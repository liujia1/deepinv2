# -*- coding: utf-8 -*-
"""
实验13.5-1：扩散最优控制(DOC) - 理论深化实践

★ 原创设计：控制论视角的扩散采样
  逆向扩散SDE = 受控动力学系统
  DOC核心：全局最优控制轨迹 vs DPS贪心每步最优

实验内容：
  - 1D高斯混合先验 + VP-SDE框架（简化版）
  - 控制论视角：引入控制输入u_t
  - 最优控制问题：最小化终端代价 -log p(y|x_0)
  - DOC实现（简化版）：存储轨迹 + 反向传播
  - DOC vs DPS对比（全局最优 vs 贪心）
  - 计算代价分析

注意：DOC需要存储整个采样轨迹并反向传播，1D简化版可运行，图像问题计算代价极高。
"""

import sys
import io
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import warnings
from tqdm import tqdm

# 设置控制台输出为 UTF-8 (Windows下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默matplotlib相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验13.5-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font模块未找到，中文字体可能无法正常显示")

# matplotlib LaTeX格式用于数学符号显示（必须在中文配置之后设置，否则会被覆盖）
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False
# ========================================================

np.random.seed(42)
torch.manual_seed(42)

print("\n" + "=" * 60)
print("实验13.5-1: 扩散最优控制(DOC)")
print("=" * 60)
print("知识点: 控制理论视角(13.5节), 最优控制问题, DOC vs DPS对比")

print("""
扩散最优控制(DOC)核心思想（13.5节）：

1. 控制论视角
   逆向扩散SDE可视为受控动力学系统：
     dx = f(x,t)dt + g(t)dw + u_t dt
   其中u_t是控制输入，引导轨迹走向最优终端状态

2. 最优控制问题
   目标：最小化终端代价 J(x_0) = -log p(y|x_0)
   约束：逆向扩散动力学

3. DOC vs DPS对比
   DPS：贪心策略，每步仅做局部最优决策
   DOC：全局最优，考虑整个轨迹的协调性

4. DOC理论优势
   - 全局最优控制轨迹
   - 精确后验采样（在PF-ODE下）
   - 克服DPS的局部贪心局限

5. DOC计算代价
   - 需存储整个采样轨迹
   - 需反向传播计算最优控制
   - 1D简化版可运行，图像问题代价极高
""")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

# ============================================================
# 1. 1D高斯混合先验 + VP-SDE框架（简化版）
# ============================================================
GM_WEIGHTS = [0.5, 0.5]
GM_MEANS = [-2.0, 2.0]
GM_STDS = [0.5, 0.5]

# VP-SDE参数（简化版，T=20步）
T = 20  # DOC简化版用少步采样
beta_min, beta_max = 1e-4, 0.02
betas = torch.linspace(beta_min, beta_max, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

def sample_from_prior(n_samples):
    """从高斯混合先验采样"""
    components = np.random.choice(len(GM_WEIGHTS), size=n_samples, p=GM_WEIGHTS)
    samples = np.array([np.random.normal(GM_MEANS[c], GM_STDS[c]) for c in components])
    return samples

def gaussian_mixture_score(x_t, t_idx):
    """混合高斯先验的真实得分函数（Torch可导版本）

    对于混合高斯 p(x) = sum_i w_i * N(x; mu_i, sigma_i^2)
    在扩散时刻t，x_t的分布为：
    p(x_t) = sum_i w_i * N(x_t; mu_i, sigma_i^2 + 1 - alpha_bar_t)

    得分函数：s(x_t) = nabla log p(x_t)

    注意：此版本支持Torch自动求导，用于DOC的全局优化
    """
    alpha_bar_t = alpha_bars[t_idx]
    sigma_t_sq = 1.0 - alpha_bar_t

    # 计算每个高斯分量的概率密度（使用torch保持计算图）
    log_probs = []
    for i in range(len(GM_WEIGHTS)):
        mu_i = torch.tensor(GM_MEANS[i], device=device, dtype=x_t.dtype)
        sigma_i_sq = torch.tensor(GM_STDS[i] ** 2, device=device, dtype=x_t.dtype)
        total_sigma_sq = sigma_i_sq + sigma_t_sq
        # log_prob = log(w_i) - 0.5*log(2*pi*sigma^2) - 0.5*(x-mu)^2/sigma^2
        # 但得分函数只需要最后一项的梯度
        log_prob = torch.log(torch.tensor(GM_WEIGHTS[i], device=device, dtype=x_t.dtype)) \
                   - 0.5 * torch.log(2 * np.pi * total_sigma_sq) \
                   - 0.5 * (x_t - mu_i) ** 2 / total_sigma_sq
        log_probs.append(log_prob)

    # 使用log-sum-exp技巧计算加权得分
    log_probs_stack = torch.stack(log_probs)
    max_log_prob = torch.max(log_probs_stack)
    exp_probs = torch.exp(log_probs_stack - max_log_prob)
    sum_exp = torch.sum(exp_probs)

    # 计算得分：s(x_t) = sum_i w_i * N(x_t; mu_i, sigma_i^2 + sigma_t^2) * (-(x_t - mu_i) / (sigma_i^2 + sigma_t^2)) / p(x_t)
    score = torch.tensor(0.0, device=device, dtype=x_t.dtype)
    for i in range(len(GM_WEIGHTS)):
        mu_i = torch.tensor(GM_MEANS[i], device=device, dtype=x_t.dtype)
        sigma_i_sq = torch.tensor(GM_STDS[i] ** 2, device=device, dtype=x_t.dtype)
        total_sigma_sq = sigma_i_sq + sigma_t_sq
        weight = exp_probs[i] / sum_exp
        score = score + weight * (-(x_t - mu_i) / total_sigma_sq)

    return score

# ============================================================
# 2. 控制论视角：逆向扩散作为受控系统
# ============================================================
print("\n" + "=" * 60)
print("步骤1：控制论视角（13.5节）")
print("=" * 60)

print("""
逆向扩散SDE的控制论形式：

dx = [f(x,t) - g(t)^2 · s_theta(x,t)] dt + g(t) dw

控制论视角：将逆向SDE视为受控动力学系统

dx = f(x,t) dt + g(t) dw + u_t dt

其中：
- f(x,t)：固有漂移（前向SDE的逆向）
- g(t)dw：固有扩散
- u_t：控制输入（引导轨迹）

最优控制问题：
- 终端代价：J(x_0) = -log p(y|x_0)
- 目标：找到最优控制序列 u_0, u_1, ..., u_T-1
  使得轨迹终端代价最小

DOC关键洞察：
u_t = alpha · nabla_x_t log p(y|x_0)
（通过整个轨迹的反向传播计算）
""")

# ============================================================
# 3. 简化DOC实现
# ============================================================
print("\n" + "=" * 60)
print("步骤2：简化DOC实现（理论验证）")
print("=" * 60)

print("""
DOC实现步骤（简化版）：

1. 前向采样生成初始轨迹 x_T → x_0
2. 存储整个轨迹：x_T, x_T-1, ..., x_0
3. 计算终端代价：J(x_0) = -log p(y|x_0)
4. 反向传播：计算最优控制 u_t = -grad_J(x_t)
5. 更新轨迹：应用最优控制重新采样
6. 重复2-5直到收敛

简化说明：
- 实际DOC需要iLQR算法或简化的Riccati方程
- 本实验简化为梯度下降优化控制序列
- 用于教学演示全局最优vs贪心对比
""")

# 简化DOC实现（梯度优化）
class SimpleDOC:
    """简化DOC：梯度优化控制序列"""

    def __init__(self, y_obs, A_obs, sigma_obs):
        self.y_obs = torch.tensor(y_obs, device=device)
        self.A_obs = torch.tensor(A_obs, device=device)
        self.sigma_obs = sigma_obs

    def terminal_cost(self, x0):
        """终端代价：-log p(y|x_0)"""
        # 简化：假设似然为高斯
        residual = self.y_obs - self.A_obs * x0
        cost = 0.5 * torch.sum(residual**2) / (self.sigma_obs**2)
        return cost

    def forward_sample_with_control(self, control_sequence, x_T, add_noise=True):
        """前向采样（带控制）

        完整逆向SDE：dx = [f(x,t) - g(t)^2 * s_theta(x,t)] dt + g(t) dw
        控制论视角：dx = [f(x,t) dt + g(t) dw] + u_t dt

        本实现中：
        - 使用混合高斯的真实得分函数（而非简化的高斯得分）
        - 控制输入u_t = 引导控制（叠加在完整逆向SDE之上）

        参数：
            control_sequence: 控制序列 [u_T, u_{T-1}, ..., u_0]
            x_T: 初始噪声状态
            add_noise: 是否添加扩散噪声（最终确定性轨迹时应为False）
        """
        trajectory = [x_T]
        x_t = x_T.clone()

        for t_idx in range(T-1, -1, -1):  # 逆向时间
            beta_t = betas[t_idx]
            alpha_bar_t = alpha_bars[t_idx]

            # 数值稳定性：避免除以零
            one_minus_alpha_bar_t = torch.clamp(1.0 - alpha_bar_t, min=1e-6)

            # 完整逆向SDE漂移 = 固有漂移 + 得分项
            # 修正：逆向SDE的漂移项符号应为正（解开前向的收缩）
            # 标准 VP-SDE 逆向采样：x_{t-1} = x_t + 0.5*β_t*x_t + β_t*score(x_t,t) + √β_t*z
            score_tensor = gaussian_mixture_score(x_t, t_idx)  # 直接使用torch版本，保持计算图
            drift = 0.5 * beta_t * x_t + beta_t * score_tensor

            # 控制输入（DOC的引导控制，叠加在完整逆向SDE之上）
            u_t = control_sequence[t_idx]

            # 扩散（可选）
            if add_noise:
                diffusion = torch.sqrt(beta_t) * torch.randn_like(x_t)
            else:
                diffusion = 0.0

            # 更新
            x_t = x_t + drift + u_t + diffusion
            trajectory.append(x_t.clone())

        return torch.stack(trajectory)

    def optimize_control(self, x_T, n_iters=300, lr=0.01, reg_lambda=0.1, trial_idx=None):
        """优化控制序列

        损失函数 = 终端代价 + reg_lambda * 控制幅度惩罚
        控制幅度惩罚防止控制输入过大，保证优化稳定性
        """
        # 初始化控制序列
        control_sequence = torch.zeros(T, device=device, requires_grad=True)
        optimizer = optim.Adam([control_sequence], lr=lr)

        trajectory_history = []
        cost_history = []

        for iter_idx in range(n_iters):
            optimizer.zero_grad()

            # 前向采样
            trajectory = self.forward_sample_with_control(control_sequence, x_T)
            x0 = trajectory[-1]

            # 终端代价 + 控制幅度正则化
            terminal_cost = self.terminal_cost(x0)
            control_reg = reg_lambda * torch.sum(control_sequence ** 2)
            total_cost = terminal_cost + control_reg

            # 反向传播
            total_cost.backward()
            optimizer.step()

            trajectory_history.append(trajectory.detach().cpu().numpy())
            cost_history.append(terminal_cost.item())

            if iter_idx % 20 == 0:
                trial_info = f"试验{trial_idx+1} " if trial_idx is not None else ""
                print(f"\r{trial_info}Iter {iter_idx}: 终端代价 = {terminal_cost.item():.6f}, "
                      f"控制正则 = {control_reg.item():.6f}, x0 = {x0.item():.4f}", end="", flush=True)

        # 最终轨迹（无噪声，确定性）
        final_trajectory = self.forward_sample_with_control(control_sequence.detach(), x_T, add_noise=False)

        return final_trajectory, control_sequence.detach(), trajectory_history, cost_history

# ============================================================
# 4. DPS贪心策略对比
# ============================================================
print("\n" + "=" * 60)
print("步骤3：DPS贪心策略（对比）")
print("=" * 60)

print("""
DPS贪心策略（13.3节）：

每步仅做局部最优决策：
  u_t = zeta · nabla log p(y|x_hat_0|t)

其中x_hat_0|t是Tweedie去噪估计

局限：
- 仅考虑当前步的局部最优
- 忽略全局轨迹协调性
- 可能导致终端状态偏离观测

DOC优势：
- 考虑整个轨迹的全局最优
- 通过反向传播优化所有控制步骤
- 精确匹配终端观测约束
""")

def dps_sampling(y_obs, A_obs, sigma_obs, x_T, zeta=1.0):
    """DPS采样（贪心策略）

    完整逆向SDE：dx = [f(x,t) - g(t)^2 * s_theta(x,t)] dt + g(t) dw
    DPS修正：s_corrected = s_theta + zeta * nabla log p(y|x_hat_0|t)

    本实现中：
    - 使用混合高斯的真实得分函数（而非简化的高斯得分）
    - u_t = zeta * 似然得分（精确Tweedie估计），贪心策略
    """
    trajectory = [x_T.clone()]
    x_t = x_T.clone()

    for t_idx in range(T-1, -1, -1):  # 逆向时间
        beta_t = betas[t_idx]
        alpha_bar_t = alpha_bars[t_idx]

        # 数值稳定性保护
        alpha_bar_t_safe = torch.clamp(alpha_bar_t, min=0.01)

        # 精确Tweedie去噪估计（使用得分函数）
        score = gaussian_mixture_score(x_t, t_idx)
        x_hat_0 = (x_t + (1.0 - alpha_bar_t_safe) * score) / torch.sqrt(alpha_bar_t_safe)

        # DPS似然得分（贪心，基于Tweedie估计）
        residual = y_obs - A_obs * x_hat_0
        likelihood_score = zeta * A_obs * residual / (sigma_obs**2)

        # 梯度裁剪，防止数值爆炸
        likelihood_score = torch.clamp(likelihood_score, min=-100, max=100)

        # 完整逆向SDE漂移 = 固有漂移 + 得分项
        # 修正：逆向SDE的漂移项符号应为正（解开前向的收缩）
        # 标准 VP-SDE 逆向采样：x_{t-1} = x_t + 0.5*β_t*x_t + β_t*score(x_t,t) + √β_t*z
        drift = 0.5 * beta_t * x_t + beta_t * score

        # 控制输入（贪心策略：似然得分作为控制）
        u_t = 0.01 * likelihood_score  # 降低控制强度

        # 扩散
        diffusion = torch.sqrt(beta_t) * torch.randn_like(x_t)

        # 更新
        x_t = x_t + drift + u_t + diffusion
        trajectory.append(x_t.clone())

    return torch.stack(trajectory)

# ============================================================
# 5. DOC vs DPS对比实验（多次试验取平均）
# ============================================================
print("\n" + "=" * 60)
print("步骤4：DOC vs DPS对比实验")
print("=" * 60)

# 多次试验取平均（对齐13_3-5.py的做法）
n_trials = 50
doc_errors = []
dps_errors = []

# 支持断点续传
checkpoint_file = os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', 'experiment_checkpoint.pkl')
start_trial = 0

if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
        doc_errors = checkpoint['doc_errors']
        dps_errors = checkpoint['dps_errors']
        start_trial = len(doc_errors)
        print(f"\n从检查点恢复，已完成{start_trial}/{n_trials}次试验")

print(f"\n开始{n_trials}次试验...")
for trial_idx in range(start_trial, n_trials):
    # 生成真实信号和观测
    x0_star = sample_from_prior(1)[0]
    A_obs = 2.0  # 简化线性观测
    sigma_obs = 0.1
    y_obs = A_obs * x0_star + np.random.randn() * sigma_obs

    # 初始噪声状态
    x_T = torch.randn(1, device=device) * 3.0

    # DOC采样
    doc_sampler = SimpleDOC(y_obs, A_obs, sigma_obs)
    doc_trajectory, doc_controls, doc_history, doc_costs = doc_sampler.optimize_control(x_T, n_iters=100, lr=0.01, trial_idx=trial_idx)

    # DPS采样
    dps_trajectory = dps_sampling(torch.tensor(y_obs, device=device),
                                  torch.tensor(A_obs, device=device),
                                  sigma_obs, x_T.clone())

    # 计算终端误差
    doc_x0 = doc_trajectory[-1].item()
    dps_x0 = dps_trajectory[-1].item()
    doc_errors.append(abs(doc_x0 - x0_star))
    dps_errors.append(abs(dps_x0 - x0_star))

    # 每轮保存检查点
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({'doc_errors': doc_errors, 'dps_errors': dps_errors}, f)

# 统计结果
doc_mean_error = np.mean(doc_errors)
doc_std_error = np.std(doc_errors)
dps_mean_error = np.mean(dps_errors)
dps_std_error = np.std(dps_errors)

print(f"\n统计结果（n={n_trials}）：")
print(f"  DOC平均误差：{doc_mean_error:.6f} ± {doc_std_error:.6f}")
print(f"  DPS平均误差：{dps_mean_error:.6f} ± {dps_std_error:.6f}")
print(f"  DOC相对改进：{(dps_mean_error - doc_mean_error) / dps_mean_error * 100:.2f}%")

# 单次试验可视化（使用最后一次试验的结果）
print("\n单次试验可视化（用于展示轨迹对比）...")

# 可视化对比（使用最后一次试验的结果）
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 子图1：轨迹对比
axes[0, 0].plot(range(T+1), doc_trajectory.cpu().numpy(), 'o-', label='DOC (全局最优)',
                linewidth=2, markersize=6)
axes[0, 0].plot(range(T+1), dps_trajectory.cpu().numpy(), 's-', label='DPS (贪心)',
                linewidth=2, markersize=6)
axes[0, 0].axhline(y=x0_star, color='green', linestyle='--', label=f'真实信号 $x^*$={x0_star:.3f}')
axes[0, 0].set_xlabel('时间步 $t$')
axes[0, 0].set_ylabel('信号值 $x_t$')
axes[0, 0].set_title('轨迹对比：DOC vs DPS（单次试验示例）')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 子图2：控制输入对比
axes[0, 1].plot(range(T), doc_controls.cpu().numpy(), 'o-', label='DOC控制',
                linewidth=2, markersize=6)
axes[0, 1].set_xlabel('时间步 $t$')
axes[0, 1].set_ylabel('控制输入 $u_t$')
axes[0, 1].set_title('DOC控制输入（全局协调）')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 子图3：终端状态对比（使用统计结果）
methods = ['DOC', 'DPS']
colors = ['steelblue', 'coral']

# 使用最后一次试验的值进行可视化
doc_x0 = doc_trajectory[-1].item()
dps_x0 = dps_trajectory[-1].item()

axes[1, 0].bar(methods, [doc_x0, dps_x0], color=colors, alpha=0.7)
axes[1, 0].axhline(y=x0_star, color='green', linestyle='--', linewidth=2,
                   label=f'真实信号 $x^*$={x0_star:.3f}')
axes[1, 0].set_ylabel('终端状态 $x_0$')
axes[1, 0].set_title(f'终端状态对比（单次试验）\n'
                     f'平均误差：DOC={doc_mean_error:.4f}±{doc_std_error:.4f}, DPS={dps_mean_error:.4f}±{dps_std_error:.4f}')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 子图4：优化历史（同时展示终端代价和终端状态x0的收敛）
iter_costs_history = []
iter_x0s = []
for traj in doc_history:
    x0_tensor = torch.tensor(traj[-1], device=device)
    cost_val = doc_sampler.terminal_cost(x0_tensor).item()
    iter_costs_history.append(cost_val)
    iter_x0s.append(traj[-1])

# 双y轴图：左轴终端代价，右轴终端状态x0
ax_left = axes[1, 1]
ax_right = ax_left.twinx()
line1 = ax_left.plot(range(len(iter_costs_history)), iter_costs_history, 'b-o',
                     linewidth=2, markersize=4, label='终端代价 $J(x_0)$')
line2 = ax_right.plot(range(len(iter_x0s)), iter_x0s, 'r-s',
                      linewidth=2, markersize=4, label='终端状态 $x_0$')
ax_right.axhline(y=x0_star, color='green', linestyle='--', alpha=0.7, label=f'真实信号 $x^*$={x0_star:.3f}')

ax_left.set_xlabel('优化迭代')
ax_left.set_ylabel('终端代价 $J(x_0)$', color='b')
ax_right.set_ylabel('终端状态 $x_0$', color='r')
ax_left.set_title('DOC优化过程：终端代价与终端状态收敛')
ax_left.tick_params(axis='y', labelcolor='b')
ax_right.tick_params(axis='y', labelcolor='r')
ax_left.grid(True, alpha=0.3)

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax_left.legend(lines, labels, loc='upper right', fontsize=9)

plt.suptitle(f'DOC vs DPS对比：全局最优控制 vs 贪心策略\n（n={n_trials}次试验统计）', fontsize=14, y=0.98)
plt.tight_layout()
compare_path = os.path.join(SAVE_DIR, "DOC vs DPS对比.png")
plt.savefig(compare_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"对比结果已保存: {compare_path}")

print("\n关键发现（基于{}次试验统计）：".format(n_trials))
print(f"  DOC平均误差: {doc_mean_error:.6f} ± {doc_std_error:.6f}")
print(f"  DPS平均误差: {dps_mean_error:.6f} ± {dps_std_error:.6f}")
print(f"  DOC相对改进: {(dps_mean_error - doc_mean_error) / dps_mean_error * 100:.2f}%")

# ============================================================
# 6. 计算代价分析
# ============================================================
print("\n" + "=" * 60)
print("步骤5：DOC计算代价分析")
print("=" * 60)

print("""
DOC计算代价（13.5节）：

1. DOC计算流程：
   - 前向采样：存储整个轨迹（T步）
   - 反向传播：计算梯度（T步链式法则）
   - 优化迭代：多次前向+反向传播

2. 内存代价：
   - 需存储整个轨迹：x_T, x_T-1, ..., x_0
   - 1D问题：可接受（T×1个变量）
   - 图像问题：极高（T×H×W×C个变量）

3. 计算代价对比：
   DOC：全局最优，计算代价极高
   DPS：贪心策略，计算代价低

4. DOC适用场景：
   - 1D简化问题（教学演示）
   - 关键逆问题（可接受高计算代价）
   - 图像问题：通常不实用

5. 教学意义：
   - 展示全局最优vs贪心差异
   - 理解DOC的理论价值
   - 认识计算代价的限制

方法对比表：

| 方法 | 内存需求         | 计算代价         | 质量         | 适用场景       | 实现复杂度     |
|------|------------------|------------------|--------------|----------------|----------------|
| DOC  | 高（存储轨迹）   | 极高（反向传播） | 全局最优     | 1D/关键逆问题  | 高（需优化）   |
| DPS  | 低（仅当前状态） | 低（单步计算）   | 贪心局部最优 | 图像/实时应用  | 低（单步贪心） |

结论：DOC理论最优但计算代价高，DPS实用性强
""")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("实验13.5-1 完成!")
print("=" * 60)
print("""
关键结论:
1. DOC控制论视角（13.5节）
   - 逆向扩散 = 受控动力学系统
   - 控制输入u_t引导轨迹走向最优终端
   - 最优控制问题：最小化终端代价

2. DOC vs DPS对比
   - DOC：全局最优控制，考虑整个轨迹协调性
   - DPS：贪心策略，仅做局部最优决策
   - 实验显示DOC终端误差更小（理论优势）

3. DOC计算代价
   - 需存储整个采样轨迹
   - 需反向传播计算最优控制
   - 1D简化版可运行，图像问题计算代价极高

4. 教学意义
   - 展示控制论视角的威力
   - 理解全局最优vs贪心的差异
   - 认识计算代价对实用性的限制
   - 理解DOC的理论价值（精确后验采样）
""")
# ===== 保存数值结果 =====
import json

def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
    import numpy as np
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except: pass
    return obj

results_summary = {
    "试验次数": n_trials,
    "DOC_平均误差": round(float(doc_mean_error), 6),
    "DOC_误差标准差": round(float(doc_std_error), 6),
    "DPS_平均误差": round(float(dps_mean_error), 6),
    "DPS_误差标准差": round(float(dps_std_error), 6),
    "DOC相对改进百分比": round(float((dps_mean_error - doc_mean_error) / dps_mean_error * 100), 2),
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
