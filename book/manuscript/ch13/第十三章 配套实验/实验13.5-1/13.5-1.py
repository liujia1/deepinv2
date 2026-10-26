# -*- coding: utf-8 -*-
"""
实验13.5-1：扩散最优控制(DOC) - 理论深化实践

★ 原创设计：控制论视角的扩散采样
  逆向扩散SDE = 受控动力学系统
  DOC核心：全局最优控制轨迹 vs DPS贪心每步最优

实验内容：
  - 1D高斯混合先验 + VP-SDE框架（连续VP离散化，T=20步，末端ᾱ_T≈0.01）
  - 控制论视角：引入控制输入u_t
  - 最优控制问题：最小化终端代价 -log p(y|x_0) + 控制能量正则(Tikhonov项)
  - DOC实现（简化版）：在确定性PF-ODE轨迹上用Adam打靶优化控制序列
    （Li & Pereira 2024 原算法为iLQR，本实验以开环梯度优化做教学简化）
  - DOC vs DPS对比（全局轨迹优化 vs 逐步贪心引导）
  - 计算代价分析

关键实现约定（与DOC原论文一致）：
  - DOC在概率流ODE(PF-ODE)的确定性rollout上优化并评估——优化与评估必须
    使用同一条动力学，否则确定性等价原理不成立，优化出的控制并非评估轨迹
    的最优控制（早期版本曾因"带噪优化+无噪评估"的动力学失配得到错误结论）。
  - DPS引导梯度对x_t求取（穿过Tweedie估计的雅可比），逆向更新采用DDPM
    后验均值/方差（末端步无噪声）。
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

4. DOC理论优势（Li & Pereira, NeurIPS 2024, arXiv:2412.16748）
   - 全局最优控制轨迹：一次性协调全部时间步的控制
   - 在PF-ODE下可恢复精确后验采样（Tikhonov常数取适当值时）
   - 对离散化误差鲁棒：DPS类方法在少步/大步长时引导强度需精细调参，
     ζ稍大即发散（"离散化脆性"），DOC把引导转化为轨迹优化，无此超参

5. DOC计算代价
   - 朴素实现需存储整个采样轨迹并反向传播（BPTT），1D可运行、图像代价高
   - 原论文用iLQR+随机低秩近似+无矩阵求值，已在FFHQ 256×256图像
     逆问题上实用化；本实验的开环Adam打靶仅为教学简化
""")

device = torch.device(os.environ.get('DEEPINV_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
print(f"\n使用设备: {device}")

# ============================================================
# 1. 1D高斯混合先验 + VP-SDE框架（简化版）
# ============================================================
GM_WEIGHTS = [0.5, 0.5]
GM_MEANS = [-2.0, 2.0]
GM_STDS = [0.5, 0.5]

# VP-SDE参数（默认 T=20 步少步采样）
# 连续VP-SDE(β(t)恒定)的离散化：令累积噪声 bbar(t)=-log ᾱ_t 从0线性增到4.6，
# 则 ᾱ_T=e^{-4.6}≈0.01，正向末端边际 p_T≈N(0,1)，与 x_T~N(0,1) 初始分布匹配。
# 反推每步系数 α_t=ᾱ_t/ᾱ_{t-1}, β_t=1-α_t（T=20 时 β≈0.215/步，属少步大步长设定）。
#
# ★ T 可由环境变量 DEEPINV_DOC_T 覆盖，用于"步数 T 对 DOC/DPS 对比的影响"补充对照：
#   T=20  主实验（少步大步长，离散化脆性最突出）；
#   T=200 补充对照（细步长；DPS 引导项 u_t=ζ·β_t·∇log p 中 β_t 随 T 等比缩小，
#         故稳定窗的 ζ 需同比放大；DOC 的控制变量由 20 维增至 200 维，而迭代次数
#         固定 300，属于对 200 维序列欠收敛，正文据此讨论"增大 T 未必利好 DOC"）。
#   补充对照不改主结果、不覆盖主图与主 JSON 字段。
T = int(os.environ.get('DEEPINV_DOC_T', '20'))
IS_MAIN_T = (T == 20)
U_MAX = 0.5  # 单步控制幅度上限：tanh饱和，防止末步β_0=0时一步投影到y/A

# DPS 引导系数配置
# T=20 主实验用实测标定值 0.02（稳定窗约 0.01~0.025，ζ≥0.03 即出现发散）；
# 补充对照（T≠20）不预设"最优 ζ"，而是扫描一组 ζ 并逐一记录，避免把单点选择
# 当成结论——离散化脆性本身就表现为"换 T 就得重新标定 ζ"。
DPS_ZETA_MAIN = 0.02 if IS_MAIN_T else float(os.environ.get('DEEPINV_DOC_ZETA', '0.3'))
DPS_ZETA_SWEEP = [DPS_ZETA_MAIN] if IS_MAIN_T else [0.1, 0.15, 0.2, 0.3, 0.5]
b_bar_end = 4.6
beta_bars = torch.linspace(0.0, b_bar_end, T, device=device)
alpha_bars = torch.exp(-beta_bars)
_prev_alpha_bars = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
alphas = alpha_bars / _prev_alpha_bars
betas = 1.0 - alphas
# DDPM逆向后验方差 β̃_t = β_t·(1-ᾱ_{t-1})/(1-ᾱ_t)，末端步(t=0)为0（x_0不带噪）
beta_tildes = betas * (1.0 - _prev_alpha_bars) / (1.0 - alpha_bars)
beta_tildes[0] = 0.0

def sample_from_prior(n_samples):
    """从高斯混合先验采样"""
    components = np.random.choice(len(GM_WEIGHTS), size=n_samples, p=GM_WEIGHTS)
    samples = np.array([np.random.normal(GM_MEANS[c], GM_STDS[c]) for c in components])
    return samples

def gaussian_mixture_score(x_t, t_idx):
    """混合高斯先验在VP-SDE下的边际得分函数（Torch可导版本）

    VP-SDE正向边际：x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·z，故
    p(x_t) = sum_i w_i * N(x_t; √ᾱ_t·mu_i, ᾱ_t·sigma_i^2 + 1 - ᾱ_t)

    得分函数：s(x_t) = nabla log p(x_t)

    注意：此版本与VP离散化/Tweedie公式使用同一约定（含√ᾱ信号衰减），
    支持Torch自动求导，用于DOC的全局优化
    """
    alpha_bar_t = alpha_bars[t_idx]
    sqrt_ab = torch.sqrt(alpha_bar_t)

    # 计算每个高斯分量的概率密度（使用torch保持计算图）
    log_probs = []
    for i in range(len(GM_WEIGHTS)):
        mu_i = torch.tensor(GM_MEANS[i], device=device, dtype=x_t.dtype)
        sigma_i_sq = torch.tensor(GM_STDS[i] ** 2, device=device, dtype=x_t.dtype)
        m_i = sqrt_ab * mu_i                        # 边际均值 √ᾱ·mu_i
        total_sigma_sq = alpha_bar_t * sigma_i_sq + (1.0 - alpha_bar_t)  # 边际方差 ᾱσ²+1-ᾱ
        # log_prob = log(w_i) - 0.5*log(2*pi*sigma^2) - 0.5*(x-mu)^2/sigma^2
        log_prob = torch.log(torch.tensor(GM_WEIGHTS[i], device=device, dtype=x_t.dtype)) \
                   - 0.5 * torch.log(2 * np.pi * total_sigma_sq) \
                   - 0.5 * (x_t - m_i) ** 2 / total_sigma_sq
        log_probs.append(log_prob)

    # 使用log-sum-exp技巧计算加权得分
    log_probs_stack = torch.stack(log_probs)
    max_log_prob = torch.max(log_probs_stack)
    exp_probs = torch.exp(log_probs_stack - max_log_prob)
    sum_exp = torch.sum(exp_probs)

    # 计算得分：s(x_t) = sum_i [posterior_weight_i] * (-(x_t - m_i) / v_i)
    score = torch.tensor(0.0, device=device, dtype=x_t.dtype)
    for i in range(len(GM_WEIGHTS)):
        mu_i = torch.tensor(GM_MEANS[i], device=device, dtype=x_t.dtype)
        sigma_i_sq = torch.tensor(GM_STDS[i] ** 2, device=device, dtype=x_t.dtype)
        m_i = sqrt_ab * mu_i
        total_sigma_sq = alpha_bar_t * sigma_i_sq + (1.0 - alpha_bar_t)
        weight = exp_probs[i] / sum_exp
        score = score + weight * (-(x_t - m_i) / total_sigma_sq)

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
- f(x,t)：固有漂移（正向SDE的逆向）
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
DOC实现步骤（简化版，确定性PF-ODE轨迹）：

1. 从 x_T~N(0,1) 出发，沿无引导PF-ODE确定性rollout生成初始轨迹 x_T → x_0
2. 存储整条轨迹（自动微分保持计算图）：x_T, x_T-1, ..., x_0
3. 计算代价：J = -log p(y|x_0) + λ·Σu_t²
4. 反向传播：计算J对每个控制量u_t的梯度（BPTT穿过整条轨迹）
5. Adam更新控制序列，重新确定性rollout
6. 重复2-5直到收敛（300次迭代）

简化说明：
- DOC原论文使用iLQR（值函数一阶/二阶展开得反馈增益），无需BPTT即可
  高效求解，并配合随机低秩近似扩展到256×256图像
- 本实验简化为开环Adam打靶：1D下直观可跑，能完整展示"全局轨迹优化"
  与"逐步贪心引导"的差异，但不含iLQR的反馈控制结构
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

    def forward_sample_with_control(self, control_sequence, x_T, add_noise=False):
        """受控逆向采样（PF-ODE确定性轨迹 / 可加噪退化为逆向SDE）

        无控制基座采用DDPM逆向后验的得分形式（大步长下比Euler-Maruyama稳定）：
            均值  μ_t = (x_t + β_t·s(x_t,t)) / √α_t
            x_{t-1} = μ_t + u_t [+ √β̃_t · z]
        其中：
            - s(x_t,t) 为混合高斯在VP-SDE下的精确边际得分
            - u_t 为DOC控制输入（加性，叠加在后验均值上）
            - β̃_t 为DDPM后验方差，末端步β̃_0=0（x_0确定性输出）

        ★关键约定：DOC优化与评估都使用 add_noise=False 的确定性PF-ODE轨迹。
          DOC原论文的"精确后验恢复"结论正是建立在PF-ODE之上；若优化时向
          rollout注入随机噪声、评估时又走确定性轨迹，二者动力学不一致，
          Adam优化的是"随机轨迹终端代价的期望"，而非评估轨迹的终端代价，
          会系统性偏离最优（早期版本的实测误差约为理论最优的2.6倍）。

        控制饱和（教学修正）：标准DOC的控制u_t无幅度约束，而本实验的离散
        DDPM调度在末步有β_0=0、α_0=1，此时后验均值就是x_t，无约束控制
        可一步把x_0设成y/A——"全局轨迹优化"退化为"末步数据一致点投影"，
        控制序列的其余分量完全不起作用（实测DOC误差与y/A误差在1e-6内相等）。
        为此对每步控制做 u_t = U_MAX·tanh(raw_u_t) 饱和，使单步权限受限、
        DOC必须在前面的步中分配控制，真正体现轨迹优化；U_MAX=0.5。

        参数：
            control_sequence: 控制序列 [u_T, u_{T-1}, ..., u_0]
            x_T: 初始噪声状态
            add_noise: 是否注入后验噪声（DOC固定为False；仅教学演示SDE时使用）
        """
        trajectory = [x_T]
        x_t = x_T.clone()

        for t_idx in range(T-1, -1, -1):  # 逆向时间
            beta_t = betas[t_idx]
            alpha_t = alphas[t_idx]

            # 逆向后验均值（得分形式），保持自动微分计算图
            score_tensor = gaussian_mixture_score(x_t, t_idx)
            posterior_mean = (x_t + beta_t * score_tensor) / torch.sqrt(alpha_t)

            # 控制输入（DOC的引导控制）
            u_t = control_sequence[t_idx]

            # 后验噪声（可选；DOC固定不加）
            if add_noise:
                posterior_noise = torch.sqrt(beta_tildes[t_idx]) * torch.randn_like(x_t)
            else:
                posterior_noise = 0.0

            # 更新（控制经tanh饱和，限制单步幅度，避免末步一步投影）
            x_t = posterior_mean + U_MAX * torch.tanh(u_t) + posterior_noise
            trajectory.append(x_t.clone())

        return torch.stack(trajectory)

    def optimize_control(self, x_T, n_iters=300, lr=0.01, reg_lambda=0.1, trial_idx=None):
        """在确定性PF-ODE轨迹上优化控制序列（开环Adam打靶，iLQR的教学简化）

        损失函数 = 终端代价 + reg_lambda * Σu_t²
        - 终端代价 = -log p(y|x_0)（高斯似然，即0.5||y-Ax_0||²/σ²）
        - Σu_t²为控制能量项，对应DOC原论文代价函数中的Tikhonov正则，
          使最优控制问题良态并与后验采样相联系。本实验的主要控制约束
          来自单步tanh饱和（U_MAX=0.5），该项仅作辅助正则。
        优化与最终评估使用同一条确定性rollout，不存在动力学失配。
        """
        # 初始化控制序列
        control_sequence = torch.zeros(T, device=device, requires_grad=True)
        optimizer = optim.Adam([control_sequence], lr=lr)

        trajectory_history = []
        cost_history = []

        for iter_idx in range(n_iters):
            optimizer.zero_grad()

            # 正向采样
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
DPS贪心引导策略（Diffusion Posterior Sampling, Chung et al. 2023）：

逆向SDE每步用似然梯度贪心修正得分：
  s_corrected(x_t,t) = s(x_t,t) + ζ · ∇_{x_t} log p(y|x̂_{0|t})
其中 x̂_{0|t} 是Tweedie去噪估计，梯度必须对 x_t 求取——即要穿过
Tweedie映射的雅可比 ∂x̂_0/∂x_t（本实验用torch自动微分精确计算）。

本实验的DPS实现要点：
- 精确边际得分：Tweedie估计在此等于精确的 E[x_0|x_t]（可数值验证）
- 规范链式梯度：∇_{x_t} log p(y|x̂_0)（不省略雅可比）
- DDPM后验均值/方差更新，末端步不加噪

DPS的固有局限（DOC原论文第4节）：
- 离散化脆性：引导强度ζ是敏感超参，少步/大步长下稳定窗很窄，
  ζ稍大即过冲发散（本实验T=20、β≈0.215时，ζ=0.02表现良好，
  ζ=0.03已出现发散试验）
- 逐步贪心：每步只最小化当前的观测不一致，不协调整条轨迹
- 依赖x̂_0近似质量：真实场景中得分网络有误差时，梯度质量随之下降
""")

def dps_sampling(y_obs, A_obs, sigma_obs, x_T, zeta=0.02):
    """DPS采样（贪心引导，规范链式梯度 + DDPM后验更新）

    每步更新：
        μ_t   = (x_t + β_t·s(x_t,t)) / √α_t + u_t
        x_{t-1} = μ_t + √β̃_t · z,   β̃_0=0
        u_t   = ζ · β_t · ∇_{x_t} log p(y|x̂_{0|t}(x_t))

    与早期简化版的区别：
    1. 梯度对x_t求取（自动微分穿过Tweedie映射 x̂_0=(x_t+(1-ᾱ_t)s)/√ᾱ_t），
       即规范的DPS条件得分近似；早期版本误用∇_{x̂_0}，漏掉雅可比∂x̂_0/∂x_t
       （少步高噪声步该雅可比中位仅约0.6，导致引导强度虚高、轨迹过冲）。
    2. 噪声采用DDPM后验方差β̃_t而非β_t：末端步β̃_0=0，避免最后一步
       向x_0直接灌入量级√β≈0.46的噪声淹没σ/A=0.05的数据精度。
    3. ζ在当前调度(β≈0.215)上实测标定：稳定窗很窄(约0.01~0.025)，
       ζ=0.02效果良好，ζ=0.03即出现发散试验——离散化脆性的直接体现。
    """
    trajectory = [x_T.clone()]
    x_t = x_T.clone()

    for t_idx in range(T-1, -1, -1):  # 逆向时间
        beta_t = betas[t_idx]
        alpha_bar_t = alpha_bars[t_idx]

        # --- 规范DPS梯度：对x_t求导，穿过Tweedie估计的雅可比 ---
        x_t_req = x_t.detach().clone().requires_grad_(True)
        score_req = gaussian_mixture_score(x_t_req, t_idx)
        x_hat_0 = (x_t_req + (1.0 - alpha_bar_t) * score_req) / torch.sqrt(alpha_bar_t)
        log_likelihood = -0.5 * (y_obs - A_obs * x_hat_0) ** 2 / (sigma_obs ** 2)
        likelihood_grad = torch.autograd.grad(log_likelihood, x_t_req)[0]
        score_val = score_req.detach()

        # --- DDPM逆向后验均值（得分形式）+ 贪心引导 ---
        posterior_mean = (x_t + beta_t * score_val) / torch.sqrt(alphas[t_idx])
        u_t = zeta * beta_t * likelihood_grad

        # --- 后验噪声（末端步β̃_0=0）---
        posterior_noise = torch.sqrt(beta_tildes[t_idx]) * torch.randn_like(x_t)

        # 更新
        x_t = posterior_mean + u_t + posterior_noise
        trajectory.append(x_t.clone())

    return torch.stack(trajectory)

# ============================================================
# 5. DOC vs DPS对比实验（多次试验取平均）
# ============================================================
print("\n" + "=" * 60)
print("步骤4：DOC vs DPS对比实验")
print("=" * 60)

def gaussian_posterior_mean(y_obs, A_obs, sigma_obs):
    """GMM先验 + 高斯线性观测下的精确后验均值 E[x0|y]（一维闭式解）

    每个高斯分量经线性高斯观测后仍为高斯：
      p(x0|y,c) = N(x0; m_c, v_c^post)
      m_c = μ_c + A·σ_c²/(A²σ_c²+σ²)·(y-Aμ_c)
    后验分量权重 ∝ w_c·N(y; Aμ_c, A²σ_c²+σ²)
    这是任何后验点估计器终端误差的MMSE参考下界。
    """
    post_weights, post_means = [], []
    for i in range(len(GM_WEIGHTS)):
        mu_i, sigma_i = GM_MEANS[i], GM_STDS[i]
        v_y = A_obs ** 2 * sigma_i ** 2 + sigma_obs ** 2
        w_i = GM_WEIGHTS[i] * np.exp(-0.5 * (y_obs - A_obs * mu_i) ** 2 / v_y) / np.sqrt(2 * np.pi * v_y)
        m_i = mu_i + (A_obs * sigma_i ** 2 / v_y) * (y_obs - A_obs * mu_i)
        post_weights.append(w_i)
        post_means.append(m_i)
    post_weights = np.array(post_weights)
    return float(np.sum(post_weights / post_weights.sum() * np.array(post_means)))

# 多次试验取平均（配对设计：DOC与DPS共享同一x*、y、x_T）
n_trials = int(os.environ.get('DEEPINV_DOC_NTRIALS', '50'))
doc_errors = []
dps_errors = []
yA_errors = []        # 参考：数据一致点 y/A 的误差（终端优化的极限之一）
postmean_errors = []  # 参考：精确后验均值的误差（MMSE点估计下界）

# 各 ζ 的终端误差分桶记录（T=20 时只有一个桶，即主实验的 DPS_平均误差）
dps_errors_by_zeta = {_z: [] for _z in DPS_ZETA_SWEEP}

# 支持断点续传（v3文件名：与修复前旧检查点区分，旧档保留不覆盖）
# 补充对照（T≠20）用带 T 后缀的独立检查点，不干扰主实验续传
_ckpt_name = 'experiment_checkpoint_v3.pkl' if IS_MAIN_T else f'experiment_checkpoint_T{T}.pkl'
checkpoint_file = os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.',
                               _ckpt_name)
start_trial = 0
viz_data = None

if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    doc_errors = checkpoint['doc_errors']
    dps_errors = checkpoint['dps_errors']
    yA_errors = checkpoint.get('yA_errors', [])
    postmean_errors = checkpoint.get('postmean_errors', [])
    start_trial = len(doc_errors)
    viz_data = checkpoint.get('viz')
    _dps_by = checkpoint.get('dps_by_zeta')
    if _dps_by:
        for _k, _v in _dps_by.items():
            if float(_k) in dps_errors_by_zeta:
                dps_errors_by_zeta[float(_k)] = list(_v)
    # 旧检查点（无 dps_by_zeta 字段）的兼容：主 ζ 桶直接复用已存的 dps_errors
    if not dps_errors_by_zeta[DPS_ZETA_MAIN] and dps_errors:
        dps_errors_by_zeta[DPS_ZETA_MAIN] = list(dps_errors)
    _meta = checkpoint.get('meta')
    if _meta is None:
        print("\n警告: 检查点无配置元数据（旧格式），仅按已完成次数续传；"
              "若最近改过 n_trials/先验参数，建议删除该文件后重跑")
    elif _meta.get('n_trials') != n_trials:
        raise RuntimeError(
            f"检查点 n_trials={_meta.get('n_trials')} 与当前脚本 n_trials={n_trials} 不一致；"
            f"请删除或改名 {os.path.basename(checkpoint_file)} 后重跑")
    if start_trial > n_trials:
        raise RuntimeError(
            f"检查点已完成 {start_trial} 次试验，超过当前 n_trials={n_trials}；"
            f"请删除或改名 {os.path.basename(checkpoint_file)} 后重跑")
    print(f"\n从检查点恢复，已完成{start_trial}/{n_trials}次试验")

print(f"\n开始{n_trials}次试验（T={T}，DPS扫描ζ={DPS_ZETA_SWEEP}，主报ζ={DPS_ZETA_MAIN}）...")
for trial_idx in range(start_trial, n_trials):
    # 生成真实信号和观测
    x0_star = sample_from_prior(1)[0]
    A_obs = 2.0  # 简化线性观测（放大观测，可理解为1维退化的超分辨模型）
    sigma_obs = 0.1
    y_obs = A_obs * x0_star + np.random.randn() * sigma_obs

    # 初始噪声状态：新调度下 ᾱ_T≈0.01，正向末端边际 p_T≈N(0,1)，直接取标准正态
    x_T = torch.randn(1, device=device)

    # DOC采样（确定性PF-ODE轨迹优化）
    doc_sampler = SimpleDOC(y_obs, A_obs, sigma_obs)
    doc_trajectory, doc_controls, doc_history, doc_costs = doc_sampler.optimize_control(x_T, n_iters=300, lr=0.01, trial_idx=trial_idx)

    # DPS采样（随机逆向SDE；与DOC共享x_T，配对设计）
    # 逐个 ζ 各跑一条链（T=20 时仅一个 ζ；同一条 x_T 保证配对可比）
    dps_trajectory = None
    for _zeta in DPS_ZETA_SWEEP:
        _traj = dps_sampling(torch.tensor(y_obs, device=device),
                             torch.tensor(A_obs, device=device),
                             sigma_obs, x_T.clone(), zeta=_zeta)
        dps_errors_by_zeta[_zeta].append(abs(_traj[-1].item() - x0_star))
        if _zeta == DPS_ZETA_MAIN:
            dps_trajectory = _traj

    # 计算终端误差
    doc_x0 = doc_trajectory[-1].item()
    dps_x0 = dps_trajectory[-1].item()
    yA_x0 = y_obs / A_obs
    pm_x0 = gaussian_posterior_mean(y_obs, A_obs, sigma_obs)
    doc_errors.append(abs(doc_x0 - x0_star))
    dps_errors.append(abs(dps_x0 - x0_star))
    yA_errors.append(abs(yA_x0 - x0_star))
    postmean_errors.append(abs(pm_x0 - x0_star))

    # 保存最后一次试验的可视化数据（随检查点一同归档）
    viz_data = {
        'x0_star': float(x0_star),
        'y_obs': float(y_obs), 'A_obs': float(A_obs), 'sigma_obs': float(sigma_obs),
        'yA_x0': float(yA_x0), 'postmean_x0': float(pm_x0),
        'doc_traj': doc_trajectory.detach().cpu().numpy().tolist(),
        'dps_traj': dps_trajectory.detach().cpu().numpy().tolist(),
        'doc_controls': doc_controls.detach().cpu().numpy().tolist(),
        'doc_history': [np.asarray(th).tolist() for th in doc_history],
    }

    # 每轮保存检查点（meta 记录配置，避免改参后误续传）
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({'meta': {'n_trials': n_trials,
                              'gm_weights': list(GM_WEIGHTS),
                              'gm_means': list(GM_MEANS),
                              'gm_stds': list(GM_STDS)},
                     'doc_errors': doc_errors, 'dps_errors': dps_errors,
                     'yA_errors': yA_errors, 'postmean_errors': postmean_errors,
                     'dps_by_zeta': {str(_k): _v for _k, _v in dps_errors_by_zeta.items()},
                     'viz': viz_data}, f)

# 统计结果
doc_mean_error = np.mean(doc_errors)
doc_std_error = np.std(doc_errors)
dps_mean_error = np.mean(dps_errors)
dps_std_error = np.std(dps_errors)
yA_mean_error = np.mean(yA_errors)
postmean_mean_error = np.mean(postmean_errors)
doc_win_count = int(np.sum(np.asarray(doc_errors) < np.asarray(dps_errors)))

print(f"\n统计结果（n={n_trials}，配对设计）：")
print(f"  DOC平均误差：{doc_mean_error:.6f} ± {doc_std_error:.6f}")
print(f"  DPS平均误差：{dps_mean_error:.6f} ± {dps_std_error:.6f}")
print(f"  参考 y/A（数据一致点）：{yA_mean_error:.6f}")
print(f"  参考 精确后验均值（MMSE下界）：{postmean_mean_error:.6f}")
print(f"  DOC相对DPS改进：{(dps_mean_error - doc_mean_error) / dps_mean_error * 100:.2f}%，"
      f"DOC胜出 {doc_win_count}/{n_trials} 次")
print(f"  口径说明：DOC为PF-ODE确定性轨迹的终端点；DPS为逆向SDE单条后验样本，"
      f"二者均值在n次配对下比较")

# 单次试验可视化（使用最后一次试验的结果）
print("\n单次试验可视化（用于展示轨迹对比）...")

# 若从完整检查点恢复（未执行任何试验），从检查点的viz数据重建可视化变量
if start_trial >= n_trials and doc_errors:
    if viz_data is None:
        raise RuntimeError(f"检查点缺少viz数据（旧格式），请删除{os.path.basename(checkpoint_file)}后重新完整运行")
    x0_star = viz_data['x0_star']
    doc_trajectory = torch.tensor(viz_data['doc_traj'], device=device)
    dps_trajectory = torch.tensor(viz_data['dps_traj'], device=device)
    doc_controls = torch.tensor(viz_data['doc_controls'], device=device)
    doc_history = [np.asarray(th) for th in viz_data['doc_history']]
    yA_x0 = viz_data['yA_x0']
    pm_x0 = viz_data['postmean_x0']
    doc_sampler = SimpleDOC(viz_data['y_obs'], viz_data['A_obs'], viz_data['sigma_obs'])

# 可视化对比（使用最后一次试验的结果）
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 子图1：轨迹对比（横轴为逆向步数：0=纯噪声x_T，20=终端x_0）
axes[0, 0].plot(range(T+1), doc_trajectory.cpu().numpy(), 'o-', label='DOC (PF-ODE全局轨迹优化)',
                linewidth=2, markersize=5)
axes[0, 0].plot(range(T+1), dps_trajectory.cpu().numpy(), 's-', label='DPS (逐步贪心引导)',
                linewidth=2, markersize=5)
axes[0, 0].axhline(y=x0_star, color='green', linestyle='--', linewidth=2,
                   label=f'真实信号 $x^*$={x0_star:.3f}')
axes[0, 0].axhline(y=yA_x0, color='purple', linestyle=':', linewidth=2,
                   label=f'数据一致点 $y/A$={yA_x0:.3f}')
axes[0, 0].axhline(y=pm_x0, color='brown', linestyle='-.', linewidth=1.5, alpha=0.8,
                   label=f'精确后验均值={pm_x0:.3f}')
axes[0, 0].set_xlabel(f'逆向步数（0=纯噪声 $x_T$，{T}=终端 $x_0$）')
axes[0, 0].set_ylabel('信号值 $x_t$')
axes[0, 0].set_title('轨迹对比：DOC vs DPS（最后一次试验示例）')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# 子图2：控制输入（横轴与子图1同向：0=高噪声端 x_T，T-1=末端 x_0 前一步）
# ★ control_sequence 按 t_idx 索引：t_idx=T-1 是首个逆向步（高噪声端），t_idx=0 是末步。
#   必须把数组倒过来画，否则末步会落在横轴 0 上、与子图1"左=高噪声"的方向相反
#   （旧版直接用 range(T) 配原数组，横轴标签与曲线方向互相矛盾）。
axes[0, 1].plot(range(T), doc_controls.flip(0).cpu().numpy(), 'o-',
                label='DOC控制变量 raw_u（实际施加 0.5·tanh(raw_u)）',
                linewidth=2, markersize=5)
axes[0, 1].axhline(y=0, color='gray', linewidth=0.8)
axes[0, 1].set_xlabel(f'逆向步序号（0=高噪声端 $x_T$，{T-1}=末端 $x_0$ 前一步）')
axes[0, 1].set_ylabel('控制输入 raw $u_t$（施加量为 0.5·tanh）')
_b_title = 'DOC控制输入（全轨迹协调，PF-ODE）'
if not IS_MAIN_T:
    # 只标注能从曲线上读出来的事实：控制量级与离饱和线的距离。
    # （控制形状对初始化/迭代次数敏感——重跑前那一版的峰值落在中段，见正文步骤4b 的重跑说明——
    #   所以标题不再对"是否收敛"下判断，避免把某一次优化路径的形态写成结论。）
    _u_abs_max = float(np.abs(doc_controls.cpu().numpy()).max())
    _b_title += f'\n（T={T}：|raw_u| 全程 ≤ {_u_abs_max:.2f}，离 U_MAX=0.5 的饱和线很远）'
axes[0, 1].set_title(_b_title, fontsize=(10 if not IS_MAIN_T else 12))
axes[0, 1].legend(fontsize=(8 if not IS_MAIN_T else 10))
axes[0, 1].grid(True, alpha=0.3)

# 子图3：n次试验的平均终端误差（总体统计，带标准差棒）
methods = ['DOC', 'DPS', 'y/A参考', '后验均值参考']
means = [doc_mean_error, dps_mean_error, yA_mean_error, postmean_mean_error]
stds = [doc_std_error, dps_std_error, np.std(yA_errors), np.std(postmean_errors)]
colors = ['steelblue', 'coral', 'purple', 'saddlebrown']
bars = axes[1, 0].bar(methods, means, yerr=stds, color=colors, alpha=0.75,
                      capsize=5, edgecolor='black', linewidth=0.6)
# ★ 注意：max(means + stds) 是**列表拼接**、不是逐元素求和，旧版据此定轴高，结果只按
#   "最大均值"取上限，DOC 与 DPS 的误差棒上须都被裁掉。逐元素求和后再留 15% 余量。
_y_top = float(np.max(np.asarray(means) + np.asarray(stds))) * 1.15
for bar, m in zip(bars, means):
    # 标签偏移按坐标轴量程取并加白底，避免被误差棒的竖线/帽线穿过（旧版固定 +0.004 会压字）
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, m + 0.035 * _y_top, f'{m:.4f}',
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
axes[1, 0].set_ylabel('$|x_0 - x^*|$ 平均终端误差')
# ★ 口径：T≠20 时"主报 ζ"是按 ζ∝T 外推选的，可能已越界（本实验 T=200 的 ζ=0.3 方差比≈40、
#   单条链过冲到 -6.6）。用"主报 ζ"的分母算改进率等于挑最有利的对照，故标题改用 ζ 扫描里
#   最有利于 DPS 的那一档，并把主报 ζ 越界这件事写进标题。
_zeta_ok = [k for k, v in dps_errors_by_zeta.items() if len(v)]
_best_zeta = min(_zeta_ok, key=lambda k: float(np.mean(dps_errors_by_zeta[k])))
_best_dps_err = float(np.mean(dps_errors_by_zeta[_best_zeta]))
if IS_MAIN_T or _best_dps_err <= 0:
    _c_line2 = (f'DOC相对DPS改进 {(dps_mean_error-doc_mean_error)/dps_mean_error*100:.1f}%，'
                f'DOC胜出 {doc_win_count}/{n_trials} 次')
else:
    _c_line2 = (f'DOC 胜出 {doc_win_count}/{n_trials} 次；对 DPS 最优 ζ={_best_zeta:g} 仍低 '
                f'{(_best_dps_err - doc_mean_error)/_best_dps_err*100:.1f}%'
                f'（主报 ζ={DPS_ZETA_MAIN:g} 已越界，见正文）')
axes[1, 0].set_title(f'终端误差对比（n={n_trials}配对试验，误差棒为±1标准差）\n{_c_line2}',
                     fontsize=(10 if not IS_MAIN_T else 12))
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].set_ylim(0, _y_top)
# 标准差远大于均值时上须会超出轴范围：标注被裁剪的上界，避免读者把"截断"当数据
if dps_mean_error + dps_std_error > 0.98 * _y_top:
    axes[1, 0].text(bars[1].get_x() + bars[1].get_width() / 2, 0.97 * _y_top,
                    f'DPS 误差棒上界 {dps_mean_error + dps_std_error:.2f}（超出轴范围，已裁剪）',
                    ha='center', va='top', fontsize=8, color='0.35')

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
ax_left.set_title('DOC优化过程（确定性PF-ODE）：代价与终端状态收敛')
ax_left.tick_params(axis='y', labelcolor='b')
ax_right.tick_params(axis='y', labelcolor='r')
ax_left.grid(True, alpha=0.3)

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax_left.legend(lines, labels, loc='upper right', fontsize=9)

plt.suptitle(f'DOC vs DPS：全局轨迹优化(PF-ODE) vs 逐步贪心引导\n'
             f'（n={n_trials}次配对试验；参考线：y/A数据一致点、精确后验均值）',
             fontsize=14, y=0.98)
plt.tight_layout()
compare_path = os.path.join(SAVE_DIR, "DOC vs DPS对比.png" if IS_MAIN_T
                            else f"DOC vs DPS对比_T{T}.png")
plt.savefig(compare_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"对比结果已保存: {compare_path}")

print("\n关键发现（基于{}次配对试验统计）：".format(n_trials))
print(f"  DOC平均误差: {doc_mean_error:.6f} ± {doc_std_error:.6f}")
print(f"  DPS平均误差: {dps_mean_error:.6f} ± {dps_std_error:.6f}")
print(f"  y/A参考:     {yA_mean_error:.6f}（终端代价最小化的可达点）")
print(f"  后验均值参考: {postmean_mean_error:.6f}（MMSE点估计下界）")
print(f"  DOC相对DPS改进: {(dps_mean_error - doc_mean_error) / dps_mean_error * 100:.2f}%，"
      f"DOC胜出 {doc_win_count}/{n_trials} 次")
print("  解读：本实验的1D线性高斯+精确得分设定对DPS最有利；DOC受单步控制饱和")
print("  与开环300步优化的限制，MMSE误差未必优于DPS。DOC的优势应在弱得分/")
print("  非线性/少步图像场景中检验，不能把本实验当作'DOC优于DPS'的证明。")

# ============================================================
# 6. 计算代价分析
# ============================================================
print("\n" + "=" * 60)
print("步骤5：DOC计算代价分析")
print("=" * 60)

print("""
DOC计算代价（13.5节）：

1. 朴素BPTT实现的代价（本实验采用的教学简化版）：
   - 每次迭代需前向存储整条轨迹并反向传播（T步链式法则）
   - 300次迭代 ≈ 300次完整采样的代价，远高于DPS的单次采样
   - 1D问题可接受；若直接照搬到图像，存储为T×H×W×C，代价很高

2. 原论文如何让高维问题可行（Li & Pereira, NeurIPS 2024）：
   - iLQR：沿轨迹做局部线性/二次展开，前向-反向扫描求前馈+反馈增益，
     不需要对外层目标做数百次BPTT
   - 随机低秩近似 + 无矩阵(matrix-free)求值：避免显式构造/存储
     H×W量级的Hessian
   - 实测在FFHQ 256×256的超分辨/修复/去模糊上建立了新SOTA，
     因此"DOC图像不实用"只对朴素实现成立，对论文算法不成立

3. 方法对比：

| 方法            | 单样本代价        | 引导超参 | 少步稳定性 | 理论定位         |
|-----------------|-------------------|----------|------------|------------------|
| DOC(本简化版)   | 高(300次BPTT)     | 无引导ζ  | 鲁棒       | 全局轨迹优化(控制饱和) |
| DOC(论文iLQR)   | 中(数次扫描)      | 无引导ζ  | 鲁棒       | 精确后验(可恢复) |
| DPS             | 低(单次采样)      | ζ需细调  | 脆性       | 逐步贪心近似     |

4. 教学意义：
   - 1D下可观察控制饱和/开环优化对DOC终端精度的影响
   - 直观看到DPS引导强度的离散化脆性（ζ窄稳定窗）
   - 理解控制论视角：后验采样是最优控制的特例
""")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("实验13.5-1 完成!")
print("=" * 60)
print(f"""
关键结论:
1. DOC控制论视角（13.5节）
   - 逆向扩散 = 受控动力学系统，控制输入u_t引导轨迹走向最优终端
   - 最优控制问题：最小化 终端代价(-log p(y|x0)) + 控制能量(Tikhonov项)

2. DOC vs DPS对比（本实验n={n_trials}次配对实测）
   - DOC：{doc_mean_error:.4f}±{doc_std_error:.4f}（单步控制饱和 U_MAX=0.5）。
     注意：若不加饱和，末步β_0=0会让DOC退化为y/A投影，误差与
     y/A={yA_mean_error:.4f}在1e-6内相等，不能当作全局优化的证据。
   - DPS：{dps_mean_error:.4f}±{dps_std_error:.4f}，规范链式梯度+标定ζ=0.02；
     ζ≥0.03即出现发散，稳定窗窄，体现"离散化脆性"
   - 本设定下DOC相对DPS的MMSE改进 {(dps_mean_error-doc_mean_error)/dps_mean_error*100:.1f}%
     （胜出{doc_win_count}/{n_trials}次）。1D线性高斯+精确得分对DPS最有利，
     不能据此断言DOC劣于DPS，也不能断言DOC优于DPS——DOC的优势需在
     弱得分/非线性/少步图像场景中检验

3. 实现教训（本实验曾踩的坑，已修复）
   - 末步β_0=0时无约束控制可一步投影到y/A，使"全局轨迹优化"名存实亡；
     必须限制单步控制幅度（本实验用tanh饱和）或改用连续时间PF-ODE离散化
   - DOC优化rollout与评估rollout必须同动力学：带噪优化+确定性评估的
     失配曾使DOC误差虚高至理论最优的约2.6倍
   - DPS梯度必须穿过Tweedie雅可比对x_t求取；逆向末端步不应再加噪
   - VP调度须使ᾱ_T≈0且x_T与p_T边际匹配，否则逆向链从分布外启动

4. 计算代价
   - 朴素BPTT简化版：约300次完整采样的代价，1D可运行
   - 论文iLQR+低秩/无矩阵求值：256×256图像逆问题上实用且SOTA
""")

# ============================================================
# 附加验证：DPS 后验校准诊断（固定观测 y，检验末端样本是否覆盖真实后验）
# ============================================================
print("\n" + "=" * 60)
print("附加验证：DPS末端分布的后验校准（方差覆盖检验）")
print("=" * 60)
print("""
做法：固定观测 y（此时真实后验是一个确定的窄高斯），从 M 个不同 x_T
独立运行DPS，统计终端样本的均值/方差，与解析真实后验对比：
  - 均值应≈后验均值，方差比 Var_DPS / Var_post 应≈1 才是合格后验样本
  - 方差比≪1 即"欠扩散"：样本过度收缩，点误差好看但丢失多样性/不确定度
注：标量版 gaussian_mixture_score 不支持批量输入，此处使用数学等价的
向量化版本；逆向更新公式与 dps_sampling() 逐行一致。
""")

def _batch_gm_score(x, t_idx):
    """gaussian_mixture_score 的批量向量化版本（x 形状 [M]），数学等价"""
    ab_t = alpha_bars[t_idx]
    mus_t = torch.sqrt(ab_t) * torch.tensor(GM_MEANS, device=device, dtype=x.dtype)
    vs_t = ab_t * torch.tensor([s ** 2 for s in GM_STDS], device=device, dtype=x.dtype) \
           + (1.0 - ab_t)
    ws = torch.tensor(GM_WEIGHTS, device=device, dtype=x.dtype)
    logp = torch.log(ws) - 0.5 * (x[:, None] - mus_t[None, :]) ** 2 / vs_t[None, :]
    post_w = torch.softmax(logp, dim=1)
    return torch.sum(post_w * (-(x[:, None] - mus_t[None, :]) / vs_t[None, :]), dim=1)

def _dps_batch(y_val, A_val, sigma_val, M, zeta):
    """与 dps_sampling 相同的更新公式，M 条链并行（仅用于校准诊断）"""
    y_t = torch.tensor(float(y_val), device=device)
    x = torch.randn(M, device=device)
    for t_idx in range(T - 1, -1, -1):
        xr = x.clone().requires_grad_(True)
        sc = _batch_gm_score(xr, t_idx)
        x0 = (xr + (1.0 - alpha_bars[t_idx]) * sc) / torch.sqrt(alpha_bars[t_idx])
        log_lik = -0.5 * (y_t - A_val * x0) ** 2 / (sigma_val ** 2)
        grad_ll = torch.autograd.grad(log_lik.sum(), xr)[0]
        with torch.no_grad():
            mu_post = (x + betas[t_idx] * sc.detach()) / torch.sqrt(alphas[t_idx])
            x = mu_post + zeta * betas[t_idx] * grad_ll \
                + torch.sqrt(beta_tildes[t_idx]) * torch.randn_like(x)
    return x.detach()

# 解析真实后验：等方差等权混合高斯的精确均值/方差（本设定错模权重~1e-14）
_cal_A, _cal_sig, _cal_tau2 = 2.0, 0.1, GM_STDS[0] ** 2
def _theory_posterior(y_val):
    mus = np.asarray(GM_MEANS, dtype=float)
    sy2 = _cal_A ** 2 * _cal_tau2 + _cal_sig ** 2
    vpost = _cal_tau2 * _cal_sig ** 2 / sy2
    logw = -0.5 * (y_val - _cal_A * mus) ** 2 / sy2
    w = np.exp(logw - logw.max()); w /= w.sum()
    mk = mus + (_cal_A * _cal_tau2 / sy2) * (y_val - _cal_A * mus)
    mean = float(np.sum(w * mk))
    var = float(np.sum(w * (vpost + (mk - mean) ** 2)))
    return mean, var

torch.manual_seed(20261004)  # 诊断专用固定种子，保证结果可复现
# 多批次重复设计：每个(y, ζ)独立重复K批、每批M条链。
# 必须报告批次间波动——ζ=0.02附近单批次方差比在0.1~1之间跳变，
# 只跑一批会因随机种子得到互相矛盾的"结论"，波动本身正是脆性的证据。
_cal_M, _cal_K, _cal_zetas, _cal_ys = 3000, 5, (0.01, 0.02, 0.025, 0.03), (4.0, -4.0)
if not IS_MAIN_T:
    # 补充对照只求量级（主实验的完整校准诊断不重复），否则 T 步数会把耗时放大 T/20 倍
    _cal_M, _cal_K = 200, 2
    _cal_zetas = tuple(DPS_ZETA_SWEEP)
_ratios02 = []                       # 汇总主报 ζ 在所有(y,批次)上的方差比
_cal_log = []                        # (y, ζ, 方差比批次均值, 尾部逃逸率)，供结论段自动定性
_cal_raw = {}                        # ζ -> 所有(y,批次)的方差比原始值（用于报中位数）
print(f"每个设定独立重复{_cal_K}批×{_cal_M}条链，报告批次均值±标准差\n")
for _y in _cal_ys:
    _th_mean, _th_var = _theory_posterior(_y)
    _th_sd = _th_var ** 0.5
    print(f"固定 y={_y:+.1f}（A={_cal_A}, σ={_cal_sig}）：真实后验 "
          f"均值={_th_mean:+.4f}，方差={_th_var:.5f}（sd={_th_sd:.4f}）")
    for _zeta in _cal_zetas:
        _rs, _bias, _tail, _div = [], [], [], []
        for _ in range(_cal_K):
            _samp = _dps_batch(_y, _cal_A, _cal_sig, _cal_M, _zeta)
            _finite = torch.isfinite(_samp)
            _xx = _samp[_finite]
            _div.append(1.0 - float(_finite.float().mean()))
            _rs.append(float(_xx.var(unbiased=False)) / _th_var)
            _bias.append(float(_xx.mean()) - _th_mean)
            _tail.append(float(((_xx - _th_mean).abs() > 3.0 * _th_sd).float().mean()))
        _rs, _bias, _tail, _div = map(np.asarray, (_rs, _bias, _tail, _div))
        if abs(_zeta - DPS_ZETA_MAIN) < 1e-12:
            _ratios02.extend(_rs.tolist())
        _cal_log.append((_y, _zeta, float(_rs.mean()), float(_tail.mean())))
        _cal_raw.setdefault(_zeta, []).extend(_rs.tolist())
        print(f"  ζ={_zeta:<5} 方差比={_rs.mean():5.2f}±{_rs.std(ddof=1):4.2f} "
              f"(范围{_rs.min():.2f}~{_rs.max():.2f})  均值偏差={_bias.mean():+.4f}  "
              f"尾部逃逸(>3σ)={_tail.mean():5.1%}  非有限={_div.mean():.1%}")
    print()

_r02 = np.asarray(_ratios02, dtype=float)
calib_ratio_002 = float(_r02.mean()) if _r02.size else float('nan')
calib_ratio_002_cv = float(_r02.std(ddof=1) / _r02.mean()) if _r02.size > 1 else float('nan')
print(f"诊断结论（ζ={DPS_ZETA_MAIN:g}=主报取值，{len(_r02)}个批次、每批{_cal_M}条链汇总）：")
print(f"  主体批次方差比 = {calib_ratio_002:.2f} ± {_r02.std(ddof=1):.2f}"
      f"（CV={calib_ratio_002_cv:.0%}，范围{_r02.min():.2f}~{_r02.max():.2f}；理想=1）")


def _verdict(r):
    """按实测量方差比自动定性，避免把某个 T 的结论写死在代码里"""
    if not np.isfinite(r):
        return "发散（非有限）"
    if r < 0.5:
        return "欠扩散：方差明显偏小，多样性被低估"
    if r <= 2.0:
        return "接近校准"
    if r <= 10.0:
        return "过扩散：方差偏大，轨迹在漂"
    return "发散：方差比失控"


for _z in _cal_zetas:
    _rr = np.array([row[2] for row in _cal_log if abs(row[1] - _z) < 1e-12])
    _tt = np.array([row[3] for row in _cal_log if abs(row[1] - _z) < 1e-12])
    if not _rr.size:
        continue
    # 定性看中位数：均方差比会被个别重尾批次（方差比可达数十）抬高，不适合直接下结论
    _med = float(np.median(_cal_raw[_z]))
    print(f"  ζ={_z:<5} 方差比 均值={_rr.mean():9.3g}  中位数={_med:7.3g}  尾部逃逸={_tt.mean():5.1%}"
          f"  → {_verdict(_med)}")

print(f"  以上为 T={T}（首步 β≈{float(betas[0]):.3g}）下的实测：本基准中没有任何 ζ 能在不同随机批次下")
print("  稳定给出校准的后验覆盖——同一 ζ 换 T（即换 β_t 尺度）后定性会整体平移。")
print("  注意：只跑单批会被'个别逃逸轨迹抬高方差'误导（方差比可能偶然接近1），")
print("  必须多批次重复才能识别。终端点误差对方差覆盖不敏感：误差好看的 ζ 往往")
print("  正是收缩（欠扩散）那一档，不能据此判断后验样本质量，多样性要单独检查。")

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

def _r(x, nd=6):
    """round 的 NaN 安全版（NaN 在 JSON 中非法，统一写 None）"""
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None

# 每个 T 的补充条目（主实验 T=20 也写一份，便于两块结果同结构对比）
_sweep_entry = {
    "试验次数": n_trials,
    "T": T,
    "DPS_主报zeta": DPS_ZETA_MAIN,
    "DOC_平均误差": _r(doc_mean_error),
    "DOC_误差标准差": _r(doc_std_error),
    "DPS_平均误差": _r(dps_mean_error),
    "DPS_误差标准差": _r(dps_std_error),
    "参考_yA数据一致点_平均误差": _r(yA_mean_error),
    "参考_精确后验均值_平均误差": _r(postmean_mean_error),
    "DOC相对改进百分比": _r((dps_mean_error - doc_mean_error) / dps_mean_error * 100, 2),
    "DOC胜出次数": doc_win_count,
    "DPS_各zeta平均误差": {str(_k): _r(np.mean(_v)) for _k, _v in dps_errors_by_zeta.items() if len(_v)},
    "DPS_各zeta误差标准差": {str(_k): _r(np.std(_v)) for _k, _v in dps_errors_by_zeta.items() if len(_v)},
    # 后验校准诊断：方差比 Var_DPS/Var_post 按 ζ 分工，理想为 1（正文据此判"欠扩散/过扩散"）
    # 均值与中位数一并存：ζ 接近发散边界时会出现个别重尾批次，只看均值会被带偏
    "DPS后验校准_各zeta方差比": {
        str(_z): _r(float(np.mean([row[2] for row in _cal_log if abs(row[1] - _z) < 1e-12])), 3)
        for _z in _cal_zetas
        if any(abs(row[1] - _z) < 1e-12 for row in _cal_log)
    },
    "DPS后验校准_各zeta方差比中位数": {
        str(_z): _r(float(np.median(_cal_raw[_z])), 3) for _z in _cal_zetas if _cal_raw.get(_z)
    },
    "后验校准_诊断规模": f"{len(_cal_ys)}个y × {_cal_K}批 × {_cal_M}条链；噪声由 torch 全局种子生成，与计算设备相关",
}

_json_path = os.path.join(SAVE_DIR, 'results_summary.json')

# 先读出已有 JSON：重跑主实验时也要保留其它 T 的补充条目（否则会把 T=200 那块抹掉）
_prev = {}
if os.path.exists(_json_path):
    with open(_json_path, encoding='utf-8') as f:
        _prev = json.load(f)

_sweep = dict(_prev.get("T扫描补充", {}))
_sweep[str(T)] = _sweep_entry

if IS_MAIN_T:
    results_summary = {
        "试验次数": n_trials,
        "实验版本": "修复版v3：PF-ODE确定性轨迹优化 + 控制tanh饱和(U_MAX=0.5) + DPS链式梯度 + VP调度修正(末态ᾱ_T≈0.01)",
        "DOC_平均误差": round(float(doc_mean_error), 6),
        "DOC_误差标准差": round(float(doc_std_error), 6),
        "DPS_平均误差": round(float(dps_mean_error), 6),
        "DPS_误差标准差": round(float(dps_std_error), 6),
        "参考_yA数据一致点_平均误差": round(float(yA_mean_error), 6),
        "参考_精确后验均值_平均误差": round(float(postmean_mean_error), 6),
        "DOC相对改进百分比": round(float((dps_mean_error - doc_mean_error) / dps_mean_error * 100), 2),
        "DOC胜出次数": doc_win_count,
        "DPS引导系数zeta": DPS_ZETA_MAIN,
        "DPS后验校准_末端方差比_zeta0.02_理想为1": _r(calib_ratio_002, 3),
        "DPS后验校准_方差比批次间变异系数CV": _r(calib_ratio_002_cv, 3),
        "实验设备": str(device),
        "T扫描补充": _sweep,
    }
else:
    # 补充对照：只并入 T扫描补充，不动主实验字段、不覆盖主图
    results_summary = dict(_prev)
    results_summary["T扫描补充"] = _sweep

results_summary = _to_native(results_summary)
with open(_json_path, 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {_json_path}")
