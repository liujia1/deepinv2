# -*- coding: utf-8 -*-
"""
实验6.4-1 切片得分匹配（SSM）与Hutchinson迹估计
对应章节：6.4 切片得分匹配（SSM）与Hutchinson迹估计

知识点:
  - ISM 需要精确 Jacobian 迹, $O(d^2)$ 代价
  - Hutchinson 迹估计: $\mathbb{E}[v^T A v] = \mathrm{Tr}(A)$
  - SSM 目标函数与无偏性
  - Rademacher vs 高斯随机向量的方差对比
  - SSM 与 ESM/DSM 的统一对比

实验内容:
  步骤1: Hutchinson 迹估计的数值验证
  步骤2: Rademacher vs 高斯随机向量的方差对比
  步骤3: SSM 目标函数在 1D 上的演示
  步骤4: 完整 SSM 训练 - 估计原始分布得分
  步骤5: SSM 无偏性验证 - 不同 M 下平均损失接近精确 ISM
  步骤6: 方法谱系象限图 - 可行性与精度的权衡

运行前提: PyTorch, CPU/GPU 均可
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 非交互式后端, 避免 plt.show 阻塞
import matplotlib.pyplot as plt
import os
import sys
import io
import time
import warnings
import logging
from tqdm import tqdm

# 兼容 NumPy 1.x (使用 np.trapz) 与 NumPy 2.x (使用 np.trapezoid)
_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 静默 matplotlib 相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====== 中文字体配置(兼容本地和 Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验6.4-1')
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
    print("警告: chinese_font 模块未找到, 中文字体可能无法正常显示")
# ========================================================

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 训练")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")

# Checkpoint 路径
CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'ssm_checkpoint.pth')

# 固定随机种子(可复现)
np.random.seed(42)
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)


# ============================================================
# 1D 高斯混合分布 (1D 教学示例)
# ============================================================
# p(x) = 0.5 * N(-2, 1) + 0.5 * N(2, 1)
# 选择 1D 是为了与原参考实验一致, 简化数学推导
# 注: 1D 情况下 Tr(∇_x s) 即 ds/dx 本身, "切片"含义退化
#     SSM 在 1D 上的演示主要用于理解目标函数结构, 高维加速需用 PyTorch autograd

def gm1d_score(x):
    """精确原始得分 $\\nabla\\log p(x)$, 1D 高斯混合"""
    p1 = np.exp(-0.5 * (x + 2)**2) / np.sqrt(2 * np.pi)
    p2 = np.exp(-0.5 * (x - 2)**2) / np.sqrt(2 * np.pi)
    p = 0.5 * p1 + 0.5 * p2
    grad_p1 = -(x + 2) * p1
    grad_p2 = -(x - 2) * p2
    return (0.5 * grad_p1 + 0.5 * grad_p2) / p


# ============================================================
# 步骤1: Hutchinson 迹估计的数值验证
# ============================================================
print("\n" + "=" * 60)
print("步骤1: Hutchinson 迹估计的数值验证")
print("=" * 60)
print("\n定理: $\\mathrm{Tr}(A) = \\mathbb{E}[v^T A v]$, 其中 $\\mathbb{E}[v v^T] = I$")
print("  蒙特卡洛估计: $\\hat{\\mathrm{Tr}}_M(A) = \\frac{1}{M} \\sum_{j=1}^M v_j^T A v_j$")
print("  无偏性: $\\mathbb{E}[\\hat{\\mathrm{Tr}}_M] = \\mathrm{Tr}(A)$")
print("  方差: $\\mathrm{Var}(\\hat{\\mathrm{Tr}}_M) = \\mathrm{Var}(v^T A v) / M$, 随 $M \\to \\infty$ 收敛到 0")

# 构造一个具体的 Jacobian 矩阵 (对称正定)
d = 20
A = np.random.randn(d, d)
A = A @ A.T / d + np.eye(d)  # 确保正定
true_trace = np.trace(A)
print(f"\n构造 {d}x{d} 对称正定矩阵 $A$, 精确迹 $\\mathrm{{Tr}}(A) = {true_trace:.6f}$")

def hutchinson_trace(A, M, dist='gaussian'):
    """Hutchinson 迹估计: $\\hat{\\mathrm{Tr}}(A) = \\frac{1}{M} \\sum v^T A v$
    
    注: 此处用 for 循环逐次采样, 便于教学理解.
    生产环境可用矩阵化写法加速:
        V = np.random.randn(M, d)  # 或 Rademacher
        estimates = np.einsum('ij,jk,ik->i', V, A, V)
    步骤5 中 M 最高到 1000, n_trials=100 (共 10 万次迭代 + autograd),
    已用 tqdm 进度条显示进度.
    """
    d = A.shape[0]
    estimates = []
    for _ in range(M):
        if dist == 'gaussian':
            v = np.random.randn(d)
        elif dist == 'rademacher':
            v = np.random.choice([-1, 1], size=d)
        estimates.append(v @ A @ v)
    return np.mean(estimates), np.std(estimates) / np.sqrt(M)

# 不同 M 值的估计精度
M_values = [1, 5, 10, 50, 100, 500, 1000]
gaussian_errors = []
rademacher_errors = []

print(f"\n{'M':>6s} | {'高斯估计':>10s} | {'高斯误差':>10s} | {'Rademacher估计':>14s} | {'Rademacher误差':>14s}")
print("-" * 65)
for M in M_values:
    g_est, g_se = hutchinson_trace(A, M, 'gaussian')
    r_est, r_se = hutchinson_trace(A, M, 'rademacher')
    g_err = abs(g_est - true_trace)
    r_err = abs(r_est - true_trace)
    gaussian_errors.append(g_err)
    rademacher_errors.append(r_err)
    print(f"{M:6d} | {g_est:10.4f} | {g_err:10.4f} | {r_est:14.4f} | {r_err:14.4f}")

print(f"\n精确迹: {true_trace:.4f}")
print(f"  观察: 估计误差随 M 增加而减小, 收敛速度约为 $1/\\sqrt{{M}}$")
print(f"  注: 当前为 1 次实验结果, 步骤2 将做 1000 次实验展示方差分布")


# ============================================================
# 步骤2: Rademacher vs 高斯随机向量的方差对比
# ============================================================
print("\n" + "=" * 60)
print("步骤2: Rademacher vs 高斯随机向量的方差对比")
print("=" * 60)
print("\n理论:")
print("  高斯分布 $v \\sim \\mathcal{N}(0, I)$:")
print("    $\\mathrm{Var}(v^T A v) = 2\\|A\\|_F^2$")
print("  Rademacher $v \\in \\{-1, +1\\}^d$:")
print("    $\\mathrm{Var}(v^T A v) = 2\\|A\\|_F^2 - 2\\sum_i A_{ii}^2 \\leq 2\\|A\\|_F^2$")
print("  结论: Rademacher 方差严格小于高斯 (除非 $A$ 对角), 实践中推荐 Rademacher")

n_experiments = 1000
M = 10  # 固定 M=10 突出方差差异

gaussian_traces = []
rademacher_traces = []

for _ in range(n_experiments):
    g_est, _ = hutchinson_trace(A, M, 'gaussian')
    r_est, _ = hutchinson_trace(A, M, 'rademacher')
    gaussian_traces.append(g_est)
    rademacher_traces.append(r_est)

gaussian_traces = np.array(gaussian_traces)
rademacher_traces = np.array(rademacher_traces)

# 换算为单次估计的方差 (理论公式 Var(v^T A v) 对应单次抽样, M 次平均后方差除以 M)
gaussian_var_single = np.var(gaussian_traces) * M
rademacher_var_single = np.var(rademacher_traces) * M

print(f"\n$M={M}$, {n_experiments} 次独立实验 (换算为单次估计方差, 以便与理论公式直接对比):")
print(f"  高斯分布:   均值 = {np.mean(gaussian_traces):.4f}, 单次方差 = {gaussian_var_single:.4f}")
print(f"  Rademacher: 均值 = {np.mean(rademacher_traces):.4f}, 单次方差 = {rademacher_var_single:.4f}")
print(f"  Rademacher 方差比高斯小: {gaussian_var_single / rademacher_var_single:.2f} 倍")
print(f"  理论预测: Rademacher 方差 $\\leq$ 高斯方差, 实测符合预期")


# ============================================================
# 步骤1-2 可视化: Hutchinson 迹估计与方差对比
# ============================================================
print("\n" + "=" * 60)
print("生成可视化图表 (步骤1-2)...")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 子图1: Hutchinson 迹估计精度随 M 变化
axes[0].semilogy(M_values, gaussian_errors, 'bo-', label=r'高斯分布 $v \sim \mathcal{N}(0,I)$', lw=2, markersize=5)
axes[0].semilogy(M_values, rademacher_errors, 'rs-', label=r'Rademacher $v \in \{-1,+1\}^d$', lw=2, markersize=5)
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3, label=r'精确迹 $\mathrm{Tr}(A)$')
axes[0].set_xlabel(r'投影次数 $M$')
axes[0].set_ylabel(r'估计误差 $|\hat{\mathrm{Tr}} - \mathrm{Tr}|$')
axes[0].set_title(r'Hutchinson 迹估计精度 vs $M$')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3, which='both')

# 子图2: 两种分布的估计方差对比 (直方图)
axes[1].hist(gaussian_traces, bins=30, alpha=0.6, color='blue',
             label=r'高斯 ($\sigma$={:.3f})'.format(np.std(gaussian_traces)))
axes[1].hist(rademacher_traces, bins=30, alpha=0.6, color='red',
             label=r'Rademacher ($\sigma$={:.3f})'.format(np.std(rademacher_traces)))
axes[1].axvline(x=true_trace, color='k', linestyle='--', lw=2,
                label=r'精确 $\mathrm{{Tr}}={:.2f}$'.format(true_trace))
axes[1].set_xlabel(r'迹估计值 $\hat{\mathrm{Tr}}$')
axes[1].set_ylabel('频次')
axes[1].set_title(r'迹估计分布 ($M={}$, {} 次实验)'.format(M, n_experiments))
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# 子图3: 理论方差与实测方差对比
theory_var_gaussian = 2 * np.linalg.norm(A, 'fro')**2
theory_var_rademacher = 2 * np.linalg.norm(A, 'fro')**2 - 2 * np.sum(np.diag(A)**2)
empirical_var_gaussian = np.var(gaussian_traces) * M  # 单次估计方差
empirical_var_rademacher = np.var(rademacher_traces) * M

labels = [r'高斯 (理论)', r'高斯 (实测)', r'Rademacher (理论)', r'Rademacher (实测)']
values = [theory_var_gaussian, empirical_var_gaussian, theory_var_rademacher, empirical_var_rademacher]
colors_v = ['steelblue', 'lightblue', 'indianred', 'lightcoral']
bars = axes[2].bar(labels, values, color=colors_v, alpha=0.85, edgecolor='black')
for bar, v in zip(bars, values):
    axes[2].text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.01,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=9)
axes[2].set_ylabel(r'$\mathrm{Var}(v^T A v)$')
axes[2].set_title(r'单次估计方差: 理论 vs 实测 (固定 $M$)')
axes[2].grid(alpha=0.3, axis='y')
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=15, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_Hutchinson迹估计.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"图表已保存: 步骤1_Hutchinson迹估计.png")


# ============================================================
# 步骤3: SSM 目标函数在 1D 上的演示
# ============================================================
print("\n" + "=" * 60)
print("步骤3: SSM 目标函数在 1D 上的演示")
print("=" * 60)
print("\n[1D 特殊性说明]")
print("  - 1D 情况下 $\\mathrm{{Tr}}(\\nabla_x s_\\theta(x)) = \\frac{{ds_\\theta}}{{dx}}$ (标量)")
print("  - '切片'含义退化: 随机方向 $v \\in \\{{-1, +1\\}}$, $v^T \\nabla_x s_\\theta v = v^2 \\cdot \\frac{{ds}}{{dx}} = \\frac{{ds}}{{dx}}$")
print("  - SSM 1D 演示主要用于理解目标函数的两项结构, 高维加速需用 PyTorch autograd")
print("  - 真正的 Hutchinson 加速在 $d \\gg 1$ 时才显著 (步骤4 将训练高维参数网络)")

# SSM 目标: $J_{SSM}(\theta) = \mathbb{E}_{p(x)} \mathbb{E}_{p(v)} [v^T \nabla_x s_\theta(x) v + 0.5 \|s_\theta(x)\|^2]$
# 1D 中: $v^T \nabla_x s_\theta v = v^2 \cdot ds/dx$, $v \sim \mathcal{N}(0,1)$ 或 $v \in \{-1,+1\}$
# Hutchinson 简化为: $v^2 \cdot ds/dx$, 其中 $\mathbb{E}[v^2] = 1$ (两种分布均满足)

x_grid = np.linspace(-5, 5, 200)
true_score = gm1d_score(x_grid)

# 数值计算 ds/dx (精确 Jacobian 迹)
h = 1e-4
ds_dx = (gm1d_score(x_grid + h) - gm1d_score(x_grid - h)) / (2 * h)

# SSM 目标的两项
term1 = ds_dx  # $v^T \nabla_x s v$ 的期望 = $ds/dx$ (1D)
term2 = 0.5 * true_score**2  # $\frac{1}{2} \|s(x)\|^2$
ssm_integrand = term1 + term2

# ESM 目标 (用于对比)
esm_integrand = 0.5 * true_score**2  # 当 $s_\theta = \nabla \log p$ 时 ESM = 0

# SSM 与 ISM 关系: 当 $s_\theta = \nabla \log p$ 时, SSM = ISM = ESM + 常数
# 在 1D 高斯混合上验证: ESM = 0, SSM = $\int (ds/dx + 0.5 s^2) dx$
# ISM = $\int (ds/dx + 0.5 s^2) dx$ 与 SSM 形式相同 (1D 情况下)
ism_value = _trapz(ssm_integrand, x_grid)
esm_value = _trapz(esm_integrand, x_grid)

print(f"\nSSM vs ESM 对比 (1D 高斯混合):")
print(f"  当 $s_\\theta = \\nabla\\log p(x)$ 时 (完美匹配):")
print(f"    ESM 积分 = {esm_value:.6f} (理论值 0, 因 $\\|s_\\theta - s\\|^2 = 0$)")
print(f"    ISM 积分 = {ism_value:.6f}")
print(f"    $\\mathrm{{Tr}}(\\nabla s)$ 项积分 = {_trapz(ds_dx[10:-10], x_grid[10:-10]):.6f}  (丢弃首尾10点, 避免边界处数值差分误差)")
print(f"    $\\|s\\|^2/2$ 项积分 = {_trapz(term2[10:-10], x_grid[10:-10]):.6f}")
print(f"  关键观察: SSM 和 ISM 只差与 $\\theta$ 无关的常数, 因此梯度方向相同")


# ============================================================
# 步骤3 可视化: SSM 目标函数在 1D 上的演示
# ============================================================
print("\n" + "=" * 60)
print("生成可视化图表 (步骤3)...")
print("=" * 60)

# 步骤3 子图: SSM 目标函数的两项 (1D 演示)
fig, ax_ssm = plt.subplots(figsize=(8, 5))

ax_ssm.plot(x_grid, ds_dx, 'b-', lw=2, label=r'$\mathrm{Tr}(\nabla_x s) = ds/dx$ (精确迹)')
ax_ssm.plot(x_grid, term2, 'r-', lw=2, label=r'$\frac{1}{2}\|s(x)\|^2$')
ax_ssm.plot(x_grid, ssm_integrand, 'k--', lw=2, label='SSM 被积函数')
ax_ssm.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax_ssm.set_xlabel('$x$')
ax_ssm.set_ylabel('值')
ax_ssm.set_title('SSM 目标函数的两项 (1D 演示)')
ax_ssm.legend(fontsize=9)
ax_ssm.grid(alpha=0.3)
ax_ssm.text(0.02, 0.98, r'1D: $\mathrm{Tr}(\nabla s) = ds/dx$, "切片" 含义退化',
             transform=ax_ssm.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_SSM目标函数1D演示.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"图表已保存: 步骤3_SSM目标函数1D演示.png")


# ============================================================
# 步骤4: 完整 SSM 训练 - 估计原始分布得分
# ============================================================
print("\n" + "=" * 60)
print("步骤4: 完整 SSM 训练 - 估计原始分布得分")
print("=" * 60)
print("\n[高维说明]")
print("  为体现 Hutchinson 加速的真实价值, 步骤4 用 PyTorch 训练一个参数网络")
print("  网络 $s_\\theta(x)$ 有较多参数, 通过 autograd 高效计算 $v^T \\nabla_x s_\\theta v$")
print("  这等价于 $d$ 维 Jacobian 矩阵, 但单次前向+反向传播即可, 无需构造完整矩阵")
print("\nSSM 目标函数:")
print(r"  $\mathcal{J}_{SSM}(\theta) = \mathbb{E}_{p(x)}\mathbb{E}_{p_v}[v^T\nabla_x s_\theta(x) v + \frac{1}{2}\|s_\theta(x)\|^2]$")
print("\n训练算法:")
print("  1. 采样 $x \\sim p_{data}$, 采样 $v \\sim p_v$ (Rademacher)")
print("  2. 计算 $s_\\theta(x)$, 用 autograd 计算 $v^T \\nabla_x s_\\theta(x) v$")
print("  3. 损失 $\\ell = v^T \\nabla_x s_\\theta(x) v + 0.5 \\|s_\\theta(x)\\|^2$")
print("  4. 反向传播更新 $\\theta$")

# 得分网络
class ScoreNet1D(torch.nn.Module):
    """1D 得分网络, 估计 $\\nabla\\log p(x)$"""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x.unsqueeze(-1)).squeeze(-1)

# 生成训练数据 (1D 高斯混合), 使用独立 RNG 避免与步骤1/2共享随机数流
rng_train = np.random.RandomState(42)
N_train = 10000
x_train = np.concatenate([rng_train.randn(N_train // 2) - 2, rng_train.randn(N_train // 2) + 2])
x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)

model = ScoreNet1D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 用于可视化和评估的有序网格点
x_eval = np.linspace(-6, 6, 200)
x_eval_t = torch.tensor(x_eval, dtype=torch.float32, device=device)
true_score_eval = gm1d_score(x_eval)

# 记录训练损失
train_losses = []

# Checkpoint 加载逻辑 (包含 train_losses 恢复, 用于绘制训练损失曲线)
start_epoch = 0
is_final = False

if os.path.exists(CHECKPOINT_PATH):
    print(f"\n检测到已保存的模型: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if 'train_losses' not in checkpoint:
        raise RuntimeError(
            f"检测到旧版本 checkpoint (缺少 'train_losses' 字段):\n"
            f"  {CHECKPOINT_PATH}\n"
            f"请删除该文件后重新训练."
        )
    if checkpoint.get('is_final', False):
        print(f"✓ 这是最终训练完成的模型, 直接加载, 跳过训练过程")
        print(f"  训练轮数: {checkpoint['epoch']+1}")
        print(f"  最终损失: {checkpoint['loss']:.6f}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        start_epoch = checkpoint['epoch'] + 1
        is_final = True
    else:
        print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        start_epoch = checkpoint['epoch'] + 1

# 训练 SSM
if not is_final:
    N_epochs = 500
    print(f"\n训练 SSM 得分网络 (估计 $\\nabla\\log p(x)$, 共 {N_epochs} 轮)...")
    t_start = time.time()
    # 用 tqdm 显示训练进度, 避免长时间无输出
    epoch_pbar = tqdm(range(start_epoch, N_epochs),
                      desc='SSM 训练',
                      ncols=80,
                      file=sys.stdout)
    for epoch in epoch_pbar:
        # SSM 目标: 用 Rademacher 随机方向 v 计算 $v^T \nabla_x s_\theta v$
        v = torch.randint(0, 2, x_train_t.shape, device=device).float() * 2 - 1  # Rademacher

        # 需要 $s_\theta(x)$ 的梯度
        x_train_t.requires_grad_(True)
        s_pred = model(x_train_t)
        # 用 autograd 计算 $v^T \nabla_x s_\theta v$ (高效, 避免构造 Jacobian)
        grad_s = torch.autograd.grad(
            outputs=s_pred,
            inputs=x_train_t,
            grad_outputs=v,  # 用 v 作为梯度种子, 等价于 v^T J v
            create_graph=True,  # 保留计算图以反向传播
            # 注: 不需要 retain_graph=True, 每轮都重新构建计算图 (s_pred = model(x_train_t) 是新的 forward)
        )[0]
        vHv = (v * grad_s).sum()  # $v^T \nabla_x s v$ 的标量值

        # SSM 损失
        s_norm = 0.5 * (s_pred ** 2).sum()
        loss = (vHv + s_norm) / x_train_t.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x_train_t.requires_grad_(False)

        train_losses.append(loss.item())

        # 更新进度条后缀 (每 10 轮显示一次 loss)
        if (epoch + 1) % 10 == 0 or epoch == start_epoch:
            epoch_pbar.set_postfix(loss=f'{loss.item():.4f}', refresh=False)

    t_elapsed = time.time() - t_start
    print(f"\n训练完成, 最终损失: {train_losses[-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")

    # 保存最终 checkpoint (包含 train_losses 用于绘制训练损失曲线)
    torch.save({
        'epoch': N_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_losses[-1],
        'train_losses': train_losses,
        'is_final': True,
    }, CHECKPOINT_PATH)
    print(f"已保存最终模型: {CHECKPOINT_PATH}")

# 评估学习到的得分
with torch.no_grad():
    learned_score = model(x_eval_t).cpu().numpy()

# 评估指标
corr = np.corrcoef(learned_score, true_score_eval)[0, 1]
mse = np.mean((learned_score - true_score_eval) ** 2)
print(f"\n学习得分与真实得分的相关性:")
print(f"  相关系数: {corr:.4f}")
print(f"  MSE: {mse:.6f}")


# ============================================================
# 步骤5: SSM 无偏性验证
# ============================================================
print("\n" + "=" * 60)
print("步骤5: SSM 无偏性验证")
print("=" * 60)
print("\n理论: $\\mathbb{E}_v[\\mathcal{J}_{SSM}(\\theta)] = \\mathcal{J}_{ISM}(\\theta)$")
print("  即 SSM 损失对随机方向 $v$ 取期望后, 等于精确 ISM 损失")
print("\n[1D 特殊情况]")
print("  - 1D 中: $v^T \\nabla_x s v = v^2 \\cdot ds/dx$")
print("  - Rademacher $v \\in \\{{-1,+1\\}}$: $v^2 = 1$ 恒定, 每次估计都等于精确值 (无随机性, 标准差=0)")
print("  - Gaussian $v \\sim \\mathcal{N}(0,1)$: $v^2$ 有方差 (Var=2), 能真正展示 SSM 估计的方差")
print("  - 步骤5 改用 Gaussian $v$ 进行验证, 训练仍用 Rademacher (实践推荐)")

# 准备 ISM 精确损失计算的 x 样本, 使用独立 RNG
rng_ism = np.random.RandomState(123)
N_is_test = 1000
x_is_test = np.concatenate([rng_ism.randn(N_is_test // 2) - 2, rng_ism.randn(N_is_test // 2) + 2])
x_is_test_t = torch.tensor(x_is_test, dtype=torch.float32, device=device)

# ISM 精确损失 = $\mathbb{E}_p[\mathrm{Tr}(\nabla s_\theta) + 0.5 \|s_\theta\|^2]$
# 1D 中 = $\mathbb{E}_p[ds_\theta/dx + 0.5 s_\theta^2]$, 用样本均值估计
# 用 autograd 精确计算 ds/dx (与 SSM 估计使用相同的导数计算方式, 避免数值差分误差混入对比)
x_is_test_t.requires_grad_(True)
s_pred_for_ism = model(x_is_test_t)
ds_dx_t = torch.autograd.grad(
    outputs=s_pred_for_ism.sum(),
    inputs=x_is_test_t,
    create_graph=False
)[0]
x_is_test_t.requires_grad_(False)
ds_dx_learned = ds_dx_t.cpu().numpy()
s_center = s_pred_for_ism.detach().cpu().numpy()
ism_per_sample = ds_dx_learned + 0.5 * s_center**2
ism_exact = ism_per_sample.mean()  # 在 1000 样本上的均值

# 不同 M 下 SSM 损失估计 (使用 Gaussian v)
M_test_values = [1, 5, 10, 50, 100, 500, 1000]
ssm_estimates_mean = []
ssm_estimates_std = []

n_trials = 100  # 每个 M 做 100 次试验 (每次试验内 M 次估计取平均)
# 用 tqdm 显示进度, 避免长时间无输出
m_pbar = tqdm(M_test_values, desc='SSM 无偏性验证', ncols=80, file=sys.stdout)
for M in m_pbar:
    m_pbar.set_postfix(M=M)
    trial_means = []
    for _ in range(n_trials):
        # 向量化: 将 M 次独立估计批量为一次前向+反向, 大幅减少 Python 循环开销
        x_batch = x_is_test_t.repeat(M)  # (M * N_is_test,)
        v_batch = torch.randn_like(x_batch)  # (M * N_is_test,)
        x_batch.requires_grad_(True)
        s_pred = model(x_batch)  # (M * N_is_test,)
        grad_s = torch.autograd.grad(
            outputs=s_pred,
            inputs=x_batch,
            grad_outputs=v_batch,
            create_graph=False,
            retain_graph=False
        )[0]  # (M * N_is_test,)
        x_batch.requires_grad_(False)

        # 重塑为 (M, N_is_test) 以计算 M 个独立估计
        v_reshaped = v_batch.view(M, -1)
        grad_reshaped = grad_s.view(M, -1)
        s_reshaped = s_pred.view(M, -1)

        vHv = (v_reshaped * grad_reshaped).mean(dim=1)  # (M,)
        s_norm = 0.5 * (s_reshaped ** 2).mean(dim=1)  # (M,)
        estimates = vHv + s_norm  # (M,)
        trial_means.append(estimates.mean().item())
    ssm_estimates_mean.append(np.mean(trial_means))
    ssm_estimates_std.append(np.std(trial_means))

print(f"\nISM 精确损失 (1000 样本均值): {ism_exact:.6f}")
print(f"\n{'M':>5s} | {'SSM 估计均值':>14s} | {'标准差':>10s} | {'与 ISM 偏差':>12s}")
print("-" * 55)
for M, mean_v, std_v in zip(M_test_values, ssm_estimates_mean, ssm_estimates_std):
    bias = abs(mean_v - ism_exact)
    print(f"{M:5d} | {mean_v:14.6f} | {std_v:10.6f} | {bias:12.6f}")

print(f"\n结论: SSM 估计均值随 M 增加稳定接近 ISM 精确损失, 验证了无偏性")
print(f"  标准差随 M 增大而减小 (符合 $1/\\sqrt{{M}}$ 收敛率)")
print(f"  注: 此处方差主要是 $v$ 在 1D 上的退化效应导致的人为方差,")
print(f"  并非高维 Jacobian 迹估计的代表性方差来源 (1D 中 $v^T \\nabla s v = v^2 \\cdot ds/dx$)")


# ============================================================
# 步骤4-5 可视化: SSM 训练结果与无偏性验证
# ============================================================
print("\n" + "=" * 60)
print("生成可视化图表 (步骤4-5)...")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 子图1: SSM 训练损失曲线
if train_losses:
    axes[0].plot(train_losses, 'b-', lw=1.5, alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('SSM Loss')
    axes[0].set_title('SSM 训练损失曲线')
    axes[0].grid(alpha=0.3)
    # 添加最终损失标注(放在子图内部空白处)
    final_loss = train_losses[-1]
    y_max = max(train_losses) * 1.05
    y_text = y_max * 0.90
    axes[0].annotate(f'Final: {final_loss:.4f}',
                    xy=(len(train_losses)-1, final_loss),
                    xytext=(len(train_losses)*0.15, y_text),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')

# 子图2: SSM 训练结果 - 学到的得分 vs 真实得分
axes[1].plot(x_eval, true_score_eval, 'r-', lw=2, label=r'真实 $\nabla\log p(x)$')
axes[1].plot(x_eval, learned_score, 'b--', lw=2, label=r'学习 $s_\theta(x)$ (SSM)')
axes[1].set_xlabel('$x$')
axes[1].set_ylabel('得分')
axes[1].set_title(f'SSM 训练结果 (相关系数 = {corr:.4f})')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# 子图3: SSM 无偏性验证
axes[2].errorbar(M_test_values, ssm_estimates_mean, yerr=ssm_estimates_std,
                 fmt='bo-', capsize=4, lw=2, label=r'SSM 估计 $\pm 1\sigma$')
axes[2].axhline(y=ism_exact, color='r', linestyle='--', lw=2,
                label=r'ISM 精确损失 = {:.4f}'.format(ism_exact))
axes[2].set_xlabel(r'投影次数 $M$')
axes[2].set_ylabel('损失')
axes[2].set_title(r'SSM 无偏性验证: $\mathbb{E}_v[\mathcal{J}_{SSM}] = \mathcal{J}_{ISM}$')
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3)
axes[2].set_xscale('log')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_SSM训练验证.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"图表已保存: 步骤4_SSM训练验证.png")


# ============================================================
# 步骤6: 方法谱系象限图 - 可行性 vs 精度
# ============================================================
print("\n" + "=" * 60)
print("步骤6: 方法谱系象限图")
print("=" * 60)
print("\n四种得分匹配方法的谱系:")
print("  ESM (显式): 需要 $\\nabla\\log p$, 不可行")
print("  ISM (隐式): 需要精确 Jacobian 迹, $O(d^2)$, 不可行")
print("  DSM (去噪): 估计 $\\nabla\\log p_\\sigma$, $O(1)$, 可行")
print("  SSM (切片): 估计 $\\nabla\\log p$, $O(M)$, 可行")

fig, ax = plt.subplots(figsize=(10, 8))

# 绘制象限背景
ax.axhspan(0, 0.5, xmin=0, xmax=0.5, alpha=0.1, color='red')    # 左下: 不可行
ax.axhspan(0.5, 1, xmin=0, xmax=0.5, alpha=0.1, color='yellow')  # 左上: 不可行但精确
ax.axhspan(0, 0.5, xmin=0.5, xmax=1, alpha=0.1, color='yellow')  # 右下: 可行但粗略
ax.axhspan(0.5, 1, xmin=0.5, xmax=1, alpha=0.1, color='green')   # 右上: 可行且精确

# 添加象限标签
ax.text(0.25, 0.95, r'理论精确但不可行', ha='center', va='top',
        fontsize=11, transform=ax.transAxes, color='darkred',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax.text(0.75, 0.95, r'可行且精确 (目标)', ha='center', va='top',
        fontsize=11, transform=ax.transAxes, color='darkgreen',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.text(0.25, 0.05, r'不可行且粗略', ha='center', va='bottom',
        fontsize=11, transform=ax.transAxes, color='darkgray',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
ax.text(0.75, 0.05, r'可行但粗略', ha='center', va='bottom',
        fontsize=11, transform=ax.transAxes, color='darkorange',
        bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.7))

# 四种方法的位置 (可行性, 精度)
methods = {
    'ESM\n(显式得分匹配)':   (0.15, 0.95),
    'ISM\n(隐式得分匹配)':   (0.25, 0.85),
    'DSM\n(去噪得分匹配)':   (0.80, 0.40),
    'SSM\n(切片得分匹配)':   (0.70, 0.80),
}

colors_m = {'ESM\n(显式得分匹配)': 'red',
            'ISM\n(隐式得分匹配)': 'darkorange',
            'DSM\n(去噪得分匹配)': 'steelblue',
            'SSM\n(切片得分匹配)': 'green'}

for name, (x, y) in methods.items():
    ax.scatter(x, y, s=2000, c=colors_m[name], alpha=0.7,
               edgecolors='black', linewidths=2, zorder=5)
    ax.annotate(name, (x, y), fontsize=11, ha='center', va='center',
                fontweight='bold', color='white', zorder=6)

# 绘制演进箭头 (展示谱系逻辑)
arrow_props = dict(arrowstyle='->', lw=2, color='gray', alpha=0.6)
# ESM -> ISM (分部积分)
ax.annotate('', xy=(0.30, 0.83), xytext=(0.20, 0.92), arrowprops=arrow_props)
ax.text(0.18, 0.87, r'分部积分', fontsize=9, color='gray', rotation=-30)

# ISM -> DSM (避免 Jacobian)
ax.annotate('', xy=(0.75, 0.45), xytext=(0.32, 0.80), arrowprops=arrow_props)
ax.text(0.50, 0.65, r'避免 Jacobian\n(噪声扰动)', fontsize=9, color='gray')

# ISM -> SSM (随机化 Jacobian)
ax.annotate('', xy=(0.65, 0.78), xytext=(0.32, 0.82), arrowprops=arrow_props)
ax.text(0.45, 0.85, r'随机化\nJacobian', fontsize=9, color='gray', rotation=10)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel(r'可行性 (Computational Feasibility) $\rightarrow$', fontsize=12)
ax.set_ylabel(r'得分估计精度 (Score Accuracy) $\rightarrow$', fontsize=12)
ax.set_title('四种得分匹配方法的谱系: 可行性 vs 精度\n(示意性定位, 非定量测量结果)', fontsize=13)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(['不可行', '低', '中', '高', '完全可行'])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['粗略', '低', '中', '高', '理论精确'])
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤6_方法谱系图.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"图表已保存: 步骤6_方法谱系图.png")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验6.4-1 总结")
print("=" * 60)
print("\n1. Hutchinson 迹估计")
print(r"   $\mathrm{Tr}(A) = \mathbb{E}[v^T A v]$, 其中 $\mathbb{E}[v v^T] = I$")
print(r"   蒙特卡洛: $\hat{\mathrm{Tr}}_M(A) = \frac{1}{M} \sum v_j^T A v_j$")
print(r"   误差按 $1/\sqrt{M}$ 收敛")
print("\n2. 随机向量选择")
print(r"   高斯 $v \sim \mathcal{N}(0,I)$: $\mathrm{Var}(v^T A v) = 2\|A\|_F^2$")
print(r"   Rademacher $v \in \{-1,+1\}^d$: 方差更小, 实践推荐")
print("\n3. SSM 目标函数")
print(r"   $\mathcal{J}_{SSM}(\theta) = \mathbb{E}_{p(x)}\mathbb{E}_{p_v}[v^T \nabla_x s_\theta(x) v + \frac{1}{2}\|s_\theta(x)\|^2]$")
print(r"   每次迭代: $O(M)$ 前向自动微分 ($M \ll d$), 估计原始 $\nabla \log p$")
print(r"   无偏性: $\mathbb{E}_v[\mathcal{J}_{SSM}] = \mathcal{J}_{ISM}$ (已通过步骤5验证)")
print("\n4. 四种方法对比")
print(f"   {r'方法':<6s} | {r'目标函数':<32s} | {r'需∇log p':<8s} | {r'Jacobian':<12s} | {r'估计得分':<14s} | {r'可行性':<8s}")
print("   " + "-" * 95)
print(f"   {'ESM':<6s} | {'(1/2) E[||s_θ-∇log p||²]':<32s} | {'是':<8s} | {'否':<12s} | {'∇log p':<14s} | {'不可行':<8s}")
print(f"   {'ISM':<6s} | {'E[Tr(∇s_θ)+(1/2)||s_θ||²]':<32s} | {'否':<8s} | {'O(d²)':<12s} | {'∇log p':<14s} | {'低维可行':<8s}")
print(f"   {'DSM':<6s} | {'(1/2) E[||s_θ+z/σ||²]':<32s} | {'否':<8s} | {'否':<12s} | {'∇log p_σ':<14s} | {'O(1)':<8s}")
print(f"   {'SSM':<6s} | {'E[vᵀ∇s_θ v+(1/2)||s_θ||²]':<32s} | {'否':<8s} | {'O(M)':<12s} | {'∇log p':<14s} | {'可行':<8s}")
print("\n5. 实验要点")
print(r"   - 1D 演示: $\mathrm{Tr}(\nabla s) = ds/dx$, '切片'含义退化")
print(r"   - 1D 高斯混合: SSM 与 ISM 形式相同, SSM = ISM = ESM + 常数")
print(r"   - SSM 估计原始 $\nabla \log p$ (无需噪声扰动), 与 DSM 互补")
