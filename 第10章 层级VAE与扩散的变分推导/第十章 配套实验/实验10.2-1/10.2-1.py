# -*- coding: utf-8 -*-
"""
实验10.2-1 扩散过程的变分下界推导
对应章节: 10.2 扩散过程的变分下界推导

知识点:
  - 高斯编码器=加噪过程 (马尔可夫高斯链的闭式边际分布)
  - VLB三项分解: L_T + ΣL_{t-1} + L_0
  - 前向过程后验的闭式解

实验内容:
  步骤1: 高斯编码器=加噪过程验证 (直接采样 vs 逐步加噪)
  步骤2: VLB三项分解验证 (L_T, L_{t-1}, L_0)
  步骤3: SNR分析与噪声调度可视化

本实验不需要GPU，通过数值实验验证10.2节的核心数学结论。
"""

import sys
import io
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import warnings

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
    SAVE_DIR = os.path.join(_gdrive, '实验10.2-1')
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
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)

print("\n" + "="*60)
print("实验10.2-1: 扩散过程的变分下界推导")
print("="*60)
print("对应章节: 10.2 扩散过程的变分下界推导")
print("使用设备: CPU (本实验不需要GPU)")


# ============================================================
# 噪声调度定义
# ============================================================
T = 1000  # 扩散步数
beta_min, beta_max = 1e-4, 0.02

# 线性噪声调度 (DDPM)
betas = torch.linspace(beta_min, beta_max, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

print("\n噪声调度: beta从" + str(beta_min) + "线性增至" + str(beta_max) + ", T=" + str(T))
print("ab_1=" + str(round(alpha_bars[0].item(), 6)) + ", ab_500=" + str(round(alpha_bars[499].item(), 6)) + ", ab_T=" + str(round(alpha_bars[999].item(), 6)))

# ★ 自检：打印典型T值的调度结果，方便读者直接验证公式推导
# 典型T的预期输出（修改beta_min/beta_max/T后这里会同步刷新）：
#   T=10,  [1e-4, 2e-2] -> t=1=0.9999, t=6=0.966663, t=10=0.903739
#   T=100, [1e-4, 2e-2] -> t=1=0.9999, t=51=0.769291, t=100=0.363563
#   T=1000,[1e-4, 2e-2] -> t=1=0.9999, t=501=0.077797, t=1000=0.00004
# 检验规则：(1) ab_t 单调递减；(2) ab_T 接近但不等于0（线性调度末态仍有微弱信号）
print("[调度自检] 几组典型(T, beta_min, beta_max)的α̅_t关键值(实际运行, 非手算):")
for _T, _bmin, _bmax in [(10, 1e-4, 2e-2), (100, 1e-4, 2e-2), (1000, 1e-4, 2e-2)]:
    _betas = torch.linspace(_bmin, _bmax, _T)
    _alphas = 1.0 - _betas
    _ab = torch.cumprod(_alphas, dim=0)
    _indices = [0, _T // 2, _T - 1]
    _vals = [round(_ab[i].item(), 6) for i in _indices]
    _labels = [f"t={i+1}" for i in _indices]
    print(f"  T={_T:4d}, beta=[{_bmin}, {_bmax}] -> " + ", ".join(f"{lab}={v}" for lab, v in zip(_labels, _vals)))


# ============================================================
# 步骤1：高斯编码器=加噪过程验证 —— 10.2节核心对应
# ============================================================
print("\n" + "="*60)
print("步骤1：高斯编码器=加噪过程验证")
print("="*60)
print("\n[核心思想]")
print("  当编码器采用固定方差的高斯转移时，编码链等价于逐步加噪:")
print("  q(x_t|x_{t-1}) = N(x_t; sqrt(alpha_t) x_{t-1}, (1-alpha_t)I)")
print("  → q(x_t|x_0) = N(x_t; sqrt(ab_t) x_0, (1-ab_t)I)")

# 数值验证：直接采样 vs 逐步加噪 (两条独立采样链)
N = 5000  # 大样本量确保统计显著性
torch.manual_seed(42)
x0 = torch.randn(N, 2)  # N个2D样本
n_steps_list = [1, 50, 200, 500, 1000]  # 1-indexed 时间步 (与步骤3、JSON 的 t 记法一致; t=T 即 x_T)

print("\n直接采样 vs 逐步加噪 (" + str(N) + "个样本, 2维):")
print("    t      均值差(max)    标准差差(max)    MC误差尺度    条件期望    理论标准差")
print("-" * 90)

step1_stats = {}  # 步骤1统计量 (写入 results_summary.json)

# 为每个t创建独立的Generator，确保可复现性（跨版本稳定）
base_seed = 42

for t_1idx in n_steps_list:
    t = t_1idx - 1  # alpha_bars 数组的 0-indexed 位置: alpha_bars[t] = ᾱ_{t_1idx}
    sqrt_ab = torch.sqrt(alpha_bars[t])
    sqrt_1mab = torch.sqrt(1 - alpha_bars[t])

    # 均值差的蒙特卡洛误差尺度: Var(mean_direct - mean_step) = 2(1-ab_t)/N (每维; 两链噪声独立)
    # mean_diff 与 mc_scale 同量级即说明两过程统计不可区分
    mc_scale = float(torch.sqrt(2 * (1 - alpha_bars[t]) / N))

    # 为当前t创建两个独立的Generator
    g_direct = torch.Generator().manual_seed(base_seed + t_1idx * 1000)
    g_step = torch.Generator().manual_seed(base_seed + t_1idx * 1000 + 1)

    # 直接采样: x_t = sqrt(ab_t) * x_0 + sqrt(1-ab_t) * eps
    eps_direct = torch.randn_like(x0, generator=g_direct)
    x_t_direct = sqrt_ab * x0 + sqrt_1mab * eps_direct

    # 逐步加噪 (独立随机性，与直接采样解耦)
    x_t_step = x0.clone()
    for s in range(t_1idx):  # t_1idx 次转移: x_0 → x_{t_1idx}
        noise = torch.randn_like(x_t_step, generator=g_step)
        x_t_step = torch.sqrt(alphas[s]) * x_t_step + torch.sqrt(betas[s]) * noise

    # 比较两条独立链的样本统计量
    mean_direct = x_t_direct.mean(dim=0)
    mean_step = x_t_step.mean(dim=0)
    std_direct = x_t_direct.std(dim=0)
    std_step = x_t_step.std(dim=0)

    mean_diff = (mean_direct - mean_step).abs().max().item()
    std_diff = (std_direct - std_step).abs().max().item()

    step1_stats["t_" + str(t_1idx)] = {
        'mean_diff_max': mean_diff,
        'std_diff_max': std_diff,
        'mc_error_scale': mc_scale,
    }

    # 条件期望值 (以当前x0样本为条件，而非绝对理论期望0)
    mean_conditional = sqrt_ab * x0.mean(dim=0)
    std_theory = sqrt_1mab

    print(str(t_1idx).rjust(5) + "  " + str(round(mean_diff, 6)).rjust(14) + "  " + str(round(std_diff, 6)).rjust(14) +
          "  " + str(round(mc_scale, 6)).rjust(14) +
          "  " + str(round(mean_conditional[0].item(), 6)).rjust(14) + "  " + str(round(std_theory.item(), 6)).rjust(14))

print("\n→ 各时间步上, 均值差与'MC误差尺度'同量级 → 差异纯属有限样本噪声, 两过程统计不可区分:")
print("  支持闭式边际分布 q(x_t|x_0) = N(sqrt(ab_t)x_0, (1-ab_t)I)")
print("  注: 该式代数上精确成立; 数值实验受蒙特卡洛精度 O(1/sqrt(N)) 限制, 验证的是统计不可区分性")
print("  这是10.2节的核心结论：高斯马尔可夫链的闭式边际分布")
print("\n  注：标准差差异是更强的判别指标——")
print("     均值本身因 E[x_0]≈0 而趋于0，判别力弱；")
print("     而方差项 (1-ab_t) 不退化为0，能更好区分两个过程是否等价")


# ============================================================
# 步骤2：VLB三项分解验证 —— 10.2节公式
# ============================================================
print("\n" + "="*60)
print("步骤2：VLB三项分解验证")
print("="*60)
print("\n[核心思想]")
print("  变分下界分解为三项:")
print("  L_VLB = L_T + sum_{t=2}^T L_{t-1} + L_0")
print("  - L_T: 先验匹配项 D_KL(q(x_T|x_0) || p(x_T))")
print("  - L_{t-1}: 一致性项 D_KL(q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t))")
print("  - L_0: 重建项 -E[log p_theta(x_0|x_1)]")

# L_T: 先验匹配项 (两端均为已知高斯, 闭式KL可直接数值计算)
s2_T = (1 - alpha_bars[-1]).item()            # q 的方差 s^2 = 1-ab_T
mean_x0_sq = (x0 ** 2).mean().item()          # 经验 E[x_0^2] (标准正态数据 ≈ 1)
# KL(N(m, s^2·I) || N(0, I)) 每维 = 0.5*[m^2 + s^2 - 1 - ln(s^2)], 其中 m^2 的样本均值 = ab_T·E[x_0^2]
L_T_per_dim = 0.5 * (alpha_bars[-1].item() * mean_x0_sq + (s2_T - 1 - float(np.log(s2_T))))
print("\nL_T (先验匹配项):")
print("  q(x_T|x_0) = N(sqrt(ab_T)x_0, (1-ab_T)I)")
print("  p(x_T) = N(0, I)")
print("  ab_T = " + str(round(alpha_bars[-1].item(), 8)) + " → sqrt(ab_T) ≈ 0, 1-ab_T ≈ 1")
print("  闭式KL (每维): L_T = 0.5*[ab_T*E[x_0^2] + (s^2 - 1 - ln s^2)], s^2 = 1-ab_T")
print("    E[x_0^2]=" + str(round(mean_x0_sq, 4)) + " → L_T ≈ " + "{:.3e}".format(L_T_per_dim) +
      " nats/维 → 确实≈0 (该性质固有于线性调度 ab_T≈e^{-Σβ}≈4e-5, 非近似口号)")

# 前向过程后验 (10.2节闭式解)
alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])

# 验证后验方差
posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
print("\n前向过程后验方差 tilde_beta_t:")
print("  t=1:   tilde_beta_1 = " + str(round(posterior_var[0].item(), 6)))
print("  t=500: tilde_beta_500 = " + str(round(posterior_var[499].item(), 6)))
print("  t=T:   tilde_beta_T = " + str(round(posterior_var[999].item(), 6)))

# 验证后验方差与前向方差的相对误差 (DDPM简化为tilde_beta_t≈beta_t的关键依据)
rel_err = (posterior_var - betas) / betas
abs_err = (posterior_var - betas).abs()
# t=1是边界点: alpha_bars_prev[0]=1导致后验方差公式分子为0 (1-ab_0=0)
# 故tilde_beta_1=0, 相对误差为-100%, 仅在t=1发生
# 实际分析相对误差时应从t=2开始
rel_err_t2_onwards = rel_err[1:]
# 注意: β̃_t < β_t 恒成立, 相对误差全为负, 故"最大相对误差"须按绝对值(最负)取, 即 .min()
worst_rel = rel_err_t2_onwards.min()
worst_rel_t = int(rel_err_t2_onwards.argmin().item()) + 2
max_abs_err = abs_err[1:].max()
max_abs_err_t = int(abs_err[1:].argmax().item()) + 2
_lt01 = (rel_err_t2_onwards.abs() < 1e-3).nonzero(as_tuple=True)[0]
t_cross_01pct = int(_lt01[0].item()) + 2 if len(_lt01) else None
print("\n  关键观察: 相对误差 (tilde_beta_t - beta_t) / beta_t")
print("    t=1:     -100%   (边界: tilde_beta_1=0, 因ab_0=1使公式分子为0)")
print("    t=2:     " + str(round(rel_err_t2_onwards[0].item() * 100, 4)) + "%")
print("    t=500:   " + str(round(rel_err_t2_onwards[498].item() * 100, 4)) + "%")
print("    t=T-1:   " + str(round(rel_err_t2_onwards[-2].item() * 100, 4)) + "%")
print("    t=T:     " + str(round(rel_err_t2_onwards[-1].item() * 100, 4)) + "%")
print("    最大相对误差(按绝对值, 忽略t=1): " + str(round(worst_rel.item() * 100, 4)) + "% (在t=" + str(worst_rel_t) + ")")
print("    最大绝对误差 |β̃_t-β_t| (忽略t=1): " + str(round(max_abs_err.item(), 6)) + " (在t=" + str(max_abs_err_t) + ")")
print("  → 相对误差在 t>=2 并非都 < 0.1%: 小 t 处偏差极大 (t=2 约 -54.5%), 直到 t≈" + str(t_cross_01pct) + " 才降到 0.1% 以内")
print("  → 为何 DDPM 仍可用 σ_t²=β_t 近似 β̃_t (三条独立理由, 而非仅凭'绝对误差小'):")
print("    (1) 训练不受影响: 简化目标 L_simple=||ε-ε_θ||² 不含 σ_t², 该选择只作用于采样阶段")
print("    (2) 采样影响可忽略: t≈" + str(t_cross_01pct) + " 之后两方差几乎重合; 小 t 处二者同为 1e-4 量级 (绝对差 ≤" +
      str(round(max_abs_err.item(), 6)) + "), 对末段采样的随机性扰动可忽略")
print("    (3) 实证支持: Improved DDPM (Nichol & Dhariwal 2021) 表明 σ 的具体取值对样本质量不敏感, 且可学习")

# 验证后验均值公式 (10.2节)
print("\n后验均值系数 (10.2节):")
print("  tilde_mu_t = [sqrt(alpha_t)(1-ab_{t-1})/(1-ab_t)] x_t + [sqrt(ab_{t-1})(1-alpha_t)/(1-ab_t)] x_0")

for t_idx in [0, 99, 499, 999]:
    t = t_idx
    a_t = alphas[t]
    ab_t = alpha_bars[t]
    ab_prev = alpha_bars_prev[t]

    coeff_xt = torch.sqrt(a_t) * (1 - ab_prev) / (1 - ab_t)
    coeff_x0 = torch.sqrt(ab_prev) * (1 - a_t) / (1 - ab_t)

    print("  t=" + str(t+1).rjust(4) + ": x_t系数=" + str(round(coeff_xt.item(), 4)) + ", x_0系数=" + str(round(coeff_x0.item(), 4)) + ", 和=" + str(round((coeff_xt+coeff_x0).item(), 4)))

# 三项的数值可评估性说明 (为何本实验只直接计算 L_T)
print("\n三项分解的数值可评估性:")
print("  - L_T    : 两端均为已知高斯 → 闭式KL直接计算 (见上, ≈" + "{:.1e}".format(L_T_per_dim) + " nats/维)")
print("  - L_{t-1}: 当 p_theta 取前向真后验时 KL≡0, 验证重点即后验闭式解本身 (已由上方 β̃_t 与均值系数完成)")
print("  - L_0    : 需要训练好的解码器 p_theta(x_0|x_1), 无模型实验无法数值评估 (固有限制, 留待10.3节训练后评估)")

# L_0: 重建项
print("\nL_0 (重建项): 离散解码器或高斯解码器的负对数似然")
print("  对连续数据通常用高斯: -log p(x_0|x_1) ∝ ||x_0 - mu_theta(x_1,1)||^2")


# ============================================================
# 步骤3：SNR分析
# ============================================================
print("\n" + "="*60)
print("步骤3：SNR分析")
print("="*60)
print("\n信噪比(SNR)随时间变化:")
print("  SNR(t) = ab_t / (1-ab_t)")

for t_idx in [0, 99, 249, 499, 749, 999]:
    snr = alpha_bars[t_idx] / (1 - alpha_bars[t_idx])
    snr_db = 10 * np.log10(snr.item()) if snr.item() > 0 else -np.inf
    print("  t=" + str(t_idx+1).rjust(4) + ": ab=" + str(round(alpha_bars[t_idx].item(), 6)) + ", SNR=" + str(round(snr.item(), 4)) + " (" + str(round(snr_db, 1)) + " dB)")

print("\n关键观察:")
print("  - t小: 信号占优，SNR高")
print("  - t大: 噪声占优，SNR低")
print("  - 当ab_t→0时，SNR→0，图像完全被噪声淹没")


# ============================================================
# 可视化
# ============================================================
print("\n" + "="*60)
print("生成可视化图表...")
print("="*60)

fig = plt.figure(figsize=(14, 11))
gs_outer = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)
gs_c = gs_outer[1, 0].subgridspec(2, 1, hspace=0.62)

# (a) 噪声调度与ab_t
ax = fig.add_subplot(gs_outer[0, 0])
t_range = range(1, T + 1)
ax.plot(t_range, alpha_bars.numpy(), 'b-', linewidth=2, label=r'$\bar{\alpha}_t$')
ax.plot(t_range, torch.sqrt(alpha_bars).numpy(), 'r--', linewidth=1.5, label=r'$\sqrt{\bar{\alpha}_t}$ ($x_0$系数)')
ax.plot(t_range, torch.sqrt(1 - alpha_bars).numpy(), 'g--', linewidth=1.5, label=r'$\sqrt{1-\bar{\alpha}_t}$ (噪声系数)')
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('系数值', fontsize=12)
ax.set_title('(a) DDPM线性噪声调度', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', alpha=0.3)

# (b) SNR随时间变化
ax = fig.add_subplot(gs_outer[0, 1])
snr = alpha_bars / (1 - alpha_bars)
snr_db = 10 * torch.log10(snr)
ax.plot(t_range, snr_db.numpy(), 'purple', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='SNR=0 dB')
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('SNR (dB)', fontsize=12)
ax.set_title('(b) 信噪比(SNR)随时间衰减', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.annotate('信号占优', xy=(0.1, 0.8), xycoords='axes fraction',
            fontsize=11, color='blue', fontweight='bold')
ax.annotate('噪声占优', xy=(0.7, 0.2), xycoords='axes fraction',
            fontsize=11, color='red', fontweight='bold')

# (c-上) 前向过程后验方差 vs 前向方差
ax = fig.add_subplot(gs_c[0])
ax.plot(t_range[1:], posterior_var[1:].numpy(), 'b-', linewidth=2, label=r'$\tilde{\beta}_t$ (后验方差)')
ax.plot(t_range[1:], betas[1:].numpy(), 'r--', linewidth=1.5, label=r'$\beta_t$ (前向方差)')
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('方差', fontsize=12)
ax.set_title(r'(c-上) 前向过程后验方差 vs 前向方差 ($\tilde{\beta}_t \approx \beta_t$)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (c-下) 相对误差: 对数轴放大显示二者的微小差异 (线性轴下 t>50 后曲线贴 0 无法分辨)
ax = fig.add_subplot(gs_c[1])
rel_err_pct = np.abs(rel_err[1:].numpy()) * 100  # |相对误差|, 从t=2开始 (跳过t=1的-100%退化点)
ax.semilogy(t_range[1:], rel_err_pct, 'b-', linewidth=2)
# 标注: t=2 误差最大 (最不近似处), t=T 误差最小 (最近似处)
worst_idx = 0  # rel_err_pct[0] 对应 t=2, 是最大的|相对误差|
worst_t = worst_idx + 2
worst_val = rel_err_pct[worst_idx]
ax.annotate(r'最大误差 $t=2$: ' + f'{worst_val:.2f}%',
            xy=(worst_t, worst_val), xytext=(0.40, 0.55),
            textcoords='axes fraction',
            fontsize=10, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))
# 在曲线右端标注最小值
best_val = rel_err_pct[-1]
ax.annotate(r'$t=T$: ' + f'{best_val:.2e}%',
            xy=(T, best_val), xytext=(0.60, 0.15),
            textcoords='axes fraction',
            fontsize=10, color='blue', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6))
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('|相对误差| (%)', fontsize=12)
ax.set_title(r'(c-下) 相对误差幅值 $\left|\frac{\tilde{\beta}_t - \beta_t}{\beta_t}\right|$ (对数轴)', fontsize=13)
ax.grid(True, alpha=0.3, which='both')

# (d) 2D扩散过程可视化
ax = fig.add_subplot(gs_outer[1, 1])
np.random.seed(42)
n_pts = 200
# 两个高斯模式
mode1 = np.random.randn(n_pts // 2, 2) * 0.3 + np.array([1.0, 0.0])
mode2 = np.random.randn(n_pts // 2, 2) * 0.3 + np.array([-1.0, 0.0])
x0_2d = np.vstack([mode1, mode2])

colors = np.array([0] * (n_pts // 2) + [1] * (n_pts // 2))
t_show = [0, 50, 200, 500, 1000]  # 1-indexed 时间步 (t=0 表示原始数据 x_0)

for i, t_idx in enumerate(t_show):
    ab = 1.0 if t_idx == 0 else alpha_bars[t_idx - 1].item()  # ᾱ_t (1-indexed)
    noise = np.random.randn(n_pts, 2)
    x_t = np.sqrt(ab) * x0_2d + np.sqrt(1 - ab) * noise
    offset_x = i * 4
    ax.scatter(x_t[:, 0] + offset_x, x_t[:, 1], c=colors, cmap='coolwarm',
               alpha=0.5, s=8)
    ax.text(offset_x, -2.5, '$t=' + str(t_idx) + '$', ha='center', fontsize=10)
    if i == 0:
        ax.text(offset_x, 2.8, '$x_0$', ha='center', fontsize=11, fontweight='bold')
    elif i == len(t_show) - 1:
        ax.text(offset_x, 2.8, '$x_T \\approx N(0,I)$', ha='center', fontsize=11, fontweight='bold')

ax.set_xlabel('前向加噪过程 →', fontsize=12)
ax.set_title('(d) 高斯编码器=加噪过程 (2D可视化)', fontsize=13)
ax.set_ylim(-3, 3.5)
ax.grid(True, alpha=0.3)
ax.set_yticks([])

fig_path = os.path.join(SAVE_DIR, '步骤1_噪声调度与VLB分解.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print("图表已保存: 步骤1_噪声调度与VLB分解.png")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "="*60)
print("实验10.2-1 总结")
print("="*60)
print("\n关键结论:")
print("\n1. 高斯编码器=加噪过程 (步骤1)")
print("   - 马尔可夫高斯链的闭式边际分布: q(x_t|x_0) = N(sqrt(ab_t)x_0, (1-ab_t)I)")
print("   - 直接采样公式允许高效地从任意时间步采样")
print("\n2. VLB三项分解 (步骤2)")
print("   - L_T≈0: 先验匹配项，当ab_T≈0时自动满足 (闭式KL数值验证: ≈" + "{:.1e}".format(L_T_per_dim) + " nats/维)")
print("   - L_{t-1}: 一致性项，核心优化目标")
print("   - L_0: 重建项")
print("\n3. 前向过程后验闭式解 (步骤2)")
print("   - 后验均值: tilde_mu_t = [sqrt(alpha_t)(1-ab_{t-1})/(1-ab_t)] x_t + [sqrt(ab_{t-1})(1-alpha_t)/(1-ab_t)] x_0")
print("   - 后验方差: tilde_beta_t = (1-ab_{t-1})/(1-ab_t) * beta_t")
print("\n4. SNR分析 (步骤3)")
print("   - SNR随t单调衰减，从信号占优到噪声占优")
print("   - 这是10.3节中损失权重差异的根源")

print("\n" + "="*60)
print("实验10.2-1 完成!")

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
    'noise_schedule': {
        'T': T,
        'beta_min': beta_min,
        'beta_max': beta_max,
        'alpha_bar_1': alpha_bars[0].item(),
        'alpha_bar_500': alpha_bars[499].item(),
        'alpha_bar_T': alpha_bars[-1].item(),
    },
    'step1_closed_form_check': step1_stats,
    'L_T_prior_matching': {
        'alpha_bar_T': alpha_bars[-1].item(),
        'sqrt_alpha_bar_T': torch.sqrt(alpha_bars[-1]).item(),
        'one_minus_alpha_bar_T': (1 - alpha_bars[-1]).item(),
        'per_dim_KL_nats': L_T_per_dim,
        'empirical_mean_x0_sq': mean_x0_sq,
    },
    'posterior_variance': {
        'tilde_beta_1': posterior_var[0].item(),
        'tilde_beta_500': posterior_var[499].item(),
        'tilde_beta_T': posterior_var[999].item(),
    },
    'relative_error_worst_pct': (rel_err[1:].min().item() * 100),  # 按绝对值取最大(最负)的相对误差, 在t=2
    'relative_error_worst_t': 2,
    'relative_error_first_below_0p1pct_t': t_cross_01pct,            # 相对误差首次降到0.1%以内的步数
    'abs_error_max': (abs_err[1:].max().item()),                     # 最大绝对误差(极小), 支撑DDPM β_t近似
    'SNR': {
        f't_{t+1}': {'alpha_bar': alpha_bars[t].item(), 'SNR': (alpha_bars[t] / (1 - alpha_bars[t])).item()}
        for t in [0, 99, 249, 499, 749, 999]
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")