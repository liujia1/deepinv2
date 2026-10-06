# -*- coding: utf-8 -*-
"""
实验10.4-2 层级VAE的前向与逆向过程 (编码/解码 ↔ 加噪/去噪)
对应章节: 10.4 层级 VAE → 扩散的极限

知识点:
  - 层级VAE编码器=逐步加噪: q(x_t|x_{t-1}) = N(sqrt(alpha_t)x_{t-1}, beta_t)
  - 层级VAE解码器=逐步去噪: p(x_{t-1}|x_t) 用正向后验均值 tilde_mu 还原上一步
  - 前向把干净信号逐步埋进噪声，逆向按后验均值一步步洗回来
  - 若已经知道 x0 (oracle)，理想解码器可"精确反演"，往返误差≈0 与层数 L 无关
  - 但解码器通常不知道 x0，必须从 x_t 估计 x0 → 往返出现误差，
    这正是去噪网络(10.3)要学的东西：去逼近理想后验均值

实验内容:
  步骤1: 层级VAE前向过程 (逐层加噪, 观察信号被噪声淹没)
  步骤2: 理想解码器 = 后验均值 (Oracle 已知 x0) → 往返精确可逆
  步骤3: 不完美解码器 (只知 x_t, x0 需估计) → 往返误差非零, 说明需训练去噪器

本实验不需要GPU，通过数值实验验证10.4节的"编码/解码 ↔ 加噪/去噪"对应。
与实验10.4-1的区别: 10.4-1验证"离散→连续极限"(L→∞),
  本实验聚焦"离散层级链本身怎么加噪/去噪"(编码器与解码器的机理)。
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
    SAVE_DIR = os.path.join(_gdrive, '实验10.4-2')
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

# 注: 本脚本所有随机性均由显式 np.random.default_rng(seed) 控制,
# 因此无需(也不应)调用顶层 np.random.seed(), 以免误以为它控制全局随机。
print("\n" + "="*60)
print("实验10.4-2: 层级VAE的前向与逆向过程")
print("="*60)
print("对应章节: 10.4 层级 VAE → 扩散的极限")
print("使用设备: CPU (本实验不需要GPU)")
print("说明: 与实验10.4-1互补——10.4-1看L→∞连续极限, 本实验看离散层级链怎么加噪/去噪")


# ============================================================
# 工具函数: 层级VAE前向(逐层加噪) 与 逆向(逐层去噪)
# ============================================================
def forward_chain(x0, L, beta_total=1.0, rng=None):
    """层级VAE前向: 从干净 x0 出发逐层加噪。

    编码器 q(x_t|x_{t-1}) = N(sqrt(alpha_t) x_{t-1}, beta_t):
        x_t = sqrt(alpha) x_{t-1} + sqrt(beta) eps
    返回 shape = (L+1,) 的每一步状态, 首元素为 x0。
    """
    if rng is None:
        rng = np.random.default_rng(42)
    beta = beta_total / L          # 每层加噪比例 (离散噪声调度)
    alpha = 1.0 - beta             # alpha_t = 1 - beta_t
    states = np.empty(L + 1)
    states[0] = x0
    x = x0
    for t in range(1, L + 1):
        noise = rng.standard_normal()                 # 标准高斯噪声 eps ~ N(0,1)
        x = np.sqrt(alpha) * x + np.sqrt(beta) * noise  # 一步加噪
        states[t] = x
    return states, beta


def reverse_chain(states, beta, x0_used):
    """层级VAE逆向: 从最末态逐层去噪, 按正向后验均值预测上一步。

    解码器 p(x_{t-1}|x_t) 的目标 = 正向后验均值 tilde_mu_t:
        tilde_mu_t = [sqrt(alpha)(1-ab_{t-1})/(1-ab_t)] x_t
                   + [sqrt(ab_{t-1}) (1-alpha) / (1-ab_t)] x0
    这里 x0 由调用方提供 —— Oracle 用真 x0 (精确反演);
    不完美解码器用 x0 的估计值 (往返有误差)。
    返回 shape = (L+1,) 的重建轨迹, 首元素为重建的 x0。
    """
    L = len(states) - 1
    alpha = 1.0 - beta
    bar_alpha = alpha ** np.arange(L + 1)     # bar_alpha_t = alpha^t
    recon = np.empty(L + 1)
    recon[L] = states[L]
    x = states[L]
    for t in range(L, 0, -1):
        # 后验均值系数 (正文 10.2 的 tilde_mu 公式)
        coeff_t = np.sqrt(alpha) * (1 - bar_alpha[t - 1]) / (1 - bar_alpha[t])
        coeff_0 = np.sqrt(bar_alpha[t - 1]) * (1 - alpha) / (1 - bar_alpha[t])
        x = coeff_t * x + coeff_0 * x0_used    # 去噪一步: 用该步可得的 x0 引导
        recon[t - 1] = x
    return recon


def reverse_chain_linear_est(states, beta):
    """不完美解码器: 每一步都用当前 x_t 按线性规则重新估计 x0 再去噪。

    真实解码器没有 x0, 每步只能从当前噪声态估计:
        x0_est(x_t) = x_t / sqrt(ab_t)   (忽略噪声的极大似然解)
    关键事实(本实验数值验证): 该估计对状态 x_t 是线性同质的,
    逆向映射逐层伸缩后乘积坍缩成单一因子, 使最终 recon[0] ≡ x_L/sqrt(ab_L),
    即"每步重估"与"只用末态估计一次"完全等价, 误差不会沿链累积。
    返回 shape = (L+1,) 的重建轨迹, 首元素为重建的 x0。
    """
    L = len(states) - 1
    alpha = 1.0 - beta
    bar_alpha = alpha ** np.arange(L + 1)
    recon = np.empty(L + 1)
    recon[L] = states[L]
    x = states[L]
    for t in range(L, 0, -1):
        x0_est = x / np.sqrt(bar_alpha[t])       # 从当前 x_t 重新估计 x0
        coeff_t = np.sqrt(alpha) * (1 - bar_alpha[t - 1]) / (1 - bar_alpha[t])
        coeff_0 = np.sqrt(bar_alpha[t - 1]) * (1 - alpha) / (1 - bar_alpha[t])
        x = coeff_t * x + coeff_0 * x0_est       # 去噪一步
        recon[t - 1] = x
    return recon


def reverse_chain_bias(states, beta, eps):
    """带乘性系统偏差的不完美解码器: 每步 x0_est(t) = (1-eps)·x_t/sqrt(ab_t)。

    eps>0 表示每一步都系统性地低估 x0 一个固定比例(方向一致)。
    由于每步偏差方向一致, 沿逆向链逐层相乘复合:
        recon[0] = B_L · (x_L/sqrt(ab_L)),
        B_L = Π_t [1 - eps·beta/(1-ab_t)] ~ L^{-eps}   (幂律压缩)
    → 系统偏差 (1-B_L)·x0 随链长 L 幂律增长, 区别于单次有偏(误差=-eps·x0, 与L无关)。
    返回 shape = (L+1,) 的重建轨迹。
    """
    L = len(states) - 1
    alpha = 1.0 - beta
    bar_alpha = alpha ** np.arange(L + 1)
    recon = np.empty(L + 1)
    recon[L] = states[L]
    x = states[L]
    for t in range(L, 0, -1):
        x0_est = (1.0 - eps) * x / np.sqrt(bar_alpha[t])   # 每步低估固定比例 eps
        coeff_t = np.sqrt(alpha) * (1 - bar_alpha[t - 1]) / (1 - bar_alpha[t])
        coeff_0 = np.sqrt(bar_alpha[t - 1]) * (1 - alpha) / (1 - bar_alpha[t])
        x = coeff_t * x + coeff_0 * x0_est
        recon[t - 1] = x
    return recon


def bias_compound_factor(L, eps, beta_total=1.0):
    """乘性偏差沿链复合的总压缩因子 B_L = Π_{t=1}^{L} [1 - eps·beta/(1-ab_t)]。

    这是"每步偏差沿链相乘"的解析表达(无需随机模拟);
    渐近 B_L ~ L^{-eps}/Gamma(1-eps), 即 log B_L vs log L 为直线, 斜率 ≈ -eps。
    """
    beta = beta_total / L
    alpha = 1.0 - beta
    bar_alpha = alpha ** np.arange(1, L + 1)
    return float(np.prod(1.0 - eps * beta / (1.0 - bar_alpha)))


# ============================================================
# 步骤1：层级VAE前向过程 —— 逐层加噪
# ============================================================
print("\n" + "="*60)
print("步骤1：层级VAE前向过程 (逐层加噪)")
print("="*60)
print("\n[核心思想]")
print("  层级 VAE 编码器就是逐层加噪:")
print("  q(x_t|x_{t-1}) = N(x_t; sqrt(1-beta_t) x_{t-1}, beta_t)")
print("  → 每步 = 乘 sqrt(alpha) 缩放 + 加 sqrt(beta)·噪声")
print("  干净信号在 L 步里被噪声逐步淹没。")

x0 = 1.5                       # 一个干净信号 (结论对高维逐分量成立)
beta_total = 1.0

rng = np.random.default_rng(0)
states_5, beta_5 = forward_chain(x0, L=5, beta_total=beta_total, rng=rng)
print("\nL=5 前向链各步:", np.round(states_5, 3))
print("末态值 =", round(states_5[-1], 3), "(单条链在 t=L 处的一次噪声态采样)")

print("\n同一 x0=1.5 出发, 不同 L 下前向链的末态(随机种子相同, L=5/50/500):")
print("  注意: beta_total=1.0 是固定总量, 只是被切分成 L 层,")
print("  sqrt(ab_L)=(1-1/L)^{L/2} 随 L→∞ 收敛到 sqrt(e^{-1})≈0.607, 并非 0,")
print("  即末态分布收敛到稳态 N(sqrt(e^{-1})x0, 1-e^{-1}), 而不是标准高斯 N(0,1)。")
for L in [5, 50, 500]:
    rng_l = np.random.default_rng(L)
    states, beta = forward_chain(x0, L=L, beta_total=beta_total, rng=rng_l)
    bar_alpha_L = (1.0 - beta_total / L) ** L   # 离散累积衰减
    print("  L=" + str(L).rjust(4) + ": 末态=" + str(round(states[-1], 3)).rjust(8) +
          ", sqrt(ab_L)=" + str(round(np.sqrt(bar_alpha_L), 4)) + " (信号残留系数)")
print("\n→ 每步乘 sqrt(alpha) 让信号指数衰减, 加 sqrt(beta) 让噪声累积。")
print("  但 L 增大只让离散链逼近其连续极限对应的稳态分布 (固定 beta_total 的饱和点),")
print("  要真正达到先验 p(x_T)=N(0,I) 需 beta_total→∞ —— 这属于 10.4-1 连续极限的范畴。")


# ============================================================
# 步骤2：理想解码器 (Oracle 已知 x0) —— 精确可逆
# ============================================================
print("\n" + "="*60)
print("步骤2：理想解码器 = 后验均值 (Oracle 已知 x0)")
print("="*60)
print("\n[核心思想]")
print("  若解码器知道真 x0, 逐层用后验均值 tilde_mu_t 去噪, 可精确反演编码链。")
print("  因此往返误差 (重建 x0 - 真 x0) 应精确为 0, 且与层级数 L 无关。")

recon_5 = reverse_chain(states_5, beta_5, x0_used=x0)
print("\nL=5 逆向重建轨迹:", np.round(recon_5, 3))
print("重建首步 (应为 x0=" + str(x0) + "):", round(recon_5[0], 6))

# 多样本统计: 理想解码在若干 L 下都能精确反演
print("\n多样本统计 (每种 L 做 5000 次随机往返):")
print("  循环前向加噪 → 理想解码(已知真 x0) → 记录平均绝对误差")
print("     L      平均往返绝对误差")
print("-" * 32)
for L in [5, 20, 100]:
    errs = []
    for trial in range(5000):
        rng_t = np.random.default_rng(L * 1000 + trial)
        x0_t = rng_t.uniform(-2.0, 2.0)
        states, beta = forward_chain(x0_t, L=L, beta_total=beta_total, rng=rng_t)
        recon = reverse_chain(states, beta, x0_used=x0_t)
        errs.append(abs(recon[0] - x0_t))
    print("  " + str(L).rjust(5) + "      " + str(round(float(np.mean(errs)), 18)).rjust(18))

print("\n→ 理想解码器的平均误差在所有 L 下都精确为 0 (t=1 处后验坍缩成 x0 点质量)。")
print("  这就说明: 编码/解码在理想情况下是严格可逆的 —— 难点从来不在层数, ")
print("  而在于解码器现实中不知道 x0 (下面步骤3)。")


# ============================================================
# 步骤3：不完美解码器 —— x0 需估计 → 往返误差 = x0 估计误差
# ============================================================
print("\n" + "="*60)
print("步骤3：不完美解码器 (只知 x_t, x0 需估计)")
print("="*60)
print("\n[核心思想]")
print("  真实解码器没有 x0, 只能从噪声态 x_t 估计 x0 (再代入后验均值 tilde_mu)。")
print("  本步骤数值验证一个关键事实: 往返误差的大小只取决于 x0 估计的质量,")
print("  而与'在哪一步估计'无关 —— 线性重估会伸缩坍缩(见对照二)。")
print("  → 要降低误差, 只能换更好的估计器, 这正是去噪网络(10.3)存在的意义。")

print("\n对照一: Oracle (已知真 x0) vs 朴素估计 (只用末态 x_L 估一次)")
print("     L      Oracle误差        朴素估计误差")
print("-" * 50)
L_list = [5, 20, 100]
oracle_errs = {}
guess_errs = {}
for L in L_list:
    orr, grr = [], []
    for trial in range(5000):
        rng_t = np.random.default_rng(L * 1000 + trial)
        x0_t = rng_t.uniform(-2.0, 2.0)
        states, beta = forward_chain(x0_t, L=L, beta_total=beta_total, rng=rng_t)
        ab_L = (1.0 - beta_total / L) ** L
        x0_guess = states[L] / np.sqrt(ab_L)     # 朴素 ML 估计 (忽略噪声)
        orr.append(abs(reverse_chain(states, beta, x0_used=x0_t)[0] - x0_t))
        grr.append(abs(reverse_chain(states, beta, x0_used=x0_guess)[0] - x0_t))
    oracle_errs[str(L)] = float(np.mean(orr))
    guess_errs[str(L)] = float(np.mean(grr))
    print("  " + str(L).rjust(5) + "      " + str(round(oracle_errs[str(L)], 6)).rjust(16) +
          "  " + str(round(guess_errs[str(L)], 6)).rjust(16))

print("\n→ 朴素估计的往返误差远大于 Oracle(精确为 0): 因为 x0_est 本身带标准差")
print("  sqrt((1-ab_L)/ab_L) ≈ sqrt(e-1) ≈ 1.31 的噪声。")
print("[附注] t=1 的后验坍缩: 后验均值在 t=1 处 tilde_mu_1 ≡ x0 (coeff_t=0, coeff_0=1),")
print("  因此任何 x0_used 在首步都被原样复现 → 往返误差恒等于 |x0_used - x0_true|,")
print("  即重建质量只取决于 x0 估计的准确度, 与链长、路径无关。")

print("\n对照二: '每步重新估计 x0' vs '只用末态估计一次' —— 重估能改善吗?")
print("     L      每步重估误差      末态一次估计误差       两者最大差")
print("-" * 62)
perstep_errs = {}
tele_maxdiff = {}
for L in L_list:
    pe, se, md = [], [], 0.0
    for trial in range(5000):
        rng_t = np.random.default_rng(L * 1000 + trial)
        x0_t = rng_t.uniform(-2.0, 2.0)
        states, beta = forward_chain(x0_t, L=L, beta_total=beta_total, rng=rng_t)
        ab_L = (1.0 - beta_total / L) ** L
        recon_step = reverse_chain_linear_est(states, beta)
        pe.append(abs(recon_step[0] - x0_t))
        se.append(abs(states[L] / np.sqrt(ab_L) - x0_t))
        md = max(md, abs(recon_step[0] - states[L] / np.sqrt(ab_L)))
    perstep_errs[str(L)] = float(np.mean(pe))
    tele_maxdiff[str(L)] = float(md)
    print("  " + str(L).rjust(5) + "      " + str(round(perstep_errs[str(L)], 6)).rjust(16) +
          "  " + str(round(float(np.mean(se)), 6)).rjust(18) + "   " + str(round(md, 3)).rjust(12))

print("\n→ 两种估计的往返误差完全相同 (最大差≈1e-15)! 即'每步重估'并不比'末态估一次'好。")
print("  原因: x0_est(x_t)=x_t/sqrt(ab_t) 对状态是线性同质的, 逆向映射逐层伸缩后")
print("  乘积坍缩成单一因子, 使最终 recon[0] ≡ x_L/sqrt(ab_L) (伸缩坍缩)。")
print("  → 误差并不会沿逆向链累积, 它只由 x0 估计的质量决定; 要降低往返误差,")
print("    只能换更好的估计器 —— 这正是 10.3 节去噪网络(学习 x0/ε 预测)要学的。")

print("\n对照三: 打破线性自洽 —— 乘性系统偏差会沿链复合 (真实网络误差的对应)")
eps_bias = 0.2          # 每步乘性偏差比例
x0_bias = 1.5           # 固定 x0, 只测系统偏差 (对噪声取均值)
n_bias = 10000          # 取均值样本数
L_bias = [5, 20, 100]   # 链长
bias_step = {}
bias_once = {}
bias_BL = {}
print("  每步 x0_est(t) = (1-eps)·x_t/sqrt(ab_t), 取 eps=" + str(eps_bias) + ", 固定 x0=" + str(x0_bias))
print("  对 " + str(n_bias) + " 次噪声取均值, 测系统偏差 E[recon[0]-x0]")
print("     L      每步偏差(随L增)   单次有偏(对照,不变)    压缩因子 B_L")
print("-" * 66)
for L in L_bias:
    s_step = 0.0
    s_once = 0.0
    for trial in range(n_bias):
        rng_t = np.random.default_rng(L * 1000 + trial)
        states, beta = forward_chain(x0_bias, L=L, beta_total=beta_total, rng=rng_t)
        ab_L = (1.0 - beta_total / L) ** L
        s_step += (reverse_chain_bias(states, beta, eps_bias)[0] - x0_bias)
        s_once += ((1.0 - eps_bias) * states[L] / np.sqrt(ab_L) - x0_bias)
    B_L = bias_compound_factor(L, eps_bias)
    bias_step[str(L)] = float(s_step / n_bias)
    bias_once[str(L)] = float(s_once / n_bias)
    bias_BL[str(L)] = B_L
    print("  " + str(L).rjust(5) + "      " + str(round(bias_step[str(L)], 4)).rjust(16) +
          "      " + str(round(bias_once[str(L)], 4)).rjust(16) +
          "      " + str(round(B_L, 4)).rjust(12))

print("\n→ 每步乘性偏差的系统误差随 L 增长, 而单次有偏恒 ≈ -eps·x0 =",
      round(-eps_bias * x0_bias, 3), "(与 L 无关)。")
print("  原因: 每步偏差方向一致, 沿逆向链逐层相乘 → 总压缩因子 B_L ~ L^{-eps},")
print("  log B_L vs log L 为直线, 斜率 ≈ -eps (幂律复合, 见下图 ii)。")
print("  → 这与 10.3 真实去噪网络一致: 网络在不同 t 的 x0 估计带方向一致的系统偏差,")
print("    且不满足'同一变量不同置信度观测'的线性自洽前提, 误差才会沿链累积。")

# 幂律斜率验证 (解析乘积, 无需随机模拟)
L_slope = [20, 100, 500, 2000]
logB_slope = [np.log(bias_compound_factor(L, eps_bias)) for L in L_slope]
slope_fit = float(np.polyfit(np.log(L_slope), logB_slope, 1)[0])
print("\n[验证] log B_L vs log L 拟合斜率 ≈", round(slope_fit, 4),
      " (理论 ≈ -eps =", str(-eps_bias) + ")")


# ============================================================
# 可视化
# ============================================================
print("\n" + "="*60)
print("生成可视化图表...")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) 前向链: 干净信号被逐步埋进噪声
ax = axes[0, 0]
rng_a = np.random.default_rng(0)
states_a, beta_a = forward_chain(x0, L=100, beta_total=beta_total, rng=rng_a)
ax.plot(range(len(states_a)), states_a, 'b-', linewidth=1.5, label='前向链 $x_t$')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(x0, color='red', linestyle='--', alpha=0.6, label='干净信号 $x_0=' + str(x0) + '$')
ax.annotate('$x_0$', xy=(0, x0), xytext=(8, x0 + 0.25), fontsize=10, color='red')
# 右上角标注真实去向: 固定 beta_total 时末态趋于稳态分布 (非 N(0,1))
ax.annotate('$\\to$ 稳态分布 (非 $N(0,1)$)', xy=(0.85, 0.92), xycoords='axes fraction',
            fontsize=11, color='blue', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='blue'))
ax.set_xlabel('编码步数 $t$', fontsize=12)
ax.set_ylabel('$x_t$', fontsize=12)
ax.set_title('(a) 前向过程: 逐层加噪埋没信号', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) 理想逆向: 从末态精确洗回干净信号
ax = axes[0, 1]
recon_a = reverse_chain(states_a, beta_a, x0_used=x0)
ax.plot(range(len(states_a)), states_a, 'b-', linewidth=1.2, alpha=0.5, label='前向 $x_t$')
ax.plot(range(len(recon_a)), recon_a, 'r-', linewidth=1.8, label='理想逆向重建 $\\tilde{x}_t$')
ax.axhline(x0, color='gray', linestyle='--', alpha=0.6, label='干净信号 $x_0$')
ax.annotate('重建精确回到 $x_0$', xy=(0, recon_a[0]), xytext=(0.42, 0.92),
            textcoords='axes fraction', fontsize=10, color='red',
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))
ax.set_xlabel('步数 (前向从左到右, 逆向从右到左)', fontsize=12)
ax.set_ylabel('$x$', fontsize=12)
ax.set_title('(b) 理想解码器: 逐层去噪精确洗回信号', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) 末态分布: 固定 beta_total 时收敛到非标准稳态分布 (不是 N(0,1))
ax = axes[1, 0]
n_pts = 500
x0_c = 1.5
for L, color in [(5, '#1f77b4'), (50, '#ff7f0e'), (500, '#2ca02c')]:
    rng_c = np.random.default_rng(L)
    finals = []
    for _ in range(n_pts):
        states_c, _ = forward_chain(x0_c, L=L, beta_total=beta_total, rng=rng_c)
        finals.append(states_c[-1])
    ax.hist(finals, bins=40, alpha=0.45, color=color, label='$L=' + str(L) + '$', density=True)
xs = np.linspace(-4, 4, 300)
# 参照1: 标准高斯 N(0,1) —— 末态并不会收敛到这里
ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), 'k--', linewidth=2, label='$N(0,1)$ (非稳态)')
# 参照2: 固定 beta_total 下的真实稳态 N(sqrt(e^{-beta_total})x0, 1-e^{-beta_total})
mu_ss = np.sqrt(np.exp(-beta_total)) * x0_c
var_ss = 1.0 - np.exp(-beta_total)
ax.plot(xs, np.exp(-(xs - mu_ss) ** 2 / (2 * var_ss)) / np.sqrt(2 * np.pi * var_ss),
        'r--', linewidth=2, label='稳态 $N(\\sqrt{e^{-1}}x_0,\\,1-e^{-1})$')
ax.set_xlabel('前向末态 $x_L$', fontsize=12)
ax.set_ylabel('概率密度', fontsize=12)
ax.set_title('(c) 末态分布: 固定 $\\beta_{total}$ 收敛到非标准稳态 (非 $N(0,1)$)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (d) Oracle vs 朴素估计: 往返误差
ax = axes[1, 1]
xpos = np.arange(len(L_list))
width = 0.35
orv = [oracle_errs[str(L)] for L in L_list]
guv = [guess_errs[str(L)] for L in L_list]
# Oracle 误差精确为 0 (线性尺度不可见), 用 max(真实计算值, 1e-15) 保留"取自计算"的形式
b1 = ax.bar(xpos - width / 2, [max(oracle_errs[str(L)], 1e-15) for L in L_list], width,
            color='steelblue', label='Oracle (已知 $x_0$)', edgecolor='steelblue', linewidth=0.5)
b2 = ax.bar(xpos + width / 2, guv, width, color='salmon', label='朴素估计 (只用 $x_L$)',
            edgecolor='salmon', linewidth=0.5)
# y 轴用线性尺度, 让两组都可见; 虚线标注 Oracle 柱的实际高度基准
ax.axhline(1e-15, color='steelblue', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylim(-0.05, max(guv) * 1.2)
ax.set_xticks(xpos)
ax.set_xticklabels(['$L=' + str(L) + '$' for L in L_list])
ax.set_ylabel('平均往返绝对误差', fontsize=12)
ax.set_title('(d) 解码器是否知 $x_0$ 决定往返成败', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.text(0.02, 0.98, 'Oracle 误差 = 0 (精确)', transform=ax.transAxes,
        fontsize=10, color='steelblue', fontweight='bold', va='top', ha='left')

fig.suptitle('实验10.4-2: 层级VAE编码/解码 <-> 加噪/去噪', fontsize=15, fontweight='bold', y=1.01)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤1_前后向过程演示.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print("图表已保存: 步骤1_前后向过程演示.png")


# 第二张图: 对照三 —— 乘性系统偏差的沿链复合
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5))

# (i) 系统偏差幅度 vs L: 每步乘性偏差(随L增长) vs 单次有偏(平坦)
ax = axes2[0]
Lp = [int(k) for k in bias_step.keys()]
stepv = [abs(bias_step[str(L)]) for L in Lp]
oncev = [abs(bias_once[str(L)]) for L in Lp]
ax.plot(Lp, stepv, 'o-', color='#d62728', linewidth=2, label='每步乘性偏差 (复合, 随 $L$ 增)')
ax.plot(Lp, oncev, 's--', color='steelblue', linewidth=2, label='单次有偏估计 (不变)')
ax.axhline(eps_bias * x0_bias, color='gray', linestyle=':', alpha=0.6,
           label='$\\epsilon\\,x_0$ 基准 ($L$ 无关)')
ax.set_xlabel('链长 $L$', fontsize=12)
ax.set_ylabel('系统偏差幅度 $|E[\\hat{x}_0]-x_0|$', fontsize=12)
ax.set_title('(i) 打破自洽: 每步偏差沿链复合增长', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (ii) log B_L vs log L: 幂律, 斜率 -eps
ax = axes2[1]
ax.plot(np.log(L_slope), logB_slope, 'o-', color='#2ca02c', linewidth=2, label='解析 $B_L$')
ax.annotate('斜率 $\\approx$ {:.3f} (理论 $-\\epsilon$ = {:.1f})'.format(slope_fit, -eps_bias),
            xy=(0.05, 0.92), xycoords='axes fraction', fontsize=11,
            color='#2ca02c', fontweight='bold')
ax.set_xlabel('$\\log L$', fontsize=12)
ax.set_ylabel('$\\log B_L$', fontsize=12)
ax.set_title('(ii) 压缩因子幂律: $B_L \\sim L^{-\\epsilon}$', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig2.suptitle('步骤3补充: 乘性系统偏差的沿链复合 ($\\epsilon={:.1f}$)'.format(eps_bias),
              fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
fig2_path = os.path.join(SAVE_DIR, '步骤3_系统偏差复合.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print("图表已保存: 步骤3_系统偏差复合.png")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "="*60)
print("实验10.4-2 总结")
print("="*60)
print("\n关键结论:")
print("\n1. 层级VAE编码器 = 逐层加噪 (步骤1)")
print("   - x_t = sqrt(alpha) x_{t-1} + sqrt(beta) eps")
print("   - 每步乘 sqrt(alpha) 缩放 + 加 sqrt(beta) 噪声")
print("   - beta_total 固定时, 末态收敛到稳态 N(sqrt(e^{-beta_total})x0, 1-e^{-beta_total}),")
print("     而非 N(0,1); 要真正趋于 N(0,1) 需 beta_total→∞ (10.4-1 连续极限范畴)")
print("\n2. 理想解码器 = 后验均值, 往返精确可逆 (步骤2)")
print("   - 若知道 x0, 逐层用 tilde_mu_t 去噪, 误差精确为 0 与 L 无关")
print("   - 难点不在层数, 而在解码器现实里不知道 x0")
print("\n3. 不完美解码器需估计 x0 → 往返误差 = x0 估计误差 (步骤3)")
print("   - t=1 处后验坍缩: 任何 x0_used 在首步都被原样复现")
print("   - 线性重估会伸缩坍缩: '每步重估' ≡ '末态估一次', 误差不沿链累积")
print("   - 打破自洽(乘性系统偏差): 总压缩因子 B_L ~ L^{-eps}, 系统误差随 L 幂律增长,")
print("     而单次有偏误差 = -eps·x0 与 L 无关 (见 步骤3_系统偏差复合.png)")
print("   - 真实去噪网络跨步估计非自洽、带方向一致偏差 → 误差才累积 = 10.3 网络的意义")
print("\n4. 与扩散模型的对应 (10.4 节)")
print("   - 前向加噪 = 编码器, 逆向去噪 = 解码器")
print("   - 本实验聚焦离散层级链的机理; 连续极限 (L→∞→SDE) 见实验10.4-1")

print("\n" + "="*60)
print("实验10.4-2 完成!")

# ===== 保存数值结果 =====
import json

def _to_native(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    return obj

results_summary = {
    'forward_chain_L5': {
        'states': states_5.tolist(),
        'final_value': float(states_5[-1]),
    },
    'ideal_reverse_L5': {
        'reconstructed': float(recon_5[0]),
        'true_x0': float(x0),
    },
    'roundtrip_mean_error_oracle_vs_L': {k: float(v) for k, v in oracle_errs.items()},
    'roundtrip_mean_error_guess_vs_L': {k: float(v) for k, v in guess_errs.items()},
    'roundtrip_mean_error_perstep_reest_vs_L': {k: float(v) for k, v in perstep_errs.items()},
    'telescope_maxdiff_perstep_vs_singleshot_vs_L': {k: float(v) for k, v in tele_maxdiff.items()},
    'bias_systematic_error_stepwise_vs_L': {k: float(v) for k, v in bias_step.items()},
    'bias_systematic_error_singleshot_vs_L': {k: float(v) for k, v in bias_once.items()},
    'bias_compound_factor_BL_vs_L': {k: float(v) for k, v in bias_BL.items()},
    'bias_loglog_slope_fit': float(slope_fit),
    'steady_state_sqrt_bar_alpha_L_inf': float(np.sqrt(np.exp(-beta_total))),
    'steady_state_variance_1_minus_e_beta_total': float(1.0 - np.exp(-beta_total)),
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")