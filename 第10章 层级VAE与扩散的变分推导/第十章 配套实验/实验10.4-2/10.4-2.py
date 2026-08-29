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

np.random.seed(42)

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
print("末态值 =", round(states_5[-1], 3), "(已接近 N(0,1) 附近采样)")

print("\n同一 x0=1.5 出发, 不同 L 下前向链的末态(随机种子相同, L=5/50/500):")
print("  注意: L 越大每步加噪越细, 但单条链的末态仍是 N(0,1) 上的一个采样,")
print("  严格'L 越大末态越接近标准高斯'是大样本分布意义, 见子图 (c)。")
for L in [5, 50, 500]:
    rng_l = np.random.default_rng(L)
    states, beta = forward_chain(x0, L=L, beta_total=beta_total, rng=rng_l)
    bar_alpha_L = (1.0 - beta_total / L) ** L   # 离散累积衰减
    print("  L=" + str(L).rjust(4) + ": 末态=" + str(round(states[-1], 3)).rjust(8) +
          ", sqrt(ab_L)=" + str(round(np.sqrt(bar_alpha_L), 4)) + " (信号残留系数)")
print("\n→ 每步乘 sqrt(alpha) 让信号指数衰减, 加 sqrt(beta) 让噪声累积。")
print("  当 L 足够大, 末态近似标准高斯 —— 这正是先验 p(x_T) = N(0,I) 的来源。")


# ============================================================
# 步骤2：理想解码器 (Oracle 已知 x0) —— 精确可逆
# ============================================================
print("\n" + "="*60)
print("步骤2：理想解码器 = 后验均值 (Oracle 已知 x0)")
print("="*60)
print("\n[核心思想]")
print("  若解码器知道真 x0, 逐层用后验均值 tilde_mu_t 去噪, 可精确反演编码链。")
print("  因此往返误差 (重建 x0 - 真 x0) 应≈0, 且与层级数 L 无关。")

rng5 = np.random.default_rng(0)
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

print("\n→ 理想解码器的平均误差在所有 L 下都接近机器精度 (≈1e-16)。")
print("  这就说明: 编码/解码在理想情况下是严格可逆的 —— 难点从来不在层数, ")
print("  而在于解码器现实中不知道 x0 (下面步骤3)。")


# ============================================================
# 步骤3：不完美解码器 —— x0 需估计 → 往返有误差
# ============================================================
print("\n" + "="*60)
print("步骤3：不完美解码器 (只知 x_t, x0 需估计)")
print("="*60)
print("\n[核心思想]")
print("  真实解码器没有 x0, 只能从噪声态 x_t 估计 x0 (再代入 tilde_mu)。")
print("  用简单的 x0 估计: x0_est ≈ x_L / sqrt(ab_L) (忽略噪声的极大似然解)。")
print("  这个估计带噪声 → 往返重建 x0 会有误差, 且误差随估计质量变差而增大。")
print("  → 这就是去噪网络(10.3)存在的意义: 学习一个更好的 x0/噪声估计。")

print("\n对照: Oracle (已知真 x0) vs 朴素估计 (只用 x_L), 多 L 平均往返误差")
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

print("\n→ 朴素估计的往返误差远大于 Oracle(≈0): 因为 x0_est 本身带标准差")
print("  sqrt((1-ab_L)/ab_L) ≈ sqrt(e-1) ≈ 1.31 的噪声, 会沿逆向链传导。")
print("  要压低这个误差, 必须用更好的去噪器(如 10.3 的 ε/x0 预测网络)去逼近理想的 tilde_mu。")


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
# 把"→N(0,1)"放到右上角空白区, 不与前向链在末态位置重叠
ax.annotate('$\\to N(0,1)$', xy=(0.85, 0.92), xycoords='axes fraction',
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

# (c) 前向末态分布随 L 变化 (越接近标准高斯)
ax = axes[1, 0]
n_pts = 500
for L, color in [(5, '#1f77b4'), (50, '#ff7f0e'), (500, '#2ca02c')]:
    rng_c = np.random.default_rng(L)
    finals = []
    for _ in range(n_pts):
        states_c, _ = forward_chain(rng_c.uniform(-2, 2), L=L, beta_total=beta_total, rng=rng_c)
        finals.append(states_c[-1])
    ax.hist(finals, bins=40, alpha=0.45, color=color, label='$L=' + str(L) + '$', density=True)
xs = np.linspace(-4, 4, 300)
ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), 'k--', linewidth=2, label='$N(0,1)$')
ax.set_xlabel('前向末态 $x_L$', fontsize=12)
ax.set_ylabel('概率密度', fontsize=12)
ax.set_title('(c) 末态分布: 加噪层越多越接近先验 $N(0,1)$', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (d) Oracle vs 朴素估计: 往返误差
ax = axes[1, 1]
xpos = np.arange(len(L_list))
width = 0.35
orv = [oracle_errs[str(L)] for L in L_list]
guv = [guess_errs[str(L)] for L in L_list]
# Oracle 误差≈0 (对数尺度不可见), 改用线性尺度 + 手动标注
b1 = ax.bar(xpos - width / 2, [1e-15] * len(L_list), width, color='steelblue',
            label='Oracle (已知 $x_0$)', edgecolor='steelblue', linewidth=0.5)
b2 = ax.bar(xpos + width / 2, guv, width, color='salmon', label='朴素估计 (只用 $x_L$)',
            edgecolor='salmon', linewidth=0.5)
# y 轴用线性尺度, 让两组都可见; 添加虚线标注 Oracle 的真实数量级
ax.axhline(1e-16, color='steelblue', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylim(-0.05, max(guv) * 1.2)
ax.set_xticks(xpos)
ax.set_xticklabels(['$L=' + str(L) + '$' for L in L_list])
ax.set_ylabel('平均往返绝对误差', fontsize=12)
ax.set_title('(d) 解码器是否知 $x_0$ 决定往返成败', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.text(0.02, 0.98, 'Oracle 误差≈$10^{-16}$', transform=ax.transAxes,
        fontsize=10, color='steelblue', fontweight='bold', va='top', ha='left')

fig.suptitle('实验10.4-2: 层级VAE编码/解码 ↔ 加噪/去噪', fontsize=15, fontweight='bold', y=1.01)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤1_前后向过程演示.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print("图表已保存: 步骤1_前后向过程演示.png")


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
print("   - L 越大末态越接近标准高斯先验 N(0,I)")
print("\n2. 理想解码器 = 后验均值, 往返精确可逆 (步骤2)")
print("   - 若知道 x0, 逐层用 tilde_mu_t 去噪, 误差≈0 与 L 无关")
print("   - 难点不在层数, 而在解码器现实里不知道 x0")
print("\n3. 不完美解码器需估计 x0 → 往返有误差 (步骤3)")
print("   - 朴素估计 x0≈x_L/sqrt(ab_L) 带噪声, 误差沿逆向链传导")
print("   - 这正是去噪网络(10.3)存在的意义: 逼近理想后验均值")
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
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")