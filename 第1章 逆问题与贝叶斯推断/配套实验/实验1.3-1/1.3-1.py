import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Jupyter 中渲染 LaTeX 公式
try:
    from IPython.display import display, Markdown
    _in_jupyter = True
except ImportError:
    _in_jupyter = False

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验1.3-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验1.3-1')
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)
# ========================================================

np.random.seed(42)

print("="*60)
print("实验1.3-1：病态性 - 误差的放大器")
print("="*60)
print("本实验演示条件数如何影响误差放大")
print("对应 1.3 节 '病态性：误差的放大器'")

# ---- 1. 构造不同条件数的对角矩阵 ----
n = 100
x_true = np.random.randn(n)

# 对角矩阵：最大对角元=1，最小对角元=1/κ
# 
# 关键说明：
# 1. 这里使用的是对角矩阵 D = diag(σ₁, ..., σₙ)
# 2. 对角矩阵的SVD分解为 D = I · D · I^T，因此对角元就是奇异值
# 3. 条件数 κ(D) = σ_max / σ_min = 1 / σ_min（因为 σ_max = 1）
# 4. 对于一般矩阵，需要通过 np.linalg.svd() 计算奇异值（见实验1.7）
# 
# 这种简化让我们能专注于理解"奇异值衰减→误差放大"的核心机制
# 条件数从 1 到 10^12
kappas = [1, 10, 1e2, 1e3, 1e4, 1e6, 1e8, 1e10, 1e12]
noise_level = 1e-6  # 数据中的噪声水平

# 预计算各条件数对应的奇异值（即对角元）
# 使用对数间隔，模拟真实不适定问题的奇异值衰减特性
singular_values_cache = {
    kappa: np.logspace(0, -np.log10(kappa), n) 
    for kappa in kappas
}

print(f"\n实验设置:")
print(f"- 向量维度: {n}")
print(f"- 噪声水平: {noise_level}")
print(f"- 条件数范围: {kappas[0]} 到 {kappas[-1]}")
print("-"*60)

rel_errors = []

for kappa in kappas:
    singular_values = singular_values_cache[kappa]

    # 正问题：y = Ax
    y = singular_values * x_true

    # 加入噪声
    y_noisy = y + noise_level * np.random.randn(n)

    # 逆问题：x = A^{-1} y_noisy（朴素重建）
    x_recon = y_noisy / singular_values

    # 朴素重建相对误差
    rel_err = np.linalg.norm(x_recon - x_true) / np.linalg.norm(x_true)
    rel_errors.append(rel_err)

# ---- 2. 可视化 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 重建误差 vs 条件数
axes[0].loglog(kappas, rel_errors, 'o-', linewidth=2, markersize=8, label='朴素逆重建')

# 理论上界: ‖δx‖/‖x‖ ≤ κ · ‖δy‖/‖y‖
# 注意：噪声向量 δy = noise_level * randn(n)，其期望范数为 noise_level * √n
theory_bound = []
for kappa in kappas:
    sv = singular_values_cache[kappa]
    y_norm = np.linalg.norm(sv * x_true)
    theory_bound.append(kappa * (noise_level * np.sqrt(n)) / y_norm)
axes[0].loglog(kappas, theory_bound, '--', alpha=0.7, label='理论上界 κ·(‖δy‖/‖y‖)')

axes[0].set_xlabel('条件数 κ(A)')
axes[0].set_ylabel('重建相对误差 ‖x-x̂‖/‖x‖')
axes[0].set_title('条件数与误差放大\nκ 越大→噪声放大越严重')
axes[0].legend()
axes[0].grid(True, which='both')

# 奇异值衰减示意
# 注：这里展示的是对角矩阵的奇异值（即对角元）的分布
# 
# 关键直觉：奇异值 σ_i 越小 → 该方向的放大因子 1/σ_i 越大
# 类比"高通滤波器"：小奇异值方向对应高频分量，求逆时被剧烈放大
# 正向算子 A 像"低通滤波器"（压制高频），A^{-1} 则像"高通滤波器"（放大高频噪声）
for kappa in [1, 1e3, 1e6, 1e10]:
    sv = singular_values_cache[kappa]
    axes[1].semilogy(np.arange(1, n + 1), sv, label=f'κ={kappa:.0e}')
axes[1].set_xlabel('奇异值索引 i')
axes[1].set_ylabel('奇异值 σ_i')
axes[1].set_title('不同条件数的奇异值分布\n小 σ_i → 1/σ_i 爆炸 → 噪声放大\n（类比：A^{-1} 是"高通滤波器"）')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_3_1_病态性.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("数值结果表：条件数 vs 误差放大")
print("="*60)
print(f"{'κ(A)':>12s}  {'朴素误差':>12s}  {'有效数字丢失':>12s}")
print("-" * 40)
for kappa, err in zip(kappas, rel_errors):
    # 估算有效数字丢失：根据1.3节，log10(κ) ≈ 丢失的有效数字位数
    # 当 κ=10^k 时，理论上可能丢失 k 位有效数字
    digits_lost_theory = int(np.log10(kappa)) if kappa > 1 else 0
    # 实际丢失位数基于误差大小（误差>1表示完全失真）
    digits_lost_actual = max(0, int(np.log10(err)) + 1) if err > 1e-16 else 0
    # 显示理论估算值（与1.3节对应）
    print(f"{kappa:12.0e}  {err:12.4e}  ~{digits_lost_theory:>8d} 位")

print("\n" + "="*60)
print("实验结论：")
print("="*60)
print("1. 条件数 κ(A) 越大，误差放大越严重")
print("2. 当 κ(A) 达到 10^10 以上时，重建误差远大于原始信号")
print("3. 理论上界 κ·(‖δy‖/‖y‖) 为误差提供了保守估计")
print("4. 这就是为什么逆问题被称为'病态'的原因")
print("5. 需要引入额外信息（如正则化）来改善重建效果")
print("\n数值解读（参考1.3节）：")
print("- κ ≈ 10^3：轻度病态，约丢失3位有效数字")
print("- κ ≈ 10^6：中度病态，约丢失6位有效数字")
print("- κ ≈ 10^10：严重病态，几乎无法恢复有意义的解")
print("\n物理直觉：")
print("- 正向算子 A 像'低通滤波器'：平滑、压缩、积分，压制高频分量")
print("- 逆向算子 A^{-1} 像'高通滤波器'：放大被压制的高频噪声")
print("- 小奇异值方向 = 高频方向 → 1/σ_i 巨大 → 噪声灾难性放大")
print("="*60)

if _in_jupyter:
    display(Markdown("""
**理论回顾**：
- **条件数**：$\\kappa(A) = \\|A\\| \\cdot \\|A^{-1}\\|$，度量误差放大效应
- **误差放大不等式**：$\\frac{\\|x - x^\\delta\\|}{\\|x\\|} \\leq \\kappa(A) \\frac{\\|y - y^\\delta\\|}{\\|y\\|}$
- **病态问题**：条件数很大（$\\kappa > 10^6$），微小噪声导致巨大重建误差

**物理直觉**：
- 正向算子 $A$ 像"低通滤波器"：平滑、压缩、积分，压制高频分量
- 逆向算子 $A^{-1}$ 像"高通滤波器"：放大被压制的高频噪声
- 小奇异值方向 = 高频方向 → $1/\\sigma_i$ 巨大 → 噪声灾难性放大

**数值示例**：
- $\\kappa \\approx 10^3$：轻度病态，约丢失3位有效数字
- $\\kappa \\approx 10^6$：中度病态，约丢失6位有效数字
- $\\kappa \\approx 10^{10}$：严重病态，几乎无法恢复有意义的解
"""))
else:
    print("\n理论回顾：")
    print("- 条件数：kappa(A) = ||A|| * ||A^{-1}||，度量误差放大效应")
    print("- 误差放大不等式：||x-x^delta||/||x|| <= kappa(A) * ||y-y^delta||/||y||")
    print("- 病态问题：条件数很大（kappa > 10^6），微小噪声导致巨大重建误差")
    print("\n物理直觉：")
    print("- 正向算子 A 像'低通滤波器'：平滑、压缩、积分，压制高频分量")
    print("- 逆向算子 A^{-1} 像'高通滤波器'：放大被压制的高频噪声")
    print("- 小奇异值方向 = 高频方向 → 1/sigma_i 巨大 → 噪声灾难性放大")
    print("\n数值示例：")
    print("- kappa ≈ 10^3：轻度病态，约丢失3位有效数字")
    print("- kappa ≈ 10^6：中度病态，约丢失6位有效数字")
    print("- kappa ≈ 10^10：严重病态，几乎无法恢复有意义的解")