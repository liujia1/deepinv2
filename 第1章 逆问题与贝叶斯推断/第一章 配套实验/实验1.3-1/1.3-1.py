import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    pass
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
# 条件数从 1 到 10^12（统一使用浮点数，避免字典键类型不一致）
kappas = [10.0**k for k in [0, 1, 2, 3, 4, 6, 7, 8, 10, 12]]
# 数据中的噪声水平。取 1e-8，正好对应 1.3 节开头那个悬念：
# "数据 y 只有亿分之一(10^-8)相对误差、条件数 κ=10^10 时最坏误差多大？"
# 跑一遍就能亲眼看到小噪声被放大成一团乱麻。
noise_level = 1e-8

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
theory_bound = []

for kappa in kappas:
    singular_values = singular_values_cache[kappa]

    # 正问题：y = Ax
    y = singular_values * x_true
    y_norm = np.linalg.norm(y)

    # 加入噪声
    y_noisy = y + noise_level * np.random.randn(n)

    # 逆问题：x = A^{-1} y_noisy（朴素重建）
    x_recon = y_noisy / singular_values

    # 朴素重建相对误差
    rel_err = np.linalg.norm(x_recon - x_true) / np.linalg.norm(x_true)
    rel_errors.append(rel_err)

    # 理论上界: ‖δx‖/‖x‖ ≤ κ · ‖δy‖/‖y‖
    # 直接用实际噪声范数 ‖δy‖，而非期望值 noise_level·√n
    delta_y_norm = np.linalg.norm(y_noisy - y)
    theory_bound.append(kappa * delta_y_norm / y_norm)

# ---- 2. 可视化 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 重建误差 vs 条件数
axes[0].loglog(kappas, rel_errors, 'o-', linewidth=2, markersize=8, label='朴素逆重建')

# 理论上界（显式用橙色，避免依赖默认色序；对应说明中的"橙色虚线"）
axes[0].loglog(kappas, theory_bound, '--', color='C1', alpha=0.7, label='理论上界 κ·(‖δy‖/‖y‖)')

# 添加垂直参考线：κ = 1/noise_level，这是误差开始超过信号的理论拐点
kappa_critical = 1.0 / noise_level
# 开头悬念对应的条件数 κ=10^10，正好超过临界点 1e8，是"完全失真"的典型
kappa_hook = 1e10
axes[0].axvline(kappa_critical, color='dimgray', linestyle=':', linewidth=2.0, alpha=0.8)
# 在参考线旁添加注释：同时说明公式和含义
axes[0].text(kappa_critical*1.3, 2e-4, 
             f'κ = 1/ε = {kappa_critical:.0e}\n(误差≈1，开始失真)', 
             fontsize=8, color='darkred',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.5))

# 高亮开头悬念那一行：κ=10^10（数据仅 10^-8 误差，最坏放大到一团乱麻）
axes[0].axvline(kappa_hook, color='purple', linestyle='-.', linewidth=1.5, alpha=0.7)
axes[0].text(kappa_hook*1.15, 1e-2,
             f'开头悬念：κ=10¹⁰\n(10⁻⁸ 误差 → 完全失真)',
             fontsize=8, color='purple',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lavender', alpha=0.6))

axes[0].set_xlabel('条件数 κ(A)', fontsize=11)
axes[0].set_ylabel(r'重建相对误差 $\|x-\hat{x}\|/\|x\|$', fontsize=11)
axes[0].set_title('条件数与误差放大', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, which='both', alpha=0.3)

# 奇异值衰减示意
# 注：这里展示的是对角矩阵的奇异值（即对角元）的分布
# 
# 关键直觉：奇异值 σ_i 越小 → 该方向的放大因子 1/σ_i 越大
# 类比"高通滤波器"：小奇异值方向对应高频分量，求逆时被剧烈放大
# 正向算子 A 像"低通滤波器"（压制高频），A^{-1} 则像"高通滤波器"（放大高频噪声）
for kappa in [1e0, 1e3, 1e6, 1e10]:
    sv = singular_values_cache[kappa]
    axes[1].semilogy(np.arange(1, n + 1), sv, label=f'κ={kappa:.0e}')
axes[1].set_xlabel(r'奇异值索引 $i$', fontsize=11)
axes[1].set_ylabel(r'奇异值 $\sigma_i$', fontsize=11)
# 将副标题整合到主标题中，使用换行符分隔
axes[1].set_title(r'不同条件数的奇异值分布\n（小 $\sigma_i$ → $1/\sigma_i$ 爆炸 → 噪声放大）',
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_3_1_病态性.png'), dpi=150, bbox_inches='tight')
plt.show()

# ===== 新增：同一信号、不同条件数下的重建对比（对应开头悬念 κ=10^10）=====
# 固定一组可复现随机种子，让对比图稳定可重绘
np.random.seed(123)
# 用一条含丰富高频细节的波形作待重建信号：高频越密，κ=10^6 的"发钝"越明显
t_demo = np.linspace(0, 1, n)
x_demo = (np.sin(2 * np.pi * 6 * t_demo)
          + 0.5 * np.cos(2 * np.pi * 16 * t_demo)
          + 0.35 * np.sin(2 * np.pi * 28 * t_demo))

# 关键：所有 κ 共用同一份观测噪声，这样唯一的变量就是条件数 → 对比最纯净
e_shared = noise_level * np.random.randn(n)

def naive_recon(kappa):
    sv = singular_values_cache[kappa]
    y_noisy = sv * x_demo + e_shared          # 同一信号 + 同一份噪声
    xr = y_noisy / sv                          # 朴素逆重建
    rel = np.linalg.norm(xr - x_demo) / np.linalg.norm(x_demo)
    return xr, rel

# 用 κ=10^7（蓝，发钝）与 κ=10^10（红，一团乱麻）作对照，
# 直接呼应开头悬念"数据仅 10^-8 误差、κ=10^10 炸成乱麻"。
demo_kappas = [1e7, 1e10]
recons = {kk: naive_recon(kk) for kk in demo_kappas}

# 方案：全部画在同一张坐标系里，用不同颜色叠加，直观对比"恢复 vs 乱麻"
fig2, ax2 = plt.subplots(1, 1, figsize=(11, 5))
# 干净信号（黑色粗线，作为基准）
ax2.plot(t_demo, x_demo, color='black', linewidth=2.5, label='干净信号 x（基准）')
# 各条件数重建：κ 越小越接近"能恢复"，越大越"一团乱麻"，颜色由蓝→红
colors = {1e7: 'steelblue', 1e10: 'crimson'}
tags = {1e7: '轻微发钝', 1e10: '一团乱麻'}
for kk in demo_kappas:
    xr, rel = recons[kk]
    ax2.plot(t_demo, xr, color=colors[kk], alpha=0.85,
             label=f'κ={kk:.0e} 重建（{tags[kk]}，误差 {rel:.1e}）')
ax2.set_xlabel('索引 / 位置', fontsize=11)
ax2.set_ylabel('信号幅度', fontsize=11)
# 夹紧 y 轴到信号量级：κ=10^10 的乱麻线会"撞"到上下边界，直观体现彻底失真
ax2.set_ylim(-1.6, 1.6)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9, loc='lower left')
fig2.suptitle('实验1.3-1 附：同一信号、不同条件数下的朴素逆重建（叠图对比）——有的能恢复，有的成一团乱麻',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_3_1_信号重建乱麻.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n[附图] 不同条件数下朴素逆重建相对误差（ε=10^-8，共用同一份噪声）：")
for kk in demo_kappas:
    print(f"  κ={kk:.0e}  相对误差 = {recons[kk][1]:.4e}")

print("\n" + "="*60)
print("数值结果表：条件数 vs 误差放大")
print("="*60)
print(f"{'κ(A)':>12s}  {'朴素误差':>12s}  {'理论丢失':>10s}  {'实际丢失':>10s}")
print("-" * 52)

# 基线精度：噪声水平决定的初始精度上限
baseline_digits = round(-np.log10(noise_level))  # = 6，表示噪声本身的精度

for kappa, err in zip(kappas, rel_errors):
    # 理论丢失：log10(κ)，衡量条件数本身的量级（上界估计）
    digits_lost_theory = round(np.log10(kappa)) if kappa > 1.0 else 0
    
    # 实际丢失：log10(err/noise_level)，衡量因条件数导致的额外精度损失
    # = 基线精度 - 重建后精度
    if err >= 1.0:
        digits_lost_str = f'>{baseline_digits}位（完全失真）'
    elif err > 1e-16:
        digits_remaining = round(-np.log10(err))  # 重建后还剩几位精度
        digits_lost_actual = max(0, baseline_digits - digits_remaining)
        digits_lost_str = f'~{digits_lost_actual:>4d}位'
    else:
        digits_lost_str = '  0位'
    
    print(f'{kappa:12.0e}  {err:12.4e}  ~{digits_lost_theory:>6d}位  {digits_lost_str:>10s}')

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

# ===== 保存数值结果 =====
import json
results_summary = {
    'vector_dimension': n,
    'noise_level': float(noise_level),
    'condition_numbers': [float(k) for k in kappas],
    'relative_errors': [float(round(e, 6)) for e in rel_errors],
    'theory_bounds': [float(round(b, 6)) for b in theory_bound],
}

def _to_native(obj):
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

results_summary = {k: _to_native(v) for k, v in results_summary.items()}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")