"""
实验3.4-1 近端算子：从收缩到软阈值
对应章节：3.4 近端方法——不可微先验的求解策略
知识点：近端算子定义与性质；四种经典近端算子（收缩、软阈值、硬阈值、投影）；
        软阈值 vs 硬阈值的几何对比；近端算子与次梯度的关系

素材来源：winter_school/BolognaWinterSchool2023-main/Matlab/proximal.m (Python翻译)
          winter_school/BolognaWinterSchool2023-main/Matlab/Smu.m (Python翻译)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验3.4-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
else:
    _chinese_path = '.chinese'
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)

# ══════════════════════════════════════════════════════════
# 1. 四种经典近端算子定义
# ══════════════════════════════════════════════════════════

def soft_threshold(v, lam):
    """软阈值 S_lam(v) = sign(v) * max(|v| - lam, 0)
    对应 g(x) = ||x||_1 的近端算子
    翻译自 Smu.m (Daubechies, Defrise and De Mol 2004)
    """
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)


def hard_threshold(v, lam):
    """近端算子 prox_{lam*||x||_0}(v)，阈值 tau = sqrt(2*lam)
    |v| < tau 的分量归零，|v| >= tau 的分量保留原值
    """
    tau = np.sqrt(2 * lam)
    return v * (np.abs(v) >= tau).astype(float)


def shrinkage(v, lam):
    """收缩算子 prox_{lam*g}(v) = v / (1 + lam)
    正则项 g(x) = 1/2 * ||x||_2^2，因子 1/2 使得梯度恰好为 x，
    从而 prox_{lam*g}(v) = v/(1+lam)（分母为 1+lam 而非 1+2lam）
    即 Tikhonov 去噪器
    """
    return v / (1 + lam)


def projection_nonneg(v):
    """投影到非负约束集 C = {x : x >= 0}
    对应 g(x) = iota_C(x) 的近端算子
    """
    return np.maximum(v, 0)


# ══════════════════════════════════════════════════════════
# 2. 近端算子可视化
# ══════════════════════════════════════════════════════════

print("=" * 60)
print("实验3.4-1 近端算子：从收缩到软阈值")
print("=" * 60)

lam = 1.0
v = np.linspace(-3, 3, 1000)

prox_l2 = shrinkage(v, lam)
prox_l1 = soft_threshold(v, lam)
prox_l0 = hard_threshold(v, lam)
prox_proj = projection_nonneg(v)

print(f"\n[参数设定]")
print(f"  正则化参数: λ = {lam}")
print(f"  输入范围: v ∈ [-3, 3]")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- 左图：四种经典近端算子对比 ----
axes[0].plot(v, v, 'k--', alpha=0.3, label='恒等 $v$')
axes[0].plot(v, prox_l2, 'g-', linewidth=2.5,
             label=r'$\ell_2^2$ 收缩: $v/(1+\lambda)$')
axes[0].plot(v, prox_l1, 'b-', linewidth=2.5,
             label=r'$\ell_1$ 软阈值: $\mathcal{S}_\lambda(v)$')
axes[0].plot(v, prox_l0, 'r-', linewidth=2.5,
             label=r'$\ell_0$ 硬阈值: $\mathcal{H}_{\sqrt{2\lambda}}(v)$')
axes[0].plot(v, prox_proj, 'm--', linewidth=2.0,
             label=r'投影: $\max(v, 0)$')

axes[0].axhline(y=0, color='gray', linewidth=0.5)
axes[0].axvline(x=0, color='gray', linewidth=0.5)
axes[0].set_xlabel(r'输入 $v$', fontsize=12)
axes[0].set_ylabel(r'$\mathrm{prox}_{\lambda g}(v)$', fontsize=12)
axes[0].set_title(r'四种经典近端算子 ($\lambda=1$)', fontsize=13)
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-3, 3)

# ---- 右图：软阈值 vs 硬阈值细节对比 ----
axes[1].plot(v, v, 'k--', alpha=0.3, label='恒等 $v$')
axes[1].plot(v, prox_l1, 'b-', linewidth=2.5,
             label=r'软阈值: 归零 + 收缩')
axes[1].plot(v, prox_l0, 'r-', linewidth=2.5,
             label=r'硬阈值: 归零 + 保留原值')

# 标注关键差异区域
# 软阈值在 |v| > lam 时向零收缩 lam，硬阈值保留原值
mask_shrink = np.abs(v) > lam
if np.any(mask_shrink):
    v_shrink = v[mask_shrink]
    mid_idx = len(v_shrink) * 3 // 4
    axes[1].annotate(
        r'shrinking bias: $|v| - \lambda$ vs $v$',
        xy=(v_shrink[mid_idx], prox_l1[mask_shrink][mid_idx]),
        xytext=(v_shrink[mid_idx] + 0.5, prox_l1[mask_shrink][mid_idx] - 0.8),
        fontsize=9, color='darkblue',
        arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.2))

# 标注归零区域
axes[1].fill_between(v, -3, 3, where=(np.abs(v) <= lam),
                     alpha=0.08, color='blue', label=r'归零区 $|v| \leq \lambda$')
axes[1].fill_between(v, -3, 3, where=(np.abs(v) < np.sqrt(2 * lam)),
                     alpha=0.08, color='red', label=r'归零区 $|v| < \sqrt{2\lambda}$')

axes[1].axhline(y=0, color='gray', linewidth=0.5)
axes[1].axvline(x=0, color='gray', linewidth=0.5)
axes[1].set_xlabel(r'输入 $v$', fontsize=12)
axes[1].set_ylabel(r'$\mathrm{prox}_{\lambda g}(v)$', fontsize=12)
axes[1].set_title(r'软阈值 vs 硬阈值：shrinking bias 对比', fontsize=13)
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(-3, 3)
axes[1].set_ylim(-3, 3)

plt.suptitle(r'实验3.4-1: 近端算子——从收缩到软阈值', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, '步骤1_近端算子对比.png'), dpi=150)
plt.close()

print(f"\n[近端算子性质验证]")
print(f"  ℓ₂² 收缩: prox(v) = v/(1+λ) = v/{1+lam:.1f}")
print(f"    - 每个分量均匀压缩向零，不产生稀疏性")
print(f"  ℓ₁ 软阈值: S_λ(v) = sign(v) * max(|v|-λ, 0)")
print(f"    - |v| ≤ λ 的分量归零（稀疏化）")
print(f"    - |v| > λ 的分量向零收缩 λ（shrinking bias）")
print(f"  ℓ₀ 硬阈值: prox_{{λ||x||₀}}(v)，阈值 τ = sqrt(2λ)")
print(f"    - |v| < τ = sqrt(2λ) = {np.sqrt(2*lam):.2f} 的分量归零")
print(f"    - |v| ≥ τ 的分量保留原值（无偏差）")
print(f"  投影: max(v, 0)")
print(f"    - 负分量归零，正分量保留")

print(f"\n[软阈值 vs 硬阈值]")
print(f"  软阈值归零区: |v| ≤ λ = {lam:.1f}")
print(f"  硬阈值归零区: |v| < τ = sqrt(2λ) = {np.sqrt(2*lam):.2f}")
print(f"  关键差异: 软阈值对非零值收缩 λ（有偏），硬阈值保留原值（无偏但非凸）")

# ══════════════════════════════════════════════════════════
# 3. 非扩张性（non-expansiveness）数值验证
# ══════════════════════════════════════════════════════════
# 近端算子的核心性质: ||prox(v1) - prox(v2)|| <= ||v1 - v2||
# 这是 ISTA/FISTA 收敛性的理论基础

print(f"\n[非扩张性验证]  ||prox(v1) - prox(v2)|| ≤ ||v1 - v2||")
print(f"  这是 ISTA/FISTA 收敛性的关键性质: 近端算子是 1-Lipschitz 的")

n_test = 10000
v1 = np.random.randn(n_test)
v2 = np.random.randn(n_test)
dist_v = np.linalg.norm(v1 - v2)

prox_fns = [
    ("ℓ₂² 收缩", lambda v: shrinkage(v, lam)),
    ("ℓ₁ 软阈值", lambda v: soft_threshold(v, lam)),
    ("ℓ₀ 硬阈值", lambda v: hard_threshold(v, lam)),
    ("投影", lambda v: projection_nonneg(v)),
]

for name, fn in prox_fns:
    # 注意：ℓ₀ 硬阈值虽然对应非凸问题，其近端算子仍是非扩张的
    dist_prox = np.linalg.norm(fn(v1) - fn(v2))
    ratio = dist_prox / dist_v
    ok = "✅" if ratio <= 1.0 + 1e-10 else "❌"
    print(f"  {ok} {name}: ratio = {ratio:.6f} {'≤' if ratio <= 1.0 + 1e-10 else '>'} 1.0")

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print("1. 近端算子 prox_{lam*g}(v) 在 v 附近寻找使 g(x) 最小的点")
print("2. 四种经典近端算子对应四种正则项:")
print("   - 收缩 (l2^2): 均匀压缩，不稀疏")
print("   - 软阈值 (l1): 归零+收缩，稀疏但有 shrinking bias")
print("   - 硬阈值 (l0): 归零+保留原值，无偏但非凸")
print("   - 投影 (约束): 满足可行约束")
print("3. 软阈值是 ISTA/FISTA 的核心操作: 先梯度步，再软阈值步")
print("4. shrinking bias 是 l1 正则化的固有代价，可用硬阈值或重加权l1缓解")

print("\n" + "=" * 60)
print(f"实验完成。结果已保存至: {SAVE_DIR}")
print("=" * 60)
