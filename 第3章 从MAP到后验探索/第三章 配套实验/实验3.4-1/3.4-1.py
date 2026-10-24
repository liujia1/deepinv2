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

    [平局点约定] 当 |v| = tau 时目标值在 x=0 与 x=v 处相等（lam = v^2/2），
    二者同为最小解；本实现按 >= 约定保留原值 v
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

# 标注归零区域（两个区域互斥，避免重叠配色造成歧义）
axes[1].fill_between(v, -3, 3, where=(np.abs(v) <= lam),
                     alpha=0.10, color='blue',
                     label=r'共同归零区 $|v| \leq \lambda$')
axes[1].fill_between(v, -3, 3,
                     where=(np.abs(v) < np.sqrt(2 * lam)) & (np.abs(v) > lam),
                     alpha=0.15, color='red', hatch='//', linewidth=0.0,
                     label=r'硬阈值额外归零区 $\lambda < |v| < \sqrt{2\lambda}$')
for _t in (-np.sqrt(2 * lam), np.sqrt(2 * lam)):
    axes[1].axvline(x=_t, color='red', linestyle=':', linewidth=1.0, alpha=0.7)

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
print(f"    - 平局点 |v| = τ 时 0 与 v 同为最优解，按 ≥ 约定保留 v")
print(f"  投影: max(v, 0)")
print(f"    - 负分量归零，正分量保留")

print(f"\n[软阈值 vs 硬阈值]")
print(f"  软阈值归零区: |v| ≤ λ = {lam:.1f}")
print(f"  硬阈值归零区: |v| < τ = sqrt(2λ) = {np.sqrt(2*lam):.2f}")
print(f"  关键差异: 软阈值对非零值收缩 λ（有偏），硬阈值保留原值（无偏但非凸）")

# ══════════════════════════════════════════════════════════
# 3. 非扩张性（non-expansiveness）数值验证
# ══════════════════════════════════════════════════════════
# 对凸且正常的 g，其近端算子非扩张（更强的 firmly nonexpansive）：
#   ||prox(v1) - prox(v2)|| <= ||v1 - v2||
# 这是 ISTA/FISTA 收敛性的理论基础。注意：非扩张性由凸性保证，
# 对非凸的 g（如 ||x||_0）不成立（见下方硬阈值条目的说明）

print(f"\n[非扩张性验证]  ||prox(v1) - prox(v2)|| ≤ ||v1 - v2||")
print(f"  对凸 g 这是 ISTA/FISTA 收敛性的关键性质: 近端算子是 1-Lipschitz 的")
print(f"  注意: 下面是【单对随机向量】的经验比值, 只是抽样, 不能当作算子的收缩常数——")
print(f"        非扩张性是全称命题(要对所有向量对成立), 故下一段另测最坏情形 Lipschitz 常数")

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
    # 注意：ℓ₀ 硬阈值对应非凸 g，其在阈值 |v|=√(2λ) 处不连续（阈值两侧各取
    # 一点可使比值任意大），理论上不保证非扩张——非扩张性由凸性保证；
    # 随机向量对几乎不会恰好跨越阈值边界，故经验 ratio≤1，这只是经验值，
    # 恰好预示 IHT 因非凸只有临界点保证、无全局收敛结论
    dist_prox = np.linalg.norm(fn(v1) - fn(v2))
    ratio = dist_prox / dist_v
    if '硬阈值' in name:
        mark, verdict = "⚠️", "经验≤1（ℓ₀非凸，不保证，见下方反例）"
    else:
        passed = ratio <= 1.0 + 1e-10
        mark = "✅" if passed else "❌"
        verdict = "≤ 1.0" if passed else "> 1.0"
    print(f"  {mark} {name}: ratio = {ratio:.6f}  {verdict}")

# ---- 最坏情形 Lipschitz 常数：非扩张性是全称命题，单对抽样不足以证明 ----
# 构造"最坏见证对"：两点同号、且都落在算子的有效区（|v|>λ）内，
# 此时输出间距 = 输入间距，比值达到上确界（对 l1/投影即 1；l2 收缩恒为 1/(1+λ)）。
_a, _b = np.array([2.0]), np.array([3.0])
worst_case = {}
worst_case["ℓ₂² 收缩"] = float(
    np.linalg.norm(shrinkage(_b, lam) - shrinkage(_a, lam)) / np.linalg.norm(_b - _a))
worst_case["ℓ₁ 软阈值"] = float(
    np.linalg.norm(soft_threshold(_b, lam) - soft_threshold(_a, lam)) / np.linalg.norm(_b - _a))
worst_case["投影"] = float(
    np.linalg.norm(projection_nonneg(_b) - projection_nonneg(_a)) / np.linalg.norm(_b - _a))
worst_case["ℓ₀ 硬阈值"] = None      # None = 上确界为 ∞（阈值处不连续，见下方反例）

print(f"\n[最坏情形 Lipschitz 常数]  上确界 sup ||prox(v1)-prox(v2)|| / ||v1-v2||")
print(f"  构造最坏见证对 v1=2.0, v2=3.0（同号且都在有效区 |v|>λ 内）:")
for _nm in ("ℓ₂² 收缩", "ℓ₁ 软阈值", "ℓ₀ 硬阈值", "投影"):
    _v = worst_case[_nm]
    print(f"  {_nm}: {'∞（阈值处不连续）' if _v is None else f'{_v:.6f}'}")
print(f"  → ℓ₁ 软阈值与非负投影的界【紧】: 最坏比值恰为 1（即非扩张但非严格收缩），")
print(f"    随机抽样得到的 0.392 / 0.586 只是这一次抽样的运气，不能当作收缩常数。")

# 随机多对抽样，确认没有比见证对更大的比值（凸算子最坏界已被见证对取到）
_rng_wc = np.random.default_rng(12345)
print(f"\n[随机多对抽样的最大比值]  （2000 对，确认见证对已达上界）")
_wc_max = {}
for _nm, _fn in prox_fns:
    _rr = []
    for _ in range(2000):
        _a2 = _rng_wc.standard_normal(200)
        _b2 = _rng_wc.standard_normal(200)
        _rr.append(np.linalg.norm(_fn(_a2) - _fn(_b2)) / np.linalg.norm(_a2 - _b2))
    _wc_max[_nm] = float(np.max(_rr))
    print(f"  {_nm}: max ratio = {_wc_max[_nm]:.6f}")

# ---- 硬阈值非扩张反例：阈值两侧各取一点，比值随 ε→0 发散 ----
print(f"\n[硬阈值非扩张反例]  v1=τ-ε, v2=τ+ε（τ=√(2λ)={np.sqrt(2*lam):.4f}）")
ce_records = []
for _eps in (1e-1, 1e-2, 1e-3, 1e-4):
    _v1 = np.array([np.sqrt(2 * lam) - _eps])
    _v2 = np.array([np.sqrt(2 * lam) + _eps])
    _r = float(np.linalg.norm(hard_threshold(_v1, lam) - hard_threshold(_v2, lam))
               / np.linalg.norm(_v1 - _v2))
    ce_records.append({'eps': float(_eps), 'ratio': round(_r, 4)})
    print(f"  ε = {_eps:.0e}: 输入间距 {2*_eps:.5f}, 输出间距 {np.sqrt(2*lam)+_eps:.5f}, "
          f"ratio = {_r:.1f}  (>1，非扩张被破坏)")
print(f"  多维情形同样违约（仅 1 个坐标紧贴阈值即可）；ratio ≈ τ/(2ε) 无上界。")
print(f"  故随机高斯对的经验 ratio≤1 只是抽样盲区（连续分布恰好骑跨阈值的概率为0），")
print(f"  不能当作非扩张成立的证据——这正是 IHT 仅有临界点保证、无全局收敛结论的根源。")

# ══════════════════════════════════════════════════════════
# 3.5 近端最优性条件（次梯度）数值验证
# ══════════════════════════════════════════════════════════
# x* = prox_{λg}(v) ⟺ 0 ∈ λ∂g(x*) + (x* - v)（凸、下半连续 g）。
# 可微情形直接算梯度残差；ℓ1 不可微，x*=0 处用次微分区间 [-λ,λ] 判定；
# 非负投影用指示函数的 KKT 条件；ℓ0 非凸、无凸次梯度，改用目标值直接比较。
print(f"\n[最优性条件] 0 ∈ λ∂g(x*) + (x* - v) 的残差（理论值应为 0）")
vs = np.array([-2.3, -1.0, -0.4, 0.0, 0.4, 1.0, np.sqrt(2 * lam), 2.3])
# ℓ2²（可微）：∂g(x*)=x*，残差 = λx* + x* - v
x_l2 = shrinkage(vs, lam)
res_l2 = np.abs(lam * x_l2 + x_l2 - vs)
# ℓ1（不可微）：x*≠0 时 ∂g={sign(x*)}；x*=0 时 ∂g=[-1,1]，残差=max(0,|v|-λ)
x_l1 = soft_threshold(vs, lam)
res_l1 = np.where(x_l1 != 0,
                  np.abs(lam * np.sign(x_l1) + x_l1 - vs),
                  np.maximum(np.abs(vs) - lam, 0.0))
# 非负投影 ι_{x≥0}：可行残差 max(0,-x*) + 对偶残差 max(0, v-x*)
x_pj = projection_nonneg(vs)
res_proj = np.maximum(-x_pj, 0.0) + np.maximum(vs - x_pj, 0.0)
print(f"  ℓ₂² 收缩   最大残差: {res_l2.max():.2e}")
print(f"  ℓ₁ 软阈值  最大残差: {res_l1.max():.2e}（x*=0 处按次微分区间判定）")
print(f"  非负投影   最大残差: {res_proj.max():.2e}（可行性 + 对偶可行性）")
# ℓ0：直接比较 φ(0)=½v² 与 φ(v)=λ（v≠0），严格小才归零，平局按 ≥ 约定保留 v；
# 注意 v=0 时两个候选点重合于 x=0（φ 均为 0），输出自然为 0
choose_zero = hard_threshold(vs, lam) == 0
phi_at_zero = 0.5 * vs ** 2
phi_at_v = np.where(vs != 0, lam, 0.0)
predicted_zero = (vs == 0) | (phi_at_zero < phi_at_v)
ok_l0 = bool(np.all(choose_zero == predicted_zero))
print(f"  ℓ₀ 硬阈值: 目标值比较 φ(0)=½v² vs φ(v)=λ 与输出完全一致: {ok_l0}")
print(f"             |v|=√(2λ) 时 φ(0)=φ(v) 平局，按 ≥ 约定保留 v")

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
print("5. 非扩张性由凸性保证: ℓ0 硬阈值存在 ratio>1 的反例（随 ε→0 发散），")
print("   故 IHT 仅有临界点保证、无全局收敛结论；随机抽样 ratio≤1 不构成证据")

print("\n" + "=" * 60)
print(f"实验完成。结果已保存至: {SAVE_DIR}")
print("=" * 60)

# ===== 保存数值结果 =====
import json
# 收集非扩张性验证比率
non_expansiveness = {}
for name, fn in prox_fns:
    dist_prox = np.linalg.norm(fn(v1) - fn(v2))
    ratio = dist_prox / dist_v
    non_expansiveness[name] = float(round(ratio, 6))

results_summary = {
    'lam': float(round(lam, 2)),
    'n_test': n_test,
    'seed': 42,
    'non_expansiveness': {
        'single_pair_ratio': non_expansiveness,
        'worst_case_lipschitz_supremum': {k: v for k, v in worst_case.items()},
        'random_multipair_max_ratio': {k: float(round(v, 6)) for k, v in _wc_max.items()},
        'note': 'single_pair_ratio 为单对随机高斯向量(seed=42)的经验比值，只是抽样；'
                '非扩张性是全称命题，算子的真实界是 worst_case_lipschitz_supremum：'
                'ℓ₁ 软阈值与投影的界紧（恰为 1），ℓ₂² 收缩恒为 1/(1+λ)=0.5，'
                'ℓ₀ 硬阈值上确界为 ∞（None）；random_multipair_max_ratio 为 2000 对随机抽样的最大值'
    },
    'hard_threshold_nonexpansive': False,
    'hard_threshold_counterexample': {
        'setup': 'v1=sqrt(2*lam)-eps, v2=sqrt(2*lam)+eps（标量，多维同理）',
        'ratios': ce_records,
        'note': '硬阈值在阈值处不连续（||x||_0 非凸），ratio≈sqrt(2λ)/(2ε) 随 ε→0 发散，'
                '非扩张不成立；随机高斯对的经验 ratio≤1 仅为抽样盲区，IHT 只有临界点保证'
    },
    'optimality_residual_0_in_lam_dg_plus_x_minus_v': {
        'l2_shrinkage_max': float(round(res_l2.max(), 12)),
        'l1_soft_threshold_max': float(round(res_l1.max(), 12)),
        'nonneg_projection_max': float(round(res_proj.max(), 12)),
        'l0_objective_value_check_passed': ok_l0,
        'note': '凸算子次梯度/KKT 残差应≈0；ℓ0 非凸无凸次梯度，改用 φ(0) vs φ(v) 目标值比较'
    },
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
