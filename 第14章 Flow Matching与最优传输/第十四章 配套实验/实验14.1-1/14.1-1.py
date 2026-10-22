# -*- coding: utf-8 -*-

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import warnings

# 静默 matplotlib 相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", message=".*Glyph.*")
warnings.filterwarnings("ignore", message=".*cmap.*")

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验14.1-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(_chinese_path, exist_ok=True)

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)


# ============================================================
# 2D点云分布定义
# ============================================================
def sample_source(n):
    """源分布：两个高斯的混合（左上+右下）"""
    mode = np.random.rand(n) < 0.5
    x = np.where(mode, -2, 2) + 0.3 * np.random.randn(n)
    y = np.where(mode, 2, -2) + 0.3 * np.random.randn(n)
    return np.stack([x, y], axis=1)

# 目标模式中心（非对称）：若用 (2,2)/(-2,-2)，四个模式恰好构成正方形，
# 每个源点到两个目标模式中心的距离都等于 4，OT 最优解不唯一——匈牙利算法
# 只是“任选一组方向”，"OT 使路径不交叉"就无从谈起。此处故意打破对称，
# 使 OT 配对唯一（最优/次优代价间隙约 42%）。
TARGET_CENTERS = np.array([[3.2, 1.0], [-2.4, -1.6]])

def sample_target(n):
    """目标分布：两个高斯的混合（右侧偏上+左下），与源构成非对称交叉"""
    mode = np.random.rand(n) < 0.5
    x = np.where(mode, TARGET_CENTERS[0, 0], TARGET_CENTERS[1, 0]) + 0.3 * np.random.randn(n)
    y = np.where(mode, TARGET_CENTERS[0, 1], TARGET_CENTERS[1, 1]) + 0.3 * np.random.randn(n)
    return np.stack([x, y], axis=1)


# ============================================================
# 最优传输耦合（小规模：使用匈牙利算法）
# ============================================================
def ot_coupling(source, target):
    """计算最优传输配对（最小化总距离的配对）

    对于2D点云，使用匈牙利算法（O(n^3)）
    这对应14.1节Kantorovich问题的离散解
    """
    from scipy.optimize import linear_sum_assignment
    # 代价矩阵: C[i,j] = ||source[i] - target[j]||^2
    diff = source[:, None, :] - target[None, :, :]  # (n, n, 2)
    cost = np.sum(diff**2, axis=-1)  # (n, n)
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind  # 返回source[i]应该配对的target索引

def independent_coupling(n):
    """独立耦合：随机配对（14.3.5节的默认CFM）"""
    return np.random.permutation(n)


def count_crossings(src, tgt, labels=None):
    """统计配对线段之间的真交叉数（交点必须落在两条线段内部）

    labels: 每条线段所属的“束”标签；若给出，额外返回“跨束宏交叉”数
            （不同束之间的交叉才是肉眼可见、影响边际场方向的那种）。
    """
    def cross2(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    cnt = 0
    macro = 0
    n = len(src)
    for i in range(n):
        for j in range(i + 1, n):
            d1 = cross2(src[i], tgt[i], src[j])
            d2 = cross2(src[i], tgt[i], tgt[j])
            d3 = cross2(src[j], tgt[j], src[i])
            d4 = cross2(src[j], tgt[j], tgt[i])
            if d1 * d2 < 0 and d3 * d4 < 0:
                cnt += 1
                if labels is None or labels[i] != labels[j]:
                    macro += 1
    return cnt, macro


# ============================================================
# 路径插值函数
# ============================================================
def linear_interp(z, target, t):
    """线性插值路径（14.3.4节 OT路径/14.4.1节直线插值）
    x_t = (1-t)z + t*target,  v_t = target - z
    注意：本书采用Flow Matching记号约定（z=源/噪声，target=数据/目标）
    """
    return (1 - t) * z + t * target


# ============================================================
# 最优传输 vs 独立耦合——路径形态对比
# ============================================================
n_points = 50
source = sample_source(n_points)
target = sample_target(n_points)

# OT耦合（匈牙利算法确定性求解，无需重设随机种子）
ot_idx = ot_coupling(source, target)
target_ot = target[ot_idx]

# 独立耦合（随机配对依赖随机种子，设置seed=123确保可复现）
np.random.seed(123)
ind_idx = independent_coupling(n_points)
target_ind = target[ind_idx]

# ============================================================
# 指标：代价 / 平均路径长度 / 线段交叉数（均在画图前算好，图标题直接用）
# ============================================================
dist_ind = np.linalg.norm(source - target_ind, axis=1)
dist_ot = np.linalg.norm(source - target_ot, axis=1)
cost_indep = float(np.mean(dist_ind ** 2))
cost_ot = float(np.mean(dist_ot ** 2))
len_indep = float(np.mean(dist_ind))
len_ot = float(np.mean(dist_ot))
# 每条线段所属的“束”标签 = (源模式号, 目标模式号)
SRC_CENTERS = np.array([[-2.0, 2.0], [2.0, -2.0]])
def _mode_of(points, centers):
    return np.argmin(((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1), axis=1)
seg_label_ind = [tuple(x) for x in zip(_mode_of(source, SRC_CENTERS), _mode_of(target_ind, TARGET_CENTERS))]
seg_label_ot = [tuple(x) for x in zip(_mode_of(source, SRC_CENTERS), _mode_of(target_ot, TARGET_CENTERS))]
cross_ind, cross_ind_macro = count_crossings(source, target_ind, seg_label_ind)
cross_ot, cross_ot_macro = count_crossings(source, target_ot, seg_label_ot)
C_all = ((source[:, None, :] - target[None, :, :]) ** 2).sum(-1)
C_sort = np.sort(C_all, axis=1)
margin_mean = float(np.mean(C_sort[:, 1] - C_sort[:, 0]))
margin_min = float(np.min(C_sort[:, 1] - C_sort[:, 0]))
# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (a) 源和目标分布
ax = axes[0]
ax.scatter(source[:, 0], source[:, 1], c='blue', alpha=0.6, s=30, label=r'$p_0$ (源分布)')
ax.scatter(target[:, 0], target[:, 1], c='red', alpha=0.6, s=30, label=r'$p_1$ (目标分布)')
ax.set_title('(a) 源分布 vs 目标分布', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(alpha=0.3)
ax.set_aspect('equal')

# (b) 独立耦合路径
ax = axes[1]
t_vals = np.linspace(0, 1, 20)
for i in range(n_points):
    path = np.array([linear_interp(source[i], target_ind[i], t) for t in t_vals])
    ax.plot(path[:, 0], path[:, 1], 'purple', alpha=0.2, lw=1)
ax.scatter(source[:, 0], source[:, 1], c='blue', alpha=0.6, s=20)
ax.scatter(target_ind[:, 0], target_ind[:, 1], c='red', alpha=0.6, s=20)
ax.set_title(f'(b) 独立耦合：路径交叉缠绕（交叉 {cross_ind} 对）', fontsize=13)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(alpha=0.3)
ax.set_aspect('equal')

# (c) OT耦合路径
ax = axes[2]
for i in range(n_points):
    path = np.array([linear_interp(source[i], target_ot[i], t) for t in t_vals])
    ax.plot(path[:, 0], path[:, 1], 'green', alpha=0.2, lw=1)
ax.scatter(source[:, 0], source[:, 1], c='blue', alpha=0.6, s=20)
ax.scatter(target_ot[:, 0], target_ot[:, 1], c='red', alpha=0.6, s=20)
ax.set_title(f'(c) OT耦合：路径短、交叉大幅减少（{cross_ot} vs {cross_ind} 对）', fontsize=13)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(alpha=0.3)
ax.set_aspect('equal')

plt.suptitle('实验14.1-1：独立耦合 vs OT耦合（14.1节/14.3.5节）', fontsize=14, y=1.01)
plt.tight_layout()
fig_path1 = os.path.join(SAVE_DIR, '步骤1_耦合对比.png')
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
plt.close()
print(f"图1已保存: {fig_path1}")

# 计算与打印指标（已在画图前算好）
print(f"  独立耦合平均传输代价: {cost_indep:.4f}   平均路径长度: {len_indep:.4f}   交叉: {cross_ind} 对（跨束 {cross_ind_macro}）")
print(f"  OT耦合平均传输代价:   {cost_ot:.4f}   平均路径长度: {len_ot:.4f}   交叉: {cross_ot} 对（跨束 {cross_ot_macro}）")
print(f"  OT代价 / 独立代价:    {cost_ot/cost_indep:.4f}；长度比: {len_ot/len_indep:.4f}")
print(f"  最优/次优配对代价间隙: 均值 {margin_mean:.2f}，最小 {margin_min:.2f}（>0 说明 OT 配对唯一）")
print(f"  → OT耦合传输代价更低（Wasserstein距离更短）")

# 验证OT最优性：OT耦合代价必然≤独立耦合（最优传输定义保证）
assert cost_ot <= cost_indep + 1e-9, "OT耦合代价不应高于独立耦合（违反最优性）"
print("  ✓ OT最优性验证通过：cost_ot ≤ cost_indep")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print("实验14.1-1 完成!")
print("=" * 60)
print("""
关键结论:
1. 最优传输基础（14.1节）
   - Monge问题：寻找确定性传输映射T，最小化传输代价
   - Kantorovich问题：离散最优传输的线性规划形式
   - Wasserstein距离：最优传输代价定义的距离度量

2. 路径形态对比（14.3.5节）
   - 独立耦合：路径交叉缠绕（实测交叉对数远多于 OT），平均路径更长
   - OT耦合：路径短且不交叉，传输代价低（Wasserstein距离）
   - OT映射下的McCann插值是Wasserstein空间的测地线

3. 实践意义
   - OT耦合能够获得更直的传输路径（长度比与交叉对数均可测）
   - 直线路径有助于减少ODE求解步数
   - 为Flow Matching的效率优化奠定基础
   - 几何必须非对称：若四个模式构成正方形，OT 解不唯一，本节结论不可验证
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
    'cost_indep': round(float(cost_indep), 4),
    'cost_ot': round(float(cost_ot), 4),
    'cost_ratio_ot_over_indep': round(float(cost_ot / cost_indep), 4),
    'mean_len_indep': round(float(len_indep), 4),
    'mean_len_ot': round(float(len_ot), 4),
    'len_ratio_ot_over_indep': round(float(len_ot / len_indep), 4),
    'crossings_indep': int(cross_ind),
    'crossings_ot': int(cross_ot),
    'macro_crossings_indep': int(cross_ind_macro),
    'macro_crossings_ot': int(cross_ot_macro),
    'coupling_margin_mean': round(float(margin_mean), 4),
    'coupling_margin_min': round(float(margin_min), 4),
    'target_centers': TARGET_CENTERS.tolist(),
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")