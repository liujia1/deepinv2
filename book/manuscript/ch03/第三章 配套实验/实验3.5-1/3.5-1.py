"""
2x2像素TV最小化：暴力搜索 vs 优化求解
对应章节：3.5 TV正则化与原始-对偶算法（开篇：TV优化的几何直觉）
知识点：TV正则化的优化地形；暴力搜索 vs 网格搜索 vs 优化求解的效率对比；
         各向异性TV的离散梯度算子；维度诅咒与迭代优化算法的必要性；
         ADMM精确参考解 vs L-BFGS-B次梯度停点的差距展示；
         L-BFGS-B在非光滑kink处的多初值×容差可靠性检验；
         0∈∂F(x*)最优性证书的数值验证；目标值与重建RMSE两种评价口径的区分
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import time
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
    SAVE_DIR = os.path.join(_gdrive, 'TV最小化暴力搜索')
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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")

np.random.seed(42)

# ══════════════════════════════════════════════════════════
# 1. 构造2x2像素CT问题
# ══════════════════════════════════════════════════════════

A = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
], dtype=float)

target = np.array([2.0, 2.0, 6.0, 7.0])
rhs = A @ target
alpha = 1.0

# 离散梯度算子（各向异性TV）
# 注意: 变量名按标准约定——DH计算水平差分(同行相邻)，DV计算垂直差分(同列相邻)
# 对于行优先排列的 2x2 图像 [[x1,x2],[x3,x4]]：
#   水平差分: x1-x2, x3-x4  (DH)
#   垂直差分: x1-x3, x2-x4  (DV)
DH = np.array([[1, -1, 0, 0],
               [0, 0, 1, -1]])
DV = np.array([[1, 0, -1, 0],
               [0, 1, 0, -1]])

def tv_cost(x_vec):
    data_fit = 0.5 * np.sum((A @ x_vec - rhs) ** 2)
    tv = np.sum(np.abs(DH @ x_vec)) + np.sum(np.abs(DV @ x_vec))
    return data_fit + alpha * tv

def tv_grad(x_vec):
    """TV目标函数的次梯度（subgradient）
    
    数据保真项梯度: ∇(0.5*||Ax-b||^2) = A^T(Ax-b)
    TV项次梯度: ∂||Dx||_1 = D^T · sign(Dx), 其中 sign(0)=0
    
    注意: TV在Dx=0处不可微，此处使用次梯度（sign(0)=0）而非经典梯度。
    
    重要说明: L-BFGS-B本质上是为光滑函数设计的拟牛顿法，对非光滑函数
    没有收敛性保证。此处以次梯度代入L-BFGS-B是实践中常用的工程做法
    （"碰巧有效"），但并非理论上正确的处理方式。严格处理TV的非光滑性
    需使用近端算子（如ISTA/FISTA）或原始-对偶算法（如Chambolle-Pock）。
    """
    # 数据保真项的梯度
    grad_data = A.T @ (A @ x_vec - rhs)
    
    # TV项的次梯度: D^T · sign(Dx)
    # DH: 水平差分算子, DV: 垂直差分算子
    grad_tv_h = DH.T @ np.sign(DH @ x_vec)
    grad_tv_v = DV.T @ np.sign(DV @ x_vec)
    
    return grad_data + alpha * (grad_tv_h + grad_tv_v)

# ══════════════════════════════════════════════════════════
# 2. 暴力搜索与随机采样
# ══════════════════════════════════════════════════════════

print("=" * 60)
print("暴力搜索 vs 优化求解：2x2 TV最小化")
print("=" * 60)
print(f"目标图像: {target.reshape(2, 2).tolist()}")
print(f"观测数据: rhs = {rhs}")
print(f"TV参数: alpha = {alpha}")

print("\n--- 方法A: 粗网格搜索 (10^4 = 1 万点) ---")
t0 = time.time()
n_grid = 10
# 搜索范围 [1, 7] 基于已知观测范围设定：
# rhs = [8, 9, 4, 13]，各像素值大致在 1~7 之间（由 target=[2,2,6,7] 决定）
# 若改变 target，需相应调整搜索范围
x_range = np.linspace(1, 7, n_grid)
best_cost_grid = np.inf
best_x_grid = None

for x1 in x_range:
    for x2 in x_range:
        for x3 in x_range:
            for x4 in x_range:
                x_vec = np.array([x1, x2, x3, x4])
                cost = tv_cost(x_vec)
                if cost < best_cost_grid:
                    best_cost_grid = cost
                    best_x_grid = x_vec.copy()

t_grid = time.time() - t0
print(f"粗网格解: x = [{best_x_grid[0]:.2f}, {best_x_grid[1]:.2f}, "
      f"{best_x_grid[2]:.2f}, {best_x_grid[3]:.2f}]")
print(f"目标函数 F = {best_cost_grid:.4f}, 耗时 = {t_grid:.3f} 秒")

print("\n--- 方法B: 随机采样 (1 万点) ---")
t0 = time.time()
N_rand = 10000
best_cost_rand = np.inf
best_x_rand = None

for _ in range(N_rand):
    x_vec = 1 + 6 * np.random.rand(4)
    cost = tv_cost(x_vec)
    if cost < best_cost_rand:
        best_cost_rand = cost
        best_x_rand = x_vec.copy()

t_rand = time.time() - t0
print(f"随机采样解: x = [{best_x_rand[0]:.2f}, {best_x_rand[1]:.2f}, "
      f"{best_x_rand[2]:.2f}, {best_x_rand[3]:.2f}]")
print(f"目标函数 F = {best_cost_rand:.4f}, 耗时 = {t_rand:.3f} 秒")

# ══════════════════════════════════════════════════════════
# 3. 优化求解 (scipy.optimize.minimize, L-BFGS-B)
# ══════════════════════════════════════════════════════════

print("\n--- 方法C: 优化求解 (scipy L-BFGS-B) ---")
t0 = time.time()

result = minimize(tv_cost, x0=[4.0, 4.0, 4.0, 4.0], method='L-BFGS-B',
                  jac=tv_grad, bounds=[(0, None)] * 4)
x_opt = result.x
t_opt = time.time() - t0
print(f"优化解: x = [{x_opt[0]:.4f}, {x_opt[1]:.4f}, "
      f"{x_opt[2]:.4f}, {x_opt[3]:.4f}]")
print(f"目标函数 F = {result.fun:.4f}, 耗时 = {t_opt:.6f} 秒")

# ══════════════════════════════════════════════════════════
# 3b. 精确参考解 (ADMM 变量分裂, 保留TV非光滑结构)
# ══════════════════════════════════════════════════════════

print("\n--- 方法D: 精确参考 (ADMM, 保留非光滑结构) ---")
t0 = time.time()
D_mat = np.vstack([DH, DV])

def admm_solve(rho=1.0, iters=50000, tol=1e-12):
    """ADMM 求解 min 0.5||Ax-b||^2 + alpha||Dx||_1

    变量分裂 z = Dx：x 步解线性系统，z 步是软阈值（L1 近端有闭式），
    完整保留 TV 的非光滑结构，对凸问题收敛到精确全局最优。

    返回: (x, 实际迭代数, 原始残差||Dx-z||, 对偶残差||rho D^T(z-z_prev)||)
    两个残差均趋于 0 是 ADMM 收敛的标准判据（Boyd et al., 2011）。
    """
    M_inv = np.linalg.inv(A.T @ A + rho * (D_mat.T @ D_mat))
    x = np.zeros(4); z = np.zeros(4); u = np.zeros(4)
    thr = alpha / rho
    r_norm = s_norm = np.inf
    admm_iters = 0
    for k in range(iters):
        x_prev, z_prev = x, z
        x_new = M_inv @ (A.T @ rhs + rho * (D_mat.T @ (z - u)))
        Dxu = D_mat @ x_new + u
        z_new = np.sign(Dxu) * np.maximum(np.abs(Dxu) - thr, 0.0)
        u = u + D_mat @ x_new - z_new
        # 缩放型 ADMM 残差（取范数，符号不影响收敛判定）
        r_norm = float(np.linalg.norm(D_mat @ x_new - z_new))
        s_norm = float(np.linalg.norm(rho * D_mat.T @ (z_new - z_prev)))
        dx = np.linalg.norm(x_new - x_prev)
        x = x_new; z = z_new
        admm_iters = k + 1
        if dx < tol and k > 1000:
            break
    return x, admm_iters, r_norm, s_norm

x_ref, admm_iters, admm_r_norm, admm_s_norm = admm_solve()
f_ref = tv_cost(x_ref)
t_ref = time.time() - t0
print(f"ADMM精确解: x = [{x_ref[0]:.4f}, {x_ref[1]:.4f}, "
      f"{x_ref[2]:.4f}, {x_ref[3]:.4f}]")
print(f"精确最优 F* = {f_ref:.6f}, 耗时 = {t_ref:.3f} 秒, "
      f"实际迭代 = {admm_iters} 次")
print(f"收敛残差: 原始 ||Dx-z|| = {admm_r_norm:.2e}, "
      f"对偶 ||rho D^T(z-z_prev)|| = {admm_s_norm:.2e}")
print(f"L-BFGS-B 与精确最优的差距: {result.fun - f_ref:.4f} "
      f"({(result.fun - f_ref) / f_ref * 100:.2f}%)")

# ══════════════════════════════════════════════════════════
# 3c. 最优性证书：数值验证 0 ∈ ∂F(x_ref)（而非仅靠"ADMM说它最优"）
# ══════════════════════════════════════════════════════════
# x_ref 处水平差分 DH x = 0（活动 kink），对应次梯度系数 s_h ∈ [-1,1]^2
# 可取（次微分是一个盒约束区间）；垂直差分非零，次梯度唯一。
# 解凸最优性条件 A^T(Ax-b) + DH^T s_h + DV^T sign(DV x) = 0，
# 若最小二乘残差≈0 且 s_h 落在 [-1,1]^2 内，则证书成立 => 全局最优。
print("\n--- 最优性证书验证: 0 ∈ ∂F(x*) ---")
g_data_ref = A.T @ (A @ x_ref - rhs)
g_vert_ref = DV.T @ np.sign(DV @ x_ref)
s_h, cert_res, _, _ = np.linalg.lstsq(DH.T, -(g_data_ref + g_vert_ref),
                                      rcond=None)
cert_resid = float(np.max(np.abs(cert_res))) if cert_res.size else 0.0
cert_ok = bool(cert_resid < 1e-8
               and np.all(s_h >= -1 - 1e-6) and np.all(s_h <= 1 + 1e-6))
print(f"水平kink次梯度系数 s_h = {np.round(s_h, 4)} (须∈[-1,1]^2), "
      f"证书方程残差 = {cert_resid:.2e}")
print(f"0 ∈ ∂F(x*) ? {cert_ok}  =>  "
      f"{'x_ref 被严格证明为全局最优（F凸）' if cert_ok else '证书验证失败，需检查!'}")

# ══════════════════════════════════════════════════════════
# 3d. L-BFGS-B 可靠性检验：多初值 × 两种容差
# ══════════════════════════════════════════════════════════
# 单个初值停在 F*+0.1% 可能只是容差偏松的偶然；用 20 个初值在
# 默认容差与紧容差下分别统计"到达精确最优的比例"和终止原因，
# 才能区分"容差问题"与"非光滑 kink 导致的本质不可靠"。
print("\n--- 方法E: L-BFGS-B 可靠性检验 (20 初值 × 默认/紧容差) ---")
rng_ms = np.random.default_rng(0)
x0_list = [np.array([4.0, 4.0, 4.0, 4.0])] + \
          [rng_ms.uniform(0.0, 8.0, 4) for _ in range(19)]

def _run_lbfgsb(x0, tight=False):
    opts = (dict(ftol=1e-15, gtol=1e-13, maxiter=10000) if tight else None)
    return minimize(tv_cost, x0, method='L-BFGS-B', jac=tv_grad,
                    bounds=[(0, None)] * 4, options=opts)

ms_summary = {}
for _tag, _tight in [("默认容差", False), ("紧容差", True)]:
    _runs = [_run_lbfgsb(x0, tight=_tight) for x0 in x0_list]
    _fvals = np.array([_r.fun for _r in _runs])
    _reach = float(np.mean(_fvals - f_ref < 1e-6))
    _msg_count = {}
    for _r in _runs:
        _key = str(_r.message).split(':')[0].strip()
        _msg_count[_key] = _msg_count.get(_key, 0) + 1
    ms_summary[_tag] = {
        'n_starts': len(x0_list),
        'f_min': float(round(_fvals.min(), 6)),
        'f_median': float(round(float(np.median(_fvals)), 6)),
        'f_max': float(round(_fvals.max(), 6)),
        'reach_Fstar_rate': round(_reach, 3),
        'termination': _msg_count,
    }
    print(f"[{_tag}] F: min={_fvals.min():.6f}, 中位={np.median(_fvals):.6f}, "
          f"max={_fvals.max():.6f}; 到达F*(gap<1e-6)比例 = {_reach:.0%}")
    print(f"        终止原因分布: {_msg_count}")

print("\n" + "=" * 60)
print("对比总结:")
print(f"  粗网格搜索:  F = {best_cost_grid:.4f}, 耗时 = {t_grid:.3f} 秒")
print(f"  随机采样:    F = {best_cost_rand:.4f}, 耗时 = {t_rand:.3f} 秒")
print(f"  优化求解:    F = {result.fun:.4f}, 耗时 = {t_opt:.6f} 秒")
print(f"  ADMM精确参考: F* = {f_ref:.4f}, 耗时 = {t_ref:.3f} 秒")
print(f"  优化优于粗网格: {best_cost_grid - result.fun:.4f}")
print(f"  优化优于随机采样: {best_cost_rand - result.fun:.4f}")
print(f"  L-BFGS-B默认容差到达F*比例: "
      f"{ms_summary['默认容差']['reach_Fstar_rate']:.0%} (20初值); "
      f"紧容差: {ms_summary['紧容差']['reach_Fstar_rate']:.0%}")

# ---- 两种评价口径：目标值 F（本实验的优化对象）vs 重建 RMSE（对真解）----
# 注意二者不可混为一谈：TV 解的 F 更低不代表更接近真解（正则化偏差/阶梯效应）。
def _rmse(x_vec):
    return float(np.sqrt(np.mean((x_vec - target) ** 2)))

rmse_all = {
    'grid': _rmse(best_x_grid),
    'random': _rmse(best_x_rand),
    'lbfgsb': _rmse(x_opt),
    'admm_ref': _rmse(x_ref),
}
print("\n  重建精度对照 (RMSE 对真解, 仅作口径区分, 非本实验优化目标):")
for _name, _v in rmse_all.items():
    print(f"    {_name:9s}: RMSE = {_v:.4f}")
print("  => F 最低的 ADMM 解 RMSE 并非最小：TV 把 [2,2;6,7] 拉平为 [2.5,2.5;6,6]，")
print("     这是 TV 正则化的固有偏差（阶梯效应），属正常现象。")

# ══════════════════════════════════════════════════════════
# 4. 可视化
# ══════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))

im0 = axes[0].imshow(target.reshape(2, 2), cmap='gray', vmin=0, vmax=7)
axes[0].set_title(r'真解 $[2, 2; 6, 7]$')
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(best_x_grid.reshape(2, 2), cmap='gray', vmin=0, vmax=7)
axes[1].set_title(
    r'粗网格搜索' + '\n' +
    rf'$F={best_cost_grid:.2f}$ $({t_grid:.2f}s)$'
)
plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(best_x_rand.reshape(2, 2), cmap='gray', vmin=0, vmax=7)
axes[2].set_title(
    r'随机采样(1万)' + '\n' +
    rf'$F={best_cost_rand:.2f}$ $({t_rand:.2f}s)$'
)
plt.colorbar(im2, ax=axes[2], fraction=0.046)

im3 = axes[3].imshow(x_opt.reshape(2, 2), cmap='gray', vmin=0, vmax=7)
axes[3].set_title(
    r'L-BFGS-B(次梯度)' + '\n' +
    rf'$F={result.fun:.3f}$ $({t_opt:.4f}s)$' + '\n' +
    rf'$F-F^*=+{result.fun - f_ref:.4f}$'
)
plt.colorbar(im3, ax=axes[3], fraction=0.046)

im4 = axes[4].imshow(x_ref.reshape(2, 2), cmap='gray', vmin=0, vmax=7)
axes[4].set_title(
    r'ADMM精确参考' + '\n' +
    rf'$F^*={f_ref:.3f}$ $({t_ref:.3f}s)$' + '\n' +
    rf'迭代${admm_iters}$次 $0\in\partial F$'
)
plt.colorbar(im4, ax=axes[4], fraction=0.046)

plt.suptitle(
    r'2$\times$2像素TV最小化 —— 暴力搜索 vs 优化求解（含精确参考解）',
    fontsize=13
)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(SAVE_DIR, '步骤1_2x2_TV搜索对比.png'), dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════
# 5. 结果输出与教学说明
# ══════════════════════════════════════════════════════════

print(f"\n[核心发现]")
print(f"  1. 2x2 像素 TV 优化是一个 4 维连续优化问题，")
print(f"     目标函数 F(x) = 0.5*||Ax - b||^2 + alpha * ||Dx||_1")
print(f"  2. 粗网格搜索 (10^4 点): 精度受网格分辨率限制，")
print(f"     加密一倍网格 → 搜索量增至 16 倍 (维度诅咒)")
print(f"  3. 随机采样 (10^4 点): 无系统性覆盖，精度有限")
print(f"  4. 优化求解 (L-BFGS-B): 高效但不保证精确——TV 是凸但非光滑函数，")
print(f"     次梯度代入拟牛顿法是工程做法，无收敛性保证。多初值检验显示：")
print(f"     默认容差下 20 个初值无一稳定到达 F*（受 ftol 偏松与 kink 双重影响）；")
print(f"     调紧容差后约半数可到 F*，但其余初值出现线搜索异常中止 (ABNORMAL)，")
print(f"     说明问题本质是【可靠性缺失】而非单纯容差选择。")
print(f"  5. ADMM 变量分裂把 L1 项交给有闭式解的软阈值处理，完整保留非光滑结构，")
print(f"     {admm_iters} 次迭代即收敛（原始/对偶残差≈0），并通过 0∈∂F(x*) 证书")
print(f"     严格验证为全局最优；大规模问题同理需 ADMM / Chambolle-Pock 等专用算法。")
print(f"  6. 评价口径提醒: 本实验比较的是目标值 F 与求解效率；F 更低不等于更接近真解——")
print(f"     TV 解的 RMSE 反而高于粗网格，正是阶梯效应（正则化偏差）的体现。")

print(f"\n[联系后续实验]")
print(f"  本实验展示了 TV 目标函数的几何地形与暴力搜索的局限。")
print(f"  3.5 节正文从 Fenchel 对偶出发，推导 Chambolle-Pock 原始-对偶算法，")
print(f"  在后续实验中将演示该算法在大规模图像上的保边去噪效果。")

print(f"\n{'=' * 60}")
print(f"实验完成")
print(f"输出图片: {os.path.join(SAVE_DIR, '步骤1_2x2_TV搜索对比.png')}")
print(f"{'=' * 60}")

# ===== 保存数值结果 =====
import json
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

results_summary = {
    'alpha': float(round(alpha, 2)),
    'target': _to_native(target),
    'rhs': _to_native(rhs),
    'best_cost_grid': float(round(best_cost_grid, 4)),
    'best_cost_rand': float(round(best_cost_rand, 4)),
    'best_x_grid': [float(round(v, 4)) for v in best_x_grid],
    'best_x_rand': [float(round(v, 4)) for v in best_x_rand],
    'opt_cost': float(round(result.fun, 4)),
    'x_opt': [float(round(v, 4)) for v in x_opt],
    'ref_cost_admm': float(round(f_ref, 6)),
    'x_ref_admm': [float(round(v, 6)) for v in x_ref],
    'gap_lbfgs_minus_ref': float(round(result.fun - f_ref, 6)),
    't_grid': float(round(t_grid, 3)),
    't_rand': float(round(t_rand, 3)),
    't_opt': float(round(t_opt, 6)),
    't_ref': float(round(t_ref, 3)),
    # --- 问题结构 ---
    'rank_A': int(np.linalg.matrix_rank(A)),
    'f_target': float(round(tv_cost(target), 6)),
    # --- ADMM 收敛诊断（原始/对偶残差应均≈0）---
    'admm_iters': int(admm_iters),
    'admm_primal_res': float(f"{admm_r_norm:.3e}"),
    'admm_dual_res': float(f"{admm_s_norm:.3e}"),
    # --- 最优性证书：0 ∈ ∂F(x*) 的数值验证 ---
    'optimality_certificate': {
        'subgrad_coef_sh': [float(round(v, 6)) for v in s_h],
        'lstsq_residual': cert_resid,
        'zero_in_subdifferential': cert_ok,
    },
    # --- L-BFGS-B 多初值 × 容差可靠性检验 ---
    'lbfgsb_multistart': ms_summary,
    # --- 重建精度口径（对真解 RMSE，非本实验优化目标，仅供区分）---
    'rmse_vs_target': {k: float(round(v, 6)) for k, v in rmse_all.items()},
}

results_summary = {k: _to_native(v) for k, v in results_summary.items()}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")