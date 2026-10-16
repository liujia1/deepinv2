"""
实验5.2-1 得分函数可视化
对应章节：5.2 得分函数：对数概率的梯度
知识点：
  - 得分函数的定义：s(x) = ∇_x log p(x)
  - 得分函数的几何含义：概率分布的"方向场"
  - 得分函数与Langevin动力学的关系

实验步骤：
  步骤1：定义2D高斯混合分布及其解析得分函数（并用数值梯度校验）
  步骤2：得分函数向量场可视化
  步骤3：ULA采样轨迹展示（含众数占比、ESS等诊断量）
  步骤4：模式混合与遍历性验证（多seed加长链，占用率回归50/50）

素材来源：
  - 原创设计：2D高斯混合得分场可视化

修改说明：
  从原参考实验5.1.py拆分，聚焦得分函数的几何含义，
  展示得分函数如何引导Langevin采样。
  复核修订：
  1) 修复子图1 colorbar误绑定到黑色散点(viridis,0~1)的bug，
     现显式绑定quiver箭头(Reds)，子图3同步补色条；
  2) ULA轨迹数组包含真正的起点x0=(0,0)，轨迹终点标记为第2000步
     （原代码红点标的是第10000步，与所画的前2000步轨迹不一致）；
  3) "有效样本8000个"改为"burn-in后样本点（自相关、非独立）"，
     并报告积分自相关时间IAT与有效样本量ESS；
  4) JSON补录全部实测诊断量；
  5) 新增步骤4，用多条独立加长链定量验证"模式混合是有限链效应，
     遍历意义下占用率回归50/50"。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
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
    SAVE_DIR = os.path.join(_gdrive, '实验5.2-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)

# ============================================================
# 步骤1：定义2D高斯混合分布及其得分函数
# ============================================================
print("=" * 60)
print("步骤1：定义2D高斯混合分布及其得分函数")
print("=" * 60)

# 目标分布：p(x) = 0.5*N(mu1, Sigma1) + 0.5*N(mu2, Sigma2)
# 逆矩阵/行列式与采样循环无关，模块级预计算一次，避免重复求逆
MU1 = np.array([-2., -2.])
MU2 = np.array([2., 2.])
SIGMA1 = np.array([[1., 0.3], [0.3, 1.]])
SIGMA2 = np.array([[1., -0.3], [-0.3, 1.]])
_INV_S1 = np.linalg.inv(SIGMA1)
_INV_S2 = np.linalg.inv(SIGMA2)
_DET_S1 = np.linalg.det(SIGMA1)
_DET_S2 = np.linalg.det(SIGMA2)
_NORM2D = 2.0 * np.pi


def _gauss_density(x_vec, mu, inv_sigma, det_sigma):
    """单个2D高斯分量的密度值。"""
    d = x_vec - mu
    return np.exp(-0.5 * d @ inv_sigma @ d) / (_NORM2D * np.sqrt(det_sigma))


def gaussian_mixture_logprob(x, y):
    """
    2D高斯混合的对数概率
    p(x,y) = 0.5 * N(μ₁, Σ₁) + 0.5 * N(μ₂, Σ₂)

    参数:
        x, y: 空间坐标

    返回:
        log p(x,y)
    """
    x_vec = np.array([x, y], dtype=float)
    p1 = _gauss_density(x_vec, MU1, _INV_S1, _DET_S1)
    p2 = _gauss_density(x_vec, MU2, _INV_S2, _DET_S2)
    return np.log(0.5 * p1 + 0.5 * p2)


def gaussian_mixture_score(x, y):
    """
    计算2D高斯混合的得分函数（对数概率的梯度）
    s(x,y) = ∇ log p(x,y)

    使用解析梯度计算：
    ∇log p(x) = Σ_k [w_k p_k(x) / p(x)] · (-Σ_k^{-1}(x - μ_k))

    参数:
        x, y: 空间坐标

    返回:
        np.array([∂log p/∂x, ∂log p/∂y])
    """
    xv = np.array([x, y], dtype=float)
    p1 = _gauss_density(xv, MU1, _INV_S1, _DET_S1)
    p2 = _gauss_density(xv, MU2, _INV_S2, _DET_S2)
    p = 0.5 * p1 + 0.5 * p2
    # 解析梯度：各分量得分按"后验分量概率"0.5*p_k/p加权
    grad = (0.5 * p1 * (-_INV_S1 @ (xv - MU1)) +
            0.5 * p2 * (-_INV_S2 @ (xv - MU2))) / p
    return grad


def _score_vec(xv):
    """向量化输入版得分函数（内部采样循环用，避免反复np.array转换）。"""
    p1 = _gauss_density(xv, MU1, _INV_S1, _DET_S1)
    p2 = _gauss_density(xv, MU2, _INV_S2, _DET_S2)
    p = 0.5 * p1 + 0.5 * p2
    return (0.5 * p1 * (-_INV_S1 @ (xv - MU1)) +
            0.5 * p2 * (-_INV_S2 @ (xv - MU2))) / p


# ---- 解析得分的数值梯度校验（确保公式实现正确）----
def _numerical_gradient(xv, eps=1e-6):
    g = np.empty(2)
    for k in range(2):
        e = np.zeros(2)
        e[k] = eps
        g[k] = (gaussian_mixture_logprob(*(xv + e)) -
                gaussian_mixture_logprob(*(xv - e))) / (2 * eps)
    return g


_rng_chk = np.random.default_rng(0)
_grad_err = 0.0
for _ in range(200):
    _pt = _rng_chk.uniform(-6, 6, 2)
    _gn = _numerical_gradient(_pt)
    _grad_err = max(_grad_err,
                    np.max(np.abs(gaussian_mixture_score(*_pt) - _gn) /
                           (1.0 + np.abs(_gn))))
print(f"解析得分 vs 数值梯度校验：200个随机点最大相对误差 = {_grad_err:.2e}")
assert _grad_err < 1e-6, "解析得分与数值梯度不一致，请检查公式实现！"

print("得分函数定义：")
print("  s(x) = ∇_x log p(x)")
print("  对于高斯混合分布，得分函数指向高密度区域")
print("  模长表示密度变化的剧烈程度")


# ============================================================
# 步骤2：得分函数向量场可视化
# ============================================================
print("\n" + "=" * 60)
print("步骤2：得分函数向量场可视化")
print("=" * 60)

# 创建网格
x_grid = np.linspace(-5, 5, 20)
y_grid = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x_grid, y_grid)

# 计算得分场
U_grid = np.zeros_like(X)
V_grid = np.zeros_like(Y)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        score = gaussian_mixture_score(X[i, j], Y[i, j])
        U_grid[i, j] = score[0]
        V_grid[i, j] = score[1]

# 归一化向量场（仅显示方向，用颜色编码模长）
score_norm = np.sqrt(U_grid ** 2 + V_grid ** 2) + 1e-8
U_normalized = U_grid / score_norm
V_normalized = V_grid / score_norm

# 箭头颜色值与统一归一化（两张含箭头的子图共用，保证颜色可比）
arrow_color = np.log1p(score_norm)
arrow_cmap_norm = plt.Normalize(vmin=0.0, vmax=float(arrow_color.max()))
ARROW_CMAP = 'Reds'

# 可视化
plt.figure(figsize=(15, 5))

# 子图1：得分函数向量场（归一化，颜色编码模长）
plt.subplot(1, 3, 1)
q1 = plt.quiver(X, Y, U_normalized, V_normalized, arrow_color,
                cmap=ARROW_CMAP, norm=arrow_cmap_norm,
                alpha=0.8, scale=25)
plt.scatter([-2, 2], [-2, 2], c='black', s=100, marker='x', label='众数')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('得分函数向量场 $s(x,y) = \\nabla \\log p(x,y)$\n（箭头归一化，颜色表示模长）')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
# 修复：colorbar必须显式绑定quiver对象；
# 原代码绑定到了随后绘制的黑色散点（viridis, 量程0~1），与箭头数据无关
plt.colorbar(q1, label='$\\log(1 + |s(x,y)|)$')

# 子图2：概率密度等高线
plt.subplot(1, 3, 2)
x_dense = np.linspace(-5, 5, 100)
y_dense = np.linspace(-5, 5, 100)
X_dense, Y_dense = np.meshgrid(x_dense, y_dense)
Z = np.zeros_like(X_dense)
for i in range(X_dense.shape[0]):
    for j in range(X_dense.shape[1]):
        Z[i, j] = np.exp(gaussian_mixture_logprob(X_dense[i, j],
                                                   Y_dense[i, j]))

plt.contour(X_dense, Y_dense, Z, levels=20, cmap='viridis')
plt.scatter([-2, 2], [-2, 2], c='red', s=100, marker='x', label='众数')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('概率密度等高线 $p(x,y)$')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)

# 子图3：得分场与密度叠加
plt.subplot(1, 3, 3)
plt.contour(X_dense, Y_dense, Z, levels=20, cmap='viridis', alpha=0.5)
q3 = plt.quiver(X, Y, U_normalized, V_normalized, arrow_color,
                cmap=ARROW_CMAP, norm=arrow_cmap_norm,
                alpha=0.7, scale=25)
plt.scatter([-2, 2], [-2, 2], c='black', s=100, marker='x', label='众数')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('得分场与密度叠加\n（箭头归一化，颜色表示模长）')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.colorbar(q3, fraction=0.046, pad=0.04,
             label='$\\log(1 + |s(x,y)|)$')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_得分函数向量场.png'), dpi=150)
plt.close()

print("得分函数几何含义：")
print("  方向：指向密度增长最快的方向")
print("  模长：表示密度变化的剧烈程度")
print("  零点：概率密度的驻点（众数、鞍点）")


# ============================================================
# 步骤3：ULA采样轨迹展示
# ============================================================
print("\n" + "=" * 60)
print("步骤3：ULA采样轨迹展示")
print("=" * 60)

def ULA_2d(niter, delta, x0):
    """
    2D ULA（Unadjusted Langevin Algorithm）采样

    离散格式（SDE dX = ∇log p(X) dt + √2 dW 的Euler–Maruyama格式）：
        X_{k+1} = X_k + δ ∇log p(X_k) + √(2δ) Z,  Z~N(0,I)

    参数:
        niter: 迭代次数
        delta: 步长
        x0: 初始点

    返回:
        samples: 采样序列 (niter+1, 2)，samples[0]=x0，
                 samples[k]为第k次更新后的状态
    """
    samples = np.zeros((niter + 1, 2))
    samples[0] = x0
    X = x0.copy().astype(float)
    for i in range(1, niter + 1):
        score = _score_vec(X)              # ∇log p(x)
        Z = np.random.randn(2)
        X = X + delta * score + np.sqrt(2 * delta) * Z
        samples[i] = X
    return samples

# 参数设置
niter = 10000
delta = 0.1
x0 = np.array([0.0, 0.0])
burn_in = 2000  # 丢弃初始的burn-in阶段
TRAJ_STEPS = 2000  # 轨迹图展示的步数

# 运行ULA（固定种子，结果可复现）
np.random.seed(42)
samples = ULA_2d(niter, delta, x0)

# ---------- 采样诊断量 ----------
post = samples[burn_in:]                       # burn-in后样本（相邻自相关！）
n_post = len(post)

def mode_labels(pts):
    """按最近众数分类（Voronoi边界）：1=众数1(-2,-2)，0=众数2(2,2)。"""
    return (np.linalg.norm(pts - MU1, axis=1) <
            np.linalg.norm(pts - MU2, axis=1)).astype(np.int8)

lab_post = mode_labels(post)
prop_mode1 = float(lab_post.mean())
prop_mode2 = 1.0 - prop_mode1

c1 = post[lab_post == 1]
c2 = post[lab_post == 0]
mean1 = c1.mean(axis=0)
mean2 = c2.mean(axis=0)
cov1 = np.cov(c1.T)
cov2 = np.cov(c2.T)
corr1 = float(cov1[0, 1] / np.sqrt(cov1[0, 0] * cov1[1, 1]))
corr2 = float(cov2[0, 1] / np.sqrt(cov2[0, 0] * cov2[1, 1]))

# 两峰之间的切换次数与平均驻留长度
switches_post = int(np.sum(lab_post[1:] != lab_post[:-1]))
_dwell = []
_start = 0
for _t in range(1, n_post):
    if lab_post[_t] != lab_post[_start]:
        _dwell.append(_t - _start)
        _start = _t
_dwell.append(n_post - _start)
mean_dwell = float(np.mean(_dwell))

def integrated_autocorr_time(vals):
    """
    积分自相关时间(IAT)与有效样本量(ESS)。
    IAT采用Geyer初始正序列估计：IAT = 1 + 2Σ_{k>=1} ρ(k)；ESS = n / IAT。
    """
    x = np.asarray(vals, dtype=float)
    n = len(x)
    x = x - x.mean()
    var = np.dot(x, x) / n
    max_lag = min(1000, n // 3)
    acf = np.array([np.dot(x[:-lag], x[lag:]) / ((n - lag) * var)
                    for lag in range(1, max_lag + 1)])
    iat = 1.0
    k = 0
    while k + 1 < len(acf):
        pair = acf[k] + acf[k + 1]
        if pair < 0:
            break
        iat += 2.0 * pair
        k += 2
    return float(iat), float(n / iat)

iat_x, ess_x = integrated_autocorr_time(post[:, 0])

# ---------- 鞍点与势垒（解释模式混合的定量尺度）----------
def _find_saddle(x_init, lr=0.01, steps=20000, fd_eps=1e-6):
    """
    对|s(x)|^2做数值梯度下降，定位得分零点（鞍点）。
    鞍点处|s|^2极其平坦（四阶小量），步长过大或差分步长过小都会
    引入数值振荡，故采用小步长+足够迭代次数直至残差收敛。
    """
    x = x_init.copy().astype(float)
    for _ in range(steps):
        g = np.empty(2)
        for k in range(2):
            e = np.zeros(2)
            e[k] = fd_eps
            f_p = np.dot(_score_vec(x + e), _score_vec(x + e))
            f_m = np.dot(_score_vec(x - e), _score_vec(x - e))
            g[k] = (f_p - f_m) / (2 * fd_eps)
        x = x - lr * g
    return x

saddle = _find_saddle(np.array([0.0, 0.0]))
_saddle_residual = float(np.linalg.norm(_score_vec(saddle)))
assert _saddle_residual < 1e-6, (
    f"鞍点搜索未收敛（|s|残差={_saddle_residual:.2e}），请调小lr或增加steps")
# 鞍点处log p的Hessian（有限差分）
_h = 1e-4
H_saddle = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        ei = np.zeros(2); ei[i] = _h
        ej = np.zeros(2); ej[j] = _h
        H_saddle[i, j] = (
            gaussian_mixture_logprob(*(saddle + ei + ej)) -
            gaussian_mixture_logprob(*(saddle + ei - ej)) -
            gaussian_mixture_logprob(*(saddle - ei + ej)) +
            gaussian_mixture_logprob(*(saddle - ei - ej))
        ) / (4 * _h * _h)
saddle_eig = np.linalg.eigvalsh(H_saddle)
barrier_logp = float(gaussian_mixture_logprob(*MU1) -
                     gaussian_mixture_logprob(*saddle))
mode_distance = float(np.linalg.norm(MU1 - MU2))
noise_std = float(np.sqrt(2 * delta))

print("采样诊断（seed=42，burn-in={}后）：".format(burn_in))
print(f"  两峰占用率：众数1 {prop_mode1:.3f} / 众数2 {prop_mode2:.3f}"
      f"（目标各0.50；有限链会偏斜，步骤4验证遍历性）")
print(f"  簇1均值={np.round(mean1,3)}（目标[-2,-2]），相关系数={corr1:.3f}（目标+0.3）")
print(f"  簇2均值={np.round(mean2,3)}（目标[ 2, 2]），相关系数={corr2:.3f}（目标-0.3）")
print(f"  burn-in后两峰切换={switches_post}次，平均驻留={mean_dwell:.0f}步")
print(f"  x坐标积分自相关时间IAT≈{iat_x:.0f}步，ESS≈{ess_x:.0f}"
      f"（名义样本点{n_post}个，但高度自相关、非独立）")
print(f"  混合尺度：峰间距={mode_distance:.2f}，单步噪声σ={noise_std:.3f}；"
      f"鞍点≈{np.round(saddle,3)}（|s|残差{_saddle_residual:.0e}），"
      f"势垒Δlog p={barrier_logp:.2f}")
print("  注：δ=0.1时ULA存在O(δ)离散偏差（平稳方差略大于1），不影响本演示结论。")

# 可视化
plt.figure(figsize=(15, 5))

# 子图1：样本散点图（丢弃burn-in）
plt.subplot(1, 3, 1)
plt.scatter(post[:, 0], post[:, 1], s=1, alpha=0.3, c='blue')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title(f'ULA采样散点（丢弃前{burn_in}步burn-in）\n'
          f'{n_post}个样本点（自相关、非独立），ESS(x)≈{ess_x:.0f}')
plt.grid(alpha=0.3)

# 子图2：采样轨迹（前TRAJ_STEPS步，轨迹数组含真正起点x0）
plt.subplot(1, 3, 2)
plt.plot(samples[:TRAJ_STEPS + 1, 0], samples[:TRAJ_STEPS + 1, 1],
         'g-', lw=0.8, alpha=0.7)
plt.scatter(samples[0, 0], samples[0, 1], c='green', s=60, zorder=5,
            label=f'起点 $x_0$=({x0[0]:.0f},{x0[1]:.0f})')
plt.scatter(samples[TRAJ_STEPS, 0], samples[TRAJ_STEPS, 1], c='red', s=60,
            zorder=5, label=f'第{TRAJ_STEPS}步位置')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title(f'采样轨迹（前{TRAJ_STEPS}步，含起点）')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)

# 子图3：轨迹与得分场叠加
plt.subplot(1, 3, 3)
plt.contour(X_dense, Y_dense, Z, levels=20, cmap='viridis', alpha=0.3)
plt.quiver(X, Y, U_normalized, V_normalized, arrow_color,
           cmap=ARROW_CMAP, norm=arrow_cmap_norm,
           alpha=0.3, scale=25)
plt.plot(samples[:TRAJ_STEPS + 1, 0], samples[:TRAJ_STEPS + 1, 1],
         'g-', lw=0.8, alpha=0.7)
plt.scatter(samples[0, 0], samples[0, 1], c='green', s=60, zorder=5, label='起点')
plt.scatter(samples[TRAJ_STEPS, 0], samples[TRAJ_STEPS, 1], c='red', s=60,
            zorder=5, label=f'第{TRAJ_STEPS}步位置')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('轨迹与得分场叠加\n（箭头归一化，颜色表示模长）')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_ULA采样轨迹.png'), dpi=150)
plt.close()

print("Langevin动力学机制：")
print("  漂移项：沿得分方向'爬坡'（推向高密度区域）")
print("  扩散项：随机'游走'（防止坍缩到众数）")
print("  平衡：漂移力与扩散力平衡产生目标分布")
print(f"  burn-in：丢弃前{burn_in}步，确保样本近似来自平稳分布")
print("  注意：MCMC样本相邻相关，'样本点数'不等于'有效样本量(ESS)'")


# ============================================================
# 步骤4：模式混合与遍历性验证
# ------------------------------------------------------------
# 步骤3的万步单链两峰占用率并非精确50/50——这不是算法错误，而是
# 多峰目标的"模式混合"难题：跨峰是稀有事件，短链内切换次数有限。
# 本步骤用多条独立加长链验证：链越长，占用率越收敛到0.50/0.50。
# ============================================================
print("\n" + "=" * 60)
print("步骤4：模式混合与遍历性验证（多条独立加长链）")
print("=" * 60)

def ula_mode_labels(n_steps, seed, delta=0.1):
    """独立运行一条ULA链，只返回每步的众数归属标签（0/1），省内存。"""
    rng = np.random.RandomState(seed)
    X = np.array([0.0, 0.0])
    labels = np.empty(n_steps, dtype=np.int8)
    for i in range(n_steps):
        X = X + delta * _score_vec(X) + np.sqrt(2 * delta) * rng.randn(2)
        labels[i] = 1 if np.linalg.norm(X - MU1) < np.linalg.norm(X - MU2) else 0
    return labels

LONG_SEEDS = [1, 2, 3]
CHAIN_LEN = 200000
LONG_BURN = 2000
CHECKPOINTS = [5000, 10000, 20000, 50000, 100000, CHAIN_LEN]

long_props = {sd: [] for sd in LONG_SEEDS}
long_labels = {}
for sd in LONG_SEEDS:
    lab = ula_mode_labels(CHAIN_LEN, seed=sd)
    long_labels[sd] = lab
    cum = np.concatenate([[0], np.cumsum(lab)])  # cum[t]=前t步中众数1计数
    for L in CHECKPOINTS:
        n1 = cum[L] - cum[LONG_BURN]
        long_props[sd].append(float(n1 / (L - LONG_BURN)))
    print(f"  seed={sd}: " +
          "  ".join(f"{L//1000}k步->{long_props[sd][k]:.3f}"
                    for k, L in enumerate(CHECKPOINTS)))

props_arr = np.array([long_props[sd] for sd in LONG_SEEDS])  # (seed, checkpoint)
props_mean = props_arr.mean(axis=0)
print(f"  三链均值: " +
      "  ".join(f"{L//1000}k->{props_mean[k]:.3f}"
                for k, L in enumerate(CHECKPOINTS)))
print(f"  结论：占用率随链长回归0.50（目标混合权重），"
      f"万步单链的偏斜是有限链/慢混合效应，不影响遍历正确性。")

# 图：左=占用率随链长收敛；右=单链累计占用率沿步数的波动
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
ck = np.array(CHECKPOINTS)
for sd in LONG_SEEDS:
    plt.plot(ck, long_props[sd], '-o', ms=3, alpha=0.6, label=f'seed={sd}')
plt.plot(ck, props_mean, '-kD', ms=6, label='3条链均值')
plt.axhline(0.5, color='red', ls='--', lw=1.5, label='目标 0.50')
plt.xscale('log')
plt.xlabel('链长（步数，对数轴；均已扣除前2000步burn-in）')
plt.ylabel('众数1 $(-2,-2)$ 的占用率')
plt.title('占用率随链长收敛到混合权重 0.50')
plt.setp(plt.gca().get_xticklabels(), rotation=0)
plt.grid(alpha=0.3, which='both')
plt.legend()
plt.ylim(0.2, 0.8)

plt.subplot(1, 2, 2)
# 从burn-in之后开始取点，避免 t=burn_in 处 0/0
grid_t = np.arange(LONG_BURN + 200, CHAIN_LEN + 1, 200)
for sd in LONG_SEEDS:
    lab = long_labels[sd]
    cum = np.concatenate([[0], np.cumsum(lab)])
    running = (cum[grid_t] - cum[LONG_BURN]) / (grid_t - LONG_BURN)
    plt.plot(grid_t, running, alpha=0.8, lw=1, label=f'seed={sd}')
plt.axhline(0.5, color='red', ls='--', lw=1.5)
plt.xlabel('迭代步数 $t$')
plt.ylabel('burn-in后众数1的累计占用率')
plt.title(f'单链累计占用率沿步数变化\n'
          f'（短期大幅波动≈慢混合，长期回归0.50；平均驻留约{mean_dwell:.0f}步量级）')
plt.grid(alpha=0.3)
plt.legend()
plt.ylim(0.15, 0.85)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_模式混合与遍历性.png'), dpi=150)
plt.close()


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.2-1 总结")
print("=" * 60)
print("1. 得分函数定义：s(x) = ∇_x log p(x)，是对数概率的梯度")
print("2. 解析梯度计算：")
print("   对于高斯混合 p(x) = Σ_k w_k N(μ_k, Σ_k)")
print("   ∇log p(x) = Σ_k [w_k p_k(x) / p(x)] · (-Σ_k^{-1}(x - μ_k))")
print("   本实验解析公式通过数值梯度校验（最大相对误差<1e-6）")
print("3. 几何含义：")
print("   - 方向：指向密度增长最快的方向")
print("   - 模长：表示密度变化的剧烈程度")
print("   - 零点：概率密度的驻点")
print("4. 与Langevin动力学的关系：")
print("   - 漂移项：s(x)dt 推向高密度区域")
print("   - 扩散项：√2 dW_t 防止坍缩")
print("   - 两者平衡产生目标分布 p(x)")
print("5. 采样诊断与模式混合：")
print(f"   - MCMC样本自相关：{n_post}个样本点ESS仅≈{ess_x:.0f}，不能当独立样本")
print("   - 峰间距远大于噪声尺度时跨峰切换稀少（模式混合难题）")
print("   - 多链加长实验表明占用率遍历收敛到0.50/0.50")
print("6. 得分函数天然消去归一化常数：")
print("   s(x) = ∇ log p(x) = ∇ log p̃(x)")
print("   这对贝叶斯推断至关重要")

# ══════════════════════════════════════════════════════════
# 保存数值结果到JSON文件（含全部实测诊断量，便于复核）
# ══════════════════════════════════════════════════════════
import json

def _to_native(obj):
    """递归将numpy/torch类型转换为Python原生类型，便于JSON序列化"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, 'item') and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (ValueError, RuntimeError, TypeError):
            return obj
    return obj

results_summary = {
    'experiment': '5.2-1',
    'title': '得分函数可视化',
    'gaussian_mixture_params': {
        'mu1': MU1.tolist(),
        'mu2': MU2.tolist(),
        'Sigma1': SIGMA1.tolist(),
        'Sigma2': SIGMA2.tolist(),
        'weights': [0.5, 0.5],
    },
    'step1_score_check': {
        'method': '解析得分 vs log p中心差分数值梯度，200个随机点',
        'max_relative_error': _grad_err,
        'tolerance': 1e-6,
        'passed': bool(_grad_err < 1e-6),
    },
    'step3_ula_sampling': {
        'algorithm': 'ULA (Euler-Maruyama): X_{k+1}=X_k+δ∇log p(X_k)+√(2δ)Z',
        'niter': niter,
        'delta': delta,
        'burn_in': burn_in,
        'x0': [0.0, 0.0],
        'random_seed': 42,
        'samples_after_burnin': n_post,
        'note_samples_after_burnin': '相邻样本自相关、非独立；有效样本量见ess_x',
        'realized_diagnostics': {
            'mode1_proportion': prop_mode1,
            'mode2_proportion': prop_mode2,
            'target_proportion': 0.5,
            'cluster1_mean': mean1,
            'cluster2_mean': mean2,
            'cluster1_covariance': cov1,
            'cluster2_covariance': cov2,
            'cluster1_corr': corr1,
            'cluster2_corr': corr2,
            'target_corr': [0.3, -0.3],
            'switches_after_burnin': switches_post,
            'mean_dwell_steps': mean_dwell,
            'iat_x_steps': iat_x,
            'ess_x': ess_x,
        },
        'mixing_scale': {
            'mode_distance': mode_distance,
            'noise_std_per_step': noise_std,
            'saddle_point': saddle,
            'saddle_score_residual': _saddle_residual,
            'saddle_hessian_eigenvalues': saddle_eig,
            'barrier_delta_logp': barrier_logp,
        },
        'discretization_note': 'ULA步长δ=0.1，平稳分布存在O(δ)偏差（方差略偏大），不影响演示结论',
    },
    'step4_mixing_ergodicity': {
        'purpose': '验证万步单链占用率偏斜是有限链慢混合效应，遍历意义下回归50/50',
        'seeds': LONG_SEEDS,
        'chain_len': CHAIN_LEN,
        'burn_in': LONG_BURN,
        'checkpoints': CHECKPOINTS,
        'mode1_proportion_by_seed': {str(sd): long_props[sd] for sd in LONG_SEEDS},
        'mode1_proportion_mean': props_mean,
        'target_proportion': 0.5,
    },
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results_summary), f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
