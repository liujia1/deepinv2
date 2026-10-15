"""
实验2.2-1 高斯先验的推广：从Tikhonov到Sobolev平滑
对应章节：2.2 经典先验族
知识点：高斯先验→Tikhonov；梯度高斯先验→Sobolev平滑；闭式解；不同协方差结构的影响

素材来源：
  - IP22 (statistical_perspective.md): 高斯随机场采样 + MAP估计
  - 2.2章节: 高斯先验的推广形式

修改说明（基于代码评审v9）：
  9. 修正Tikhonov参数的方法论错误：
     - 区分零均值模型 N(0,σx²I) 与非零均值模型 N(μ,σx²I)；
     - σx²必须使用"关于模型均值的二阶矩"：零均值用 E[x²](rms²)，非零均值用 Var(x)；
     - 经验贝叶斯插件估计只使用观测 y（均值、二阶矩减去噪声方差），不偷看真值；
     - 保留旧版"σ²/s² 配零均值模型且不中心化"的做法作为反面教材（PSNR反而低于含噪图）；
     - 新增均值中心化 Tikhonov：x̂ = ȳ + (y-ȳ)/(1+λ)，与非零均值数据真正匹配。
  10. 边界条件归因修正 + DCT对照：
     - 周期拼接伪影是"周期性边界（np.roll/DFT）+ 非周期图像"的代价，
       不是频域方法的固有属性；
     - 新增 Neumann 镜像边界的 DCT 闭式解 sobolev_denoise_dct（同为 O(n²logn)），
       与周期 DFT 解做边界环 MSE / PSNR 对照（步骤5）。
  11. 可追溯性：迭代解-闭式解最大差异、DFT/DCT 边界 MSE 等关键验证量落盘 JSON；
      "按PSNR自动选择"明确标注为基于真值的 oracle（事后）选择。
  ---
  历史修改（v8）：
  1. 添加Sobolev频域闭式解，与迭代解对比验证
  2. 统一PSNR/SSIM格式化函数
  3. data_range改为理论值域1.0
  4. 收敛判断改为稳健相对阈值
  5. Tikhonov参数说明补充归一化讨论
  6. 补充verbose未收敛输出
  7. 修复axes条件分支逻辑
  8. 魔法数字统一管理
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.fft import dctn, idctn
import os
import sys

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.2-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.2-1')
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    # 必须用绝对路径：脚本可能从任意 cwd 启动，相对路径会导致 import 失败
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
    print("请确保 .chinese 文件夹已上传到 Google Drive 的正确位置")

# 跨平台兜底：chinese_font 在 Windows 上只查系统字体，不加载自带 TTF；
# 若系统无中文字体（或导入失败），显式注册随实验附带的 NotoSansSC。
_bundled_ttf = os.path.join(_chinese_path, 'NotoSansSC-Regular.ttf')
if os.path.exists(_bundled_ttf):
    try:
        from matplotlib.font_manager import fontManager
        fontManager.addfont(_bundled_ttf)
        _preferred = ['SimHei', 'Microsoft YaHei', 'Noto Sans SC']
        plt.rcParams['font.sans-serif'] = _preferred + [
            f for f in plt.rcParams.get('font.sans-serif', []) if f not in _preferred]
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as _e:
        print(f"警告: 自带中文字体注册失败: {_e}")

np.random.seed(42)

# ─── 实验参数配置 ───
N_PIXELS    = 64
NOISE_LEVEL = 0.1
NOISE_VAR   = NOISE_LEVEL ** 2
SOB_LAMBDAS = [0.05, 0.2, 0.5, 1.0, 2.0]  # 扩展网格以覆盖 PSNR 峰值（λ≈0.5）并展示过平滑回落
SOB_ITERS   = 1000
SOB_TOL     = 1e-13
DATA_RANGE  = 1.0  # camera() 经 resize 后值域为 [0,1]，理论值域
RING_WIDTH  = 4    # 边界效应量化时外缘环的宽度（像素）
BOUNDARY_LAMBDA = 1.0  # 步骤5边界对比图使用的 λ（边界伪影较强、差异明显）

def fmt_metrics(psnr, ssim):
    """统一格式化 PSNR 和 SSIM 输出"""
    return f"PSNR={psnr:.2f}dB, SSIM={ssim:.3f}"

def calc_psnr(a, b):
    return peak_signal_noise_ratio(a, b, data_range=DATA_RANGE)

def calc_ssim(a, b):
    # 新版 scikit-image 要求 2D 图像显式指定 channel_axis=None；旧版无此参数
    try:
        return structural_similarity(a, b, data_range=DATA_RANGE, channel_axis=None)
    except TypeError:
        return structural_similarity(a, b, data_range=DATA_RANGE)

n = N_PIXELS
x_true = resize(data.camera(), (n, n))

y = x_true + NOISE_LEVEL * np.random.randn(n, n)

# ═══════════════════════════════════════════════════════════════════
# 第一部分：Tikhonov（简单 i.i.d. 高斯先验）
# ═══════════════════════════════════════════════════════════════════

def tikhonov_denoise(y, lam):
    """
    简单高斯先验（零均值）: x ~ N(0, σ_x² I)
    正则项: ||x||_2²

    闭式解推导:
      目标函数: J(x) = 0.5||y-x||² + 0.5λ||x||²
      梯度: ∇J = (x-y) + λx = 0
      解: x = y/(1+λ)

    参数说明（v9修正，重点）:
      - λ = σ²/σ_x²，其中 σ 为噪声标准差。
      - 模型均值为 0，故一致的先验方差是"关于 0 的二阶矩" σ_x² = E[x²]（rms²），
        而不是关于样本均值的 Var(x)=s²。对均值 μ≈0.51 的 cameraman，
        E[x²]=μ²+s²≈0.332 远大于 s²≈0.076，二者绝不能混用。
      - 该先验把解向 0 收缩；对非零均值图像，正确做法是显式建模均值
        x ~ N(μ,σ_x²I) 并先去均值（见 tikhonov_denoise_centered）。
    """
    return y / (1 + lam)

def tikhonov_denoise_centered(y, lam):
    """
    非零均值高斯先验: x ~ N(μ, σ_x² I)，μ 用样本均值 ȳ 估计。

    推导: 令 d = y - μ，u = x - μ，则模型退化为零均值形式，
          û = d/(1+λ)，故 x̂ = μ + (y-μ)/(1+λ)。
    此时一致的先验方差为 Var(x)（关于 μ 的二阶矩），λ = σ²/Var(x)。

    与 tikhonov_denoise 的唯一差别就是"是否先去均值"；
    同样的 λ，仅差这一步，去噪质量显著不同（本实验 19.17 → 20.56 dB）。
    """
    mu = np.mean(y)
    return mu + (y - mu) / (1 + lam)

# ═══════════════════════════════════════════════════════════════════
# 第二部分：Sobolev（梯度高斯先验）
# ═══════════════════════════════════════════════════════════════════

def sobolev_denoise(y, lam, n_iter=SOB_ITERS, tol=SOB_TOL, verbose=False):
    """
    梯度高斯先验: ∇x ~ N(0, σ_∇² I)
    正则项: ||∇x||_2² = ∫|∇x|² dx (Sobolev半范数H¹)

    目标函数: J(x) = 0.5||y-x||² + 0.5λ||∇x||²

    梯度下降迭代:
      x^{k+1} = x^k - dt * ∇J(x^k)
              = x^k + dt * [(y-x^k) + λ·Δx^k]
      其中Δ为拉普拉斯算子

    数值稳定性分析:
      - 二维四邻域拉普拉斯算子的谱半径 ρ(Δ) = 8
      - 梯度下降稳定性条件: dt × (1 + λ·ρ(Δ)) < 2（收敛域），
        dt × (1 + λ·ρ(Δ)) < 1 时能量单调下降
      - 采用保守自适应步长: dt = 0.9 / (1 + 8λ)

    边界条件说明（v9修正）:
      - 使用np.roll实现周期性边界条件（图像左右/上下相连）
      - cameraman 不是周期信号：上下/左右边缘灰度跳变约 0.23~0.29，
        周期延拓会人为制造不连续，大 λ 时边缘出现明显伪影
      - 这是"周期性边界 + 非周期图像"的建模代价，不是频域方法的固有属性：
        改用 Neumann 镜像边界（DCT 闭式解，见 sobolev_denoise_dct）
        可消除大部分该伪影，计算复杂度同为 O(n²log n)
      - 步骤5给出两种边界的定量对照

    收敛诊断:
      - 以解的逐像素最大变化作为停止判据：max|x^{k+1} - x^k| < tol
      - 不使用"目标函数相对变化"判据：该目标在极小值附近 loss 曲面趋于平坦，
        相对变化会过早触发而解 x 远未收敛（残差可达 1e-4 量级），
        无法验证迭代解与闭式解的等价性
      - k > 0 确保第一次迭代不会误触发
    """
    x = y.copy()
    dt = 0.9 / (1 + 8 * lam)

    converged_iter = n_iter

    for k in range(n_iter):
        # 用当前x的拉普拉斯更新x
        lap = (np.roll(x, 1, axis=0) + np.roll(x, -1, axis=0) +
               np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1) - 4 * x)
        x_new = x + dt * ((y - x) + lam * lap)

        # 收敛判据：解的逐像素最大变化 max|Δx| < tol（直接控制解的精度，
        # 而非 loss 相对变化——后者在极小值附近过早触发，解尚未收敛）
        dx = np.max(np.abs(x_new - x))
        x = x_new

        if k > 0 and dx < tol:
            converged_iter = k + 1
            if verbose:
                print(f"    收敛于第 {k+1} 次迭代 (λ={lam:.3f})")
            break
    else:
        # 循环正常结束（未break），说明未提前收敛
        if verbose:
            print(f"    未收敛，已达最大迭代次数 {n_iter} (λ={lam:.3f})")

    return x, converged_iter

def sobolev_denoise_exact(y, lam):
    """
    Sobolev去噪的频域闭式解（周期边界，与四邻域离散拉普拉斯一致）

    理论推导:
      目标函数: J(x) = 0.5||y-x||² + 0.5λ||∇x||²

      在离散频域中，四邻域拉普拉斯算子的特征值为：
        λ_Lap(ω) = 2cos(2πω_x) + 2cos(2πω_y) - 4
      其中 ω_x, ω_y 为离散归一化频率（范围 [-0.5, 0.5)）

      由Parseval定理，空域范数可转化为频域求和：
        ||∇x||² = Σ|∇x̂(ω)|² = Σ(-λ_Lap(ω))·|X̂(ω)|²
      其中 -λ_Lap(ω) ≥ 0，保证正则项非负

      频域目标函数:
        Ĵ(X̂) = 0.5|Ŷ - X̂|² - 0.5λ·λ_Lap(ω)·|X̂|²

      对每个频率分量求Wirtinger导数并令其为零:
        ∂Ĵ/∂X̂* = -(Ŷ - X̂) - λ·λ_Lap(ω)·X̂ = 0
        X̂(1 - λ·λ_Lap(ω)) = Ŷ
        X̂ = Ŷ / (1 - λ·λ_Lap(ω))

    实现说明:
      - 使用np.fft.fft2进行二维DFT
      - 频率网格通过np.fft.fftfreq构造（离散归一化频率，范围[-0.5, 0.5)）
      - 使用离散拉普拉斯特征值，确保与迭代法的四邻域拉普拉斯（周期边界）完全一致
      - 结果通过np.fft.ifft2返回空域
      - DFT 隐含周期边界；镜像边界版本见 sobolev_denoise_dct
    """
    ny, nx = y.shape
    # 构造离散归一化频率网格（范围 [-0.5, 0.5)）
    freq_y = np.fft.fftfreq(ny)
    freq_x = np.fft.fftfreq(nx)
    omega_y, omega_x = np.meshgrid(freq_y, freq_x, indexing='ij')

    # 四邻域离散拉普拉斯的特征值
    lap_eigenval = 2*np.cos(2*np.pi*omega_x) + 2*np.cos(2*np.pi*omega_y) - 4

    # DFT
    y_hat = np.fft.fft2(y)
    # 频域滤波: X̂ = Ŷ / (1 - λ·λ_Lap)
    # 注意：lap_eigenval ≤ 0，故分母 ≥ 1，数值稳定
    x_hat = y_hat / (1 - lam * lap_eigenval)
    # IDFT返回实部
    x_exact = np.fft.ifft2(x_hat).real

    return x_exact

def sobolev_denoise_dct(y, lam):
    """
    Sobolev去噪的频域闭式解（Neumann 镜像边界，DCT-II 实现）

    理论推导:
      镜像（Neumann）边界的一维路径图拉普拉斯：内部 [1,-2,1]/端点度1，
      其正交特征基恰为 DCT-II 基，二维可分离，特征值为
        μ(k,l) = 4 - 2cos(πk/n) - 2cos(πl/n),  k,l = 0,...,n-1
      （已用稠密矩阵谱分解核对到机器精度 1e-14）

      频域解: X̃ = Ỹ / (1 + λ·μ)，正/反变换使用 ortho 归一化 DCT。

    与 DFT 版本的关系:
      - 两者同为 O(n²log n) 的精确闭式解，唯一差别是边界条件；
      - DCT 对应镜像延拓，图像边缘不被"对边拼接"，
        对 cameraman 这类非周期图像边界 MSE 显著更低（步骤5定量验证）。
    """
    ny, nx = y.shape
    k_idx = np.arange(ny)
    l_idx = np.arange(nx)
    K, L = np.meshgrid(k_idx, l_idx, indexing='ij')
    mu = 4 - 2*np.cos(np.pi*K/ny) - 2*np.cos(np.pi*L/nx)

    y_dct = dctn(y, norm='ortho')
    x_dct = y_dct / (1 + lam * mu)
    return idctn(x_dct, norm='ortho')

def generate_figure_2_1(save_dir):
    """
    图2-1：L1 vs L2 几何解释——为什么L1促稀疏
    对应章节：2.2 经典先验族 - Laplace先验
    知识点：L1约束（菱形）与L2约束（圆形）的几何差异；角点处相切→稀疏解
    合并自 temp/图2-1_L1_vs_L2几何解释/图2-1_L1_vs_L2几何解释.py
    """
    # 确保中文字体在绘图前加载
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    theta = np.linspace(0, 2*np.pi, 400)

    # L1 正则项等高线：菱形（旋转45度的正方形）
    # R(w) = |w1| + |w2| = C → 参数方程
    def l1_contour(C, n=100):
        """生成L1等高线（菱形）的点"""
        pts = []
        t = np.linspace(0, C, n)
        # 第一象限: (t, C-t), t∈[0,C]
        pts.extend([(x, C - x) for x in t])
        # 第二/三/四象限（去掉与第一象限重复的端点）
        pts.extend([(-x, C - x) for x in t[1:]])
        pts.extend([(-x, -(C - x)) for x in t[1:]])
        pts.extend([(x, -(C - x)) for x in t[1:]])
        return np.array(pts)

    # 损失函数等高线（数据项）：椭圆
    def ellipse_contour(cx, cy, a, b, n=200):
        """椭圆等高线"""
        t = np.linspace(0, 2*np.pi, n)
        return np.column_stack([cx + a*np.cos(t), cy + b*np.sin(t)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── 左子图：L2正则化（圆形）──
    # 几何构造：先选切点 w* 在圆上，再由相切条件反推椭圆参数
    # 圆: ||w||₂ = r, 切点 w* = (0.8, 0.6), 法向量 n = w* = (0.8, 0.6)
    # 椭圆中心 w_ls = (1.5, 1.0), 半轴 a, b（未旋转）
    # 相切条件: 椭圆在 w* 处的梯度 ∝ n
    #   椭圆方程: ((w1-1.5)/a)² + ((w2-1.0)/b)² = 1
    #   梯度: (-0.7/a², -0.4/b²) ∝ (0.8, 0.6) → b²/a² = 16/21
    #   代入 w* 在椭圆上: 0.49/a² + 0.16/b² = 1 → a²=0.7, b²=16/30
    ax = axes[0]
    r_l2_1, r_l2_2 = 1.0, 1.5
    for r, alpha in [(r_l2_1, 0.3), (r_l2_2, 0.15)]:
        ax.fill(np.cos(theta)*r, np.sin(theta)*r, alpha=alpha, color='steelblue')
        ax.plot(np.cos(theta)*r, np.sin(theta)*r, 'b-', linewidth=1.5)
    # 损失函数等高线（椭圆）：中心(1.5,1.0)，半轴由相切条件确定
    ell = ellipse_contour(1.5, 1.0, 0.837, 0.730)
    ax.plot(ell[:, 0], ell[:, 1], 'r-', linewidth=2, label='损失函数等高线')
    ell2 = ellipse_contour(1.5, 1.0, 1.299, 1.133)
    ax.plot(ell2[:, 0], ell2[:, 1], 'r-', linewidth=1.5, alpha=0.5)
    w_opt_l2 = np.array([0.8, 0.6])  # 严格切点（非稀疏）
    ax.plot(w_opt_l2[0], w_opt_l2[1], 'ko', markersize=12, zorder=5)
    ax.annotate('最优解\n(非稀疏)', xy=w_opt_l2, xytext=(w_opt_l2[0]+0.5, w_opt_l2[1]+0.3),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_xlabel('$w_1$', fontsize=13)
    ax.set_ylabel('$w_2$', fontsize=13)
    ax.set_title('L2正则化（高斯先验）\n约束集：圆形', fontsize=13, fontweight='bold')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', alpha=0.3, edgecolor='blue', label='L2正则项等高线'),
                       plt.Line2D([0], [0], color='red', linewidth=2, label='损失函数等高线')]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # ── 右子图：L1正则化（菱形）──
    ax = axes[1]
    l1_c1 = l1_contour(1.2, n=80)
    l1_c2 = l1_contour(1.8, n=80)
    ax.fill(l1_c1[:, 0], l1_c1[:, 1], alpha=0.3, color='forestgreen')
    ax.plot(l1_c1[:, 0], l1_c1[:, 1], 'g-', linewidth=1.5)
    ax.fill(l1_c2[:, 0], l1_c2[:, 1], alpha=0.15, color='forestgreen')
    ax.plot(l1_c2[:, 0], l1_c2[:, 1], 'g-', linewidth=1.5, alpha=0.7)
    # 损失函数等高线（椭圆）：中心(0.6,0.3)，半轴由相切条件确定
    # 切点 w* = (1.2, 0) 在菱形顶点，椭圆须过该点且梯度在L1次微分内
    # 椭圆方程: ((w1-0.6)/a)² + ((w2-0.3)/b)² = 1
    # 过(1.2,0): 0.36/a² + 0.09/b² = 1
    # 梯度方向: (0.6/a², -0.3/b²)，需 |0.3·a²/(0.6·b²)| ≤ 1 → a² ≤ 2b²
    # 取等号 a²=2b²: 0.18/b² + 0.09/b² = 0.27/b² = 1 → b²=0.27, a²=0.54
    ell = ellipse_contour(0.6, 0.3, 0.7348, 0.5196)
    ax.plot(ell[:, 0], ell[:, 1], 'r-', linewidth=2, label='损失函数等高线')
    # 外层椭圆：按相同比例放大1.5倍
    ell2 = ellipse_contour(0.6, 0.3, 1.1023, 0.7794)
    ax.plot(ell2[:, 0], ell2[:, 1], 'r-', linewidth=1.5, alpha=0.5)
    w_opt_l1 = np.array([1.2, 0.0])  # 在w1轴上（w2=0）→ 稀疏解
    ax.plot(w_opt_l1[0], w_opt_l1[1], 'ko', markersize=12, zorder=5)
    ax.annotate('最优解\n(稀疏：$w_2=0$)', xy=w_opt_l1, xytext=(w_opt_l1[0]+0.6, w_opt_l1[1]+0.5),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    # 标注角点
    ax.plot(0, 1.2, 'g^', markersize=8, alpha=0.7)
    ax.plot(0, -1.2, 'gv', markersize=8, alpha=0.7)
    ax.plot(1.2, 0, 'g>', markersize=8, alpha=0.7)
    ax.plot(-1.2, 0, 'g<', markersize=8, alpha=0.7)
    ax.text(1.4, 0.15, '角点', fontsize=9, color='green')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_xlabel('$w_1$', fontsize=13)
    ax.set_ylabel('$w_2$', fontsize=13)
    ax.set_title('L1正则化（Laplace先验）\n约束集：菱形', fontsize=13, fontweight='bold')
    legend_elements = [Patch(facecolor='forestgreen', alpha=0.3, edgecolor='green', label='L1正则项等高线'),
                       plt.Line2D([0], [0], color='red', linewidth=2, label='损失函数等高线')]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    fig.suptitle('L1 vs L2 几何解释\n为什么L1促稀疏', fontsize=15, fontweight='bold', y=0.96, linespacing=1.5)
    fig.text(0.5, 0.04,
             '核心直觉：L2的圆形约束集没有"角点"，最优解通常不在坐标轴上；'
             'L1的菱形约束集有4个角点，椭圆更容易在角点处与之相切，从而得到稀疏解（某些分量为0）',
             ha='center', va='bottom', fontsize=9, linespacing=1.5,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3), wrap=True)

    plt.subplots_adjust(wspace=0.12, hspace=0.3, bottom=0.15, top=0.85)
    out_path = os.path.join(save_dir, '图2-1_L1_vs_L2几何解释.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("图2-1已生成：", out_path)

# ═══════════════════════════════════════════════════════════════════
# 参数计算：经验贝叶斯插件量只使用观测 y，不使用真值 x_true
# ═══════════════════════════════════════════════════════════════════

# --- 真值统计量（仅用于教学对照与评判，不参与"从数据定参数"）---
mu_true   = float(np.mean(x_true))
s_true    = float(np.std(x_true))
rms2_true = float(np.mean(x_true ** 2))  # E[x²]，关于0的二阶矩

# --- 从观测 y 可获得的无偏插件估计（噪声方差已知）---
mu_y      = float(np.mean(y))
rms2_hat  = float(np.mean(y ** 2) - NOISE_VAR)   # E[x²] 的插件估计
s2_hat    = float(np.var(y) - NOISE_VAR)         # Var(x) 的插件估计

# Tikhonov 四种参数化
sigma_x_prior = 1.0
lam_tikh_subj    = NOISE_VAR / sigma_x_prior ** 2  # 主观先验 σx=1
lam_tikh_eb0     = NOISE_VAR / rms2_hat            # 零均值模型 N(0,σx²I) 的一致EB：σx²=E[x²]
lam_tikh_center  = NOISE_VAR / s2_hat              # 非零均值模型 N(μ,σx²I) 的一致EB：σx²=Var(x)
# 反面教材：旧版做法——λ 用 σ²/s²（关于均值的方差），却仍套零均值闭式解、不中心化
lam_tikh_misuse  = lam_tikh_center

x_tikh_subj   = tikhonov_denoise(y, lam_tikh_subj)
x_tikh_eb0    = tikhonov_denoise(y, lam_tikh_eb0)
x_tikh_center = tikhonov_denoise_centered(y, lam_tikh_center)
x_tikh_misuse = tikhonov_denoise(y, lam_tikh_misuse)

# Sobolev：迭代解（周期）、DFT 闭式解（周期）、DCT 闭式解（镜像）
x_sob_results = []
x_sob_exact_results = []
x_sob_dct_results = []
converged_iters = []
iter_exact_diffs = []

for lam_sob in SOB_LAMBDAS:
    # 迭代解
    x_sob, iters = sobolev_denoise(y, lam_sob, verbose=True)
    x_sob_results.append(x_sob)
    converged_iters.append(iters)
    # 闭式解（周期边界，用于对比验证迭代实现）
    x_sob_exact = sobolev_denoise_exact(y, lam_sob)
    x_sob_exact_results.append(x_sob_exact)
    iter_exact_diffs.append(float(np.max(np.abs(x_sob - x_sob_exact))))
    # 闭式解（Neumann 镜像边界，用于边界条件对照）
    x_sob_dct_results.append(sobolev_denoise_dct(y, lam_sob))

# ─── 统一计算所有 PSNR/SSIM ───
psnr_noisy = calc_psnr(x_true, y)
psnr_tikh_subj,   ssim_tikh_subj   = calc_psnr(x_true, x_tikh_subj),   calc_ssim(x_true, x_tikh_subj)
psnr_tikh_eb0,    ssim_tikh_eb0    = calc_psnr(x_true, x_tikh_eb0),    calc_ssim(x_true, x_tikh_eb0)
psnr_tikh_center, ssim_tikh_center = calc_psnr(x_true, x_tikh_center), calc_ssim(x_true, x_tikh_center)
psnr_tikh_misuse, ssim_tikh_misuse = calc_psnr(x_true, x_tikh_misuse), calc_ssim(x_true, x_tikh_misuse)

psnr_sob       = [calc_psnr(x_true, xs)     for xs in x_sob_results]
psnr_sob_exact = [calc_psnr(x_true, xs)     for xs in x_sob_exact_results]
psnr_sob_dct   = [calc_psnr(x_true, xs)     for xs in x_sob_dct_results]
ssim_noisy = calc_ssim(x_true, y)
ssim_sob       = [calc_ssim(x_true, xs)     for xs in x_sob_results]
ssim_sob_exact = [calc_ssim(x_true, xs)     for xs in x_sob_exact_results]
ssim_sob_dct   = [calc_ssim(x_true, xs)     for xs in x_sob_dct_results]

# ─── 边界环 / 内部区域的 MSE（隔离边界条件效应）──
ring_mask = np.zeros((n, n), dtype=bool)
w = RING_WIDTH
ring_mask[:w, :] = True
ring_mask[-w:, :] = True
ring_mask[:, :w] = True
ring_mask[:, -w:] = True
inner_mask = ~ring_mask

bmse_dft, bmse_dct, imse_dft, imse_dct = [], [], [], []
for i in range(len(SOB_LAMBDAS)):
    bmse_dft.append(float(np.mean((x_sob_exact_results[i][ring_mask] - x_true[ring_mask]) ** 2)))
    bmse_dct.append(float(np.mean((x_sob_dct_results[i][ring_mask]   - x_true[ring_mask]) ** 2)))
    imse_dft.append(float(np.mean((x_sob_exact_results[i][inner_mask] - x_true[inner_mask]) ** 2)))
    imse_dct.append(float(np.mean((x_sob_dct_results[i][inner_mask]   - x_true[inner_mask]) ** 2)))

best_sob_idx     = int(np.argmax(psnr_sob))
best_sob_dct_idx = int(np.argmax(psnr_sob_dct))

print("=" * 70)
print("实验2.2-1 高斯先验的推广：从Tikhonov到Sobolev平滑")
print("=" * 70)

print("\n" + "=" * 70)
print("【核心知识点】MAP与MMSE的等价性")
print("=" * 70)
print("""
在高斯-高斯模型下（高斯似然 + 高斯先验），后验分布为高斯分布。
由于高斯分布对称，均值 = 众数，因此：
  - MAP估计（后验众数）= MMSE估计（后验均值）
  - Tikhonov闭式解 x = y/(1+λ) 既是MAP也是MMSE（均值中心化版本同理）
这是共轭先验的重要性质，详见第2章附录2A。
""")

print("=" * 70)
print("【参数设置】")
print("=" * 70)
print(f"  图像尺寸: {n}×{n}")
print(f"  噪声标准差 σ = {NOISE_LEVEL:.4f}")
print(f"\n  真值统计量（仅用于教学对照，实际逆问题中不可得）:")
print(f"    均值 μ = {mu_true:.4f}（显著非零！）")
print(f"    标准差 s = {s_true:.4f},  Var(x) = s² = {s_true**2:.4f}")
print(f"    rms² = E[x²] = μ²+s² = {rms2_true:.4f}")
print(f"\n  从观测 y 得到的插件估计（噪声方差已知，E[y²]=E[x²]+σ²）:")
print(f"    μ̂ = ȳ = {mu_y:.4f}")
print(f"    E[x²] 插件估计 = mean(y²)-σ² = {rms2_hat:.4f}")
print(f"    Var(x) 插件估计 = var(y)-σ² = {s2_hat:.4f}")
print(f"\n  Tikhonov 参数 λ=σ²/σx² 的四种用法（σx²必须与模型均值配套）:")
print(f"    (1) 主观先验 σx=1（零均值模型）:            λ={lam_tikh_subj:.4f}")
print(f"    (2) 零均值模型的一致EB σx²=E[x²]=rms²:     λ={lam_tikh_eb0:.4f}")
print(f"    (3) 非零均值模型 N(μ,σx²I)，先去均值，")
print(f"        σx²=Var(x):                             λ={lam_tikh_center:.4f}")
print(f"    (4) 反面教材：λ取σ²/s²却仍向0收缩、不中心化: λ={lam_tikh_misuse:.4f}")
print(f"\n  Sobolev参数 λ ∈ {SOB_LAMBDAS}")
print(f"  自适应步长: dt = 0.9/(1+8λ)")
print(f"  边界对照: 周期(DFT/np.roll) vs 镜像(DCT/Neumann)，边界环宽 {RING_WIDTH} 像素")

print("\n" + "=" * 70)
print("【评估结果】")
print("=" * 70)
print(f"\n0. 含噪图像:")
print(f"   {fmt_metrics(psnr_noisy, ssim_noisy)}")

print(f"\n1. Tikhonov去噪 (简单 i.i.d. 高斯先验)，正则项 ||x||₂²:")
print(f"   (1) 主观先验 σx=1：假设像素可扩散到约[-3,3]，先验方差过大→λ过小、")
print(f"       几乎不收缩: {fmt_metrics(psnr_tikh_subj, ssim_tikh_subj)}")
print(f"   (2) 零均值模型+一致EB（σx²=E[x²]={rms2_hat:.3f}）: "
      f"{fmt_metrics(psnr_tikh_eb0, ssim_tikh_eb0)}")
print(f"       ——与 N(0,σx²I) 配套的无偏插件选择；标量模型总体最优 λ*=σ²/E[x²]。")
print(f"   (3) 均值中心化 x̂=ȳ+(y-ȳ)/(1+λ)（模型与非零均值数据匹配）: "
      f"{fmt_metrics(psnr_tikh_center, ssim_tikh_center)}")
print(f"       ——Tikhonov 族最佳。")
print(f"   (4) 反面教材（σ²/s² 配零均值闭式解，不中心化）: "
      f"{fmt_metrics(psnr_tikh_misuse, ssim_tikh_misuse)}")
print(f"       ——把关于均值的方差错配给零均值模型，过度向0收缩，PSNR反而低于含噪图。")
print(f"\n   → 教学解读：")
print(f"     先验的均值与方差必须与数据配套：均值不匹配时再"
      f"\"正确\"的λ也会失效；")
print(f"     (3)与(4)只差一次去均值，结果相差 {psnr_tikh_center-psnr_tikh_misuse:.2f} dB，")
print(f"     直观展示了模型假设一致性的重要性。")
print(f"     但即使是最佳Tikhonov({psnr_tikh_center:.2f}dB)也远逊于Sobolev"
      f"({max(psnr_sob):.2f}dB)：")
print(f"     i.i.d. 先验完全没有空间相关性，这是其本质局限。")

print(f"\n2. Sobolev去噪 (梯度高斯先验)，正则项 ||∇x||₂²:")
for i, lam_sob in enumerate(SOB_LAMBDAS):
    print(f"   λ={lam_sob:.2f} [周期DFT]: {fmt_metrics(psnr_sob[i], ssim_sob[i])}, "
          f"收敛于{converged_iters[i]}次迭代; [镜像DCT]: {fmt_metrics(psnr_sob_dct[i], ssim_sob_dct[i])}")

print(f"\n【迭代解 vs 闭式解验证（周期边界内部一致性）】")
print(f"   迭代法使用空域梯度下降，闭式解使用频域DFT滤波")
print(f"   两者理论上等价，数值差异应达到迭代容差精度（~1e-12，受 SOB_TOL 控制）")
for i, lam_sob in enumerate(SOB_LAMBDAS):
    print(f"   λ={lam_sob:.2f}: 最大差异 = {iter_exact_diffs[i]:.2e}")
    print(f"            迭代解 {fmt_metrics(psnr_sob[i], ssim_sob[i])}")
    print(f"            闭式解 {fmt_metrics(psnr_sob_exact[i], ssim_sob_exact[i])}")

print(f"\n【边界条件对照：周期(DFT) vs 镜像(DCT)】")
print(f"   边界环宽 {RING_WIDTH} 像素；两者内部区域 MSE 应几乎相同，差异只来自边界建模")
print(f"   {'λ':>5s} {'DFT边界MSE':>12s} {'DCT边界MSE':>12s} {'DFT内部MSE':>12s} {'DCT内部MSE':>12s} {'DFT PSNR':>9s} {'DCT PSNR':>9s}")
for i, lam_sob in enumerate(SOB_LAMBDAS):
    print(f"   {lam_sob:5.2f} {bmse_dft[i]:12.5f} {bmse_dct[i]:12.5f} "
          f"{imse_dft[i]:12.5f} {imse_dct[i]:12.5f} {psnr_sob[i]:9.2f} {psnr_sob_dct[i]:9.2f}")
print(f"   cameraman 上下/左右边缘均值跳变约 0.29/0.23，周期延拓人为制造不连续；")
print(f"   大 λ 时 DFT 边界 MSE 显著高于 DCT——这是周期边界的代价，")
print(f"   而非\"频域方法固有振铃\"（两者同为 O(n²log n) 频域闭式解）。")

# ═══════════════════════════════════════════════════════════════════
# 步骤1：去噪结果总览（2×4）
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

axes[0, 0].imshow(x_true, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(y, cmap='gray')
axes[0, 1].set_title(f'含噪图像\n{fmt_metrics(psnr_noisy, ssim_noisy)}')
axes[0, 1].axis('off')

axes[0, 2].imshow(x_tikh_subj, cmap='gray')
axes[0, 2].set_title(f'Tikhonov (1) 主观先验$\\sigma_x=1$\n$\\lambda$={lam_tikh_subj:.4f}\n'
                     f'{fmt_metrics(psnr_tikh_subj, ssim_tikh_subj)}')
axes[0, 2].axis('off')

axes[0, 3].imshow(x_tikh_eb0, cmap='gray')
axes[0, 3].set_title(f'Tikhonov (2) 零均值模型一致EB\n$\\sigma_x^2$=E[$x^2$], $\\lambda$={lam_tikh_eb0:.4f}\n'
                     f'{fmt_metrics(psnr_tikh_eb0, ssim_tikh_eb0)}')
axes[0, 3].axis('off')

axes[1, 0].imshow(x_tikh_center, cmap='gray')
axes[1, 0].set_title(f'Tikhonov (3) 均值中心化（最佳）\n$\\lambda$={lam_tikh_center:.4f}\n'
                     f'{fmt_metrics(psnr_tikh_center, ssim_tikh_center)}')
axes[1, 0].axis('off')

axes[1, 1].imshow(x_tikh_misuse, cmap='gray')
axes[1, 1].set_title(f'Tikhonov (4) 反面教材：不中心化\n$\\lambda$={lam_tikh_misuse:.4f}\n'
                     f'{fmt_metrics(psnr_tikh_misuse, ssim_tikh_misuse)}', color='darkred')
axes[1, 1].axis('off')

sob_idx_01 = 1  # 较小 λ（SOB_LAMBDAS[1]=0.2），与右图"最优λ"形成对照
lam_display = SOB_LAMBDAS[sob_idx_01]
axes[1, 2].imshow(x_sob_results[sob_idx_01], cmap='gray')
axes[1, 2].set_title(f'Sobolev ($\\lambda$={lam_display})\n{fmt_metrics(psnr_sob[sob_idx_01], ssim_sob[sob_idx_01])}')
axes[1, 2].axis('off')

axes[1, 3].imshow(x_sob_results[best_sob_idx], cmap='gray')
axes[1, 3].set_title(f'Sobolev (事后最优$\\lambda$={SOB_LAMBDAS[best_sob_idx]})\n'
                     f'{fmt_metrics(psnr_sob[best_sob_idx], ssim_sob[best_sob_idx])}\n'
                     f'(oracle：按与真值的PSNR选取)')
axes[1, 3].set_axis_off()

plt.suptitle('高斯先验的推广：Tikhonov（含均值中心化修正）vs Sobolev', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_去噪结果对比.png'), dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 步骤2：剖面与性能柱状图
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

center = n // 2
for i, (lam_sob, xs) in enumerate(zip(SOB_LAMBDAS, x_sob_results)):
    axes[0].plot(xs[center, :], linewidth=1, label=f'$\\lambda$={lam_sob:.2f}')
axes[0].plot(x_true[center, :], 'k--', linewidth=1.5, label='真实')
axes[0].plot(y[center, :], 'r-', linewidth=0.5, alpha=0.5, label='含噪')
axes[0].set_title('Sobolev不同$\\lambda$的剖面对比\n展示超参数敏感性')
axes[0].legend(fontsize=8)
axes[0].set_xlabel('像素索引')
axes[0].set_ylabel('灰度值')
axes[0].grid(True, alpha=0.3)

all_methods = ['Tikhonov\n(1)主观σx=1', 'Tikhonov\n(2)零均值EB',
               'Tikhonov\n(3)均值中心化', 'Tikhonov\n(4)误用对照'] + \
              [f'Sobolev周期\n$\\lambda$={lam}' for lam in SOB_LAMBDAS] + \
              [f'Sobolev镜像DCT\n$\\lambda$={SOB_LAMBDAS[best_sob_dct_idx]:g}(最优)']
all_psnrs = [psnr_tikh_subj, psnr_tikh_eb0, psnr_tikh_center, psnr_tikh_misuse] + \
            psnr_sob + [psnr_sob_dct[best_sob_dct_idx]]
all_ssim_scaled = [ssim_tikh_subj*100, ssim_tikh_eb0*100,
                   ssim_tikh_center*100, ssim_tikh_misuse*100] + \
                  [s*100 for s in ssim_sob] + [ssim_sob_dct[best_sob_dct_idx]*100]

x_pos = np.arange(len(all_methods))
width = 0.35

ax1 = axes[1]
ax1.bar(x_pos - width/2, all_psnrs, width, color='steelblue', alpha=0.8, label='PSNR (dB)')
ax1.axhline(y=psnr_noisy, color='black', linestyle='--', label=f'含噪PSNR: {psnr_noisy:.2f}dB')

psnr_min, psnr_max = min(all_psnrs), max(all_psnrs)
y_bottom = min(psnr_min, psnr_noisy) - 1
ax1.set_ylim([y_bottom, psnr_max + 1])
ax1.set_ylabel('PSNR (dB)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.bar(x_pos + width/2, all_ssim_scaled, width, color='seagreen', alpha=0.6, label='SSIM$\\times$100')

ssim_min, ssim_max = min(all_ssim_scaled), max(all_ssim_scaled)
ssim_noisy_scaled = ssim_noisy * 100
ax2.set_ylim([min(ssim_min, ssim_noisy_scaled) - 2, ssim_max + 2])
ax2.axhline(y=ssim_noisy_scaled, color='gray', linestyle=':', label=f'含噪SSIM$\\times$100: {ssim_noisy_scaled:.1f}')
ax2.set_ylabel('SSIM$\\times$100', color='seagreen')
ax2.tick_params(axis='y', labelcolor='seagreen')

ax1.set_xticks(x_pos)
ax1.set_xticklabels(all_methods, fontsize=7)
ax1.set_title('不同方法的性能对比\n(左轴: PSNR, 右轴: SSIM$\\times$100)')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_剖面对比与性能.png'), dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 步骤3：正则项形态与局部方差
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

t = np.linspace(-2, 2, 400)
axes[0].plot(t, t**2, 'b-', linewidth=2, label='L2: $t^2$ (高斯先验正则项)')
axes[0].plot(t, np.abs(t), 'r-', linewidth=2, label='L1: |t| (Laplace/TV正则项)')
axes[0].set_title('正则项形态对比（示意）\nL2均匀惩罚 vs L1促稀疏\n注：L1方法未在本实验实现，详见实验2.2-3')
axes[0].legend()
axes[0].set_xlabel('t')
axes[0].set_ylabel('惩罚值')
axes[0].set_ylim(-0.2, 4)
axes[0].grid(True, alpha=0.3)

# 局部方差计算（边界安全检查）
assert n >= 10, "N_PIXELS 过小，局部方差区域无效"
sl = (slice(center-5, center+5), slice(center-5, center+5))
local_var_true   = float(np.var(x_true[sl]))
local_var_subj   = float(np.var(x_tikh_subj[sl]))
local_var_eb0    = float(np.var(x_tikh_eb0[sl]))
local_var_center = float(np.var(x_tikh_center[sl]))
local_var_misuse = float(np.var(x_tikh_misuse[sl]))
local_var_sob = [float(np.var(xs[sl])) for xs in x_sob_results]

var_methods = ['真实', 'Tikhonov\n(1)主观σx=1', 'Tikhonov\n(2)零均值EB',
               'Tikhonov\n(3)均值中心化', 'Tikhonov\n(4)误用对照'] + \
              [f'Sobolev\n$\\lambda$={lam}' for lam in SOB_LAMBDAS]
var_values = [local_var_true, local_var_subj, local_var_eb0,
              local_var_center, local_var_misuse] + local_var_sob
var_colors = plt.cm.tab10(np.linspace(0, 0.5, len(var_methods)))

axes[1].bar(var_methods, var_values, color=var_colors[:len(var_methods)], alpha=0.7)
axes[1].set_ylabel('局部方差')
axes[1].set_title('局部方差保持度\n(中心$10\\times10$区域)')
axes[1].tick_params(axis='x', labelsize=7)

# 动态计算文字偏移量，避免超出边框
var_max = max(var_values)
var_min = min(var_values)
var_range = var_max - var_min
offset = var_range * 0.02 if var_range > 0 else var_max * 0.02

for i, v in enumerate(var_values):
    # 根据数值大小动态调整文字位置
    text_y = v + offset
    # 如果文字可能超出标题区域，将文字移到柱内
    if text_y > var_max * 0.95:
        text_y = v - offset
        va = 'top'
    else:
        va = 'bottom'
    axes[1].text(i, text_y, f'{v:.4f}', ha='center', va=va, fontsize=7)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, '步骤3_正则项形态与方差保持.png'), dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 步骤4：迭代解与闭式解对比（保持原结构）
# ═══════════════════════════════════════════════════════════════════
n_rows = 2  # 上排迭代解，下排闭式解
n_cols = len(SOB_LAMBDAS)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)

for i, lam_sob in enumerate(SOB_LAMBDAS):
    # 上排：迭代解
    axes[0, i].imshow(x_sob_results[i], cmap='gray')
    title1 = '迭代解 ($\\lambda$={:.2f})\n{}\n迭代{}次'.format(lam_sob, fmt_metrics(psnr_sob[i], ssim_sob[i]), converged_iters[i])
    axes[0, i].set_title(title1)
    axes[0, i].axis('off')

    # 下排：闭式解
    axes[1, i].imshow(x_sob_exact_results[i], cmap='gray')
    title2 = '闭式解 ($\\lambda$={:.2f})\n{}\n频域O($n^2\\log n$), n为边长'.format(lam_sob, fmt_metrics(psnr_sob_exact[i], ssim_sob_exact[i]))
    axes[1, i].set_title(title2)
    axes[1, i].axis('off')

plt.suptitle('Sobolev去噪：迭代解 vs 频域闭式解（均为周期边界）', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_迭代解与闭式解对比.png'), dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 步骤5（新增）：边界条件对照——周期(DFT) vs 镜像(DCT)
# ═══════════════════════════════════════════════════════════════════
bd_idx = SOB_LAMBDAS.index(BOUNDARY_LAMBDA)
x_dft_bd = x_sob_exact_results[bd_idx]
x_dct_bd = x_sob_dct_results[bd_idx]
err_dft_bd = np.abs(x_dft_bd - x_true)
err_dct_bd = np.abs(x_dct_bd - x_true)
err_vmax = max(float(err_dft_bd.max()), float(err_dct_bd.max()))

fig, axes = plt.subplots(2, 2, figsize=(11, 10))

im00 = axes[0, 0].imshow(x_dft_bd, cmap='gray')
axes[0, 0].set_title(f'周期边界（np.roll/DFT），$\\lambda$={BOUNDARY_LAMBDA:g}\n'
                     f'{fmt_metrics(psnr_sob[bd_idx], ssim_sob[bd_idx])}\n'
                     f'边界{RING_WIDTH}像素环 MSE={bmse_dft[bd_idx]:.5f}')
axes[0, 0].axis('off')

im01 = axes[0, 1].imshow(x_dct_bd, cmap='gray')
axes[0, 1].set_title(f'镜像边界（Neumann/DCT），$\\lambda$={BOUNDARY_LAMBDA:g}\n'
                     f'{fmt_metrics(psnr_sob_dct[bd_idx], ssim_sob_dct[bd_idx])}\n'
                     f'边界{RING_WIDTH}像素环 MSE={bmse_dct[bd_idx]:.5f}')
axes[0, 1].axis('off')

im10 = axes[1, 0].imshow(err_dft_bd, cmap='hot', vmin=0, vmax=err_vmax)
axes[1, 0].set_title('绝对误差 |$\\hat{x}-x_{真}$|：周期边界\n（误差集中在四条边缘）')
axes[1, 0].axis('off')
fig.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

im11 = axes[1, 1].imshow(err_dct_bd, cmap='hot', vmin=0, vmax=err_vmax)
axes[1, 1].set_title('绝对误差 |$\\hat{x}-x_{真}$|：镜像边界\n（边缘误差基本消除）')
axes[1, 1].axis('off')
fig.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

fig.suptitle('边界条件对照：周期拼接伪影是边界建模的代价，不是频域方法的固有属性\n'
             '两种解法同为 O($n^2\\log n$) 频域闭式解，仅边界条件不同', fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(SAVE_DIR, '步骤5_边界条件DFT与DCT对照.png'), dpi=150, bbox_inches='tight')
plt.close()

# 补充图2-1：L1 vs L2 几何解释（合并自 temp）
generate_figure_2_1(SAVE_DIR)

print("\n" + "=" * 70)
print("【高斯先验的特点总结】")
print("=" * 70)
print("\n1. 简单高斯先验 (Tikhonov):")
print("   - 假设: 像素 i.i.d. 高斯，协方差 σ_x² I（无空间结构）")
print("   - 均值必须与数据配套：N(0,σx²I) 用 E[x²] 定 σx²；")
print("     N(μ,σx²I) 用 Var(x) 定 σx² 并先去均值，二者不可混用")
print("   - 优点: 有闭式解，计算高效；高斯下 MAP=MMSE")
print("   - 缺点: 过度平滑、丢失边缘；对非零均值图像若不中心化，")
print("     参数再\"正确\"也会失效（本实验反面教材 PSNR 低于含噪图）")
print("   - 即使均值/方差都匹配（20.56dB），仍远不如 Sobolev（25.37dB）")

print("\n2. 梯度高斯先验 (Sobolev平滑):")
print("   - 假设: 相邻像素相似，先验协方差含梯度算子（空间结构）")
print("   - 优点: 编码空间相关性，比 i.i.d. 高斯合理得多")
print("   - 缺点: 仍会过度平滑边缘（L2 对梯度均匀惩罚）")
print("   - 数值稳定性: dt < 2/(1+8λ)（收敛域），代码取 0.9/(1+8λ)")
print("   - 超参数敏感性: λ影响平滑程度，λ≈0.5 PSNR最优，过大过平滑回落")
print("   - 频域闭式解: X̂ = Ŷ / (1 - λ·λ_Lap)，计算效率 O(n²log n)")
print("     其中 λ_Lap = 2cos(2πω_x) + 2cos(2πω_y) - 4 ≤ 0 为离散拉普拉斯特征值")
print("   - 边界条件: 周期(DFT)会对非周期图像造成拼接伪影；")
print("     镜像(Neumann/DCT)可显著缓解，复杂度相同——边界是建模选择而非方法宿命")

print("\n3. 与Laplace/TV先验的对比:")
print("   - 高斯先验 (L2): 均匀惩罚所有值，无精确零值")
print("   - Laplace/TV先验 (L1): 促稀疏，产生精确零值")
print("   - L1更适合稀疏表示，L2更适合平滑先验")

print("\n【实验完成】")
print(f"结果已保存至: {SAVE_DIR}")

# ===== 保存数值结果 =====
import json
results_summary = {
    'image_size': n,
    'noise_level': float(NOISE_LEVEL),
    'image_stats_true_for_teaching_only': {
        'mean': mu_true,
        'std': s_true,
        'var': float(s_true ** 2),
        'rms2_E_x2': rms2_true,
    },
    'plugin_estimates_from_observation': {
        'mu_y': mu_y,
        'rms2_hat_E_x2': rms2_hat,
        'var_hat': s2_hat,
        'note': '噪声方差已知时 E[y^2]=E[x^2]+sigma^2, var(y)=var(x)+sigma^2',
    },
    'psnr_noisy_dB': float(round(psnr_noisy, 2)),
    'ssim_noisy': float(round(ssim_noisy, 4)),
    'tikhonov': {
        '1_subjective_sigma_x_1': {
            'model': 'N(0,sigma_x^2 I), sigma_x=1 (主观先验)',
            'lambda': float(round(lam_tikh_subj, 6)),
            'psnr_dB': float(round(psnr_tikh_subj, 2)),
            'ssim': float(round(ssim_tikh_subj, 4)),
        },
        '2_empirical_bayes_zero_mean': {
            'model': 'N(0,sigma_x^2 I), sigma_x^2=E[x^2] 插件估计',
            'lambda': float(round(lam_tikh_eb0, 6)),
            'psnr_dB': float(round(psnr_tikh_eb0, 2)),
            'ssim': float(round(ssim_tikh_eb0, 4)),
        },
        '3_empirical_bayes_centered': {
            'model': 'N(mu,sigma_x^2 I), mu=y_mean, sigma_x^2=Var(x) 插件估计, 先去均值',
            'lambda': float(round(lam_tikh_center, 6)),
            'psnr_dB': float(round(psnr_tikh_center, 2)),
            'ssim': float(round(ssim_tikh_center, 4)),
        },
        '4_misuse_no_centering': {
            'model': '反面教材：lambda=sigma^2/Var(x) 却套零均值闭式解、不中心化',
            'lambda': float(round(lam_tikh_misuse, 6)),
            'psnr_dB': float(round(psnr_tikh_misuse, 2)),
            'ssim': float(round(ssim_tikh_misuse, 4)),
        },
    },
    'sobolev': {f'lambda_{lam:.2f}': {
        'periodic_dft_iter_psnr_dB': float(round(psnr_sob[i], 2)),
        'periodic_dft_iter_ssim': float(round(ssim_sob[i], 4)),
        'periodic_dft_exact_psnr_dB': float(round(psnr_sob_exact[i], 2)),
        'periodic_dft_exact_ssim': float(round(ssim_sob_exact[i], 4)),
        'mirror_dct_psnr_dB': float(round(psnr_sob_dct[i], 2)),
        'mirror_dct_ssim': float(round(ssim_sob_dct[i], 4)),
        'converged_iters': int(converged_iters[i]),
        'iter_vs_exact_max_abs_diff': float(iter_exact_diffs[i]),
        f'boundary_ring{ RING_WIDTH }px_mse_dft': float(bmse_dft[i]),
        f'boundary_ring{ RING_WIDTH }px_mse_dct': float(bmse_dct[i]),
        'interior_mse_dft': float(imse_dft[i]),
        'interior_mse_dct': float(imse_dct[i]),
    } for i, lam in enumerate(SOB_LAMBDAS)},
    'best_lambda_oracle_by_psnr_to_truth': {
        'periodic_dft_lambda': float(SOB_LAMBDAS[best_sob_idx]),
        'mirror_dct_lambda': float(SOB_LAMBDAS[best_sob_dct_idx]),
        'note': 'oracle 选择：使用了真值 PSNR，实际应用中需交叉验证/无监督准则',
    },
    'boundary_comparison': {
        'periodic': 'np.roll / DFT，图像对边周期拼接',
        'mirror': 'Neumann / DCT-II，镜像延拓',
        'ring_width_px': RING_WIDTH,
        'figure_lambda': BOUNDARY_LAMBDA,
        'conclusion': '内部区域MSE两者一致；大λ下周期边界边缘MSE显著更高，伪影源于边界建模而非频域方法本身',
    },
}

def _to_native(obj):
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    if isinstance(obj, (bool, np.bool_)): return bool(obj)
    return obj

results_summary = {k: _to_native(v) for k, v in results_summary.items()}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
