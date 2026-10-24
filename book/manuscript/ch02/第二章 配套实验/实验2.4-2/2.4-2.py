"""
实验2.4-2 Plug-and-Play先验：去噪器即先验
对应章节：2.4 从显式先验到隐式先验
知识点：PnP概念；去噪器=先验；PnP-GD迭代；隐式先验 vs 显式先验对比

素材来源：
  - examples/blind-inverse-problems/demo_blind_deblurring.py (PnP部分)
  - 2.4章节: PnP先验

修改说明（基于代码评审）：
  1. 删除原实验2.4-1(DIP)，原实验2.4-2(PnP)改为2.4-2（原2.4-1编号留给新增实验）
  2. 增加显式先验对比基线：Tikhonov(Wiener滤波)和TV去模糊
  3. 修正"数据驱动"相关注释，阐明simple_denoiser非数据驱动
  4. 补充去噪器类型分析及其与章节论点的关系
  5. （复核修正）PnP 从"对 A^T y 单次去噪"改为结论页声称的迭代 PnP-GD 形式，
     与正文公式 x^{k+1}=D_sigma(x^k - tau*A^T(A x^k - y)) 一致；
     同一手工去噪器下迭代版 PSNR 显著超过 TV，原"单次版劣于线性反演"系算法未跑满
  6. （复核修正）Tikhonov 低 PSNR 的归因由"循环卷积边界效应"改为
     "L2 先验与分段常数图像失配 + 小 lambda 反卷积噪声放大"（实测内部误差约为边界带 2 倍）
  7. （复核修正）修复 results_summary 段引用未定义变量 tau_pnp/num_iter_pnp 导致
     NameError 崩溃、JSON 从未写出的 bug；补归档 results_summary.json
  8. （第二轮复核修正）TV 基线由 Adam(lr=0.05,100步,lambda=0.05) 改为
     L-BFGS+强Wolfe(解至收敛, lambda=0.03 扫参最优)。原"100步已到收敛平台"
     系固定学习率 Adam 在极小点附近停滞，并非 TV 模型上限；PnP 对调优 TV 的
     真实优势为 16 seed 平均 +0.4 dB、胜率 75%（旧口径 1.78 dB 含弱基线水分）
  9. （第二轮复核修正）"单次去噪劣于线性反演"系稻草人对照：旧对照硬编码
     sigma=0.2（去噪器此时退化为均值模糊），与迭代版 sigma=0.05 不同口径。
     改为以 sigma=0.05 为主对照（实测稳定小幅优于线性反演），sigma=0.2 仅作
     "去噪器强度失配"演示；结论改写为"迭代把约+0.4 dB放大到约+10 dB"
 10. （第二轮复核修正）新增 16 个独立噪声实现的均值/标准差/胜率统计与
     稳健性图；弱化"隐式性才是收益来源"的过度归因（本实验手工去噪器编码的
     是与 TV 同族的保边先验，无"同先验、仅形式不同"的对照）
"""

import numpy as np
import os
import sys
# ====== 静默模式配置 ======
SILENT_MODE = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
import torch

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.4-2', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.4-2')
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    cn_font = setup_chinese_font(save_dir=_chinese_path)
    # 额外配置：确保LaTeX模式下也能使用中文字体
    if cn_font:
        plt.rcParams['font.sans-serif'] = [cn_font] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['font.family'] = 'sans-serif'
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
    print("请确保 .chinese 文件夹已上传到 Google Drive 的正确位置")

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ══════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════

def create_test_image(size=64):
    """创建测试图像（嵌套矩形）"""
    x = np.zeros((1, size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            if abs(i - center) < size // 4 and abs(j - center) < size // 4:
                x[0, i, j] = 1.0
            elif abs(i - center) < size // 3 and abs(j - center) < size // 3:
                x[0, i, j] = 0.5
    return torch.tensor(x, dtype=torch.float32)

def create_blur_kernel(size=5, sigma=1.0):
    """创建高斯模糊核"""
    x = np.arange(size) - size // 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return torch.tensor(kernel_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def conv2d(x, kernel, padding=None):
    """2D卷积"""
    if padding is None:
        padding = kernel.shape[-1] // 2
    return torch.nn.functional.conv2d(x, kernel, padding=padding)

def compute_psnr(x, x_true):
    """计算PSNR（值域[0,1]）"""
    mse = torch.mean((x - x_true)**2).item()
    if mse < 1e-12:
        return float('inf')
    return 10 * np.log10(1.0 / mse)

# ══════════════════════════════════════════════════════════
# 显式先验基线方法
# ══════════════════════════════════════════════════════════

def tikhonov_wiener(y, kernel, lam=0.01):
    """
    Tikhonov去模糊（Wiener滤波，频域闭式解）
    
    目标函数: min_x 0.5||Ax-y||^2 + lam * ||x||^2
    正则项: ||x||^2 (L2, 对应高斯先验)
    频域解: X_hat = conj(H)*Y / (|H|^2 + lam)
    
    注意: 频域FFT实现的是循环卷积，与前向算子 conv2d 的线性卷积在边界处
    行为不同，存在模型失配；但复核实测该失配是次要因素（边界12像素带
    MSE≈0.020，内部 MSE≈0.040，内部误差反而更大）。Tikhonov 表现差的
    主因是 L2 先验与分段常数图像失配，且 lam 偏小时反卷积放大噪声
    （换用无边界问题的 CG 精确求解同一目标, lam=0.01 也仅 ~16.6 dB;
    扫参最优 lam≈0.05 也只 ~21.9 dB, 仍低于线性反演）。
    """
    h, w = y.shape[-2], y.shape[-1]
    kh, kw = kernel.shape[-2], kernel.shape[-1]
    
# 将核放在左上角，然后循环移位使核中心对准(0,0)（FFT卷积要求）
    kernel_pad = torch.zeros(h, w, device=y.device)
    kernel_pad[:kh, :kw] = kernel.squeeze()
    kernel_pad = torch.roll(kernel_pad, shifts=(-kh//2, -kw//2), dims=(-2, -1))
    
    H = torch.fft.fft2(kernel_pad)
    Y = torch.fft.fft2(y[0])
    
    H_conj = torch.conj(H)
    X_hat = H_conj * Y / (H * H_conj + lam)
    x = torch.fft.ifft2(X_hat).real
    return x.unsqueeze(0)  # shape: (1, H, W) 与 x_true 一致


def tv_deblur(y, A, A_T, lam=0.03, max_iter=500):
    """
    TV去模糊: min_x 0.5||Ax-y||^2 + lam * TV(x)
    正则项: TV(x) = sum_i |(nabla x)_i| (梯度L1, 对应TV先验)

    x 形状为 (C, H, W), torch.diff 的 dim:
      dim=1: 沿行方向(垂直)差分
      dim=2: 沿列方向(水平)差分

    求解器: L-BFGS + 强Wolfe线搜索，把(1e-10光滑化后的)凸目标解到真正收敛。
    上一版用 Adam(lr=0.05,100步,lam=0.05) 并把 31.79 dB 称为"收敛平台"；
    第二轮复核实测那只是固定学习率 Adam 在极小点附近停滞：
      - 同一目标 lam=0.05: Adam-100 31.79 dB -> L-BFGS 31.93 dB
      - 公平扫参 lam=0.03: L-BFGS seed42 达 32.88 dB, 16 seed 均值 32.23 dB
    因此 TV 基线必须调参 + 解到收敛，与 PnP 的对比才公平。
    """
    x = A_T(y).detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [x], max_iter=max_iter, history_size=50,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-11, tolerance_change=1e-13)

    def closure():
        optimizer.zero_grad()
        # 数据保真项
        data_term = 0.5 * torch.mean((A(x) - y)**2)
        # 各向同性TV, 使用mean而非sum使lam对图像尺寸不敏感
        dx = torch.diff(x, dim=1)  # (C, H-1, W)
        dy = torch.diff(x, dim=2)  # (C, H, W-1)
        tv = torch.mean(torch.sqrt(dx[:, :, :dy.shape[2]]**2 +
                                   dy[:, :dx.shape[1], :]**2 + 1e-10))
        loss = data_term + lam * tv
        loss.backward()
        return loss

    optimizer.step(closure)
    return x.detach()


def cg_tikhonov(y, A, A_T, lam=0.05, n_iter=2000, tol=1e-10):
    """
    在线性(零填充)卷积模型下用共轭梯度精确求解 Tikhonov 目标:
        min_x 0.5||Ax-y||^2 + 0.5*lam*||x||^2
    即正规方程 (A^T A + lam I) x = A^T y。

    用途: tikhonov_wiener 的 FFT 实现隐含循环卷积边界假设，与前向算子的
    零填充卷积存在边界失配。CG 在真实算子上求解，可排除边界因素，检验
    L2 先验"调优 lambda 后"的真实上限(实测 lam=0.05 约 21.9 dB，
    仍低于线性反演，证明 L2 失配才是主因、FFT 边界只是次因)。
    """
    b = A_T(y)
    x = torch.zeros_like(b)
    r = b - (A_T(A(x)) + lam * x)
    p = r.detach().clone()
    rr = (r * r).sum()
    for _ in range(n_iter):
        Ap = A_T(A(p)) + lam * p
        alpha = rr / (p * Ap).sum()
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = (r * r).sum()
        if rr_new.sqrt() < tol:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


# ══════════════════════════════════════════════════════════
# PnP方法（去噪器 = 隐式先验）
# ══════════════════════════════════════════════════════════

def simple_denoiser(x, sigma_denoise=0.1):
    """
    简单去噪器 (邻域自适应加权平均)
    
    去噪方式: 以3x3邻域块的高斯相似度为权重，加权平均中心像素
    权重: w_ij = exp(-||patch_i - patch_j||^2 / (2*sigma^2))
    
    注意: 本实现基于逐像素邻域差值加权，非标准非局部均值(NLM)。
    标准NLM的权重基于patch之间的距离（邻域窗口内多像素比较），
    本实现退化为一类逐像素自适应加权平滑。
    自适应强度依赖 sigma_denoise 与输入噪声尺度匹配: 复核实测
    sigma_denoise=0.2 时 9 方向权重接近均匀(≈1/9)，退化为均值模糊;
    降至 0.05 后权重才真正随局部差异自适应。PnP-GD 中取 0.05。
    
    去噪器类型说明:
      本去噪器是手工设计的算法（基于块相似度加权平均），
      不需要任何外部训练数据，属于非数据驱动方法。
      这与2.4章节"隐式先验无需显式定义p(x)"的基本论点一致。
      数据驱动去噪器（DnCNN/DRUNet）虽可插入PnP框架，
      但已进入第3章"学习型先验"范畴，不在本章讨论。
    """
    # 兼容 3D (C,H,W) 和 4D (B,C,H,W) 输入
    is_3d = x.dim() == 3
    if is_3d:
        x = x.unsqueeze(0)
    
    h, w = x.shape[-2], x.shape[-1]
    x_pad = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')

    # 收集3x3邻域的所有平移patch
    # 1+di:1+di+h 正确地将切片对齐到h×w大小
    patches = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            patches.append(x_pad[:, :, 1+di:1+di+h, 1+dj:1+dj+w])

    patches = torch.stack(patches, dim=0)  # (9, B, C, H, W)
    # 每个移位方向单独计算权重，不跨移位方向求和
    diff_sq = (patches - x)**2          # (9, B, C, H, W)
    # 对单通道图像：逐像素、逐移位方向的权重
    weights = torch.exp(-diff_sq / (2 * sigma_denoise**2))
    # 沿移位方向归一化
    weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-10)

    result = (weights * patches).sum(dim=0)
    
    if is_3d:
        result = result.squeeze(0)
    return result


def pnp_gd(y, A, A_T, x0, tau=0.5, sigma_denoise=0.05, n_iter=50):
    """
    PnP-GD 迭代: x^{k+1} = D_sigma(x^k - tau * A^T(A x^k - y))
    用去噪器替代近端算子，梯度步(数据保真)与去噪(隐式先验)交替。
    """
    x = x0.detach().clone()
    for _ in range(n_iter):
        grad = A_T(A(x) - y)
        x = simple_denoiser(x - tau * grad, sigma_denoise)
    return x


# ══════════════════════════════════════════════════════════
# 实验设置
# ══════════════════════════════════════════════════════════

x_true = create_test_image(64).to(device)

kernel = create_blur_kernel(size=5, sigma=1.0).to(device)
A = lambda x: conv2d(x, kernel)
A_T = lambda x: conv2d(x, kernel.flip(-1).flip(-2))

noise_level = 0.05
y = A(x_true) + noise_level * torch.randn_like(x_true)

x_lin = A_T(y)

print("=" * 70)
print("实验2.4-2 Plug-and-Play先验：去噪器即先验")
print("=" * 70)
print(f"\n实验设定:")
print(f"  问题: 图像去模糊")
print(f"  模糊核: 高斯模糊 (5x5, sigma=1.0)")
print(f"  噪声水平: {noise_level}")
print(f"  图像尺寸: 64x64")

# ── 显式先验基线 ──
print("\n" + "-" * 50)
print("显式先验基线 (对比基准)")
print("-" * 50)

lam_tikh = 0.01
x_tikh = tikhonov_wiener(y, kernel, lam=lam_tikh)
psnr_tikh = compute_psnr(x_tikh, x_true)
print(f"  Tikhonov (L2, lam={lam_tikh}): PSNR={psnr_tikh:.2f} dB")

lam_tv = 0.03  # L-BFGS 下扫参所得最优(seed42: 0.02->32.52, 0.03->32.88, 0.04->32.50, 0.05->31.93)
x_tv = tv_deblur(y, A, A_T, lam=lam_tv)
psnr_tv = compute_psnr(x_tv, x_true)
print(f"  TV (L1 grad, L-BFGS收敛解, lam={lam_tv}): PSNR={psnr_tv:.2f} dB")

# Tikhonov 公平对照: CG 在真实(零填充)卷积算子上精确求解 + 扫参最优 lambda
lam_tikh_tuned = 0.05
x_tikh_tuned = cg_tikhonov(y, A, A_T, lam=lam_tikh_tuned)
psnr_tikh_tuned = compute_psnr(x_tikh_tuned, x_true)
print(f"  Tikhonov (CG精确解, 扫参最优lam={lam_tikh_tuned}): PSNR={psnr_tikh_tuned:.2f} dB")

# 与 FFT-Wiener 同一 lambda 的 CG 精确解：单独隔离"边界实现"这一因素
# （正文引用约 16.6 dB，指 lam=0.01 未调优；此前只写在注释里，现落盘 JSON）
lam_tikh_cg_untuned = lam_tikh
x_tikh_cg_untuned = cg_tikhonov(y, A, A_T, lam=lam_tikh_cg_untuned)
psnr_tikh_cg_untuned = compute_psnr(x_tikh_cg_untuned, x_true)
print(f"  Tikhonov (CG精确解, 与Wiener同lam={lam_tikh_cg_untuned}未调优): "
      f"PSNR={psnr_tikh_cg_untuned:.2f} dB")

# ── PnP隐式先验 ──
print("\n" + "-" * 50)
print("PnP隐式先验 (去噪器即先验)")
print("-" * 50)

# PnP核心思想: 去噪器 D_sigma 替代近端算子 prox_{tau*R}, 嵌入迭代格式
#   x^{k+1} = D_sigma( x^k - tau * A^T(A x^k - y) )
# 即 PnP-GD（与本页结论及 2.4 节正文公式一致）。tau 为梯度步长
# (tau < 1/||A^T A|| ≈ 1), sigma_denoise 为去噪器噪声水平假设(≈观测噪声 0.05)。
tau = 0.5            # 梯度步长
sigma_denoise = 0.05 # 去噪器工作噪声水平
num_iter_pnp = 50    # 复核实测 50 步已收敛(200 步仅 -0.1 dB)

x_pnp = pnp_gd(y, A, A_T, x_lin, tau, sigma_denoise, num_iter_pnp)

psnr_pnp = compute_psnr(x_pnp, x_true)
psnr_lin = compute_psnr(x_lin, x_true)
# 单次去噪对照: 主口径 sigma=0.05(与迭代版同口径); sigma=0.2 仅演示"去噪器失配"
psnr_single_05 = compute_psnr(simple_denoiser(x_lin, sigma_denoise), x_true)
psnr_single_02 = compute_psnr(simple_denoiser(x_lin, 0.2), x_true)

print(f"  线性反演: PSNR={psnr_lin:.2f} dB")
print(f"  PnP-GD 迭代({num_iter_pnp}步, tau={tau}, sigma_d={sigma_denoise}): PSNR={psnr_pnp:.2f} dB")
print(f"  [对照] 单次去噪 sigma=0.05(与迭代同口径)不迭代: PSNR={psnr_single_05:.2f} dB (小幅好于线性反演)")
print(f"  [对照] 单次去噪 sigma=0.2(去噪器失配,退化为均值模糊): PSNR={psnr_single_02:.2f} dB")

# ── 全部对比总结 ──
print("\n" + "-" * 50)
print("四种方法对比")
print("-" * 50)
methods = ['线性反演', 'Tikhonov', 'TV', 'PnP-GD']
psnrs_all = [psnr_lin, psnr_tikh, psnr_tv, psnr_pnp]
for name, p in zip(methods, psnrs_all):
    print(f"  {name}: PSNR={p:.2f} dB")

# ══════════════════════════════════════════════════════════
# 多噪声实现稳健性 (16 个独立 seed, 避免单张实现的偶然性)
# ══════════════════════════════════════════════════════════
print("\n" + "-" * 50)
print("多噪声实现稳健性 (16 个独立噪声实现)")
print("-" * 50)

NUM_SEEDS = 16
SEED_START = 1000
robust_rows = {k: [] for k in
               ['linear', 'tikhonov_wiener_0.01', 'tikhonov_cg_0.05',
                'single_pass_0.05', 'single_pass_0.2',
                'tv_lbfgs_0.03', 'pnp_gd_50']}
for _seed in range(SEED_START, SEED_START + NUM_SEEDS):
    torch.manual_seed(_seed)
    _y = A(x_true) + noise_level * torch.randn_like(x_true)
    _x_lin = A_T(_y)
    robust_rows['linear'].append(compute_psnr(_x_lin, x_true))
    robust_rows['tikhonov_wiener_0.01'].append(
        compute_psnr(tikhonov_wiener(_y, kernel, lam=lam_tikh), x_true))
    robust_rows['tikhonov_cg_0.05'].append(
        compute_psnr(cg_tikhonov(_y, A, A_T, lam=lam_tikh_tuned), x_true))
    robust_rows['single_pass_0.05'].append(
        compute_psnr(simple_denoiser(_x_lin, sigma_denoise), x_true))
    robust_rows['single_pass_0.2'].append(
        compute_psnr(simple_denoiser(_x_lin, 0.2), x_true))
    robust_rows['tv_lbfgs_0.03'].append(
        compute_psnr(tv_deblur(_y, A, A_T, lam=lam_tv), x_true))
    robust_rows['pnp_gd_50'].append(
        compute_psnr(pnp_gd(_y, A, A_T, _x_lin, tau, sigma_denoise, num_iter_pnp), x_true))

robust_stats = {k: (float(np.mean(v)), float(np.std(v)))
                for k, v in robust_rows.items()}
_pnp_arr = np.array(robust_rows['pnp_gd_50'])
_tv_arr = np.array(robust_rows['tv_lbfgs_0.03'])
_single05_arr = np.array(robust_rows['single_pass_0.05'])
_linear_arr = np.array(robust_rows['linear'])
pnp_vs_tv_winrate = float((_pnp_arr > _tv_arr).mean())
pnp_minus_tv_mean = float((_pnp_arr - _tv_arr).mean())
single05_vs_linear_winrate = float((_single05_arr > _linear_arr).mean())
single05_minus_linear_mean = float((_single05_arr - _linear_arr).mean())

for _k, (_m, _s) in robust_stats.items():
    print(f"  {_k:<22}: {_m:6.2f} +/- {_s:.2f} dB")
print(f"  PnP-GD > TV 胜率: {pnp_vs_tv_winrate*100:.0f}%, 平均增益 {pnp_minus_tv_mean:+.2f} dB")
print(f"  单次(0.05) > 线性反演 胜率: {single05_vs_linear_winrate*100:.0f}%, "
      f"平均增益 {single05_minus_linear_mean:+.2f} dB")

# ══════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第1行：原始 -> 退化 -> 显式先验基线
axes[0, 0].imshow(x_true.cpu().squeeze(), cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(y.cpu().squeeze(), cmap='gray')
axes[0, 1].set_title('模糊+噪声图像')
axes[0, 1].axis('off')

axes[0, 2].imshow(x_tikh.cpu().squeeze(), cmap='gray')
axes[0, 2].set_title(f'Tikhonov (L2显式先验, λ={lam_tikh}未调优)\nPSNR={psnr_tikh:.2f}dB')
axes[0, 2].axis('off')

# 第2行：线性反演 -> PnP -> TV
axes[1, 0].imshow(x_lin.cpu().squeeze(), cmap='gray')
axes[1, 0].set_title(f'线性反演\nPSNR={psnr_lin:.2f}dB')
axes[1, 0].axis('off')

axes[1, 1].imshow(x_pnp.cpu().squeeze(), cmap='gray')
axes[1, 1].set_title(f'PnP-GD 迭代{num_iter_pnp}步 (隐式先验)\nPSNR={psnr_pnp:.2f}dB')
axes[1, 1].axis('off')

axes[1, 2].imshow(x_tv.cpu().squeeze(), cmap='gray')
axes[1, 2].set_title(f'TV (L1梯度显式先验, L-BFGS λ={lam_tv}调优)\nPSNR={psnr_tv:.2f}dB')
axes[1, 2].axis('off')

plt.suptitle('隐式先验 vs 显式先验：图像去模糊任务对比', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # 调整布局，为suptitle留出更多空间
plt.subplots_adjust(hspace=0.175)  # 增加行间距，使两行子图之间有更多间隔
plt.savefig(os.path.join(SAVE_DIR, '步骤1_PnP与显式先验对比.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── 图2：多噪声实现稳健性 (均值±标准差) ──
_fig2_order = [
    ('linear',              '线性反演'),
    ('tikhonov_wiener_0.01', 'Tikhonov\nλ=0.01\n未调优'),
    ('tikhonov_cg_0.05',    'Tikhonov(CG)\nλ=0.05调优'),
    ('single_pass_0.05',    '单次去噪\nσ=0.05'),
    ('single_pass_0.2',     '单次去噪\nσ=0.2失配'),
    ('tv_lbfgs_0.03',       'TV\nL-BFGS调优'),
    ('pnp_gd_50',           'PnP-GD\n50步'),
]
_colors = ['#9e9e9e', '#d6604d', '#f4a582', '#92c5de', '#d1e5f0', '#4393c3', '#2166ac']
fig2, ax2 = plt.subplots(figsize=(11, 5.5))
_labels = [item[1] for item in _fig2_order]
_means = [robust_stats[key][0] for key, _ in _fig2_order]
_stds = [robust_stats[key][1] for key, _ in _fig2_order]
_bars = ax2.bar(_labels, _means, yerr=_stds, capsize=4, color=_colors, alpha=0.9)
for _bar, _m in zip(_bars, _means):
    ax2.text(_bar.get_x() + _bar.get_width() / 2, _m + 0.45,
             f'{_m:.1f}', ha='center', fontsize=10)
ax2.set_ylabel('PSNR (dB)', fontsize=12)
ax2.set_ylim(10, 36)
ax2.set_title(f'16个独立噪声实现的均值±标准差：PnP-GD 对调优TV平均 {pnp_minus_tv_mean:+.2f} dB、'
              f'胜率 {pnp_vs_tv_winrate*100:.0f}%；单次去噪σ=0.05 对线性反演 '
              f'{single05_minus_linear_mean:+.2f} dB', fontsize=12)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_多噪声实现稳健性.png'), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════
# 核心结论
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("【PnP先验核心结论】")
print("=" * 70)

print("\n1. PnP的定义:")
print("   用去噪器 D_tau 替代近端算子 prox_{tau*R}")
print("   x^{k+1} = D_tau(x^k - tau * A^T(A x^k - y))")

print("\n2. 为什么PnP有效? (本实验已按此迭代形式实现)")
print("   - Tweedie等式: D_tau(x) - x propto grad_x ln p(x)")
print("   - 去噪器隐式编码了图像先验的梯度信息")
print("   - 好的去噪器 = 好的先验")

print("\n3. 去噪器的类型与数据驱动的关系:")
print("   - 当前使用的 simple_denoiser 是邻域自适应加权平均算法")
print("   - 它不需要外部训练数据，属于非数据驱动方法")
print("   - 这与'隐式先验无需显式定义 p(x)'的章节论点一致")
print("   - PnP框架本身对去噪器来源无限制:")
print("     a) 手工设计去噪器 (NLM, BM3D) --- 不需要训练数据")
print("     b) 数据驱动去噪器 (DnCNN, DRUNet) --- 需要外部训练")
print("   - 数据驱动去噪器已进入第3章'学习型先验'范畴")

print("\n4. PnP vs 显式先验 (16 个独立噪声实现的稳健口径):")
print(f"   - Tikhonov (L2, λ={lam_tikh} 未调优): 最差; 即便 CG 排除边界失配且扫参到 "
      f"λ={lam_tikh_tuned}, 均值也仅 {robust_stats['tikhonov_cg_0.05'][0]:.1f} dB")
print(f"   - TV (L-BFGS 收敛解, λ={lam_tv} 调优): {robust_stats['tv_lbfgs_0.03'][0]:.2f}"
      f"±{robust_stats['tv_lbfgs_0.03'][1]:.2f} dB —— 需认真调参的强显式基线")
print(f"   - PnP-GD: {robust_stats['pnp_gd_50'][0]:.2f}±{robust_stats['pnp_gd_50'][1]:.2f} dB, "
      f"对调优TV平均 {pnp_minus_tv_mean:+.2f} dB、胜率 {pnp_vs_tv_winrate*100:.0f}% —— 占优但优势不大")
print(f"   - 单次去噪(σ=0.05, 与迭代同口径): 均值 {robust_stats['single_pass_0.05'][0]:.2f} dB, "
      f"对线性反演 {single05_minus_linear_mean:+.2f} dB、胜率 {single05_vs_linear_winrate*100:.0f}%")
print("     -> 单次并非'不升反降'; 真正的结论是: 迭代交替把约 +0.4 dB 放大到约 +10 dB")
print("        σ=0.2 低于线性反演只是去噪器强度失配(退化为均值模糊), 不能作为'必须交替'的证据")

print("\n5. PnP的优势与归因边界:")
print("   - 无需显式定义先验 p(x) 或正则项 R(x)")
print("   - 解耦数据保真项与先验项 (模块化)")
print("   - 适用于各种逆问题 (去模糊、超分、修复等)")
print("   - 归因边界: 本实验手工去噪器编码的是与 TV 同族的保边平滑, 无'同先验、")
print("     仅显式/隐式形式不同'的对照, 故小幅领先不能单独归因于'隐式性本身';")
print("     可下的结论是'不写显式正则项, 仅凭去噪器交替即可达到并略超调优TV'")

print("\n6. 理论保证:")
print("   PnP-ULA 收敛到 oracle 后验邻域")

print("\n" + "=" * 70)
print("实验完成。结果已保存至:", SAVE_DIR)
print("=" * 70)

# ===== 保存数值结果 =====
import json
results_summary = {
    'single_realization_seed42': {
        'psnr_linear_dB': round(psnr_lin, 2),
        'psnr_tikhonov_wiener_lam0.01_dB': round(psnr_tikh, 2),
        'psnr_tikhonov_cg_lam0.01_untuned_dB': round(psnr_tikh_cg_untuned, 2),
        'psnr_tikhonov_cg_lam0.05_dB': round(psnr_tikh_tuned, 2),
        'psnr_tv_lbfgs_lam0.03_dB': round(psnr_tv, 2),
        'psnr_pnp_gd_dB': round(psnr_pnp, 2),
        'psnr_single_pass_sigma0.05_dB': round(psnr_single_05, 2),
        'psnr_single_pass_sigma0.2_dB': round(psnr_single_02, 2),
    },
    'parameters': {
        'noise_sigma': noise_level,
        'lambda_tikhonov_wiener_headline': lam_tikh,
        'lambda_tikhonov_cg_untuned': lam_tikh_cg_untuned,
        'lambda_tikhonov_cg_tuned': lam_tikh_tuned,
        'lambda_tv': lam_tv,
        'tv_solver': 'L-BFGS + strong_wolfe (max_iter=500, 解至收敛)',
        'pnp_tau_step': tau,
        'pnp_sigma_denoise': sigma_denoise,
        'pnp_num_iters': num_iter_pnp,
    },
    'multi_seed_robustness': {
        'num_seeds': NUM_SEEDS,
        'seed_range': [SEED_START, SEED_START + NUM_SEEDS - 1],
        'mean_std_dB': {
            k: {'mean': round(m, 3), 'std': round(s, 3)}
            for k, (m, s) in robust_stats.items()
        },
        'pnp_gd_minus_tv_mean_dB': round(pnp_minus_tv_mean, 3),
        'pnp_gd_vs_tv_win_rate': round(pnp_vs_tv_winrate, 3),
        'single0.05_minus_linear_mean_dB': round(single05_minus_linear_mean, 3),
        'single0.05_vs_linear_win_rate': round(single05_vs_linear_winrate, 3),
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