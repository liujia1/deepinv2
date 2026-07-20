"""
实验2.4-1 Plug-and-Play先验：去噪器即先验
对应章节：2.4 从显式先验到隐式先验
知识点：PnP概念；去噪器=先验；DPIR算法；隐式先验 vs 显式先验对比

素材来源：
  - examples/blind-inverse-problems/demo_blind_deblurring.py (PnP部分)
  - 2.4章节: PnP先验

修改说明（基于代码评审）：
  1. 删除原实验2.4-1(DIP)，原实验2.4-2(PnP)改为2.4-1
  2. 增加显式先验对比基线：Tikhonov(Wiener滤波)和TV去模糊
  3. 修正"数据驱动"相关注释，阐明simple_denoiser非数据驱动
  4. 补充去噪器类型分析及其与章节论点的关系
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
    _chinese_path = os.path.join(_gdrive, '实验2.4-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.4-1')
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
    
    注意: 频域FFT实现的是循环卷积，与conv2d的线性卷积在边界处行为不同。
    这会导致边界区域误差较大，小图像(64x64)上影响更明显。
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


def tv_deblur(y, A, A_T, lam=0.05, n_iter=100):
    """
    TV去模糊: min_x 0.5||Ax-y||^2 + lam * TV(x)
    正则项: TV(x) = sum_i |(nabla x)_i| (梯度L1, 对应TV先验)
    
    x 形状为 (C, H, W), torch.diff 的 dim:
      dim=1: 沿行方向(垂直)差分
      dim=2: 沿列方向(水平)差分
    
    n_iter=100 为速度权衡, 未完全收敛, 更公平对比可增至300
    """
    x = A_T(y).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=0.05)

    for _ in range(n_iter):
        optimizer.zero_grad()

        # 数据保真项
        data_term = 0.5 * torch.mean((A(x) - y)**2)

        # 各向同性TV (内部区域, 边界项用绝对值近似)
        dx = torch.diff(x, dim=1)  # (C, H-1, W)
        dy = torch.diff(x, dim=2)  # (C, H, W-1)
        # 取重叠区域: (C, H-1, W-1)
        # 注意: 使用mean而非sum，使lam参数对图像尺寸不敏感
        tv = torch.mean(torch.sqrt(dx[:, :, :dy.shape[2]]**2 +
                                   dy[:, :dx.shape[1], :]**2 + 1e-10))

        loss = data_term + lam * tv
        loss.backward()
        optimizer.step()

    return x.detach()


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
print("实验2.4-1 Plug-and-Play先验：去噪器即先验")
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

lam_tv = 0.05
x_tv = tv_deblur(y, A, A_T, lam=lam_tv)
psnr_tv = compute_psnr(x_tv, x_true)
print(f"  TV (L1 grad, lam={lam_tv}): PSNR={psnr_tv:.2f} dB")

# ── PnP隐式先验 ──
print("\n" + "-" * 50)
print("PnP隐式先验 (去噪器即先验)")
print("-" * 50)

tau = 0.2
# PnP核心思想: 去噪器 D_tau 可替代近端算子 prox_{tau*R}
# 此处演示最简形式：对线性反演 x_lin = A^T y 应用去噪器，
# 去噪器隐式编码了先验信息（Tweedie等式），对非对称问题可嵌入迭代框架。
# 完整PnP-ADMM/GD需要更强的去噪器(BM3D/DnCNN, 见第3章)。
x_pnp = simple_denoiser(x_lin, tau)  # 对线性反演结果去噪
psnr_pnp = compute_psnr(x_pnp, x_true)
psnr_lin = compute_psnr(x_lin, x_true)

print(f"  线性反演: PSNR={psnr_lin:.2f} dB")
print(f"  PnP (去噪器直接应用): PSNR={psnr_pnp:.2f} dB")

# ── 全部对比总结 ──
print("\n" + "-" * 50)
print("四种方法对比")
print("-" * 50)
methods = ['线性反演', 'Tikhonov', 'TV', 'PnP']
psnrs_all = [psnr_lin, psnr_tikh, psnr_tv, psnr_pnp]
for name, p in zip(methods, psnrs_all):
    print(f"  {name}: PSNR={p:.2f} dB")

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
axes[0, 2].set_title(f'Tikhonov (L2显式先验)\nPSNR={psnr_tikh:.2f}dB')
axes[0, 2].axis('off')

# 第2行：线性反演 -> PnP -> TV
axes[1, 0].imshow(x_lin.cpu().squeeze(), cmap='gray')
axes[1, 0].set_title(f'线性反演\nPSNR={psnr_lin:.2f}dB')
axes[1, 0].axis('off')

axes[1, 1].imshow(x_pnp.cpu().squeeze(), cmap='gray')
axes[1, 1].set_title(f'PnP (隐式先验)\nPSNR={psnr_pnp:.2f}dB')
axes[1, 1].axis('off')

axes[1, 2].imshow(x_tv.cpu().squeeze(), cmap='gray')
axes[1, 2].set_title(f'TV (L1梯度显式先验)\nPSNR={psnr_tv:.2f}dB')
axes[1, 2].axis('off')

plt.suptitle('隐式先验 vs 显式先验：图像去模糊任务对比', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # 调整布局，为suptitle留出更多空间
plt.subplots_adjust(hspace=0.175)  # 增加行间距，使两行子图之间有更多间隔
plt.savefig(os.path.join(SAVE_DIR, '步骤1_PnP与显式先验对比.png'), dpi=150, bbox_inches='tight')
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

print("\n2. 为什么PnP有效?")
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

print("\n4. PnP vs 显式先验 (本实验对比基线):")
print("   - Tikhonov (L2): 闭式解, 计算快, 但过度平滑")
print("   - TV (L1梯度): 保边, 需迭代求解")
print("   - PnP: 无需显式定义正则项, 可嵌入手工或学习型去噪器")

print("\n5. PnP的优势:")
print("   - 无需显式定义先验 p(x) 或正则项 R(x)")
print("   - 解耦数据保真项与先验项 (模块化)")
print("   - 适用于各种逆问题 (去模糊、超分、修复等)")

print("\n6. 理论保证:")
print("   PnP-ULA 收敛到 oracle 后验邻域")

print("\n" + "=" * 70)
print("实验完成。结果已保存至:", SAVE_DIR)
print("=" * 70)

# ===== 保存数值结果 =====
import json
results_summary = {
    'psnr_linear_dB': float(round(psnr_lin, 2)),
    'psnr_tikhonov_dB': float(round(psnr_tikh, 2)),
    'psnr_tv_dB': float(round(psnr_tv, 2)),
    'psnr_pnp_dB': float(round(psnr_pnp, 2)),
    'lambda_tikhonov': float(lam_tikh),
    'lambda_tv': float(lam_tv),
    'pnp_tau': float(tau_pnp),
    'pnp_num_iters': int(num_iter_pnp),
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