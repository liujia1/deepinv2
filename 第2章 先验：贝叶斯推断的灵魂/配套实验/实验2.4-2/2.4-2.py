"""
实验2.4-2 Plug-and-Play先验：去噪器即先验
对应章节：2.4 从显式先验到隐式先验
知识点：PnP概念；去噪器=先验；DPIR算法；隐式先验的优势

素材来源：
  - examples/blind-inverse-problems/demo_blind_deblurring.py (PnP部分)
  - 2.4章节: PnP先验
"""

import numpy as np
import os
import sys
import warnings

# ====== 静默模式配置 ======
SILENT_MODE = True  # True: 不弹窗、不显示警告；False: 正常交互模式

if SILENT_MODE:
    import matplotlib
    matplotlib.use('Agg')
    warnings.filterwarnings('ignore')
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
else:
    import matplotlib

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.4-2', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.4-2')
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
    print("请确保 .chinese 文件夹已上传到 Google Drive 的正确位置")

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_test_image(size=64):
    """创建测试图像"""
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

def simple_denoiser(x, sigma_denoise=0.1):
    """简单去噪器 (非局部均值简化版)"""
    h, w = x.shape[-2], x.shape[-1]
    x_pad = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
    
    patches = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            patches.append(x_pad[:, :, di:di+h, dj:dj+w])
    
    patches = torch.stack(patches, dim=0)
    weights = torch.exp(-torch.sum((patches - x)**2, dim=0) / (2 * sigma_denoise**2))
    weights = weights / weights.sum(dim=0, keepdim=True)
    
    result = (weights * patches).sum(dim=0)
    return result

def proximal_operator(x, tau, denoiser):
    """近端算子 = 去噪器"""
    return denoiser(x, tau)

def pnp_admm(y, A, A_T, denoiser, sigma, max_iter=50, tau=0.1):
    """
    PnP-ADMM算法
    
    问题: min_x 0.5||Ax-y||² + λ·R(x)
    
    ADMM迭代:
    x^{k+1} = (A^T A + ρI)^{-1} (A^T y + ρ(v^k - u^k))
    v^{k+1} = prox_{λ/ρ·R}(x^{k+1} + u^k)  ← 用去噪器替代
    u^{k+1} = u^k + x^{k+1} - v^{k+1}
    """
    h, w = y.shape[-2], y.shape[-1]
    x = A_T(y).clone()
    v = x.clone()
    u = torch.zeros_like(x)
    
    rho = 1.0
    
    losses = []
    psnrs = []
    
    for k in range(max_iter):
        x_new = (A_T(y) + rho * (v - u)) / (1 + rho)
        
        v_new = denoiser(x_new + u, tau)
        
        u = u + x_new - v_new
        
        x = x_new
        v = v_new
        
        loss = torch.mean((A(x) - y)**2).item()
        losses.append(loss)
    
    return x, losses

img_size = (1, 64, 64)
x_true = create_test_image(64).to(device)

kernel = create_blur_kernel(size=5, sigma=1.0).to(device)
A = lambda x: conv2d(x, kernel)
A_T = lambda x: conv2d(x, kernel.flip(-1))

noise_level = 0.05
y = A(x_true) + noise_level * torch.randn_like(x_true)

x_lin = A_T(y)

max_iter = 30
tau = 0.1

print("===== Plug-and-Play先验：去噪器即先验 =====")
print(f"\n实验设定:")
print(f"  问题: 图像去模糊")
print(f"  模糊核: 高斯模糊 (σ=1.0)")
print(f"  噪声水平: {noise_level}")
print(f"  迭代次数: {max_iter}")
print(f"\nPnP核心思想:")
print(f"  近端算子 prox_{τR}(x) = 去噪器 D_τ(x)")
print(f"  去噪器隐式编码了先验信息")
print(f"  残差 D_τ(x) - x ∝ ∇_x ln p(x) (Tweedie等式)")

x_pnp, losses = pnp_admm(y, A, A_T, simple_denoiser, noise_level, max_iter, tau)

def compute_psnr(x, x_true):
    mse = torch.mean((x - x_true)**2).item()
    return 10 * np.log10(1.0 / mse)

psnr_lin = compute_psnr(x_lin, x_true)
psnr_pnp = compute_psnr(x_pnp, x_true)

print(f"\n结果:")
print(f"  线性反演 PSNR: {psnr_lin:.2f} dB")
print(f"  PnP重建 PSNR: {psnr_pnp:.2f} dB")
print(f"  提升: {psnr_pnp - psnr_lin:.2f} dB")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(x_true.cpu().squeeze(), cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(y.cpu().squeeze(), cmap='gray')
axes[0, 1].set_title('模糊+噪声图像')
axes[0, 1].axis('off')

axes[0, 2].imshow(x_lin.cpu().squeeze(), cmap='gray')
axes[0, 2].set_title(f'线性反演\nPSNR={psnr_lin:.2f}dB')
axes[0, 2].axis('off')

axes[1, 0].imshow(x_pnp.cpu().squeeze(), cmap='gray')
axes[1, 0].set_title(f'PnP重建\nPSNR={psnr_pnp:.2f}dB')
axes[1, 0].axis('off')

axes[1, 1].plot(losses)
axes[1, 1].set_xlabel('迭代次数')
axes[1, 1].set_ylabel('数据保真项')
axes[1, 1].set_title('收敛曲线')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].text(0.5, 0.8, 'PnP核心公式', fontsize=14, ha='center', fontweight='bold')
axes[1, 2].text(0.5, 0.6, 'prox_{τR}(x) = D_τ(x)', fontsize=12, ha='center')
axes[1, 2].text(0.5, 0.45, '去噪器 = 近端算子', fontsize=12, ha='center')
axes[1, 2].text(0.5, 0.3, 'D_τ(x) - x ∝ ∇_x ln p(x)', fontsize=12, ha='center')
axes[1, 2].text(0.5, 0.15, '(Tweedie等式, 第5章)', fontsize=10, ha='center', style='italic')
axes[1, 2].axis('off')

plt.suptitle('Plug-and-Play先验: 去噪器即先验', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_PnP实验.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()

print("\n===== 2.4章节核心结论 =====")
print("\n1. PnP的定义:")
print("   用预训练去噪器 D_ε 替代近端算子 prox_{τR}")
print("   x^{k+1} = D_ε(x^k - τA^T(Ax^k - y))")
print("\n2. 为什么PnP有效?")
print("   - 去噪器隐式编码了先验的梯度信息")
print("   - D_ε(x) - x ∝ ∇_x ln p(x) (Tweedie等式)")
print("   - 好的去噪器 = 好的先验")
print("\n3. PnP的优势:")
print("   - 无需显式定义先验 p(x)")
print("   - 可以使用任何去噪器 (BM3D, DnCNN等)")
print("   - 适用于各种逆问题")
print("\n4. 与显式先验的对比:")
print("   显式先验: 需要手工指定 R(x)")
print("   PnP: 通过去噪器隐式编码先验")
print("   数据驱动: 去噪器可从数据中学习")
print("\n5. 理论保证:")
print("   PnP-ULA 收敛到 oracle 后验邻域")
print("   (详见 Pereyra 等人的理论分析)")
