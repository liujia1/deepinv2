# -*- coding: utf-8 -*-
"""
实验16.3-1 MRI成像基础——k-space采样与重建
对应章节：16.3.2节（k-space采样与傅里叶重建）
           16.3.3节（欠采样掩码与零填充重建）
           16.3.4节（压缩感知MRI）

实验目的：理解MRI正向模型（k-space = 图像的傅里叶变换），
          验证欠采样导致混叠伪影，对比三种采样策略，
          实现CS-MRI的ISTA重建

素材来源：全部原创
运行前提：CPU可运行
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import sys
import io
import warnings
import logging

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.rcParams['axes.unicode_minus'] = False

_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验16.3-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")

np.random.seed(42)
torch.manual_seed(42)

print(f"\n{'='*60}")
print(f"实验16.3-1: MRI成像基础——k-space采样与重建")
print(f"{'='*60}")

# ---- 准备幻影图像（模拟MRI脑图像）----
n = 128
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
phantom = phantom / phantom.max()
x_true = torch.tensor(phantom, dtype=torch.float32)


# ========================================================================
# 步骤1：k-space可视化（16.3.2节 k-space采样与傅里叶重建）
# ========================================================================
print("\n" + "=" * 60)
print("步骤1：k-space可视化（16.3.2节）")
print("=" * 60)

kspace = torch.fft.fftshift(torch.fft.fft2(x_true))
kspace_mag = torch.abs(kspace)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].imshow(phantom, cmap='gray')
axes[0].set_title('图像 $u(\\mathbf{r})$（Shepp-Logan幻影模拟脑）')
axes[0].axis('off')

axes[1].imshow(torch.log1p(kspace_mag).numpy(), cmap='gray')
axes[1].set_title('k-space $|s(\\mathbf{k})|$（对数尺度）\n低频能量集中在中心')
axes[1].axis('off')

mask_low = torch.zeros(n, n)
center = n // 4
mask_low[n//2-center:n//2+center, n//2-center:n//2+center] = 1
mask_high = 1 - mask_low

kspace_low = kspace * mask_low
kspace_high = kspace * mask_high

recon_low = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(kspace_low)))
recon_high = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(kspace_high)))

axes[2].imshow(torch.cat([recon_low[:, :n//2], recon_high[:, n//2:]], dim=1).numpy(), cmap='gray')
axes[2].set_title('左半: 低频重建（轮廓）| 右半: 高频重建（细节）')
axes[2].axis('off')

plt.suptitle('步骤1：MRI的k-space = 图像的傅里叶变换（16.3.2节）', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_k空间可视化.png'), dpi=150, bbox_inches='tight')
plt.show()


# ========================================================================
# 步骤2：欠采样掩码设计（16.3.3节）
# ========================================================================
print("\n" + "=" * 60)
print("步骤2：欠采样掩码设计（16.3.3节）")
print("=" * 60)

R = 4
n_lines = n // R

def mask_equispaced(n, R, seed=42):
    """等间距欠采样：沿相位编码方向每隔R行采样1行"""
    mask = torch.zeros(n)
    mask[::R] = 1
    mask[n//2] = 1
    return mask

def mask_random(n, R, seed=42):
    """随机欠采样：随机选择n/R行"""
    torch.manual_seed(seed)
    n_sample = max(n // R, 1)
    indices = torch.randperm(n)[:n_sample]
    mask = torch.zeros(n)
    mask[indices] = 1
    mask[n//2] = 1
    return mask

def mask_variable_density_topk(n, R, seed=None):
    """可变密度Top-K采样：按概率密度选择前K个位置（中心密集，外围稀疏）

    注：此实现为确定性Top-K选择（按概率密度贪心选取），非真随机采样。
    seed参数保留用于接口兼容，但不影响输出结果。
    """
    n_sample = max(n // R, 1)
    prob = torch.zeros(n)
    for i in range(n):
        dist = abs(i - n//2) / (n//2)
        prob[i] = (1 - dist**2) ** 1.5 + 0.02
    prob = prob / prob.sum() * n_sample
    mask = torch.zeros(n)
    sorted_indices = torch.argsort(prob, descending=True)
    mask[sorted_indices[:n_sample]] = 1
    mask[n//2] = 1
    return mask

mask_eq = mask_equispaced(n, R)
mask_rand = mask_random(n, R)
mask_vd = mask_variable_density_topk(n, R)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, mask, title in zip(axes,
    [mask_eq, mask_rand, mask_vd],
    [f'等间距采样 (R={R})\n相干混叠伪影——周期性条纹',
     f'随机采样 (R={R})\n不相干伪影——类噪声',
     f'可变密度采样 (R={R})\n中心密集+外围随机']):
    mask_2d = mask.unsqueeze(1).expand(-1, n)  # 沿行(k_y)方向欠采样
    ax.imshow(mask_2d.numpy(), cmap='gray', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('$k_x$（读出方向）')
    ax.set_ylabel('$k_y$（相位编码方向）')
    ax.axis('off')

plt.suptitle('步骤2：MRI欠采样掩码——三种采样策略', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_欠采样掩码.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"  等间距采样保留: {mask_eq.sum():.0f}/{n} 条相位编码线 (加速比={n/mask_eq.sum():.1f}x)")
print(f"  随机采样保留:   {mask_rand.sum():.0f}/{n} 条相位编码线")
print(f"  可变密度保留:   {mask_vd.sum():.0f}/{n} 条相位编码线")


# ========================================================================
# 步骤3：零填充重建与混叠伪影
# ========================================================================
print("\n" + "=" * 60)
print("步骤3：零填充重建与混叠伪影")
print("=" * 60)

def mri_forward(x, mask):
    """MRI正向算子: y = M_Ω * F * x"""
    kspace = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    mask_2d = mask.unsqueeze(1).expand(-1, x.shape[-1])  # 沿行(k_y)方向欠采样
    return kspace * mask_2d

def mri_adjoint(y, mask):
    """MRI伴随算子: x_zf = F^H * M_Ω^T * y（零填充重建）"""
    mask_2d = mask.unsqueeze(1).expand(-1, y.shape[-1])  # 沿行(k_y)方向欠采样
    return torch.abs(torch.fft.ifft2(torch.fft.ifftshift(y * mask_2d, dim=(-2, -1))))

def zero_filled_recon(x, mask):
    """零填充重建"""
    y = mri_forward(x, mask)
    return mri_adjoint(y, mask)

recon_eq = zero_filled_recon(x_true, mask_eq)
recon_rand = zero_filled_recon(x_true, mask_rand)
recon_vd = zero_filled_recon(x_true, mask_vd)

p_eq = psnr(phantom, recon_eq.numpy(), data_range=1.0)
p_rand = psnr(phantom, recon_rand.numpy(), data_range=1.0)
p_vd = psnr(phantom, recon_vd.numpy(), data_range=1.0)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for i, (mask, label) in enumerate(zip(
    [mask_eq, mask_rand, mask_vd],
    ['等间距', '随机', '可变密度'])):
    mask_2d = mask.unsqueeze(1).expand(-1, n)  # 沿行(k_y)方向欠采样
    axes[0, i].imshow(mask_2d.numpy(), cmap='gray', aspect='auto')
    axes[0, i].set_title(f'{label}掩码')
    axes[0, i].axis('off')

axes[0, 3].imshow(phantom, cmap='gray')
axes[0, 3].set_title('完整k-space重建（参考）\nPSNR=$\\infty$')
axes[0, 3].axis('off')

for i, (recon, label, p) in enumerate(zip(
    [recon_eq, recon_rand, recon_vd],
    ['等间距采样', '随机采样', '可变密度采样'],
    [p_eq, p_rand, p_vd])):
    axes[1, i].imshow(recon.numpy(), cmap='gray')
    axes[1, i].set_title(f'{label}零填充重建\nPSNR={p:.1f}dB')
    axes[1, i].axis('off')

diff = np.abs(recon_vd.numpy() - phantom)
axes[1, 3].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
axes[1, 3].set_title('可变密度重建误差图')
axes[1, 3].axis('off')

plt.suptitle('步骤3：零填充重建与混叠伪影（16.3.3节）', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_零填充重建.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"  等间距零填充:  PSNR={p_eq:.1f}dB")
print(f"  随机零填充:    PSNR={p_rand:.1f}dB")
print(f"  可变密度零填充: PSNR={p_vd:.1f}dB")


# ========================================================================
# 步骤4：压缩感知MRI——ISTA算法（16.3.4节）
# ========================================================================
print("\n" + "=" * 60)
print("步骤4：压缩感知MRI——ISTA算法（16.3.4节）")
print("=" * 60)

def cs_mri_ista(y, mask, n_iter=50, lam=0.01):
    """
    CS-MRI迭代重建（数据一致性+TV正则化梯度下降）
    min_x ||M_Ω * F * x - y||^2 + lambda * TV(x)

    由于MRI算子的特殊结构（F^H F = I），数据一致性步可高效实现：
    - 已采样k-space位置：用测量值替换
    - 未采样k-space位置：保持当前预测
    然后施加TV正则化作为投影/梯度步骤
    """
    mask_2d = mask.unsqueeze(1).expand(-1, y.shape[-1])  # 沿行(k_y)方向欠采样
    x = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(y * mask_2d, dim=(-2, -1))))

    for k in tqdm(range(n_iter), desc='  CS-MRI ISTA', leave=False):
        kspace_x = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        kspace_dc = mask_2d * y + (1 - mask_2d) * kspace_x
        x_dc = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(kspace_dc, dim=(-2, -1))))

        eps = 1e-8
        dx = torch.diff(x_dc, dim=1, prepend=x_dc[:, :1])
        dy = torch.diff(x_dc, dim=0, prepend=x_dc[:1, :])
        mag = torch.sqrt(dx**2 + dy**2 + eps)
        div_x = torch.diff(dx / mag, dim=1, append=(dx / mag)[:, -1:])
        div_y = torch.diff(dy / mag, dim=0, append=(dy / mag)[-1:, :])
        tv_grad = -(div_x + div_y)

        x = x_dc - lam * tv_grad
        x = torch.clamp(x, 0, None)

    return x

y_vd = mri_forward(x_true, mask_vd)
y_vd_noisy = y_vd + 0.01 * torch.randn_like(y_vd)

print("  正在执行CS-MRI ISTA重建...")
recon_cs = cs_mri_ista(y_vd, mask_vd, n_iter=50, lam=0.01)
p_cs = psnr(phantom, recon_cs.numpy(), data_range=1.0)

# 收敛曲线
psnr_curve = []
mask_2d = mask_vd.unsqueeze(1).expand(-1, n)  # 沿行(k_y)方向欠采样
x_ista = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(y_vd * mask_2d, dim=(-2, -1))))
for k in tqdm(range(50), desc='  收敛曲线', leave=False):
    kspace_x = torch.fft.fftshift(torch.fft.fft2(x_ista), dim=(-2, -1))
    kspace_dc = mask_2d * y_vd + (1 - mask_2d) * kspace_x
    x_dc = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(kspace_dc, dim=(-2, -1))))
    eps = 1e-8
    dx_i = torch.diff(x_dc, dim=1, prepend=x_dc[:, :1])
    dy_i = torch.diff(x_dc, dim=0, prepend=x_dc[:1, :])
    mag = torch.sqrt(dx_i**2 + dy_i**2 + eps)
    div_x = torch.diff(dx_i / mag, dim=1, append=(dx_i / mag)[:, -1:])
    div_y = torch.diff(dy_i / mag, dim=0, append=(dy_i / mag)[-1:, :])
    tv_grad = -(div_x + div_y)
    x_ista = x_dc - 0.01 * tv_grad
    x_ista = torch.clamp(x_ista, 0, None)
    psnr_curve.append(psnr(phantom, x_ista.numpy(), data_range=1.0))

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

axes[0].imshow(phantom, cmap='gray')
axes[0].set_title('原始幻影')
axes[0].axis('off')

axes[1].imshow(recon_vd.numpy(), cmap='gray')
axes[1].set_title(f'零填充重建 (R={R})\nPSNR={p_vd:.1f}dB')
axes[1].axis('off')

axes[2].imshow(recon_cs.numpy(), cmap='gray')
axes[2].set_title(f'CS-MRI ISTA (R={R}, $\\lambda$=0.005)\nPSNR={p_cs:.1f}dB\nTV稀疏性抑制伪影')
axes[2].axis('off')

axes[3].plot(range(len(psnr_curve)), psnr_curve, 'bo-')
axes[3].set_xlabel('ISTA迭代次数')
axes[3].set_ylabel('PSNR (dB)')
axes[3].set_title('ISTA收敛曲线')
axes[3].grid(True)

plt.suptitle('步骤4：压缩感知MRI——零填充 vs CS-ISTA（16.3.4节）', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_CS_MRI.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"  零填充:   PSNR={p_vd:.1f}dB")
print(f"  CS-ISTA:  PSNR={p_cs:.1f}dB")


# ========================================================================
# 步骤5：加速因子对重建质量的影响
# ========================================================================
print("\n" + "=" * 60)
print("步骤5：加速因子对重建质量的影响")
print("=" * 60)

R_list = [2, 4, 6, 8]
results_zf = []
results_cs = []

for R_val in tqdm(R_list, desc='  加速因子扫描'):
    mask_vd_R = mask_variable_density_topk(n, R_val)
    y_R = mri_forward(x_true, mask_vd_R)

    recon_zf = zero_filled_recon(x_true, mask_vd_R)
    p_zf = psnr(phantom, recon_zf.numpy(), data_range=1.0)
    results_zf.append(p_zf)

    recon_cs_R = cs_mri_ista(y_R, mask_vd_R, n_iter=50, lam=0.01)
    p_cs_R = psnr(phantom, recon_cs_R.numpy(), data_range=1.0)
    results_cs.append(p_cs_R)

    print(f"  R={R_val}: 保留{mask_vd_R.sum():.0f}条相位编码线, 零填充={p_zf:.1f}dB, CS-ISTA={p_cs_R:.1f}dB")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(R_list, results_zf, 'ro-', label='零填充重建')
ax.plot(R_list, results_cs, 'bs-', label='CS-MRI ISTA')
ax.set_xlabel('加速因子 R')
ax.set_ylabel('PSNR (dB)')
ax.set_title('MRI重建质量 vs 加速因子\nR越大→欠采样越严重→需要更强的先验')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤5_加速因子对比.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n实验16.3-1完成！")
