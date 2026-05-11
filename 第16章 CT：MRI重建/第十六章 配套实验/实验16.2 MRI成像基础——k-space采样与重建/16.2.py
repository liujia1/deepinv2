# -*- coding: utf-8 -*-
"""
实验16.2 MRI成像基础——k-space采样与重建
对应知识点：16.3.2节（k-space采样）、16.3.3节（欠采样掩码与零填充重建）
           16.3.4节（压缩感知MRI）

★原创设计：全部代码从零编写，无现有MRI代码可复用
- 使用torch.fft实现k-space采样与重建
- 实现三种欠采样掩码（等间距/随机/可变密度）
- 实现CS-MRI的ISTA算法（TV正则化）
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib as mpl
import warnings
import logging

# ====== 解决中文乱码的核心代码（Windows + Linux 自动适配）======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

import platform
from matplotlib.font_manager import FontManager, FontProperties

def _find_chinese_font():
    """自动检测系统中可用的中文字体，兼容 Windows / Linux"""
    candidates = []
    if platform.system() == 'Windows':
        candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    else:
        candidates = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
            'Noto Sans CJK SC', 'Noto Sans CJK',
            'Source Han Sans SC', 'AR PL UMing CN',
            'SimHei',
        ]
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import os, re
    cjk_patterns = ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']
    for f in fm.ttflist:
        name_lower = f.name.lower()
        fname_lower = (os.path.basename(f.fname) if hasattr(f, 'fname') else '').lower()
        for pat in cjk_patterns:
            if re.search(pat, name_lower) or re.search(pat, fname_lower):
                return f.name
    return None

_cn_font = _find_chinese_font()
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"[Font] 已检测到中文字体: {_cn_font}")
else:
    print("[Font] 未找到中文字体，中文可能显示为方框")
# ========================================================

np.random.seed(42)
torch.manual_seed(42)


# ---- 准备幻影图像（模拟MRI脑图像）----
n = 128
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
phantom = phantom / phantom.max()
x_true = torch.tensor(phantom, dtype=torch.float32)


# ========================================================================
# ★ 步骤1：k-space可视化（16.3.2节 k-space采样与傅里叶重建）
# ========================================================================
print("=" * 60)
print("步骤1：k-space可视化（16.3.2节）")
print("=" * 60)

# MRI测量 = 图像的2D傅里叶变换
# s(k) = ∫ u(r) exp(-2πi k·r) dr
kspace = torch.fft.fftshift(torch.fft.fft2(x_true))
kspace_mag = torch.abs(kspace)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].imshow(phantom, cmap='gray')
axes[0].set_title('图像 $u(\\mathbf{r})$（Shepp-Logan幻影模拟脑）')
axes[0].axis('off')

axes[1].imshow(torch.log1p(kspace_mag).numpy(), cmap='gray')
axes[1].set_title('k-space $|s(\\mathbf{k})|$（对数尺度）\n低频能量集中在中心')
axes[1].axis('off')

# 分别显示低频和高频的贡献
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
plt.savefig('步骤1_k空间可视化.png', dpi=150, bbox_inches='tight')
plt.show()


# ========================================================================
# ★ 步骤2：欠采样掩码设计（16.3.3节）
# ★原创设计：三种掩码类型的系统对比
# ========================================================================
print("\n" + "=" * 60)
print("步骤2：欠采样掩码设计（16.3.3节）")
print("=" * 60)

R = 4  # 加速因子（4倍加速）
n_lines = n // R  # 保留的k-space行数

def mask_equispaced(n, R, seed=42):
    """等间距欠采样：每隔R行采样1行"""
    mask = torch.zeros(n)
    mask[::R] = 1
    # 始终保留中心k-space行
    mask[n//2] = 1
    return mask

def mask_random(n, R, seed=42):
    """随机欠采样：随机选择n/R行"""
    torch.manual_seed(seed)
    n_sample = max(n // R, 1)
    indices = torch.randperm(n)[:n_sample]
    mask = torch.zeros(n)
    mask[indices] = 1
    mask[n//2] = 1  # 保留中心
    return mask

def mask_variable_density(n, R, seed=42):
    """★ 可变密度随机采样：中心密集，外围稀疏"""
    torch.manual_seed(seed)
    n_sample = max(n // R, 1)
    # 概率密度：中心高，外围低
    prob = torch.zeros(n)
    for i in range(n):
        dist = abs(i - n//2) / (n//2)
        prob[i] = (1 - dist**2) ** 1.5 + 0.02
    prob = prob / prob.sum() * n_sample
    # 按概率采样
    mask = torch.zeros(n)
    sorted_indices = torch.argsort(prob, descending=True)
    mask[sorted_indices[:n_sample]] = 1
    mask[n//2] = 1  # 确保保留中心
    return mask

mask_eq = mask_equispaced(n, R)
mask_rand = mask_random(n, R)
mask_vd = mask_variable_density(n, R)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, mask, title in zip(axes,
    [mask_eq, mask_rand, mask_vd],
    [f'等间距采样 (R={R})\n相干混叠伪影——周期性条纹',
     f'随机采样 (R={R})\n不相干伪影——类噪声',
     f'★可变密度采样 (R={R})\n中心密集+外围随机']):
    mask_2d = mask.unsqueeze(0).expand(n, -1)
    ax.imshow(mask_2d.numpy(), cmap='gray', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('kx')
    ax.set_ylabel('ky (相位编码方向)')
    ax.axis('off')

plt.suptitle('步骤2：MRI欠采样掩码——三种采样策略（16.3.3节）', fontsize=13)
plt.tight_layout()
plt.savefig('步骤2_欠采样掩码.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  等间距采样保留: {mask_eq.sum():.0f}/{n} 行 (加速比={n/mask_eq.sum():.1f}x)")
print(f"  随机采样保留:   {mask_rand.sum():.0f}/{n} 行")
print(f"  可变密度保留:   {mask_vd.sum():.0f}/{n} 行")


# ========================================================================
# 步骤3：零填充重建与混叠伪影（16.3.3节）
# ========================================================================
print("\n" + "=" * 60)
print("步骤3：零填充重建与混叠伪影（16.3.3节）")
print("=" * 60)

def mri_forward(x, mask):
    """MRI正向算子: y = M_Ω * F * x
    mask: (H,) 1D采样掩码, 1=采样, 0=未采样 (以k-space中心为基准)
    """
    kspace = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    mask_2d = mask.unsqueeze(0).expand(x.shape[0], -1)  # (H, W), 沿ky方向采样
    return kspace * mask_2d

def mri_adjoint(y, mask):
    """MRI伴随算子: x_zf = F^H * M_Ω^T * y（零填充重建）"""
    mask_2d = mask.unsqueeze(0).expand(y.shape[0], -1)
    return torch.abs(torch.fft.ifft2(torch.fft.ifftshift(y * mask_2d, dim=(-2, -1))))

def zero_filled_recon(x, mask):
    """零填充重建"""
    y = mri_forward(x, mask)
    return mri_adjoint(y, mask)

# 对三种掩码分别做零填充重建
recon_eq = zero_filled_recon(x_true, mask_eq)
recon_rand = zero_filled_recon(x_true, mask_rand)
recon_vd = zero_filled_recon(x_true, mask_vd)

p_full = psnr(phantom, phantom, data_range=1.0)
p_eq = psnr(phantom, recon_eq.numpy(), data_range=1.0)
p_rand = psnr(phantom, recon_rand.numpy(), data_range=1.0)
p_vd = psnr(phantom, recon_vd.numpy(), data_range=1.0)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# 第一行：掩码
for i, (mask, label) in enumerate(zip(
    [mask_eq, mask_rand, mask_vd],
    ['等间距', '随机', '★可变密度'])):
    mask_2d = mask.unsqueeze(0).expand(n, -1)
    axes[0, i].imshow(mask_2d.numpy(), cmap='gray', aspect='auto')
    axes[0, i].set_title(f'{label}掩码')
    axes[0, i].axis('off')

# 完整k-space重建（参考）
axes[0, 3].imshow(phantom, cmap='gray')
axes[0, 3].set_title('完整k-space重建（参考）\nPSNR=∞')
axes[0, 3].axis('off')

# 第二行：零填充重建
for i, (recon, label, p) in enumerate(zip(
    [recon_eq, recon_rand, recon_vd],
    ['等间距采样', '随机采样', '★可变密度采样'],
    [p_eq, p_rand, p_vd])):
    axes[1, i].imshow(recon.numpy(), cmap='gray')
    axes[1, i].set_title(f'{label}零填充重建\nPSNR={p:.1f}dB')
    axes[1, i].axis('off')

# 差异图（可变密度 vs 真值）
diff = np.abs(recon_vd.numpy() - phantom)
axes[1, 3].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
axes[1, 3].set_title('★可变密度重建误差图')
axes[1, 3].axis('off')

plt.suptitle('步骤3：零填充重建与混叠伪影（16.3.3节）', fontsize=13)
plt.tight_layout()
plt.savefig('步骤3_零填充重建.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  等间距零填充:  PSNR={p_eq:.1f}dB")
print(f"  随机零填充:    PSNR={p_rand:.1f}dB")
print(f"  可变密度零填充: PSNR={p_vd:.1f}dB")


# ========================================================================
# ★ 步骤4：压缩感知MRI——ISTA算法（16.3.4节）
# ★原创设计：使用TV正则化的ISTA迭代重建
# ========================================================================
print("\n" + "=" * 60)
print("步骤4：压缩感知MRI——ISTA算法（16.3.4节）")
print("=" * 60)

def cs_mri_ista(y, mask, n_iter=50, lam=0.01, step_size=0.5):
    """
    ★ CS-MRI迭代重建（数据一致性+TV正则化梯度下降）
    min_x ||M_Ω * F * x - y||^2 + lambda * TV(x)

    由于MRI算子的特殊结构（F^H F = I），数据一致性步可高效实现：
    - 已采样k-space位置：用测量值替换
    - 未采样k-space位置：保持当前预测
    然后施加TV正则化作为投影/梯度步骤
    """
    mask_2d = mask.unsqueeze(0).expand(y.shape[0], -1)
    x = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(y * mask_2d, dim=(-2, -1))))  # 零填充初始化

    for k in range(n_iter):
        # ★ 数据一致性步：在已采样位置替换为测量值
        kspace_x = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        kspace_dc = mask_2d * y + (1 - mask_2d) * kspace_x  # 已采样用y，未采样用当前预测
        x_dc = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(kspace_dc, dim=(-2, -1))))

        # TV正则化梯度步（去噪步）
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

# 在可变密度掩码下测试CS-MRI
y_vd = mri_forward(x_true, mask_vd)
# 添加小噪声
y_vd_noisy = y_vd + 0.01 * torch.randn_like(y_vd)

print("  正在执行CS-MRI ISTA重建...")
recon_cs = cs_mri_ista(y_vd, mask_vd, n_iter=50, lam=0.01)
p_cs = psnr(phantom, recon_cs.numpy(), data_range=1.0)

# 收敛曲线
psnr_curve = []
mask_2d = mask_vd.unsqueeze(0).expand(n, -1)
x_ista = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(y_vd * mask_2d, dim=(-2, -1))))
for k in range(50):
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
axes[2].set_title(f'★ CS-MRI ISTA (R={R}, λ=0.005)\nPSNR={p_cs:.1f}dB\nTV稀疏性抑制伪影')
axes[2].axis('off')

axes[3].plot(range(len(psnr_curve)), psnr_curve, 'bo-')
axes[3].set_xlabel('ISTA迭代次数')
axes[3].set_ylabel('PSNR (dB)')
axes[3].set_title('ISTA收敛曲线')
axes[3].grid(True)

plt.suptitle('步骤4：压缩感知MRI——零填充 vs CS-ISTA（16.3.4节）', fontsize=13)
plt.tight_layout()
plt.savefig('步骤4_CS_MRI.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  零填充:   PSNR={p_vd:.1f}dB")
print(f"  CS-ISTA:  PSNR={p_cs:.1f}dB")


# ========================================================================
# ★ 步骤5：加速因子对重建质量的影响（16.3.3/16.3.4节）
# ========================================================================
print("\n" + "=" * 60)
print("步骤5：加速因子对重建质量的影响（16.3.3/16.3.4节）")
print("=" * 60)

R_list = [2, 4, 6, 8]
results_zf = []
results_cs = []

for R_val in R_list:
    mask_vd_R = mask_variable_density(n, R_val, seed=42)
    y_R = mri_forward(x_true, mask_vd_R)

    # 零填充
    recon_zf = zero_filled_recon(x_true, mask_vd_R)
    p_zf = psnr(phantom, recon_zf.numpy(), data_range=1.0)
    results_zf.append(p_zf)

    # CS-MRI
    recon_cs_R = cs_mri_ista(y_R, mask_vd_R, n_iter=50, lam=0.01)
    p_cs_R = psnr(phantom, recon_cs_R.numpy(), data_range=1.0)
    results_cs.append(p_cs_R)

    print(f"  R={R_val}: 保留{mask_vd_R.sum():.0f}行, 零填充={p_zf:.1f}dB, CS-ISTA={p_cs_R:.1f}dB")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(R_list, results_zf, 'ro-', label='零填充重建')
ax.plot(R_list, results_cs, 'bs-', label='CS-MRI ISTA')
ax.set_xlabel('加速因子 R')
ax.set_ylabel('PSNR (dB)')
ax.set_title('MRI重建质量 vs 加速因子\nR越大→欠采样越严重→需要更强的先验')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('步骤5_加速因子对比.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n实验16.2完成！")
