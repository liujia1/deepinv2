# -*- coding: utf-8 -*-
"""
实验16.1-2 ASTRA工具箱——CT正向模型与FBP

实验目的：使用工业级CT重建工具箱ASTRA实现CT正向模型和FBP重建，
          理解不同投影几何（平行束/扇形束）的差异和GPU加速的优势

素材来源：基于astra_operators_example.ipynb
运行前提：ASTRA仅支持Linux+CUDA；非Linux+CUDA环境自动回退到skimage版本
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import os
import sys
import io
import warnings
import logging

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
    SAVE_DIR = os.path.join(_gdrive, '实验16.1-2')
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

# ASTRA可用性检测
try:
    import astra
    ASTRA_AVAILABLE = True
except ImportError:
    ASTRA_AVAILABLE = False
    print("ASTRA工具箱未安装，将使用skimage回退版本")

np.random.seed(42)


# ========================================================================
# 工具函数
# ========================================================================

def random_ellipses(num_of_ellipses, size):
    """生成随机椭圆幻影

    参数:
        num_of_ellipses: 椭圆数量
        size: (H, W) 图像尺寸
    返回:
        归一化的numpy数组, 值域[0, 1]
    """
    ellipses = torch.zeros((num_of_ellipses, size[0], size[1]))
    x = torch.linspace(-size[0], size[1], size[1])
    y = torch.linspace(-size[0], size[1], size[0])[:, None]

    for k in range(num_of_ellipses):
        x0 = np.random.randint(-size[0], size[0])
        y0 = np.random.randint(-size[1], size[1])
        a = np.random.choice([i for i in range(-size[0], size[0]) if i != 0])
        b = np.random.choice([i for i in range(-size[1], size[1]) if i != 0])
        ellipses[k] = ((x - x0) / a) ** 2 + ((y - y0) / b) ** 2 <= 1
        ellipses[k][ellipses[k] == 1] = torch.rand(1)

    ellipses = torch.sum(ellipses, dim=0)
    return (ellipses / torch.max(ellipses)).numpy()


def compute_psnr(img1, img2, data_range=1.0):
    """计算PSNR，优先使用skimage指标，回退到自定义实现"""
    try:
        return psnr_metric(img1, img2, data_range=data_range)
    except Exception:
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(data_range ** 2 / mse)


# ========================================================================
# 步骤1：随机椭圆幻影生成（16.1.2节）
# ========================================================================
print("\n" + "=" * 60)
print("步骤1：随机椭圆幻影生成（16.1.2节）")
print("=" * 60)

size = (256, 256)
amount_of_ellipses = 50
ellipses = random_ellipses(amount_of_ellipses, size)

# Shepp-Logan幻影作为对比
shepp = shepp_logan_phantom()
from skimage.transform import resize as sk_resize
shepp = sk_resize(shepp, size, order=0, preserve_range=True, anti_aliasing=False)
shepp = (shepp - shepp.min()) / (shepp.max() - shepp.min())

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(shepp, cmap='gray')
axes[0].set_title('Shepp-Logan幻影（解析定义）')
axes[0].axis('off')

axes[1].imshow(ellipses, cmap='gray')
axes[1].set_title(f'随机椭圆幻影（{amount_of_ellipses}个椭圆, {size[0]}$\\times${size[1]}）')
axes[1].axis('off')

plt.suptitle('步骤1：CT幻影对比', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_随机椭圆phantom.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"  随机椭圆幻影尺寸: {ellipses.shape}")
print(f"  像素值范围: [{ellipses.min():.3f}, {ellipses.max():.3f}]")


# ========================================================================
# 步骤2：平行束投影与FBP重建
# ========================================================================
print("\n" + "=" * 60)
print("步骤2：平行束投影与FBP重建")
print("=" * 60)

if ASTRA_AVAILABLE:
    # ---- ASTRA版本 ----
    vol_geom = astra.create_vol_geom(size)
    num_of_lines = 512
    amount_of_angles = 720
    angles = np.linspace(0, np.pi, amount_of_angles)

    proj_geom = astra.create_proj_geom('parallel', 1.0, num_of_lines, angles)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    projector_type = 'cuda' if device == 'cuda' else 'line'
    FBP_type = 'FBP_CUDA' if device == 'cuda' else 'FBP'
    print(f"  ASTRA设备: {device}, FBP类型: {FBP_type}")

    proj_id_parall = astra.create_projector(projector_type, proj_geom, vol_geom)
    parall_sinogram_id, parall_sinogram = astra.create_sino(ellipses, proj_id_parall)
    astra.data2d.delete(proj_id_parall)

    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict(FBP_type)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = parall_sinogram_id
    cfg['option'] = {'FilterType': 'ram-lak'}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    parall_rec = astra.data2d.get(rec_id)
    parall_rec = np.maximum(0, parall_rec)
    if parall_rec.max() > 0:
        parall_rec = parall_rec / parall_rec.max()

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(parall_sinogram_id)

    parall_psnr = compute_psnr(ellipses, parall_rec, data_range=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ellipses, cmap='gray')
    axes[0].set_title('原始幻影')
    axes[0].axis('off')

    im1 = axes[1].imshow(parall_sinogram, cmap='gray', aspect='auto')
    axes[1].set_title(f'平行束Sinogram\n({amount_of_angles}角度, {num_of_lines}探测器)')
    axes[1].set_xlabel('投影角度 $\\theta$')
    axes[1].set_ylabel('探测器位置 s')
    plt.colorbar(im1, ax=axes[1])

    axes[2].imshow(parall_rec, cmap='gray')
    axes[2].set_title(f'FBP重建\nPSNR={parall_psnr:.1f} dB')
    axes[2].axis('off')

    plt.suptitle('步骤2：ASTRA平行束投影与FBP重建', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤2_平行束对比.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"  平行束FBP PSNR: {parall_psnr:.1f} dB")

else:
    # ---- skimage回退版本 ----
    theta_full = np.linspace(0., 180., 360, endpoint=False)
    sinogram_full = radon(ellipses, theta=theta_full, circle=True)
    recon_full = iradon(sinogram_full, theta=theta_full, circle=True, filter_name='ramp')
    recon_full = np.clip(recon_full, 0, None)
    if recon_full.max() > 0:
        recon_full = recon_full / recon_full.max()

    parall_psnr = compute_psnr(ellipses, recon_full, data_range=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ellipses, cmap='gray')
    axes[0].set_title('原始幻影')
    axes[0].axis('off')

    im1 = axes[1].imshow(sinogram_full, cmap='gray', aspect='auto')
    axes[1].set_title(f'平行束Sinogram\n({len(theta_full)}角度)')
    axes[1].set_xlabel('投影角度 $\\theta$ ($^{\\circ}$)')
    axes[1].set_ylabel('探测器位置 s')
    plt.colorbar(im1, ax=axes[1])

    axes[2].imshow(recon_full, cmap='gray')
    axes[2].set_title(f'FBP重建 (skimage)\nPSNR={parall_psnr:.1f} dB')
    axes[2].axis('off')

    plt.suptitle('步骤2：skimage平行束投影与FBP重建', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤2_平行束对比.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"  平行束FBP PSNR (skimage): {parall_psnr:.1f} dB")


# ========================================================================
# 步骤3：扇形束投影
# ========================================================================
print("\n" + "=" * 60)
print("步骤3：扇形束投影")
print("=" * 60)

if ASTRA_AVAILABLE:
    # ---- ASTRA版本 ----
    num_of_lines = 512
    amount_of_angles = 720
    angles = np.linspace(0, 2 * np.pi, amount_of_angles)
    SOD = 250  # 源到物体中心距离
    SDD = 260  # 源到探测器距离

    proj_geom = astra.create_proj_geom('fanflat', 1.0, num_of_lines, angles, SOD, SDD - SOD)
    # 扇形束几何需要使用'line_fanflat'投影器类型（平行束用'line'，这里必须区分）
    fan_projector_type = 'cuda' if torch.cuda.is_available() else 'line_fanflat'
    proj_id_full = astra.create_projector(fan_projector_type, proj_geom, vol_geom)

    full_sinogram_id, full_sinogram = astra.create_sino(ellipses, proj_id_full)
    astra.data2d.delete(proj_id_full)

    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict(FBP_type)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = full_sinogram_id
    cfg['option'] = {'FilterType': 'ram-lak'}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    full_rec = astra.data2d.get(rec_id)
    full_rec = np.maximum(0, full_rec)
    if full_rec.max() > 0:
        full_rec = full_rec / full_rec.max()

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(full_sinogram_id)

    fan_psnr = compute_psnr(ellipses, full_rec, data_range=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ellipses, cmap='gray')
    axes[0].set_title('原始幻影')
    axes[0].axis('off')

    axes[1].imshow(full_sinogram, cmap='gray', aspect='auto')
    axes[1].set_title(f'扇形束Sinogram\n(0-360$^\\circ$, {amount_of_angles}角度)')
    axes[1].set_xlabel('投影角度 $\\theta$')
    axes[1].set_ylabel('探测器位置 s')

    axes[2].imshow(full_rec, cmap='gray')
    axes[2].set_title(f'扇形束FBP重建\nPSNR={fan_psnr:.1f} dB')
    axes[2].axis('off')

    plt.suptitle('步骤3：ASTRA扇形束投影与FBP重建', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤3_扇形束投影.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"  扇形束FBP PSNR: {fan_psnr:.1f} dB")
    print(f"  SOD={SOD}, SDD={SDD}, 放大倍率M={SDD/SOD:.3f}")

else:
    # ---- skimage回退版本：概念讲解 + 角度采样对比 ----
    print("  扇形束投影需要ASTRA工具箱（Linux+CUDA环境）。")
    print("  关键参数:")
    print("    SOD = 250  # 源到物体中心距离 (Source-Object Distance)")
    print("    SDD = 260  # 源到探测器距离 (Source-Detector Distance)")
    print("    proj_geom = astra.create_proj_geom('fanflat', 1.0, 512, angles, SOD, SDD-SOD)")
    print()
    print("  扇形束 vs 平行束的区别:")
    print("    - 扇形束：射线从点光源发散")
    print("    - 平行束：射线互相平行")
    print("    - 扇形束sinogram：正弦曲线不对称（发散效应）")
    print("    - 扇形束有放大效应：M = SDD/SOD")
    print()
    print("  注意：由于skimage仅支持平行束几何，下方对比图无法展示扇形束的")
    print("        发散效应和非对称sinogram，仅作角度采样数量的对比展示，")
    print("        扇形束的真实效果需要ASTRA环境才能观察到。")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 平行束sinogram（标准0-180°）
    theta_par = np.linspace(0., 180., 180, endpoint=False)
    sino_par = radon(ellipses, theta=theta_par, circle=True)
    axes[0].imshow(sino_par, cmap='gray', aspect='auto')
    axes[0].set_title('平行束Sinogram (0-180°)\n对称正弦曲线')
    axes[0].set_xlabel('投影角度 $\\theta$ ($^{\\circ}$)')
    axes[0].set_ylabel('探测器位置 s')

    # 0-360°采样（仍为平行束，仅增加角度数量）
    theta_fan = np.linspace(0., 360., 720, endpoint=False)
    sino_fan = radon(ellipses, theta=theta_fan, circle=True)
    axes[1].imshow(sino_fan, cmap='gray', aspect='auto')
    axes[1].set_title('平行束Sinogram (0-360°)\n（注意：仍是对称的，非真实扇形束）')
    axes[1].set_xlabel('投影角度 $\\theta$ ($^{\\circ}$)')
    axes[1].set_ylabel('探测器位置 s')

    plt.suptitle('步骤3：扇形束几何概念对比——skimage回退', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤3_扇形束投影.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print("  扇形束对比图已保存（skimage模拟版本）")


# ========================================================================
# 步骤4：FBP滤波器对比
# ========================================================================
print("\n" + "=" * 60)
print("步骤4：FBP滤波器对比")
print("=" * 60)

if ASTRA_AVAILABLE:
    # ---- ASTRA版本：使用扇形束sinogram ----
    num_of_lines = 512
    amount_of_angles = 720
    angles = np.linspace(0, 2 * np.pi, amount_of_angles)
    SOD = 250
    SDD = 260

    proj_geom = astra.create_proj_geom('fanflat', 1.0, num_of_lines, angles, SOD, SDD - SOD)
    # 扇形束几何需要使用'line_fanflat'投影器类型
    fan_projector_type = 'cuda' if torch.cuda.is_available() else 'line_fanflat'
    proj_id = astra.create_projector(fan_projector_type, proj_geom, vol_geom)
    sino_id, sino = astra.create_sino(ellipses, proj_id)
    astra.data2d.delete(proj_id)

    filter_types = ['none', 'ram-lak', 'shepp-logan', 'cosine']
    filter_labels = ['无滤波', 'Ram-Lak $|\\omega|$', 'Shepp-Logan $|\\omega|\\mathrm{sinc}$', 'Cosine $|\\omega|\\cos$']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    filter_psnrs = []

    for idx, (filt, flabel) in enumerate(tqdm(zip(filter_types, filter_labels),
                                               total=len(filter_types), desc="  滤波器对比")):
        rec_id = astra.data2d.create('-vol', vol_geom)
        cfg = astra.astra_dict(FBP_type)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['option'] = {'FilterType': filt}

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data2d.get(rec_id)
        rec = np.maximum(0, rec)
        if rec.max() > 0:
            rec = rec / rec.max()

        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)

        p = compute_psnr(ellipses, rec, data_range=1.0)
        filter_psnrs.append(p)

        axes[idx].imshow(rec, cmap='gray')
        axes[idx].set_title(f'{flabel}\nPSNR={p:.1f} dB')
        axes[idx].axis('off')

    astra.data2d.delete(sino_id)

    plt.suptitle('步骤4：FBP滤波器对比——扇形束', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤4_滤波器对比.png'), dpi=150, bbox_inches='tight')
    plt.show()

    for filt, p in zip(filter_types, filter_psnrs):
        print(f"  滤波器 '{filt}': PSNR = {p:.1f} dB")

else:
    # ---- skimage回退版本 ----
    filters = [None, 'ramp', 'shepp-logan', 'cosine']
    filter_labels = ['无滤波', 'Ramp $|\\omega|$', 'Shepp-Logan $|\\omega|\\mathrm{sinc}$', 'Cosine $|\\omega|\\cos$']

    theta = np.linspace(0., 180., 360, endpoint=False)
    sino = radon(ellipses, theta=theta, circle=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    filter_psnrs = []

    for idx, (filt, flabel) in enumerate(tqdm(zip(filters, filter_labels),
                                               total=len(filters), desc="  滤波器对比")):
        recon = iradon(sino, theta=theta, circle=True, filter_name=filt)
        recon = np.clip(recon, 0, None)
        if recon.max() > 0:
            recon = recon / recon.max()

        p = compute_psnr(ellipses, recon, data_range=1.0)
        filter_psnrs.append(p)

        axes[idx].imshow(recon, cmap='gray')
        axes[idx].set_title(f'{flabel}\nPSNR={p:.1f} dB')
        axes[idx].axis('off')

    plt.suptitle('步骤4：FBP滤波器对比——平行束', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤4_滤波器对比.png'), dpi=150, bbox_inches='tight')
    plt.show()

    filter_names_display = ['None', 'ramp', 'shepp-logan', 'cosine']
    for filt, p in zip(filter_names_display, filter_psnrs):
        print(f"  滤波器 '{filt}': PSNR = {p:.1f} dB")


# ========================================================================
# ASTRA内存管理提醒
# ========================================================================
if ASTRA_AVAILABLE:
    print("\n" + "-" * 60)
    print("ASTRA内存管理提醒:")
    print("  ASTRA使用C++后端，必须手动释放内存：")
    print("    astra.data2d.delete(id)")
    print("    astra.algorithm.delete(id)")
    print("    astra.projector.delete(id)")
    print("  未释放会导致GPU显存泄漏！")


# ========================================================================
# ASTRA vs skimage总结
# ========================================================================
print("\n" + "=" * 60)
print("ASTRA vs skimage 总结")
print("=" * 60)
print()
print("skimage:")
print("  + 跨平台（Windows/macOS/Linux）")
print("  + 简单API（radon/iradon）")
print("  - 仅支持平行束")
print("  - 无GPU加速")
print("  - 探测器数量固定")
print()
print("ASTRA（需Linux+CUDA）:")
print("  + 支持扇形束/锥形束几何")
print("  + GPU加速（50-100倍提速）")
print("  + 自定义探测器数量")
print("  + 多种重建算法（FBP/SIRT/SART/CGLS）")
print("  - 仅Linux+CUDA")
print("  - 需手动内存管理")

print(f"\n实验16.1-2完成！结果已保存至: {SAVE_DIR}")
