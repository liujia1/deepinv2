# -*- coding: utf-8 -*-
"""
实验16.2-2 ASTRA工具箱——CT不适定性演示

实验目的：使用ASTRA/skimage对比全角、稀疏角度和有限角度CT的重建效果，
          直观验证两种不适定性的差异——稀疏角度产生混叠，有限角度丢失不可见边缘

素材来源：基于16.5.py步骤4
运行前提：ASTRA仅支持Linux+CUDA；非Linux+CUDA环境自动回退到skimage版本
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, resize
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
    SAVE_DIR = os.path.join(_gdrive, '实验16.2-2')
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

try:
    import astra
    ASTRA_AVAILABLE = True
except ImportError:
    ASTRA_AVAILABLE = False
    print("ASTRA工具箱未安装，将使用skimage回退版本")

np.random.seed(42)


# ========================================================================
# 辅助函数
# ========================================================================

def random_ellipses(num_of_ellipses, size):
    """生成随机椭圆幻影（用于ASTRA版本）"""
    ellipses = torch.zeros((num_of_ellipses, size[0], size[1]))
    x = torch.linspace(-size[0], size[1], size[1])
    y = torch.linspace(-size[0], size[1], size[0])[:, None]

    for k in range(num_of_ellipses):
        x0 = np.random.randint(-size[0] // 2, size[0] // 2)
        y0 = np.random.randint(-size[1] // 2, size[1] // 2)
        a = np.random.randint(size[0] // 8, size[0] // 2)
        b = np.random.randint(size[1] // 8, size[1] // 2)
        if a == 0 or b == 0:
            continue
        ellipses[k] = ((x - x0) / a) ** 2 + ((y - y0) / b) ** 2 <= 1
        ellipses[k][ellipses[k] == 1] = torch.rand(1)

    ellipses = torch.sum(ellipses, dim=0)
    return (ellipses / torch.max(ellipses)).numpy()


def compute_psnr(ref, test):
    """计算PSNR"""
    return psnr_metric(ref, test, data_range=ref.max() - ref.min())


def normalize_reconstruction(rec):
    """将重建结果裁剪并归一化到[0,1]"""
    rec = np.maximum(0, rec)
    if rec.max() > 0:
        rec = rec / rec.max()
    return rec


# ========================================================================
# ASTRA版本：扇形束CT
# ========================================================================

def astra_fanbeam_reconstruct(phantom, angles, vol_geom, num_of_lines=512,
                               SOD=250, SDD=260, filter_type='ram-lak'):
    """
    ASTRA扇形束投影与FBP重建

    参数:
        phantom: 原始图像 (2D numpy array)
        angles: 投影角度数组 (弧度制)
        vol_geom: ASTRA体积几何
        num_of_lines: 探测器线数
        SOD: 源到物体中心距离
        SDD: 源到探测器距离
        filter_type: FBP滤波器类型

    返回:
        sinogram: 正弦图
        reconstruction: 重建图像
    """
    proj_geom = astra.create_proj_geom('fanflat', 1.0, num_of_lines, angles, SOD, SDD - SOD)

    device = 'cuda' if torch.cuda.is_available() else 'line_fanflat'
    FBP_type = 'FBP_CUDA' if device == 'cuda' else 'FBP'

    proj_id = astra.create_projector(device, proj_geom, vol_geom)
    sinogram_id, sinogram = astra.create_sino(phantom, proj_id)
    astra.data2d.delete(proj_id)

    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict(FBP_type)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = {'FilterType': filter_type}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    reconstruction = astra.data2d.get(rec_id)
    reconstruction = normalize_reconstruction(reconstruction)

    # 清理ASTRA内存
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)

    return sinogram, reconstruction


# ========================================================================
# skimage回退版本：平行束CT
# ========================================================================

def skimage_parallel_reconstruct(phantom, theta_deg, filter_name='ramp', sinogram=None):
    """
    skimage平行束投影与FBP重建

    参数:
        phantom: 原始图像 (2D numpy array)（仅用于形状参考）
        theta_deg: 投影角度数组 (角度制)
        filter_name: FBP滤波器名称
        sinogram: 预生成的sinogram（IC-free）。如果为None，则用radon直接生成（不推荐）

    返回:
        sinogram: 正弦图
        reconstruction: 重建图像
    """
    if sinogram is None:
        sinogram = radon(phantom, theta=theta_deg, circle=True)
    reconstruction = iradon(sinogram, theta=theta_deg, circle=True, filter_name=filter_name)
    reconstruction = normalize_reconstruction(reconstruction)
    return sinogram, reconstruction


# ========================================================================
# 主实验
# ========================================================================

print(f"\n{'=' * 60}")
print(f"实验16.2-2: ASTRA工具箱——CT不适定性演示")
print(f"{'=' * 60}")
print(f"运行模式: {'ASTRA扇形束 (CUDA)' if ASTRA_AVAILABLE and torch.cuda.is_available() else 'ASTRA扇形束 (CPU)' if ASTRA_AVAILABLE else 'skimage平行束 (回退)'}")


# ========================================================================
# 步骤1：生成幻影
# ========================================================================
print("\n" + "=" * 60)
print("步骤1：生成幻影")
print("=" * 60)

size = (256, 256)

if ASTRA_AVAILABLE:
    phantom = random_ellipses(50, size)
    phantom_name = "随机椭圆幻影 (50个椭圆)"
    print(f"  使用ASTRA版本 -> 随机椭圆幻影 ({size[0]}x{size[1]})")
else:
    phantom = shepp_logan_phantom()
    from skimage.transform import resize
    phantom = resize(phantom, size, order=0, preserve_range=True, anti_aliasing=False)
    phantom = phantom / phantom.max()
    phantom_name = "Shepp-Logan幻影"
    print(f"  使用skimage回退版本 -> Shepp-Logan幻影 ({size[0]}x{size[1]})")

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(phantom, cmap='gray')
ax.set_title(f'{phantom_name}\n{size[0]}$\\times${size[1]}')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_幻影.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  幻影已生成并保存。")


# ========================================================================
# 避免inverse crime：数据生成使用高分辨率模型，反演使用低分辨率模型
# ========================================================================
n_data = (512, 512)  # 数据生成分辨率（2倍）

# 高分辨率phantom（用于数据生成）
if ASTRA_AVAILABLE:
    phantom_fine = random_ellipses(50, n_data)
else:
    from skimage.transform import resize as sk_resize
    phantom_fine = sk_resize(phantom, n_data, order=1, preserve_range=True, anti_aliasing=True)
    phantom_fine = phantom_fine / phantom_fine.max()

def generate_sinogram_ic_free(phantom_fine, theta, n_recon_size, noise_sigma=0.0):
    """避免inverse crime的sinogram生成（skimage路径）：
    在n_data高分辨率下用radon正向投影，添加噪声后
    resize到n_recon对应的探测器尺寸。"""
    sinogram_fine = radon(phantom_fine, theta=theta, circle=True)
    if noise_sigma > 0:
        sinogram_fine = sinogram_fine + noise_sigma * np.random.randn(*sinogram_fine.shape)
    # circle=True时探测器尺寸=图像尺寸
    target_det = n_recon_size[0]
    from skimage.transform import resize as sk_resize
    sinogram_recon = sk_resize(sinogram_fine, (target_det, len(theta)), order=1, preserve_range=True, anti_aliasing=True)
    return sinogram_recon


# ========================================================================
# 步骤2：三种CT配置对比——全角/稀疏角度/有限角度
# ========================================================================
print("\n" + "=" * 60)
print("步骤2：三种CT配置对比——全角/稀疏角度/有限角度")
print("  稀疏角度CT的奇异性分析（稀疏角度→混叠伪影）")
print("  有限角CT的奇异值指数衰减（缺失角度→边缘丢失）")
print("=" * 60)

if ASTRA_AVAILABLE:
    # ASTRA参数
    num_of_lines = 512
    SOD = 250
    SDD = 260
    vol_geom = astra.create_vol_geom(size)

    # 三种配置：全角(360)、稀疏角度(30)、有限角度(0-90度, 180角度)
    configs = [
        {
            'name': '全角CT',
            'name_en': 'Full Angle',
            'angles': np.linspace(0, 2 * np.pi, 360, endpoint=False),
            'desc': '360角度，覆盖0$^{\\circ}$-360$^{\\circ}$',
            'angle_info': '$\\theta \\in [0^{\\circ}, 360^{\\circ})$, 360角度',
        },
        {
            'name': '稀疏角度CT',
            'name_en': 'Sparse Angle',
            'angles': np.linspace(0, 2 * np.pi, 30, endpoint=False),
            'desc': '30角度，覆盖0$^{\\circ}$-360$^{\\circ}$（间隔大）',
            'angle_info': '$\\theta \\in [0^{\\circ}, 360^{\\circ})$, 30角度（稀疏）',
        },
        {
            'name': '有限角度CT',
            'name_en': 'Limited Angle',
            'angles': np.linspace(0, np.pi / 2, 180, endpoint=False),
            'desc': '180角度，仅覆盖0$^{\\circ}$-90$^{\\circ}$',
            'angle_info': '$\\theta \\in [0^{\\circ}, 90^{\\circ}]$, 180角度（有限角）',
        },
    ]

    results = {}
    for cfg in tqdm(configs, desc='  ASTRA重建进度'):
        sinogram, reconstruction = astra_fanbeam_reconstruct(
            phantom, cfg['angles'], vol_geom,
            num_of_lines=num_of_lines, SOD=SOD, SDD=SDD,
            filter_type='ram-lak'
        )
        p = compute_psnr(phantom, reconstruction)
        results[cfg['name']] = {
            'sinogram': sinogram,
            'reconstruction': reconstruction,
            'psnr': p,
            'desc': cfg['desc'],
            'angle_info': cfg['angle_info'],
            'name_en': cfg['name_en'],
        }

else:
    # skimage回退：平行束
    # 三种配置：全角(360)、稀疏角度(30)、有限角度(0-90度, 180角度)
    configs = [
        {
            'name': '全角CT',
            'name_en': 'Full Angle',
            'theta': np.linspace(0, 180, 360, endpoint=False),
            'desc': '360角度，覆盖0$^{\\circ}$-180$^{\\circ}$',
            'angle_info': '$\\theta \\in [0^{\\circ}, 180^{\\circ})$, 360角度',
        },
        {
            'name': '稀疏角度CT',
            'name_en': 'Sparse Angle',
            'theta': np.linspace(0, 180, 30, endpoint=False),
            'desc': '30角度，覆盖0$^{\\circ}$-180$^{\\circ}$（间隔大）',
            'angle_info': '$\\theta \\in [0^{\\circ}, 180^{\\circ})$, 30角度（稀疏）',
        },
        {
            'name': '有限角度CT',
            'name_en': 'Limited Angle',
            'theta': np.linspace(0, 90, 180, endpoint=False),
            'desc': '180角度，仅覆盖0$^{\\circ}$-90$^{\\circ}$',
            'angle_info': '$\\theta \\in [0^{\\circ}, 90^{\\circ}]$, 180角度（有限角）',
        },
    ]

    results = {}
    for cfg in tqdm(configs, desc='  skimage重建进度'):
        # IC-free：高分辨率生成sinogram，低分辨率重建
        sino_ic = generate_sinogram_ic_free(phantom_fine, cfg['theta'], size)
        sinogram, reconstruction = skimage_parallel_reconstruct(
            phantom, cfg['theta'], filter_name='ramp', sinogram=sino_ic
        )
        p = compute_psnr(phantom, reconstruction)
        results[cfg['name']] = {
            'sinogram': sinogram,
            'reconstruction': reconstruction,
            'psnr': p,
            'desc': cfg['desc'],
            'angle_info': cfg['angle_info'],
            'name_en': cfg['name_en'],
        }


# ========================================================================
# 绘制三种配置对比图
# ========================================================================
names = ['全角CT', '稀疏角度CT', '有限角度CT']

fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for idx, name in enumerate(names):
    r = results[name]

    # 第一列：原始幻影（仅第一行显示）
    if idx == 0:
        axes[idx, 0].imshow(phantom, cmap='gray')
        axes[idx, 0].set_title(f'原始幻影\n{phantom_name}')
    else:
        axes[idx, 0].imshow(phantom, cmap='gray')
        axes[idx, 0].set_title('原始幻影')
    axes[idx, 0].axis('off')

    # 第二列：正弦图
    sino = r['sinogram']
    axes[idx, 1].imshow(sino, cmap='gray', aspect='auto')
    axes[idx, 1].set_title(f'{name} — 正弦图\n{r["angle_info"]}')
    axes[idx, 1].set_xlabel('$\\theta$')
    axes[idx, 1].set_ylabel('探测器')

    # 第三列：FBP重建
    rec = r['reconstruction']
    axes[idx, 2].imshow(rec, cmap='gray')
    p_val = r['psnr']
    # 为每种配置添加不同的伪影描述
    if name == '全角CT':
        artifact_desc = '高质量重建'
    elif name == '稀疏角度CT':
        artifact_desc = '混叠伪影（条纹）'
    else:
        artifact_desc = '不可见边缘丢失'
    axes[idx, 2].set_title(f'FBP重建\nPSNR={p_val:.1f}dB — {artifact_desc}')
    axes[idx, 2].axis('off')

mode_label = 'ASTRA扇形束' if ASTRA_AVAILABLE else 'skimage平行束（回退）'
plt.suptitle(
    f'步骤2：三种CT配置对比 — {mode_label}\n'
    f'稀疏角度$\\rightarrow$混叠伪影  |  有限角度$\\rightarrow$边缘丢失',
    fontsize=13
)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_三种CT配置对比.png'), dpi=100, bbox_inches='tight')
plt.close()

# 打印结果
print(f"\n  {'配置':<16} {'PSNR (dB)':<12} {'伪影特征'}")
print(f"  {'-'*50}")
for name in names:
    r = results[name]
    if name == '全角CT':
        artifact = '高质量重建'
    elif name == '稀疏角度CT':
        artifact = '混叠伪影（条纹）'
    else:
        artifact = '不可见边缘丢失'
    print(f"  {name:<16} {r['psnr']:<12.1f} {artifact}")


# ========================================================================
# 步骤3：差异图与伪影分析——直观对比两种不适定性
# ========================================================================
print("\n" + "=" * 60)
print("步骤3：差异图与伪影分析——直观对比两种不适定性")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for row, name in enumerate(['稀疏角度CT', '有限角度CT']):
    r = results[name]
    rec = r['reconstruction']
    diff = np.abs(phantom - rec)

    # 重建结果
    axes[row, 0].imshow(rec, cmap='gray')
    axes[row, 0].set_title(f'{name} FBP重建\nPSNR={r["psnr"]:.1f}dB')
    axes[row, 0].axis('off')

    # 差异图（增强显示）
    diff_enhanced = diff / (diff.max() + 1e-10)
    axes[row, 1].imshow(diff_enhanced, cmap='hot')
    axes[row, 1].set_title(f'差异图 $|x - \\hat{{x}}|$\n（增强显示）')
    axes[row, 1].axis('off')

    # 差异图直方图
    axes[row, 2].hist(diff.ravel(), bins=100, color='steelblue', alpha=0.8)
    axes[row, 2].set_xlabel('差异值 $|x - \\hat{x}|$')
    axes[row, 2].set_ylabel('像素数')
    mean_diff = diff.mean()
    max_diff = diff.max()
    if name == '稀疏角度CT':
        axes[row, 2].set_title(f'差异分布（稀疏角度）\n均值={mean_diff:.4f}, 最大={max_diff:.3f}')
    else:
        axes[row, 2].set_title(f'差异分布（有限角度）\n均值={mean_diff:.4f}, 最大={max_diff:.3f}')

plt.suptitle(
    '步骤3：差异图分析\n'
    '稀疏角度：全局条纹伪影（混叠，$\\sigma$均匀分布）\n'
    '有限角度：特定方向边缘丢失（信息缺失不可恢复）',
    fontsize=12
)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_差异图分析.png'), dpi=100, bbox_inches='tight')
plt.close()

print(f"  稀疏角度CT: 差异均值={np.abs(phantom - results['稀疏角度CT']['reconstruction']).mean():.4f}")
print(f"  有限角度CT: 差异均值={np.abs(phantom - results['有限角度CT']['reconstruction']).mean():.4f}")
print("  分析：稀疏角度产生全局混叠条纹，有限角度导致特定方向边缘丢失")


# ========================================================================
# 步骤4：角度数量/范围扫描——不适定性的量化
# ========================================================================
print("\n" + "=" * 60)
print("步骤4：角度数量/范围扫描——不适定性的量化")
print("=" * 60)

# 扫描1：稀疏角度——改变投影角度数量
sparse_angle_counts = [10, 20, 30, 45, 60, 90, 120, 180, 360]
psnr_sparse_curve = []

if ASTRA_AVAILABLE:
    for na in tqdm(sparse_angle_counts, desc='  稀疏角度扫描'):
        angles = np.linspace(0, 2 * np.pi, na, endpoint=False)
        sino, rec = astra_fanbeam_reconstruct(
            phantom, angles, vol_geom,
            num_of_lines=num_of_lines, SOD=SOD, SDD=SDD,
            filter_type='ram-lak'
        )
        psnr_sparse_curve.append(compute_psnr(phantom, rec))
else:
    for na in tqdm(sparse_angle_counts, desc='  稀疏角度扫描'):
        theta = np.linspace(0, 180, na, endpoint=False)
        # IC-free：高分辨率生成sinogram
        sino_ic = generate_sinogram_ic_free(phantom_fine, theta, size)
        sino, rec = skimage_parallel_reconstruct(phantom, theta, filter_name='ramp', sinogram=sino_ic)
        psnr_sparse_curve.append(compute_psnr(phantom, rec))

# 扫描2：有限角度——改变角度覆盖范围
limited_ranges = [30, 45, 60, 90, 120, 150, 180]
psnr_limited_curve = []

if ASTRA_AVAILABLE:
    for angle_range_deg in tqdm(limited_ranges, desc='  有限角度扫描'):
        angle_range_rad = np.deg2rad(angle_range_deg)
        angles = np.linspace(0, angle_range_rad, 180, endpoint=False)
        sino, rec = astra_fanbeam_reconstruct(
            phantom, angles, vol_geom,
            num_of_lines=num_of_lines, SOD=SOD, SDD=SDD,
            filter_type='ram-lak'
        )
        psnr_limited_curve.append(compute_psnr(phantom, rec))
else:
    for angle_range_deg in tqdm(limited_ranges, desc='  有限角度扫描'):
        theta = np.linspace(0, angle_range_deg, 180, endpoint=False)
        # IC-free：高分辨率生成sinogram
        sino_ic = generate_sinogram_ic_free(phantom_fine, theta, size)
        sino, rec = skimage_parallel_reconstruct(phantom, theta, filter_name='ramp', sinogram=sino_ic)
        psnr_limited_curve.append(compute_psnr(phantom, rec))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 稀疏角度PSNR曲线
ax1.plot(sparse_angle_counts, psnr_sparse_curve, 'bo-', markersize=6)
ax1.set_xlabel('投影角度数')
ax1.set_ylabel('PSNR (dB)')
ax1.set_title('稀疏角度CT：角度数 vs 重建质量\n角度越少$\\rightarrow$信息不足$\\rightarrow$混叠伪影加重')
ax1.grid(True, alpha=0.3)

# 标注关键点
for i, (na, p) in enumerate(zip(sparse_angle_counts, psnr_sparse_curve)):
    if na in [30, 360]:
        ax1.annotate(f'{na}角: {p:.1f}dB', (na, p),
                     textcoords="offset points", xytext=(10, 5), fontsize=8)

# 有限角度PSNR曲线
ax2.plot(limited_ranges, psnr_limited_curve, 'rs-', markersize=6)
ax2.set_xlabel('角度覆盖范围 ($^{\\circ}$)')
ax2.set_ylabel('PSNR (dB)')
ax2.set_title('有限角度CT：角度范围 vs 重建质量\n缺失角度$\\rightarrow$频域缺口$\\rightarrow$边缘不可恢复')
ax2.grid(True, alpha=0.3)

# 标注关键点
for i, (ar, p) in enumerate(zip(limited_ranges, psnr_limited_curve)):
    if ar in [90, 180]:
        ax2.annotate(f'{ar}$^{{\\circ}}$: {p:.1f}dB', (ar, p),
                     textcoords="offset points", xytext=(10, 5), fontsize=8)

mode_label = 'ASTRA扇形束' if ASTRA_AVAILABLE else 'skimage平行束'
plt.suptitle(f'步骤4：CT不适定性的量化 — {mode_label}', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_角度扫描量化.png'), dpi=100, bbox_inches='tight')
plt.close()

# 打印结果
print(f"\n  稀疏角度扫描:")
print(f"  {'角度数':<10} {'PSNR (dB)'}")
print(f"  {'-'*25}")
for na, p in zip(sparse_angle_counts, psnr_sparse_curve):
    print(f"  {na:<10} {p:.1f}")

print(f"\n  有限角度扫描:")
print(f"  {'角度范围(°)':<14} {'PSNR (dB)'}")
print(f"  {'-'*30}")
for ar, p in zip(limited_ranges, psnr_limited_curve):
    print(f"  {ar:<14} {p:.1f}")

print("\n  关键观察:")
print("  - 稀疏角度：角度数增加时PSNR稳步提升，但总有混叠伪影残留")
print("  - 有限角度：角度范围扩大时PSNR快速提升，但缺失角度的边缘信息不可恢复")
print("  - 两种不适定性本质不同：稀疏角度是采样不足，有限角度是信息缺失")


# ========================================================================
# ASTRA内存管理提示
# ========================================================================
if ASTRA_AVAILABLE:
    print("\n" + "=" * 60)
    print("ASTRA内存管理提示")
    print("=" * 60)
    print("  ASTRA使用C++后端，必须手动释放内存：")
    print("    astra.data2d.delete(id)     # 删除数据对象")
    print("    astra.algorithm.delete(id)   # 删除算法对象")
    print("    astra.projector.delete(id)   # 删除投影器对象")
    print("  未释放将导致GPU内存泄漏！")


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
    "experiment": "实验16.2-2 ASTRA工具箱——CT不适定性演示",
    "ASTRA_AVAILABLE": ASTRA_AVAILABLE,
    "步骤2_三种CT配置_PSNR_dB": {name: round(float(results[name]['psnr']), 2) for name in names},
    "步骤4_稀疏角度扫描": {
        "角度数": sparse_angle_counts,
        "PSNR_dB": [round(float(v), 2) for v in psnr_sparse_curve],
    },
    "步骤4_有限角度扫描": {
        "角度范围": limited_ranges,
        "PSNR_dB": [round(float(v), 2) for v in psnr_limited_curve],
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")

print(f"\n实验16.2-2完成！所有图片已保存至: {SAVE_DIR}")
