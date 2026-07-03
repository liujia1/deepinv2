"""
实验16.5：ASTRA工具箱CT算子演示（M11）——ASTRA版本
对应章节：16.1.2（Radon变换）、16.2.1（全角CT）、16.2.2（有限角CT）
参考素材：astra_operators_example.ipynb（Teaching Unit 4）

注意：本文件需要Linux+CUDA环境运行ASTRA工具箱。
Windows/macOS用户请使用16.5_fallback.py（skimage回退版本）。
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    import astra
    ASTRA_AVAILABLE = True
except ImportError:
    ASTRA_AVAILABLE = False
    print("ASTRA not installed. Install with: pip install astra-toolbox")
    print("Falling back to skimage version (16.5_fallback.py)")
    print("Run: python 16.5_fallback.py")


def random_ellipses(num_of_ellipses, size):
    ellipses = torch.zeros((num_of_ellipses, size[0], size[1]))
    x = torch.linspace(-size[0], size[1], size[1])
    y = torch.linspace(-size[0], size[1], size[0])[:, None]

    for k in range(num_of_ellipses):
        x0 = np.random.randint(-size[0], size[0])
        y0 = np.random.randint(-size[1], size[1])
        a = np.random.randint(-size[0], size[0])
        b = np.random.randint(-size[1], size[1])
        ellipses[k] = ((x - x0) / a)**2 + ((y - y0) / b)**2 <= 1
        ellipses[k][ellipses[k] == 1] = torch.rand(1)

    ellipses = torch.sum(ellipses, dim=0)
    return (ellipses / torch.max(ellipses)).numpy()


def psnr(image1, image2):
    return 10 * np.log10((np.max(image1)) / (((image1 - image2)**2).mean() + 1e-10))


if not ASTRA_AVAILABLE:
    print("\nASTRA not available. Exiting.")
    exit(0)


device = 'cuda' if torch.cuda.is_available() else ['line', 'line_fanflat']
FBP_type = 'FBP_CUDA' if device == 'cuda' else 'FBP'
print(f"Using device: {device}, FBP type: {FBP_type}")


print("\n=== Step 1: Random Ellipse Phantom Generation ===")
size = (256, 256)
amount_of_ellipses = 50
np.random.seed(42)
ellipses = random_ellipses(amount_of_ellipses, size)

plt.figure(figsize=(5, 5))
plt.imshow(ellipses, cmap='gray')
plt.title(f'Random Ellipses Phantom\n({amount_of_ellipses} ellipses, {size[0]}x{size[1]})')
plt.axis('off')
plt.colorbar()
plt.tight_layout()
plt.savefig('步骤1_随机椭圆phantom.png', dpi=150, bbox_inches='tight')
plt.close()
print("Phantom generated and saved.")


print("\n=== Step 2: Parallel Beam Projection & FBP ===")
vol_geom = astra.create_vol_geom(size)
num_of_lines = 512
amount_of_angles = 720
angles = np.linspace(0, np.pi, amount_of_angles)

proj_geom = astra.create_proj_geom('parallel', 1.0, num_of_lines, angles)
if device == 'cuda':
    proj_id_parall = astra.create_projector('cuda', proj_geom, vol_geom)
else:
    proj_id_parall = astra.create_projector(device[0], proj_geom, vol_geom)

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
parall_rec = np.maximum(0, parall_rec) / np.max(parall_rec)

astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(parall_sinogram_id)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im0 = axes[0].imshow(ellipses, cmap='gray')
axes[0].set_title('Original Phantom')
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(parall_sinogram, cmap='gray', aspect='auto')
axes[1].set_title(f'Parallel Sinogram\n({amount_of_angles} angles, {num_of_lines} lines)')
axes[1].set_xlabel('Angle')
axes[1].set_ylabel('Detector')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(parall_rec, cmap='gray')
axes[2].set_title(f'FBP Reconstruction\nPSNR: {psnr(ellipses, parall_rec):.1f} dB')
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2])

plt.suptitle('Step 2: ASTRA Parallel Beam')
plt.tight_layout()
plt.savefig('步骤2_平行束对比.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Parallel beam FBP PSNR: {psnr(ellipses, parall_rec):.1f} dB")


print("\n=== Step 3: Fan-Beam Projection & FBP ===")
num_of_lines = 512
amount_of_angles = 720
angles = np.linspace(0, 2 * np.pi, amount_of_angles)
SOD = 250
SDD = 260

proj_geom = astra.create_proj_geom('fanflat', 1.0, num_of_lines, angles, SOD, SDD - SOD)
if device == 'cuda':
    proj_id_full = astra.create_projector('cuda', proj_geom, vol_geom)
else:
    proj_id_full = astra.create_projector(device[1], proj_geom, vol_geom)

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
full_rec = np.maximum(0, full_rec) / np.max(full_rec)

astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(ellipses, cmap='gray')
axes[0].set_title('Original Phantom')
axes[0].axis('off')
axes[1].imshow(full_sinogram, cmap='gray', aspect='auto')
axes[1].set_title(f'Fan-Beam Sinogram\n(0-360 deg, {amount_of_angles} angles)')
axes[1].set_xlabel('Angle')
axes[1].set_ylabel('Detector')
axes[2].imshow(full_rec, cmap='gray')
axes[2].set_title(f'Fan-Beam FBP\nPSNR: {psnr(ellipses, full_rec):.1f} dB')
axes[2].axis('off')
plt.suptitle('Step 3: ASTRA Fan-Beam')
plt.tight_layout()
plt.savefig('步骤3_扇形束投影.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Fan-beam FBP PSNR: {psnr(ellipses, full_rec):.1f} dB")


print("\n=== Step 4: Sparse & Limited Angle Fan-Beam CT ===")
configs = [
    ('Full Angle (720 angles)', np.linspace(0, 2*np.pi, 720)),
    ('Sparse Angle (30 angles)', np.linspace(0, 2*np.pi, 30)),
    ('Limited Angle (0-90 deg)', np.linspace(0, np.pi/2, 360)),
]

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
reconstructions = {}

for idx, (name, angles) in enumerate(configs):
    proj_geom = astra.create_proj_geom('fanflat', 1.0, num_of_lines, angles, SOD, SDD - SOD)
    if device == 'cuda':
        proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    else:
        proj_id = astra.create_projector(device[1], proj_geom, vol_geom)

    sino_id, sino = astra.create_sino(ellipses, proj_id)
    astra.data2d.delete(proj_id)

    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict(FBP_type)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id
    cfg['option'] = {'FilterType': 'ram-lak'}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    rec = astra.data2d.get(rec_id)
    rec = np.maximum(0, rec) / np.max(rec)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sino_id)

    p = psnr(ellipses, rec)
    reconstructions[name] = (sino, rec, p)

    axes[idx, 0].imshow(sino, cmap='gray', aspect='auto')
    axes[idx, 0].set_title(f'{name}\nSinogram ({len(angles)} angles)')
    axes[idx, 0].set_xlabel('Angle')
    axes[idx, 0].set_ylabel('Detector')

    axes[idx, 1].imshow(rec, cmap='gray')
    axes[idx, 1].set_title(f'FBP Reconstruction\nPSNR: {p:.1f} dB')
    axes[idx, 1].axis('off')

plt.suptitle('Step 4: Sparse & Limited Angle Fan-Beam CT')
plt.tight_layout()
plt.savefig('步骤4_三种投影对比.png', dpi=150, bbox_inches='tight')
plt.close()

for name in reconstructions:
    print(f"  {name}: PSNR = {reconstructions[name][2]:.1f} dB")


print("\n=== Step 5: FBP Filter Comparison ===")
num_of_lines = 512
amount_of_angles = 720
angles = np.linspace(0, 2*np.pi, amount_of_angles)

proj_geom = astra.create_proj_geom('fanflat', 1.0, num_of_lines, angles, SOD, SDD - SOD)
if device == 'cuda':
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
else:
    proj_id = astra.create_projector(device[1], proj_geom, vol_geom)

sino_id, sino = astra.create_sino(ellipses, proj_id)
astra.data2d.delete(proj_id)

filter_types = ['none', 'ram-lak', 'shepp-logan', 'cosine']
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, filt in enumerate(filter_types):
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict(FBP_type)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id
    cfg['option'] = {'FilterType': filt}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    rec = astra.data2d.get(rec_id)
    rec = np.maximum(0, rec) / np.max(rec)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)

    p = psnr(ellipses, rec)
    axes[idx].imshow(rec, cmap='gray')
    axes[idx].set_title(f'Filter: {filt}\nPSNR: {p:.1f} dB')
    axes[idx].axis('off')
    print(f"  Filter '{filt}': PSNR = {p:.1f} dB")

astra.data2d.delete(sino_id)

plt.suptitle('Step 5: FBP Filter Comparison (Fan-Beam)')
plt.tight_layout()
plt.savefig('步骤5_滤波器对比.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n=== ASTRA Memory Management ===")
print("IMPORTANT: ASTRA uses C++ backend, must manually free memory:")
print("  astra.data2d.delete(id)")
print("  astra.algorithm.delete(id)")
print("  astra.projector.delete(id)")
print("Failure to do so causes GPU memory leaks!")

print("\nDone!")