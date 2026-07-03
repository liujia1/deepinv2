"""
实验16.5：ASTRA工具箱CT算子演示（M11）——跨平台回退版本
对应章节：16.1.2（Radon变换）、16.2.1（全角CT）、16.2.2（有限角CT）
参考素材：astra_operators_example.ipynb（Teaching Unit 4）

注意：本文件为跨平台回退版本，使用skimage替代ASTRA。
ASTRA完整版本（16.5.py）需要Linux+CUDA环境。
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom


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


def psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(np.max(img1)**2 / mse)


print("=== Step 1: Random Ellipse Phantom Generation ===")
size = 256
np.random.seed(42)

shepp = shepp_logan_phantom()
shepp = (shepp - shepp.min()) / (shepp.max() - shepp.min())

random_phantom = random_ellipses(50, (size, size))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(shepp, cmap='gray')
axes[0].set_title('Shepp-Logan Phantom')
axes[0].axis('off')
axes[1].imshow(random_phantom, cmap='gray')
axes[1].set_title('Random Ellipses Phantom (50 ellipses)')
axes[1].axis('off')
plt.suptitle('Step 1: Phantom Comparison')
plt.tight_layout()
plt.savefig('步骤1_随机椭圆phantom.png', dpi=150, bbox_inches='tight')
plt.close()
print("Phantom comparison saved.")


print("\n=== Step 2: Parallel Beam Projection & FBP ===")
theta_full = np.linspace(0., 180., 360, endpoint=False)
sinogram_full = radon(random_phantom, theta=theta_full, circle=True)
recon_full = iradon(sinogram_full, theta=theta_full, circle=True, filter_name='ramp')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(random_phantom, cmap='gray')
axes[0].set_title('Original Phantom')
axes[0].axis('off')
axes[1].imshow(sinogram_full, cmap='gray', aspect='auto')
axes[1].set_title(f'Sinogram ({len(theta_full)} angles)')
axes[1].set_xlabel('Projection angle')
axes[1].set_ylabel('Detector position')
axes[2].imshow(recon_full, cmap='gray')
axes[2].set_title(f'FBP Reconstruction\nPSNR: {psnr(random_phantom, recon_full):.1f} dB')
axes[2].axis('off')
plt.suptitle('Step 2: Parallel Beam (skimage)')
plt.tight_layout()
plt.savefig('步骤2_平行束对比.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Parallel beam FBP PSNR: {psnr(random_phantom, recon_full):.1f} dB")


print("\n=== Step 3: Fan-Beam Geometry (Conceptual) ===")
print("Fan-beam projection requires ASTRA toolbox (Linux+CUDA).")
print("Key parameters:")
print("  SOD = 250  # Source-Object Distance")
print("  SDD = 260  # Source-Detector Distance")
print("  proj_geom = astra.create_proj_geom('fanflat', 1.0, 512, angles, SOD, SDD-SOD)")
print()
print("Fan-beam vs Parallel beam differences:")
print("  - Fan-beam: rays diverge from point source")
print("  - Parallel beam: rays are parallel")
print("  - Fan-beam sinogram: asymmetric sine curves (divergence effect)")
print("  - Fan-beam has magnification: M = SDD/SOD")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

theta_par = np.linspace(0., 180., 180, endpoint=False)
sino_par = radon(random_phantom, theta=theta_par, circle=True)
axes[0].imshow(sino_par, cmap='gray', aspect='auto')
axes[0].set_title('Parallel Beam Sinogram\n(symmetric sine curves)')
axes[0].set_xlabel('Angle')
axes[0].set_ylabel('Detector')

theta_fan = np.linspace(0., 360., 720, endpoint=False)
sino_fan = radon(random_phantom, theta=theta_fan, circle=True)
axes[1].imshow(sino_fan, cmap='gray', aspect='auto')
axes[1].set_title('Fan-Beam Sinogram (simulated)\n(0-360 degrees, asymmetric)')
axes[1].set_xlabel('Angle')
axes[1].set_ylabel('Detector')

plt.suptitle('Step 3: Fan-Beam Geometry Comparison')
plt.tight_layout()
plt.savefig('步骤3_扇形束投影.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fan-beam comparison saved.")


print("\n=== Step 4: Sparse & Limited Angle CT ===")
configs = [
    ('Full Angle (360 angles)', np.linspace(0., 180., 360, endpoint=False)),
    ('Sparse Angle (30 angles)', np.linspace(0., 180., 30, endpoint=False)),
    ('Limited Angle (0-90 deg, 180 angles)', np.linspace(0., 90., 180, endpoint=False)),
]

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
for idx, (name, theta) in enumerate(configs):
    sino = radon(random_phantom, theta=theta, circle=True)
    recon = iradon(sino, theta=theta, circle=True, filter_name='ramp')
    p = psnr(random_phantom, recon)

    axes[idx, 0].imshow(sino, cmap='gray', aspect='auto')
    axes[idx, 0].set_title(f'{name}\nSinogram ({len(theta)} angles)')
    axes[idx, 0].set_xlabel('Angle')
    axes[idx, 0].set_ylabel('Detector')

    axes[idx, 1].imshow(recon, cmap='gray')
    axes[idx, 1].set_title(f'FBP Reconstruction\nPSNR: {p:.1f} dB')
    axes[idx, 1].axis('off')

plt.suptitle('Step 4: Sparse & Limited Angle CT')
plt.tight_layout()
plt.savefig('步骤4_三种投影对比.png', dpi=150, bbox_inches='tight')
plt.close()
print("Sparse/limited angle comparison saved.")


print("\n=== Step 5: FBP Filter Comparison ===")
filters = ['None', 'ramp', 'shepp-logan', 'cosine']
theta = np.linspace(0., 180., 360, endpoint=False)
sino = radon(random_phantom, theta=theta, circle=True)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for idx, filt in enumerate(filters):
    if filt == 'None':
        recon = iradon(sino, theta=theta, circle=True, filter_name=None)
    else:
        recon = iradon(sino, theta=theta, circle=True, filter_name=filt)
    p = psnr(random_phantom, recon)

    axes[idx].imshow(recon, cmap='gray')
    axes[idx].set_title(f'{filt}\nPSNR: {p:.1f} dB')
    axes[idx].axis('off')

plt.suptitle('Step 5: FBP Filter Comparison')
plt.tight_layout()
plt.savefig('步骤5_滤波器对比.png', dpi=150, bbox_inches='tight')
plt.close()

print("Filter comparison results:")
for filt in filters:
    if filt == 'None':
        recon = iradon(sino, theta=theta, circle=True, filter_name=None)
    else:
        recon = iradon(sino, theta=theta, circle=True, filter_name=filt)
    print(f"  {filt:15s}: PSNR = {psnr(random_phantom, recon):.1f} dB")

print("\n=== ASTRA vs skimage Summary ===")
print("skimage (this fallback):")
print("  + Cross-platform (Windows/macOS/Linux)")
print("  + Simple API (radon/iradon)")
print("  - Parallel beam only")
print("  - No GPU acceleration")
print("  - Fixed detector count")
print()
print("ASTRA (16.5.py, Linux+CUDA only):")
print("  + Fan-beam/cone-beam geometry")
print("  + GPU acceleration (50-100x faster)")
print("  + Custom detector count")
print("  + Multiple reconstruction algorithms (FBP/SIRT/SART/CGLS)")
print("  - Linux+CUDA only")
print("  - Manual memory management required")

print("\nDone!")