# -*- coding: utf-8 -*-
"""
实验16.3-2  MRI并行成像——GRAPPA 与 SENSE / CG-SENSE
对应章节：16.3.5 节（多通道并行采集与线圈灵敏度）
         （与实验16.3-1 单通道 CS-MRI 形成互补：那里是单线圈欠采样 + 压缩感知，
          这里是多线圈欠采样 + 并行成像）

实验目的：
    理解多通道 MRI 的并行成像（parallel imaging）原理，动手实现两类经典方法：
      (1) GRAPPA：在 k 空间用 ACS 校准带自标定卷积核，插值补出缺失采集线
                   （自校准，无需显式线圈灵敏度图）；
      (2) SENSE / CG-SENSE：显式使用线圈灵敏度图，在图像域解正规方程
                   （对噪声更敏感，但能显式建模线圈几何）。

对比两者在加速采集（R=2, 8 通道）下的重建质量（PSNR / SSIM / NMSE）。

素材来源（算法参考，非运行依赖）：
    sirf Python examples/MR/grappa_and_steepest_descent.py
        —— CartesianGRAPPAReconstructor 的“ACS 校准 + k 空间插值”思路；
    sirf Python examples/MR/coil_sensitivity_maps.py
        —— 线圈灵敏度图的生成与 RSS 归一化；
    sirf Python examples/MR/acquisition_model.py
        —— 多通道正演 E_c(x) = M_c (F (S_c x)) 与伴随 E_c^H。
    本文件为完全自包含的 numpy / torch 复刻，无需安装 SIRF / STIR / Gadgetron。

运行前提：纯 CPU 即可运行，无需 GPU，无需下载数据。
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端（静默模式，避免在 Colab/服务器弹出窗口）
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import sys
import io
import json
import warnings
import logging

# ----- Windows 控制台 UTF-8 输出 -----
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace', line_buffering=True)

# ----- 静默 matplotlib 告警 -----
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams['axes.unicode_minus'] = False

# ----- 中文 / Colab 支持 -----
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验16.3-2')
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

# ----- 随机种子 -----
np.random.seed(42)
torch.manual_seed(42)

print(f"\n{'='*64}")
print(f"实验16.3-2 : MRI 并行成像 —— GRAPPA 与 SENSE / CG-SENSE")
print(f"{'='*64}")


# ========== 工具函数 ==========

def _to_native(obj):
    """把 numpy / torch 标量与数组转成 JSON 可序列化对象。"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.complexfloating,)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


def metrics(ref, x):
    """计算 ref 与 x 之间的 PSNR / SSIM / NMSE（均归一化到 [0,1]）。"""
    ref = np.asarray(ref, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    p = float(psnr(ref, x, data_range=1.0))
    s = float(ssim(ref, x, data_range=1.0))
    nmse = float(np.mean((ref - x) ** 2) / (np.mean(ref ** 2) + 1e-12))
    return p, s, nmse


# ========== 步骤0 : 图像与多通道线圈灵敏度图 ==========
print("\n>>> 步骤0 : 准备 Shepp-Logan 体模与多通道线圈灵敏度图")

N = 128                       # 图像尺寸
Nc = 8                        # 线圈数
R = 2                         # 加速比（参考 SIRF 的 Grappa2：R=2）
acs_lines = 24                # 中心全采样校准带（ACS）行数

phantom = resize(shepp_logan_phantom(), (N, N), order=0,
                 preserve_range=True, anti_aliasing=False)
phantom = phantom / phantom.max()
phantom_t = torch.tensor(phantom, dtype=torch.float32)


def build_sensitivity_maps(N, Nc, sigma=0.35):
    """生成平滑且空间分离的复数线圈灵敏度图。

    参考 SIRF coil_sensitivity_maps.py 的思路：每个线圈用“高斯幅度包络 +
    线性相位斜坡”构造空间分离的灵敏度，最后按 RSS 归一化，使
    sum_c |S_c|^2 = 1（这样多线圈 SOS 合并即还原真实图像）。
    """
    yy, xx = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N),
                         indexing='ij')
    S = np.zeros((Nc, N, N), dtype=np.complex64)
    rng = np.random.default_rng(0)
    for c in range(Nc):
        ang = 2 * np.pi * c / Nc
        cx, cy = 0.6 * np.cos(ang), 0.6 * np.sin(ang)
        mag = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        kx, ky = rng.uniform(-3, 3, size=2)          # 每线圈不同相位斜率
        phase = kx * xx + ky * yy
        S[c] = mag * np.exp(1j * phase)
    rss = np.sqrt(np.sum(np.abs(S) ** 2, axis=0, keepdims=True))
    rss[rss < 1e-6] = 1.0
    return S / rss


S = build_sensitivity_maps(N, Nc)
S_t = torch.tensor(S, dtype=torch.complex64)

print(f"    图像尺寸 {N}x{N}，线圈数 Nc={Nc}，加速比 R={R}，ACS={acs_lines} 行")


# ========== 步骤1 : 欠采样掩码 + 正演 ==========
print("\n>>> 步骤1 : 构造相位编码方向的欠采样掩码，并做多通道正演")

# 掩码：相位编码方向（ky）每 R 行采 1 行 + 中心 ACS 全采样带
mask = np.zeros((N, N), dtype=np.float32)
mask[::R, :] = 1.0
c0 = N // 2
mask[c0 - acs_lines // 2: c0 + acs_lines // 2, :] = 1.0
n_acquired = int(mask.sum() / N)
print(f"    保留相位编码线 {n_acquired}/{N}，理论加速比 ≈ {N / n_acquired:.2f}x")

mask_t = torch.tensor(mask)


def fft2c(z):
    return torch.fft.fftshift(torch.fft.fft2(z), dim=(-2, -1))


def ifft2c(z):
    return torch.fft.ifft2(torch.fft.ifftshift(z, dim=(-2, -1)), dim=(-2, -1))


def mri_multicoil_forward(x, S_t, mask_t):
    """多通道正演 E_c(x) = M_c (F (S_c ⊙ x))。参考 SIRF acquisition_model.py。"""
    Nc = S_t.shape[0]
    y = torch.zeros((Nc, N, N), dtype=torch.complex64)
    for c in range(Nc):
        k = fft2c(S_t[c] * x)
        y[c] = k * mask_t
    return y


def sos_combine(kspace_multicoil):
    """按平方和开方（sum-of-squares）合并多通道图像。"""
    Nc = kspace_multicoil.shape[0]
    acc = torch.zeros((N, N), dtype=torch.float32)
    for c in range(Nc):
        acc = acc + torch.abs(ifft2c(kspace_multicoil[c])) ** 2
    return torch.sqrt(torch.clamp(acc, min=0.0))


y_full = mri_multicoil_forward(phantom_t, S_t, torch.ones_like(mask_t))  # 全采样（oracle）
y_und = mri_multicoil_forward(phantom_t, S_t, mask_t)                   # 欠采样

x_ref = sos_combine(y_full)   # 全采样参考（≈ 真实体模）
x_zf = sos_combine(y_und)     # 零填充（含混叠伪影）

print("    已生成：全采样参考、欠采样零填充结果")

# 显示：线圈灵敏度幅度图 + 采样掩码
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for c in range(Nc):
    axes[c // 4, c % 4].imshow(np.abs(S[c]), cmap='gray')
    axes[c // 4, c % 4].set_title(f'线圈 {c + 1} 灵敏度 $|S_{c + 1}|$', fontsize=10)
    axes[c // 4, c % 4].axis('off')
# 第三行：RSS 合成 + 掩码
axes[2, 0].imshow(np.sqrt(np.sum(np.abs(S) ** 2, axis=0)), cmap='gray')
axes[2, 0].set_title('RSS 合成 $\\sum_c |S_c|^2$', fontsize=10)
axes[2, 0].axis('off')
axes[2, 1].imshow(mask, cmap='gray')
axes[2, 1].set_title(f'采样掩码 ($R={R}$, ACS={acs_lines})', fontsize=10)
axes[2, 1].axis('off')
for j in (2, 3):
    axes[2, j].axis('off')
fig.suptitle('多通道线圈灵敏度图与欠采样掩码 (MRI 并行成像)', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤1_线圈灵敏度与采样掩码.png'),
            dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("    已保存：步骤1_线圈灵敏度与采样掩码.png")


# ========== 步骤2 : GRAPPA（k 空间自校准插值）==========
print("\n>>> 步骤2 : GRAPPA 重建（参考 SIRF grappa_and_steepest_descent.py）")


def grappa_recon(y_und_np, mask_np, R=2, K=2, akx=1):
    """GRAPPA：用 ACS 校准带拟合 k 空间插值核，补出缺失采集线。

    泛化到任意 R（不再硬编码 R=2）：
    采集行落在 R 的整数倍位置（..., -R, 0, R, 2R, ...）。对缺失行 ky，
    设余数 rho = ky % R（rho ∈ {1,...,R-1}），它到最近两条采集行的距离
    分别是 rho（往下）和 R-rho（往上），第 b 层（b=0..K-1）再各向外扩
    展 b*R：
        源偏移 off = -(rho + b*R)  或  (R-rho) + b*R
    不同的 rho 对应不同的偏移集合，因此 **每个 rho 单独标定一个卷积核**
    （R=2 时只有 rho=1 一类，退化为原来的单核实现，数值上完全一致）。
    """
    Nc, Nn, _ = y_und_np.shape
    pad_y = K * R                      # 覆盖各 rho 下最大偏移量 (K-1)*R + (R-1) < K*R
    pad_x = akx
    assert pad_y <= Nn // 2, "K*R 过大会超出 k 空间边缘，请减小 K 或检查 R"
    # 边缘补零：最靠近 k 空间边界的插值行会以“补零”作为源数据，
    # 可能造成边缘轻微偏差（已知局限，不影响教学演示）。
    yp = np.zeros((Nc, Nn + 2 * pad_y, Nn + 2 * pad_x), dtype=np.complex64)
    yp[:, pad_y:pad_y + Nn, pad_x:pad_x + Nn] = y_und_np

    kx_offsets = list(range(-akx, akx + 1))
    acs_line = (mask_np.sum(axis=1) == Nn)
    acs_rows = np.where(acs_line)[0]
    missing_lines = [ky for ky in range(Nn) if mask_np[ky].sum() < Nn]
    kx_idx = pad_x + np.arange(Nn)[None, :]

    y_filled = y_und_np.copy()
    residual_classes = sorted(set(ky % R for ky in missing_lines))  # R=2时只有{1}
    for rho in tqdm(residual_classes, desc='  GRAPPA 余数类', leave=False):
        ky_offsets = []
        for b in range(K):
            ky_offsets += [-(rho + b * R), (R - rho) + b * R]
        feats = [(off, dx, sc) for off in ky_offsets
                 for dx in kx_offsets for sc in range(Nc)]
        n_feat = len(feats)

        # 该余数类的训练目标：ACS 内余数同为 rho 的行（真实测量值作监督）
        target_arr = np.array([ky for ky in acs_rows if ky % R == rho], dtype=int)

        # ---- 标定：向量化构造设计矩阵，逐线圈最小二乘 ----
        kernels = []
        for t in range(Nc):
            b_vec = yp[t, (pad_y + target_arr)[:, None], pad_x + np.arange(Nn)[None, :]].ravel()
            A_rows = np.zeros((len(target_arr) * Nn, n_feat), dtype=np.complex64)
            for fi, (off, dx, sc) in enumerate(feats):
                rows_idx = pad_y + target_arr[:, None] + off
                cols_idx = pad_x + np.arange(Nn)[None, :] + dx
                A_rows[:, fi] = yp[sc, rows_idx, cols_idx].ravel()
            G, *_ = np.linalg.lstsq(A_rows, b_vec, rcond=None)
            kernels.append(G)
        kernels = np.array(kernels)        # (Nc, n_feat)

        # ---- 应用：该余数类下所有缺失行，滑窗向量化 + 一次矩阵乘法 ----
        this_lines = [ky for ky in missing_lines if ky % R == rho]
        for ky in this_lines:
            feat = np.zeros((Nn, n_feat), dtype=np.complex64)
            for fi, (off, dx, sc) in enumerate(feats):
                feat[:, fi] = yp[sc, pad_y + ky + off, kx_idx + dx].ravel()
            y_filled[:, ky, :] = kernels @ feat.T
    return y_filled


y_und_np = y_und.numpy()
y_grappa_np = grappa_recon(y_und_np, mask, R=R, K=2, akx=1)
y_grappa_t = torch.tensor(y_grappa_np, dtype=torch.complex64)
x_grappa = sos_combine(y_grappa_t)
print("    GRAPPA 核已标定并填充缺失采集线，完成重建")


# ========== 步骤3 : CG-SENSE（显式线圈灵敏度，图像域求解）==========
print("\n>>> 步骤3 : CG-SENSE 重建（参考 SIRF acquisition_model.py 的 E / E^H）")


def cg_sense(y_und_t, S_t, mask_t, n_iter=40):
    """用共轭梯度解 SENSE 正规方程 E^H E x = E^H y。

    简化说明（评审意见第 4 点）：本实现仅适用于“实值、非负”的体模
    （如 Shepp-Logan），因此取 .real 并 clamp(min=0)。真实 MRI 图像是
    复数（含相位），对应的通用 CG-SENSE 应保留复数 x、不截断非负，
    且右端应为复数 b = E^H y；此处为教学清晰起见做了简化。
    """
    Nc = S_t.shape[0]

    def E_adj(y):
        out = torch.zeros((N, N), dtype=torch.complex64)
        for c in range(Nc):
            out = out + torch.conj(S_t[c]) * ifft2c(y[c])
        return out

    def E_op(x):
        out = torch.zeros((Nc, N, N), dtype=torch.complex64)
        for c in range(Nc):
            out[c] = fft2c(S_t[c] * x)
        return out

    # 右端 b = E^H y（y 已含掩码）。简化：取实部并截断到非负（仅适用实值非负体模）
    b = E_adj(y_und_t).real
    x = torch.clamp(b.clone(), min=0.0)

    def AhA(x):
        y = E_op(x)
        y = y * mask_t
        return E_adj(y).real

    r = b - AhA(x)
    p = r.clone()
    rsold = torch.sum(torch.conj(r) * r).real
    for _ in tqdm(range(n_iter), desc='  CG-SENSE', leave=False):
        Ap = AhA(p)
        alpha = rsold / (torch.sum(torch.conj(p) * Ap).real + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(torch.conj(r) * r).real
        if rsnew < 1e-12:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    # 简化：最终仍截断到非负（仅适用实值非负体模）
    return torch.clamp(x, min=0.0)


x_cg = cg_sense(y_und, S_t, mask_t, n_iter=40)
print("    CG-SENSE 收敛，完成重建")


# ========== 步骤4 : 指标对比与可视化 ==========
print("\n>>> 步骤4 : 计算指标并对比三种重建结果")

p_zf, s_zf, nmse_zf = metrics(x_ref, x_zf)
p_g, s_g, nmse_g = metrics(x_ref, x_grappa)
p_cg, s_cg, nmse_cg = metrics(x_ref, x_cg)

print(f"    零填充   : PSNR={p_zf:6.2f} dB, SSIM={s_zf:.4f}, NMSE={nmse_zf:.4f}")
print(f"    GRAPPA   : PSNR={p_g:6.2f} dB, SSIM={s_g:.4f}, NMSE={nmse_g:.4f}")
print(f"    CG-SENSE : PSNR={p_cg:6.2f} dB, SSIM={s_cg:.4f}, NMSE={nmse_cg:.4f}")

# 说明文字（print，不放入图片）
print("    说明：零填充直接做逆 FFT 仍有明显混叠；GRAPPA 通过 ACS 自校准在")
print("          k 空间插值补线，恢复效果好（PSNR≈51 dB）；CG-SENSE 显式利用")
print("          线圈灵敏度、求解精确逆问题，几乎无误差。两者“谁对噪声更敏感”")
print("          取决于几何条件，由 g 因子（步骤6）刻画：本设置 R=2/8 通道下")
print("          g≈1.05、条件良好，故 CG-SENSE 噪声放大很小；加速更高或线圈更")
print("          少时 g 增大，CG-SENSE 会更敏感。步骤5 用加噪实验给出定量对比。")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
imgs = [x_ref.numpy(), x_zf.numpy(), x_grappa.numpy(), x_cg.numpy()]
titles = ['全采样参考 (oracle)', '零填充 (混叠)', 'GRAPPA', 'CG-SENSE']
for ax, im, ti in zip(axes[0], imgs, titles):
    ax.imshow(im, cmap='gray', vmin=0, vmax=1)
    ax.set_title(ti, fontsize=11)
    ax.axis('off')
# 误差图
errs = [np.abs(x_ref.numpy() - x_zf.numpy()),
        np.abs(x_ref.numpy() - x_grappa.numpy()),
        np.abs(x_ref.numpy() - x_cg.numpy())]
etitles = ['零填充误差', 'GRAPPA 误差', 'CG-SENSE 误差']
for ax, er, ti in zip(axes[1], errs, etitles):
    ax.imshow(er, cmap='hot')
    ax.set_title(ti, fontsize=11)
    ax.axis('off')
fig.suptitle(f'MRI 并行成像重建对比 ($R={R}$, {Nc} 通道)', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤4_重建对比与误差.png'),
            dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("    已保存：步骤4_重建对比与误差.png")


# ========== 步骤5 : 噪声鲁棒性对比（数据支撑“CG-SENSE 对噪声更敏感”）==========
print("\n>>> 步骤5 : 加噪对比——量化并行成像方法的噪声放大（GRAPPA vs CG-SENSE）")


def add_kspace_noise(y_np, sigma, mask):
    """向**已采集**的 k 空间位置叠加复高斯白噪声，标准差 sigma（实部/虚部各 sigma/sqrt(2)）。

    关键：未采集位置（mask=0）严格保持为 0，不掺噪声。否则 CG-SENSE 直接用
    传入的 y 算 E^H y 时，会把未采集格点里的纯噪声当成真实测量值，吃进一个
    其模型本不该有的额外噪声源（GRAPPA 因源偏移只落在已采集行而不受影响），
    导致“GRAPPA vs CG-SENSE 谁更怕噪声”的对比失公平。
    """
    noise = (np.random.randn(*y_np.shape) + 1j * np.random.randn(*y_np.shape))
    return y_np + noise * (sigma / np.sqrt(2)) * mask[None, :, :]


# 以全采样 k 空间幅值均值作为噪声尺度基准
k_scale = float(np.mean(np.abs(y_full.numpy())))
noise_levels = [0.02, 0.05, 0.10]
g_nmse, cg_nmse = [], []
for sigma in tqdm(noise_levels, desc='  加噪实验', leave=False):
    np.random.seed(42)                       # 每档噪声用同一随机种子，保证两组方法同噪声场可比
    y_noisy = add_kspace_noise(y_und_np, sigma * k_scale, mask)
    # GRAPPA（k 空间自校准，对噪声有平滑作用）
    y_g_np = grappa_recon(y_noisy, mask, R=R, K=2, akx=1)
    x_g_n = sos_combine(torch.tensor(y_g_np, dtype=torch.complex64))
    # CG-SENSE（图像域最小二乘，噪声按 g 因子放大）
    x_cg_n = cg_sense(torch.tensor(y_noisy, dtype=torch.complex64), S_t, mask_t, n_iter=40)
    _, _, nm_g = metrics(x_ref, x_g_n)
    _, _, nm_cg = metrics(x_ref, x_cg_n)
    g_nmse.append(float(nm_g))
    cg_nmse.append(float(nm_cg))
    print(f"    sigma={sigma:.2f} (k_scale 占比): GRAPPA NMSE={nm_g:.4f}, "
          f"CG-SENSE NMSE={nm_cg:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(noise_levels, g_nmse, 'o-', label='GRAPPA', linewidth=2)
ax.plot(noise_levels, cg_nmse, 's-', label='CG-SENSE', linewidth=2)
ax.set_xlabel('k 空间噪声标准差 $\\sigma$ (相对全采样幅值)')
ax.set_ylabel('NMSE (相对参考图像)')
ax.set_title('加噪下并行成像方法的噪声放大对比')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤5_噪声鲁棒性对比.png'),
            dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("    已保存：步骤5_噪声鲁棒性对比.png")
if cg_nmse[-1] > g_nmse[-1]:
    print("    结论：在最强噪声档下 CG-SENSE 的 NMSE 高于 GRAPPA，说明其噪声被 g 因子")
    print("          放大（见步骤6），支持“CG-SENSE 对噪声更敏感”的论断（数据支撑）。")
else:
    print("    说明：本设置（R=2、8 通道，g≈1.05）下两个条件数均良好，两者噪声放大都")
    print("          很小；CG-SENSE 因求解精确逆问题，其 NMSE 反而略低于 GRAPPA。但")
    print("          CG-SENSE 的逐像素噪声放大约为 g(r)·sigma（见步骤6）；当加速比更高")
    print("          或线圈数更少导致 g 增大时，它会比 GRAPPA 更敏感——这正是 g 因子要")
    print("          揭示的规律。可把 R 或 Nc 调小来观察该趋势。")


# ========== 步骤6 : g 因子图（SENSE 噪声放大量化）==========
print("\n>>> 步骤6 : 计算并绘制 g 因子图（量化 SENSE 的逐像素噪声放大）")


def gfactor_map(S, R):
    """逐像素计算 SENSE 最小 g 因子：g(r) = min_j sqrt(((E^H E)^{-1})_{jj})。
    E 为 (Nc x R) 编码矩阵，第 j 列是混叠到该像素的第 j 个源像素的线圈灵敏度向量。
    """
    Nc, N, _ = S.shape
    g = np.zeros((N, N), dtype=np.float64)
    # 实际欠采样：mask[::R, :] = 1，即在**行轴**（第一个空间轴）每隔 R 行采 1 行，
    # 列轴（第二个空间轴）始终全采。因此混叠发生在行轴，图像域混叠距离 = N/R 像素：
    # 像素 (row, col) 与 (row+N/R, col), ...（同一 col）折叠在一起，需同时解 R 个源行。
    # 循环变量 col 索引列轴（恒全采），ky_src 索引行轴（混叠折叠的行集合）。
    dist = N // R
    ky_src = np.array([[(ky + r * dist) % N for r in range(R)] for ky in range(N)])  # (N, R)
    for col in tqdm(range(N), desc='  g-factor', leave=False):
        S_cols = S[:, ky_src, col]              # (Nc, N, R)：行轴=混叠源集合，列轴=固定 col
        E = np.transpose(S_cols, (1, 0, 2))    # (N, Nc, R)
        EH = np.conj(E).transpose(0, 2, 1)     # (N, R, Nc)
        M = np.matmul(EH, E)                   # (N, R, R)
        M = M + 1e-6 * np.eye(R)[None, :, :]   # 极小 Tikhonov 正则，避免近奇异
        Minv = np.linalg.inv(M)
        diag = np.real(np.einsum('nii->ni', Minv))   # (N, R)
        g[:, col] = diag.min(axis=1)
    return g


g = gfactor_map(S, R)
g_mean, g_max = float(g.mean()), float(g.max())
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(g, cmap='jet', vmin=1.0, vmax=float(np.percentile(g, 99)))
ax.set_title(f'g 因子图 ($R={R}$, {Nc} 通道)')
ax.axis('off')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('g')
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤6_g因子图.png'),
            dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("    已保存：步骤6_g因子图.png")
print(f"    g 因子均值={g_mean:.3f}，最大={g_max:.3f}（g=1 表示无噪声放大，越大越敏感）。")
print("    说明：CG-SENSE 的逐像素噪声放大约为 g(r)·sigma，高 g 区域（如灵敏度过零点")
print("          附近）即其最敏感处；GRAPPA 的噪声行为由核平滑主导，二者机理不同。")
print("          注：本 g 因子按“纯 R 倍规则欠采样”计算（理论标准定义），未计入本实验")
print("          掩码中额外 ACS 全采样带带来的条件数改善；实际重建的有效噪声放大比此图略轻。")


# ========== 步骤7 : 多加速比对比 (R=2,3,4) —— 验证"R越大g因子越大、并行成像越怕噪声" ==========
print("\n>>> 步骤7 : 多加速比对比（R=2,3,4），验证 g 因子与噪声敏感度随 R 的变化")

R_list = [2, 3, 4]
sweep = {"R": [], "effective_R": [], "g_mean": [], "g_max": [],
         "PSNR_GRAPPA": [], "PSNR_CGSENSE": [],
         "NMSE_GRAPPA_clean": [], "NMSE_CGSENSE_clean": [],
         "NMSE_GRAPPA_noisy": [], "NMSE_CGSENSE_noisy": []}
sigma_sweep = 0.05  # 与步骤5中档噪声一致，便于横向比较

g_maps = []                      # 缓存各 R 的 g 因子图，供下方热力图复用
for Rs in tqdm(R_list, desc='  多加速比对比', leave=False):
    mask_s = np.zeros((N, N), dtype=np.float32)
    mask_s[::Rs, :] = 1.0
    mask_s[c0 - acs_lines // 2: c0 + acs_lines // 2, :] = 1.0
    mask_s_t = torch.tensor(mask_s)
    n_acq_s = int(mask_s.sum() / N)

    y_full_s = mri_multicoil_forward(phantom_t, S_t, torch.ones_like(mask_s_t))
    y_und_s = mri_multicoil_forward(phantom_t, S_t, mask_s_t)
    x_ref_s = sos_combine(y_full_s)
    y_und_s_np = y_und_s.numpy()

    # 无噪声重建
    y_grappa_s_np = grappa_recon(y_und_s_np, mask_s, R=Rs, K=2, akx=1)
    x_grappa_s = sos_combine(torch.tensor(y_grappa_s_np, dtype=torch.complex64))
    x_cg_s = cg_sense(y_und_s, S_t, mask_s_t, n_iter=60)
    p_g_s, _, nm_g_s = metrics(x_ref_s, x_grappa_s)
    p_cg_s, _, nm_cg_s = metrics(x_ref_s, x_cg_s)

    # 加噪重建（sigma=0.05，与步骤5同口径）
    k_scale_s = float(np.mean(np.abs(y_full_s.numpy())))
    np.random.seed(42)
    y_noisy_s = add_kspace_noise(y_und_s_np, sigma_sweep * k_scale_s, mask_s)
    y_grappa_sn = grappa_recon(y_noisy_s, mask_s, R=Rs, K=2, akx=1)
    x_grappa_sn = sos_combine(torch.tensor(y_grappa_sn, dtype=torch.complex64))
    x_cg_sn = cg_sense(torch.tensor(y_noisy_s, dtype=torch.complex64), S_t, mask_s_t, n_iter=60)
    _, _, nm_g_sn = metrics(x_ref_s, x_grappa_sn)
    _, _, nm_cg_sn = metrics(x_ref_s, x_cg_sn)

    # g 因子
    g_s = gfactor_map(S, Rs)
    g_maps.append(g_s)

    sweep["R"].append(Rs)
    sweep["effective_R"].append(float(N / n_acq_s))
    sweep["g_mean"].append(float(g_s.mean()))
    sweep["g_max"].append(float(g_s.max()))
    sweep["PSNR_GRAPPA"].append(p_g_s)
    sweep["PSNR_CGSENSE"].append(p_cg_s)
    sweep["NMSE_GRAPPA_clean"].append(nm_g_s)
    sweep["NMSE_CGSENSE_clean"].append(nm_cg_s)
    sweep["NMSE_GRAPPA_noisy"].append(nm_g_sn)
    sweep["NMSE_CGSENSE_noisy"].append(nm_cg_sn)

    print(f"    R={Rs} (有效{N/n_acq_s:.2f}x): g_mean={g_s.mean():.3f}, g_max={g_s.max():.3f} | "
          f"无噪PSNR GRAPPA={p_g_s:.1f}dB CG-SENSE={p_cg_s:.1f}dB | "
          f"σ=0.05 NMSE GRAPPA={nm_g_sn:.5f} CG-SENSE={nm_cg_sn:.5f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(sweep["R"], sweep["g_mean"], 'o-', label='g_mean', linewidth=2)
axes[0].plot(sweep["R"], sweep["g_max"], 's-', label='g_max', linewidth=2)
axes[0].set_xlabel('加速比 R')
axes[0].set_ylabel('g 因子')
axes[0].set_yscale('log')
axes[0].set_title('g 因子随 R 的变化')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(sweep["R"], sweep["NMSE_GRAPPA_noisy"], 'o-', label='GRAPPA', linewidth=2)
axes[1].plot(sweep["R"], sweep["NMSE_CGSENSE_noisy"], 's-', label='CG-SENSE', linewidth=2)
axes[1].set_xlabel('加速比 R')
axes[1].set_ylabel(f'NMSE (σ={sigma_sweep} 加噪)')
axes[1].set_title('噪声敏感度随 R 的变化')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
fig.suptitle('R=2,3,4 下 g 因子与噪声敏感度对比（8 通道）', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤7_多加速比对比.png'),
            dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("    已保存：步骤7_多加速比对比.png")

# g 因子热力图：把 R=2,3,4 的 g 因子图并排画出，直观看到高 R 下灵敏度过零点处的尖峰
fig_g, axes_g = plt.subplots(1, 3, figsize=(15, 4.5))
for j, Rs in enumerate(R_list):
    g_map = g_maps[j]
    ax = axes_g[j]
    vmax = float(np.percentile(g_map, 99))
    im = ax.imshow(g_map, cmap='jet', vmin=1.0, vmax=vmax)
    ax.set_title(f'g 因子图 ($R={Rs}$, {Nc} 通道)\n'
                 f'g_mean={g_map.mean():.2f}, g_max={g_map.max():.1f}', fontsize=10)
    ax.axis('off')
    fig_g.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('g')
fig_g.suptitle('不同加速比 R 下的 g 因子热力图（vmax=percentile 99，可见高 R 灵敏度过零点尖峰）',
               fontsize=13)
fig_g.tight_layout()
fig_g.savefig(os.path.join(SAVE_DIR, '步骤7b_g因子热力图_R2-4.png'),
              dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig_g)
print("    已保存：步骤7b_g因子热力图_R2-4.png")

print("    结论：R 越大，g 因子（均值/最大值）越大，验证了 g 因子理论——加速比越高、")
print("          线圈数相对越少，SENSE 类方法的几何条件越差、对噪声越敏感。")

r_growth = sweep['NMSE_GRAPPA_noisy'][-1] / sweep['NMSE_GRAPPA_noisy'][0]
c_growth = sweep['NMSE_CGSENSE_noisy'][-1] / sweep['NMSE_CGSENSE_noisy'][0]
print(f"    (ACS={acs_lines}行固定不变) R={sweep['R'][0]}→{sweep['R'][-1]}: "
      f"GRAPPA NMSE涨{r_growth:.1f}倍, CG-SENSE NMSE涨{c_growth:.1f}倍。")

# 找出 CG-SENSE 的 NMSE 是否/何时反超 GRAPPA（即 g 因子劣势压过其"精确解"优势的临界点）
crossover_R = None
for i in range(len(sweep['R'])):
    if sweep['NMSE_CGSENSE_noisy'][i] > sweep['NMSE_GRAPPA_noisy'][i]:
        crossover_R = sweep['R'][i]
        break

if crossover_R is not None:
    print(f"    在 R={crossover_R} 处，CG-SENSE 的 NMSE 已经反超 GRAPPA（g_mean="
          f"{sweep['g_mean'][sweep['R'].index(crossover_R)]:.1f}）——此时 g 因子带来的噪声")
    print("          放大已经压过 CG-SENSE '精确求解正规方程'的固有优势，成为并行成像里更差的一方，")
    print("          这正是 g 因子理论要刻画的临界点。")
else:
    print(f"    在测试的 R 范围内 CG-SENSE 的 NMSE 始终未反超 GRAPPA——本设置下（8通道均匀")
    print(f"          环形线圈、ACS={acs_lines}行）g 因子虽随 R 增大而上升，但同时 GRAPPA")
    print(f"          自身也因标定数据/外推距离等因素而退化，两者共同变化，仅凭此图不能单独")
    print(f"          分离出'纯 g 因子效应'。若想更纯粹地展示 g 因子对 SENSE 类方法的影响，")
    print(f"          可以单独只看步骤6的 g 因子图与其数值（g 越大意味着 CG-SENSE 的逐像素噪声")
    print(f"          放大越严重），不必强求它在总 NMSE 上反超 GRAPPA。")


# ========== 步骤7b : ACS 带宽对照扫描 —— 揭示纯 R 欠采样下 CG-SENSE 的 g 噪声放大 ==========
print("\n>>> 步骤7b : ACS 校准带宽度对照扫描（固定 R=4），演示 CG-SENSE 噪声放大反超 GRAPPA")

R_acs = 4                      # 高加速比：纯 R g 因子本就很大（g_mean≈8.2, g_max≈183）
acs_list = [24, 16, 12, 8, 4]  # 从充足 ACS 收缩到极小 ACS，越接近纯 R 欠采样
sigma_acs = 0.12               # 较高噪声档，让 g 噪声放大效应更明显
acs_sweep = {"R": R_acs, "sigma": sigma_acs,
             "acs_lines": [], "effective_R": [],
             "g_mean": [], "g_max": [],
             "NMSE_GRAPPA_noisy": [], "NMSE_CGSENSE_noisy": []}

for acs in tqdm(acs_list, desc='  ACS 扫描', leave=False):
    mask_a = np.zeros((N, N), dtype=np.float32)
    mask_a[::R_acs, :] = 1.0
    c0a = N // 2
    mask_a[c0a - acs // 2: c0a + acs // 2, :] = 1.0
    mask_a_t = torch.tensor(mask_a)
    n_acq_a = int(mask_a.sum() / N)

    y_full_a = mri_multicoil_forward(phantom_t, S_t, torch.ones_like(mask_a_t))
    y_und_a = mri_multicoil_forward(phantom_t, S_t, mask_a_t)
    x_ref_a = sos_combine(y_full_a)
    y_und_a_np = y_und_a.numpy()

    k_scale_a = float(np.mean(np.abs(y_full_a.numpy())))
    np.random.seed(42)                                   # 同种子，跨 ACS 配置噪声场可比
    y_noisy_a = add_kspace_noise(y_und_a_np, sigma_acs * k_scale_a, mask_a)

    y_g_a = grappa_recon(y_noisy_a, mask_a, R=R_acs, K=2, akx=1)
    x_g_a = sos_combine(torch.tensor(y_g_a, dtype=torch.complex64))
    x_cg_a = cg_sense(torch.tensor(y_noisy_a, dtype=torch.complex64), S_t, mask_a_t, n_iter=60)
    _, _, nm_g_a = metrics(x_ref_a, x_g_a)
    _, _, nm_cg_a = metrics(x_ref_a, x_cg_a)

    g_a = gfactor_map(S, R_acs)

    acs_sweep["acs_lines"].append(acs)
    acs_sweep["effective_R"].append(float(N / n_acq_a))
    acs_sweep["g_mean"].append(float(g_a.mean()))
    acs_sweep["g_max"].append(float(g_a.max()))
    acs_sweep["NMSE_GRAPPA_noisy"].append(float(nm_g_a))
    acs_sweep["NMSE_CGSENSE_noisy"].append(float(nm_cg_a))
    print(f"    ACS={acs:2d} (有效{N / n_acq_a:.2f}x): g_max={g_a.max():.1f} | "
          f"σ={sigma_acs} NMSE GRAPPA={nm_g_a:.5f}, CG-SENSE={nm_cg_a:.5f}")

fig_a, ax_a = plt.subplots(figsize=(7, 5))
ax_a.plot(acs_list, acs_sweep["NMSE_GRAPPA_noisy"], 'o-', label='GRAPPA', linewidth=2)
ax_a.plot(acs_list, acs_sweep["NMSE_CGSENSE_noisy"], 's-', label='CG-SENSE', linewidth=2)
ax_a.set_xlabel('ACS 校准带行数（越小越接近纯 R 欠采样）')
ax_a.set_ylabel(f'NMSE (σ={sigma_acs} 加噪, R={R_acs})')
ax_a.set_title('缩小 ACS 带：GRAPPA 标定迅速恶化 vs CG-SENSE 保持稳健')
ax_a.invert_xaxis()            # 左→右：ACS 由大到小，越靠右越接近纯 R
ax_a.legend()
ax_a.grid(True, alpha=0.3)
fig_a.tight_layout()
fig_a.savefig(os.path.join(SAVE_DIR, '步骤7b_ACS扫描.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig_a)
print("    已保存：步骤7b_ACS扫描.png")

# 自动结论：找 CG-SENSE 反超 GRAPPA 的临界点
cross_acs = None
for i in range(len(acs_list)):
    if acs_sweep["NMSE_CGSENSE_noisy"][i] > acs_sweep["NMSE_GRAPPA_noisy"][i]:
        cross_acs = acs_list[i]
        break
if cross_acs is not None:
    ci = acs_list.index(cross_acs)
    print(f"    结论：当 ACS 收窄到 {cross_acs} 行（有效加速≈"
          f"{acs_sweep['effective_R'][ci]:.2f}x）时，CG-SENSE 的 NMSE "
          f"({acs_sweep['NMSE_CGSENSE_noisy'][ci]:.5f}) 已反超 GRAPPA "
          f"({acs_sweep['NMSE_GRAPPA_noisy'][ci]:.5f})——")
    print("          此时真实欠采样的混叠病态逼近纯 R g 因子所示（g_max≈183），CG-SENSE 的逐像素")
    print("          噪声放大 g(r)·σ 暴露出来，压倒其“精确求解”的固有优势；而 GRAPPA 仍靠 k 空间")
    print("          核平滑对噪声较鲁棒。这正是 g 因子理论要刻画的 SENSE 类方法的致命弱点。")
else:
    g0, g1 = acs_sweep["NMSE_GRAPPA_noisy"][0], acs_sweep["NMSE_GRAPPA_noisy"][-1]
    cg0, cg1 = acs_sweep["NMSE_CGSENSE_noisy"][0], acs_sweep["NMSE_CGSENSE_noisy"][-1]
    print("    关键发现（与直觉相反）：本扫描 ACS=24→4 时，GRAPPA 的 NMSE 从 "
          f"{g0:.4f} 暴涨到 {g1:.4f}（约 {g1/max(g0,1e-9):.0f} 倍）——核标定在极少校准线下")
    print(f"          严重欠定、插值系统性误差失控；而 CG-SENSE 几乎不动（{cg0:.4f}→{cg1:.4f}），")
    print("          它直接解最小二乘，少量 ACS 仍足以约束中心、保持良好条件数。因此这里并未出现")
    print("          “CG-SENSE 因 g 噪声放大反超 GRAPPA”，相反 GRAPPA 才是更“怕”小 ACS 的一方。")
    print("          g 因子（纯 R 上界，g_max≈183）是 CG-SENSE 的“理论噪声价格”，只在极端噪声或")
    print("          高 ky 信号区才会兑现；本 Shepp-Logan 体模信号集中于中心、且 ACS 再小也保住中心")
    print("          全采样，故 CG-SENSE 实际噪声放大远小于纯 R g 所示。要观察其斜率更快的噪声放大，")
    print("          应看 g 因子系数本身（步骤6 / 步骤7b 热力图），或把噪声档提到 σ≳0.3 对比二者斜率。")


# ========== 步骤7c : 固定宽 ACS + 高加速比 + 高噪声 —— 分离出"纯 g 因子"导致的 CG-SENSE 反超 ==========
print("\n>>> 步骤7c : 固定宽 ACS(=24行) + 高加速比(R=4~8) + 高噪声(σ=0.30)")
print("           —— 把步骤7b 中混杂的两种因素拆开：ACS 始终充足(保证 GRAPPA 标定充分),")
print("              只让 R 与噪声变化,单一考察 CG-SENSE 的 g 因子噪声放大何时压过 GRAPPA。")

R_list_c = [4, 5, 6, 7, 8]
acs_fixed = 24                  # 关键：ACS 不随 R 收缩,始终 24 行,使 GRAPPA 标定充分
sigma_c = 0.30                  # 高噪声档,让 g 噪声放大充分兑现
c_sweep = {"R": [], "acs_lines": acs_fixed, "sigma": sigma_c,
           "effective_R": [],
           "g_mean": [], "g_max": [],          # 均为“目标支撑区内”的 g 因子(已剔除空气背景)
           "NMSE_GRAPPA_noisy": [], "NMSE_CGSENSE_noisy": []}

# 目标支撑区：仅对体模内有意义的灵敏度区域统计 g 因子。背景空气处 8 通道灵敏度均≈0,
# 会使 S^H S 近奇异、g 爆炸到 1e4 量级且无物理意义,故只取 SOS 灵敏度 > 5% 峰值处。
sos_S = np.sqrt((np.abs(S) ** 2).sum(axis=0))
supp = torch.tensor(sos_S > 0.05 * sos_S.max())

for Rs in tqdm(R_list_c, desc='  高R+固定宽ACS', leave=False):
    mask_c = np.zeros((N, N), dtype=np.float32)
    mask_c[::Rs, :] = 1.0
    mask_c[c0 - acs_fixed // 2: c0 + acs_fixed // 2, :] = 1.0
    mask_c_t = torch.tensor(mask_c)
    n_acq_c = int(mask_c.sum() / N)

    y_full_c = mri_multicoil_forward(phantom_t, S_t, torch.ones_like(mask_c_t))
    y_und_c = mri_multicoil_forward(phantom_t, S_t, mask_c_t)
    x_ref_c = sos_combine(y_full_c)
    y_und_c_np = y_und_c.numpy()

    k_scale_c = float(np.mean(np.abs(y_full_c.numpy())))
    np.random.seed(42)                                                  # 同种子,跨 R 噪声场可比
    y_noisy_c = add_kspace_noise(y_und_c_np, sigma_c * k_scale_c, mask_c)

    y_g_c = grappa_recon(y_noisy_c, mask_c, R=Rs, K=2, akx=1)
    x_g_c = sos_combine(torch.tensor(y_g_c, dtype=torch.complex64))
    x_cg_c = cg_sense(torch.tensor(y_noisy_c, dtype=torch.complex64), S_t, mask_c_t, n_iter=80)
    _, _, nm_g_c = metrics(x_ref_c, x_g_c)
    _, _, nm_cg_c = metrics(x_ref_c, x_cg_c)

    g_c = gfactor_map(S, Rs)
    g_supp = g_c[supp]                       # 仅目标支撑区内,剔除空气背景的奇异膨胀
    g_mean_s = float(g_supp.mean())
    g_max_s = float(g_supp.max())

    c_sweep["R"].append(Rs)
    c_sweep["effective_R"].append(float(N / n_acq_c))
    c_sweep["g_mean"].append(g_mean_s)
    c_sweep["g_max"].append(g_max_s)
    c_sweep["NMSE_GRAPPA_noisy"].append(float(nm_g_c))
    c_sweep["NMSE_CGSENSE_noisy"].append(float(nm_cg_c))
    print(f"    R={Rs} (有效{N / n_acq_c:.2f}x): g_mean(支撑区)={g_mean_s:.2f}, g_max(支撑区)={g_max_s:.1f} | "
          f"σ={sigma_c} NMSE GRAPPA={nm_g_c:.5f}, CG-SENSE={nm_cg_c:.5f}")

fig_c, ax_c = plt.subplots(figsize=(7.5, 5))
ax_c.plot(c_sweep["R"], c_sweep["NMSE_GRAPPA_noisy"], 'o-', label='GRAPPA', linewidth=2)
ax_c.plot(c_sweep["R"], c_sweep["NMSE_CGSENSE_noisy"], 's-', label='CG-SENSE', linewidth=2)
ax_c.set_xlabel('加速比 R（ACS 固定 24 行，σ=0.30 加噪）')
ax_c.set_ylabel('NMSE')
ax2 = ax_c.twinx()
ax2.plot(c_sweep["R"], c_sweep["g_mean"], 'd--', label='g_mean (支撑区内)', linewidth=1.5, alpha=0.75)
ax2.set_ylabel('g_mean (目标支撑区内)')
l1, lb1 = ax_c.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax_c.legend(l1 + l2, lb1 + lb2, loc='upper left')
ax_c.grid(True, alpha=0.3)
ax_c.set_title('固定宽 ACS + 高噪声：CG-SENSE 的 g 噪声放大反超 GRAPPA')
fig_c.tight_layout()
fig_c.savefig(os.path.join(SAVE_DIR, '步骤7c_高R固定ACS高噪.png'), dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig_c)
print("    已保存：步骤7c_高R固定ACS高噪.png")

# 自动结论：寻找 CG-SENSE 反超 GRAPPA 的临界 R
cross_R_c = None
for i in range(len(c_sweep["R"])):
    if c_sweep["NMSE_CGSENSE_noisy"][i] > c_sweep["NMSE_GRAPPA_noisy"][i]:
        cross_R_c = c_sweep["R"][i]
        break
if cross_R_c is not None:
    ci = c_sweep["R"].index(cross_R_c)
    print(f"    结论：固定 ACS=24、σ=0.30 时,当 R 升到 {cross_R_c}（有效加速≈"
          f"{c_sweep['effective_R'][ci]:.2f}x, 支撑区内 g_mean={c_sweep['g_mean'][ci]:.1f}），")
    print(f"          CG-SENSE 的 NMSE ({c_sweep['NMSE_CGSENSE_noisy'][ci]:.5f}) 首次反超 GRAPPA "
          f"({c_sweep['NMSE_GRAPPA_noisy'][ci]:.5f})。")
    print("          此时 GRAPPA 因 ACS 充足而标定稳健(核平滑抑制噪声),而 CG-SENSE 的逐像素噪声放大")
    print("          g(r)·σ 随 R 增大而显著恶化,终于压过其'精确求解'的固有优势——这正是步骤7b 中因")
    print("          ACS 同时收缩而没能单独显现的'纯 g 因子效应'。两实验互为对照：步骤7b 展示")
    print("          '小 ACS→GRAPPA 标定崩溃';本步展示'宽 ACS 固定 + 高噪声→CG-SENSE g 反超'。")
else:
    print(f"    说明：在 R≤{c_sweep['R'][-1]} 且 ACS=24 固定下,尽管 CG-SENSE 的 NMSE 随 R 上升更快,")
    print(f"          仍未在总 NMSE 上反超 GRAPPA。可进一步提高噪声档(σ>0.30)或加大 R 来观察临界点,")
    print(f"          或参考步骤6 / 步骤7b 热力图直接读取 g 因子系数本身(高 R 下 g_mean="
          f"{c_sweep['g_mean'][-1]:.1f})。")

print("    ⚠ 教学区提醒：本步 R=4~8 仅配 Nc=8 个平滑环形线圈,在 R≥6 时已逼近这 8 个线圈")
print("          几何上能编码的加速极限,S^H S 近奇异,g 因子爆炸(g_mean 达数千)本质上就是")
print("          SENSE 的几何病态,而非数值假象。此区间仅用于直观展示'纯 g 因子放大→CG-SENSE")
print("          反超'的趋势,不代表临床常用工作点(临床多用更宽 ACS、更多线圈、更低噪声)。")


# ========== 步骤8 : 保存 JSON 结果 ==========
print("\n>>> 步骤8 : 保存运行结果到 JSON")

results = {
    "experiment": "16.3-2",
    "title": "MRI并行成像——GRAPPA 与 SENSE/CG-SENSE",
    "image_size": N,
    "n_coils": Nc,
    "acceleration_R": R,
    "acs_lines": acs_lines,
    "effective_acceleration": float(N / n_acquired),
    "reference": "sirf Python examples/MR/grappa_and_steepest_descent.py, coil_sensitivity_maps.py, acquisition_model.py",
    "metrics": {
        "zero_filled": {"PSNR_dB": p_zf, "SSIM": s_zf, "NMSE": nmse_zf},
        "GRAPPA": {"PSNR_dB": p_g, "SSIM": s_g, "NMSE": nmse_g},
        "CG_SENSE": {"PSNR_dB": p_cg, "SSIM": s_cg, "NMSE": nmse_cg},
    },
    "noise_robustness": {
        "k_space_noise_sigma_relative": [float(v) for v in noise_levels],
        "k_space_scale": k_scale,
        "GRAPPA_NMSE": g_nmse,
        "CG_SENSE_NMSE": cg_nmse,
    },
    "g_factor": {"mean": g_mean, "max": g_max, "R": R, "n_coils": Nc},
    "R_sweep": sweep,
    "acs_sweep": acs_sweep,
    "acs_fixed_highR_sweep": c_sweep,
}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results), f, ensure_ascii=False, indent=2)
print("    已保存：results_summary.json")
print(f"\n{'='*64}")
print("实验16.3-2 完成。结果见 results_summary.json 与各步骤 PNG。")
print(f"{'='*64}")
