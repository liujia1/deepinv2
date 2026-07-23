# -*- coding: utf-8 -*-
"""
附录16D  PET 发射断层重建——射线追踪正演 + Poisson 统计 + OSEM / OSL
对应章节：16.1 节（Radon / 线积分正演）的 PET 推广；本书附录：发射断层

实验目的：
    PET 与 CT 共享同一套线积分正演模型 A（投影 = 沿射线的线积分），
    但 PET 的探测计数服从泊松统计，且需从投影数据里“反演”活度分布。
    本实验动手实现：
      (1) 用 Radon 变换作为 2D 射线追踪正演（与 CT 的 16.1-1 一致）；
      (2) 在投影域叠加泊松噪声（计数越高的 bin 方差越大）；
      (3) OSEM：有序子集期望最大化，即最大似然估计（MLE）；
      (4) OSL：一步延迟 MAP，叠加二次先验抑制噪声。
    对比 OSEM 与 OSL 的重建质量与噪声特性。

素材来源（算法参考，非运行依赖）：
    sirf Python examples/PET/osem_reconstruction.py
        —— OSMAPOSL / OSEM 主流程；
    sirf Python examples/PET/user_osmaposl.py
        —— OSMAPOSL 迭代更新公式
           x <- x * (A_s^T (y_s / (A_s x + b))) / (A_s^T 1)；
    sirf Python examples/PET/osl_reconstruction.py
        —— QuadraticPrior / OSL 先验梯度；
    sirf Python examples/PET/acquisition_model.py
        —— 射线追踪正演 A（这里以 Radon 替代 3D 环形几何，原理一致）。
    本文件为完全自包含的 numpy / skimage 复刻，无需安装 SIRF / STIR。

运行前提：纯 CPU 即可运行，无需 GPU，无需下载数据。
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端（静默模式）
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon
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
    SAVE_DIR = os.path.join(_gdrive, '附录16D')
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
print(f"附录16D : PET 发射断层重建 —— OSEM / OSL")
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
    """计算 ref 与 x 之间的 PSNR / SSIM / NMSE（归一化到 [0,1]）。"""
    ref = np.asarray(ref, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    p = float(psnr(ref, x, data_range=1.0))
    s = float(ssim(ref, x, data_range=1.0))
    nmse = float(np.mean((ref - x) ** 2) / (np.mean(ref ** 2) + 1e-12))
    return p, s, nmse


# ========== 步骤0 : 发射活度体模 ==========
print("\n>>> 步骤0 : 准备 PET 发射活度体模")

N = 111                                   # 图像尺寸（与 SIRF PET 示例一致）
theta = np.linspace(0, 180, 180, endpoint=False)   # 投影角度

base = resize(shepp_logan_phantom(), (N, N), order=0,
              preserve_range=True, anti_aliasing=False)
x_true = base / base.max()

# 叠加若干“热区”模拟病灶/高摄取（PET 典型对比度更高）
yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
for (cy, cx, r, v) in [(40, 40, 8, 1.8), (72, 70, 6, 1.5),
                       (55, 55, 5, 1.3), (30, 80, 5, 1.4)]:
    x_true += v * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
x_true = np.clip(x_true, 0, None)
x_true = x_true / x_true.max()
print(f"    活度体模尺寸 {N}x{N}，含 4 个高摄取热区，已归一化到 [0,1]")


# ========== 步骤1 : 射线追踪正演 + 泊松噪声 ==========
print("\n>>> 步骤1 : 射线追踪正演（Radon）+ 投影域泊松噪声")

# 正演 A 与伴随 A^T（Radon / 无滤波反投影）
def A(x):
    return radon(x, theta=theta)


def AT(sino):
    return iradon(sino, theta=theta, filter_name=None, output_size=N)


proj_true = A(x_true)                     # 真值线积分（活度沿射线求和）
total_counts = 3.0e5                      # 期望总计数级别（PET 典型量级）
scale = total_counts / proj_true.sum()    # 把线积分换算成计数水平
background = 0.05 * proj_true.mean()      # 均匀本底（randoms / scatter 近似）
y_rate = scale * proj_true + background
y_counts = np.random.poisson(y_rate).astype(np.float64)   # 含泊松噪声的实测计数

print(f"    总期望计数 ≈ {total_counts:.0f}，叠加 Poisson 噪声")
print(f"    实测计数均值 ≈ {y_counts.mean():.2f}，本底 b ≈ {background:.2f}")

# 显示：体模 + 真值 sinogram + 含噪 sinogram
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
axes[0].imshow(x_true, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('发射活度体模 $x$', fontsize=12)
axes[0].axis('off')
axes[1].imshow(proj_true, cmap='gray')
axes[1].set_title('无噪投影 $Ax$ (Radon)', fontsize=12)
axes[1].axis('off')
axes[2].imshow(y_counts, cmap='gray')
axes[2].set_title('含 Poisson 噪声投影 $y$', fontsize=12)
axes[2].axis('off')
fig.suptitle('PET 正演：射线追踪 + 泊松统计', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤1_正演与泊松噪声.png'),
            dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("    已保存：步骤1_正演与泊松噪声.png")


# ========== 步骤2 : OSEM 重建（MLE）==========
print("\n>>> 步骤2 : OSEM 重建（参考 SIRF user_osmaposl.py 的 OSMAPOSL 更新公式）")


def osem_recon(y_counts, scale, background, num_subsets=10, n_subiter=20,
               prior_beta=0.0):
    """有序子集期望最大化（OSEM），可选 OSL 二次先验。

    每次子集更新（参考 user_osmaposl.py）：
        grad = A_s^T ( y_s / (A_s x + b) )
        sens = A_s^T 1
        denom = sens + prior_grad / num_subsets   （prior_beta>0 时）
        x <- x * grad / denom
    """
    n_ang = y_counts.shape[0]
    x = np.ones((N, N), dtype=np.float64)         # 初值：均匀活度
    rows_list = [np.arange(s, n_ang, num_subsets) for s in range(num_subsets)]

    for it in tqdm(range(1, n_subiter + 1), desc='  OSEM/OSL', leave=False):
        subset = (it - 1) % num_subsets
        rows = rows_list[subset]
        # 正演（仅当前子集的角度）
        full = np.zeros_like(y_counts)
        full[rows] = (scale * A(x))[rows]
        ratio = y_counts[rows] / (full[rows] + background)
        # 伴随：把子集投影填回整幅 sinogram 再反投影（零行不贡献）
        tmp = np.zeros_like(y_counts)
        tmp[rows] = ratio
        grad = AT(tmp)
        # 子集灵敏度图 A_s^T 1（OSEM 每子集用各自灵敏度，保证无偏）
        ones_s = np.zeros_like(y_counts)
        ones_s[rows] = 1.0
        sens_s = np.maximum(AT(ones_s), 1e-6)
        if prior_beta > 0:
            lap = (4 * x - np.roll(x, 1, 0) - np.roll(x, -1, 0)
                   - np.roll(x, 1, 1) - np.roll(x, -1, 1))
            denom = sens_s + (prior_beta * lap) / num_subsets
        else:
            denom = sens_s
        denom = np.maximum(denom, 1e-6)
        x = x * (grad / denom)
        x = np.clip(x, 0, None)
    return x


x_osem = osem_recon(y_counts, scale, background,
                    num_subsets=10, n_subiter=20, prior_beta=0.0)
print("    OSEM 完成（num_subsets=10, subiterations=20）")


# ========== 步骤3 : OSL 重建（带二次先验的 MAP）==========
print("\n>>> 步骤3 : OSL 重建（参考 SIRF osl_reconstruction.py 的 QuadraticPrior）")

x_osl = osem_recon(y_counts, scale, background,
                   num_subsets=10, n_subiter=20, prior_beta=0.05)
print("    OSL 完成（叠加二次先验 beta=0.05，抑制噪声但轻微平滑）")


# ========== 步骤4 : 指标对比与可视化 ==========
print("\n>>> 步骤4 : 计算指标并对比两种重建结果")

p_o, s_o, nmse_o = metrics(x_true, x_osem)
p_l, s_l, nmse_l = metrics(x_true, x_osl)

print(f"    OSEM : PSNR={p_o:6.2f} dB, SSIM={s_o:.4f}, NMSE={nmse_o:.4f}")
print(f"    OSL  : PSNR={p_l:6.2f} dB, SSIM={s_l:.4f}, NMSE={nmse_l:.4f}")
print("    说明：OSEM 为最大似然估计，计数充足时分辨率好但偏 noisy；")
print("          OSL 在分母加入二次先验梯度，抑制斑点噪声、分辨率略降，")
print("          是典型的‘保真度 ↔ 正则’折中，与 16.2 节的正则化思想一致。")

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
imgs = [x_true, x_osem, x_osl]
titles = ['真值活度 $x$', 'OSEM (MLE)', 'OSL (MAP, 二次先验)']
for ax, im, ti in zip(axes[0], imgs, titles):
    ax.imshow(im, cmap='gray', vmin=0, vmax=1)
    ax.set_title(ti, fontsize=12)
    ax.axis('off')
errs = [np.zeros_like(x_true),
        np.abs(x_true - x_osem),
        np.abs(x_true - x_osl)]
etitles = ['', 'OSEM 误差', 'OSL 误差']
for ax, er, ti in zip(axes[1], errs, etitles):
    if ti == '':
        ax.axis('off')
        continue
    ax.imshow(er, cmap='hot')
    ax.set_title(ti, fontsize=12)
    ax.axis('off')
fig.suptitle('PET 重建对比：OSEM (MLE) vs OSL (带先验 MAP)', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '步骤4_OSEM与OSL对比.png'),
            dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
print("    已保存：步骤4_OSEM与OSL对比.png")


# ========== 步骤5 : 保存 JSON 结果 ==========
print("\n>>> 步骤5 : 保存运行结果到 JSON")

results = {
    "experiment": "附录16D",
    "title": "PET发射断层重建——OSEM / OSL",
    "image_size": N,
    "n_angles": int(theta.size),
    "total_expected_counts": float(total_counts),
    "background": float(background),
    "num_subsets": 10,
    "n_subiterations": 20,
    "osl_prior_beta": 0.05,
    "reference": "sirf Python examples/PET/osem_reconstruction.py, user_osmaposl.py, osl_reconstruction.py, acquisition_model.py",
    "metrics": {
        "OSEM": {"PSNR_dB": p_o, "SSIM": s_o, "NMSE": nmse_o},
        "OSL": {"PSNR_dB": p_l, "SSIM": s_l, "NMSE": nmse_l},
    },
}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results), f, ensure_ascii=False, indent=2)
print("    已保存：results_summary.json")
print(f"\n{'='*64}")
print("附录16D 完成。结果见 results_summary.json 与各步骤 PNG。")
print(f"{'='*64}")
