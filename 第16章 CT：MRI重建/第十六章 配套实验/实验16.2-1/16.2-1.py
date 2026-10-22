# -*- coding: utf-8 -*-
"""
实验16.2-1 CT不适定性与正则化重建

实验目的：理解CT逆问题的两种不适定性（稀疏角度vs有限角度），
          验证迭代正则化重建方法（Tikhonov/TV）的效果，
          量化不适定程度与重建质量的关系

素材来源：基于16.1.py步骤4-6
运行前提：CPU可运行
"""

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
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
    SAVE_DIR = os.path.join(_gdrive, '实验16.2-1')
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

print(f"\n{'='*60}")
print(f"实验16.2-1: CT不适定性与正则化重建")
print(f"{'='*60}")

# ---- 准备Shepp-Logan幻影 ----
# 避免inverse crime：数据生成使用高分辨率模型，反演使用低分辨率模型
n = 128           # 重建分辨率
n_data = 256      # 数据生成分辨率（2倍，制造模型失配）

# 高分辨率phantom（用于数据生成）
phantom_fine = resize(shepp_logan_phantom(), (n_data, n_data), order=0, preserve_range=True, anti_aliasing=False)
phantom_fine = phantom_fine / phantom_fine.max()
# 低分辨率phantom（用于重建和PSNR比较，由高分辨率降采样得到）
phantom = resize(phantom_fine, (n, n), order=1, preserve_range=True, anti_aliasing=True)
phantom = phantom / phantom.max()

def generate_sinogram_ic_free(theta, noise_level=0.0):
    """避免inverse crime的sinogram生成：
    在n_data=256高分辨率下用radon正向投影，再resize到n=128低分辨率探测器尺寸，
    模拟真实CT中"物体连续、探测器binning离散"的模型失配
    （数据生成的离散化算子与反演用的forward_radon不是同一矩阵）。
    ★探测器尺度归一化：radon返回的线积分以像素为单位，同一物体在256网格上
    跨越的像素数是128网格的2倍，故细网格线积分是粗网格的n_data/n倍；
    resize只做插值平均不改变数值尺度，因此必须乘n/n_data，
    否则sinogram整体偏大2倍，重建值也整体偏大2倍（PSNR被强度误差主导）。"""
    sinogram_fine = radon(phantom_fine, theta=theta, circle=True)
    # circle=True时探测器尺寸=图像尺寸，从256 resize到128，并归一化到粗网格像素单位
    sinogram_recon = resize(sinogram_fine, (n, len(theta)), order=1, preserve_range=True, anti_aliasing=True)
    sinogram_recon = sinogram_recon * (n / n_data)
    if noise_level > 0:
        # 相对噪声：以sinogram峰值为基准的百分比（绝对sigma对量级O(30)的sinogram毫无作用）
        sinogram_recon = sinogram_recon + noise_level * sinogram_recon.max() * np.random.randn(*sinogram_recon.shape)
    return sinogram_recon

# 先做全角FBP作为参考（使用IC-free sinogram）
theta_full = np.linspace(0, 180, 180, endpoint=False)
sinogram_full = generate_sinogram_ic_free(theta_full)
recon_fbp_full = iradon(sinogram_full, theta=theta_full, circle=True, filter_name='ramp')
psnr_fbp_full = psnr(phantom, recon_fbp_full, data_range=1.0)
noise_level = 0.05  # 相对噪声水平（占sinogram峰值的5%）


# ========================================================================
# 步骤1：稀疏角度与有限角度CT——两种不同的不适定性
# ========================================================================
print("\n" + "=" * 60)
print("步骤1：稀疏角度与有限角度CT——两种不同的不适定性")
print("=" * 60)

# 稀疏角度：30个均匀分布在[0,180)（IC-free数据生成）
theta_sparse30 = np.linspace(0, 180, 30, endpoint=False)
sino_sparse = generate_sinogram_ic_free(theta_sparse30)
recon_sparse = iradon(sino_sparse, theta=theta_sparse30, circle=True, filter_name='ramp')

# 有限角度：0°-120°（缺失60°-180°）（IC-free数据生成）
theta_limited = np.linspace(0, 120, 120, endpoint=False)
sino_limited = generate_sinogram_ic_free(theta_limited)
recon_limited = iradon(sino_limited, theta=theta_limited, circle=True, filter_name='ramp')

p_sparse = psnr(phantom, recon_sparse, data_range=1.0)
p_limited = psnr(phantom, recon_limited, data_range=1.0)
s_sparse = ssim(phantom, recon_sparse, data_range=1.0)
s_limited = ssim(phantom, recon_limited, data_range=1.0)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].imshow(phantom, cmap='gray')
axes[0, 0].set_title('原始幻影')
axes[0, 0].axis('off')

axes[0, 1].imshow(sino_sparse, cmap='gray', aspect='auto',
                   extent=[0, 180, sino_sparse.shape[0], 0])
axes[0, 1].set_title('稀疏Sinogram (30角度)\n角度覆盖[0$^{\\circ}$, 180$^{\\circ}$)但间隔大')
axes[0, 1].set_xlabel('$\\theta$ ($^{\\circ}$)')

axes[0, 2].imshow(sino_limited, cmap='gray', aspect='auto',
                   extent=[0, 120, sino_limited.shape[0], 0])
axes[0, 2].set_title('有限角Sinogram (0$^{\\circ}$-120$^{\\circ}$)\n缺失60$^{\\circ}$-180$^{\\circ}$角度范围')
axes[0, 2].set_xlabel('$\\theta$ ($^{\\circ}$)')

axes[1, 0].imshow(recon_fbp_full, cmap='gray')
_deg = '$^{\\circ}$'  # 度数符号，预定义以避免f-string中反斜杠语法错误
axes[1, 0].set_title(f'FBP 全角(180{_deg})参考\nPSNR={psnr_fbp_full:.1f}dB')
axes[1, 0].axis('off')

axes[1, 1].imshow(recon_sparse, cmap='gray')
axes[1, 1].set_title(f'FBP 稀疏30角度\nPSNR={p_sparse:.1f}dB, SSIM={s_sparse:.3f}\n条纹伪影（混叠）')
axes[1, 1].axis('off')

axes[1, 2].imshow(recon_limited, cmap='gray')
axes[1, 2].set_title(f'FBP 有限角0{_deg}-120{_deg}\nPSNR={p_limited:.1f}dB, SSIM={s_limited:.3f}\n条纹伪影+边缘模糊丢失')
axes[1, 2].axis('off')

plt.suptitle('步骤1：稀疏角度 vs 有限角度——两种不同的不适定性', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_稀疏vs有限角.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"  全角 FBP:     PSNR={psnr_fbp_full:.1f}dB")
print(f"  稀疏30角度:   PSNR={p_sparse:.1f}dB, SSIM={s_sparse:.3f}")
print(f"  有限角0-120°: PSNR={p_limited:.1f}dB, SSIM={s_limited:.3f}")

# ---- 步骤1b：有限角误差的“边缘集中”与“方向选择性”定量诊断 ----
# 文档需要回答两个问题：有限角的误差是否真的集中在边缘？缺失方向的边缘是否
# 真的丢失？这里用真值梯度定义边缘掩码，算三个可复现的数字：
#   ① 边缘/内部平均绝对误差之比（误差的空间集中度）
#   ② 边缘梯度能量比（边缘幅度保留度，相对真值与相对全角重建两种口径）
#   ③ 水平/垂直边缘的梯度保留度（方向选择性；有限角只应丢失缺失方向的高频）
print("\n  [边缘诊断] 量化有限角的误差集中与方向选择性")
_gy, _gx = np.gradient(phantom)
_gmag = np.hypot(_gy, _gx)
_edge = _gmag > 0.15 * _gmag.max()   # 强度跳变处的强边缘
_inner = _gmag < 0.01 * _gmag.max()  # 平坦区域

_err_lim = np.abs(recon_limited - phantom)
edge_err_ratio_inner = float(_err_lim[_edge].mean() / _err_lim[_inner].mean())
edge_err_ratio_all = float(_err_lim[_edge].mean() / _err_lim.mean())

def _gmag_of(a):
    gy_, gx_ = np.gradient(a)
    return np.hypot(gy_, gx_)

_gm_limited = _gmag_of(recon_limited)
_gm_full = _gmag_of(recon_fbp_full)
edge_amp_vs_true = float(np.median(_gm_limited[_edge]) / np.median(_gmag[_edge]))
edge_amp_vs_full = float(np.median(_gm_limited[_edge]) / np.median(_gm_full[_edge]))

# 方向拆分：水平边缘的灰度沿 y 跳变（|gy|>|gx|），垂直边缘沿 x 跳变
_hor = _edge & (np.abs(_gy) > np.abs(_gx))
_ver = _edge & (np.abs(_gx) >= np.abs(_gy))
_gm_gy, _gm_gx = np.gradient(recon_limited)
ret_h = float(np.median(np.abs(_gm_gy[_hor])) / np.median(np.abs(_gy[_hor])))
ret_v = float(np.median(np.abs(_gm_gx[_ver])) / np.median(np.abs(_gx[_ver])))

print(f"    边缘 {int(_edge.sum())} 像素；边缘/内部误差比 = {edge_err_ratio_inner:.2f}×"
      f"（边缘/全图 = {edge_err_ratio_all:.2f}×）")
print(f"    边缘梯度幅度保留（相对真值）= {edge_amp_vs_true:.3f}，"
      f"（相对全角重建）= {edge_amp_vs_full:.3f}")
print(f"    方向选择性：水平边保留 {ret_h:.3f}，垂直边保留 {ret_v:.3f}"
      f"（缺失 120°-180° 投影，垂直方向高频应显著丢失）")


# ========================================================================
# 步骤2：迭代正则化重建——Tikhonov/TV正则化
# ========================================================================
print("\n" + "=" * 60)
print("步骤2：迭代正则化重建")
print("=" * 60)

def forward_radon(x, theta):
    """正向Radon变换（低分辨率n=128模型，与数据生成模型不同→避免inverse crime）"""
    return radon(x, theta=theta, circle=True)

def backward_radon(sino, theta):
    """伴随Radon变换（无滤波反投影）"""
    return iradon(sino, theta=theta, circle=True, filter_name=None)

def gradient_descent_recon(y, theta, n_iter=100, lam=0.1, reg_type='tikhonov'):
    """
    迭代正则化重建 (梯度下降 + 正则化，即带正则的Landweber迭代)
    x_{k+1} = x_k - alpha * (A^T(Ax_k - y) + lambda * R'(x_k))

    reg_type: 'tikhonov' -> R(x) = (1/2)||x||^2, R'(x) = x
              'tv'       -> R(x) = |||grad x|||_1 (各向同性),
                            R'(x) = -div(grad(x) / |grad(x)|)
    注：n_iter=100为经验停止点——实测TV在50-100次迭代进入平台期（PSNR峰值），
    Tikhonov约50次达峰后缓降（半收敛：继续迭代开始拟合噪声）。
    """
    x = iradon(y, theta=theta, circle=True, filter_name='shepp-logan')
    alpha = 0.01  # 小步长，避免发散

    for k in tqdm(range(n_iter), desc=f'  {reg_type}重建', leave=False):
        residual = forward_radon(x, theta) - y
        grad_data = backward_radon(residual, theta)

        if reg_type == 'tikhonov':
            grad_reg = lam * x
        elif reg_type == 'tv':
            eps = 1e-8
            dx = np.diff(x, axis=1, prepend=x[:, :1])
            dy = np.diff(x, axis=0, prepend=x[:1, :])
            mag = np.sqrt(dx**2 + dy**2 + eps)
            grad_tv_x = dx / mag
            grad_tv_y = dy / mag
            div_x = np.diff(grad_tv_x, axis=1, append=grad_tv_x[:, -1:])
            div_y = np.diff(grad_tv_y, axis=0, append=grad_tv_y[-1:, :])
            grad_reg = -lam * (div_x + div_y)
        else:
            grad_reg = 0

        x = x - alpha * (grad_data + grad_reg)
        x = np.clip(x, 0, None)

    return x

sino_sparse_noisy = generate_sinogram_ic_free(theta_sparse30, noise_level=noise_level)

print("  正在执行Tikhonov正则化重建...")
recon_tikh = gradient_descent_recon(sino_sparse_noisy, theta_sparse30, n_iter=100, lam=0.1, reg_type='tikhonov')
print("  正在执行TV正则化重建...")
recon_tv = gradient_descent_recon(sino_sparse_noisy, theta_sparse30, n_iter=100, lam=0.1, reg_type='tv')

recon_sparse_noisy = iradon(sino_sparse_noisy, theta=theta_sparse30, circle=True, filter_name='ramp')

p_fbp = psnr(phantom, recon_sparse_noisy, data_range=1.0)
p_tikh = psnr(phantom, recon_tikh, data_range=1.0)
p_tv = psnr(phantom, recon_tv, data_range=1.0)
s_fbp = ssim(phantom, recon_sparse_noisy, data_range=1.0)
s_tikh = ssim(phantom, recon_tikh, data_range=1.0)
s_tv = ssim(phantom, recon_tv, data_range=1.0)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

axes[0].imshow(phantom, cmap='gray')
axes[0].set_title('原始幻影')
axes[0].axis('off')

axes[1].imshow(recon_sparse_noisy, cmap='gray')
axes[1].set_title(f'FBP (无正则化)\nPSNR={p_fbp:.1f}dB\n噪声放大')
axes[1].axis('off')

axes[2].imshow(recon_tikh, cmap='gray')
axes[2].set_title(f'Tikhonov正则化 ($\\lambda$=0.1)\nPSNR={p_tikh:.1f}dB\n过度平滑')
axes[2].axis('off')

axes[3].imshow(recon_tv, cmap='gray')
axes[3].set_title(f'TV正则化 ($\\lambda$=0.1)\nPSNR={p_tv:.1f}dB\n保持边缘')
axes[3].axis('off')

plt.suptitle('步骤2：正则化重建对比——FBP噪声放大 vs Tikhonov平滑 vs TV保边缘', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_正则化对比.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"  FBP(稀疏含噪):   PSNR={p_fbp:.1f}dB, SSIM={s_fbp:.3f}")
print(f"  Tikhonov(λ=0.1): PSNR={p_tikh:.1f}dB, SSIM={s_tikh:.3f}")
print(f"  TV(λ=0.1):       PSNR={p_tv:.1f}dB, SSIM={s_tv:.3f}")


# ========================================================================
# 步骤3：角度数量 vs 重建质量——不适定性的量化
# ========================================================================
print("\n" + "=" * 60)
print("步骤3：角度数量 vs 重建质量——不适定性的量化")
print("=" * 60)

n_angles_list = [10, 15, 20, 30, 45, 60, 90, 120, 180]
psnr_clean = []
psnr_noisy_curve = []

for na in tqdm(n_angles_list, desc='  稀疏角度扫描'):
    theta = np.linspace(0, 180, na, endpoint=False)
    # IC-free：高分辨率生成，低分辨率重建
    sino = generate_sinogram_ic_free(theta)
    recon = iradon(sino, theta=theta, circle=True, filter_name='ramp')
    psnr_clean.append(psnr(phantom, recon, data_range=1.0))

    sino_n = generate_sinogram_ic_free(theta, noise_level=noise_level)
    # 与无噪曲线使用同一ramp滤波器，保证两条曲线唯一变量是噪声
    recon_n = iradon(sino_n, theta=theta, circle=True, filter_name='ramp')
    psnr_noisy_curve.append(psnr(phantom, recon_n, data_range=1.0))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(n_angles_list, psnr_clean, 'bo-', label='FBP (无噪声)')
ax1.plot(n_angles_list, psnr_noisy_curve, 'rs-', label=f'FBP (含相对噪声{noise_level*100:.0f}%)')
ax1.set_xlabel('投影角度数')
ax1.set_ylabel('PSNR (dB)')
ax1.set_title('重建质量 vs 投影角度数\n角度越少→信息越少→不适定性越严重')
ax1.legend()
ax1.grid(True)

limited_ranges = [30, 60, 90, 120, 150, 180]
psnr_limited = []
for angle_range in tqdm(limited_ranges, desc='  有限角度扫描'):
    theta = np.linspace(0, angle_range, min(angle_range, 180), endpoint=False)
    # IC-free：高分辨率生成，低分辨率重建
    sino = generate_sinogram_ic_free(theta)
    recon = iradon(sino, theta=theta, circle=True, filter_name='ramp')
    psnr_limited.append(psnr(phantom, recon, data_range=1.0))

ax2.plot(limited_ranges, psnr_limited, 'gs-')
ax2.set_xlabel('角度覆盖范围 ($^{\\circ}$)')
ax2.set_ylabel('PSNR (dB)')
ax2.set_title('有限角CT：角度范围 vs 重建质量\n缺失角度→频域信息缺口→条纹伪影')
ax2.grid(True)

plt.suptitle('步骤3：CT不适定性的量化', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_角度vs质量.png'), dpi=150, bbox_inches='tight')
plt.show()

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
    "experiment": "实验16.2-1 CT不适定性与正则化重建",
    "相对噪声水平": noise_level,
    "噪声说明": "占sinogram峰值的比例（相对噪声）",
    "步骤1_稀疏vs有限角": {
        "全角FBP_PSNR_dB": round(psnr_fbp_full, 2),
        "稀疏30角度": {"PSNR_dB": round(p_sparse, 2), "SSIM": round(s_sparse, 4)},
        "有限角0_120": {"PSNR_dB": round(p_limited, 2), "SSIM": round(s_limited, 4)},
    },
    "步骤1b_边缘诊断": {
        "边缘像素数": int(_edge.sum()),
        "边缘/内部_误差比": round(edge_err_ratio_inner, 2),
        "边缘/全图_误差比": round(edge_err_ratio_all, 2),
        "边缘幅度保留_相对真值": round(edge_amp_vs_true, 4),
        "边缘幅度保留_相对全角": round(edge_amp_vs_full, 4),
        "水平边保留度": round(ret_h, 4),
        "垂直边保留度": round(ret_v, 4),
        "说明": "边缘掩码=真值梯度>15%最大；方向拆分按|gy|与|gx|大小",
    },
    "步骤2_正则化对比": {
        "FBP_PSNR_dB": round(p_fbp, 2),
        "FBP_SSIM": round(s_fbp, 4),
        "Tikhonov_PSNR_dB": round(p_tikh, 2),
        "Tikhonov_SSIM": round(s_tikh, 4),
        "TV_PSNR_dB": round(p_tv, 2),
        "TV_SSIM": round(s_tv, 4),
    },
    "步骤3_稀疏角度扫描": {
        "角度数": n_angles_list,
        "PSNR_无噪声_dB": [round(v, 2) for v in psnr_clean],
        "PSNR_含噪_dB": [round(v, 2) for v in psnr_noisy_curve],
    },
    "步骤3_有限角度扫描": {
        "角度范围": limited_ranges,
        "PSNR_dB": [round(v, 2) for v in psnr_limited],
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")

print("\n实验16.2-1完成！")
