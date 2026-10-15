# -*- coding: utf-8 -*-
"""诊断实验1.4-1：
1) 幻影离散值分布（解释为何20桶只有零星几个点）
2) 高斯分桶σ：clip截断免责 vs 小样本MC噪声 vs 真实信号依赖
3) Poisson分桶σ vs 理论sqrt(x*gain)：吻合度、顶端clip
4) SSIM=0.1856(高斯) vs 0.66(泊松) 的机理：平坦区结构项塌缩
5) 脉冲噪声跨版本/跨种子敏感性
"""
import io, sys, json
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

np.random.seed(42)
n = 256
x = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
x = x / x.max()

vals, cnts = np.unique(np.round(x, 6), return_counts=True)
print("== 幻影离散像素值分布 ==")
for v, c in zip(vals, cnts):
    print(f"  x={v:.4f}  npix={c}  ({100*c/x.size:.2f}%)")

sigma_gauss = 0.1
y_gauss = x + sigma_gauss * np.random.randn(n, n)
y_gauss = np.clip(y_gauss, 0, 1)

gain = 0.08
y_poisson = np.random.poisson(x / gain).astype(np.float64) * gain
y_poisson = np.clip(y_poisson, 0, 1)

amount_sp = 0.01
y_sp = random_noise(x, mode='s&p', amount=amount_sp)

print(f"\nPSNR: gauss={peak_signal_noise_ratio(x, y_gauss, data_range=1.0):.2f} "
      f"poisson={peak_signal_noise_ratio(x, y_poisson, data_range=1.0):.2f} "
      f"sp={peak_signal_noise_ratio(x, y_sp, data_range=1.0):.2f}")
print(f"SSIM: gauss={structural_similarity(x, y_gauss, data_range=1.0, win_size=7):.4f} "
      f"poisson={structural_similarity(x, y_poisson, data_range=1.0, win_size=7):.4f} "
      f"sp={structural_similarity(x, y_sp, data_range=1.0, win_size=7):.4f}")

# ---- 分桶诊断 ----
n_bins = 20
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
noise_g = y_gauss - x
noise_p = y_poisson - x
print("\n== 分桶诊断（高斯σ / Poissonσ）==")
print(f"{'bin':>6} {'npix':>7} {'mean x':>7} {'gauss σ':>8} {'理论(无clip)':>10} {'clip后精确理论':>12} {'pois σ':>7} {'sqrt(xg)':>8}")
from scipy.stats import norm
for j in range(n_bins):
    m = (x >= bin_edges[j]) & (x <= bin_edges[j+1]) if j == n_bins-1 else (x >= bin_edges[j]) & (x < bin_edges[j+1])
    c = m.sum()
    if c == 0:
        continue
    mx = x[m].mean()
    sg = noise_g[m].std()
    sp_ = noise_p[m].std()
    # clip后高斯残差的精确std：residual = clip(x+N(0,σ²))-x, 逐像素解析
    lo, hi = -mx, 1 - mx  # 截断边界（对残差）
    s = sigma_gauss
    a_, b_ = lo / s, hi / s
    Pa, Pb = norm.cdf(a_), norm.cdf(b_)
    Za, Zb = norm.pdf(a_), norm.pdf(b_)
    # E[r] = σ(Za - Pb + Pa... ) 用截断正态矩公式: E = μ + σ*(Za-Zb)/(Pb-Pa) ... 数值积分更稳
    t = np.linspace(-6*s, 6*s, 200001)
    pdf = norm.pdf(t, 0, s)
    r = np.clip(t, lo, hi)
    Er = np.trapezoid(pdf, t) and (np.trapezoid(r*pdf, t) / np.trapezoid(pdf, t))
    Er2 = np.trapezoid(r**2*pdf, t) / np.trapezoid(pdf, t)
    sd_clip = np.sqrt(max(Er2 - Er**2, 0))
    print(f"{bin_centers[j]:6.3f} {c:7d} {mx:7.3f} {sg:8.4f} {s:10.4f} {sd_clip:12.4f} {sp_:7.4f} {np.sqrt(mx*gain):8.4f}")

# ---- SSIM 机理：平坦区 vs 边缘区 ----
print("\n== SSIM 分解：背景(x=0)区域 vs 全图 ==")
from skimage.metrics import structural_similarity as ssim
bg = x == 0
print(f"背景像素占比: {bg.mean():.2%}")
# 手工局部统计验证：平坦内点（远离边缘）的SSIM图应≈C3/(2σy²+C3)
ssim_map_g = ssim(x, y_gauss, data_range=1.0, win_size=7, full=True)
ssim_map_p = ssim(x, y_poisson, data_range=1.0, win_size=7, full=True)
inner = np.zeros_like(bg)
inner[20:-20, 20:-20] = True
flat_g = bg & inner  # 背景平坦区
print(f"高斯SSIM图: 背景平坦区均值={ssim_map_g[flat_g].mean():.4f}, 其余={ssim_map_g[~flat_g].mean():.4f}, 全图={ssim_map_g.mean():.4f}")
print(f"泊松SSIM图: 背景平坦区均值={ssim_map_p[flat_g].mean():.4f}, 其余={ssim_map_p[~flat_g].mean():.4f}, 全图={ssim_map_p.mean():.4f}")
C3 = (0.03**2)
print(f"理论塌缩值 C3/(2σ²+C3) (σ=0.1): {C3/(2*0.01+C3):.4f}")

# ---- 脉冲噪声敏感性：不同种子 & 手工实现 ----
print("\n== 脉冲噪声敏感性 ==")
amts_changed = np.sum(np.abs(y_sp - x) > 1e-12)
print(f"实际被改变的像素数: {amts_changed} ({100*amts_changed/x.size:.2f}%)  (amount=0.01 → 名义1%, 其中一半salt一半pepper)")
# 已是0或1的像素被"替换"后不变
at_rail = np.sum((x < 1e-12) | (np.abs(x - 1) < 1e-12))
print(f"位于轨(0或1)的像素占比: {100*at_rail/x.size:.2f}%")
psnr_sp_list = []
for sd in range(20):
    rng = np.random.default_rng(sd)
    mask = rng.random(x.shape) < 0.01
    yy = x.copy()
    n_salt = mask.sum() // 2
    idx = np.argwhere(mask)
    yy[x.shape[0]*idx[:,0]+idx[:,1] < 0] = 0  # noop placeholder
    sel = np.flatnonzero(mask.ravel())
    salt = rng.choice(sel, size=sel.size//2, replace=False)
    pep = np.setdiff1d(sel, salt)
    yv = x.ravel().copy()
    yv[salt] = 1.0; yv[pep] = 0.0
    yy = yv.reshape(x.shape)
    psnr_sp_list.append(peak_signal_noise_ratio(x, yy, data_range=1.0))
print(f"手工实现20个种子的PSNR: mean={np.mean(psnr_sp_list):.2f}, min={np.min(psnr_sp_list):.2f}, max={np.max(psnr_sp_list):.2f}")

# 未clip高斯的理论PSNR
mse_theory = sigma_gauss**2
print(f"\n未clip高斯理论PSNR: {10*np.log10(1/mse_theory):.2f} dB; 实测: {peak_signal_noise_ratio(x, y_gauss, data_range=1.0):.2f} dB")
# 残差直方图0处尖峰来源
res = (y_gauss - x).ravel()
zero_frac = np.mean(np.abs(res) < 1e-12)
print(f"高斯残差恰为0的比例: {zero_frac:.2%} (x=0且噪声<0被clip)")
