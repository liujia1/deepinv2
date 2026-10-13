# 实验1.2-2 复核诊断：验证讲义定性断言与数值
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize

np.random.seed(42)
n = 128
phantom = resize(shepp_logan_phantom(), (n, n), order=0, preserve_range=True, anti_aliasing=False)
phantom = phantom / phantom.max()
theta = np.linspace(0, 180, 90, endpoint=False)
sinogram = radon(phantom, theta=theta, circle=True)

print("== 基本信息 ==")
print("phantom range:", phantom.min(), phantom.max())
print("sinogram shape:", sinogram.shape, "max:", sinogram.max(), " (讲义: 128x90, 线积分最大值~33)")

rec_raw = iradon(sinogram, theta=theta, filter_name='ramp', circle=True)
rec_clip = np.clip(rec_raw, 0, 1)
print("\n== 裁剪影响 ==")
print("FBP raw range:", rec_raw.min(), rec_raw.max())
print("frac < 0:", (rec_raw < 0).mean(), " frac > 1:", (rec_raw > 1).mean())
rmse_raw = np.sqrt(np.mean((phantom - rec_raw) ** 2))
rmse_clip = np.sqrt(np.mean((phantom - rec_clip) ** 2))
print("RMSE no-clip: %.6f  PSNR: %.2f" % (rmse_raw, 20 * np.log10(1 / rmse_raw)))
print("RMSE clip   : %.6f  PSNR: %.2f  (json: 0.055378 / 25.13)" % (rmse_clip, 20 * np.log10(1 / rmse_clip)))

err = np.abs(phantom - rec_clip)
print("\n== 误差分布断言核验 ==")
print("error max: %.4f  (热力图 vmax=0.3, 饱和像素占比: %.4f%%)" % (err.max(), (err > 0.3).mean() * 100))
# 中心区域 vs 边缘
yy, xx = np.mgrid[0:n, 0:n]
r = np.sqrt((yy - n / 2) ** 2 + (xx - n / 2) ** 2)
R = n / 2
center = r < 0.5 * R
ring = (phantom > 0.9)  # 高亮外颅边缘像素
interior_low = (phantom > 0.18) & (phantom < 0.22) & ~ring
print("中心区域(r<R/2)平均误差: %.4f" % err[center].mean())
print("全图平均误差: %.4f" % err.mean())
print("外颅高亮环(>0.9)平均误差: %.4f" % err[ring].mean())
# 梯度大处 vs 平坦处
gy, gx = np.gradient(phantom)
grad_mag = np.sqrt(gy ** 2 + gx ** 2)
hi_grad = grad_mag > np.quantile(grad_mag[grad_mag > 0], 0.9)
flat = (grad_mag == 0) & (phantom > 0)
print("高梯度像素(边缘,前10%%)平均误差: %.4f  像素数: %d" % (err[hi_grad].mean(), hi_grad.sum()))
print("平坦内部像素平均误差: %.4f  像素数: %d" % (err[flat].mean(), flat.sum()))

print("\n== 单角度投影断言 ==")
idx = len(theta) // 2
print("example idx:", idx, "theta:", theta[idx], " 投影max: %.2f" % sinogram[:, idx].max())

print("\n== 对照：180角度 & 无裁剪不resize(order=1) 的量级敏感性 ==")
theta180 = np.linspace(0, 180, 180, endpoint=False)
sino180 = radon(phantom, theta=theta180, circle=True)
rec180 = np.clip(iradon(sino180, theta=theta180, filter_name='ramp', circle=True), 0, 1)
r180 = np.sqrt(np.mean((phantom - rec180) ** 2))
print("180角度: RMSE %.4f PSNR %.2f" % (r180, 20 * np.log10(1 / r180)))
phantom_smooth = resize(shepp_logan_phantom(), (n, n), order=1, preserve_range=True)
phantom_smooth = phantom_smooth / phantom_smooth.max()
s2 = radon(phantom_smooth, theta=theta, circle=True)
r2 = np.clip(iradon(s2, theta=theta, filter_name='ramp', circle=True), 0, 1)
rm = np.sqrt(np.mean((phantom_smooth - r2) ** 2))
print("双线性phantom(90角度): RMSE %.4f PSNR %.2f" % (rm, 20 * np.log10(1 / rm)))

print("\n== 稀疏30角度重建（讲义未展示但可对照信息缺失的实际后果） ==")
theta_sparse = np.linspace(0, 180, 30, endpoint=False)
sino_sp = radon(phantom, theta=theta_sparse, circle=True)
rec_sp = np.clip(iradon(sino_sp, theta=theta_sparse, filter_name='ramp', circle=True), 0, 1)
rmsp = np.sqrt(np.mean((phantom - rec_sp) ** 2))
print("30角度FBP: RMSE %.4f PSNR %.2f" % (rmsp, 20 * np.log10(1 / rmsp)))
