import numpy as np

sigma = np.array([0.01, 0.03, 0.05, 0.10, 0.20, 0.40])
mse = np.array([0.000093, 0.000828, 0.002269, 0.008731, 0.031332, 0.092552])
psnr = np.array([40.31, 30.82, 26.44, 20.59, 15.04, 10.34])
ssim = np.array([0.9486, 0.7303, 0.5645, 0.3552, 0.1926, 0.0890])

def norm(a): return (a - a.min()) / (a.max() - a.min())

mse_n = 1 - norm(mse)
psnr_n = norm(psnr)
ssim_n = norm(ssim)

print("归一化后（越高越好）：")
print("sigma   mse_n   psnr_n  ssim_n")
for i in range(6):
    print(f"{sigma[i]:5.2f}  {mse_n[i]:.4f}  {psnr_n[i]:.4f}  {ssim_n[i]:.4f}")

print("\n第一段 0.01→0.03 归一化跌幅：")
print(f"  MSE反转: {mse_n[0]-mse_n[1]:.4f} ({(mse_n[0]-mse_n[1])/mse_n[0]*100:.1f}%)")
print(f"  PSNR:    {psnr_n[0]-psnr_n[1]:.4f} ({(psnr_n[0]-psnr_n[1])/psnr_n[0]*100:.1f}%)")
print(f"  SSIM:    {ssim_n[0]-ssim_n[1]:.4f} ({(ssim_n[0]-ssim_n[1])/ssim_n[0]*100:.1f}%)")

print("\n原始值跌幅（0.01→0.03）：")
print(f"  PSNR 绝对降 {psnr[0]-psnr[1]:.2f} dB；相对降 {(psnr[0]-psnr[1])/psnr[0]*100:.1f}%")
print(f"  SSIM 绝对降 {ssim[0]-ssim[1]:.4f}；相对降 {(ssim[0]-ssim[1])/ssim[0]*100:.1f}%")

print("\nMSE 全程倍数与 PSNR 倍数：")
print(f"  MSE比值 {mse[-1]/mse[0]:.1f}x（理论 (0.4/0.01)^2=1600x，clip 削顶后更小）")
print(f"  PSNR 40.31→10.34，比值 {psnr[0]/psnr[-1]:.2f}（讲义称'仅降约4倍'）")

# SSIM 非负性检验：极恶劣情形下 SSIM 是否会为负（讲义关键观察4）
from skimage.metrics import structural_similarity
xs = np.zeros((64, 64)); xs[:, 32:] = 1.0          # 二值方块图
ys1 = 1 - xs                                        # 反相
ys2 = np.random.default_rng(1).random((64, 64))     # 纯随机
print("\nSSIM 反相图:", structural_similarity(xs, ys1, data_range=1.0, win_size=7))
print("SSIM 纯噪声:", structural_similarity(xs, ys2, data_range=1.0, win_size=7))
