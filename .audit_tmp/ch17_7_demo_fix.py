import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)


def roll2d(x, k):
    """循环平移 (dy,dx)=(k,k)。这是平移，不是旋转。"""
    return np.roll(np.roll(x, k, axis=-2), k, axis=-1)


def rot90_c(x, k):
    """循环意义下的 90 度旋转 k 次（方形图，旋转不丢信息）。"""
    return np.ascontiguousarray(np.rot90(x, k))


def circ_conv(x, w):
    """循环边界卷积（F.conv2d 没有 padding_mode 参数，必须先 F.pad）。"""
    xt = torch.tensor(x, dtype=torch.float32)[None, None]
    wt = torch.tensor(w, dtype=torch.float32)[None, None]
    xt = F.pad(xt, (1, 1, 1, 1), mode='circular')
    return F.conv2d(xt, wt, padding=0)[0, 0].numpy()


H = W = 16
img = np.zeros((H, W))
img[5:9, 5:9] = 1.0
kernel = np.random.randn(3, 3)

# A) 平移等变：随机核即可严格成立
out0 = circ_conv(img, kernel)
err_shift = np.max(np.abs(circ_conv(roll2d(img, 3), kernel) - roll2d(out0, 3)))
print(f"[A 平移等变/随机核] 最大像素差 = {err_shift:.2e}")

# B) 旋转等变：随机核不满足
err_rot_plain = np.max(np.abs(circ_conv(rot90_c(img, 1), kernel) - rot90_c(out0, 1)))
print(f"[B 旋转等变/随机核] 最大像素差 = {err_rot_plain:.2e}  (尺度 {np.abs(out0).max():.2f})")

# C) 旋转等变：C4 群平均核（对核做 4 个 90 度旋转再平均）
kernel_c4 = sum(np.rot90(kernel, k) for k in range(4)) / 4.0
out_eq = circ_conv(img, kernel_c4)
err_rot_c4 = np.max(np.abs(circ_conv(rot90_c(img, 1), kernel_c4) - rot90_c(out_eq, 1)))
print(f"[C 旋转等变/C4平均核] 最大像素差 = {err_rot_c4:.2e}  (尺度 {np.abs(out_eq).max():.2f})")

# 再验证 C4 平均核仍保持平移等变
err_shift_c4 = np.max(np.abs(circ_conv(roll2d(img, 3), kernel_c4) - roll2d(out_eq, 3)))
print(f"[D 平移等变/C4平均核] 最大像素差 = {err_shift_c4:.2e}")
