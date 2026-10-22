import numpy as np
import torch

# 直接读 MNIST idx 文件，避免依赖 torchvision 下载
p = r"h:\deepinv\deepinv2\第16章 CT：MRI重建\第十六章 配套实验\实验16.5-1\data\MNIST\raw\t10k-images-idx3-ubyte"
with open(p, 'rb') as f:
    buf = f.read()
n = int.from_bytes(buf[4:8], 'big')
h = int.from_bytes(buf[8:12], 'big')
w = int.from_bytes(buf[12:16], 'big')
imgs = np.frombuffer(buf, dtype=np.uint8, offset=16).reshape(n, h, w)[:4].astype(np.float64) / 255.0
x = torch.tensor(imgs, dtype=torch.float64).unsqueeze(1)

N = 28
R = 4
# 复刻 create_mri_mask_topk(28, R=4)
n_rows = N
n_sample = max(n_rows // R, 1)
prob = torch.zeros(n_rows)
for i in range(n_rows):
    dist = abs(i - n_rows // 2) / (n_rows // 2)
    prob[i] = (1 - dist ** 2) ** 1.5 + 0.02
prob = prob / prob.sum() * n_sample
mask = torch.zeros(n_rows)
idx = torch.argsort(prob, descending=True)
mask[idx[:n_sample]] = 1
mask[n_rows // 2] = 1
print('mask rows kept =', int(mask.sum()), ' / ', N)

k = torch.fft.fftshift(torch.fft.fft2(x), dim=-2)
m2 = mask.view(1, 1, N, 1).expand_as(k)
y = k * m2
x_zf = torch.real(torch.fft.ifft2(torch.fft.ifftshift(y * m2, dim=-2)))
print('gt   mean=%.4f  max=%.3f' % (x.mean().item(), x.max().item()))
print('x_zf mean=%.4f  max=%.3f   (diag(A^T A)=%.3f)' % (x_zf.mean().item(), x_zf.max().item(), mask.sum().item() / N))


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(1.0 / mse)


for i in range(4):
    g = imgs[i]
    raw = np.clip(x_zf[i, 0].numpy(), 0, 1)
    sc = np.clip(x_zf[i, 0].numpy() * (N / mask.sum().item()), 0, 1)
    # 也可做逐图 max 归一化（16.1-2 的做法）
    mx = np.clip(x_zf[i, 0].numpy() / x_zf[i, 0].numpy().max(), 0, 1)
    print(f'样本{i}: 零填充原始={psnr(g, raw):.2f} dB | 乘 N/行数={psnr(g, sc):.2f} dB | max归一化={psnr(g, mx):.2f} dB')
