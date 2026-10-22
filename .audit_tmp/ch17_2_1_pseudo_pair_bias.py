"""验证 N2B 伪配对目标是否有偏：
   (1) 噪声独立零均值 => E[y2|y1] = E[x2|y1]  （"噪声无偏"这一半是对的）
   (2) 但 E[x2|y1] != x1                       （目标本身对 x 有偏）
   (3) 相邻像素相关系数 rho 是偏差的控制因素：rho 越大偏差越小
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from torchvision import datasets, transforms

SIGMA = 0.3
ROOT = r'h:\deepinv\deepinv2\第17章 自监督学习与等变架构\第十七章 配套实验\实验17.2-1\mnist_data'
tf = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
ds = datasets.MNIST(root=ROOT, train=False, download=False, transform=tf)
X = torch.stack([ds[i][0] for i in range(3000)])[:, 0].numpy()  # (N,32,32)

# 相邻像素对：(i,j) 与 (i,j+1)，对应 sub1=(偶,偶) / sub2=(偶,奇) 的关系
a = X[:, :, :-1].reshape(-1).astype(np.float64)
b = X[:, :, 1:].reshape(-1).astype(np.float64)
vx = a.var()
cov = ((a - a.mean()) * (b - b.mean())).mean()
rho = cov / vx
print(f'[真实MNIST] 相邻像素 rho = {rho:.4f},  cov = {cov:.5f},  var_x = {vx:.5f}')


def mc_expectation(a_src, b_src, rho_tag, n=1_500_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a_src), n)
    x1, x2 = a_src[idx], b_src[idx]
    y1 = x1 + rng.normal(0, SIGMA, n)
    y2 = x2 + rng.normal(0, SIGMA, n)

    bins = np.linspace(-0.1, 1.1, 25)
    k = np.digitize(y1, bins)
    rows = []
    for t in range(1, len(bins)):
        m = k == t
        if m.sum() < 3000:
            continue
        rows.append((y1[m].mean(), y2[m].mean(), x2[m].mean(), x1[m].mean(), m.sum()))

    y1m = np.array([r[0] for r in rows])
    ey2 = np.array([r[1] for r in rows])
    ex2 = np.array([r[2] for r in rows])
    ex1 = np.array([r[3] for r in rows])
    w = np.array([r[4] for r in rows], dtype=float); w /= w.sum()

    print(f'--- {rho_tag} ---')
    print(f'  max|E[y2|y1] - E[x2|y1]| = {np.abs(ey2 - ex2).max():.5f}   '
          f'(噪声独立时二者应相等)')
    print(f'  max|E[x2|y1] - E[x1|y1]| = {np.abs(ex2 - ex1).max():.5f}   '
          f'(若=0 才叫"对 x 无偏")')
    print(f'  加权MSE(E[x2|y1], x1) = {np.sum(w * (ex2 - ex1) ** 2):.5f}')
    print(f'  加权MSE(E[x1|y1], x1) = {np.sum(w * (ex1 - ex1) ** 2):.5f}  <- 对照：x1 相对自身为 0')
    # 线性收缩系数：E[x2|y1] 相对 E[x1|y1] 的回归斜率（稳态高斯近似应≈rho）
    slope = np.polyfit(ex1 - ex1.mean(), ex2 - ex2.mean(), 1)[0]
    print(f'  E[x2|y1] 对 E[x1|y1] 的斜率 = {slope:.4f}  (理论≈rho={rho:.4f})')


mc_expectation(a, b, '真实邻域（信号强相关）')

# 破坏空间相关：每张图的行内像素随机置换，再取"相邻"对
rng0 = np.random.default_rng(1)
Xs = X.copy()
for i in range(Xs.shape[0]):
    for r in range(Xs.shape[1]):
        rng0.shuffle(Xs[i, r])
a_s = Xs[:, :, :-1].reshape(-1).astype(np.float64)
b_s = Xs[:, :, 1:].reshape(-1).astype(np.float64)
rho_s = ((a_s - a_s.mean()) * (b_s - b_s.mean())).mean() / a_s.var()
print(f'\n[置换后] rho = {rho_s:.4f}')
mc_expectation(a_s, b_s, '随机置换（信号无关）')
