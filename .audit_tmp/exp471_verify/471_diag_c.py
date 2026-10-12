"""复核脚本 C：
C1. 短链 ACF 的"周期性/负相关"是否为有限样本伪影（对独立高斯白噪声、M=49 求 ACF 期望形状）
C2. 单次 500 迭代 CPU 运行，保存与实验同布局的图，用于验证"边缘高 std / 平坦低 std / 暗区高 CV"断言
"""
import math, os, sys, json, time, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
EXP = r"D:\works\deepinv\第4章 MCMC与ULA算法\第四章 配套实验\实验4.7-1"
OUT = r"D:\works\deepinv\.audit_tmp\exp471_verify"
sys.path.insert(0, EXP)
device = torch.device('cpu')
from sampling_tools import *   # noqa
from PIL import Image
import arviz
from statsmodels.graphics.tsaplots import plot_acf

# ---------- C1: 白噪声短链 ACF 伪影 ----------
rng = np.random.default_rng(0)
M = 49
lags = np.arange(1, 49)
acf_list = []
for _ in range(4000):
    c = rng.standard_normal(M)
    c = c - c.mean()
    ac = np.correlate(c, c, 'full')[M-1:] / np.var(c) / M
    acf_list.append(ac[1:])
mean_acf = np.mean(acf_list, axis=0)
print("C1: 独立白噪声 M=49 的 E[rho_k]（应全为 0，实际系统性负偏）")
for k in [1, 5, 10, 17, 25, 32, 40, 48]:
    print(f"  lag {k:2d}: E[rho]={mean_acf[k-1]:+.4f}   (理论偏差 -1/M = {-1/M:+.4f})")
print(f"  最小值 lag {int(np.argmin(mean_acf))+1}: {mean_acf.min():+.4f}")
plt.figure(figsize=(6, 4))
plt.plot(lags, mean_acf, 'o-', ms=3)
plt.axhline(0, color='k', lw=0.8)
plt.axhline(-1/M, color='r', ls='--', lw=0.8, label='-1/M')
plt.title(f'White noise, M={M}: mean ACF (finite-sample artifact)')
plt.xlabel('lag'); plt.ylabel('rho_k'); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "acf_white_noise_M49.png"), dpi=130)
print("C1 saved acf_white_noise_M49.png")

# ---------- C2: 单次运行 + 不确定性图验证 ----------
im = np.array(Image.open(os.path.join(EXP, "cman.png")))
x = torch.Tensor(im / 255.).to(device)
A, AT, AAT_norm = blur_operators([5, 5], list(im.shape[:2]), "uniform", device)
model = load_model(os.path.join(EXP, "Pretrained_models", "RealSN_DnCNN_noise5.pth"), device)
denoise = lambda z: (z - model(z[None][None])[0][0]).detach()
eps = (5 / 255) ** 2

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
sigma = torch.linalg.matrix_norm(A(x) - torch.mean(A(x)), ord='fro') / math.sqrt(torch.numel(x) * 10 ** 4)
y = A(x) + sigma * torch.randn_like(x)
gradf = lambda z: AT(A(z) - y) / sigma ** 2
L_y = AAT_norm / sigma ** 2
delta = 0.99 / (1.0 / eps + L_y)

maxit, burnin, n_samples = 500, 50, 50
thinning = max(1, int((maxit - burnin) / n_samples))
X = y.clone(); MC = []; psnr_v = []; ssim_v = []
with torch.no_grad():
    for i in range(maxit):
        X = X - delta * gradf(X) + delta / eps * (denoise(X) - X) + math.sqrt(2 * delta) * torch.randn_like(X)
        if i == burnin:
            pm = welford(X); count = 0
        elif i > burnin:
            pm.update(X)
            psnr_v.append(PSNR(x, pm.get_mean())); ssim_v.append(SSIM(x, pm.get_mean()))
            if count == thinning - 1:
                MC.append(X.numpy()); count = 0
            else:
                count += 1

mean = pm.get_mean().numpy(); std = np.sqrt(pm.get_var().numpy())
cv = std / (mean + 1e-8)
gt = x.numpy()
print(f"\nC2: CPU seed42 PSNR={PSNR(x, pm.get_mean()):.2f} SSIM={SSIM(x, pm.get_mean()):.4f}")

# 用真值梯度幅值定义"边缘"，用局部方差定义"平坦区"
gy, gx = np.gradient(gt)
gmag = np.hypot(gx, gy)
edge_mask = gmag > np.percentile(gmag, 90)          # 真值最强 10% 梯度 = 边缘
flat_mask = (gmag < np.percentile(gmag, 50)) & (np.abs(gt - np.mean(gt)) < 0.05)
dark_mask = gt < np.percentile(gt[gt > 0], 20)      # 暗区（真值最低 20% 非零像素以下）
bright_mask = gt > np.percentile(gt, 80)
print(f"  后验std: 边缘区 mean={std[edge_mask].mean():.5f}  平坦区 mean={std[flat_mask].mean():.5f}  比值={std[edge_mask].mean()/std[flat_mask].mean():.2f}")
print(f"  变异系数CV: 暗区 mean={cv[dark_mask].mean():.4f}  亮区 mean={cv[bright_mask].mean():.4f}")
print(f"  绝对std:   暗区 mean={std[dark_mask].mean():.5f}  亮区 mean={std[bright_mask].mean():.5f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gt, cmap='gray'); axes[0].set_title('truth'); axes[0].axis('off')
axes[1].imshow(mean, cmap='gray'); axes[1].set_title(f'posterior mean PSNR={PSNR(x, pm.get_mean()):.2f}'); axes[1].axis('off')
sc = axes[2].imshow(std, cmap='hot'); axes[2].set_title('posterior std'); axes[2].axis('off')
plt.colorbar(sc, ax=axes[2])
plt.tight_layout(); plt.savefig(os.path.join(OUT, "c2_maps.png"), dpi=130)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].imshow(cv, cmap='gray'); axes[0].set_title('CV = std/mean'); axes[0].axis('off')
axes[1].imshow(gmag > np.percentile(gmag, 90), cmap='gray'); axes[1].set_title('edge mask (truth grad top10%)'); axes[1].axis('off')
plt.tight_layout(); plt.savefig(os.path.join(OUT, "c2_cv.png"), dpi=130)

chain = np.array(MC).reshape(len(MC), -1)
var_sp = np.var(chain, axis=0)
order = np.argsort(var_sp)
for nm, idx in [('slow', order[-1]), ('med', order[len(order)//2]), ('fast', order[0])]:
    tr = chain[:, idx]
    e = float(arviz.ess(tr))
    print(f"  {nm}: std(x_i)={np.sqrt(var_sp[idx]):.5f} 位置=({idx//256},{idx%256})  ESS={e:.2f}")
plt.figure(figsize=(12, 4))
for ax, nm, idx in zip(plt.gcf().axes, ['med', 'slow', 'fast'], [order[len(order)//2], order[-1], order[0]]):
    plot_acf(chain[:, idx], ax=ax, lags=min(50, len(MC)-1), alpha=None)
    ax.set_title(nm)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "c2_acf.png"), dpi=130)
print("C2 saved c2_maps.png c2_cv.png c2_acf.png")
