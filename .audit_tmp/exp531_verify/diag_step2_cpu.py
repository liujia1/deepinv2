"""
实验5.3-1 步骤2 诊断脚本（复核用，不改动原实验目录）
目的：在 CPU 上运行 RealSN-DnCNN，检验讲义/代码中对步骤2的三个定性断言：
  A1 得分 magnitude 在边缘区域大、平坦区域小
  A2 残差 D_eps(y)-y 与 eps*得分 一致（按构造恒等，检查数值量级是否合理）
  A3 噪声越大，得分 magnitude 越大
同时记录 PSNR 提升，作为"去噪器确为有效MMSE近似"的证据。
"""
import os, sys, json
import numpy as np
import torch
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # 使 from sampling_tools.xxx 可用
from sampling_tools.load_model import load_model  # 自制CPU版加载（见下）

IMG = r"d:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验\实验5.5-1\cman.png"

def load_model_cpu(path):
    """复刻 sampling_tools.load_model，但去掉硬编码 .cuda()，改在 CPU 上推理"""
    import torch.nn as nn
    from sampling_tools.load_model import DnCNN
    net = DnCNN(channels=1, num_of_layers=17)
    model = nn.DataParallel(net)
    model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
    model.eval()
    return model

cases = [("noise5", 5 / 255), ("noise15", 15 / 255), ("noise40", 40 / 255)]

im = np.array(Image.open(IMG))
print("cman.png shape:", im.shape, "dtype:", im.dtype)
x = torch.Tensor(im / 255.).float()
if x.dim() == 2:
    x_g = x
else:
    x_g = x[:, :, 0]  # 取第一通道，与原始灰度一致
print("used image:", tuple(x_g.shape))

g = torch.Generator().manual_seed(0)
out = {}
for name, sigma in cases:
    mpath = os.path.join(HERE, "Pretrained_models", f"RealSN_DnCNN_{name}.pth")
    if not os.path.exists(mpath):
        print("missing model:", mpath)
        continue
    model = load_model_cpu(mpath)
    eps = sigma ** 2
    y = x_g + torch.randn(x_g.shape, generator=g) * sigma
    with torch.no_grad():
        n_hat = model(y[None][None])[0][0]
    denoised = y - n_hat
    resid = denoised - y              # = -n_hat = D(y)-y
    score = resid / eps               # Tweedie
    # PSNR（相对真值 x）
    def psnr(a, b):
        mse = float(torch.mean((a - b) ** 2))
        return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    # 边缘 vs 平坦：用真值 x 的梯度幅值分位数划分
    gx = torch.zeros_like(x_g); gy = torch.zeros_like(x_g)
    gx[:, :-1] = x_g[:, 1:] - x_g[:, :-1]
    gy[:-1, :] = x_g[1:, :] - x_g[:-1, :]
    edge_mag = torch.sqrt(gx ** 2 + gy ** 2)
    thr_hi = torch.quantile(edge_mag.flatten(), 0.90)
    thr_lo = torch.quantile(edge_mag.flatten(), 0.50)
    mask_edge = edge_mag >= thr_hi
    mask_flat = edge_mag <= thr_lo
    sm = score.abs()
    out[name] = {
        "sigma": round(sigma, 6),
        "eps": round(eps, 9),
        "resid_rms": round(float(resid.pow(2).mean().sqrt()), 6),
        "score_rms": round(float(sm.pow(2).mean().sqrt()), 3),
        "score_mean_abs": round(float(sm.mean()), 3),
        "score_edge_mean": round(float(sm[mask_edge].mean()), 3),
        "score_flat_mean": round(float(sm[mask_flat].mean()), 3),
        "edge_over_flat": round(float(sm[mask_edge].mean() / sm[mask_flat].mean()), 3),
        "psnr_noisy": round(psnr(y, x_g), 3),
        "psnr_denoised": round(psnr(denoised, x_g), 3),
        "psnr_gain": round(psnr(denoised, x_g) - psnr(y, x_g), 3),
    }
    print(name, json.dumps(out[name], ensure_ascii=False))

with open(os.path.join(HERE, "diag_step2_cpu.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("saved diag_step2_cpu.json")
