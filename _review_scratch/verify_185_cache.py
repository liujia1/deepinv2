# -*- coding: utf-8 -*-
"""独立复核 18.5-1 v3 缓存：掩码/真值复现、30样本全部统计量重算、零模型解析值"""
import os, json, math, pickle
import numpy as np
import torch
from PIL import Image

EXP = r"d:\works\deepinv\第18章 综合项目：用扩散模型求解自定义逆问题\第十八章 配套实验\实验18.5-1"
with open(os.path.join(EXP, "experiment_cache_S30.pkl"), "rb") as f:
    cache = torch.load(f, map_location="cpu", weights_only=False)
with open(os.path.join(EXP, "uq_results.json"), encoding="utf-8") as f:
    js = json.load(f)

fp = cache["fingerprint"]
print("指纹关键字段:")
for k in ("cache_version", "sampler", "device", "n_samples", "img_size", "mask_seed",
          "mask_hash", "mask_keep_ratio", "init_mode", "data_fidelity", "lambda_data",
          "steps_each_sigma", "map_levels", "map_steps_per_level"):
    print(f"  {k} = {fp.get(k)}")
sched = fp["sigma_schedule"]
print("  sigma_schedule:", [round(s, 4) for s in sched])

samples = torch.cat(cache["samples"], 0) if isinstance(cache["samples"], list) else cache["samples"]
x_true = cache["x_true"]; mask = cache["mask"]; y = cache["y_inp"]; map_x = cache.get("map_x")
print("\nsamples:", tuple(samples.shape), "x_true:", tuple(x_true.shape),
      "mask:", tuple(mask.shape), "map_x:", None if map_x is None else tuple(map_x.shape))
print("sample range:", float(samples.min()), float(samples.max()))

# ---- 1) 掩码按 mask_seed 复现 ----
g = torch.Generator(device="cpu").manual_seed(fp["mask_seed"])
aux = torch.rand((3, 128, 128), generator=g)
m2 = torch.ones((3, 128, 128)); m2[:, aux[0] > 0.5] = 0.0
print("\n掩码复现 max|Δ| =", float((m2 - mask[0]).abs().max()),
      " keep=", round(float(mask.mean()), 4))

# ---- 2) x_true 与本地 celeba(128 bilinear) ----
img = Image.open(os.path.join(EXP, "celeba_example.jpg")).convert("RGB").resize((128, 128), Image.BILINEAR)
xt2 = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0)
print("x_true 复现 max|Δ| =", float((xt2 - x_true).abs().max()))

# ---- 3) 观测一致性：M x + 噪声不可复现(噪声种子流)，只检查零填充像素 ----
zf = mask * y
print("y[观测] 与 x 偏差 RMS =", float((y[mask > 0.5] - x_true[mask > 0.5]).pow(2).mean().sqrt()),
      "(应≈0.01)")

# ---- 4) 重算全部统计量 ----
S = samples.shape[0]
pm = samples.mean(0, keepdim=True)
psd = samples.std(0, keepdim=True)
obs = mask[0].bool(); mis = ~obs

def psnr_t(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(1.0 / mse)

stats = {}
stats["后验均值PSNR_整体"] = psnr_t(x_true, pm)
stats["后验均值PSNR_观测半区"] = psnr_t(x_true[0][obs], pm[0][obs])
stats["后验均值PSNR_缺失半区"] = psnr_t(x_true[0][mis], pm[0][mis])
sp = [psnr_t(x_true, samples[i:i+1]) for i in range(S)]
stats["单样本min"], stats["单样本max"] = min(sp), max(sp)
stats["单样本PSNR标准差"] = float(np.std(sp))
err = pm - x_true
stats["缺失半区占平方误差比例"] = float(err[0][mis].pow(2).sum() / err.pow(2).sum())
stats["std_观测半区"] = float(psd[0][obs].pow(2).mean().sqrt())
stats["std_缺失半区"] = float(psd[0][mis].pow(2).mean().sqrt())
stats["std_整体"] = float(psd.mean())
# lag1
def lag1(field, w=None):
    vals = []
    for c in range(field.shape[0]):
        a = field[c].numpy()
        if w is not None:
            a = np.where(w[c].numpy() > 0.5, a, 0.0)
        a = a - a.mean(); v = (a * a).mean()
        if v > 1e-12:
            vals.append(0.5 * ((a[:-1] * a[1:]).mean() / v + (a[:, :-1] * a[:, 1:]).mean() / v))
    return float(np.mean(vals))
wmis = (1.0 - mask[0]).float()
dev = samples - pm
stats["lag1_缺失"] = float(np.mean([lag1(dev[s], wmis) for s in range(S)]))
stats["lag1_全图"] = float(np.mean([lag1(dev[s]) for s in range(S)]))
# 覆盖率（torch.quantile 与实验一致）
cl_levels = sorted({round(2*k/S, 4) for k in range(1, S//2+1) if 2*k/S < 1})
def cov_at(cl):
    ql = torch.quantile(samples, (1-cl)/2, dim=0); qh = torch.quantile(samples, 1-(1-cl)/2, dim=0)
    inside = (x_true[0] >= ql) & (x_true[0] <= qh)
    return float(inside.float().mean()), float(inside[obs].float().mean()), float(inside[mis].float().mean())
covs = {cl: cov_at(cl) for cl in cl_levels}
# PIT
lt = (samples < x_true).sum(0).float(); eq = (samples == x_true).sum(0).float()
gg = torch.Generator().manual_seed(20261003)
rank = lt + torch.rand(lt.shape, generator=gg) * (eq + 1.0)
stats["PIT_obs"], stats["PIT_mis"] = float(rank[obs].mean()), float(rank[mis].mean())
# MAP
if map_x is not None:
    stats["MAP缺失PSNR"] = psnr_t(x_true[0][mis], map_x[0][mis])
    stats["MAP整体PSNR"] = psnr_t(x_true, map_x)
    stats["RMS比_均值除MAP"] = (float((pm[0][mis]-x_true[0][mis]).pow(2).mean().sqrt())
                           / float((map_x[0][mis]-x_true[0][mis]).pow(2).mean().sqrt()))
# S4
m4 = samples[:4].mean(0)
stats["S4_PSNR"] = psnr_t(x_true, m4)
# 链间 rho
D = (samples - pm).reshape(S, -1); D = D - D.mean(0)
Dn = D / D.norm(dim=1, keepdim=True)
C = (Dn @ Dn.T).numpy(); off = C[np.triu_indices(S, 1)]
stats["rho"] = float(off.mean())

print("\n=== 重算 vs JSON ===")
keymap = [("后验均值PSNR_整体", "后验均值PSNR_整体"), ("后验均值PSNR_观测半区", "后验均值PSNR_观测半区"),
          ("后验均值PSNR_缺失半区", "后验均值PSNR_缺失半区"), ("单样本PSNR标准差", "单样本PSNR标准差"),
          ("缺失半区占平方误差比例", "缺失半区占后验均值平方误差比例"),
          ("std_观测半区", "std_观测半区"), ("std_缺失半区", "std_缺失半区"), ("std_整体", "std_整体"),
          ("lag1_缺失", "偏差场lag1_缺失半区"), ("lag1_全图", "偏差场lag1_全图"),
          ("PIT_obs", "PIT秩均值_观测半区"), ("PIT_mis", "PIT秩均值_缺失半区"),
          ("MAP缺失PSNR", "MAP基线PSNR_缺失半区"), ("MAP整体PSNR", "MAP基线PSNR_整体"),
          ("RMS比_均值除MAP", "缺失RMS比_后验均值除MAP"), ("S4_PSNR", "S4_PSNR"), ("rho", "链间相关系数均值")]
for a, b in keymap:
    v = stats.get(a); jv = js.get(b)
    tol = 0.02 if "PSNR" in a or a == "S4_PSNR" else (0.01 if "RMS" in a or "比例" in a or "std" in a or "lag1" in a or "PIT" in a or a == "rho" else 0.1)
    flag = "OK" if (v is not None and jv is not None and abs(v - jv) < tol) else "!!"
    print(f"  {a:28s} 重算={v!s:>10}  json={jv!s:>10} {flag}")
print(f"  单样本范围 重算=[{stats['单样本min']:.2f},{stats['单样本max']:.2f}] json={js['单样本PSNR范围']}")

print("\n=== 覆盖率重算 (pool/obs/mis) ===")
for cl, (p, o, m) in covs.items():
    jj = js["各置信水平覆盖率"].get(str(cl)) or js["各置信水平覆盖率"].get(str(round(cl,4)))
    jn = js["各置信水平零模型期望"].get(str(cl)) or js["各置信水平零模型期望"].get(str(round(cl,4)))
    print(f"  cl={cl:.4f}: 重算 {p:.4f}/{o:.4f}/{m:.4f}  json {jj}  零模型json {jn}")

# 零模型解析值：min-max 区间 (28/30 名义)
cl0 = 28/30
w = (1-cl0)/2 * (S-1)          # 分位数插值位置
tail = (1 - (w % 1) + (math.floor(w)+1)) / (S+1) if False else None
# ql 在第1、2次序统计量间：权重 (1-frac)*X_(1)+frac*X_(2)
frac = ((1-cl0)/2 * (S-1)) % 1
P_below = ((1-frac)*1 + frac*2) / (S+1)
print(f"\n零模型解析: 名义{cl0:.4f} 期望覆盖率 = {1-2*P_below:.4f} (json {js['主置信水平零模型期望覆盖率']})")

# 样本间多样性：缺失半区每像素 std / 样本间 L2
print("\n多样性检查: 样本-均值 L2/像素 =",
      [round(float((samples[i]-pm[0]).pow(2).mean().sqrt()), 4) for i in range(6)])
