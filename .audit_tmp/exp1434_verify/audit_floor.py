# -*- coding: utf-8 -*-
"""补充检查: 切片W2的有限样本地板 & 图3面板(800样本,seed2029)与表(2000样本)差异来源"""
import numpy as np, torch, importlib.util, sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
spec = importlib.util.spec_from_file_location("p", r"d:\works\deepinv\.audit_tmp\exp1434_verify\14.3-4.py")

# 不重跑训练: 只复制所需函数
def sliced_w2(a, b, n_proj=32, seed=42):
    g = torch.Generator().manual_seed(seed)
    a = torch.tensor(a, dtype=torch.float64); b = torch.tensor(b, dtype=torch.float64)
    tot = 0.0
    for _ in range(n_proj):
        u = torch.randn(2, generator=g, dtype=torch.float64); u = u / u.norm()
        pa = (a @ u).sort().values; pb = (b @ u).sort().values
        tot += float(((pa - pb) ** 2).mean())
    return np.sqrt(tot / n_proj)

MU1 = np.array([-2., -2.]); S1 = np.array([.36, .81])
MU2 = np.array([2., 2.]);  S2 = np.array([.81, .36])
def sample_ref(n, seed):
    rng = np.random.default_rng(seed)
    c = rng.choice(2, size=n, p=[.5, .5])
    out = np.empty((n, 2))
    for k, (mu, sg) in enumerate([(MU1, S1), (MU2, S2)]):
        msk = c == k
        out[msk] = mu + rng.standard_normal((msk.sum(), 2)) * np.sqrt(sg)
    return out

for n in [2000, 800]:
    vals = [sliced_w2(sample_ref(n, 100 + i), sample_ref(n, 500 + i)) for i in range(5)]
    print(f"真值vs真值 ({n}样本) 地板: " + "  ".join(f"{v:.4f}" for v in vals) + f"   mean={np.mean(vals):.4f}")
