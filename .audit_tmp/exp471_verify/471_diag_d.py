"""复核脚本 D：单链 M=49 白噪声的 ACF 波动幅度 + '从大量分量中选最大方差者'的选择效应
目的：判断讲义中最慢分量 ACF 的'明显周期性波动'（-0.4 波谷）是真实混合行为，还是短链+选择效应伪影。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
M = 49
n_rep = 3000          # 独立链条数
n_pix = 2000          # 每条链的"像素"数（模拟高维选择）

dips = []
dips_maxvar = []
for _ in range(n_rep):
    ch = rng.standard_normal((n_pix, M))
    ch = ch - ch.mean(axis=1, keepdims=True)
    ac = np.fft.irfft(np.abs(np.fft.rfft(ch, n=2 * M, axis=1)) ** 2, n=2 * M, axis=1)[:, :M]
    ac = ac / ac[:, :1]                      # 归一化 ACF
    d = ac[:, 1:].min(axis=1)                # 每个分量的 ACF 最小值
    dips.append(d)
    idx = int(np.argmax(ch.var(axis=1)))     # 选择效应：方差最大的分量
    dips_maxvar.append(d[idx])

dips = np.array(dips); dips_maxvar = np.array(dips_maxvar)
print(f"独立白噪声链 M={M}, 每链 {n_pix} 分量, {n_rep} 次重复")
print(f"  单分量 ACF 最小值分布: 中位数={np.median(dips):+.3f}, 5%分位={np.percentile(dips,5):+.3f}, 1%分位={np.percentile(dips,1):+.3f}")
print(f"  P(min ACF < -0.40) = {(dips < -0.40).mean():.3f}   （讲义最慢分量波谷约 -0.41）")
print(f"  '最大方差分量'的 ACF 最小值: 中位数={np.median(dips_maxvar):+.3f}, 1%分位={np.percentile(dips_maxvar,1):+.3f}")
print(f"  P(max-var 分量 min ACF < -0.40) = {(dips_maxvar < -0.40).mean():.3f}")

plt.figure(figsize=(7, 4))
plt.hist(dips, bins=60, alpha=0.6, label='all components')
plt.hist(dips_maxvar, bins=60, alpha=0.6, label='max-variance component')
plt.axvline(-0.41, color='r', ls='--', label='slowest component in exp (-0.41)')
plt.xlabel('min ACF value of a single chain (M=49)'); plt.legend(); plt.tight_layout()
plt.savefig(r"D:\works\deepinv\.audit_tmp\exp471_verify\acf_dip_null.png", dpi=130)
print("saved acf_dip_null.png")
