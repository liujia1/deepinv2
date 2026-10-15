# -*- coding: utf-8 -*-
"""实验3.7-1 独立复核脚本（不修改实验目录任何文件）。
1) 逐位复现 seed=42 的 JSON 数值；
2) 多种子独立抽样对照 scipy.stats.chi 精确矩/分位数；
3) 校验代码里手写的 chi(n) PDF 公式、CLT 标准化、壳层概率、近众数概率。
"""
import json
import math
import numpy as np
from scipy import stats

EXP_DIR = r"d:\works\deepinv\第3章 从MAP到后验探索\第三章 配套实验\实验3.7-1"
dims = [2, 10, 100, 1000]
N = 500

print("=" * 70)
print("【1】逐位复现：seed=42 + 与原脚本完全相同的 RNG 调用顺序")
print("=" * 70)
with open(EXP_DIR + r"\results_summary.json", encoding="utf-8") as f:
    saved = json.load(f)["scene_a_typical_set"]

np.random.seed(42)
repro = {}
for d in dims:  # 原脚本按 [2,10,100,1000] 顺序各 randn(500,d)
    x = np.random.randn(N, d)
    r = np.linalg.norm(x, axis=1)
    repro[d] = (np.median(r), np.mean(r))

ok = True
for d in dims:
    med_json = saved["typical_norm_median"][str(d)]
    mean_json = saved["mean_norm"][str(d)]
    med_r, mean_r = repro[d]
    m_ok = math.isclose(med_r, med_json, rel_tol=0, abs_tol=1e-12)
    a_ok = math.isclose(mean_r, mean_json, rel_tol=0, abs_tol=1e-12)
    ok &= m_ok and a_ok
    print(f"n={d:<5}: median repro={med_r:.10f} json={med_json:.10f} "
          f"{'OK' if m_ok else 'MISMATCH'} | mean repro={mean_r:.10f} "
          f"json={mean_json:.10f} {'OK' if a_ok else 'MISMATCH'}")
print("逐位复现结论:", "全部一致，JSON 确为该脚本 seed=42 的真实输出" if ok else "存在不一致！")

print()
print("=" * 70)
print("【2】理论精确值（chi(n) 分布）与 seed=42 结果的蒙特卡洛误差")
print("=" * 70)
print(f"{'n':>5} {'量':>6} {'经验(seed42)':>14} {'理论精确值':>14} "
      f"{'偏差':>10} {'MC标准误':>10} {'z(偏差/SE)':>10}")
theory_rows = []
for d in dims:
    chi = stats.chi(d)
    med_t, mean_t, std_t = chi.ppf(0.5), chi.mean(), chi.std()
    med_e, mean_e = repro[d]
    # 样本均值 SE = std/sqrt(N)；样本中位数渐近 SE = 1/(2 f(m) sqrt(N))
    se_mean = std_t / math.sqrt(N)
    se_med = 1.0 / (2 * chi.pdf(med_t) * math.sqrt(N))
    print(f"{d:>5} {'median':>6} {med_e:>14.6f} {med_t:>14.6f} "
          f"{med_e-med_t:>+10.5f} {se_med:>10.5f} {(med_e-med_t)/se_med:>+10.2f}")
    print(f"{d:>5} {'mean':>6} {mean_e:>14.6f} {mean_t:>14.6f} "
          f"{mean_e-mean_t:>+10.5f} {se_mean:>10.5f} {(mean_e-mean_t)/se_mean:>+10.2f}")
    theory_rows.append((d, med_t, mean_t, std_t))

print()
print("sqrt(n) 集中半径 vs 精确期望：")
for d in dims:
    mean_t = stats.chi(d).mean()
    print(f"  n={d:<5}: sqrt(n)={math.sqrt(d):.6f}, E||x||={mean_t:.6f}, "
          f"相对差={ (math.sqrt(d)-mean_t)/math.sqrt(d)*100:.3f}%  "
          f"(正文称 sqrt(n) 是集中半径而非均值，与 E||x||<sqrt(n) 一致: "
          f"{mean_t < math.sqrt(d)})")

print()
print("=" * 70)
print("【3】多种子（200 个独立种子, 每种子500样本）独立验证覆盖性")
print("=" * 70)
rng = np.random.default_rng(20261014)
N_SEEDS = 200
for d in dims:
    meds, means = [], []
    chi = stats.chi(d)
    for _ in range(N_SEEDS):
        x = rng.standard_normal((N, d))
        r = np.linalg.norm(x, axis=1)
        meds.append(np.median(r))
        means.append(np.mean(r))
    meds, means = np.array(meds), np.array(means)
    med_e, mean_e = repro[d]
    med_t, mean_t = chi.ppf(0.5), chi.mean()
    pct_med = (meds < med_e).mean()
    pct_mean = (means < mean_e).mean()
    print(f"n={d:<5}: seed42 median={med_e:.4f} 在{N_SEEDS}种子分布中的分位={pct_med*100:5.1f}% "
          f"(理论中位{med_t:.4f}, 跨种子2.5%-97.5%=[{np.percentile(meds,2.5):.4f},"
          f"{np.percentile(meds,97.5):.4f}])")
    print(f"       seed42 mean  ={mean_e:.4f} 分位={pct_mean*100:5.1f}% "
          f"(理论均值{mean_t:.4f}, 区间=[{np.percentile(means,2.5):.4f},"
          f"{np.percentile(means,97.5):.4f}])")

print()
print("=" * 70)
print("【4】代码手写 chi(n) PDF 公式 vs scipy 精确值")
print("=" * 70)
for d in [10]:
    grid = np.linspace(0.01, 8, 200)
    hand = (grid ** (d - 1)) * np.exp(-grid ** 2 / 2) / \
           (2 ** (d / 2 - 1) * math.gamma(d / 2))
    ref = stats.chi(d).pdf(grid)
    print(f"  n={d}: max|手写PDF - scipy chi.pdf| = {np.max(np.abs(hand-ref)):.3e} "
          f"({'公式正确' if np.allclose(hand, ref) else '公式错误'})")

print()
print("=" * 70)
print("【5】n=1000 标准化范数 (R-sqrt(n))/(1/sqrt(2)) 的 CLT 检验")
print("=" * 70)
np.random.seed(42)
for _ in dims:
    pass
np.random.seed(42)
_ = {d: np.random.randn(N, d) for d in dims}  # 复刻调用顺序
r1000 = np.linalg.norm(_[1000], axis=1)
z = (r1000 - math.sqrt(1000)) / (1 / math.sqrt(2))
print(f"  z 均值={z.mean():+.4f} (理论≈{(stats.chi(1000).mean()-math.sqrt(1000))/(1/math.sqrt(2)):+.4f})")
print(f"  z 标准差={z.std(ddof=1):.4f} (理论精确 std(R)/(1/sqrt2)="
      f"{stats.chi(1000).std()/(1/math.sqrt(2)):.4f}, CLT 目标 1)")
print(f"  z 偏度={stats.skew(z):+.4f} (CLT 目标 0)")
# KS 检验对 N(0,1)
ks = stats.kstest(z, "norm")
print(f"  Kolmogorov-Smirnov vs N(0,1): D={ks.statistic:.4f}, p={ks.pvalue:.4f} "
          f"({'不拒绝正态' if ks.pvalue > 0.05 else '拒绝'})")

print()
print("=" * 70)
print("【6】核心论点量化：众数附近的概率 vs 薄壳集中")
print("=" * 70)
for d in dims:
    r = np.linalg.norm(_[d], axis=1)
    # 球 B(0, radius) 的精确概率 = P(chi2(d) <= radius^2)
    for radius in ([0.5, 1.0] if d <= 10 else [1.0, 2.0, 5.0]):
        p = stats.chi2.cdf(radius ** 2, d)
        emp = (r <= radius).mean()
        print(f"  n={d:<5}: P(||x||<={radius}) 精确={p:.3e} 经验({N}样本)={emp:.3e}")
    # 薄壳：R 落在 E[R]±2std 内的概率（精确）
    chi = stats.chi(d)
    lo, hi = chi.mean() - 2 * chi.std(), chi.mean() + 2 * chi.std()
    shell_p = chi.cdf(hi) - chi.cdf(lo)
    cv = chi.std() / chi.mean()
    print(f"  n={d:<5}: 相对壳厚 2std/E[R]={2*cv:.4f} (随 n→0 => {2*cv < (0.5 if d>2 else 1)}), "
          f"±2std 带内精确质量={shell_p:.4f}, 样本最小范数={r.min():.3f}")
    print()

print("=" * 70)
print("【7】左图 n=2 “典型圆 r=sqrt(2)” 的实际含义")
print("=" * 70)
# R^2~chi2(2): P(R<sqrt(2)) = 1-exp(-1)
p_in = 1 - math.exp(-1)
r2 = np.linalg.norm(_[2], axis=1)
print(f"  n=2: 圆内样本占比经验={ (r2<math.sqrt(2)).mean():.3f}, "
      f"理论 1-e^-1={p_in:.3f} -> 2维时近 63% 样本在“典型圆”内，壳层很不集中（符合小维度假定）")

print()
print("=" * 70)
print("【8】正文‘n=1000 median/mean 均在 sqrt(n) 的 0.08% 以内’核算")
print("=" * 70)
s = math.sqrt(1000)
med_e, mean_e = repro[1000]
print(f"  |median-sqrt(n)|/sqrt(n)={abs(med_e-s)/s*100:.4f}%")
print(f"  |mean  -sqrt(n)|/sqrt(n)={abs(mean_e-s)/s*100:.4f}%  (声称<0.08%: "
      f"{abs(med_e-s)/s*100 < 0.08 and abs(mean_e-s)/s*100 < 0.08})")
