# -*- coding: utf-8 -*-
"""独立复核 3.7-1：复现 JSON median/mean/√n + 标准误裁决 + 偏差断言"""
import numpy as np
np.random.seed(42)
dims=[2,10,100,1000]; ns=500
samples={d:np.random.randn(ns,d) for d in dims}
print("dim  median(J)        mean(J)          sqrt(n)      E||x||理论      SE")
for d in dims:
    norms=np.linalg.norm(samples[d],axis=1)
    med=np.median(norms); mn=norms.mean()
    # 理论 E||x|| = sqrt(2)*Gamma((d+1)/2)/Gamma(d/2)
    from math import lgamma, exp, sqrt as _s, log
    logE=0.5*log(2)+lgamma((d+1)/2)-lgamma(d/2)
    Eth=_s(exp(logE))
    se=norms.std(ddof=1)/np.sqrt(ns)
    print(f"{d:<5} {med:.4f}  {mn:.4f}  {np.sqrt(d):.4f}  {Eth:.4f}  {se:.4f}")
# n=2 讲义旧值1.392距理论mean/median的SE数
print("\n[旧捏造值裁决] n=2 讲义曾写1.392")
norms2=np.linalg.norm(samples[2],axis=1)
from math import lgamma,exp,sqrt,log
logE=0.5*log(2)+lgamma(1.5)-lgamma(1.0); Eth2=sqrt(exp(logE))
se2=norms2.std(ddof=1)/np.sqrt(ns)
print(f"  实测median={np.median(norms2):.4f} mean={norms2.mean():.4f} 理论E||x||={Eth2:.4f} sqrt2={sqrt(2):.4f} SE={se2:.4f}")
print(f"  1.392 距理论mean: {(1.392-Eth2)/se2:.1f} SE")
# 对√n最大偏差（讲义旧称1.17%）
print("\n[对√n偏差%] (median-sqrt(n))/sqrt(n)*100")
for d in dims:
    norms=np.linalg.norm(samples[d],axis=1); med=np.median(norms)
    print(f"  n={d}: {(med-np.sqrt(d))/np.sqrt(d)*100:+.2f}%")
