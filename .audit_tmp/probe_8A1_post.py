# -*- coding: utf-8 -*-
"""8A-1 后验形态与间隙构成探针：确认配置A后验是否双峰、配置B间隙是否≈放弃模式的质量代价。"""
import numpy as np
from scipy.integrate import quad

def analyze(wts, means, stds, sobs, xobs, label):
    def log_prior(z):
        lp = np.full_like(np.asarray(z, dtype=float), -1e30)
        for w, mu, tau in zip(wts, means, stds):
            lp = np.logaddexp(lp, np.log(w) - 0.5*np.log(2*np.pi*tau**2) - 0.5*((z-mu)/tau)**2)
        return lp
    def log_joint(z):
        z = np.asarray(z, dtype=float)
        return (-0.5*np.log(2*np.pi*sobs**2) - 0.5*((xobs-z)/sobs)**2) + log_prior(z)
    Zc, err = quad(lambda t: np.exp(log_joint(t)), min(means)-12, max(means)+12, limit=800)
    log_px = np.log(Zc)
    # 后验各分量解析参数
    prec = [1/tau**2 for tau in stds]; lp_ = 1/sobs**2
    comp_mean = [(m*p + xobs*lp_)/(p+lp_) for m, p in zip(means, prec)]
    comp_std = [np.sqrt(1/(p+lp_)) for p in prec]
    raw = [w*np.exp(log_joint(m) - (-0.5*np.log(2*np.pi*sobs**2)-0.5*((xobs-m)/sobs)**2))
           for w, m in zip(wts, means)]  # = w * N(x; m, sqrt(tau^2+s^2)) 差常数，直接按边际算
    raw = [w * np.exp(-0.5*np.log(2*np.pi*(tau**2+sobs**2)) - 0.5*((xobs-m)/np.sqrt(tau**2+sobs**2))**2)
           for w, m, tau in zip(wts, means, stds)]
    tot = sum(raw); pis = [r/tot for r in raw]
    sep = abs(comp_mean[1]-comp_mean[0]); cstd = comp_std[0]
    zs = np.linspace(min(means)-6, max(means)+6, 12001)
    p = np.exp(log_joint(zs)); p /= p.sum()
    loc = [zs[i] for i in range(1, len(zs)-1) if p[i] > p[i-1] and p[i] >= p[i+1] and p[i] > 1e-4*p.max()]
    print(f"== {label}")
    print(f"   log p(x)={log_px:.6f}  后验权重={['%.4f' % v for v in pis]}")
    print(f"   后验分量均值={['%.4f' % v for v in comp_mean]}  分量std={cstd:.4f}  间距={sep:.4f}  双峰阈值2*std={2*cstd:.4f} → {'双峰' if sep > 2*cstd else '单峰'}")
    print(f"   密度局部极大(>1e-4峰值): {['%.3f' % m for m in loc]}")
    print(f"   -log(max pi) = {-np.log(max(pis)):.6f}")
    return log_px, pis

analyze([0.3, 0.7], [-2.0, 1.0], [1.0, 1.0], 0.5, 0.5, "配置A 原设定")
analyze([0.3, 0.7], [-2.5, 2.5], [0.5, 0.5], 0.4, 0.0, "配置B 强分离")
analyze([0.5, 0.5], [-3.0, 3.0], [0.6, 0.6], 0.5, 0.0, "配置B' 对称")
