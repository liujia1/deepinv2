# -*- coding: utf-8 -*-
import numpy as np, os
base = r"d:\works\deepinv\第5章 朗之万动力学与得分函数\第五章配套实验"
p = os.path.join(base, "实验5.5-1", "sampling_results.npz")
d = np.load(p)
print("keys:", list(d.keys()))
for k in d.keys():
    print(k, d[k].shape, d[k].dtype)
mc = d["mc_samples"]; pm = d["post_mean"]; pv = d["post_var"]; xt = d["x_true"]; yo = d["y_obs"]
print("num mc_samples:", len(mc))
# recompute as 5.6-1 does
pmr = np.mean(mc, axis=0)
pvr = np.var(mc, axis=0, ddof=1)
print("mean L2 diff:", np.linalg.norm(pm - pmr))
print("var  L2 diff:", np.linalg.norm(pv - pvr))
std = np.sqrt(pvr)
print("post_std mean %.6f max %.6f min %.6f median %.6f" % (np.mean(std), np.max(std), np.min(std), np.median(std)))
thr = np.mean(std) + np.std(std)
hu = std > thr
print("threshold %.6f  high-unc pct %.4f" % (thr, np.sum(hu)/hu.size*100))
err = np.abs(xt - pm)
corr = np.corrcoef(std.flatten(), err.flatten())[0,1]
print("corr(std,err) = %.6f" % corr)
# also corr using original post_var from npz (not recomputed)
std_o = np.sqrt(pv)
print("corr(std_orig,err) = %.6f" % np.corrcoef(std_o.flatten(), err.flatten())[0,1])
# range checks
print("x_true range", xt.min(), xt.max(), " post_mean range", pm.min(), pm.max())
print("y_obs range", yo.min(), yo.max())
# quantile bins
nb=5
bins=np.percentile(std.flatten(), np.linspace(0,100,nb+1))
sf=std.flatten(); ef=err.flatten()
bm=[]
for i in range(nb):
    m=(std>=bins[i])&(std<bins[i+1])
    bm.append(np.mean(err[m]) if m.sum()>0 else np.nan)
print("bin means:", np.round(bm,5))
