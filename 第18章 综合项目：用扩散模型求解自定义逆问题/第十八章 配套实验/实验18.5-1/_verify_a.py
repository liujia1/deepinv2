# -*- coding: utf-8 -*-
"""独立验证脚本：不复用 18.5-1.py 的统计代码，从缓存与第一性原理重算。

路径按脚本所在目录解析，可在任意工作区直接运行；所有可严格复算的指标
都会与 uq_results.json 比较，任一失败则退出码为 1。
"""
import math, json, sys, os
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "experiment_cache_S30.pkl")
JSONF = os.path.join(HERE, "uq_results.json")

obj = torch.load(CACHE, map_location="cpu", weights_only=False)
S_samples = obj["samples"]; mask = obj["mask"]; x_true = obj["x_true"]; map_x = obj["map_x"]; y_inp = obj["y_inp"]
SAMPLES = torch.cat(S_samples, 0)  # (S,3,128,128)
S = SAMPLES.shape[0]
js = json.load(open(JSONF, encoding="utf-8"))

failures = []


def check(name, mine, ref, tol):
    ok = abs(float(mine) - float(ref)) <= tol
    print(f"{'OK  ' if ok else 'FAIL'} {name}: 重算={float(mine):.6f}  参考={ref}  tol={tol:g}")
    if not ok:
        failures.append(name)
    return ok


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)


pm = SAMPLES.mean(0, keepdim=True)
ps = SAMPLES.std(0, keepdim=True, unbiased=True)
m_el = mask.bool()
xt, pmv, psv = x_true[0], pm[0], ps[0]

print("=== A. 从缓存独立重算 vs uq_results.json ===")
rows = [
    ("后验均值PSNR_整体", psnr(x_true, pm), js["后验均值PSNR_整体"]),
    ("后验均值PSNR_观测半区", psnr(xt[m_el[0]], pmv[m_el[0]]), js["后验均值PSNR_观测半区"]),
    ("后验均值PSNR_缺失半区", psnr(xt[~m_el[0]], pmv[~m_el[0]]), js["后验均值PSNR_缺失半区"]),
    # 观测/缺失半区：JSON 记录的是该半区逐像素 std 的 RMS
    ("std_观测半区", float(psv[m_el[0]].pow(2).mean().sqrt()), js["std_观测半区"]),
    ("std_缺失半区", float(psv[~m_el[0]].pow(2).mean().sqrt()), js["std_缺失半区"]),
    # 整体：JSON 记录的是逐像素 std 的均值（不是 RMS），口径必须与主脚本一致
    ("std_整体", float(psv.mean()), js["std_整体"]),
    ("MAP缺失PSNR", psnr(xt[~m_el[0]], map_x[0][~m_el[0]]), js["MAP基线PSNR_缺失半区"]),
    ("MAP整体PSNR", psnr(x_true, map_x), js["MAP基线PSNR_整体"]),
]
for name, mine, ref in rows:
    # JSON 只保留 2~4 位小数；PSNR 用 0.01，std 量级小、用 1e-4 才能抓住实质差异
    tol = 1e-4 if name.startswith("std_") else 0.01
    check(name, mine, ref, tol)

sps = [psnr(x_true, SAMPLES[i:i+1]) for i in range(S)]
js_range = js["单样本PSNR范围"]
check("单样本PSNR_min", min(sps), js_range[0], 0.01)
check("单样本PSNR_max", max(sps), js_range[1], 0.01)
check("单样本PSNR标准差", float(np.std(sps)), js["单样本PSNR标准差"], 1e-3)

# 覆盖率（主水平 0.9333）
cl = 0.9333
ql = torch.quantile(SAMPLES, (1-cl)/2, dim=0); qh = torch.quantile(SAMPLES, 1-(1-cl)/2, dim=0)
inside = (x_true[0] >= ql) & (x_true[0] <= qh)
pool = float(inside.float().mean()); obs = float(inside[m_el[0]].float().mean()); mis = float(inside[~m_el[0]].float().mean())
check("覆盖率_池化", pool, js["主置信水平覆盖率_池化"], 1e-4)
check("覆盖率_观测半区", obs, js["主置信水平覆盖率_观测半区"], 1e-4)
check("覆盖率_缺失半区", mis, js["主置信水平覆盖率_缺失半区"], 1e-4)

# PIT（脚本用中点化结随机化 rank = #below + U(-0.5,0.5)，均值=#below；理论均值 S/2=15）
ranks = (SAMPLES < x_true).sum(0).float()
check("PIT秩_观测半区", float(ranks[m_el[0]].mean()), js["PIT秩均值_观测半区"], 0.01)
check("PIT秩_缺失半区", float(ranks[~m_el[0]].mean()), js["PIT秩均值_缺失半区"], 0.01)

# 零模型期望覆盖率（独立蒙特卡洛，比脚本的 12 万次更大）
rng = np.random.default_rng(123)
z = rng.standard_normal((2_000_000, S)); t = rng.standard_normal(2_000_000)
a = (1-cl)/2
null = float(((t >= np.quantile(z, a, axis=1)) & (t <= np.quantile(z, 1-a, axis=1))).mean())
check("零模型期望覆盖率", null, js["主置信水平零模型期望覆盖率"], 0.002)
check("实测覆盖率偏差_对零模型", pool - null, js["主水平覆盖率偏差_对零模型"], 0.002)
print(f"     -> 实测池化 {pool:.4f} 对名义偏差 {pool-cl:+.4f}（json: {js['主水平覆盖率偏差_对名义']:+.4f}）")

# MYULA prox 公式核验（解析）
# g(x)=½||Mx-y||²/σ_d²; prox_{εg}(x)=argmin ½(x'-x)²+ε·g(x')
# 观测像素: (x'-x)+(ε/σ_d²)(x'-y)=0 -> x'=(x+(ε/σ_d²)y)/(1+ε/σ_d²)
# 脚本: prox_a=ε·λ/σ_d², prox=(x+prox_a·y)/(1+prox_a) —— λ=1 时应与上式一致
sd2 = 0.01**2
for eps_test in [1.92e-4, 3.07e-7]:
    a_p = eps_test * 1.0 / sd2          # = prox_a
    x_val, y_val = 0.3, 0.8
    closed = (x_val + a_p*y_val)/(1.0 + a_p)          # 脚本公式
    xs = np.linspace(-1, 2, 3000001)
    f = (xs-x_val)**2 + a_p*(xs-y_val)**2             # 正确目标（½ 因子对方程无影响）
    num = xs[f.argmin()]
    check(f"prox公式_eps{eps_test:.1e}", closed, num, 1e-6)

# 图像边界：观测半区样本是否真的被数据钉住
err_obs = float((SAMPLES[:, m_el[0]] - y_inp[0][m_el[0]]).pow(2).mean().sqrt())
print(f"INFO 样本-观测 在观测半区 RMS = {err_obs:.5f} (应远小于 sigma_d=0.01 量级; 含 MYULA O(eps) 近似误差)")

# 结构：缺失半区样本偏差 lag1 独立重算
# 注意：本脚本是"先对通道平均再算相关"的近似；脚本 _deviation_lag1 是"逐通道算相关再平均"，
# 两者在通道强相关时数值不同，以脚本口径（json 记录值）为准，因此这里只打印、不做断言。
dev = (SAMPLES - pm)
w = (1.0 - mask[0].float())
def lag1(field, wt):
    a = field.numpy(); a = np.where(wt.numpy() > 0.5, a, 0.0); a = a - a.mean()
    v = float((a*a).mean())
    return 0.5*(float((a[:-1,:]*a[1:,:]).mean()/v) + float((a[:,:-1]*a[:,1:]).mean()/v)) if v > 1e-12 else float('nan')
lag_dev = float(np.mean([lag1(dev[s].mean(0, keepdim=True)[0], w) for s in range(S)]))
print(f"INFO 缺失半区偏差 lag1: 重算={lag_dev:.4f} json={js['偏差场lag1_缺失半区']}（口径不同，仅参考）")

# 偏差分解：后验均值的 PSNR 只由偏差^2 决定；bias^2+var 是贝叶斯预测 MSE，不是 PSNR 口径
err_m = pmv[~m_el[0]] - xt[~m_el[0]]
bias2 = float(err_m.pow(2).mean()); var_m = float(psv[~m_el[0]].pow(2).mean())
check("缺失半区偏差^2隐含PSNR", 10*math.log10(1/bias2), js["后验均值PSNR_缺失半区"], 0.01)
print(f"INFO 缺失半区: 偏差^2={bias2:.6f} 后验方差={var_m:.6f} -> 贝叶斯预测 MSE={bias2+var_m:.6f}")

print()
if failures:
    print(f"[FAIL] {len(failures)} 项未通过: {failures}")
    sys.exit(1)
print("[PASS] 所有可严格复算的指标均与 uq_results.json 一致")
