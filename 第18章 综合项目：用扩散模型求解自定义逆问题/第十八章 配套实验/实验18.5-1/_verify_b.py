# -*- coding: utf-8 -*-
"""B: 提取 18.5-1.py 的 Step 0 之前全部定义，在解析可解问题上：
1) 原样重跑 Step 0（显式梯度路径）→ 核对 json 的 1.008/1.104/0.91
2) 额外用 MYULA 近端路径（v3 真实实验所用、Step 0 未覆盖）跑同一问题 → 验证其标定
"""
import os, sys, json, math
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "18.5-1.py")
src = open(SRC, encoding="utf-8").read()
cut = src.index("# ========================================================================\n# Step 0\n")
exec(compile(src[:cut], SRC, "exec"), globals())

print("\n>>> B1: 原样重跑 Step 0（显式梯度路径，与 json 记录同配置）")
st = run_analytic_selftest(CFG)
js = json.load(open(os.path.join(SAVE_DIR, "uq_results.json"), encoding="utf-8"))
print(f"\n[B1 核对] Step0 std比: 重算 obs={st['ratio_obs']:.3f} mis={st['ratio_mis']:.3f} cov={st['cov']:.4f}")
print(f"[B1 核对] json 记录  obs={js['Step0_std比_观测']} mis={js['Step0_std比_缺失']} cov={js['Step0_覆盖率']}")
# std 比应与 JSON 一致（覆盖率允许在正文声明的 91~96% 波动内，不做硬断言）
b1_ok = (abs(st['ratio_obs'] - js['Step0_std比_观测']) < 0.01
         and abs(st['ratio_mis'] - js['Step0_std比_缺失']) < 0.01)
print(f"[B1 结论] {'✓ std 比与 JSON 一致' if b1_ok else '✗ std 比与 JSON 不一致'}")

# ---------------- B2: MYULA 近端路径（Step 0 未覆盖） ----------------
print("\n>>> B2: 同一解析问题，改用 MYULA 近端数据项（prox 路径，v3 真实配置）")
torch.set_default_dtype(torch.float64)
prob = _AnalyticGaussian(side=10)
N = prob.N
gen = torch.Generator().manual_seed(7)
mask = (torch.rand(N, generator=gen) < 0.5).double()
x_true = (torch.randn(1, N, generator=gen) * prob.lam.sqrt()) @ prob.U.T + prob.mean
x_true = x_true[0]
y = mask * x_true + SIGMA_DATA * torch.randn(N, generator=gen)
ym = mask * y
mu0, C0 = prob.posterior(mask, ym, 0.0)
sd_ref = C0.diag().clamp_min(0).sqrt()
obs, mis = mask > 0.5, mask < 0.5
A = lambda t: t * mask
At = lambda t: t * mask

def inpaint_prox(x, a):
    # 与 18.5-1.py 中 _inpaint_prox 完全一致的形式（批量版）
    return (x + a * ym.reshape(1, 1, -1)) / (1.0 + a * mask.reshape(1, 1, -1))

n_chains = 300
smp = sample_annealed_ula(ym.reshape(1, 1, -1), A, At, prob.denoiser, CFG,
                          n_chains, torch.device('cpu'), seed=11,
                          desc="自检prox", verbose=False, prox_op=inpaint_prox)
smp = smp.reshape(n_chains, N)
emp_sd = smp.var(0, unbiased=True).clamp_min(0).sqrt()
ql = torch.quantile(smp, (1 - 0.9333333) / 2, dim=0)
qh = torch.quantile(smp, 1 - (1 - 0.9333333) / 2, dim=0)
inside = (x_true >= ql) & (x_true <= qh)
# emp_sd 与 sd_ref 都已是标准差，直接作比即为“实测 std / 真实后验 std”。
r_o = float(emp_sd[obs].mean() / sd_ref[obs].mean())
r_m = float(emp_sd[mis].mean() / sd_ref[mis].mean())
cov = float(inside.float().mean())
print(f"[B2 结果] MYULA prox 路径: std比 观测={r_o:.3f} 缺失={r_m:.3f} 93.3%覆盖率={cov*100:.1f}%")
ok = (0.8 <= r_o <= 1.25) and (0.7 <= r_m <= 1.4) and (cov >= 0.85)
print(f"[B2 结论] {'✓ prox 路径通过解析验证（与显式梯度路径同样正确）' if ok else '✗ prox 路径未通过'}")
if not (b1_ok and ok):
    sys.exit(1)

# 参考：精确后验样本
ref = prob.sample_posterior(mu0, C0, n_chains, gen)
emp_sd_ref = ref.var(0, unbiased=True).clamp_min(0).sqrt()
r_o_ref = float(emp_sd_ref[obs].mean() / sd_ref[obs].mean())
r_m_ref = float(emp_sd_ref[mis].mean() / sd_ref[mis].mean())
print(f"[B2 参考] 精确后验样本自身（蒙特卡洛误差）: 观测={r_o_ref:.3f} 缺失={r_m_ref:.3f}")

torch.set_default_dtype(torch.float32)
