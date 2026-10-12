# -*- coding: utf-8 -*-
"""4.4-1 复核：数学单元测试
1) GradientIm / DivergenceIm 的伴随关系 <Du,p> = -<u, div p>
2) Chambolle prox 精度：MaxIter=1..100 相对收敛参考解的误差
3) Moreau 包络梯度 (x-prox)/lambda 的 1/lambda-Lipschitz 与单调性（与 beta 无关）
4) 各梯度分量与 prox 误差的量级对比
"""
import sys, math, json
import numpy as np
import torch
from skimage.data import camera
from skimage.transform import resize

EXP_DIR = r"d:\works\deepinv\第4章 MCMC与ULA算法\第四章 配套实验\实验4.4-1"
sys.path.insert(0, EXP_DIR)
from sampling_tools.blur_operators import blur_operators
from sampling_tools.chambolle_prox_TV import chambolle_prox_TV, GradientIm, DivergenceIm

torch.set_num_threads(8)
dev = torch.device('cpu')
results = {}

# ---------- 1) 伴随测试 ----------
torch.manual_seed(0)
u = torch.randn(64, 48)
p1 = torch.randn(64, 48); p2 = torch.randn(64, 48)
dux, duy = GradientIm(u, dev)
divp = DivergenceIm(p1, p2)
lhs = torch.sum(dux*p1 + duy*p2).item()
rhs = torch.sum(u*divp).item()
rel = abs(lhs + rhs) / abs(lhs)
results['adjoint'] = {'<Du,p>': lhs, '<u,div p>': rhs, 'rel_err_|<Du,p>+<u,divp>|/|<Du,p>|': rel}
print(f"[1] 伴随关系: <Du,p>={lhs:.6e}, <u,div p>={rhs:.6e}, 相对残差={rel:.3e} (应≈0)")

# 边界伴随单独检验：解析核对最后一列/最后一行系数
# 已在代码审查中推导，这里用有限结构验证 div 恰为 -D*
print("    -> 若相对残差<1e-6，则修复后的 DivergenceIm 严格等于 -D*")

# ---------- 2) prox 精度 ----------
u_np = resize(camera().astype(np.float64), (128, 128), anti_aliasing=True)
x_true = torch.tensor(u_np)
A, AT, AAT_norm = blur_operators([9, 9], (128, 128), "uniform", dev)
Ax = A(x_true)
sigma = torch.linalg.matrix_norm(Ax - torch.mean(Ax), ord='fro') / math.sqrt(128*128 * 10**(20/10))
y = Ax + sigma*torch.randn_like(x_true)
sigma2 = sigma.item()**2

for name, X in [('观测 y (链初态)', y), ('强平滑图像 (类似链稳态)', 0.5*(y+x_true))]:
    pref = chambolle_prox_TV(X, dev, {'lambda': 0.3, 'MaxIter': 5000, 'tol': 1e-13})
    g_ref = (X - pref)/1.0
    rows = {}
    for m in [1, 2, 5, 10, 20, 50, 100]:
        pm = chambolle_prox_TV(X, dev, {'lambda': 0.3, 'MaxIter': m, 'tol': 1e-13})
        e_prox = torch.norm(pm-pref).item()/torch.norm(pref).item()
        e_grad = torch.norm((X-pm)-(X-pref)).item()/torch.norm(X-pref).item()
        rows[m] = {'rel_err_prox': e_prox, 'rel_err_moreau_grad': e_grad}
    results[f'prox_accuracy[{name}]'] = rows
    print(f"\n[2] prox 精度 ({name}):")
    print(f"    {'MaxIter':>8} | {'prox相对误差':>14} | {'包络梯度相对误差':>16}")
    for m, r in rows.items():
        print(f"    {m:>8} | {r['rel_err_prox']:>14.3e} | {r['rel_err_moreau_grad']:>16.3e}")

# ---------- 3) Moreau 梯度 Lipschitz / 单调性 ----------
lam = 1.0
for beta in [0.0, 0.01, 0.3, 3.0]:
    ratios, monos = [], []
    for _ in range(200):
        x = torch.randn(32, 32)*50 + 128
        z = torch.randn(32, 32)*50 + 128
        px = chambolle_prox_TV(x, dev, {'lambda': lam*beta if beta > 0 else 1e-12, 'MaxIter': 500, 'tol': 1e-12})
        pz = chambolle_prox_TV(z, dev, {'lambda': lam*beta if beta > 0 else 1e-12, 'MaxIter': 500, 'tol': 1e-12})
        hx, hz = (x-px)/lam, (z-pz)/lam
        ratios.append((torch.norm(hx-hz)/torch.norm(x-z)).item())
        monos.append(torch.sum((hx-hz)*(x-z)).item())
    L_emp = max(ratios)
    mono_min = min(monos)
    results[f'moreau_lipschitz_beta={beta}'] = {'empirical_L': L_emp, 'theory_bound': 1/lam,
                                                 'monotonicity_min_inner': mono_min}
    print(f"[3] beta={beta:>5}: 经验L={L_emp:.4f} (理论界 {1/lam:.3f}), 单调性最小内积={mono_min:.3e} (应>=0)")

# ---------- 4) 梯度量级对比（在 y 处） ----------
gf = AT(A(y)-y)/sigma2
gg_ref = (y - chambolle_prox_TV(y, dev, {'lambda': 0.3, 'MaxIter': 5000, 'tol': 1e-13}))/1.0
gg_10 = (y - chambolle_prox_TV(y, dev, {'lambda': 0.3, 'MaxIter': 10}))/1.0
err = torch.norm(gg_10-gg_ref).item()
print(f"\n[4] 在 y 处: ||grad_f||={torch.norm(gf).item():.2f}, "
      f"||grad_g_lambda(精确)||={torch.norm(gg_ref).item():.2f}, "
      f"MaxIter=10 误差范数={err:.2f} (占先验梯度 {err/torch.norm(gg_ref).item()*100:.1f}%)")
results['gradient_scale_at_y'] = {'grad_f_norm': torch.norm(gf).item(),
                                  'grad_g_norm_exact': torch.norm(gg_ref).item(),
                                  'prox10_grad_err_norm': err,
                                  'rel_err_pct': err/torch.norm(gg_ref).item()*100}

with open(r'd:\works\deepinv\_review_scratch\units_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n已保存 units_results.json")
