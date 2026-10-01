# -*- coding: utf-8 -*-
"""实验13.1-1 审计：解析后验 vs DPS采样 对照检验"""
import sys, io, json, os
import numpy as np
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# ---------- 复现脚本的随机流 ----------
np.random.seed(42)
r = np.random.rand()
branch = 0 if r < 0.3 else 1
x0_star = (np.random.randn() - 2) if branch == 0 else (np.random.randn() + 1)
sigma_obs = 0.5
y_obs = x0_star + sigma_obs * np.random.randn()
print(f"[seed流] rand()={r:.6f} -> 分支={'N(-2,1)' if branch==0 else 'N(1,1)'}")
print(f"[seed流] x0_star={x0_star:.4f}, y_obs={y_obs:.4f}")

# ---------- SDE 公共函数（与脚本一致） ----------
def vp_marginal(t, beta_min=0.1, beta_max=20.0):
    log_mean = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    return np.exp(log_mean), np.sqrt(1 - np.exp(2 * log_mean))

def vp_beta(t, beta_min=0.1, beta_max=20.0):
    return beta_min + t * (beta_max - beta_min)

W = [0.3, 0.7]; M = [-2.0, 1.0]; S = [1.0, 1.0]

def gm_mixture_score(x, mus, varis, wts):
    pdf = np.zeros_like(x); dpdf = np.zeros_like(x)
    for w, m, v in zip(wts, mus, varis):
        e = np.exp(-0.5 * ((x - m)**2) / v) / np.sqrt(2 * np.pi * v)
        pdf += w * e
        dpdf += w * (-(x - m) / v) * e
    return dpdf / (pdf + 1e-30)

def prior_score(x, t):
    mean_t, std_t = vp_marginal(t)
    mus = [mean_t * m for m in M]
    varis = [mean_t**2 * s**2 + std_t**2 for s in S]
    return gm_mixture_score(x, mus, varis, W)

def dps_score(x, t, zeta=1.0, scale='mean_t'):
    mean_t, std_t = vp_marginal(t)
    ps = prior_score(x, t)
    x0_hat = (x + std_t**2 * ps) / (mean_t + 1e-10)
    if scale == 'mean_t':
        lg = mean_t / sigma_obs**2 * (y_obs - x0_hat)
    elif scale == 'none':
        lg = (y_obs - x0_hat) / sigma_obs**2
    return ps + zeta * lg

def exact_post_score(x, t):
    """真后验 p(x0|y) 经前向SDE扩散后的精确得分（金标准）"""
    # t=0 后验混合
    comp_var = np.array(S)**2
    obs_var = comp_var + sigma_obs**2
    alpha = np.array(W) * np.exp(-0.5 * (y_obs - np.array(M))**2 / obs_var) / np.sqrt(obs_var)
    alpha /= alpha.sum()
    vpost = (np.array(S)**2 * sigma_obs**2) / (np.array(S)**2 + sigma_obs**2)
    mpost = vpost * (np.array(M) / np.array(S)**2 + y_obs / sigma_obs**2)
    mean_t, std_t = vp_marginal(t)
    mus = mean_t * mpost
    varis = mean_t**2 * vpost + std_t**2
    return gm_mixture_score(x, mus, varis, alpha), alpha, mpost, vpost

def run_reverse(score_fn, N=5000, steps=300, T=1.0, seed=42):
    np.random.seed(seed)
    h = T / steps
    x = np.random.randn(N)
    for i in range(steps):
        t = T - i * h
        b = vp_beta(t)
        x = x + b * h * (0.5 * x + score_fn(x, t)) + np.sqrt(b * h) * np.random.randn(N)
    return x

# ---------- 解析真后验 ----------
_, alpha, mpost, vpost = exact_post_score(np.zeros(1), 0.0)
E_exact = float(np.sum(alpha * mpost))
V_exact = float(np.sum(alpha * (vpost + mpost**2)) - E_exact**2)
print(f"\n[解析] 后验分量权重 alpha = {np.round(alpha,4)}  (对应分量均值 {mpost.round(4)})")
print(f"[解析] 真后验均值={E_exact:.4f}, 方差={V_exact:.4f}")

# 解析 E|x-y|, E|x-x0*|（网格数值积分）
xg = np.linspace(-8, 8, 200001)
dens = sum(a * np.exp(-0.5 * ((xg - m)**2) / v) / np.sqrt(2 * np.pi * v)
           for a, m, v in zip(alpha, mpost, vpost))
dens /= np.trapezoid(dens, xg)
Exy = np.trapezoid(np.abs(xg - y_obs) * dens, xg)
Exx0 = np.trapezoid(np.abs(xg - x0_star) * dens, xg)
print(f"[解析] 真后验 E|x-y|={Exy:.4f}, E|x-x0*|={Exx0:.4f}")

# 先验矩
E_prior = sum(w * m for w, m in zip(W, M))
V_prior = sum(w * (m**2 + s**2) for w, m, s in zip(W, M, S)) - E_prior**2
print(f"[解析] 先验均值={E_prior:.4f}, 先验方差={V_prior:.4f}")

# ---------- 采样对照 ----------
res = {}
x_unc = run_reverse(lambda x, t: prior_score(x, t))
res['DPS(zeta=1,mean_t因子) [脚本原样]'] = run_reverse(lambda x, t: dps_score(x, t, 1.0))
res['精确后验得分(金标准)'] = run_reverse(lambda x, t: exact_post_score(x, t)[0])
res['DPS变体: 无mean_t因子'] = run_reverse(lambda x, t: dps_score(x, t, 1.0, scale='none'))
res['DPS: zeta=0.25'] = run_reverse(lambda x, t: dps_score(x, t, 0.25))
res['DPS: zeta=0.5'] = run_reverse(lambda x, t: dps_score(x, t, 0.5))

print(f"\n{'方法':<34}{'均值':>9}{'方差':>9}{'E|x-y|':>9}{'E|x-x0*|':>10}{'左模式占比':>10}")
def left_frac(xs): return float(np.mean(xs < -0.6))
rows = [('无条件(先验)', x_unc, E_prior, V_prior)]
for k, v in res.items():
    tb = exact_post_score(v, 0.0)[0] if False else None
    rows.append((k, v, E_exact, V_exact))
for name, xs, _, _ in rows:
    print(f"{name:<34}{np.mean(xs):>9.4f}{np.var(xs):>9.4f}"
          f"{np.mean(np.abs(xs - y_obs)):>9.4f}{np.mean(np.abs(xs - x0_star)):>10.4f}{left_frac(xs):>10.4f}")
left_mask = xg < -0.6
left_mass = sum(a * np.trapezoid(np.exp(-0.5 * ((xg[left_mask] - m)**2) / v) / np.sqrt(2 * np.pi * v), xg[left_mask])
                for a, m, v in zip(alpha, mpost, vpost))
print(f"{'[解析真后验]':<34}{E_exact:>9.4f}{V_exact:>9.4f}{Exy:>9.4f}{Exx0:>10.4f}{float(left_mass):>10.4f}")

# ---------- 图中数值 vs 重跑JSON ----------
jp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_summary.json')
if os.path.exists(jp):
    print("\n[JSON] 重跑生成:", json.dumps(json.load(open(jp, encoding='utf-8')), ensure_ascii=False))

# ---------- 新旧图对比 ----------
try:
    from PIL import Image
    old = np.asarray(Image.open(r"d:\works\deepinv\第13章 条件生成与逆问题求解\第十三章 配套实验\实验13.1-1\1D逆问题原理演示.png").convert('RGB')).astype(float)
    new = np.asarray(Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '1D逆问题原理演示.png')).convert('RGB')).astype(float)
    if old.shape == new.shape:
        d = np.abs(old - new).mean()
        print(f"\n[图对比] 新旧PNG平均像素差={d:.3f} (0=完全一致), 最大差={np.abs(old-new).max():.0f}")
    else:
        print(f"\n[图对比] 尺寸不同 old={old.shape} new={new.shape}")
except Exception as e:
    print("[图对比] 失败:", e)
