# -*- coding: utf-8 -*-
"""实验3.4-2 复核诊断：
1) F* 参考值收敛性（20k vs 100k vs 300k）+ KKT 最优性验证
2) 斜率拟合窗口敏感性（ISTA/FISTA 多窗口 + 逐段局部斜率）
3) 容差鲁棒性（0.5%~5% 各容差下的迭代数与加速比）
4) 活跃集（非零支撑）随迭代的变化，检验"k>500 后活跃集固定"免责
复现要求：与 3.4-2.py 完全相同的种子与构造。
"""
import sys, io, math
import numpy as np
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

np.random.seed(42)
n, sigma_blur, sigma_noise, lam = 256, 2.0, 0.05, 0.1

def make_piecewise_constant(n, seed=17):
    rng = np.random.RandomState(seed)
    x = np.zeros(n); i = 0
    while i < n:
        seg = rng.randint(15, 40)
        amp = rng.uniform(0.3, 1.0) * rng.choice([-1.0, 1.0])
        x[i:i+seg] = amp; i += seg
    return x

x_true = make_piecewise_constant(n, seed=17)
t = np.fft.fftfreq(n) * n
h = np.exp(-t**2 / (2*sigma_blur**2)); h = h / h.sum()
H_fft = np.fft.fft(h)
forward = lambda x: np.real(np.fft.ifft(H_fft * np.fft.fft(x)))
adjoint = lambda z: np.real(np.fft.ifft(np.conj(H_fft) * np.fft.fft(z)))
y = forward(x_true) + sigma_noise * np.random.randn(n)
L_exact = float(np.max(np.abs(H_fft))**2)
gamma = 0.99 / L_exact

fidelity = lambda x: 0.5*float(np.sum((forward(x)-y)**2))
grad_f = lambda x: adjoint(forward(x)-y)
soft = lambda v, thr: np.sign(v)*np.maximum(np.abs(v)-thr, 0.0)
cost = lambda x: fidelity(x) + lam*float(np.sum(np.abs(x)))

def run_ista(mi):
    xk = np.zeros(n); c = np.zeros(mi)
    for k in range(mi):
        xk = soft(xk - gamma*grad_f(xk), gamma*lam); c[k] = cost(xk)
    return xk, c

def run_fista(mi, ret_support=False):
    xk = np.zeros(n); xo = np.zeros(n); to = 1.0
    c = np.zeros(mi); sup = np.zeros(mi, dtype=int)
    for k in range(mi):
        tk = 0.5*(1.0+math.sqrt(1.0+4.0*to**2))
        yk = xk + (to-1.0)/tk*(xk-xo)
        xn = soft(yk - gamma*grad_f(yk), gamma*lam)
        c[k] = cost(xn)
        if ret_support: sup[k] = int(np.sum(np.abs(xn) > 1e-9))
        to, xo, xk = tk, xk, xn
    return xk, c, sup

x_ista, c_ista = run_ista(2000)
_, c_fista, sup_fista = run_fista(2000, ret_support=True)

# ---------- 1) F* 收敛性与 KKT ----------
print("="*60)
print("[1] 参考最优值 F* 的收敛性")
for R in (20000, 100000, 300000):
    xr, cr, _ = run_fista(R)
    print(f"  FISTA {R:>6} 次: F_ref = {cr[-1]:.9f}  (tail slope k={R//2}~{R}: "
          f"{np.polyfit(np.log(np.arange(R//2+1,R+1)), np.log(np.maximum(cr[R//2:]-0,1e-300)+1e-300),1)[0]:.2f})")
x_ref, c_ref, _ = run_fista(20000)
F_star = float(c_ref[-1])
# KKT：对最优解 x*，grad f(x*) 分量满足 |gi|<=λ (xi=0)；gi = -λ·sign(xi) (xi≠0)
g = grad_f(x_ref)
nz = np.abs(x_ref) > 1e-9
v_eq = float(np.max(np.abs(g[nz] + lam*np.sign(x_ref[nz])))) if nz.any() else 0.0
v_ineq = float(np.max(np.abs(g[~nz])) - lam) if (~nz).any() else -1.0
print(f"  KKT 验证 @FISTA20k 解: 非零分量等式最大违反 {v_eq:.2e} | 零分量 |g_i|<=λ 最大违反 {max(v_ineq,0):.2e}")
print(f"  非零分量数: {int(nz.sum())}/{n}")

# ---------- 2) 斜率窗口敏感性 ----------
print("="*60)
print("[2] 斜率拟合窗口敏感性（gap = F(x_k)-F*）")
gap_i = np.maximum(c_ista - F_star, 1e-15)
gap_f = np.maximum(c_fista - F_star, 1e-15)
def slope(gap, a, b):
    ka = np.arange(a, b+1)
    return float(np.polyfit(np.log(ka), np.log(gap[a-1:b]), 1)[0])
wins = [(50,300),(100,500),(300,800),(500,2000),(1000,2000)]
for (a,b) in wins:
    print(f"  窗口 k={a:>4}~{b:<5} ISTA {slope(gap_i,a,b):+.3f}   FISTA {slope(gap_f,a,b):+.3f}")
# 局部斜率（log-log 差分，分段中点报告）
print("  FISTA 逐段局部斜率 dlog(gap)/dlog(k):")
kk = np.arange(1, 2001)
loc = np.gradient(np.log(gap_f), np.log(kk))
for a in (20, 50, 100, 200, 300, 500, 800, 1200, 1800):
    print(f"    k≈{a:>4}: {loc[a-1]:+.2f}")
print("  ISTA 逐段局部斜率:")
loci = np.gradient(np.log(gap_i), np.log(kk))
for a in (50, 200, 500, 1000, 1500, 1900):
    print(f"    k≈{a:>4}: {loci[a-1]:+.2f}")

# ---------- 3) 容差鲁棒性 ----------
print("="*60)
print("[3] 容差鲁棒性（免责：更紧容差只会放大差距，不改方向）")
gap0 = float(c_ista[0]-F_star)
def first_reach(gap, tol):
    hit = np.nonzero(gap <= tol)[0]
    return int(hit[0])+1 if len(hit) else None
for rel in (5e-3, 1e-2, 2e-2, 5e-2):
    t_ = rel*gap0
    ki, kf = first_reach(gap_i, t_), first_reach(gap_f, t_)
    if ki and kf: print(f"  tol={rel:.0e}: ISTA {ki}  FISTA {kf}  加速 {ki/kf:.2f}x")
    else: print(f"  tol={rel:.0e}: ISTA {ki}  FISTA {kf}（None=2000次内未达）")

# ---------- 4) 活跃集稳定性 ----------
print("="*60)
print("[4] FISTA 迭代点非零分量数（活跃集）")
for k in (50, 100, 200, 300, 500, 800, 1200, 1999):
    print(f"  k={k:>4}: nnz={sup_fista[k-1]}")
sup_ref = int(np.sum(np.abs(x_ref) > 1e-9))
print(f"  参考解 nnz={sup_ref}")
# 与真信号误差（检验讲义为何不用）
print("="*60)
print("[5] 附：与真信号相对误差（讲义声明不用于收敛度量，仅核对其说法）")
print(f"  ISTA2000 vs x_true: {np.linalg.norm(x_ista-x_true)/np.linalg.norm(x_true):.3f}")
print(f"  参考解  vs x_true: {np.linalg.norm(x_ref-x_true)/np.linalg.norm(x_true):.3f}")
