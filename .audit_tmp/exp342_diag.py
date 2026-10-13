# -*- coding: utf-8 -*-
"""复核 3.4-2：不写图，仅复现 JSON 数值 + 独立验证 LASSO 最优性与算法正确性"""
import math, sys
import numpy as np

# ---- 复刻 3.4-2 的正向模型与算法（逐行对齐原脚本）----
np.random.seed(42)
n, sigma_blur, sigma_noise, lam = 256, 2.0, 0.05, 0.1

def make_piecewise_constant(n, seed=17):
    rng = np.random.RandomState(seed); x = np.zeros(n); i = 0
    while i < n:
        seg = rng.randint(15, 40); amp = rng.uniform(0.3, 1.0) * rng.choice([-1.0, 1.0])
        x[i:i+seg] = amp; i += seg
    return x

x_true = make_piecewise_constant(n, seed=17)

def gaussian_kernel_fft(n, sigma):
    t = np.fft.fftfreq(n) * n; h = np.exp(-t**2/(2*sigma**2)); h = h/h.sum()
    return np.fft.fft(h)

H_fft = gaussian_kernel_fft(n, sigma_blur)
forward = lambda x: np.real(np.fft.ifft(H_fft*np.fft.fft(x)))
adjoint = lambda z: np.real(np.fft.ifft(np.conj(H_fft)*np.fft.fft(z)))
y = forward(x_true) + sigma_noise*np.random.randn(n)

L_exact = float(np.max(np.abs(H_fft))**2)
np.random.seed(0)
v = np.random.randn(n); v /= np.linalg.norm(v)
for _ in range(50):
    w = adjoint(forward(v)); v = w/np.linalg.norm(w)
L_power = float(np.sum(v*adjoint(forward(v))))
gamma = 0.99/L_exact

def grad_f(x): return adjoint(forward(x)-y)
def soft(v,t): return np.sign(v)*np.maximum(np.abs(v)-t,0.0)
def cost(x): return 0.5*float(np.sum((forward(x)-y)**2)) + lam*float(np.sum(np.abs(x)))

def run_ista(mi):
    x = np.zeros(n); c = np.zeros(mi)
    for k in range(mi):
        z = x - gamma*grad_f(x); x = soft(z, gamma*lam); c[k]=cost(x)
    return x,c
def run_fista(mi):
    x=np.zeros(n); xo=np.zeros(n); to=1.0; c=np.zeros(mi)
    for k in range(mi):
        tk=0.5*(1+math.sqrt(1+4*to**2)); yk=x+(to-1)/tk*(x-xo)
        z=yk-gamma*grad_f(yk); xn=soft(z,gamma*lam); c[k]=cost(xn)
        to=tk; xo=x; x=xn
    return x,c

max_iter=2000
x_ista,ci=run_ista(max_iter); x_fista,cf=run_fista(max_iter)
x_ref,cr=run_fista(20000); F_star=float(cr[-1])
gi=np.maximum(ci-F_star,1e-15); gf=np.maximum(cf-F_star,1e-15)
gap0=float(ci[0]-F_star); tol=2e-2*gap0
def first(g,t):
    h=np.nonzero(g<=t)[0]; return int(h[0])+1 if len(h) else None
ki,kf=first(gi,tol),first(gf,tol)
def slope(g,a,b):
    ka=np.arange(a,b+1); return float(np.polyfit(np.log(ka),np.log(g[a-1:b]),1)[0])
si=slope(gi,500,max_iter); sf=slope(gf,50,300)
ei=np.linalg.norm(x_ista-x_ref)/np.linalg.norm(x_ref)
ef=np.linalg.norm(x_fista-x_ref)/np.linalg.norm(x_ref)

print("=== JSON 复现 ===")
print(f"L_exact={L_exact:.6f} (J=1.0)  L_power={L_power:.10f} (J=0.9897594646799062)")
print(f"gamma={gamma:.6f} (J=0.99)  gap0={gap0:.12f}  tol_abs={tol:.12f}")
print(f"k_ista={ki} (J=1230)  k_fista={kf} (J=94)  speedup={ki/kf:.12f} (J=13.085106382978724)")
print(f"slope_ista={si:.12f} (J=-0.9815819935242219)")
print(f"slope_fista={sf:.12f} (J=-2.2357311205549752)")
print(f"err_ista={ei:.12f} (J=0.5054771546605793)  err_fista={ef:.12f} (J=0.01844154258637924)")

# ---- KKT 最优性：x* 为 LASSO 全局最优 <=> A^T(Ax*-y)= -lam*s, s in del||x*||_1 ----
# 即 |grad_i| <= lam 对所有 i；x_i != 0 时 grad_i = -lam*sign(x_i)
print("\n=== KKT 最优性验证 (x_ref 是否 LASSO 全局最优) ===")
for tag,xx in [("x_ref(FISTA2e4)",x_ref),("x_fista(2e3)",x_fista),("x_ista(2e3)",x_ista)]:
    g=grad_f(xx)  # A^T(Ax-y)
    viol_box=np.maximum(np.abs(g)-lam,0.0).max()  # 应<=lam
    nz=np.abs(xx)>1e-8
    # 非零分量: g_i 应 = -lam*sign(x_i)
    st_res=np.abs(g[nz]+lam*np.sign(xx[nz])).max() if nz.any() else 0.0
    print(f"  {tag}: 支撑外|grad|最大={np.abs(g[~nz]).max():.6f}(<=lam={lam})  "
          f"box违反={viol_box:.2e}  非零KKT残差={st_res:.2e}  nnz={int(nz.sum())}")

# ---- 幂迭代是否被 max|H|^2 支配 (L 正确性) ----
print("\n=== L_f 一致性 ===")
# 循环矩阵 A^T A 特征值 = |H|^2, 最大特征值 = max|H|^2
print(f"  max|H|^2={L_exact:.8f}  谱半径(数值A^TA via fft)={np.max(np.abs(H_fft)**2):.8f}")
print(f"  幂迭代50步={L_power:.6f}  (略低于真值属正常, 未完全收敛)")

# ---- sklearn Lasso 交叉验证(若可用) ----
try:
    from sklearn.linear_model import Lasso
    # 构造显式 circulant A
    idx=np.arange(n); A=np.zeros((n,n))
    for j in range(n):
        A[:,j]=forward(np.eye(n)[j] if False else _e(n,j)) if False else 0
    # 用基向量逐列
    E=np.eye(n); A=np.column_stack([forward(E[:,k]) for k in range(n)])
    # sklearn: 1/(2n)||y-Ax||^2 + alpha||x||_1  -> 匹配 0.5||Ax-y||^2+lam||x||_1 需 alpha=lam/n
    las=Lasso(alpha=lam/n, fit_intercept=False, tol=1e-10, max_iter=200000).fit(A,y)
    coef=las.coef_
    print("\n=== sklearn Lasso 交叉验证 ===")
    print(f"  ||x_ref - sklearn||/||sklearn|| = {np.linalg.norm(x_ref-coef)/np.linalg.norm(coef):.6f}")
    print(f"  sklearn nnz={int((np.abs(coef)>1e-6).sum())}  x_ref nnz={int((np.abs(x_ref)>1e-8).sum())}")
    print(f"  sklearn F(x)={cost(coef):.8f}  x_ref F(x)={cost(x_ref):.8f}")
except Exception as ex:
    print("\nsklearn 交叉验证跳过:", type(ex).__name__, ex)
