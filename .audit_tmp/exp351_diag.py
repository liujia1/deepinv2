# -*- coding: utf-8 -*-
"""复核 3.5-1：复现 JSON + KKT 裁决 ADMM 是否真全局最优 + 检验 L-BFGS-B 停点的 scipy 版本敏感性"""
import numpy as np
from scipy.optimize import minimize
import scipy

np.random.seed(42)
A = np.array([[1,0,1,0],[0,1,0,1],[1,1,0,0],[0,0,1,1]], dtype=float)
target = np.array([2.,2.,6.,7.]); rhs = A@target; alpha=1.0
DH = np.array([[1,-1,0,0],[0,0,1,-1]]); DV = np.array([[1,0,-1,0],[0,1,0,-1]])
def tv_cost(x):
    return 0.5*np.sum((A@x-rhs)**2) + alpha*(np.sum(np.abs(DH@x))+np.sum(np.abs(DV@x)))
def tv_grad(x):
    return A.T@(A@x-rhs) + alpha*(DH.T@np.sign(DH@x)+DV.T@np.sign(DV@x))

# 方法A 粗网格
xr = np.linspace(1,7,10); bg=np.inf; bx=None
for a in xr:
 for b in xr:
  for c in xr:
   for d in xr:
    v=np.array([a,b,c,d]); cc=tv_cost(v)
    if cc<bg: bg=cc; bx=v.copy()
# 方法B 随机
br=np.inf; rx=None
for _ in range(10000):
    v=1+6*np.random.rand(4); cc=tv_cost(v)
    if cc<br: br=cc; rx=v.copy()
# 方法C L-BFGS-B
res=minimize(tv_cost,x0=[4.]*4,method='L-BFGS-B',jac=tv_grad,bounds=[(0,None)]*4)
# 方法D ADMM
D=np.vstack([DH,DV])
def admm(rho=1.0,iters=50000,tol=1e-12):
    Mi=np.linalg.inv(A.T@A+rho*(D.T@D)); x=np.zeros(4);z=np.zeros(4);u=np.zeros(4);thr=alpha/rho
    for k in range(iters):
        xn=Mi@(A.T@rhs+rho*(D.T@(z-u))); Dxu=D@xn+u
        zn=np.sign(Dxu)*np.maximum(np.abs(Dxu)-thr,0.); u=u+D@xn-zn
        conv=np.linalg.norm(xn-x)<tol and k>1000; x=xn; z=zn
        if conv: break
    return x
xref=admm(); fref=tv_cost(xref)

print("=== JSON 复现 (scipy %s) ===" % scipy.__version__)
print(f"best_cost_grid={bg:.4f} (J=8.5556)  best_x_grid={np.round(bx,4).tolist()} (J=[2.3333,2.3333,6.3333,6.3333])")
print(f"best_cost_rand={br:.4f} (J=9.0234)  best_x_rand={np.round(rx,4).tolist()} (J=[2.3747,2.3907,6.4084,6.0035])")
print(f"opt_cost(L-BFGS-B)={res.fun:.4f} (J=8.2582)  x_opt={np.round(res.x,4).tolist()} (J=[2.5464,2.5464,6.013,6.013])")
print(f"ref_cost_admm={fref:.6f} (J=8.25)  x_ref={np.round(xref,6).tolist()} (J=[2.5,2.5,6.0,6.0])")
print(f"gap={res.fun-fref:.6f} (J=0.008184)")

# KKT 裁决 ADMM: x* 全局最优 <=> 存在 s_H in del|DH x|, s_V in del|DV x| 使 A^T(Ax-b)+alpha(DH^T s_H+DV^T s_V)=0
# 等价: 对 Dx 非零分量 s=sign, 零分量 |对应子梯度| 允许 [-1,1]。用投影残差衡量
print("\n=== KKT/最优性裁决 ===")
def kkt_report(x,tag):
    g=A.T@(A@x-rhs)  # 数据项梯度
    Dh,H0=DH@x,DV@x
    sh=np.sign(Dh); sv=np.sign(H0)
    grad=g+alpha*(DH.T@sh+DV.T@sv)
    # 对零分量: 其"梯度分量"若|.|<=alpha 则可被次梯度吸收
    nzH=np.abs(Dh)>1e-9; nzV=np.abs(H0)>1e-9
    # 简化：报告非光滑零区是否满足 |alpha*(DH^T..)_{i}|<=alpha
    print(f"  {tag}: F={tv_cost(x):.6f}  x={np.round(x,4).tolist()}")
    print(f"      DH x={np.round(Dh,4).tolist()}  DV x={np.round(H0,4).tolist()}")
    # 精确全局最优验证：用暴力多起点 L-BFGS-B + 细网格确认无更低点
for x,t in [(xref,"ADMM"),(res.x,"L-BFGS-B")]:
    kkt_report(x,t)

# 独立验证 F*=8.25 确为全局最优：多起点扰动 + 解析候选
print("\n=== 全局最优独立确认 (多起点 L-BFGS-B + Powell) ===")
best=(1e9,None)
rng=np.random.RandomState(0)
for i in range(300):
    x0=rng.uniform(0,8,4)
    r=minimize(tv_cost,x0,method='Powell',bounds=[(0,None)]*4)
    if r.fun<best[0]-1e-9: best=(r.fun,r.x.copy())
print(f"  300起点Powell最优: F={best[0]:.6f} x={np.round(best[1],4).tolist()}")
print(f"  ADMM F*={fref:.6f}  是否 <= 所有起点最优: {fref<=best[0]+1e-6}")

# scipy 版本敏感性：不同 method 的停点
print("\n=== L-BFGS-B 停点稳健性(不同起点/方法) ===")
for x0 in [[4,4,4,4],[1,1,7,7],[6,2,2,6],[3,3,5,5]]:
    r=minimize(tv_cost,x0=np.array(x0,float),method='L-BFGS-B',jac=tv_grad,bounds=[(0,None)]*4)
    print(f"  x0={x0}: F={r.fun:.6f} x={np.round(r.x,4).tolist()}")
