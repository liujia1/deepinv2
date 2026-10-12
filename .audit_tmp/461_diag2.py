# -*- coding: utf-8 -*-
"""实验4.6-1 深度诊断：ILA采样正确性 & MMSE最优性"""
import numpy as np, torch, builtins
_out = open(r'd:\works\deepinv\.audit_tmp\461_diag2_out.txt', 'w', encoding='utf-8')
_rp = builtins.print
def print(*a, **k): _rp(*a, file=_out, **k); _out.flush()

def Du(u):
    M,N=u.shape; d=np.zeros((2,M,N))
    d[0,:,:-1]=u[:,1:]-u[:,:-1]; d[1,:-1,:]=u[1:,:]-u[:-1,:]; return d
def DTp(p):
    C,M,N=p.shape; o=np.zeros((M,N))
    o[:,1:]+=p[0,:,:-1]; o[:,:-1]-=p[0,:,:-1]; o[1:,:]+=p[1,:-1,:]; o[:-1,:]-=p[1,:-1,:]; return o

N=100; x=np.linspace(-1,1,N); y=np.linspace(-1,1,N); X,Y=np.meshgrid(x,y)
u1=0*X+0.5; u2=X+Y; u2-=u2.min(); u2/=u2.max()
u3=np.clip(1-((1-X)**2+(1-Y)**2),0,None); u3-=u3.min(); u3/=u3.max()
m1=0*X; m1[N//8:-N//8,N//8:-N//8]=1; u=u1.copy(); u[m1==1]=u2[m1==1]*1.25
m2=0*X; m2[(X-1)**2+(Y-1)**2<=1]=1; u[m2==1]=u3[m2==1]
sigma=0.1; np.random.seed(42); f=u+np.random.randn(N,N)*sigma
lamb,eps,theta=10.0,1e-3,0.99
L=lamb*8/eps+1/sigma**2; tau=2/L

def grad_E(v):
    d=Du(v); return lamb*DTp(d/np.clip(np.abs(d),eps,None))+(v-f)/sigma**2
def psnr(a,b): mse=np.mean((a-b)**2); return 10*np.log10(1/mse) if mse>0 else float('inf')

# MAP
u_map=f.copy(); u_old=f.copy()
for it in range(10000):
    g=grad_E(u_map); ui=u_map+theta*(u_map-u_old); u_old=u_map.copy(); u_map=ui-tau*g

# ---- A. 稳定域分析：γλ 分布 ----
# 在 MAP 处线性化 Hessian 的特征值范围（用幂迭代估计极值）
import scipy.sparse as sp, scipy.sparse.linalg as spla
d_map=Du(u_map); w=lamb/np.clip(np.abs(d_map),eps,None)
D1=sp.diags([-np.ones(N-1),np.ones(N-1)],[0,1],shape=(N-1,N),format='csr'); IN=sp.eye(N,format='csr')
Bc=sp.kron(IN,D1,format='csr'); Wc=sp.diags(w[0,:,:-1].ravel(order='C'),0,format='csr'); H1=Bc.T@Wc@Bc
Br=sp.kron(D1,IN,format='csr'); Wr=sp.diags(w[1,:-1,:].ravel(order='C'),0,format='csr'); H2=Br.T@Wr@Br
H=(H1+H2+sp.eye(N*N,format='csr')/sigma**2).tocsr()
lmax=spla.eigsh(H,k=1,which='LM',return_eigenvectors=False)[0]
lmin=spla.eigsh(H,k=1,which='SA',return_eigenvectors=False)[0]
print(f"[A] Hessian特征值: lmin={lmin:.1f} lmax={lmax:.1f}")
print(f"[A] gamma*lambda: min={tau*lmin:.4f} max={tau*lmax:.4f}  (2beta={2*theta:.3f})")
print(f"[A] ILA/AR2稳定Jury条件要求 2beta<gamma_l<2 即 1.98<gamma_l<2.0")
print(f"[A] 绝大多数方向 gamma_l={tau*lmin:.4f}<<1.98 → 违反下界(动量过强,欠阻尼)；gamma_lmax={tau*lmax:.4f}{'>2 违反上界(发散)' if tau*lmax>2 else '<2 ok'}")

# ---- B. 采样方差 vs 理论 ----
def ila(maxit,burn,seed,noise_scale=1.0):
    gt=torch.Generator().manual_seed(seed)
    ui_=f.copy(); uo=f.copy(); s1=np.zeros((N,N)); s2=np.zeros((N,N)); stds=[]
    amp=np.sqrt(2*tau*(1-theta))*noise_scale
    for it in range(maxit):
        g=grad_E(ui_); uu=ui_+theta*(ui_-uo); uo=ui_.copy()
        ui_=uu-tau*g+torch.randn(N,N,generator=gt).numpy()*amp
        if it>=burn: s1+=ui_; s2+=ui_**2
        if it in (2000,5000,10000,20000,40000) and it>=burn:
            a=s1/(it-burn+1); v=s2/(it-burn+1)-a**2; stds.append((it,np.sqrt(np.clip(v,0,None)).mean()))
    a=s1/(maxit-burn); v=s2/(maxit-burn)-a**2
    return a, np.sqrt(np.clip(v,0,None)), stds

avg, std_map, stds = ila(40000, 1000, 42)
print(f"[B] ILA采样 std图均值={std_map.mean():.4f}")
print(f"[B] 单像素方差随迭代演化(it, 平均std): {[(int(i),round(float(s),4)) for i,s in stds]}")
print(f"[B] 若std随迭代持续增=不稳定(未达平稳); 若趋于常数=平稳但温度错误")

# 理论后验方差(高斯近似): 特征值方向1/λ, 空间各像素=H^-1对角
lu=spla.splu(H.tocsc())
Z=np.column_stack([lu.solve(np.eye(N*N)[:,j]) for j in range(N*N)])
# 正确: 边际后验方差 = (H^-1)_{jj} = Z[j,j] (H对称)
post_var=np.diag(Z).copy()
post_var=np.clip(post_var,1e-12,None)
post_std=np.sqrt(post_var).reshape(N,N)
print(f"[B] 理论(H^-1对角)后验std均值={post_std.mean():.4f}  采样/理论={std_map.mean()/post_std.mean():.2f}x")
print(f"[B] 采样std与理论std空间相关={np.corrcoef(std_map.ravel(),post_std.ravel())[0,1]:.3f}")

# ---- C. MMSE最优性检验 ----
# 正确后验均值应最小化对真值u的MSE. 用高斯近似的后验均值:
g_map=grad_E(u_map); b=g_map.ravel()-H@u_map.ravel()
gm=(u_map.ravel()+lu.solve(lu.solve(b))).reshape(N,N)
print(f"[C] psnr MAP={psnr(u_map,u):.2f}  psnr_高斯近似后验均值={psnr(gm,u):.2f}  psnr_ILA采样均值={psnr(avg,u):.2f}")
print(f"[C] 理论MMSE应≥MAP(均值是最小MSE估计). 高斯均值{psnr(gm,u):.2f} vs MAP{psnr(u_map,u):.2f}")

# ---- D. 正确温度采样: 过阻尼ULA √(2τ)无动量, 小步 ----
def ula(maxit,burn,tau_):
    gt=torch.Generator().manual_seed(42)
    ui_=f.copy(); s1=np.zeros((N,N)); s2=np.zeros((N,N))
    amp=np.sqrt(2*tau_)
    for it in range(maxit):
        g=grad_E(ui_); ui_=ui_-tau_*g+torch.randn(N,N,generator=gt).numpy()*amp
        if it>=burn: s1+=ui_; s2+=ui_**2
    a=s1/(maxit-burn); v=s2/(maxit-burn)-a**2
    return a,np.sqrt(np.clip(v,0,None))
ua,us=ula(60000,10000,tau)
print(f"[D] 过阻尼ULA(√2τ,无动量) std均值={us.mean():.4f} (理论{post_std.mean():.4f}) psnr均值={psnr(ua,u):.2f} 空间相关={np.corrcoef(us.ravel(),post_std.ravel())[0,1]:.3f}")

# ---- E. ILA若β=0(退化ULA)应匹配理论 ----
a0,s0,_=ila(20000,2000,42,noise_scale=1.0); # theta仍0.99,需单独β=0
gt=torch.Generator().manual_seed(42); ui_=f.copy(); uo=f.copy(); s1=np.zeros((N,N)); s2=np.zeros((N,N))
beta=0.0; amp=np.sqrt(2*tau*(1-beta))
for it in range(20000):
    g=grad_E(ui_); uu=ui_+beta*(ui_-uo); uo=ui_.copy(); ui_=uu-tau*g+torch.randn(N,N,generator=gt).numpy()*amp
    if it>=2000: s1+=ui_; s2+=ui_**2
a1=s1/18000; v1=s2/18000-a1**2; std1=np.sqrt(np.clip(v1,0,None))
print(f"[E] ILA β=0(=ULA) std均值={std1.mean():.4f} (理论{post_std.mean():.4f}) psnr={psnr(a1,u):.2f} 空间相关={np.corrcoef(std1.ravel(),post_std.ravel())[0,1]:.3f}")
_out.close()
print_=None
