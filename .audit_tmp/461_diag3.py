# -*- coding: utf-8 -*-
"""实验4.6-1 诊断3: 线性高斯ILA精确性 + 配图std空间模式"""
import numpy as np, torch, builtins
import scipy.sparse as sp, scipy.sparse.linalg as spla
_out = open(r'd:\works\deepinv\.audit_tmp\461_diag3_out.txt','w',encoding='utf-8')
_rp=builtins.print
def print(*a,**k): _rp(*a,file=_out,**k); _out.flush()

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

u_map=f.copy(); uo=f.copy()
for it in range(10000):
    g=grad_E(u_map); ui=u_map+theta*(u_map-uo); uo=u_map.copy(); u_map=ui-tau*g
d_map=Du(u_map); w=lamb/np.clip(np.abs(d_map),eps,None)

D1=sp.diags([-np.ones(N-1),np.ones(N-1)],[0,1],shape=(N-1,N),format='csr'); IN=sp.eye(N,format='csr')
Bc=sp.kron(IN,D1,format='csr'); Wc=sp.diags(w[0,:,:-1].ravel(order='C'),0,format='csr'); H1=Bc.T@Wc@Bc
Br=sp.kron(D1,IN,format='csr'); Wr=sp.diags(w[1,:-1,:].ravel(order='C'),0,format='csr'); H2=Br.T@Wr@Br
H=(H1+H2+sp.eye(N*N,format='csr')/sigma**2).tocsr()
lu=spla.splu(H.tocsc())

rep=[30*N+30,70*N+70,50*N+20,20*N+80,5*N+5,95*N+95,50*N+50,40*N+60]
pv=[lu.solve(np.eye(N*N)[j])[j] for j in rep]
print("[1] 线性高斯 代表像素 理论std=sqrt(H^-1对角): " + " ".join(f"{j}:{np.sqrt(max(v,0)):.4f}" for j,v in zip(rep,pv)))

def ila_linear(tau_, theta_, maxit, burn, seed):
    gt=torch.Generator().manual_seed(seed)
    v=np.zeros((N,N)); vo=np.zeros((N,N)); s1=np.zeros((N,N)); s2=np.zeros((N,N))
    amp=np.sqrt(2*tau_*(1-theta_)); Hm=lambda u: np.asarray(H@u.ravel()).reshape(N,N)
    for it in range(maxit):
        g=Hm(v); vv=v+theta_*(v-vo); vo=v.copy()
        v=vv-tau_*g+torch.randn(N,N,generator=gt).numpy()*amp
        if it>=burn: s1+=v; s2+=v**2
    n=maxit-burn; a=s1/n; var=s2/n-a**2
    return a,np.sqrt(np.clip(var,0,None))

for tau_scale in [1.0, 0.5, 0.2]:
    va,vs=ila_linear(tau*tau_scale, theta, 40000, 5000, 42)
    rat=[vs[divmod(j,N)[0],divmod(j,N)[1]]/max(np.sqrt(max(v,0)),1e-9) for j,v in zip(rep,pv)]
    print(f"[2] 线性高斯 tau_x{tau_scale}: 采样/理论std(代表像素) 均值={np.mean(rat):.2f} 明细={[f'{r:.2f}' for r in rat]}")
print("   -> 若tau减小时比值趋1: 偏差是O(gamma)离散固有(讲义已承认); 若恒远大于1: 噪声缩放实现有误")

def ila_nonlin(maxit,burn,seed):
    gt=torch.Generator().manual_seed(seed); ui_=f.copy(); uo=f.copy(); s1=np.zeros((N,N)); s2=np.zeros((N,N))
    amp=np.sqrt(2*tau*(1-theta))
    for it in range(maxit):
        g=grad_E(ui_); uu=ui_+theta*(ui_-uo); uo=ui_.copy(); ui_=uu-tau*g+torch.randn(N,N,generator=gt).numpy()*amp
        if it>=burn: s1+=ui_; s2+=ui_**2
    n=maxit-burn; a=s1/n; vr=s2/n-a**2; return a,np.sqrt(np.clip(vr,0,None))
avg,ss=ila_nonlin(10000,1000,42)
grad_mag=np.sqrt(d_map[0]**2+d_map[1]**2)
flat=grad_mag<0.01; edge=grad_mag>0.05
print(f"[3] 配图std: 平坦区(grad<0.01)均值={ss[flat].mean():.4f} 边缘区(grad>0.05)均值={ss[edge].mean():.4f} 比={ss[edge].mean()/max(ss[flat].mean(),1e-9):.2f}x")
print(f"[3] std图 vs 边缘强度 空间相关={np.corrcoef(ss.ravel(),grad_mag.ravel())[0,1]:.3f}")
rng=np.random.default_rng(0); acc=np.zeros(N*N)
for _ in range(300):
    z=rng.choice([-1.0,1.0],N*N); y=lu.solve(z); acc+=z*y
th=acc/300; th_std=np.sqrt(np.clip(th,0,None)).reshape(N,N)
print(f"[4] 理论Hutchinson std: 平坦={th_std[flat].mean():.4f} 边缘={th_std[edge].mean():.4f} 比={th_std[edge].mean()/max(th_std[flat].mean(),1e-9):.2f}x vs边缘相关={np.corrcoef(th_std.ravel(),grad_mag.ravel())[0,1]:.3f}")
print(f"[4] 采样std vs 理论std 全图空间相关={np.corrcoef(ss.ravel(),th_std.ravel())[0,1]:.3f} 采样/理论均值比={ss.mean()/th_std.mean():.2f}x")
_out.close()
