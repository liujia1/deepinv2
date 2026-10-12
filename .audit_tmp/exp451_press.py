import numpy as np, torch, time
from scipy.stats import geninvgauss
torch.set_num_threads(1)

def gig_mh(chi, psi, rng, steps=120, init_logz=None):
    chi = np.maximum(chi, 1e-30)
    logz = init_logz if init_logz is not None else np.log(np.maximum(chi/psi,1e-12))
    step=0.5
    def lpz(lz): return 0.5*lz - 0.5*(chi*np.exp(-lz) + psi*np.exp(lz))
    cur=lpz(logz)
    for _ in range(steps):
        prop=logz+rng.normal(0,step,size=logz.shape); pc=lpz(prop)
        a=np.log(rng.random(size=logz.shape))<(pc-cur)
        logz=np.where(a,prop,logz); cur=np.where(a,pc,cur)
    return np.exp(logz)

def CG(x, Ax, sigma, lamb, z, b, maxit=200):
    r=b-Ax(sigma,lamb,z,x); p=r; rsold=torch.sum(r**2)
    for it in range(maxit):
        Ap=Ax(sigma,lamb,z,p); alpha=rsold/torch.sum(p*Ap)
        x=x+alpha*p; r=r-alpha*Ap; rsnew=torch.sum(r**2); beta=rsnew/rsold
        rsold=rsnew.clone(); p=r+beta*p
        if torch.mean(r**2)<1e-8: break
    return x
def Du(u):
    M,N=u.shape; o=torch.zeros((2,M,N),dtype=u.dtype)
    o[0,:, :-1]+=u[:,1:]-u[:,:-1]; o[1,:-1,:]+=u[1:,:]-u[:-1,:]; return o
def DTp(p):
    C,M,N=p.shape; o=torch.zeros((M,N),dtype=p.dtype)
    o[:,1:]+=p[0,:,:-1]; o[:,:-1]-=p[0,:,:-1]; o[1:,:]+=p[1,:-1,:]; o[:-1,:]-=p[1,:-1,:]; return o

N=100; xg=np.linspace(-1,1,N); yg=np.linspace(-1,1,N); X,Y=np.meshgrid(xg,yg)
u1=0*X+0.5; u2=X+Y; u2-=u2.min(); u2/=u2.max()
u3=np.clip(1-((1-X)**2+(1-Y)**2),a_min=0,a_max=None); u3-=u3.min(); u3/=u3.max()
m1=0*X; m1[N//8:-N//8,N//8:-N//8]=1; u=u1.copy(); u[m1==1]=u2[m1==1]*1.25
m2=0*X; m2[(X-1)**2+(Y-1)**2<=1]=1; u[m2==1]=u3[m2==1]
sigma=0.1; lamb=10.0
def psnr(x,ref): return 10*np.log10(1.0/np.mean((x-ref)**2))
gx=np.abs(np.diff(u,axis=1)); gy=np.abs(np.diff(u,axis=0))
edge=np.zeros((N,N)); edge[:,:-1]+=gx; edge[:-1,:]+=gy; edge/=edge.max()

np.random.seed(42); torch.manual_seed(42)
f=u+np.random.randn(N,N)*sigma; f_=torch.from_numpy(f).float()
Au_hq=lambda s,l,z,uu: lamb*DTp(Du(uu)/z)+uu/s**2
uhq=f_.clone()
for it in range(50):
    z_=torch.from_numpy(np.clip(torch.abs(Du(uhq)).numpy(),1e-6,None)).float()
    uhq=CG(uhq,Au_hq,sigma,lamb,z_,f_/sigma**2)
print(f"PSNR_MAP={psnr(uhq.numpy(),u):.2f}", flush=True)

rng=np.random.default_rng(123)
Au_glm=lambda s,l,z,uu: DTp(Du(uu)/z)+uu/s**2
MAXIT=800; NB=100
ug=f_.clone(); us=torch.zeros_like(ug); usq=torch.zeros_like(ug)
prev_logz=None
t0=time.time()
for it in range(1,MAXIT+1):
    d=Du(ug); diff=torch.abs(d).numpy().ravel(); diff=np.maximum(diff,1e-30)
    z_=torch.from_numpy(gig_mh(diff**2, lamb**2, rng, steps=80, init_logz=prev_logz)).float().reshape(2,N,N)
    prev_logz=np.log(z_.numpy().ravel())
    eta=torch.normal(0,1,size=(3,N,N))
    rhs=DTp(eta[:2]/torch.sqrt(z_))+eta[2]/sigma+f_/sigma**2
    ug=CG(ug,Au_glm,sigma,lamb,z_,rhs)
    if it>=NB: us+=ug; usq+=ug**2
    if it in (NB+30,200,300,500,800):
        ns=it-NB; uavg=us/ns; uvar=usq/ns-uavg**2
        std=np.sqrt(torch.clamp(uvar,min=0)).numpy()
        corr=np.corrcoef(std.ravel(),edge.ravel())[0,1]
        print(f"n_samples={ns:4d} | PSNR_MMSE={psnr(uavg.numpy(),u):6.2f} | corr(std,edge)={corr:.3f} | std_mean={std.mean():.4f} | {time.time()-t0:.0f}s", flush=True)
