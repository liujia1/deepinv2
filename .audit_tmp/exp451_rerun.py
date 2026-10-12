import numpy as np, torch, time
import matplotlib; matplotlib.use('Agg')
from scipy.stats import geninvgauss

def CG(x, Ax, sigma, lamb, z, b, maxit=1000):
    r = b - Ax(sigma, lamb, z, x); p = r; rsold = torch.sum(r**2)
    for it in range(maxit):
        Ap = Ax(sigma, lamb, z, p); alpha = rsold/torch.sum(p*Ap)
        x = x+alpha*p; r = r-alpha*Ap; rsnew = torch.sum(r**2); beta = rsnew/rsold
        rsold = rsnew.clone(); p = r+beta*p
        if torch.mean(r**2) < 1e-9: break
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
sigma=0.1
def psnr(x,ref): return 10*np.log10(1.0/np.mean((x-ref)**2))

def run(MIT, NB, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    f = u + np.random.randn(N,N)*sigma
    f_ = torch.from_numpy(f).float()
    lamb=10.0
    Au_hq=lambda s,l,z,uu: lamb*DTp(Du(uu)/z)+uu/s**2
    uhq=f_.clone()
    for it in range(50):
        z_=np.clip(torch.abs(Du(uhq)).numpy(),1e-6,None); z_=torch.from_numpy(z_).float()
        uhq=CG(uhq,Au_hq,sigma,lamb,z_,f_/sigma**2)
    u_hq=uhq
    Au_glm=lambda s,l,z,uu: DTp(Du(uu)/z)+uu/s**2
    ug=f_.clone(); us=0; usq=0
    for it in range(MIT):
        diff=torch.abs(Du(ug)).numpy().ravel(); diff[diff==0]=1e-30
        z_=torch.from_numpy(geninvgauss.rvs(0.5,lamb*diff)*diff/lamb).float().reshape(2,N,N)
        eta=torch.normal(0,1,size=(3,N,N))
        rhs=DTp(eta[:2]/torch.sqrt(z_))+eta[2]/sigma+f_/sigma**2
        ug=CG(ug,Au_glm,sigma,lamb,z_,rhs)
        if it>=NB: us+=ug; usq+=ug**2
    ns=MIT-NB; uavg=us/ns; uvar=usq/ns-uavg**2
    std=np.sqrt(np.clip(uvar.numpy(),0,None))
    gx=np.abs(np.diff(u,axis=1)); gy=np.abs(np.diff(u,axis=0))
    edge=np.zeros((N,N)); edge[:,:-1]+=gx; edge[:-1,:]+=gy; edge/=edge.max()
    corr=np.corrcoef(std.ravel(),edge.ravel())[0,1]
    return psnr(u_hq.numpy(),u), psnr(uavg.numpy(),u), corr, std, uavg.numpy()

print(f"{'iters(burn)':>14} | {'PSNR_MAP':>8} | {'PSNR_MMSE':>9} | {'corr(std,edge)':>14} | {'std_mean':>8}")
res={}
for MIT,NB in [(50,20),(100,30),(200,50),(500,100),(1000,200)]:
    t0=time.time(); pm,pa,corr,std,ua=run(MIT,NB,42); res[(MIT,NB)]=(pm,pa,corr)
    print(f"{MIT:>6}(bi{NB:>3})   | {pm:8.2f} | {pa:9.2f} | {corr:14.3f} | {std.mean():8.4f}  [{time.time()-t0:.0f}s]")

c=torch.load(r"d:\works\deepinv\第4章 MCMC与ULA算法\第四章 配套实验\实验4.5-1\experiment_4.5-1_cache.pth",map_location='cpu',weights_only=False)
print("\nCached  PSNRs:", f"MAP={psnr(c['u_hq'].numpy(),u):.2f} MMSE={psnr(c['u_avg_'].numpy(),u):.2f}")
pm,pa,corr,std,ua=run(50,20,42)
print("Fresh50 PSNRs:", f"MAP={pm:.2f} MMSE={pa:.2f}")
print("max|fresh-MMSE - cached-MMSE| =", f"{np.abs(ua-c['u_avg_'].numpy()).max():.3e}")
