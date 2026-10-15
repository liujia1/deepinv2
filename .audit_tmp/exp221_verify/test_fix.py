import numpy as np
from skimage import data
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

np.random.seed(42)
n=64; NOISE=0.1; DR=1.0
x_true = resize(data.camera(),(n,n))
y = x_true + NOISE*np.random.randn(n,n)

def sob_exact(y,lam):
    ny,nx=y.shape
    fy=np.fft.fftfreq(ny);fx=np.fft.fftfreq(nx)
    oy,ox=np.meshgrid(fy,fx,indexing='ij')
    le=2*np.cos(2*np.pi*ox)+2*np.cos(2*np.pi*oy)-4
    return np.fft.ifft2(np.fft.fft2(y)/(1-lam*le)).real

# 新判据：基于解的变化 max|Δx|
def sob_iter_new(y,lam,n_iter=200,tol=1e-12):
    x=y.copy(); dt=0.9/(1+8*lam); conv=n_iter
    for k in range(n_iter):
        lap=(np.roll(x,1,0)+np.roll(x,-1,0)+np.roll(x,1,1)+np.roll(x,-1,1)-4*x)
        xn=x+dt*((y-x)+lam*lap)
        dx=np.max(np.abs(xn-x)); x=xn
        if k>0 and dx<tol:
            conv=k+1;break
    return x,conv

print("新判据 tol=1e-12 (解变化):")
for lam in [0.05,0.2,0.5,1.0,2.0]:
    xi,it=sob_iter_new(y,lam); xe=sob_exact(y,lam)
    print(f" lam={lam}: iters={it} maxdiff={np.max(np.abs(xi-xe)):.2e} PSNR={psnr(x_true,xi,data_range=DR):.2f} SSIM={ssim(x_true,xi,data_range=DR):.4f}")

print("\n扩展网格 PSNR/SSIM(闭式解):")
for lam in [0.05,0.2,0.5,1.0,2.0]:
    xe=sob_exact(y,lam)
    print(f" lam={lam}: PSNR={psnr(x_true,xe,data_range=DR):.2f} SSIM={ssim(x_true,xe,data_range=DR):.4f}")

c=n//2
print("\n局部方差(新网格):")
print(" true",round(np.var(x_true[c-5:c+5,c-5:c+5]),4))
lam_d=NOISE**2/x_true.std()**2
print(" tikh",round(np.var(y[c-5:c+5,c-5:c+5]/(1+lam_d)),4))
for lam in [0.05,0.2,0.5,1.0,2.0]:
    xi,_=sob_iter_new(y,lam)
    print(f" sob{lam}",round(np.var(xi[c-5:c+5,c-5:c+5]),4))
