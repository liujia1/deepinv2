import numpy as np
from skimage import data
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

np.random.seed(42)
n = 64
NOISE = 0.1
DR = 1.0
x_true = resize(data.camera(), (n, n))
y = x_true + NOISE * np.random.randn(n, n)

print("=== 基础量 ===")
print("x_true.std() =", round(x_true.std(), 4), " (md声称0.2755)")
print("x_true.mean() =", round(x_true.mean(), 4))
print("min,max =", round(x_true.min(),4), round(x_true.max(),4))
print("y realization std =", round((y-x_true).std(),4))

sigma_x_data = x_true.std()
lam_data = NOISE**2/sigma_x_data**2
lam_prior = NOISE**2/1.0**2
print("lam_tikh_data =", round(lam_data,4), " (md:0.1317)  lam_prior =", lam_prior)

xt_d = y/(1+lam_data)
xt_p = y/(1+lam_prior)
print("\n=== PSNR/SSIM ===")
print("noisy    ", round(psnr(x_true,y,data_range=DR),2), round(ssim(x_true,y,data_range=DR),4))
print("tikh_data", round(psnr(x_true,xt_d,data_range=DR),2), round(ssim(x_true,xt_d,data_range=DR),4))
print("tikh_pr  ", round(psnr(x_true,xt_p,data_range=DR),2), round(ssim(x_true,xt_p,data_range=DR),4))

def sob_iter(y, lam, n_iter, tol=1e-6, do_break=True):
    x = y.copy()
    dt = 0.9/(1+8*lam)
    prev = np.inf
    conv = n_iter
    for k in range(n_iter):
        lap = (np.roll(x,1,0)+np.roll(x,-1,0)+np.roll(x,1,1)+np.roll(x,-1,1)-4*x)
        x = x + dt*((y-x)+lam*lap)
        gx = np.roll(x,-1,1)-x; gy = np.roll(x,-1,0)-x
        loss = 0.5*np.sum((x-y)**2)+0.5*lam*np.sum(gx**2+gy**2)
        if do_break and k>0 and abs(prev-loss)/(abs(prev)+1e-12) < tol:
            conv = k+1; break
        prev = loss
    return x, conv

def sob_exact(y, lam):
    ny,nx = y.shape
    fy = np.fft.fftfreq(ny); fx = np.fft.fftfreq(nx)
    oy,ox = np.meshgrid(fy,fx,indexing='ij')
    le = 2*np.cos(2*np.pi*ox)+2*np.cos(2*np.pi*oy)-4
    return np.fft.ifft2(np.fft.fft2(y)/(1-lam*le)).real

print("\n=== Sobolev (原代码 n_iter=200, tol=1e-6) ===")
for lam in [0.05,0.1,0.2]:
    xi,_ = sob_iter(y,lam,200)
    xe = sob_exact(y,lam)
    d = np.max(np.abs(xi-xe))
    print(f"lam={lam}: PSNR={psnr(x_true,xi,data_range=DR):.2f} SSIM={ssim(x_true,xi,data_range=DR):.4f}  迭代vs闭式 maxdiff={d:.3e}")

print("\n=== 关键核查：迭代是否真的收敛到机器精度？ ===")
for lam in [0.05,0.1,0.2]:
    for it in [200, 5000, 50000]:
        xi,_ = sob_iter(y,lam,it,do_break=False)  # 不提前break，跑满
        xe = sob_exact(y,lam)
        d = np.max(np.abs(xi-xe))
        print(f"lam={lam} n_iter={it}(不break): maxdiff={d:.3e}")

print("\n=== 收敛因子分析：慢模态 1-dt*1 ===")
for lam in [0.05,0.1,0.2]:
    dt = 0.9/(1+8*lam)
    mu_min = 1.0; mu_max = 1+8*lam
    slow = abs(1-dt*mu_min); fast = abs(1-dt*mu_max)
    rho = max(slow,fast)
    print(f"lam={lam}: dt={dt:.4f} 慢模态收缩={slow:.4f} 快模态={fast:.4f} 谱半径={rho:.4f}")

print("\n=== 免责核查：更大lam是否更好（0.2是否只是区间边缘）===")
for lam in [0.2,0.4,0.8,1.5,3.0,6.0]:
    xe = sob_exact(y,lam)
    print(f"lam={lam}: PSNR={psnr(x_true,xe,data_range=DR):.2f} SSIM={ssim(x_true,xe,data_range=DR):.4f}")

print("\n=== 边界振铃：周期性vs零边界对比（闭式解）===")
xe = sob_exact(y,0.2)
# 边缘一行的相邻差 vs 内部
edge = np.abs(xe[0,:]-xe[-1,:]).mean()  # 周期耦合的首尾差异
inner = np.abs(xe[32,:]-xe[31,:]).mean()
print(f"首尾行平均差(周期性耦合)={edge:.4f}  内部行差={inner:.4f}")

print("\n=== 局部方差(步骤3表格) ===")
c = n//2
lv_t = np.var(x_true[c-5:c+5,c-5:c+5])
lv_tk = np.var(xt_d[c-5:c+5,c-5:c+5])
print("真实",round(lv_t,4),"(md:0.0234)  Tikh_data",round(lv_tk,4),"(md:0.0174)")
for lam,md in zip([0.05,0.1,0.2],[0.0206,0.0190,0.0166]):
    xi,_ = sob_iter(y,lam,200)
    print(f"Sob lam={lam}",round(np.var(xi[c-5:c+5,c-5:c+5]),4),"(md:",md,")")
