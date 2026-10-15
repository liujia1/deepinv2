# 修正版 TGV：div = -D^T 严格伴随；并做 scipy 参考解对比 + 完整重跑
import numpy as np, time, json, os
from skimage import data
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def grad_fwd(x):
    dh = np.zeros_like(x); dv = np.zeros_like(x)
    dh[:, :-1] = x[:, 1:] - x[:, :-1]
    dv[:-1, :] = x[1:, :] - x[:-1, :]
    return dh, dv

def div_true(p, q):   # = -D^T p
    out = np.zeros(p.shape)
    out[:, :-1] += p[:, 1:] - p[:, :-1]
    out[:, -1] -= p[:, -1]
    out[:-1, :] += q[1:, :] - q[:-1, :]
    out[-1, :] -= q[-1, :]
    return out

def sym_grad_fwd(w1, w2):
    E1 = np.zeros_like(w1); E2 = np.zeros_like(w2); d_wv_h = np.zeros_like(w2); d_wh_v = np.zeros_like(w1)
    E1[:, :-1] = w1[:, 1:] - w1[:, :-1]
    E2[:-1, :] = w2[1:, :] - w2[:-1, :]
    d_wv_h[:, :-1] = w2[:, 1:] - w2[:, :-1]
    d_wh_v[:-1, :] = w1[1:, :] - w1[:-1, :]
    E3 = 0.5 * (d_wv_h + d_wh_v)
    return E1, E2, E3

def sym_div_true(q1, q2, q3):  # = -E^T q
    out_h = np.zeros(q1.shape); out_v = np.zeros(q2.shape)
    out_h[:, :-1] += q1[:, 1:] - q1[:, :-1]
    out_h[:, -1] -= q1[:, -1]
    t = np.zeros(q3.shape)
    t[:, :-1] += 0.5 * (q3[:, 1:] - q3[:, :-1])
    t[:, -1] -= 0.5 * q3[:, -1]
    out_h += t
    out_v[:-1, :] += q2[1:, :] - q2[:-1, :]
    out_v[-1, :] -= q2[-1, :]
    t2 = np.zeros(q3.shape)
    t2[:-1, :] += 0.5 * (q3[1:, :] - q3[:-1, :])
    t2[-1, :] -= 0.5 * q3[-1, :]
    out_v += t2
    return out_h, out_v

# 伴随性数值终检
rng = np.random.default_rng(2)
N = 7
errs = []
for _ in range(20):
    x = rng.standard_normal((N,N)); p = rng.standard_normal((N,N)); q = rng.standard_normal((N,N))
    w1 = rng.standard_normal((N,N)); w2 = rng.standard_normal((N,N))
    q1 = rng.standard_normal((N,N)); q2 = rng.standard_normal((N,N)); q3 = rng.standard_normal((N,N))
    Dh, Dv = grad_fwd(x)
    errs.append(abs(np.sum(Dh*p)+np.sum(Dv*q) + np.sum(x*div_true(p,q))))
    E = sym_grad_fwd(w1, w2)
    S = sym_div_true(q1, q2, q3)
    errs.append(abs(np.sum(E[0]*q1)+np.sum(E[1]*q2)+np.sum(E[2]*q3) + np.sum(w1*S[0])+np.sum(w2*S[1])))
print(f"[adjoint] 修正版算子伴随性最大误差 = {max(errs):.3e}")

def tgv_pd(y, alpha0, alpha1, n_iter=2000, tau=0.25, sigma=0.25, trace=False):
    h, w = y.shape
    x = y.copy()
    w1 = np.zeros((h,w)); w2 = np.zeros((h,w))
    p1 = np.zeros((h,w)); p2 = np.zeros((h,w))
    q1 = np.zeros((h,w)); q2 = np.zeros((h,w)); q3 = np.zeros((h,w))
    x_bar = x.copy(); w1_bar = w1.copy(); w2_bar = w2.copy()
    tr = []
    for k in range(n_iter):
        Dx = grad_fwd(x_bar)
        p1 += sigma * (Dx[0] - w1_bar); p2 += sigma * (Dx[1] - w2_bar)
        nrm = np.sqrt(p1**2 + p2**2)
        p1 /= np.maximum(1, nrm/alpha1); p2 /= np.maximum(1, nrm/alpha1)
        E = sym_grad_fwd(w1_bar, w2_bar)
        q1 += sigma*E[0]; q2 += sigma*E[1]; q3 += sigma*E[2]
        nrmq = np.sqrt(q1**2 + q2**2 + 2*q3**2)
        q1 /= np.maximum(1, nrmq/alpha0); q2 /= np.maximum(1, nrmq/alpha0); q3 /= np.maximum(1, nrmq/alpha0)
        dp = div_true(p1, p2)
        sq = sym_div_true(q1, q2, q3)
        x_new = (x + tau*dp + tau*y) / (1+tau)
        w1_new = w1 + tau*(sq[0] - p1); w2_new = w2 + tau*(sq[1] - p2)
        x_bar = x_new + (x_new - x); w1_bar = w1_new + (w1_new - w1); w2_bar = w2_new + (w2_new - w2)
        x, w1, w2 = x_new, w1_new, w2_new
        if trace and (k+1) in (50,100,200,300,500,800,1200,1600,2000):
            Dx = grad_fwd(x); E = sym_grad_fwd(w1, w2)
            obj = 0.5*np.sum((x-y)**2) + alpha1*np.sum(np.sqrt(Dx[0]**2+Dx[1]**2+1e-24)) + alpha0*np.sum(np.sqrt(E[0]**2+E[1]**2+2*E[2]**2))
            tr.append((k+1, obj))
    if trace: return x, w1, w2, tr
    return x, w1, w2

# ===== 主实验（修正版）=====
n = 128
camera = resize(data.camera(), (n, n))
np.random.seed(42)
y = random_noise(camera, mode='gaussian', var=0.01)

t0=time.time(); x_tv = denoise_tv_chambolle(y, weight=0.2); t_tv=time.time()-t0
t0=time.time(); x_tgv, w1, w2, tr = tgv_pd(y, 0.3, 0.15, n_iter=2000, trace=True); t_tgv=time.time()-t0
print(f"[trace] 修正版目标函数下降: " + ", ".join(f"{k}:{o:.1f}" for k,o in tr))
print(f"TV  : PSNR={peak_signal_noise_ratio(camera,x_tv):.2f} SSIM={structural_similarity(camera,x_tv,data_range=1):.4f} 用时{t_tv:.2f}s")
print(f"TGV : PSNR={peak_signal_noise_ratio(camera,x_tgv):.2f} SSIM={structural_similarity(camera,x_tgv,data_range=1):.4f} 用时{t_tgv:.2f}s")
x_tgv500,_,_ = tgv_pd(y, 0.3, 0.15, n_iter=500)
print(f"TGV@500: PSNR={peak_signal_noise_ratio(camera,x_tgv500):.2f} SSIM={structural_similarity(camera,x_tgv500,data_range=1):.4f}")

# ===== scipy 参考解（小规模 32x32 全凸优化）=====
from scipy.optimize import minimize
m = 32
cam32 = resize(data.camera(), (m, m))
np.random.seed(42)
_ = random_noise(cam32, mode='gaussian', var=0.01)  # 保持随机流一致? 不, 直接重新取
rngy = np.random.default_rng(0)
y32 = np.clip(cam32 + rngy.normal(0, 0.1, cam32.shape), 0, 1)
a0, a1 = 0.3, 0.15
def obj(z):
    x = z[:m*m].reshape(m,m); w1_ = z[m*m:2*m*m].reshape(m,m); w2_ = z[2*m*m:].reshape(m,m)
    Dh, Dv = grad_fwd(x); E = sym_grad_fwd(w1_, w2_)
    return 0.5*np.sum((x-y32)**2) + a1*np.sum(np.sqrt(Dh**2+Dv**2+1e-12)) + a0*np.sum(np.sqrt(E[0]**2+E[1]**2+2*E[2]**2+1e-12))
z0 = np.concatenate([y32.reshape(-1), np.zeros(2*m*m)])
res = minimize(obj, z0, method='L-BFGS-B', options=dict(maxiter=20000, ftol=1e-14, gtol=1e-10))
x_ref = res.x[:m*m].reshape(m,m)
print(f"[scipy] L-BFGS-B obj={res.fun:.6f} nit={res.nit} status={res.status}")
x_pd32, w1p, w2p, tr32 = tgv_pd(y32, a0, a1, n_iter=20000, trace=True)
obj_pd = obj(np.concatenate([x_pd32.reshape(-1), w1p.reshape(-1), w2p.reshape(-1)]))
print(f"[scipy] PD@20000 obj={obj_pd:.6f}  目标差={obj_pd-res.fun:.3e}")
print(f"[scipy] rel||x_pd - x_ref|| = {np.linalg.norm(x_pd32-x_ref)/np.linalg.norm(x_ref):.5f}")
print(f"[scipy] PSNR(32) ref={peak_signal_noise_ratio(cam32,x_ref):.2f} pd={peak_signal_noise_ratio(cam32,x_pd32):.2f}")

# ===== 参数敏感性（修正版, 128, 500 与 2000 迭代各测一次）=====
print("\n[sensitivity] alpha1=0.15")
for it in (500, 2000):
    row = []
    for ratio in [0.5, 1.0, 2.0, 4.0, 8.0]:
        xx,_,_ = tgv_pd(y, ratio*0.15, 0.15, n_iter=it)
        row.append(f"{ratio}:{peak_signal_noise_ratio(camera,xx):.2f}/{structural_similarity(camera,xx,data_range=1):.4f}")
    print(f"  n_iter={it}: " + "  ".join(row))

# ===== alpha0 -> inf 退化为 TV 检验（修正版, alpha1=0.15 对齐）=====
x_tv15 = denoise_tv_chambolle(y, weight=0.15)
for a0b in [1.5, 15.0, 150.0]:
    xa,_,_ = tgv_pd(y, a0b, 0.15, n_iter=2000)
    print(f"[退化] a0={a0b:6.1f} ratio={a0b/0.15:6.0f}: rel||x-TV(0.15)||={np.linalg.norm(xa-x_tv15)/np.linalg.norm(x_tv15):.4f} PSNR={peak_signal_noise_ratio(camera,xa):.2f}")

# ===== 局部渐变区域指标（讲义核心卖点：阶梯效应）=====
row = n//2
def staircase_metric(img, r, sl):
    prof = img[r, sl]
    d1 = np.diff(prof)
    plateau_frac = np.mean(np.abs(d1) < 1e-3)   # 完全平坦像素对占比（阶梯平台）
    mse = np.mean((prof - camera[r, sl])**2)
    return plateau_frac, mse
sl = slice(32, 64)
for name, img in [("true", camera), ("TV", x_tv), ("TGV", x_tgv)]:
    pf, mse = staircase_metric(img, row, sl)
    print(f"[局部] {name}: 平台占比={pf:.3f} 渐变区MSE={mse:.5f}")
