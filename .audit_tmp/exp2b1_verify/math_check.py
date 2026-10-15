# 实验2.b-1 数学核验：算子伴随性 / 步长条件 ||K||^2 / 收敛性 / alpha0->inf 退化
import numpy as np, sys, time
sys.path.insert(0, r"D:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\第二章 配套实验\实验2.b-1")
import importlib.util
spec = importlib.util.spec_from_file_location("m", r"D:\works\deepinv\.audit_tmp\exp2b1_verify\exp2b1.py")

# 直接复制算子（避免import执行整脚本）
def gradient_forward(x):
    h, w = x.shape
    dx_h = np.zeros_like(x); dx_v = np.zeros_like(x)
    dx_h[:, :-1] = x[:, 1:] - x[:, :-1]
    dx_v[:-1, :] = x[1:, :] - x[:-1, :]
    return dx_h, dx_v

def divergence_backward(p_h, p_v):
    h, w = p_h.shape
    div = np.zeros((h, w))
    div[:, 1:] += p_h[:, :-1]
    div -= p_h
    div[1:, :] += p_v[:-1, :]
    div -= p_v
    return div

def symmetrized_gradient_forward(w_h, w_v):
    Ew_1 = np.zeros_like(w_h); Ew_2 = np.zeros_like(w_v); Ew_3 = np.zeros_like(w_h)
    Ew_1[:, :-1] = w_h[:, 1:] - w_h[:, :-1]
    Ew_2[:-1, :] = w_v[1:, :] - w_v[:-1, :]
    d_wv_h = np.zeros_like(w_v); d_wh_v = np.zeros_like(w_h)
    d_wv_h[:, :-1] = w_v[:, 1:] - w_v[:, :-1]
    d_wh_v[:-1, :] = w_h[1:, :] - w_h[:-1, :]
    Ew_3 = 0.5 * (d_wv_h + d_wh_v)
    return Ew_1, Ew_2, Ew_3

def symmetrized_divergence_backward(q_1, q_2, q_3):
    div_w_h = np.zeros_like(q_1); div_w_v = np.zeros_like(q_2)
    div_w_h[:, 1:] += q_1[:, :-1]; div_w_h -= q_1
    div_w_v[1:, :] += q_2[:-1, :]; div_w_v -= q_2
    dq3_dh = np.zeros_like(q_3); dq3_dv = np.zeros_like(q_3)
    dq3_dh[:, 1:] += 0.5 * q_3[:, :-1]; dq3_dh -= 0.5 * q_3
    dq3_dv[1:, :] += 0.5 * q_3[:-1, :]; dq3_dv -= 0.5 * q_3
    div_w_h += dq3_dv; div_w_v += dq3_dh
    return div_w_h, div_w_v

def tgv_denoise_primal_dual(y, alpha0, alpha1, n_iter=500, tau=0.25, sigma=0.25, ret_obj=False):
    h, w = y.shape
    x = y.copy()
    w_h = np.zeros((h, w)); w_v = np.zeros((h, w))
    p_h = np.zeros((h, w)); p_v = np.zeros((h, w))
    q_1 = np.zeros((h, w)); q_2 = np.zeros((h, w)); q_3 = np.zeros((h, w))
    theta = 1.0
    x_bar = x.copy(); w_h_bar = w_h.copy(); w_v_bar = w_v.copy()
    objs = []
    for k in range(n_iter):
        Dx_h, Dx_v = gradient_forward(x_bar)
        p_h = p_h + sigma * (Dx_h - w_h_bar); p_v = p_v + sigma * (Dx_v - w_v_bar)
        norm_p = np.sqrt(p_h**2 + p_v**2)
        p_h = p_h / np.maximum(1, norm_p / alpha1); p_v = p_v / np.maximum(1, norm_p / alpha1)
        Ew_1, Ew_2, Ew_3 = symmetrized_gradient_forward(w_h_bar, w_v_bar)
        q_1 = q_1 + sigma * Ew_1; q_2 = q_2 + sigma * Ew_2; q_3 = q_3 + sigma * Ew_3
        norm_q = np.sqrt(q_1**2 + q_2**2 + 2*q_3**2)
        q_1 = q_1 / np.maximum(1, norm_q / alpha0); q_2 = q_2 / np.maximum(1, norm_q / alpha0); q_3 = q_3 / np.maximum(1, norm_q / alpha0)
        div_p = divergence_backward(p_h, p_v)
        div_q_h, div_q_v = symmetrized_divergence_backward(q_1, q_2, q_3)
        x_new = (x + tau * div_p + tau * y) / (1 + tau)
        w_h_new = w_h + tau * (div_q_h - p_h); w_v_new = w_v + tau * (div_q_v - p_v)
        x_bar = x_new + theta * (x_new - x); w_h_bar = w_h_new + theta * (w_h_new - w_h); w_v_bar = w_v_new + theta * (w_v_new - w_v)
        x = x_new; w_h = w_h_new; w_v = w_v_new
        if ret_obj and (k+1) % 50 == 0:
            Dh, Dv = gradient_forward(x)
            E1, E2, E3 = symmetrized_gradient_forward(w_h, w_v)
            # 加权E范数要与对偶球一致: sqrt(||q1||^2+||q2||^2+2||q3||^2)
            e_norm = np.sqrt(E1**2 + E2**2 + 2*E3**2)
            obj = 0.5*np.sum((x-y)**2) + alpha1*np.sum(np.sqrt(Dh**2+Dv**2+1e-24)) + alpha0*np.sum(e_norm)
            objs.append((k+1, obj))
    if ret_obj: return x, w_h, w_v, objs
    return x, w_h, w_v

rng = np.random.default_rng(0)
h, w = 16, 16
# ==== 测试1：D 与 -div 的伴随性  <Dx,p> == <x,-div p> ====
x1 = rng.standard_normal((h, w)); p1 = rng.standard_normal((h, w)); p2 = rng.standard_normal((h, w))
Dx = gradient_forward(x1)
lhs = np.sum(Dx[0]*p1) + np.sum(Dx[1]*p2)
dv = divergence_backward(p1, p2)
rhs = -np.sum(x1*dv)
print(f"[T1] D-div adjoint: <Dx,p>={lhs:.6f}, <-x,div p>={rhs:.6f}, rel_err={abs(lhs-rhs)/max(abs(lhs),1e-9):.2e}")

# ==== 测试2：E 与 sym_div 的伴随性  <Ew,q> == <w, -div_sym q> ====
w1 = rng.standard_normal((h, w)); w2 = rng.standard_normal((h, w))
q1 = rng.standard_normal((h, w)); q2 = rng.standard_normal((h, w)); q3 = rng.standard_normal((h, w))
E = symmetrized_gradient_forward(w1, w2)
lhs = np.sum(E[0]*q1) + np.sum(E[1]*q2) + np.sum(E[2]*q3)
dq = symmetrized_divergence_backward(q1, q2, q3)
rhs = -np.sum(w1*dq[0]) - np.sum(w2*dq[1])
print(f"[T2] E-symdiv adjoint: <Ew,q>={lhs:.6f}, <-w,div q>={rhs:.6f}, rel_err={abs(lhs-rhs)/max(abs(lhs),1e-9):.2e}")

# ==== 测试3：联合算子 K=[D;E] 的谱范数（幂迭代），验证 tau*sigma*||K||^2<=1 ====
def apply_K(u, v1, v2):
    Dx = gradient_forward(u)
    E = symmetrized_gradient_forward(v1, v2)
    return (Dx[0], Dx[1], E[0], E[1], E[2])
def apply_Kt(a, b, c, d, e):
    dv = divergence_backward(a, b)
    sq = symmetrized_divergence_backward(c, d, e)
    return dv, sq[0], sq[1]
# 加权内积：q部分与代码norm_q一致，用 sqrt(q1^2+q2^2+2q3^2) => 即 E 的第三个分量按 sqrt(2) 计权
# ||K||^2 = max (||Dx||^2 + ||E1w||^2+||E2w||^2+2||E3w||^2) / ||x||^2+||w||^2
vecs = (rng.standard_normal((h,w)), rng.standard_normal((h,w)), rng.standard_normal((h,w)))
norm = np.sqrt(sum(np.sum(v**2) for v in vecs))
vecs = tuple(v/norm for v in vecs)
lam = 0.0
for it in range(3000):
    Kp = apply_K(*vecs)
    # 加权：第5分量乘2（对应2*q3^2）
    wts = (Kp[0], Kp[1], Kp[2], Kp[3], 2*Kp[4])
    KtKv = apply_Kt(*wts)
    sq = np.sqrt(sum(np.sum(v**2) for v in KtKv))
    new_lam = sq
    vecs = tuple(v/sq for v in KtKv)
    lam = new_lam
print(f"[T3] ||K||^2 (weighted, 幂迭代) = {lam:.4f}  -> 需要 tau*sigma <= {1/lam:.4f}; 实际 tau*sigma = 0.0625")
print(f"     代码注释声称 ||K||≈sqrt(8)={np.sqrt(8):.3f}, ||K||^2=8")

# ==== 测试4：目标函数随迭代单调性与收敛（500 vs 2000 iter 对比）====
from skimage import data
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from skimage.transform import resize
n = 128
camera = resize(data.camera(), (n, n))
np.random.seed(42)
camera_noisy = random_noise(camera, mode='gaussian', var=0.01)
t0 = time.time()
x500, wh, wv, objs = tgv_denoise_primal_dual(camera_noisy, 0.3, 0.15, n_iter=2000, ret_obj=True)
t1 = time.time()
print(f"[T4] 2000次迭代耗时 {t1-t0:.1f}s (128x128, CPU numpy)")
for k, o in objs[::2]:
    print(f"     iter {k:5d}: objective = {o:.6f}")
# 取500次与2000次的x差
x500only, _, _ = tgv_denoise_primal_dual(camera_noisy, 0.3, 0.15, n_iter=500)
print(f"[T4] ||x_2000 - x_500||_F / ||x_500|| = {np.linalg.norm(x500only-x500[:,:,:] if False else x500)/np.linalg.norm(x500only):.5f}" if False else "")
print(f"[T4] PSNR@500 = {peak_signal_noise_ratio(camera, x500only):.2f} dB; PSNR@2000 = {peak_signal_noise_ratio(camera, x500):.2f} dB")

# ==== 测试5：alpha0 -> 大 时是否退化接近 TV（与 alpha1*TV 比较）====
from skimage.restoration import denoise_tv_chambolle
x_tv15 = denoise_tv_chambolle(camera_noisy, weight=0.15)
for a0 in [3.0, 30.0, 300.0]:
    xa, _, _ = tgv_denoise_primal_dual(camera_noisy, a0, 0.15, n_iter=2000)
    d = np.linalg.norm(xa - x_tv15)/np.linalg.norm(x_tv15)
    print(f"[T5] alpha0={a0:6.1f} (ratio={a0/0.15:.0f}): rel ||x - TV(w=0.15)|| = {d:.4f}, PSNR={peak_signal_noise_ratio(camera, xa):.2f}dB")
