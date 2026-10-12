"""
实验4.4-1 诊断脚本（复核用，不覆盖实验目录任何文件）
A. max_eigenval 幂迭代是否收敛（AAT_norm 是否被低估）
B. Chambolle prox MaxIter=10 相对收敛解的误差（近似质量）
C. MYULA 主循环变体：
   - baseline: 与 4.4-1.py 完全一致（seed42, gamma=3.0316, MaxIter=10）
   - prox200: 近端算子充分收敛
   - gamma_small: gamma=0.98/(L_f+1/lambda)=0.977（满足 delta<lambda 的严格MYULA步长）
   - seeds: baseline 换 seed 1/7 看 MC 波动
   - long: baseline 跑 20000 步，看 4000-10000 窗口均值 vs 10000-20000 窗口均值漂移（burn-in充分性）
   统计：末端 RMSE、自相关时间、有效样本数、后验方差 vs 理论 2gamma
"""
import numpy as np, torch, math, sys, os, time
torch.set_num_threads(4)
sys.path.insert(0, r"D:\works\deepinv\第4章 MCMC与ULA算法\第四章 配套实验\实验4.4-1")
from skimage.data import camera
from skimage.transform import resize
from sampling_tools import *

device = torch.device('cpu')

# ---------- A. 幂迭代收敛性 ----------
u_np = resize(camera().astype(np.float64), (128, 128), anti_aliasing=True)
x_true = torch.Tensor(u_np).to(device)
nx, ny = x_true.shape
dimx = nx * ny
A, AT, AAT_norm = blur_operators([9, 9], (nx, ny), "uniform", device)

# 手工重跑幂迭代，记录历史
torch.manual_seed(42)
def power_iter_hist(max_iter):
    x = torch.normal(mean=0, std=1, size=(1, 1, nx, ny))
    x = x / torch.norm(torch.ravel(x), 2)
    init_val = 1.0
    hist = []
    for k in range(max_iter):
        y = A(x); x = AT(y)
        val = torch.norm(torch.ravel(x), 2).item()
        rel = abs(val - init_val) / init_val
        hist.append((k, val, rel))
        if rel < 1e-4:
            print(f"  幂迭代在 k={k} 收敛, val={val:.6f}")
            return val, hist
        init_val = val
        x = x / val
    print(f"  !! 幂迭代达到 max_iter={max_iter} 未收敛, 最后 val={val:.6f}, rel={rel:.3e}")
    return val, hist

print("A. max_eigenval 检查:")
val_api, _ = power_iter_hist(int(1e4))  # 复现 blur_operators 内部(其内部自己播种消耗全局RNG? 不,它用全局RNG)
# 理论值：均匀9x9核 |H|max 应为 1（DC），循环卷积下 A^TA 最大特征值=1
# 用解析 FFT 直接算 max |H|^2
h = torch.zeros(nx, ny); h[0:9, 0:9] = 1/81
c = np.ceil((np.array([9, 9]) - 1) / 2).astype("int64")
H = torch.fft.fft2(torch.roll(h, shifts=(-c[0], -c[1]), dims=(0, 1)))
print(f"  blur_operators 返回 AAT_norm = {AAT_norm:.6f}")
print(f"  解析 max|H|^2 = {(H.abs()**2).max().item():.6f}")

# ---------- 共同设置（与4.4-1.py一致） ----------
Ax = A(x_true)
BSNR = 20
sigma = torch.linalg.matrix_norm(Ax - torch.mean(Ax), ord='fro') / math.sqrt(dimx * 10 ** (BSNR / 10))
sigma2 = (sigma ** 2).item()
beta, lambda_prox = 0.3, 1.0
lam_beta = beta * lambda_prox
L_f = AAT_norm / sigma2
L_g = beta / lambda_prox

def psnr(x_ref, x_est, max_val=255.0):
    mse = torch.mean((x_ref - x_est) ** 2).item()
    return 10 * math.log10(max_val ** 2 / mse)

def run_myula(seed, gamma, n_iter, burn, maxiter_prox, tag):
    torch.manual_seed(seed); np.random.seed(seed)
    # 与主脚本RNG消耗顺序一致：blur_operators 内部幂迭代先消耗RNG，再生成 y
    A_, AT_, _ = blur_operators([9, 9], (nx, ny), "uniform", device)
    y = A_(x_true) + sigma * torch.randn_like(x_true)
    def grad_f(x): return AT_(A_(x) - y) / sigma2
    X = y.clone()
    ws = None; curve = []
    t0 = time.time()
    for i in range(n_iter):
        gp = beta * (X - chambolle_prox_TV(X, device, {'lambda': lam_beta, 'MaxIter': maxiter_prox})) / lambda_prox
        Z = torch.randn_like(X)
        X = X - gamma * (grad_f(X) + gp) + math.sqrt(2 * gamma) * Z
        if i >= burn:
            if ws is None: ws = welford(X)
            else: ws.update(X)
            if (i + 1) % 1000 == 0:
                curve.append((i + 1, torch.sqrt(torch.mean((ws.get_mean() - x_true) ** 2)).item()))
    Xm = ws.get_mean(); Xv = ws.get_var()
    print(f"[{tag}] seed={seed} gamma={gamma:.4f} MaxIter={maxiter_prox} n={n_iter}: "
          f"psnr_obs={psnr(x_true, y):.2f} psnr_mean={psnr(x_true, Xm):.2f} psnr_last={psnr(x_true, X):.2f} "
          f"var_mean={Xv.mean().item():.2f} (2gamma={2*gamma:.2f}) 用时{time.time()-t0:.0f}s")
    return Xm, X, curve, y

print("\nB. Chambolle prox 收敛质量（在链中点样本上）:")
torch.manual_seed(42); np.random.seed(42)
y0 = Ax + sigma * torch.randn_like(x_true)
Xmid = y0 + 20 * torch.randn_like(y0)  # 粗略扰动作为测试点
from sampling_tools.Grad_Image import TVnorm
for mi in [10, 50, 200, 1000]:
    p = chambolle_prox_TV(Xmid, device, {'lambda': lam_beta, 'MaxIter': mi})
    obj = lam_beta * TVnorm(p).item() + 0.5 * torch.sum((Xmid - p) ** 2).item()
    print(f"  MaxIter={mi:4d}: obj={obj:.4f}")
p200 = chambolle_prox_TV(Xmid, device, {'lambda': lam_beta, 'MaxIter': 200})
p10 = chambolle_prox_TV(Xmid, device, {'lambda': lam_beta, 'MaxIter': 10})
print(f"  ||prox10 - prox200||/||prox200|| = {(torch.norm(p10-p200)/torch.norm(p200)).item():.4e}")

print("\nC. MYULA 变体:")
g_base = 0.98 / (L_f + L_g)
g_strict = 0.98 / (L_f + 1.0 / lambda_prox)  # delta_lambda^max = 1/(L_f+1/lambda)
print(f"  baseline gamma={g_base:.4f}, 严格MYULA gamma={g_strict:.4f}")
Xm_b, Xl_b, curve_b, _ = run_myula(42, g_base, 10000, 4000, 10, "baseline")
run_myula(42, g_strict, 10000, 4000, 10, "gamma_strict")
run_myula(1, g_base, 10000, 4000, 10, "seedvar")
run_myula(7, g_base, 10000, 4000, 10, "seedvar")
Xm_l, Xl_l, curve_l, _ = run_myula(42, g_base, 20000, 4000, 10, "long20k")

print("\nD. baseline 曲线（RMSE vs 迭代）:", [(k, round(v, 3)) for k, v in curve_b])
print("   long20k 曲线:", [(k, round(v, 3)) for k, v in curve_l])
print(f"   窗口对比: mean(4000-10000) psnr={psnr(x_true, Xm_b):.2f}")
# long run 的前6000样本均值≈baseline末端均值；再算 10000-20000 段均值漂移
print(f"   |Xm_long - Xm_base|_inf = {torch.max(torch.abs(Xm_l - Xm_b)).item():.3f}")

# 自相关时间：用标号投影 s_i = <X_i - mean, x_true - mean>
torch.manual_seed(42); np.random.seed(42)
A2, AT2, _ = blur_operators([9, 9], (nx, ny), "uniform", device)
y0 = A2(x_true) + sigma * torch.randn_like(x_true)
X = y0.clone(); samples = []
for i in range(10000):
    gp = beta * (X - chambolle_prox_TV(X, device, {'lambda': lam_beta, 'MaxIter': 10})) / lambda_prox
    X = X - g_base * (AT2(A2(X) - y0) / sigma2 + gp) + math.sqrt(2 * g_base) * torch.randn_like(X)
    if i >= 4000:
        samples.append(torch.sum((X - Xm_b) * (Xm_b - x_true)).item())
s = np.array(samples); s = s - s.mean()
acf = np.correlate(s, s, 'full')[len(s)-1:] / (np.sum(s**2))
tau = 1 + 2 * np.sum(acf[1:100])
print(f"E. 投影自相关时间 tau ≈ {tau:.0f} -> 6000步有效样本 ≈ {6000/tau:.0f}")
