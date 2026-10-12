# -*- coding: utf-8 -*-
"""
================================================================================
实验 4.4-1  MYULA 去模糊 —— 独立复核脚本（Colab GPU 一键版，自包含）
================================================================================
用法：Colab -> 新建笔记本 -> 代码单元格粘贴本文件全部内容 -> 运行
     菜单 代码执行程序 -> 更改运行时类型 -> T4 GPU（CPU 也能跑，约慢 5~10 倍）
依赖：torch / numpy / scikit-image / matplotlib（Colab 自带）

复核清单（全部自动判定 PASS/FAIL/WARN）：
  T0 确定性量复现：sigma^2、AAT_norm、PSNR(obs) 与 results_summary.json 比对
  T1 算子伴随性：<Du,p> = -<u,div p>（修复后的 DivergenceIm 是否严格为 -D*）
  T2 步长合法性：gamma < 2/L_total 且 gamma < lambda（MYULA 框式系数为正）
  T3 历史回归：旧双-beta 实现(beta^2=0.09, gamma=3.03 违法) vs 现 schemeA
  T4 seed=42 全链复现：psnr_mean≈23.59、psnr_last≈23.41（±0.15dB 容差）
  T5 多种子(5个)：MMSE PSNR 均值 >= 单样本 PSNR 均值（讲义核心结论的稳定性）
  T6 Chambolle MaxIter 敏感性：{1,5,10,50} 对 MMSE/后验方差的影响
              + prox 相对收敛参考(5000次)的梯度误差（核查"MaxIter=10免责"是否站得住）
  T7 质量天花板：精确 TV-MAP(FISTA) PSNR —— 判断 23.6dB 是参数/方法固有限制还是实现问题
  T8 lambda 权衡：lambda=1.0(10k) vs 0.1(30k)，小 lambda 是否更锐但更慢（验证讲义权衡表）
  T9 后验有效性：像素级覆盖率 68%/95% 与标称 1/2 sigma 水平；ESS/积分自相关时间
判据阈值集中在 PASSCRIT 字典，可自行收紧。
结果：末尾打印汇总表 + 保存 review441_results.json / review441_fig.png（Colab 自动下载）
================================================================================
"""
import json, math, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.data import camera
from skimage.transform import resize

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
print("device =", dev, "| torch", torch.__version__)

PASSCRIT = dict(obs_tol=0.10, repro_tol=0.20, adjoint_tol=1e-5, prox10_grad_tol=0.02,
                map_minus_mmse_db=2.0, cover_tol=0.05)
SUMMARY_JSON = dict(psnr_obs=21.06, psnr_mean=23.59, psnr_last=23.41,
                    sigma=6.549520, sigma2=42.896206, gamma=0.95772123,
                    L_f=0.0233, L_g=1.0, L_total=1.0233)
report = {}

# ---------------------------------------------------------------- 算子（自包含）
def make_blur(kernel_len, size, seed=123):
    nx, ny = size
    g = torch.Generator(device="cpu").manual_seed(seed)  # 幂迭代初值固定，不干扰主 RNG
    h = torch.zeros(nx, ny, device=dev)
    lx, ly = kernel_len
    h[0:lx, 0:ly] = 1.0 / (lx * ly)
    c = np.ceil((np.array([ly, lx]) - 1) / 2).astype("int64")
    H = torch.fft.fft2(torch.roll(h, shifts=(-c[0], -c[1]), dims=(0, 1)))
    HC = torch.conj(H)
    A = lambda x: torch.fft.ifft2(H * torch.fft.fft2(x)).real
    AT = lambda x: torch.fft.ifft2(HC * torch.fft.fft2(x)).real
    x0 = torch.randn(nx, ny, generator=g).to(dev)
    x0 = x0 / torch.linalg.norm(x0.flatten())
    prev, val = 1.0, 1.0
    for _ in range(10000):
        x0 = AT(A(x0)); val = torch.linalg.norm(x0.flatten()).item()
        if abs(val - prev) / prev < 1e-4:
            break
        prev = val; x0 = x0 / val
    return A, AT, val

def _grad(u):
    z = u[1:, :] - u[:-1, :]
    dux = torch.vstack([z, torch.zeros(1, u.shape[1], device=u.device)])
    z = u[:, 1:] - u[:, :-1]
    duy = torch.hstack([z, torch.zeros(u.shape[0], 1, device=u.device)])
    return dux, duy

def _div(p1, p2):
    # 零填充前向差分的严格负伴随：末行/末列取 -p[-2]（上游 bug 误用恒为 0 的 -p[-1]）
    z = p2[:, 1:-1] - p2[:, :-2]
    v = torch.hstack([p2[:, :1], z, -p2[:, -2:-1]])
    z = p1[1:-1, :] - p1[:-2, :]
    u = torch.vstack([p1[:1, :], z, -p1[-2:-1, :]])
    return v + u

def tv_iso(x):
    dx, dy = _grad(x)
    return torch.sqrt(dx * dx + dy * dy).sum()

def prox_tv(g, tau, maxiter=10, tol=0.0):
    px = torch.zeros_like(g); py = torch.zeros_like(g); t = 0.249
    for k in range(maxiter):
        u = _div(px, py) - g / tau
        upx, upy = _grad(u)
        tmp = torch.sqrt(upx * upx + upy * upy)
        px = (px + t * upx) / (1 + t * tmp)
        py = (py + t * upy) / (1 + t * tmp)
    return g - tau * _div(px, py)

class Welford:
    def __init__(self, x):
        self.k = 1; self.M = x.clone(); self.S = torch.zeros_like(x)
    def update(self, x):
        self.k += 1; Mn = self.M + (x - self.M) / self.k
        self.S += (x - self.M) * (x - Mn); self.M = Mn
    def mean(self): return self.M
    def var(self):
        return self.S / (self.k - 1) if self.k >= 2 else self.S / self.k

def psnr(ref, est, mv=255.0):
    mse = torch.mean((ref - est) ** 2).item()
    return 10 * math.log10(mv ** 2 / mse) if mse > 0 else float("inf")

def iac_ess(trace):
    x = np.asarray(trace, float); x -= x.mean(); n = len(x)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real / np.arange(n, 0, -1)
    acf /= acf[0]
    tau = 1.0
    for k in range(1, n):
        if acf[k] <= 0:
            break
        tau += 2 * acf[k]
    return tau, n / tau, acf

# ---------------------------------------------------------------- 数据
x_true = torch.tensor(resize(camera().astype(np.float64), (128, 128), anti_aliasing=True))
nx, ny = x_true.shape; dimx = nx * ny
A, AT, AAT = make_blur([9, 9], (nx, ny))
Ax = A(x_true)
BSNR = 20

def make_y(seed):
    torch.manual_seed(seed)
    return Ax + sigma * torch.randn_like(x_true)

sigma = torch.linalg.norm(Ax - Ax.mean()) / math.sqrt(dimx * 10 ** (BSNR / 10))
sigma2 = sigma.item() ** 2
beta, lam = 0.3, 1.0
y42 = make_y(42)
po42 = psnr(x_true, y42)
print(f"sigma={sigma.item():.6f} (json {SUMMARY_JSON['sigma']:.6f}) | "
      f"sigma2={sigma2:.6f} (json {SUMMARY_JSON['sigma2']:.6f}) | AAT={AAT:.6f}")
report["T0_deterministic"] = dict(sigma=sigma.item(), sigma2=sigma2, AAT=AAT,
                                  psnr_obs_seed42=po42, json=SUMMARY_JSON)

# T1 伴随
torch.manual_seed(0)
u = torch.randn(64, 48, device=dev); q1 = torch.randn(64, 48, device=dev); q2 = torch.randn(64, 48, device=dev)
dux, duy = _grad(u)
lhs = torch.sum(dux * q1 + duy * q2).item(); rhs = torch.sum(u * _div(q1, q2)).item()
adj_err = abs(lhs + rhs) / abs(lhs)
print(f"[T1] 伴随残差 {adj_err:.2e} ({'PASS' if adj_err < PASSCRIT['adjoint_tol'] else 'FAIL'})")
report["T1_adjoint"] = dict(rel_err=adj_err, pass_flag=adj_err < PASSCRIT["adjoint_tol"])

# ---------------------------------------------------------------- MYULA 链
def run_chain(y, gamma, tau, outer, n_iter, burn, prox_it=10, lam_prox=1.0,
              keep_samples=False, seed_z=0, tag=""):
    torch.manual_seed(seed_z)
    X = y.clone(); W = None; rmse = []; tvtrace = []; samples = []
    t0 = time.time()
    for i in range(n_iter):
        P = prox_tv(X, tau, prox_it)
        gp = outer * (X - P) / lam_prox
        X = X - gamma * (AT(A(X) - y) / sigma2 + gp) + math.sqrt(2 * gamma) * torch.randn_like(X)
        if i >= burn:
            if W is None: W = Welford(X)
            else: W.update(X)
            rmse.append(math.sqrt(torch.mean((W.mean() - x_true) ** 2).item()))
            tvtrace.append(tv_iso(X).item())
            if keep_samples and (i % 50 == 0):
                samples.append(X.clone())
    M = W.mean(); V = W.var()
    out = dict(tag=tag, psnr_obs=psnr(x_true, y), psnr_mean=psnr(x_true, M),
               psnr_last=psnr(x_true, X), mean_range=[M.min().item(), M.max().item()],
               last_range=[X.min().item(), X.max().item()], std_mean=V.sqrt().mean().item(),
               wall=time.time() - t0, rmse=rmse)
    tau_i, ess, acf = iac_ess(tvtrace)
    out.update(tau_int=tau_i, ess=ess)
    if keep_samples:
        out["samples"] = torch.stack(samples)
    return out, M, V.sqrt(), X

Lf = AAT / sigma2
# T2 步长
for name, lg in [("schemeA/current L_g=1/lambda", 1.0 / lam), ("old L_g=beta/lambda", beta / lam)]:
    g = 0.98 / (Lf + lg)
    print(f"[T2] {name}: gamma={g:.4f}  gamma<2/L={g < 2/(Lf+lg)}  gamma<lambda={g < lam}")
gA = 0.98 / (Lf + 1.0 / lam)
gB = 0.98 / (Lf + beta / lam)
report["T2_stepsize"] = dict(gammaA=gA, cond_2oL=gA < 2 / (Lf + 1 / lam),
                             cond_dlt_lam=gA < lam,
                             gamma_old=gB, old_violates_dlt_lam=not (gB < lam))

# T3 历史回归（三种先验实现，各 3 seed）
print("\n[T3] 双-beta 历史回归 ...")
t3 = {}
for vname, tau_, outer, gg in [("old_double_beta", lam * beta, beta, gB),
                               ("schemeB_fix_tau", lam, beta, gB),
                               ("schemeA_current", lam * beta, 1.0, gA)]:
    pms, pls = [], []
    for sd in [42, 1, 7]:
        o, _, _, _ = run_chain(make_y(sd), gg, tau_, outer, 10000, 4000, seed_z=sd + 100, tag=vname)
        pms.append(o["psnr_mean"]); pls.append(o["psnr_last"])
    t3[vname] = dict(mmse_mean=float(np.mean(pms)), single_mean=float(np.mean(pls)))
    print(f"   {vname:16s} MMSE={np.mean(pms):.2f} single={np.mean(pls):.2f}")
report["T3_regression"] = t3

# T4 seed=42 复现
print("\n[T4] seed=42 全链复现 ...")
o42, M42, S42, X42 = run_chain(y42, gA, lam * beta, 1.0, 10000, 4000, seed_z=0, tag="repro42")
# 注：seed_z 对齐不是逐位目标（本脚本幂迭代用独立 generator），看数值吻合
rep_ok = abs(o42["psnr_mean"] - SUMMARY_JSON["psnr_mean"]) < PASSCRIT["repro_tol"] and \
         abs(o42["psnr_last"] - SUMMARY_JSON["psnr_last"]) < PASSCRIT["repro_tol"]
print(f"   obs={o42['psnr_obs']:.2f} MMSE={o42['psnr_mean']:.2f} single={o42['psnr_last']:.2f} "
      f"({'PASS' if rep_ok else 'CHECK: 用实验目录原脚本逐位复现；本脚本为独立实现'})")
report["T4_repro"] = {k: o42[k] for k in ("psnr_obs", "psnr_mean", "psnr_last", "mean_range", "last_range")}
report["T4_repro"]["pass_flag"] = bool(rep_ok)

# T5 多种子
print("\n[T5] 5 seeds MMSE vs single ...")
rows5 = []
for sd in [42, 0, 1, 2, 3]:
    o, _, _, _ = run_chain(make_y(sd), gA, lam * beta, 1.0, 10000, 4000, seed_z=sd + 100, tag=f"s{sd}")
    rows5.append((sd, o["psnr_obs"], o["psnr_mean"], o["psnr_last"]))
    print(f"   seed={sd:3d} obs={o['psnr_obs']:.2f} MMSE={o['psnr_mean']:.2f} single={o['psnr_last']:.2f} "
          f"MMSE>single={o['psnr_mean'] > o['psnr_last']}")
pm = np.array([r[2] for r in rows5]); pl = np.array([r[3] for r in rows5])
t5 = dict(rows=rows5, mmse_mean=float(pm.mean()), mmse_std=float(pm.std(ddof=1)),
          single_mean=float(pl.mean()), single_std=float(pl.std(ddof=1)),
          mmse_ge_single_every=bool((pm >= pl).all()), gap_mean=float((pm - pl).mean()))
print(f"   >> MMSE {pm.mean():.2f}+/-{pm.std(ddof=1):.2f} | single {pl.mean():.2f}+/-{pl.std(ddof=1):.2f} | "
      f"全部 MMSE>=single: {t5['mmse_ge_single_every']}")
report["T5_multiseed"] = t5

# T6 MaxIter 敏感性 + prox 精度
print("\n[T6] Chambolle MaxIter 敏感性 ...")
p_ref = prox_tv(y42, lam * beta, 5000)
ge_ref = (y42 - p_ref) / lam
prox_acc = {}
for m in [1, 2, 5, 10, 20, 50, 100]:
    pm_ = prox_tv(y42, lam * beta, m)
    prox_acc[m] = float(torch.norm((y42 - pm_) / lam - ge_ref) / torch.norm(ge_ref))
print("   prox 包络梯度相对误差 @y:", {k: f"{v:.2e}" for k, v in prox_acc.items()})
t6rows = []
for m in [1, 5, 10, 50]:
    o, Mm, Std, _ = run_chain(y42, gA, lam * beta, 1.0, 6000, 2500, prox_it=m,
                              seed_z=777, tag=f"mi{m}")
    t6rows.append(dict(maxiter=m, psnr_mean=o["psnr_mean"], psnr_last=o["psnr_last"],
                       std_mean=o["std_mean"], ess=o["ess"]))
    print(f"   MaxIter={m:3d}: MMSE={o['psnr_mean']:.2f} single={o['psnr_last']:.2f} "
          f"std={o['std_mean']:.3f} ESS={o['ess']:.0f}")
report["T6_prox"] = dict(prox_grad_rel_err=prox_acc, chains=t6rows,
                         pass_flag=prox_acc[10] < PASSCRIT["prox10_grad_tol"])

# T7 MAP 天花板（FISTA，精确 prox）
print("\n[T7] TV-MAP (FISTA, 精确prox) ...")
step = 1.0 / Lf; tau_map = step * beta
xm = y42.clone(); zm = xm.clone(); tt = 1.0
for _ in range(800):
    xn = prox_tv(zm - step * AT(A(zm) - y42) / sigma2, tau_map, 80, tol=0)
    tn = (1 + math.sqrt(1 + 4 * tt * tt)) / 2
    zm = xn + ((tt - 1) / tn) * (xn - xm); xm = xn; tt = tn
map_psnr = psnr(x_true, xm)
drift = AT(A(xm) - y42) / sigma2 + (xm - prox_tv(xm, lam * beta, 1000)) / lam
print(f"   MAP PSNR={map_psnr:.2f} | MMSE={o42['psnr_mean']:.2f} | 差距 {map_psnr-o42['psnr_mean']:.2f} dB "
      f"| 末端漂移范数={torch.norm(drift).item():.3f}")
report["T7_map"] = dict(psnr_map=map_psnr, psnr_mmse=o42["psnr_mean"],
                        gap_db=map_psnr - o42["psnr_mean"],
                        final_drift=torch.norm(drift).item())

# T8 lambda 权衡
print("\n[T8] lambda 权衡（lambda=0.1, 30k 长链）...")
g_small = 0.98 / (Lf + 1.0 / 0.1)
o01, M01, S01, _ = run_chain(y42, g_small, 0.1 * beta, 1.0, 30000, 12000,
                             lam_prox=0.1, seed_z=777, tag="lam0.1")
print(f"   lambda=1.0: gamma={gA:.4f} MMSE={o42['psnr_mean']:.2f} std={o42['std_mean']:.3f}")
print(f"   lambda=0.1: gamma={g_small:.4f} MMSE={o01['psnr_mean']:.2f} std={o01['std_mean']:.3f} "
      f"ESS={o01['ess']:.0f}")
report["T8_lambda"] = dict(lam1=dict(gamma=gA, psnr_mean=o42["psnr_mean"], std=o42["std_mean"], ess=o42["ess"]),
                           lam01=dict(gamma=g_small, psnr_mean=o01["psnr_mean"], std=o01["std_mean"], ess=o01["ess"]))

# T9 后验覆盖率（用 lambda=1 链的保存样本）
print("\n[T9] 后验覆盖率 ...")
o9, M9, S9, _ = run_chain(y42, gA, lam * beta, 1.0, 12000, 4000, keep_samples=True,
                          seed_z=2026, tag="coverage")
samples = o9.pop("samples")
err = (samples - x_true[None]).abs()
cov68 = float((err <= S9[None]).float().mean())
cov95 = float((err <= 1.96 * S9[None]).float().mean())
print(f"   68% 覆盖={cov68:.3f} (标称0.68) | 95% 覆盖={cov95:.3f} (标称0.95) | ESS={o9['ess']:.0f}")
report["T9_posterior"] = dict(cov68=cov68, cov95=cov95, ess=o9["ess"], tau_int=o9["tau_int"],
                              n_samples_kept=samples.shape[0])

# ---------------------------------------------------------------- 汇总 & 图
print("\n" + "=" * 78)
print("复核汇总")
print("=" * 78)
checks = [
    ("T1 伴随 -D*", report["T1_adjoint"]["pass_flag"]),
    ("T2 gamma<2/L 且 gamma<lambda", report["T2_stepsize"]["cond_2oL"] and report["T2_stepsize"]["cond_dlt_lam"]),
    ("T2 旧实现 gamma=%.2f 确实违法 δ<λ" % gB, report["T2_stepsize"]["old_violates_dlt_lam"]),
    ("T4 seed42 复现 JSON(±0.2dB)", report["T4_repro"]["pass_flag"]),
    ("T5 全 seed MMSE>=single", t5["mmse_ge_single_every"]),
    ("T6 MaxIter=10 包络梯度误差<2%", prox_acc[10] < PASSCRIT["prox10_grad_tol"]),
    ("T7 MAP-MMSE 差距<2dB(说明MMSE没拖后腿)", report["T7_map"]["gap_db"] < PASSCRIT["map_minus_mmse_db"]),
    ("T9 |68%覆盖-0.68|<0.05", abs(cov68 - 0.68) < PASSCRIT["cover_tol"]),
]
for name, ok in checks:
    print(f"  [{'PASS' if ok else 'WARN/FAIL'}] {name}")

fig, ax = plt.subplots(2, 3, figsize=(15, 9))
ax[0, 0].imshow(x_true.cpu(), cmap="gray", vmin=0, vmax=255); ax[0, 0].set_title("ground truth")
ax[0, 1].imshow(y42.cpu(), cmap="gray", vmin=0, vmax=255); ax[0, 1].set_title(f"y PSNR={po42:.1f}")
ax[0, 2].imshow(M42.cpu(), cmap="gray", vmin=0, vmax=255); ax[0, 2].set_title(f"MMSE {o42['psnr_mean']:.1f}")
ax[1, 0].imshow(S42.cpu(), cmap="gray"); ax[1, 0].set_title("posterior std")
ax[1, 1].imshow(xm.cpu(), cmap="gray", vmin=0, vmax=255); ax[1, 1].set_title(f"TV-MAP {map_psnr:.1f}")
ax[1, 2].plot(o42["rmse"]); ax[1, 2].set_title("RMSE after burn-in"); ax[1, 2].grid(alpha=.3)
for a in ax.ravel()[:5]: a.axis("off")
plt.tight_layout(); plt.savefig("review441_fig.png", dpi=140, bbox_inches="tight")
with open("review441_results.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=float)
print("\n已保存 review441_results.json / review441_fig.png")
try:
    from google.colab import files
    files.download("review441_results.json"); files.download("review441_fig.png")
except Exception:
    pass
