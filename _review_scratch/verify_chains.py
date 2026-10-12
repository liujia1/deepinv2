# -*- coding: utf-8 -*-
"""4.4-1 复核：链级实验
A. 精确复现 seed=42 原始配置 -> 与 JSON 对比
B. Chambolle MaxIter 敏感性 {1,5,10,50}（同一 y、同一 Z 随机流）
C. 5 个随机种子 -> MMSE vs single 分布
D. 真实 TV-MAP（FISTA + 精确prox）-> 判断重建质量天花板
E. lambda=0.1 长链 -> 验证 lambda 权衡 / 平滑偏差
"""
import sys, math, json, time, argparse
import numpy as np
import torch
from skimage.data import camera
from skimage.transform import resize

EXP_DIR = r"d:\works\deepinv\第4章 MCMC与ULA算法\第四章 配套实验\实验4.4-1"
sys.path.insert(0, EXP_DIR)
from sampling_tools import blur_operators, chambolle_prox_TV, TVnorm

torch.set_num_threads(8)
dev = torch.device('cpu')
OUT = r'd:\works\deepinv\_review_scratch'


def setup(seed=42):
    """与原脚本完全相同的调用顺序（幂迭代也消耗 RNG）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    u = resize(camera().astype(np.float64), (128, 128), anti_aliasing=True)
    x_true = torch.tensor(u, dtype=torch.float32)  # 与原脚本 torch.Tensor(...) 一致：float32
    A, AT, AAT_norm = blur_operators([9, 9], (128, 128), "uniform", dev)
    Ax = A(x_true)
    sigma = torch.linalg.matrix_norm(Ax - torch.mean(Ax), ord='fro') / math.sqrt(128*128 * 10**(20/10))
    y = Ax + sigma * torch.randn_like(x_true)
    return x_true, A, AT, AAT_norm, y, sigma.item(), sigma.item()**2


def psnr(ref, est, mx=255.0):
    mse = torch.mean((ref-est)**2).item()
    return 10*math.log10(mx**2/mse)


def tv_iso(x):
    """正确的各向同性 TV（避免 sampling_tools.TVnorm 的 diffv.T 转置怪癖）"""
    dh = torch.zeros_like(x); dh[:, :-1] = x[:, 1:] - x[:, :-1]
    dv = torch.zeros_like(x); dv[:-1, :] = x[1:, :] - x[:-1, :]
    return torch.sqrt(dh*dh + dv*dv).sum()


def iac_ess(trace):
    """FFT 自协方差 + 初始正序列截断，返回积分自相关时间与 ESS"""
    x = np.asarray(trace, float)
    x = x - x.mean()
    n = len(x)
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f*np.conj(f))[:n].real / np.arange(n, 0, -1)
    acf /= acf[0]
    tau = 1.0
    for k in range(1, n):
        if acf[k] <= 0:
            break
        tau += 2*acf[k]
    return tau, n/tau, acf


def run_chain(x_true, A, AT, AAT_norm, y, sigma2, beta=0.3, lam_prox=1.0,
              maxiter=10, n_iter=10000, burn=4000, seed_after_y=None, tag=''):
    if seed_after_y is not None:
        torch.manual_seed(seed_after_y)
    Lf = float(AAT_norm/sigma2)
    gamma = float(0.98/(Lf + 1.0/lam_prox))
    tau = lam_prox*beta
    X = y.clone()
    M = None; S = None; k = 0
    rmse_curve = []
    tv_trace = []
    t0 = time.time()
    for i in range(n_iter):
        gl = AT(A(X)-y)/sigma2
        prx = chambolle_prox_TV(X, dev, {'lambda': tau, 'MaxIter': maxiter})
        gg = (X-prx)/lam_prox
        X = X - gamma*(gl+gg) + math.sqrt(2*gamma)*torch.randn_like(X)
        if i >= burn:
            if M is None:
                M = X.clone(); S = torch.zeros_like(X); k = 1
            else:
                k += 1
                Mn = M + (X-M)/k
                S += (X-M)*(X-Mn)
                M = Mn
            rmse_curve.append(math.sqrt(torch.mean((M-x_true)**2).item()))
            tv_trace.append(tv_iso(X).item())
    var = S/(k-1)
    std = torch.sqrt(torch.clamp(var, min=0))
    tau_int, ess, _ = iac_ess(tv_trace)
    res = {'tag': tag, 'maxiter': maxiter, 'lam_prox': lam_prox, 'beta': beta,
           'gamma': gamma, 'n_iter': n_iter, 'burn': burn, 'n_samples': k,
           'psnr_obs': psnr(x_true, y), 'psnr_mean': psnr(x_true, M),
           'psnr_last': psnr(x_true, X),
           'mean_minus_obs_db': psnr(x_true, M) - psnr(x_true, y),
           'mean_minus_single_db': psnr(x_true, M) - psnr(x_true, X),
           'mean_range': [M.min().item(), M.max().item()],
           'last_range': [X.min().item(), X.max().item()],
           'posterior_std_mean': std.mean().item(),
           'posterior_std_max': std.max().item(),
           'tau_int_TVtrace': tau_int, 'ess_TVtrace': ess,
           'rmse_start': rmse_curve[0], 'rmse_end': rmse_curve[-1],
           'wall_s': time.time()-t0}
    return res, M, std, X, rmse_curve


def fista_map(x_true, A, AT, AAT_norm, y, sigma2, beta=0.3, n_outer=800, inner=80):
    """min 1/(2σ²)||y-Ax||² + β TV 的 FISTA，prox 用高迭代次数近似精确"""
    Lf = AAT_norm/sigma2
    step = 1.0/Lf
    tau_tv = step*beta
    x = y.clone(); z = x.clone(); t = 1.0
    for _ in range(n_outer):
        xn = chambolle_prox_TV(z - step*AT(A(z)-y)/sigma2, dev,
                               {'lambda': tau_tv, 'MaxIter': inner, 'tol': 1e-9})
        tn = (1+math.sqrt(1+4*t*t))/2
        z = xn + ((t-1)/tn)*(xn-x)
        x = xn; t = tn
    g = AT(A(x)-y)/sigma2 + (x - chambolle_prox_TV(x, dev, {'lambda': tau_tv, 'MaxIter': 500, 'tol': 1e-11}))/1.0
    return {'psnr_map': psnr(x_true, x), 'map_range': [x.min().item(), x.max().item()],
            'final_drift_norm': torch.norm(g).item(), 'n_outer': n_outer}, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    args = ap.parse_args()
    allres = {}

    if args.stage in ('A', 'AB'):
        x_true, A, AT, AAT_norm, y, sigma, sigma2 = setup(42)
        print(f"setup: sigma={sigma:.6f} sigma2={sigma2:.6f} AAT_norm={AAT_norm:.6f}")
        rA, M, std, Xlast, curve = run_chain(x_true, A, AT, AAT_norm, y, sigma2,
                                             n_iter=10000, burn=4000, seed_after_y=None,
                                             tag='A_repro_seed42')
        allres['A_repro'] = rA
        print(json.dumps(rA, indent=2, ensure_ascii=False))
        np.save(OUT + '/A_mean.npy', M.numpy())
        np.save(OUT + '/A_std.npy', std.numpy())
        np.save(OUT + '/A_rmse.npy', np.array(curve))

    if args.stage == 'B':
        x_true, A, AT, AAT_norm, y, sigma, sigma2 = setup(42)
        bs = []
        for mi in [1, 5, 10, 50]:
            r, M, std, Xlast, curve = run_chain(x_true, A, AT, AAT_norm, y, sigma2,
                                                 maxiter=mi, n_iter=6000, burn=2500,
                                                 seed_after_y=777, tag=f'B_maxiter{mi}')
            bs.append(r)
            print(f"MaxIter={mi:>3}: psnr_mean={r['psnr_mean']:.3f} psnr_last={r['psnr_last']:.3f} "
                  f"std_mean={r['posterior_std_mean']:.3f} ess={r['ess_TVtrace']:.0f} wall={r['wall_s']:.0f}s")
        allres['B_maxiter_sweep'] = bs

    if args.stage == 'C':
        cs = []
        for sd in [42, 0, 1, 2, 3]:
            x_true, A, AT, AAT_norm, y, sigma, sigma2 = setup(sd)
            r, *_ = run_chain(x_true, A, AT, AAT_norm, y, sigma2,
                              n_iter=10000, burn=4000, seed_after_y=None, tag=f'C_seed{sd}')
            cs.append(r)
            print(f"seed={sd:>3}: psnr_mean={r['psnr_mean']:.3f} psnr_last={r['psnr_last']:.3f} "
                  f"mean-single={r['mean_minus_single_db']:+.3f}")
        pm = np.array([r['psnr_mean'] for r in cs]); pl = np.array([r['psnr_last'] for r in cs])
        allres['C_multiseed'] = cs
        allres['C_summary'] = {'psnr_mean_mean': pm.mean(), 'psnr_mean_std': pm.std(ddof=1),
                               'psnr_single_mean': pl.mean(), 'psnr_single_std': pl.std(ddof=1),
                               'all_mean_gt_single': bool((pm > pl).all()),
                               'mean_minus_single_mean': (pm-pl).mean()}
        print(json.dumps(allres['C_summary'], indent=2))

    if args.stage == 'D':
        x_true, A, AT, AAT_norm, y, sigma, sigma2 = setup(42)
        rD, xmap = fista_map(x_true, A, AT, AAT_norm, y, sigma2)
        allres['D_map'] = rD
        print(json.dumps(rD, indent=2))
        np.save(OUT + '/D_map.npy', xmap.numpy())

    if args.stage == 'E':
        x_true, A, AT, AAT_norm, y, sigma, sigma2 = setup(42)
        rE, M, std, Xlast, curve = run_chain(x_true, A, AT, AAT_norm, y, sigma2,
                                             lam_prox=0.1, maxiter=10,
                                             n_iter=50000, burn=20000,
                                             seed_after_y=777, tag='E_lambda0.1')
        allres['E_lambda0.1_long'] = rE
        print(json.dumps(rE, indent=2, ensure_ascii=False))
        np.save(OUT + '/E_mean_lam0.1.npy', M.numpy())

    with open(OUT + f'/chains_{args.stage}.json', 'w', encoding='utf-8') as f:
        json.dump(allres, f, ensure_ascii=False, indent=2)
    print('saved', OUT + f'/chains_{args.stage}.json')


if __name__ == '__main__':
    main()
