import numpy as np, torch, os
SAVE = r"d:\works\deepinv\第4章 MCMC与ULA算法\第四章 配套实验\实验4.5-1"
c = torch.load(os.path.join(SAVE,'experiment_4.5-1_cache.pth'), map_location='cpu', weights_only=False)
u = c['u']; f = c['f']; u_hq = c['u_hq'].numpy(); u_avg = c['u_avg_'].numpy(); u_var = c['u_var_'].numpy()
sigma = c['sigma']; N = c['N']
print("keys:", list(c.keys()))
print("sigma", sigma, "N", N)
print("dtype u", u.dtype, "u range", u.min(), u.max())

def psnr(x, ref, peak=1.0):
    mse = np.mean((x-ref)**2)
    return 10*np.log10(peak**2/mse), mse

p_f, m_f = psnr(f, u)
p_hq, m_hq = psnr(u_hq, u)
p_avg, m_avg = psnr(u_avg, u)
print(f"\nPSNR noisy vs GT      : {p_f:.2f} dB (mse {m_f:.5f})")
print(f"PSNR MAP (hq) vs GT   : {p_hq:.2f} dB (mse {m_hq:.5f})")
print(f"PSNR MMSE (glm) vs GT : {p_avg:.2f} dB (mse {m_avg:.5f})")
print(f"MAP better than MMSE?  {m_hq < m_avg}")

# posterior std map analysis
std = np.sqrt(np.clip(u_var, 0, None))
print(f"\nposterior std: min {std.min():.4f} max {std.max():.4f} mean {std.mean():.4f}")
# edge magnitude of GT
gx = np.abs(np.diff(u, axis=1)); gy = np.abs(np.diff(u, axis=0))
edge = np.zeros_like(u); edge[:, :-1] += gx; edge[:-1, :] += gy
edge = edge/edge.max()
flat_mask = edge < 0.05
edge_mask = edge > 0.3
print(f"mean std in flat regions  : {std[flat_mask].mean():.4f}  (n={flat_mask.sum()})")
print(f"mean std in strong-edge   : {std[edge_mask].mean():.4f}  (n={edge_mask.sum()})")
print(f"corr(std, edge_mag)       : {np.corrcoef(std.ravel(), edge.ravel())[0,1]:.3f}")
# fraction of std map that is 'saturated high'
q = np.percentile(std, [50,90,99])
print(f"std percentiles 50/90/99  : {q}")
print(f"fraction std > 0.5*max    : {(std> 0.5*std.max()).mean():.3f}")
