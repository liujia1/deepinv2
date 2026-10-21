"""14.2-2 淇鏂规楠岃瘉锛氭妸 RealNVP锛坸1 閫忎紶锛夋敼鎴?2D MAF锛堣嚜鍥炲綊娴侊級銆?

v1 = 浜や粯鑴氭湰鐨?RealNVP锛歾2 = u2 閫忎紶 -> p(x2|y) 涓ユ牸鍗曢珮鏂?
v3 = MAF锛歾1 = (x1-t1)/exp(s1)锛寊2 = (x2-t2(y,z1))/exp(s2(y,z1))
     -> p(x1|y) 鍙换鎰忓舰鐘讹紙鍗曞彉閲忓崟璋冨彉鎹級锛宲(x2|y) 鏄潯浠堕珮鏂殑娣峰悎锛屽彲鍙屽嘲
瀵圭収瑙ｆ瀽鍚庨獙锛坹=0 鏃?x2 鐨勫弻宄拌竟缂橈級銆?
"""
import sys
import io
import time
import numpy as np
import torch
import torch.nn as nn

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}")

mu1 = torch.tensor([-2.5, -2.5])
mu2 = torch.tensor([2.5, 2.5])
SIGMA_P, W_PRIOR, SIGMA_OBS = 1.0, 0.5, 1.0
A = torch.tensor([[1.0, 0.0]])


def sample_prior(n, g):
    mode = torch.rand(n, 1, generator=g) < W_PRIOR
    x = torch.empty(n, 2)
    n1 = int(mode.sum().item())
    x[mode.squeeze(1)] = mu1 + SIGMA_P * torch.randn(n1, 2, generator=g)
    x[~mode.squeeze(1)] = mu2 + SIGMA_P * torch.randn(n - n1, 2, generator=g)
    return x


def forward_obs(x, g):
    noise = SIGMA_OBS * torch.randn(x.shape[0], 1, generator=g)
    return x @ A.to(x.device).t() + noise.to(x.device)


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 2))

    def forward(self, inp):
        out = self.net(inp)
        return 1.5 * torch.tanh(out[:, 0:1]), out[:, 1:2]   # s 闄愬箙 卤1.5


class MAF(nn.Module):
    """2D 鏉′欢 MAF锛歾1 鍙樻崲 x1锛寊2 浠?z1 涓烘潯浠跺彉鎹?x2銆?""

    def __init__(self):
        super().__init__()
        self.net1 = MLP(1)       # y -> (s1,t1)
        self.net2 = MLP(2)       # (y,z1) -> (s2,t2)

    def forward(self, x, y):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        s1, t1 = self.net1(y)
        z1 = (x1 - t1) * torch.exp(-s1)
        s2, t2 = self.net2(torch.cat([y, z1], dim=1))
        z2 = (x2 - t2) * torch.exp(-s2)
        return torch.cat([z1, z2], dim=1), -(s1.sum(1) + s2.sum(1))

    def inverse(self, z, y):
        z1, z2 = z[:, 0:1], z[:, 1:2]
        s1, t1 = self.net1(y)
        x1 = z1 * torch.exp(s1) + t1
        s2, t2 = self.net2(torch.cat([y, z1], dim=1))
        x2 = z2 * torch.exp(s2) + t2
        return torch.cat([x1, x2], dim=1)


def train(n_iters=6000, batch=256, seed=0):
    torch.manual_seed(seed)
    model = MAF().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed + 1)
    losses = []
    for _ in range(n_iters):
        x = sample_prior(batch, g).to(device)
        y = forward_obs(x, g).to(device)
        z, logdet = model.forward(x, y)
        nll = (0.5 * (z ** 2).sum(1) + 0.5 * z.shape[1] * np.log(2 * np.pi) - logdet).mean()
        opt.zero_grad()
        nll.backward()
        opt.step()
        losses.append(nll.item())
    return model, np.array(losses)


def analytic_components(yv):
    v_post = 1.0 / (1.0 / SIGMA_P ** 2 + 1.0 / SIGMA_OBS ** 2)
    marg_var = SIGMA_P ** 2 + SIGMA_OBS ** 2
    comps = []
    for mu, w in [(mu1, W_PRIOR), (mu2, W_PRIOR)]:
        m1 = v_post * (mu[0].item() / SIGMA_P ** 2 + yv / SIGMA_OBS ** 2)
        wp = w * np.exp(-0.5 * (yv - mu[0].item()) ** 2 / marg_var)
        comps.append((wp, m1, mu[1].item(), v_post, SIGMA_P ** 2))
    tot = sum(c[0] for c in comps)
    return [(c[0] / tot, c[1], c[2], c[3], c[4]) for c in comps]


def analytic_x2(yv):
    comps = analytic_components(yv)
    e1 = sum(c[0] * c[2] for c in comps)
    e2 = sum(c[0] * (c[4] + c[2] ** 2) for c in comps)
    e3 = sum(c[0] * (c[2] ** 3 + 3 * c[2] * c[4]) for c in comps)
    e4 = sum(c[0] * (c[2] ** 4 + 6 * c[2] ** 2 * c[4] + 3 * c[4] ** 2) for c in comps)
    var = e2 - e1 ** 2
    mu3 = e3 - 3 * e1 * e2 + 2 * e1 ** 3
    mu4 = e4 - 4 * e1 * e3 + 6 * e1 ** 2 * e2 - 3 * e1 ** 4
    return e1, var, mu3, mu4


def bc_kurt(x):
    x = np.asarray(x, dtype=np.float64)
    d = x - x.mean()
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    skew = m3 / m2 ** 1.5
    kurt = m4 / m2 ** 2
    return (skew ** 2 + 1) / kurt, kurt, skew


def gap(x, thr=1.25):
    return float(np.mean(np.abs(x) <= thr))


if __name__ == '__main__':
    n_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    t0 = time.time()
    model, losses = train(n_iters=n_iters)
    print(f"MAF 璁粌 {n_iters} iters, {time.time()-t0:.1f}s, 鏈€缁?NLL={losses[-1]:.4f}")

    for yv in [0.0, -2.5, 2.5]:
        print(f"\n===== y = {yv} =====")
        comps = analytic_components(yv)
        print("  瑙ｆ瀽鍒嗛噺:", [(round(c[0], 4), round(c[1], 4), c[2], round(c[3], 4)) for c in comps])
        m1, var, mu3, mu4 = analytic_x2(yv)
        skew = mu3 / var ** 1.5
        kurt = mu4 / var ** 2
        bc = (skew ** 2 + 1) / kurt
        # 鏁板€肩Н鍒?gap 璐ㄩ噺
        xs = np.linspace(-9, 9, 600001)
        from scipy.stats import norm
        p = sum(w * norm.pdf(xs, m, np.sqrt(v)) for w, _, m, v, _ in comps)
        sel = np.abs(xs) <= 1.25
        gap_a = np.trapezoid(p[sel], xs[sel])
        print(f"  瑙ｆ瀽 x2: mean={m1:+.4f} var={var:.4f} kurt={kurt:.4f} BC={bc:.4f} gap={gap_a:.4f}")

        torch.manual_seed(999)
        with torch.no_grad():
            z = torch.randn(20000, 2, device=device)
            y = torch.full((20000, 1), yv, device=device)
            xs_s = model.inverse(z, y).cpu().numpy()
        b, k, s = bc_kurt(xs_s[:, 1])
        print(f"  MAF:  mean=({xs_s[:,0].mean():+.4f},{xs_s[:,1].mean():+.4f}) "
              f"std=({xs_s[:,0].std():.4f},{xs_s[:,1].std():.4f}) "
              f"x2: kurt={k:.4f} skew={s:+.4f} BC={b:.4f} gap={gap(xs_s[:,1]):.4f}")

