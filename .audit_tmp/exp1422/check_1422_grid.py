"""14.2-2 定方案：结构 x 观测强度的 2x2 网格。

结构: nvp = 交付脚本的两层仿射耦合（x1 透传，p(x2|y) 必为单高斯）
      maf = z1 变换 x1、z2 以 z1 为条件变换 x2（p(x2|y) 可为混合）
观测: sigma_obs = 0.25（当前交付值，x1 后验双峰仅 1.2 个标准差）
      sigma_obs = 1.0 （弱观测，x1 后验双峰约 3.5 个标准差）
指标: y=0 时 x2 边缘的双峰系数 BC / 峰度 / 谷区质量 P(|x2|<=1.25)，对照解析值。
"""
import sys
import io
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}")

MU = torch.tensor([[-2.5, -2.5], [2.5, 2.5]])
SIGMA_P, W_PRIOR = 1.0, 0.5
A = torch.tensor([[1.0, 0.0]])


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 2))

    def forward(self, inp):
        out = self.net(inp)
        return 1.5 * torch.tanh(out[:, 0:1]), out[:, 1:2]


class Flow(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.net1 = MLP(1)                 # y -> (s1,t1)
        self.net2 = MLP(2)                 # 层2 条件输入（nvp: y+u2；maf: y+z1）

    def forward(self, x, y):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        s1, t1 = self.net1(y)
        if self.kind == 'nvp':
            u2 = x2 * torch.exp(s1) + t1          # x1 透传
            s2, t2 = self.net2(torch.cat([y, u2], dim=1))
            z1 = x1 * torch.exp(s2) + t2
            z2 = u2
            logdet = s1.sum(1) + s2.sum(1)        # z = x*exp(s)+t -> +s
        else:
            z1 = (x1 - t1) * torch.exp(-s1)       # 先变换 x1
            s2, t2 = self.net2(torch.cat([y, z1], dim=1))
            z2 = (x2 - t2) * torch.exp(-s2)
            logdet = -(s1.sum(1) + s2.sum(1))     # z = (x-t)*exp(-s) -> -s
        return torch.cat([z1, z2], dim=1), logdet

    def inverse(self, z, y):
        z1, z2 = z[:, 0:1], z[:, 1:2]
        s1, t1 = self.net1(y)
        if self.kind == 'nvp':
            u2 = z2
            s2, t2 = self.net2(torch.cat([y, u2], dim=1))
            x1 = (z1 - t2) * torch.exp(-s2)
            x2 = (u2 - t1) * torch.exp(-s1)
        else:
            x1 = z1 * torch.exp(s1) + t1
            s2, t2 = self.net2(torch.cat([y, z1], dim=1))
            x2 = z2 * torch.exp(s2) + t2
        return torch.cat([x1, x2], dim=1)


def make_data_fns(sigma_obs):
    def sample_prior(n, g):
        mode = torch.rand(n, 1, generator=g) < W_PRIOR
        x = torch.empty(n, 2)
        n1 = int(mode.sum().item())
        x[mode.squeeze(1)] = MU[0] + SIGMA_P * torch.randn(n1, 2, generator=g)
        x[~mode.squeeze(1)] = MU[1] + SIGMA_P * torch.randn(n - n1, 2, generator=g)
        return x

    def forward_obs(x, g):
        noise = sigma_obs * torch.randn(x.shape[0], 1, generator=g)
        return x @ A.to(x.device).t() + noise.to(x.device)

    return sample_prior, forward_obs


def train(kind, sigma_obs, n_iters=6000, batch=256, seed=0):
    sample_prior, forward_obs = make_data_fns(sigma_obs)
    torch.manual_seed(seed)
    model = Flow(kind).to(device)
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


def analytic(yv, sigma_obs):
    v_post = 1.0 / (1.0 / SIGMA_P ** 2 + 1.0 / sigma_obs ** 2)
    marg_var = SIGMA_P ** 2 + sigma_obs ** 2
    comps = []
    for k in range(2):
        m1 = v_post * (MU[k, 0].item() / SIGMA_P ** 2 + yv / sigma_obs ** 2)
        wp = W_PRIOR * np.exp(-0.5 * (yv - MU[k, 0].item()) ** 2 / marg_var)
        comps.append((wp, m1, MU[k, 1].item(), v_post, SIGMA_P ** 2))
    tot = sum(c[0] for c in comps)
    return [(c[0] / tot, c[1], c[2], c[3], c[4]) for c in comps]


def analytic_x2_stats(yv, sigma_obs):
    comps = analytic(yv, sigma_obs)
    e1 = sum(c[0] * c[2] for c in comps)
    e2 = sum(c[0] * (c[4] + c[2] ** 2) for c in comps)
    e3 = sum(c[0] * (c[2] ** 3 + 3 * c[2] * c[4]) for c in comps)
    e4 = sum(c[0] * (c[2] ** 4 + 6 * c[2] ** 2 * c[4] + 3 * c[4] ** 2) for c in comps)
    var = e2 - e1 ** 2
    mu3 = e3 - 3 * e1 * e2 + 2 * e1 ** 3
    mu4 = e4 - 4 * e1 * e3 + 6 * e1 ** 2 * e2 - 3 * e1 ** 4
    skew = mu3 / var ** 1.5
    kurt = mu4 / var ** 2
    xs = np.linspace(-12, 12, 800001)
    # 分量元组: (w, x1均值, x2均值, x1后验方差, x2后验方差)
    p = sum(w * norm.pdf(xs, m2v, np.sqrt(v2v)) for w, _, m2v, _, v2v in comps)
    sel = np.abs(xs) <= 1.25
    return var, skew, kurt, (skew ** 2 + 1) / kurt, np.trapezoid(p[sel], xs[sel])


def emp_stats(v):
    v = np.asarray(v, dtype=np.float64)
    d = v - v.mean()
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    skew = m3 / m2 ** 1.5
    kurt = m4 / m2 ** 2
    return m2, skew, kurt, (skew ** 2 + 1) / kurt, float(np.mean(np.abs(v) <= 1.25))


if __name__ == '__main__':
    n_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    print(f"n_iters={n_iters}\n")
    print(f"{'sigma':>6} {'结构':>5} | {'x1后验双峰间距':>14} | {'x2 var':>8} {'kurt':>7} {'BC':>7} {'gap':>7}   (解析)")
    for sigma_obs in [0.25, 1.0]:
        var_a, sk_a, ku_a, bc_a, gap_a = analytic_x2_stats(0.0, sigma_obs)
        v_post = 1.0 / (1.0 / SIGMA_P ** 2 + 1.0 / sigma_obs ** 2)
        sep = 2 * 2.5 * v_post / np.sqrt(v_post)
        print(f"\n--- sigma_obs = {sigma_obs} (x1 后验两峰间距 {sep:.2f} sigma) ---")
        print(f"{'':>6} {'解析':>5} | {'':>14} | {var_a:8.4f} {ku_a:7.4f} {bc_a:7.4f} {gap_a:7.4f}")
        for kind in ['nvp', 'maf']:
            t0 = time.time()
            model, losses = train(kind, sigma_obs, n_iters=n_iters)
            torch.manual_seed(999)
            with torch.no_grad():
                z = torch.randn(20000, 2, device=device)
                y = torch.zeros(20000, 1, device=device)
                xs = model.inverse(z, y).cpu().numpy()
            var, sk, ku, bc, gap = emp_stats(xs[:, 1])
            print(f"{sigma_obs:6.2f} {kind:>5} | {'':>14} | {var:8.4f} {ku:7.4f} {bc:7.4f} {gap:7.4f}"
                  f"   [NLL={losses[-1]:.4f}, {time.time()-t0:.0f}s]")
