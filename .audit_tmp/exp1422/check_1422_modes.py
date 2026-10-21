"""14.2-2 复验：cINN 的 p(x2|y) 是否被结构强制为单高斯。

对照两个结构（同数据流、同种子、同迭代数）：
  v1 = 当前交付脚本：层1 的仿射参数只看 y  -> x2|y 单高斯
  v2 = 层1 的仿射参数看 (y, x1)          -> x2|y 可为混合（标准 RealNVP 写法）
在 y=0（歧义观测）下比较 x2 边缘的双峰性指标与解析后验。
"""
import sys
import io
import numpy as np
import torch
import torch.nn as nn

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}  torch={torch.__version__}")

# ---------------- 问题设定（与交付脚本一致） ----------------
mu1 = torch.tensor([-2.5, -2.5])
mu2 = torch.tensor([2.5, 2.5])
SIGMA_P = 1.0
W1 = 0.5
A = torch.tensor([[1.0, 0.0]])
SIGMA_OBS = 0.25


def sample_prior(n, g):
    mode = torch.rand(n, 1, generator=g) < W1
    x = torch.empty(n, 2)
    n1 = int(mode.sum().item())
    x[mode.squeeze(1)] = mu1 + SIGMA_P * torch.randn(n1, 2, generator=g)
    x[~mode.squeeze(1)] = mu2 + SIGMA_P * torch.randn(n - n1, 2, generator=g)
    return x


def forward_obs(x, g):
    noise = SIGMA_OBS * torch.randn(x.shape[0], 1, generator=g)  # generator 在 CPU
    return x @ A.to(x.device).t() + noise.to(x.device)


# ---------------- 两种结构 ----------------
class Net2(nn.Module):
    """(y, u2) -> (s2, t2)"""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 2))

    def forward(self, y, u2):
        out = self.mlp(torch.cat([y, u2], dim=1))
        return 1.5 * torch.tanh(out[:, 0:1]), out[:, 1:2]


class CINN(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        in_dim = 1 if variant == 'v1' else 2      # v2: 额外把 x1 喂给层1
        self.net1 = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh(),
                                  nn.Linear(64, 64), nn.Tanh(),
                                  nn.Linear(64, 2))
        self.net2 = Net2()

    def _s1t1(self, y, x1):
        inp = y if self.variant == 'v1' else torch.cat([y, x1], dim=1)
        out = self.net1(inp)
        return 1.5 * torch.tanh(out[:, 0:1]), out[:, 1:2]

    def forward(self, x, y):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        s1, t1 = self._s1t1(y, x1)
        u1 = x1
        u2 = x2 * torch.exp(s1) + t1
        s2, t2 = self.net2(y, u2)
        z1 = u1 * torch.exp(s2) + t2
        z2 = u2
        return torch.cat([z1, z2], dim=1), s1.sum(1) + s2.sum(1)

    def inverse(self, z, y):
        z1, z2 = z[:, 0:1], z[:, 1:2]
        u2 = z2
        s2, t2 = self.net2(y, u2)
        u1 = (z1 - t2) * torch.exp(-s2)
        x1 = u1
        s1, t1 = self._s1t1(y, x1)
        x2 = (u2 - t1) * torch.exp(-s1)
        return torch.cat([x1, x2], dim=1)


def train(variant, n_iters=6000, batch=256, seed=0):
    torch.manual_seed(seed)
    model = CINN(variant).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed + 1)
    losses = []
    for it in range(n_iters):
        x = sample_prior(batch, g).to(device)
        y = forward_obs(x, g).to(device)
        z, logdet = model.forward(x, y)
        nll = (0.5 * (z ** 2).sum(1) + 0.5 * z.shape[1] * np.log(2 * np.pi) - logdet).mean()
        opt.zero_grad()
        nll.backward()
        opt.step()
        losses.append(nll.item())
    return model, np.array(losses)


# ---------------- 解析后验 ----------------
def analytic_components(yv):
    v_post = 1.0 / (1.0 / SIGMA_P ** 2 + 1.0 / SIGMA_OBS ** 2)
    marg_var = SIGMA_P ** 2 + SIGMA_OBS ** 2
    comps = []
    for mu, w in [(mu1, W1), (mu2, W1)]:
        m1 = v_post * (mu[0].item() / SIGMA_P ** 2 + yv / SIGMA_OBS ** 2)
        wp = w * np.exp(-0.5 * (yv - mu[0].item()) ** 2 / marg_var)
        comps.append((wp, m1, mu[1].item(), v_post, SIGMA_P ** 2))
    tot = sum(c[0] for c in comps)
    return [(c[0] / tot, c[1], c[2], c[3], c[4]) for c in comps]


def analytic_x2_moments(yv):
    comps = analytic_components(yv)
    m1 = sum(c[0] * c[2] for c in comps)
    m2 = sum(c[0] * (c[4] + c[2] ** 2) for c in comps) - m1 ** 2
    m3 = 0.0
    m4 = sum(c[0] * (c[2] ** 4 + 6 * c[2] ** 2 * c[4] + 3 * c[4] ** 2) for c in comps)
    # 四阶中心矩：用原始矩展开
    e1 = sum(c[0] * c[2] for c in comps)
    e2 = sum(c[0] * (c[4] + c[2] ** 2) for c in comps)
    e3 = sum(c[0] * (c[2] ** 3 + 3 * c[2] * c[4]) for c in comps)
    e4 = sum(c[0] * (c[2] ** 4 + 6 * c[2] ** 2 * c[4] + 3 * c[4] ** 2) for c in comps)
    mu = e1
    var = e2 - mu ** 2
    mu3 = e3 - 3 * mu * e2 + 2 * mu ** 3
    mu4 = e4 - 4 * mu * e3 + 6 * mu ** 2 * e2 - 3 * mu ** 4
    return mu, var, mu3, mu4


def bc_and_kurt(x2):
    x2 = np.asarray(x2, dtype=np.float64)
    mu = x2.mean()
    d = x2 - mu
    m2 = (d ** 2).mean()
    m3 = (d ** 3).mean()
    m4 = (d ** 4).mean()
    skew = m3 / m2 ** 1.5
    kurt = m4 / m2 ** 2           # 非超额峰度
    bc = (skew ** 2 + 1.0) / kurt  # Sarle 双峰性系数：<0.555 单峰，>0.555 双峰
    return bc, kurt, skew


def gap_mass(x2, thr=1.25):
    return float(np.mean(np.abs(x2) <= thr))


# ---------------- 主流程 ----------------
if __name__ == '__main__':
    n_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    models = {}
    for v in ['v1', 'v2']:
        print(f"\n训练 {v} ({n_iters} iters)...")
        t0 = __import__('time').time()
        m, lo = train(v, n_iters=n_iters)
        print(f"  耗时 {__import__('time').time()-t0:.1f}s  最终 NLL={lo[-1]:.4f}  前100均值={lo[:100].mean():.4f}")
        models[v] = (m, lo)

    for yv in [0.0, -2.5, 2.5]:
        print(f"\n===== y = {yv} =====")
        comps = analytic_components(yv)
        print("  解析后验分量:", [(round(c[0], 4), round(c[1], 4), c[2], round(c[3], 4), c[4]) for c in comps])
        mu, var, mu3, mu4 = analytic_x2_moments(yv)
        skew = mu3 / var ** 1.5
        kurt = mu4 / var ** 2
        bc = (skew ** 2 + 1) / kurt
        print(f"  解析 x2: mean={mu:+.4f} var={var:.4f} skew={skew:+.4f} kurt={kurt:.4f} BC={bc:.4f} "
              f"gap(|x2|<=1.25)={0.0:.4f}(下面用数值积分)")
        # 数值积分 gap 质量
        xs = np.linspace(-8, 8, 400001)
        from scipy.stats import norm
        p = np.zeros_like(xs)
        for w, m1, m2v, v1, v2v in comps:
            p += w * norm.pdf(xs, m2v, np.sqrt(v2v))
        gap = np.trapz(p[xs ** 2 <= 1.25 ** 2], xs[xs ** 2 <= 1.25 ** 2])
        print(f"  解析 x2 gap 质量 = {gap:.4f}")
        for v in ['v1', 'v2']:
            model = models[v][0]
            torch.manual_seed(999)
            with torch.no_grad():
                z = torch.randn(20000, 2, device=device)
                y = torch.full((20000, 1), yv, device=device)
                xs_s = model.inverse(z, y).cpu().numpy()
            b, k, s = bc_and_kurt(xs_s[:, 1])
            print(f"  {v}: mean=({xs_s[:,0].mean():+.4f},{xs_s[:,1].mean():+.4f}) "
                  f"std=({xs_s[:,0].std():.4f},{xs_s[:,1].std():.4f}) "
                  f"x2: kurt={k:.4f} skew={s:+.4f} BC={b:.4f} gap={gap_mass(xs_s[:,1]):.4f}")
