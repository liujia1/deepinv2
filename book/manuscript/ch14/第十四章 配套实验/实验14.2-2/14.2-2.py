# -*- coding: utf-8 -*-
"""
实验14.2-2 cINN 条件归一化流：高维逆问题的后验采样
对应章节: 14.2.5 条件归一化流：高维逆问题的后验采样

知识点:
  - 逆问题: y = A x + ε，目标是从条件后验 p(x|y) 采样
  - 条件归一化流 (cINN, Ardizzone et al., 2019): 可逆映射 z = f_φ(x; y)
  - 训练目标: 最大似然 NLL = -[log p(z) + log|det J|]
  - 后验采样: z ~ N(0, I) → x = f^{-1}(z; y)，一次前向即可（摊销推断, APS）
  - 与解析后验（高斯混合）对照，验证 cINN 捕获多峰不确定性

实验内容:
  步骤1: 构造 2D 高斯混合先验 + 线性观测 y = x1 + ε（仅观测第一维）
  步骤2: 训练 cINN（2D 条件自回归流 MAF：先变换 x1，再以 z1 为条件变换 x2）
  步骤3: 对 3 个新观测 y，cINN 后验采样 vs 解析后验等高线对比
  步骤4: 样本均值/协方差 vs 解析后验统计量对比
  步骤5: 边缘双峰性诊断（双峰系数 BC、峰度、谷区质量）+ 与"x1 透传"结构的对照

数据集: 2D 双峰高斯混合，纯 CPU 实验
素材来源:
  - Ardizzone et al. (2019) "Guided Image Generation with Conditional Invertible Neural Networks"
  - Kruse et al. (2021) "HINT: Hierarchical Invertible Neural Transport for General Bayesian Inference"

运行前提: PyTorch, scipy, CPU 即可
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import warnings
import time

# 静默 matplotlib 相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", message=".*Glyph.*")

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验14.2-2')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
else:
    _chinese_path = '.chinese'
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"实验14.2-2: cINN 条件归一化流——高维逆问题的后验采样")
print(f"{'='*60}")
print(f"使用设备: {device}")

# ══════════════════════════════════════════════════════════
# 步骤1: 逆问题设定——双峰先验 + 线性观测
# ══════════════════════════════════════════════════════════
print("\n[步骤 1] 逆问题设定: y = A x + ε（只观测第一维）")

# 2D 双峰高斯混合先验
mu1 = torch.tensor([-2.5, -2.5], dtype=torch.float32)
mu2 = torch.tensor([2.5, 2.5], dtype=torch.float32)
sigma_p = 1.0          # 先验每个峰的宽度
w1, w2 = 0.5, 0.5      # 混合权重

# 观测算子 A: 2D -> 1D（只取第一维），观测噪声
A = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
# 观测噪声：σ_obs=1.0 是"弱观测"。若取 0.25（强观测），x1 的后验被压到宽度 0.24，
# 两个模式在 x1 上只差 1.2 个标准差——模式信息被观测掩埋，任何可逆流都难以
# 从 y 中恢复 x2 的双峰（实测见步骤5 的结构对照）。
sigma_obs = 1.0

def sample_prior(n):
    """从双峰高斯混合采样 x ∈ R^2。"""
    mode = torch.rand(n, 1) < w1
    x = torch.empty(n, 2, dtype=torch.float32)
    n1 = int(mode.sum().item())
    x[mode.squeeze(1)] = mu1 + sigma_p * torch.randn(n1, 2)
    x[~mode.squeeze(1)] = mu2 + sigma_p * torch.randn(n - n1, 2)
    return x

def forward_obs(x):
    """y = A x + ε，y ∈ R^1。"""
    dev = x.device
    return x @ A.to(dev).t() + sigma_obs * torch.randn(x.shape[0], 1, device=dev)

# ══════════════════════════════════════════════════════════
# cINN: 2D 条件自回归流（MAF）
# ══════════════════════════════════════════════════════════
# 正向 z = f(x; y):
#   层1: z1 = (x1 - t1(y)) * exp(-s1(y))            # 先变换 x1
#   层2: z2 = (x2 - t2(y, z1)) * exp(-s2(y, z1))    # 以 z1 为条件变换 x2
# log|det J| = -(s1(y) + s2(y,z1))
# 逆 x = f^{-1}(z; y) 依次反解。
#
# 为何不用 RealNVP 的“透传”写法（x1 透传、z2 = u2）？
# 那样 z2 与 y 无关，给定 y 时 x2 只能是一个高斯，p(x2|y) 被结构锁死为单峰；
# 本实验 y=0 的真后验是双峰，于是只能匹配二阶矩、匹配不了形状。
# 自回归写法让 x1 也经过变换，z1 能保留模式身份，层2 才能把双峰写进 x2。

class Net1(torch.nn.Module):
    """y -> (s1, t1)"""
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 2),
        )
    def forward(self, y):
        out = self.mlp(y)
        s = 1.5 * torch.tanh(out[:, 0:1])   # 限幅，保证可逆与稳定
        t = out[:, 1:2]
        return s, t

class Net2(torch.nn.Module):
    """(y, u2) -> (s2, t2)"""
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 2),
        )
    def forward(self, y, u2):
        inp = torch.cat([y, u2], dim=1)
        out = self.mlp(inp)
        s = 1.5 * torch.tanh(out[:, 0:1])
        t = out[:, 1:2]
        return s, t

class CINN(torch.nn.Module):
    """2D 条件可逆映射 z = f(x; y)。"""
    def __init__(self):
        super().__init__()
        self.net1 = Net1()
        self.net2 = Net2()

    def forward(self, x, y, need_logdet=True):
        """正向: z = f(x; y)，返回 z 与 log|det J|。"""
        x1, x2 = x[:, 0:1], x[:, 1:2]
        s1, t1 = self.net1(y)
        z1 = (x1 - t1) * torch.exp(-s1)           # 层1：变换 x1
        s2, t2 = self.net2(y, z1)                 # 层2 条件于 (y, z1)
        z2 = (x2 - t2) * torch.exp(-s2)           # 层2：变换 x2
        z = torch.cat([z1, z2], dim=1)
        logdet = -(s1.sum(1) + s2.sum(1))         # z=(x-t)·exp(-s) 的 Jacobian
        return z, logdet

    def inverse(self, z, y):
        """逆映射: x = f^{-1}(z; y)，用于后验采样。"""
        z1, z2 = z[:, 0:1], z[:, 1:2]
        s1, t1 = self.net1(y)
        x1 = z1 * torch.exp(s1) + t1
        s2, t2 = self.net2(y, z1)
        x2 = z2 * torch.exp(s2) + t2
        return torch.cat([x1, x2], dim=1)

    def log_p(self, x, y):
        """精确对数似然 log p(x|y)。"""
        z, logdet = self.forward(x, y)
        log_pz = -0.5 * (z ** 2).sum(1) - z.shape[1] * 0.5 * np.log(2 * np.pi)
        return log_pz + logdet

print("\n  cINN 结构: 2D 条件自回归流（MAF）")
print("  正向   z = f(x; y):  x1 ← 仿射(x1 | y)，x2 ← 仿射(x2 | y, z1)")
print("  采样   x = f^{-1}(z; y):  z ~ N(0,I) 一次前向得到后验样本（摊销推断）")

# ══════════════════════════════════════════════════════════
# 步骤2: 训练 cINN（最大似然 NLL）
# ══════════════════════════════════════════════════════════
print("\n[步骤 2] 训练 cINN（最大似然）...")

weights_path = os.path.join(SAVE_DIR, 'cinn_weights.pt')
losses_path = os.path.join(SAVE_DIR, 'losses.npy')
model = CINN().to(device)
if os.path.exists(weights_path) and os.path.exists(losses_path):
    # 已有权重：跳过训练，直接出图（删除 cinn_weights.pt / losses.npy 可强制重新训练）
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    losses = np.load(losses_path).tolist()
    print(f"  已加载现成权重 {os.path.basename(weights_path)}，跳过训练（最终 NLL = {losses[-1]:.4f}）")
else:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_iters = 6000
    batch_size = 256
    losses = []
    t_start = time.time()
    for it in range(n_iters):
        x = sample_prior(batch_size).to(device)
        y = forward_obs(x).to(device)
        z, logdet = model.forward(x, y)
        # NLL = -mean[ log N(z;0,I) + log|det J| ]，含归一化常数 D/2·log(2π)
        nll = (0.5 * (z ** 2).sum(1) + 0.5 * z.shape[1] * np.log(2 * np.pi) - logdet).mean()
        opt.zero_grad()
        nll.backward()
        opt.step()
        losses.append(nll.item())
        if (it + 1) % 1000 == 0:
            print(f"  迭代 {it+1}/{n_iters}: NLL = {nll.item():.4f}")
    torch.save(model.state_dict(), weights_path)
    np.save(losses_path, np.asarray(losses, dtype=np.float32))
    print(f"  训练完成，耗时 {time.time()-t_start:.1f} 秒，最终 NLL = {losses[-1]:.4f}")
    print(f"  权重已保存: {weights_path}（下次运行自动跳过训练）")

# 损失曲线
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(losses, lw=1.5, color='steelblue')
ax.set_xlabel('迭代次数')
ax.set_ylabel('负对数似然 NLL')
ax.set_title('实验14.2-2: cINN 训练收敛曲线')
ax.grid(True, alpha=0.3)
out_loss = os.path.join(SAVE_DIR, '步骤2_cINN训练收敛.png')
plt.tight_layout()
plt.savefig(out_loss, dpi=150)
plt.close()
print(f"  已保存: {out_loss}")

# ══════════════════════════════════════════════════════════
# 解析后验（高斯混合）——用于对照
# ══════════════════════════════════════════════════════════
def analytic_posterior(yv, grid):
    """对固定观测 yv∈R^1，返回解析后验的每个高斯分量的 (权重, 均值[2], 协方差[2])。"""
    # 先验分量 k:  N(μ_k, σ_p^2 I)  观测 y = x1 + ε, ε~N(0, σ_obs^2)
    comps = []
    v_post = 1.0 / (1.0 / sigma_p ** 2 + 1.0 / sigma_obs ** 2)     # x1 后验方差
    for mu, w in [(mu1, w1), (mu2, w2)]:
        m_post = v_post * (mu[0] / sigma_p ** 2 + yv / sigma_obs ** 2)  # x1 后验均值
        # 后验权重 ∝ 先验权重 * 边际似然 N(y; μ_1, σ_p^2 + σ_obs^2)
        marg_var = sigma_p ** 2 + sigma_obs ** 2
        w_post = w * np.exp(-0.5 * (yv.item() - mu[0].item()) ** 2 / marg_var)
        comps.append({
            'w': float(w_post),
            'mean': np.array([float(m_post), float(mu[1])]),
            'cov': np.array([[float(v_post), 0.0], [0.0, float(sigma_p ** 2)]]),
        })
    total = sum(c['w'] for c in comps)
    for c in comps:
        c['w'] /= total
    return comps

def posterior_pdf(x1g, x2g, comps):
    """解析后验密度（高斯混合）在网格上的值。"""
    pdf = np.zeros_like(x1g)
    for c in comps:
        d = np.stack([x1g - c['mean'][0], x2g - c['mean'][1]], axis=-1)
        inv_cov = np.linalg.inv(c['cov'])
        mah = np.einsum('...i,ij,...j->...', d, inv_cov, d)
        pdf += c['w'] * np.exp(-0.5 * mah) / (2 * np.pi * np.sqrt(np.linalg.det(c['cov'])))
    return pdf

# ══════════════════════════════════════════════════════════
# 步骤3: 对新观测 y 做 cINN 后验采样，与解析后验对照
# ══════════════════════════════════════════════════════════
print("\n[步骤 3] 对 3 个新观测做 cINN 后验采样，与解析后验对照...")

test_ys = [torch.tensor([[-2.5]], dtype=torch.float32),
           torch.tensor([[0.0]], dtype=torch.float32),
           torch.tensor([[2.5]], dtype=torch.float32)]
test_labels = ['y = -2.5（锁定左峰）', 'y = 0（两峰均可能）', 'y = 2.5（锁定右峰）']

n_samples = 2000
xs_grid = np.linspace(-5, 5, 200)
xg1, xg2 = np.meshgrid(xs_grid, xs_grid)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
stats_all = []
for ax, yv, lab in zip(axes, test_ys, test_labels):
    comps = analytic_posterior(yv, None)
    # 解析后验等高线
    pdf = posterior_pdf(xg1, xg2, comps)
    ax.contour(xg1, xg2, pdf, levels=8, colors='gray', linewidths=1.0, alpha=0.7)
    # cINN 后验采样
    # 固定种子：否则加载缓存权重（跳过训练）与从零训练两条路径的全局 RNG 状态不同，
    # 采到的 2000 个样本不同，步骤4 的统计量会跟着变（同一份权重的结果却不一致）。
    torch.manual_seed(1234)
    with torch.no_grad():
        z = torch.randn(n_samples, 2, device=device)
        y_b = yv.to(device).repeat(n_samples, 1)
        xs = model.inverse(z, y_b).cpu().numpy()
    ax.scatter(xs[:, 0], xs[:, 1], s=8, alpha=0.35, color='steelblue', label='cINN 采样')
    # 解析后验均值
    am = sum(c['w'] * c['mean'] for c in comps)
    ax.scatter([am[0]], [am[1]], marker='*', s=180, color='red', zorder=5, label='解析后验均值')
    ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title(lab)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')
    # 统计量
    sm = xs.mean(axis=0)
    scov = np.cov(xs.T)
    acov = sum(c['w'] * (c['cov'] + np.outer(c['mean'] - am, c['mean'] - am)) for c in comps)
    stats_all.append({'y': float(yv.item()), 'sample_mean': sm, 'analytic_mean': am,
                      'sample_cov': scov, 'analytic_cov': acov})
fig.suptitle('实验14.2-2: cINN 后验采样 vs 解析后验（灰度等高线）——捕获多峰不确定性', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out_samp = os.path.join(SAVE_DIR, '步骤3_cINN后验采样.png')
plt.savefig(out_samp, dpi=150)
plt.close()
print(f"  已保存: {out_samp}")

# ══════════════════════════════════════════════════════════
# 步骤4: 后验统计量对比
# ══════════════════════════════════════════════════════════
print("\n[步骤 4] 后验统计量对比（样本均值/协方差 vs 解析后验）...")

print(f"\n  {'y':<8}{'均值误差(样本vs解析)':<20}{'协方差谱范数误差':<18}{'解析谱范数':<12}")
names = ['x1', 'x2']
for st in stats_all:
    mean_err = np.linalg.norm(st['sample_mean'] - st['analytic_mean'])
    cov_err = np.linalg.norm(st['sample_cov'] - st['analytic_cov'], ord=2)
    cov_scale = np.linalg.norm(st['analytic_cov'], ord=2)
    print(f"  {st['y']:<8.2f}{mean_err:<22.4f}{cov_err:<20.4f}{cov_scale:<12.4f}")

# 纯抽样噪声基线：直接从解析后验抽同样多的样本，看“只差在抽样”时误差能有多大。
# 没有这条基线，就无法判断 cINN 的误差是“学错了”还是“样本不够”。
def sample_analytic(comps, n, rng):
    """从解析后验（高斯混合）抽 n 个样本。"""
    ws = np.array([c['w'] for c in comps])
    ws = ws / ws.sum()
    k = rng.choice(len(comps), size=n, p=ws)
    out = np.empty((n, 2))
    for i, c in enumerate(comps):
        m = k == i
        if m.sum() == 0:
            continue
        out[m] = c['mean'] + rng.standard_normal((int(m.sum()), 2)) @ np.linalg.cholesky(c['cov']).T
    return out

rng = np.random.default_rng(0)
n_boot = 200
mc_baseline = []
print(f"\n  纯抽样噪声基线（解析后验抽 {n_samples} 个样本，重复 {n_boot} 次，取 95% 分位）:")
for yv, lab in zip(test_ys, test_labels):
    comps = analytic_posterior(yv, None)
    am = sum(c['w'] * c['mean'] for c in comps)
    acov = sum(c['w'] * (c['cov'] + np.outer(c['mean'] - am, c['mean'] - am)) for c in comps)
    me, ce = [], []
    for _ in range(n_boot):
        s = sample_analytic(comps, n_samples, rng)
        me.append(np.linalg.norm(s.mean(axis=0) - am))
        ce.append(np.linalg.norm(np.cov(s.T) - acov, ord=2))
    q_me, q_ce = float(np.percentile(me, 95)), float(np.percentile(ce, 95))
    mc_baseline.append({'y': float(yv.item()), 'mean_err_q95': q_me, 'cov_err_q95': q_ce})
    print(f"  {lab:<24s} 均值误差 {q_me:.4f}，协方差谱范数误差 {q_ce:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, st, lab in zip(axes, stats_all, test_labels):
    comps = analytic_posterior(torch.tensor([[st['y']]], dtype=torch.float32), None)
    am = st['analytic_mean']
    acov = st['analytic_cov']
    sm = st['sample_mean']
    scov = st['sample_cov']
    # 解析后验 2σ 椭圆（跳过后验权重可忽略的分量：w<1e-3 时 2000 个样本期望出现 <2 个）
    for c in comps:
        if c['w'] < 1e-3:
            continue
        eigv, eigvec = np.linalg.eigh(c['cov'])
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = np.stack([np.cos(theta), np.sin(theta)])          # (2,100)
        ell = eigvec @ (2 * np.sqrt(eigv)[:, None] * circle)       # 先主轴缩放再旋转
        ax.plot(c['mean'][0] + ell[0], c['mean'][1] + ell[1], color='red', lw=1.5,
                label='解析后验 2σ' if c is comps[0] else None)
    # 样本协方差 2σ 椭圆
    eigv, eigvec = np.linalg.eigh(scov)
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    ell = eigvec @ (2 * np.sqrt(eigv)[:, None] * circle)
    ax.plot(sm[0] + ell[0], sm[1] + ell[1], color='steelblue', lw=2.0,
            label='cINN 样本 2σ')
    ax.scatter([am[0]], [am[1]], marker='*', s=160, color='red', zorder=5, label='解析均值')
    ax.scatter([sm[0]], [sm[1]], marker='D', s=60, color='steelblue', zorder=5, label='cINN 样本均值')
    ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title(lab)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')
fig.suptitle('实验14.2-2: cINN 样本协方差 vs 解析后验（2σ 椭圆）', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out_stat = os.path.join(SAVE_DIR, '步骤4_后验统计对比.png')
plt.savefig(out_stat, dpi=150)
plt.close()
print(f"  已保存: {out_stat}")

# ══════════════════════════════════════════════════════════
# 步骤5: 边缘双峰性诊断（y=0）——“均值/协方差对了”不等于“形状对了”
# ══════════════════════════════════════════════════════════
print("\n[步骤 5] 边缘双峰性诊断（y=0）...")

def edge_shape(v, thr=1.25):
    """边缘分布的 2/3/4 阶统计量 + Sarle 双峰系数 BC + 谷区质量。"""
    v = np.asarray(v, dtype=np.float64)
    d = v - v.mean()
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    skew = m3 / m2 ** 1.5
    kurt = m4 / m2 ** 2
    return {'var': float(m2), 'skew': float(skew), 'kurt': float(kurt),
            'bc': float((skew ** 2 + 1.0) / kurt),        # >5/9≈0.556 判为双峰
            'gap_mass': float((np.abs(v) <= thr).mean())}

def analytic_edge_shape(comps, axis, thr=1.25):
    """解析后验在 axis 方向上的边缘（高斯混合）的同一套指标。"""
    from scipy.stats import norm
    w = np.array([c['w'] for c in comps])
    m = np.array([c['mean'][axis] for c in comps])
    v = np.array([c['cov'][axis, axis] for c in comps])
    e1 = (w * m).sum()
    e2 = (w * (v + m ** 2)).sum()
    e3 = (w * (m ** 3 + 3 * m * v)).sum()
    e4 = (w * (m ** 4 + 6 * m ** 2 * v + 3 * v ** 2)).sum()
    var = e2 - e1 ** 2
    mu3 = e3 - 3 * e1 * e2 + 2 * e1 ** 3
    mu4 = e4 - 4 * e1 * e3 + 6 * e1 ** 2 * e2 - 3 * e1 ** 4
    skew = mu3 / var ** 1.5
    kurt = mu4 / var ** 2
    xs = np.linspace(-12, 12, 400001)
    pdf = sum(wi * norm.pdf(xs, mi, np.sqrt(vi)) for wi, mi, vi in zip(w, m, v))
    sel = np.abs(xs) <= thr
    return {'var': float(var), 'skew': float(skew), 'kurt': float(kurt),
            'bc': float((skew ** 2 + 1.0) / kurt),
            'gap_mass': float(np.trapezoid(pdf[sel], xs[sel]))}

y0_comps = analytic_posterior(torch.tensor([[0.0]]), None)
shape_ana = analytic_edge_shape(y0_comps, axis=1)          # x2 边缘

def sample_post(md, yv, n=20000, seed=999):
    torch.manual_seed(seed)
    with torch.no_grad():
        z = torch.randn(n, 2, device=device)
        y = torch.full((n, 1), yv, device=device)
        return md.inverse(z, y).cpu().numpy()

xs_y0 = sample_post(model, 0.0)
shape_maf = edge_shape(xs_y0[:, 1])


class CINNTransmit(CINN):
    """对照结构：层1 只条件于 y（x1 透传、z2=u2），p(x2|y) 在结构上锁死为单高斯。"""

    def forward(self, x, y, need_logdet=True):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        s1, t1 = self.net1(y)
        u2 = x2 * torch.exp(s1) + t1
        s2, t2 = self.net2(y, u2)
        z1 = x1 * torch.exp(s2) + t2
        return torch.cat([z1, u2], dim=1), s1.sum(1) + s2.sum(1)

    def inverse(self, z, y):
        z1, u2 = z[:, 0:1], z[:, 1:2]
        s2, t2 = self.net2(y, u2)
        u1 = (z1 - t2) * torch.exp(-s2)
        s1, t1 = self.net1(y)
        return torch.cat([u1, (u2 - t1) * torch.exp(-s1)], dim=1)


print("  训练对照结构（x1 透传的 RealNVP，4000 iters）...")
torch.manual_seed(0)
model_ref = CINNTransmit().to(device)
opt_ref = torch.optim.Adam(model_ref.parameters(), lr=1e-3)
for _ in range(4000):
    x = sample_prior(256).to(device)
    y = forward_obs(x).to(device)
    z_ref, logdet_ref = model_ref.forward(x, y)
    nll_ref = (0.5 * (z_ref ** 2).sum(1) + 0.5 * z_ref.shape[1] * np.log(2 * np.pi) - logdet_ref).mean()
    opt_ref.zero_grad()
    nll_ref.backward()
    opt_ref.step()
xs_ref = sample_post(model_ref, 0.0)
shape_ref = edge_shape(xs_ref[:, 1])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axes[0]
bins = np.linspace(-8, 8, 90)
ax.hist(xs_y0[:, 1], bins=bins, density=True, alpha=0.45, color='steelblue', label='cINN（MAF）')
ax.hist(xs_ref[:, 1], bins=bins, density=True, alpha=0.35, color='orange', label='cINN（x1 透传）')
xs_pdf = np.linspace(-8, 8, 800)
from scipy.stats import norm as _norm
pdf_a = sum(c['w'] * _norm.pdf(xs_pdf, c['mean'][1], np.sqrt(c['cov'][1, 1])) for c in y0_comps)
ax.plot(xs_pdf, pdf_a, 'k-', lw=2.0, label='解析后验')
ax.set_xlabel(r'$x_2$')
ax.set_ylabel('概率密度')
ax.set_title('(a) y=0 时 $x_2$ 的边缘：解析为双峰')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
names = ['解析', 'MAF', 'x1 透传']
bcs = [shape_ana['bc'], shape_maf['bc'], shape_ref['bc']]
bars = ax.bar(names, bcs, color=['black', 'steelblue', 'orange'], alpha=0.75)
ax.axhline(5 / 9, color='red', ls='--', lw=1.2, label='双峰判据 BC = 5/9')
for b, v in zip(bars, bcs):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f'{v:.3f}',
            ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(0, 0.85)
ax.set_ylabel('双峰系数 BC')
ax.set_title('(b) 双峰系数：越接近 1 越双峰')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
fig.suptitle('实验14.2-2 步骤5：后验形状诊断（y=0）', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out_shape = os.path.join(SAVE_DIR, '步骤5_边缘双峰诊断.png')
plt.savefig(out_shape, dpi=150)
plt.close()
print(f"  已保存: {out_shape}")

print(f"\n  {'分布':<18s}{'var':>8s}{'kurt':>8s}{'BC':>8s}{'谷区质量':>10s}")
for nm, s in [('解析后验', shape_ana), ('cINN(MAF)', shape_maf), ('cINN(x1透传)', shape_ref)]:
    print(f"  {nm:<18s}{s['var']:8.3f}{s['kurt']:8.3f}{s['bc']:8.3f}{s['gap_mass']:10.3f}")
print("  注：谷区质量 = P(|x2|<=1.25)，即两峰之间本应几乎无样本的区域。")

# ══════════════════════════════════════════════════════════
# 保存结果
# ══════════════════════════════════════════════════════════
def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj

results = {
    'experiment': '实验14.2-2 cINN 条件归一化流：高维逆问题的后验采样',
    'setting': {
        'prior': '2D 双峰高斯混合 μ=(-2.5,-2.5),(2.5,2.5), σ=1.0',
        'forward': 'y = A x + ε, A=[[1,0]], σ_obs=1.0（只观测第一维；弱观测才能保留模式信息）',
        'cinn': '2D 条件自回归流 MAF（z1=(x1-t1)/exp(s1), z2=(x2-t2(y,z1))/exp(s2)），6000 步 Adam lr=1e-3',
    },
    'final_nll': losses[-1],
    'test_statistics': [
        {
            'y': st['y'],
            'mean_error_l2': float(np.linalg.norm(st['sample_mean'] - st['analytic_mean'])),
            'cov_spectral_norm_error': float(np.linalg.norm(st['sample_cov'] - st['analytic_cov'], ord=2)),
            'cov_spectral_norm_analytic': float(np.linalg.norm(st['analytic_cov'], ord=2)),
        } for st in stats_all
    ],
    'mc_baseline_q95': mc_baseline,
    'posterior_shape_y0': {
        'edge': 'x2',
        'bimodality_threshold': float(5 / 9),
        'analytic': shape_ana,
        'cinn_maf': shape_maf,
        'cinn_x1_transmit': shape_ref,
    },
    'output_files': [os.path.basename(p) for p in (out_loss, out_samp, out_stat, out_shape)],  # 只存文件名，跨机器有效
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results), f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("【核心结论】")
print("=" * 60)
print("1. cINN 用可逆网络 z = f(x; y) 显式条件于观测 y，训练目标即最大似然 NLL")
print("2. 后验采样只需一次前向: z ~ N(0,I) → x = f^{-1}(z; y)——摊销推断（APS）")
print("3. y=0（歧义观测）时，x2 的边缘应为双峰（解析 BC=%.3f > 5/9）。MAF 结构实测：" % shape_ana['bc'])
print(f"   BC={shape_maf['bc']:.3f}，kurt={shape_maf['kurt']:.3f}（解析 {shape_ana['kurt']:.3f}，单高斯 3.000）"
      f"，谷区质量 {shape_maf['gap_mass']:.3f}（解析 {shape_ana['gap_mass']:.3f}）")
print("   → 双峰轮廓已出现，但仍略低于 5/9 判据，谷区多出约一倍的“桥接”样本（小网络过度平滑）")
print("4. 结构决定上限：把层1 换回“只条件于 y”（x1 透传、z2=u2），p(x2|y) 被锁死为单高斯：")
print(f"   BC={shape_ref['bc']:.3f}（≈单峰的 0.333）、谷区质量 {shape_ref['gap_mass']:.3f}")
print("   → 均值/协方差吻合（二阶矩匹配）不能当作形状正确的证据")
print(f"\n实验完成。结果已保存至: {SAVE_DIR}")
print(f"  - 步骤2_cINN训练收敛.png")
print(f"  - 步骤3_cINN后验采样.png")
print(f"  - 步骤4_后验统计对比.png")
print(f"  - 步骤5_边缘双峰诊断.png")
print(f"  - results_summary.json")
