# -*- coding: utf-8 -*-
"""
实验7.7-1 采样加速：高阶指数积分器 vs 一致性模型
对应章节: 7.7（采样加速：从高阶求解器到一致性模型）
素材来源:
  - 7.4节概率流ODE, 7.5节数值离散化
  - Lu et al. (2022) DPM-Solver / DPM-Solver++（指数积分器精确解）
  - Song et al. (2023) Consistency Models（自一致性 + CD蒸馏）

实验内容:
  步骤1: 高阶指数积分器步数-质量曲线
         DDIM(1阶, = DPM-Solver-1) vs DPM-Solver++(2阶, multistep)
         用"解析完美网络"在2D高斯混合上演示: 阶数越高, 达到同样质量所需步数越少。
  步骤2: 一致性模型CD蒸馏——把整条轨迹压成一步
         训练一个极小的MLP一致性函数 f(x_t,t)->\hat x0 (自一致性 + 边界条件),
         对比: 朴素一步x0预测 / 一致性模型一步 / 一致性模型链式多步 / 多步DDIM。

运行前提: 纯NumPy即可, 无需PyTorch/预训练模型; 目标分布的解析得分闭式可得。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import io
import warnings
import logging

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验7.7-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)


# ============================================================
# 目标分布: 2D 高斯混合（解析 pdf + 解析得分）
#    p0(x) = 0.25 N(mu_i, 0.8^2 I),  mu_i 在四角相距较远(制造轨迹曲率)
# ============================================================
MU = 1.2 * np.array([[-4.0, -4.0], [4.0, -4.0], [-4.0, 4.0], [4.0, 4.0]])
W = np.ones(4) / 4.0
SIG = 0.8
DIM = 2


def p0_pdf(x):
    y = np.zeros(x.shape[0])
    for i in range(2):
        d = x - MU[i]
        y += W[i] * np.exp(-0.5 * np.sum(d * d, axis=-1) / (SIG ** 2)) \
             / ((2 * np.pi * SIG ** 2) ** (DIM / 2.0))
    return y


def sample_p0(n):
    k = np.random.choice(len(W), p=W, size=n)
    return np.random.randn(n, DIM) * SIG + MU[k]


def neg_loglik0(x):
    return -np.log(p0_pdf(x) + 1e-30)


# ============================================================
# VP-SDE 正向参数化（线性 beta 调度, 与书中 7.6 一致）
#   x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) eps
# ============================================================
B_MIN, B_MAX = 0.1, 20.0


def alpha_bar(t):
    return np.exp(-B_MIN * t - 0.5 * (B_MAX - B_MIN) * t * t)


def alpha_t(t):
    return np.sqrt(alpha_bar(t))


def sigma_t(t):
    return np.sqrt(1.0 - alpha_bar(t))


def lambda_t(t):
    ab = alpha_bar(t)
    return 0.5 * np.log(ab / (1.0 - ab) + 1e-30)


def _ab_from_lambda(lam):
    q = np.exp(-2.0 * lam)
    return 1.0 / (1.0 + q + 1e-30)


def _t_from_ab(ab):
    ts = np.empty_like(ab)
    for idx, target in enumerate(ab):
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if alpha_bar(mid) > target:
                lo = mid
            else:
                hi = mid
        ts[idx] = 0.5 * (lo + hi)
    return ts


def schedule_times(n_steps, t_start=1.0, t_end=0.01):
    """在 log-SNR λ 上均匀取点 -> 转回时间 t（递减序列, 首元素为 t_start）"""
    lam_s = lambda_t(t_start)
    lam_e = lambda_t(t_end)
    lam = np.linspace(lam_s, lam_e, n_steps + 1)
    return _t_from_ab(_ab_from_lambda(lam))


def _marginal(t):
    """正向边际 p_t(x) 的高斯混合参数 (解析)"""
    ab = alpha_bar(t)
    A = ab * SIG ** 2 + (1.0 - ab)          # 每个模态方差(A*I)
    alpha = np.sqrt(ab)
    return ab, A, alpha


def forward_score(x, t):
    """∇ log p_t(x)（解析得分，作为"完美老师"）"""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    ab, A, alpha = _marginal(t)
    li = np.zeros(x.shape[0])
    score = np.zeros_like(x)
    for i in range(len(W)):
        mean_i = alpha[:, None] * MU[i]         # 逐样本缩放的模态中心 (N,2)
        d = x - mean_i
        lik = W[i] * np.exp(-0.5 * np.sum(d * d, axis=-1) / A)
        li += lik
        score += lik[:, None] * (mean_i - x) / A[:, None]
    score /= (li[:, None] + 1e-30)
    return score


def dpm_data_prediction(x, t):
    """x0-预测: E[x0 | x_t]（解析高斯混合后验; 作为"完美 xθ 网络"）"""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    ab, A, alpha = _marginal(t)
    reg = alpha * SIG ** 2 / A
    li = np.zeros(x.shape[0])
    out = np.zeros_like(x)
    for i in range(len(W)):
        mean_i = alpha[:, None] * MU[i]
        d = x - mean_i
        lik = W[i] * np.exp(-0.5 * np.sum(d * d, axis=-1) / A)
        e_x0 = MU[i] + reg[:, None] * (x - mean_i)
        li += lik
        out += lik[:, None] * e_x0
    out /= (li[:, None] + 1e-30)
    return out


# ============================================================
# 确定性 ODE 求解器（消费 NFE = 模型求值次数）
# ============================================================
def ode_euler_order1(xT, ts):
    """DDIM / DPM-Solver-1（1阶指数积分器）。数据预测器 -> x0 形式, 等价于 DDIM。"""
    x = xT.copy()
    n = len(ts) - 1
    for i in range(1, n + 1):
        t_prev, t_cur = ts[i - 1], ts[i]     # t_prev 较大, t_cur 较小
        a_cur, a_new = alpha_t(t_prev), alpha_t(t_cur)
        s_cur, s_new = sigma_t(t_prev), sigma_t(t_cur)
        x0_pred = dpm_data_prediction(x, t_prev)
        eps = (x - a_cur * x0_pred) / (s_cur + 1e-30)
        x = a_new * x0_pred + s_new * eps    # 标准 DDIM: x = α_new x0 + σ_new ε
    return x


def ode_dpmpp2(xT, ts):
    """DPM-Solver++(2阶, multistep)。数据预测器 xθ。复用上一步求值, ~1 次求值/步。"""
    n = len(ts) - 1
    x = xT.copy()
    x0_prev2 = None
    x0_prev1 = dpm_data_prediction(x, ts[0])   # 在初始点求值
    prev_h = None
    for i in range(1, n + 1):
        t_prev, t_cur = ts[i - 1], ts[i]
        a_c = alpha_t(t_cur)
        s_prev, s_c = sigma_t(t_prev), sigma_t(t_cur)
        h = lambda_t(t_cur) - lambda_t(t_prev)
        if x0_prev2 is None:
            D = x0_prev1                       # 第1步退化为1阶
        else:
            ri = prev_h / h
            D = (1.0 + 1.0 / (2.0 * ri)) * x0_prev1 \
                - (1.0 / (2.0 * ri)) * x0_prev2
        x = (s_c / s_prev) * x - a_c * (np.exp(-h) - 1.0) * D
        prev_h = h
        x0_prev2 = x0_prev1
        x0_prev1 = dpm_data_prediction(x, t_cur)   # 在新点求值, 供下一步使用
    return x


def odefixed(xT, n):
    return ode_euler_order1(xT, schedule_times(n, 1.0, 0.01))


# ============================================================
# 一致性模型: 自一致性 + 边界条件 + 极小MLP（CD蒸馏）
#   fθ(x_t, t) = c_out(t)*Fθ(x_t, t) + c_skip(t)*x_t
#   c_skip(0)=1, c_out(0)=0 -> 边界 fθ(x,ε)=x
# ============================================================
class TinyMLP:
    """2层tanh MLP: 输入 [x(2), t_enc(1)] -> 输出(2)。纯numpy前向/反向。"""
    def __init__(self, h=64, seed=0):
        self.h = h
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(3, h) * 0.5
        self.b1 = np.zeros(h)
        self.W2 = rng.randn(h, 2) * 0.5
        self.b2 = np.zeros(2)

    def forward(self, z):
        a1 = np.tanh(z @ self.W1 + self.b1)
        out = a1 @ self.W2 + self.b2
        return out, a1

    def backward(self, z, a1, grad_out):
        gw = a1.T @ grad_out
        gb = np.sum(grad_out, axis=0)
        g_a1 = grad_out @ self.W2.T
        g_in = (1 - a1 * a1) * g_a1
        gw1 = z.T @ g_in
        gb1 = np.sum(g_in, axis=0)
        return gw, gb, gw1, gb1


def c_skip_out(t, lam=4.0):
    w = np.clip(t / 1.0, 0.0, 1.0)
    g = 1.0 - np.exp(-lam * w)
    return 1.0 - g, g


def cm_forward(model, x, t):
    """fθ(x_t, t) -> x0 预测。t: (N,) 标量"""
    t = np.asarray(t, dtype=float)
    t_enc = (t - 0.5) * 2.0
    z = np.concatenate([x, t_enc[:, None]], axis=-1)
    F, _ = model.forward(z)
    cs, co = c_skip_out(t)
    return cs[:, None] * x + co[:, None] * F


def step1():
    tier = [3, 5, 8, 15, 30, 60]
    n_eval = 10000
    xT = np.random.randn(n_eval, DIM)          # t=1 处 x≈标准高斯(alp_bar≈0)

    x_ref = odefixed(xT, 1000)
    nll_ref = np.mean(neg_loglik0(x_ref))

    nll_o1, nll_o2 = [], []
    for n in tier:
        ts = schedule_times(n, 1.0, 0.01)
        nll_o1.append(np.mean(neg_loglik0(ode_euler_order1(xT, ts))))
        nll_o2.append(np.mean(neg_loglik0(ode_dpmpp2(xT, ts))))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(tier, nll_o1, 'o-', label='DDIM (1阶, =DPM-Solver-1)')
    ax.plot(tier, nll_o2, 's-', label='DPM-Solver++ (2阶)')
    ax.axhline(nll_ref, ls='--', color='gray', label='DDIM 1000步 (上界)')
    ax.set_xscale('log')
    ax.set_xticks(tier)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('采样步数 N（NFE）')
    ax.set_ylabel('负对数似然 -log p0(x̂)（越低越好）')
    ax.set_title('高阶指数积分器：用更少的步数逼近高步数质量')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, '步骤1_求解器阶数对比.png'), dpi=150)
    plt.close(fig)

    print("步骤1 负对数似然(越低越好):")
    print("  步数      DDIM(1阶)   DPM-Solver++(2阶)")
    for i, n in enumerate(tier):
        print("  %4d      %8.4f    %8.4f" % (n, nll_o1[i], nll_o2[i]))
    print("  参考 DDIM 1000步: %.4f" % nll_ref)
    return xT


def step2(xT):
    # ---- 生成 CD 训练对: 老师=解析得分, 单步ODE连接相邻轨迹点 ----
    nint = 16
    t_grid = schedule_times(nint, 1.0, 0.01)[::-1]   # 升序, 从0.01到1.0
    t_grid = t_grid[1:]                              # 去掉近 0 端
    n_pairs = 20000
    idx = np.random.randint(0, len(t_grid) - 1, size=n_pairs)
    t1 = t_grid[idx]                                 # 较大时间(噪声侧)
    t2 = t_grid[idx + 1]                             # 较小时间(数据侧)
    x0 = sample_p0(n_pairs)
    ab1 = alpha_bar(t1)
    eps = np.random.randn(n_pairs, DIM)
    x_t1 = np.sqrt(ab1)[:, None] * x0 + np.sqrt(1 - ab1)[:, None] * eps
    score = forward_score(x_t1, t1)
    beta = B_MIN + t1 * (B_MAX - B_MIN)
    vel = -0.5 * beta[:, None] * (x_t1 + score)
    x_t2 = x_t1 + vel * (t2 - t1)[:, None]

    # ---- 训练极小 MLP (自一致性 + EMA 停梯度) ----
    model = TinyMLP(h=64, seed=1)
    model_ema = TinyMLP(h=64, seed=1)
    def copy_to(src, dst):
        dst.W1[:], dst.b1[:] = src.W1.copy(), src.b1.copy()
        dst.W2[:], dst.b2[:] = src.W2.copy(), src.b2.copy()
    copy_to(model, model_ema)

    lr, batch, iters = 1e-2, 512, 2500
    ema_decay = 0.999
    mW1 = np.zeros_like(model.W1); vW1 = np.zeros_like(model.W1)
    mb1 = np.zeros_like(model.b1); vb1 = np.zeros_like(model.b1)
    mW2 = np.zeros_like(model.W2); vW2 = np.zeros_like(model.W2)
    mb2 = np.zeros_like(model.b2); vb2 = np.zeros_like(model.b2)
    b1m, b2m, epsa = 0.9, 0.999, 1e-8

    for it in range(1, iters + 1):
        sel = np.random.choice(n_pairs, batch, replace=False)
        x_a, ta = x_t1[sel], t1[sel]
        x_b, tb = x_t2[sel], t2[sel]
        with_ema = cm_forward(model_ema, x_a, ta)        # 停梯度老师目标
        tb_enc = (tb - 0.5) * 2.0
        z = np.concatenate([x_b, tb_enc[:, None]], axis=-1)
        F, a1 = model.forward(z)
        cs_b, co_b = c_skip_out(tb)
        pred = cs_b[:, None] * x_b + co_b[:, None] * F
        diff = pred - with_ema
        loss = np.mean(np.sum(diff * diff, axis=-1))
        grad = 2.0 * diff * co_b[:, None]
        gw2, gb2p, gw1, gb1p = model.backward(z, a1, grad)
        for g, m_, v_, w in [(gw1, mW1, vW1, model.W1), (gb1p, mb1, vb1, model.b1),
                             (gw2, mW2, vW2, model.W2), (gb2p, mb2, vb2, model.b2)]:
            m_[:] = b1m * m_ + (1 - b1m) * g
            v_[:] = b2m * v_ + (1 - b2m) * (g * g)
            m_h = m_ / (1 - b1m ** it)
            v_h = v_ / (1 - b2m ** it)
            w[:] -= lr * m_h / (np.sqrt(v_h) + epsa)
        for w, wema in [(model.W1, model_ema.W1), (model.b1, model_ema.b1),
                        (model.W2, model_ema.W2), (model.b2, model_ema.b2)]:
            wema[:] = ema_decay * wema + (1 - ema_decay) * w
        if it % 500 == 0:
            print("  [CD %5d/%d] loss=%.5f" % (it, iters, loss))

    # ---- 生成对比 ----
    n = xT.shape[0]
    x_naive = dpm_data_prediction(xT, np.full(n, 1.0))   # 朴素一步x0预测
    x_cm1 = cm_forward(model_ema, xT, np.full(n, 1.0))   # 一致性模型一步

    def cm_multistep(k):
        ts = schedule_times(k, 1.0, 0.01)                # 递减
        x = xT.copy()
        for i in range(k):
            t_node = ts[i]
            x = cm_forward(model_ema, x, np.full(n, t_node))
            if i < k - 1:
                x = x + sigma_t(ts[i + 1]) * np.random.randn(n, DIM)
        return x
    x_cm4 = cm_multistep(4)
    x_ddim50 = odefixed(xT, 50)

    names = ['朴素一步(x0预测)', '一致性模型 1步', '一致性模型 4步', 'DDIM 50步']
    samples = [x_naive, x_cm1, x_cm4, x_ddim50]
    nll = [np.mean(neg_loglik0(s)) for s in samples]

    print("\n步骤2 负对数似然(越低越好):")
    for name, v in zip(names, nll):
        print("  %-18s %.4f" % (name, v))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    x0_gt = sample_p0(n)
    subs = {'真实分布': x0_gt, names[0]: samples[0], names[1]: samples[1], names[3]: samples[3]}
    for ax, (title, pts) in zip(axes, subs.items()):
        s = pts[:1500]
        ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.35)
        ax.set_title("%s\nNLL=%.3f" % (title, np.mean(neg_loglik0(s))))
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
    fig.suptitle('一致性模型把整条采样轨迹压成一步（2D高斯混合）')
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, '步骤2_一致性模型一步生成.png'), dpi=150)
    plt.close(fig)

    return names, nll


if __name__ == '__main__':
    print("===== 实验7.7-1: 采样加速：从高阶求解器到一致性模型 =====")
    xT = step1()
    names2, nll2 = step2(xT)
    print("\n全部结果已保存至:", SAVE_DIR)