# -*- coding: utf-8 -*-
"""
实验14.3-4 得分-速度-去噪器三角转换 + SDE扩展(σ_t)数值验证
对应章节: 14.3.4 三角转换框 / 14.3.7 SDE扩展技巧

知识点:
  - 三角转换（14.3.4框）: 对高斯路径 p_t(x|x0)=N(α_t x0, β_t²I), 得分/向量场/去噪器
    是同一个量(后验均值)的三种线性重参数化, 三条边际转换边全部闭式:
      得分→速度:  v = a_t·s + b_t·x,  a_t = β²(α̇/α) − β̇β,  b_t = α̇/α
      速度→去噪器: D = (β·v − β̇·x)/(α̇β − αβ̇)
      去噪器→得分: s = (α·D − x)/β²   (Tweedie)
  - SDE扩展（14.3.7）: 训练后任意 σ_t≥0, dX=[v + (σ²/2)∇log p]dt + σdW 边际不变
  - 工程含义: 只训练 v_θ 一个网络, 三角公式免费换算出 score, 即可跑任意 σ 的随机采样

实验内容:
  - 2D 两高斯混合(各向异性): 边际 p_t 与 score*/D*/v* 全部有解析解(共轭计算)
  - 调度 α_t=t, β_t=1−t(直线插值), 独立耦合 CFM 训练 v_θ
  - 三角验证三重: ①闭环 v_θ→D_θ→s_θ→v'_θ 误差(浮点级) ②三对象 vs 解析真值
    ③解析 score* 用有限差分独立验证(抓住推导错误的最后一道防线)
  - σ 扫描 {0,0.5,1,2,4} × 步数 {100,5} × 5 采样种子: 切片W2 均值±std——检验"多步对 σ 鲁棒"
    与"少步中间 σ 红利"（14.3.7 的边际不变仅对精确 score 成立, 模型 score 误差会被 σ²/2 放大）
  - 解析对照: 精确 v*/s* 跑同款采样器, 分离"粗步离散化效应"（对照中仍在的红利）
    与"模型误差放大"（模型版多出的劣化）——结论写在输出里, 证据也在代码里

本实验不需要GPU，CPU 约 1~2 分钟可跑完。
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import warnings

# 静默 matplotlib 相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", message=".*Glyph.*")
warnings.filterwarnings("ignore", message=".*cmap.*")

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验14.3-4')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(_chinese_path, exist_ok=True)

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
import torch.nn as nn

# 精度策略: 解析量全部用 numpy(天然 float64) 计算, 保证 t->1 时精度;
# 网络训练/采样用 torch 默认 float32, 与解析量对比前再统一转换
# （见下方探针处的 astype(np.float64) 与采样处的 .double()）。
# 注意不要 torch.set_default_dtype(torch.float64)——那会让训练输入变成 float64,
# 与 .float() 的模型 dtype 不匹配而直接报错。

# ============================================================
# 步骤0：数据分布与解析真值（两高斯混合, 全对角协方差 -> 逐坐标向量化）
# ============================================================
# p_data = 0.5·N(μ1, Σ1) + 0.5·N(μ2, Σ2), 各向异性(两个方向方差不同)
MU1 = np.array([-2.0, -2.0]); SIG1 = np.array([0.6**2, 0.9**2])   # 左下: 窄x宽y
MU2 = np.array([ 2.0,  2.0]); SIG2 = np.array([0.9**2, 0.6**2])   # 右上: 宽x窄y
WEIGHTS = np.array([0.5, 0.5])
MUS = np.stack([MU1, MU2])            # (K, 2)
SIGS = np.stack([SIG1, SIG2])         # (K, 2) 对角协方差

# 调度（14.3.4 框的统一记法）: α_t = t, β_t = 1-t（直线插值; β_t>0 对 t<1）
ALPHA = lambda t: t
BETA = lambda t: 1.0 - t

def gauss_logpdf_diag(x, mean, var):
    """对角协方差高斯的 log N(x; mean, diag(var)), x:(N,2), mean/var:(N,2) 或标量广播"""
    return -0.5 * np.sum((x - mean) ** 2 / var + np.log(var) + np.log(2 * np.pi), axis=-1)

def mixture_marginal(x, t):
    """边际 p_t(x) = Σ_k w_k N(x; α_t μ_k, β_t² I + α_t² Σ_k)（14.3.4 情形二的推广）

    返回 (dens (N,), comp_logpdf (N,K))——分量 log 密度同时返回, 供后验权重复用
    """
    a_t, b_t2 = ALPHA(t), BETA(t) ** 2
    comp = np.stack([WEIGHTS[k] * np.exp(gauss_logpdf_diag(
        x, a_t * MUS[k][None, :], b_t2 + a_t ** 2 * SIGS[k][None, :])) for k in range(2)], axis=-1)
    return comp.sum(-1), comp

def mixture_posterior_weights(x, t, comp_logpdf=None):
    """分量后验权重 r_k(x) = w_k·N(x; t μ_k, C_k) / Σ_j（x 的函数, 与后验 m_k 的权重不同!）"""
    if comp_logpdf is None:
        _, comp = mixture_marginal(x, t)
    else:
        comp = comp_logpdf
    return comp / comp.sum(-1, keepdims=True)

def mixture_score(x, t):
    """解析边际得分 s*(x) = ∇log p_t = Σ_k r_k·(α_t μ_k − x)/C_k(x)"""
    a_t, b_t2 = ALPHA(t), BETA(t) ** 2
    C_k = b_t2 + a_t ** 2 * SIGS            # (K, 2) 两个分量的边际方差
    _, comp = mixture_marginal(x, t)
    r = comp / comp.sum(-1, keepdims=True)  # (N, K) 后验权重
    contrib = (a_t * MUS[None, :, :] - x[:, None, :]) / C_k[None, :, :]  # (N, K, 2)
    return (r[:, :, None] * contrib).sum(1)

def mixture_denoiser(x, t):
    """解析去噪器 D*(x) = E[x0 | x_t = x]（高斯共轭后验的精确解）

    观测模型: x_t = α_t x0 + β_t ζ, ζ~N(0,I) -> 似然关于 x0 的精度 = α_t²/β_t²
    分量后验: N(m_k, S_k), S_k = (Σ_k^{-1} + (α_t²/β_t²) I)^{-1},
              m_k = S_k(Σ_k^{-1} μ_k + (α_t/β_t²) x)
    后验权重: w̃_k ∝ w_k·N(x; α_t μ_k, β_t² I + α_t² Σ_k)（边际似然, 同 r_k 的分子）
    """
    a_t, b_t2 = ALPHA(t), BETA(t) ** 2
    lam = a_t ** 2 / b_t2                                  # 似然精度（逐坐标同值）
    S_k = 1.0 / (1.0 / SIGS + lam)                         # (K, 2) 分量后验方差
    m_k = S_k * (MUS / SIGS + (a_t / b_t2) * x[:, None, :])  # (N, K, 2)
    wt = mixture_posterior_weights(x, t)                   # (N, K)
    return (wt[:, :, None] * m_k).sum(1)

def mixture_velocity(x, t):
    """解析边际流 v*(x) = (E[x0|x_t] − x_t)/(1−t)

    推导: 直线插值 x_t=(1−t)z+t·x0 => z=(x_t−t·x0)/(1−t) => x0−z=(x0−x_t)/(1−t)
          取后验期望即得（独立耦合下 E[x0−z|x_t] 正是 CFM 回归的最优解, 与14.3.3一致）
    """
    return (mixture_denoiser(x, t) - x) / BETA(t)

def score_from_velocity(v, x, t):
    """三角反解: 由 v_θ 换算 s_θ = (t·v − x)/(1−t)

    由 v = a_t·s + b_t·x（a_t=(1−t)/t, b_t=1/t）反解得 s=(t·v−x)/(1−t)。
    该形式在 t=0 处也良定义: s(0) = −x, 恰是 p_0=N(0,I) 的精确得分——
    端点无需任何特殊处理（工程实现比逐项代入 a_t/b_t 更稳）。
    """
    return (ALPHA(t) * v - x) / BETA(t)

# ---------- 自检1：解析 score* vs 有限差分（独立验证解析推导） ----------
print("[自检1] 解析得分 vs 有限差分（互相独立的两种计算, 必须一致）:")
fd_errs = []
for t_chk in [0.2, 0.5, 0.8, 0.95]:
    rng = np.random.RandomState(7)
    x_chk = rng.uniform(-3, 3, size=(64, 2))
    eps = 1e-5
    s_fd = np.stack([(np.log(mixture_marginal(x_chk + eps * np.eye(2)[i], t_chk)[0])
                      - np.log(mixture_marginal(x_chk - eps * np.eye(2)[i], t_chk)[0]))
                     / (2 * eps) for i in range(2)], axis=-1)
    s_an = mixture_score(x_chk, t_chk)
    err = np.abs(s_fd - s_an).max()
    fd_errs.append(err)
    print(f"    t={t_chk:.2f}: max|s_fd − s*| = {err:.2e}")
assert max(fd_errs) < 1e-4, "解析得分与有限差分不一致——推导有误!"
print("    -> 全部一致, 解析得分可信")

# ---------- 自检2：三角闭环（解析量之间必须精确闭合） ----------
print("[自检2] 三角公式闭环（解析真值自乘自除）:")
for t_chk in [0.2, 0.5, 0.8]:
    x_chk = np.random.RandomState(7).uniform(-3, 3, size=(32, 2))
    s_an = mixture_score(x_chk, t_chk)
    a_t = BETA(t_chk) ** 2 * (1.0 / ALPHA(t_chk)) - (-1.0) * BETA(t_chk)  # β²·(α̇/α) − β̇·β
    b_t = 1.0 / ALPHA(t_chk)                                              # α̇/α
    v_via_s = a_t * s_an + b_t * x_chk            # 得分→速度
    v_an = mixture_velocity(x_chk, t_chk)
    e1 = np.abs(v_via_s - v_an).max()
    D_via_v = (BETA(t_chk) * v_an - (-1.0) * x_chk) / (1.0 * BETA(t_chk) - ALPHA(t_chk) * (-1.0))
    D_an = mixture_denoiser(x_chk, t_chk)
    e2 = np.abs(D_via_v - D_an).max()
    s_via_D = (ALPHA(t_chk) * D_an - x_chk) / BETA(t_chk) ** 2
    e3 = np.abs(s_via_D - s_an).max()
    print(f"    t={t_chk:.2f}: 得分→速度误差 {e1:.2e} | 速度→去噪器误差 {e2:.2e} | 去噪器→得分误差 {e3:.2e}")
    assert max(e1, e2, e3) < 1e-8, "三角公式未闭合——实现有误!"
print("    -> 三条转换边在解析真值上精确闭合（浮点级）")


# ============================================================
# 步骤1：CFM 训练——只训练 v_θ 一个网络（直线插值, 独立耦合）
# ============================================================
class VectorFieldNet(nn.Module):
    """2D 向量场网络 v_θ(x_t, t)（与实验14.3-2 同构）"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_t, t):
        inp = torch.cat([x_t, t], dim=-1)
        return self.net(inp)

N_PARAMS = sum(p.numel() for p in VectorFieldNet().parameters())
print(f"\n[模型] v_θ 向量场网络: 参数量 {N_PARAMS:,}")

def train_cfm(n_epochs=2000, batch_size=256, lr=1e-3, eval_every=25, eval_n=4096):
    """条件 Flow Matching 训练（直线插值 x_t=(1−t)z+t·x0, 条件场 v=x0−z）

    训练/网络内部用 float32 提速; 解析量比较时统一升回 float64。
    收敛曲线用固定评估 batch（4096 样本, 独立生成器, 不扰动训练的随机数流）每 25 epoch
    评估一次——单 batch 训练损失的锯齿只是蒙特卡洛噪声, 不反映真实收敛。
    """
    model = VectorFieldNet().float()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 固定评估 batch（显式生成器 -> 全局随机数流不变, 训练结果与不评估时逐位一致）
    g_eval = torch.Generator().manual_seed(123)
    comp_e = torch.multinomial(torch.tensor(WEIGHTS, dtype=torch.float32),
                               eval_n, replacement=True, generator=g_eval)
    z_e = torch.randn(eval_n, 2, generator=g_eval)
    x0_e = torch.tensor(MUS, dtype=torch.float32)[comp_e] \
        + torch.randn(eval_n, 2, generator=g_eval) \
        * torch.tensor(SIGS, dtype=torch.float32)[comp_e].sqrt()
    t_e = torch.rand(eval_n, 1, generator=g_eval)
    v_e = x0_e - z_e
    eval_curve = []
    model.train()
    for epoch in range(1, n_epochs + 1):
        # 1. 采噪声 z~N(0,I) 与数据 x0~p_data（独立耦合, 14.3.3 默认配置）
        z = torch.randn(batch_size, 2)
        comp = torch.multinomial(torch.tensor(WEIGHTS, dtype=torch.float32),
                                 batch_size, replacement=True)
        mu_b = torch.tensor(MUS, dtype=torch.float32)[comp]
        sg_b = torch.tensor(SIGS, dtype=torch.float32)[comp]
        x0 = mu_b + torch.randn(batch_size, 2) * sg_b.sqrt()

        # 2. 采 t~U(0,1), 直线插值加噪, 条件向量场 = x0 − z
        t = torch.rand(batch_size, 1)
        x_t = (1 - t) * z + t * x0
        v_target = x0 - z

        # 3. MSE 回归（CFM 损失的蒙特卡洛估计）
        v_pred = model(x_t, t)
        loss = nn.functional.mse_loss(v_pred, v_target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % eval_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                eval_loss = nn.functional.mse_loss(
                    model((1 - t_e) * z_e + t_e * x0_e, t_e), v_e).item()
            model.train()
            eval_curve.append((epoch, eval_loss))
        if epoch % 500 == 0:
            print(f"  epoch {epoch}/{n_epochs}  评估 loss = {eval_curve[-1][1]:.6f}")
    return model, eval_curve

print("\n训练 CFM（直线插值, 独立耦合）...")
model_v, losses_v = train_cfm()
model_v.eval()


# ============================================================
# 步骤2：三角验证——网络闭环误差 + 三对象 vs 解析真值
# ============================================================
print("\n步骤2: 探针网格上验证三角转换（5 个时间 × 15×15 空间网格）...")
T_PROBES = [0.1, 0.3, 0.5, 0.7, 0.9]
_g1d = np.linspace(-4, 4, 15)
_probe_t, _probe_x = [], []
for tp in T_PROBES:
    X, Y = np.meshgrid(_g1d, _g1d)
    _probe_t.append(np.full(X.size, tp))
    _probe_x.append(np.stack([X.ravel(), Y.ravel()], axis=1))
probe_t = np.concatenate(_probe_t)      # (M,)
probe_x = np.concatenate(_probe_x)      # (M, 2)

with torch.no_grad():
    v_net = model_v(torch.tensor(probe_x, dtype=torch.float32),
                    torch.tensor(probe_t, dtype=torch.float32)[:, None]).numpy().astype(np.float64)
D_net = BETA(probe_t)[:, None] * v_net + probe_x                 # 速度→去噪器（分母=1）
s_net = (ALPHA(probe_t)[:, None] * D_net - probe_x) / (BETA(probe_t) ** 2)[:, None]  # 去噪器→得分
v_back = (BETA(probe_t) / ALPHA(probe_t))[:, None] * s_net + (1.0 / ALPHA(probe_t))[:, None] * probe_x
loop_err = np.linalg.norm(v_back - v_net, axis=1)                # 闭环: v→D→s→v'

# 解析函数按"标量 t"契约实现（公式与正文一致、广播最直观）;
# probe_t 是 1125 个逐样本时间, 需按时间分块后逐块以标量 t 计算再拼接
# （若把数组 t 直接传入, a_t**2 * SIGS 之类会因 (N,)*(K,2) 无法广播而崩溃）
s_ana = np.concatenate([mixture_score(probe_x[probe_t == tp], tp) for tp in T_PROBES])
D_ana = np.concatenate([mixture_denoiser(probe_x[probe_t == tp], tp) for tp in T_PROBES])
v_ana = np.concatenate([mixture_velocity(probe_x[probe_t == tp], tp) for tp in T_PROBES])
err_v = np.linalg.norm(v_net - v_ana, axis=1)
err_D = np.linalg.norm(D_net - D_ana, axis=1)
err_s = np.linalg.norm(s_net - s_ana, axis=1)
print(f"  闭环误差 v→D→s→v':    max {loop_err.max():.2e}  （浮点级 -> 公式实现自洽）")
print(f"  网络速度 vs 解析 v*:   mean {err_v.mean():.4f} / max {err_v.max():.4f}")
print(f"  换算去噪器 vs 解析 D*: mean {err_D.mean():.4f} / max {err_D.max():.4f}")
print(f"  换算得分   vs 解析 s*: mean {err_s.mean():.4f} / max {err_s.max():.4f}")
print("  -> 学一个 v_θ, 三角公式免费得到 D_θ 与 s_θ; 三者误差按三角系数精确线性缩放(代数恒等), 不必同量级")

per_t_err = {tp: {'v': float(err_v[probe_t == tp].mean()),
                  'D': float(err_D[probe_t == tp].mean()),
                  's': float(err_s[probe_t == tp].mean())} for tp in T_PROBES}

# 图1: 三面板散点（解析真值 vs 网络换算值, 颜色=时间）
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
sc_specs = [('a', v_ana, v_net, '速度 v', err_v), ('b', D_ana, D_net, '去噪器 D', err_D),
            ('c', s_ana, s_net, '得分 s', err_s)]
for ax, (tag, ana, net, name, err) in zip(axes, sc_specs):
    sc = ax.scatter(ana[:, 0], net[:, 0], c=probe_t, cmap='viridis', s=12, alpha=0.75)
    ax.scatter(ana[:, 1], net[:, 1], c=probe_t, cmap='viridis', s=12, alpha=0.75)
    lim = max(abs(ana).max(), abs(net).max())
    if name == '得分 s':
        lim = 50.0  # s 在 t->1 时发散, 截窗显示主体区域（远离端点的发散尾注明在文字里）
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.8, alpha=0.6, label='y = x')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(f'解析真值 {name}*', fontsize=11)
    ax.set_ylabel(f'网络换算 ${name[-1]}_{{\\theta}}$', fontsize=11)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title(f'({tag}) {name}: mean误差 {err.mean():.3f} / max {err.max():.3f}', fontsize=11)
plt.colorbar(sc, ax=axes, label='时间 $t$', shrink=0.85)
plt.suptitle('实验14.3-4 步骤2: 得分-速度-去噪器三角——一个网络, 三个等价对象（两高斯混合, $\\alpha=t$, $\\beta=1-t$）',
             fontsize=12.5, y=1.03)
plt.savefig(os.path.join(SAVE_DIR, '步骤2_三角验证.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"图已保存: {os.path.join(SAVE_DIR, '步骤2_三角验证.png')}")


# ============================================================
# 步骤3：SDE 扩展 σ_t 扫描——同一 v_θ 配不同 σ（14.3.7 的数值版）
# ============================================================
# 采样 SDE: dX = [v_θ + (σ²/2)·s_θ] dt + σ dW,  s_θ 由三角公式从 v_θ 免费换算
# 欧拉-马里亚马: X_{t+h} = X + h·[v + σ²/2·s] + σ·√h·ξ
# σ=0 时退化为概率流 ODE 的欧拉步——同一个实现覆盖"随机性旋钮"全程。
def sliced_w2(samples_a, samples_b, n_proj=32, seed=42):
    """切片 W2 距离（随机投影近似, 教学标准做法）:
    W2²(a,b) ≈ E_u[W2²(u·a, u·b)], 1D W2 用排序后的逐点差分计算"""
    if not (np.isfinite(samples_a).all() and np.isfinite(samples_b).all()):
        return float('nan')   # 采样发散（如解析对照 5步σ=4）——如实记 NaN
    g = torch.Generator().manual_seed(seed)
    a = torch.tensor(samples_a, dtype=torch.float64)
    b = torch.tensor(samples_b, dtype=torch.float64)
    tot = 0.0
    for _ in range(n_proj):
        u = torch.randn(2, generator=g, dtype=torch.float64)
        u = u / u.norm()
        pa = (a @ u).sort().values
        pb = (b @ u).sort().values
        tot += float(((pa - pb) ** 2).mean())
    return np.sqrt(tot / n_proj)

def sample_sde_sigma(model, sigma, n_steps, n_samples, seed):
    """欧拉-马里亚马采样（σ=0 即 ODE 欧拉步）; s_θ 用三角公式从 v_θ 换算"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, 2, generator=g)
    h = 1.0 / n_steps
    with torch.no_grad():
        for n in range(n_steps):
            t_n = n / n_steps                                  # 时间网格 [0, 1-h]
            v = model(x.float(), torch.full((n_samples, 1), t_n)).double()
            s = score_from_velocity(v.numpy(), x.numpy(), t_n)  # 免费的 score
            drift = v.numpy() + 0.5 * sigma ** 2 * s
            noise = np.sqrt(h) * sigma * np.random.default_rng(seed * 1000 + n).standard_normal(x.shape)
            x = torch.tensor(x.numpy() + h * drift + noise)
    return x.numpy()

def sample_sde_analytic(sigma, n_steps, n_samples, seed):
    """解析对照采样器: 用精确 v* 与 s* 替换网络, 其余与 sample_sde_sigma 完全同款。

    用途: 把 σ 扫描的总误差分解为"粗步离散化效应"（解析对照中仍存在的部分）
    与"模型 score 误差被 σ²/2 放大"（模型版超出解析对照的部分）。
    """
    rng0 = np.random.default_rng(seed)
    x = rng0.standard_normal((n_samples, 2))
    h = 1.0 / n_steps
    with np.errstate(all='ignore'):   # σ 大+步少时会数值发散, 如实让它溢出为 NaN
        for n in range(n_steps):
            t_n = n / n_steps
            drift = mixture_velocity(x, t_n) + 0.5 * sigma ** 2 * mixture_score(x, t_n)
            noise = np.sqrt(h) * sigma * np.random.default_rng(seed * 1000 + n).standard_normal(x.shape)
            x = x + h * drift + noise
    return x

# 真值参考样本: 直接从解析混合高斯采
def sample_reference(n_samples, seed):
    rng = np.random.default_rng(seed)
    comp = rng.choice(2, size=n_samples, p=WEIGHTS)
    out = np.empty((n_samples, 2))
    for k in range(2):
        m = comp == k
        out[m] = MUS[k] + rng.standard_normal((m.sum(), 2)) * np.sqrt(SIGS[k])
    return out

N_SAMPLES_EVAL = 2000
SIGMA_SWEEP = [0.0, 0.5, 1.0, 2.0, 4.0]
STEPS_LIST = [100, 5]
N_SEEDS_SWEEP = 5

def sweep_seeds(n_steps):
    """每个步数档的采样种子列表; 首种子与旧版单种子一致(2028+n_steps)"""
    return [2028 + n_steps + 1000 * s for s in range(N_SEEDS_SWEEP)]

ref_samples = sample_reference(N_SAMPLES_EVAL, seed=42)

def run_sweep(sampler):
    """对 SIGMA_SWEEP × STEPS_LIST × N_SEEDS_SWEEP 跑采样器, 返回 {步数: (均值, std, 原始矩阵)}"""
    out = {}
    for n_steps in STEPS_LIST:
        mat = np.array([[sliced_w2(sampler(sigma, n_steps, s), ref_samples)
                         for sigma in SIGMA_SWEEP] for s in range(N_SEEDS_SWEEP)])
        out[n_steps] = (mat.mean(0), mat.std(0), mat)
    return out

print(f"\n步骤3: σ 扫描（每配置 {N_SAMPLES_EVAL} 样本 × {N_SEEDS_SWEEP} 采样种子, 切片W2 对比解析真值样本）")
print(f"  {'σ':>4} | {'SW2 100步(均值±std)':>20} | {'SW2 5步(均值±std)':>18}   <- 14.3.7 边际不变仅对精确 score 成立")
sw2_model = run_sweep(lambda sigma, n_steps, s:
                      sample_sde_sigma(model_v, sigma, n_steps, N_SAMPLES_EVAL, sweep_seeds(n_steps)[s]))
for i, sigma in enumerate(SIGMA_SWEEP):
    m100, m5 = sw2_model[100], sw2_model[5]
    print(f"  {sigma:>4.1f} | {m100[0][i]:>10.4f}±{m100[1][i]:<8.4f} | {m5[0][i]:>8.4f}±{m5[1][i]:<8.4f}")

mean5, mean100 = sw2_model[5][0], sw2_model[100][0]
best_5 = SIGMA_SWEEP[int(np.argmin(mean5))]
best_100 = SIGMA_SWEEP[int(np.argmin(mean100))]
second_5 = SIGMA_SWEEP[int(np.argsort(mean5)[1])]
best_per_seed = [SIGMA_SWEEP[int(np.argmin(row))] for row in sw2_model[5][2]]
n_gain = int(sum(min(row[1:]) < row[0] for row in sw2_model[5][2]))  # 任一 σ>0 胜过 σ=0 的种子数
print(f"  -> 少步(5步): {n_gain}/{N_SEEDS_SWEEP} 个种子存在中间 σ 红利(最优σ>0); 逐种子最优 σ={best_per_seed}")
print(f"     种子平均最优 σ={best_5}（次优 {second_5}）-> 结论取'中间 σ∈[0.5,1] 有红利', 单点最优随种子浮动")
print(f"  -> 多步(100步): SW2 自 σ=0 的 {mean100[0]:.3f} 升至 σ=4 的 {mean100[-1]:.3f}"
      f"（对 σ 的鲁棒性远高于 5 步; 劣化来源由下方解析对照分离）")

print(f"\n  解析对照（精确 v*/s* 替换网络, 同款采样器——分离粗步离散化效应与模型误差）:")
sw2_ana = run_sweep(lambda sigma, n_steps, s:
                    sample_sde_analytic(sigma, n_steps, N_SAMPLES_EVAL, sweep_seeds(n_steps)[s]))
for i, sigma in enumerate(SIGMA_SWEEP):
    a100, a5 = sw2_ana[100], sw2_ana[5]
    f100 = "发散" if not np.isfinite(a100[0][i]) else f"{a100[0][i]:.4f}±{a100[1][i]:.4f}"
    f5 = "发散" if not np.isfinite(a5[0][i]) else f"{a5[0][i]:.4f}±{a5[1][i]:.4f}"
    print(f"  {sigma:>4.1f} | {f100:>18} | {f5:>16}")
ana_best_5 = SIGMA_SWEEP[int(np.nanargmin(sw2_ana[5][0]))]
ana_best_100 = SIGMA_SWEEP[int(np.nanargmin(sw2_ana[100][0]))]
print(f"  -> 解析 5 步: 中间 σ 仍占优（最优 σ={ana_best_5}: SW2={np.nanmin(sw2_ana[5][0]):.3f} vs σ=0 的 {sw2_ana[5][0][0]:.3f}）"
      f"-> 少步中间 σ 红利来自粗步离散化本身; 模型误差只是叠加（精确 score 下红利更大: "
      f"{np.nanmin(sw2_ana[5][0]):.3f} vs 模型 {mean5.min():.3f}）")
print(f"  -> 解析 100 步: SW2 不随 σ 劣化（最优 σ={ana_best_100}, σ=4 仍 {sw2_ana[100][0][-1]:.3f}）"
      f"-> 模型版 100 步的劣化并非离散化假象, 而是模型 score 误差被 σ²/2 放大"
      f"（14.3.7 的'边际不变'前提是精确 score）")

# 图2: (a) 少步散点对比 (b) SW2 vs σ 曲线
fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.4))
demo_steps = 5
# 注: sample_sde_sigma/sliced_w2 内部自带生成器种子, 无需(也不应)设全局种子
for ax, sigma, tag in zip(axes[:3], [0.0, 1.0, 4.0], ['a', 'b', 'c']):
    xs = sample_sde_sigma(model_v, sigma, demo_steps, 800, seed=2029)
    ax.scatter(ref_samples[:800, 0], ref_samples[:800, 1], s=5, c='gray', alpha=0.25, label='解析真值样本')
    ax.scatter(xs[:, 0], xs[:, 1], s=5, c='#d1495b', alpha=0.45, label=f'生成($\\sigma$={sigma})')
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect('equal')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper left')
    sw2_demo = sliced_w2(xs, ref_samples[:800])
    ax.set_title(f'({tag}) {demo_steps}步, $\\sigma$={sigma}: SW2={sw2_demo:.3f}\n(800 样本演示组)', fontsize=11)
ax = axes[3]
for n_steps, mk, color in zip(STEPS_LIST, ['o', 's'], ['#2a7de1', '#d1495b']):
    m, sd, _ = sw2_model[n_steps]
    ax.errorbar(SIGMA_SWEEP, m, yerr=sd, fmt=mk + '-', color=color, capsize=3, lw=1.5,
                label=f'{n_steps}步·模型({N_SEEDS_SWEEP}种子)')
for n_steps, mk, color in zip(STEPS_LIST, ['o', 's'], ['#3a9d23', '#8c564b']):
    m, sd, _ = sw2_ana[n_steps]
    ax.errorbar(SIGMA_SWEEP, m, yerr=sd, fmt=mk + '--', color=color, capsize=3, lw=1.2, alpha=0.85,
                label=f'{n_steps}步·解析对照')
if not np.isfinite(sw2_ana[5][0][-1]):
    ax.text(0.98, 0.72, '解析对照 5步 σ=4 发散(NaN, 未画出)', transform=ax.transAxes,
            ha='right', fontsize=8, color='#3a9d23')
ax.set_xlabel('噪声强度 $\\sigma$（常数调度）', fontsize=11)
ax.set_ylabel('切片 $W_2$ 距离（vs 解析真值）', fontsize=11)
ax.set_yscale('log'); ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8, loc='upper left')
ax.set_title('(d) $\\sigma$ 扫描: 模型 vs 解析对照（均值±std）', fontsize=11)
plt.suptitle('实验14.3-4 步骤3: SDE 扩展——多步对 $\\sigma$ 鲁棒, 少步中间 $\\sigma$ 有红利; 解析对照分离粗步离散化与模型 score 误差\n'
             '（(a)-(c) 为 800 样本单种子演示组, 与 (d) 的多种子统计不可直接对比）',
             fontsize=12.5, y=1.06)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_sigma扫描.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"图已保存: {os.path.join(SAVE_DIR, '步骤3_sigma扫描.png')}")

# ============================================================
# 步骤4：训练损失曲线
# ============================================================
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot([e for e, _ in losses_v], [l for _, l in losses_v], '-', color='#2a7de1')
ax.set_xlabel('训练 epoch', fontsize=11); ax.set_ylabel('CFM 损失（MSE）', fontsize=11)
ax.grid(alpha=0.3)
ax.set_title('实验14.3-4 步骤4: CFM 训练收敛（固定评估 batch 4096, 每 25 epoch）', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_训练损失.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"图已保存: {os.path.join(SAVE_DIR, '步骤4_训练损失.png')}")
print(f"  最终 CFM loss（固定评估 batch） = {losses_v[-1][1]:.6f}")

# ============================================================
# 保存数值结果
# ============================================================
import json

def _to_native(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    return obj

def _r5(v):
    """NaN -> None(JSON null, 表示数值发散), 其余保留 5 位小数"""
    return None if not np.isfinite(v) else round(float(v), 5)

results_summary = {
    'setup': {
        'mixture': {'mus': MUS.tolist(), 'sigmas_diag': SIGS.tolist(), 'weights': WEIGHTS.tolist()},
        'schedule': 'alpha_t = t, beta_t = 1-t (直线插值; beta>0 对 t<1)',
        'model_params': int(N_PARAMS),
    },
    'self_check': {
        'score_fd_max_err': float(f"{max(fd_errs):.2e}"),
        'analytic_triangle_closed': True,
    },
    'triangle_check': {
        'loop_err_max': float(f"{loop_err.max():.2e}"),
        'err_v_mean': round(float(err_v.mean()), 5), 'err_v_max': round(float(err_v.max()), 5),
        'err_D_mean': round(float(err_D.mean()), 5), 'err_D_max': round(float(err_D.max()), 5),
        'err_s_mean': round(float(err_s.mean()), 5), 'err_s_max': round(float(err_s.max()), 5),
        'per_t_mean_err': {str(tp): {k: round(v, 5) for k, v in d.items()} for tp, d in per_t_err.items()},
    },
    'sigma_sweep': {
        'sigma_list': SIGMA_SWEEP,
        'steps_list': STEPS_LIST,
        'n_seeds': N_SEEDS_SWEEP,
        'n_samples': N_SAMPLES_EVAL,
        'sliced_w2_model_mean': {str(k): [_r5(v) for v in sw2_model[k][0]] for k in STEPS_LIST},
        'sliced_w2_model_std': {str(k): [_r5(v) for v in sw2_model[k][1]] for k in STEPS_LIST},
        'best_sigma_per_seed_5steps': [float(b) for b in best_per_seed],
        'intermediate_sigma_gain': f'{n_gain}/{N_SEEDS_SWEEP} 个采样种子存在中间σ红利(最优σ>0)',
        'best_sigma_5steps_seed_mean': float(best_5),
        'best_sigma_100steps_seed_mean': float(best_100),
        'analytic_control': {
            'desc': '精确 v*/s* 同款采样器: 分离粗步离散化效应与模型 score 误差的 σ² 放大',
            'sliced_w2_mean': {str(k): [_r5(v) for v in sw2_ana[k][0]] for k in STEPS_LIST},
            'sliced_w2_std': {str(k): [_r5(v) for v in sw2_ana[k][1]] for k in STEPS_LIST},
            'best_sigma_5steps': float(ana_best_5),
            'best_sigma_100steps': float(ana_best_100),
        },
    },
    'final_cfm_loss': round(float(losses_v[-1][1]), 6),
    'final_cfm_loss_note': '固定评估 batch(4096 样本)上的 MSE, 非单训练 batch',
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"\n数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")

print("\n" + "=" * 62)
print("实验14.3-4 总结")
print("=" * 62)
print(f"""1. 解析真值三重自检: 有限差分验证 score*（误差 {max(fd_errs):.1e}）,
   三角公式在解析量上精确闭合 -> 后续所有对比的地基可靠

2. 三角验证（14.3.4 框的数值版, {len(probe_t)} 个探针点）:
   - 闭环 v→D→s→v' 误差 {loop_err.max():.1e}（浮点级 -> 公式实现零漂移）
   - 换算误差随三角系数线性缩放: mean err_v={err_v.mean():.4f} / err_D={err_D.mean():.4f} / err_s={err_s.mean():.4f}
     （D=β·v+x、s=(t·v−x)/(1−t) 是精确线性换算, 系数即缩放因子, 三者不必同量级）
   -> 学一个 v_θ, 免费拿到去噪器与得分——三角的红利在数值上兑现

3. SDE 扩展（14.3.7 的数值版）: s_θ 由三角公式从 v_θ 换算, 同一网络跑 5 档 σ × {N_SEEDS_SWEEP} 种子
   - 多步(100步): SW2 自 {mean100[0]:.3f}（σ=0）升至 {mean100[-1]:.3f}（σ=4）; 解析对照（精确 v*/s*）下
     100 步各 σ SW2 ≤ {np.nanmax(sw2_ana[100][0]):.3f} 且最优 σ={ana_best_100}
     -> 劣化不是离散化假象, 而是模型 score 误差被 σ²/2 放大（"边际不变"前提是精确 score）
   - 少步(5步): {n_gain}/{N_SEEDS_SWEEP} 种子存在中间 σ 红利; 种子平均最优 σ={best_5}（次优 {second_5}）,
     解析对照最优 σ={ana_best_5} -> 红利来自粗步离散化本身, 模型误差只是叠加并压小红利
   -> 随机性是"旋钮而非配方": 训练定路径, 采样选 σ; 最优档随步数/模型质量移动, 必须实测""")
