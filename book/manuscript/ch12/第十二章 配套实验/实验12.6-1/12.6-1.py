# -*- coding: utf-8 -*-
"""
实验12.6-1 实践意义与训练目标选择
对应章节: 12.6 实践意义与训练目标选择

知识点:
  - 简化目标(c=1)、完整VLB(c=w_t)、重要性采样(c=IS_t) 是同一个 ε-目标的不同加权
  - 三种加权可以用同一份训练代码实现, 只需换一个系数向量
  - 权重选择改变的是"模型把容量花在哪些噪声水平上", 因此可用似然(BPD)与去噪MSE量化
  - 同一个网络输出 ε̂_θ 同时承载得分函数、逆向均值、x0估计三种信息

实验内容:
  步骤1: 权重函数画像 —— c=1 / c=w_t / c=IS_t 的形状与数值范围
  步骤2: 三模型受控训练 —— 同种子、同初始化、同数据顺序, 只改损失权重
  步骤3: 采样结果对比
  步骤4: 定量指标 —— 测试集 BPD(负对数似然上界)、分时步去噪 MSE
  步骤5: 双向验证 —— 从 ε̂_θ 提取得分函数、逆向均值、x0估计

受控设计说明(与旧版的关键差别):
  旧版训练两个"目标完全相同、只有随机种子不同"的模型, 只能展示训练噪声,
  无法支撑正文关于权重选择的任何结论。本版改为:
    - 三个模型的初始化、数据顺序、噪声流完全相同(同一手动种子与专用 Generator)
    - 唯一的自变量是损失权重 c(t) (以及 IS 的时步采样分布 p_t)
    - 评价用固定测试集上的完整 VLB/BPD, 与训练时的加权口径无关, 三者可比

运行前提: PyTorch + GPU推荐
"""

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import io
import math
import time
import warnings
import logging

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默 matplotlib 相关警告
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
    SAVE_DIR = os.path.join(_gdrive, '实验12.6-1')
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
    print("警告: chinese_font模块未找到，中文字体可能无法正常显示")
# ========================================================

# 设置随机种子
np.random.seed(42)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"实验12.6-1: 实践意义与训练目标选择")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 训练")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")


# ============================================================
# 噪声调度: cosine调度 (Nichol & Dhariwal 2021)
# 说明: 线性调度(1e-4, 0.02)是为T=1000设计的; 若直接用于T=200,
# 终端 √ᾱ_T≈0.36 仍残留信号, 与采样起点 x_T~N(0,I) 分布失配,
# 导致样本模糊/畸变。cosine调度在短T下终端SNR自然衰减到≈0。
# 数组下标约定: 下标 i 对应数学时步 t = i+1 (betas[0]=β_1)
# ============================================================
T = 200
_s = 0.008
SCHEDULE_TAG = f'cosine_s{_s}_T{T}'  # 写入checkpoint的调度指纹, 防止旧调度的ckpt被新代码误用
_steps = torch.arange(T + 1, dtype=torch.float64)
_ac = torch.cos((_steps / T + _s) / (1 + _s) * math.pi / 2) ** 2
_ac = _ac / _ac[0]
betas = (1 - _ac[1:] / _ac[:-1]).clamp(max=0.999).float().to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
posterior_var = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)
beta_over_sqrt_1m_ab = betas / sqrt_one_minus_alpha_bars

print(f"\n噪声调度: T={T} (cosine调度, s={_s}), 终端√ᾱ_T={sqrt_alpha_bars[-1].item():.4e} (采样起点N(0,I)与终端分布匹配)")

# 训练只在数组下标 [1, T-1] 上采样, 即数学时步 t ∈ [2, T]
# 原因: 下标0对应 t=1, 此时 β̃_1 = 0 ⇒ 完整VLB权重 w_1 = ∞ (该步由离散解码器 L_0 单独处理)
T_START, T_END = 1, T - 1
T_EFF = T_END - T_START + 1   # 199
N_DIM = 28 * 28               # 每样本维数, BPD 归一化用


# ============================================================
# 三种权重: 全部写成 c(t) 向量, 训练循环只改这一个系数
# ============================================================
def _build_vlb_weight():
    """完整VLB权重 w_t = β_t²/(2β̃_t α_t (1-ᾱ_t)) = β_t/(2(1-ᾱ_{t-1})α_t)

    下标0 (t=1) 处 β̃_1 = 0, w_1 为 +∞, 置 inf 以便下方显式排除。
    """
    w = torch.full((T,), float('inf'), dtype=torch.float64, device=device)
    bt = betas.double()
    pv = posterior_var.double()
    al = alphas.double()
    ab = alpha_bars.double()
    for i in range(T_START, T_END + 1):
        w[i] = bt[i] ** 2 / (2 * pv[i] * al[i] * (1 - ab[i]))
    return w


W_VLB_RAW = _build_vlb_weight()                    # 原始 w_t (含末端尖峰)
W_Q99 = torch.quantile(W_VLB_RAW[T_START:], 0.99)  # t≥2 的 99% 分位
W_VLB = W_VLB_RAW.clamp(max=W_Q99.item())          # 截断版
W_VLB = W_VLB / W_VLB[T_START:].mean()             # 均值归一化 (Adam 近似尺度不变, 归一化只为数值可比)
W_SIMPLE = torch.ones(T, dtype=torch.float64, device=device)

_w = W_VLB_RAW[T_START:]
print(f"完整VLB权重: w_1=+∞(排除), min={_w.min():.5f}(t={(int(_w.argmin())+T_START+1)}), "
      f"max={_w.max():.4f}(t={T}), 跨度 {_w.max()/_w.min():.2e} 倍")
print(f"  末两步(t={T-1},{T})合计占总权重 {(_w[-2:].sum()/_w.sum()*100):.2f}% ⇒ t≥2 的99%分位={W_Q99.item():.4f}, 仅截断{int((W_VLB_RAW[T_START:] > W_Q99).sum())}个时步")


class ISScheme:
    """重要性采样: t ~ p_t, 系数 c_t/(T_eff·p_t), 使 Σ_t 的估计保持无偏

    p_t ∝ sqrt(EMA[c_t²·per²]): 对单样本估计量 Σ_t f_t/p_t 而言, 该提议分布
    最小化方差 (标准的 sqrt(二阶矩) 规则)。为防退化, p_t 先被夹在
    [0.2/T_eff, 5/T_eff] 内再归一化 (归一化会让实际频率略越出夹逼区间)。
    这正是正文所说"实现更复杂、需额外调参"的具体形态。
    """

    def __init__(self, c_vec, momentum=0.9):
        self.c_vec = c_vec
        self.momentum = momentum
        self.ema = torch.ones(T, dtype=torch.float64, device=device)  # 二阶矩 EMA
        self.p = torch.full((T,), 1.0 / T_EFF, dtype=torch.float64, device=device)

    def probs(self):
        raw = self.ema[T_START:T_END + 1].clamp(min=0.0).sqrt()
        if not torch.isfinite(raw).all() or raw.sum() <= 0:
            return self.p
        p = raw / raw.sum()
        p = p.clamp(min=0.2 / T_EFF, max=5.0 / T_EFF)
        p = p / p.sum()
        out = torch.full((T,), 0.0, dtype=torch.float64, device=device)
        out[T_START:T_END + 1] = p
        return out

    @torch.no_grad()
    def update(self, t, per_sample):
        f = (self.c_vec[t] * per_sample.double()) ** 2
        s = torch.zeros(T, dtype=torch.float64, device=device).index_add_(0, t, f)
        c = torch.zeros(T, dtype=torch.float64, device=device).index_add_(
            0, t, torch.ones_like(f))
        m = c > 0
        self.ema[m] = self.momentum * self.ema[m] + (1 - self.momentum) * (s[m] / c[m])
        self.p = self.probs()


# ============================================================
# 正向过程: q(x_t|x_0)
# ============================================================
def q_sample(x_0, t, noise=None):
    """x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε"""
    if noise is None:
        noise = torch.randn_like(x_0)
    return (
        sqrt_alpha_bars[t][:, None, None, None] * x_0 +
        sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
    )


# ============================================================
# 去噪网络: SmallUNet
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ConvBlock(nn.Module):
    """卷积块: Conv + GroupNorm + SiLU + 时间注入 + 残差连接"""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        gn_groups = min(4, out_ch)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(gn_groups, out_ch),
            nn.SiLU(),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(gn_groups, out_ch),
            nn.SiLU(),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class SmallUNet(nn.Module):
    """小型UNet去噪网络 (通道数 [1, 16, 32, 64], 时间嵌入维度64)"""
    def __init__(self, time_dim=64):
        super().__init__()
        ch = [1, 16, 32, 64]
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.down1 = ConvBlock(ch[0], ch[1], time_dim)
        self.down2 = ConvBlock(ch[1], ch[2], time_dim)
        self.down3 = ConvBlock(ch[2], ch[3], time_dim)
        self.bottleneck = ConvBlock(ch[3], ch[3], time_dim)
        self.up3 = ConvBlock(ch[3] + ch[2], ch[2], time_dim)
        self.up2 = ConvBlock(ch[2] + ch[1], ch[1], time_dim)
        self.up1 = ConvBlock(ch[1] + ch[0], ch[0], time_dim)
        self.out_conv = nn.Conv2d(ch[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x_t, t):
        t_emb = self.time_mlp(t)
        h1 = self.down1(x_t, t_emb)
        h2 = self.down2(self.pool(h1), t_emb)
        h3 = self.down3(self.pool(h2), t_emb)
        h = self.bottleneck(h3, t_emb)           # bottleneck只调用一次
        h = F.interpolate(h, size=(14, 14), mode='nearest')
        h = self.up3(torch.cat([h, h2], dim=1), t_emb)
        h = F.interpolate(h, size=(28, 28), mode='nearest')
        h = self.up2(torch.cat([h, h1], dim=1), t_emb)
        h = self.up1(torch.cat([h, x_t], dim=1), t_emb)
        return self.out_conv(h)


# ============================================================
# DDPM采样算法
# ============================================================
@torch.no_grad()
def ddpm_sample(model, shape, generator=None):
    """DDPM反向采样 x_T→x_0 (ε-prediction)

    传入同一个 generator 可让不同模型从同一份初始噪声出发, 便于逐格对比。
    """
    model.eval()
    x = torch.randn(shape, device=device, generator=generator)
    for t_idx in reversed(range(T)):
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        pred = model(x, t)
        model_mean = sqrt_recip_alphas[t_idx] * (
            x - beta_over_sqrt_1m_ab[t_idx] * pred
        )
        if t_idx == 0:
            x = model_mean
        else:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x = model_mean + torch.sqrt(posterior_var[t_idx]) * noise
    return x


# ============================================================
# 数据加载 (归一化到[-1,1])
# ============================================================
print("\n加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # 归一化到[-1,1]
])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
print(f"训练集: {len(train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")

DATA_SEED = 2024      # 数据打乱顺序 (三个模型共用)
NOISE_SEED = 777      # 训练噪声流 (三个模型共用)
T_SEED = 778          # 时步采样流 (simple/vlb 共用; is 走 multinomial)
EVAL_SEED = 10007     # 测试集评估的噪声流 (三个模型共用, 保证可比)
N_VAL = 1000          # 训练过程中周期评估的样本数
N_TEST = 5000         # 最终 BPD 评估的样本数


def _stack_dataset(ds, n):
    xs = torch.stack([ds[i][0] for i in range(n)])
    return xs.to(device)


with torch.no_grad():
    X_VAL = _stack_dataset(test_dataset, N_VAL)
    X_TEST = _stack_dataset(test_dataset, N_TEST)


# ============================================================
# 评估: 固定样本上的完整 VLB (nats) 与 BPD
# ============================================================
@torch.no_grad()
def eval_metrics(model, x0, chunk=500, desc=None):
    """在固定 x0 上精确求和完整 VLB, 返回 {vlb_nats, bpd, mse_curve}

    VLB = L_T + Σ_{t=2}^{T} L_{t-1} + L_0
        = 0  + Σ_{i=1}^{T-1} w_{i+1}·||ε-ε̂||²  + 解码器项
    其中 L_T = KL(q(x_T|x_0)||N(0,I)) ≈ 0 (cosine 下 ᾱ_T ≈ 6e-8, 量级 1e-7 nats)
    L_0 用固定方差 β_1 的离散解码器 (Ho et al. 2020 的 Gaussian 分箱积分)
    噪声按 t 固定播种, 三个模型看到完全相同的 (x0, ε) 对。
    """
    model.eval()
    N = x0.shape[0]
    total = torch.zeros(N, dtype=torch.float64, device=device)
    mse_curve = {}
    it = range(T_START, T_END + 1)
    if desc:
        it = tqdm(it, desc=desc, leave=False, unit='t')
    for i in it:
        g = torch.Generator(device=device).manual_seed(EVAL_SEED + i)
        eps = torch.randn(x0.shape, generator=g, device=device)
        acc = torch.zeros(N, dtype=torch.float64, device=device)
        for s in range(0, N, chunk):
            sl = slice(s, min(N, s + chunk))
            tb = torch.full((x0[sl].shape[0],), i, device=device, dtype=torch.long)
            xt = q_sample(x0[sl], tb, eps[sl])
            pred = model(xt, tb)
            acc[sl] = ((eps[sl] - pred) ** 2).flatten(1).sum(1).double()
        total += W_VLB_RAW[i].double() * acc          # 注意: 用未截断的 w_t, BPD 口径才严格
        mse_curve[i + 1] = float(acc.mean().item() / N_DIM)

    # ---- L_0: 离散解码器 (方差取 β_1, 与 Ho et al. 2020 一致) ----
    g0 = torch.Generator(device=device).manual_seed(EVAL_SEED)
    eps0 = torch.randn(x0.shape, generator=g0, device=device)
    t0 = torch.zeros(N, device=device, dtype=torch.long)
    x1 = q_sample(x0, t0, eps0)
    pred0 = model(x1, t0)
    mu0 = sqrt_recip_alphas[0] * (x1 - beta_over_sqrt_1m_ab[0] * pred0)
    sig0 = torch.sqrt(betas[0])
    dlt = 1.0 / 255.0
    cdf_u = 0.5 * (1 + torch.erf((x0 + dlt - mu0) / (sig0 * math.sqrt(2))))
    cdf_l = 0.5 * (1 + torch.erf((x0 - dlt - mu0) / (sig0 * math.sqrt(2))))
    nll0 = -torch.log((cdf_u - cdf_l).clamp(min=1e-20)).flatten(1).sum(1).double()

    nats = total + nll0
    return {
        'vlb_nats': float(nats.mean().item()),
        'bpd': float(nats.mean().item() / (N_DIM * math.log(2))),
        'nll0_nats': float(nll0.mean().item()),
        'kl_nats': float(total.mean().item()),
        'mse_curve': mse_curve,
    }


# ============================================================
# 步骤1：权重函数画像
# ============================================================
print(f"\n{'='*60}")
print("步骤1：权重函数画像（c=1 / c=w_t / c=IS_t）")
print(f"{'='*60}")

_w_np = W_VLB_RAW[T_START:].cpu().numpy()
_inv_beta = (1.0 / betas[T_START:].double()).cpu().numpy()
_tt = np.arange(T_START + 1, T + 1)
_corr = float(np.corrcoef(_w_np, _inv_beta)[0, 1])
print(f"  c=1: 常数 1")
print(f"  c=w_t: {_w_np.min():.4f} ~ {_w_np.max():.2f}, 与 1/β_t 的相关系数 {_corr:+.3f}")
print(f"  c=IS_t: 由 EMA 二阶矩自适应生成, 上限 5/T_eff")
print(f"  ⇒ cosine T={T} 下 w_t 呈 U 形（谷底 t={int(_w_np.argmin())+T_START+1}）,")
print(f"    不是'小 t 时极大(∝1/β_t)'; 真正极大的是末端 t={T}（β 被 clamp 到 0.999）")


# ============================================================
# 步骤2：三模型受控训练（同初始化、同数据顺序、只改权重）
# ============================================================
print(f"\n{'='*60}")
print("步骤2：三模型受控训练（唯一自变量 = 损失权重）")
print(f"{'='*60}")
print("""
核心设计:
  - 模型A (简化目标):  c(t) = 1                    —— Ho et al. 2020 的默认做法
  - 模型B (完整VLB):   c(t) = w_t (99%分位截断+归)  —— 严格似然下界
  - 模型C (重要性采样): c(t) = w_t, 但 t ~ p_t      —— 少采"白费力"的时步
  三者网络参数初始化、数据顺序、噪声流完全相同, 因此差异只能归因于权重。
  评价指标(完整VLB/BPD)在固定测试集上算, 与训练口径无关, 三者可比。
""")

num_epochs = 50
EVAL_EVERY = 10


def train_model(label, kind, seed, checkpoint_path):
    """训练去噪网络

    kind: 'simple' (c=1) / 'vlb' (c=w_t) / 'is' (c=w_t, 非均匀 t)
    """
    # 同一种子 ⇒ 三个模型的初始化权重逐比特相同
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    if kind == 'simple':
        c_vec = W_SIMPLE
        is_scheme = None
    elif kind == 'vlb':
        c_vec = W_VLB
        is_scheme = None
    elif kind == 'is':
        c_vec = W_VLB
        is_scheme = ISScheme(W_VLB)
    else:
        raise ValueError(kind)

    start_epoch = 0
    is_final = False
    train_losses = []
    train_mse = []
    val_epochs, val_nats = [], []

    if os.path.exists(checkpoint_path):
        print(f"\n检测到已保存的模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        for _k in ('train_losses', 'kind'):
            if _k not in checkpoint:
                raise RuntimeError(
                    f"检测到旧版本 checkpoint (缺少 '{_k}' 字段):\n  {checkpoint_path}\n"
                    f"请删除该文件后重新训练.")
        if checkpoint.get('schedule') != SCHEDULE_TAG or checkpoint.get('kind') != kind:
            raise RuntimeError(
                f"checkpoint 的调度/权重口径 ({checkpoint.get('schedule')}/{checkpoint.get('kind')}) "
                f"与当前代码 ({SCHEDULE_TAG}/{kind}) 不匹配:\n  {checkpoint_path}\n请删除该文件后重新训练.")
        model.load_state_dict(checkpoint['model_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        train_mse = checkpoint.get('train_mse', [])
        val_epochs = checkpoint.get('val_epochs', [])
        val_nats = checkpoint.get('val_nats', [])
        if checkpoint.get('is_final', False):
            print(f"✓ 最终模型, 跳过训练 (已训练 {checkpoint['epoch']+1} 轮, 损失 {checkpoint['loss']:.6f})")
            start_epoch = checkpoint['epoch'] + 1
            is_final = True
        else:
            print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if is_scheme is not None and 'is_ema' in checkpoint:
                is_scheme.ema = checkpoint['is_ema'].to(device)
                is_scheme.p = is_scheme.probs()
            start_epoch = checkpoint['epoch'] + 1

    if not is_final:
        _num_epochs = num_epochs
        if os.environ.get('QUICK_TEST', '') == '1':
            _num_epochs = 3
            print(f"  [快速验证模式] 仅训练 {_num_epochs} 轮")

        print(f"\n训练 [{label}] ({_num_epochs} 轮)...")
        t_start = time.time()

        # 数据顺序: 每个模型都用同一个 generator 重新播种, 因此 batch 序列完全一致
        loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                            num_workers=0, generator=torch.Generator().manual_seed(DATA_SEED))
        g_noise = torch.Generator(device=device).manual_seed(NOISE_SEED)
        g_t = torch.Generator(device=device).manual_seed(T_SEED)

        # 训练前的基线
        if not val_nats:
            _m = eval_metrics(model, X_VAL, desc=f'[{label}] 初始评估')
            val_epochs.append(0)
            val_nats.append(_m['vlb_nats'])
            print(f"  [{label}] 初始(未训练) 验证VLB = {_m['vlb_nats']:.2f} nats, BPD = {_m['bpd']:.4f}")

        for epoch in range(start_epoch, _num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_mse = 0.0
            n_seen = 0
            t_hist = torch.zeros(T, dtype=torch.long)
            pbar = tqdm(loader, desc=f'[{label}] Epoch {epoch+1}/{_num_epochs}',
                        leave=False, unit='batch')
            for x_batch, _ in pbar:
                x_batch = x_batch.to(device)
                batch = x_batch.shape[0]
                if is_scheme is None:
                    t = torch.randint(T_START, T_END + 1, (batch,), generator=g_t, device=device)
                    coef = c_vec[t]
                else:
                    p = is_scheme.p
                    t = torch.multinomial(p, batch, replacement=True, generator=g_t)
                    coef = c_vec[t] / (T_EFF * p[t])
                # batch 内归一化: 只差一个随步数波动的标量, 不改变梯度方向的期望
                coef = coef / coef.mean()

                noise = torch.randn(x_batch.shape, generator=g_noise, device=device)
                x_t = q_sample(x_batch, t, noise)
                pred = model(x_t, t)
                per = ((noise - pred) ** 2).flatten(1).sum(1)
                loss = (coef.float() * per).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 三者同样处理
                optimizer.step()

                if is_scheme is not None:
                    is_scheme.update(t, per)

                epoch_loss += loss.item() * batch
                epoch_mse += per.detach().mean().item() * batch
                t_hist += torch.bincount(t, minlength=T).cpu()
                n_seen += batch
                pbar.set_postfix(loss=f'{loss.item():.4f}')

            train_losses.append(epoch_loss / n_seen)
            train_mse.append(epoch_mse / n_seen)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  [{label}] Epoch {epoch+1:3d}/{_num_epochs}  obj={train_losses[-1]:.4f}  "
                      f"未加权MSE={train_mse[-1]:.5f}")

            if (epoch + 1) % EVAL_EVERY == 0 or epoch + 1 == _num_epochs:
                _m = eval_metrics(model, X_VAL, desc=f'[{label}] 验证评估')
                val_epochs.append(epoch + 1)
                val_nats.append(_m['vlb_nats'])
                print(f"    → 验证VLB={_m['vlb_nats']:.2f} nats (BPD={_m['bpd']:.4f})")

            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'kind': kind,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_losses[-1],
                    'train_losses': train_losses,
                    'train_mse': train_mse,
                    'val_epochs': val_epochs,
                    'val_nats': val_nats,
                    'is_ema': (is_scheme.ema if is_scheme is not None else None),
                    'schedule': SCHEDULE_TAG,
                    'is_final': False
                }, checkpoint_path)

        print(f"  [{label}] 训练完成, 未加权MSE={train_mse[-1]:.5f}, 耗时: {time.time()-t_start:.1f} 秒")
        if train_losses:
            torch.save({
                'epoch': _num_epochs - 1,
                'kind': kind,
                'model_state_dict': model.state_dict(),
                'loss': train_losses[-1],
                'train_losses': train_losses,
                'train_mse': train_mse,
                'val_epochs': val_epochs,
                'val_nats': val_nats,
                'is_ema': (is_scheme.ema if is_scheme is not None else None),
                'schedule': SCHEDULE_TAG,
                'is_final': True
            }, checkpoint_path)
            print(f"✓ [{label}] 模型已保存: {checkpoint_path}")
    else:
        print(f"\n使用已训练完成的 [{label}] 模型, 跳过训练过程")

    if is_scheme is not None:
        is_scheme.p = is_scheme.probs()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    return model, {'obj': train_losses, 'mse': train_mse,
                   'val_epochs': val_epochs, 'val_nats': val_nats,
                   'is_scheme': is_scheme}


CHECKPOINTS = {k: os.path.join(SAVE_DIR, f'ckpt_{k}.pth') for k in ('simple', 'vlb', 'is')}

# 初始化相同性检查: 同种子建两个网络, 参数应逐比特相同
torch.manual_seed(42)
_m1 = SmallUNet()
torch.manual_seed(42)
_m2 = SmallUNet()
_init_diff = max((a - b).abs().max().item() for a, b in zip(_m1.parameters(), _m2.parameters()))
del _m1, _m2
print(f"\n初始化一致性检查: 同种子两次建网的参数最大差 = {_init_diff:.3e}"
      f"{'  ✓ 三模型起点相同' if _init_diff == 0 else '  (注意: 起点不完全相同)'}")

model_a, hist_a = train_model('简化目标 c=1', 'simple', seed=42, checkpoint_path=CHECKPOINTS['simple'])
model_b, hist_b = train_model('完整VLB c=w_t', 'vlb', seed=42, checkpoint_path=CHECKPOINTS['vlb'])
model_c, hist_c = train_model('重要性采样 c=IS_t', 'is', seed=42, checkpoint_path=CHECKPOINTS['is'])


# ============================================================
# 步骤1(图): 权重函数画像
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
ax = axes[0]
ax.semilogy(_tt, _w_np, color='C1', lw=1.8)
ax.axhline(1.0, color='C0', ls='--', lw=1.2, label=r'$c(t)=1$ (简化目标)')
ax.set_xlabel('时步 $t$')
ax.set_ylabel(r'$w_t$ (log)')
ax.set_title(r'(a) 完整VLB权重 $w_t=\beta_t/(2(1-\bar\alpha_{t-1})\alpha_t)$')
ax.annotate(f'min {_w_np.min():.4f}\n@t={int(_w_np.argmin())+T_START+1}',
            xy=(int(_w_np.argmin()) + T_START + 1, _w_np.min()), xytext=(60, 0.5),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate(f'max {_w_np.max():.1f}', xy=(T, _w_np.max()), xytext=(150, 100),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, which='both')

ax = axes[1]
ax.loglog(_inv_beta, _w_np, '.', ms=3, color='C1')
ax.set_xlabel(r'$1/\beta_t$ (log)')
ax.set_ylabel(r'$w_t$ (log)')
ax.set_title(f'(b) $w_t$ 与 $1/\\beta_t$ 的关系 (相关系数 {_corr:+.3f})')
ax.grid(True, alpha=0.3, which='both')
ax.text(0.03, 0.95, '若 $w_t\\propto 1/\\beta_t$，散点应沿对角线排列',
        transform=ax.transAxes, fontsize=9, va='top', color='dimgray')

ax = axes[2]
_p_is = hist_c['is_scheme'].p[T_START:T_END + 1].cpu().numpy() * T_EFF if hist_c['is_scheme'] else np.ones(T_EFF)
ax.plot(_tt, _p_is, color='C2', lw=1.0, label='IS 采样分布 $p_t\\cdot T_{eff}$')
ax.axhline(1.0, color='k', ls=':', lw=1.2, label='均匀采样')
ax.set_xlabel('时步 $t$')
ax.set_ylabel('相对采样频率')
ax.set_title('(c) 重要性采样把时步采到哪里')
ax.set_yscale('log')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

fig.suptitle('三种损失权重: 同一目标的不同加权', fontsize=14, y=1.02)
fig.text(0.5, -0.04,
         f'注：cosine T={T} 下 $w_t$ 是被 $\\beta$ 的末端 clamp 抬高的 U 形曲线，末两步占总权重 '
         f'{(_w_np[-2:].sum()/_w_np.sum()*100):.1f}%；因此训练用的 $w_t$ 在 {int((W_VLB_RAW[T_START:] > W_Q99).sum())} 个时步上按 99% 分位截断',
         ha='center', fontsize=9, color='dimgray')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_权重函数对比.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n图表已保存: 步骤1_权重函数对比.png")


# ============================================================
# 步骤2(图)：训练曲线对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
_names = ['简化目标 $c=1$', '完整VLB $c=w_t$', '重要性采样 $c=IS_t$']
_hists = [hist_a, hist_b, hist_c]
_colors = ['C0', 'C1', 'C2']

ax = axes[0]
for nm, h, cc in zip(_names, _hists, _colors):
    if h['val_nats']:
        ax.plot(h['val_epochs'], h['val_nats'], 'o-', color=cc, ms=4, label=nm)
ax.set_xlabel('Epoch')
ax.set_ylabel('固定验证集上的完整 VLB (nats)')
ax.set_title('(a) 统一的似然指标: 完整 VLB')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
for nm, h, cc in zip(_names, _hists, _colors):
    if h['mse']:
        ax.plot(range(1, len(h['mse']) + 1), h['mse'], '-', color=cc, lw=1.4, label=nm)
ax.set_xlabel('Epoch')
ax.set_ylabel(r'训练 batch 未加权 $\|\varepsilon-\hat\varepsilon\|^2/784$')
ax.set_title('(b) 训练集上的未加权去噪 MSE')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle('三种权重的训练动态 (同初始化/同数据顺序)', fontsize=14, y=1.02)
fig.text(0.5, -0.04,
         '注：(a) 是与训练口径无关的统一标尺，可直接比较；(b) 各模型的 t 分布不同，仅作参考。'
         '单个 epoch 的训练损失受 t 随机采样影响方差很大，不宜用来判断收敛。',
         ha='center', fontsize=9, color='dimgray')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_训练曲线对比.png'), dpi=150, bbox_inches='tight')
plt.close()
print("图表已保存: 步骤2_训练曲线对比.png")


# ============================================================
# 步骤3：采样结果对比
# ============================================================
print(f"\n{'='*60}")
print("步骤3：采样结果对比")
print(f"{'='*60}")

n_samples = 8
sample_shape = (n_samples, 1, 28, 28)
_models = [model_a, model_b, model_c]
_labels = ['简化目标 c=1', '完整VLB c=w_t', '重要性采样 c=IS_t']
_samples = []
for nm, mdl in zip(_labels, _models):
    print(f"  [{nm}] 采样中...")
    _samples.append(ddpm_sample(mdl, sample_shape,
                                generator=torch.Generator(device=device).manual_seed(31337)))

fig, axes = plt.subplots(3, n_samples, figsize=(16, 6))
fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.08)
for r in range(3):
    img = ((_samples[r] + 1) / 2).clamp(0, 1)
    for i in range(n_samples):
        axes[r, i].imshow(img[i, 0].cpu().numpy(), cmap='gray')
        axes[r, i].axis('off')
    fig.text(0.02, 0.78 - 0.29 * r, _labels[r], fontsize=10, ha='left', va='center', rotation=90)

fig.suptitle('三种权重训练出的模型采样结果 (同一初始噪声)', fontsize=14, y=0.97)
fig.text(0.5, 0.015,
         '注：三行使用同一个 DDPM 采样器与同一份初始噪声，差异只来自训练权重；50 轮 SmallUNet 的样本仍偏模糊。',
         ha='center', fontsize=9, color='dimgray')
plt.savefig(os.path.join(SAVE_DIR, '步骤3_采样结果对比.png'), dpi=150)
plt.close()
print("图表已保存: 步骤3_采样结果对比.png")


# ============================================================
# 步骤4：定量指标 (BPD + 分时步去噪 MSE)
# ============================================================
print(f"\n{'='*60}")
print("步骤4：定量指标（测试集 BPD / 分时步去噪 MSE）")
print(f"{'='*60}")
print(f"在 {N_TEST} 张测试图像上精确求和完整 VLB，噪声按 t 固定播种，三模型共用同一份数据与噪声")

final_metrics = {}
for nm, mdl in zip(_labels, _models):
    _m = eval_metrics(mdl, X_TEST, desc=f'[{nm}] BPD 评估')
    final_metrics[nm] = _m
    print(f"  [{nm}] BPD={_m['bpd']:.4f}  VLB={_m['vlb_nats']:.2f} nats "
          f"(KL项 {_m['kl_nats']:.2f} + 解码项 {_m['nll0_nats']:.2f})")

T_PROBE = [10, 30, 50, 80]
print("\n  分时步去噪 MSE (每像素, 越低越好):")
_hdr = "  {:>6s}".format('t') + "".join(f"{n:>22s}" for n in _labels)
print(_hdr)
for tv in T_PROBE:
    row = f"  {tv:>6d}"
    best = min(final_metrics[n]['mse_curve'][tv] for n in _labels)
    for n in _labels:
        v = final_metrics[n]['mse_curve'][tv]
        row += f"{v:>18.5f}{'  *' if v == best else '   '}"
    print(row)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
ax = axes[0]
_bpds = [final_metrics[n]['bpd'] for n in _labels]
bars = ax.bar(range(3), _bpds, color=_colors, width=0.55)
for i, v in enumerate(_bpds):
    ax.text(i, v + max(_bpds) * 0.02, f'{v:.4f}', ha='center', fontsize=10)
ax.set_xticks(range(3))
ax.set_xticklabels(_labels, fontsize=9)
ax.set_ylabel('BPD (bits/dim, 越低越好)')
ax.set_title('(a) 测试集似然: 完整 VLB 的 BPD')
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
for nm, cc in zip(_labels, _colors):
    cur = final_metrics[nm]['mse_curve']
    ks = sorted(cur)
    ax.semilogy(ks, [cur[k] for k in ks], color=cc, lw=1.4, label=nm)
ax.set_xlabel('时步 $t$')
ax.set_ylabel('去噪 MSE (每像素, log)')
ax.set_title('(b) 各噪声水平上的去噪精度')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

fig.suptitle('定量比较: 权重选择把容量分配到了哪些噪声水平', fontsize=14, y=1.02)
fig.text(0.5, -0.04,
         '注：BPD 用未截断的 $w_t$ 精确求和，与训练时的截断/采样方式无关；'
         'MSE 在固定的 $t$ 上评估，三模型共用同一批 $(x_0,\\varepsilon)$。',
         ha='center', fontsize=9, color='dimgray')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_定量指标对比.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n图表已保存: 步骤4_定量指标对比.png")


# ============================================================
# 步骤5：双向验证
# ============================================================
print(f"\n{'='*60}")
print("步骤5：双向验证")
print(f"{'='*60}")
print("""
12.4节等价性的实践含义:
  - 从同一个网络输出 ε̂_θ，可以同时提取三种信息:
    1. 得分函数 s_θ = -ε̂_θ / √(1-ᾱ_t)  (采样路径)
    2. 逆向均值 μ_θ = (1/√α_t)(x_t - β_t/√(1-ᾱ_t)·ε̂_θ)  (变分路径)
    3. x₀估计 x̂₀ = (x_t - √(1-ᾱ_t)·ε̂_θ) / √ᾱ_t  (Tweedie等式)
""")

# 使用简化目标模型(模型A)进行验证
model_a.eval()
gv = torch.Generator(device=device).manual_seed(20240601)
x_test = torch.randn(1, 1, 28, 28, device=device, generator=gv)
t_val = 50
t_test = torch.tensor([t_val], device=device)

with torch.no_grad():
    eps_pred = model_a(x_test, t_test)  # 网络输出: ε̂_θ

    # 采样路径视角: 提取得分函数
    ab_t = alpha_bars[t_val]
    score_from_eps = -eps_pred / torch.sqrt(1 - ab_t)

    # 变分路径视角: 计算逆向均值
    mu_from_eps = (1 / torch.sqrt(alphas[t_val])) * (
        x_test - (betas[t_val] / torch.sqrt(1 - ab_t)) * eps_pred
    )

    # Tweedie等式: 提取x₀估计
    x0_from_eps = (x_test - torch.sqrt(1 - ab_t) * eps_pred) / torch.sqrt(ab_t)

print("从同一个网络输出 ε̂_θ 提取的不同量:")
print(f"  ε̂_θ 范数: {eps_pred.norm().item():.4f}  (网络直接输出)")
print(f"  s_θ = -ε̂_θ/√(1-ᾱ_t) 范数: {score_from_eps.norm().item():.4f}  (采样路径: 得分函数)")
print(f"  μ_θ 范数: {mu_from_eps.norm().item():.4f}  (变分路径: 逆向均值)")
print(f"  x̂₀ = (x_t-√(1-ᾱ_t)ε̂)/√ᾱ_t 范数: {x0_from_eps.norm().item():.4f}  (去噪估计)")
print()
print("→ 一个网络输出，三种解读，三种用途")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验12.6-1 总结")
print(f"{'='*60}")

_best_bpd = min(_labels, key=lambda n: final_metrics[n]['bpd'])
print(f"""
1. 三种权重是同一个 ε-目标的三种加权 (12.6节)
   - c(t)=1 / c(t)=w_t / c(t)=IS_t 共用同一份训练代码, 只换一个系数向量
   - 权重函数形态: w_t 在 cosine T={T} 下呈 U 形, 谷底 t={int(_w_np.argmin())+T_START+1},
     末端 t={T} 为 {_w_np.max():.1f} (β 被 clamp 到 0.999 所致), 末两步占总权重 {(_w_np[-2:].sum()/_w_np.sum()*100):.1f}%

2. 定量结果 ({N_TEST} 张测试图, 完整VLB口径)
""")
for nm in _labels:
    print(f"   - {nm}: BPD={final_metrics[nm]['bpd']:.4f}, "
          f"MSE@t=50={final_metrics[nm]['mse_curve'][50]:.5f}")
print(f"""   → 本次运行中 BPD 最低的是「{_best_bpd}」

3. 双向验证 (12.4节实践验证)
   - 从同一个 ε̂_θ 同时提取得分函数、逆向均值、x₀估计
   - 三种视角共享同一参数化, 差异只在"如何解读"

4. 实践意义 (12.6节)
   - 选择哪种视角解释不影响训练和采样——代码完全相同
   - 但选择哪种"权重"会真实改变模型把容量花在哪里, 这是需要实验回答的工程问题
   - 单个种子、50 轮、SmallUNet 的结论只具描述性; 要下普适结论需多种子与更大模型
""")

print(f"{'='*60}")
print("第十二章配套实验12.6-1完成!")

# ===== 保存数值结果 =====
import json


def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return _to_native(obj.detach().cpu().tolist())
    return obj


results_summary = {
    'config': {
        'T': T, 'schedule': SCHEDULE_TAG, 'num_epochs': num_epochs,
        'batch_size': 128, 'lr': 2e-4, 'seed_all_models': 42,
        'data_seed': DATA_SEED, 'noise_seed': NOISE_SEED, 'eval_seed': EVAL_SEED,
        't_range': [T_START + 1, T], 'grad_clip': 1.0,
        'n_val': N_VAL, 'n_test': N_TEST,
        'init_max_abs_diff': _init_diff,
    },
    'weight_functions': {
        'w_vlb_raw': W_VLB_RAW[T_START:].cpu().tolist(),
        'w_vlb_used_for_training': W_VLB[T_START:].cpu().tolist(),
        'w_min': float(_w_np.min()), 'w_min_t': int(_w_np.argmin()) + T_START + 1,
        'w_max': float(_w_np.max()), 'w_max_t': T,
        'w_q99': float(W_Q99.item()),
        'n_clipped_steps': int((W_VLB_RAW > W_Q99).sum().item()),
        'share_top2_steps': float(_w_np[-2:].sum() / _w_np.sum()),
        'corr_w_inv_beta': _corr,
        'is_p_t': _p_is.tolist(),
    },
    'models': {
        'simple': {'train_obj': hist_a['obj'], 'train_mse': hist_a['mse']},
        'vlb': {'train_obj': hist_b['obj'], 'train_mse': hist_b['mse']},
        'is': {'train_obj': hist_c['obj'], 'train_mse': hist_c['mse']},
    },
    'validation_curve': {
        'epochs': hist_a['val_epochs'],
        'vlb_nats': {'simple': hist_a['val_nats'],
                     'vlb': hist_b['val_nats'],
                     'is': hist_c['val_nats']},
        'note': '固定验证集(1000张)上精确求和完整VLB, 与训练口径无关, 三模型可比',
    },
    'final_metrics': {
        k: {kk: vv for kk, vv in v.items() if kk != 'mse_curve'}
        for k, v in final_metrics.items()
    },
    'mse_at_t': {
        str(tv): {n: final_metrics[n]['mse_curve'][tv] for n in _labels}
        for tv in T_PROBE
    },
    'mse_curve': {n: {str(k): v for k, v in final_metrics[n]['mse_curve'].items()}
                  for n in _labels},
    'is_sampling': {
        'p_t': hist_c['is_scheme'].p[T_START:T_END + 1].cpu().tolist(),
        'relative_freq': _p_is.tolist(),
    },
    'dual_verification': {
        'eps_pred_norm': eps_pred.norm().item(),
        'score_norm': score_from_eps.norm().item(),
        'mu_norm': mu_from_eps.norm().item(),
        'x0_est_norm': x0_from_eps.norm().item(),
        'note': '同一网络输出承载三种信息: score, 逆向均值, x0估计',
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
