# -*- coding: utf-8 -*-
"""
验证实验15.4-1（增强版）: adaLN-Zero 的真实检验
====================================================

背景: 原实验15.4-1步骤3的消融存在三个硬伤, 使其先天无法检验
     DiT论文(Peebles & Xie 2023)关于 adaLN-Zero 的结论:
       (1) 判据错位: 论文优势体现在生成质量FID(25.21→19.47),
           而原实验用训练MSE($L_{simple}$)——MSE是FID的弱代理;
       (2) 深度不足: depth=4 时恒等初始化的"深层信号传播"收益本就有限,
           检验"Zero优势随深度增大"必须把深度作为自变量;
       (3) 单次运行: 两种初始化末轮Loss差异~0.001属随机波动量级,
           没有多种子配对比较就无法区分效应与噪声。

本脚本的设计:
  A. 深度作为自变量: depth ∈ {4, 12}（12层对齐DiT论文消融所用DiT-B/2的深度）
  B. 判据升级:
     - 每轮在固定验证集(2000张测试图+固定t网格+固定噪声)上算验证Loss
     - 训练后做DDPM祖先采样, 计算两个生成质量代理指标:
       · 像素Fréchet距离(8×8降采样特征, 类FID, 诚实标注为代理)
       · 最近邻PSNR(生成样本对测试集的NN-PSNR, 衡量样本真实感)
  C. 多种子配对: 同一seed下 Zero/NoZero 两组除adaLN初始化外
     全部权重、数据顺序、噪声流完全一致(严格配对比较)
  D. 早期振荡定量: 记录前400个step的batch loss, 计算相邻step
     loss变化幅度——直接检验原脚本"随机初始化振荡明显"的断言
  E. (可选)学习率鲁棒性: LRS中加入1e-3即可检验"adaLN-Zero允许更大lr"

判读标准(脚本末尾自动打印):
  - 若 depth=12 上 Zero 的验证Loss/采样指标显著优于 NoZero,
    而 depth=4 上两者接近 → 支持"Zero优势随深度增大"的固有属性论
  - 若两组在所有指标上均无差异 → 本设定仍不足以检验(受限于MNIST/MSE代理),
    结论应表述为"无法证伪也无法证实", 不得包装成验证
  - 若 NoZero 早期振荡显著更大 → 支持"恒等初始化稳定早期训练"的机制断言

运行前提: PyTorch, 强烈建议GPU。
  默认配置(2深度×2初始化×2种子=8次训练, MNIST 6万张, 20轮):
    GPU(T4/3060级)约 1.5~3 小时; CPU 不可行。
  快速冒烟测试: 把 QUICK_MODE 改为 True (约10~20分钟GPU)。
  Colab: 菜单 运行时->更改运行时类型 选GPU; 需先挂载Drive(脚本会自动提示),
        权重/图表保存到 MyDrive/实验15.4-1, 断线重跑可续训;
        MNIST数据与.chinese字体与原实验同目录复用, 不会重复下载。

输出文件(本地=本脚本同目录; Colab=Google Drive/实验15.4-1):
  ablation_d{depth}_{init}_lr{lr}_seed{seed}[_quick].pth   各run的checkpoint(支持断点续训, quick/全量互不污染)
  验证1_验证Loss收敛曲线.png
  验证2_早期batch振荡对比.png
  验证3_采样质量对比.png
  验证结果_summary.json
"""

import sys
import os
import io
import json
import time
import math
import warnings
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Windows 控制台 UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*glyph.*")

# ====== 路径与中文字体配置(兼容本地和Google Colab, 与15.4-1行为一致) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验15.4-1')
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
    print("警告: chinese_font 模块未找到, 中文字体可能无法正常显示")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ============================================================
# ★ 实验配置(自变量与预算都在这里)
# ============================================================
QUICK_MODE = False           # True=快速冒烟测试(结论仅供调试, 不可用于判读)

DEPTHS = [4, 12]             # ★ 深度作为自变量: 浅网 vs 论文消融同级深度
INITS = ['zero', 'nozero']   # adaLN-Zero vs 随机初始化(其余完全同seed配对)
LRS = [2e-4]                 # 检验lr鲁棒性时改为 [2e-4, 1e-3]
SEEDS = [0, 1]               # 多种子配对比较; 预算允许建议3~5个

HIDDEN = 128                 # 固定宽度, 只动深度, 混杂变量最小化
NUM_HEADS = 4
PATCH = 2
BATCH = 128
EPOCHS_BY_DEPTH = {4: 20, 12: 20}
WEIGHT_DECAY = 0.01

T = 200                      # DDPM步数(与15.4-1一致)
BETA_MIN, BETA_MAX = 1e-4, 0.02

N_VAL = 2000                 # 验证集大小(取测试集前N_VAL张, 固定)
VAL_T_GRID = 20              # 验证用固定t网格点数
TRACE_STEPS = 400            # 早期batch loss轨迹长度
N_SAMPLE = 600               # 采样质量评估的生成样本数
NN_POOL = 5000               # NN-PSNR的参考池(测试图数量)

if QUICK_MODE:
    SEEDS = [0]
    EPOCHS_BY_DEPTH = {4: 6, 12: 6}
    N_VAL, N_SAMPLE, NN_POOL = 500, 200, 1000
    print("!!! QUICK_MODE: 仅用于冒烟测试, 样本/轮数不足, 结果不可用于判读 !!!")

# ★ 模式标记入文件名: 避免quick的6轮权重被全量运行误认为"已完成"而跳过训练
RUN_TAG = '_quick' if QUICK_MODE else ''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*64}")
print("验证实验15.4-1(增强版): adaLN-Zero 的真实检验")
print(f"{'='*64}")
print(f"设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
if device.type == 'cpu':
    print("警告: CPU上本实验不可行(8次训练), 请使用GPU或QUICK_MODE")
print(f"设计: depth∈{DEPTHS} × init∈{INITS} × seed∈{SEEDS} × lr∈{LRS}")
print(f"判据: 固定验证Loss + DDPM采样(像素Fréchet代理 + NN-PSNR) + 早期振荡轨迹")

# ============================================================
# DDPM噪声调度(与15.4-1一致)
# ============================================================
betas = torch.linspace(BETA_MIN, BETA_MAX, T).to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)


# ============================================================
# DiT(与15.4-1相同的架构, 增加 zero_init 开关)
# ============================================================
def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiTBlock(nn.Module):
    """Transformer Block + adaLN-Zero(15.3.3节)"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        h = self.norm1(x)
        h = h * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        h, _ = self.attn(h, h, h)
        x = x + gate_msa[:, None, :] * h
        h = self.norm2(x)
        h = h * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        h = self.mlp(h)
        x = x + gate_mlp[:, None, :] * h
        return x


class Patchify(nn.Module):
    def __init__(self, patch_size=2, in_channels=1, hidden_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size * in_channels, hidden_size)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, (H // p) * (W // p), p * p * C)
        return self.proj(x)


class Unpatchify(nn.Module):
    def __init__(self, patch_size=2, out_channels=1, hidden_size=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.out_channels = out_channels

    def forward(self, x, H, W):
        B = x.shape[0]
        p = self.patch_size
        x = self.proj(x)
        x = x.reshape(B, H // p, W // p, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, self.out_channels, H, W)


class DiT(nn.Module):
    """Patchify → depth×DiTBlock → Unpatchify

    zero_init=True : adaLN调制层全部Zero初始化(→恒等映射起步)
    zero_init=False: 同一模型构建后, 用独立随机流把adaLN层覆盖为
                     Xavier/Normal初始化——★ 其余所有权重(pos_embed/attn/
                     mlp/patchify等)与zero组逐位相同, 保证严格配对,
                     唯一差异就是"adaLN是否Zero初始化"。
    """
    def __init__(self, hidden_size=128, depth=4, num_heads=4,
                 patch_size=2, in_channels=1, zero_init=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.patchify = Patchify(patch_size, in_channels, hidden_size)
        self.unpatchify = Unpatchify(patch_size, in_channels, hidden_size)
        num_patches = (28 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        # adaLN调制层初始化(默认Zero)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        if not zero_init:
            self._randomize_adaln()

    def _randomize_adaln(self):
        """仅覆盖adaLN调制层为随机初始化(用独立seed, 不扰动其他权重)"""
        for block in self.blocks:
            nn.init.xavier_uniform_(block.adaLN_modulation[-1].weight)
            nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.02)
        nn.init.xavier_uniform_(self.final_adaLN[-1].weight)
        nn.init.normal_(self.final_adaLN[-1].bias, std=0.02)

    def forward(self, x_t, t):
        c = self.time_mlp(timestep_embedding(t, self.hidden_size))
        h = self.patchify(x_t) + self.pos_embed
        for block in self.blocks:
            h = block(h, c)
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale[:, None, :]) + shift[:, None, :]
        return self.unpatchify(h, 28, 28)


# ============================================================
# 数据
# ============================================================
print("\n加载MNIST...")
data_dir = os.path.join(SAVE_DIR, 'data')
os.makedirs(data_dir, exist_ok=True)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
print(f"  训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")


def make_val_inputs(seed=999):
    """固定验证输入: N_VAL张测试图 + 固定t网格 + 固定噪声 → 确定性验证Loss"""
    g = torch.Generator().manual_seed(seed)
    x = torch.stack([test_dataset[i][0] for i in range(N_VAL)]).to(device)
    t_grid = torch.linspace(0, T - 1, VAL_T_GRID).long().to(device)
    t = t_grid[torch.arange(N_VAL) % VAL_T_GRID]
    eps = torch.randn(x.shape, generator=g).to(device)
    x_t = sqrt_alpha_bars[t][:, None, None, None] * x \
        + sqrt_one_minus_alpha_bars[t][:, None, None, None] * eps
    return x_t, t, eps


@torch.no_grad()
def val_loss(model, x_t, t, eps):
    model.eval()
    total, n = 0.0, x_t.shape[0]
    for i in range(0, n, 250):
        pred = model(x_t[i:i + 250], t[i:i + 250])
        total += F.mse_loss(pred, eps[i:i + 250], reduction='sum').item()
    return total / (n * eps[0].numel())


# ============================================================
# DDPM祖先采样 + 生成质量代理指标
# ============================================================
@torch.no_grad()
def ddpm_sample(model, n_samples, batch=100):
    model.eval()
    out = []
    for _ in range(math.ceil(n_samples / batch)):
        b = batch  # 固定batch采样, 最后按n_samples截断
        x = torch.randn(b, 1, 28, 28, device=device)
        for t_idx in reversed(range(T)):
            tt = torch.full((b,), t_idx, device=device, dtype=torch.long)
            eps = model(x, tt)
            ab = alpha_bars[t_idx]
            mean = (x - betas[t_idx] / sqrt_one_minus_alpha_bars[t_idx] * eps) / torch.sqrt(alphas[t_idx])
            if t_idx > 0:
                x = mean + torch.sqrt(betas[t_idx]) * torch.randn_like(x)
            else:
                x = mean
        out.append(x.clamp(-1, 1).cpu())
    return torch.cat(out)[:n_samples]


def to_feat(x, size=8):
    """[−1,1]图像 → [0,1] → size×size降采样展平特征(像素Fréchet用)"""
    x = (x + 1) / 2
    x = F.adaptive_avg_pool2d(x, size)
    return x.reshape(x.shape[0], -1).numpy()


def frechet_distance(a, b):
    """两个高斯(均值,协方差)之间的Fréchet距离, 协方差开方用eigh实现"""
    ma, sa = a.mean(0), np.cov(a, rowvar=False)
    mb, sb = b.mean(0), np.cov(b, rowvar=False)
    diff = ma - mb
    # sqrt(sa) via eigh
    w, v = np.linalg.eigh(sa)
    sqrt_sa = v @ np.diag(np.sqrt(np.clip(w, 0, None))) @ v.T
    m = sqrt_sa @ sb @ sqrt_sa
    w2, _ = np.linalg.eigh(m)
    tr = np.sqrt(np.clip(w2, 0, None)).sum()
    return float(diff @ diff + np.trace(sa) + np.trace(sb) - 2 * tr)


def nn_psnr(gen, pool):
    """生成样本到参考图像池的最近邻PSNR均值(样本真实感代理)"""
    g = gen.reshape(gen.shape[0], -1)
    p = pool.reshape(pool.shape[0], -1)
    psnrs = []
    for i in range(0, g.shape[0], 100):
        d = torch.cdist(g[i:i + 100], p)
        idx = d.argmin(dim=1)
        mse = ((g[i:i + 100] - p[idx]) ** 2).mean(dim=1)
        psnrs.append(10 * torch.log10(1.0 / mse.clamp_min(1e-10)))
    return float(torch.cat(psnrs).mean())


# ============================================================
# 训练(带batch轨迹记录与断点续训)
# ============================================================
def run_name(depth, init, lr, seed):
    return f'ablation_d{depth}_{init}_lr{lr:g}_seed{seed}{RUN_TAG}'


def train_one(depth, init, lr, seed):
    name = run_name(depth, init, lr, seed)
    ckpt_path = os.path.join(SAVE_DIR, name + '.pth')
    epochs = EPOCHS_BY_DEPTH[depth]

    # ★ 配对关键: 两组用同一seed构建模型 → 除adaLN覆盖外所有随机流一致
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DiT(hidden_size=HIDDEN, depth=depth, num_heads=NUM_HEADS,
                patch_size=PATCH, zero_init=(init == 'zero')).to(device)
    # nozero组的adaLN覆盖在__init__内完成, 使用的随机数发生在构建时,
    # 为消除这一顺序差异, 这里用独立seed重新覆盖一次(结果仍确定性):
    if init == 'nozero':
        torch.manual_seed(seed + 100000)
        model._randomize_adaln()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    data_gen = torch.Generator().manual_seed(seed)  # 相同数据顺序
    loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, generator=data_gen)

    hist = {'train': [], 'val': [], 'batch_trace': [], 'epochs_done': 0}
    start_epoch = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck.get('is_final', False):
            print(f"  [{name}] 已完成({len(ck['hist']['train'])}轮), 跳过训练")
            model.load_state_dict(ck['model_state_dict'])
            return model, ck['hist']
        model.load_state_dict(ck['model_state_dict'])
        opt.load_state_dict(ck['opt_state_dict'])
        hist = ck['hist']
        start_epoch = ck['epoch'] + 1
        print(f"  [{name}] 从第{start_epoch+1}轮续训")

    x_t_val, t_val, eps_val = make_val_inputs()
    step_in_run = len(hist['train']) * len(loader)
    t0 = time.time()
    torch.manual_seed(seed)  # ★ 重置全局RNG: 消除nozero构建时多消耗的随机数, 保证两组训练噪声流一致
    epoch_bar = tqdm(range(start_epoch, epochs), desc=f"  [{name}]", unit="epoch")
    n_batches = len(loader)
    for epoch in epoch_bar:
        model.train()
        total = 0.0
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            b = x.shape[0]
            t = torch.randint(0, T, (b,), device=device)
            eps = torch.randn_like(x)
            x_t = sqrt_alpha_bars[t][:, None, None, None] * x \
                + sqrt_one_minus_alpha_bars[t][:, None, None, None] * eps
            pred = model(x_t, t)
            loss = F.mse_loss(pred, eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * b
            step_in_run += 1
            if step_in_run <= TRACE_STEPS:          # 早期batch轨迹
                hist['batch_trace'].append(loss.item())
            epoch_bar.set_postfix(epoch=f"{epoch+1}/{epochs}", batch=f"{i+1}/{n_batches}", loss=f"{loss.item():.4f}")
        tr = total / len(train_dataset)
        vl = val_loss(model, x_t_val, t_val, eps_val)
        hist['train'].append(tr)
        hist['val'].append(vl)
        diverged = not np.isfinite(tr)
        epoch_bar.set_postfix(epoch=f"{epoch+1}/{epochs}", train=f"{tr:.5f}", val=f"{vl:.5f}")
        if diverged:
            epoch_bar.write(f"  [{name}] ep{epoch+1}: !!! 发散(NaN) !!!")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'opt_state_dict': opt.state_dict(), 'hist': hist,
                    'is_final': epoch == epochs - 1}, ckpt_path)
        if diverged:
            break
    print(f"  [{name}] 完成, 用时 {time.time()-t0:.0f}s")
    return model, hist


# ============================================================
# 评估一个run的三项指标
# ============================================================
def evaluate_one(model, depth, init, lr, seed):
    name = run_name(depth, init, lr, seed)
    eval_path = os.path.join(SAVE_DIR, name + '_eval.pth')
    if os.path.exists(eval_path):
        return torch.load(eval_path, map_location=device, weights_only=False)
    gen = ddpm_sample(model, N_SAMPLE)
    feat_gen = to_feat(gen)
    pool = torch.stack([test_dataset[i][0] for i in range(NN_POOL)])
    feat_pool = to_feat(pool)
    res = {
        'pixel_frechet': frechet_distance(feat_gen, feat_pool),
        'nn_psnr': None,  # 下方在GPU上计算并覆盖
    }
    res['nn_psnr'] = nn_psnr(((gen + 1) / 2).flatten(1).to(device),
                             ((pool + 1) / 2).flatten(1).to(device))
    # 保存少量样例供目视检查
    res['sample_grid'] = gen[:64].numpy()
    torch.save(res, eval_path)
    print(f"  [{name}] 采样评估: pixel-FD={res['pixel_frechet']:.3f}, "
          f"NN-PSNR={res['nn_psnr']:.2f}dB")
    return res


# ============================================================
# 主循环
# ============================================================
print(f"\n{'='*64}")
print("开始训练矩阵: depth × init × seed (× lr)")
print("=" * 64)

results = {}   # (depth, init, lr) -> {seed: (hist, eval)}
for lr in LRS:
    for depth in DEPTHS:
        for init in INITS:
            for seed in SEEDS:
                key = (depth, init, lr)
                results.setdefault(key, {})
                model, hist = train_one(depth, init, lr, seed)
                ev = evaluate_one(model, depth, init, lr, seed)
                results[key][seed] = (hist, ev)


# ============================================================
# 汇总与统计(配对: 同seed相减)
# ============================================================
def paired_stats(depth, lr, metric):
    """返回末轮配对差值 mean±std (nozero - zero; 正=zero更优)"""
    diffs = []
    for seed in SEEDS:
        z = results[(depth, 'zero', lr)][seed][1][metric]
        n = results[(depth, 'nozero', lr)][seed][1][metric]
        diffs.append(n - z)
    diffs = np.array(diffs)
    return diffs


def instability(hist):
    """早期batch轨迹的相邻step loss变化幅度均值(前400步, 去掉前50步陡降)"""
    bt = np.array(hist['batch_trace'][50:TRACE_STEPS])
    if len(bt) < 2:
        return float('nan')
    return float(np.abs(np.diff(bt)).mean())


print(f"\n{'='*64}")
print("指标汇总 (差值 = nozero - zero; 正数表示Zero更优)")
print("=" * 64)
summary = {}
for lr in LRS:
    for depth in DEPTHS:
        z_val = np.mean([results[(depth, 'zero', lr)][s][0]['val'][-1] for s in SEEDS])
        n_val = np.mean([results[(depth, 'nozero', lr)][s][0]['val'][-1] for s in SEEDS])
        d_val = paired_stats(depth, lr, 'pixel_frechet')
        d_psnr = paired_stats(depth, lr, 'nn_psnr')
        z_inst = np.mean([instability(results[(depth, 'zero', lr)][s][0]) for s in SEEDS])
        n_inst = np.mean([instability(results[(depth, 'nozero', lr)][s][0]) for s in SEEDS])
        summary[f'depth{depth}_lr{lr:g}'] = {
            'final_val_zero': float(z_val), 'final_val_nozero': float(n_val),
            'paired_diff_pixel_frechet_mean': float(d_val.mean()),
            'paired_diff_pixel_frechet_std': float(d_val.std(ddof=0)) if len(SEEDS) > 1 else None,
            'paired_diff_nn_psnr_mean': float(d_psnr.mean()),
            'early_instability_zero': float(z_inst),
            'early_instability_nozero': float(n_inst),
        }
        s = summary[f'depth{depth}_lr{lr:g}']
        print(f"\n  depth={depth}, lr={lr:g}:")
        print(f"    末轮验证Loss:  Zero={s['final_val_zero']:.5f}  NoZero={s['final_val_nozero']:.5f}")
        print(f"    pixel-FD差值:  {s['paired_diff_pixel_frechet_mean']:+.3f} "
              f"(±{s['paired_diff_pixel_frechet_std'] or 0:.3f})  正=Zero生成更真实")
        print(f"    NN-PSNR差值:   {s['paired_diff_nn_psnr_mean']:+.2f} dB      正=Zero更优")
        print(f"    早期振荡幅度:  Zero={s['early_instability_zero']:.5f}  "
              f"NoZero={s['early_instability_nozero']:.5f}")


# ============================================================
# 图1: 验证Loss收敛曲线(均值±区间, 按深度分panel)
# ============================================================
fig, axes = plt.subplots(1, len(DEPTHS), figsize=(7 * len(DEPTHS), 5), sharey=True)
if len(DEPTHS) == 1:
    axes = [axes]
for ax, depth in zip(axes, DEPTHS):
    for init, color, marker in [('zero', 'tab:blue', 'o'), ('nozero', 'tab:orange', 's')]:
        curves = np.array([results[(depth, init, LRS[0])][s][0]['val'] for s in SEEDS])
        m = curves.mean(0)
        ax.plot(range(1, len(m) + 1), m, f'-{marker}', ms=3, color=color,
                label=f"adaLN-{'Zero' if init == 'zero' else '随机初始化'}")
        if len(SEEDS) > 1 and curves.std(0).sum() > 0:
            half = curves.std(0)
            ax.fill_between(range(1, len(m) + 1), m - half, m + half, color=color, alpha=0.15)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_title(f'depth={depth} (hidden={HIDDEN})', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
axes[0].set_ylabel('固定验证Loss (eps-MSE)', fontsize=12)
plt.suptitle('验证1: 验证Loss收敛 — 深度作为自变量, 多种子均值±std', fontsize=14, y=1.02)
plt.tight_layout()
p1 = os.path.join(SAVE_DIR, '验证1_验证Loss收敛曲线.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图1已保存: {p1}")

# ============================================================
# 图2: 早期batch loss轨迹(检验"随机初始化振荡明显")
# ============================================================
fig, axes = plt.subplots(1, len(DEPTHS), figsize=(7 * len(DEPTHS), 5), sharey=True)
if len(DEPTHS) == 1:
    axes = [axes]
for ax, depth in zip(axes, DEPTHS):
    for init, color, ls in [('zero', 'tab:blue', '-'), ('nozero', 'tab:orange', '--')]:
        traces = np.array([results[(depth, init, LRS[0])][s][0]['batch_trace'][:TRACE_STEPS]
                           for s in SEEDS])
        m = traces.mean(0)
        # 滑动平均(窗口20)突出振荡而非整体下降
        k = 20
        sm = np.convolve(m, np.ones(k) / k, mode='valid')
        resid = m[k - 1:] - sm
        ax.plot(range(1, len(sm) + 1), sm, color=color, ls=ls,
                label=f"{'Zero' if init == 'zero' else '随机'} (振荡σ={resid.std():.4f})")
    ax.set_xlabel('训练step (前400步)', fontsize=12)
    ax.set_title(f'depth={depth}: 早期batch loss(平滑)与振荡幅度', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
axes[0].set_ylabel('batch loss', fontsize=12)
plt.suptitle('验证2: 早期训练稳定性 — 直接检验"随机初始化振荡明显"断言', fontsize=14, y=1.02)
plt.tight_layout()
p2 = os.path.join(SAVE_DIR, '验证2_早期batch振荡对比.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f"图2已保存: {p2}")

# ============================================================
# 图3: 采样质量对比(pixel-Fréchet 与 NN-PSNR)
# ============================================================
labels = [f"d={d}\n{m}" for d in DEPTHS for m in ['zero', 'nozero']]
fd_vals = [np.mean([results[(d, m, LRS[0])][s][1]['pixel_frechet'] for s in SEEDS])
           for d in DEPTHS for m in ['zero', 'nozero']]
ps_vals = [np.mean([results[(d, m, LRS[0])][s][1]['nn_psnr'] for s in SEEDS])
           for d in DEPTHS for m in ['zero', 'nozero']]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = ['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange']
axes[0].bar(labels, fd_vals, color=colors, alpha=0.75)
axes[0].set_title('像素Fréchet距离 ↓ (8×8特征, FID代理)', fontsize=12)
axes[0].grid(alpha=0.3, axis='y')
axes[1].bar(labels, ps_vals, color=colors, alpha=0.75)
axes[1].set_title('最近邻PSNR ↑ (dB)', fontsize=12)
axes[1].grid(alpha=0.3, axis='y')
plt.suptitle('验证3: DDPM采样质量 (blue=adaLN-Zero, orange=随机初始化)', fontsize=14, y=1.03)
plt.tight_layout()
p3 = os.path.join(SAVE_DIR, '验证3_采样质量对比.png')
plt.savefig(p3, dpi=150, bbox_inches='tight')
plt.close()
print(f"图3已保存: {p3}")

# ============================================================
# 自动判读(诚实版结论)
# ============================================================
print(f"\n{'='*64}")
print("★ 判读结论")
print("=" * 64)
d_deep = summary.get(f'depth{DEPTHS[-1]}_lr{LRS[0]:g}', {})
d_shallow = summary.get(f'depth{DEPTHS[0]}_lr{LRS[0]:g}', {})


def verdict_line(tag, s):
    if not s:
        return
    fd = s['paired_diff_pixel_frechet_mean']
    val = s['final_val_nozero'] - s['final_val_zero']
    inst = s['early_instability_nozero'] - s['early_instability_zero']
    print(f"  [{tag}] 验证Loss方向: {'Zero优' if val > 0 else 'NoZero优'} ({val:+.5f}); "
          f"pixel-FD: {'Zero优' if fd > 0 else 'NoZero优'} ({fd:+.3f}); "
          f"早期振荡: {'NoZero更大→支持稳定论' if inst > 0 else 'Zero更大/相当→不支持'} ({inst:+.5f})")


verdict_line(f"depth={DEPTHS[0]} (浅)", d_shallow)
verdict_line(f"depth={DEPTHS[-1]} (深)", d_deep)

n_seeds = len(SEEDS)
print(f"""
判读注意(不可省略):
  1. 本次共 {n_seeds} 个种子/run。配对差值的±std在{n_seeds}个样本下极不可靠,
     |mean| < 2×std 时应视为"无差异"; 欲下强结论请至少3~5个种子。
  2. pixel-Fréchet(8×8)与NN-PSNR只是FID的粗代理, 它们对纹理/结构失真
     不敏感; 若三者方向互相矛盾, 结论只能是"不确定"。
  3. 若所有指标均无显著差异: 正确表述是"MNIST+小模型+短训练设定下无法
     复现论文级adaLN-Zero优势", 而不是"adaLN-Zero无效"——这正是原实验
     犯过的方向性错误(把检验能力不足当成被检验对象的属性)。
  4. 若希望进一步逼近论文设定: 换CIFAR-10/FashionGen子集、hidden加到384、
     训练>100k step、用真正的FID(Inception-v3, 需在[0,1]上预训练的变体)。
""")

with open(os.path.join(SAVE_DIR, '验证结果_summary.json'), 'w', encoding='utf-8') as f:
    json.dump({'config': {'depths': DEPTHS, 'inits': INITS, 'seeds': SEEDS, 'lrs': LRS,
                          'hidden': HIDDEN, 'epochs_by_depth': EPOCHS_BY_DEPTH,
                          'quick_mode': QUICK_MODE},
               'summary': summary}, f, ensure_ascii=False, indent=2, default=float)
print(f"数值汇总已保存: {os.path.join(SAVE_DIR, '验证结果_summary.json')}")
print("\n验证实验15.4-1(增强版) 完成!")
