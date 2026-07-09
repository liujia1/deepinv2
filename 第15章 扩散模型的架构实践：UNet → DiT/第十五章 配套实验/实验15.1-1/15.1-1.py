# -*- coding: utf-8 -*-
"""
实验15.1-1 扩散UNet深度架构实现（D9）
对应章节：15.1.2（UNet架构）、15.2.2（正弦位置编码）、15.4.1（UNet归纳偏置）

知识点:
  - ConvNext块：深度可分离卷积 + MLP时间步注入
  - 线性注意力 vs 全注意力的复杂度对比 O(N·d²) vs O(N²)
  - PreNorm + Residual 模式
  - 完整DiffusionUNet（D9风格）的训练与收敛

实验内容:
  步骤1: ConvNext块架构解析（参数量、输入/输出形状）
  步骤2: 注意力机制复杂度对比（Full vs Linear）
  步骤3: PreNorm模式演示
  步骤4: 完整UNet（D9）训练（MNIST, 5 epoch）
  步骤5: 各组件消融贡献总结

素材来源:
  - 15.3.py（参考实验，深扩散UNet）
  - scripts/unet.py（Diffusion_models_tutorial-main）

运行前提: PyTorch, GPU加速推荐（MNIST + 较深网络）
        数据集: MNIST（torchvision自动下载, 约11MB）
"""

import sys
import os
import io
import time
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免弹出窗口
import matplotlib.pyplot as plt
import warnings
import logging
from functools import partial

# 设置控制台输出为 UTF-8（Windows 下避免中文乱码）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默 matplotlib 相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====== 中文字体配置（兼容本地和Google Colab） ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验15.1-1')
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

# 设置随机种子
np.random.seed(42)

import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from einops import rearrange
from tqdm import tqdm

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"实验15.1-1: 扩散UNet深度架构实现（D9）")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 训练")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")

# Checkpoint路径
CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'diffusion_unet_mnist_checkpoint.pth')


# ============================================================
# 网络架构
# ============================================================
class SinusoidalPositionEmbeddings(nn.Module):
    """正弦时间步嵌入（15.2.2节）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ConvNextBlock(nn.Module):
    """ConvNext块: 深度可分离卷积 + 时间步注入 + 残差

    核心设计（15.1.2节）:
      - ds_conv: 7×7 深度可分离卷积，捕获大感受野
      - 时间步通过 MLP 注入到通道维度
      - 残差连接稳定训练
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if time_emb_dim is not None else None
        )
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if self.mlp is not None and time_emb is not None:
            h = h + self.mlp(time_emb)[:, :, None, None]
        return self.net(h) + self.res_conv(x)


class Attention(nn.Module):
    """全注意力: O(N²) 复杂度, 精确但显存消耗大"""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        # ★ 数值稳定性保护: 空特征图（h=0或w=0）直接返回零张量
        if h == 0 or w == 0:
            import warnings
            warnings.warn(f"Attention收到空特征图: shape={x.shape}, 返回零张量")
            return torch.zeros_like(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv
        )
        q = q * self.scale
        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        # 数值稳定性: 减去最大值避免 softmax 溢出
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """线性注意力: O(N·d²) 复杂度, 显存友好但精度略低"""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    """PreNorm: 在注意力/MLP之前做归一化, 稳定深层训练"""
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class Residual(nn.Module):
    """残差连接: x + fn(x)"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    """2×上采样: ConvTranspose2d"""
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    """2×下采样: 卷积stride=2"""
    return nn.Conv2d(dim, dim, 4, 2, 1)


class DiffusionUNet(nn.Module):
    """深度扩散UNet（D9风格，参考 lilianweng/diffusion-models 论文实现）

    架构: init_conv → 下采样块（ConvNext×2 + LinearAttention + Downsample）×N
                  → 中间块（ConvNext + FullAttention + ConvNext）
                  → 上采样块（ConvNext×2 + LinearAttention + Upsample）×N
                  → final_conv

    关键设计（15.1.2/15.4.1节）:
      - ConvNext块提供大感受野 + 时间步注入
      - LinearAttention（而非FullAttention）降低显存消耗
      - PreNorm + Residual 模式稳定深层训练
      - 时间步通过正弦位置编码 + MLP 注入（15.2.2节）
    """
    def __init__(
        self,
        dim=64,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()
        self.channels = channels
        init_dim = init_dim or dim // 3 * 2
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ConvNextBlock, mult=convnext_mult) if use_convnext else None

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity(),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity(),
            ]))

        out_dim = out_dim or channels
        self.final_conv = nn.Sequential(
            block_klass(dim, dim),
            nn.Conv2d(dim, out_dim, 1),
        )

    def forward(self, x, time):
        x = self.init_conv(x)
        t = self.time_mlp(time) if self.time_mlp is not None else None

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


# ============================================================
# 步骤1: ConvNext块架构解析
# ============================================================
print(f"\n{'='*60}")
print("步骤1: ConvNext块架构解析")
print(f"{'='*60}")
print("""
[ConvNext块设计要点]
  - ds_conv: 7×7 深度可分离卷积, 大感受野低参数量
  - MLP时间步注入: h + mlp(t)[:, :, None, None]
  - 残差连接: out = net(h) + res_conv(x)
  - 优势: 相比传统ResNet, ConvNext在视觉任务上更优
""")

block = ConvNextBlock(64, 128, time_emb_dim=256)
x_test = torch.randn(2, 64, 32, 32)
t_test = torch.randn(2, 256)
out = block(x_test, t_test)
print(f"ConvNextBlock: input {tuple(x_test.shape)} -> output {tuple(out.shape)}")
print(f"  ds_conv 参数量: {sum(p.numel() for p in block.ds_conv.parameters()):,}")
print(f"  net 参数量:     {sum(p.numel() for p in block.net.parameters()):,}")
print(f"  res_conv 参数量: {sum(p.numel() for p in block.res_conv.parameters()):,}")


# ============================================================
# 步骤2: 注意力机制复杂度对比
# ============================================================
print(f"\n{'='*60}")
print("步骤2: 注意力机制复杂度对比")
print(f"{'='*60}")
print("""
[Full Attention vs Linear Attention]
  - Full:    注意力矩阵 N×N, 复杂度 O(N²)
  - Linear:  上下文向量 N×d, 复杂度 O(N·d²)
  - 当 N >> d 时（如64×64特征图）, Linear显著节省显存
""")

attn_full = Attention(64, heads=4, dim_head=32)
attn_linear = LinearAttention(64, heads=4, dim_head=32)
print(f"Full Attention 参数量:    {sum(p.numel() for p in attn_full.parameters()):,}")
print(f"Linear Attention 参数量:  {sum(p.numel() for p in attn_linear.parameters()):,}")

for size in [16, 32, 64]:
    x_s = torch.randn(1, 64, size, size)
    N = size * size
    d_head = 32
    print(f"  特征图 {size}×{size} (N={N}):")
    print(f"    Full Attn  复杂度: O(N^2)    = O({N**2:,})")
    print(f"    Linear Attn 复杂度: O(N·d^2) = O({N * d_head**2:,})")
    if N > d_head:
        ratio = (N**2) / (N * d_head**2)
        print(f"    节省比: {ratio:.1f}×")


# ============================================================
# 步骤3: PreNorm模式演示
# ============================================================
print(f"\n{'='*60}")
print("步骤3: PreNorm模式演示")
print(f"{'='*60}")
print("""
[PreNorm vs PostNorm]
  - PreNorm:  y = x + fn(Norm(x))   归一化在注意力/MLP之前
  - PostNorm: y = Norm(x + fn(x))   归一化在残差之后
  - 深层网络中PreNorm训练更稳定, 是现代Transformer的默认选择
""")

prenorm = PreNorm(64, Attention(64))
x_norm = torch.randn(2, 64, 16, 16)
out_norm = prenorm(x_norm)
print(f"PreNorm: input {tuple(x_norm.shape)} -> output {tuple(out_norm.shape)}")


# ============================================================
# 步骤4: 完整UNet（D9）训练
# ============================================================
print(f"\n{'='*60}")
print("步骤4: 完整UNet（D9）训练")
print(f"{'='*60}")

model = DiffusionUNet(
    dim=64,
    channels=1,
    dim_mults=(1, 2, 4, 8),
    with_time_emb=True,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"DiffusionUNet 参数量: {total_params:,} ({total_params/1e6:.1f}M)")

# 数据准备
print("\n加载MNIST数据集...")
data_dir = os.path.join(SAVE_DIR, 'data')
os.makedirs(data_dir, exist_ok=True)
# 注意: MNIST torchvision自动下载, 约11MB
# 归一化到 [-1, 1] 区间以匹配标准高斯噪声先验
transform = transforms.Compose([
    # Resize 64: MNIST原分辨率28×28, 放大至64×64使三次下采样后中间层落在8×8
    # 与dim_mults=(1,2,4,8)的深层UNet设计匹配
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
print(f"  训练集大小: {len(train_dataset)}")

# 噪声调度（DDPM线性）
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars = torch.sqrt(alpha_bars).to(device)
sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
num_epochs = 20

# Checkpoint加载逻辑（支持 resume + 最终权重检测）
start_epoch = 0
is_final = False
loss_history = []

if os.path.exists(CHECKPOINT_PATH):
    print(f"\n检测到已保存的模型: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if 'loss_history' not in checkpoint:
        raise RuntimeError(
            f"检测到旧版本 checkpoint (缺少 'loss_history' 字段):\n"
            f"  {CHECKPOINT_PATH}\n"
            f"请删除该文件后重新训练."
        )
    if checkpoint.get('is_final', False):
        print(f"✓ 这是最终训练完成的模型, 直接加载, 跳过训练过程")
        print(f"  训练轮数: {checkpoint['epoch']+1}")
        print(f"  最终损失: {checkpoint['loss']:.4f}")
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            raise RuntimeError(
                f"检测到 checkpoint 与当前模型架构不兼容:\n{e}\n"
                f"可能原因: 模型架构已更新 (dim/dim_mults 等变更).\n"
                f"请删除 checkpoint 文件后重新训练:\n"
                f"  {CHECKPOINT_PATH}"
            )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_history = checkpoint.get('loss_history', [])
        start_epoch = checkpoint['epoch'] + 1
        is_final = True
    else:
        print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            raise RuntimeError(
                f"检测到 checkpoint 与当前模型架构不兼容:\n{e}\n"
                f"可能原因: 模型架构已更新.\n"
                f"请删除 checkpoint 文件后重新训练:\n"
                f"  {CHECKPOINT_PATH}"
            )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_history = checkpoint.get('loss_history', [])
        start_epoch = checkpoint['epoch'] + 1

# 训练循环
if not is_final:
    # 快速验证模式: 设置环境变量 QUICK_TEST=1 可仅训练1轮
    import os as _os
    if _os.environ.get('QUICK_TEST', '') == '1':
        num_epochs = 1
        print(f"\n[快速验证模式] 仅训练 {num_epochs} 轮")

    # 边界保护
    if start_epoch >= num_epochs:
        print(f"  注意: start_epoch({start_epoch}) >= num_epochs({num_epochs}), 无需继续训练")
        print(f"        该判断基于当前num_epochs设置，非模型真实收敛状态")
        if not loss_history:
            print(f"  警告: 无历史损失记录")
        is_final = True

    if not is_final:
        print(f"\n开始训练 (共 {num_epochs} 轮, 当前从第 {start_epoch+1} 轮开始)...")
        t_start = time.time()

        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, unit='batch')
            for x_batch, _ in pbar:
                x_batch = x_batch.to(device)
                b = x_batch.size(0)

                t = torch.randint(0, T, (b,), device=device)
                eps = torch.randn_like(x_batch)
                x_t = (sqrt_alpha_bars[t].view(b, 1, 1, 1) * x_batch +
                       sqrt_one_minus_alpha_bars[t].view(b, 1, 1, 1) * eps)

                pred = model(x_t, t)
                loss = F.mse_loss(pred, eps)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            print(f"  Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}")

            # 每个 epoch 结束后保存中间 checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'loss_history': loss_history,
                'is_final': False
            }, CHECKPOINT_PATH)

        t_elapsed = time.time() - t_start
        print(f"\n训练完成, 总耗时: {t_elapsed:.1f} 秒, 最终损失: {loss_history[-1]:.4f}")

        # 保存最终 checkpoint
        if loss_history:
            torch.save({
                'epoch': num_epochs - 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_history[-1],
                'loss_history': loss_history,
                'is_final': True
            }, CHECKPOINT_PATH)
            print(f"✓ 训练完成, 模型已保存: {CHECKPOINT_PATH}")
        else:
            print(f"⚠ 警告: loss_history 为空, 未保存最终 checkpoint")
else:
    print(f"\n使用已训练完成的 DiffusionUNet 模型, 跳过训练过程")

# 训练收敛曲线
if loss_history:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history)+1), loss_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$\mathcal{L}_{\mathrm{MSE}}$')
    ax.set_title(r'步骤4: DiffusionUNet 训练收敛曲线')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, '步骤4_训练收敛.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: 步骤4_训练收敛.png")
else:
    print(f"\n无训练历史, 跳过训练曲线绘制")


# ============================================================
# 步骤4+1: 训练后采样结果
# ============================================================
print(f"\n{'='*60}")
print("步骤4+1: 训练后采样结果")
print(f"{'='*60}")

@torch.no_grad()
def sample_ddpm(model, shape, T, alphas, alpha_bars, betas, device):
    """DDPM 标准采样: 从纯噪声逐步去噪生成图像"""
    model.eval()
    x = torch.randn(shape, device=device)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
    pbar = tqdm(reversed(range(T)), total=T, desc="DDPM 采样", leave=False, unit='step')
    for t in pbar:
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)
        alpha = alphas[t]
        alpha_bar = alpha_bars[t]
        beta = betas[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0.0

        x = (
            1 / torch.sqrt(alpha) *
            (x - (1 - alpha) / sqrt_one_minus_alpha_bars[t] * predicted_noise)
            + torch.sqrt(beta) * noise
        )
    return x

num_samples = 16
print(f"  正在生成 {num_samples} 张样本 (DDPM 采样, T={T})...")
# ★ 注意：dim_mults=(1,2,4,8) 需要图像尺寸能下采样4次：64→32→16→8→4
samples = sample_ddpm(model, (num_samples, 1, 64, 64), T, alphas, alpha_bars, betas, device)
# 反归一化到 [0, 1]
samples = (samples + 1) / 2.0
samples = torch.clamp(samples, 0.0, 1.0)

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(samples[i, 0].cpu().numpy(), cmap='gray')
    ax.axis('off')
plt.suptitle('步骤4+1: 采样生成结果 (4×4)')
plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤4+1_采样结果.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  图表已保存: 步骤4+1_采样结果.png")


# ============================================================
# 步骤5: 各组件消融贡献总结
# ============================================================
print(f"\n{'='*60}")
print("步骤5: 各组件消融贡献总结")
print(f"{'='*60}")
print("""
[各组件对扩散去噪的贡献分析]

  完整UNet (ConvNext + Attention + PreNorm): 基线架构

  当前配置:
    - dim=64, dim_mults=(1,2,4,8): 通道数64→128→256→512
    - T=500: 采样步数（相比T=1000减少误差累积）
    - num_epochs=20: 充分训练以收敛深层网络

  设计考量:
    - Attention: 实践中通常观察到其对生成质量影响较大，移除后细节捕获能力明显下降
    - PreNorm: 通过在残差前归一化稳定深层训练，是现代UNet/Transformer的事实标准
    - ConvNext深度可分离卷积: 提供7×7大感受野，兼顾参数效率与空间建模能力

  说明: 本实验未进行定量消融实验，上述为架构设计的定性分析
""")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验15.1-1 总结")
print(f"{'='*60}")
print("""
1. ConvNext块（步骤1）
   - 深度可分离7×7卷积 + 时间步MLP注入 + 残差
   - 大感受野 + 低参数量, 兼顾效率与表达力

2. 注意力复杂度（步骤2）
   - Full Attention:    O(N²)    精确但显存大
   - Linear Attention:  O(N·d²)  显存友好, 实际可行

3. PreNorm模式（步骤3）
   - 归一化在残差之前, 深层网络训练稳定
   - 是现代UNet/Transformer的事实标准

4. 完整UNet训练（步骤4）
   - MNIST数据集（torchvision自动下载）
   - 配置: dim=64, dim_mults=(1,2,4,8), 通道数64→128→256→512
   - DDPM线性噪声调度, T=500（减少采样误差累积）
   - 训练轮数: 20 epochs（充分收敛深层网络）
   - 支持resume + 最终权重自动跳过训练

5. 架构分析（步骤5）
   - Attention对生成质量影响显著（定性分析）
   - PreNorm稳定深层训练，现代架构标配
   - ConvNext深度可分离卷积兼顾效率与空间建模
""")

print(f"\n{'='*60}")
print("实验15.1-1 完成!")
print(f"{'='*60}")
