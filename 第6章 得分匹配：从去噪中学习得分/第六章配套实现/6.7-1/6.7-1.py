# -*- coding: utf-8 -*-
"""
实验6.7-1 SMLD完整实现——从零训练到图像生成
对应章节: 6.7 用学习到的得分驱动采样

知识点:
  - NCSN多噪声水平训练 (MNIST + ConditionalBatchNorm2d)
  - DSM训练目标与 $\\lambda(\\sigma) = \\sigma^2$ 加权
  - 退火Langevin动力学无条件采样
  - 采样轨迹可视化: 从大噪声到小噪声逐步去噪
  - SMLD与扩散模型的桥梁

实验内容:
  步骤1: NCSN训练 (MNIST + ConditionalBatchNorm2d)
  步骤2: 退火Langevin无条件采样生成图像
  步骤3: 采样轨迹可视化

素材来源:
  - 03-smld.ipynb (diffusion-tutorials-master)
  - 参考实验6.6.py

运行前提: PyTorch, CPU/GPU均可 (GPU加速推荐)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import io
import time
import warnings
import logging

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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
    SAVE_DIR = os.path.join(_gdrive, '实验6.7-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)

# 在Colab或本地首次运行时自动创建chinese_font.py
_chinese_font_path = os.path.join(_chinese_path, 'chinese_font.py')
if not os.path.exists(_chinese_font_path):
    print("正在创建中文字体配置模块...")
    _chinese_font_code = '''# -*- coding: utf-8 -*-
"""
中文显示支持模块 - 兼容 Windows / Linux / Colab
"""
import os
import sys
import platform
import warnings
import logging
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

def _find_chinese_font():
    candidates = []
    if platform.system() == 'Windows':
        candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    else:
        candidates = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK', 'Source Han Sans SC', 'AR PL UMing CN', 'SimHei']
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import re
    cjk_patterns = ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']
    for f in fm.ttflist:
        name_lower = f.name.lower()
        fname_lower = (os.path.basename(f.fname) if hasattr(f, 'fname') else '').lower()
        for pat in cjk_patterns:
            if re.search(pat, name_lower) or re.search(pat, fname_lower):
                return f.name
    return None

def setup_chinese_font(save_dir=None):
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    _cn_font = _find_chinese_font()
    if _cn_font:
        plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['font.family'] = 'sans-serif'
        print(f"[Font] 已检测到中文字体: {_cn_font}")
        return _cn_font
    if platform.system() != 'Windows':
        _font_url = 'https://github.com/jsntn/webfonts/raw/master/NotoSansSC-Regular.ttf'
        _font_file = os.path.join(save_dir, 'NotoSansSC-Regular.ttf')
        if os.path.exists(_font_file):
            from matplotlib.font_manager import fontManager
            fontManager.addfont(_font_file)
            plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['font.family'] = 'sans-serif'
            print(f"[Font] 已加载缓存字体: Noto Sans SC")
            return 'Noto Sans SC'
        else:
            try:
                import urllib.request
                print(f"[Font] 正在下载中文字体 NotoSansSC...")
                urllib.request.urlretrieve(_font_url, _font_file)
                from matplotlib.font_manager import fontManager
                fontManager.addfont(_font_file)
                plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
                plt.rcParams['font.family'] = 'sans-serif'
                print(f"[Font] 已下载并注册中文字体: Noto Sans SC")
                return 'Noto Sans SC'
            except Exception as e:
                print(f"[Font] 字体下载失败: {e}")
    else:
        print("[Font] 未找到中文字体")
    return None

__all__ = ['setup_chinese_font']
'''
    with open(_chinese_font_path, 'w', encoding='utf-8') as f:
        f.write(_chinese_font_code)
    print(f"[Font] 已创建字体配置模块: {_chinese_font_path}")

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

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"实验6.7-1: SMLD完整实现——从零训练到图像生成")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 训练")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")

# Checkpoint路径
CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'ncsn_mnist_checkpoint.pth')

# ============================================================
# 噪声调度: 几何级数
# ============================================================
L = 10
sigma_min = 0.01
sigma_max = 1.0
sigmas = torch.tensor([sigma_min * (sigma_max / sigma_min) ** (i / (L - 1))
                       for i in range(L)])
sigmas_dev = sigmas.to(device)
print(f"\n噪声调度 (L={L}): $\\sigma$ = {[f'{s:.4f}' for s in sigmas.tolist()]}")


# ============================================================
# NCSN网络架构 (ConditionalBatchNorm2d + UNet风格CNN)
# ============================================================
class ConditionalBatchNorm2d(torch.nn.Module):
    """条件批归一化: 将噪声水平 $\\sigma$ 编码后注入每个残差块"""
    def __init__(self, num_features, num_sigmas):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(num_features, affine=False)
        self.gamma = torch.nn.Embedding(num_sigmas, num_features)
        self.beta = torch.nn.Embedding(num_sigmas, num_features)
        self.gamma.weight.data.fill_(1.0)
        self.beta.weight.data.fill_(0.0)

    def forward(self, x, sigma_idx):
        out = self.bn(x)
        gamma = self.gamma(sigma_idx).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(sigma_idx).unsqueeze(-1).unsqueeze(-1)
        return gamma * out + beta


class NCSN_MNIST(torch.nn.Module):
    """NCSN得分网络 (MNIST, UNet风格 + 条件批归一化)

    输入: 含噪图像 x (B,1,28,28) + 噪声水平索引 sigma_idx (B,)
    输出: 得分估计 $s_\\theta(x, \\sigma)$ (B,1,28,28)
    """
    def __init__(self, num_sigmas=10, base_ch=64):
        super().__init__()
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, base_ch, 3, padding=1),
            ConditionalBatchNorm2d(base_ch, num_sigmas),
            torch.nn.SiLU())
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            ConditionalBatchNorm2d(base_ch * 2, num_sigmas),
            torch.nn.SiLU())
        self.enc3 = torch.nn.Sequential(
            torch.nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1),
            ConditionalBatchNorm2d(base_ch * 2, num_sigmas),
            torch.nn.SiLU())
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            ConditionalBatchNorm2d(base_ch * 2, num_sigmas),
            torch.nn.SiLU())
        self.dec1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            torch.nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            ConditionalBatchNorm2d(base_ch * 2, num_sigmas),
            torch.nn.SiLU())
        self.dec2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            torch.nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            ConditionalBatchNorm2d(base_ch, num_sigmas),
            torch.nn.SiLU())
        self.out_conv = torch.nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x, sigma_idx):
        e1 = self.enc1[0](x)
        e1 = self.enc1[1](e1, sigma_idx)
        e1 = self.enc1[2](e1)           # (B, base_ch, 28, 28)
        e2 = self.enc2[0](e1)
        e2 = self.enc2[1](e2, sigma_idx)
        e2 = self.enc2[2](e2)           # (B, base_ch*2, 14, 14)
        e3 = self.enc3[0](e2)
        e3 = self.enc3[1](e3, sigma_idx)
        e3 = self.enc3[2](e3)           # (B, base_ch*2, 7, 7)
        b = self.bottleneck[0](e3)
        b = self.bottleneck[1](b, sigma_idx)
        b = self.bottleneck[2](b)       # (B, base_ch*2, 7, 7)
        d1 = self.dec1[0](b)            # Upsample: 7->14
        d1 = self.dec1[1](d1)           # Conv2d
        d1 = self.dec1[2](d1, sigma_idx)  # ConditionalBatchNorm2d
        d1 = self.dec1[3](d1)           # SiLU  -> (B, base_ch, 14, 14)
        d2 = self.dec2[0](d1 + e2)      # Upsample: 14->28, skip=e2(14x14)
        d2 = self.dec2[1](d2)           # Conv2d
        d2 = self.dec2[2](d2, sigma_idx)  # ConditionalBatchNorm2d
        d2 = self.dec2[3](d2)           # SiLU  -> (B, base_ch, 28, 28)
        return self.out_conv(d2)


# ============================================================
# 步骤1: NCSN训练 (MNIST)
# ============================================================
print(f"\n{'='*60}")
print("步骤1: NCSN训练 (MNIST)")
print(f"{'='*60}")
print("\n[核心思想]")
print("  使用 $L$ 个噪声水平 $\\sigma_1 > \\sigma_2 > \\cdots > \\sigma_L$ 训练NCSN,")
print("  大噪声'填满'低密度区域, 小噪声保留分布细节.")
print("  DSM目标: $\\sigma^2 \\cdot \\|s_\\theta(\\tilde{x}, \\sigma) + z/\\sigma\\|^2$")
print("  $\\lambda(\\sigma) = \\sigma^2$ 加权使各噪声水平贡献均匀.")

# 加载MNIST数据集
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root=os.path.join(SAVE_DIR, 'data'), train=True,
                               download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

# 初始化模型
model = NCSN_MNIST(num_sigmas=L).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Checkpoint加载逻辑
start_epoch = 0
is_final = False
train_losses = []

if os.path.exists(CHECKPOINT_PATH):
    print(f"\n检测到已保存的模型: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if 'train_losses' not in checkpoint:
        raise RuntimeError(
            f"检测到旧版本 checkpoint (缺少 'train_losses' 字段):\n"
            f"  {CHECKPOINT_PATH}\n"
            f"请删除该文件后重新训练."
        )
    if checkpoint.get('is_final', False):
        print(f"✓ 这是最终训练完成的模型, 直接加载, 跳过训练过程")
        print(f"  训练轮数: {checkpoint['epoch']+1}")
        print(f"  最终损失: {checkpoint['loss']:.6f}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        start_epoch = checkpoint['epoch'] + 1
        is_final = True
    else:
        print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        start_epoch = checkpoint['epoch'] + 1

# 训练循环
if not is_final:
    num_epochs = 50
    # 快速验证模式: 设置环境变量 QUICK_TEST=1 可仅训练2轮
    import os as _os
    if _os.environ.get('QUICK_TEST', '') == '1':
        num_epochs = 2
        print(f"\n[快速验证模式] 仅训练 {num_epochs} 轮")
    print(f"\n训练 NCSN ({L} 个噪声水平, 共 {num_epochs} 轮)...")
    t_start = time.time()

    # 边界保护
    if start_epoch >= num_epochs:
        print(f"  注意: start_epoch({start_epoch}) >= num_epochs({num_epochs}), 无需继续训练")
        if not train_losses:
            print(f"  警告: 无历史损失记录")
        is_final = True

    if not is_final:
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(device)
                b = x_batch.size(0)

                # 随机选择噪声水平
                sigma_idx = torch.randint(0, L, (b,)).to(device)
                sigma_i = sigmas_dev[sigma_idx].view(b, 1, 1, 1)

                # 加噪声
                z = torch.randn_like(x_batch)
                x_noisy = x_batch + sigma_i * z

                # DSM目标: $\\sigma^2 \\cdot \\|s_\\theta(\\tilde{x}, \\sigma) + z/\\sigma\\|^2$
                pred = model(x_noisy, sigma_idx)
                target = -z / sigma_i
                loss = torch.mean((pred - target) ** 2 * sigma_i ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            if (epoch + 1) % (10 if num_epochs >= 10 else 1) == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: DSM Loss = {avg_loss:.4f}")

                # 保存中间checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'train_losses': train_losses,
                    'is_final': False
                }, CHECKPOINT_PATH)

        t_elapsed = time.time() - t_start
        print(f"\n训练完成, 最终损失: {train_losses[-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")

    # 保存最终checkpoint
    if train_losses:
        torch.save({
            'epoch': start_epoch - 1 if start_epoch >= num_epochs else num_epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses[-1],
            'train_losses': train_losses,
            'is_final': True
        }, CHECKPOINT_PATH)
        print(f"✓ 训练完成, 模型已保存: {CHECKPOINT_PATH}")
else:
    print(f"\n使用已训练完成的 NCSN 模型, 跳过训练过程")

# 训练损失曲线
if train_losses:
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(train_losses, 'b-', lw=1.5, alpha=0.7)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel(r'$\mathcal{J}_{\mathrm{DSM}}$ Loss')
    ax_loss.set_title('NCSN 训练损失曲线')
    ax_loss.grid(alpha=0.3)
    ax_loss.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤1_NCSN训练损失.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: 步骤1_NCSN训练损失.png")


# ============================================================
# 步骤2: 退火Langevin无条件采样生成图像
# ============================================================
print(f"\n{'='*60}")
print("步骤2: 退火Langevin无条件采样生成图像")
print(f"{'='*60}")
print("\n[退火Langevin动力学]")
print("  从纯噪声开始, 从大噪声到小噪声逐步采样:")
print("  $x_{t+1} = x_t + \\frac{\\alpha_i}{2} s_\\theta(x_t, \\sigma_i) + \\sqrt{\\alpha_i} z$")
print("  步长 $\\alpha_i = \\epsilon \\cdot \\sigma_i^2 / \\sigma_L^2$ 随噪声水平自适应调整.")


def annealed_langevin_sample(model, sigmas, n_samples=16, T=100, eps=2e-5):
    """退火Langevin采样

    参数:
        model: NCSN得分网络
        sigmas: 噪声调度张量 (L,)
        n_samples: 生成样本数
        T: 每个噪声水平的Langevin步数
        eps: 基础步长

    返回:
        生成的图像张量 (n_samples, 1, 28, 28)
    """
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28, device=device) * sigmas[-1]  # 从最大噪声初始化
    L_local = len(sigmas)

    with torch.no_grad():
        for i in range(L_local - 1, -1, -1):
            alpha = eps * (sigmas[i] / sigmas[-1]) ** 2
            alpha = torch.clamp(alpha, min=0)  # 数值保护
            sigma_idx = torch.full((n_samples,), i, dtype=torch.long, device=device)
            for t in range(T):
                score = model(x, sigma_idx)
                z = torch.randn_like(x)
                x = x + alpha / 2 * score + torch.sqrt(alpha) * z

    return torch.clamp(x, 0, 1)


n_samples = 16
print(f"\n运行退火Langevin采样 (n_samples={n_samples}, T=100, $\\epsilon$=2e-5)...")
t_start = time.time()
samples = annealed_langevin_sample(model, sigmas_dev, n_samples=n_samples, T=100, eps=2e-5)
t_elapsed = time.time() - t_start
print(f"采样完成, 耗时: {t_elapsed:.1f} 秒")

# 可视化生成样本
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i in range(n_samples):
    row, col = i // 4, i % 4
    axes[row, col].imshow(samples[i, 0].cpu().numpy(), cmap='gray')
    axes[row, col].axis('off')
plt.suptitle('步骤2: 退火Langevin生成样本')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_生成样本.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"图表已保存: 步骤2_生成样本.png")


# ============================================================
# 步骤3: 采样轨迹可视化
# ============================================================
print(f"\n{'='*60}")
print("步骤3: 采样轨迹可视化")
print(f"{'='*60}")
print("\n[轨迹演化]")
print("  大噪声阶段: 图像从纯噪声中出现大致的亮暗结构")
print("  中噪声阶段: 主要形状和纹理逐渐清晰")
print("  小噪声阶段: 细节精细化, 最终得到高质量图像")

model.eval()
x_traj = torch.randn(1, 1, 28, 28, device=device) * sigmas_dev[-1]  # 从最大噪声初始化
trajectory = [x_traj[0, 0].cpu().numpy()]

# 记录关键噪声水平处的图像
checkpoints = [9, 7, 5, 3, 1, 0]
cp_idx = 0
with torch.no_grad():
    for i in range(L - 1, -1, -1):
        alpha = 2e-5 * (sigmas_dev[i] / sigmas_dev[-1]) ** 2
        sigma_idx = torch.full((1,), i, dtype=torch.long, device=device)
        for t in range(100):
            score = model(x_traj, sigma_idx)
            z = torch.randn_like(x_traj)
            x_traj = x_traj + alpha / 2 * score + torch.sqrt(alpha) * z
        if cp_idx < len(checkpoints) and i == checkpoints[cp_idx]:
            trajectory.append(torch.clamp(x_traj, 0, 1)[0, 0].cpu().numpy())
            cp_idx += 1

# 轨迹可视化
sigma_labels = [r'$\mathrm{init}$', r'$\sigma\approx0.60$', r'$\sigma\approx0.36$',
                r'$\sigma\approx0.13$', r'$\sigma\approx0.05$', r'$\sigma\approx0.01$']

fig, axes = plt.subplots(1, len(trajectory), figsize=(12, 3))
for j, (ax, img) in enumerate(zip(axes, trajectory)):
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(sigma_labels[j] if j < len(sigma_labels) else '')
plt.suptitle('步骤3: 采样轨迹演化')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_采样轨迹.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"图表已保存: 步骤3_采样轨迹.png")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验6.7-1 总结")
print(f"{'='*60}")
print("\n1. NCSN训练 (步骤1):")
print(f"   - 噪声调度: $\\sigma_1={sigmas[0]:.2f} \\to \\sigma_L={sigmas[-1]:.4f}$, 几何级数排列")
print("   - ConditionalBatchNorm2d: 将噪声水平编码后注入每个残差块")
print("   - DSM目标: $\\sigma^2 \\cdot \\|s_\\theta(\\tilde{x}, \\sigma) + z/\\sigma\\|^2$")
if train_losses:
    print(f"   - 最终训练损失: {train_losses[-1]:.6f}")
print("\n2. 退火Langevin采样 (步骤2):")
print("   - 从纯噪声开始, 从大噪声到小噪声逐步采样")
print("   - 步长 $\\alpha_i = \\epsilon \\cdot \\sigma_i^2 / \\sigma_L^2$ 自适应调整")
print("   - 生成结果: MNIST风格的手写数字图像")
print("\n3. 采样轨迹 (步骤3):")
print("   - 大噪声阶段: 粗粒度结构出现")
print("   - 中噪声阶段: 形状和纹理清晰化")
print("   - 小噪声阶段: 细节精细化")
print("\n4. SMLD与扩散模型的桥梁:")
print("   - 噪声调度 $\\leftrightarrow$ 时间步, 退火Langevin $\\leftrightarrow$ 逆向SDE")
print("   - NCSN = 离散化扩散模型, 第7章将展示连续化版本")
print("   - 核心洞见: 得分匹配解决了'得分从哪来'的问题,")
print("     退火Langevin解决了'如何用多时间步得分做高质量采样'的问题")

print(f"\n{'='*60}")
print("第六章配套实验完成!")
print(f"{'='*60}")
print("实验列表:")
print("  6.1-1: 归一化常数困境与得分匹配动机")
print("  6.2-1: ESM与ISM的验证")
print("  6.3-1: 去噪得分匹配(DSM)训练与验证")
print("  6.4-1: Hutchinson迹估计与切片得分匹配(SSM)")
print("  6.5-1: 多尺度得分匹配与退火Langevin采样")
print("  6.6-1: 从去噪器中提取得分函数(Tweedie等式实践)")
print("  6.7-1: SMLD完整实现——从零训练到图像生成")
