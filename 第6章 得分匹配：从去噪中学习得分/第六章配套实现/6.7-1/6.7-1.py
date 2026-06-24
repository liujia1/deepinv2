# -*- coding: utf-8 -*-
"""
实验6.7-1 SMLD完整实现——从零训练到图像生成
对应章节: 6.7 用学习到的得分驱动采样

知识点:
  - NCSN多噪声水平训练 (MNIST + ConditionalBatchNorm2d)
  - DSM训练目标与 λ(σ) = σ² 加权
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
from tqdm import tqdm
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
L = 10  # 噪声水平数量
sigma_min = 0.01
sigma_max = 1.0
sigmas = torch.tensor([sigma_min * (sigma_max / sigma_min) ** (i / (L - 1))
                       for i in range(L)])
sigmas_dev = sigmas.to(device)
print(f"\n噪声调度 (L={L}): σ = {[f'{s:.4f}' for s in sigmas.tolist()]}")


# ============================================================
# NCSN网络架构 (ConditionalBatchNorm2d + UNet风格CNN)
# ============================================================
class ConditionalBatchNorm2d(torch.nn.Module):
    """条件批归一化: 将噪声水平 σ 编码后注入每个残差块

    与 NCSN 官方实现一致 (songyang/ncsn):
    - BatchNorm2d(affine=False): 去除可学习 affine 参数
    - 通过 Embedding 层根据噪声水平注入 gamma/beta
    - 训练时依赖 batch 统计量, 需要足够大的 batch size
    """
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
    """NCSN得分网络 (MNIST, UNet风格 + 条件批归一化 + skip connection)

    输入: 含噪图像 x (B,1,28,28) + 噪声水平索引 sigma_idx (B,)
    输出: 得分估计 s_θ(x, σ) (B,1,28,28)
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
        # decoder 输入通道包含 skip connection 的特征
        self.dec1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            torch.nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            ConditionalBatchNorm2d(base_ch * 2, num_sigmas),
            torch.nn.SiLU())
        self.dec2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            torch.nn.Conv2d(base_ch * 3, base_ch, 3, padding=1),  # base_ch*2 + base_ch = base_ch*3
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
        d1 = torch.cat([d1, e2], dim=1)  # skip connection: (B, base_ch*4, 14, 14)
        d1 = self.dec1[1](d1)           # Conv2d
        d1 = self.dec1[2](d1, sigma_idx)  # ConditionalBatchNorm2d
        d1 = self.dec1[3](d1)           # SiLU  -> (B, base_ch*2, 14, 14)
        d2 = self.dec2[0](d1)           # Upsample: 14->28
        d2 = torch.cat([d2, e1], dim=1)  # skip connection: (B, base_ch*4, 28, 28)
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
print("  使用 L 个噪声水平 σ₁ > σ₂ > ... > σ_L 训练NCSN,")
print("  大噪声'填满'低密度区域, 小噪声保留分布细节.")
print("  DSM目标: σ² · ||s_θ(x̃, σ) + z/σ||²")
print("  λ(σ) = σ² 加权使各噪声水平贡献均匀.")

# 加载MNIST数据集
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root=os.path.join(SAVE_DIR, 'data'), train=True,
                               download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

# 初始化模型
model = NCSN_MNIST(num_sigmas=L, base_ch=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

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
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            raise RuntimeError(
                f"检测到 checkpoint 与当前模型架构不兼容:\n{e}\n"
                f"可能原因: 模型架构已更新 (如 base_ch 或 skip connection 变更).\n"
                f"请删除 checkpoint 文件后重新训练:\n"
                f"  {CHECKPOINT_PATH}"
            )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        start_epoch = checkpoint['epoch'] + 1
        is_final = True
    else:
        print(f"检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            raise RuntimeError(
                f"检测到 checkpoint 与当前模型架构不兼容:\n{e}\n"
                f"可能原因: 模型架构已更新 (如 base_ch 或 skip connection 变更).\n"
                f"请删除 checkpoint 文件后重新训练:\n"
                f"  {CHECKPOINT_PATH}"
            )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        start_epoch = checkpoint['epoch'] + 1

# 训练循环
if not is_final:
    num_epochs = 200  # 训练轮数 (NCSN论文中训练数百轮, 80轮通常不够)
    # 快速验证模式: 设置环境变量 QUICK_TEST=1 可仅训练3轮
    import os as _os
    if _os.environ.get('QUICK_TEST', '') == '1':
        num_epochs = 3
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
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False, unit='batch')
            for x_batch, _ in pbar:
                x_batch = x_batch.to(device)
                b = x_batch.size(0)

                # 随机选择噪声水平
                sigma_idx = torch.randint(0, L, (b,)).to(device)
                sigma_i = sigmas_dev[sigma_idx].view(b, 1, 1, 1)

                # 加噪声
                z = torch.randn_like(x_batch)
                x_noisy = x_batch + sigma_i * z

                # DSM目标: ||σ · s_θ(x̃, σ) + z||²
                # 等价于 σ² · ||s_θ(x̃, σ) + z/σ||², 但数值更稳定
                pred = model(x_noisy, sigma_idx)
                loss = torch.mean((sigma_i * pred + z) ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=f'{loss.item():.4f}')

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            if (epoch + 1) % (10 if num_epochs >= 10 else 1) == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: DSM Loss = {avg_loss:.4f}, LR = {current_lr:.6f}")

                # 保存中间checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'train_losses': train_losses,
                    'is_final': False
                }, CHECKPOINT_PATH)

                # 每50轮做一次中间采样, 监控生成质量
                if (epoch + 1) % 50 == 0:
                    print(f"  [中间采样] Epoch {epoch+1}, 生成样本中...")
                    model.eval()
                    mid_samples = annealed_langevin_sample(model, sigmas_dev, n_samples=8, T=50, eps=2e-5)
                    fig_mid, axes_mid = plt.subplots(2, 4, figsize=(6, 3))
                    for k in range(8):
                        r, c = k // 4, k % 4
                        axes_mid[r, c].imshow(mid_samples[k, 0].cpu().numpy(), cmap='gray')
                        axes_mid[r, c].axis('off')
                    plt.suptitle(f'中间采样 Epoch {epoch+1}')
                    plt.tight_layout()
                    mid_path = os.path.join(SAVE_DIR, f'中间采样_epoch{epoch+1:03d}.png')
                    plt.savefig(mid_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  [中间采样] 已保存: {mid_path}")
                    model.train()  # 恢复训练模式

        t_elapsed = time.time() - t_start
        print(f"\n训练完成, 最终损失: {train_losses[-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")

    # 保存最终checkpoint
    if train_losses:
        torch.save({
            'epoch': num_epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_losses[-1],
            'train_losses': train_losses,
            'is_final': True
        }, CHECKPOINT_PATH)
        print(f"✓ 训练完成, 模型已保存: {CHECKPOINT_PATH}")
    else:
        print(f"⚠ 警告: train_losses 为空, 未保存最终 checkpoint (可能 start_epoch >= num_epochs 且无历史训练记录)")

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
print("  x_{t+1} = x_t + (α_i/2) · s_θ(x_t, σ_i) + √α_i · z")
print("  步长 α_i = ε · σ_i² / σ_L² 随噪声水平自适应调整.")


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
    # NCSN 采样时使用 eval 模式: 使用 running statistics 而非 batch statistics,
    # 避免小 batch (n_samples=16) 下 BatchNorm 统计量噪声过大导致采样不稳定
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28, device=device) * sigmas[-1]  # 从最大噪声初始化
    L_local = len(sigmas)

    with torch.no_grad():
        for i in range(L_local - 1, -1, -1):
            # 步长 α_i = ε · σ_i² / σ_min²  (NCSN论文公式, 分母为最小σ)
            # 注意: sigmas[0] 是最小σ(0.01), sigmas[-1] 是最大σ(1.0)
            # 此处必须除以最小σ, 否则步长会偏小 (σ_max/σ_min)²=10⁴ 倍, 样本无法收敛
            alpha = eps * (sigmas[i] / sigmas[0]) ** 2
            alpha = torch.clamp(alpha, min=0)  # 数值保护
            sigma_idx = torch.full((n_samples,), i, dtype=torch.long, device=device)
            for t in range(T):
                score = model(x, sigma_idx)
                z = torch.randn_like(x)
                x = x + alpha / 2 * score + torch.sqrt(alpha) * z

    return torch.clamp(x, 0, 1)


n_samples = 16  # 生成样本数
# 退火Langevin采样超参数: 参考 NCSN 论文 (Song & Ermon, 2019)
# eps=2e-5, T=100 在论文中效果最佳
print(f"\n运行退火Langevin采样 (n_samples={n_samples}, T=100, ε=2e-5)...")
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

model.eval()  # 轨迹可视化也使用 eval 模式, 与 annealed_langevin_sample 保持一致
# 注意: 使用 batch=16 进行轨迹采样, 只取第一条轨迹可视化
# 原因: 单样本(batch=1)过 BatchNorm2d 时统计量噪声极大,
#       使用较大 batch 可获得更稳定的 BatchNorm 统计量
n_traj_batch = 16
x_traj = torch.randn(n_traj_batch, 1, 28, 28, device=device) * sigmas_dev[-1]
trajectory = [x_traj[0, 0].cpu().numpy()]  # 只记录第一条轨迹

# 记录关键噪声水平处的图像（动态生成）
n_checkpoints = min(6, L)  # 最多6个checkpoint
checkpoints = [int(i) for i in np.linspace(L-1, 0, n_checkpoints).astype(int)]
cp_idx = 0
with torch.no_grad():
    for i in range(L - 1, -1, -1):
        # 步长 α_i = ε · σ_i² / σ_min²  (与 annealed_langevin_sample 保持一致)
        alpha = 2e-5 * (sigmas_dev[i] / sigmas_dev[0]) ** 2
        alpha = torch.clamp(alpha, min=0)  # 数值保护（防御性编程）
        sigma_idx = torch.full((n_traj_batch,), i, dtype=torch.long, device=device)
        for t in range(100):
            score = model(x_traj, sigma_idx)
            z = torch.randn_like(x_traj)
            x_traj = x_traj + alpha / 2 * score + torch.sqrt(alpha) * z
        if cp_idx < len(checkpoints) and i == checkpoints[cp_idx]:
            trajectory.append(torch.clamp(x_traj, 0, 1)[0, 0].cpu().numpy())  # 只取第一条
            cp_idx += 1

# 轨迹可视化（动态生成标签）
sigma_labels = ['init']
for idx in checkpoints:
    sigma_labels.append(f'σ≈{sigmas[idx].item():.2f}')

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
print(f"   - 噪声调度: σ₁={sigmas[0]:.2f} → σ_L={sigmas[-1]:.4f}, 几何级数排列")
print("   - ConditionalBatchNorm2d: 将噪声水平编码后注入每个残差块")
print("   - DSM目标: σ² · ||s_θ(x̃, σ) + z/σ||²")
if train_losses:
    print(f"   - 最终训练损失: {train_losses[-1]:.6f}")
print("\n2. 退火Langevin采样 (步骤2):")
print("   - 从纯噪声开始, 从大噪声到小噪声逐步采样")
print("   - 步长 α_i = ε · σ_i² / σ_L² 自适应调整")
print("   - 生成结果: MNIST风格的手写数字图像")
print("\n3. 采样轨迹 (步骤3):")
print("   - 大噪声阶段: 粗粒度结构出现")
print("   - 中噪声阶段: 形状和纹理清晰化")
print("   - 小噪声阶段: 细节精细化")
print("\n4. SMLD与扩散模型的桥梁:")
print("   - 噪声调度 <-> 时间步, 退火Langevin <-> 逆向SDE")
print("   - NCSN = 离散化扩散模型, 第7章将展示连续化版本")
print("   - 核心洞见: 得分匹配解决了'得分从哪来'的问题,")
print("     退火Langevin解决了'如何用多时间步得分做高质量采样'的问题")

print(f"\n{'='*60}")
print("第六章配套实验完成!")