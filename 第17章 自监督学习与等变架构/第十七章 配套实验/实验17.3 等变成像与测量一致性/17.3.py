# -*- coding: utf-8 -*-
"""
实验17.3 等变成像与测量一致性
对应知识点：17.5节（等变成像：从不完整测量中学习）、17.6节（测量一致性与等变架构）

实验内容：
Step 1: 零空间问题——inpainting中自监督损失不约束缺失区域
Step 2: 等变成像（EI）损失——利用平移对称性约束零空间
Step 3: 测量一致性（MC）损失——约束值空间方向
Step 4: MC + EI互补性验证——值空间+零空间联合约束
Step 5: 算子-等变性对照实验——不同算子的等变性验证

★原创设计：
- 用inpainting（最直观的零空间问题）展示自监督失效
- 实现EI损失利用平移不变性虚拟产生多算子
- 可视化MC约束值空间、EI约束零空间的互补效果
- 验证算子-等变性对照表中的结论

素材来源：deepinv.loss.EILoss思路、17.5节理论、17.6节MC损失
运行前提：需GPU（Colab T4即可）
"""

import os, sys, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib as mpl
import warnings
import logging

# ====== 解决中文乱码的核心代码（Windows + Linux 自动适配）======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

import platform
from matplotlib.font_manager import FontManager, FontProperties

def _find_chinese_font():
    """自动检测系统中可用的中文字体，兼容 Windows / Linux / Colab"""
    candidates = []
    if platform.system() == 'Windows':
        candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    else:
        candidates = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
            'Noto Sans CJK SC', 'Noto Sans CJK',
            'Source Han Sans SC', 'AR PL UMing CN', 'SimHei',
        ]
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import os as _os, re
    cjk_patterns = ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']
    for f in fm.ttflist:
        name_lower = f.name.lower()
        fname_lower = (_os.path.basename(f.fname) if hasattr(f, 'fname') else '').lower()
        for pat in cjk_patterns:
            if re.search(pat, name_lower) or re.search(pat, fname_lower):
                return f.name
    return None

_cn_font = _find_chinese_font()
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"[Font] 已检测到中文字体: {_cn_font}")
else:
    # Linux/Colab 未找到中文字体，尝试加载或下载 Noto Sans SC
    if platform.system() != 'Windows':
        _font_url = 'https://github.com/jsntn/webfonts/raw/master/NotoSansSC-Regular.ttf'
        _font_file = os.path.join(SAVE_DIR if 'SAVE_DIR' in dir() else '.', 'NotoSansSC-Regular.ttf')
        if os.path.exists(_font_file):
            from matplotlib.font_manager import fontManager
            fontManager.addfont(_font_file)
            plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['font.family'] = 'sans-serif'
            _cn_font = 'Noto Sans SC'
            print(f"[Font] 已加载缓存字体: {_cn_font}")
        else:
            try:
                import urllib.request
                print(f"[Font] 正在下载中文字体 NotoSansSC...")
                urllib.request.urlretrieve(_font_url, _font_file)
                from matplotlib.font_manager import fontManager
                fontManager.addfont(_font_file)
                plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
                plt.rcParams['font.family'] = 'sans-serif'
                _cn_font = 'Noto Sans SC'
                print(f"[Font] 已下载并注册中文字体: {_cn_font}")
            except Exception as e:
                print(f"[Font] 字体下载失败: {e}，中文可能显示为方框")
    else:
        print("[Font] 未找到中文字体，中文可能显示为方框")
# ========================================================

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    SAVE_DIR = os.path.join(_gdrive, '实验17_3_等变成像与测量一致性')
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"检测到 Google Drive，结果将保存至: {SAVE_DIR}")
else:
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    print(f"本地环境，结果将保存至: {SAVE_DIR}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ========================================================================
# 网络架构
# ========================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class SmallUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec3 = DoubleConv(base*4, base*2)
        self.dec2 = DoubleConv(base*2, base)
        self.out_conv = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d3 = self.up3(e3)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        return self.out_conv(d2)


# ========================================================================
# Inpainting正向算子
# ========================================================================
class InpaintingOperator:
    """图像修复算子: y = M ⊙ x + ε
    对应17.5.1节：最直观的零空间问题
    
    M是二值掩码，1=保留像素，0=缺失像素
    A的零空间 = 被遮蔽的像素位置 → 自监督损失不约束这些位置
    """
    def __init__(self, mask):
        """mask: (H, W) 二值掩码"""
        self.mask = mask  # (H, W)
    
    def A(self, x):
        """正向: y = M ⊙ x + ε"""
        mask_2d = self.mask.unsqueeze(0).unsqueeze(0).to(x.device)
        return x * mask_2d
    
    def AT(self, y):
        """伴随: A^T y = M ⊙ y (自伴随)"""
        return self.A(y)
    
    def zero_filled(self, y):
        """零填充重建"""
        return y


def create_inpainting_mask(H, W, keep_ratio=0.5, seed=42):
    """创建随机inpainting掩码
    keep_ratio: 保留像素比例
    """
    rng = np.random.RandomState(seed)
    mask = np.zeros((H, W), dtype=np.float32)
    n_keep = int(H * W * keep_ratio)
    indices = rng.choice(H * W, n_keep, replace=False)
    mask.flat[indices] = 1.0
    return torch.from_numpy(mask)


def create_random_mask_batch(batch_size, H, W, keep_ratio=0.5):
    """为每个batch样本创建不同的随机掩码
    ★原创：每张图用不同掩码（模拟MOI场景）
    """
    masks = torch.zeros(batch_size, 1, H, W)
    n_keep = int(H * W * keep_ratio)
    for i in range(batch_size):
        indices = torch.randperm(H * W)[:n_keep]
        masks[i].view(-1)[indices] = 1.0
    return masks


# ========================================================================
# 数据准备
# ========================================================================
IMG_SIZE = 32
SIGMA = 0.05   # 低噪声——强调零空间问题
KEEP_RATIO = 0.5  # 保留50%像素
BATCH_SIZE = 128
N_EPOCHS = 40
LR = 1e-3

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

print("加载MNIST数据集...")
mnist_train = datasets.MNIST(root=os.path.join(SAVE_DIR, 'mnist_data'),
                              train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root=os.path.join(SAVE_DIR, 'mnist_data'),
                             train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 固定测试掩码
test_mask = create_inpainting_mask(IMG_SIZE, IMG_SIZE, KEEP_RATIO)
inpainting_op = InpaintingOperator(test_mask)


# ========================================================================
# Step 1: 零空间问题——inpainting中自监督损失不约束缺失区域
# 对应17.5.1节：A≠I时，L_SURE仅约束值空间方向
# ========================================================================
print("\n" + "="*70)
print("Step 1: 零空间问题——自监督损失不约束缺失区域")
print("="*70)

# 训练朴素自监督: loss = ‖y - A f(y)‖²
# 这仅约束被观测到的像素（值空间），不约束缺失像素（零空间）
print("\n  训练朴素自监督 (仅MC损失)...")
model_naive = SmallUNet().to(device)
optimizer_naive = optim.Adam(model_naive.parameters(), lr=LR)

for epoch in range(N_EPOCHS):
    model_naive.train()
    for batch_x, _ in train_loader:
        batch_x = batch_x.to(device)
        # 用随机掩码做inpainting
        masks = create_random_mask_batch(batch_x.shape[0], IMG_SIZE, IMG_SIZE, KEEP_RATIO).to(device)
        y = batch_x * masks + SIGMA * torch.randn_like(batch_x) * masks  # 噪声仅在观测像素上
        
        optimizer_naive.zero_grad()
        f_y = model_naive(y)
        # ★ 朴素MC损失：仅在被观测像素上计算
        loss = ((y - f_y * masks) ** 2 * masks).sum() / masks.sum()
        loss.backward()
        optimizer_naive.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"    Epoch {epoch+1}/{N_EPOCHS}")

# 评估
def evaluate_inpainting(model, test_loader, mask, sigma=SIGMA):
    model.eval()
    psnr_vals = []
    mask_dev = mask.to(device)
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            mask_2d = mask_dev.unsqueeze(0).unsqueeze(0).expand_as(batch_x)
            y = batch_x * mask_2d + sigma * torch.randn_like(batch_x) * mask_2d
            pred = model(y).clip(0, 1)
            pred_np = pred.cpu().numpy()
            x_np = batch_x.cpu().numpy()
            for i in range(pred_np.shape[0]):
                psnr_vals.append(psnr(x_np[i, 0], pred_np[i, 0], data_range=1.0))
    return np.mean(psnr_vals)

psnr_naive = evaluate_inpainting(model_naive, test_loader, test_mask)
print(f"  朴素MC PSNR = {psnr_naive:.2f} dB")

# 可视化零空间问题
test_imgs, _ = next(iter(test_loader))
test_imgs = test_imgs[:6].to(device)
mask_2d = test_mask.unsqueeze(0).unsqueeze(0).expand_as(test_imgs).to(device)
test_y = test_imgs * mask_2d

with torch.no_grad():
    pred_naive = model_naive(test_y).clip(0, 1)

fig, axes = plt.subplots(3, 6, figsize=(15, 7))
for i in range(6):
    axes[0, i].imshow(test_imgs[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    axes[1, i].imshow(test_y[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
    axes[2, i].imshow(pred_naive[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[2, i].axis('off')

axes[0, 0].set_ylabel('干净图像x', fontsize=11)
axes[1, 0].set_ylabel('观测y=M⊙x', fontsize=11)
axes[2, 0].set_ylabel('朴素MC重建', fontsize=11)
fig.suptitle(f'Step 1: 零空间问题——仅MC损失无法约束缺失区域 (PSNR={psnr_naive:.1f}dB)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step1_nullspace.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step1_nullspace.png")
print("  问题: 缺失像素区域完全不受约束，网络可以输出任意值")


# ========================================================================
# Step 2: 等变成像（EI）损失——利用平移对称性约束零空间
# 对应17.5.4节：L_EI = E_g ‖T_g x̂ - f(A T_g x̂)‖²
# ★原创：实现平移变换的EI损失
# ========================================================================
print("\n" + "="*70)
print("Step 2: 等变成像（EI）损失——平移对称性约束零空间")
print("="*70)

def random_shift(x, max_shift=8):
    """随机平移变换 T_g
    对应17.5.3节：平移是最常见的对称性
    """
    B, C, H, W = x.shape
    dy = torch.randint(-max_shift, max_shift+1, (1,)).item()
    dx = torch.randint(-max_shift, max_shift+1, (1,)).item()
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))

def ei_loss(model, y, A_fn, n_transforms=4):
    """等变成像损失
    对应17.5.4节：Chen, Tachella & Davies (ICCV 2021)
    
    L_EI = (1/G) Σ_g ‖T_g x̂ - f(A T_g x̂)‖²
    
    其中 x̂ = f(y) 是参考重建
    A_fn: 正向算子函数
    T_g: 随机平移变换
    
    ★原创实现：简化版EI损失，使用平移变换
    """
    # 参考重建
    x_hat = model(y)
    
    total_loss = 0
    for _ in range(n_transforms):
        # 随机平移
        x_hat_shifted = random_shift(x_hat.detach())
        # 对平移后的重建做"重新测量"
        y_virtual = A_fn(x_hat_shifted)
        # 重建虚拟测量
        f_virtual = model(y_virtual)
        # 等变性约束：f(AT_g x̂) ≈ T_g x̂
        total_loss += nn.MSELoss()(f_virtual, x_hat_shifted)
    
    return total_loss / n_transforms


# 训练EI模型
print("\n  训练EI模型 (MC + EI)...")
model_ei = SmallUNet().to(device)
optimizer_ei = optim.Adam(model_ei.parameters(), lr=LR)
lambda_ei = 0.5

for epoch in range(N_EPOCHS):
    model_ei.train()
    for batch_x, _ in train_loader:
        batch_x = batch_x.to(device)
        masks = create_random_mask_batch(batch_x.shape[0], IMG_SIZE, IMG_SIZE, KEEP_RATIO).to(device)
        y = batch_x * masks + SIGMA * torch.randn_like(batch_x) * masks
        
        optimizer_ei.zero_grad()
        
        # MC损失
        f_y = model_ei(y)
        loss_mc = ((y - f_y * masks) ** 2 * masks).sum() / masks.sum()
        
        # EI损失
        def A_fn(x):
            """用当前batch的掩码做正向测量"""
            return x * masks
        
        loss_ei = ei_loss(model_ei, y, A_fn, n_transforms=4)
        
        loss = loss_mc + lambda_ei * loss_ei
        loss.backward()
        optimizer_ei.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"    Epoch {epoch+1}/{N_EPOCHS}")

psnr_ei = evaluate_inpainting(model_ei, test_loader, test_mask)
print(f"  EI (MC+EI) PSNR = {psnr_ei:.2f} dB")


# ========================================================================
# Step 3: 测量一致性（MC）损失增强
# 对应17.6.1节：L_MC = ‖y - A f(y)‖²
# ========================================================================
print("\n" + "="*70)
print("Step 3: 测量一致性损失分析")
print("="*70)

# 训练纯MC模型（更强的MC权重）
print("\n  训练增强MC模型...")
model_mc = SmallUNet().to(device)
optimizer_mc = optim.Adam(model_mc.parameters(), lr=LR)

for epoch in range(N_EPOCHS):
    model_mc.train()
    for batch_x, _ in train_loader:
        batch_x = batch_x.to(device)
        masks = create_random_mask_batch(batch_x.shape[0], IMG_SIZE, IMG_SIZE, KEEP_RATIO).to(device)
        y = batch_x * masks + SIGMA * torch.randn_like(batch_x) * masks
        
        optimizer_mc.zero_grad()
        f_y = model_mc(y)
        # 增强MC：在所有像素上计算（包括缺失位置的零填充）
        loss = nn.MSELoss()(f_y * masks, y * masks)
        loss.backward()
        optimizer_mc.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"    Epoch {epoch+1}/{N_EPOCHS}")

psnr_mc = evaluate_inpainting(model_mc, test_loader, test_mask)
print(f"  纯MC PSNR = {psnr_mc:.2f} dB")


# ========================================================================
# Step 4: MC + EI互补性验证
# 对应17.6.3节：MC约束值空间，EI约束零空间
# ★原创：可视化两种约束在不同像素位置的贡献
# ========================================================================
print("\n" + "="*70)
print("Step 4: MC + EI互补性验证")
print("="*70)

# 监督基线
print("\n  训练监督基线...")
model_sup = SmallUNet().to(device)
optimizer_sup = optim.Adam(model_sup.parameters(), lr=LR)

for epoch in range(N_EPOCHS):
    model_sup.train()
    for batch_x, _ in train_loader:
        batch_x = batch_x.to(device)
        masks = create_random_mask_batch(batch_x.shape[0], IMG_SIZE, IMG_SIZE, KEEP_RATIO).to(device)
        y = batch_x * masks + SIGMA * torch.randn_like(batch_x) * masks
        
        optimizer_sup.zero_grad()
        f_y = model_sup(y)
        loss = nn.MSELoss()(f_y, batch_x)  # ★监督：知道干净x
        loss.backward()
        optimizer_sup.step()

psnr_sup = evaluate_inpainting(model_sup, test_loader, test_mask)
print(f"  监督 PSNR = {psnr_sup:.2f} dB")

# 误差分析：分观测像素和缺失像素分别计算
def evaluate_psnr_split(model, test_loader, mask, sigma=SIGMA):
    """分别计算观测像素和缺失像素的PSNR"""
    model.eval()
    mask_dev = mask.to(device)
    obs_psnr = []
    miss_psnr = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            mask_2d = mask_dev.unsqueeze(0).unsqueeze(0).expand_as(batch_x)
            y = batch_x * mask_2d + sigma * torch.randn_like(batch_x) * mask_2d
            pred = model(y).clip(0, 1)
            
            for i in range(batch_x.shape[0]):
                x_np = batch_x[i, 0].cpu().numpy()
                p_np = pred[i, 0].cpu().numpy()
                m_np = mask_2d[i, 0].cpu().numpy()
                
                # 观测像素的MSE
                obs_pixels = m_np > 0.5
                if obs_pixels.sum() > 0:
                    mse_obs = ((x_np[obs_pixels] - p_np[obs_pixels])**2).mean()
                    obs_psnr.append(10 * np.log10(1.0 / max(mse_obs, 1e-10)))
                
                # 缺失像素的MSE
                miss_pixels = m_np < 0.5
                if miss_pixels.sum() > 0:
                    mse_miss = ((x_np[miss_pixels] - p_np[miss_pixels])**2).mean()
                    miss_psnr.append(10 * np.log10(1.0 / max(mse_miss, 1e-10)))
    
    return np.mean(obs_psnr), np.mean(miss_psnr)

# 各方法在观测/缺失像素的PSNR
methods = {
    '监督': model_sup,
    'MC+EI': model_ei,
    '纯MC': model_mc,
    '朴素MC': model_naive,
}

obs_psnrs = {}
miss_psnrs = {}
total_psnrs = {}
for name, model in methods.items():
    obs, miss = evaluate_psnr_split(model, test_loader, test_mask)
    total = evaluate_inpainting(model, test_loader, test_mask)
    obs_psnrs[name] = obs
    miss_psnrs[name] = miss
    total_psnrs[name] = total
    print(f"  {name:8s}: 总PSNR={total:.1f}dB, 观测像素={obs:.1f}dB, 缺失像素={miss:.1f}dB")

# 可视化互补性
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 总PSNR对比
names = list(methods.keys())
totals = [total_psnrs[n] for n in names]
colors_bar = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
bars = ax1.bar(names, totals, color=colors_bar, width=0.5)
for bar, v in zip(bars, totals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{v:.1f}dB', ha='center', fontsize=11)
ax1.set_ylabel('PSNR (dB)')
ax1.set_title('Step 4a: 总体PSNR对比')
ax1.grid(True, alpha=0.3, axis='y')

# 观测/缺失像素分别对比
x_pos = np.arange(len(names))
width = 0.35
bars1 = ax2.bar(x_pos - width/2, [obs_psnrs[n] for n in names], width,
                label='观测像素 (值空间)', color='#2196F3', alpha=0.8)
bars2 = ax2.bar(x_pos + width/2, [miss_psnrs[n] for n in names], width,
                label='缺失像素 (零空间)', color='#FF9800', alpha=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(names)
ax2.set_ylabel('PSNR (dB)')
ax2.set_title('Step 4b: 值空间 vs 零空间约束效果')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('Step 4: MC + EI互补——MC约束值空间，EI约束零空间', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step4_mc_ei_complement.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step4_mc_ei_complement.png")

# 重建结果可视化
fig, axes = plt.subplots(5, 6, figsize=(15, 12))
vis_imgs, _ = next(iter(test_loader))
vis_imgs = vis_imgs[:6].to(device)
mask_vis = test_mask.unsqueeze(0).unsqueeze(0).expand_as(vis_imgs).to(device)
y_vis = vis_imgs * mask_vis

with torch.no_grad():
    pred_sup = model_sup(y_vis).clip(0, 1)
    pred_ei = model_ei(y_vis).clip(0, 1)
    pred_mc = model_mc(y_vis).clip(0, 1)
    pred_naive = model_naive(y_vis).clip(0, 1)

row_data = [
    ('干净图像x', vis_imgs.cpu()),
    ('观测y=M⊙x', y_vis.cpu()),
    (f'监督 ({total_psnrs["监督"]:.1f}dB)', pred_sup.cpu()),
    (f'MC+EI ({total_psnrs["MC+EI"]:.1f}dB)', pred_ei.cpu()),
    (f'朴素MC ({total_psnrs["朴素MC"]:.1f}dB)', pred_naive.cpu()),
]

for r, (label, imgs) in enumerate(row_data):
    for i in range(6):
        axes[r, i].imshow(imgs[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[r, i].axis('off')
    axes[r, 0].set_ylabel(label, fontsize=10, rotation=0, labelpad=80)

fig.suptitle('Step 4: MC + EI互补性——重建结果对比', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step4_reconstruction.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step4_reconstruction.png")


# ========================================================================
# Step 5: 算子-等变性对照实验
# 对应17.5.3节：验证不同算子对不同变换的等变性
# ★原创：数值验证AT_g vs T_g A的等价性
# ========================================================================
print("\n" + "="*70)
print("Step 5: 算子-等变性对照实验")
print("="*70)

def check_equivariance(A_fn, x, shift_fn, n_tests=10):
    """检查算子A是否关于变换T_g等变
    对应17.5.3节：AT_g = T_g A ?
    
    如果 AT_g ≈ T_g A → 等变 → EI无法利用此对称性
    如果 AT_g ≠ T_g A → 非等变 → EI可以利用此对称性
    """
    errors = []
    for _ in range(n_tests):
        # AT_g x
        Tg_x = shift_fn(x)
        ATg_x = A_fn(Tg_x)
        
        # T_g A x
        Ax = A_fn(x)
        TgAx = shift_fn(Ax)
        
        # 差异
        err = ((ATg_x - TgAx) ** 2).mean().sqrt().item()
        errors.append(err)
    return np.mean(errors)

# 测试图像
test_x = test_imgs[:4]

# 1. Inpainting掩码 + 平移
def inpainting_A(x):
    mask_2d = test_mask.unsqueeze(0).unsqueeze(0).expand_as(x).to(x.device)
    return x * mask_2d

def shift_fn(x, max_shift=5):
    dy = torch.randint(-max_shift, max_shift+1, (1,)).item()
    dx = torch.randint(-max_shift, max_shift+1, (1,)).item()
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))

# 2. MRI欠采样 + 平移
def mri_A(x):
    """MRI正向：FFT + 中心采样"""
    kspace = torch.fft.fft2(x)
    H, W = x.shape[2], x.shape[3]
    # 中心1D掩码
    center = H // 4
    mask = torch.zeros(H, device=x.device)
    mask[:center] = 1.0
    mask[-center:] = 1.0
    mask_2d = mask.view(1, 1, H, 1).expand_as(kspace)
    return torch.real(torch.fft.ifft2(kspace * mask_2d))

# 3. 高斯模糊 + 平移
def blur_A(x):
    """高斯模糊：关于平移等变"""
    kernel_size = 7
    sigma_k = 1.5
    kernel = torch.zeros(1, 1, kernel_size, kernel_size)
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[0, 0, i, j] = np.exp(-((i-kernel_size//2)**2 + (j-kernel_size//2)**2) / (2*sigma_k**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.to(x.device)
    return torch.nn.functional.conv2d(x, kernel, padding=kernel_size//2)

# 4. MRI + 旋转
def rotate_fn(x, angle=90):
    """90度旋转"""
    return torch.rot90(x, k=angle//90, dims=[2, 3])

# 运行等变性检查
results = {}

print("\n  算子-等变性验证 (AT_g vs T_g A 的差异, 越大越非等变):")
print(f"  {'算子':20s} {'变换':10s} {'差异':10s} {'等变?':8s}")
print(f"  {'─'*50}")

# 高斯模糊 + 平移 → 等变
err = check_equivariance(blur_A, test_x, shift_fn)
results[('高斯模糊', '平移')] = err
equiv = '✓ 等变' if err < 0.1 else '✗ 非等变'
print(f"  {'高斯模糊':20s} {'平移':10s} {err:.4f}     {equiv}")

# Inpainting + 平移 → 等变(周期性掩码)或非等变(随机掩码)
err = check_equivariance(inpainting_A, test_x, shift_fn)
results[('Inpainting', '平移')] = err
equiv = '✓ 等变' if err < 0.1 else '✗ 非等变'
print(f"  {'Inpainting(随机)':20s} {'平移':10s} {err:.4f}     {equiv}")

# MRI + 平移 → 近似等变
err = check_equivariance(mri_A, test_x, shift_fn)
results[('MRI欠采样', '平移')] = err
equiv = '✓ 等变' if err < 0.1 else '✗ 非等变'
print(f"  {'MRI欠采样':20s} {'平移':10s} {err:.4f}     {equiv}")

# 高斯模糊 + 旋转 → 非等变
err = check_equivariance(blur_A, test_x, lambda x: rotate_fn(x, 90))
results[('高斯模糊', '旋转')] = err
equiv = '✓ 等变' if err < 0.1 else '✗ 非等变'
print(f"  {'高斯模糊':20s} {'旋转90°':10s} {err:.4f}     {equiv}")

# Inpainting + 旋转 → 非等变
err = check_equivariance(inpainting_A, test_x, lambda x: rotate_fn(x, 90))
results[('Inpainting', '旋转')] = err
equiv = '✓ 等变' if err < 0.1 else '✗ 非等变'
print(f"  {'Inpainting(随机)':20s} {'旋转90°':10s} {err:.4f}     {equiv}")

# MRI + 旋转 → 非等变
err = check_equivariance(mri_A, test_x, lambda x: rotate_fn(x, 90))
results[('MRI欠采样', '旋转')] = err
equiv = '✓ 等变' if err < 0.1 else '✗ 非等变'
print(f"  {'MRI欠采样':20s} {'旋转90°':10s} {err:.4f}     {equiv}")

# 可视化
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
operators = ['高斯模糊', 'Inpainting', 'MRI欠采样']
transforms_list = ['平移', '旋转90°']
x_pos = np.arange(len(operators))
width = 0.3

for j, t in enumerate(transforms_list):
    vals = [results.get((op, t), 0) for op in operators]
    bars = ax.bar(x_pos + j * width, vals, width, label=t, alpha=0.8)
    for bar, v in zip(bars, vals):
        if v > 0.001:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', fontsize=9)

ax.set_xticks(x_pos + width/2)
ax.set_xticklabels(operators)
ax.set_ylabel('‖AT_g - T_gA‖ (越小越等变)')
ax.set_title('Step 5: 算子-等变性对照实验\n(差异大→非等变→EI可利用此对称性)')
ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='等变阈值')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step5_equivariance_check.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step5_equivariance_check.png")

print("\n  结论:")
print("  - 高斯模糊+平移: 等变→EI无法利用平移对称性(与17.5.3节一致)")
print("  - Inpainting+平移: 非等变(随机掩码)→EI可利用平移对称性")
print("  - Inpainting+旋转: 非等变→EI可利用旋转对称性")
print("  - MRI+平移: 近似等变→EI难以利用平移对称性")
print("  - MRI+旋转: 非等变→EI可利用旋转对称性(与17.5.6节FastMRI结果一致)")


# ========================================================================
# 总结
# ========================================================================
print("\n" + "="*70)
print("实验17.3 总结")
print("="*70)
print(f"  方法                  PSNR (dB)    观测像素    缺失像素    说明")
print(f"  ──────────────────────────────────────────────────────────────")
print(f"  监督 (有干净x)        {total_psnrs['监督']:.1f}       {obs_psnrs['监督']:.1f}       {miss_psnrs['监督']:.1f}       基线")
print(f"  MC+EI (自监督)        {total_psnrs['MC+EI']:.1f}       {obs_psnrs['MC+EI']:.1f}       {miss_psnrs['MC+EI']:.1f}       值空间+零空间")
print(f"  纯MC (自监督)         {total_psnrs['纯MC']:.1f}       {obs_psnrs['纯MC']:.1f}       {miss_psnrs['纯MC']:.1f}       仅值空间")
print(f"  朴素MC (自监督)       {total_psnrs['朴素MC']:.1f}       {obs_psnrs['朴素MC']:.1f}       {miss_psnrs['朴素MC']:.1f}       无约束")
print(f"\n  核心结论:")
print(f"  1. 朴素MC不约束零空间→缺失像素重建差")
print(f"  2. MC约束值空间(Af(y)≈y)，EI约束零空间(等变性)")
print(f"  3. MC+EI互补: MC保证观测一致性，EI利用对称性填补缺失")
print(f"  4. 算子非等变→EI有效: 随机inpainting关于平移/旋转非等变")
print(f"  5. 算子等变→EI无效: 高斯模糊关于平移等变(无法提供新信息)")
