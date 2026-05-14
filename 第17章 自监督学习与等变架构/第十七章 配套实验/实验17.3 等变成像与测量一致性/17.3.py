# -*- coding: utf-8 -*-
"""
实验17.3 等变成像与测量一致性
对应知识点：17.5节（等变成像：从不完整测量中学习）、17.6节（测量一致性与等变架构）

实验内容：
Step 1: 零空间问题——inpainting中自监督损失不约束缺失区域
Step 2: 等变成像（EI）损失——利用平移对称性约束零空间
Step 3: MC + EI互补性验证——值空间+零空间联合约束
Step 4: 算子-等变性对照实验——不同算子的等变性验证

★原创设计：
- 用inpainting（最直观的零空间问题）展示自监督失效
- 实现EI损失利用平移不变性虚拟产生多算子
- 可视化MC约束值空间、EI约束零空间的互补效果
- 验证算子-等变性对照表中的结论

★代码质量改进（响应评审意见）：
- 设备管理重构：消除硬编码 .to(device)，统一在类/函数构造函数中指定设备
  • InpaintingOperator.__init__(mask, device=None)
  • create_random_mask_batch(..., device=None)
  • evaluate_inpainting/evaluate_combined(..., device=None)
  • ei_loss(...): 移除未使用的device参数，从输入y推断设备
  • 全局变量 test_mask, _blur_kernel 在初始化时指定device
  • test_x 显式指定设备，避免设备不一致
  • inpainting_A 添加设备检查，确保mask和输入在同一设备
- Stop-Gradient教学说明：在ei_loss中添加详细注释，解释.detach()的必要性与替代方案

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

def _setup_chinese_font(save_dir):
    """设置中文字体（修复：在SAVE_DIR定义后调用）"""
    _cn_font = _find_chinese_font()
    if _cn_font:
        plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['font.family'] = 'sans-serif'
        print(f"[Font] 已检测到中文字体: {_cn_font}")
    else:
        # Linux/Colab 未找到中文字体，尝试加载或下载 Noto Sans SC
        if platform.system() != 'Windows':
            _font_url = 'https://github.com/jsntn/webfonts/raw/master/NotoSansSC-Regular.ttf'
            _font_file = os.path.join(save_dir, 'NotoSansSC-Regular.ttf')
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

# ★设备管理重构说明（响应评审意见）
# ──────────────────────────────────────────────────────
# 原问题：代码在各处零散使用了 .to(device)，容易导致 device mismatch 错误
# 修复方案：
#   1. InpaintingOperator类：在__init__中接收device参数，统一设备管理
#   2. create_random_mask_batch函数：添加device参数，避免调用方手动.to(device)
#   3. evaluate_inpainting/evaluate_combined函数：添加device参数
#   4. ei_loss函数：添加device参数并传递给create_random_mask_batch
#   5. 全局变量（test_mask, _blur_kernel）：在初始化时指定device
#   6. 所有训练循环：使用create_random_mask_batch(..., device=device)替代.to(device)
# 优势：
#   - 避免硬编码设备，提高代码可维护性
#   - 减少device mismatch错误风险
#   - 便于未来支持多GPU或CPU-only环境

# 修复：在SAVE_DIR定义后设置中文字体
_setup_chinese_font(SAVE_DIR)


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
        # 修复：正确的UNet skip connection设计
        self.dec3 = DoubleConv(base*2 + base*2, base*2)  # up3(e3) + e2
        self.dec2 = DoubleConv(base + base, base)        # up2(d3) + e1
        self.out_conv = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d3 = self.up3(e3)                          # base*2 (从e3上采样)
        d3 = self.dec3(torch.cat([d3, e2], dim=1)) # base*2 + base*2 (标准UNet skip)
        d2 = self.up2(d3)                          # base
        d2 = self.dec2(torch.cat([d2, e1], dim=1)) # base + base
        return self.out_conv(d2)


# ========================================================================
# Inpainting正向算子
# ========================================================================
class InpaintingOperator:
    """图像修复算子: y = M ⊙ x + ε
    对应17.5.1节：最直观的零空间问题
    
    M是二值掩码，1=保留像素，0=缺失像素
    A的零空间 = 被遮蔽的像素位置 → 自监督损失不约束这些位置
    
    ★修复：在构造函数中统一指定设备，避免硬编码.to(device)
    """
    def __init__(self, mask, device=None):
        """mask: (H, W) 二值掩码
        device: 指定计算设备（可选），如果不指定则使用mask的设备
        """
        if device is not None:
            self.mask = mask.to(device)
        else:
            self.mask = mask
        self.device = self.mask.device
    
    def A(self, x):
        """正向: y = M ⊙ x + ε"""
        # 确保mask和输入在同一设备上（防御性编程）
        if self.mask.device != x.device:
            mask_2d = self.mask.unsqueeze(0).unsqueeze(0).to(x.device)
        else:
            mask_2d = self.mask.unsqueeze(0).unsqueeze(0)
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


def create_random_mask_batch(batch_size, H, W, keep_ratio=0.5, device=None):
    """为每个batch样本创建不同的随机掩码
    ★原创：每张图用不同掩码（模拟MOI场景）
    ★修复：添加device参数，避免调用方手动.to(device)
    
    参数说明：
        keep_ratio: 保留像素比例，默认0.5
            - 较大的keep_ratio（如0.7）：更多观测像素，MC约束更强
            - 较小的keep_ratio（如0.3）：更少观测像素，零空间更大
            - 在EI损失中，虚拟样本可使用不同的keep_ratio以增强正则化
    """
    if device is None:
        device = torch.device('cpu')
    masks = torch.zeros(batch_size, 1, H, W, device=device)
    n_keep = int(H * W * keep_ratio)
    for i in range(batch_size):
        indices = torch.randperm(H * W, device=device)[:n_keep]
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

num_workers = 0 if platform.system() == 'Windows' else 2
train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

# 固定测试掩码（在device确定后创建）
test_mask = create_inpainting_mask(IMG_SIZE, IMG_SIZE, KEEP_RATIO).to(device)
inpainting_op = InpaintingOperator(test_mask, device=device)


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
naive_ckpt_path = os.path.join(SAVE_DIR, 'ckpt_Naive.pt')
naive_start = 0
if os.path.exists(naive_ckpt_path):
    ckpt = torch.load(naive_ckpt_path, map_location=device)
    model_naive.load_state_dict(ckpt['model_state'])
    optimizer_naive.load_state_dict(ckpt['optimizer_state'])
    naive_start = ckpt['epoch'] + 1
    print(f"  [Naive] 检测到已有checkpoint，从第 {naive_start+1} 轮继续训练")
if naive_start >= N_EPOCHS:
    print("  [Naive] 模型已训练完毕，跳过。")
else:
    for epoch in range(naive_start, N_EPOCHS):
        model_naive.train()
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            # 用随机掩码做inpainting（★修复：使用device参数）
            masks = create_random_mask_batch(batch_x.shape[0], IMG_SIZE, IMG_SIZE, KEEP_RATIO, device=device)
            y = batch_x * masks + SIGMA * torch.randn_like(batch_x) * masks  # 噪声仅在观测像素上
            
            optimizer_naive.zero_grad()
            f_y = model_naive(y)
            # ★ 朴素MC损失：仅在被观测像素上计算
            # 修复：正确的MC损失应为 ‖M ⊙ (y - f(y))‖²
            loss = ((masks * (y - f_y)) ** 2).sum() / masks.sum()
            loss.backward()
            optimizer_naive.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_EPOCHS}")
            torch.save({'epoch': epoch, 'model_state': model_naive.state_dict(),
                        'optimizer_state': optimizer_naive.state_dict()}, naive_ckpt_path)
            print(f"  [Naive] ✓ checkpoint已保存 (epoch {epoch+1})")

# 评估
def evaluate_inpainting(model, test_loader, mask, sigma=SIGMA, device=None):
    """评估inpainting性能
    ★修复：添加device参数，统一设备管理
    ★修复：固定随机种子，确保评估结果可重复
    """
    model.eval()
    psnr_vals = []
    if device is not None:
        mask_dev = mask.to(device)
    else:
        mask_dev = mask
    with torch.no_grad():
        torch.manual_seed(0)
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(mask_dev.device)
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
mask_2d = test_mask.unsqueeze(0).unsqueeze(0).expand_as(test_imgs)  # ★修复：test_mask已在device上
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

def ei_loss(model, y, n_transforms=4, keep_ratio=KEEP_RATIO, sigma=SIGMA):
    """等变成像损失
    对应17.5.4节：Chen, Tachella & Davies (ICCV 2021)
    
    L_EI = (1/G) Σ_g ‖T_g x̂ - f(A T_g x̂)‖²
    
    其中 x̂ = f(y) 是参考重建
    T_g: 随机平移变换
    
    ★原创实现：简化版EI损失，使用平移变换
    ★修复：移除未使用的A_fn参数，统一从输入y推断设备
    修复：虚拟样本使用独立采样的掩码，而非当前batch的掩码
    
    参数说明：
        keep_ratio: 虚拟样本的掩码保留比例
            - 默认使用全局KEEP_RATIO=0.5
            - 可调参数：不同keep_ratio可能产生更好的正则化效果
            - 例如：虚拟样本使用更稀疏的掩码（keep_ratio=0.3）可增强零空间约束
            - 或使用更密集的掩码（keep_ratio=0.7）可增强值空间约束
        sigma: 虚拟样本的噪声水平（默认使用全局SIGMA）
    
    ⚠️ 教学简化说明：
    ─────────────────────────────────────────
    1. 针对Inpainting算子简化：
       - 理论上：y_virtual = A(T_g x̂) + noise
       - 当前实现：y_virtual = T_g x̂ * mask（直接掩码乘法）
       - 原因：对于inpainting，A就是掩码乘法，两者等价
       - 通用性：如需支持其他算子（如MRI），应添加A_fn参数
    
    2. 虚拟样本未添加噪声：
       - 理论上：y_virtual = A(T_g x̂) + ε，ε ~ N(0, σ²)
       - 当前实现：y_virtual = A(T_g x̂)（无噪声）
       - 原因：教学简化，EI主要约束零空间，噪声影响较小
       - 影响：定性结论不变，定量结果可能略有差异
    ─────────────────────────────────────────
    """
    # 参考重建
    x_hat = model(y)
    
    total_loss = 0
    for _ in range(n_transforms):
        # 随机平移
        # ★ stop-gradient：对 x_hat 使用 .detach()，将 T_g x̂ 视为固定目标
        # 这是 EI 论文 (Chen et al., ICCV 2021) 中的标准做法——防止双侧梯度
        # 导致 trivial collapse（x̂ 退化为平凡解来最小化 EI 损失）。
        # 去掉 .detach() 后梯度从 T_g x̂ 和 f(AT_g x̂) 双路流回，理论上更完整，
        # 但实践中训练不稳定。此设计类似于 BYOL 中的 stop-gradient 机制。
        #
        # ⚠️ 教学要点：Stop-Gradient 的必要性与替代方案
        # ──────────────────────────────────────────────────────
        # 1. 为什么需要 .detach()？
        #    - 防止平凡解（Trivial Solution）：如果不 detach，模型可能学习到
        #      x̂ = f(y) ≈ 常数，使得 T_g x̂ ≈ f(A T_g x̂) 对所有变换都成立
        #    - 切断一半参数更新路径：梯度只从 f(AT_g x̂) 流回，不从 T_g x̂ 流回
        #    - 稳定训练：避免优化过程中的振荡和发散
        #
        # 2. 潜在误解：学生可能认为这是"唯一"的实现方式
        #    - 实际上，在某些特定对称群下（如旋转群 SO(2)），不使用 detach
        #      配合适当的正则化也是可行的
        #    - 例如：添加约束 ‖x̂‖² 或总变分正则化 TV(x̂) 可防止平凡解
        #    - 关键区别：平移群 T(2) 的平凡解风险 > 旋转群 SO(2)
        #
        # 3. 理论视角：
        #    - Stop-gradient 使损失函数变为 Bregman 散度形式
        #    - 等价于在每次迭代中固定参考点 x̂，进行单侧投影
        #    - 这类似于 EM 算法中的 E-step（固定参数）和 M-step（优化参数）
        #
        # 4. 实践建议：
        #    - 对于初学者：使用 .detach() 确保训练稳定性
        #    - 对于进阶研究：尝试无 detach + 正则化的组合，比较收敛性
        #    - 验证方法：监控 ‖x̂‖ 是否退化到零或常数
        x_hat_shifted = random_shift(x_hat.detach())
        
        # 修复：为虚拟样本独立生成掩码，使用统一的device参数
        B, C, H, W = x_hat_shifted.shape
        virtual_masks = create_random_mask_batch(B, H, W, keep_ratio, device=x_hat_shifted.device)
        y_virtual = x_hat_shifted * virtual_masks
        
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
ei_ckpt_path = os.path.join(SAVE_DIR, 'ckpt_EI.pt')
ei_start = 0
if os.path.exists(ei_ckpt_path):
    ckpt = torch.load(ei_ckpt_path, map_location=device)
    model_ei.load_state_dict(ckpt['model_state'])
    optimizer_ei.load_state_dict(ckpt['optimizer_state'])
    ei_start = ckpt['epoch'] + 1
    print(f"  [EI] 检测到已有checkpoint，从第 {ei_start+1} 轮继续训练")
if ei_start >= N_EPOCHS:
    print("  [EI] 模型已训练完毕，跳过。")
else:
    for epoch in range(ei_start, N_EPOCHS):
        model_ei.train()
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            masks = create_random_mask_batch(batch_x.shape[0], IMG_SIZE, IMG_SIZE, KEEP_RATIO, device=device)
            y = batch_x * masks + SIGMA * torch.randn_like(batch_x) * masks
            
            optimizer_ei.zero_grad()
            
            # MC损失
            f_y = model_ei(y)
            # 修复：正确的MC损失应为 ‖M ⊙ (y - f(y))‖²
            loss_mc = ((masks * (y - f_y)) ** 2).sum() / masks.sum()
            
            # EI损失（修复：虚拟掩码在ei_loss内部独立生成）
            loss_ei = ei_loss(model_ei, y, n_transforms=4)
            
            loss = loss_mc + lambda_ei * loss_ei
            loss.backward()
            optimizer_ei.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_EPOCHS}")
            torch.save({'epoch': epoch, 'model_state': model_ei.state_dict(),
                        'optimizer_state': optimizer_ei.state_dict()}, ei_ckpt_path)
            print(f"  [EI] ✓ checkpoint已保存 (epoch {epoch+1})")

psnr_ei = evaluate_inpainting(model_ei, test_loader, test_mask)
print(f"  EI (MC+EI) PSNR = {psnr_ei:.2f} dB")


# ========================================================================
# Step 3: MC + EI互补性验证
# 对应17.6.3节：MC约束值空间，EI约束零空间
# ★原创：可视化两种约束在不同像素位置的贡献
# ========================================================================
print("\n" + "="*70)
print("Step 3: MC + EI互补性验证")
print("="*70)

# 监督基线
print("\n  训练监督基线...")
model_sup = SmallUNet().to(device)
optimizer_sup = optim.Adam(model_sup.parameters(), lr=LR)
sup_ckpt_path = os.path.join(SAVE_DIR, 'ckpt_Supervised.pt')
sup_start = 0
if os.path.exists(sup_ckpt_path):
    ckpt = torch.load(sup_ckpt_path, map_location=device)
    model_sup.load_state_dict(ckpt['model_state'])
    optimizer_sup.load_state_dict(ckpt['optimizer_state'])
    sup_start = ckpt['epoch'] + 1
    print(f"  [Supervised] 检测到已有checkpoint，从第 {sup_start+1} 轮继续训练")
if sup_start >= N_EPOCHS:
    print("  [Supervised] 模型已训练完毕，跳过。")
else:
    for epoch in range(sup_start, N_EPOCHS):
        model_sup.train()
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            masks = create_random_mask_batch(batch_x.shape[0], IMG_SIZE, IMG_SIZE, KEEP_RATIO, device=device)
            y = batch_x * masks + SIGMA * torch.randn_like(batch_x) * masks
            
            optimizer_sup.zero_grad()
            f_y = model_sup(y)
            loss = nn.MSELoss()(f_y, batch_x)  # ★监督：知道干净x
            loss.backward()
            optimizer_sup.step()
        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch, 'model_state': model_sup.state_dict(),
                        'optimizer_state': optimizer_sup.state_dict()}, sup_ckpt_path)
            print(f"  [Supervised] ✓ checkpoint已保存 (epoch {epoch+1})")

psnr_sup = evaluate_inpainting(model_sup, test_loader, test_mask)
print(f"  监督 PSNR = {psnr_sup:.2f} dB")


# 各方法在观测/缺失像素的PSNR
NAIVE_KEY = '朴素MC'  # 简化标签，避免柱状图x轴拥挤
methods = {
    '监督': model_sup,
    'MC+EI': model_ei,
    NAIVE_KEY: model_naive,
}

# 修复：合并评估函数，消除重复计算
def evaluate_combined(model, test_loader, mask, sigma=SIGMA, device=None):
    """合并评估：一次推理计算所有PSNR指标
    ★修复：添加device参数，统一设备管理
    ★修复：固定随机种子，确保评估结果可重复
    """
    model.eval()
    if device is not None:
        mask_dev = mask.to(device)
    else:
        mask_dev = mask
    
    total_psnr_vals = []
    obs_psnr_vals = []
    miss_psnr_vals = []
    
    with torch.no_grad():
        torch.manual_seed(0)
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(mask_dev.device)
            mask_2d = mask_dev.unsqueeze(0).unsqueeze(0).expand_as(batch_x)
            y = batch_x * mask_2d + sigma * torch.randn_like(batch_x) * mask_2d
            pred = model(y).clip(0, 1)
            
            pred_np = pred.cpu().numpy()
            x_np = batch_x.cpu().numpy()
            m_np = mask_2d.cpu().numpy()
            
            for i in range(batch_x.shape[0]):
                # 总体PSNR
                total_psnr_vals.append(psnr(x_np[i, 0], pred_np[i, 0], data_range=1.0))
                
                # 观测像素PSNR
                obs_pixels = m_np[i, 0] > 0.5
                if obs_pixels.sum() > 0:
                    mse_obs = ((x_np[i, 0][obs_pixels] - pred_np[i, 0][obs_pixels])**2).mean()
                    obs_psnr_vals.append(10 * np.log10(1.0 / max(mse_obs, 1e-10)))
                
                # 缺失像素PSNR
                miss_pixels = m_np[i, 0] < 0.5
                if miss_pixels.sum() > 0:
                    mse_miss = ((x_np[i, 0][miss_pixels] - pred_np[i, 0][miss_pixels])**2).mean()
                    miss_psnr_vals.append(10 * np.log10(1.0 / max(mse_miss, 1e-10)))
    
    return (np.mean(total_psnr_vals), 
            np.mean(obs_psnr_vals) if obs_psnr_vals else 0, 
            np.mean(miss_psnr_vals) if miss_psnr_vals else 0)

obs_psnrs = {}
miss_psnrs = {}
total_psnrs = {}
for name, model in methods.items():
    total, obs, miss = evaluate_combined(model, test_loader, test_mask)
    obs_psnrs[name] = obs
    miss_psnrs[name] = miss
    total_psnrs[name] = total
    print(f"  {name:8s}: 总PSNR={total:.1f}dB, 观测像素={obs:.1f}dB, 缺失像素={miss:.1f}dB")

# 可视化互补性
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 总PSNR对比
names = list(methods.keys())
totals = [total_psnrs[n] for n in names]
colors_bar = ['#2196F3', '#4CAF50', '#FF9800']
bars = ax1.bar(names, totals, color=colors_bar, width=0.5)
for bar, v in zip(bars, totals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{v:.1f}dB', ha='center', fontsize=11)
ax1.set_ylabel('PSNR (dB)')
ax1.set_title('Step 3a: 总体PSNR对比')
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
ax2.set_title('Step 3b: 值空间 vs 零空间约束效果')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('Step 3: MC + EI互补——MC约束值空间，EI约束零空间', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step3_mc_ei_complement.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step3_mc_ei_complement.png")

# 重建结果可视化
fig, axes = plt.subplots(5, 6, figsize=(15, 12))
vis_imgs, _ = next(iter(test_loader))
vis_imgs = vis_imgs[:6].to(device)
mask_vis = test_mask.unsqueeze(0).unsqueeze(0).expand_as(vis_imgs)  # ★修复：test_mask已在device上
y_vis = vis_imgs * mask_vis

with torch.no_grad():
    pred_sup = model_sup(y_vis).clip(0, 1)
    pred_ei = model_ei(y_vis).clip(0, 1)
    pred_naive = model_naive(y_vis).clip(0, 1)

row_data = [
    ('干净图像x', vis_imgs.cpu()),
    ('观测y=M⊙x', y_vis.cpu()),
    (f'监督 ({total_psnrs["监督"]:.1f}dB)', pred_sup.cpu()),
    (f'MC+EI ({total_psnrs["MC+EI"]:.1f}dB)', pred_ei.cpu()),
]

# 将朴素MC的key映射回来
row_data.append((f'朴素MC ({total_psnrs[NAIVE_KEY]:.1f}dB)', pred_naive.cpu()))

for r, (label, imgs) in enumerate(row_data):
    for i in range(6):
        axes[r, i].imshow(imgs[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[r, i].axis('off')
    axes[r, 0].set_ylabel(label, fontsize=10, rotation=0, labelpad=80)

fig.suptitle('Step 3: MC + EI互补性——重建结果对比', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step3_reconstruction.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step3_reconstruction.png")


# ========================================================================
# Step 4: 算子-等变性对照实验
# 对应17.5.3节：验证不同算子对不同变换的等变性
# ★原创：数值验证AT_g vs T_g A的等价性
# ========================================================================
print("\n" + "="*70)
print("Step 4: 算子-等变性对照实验")
print("="*70)

def check_equivariance_shift(A_fn, x, n_tests=5):
    """检查算子A是否关于平移变换等变
    对应17.5.3节：AT_g = T_g A ?
    
    如果 AT_g ≈ T_g A → 等变 → EI无法利用此对称性
    如果 AT_g ≠ T_g A → 非等变 → EI可以利用此对称性
    
    修复：移除无用的shift_fn参数，在函数内部固定变换参数
    
    ⚠️ 局限性说明（针对Inpainting算子）：
    在测试阶段，test_mask 是固定的。然而在等变性检查中，随机平移后的掩码
    位置发生了变化：
    - AT_g x: 图像平移后，被固定位置的掩码遮罩
    - T_g A x: 先遮罩再平移，相当于掩码也跟着平移
    
    如果 test_mask 的随机分布不均匀（例如某块区域全黑），那么计算出的
    相对误差可能会有较大的随机波动。这并非实验设计错误，而是反映了
    Inpainting算子本身对平移的非等变性——不同平移量会导致不同的误差。
    
    参数说明：
        n_tests: 测试不同平移量的次数（默认5次）
            - 由于 test_x 固定，不同平移量产生的误差差异较大
            - 多次测试取均值可减少偶然性，但稳定性有限
            - 建议值 3-5，过多无意义
    
    返回 (绝对误差, 相对误差百分比)
    """
    errors = []
    Ax_norms = []
    for _ in range(n_tests):
        # 修复：固定每次循环的平移量
        max_shift = 5
        dy = torch.randint(-max_shift, max_shift+1, (1,)).item()
        dx = torch.randint(-max_shift, max_shift+1, (1,)).item()
        Tg = lambda x: torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        
        # AT_g x
        ATg_x = A_fn(Tg(x))
        
        # T_g A x
        Ax = A_fn(x)
        TgAx = Tg(Ax)
        
        # 差异
        err = ((ATg_x - TgAx) ** 2).mean().sqrt().item()
        errors.append(err)
        Ax_norms.append(Ax.abs().mean().item())
    abs_err = np.mean(errors)
    rel_err = abs_err / (np.mean(Ax_norms) + 1e-8)
    return abs_err, rel_err

def check_equivariance_rotate(A_fn, x, angle=90):
    """检查算子A是否关于旋转变换等变
    
    ★修复：旋转是确定性变换，无需多次测试，去掉无意义的循环
    
    返回 (绝对误差, 相对误差百分比)
    """
    Tg = lambda x: torch.rot90(x, k=angle//90, dims=[2, 3])
    
    # AT_g x
    ATg_x = A_fn(Tg(x))
    
    # T_g A x
    Ax = A_fn(x)
    TgAx = Tg(Ax)
    
    # 差异
    abs_err = ((ATg_x - TgAx) ** 2).mean().sqrt().item()
    rel_err = abs_err / (Ax.abs().mean().item() + 1e-8)
    return abs_err, rel_err

# 测试图像（★修复：显式指定设备，避免设备不一致）
test_x = test_imgs[:4].to(device)

# 1. Inpainting掩码 + 平移
def inpainting_A(x):
    """★修复：确保mask和输入在同一设备上（防御性编程）"""
    if test_mask.device != x.device:
        mask_2d = test_mask.unsqueeze(0).unsqueeze(0).expand_as(x).to(x.device)
    else:
        mask_2d = test_mask.unsqueeze(0).unsqueeze(0).expand_as(x)
    return x * mask_2d

# 2. MRI欠采样 + 平移 (多种采样模式对比)
def mri_A(x, sampling_mode='vertical'):
    """MRI正向：FFT + 不同采样模式
    
    ★修复：使用x.device自动适配，避免硬编码device
    
    采样模式对比：
    - vertical: 垂直条带（原版，近似等变）
    - random: 随机采样（非等变，对比更明显）
    - cartesian: 规律性网格采样（非等变）
    
    目的：展示不同采样模式对等变性的影响
    
    ⚠️ 注意：random模式使用固定种子，确保等变性测试时掩码一致
    """
    kspace = torch.fft.fft2(x)
    H, W = x.shape[2], x.shape[3]
    
    if sampling_mode == 'vertical':
        # 原版：垂直条带采样（近似等变）
        center = H // 4
        mask = torch.zeros(H, device=x.device)
        mask[:center] = 1.0
        mask[-center:] = 1.0
        
    elif sampling_mode == 'random':
        # 随机采样（非等变）
        # ★修复：固定种子，确保等变性测试时两次调用生成相同掩码
        # 否则测出的误差是"掩码不同"而非"平移非等变"
        state = torch.get_rng_state()
        torch.manual_seed(42)
        mask = torch.rand(H, device=x.device) < 0.25  # 25%采样率
        torch.set_rng_state(state)
        
    elif sampling_mode == 'cartesian':
        # Cartesian网格采样（非等变）
        mask = torch.zeros(H, device=x.device)
        mask[::4] = 1.0  # 每4行采样1行
        
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")
    
    mask_2d = mask.view(1, 1, H, 1).expand_as(kspace)
    return torch.real(torch.fft.ifft2(kspace * mask_2d))

# 3. 高斯模糊 + 平移
# ★修复：将模糊核的创建移到device确定后，避免每次调用都.to(device)
_kernel_size = 7
_sigma_k = 1.5
_blur_kernel = torch.zeros(1, 1, _kernel_size, _kernel_size, device=device)  # ★修复：在初始化时指定device
for _i in range(_kernel_size):
    for _j in range(_kernel_size):
        _blur_kernel[0, 0, _i, _j] = np.exp(-((_i-_kernel_size//2)**2 + (_j-_kernel_size//2)**2) / (2*_sigma_k**2))
_blur_kernel = _blur_kernel / _blur_kernel.sum()

def blur_A(x):
    """高斯模糊：关于平移等变
    ★修复：使用预初始化的_blur_kernel（已在device上），避免每次调用都.to(device)
    """
    return torch.nn.functional.conv2d(x, _blur_kernel, padding=_kernel_size//2)

# 运行等变性检查
results = {}
rel_results = {}

print("\n  算子-等变性验证 (AT_g vs T_g A):")
print(f"  {'算子':20s} {'变换':10s} {'绝对误差':10s} {'相对误差':10s}")
print(f"  {'─'*55}")

# 高斯模糊 + 平移 → 等变
abs_err, rel_err = check_equivariance_shift(blur_A, test_x)
results[('高斯模糊', '平移')] = abs_err
rel_results[('高斯模糊', '平移')] = rel_err
print(f"  {'高斯模糊':20s} {'平移':10s} {abs_err:.4f}       {rel_err*100:.1f}%")

# Inpainting + 平移 → 等变(周期性掩码)或非等变(随机掩码)
abs_err, rel_err = check_equivariance_shift(inpainting_A, test_x)
results[('Inpainting', '平移')] = abs_err
rel_results[('Inpainting', '平移')] = rel_err
print(f"  {'Inpainting(随机)':20s} {'平移':10s} {abs_err:.4f}       {rel_err*100:.1f}%")

# MRI + 平移 → 不同采样模式的等变性对比
sampling_modes = ['vertical', 'random', 'cartesian']
for mode in sampling_modes:
    abs_err, rel_err = check_equivariance_shift(
        lambda x: mri_A(x, sampling_mode=mode), test_x)
    results[(f'MRI欠采样({mode})', '平移')] = abs_err
    rel_results[(f'MRI欠采样({mode})', '平移')] = rel_err
    print(f"  {'MRI欠采样({mode})':20s} {'平移':10s} {abs_err:.4f}       {rel_err*100:.1f}%")

# 高斯模糊 + 旋转 → 非等变
abs_err, rel_err = check_equivariance_rotate(blur_A, test_x, angle=90)
results[('高斯模糊', '旋转')] = abs_err
rel_results[('高斯模糊', '旋转')] = rel_err
print(f"  {'高斯模糊':20s} {'旋转90°':10s} {abs_err:.4f}       {rel_err*100:.1f}%")

# Inpainting + 旋转 → 非等变
abs_err, rel_err = check_equivariance_rotate(inpainting_A, test_x, angle=90)
results[('Inpainting', '旋转')] = abs_err
rel_results[('Inpainting', '旋转')] = rel_err
print(f"  {'Inpainting(随机)':20s} {'旋转90°':10s} {abs_err:.4f}       {rel_err*100:.1f}%")

# MRI + 旋转 → 非等变
abs_err, rel_err = check_equivariance_rotate(mri_A, test_x, angle=90)
results[('MRI欠采样', '旋转')] = abs_err
rel_results[('MRI欠采样', '旋转')] = rel_err
print(f"  {'MRI欠采样':20s} {'旋转90°':10s} {abs_err:.4f}       {rel_err*100:.1f}%")

# 可视化：使用相对误差百分比，展示不同采样模式的对比效果
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# 更新算子列表，包含不同采样模式
operators = ['高斯模糊', 'Inpainting', 'MRI欠采样(vertical)', 
            'MRI欠采样(random)', 'MRI欠采样(cartesian)']
transforms_list = ['平移', '旋转90°']
x_pos = np.arange(len(operators))
width = 0.25

for j, t in enumerate(transforms_list):
    vals = [rel_results.get((op, t), 0) * 100 for op in operators]
    bars = ax.bar(x_pos + j * width, vals, width, label=t, alpha=0.8)
    for bar, v in zip(bars, vals):
        if v > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.1f}%', ha='center', fontsize=8)

ax.set_xticks(x_pos + width)
ax.set_xticklabels(operators, rotation=45, ha='right')
ax.set_ylabel('相对误差 ‖AT_g-T_gA‖/‖Ax‖ (%)')
ax.set_title('Step 4: 算子-等变性对照实验\n(bar越高→越非等变→EI越可利用此对称性)\n对比不同MRI采样模式的等变性差异')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step4_equivariance_check.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step4_equivariance_check.png")

print("\n  结论:")
print("  - 高斯模糊+平移: 相对误差≈0% → 等变 → EI无法利用平移对称性(与17.5.3节一致)")
print("  - Inpainting(固定掩码)+平移: 相对误差较大 → 非等变 → EI可利用平移对称性")
print("  - Inpainting(固定掩码)+旋转: 非等变 → EI可利用旋转对称性")
print("  - MRI+平移: 相对误差较小 → 近似等变 → EI难以利用平移对称性")
print("  - MRI+旋转: 非等变 → EI可利用旋转对称性(与17.5.6节FastMRI结果一致)")
print("  注: Inpainting测试使用固定掩码，若使用随机掩码行为会不同")


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
print(f"  朴素MC (自监督)       {total_psnrs[NAIVE_KEY]:.1f}       {obs_psnrs[NAIVE_KEY]:.1f}       {miss_psnrs[NAIVE_KEY]:.1f}       仅值空间")
print(f"\n  核心结论:")
print(f"  1. 朴素MC不约束零空间→缺失像素重建差")
print(f"  2. MC约束值空间(Af(y)≈y)，EI约束零空间(等变性)")
print(f"  3. MC+EI互补: MC保证观测一致性，EI利用对称性填补缺失")
print(f"  4. 算子非等变→EI有效: 随机inpainting关于平移/旋转非等变")
print(f"  5. 算子等变→EI无效: 高斯模糊关于平移等变(无法提供新信息)")

# 三部曲逻辑链条
print(f"""
  ╔═══════════════════════════════════════════════════════════════════╗
  ║           实验17.1-17.3 三部曲逻辑链条                              ║
  ╠═══════════════════════════════════════════════════════════════════╣
  ║  实验    │   核心问题          │   解决方法      │   连接点         ║
  ╠═══════════════════════════════════════════════════════════════════╣
  ║  17.1    │   朴素MSE有偏       │   N2N / N2B     │   配对噪声/空间配对 ║
  ║  17.2    │   需要配对噪声      │   SURE (修正项)  │   从N2N进化到SURE  ║
  ║  17.3    │   A≠I时SURE只约束值空间 │ EI (等变约束) │   SURE+EI=完整约束 ║
  ╠═══════════════════════════════════════════════════════════════════╣
  ║  17.1：当 y = x + ε（噪声）时，问：如何自监督？                     ║
  ║  17.2：当 y = x + ε 且 A = I 时，答：SURE修正偏差                   ║
  ║  17.3：当 A ≠ I（inpainting/MRI）时，问：SURE失效怎么办？           ║
  ║        答：EI加约束，利用对称性约束零空间                            ║
  ╚═══════════════════════════════════════════════════════════════════╝
""")
