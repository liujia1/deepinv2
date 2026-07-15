# -*- coding: utf-8 -*-
"""
实验17.2-1 Noise2Noise与自监督去噪
对应知识点：17.1节（数据困境与学习设定谱系）、17.2节（Noise2Noise原理）

实验内容：
Step 1: 四种学习设定的数据构造与对比 —— 在MNIST上构造四种设定的数据
Step 2: Noise2Noise原理验证 —— 用配对噪声数据训练，对比监督基线
Step 3: Noise2Noise与朴素自监督损失对比 —— 展示‖y-f(y)‖²的偏差
Step 4: Neighbor2Neighbor空间配对 —— 利用邻域结构近似配对

★原创设计：
- 在MNIST上系统对比四种学习设定的训练效果
- 用散度分析可视化朴素MSE的偏差来源
- Neighbor2Neighbor从单帧噪声图像构造伪配对

素材来源：MiniProject_Self_Supervised中N2N思路、deepinv.loss API
运行前提：需GPU（Colab T4即可）
"""

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os, sys, io, time, warnings, logging

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
    SAVE_DIR = os.path.join(_gdrive, '实验17.2-1')
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
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ★可复现性说明：
# torch.manual_seed(42)只在文件开头设置一次，全局RNG会被add_noise、
# 模型初始化、训练循环等依次消耗。断点续训会改变后续模型初始化时的
# RNG状态，导致"从头跑"和"从checkpoint恢复"得到的最终数值略有差异。
# 对教学实验影响有限（结论方向一般不变），工程应用如需完全可复现，
# 建议为每个训练函数传入独立的torch.Generator。
# evaluate_psnr中已使用局部Generator+固定seed=12345的设计，不污染全局RNG。

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"实验17.2-1: Noise2Noise与自监督去噪")
print(f"{'='*70}")
print(f"使用设备: {device}")

num_workers = 0 if sys.platform == 'win32' else 2


# ========================================================================
# 轻量级UNet（复用自15.2/16.3的SmallUNet架构）
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
    """轻量UNet，用于MNIST 28×28→上采样到32×32"""
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
        e1 = self.enc1(x)        # (B, base, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, base*2, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, base*4, H/4, W/4)
        d3 = self.up3(e3)             # (B, base*2, H/2, W/2)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d3)             # (B, base, H, W)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        return self.out_conv(d2)


# ========================================================================
# 数据准备：MNIST + 高斯噪声
# ========================================================================
IMG_SIZE = 32  # 上采样到32×32以便UNet下采样
SIGMA = 0.3    # 噪声标准差
BATCH_SIZE = 128
N_EPOCHS = 30
LR = 1e-3

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),  # [0,1]
])

print("加载MNIST数据集...")
mnist_train = datasets.MNIST(root=os.path.join(SAVE_DIR, 'mnist_data'),
                              train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root=os.path.join(SAVE_DIR, 'mnist_data'),
                             train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)


def add_noise(x, sigma=SIGMA):
    """给干净图像添加高斯噪声并 clamp 到 [0,1]

    ★ 关于 clamp 对 N2N 理论假设的影响（重要说明）：
    - N2N 的理论保证依赖于 E[y'|x] = x，即噪声是零均值的
    - clamp 操作会在边界处（接近 0 或 1 的像素）引入截断偏差，
      使得 E[clamp(x+ε)|x] ≠ x，破坏了严格的零均值假设
    - 当 σ=0.3 时，MNIST 中接近 0 的背景区域受截断影响较大
    - 本实验选择 clamp 的原因：
      1. 保持训练/评估分布一致（PSNR 要求 data_range=1.0）
      2. 避免负值或超 1 的像素导致可视化异常
      3. 教学实验中，clamp 带来的偏差对定性结论影响有限
    - 严谨的论文实验中应讨论此偏差，或使用更小的 σ 减小截断影响
    """
    return (x + sigma * torch.randn_like(x)).clamp(0, 1)


def evaluate_psnr(model, test_loader, sigma=SIGMA, noise_seed=12345):
    """在测试集上评估PSNR和SSIM

    ★为什么同时使用PSNR和SSIM：
    - PSNR反映像素级均方误差，但与人眼感知不完全一致
    - SSIM考虑亮度、对比度、结构信息，更符合人眼对图像质量的感知
    - 自监督方法（尤其是N2B）可能产生轻微模糊，SSIM能更好地反映结构完整性

    ★关于noise_seed：
    - 使用固定种子确保所有模型评估时使用同一份噪声
    - 这是严格意义上的"同一测试条件"对比
    """
    model.eval()
    psnr_vals = []
    ssim_vals = []
    # ★使用固定种子，确保所有模型评估时噪声一致
    noise_rng = torch.Generator(device=device)
    noise_rng.manual_seed(noise_seed)

    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader, desc='评估PSNR', leave=False):
            batch_x = batch_x.to(device)
            # 使用固定种子生成噪声，确保跨模型评估一致性
            noise = torch.randn(batch_x.shape, generator=noise_rng, device=device)
            y = (batch_x + sigma * noise).clamp(0, 1)
            pred = model(y)
            pred_np = pred.cpu().numpy().clip(0, 1)
            x_np = batch_x.cpu().numpy()
            for i in range(pred_np.shape[0]):
                psnr_vals.append(psnr(x_np[i, 0], pred_np[i, 0], data_range=1.0))
                ssim_vals.append(ssim(x_np[i, 0], pred_np[i, 0], data_range=1.0))
    return np.mean(psnr_vals), np.mean(ssim_vals)


# ========================================================================
# Step 1: 四种学习设定的数据构造与性能差异对比
# 对应17.1.2节：监督 / 合成配对 / 无监督-x / 无监督-y
# ★修正：除了展示数据，还训练轻量模型对比不同设定的性能差异
# ========================================================================
print("\n" + "="*70)
print("Step 1: 四种学习设定的数据构造与性能差异对比")
print("="*70)

# 取测试图像做可视化
# ★使用共享的test_iter，让各步骤可视化使用不同batch（避免总是同一批图像）
test_iter = iter(test_loader)
test_imgs, _ = next(test_iter)
test_imgs = test_imgs[:8].to(device)
test_noisy = add_noise(test_imgs, SIGMA)  # 第一次噪声观测 y
test_noisy2 = add_noise(test_imgs, SIGMA)  # ★第二次独立噪声观测 y'（用于设定2的合成配对）

# ★命名规范说明（避免后续s1/s2术语混淆）：
# - 本文件Step 1使用s1/s2/s3/s4指代"设定1/设定2/设定3/设定4"（learning setting）
# - Step 4起使用sub1/sub2指代"子图1/子图2"（neighbor sub-image）
# - 两者是不同概念，请按上下文区分

# --- 1a. 数据可视化 ---
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
titles = ['设定1: 监督 (x,y)', '设定2: 合成配对 (y,y\')',
          '设定3: 无监督-x (仅x)', '设定4: 无监督-y (仅y)']
for i in range(8):
    # 设定1: 监督学习 - 有配对的干净x和噪声y
    axes[0, i].imshow(test_imgs[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    # 设定2: 合成配对 - 有两次独立噪声观测y, y'（这里展示y'，y在设定4行展示）
    axes[1, i].imshow(test_noisy2[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
    # 设定3: 无监督-x - 仅有干净图像（无噪声配对，无法训练去噪）
    axes[2, i].imshow(test_imgs[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[2, i].axis('off')
    # 设定4: 无监督-y - 仅需要噪声观测（本章核心挑战）
    axes[3, i].imshow(test_noisy[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[3, i].axis('off')

for r in range(4):
    axes[r, 0].set_ylabel(titles[r], fontsize=9, rotation=0, labelpad=100)

fig.suptitle(f'Step 1a: 四种学习设定的数据对比 ($\\sigma$={SIGMA:.1f})', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step1a_four_settings_data.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step1a_four_settings_data.png")
print("  说明: 设定1-3需要干净数据x，设定4仅需要噪声观测y——本章的核心挑战")

# --- 1b. 四种设定的可行性与性能对比实验 ---
# 设定3（仅x）无法训练去噪模型，因为去噪需要学习从y到x的映射
# 这里我们对比设定1、2、4的训练效果

# ★说明：Step1的model_s1/model_s2与Step2的model_sup/model_n2n是独立实验，非复用
# - Step1：快速训练（5 epochs），用于四种设定的定性对比，了解各设定的相对性能
# - Step2：完整训练（30 epochs），用于N2N原理的定量验证
# 虽然loss函数相同，但训练目的不同，Step1侧重"哪种设定可行"，Step2侧重"N2N是否等价于监督"
print("\n  快速训练对比四种设定的性能差异（各训练5个epoch）...")

def quick_train(model, loss_fn, n_epochs=5, tag=""):
    """快速训练用于对比（支持resume，兼容新旧checkpoint格式）"""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ckpt_path = os.path.join(SAVE_DIR, f'ckpt_quick_{tag}.pt') if tag else None
    start_epoch = 0
    is_final = False
    train_losses = []

    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
        optimizer_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer_state'))
        train_losses = checkpoint.get('train_losses', checkpoint.get('losses', []))

        if checkpoint.get('is_final', False):
            print(f"  [{tag}] ✓ 检测到最终权重，直接加载，跳过训练过程")
            print(f"  训练轮数: {checkpoint['epoch']+1}")
            model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1
            is_final = True
        else:
            print(f"  [{tag}] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
            model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1

    if is_final or start_epoch >= n_epochs:
        print(f"  [{tag}] 模型已训练完毕，跳过。")
        psnr_val, ssim_val = evaluate_psnr(model, test_loader)
        return psnr_val, ssim_val

    model.train()
    for epoch in range(start_epoch, n_epochs):
        epoch_loss = 0.0
        n_batch = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [{tag}]', leave=False, unit='batch')
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            y = add_noise(batch_x, SIGMA)
            optimizer.zero_grad()
            pred = model(y)
            loss = loss_fn(pred, batch_x, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        avg_loss = epoch_loss / n_batch
        train_losses.append(avg_loss)
        # 保存checkpoint
        if ckpt_path:
            if epoch == n_epochs - 1:
                # 最终checkpoint，不保存optimizer_state_dict
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': train_losses[-1],
                    'train_losses': train_losses,
                    'is_final': True
                }, ckpt_path)
            else:
                # 中间checkpoint，保存optimizer_state_dict
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'train_losses': train_losses,
                    'is_final': False
                }, ckpt_path)

    psnr_val, ssim_val = evaluate_psnr(model, test_loader)
    return psnr_val, ssim_val

# 设定1: 监督学习 (有配对的x和y)
model_s1 = SmallUNet().to(device)
psnr_s1, ssim_s1 = quick_train(model_s1, lambda p, x, y: nn.MSELoss()(p, x), tag="S1-supervised")

# 设定2: 合成配对 (有两次独立噪声观测)
# ★语义说明：这里"设定2"描述的是数据可得性层次——有(y, y')两次独立噪声观测。
# 训练算法本质上就是N2N损失‖y'-f(y)‖²，与Step 2的model_n2n是同一件事。
# 区别仅在于：Step 1的设定2用5 epoch快速验证可行性，Step 2的N2N用30 epoch完整训练。
# 不要把"设定2"和"Step 2的N2N"误解为两种不同方法。
def train_synthetic_pair(model, n_epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ckpt_path = os.path.join(SAVE_DIR, 'ckpt_quick_S2-synthetic-pair.pt')
    start_epoch = 0
    is_final = False
    train_losses = []

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
        optimizer_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer_state'))
        train_losses = checkpoint.get('train_losses', checkpoint.get('losses', []))

        if checkpoint.get('is_final', False):
            print(f"  [设定2-合成配对] ✓ 检测到最终权重，直接加载，跳过训练过程")
            print(f"  训练轮数: {checkpoint['epoch']+1}")
            model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1
            is_final = True
        else:
            print(f"  [设定2-合成配对] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
            model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1

    if is_final or start_epoch >= n_epochs:
        print(f"  [设定2-合成配对] 模型已训练完毕，跳过。")
        psnr_val, ssim_val = evaluate_psnr(model, test_loader)
        return psnr_val, ssim_val

    model.train()
    for epoch in range(start_epoch, n_epochs):
        epoch_loss = 0.0
        n_batch = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [设定2-合成配对]', leave=False, unit='batch')
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            y1 = add_noise(batch_x, SIGMA)
            y2 = add_noise(batch_x, SIGMA)  # 第二次独立采样
            optimizer.zero_grad()
            pred = model(y1)
            loss = nn.MSELoss()(pred, y2.detach())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        avg_loss = epoch_loss / n_batch
        train_losses.append(avg_loss)
        # 保存checkpoint
        if epoch == n_epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': train_losses[-1],
                'train_losses': train_losses,
                'is_final': True
            }, ckpt_path)
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'train_losses': train_losses,
                'is_final': False
            }, ckpt_path)

    psnr_val, ssim_val = evaluate_psnr(model, test_loader)
    return psnr_val, ssim_val

model_s2 = SmallUNet().to(device)
psnr_s2, ssim_s2 = train_synthetic_pair(model_s2)

# 设定3: 无监督-x (仅干净图像) - 无法训练去噪
# 解释：去噪需要学习f: y→x，但没有y，无法训练
psnr_s3 = None  # 未使用，仅为占位说明设定3不可行
ssim_s3 = None  # 未使用，仅为占位说明设定3不可行

# 设定4: 无监督-y (仅噪声观测) - 朴素方法
model_s4 = SmallUNet().to(device)
psnr_s4, ssim_s4 = quick_train(model_s4, lambda p, x, y: nn.MSELoss()(p, y), tag="S4-naive")

print(f"\n  四种设定的快速训练结果（5 epochs）:")
print(f"    设定1 (监督):     PSNR = {psnr_s1:.2f} dB, SSIM = {ssim_s1:.4f}")
print(f"    设定2 (合成配对): PSNR = {psnr_s2:.2f} dB, SSIM = {ssim_s2:.4f}")
print(f"    设定3 (仅x):      不可行 - 缺少噪声输入，无法训练去噪模型")
print(f"    设定4 (仅y-朴素): PSNR = {psnr_s4:.2f} dB, SSIM = {ssim_s4:.4f}")

# 可视化对比
fig, axes = plt.subplots(1, 4, figsize=(14, 3))
vis_img = test_imgs[:1]
vis_noisy = add_noise(vis_img, SIGMA)

with torch.no_grad():
    pred_s1 = model_s1(vis_noisy).cpu().clip(0, 1)
    pred_s2 = model_s2(vis_noisy).cpu().clip(0, 1)
    pred_s4 = model_s4(vis_noisy).cpu().clip(0, 1)

methods = [
    (f'设定1: 监督\n{psnr_s1:.1f}dB', pred_s1[0,0], '#2196F3'),
    (f'设定2: 合成配对\n{psnr_s2:.1f}dB', pred_s2[0,0], '#4CAF50'),
    (f'设定3: 仅x\n不可行', vis_img[0,0].cpu(), '#9E9E9E'),
    (f'设定4: 仅y-朴素\n{psnr_s4:.1f}dB', pred_s4[0,0], '#FF9800'),
]

for ax, (title, img, color) in zip(axes, methods):
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10, color=color if '不可行' not in title else 'gray')
    ax.axis('off')

fig.suptitle(f'Step 1b: 四种学习设定的性能对比 ($\\sigma$={SIGMA:.1f}, 5 epochs)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step1b_four_settings_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step1b_four_settings_comparison.png")
# ★动态校验：根据实际PSNR值生成结论，而非硬编码断言
if psnr_s4 < min(psnr_s1, psnr_s2):
    print(f"  核心发现: 设定4（仅y）的朴素方法PSNR最低({psnr_s4:.2f}dB < {min(psnr_s1, psnr_s2):.2f}dB)，需要更聪明的自监督策略（N2N/SURE）")
else:
    print(f"  反常提示: 设定4的PSNR({psnr_s4:.2f}dB)并非最低，与理论预期不符，建议检查训练过程")


# ========================================================================
# Step 2: Noise2Noise原理验证
# 对应17.2.1-17.2.2节：配对噪声数据训练，验证N2N等价性
# ========================================================================
print("\n" + "="*70)
print("Step 2: Noise2Noise原理验证")
print("="*70)

def train_model(model, loss_fn, train_loader, n_epochs=N_EPOCHS, tag=""):
    """通用训练循环（支持断点续训，兼容新旧checkpoint格式）"""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ckpt_path = os.path.join(SAVE_DIR, f'ckpt_{tag}.pt') if tag else None
    start_epoch = 0
    is_final = False
    train_losses = []

    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
        optimizer_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer_state'))
        train_losses = checkpoint.get('train_losses', checkpoint.get('losses', []))

        if checkpoint.get('is_final', False):
            print(f"  [{tag}] ✓ 检测到最终权重，直接加载，跳过训练过程")
            print(f"  训练轮数: {checkpoint['epoch']+1}")
            model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1
            is_final = True
        else:
            print(f"  [{tag}] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
            model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1

    if is_final or start_epoch >= n_epochs:
        print(f"  [{tag}] 模型已训练完毕，跳过。")
        return train_losses

    model.train()
    for epoch in range(start_epoch, n_epochs):
        epoch_loss = 0
        n_batch = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [{tag}]', leave=False, unit='batch')
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            # 构造噪声观测 y = x + ε
            y1 = add_noise(batch_x, SIGMA)
            optimizer.zero_grad()
            pred = model(y1)
            loss = loss_fn(pred, batch_x, y1)  # loss_fn决定用x还是y'作为target
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        avg_loss = epoch_loss / n_batch
        train_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  [{tag}] Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
        # 保存checkpoint
        if ckpt_path and (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'train_losses': train_losses,
                'is_final': False
            }, ckpt_path)
            print(f"  [{tag}] ✓ checkpoint已保存 (epoch {epoch+1})")

    # 保存最终checkpoint
    if ckpt_path and train_losses:
        torch.save({
            'epoch': n_epochs - 1,
            'model_state_dict': model.state_dict(),
            'loss': train_losses[-1],
            'train_losses': train_losses,
            'is_final': True
        }, ckpt_path)
        print(f"  [{tag}] ✓ 训练完成, 最终checkpoint已保存")

    return train_losses

# --- 2a. 监督基线：loss = ‖x - f(y)‖² ---
print("\n  训练监督基线 (Supervised)...")
model_sup = SmallUNet().to(device)
losses_sup = train_model(
    model_sup,
    loss_fn=lambda pred, x, y: nn.MSELoss()(pred, x),  # 用干净x作target
    train_loader=train_loader,
    tag="Supervised"
)
psnr_sup, ssim_sup = evaluate_psnr(model_sup, test_loader)
print(f"  监督基线 PSNR = {psnr_sup:.2f} dB, SSIM = {ssim_sup:.4f}")

# --- 2b. Noise2Noise：loss = ‖y' - f(y)‖² ---
# ★修正：真正的N2N要求对同一批数据独立采样两次噪声观测，不使用干净数据x
print("\n  训练Noise2Noise...")

def train_n2n_model(model, train_loader, n_epochs=N_EPOCHS, tag="N2N"):
    """N2N专用训练循环：对每个batch独立采样两次噪声"""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ckpt_path = os.path.join(SAVE_DIR, f'ckpt_{tag}.pt')
    start_epoch = 0
    is_final = False
    train_losses = []

    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
        optimizer_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer_state'))
        train_losses = checkpoint.get('train_losses', checkpoint.get('losses', []))

        if checkpoint.get('is_final', False):
            print(f"  [{tag}] ✓ 检测到最终权重，直接加载，跳过训练过程")
            print(f"  训练轮数: {checkpoint['epoch']+1}")
            model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1
            is_final = True
        else:
            print(f"  [{tag}] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
            model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1

    if is_final or start_epoch >= n_epochs:
        print(f"  [{tag}] 模型已训练完毕，跳过。")
        return train_losses

    model.train()
    for epoch in range(start_epoch, n_epochs):
        epoch_loss = 0
        n_batch = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [{tag}]', leave=False, unit='batch')
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            # ★核心修正：对同一批干净图像独立采样两次噪声，模拟两次独立观测
            y1 = add_noise(batch_x, SIGMA)  # 第一次噪声观测
            y2 = add_noise(batch_x, SIGMA)  # 第二次独立噪声观测
            optimizer.zero_grad()
            pred = model(y1)
            # N2N损失：用y2作为target，不使用x
            loss = nn.MSELoss()(pred, y2.detach())  # ★ detach防止梯度回传到y2
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        avg_loss = epoch_loss / n_batch
        train_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  [{tag}] Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
        if ckpt_path and (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'train_losses': train_losses,
                'is_final': False
            }, ckpt_path)
            print(f"  [{tag}] ✓ checkpoint已保存 (epoch {epoch+1})")

    # 保存最终checkpoint
    if ckpt_path and train_losses:
        torch.save({
            'epoch': n_epochs - 1,
            'model_state_dict': model.state_dict(),
            'loss': train_losses[-1],
            'train_losses': train_losses,
            'is_final': True
        }, ckpt_path)
        print(f"  [{tag}] ✓ 训练完成, 最终checkpoint已保存")

    return train_losses

model_n2n = SmallUNet().to(device)
losses_n2n = train_n2n_model(model_n2n, train_loader, tag="N2N")
psnr_n2n, ssim_n2n = evaluate_psnr(model_n2n, test_loader)
print(f"  Noise2Noise PSNR = {psnr_n2n:.2f} dB, SSIM = {ssim_n2n:.4f}")

# 可视化对比
fig, axes = plt.subplots(3, 6, figsize=(15, 7))
vis_imgs, _ = next(test_iter)  # ★使用共享test_iter，与Step 1不同batch
vis_imgs = vis_imgs[:6].to(device)
vis_noisy = add_noise(vis_imgs, SIGMA)

with torch.no_grad():
    pred_sup = model_sup(vis_noisy).cpu().clip(0, 1)
    pred_n2n = model_n2n(vis_noisy).cpu().clip(0, 1)

for i in range(6):
    axes[0, i].imshow(vis_noisy[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    axes[0, i].set_title(f'噪声输入', fontsize=9)
    axes[1, i].imshow(pred_sup[i, 0], cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
    p_s = psnr(vis_imgs[i,0].cpu().numpy(), pred_sup[i,0].numpy(), data_range=1.0)
    axes[1, i].set_title(f'监督 {p_s:.1f}dB', fontsize=9)
    axes[2, i].imshow(pred_n2n[i, 0], cmap='gray', vmin=0, vmax=1)
    axes[2, i].axis('off')
    p_n = psnr(vis_imgs[i,0].cpu().numpy(), pred_n2n[i,0].numpy(), data_range=1.0)
    axes[2, i].set_title(f'N2N {p_n:.1f}dB', fontsize=9)

axes[0, 0].set_ylabel('噪声输入 y', fontsize=11)
axes[1, 0].set_ylabel(r'监督: $\|x-f(y)\|^2$', fontsize=11)
axes[2, 0].set_ylabel(r"N2N: $\|y'-f(y)\|^2$", fontsize=11)
fig.suptitle(f'Step 2: Noise2Noise vs 监督 (PSNR: 监督={psnr_sup:.1f}dB, N2N={psnr_n2n:.1f}dB)',
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step2_n2n_vs_supervised.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  已保存: step2_n2n_vs_supervised.png")
# ★动态校验：根据实际PSNR差值判断是否"≈"，而非硬编码断言
n2n_sup_diff = abs(psnr_n2n - psnr_sup)
if n2n_sup_diff < 2.0:  # 2dB阈值，教学实验中认为"≈"
    print(f"  结论: N2N PSNR ({psnr_n2n:.1f}dB) ≈ 监督PSNR ({psnr_sup:.1f}dB)")
    print(f"  验证了定理: argmin E‖y'-f(y)‖² = argmin E‖x-f(y)‖² = E[x|y]")
else:
    print(f"  反常：N2N与监督差异较大 ({n2n_sup_diff:.2f}dB)，与理论预期不符")
    print(f"  建议：检查训练是否充分或增大epoch数")


# ========================================================================
# Step 3: 朴素自监督损失偏差分析
# 对应17.3.1节：‖y-f(y)‖²有偏，低估真实风险
# ★原创：可视化散度项‖div f(y)‖随训练的变化
# ========================================================================
print("\n" + "="*70)
print("Step 3: 朴素自监督损失偏差分析")
print("="*70)

# --- 3a. 训练朴素自监督：loss = ‖y - f(y)‖² ---
# ★说明：此处为独立训练，非复用Step1的model_s4
# - Step1的model_s4：仅5 epochs，用于四种设定的快速对比
# - Step3的model_naive：30 epochs，用于详细分析偏差来源和训练曲线
# 两者loss函数相同（‖y-f(y)‖²），但训练目的不同
print("\n  训练朴素自监督 (Naive: ‖y-f(y)‖²)...")
model_naive = SmallUNet().to(device)
losses_naive = train_model(
    model_naive,
    loss_fn=lambda pred, x, y: nn.MSELoss()(pred, y),  # ★用y作target（错误！）
    train_loader=train_loader,
    tag="Naive"
)
psnr_naive, ssim_naive = evaluate_psnr(model_naive, test_loader)
print(f"  朴素自监督 PSNR = {psnr_naive:.2f} dB, SSIM = {ssim_naive:.4f}")

# --- 3b. 计算散度项分析偏差 ---
def compute_divergence_mc(model, y, n_samples=10, alpha=1e-3, divergence_seed=12345):
    """用Monte Carlo方法估计散度 div f(y) = Σ ∂f_i/∂y_i

    ★原创：用于可视化偏差来源
    参考: Ramani et al. (2007)

    返回值：单图平均散度（batch内所有图像的散度均值）
    注意：散度是单张图像的属性，对batch求和会导致数值被放大约batch_size倍

    ★关于样本量的教学说明：
    - 默认使用10个随机方向（Hutchinson迹估计），在16张测试图上估计
    - n_samples=10比n_samples=5方差更小，对散度估计的稳定性有显著改善
    - 如果bias_term的符号不稳定，可进一步增加n_samples
    - 工程应用中建议使用更多样本以获得稳定估计

    ★关于随机方向分布：
    - 使用Rademacher分布（±1），方差比高斯分布小
    - 数学上：E[ωω^T] = I 同样满足Hutchinson迹估计器要求
    - 优势：在n_samples较小时方差更小，估计更稳定
    - 参考: Hutchinson (1990) "A Stochastic Estimator of the Trace of the Influence Matrix"

    ★关于可复现性：
    - 使用独立的torch.Generator和固定seed，不污染全局RNG流
    - 多次运行结果完全一致

    ★关于有限差分步长alpha的偏差-方差权衡：
    - alpha太小：受float32精度限制，(f(y+αω)-f(y))/α的截断误差增大
    - alpha太大：引入非线性偏差，不再近似Jacobian
    - alpha=1e-3是经验值，在float32精度和线性近似之间取折中
    - 可做敏感性检查：尝试alpha=1e-2, 1e-3, 1e-4，观察div估计是否稳定
    - 参考: Ramani et al. (2007) 建议alpha≈1e-3~1e-4
    """
    model.eval()
    div_estimates = []
    B = y.shape[0]  # batch size

    # ★使用独立的Generator和固定种子，确保可复现
    div_rng = torch.Generator(device=device)
    div_rng.manual_seed(divergence_seed)

    with torch.no_grad():
        f_y = model(y)
        for _ in range(n_samples):
            # ★改用Rademacher分布（±1），方差比高斯分布小
            # 使用torch.randint生成0/1，然后映射到-1/+1，效率更高
            omega = (torch.randint(0, 2, y.shape, generator=div_rng, device=device).float() * 2 - 1)
            f_y_perturbed = model(y + alpha * omega)
            # ★修正：除以batch_size，返回单图平均散度
            div_est = (omega * (f_y_perturbed - f_y)).sum() / alpha / B
            div_estimates.append(div_est.item())
    return np.mean(div_estimates)

# 对测试图像计算散度
# ★使用固定Generator生成噪声，与evaluate_psnr保持一致，便于复现
model_naive.eval()
test_batch, _ = next(iter(test_loader))
div_noise_rng = torch.Generator(device=device)
div_noise_rng.manual_seed(12345)  # 与evaluate_psnr的noise_seed一致
test_batch_noise = torch.randn(test_batch[:16].shape, generator=div_noise_rng, device=device)
test_batch_y = (test_batch[:16].to(device) + SIGMA * test_batch_noise).clamp(0, 1)
div_naive = compute_divergence_mc(model_naive, test_batch_y)
model_sup.eval()
div_sup = compute_divergence_mc(model_sup, test_batch_y)
print(f"\n  朴素自监督散度 div f(y) ≈ {div_naive:.2f}")
print(f"  监督模型散度 div f(y) ≈ {div_sup:.2f}")

# ★动态校验：根据Stein恒等式判断偏差方向
# E[‖x-f(y)‖²] = E[‖y-f(y)‖²] + 2σ²·div f - nσ²
# 是否"低估"取决于 2σ²·div > nσ²，即 div > n/2
# ★注：n是总像素数（含通道），MNIST灰度图C=1，复用到RGB时需要乘以C
# ★量纲一致性：div_naive是单图平均散度（已除以batch_size B），
#   n_pixels是单图像素数，两者都是单图尺度，量纲匹配。
n_pixels = test_batch_y.shape[-3] * test_batch_y.shape[-2] * test_batch_y.shape[-1]  # C×H×W
bias_term = 2 * SIGMA**2 * div_naive - n_pixels * SIGMA**2
print(f"  偏差项 = 2σ²·div f - nσ² ≈ {bias_term:.2f}")
if bias_term > 0:
    print(f"  偏差项 > 0 → 朴素损失‖y-f(y)‖²低估真实风险‖x-f(y)‖²")
else:
    print(f"  偏差项 ≤ 0 → 本次实验中散度不足以导致低估")
    # ★教学讨论：为什么小图像上bias_term可能不稳定
    # 要使偏差项>0，需要 div > n/2。这里n=1024，div需要>512。
    # 对轻量UNet(32×32 MNIST)，训练不充分时散度可能不够大。
    # 这不是代码bug，而是"小图像+低分辨率"的固有限制：
    # 小图像上-nσ²项（常数≈92）相对散度修正项2σ²·div更大，
    # Stein偏差容易被常数项"淹没"。
    # 工程上可用更大图像(如64×64)或更低σ来使偏差项更稳定为正。
    print(f"  ★教学讨论：bias_term≤0并不意味着Stein恒等式不成立，而是说明")
    print(f"    在当前设定下(n={n_pixels}, σ={SIGMA})，常数项nσ²≈{n_pixels*SIGMA**2:.1f}较大，")
    print(f"    散度修正项2σ²·div≈{2*SIGMA**2*div_naive:.1f}不足以抵消。")
    print(f"    这在小图像上是常见现象——更大的图像或更低σ会使偏差项更稳定为正。")

# --- ★★★ clamp影响的定量对照实验 ★★★ ---
# Stein引理要求 E[y'|x]=x（无偏），而clamp到[0,1]会破坏这个假设。
# 以下量化clamp对散度估计的影响。
print(f"\n  ★★★ clamp对散度估计影响的定量对照 ★★★")
print("  Stein引理要求E[y'|x]=x（无偏噪声），但add_noise后clamp到[0,1]会破坏这个假设。")
print("  以下对比'clamp'与'不clamp'下的散度估计值，量化clamp的影响：")
# 不clamp的噪声数据
test_batch_y_noclamps = (test_batch[:16].to(device) + SIGMA * torch.randn_like(test_batch[:16].to(device)))
# clamp到[-0.5, 1.5]（避免完全clamp但保持范围合理）
test_batch_y_softclamp = test_batch_y_noclamps.clamp(-0.5, 1.5)

div_naive_noclamp = compute_divergence_mc(model_naive, test_batch_y_noclamps)
div_naive_softclamp = compute_divergence_mc(model_naive, test_batch_y_softclamp)

print(f"    严格clamp [0,1]:     div_naive ≈ {div_naive:.2f}")
print(f"    软clamp [-0.5,1.5]: div_naive ≈ {div_naive_softclamp:.2f}")
print(f"    不clamp (理论值):    div_naive ≈ {div_naive_noclamp:.2f}")
print(f"    ★相对变化: clamp版 vs 不clamp版 = {(div_naive - div_naive_noclamp) / (abs(div_naive_noclamp) + 1e-6) * 100:+.1f}%")
if abs(div_naive - div_naive_noclamp) / (abs(div_naive_noclamp) + 1e-6) < 0.1:
    print(f"    结论：clamp对散度估计影响<10%，Stein恒等式近似成立")
else:
    print(f"    结论：clamp对散度估计影响显著（>10%），定量分析需考虑此偏差")

# --- ★alpha敏感性检查（定量验证有限差分步长选择） ---
print(f"\n  ★ alpha敏感性检查（验证alpha=1e-3的选择是否稳健）：")
print(f"  不同alpha下散度估计的偏差-方差权衡：alpha太小→float32精度误差，太大→非线性偏差")
alphas_to_test = [1e-2, 1e-3, 1e-4, 1e-5]
div_at_alphas = []
for a in alphas_to_test:
    d = compute_divergence_mc(model_naive, test_batch_y, n_samples=10, alpha=a)
    div_at_alphas.append(d)
    print(f"    alpha={a:.0e} → div ≈ {d:.2f}")
# 判断alpha=1e-3是否在稳定区间
div_range = max(div_at_alphas) - min(div_at_alphas)
div_mean = np.mean(div_at_alphas)
if div_range / (abs(div_mean) + 1e-6) < 0.2:
    print(f"  ★ 散度估计在alpha∈[1e-5, 1e-2]范围内变化<20%，alpha=1e-3选择稳健")
else:
    print(f"  ⚠️ 散度估计随alpha变化较大（{(div_range/(abs(div_mean)+1e-6))*100:.0f}%），建议增加n_samples或检查模型平滑性")

# --- 3c. 三种方法训练曲线对比 ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.plot(losses_sup, label=r'监督: $\|x-f(y)\|^2$', linewidth=2)
ax1.plot(losses_n2n, label=r"N2N: $\|y'-f(y)\|^2$", linewidth=2)
ax1.plot(losses_naive, label=r'朴素: $\|y-f(y)\|^2$', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('训练损失')
ax1.set_title('Step 3a: 训练损失对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

# PSNR柱状图
methods = ['监督', 'N2N', r'朴素$\|y-f(y)\|^2$']
psnrs = [psnr_sup, psnr_n2n, psnr_naive]
ssims = [ssim_sup, ssim_n2n, ssim_naive]
colors = ['#2196F3', '#4CAF50', '#FF9800']

bars = ax2.bar(methods, psnrs, color=colors, width=0.5)
for bar, v in zip(bars, psnrs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{v:.1f}dB', ha='center', fontsize=11)
ax2.set_ylabel('PSNR (dB)')
ax2.set_title('Step 3b: 去噪PSNR对比')
ax2.grid(True, alpha=0.3, axis='y')

# SSIM柱状图
bars = ax3.bar(methods, ssims, color=colors, width=0.5)
for bar, v in zip(bars, ssims):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{v:.3f}', ha='center', fontsize=11)
ax3.set_ylabel('SSIM')
ax3.set_title('Step 3c: 去噪SSIM对比')
ax3.grid(True, alpha=0.3, axis='y')

fig.suptitle(r'Step 3: 朴素自监督损失$\|y-f(y)\|^2$有偏——低估真实风险', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step3_naive_bias.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step3_naive_bias.png")


# ========================================================================
# Step 4: Neighbor2Neighbor空间配对
# 对应17.2.3节：N2N的局限与Neighbor2Neighbor缓解方案
# ★原创：从单帧噪声图像构造伪配对
# ========================================================================
print("\n" + "="*70)
print("Step 4: Neighbor2Neighbor空间配对")
print("="*70)

def neighbor_split(y):
    """将噪声图像分成两个子图像（仅使用偶数行）

    原理：空间独立的噪声在相邻像素间也独立，
    因此两个子图可近似作为同一信号的配对噪声观测

    ★简化实现：取每个2×2块的(0,0)作为子图1，(0,1)作为子图2
    即仅使用原图偶数行的像素，奇数行不参与训练信号构造
    这样两个子图的像素空间相邻（水平方向），噪声独立，满足N2B核心假设

    ★★★ 重要理论缺陷（教学要点）★★★

    本实现采用固定位置配对，与原论文（Huang et al. 2021）的随机采样策略不同：

    1. 原论文设计：每次迭代随机选择2×2块内的配对位置（如(0,0)-(0,1),
       (0,1)-(1,1), (1,1)-(1,0), (1,0)-(0,0)），避免网络学到固定的空间偏移模式

    2. 本实现：固定取(0,0)和(0,1)，每个epoch都一样

    3. 潜在风险：网络可能学到"往右挪一个像素"这种确定性的空间映射捷径，
       而不是真正的去噪。这是简化实现的教学代价

    4. 教学价值：本实验展示了"实现简化 vs 理论严格"的权衡，
       工程应用建议参考原论文的随机采样实现

    参考: Huang et al. (2021) Neighbor2Neighbor
    """
    B, C, H, W = y.shape
    # 确保 H, W 为偶数
    if H % 2 != 0:
        y = y[:, :, :-1, :]
    if W % 2 != 0:
        y = y[:, :, :, :-1]
    B, C, H, W = y.shape

    # 重塑为2×2块结构
    y_blocks = y.reshape(B, C, H//2, 2, W//2, 2)

    # 子图1：每个2×2块的(0,0)位置
    sub1 = y_blocks[:, :, :, 0, :, 0]  # shape: (B, C, H//2, W//2)
    # 子图2：每个2×2块的(0,1)位置（相邻像素，噪声独立）
    sub2 = y_blocks[:, :, :, 0, :, 1]  # shape: (B, C, H//2, W//2)

    return sub1, sub2

def neighbor_merge(sub1, sub2, H, W):
    """将两个子图合并回原始尺寸

    ★重要局限性（教学说明）：
    由于neighbor_split只取了每个2×2块的偶数行像素（第0行），
    奇数行的真实像素信息从未被采样使用。
    本函数采用棋盘式对角交换填充奇数行（非真实像素值）：
      - 偶数行：col0=sub1, col1=sub2
      - 奇数行：col0=sub2, col1=sub1（对角交换）
    这会引入额外的棋盘格伪影，使得下采样版N2B的重建质量低于真实水平。

    因此，下采样版的PSNR评估存在系统性偏差，不应与全分辨率版
    做"公平"对比。全分辨率版才是更准确的N2B实现。

    参考: Huang et al. (2021) Neighbor2Neighbor（原论文使用行平均插值）
    """
    B, C, h, w = sub1.shape  # h=H//2, w=W//2
    out = torch.zeros(B, C, H, W, device=sub1.device)
    # 将子图填回对应位置（棋盘式填充）
    out[:, :, 0::2, 0::2] = sub1  # 偶数行偶数列 ← sub1
    out[:, :, 0::2, 1::2] = sub2  # 偶数行奇数列 ← sub2
    out[:, :, 1::2, 0::2] = sub2  # 奇数行偶数列 ← sub2（对角交换，引入伪影）
    out[:, :, 1::2, 1::2] = sub1  # 奇数行奇数列 ← sub1（对角交换，引入伪影）
    return out

# --- 训练Neighbor2Neighbor ---
print("\n  训练Neighbor2Neighbor...")

"""
★ Neighbor2Neighbors 的分辨率问题与伪影风险（重要说明）：

1. 分辨率减半：
   - neighbor_split 后，图像从 32×32 变为 16×16
   - 模型在半分辨率下训练，感受野相对变大

2. ★奇数行丢失问题（关键局限性）：
   - neighbor_split 只取偶数行像素，奇数行从未被采样
   - neighbor_merge 用偶数行值复制填充奇数行，引入棋盘格伪影
   - 这使得下采样版的PSNR存在系统性偏差，低于真实水平
   - 因此，下采样版与全分辨率版的对比主要展示"实现差异"，而非"算法优劣"

3. 尺度不匹配风险：
   - 测试时直接用全分辨率输入可能导致纹理偏移
   - 本实验通过 neighbor_merge 将预测结果合并回全分辨率来缓解

4. 实践中的改进策略（论文推荐）：
   - 正则项：添加 ‖f(sub1) - sub1‖² 或类似约束，防止过度平滑
   - 数据增强：随机翻转/旋转子图对，增强泛化能力
   - 多尺度训练：混合不同下采样比例

5. 本实验的简化处理：
   - 为教学目的，本实验未添加额外正则项
   - 实际应用中建议参考原论文的完整实现（含行平均插值）
"""

model_n2nb = SmallUNet().to(device)
optimizer_n2nb = optim.Adam(model_n2nb.parameters(), lr=LR)
n2nb_ckpt_path = os.path.join(SAVE_DIR, 'ckpt_N2B.pt')
n2nb_start = 0
is_n2nb_final = False
losses_n2nb = []

# ★ Resume: 检测已有checkpoint（兼容新旧格式）
if os.path.exists(n2nb_ckpt_path):
    checkpoint = torch.load(n2nb_ckpt_path, map_location=device, weights_only=False)
    model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
    optimizer_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer_state'))
    losses_n2nb = checkpoint.get('train_losses', checkpoint.get('losses', []))

    if checkpoint.get('is_final', False):
        print(f"  [N2B] ✓ 检测到最终权重，直接加载，跳过训练过程")
        print(f"  训练轮数: {checkpoint['epoch']+1}")
        model_n2nb.load_state_dict(model_state)
        if optimizer_state:
            optimizer_n2nb.load_state_dict(optimizer_state)
        n2nb_start = checkpoint['epoch'] + 1
        is_n2nb_final = True
    else:
        print(f"  [N2B] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
        model_n2nb.load_state_dict(model_state)
        if optimizer_state:
            optimizer_n2nb.load_state_dict(optimizer_state)
        n2nb_start = checkpoint['epoch'] + 1

if is_n2nb_final or n2nb_start >= N_EPOCHS:
    print("  [N2B] 模型已训练完毕，跳过。")
else:
    model_n2nb.train()
    for epoch in range(n2nb_start, N_EPOCHS):
        epoch_loss = 0
        n_batch = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{N_EPOCHS} [N2B]', leave=False, unit='batch')
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            y = add_noise(batch_x, SIGMA)
            # 从单帧y构造伪配对
            sub1, sub2 = neighbor_split(y)
            optimizer_n2nb.zero_grad()
            pred1 = model_n2nb(sub1)
            # ★ N2B核心：用sub2作为sub1的"噪声标签"
            loss = nn.MSELoss()(pred1, sub2.detach())
            loss.backward()
            optimizer_n2nb.step()
            epoch_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        avg_loss = epoch_loss / n_batch
        losses_n2nb.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  [N2B] Epoch {epoch+1}/{N_EPOCHS}, Loss: {avg_loss:.6f}")
        # 每10轮保存checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_n2nb.state_dict(),
                'optimizer_state_dict': optimizer_n2nb.state_dict(),
                'loss': avg_loss,
                'train_losses': losses_n2nb,
                'is_final': False
            }, n2nb_ckpt_path)
            print(f"  [N2B] ✓ checkpoint已保存 (epoch {epoch+1})")

    # 保存最终checkpoint
    if losses_n2nb:
        torch.save({
            'epoch': N_EPOCHS - 1,
            'model_state_dict': model_n2nb.state_dict(),
            'loss': losses_n2nb[-1],
            'train_losses': losses_n2nb,
            'is_final': True
        }, n2nb_ckpt_path)
        print(f"  [N2B] ✓ 训练完成, 最终checkpoint已保存")

# 评估N2B：对完整尺寸输入去噪
# ★使用固定噪声（与evaluate_psnr一致），确保跨模型评估一致性
model_n2nb.eval()
psnr_n2nb_vals = []
ssim_n2nb_vals = []
n2b_noise_rng = torch.Generator(device=device)
n2b_noise_rng.manual_seed(12345)  # 与evaluate_psnr的noise_seed一致
with torch.no_grad():
    for batch_x, _ in tqdm(test_loader, desc='评估N2B PSNR', leave=False):
        batch_x = batch_x.to(device)
        noise = torch.randn(batch_x.shape, generator=n2b_noise_rng, device=device)
        y = (batch_x + SIGMA * noise).clamp(0, 1)
        # 用子图去噪后合并
        sub1, sub2 = neighbor_split(y)
        pred_sub1 = model_n2nb(sub1).clip(0, 1)
        pred_sub2 = model_n2nb(sub2).clip(0, 1)
        # 合并回原始尺寸
        pred_full = neighbor_merge(pred_sub1, pred_sub2, IMG_SIZE, IMG_SIZE)
        pred_np = pred_full.cpu().numpy()
        x_np = batch_x.cpu().numpy()
        for i in range(min(pred_np.shape[0], x_np.shape[0])):
            h = min(pred_np.shape[2], x_np.shape[2])
            w = min(pred_np.shape[3], x_np.shape[3])
            psnr_n2nb_vals.append(psnr(x_np[i, 0, :h, :w], pred_np[i, 0, :h, :w], data_range=1.0))
            ssim_n2nb_vals.append(ssim(x_np[i, 0, :h, :w], pred_np[i, 0, :h, :w], data_range=1.0))

psnr_n2nb = np.mean(psnr_n2nb_vals)
ssim_n2nb = np.mean(ssim_n2nb_vals)
# ★命名说明：n2nb = N2B下采样版（neighbor_split/merge），n2b_fullres = N2B全分辨率版
print(f"  Neighbor2Neighbor PSNR = {psnr_n2nb:.2f} dB, SSIM = {ssim_n2nb:.4f}")

# --- Step 4b: 改进版N2B（保持分辨率）---
print("\n  训练改进版N2B（保持分辨率，消除合并误差）...")

def get_subgrid_indices(H, W, device):
    """获取棋盘格子图位置的索引

    子图1：偶数行偶数列 (i%2==0, j%2==0)
    子图2：偶数行奇数列 (i%2==0, j%2==1)

    返回：
        idx_sub1: 子图1位置的扁平索引
        idx_sub2: 子图2位置的扁平索引

    ★ 与原始N2B下采样方案的对比：
    ─────────────────────────────────────────
    原始方案（neighbor_split）：
      - 物理下采样：32×32 → 16×16 子图
      - 模型在低分辨率上训练
      - 测试时需neighbor_merge合并，引入近似误差

    本方案（保持分辨率）：
      - 逻辑下采样：仅提取子图位置的索引
      - 模型在全分辨率上训练，损失只在子图位置计算
      - 测试时直接推理，无需合并，消除近似误差
    ─────────────────────────────────────────
    """
    idx_sub1 = []
    idx_sub2 = []

    for i in range(0, H, 2):
        for j in range(0, W, 2):
            idx_sub1.append(i * W + j)
            # ★注：j步长为2且W为偶数时，j+1 < W 恒成立，else分支为保险措施
            # 保留else分支是为了代码在W为奇数时也能工作（实际IMG_SIZE=32恒为偶数）
            if j + 1 < W:
                idx_sub2.append(i * W + (j + 1))
            else:
                idx_sub2.append(i * W + j)

    return (torch.tensor(idx_sub1, device=device),
            torch.tensor(idx_sub2, device=device))

def train_n2b_fullres(model, train_loader, n_epochs=N_EPOCHS, tag="N2B_FullRes"):
    """保持分辨率的N2B训练

    核心思想：
    - 模型在全分辨率上训练：f: R^{H×W} → R^{H×W}
    - 损失只在子图位置计算，用相邻像素作为target
    - 测试时直接在原图上推理，无需合并

    优点：
    1. 消除neighbor_merge引入的近似误差
    2. 训练和测试分辨率一致

    ★★★ 重要理论缺陷（教学要点）★★★

    本实现存在"感受野泄漏"问题，这是blind-spot自监督去噪方法的经典陷阱：

    1. 问题本质：
       - 原始neighbor_split版本：模型输入是16×16的下采样图，sub2位置的信息
         从未进入模型输入，天然满足blind-spot约束
       - 本版本：模型输入是完整的y，包含了位置i自身的像素值
       - SmallUNet的skip connection会让网络学到恒等映射 f(y)[i] ≈ y[i]
       - 相邻像素y[i]与y[j]（sub2的独立噪声实现）相关性很强，
         用y[i]直接逼近y[j]就能把loss压得很低

    2. 这不代表模型学会了去噪：
       - 网络可能只是学会了"抄输入"（identity shortcut）
       - 这与经典blind-spot网络文献（Laine et al. 2019）要解决的问题一致

    3. 正确的blind-spot实现需要：
       - 输入掩码：对目标位置置零或用掩码卷积
       - 或使用旋转不变架构等标准blind-spot技巧

    4. 教学价值：
       - 本实验展示了"消除合并伪影 vs 引入感受野泄漏"的权衡
       - 下采样版消除了感受野泄漏但有合并伪影
       - 全分辨率版消除了合并伪影但有感受野泄漏
       - 两者各有缺陷，而非简单的"谁更准确"

    参考: Laine et al. (2019) "High-Quality Self-Supervised Deep Image Denoising"
    """
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ckpt_path = os.path.join(SAVE_DIR, f'ckpt_{tag}.pt')
    start_epoch = 0
    is_final = False
    train_losses = []

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
        optimizer_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer_state'))
        train_losses = checkpoint.get('train_losses', checkpoint.get('losses', []))

        if checkpoint.get('is_final', False):
            print(f"  [{tag}] ✓ 检测到最终权重，直接加载，跳过训练过程")
            print(f"  训练轮数: {checkpoint['epoch']+1}")
            model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1
            is_final = True
        else:
            print(f"  [{tag}] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
            model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1

    if is_final or start_epoch >= n_epochs:
        print(f"  [{tag}] 模型已训练完毕，跳过。")
        return train_losses

    # ★效率优化：预先计算子图索引，避免在循环内重复计算
    idx_sub1, idx_sub2 = get_subgrid_indices(IMG_SIZE, IMG_SIZE, device)

    model.train()
    for epoch in range(start_epoch, n_epochs):
        epoch_loss = 0
        n_batch = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [{tag}]', leave=False, unit='batch')
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            y = add_noise(batch_x, SIGMA)
            B, C, H, W = y.shape

            pred = model(y)

            pred_flat = pred.view(B, C, -1)
            y_flat = y.view(B, C, -1)

            pred_sub1 = pred_flat[:, :, idx_sub1]
            y_sub2 = y_flat[:, :, idx_sub2]

            loss = ((pred_sub1 - y_sub2.detach()) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = epoch_loss / n_batch
        train_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  [{tag}] Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'train_losses': train_losses,
                'is_final': False
            }, ckpt_path)
            print(f"  [{tag}] ✓ checkpoint已保存 (epoch {epoch+1})")

    # 保存最终checkpoint
    if ckpt_path and train_losses:
        torch.save({
            'epoch': n_epochs - 1,
            'model_state_dict': model.state_dict(),
            'loss': train_losses[-1],
            'train_losses': train_losses,
            'is_final': True
        }, ckpt_path)
        print(f"  [{tag}] ✓ 训练完成, 最终checkpoint已保存")

    return train_losses

model_n2b_fullres = SmallUNet().to(device)
losses_n2b_fullres = train_n2b_fullres(model_n2b_fullres, train_loader)

model_n2b_fullres.eval()
psnr_n2b_fullres_vals = []
ssim_n2b_fullres_vals = []
# ★使用固定噪声（与evaluate_psnr一致），确保跨模型评估一致性
fullres_noise_rng = torch.Generator(device=device)
fullres_noise_rng.manual_seed(12345)
with torch.no_grad():
    for batch_x, _ in tqdm(test_loader, desc='评估N2B FullRes PSNR', leave=False):
        batch_x = batch_x.to(device)
        noise = torch.randn(batch_x.shape, generator=fullres_noise_rng, device=device)
        y = (batch_x + SIGMA * noise).clamp(0, 1)
        pred = model_n2b_fullres(y).clip(0, 1)
        pred_np = pred.cpu().numpy()
        x_np = batch_x.cpu().numpy()
        for i in range(pred_np.shape[0]):
            psnr_n2b_fullres_vals.append(psnr(x_np[i, 0], pred_np[i, 0], data_range=1.0))
            ssim_n2b_fullres_vals.append(ssim(x_np[i, 0], pred_np[i, 0], data_range=1.0))

psnr_n2b_fullres = np.mean(psnr_n2b_fullres_vals)
ssim_n2b_fullres = np.mean(ssim_n2b_fullres_vals)
print(f"  改进版N2B PSNR = {psnr_n2b_fullres:.2f} dB, SSIM = {ssim_n2b_fullres:.4f}")

# --- ★公平评估：仅在idx_sub1位置计算PSNR/SSNR ---
# 目的：让下采样版（仅在sub1位置有真实预测）和全分辨率版使用相同评估口径
# 避免下采样版的"合并伪影"奇数行被错误地拉低分数
print(f"\n  ★公平评估：仅在idx_sub1位置计算（与下采样版监督范围一致）")
model_n2b_fullres.eval()
idx_sub1_fair, _ = get_subgrid_indices(IMG_SIZE, IMG_SIZE, device)
psnr_n2b_fullres_fair = []
ssim_n2b_fullres_fair = []
fair_noise_rng = torch.Generator(device=device)
fair_noise_rng.manual_seed(12345)
with torch.no_grad():
    for batch_x, _ in tqdm(test_loader, desc='公平评估N2B FullRes', leave=False):
        batch_x = batch_x.to(device)
        noise = torch.randn(batch_x.shape, generator=fair_noise_rng, device=device)
        y = (batch_x + SIGMA * noise).clamp(0, 1)
        pred = model_n2b_fullres(y).clip(0, 1)
        B, C, H, W = pred.shape
        pred_flat = pred.view(B, C, -1)
        x_flat = batch_x.view(B, C, -1)
        # ★只取idx_sub1位置（与下采样版评估口径一致）
        pred_sub1 = pred_flat[:, :, idx_sub1_fair]
        x_sub1 = x_flat[:, :, idx_sub1_fair]
        # 重塑为2D图像计算PSNR/SSIM
        h_sub = H // 2
        w_sub = W // 2
        pred_2d = pred_sub1.view(B, C, h_sub, w_sub).cpu().numpy()
        x_2d = x_sub1.view(B, C, h_sub, w_sub).cpu().numpy()
        for i in range(pred_2d.shape[0]):
            psnr_n2b_fullres_fair.append(psnr(x_2d[i, 0], pred_2d[i, 0], data_range=1.0))
            ssim_n2b_fullres_fair.append(ssim(x_2d[i, 0], pred_2d[i, 0], data_range=1.0))

psnr_n2b_fullres_fair_mean = np.mean(psnr_n2b_fullres_fair)
ssim_n2b_fullres_fair_mean = np.mean(ssim_n2b_fullres_fair)
print(f"  公平评估（全分辨率版）: PSNR = {psnr_n2b_fullres_fair_mean:.2f} dB, SSIM = {ssim_n2b_fullres_fair_mean:.4f}")

# --- ★偶数行vs奇数行PSNR对比（量化"卷积空间共享泛化"假设） ---
print(f"\n  ★★ 偶数行vs奇数行PSNR对比（验证卷积空间共享泛化假设）★★")
print("  两种N2B实现都只训练了偶数行（sub1位置），奇数行依赖'卷积核空间共享→泛化'。")
print("  量化对比可以验证这个隐含假设是否成立：")
model_n2b_fullres.eval()
# 定义偶数行/奇数行的索引
even_row_idx = []
odd_row_idx = []
for i in range(IMG_SIZE):
    for j in range(IMG_SIZE):
        if i % 2 == 0:
            even_row_idx.append(i * IMG_SIZE + j)
        else:
            odd_row_idx.append(i * IMG_SIZE + j)
even_row_idx = torch.tensor(even_row_idx, device=device)
odd_row_idx = torch.tensor(odd_row_idx, device=device)

psnr_even_fullres = []
psnr_odd_fullres = []
evenodd_noise_rng = torch.Generator(device=device)
evenodd_noise_rng.manual_seed(12345)
with torch.no_grad():
    for batch_x, _ in tqdm(test_loader, desc='偶/奇行PSNR评估', leave=False):
        batch_x = batch_x.to(device)
        noise = torch.randn(batch_x.shape, generator=evenodd_noise_rng, device=device)
        y = (batch_x + SIGMA * noise).clamp(0, 1)
        pred = model_n2b_fullres(y).clip(0, 1)
        B, C, H, W = pred.shape
        pred_flat = pred.view(B, C, -1)
        x_flat = batch_x.view(B, C, -1)
        pred_even = pred_flat[:, :, even_row_idx]
        x_even = x_flat[:, :, even_row_idx]
        pred_odd = pred_flat[:, :, odd_row_idx]
        x_odd = x_flat[:, :, odd_row_idx]
        for i in range(B):
            pe_2d = pred_even[i, 0].view(H // 2, W).cpu().numpy()
            xe_2d = x_even[i, 0].view(H // 2, W).cpu().numpy()
            po_2d = pred_odd[i, 0].view(H // 2, W).cpu().numpy()
            xo_2d = x_odd[i, 0].view(H // 2, W).cpu().numpy()
            psnr_even_fullres.append(psnr(xe_2d, pe_2d, data_range=1.0))
            psnr_odd_fullres.append(psnr(xo_2d, po_2d, data_range=1.0))

psnr_even_mean = np.mean(psnr_even_fullres)
psnr_odd_mean = np.mean(psnr_odd_fullres)
gap_even_odd = psnr_even_mean - psnr_odd_mean
print(f"    偶数行PSNR (被监督位置): {psnr_even_mean:.2f} dB")
print(f"    奇数行PSNR (未被监督):   {psnr_odd_mean:.2f} dB")
print(f"    差距: {gap_even_odd:+.2f} dB")
if abs(gap_even_odd) < 1.0:
    print(f"    ★ 差距<1dB → 卷积空间共享泛化假设成立")
else:
    print(f"    ⚠️  差距>1dB → 空间共享泛化能力有限，奇数行质量不可靠")
print(f"  （注：此差距可作为'感受野泄漏'的辅助证据）")

# --- ★诊断实验：验证identity shortcut是否发生 ---
print(f"\n  ★★ 感受野泄漏诊断（教学要点）★★")
print(f"  全分辨率版可能存在identity shortcut问题，以下进行定量诊断：")
print(f"  ★同时计算model_sup、model_n2n作为对照组，验证0.9阈值是否合理")
model_n2b_fullres.eval()
# ★获取子图索引（与训练时一致）
idx_sub1_diag, idx_sub2_diag = get_subgrid_indices(IMG_SIZE, IMG_SIZE, device)

# ★定义一个函数计算f(y)与y在idx_sub1位置的相关系数（移到with块外面，更清晰）
def compute_identity_correlation(model, y, idx_sub1):
    """计算模型输出f(y)与输入y在指定位置的相关系数"""
    model.eval()
    with torch.no_grad():
        pred = model(y)
        B, C, H, W = pred.shape
        pred_flat = pred.view(B, C, -1)
        y_flat = y.view(B, C, -1)
        pred_sub1 = pred_flat[:, :, idx_sub1].flatten()
        y_sub1 = y_flat[:, :, idx_sub1].flatten()
        correlation = np.corrcoef(y_sub1.cpu().numpy(), pred_sub1.cpu().numpy())[0, 1]
        return correlation

with torch.no_grad():
    test_diag, _ = next(iter(test_loader))
    test_diag_y = add_noise(test_diag[:64].to(device), SIGMA)
    pred_diag = model_n2b_fullres(test_diag_y)

    # ★只在被监督的idx_sub1位置计算相关性（与训练监督范围一致）
    B, C, H, W = pred_diag.shape
    pred_flat = pred_diag.view(B, C, -1)
    y_flat = test_diag_y.view(B, C, -1)

    pred_sub1_diag = pred_flat[:, :, idx_sub1_diag].flatten()
    y_sub1_diag = y_flat[:, :, idx_sub1_diag].flatten()

    correlation = np.corrcoef(y_sub1_diag.cpu().numpy(), pred_sub1_diag.cpu().numpy())[0, 1]
    mse_identity = ((pred_sub1_diag - y_sub1_diag) ** 2).mean().item()

    # ★计算对照组：model_sup和model_n2n在相同位置的相关系数
    # 这两个模型不是blind-spot网络，相关系数代表"正常基线"
    corr_sup = compute_identity_correlation(model_sup, test_diag_y, idx_sub1_diag)
    corr_n2n = compute_identity_correlation(model_n2n, test_diag_y, idx_sub1_diag)

    print(f"    f(y)[sub1]与y[sub1]的相关系数 (对照组与待诊断模型):")
    print(f"      model_sup (正常基线):       {corr_sup:.4f}")
    print(f"      model_n2n (正常基线):        {corr_n2n:.4f}")
    print(f"      model_n2b_fullres (待诊断): {correlation:.4f}")
    print(f"    f(y)[sub1]与y[sub1]的MSE (待诊断): {mse_identity:.6f}")
    print(f"  （仅在训练时被监督的idx_sub1位置计算，更准确反映identity shortcut风险）")
    # ★动态校验：使用相对阈值（待诊断-正常基线）
    # 注意：0.1和0.05是教学经验值，非统计显著性检验阈值
    # 实际判断应结合具体场景和领域知识
    baseline = (corr_sup + corr_n2n) / 2  # 正常基线均值
    excess = correlation - baseline
    print(f"    正常基线均值: {baseline:.4f}, 待诊断模型超出量: {excess:+.4f}")
    if excess > 0.1:  # 经验阈值，仅供参考
        print(f"    ⚠️  警告：超出正常基线{excess:.3f}，存在identity shortcut风险")
    elif excess > 0.05:  # 经验阈值，仅供参考
        print(f"    相关系数略高于正常基线，需关注但风险有限")
    else:
        print(f"    相关系数与正常基线相当，identity shortcut风险较低")
    print(f"  （注：理想的blind-spot实现应确保目标像素不在感受野内）")

print(f"\n  ★ N2B两种实现对比：")
print(f"    下采样+合并版：PSNR = {psnr_n2nb:.2f} dB, SSIM = {ssim_n2nb:.4f}")
print(f"    保持分辨率版：  PSNR = {psnr_n2b_fullres:.2f} dB, SSIM = {ssim_n2b_fullres:.4f}")
if psnr_n2b_fullres > psnr_n2nb:
    print(f"    改进版提升：    +{psnr_n2b_fullres - psnr_n2nb:.2f} dB")
else:
    print(f"    性能差异：      {psnr_n2b_fullres - psnr_n2nb:.2f} dB")

# --- 可视化六种方法对比 ---
vis_imgs, _ = next(test_iter)  # ★使用共享test_iter，与前两步不同batch
vis_imgs = vis_imgs[:3].to(device)
vis_noisy = add_noise(vis_imgs, SIGMA)

with torch.no_grad():
    pred_sup_vis = model_sup(vis_noisy).cpu().clip(0, 1)
    pred_n2n_vis = model_n2n(vis_noisy).cpu().clip(0, 1)
    pred_naive_vis = model_naive(vis_noisy).cpu().clip(0, 1)
    # N2B 下采样+合并版
    sub1_v, sub2_v = neighbor_split(vis_noisy)
    # ★变量名更具体：pred_s1_n2b/pred_s2_n2b 避免与Step 1的 pred_s1/pred_s2 混淆
    pred_s1_n2b = model_n2nb(sub1_v).clip(0, 1)
    pred_s2_n2b = model_n2nb(sub2_v).clip(0, 1)
    pred_n2b_vis = neighbor_merge(pred_s1_n2b, pred_s2_n2b, IMG_SIZE, IMG_SIZE).cpu()
    # N2B 保持分辨率版
    pred_n2b_fullres_vis = model_n2b_fullres(vis_noisy).cpu().clip(0, 1)

fig, axes = plt.subplots(7, 3, figsize=(10, 21))
row_labels = ['干净图像x', '噪声输入y',
              f'监督 ({psnr_sup:.1f}dB, SSIM={ssim_sup:.3f})',
              f'N2N ({psnr_n2n:.1f}dB, SSIM={ssim_n2n:.3f})',
              f'朴素 ({psnr_naive:.1f}dB, SSIM={ssim_naive:.3f})',
              f'N2B下采样 ({psnr_n2nb:.1f}dB, SSIM={ssim_n2nb:.3f})',
              f'N2B全分辨率 ({psnr_n2b_fullres:.1f}dB, SSIM={ssim_n2b_fullres:.3f})']

for i in range(3):
    axes[0, i].imshow(vis_imgs[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    axes[1, i].imshow(vis_noisy[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
    axes[2, i].imshow(pred_sup_vis[i, 0], cmap='gray', vmin=0, vmax=1)
    axes[2, i].axis('off')
    axes[3, i].imshow(pred_n2n_vis[i, 0], cmap='gray', vmin=0, vmax=1)
    axes[3, i].axis('off')
    axes[4, i].imshow(pred_naive_vis[i, 0], cmap='gray', vmin=0, vmax=1)
    axes[4, i].axis('off')
    # N2B下采样版
    axes[5, i].imshow(pred_n2b_vis[i, 0], cmap='gray', vmin=0, vmax=1)
    axes[5, i].axis('off')
    # N2B全分辨率版
    axes[6, i].imshow(pred_n2b_fullres_vis[i, 0], cmap='gray', vmin=0, vmax=1)
    axes[6, i].axis('off')

for r, label in enumerate(row_labels):
    axes[r, 0].set_ylabel(label, fontsize=10, rotation=0, labelpad=80)

fig.suptitle('Step 4: 六种去噪方法完整对比', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step4_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step4_comparison.png")

# --- PSNR/SSIM汇总 ---
print("\n" + "="*70)
print("实验17.2-1 总结")
print("="*70)
print(f"  方法                  PSNR (dB)    SSIM       说明")
print(f"  ─────────────────────────────────────────────────────────")
# ★使用固定宽度格式化，确保数值变化时列对齐
print(f"  监督 ‖x-f(y)‖²       {psnr_sup:6.2f}       {ssim_sup:6.4f}    基线（需要配对数据）")
print(f"  N2N  ‖y'-f(y)‖²      {psnr_n2n:6.2f}       {ssim_n2n:6.4f}    等价于监督（需配对噪声）")
print(f"  朴素 ‖y-f(y)‖²       {psnr_naive:6.2f}       {ssim_naive:6.4f}    有偏——低估风险")
print(f"  N2B  下采样+合并      {psnr_n2nb:6.2f}       {ssim_n2nb:6.4f}    ⚠️系统性偏低，仅供参考")
print(f"  N2B  保持分辨率        {psnr_n2b_fullres:6.2f}       {ssim_n2b_fullres:6.4f}    消除合并误差")
print(f"  ─────────────────────────────────────────────────────────")
print(f"  ★公平评估（仅在idx_sub1位置，两种N2B可比口径）：")
print(f"  N2B  保持分辨率(公平)  {psnr_n2b_fullres_fair_mean:6.2f}       {ssim_n2b_fullres_fair_mean:6.4f}    消除合并误差+公平口径")
print(f"\n  ★注：下采样+合并版的全图PSNR因奇数行棋盘伪影填充而系统性偏低，")
print(f"        上方'公平评估'行与下采样版使用相同评估口径（仅sub1位置），可做苹果对苹果比较。")
print(f"\n  核心结论:")
# ★动态校验：根据实际PSNR值生成结论，而非硬编码断言
n2n_sup_diff = abs(psnr_n2n - psnr_sup)
if n2n_sup_diff < 2.0:  # 2dB阈值，教学实验中认为"≈"
    print(f"  1. N2N ≈ 监督 (相差{n2n_sup_diff:.2f}dB) → 验证了零均值噪声下 E[y'|x]=x 的等价性")
else:
    print(f"  1. N2N与监督差异较大 ({n2n_sup_diff:.2f}dB)，可能与随机性或训练不足有关")

if psnr_naive < min(psnr_sup, psnr_n2n):
    print(f"  2. 朴素MSE偏低 ({psnr_naive:.2f}dB) → ‖y-f(y)‖²系统低估真实风险(散度偏差)")
else:
    print(f"  2. 反常：朴素MSE并未最低，与理论预期不符，建议检查训练过程")

if psnr_n2nb < psnr_n2n:
    print(f"  3. N2B ≈ N2N-δ (低{psnr_n2n - psnr_n2nb:.2f}dB) → 空间独立性假设下的近似配对")
else:
    print(f"  3. N2B接近或超过N2N，与理论预期一致")

print(f"\n  ★ N2B两种实现对比（简表）：")
print(f"  ┌───────────────┬──────────────────────┬──────────────────────┐")
print(f"  │               │  下采样+合并版        │  保持分辨率版        │")
print(f"  ├───────────────┼──────────────────────┼──────────────────────┤")
print(f"  │ 盲点约束      │  ✓ 天然满足           │  ✗ 存在感受野泄漏    │")
print(f"  │               │  (sub2不在输入中)     │  (输入包含完整y)     │")
print(f"  ├───────────────┼──────────────────────┼──────────────────────┤")
print(f"  │ 分辨率        │  ✗ 减半(32→16)       │  ✓ 保持(32×32)      │")
print(f"  ├───────────────┼──────────────────────┼──────────────────────┤")
print(f"  │ 合并伪影      │  ✗ 棋盘式对角交换    │  ✓ 无合并步骤        │")
print(f"  │               │  (奇数行非真实像素)   │                      │")
print(f"  ├───────────────┼──────────────────────┼──────────────────────┤")
print(f"  │ identity风险  │  ✓ 无                │  ✗ 可能f(y)≈y       │")
print(f"  ├───────────────┼──────────────────────┼──────────────────────┤")
print(f"  │ PSNR评估      │  ⚠️ 系统性偏低       │  可能虚高(若泄漏)    │")
print(f"  └───────────────┴──────────────────────┴──────────────────────┘")
print(f"  性能差异：{psnr_n2b_fullres - psnr_n2nb:+.2f} dB（保持分辨率版）")
# ★动态校验：根据诊断结果给出风险评估
if correlation > 0.9:
    print(f"  ★诊断：相关系数{correlation:.2f}>0.9，identity shortcut风险明显")
    print(f"  建议：全分辨率版PSNR可能虚高，下采样版更可靠")
else:
    print(f"  ★诊断：相关系数{correlation:.2f}≤0.9，identity shortcut风险较低（但仍需关注理论缺陷）")
print(f"  ★结论：两种实现各有缺陷，正确的blind-spot实现需输入掩码等额外措施")
print(f"\n  ★ 从N2N到SURE的逻辑演进（关键发现）：")
print(f"  ─────────────────────────────────────────────────────────")
print(f"  本实验Step 3量化了朴素MSE的偏差来源：")
print(f"    SURE修正项 = 2σ²·div f(y) ≈ {2*SIGMA**2*div_naive:.4f}")
print(f"  （注：完整的偏差项 = 2σ²·div f - nσ²，这里仅展示SURE损失构造所需的修正部分）")
print(f"  ")
print(f"  核心洞察：如果把这个修正项\"加回去\"，即：")
print(f"    L_SURE = ‖y-f(y)‖² + 2σ²·div f(y)")
print(f"  （注：完整SURE公式为 ‖y-f(y)‖² - nσ² + 2σ²·div f(y)，")
print(f"   此处省略了常数项 -nσ²，因常数不影响梯度优化）")
print(f"  ")
print(f"  ★理论预告（非本实验实测）：")
print(f"  这一发现将是下一实验（SURE）的出发点——理论上加入偏差修正项后，")
print(f"  无需y₂（配对噪声）即可逼近N2N效果。我们将在下一实验中训练验证这一点。")
print(f"  ─────────────────────────────────────────────────────────")
print(f"\n  ★ 关于SSIM指标：")
print(f"     - SSIM更关注结构相似性，能更好反映人眼感知质量")
# ★动态校验：检查N2B的SSIM与PSNR差距的关系
ssim_gap_n2nb = ssim_n2n - ssim_n2nb  # N2N比N2B的SSIM差距
psnr_gap_n2nb = psnr_n2n - psnr_n2nb  # N2N比N2B的PSNR差距
if ssim_gap_n2nb < psnr_gap_n2nb:
    print(f"     - 本次实验：N2B的SSIM差距({ssim_gap_n2nb:.4f})小于PSNR差距({psnr_gap_n2nb:.2f}dB)，结构保持较好")
else:
    print(f"     - 本次实验：N2B的SSIM差距({ssim_gap_n2nb:.4f})与PSNR差距({psnr_gap_n2nb:.2f}dB)相近")
print(f"\n  ★ 关于收敛曲线（建议学生观察）：")
# ★动态校验：检查收敛速度
final_loss_sup = losses_sup[-1] if losses_sup else 0
final_loss_n2n = losses_n2n[-1] if losses_n2n else 0
final_loss_n2b = losses_n2b_fullres[-1] if losses_n2b_fullres else 0
print(f"     - 本次最终损失：监督={final_loss_sup:.4f}, N2N={final_loss_n2n:.4f}, N2B={final_loss_n2b:.4f}")
print(f"     - 可尝试增加EPOCHS观察不同方法的最终收敛性能")
print(f"     - 朴素方法的损失下降最快但PSNR最低——这是有偏估计的典型特征")

print(f"\n{'='*70}")
print("★ 已知代码工程问题（供未来改进参考，未影响本次实验正确性）")
print("="*70)
print("  1. checkpoint存取逻辑高度重复：quick_train、train_synthetic_pair、train_model、")
print("     train_n2n_model、train_n2b_fullres及N2B内联训练均复制了")
print("     'load checkpoint → 判断is_final → resume → 训练循环 → 保存'逻辑。")
print("     建议未来重构为统一的resumable_train(model, ckpt_name, step_fn, ...)，")
print("     其中step_fn(batch_x) -> loss作为参数传入，可减少代码量约50%并避免不一致。")
print("  2. 可复现性：torch.manual_seed(42)只设一次，断点续训会改变后续RNG状态，")
print("     导致'从头跑'与'从checkpoint恢复'的最终数值略有差异。")
print("     工程应用建议为每个训练函数传入独立的torch.Generator。")
