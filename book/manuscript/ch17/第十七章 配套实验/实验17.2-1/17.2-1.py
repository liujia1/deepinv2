# -*- coding: utf-8 -*-
"""
实验17.2-1 Noise2Noise与自监督去噪
对应知识点：17.1节（数据困境与学习设定谱系）、17.2节（Noise2Noise原理）

实验内容：
Step 1: 各学习设定的数据构造与对比 —— 在MNIST上构造各设定的数据（含正文设定4的"配对噪声观测"子档）
Step 2: Noise2Noise原理验证 —— 用配对噪声数据训练，对比监督基线
Step 3: Noise2Noise与朴素自监督损失对比 —— 展示‖y-f(y)‖²的偏差
Step 4: Neighbor2Neighbor空间配对 —— 利用邻域结构近似配对

★原创设计：
- 在MNIST上系统对比四种学习设定的训练效果
- 用散度分析可视化朴素MSE的偏差来源
- Neighbor2Neighbor从单帧噪声图像构造伪配对

素材来源：MiniProject_Self_Supervised中N2N思路、deepinv.loss API
运行前提：需GPU（Colab T4即可）

★v2修订（修复v1结果与理论预期不符的问题）：
1. 噪声管线去掉clamp：N2N理论要求E[y'|x]=x，v1的clamp(0,1)在σ=0.3下使背景
   产生≈σ/√(2π)≈0.12的系统性正偏差，导致N2N(≈18.3dB)与监督(≈26.5dB)相差
   约8dB，掩盖了"N2N≈监督"的核心教学结论。v2起训练/评估均不clamp。
2. checkpoint文件名统一加_v2后缀：旧权重基于clamp管线训练，与新管线评估
   不匹配，通过后缀自动隔离，避免resume误加载。
3. N2B全分辨率版的identity诊断改为对齐训练目标y[sub2]：v1误测f(y)[sub1]
   与y[sub1]的相关性，漏检了"直接抄目标像素"的identity shortcut。

★v3修订（修复N2B下采样版的实现缺陷）：
v2的neighbor_split只取每个2×2块的(0,0)/(0,1)两个像素，原图一半像素（奇数行）
从未进入训练信号；neighbor_merge又用"对角交换"把奇数行填成非真实像素，
既损失训练数据又引入棋盘伪影，使下采样版PSNR降到12.2dB——该数字反映的是
**实现缺陷**而非算法固有局限。此前把该差距归因为"空间独立性假设下的近似配对"
属于错误归因。v3改用Huang et al. (2021)原文的随机邻域降采样：
  - 每个2×2块随机选一对相邻像素构成sub1，互补的两个像素构成sub2
    （两张子图各含2像素/块，尺寸(B,2C,H/2,W/2)，与原论文一致）
  - 随机模式使网络无法学到固定的空间偏移捷径
  - sub1与sub2的像素位置互补且无重叠，配合记录下来的模式可**无损还原**原图，
    因此下采样版PSNR恢复为全图口径、可直接与其他方法比较，不再需要"公平评估"补丁
  - N2B模型输入/输出通道数由1改为2（对应块内2个像素），旧权重不兼容，
    通过N2B_CKPT_VER='v3'自动隔离
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
QUICK_EPOCHS = 5  # Step1快速对比的训练轮数
LR = 1e-3
# ★checkpoint版本后缀：v2起噪声管线去掉clamp（见add_noise说明），
# 旧v1权重基于clamp噪声训练，与新管线的训练/评估不匹配，
# 通过版本后缀自动隔离，避免resume逻辑误加载旧权重导致结果混杂
CKPT_VER = 'v2'
# ★N2B专用版本后缀：v3起Step 4改用原论文的随机邻域降采样（2通道子图），
# 与v2的1通道固定配对权重不兼容，单独设版本避免影响其他已训练好的模型
N2B_CKPT_VER = 'v3'

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
    """给干净图像添加高斯噪声（★v2起不clamp）

    ★ 为什么必须去掉clamp（v1的核心缺陷）：
    - N2N 的理论保证依赖于 E[y'|x] = x，即噪声零均值且无截断
    - v1在加噪后clamp到[0,1]，背景像素(x=0)处 E[clamp(ε)] = σ/√(2π) ≈ 0.12，
      相当于给N2N的训练目标注入了系统性正偏差
    - 后果（v1实测）：N2N把背景学成"灰雾"，PSNR被压到≈18.4dB的理论上限，
      与监督(≈26.5dB)相差约8dB，完全掩盖了"N2N≈监督"的教学结论
    - v2起训练与评估均不clamp；可视化时matplotlib按vmin/vmax自动截断显示，
      PSNR评估对模型输出pred.clip(0,1)仅影响指标计算，不涉及训练假设
    """
    return x + sigma * torch.randn_like(x)


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
            y = batch_x + sigma * noise  # ★v2：不clamp，与训练管线一致
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
test_noisy2 = add_noise(test_imgs, SIGMA)  # ★第二次独立噪声观测 y'（用于“配对噪声观测”子档 = 正文设定4的子档）

# ★命名规范说明（避免后续s1/s2术语混淆）：
# - 本文件Step 1使用s1/s2/s3/s4指代四行数据（learning setting）：
#   s1 = 设定1（监督）；s2 = 设定4的“配对噪声观测”子档（★不是正文的设定2！）；
#   s3 = 设定3（仅x）；s4 = 设定4（仅y）
#   正文17.1 的设定2 是“干净x + 已知噪声模型”，其训练方式与设定1同构，本实验未单列。
# - Step 4起使用sub1/sub2指代"子图1/子图2"（neighbor sub-image）
# - 两者是不同概念，请按上下文区分

# --- 1a. 数据可视化 ---
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
titles = ['设定1: 监督 (x,y)', "设定4子档: 配对噪声 (y,y')",
          '设定3: 无监督-x (仅x)', '设定4: 无监督-y (仅y)']
for i in range(8):
    # 设定1: 监督学习 - 有配对的干净x和噪声y
    axes[0, i].imshow(test_imgs[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    # 设定4子档: 配对噪声观测 - 有两次独立噪声观测y, y'（这里展示y'，y在设定4行展示）
    axes[1, i].imshow(test_noisy2[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
    # 设定3: 无监督-x - 仅有干净图像（本实验框架下无法开工；正文指出得分匹配/扩散正站此档，见第6–7章）
    axes[2, i].imshow(test_imgs[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[2, i].axis('off')
    # 设定4: 无监督-y - 仅需要噪声观测（本章核心挑战）
    axes[3, i].imshow(test_noisy[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[3, i].axis('off')

for r in range(4):
    axes[r, 0].set_ylabel(titles[r], fontsize=9, rotation=0, labelpad=100)

fig.suptitle(f'Step 1a: 各学习设定的数据对比（含设定4子档） ($\\sigma$={SIGMA:.1f})', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step1a_four_settings_data.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step1a_four_settings_data.png")
print("  说明: 设定1/2/3需要干净数据x；设定4（含其配对噪声子档）只需噪声观测y——本章的核心挑战")

# --- 1b. 各设定的可行性与性能对比实验 ---
# 设定3（仅x）在“两两映射式”训练框架下无法开工：既没有 y，也没有噪声模型。
# 但这不是“不可训练”——正文17.1指出得分匹配/扩散（第6–7章）正是用仅有的 x 学 p(x) 的得分，
# 再反推去噪器，只是超出本实验范围。故这里只对比设定1、设定4的配对噪声子档、设定4。

# ★说明：Step1的model_s1/model_s2与Step2的model_sup/model_n2n是独立实验，非复用
# - Step1：快速训练（5 epochs），用于各设定的定性对比，了解各设定的相对性能
# - Step2：完整训练（30 epochs），用于N2N原理的定量验证
# 虽然loss函数相同，但训练目的不同，Step1侧重"哪种设定可行"，Step2侧重"N2N是否等价于监督"
print(f"\n  快速训练对比各设定的性能差异（各训练{QUICK_EPOCHS}个epoch）...")

def quick_train(model, loss_fn, n_epochs=QUICK_EPOCHS, tag=""):
    """快速训练用于对比（支持resume，兼容新旧checkpoint格式）"""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ckpt_path = os.path.join(SAVE_DIR, f'ckpt_quick_{tag}_{CKPT_VER}.pt') if tag else None
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

# 设定4 的“配对噪声观测”子档 (有两次独立噪声观测)
# ★术语更正：正文17.1 的设定2 是“干净x + 已知噪声模型”，与这里的 (y,y') 不是一回事；
#   只有噪声观测、且有两帧独立噪声，属于正文设定4 里明确提到的“特殊子档”。
# 训练算法本质就是N2N损失‖y'-f(y)‖²，与Step 2的model_n2n是同一件事；
# 区别仅在于：这里用 QUICK_EPOCHS 快速验证可行性，Step 2 用 N_EPOCHS 完整训练。
# 变量名 s2 / 文件名 S2-synthetic-pair 沿用旧名，请按本注释理解。
def train_synthetic_pair(model, n_epochs=QUICK_EPOCHS):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ckpt_path = os.path.join(SAVE_DIR, f'ckpt_quick_S2-synthetic-pair_{CKPT_VER}.pt')
    start_epoch = 0
    is_final = False
    train_losses = []

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
        optimizer_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer_state'))
        train_losses = checkpoint.get('train_losses', checkpoint.get('losses', []))

        if checkpoint.get('is_final', False):
            print(f"  [设定4子档-配对噪声] ✓ 检测到最终权重，直接加载，跳过训练过程")
            print(f"  训练轮数: {checkpoint['epoch']+1}")
            model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1
            is_final = True
        else:
            print(f"  [设定4子档-配对噪声] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
            model.load_state_dict(model_state)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1

    if is_final or start_epoch >= n_epochs:
        print(f"  [设定4子档-配对噪声] 模型已训练完毕，跳过。")
        psnr_val, ssim_val = evaluate_psnr(model, test_loader)
        return psnr_val, ssim_val

    model.train()
    for epoch in range(start_epoch, n_epochs):
        epoch_loss = 0.0
        n_batch = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [设定4子档-配对噪声]', leave=False, unit='batch')
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

# 设定3: 无监督-x (仅干净图像) - 本实验不训练
# 解释：本实验的所有训练路径都要求“输入是带噪观测 y”，设定3 连 y 都没有；
#       正文17.1 指出该档应改用得分匹配/扩散（第6–7章），超出本实验范围。
psnr_s3 = None  # 占位：设定3 不在本实验的训练范围内
ssim_s3 = None  # 占位：同上

# 设定4: 无监督-y (仅噪声观测) - 朴素方法
model_s4 = SmallUNet().to(device)
psnr_s4, ssim_s4 = quick_train(model_s4, lambda p, x, y: nn.MSELoss()(p, y), tag="S4-naive")

print(f"\n  各设定（含设定4子档）的快速训练结果（{QUICK_EPOCHS} epochs）:")
print(f"    设定1 (监督):        PSNR = {psnr_s1:.2f} dB, SSIM = {ssim_s1:.4f}")
print(f"    设定4子档 (配对噪声): PSNR = {psnr_s2:.2f} dB, SSIM = {ssim_s2:.4f}")
print(f"    设定3 (仅x):         本实验不训练（需得分匹配/扩散，见第6–7章）")
print(f"    设定4 (仅y-朴素):    PSNR = {psnr_s4:.2f} dB, SSIM = {ssim_s4:.4f}")

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
    (f'设定4子档: 配对噪声\n{psnr_s2:.1f}dB', pred_s2[0,0], '#4CAF50'),
    (f'设定3: 仅x\n本实验不训练', vis_img[0,0].cpu(), '#9E9E9E'),
    (f'设定4: 仅y-朴素\n{psnr_s4:.1f}dB', pred_s4[0,0], '#FF9800'),
]

for ax, (title, img, color) in zip(axes, methods):
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10, color=color if '本实验不训练' not in title else 'gray')
    ax.axis('off')

fig.suptitle(f'Step 1b: 各学习设定的性能对比 ($\\sigma$={SIGMA:.1f}, {QUICK_EPOCHS} epochs)', fontsize=13)
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
    ckpt_path = os.path.join(SAVE_DIR, f'ckpt_{tag}_{CKPT_VER}.pt') if tag else None
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
    ckpt_path = os.path.join(SAVE_DIR, f'ckpt_{tag}_{CKPT_VER}.pt')
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
# - Step1的model_s4：仅5 epochs，用于各设定的快速对比
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
# ★v2：主口径不clamp（与训练/评估管线一致，保证Stein恒等式的零均值前提成立）
test_batch_y = test_batch[:16].to(device) + SIGMA * test_batch_noise
test_batch_y_clamped = test_batch_y.clamp(0, 1)  # 仅用于clamp对照实验
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
    # ★教学讨论：为什么bias_term可能≤0
    # bias_term = σ²(2·div − n)，σ²是公因子不影响符号；bias_term>0 等价于 div > n/2。
    # 这里n=1024，div需>512，即模型必须足够"保噪"：恒等式映射div≈n
    # （v2实测朴素模型div≈n，bias_term≈+92>0，走的是上一分支）。
    # 走到本分支说明当前训练出的模型散度不足n/2（尚未收敛到近恒等映射），
    # 例如训练轮数不足。注意：该判据是尺度不变的（div与n同随像素数增长），
    # 并非"小图像的固有限制"，而是训练动力学现象。
    # 工程上应增加训练轮数使朴素模型更"保噪"，而非换大图像/调σ来改变符号。
    print(f"  ★教学讨论：bias_term≤0并不意味着Stein恒等式不成立，而是说明")
    print(f"    在当前设定下(n={n_pixels}, σ={SIGMA})，散度修正项2σ²·div≈{2*SIGMA**2*div_naive:.1f}")
    print(f"    小于常数项nσ²≈{n_pixels*SIGMA**2:.1f}，即div={div_naive:.1f} < n/2={n_pixels/2:.0f}，")
    print(f"    模型尚未学到足够保噪（近恒等）的映射。增加训练轮数可使偏差项更稳定为正。")

# --- ★★★ clamp影响的定量对照实验 ★★★ ---
# ★v2说明：主口径（div_naive/bias_term）已不clamp，Stein引理前提E[y|x]=x严格成立。
# 本对照用同一份噪声量化"若clamp到[0,1]"会使散度估计偏离多少（教学参考）。
print(f"\n  ★★★ clamp对散度估计影响的定量对照（v2主口径已不clamp） ★★★")
print("  Stein引理要求E[y'|x]=x（无偏噪声）。v2管线已不clamp，该前提严格成立；")
print("  以下用同一份噪声对比'clamp到[0,1]'与'不clamp'输入下的散度估计差异：")
div_naive_clamped = compute_divergence_mc(model_naive, test_batch_y_clamped)

print(f"    不clamp（v2主口径，Stein前提成立）: div_naive ≈ {div_naive:.2f}")
print(f"    clamp [0,1]（v1旧管线的做法）:      div_naive ≈ {div_naive_clamped:.2f}")
print(f"    ★相对变化: clamp版 vs 不clamp版 = {(div_naive_clamped - div_naive) / (abs(div_naive) + 1e-6) * 100:+.1f}%")
if abs(div_naive_clamped - div_naive) / (abs(div_naive) + 1e-6) < 0.1:
    print(f"    结论：clamp对散度估计影响<10%，散度分析对输入截断不敏感")
    print(f"    （注：clamp的主要危害在于N2N训练目标E[y2|x]≠x，见add_noise的说明）")
else:
    print(f"    结论：clamp对散度估计影响显著（>10%），散度定量分析需避免输入截断")

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

# 2×2块内四个位置的编号：k = 2*行内偏移 + 列内偏移
#   0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
# 四种模式各把一对相邻像素给sub1、互补的一对给sub2，恰好覆盖整块4个像素
_PAT_SUB1_A = (0, 0, 1, 2)
_PAT_SUB1_B = (1, 2, 3, 3)
_PAT_SUB2_A = (2, 1, 0, 0)
_PAT_SUB2_B = (3, 3, 2, 1)

def neighbor_split(y, pattern=None, generator=None):
    """随机邻域降采样：由单帧噪声图像构造伪配对（Huang et al. 2021）

    对每个2×2块随机选一种相邻像素对模式 p ∈ {0,1,2,3}：
      p=0: sub1=(0,0),(0,1)  sub2=(1,0),(1,1)   （水平：上行 / 下行）
      p=1: sub1=(0,0),(1,0)  sub2=(0,1),(1,1)   （垂直：左列 / 右列）
      p=2: sub1=(0,1),(1,1)  sub2=(0,0),(1,0)   （垂直：右列 / 左列）
      p=3: sub1=(1,0),(1,1)  sub2=(0,0),(0,1)   （水平：下行 / 上行）

    ★为什么是"一对相邻像素×2"而不是"两个相邻的单像素"：
    每个2×2块共4个像素。sub1与sub2各取2个互补像素（无重叠、无遗漏），
    合起来正好覆盖整块，因此两张子图尺寸为 (B, 2C, H//2, W//2)，
    与原论文一致；并且可由 (子图, pattern) **无损还原**原图（见neighbor_merge），
    使下采样版PSNR就是全图口径，可与其他方法直接比较。
    （注：空间独立的噪声在相邻像素间也独立，故互补像素可互为"噪声标签"。）

    ★随机模式的作用：若模式固定（如恒取(0,0)-(0,1)），网络容易学到
    "往右挪一个像素"这种确定性的空间偏移捷径；每个样本/每个batch随机取模式可消除该捷径。

    参考: Huang et al. (2021) Neighbor2Neighbor

    Args:
        y: (B, C, H, W) 噪声图像
        pattern: 可选，(B, H//2, W//2) int64；为None时随机采样
        generator: 可选，torch.Generator，用于随机模式的可复现性

    Returns:
        sub1, sub2: (B, 2C, H//2, W//2)
        pattern:    (B, H//2, W//2)
    """
    B, C, H, W = y.shape
    # 确保 H, W 为偶数
    if H % 2 != 0:
        y = y[:, :, :-1, :]
    if W % 2 != 0:
        y = y[:, :, :, :-1]
    B, C, H, W = y.shape
    H2, W2 = H // 2, W // 2

    # 把每个2×2块的4个像素摊到最后一维：blk[..., k]
    blk = y.reshape(B, C, H2, 2, W2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, H2, W2, 4)

    if pattern is None:
        pattern = torch.randint(0, 4, (B, H2, W2), device=y.device, generator=generator)

    def _take(flat_table):
        table = torch.tensor(flat_table, device=y.device, dtype=torch.long)
        idx = table[pattern].unsqueeze(1).expand(B, C, H2, W2).unsqueeze(-1)
        return torch.gather(blk, 4, idx).squeeze(-1)

    sub1 = torch.cat([_take(_PAT_SUB1_A), _take(_PAT_SUB1_B)], dim=1)  # (B, 2C, H2, W2)
    sub2 = torch.cat([_take(_PAT_SUB2_A), _take(_PAT_SUB2_B)], dim=1)
    return sub1, sub2, pattern

def neighbor_merge(sub1, sub2, pattern):
    """按 pattern 将两张子图还原为原图（每个像素恰好被填一次，无损）

    由于 sub1/sub2 在每个2×2块内互补且无重叠，还原结果不含任何
    插值或复制填充的"伪像素"，下采样版的PSNR因此是全图口径，
    可直接与全分辨率方法、监督基线比较。
    """
    B, C2, H2, W2 = sub1.shape
    C = C2 // 2
    canvas = torch.empty(B, C, H2, W2, 4, device=sub1.device, dtype=sub1.dtype)

    def _put(flat_table, val):
        table = torch.tensor(flat_table, device=sub1.device, dtype=torch.long)
        idx = table[pattern].unsqueeze(1).expand(B, C, H2, W2).unsqueeze(-1)
        canvas.scatter_(4, idx, val.unsqueeze(-1))

    _put(_PAT_SUB1_A, sub1[:, :C])
    _put(_PAT_SUB1_B, sub1[:, C:])
    _put(_PAT_SUB2_A, sub2[:, :C])
    _put(_PAT_SUB2_B, sub2[:, C:])

    return canvas.reshape(B, C, H2, W2, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, H2 * 2, W2 * 2)

# --- 训练Neighbor2Neighbor ---
print("\n  训练Neighbor2Neighbor...")

"""
★ Neighbor2Neighbor 的实现要点（v3起）

1. 分辨率减半：
   - 每个2×2块取2个像素作为子图，图像从 32×32 变为 16×16（2通道）
   - 模型在半分辨率下训练，感受野相对变大
   - 预测结果由 neighbor_merge 按记录的模式**无损还原**回 32×32，
     所以下采样版的PSNR就是全图口径，可直接与监督/N2N比较

2. 与原文一致之处：
   - 每块随机选一对相邻像素（4种模式等概率随机），互补像素对作为配对目标
   - 训练时每batch随机模式，测试时用固定模式（保证可复现）

3. 相对原文的已知简化：
   - 未实现原论文的 reg 项（‖f(sub1)-sub1‖² 之类的额外正则项），
     这会使本实验的N2B结果略低于论文设定的上限，属已知简化而非算法缺陷
   - 未做模型选择/超参搜索

4. 差距拆分对照（见下方Step 4a2）：
   - 额外训练一个"半分辨率监督对照"，其输入表示、架构、优化器、轮数与推理/还原方式
     与N2B降采样**完全相同**，仅把训练目标换成分辨率表示下的干净像素；
   - 用途：把 N2B−N2N 的差距实测拆分为"表示/分辨率代价"与"伪配对目标代价"两部分，
     避免把差距笼统归因于未经检验的原因
"""

model_n2nb = SmallUNet(in_ch=2, out_ch=2).to(device)  # ★子图为2通道（每2×2块取2像素）
optimizer_n2nb = optim.Adam(model_n2nb.parameters(), lr=LR)
n2nb_ckpt_path = os.path.join(SAVE_DIR, f'ckpt_N2B_{N2B_CKPT_VER}.pt')
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
            # 从单帧y构造伪配对（★每个batch随机采样邻域模式）
            sub1, sub2, _ = neighbor_split(y)
            optimizer_n2nb.zero_grad()
            pred1 = model_n2nb(sub1)
            # ★ N2B核心：用sub2（互补位置的独立噪声实现）作为sub1的"噪声标签"
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

# 评估N2B：降采样→去噪→按同一模式无损还原
# ★使用固定噪声（与evaluate_psnr一致）+固定邻域模式，确保评估可复现
model_n2nb.eval()
psnr_n2nb_vals = []
ssim_n2nb_vals = []
n2b_noise_rng = torch.Generator(device=device)
n2b_noise_rng.manual_seed(12345)  # 与evaluate_psnr的noise_seed一致
n2b_pat_rng = torch.Generator(device=device)
n2b_pat_rng.manual_seed(20240)    # 评估用固定邻域模式
with torch.no_grad():
    for batch_x, _ in tqdm(test_loader, desc='评估N2B PSNR', leave=False):
        batch_x = batch_x.to(device)
        noise = torch.randn(batch_x.shape, generator=n2b_noise_rng, device=device)
        y = batch_x + SIGMA * noise
        pattern = torch.randint(0, 4, (y.shape[0], IMG_SIZE // 2, IMG_SIZE // 2),
                                device=device, generator=n2b_pat_rng)
        sub1, sub2, pat = neighbor_split(y, pattern=pattern)
        pred_sub1 = model_n2nb(sub1).clip(0, 1)
        pred_sub2 = model_n2nb(sub2).clip(0, 1)
        # ★无损还原：每个像素恰好被sub1或sub2的预测填充一次，无棋盘伪影
        pred_full = neighbor_merge(pred_sub1, pred_sub2, pat)
        pred_np = pred_full.cpu().numpy()
        x_np = batch_x.cpu().numpy()
        for i in range(min(pred_np.shape[0], x_np.shape[0])):
            h = min(pred_np.shape[2], x_np.shape[2])
            w = min(pred_np.shape[3], x_np.shape[3])
            psnr_n2nb_vals.append(psnr(x_np[i, 0, :h, :w], pred_np[i, 0, :h, :w], data_range=1.0))
            ssim_n2nb_vals.append(ssim(x_np[i, 0, :h, :w], pred_np[i, 0, :h, :w], data_range=1.0))

psnr_n2nb = np.mean(psnr_n2nb_vals)
ssim_n2nb = np.mean(ssim_n2nb_vals)
# ★命名说明：n2nb = N2B降采样版（neighbor_split/merge，全图口径）；n2b_fullres = N2B全分辨率版
print(f"  Neighbor2Neighbor PSNR = {psnr_n2nb:.2f} dB, SSIM = {ssim_n2nb:.4f}")

# --- Step 4a2: 半分辨率监督对照（★用于实际拆分 N2B 与 N2N 的差距来源）---
# ★动机：N2B降采样与N2N的差距里，有两个本来纠缠在一起的因素：
#   (i)  表示/分辨率：N2B在半分辨率(32→16)的2通道子图上训练与推理；
#   (ii) 目标位置错位：N2B的目标 y₂ 取自**相邻位置**，最优解是 E[x₂|y₁] 而非 x₁。
#        （注意：目标"带噪"本身不引入偏差——噪声独立零均值，E[y₂|y₁]=E[x₂|y₁]；
#          偏差全部来自 x₂≠x₁，即相邻像素信号不同，信号相关性只决定偏差有多大。）
# 本对照与N2B降采样**共用完全相同的输入表示、网络架构、优化器、轮数与推理/还原方式**，
# 唯一差别是把训练目标换成同位置的**干净**像素（即同位置的有监督去噪）。于是可实测拆分为：
#   表示/分辨率代价 = PSNR(N2N) − PSNR(本对照)
#   伪配对目标代价   = PSNR(本对照) − PSNR(N2B降采样)
print("\n  训练半分辨率监督对照（同表示、同架构，仅把目标换成干净像素）...")

model_hfsup = SmallUNet(in_ch=2, out_ch=2).to(device)
optimizer_hfsup = optim.Adam(model_hfsup.parameters(), lr=LR)
hfsup_ckpt_path = os.path.join(SAVE_DIR, f'ckpt_HalfResSup_{N2B_CKPT_VER}.pt')
hfsup_start = 0
is_hfsup_final = False
losses_hfsup = []

if os.path.exists(hfsup_ckpt_path):
    checkpoint = torch.load(hfsup_ckpt_path, map_location=device, weights_only=False)
    model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
    optimizer_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer_state'))
    losses_hfsup = checkpoint.get('train_losses', [])
    if checkpoint.get('is_final', False):
        print(f"  [半分辨率监督] ✓ 检测到最终权重，直接加载，跳过训练过程")
        print(f"  训练轮数: {checkpoint['epoch']+1}")
        model_hfsup.load_state_dict(model_state)
        if optimizer_state:
            optimizer_hfsup.load_state_dict(optimizer_state)
        hfsup_start = checkpoint['epoch'] + 1
        is_hfsup_final = True
    else:
        print(f"  [半分辨率监督] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
        model_hfsup.load_state_dict(model_state)
        if optimizer_state:
            optimizer_hfsup.load_state_dict(optimizer_state)
        hfsup_start = checkpoint['epoch'] + 1

if is_hfsup_final or hfsup_start >= N_EPOCHS:
    print("  [半分辨率监督] 模型已训练完毕，跳过。")
else:
    model_hfsup.train()
    for epoch in range(hfsup_start, N_EPOCHS):
        epoch_loss = 0
        n_batch = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{N_EPOCHS} [半分辨率监督]', leave=False, unit='batch')
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            y = add_noise(batch_x, SIGMA)
            # ★对噪声图与干净图施加**同一**邻域模式，确保输入表示与N2B降采样完全一致
            sub1, _, pattern = neighbor_split(y)          # 与N2B训练完全相同的模式采样方式
            clean_sub1, _, _ = neighbor_split(batch_x, pattern=pattern)
            optimizer_hfsup.zero_grad()
            loss = nn.MSELoss()(model_hfsup(sub1), clean_sub1.detach())
            loss.backward()
            optimizer_hfsup.step()
            epoch_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        avg_loss = epoch_loss / n_batch
        losses_hfsup.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  [半分辨率监督] Epoch {epoch+1}/{N_EPOCHS}, Loss: {avg_loss:.6f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_hfsup.state_dict(),
                'optimizer_state_dict': optimizer_hfsup.state_dict(),
                'loss': avg_loss,
                'train_losses': losses_hfsup,
                'is_final': False
            }, hfsup_ckpt_path)
            print(f"  [半分辨率监督] ✓ checkpoint已保存 (epoch {epoch+1})")
    if losses_hfsup:
        torch.save({
            'epoch': N_EPOCHS - 1,
            'model_state_dict': model_hfsup.state_dict(),
            'loss': losses_hfsup[-1],
            'train_losses': losses_hfsup,
            'is_final': True
        }, hfsup_ckpt_path)
        print(f"  [半分辨率监督] ✓ 训练完成, 最终checkpoint已保存")

# 评估半分辨率监督对照：与N2B降采样使用完全相同的评估口径（同噪声、同模式、同还原方式）
model_hfsup.eval()
psnr_hfsup_vals, ssim_hfsup_vals = [], []
hfsup_noise_rng = torch.Generator(device=device); hfsup_noise_rng.manual_seed(12345)
hfsup_pat_rng = torch.Generator(device=device); hfsup_pat_rng.manual_seed(20240)
with torch.no_grad():
    for batch_x, _ in tqdm(test_loader, desc='评估半分辨率监督 PSNR', leave=False):
        batch_x = batch_x.to(device)
        noise = torch.randn(batch_x.shape, generator=hfsup_noise_rng, device=device)
        y = batch_x + SIGMA * noise
        pattern = torch.randint(0, 4, (y.shape[0], IMG_SIZE // 2, IMG_SIZE // 2),
                                device=device, generator=hfsup_pat_rng)
        sub1, sub2, pat = neighbor_split(y, pattern=pattern)
        pred_sub1 = model_hfsup(sub1).clip(0, 1)
        pred_sub2 = model_hfsup(sub2).clip(0, 1)
        pred_full = neighbor_merge(pred_sub1, pred_sub2, pat)
        pred_np = pred_full.cpu().numpy()
        x_np = batch_x.cpu().numpy()
        for i in range(min(pred_np.shape[0], x_np.shape[0])):
            h = min(pred_np.shape[2], x_np.shape[2])
            w = min(pred_np.shape[3], x_np.shape[3])
            psnr_hfsup_vals.append(psnr(x_np[i, 0, :h, :w], pred_np[i, 0, :h, :w], data_range=1.0))
            ssim_hfsup_vals.append(ssim(x_np[i, 0, :h, :w], pred_np[i, 0, :h, :w], data_range=1.0))

psnr_hfsup = np.mean(psnr_hfsup_vals)
ssim_hfsup = np.mean(ssim_hfsup_vals)
print(f"  半分辨率监督对照 PSNR = {psnr_hfsup:.2f} dB, SSIM = {ssim_hfsup:.4f}")
print(f"    → 表示/分辨率代价 = PSNR(N2N) − PSNR(对照) = {psnr_n2n - psnr_hfsup:+.2f} dB")
print(f"    → 伪配对目标代价   = PSNR(对照) − PSNR(N2B降采样) = {psnr_hfsup - psnr_n2nb:+.2f} dB")

# --- Step 4b: 保持分辨率版N2B（★作为"感受野泄漏/identity shortcut"的对照负例）---
print("\n  训练保持分辨率版N2B（该版本有意保留感受野泄漏，作为对照负例）...")

def get_subgrid_indices(H, W, device):
    """获取棋盘格子图位置的索引

    子图1：偶数行偶数列 (i%2==0, j%2==0)
    子图2：偶数行奇数列 (i%2==0, j%2==1)

    返回：
        idx_sub1: 子图1位置的扁平索引
        idx_sub2: 子图2位置的扁平索引

    ★ 与N2B降采样方案的对比：
    ─────────────────────────────────────────
    N2B降采样方案（neighbor_split/merge，v3）：
      - 物理下采样：32×32 → 16×16（2通道）子图
      - 模型在低分辨率上训练，sub2位置不在输入中 → 无感受野泄漏
      - 测试时按记录的模式无损还原，无需近似填充

    本方案（保持分辨率）：
      - 逻辑下采样：仅提取子图位置的索引
      - 模型在全分辨率上训练，输入包含全部像素 → 存在感受野泄漏
      - 损失只在idx_sub1位置计算，目标取相邻的idx_sub2像素
      - 测试时直接推理、无需合并，但泄漏使其退化为identity shortcut
    ─────────────────────────────────────────
    ★教学定位：本方案是"保持分辨率但未屏蔽输入像素"的**负例**，
      用于对照说明真正的blind-spot必须保证目标像素不在感受野内。
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
    1. 训练和测试分辨率一致（都是32×32），无需降采样/还原
    2. 与N2B降采样版形成对照：唯一差异是"输入是否包含目标像素"

    ★★★ 重要理论缺陷（教学要点）★★★

    本实现存在"感受野泄漏"问题，这是blind-spot自监督去噪方法的经典陷阱：

    1. 问题本质：
       - 原始neighbor_split版本：模型输入是16×16的下采样子图，sub2位置的信息
         从未进入模型输入，天然满足blind-spot约束
       - 本版本：模型输入是完整的y，包含了位置i自身的像素值
       - SmallUNet的skip connection会让网络学到恒等映射 f(y)[i] ≈ y[i]
       - 相邻像素y[i]与y[j]（sub2的独立噪声实现）相关性很强，
         用y[i]直接逼近y[j]就能把loss压得很低

    2. 这不代表模型学会了去噪：
       - 网络可能只是学会了"抄输入"（identity shortcut）
       - 这与经典blind-spot网络文献（Laine et al. 2019）要解决的问题一致

    3. 正确的blind-spot实现需要：
       - 输入掩码：对目标位置置零或用掩码卷积（见实验17.4-1的N2V masking）
       - 或使用旋转不变架构等标准blind-spot技巧

    4. 教学价值：
       - 本版本是"未屏蔽输入像素"的**负例**：保留分辨率却引入泄漏，
         结果被吸向恒等映射、PSNR反而不如下采样版
       - N2B降采样版（v3）既无感受野泄漏、也无合并伪影，
         是本实验中更接近原论文的实现

    参考: Laine et al. (2019) "High-Quality Self-Supervised Deep Image Denoising"
    """
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ckpt_path = os.path.join(SAVE_DIR, f'ckpt_{tag}_{CKPT_VER}.pt')
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
        y = batch_x + SIGMA * noise
        pred = model_n2b_fullres(y).clip(0, 1)
        pred_np = pred.cpu().numpy()
        x_np = batch_x.cpu().numpy()
        for i in range(pred_np.shape[0]):
            psnr_n2b_fullres_vals.append(psnr(x_np[i, 0], pred_np[i, 0], data_range=1.0))
            ssim_n2b_fullres_vals.append(ssim(x_np[i, 0], pred_np[i, 0], data_range=1.0))

psnr_n2b_fullres = np.mean(psnr_n2b_fullres_vals)
ssim_n2b_fullres = np.mean(ssim_n2b_fullres_vals)
print(f"  保持分辨率版N2B PSNR = {psnr_n2b_fullres:.2f} dB, SSIM = {ssim_n2b_fullres:.4f}")

# 注（v3）：原"公平评估（仅在idx_sub1位置）"补丁已删除。
# 该补丁是为绕开旧版neighbor_merge的棋盘伪影而设的补偿口径；
# v3的降采样版按记录模式无损还原，PSNR本就是全图口径，
# 与监督/N2N/全分辨率版同口径可比，无需再做非对称补偿。

# --- ★偶数行vs奇数行PSNR对比（量化"卷积空间共享泛化"假设） ---
print(f"\n  ★★ 偶数行vs奇数行PSNR对比（仅针对全分辨率负例）★★")
print("  全分辨率版只在偶数行的idx_sub1位置被监督，奇数行依赖'卷积核空间共享→泛化'。")
print("  量化对比可以验证这个隐含假设是否成立：")
print("  （N2B降采样版v3的4种随机模式在期望意义上覆盖了全部像素，不存在该问题）")
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
        y = batch_x + SIGMA * noise
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
print(f"  全分辨率版的损失是‖f(y)[sub1] - y[sub2]‖²，identity shortcut的表现是")
print(f"  f(y)[sub1] ≈ y[sub2]（直接抄目标位置的相邻像素），因此诊断必须对齐y[sub2]。")
print(f"  ★v2修正：v1误测f(y)[sub1]与y[sub1]（输入自身位置）的相关性，导致泄漏漏检")
model_n2b_fullres.eval()
# ★获取子图索引（与训练时一致）
idx_sub1_diag, idx_sub2_diag = get_subgrid_indices(IMG_SIZE, IMG_SIZE, device)

# ★计算f(y)[sub1]与训练目标y[sub2]的相关系数和MSE
def compute_target_correlation(model, y, idx_sub1, idx_sub2):
    """计算f(y)[sub1]与训练目标y[sub2]的相关系数与MSE

    ★为什么必须对齐训练目标：
    - 全分辨率版N2B用y[sub2]作target，若发生identity shortcut，
      f(y)[sub1]会逼近y[sub2]而非y[sub1]
    - 判据：corr→1 且 MSE≪σ²（纯复制时MSE→0；真去噪模型输出≈E[x|y]，
      与单次噪声实现y[sub2]的固有差异为σ²，故基线MSE≈σ²）
    """
    model.eval()
    with torch.no_grad():
        pred = model(y)
        B, C, H, W = pred.shape
        pred_flat = pred.view(B, C, -1)
        y_flat = y.view(B, C, -1)
        pred_sub1 = pred_flat[:, :, idx_sub1].flatten()
        y_target = y_flat[:, :, idx_sub2].flatten()
        correlation = np.corrcoef(y_target.cpu().numpy(), pred_sub1.cpu().numpy())[0, 1]
        mse_to_target = ((pred_sub1 - y_target) ** 2).mean().item()
        return correlation, mse_to_target

test_diag, _ = next(iter(test_loader))
test_diag_y = add_noise(test_diag[:64].to(device), SIGMA)

# ★对照组：model_sup/model_n2n不以y[sub2]为target，代表"正常去噪模型"的基线
correlation, mse_to_target = compute_target_correlation(model_n2b_fullres, test_diag_y, idx_sub1_diag, idx_sub2_diag)
corr_sup, mse_sup_target = compute_target_correlation(model_sup, test_diag_y, idx_sub1_diag, idx_sub2_diag)
corr_n2n, mse_n2n_target = compute_target_correlation(model_n2n, test_diag_y, idx_sub1_diag, idx_sub2_diag)

print(f"    f(y)[sub1]与训练目标y[sub2]的相关系数:")
print(f"      model_sup (正常基线):       {corr_sup:.4f} (MSE={mse_sup_target:.4f})")
print(f"      model_n2n (正常基线):        {corr_n2n:.4f} (MSE={mse_n2n_target:.4f})")
print(f"      model_n2b_fullres (待诊断): {correlation:.4f} (MSE={mse_to_target:.6f})")
print(f"    ★判据参考: σ²={SIGMA**2:.4f}；若待诊断模型MSE≪σ²而基线MSE≈σ²，")
print(f"      说明其输出几乎逐像素复制了y[sub2]，而非真正去噪")
# ★动态校验：使用相对阈值（待诊断-正常基线）
# 注意：0.1和0.05是教学经验值，非统计显著性检验阈值
# 实际判断应结合具体场景和领域知识
baseline = (corr_sup + corr_n2n) / 2  # 正常基线均值
excess = correlation - baseline
print(f"    正常基线均值: {baseline:.4f}, 待诊断模型超出量: {excess:+.4f}")
if excess > 0.1:  # 经验阈值，仅供参考
    print(f"    ⚠️  警告：超出正常基线{excess:.3f}，存在identity shortcut（抄目标像素）风险")
elif excess > 0.05:  # 经验阈值，仅供参考
    print(f"    相关系数略高于正常基线，需关注但风险有限")
else:
    print(f"    相关系数与正常基线相当，identity shortcut风险较低")
print(f"  （注：理想的blind-spot实现应确保目标像素不在感受野内）")

print(f"\n  ★ N2B两种实现对比：")
print(f"    下采样+合并版：PSNR = {psnr_n2nb:.2f} dB, SSIM = {ssim_n2nb:.4f}")
print(f"    保持分辨率版：  PSNR = {psnr_n2b_fullres:.2f} dB, SSIM = {ssim_n2b_fullres:.4f}")
if psnr_n2b_fullres > psnr_n2nb:
    print(f"    保持分辨率版更高：+{psnr_n2b_fullres - psnr_n2nb:.2f} dB（需结合下方泄漏诊断判断可信度）")
else:
    print(f"    下采样版更高：    {psnr_n2b_fullres - psnr_n2nb:+.2f} dB（保持分辨率版可能因identity shortcut退化）")

# --- 可视化六种方法对比 ---
vis_imgs, _ = next(test_iter)  # ★使用共享test_iter，与前两步不同batch
vis_imgs = vis_imgs[:3].to(device)
vis_noisy = add_noise(vis_imgs, SIGMA)

with torch.no_grad():
    pred_sup_vis = model_sup(vis_noisy).cpu().clip(0, 1)
    pred_n2n_vis = model_n2n(vis_noisy).cpu().clip(0, 1)
    pred_naive_vis = model_naive(vis_noisy).cpu().clip(0, 1)
    # N2B 降采样+无损还原版（可视化用固定邻域模式，与评估口径一致）
    vis_pat = torch.randint(0, 4, (vis_noisy.shape[0], IMG_SIZE // 2, IMG_SIZE // 2),
                            device=device, generator=torch.Generator(device=device).manual_seed(20240))
    sub1_v, sub2_v, pat_v = neighbor_split(vis_noisy, pattern=vis_pat)
    # ★变量名更具体：pred_s1_n2b/pred_s2_n2b 避免与Step 1的 pred_s1/pred_s2 混淆
    pred_s1_n2b = model_n2nb(sub1_v).clip(0, 1)
    pred_s2_n2b = model_n2nb(sub2_v).clip(0, 1)
    pred_n2b_vis = neighbor_merge(pred_s1_n2b, pred_s2_n2b, pat_v).cpu()
    # 半分辨率监督对照（与N2B降采样同口径还原，便于目视对比）
    pred_hfsup_vis = neighbor_merge(model_hfsup(sub1_v).clip(0, 1),
                                    model_hfsup(sub2_v).clip(0, 1), pat_v).cpu()
    # N2B 保持分辨率版
    pred_n2b_fullres_vis = model_n2b_fullres(vis_noisy).cpu().clip(0, 1)

fig, axes = plt.subplots(8, 3, figsize=(10, 24))
row_labels = ['干净图像x', '噪声输入y',
              f'监督 ({psnr_sup:.1f}dB, SSIM={ssim_sup:.3f})',
              f'N2N ({psnr_n2n:.1f}dB, SSIM={ssim_n2n:.3f})',
              f'朴素 ({psnr_naive:.1f}dB, SSIM={ssim_naive:.3f})',
              f'N2B降采样 ({psnr_n2nb:.1f}dB, SSIM={ssim_n2nb:.3f})',
              f'半分辨率监督对照 ({psnr_hfsup:.1f}dB, SSIM={ssim_hfsup:.3f})',
              f'N2B全分辨率负例 ({psnr_n2b_fullres:.1f}dB, SSIM={ssim_n2b_fullres:.3f})']

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
    # 半分辨率监督对照
    axes[6, i].imshow(pred_hfsup_vis[i, 0], cmap='gray', vmin=0, vmax=1)
    axes[6, i].axis('off')
    # N2B全分辨率版
    axes[7, i].imshow(pred_n2b_fullres_vis[i, 0], cmap='gray', vmin=0, vmax=1)
    axes[7, i].axis('off')

for r, label in enumerate(row_labels):
    axes[r, 0].set_ylabel(label, fontsize=10, rotation=0, labelpad=80)

fig.suptitle('Step 4: 六种去噪方法完整对比（含干净/噪声参考）', fontsize=13)
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
print(f"  N2B  降采样(v3)        {psnr_n2nb:6.2f}       {ssim_n2nb:6.4f}    单帧伪配对，全图口径")
print(f"  半分辨率监督对照        {psnr_hfsup:6.2f}       {ssim_hfsup:6.4f}    同表示/同架构，仅目标换干净像素")
print(f"  N2B  保持分辨率(负例)  {psnr_n2b_fullres:6.2f}       {ssim_n2b_fullres:6.4f}    感受野泄漏→identity shortcut")
print(f"  ─────────────────────────────────────────────────────────")
print(f"\n  ★注（v3）：以上六种方法的PSNR均为**全图口径**，可直接互相比较。")
print(f"     '保持分辨率(负例)'一行是为暴露identity shortcut而刻意保留的对照实现，")
print(f"     其低分来自泄漏（网络抄输入而非去噪），不代表N2B的算法上限。")
print(f"     '半分辨率监督对照'与'N2B降采样'共用同一输入表示、架构、轮数与推理方式，")
print(f"     仅训练目标不同（干净像素 vs 互补位置的带噪像素），用于实测拆分N2B−N2N的差距。")
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

# ★N2B与N2N的差距归因（v4）：用"半分辨率监督对照"把差距**实测拆分**，不再停留于定性判断
n2b_gap = psnr_n2n - psnr_n2nb
gap_repr = psnr_n2n - psnr_hfsup      # 表示/分辨率代价
gap_target = psnr_hfsup - psnr_n2nb   # 伪配对目标代价
if n2b_gap < 2.0:
    print(f"  3. N2B ≈ N2N (低{n2b_gap:.2f}dB < 2dB) → 单帧噪声下用相邻互补像素构造伪配对有效")
else:
    print(f"  3. N2B 比 N2N 低 {n2b_gap:.2f}dB，由'半分辨率监督对照'实测拆分：")
    print(f"     · 表示/分辨率代价 PSNR(N2N)−PSNR(对照) = {gap_repr:+.2f} dB")
    print(f"       （半分辨率2通道子图 vs 全分辨率，属结构性代价）")
    print(f"     · 伪配对目标代价   PSNR(对照)−PSNR(N2B) = {gap_target:+.2f} dB")
    print(f"       （目标用互补位置的带噪像素，而非同位置的干净像素）")
    if gap_repr > gap_target + 0.5:
        main_cause = "半分辨率表示(i)"
    elif gap_target > gap_repr + 0.5:
        main_cause = "伪配对目标位置错位(ii)：目标取相邻像素而非同位置干净像素"
    else:
        main_cause = "(i)与(ii)相当"
    print(f"     → 主导因素：{main_cause}")
    print(f'     注：该拆分是"先表示、后目标"的单路径可加分解，两项之和恒等于总差距，')
    print(f"         但未分离二者的交互项（例如分辨率代价在目标带噪时可能略有不同）。")
    print(f"     注（伪配对目标为何有偏）：噪声独立零均值只给出 E[y₂|y₁]=E[x₂|y₁]，")
    print(f"         而该位置本应给出 x₁，且 E[x₂|y₁]≠x₁——所以伪配对目标对 x **有偏**。")
    print(f"         相邻像素的信号相关性正是该偏差的控制因素（线性近似下 E[x₂|y₁] 相当于")
    print(f"         把 E[x₁|y₁] 按相关系数ρ收缩）：ρ↑→偏差↓；ρ→0 时网络无法从 y₁ 推断 x₂，")
    print(f"         只能输出局部统计均值。（本实验数据实测相邻ρ≈0.87，偏差受控，N2B 才可用；")
    print(f"         这也是它仍不及同位置 N2N 的根因。）")

print(f"\n  ★ N2B两种实现对比（简表）：")
print(f"  ┌───────────────┬──────────────────────┬──────────────────────┐")
print(f"  │               │  降采样版(v3)         │  保持分辨率版(负例)   │")
print(f"  ├───────────────┼──────────────────────┼──────────────────────┤")
print(f"  │ 盲点约束      │  ✓ 天然满足           │  ✗ 存在感受野泄漏    │")
print(f"  │               │  (sub2不在输入中)     │  (输入包含完整y)     │")
print(f"  ├───────────────┼──────────────────────┼──────────────────────┤")
print(f"  │ 分辨率        │  减半(32→16，2通道)   │  ✓ 保持(32×32)      │")
print(f"  ├───────────────┼──────────────────────┼──────────────────────┤")
print(f"  │ 还原方式      │  ✓ 按模式无损还原     │  无需还原步骤        │")
print(f"  │               │  (无插值/无伪影)      │                      │")
print(f"  ├───────────────┼──────────────────────┼──────────────────────┤")
print(f"  │ identity风险  │  ✓ 无                │  ✗ 实测f(y)≈y       │")
print(f"  ├───────────────┼──────────────────────┼──────────────────────┤")
print(f"  │ PSNR口径      │  ✓ 全图，可比较       │  被泄漏扭曲，仅作警示")
print(f"  └───────────────┴──────────────────────┴──────────────────────┘")
print(f"  性能差异：{psnr_n2b_fullres - psnr_n2nb:+.2f} dB（保持分辨率版 − 降采样版）")
# ★动态校验：根据诊断结果给出风险评估（对齐训练目标y[sub2]的口径）
if correlation > 0.9 and mse_to_target < SIGMA**2 / 10:
    print(f"  ★诊断：f(y)[sub1]与目标y[sub2]的相关系数{correlation:.2f}>0.9，")
    print(f"  且MSE({mse_to_target:.2e})≪σ²({SIGMA**2:.4f})——输出几乎逐像素复制目标位置，")
    print(f"  identity shortcut确凿，保持分辨率版的PSNR反映的是'抄像素'而非去噪能力")
elif correlation > 0.9:
    print(f"  ★诊断：相关系数{correlation:.2f}>0.9，identity shortcut风险明显")
    print(f"  建议：以N2B降采样版(v3)的结果为准；全分辨率版分数被泄漏扭曲，不可比")
else:
    print(f"  ★诊断：相关系数{correlation:.2f}≤0.9，identity shortcut风险较低（但仍需关注理论缺陷）")
print(f"  ★结论：降采样版(v3)是本实验的正确实现；保持分辨率版是刻意保留的泄漏负例，")
print(f"     其低分不构成N2B的算法上限")
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
print(f"     - 可尝试增加训练轮数(N_EPOCHS)观察不同方法的最终收敛性能")
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

# ===== 保存数值结果 =====
import json

def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
    import numpy as np
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
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return _to_native(obj.detach().cpu().tolist())
    except:
        pass
    return obj

# 收集关键数值结果
results_summary = {
    '实验': '17.2-1 Noise2Noise与Noise2Blind',
    '不同训练设定的性能': {
        '设定1(监督)': {'PSNR': psnr_s1, 'SSIM': ssim_s1},
        '设定4子档(配对噪声观测)': {'PSNR': psnr_s2, 'SSIM': ssim_s2},
        '设定4(朴素自监督)': {'PSNR': psnr_s4, 'SSIM': ssim_s4},
    },
    '完整训练结果': {
        '监督': {'PSNR': psnr_sup, 'SSIM': ssim_sup},
        'Noise2Noise': {'PSNR': psnr_n2n, 'SSIM': ssim_n2n},
        '朴素自监督': {'PSNR': psnr_naive, 'SSIM': ssim_naive},
        'N2B降采样(全图口径)': {'PSNR': psnr_n2nb, 'SSIM': ssim_n2nb},
        '半分辨率监督对照(同表示同架构)': {'PSNR': psnr_hfsup, 'SSIM': ssim_hfsup},
        'N2B保持分辨率(泄漏负例)': {'PSNR': psnr_n2b_fullres, 'SSIM': ssim_n2b_fullres},
    },
    'N2B−N2N差距拆分': {
        '总差距(dB)': psnr_n2n - psnr_n2nb,
        '表示_分辨率代价(dB)': psnr_n2n - psnr_hfsup,
        '伪配对目标代价(dB)': psnr_hfsup - psnr_n2nb,
        '说明': '由"半分辨率监督对照"实测拆分：对照与N2B降采样共用输入表示/架构/轮数/推理方式，仅训练目标不同',
    },
    '偏差分析': {
        '朴素散度': div_naive,
        '监督散度': div_sup,
        '偏差项': bias_term,
    },
    'N2B泄漏诊断(保持分辨率负例)': {
        '说明': '★以下两项均由 model_n2b_fullres（泄漏负例）计算，不是降采样N2B；'
                'corr≈1 且 MSE≪σ² 正是它在抄目标像素 y[sub2] 的证据，不是“盲点性好”的证据；'
                '正常去噪模型对单帧噪声目标的 MSE 基线应≈σ²',
        '相关系数(f[sub1] vs 目标y[sub2])': correlation,
        'MSE(f[sub1] vs 目标y[sub2])': mse_to_target,
        'σ^2(基线参考)': SIGMA ** 2,
    },
    '实验参数': {
        '噪声水平': SIGMA,
        '训练轮数快速验证': QUICK_EPOCHS,
        '训练轮数完整训练': N_EPOCHS,
    }
}

results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"\n数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
