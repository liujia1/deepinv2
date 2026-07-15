# -*- coding: utf-8 -*-
"""
实验17.6-1 等变成像与测量一致性
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

素材来源：deepinv.loss.EILoss思路、17.5节理论、17.6节MC损失
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
    SAVE_DIR = os.path.join(_gdrive, '实验17.6-1')
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

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

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
        self.dec3 = DoubleConv(base*2 + base*2, base*2)
        self.dec2 = DoubleConv(base + base, base)
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
    def __init__(self, mask, device=None):
        if device is not None:
            self.mask = mask.to(device)
        else:
            self.mask = mask
        self.device = self.mask.device

    def A(self, x):
        if self.mask.device != x.device:
            mask_2d = self.mask.unsqueeze(0).unsqueeze(0).to(x.device)
        else:
            mask_2d = self.mask.unsqueeze(0).unsqueeze(0)
        return x * mask_2d

    def AT(self, y):
        return self.A(y)

    def zero_filled(self, y):
        return y


def create_inpainting_mask(H, W, keep_ratio=0.5, seed=42):
    rng = np.random.RandomState(seed)
    mask = np.zeros((H, W), dtype=np.float32)
    n_keep = int(H * W * keep_ratio)
    indices = rng.choice(H * W, n_keep, replace=False)
    mask.flat[indices] = 1.0
    return torch.from_numpy(mask)


def create_random_mask_batch(batch_size, H, W, keep_ratio=0.5, device=None):
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
SIGMA = 0.05
KEEP_RATIO = 0.5
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

num_workers = 0 if sys.platform == 'win32' else 2
train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

test_mask = create_inpainting_mask(IMG_SIZE, IMG_SIZE, KEEP_RATIO).to(device)
inpainting_op = InpaintingOperator(test_mask, device=device)


# ========================================================================
# Step 1: 零空间问题——inpainting中自监督损失不约束缺失区域
# ========================================================================
print("\n" + "="*70)
print("Step 1: 零空间问题——自监督损失不约束缺失区域")
print("="*70)

print("\n  训练朴素自监督 (仅MC损失)...")
model_naive = SmallUNet().to(device)
optimizer_naive = optim.Adam(model_naive.parameters(), lr=LR)
naive_ckpt_path = os.path.join(SAVE_DIR, 'ckpt_Naive.pt')
naive_start = 0
naive_train_losses = []
naive_is_final = False

if os.path.exists(naive_ckpt_path):
    checkpoint = torch.load(naive_ckpt_path, map_location=device, weights_only=False)
    model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
    optimizer_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer_state'))

    if checkpoint.get('is_final', False):
        print(f"✓ [Naive] 检测到最终权重，直接加载，跳过训练过程")
        print(f"  训练轮数: {checkpoint['epoch']+1}")
        model_naive.load_state_dict(model_state)
        if optimizer_state is not None:
            optimizer_naive.load_state_dict(optimizer_state)
        naive_train_losses = checkpoint.get('train_losses', [])
        naive_start = checkpoint['epoch'] + 1
        naive_is_final = True
    else:
        print(f"  [Naive] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
        model_naive.load_state_dict(model_state)
        if optimizer_state is not None:
            optimizer_naive.load_state_dict(optimizer_state)
        naive_train_losses = checkpoint.get('train_losses', [])
        naive_start = checkpoint['epoch'] + 1

if naive_is_final:
    print("  [Naive] 模型已训练完毕，跳过。")
else:
    for epoch in range(naive_start, N_EPOCHS):
        model_naive.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{N_EPOCHS}', leave=False, unit='batch')
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            masks = create_random_mask_batch(batch_x.shape[0], IMG_SIZE, IMG_SIZE, KEEP_RATIO, device=device)
            y = batch_x * masks + SIGMA * torch.randn_like(batch_x) * masks

            optimizer_naive.zero_grad()
            f_y = model_naive(y)
            loss = ((masks * (y - f_y)) ** 2).sum() / masks.sum()
            loss.backward()
            optimizer_naive.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = epoch_loss / max(n_batches, 1)
        naive_train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_EPOCHS}, avg_loss={avg_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_naive.state_dict(),
                'optimizer_state_dict': optimizer_naive.state_dict(),
                'loss': avg_loss,
                'train_losses': naive_train_losses,
                'is_final': False
            }, naive_ckpt_path)
            print(f"  [Naive] ✓ checkpoint已保存 (epoch {epoch+1})")

    # 保存最终权重
    torch.save({
        'epoch': N_EPOCHS - 1,
        'model_state_dict': model_naive.state_dict(),
        'loss': naive_train_losses[-1],
        'train_losses': naive_train_losses,
        'is_final': True
    }, naive_ckpt_path)
    print(f"  [Naive] ✓ 最终模型已保存")


# 评估
def evaluate_inpainting(model, test_loader, mask, sigma=SIGMA, device=None):
    model.eval()
    psnr_vals = []
    if device is not None:
        mask_dev = mask.to(device)
    else:
        mask_dev = mask
    with torch.no_grad():
        torch.manual_seed(0)
        for batch_x, _ in tqdm(test_loader, desc='评估PSNR', leave=False):
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
mask_2d = test_mask.unsqueeze(0).unsqueeze(0).expand_as(test_imgs)
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

axes[0, 0].set_ylabel('干净图像 $x$', fontsize=11)
axes[1, 0].set_ylabel(r'观测 $y=M \odot x$', fontsize=11)
axes[2, 0].set_ylabel('朴素MC重建', fontsize=11)
fig.suptitle(f'Step 1: 零空间问题——仅MC损失无法约束缺失区域 (PSNR={psnr_naive:.1f}dB)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step1_nullspace.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step1_nullspace.png")
print("  问题: 缺失像素区域完全不受约束，网络可以输出任意值")


# ========================================================================
# Step 2: 等变成像（EI）损失——利用平移对称性约束零空间
# ========================================================================
print("\n" + "="*70)
print("Step 2: 等变成像（EI）损失——平移对称性约束零空间")
print("="*70)

def random_shift(x, max_shift=8):
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

    ★ stop-gradient：对 x_hat 使用 .detach()，将 T_g x̂ 视为固定目标
    这是 EI 论文 (Chen et al., ICCV 2021) 中的标准做法——防止双侧梯度
    导致 trivial collapse（x̂ 退化为平凡解来最小化 EI 损失）。
    """
    x_hat = model(y)

    total_loss = 0
    for _ in range(n_transforms):
        x_hat_shifted = random_shift(x_hat.detach())

        B, C, H, W = x_hat_shifted.shape
        virtual_masks = create_random_mask_batch(B, H, W, keep_ratio, device=x_hat_shifted.device)
        y_virtual = x_hat_shifted * virtual_masks

        f_virtual = model(y_virtual)
        total_loss += nn.MSELoss()(f_virtual, x_hat_shifted)

    return total_loss / n_transforms


# 训练EI模型
print("\n  训练EI模型 (MC + EI)...")
model_ei = SmallUNet().to(device)
optimizer_ei = optim.Adam(model_ei.parameters(), lr=LR)
lambda_ei = 0.5
ei_ckpt_path = os.path.join(SAVE_DIR, 'ckpt_EI.pt')
ei_start = 0
ei_train_losses = []
ei_is_final = False

if os.path.exists(ei_ckpt_path):
    checkpoint = torch.load(ei_ckpt_path, map_location=device, weights_only=False)
    model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
    optimizer_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer_state'))

    if checkpoint.get('is_final', False):
        print(f"✓ [EI] 检测到最终权重，直接加载，跳过训练过程")
        print(f"  训练轮数: {checkpoint['epoch']+1}")
        model_ei.load_state_dict(model_state)
        if optimizer_state is not None:
            optimizer_ei.load_state_dict(optimizer_state)
        ei_train_losses = checkpoint.get('train_losses', [])
        ei_start = checkpoint['epoch'] + 1
        ei_is_final = True
    else:
        print(f"  [EI] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
        model_ei.load_state_dict(model_state)
        if optimizer_state is not None:
            optimizer_ei.load_state_dict(optimizer_state)
        ei_train_losses = checkpoint.get('train_losses', [])
        ei_start = checkpoint['epoch'] + 1

if ei_is_final:
    print("  [EI] 模型已训练完毕，跳过。")
else:
    for epoch in range(ei_start, N_EPOCHS):
        model_ei.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{N_EPOCHS}', leave=False, unit='batch')
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            masks = create_random_mask_batch(batch_x.shape[0], IMG_SIZE, IMG_SIZE, KEEP_RATIO, device=device)
            y = batch_x * masks + SIGMA * torch.randn_like(batch_x) * masks

            optimizer_ei.zero_grad()

            f_y = model_ei(y)
            loss_mc = ((masks * (y - f_y)) ** 2).sum() / masks.sum()

            loss_ei = ei_loss(model_ei, y, n_transforms=4)

            loss = loss_mc + lambda_ei * loss_ei
            loss.backward()
            optimizer_ei.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = epoch_loss / max(n_batches, 1)
        ei_train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_EPOCHS}, avg_loss={avg_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_ei.state_dict(),
                'optimizer_state_dict': optimizer_ei.state_dict(),
                'loss': avg_loss,
                'train_losses': ei_train_losses,
                'is_final': False
            }, ei_ckpt_path)
            print(f"  [EI] ✓ checkpoint已保存 (epoch {epoch+1})")

    # 保存最终权重
    torch.save({
        'epoch': N_EPOCHS - 1,
        'model_state_dict': model_ei.state_dict(),
        'loss': ei_train_losses[-1],
        'train_losses': ei_train_losses,
        'is_final': True
    }, ei_ckpt_path)
    print(f"  [EI] ✓ 最终模型已保存")

psnr_ei = evaluate_inpainting(model_ei, test_loader, test_mask)
print(f"  EI (MC+EI) PSNR = {psnr_ei:.2f} dB")


# ========================================================================
# Step 3: MC + EI互补性验证
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
sup_train_losses = []
sup_is_final = False

if os.path.exists(sup_ckpt_path):
    checkpoint = torch.load(sup_ckpt_path, map_location=device, weights_only=False)
    model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
    optimizer_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer_state'))

    if checkpoint.get('is_final', False):
        print(f"✓ [Supervised] 检测到最终权重，直接加载，跳过训练过程")
        print(f"  训练轮数: {checkpoint['epoch']+1}")
        model_sup.load_state_dict(model_state)
        if optimizer_state is not None:
            optimizer_sup.load_state_dict(optimizer_state)
        sup_train_losses = checkpoint.get('train_losses', [])
        sup_start = checkpoint['epoch'] + 1
        sup_is_final = True
    else:
        print(f"  [Supervised] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
        model_sup.load_state_dict(model_state)
        if optimizer_state is not None:
            optimizer_sup.load_state_dict(optimizer_state)
        sup_train_losses = checkpoint.get('train_losses', [])
        sup_start = checkpoint['epoch'] + 1

if sup_is_final:
    print("  [Supervised] 模型已训练完毕，跳过。")
else:
    for epoch in range(sup_start, N_EPOCHS):
        model_sup.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{N_EPOCHS}', leave=False, unit='batch')
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            masks = create_random_mask_batch(batch_x.shape[0], IMG_SIZE, IMG_SIZE, KEEP_RATIO, device=device)
            y = batch_x * masks + SIGMA * torch.randn_like(batch_x) * masks

            optimizer_sup.zero_grad()
            f_y = model_sup(y)
            loss = nn.MSELoss()(f_y, batch_x)
            loss.backward()
            optimizer_sup.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = epoch_loss / max(n_batches, 1)
        sup_train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_EPOCHS}, avg_loss={avg_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_sup.state_dict(),
                'optimizer_state_dict': optimizer_sup.state_dict(),
                'loss': avg_loss,
                'train_losses': sup_train_losses,
                'is_final': False
            }, sup_ckpt_path)
            print(f"  [Supervised] ✓ checkpoint已保存 (epoch {epoch+1})")

    # 保存最终权重
    torch.save({
        'epoch': N_EPOCHS - 1,
        'model_state_dict': model_sup.state_dict(),
        'loss': sup_train_losses[-1],
        'train_losses': sup_train_losses,
        'is_final': True
    }, sup_ckpt_path)
    print(f"  [Supervised] ✓ 最终模型已保存")

psnr_sup = evaluate_inpainting(model_sup, test_loader, test_mask)
print(f"  监督 PSNR = {psnr_sup:.2f} dB")


# 各方法在观测/缺失像素的PSNR
NAIVE_KEY = '朴素MC'
methods = {
    '监督': model_sup,
    'MC+EI': model_ei,
    NAIVE_KEY: model_naive,
}

def evaluate_combined(model, test_loader, mask, sigma=SIGMA, device=None):
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
        for batch_x, _ in tqdm(test_loader, desc='评估PSNR', leave=False):
            batch_x = batch_x.to(mask_dev.device)
            mask_2d = mask_dev.unsqueeze(0).unsqueeze(0).expand_as(batch_x)
            y = batch_x * mask_2d + sigma * torch.randn_like(batch_x) * mask_2d
            pred = model(y).clip(0, 1)

            pred_np = pred.cpu().numpy()
            x_np = batch_x.cpu().numpy()
            m_np = mask_2d.cpu().numpy()

            for i in range(batch_x.shape[0]):
                total_psnr_vals.append(psnr(x_np[i, 0], pred_np[i, 0], data_range=1.0))

                obs_pixels = m_np[i, 0] > 0.5
                if obs_pixels.sum() > 0:
                    mse_obs = ((x_np[i, 0][obs_pixels] - pred_np[i, 0][obs_pixels])**2).mean()
                    obs_psnr_vals.append(10 * np.log10(1.0 / max(mse_obs, 1e-10)))

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

fig.suptitle(r'Step 3: MC + EI互补——MC约束值空间，EI约束零空间', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step3_mc_ei_complement.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step3_mc_ei_complement.png")

# 重建结果可视化
fig, axes = plt.subplots(5, 6, figsize=(15, 12))
vis_imgs, _ = next(iter(test_loader))
vis_imgs = vis_imgs[:6].to(device)
mask_vis = test_mask.unsqueeze(0).unsqueeze(0).expand_as(vis_imgs)
y_vis = vis_imgs * mask_vis

with torch.no_grad():
    pred_sup = model_sup(y_vis).clip(0, 1)
    pred_ei = model_ei(y_vis).clip(0, 1)
    pred_naive = model_naive(y_vis).clip(0, 1)

row_data = [
    ('干净图像 $x$', vis_imgs.cpu()),
    (r'观测 $y=M \odot x$', y_vis.cpu()),
    (f'监督 ({total_psnrs["监督"]:.1f}dB)', pred_sup.cpu()),
    (f'MC+EI ({total_psnrs["MC+EI"]:.1f}dB)', pred_ei.cpu()),
]
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
# ========================================================================
print("\n" + "="*70)
print("Step 4: 算子-等变性对照实验")
print("="*70)

def check_equivariance_shift(A_fn, x, n_tests=5):
    errors = []
    Ax_norms = []
    for _ in range(n_tests):
        max_shift = 5
        dy = torch.randint(-max_shift, max_shift+1, (1,)).item()
        dx = torch.randint(-max_shift, max_shift+1, (1,)).item()
        Tg = lambda x: torch.roll(x, shifts=(dy, dx), dims=(2, 3))

        ATg_x = A_fn(Tg(x))
        Ax = A_fn(x)
        TgAx = Tg(Ax)

        err = ((ATg_x - TgAx) ** 2).mean().sqrt().item()
        errors.append(err)
        Ax_norms.append(Ax.abs().mean().item())
    abs_err = np.mean(errors)
    rel_err = abs_err / (np.mean(Ax_norms) + 1e-8)
    return abs_err, rel_err

def check_equivariance_rotate(A_fn, x, angle=90):
    Tg = lambda x: torch.rot90(x, k=angle//90, dims=[2, 3])

    ATg_x = A_fn(Tg(x))
    Ax = A_fn(x)
    TgAx = Tg(Ax)

    abs_err = ((ATg_x - TgAx) ** 2).mean().sqrt().item()
    rel_err = abs_err / (Ax.abs().mean().item() + 1e-8)
    return abs_err, rel_err

# 测试图像
test_x = test_imgs[:4].to(device)

# 1. Inpainting掩码 + 平移
def inpainting_A(x):
    if test_mask.device != x.device:
        mask_2d = test_mask.unsqueeze(0).unsqueeze(0).expand_as(x).to(x.device)
    else:
        mask_2d = test_mask.unsqueeze(0).unsqueeze(0).expand_as(x)
    return x * mask_2d

# 2. MRI欠采样 + 平移
def mri_A(x, sampling_mode='vertical'):
    kspace = torch.fft.fft2(x)
    H, W = x.shape[2], x.shape[3]

    if sampling_mode == 'vertical':
        center = H // 4
        mask = torch.zeros(H, device=x.device)
        mask[:center] = 1.0
        mask[-center:] = 1.0

    elif sampling_mode == 'random':
        state = torch.get_rng_state()
        torch.manual_seed(42)
        mask = torch.rand(H, device=x.device) < 0.25
        torch.set_rng_state(state)

    elif sampling_mode == 'cartesian':
        mask = torch.zeros(H, device=x.device)
        mask[::4] = 1.0

    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    mask_2d = mask.view(1, 1, H, 1).expand_as(kspace)
    return torch.real(torch.fft.ifft2(kspace * mask_2d))

# 3. 高斯模糊 + 平移
_kernel_size = 7
_sigma_k = 1.5
_blur_kernel = torch.zeros(1, 1, _kernel_size, _kernel_size, device=device)
for _i in range(_kernel_size):
    for _j in range(_kernel_size):
        _blur_kernel[0, 0, _i, _j] = np.exp(-((_i-_kernel_size//2)**2 + (_j-_kernel_size//2)**2) / (2*_sigma_k**2))
_blur_kernel = _blur_kernel / _blur_kernel.sum()

def blur_A(x):
    return torch.nn.functional.conv2d(x, _blur_kernel, padding=_kernel_size//2)

# 运行等变性检查
results = {}
rel_results = {}

print(f"\n  算子-等变性验证 ($AT_g$ vs $T_g A$):")
print(f"  {'算子':20s} {'变换':10s} {'绝对误差':10s} {'相对误差':10s}")
print(f"  {'─'*55}")

abs_err, rel_err = check_equivariance_shift(blur_A, test_x)
results[('高斯模糊', '平移')] = abs_err
rel_results[('高斯模糊', '平移')] = rel_err
print(f"  {'高斯模糊':20s} {'平移':10s} {abs_err:.4f}       {rel_err*100:.1f}%")

abs_err, rel_err = check_equivariance_shift(inpainting_A, test_x)
results[('Inpainting', '平移')] = abs_err
rel_results[('Inpainting', '平移')] = rel_err
print(f"  {'Inpainting(随机)':20s} {'平移':10s} {abs_err:.4f}       {rel_err*100:.1f}%")

sampling_modes = ['vertical', 'random', 'cartesian']
for mode in sampling_modes:
    abs_err, rel_err = check_equivariance_shift(
        lambda x: mri_A(x, sampling_mode=mode), test_x)
    results[(f'MRI欠采样({mode})', '平移')] = abs_err
    rel_results[(f'MRI欠采样({mode})', '平移')] = rel_err
    print(f"  {'MRI欠采样('+mode+')':20s} {'平移':10s} {abs_err:.4f}       {rel_err*100:.1f}%")

abs_err, rel_err = check_equivariance_rotate(blur_A, test_x, angle=90)
results[('高斯模糊', '旋转')] = abs_err
rel_results[('高斯模糊', '旋转')] = rel_err
print(f"  {'高斯模糊':20s} {'旋转90°':10s} {abs_err:.4f}       {rel_err*100:.1f}%")

abs_err, rel_err = check_equivariance_rotate(inpainting_A, test_x, angle=90)
results[('Inpainting', '旋转')] = abs_err
rel_results[('Inpainting', '旋转')] = rel_err
print(f"  {'Inpainting(随机)':20s} {'旋转90°':10s} {abs_err:.4f}       {rel_err*100:.1f}%")

abs_err, rel_err = check_equivariance_rotate(mri_A, test_x, angle=90)
results[('MRI欠采样', '旋转')] = abs_err
rel_results[('MRI欠采样', '旋转')] = rel_err
print(f"  {'MRI欠采样':20s} {'旋转90°':10s} {abs_err:.4f}       {rel_err*100:.1f}%")

# 可视化
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

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
ax.set_ylabel(r'相对误差 $\|AT_g - T_gA\| / \|Ax\|$ (%)')
ax.set_title(r'Step 4: 算子-等变性对照实验' + '\n'
             + r'(bar越高→越非等变→EI越可利用此对称性)' + '\n'
             + '对比不同MRI采样模式的等变性差异')
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
print("实验17.6-1 总结")
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

print(f"""
  ╔═══════════════════════════════════════════════════════════════════╗
  ║           实验17.4-17.6-1 三部曲逻辑链条                          ║
  ╠═══════════════════════════════════════════════════════════════════╣
  ║  实验      │   核心问题            │   解决方法      │   连接点       ║
  ╠═══════════════════════════════════════════════════════════════════╣
  ║  17.4-1    │   朴素MSE有偏         │   N2N / N2B     │   配对噪声/空间配对 ║
  ║  17.5-1    │   需要配对噪声        │   SURE (修正项)  │   从N2N进化到SURE  ║
  ║  17.6-1    │   A≠I时SURE只约束值空间 │ EI (等变约束) │   SURE+EI=完整约束 ║
  ╠═══════════════════════════════════════════════════════════════════╣
  ║  17.4-1：当 y = x + ε（噪声）时，问：如何自监督？                     ║
  ║  17.5-1：当 y = x + ε 且 A = I 时，答：SURE修正偏差                   ║
  ║  17.6-1：当 A ≠ I（inpainting/MRI）时，问：SURE失效怎么办？           ║
  ║          答：EI加约束，利用对称性约束零空间                            ║
  ╚═══════════════════════════════════════════════════════════════════╝
""")
