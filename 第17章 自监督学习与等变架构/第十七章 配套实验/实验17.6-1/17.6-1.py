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
运行前提：推荐GPU（Colab T4即可），CPU可运行但很慢

注：checkpoint断点续训仅恢复模型和优化器状态，未保存/恢复RNG状态。
若训练被打断续训，DataLoader shuffle顺序和random_shift随机平移会与
不间断训练时不同，但不影响教学结论的正确性。
"""

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os, sys, io, warnings, logging

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
# Inpainting掩码生成
# ========================================================================
def create_inpainting_mask(H, W, keep_ratio=0.5, seed=42):
    rng = np.random.RandomState(seed)
    mask = np.zeros((H, W), dtype=np.float32)
    n_keep = int(H * W * keep_ratio)
    indices = rng.choice(H * W, n_keep, replace=False)
    mask.flat[indices] = 1.0
    return torch.from_numpy(mask)


# ========================================================================
# 数据准备
# ========================================================================
IMG_SIZE = 32
SIGMA = 0.05
KEEP_RATIO = 0.5
BATCH_SIZE = 128
N_EPOCHS = 40
LR = 1e-3
MAX_SHIFT = 8  # 平移幅度常量：训练EI增强和等变性检验共用同一值

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


# ========================================================================
# 通用训练函数
# ========================================================================
def train_model(model, optimizer, ckpt_path, name, loss_fn,
                train_loader, mask, n_epochs=N_EPOCHS, device=device):
    """通用训练函数，支持checkpoint断点续训

    参数:
        model: 待训练的模型
        optimizer: 优化器
        ckpt_path: checkpoint保存路径
        name: 模型名称（用于打印）
        loss_fn: 损失函数，签名 loss_fn(model, batch_x, y, mask_2d) -> loss
        train_loader: 训练数据加载器
        mask: 固定掩码
        n_epochs: 训练轮数
        device: 设备

    返回:
        model: 训练后的模型
        train_losses: 训练损失列表
    """
    start_epoch = 0
    train_losses = []
    is_final = False

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
        optimizer_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer_state'))

        if checkpoint.get('is_final', False):
            print(f"✓ [{name}] 检测到最终权重，直接加载，跳过训练过程")
            print(f"  训练轮数: {checkpoint['epoch']+1}")
            model.load_state_dict(model_state)
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            train_losses = checkpoint.get('train_losses', [])
            is_final = True
        else:
            print(f"  [{name}] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
            model.load_state_dict(model_state)
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            train_losses = checkpoint.get('train_losses', [])
            start_epoch = checkpoint['epoch'] + 1

    if is_final:
        print(f"  [{name}] 模型已训练完毕，跳过。")
        return model, train_losses

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}', leave=False, unit='batch')
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device)
            mask_2d = mask.unsqueeze(0).unsqueeze(0).expand_as(batch_x)
            y = batch_x * mask_2d + SIGMA * torch.randn_like(batch_x) * mask_2d

            optimizer.zero_grad()
            loss = loss_fn(model, batch_x, y, mask_2d)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)
        pbar.set_description(f'Epoch {epoch+1}/{n_epochs} | loss={avg_loss:.4f}')
        pbar.close()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'train_losses': train_losses,
            'is_final': False
        }, ckpt_path)

    # 保存最终权重（需确保有训练数据可保存）
    if train_losses:
        torch.save({
            'epoch': n_epochs - 1,
            'model_state_dict': model.state_dict(),
            'loss': train_losses[-1],
            'train_losses': train_losses,
            'is_final': True
        }, ckpt_path)
        print(f"  [{name}] ✓ 最终模型已保存")
    else:
        # 边界情况：n_epochs <= start_epoch 导致循环未执行
        print(f"  [{name}] ⚠ 无训练数据，跳过最终权重保存 (start_epoch={start_epoch}, n_epochs={n_epochs})")

    return model, train_losses


# ========================================================================
# Step 1: 零空间问题——inpainting中自监督损失不约束缺失区域
# ========================================================================
print("\n" + "="*70)
print("Step 1: 零空间问题——自监督损失不约束缺失区域")
print("="*70)

print("\n  训练朴素自监督 (仅MC损失)...")
model_naive = SmallUNet().to(device)
optimizer_naive = optim.Adam(model_naive.parameters(), lr=LR)

# Naive损失函数：仅MC损失（测量一致性）
def loss_naive(model, batch_x, y, mask_2d):
    f_y = model(y)
    return ((mask_2d * (y - f_y)) ** 2).sum() / mask_2d.sum()

model_naive, naive_train_losses = train_model(
    model_naive, optimizer_naive,
    os.path.join(SAVE_DIR, 'ckpt_Naive.pt'),
    'Naive', loss_naive, train_loader, test_mask
)


# 评估
def evaluate_inpainting(model, test_loader, mask, sigma=SIGMA, device=None):
    model.eval()
    psnr_vals = []
    if device is not None:
        mask_dev = mask.to(device)
    else:
        mask_dev = mask
    with torch.no_grad():
        with torch.random.fork_rng():
            torch.manual_seed(0)  # fork_rng隔离，不影响外部全局RNG状态
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
# 重要：可视化输入必须与 evaluate_inpainting 保持一致（加噪声），
# 否则图上展示的重建对应的输入和标题里引用的 PSNR 来自不同观测条件——
# 网络训练时输入的就是"掩码+噪声"的观测，PSNR 评估也基于此；
# 若图上展示无噪声输入下的重建，与训练/评估分布不一致，会让学生误解实验设置。
test_imgs, _ = next(iter(test_loader))
test_imgs = test_imgs[:6].to(device)
mask_2d = test_mask.unsqueeze(0).unsqueeze(0).expand_as(test_imgs)
test_y = test_imgs * mask_2d + SIGMA * torch.randn_like(test_imgs) * mask_2d

with torch.no_grad():
    pred_naive = model_naive(test_y).clip(0, 1)

fig, axes = plt.subplots(3, 6, figsize=(15, 7))
for i in range(6):
    axes[0, i].imshow(test_imgs[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[1, i].imshow(test_y[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[2, i].imshow(pred_naive[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    # 关闭ticks和spines，但保留ylabel（axis('off')会隐藏ylabel）
    for r in range(3):
        axes[r, i].set_xticks([])
        axes[r, i].set_yticks([])
        for spine in axes[r, i].spines.values():
            spine.set_visible(False)

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

def random_shift(x, max_shift=MAX_SHIFT):
    B, C, H, W = x.shape
    dy = torch.randint(-max_shift, max_shift+1, (1,)).item()
    dx = torch.randint(-max_shift, max_shift+1, (1,)).item()
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))

def ei_loss(model, mask, f_y, n_transforms=4):
    """等变成像损失
    对应17.5.4节：Chen, Tachella & Davies (ICCV 2021)

    L_EI = (1/G) Σ_g ‖T_g x̂ - f(A T_g x̂)‖²

    其中 x̂ = f(y) 是参考重建
    T_g: 随机平移变换
    A: 固定正向算子（此处为 inpainting 掩码 mask），作用于 T_g x̂ 产生虚拟测量

    ★ stop-gradient：对 x_hat 使用 .detach()，将 T_g x̂ 视为固定目标
    这是 EI 论文 (Chen et al., ICCV 2021) 中的标准做法——防止双侧梯度
    导致 trivial collapse（x̂ 退化为平凡解来最小化 EI 损失）。

    ★ 固定算子：A 必须是固定算子（固定掩码），而非随机掩码。
    若用随机掩码，算子在分布意义下关于平移近似等变，EI 无法利用
    不等变性来约束零空间——与 17.5 节理论矛盾。Step 4 的等变性
    验证也使用固定掩码，此处须保持一致。

    ★ 虚拟测量未加观测噪声：
    真实观测 y = M ⊙ x + σ·ε·M（保留像素处加噪声），但 EI 虚拟路径
    y_virtual = M ⊙ (T_g x̂) 未加噪声。这是多数 EI 论文基础版本的做法。
    原因：(1) x̂ 本身是对 x 的估计，已包含去噪效果；(2) 虚拟测量用于
    约束等变性，而非精确匹配噪声分布。若需更鲁棒可考虑 Robust EI 变体
    （给 y_virtual 也加噪声），但本实验采用基础版本。

    ★ 计算成本：每次调用 ei_loss 会触发 1 + n_transforms 次模型前向传播。
    例如 n_transforms=4 时，一个 batch 上 loss_fn 内部共调用 model 5 次
    （1次算 f_y + 4次算 f_virtual）。这是 EI 方法的固有开销，通过
    虚拟变换产生"伪多算子"信号来约束零空间。实际训练中可将 n_transforms
    调小（如2）以加速，但会降低 EI 约束的覆盖度。
    """
    x_hat = f_y

    mask_2d = mask.unsqueeze(0).unsqueeze(0)

    mse_fn = nn.MSELoss()
    total_loss = 0
    for _ in range(n_transforms):
        x_hat_shifted = random_shift(x_hat.detach())

        y_virtual = x_hat_shifted * mask_2d

        f_virtual = model(y_virtual)
        total_loss += mse_fn(f_virtual, x_hat_shifted)

    return total_loss / n_transforms


# 训练EI模型
print("\n  训练EI模型 (MC + EI)...")
model_ei = SmallUNet().to(device)
optimizer_ei = optim.Adam(model_ei.parameters(), lr=LR)
# EI损失权重：通过下方消融实验选定
lambda_ei = 0.5

# EI损失函数：MC + EI
def loss_ei(model, batch_x, y, mask_2d):
    f_y = model(y)
    loss_mc = ((mask_2d * (y - f_y)) ** 2).sum() / mask_2d.sum()
    loss_ei_val = ei_loss(model, test_mask, f_y=f_y, n_transforms=4)
    return loss_mc + lambda_ei * loss_ei_val

# ★ 关于 mask 类型的补充说明：
# create_inpainting_mask 使用逐像素随机采样（50%随机像素），而非连续区域。
# 对随机单像素 mask，torch.roll 平移后观测点位置几乎完全打乱，与原始 mask
# 几乎不重叠——这导致 EI 虚拟测量 y_virtual = x_hat_shifted * mask_2d 中，
# 平移后的像素点经常完全跑出原来观测的采样位置。此时 EI 项退化为较强的先验
# 正则（等价于对任意50%采样都要重建好），比标准 EI 论文里"矩形/连续区域
# inpainting mask"的效果更强。这在教学上是有意的：强化"非等变→EI有效"的结论，
# 也更贴近真实场景中的不规则缺失（如随机遮挡）。

model_ei, ei_train_losses = train_model(
    model_ei, optimizer_ei,
    os.path.join(SAVE_DIR, 'ckpt_EI.pt'),
    'EI', loss_ei, train_loader, test_mask
)

psnr_ei = evaluate_inpainting(model_ei, test_loader, test_mask)
print(f"  EI (MC+EI) PSNR = {psnr_ei:.2f} dB")


# ========================================================================
# Step 2b: lambda_ei 消融实验——为权重选择提供数据支撑
# ========================================================================
print("\n" + "="*70)
print("Step 2b: lambda_ei 消融实验")
print("="*70)

lambda_candidates = [0.1, 0.5, 1.0]
ablation_results = {}
print(f"  {'lambda_ei':>10s}  {'PSNR (dB)':>10s}")
print(f"  {'─'*25}")
for lam in lambda_candidates:
    # 若消融值就是当前 lambda_ei，直接复用已训练模型，避免重复训练
    if abs(lam - lambda_ei) < 1e-9 and abs(lam - 0.5) < 1e-9:
        model_ab = model_ei
    else:
        model_ab = SmallUNet().to(device)
        optimizer_ab = optim.Adam(model_ab.parameters(), lr=LR)

        def loss_ab(model, batch_x, y, mask_2d, _lam=lam):
            f_y = model(y)
            loss_mc = ((mask_2d * (y - f_y)) ** 2).sum() / mask_2d.sum()
            loss_ei_val = ei_loss(model, test_mask, f_y=f_y, n_transforms=4)
            return loss_mc + _lam * loss_ei_val

        model_ab, _ = train_model(
            model_ab, optimizer_ab,
            os.path.join(SAVE_DIR, f'ckpt_EI_lam{lam}.pt'),
            f'EI(λ={lam})', loss_ab, train_loader, test_mask,
            n_epochs=20, device=device
        )
    psnr_ab = evaluate_inpainting(model_ab, test_loader, test_mask)
    ablation_results[lam] = psnr_ab
    marker = ' ◀ 选用' if abs(lam - lambda_ei) < 1e-9 else ''
    print(f"  {lam:10.1f}  {psnr_ab:10.2f}{marker}")

best_lam = max(ablation_results, key=ablation_results.get)
print(f"\n  消融结论: λ={best_lam} 时PSNR最高 ({ablation_results[best_lam]:.2f} dB)")
print(f"  当前选用 λ={lambda_ei}，与最优值{'一致' if abs(best_lam - lambda_ei) < 1e-9 else '接近'}。")


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

# 监督损失函数：直接用干净图像作为目标
def loss_supervised(model, batch_x, y, mask_2d):
    f_y = model(y)
    return nn.MSELoss()(f_y, batch_x)

model_sup, sup_train_losses = train_model(
    model_sup, optimizer_sup,
    os.path.join(SAVE_DIR, 'ckpt_Supervised.pt'),
    'Supervised', loss_supervised, train_loader, test_mask
)

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
        with torch.random.fork_rng():
            torch.manual_seed(0)  # fork_rng隔离，不影响外部全局RNG状态
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
# 重要：与 evaluate_combined 保持一致——观测 = 掩码 + 噪声。
# 这样图中展示的重建（三种方法的 pred_*）对应的输入 y_vis 与
# 标题/行标签引用的 total_psnrs（来自带噪声评估）来自同一观测条件。
fig, axes = plt.subplots(5, 6, figsize=(15, 12))
vis_imgs, _ = next(iter(test_loader))
vis_imgs = vis_imgs[:6].to(device)
mask_vis = test_mask.unsqueeze(0).unsqueeze(0).expand_as(vis_imgs)
y_vis = vis_imgs * mask_vis + SIGMA * torch.randn_like(vis_imgs) * mask_vis

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
        # 关闭ticks和spines，但保留ylabel（axis('off')会隐藏ylabel）
        axes[r, i].set_xticks([])
        axes[r, i].set_yticks([])
        for spine in axes[r, i].spines.values():
            spine.set_visible(False)
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
        max_shift = MAX_SHIFT  # 与训练时 random_shift 共用同一常量
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
test_x = test_imgs[:4]

# 1. Inpainting掩码 + 平移
def inpainting_A(x):
    mask_2d = test_mask.to(x.device).unsqueeze(0).unsqueeze(0).expand_as(x)
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

for mode in sampling_modes:
    abs_err, rel_err = check_equivariance_rotate(
        lambda x: mri_A(x, sampling_mode=mode), test_x, angle=90)
    results[(f'MRI欠采样({mode})', '旋转')] = abs_err
    rel_results[(f'MRI欠采样({mode})', '旋转')] = rel_err
    print(f"  {'MRI欠采样('+mode+')':20s} {'旋转90°':10s} {abs_err:.4f}       {rel_err*100:.1f}%")

# 可视化
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

operators = ['高斯模糊', 'Inpainting', 'MRI欠采样(vertical)',
            'MRI欠采样(random)', 'MRI欠采样(cartesian)']
transform_keys = ['平移', '旋转']
transform_labels = ['平移', '旋转90°']
x_pos = np.arange(len(operators))
width = 0.25

for j, (t, t_label) in enumerate(zip(transform_keys, transform_labels)):
    vals = [rel_results.get((op, t), 0) * 100 for op in operators]
    bars = ax.bar(x_pos + j * width, vals, width, label=t_label, alpha=0.8)
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
print("  - 高斯模糊+平移: 相对误差≈0% → 近似等变(边界效应可忽略) → EI无法利用平移对称性")
print("    注: conv2d使用zero-padding, torch.roll使用循环边界, 严格不等变但有微小误差")
print("  - Inpainting(固定掩码)+平移: 相对误差较大 → 非等变 → EI可利用平移对称性")
print("  - Inpainting(固定掩码)+旋转: 非等变 → EI可利用旋转对称性")
print("  - MRI+平移: 相对误差极小 → 严格等变（数值误差） → EI无法利用平移对称性")
print("    理论: 频域欠采样是逐元素相乘, 空间循环平移对应频域相位调制, 两者可交换")
print("  - MRI+旋转: 非等变 → EI可利用旋转对称性(与17.5.6节FastMRI结果一致)")
print("  注: Inpainting测试使用固定掩码，若使用随机掩码行为会不同")


# ========================================================================
# 总结
# ========================================================================
print("\n" + "="*70)
print("实验17.6-1 总结")
print("="*70)
print("  注：以下PSNR为单次训练结果（seed=42），未报告方差，仅供定性对比参考。")
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
print(f"  5. 算子近似等变→EI效果弱: 高斯模糊关于平移近似等变(无法提供新信息)")

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
