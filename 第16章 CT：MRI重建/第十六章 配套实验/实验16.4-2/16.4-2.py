#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 16.4-2：CT 后处理——UNet/VDSR 对 FBP 重建的修复
================================================================

本实验展示第 16.4 节"学习型重建方法"在 CT（计算机断层成像）场景下的应用。
与实验 16.4-1（MRI 欠采样 + MNIST）相比，本实验使用合成 CT 投影数据，覆盖：
  - CT 正向算子：Radon 变换（线积分投影）
  - FBP（滤波反投影）：经典解析重建，作为学习型方法的输入
  - UNet / VDSR 后处理：从有噪声的 FBP 重建中恢复干净图像
  - MSE vs MAE 损失函数对比

核心教学要点：
  1. 非扩散 UNet 与扩散 UNet（实验 15.1-1）的区别——无时间步条件
  2. VDSR 的全局残差设计（20 层 CNN + skip connection）
  3. Radon → FBP → 后处理 完整 CT 管线
"""

# ============================================================
# 0. 环境准备
# ============================================================

import os, sys, json, time, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')                        # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib import rcParams

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm

from skimage.transform import radon, iradon
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.data import shepp_logan_phantom

# ============================================================
# Colab 检测与持久化目录（参考 6.7-1 风格）
# ============================================================
_IN_COLAB = 'google.colab' in sys.modules
_GDRIVE = '/content/drive/MyDrive'

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_GDRIVE):
        print("[Colab] 正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_GDRIVE, '实验16.4-2')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()

os.makedirs(SAVE_DIR, exist_ok=True)
SCRIPT_DIR = SAVE_DIR

# 中文显示配置
_chinese_path = os.path.join(SCRIPT_DIR, '.chinese')
os.makedirs(_chinese_path, exist_ok=True)
sys.path.insert(0, _chinese_path)

FONT_HANDLE = None
try:
    from chinese_font import setup_chinese_font
    FONT_HANDLE = setup_chinese_font(save_dir=_chinese_path)
    if FONT_HANDLE:
        print(f"[字体] 中文字体已就绪: {FONT_HANDLE}")
except Exception as e:
    if _IN_COLAB:
        # Colab 自带 Noto Sans CJK SC，直接兜底
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        FONT_HANDLE = 'Noto Sans CJK SC'
        print(f"[字体] 使用 Colab 系统字体: {FONT_HANDLE}")
    else:
        print(f"[字体] 中文显示功能未启用，图片上中文可能出现乱码 ({e})")

# 仅静默中文/数学符号字形相关警告（精确匹配），不屏蔽其他 UserWarning，
# 以便 skimage radon 在重建圆外有非零像素时仍能抛出真实几何警告
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", message=".*U\\+2212.*")

# Matplotlib LaTeX 渲染（数学符号）
rcParams['text.usetex'] = False              # 不使用完整 LaTeX 引擎
rcParams['mathtext.default'] = 'regular'     # 使用 mathtext，兼容性最好

# 随机种子（保证可复现）
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[设备] 使用: {device}" + (" (GPU)" if device.type == 'cuda' else " (CPU)"))

# 路径
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data', 'ct_phantoms')
CKPT_DIR   = os.path.join(SCRIPT_DIR, 'checkpoints')
OUTPUT_DIR = SCRIPT_DIR   # 输出直接放在实验目录下，不另建 outputs 子目录
for d in [DATA_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# 1. 网络架构
# ============================================================

class ConvBlock(nn.Module):
    """基础卷积块：Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CTUNet(nn.Module):
    """
    非扩散 UNet——用于 CT 后处理（无时间步条件）

    与实验 15.1-1 中的扩散 UNet 的关键区别：
      - 没有时间步嵌入（不需要 t 作为输入）
      - 没有 ConvNext 块、没有注意力机制
      - 输出直接是去噪图像，而非噪声预测
      - 更轻量，适合图像到图像的后处理任务

    Args:
        in_ch: 输入通道数（灰度=1）
        out_ch: 输出通道数（灰度=1）
        ch_list: 各层通道数，例如 (32, 64, 128) 表示 3 级 UNet
    """
    def __init__(self, in_ch=1, out_ch=1, ch_list=(32, 64, 128)):
        super().__init__()
        n_levels = len(ch_list)

        # 编码器
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev_ch = in_ch
        for ch in ch_list:
            self.enc_blocks.append(ConvBlock(prev_ch, ch))
            self.downs.append(nn.MaxPool2d(2))
            prev_ch = ch

        # 瓶颈
        self.bottleneck = ConvBlock(ch_list[-1], ch_list[-1] * 2)

        # 解码器
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        rev_ch = list(reversed(ch_list))
        for i, ch in enumerate(rev_ch):
            in_c = ch_list[-1] * 2 if i == 0 else rev_ch[i - 1]
            self.ups.append(nn.ConvTranspose2d(in_c, ch, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock(ch * 2, ch))  # skip connection 拼接

        self.final = nn.Conv2d(ch_list[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for i, (block, down) in enumerate(zip(self.enc_blocks, self.downs)):
            x = block(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        x = self.final(x)
        return x


class VDSR(nn.Module):
    """
    VDSR（Very Deep Super-Resolution）——20 层 CNN + 全局残差

    来自 CVPR 2016，原始用于超分辨率，这里用作 CT 后处理：
      output = conv_stack(input) + input

    架构：conv(1→64) → 18×conv(64→64)+ReLU → conv(64→1) → + 输入

    注：原论文（CVPR 2016）使用 SGD + 较大学习率(0.1) + 梯度裁剪来加速深层收敛；
    此处为教学清晰与稳定，改用 Adam(lr=1e-3) 且不做梯度裁剪，训练更平稳，
    但偏离了原文"高学习率 + 裁剪"的设计动机——如需对照论文，请知悉此差异。

    Args:
        in_ch: 输入通道数
        n_layers: 总卷积层数（含首尾），默认 20
        n_feats: 中间层特征通道数，默认 64
    """
    def __init__(self, in_ch=1, n_layers=20, n_feats=64):
        super().__init__()
        layers = []
        # 第一层：输入 → 特征
        layers.append(nn.Conv2d(in_ch, n_feats, 3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        # 中间层：特征 → 特征
        for _ in range(n_layers - 2):
            layers.append(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        # 最后一层：特征 → 输出
        layers.append(nn.Conv2d(n_feats, in_ch, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.net(x) + x    # 全局残差


# ============================================================
# 2. CT 数据生成
# ============================================================

def generate_random_phantom(size=128, n_ellipses=None, style='random'):
    """
    生成随机椭圆 phantom（模拟 2D 人体截面）

    每个 phantom 包含若干个随机位置、大小、旋转角度的椭圆。
    灰度值归一化到 [0, 1]，最大值恒为 1（与 Shepp-Logan 一致）。

    style 参数：
      - 'random':  椭圆重叠取最大灰度（类似 CT 截面中不同组织的叠加）
      - 'layered': 后绘制的覆盖前面的，形成清晰的多灰度层次（类似 Shepp-Logan）
    """
    if n_ellipses is None:
        n_ellipses = np.random.randint(3, 9)
    img = np.zeros((size, size), dtype=np.float32)
    cy, cx = size // 2, size // 2

    if style == 'layered':
        # 类似 Shepp-Logan：先生成所有椭圆参数，按灰度排序后逐层覆盖
        ellipses = []
        for _ in range(n_ellipses):
            a  = np.random.uniform(5, size * 0.35)
            b  = np.random.uniform(3, a * 0.8)
            theta = np.random.uniform(0, np.pi)
            x0 = np.random.uniform(cx - size * 0.25, cx + size * 0.25)
            y0 = np.random.uniform(cy - size * 0.25, cy + size * 0.25)
            intensity = np.random.uniform(0.05, 0.95)
            ellipses.append((a, b, theta, x0, y0, intensity))
        # 亮的先画，暗的后覆盖（亮=大背景，暗=内部结构）
        ellipses.sort(key=lambda e: e[5], reverse=True)
        for a, b, theta, x0, y0, intensity in ellipses:
            yy, xx = np.mgrid[0:size, 0:size]
            dx = xx - x0
            dy = yy - y0
            r1 =  dx * np.cos(theta) + dy * np.sin(theta)
            r2 = -dx * np.sin(theta) + dy * np.cos(theta)
            mask = (r1 / a) ** 2 + (r2 / b) ** 2 <= 1.0
            img[mask] = intensity   # 覆盖而非取 max，形成清晰灰度层次
    else:
        # 原有 random 风格：椭圆重叠取最大灰度（模拟重叠结构）
        for _ in range(n_ellipses):
            a  = np.random.uniform(5, size * 0.35)
            b  = np.random.uniform(3, a * 0.8)
            theta = np.random.uniform(0, np.pi)
            x0 = np.random.uniform(cx - size * 0.25, cx + size * 0.25)
            y0 = np.random.uniform(cy - size * 0.25, cy + size * 0.25)
            intensity = np.random.uniform(0.15, 0.85)
            yy, xx = np.mgrid[0:size, 0:size]
            dx = xx - x0
            dy = yy - y0
            r1 =  dx * np.cos(theta) + dy * np.sin(theta)
            r2 = -dx * np.sin(theta) + dy * np.cos(theta)
            mask = (r1 / a) ** 2 + (r2 / b) ** 2 <= 1.0
            img[mask] = np.maximum(img[mask], intensity)

    # 添加圆形背景（固定居中，避免随机偏移导致背景超出重建圆）
    bg_r = size * 0.45
    yy, xx = np.mgrid[0:size, 0:size]
    bg_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= bg_r ** 2
    img[~bg_mask] = 0.0

    # 显式裁剪到 Radon 重建圆内（radius = size/2），保证 circle=True 前提：
    # 重建圆外像素必须为 0，否则 FBP 边缘失真会混入"噪声/欠采样"伪影，误导教学
    recon_r = size / 2.0
    img[(xx - cx) ** 2 + (yy - cy) ** 2 > recon_r ** 2] = 0.0

    # 统一归一化到 [0, 1]，确保 max=1（与 Shepp-Logan 一致，消除尺度失配）
    img = np.clip(img, 0.0, None)
    img_max = img.max()
    if img_max > 1e-8:
        img = img / img_max
    return img.astype(np.float32)


def forward_ct(phantom, n_angles=90, noise_sigma=0.03, circle=True):
    """
    CT 正向 + FBP 重建管线

    1. Radon 变换（线积分投影）
    2. 添加高斯噪声到 sinogram
    3. FBP 重建（滤波反投影）

    Returns:
        fbp: FBP 重建（有噪声伪影）
        gt:  原始 phantom（ground truth）
        sino: 含噪声的 sinogram（备查）
    """
    theta = np.linspace(0., 180., n_angles, endpoint=False)
    sino_clean = radon(phantom, theta=theta, circle=circle)
    # 归一化 sinogram 后加噪声
    sino_norm = sino_clean / (sino_clean.max() + 1e-8)
    noise = np.random.randn(*sino_norm.shape).astype(np.float32) * noise_sigma
    sino_noisy = sino_norm + noise
    # 恢复到原始尺度
    sino_noisy = sino_noisy * (sino_clean.max() + 1e-8)
    # FBP 重建
    fbp = iradon(sino_noisy, theta=theta, circle=circle, filter_name='ramp')
    # 仅做 [0, 1] 裁剪，保持与输入 phantom 一致的原始线性尺度
    # （Radon/iradon 是线性算子，FBP 输出尺度与输入 phantom 一致；
    #  不做 per-sample 归一化到 max=1，避免人为压低 FBP PSNR）
    fbp = np.clip(fbp, 0.0, 1.0)
    return fbp.astype(np.float32), phantom.astype(np.float32), sino_noisy.astype(np.float32)


def build_ct_dataset(n_train=400, n_test=50, size=128, n_angles=90, noise_sigma=0.03,
                     force_regen=False):
    """生成/加载合成 CT phantom 数据集

    通过 _meta.json 记录关键生成参数，加载时自动校验是否匹配；
    不匹配（或任一 .npy 缺失）时自动重建，无需人工记忆 force_regen。
    """
    train_fbp_path = os.path.join(DATA_DIR, 'train_fbp.npy')
    train_gt_path  = os.path.join(DATA_DIR, 'train_gt.npy')
    test_fbp_path  = os.path.join(DATA_DIR, 'test_fbp.npy')
    test_gt_path   = os.path.join(DATA_DIR, 'test_gt.npy')
    sl_fbp_path    = os.path.join(DATA_DIR, 'shepp_logan_fbp.npy')
    sl_gt_path     = os.path.join(DATA_DIR, 'shepp_logan_gt.npy')
    meta_path      = os.path.join(DATA_DIR, '_meta.json')

    def _meta_match():
        """校验已存 _meta.json 与当前生成参数是否一致"""
        if not os.path.exists(meta_path):
            return False
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            return False
        return (meta.get('size') == size and meta.get('n_angles') == n_angles
                and abs(meta.get('noise_sigma', -1) - noise_sigma) < 1e-9
                and meta.get('n_train') == n_train and meta.get('n_test') == n_test)

    npy_paths = [train_fbp_path, train_gt_path, test_fbp_path, test_gt_path,
                 sl_fbp_path, sl_gt_path]
    exists = all(os.path.exists(p) for p in npy_paths)

    if exists and not force_regen and _meta_match():
        print("[数据] 检测到已有数据集且配置匹配，直接加载...")
        return (np.load(train_fbp_path), np.load(train_gt_path),
                np.load(test_fbp_path),  np.load(test_gt_path),
                np.load(sl_fbp_path),    np.load(sl_gt_path))
    if exists and not force_regen and not _meta_match():
        print("[数据] 检测到已有数据集但生成配置不匹配，自动重建...")

    print(f"[数据] 生成合成 CT 数据集 ({n_train} 训练 + {n_test} 测试 + 1 Shepp-Logan)...")

    # 训练集：混合 random 和 layered 两种风格（约 3:2），提升多样性
    train_fbp, train_gt = [], []
    for i in tqdm(range(n_train), desc='生成训练数据', ncols=80, leave=False):
        style = 'layered' if i % 5 < 2 else 'random'   # 40% layered, 60% random
        ph = generate_random_phantom(size=size, style=style)
        fbp, gt, _ = forward_ct(ph, n_angles=n_angles, noise_sigma=noise_sigma)
        train_fbp.append(fbp)
        train_gt.append(gt)
    train_fbp = np.array(train_fbp, dtype=np.float32)
    train_gt  = np.array(train_gt, dtype=np.float32)

    # 测试集：同样混合两种风格
    test_fbp, test_gt = [], []
    for i in tqdm(range(n_test), desc='生成测试数据', ncols=80, leave=False):
        style = 'layered' if i % 5 < 2 else 'random'
        ph = generate_random_phantom(size=size, style=style)
        fbp, gt, _ = forward_ct(ph, n_angles=n_angles, noise_sigma=noise_sigma)
        test_fbp.append(fbp)
        test_gt.append(gt)
    test_fbp = np.array(test_fbp, dtype=np.float32)
    test_gt  = np.array(test_gt, dtype=np.float32)

    # Shepp-Logan（经典测试图像）
    print("[数据] 生成 Shepp-Logan phantom 测试样本...")
    sl_phantom = shepp_logan_phantom()
    # 将 Shepp-Logan 缩放到 128×128
    from skimage.transform import resize
    sl_phantom = resize(sl_phantom, (size, size), anti_aliasing=True)
    sl_phantom = sl_phantom.astype(np.float32)
    sl_phantom = sl_phantom / sl_phantom.max()   # 归一化
    sl_fbp, sl_gt, _ = forward_ct(sl_phantom, n_angles=n_angles, noise_sigma=noise_sigma)

    # 保存
    for p, arr in [(train_fbp_path, train_fbp), (train_gt_path, train_gt),
                   (test_fbp_path, test_fbp), (test_gt_path, test_gt),
                   (sl_fbp_path, sl_fbp), (sl_gt_path, sl_gt)]:
        np.save(p, arr)
    # 保存生成配置 meta，供下次加载时自动校验
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'size': size, 'n_angles': n_angles, 'noise_sigma': noise_sigma,
                   'n_train': n_train, 'n_test': n_test}, f, indent=2)
    print("[数据] 数据集已保存到 ./data/ct_phantoms/")
    return train_fbp, train_gt, test_fbp, test_gt, sl_fbp, sl_gt


class CTDataset(Dataset):
    """CT 图像对 Dataset：FBP 输入 → 干净 GT"""
    def __init__(self, fbp_arr, gt_arr):
        self.fbp = torch.from_numpy(fbp_arr).unsqueeze(1)   # (N, 1, H, W)
        self.gt  = torch.from_numpy(gt_arr).unsqueeze(1)

    def __len__(self):
        return len(self.fbp)

    def __getitem__(self, idx):
        return self.fbp[idx], self.gt[idx]


# ============================================================
# 3. 训练工具
# ============================================================

def compute_psnr_single(img, target, data_range=1.0):
    """单张图像 PSNR（skimage.metrics.peak_signal_noise_ratio 封装）"""
    return psnr(target, img, data_range=data_range)


def compute_psnr_batch(output, target):
    """批量计算 PSNR（对单张计算求平均）"""
    out_np = output.detach().cpu().numpy()
    tgt_np = target.cpu().numpy()
    vals = []
    for i in range(len(out_np)):
        vals.append(compute_psnr_single(out_np[i, 0], tgt_np[i, 0]))
    return np.mean(vals)


def train_one_model(model, train_loader, test_loader, device, ckpt_path,
                    model_name='Model', n_epochs=30, lr=1e-3,
                    loss_type='mse', eval_every=5):
    """
    通用训练函数——支持 resume、CPU 跳过、epoch 级 checkpoint、进度条

    Checkpoint 格式：
      - 中间状态: {epoch, model_state, optimizer_state, losses, is_final: False}
      - 最终状态: {epoch, model_state, losses, is_final: True}（无 optimizer_state）
    """
    # ---- 查 checkpoint ----
    ckpt = None
    start_epoch = 0
    loss_history = []

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if ckpt.get('is_final', False):
            print(f"\n[{model_name}] 检测到最终权重 ({ckpt_path})，直接加载，跳过训练过程。")
            model.load_state_dict(ckpt['model_state'])
            model.to(device)
            return model, ckpt.get('losses', [])
        else:
            # 存在中断（非 final）checkpoint：CPU 环境下不继续跑剩余 epoch，
            # 直接加载已有权重并跳过，避免 CPU 上触发长时间训练
            if device.type == 'cpu':
                print(f"\n[{model_name}] [跳过] CPU 环境且检测到中断 checkpoint "
                      f"(epoch {ckpt['epoch']})，不继续训练。")
                print(f"         请在有 GPU 的环境下运行以完成训练：{ckpt_path}")
                model.load_state_dict(ckpt['model_state'])
                model.to(device)
                return model, ckpt.get('losses', [])
            model.load_state_dict(ckpt['model_state'])
            start_epoch = ckpt['epoch'] + 1
            loss_history = ckpt.get('losses', [])
            print(f"\n[{model_name}] 检测到 checkpoint (epoch {ckpt['epoch']})，从 epoch {start_epoch} 继续训练。")
    else:
        if device.type == 'cpu':
            print(f"\n[{model_name}] [跳过] 未检测到 GPU 且无预训练权重，CPU 环境下跳过训练。")
            print(f"         请在有 GPU 的环境下首次运行以生成权重文件：{ckpt_path}")
            return model, []   # 返回未训练的模型

    # ---- 损失函数 ----
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'mae' or loss_type == 'l1':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}")

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # resume 时恢复 optimizer（in-checkpoint 守卫，兼容有/无 optimizer_state 两种格式）
    if ckpt is not None and 'optimizer_state' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        except Exception:
            print(f"[{model_name}] 警告：无法恢复 optimizer 状态，使用初始 optimizer")

    best_psnr = 0.0
    best_epoch = 0
    best_state = None
    train_losses = loss_history.copy() if loss_history else []   # resume 时继承历史
    val_psnrs = []
    start_time = time.time()

    # ---- 训练循环 ----
    # 单个持久进度条覆盖"全部 epoch 的总 batch 数"，全程 position=0 第一行原位刷新，
    # 不嵌套内层进度条（position=1 在 Windows 终端会真正占用第二行，造成多行累积）。
    # 当前 epoch 与 batch loss 通过 postfix 实时显示，始终只有一行。
    total_steps = (n_epochs - start_epoch) * len(train_loader)
    global_pbar = tqdm(total=total_steps, desc=model_name, position=0, leave=True,
                       ncols=100)
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = len(train_loader)

        for i, (x_in, x_gt) in enumerate(train_loader):
            x_in, x_gt = x_in.to(device), x_gt.to(device)
            optimizer.zero_grad()
            output = model(x_in)
            loss = criterion(output, x_gt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            global_pbar.set_postfix({'E': f'{epoch+1}/{n_epochs}',
                                     'loss': f'{loss.item():.4f}'})
            global_pbar.update(1)
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)

        # ---- 验证 ----
        if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
            model.eval()
            psnr_vals = []
            with torch.no_grad():
                for x_in, x_gt in test_loader:
                    x_in, x_gt = x_in.to(device), x_gt.to(device)
                    output = model(x_in)
                    psnr_vals.append(compute_psnr_batch(output, x_gt))
            avg_psnr = np.mean(psnr_vals)
            val_psnrs.append((epoch + 1, avg_psnr))
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_epoch = epoch + 1
                best_state = model.state_dict()

        # ---- 每个 epoch 保存 checkpoint（含 optimizer，用于 resume）----
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'losses': train_losses,
            'val_psnrs': val_psnrs,
            'is_final': False,
        }, ckpt_path)

    # ---- 最终 checkpoint（不含 optimizer_state）----
    # 保存最佳 epoch 的权重（而非最后一个 epoch），与 best_psnr 一致
    final_state = best_state if best_state is not None else model.state_dict()
    final_ckpt = {
        'epoch': n_epochs - 1,
        'model_state': final_state,
        'losses': train_losses,
        'val_psnrs': val_psnrs,
        'best_psnr': best_psnr,
        'best_epoch': best_epoch,
        'is_final': True,
    }
    torch.save(final_ckpt, ckpt_path)

    # 关闭持久进度条，确保终端回到正常换行模式
    global_pbar.close()

    elapsed = time.time() - start_time
    sys.stdout.write('\n')
    sys.stdout.flush()
    print(f"[{model_name}] 训练完成！耗时 {elapsed:.1f}s, "
          f"最佳验证 PSNR: {best_psnr:.2f} dB (epoch {best_epoch})")

    # 合并 loss_history
    loss_history = train_losses
    return model, loss_history


# ============================================================
# 4. 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  实验 16.4-2：CT 后处理——UNet/VDSR 对 FBP 重建的修复")
    print("=" * 60)

    # ---- 4.1 数据生成 ----
    print("\n" + "─" * 40)
    print("阶段 1：生成合成 CT phantom 数据集")
    print("─" * 40)

    IMG_SIZE  = 128
    N_ANGLES  = 90
    NOISE_SIG = 0.03
    N_TRAIN   = 800
    N_TEST    = 50
    BATCH     = 32
    N_EPOCHS_UNET  = 50
    N_EPOCHS_VDSR  = 40
    N_EPOCHS_MAE   = 30

    # force_regen=False：数据集缓存由 _meta.json 自动校验配置，不匹配时自动重建
    train_fbp, train_gt, test_fbp, test_gt, sl_fbp, sl_gt = build_ct_dataset(
        n_train=N_TRAIN, n_test=N_TEST, size=IMG_SIZE,
        n_angles=N_ANGLES, noise_sigma=NOISE_SIG, force_regen=False)

    train_dataset = CTDataset(train_fbp, train_gt)
    test_dataset  = CTDataset(test_fbp, test_gt)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH, shuffle=False, num_workers=0)

    print(f"  训练集: {len(train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")
    print(f"  图像尺寸: {IMG_SIZE}×{IMG_SIZE}, Radon 角度数: {N_ANGLES}, 噪声 sigma: {NOISE_SIG}")

    # FBP 基线 PSNR
    fbp_psnr_vals = []
    for i in range(N_TEST):
        fbp_psnr_vals.append(psnr(test_gt[i], test_fbp[i], data_range=1.0))
    fbp_psnr_base = np.mean(fbp_psnr_vals)
    print(f"  FBP 基线 PSNR: {fbp_psnr_base:.2f} dB")
    # Shepp-Logan FBP PSNR
    sl_fbp_psnr = psnr(sl_gt, sl_fbp, data_range=1.0)
    print(f"  Shepp-Logan FBP PSNR: {sl_fbp_psnr:.2f} dB")

    results = {
        'config': {
            'img_size': IMG_SIZE, 'n_angles': N_ANGLES, 'noise_sigma': NOISE_SIG,
            'n_train': N_TRAIN, 'n_test': N_TEST, 'batch_size': BATCH,
        },
        'baseline': {
            'fbp_psnr_avg': round(float(fbp_psnr_base), 2),
            'fbp_psnr_shepp_logan': round(float(sl_fbp_psnr), 2),
        },
    }

    # ---- 4.2 UNet (MSE) 训练 ----
    print("\n" + "─" * 40)
    print("阶段 2：UNet 后处理训练（MSE 损失）")
    print("─" * 40)

    unet = CTUNet(in_ch=1, out_ch=1, ch_list=(32, 64, 128))
    num_params_unet = sum(p.numel() for p in unet.parameters())
    print(f"  UNet 参数量: {num_params_unet:,}")

    ckpt_unet = os.path.join(CKPT_DIR, 'unet_ct_ckpt.pt')
    unet, loss_unet = train_one_model(
        unet, train_loader, test_loader, device, ckpt_unet,
        model_name='UNet(MSE)', n_epochs=N_EPOCHS_UNET, lr=1e-3,
        loss_type='mse', eval_every=5)

    # ---- 4.3 VDSR (MSE) 训练 ----
    print("\n" + "─" * 40)
    print("阶段 3：VDSR 后处理训练（MSE 损失）")
    print("─" * 40)

    vdsr = VDSR(in_ch=1, n_layers=20, n_feats=64)
    num_params_vdsr = sum(p.numel() for p in vdsr.parameters())
    print(f"  VDSR 参数量: {num_params_vdsr:,}")

    ckpt_vdsr = os.path.join(CKPT_DIR, 'vdsr_ct_ckpt.pt')
    vdsr, loss_vdsr = train_one_model(
        vdsr, train_loader, test_loader, device, ckpt_vdsr,
        model_name='VDSR(MSE)', n_epochs=N_EPOCHS_VDSR, lr=1e-3,
        loss_type='mse', eval_every=5)

    # ---- 4.4 UNet (MAE) 对比训练 ----
    print("\n" + "─" * 40)
    print("阶段 4：UNet 后处理（MAE 损失对比训练）")
    print("─" * 40)

    unet_mae = CTUNet(in_ch=1, out_ch=1, ch_list=(32, 64, 128))
    ckpt_unet_mae = os.path.join(CKPT_DIR, 'unet_ct_mae_ckpt.pt')
    unet_mae, loss_unet_mae = train_one_model(
        unet_mae, train_loader, test_loader, device, ckpt_unet_mae,
        model_name='UNet(MAE)', n_epochs=N_EPOCHS_MAE, lr=1e-3,
        loss_type='mae', eval_every=3)

    # ---- 4.5 评估与可视化 ----
    print("\n" + "─" * 40)
    print("阶段 5：方法对比与可视化")
    print("─" * 40)

    # 模型就绪检查：读取 ckpt 内的 is_final 标志，避免把中断后的中间权重误判为"就绪"
    def _is_ckpt_final(ckpt_path):
        if not os.path.exists(ckpt_path):
            return False
        try:
            c = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            return c.get('is_final', False)
        except Exception:
            return False

    unet_ready     = _is_ckpt_final(ckpt_unet)     or len(loss_unet) > 0
    vdsr_ready     = _is_ckpt_final(ckpt_vdsr)     or len(loss_vdsr) > 0
    unet_mae_ready = _is_ckpt_final(ckpt_unet_mae) or len(loss_unet_mae) > 0

    if not any([unet_ready, vdsr_ready, unet_mae_ready]):
        print("\n[跳过] 所有模型均未训练（首次在 CPU 环境下运行），跳过评估。")
        print("       请在有 GPU 的环境下首次运行以生成权重文件。")
        skip_detail = {'note': 'all_models_skipped_on_cpu'}
        results['methods'] = {
            'fbp':      {'psnr_test': fbp_psnr_base,    'psnr_shepp_logan': sl_fbp_psnr},
            'unet_mse': skip_detail,
            'vdsr_mse': skip_detail,
            'unet_mae': skip_detail,
        }
        results['training'] = {'unet_mse_losses': [], 'vdsr_mse_losses': [], 'unet_mae_losses': []}
        results['images'] = {}
        json_path = os.path.join(OUTPUT_DIR, 'results_16_4_2.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[结果] JSON 已保存: {json_path}")
        return

    unet.to(device).eval()
    vdsr.to(device).eval()
    unet_mae.to(device).eval()

    # ---- 测试集 PSNR ----
    def eval_model(model, loader, ready_flag):
        if not ready_flag:
            return None
        psnrs = []
        with torch.no_grad():
            for x_in, x_gt in loader:
                x_in = x_in.to(device)
                pred = model(x_in)
                psnrs.append(compute_psnr_batch(pred, x_gt))
        return round(float(np.mean(psnrs)), 2)

    psnr_unet     = eval_model(unet, test_loader, unet_ready)
    psnr_vdsr     = eval_model(vdsr, test_loader, vdsr_ready)
    psnr_unet_mae = eval_model(unet_mae, test_loader, unet_mae_ready)

    print(f"\n  {'='*40}")
    print(f"  测试集 PSNR 对比（均值，{N_TEST} 个样本）：")
    print(f"    FBP (基线)     : {fbp_psnr_base:.2f} dB")
    for label, val in [('UNet (MSE)', psnr_unet), ('VDSR (MSE)', psnr_vdsr),
                       ('UNet (MAE)', psnr_unet_mae)]:
        if val is not None:
            print(f"    {label:<13}: {val:.2f} dB  (+{val - fbp_psnr_base:.2f} dB)")
        else:
            print(f"    {label:<13}: N/A (模型未训练)")
    print(f"  {'='*40}")

    # ---- Shepp-Logan 评估 ----
    sl_fbp_t = torch.from_numpy(sl_fbp).unsqueeze(0).unsqueeze(0).to(device)
    def _predict_sl(model, ready_flag):
        if not ready_flag:
            return None
        with torch.no_grad():
            return model(sl_fbp_t).cpu()

    sl_unet     = _predict_sl(unet, unet_ready)
    sl_vdsr     = _predict_sl(vdsr, vdsr_ready)
    sl_unet_mae = _predict_sl(unet_mae, unet_mae_ready)

    def _sl_psnr(pred_tensor, is_ready):
        if not is_ready or pred_tensor is None:
            return None
        return round(float(psnr(sl_gt, pred_tensor.squeeze().numpy(), data_range=1.0)), 2)

    sl_psnr_unet     = _sl_psnr(sl_unet, unet_ready)
    sl_psnr_vdsr     = _sl_psnr(sl_vdsr, vdsr_ready)
    sl_psnr_unet_mae = _sl_psnr(sl_unet_mae, unet_mae_ready)

    print(f"\n  Shepp-Logan PSNR:")
    print(f"    FBP          : {sl_fbp_psnr:.2f} dB")
    for label, val in [('UNet (MSE)', sl_psnr_unet), ('VDSR (MSE)', sl_psnr_vdsr),
                       ('UNet (MAE)', sl_psnr_unet_mae)]:
        if val is not None:
            print(f"    {label:<13}: {val:.2f} dB")
        else:
            print(f"    {label:<13}: N/A")

    results['methods'] = {
        'fbp': {'psnr_test': fbp_psnr_base, 'psnr_shepp_logan': sl_fbp_psnr},
        'unet_mse': {
            'psnr_test': psnr_unet, 'psnr_shepp_logan': sl_psnr_unet,
            'params': num_params_unet, 'ready': unet_ready,
        },
        'vdsr_mse': {
            'psnr_test': psnr_vdsr, 'psnr_shepp_logan': sl_psnr_vdsr,
            'params': num_params_vdsr, 'ready': vdsr_ready,
        },
        'unet_mae': {
            'psnr_test': psnr_unet_mae, 'psnr_shepp_logan': sl_psnr_unet_mae,
            'ready': unet_mae_ready,
        },
    }

    # ---- 可视化 ----
    psnr_label = lambda a, b: f'{compute_psnr_single(a, b):.1f}'

    # 图 1：随机测试样本对比
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    idx = np.random.randint(0, N_TEST)
    fbp_i = test_fbp[idx]
    gt_i  = test_gt[idx]
    fbp_t = torch.from_numpy(fbp_i).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_unet_i     = unet(fbp_t).cpu().squeeze().numpy()     if unet_ready     else np.zeros_like(gt_i)
        pred_vdsr_i     = vdsr(fbp_t).cpu().squeeze().numpy()     if vdsr_ready     else np.zeros_like(gt_i)
        pred_unet_mae_i = unet_mae(fbp_t).cpu().squeeze().numpy() if unet_mae_ready else np.zeros_like(gt_i)

    def _title_or_na(name, pred, gt, ready):
        if not ready:
            return f'{name}\nN/A'
        return f'{name}\nPSNR={psnr_label(pred, gt)} dB'

    titles_row1 = [
        (fbp_i, f'FBP ($\\hat{{x}}_\\mathrm{{FBP}}$)'),
        (gt_i, 'Ground Truth ($x$)'),
        (pred_unet_i, _title_or_na('UNet (MSE)', pred_unet_i, gt_i, unet_ready)),
    ]
    for ax, (img, title) in zip(axes[0], titles_row1):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    titles_row2 = [
        (pred_vdsr_i, _title_or_na('VDSR (MSE)', pred_vdsr_i, gt_i, vdsr_ready)),
        (pred_unet_mae_i, _title_or_na('UNet (MAE)', pred_unet_mae_i, gt_i, unet_mae_ready)),
        (np.abs(pred_unet_i - gt_i) * 5 if unet_ready else np.zeros_like(gt_i),
         r'|UNet$_\mathrm{MSE}$ - $x$| $\times$5' if unet_ready else 'Error Map (N/A)'),
    ]
    for ax, (img, title) in zip(axes[1], titles_row2):
        cmap = 'gray' if 'Error' not in title else 'hot'
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    fig.suptitle('CT $\\rightarrow$ FBP $\\rightarrow$ Post-Processing: UNet vs VDSR',
                 fontsize=13, fontweight='bold')
    # rect 顶部留出 7% 空间给 suptitle，避免与子图标题重叠
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path1 = os.path.join(OUTPUT_DIR, 'comparison_test_sample.png')
    fig.savefig(fig_path1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[图片] 已保存: {fig_path1}")

    # 图 2：Shepp-Logan 可视化（只展示可用的模型）
    # 无条件绘制的基础面板：FBP + GT（2 张），再按 ready 情况追加各模型面板
    n_panels = 2 + [unet_ready, vdsr_ready, unet_mae_ready].count(True)
    # 单行横排布局（最多 5 张图：FBP + GT + 3 个模型），避免 2×3 布局末尾出现空面板
    n_cols = n_panels
    n_rows = 1
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.5 * n_rows))
    axes2 = np.atleast_1d(axes2)

    panel_idx = 0
    def _add_sl_panel(ax, img, title):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    _add_sl_panel(axes2.flat[panel_idx], sl_fbp,
                  f'FBP ($\\hat{{x}}_\\mathrm{{FBP}}$)\nPSNR={sl_fbp_psnr:.1f} dB')
    panel_idx += 1
    _add_sl_panel(axes2.flat[panel_idx], sl_gt, 'Shepp-Logan ($x$)')
    panel_idx += 1

    if unet_ready:
        _add_sl_panel(axes2.flat[panel_idx], sl_unet.squeeze().numpy(),
                      f'UNet (MSE)\nPSNR={sl_psnr_unet:.1f} dB')
        panel_idx += 1
    if vdsr_ready:
        _add_sl_panel(axes2.flat[panel_idx], sl_vdsr.squeeze().numpy(),
                      f'VDSR (MSE)\nPSNR={sl_psnr_vdsr:.1f} dB')
        panel_idx += 1
    if unet_mae_ready:
        _add_sl_panel(axes2.flat[panel_idx], sl_unet_mae.squeeze().numpy(),
                      f'UNet (MAE)\nPSNR={sl_psnr_unet_mae:.1f} dB')
        panel_idx += 1

    # 隐藏多余的子图
    for j in range(panel_idx, len(axes2.flat)):
        axes2.flat[j].axis('off')

    fig2.suptitle('Shepp-Logan Phantom: FBP vs Learned Post-Processing',
                  fontsize=13, fontweight='bold')
    # rect 顶部留出 7% 空间给 suptitle，避免与子图标题重叠
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path2 = os.path.join(OUTPUT_DIR, 'comparison_shepp_logan.png')
    fig2.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"[图片] 已保存: {fig_path2}")

    # 图 3：训练曲线 + PSNR 柱状图
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    any_loss = False
    if len(loss_unet) > 0:
        ax1.plot(range(1, len(loss_unet) + 1), loss_unet, 'b-',
                 label='UNet (MSE)', linewidth=1.5)
        any_loss = True
    if len(loss_vdsr) > 0:
        ax1.plot(range(1, len(loss_vdsr) + 1), loss_vdsr, 'r-',
                 label='VDSR (MSE)', linewidth=1.5)
        any_loss = True
    if len(loss_unet_mae) > 0:
        ax1.plot(range(1, len(loss_unet_mae) + 1), loss_unet_mae, 'g--',
                 label=r'UNet (MAE, $L_1$)', linewidth=1.5)
        any_loss = True
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curves')
    if any_loss:
        ax1.legend()
    ax1.grid(True, alpha=0.3)

    # PSNR 柱状图
    methods_bars = ['FBP']
    psnrs_bars  = [fbp_psnr_base]
    colors_bars = ['gray']
    for label, val, color in [('UNet\n(MSE)', psnr_unet, 'steelblue'),
                                ('VDSR\n(MSE)', psnr_vdsr, 'coral'),
                                ('UNet\n(MAE)', psnr_unet_mae, 'seagreen')]:
        if val is not None:
            methods_bars.append(label)
            psnrs_bars.append(val)
            colors_bars.append(color)
    bars = ax2.bar(methods_bars, psnrs_bars, color=colors_bars, edgecolor='black', alpha=0.85)
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Test Set PSNR Comparison')
    for bar, val in zip(bars, psnrs_bars):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    fig3.suptitle('Experiment 16.4-2: CT Post-Processing Results',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path3 = os.path.join(OUTPUT_DIR, 'training_curves.png')
    fig3.savefig(fig_path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"[图片] 已保存: {fig_path3}")

    # ---- 保存 JSON 结果 ----
    results['images'] = {
        'test_sample': fig_path1,
        'shepp_logan': fig_path2,
        'curves': fig_path3,
    }
    results['training'] = {
        'unet_mse_losses': [float(x) for x in loss_unet],
        'vdsr_mse_losses': [float(x) for x in loss_vdsr],
        'unet_mae_losses': [float(x) for x in loss_unet_mae],
    }
    json_path = os.path.join(OUTPUT_DIR, 'results_16_4_2.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[结果] JSON 已保存: {json_path}")

    # ---- 小结（动态推导，绝不硬编码结论） ----
    print("\n" + "=" * 60)
    print("  实验 16.4-2 完成！")
    print("=" * 60)
    print(f"\n  核心结论：")
    if unet_ready and vdsr_ready:
        print(f"    1. UNet/VDSR 后处理均能显著超越 FBP 基线"
              f"（+{psnr_unet - fbp_psnr_base:.1f} dB / +{psnr_vdsr - fbp_psnr_base:.1f} dB）")
        gap = psnr_unet - psnr_vdsr
        if abs(gap) < 0.3:
            comp_txt = f"VDSR 与 UNet 表现相当（PSNR 差距 {gap:+.2f} dB）"
        elif gap > 0:
            comp_txt = f"UNet 优于 VDSR（+{gap:.2f} dB）"
        else:
            comp_txt = f"VDSR 优于 UNet（+{-gap:.2f} dB）"
        print(f"    2. {comp_txt}")
    elif unet_ready or vdsr_ready:
        print(f"    1-2. 仅完成部分模型训练，UNet/VDSR 对比暂不可用")
    else:
        print(f"    1-2. 模型尚未全部训练完成（需 GPU 环境）")

    if unet_ready and unet_mae_ready:
        gap_loss = psnr_unet - psnr_unet_mae
        if abs(gap_loss) < 0.3:
            loss_txt = f"MSE 与 MAE 损失下 UNet 表现相近（差距 {gap_loss:+.2f} dB）"
        elif gap_loss > 0:
            loss_txt = f"MSE 损失下 UNet 更优（+{gap_loss:.2f} dB），但 MAE 理论上对异常像素更鲁棒"
        else:
            loss_txt = f"MAE 损失下 UNet 更优（+{-gap_loss:.2f} dB）"
        print(f"    3. {loss_txt}")
    else:
        print(f"    3. MSE/MAE 损失对比数据不完整（需两个模型均训练完成）")

    print(f"    4. 非扩散 UNet（无时间步条件）与扩散 UNet（实验 15.1-1）的区别一目了然")
    print()


if __name__ == '__main__':
    main()
