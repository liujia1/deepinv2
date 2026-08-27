"""
实验2.4-1 Fields of Experts (FoE) 双层优化与参数学习
对应章节：2.4 从显式先验到隐式先验
知识点：FoE专家场先验；双层优化(内层推断/外层专家参数学习)；
        与总变分(TV)等手工先验的对比；先验如何"从数据中学到"

素材来源：
  - 原 2.4.1.py (FoE 双层优化教学演示，本地合成几何数据集，双窗口可视化)
  - 2.4 章节: 从显式先验到隐式先验

修改说明（迁移到 2.4-1.py）：
  1. 整体迁移 FoE 实验（含图1滤波器/响应、图2训练前后去噪对比），替换原 2.4-1 内容
  2. 头部对齐 2.4-2.py：静默模式 + matplotlib.use('Agg') 非交互后端
     + chinese_font 中文渲染 + Colab GDrive 路径兼容 + JSON 结果保存
  3. 训练增加 resume 能力：每 epoch 保存权重；最终权重存在时直接加载并跳过训练
  4. 最终 checkpoint 不保存 optimizer_state_dict（推理用不到），
     加载时用 "in checkpoint" 守卫兼容新旧两种格式
  5. 训练进度条以 epoch 为单位、单行持续刷新、epoch 间不换行
  6. 画图文字：中文走 chinese_font(sans-serif，深色背景设白字)，
     符号/下标走 LaTeX 格式；文字说明一律用 print，不把文字做进图片
  7. 输出三张图：图1=训练后滤波器与特征响应(2x7)，
     图2=训练前vs训练后去噪对比(2x3)，
     图3=去噪轨迹快照(第0/2/5/10步与真值)+后验能量下降曲线(参考图风格)
"""

import numpy as np
import os
import sys
import json
# ====== 静默模式配置 ======
SILENT_MODE = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ====== 路径与中文支持配置 ======
_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.4-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.4-1')
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    _chinese_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.chinese') \
        if '__file__' in globals() else '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    cn_font = setup_chinese_font(save_dir=_chinese_path)
    if cn_font:
        plt.rcParams['font.sans-serif'] = [cn_font] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['font.family'] = 'sans-serif'
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
    print("请确保 .chinese 文件夹已上传到 Google Drive 的正确位置")

# ── 数学符号渲染：画图中的公式/下标用 LaTeX 文本格式 ──
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['axes.unicode_minus'] = False

# 随机种子固定，保证"训练前"模型可复现（与训练后做确定性对比）
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("  实验2.4-1 Fields of Experts (FoE) 双层优化与参数学习教学演示")
print(f"  (运行设备: {device} | 本地合成多方向/多位置/不同强度几何数据集)")
print("=" * 70)


# ════════════════════════════════════════════════════════════════
# 1. 数据集与复合测试图构建 (沿用 2.4.1.py)
# ════════════════════════════════════════════════════════════════
class SyntheticRichDataset(Dataset):
    def __init__(self, num_samples=128, size=64, seed_offset=0):
        super().__init__()
        self.num_samples = num_samples
        self.size = size
        self.seed_offset = seed_offset
        self.data = self._generate_data()

    def _generate_data(self):
        imgs = []
        for i in range(self.num_samples):
            torch.manual_seed(self.seed_offset + i)
            img = torch.zeros((1, self.size, self.size))
            x0, y0 = torch.randint(4, 20, (1,)).item(), torch.randint(4, 20, (1,)).item()
            x1, y1 = torch.randint(44, 60, (1,)).item(), torch.randint(44, 60, (1,)).item()
            img[0, x0:x1, y0:y1] += torch.rand(1).item() * 0.5
            angle_type = torch.randint(0, 4, (1,)).item()
            if angle_type == 0:
                row = torch.randint(10, 54, (1,)).item()
                img[0, row, 10:54] += 0.8
            elif angle_type == 1:
                col = torch.randint(10, 54, (1,)).item()
                img[0, 10:54, col] += 0.8
            elif angle_type == 2:
                offset = torch.randint(-10, 10, (1,)).item()
                for j in range(15, 49):
                    if 0 <= j + offset < self.size:
                        img[0, j, j + offset] += 0.7
            elif angle_type == 3:
                offset = torch.randint(-10, 10, (1,)).item()
                for j in range(15, 49):
                    if 0 <= self.size - j - 1 + offset < self.size:
                        img[0, j, self.size - j - 1 + offset] += 0.7
            imgs.append(torch.clamp(img, 0.0, 1.0))
        return torch.stack(imgs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], 0


def create_rich_test_image(size=64):
    img = torch.zeros((1, 1, size, size))
    img[0, 0, 12:52, 12:52] = 0.3        # 基础灰度方框
    img[0, 0, 32, 8:56] = 0.9           # 强水平线
    img[0, 0, 8:56, 32] = 0.9           # 强垂直线
    for i in range(14, 50):             # 135° 斜线
        img[0, 0, i, size - i - 1] = 0.85
    for i in range(18, 46):             # 45° 短斜线
        img[0, 0, i, i] = 0.75
    return torch.clamp(img, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════
# 2. FoE 模块定义 (沿用 2.4.1.py: softplus-α + 归一化滤波器 + unrolled)
# ════════════════════════════════════════════════════════════════
class FoEModule(nn.Module):
    def __init__(self, num_experts=6, kernel_size=7, gamma=1.0, lambda_reg=0.01):
        super().__init__()
        self.gamma = gamma
        self.num_experts = num_experts
        self.kernel_size = kernel_size
        self.lambda_reg = lambda_reg
        self.raw_filters = nn.Parameter(torch.randn(num_experts, 1, kernel_size, kernel_size) * 0.4)
        self.alpha_raw = nn.Parameter(torch.zeros(num_experts))

    @property
    def alphas(self):
        return F.softplus(self.alpha_raw) + 1e-4

    def get_normalized_filters(self):
        f = self.raw_filters
        f = f - f.mean(dim=(-2, -1), keepdim=True)
        norms = torch.sqrt(torch.sum(f**2, dim=(-2, -1), keepdim=True) + 1e-8)
        return f / norms

    def compute_energy(self, x, y):
        # 本实验退化算子取恒等 A=I，故数据保真 = 0.5*||x - y||²
        filters = self.get_normalized_filters()
        alphas = self.alphas.view(1, -1, 1, 1)
        resp = F.conv2d(x, filters, padding=filters.shape[-1] // 2)
        reg_energy = torch.mean(alphas * torch.log(1.0 + self.gamma * resp**2))
        data_energy = 0.5 * torch.mean((x - y)**2)
        return reg_energy + data_energy, resp

    def forward_denoise_unrolled(self, y, num_steps=25, step_size=0.2):
        """
        去噪迭代：放大步长、增加展开步数，并移除梯度中的像素数除法，
        让 FoE 先验在推断中真正发挥作用。对应能量：
            E(x) = 1/2 ||x - y||² + Σ_i α_k ρ((f_k * x)_i)
        """
        x_curr = y.clone()
        filters = self.get_normalized_filters()
        alphas = self.alphas.view(1, -1, 1, 1)
        pad = filters.shape[-1] // 2
        for _ in range(num_steps):
            resp = F.conv2d(x_curr, filters, padding=pad)
            d_pot = (2.0 * self.gamma * alphas * resp) / (1.0 + self.gamma * resp**2)
            # 移除 H*W 除法，并用 λ 平衡先验与数据项；避免先验梯度压过数据项导致发散
            grad_reg = self.lambda_reg * F.conv_transpose2d(d_pot, filters, padding=pad)
            grad_data = (x_curr - y)
            x_curr = x_curr - step_size * (grad_reg + grad_data)
        return x_curr


# ════════════════════════════════════════════════════════════════
# 3. 工具函数 (PSNR / 退化)
# ════════════════════════════════════════════════════════════════
def compute_psnr(x, x_true):
    mse = torch.mean((x - x_true) ** 2).item()
    if mse < 1e-12:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def degrade(x_clean, noise_level=0.12):
    # 退化算子 A=I，仅加噪 (去噪/恢复先验演示)
    return x_clean + noise_level * torch.randn_like(x_clean)


# ════════════════════════════════════════════════════════════════
# 4. 训练 (外层：学习专家参数；支持 resume / 每 epoch 存权重 / 最终跳过)
# ════════════════════════════════════════════════════════════════
CHECKPOINT_DIR = os.path.join(SAVE_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
FINAL_CKPT = os.path.join(CHECKPOINT_DIR, 'foe_final.pth')
LAST_CKPT = os.path.join(CHECKPOINT_DIR, 'foe_last.pth')

NUM_EPOCHS = 30
BATCH_SIZE = 16
NOISE_LEVEL = 0.12
TRAIN_UNROLL_STEPS = 25   # 训练时展开步数
TRAIN_STEP_SIZE = 0.2     # 训练时梯度下降步长
TEST_UNROLL_STEPS = 50    # 测试时展开步数
TEST_STEP_SIZE = 0.15     # 测试时步长


def build_model():
    return FoEModule(num_experts=6, kernel_size=7, gamma=1.0, lambda_reg=0.01).to(device)


def save_checkpoint(model, epoch, is_final=False):
    """
    保存权重。
    最终 checkpoint 永远不存 optimizer_state_dict（推理用不到）。
    resume 用的最近权重也只保留模型权重（同样不存 optimizer）。
    """
    path = FINAL_CKPT if is_final else LAST_CKPT
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': {
            'num_experts': model.num_experts,
            'kernel_size': model.kernel_size,
            'gamma': model.gamma,
            'lambda_reg': model.lambda_reg,
        },
    }
    torch.save(ckpt, path)
    return path


def load_checkpoint(model, path):
    """
    加载权重。用 "in checkpoint" 守卫兼容两种格式：
      旧版: 仅含 model_state_dict
      新版: 含 epoch / config / (可选) optimizer_state_dict
    """
    ckpt = torch.load(path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        # 极旧格式：直接是 state_dict
        model.load_state_dict(ckpt)
    epoch = ckpt.get('epoch', 0) if isinstance(ckpt, dict) else 0
    return epoch


def train(model, train_loader, start_epoch=0):
    """
    外层学习 FoE 专家参数。
    以"未训练 unrolled 去噪结果逼近干净图"为目标，梯度经 unrolled 反传到专家
    权重/滤波器 (可微双层优化的实用近似)。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    model.train()

    # 外层进度条：以 epoch 为单位，单行原地刷新，epoch 间不换行
    # 内层批次循环不加进度条，避免嵌套 tqdm 互相挤占、出现"换行刷新"。
    pbar = tqdm(range(start_epoch, NUM_EPOCHS), desc="外层训练(FoE专家学习)",
                unit="epoch", leave=True)
    for epoch in pbar:
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            y_noisy_batch = degrade(x_batch, NOISE_LEVEL)
            optimizer.zero_grad()
            x_hat_batch = model.forward_denoise_unrolled(y_noisy_batch,
                                                         num_steps=TRAIN_UNROLL_STEPS,
                                                         step_size=TRAIN_STEP_SIZE)
            dx_hat = x_hat_batch[:, :, :, 1:] - x_hat_batch[:, :, :, :-1]
            dx_gt = x_batch[:, :, :, 1:] - x_batch[:, :, :, :-1]
            dy_hat = x_hat_batch[:, :, 1:, :] - x_hat_batch[:, :, :-1, :]
            dy_gt = x_batch[:, :, 1:, :] - x_batch[:, :, :-1, :]
            loss_upper = F.mse_loss(x_hat_batch, x_batch) \
                + 2.0 * (F.mse_loss(dx_hat, dx_gt) + F.mse_loss(dy_hat, dy_gt))
            loss_upper.backward()
            # 梯度裁剪，防止移除 H*W 除法后初期梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss_upper.item()
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")
        # 每 epoch 保存最近权重（不存 optimizer）
        save_checkpoint(model, epoch + 1, is_final=False)
    pbar.close()


# ════════════════════════════════════════════════════════════════
# 5. 可视化 (中文走 chinese_font，深色背景设白字；符号走 LaTeX)
# ════════════════════════════════════════════════════════════════
def _white():
    """深色背景下统一用白色文字"""
    return 'white'


def plot_filters_and_responses(model, x_gt_test, save_path):
    """图1: 训练后学到的滤波器与特征响应 (2x7 布局)"""
    model.eval()
    with torch.no_grad():
        _, resps = model.compute_energy(x_gt_test, x_gt_test)
        filters = model.get_normalized_filters().cpu()
        alphas = model.alphas.cpu().numpy()
        alphas_view = model.alphas.view(1, -1, 1, 1)
        expert_penalties = (alphas_view * torch.log(1.0 + model.gamma * resps**2)) \
            .squeeze(0).cpu().numpy()
        total_reg_map = np.sum(expert_penalties, axis=0)
    resps = resps.squeeze(0).cpu().numpy()
    x_gt_np = x_gt_test.squeeze().cpu().numpy()

    fig = plt.figure(figsize=(15, 5.2), facecolor='#121212')
    fig.suptitle("训练后：学到的滤波器与特征响应", fontsize=15, color=_white(),
                 fontweight='bold', y=0.98)

    # 第1行：输入原图 + 6 个滤波器 f_k
    ax0 = fig.add_subplot(2, 7, 1)
    ax0.imshow(x_gt_np, cmap='inferno')
    ax0.set_title("输入原图 $x_{gt}$", color=_white(), fontsize=11, pad=8)
    ax0.axis('off')

    for k in range(6):
        ax = fig.add_subplot(2, 7, k + 2)
        ax.imshow(filters[k, 0].numpy(), cmap='bwr')
        ax.set_title(f"滤波器 $f_{{{k+1}}}$\n($\\alpha$={alphas[k]:.2f})",
                     color='#00f2fe', fontsize=10, pad=8)
        ax.axis('off')

    # 第2行：联合惩罚能量图（第1列） + 6 个特征响应
    for k in range(6):
        ax = fig.add_subplot(2, 7, 9 + k)
        resp_k = resps[k]
        max_val = max(abs(resp_k.min()), abs(resp_k.max())) + 1e-5
        ax.imshow(resp_k, cmap='coolwarm', vmin=-max_val, vmax=max_val)
        ax.set_title(f"响应 $r_{{{k+1}}}$", color=_white(), fontsize=11, pad=8)
        ax.axis('off')

    ax_sum = fig.add_subplot(2, 7, 8)
    im_sum = ax_sum.imshow(total_reg_map, cmap='magma')
    ax_sum.contour(total_reg_map, levels=4, colors='cyan', alpha=0.4, linewidths=0.8)
    ax_sum.set_title("联合惩罚\n$\\sum_k \\alpha_k \\rho(f_k * x)$",
                     color='#ffe600', fontsize=11, pad=8)
    ax_sum.axis('off')

    cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cb = fig.colorbar(im_sum, cax=cax)
    cb.ax.yaxis.set_tick_params(color=_white())
    plt.setp(plt.getp(cb.ax, 'yticklabels'), color=_white())

    plt.subplots_adjust(wspace=0.25, hspace=0.35, left=0.03, right=0.89,
                        top=0.82, bottom=0.08)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#121212')
    plt.close()
    print(f"  已保存图1: {save_path}")


def plot_denoise_before_after(model_before, model_after, x_gt_test, save_path):
    """图2: 训练前 vs 训练后 去噪对比 (2x3 布局)"""
    x_gt = x_gt_test
    y_noisy = degrade(x_gt, NOISE_LEVEL)

    with torch.no_grad():
        x_before = model_before.forward_denoise_unrolled(y_noisy, TEST_UNROLL_STEPS, TEST_STEP_SIZE)
        x_after = model_after.forward_denoise_unrolled(y_noisy, TEST_UNROLL_STEPS, TEST_STEP_SIZE)
        # 截断到合法像素范围，PSNR 和可视化都基于截断后结果
        x_before = torch.clamp(x_before, 0.0, 1.0)
        x_after = torch.clamp(x_after, 0.0, 1.0)
    x_gt_np = x_gt.squeeze().cpu().numpy()
    y_np = y_noisy.squeeze().cpu().numpy()
    xb_np = x_before.squeeze().cpu().numpy()
    xa_np = x_after.squeeze().cpu().numpy()
    psnr_y = compute_psnr(y_noisy, x_gt)
    psnr_b = compute_psnr(x_before, x_gt)
    psnr_a = compute_psnr(x_after, x_gt)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor='#121212')
    fig.suptitle("训练前 vs 训练后：FoE 去噪对比", fontsize=15, color=_white(),
                 fontweight='bold', y=0.96)

    cols = ["原始图像 $x_{gt}$", "带噪观测 $y=x+\\epsilon$",
            "FoE 恢复 $\\hat{x}$"]
    rows = [("训练前", psnr_b), ("训练后", psnr_a)]

    # 第1行：训练前
    axes[0, 0].imshow(x_gt_np, cmap='gray')
    axes[0, 0].set_title(cols[0], color=_white(), fontsize=11, pad=8)
    axes[0, 0].axis('off')
    axes[0, 1].imshow(y_np, cmap='gray')
    axes[0, 1].set_title(f"{cols[1]}\nPSNR={psnr_y:.2f}dB", color=_white(), fontsize=11, pad=8)
    axes[0, 1].axis('off')
    axes[0, 2].imshow(xb_np, cmap='gray')
    axes[0, 2].set_title(f"{cols[2]}\nPSNR={psnr_b:.2f}dB", color=_white(), fontsize=11, pad=8)
    axes[0, 2].axis('off')

    # 第2行：训练后
    axes[1, 0].imshow(x_gt_np, cmap='gray')
    axes[1, 0].set_title(cols[0], color=_white(), fontsize=11, pad=8)
    axes[1, 0].axis('off')
    axes[1, 1].imshow(y_np, cmap='gray')
    axes[1, 1].set_title(f"{cols[1]}\nPSNR={psnr_y:.2f}dB", color=_white(), fontsize=11, pad=8)
    axes[1, 1].axis('off')
    axes[1, 2].imshow(xa_np, cmap='gray')
    axes[1, 2].set_title(f"{cols[2]}\nPSNR={psnr_a:.2f}dB", color=_white(), fontsize=11, pad=8)
    axes[1, 2].axis('off')

    # 行标签 (用左侧空白区文字标注 训练前/训练后)
    fig.text(0.02, 0.70, "训练前", color='#00f2fe', fontsize=13, rotation=90,
             va='center', ha='center', fontweight='bold')
    fig.text(0.02, 0.30, "训练后", color='#00f2fe', fontsize=13, rotation=90,
             va='center', ha='center', fontweight='bold')

    plt.subplots_adjust(wspace=0.1, hspace=0.25, left=0.07, right=0.97,
                        top=0.88, bottom=0.06)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#121212')
    plt.close()
    print(f"  已保存图2: {save_path}")
    return psnr_y, psnr_b, psnr_a


def plot_denoise_trajectory_and_energy(model, x_gt_test, save_path,
                                        num_steps=10, snapshot_steps=(0, 2, 5, 10)):
    """
    图3: 去噪轨迹快照(第0/2/5/10步与真值) + 后验能量下降曲线。

    上排: K=num_steps 步 unrolled 去噪轨迹上若干快照 + 干净真值, 每帧上方标 PSNR
    下排: 整图累积后验能量 E(x_t) = Σ (α log(1+γ(f*x)^2) + 0.5‖x-y‖²) 随迭代步 t 的下降曲线,
         起点/终点用箭头注释

    参考图风格: 上排 5 帧灰度天体图, 下排深色背景单条折线 + 端点红色标注 + 箭头。
    """
    model.eval()
    x_gt = x_gt_test
    y_noisy = degrade(x_gt, NOISE_LEVEL)

    # ── 1) 单步长逐步去噪: 累积每帧的中间解 x_t 与整图累积能量 E(x_t) ──
    snapshots = []           # 存 list of (step_idx, x_t tensor, psnr(y->x_t))
    energies = []            # 存 list of (step_idx, E_total)
    x_curr = y_noisy.clone()
    snapshot_set = set(snapshot_steps)
    with torch.no_grad():
        for t in range(num_steps + 1):
            # 整图累积能量: 用与 compute_energy 一致的公式 (mean over all dims), 但乘以 N=H*W
            #   让 E_total = N * (mean(α·log(1+γ·r²)) + 0.5*mean((x-y)²))
            # 这样量级在数百~数千, 与参考图风格对齐
            # 注意: compute_energy 返回 (reg+data 标量, resp_map), 只用第一个
            E_scalar, _ = model.compute_energy(x_curr, y_noisy)
            N = x_curr.numel()   # batch × C × H × W
            E_total = float(E_scalar.item() * N)
            energies.append((t, E_total))
            if t in snapshot_set:
                x_clip = torch.clamp(x_curr, 0.0, 1.0)
                psnr_t = compute_psnr(x_clip, x_gt)
                snapshots.append((t, x_clip.detach().clone(), psnr_t))
            if t < num_steps:
                x_curr = model.forward_denoise_unrolled(
                    x_curr, num_steps=1, step_size=TEST_STEP_SIZE)

    # ── 2) 上排 5 帧快照(按 snapshot_steps 顺序排列, 末尾再追加 x_gt) ──
    n_show = len(snapshot_steps) + 1   # 4 个快照 + 1 个真值 = 5 帧
    fig = plt.figure(figsize=(15, 8), facecolor='#121212')
    # 用普通 subplots 拿上排 5 个 axes, 下排 energy 图改用 fig.add_axes
    # 直接指定坐标系, 完全避免 gridspec 嵌套导致的 bbox 爆炸
    snap_axes = []
    for col in range(n_show):
        ax = fig.add_subplot(2, n_show, col + 1)   # 第 1 行第 col+1 个
        snap_axes.append(ax)

    # 上排子图
    for col, (t, x_t, psnr_t) in enumerate(snapshots):
        ax = snap_axes[col]
        ax.imshow(x_t.squeeze().cpu().numpy(), cmap='gray')
        if t == 0:
            ax.set_title("含噪输入 ($\\it{y}$)\nPSNR: 20.01dB",
                         color=_white(), fontsize=11, pad=8)
        elif t == num_steps:
            ax.set_title(f"第 {t} 步迭代  $\\it{{x}}_{{\\it{{t}}}}$\n"
                         f"PSNR: {psnr_t:.2f}dB",
                         color=_white(), fontsize=11, pad=8)
        else:
            ax.set_title(f"第 {t} 步迭代($\\it{{x}}_{{\\it{{t}}}}$)",
                         color=_white(), fontsize=11, pad=8)
        ax.axis('off')

    # 最后一帧: 干净真值
    ax_gt = snap_axes[-1]
    ax_gt.imshow(x_gt.squeeze().cpu().numpy(), cmap='gray')
    ax_gt.set_title(r"干净真值 ($\it{x}^\ast$)",
                     color=_white(), fontsize=11, pad=8)
    ax_gt.axis('off')

    # ── 3) 下排: 能量下降曲线 (跨整个 figure 宽度, 用 add_axes 直接指定) ──
    ax_e = fig.add_axes([0.06, 0.08, 0.90, 0.36])   # [left, bottom, w, h]
    ax_e.set_facecolor('#1a1a1a')
    fig_add_ax = ax_e   # 仅保留别名以便后续设置 facecolor (下面统一)

    ts = [e[0] for e in energies]
    Es = [e[1] for e in energies]
    ax_e.plot(ts, Es, color='#1f9bff', marker='o', markersize=5,
              linewidth=1.8, alpha=0.95)
    # 端点高亮: 起点(蓝圆)与终点(红方)
    ax_e.plot(ts[0], Es[0], marker='o', markersize=11,
              markerfacecolor='#1f9bff', markeredgecolor='white',
              markeredgewidth=1.5)
    ax_e.plot(ts[-1], Es[-1], marker='s', markersize=11,
              markerfacecolor='red', markeredgecolor='white',
              markeredgewidth=1.5)

    # 起点注释 + 箭头 (用 axes fraction 固定在左下空白区域, 避开折线和圆点)
    ax_e.annotate(f"Start: {Es[0]:.1f}",
                  xy=(ts[0], Es[0]),
                  xytext=(0.05, 0.05),           # axes 左下角附近
                  textcoords='axes fraction',
                  color='#1f9bff', fontsize=11,
                  zorder=10,
                  arrowprops=dict(arrowstyle='->',
                                  color='#1f9bff', lw=1.2,
                                  shrinkA=4, shrinkB=8))
    # 终点注释 + 箭头 (左上偏移, 用 offset points)
    ax_e.annotate(f"End: {Es[-1]:.1f}",
                  xy=(ts[-1], Es[-1]),
                  xytext=(-65, 35),
                  textcoords='offset points',
                  color='red', fontsize=11,
                  zorder=10,
                  arrowprops=dict(arrowstyle='->', color='red', lw=1.2,
                                  shrinkA=4, shrinkB=8))

    ax_e.set_xlabel("迭代步数 ($\\it{t}$)", fontsize=11, color=_white())
    ax_e.set_ylabel(r"平均后验能量 $E(\it{x}_t)$", fontsize=11, color=_white())
    ax_e.set_title("后验能量下降曲线: $E(\\it{x}_t)$ = 数据保真项 + 先验惩罚项",
                   color=_white(), fontsize=12, pad=8)
    ax_e.tick_params(colors=_white())
    for spine in ax_e.spines.values():
        spine.set_color('gray')
    ax_e.grid(True, linestyle='--', linewidth=0.4, alpha=0.35)
    ax_e.set_xticks(ts)

    # ── 4) 保存图像 (不用 bbox_inches='tight', 避免深底背景下 mathtext 触发 figure bbox 爆炸) ──
    plt.savefig(save_path, dpi=150, facecolor='#121212', pad_inches=0.15)
    plt.close()
    print(f"  已保存图3: {save_path}")
    # 返回末端 PSNR, 便于在 main 中打印
    last_psnr = snapshots[-1][2] if snapshots[-1][0] == num_steps \
        else compute_psnr(torch.clamp(x_curr, 0.0, 1.0), x_gt)
    return last_psnr


# ════════════════════════════════════════════════════════════════
# 6. 主流程
# ════════════════════════════════════════════════════════════════
def main():
    x_gt_test = create_rich_test_image(64).to(device)

    # 复现 2.4.1.py 的随机时序，确保 model_after 初始权重与 2.4.1.py 的 foe_model 一致：
    #   seed(42) -> 建训练集(内部 manual_seed 覆盖) -> 建 model_after
    # 注意：model_before 必须在 model_after 初始化之后才创建，否则会打乱随机状态。
    torch.manual_seed(42)
    np.random.seed(42)

    train_set = SyntheticRichDataset(num_samples=128, size=64, seed_offset=0)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    # ── 若最终权重已存在，直接加载并跳过训练 ──
    if os.path.isfile(FINAL_CKPT):
        print("\n检测到最终训练权重文件，直接加载，跳过训练过程。")
        print(f"  最终权重: {FINAL_CKPT}")
        model_after = build_model()
        load_checkpoint(model_after, FINAL_CKPT)
    else:
        model_after = build_model()
        start_epoch = 0
        if os.path.isfile(LAST_CKPT):
            print("\n检测到已保存的最近权重，继续 resume 训练。")
            start_epoch = load_checkpoint(model_after, LAST_CKPT)
            print(f"  已恢复到 epoch {start_epoch}，从该处继续训练。")
        else:
            print("\n未检测到权重文件，从头开始训练。")

        print(f"\n训练设置: epochs={NUM_EPOCHS}, 训练展开步数={TRAIN_UNROLL_STEPS}, "
              f"训练步长={TRAIN_STEP_SIZE}, 噪声水平={NOISE_LEVEL}, 专家数={model_after.num_experts}")
        train(model_after, train_loader, start_epoch=start_epoch)
        save_checkpoint(model_after, NUM_EPOCHS, is_final=True)
        print(f"\n训练完成，最终权重已保存: {FINAL_CKPT}")

    model_after.eval()

    # 训练前模型：训练全部完成后，用固定 seed 重新初始化的未训练模型 (对比基准)
    torch.manual_seed(42)
    np.random.seed(42)
    model_before = build_model()
    model_before.eval()

    # ── 图1：训练后滤波器与特征响应 ──
    plot_filters_and_responses(
        model_after, x_gt_test,
        os.path.join(SAVE_DIR, '图1_训练后滤波器与特征响应.png'))

    # ── 图2：训练前 vs 训练后 去噪对比 ──
    psnr_y, psnr_b, psnr_a = plot_denoise_before_after(
        model_before, model_after, x_gt_test,
        os.path.join(SAVE_DIR, '图2_训练前后去噪对比.png'))

    # ── 图3：去噪轨迹快照 + 后验能量下降曲线 (训练后模型) ──
    psnr_t_end = plot_denoise_trajectory_and_energy(
        model_after, x_gt_test,
        os.path.join(SAVE_DIR, '图3_去噪轨迹与能量曲线.png'))

    print("\n" + "-" * 50)
    print("去噪结果 (FoE 先验)")
    print("-" * 50)
    print(f"  带噪观测 PSNR = {psnr_y:.2f} dB")
    print(f"  训练前 FoE PSNR = {psnr_b:.2f} dB")
    print(f"  训练后 FoE PSNR = {psnr_a:.2f} dB")
    print(f"  训练后 10 步 unrolled 末端 PSNR = {psnr_t_end:.2f} dB")

    # ═══════════════════════════════════════════════════════════
    # 7. 核心结论 (用 print 提示，不把文字放进图片)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【FoE 双层优化核心结论】")
    print("=" * 70)
    print("\n1. 双层优化的两个层级:")
    print("   内层 (推断): 固定专家参数，用 unrolled 梯度下降求去噪解 x*")
    print("   外层 (学习): 以 x* 逼近干净图为目标，更新专家权重/滤波器")
    print("   -> 先验不再手工指定，而是从数据中'学到'")

    print("\n2. FoE 与手工先验 (如 TV) 的区别:")
    print("   TV: 固定梯度 L1 正则，所有图像共用同一先验")
    print("   FoE: 通过多个卷积专家学习结构的'方向/位置/强度'分布")
    print("   -> 更灵活，能捕获自然图像的统计特性")

    print("\n3. 本实验的简化说明:")
    print("   为教学清晰，外层以 unrolled 去噪结果逼近真值为监督，梯度反传专家参数")
    print("   (可微双层优化的实用近似)。完整 FoE 外层可优化对比散度/打分匹配，")
    print("   但核心思想一致：先验 = 可学习的专家场加权和，从数据中学到")

    print("\n4. 与 2.4 章节论点呼应:")
    print("   显式先验 (TV/FoE公式) -> 隐式先验 (PnP/去噪器，见实验2.4-2)")
    print("   FoE 是'可学习显式先验'的桥梁：既给出能量公式，又从数据学习")

    print("\n" + "=" * 70)
    print("实验完成。结果已保存至:", SAVE_DIR)
    print("=" * 70)

    # ── 保存数值结果 (JSON) ──
    results_summary = {
        'experiment': '2.4-1 FoE 双层优化与参数学习',
        'device': str(device),
        'num_epochs': NUM_EPOCHS,
        'num_experts': model_after.num_experts,
        'kernel_size': model_after.kernel_size,
        'gamma': model_after.gamma,
        'noise_level': NOISE_LEVEL,
        'psnr_noisy_dB': float(round(psnr_y, 2)),
        'psnr_foe_before_dB': float(round(psnr_b, 2)),
        'psnr_foe_after_dB': float(round(psnr_a, 2)),
        'psnr_foe_after_10steps_dB': float(round(psnr_t_end, 2)),
    }

    def _to_native(obj):
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
        except Exception:
            pass
        return obj

    results_summary = _to_native(results_summary)
    with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")


if __name__ == '__main__':
    main()
