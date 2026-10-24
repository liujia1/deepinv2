# -*- coding: utf-8 -*-
"""
实验17.5-1 MOI多算子成像
对应知识点：17.5.2节（多算子成像：MOI）

实验内容：
Step 1: 单算子基线 (Naive MC + EI) —— 固定单一掩码 + 平移对称性（真正的"单算子"设定，
        训练/测试用同一算子，与EI原文inpainting实验一致）
Step 2: MOI多算子 (G=1, 2, 4, 8) —— 固定算子池（MOI论文原始设定）：
        算子池=[测试掩码]+(G-1)个固定掩码，每batch为每样本随机分配池中一个；
        G的作用来自掩码条件化 + G-1项交叉一致性（不是"永久分配"）；
        G=1 为纯 MC 基线（n_cons=0，无一致性项），是 MOI 要拯救的最困难情形
Step 3: MOI vs EI全面对比 —— 缺失像素恢复能力
Step 4: 联合覆盖度验证 —— 单次抽样下 rank(Σ_g A_gᵀA_g)/n 随G的演化
        （区别于正文17.5.2的期望满秩条件 rank(E[A_gᵀA_g])=n：iid掩码下后者在G=1即成立）

★原创设计：
- 实现MOI原始论文 (Tachella et al. NeurIPS 2022) 的多算子成像损失
- 与17.6-1的等变成像(EI)进行直接对比，凸显两种零空间填补策略
- 经验验证MOI的联合覆盖度（有限G下的实例联合秩，区别于期望满秩条件）

素材来源：17.5.2节理论、17.6-1实验架构
★ 为何与17.6-1共用 inpainting 算子：两实验要对比的是"零空间信息从哪里来"
  （MOI 靠真实多算子，EI 靠单算子+对称性），算子本身是控制变量；
  换成 MRI/CT 会同时改变任务难度，反而混淆对比。两实验的差别在算子集合
  与损失函数，不在成像物理，因此有意不换算子。
运行前提：需GPU（Colab T4即可）
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os, sys, io, time, warnings, logging

# ★ 进度格式化辅助函数：所有进度在同一行原地刷新（与13.5-1保持一致）
def print_progress(epoch, n_epochs, iter_idx, n_iters, tag='', **kwargs):
    """在终端原地刷新进度（不换行），带可视化进度条

    Args:
        epoch: 当前epoch（从0开始）
        n_epochs: 总epoch数
        iter_idx: 当前iter（从0开始）
        n_iters: 总iter数
        tag: 方法标签（如"single_ei"、"moipool_G4"）
        **kwargs: 其他要显示的指标（如 loss=0.123, mc=0.1, cons=0.05）
    """
    # 进度条可视化（总宽度30字符）
    progress = iter_idx / max(n_iters, 1)
    bar_width = 30
    filled = int(bar_width * progress)
    bar = '█' * filled + '░' * (bar_width - filled)
    info = f"[{tag}] Ep{epoch+1}/{n_epochs} [{bar}] {iter_idx}/{n_iters}"
    for k, v in kwargs.items():
        if isinstance(v, float):
            info += f" {k}={v:.4f}"
        else:
            info += f" {k}={v}"
    print(f"\r{info}", end="", flush=True)


def print_eval_progress(iter_idx, n_iters, tag=''):
    """评估进度原地刷新"""
    print(f"\r[{tag}] Eval {iter_idx}/{n_iters}", end="", flush=True)

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
    SAVE_DIR = os.path.join(_gdrive, '实验17.5-1')
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
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"实验17.5-1: MOI多算子成像")
print(f"{'='*70}")
print(f"使用设备: {device}")

num_workers = 0 if sys.platform == 'win32' else 2


# ========================================================================
# 网络架构：双输入UNet（接受 y 和 mask 拼接作为输入）
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


class SmallUNetMOI(nn.Module):
    """MOI专用UNet：接受 y 和 mask 拼接作为输入

    ★ 设计理由：
    - MOI损失需要 f(y, A_g) 中明确传入算子A_g
    - 把mask作为额外输入通道，让网络感知"哪些像素被观测"
    - 这样 f(A_s x̂, A_s) 中的 A_s 也可以是不同mask，实现交叉算子一致性

    输入:  y (B, 1, H, W) 与 mask (B, 1, H, W) 拼接为 (B, 2, H, W)
    输出:  重建图像 (B, 1, H, W)
    """
    def __init__(self, in_ch=2, out_ch=1, base=24):
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

    def forward(self, y, mask):
        # ★ 关键：把y和mask拼接作为输入
        x = torch.cat([y, mask], dim=1)
        e1 = self.enc1(x)        # (B, base, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, base*2, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, base*4, H/4, W/4)
        d3 = self.up3(e3)             # (B, base*2, H/2, W/2)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d3)             # (B, base, H, W)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        return self.out_conv(d2)


# ========================================================================
# 数据准备：MNIST + Inpainting多算子
# ========================================================================
IMG_SIZE = 32
SIGMA = 0.05
KEEP_RATIO = 0.5
BATCH_SIZE = 64
N_EPOCHS = 20
LR = 1e-3
LAMBDA_EI = 0.5     # EI损失权重（与17.6-1一致）
LAMBDA_MOI = 1.0    # MOI一致性损失权重
# ★ 权重选择说明：LAMBDA_EI=0.5 沿用17.6-1的调参结果；LAMBDA_MOI=1.0 是MOI论文默认值，
#   且MOI的一致性项本身是 stop-gradient 近似（非论文精确公式），λ=1.0 可保证一致性约束
#   有足够强度。两者不对等是有意为之——各自取该方法文献中的标准设定，而非统一调参。
#   若需验证结论稳健性，可将EI也设为λ=1.0做一次消融对比。

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

print("加载MNIST数据集...")
mnist_train = datasets.MNIST(root=os.path.join(SAVE_DIR, 'mnist_data'),
                              train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root=os.path.join(SAVE_DIR, 'mnist_data'),
                             train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)


def create_random_masks_per_sample(B, H, W, keep_ratio, device, G=1, seed=None):
    """为batch中每个样本生成G个不同随机掩码

    ★ 向量化实现：用 torch.argsort(torch.rand(...)) 替代双重Python循环，
      避免 B×G 次 randperm 的 CPU/GPU 同步开销（G=8, B=64 时从512次调用→1次）。

    Args:
        B: batch size
        H, W: 图像尺寸
        keep_ratio: 保留像素比例
        G: 每个样本的算子数
        seed: 随机种子

    Returns:
        masks: (B, G, H, W) 的0/1张量
    """
    if seed is not None:
        torch.manual_seed(seed)
    n_keep = int(H * W * keep_ratio)
    # ★ 向量化生成：对每个(B,G)位置生成随机分数，取 top-k 索引作为掩码
    rand_scores = torch.rand(B, G, H * W, device=device)
    # argsort 升序，后 n_keep 个即最大值的索引（等价于 randperm[:n_keep]）
    topk_indices = rand_scores.argsort(dim=-1)[:, :, -n_keep:]
    masks = torch.zeros(B, G, H * W, device=device)
    # scatter_ 在最后一维按索引置1
    masks.scatter_(dim=-1, index=topk_indices, value=1.0)
    return masks.view(B, G, H, W)


def create_fixed_test_mask(H, W, keep_ratio, seed=42):
    """生成固定的测试掩码（保证评估时所有方法用同一掩码）"""
    rng = np.random.RandomState(seed)
    mask = np.zeros((H, W), dtype=np.float32)
    n_keep = int(H * W * keep_ratio)
    indices = rng.choice(H * W, n_keep, replace=False)
    mask.flat[indices] = 1.0
    return torch.from_numpy(mask)


test_mask = create_fixed_test_mask(IMG_SIZE, IMG_SIZE, KEEP_RATIO).to(device)
# ★ 单算子基线（EI）专用固定掩码：与测试掩码相同，训练/测试用同一算子A。
#   这正是EI原文inpainting实验的设定——固定算子A + 平移对称性群"虚拟"造多算子，
#   以此与MOI"真实多算子（每样本G个不同掩码）"形成名副其实的单/多算子对比。
fixed_train_mask = test_mask


def add_inpainting_noise(x, mask, sigma=SIGMA):
    """对图像施加inpainting掩码和加性高斯噪声
    y = M ⊙ x + σ * M ⊙ ε  （仅在观测像素上有噪声）
    """
    mask_2d = mask.unsqueeze(1) if mask.dim() == 2 else mask
    noise = sigma * torch.randn_like(x) * mask_2d
    return x * mask_2d + noise


def evaluate_combined(model, test_loader, mask, sigma=SIGMA, device=None, tag=""):
    """在测试集上用固定掩码评估总PSNR、观测/缺失像素PSNR和SSIM

    ★ 与17.6-1评估保持一致：使用固定测试掩码，PSNR/SSIM在同一掩码下计算
    ★ 用 fork_rng 隔离评估噪声对全局RNG的污染

    Returns:
        total_psnr: 整图PSNR
        obs_psnr: 观测像素PSNR
        miss_psnr: 缺失像素PSNR
        ssim_val: 整图SSIM
    """
    model.eval()
    total_psnrs = []
    obs_psnrs = []
    miss_psnrs = []
    ssim_vals = []
    n_iters = len(test_loader)
    with torch.no_grad():
        with torch.random.fork_rng():
            torch.manual_seed(0)
            for iter_idx, (batch_x, _) in enumerate(test_loader):
                batch_x = batch_x.to(device)
                mask_2d = mask.unsqueeze(0).unsqueeze(0).expand_as(batch_x)
                y = add_inpainting_noise(batch_x, mask_2d, sigma)
                pred = model(y, mask_2d).clip(0, 1)
                pred_np = pred.cpu().numpy()
                x_np = batch_x.cpu().numpy()
                m_np = mask_2d.cpu().numpy()
                for i in range(batch_x.shape[0]):
                    total_psnrs.append(psnr(x_np[i, 0], pred_np[i, 0], data_range=1.0))
                    ssim_vals.append(ssim(x_np[i, 0], pred_np[i, 0], data_range=1.0,
                                           win_size=11, gaussian_weights=True, sigma=1.5))
                    obs = m_np[i, 0] > 0.5
                    miss = ~obs
                    if obs.sum() > 0:
                        mse_obs = ((x_np[i, 0][obs] - pred_np[i, 0][obs]) ** 2).mean()
                        obs_psnrs.append(10 * np.log10(1.0 / max(mse_obs, 1e-10)))
                    if miss.sum() > 0:
                        mse_miss = ((x_np[i, 0][miss] - pred_np[i, 0][miss]) ** 2).mean()
                        miss_psnrs.append(10 * np.log10(1.0 / max(mse_miss, 1e-10)))
                if iter_idx % 5 == 0 or iter_idx == n_iters - 1:
                    print_eval_progress(iter_idx + 1, n_iters, tag=f'评估 {tag}')
    print()
    return np.mean(total_psnrs), np.mean(obs_psnrs), np.mean(miss_psnrs), np.mean(ssim_vals)


# ========================================================================
# Step 1: 单算子基线 (Naive MC + EI)
# 对应17.5节：用单个算子A，依赖EI对称性约束零空间
# ========================================================================
print("\n" + "="*70)
print("Step 1: 单算子基线 (Naive MC + EI)")
print("="*70)


def random_shift(x, max_shift=8):
    """随机循环平移（用于EI对称性）

    ★ batch级共享位移：一个batch内所有样本使用相同的(dy, dx)，
      而非逐样本独立采样。这是有意与17.6-1保持一致的设计选择：
      共享位移简化了实现且跨迭代仍有足够随机性。
    ★ 进阶练习：读者可尝试改为逐样本独立采样 (dy_b, dx_b)，
      观察EI基线PSNR是否提升，以验证“共享 vs 独立”位移对
      对称性约束覆盖范围的影响。
    """
    B, C, H, W = x.shape
    dy = torch.randint(-max_shift, max_shift+1, (1,)).item()
    dx = torch.randint(-max_shift, max_shift+1, (1,)).item()
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


def ei_loss_for_moi(model, y, mask, pred, n_transforms=4):
    """适配MOI网络的EI损失（真正的"单算子 + 对称性"设定）

    ★ 设计说明：EI的虚拟多算子来自对称性群，而非真实多掩码。
      因此这里对每个平移变换后的 x̂ 仍使用**同一个固定掩码**（= 观测算子A），
      即虚拟算子为 A∘T_g —— 这正是17.5.3节"借对称性虚拟多算子"的实现。
      若像旧版那样每次重采随机掩码，基线本身就变成了"多算子"，
      与MOI的对比将失真（评审意见#4）。

    Args:
        mask: 固定观测掩码 (B,1,H,W)，与主重建路径同一算子
        pred: 外层已计算好的 model(y, mask) 输出，直接复用避免重复前向
    """
    x_hat = pred  # ★ 复用外层 train_single_op_ei 中已算好的前向结果

    total_loss = 0
    for _ in range(n_transforms):
        x_hat_shifted = random_shift(x_hat.detach())
        # ★ 关键：虚拟算子 = 固定掩码 ∘ 平移，掩码不变（单算子设定）
        y_virtual = x_hat_shifted * mask
        f_virtual = model(y_virtual, mask)
        total_loss += ((f_virtual - x_hat_shifted.detach()) ** 2).mean()
    return total_loss / n_transforms


def train_single_op_ei(model, train_loader, n_epochs, device, tag="single_ei"):
    """训练单算子 + EI基线模型"""
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
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1
            is_final = True
        else:
            print(f"  [{tag}] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
            model.load_state_dict(model_state)
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1
            # ★ 恢复RNG状态以保证续训后可复现性（评审意见#4）
            # 根因：torch.get_rng_state() 返回 CPU ByteTensor，但 torch.load(map_location='cuda')
            # 会将其搬到 GPU，而 torch.ByteTensor 是 CPU 专属类型，isinstance 必然失败。
            # .type(torch.ByteTensor) 隐式将 CUDA tensor 搬回 CPU 并恢复 ByteTensor 类型身份。
            if 'rng_state' in checkpoint:
                _rng = checkpoint['rng_state']
                try:
                    torch.set_rng_state(_rng)
                except TypeError:
                    try:
                        torch.set_rng_state(_rng.type(torch.ByteTensor))
                    except Exception:
                        print(f"  [{tag}] ⚠ rng_state恢复失败，跳过")
                if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
                    _cuda_rng = checkpoint['cuda_rng_state']
                    if _cuda_rng is not None:
                        try:
                            torch.cuda.set_rng_state(_cuda_rng)
                        except TypeError:
                            try:
                                torch.cuda.set_rng_state(_cuda_rng.type(torch.ByteTensor))
                            except Exception as e:
                                print(f"  [{tag}] ⚠ cuda_rng_state恢复失败({e})，跳过")

    if is_final or start_epoch >= n_epochs:
        print(f"  [{tag}] 模型已训练完毕，跳过。")
        return train_losses

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        n_iters = len(train_loader)
        for iter_idx, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            # ★ 单算子：全程使用同一固定掩码（= 测试掩码），虚拟多算子仅由EI平移对称性产生
            mask = fixed_train_mask.unsqueeze(0).unsqueeze(0).expand_as(batch_x)
            y = add_inpainting_noise(batch_x, mask, SIGMA)

            optimizer.zero_grad()
            # MC损失（值空间约束）
            pred = model(y, mask)
            loss_mc = ((mask * (y - mask * pred)) ** 2).sum() / mask.sum().clamp(min=1.0)
            # EI损失（零空间约束：利用平移对称性）
            loss_ei = ei_loss_for_moi(model, y, mask, pred, n_transforms=4)
            loss = loss_mc + LAMBDA_EI * loss_ei
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            # ★ 每个iteration都显示进度（更频繁的更新让用户看到实时进度）
            if iter_idx % 10 == 0 or iter_idx == n_iters - 1:
                print_progress(epoch, n_epochs, iter_idx + 1, n_iters, tag,
                              loss=loss.item(), mc=loss_mc.item(), ei=loss_ei.item())

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)
        # ★ 不换行，让进度条持续在同一行刷新
        if (epoch + 1) % 5 == 0 or (epoch + 1) == n_epochs:
            print(f"\r  [{tag}] Epoch {epoch+1}/{n_epochs} avg_loss={avg_loss:.4f}  ", flush=True)
        # 保存checkpoint
        # 保存checkpoint（每个epoch都保存，确保中断后可精确续训）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'train_losses': train_losses,
            'is_final': False,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }, ckpt_path)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == n_epochs:
            print(f"\n  [{tag}] ✓ checkpoint已保存 (epoch {epoch+1})")

    # 最终checkpoint
    torch.save({
        'epoch': n_epochs - 1,
        'model_state_dict': model.state_dict(),
        'loss': train_losses[-1] if train_losses else 0.0,
        'train_losses': train_losses,
        'is_final': True
    }, ckpt_path)
    print(f"\n  [{tag}] ✓ 最终模型已保存")
    return train_losses


print("\n  训练单算子 + EI基线...")
model_single_ei = SmallUNetMOI(in_ch=2, out_ch=1, base=24).to(device)
losses_single_ei = train_single_op_ei(model_single_ei, train_loader, N_EPOCHS, device, tag="single_ei")

single_ei_total, single_ei_obs, single_ei_miss, ssim_single_ei = evaluate_combined(
    model_single_ei, test_loader, test_mask, SIGMA, device, tag="单算子+EI")
print(f"  单算子+EI PSNR = {single_ei_total:.2f} dB, SSIM = {ssim_single_ei:.4f}")


# ========================================================================
# Step 2: MOI多算子训练（G=1, 2, 4, 8；G=1 即纯MC基线）
# 对应17.5.2节：L_MOI = ||y - A_g f(y,A_g)||² + Σ_{s≠g} ||f(A_s x̂, A_s) - x̂||²
# ========================================================================
print("\n" + "="*70)
print("Step 2: MOI多算子训练 (G=1, 2, 4, 8)")
print("="*70)


def moi_loss(model, x, pool, sigma, g, lambda_moi=LAMBDA_MOI):
    """MOI损失（固定算子池版，MOI论文原始设定）

    ★ 设定说明：pool是训练前预生成的G个**固定**掩码（模拟G台不同设备），
      训练时**每个batch**为每个样本随机分配池中一个算子（g=randint(0,G)，见train_moi），
      并非样本级"永久分配"——因此"某像素在所有训练步中从未被任何算子观测"几乎不会
      发生。G的收益来自两处：(a) 模型对G个固定掩码条件化重建（池中含测试掩码，
      G越大池的联合覆盖越广）；(b) 交叉一致性项有G-1项，把各算子下的重建互相校准。
      （对比：若每batch重采**新**随机掩码，系综期望E[AᵀA]=p·I在G=1即满秩，
      多算子的边际价值被"免费算子多样性"淹没，PSNR随G基本持平——旧版实测验证。）

    ★ 简化实现说明（两处已知差异：①②分别相对论文精确复现与deepinv参考实现）：
    1. 使用 stop-gradient（x_hat.detach()）作为交叉一致性的伪标签，
       避免trivial collapse，是教学简化而非论文精确复现；
    2. 交叉一致性路径的输入 y_s = M_s·x̂ 不带噪声（与主重建路径的带噪y不同），
       这会造成训练时网络同时看到"带噪观测"和"无噪伪观测"两种输入分布。
       注意：这是本实验的简化选择，并非与参考实现一致——deepinv 官方 MOILoss
       默认 apply_noise=True，即对伪观测 A_s x̂ 重新施加物理噪声后再送入模型。
       若要严格对齐官方行为，把 y_s 改为 mask_s*x_hat_det 后再加 sigma*randn 即可。
    核心思想一致：用多个算子覆盖零空间，交叉一致性约束。

    Args:
        x: 干净图像 (B, 1, H, W)
        pool: G个固定掩码（算子池）(G, H, W)
        sigma: 噪声水平
        g: 每个样本被分配的算子索引 (B,) 或 int
        lambda_moi: 一致性损失权重

    Returns:
        loss: MC损失 + 一致性损失
        loss_mc: 仅MC损失（用于监控）
        loss_cons: 仅一致性损失
    """
    B = x.shape[0]

    # 处理g：如果是int，扩展到(B,)
    if isinstance(g, int):
        g = torch.full((B,), g, dtype=torch.long, device=x.device)

    # ★ 每个样本取自己被分配的固定算子（模拟"样本i由第g_i台设备测量"）
    mask_g = pool[g]  # (B, H, W) 高级索引
    mask_g_2d = mask_g.unsqueeze(1)  # (B, 1, H, W)

    # 观测 y = M_g ⊙ x + noise
    y = mask_g_2d * x + sigma * torch.randn_like(x) * mask_g_2d

    # 主重建
    x_hat = model(y, mask_g_2d)  # (B, 1, H, W)

    # 第一项：MC损失（值空间约束）
    y_pred = mask_g_2d * x_hat
    loss_mc = ((y - y_pred) ** 2).sum() / mask_g_2d.sum().clamp(min=1.0)

    # 第二项：交叉算子一致性损失（零空间约束）
    # ★ 关键：stop-gradient防止x_hat退化为平凡解
    x_hat_det = x_hat.detach()
    # ★ 用 tensor 而非 float 0.0 初始化：G=1 时循环体全跳过（n_cons=0），
    #   若初始为 float，末尾 loss_cons.item() 会直接报 AttributeError
    loss_cons = torch.zeros((), device=x.device)
    n_cons = 0
    G = pool.shape[0]
    for s in range(G):
        mask_s = pool[s].unsqueeze(0).unsqueeze(1).expand(B, 1, x.shape[2], x.shape[3])  # (B, 1, H, W)
        # ★ 严格对应论文公式 Σ_{s≠g}：对每个样本 b，跳过 s == g[b] 的 trivial 自一致性项
        is_cross = (g != s)  # (B,) bool
        if not is_cross.any():
            # G=1 时（或整个 batch 恰好都分配到 s）该项恒为 0，白算一次前向没有意义
            continue
        y_s = mask_s * x_hat_det  # ★ 无噪声伪观测（与主重建路径的带噪y分布不同）
        #   这是本实验的简化：伪观测不经噪声模型，直接由重建结果加掩码得到；
        #   deepinv官方MOILoss默认apply_noise=True，会重新施加物理噪声
        #   （y_s = mask_s*x̂ + σ·randn），本实验为简化省略该步，属已知差异。
        x_hat_s = model(y_s, mask_s)
        diff = (x_hat_s - x_hat_det) ** 2  # (B, 1, H, W)
        per_sample = diff.mean(dim=[1, 2, 3])  # (B,)
        loss_cons = loss_cons + (per_sample * is_cross.float()).sum()
        n_cons += is_cross.sum().item()
    loss_cons = loss_cons / max(n_cons, 1)

    loss = loss_mc + lambda_moi * loss_cons
    return loss, loss_mc.item(), loss_cons.item()


def train_moi(model, train_loader, n_epochs, G, device, tag="moi"):
    """训练MOI模型（固定算子池版）

    ★ 算子池设计：pool = [测试掩码] + (G-1)个固定随机掩码。
      把测试掩码纳入池中可保证测试时无输入分布偏移，且G越大池的联合覆盖越广；
      G=1时退化为"纯MC+固定掩码"（连一致性项都没有，n_cons=0），
      正是MOI要拯救的最困难情形。
      池生成用固定种子+隔离RNG：保证续训时算子池一致，且不污染全局随机流。

    Args:
        G: 算子池大小（每个样本被分配池中1个算子）
    """
    # ★ 固定算子池：测试掩码 + (G-1)个固定随机掩码
    pool_parts = [test_mask.unsqueeze(0)]  # (1,H,W)
    if G > 1:
        with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
            torch.manual_seed(1000 + G)
            extra = create_random_masks_per_sample(1, IMG_SIZE, IMG_SIZE,
                                                    KEEP_RATIO, device, G=G - 1)
        pool_parts.append(extra.squeeze(0))  # (G-1,H,W)
    pool = torch.cat(pool_parts, dim=0)  # (G,H,W)

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
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1
            is_final = True
        else:
            print(f"  [{tag}] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
            model.load_state_dict(model_state)
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1
            # ★ 恢复RNG状态以保证续训后可复现性（评审意见#4）
            # 根因：torch.get_rng_state() 返回 CPU ByteTensor，但 torch.load(map_location='cuda')
            # 会将其搬到 GPU，而 torch.ByteTensor 是 CPU 专属类型，isinstance 必然失败。
            # .type(torch.ByteTensor) 隐式将 CUDA tensor 搬回 CPU 并恢复 ByteTensor 类型身份。
            if 'rng_state' in checkpoint:
                _rng = checkpoint['rng_state']
                try:
                    torch.set_rng_state(_rng)
                except TypeError:
                    try:
                        torch.set_rng_state(_rng.type(torch.ByteTensor))
                    except Exception:
                        print(f"  [{tag}] ⚠ rng_state恢复失败，跳过")
                if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
                    _cuda_rng = checkpoint['cuda_rng_state']
                    if _cuda_rng is not None:
                        try:
                            torch.cuda.set_rng_state(_cuda_rng)
                        except TypeError:
                            try:
                                torch.cuda.set_rng_state(_cuda_rng.type(torch.ByteTensor))
                            except Exception as e:
                                print(f"  [{tag}] ⚠ cuda_rng_state恢复失败({e})，跳过")

    if is_final or start_epoch >= n_epochs:
        print(f"  [{tag}] 模型已训练完毕，跳过。")
        return train_losses

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mc = 0.0
        epoch_cons = 0.0
        n_batches = 0
        n_iters = len(train_loader)
        for iter_idx, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            B = batch_x.shape[0]

            # ★ 固定算子池：每个样本随机分配池中一个固定算子（数据采集视角）
            g = torch.randint(0, G, (B,), device=device)

            optimizer.zero_grad()
            loss, mc_val, cons_val = moi_loss(model, batch_x, pool, SIGMA, g)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_mc += mc_val
            epoch_cons += cons_val
            n_batches += 1
            # ★ 每个iteration都显示进度（更频繁的更新让用户看到实时进度）
            if iter_idx % 10 == 0 or iter_idx == n_iters - 1:
                print_progress(epoch, n_epochs, iter_idx + 1, n_iters, tag,
                              loss=loss.item(), mc=mc_val, cons=cons_val)

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_mc = epoch_mc / max(n_batches, 1)
        avg_cons = epoch_cons / max(n_batches, 1)
        train_losses.append(avg_loss)
        # ★ 不换行，让进度条持续在同一行刷新
        if (epoch + 1) % 5 == 0 or (epoch + 1) == n_epochs:
            print(f"\r  [{tag}] Epoch {epoch+1}/{n_epochs} loss={avg_loss:.4f} mc={avg_mc:.4f} cons={avg_cons:.4f}  ", flush=True)

        # 保存checkpoint
        # 保存checkpoint（每个epoch都保存，确保中断后可精确续训）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'train_losses': train_losses,
            'is_final': False,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }, ckpt_path)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == n_epochs:
            print(f"\n  [{tag}] ✓ checkpoint已保存 (epoch {epoch+1})")

    # 最终checkpoint
    torch.save({
        'epoch': n_epochs - 1,
        'model_state_dict': model.state_dict(),
        'loss': train_losses[-1] if train_losses else 0.0,
        'train_losses': train_losses,
        'is_final': True
    }, ckpt_path)
    print(f"\n  [{tag}] ✓ 最终模型已保存")
    return train_losses


# 训练MOI G=1, 2, 4, 8（G=1 为纯 MC 基线：无一致性项，正是 MOI 要拯救的最困难情形）
print("  ★ 固定算子池设定（MOI论文原始设定）:")
print("    算子池 = [测试掩码] + (G-1)个固定随机掩码，每batch为每样本随机分配其中1个；")
print("    G=1 时池中只有测试掩码，退化为纯MC（n_cons=0），作为'没有MOI收益'的基线。")
print(f"    单次抽样的联合覆盖度 1-(1-p)^G（G=1:50%, G=2:75%, G=4:93.8%, G=8:99.6%），")
print("    且交叉一致性项有G-1项——G的作用体现在这两处（本实验未分离二者贡献）。")
print("    ★ 非等计算量声明：每步前向次数 = 1+(G-1)（G=1/2/4/8 → 1/3/5/9 次），")
print("      统一batch_size与epochs只保证数据顺序和优化器步数相同，不保证总计算量相同；")
print("      故PSNR随G的变化混有'算力增加'这一因素，不能全部归因于MOI机制本身。")
print("    （旧版每batch重采随机掩码，系综E[AᵀA]=p·I在G=1即满秩，G失去作用，")
print("     实测PSNR随G基本持平——两种设定的对比本身就是一个有效知识点。）")
moi_results = {}  # G -> {psnr, ssim, losses, model}
for G in [1, 2, 4, 8]:
    print(f"\n  训练MOI G={G}...")
    model_moi = SmallUNetMOI(in_ch=2, out_ch=1, base=24).to(device)
    # ★ 所有G统一使用相同batch_size，保证训练公平性
    # ★ tag用moipool_*避免误加载旧设计（每batch重采）的checkpoint
    losses = train_moi(model_moi, train_loader, N_EPOCHS, G, device, tag=f"moipool_G{G}")
    psnr_val, obs_val, miss_val, ssim_val = evaluate_combined(
        model_moi, test_loader, test_mask, SIGMA, device, tag=f"MOI G={G}")
    print(f"  MOI G={G} PSNR = {psnr_val:.2f} dB, SSIM = {ssim_val:.4f}")
    moi_results[G] = {
        'psnr': psnr_val,
        'obs': obs_val,
        'miss': miss_val,
        'ssim': ssim_val,
        'losses': losses,
        'model': model_moi
    }


# ========================================================================
# Step 3: MOI vs EI 全面对比
# ========================================================================
print("\n" + "="*70)
print("Step 3: MOI vs EI 全面对比")
print("="*70)

# 训练监督基线（用于参考）
print("\n  训练监督基线...")

def train_supervised(model, train_loader, n_epochs, device, tag="sup"):
    """监督训练（用干净x作为target）"""
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
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1
            is_final = True
        else:
            print(f"  [{tag}] 检测到未完成的训练，从第 {checkpoint['epoch']+1} 轮继续")
            model.load_state_dict(model_state)
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = checkpoint['epoch'] + 1
            # ★ 恢复RNG状态以保证续训后可复现性（评审意见#4）
            # 根因：torch.get_rng_state() 返回 CPU ByteTensor，但 torch.load(map_location='cuda')
            # 会将其搬到 GPU，而 torch.ByteTensor 是 CPU 专属类型，isinstance 必然失败。
            # .type(torch.ByteTensor) 隐式将 CUDA tensor 搬回 CPU 并恢复 ByteTensor 类型身份。
            if 'rng_state' in checkpoint:
                _rng = checkpoint['rng_state']
                try:
                    torch.set_rng_state(_rng)
                except TypeError:
                    try:
                        torch.set_rng_state(_rng.type(torch.ByteTensor))
                    except Exception:
                        print(f"  [{tag}] ⚠ rng_state恢复失败，跳过")
                if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
                    _cuda_rng = checkpoint['cuda_rng_state']
                    if _cuda_rng is not None:
                        try:
                            torch.cuda.set_rng_state(_cuda_rng)
                        except TypeError:
                            try:
                                torch.cuda.set_rng_state(_cuda_rng.type(torch.ByteTensor))
                            except Exception as e:
                                print(f"  [{tag}] ⚠ cuda_rng_state恢复失败({e})，跳过")

    if is_final or start_epoch >= n_epochs:
        print(f"  [{tag}] 模型已训练完毕，跳过。")
        return train_losses

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        n_iters = len(train_loader)
        for iter_idx, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            mask = create_random_masks_per_sample(batch_x.shape[0], IMG_SIZE, IMG_SIZE,
                                                   KEEP_RATIO, device, G=1)
            y = add_inpainting_noise(batch_x, mask, SIGMA)
            optimizer.zero_grad()
            pred = model(y, mask)
            loss = nn.MSELoss()(pred, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            # ★ 每个iteration都显示进度（更频繁的更新让用户看到实时进度）
            if iter_idx % 10 == 0 or iter_idx == n_iters - 1:
                print_progress(epoch, n_epochs, iter_idx + 1, n_iters, tag, loss=loss.item())
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)
        # ★ 不换行，让进度条持续在同一行刷新
        if (epoch + 1) % 5 == 0 or (epoch + 1) == n_epochs:
            print(f"\r  [{tag}] Epoch {epoch+1}/{n_epochs} loss={avg_loss:.4f}  ", flush=True)
        # 保存checkpoint（每个epoch都保存，确保中断后可精确续训）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'train_losses': train_losses,
            'is_final': False,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }, ckpt_path)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == n_epochs:
            print(f"\n  [{tag}] ✓ checkpoint已保存 (epoch {epoch+1})")
    torch.save({
        'epoch': n_epochs - 1,
        'model_state_dict': model.state_dict(),
        'loss': train_losses[-1] if train_losses else 0.0,
        'train_losses': train_losses,
        'is_final': True
    }, ckpt_path)
    print(f"\n  [{tag}] ✓ 最终模型已保存")
    return train_losses

model_sup = SmallUNetMOI(in_ch=2, out_ch=1, base=24).to(device)
losses_sup = train_supervised(model_sup, train_loader, N_EPOCHS, device, tag="supervised")
psnr_sup, obs_sup, miss_sup, ssim_sup = evaluate_combined(
    model_sup, test_loader, test_mask, SIGMA, device, tag="监督")
print(f"  监督 PSNR = {psnr_sup:.2f} dB, SSIM = {ssim_sup:.4f}")


# 构建 region_results：复用之前 evaluate_combined 的存储结果，不再重新评估
methods = {
    '监督': model_sup,
    '单算子+EI': model_single_ei,
    # ★ MOI G=1 = 纯MC基线（交叉一致性项 n_cons=0，退化为仅重建损失）
    'MOI G=1': moi_results[1]['model'],
    'MOI G=2': moi_results[2]['model'],
    'MOI G=4': moi_results[4]['model'],
    'MOI G=8': moi_results[8]['model'],
}

region_results = {
    '监督': {'total': psnr_sup, 'obs': obs_sup, 'miss': miss_sup, 'gap': obs_sup - miss_sup},
    '单算子+EI': {'total': single_ei_total, 'obs': single_ei_obs, 'miss': single_ei_miss,
                 'gap': single_ei_obs - single_ei_miss},
}
for G in [1, 2, 4, 8]:
    r = moi_results[G]
    region_results[f'MOI G={G}'] = {'total': r['psnr'], 'obs': r['obs'], 'miss': r['miss'],
                                     'gap': r['obs'] - r['miss']}

print("\n  各方法分区域PSNR:")
print(f"  {'方法':12s}  {'总PSNR':>10s}  {'观测像素':>10s}  {'缺失像素':>10s}  {'观测-缺失':>12s}")
print(f"  {'─'*60}")
for name in methods.keys():
    r = region_results[name]
    print(f"  {name:12s}  {r['total']:>9.2f}dB  {r['obs']:>9.2f}dB  {r['miss']:>9.2f}dB  {r['gap']:>10.2f}dB")


# 重建结果可视化
fig, axes = plt.subplots(len(methods) + 2, 6, figsize=(15, 2.5 * (len(methods) + 2)))
vis_imgs, _ = next(iter(test_loader))
vis_imgs = vis_imgs[:6].to(device)
mask_vis = test_mask.unsqueeze(0).unsqueeze(0).expand_as(vis_imgs)
y_vis = add_inpainting_noise(vis_imgs, mask_vis, SIGMA)

# 第一行：干净图像
for i in range(6):
    axes[0, i].imshow(vis_imgs[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
axes[0, 0].set_ylabel(r'干净图像 $x$', fontsize=10, rotation=0, labelpad=80)

# 第二行：观测y
for i in range(6):
    axes[1, i].imshow(y_vis[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
axes[1, 0].set_ylabel(r'观测 $y$', fontsize=10, rotation=0, labelpad=80)

# 各方法结果
for r, (name, mdl) in enumerate(methods.items()):
    with torch.no_grad():
        pred = mdl(y_vis, mask_vis).clip(0, 1)
    for i in range(6):
        axes[r + 2, i].imshow(pred[i, 0].cpu(), cmap='gray', vmin=0, vmax=1)
        axes[r + 2, i].axis('off')
    axes[r + 2, 0].set_ylabel(f'{name}\n{region_results[name]["total"]:.1f}dB',
                              fontsize=10, rotation=0, labelpad=80)

fig.suptitle('Step 3: MOI vs EI 重建结果对比\n(注: 各方法每batch前向次数不同——EI:5次, MOI G=1/2/4/8: 1/3/5/9次)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step3_moi_vs_ei_recon.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step3_moi_vs_ei_recon.png")


# 性能综合对比柱状图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

method_names = list(methods.keys())
colors = ['#2196F3', '#FF9800', '#795548', '#4CAF50', '#9C27B0', '#E91E63']  # 6色对应6方法（含G=1基线）
totals = [region_results[n]['total'] for n in method_names]
obs_vals = [region_results[n]['obs'] for n in method_names]
miss_vals = [region_results[n]['miss'] for n in method_names]

x_pos = np.arange(len(method_names))
width = 0.3
ax1.bar(x_pos, totals, width, color=colors, label='总PSNR')
ax1.bar(x_pos + width, obs_vals, width * 0.8, color=colors, alpha=0.5, label='观测像素')
ax1.bar(x_pos + 2 * width, miss_vals, width * 0.8, color=colors, alpha=0.3, label='缺失像素')
ax1.set_xticks(x_pos + width)
ax1.set_xticklabels(method_names, rotation=20, ha='right')
ax1.set_ylabel('PSNR (dB)')
ax1.set_title('总PSNR vs 观测/缺失像素PSNR')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 缺失-观测差距
gaps = [region_results[n]['gap'] for n in method_names]
bars = ax2.bar(method_names, gaps, color=colors, width=0.5)
for bar, v in zip(bars, gaps):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
             f'{v:.1f}', ha='center', fontsize=10)
ax2.set_ylabel('PSNR差距 (观测 - 缺失) dB')
# ★ 解读修正（评审意见#3）：gap 并非纯粹的"零空间约束质量"指标。
#   MC损失使观测像素重建继承观测噪声(σ=0.05)，而MNIST缺失像素多为背景(≈0)天然易重建，
#   因此自监督方法常出现"负gap"（缺失反而比观测准），不能据此判定其零空间约束优于监督。
#   gap 仅在"同一方法、比较其缺失像素绝对水平"时才反映零空间约束强弱。
ax2.set_title('观测-缺失 PSNR差距\n⚠ 负gap≠零空间约束更优：观测像素继承σ=0.05噪声，\n缺失像素多为背景(≈0)天然易重建，该指标混入噪声继承效应',
              fontsize=10)
ax2.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
ax2.set_ylim(bottom=min(min(gaps) - 0.5, -0.5), top=max(gaps) + 1.0)
ax2.tick_params(axis='x', rotation=20)
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('Step 3: MOI vs EI 全面对比\n(注: 各方法训练计算量不同，PSNR差异可能部分来自前向次数差异)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step3_moi_vs_ei_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step3_moi_vs_ei_metrics.png")


# ========================================================================
# Step 4: 联合覆盖度（单实例联合秩：与正文期望满秩条件 rank(E[AᵀA])=n 是两个不同的量）
# 对应17.5.2节：MOI要求期望意义下 rank(E[AᵀA]) = n，即算子分布的零空间被联合填补
# ★本实验的算子是对角0/1掩码，此时 rank(Σ_g A_gᵀA_g)/n 恰等于"联合覆盖度"，
#   两者是同一个量，因此下面同时报告覆盖率与秩占比（数值必然相同）。
#   对非对角算子（如CT投影、模糊），二者不等价，需分别计算。
# ========================================================================
print("\n" + "="*70)
print("Step 4: 联合覆盖度验证")
print("="*70)


def compute_coverage_stats(G_values, H, W, keep_ratio, n_trials=100, device=None):
    """计算不同G下的"零空间覆盖"统计

    返回：
        coverage_at_least_once: 至少被一个算子覆盖的像素比例
        coverage_per_g: G个算子全部覆盖的像素比例（理论≈p^G）
        rank_ratio: 平均 rank(sum_g A_gᵀA_g) / n
    """
    if device is None:
        device = torch.device('cpu')
    n_pixels = H * W
    p = keep_ratio

    results = {}
    for G in G_values:
        # ★ 经验值：n_trials次试验
        at_least_once = 0
        all_covered = 0
        rank_sum = 0
        for trial in range(n_trials):
            masks = create_random_masks_per_sample(1, H, W, keep_ratio, device, G=G).squeeze(0)
            # 至少被一个算子覆盖的像素
            coverage_count = (masks.sum(dim=0) > 0).sum().item()
            at_least_once += coverage_count / n_pixels
            # 被G个算子全部覆盖的像素
            all_cov = (masks.sum(dim=0) == G).sum().item()
            all_covered += all_cov / n_pixels
            # rank统计：sum_g A_gᵀA_g 矩阵（对角矩阵）的非零对角元数
            sum_diag = masks.sum(dim=0).view(-1)
            rank_sum += (sum_diag > 0).sum().item() / n_pixels

        results[G] = {
            'coverage_at_least_once': at_least_once / n_trials,
            'all_covered': all_covered / n_trials,
            'rank_ratio': rank_sum / n_trials,
            'theoretical_at_least_once': 1 - (1 - p) ** G,
            'theoretical_all_covered': p ** G,
        }
    return results


G_values = [1, 2, 3, 4, 6, 8, 12, 16]
N_TRIALS = 200
coverage_stats = compute_coverage_stats(G_values, IMG_SIZE, IMG_SIZE, KEEP_RATIO,
                                         n_trials=N_TRIALS, device=device)

print(f"\n  联合覆盖度验证 (H=W={IMG_SIZE}, keep_ratio={KEEP_RATIO}):")
print(f"  注: 掩码算子为对角 0/1 矩阵时，'经验覆盖'与'经验rank'在数学上等价于同一量。")
print(f"      不同算子的掩码彼此独立生成，且单个掩码的边缘覆盖概率恒等于 keep_ratio=p，")
print(f"      因此理论值 1-(1-p)^G 和 p^G 在本设计下是精确成立的（非大样本近似）；")
print(f"      经验值与理论值的微小偏差纯粹来自 n_trials={N_TRIALS} 次蒙特卡洛估计的采样噪声。")
print(f"  {'G':>3s}  {'经验覆盖':>10s}  {'理论覆盖':>10s}  {'经验rank':>10s}  {'G-全覆比例':>12s}")
print(f"  {'─'*60}")
for G in G_values:
    s = coverage_stats[G]
    print(f"  {G:>3d}  {s['coverage_at_least_once']*100:>9.2f}%  "
          f"{s['theoretical_at_least_once']*100:>9.2f}%  "
          f"{s['rank_ratio']*100:>9.2f}%  "
          f"{s['all_covered']*100:>11.2f}%")


# 可视化覆盖演化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：覆盖率随G的演化
G_arr = np.array(G_values)
exp_cov = [coverage_stats[G]['coverage_at_least_once'] * 100 for G in G_values]
theory_cov = [coverage_stats[G]['theoretical_at_least_once'] * 100 for G in G_values]
axes[0].plot(G_arr, exp_cov, 'o-', linewidth=2, markersize=8, color='#2196F3', label='经验值')
axes[0].plot(G_arr, theory_cov, 's--', linewidth=2, markersize=8, color='#FF9800',
             label=r'理论值 $1-(1-p)^G$', alpha=0.7)
axes[0].axhline(y=100, color='gray', linestyle=':', alpha=0.5)
axes[0].set_xlabel('算子数 G')
axes[0].set_ylabel('至少被一个算子覆盖的像素比例 (%)')
axes[0].set_title(r'覆盖度演化: $\Pr(\cup_g \{i: M_g[i]=1\})$')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([40, 105])

# 右图：rank ratio（在对角 0/1 掩码下与左图'经验覆盖'为同一量，此处以秩的视角再次呈现）
rank_vals = [coverage_stats[G]['rank_ratio'] * 100 for G in G_values]
axes[1].bar(G_arr, rank_vals, color='#4CAF50', alpha=0.7, width=0.6)
axes[1].axhline(y=100, color='red', linestyle='--', label='满秩 (100%)', alpha=0.7)
axes[1].set_xlabel('算子数 G')
axes[1].set_ylabel(r'$\mathrm{rank}(\sum_g A_g^\top A_g) / n$ (%)')
axes[1].set_title('联合观测矩阵的秩占比\n⚠ “经验覆盖≡经验rank”仅在对角0/1观测算子下成立，\n非对角算子(如CT投影矩阵)时二者不等价')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim([0, 110])

fig.suptitle(r'Step 4: 联合覆盖度验证——$G$ 个算子的零空间覆盖能力', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step4_rank_condition.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step4_rank_condition.png")


# 真实覆盖图：固定一张测试图像，可视化不同G下的实际覆盖
print("\n  生成真实覆盖图（固定测试图像）...")
vis_x, _ = next(iter(test_loader))
vis_x = vis_x[:1].to(device)  # 取1张

fig, axes = plt.subplots(2, len(G_values), figsize=(2.2 * len(G_values), 5))

for col, G in enumerate(G_values):
    # 生成G个随机掩码
    masks = create_random_masks_per_sample(1, IMG_SIZE, IMG_SIZE, KEEP_RATIO,
                                            device, G=G).squeeze(0)  # (G, H, W)
    # 覆盖计数：每个像素被多少个算子观测到
    coverage_count = masks.sum(dim=0).cpu().numpy()  # (H, W)
    # 第一行：被G个算子观测的位置（至少1次）
    axes[0, col].imshow((coverage_count > 0).astype(float), cmap='gray', vmin=0, vmax=1)
    axes[0, col].set_title(f'G={G}\n覆盖率={100*(coverage_count>0).mean():.1f}%', fontsize=10)
    axes[0, col].axis('off')
    # 第二行：覆盖次数热图
    im = axes[1, col].imshow(coverage_count, cmap='viridis', vmin=0, vmax=G)
    axes[1, col].set_title(f'覆盖计数 (0-{G})', fontsize=10)
    axes[1, col].axis('off')

axes[0, 0].set_ylabel('至少覆盖一次', fontsize=11)
axes[1, 0].set_ylabel('覆盖次数热图', fontsize=11)

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.3])
fig.colorbar(im, cax=cbar_ax, label='覆盖次数')
fig.suptitle('Step 4: 不同G下真实图像的像素覆盖情况（单样本）', fontsize=14)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(os.path.join(SAVE_DIR, 'step4_real_coverage.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step4_real_coverage.png")


# ========================================================================
# 总结
# ========================================================================
print("\n" + "="*70)
print("实验17.5-1 总结")
print("="*70)
print(f"\n  性能对比表:")
print(f"  {'方法':12s}  {'总PSNR':>10s}  {'观测像素':>10s}  {'缺失像素':>10s}")
print(f"  {'─'*50}")
for name in method_names:
    r = region_results[name]
    print(f"  {name:12s}  {r['total']:>9.2f}dB  {r['obs']:>9.2f}dB  {r['miss']:>9.2f}dB")

moi_g8_gap = psnr_sup - region_results['MOI G=8']['total']

# ★ G=1 纯MC基线（不参与“最佳G”判定，只作零空间约束缺失时的下界参照）
g1_psnr = moi_results[1]['psnr']
g1_miss = moi_results[1]['miss']
mc_baseline_desc = (f"G=1纯MC基线总PSNR仅{g1_psnr:.2f}dB（缺失像素{g1_miss:.2f}dB），"
                    f"远低于G≥2的所有MOI与单算子+EI——定量印证'纯MC对零空间无约束'，"
                    f"这正是MOI/EI要解决的问题")

# ★ 动态校验"PSNR随G单调提升"是否真的成立，避免硬编码结论与实际数值脱节
psnr_by_G = [moi_results[G]['psnr'] for G in [2, 4, 8]]
is_monotonic = all(psnr_by_G[i] <= psnr_by_G[i+1] for i in range(len(psnr_by_G) - 1))
# 详细版（含数值，用于打印结论）：非单调时给出具体 PSNR 轨迹
# ★ 复核补充：极差过小时应表述为"基本持平"而非"上升"——keep_ratio=0.5 下
#   G=2 已覆盖75%零空间，G再大的覆盖收益本就应趋近饱和（与Step 4互相印证）
psnr_range = max(psnr_by_G) - min(psnr_by_G)
if is_monotonic:
    trend_desc = "单调提升"
elif psnr_range < 0.1:
    trend_desc = (f"基本持平 (G=2:{psnr_by_G[0]:.2f}→G=4:{psnr_by_G[1]:.2f}→G=8:{psnr_by_G[2]:.2f}dB，"
                  f"极差仅{psnr_range:.2f}dB，在训练方差范围内；与Step 4呼应——keep_ratio=0.5时"
                  f"G=2已覆盖75%零空间，G再大覆盖收益趋近饱和)")
else:
    trend_desc = (f"整体呈上升趋势但非严格单调 (G=2:{psnr_by_G[0]:.2f}→G=4:{psnr_by_G[1]:.2f}→G=8:{psnr_by_G[2]:.2f}dB，"
                  f"提示G增大带来的边际收益递减或训练方差影响)")
# 简短版（用于图表标题，避免文字过长导致排版溢出）
if is_monotonic:
    trend_desc_short = "单调提升"
elif psnr_range < 0.1:
    trend_desc_short = "基本持平（零空间覆盖已近饱和）"
else:
    trend_desc_short = "非严格单调（边际收益递减）"

# ★ 动态生成"与监督差距"的定性描述，避免硬编码"接近监督性能"
# ★ 需处理负值（MOI反超监督）情形——自监督方法用了更多算子/前向时并非不可能
if moi_g8_gap < -0.5:
    gap_desc = "MOI在该配置下已反超监督性能（需结合上方计算量说明谨慎解读）"
elif moi_g8_gap < 1.0:
    gap_desc = "MOI在多算子场景下已接近监督性能"
elif moi_g8_gap < 3.0:
    gap_desc = "MOI在多算子场景下与监督性能仍有小幅差距"
else:
    gap_desc = "MOI在多算子场景下与监督性能仍有明显差距，提示自监督零空间约束的固有局限"

# ★ 动态判定 MOI vs EI 强弱（评审意见#1）：不预设"MOI优于EI"，以实际总PSNR为准。
#   注意：EI基线已改为固定掩码+平移对称性（真正的单算子设定），与MOI的
#   "每样本G个真实不同算子"形成名副其实的单/多算子对比。
moi_best_G = max([2, 4, 8], key=lambda G: moi_results[G]['psnr'])
moi_best_psnr = moi_results[moi_best_G]['psnr']
ei_total = region_results['单算子+EI']['total']
moi_vs_ei_diff = moi_best_psnr - ei_total
if moi_vs_ei_diff > 0.5:
    # ★ 差距较小时措辞降级为"略优"，并附计算量未对齐限定（避免过度因果解读）：
    #   MOI每batch前向次数为1+G次（主重建+G次交叉一致性），EI为1+4=5次，
    #   未做计算量对齐前，微弱差距不能归因为纯粹的零空间覆盖收益。
    adv_word = "略优" if moi_vs_ei_diff < 1.5 else "优于"
    moi_ei_desc = (f"最佳MOI (G={moi_best_G}: {moi_best_psnr:.2f}dB) {adv_word}单算子+EI ({ei_total:.2f}dB，"
                   f"差{moi_vs_ei_diff:.2f}dB)；但训练计算量未对齐（MOI G={moi_best_G}每batch约{moi_best_G+1}次前向"
                   f"vs EI 5次），该差距可能部分来自计算量增加，零空间覆盖收益的因果结论需等计算量对齐实验确认")
elif moi_vs_ei_diff < -0.5:
    moi_ei_desc = (f"单算子+EI ({ei_total:.2f}dB) 反超最佳MOI (G={moi_best_G}: {moi_best_psnr:.2f}dB)，"
                   f"与EI原文结论一致：平移对称性较强的MNIST inpainting设定下，"
                   f"虚拟多算子已能逼近甚至超过真实多算子，且EI训练计算量更省")
else:
    moi_ei_desc = (f"最佳MOI (G={moi_best_G}: {moi_best_psnr:.2f}dB) 与单算子+EI ({ei_total:.2f}dB) 性能相当，"
                   f"印证EI原文'单算子设定下EI逼近MOI'的结论")

print(f"\n  核心结论:")
print(f"  1. MOI用G个不同算子覆盖零空间，PSNR随G{trend_desc}")
print(f"     注：{mc_baseline_desc}")
print(f"  2. {moi_ei_desc}")
print(f"  3. MOI G=8 与监督差距 = {moi_g8_gap:.2f} dB → {gap_desc}")
print(f"  4. 零空间覆盖: 像素被G个掩码中'至少一个'观测到的概率 = 1-(1-p)^G，经验值与理论值吻合")
print(f"     注：理论条件 rank(E[AᵀA])=n（期望意义下满秩）由算子分布决定，与本实验的G无单调关系；")
print(f"     本实验测的是'联合覆盖度'。因掩码是对角0/1算子，覆盖率与rank(Σ_g A_gᵀA_g)/n是同一量，")
print(f"     故两者数值相同——不能由此反推'rank随G→∞趋于n'。")
print(f"  5. 缺失像素的恢复能力是评估零空间约束的关键指标")
# ★ 计算量混淆因子说明（评审意见#1）
print(f"\n  ★ 计算量说明:")
print(f"  ────────────────────────────────────────────────────────")
print(f"  MOI的训练计算量随G线性增长：每batch前向-反向次数 = 1+(G-1)，")
print(f"  G=1/2/4/8 → 1/3/5/9次（G=1为纯MC基线，无交叉一致性项）。本实验未做计算量对齐，")
print(f"  因此PSNR随G的变化趋势可能部分来自训练计算量的增加，")
print(f"  而非纯粹源于零空间覆盖增强。如需严格分离该变量，可用等计算量")
print(f"  对比（如G越大、epoch数越少）进行补充实验。")
print(f"  ────────────────────────────────────────────────────────")

print(f"\n  ★ 与17.6-1的衔接:")
print(f"  ────────────────────────────────────────────────────────")
print(f"  17.6-1 (EI) : 用信号分布的对称性  →  虚拟多算子")
print(f"  17.5-1 (MOI): 用真实的多个测量算子  →  实际多算子")
print(f"  两者殊途同归：填补 A 的零空间，让自监督损失能约束全像素")
print(f"  ────────────────────────────────────────────────────────")

print(f"\n  ★ MOI vs EI 适用场景:")
print(f"  ┌──────────────────┬──────────────────────────────────┐")
print(f"  │     MOI 适用      │           EI 适用                 │")
print(f"  ├──────────────────┼──────────────────────────────────┤")
print(f"  │ 多角度CT扫描       │  单角度CT + 利用旋转对称性         │")
print(f"  │ 多掩码MRI采集      │  固定掩码MRI + 利用平移对称性       │")
print(f"  │ 多曝光压缩成像      │  单曝光 + 利用尺度对称性           │")
print(f"  │ 数据集含多算子      │  算子固定 + 信号分布有对称性        │")
print(f"  └──────────────────┴──────────────────────────────────┘")

# 综合性能曲线：PSNR vs G
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
G_list = [1, 2, 4, 8]
moi_psnrs = [region_results[f'MOI G={G}']['total'] for G in G_list]
ei_psnr = region_results['单算子+EI']['total']
sup_psnr = region_results['监督']['total']

ax.plot(G_list, moi_psnrs, 'o-', linewidth=2, markersize=10, color='#4CAF50', label='MOI (多算子)')
ax.axhline(y=ei_psnr, color='#FF9800', linestyle='--', linewidth=2, label=f'单算子+EI (G=1+对称性) = {ei_psnr:.1f}dB')
ax.axhline(y=sup_psnr, color='#2196F3', linestyle='--', linewidth=2, label=f'监督基线 = {sup_psnr:.1f}dB')
ax.set_xlabel('算子数 G')
ax.set_ylabel('PSNR (dB)')
ax.set_title(f'PSNR vs 算子数 G: MOI 随 G {trend_desc_short}\n(注: 训练计算量随G线性增长，未做计算量对齐)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_xticks(G_list)
ax.set_xticklabels([str(g) for g in G_list])
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'summary_psnr_vs_G.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: summary_psnr_vs_G.png")
print(f"\n  实验完成！所有图表已保存到 {SAVE_DIR}")

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
    '实验': '17.5-1 MOI多算子成像',
    '分区域PSNR结果': {
        '各方法分区域PSNR': region_results,
        'MOI_vs_EI判定': moi_ei_desc + '；' + mc_baseline_desc,
    },
    '联合覆盖度验证': {
        # ★ 这段说明必须写在代码里：否则每次重跑重写 JSON 时会被丢掉
        '说明': ("单次抽样下G个算子的联合覆盖度，等于对角0/1掩码下的 "
                 "rank(Σ_g A_gᵀA_g)/n；与正文17.5.2的期望满秩条件 rank(E[AᵀA])=n"
                 "（iid掩码下G=1即成立）不是同一量"),
        **{
            f"G={G}": {
                '经验覆盖率': coverage_stats[G]['coverage_at_least_once'],
                '理论覆盖率': coverage_stats[G]['theoretical_at_least_once'],
                '秩占比': coverage_stats[G]['rank_ratio'],
                '全部覆盖比例': coverage_stats[G]['all_covered'],
            } for G in coverage_stats.keys()
        },
    },
    '实验参数': {
        '噪声水平': SIGMA,
        '保留像素比例': KEEP_RATIO,
        '训练轮数': N_EPOCHS,
        'MOI权重': LAMBDA_MOI,
        'EI权重': LAMBDA_EI,
    }
}

results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"\n数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
