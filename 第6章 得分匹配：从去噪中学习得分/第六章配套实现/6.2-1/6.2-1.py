# -*- coding: utf-8 -*-
"""
实验6.2-1 ESM与ISM的验证
对应章节：6.2 显式得分匹配（ESM）与隐式得分匹配（ISM）
知识点：
  - ESM目标函数需要∇log p(x)作为监督信号（不可行）
  - ISM通过分部积分消去∇log p(x)
  - ISM与ESM的等价性（最优解相同，差一个常数）
  - ISM的Jacobian迹计算瓶颈（高维不可行）

素材来源：
  - 原参考实验6.1.py 拆分 + 新增ISM验证
  - 6.2节"ESM不可行性"、"ISM分部积分推导"、"Jacobian迹瓶颈"

实验内容：
  步骤1：ESM目标函数验证——需要∇log p(x)作为监督
  步骤2：ISM目标函数验证——分部积分消去∇log p(x)
  步骤3：ESM与ISM等价性验证（训练得分网络）
  步骤4：Jacobian迹的计算代价随维度增长 + 等价性证据链（不同容量模型）

运行前提：
  步骤1-2：纯NumPy，无需GPU
  步骤3-4：需要PyTorch（CPU即可，可选GPU加速）
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import torch

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验6.2-1')
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
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Colab环境GPU提示
if _IN_COLAB and device.type == 'cpu':
    print("\n" + "=" * 60)
    print("Colab GPU 启用提示")
    print("=" * 60)
    print("  当前未检测到GPU，建议使用CPU运行（本实验计算量较小）")
    print("  如需启用GPU：")
    print("    1. 点击菜单栏 '运行时' -> '更改运行时类型'")
    print("    2. 在'硬件加速器'中选择 'GPU'")
    print("    3. 点击'保存'")
    print("=" * 60)


# ============================================================
# 公共定义：1D高斯混合分布
# ============================================================
def gm_score(x):
    """精确得分函数 ∇log p(x)"""
    p1 = np.exp(-0.5 * (x + 2)**2) / np.sqrt(2*np.pi)
    p2 = np.exp(-0.5 * (x - 2)**2) / np.sqrt(2*np.pi)
    p = 0.5 * p1 + 0.5 * p2
    grad_p1 = -(x + 2) * p1
    grad_p2 = -(x - 2) * p2
    return (0.5 * grad_p1 + 0.5 * grad_p2) / p

x_grid = np.linspace(-6, 6, 200)
true_score = gm_score(x_grid)

# 采样
N_samples = 50000
x_samples = np.concatenate([
    np.random.randn(N_samples // 2) - 2,
    np.random.randn(N_samples // 2) + 2
])


# ============================================================
# 步骤1：ESM目标函数验证——需要∇log p(x)作为监督
# ============================================================
print("\n" + "=" * 60)
print("步骤1：ESM目标函数验证")
print("=" * 60)

# ESM目标：J_ESM(θ) = 0.5 * E_p(x)[||s_θ(x) - ∇log p(x)||^2]
# 用线性模型 s_θ(x) = θ*x 测试
def esm_loss(theta):
    """线性模型 s_θ(x) = θ*x 的ESM损失"""
    s_theta = theta * x_samples
    true_s = gm_score(x_samples)
    return 0.5 * np.mean((s_theta - true_s)**2)

thetas = np.linspace(-1.5, 0.5, 100)
esm_losses = [esm_loss(t) for t in thetas]
best_theta_esm = thetas[np.argmin(esm_losses)]

print(f"线性模型 $s_\\theta(x) = \\theta x$ 的ESM最优解: $\\theta^* = {best_theta_esm:.4f}$")
print(f"对应ESM损失: {min(esm_losses):.4f}")
print("\nESM的悖论：")
print("  - ESM需要知道 $\\nabla\\log p(x)$ 才能计算损失")
print("  - 但 $\\nabla\\log p(x)$ 恰恰是我们要学习的量")
print("  - 在真实问题中，只能用KDE近似 $p(x)$，但KDE遭遇维数灾难（参见6.2节文字描述）")
print("  - 结论：ESM理论正确（Fisher一致性），但实践中不可行")


# ============================================================
# 步骤2：ISM目标函数验证——分部积分消去∇log p(x)
# ============================================================
print("\n" + "=" * 60)
print("步骤2：ISM目标函数验证")
print("=" * 60)

print("\nISM推导核心（分部积分）：")
print(r"  ESM交叉项: $\mathbb{E}_{p(x)}[s_\theta(x)^T \nabla\log p(x)]$")
print(r"  $= \int s_\theta(x)^T \nabla p(x) dx$")
print(r"  $= -\int \nabla \cdot s_\theta(x) \, p(x) dx$  （分部积分）")
print(r"  $= -\mathbb{E}_{p(x)}[\nabla \cdot s_\theta(x)]$")
print(r"  $= -\mathbb{E}_{p(x)}[\mathrm{Tr}(\nabla_x s_\theta(x))]$")
print()
print("ISM目标函数：")
print(r"  $\mathcal{J}_{ISM}(\theta) = \mathbb{E}_{p(x)}\left[\mathrm{Tr}(\nabla_x s_\theta(x)) + \frac{1}{2}\|s_\theta(x)\|^2\right]$")
print("  不需要 $\\nabla\\log p(x)$！")

# 对线性模型 s_θ(x) = θ*x（1D情况）：
# Tr(∇_x s_θ(x)) = dθ/dx = θ （1D时Jacobian就是θ）
# ||s_θ(x)||^2 = θ^2 * x^2
# J_ISM(θ) = E[θ + 0.5 * θ^2 * x^2] = θ + 0.5 * θ^2 * E[x^2]

E_x2 = np.mean(x_samples**2)
print(f"\n1D线性模型的ISM解析：")
print(f"  $\\mathrm{{Tr}}(\\nabla_x s_\\theta(x)) = \\theta$")
print(f"  $\\|s_\\theta(x)\\|^2 = \\theta^2 x^2$")
print(f"  $\\mathcal{{J}}_{{ISM}}(\\theta) = \\theta + 0.5 \\theta^2 \\mathbb{{E}}[x^2]$")
print(f"  $\\mathbb{{E}}[x^2] = {E_x2:.4f}$")

def ism_loss_linear(theta):
    """线性模型 s_θ(x) = θ*x 的ISM损失（1D）"""
    # Tr(∇s) = θ, ||s||^2 = θ^2 * x^2
    return theta + 0.5 * theta**2 * E_x2

ism_losses_linear = [ism_loss_linear(t) for t in thetas]
best_theta_ism = thetas[np.argmin(ism_losses_linear)]

# 解析最优解：dJ/dθ = 1 + θ*E[x^2] = 0 → θ* = -1/E[x^2]
theta_ism_analytical = -1.0 / E_x2

print(f"\nISM最优解（数值搜索）: $\\theta^* = {best_theta_ism:.4f}$")
print(f"ISM最优解（解析公式）: $\\theta^* = -1/\\mathbb{{E}}[x^2] = {theta_ism_analytical:.4f}$")
print(f"ESM最优解（步骤1）:    $\\theta^* = {best_theta_esm:.4f}$")
print(f"\n注意：ESM和ISM的最优解在理论上相同（均约为 {theta_ism_analytical:.4f}）")
print(f"  ESM数值结果: {best_theta_esm:.4f}，ISM数值结果: {best_theta_ism:.4f}")
print(f"  微小差异来自蒙特卡罗估计误差（N={N_samples}）和网格精度（步长≈{thetas[1]-thetas[0]:.3f}）")
print(f"  这验证了等价性定理：对任意模型族，ESM与ISM具有相同最优解")
print(f"  关键在于：即使最优解相同，线性模型也无法精确表示真实得分函数")

# ====== 步骤1/2可视化：ESM与ISM损失曲线对比 ======
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 子图1：ESM与ISM损失曲线对比（归一化：各自减去最小值，让形状对比更直观）
esm_norm = np.array(esm_losses) - min(esm_losses)
ism_norm = np.array(ism_losses_linear) - min(ism_losses_linear)
axes[0].plot(thetas, esm_norm, 'r-', lw=2, label='ESM loss (shifted to min=0)')
axes[0].plot(thetas, ism_norm, 'b--', lw=2, label='ISM loss (shifted to min=0)')
axes[0].axvline(x=best_theta_esm, color='red', linestyle=':', alpha=0.6,
                label=f'ESM optimum $\\theta^*={best_theta_esm:.3f}$')
axes[0].axvline(x=best_theta_ism, color='blue', linestyle=':', alpha=0.6,
                label=f'ISM optimum $\\theta^*={best_theta_ism:.3f}$')
axes[0].set_xlabel('$\\theta$')
axes[0].set_ylabel('Loss (shifted)')
axes[0].set_title('ESM vs ISM Loss 形状对比（各自减去最小值）')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# 子图2：ESM - ISM 差值（应接近水平线，验证"差一个常数"）
diff_curve = np.array(esm_losses) - np.array(ism_losses_linear)
axes[1].plot(thetas, diff_curve, 'g-', lw=2)
axes[1].axhline(y=np.mean(diff_curve), color='orange', linestyle='--', alpha=0.7,
                label=f'Mean diff = {np.mean(diff_curve):.4f}')
axes[1].set_xlabel('$\\theta$')
axes[1].set_ylabel('ESM loss - ISM loss')
axes[1].set_title('ESM - ISM (should be constant w.r.t. $\\theta$)')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1-2_ESM与ISM损失对比.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图表已保存: 步骤1-2_ESM与ISM损失对比.png")


# ============================================================
# 步骤3：ESM与ISM等价性验证（训练得分网络）
# ============================================================
print("\n" + "=" * 60)
print("步骤3：ESM与ISM等价性验证（训练得分网络）")
print("=" * 60)

# 定义一个简单的MLP作为得分网络
class ScoreNet1D(torch.nn.Module):
    """1D得分网络：输入x（标量），输出s_θ(x)（标量）"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# ====== 训练配置 ======
N_TRAIN = 20000
N_VAL = 5000
N_EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-3
HIDDEN_DIM = 64

# 生成训练数据
x_train = np.concatenate([
    np.random.randn(N_TRAIN // 2) - 2,
    np.random.randn(N_TRAIN // 2) + 2
]).astype(np.float32)

x_val = np.concatenate([
    np.random.randn(N_VAL // 2) - 2,
    np.random.randn(N_VAL // 2) + 2
]).astype(np.float32)

# 真实得分（用于评估，不用于ISM训练）
true_score_train = gm_score(x_train).astype(np.float32)
true_score_val = gm_score(x_val).astype(np.float32)

# 转换为tensor
x_train_t = torch.tensor(x_train, device=device).unsqueeze(1)  # (N, 1)
x_val_t = torch.tensor(x_val, device=device).unsqueeze(1)
true_score_train_t = torch.tensor(true_score_train, device=device).unsqueeze(1)
true_score_val_t = torch.tensor(true_score_val, device=device).unsqueeze(1)


# ====== Checkpoint管理（支持断点续训）======
def get_checkpoint_path(method):
    """获取checkpoint文件路径"""
    return os.path.join(SAVE_DIR, f'checkpoint_{method}_final.pth')

def get_final_marker_path(method):
    """获取训练完成标记文件路径"""
    return os.path.join(SAVE_DIR, f'checkpoint_{method}_DONE.txt')

def load_checkpoint(model, optimizer, method):
    """
    加载checkpoint，支持断点续训

    返回:
        start_epoch: 起始epoch（0表示从头训练）
        is_final: 是否为最终训练完成的权重
        history: 训练历史 (train_losses, val_losses)
    """
    ckpt_path = get_checkpoint_path(method)
    final_marker = get_final_marker_path(method)

    if not os.path.exists(ckpt_path):
        print(f"  [Checkpoint] 未找到 {method} 的checkpoint，从头开始训练")
        return 0, False, ([], [])

    is_final = os.path.exists(final_marker)

    if is_final:
        print(f"  [Checkpoint] 检测到 {method} 的最终训练完成权重")
        print(f"  [Checkpoint] 直接加载最终权重，跳过训练过程")
    else:
        print(f"  [Checkpoint] 检测到 {method} 的未完成checkpoint，从断点恢复训练")

    # weights_only=False: checkpoint 中包含 train_losses 等 Python 列表，
    # weights_only=True 无法反序列化这些非张量对象
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    # 恢复训练历史
    history = (checkpoint.get('train_losses', []), checkpoint.get('val_losses', []))

    if not is_final:
        print(f"  [Checkpoint] 从 epoch {start_epoch} 继续训练")

    return start_epoch, is_final, history

def save_checkpoint(model, optimizer, epoch, method, train_losses, val_losses, is_final=False):
    """保存checkpoint"""
    ckpt_path = get_checkpoint_path(method)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, ckpt_path)

    if is_final:
        # 写入完成标记
        final_marker = get_final_marker_path(method)
        with open(final_marker, 'w') as f:
            f.write(f"Training completed at epoch {epoch}\n")


# ====== ESM训练 ======
def train_esm(model, optimizer, start_epoch, is_final, history):
    """ESM训练：直接用∇log p(x)作为监督"""
    esm_train_losses, esm_val_losses = history

    if is_final:
        return esm_train_losses, esm_val_losses  # 跳过训练，返回历史

    for epoch in range(start_epoch, N_EPOCHS):
        model.train()

        # 小批量训练
        perm = np.random.permutation(N_TRAIN)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N_TRAIN, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_TRAIN)
            idx = perm[start:end]

            x_batch = x_train_t[idx]
            score_batch = true_score_train_t[idx]

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = 0.5 * torch.mean((pred - score_batch)**2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        esm_train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = 0.5 * torch.mean((val_pred - true_score_val_t)**2).item()
        esm_val_losses.append(val_loss)

        # 每20个epoch保存checkpoint
        if (epoch + 1) % 20 == 0 or epoch == N_EPOCHS - 1:
            is_last = (epoch == N_EPOCHS - 1)
            save_checkpoint(model, optimizer, epoch, 'esm',
                          esm_train_losses, esm_val_losses, is_final=is_last)
            if (epoch + 1) % 50 == 0:
                print(f"    ESM Epoch {epoch+1}/{N_EPOCHS}: "
                      f"train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")

    return esm_train_losses, esm_val_losses


# ====== ISM训练 ======
def train_ism(model, optimizer, start_epoch, is_final, history):
    """ISM训练：不需要∇log p(x)，使用分部积分后的目标"""
    ism_train_losses, ism_val_losses = history

    if is_final:
        return ism_train_losses, ism_val_losses  # 跳过训练，返回历史

    for epoch in range(start_epoch, N_EPOCHS):
        model.train()

        perm = np.random.permutation(N_TRAIN)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N_TRAIN, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_TRAIN)
            idx = perm[start:end]

            x_batch = x_train_t[idx].clone().requires_grad_(True)

            optimizer.zero_grad()
            s = model(x_batch)  # (batch, 1)  唯一一次前向传播

            # Jacobian迹: Tr(∇_x s_θ(x))，复用 s 而非重新前向
            trace_term = torch.autograd.grad(
                s.sum(), x_batch, create_graph=True
            )[0]  # (batch, 1)

            # ISM目标: E[Tr(∇s) + 0.5 * ||s||^2]
            loss = torch.mean(trace_term + 0.5 * s**2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        ism_train_losses.append(avg_train_loss)

        # 验证：用ESM指标评估（因为ISM损失不含常数项，不能直接比较）
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            # 用ESM loss作为验证指标（需要真实得分）
            val_esm_loss = 0.5 * torch.mean((val_pred - true_score_val_t)**2).item()
        ism_val_losses.append(val_esm_loss)

        # 每20个epoch保存checkpoint
        if (epoch + 1) % 20 == 0 or epoch == N_EPOCHS - 1:
            is_last = (epoch == N_EPOCHS - 1)
            save_checkpoint(model, optimizer, epoch, 'ism',
                          ism_train_losses, ism_val_losses, is_final=is_last)
            if (epoch + 1) % 50 == 0:
                print(f"    ISM Epoch {epoch+1}/{N_EPOCHS}: "
                      f"train_loss(ISM)={avg_train_loss:.6f}, val_loss(ESM)={val_esm_loss:.6f}")

    return ism_train_losses, ism_val_losses


# ====== 执行训练 ======
print(f"\n训练配置：")
print(f"  训练样本数: {N_TRAIN}")
print(f"  验证样本数: {N_VAL}")
print(f"  网络结构: MLP(1 → {HIDDEN_DIM} → {HIDDEN_DIM} → 1)")
print(f"  训练轮数: {N_EPOCHS}")
print(f"  学习率: {LR}")
print(f"  批量大小: {BATCH_SIZE}")

# --- ESM训练 ---
print(f"\n--- ESM训练 ---")
model_esm = ScoreNet1D(HIDDEN_DIM).to(device)
optimizer_esm = torch.optim.Adam(model_esm.parameters(), lr=LR)
start_epoch_esm, is_final_esm, esm_history = load_checkpoint(model_esm, optimizer_esm, 'esm')
esm_results = train_esm(model_esm, optimizer_esm, start_epoch_esm, is_final_esm, esm_history)

# --- ISM训练 ---
print(f"\n--- ISM训练 ---")
model_ism = ScoreNet1D(HIDDEN_DIM).to(device)
optimizer_ism = torch.optim.Adam(model_ism.parameters(), lr=LR)
start_epoch_ism, is_final_ism, ism_history = load_checkpoint(model_ism, optimizer_ism, 'ism')
ism_results = train_ism(model_ism, optimizer_ism, start_epoch_ism, is_final_ism, ism_history)


# ====== 评估与对比 ======
print(f"\n--- 评估对比 ---")

model_esm.eval()
model_ism.eval()

with torch.no_grad():
    x_grid_t = torch.tensor(x_grid, device=device, dtype=torch.float32).unsqueeze(1)

    score_esm = model_esm(x_grid_t).cpu().numpy().flatten()
    score_ism = model_ism(x_grid_t).cpu().numpy().flatten()

    # ESM验证loss
    val_pred_esm = model_esm(x_val_t)
    esm_val_final = 0.5 * torch.mean((val_pred_esm - true_score_val_t)**2).item()

    # ISM验证loss（用ESM指标）
    val_pred_ism = model_ism(x_val_t)
    ism_val_final = 0.5 * torch.mean((val_pred_ism - true_score_val_t)**2).item()

print(f"ESM验证损失 (ESM metric): {esm_val_final:.6f}")
print(f"ISM验证损失 (ESM metric): {ism_val_final:.6f}")
print(f"\n结论：ESM和ISM训练出的网络都逼近了真实得分函数")
print(f"  两者验证损失接近，验证了等价性")
print(f"  关键区别：ISM不需要 $\\nabla\\log p(x)$ 作为监督！")


# ====== 可视化步骤3 ======
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1：ESM vs ISM 学到的得分函数
axes[0, 0].plot(x_grid, true_score, 'k-', lw=2, label='True $\\nabla\\log p(x)$')
axes[0, 0].plot(x_grid, score_esm, 'r--', lw=1.5, label=f'ESM (val_loss={esm_val_final:.4f})')
axes[0, 0].plot(x_grid, score_ism, 'b:', lw=1.5, label=f'ISM (val_loss={ism_val_final:.4f})')
axes[0, 0].set_xlabel('$x$')
axes[0, 0].set_ylabel('$s(x)$')
axes[0, 0].set_title('ESM vs ISM 学到的得分函数')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 子图2：ESM训练曲线
if len(esm_results[0]) > 0:
    axes[0, 1].plot(esm_results[0], 'r-', lw=1.5, label='Train (ESM loss)')
    axes[0, 1].plot(esm_results[1], 'r--', lw=1.5, label='Val (ESM loss)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
else:
    axes[0, 1].text(0.5, 0.5, '无训练历史\n(使用预训练权重)',
                    ha='center', va='center', transform=axes[0, 1].transAxes,
                    fontsize=12, color='gray')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('ESM训练曲线')

# 子图3：ISM训练曲线
if len(ism_results[0]) > 0:
    axes[1, 0].plot(ism_results[0], 'b-', lw=1.5, label='Train (ISM loss)')
    # 注意：ISM损失不含常数项，不能直接与ESM比较，这里用ESM作为代理指标
    axes[1, 0].plot(ism_results[1], 'b--', lw=1.5, label='Val (ESM proxy)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, '无训练历史\n(使用预训练权重)',
                    ha='center', va='center', transform=axes[1, 0].transAxes,
                    fontsize=12, color='gray')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('ISM训练曲线')

# 子图4：误差对比
error_esm = np.abs(score_esm - true_score)
error_ism = np.abs(score_ism - true_score)
axes[1, 1].plot(x_grid, error_esm, 'r-', lw=1.5, alpha=0.7, label='ESM error')
axes[1, 1].plot(x_grid, error_ism, 'b-', lw=1.5, alpha=0.7, label='ISM error')
axes[1, 1].set_xlabel('$x$')
axes[1, 1].set_ylabel('$|s_\\theta(x) - \\nabla\\log p(x)|$')
axes[1, 1].set_title('得分估计误差')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_ESM与ISM等价性.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图表已保存: 步骤3_ESM与ISM等价性.png")


# ============================================================
# 步骤4：Jacobian迹的计算代价随维度增长
# ============================================================
print("\n" + "=" * 60)
print("步骤4：Jacobian迹的计算代价")
print("=" * 60)

# 防御性检查：步骤4依赖步骤3初始化的变量
# 注意：dir() 不返回模块级全局变量，必须用 globals() 检查
_required_vars = ['x_train_t', 'true_score_train_t', 'x_grid_t', 'x_grid', 'true_score',
                  'N_TRAIN', 'BATCH_SIZE', 'LR', 'device', 'SAVE_DIR']
_globals = globals()
_missing = [v for v in _required_vars if v not in _globals]
if _missing:
    raise NameError(
        f"步骤4依赖步骤3的变量，请先运行步骤3。缺失: {_missing}"
    )

print("\nISM的瓶颈：$\\mathrm{Tr}(\\nabla_x s_\\theta(x))$ 的计算代价")
print("-" * 50)

# 模拟不同维度下的计算代价
# 对于 d 维输入，计算 Jacobian 迹的几种方法：
dims = [1, 2, 5, 10, 50, 100, 256, 512, 1024, 4096, 16384, 65536]

# 方法1：直接构造完整Jacobian → O(d^2)
cost_full_jacobian = [d**2 for d in dims]

# 方法2：逐列计算（d次反向传播）→ O(d) 次前向/反向
cost_per_column = [d for d in dims]

# 方法3：Hutchinson估计（单次随机估计）→ O(1) 次前向/反向
# 但这是SSM/Hutchinson方法，不是精确ISM
cost_hutchinson = [1 for d in dims]

print(f"{'维度 $d$':>10s} | {'完整Jacobian $O(d^2)$':>20s} | {'逐列计算 $O(d)$次':>20s} | {'Hutchinson $O(1)$次':>20s}")
print("-" * 80)
for d, c1, c2, c3 in zip(dims, cost_full_jacobian, cost_per_column, cost_hutchinson):
    print(f"{d:10d} | {c1:20d} | {c2:20d} | {c3:20d}")

print(f"\n关键观察：")
print(f"  - 图像数据 $d = 256 \\times 256 = 65536$")
print(f"  - 完整Jacobian: $d^2 = {65536**2:,}$ 次运算 → 完全不可行")
print(f"  - 逐列计算: $d = 65536$ 次前向/反向传播 → 不可行")
print(f"  - Hutchinson估计: $O(1)$ 次 → 可行！（引出SSM方法，见实验6.3-1）")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图1：计算代价对比
axes[0].loglog(dims, cost_full_jacobian, 'ro-', lw=2, markersize=6, label='Full Jacobian $O(d^2)$')
axes[0].loglog(dims, cost_per_column, 'bs-', lw=2, markersize=6, label='Per-column $O(d)$ passes')
axes[0].loglog(dims, cost_hutchinson, 'g^-', lw=2, markersize=6, label='Hutchinson $O(1)$ pass')
axes[0].axvline(x=65536, color='gray', linestyle='--', alpha=0.5, label='Image $d=65536$')
axes[0].set_xlabel('Dimension $d$')
axes[0].set_ylabel('Computational cost (relative)')
axes[0].set_title('Jacobian trace computation cost')
axes[0].legend()
axes[0].grid(alpha=0.3, which='both')

# 子图2：ESM与ISM等价性证据链（不同模型容量下的差异）
# 这一步需要额外训练几个不同容量的模型来展示等价性
print("\n  训练不同容量的模型以构建等价性证据链...")

# 定义不同容量的得分网络
class ScoreNetFlex(torch.nn.Module):
    """可变容量的1D得分网络"""
    def __init__(self, hidden_dim=64, n_hidden=2):
        super().__init__()
        layers = [torch.nn.Linear(1, hidden_dim), torch.nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Tanh()]
        layers += [torch.nn.Linear(hidden_dim, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 三种容量配置：浅层MLP、中层MLP、深层MLP
configs = [
    ('Shallow MLP\n(1->8->1)', 8, 1),      # 浅层MLP
    ('Medium MLP\n(1->32->1)', 32, 1),     # 中层MLP
    ('Deep MLP\n(1->64->64->1)', 64, 2),   # 深层MLP
]

equiv_diffs = []
equiv_esm_losses = []
equiv_ism_losses = []

for name, hidden, n_layers in configs:
    # ESM训练
    np.random.seed(42)  # 对齐numpy随机状态，确保ESM和ISM使用相同的minibatch顺序
    torch.manual_seed(42)
    model_e = ScoreNetFlex(hidden, n_layers).to(device)
    opt_e = torch.optim.Adam(model_e.parameters(), lr=LR)
    for _ in range(100):
        perm = np.random.permutation(N_TRAIN)
        for start in range(0, N_TRAIN, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_TRAIN)
            idx = perm[start:end]
            x_batch = x_train_t[idx]
            score_batch = true_score_train_t[idx]
            opt_e.zero_grad()
            pred = model_e(x_batch)
            loss = 0.5 * torch.mean((pred - score_batch)**2)
            loss.backward()
            opt_e.step()
    model_e.eval()
    with torch.no_grad():
        esm_pred = model_e(x_grid_t).cpu().numpy().flatten()
        esm_l = 0.5 * np.mean((esm_pred - true_score)**2)

    # ISM训练
    np.random.seed(42)  # 对齐numpy随机状态，确保ESM和ISM使用相同的minibatch顺序
    torch.manual_seed(42)
    model_i = ScoreNetFlex(hidden, n_layers).to(device)
    opt_i = torch.optim.Adam(model_i.parameters(), lr=LR)
    for _ in range(100):
        perm = np.random.permutation(N_TRAIN)
        for start in range(0, N_TRAIN, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_TRAIN)
            idx = perm[start:end]
            x_batch = x_train_t[idx].clone().requires_grad_(True)
            opt_i.zero_grad()
            s = model_i(x_batch)
            trace_term = torch.autograd.grad(s.sum(), x_batch, create_graph=True)[0]
            loss = torch.mean(trace_term + 0.5 * s**2)
            loss.backward()
            opt_i.step()
    model_i.eval()
    with torch.no_grad():
        ism_pred = model_i(x_grid_t).cpu().numpy().flatten()
        ism_l = 0.5 * np.mean((ism_pred - true_score)**2)

    diff = np.sqrt(np.mean((esm_pred - ism_pred)**2))
    equiv_diffs.append(diff)
    equiv_esm_losses.append(esm_l)
    equiv_ism_losses.append(ism_l)
    print(f"    {name}: ESM loss={esm_l:.4f}, ISM loss={ism_l:.4f}, ESM-ISM diff={diff:.4f}")

# 绘制等价性证据链
x_pos = np.arange(len(configs))
width = 0.25
axes[1].bar(x_pos - width, equiv_esm_losses, width, label='ESM loss', color='red', alpha=0.7)
axes[1].bar(x_pos, equiv_ism_losses, width, label='ISM loss', color='blue', alpha=0.7)
axes[1].bar(x_pos + width, equiv_diffs, width, label='ESM-ISM diff', color='green', alpha=0.7)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([c[0] for c in configs], fontsize=9)
axes[1].set_ylabel('Loss / Difference')
axes[1].set_title('ESM与ISM等价性证据链（容量越大，差异越小）')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3, axis='y')
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_Jacobian迹与等价性.png'), dpi=150, bbox_inches='tight')
plt.close()

# 删除旧文件名
old_path = os.path.join(SAVE_DIR, '步骤4_Jacobian迹计算代价.png')
if os.path.exists(old_path):
    os.remove(old_path)

print(f"\n图表已保存: 步骤4_Jacobian迹与等价性.png")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验6.2-1 总结")
print("=" * 60)
print("1. ESM（显式得分匹配）")
print(r"   $\mathcal{J}_{ESM}(\theta) = \frac{1}{2}\mathbb{E}_{p(x)}[\|s_\theta(x) - \nabla\log p(x)\|^2]$")
print("   - Fisher一致性：最优解 $s_{\\theta^*}(x) = \\nabla\\log p(x)$")
print("   - 不可行：需要 $\\nabla\\log p(x)$ 作为监督信号")
print("   - KDE近似在高维中遭遇维数灾难")
print()
print("2. ISM（隐式得分匹配）")
print(r"   $\mathcal{J}_{ISM}(\theta) = \mathbb{E}_{p(x)}[\mathrm{Tr}(\nabla_x s_\theta(x)) + \frac{1}{2}\|s_\theta(x)\|^2]$")
print("   - 通过分部积分消去了 $\\nabla\\log p(x)$")
print("   - 与ESM等价（差一个常数）")
print("   - 但引入了Jacobian迹的计算")
print()
print("3. ESM与ISM等价性验证")
print(f"   - ESM验证损失: {esm_val_final:.6f}")
print(f"   - ISM验证损失: {ism_val_final:.6f}")
print("   - 两者都逼近了真实得分函数，验证了等价性")
print()
print("   重要概念澄清：")
print("   - 等价性定理保证的是损失泛函在期望下相差一个常数（最优解相同）")
print("   - 并不直接保证两次独立训练会收敛到完全相同的参数")
print("   - 步骤4观察到的\"容量越大、ESM和ISM学出的网络越接近\"")
print("   - 更准确的解释是：两者都在逐渐逼近同一个真实得分函数，所以彼此趋同")
print("   - 这体现了\"理论等价\"与\"实践收敛\"的区别")
print()
print("4. ISM的计算瓶颈")
print("   - Jacobian迹 $\\mathrm{Tr}(\\nabla_x s_\\theta(x))$ 在高维中计算代价过高")
print("   - 图像 $d=65536$: 完整Jacobian $O(d^2)$ 不可行")
print("   - 逐列计算 $O(d)$ 次前向/反向传播也不可行")
print()
print("5. 两条解决路径")
print("   - DSM（去噪得分匹配）：噪声扰动，完全避免Jacobian → 实用！")
print("   - SSM（切片得分匹配）：随机投影，近似Jacobian迹 → 可行！")
print()
print("=" * 60)
print("下一步：实验6.3-1 DSM训练与得分提取")
print("=" * 60)
