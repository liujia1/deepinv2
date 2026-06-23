# -*- coding: utf-8 -*-
"""
实验6.5-1 多尺度得分匹配与退火Langevin采样
对应章节: 6.5 多尺度得分匹配: 从单一噪声到噪声条件网络

知识点:
  - 单一噪声水平的困境: 低密度区域得分不准确
  - 流形假设与覆盖率问题
  - NCSN(Noise Conditional Score Network) 多噪声水平训练
  - 噪声调度的几何级数设计
  - $\\lambda(\\sigma) = \\sigma^2$ 加权使各噪声水平贡献均匀
  - 退火Langevin采样: 从大噪声到小噪声逐步采样
  - 步长 $\\alpha_i = \\epsilon \\cdot \\sigma_i^2 / \\sigma_L^2$ 的设计
  - NCSN与扩散模型的桥梁

实验内容:
  步骤1: 单一噪声水平的困境 - 不同 $\\sigma$ 下的得分场覆盖范围
  步骤2: NCSN多噪声水平训练 (2D高斯混合)
  步骤3: 退火Langevin采样 vs 单噪声Langevin采样对比
  步骤4: 退火Langevin采样轨迹演化
  步骤5: 噪声调度设计原则验证 ( $\\lambda(\\sigma) = \\sigma^2$ 加权与步长设计)

素材来源:
  - 03-smld.ipynb 的 NCSN 训练和退火Langevin代码
  - Song & Ermon (2019) NCSN论文
  - 参考实验6.4.py

运行前提: PyTorch CPU即可
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端, 避免 plt.show 阻塞
import matplotlib.pyplot as plt
import os
import sys
import io
import time
import warnings
import logging

# 设置控制台输出为 UTF-8 (Windows 下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 静默 matplotlib 相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====== 中文字体配置(兼容本地和 Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验6.5-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)

# 在 Colab 或本地首次运行时自动创建 chinese_font.py
_chinese_font_path = os.path.join(_chinese_path, 'chinese_font.py')
if not os.path.exists(_chinese_font_path):
    print("正在创建中文字体配置模块...")
    _chinese_font_code = '''# -*- coding: utf-8 -*-
"""
中文显示支持模块 - 兼容 Windows / Linux / Colab
"""
import os
import sys
import platform
import warnings
import logging
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

def _find_chinese_font():
    candidates = []
    if platform.system() == 'Windows':
        candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    else:
        candidates = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK', 'Source Han Sans SC', 'AR PL UMing CN', 'SimHei']
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import re
    cjk_patterns = ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']
    for f in fm.ttflist:
        name_lower = f.name.lower()
        fname_lower = (os.path.basename(f.fname) if hasattr(f, 'fname') else '').lower()
        for pat in cjk_patterns:
            if re.search(pat, name_lower) or re.search(pat, fname_lower):
                return f.name
    return None

def setup_chinese_font(save_dir=None):
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    _cn_font = _find_chinese_font()
    if _cn_font:
        plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['font.family'] = 'sans-serif'
        print(f"[Font] 已检测到中文字体: {_cn_font}")
        return _cn_font
    if platform.system() != 'Windows':
        _font_url = 'https://github.com/jsntn/webfonts/raw/master/NotoSansSC-Regular.ttf'
        _font_file = os.path.join(save_dir, 'NotoSansSC-Regular.ttf')
        if os.path.exists(_font_file):
            from matplotlib.font_manager import fontManager
            fontManager.addfont(_font_file)
            plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['font.family'] = 'sans-serif'
            print(f"[Font] 已加载缓存字体: Noto Sans SC")
            return 'Noto Sans SC'
        else:
            try:
                import urllib.request
                print(f"[Font] 正在下载中文字体 NotoSansSC...")
                urllib.request.urlretrieve(_font_url, _font_file)
                from matplotlib.font_manager import fontManager
                fontManager.addfont(_font_file)
                plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
                plt.rcParams['font.family'] = 'sans-serif'
                print(f"[Font] 已下载并注册中文字体: Noto Sans SC")
                return 'Noto Sans SC'
            except Exception as e:
                print(f"[Font] 字体下载失败: {e}")
    else:
        print("[Font] 未找到中文字体")
    return None

__all__ = ['setup_chinese_font']
'''
    with open(_chinese_font_path, 'w', encoding='utf-8') as f:
        f.write(_chinese_font_code)
    print(f"[Font] 已创建字体配置模块: {_chinese_font_path}")

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到, 中文字体可能无法正常显示")
# ========================================================

# 设置随机种子
np.random.seed(42)

import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"实验6.5-1: 多尺度得分匹配与退火Langevin采样")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 训练")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")

# Checkpoint 路径 (用于训练 resume)
CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'ncsn_checkpoint.pth')


# ============================================================
# 2D 高斯混合分布 (多模态, 适合演示低密度区域问题)
# ============================================================
# 4 个高斯分量, 呈方形排列
# 注意: 以下实现假设各分量协方差为各向同性对角阵 (covs[k] = v_k * I)。
# gm2d_pdf / gm2d_score_noisy_np 中使用标量 v = covs[k][0,0] 也基于此假设。
# 若需扩展为非对角/非各向同性协方差, 需同步修改这三个函数 (用 Cholesky 分解)。
mus = torch.tensor([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])
covs = [torch.eye(2) * 0.5 for _ in range(4)]
weights = [0.25, 0.25, 0.25, 0.25]

def sample_gm2d(n):
    """从 2D 高斯混合采样 (向量化实现)

    假设各分量协方差为各向同性对角阵 covs[k] = v_k * I,
    因此 x = mu + sqrt(v) * randn(2) 等价于 mu + chol(cov) @ z。
    """
    k = np.random.choice(4, size=n, p=weights)
    k_t = torch.as_tensor(k, dtype=torch.long)
    means = mus[k_t]                                          # (n, 2)
    stds = torch.sqrt(torch.stack([covs[i][0, 0] for i in range(4)]))[k_t].unsqueeze(-1)  # (n, 1)
    return means + torch.randn(n, 2) * stds

def gm2d_pdf(x1, x2):
    """2D 高斯混合的密度值 (假设各向同性协方差)"""
    p = 0
    for k in range(4):
        mu = mus[k].numpy()
        v = covs[k][0, 0].item()
        p += weights[k] * np.exp(-0.5*((x1-mu[0])**2 + (x2-mu[1])**2)/v) / (2*np.pi*v)
    return p

def gm2d_score_noisy_np(x1, x2, sigma):
    """噪声扰动后的解析得分 $\\nabla\\log p_\\sigma(x)$ (作为 ground truth 参照)

    推导: 对各向同性高斯分量 N(mu, v*I) 加独立高斯噪声 N(0, sigma^2*I),
    卷积结果仍为高斯, 协方差变为 (v + sigma^2) * I (此处 v=0.5)。
    此解析得分用于步骤1可视化, 作为训练得分网络是否准确的参照,
    并非 NCSN 网络本身的一部分。
    """
    v_total = 0.5 + sigma**2
    score1, score2 = 0, 0
    p_total = 0
    for k in range(4):
        mu = mus[k].numpy()
        pk = weights[k] * np.exp(-0.5*((x1-mu[0])**2 + (x2-mu[1])**2)/v_total) / (2*np.pi*v_total)
        score1 += pk * (-(x1-mu[0])/v_total)
        score2 += pk * (-(x2-mu[1])/v_total)
        p_total += pk
    return score1/p_total, score2/p_total


# ============================================================
# 步骤1: 单一噪声水平的困境
# 在 2D 高斯混合上展示不同 $\\sigma$ 的得分场覆盖范围
# ============================================================
print("\n" + "=" * 60)
print("步骤1: 单一噪声水平的困境")
print("=" * 60)
print("\n[问题阐述]")
print("  单一噪声水平的 DSM 在实践中面临'低密度区域'困境:")
print("  - 数据集中在低维流形上, 远离流形的区域没有足够训练样本")
print("  - $\\sigma$ 太小: 低密度区域得分不准确, Langevin采样'迷路'")
print("  - $\\sigma$ 太大: 得分只提供全局方向, 模态细节丢失")

fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5))
x_grid = np.linspace(-5, 5, 20)
y_grid = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x_grid, y_grid)

for idx, sigma in enumerate([0.1, 0.5, 1.0, 3.0]):
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    p_grid = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            s1, s2 = gm2d_score_noisy_np(X[i,j], Y[i,j], sigma)
            U[i,j] = s1
            V[i,j] = s2
            p_grid[i,j] = gm2d_pdf(X[i,j], Y[i,j])

    # 归一化箭头长度以便可视化
    mag = np.sqrt(U**2 + V**2)
    mag_max = np.percentile(mag, 95) + 1e-8
    U_norm = U / (mag + 0.1*mag_max)
    V_norm = V / (mag + 0.1*mag_max)

    axes1[idx].contourf(X, Y, p_grid, levels=20, cmap='Blues', alpha=0.4)
    axes1[idx].quiver(X, Y, U_norm, V_norm, mag, cmap='Reds', alpha=0.7, scale=30)
    axes1[idx].set_title(r'$\sigma={}$'.format(sigma))
    axes1[idx].set_xlim(-5, 5)
    axes1[idx].set_ylim(-5, 5)
    axes1[idx].set_aspect('equal')
    axes1[idx].set_xlabel(r'$x_1$')
    axes1[idx].set_ylabel(r'$x_2$')

axes1[0].set_ylabel(r'$x_2$')
fig1.suptitle(r'不同噪声水平的得分场 (红色箭头=得分方向, 蓝色=密度轮廓)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_单一噪声水平困境.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n观察:")
print("  $\\sigma=0.1$: 得分仅在模态附近非零, 模态之间 (低密度区域) 几乎无方向指引")
print("  $\\sigma=0.5$: 得分开始覆盖模态间区域")
print("  $\\sigma=1.0$: 得分在整个区域都有方向指引, 但模态细节开始模糊")
print("  $\\sigma=3.0$: 得分全局覆盖良好, 但无法区分不同模态 -- '只见森林不见树木'")
print(f"\n图表已保存: 步骤1_单一噪声水平困境.png")


# ============================================================
# 步骤2: NCSN 多噪声水平训练 (2D 高斯混合)
# ============================================================
print("\n" + "=" * 60)
print("步骤2: NCSN 多噪声水平训练")
print("=" * 60)
print("\n[核心思想]")
print("  使用一系列噪声水平 $\\sigma_1 > \\sigma_2 > \\cdots > \\sigma_L$,")
print("  大噪声'填满'低密度区域保证全局覆盖, 小噪声保留分布细节保证局部精度.")
print("  噪声条件得分网络 $s_\\theta(x, \\sigma_i)$ 同时学习所有噪声水平的得分.")

# 噪声调度: 几何级数
L = 5
sigma_1 = 3.0
sigma_L = 0.1
ratio = (sigma_1 / sigma_L) ** (1.0 / (L - 1))
sigmas = [sigma_1 * ratio ** (-i) for i in range(L)]
sigmas_t = torch.tensor(sigmas, dtype=torch.float32, device=device)
print(f"\n噪声调度 (L={L}): $\\sigma$ = {[f'{s:.3f}' for s in sigmas]}")
print(f"几何级数公比 r = {ratio:.3f}")

# 噪声条件得分网络 (2D MLP, $\\sigma$ 编码后注入)
class NCSNet2D(torch.nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        # $\\sigma$ 编码: 将 $\\sigma$ 映射到 hidden 维向量
        self.sigma_embed = torch.nn.Sequential(
            torch.nn.Linear(1, hidden),
            torch.nn.SiLU(),
        )
        # 主网络: $x$(2) + $\\sigma$_embedding(hidden) $\\to$ score(2)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 + hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, x, sigma_idx):
        """x: (B, 2), sigma_idx: (B,)"""
        s = sigmas_t[sigma_idx].unsqueeze(-1)  # (B, 1)
        s_embed = self.sigma_embed(s)  # (B, hidden)
        inp = torch.cat([x, s_embed], dim=-1)  # (B, 2+hidden)
        return self.net(inp)

# 训练数据
N_train = 8000
x_train = sample_gm2d(N_train).to(device)

# NCSN 训练 (支持 resume)
model = NCSNet2D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

start_epoch = 0
is_final = False
train_losses = []

# Checkpoint 加载逻辑
if os.path.exists(CHECKPOINT_PATH):
    print(f"\n检测到已保存的模型: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if 'train_losses' not in checkpoint:
        raise RuntimeError(
            f"检测到旧版本 checkpoint (缺少 'train_losses' 字段):\n"
            f"  {CHECKPOINT_PATH}\n"
            f"请删除该文件后重新训练."
        )
    if checkpoint.get('is_final', False):
        print(f"  这是最终训练完成的模型, 直接加载, 跳过训练过程")
        print(f"  训练轮数: {checkpoint['epoch']+1}")
        print(f"  最终损失: {checkpoint['loss']:.6f}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        start_epoch = checkpoint['epoch'] + 1
        is_final = True
    else:
        print(f"  检测到未完成的训练, 从第 {checkpoint['epoch']+1} 轮继续")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        start_epoch = checkpoint['epoch'] + 1

# NCSN 训练循环
if not is_final:
    N_epochs = 1000
    print(f"\n训练 NCSN ({L} 个噪声水平, 共 {N_epochs} 轮)...")

    # 边界保护: 若 start_epoch >= N_epochs (如学生改小了 N_epochs 复现),
    # range 为空, 需跳过训练避免 train_losses[-1] 越界
    if start_epoch >= N_epochs:
        print(f"  注意: start_epoch({start_epoch}) >= N_epochs({N_epochs}), 无需继续训练")
        if not train_losses:
            print(f"  警告: 无历史损失记录")
        is_final = True

    if not is_final:
        t_start = time.time()

        for epoch in range(start_epoch, N_epochs):
            # 随机选择噪声水平
            sigma_idx = torch.randint(0, L, (N_train,), device=device)
            sigma_vals = sigmas_t[sigma_idx].unsqueeze(-1)  # (N, 1)

            # 加噪声
            z = torch.randn(N_train, 2, device=device)
            x_noisy = x_train + sigma_vals * z

            # DSM 目标: $\\sigma^2/2 \\cdot \\|s_\\theta(\\tilde{x}, \\sigma) + z/\\sigma\\|^2$
            # ($\\lambda(\\sigma) = \\sigma^2$ 加权)
            pred = model(x_noisy, sigma_idx)
            target = -z / sigma_vals  # $-z/\\sigma$
            loss_per_sample = 0.5 * sigma_vals.squeeze()**2 * torch.sum((pred - target)**2, dim=-1)
            loss = loss_per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch+1}/{N_epochs}: NCSN loss = {loss.item():.6f}")

        t_elapsed = time.time() - t_start
        print(f"\n训练完成, 最终损失: {train_losses[-1]:.6f}, 耗时: {t_elapsed:.1f} 秒")

    # 保存最终 checkpoint (无论本次是否训练, 只要 train_losses 非空就保存)
    if train_losses:
        torch.save({
            'epoch': start_epoch - 1 if start_epoch >= N_epochs else N_epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses[-1],
            'train_losses': train_losses,
            'is_final': True,
        }, CHECKPOINT_PATH)
        print(f"已保存最终模型: {CHECKPOINT_PATH}")
    else:
        print(f"无训练损失记录, 跳过保存 checkpoint")
else:
    print(f"\n使用已训练完成的 NCSN 模型, 跳过训练过程")
    print(f"  历史损失曲线点数: {len(train_losses)}")

# 训练损失曲线 (如果有历史记录)
if train_losses and not is_final:
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(train_losses, 'b-', lw=1.5, alpha=0.7)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel(r'$\mathcal{J}_{\mathrm{NCSN}}$ Loss')
    ax_loss.set_title('NCSN 训练损失曲线')
    ax_loss.grid(alpha=0.3)
    ax_loss.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤2_NCSN训练损失.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: 步骤2_NCSN训练损失.png")


# ============================================================
# 步骤3: 退火Langevin采样 vs 单噪声Langevin采样对比
# ============================================================
print("\n" + "=" * 60)
print("步骤3: 退火Langevin vs 单噪声Langevin采样对比")
print("=" * 60)
print("\n[退火Langevin核心思想]")
print("  从大噪声到小噪声逐步采样, 大噪声阶段粗定位, 小噪声阶段精修.")
print("  步长 $\\alpha_i = \\epsilon \\cdot \\sigma_i^2 / \\sigma_L^2$ 随噪声水平自适应调整.")

def langevin_sample(model, n_samples, T=200, epsilon=0.01, sigma_idx=0, init_scale=None):
    """单噪声水平的Langevin采样

    init_scale: 初始化标准差。若为 None, 默认用 sigmas_t[sigma_idx] (原行为);
                传入 sigmas_t[0] 可实现"从大噪声初始化", 与退火Langevin同起点,
                使对比更公平 (差异仅来自采样策略而非初始化)。
    """
    if init_scale is None:
        init_scale = sigmas_t[sigma_idx]
    x = torch.randn(n_samples, 2, device=device) * init_scale
    for t in range(T):
        idx = torch.full((n_samples,), sigma_idx, dtype=torch.long, device=device)
        with torch.no_grad():
            score = model(x, idx)
        x = x + epsilon/2 * score + np.sqrt(epsilon) * torch.randn(n_samples, 2, device=device)
    return x

def annealed_langevin_sample(model, n_samples, T_per_level=50, epsilon=2e-3):
    """退火Langevin采样 (参考 03-smld.ipynb 的 run_inference)"""
    x = torch.randn(n_samples, 2, device=device) * sigmas_t[0]
    trajectories = [x.clone().cpu().numpy()]

    for i in range(L):
        # 步长 $\\alpha_i = \\epsilon \\cdot \\sigma_i^2 / \\sigma_L^2$
        alpha_i = epsilon * (sigmas_t[i] / sigmas_t[-1])**2
        for t in range(T_per_level):
            idx = torch.full((n_samples,), i, dtype=torch.long, device=device)
            with torch.no_grad():
                score = model(x, idx)
            x = x + alpha_i/2 * score + torch.sqrt(alpha_i) * torch.randn(n_samples, 2, device=device)
        trajectories.append(x.clone().cpu().numpy())

    return x, trajectories

# 单噪声Langevin (小噪声, 但从大噪声初始化以保证公平对比)
n_samples = 2000
print(f"\n运行单噪声Langevin ($\\sigma={sigmas[-1]:.2f}$, T=500)...")
print(f"  (从大噪声 $\\sigma={sigmas[0]:.2f}$ 初始化, 与退火Langevin同起点, 公平对比)")
samples_single = langevin_sample(model, n_samples, T=500, epsilon=0.005,
                                 sigma_idx=L-1, init_scale=sigmas_t[0])

# 退火Langevin
print(f"运行退火Langevin (L={L}, T_per_level=100)...")
samples_annealed, trajectories = annealed_langevin_sample(model, n_samples, T_per_level=100, epsilon=2e-3)

# 真实样本
samples_true = sample_gm2d(n_samples).cpu().numpy()

# 评估采样质量 (各象限的样本比例)
def quadrant_stats(samples_np):
    q1 = np.mean((samples_np[:,0] > 0) & (samples_np[:,1] > 0))
    q2 = np.mean((samples_np[:,0] < 0) & (samples_np[:,1] > 0))
    q3 = np.mean((samples_np[:,0] < 0) & (samples_np[:,1] < 0))
    q4 = np.mean((samples_np[:,0] > 0) & (samples_np[:,1] < 0))
    return [q1, q2, q3, q4]

true_stats = quadrant_stats(samples_true)
single_stats = quadrant_stats(samples_single.cpu().numpy())
annealed_stats = quadrant_stats(samples_annealed.cpu().numpy())

# 样本到最近模态中心的平均距离 (衡量"紧致度": 距离越小, 样本越集中在模态中心)
def mean_dist_to_nearest_mode(samples_np):
    mus_np = mus.numpy()  # (4, 2)
    dists = np.sqrt(((samples_np[:, None, :] - mus_np[None, :, :])**2).sum(-1))  # (n, 4)
    return dists.min(axis=1).mean()

true_dist = mean_dist_to_nearest_mode(samples_true)
single_dist = mean_dist_to_nearest_mode(samples_single.cpu().numpy())
annealed_dist = mean_dist_to_nearest_mode(samples_annealed.cpu().numpy())

print(f"\n各象限样本比例:")
print(f"{'象限':>8s} | {'真实':>8s} | {'单噪声ULA':>10s} | {'退火ULA':>8s}")
print("-" * 45)
for i, name in enumerate(['Q1(++)', 'Q2(-+)', 'Q3(--)', 'Q4(+-)']):
    print(f"{name:>8s} | {true_stats[i]:8.3f} | {single_stats[i]:10.3f} | {annealed_stats[i]:8.3f}")

print(f"\n样本到最近模态中心的平均距离 (越小越紧致):")
print(f"  真实分布: {true_dist:.4f}")
print(f"  单噪声ULA: {single_dist:.4f}")
print(f"  退火ULA: {annealed_dist:.4f}")

# 可视化: 三个子图对比 (含真实分布背景对比)
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5))

# 子图1: 真实分布
axes2[0].scatter(samples_true[:,0], samples_true[:,1], s=2, alpha=0.3, c='blue')
axes2[0].set_title('真实分布样本')
axes2[0].set_xlim(-5, 5)
axes2[0].set_ylim(-5, 5)
axes2[0].set_aspect('equal')
axes2[0].grid(alpha=0.3)
axes2[0].set_xlabel(r'$x_1$')
axes2[0].set_ylabel(r'$x_2$')

# 子图2: 单噪声Langevin (叠加真实分布散点作为背景)
axes2[1].scatter(samples_true[:,0], samples_true[:,1], s=1, alpha=0.1, c='gray', label='真实')
axes2[1].scatter(samples_single.cpu().numpy()[:,0], samples_single.cpu().numpy()[:,1],
                 s=2, alpha=0.4, c='red', label='单噪声采样')
axes2[1].set_title(r'单噪声Langevin ($\sigma={:.2f}$)'.format(sigmas[-1]))
axes2[1].set_xlim(-5, 5)
axes2[1].set_ylim(-5, 5)
axes2[1].set_aspect('equal')
axes2[1].grid(alpha=0.3)
axes2[1].legend(fontsize=8)
axes2[1].set_xlabel(r'$x_1$')
axes2[1].set_ylabel(r'$x_2$')

# 子图3: 退火Langevin (叠加真实分布散点作为背景)
axes2[2].scatter(samples_true[:,0], samples_true[:,1], s=1, alpha=0.1, c='gray', label='真实')
axes2[2].scatter(samples_annealed.cpu().numpy()[:,0], samples_annealed.cpu().numpy()[:,1],
                 s=2, alpha=0.4, c='green', label='退火采样')
axes2[2].set_title(r'退火Langevin ($L={}$)'.format(L))
axes2[2].set_xlim(-5, 5)
axes2[2].set_ylim(-5, 5)
axes2[2].set_aspect('equal')
axes2[2].grid(alpha=0.3)
axes2[2].legend(fontsize=8)
axes2[2].set_xlabel(r'$x_1$')
axes2[2].set_ylabel(r'$x_2$')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_退火Langevin对比.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图表已保存: 步骤2_退火Langevin对比.png")
print("\n观察 (公平对比: 两种方法均从大噪声 $\\sigma={:.2f}$ 初始化):".format(sigmas[0]))
print("  单噪声Langevin: 全程用小噪声 $\\sigma={:.2f}$ 采样, 虽能覆盖4个模态,".format(sigmas[-1]))
print("    但步长固定且较小, 粒子收敛到模态中心的精度受限, 分布形状偏散。")
print("  退火Langevin: 步长 $\\alpha_i \\propto \\sigma_i^2$ 自适应, 大噪声阶段粗定位,")
print("    小噪声阶段精修, 样本更集中地落在模态中心, 更接近真实分布的紧致形态。")
print("  注: 在此2D简单分布上两者差异不大; 在高维/复杂分布上退火优势更显著。")


# ============================================================
# 步骤4: 退火Langevin的逐步演化
# ============================================================
print("\n" + "=" * 60)
print("步骤4: 退火Langevin采样轨迹演化")
print("=" * 60)

fig3, axes3 = plt.subplots(1, L+1, figsize=(4*(L+1), 4))
for i, traj in enumerate(trajectories):
    if i == 0:
        title = r'初始 (纯噪声, $\sigma={:.2f}$)'.format(sigmas[0])
    else:
        title = r'$\sigma={:.2f}$ 后'.format(sigmas[i-1])
    axes3[i].scatter(traj[:,0], traj[:,1], s=2, alpha=0.3)
    axes3[i].set_title(title, fontsize=10)
    axes3[i].set_xlim(-5, 5)
    axes3[i].set_ylim(-5, 5)
    axes3[i].set_aspect('equal')
    axes3[i].grid(alpha=0.3)
    axes3[i].set_xlabel(r'$x_1$')
    axes3[i].set_ylabel(r'$x_2$')

fig3.suptitle(r'退火Langevin采样轨迹: 从大噪声到小噪声', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_退火轨迹演化.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"图表已保存: 步骤3_退火轨迹演化.png")
print("\n观察:")
print("  大噪声阶段: 粒子快速分布到整个空间, 找到大致的高密度区域")
print("  小噪声阶段: 粒子在每个模态附近精修, 形成最终的清晰分布")


# ============================================================
# 步骤5: 噪声调度设计原则验证
# ============================================================
print("\n" + "=" * 60)
print("步骤5: 噪声调度设计原则验证")
print("=" * 60)

# 验证 $\\lambda(\\sigma) = \\sigma^2$ 加权使各噪声水平损失量级均匀
print("\n未加权DSM损失的相对量级 ($1/\\sigma^2$ 规律):")
for i, s in enumerate(sigmas):
    print(f"  $\\sigma_{i+1}={s:.3f}$: $1/\\sigma^2 = {1/s**2:.3f}$")

print("\n$\\sigma^2$ 加权后的损失量级 (应接近均匀):")
for i, s in enumerate(sigmas):
    print(f"  $\\sigma_{i+1}={s:.3f}$: $\\sigma^2 \\times (1/\\sigma^2) = 1.000$ (理论上)")

print("\n步长 $\\alpha_i = \\epsilon \\cdot \\sigma_i^2 / \\sigma_L^2$:")
epsilon = 2e-3
for i, s in enumerate(sigmas):
    alpha_i = epsilon * (s / sigmas[-1])**2
    print(f"  $\\sigma_{i+1}={s:.3f}$: $\\alpha_{i+1} = {alpha_i:.6f}$")

# 对照: 固定步长 (不按 $\\sigma^2$ 缩放) 的退火Langevin
print("\n--- 对照实验: 固定步长 vs 自适应步长 ---")
print("  如果步长不随噪声水平缩放, 而是固定为 $\\alpha = \\epsilon$:")
print("  - 大噪声阶段 ($\\sigma$ 大): 步长相对噪声太小, 采样效率低, 收敛慢")
print("  - 小噪声阶段 ($\\sigma$ 小): 步长相对噪声太大, 可能发散或震荡")
print("  自适应步长 $\\alpha_i \\propto \\sigma_i^2$ 使信噪比在各噪声水平保持一致。")

def annealed_langevin_fixed_step(model, n_samples, T_per_level=100, epsilon=2e-3):
    """固定步长退火Langevin (对照: 不按 $\\sigma^2$ 缩放步长)"""
    x = torch.randn(n_samples, 2, device=device) * sigmas_t[0]
    for i in range(L):
        for t in range(T_per_level):
            idx = torch.full((n_samples,), i, dtype=torch.long, device=device)
            with torch.no_grad():
                score = model(x, idx)
            x = x + epsilon/2 * score + np.sqrt(epsilon) * torch.randn(n_samples, 2, device=device)
    return x

print(f"\n运行固定步长退火Langevin ($\\alpha = {epsilon}$)...")
samples_fixed = annealed_langevin_fixed_step(model, n_samples, T_per_level=100, epsilon=2e-3)
fixed_dist = mean_dist_to_nearest_mode(samples_fixed.cpu().numpy())
print(f"  固定步长ULA 到最近模态中心平均距离: {fixed_dist:.4f}")
print(f"  自适应步长ULA 到最近模态中心平均距离: {annealed_dist:.4f}")
print(f"  真实分布 到最近模态中心平均距离: {true_dist:.4f}")
if fixed_dist > annealed_dist * 1.1:
    print("  $\\Rightarrow$ 固定步长效果明显更差, 验证了 $\\alpha_i \\propto \\sigma_i^2$ 的必要性")
else:
    print("  $\\Rightarrow$ 在此2D简单分布上差异不大, 但在高维/复杂分布上自适应步长优势更显著")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验6.5-1 总结")
print("=" * 60)
print("\n1. 单一噪声水平困境:")
print("   - $\\sigma$ 太小: 低密度区域得分不准确, Langevin采样'迷路'")
print("   - $\\sigma$ 太大: 得分只提供全局方向, 模态细节丢失")
print("\n2. NCSN 多噪声水平策略:")
print(f"   - 噪声调度 $\\sigma_1={sigmas[0]:.1f} \\to \\sigma_L={sigmas[-1]:.1f}$, 几何级数排列")
print("   - 大噪声'填满'低密度区域, 小噪声保留分布细节")
print("   - $\\lambda(\\sigma) = \\sigma^2$ 加权使各噪声水平贡献均匀")
print("\n3. 退火Langevin采样:")
print("   - 从大噪声到小噪声逐步采样, 步长 $\\alpha_i \\propto \\sigma_i^2$")
print("   - 大噪声阶段: 粗粒度调整, 快速定位高密度区域")
print("   - 小噪声阶段: 细粒度调整, 精确刻画模态细节")
print("   - 公平对比 (同起点初始化): 退火ULA样本更紧致地集中于模态中心")
print("\n4. NCSN与扩散模型的桥梁:")
print("   - 噪声调度 $\\leftrightarrow$ 时间步, 退火Langevin $\\leftrightarrow$ 逆向SDE")
print("   - NCSN = 离散化扩散模型, 第7章将展示连续化版本")
print("\n本次实验使用了以下技术:")
print("   - matplotlib.use('Agg') 非交互式后端")
print("   - Google Colab 兼容 (Google Drive 自动挂载与保存)")
print("   - 中文字体自动检测与下载 (NotoSansSC)")
print("   - LaTeX 格式数学符号")
print("   - 训练 checkpoint resume (is_final=True 时直接加载跳过训练)")
