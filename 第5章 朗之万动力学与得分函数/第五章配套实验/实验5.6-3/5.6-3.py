# -*- coding: utf-8 -*-
"""
实验5.6-3 MAP vs MMSE综合对比
对应章节：5.6 近似理论与收敛保证 -> 5.6.5 MAP vs MMSE + 不确定性量化的实践指导意义
知识点：
  - MAP估计：后验众数，"最可能"的重建（本实验用确定性PnP迭代真实求解）
  - MMSE估计：后验均值，"平均意义下最好"的重建
  - MMSE在MSE/PSNR意义下最优但更平滑；MAP保留更多高频细节（可能带振铃）
  - 不确定性量化的4个应用场景

设计说明（相对旧版的修订）：
  - 旧版用"含噪观测 y"冒充 MAP 基准，实际只演示了"重建优于退化观测"，
    并未真正对比 MAP vs MMSE。本版用确定性 PnP（去掉朗之万噪声项的不动点迭代）
    真实求解平滑后验的 MAP，兑现"两种估计哲学"的教学对比。
  - 旧版步骤6为闭式公式画的"理论示意"曲线。本版改为真实跑多条 ULA 链，
    分别扫描 ε、δ、M 三个旋钮，用实测数据作图（CPU 上较慢，建议 GPU/Colab）。

前置实验 / 依赖：
  - 需要先运行实验5.5-1，生成 sampling_results.npz（本实验读取其中的
    post_mean / mc_samples / x_true / y_obs）
  - 步骤2/6 的真实 MAP 与参数扫描需要 5.5-1 目录下的 sampling_tools 与预训练去噪器；
    若缺失则自动回退到旧版行为（y_obs 作基准 + 理论示意曲线），并打印警告。

素材来源：
  - 实验5.5-1.py（前向算子/去噪器/PnP-ULA 设定）
  - Terris et al. (2022); Laumont et al. (2021)
"""

import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验5.6-3')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)

    # 在Colab中自动创建chinese_font.py
    _chinese_font_path = os.path.join(_chinese_path, 'chinese_font.py')
    if not os.path.exists(_chinese_font_path):
        print("正在创建中文字体配置模块...")
        _chinese_font_code = '''# -*- coding: utf-8 -*-
import os, sys, platform, warnings, logging
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

def _find_chinese_font():
    candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong'] if platform.system() == 'Windows' else ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK', 'Source Han Sans SC', 'AR PL UMing CN', 'SimHei']
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available: return font
    import re
    for f in fm.ttflist:
        for pat in ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']:
            if re.search(pat, f.name.lower()): return f.name
    return None

def setup_chinese_font(save_dir=None):
    if save_dir is None: save_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
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
            return 'Noto Sans SC'
        try:
            import urllib.request
            urllib.request.urlretrieve(_font_url, _font_file)
            from matplotlib.font_manager import fontManager
            fontManager.addfont(_font_file)
            plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
            return 'Noto Sans SC'
        except: pass
    return None

__all__ = ['setup_chinese_font']
'''
        with open(_chinese_font_path, 'w', encoding='utf-8') as f:
            f.write(_chinese_font_code)
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    os.makedirs(SAVE_DIR, exist_ok=True)
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到")
# ========================================================

np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# 辅助函数：图像质量指标
# 注意：这是简化版实现，用于相对比较
# 与sampling_tools中的实现可能有数值差异（尤其NRMSE归一化口径不同）
# ============================================================
def compute_psnr(x, y):
    """计算PSNR（以 max(x.max(),1.0) 为峰值，[0,1]图像即标准PSNR）"""
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return float('inf')
    max_val = max(x.max(), 1.0)
    return 10 * np.log10(max_val ** 2 / mse)

def compute_global_similarity(x, y):
    """
    全局相似度指标（简化版，用于相对比较）
    注意：这不是标准的局部窗口SSIM，仅用于对比不同方法的相对性能。
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = np.mean(x); mu_y = np.mean(y)
    sigma_x = np.var(x); sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim

def compute_nrmse(x, y):
    """NRMSE：RMSE 除以 std(x_true)。注意与5.5-1的归一化口径不同。"""
    std_x = np.std(x)
    if std_x == 0:
        return float('inf')
    return np.sqrt(np.mean((x - y) ** 2)) / std_x


# ============================================================
# 步骤1：加载采样结果 + 重建前向算子/去噪器（供真实MAP与扫描用）
# ============================================================
print("=" * 60)
print("步骤1：加载采样结果与依赖资源")
print("=" * 60)

_possible_paths = [
    os.path.join(SAVE_DIR, 'sampling_results.npz'),
    os.path.join(os.path.dirname(SAVE_DIR), '实验5.5-1', 'sampling_results.npz'),
]
if _IN_COLAB:
    _possible_paths.insert(0, os.path.join(_gdrive, '实验5.5-1', 'sampling_results.npz'))

_data_loaded = False
for _path in _possible_paths:
    if os.path.exists(_path):
        print(f"[Data] 从 {_path} 加载采样结果")
        data = np.load(_path)
        post_mean = data['post_mean']
        mc_samples = data['mc_samples']
        x_true = data['x_true']
        y_obs = data['y_obs']
        _data_loaded = True
        break

if not _data_loaded:
    raise FileNotFoundError(
        "未找到 sampling_results.npz（请先运行实验5.5-1）。本实验依赖其采样结果。")

# 从样本重新估计后验方差（统一口径，ddof=1）
post_var = np.var(mc_samples, axis=0, ddof=1)
print(f"后验均值形状: {post_mean.shape}, 样本数量: {len(mc_samples)}")

# ---- 尝试重建前向算子与去噪器（真实MAP / 参数扫描所需）----
_exp551_dir = os.path.join(os.path.dirname(SAVE_DIR), '实验5.5-1')
_has_tools = False
_has_model = False
A = AT = AAT_norm = sigma = L_y = None
denoise = None
eps0 = (5 / 255) ** 2
L_net = 1.0
alpha = 1.0

if _exp551_dir not in sys.path:
    sys.path.insert(0, _exp551_dir)
try:
    from sampling_tools.blur_operators import blur_operators
    _has_tools = True
except Exception as e:
    print(f"[Warn] 无法导入 sampling_tools.blur_operators: {e}")

_model_path = os.path.join(_exp551_dir, 'Pretrained_models', 'RealSN_DnCNN_noise5.pth')
if _has_tools:
    try:
        from sampling_tools.load_model import DnCNN

        def _load_model_cpu(path, dev):
            net = DnCNN(channels=1, num_of_layers=17)
            sd = torch.load(path, map_location=dev, weights_only=True)
            sd2 = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
            net.load_state_dict(sd2)
            net.eval()
            return net

        if os.path.exists(_model_path):
            _net = _load_model_cpu(_model_path, device)
            denoise = lambda xx: (xx - _net(xx[None][None].to(device))[0][0]).detach()
            _has_model = True
        else:
            print(f"[Warn] 未找到预训练去噪器: {_model_path}")
    except Exception as e:
        print(f"[Warn] 去噪器加载失败: {e}")

if _has_tools:
    # 用 npz 中的 x_true 重建确定性量（前向算子、sigma 与随机数无关）
    _x_t = torch.tensor(x_true, dtype=torch.float32, device=device)
    _y_t = torch.tensor(y_obs, dtype=torch.float32, device=device)
    A, AT, AAT_norm = blur_operators([5, 5], list(x_true.shape), "uniform", device)
    BSNRdb = 40.0
    sigma = torch.linalg.matrix_norm(A(_x_t) - torch.mean(A(_x_t)), ord='fro') / \
        math.sqrt(torch.numel(_x_t) * 10 ** (BSNRdb / 10))
    L_y = AAT_norm / (sigma ** 2)
    gradf = lambda xx: AT(A(xx) - _y_t) / sigma ** 2
    _C_hi = torch.tensor(1.0, device=device)
    _C_lo = torch.tensor(0.0, device=device)
    projbox = lambda xx: torch.clamp(xx, min=_C_lo, max=_C_hi)
    _gradf = gradf
else:
    _gradf = None

_can_real_map = bool(_has_tools and _has_model and _gradf is not None)
print(f"[Env] device={device}, sampling_tools={_has_tools}, 去噪器={_has_model} -> 真实MAP/扫描可用={_can_real_map}")


# ============================================================
# 步骤2：MAP vs MMSE 对比（真实MAP）
# ============================================================
print("\n" + "=" * 60)
print("步骤2：MAP vs MMSE 对比")
print("=" * 60)

# MMSE估计 = 后验均值
x_mmse = post_mean

def pnp_map(maxit=300, delta_frac=0.99):
    """确定性PnP不动点迭代（去掉朗之万噪声项）求平滑后验的MAP。
    收敛点满足 ∇f(x) = (α/ε)(D_ε(x)-x) = -α∇g_ε(x)，即 f+α·g_ε 的驻点。"""
    delta_max = 1.0 / (L_net / eps0 + L_y)
    delta = float(delta_max) * delta_frac
    X = _y_t.clone()
    for _ in range(maxit):
        X = projbox(X - delta * _gradf(X) + alpha * delta / eps0 * (denoise(X) - X))
    return X.detach().cpu().numpy()

if _can_real_map:
    x_map = pnp_map()
    map_label = 'MAP (确定性PnP)'
    print("[Info] 使用确定性PnP迭代真实求解MAP（后验众数）")
else:
    x_map = y_obs
    map_label = '含噪观测(退化基准)'
    print("[Warn] 缺少去噪器/sampling_tools，回退：用含噪观测作退化基准（非真实MAP）")

# 计算质量指标
psnr_mmse = compute_psnr(x_true, x_mmse)
psnr_map = compute_psnr(x_true, x_map)
sim_mmse = compute_global_similarity(x_true, x_mmse)
sim_map = compute_global_similarity(x_true, x_map)
nrmse_mmse = compute_nrmse(x_true, x_mmse)
nrmse_map = compute_nrmse(x_true, x_map)

print(f"\nMMSE估计（后验均值）:")
print(f"  PSNR: {psnr_mmse:.2f} dB")
print(f"  全局相似度: {sim_mmse:.4f} (全局近似，非标准SSIM)")
print(f"  NRMSE: {nrmse_mmse:.4f}")
print(f"\n{map_label}:")
print(f"  PSNR: {psnr_map:.2f} dB")
print(f"  全局相似度: {sim_map:.4f} (全局近似，非标准SSIM)")
print(f"  NRMSE: {nrmse_map:.4f}")

if _can_real_map:
    print("\n[预期教学点] MMSE在MSE/PSNR意义下最优（PSNR≥MAP），但更平滑；")
    print("             MAP保留更多高频细节（边缘更锐利，可能有振铃），PSNR通常略低。")

# 高频能量对比（量化"MAP更锐利/MMSE更平滑"，用梯度幅值均值）
def _grad_energy(img):
    gy, gx = np.gradient(img)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
hf_mmse = _grad_energy(x_mmse)
hf_map = _grad_energy(x_map)
print(f"[高频能量] MMSE={hf_mmse:.5f}, {map_label}={hf_map:.5f}")


# ============================================================
# 步骤3：可视化对比
# ============================================================
print("\n" + "=" * 60)
print("步骤3：可视化对比")
print("=" * 60)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

axes[0, 0].imshow(x_true, cmap='gray')
axes[0, 0].set_title('原始图像 $x$')
axes[0, 0].axis('off')

axes[0, 1].imshow(y_obs, cmap='gray')
axes[0, 1].set_title('含噪+模糊观测 $y$')
axes[0, 1].axis('off')

axes[0, 2].imshow(x_mmse, cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title(f'MMSE估计 (PSNR: {psnr_mmse:.2f}dB)')
axes[0, 2].axis('off')

axes[0, 3].imshow(x_map, cmap='gray', vmin=0, vmax=1)
axes[0, 3].set_title(f'{map_label} (PSNR: {psnr_map:.2f}dB)')
axes[0, 3].axis('off')

error_mmse = np.abs(x_true - x_mmse)
error_map = np.abs(x_true - x_map)
diff = np.abs(x_mmse - x_map)

axes[1, 0].imshow(error_mmse, cmap='hot')
axes[1, 0].set_title('MMSE误差')
axes[1, 0].axis('off')

axes[1, 1].imshow(error_map, cmap='hot')
axes[1, 1].set_title(f'{map_label}误差')
axes[1, 1].axis('off')

axes[1, 2].imshow(diff, cmap='hot')
axes[1, 2].set_title('MMSE与MAP的差异')
axes[1, 2].axis('off')

# 局部放大对比（MMSE vs MAP）
h, w = x_true.shape
zoom_size = min(32, h // 2, w // 2)
zoom_region = (h // 2 - zoom_size // 2, h // 2 + zoom_size // 2,
               w // 2 - zoom_size // 2, w // 2 + zoom_size // 2)
axes[1, 3].axis('off')
ax_mmse_zoom = axes[1, 3].inset_axes([0, 0, 0.48, 1])
ax_map_zoom = axes[1, 3].inset_axes([0.52, 0, 0.48, 1])
ax_mmse_zoom.imshow(x_mmse[zoom_region[0]:zoom_region[1], zoom_region[2]:zoom_region[3]], cmap='gray')
ax_mmse_zoom.set_title('MMSE', fontsize=10)
ax_mmse_zoom.axis('off')
ax_map_zoom.imshow(x_map[zoom_region[0]:zoom_region[1], zoom_region[2]:zoom_region[3]], cmap='gray')
ax_map_zoom.set_title('MAP' if _can_real_map else '含噪观测', fontsize=10)
ax_map_zoom.axis('off')
axes[1, 3].set_title('局部放大对比', fontsize=11)

fig.suptitle('实验5.6-3 MMSE vs MAP 对比' + ('' if _can_real_map else '（退化：MAP不可用）'),
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_MAP_vs_MMSE对比.png'), dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 步骤4：两种估计哲学的对比
# ============================================================
print("\n" + "=" * 60)
print("步骤4：两种估计哲学的对比")
print("=" * 60)

print("\nMAP估计（第3-5章优化路径）:")
print("  $\\hat{x}_{MAP} = \\arg\\max_x \\log p(x|y) = \\arg\\min_x \\{f(x) + g(x)\\}$")
print("  - 给出'最可能的'重建——后验分布的众数")
print("  - 本实验用确定性PnP（去噪声项的不动点迭代）真实求解")
print("  - 保留更多高频细节，但可能放大噪声/振铃，PSNR通常不占优")

print("\nMMSE估计（第5章采样路径）:")
print("  $\\hat{x}_{MMSE} = E[x|y] = \\int x\\,p(x|y)\\,dx$")
print("  - 给出'平均意义下最好'的重建——后验分布的均值")
print("  - 是估计+不确定性：均值是点估计，方差是置信度")
print("  - MSE最优 => PSNR通常≥MAP，视觉上更平滑")

print("\n关键洞察:")
print("  后验均值天然平滑多峰不确定性；MAP取峰值、MMSE平均所有峰。")
print("  因此MMSE的PSNR更高（更平滑），MAP的边缘更锐利（高频能量更大）。")


# ============================================================
# 步骤5：不确定性量化的实践指导意义
# ============================================================
print("\n" + "=" * 60)
print("步骤5：不确定性量化的实践指导意义")
print("=" * 60)

post_std = np.sqrt(post_var)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(x_mmse, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('MMSE重建结果')
axes[0].axis('off')

im1 = axes[1].imshow(post_std, cmap='hot')
axes[1].set_title('不确定性（后验标准差）')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

uncertainty_norm = (post_std - post_std.min()) / (post_std.max() - post_std.min() + 1e-10)
overlay = np.zeros((*x_mmse.shape, 3))
overlay[:, :, 0] = x_mmse
overlay[:, :, 1] = x_mmse
overlay[:, :, 2] = x_mmse
threshold = np.percentile(uncertainty_norm, 90)
high_uncertainty_mask = uncertainty_norm > threshold
overlay[high_uncertainty_mask, 0] = 1.0
overlay[high_uncertainty_mask, 1] = 0.3
overlay[high_uncertainty_mask, 2] = 0.3

axes[2].imshow(overlay)
axes[2].set_title(f'不确定性叠加（红色=高不确定区域，占{np.mean(high_uncertainty_mask)*100:.1f}%）')
axes[2].axis('off')

fig.suptitle('不确定性量化在医学影像中的应用示意', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤5_实践指导意义.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n应用场景说明:")
print("  医学影像：高不确定性区域（红色）应标记为'需医生复核'")
print("  遥感图像：不确定性图指导采样策略——高不确定区域增加采样")
print("  科学计算：不确定性量化是实验结果可信度的核心指标")
print("  主动学习：选择高不确定性样本进行主动标注或重新测量")


# ============================================================
# 步骤6：参数敏感性分析（真实ULA扫描）
# ============================================================
print("\n" + "=" * 60)
print("步骤6：参数敏感性分析")
print("=" * 60)

step6_mode = 'sketch'  # 'real' 或 'sketch'
eps_test = np.array([eps0 * f for f in [0.5, 1.0, 2.0, 4.0, 8.0]])
M_test = np.array([10, 20, 50, 100, 200])
delta_test = np.array([0.1, 0.3, 0.5, 0.7, 0.99])

if _can_real_map:
    step6_mode = 'real'
    print("[Info] 真实扫描模式：跑多条ULA链测量ε/δ/M的影响（GPU建议，CPU较慢）...")

    def run_ula(eps, delta_frac, maxit, n_samples, seed=0, burn_frac=0.05):
        """运行一条PnP-ULA链，返回(粗化样本数组, 链均值轨迹)。"""
        torch.manual_seed(seed)
        delta_max = 1.0 / (L_net / eps + L_y)
        delta = float(delta_max) * delta_frac
        X = _y_t.clone()
        burnin = int(maxit * burn_frac)
        thin = max(1, int((maxit - burnin) / n_samples))
        samples, trace = [], []
        count = 0
        for i in range(maxit):
            X = projbox(X - delta * _gradf(X) + alpha * delta / eps * (denoise(X) - X)
                        + math.sqrt(2 * delta) * torch.randn_like(X))
            if i > burnin:
                trace.append(float(X.mean()))
                if count == thin - 1:
                    samples.append(X.detach().cpu().numpy())
                    count = 0
                else:
                    count += 1
        return np.array(samples), np.array(trace)

    # --- 实验1：ε 对后验方差的影响 ---
    var_vs_epsilon = []
    for j, e in enumerate(eps_test):
        s, _ = run_ula(float(e), 0.99, maxit=400, n_samples=60, seed=100 + j)
        var_vs_epsilon.append(float(np.mean(np.var(s, axis=0, ddof=1))))
        print(f"  ε={e:.2e} -> 平均后验方差={var_vs_epsilon[-1]:.3e}")
    var_vs_epsilon = np.array(var_vs_epsilon)

    # --- 实验2：M 对方差估计稳定性的影响（长链bootstrap）---
    long_samples, _ = run_ula(eps0, 0.99, maxit=2000, n_samples=500, seed=7)
    n_long = len(long_samples)
    rng = np.random.default_rng(0)
    std_error_vs_M = []
    for M in M_test:
        M_use = int(min(M, n_long))
        est = []
        for _ in range(60):
            idx = rng.choice(n_long, size=M_use, replace=False)
            est.append(float(np.mean(np.var(long_samples[idx], axis=0, ddof=1))))
        std_error_vs_M.append(float(np.std(est, ddof=1)))
        print(f"  M={M_use} -> 方差估计标准误差={std_error_vs_M[-1]:.3e}")
    std_error_vs_M = np.array(std_error_vs_M)

    # --- 实验3：δ 对估计稳定性的影响（用链自相关估计有效样本数）---
    var_stability = []
    for j, df in enumerate(delta_test):
        _, tr = run_ula(eps0, float(df), maxit=600, n_samples=120, seed=200 + j)
        tr = tr - tr.mean()
        if len(tr) > 2:
            rho = float(np.corrcoef(tr[:-1], tr[1:])[0, 1])
            rho = min(max(rho, 0.0), 0.999)
            ess = len(tr) * (1 - rho) / (1 + rho)  # 有效样本数
        else:
            ess = len(tr)
        se = float(np.std(tr, ddof=1) / (math.sqrt(max(ess, 1.0)) + 1e-12))
        var_stability.append(se)
        print(f"  δ_frac={df} -> ESS≈{ess:.1f}, 均值标准误差={se:.3e}")
    var_stability = np.array(var_stability)

    base_var = float(np.mean(post_var))
else:
    # 回退：闭式理论示意（与旧版一致，明确标注非真实跑数）
    print("[Warn] 缺少去噪器，回退到理论示意曲线（非真实跑数）")
    base_var = float(np.mean(post_var))
    var_vs_epsilon = base_var * (np.array([0.01, 0.02, 0.05, 0.1, 0.2]) / 0.05) ** 2
    eps_test = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
    std_error_vs_M = base_var * 0.5 / np.sqrt(M_test)
    effective_samples = np.clip(100 * np.array([0.001, 0.005, 0.01, 0.05, 0.1]) / 0.01, 5, 200)
    var_stability = base_var * (1 + 0.5 / np.sqrt(effective_samples))
    delta_test = np.array([0.001, 0.005, 0.01, 0.05, 0.1])

# --- 绘图 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(eps_test, var_vs_epsilon, 'o-', color='#2E86AB', linewidth=2, markersize=8)
axes[0].set_xlabel('去噪器噪声水平 ε')
axes[0].set_ylabel('平均后验方差')
axes[0].set_title('ε对后验方差的影响' + ('（实测）' if step6_mode == 'real' else '（理论示意）'))
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale('log')
axes[0].annotate('ε↑ → 先验约束弱 → 方差↑', xy=(0.7, 0.7), xycoords='axes fraction',
                 fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[1].plot(M_test, std_error_vs_M, 's-', color='#A23B72', linewidth=2, markersize=8)
axes[1].set_xlabel('样本数 M')
axes[1].set_ylabel('方差估计标准误差')
axes[1].set_title('M对估计稳定性的影响' + ('（实测）' if step6_mode == 'real' else '（理论示意）'))
axes[1].grid(True, alpha=0.3)
axes[1].annotate('M↑ → 估计更稳定', xy=(0.7, 0.7), xycoords='axes fraction',
                 fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[2].plot(delta_test, var_stability, '^-', color='#F18F01', linewidth=2, markersize=8)
axes[2].set_xlabel('步长比例 δ/δ_max' if step6_mode == 'real' else '步长 δ')
axes[2].set_ylabel('均值估计标准误差')
axes[2].set_title('δ对估计稳定性的影响' + ('（实测）' if step6_mode == 'real' else '（理论示意）'))
axes[2].grid(True, alpha=0.3)
axes[2].annotate('δ过小 → 有效样本少 → 不稳定', xy=(0.25, 0.8), xycoords='axes fraction',
                 fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('关键参数对不确定性估计的影响' +
             ('（真实ULA扫描实测）' if step6_mode == 'real' else '（理论示意）'), fontsize=14, y=1.02)
if step6_mode != 'real':
    fig.text(0.5, 0.01, '注：以上曲线为理论模型示意，非真实实验数据', ha='center', fontsize=9, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤6_参数敏感性.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n参数敏感性结论:")
print("  1. ε（去噪器噪声水平/先验温度）: ε越大 → 先验约束越弱 → 后验方差越大")
print("  2. M（样本数）: M越大 → 方差估计越稳定（标准误差 ∝ 1/√M）")
print("  3. δ（步长）: δ过小 → 链相关性强、有效样本少 → 估计不稳定")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.6-3 总结")
print("=" * 60)
print("1. MAP估计: 后验众数（确定性PnP求解），保留更多细节，PSNR通常不占优")
print("2. MMSE估计: 后验均值，MSE最优 => PSNR通常≥MAP，更平滑")
print("3. 二者PSNR差异不大但视觉/高频特征不同（见步骤3局部放大）")
print("4. 不确定性量化在医学、遥感、科学计算、主动学习中有重要应用")
print(f"5. 参数（ε、δ、M）影响不确定性估计——步骤6为{'真实扫描' if step6_mode=='real' else '理论示意'}")
print("\n来源: 实验5.5-1.py; Terris et al. (2022); Laumont et al. (2021)")


# ===== 保存数值结果 =====
import json

def _to_native(obj):
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except: pass
    return obj

results_summary = {
    "实验名称": "实验5.6-3 MAP vs MMSE综合对比",
    "数据信息": {
        "后验均值形状": list(post_mean.shape),
        "样本数量": int(len(mc_samples)),
        "真实MAP可用": bool(_can_real_map),
        "device": str(device),
    },
    "步骤2_MAP_vs_MMSE对比": {
        "MMSE估计_后验均值": {
            "PSNR_dB": round(float(psnr_mmse), 4),
            "全局相似度": round(float(sim_mmse), 6),
            "NRMSE": round(float(nrmse_mmse), 6),
            "高频能量": round(hf_mmse, 6),
        },
        ("MAP估计_确定性PnP" if _can_real_map else "含噪观测_退化基准"): {
            "PSNR_dB": round(float(psnr_map), 4),
            "全局相似度": round(float(sim_map), 6),
            "NRMSE": round(float(nrmse_map), 6),
            "高频能量": round(hf_map, 6),
        },
    },
    "步骤5_不确定性量化": {
        "后验标准差_平均值": round(float(np.mean(post_std)), 6),
        "后验标准差_最大值": round(float(np.max(post_std)), 6),
        "高不确定性区域占比_百分比": round(float(np.mean(high_uncertainty_mask) * 100), 4),
    },
    "步骤6_参数敏感性": {
        "模式": step6_mode,
        "基准平均方差": round(base_var, 10),
        "epsilon测试值": [float(v) for v in eps_test],
        "epsilon对应平均后验方差": [float(v) for v in var_vs_epsilon],
        "样本数测试值": [int(v) for v in M_test],
        "样本数对应方差估计标准误差": [float(v) for v in std_error_vs_M],
        "步长测试值": [float(v) for v in delta_test],
        "步长对应均值估计标准误差": [float(v) for v in var_stability],
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
