"""
实验5.3-1 Tweedie等式验证——从去噪器提取得分
对应章节：5.3（Tweedie等式：从去噪器到得分函数）
知识点：
  - Tweedie等式：得分函数 = 去噪器残差 / 噪声方差
  - MMSE去噪器与得分函数的等价性
  - 从去噪器构建隐式先验

素材来源：
  - Mathematics.../Teaching Unit 2/labs/lab2_PnP_sol.ipynb
    - 去噪器定义与Tweedie替换
    - PnP-ULA中得分函数的计算方式
  - 原创设计：1D解析情形的Tweedie等式验证

实验内容：
  步骤1：1D情形的Tweedie等式验证（解析解）
  步骤2：图像去噪中的Tweedie等式验证（学习去噪器）
  步骤3：Tweedie等式对PnP的指导意义

运行前提：
  步骤2需要预训练RealSN-DnCNN模型与测试图cman.png（脚本会自动在本实验目录、
    第五章配套实验的兄弟实验目录、以及教学材料目录中搜索并拷贝到当前目录）
  步骤2在GPU与CPU上都可推理（DnCNN 17层，256x256单张CPU推理秒级），
    原始sampling_tools.load_model硬编码.cuda()，本脚本另提供CPU加载分支
  步骤1和3不依赖任何外部资源，纯numpy即可完成
"""

import numpy as np
import glob
import shutil
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
    SAVE_DIR = os.path.join(_gdrive, '实验5.3-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)


# ====== Windows GBK 控制台下 ∇/ε 等符号的安全打印 ======
_raw_print = print


def print(*args, **kwargs):  # noqa: A001 - 覆盖内建print，仅做编码兜底
    try:
        _raw_print(*args, **kwargs)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, 'encoding', None) or 'utf-8'
        _raw_print(*[str(a).encode(enc, 'replace').decode(enc, 'replace') for a in args], **kwargs)

# ============================================================
# 步骤1：1D情形的Tweedie等式验证
# ============================================================
print("=" * 60)
print("步骤1：1D情形的Tweedie等式验证")
print("=" * 60)

def exact_mmse_denoiser_1d(y, eps):
    """
    1D高斯先验的精确MMSE去噪器
    x ~ N(0,1), y = x + sqrt(eps)*z
    E[x|y] = y / (1 + eps)
    """
    return y / (1 + eps)

def numerical_score_1d(y, eps, h=1e-5):
    """
    用中心差分数值估计得分函数，用于与Tweedie等式对比验证

    p_eps(y) = N(y; 0, 1+eps) 的对数概率
    log p_eps(y) = -y^2 / (2(1+eps)) + const

    数值梯度：∇log p_eps(y) ≈ (log p(y+h) - log p(y-h)) / (2h)

    注：本例 log p_eps 严格二次，中心差分无截断误差（三阶导为零），
    实测残量完全来自两个几乎相等的浮点数相减后被 1/(2h) 放大，
    因此误差随 h 减小而降低（标度 ~1/h），不是算法误差。
    """
    log_p = lambda y_val: -0.5 * y_val**2 / (1 + eps)  # 忽略常数项
    return (log_p(y + h) - log_p(y - h)) / (2 * h)

# 测试Tweedie等式
eps_step1 = 0.5
y_values = np.linspace(-3, 3, 200)

denoised = exact_mmse_denoiser_1d(y_values, eps_step1)
score_numerical = numerical_score_1d(y_values, eps_step1)  # 数值梯度估计的得分
score_from_denoiser = (denoised - y_values) / eps_step1    # Tweedie公式给出的得分

# 可视化验证
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(y_values, denoised, 'b-', lw=2, label=r'$D_\varepsilon^*(y) = y/(1+\varepsilon)$')
plt.plot(y_values, y_values, 'k--', lw=1, label='$y$ (含噪观测)')
plt.xlabel('$y$')
plt.ylabel('去噪输出')
plt.title('MMSE去噪器 (1D)')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(y_values, score_numerical, 'r-', lw=2, label=r'数值梯度 $\nabla \log p_\varepsilon(y)$')
plt.plot(y_values, score_from_denoiser, 'b--', lw=2, label=r'Tweedie: $(D^*-y)/\varepsilon$')
plt.xlabel('$y$')
plt.ylabel('得分')
plt.title('Tweedie等式验证')
plt.legend()
plt.grid(alpha=0.3)

# 差异图
plt.subplot(1, 3, 3)
diff = score_numerical - score_from_denoiser
plt.plot(y_values, diff, 'g-', lw=2)
plt.xlabel('$y$')
plt.ylabel('差异 (数值梯度 - Tweedie)')
plt.title('验证误差（浮点舍入被 1/(2h) 放大，非等式偏差）')
plt.axhline(y=0, color='k', linestyle='--')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_Tweedie等式验证.png'), dpi=150)
plt.close()

# 打印最大误差
print(f"最大绝对误差: {np.max(np.abs(diff)):.2e}")
print(f"最大相对误差: {np.max(np.abs(diff)) / (np.max(np.abs(score_numerical)) + 1e-10):.2e}")
print("说明：log p_eps 为二次函数，中心差分无截断误差；上述残量全来自浮点舍入被 1/(2h) 放大")
print("      因此它随 h 减小而减小（标度 ~1/h），解析上 Tweedie 等式两边恒等")


# ============================================================
# 多噪声水平验证
# ============================================================
print("\n" + "-" * 60)
print("多噪声水平Tweedie等式验证")
print("-" * 60)

eps_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
_multi_err = {}
for eps_val in eps_values:
    denoised = exact_mmse_denoiser_1d(y_values, eps_val)
    score_numerical = numerical_score_1d(y_values, eps_val)
    score_from_denoiser = (denoised - y_values) / eps_val
    max_err = np.max(np.abs(score_numerical - score_from_denoiser))
    _multi_err[float(eps_val)] = float(max_err)
    print(f"  ε={eps_val:5.2f}: 最大绝对误差 = {max_err:.2e}")


# ============================================================
# 步骤2：图像去噪中的Tweedie等式验证（学习去噪器）
# 取自 lab2_PnP_sol.ipynb 的去噪器
# ============================================================
print("\n" + "=" * 60)
print("步骤2：图像去噪中的Tweedie等式验证（学习去噪器）")
print("=" * 60)

from PIL import Image


def _find_sampling_tools_parent():
    """在本目录/父目录/第五章兄弟实验目录/整个工作区中搜索 sampling_tools 包，返回其父目录"""
    local = os.path.join(SAVE_DIR, 'sampling_tools')
    if os.path.isdir(local):
        return SAVE_DIR
    parent = os.path.join(os.path.dirname(SAVE_DIR), 'sampling_tools')
    if os.path.isdir(parent):
        return os.path.dirname(SAVE_DIR)
    ws_root = os.path.dirname(os.path.dirname(os.path.dirname(SAVE_DIR)))  # 实验目录/第五章配套实验/第5章/工作区根
    for cand in glob.glob(os.path.join(ws_root, '*', '*', 'sampling_tools')):
        if os.path.isdir(cand):
            return os.path.dirname(cand)
    for cand in glob.glob(os.path.join(ws_root, '**', 'sampling_tools'), recursive=True):
        if os.path.isdir(cand):
            return os.path.dirname(cand)
    return None


def psnr_db(ref, est, peak=1.0):
    """峰值信噪比（图像范围[0,1]，故peak=1）"""
    mse = float(torch.mean((est - ref) ** 2))
    return float('inf') if mse <= 0 else 10.0 * np.log10(peak ** 2 / mse)


def edge_flat_masks(clean, q_edge=0.90, q_flat=0.50):
    """用干净图（真值）的梯度幅值分位数划分边缘(前10%)与平坦(后50%)区域，
    划分只依赖真值，不依赖去噪结果，避免循环论证"""
    gx = torch.zeros_like(clean)
    gy = torch.zeros_like(clean)
    gx[:, :-1] = clean[:, 1:] - clean[:, :-1]
    gy[:-1, :] = clean[1:, :] - clean[:-1, :]
    mag = torch.sqrt(gx ** 2 + gy ** 2)
    m_edge = mag >= torch.quantile(mag.flatten(), q_edge)
    m_flat = mag <= torch.quantile(mag.flatten(), q_flat)
    return m_edge, m_flat


# 检查sampling_tools（提供DnCNN结构与权重键名定义，不修改兄弟实验的拷贝）
_tools_parent = _find_sampling_tools_parent()
_DnCNN = None
if _tools_parent is not None:
    sys.path.insert(0, _tools_parent)
    from sampling_tools import *  # noqa: F401,F403
    from sampling_tools.load_model import DnCNN as _DnCNN
    _has_sampling_tools = True
else:
    _has_sampling_tools = False
    print("警告: sampling_tools 模块未找到，步骤2将被跳过")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def _load_denoiser(path, dev):
    """加载RealSN-DnCNN；GPU走原始load_model，CPU走等价分支（原模块硬编码.cuda()）"""
    if dev.type == 'cuda':
        return load_model(path, dev)
    import torch.nn as nn
    net = _DnCNN(channels=1, num_of_layers=17)
    model = nn.DataParallel(net)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def _ensure_resource(rel_name, patterns, is_dir=False):
    """把外部资源定位并拷贝到当前实验目录（已存在则直接返回）"""
    target = os.path.join(SAVE_DIR, rel_name)
    if os.path.isdir(target) if is_dir else os.path.exists(target):
        return target
    ws_root = os.path.dirname(os.path.dirname(os.path.dirname(SAVE_DIR)))
    found = None
    for pat in patterns:
        for cand in glob.glob(os.path.join(ws_root, '**', pat), recursive=True):
            if (os.path.isdir(cand) if is_dir else os.path.isfile(cand)):
                found = cand
                break
        if found:
            break
    if not found:
        return None
    parent_dir = os.path.dirname(target)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    (shutil.copytree if is_dir else shutil.copy)(found, target)
    print(f"[资源] 已从 {found} 拷贝到 {target}")
    return target


# 提前检查运行条件：推理本身CPU即可，缺的只是模型权重/测试图/模块
_skip_step2 = False
if _DnCNN is None:
    print("缺少 sampling_tools（DnCNN结构定义），跳过步骤2（图像去噪Tweedie验证）")
    _skip_step2 = True

if not _skip_step2:
    # 加载图像（自动从工作区搜索cman.png并拷贝到当前目录）
    im_path = _ensure_resource('cman.png', ['cman.png'])
    if im_path is None:
        print("警告: 测试图像 cman.png 未找到，跳过步骤2")
        _skip_step2 = True

_model_paths = {}
if not _skip_step2:
    # 预训练模型（自动从工作区搜索并拷贝到 Pretrained_models/）
    for _mf in ['RealSN_DnCNN_noise5.pth', 'RealSN_DnCNN_noise15.pth', 'RealSN_DnCNN_noise40.pth']:
        _p = _ensure_resource(os.path.join('Pretrained_models', _mf), [_mf])
        if _p is None:
            print(f"警告: 预训练模型 {_mf} 未找到，对应行将标注缺失")
        _model_paths[_mf] = _p

if not _skip_step2:
    im = np.array(Image.open(im_path))
    x = torch.Tensor(im / 255.).to(device)
    if x.dim() == 3:  # PNG可能带alpha通道，取灰度首通道
        x = x[:, :, 0]
    x = x.float()

if not _skip_step2:
    # 测试不同噪声水平
    noise_levels = [5/255, 15/255, 40/255]
    eps_values_img = [(5/255)**2, (15/255)**2, (40/255)**2]
    model_files = ['RealSN_DnCNN_noise5.pth', 'RealSN_DnCNN_noise15.pth', 'RealSN_DnCNN_noise40.pth']

    # 先收集量化指标与图像，再统一排版（避免硬编码定性结论与数据脱节）
    step2_stats = {}
    rows = []
    for noise_lvl, eps_img, model_file in zip(noise_levels, eps_values_img, model_files):
        model_i_path = _model_paths.get(model_file)
        if model_i_path is None or not os.path.exists(model_i_path):
            print(f"  [跳过] 模型 {model_file} 未找到")
            continue
        model_i = _load_denoiser(model_i_path, device)
        # RealSN-DnCNN 输出的是估计噪声 n̂，故干净图像 D_ε(y) = y - n̂(y)
        with torch.no_grad():
            y = x + torch.randn_like(x) * noise_lvl
            denoised = (y - model_i(y[None][None].to(device))[0][0]).detach()

        # 通过Tweedie等式计算得分函数
        score = (denoised - y) / eps_img
        residual = denoised - y
        m_edge, m_flat = edge_flat_masks(x)
        step2_stats[model_file.replace('RealSN_DnCNN_', '').replace('.pth', '')] = {
            'sigma': float(noise_lvl),
            'eps': float(eps_img),
            'score_abs_mean': float(score.abs().mean()),
            'residual_rms': float(residual.pow(2).mean().sqrt()),
            'edge_over_flat': float(score.abs()[m_edge].mean() / score.abs()[m_flat].mean()),
            'psnr_noisy': psnr_db(x, y),
            'psnr_denoised': psnr_db(x, denoised),
        }
        rows.append((y, denoised, score, residual, noise_lvl, eps_img, model_file))

    n_rows = max(len(rows), 1)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for i, (y, denoised, score, residual, noise_lvl, eps_img, model_file) in enumerate(rows):
        axes[i][0].imshow(y.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i][0].set_title(f'含噪 ($\\sigma$={noise_lvl:.3f})')
        axes[i][0].axis('off')

        axes[i][1].imshow(denoised.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i][1].set_title(r'去噪输出 $D_\varepsilon(y)$')
        axes[i][1].axis('off')

        # 得分函数的magnitude（用99.5百分位截断显示，否则hot色标整体饱和看不出结构）
        score_mag = torch.abs(score)
        vmax_pct = float(torch.quantile(score_mag.flatten(), 0.995))
        im2 = axes[i][2].imshow(score_mag.cpu().numpy(), cmap='hot', vmin=0, vmax=max(vmax_pct, 1e-12))
        axes[i][2].set_title(r'$|s_\varepsilon|$ = $|(D_\varepsilon-y)/\varepsilon|$' + f'  (99.5%={vmax_pct:.1f})')
        axes[i][2].axis('off')
        plt.colorbar(im2, ax=axes[i][2], fraction=0.046, pad=0.04)

        # 残差 = eps × 得分
        im3 = axes[i][3].imshow(residual.cpu().numpy(), cmap='RdBu_r')
        axes[i][3].set_title(r'残差 $D_\varepsilon-y = \varepsilon \cdot$ 得分')
        axes[i][3].axis('off')
        plt.colorbar(im3, ax=axes[i][3], fraction=0.046, pad=0.04)

    if len(rows) < n_rows:
        for j in range(4):
            axes[len(rows)][j].axis('off')

    fig.suptitle('图像去噪中的Tweedie等式验证', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤2_图像Tweedie验证.png'), dpi=150, bbox_inches='tight')
    plt.close()

    _keys = list(step2_stats.keys())
    _sm = [step2_stats[k]['score_abs_mean'] for k in _keys]
    _ef = [step2_stats[k]['edge_over_flat'] for k in _keys]
    _gain = [step2_stats[k]['psnr_denoised'] - step2_stats[k]['psnr_noisy'] for k in _keys]

    # 量化面板：把三个定性断言变成可判读的数据
    fig, axes2 = plt.subplots(1, 3, figsize=(15, 4.2))
    axes2[0].bar(_keys, _sm, color='steelblue')
    axes2[0].set_ylabel(r'$\overline{|s_\varepsilon|}$')
    axes2[0].set_title('得分幅值均值 vs 噪声水平')
    axes2[1].bar(_keys, _ef, color='darkorange')
    axes2[1].axhline(1.0, color='k', ls='--', label='边缘=平坦(比值1)')
    axes2[1].set_ylabel('边缘得分均值 / 平坦得分均值')
    axes2[1].set_title('断言“边缘得分更大”是否成立')
    axes2[1].legend(fontsize=8)
    axes2[2].bar(_keys, _gain, color='seagreen')
    axes2[2].set_ylabel('PSNR提升 (dB)')
    axes2[2].set_title('去噪器有效性（D_ε≈MMSE）')
    for ax_ in axes2:
        ax_.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤2_量化指标.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("图像Tweedie验证的量化结果（由实测生成，非预设结论）：")
    for k, v in step2_stats.items():
        print(f"  [{k}] σ={v['sigma']:.4f}, ε={v['eps']:.6f}: "
              f"得分幅值均值={v['score_abs_mean']:.3f}, 残差RMS={v['residual_rms']:.4f}, "
              f"边缘/平坦得分比={v['edge_over_flat']:.3f}, "
              f"PSNR {v['psnr_noisy']:.2f}→{v['psnr_denoised']:.2f} dB")
    print("  - 残差 D_ε(y)-y = ε·得分 按构造恒等，量级见上表；等式的实质检验见步骤1的解析对比")
    print("  - 去噪器质量：PSNR提升越大说明D_ε越接近MMSE去噪器，Tweedie给出的得分越可信")


# ============================================================
# 步骤3：Tweedie等式对PnP的指导意义
# ============================================================
print("\n" + "=" * 60)
print("步骤3：Tweedie等式对PnP框架的核心意义")
print("=" * 60)

print("\n1. 传统ULA需要先验得分:")
print("   ∇log p(x)")
print("   问题：复杂先验的得分函数不可直接计算（归一化常数Z未知）")

print("\n2. Tweedie等式提供桥梁:")
print("   ∇log p_ε(x) = (D_ε(x) - x) / ε")
print("   含义：得分函数 = 去噪器残差 / 噪声方差")

print("\n3. PnP框架的核心思想:")
print("   用学习到的去噪器 D_ε 替换不可计算的先验得分")
print("   X_{k+1} = X_k + δ[∇log p(y|X_k) + (D_ε(X_k)-X_k)/ε] + √(2δ)·Z_{k+1}")

print("\n4. 实现步骤:")
print("   a) 训练去噪器 D_ε（监督学习，数据充足）")
print("   b) Tweedie等式给出 s_ε(x) = (D_ε(x)-x)/ε")
print("   c) 将 s_ε(x) 代入ULA替换先验梯度 → PnP-ULA")

print("\n5. 去噪器成为'隐式先验':")
print("   - 不需要显式指定先验分布的形式")
print("   - 只需要一个能去噪的函数")
print("   - 去噪器学到了数据中的先验知识（如'自然图像长什么样'）")


# ============================================================
# 可视化：去噪器残差 = 噪声方差 × 得分函数
# ============================================================
eps_step3 = 0.5
y_values = np.linspace(-3, 3, 200)

denoised = exact_mmse_denoiser_1d(y_values, eps_step3)
residual = denoised - y_values              # 去噪器残差
score = numerical_score_1d(y_values, eps_step3)  # 数值梯度估计的得分
scaled_score = eps_step3 * score             # 噪声方差 × 得分

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(y_values, residual, 'b-', lw=2, label=r'去噪器残差 $D_\varepsilon^*(y) - y$')
plt.plot(y_values, scaled_score, 'r--', lw=2, label=r'$\varepsilon \cdot \nabla\log p_\varepsilon(y)$')
plt.xlabel('$y$')
plt.ylabel('值')
plt.title(f'Tweedie等式：残差 = $\\varepsilon \\times$ 得分 ($\\varepsilon$={eps_step3})')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
# 可视化去噪器的几何含义
plt.plot(y_values, y_values, 'k--', lw=1, alpha=0.5, label='$y=x$ (无去噪)')
plt.plot(y_values, denoised, 'b-', lw=2, label=r'$D_\varepsilon^*(y) = y/(1+\varepsilon)$')
# 箭头表示去噪方向（残差方向）
for y_val in np.linspace(-3, 3, 7):
    d_val = exact_mmse_denoiser_1d(y_val, eps_step3)
    plt.annotate('', xy=(y_val, d_val), xytext=(y_val, y_val),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
plt.xlabel('含噪观测 $y$')
plt.ylabel('去噪输出')
plt.title('去噪器的收缩效应')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_Tweedie几何含义.png'), dpi=150)
plt.close()

# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.3-1 总结")
print("=" * 60)
_max_err_level = 10 ** np.floor(np.log10(max(_multi_err.values())))
print("1. Tweedie等式的数学正确性：通过数值梯度验证了等式")
print(f"   实测最大绝对误差量级 ~{_max_err_level:.0e}（浮点舍入被 1/(2h) 放大所致，非~1e-5，也非等式偏差）")
print("2. 从去噪器提取得分：得分函数 = 去噪器残差 / 噪声方差")
if _skip_step2:
    print("3. 图像去噪验证：本次未运行步骤2（缺资源），结论仅限1D解析情形")
else:
    print("3. 图像去噪验证：学习去噪器的残差按构造等于ε×得分，量化指标见『步骤2_量化指标.png』")
print("4. PnP的桥梁作用：Tweedie等式是连接去噪与采样的核心数学工具")
print("5. 隐式先验：去噪器学习到的'干净图像长什么样'等价于学习了先验分布 p(x)")


# ===== 保存数值结果 =====
import json

def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
    import numpy as np
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

# 重新计算各噪声水平下的Tweedie等式验证误差
_tweedie_errors = {f"eps_{_eps}": float(f"{_multi_err[float(_eps)]:.3e}") for _eps in eps_values}

# 重新计算eps_step1对应的验证误差（确保数值正确）
_denoised_1 = exact_mmse_denoiser_1d(y_values, eps_step1)
_score_num_1 = numerical_score_1d(y_values, eps_step1)
_score_tweedie_1 = (_denoised_1 - y_values) / eps_step1
_diff_1 = _score_num_1 - _score_tweedie_1
_max_abs_err = float(np.max(np.abs(_diff_1)))
_max_rel_err = float(_max_abs_err / (np.max(np.abs(_score_num_1)) + 1e-10))

results_summary = {
    "实验名称": "实验5.3-1 Tweedie等式验证——从去噪器提取得分",
    "步骤1_1D_Tweedie验证": {
        "噪声方差_eps": round(float(eps_step1), 6),
        "最大绝对误差": float(f"{_max_abs_err:.3e}"),
        "最大相对误差": float(f"{_max_rel_err:.3e}"),
        "多噪声水平验证": _tweedie_errors,
        "说明": "log p_eps严格二次，中心差分无截断误差；残量来自浮点舍入被1/(2h)放大，标度~1/h；解析上两边恒等",
    },
    "步骤3_几何含义": {
        "噪声方差_eps": round(float(eps_step3), 6),
    },
}
if not _skip_step2:
    results_summary["步骤2_图像Tweedie验证"] = {
        "device": str(device),
        "各噪声水平指标": step2_stats,
        "说明": "残差=ε·得分按构造恒等，本表用于核验去噪器有效性与两个定性断言（边缘/噪声趋势）",
    }
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
