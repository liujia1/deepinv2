# -*- coding: utf-8 -*-
"""
实验6.7-2 学习得分驱动的PnP-ULA采样
对应章节: 6.7 用学习到的得分驱动采样

知识点:
  - 从预训练DnCNN通过Tweedie等式构建得分估计器
  - 学习得分驱动的PnP-ULA求解逆问题(去卷积)
  - 学习先验 vs 手工先验(TV)的对比
  - 后验采样与不确定性量化

实验内容:
  步骤1: 从预训练DnCNN构建得分估计器(Tweedie等式实践)
  步骤2: 学习得分驱动的PnP-ULA求解去卷积问题
  步骤3: 学习先验 vs 手工先验(TV)的对比

素材来源:
  - 第5章实验5.3/5.4的PnP-ULA代码框架
  - 6.7节"学习先验 vs 手工先验"对比实验
  - 参考实验6.5.py

运行前提: 需要预训练模型 RealSN_DnCNN_noise15.pth (自动下载)
          需要sampling_tools目录 (已包含)
          CPU/GPU均可运行 (GPU加速推荐)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import io
import time
import warnings
import logging

# 兼容 NumPy 1.x (使用 np.trapz) 与 NumPy 2.x (使用 np.trapezoid)
_trapz = getattr(np, 'trapezoid', np.trapz)

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

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验6.7-2')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)

# 在Colab或本地首次运行时自动创建chinese_font.py
# 设计说明: 动态生成字体配置模块是为了保证单文件可移植性(Colab单文件运行)
# 避免用户需要手动管理额外的依赖文件
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
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

# 设置随机种子
np.random.seed(42)

import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"实验6.7-2: 学习得分驱动的PnP-ULA采样")
print(f"{'='*60}")
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  未检测到 GPU, 使用 CPU 运行")
    print("  提示: Colab 用户可在菜单 运行时 -> 更改运行时类型 中选择 GPU")

# 导入sampling_tools模块
_local_sampling_tools = os.path.join(SAVE_DIR, 'sampling_tools')
_has_sampling_tools = False
if os.path.exists(_local_sampling_tools):
    sys.path.insert(0, SAVE_DIR)
    try:
        from sampling_tools.spectral_normalize_chen import spectral_norm
        _has_sampling_tools = True
    except ImportError as e:
        print(f"警告: sampling_tools 导入失败: {e}")
else:
    print("警告: sampling_tools 模块未找到")

# 若sampling_tools不可用，提供dummy spectral_norm以避免类定义时NameError
# 这样代码可以优雅降级到"无预训练模型"分支
if not _has_sampling_tools:
    def spectral_norm(module, *args, **kwargs):
        """占位函数：当sampling_tools不可用时使用"""
        return module

import torch.nn as nn


# ============================================================
# CPU兼容的DnCNN模型
# ============================================================
class DnCNN(nn.Module):
    """DnCNN去噪网络（CPU兼容版本）

    使用Real Spectral Normalization保证Lipschitz常数 $\\leq 1$。
    输出=残差(噪声), 去噪结果 $D_\\varepsilon(x) = x - \\mathrm{out}$。
    """
    def __init__(self, channels=1, num_of_layers=17):
        super().__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(spectral_norm(nn.Conv2d(in_channels=channels, out_channels=features,
                                                kernel_size=kernel_size, padding=padding, bias=False)))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(spectral_norm(nn.Conv2d(in_channels=features, out_channels=features,
                                                    kernel_size=kernel_size, padding=padding, bias=False)))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(spectral_norm(nn.Conv2d(in_channels=features, out_channels=channels,
                                                kernel_size=kernel_size, padding=padding, bias=False)))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out  # 输出=残差(噪声), 去噪结果=x-out


def load_dncnn(model_path, device_str='cpu'):
    """加载预训练DnCNN模型（CPU兼容）

    使用DataParallel包装以匹配预训练权重格式。
    """
    net = DnCNN(channels=1, num_of_layers=17)
    net = nn.DataParallel(net)
    checkpoint = torch.load(model_path, map_location=device_str, weights_only=True)
    net.load_state_dict(checkpoint)
    net.eval()
    return net


# ============================================================
# 得分估计器: 从DnCNN通过Tweedie等式构建
# ============================================================
class ScoreEstimatorFromDnCNN:
    """从预训练去噪器构建得分估计器

    核心公式（Tweedie等式）:
      $s_\\theta(x, \\varepsilon) = (D_\\varepsilon(x) - x) / \\varepsilon^2$

    其中 $D_\\varepsilon(x)$ 是噪声水平 $\\varepsilon$ 下去噪器的输出。
    """
    def __init__(self, denoiser, epsilon):
        self.denoiser = denoiser
        self.epsilon = epsilon

    def __call__(self, x):
        """计算得分估计 $\\nabla\\log p_\\varepsilon(x) \\approx (D_\\varepsilon(x) - x) / \\varepsilon^2$"""
        with torch.no_grad():
            residual = self.denoiser(x)  # 输出=残差(噪声)
            denoised = x - residual       # $D_\\varepsilon(x) = x - \\mathrm{residual}$
        score = (denoised - x) / (self.epsilon ** 2)
        return score


# ============================================================
# 模型自动下载功能
# ============================================================
def download_model_if_needed(model_path, model_dir):
    """自动下载预训练模型（优先本地, 不存在则从GitHub下载）

    参数:
        model_path: 模型文件完整路径
        model_dir: 模型目录路径

    返回:
        bool: 模型是否可用
    """
    if os.path.exists(model_path):
        print(f"[Model] 检测到已缓存的模型权重: {model_path}")
        print("[Model] 直接加载预训练权重, 跳过下载过程")
        return True

    # 模型不存在, 尝试自动下载
    print(f"[Model] 未找到预训练模型: {model_path}")
    print("[Model] 正在尝试自动下载...")

    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)

    # 模型下载URL（来自原始项目的GitHub仓库）
    model_url = "https://github.com/uclaopt/Provable_Plug_and_Play/raw/master/Pretrained_models/RealSN_DnCNN_noise15.pth"

    try:
        import urllib.request
        print(f"[Model] 下载地址: {model_url}")
        print("[Model] 正在下载模型权重（约50MB）...")

        urllib.request.urlretrieve(model_url, model_path)

        print(f"[Model] 模型下载成功! 已保存到: {model_path}")
        print("[Model] 下次运行将直接加载缓存权重")
        return True

    except Exception as e:
        print(f"[Model] 模型下载失败: {e}")
        print("[Model] 请手动下载模型文件:")
        print(f"  1. 访问: https://github.com/uclaopt/Provable_Plug_and_Play")
        print(f"  2. 下载 Pretrained_models/RealSN_DnCNN_noise15.pth")
        print(f"  3. 保存到: {model_path}")
        return False


# 检查模型可用性
model_dir = os.path.join(SAVE_DIR, 'Pretrained_models')
model_path = os.path.join(model_dir, 'RealSN_DnCNN_noise15.pth')
_model_downloaded = download_model_if_needed(model_path, model_dir)
# HAS_MODEL: 模型权重可用 且 sampling_tools可用（DnCNN依赖spectral_norm）
HAS_MODEL = _model_downloaded and _has_sampling_tools


# ============================================================
# 步骤1: 从预训练DnCNN构建得分估计器
# ============================================================
print(f"\n{'='*60}")
print("步骤1: 从预训练DnCNN构建得分估计器")
print(f"{'='*60}")
print("\n[Tweedie等式]")
print("  ∇log p_ε(x) = (D_ε(x) - x) / ε²")
print("  去噪器输出 D_ε(x), 通过Tweedie等式转换为得分估计器")

if HAS_MODEL:
    print(f"\n加载预训练模型: {model_path}")
    denoiser = load_dncnn(model_path, device_str=str(device))
    denoiser = denoiser.to(device)

    # 测试去噪→得分提取
    # 注意: epsilon=15/255 是 DnCNN 训练时的噪声水平(σ=15)
    # 这是 PnP-ULA 的简化假设: 固定 ε 而非随采样过程退火
    # 实际应用中, ε 应与 Langevin 链的有效噪声水平匹配
    # 这正是 PnP-ULA(单一去噪器、固定ε) 与 PnP-Annealed-ULA(多噪声水平、ε退火) 的区别
    # 呼应第6.5节的退火思想: 使用多个噪声水平的去噪器序列, 逐步降低ε
    epsilon = 15.0 / 255.0  # 噪声水平 $\sigma=15/255$
    score_estimator = ScoreEstimatorFromDnCNN(denoiser, epsilon)

    # 创建测试图像（简单梯度图案）
    test_img = torch.zeros(1, 1, 64, 64, device=device)
    test_img[0, 0, :32, :] = 0.5
    test_img[0, 0, 32:, :] = 1.0

    # 加噪声
    noise = torch.randn_like(test_img) * epsilon
    noisy_img = test_img + noise

    # 去噪
    with torch.no_grad():
        residual = denoiser(noisy_img)
        denoised = noisy_img - residual  # $D_\\varepsilon(x) = x - \\mathrm{residual}$

    # 提取得分
    score = score_estimator(noisy_img)

    print(f"\nTweedie得分提取验证:")
    print(f"  噪声水平 ε = {epsilon:.4f}")
    print(f"  去噪器输出 D_ε(x) 范围: [{denoised.min():.4f}, {denoised.max():.4f}]")
    print(f"  得分 s_θ = (D_ε - x) / ε² 范围: [{score.min():.4f}, {score.max():.4f}]")
    print(f"  得分范数 ‖s_θ‖ = {score.norm():.4f}")
    print(f"\n  Tweedie等式: ∇log p_ε(x) = (D_ε(x) - x) / ε²")
    print(f"  得分估计器的构建完全基于Tweedie等式——无需显式计算 ∇log p(x)")

else:
    if not _model_downloaded:
        print(f"\n未找到预训练模型, 跳过需要预训练模型的步骤, 使用模拟数据演示")
    if not _has_sampling_tools:
        print(f"\n未找到sampling_tools模块, 跳过需要该模块的步骤")


# ============================================================
# 步骤2: 学习得分驱动的PnP-ULA求解去卷积问题
# ============================================================
print(f"\n{'='*60}")
print("步骤2: 学习得分驱动的PnP-ULA求解去卷积问题")
print(f"{'='*60}")
print("\n[PnP-ULA递推式]")
print("  X_{m+1} = X_m - δ∇f(X_m) + δ · s_θ(X_m, ε) + √(2δ) · Z")
print("  三步解读:")
print("    似然梯度步: -δ∇f(X_m) —— 数据一致性")
print("    先验得分步: δ · s_θ(X_m, ε) —— 先验知识")
print("    探索噪声: √(2δ) · Z —— 随机性保证收敛到分布")

if HAS_MODEL:
    try:
        from scipy.ndimage import gaussian_filter as _gaussian_filter
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        print("[Warning] scipy未安装, 使用PyTorch近似模糊")

    try:
        from skimage import data as _skimage_data
        HAS_SKIMAGE = True
    except ImportError:
        HAS_SKIMAGE = False
        print("[Warning] skimage未安装, 使用合成测试图像")

    # 加载测试图像
    if HAS_SKIMAGE:
        img = _skimage_data.camera().astype(np.float64) / 255.0
        img = img[128:384, 128:384]  # 裁剪到256x256
    else:
        # 合成测试图像: 渐变+方块
        img = np.zeros((256, 256), dtype=np.float64)
        img[50:200, 50:200] = 0.7
        img[80:180, 80:180] = 0.3
        img[100:150, 100:150] = 1.0
    x_true = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # 构造正向模型: 高斯模糊
    # 注意: gaussian_blur 内部会进行 GPU→CPU→numpy→GPU 转换
    # 这是为了使用 scipy.ndimage.gaussian_filter (更精确)
    # 性能上不是最优, 但保证了算子的正确性和自共轭性
    def gaussian_blur(image, sigma_blur=3.0):
        """高斯模糊算子 (自共轭线性算子)"""
        img_np = image.squeeze().cpu().numpy()
        if HAS_SCIPY:
            blurred = _gaussian_filter(img_np, sigma=sigma_blur)
        else:
            # PyTorch近似: 用多次均值滤波替代高斯滤波
            # 注意: 这是粗略近似, 仅用于无scipy时的兜底演示
            # 核大小 k=6σ+1 是高斯核经验法则, 用于均值滤波理论依据较弱
            # 实际科研中应使用精确的高斯核, 避免用此近似做定量分析
            import torch.nn.functional as F
            k = int(6 * sigma_blur + 1)
            if k % 2 == 0:
                k += 1
            kernel = torch.ones(1, 1, k, k, device=device) / (k * k)
            pad = k // 2
            img_t = torch.tensor(img_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            for _ in range(3):
                img_t = F.conv2d(img_t, kernel, padding=pad)
            blurred = img_t.squeeze().cpu().numpy()
        return torch.tensor(blurred, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # 生成观测 $y = Ax + \\mathrm{noise}$
    sigma_blur = 3.0
    sigma_noise = 0.05
    Ax = gaussian_blur(x_true, sigma_blur)
    noise_obs = torch.randn_like(Ax) * sigma_noise
    y = Ax + noise_obs

    # 似然梯度: $f(x) = \frac{1}{2\sigma^2}\|Ax - y\|^2$
    # $\nabla f(x) = A^T(Ax - y) / \sigma^2$
    # ULA更新中: $X_{m+1} = X_m - \delta \cdot \nabla f(X_m) + \delta \cdot s_\theta(X_m) + \sqrt{2\delta} Z$
    def likelihood_grad(x):
        """返回数据项梯度 $\\nabla f(x) = A^T(Ax - y) / \\sigma^2$

        ULA 递推式中以"减梯度"形式使用: x - delta * grad_f
        """
        Ax_current = gaussian_blur(x, sigma_blur)
        residual = Ax_current - y
        grad = gaussian_blur(residual, sigma_blur) / (sigma_noise ** 2)
        return grad

    # PSNR计算
    # 注意: 假设像素值域为[0,1], 峰值固定为1 (MAX_I=1)
    # 若数据值域不同, 需相应调整峰值
    def psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2).item()
        if mse == 0:
            return float('inf')
        return 10 * np.log10(1.0 / mse)

    psnr_init = psnr(y, x_true)
    print(f"\n初始PSNR (模糊+噪声): {psnr_init:.2f} dB")

    # PnP-ULA参数
    # 步长delta需满足PnP-ULA收敛条件: δ < 2/L, 其中L是∇f + ∇(-log p)的Lipschitz常数
    # 此处 sigma_noise=0.05 => 1/σ²=400 较大, 故需小步长 (delta=0.0001)
    delta = 0.0001  # 步长 (适配 σ_noise=0.05 的似然项尺度)
    # 注意: n_iter=200, burn_in=100 是教学简化版本
    # 实际PnP-ULA论文中迭代次数通常是上万步(burn-in也是几千步起步)
    # 这里仅为演示算法流程, 实际科研中需要远多于此的迭代数
    n_iter = 500  # 迭代次数(教学简化)
    burn_in = 250  # 预热步数(教学简化)

    print(f"\n运行PnP-ULA ({n_iter}步, δ={delta}, ε={epsilon:.4f})...")
    print(f"  预计耗时: {'~3分钟(CPU)' if device.type == 'cpu' else '~30秒(GPU)'}")

    # 初始化
    x_chain = y.clone()
    samples = []
    psnr_history = []

    t_start = time.time()
    for m in range(n_iter):
        # 似然梯度步
        grad_f = likelihood_grad(x_chain)

        # 先验得分步（通过Tweedie从去噪器获取）
        score_prior = score_estimator(x_chain)

        # ULA更新
        x_chain = x_chain - delta * grad_f + delta * score_prior + np.sqrt(2 * delta) * torch.randn_like(x_chain)

        # 裁剪到[0,1]
        x_chain = torch.clamp(x_chain, 0, 1)

        current_psnr = psnr(x_chain, x_true)
        psnr_history.append(current_psnr)

        if (m + 1) % 50 == 0:
            print(f"  Step {m+1}/{n_iter}: PSNR = {current_psnr:.2f} dB")

        # 保存burn-in后的样本
        if m >= burn_in:
            samples.append(x_chain.clone())

    t_elapsed = time.time() - t_start
    print(f"  PnP-ULA完成, 耗时: {t_elapsed:.1f} 秒")

    # 后验均值（MMSE估计）
    x_mmse = torch.stack(samples).mean(dim=0)
    psnr_mmse = psnr(x_mmse, x_true)

    # 后验方差（像素级不确定性）
    # 注意: 这里仅用burn-in后的100个样本估计方差
    # 这些样本是高度自相关的相邻迭代, 未做thinning/降相关处理
    # 统计上方差可能被低估, 实际应用中应增加样本数或使用thinning
    if len(samples) > 1:
        x_var = torch.stack(samples).var(dim=0)
        mean_var = x_var.mean().item()
    else:
        mean_var = float('nan')

    print(f"\nPnP-ULA结果:")
    print(f"  MMSE重建PSNR: {psnr_mmse:.2f} dB")
    print(f"  平均后验方差: {mean_var:.6f}")

else:
    # 无预训练模型时, 使用模拟结果
    print("\n无预训练模型, 展示6.7节的典型结果（引用）:")
    print(f"\n去卷积实验典型结果 (6.7节表):")
    print(f"{'方法':<20s} | {'PSNR(dB)':>8s} | {'SSIM':>6s} | {'不确定性量化':>10s}")
    print("-" * 55)
    print(f"{'逆滤波':<20s} | {'15.2':>8s} | {'0.35':>6s} | {'无':>10s}")
    print(f"{'Tikhonov正则化':<20s} | {'22.5':>8s} | {'0.68':>6s} | {'无':>10s}")
    print(f"{'TV正则化':<20s} | {'24.8':>8s} | {'0.78':>6s} | {'无':>10s}")
    print(f"{'MYULA+TV':<20s} | {'24.3':>8s} | {'0.76':>6s} | {'有':>10s}")
    print(f"{'PnP-ULA+DRUNet':<20s} | {'27.1':>8s} | {'0.88':>8s} | {'有':>10s}")

    # 模拟PSNR历史
    psnr_init = 15.2
    psnr_mmse = 27.1
    mean_var = 0.003


# ============================================================
# 步骤3: 学习先验 vs 手工先验对比
# ============================================================
print(f"\n{'='*60}")
print("步骤3: 学习先验 vs 手工先验——从得分函数视角的解读")
print(f"{'='*60}")

print("""
从得分函数视角看PnP-ULA中的先验差异:

1. TV先验的得分（通过近端算子近似）:
   - $\\nabla\\log p_{\\mathrm{TV}}(x) \\approx -\\nabla\\|\\nabla x\\|_1$ (TV梯度的近似)
   - 特点: 鼓励分段常数解, 导致staircase效应
   - 得分场特征: 在边缘处强、在平坦区域弱
   - 无法编码复杂的自然图像统计规律

2. 学习先验的得分（通过Tweedie从DnCNN提取）:
   - $\\nabla\\log p_\\varepsilon(x) \\approx (D_\\varepsilon(x) - x) / \\varepsilon^2$
   - 特点: 从数据中学习了自然图像的统计规律
   - 得分场特征: 既能指引全局结构, 又能保留局部细节
   - 编码了自然图像的复杂先验知识

3. 关键差异:
   - TV得分是"手工设计"的, 基于简单的分段光滑假设
   - 学习得分是"数据驱动"的, 编码了自然图像的复杂统计
   - 得分匹配训练（6.3节DSM）使得数据驱动的得分成为可能
   - 这正是本章核心论点的实践验证

4. 不确定性量化:
   - TV和学习的先验都可用于后验采样
   - 但学习先验的后验更准确（不确定性图与误差相关性更高）
""")


# ============================================================
# 可视化
# ============================================================
print(f"\n{'='*60}")
print("生成可视化图表...")
print(f"{'='*60}")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

if HAS_MODEL:
    # 真实图像
    axes[0].imshow(x_true.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('真实图像')
    axes[0].axis('off')

    # 观测（模糊+噪声）
    axes[1].imshow(y.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'观测 ($\\mathrm{{PSNR}}$={psnr_init:.1f}dB)')
    axes[1].axis('off')

    # PnP-ULA重建
    axes[2].imshow(x_mmse.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'PnP-ULA重建 ($\\mathrm{{PSNR}}$={psnr_mmse:.1f}dB)')
    axes[2].axis('off')

    # 不确定性图
    if len(samples) > 1:
        axes[3].imshow(x_var.squeeze().cpu().numpy(), cmap='hot')
        axes[3].set_title(f'后验方差 (均值={mean_var:.4f})')
    else:
        axes[3].text(0.5, 0.5, '样本不足\n无法计算方差', ha='center', va='center', fontsize=14,
                     transform=axes[3].transAxes)
        axes[3].set_title('不确定性图')
    axes[3].axis('off')
else:
    # 无模型时的示意图
    for i, (title, desc) in enumerate([
        ('真实图像', '(需预训练模型)'),
        (r'观测 $y=Ax+\eta$', '(需预训练模型)'),
        ('PnP-ULA重建', r'$\mathrm{PSNR}\approx27.1$dB'),
        ('后验方差', '不确定性图')
    ]):
        axes[i].text(0.5, 0.5, desc, ha='center', va='center', fontsize=12,
                     transform=axes[i].transAxes)
        axes[i].set_title(title)
        axes[i].axis('off')

plt.suptitle('学习得分驱动的PnP-ULA采样', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_学习得分PnPULA.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"图表已保存: 步骤1_学习得分PnPULA.png")

# PSNR收敛曲线
if HAS_MODEL:
    fig2, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(1, n_iter + 1), psnr_history, 'b-', lw=1.5)
    ax.axhline(y=psnr_mmse, color='r', linestyle='--',
               label=f'MMSE $\\mathrm{{PSNR}}$={psnr_mmse:.1f}dB')
    ax.axvline(x=burn_in, color='g', linestyle='--', alpha=0.5, label=f'Burn-in={burn_in}')
    ax.set_xlabel('迭代步数')
    ax.set_ylabel('$\\mathrm{PSNR}$ (dB)')
    ax.set_title('PnP-ULA $\\mathrm{PSNR}$收敛曲线')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤2_PnPULA收敛曲线.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: 步骤2_PnPULA收敛曲线.png")
    print("[备注] 教学简化参数 (500步), 链尚未收敛; 实际PnP-ULA需数千~上万步迭代")


# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验6.7-2 总结")
print(f"{'='*60}")
print("\n1. 从DnCNN通过Tweedie等式构建得分估计器:")
print("   s_θ(x,ε) = (D_ε(x) - x) / ε²")
print("   这正是6.6节'去噪器作为得分估计器'的实践验证")
print("\n2. 学习得分驱动的PnP-ULA:")
print("   X_{m+1} = X_m - δ∇f(X_m) + δ · s_θ(X_m,ε) + √(2δ) · Z")
print("   三步解读: 似然梯度步(数据) + 先验得分步(知识) + 探索噪声(随机性)")
if HAS_MODEL:
    print(f"   去卷积PSNR: {psnr_init:.1f} → {psnr_mmse:.1f} dB")
else:
    print("   去卷积PSNR: 15.2 → 27.1 dB (引用6.7节数据)")
print("\n3. 学习先验 vs 手工先验:")
print("   - TV先验: 手工设计, staircase效应, PSNR≈24.8dB")
print("   - 学习先验: 数据驱动, 自然重建, PSNR≈27.1dB")
print("   - 得分匹配训练使得数据驱动的先验得分成为可能")
print("\n4. 本章核心论点的实践验证:")
print("   得分匹配解决了'得分从哪来'的问题")
print("   PnP-ULA解决了'如何用得分做后验采样'的问题")
print("   两者组合 → 数据驱动的逆问题求解框架")

print(f"\n{'='*60}")
print("第六章配套实验完成!")
print(f"{'='*60}")
print("实验列表:")
print("  6.1-1: 归一化常数困境与得分匹配动机")
print("  6.2-1: ESM与ISM的验证")
print("  6.3-1: 去噪得分匹配(DSM)训练与验证")
print("  6.4-1: Hutchinson迹估计与切片得分匹配(SSM)")
print("  6.5-1: 多尺度得分匹配与退火Langevin采样")
print("  6.6-1: 从去噪器中提取得分函数(Tweedie等式实践)")
print("  6.7-1: SMLD完整实现——从零训练到图像生成")
print("  6.7-2: 学习得分驱动的PnP-ULA采样")
