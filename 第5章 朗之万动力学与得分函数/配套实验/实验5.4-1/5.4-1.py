# -*- coding: utf-8 -*-
"""
实验5.4-1 近端算子 vs 学习去噪器：PnP中的先验替换
对应章节：5.4（MAP与MMSE的结构对偶性）
知识点：
  - 近端算子与去噪器的结构对偶性
  - 显式先验（TV）vs 隐式先验（学习去噪器）
  - MAP方向 vs MMSE方向

素材来源：
  - Mathematics.../Teaching Unit 2/labs/lab2_PnP_sol.ipynb（PnP-ULA、去噪器）
  - sampling_tools/chambolle_prox_TV.py（TV近端算子，Chambolle投影算法）

核心思想：
  - 近端算子（MAP方向）：$\\mathrm{prox}_{\\lambda R}(y) = \\arg\\min_x \\{R(x) + \\|x-y\\|^2/(2\\lambda)\\}$
  - 去噪器（MMSE方向）：$D_\\varepsilon^*(y) = E[x|y] = y - \\varepsilon\\nabla\\log p_\\varepsilon(y)$
  - 结构对偶：两者形式相同 $y - c\\cdot\\nabla(\\cdot)$，但目标不同（众数 vs 均值）

运行前提：
  步骤1：无特殊要求，CPU即可运行
  步骤2-4：需要GPU和预训练RealSN-DnCNN模型
    - sampling_tools/ 已包含在当前目录
    - Pretrained_models/ 需要用户自行准备（包含RealSN_DnCNN_noise5.pth）
    - 若无GPU或缺少模型，步骤2-4将被跳过，仅运行步骤1
"""

import math
import torch
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
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
    SAVE_DIR = os.path.join(_gdrive, '实验5.4-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)

    # 在Colab中自动创建chinese_font.py
    _chinese_font_path = os.path.join(_chinese_path, 'chinese_font.py')
    if not os.path.exists(_chinese_font_path):
        print("正在创建中文字体配置模块...")
        _chinese_font_code = '''# -*- coding: utf-8 -*-
"""
中文显示支持模块 - 兼容 Windows / Linux / Colab

使用方法：
    from chinese_font import setup_chinese_font
    setup_chinese_font()
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
    """自动检测系统中可用的中文字体，兼容 Windows / Linux / Colab"""
    candidates = []
    if platform.system() == 'Windows':
        candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    else:
        candidates = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
            'Noto Sans CJK SC', 'Noto Sans CJK',
            'Source Han Sans SC', 'AR PL UMing CN', 'SimHei',
        ]
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
    """
    设置中文字体支持

    参数:
        save_dir: 字体缓存目录（可选，默认使用模块所在目录）

    返回:
        str: 检测到的中文字体名称，或 None
    """
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
            _cn_font = 'Noto Sans SC'
            print(f"[Font] 已加载缓存字体: {_cn_font}")
            return _cn_font
        else:
            try:
                import urllib.request
                print(f"[Font] 正在下载中文字体 NotoSansSC...")
                urllib.request.urlretrieve(_font_url, _font_file)
                from matplotlib.font_manager import fontManager
                fontManager.addfont(_font_file)
                plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
                plt.rcParams['font.family'] = 'sans-serif'
                _cn_font = 'Noto Sans SC'
                print(f"[Font] 已下载并注册中文字体: {_cn_font}")
                return _cn_font
            except Exception as e:
                print(f"[Font] 字体下载失败: {e}，中文可能显示为方框")
    else:
        print("[Font] 未找到中文字体，中文可能显示为方框")

    return None


__all__ = ['setup_chinese_font']
'''
        with open(_chinese_font_path, 'w', encoding='utf-8') as f:
            f.write(_chinese_font_code)
        print(f"[Font] 已创建字体配置模块: {_chinese_font_path}")
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
torch.manual_seed(42)

# 导入sampling_tools模块
_local_sampling_tools = os.path.join(SAVE_DIR, 'sampling_tools')
if os.path.exists(_local_sampling_tools):
    sys.path.insert(0, SAVE_DIR)
    try:
        from sampling_tools import *
        _has_sampling_tools = True
    except ImportError as e:
        print(f"警告: sampling_tools 导入失败: {e}")
        _has_sampling_tools = False
else:
    _has_sampling_tools = False
    if _IN_COLAB:
        print("\n" + "=" * 60)
        print("Colab环境提示")
        print("=" * 60)
        print("  sampling_tools 模块未找到")
        print(f"  请确保已将整个实验目录上传到 Google Drive:")
        print(f"  路径: {_gdrive}/实验5.4-1/")
        print("  需要上传的文件:")
        print("    - 5.4-1.py")
        print("    - sampling_tools/ (整个目录)")
        print("    - cman.png")
        print("    - .chinese/ (可选，会自动创建)")
        print("=" * 60)
    else:
        print("警告: sampling_tools 模块未找到")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Colab环境GPU提示
if _IN_COLAB and device.type == 'cpu':
    print("\n" + "=" * 60)
    print("Colab GPU 启用提示")
    print("=" * 60)
    print("  当前未检测到GPU，建议启用GPU以加速运行")
    print("  启用方法：")
    print("    1. 点击菜单栏 '运行时' -> '更改运行时类型'")
    print("    2. 在'硬件加速器'中选择 'GPU'")
    print("    3. 点击'保存'，运行时会重新启动")
    print("    4. 重新运行代码")
    print("=" * 60)

# 检查运行条件
_has_gpu = device.type == 'cuda'
_model_dir = os.path.join(SAVE_DIR, 'Pretrained_models')
_model_path = os.path.join(_model_dir, 'RealSN_DnCNN_noise5.pth')

# 模型自动下载功能
def download_model_if_needed(model_path, model_dir):
    """
    自动下载预训练模型（如果不存在）

    参数:
        model_path: 模型文件完整路径
        model_dir: 模型目录

    返回:
        bool: 模型是否可用
    """
    if os.path.exists(model_path):
        print(f"[Model] 检测到已缓存的模型权重: {model_path}")
        print("[Model] 直接加载预训练权重，跳过训练过程")
        return True

    # 模型不存在，尝试自动下载
    print(f"[Model] 未找到预训练模型: {model_path}")
    print("[Model] 正在尝试自动下载...")

    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)

    # 模型下载URL（来自原始项目的GitHub仓库）
    # 注意：这里需要用户提供正确的下载链接
    model_url = "https://github.com/uclaopt/Provable_Plug_and_Play/raw/master/Pretrained_models/RealSN_DnCNN_noise5.pth"

    try:
        import urllib.request
        print(f"[Model] 下载地址: {model_url}")
        print("[Model] 正在下载模型权重（约50MB）...")

        # 下载模型
        urllib.request.urlretrieve(model_url, model_path)

        print(f"[Model] 模型下载成功！已保存到: {model_path}")
        print("[Model] 下次运行将直接加载缓存权重")
        return True

    except Exception as e:
        print(f"[Model] 模型下载失败: {e}")
        print("[Model] 请手动下载模型文件:")
        print(f"  1. 访问: https://github.com/uclaopt/Provable_Plug_and_Play")
        print(f"  2. 下载 Pretrained_models/RealSN_DnCNN_noise5.pth")
        print(f"  3. 保存到: {model_path}")
        return False

# 检查模型可用性
_has_model = download_model_if_needed(_model_path, _model_dir)

if not (_has_gpu and _has_sampling_tools and _has_model):
    print("\n" + "=" * 60)
    print("运行环境检查")
    print("=" * 60)
    print(f"  GPU可用: {_has_gpu}")
    print(f"  sampling_tools可用: {_has_sampling_tools}")
    print(f"  预训练模型可用: {_has_model}")
    print("\n  缺少必要资源，将仅运行步骤1（TV近端算子演示）")
    print("  步骤2-4需要GPU和预训练模型")
    print("=" * 60)


# ============================================================
# 步骤1：TV近端算子（显式先验）演示
# 使用 sampling_tools/chambolle_prox_TV.py
# ============================================================
print("=" * 60)
print("步骤1：TV近端算子（显式先验）演示")
print("=" * 60)

# 创建测试图像（简单形状）
test_image = np.zeros((64, 64))
test_image[10:20, 10:20] = 1.0  # 正方形
test_image[40:50, 40:50] = 0.8  # 另一个正方形
test_image_t = torch.from_numpy(test_image).float().to(device)

# 添加噪声
noisy_image_t = test_image_t + 0.2 * torch.randn_like(test_image_t)

# 测试不同$\\lambda$值的TV近端算子
lambda_values = [0.01, 0.05, 0.1, 0.5]

plt.figure(figsize=(15, 4))

plt.subplot(1, len(lambda_values)+2, 1)
plt.imshow(test_image, cmap='gray', vmin=0, vmax=1)
plt.title('原始图像')
plt.axis('off')

plt.subplot(1, len(lambda_values)+2, 2)
plt.imshow(noisy_image_t.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
plt.title('含噪图像')
plt.axis('off')

for i, lambda_ in enumerate(lambda_values):
    result = chambolle_prox_TV(noisy_image_t, device, {'lambda': lambda_, 'maxiter': 200})
    plt.subplot(1, len(lambda_values)+2, i+3)
    plt.imshow(result.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    plt.title(r'TV近端 ($\lambda$={})'.format(lambda_))
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_TV近端算子.png'), dpi=150)
plt.close()

print("TV近端算子说明：")
print(r"  $\lambda$小（0.01）：近端算子接近输入，TV正则化弱，保留更多噪声")
print(r"  $\lambda$大（0.5）：近端算子趋向于常数，TV正则化强，图像被过度平滑")
print(r"  适当$\lambda$：平衡去噪与保真度")


# ============================================================
# 步骤2：加载图像与去噪器，准备对比实验
# ============================================================
if _has_gpu and _has_sampling_tools and _has_model:
    print("\n" + "=" * 60)
    print("步骤2：显式先验（TV） vs 隐式先验（学习去噪器）对比")
    print("=" * 60)

    # 加载图像
    im = np.array(Image.open(os.path.join(SAVE_DIR, "cman.png")))
    x = torch.Tensor(im/255.).to(device)

    # 模糊算子
    kernel_len = [5,5]
    size = [im.shape[0],im.shape[1]]
    type_blur = "uniform"
    A, AT, AAT_norm = blur_operators(kernel_len, size, type_blur, device)

    # 含噪观测
    y0 = A(x)
    BSNRdb = 40
    sigma = torch.linalg.matrix_norm(A(x)-torch.mean(A(x)), ord='fro')/math.sqrt(torch.numel(x)*10**(BSNRdb/10))
    y = y0 + sigma * torch.randn_like(x)

    # 似然
    gradf = lambda x,A,AT : AT(A(x)-y)/sigma**2
    L_y = AAT_norm/(sigma**2)

    # 去噪器（隐式先验）
    L_net = 1.0
    model = load_model(_model_path, device)
    denoise = lambda x: (x - model(x[None][None].to(device))[0][0]).detach()

    eps = (5/255)**2

    projbox = lambda x: torch.clamp(x, min=0, max=1)


    # ============================================================
    # 步骤2a：TV-ULA（显式TV先验采样）
    # 使用 chambolle_prox_TV 提供TV近端，再通过Moreau包络梯度获得TV梯度
    # Moreau包络梯度: $\\nabla R_\\lambda(y) = (y - \\mathrm{prox}_{\\lambda R}(y)) / \\lambda$
    # ============================================================
    def tv_ula(y, A, AT, sigma, niter, delta, lambda_tv, device):
        """
        TV-ULA：使用TV先验的ULA采样
        TV梯度通过Moreau包络的梯度获取：
          $\\nabla \\mathrm{TV}_\\lambda(x) = (x - \\mathrm{prox}_{\\lambda \\mathrm{TV}}(x)) / \\lambda$
        """
        X = y.clone()
        post_sum = torch.zeros_like(X)
        post_sqr = torch.zeros_like(X)

        for k in tqdm(range(niter), desc="TV-ULA"):
            # 似然梯度
            grad_likelihood = gradf(X, A, AT)

            # TV梯度（通过Moreau包络）
            prox_tv = chambolle_prox_TV(X, device, {'lambda': lambda_tv, 'maxiter': 50})
            grad_tv = (X - prox_tv) / lambda_tv  # $\\nabla R_\\lambda(x) = (x - \\mathrm{prox}) / \\lambda$

            # ULA更新
            noise = torch.randn_like(X) * math.sqrt(2 * delta)
            X = X - delta * (grad_likelihood + grad_tv) + noise

            # 投影到[0,1]
            X = projbox(X)

            # 累计统计量
            if k >= niter // 2:
                post_sum += X
                post_sqr += X**2

        n_samples = niter - niter // 2
        post_mean = post_sum / n_samples
        post_var = post_sqr / n_samples - post_mean**2
        post_var = torch.clamp(post_var, min=0)

        return X, post_mean, torch.sqrt(post_var)


    # ============================================================
    # 步骤2b：PnP-ULA（隐式学习先验采样）
    # 使用Tweedie等式从去噪器提取得分函数
    # ============================================================
    def pnp_ula(y, A, AT, sigma, denoiser, niter, delta, eps, device):
        """
        PnP-ULA：使用学习去噪器的ULA采样
        先验得分通过Tweedie等式获取：
          $\\nabla\\log p_\\varepsilon(x) = (D_\\varepsilon(x) - x) / \\varepsilon$
        """
        X = y.clone()
        post_sum = torch.zeros_like(X)
        post_sqr = torch.zeros_like(X)

        for k in tqdm(range(niter), desc="PnP-ULA"):
            # 似然梯度
            grad_likelihood = gradf(X, A, AT)

            # 先验得分（Tweedie替换）
            score_prior = (denoiser(X) - X) / eps  # $\\nabla\\log p_\\varepsilon(x)$

            # ULA更新
            noise = torch.randn_like(X) * math.sqrt(2 * delta)
            X = X - delta * (grad_likelihood - score_prior) + noise

            # 投影到[0,1]
            X = projbox(X)

            # 累计统计量
            if k >= niter // 2:
                post_sum += X
                post_sqr += X**2

        n_samples = niter - niter // 2
        post_mean = post_sum / n_samples
        post_var = post_sqr / n_samples - post_mean**2
        post_var = torch.clamp(post_var, min=0)

        return X, post_mean, torch.sqrt(post_var)


    # ============================================================
    # 运行对比实验
    # ============================================================
    niter = 500    # 原始10000，此处缩小以便快速演示
    lambda_tv = 0.05

    # PnP-ULA步长
    delta_pnp = 0.99 / (L_net/eps + L_y)

    # TV-ULA步长（需要考虑TV梯度的Lipschitz常数，此处简化）
    delta_tv = 0.99 / (1.0/lambda_tv + L_y)

    print(f"\n实验参数:")
    print(rf"  TV-ULA: $\lambda_{{\mathrm{{TV}}}}$={lambda_tv}, $\delta$={delta_tv:.6f}")
    print(rf"  PnP-ULA: $\varepsilon$={eps:.6f}, $\delta$={delta_pnp:.6f}")
    print(f"  迭代次数: {niter}")

    print(f"\n运行TV-ULA...")
    x_tv_final, x_tv_mean, x_tv_std = tv_ula(y, A, AT, sigma, niter, delta_tv, lambda_tv, device)

    print(f"\n运行PnP-ULA...")
    x_pnp_final, x_pnp_mean, x_pnp_std = pnp_ula(y, A, AT, sigma, denoise, niter, delta_pnp, eps, device)


    # ============================================================
    # 步骤3：可视化对比
    # ============================================================
    print("\n" + "=" * 60)
    print("步骤3：显式先验（TV）vs 隐式先验（学习去噪器）对比")
    print("=" * 60)

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # 第1行：原始、含噪
    axes[0][0].imshow(im, cmap='gray')
    axes[0][0].set_title('原始图像')
    axes[0][0].axis('off')

    axes[0][1].imshow(y.cpu().numpy(), cmap='gray')
    axes[0][1].set_title(r'含噪+模糊' + '\n' + r'PSNR: {:.2f} dB'.format(PSNR(x,y)))
    axes[0][1].axis('off')

    # TV结果
    axes[0][2].imshow(x_tv_mean.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0][2].set_title(r'TV-ULA 后验均值' + '\n' + r'PSNR: {:.2f} dB'.format(PSNR(x_tv_mean,x)))
    axes[0][2].axis('off')

    # PnP结果
    axes[0][3].imshow(x_pnp_mean.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0][3].set_title(r'PnP-ULA 后验均值' + '\n' + r'PSNR: {:.2f} dB'.format(PSNR(x_pnp_mean,x)))
    axes[0][3].axis('off')

    # 第2行：最终样本
    axes[1][0].axis('off')

    axes[1][1].axis('off')

    axes[1][2].imshow(x_tv_final.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1][2].set_title('TV-ULA 最终样本')
    axes[1][2].axis('off')

    axes[1][3].imshow(x_pnp_final.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1][3].set_title('PnP-ULA 最终样本')
    axes[1][3].axis('off')

    # 第3行：不确定性
    axes[2][0].axis('off')

    axes[2][1].axis('off')

    axes[2][2].imshow(x_tv_std.cpu().numpy(), cmap='hot')
    axes[2][2].set_title('TV-ULA 后验标准差')
    axes[2][2].axis('off')

    axes[2][3].imshow(x_pnp_std.cpu().numpy(), cmap='hot')
    axes[2][3].set_title('PnP-ULA 后验标准差')
    axes[2][3].axis('off')

    fig.suptitle('实验5.4-1 显式先验(TV) vs 隐式先验(学习去噪器)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤3_显式vs隐式先验.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 对比统计
    print(f"\n重建质量对比:")
    print(f"  含噪图像:     PSNR = {PSNR(x,y):.2f} dB")
    print(f"  TV-ULA均值:   PSNR = {PSNR(x_tv_mean,x):.2f} dB")
    print(f"  PnP-ULA均值:  PSNR = {PSNR(x_pnp_mean,x):.2f} dB")
    print(f"\n不确定性对比:")
    print(f"  TV-ULA 平均标准差:  {torch.mean(x_tv_std).item():.4f}")
    print(f"  PnP-ULA 平均标准差: {torch.mean(x_pnp_std).item():.4f}")


    # ============================================================
    # 步骤4：结构对偶性展示
    # ============================================================
    print("\n" + "=" * 60)
    print("步骤4：近端算子 vs 去噪器的结构对偶性")
    print("=" * 60)

    # 在含噪图像上分别应用TV近端算子和去噪器
    y_denoise = y.clone()
    sigma_denoise = sigma.item()

    # TV近端算子
    prox_result = chambolle_prox_TV(y_denoise, device, {'lambda': 0.05, 'maxiter': 200})

    # 学习去噪器
    denoise_result = denoise(y_denoise)

    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0][0].imshow(y.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0][0].set_title(r'含噪观测 $y$')
    axes[0][0].axis('off')

    axes[0][1].imshow(prox_result.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0][1].set_title(r'TV近端算子 $\mathrm{prox}_{\lambda\mathrm{TV}}(y)$' + '\n(MAP方向)')
    axes[0][1].axis('off')

    axes[0][2].imshow(denoise_result.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0][2].set_title(r'去噪器 $D_\varepsilon(y)$' + '\n(MMSE方向)')
    axes[0][2].axis('off')

    # 残差对比：y - prox vs y - denoiser
    residual_prox = y - prox_result  # $\\lambda\\cdot\\nabla R_\\lambda(y)$ (Moreau包络梯度方向)
    residual_denoise = y - denoise_result  # $\\varepsilon\\cdot\\nabla\\log p_\\varepsilon(y)$ (Tweedie得分方向)

    axes[1][0].axis('off')

    axes[1][1].imshow(residual_prox.cpu().numpy(), cmap='RdBu_r')
    axes[1][1].set_title(r'$y - \mathrm{prox}_{\lambda\mathrm{TV}}(y)$' + '\n' + r'$= \lambda\cdot\nabla R_\lambda(y)$ (Moreau梯度)')
    axes[1][1].axis('off')

    axes[1][2].imshow(residual_denoise.cpu().numpy(), cmap='RdBu_r')
    axes[1][2].set_title(r'$y - D_\varepsilon(y)$' + '\n' + r'$= \varepsilon\cdot\nabla\log p_\varepsilon(y)$ (Tweedie得分)')
    axes[1][2].axis('off')

    fig.suptitle('近端算子 vs 去噪器：结构对偶性', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤4_结构对偶性.png'), dpi=150, bbox_inches='tight')
    plt.close()


    # ============================================================
    # 结构对偶性总结表
    # ============================================================
    print("\n结构对偶性总结：")
    print("-" * 70)
    print(f"{'性质':<15} {'Moreau包络（MAP）':<25} {'软下卷积（MMSE）':<25}")
    print("-" * 70)

    # 定义LaTeX字符串（避免f-string中的反斜杠问题）
    def_str1 = r'$R_\lambda(y)=\inf\{R(x)+\|x-y\|^2/(2\lambda)\}$'
    def_str2 = r'$\bar{R}_\varepsilon(y)=-\log\int\exp(...)dx$'
    op_str1 = r'$\mathrm{prox}_{\lambda R}$'
    op_str2 = r'$D_\varepsilon^*$'
    grad_str1 = r'$\nabla R_\lambda=(y-\mathrm{prox})/\lambda$'
    grad_str2 = r'$\nabla\bar{R}_\varepsilon=(y-D_\varepsilon^*)/\varepsilon$'
    step_str1 = r'$\mathrm{prox}=y-\lambda\nabla R_\lambda$'
    step_str2 = r'$D_\varepsilon^*=y-\varepsilon\nabla\bar{R}_\varepsilon$'
    temp_str1 = r'$T=0$（绝对零度）'
    temp_str2 = r'$T=1$（满温度）'

    print(f"{'定义':<15} {def_str1:<25} {def_str2:<25}")
    print(f"{'算子':<15} {op_str1:<25} {op_str2:<25}")
    print(f"{'梯度':<15} {grad_str1:<25} {grad_str2:<25}")
    print(f"{'一步':<15} {step_str1:<25} {step_str2:<25}")
    print(f"{'目标':<15} {'众数（MAP）':<25} {'均值（MMSE）':<25}")
    print(f"{'温度':<15} {temp_str1:<25} {temp_str2:<25}")
    print("-" * 70)


    # ============================================================
    # 实验总结
    # ============================================================
    print("\n" + "=" * 60)
    print("实验5.4-1 总结")
    print("=" * 60)
    print("1. 近端算子与去噪器的结构对偶：")
    print(r"   数学形式相同：$y - c\cdot\nabla(\cdot)$")
    print(r"   近端算子：$c=\lambda$，$\nabla$是Moreau包络梯度")
    print(r"   去噪器：$c=\varepsilon$，$\nabla$是软下卷积梯度")
    print("2. 显式先验 vs 隐式先验：")
    print(r"   显式先验（TV）：需要手动选择$\lambda$，表达能力有限")
    print("   隐式先验（学习去噪器）：自动学习，表达能力强")
    print("   学习去噪器通常优于手工先验")
    print("3. MAP vs MMSE：")
    print(r"   MAP（近端算子）：输出众数，对应'绝对零度'扩散")
    print(r"   MMSE（去噪器）：输出均值，对应'满温度'扩散")
    print(r"   温度参数$T$控制从众数到均值的过渡")
else:
    print("\n" + "=" * 60)
    print("实验5.4-1 总结")
    print("=" * 60)
    print("步骤1已完成：TV近端算子演示")
    print("\n步骤2-4需要以下条件才能运行：")
    print("  1. GPU支持（当前设备：{}）".format(device))
    print("  2. sampling_tools模块（{}）".format('可用' if _has_sampling_tools else '不可用'))
    print("  3. 预训练模型（{}）".format('可用' if _has_model else '不可用'))
    if not _has_model:
        print(f"\n  模型路径: {_model_path}")
        print("  请将RealSN_DnCNN_noise5.pth放入Pretrained_models目录")
    print("\n核心概念：")
    print(r"  近端算子（MAP方向）：$\mathrm{prox}_{\lambda R}(y)$")
    print(r"  去噪器（MMSE方向）：$D_\varepsilon^*(y)$")
    print(r"  结构对偶：$y - c\cdot\nabla(\cdot)$")
    print("=" * 60)
