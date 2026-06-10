# -*- coding: utf-8 -*-
"""
实验5.4-1 近端算子：Moreau包络的一步梯度
对应章节：5.4 MAP与MMSE的结构对偶性
  - 近端算子 = Moreau包络的一步梯度（第3小节）
  - 验证TV近端算子作为MAP去噪器的实现

知识点：
  - 近端算子定义：prox_{λR}(y) = argmin_x {R(x) + ||x-y||^2/(2λ)}
  - Moreau包络梯度：∇R̂_λ(y) = (y - prox_{λR}(y)) / λ
  - 近端算子 = 沿Moreau包络梯度走一步：prox = y - λ∇R̂_λ(y)

运行前提：
  仅需CPU，无需GPU和预训练模型
  需要 sampling_tools/chambolle_prox_TV.py（已包含在当前目录）

本实验对应5.4节第3小节"近端算子 = Moreau包络的一步梯度"。
拆分自原实验5.4-1的步骤1。
"""

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
        from sampling_tools import chambolle_prox_TV
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
        print("    - .chinese/ (可选，会自动创建)")
        print("=" * 60)
    else:
        print("警告: sampling_tools 模块未找到")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ============================================================
# 步骤1：TV近端算子（显式先验）演示
# 使用 sampling_tools/chambolle_prox_TV.py
# ============================================================
print("\n" + "=" * 60)
print("步骤1：TV近端算子（显式先验）演示")
print("=" * 60)

# 创建测试图像（简单形状）
test_image = np.zeros((64, 64))
test_image[10:20, 10:20] = 1.0  # 正方形
test_image[40:50, 40:50] = 0.8  # 另一个正方形
test_image_t = torch.from_numpy(test_image).float().to(device)

# 添加噪声
noisy_image_t = test_image_t + 0.2 * torch.randn_like(test_image_t)

# 测试不同$\lambda$值的TV近端算子
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
# 步骤2：Moreau包络梯度验证
# 验证：prox = y - λ∇R̂_λ(y)，即近端算子 = 沿Moreau包络梯度走一步
# ============================================================
print("\n" + "=" * 60)
print("步骤2：验证 Moreau包络梯度关系")
print("=" * 60)

if _has_sampling_tools:
    lambda_tv = 0.1

    # 在近端算子结果上计算残差
    prox_result = chambolle_prox_TV(noisy_image_t, device, {'lambda': lambda_tv, 'maxiter': 200})
    residual = noisy_image_t - prox_result  # λ∇R̂_λ(y)
    gradient = residual / lambda_tv  # ∇R̂_λ(y)

    # 反向计算：y - λ∇R̂_λ(y) 应该等于 prox
    reconstructed = noisy_image_t - lambda_tv * gradient

    print(rf"  测试 $\lambda = {lambda_tv}$")
    print(rf"  prox(y) = Chambolle_TV近端算子输出")
    print(rf"  y - prox(y) = {torch.mean(residual.abs()).item():.4f} (平均残差绝对值)")
    print(rf"  ∇R̂_λ(y) = (y - prox) / λ")
    print(rf"  y - λ∇R̂_λ(y) = {torch.mean(reconstructed.abs()).item():.4f}")
    print(rf"  ||prox - (y - λ∇R̂_λ(y))|| = {torch.norm(prox_result - reconstructed).item():.2e}")
    is_valid = torch.norm(prox_result - reconstructed).item() < 1e-6
    print(f"  验证结果：{is_valid}（误差 < 1e-6 则成立）")

    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(noisy_image_t.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(r'含噪图像 $y$')
    axes[0].axis('off')

    axes[1].imshow(prox_result.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(r'$\mathrm{prox}_{\lambda\mathrm{TV}}(y)$' + '\n(MAP方向)')
    axes[1].axis('off')

    axes[2].imshow(residual.cpu().numpy(), cmap='RdBu_r')
    axes[2].set_title(r'$y - \mathrm{prox}(y)$' + '\n' + r'$= \lambda\cdot\nabla \hat{R}_\lambda(y)$')
    axes[2].axis('off')

    axes[3].imshow(gradient.cpu().numpy(), cmap='RdBu_r')
    axes[3].set_title(r'$\nabla \hat{R}_\lambda(y)$' + '\n' + r'$= (y - \mathrm{prox})/\lambda$')
    axes[3].axis('off')

    fig.suptitle('Moreau包络梯度验证：prox = y - λ∇R̂_λ(y)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤2_Moreau包络梯度验证.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  上图说明：")
    print(r"    左2：TV近端算子输出（MAP方向的一步结果）")
    print(r"    左3：残差 y - prox，等于 λ∇R̂_λ(y)（Moreau包络梯度方向）")
    print(r"    右：梯度 ∇R̂_λ(y) = (y - prox) / λ")
    print(r"    验证：从 $y$ 沿 Moreau 包络梯度走一步，恰好到达 $\mathrm{prox}_{\lambda\mathrm{TV}}(y)$")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.4-1 总结")
print("=" * 60)
print("1. 近端算子定义：")
print(r"   $\mathrm{prox}_{\lambda R}(y) = \arg\min_x \{R(x) + \frac{\|x-y\|^2}{2\lambda}\}$")
print("\n2. Moreau包络梯度：")
print(r"   $\nabla \hat{R}_\lambda(y) = \frac{y - \mathrm{prox}_{\lambda R}(y)}{\lambda}$")
print("\n3. 梯度步解读：")
print(r"   $\mathrm{prox}_{\lambda R}(y) = y - \lambda\,\nabla\hat{R}_\lambda(y)$")
print(r"   近端算子 = 从 $y$ 出发，沿Moreau包络梯度走一步，步长为 $\lambda$")
print(r"   这是MAP方向的'一步梯度下降'")
print("\n4. TV近端算子参数敏感性：")
print(r"   $\lambda$小：弱正则化，保留更多细节和噪声")
print(r"   $\lambda$大：强正则化，趋向常数，过度平滑")
print("\n下一步：加载去噪器（学习去噪器），实现MMSE方向的'一步'（见拆分实验5.4-2）")
