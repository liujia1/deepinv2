# -*- coding: utf-8 -*-
"""
实验18.1 自定义前向算子与伴随验证
对应知识点：18.2节（定义自定义前向算子）

实验内容：
Step 1: 深入理解deepinv Physics类体系 —— 浏览内置算子并理解三原则（配对A/A^T、噪声分离、可组合性）
Step 2: 实现多视角下采样算子 MultiViewPhysics —— 仿射变换+下采样的复合前向模型
Step 3: 伴随验证与自伴随检测 —— dot product test验证手动伴随，autograd自动伴随对比
Step 4: 噪声模型对比 —— 高斯/泊松/泊松-高斯复合噪声的视觉效果与统计特性
Step 5: ★自定义模糊核与MRI掩模算子 —— 将自定义退化集成到deepinv框架中

★原创设计：
- Step 5的自定义模糊核+MRI掩模综合实验，验证算子可组合性原则
- 对比手动伴随 vs autograd自动伴随的数值精度
- 泊松-高斯复合噪声模型的可视化对比

素材来源：MiniProject_DefiningOperator.ipynb、deepinv Physics API
运行前提：无需GPU，CPU即可运行
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import warnings
import logging

# ====== 解决中文乱码的核心代码（Windows + Linux 自动适配）======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

import platform
from matplotlib.font_manager import FontManager, FontProperties

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
    import os as _os, re
    cjk_patterns = ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']
    for f in fm.ttflist:
        name_lower = f.name.lower()
        fname_lower = (_os.path.basename(f.fname) if hasattr(f, 'fname') else '').lower()
        for pat in cjk_patterns:
            if re.search(pat, name_lower) or re.search(pat, fname_lower):
                return f.name
    return None

# ====== 保存目录（优先Google Drive）======
_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    SAVE_DIR = os.path.join(_gdrive, '实验18_1_自定义前向算子与伴随验证')
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"检测到 Google Drive，结果将保存至: {SAVE_DIR}")
else:
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    print(f"本地环境，结果将保存至: {SAVE_DIR}")

_cn_font = _find_chinese_font()
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"[Font] 已检测到中文字体: {_cn_font}")
else:
    if platform.system() != 'Windows':
        _font_url = 'https://github.com/jsntn/webfonts/raw/master/NotoSansSC-Regular.ttf'
        _font_file = os.path.join(SAVE_DIR, 'NotoSansSC-Regular.ttf')
        if os.path.exists(_font_file):
            from matplotlib.font_manager import fontManager
            fontManager.addfont(_font_file)
            plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['font.family'] = 'sans-serif'
            _cn_font = 'Noto Sans SC'
            print(f"[Font] 已加载缓存字体: {_cn_font}")
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
            except Exception as e:
                print(f"[Font] 字体下载失败: {e}，中文可能显示为方框")
    else:
        print("[Font] 未找到中文字体，中文可能显示为方框")
# ========================================================

# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 安装deepinv（Colab环境）
try:
    import deepinv
except ImportError:
    print("正在安装 deepinv ...")
    os.system('pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv')
    import deepinv as dinv
else:
    dinv = deepinv
    print(f"deepinv 版本: {dinv.__version__}")

from deepinv.physics import LinearPhysics, Downsampling, GaussianNoise, PoissonNoise, Denoising
from deepinv.utils import load_example


# ========================================================================
# Step 1: 浏览deepinv Physics类体系
# 对应18.2节知识点：Physics类设计三原则
# ========================================================================
print("\n" + "="*70)
print("Step 1: deepinv Physics 类体系与设计三原则")
print("="*70)

print("""
deepinv Physics 类体系的设计三原则：
1. 配对性: 每个算子 A 必须提供对应的伴随 A_adjoint（或 A^T）
2. 噪声分离: 前向模型 y = A(x) + ε，噪声独立于算子可组合
3. 可组合性: 多个算子可通过乘法组合成复合前向模型

常用内置 Physics 类：
- Denoising:        A = I (恒等算子)，加性噪声
- Downsampling:     A = 下采样算子（最近邻/双线性）
- Blur:             A = 卷积模糊算子
- Inpainting:       A = 逐像素掩模
- Tomography:       A = Radon 变换（CT重建）
- MRI:              A = 傅里叶+子采样掩模
""")

# 加载示例图像
x_true = load_example("celeba_example.jpg", img_size=(256, 256), resize_mode='resize')
x_true = x_true.unsqueeze(0).to(device)  # (1, 3, 256, 256)
print(f"示例图像 shape: {x_true.shape}")


# ========================================================================
# Step 2: 实现多视角下采样算子 MultiViewPhysics
# 对应18.2节知识点：自定义LinearPhysics子类
# 素材来源: MiniProject_DefiningOperator.ipynb
# ========================================================================
print("\n" + "="*70)
print("Step 2: 实现多视角下采样算子 MultiViewPhysics")
print("="*70)

class MultiViewPhysics(LinearPhysics):
    """
    多视角下采样算子：y_j = A * T_j * x
    其中 A 是基础算子（如下采样），T_j 是仿射变换（旋转+平移）
    
    输入 x: (B, C, H, W)
    输出 y: (B, J, C, H', W')，其中 H'=H/factor, W'=W/factor
    """
    def __init__(self, base_physics, transf=None, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.base_physics = base_physics
        self.transf = transf
        self.device = device

    def A(self, x, **kwargs):
        """前向传播: 对x施加J个仿射变换，再分别下采样"""
        transf = kwargs.get('transf', self.transf)
        if transf is None:
            raise ValueError("需要提供变换矩阵 transf")
        
        B, C, H, W = x.shape
        J = transf.shape[0]
        
        # 对每个视角施加仿射变换
        views = []
        for j in range(J):
            # 构造仿射网格
            theta_j = transf[j:j+1].to(x.device).float()  # (1, 2, 3)
            grid = F.affine_grid(theta_j.expand(B, -1, -1, -1),
                                  size=(B, C, H, W),
                                  align_corners=False)
            x_transformed = F.grid_sample(x, grid, align_corners=False)
            # 下采样
            y_j = self.base_physics.A(x_transformed)
            views.append(y_j)
        
        y = torch.stack(views, dim=1)  # (B, J, C, H', W')
        return y

    def A_adjoint(self, y, **kwargs):
        """
        伴随算子: A^T y = sum_j T_j^T A^T y_j
        对每个视角：先上采样(A^T)，再反向仿射变换(T_j^T)
        """
        transf = kwargs.get('transf', self.transf)
        if transf is None:
            raise ValueError("需要提供变换矩阵 transf")
        
        B, J, C, H_prime, W_prime = y.shape
        # 从base_physics推断原始尺寸
        if hasattr(self.base_physics, 'img_size'):
            _, _, H_full, W_full = self.base_physics.img_size
        else:
            factor = self.base_physics.factor if hasattr(self.base_physics, 'factor') else 4
            H_full = H_prime * factor
            W_full = W_prime * factor
        
        x_adj = torch.zeros(B, C, H_full, W_full, device=y.device)
        
        for j in range(J):
            y_j = y[:, j]  # (B, C, H', W')
            # A^T: 上采样（伴随下采样）
            x_up = self.base_physics.A_adjoint(y_j)  # (B, C, H, W)
            # T_j^T: 反向仿射变换（求完整2×3仿射矩阵的逆）
            theta_2x3 = transf[j].to(y.device).float()  # (2, 3)
            A_2x2 = theta_2x3[:, :2]  # (2, 2) 旋转+缩放部分
            t_2x1 = theta_2x3[:, 2:3]  # (2, 1) 平移部分
            try:
                A_inv = torch.inverse(A_2x2)
                t_inv = -A_inv @ t_2x1  # 逆变换的平移: t' = -A^{-1} t
                theta_inv_full = torch.cat([A_inv, t_inv], dim=1).unsqueeze(0)  # (1, 2, 3)
            except RuntimeError:
                # 矩阵不可逆时用转置近似（退化情况）
                theta_inv_full = theta_2x3.unsqueeze(0)
            
            grid = F.affine_grid(theta_inv_full.expand(B, -1, -1, -1),
                                  size=(B, C, H_full, W_full),
                                  align_corners=False)
            # 累加: A^T y = Σ_j T_j^T A^T y_j （堆叠算子的伴随是各子伴随之和，
            # 重叠区域值大反映观测更多，后续求解器会归一化）
            x_adj = x_adj + F.grid_sample(x_up, grid, align_corners=False)
        
        return x_adj

    def update_parameters(self, transf=None, **kwargs):
        """更新变换参数"""
        if transf is not None:
            self.transf = transf


def generate_random_transforms(J, scale=0.8, max_angle=np.pi/8, max_shift=0.05):
    """生成J个随机仿射变换矩阵（旋转+缩放+平移）"""
    transf = torch.zeros(J, 2, 3)
    for i in range(J):
        angle = torch.rand(1) * max_angle
        transf[i, 0, 0] = torch.cos(angle) * scale
        transf[i, 0, 1] = -torch.sin(angle) * scale
        transf[i, 1, 0] = torch.sin(angle) * scale
        transf[i, 1, 1] = torch.cos(angle) * scale
        transf[i, :, -1] = torch.randn(2) * max_shift
    return transf


# 创建多视角算子并测试
print("\n创建多视角下采样算子 (4×下采样, J=16个视角)...")
base_physics = Downsampling(factor=4, img_size=(3, 256, 256), device=device,
                             filter=None, padding="zeros")
torch.manual_seed(42)
transf = generate_random_transforms(J=16)

multi_physics = MultiViewPhysics(base_physics, transf=transf, device=device)
y_multi = multi_physics(x_true)
print(f"  输入 x shape:  {x_true.shape}")
print(f"  输出 y shape:  {y_multi.shape}")
print(f"  J={y_multi.shape[1]} 个视角, 每个视角 {y_multi.shape[3]}×{y_multi.shape[4]}")


# ========================================================================
# Step 3: 伴随验证与自伴随检测
# 对应18.2节知识点：dot product test与autograd自动伴随
# ========================================================================
print("\n" + "="*70)
print("Step 3: 伴随验证与自伴随检测")
print("="*70)

print("""
伴随验证原理 (Dot Product Test):
  对任意 x, y: <Ax, y> ≈ <x, A^T y>
  相对误差 |<Ax,y> - <x,A^Ty>| / max(|<Ax,y>|, |<x,A^Ty>|) 应接近0

deepinv 提供:
  physics.adjointness_test(x) → 相对误差
  dinv.physics.adjoint_function(A) → 基于autograd的自动伴随
""")


def detailed_adjointness_test(physics, x, n_tests=5):
    """多次随机测试伴随的精度，返回均值和标准差"""
    errors = []
    for i in range(n_tests):
        torch.manual_seed(100 + i)
        y_rand = torch.randn_like(physics(x))
        lhs = (physics(x) * y_rand).sum().item()
        rhs = (x * physics.A_adjoint(y_rand)).sum().item()
        rel_err = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-10)
        errors.append(rel_err)
    return np.mean(errors), np.std(errors)

# 3a. 对内置算子做伴随验证
print("\n--- 3a. 内置算子伴随验证 ---")

# 下采样算子
down_phys = Downsampling(factor=2, img_size=(3, 256, 256), device=device)
x_test = torch.randn(1, 3, 256, 256, device=device)
adj_err_down = down_phys.adjointness_test(x_test)
mean_err, std_err = detailed_adjointness_test(down_phys, x_test, n_tests=5)
print(f"  下采样 (factor=2) 伴随误差: {adj_err_down:.2e} (多次测试: {mean_err:.2e} ± {std_err:.2e})")

# 3b. 对MultiViewPhysics做伴随验证
print("\n--- 3b. MultiViewPhysics 伴随验证 ---")
adj_err_multi = multi_physics.adjointness_test(x_true)
mean_multi, std_multi = detailed_adjointness_test(multi_physics, x_true, n_tests=5)
print(f"  MultiViewPhysics 伴随误差: {adj_err_multi:.2e} (多次测试: {mean_multi:.2e} ± {std_multi:.2e})")
if adj_err_multi < 1e-3:
    print("  ✓ 伴随验证通过！")
else:
    print("  ✗ 伴随误差较大，手动伴随可能不够精确，建议使用autograd自动伴随")

# 3c. ★ 使用autograd自动伴随对比
print("\n--- 3c. ★ autograd 自动伴随 vs 手动伴随 ---")
try:
    # 兼容不同deepinv版本的导入路径
    try:
        from deepinv.physics import adjoint_function
    except ImportError:
        from deepinv.utils import adjoint_function
    
    # 创建使用autograd伴随的版本
    class MultiViewPhysicsAutoAdj(MultiViewPhysics):
        """使用autograd自动计算伴随的版本"""
        def A_adjoint(self, y, **kwargs):
            # 利用autograd自动计算A^T
            adj_fn = adjoint_function(self.A, y.shape, device=y.device)
            return adj_fn(y)
    
    multi_physics_auto = MultiViewPhysicsAutoAdj(base_physics, transf=transf, device=device)
    adj_err_auto = multi_physics_auto.adjointness_test(x_true)
    print(f"  autograd伴随误差: {adj_err_auto:.2e}")
    print(f"  手动伴随误差:     {adj_err_multi:.2e}")
    
    if adj_err_auto < adj_err_multi:
        print("  ★ autograd自动伴随更精确！推荐在手动伴随不精确时使用")
    else:
        print("  手动伴随已经足够精确")
        
except (ImportError, AttributeError):
    print("  autograd伴随功能在当前deepinv版本不可用，跳过对比")

# 3d. 算子范数
print("\n--- 3d. 算子范数估计 ---")
try:
    norm_down = down_phys.compute_norm(x_test)
    print(f"  下采样算子范数: {norm_down:.4f}")
except Exception as e:
    print(f"  算子范数计算失败: {e}")

# --- Step 3 可视化 ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
# 原始图像
axes[0].imshow(x_true[0].cpu().permute(1, 2, 0).clamp(0, 1))
axes[0].set_title('原始图像 x', fontsize=12)
axes[0].axis('off')

# 多视角观测（前4个视角）
for i in range(3):
    axes[i+1].imshow(y_multi[0, i].cpu().permute(1, 2, 0).clamp(0, 1))
    angle_deg = torch.atan2(transf[i, 1, 0], transf[i, 0, 0]).item() * 180 / np.pi
    axes[i+1].set_title(f'视角 {i+1}\n旋转≈{angle_deg:.1f}°', fontsize=11)
    axes[i+1].axis('off')

fig.suptitle('Step 3: 多视角下采样算子的前向结果', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step3_multiview_forward.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step3_multiview_forward.png")


# ========================================================================
# Step 4: 噪声模型对比
# 对应18.2节知识点：噪声分离原则与常见噪声模型
# ========================================================================
print("\n" + "="*70)
print("Step 4: 噪声模型对比（高斯/泊松/泊松-高斯）")
print("="*70)

print("""
噪声模型对比:
  高斯噪声:     y = Ax + ε,           ε ~ N(0, σ²I)        — 加性，方差恒定
  泊松噪声:     y = Poisson(Ax/λ)·λ   — 信号依赖型，方差随信号强度变化
  泊松-高斯:    y = Poisson(Ax/λ)·λ + ε — 传感器噪声的更真实模型
""")

# 设置不同噪声模型
sigma_gauss = 0.05  # 高斯噪声标准差
gain_poisson = 0.1  # 泊松噪声增益（越小噪声越大）

# 使用下采样算子 + 不同噪声
down_phys_clean = Downsampling(factor=4, img_size=(3, 256, 256), device=device)

# 高斯噪声
phys_gauss = Downsampling(factor=4, img_size=(3, 256, 256), device=device)
phys_gauss.set_noise_model(GaussianNoise(sigma=sigma_gauss))

# 泊松噪声
phys_poisson = Downsampling(factor=4, img_size=(3, 256, 256), device=device)
phys_poisson.set_noise_model(PoissonNoise(gain=gain_poisson))

# 泊松-高斯复合噪声（★模拟真实传感器：光子散粒噪声+电子读出噪声）
class PoissonGaussianNoise(nn.Module):
    """泊松-高斯复合噪声模型
    y = Poisson(x/λ)·λ + N(0, σ²)
    - 输入x为 A(x_out)（由deepinv Physics自动传入）
    - Poisson: 光子散粒噪声（信号依赖型，方差∝信号强度）
    - Gaussian: 电子读出噪声（信号无关型，方差恒定）
    - gain(λ)越小，泊松噪声越强；sigma越大，高斯噪声越强
    """
    def __init__(self, gain=0.1, sigma=0.02):
        super().__init__()
        self.gain = gain
        self.sigma = sigma
    
    def forward(self, x, **kwargs):
        # 1. 泊松噪声：模拟光子计数过程
        x_poisson = torch.poisson(x.clamp(0, 1) / self.gain) * self.gain
        # 2. 高斯噪声：模拟电子读出噪声（独立于信号）
        x_noisy = x_poisson + self.sigma * torch.randn_like(x)
        return x_noisy.clamp(0, 1)

phys_pg = Downsampling(factor=4, img_size=(3, 256, 256), device=device)
phys_pg.set_noise_model(PoissonGaussianNoise(gain=0.1, sigma=0.02))

# 生成观测
torch.manual_seed(42)
y_clean = down_phys_clean(x_true)
y_gauss = phys_gauss(x_true)
y_poisson = phys_poisson(x_true)
y_pg = phys_pg(x_true)

# 计算噪声水平
noise_gauss = (y_gauss - y_clean).abs().mean().item()
noise_poisson = (y_poisson - y_clean).abs().mean().item()
noise_pg = (y_pg - y_clean).abs().mean().item()

print(f"  无噪声观测:    MAE = 0")
print(f"  高斯噪声:      MAE = {noise_gauss:.4f}  (σ={sigma_gauss})")
print(f"  泊松噪声:      MAE = {noise_poisson:.4f}  (gain={gain_poisson})")
print(f"  泊松-高斯:     MAE = {noise_pg:.4f}  (gain=0.1, σ=0.02)")

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
titles = ['无噪声', '高斯噪声', '泊松噪声', '泊松-高斯']
images = [y_clean, y_gauss, y_poisson, y_pg]

for i, (title, img) in enumerate(zip(titles, images)):
    axes[0, i].imshow(img[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0, i].set_title(title, fontsize=12)
    axes[0, i].axis('off')
    
    if i > 0:
        residual = (img - y_clean)[0].cpu().permute(1, 2, 0)
        axes[1, i].imshow(residual.clamp(-0.1, 0.1) + 0.5, vmin=0, vmax=1)
        axes[1, i].set_title(f'残差 (MAE={images[i].abs().mean():.4f})', fontsize=10)
    else:
        axes[1, i].text(0.5, 0.5, '(原始观测)', ha='center', va='center', fontsize=12)
    axes[1, i].axis('off')

# 隐藏左下角的空位
axes[1, 0].imshow(y_clean[0].cpu().permute(1, 2, 0).clamp(0, 1))
axes[1, 0].set_title('原始观测', fontsize=10)
axes[1, 0].axis('off')

fig.suptitle('Step 4: 噪声模型对比（4×下采样观测）', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step4_noise_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step4_noise_comparison.png")

# 噪声直方图对比
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i, (title, img) in enumerate(zip(titles[1:], images[1:])):
    residual = (img - y_clean).cpu().numpy().flatten()
    axes[i].hist(residual, bins=100, density=True, alpha=0.7, color=f'C{i}')
    axes[i].set_title(f'{title} 残差分布', fontsize=12)
    axes[i].set_xlabel('残差值')
    axes[i].set_ylabel('概率密度')
    axes[i].axvline(0, color='red', linestyle='--', alpha=0.5)

fig.suptitle('Step 4: 噪声残差分布对比', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step4_noise_histogram.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step4_noise_histogram.png")


# ========================================================================
# Step 5: ★自定义模糊核与MRI掩模算子
# 对应18.2节知识点：算子可组合性与自定义退化
# ========================================================================
print("\n" + "="*70)
print("Step 5: ★ 自定义模糊核与MRI掩模算子（算子组合性验证）")
print("="*70)

print("""
★ 本步骤演示deepinv的算子可组合性原则：
  将自定义模糊核 → 下采样 → MRI子采样 组合为复合前向模型
  复合模型 y = M·D·K·x + ε 中每个子算子都有独立的伴随
""")

# 5a. 自定义模糊核
print("\n--- 5a. 自定义运动模糊核 ---")

def create_motion_blur_kernel(length=15, angle=45):
    """创建运动模糊核"""
    kernel = np.zeros((length, length))
    center = length // 2
    angle_rad = np.deg2rad(angle)
    for i in range(length):
        offset = i - center
        x = int(round(center + offset * np.cos(angle_rad)))
        y = int(round(center + offset * np.sin(angle_rad)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    kernel = kernel / kernel.sum()
    return kernel

# 创建两种模糊核
motion_kernel = create_motion_blur_kernel(length=15, angle=30)
gauss_kernel = torch.tensor(
    dinv.physics.blur.gaussian_blur(sigma=(2.0, 2.0)).numpy() 
    if hasattr(dinv.physics.blur, 'gaussian_blur') else 
    np.zeros((1, 1, 15, 15)),
    dtype=torch.float32
)

# 使用deepinv的Blur算子
try:
    from deepinv.physics import Blur
    motion_k_tensor = torch.tensor(motion_kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # 扩展到3通道
    motion_k_3ch = motion_k_tensor.repeat(1, 3, 1, 1)
    blur_phys = Blur(filter=motion_k_3ch, device=device)
    
    y_blurred = blur_phys(x_true)
    
    # 伴随验证
    adj_err_blur = blur_phys.adjointness_test(x_true)
    print(f"  运动模糊算子伴随误差: {adj_err_blur:.2e}")
    
    has_blur = True
except Exception as e:
    print(f"  Blur算子创建失败: {e}，跳过模糊部分")
    has_blur = False
    has_composite = False

# 5b. ★ 组合算子：模糊 → 下采样
if has_blur:
    print("\n--- 5b. ★ 组合算子: 模糊 → 下采样 ---")
    try:
        # deepinv 支持算子乘法组合
        composite_phys = blur_phys * Downsampling(factor=4, img_size=(3, 256, 256), device=device)
        y_composite = composite_phys(x_true)
        adj_err_composite = composite_phys.adjointness_test(x_true)
        print(f"  组合算子(模糊+下采样)伴随误差: {adj_err_composite:.2e}")
        print(f"  组合观测 shape: {y_composite.shape}")
        has_composite = True
    except Exception as e:
        print(f"  算子组合失败: {e}")
        has_composite = False

# 5c. MRI子采样掩模
print("\n--- 5c. MRI子采样掩模 ---")
try:
    from deepinv.physics import MRI
    
    # 创建4倍加速的MRI物理模型
    mri_phys = MRI(img_size=(256, 256), device=device, acceleration=4)
    # MRI通常是单通道，将RGB转灰度
    x_gray = x_true.mean(dim=1, keepdim=True)  # (1, 1, 256, 256)
    y_mri = mri_phys(x_gray)
    
    adj_err_mri = mri_phys.adjointness_test(x_gray)
    print(f"  MRI算子伴随误差: {adj_err_mri:.2e}")
    print(f"  MRI观测 shape: {y_mri.shape}")
    has_mri = True
except Exception as e:
    print(f"  MRI算子创建失败: {e}")
    has_mri = False

# 5d. 可视化
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
# 第一行: 原图 + 模糊 + 组合
axes[0, 0].imshow(x_true[0].cpu().permute(1, 2, 0).clamp(0, 1))
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

if has_blur:
    axes[0, 1].imshow(y_blurred[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0, 1].set_title('运动模糊', fontsize=12)
    axes[0, 1].axis('off')
    # ★ 模糊核的幅度谱（频域视角理解模糊）
    kernel_fft = torch.fft.fftshift(
        torch.fft.fft2(torch.tensor(motion_kernel, dtype=torch.float32))
    )
    axes[0, 2].imshow(np.log1p(kernel_fft.abs().numpy()), cmap='viridis')
    axes[0, 2].set_title('★ 模糊核幅度谱', fontsize=12)
    axes[0, 2].axis('off')
else:
    axes[0, 1].text(0.5, 0.5, '模糊算子不可用', ha='center', va='center', fontsize=12)
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')

# 第二行: 组合 + MRI重建 + MRI k空间
if has_composite:
    axes[1, 0].imshow(y_composite[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[1, 0].set_title('★ 模糊+下采样组合', fontsize=12)
    axes[1, 0].axis('off')
else:
    axes[1, 0].text(0.5, 0.5, '组合算子不可用', ha='center', va='center', fontsize=12)
    axes[1, 0].axis('off')

if has_mri:
    # 零填充重建
    x_adj_mri = mri_phys.A_adjoint(y_mri)
    axes[1, 1].imshow(x_adj_mri[0, 0].cpu(), cmap='gray')
    axes[1, 1].set_title('MRI零填充重建', fontsize=12)
    axes[1, 1].axis('off')
    
    # k-space
    kspace = torch.fft.fftshift(y_mri[0, 0].cpu().abs(), dim=(-2, -1))
    axes[1, 2].imshow(np.log1p(kspace.numpy()), cmap='gray')
    axes[1, 2].set_title('MRI k空间(对数)', fontsize=12)
    axes[1, 2].axis('off')
else:
    axes[1, 1].text(0.5, 0.5, 'MRI不可用', ha='center', va='center', fontsize=12)
    axes[1, 1].axis('off')
    axes[1, 2].text(0.5, 0.5, 'MRI不可用', ha='center', va='center', fontsize=12)
    axes[1, 2].axis('off')

# 第三行: 伴随验证汇总
for c in range(3):
    axes[2, c].axis('off')

summary_text = "伴随验证汇总:\n\n"
summary_text += f"下采样算子:       {adj_err_down:.2e}\n"
summary_text += f"MultiViewPhysics: {adj_err_multi:.2e}\n"
if has_blur:
    summary_text += f"运动模糊:         {adj_err_blur:.2e}\n"
if has_composite:
    summary_text += f"模糊+下采样:      {adj_err_composite:.2e}\n"
if has_mri:
    summary_text += f"MRI子采样:        {adj_err_mri:.2e}\n"
summary_text += "\n阈值: < 1e-3 通过"
axes[2, 0].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=axes[2, 0].transAxes)

# ★ 模糊核空间域可视化
if has_blur:
    axes[2, 1].imshow(motion_kernel, cmap='gray')
    axes[2, 1].set_title('★ 运动模糊核(空间域)', fontsize=12)
    axes[2, 1].axis('off')

# MRI灰度输入
if has_mri:
    axes[2, 2].imshow(x_gray[0, 0].cpu(), cmap='gray')
    axes[2, 2].set_title('MRI输入(灰度)', fontsize=12)
    axes[2, 2].axis('off')

fig.suptitle('Step 5: ★ 自定义算子与算子组合性验证', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step5_custom_operators.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step5_custom_operators.png")


# ========================================================================
# 实验总结
# ========================================================================
print("\n" + "="*70)
print("实验18.1 总结")
print("="*70)
print("""
本实验对应18.2节核心知识点：

1. Physics类设计三原则 ✓
   - 配对性: A 与 A^T 必须配对
   - 噪声分离: 前向模型 + 噪声模型独立设置
   - 可组合性: 算子可通过乘法组合

2. 自定义LinearPhysics子类 ✓
   - MultiViewPhysics: 仿射变换 + 下采样
   - A(x) 和 A_adjoint(y) 的实现
   - update_parameters 动态参数更新

3. 伴随验证方法 ✓
   - dot product test: <Ax,y> ≈ <x,A^T y>
   - autograd自动伴随: 精度更高但计算更慢
   - 算子范数估计: compute_norm (power iteration)

4. 噪声模型体系 ✓
   - 高斯噪声: 加性，方差恒定
   - 泊松噪声: 信号依赖型
   - 泊松-高斯: 复合噪声，更贴近实际传感器

5. ★ 自定义算子组合 ✓
   - 模糊 → 下采样组合
   - MRI子采样验证
   - 验证组合算子伴随仍然正确

预期结果：
- 伴随误差均 < 1e-3（通过验证）
- 泊松噪声残差分布非高斯（信号依赖）
- 组合算子伴随误差与子算子同量级
""")

print(f"所有图像已保存至: {SAVE_DIR}")
