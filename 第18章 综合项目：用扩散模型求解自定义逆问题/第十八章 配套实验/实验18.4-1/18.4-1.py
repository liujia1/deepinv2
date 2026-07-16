# -*- coding: utf-8 -*-
"""
实验18.4-1 端到端求解策略对比
对应知识点：18.3节（求解策略选择与对比）、18.4节（端到端实战）

实验内容：
Step 1: 三种退化场景构造 —— 去模糊(轻度/重度)、超分辨率(2×/4×)、修复(50%/80%)
Step 2: 优化方法求解 —— Tikhonov正则化与TV正则化
Step 3: PnP方法求解 —— DPIR (DRUNet去噪器)
Step 4: 扩散方法求解 —— DiffPIR (DiffUNet扩散去噪器)
Step 5: ★方法对比与决策指南 —— PSNR/SSIM表格、运行时间、适用场景分析

★原创设计：
- 在统一框架下对比优化/PnP/扩散三种方法族的完整性能
- 轻度vs重度退化的方法选择决策可视化
- 运行时间-质量权衡图（Pareto前沿）

素材来源：MiniProject_DenoisingPrior.ipynb、18.4节代码模板
运行前提：需GPU（Colab T4即可），需下载预训练模型(DRUNet/DiffUNet)
"""

import os, sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # ★ 设置非交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import warnings
import logging
import pickle  # ★ 用于缓存结果

# ====== 解决中文乱码的核心配置 ======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

# ====== 保存目录（优先Google Drive）======
_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    SAVE_DIR = os.path.join(_gdrive, '实验18_4-1_端到端求解策略对比')
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"检测到 Google Drive，结果将保存至: {SAVE_DIR}")
else:
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    print(f"本地环境，结果将保存至: {SAVE_DIR}")

# ====== 导入中文字体模块 ======
# ★ 动态导入中文字体模块（兼容直接运行脚本）
_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
_chinese_font_path = os.path.join(_script_dir, '.chinese', 'chinese_font.py')
import importlib.util
_spec = importlib.util.spec_from_file_location("chinese_font", _chinese_font_path)
_chinese_font = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chinese_font)
setup_chinese_font = _chinese_font.setup_chinese_font

# ★ 设置中文字体
setup_chinese_font(os.path.join(_script_dir, '.chinese'))

# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cpu':
    print("⚠ 警告: 扩散模型推理在CPU上会非常慢，建议使用GPU")

# ====== 实验参数配置 ======
n_diffusion_steps = 100  # 扩散采样步数: 100(快速), 300(平衡), 1000(高质量)
show_progress = True  # 是否显示扩散采样进度条
use_cache = True  # ★ 是否启用结果缓存（避免重复计算）
cache_file = os.path.join(SAVE_DIR, 'experiment_cache.pkl')  # ★ 缓存文件路径

# 安装deepinv
try:
    import deepinv
except ImportError:
    print("正在安装 deepinv ...")
    os.system('pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv')
    import deepinv as dinv
else:
    dinv = deepinv
    print(f"deepinv 版本: {dinv.__version__}")

from deepinv.physics import Downsampling, Blur, Inpainting, GaussianNoise
from deepinv.utils import load_example


# ========================================================================
# 辅助函数
# ========================================================================
def compute_psnr(img1, img2):
    """计算PSNR (dB)"""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)

def compute_ssim(img1, img2):
    """简化SSIM计算（基于局部统计量）"""
    try:
        from skimage.metrics import structural_similarity as ssim
        img1_np = img1.squeeze().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        img2_np = img2.squeeze().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)
    except ImportError:
        # skimage不可用时返回NaN，避免不可比的近似值误导结果
        return float('nan')


# ========================================================================
# Step 1: 三种退化场景构造
# 对应18.4节知识点：去模糊/超分辨率/修复场景
# ========================================================================
print("\n" + "="*70)
print("Step 1: 三种退化场景构造")
print("="*70)

# 加载测试图像
x_true = load_example("celeba_example.jpg", img_size=(256, 256), resize_mode='resize')
print(f"load_example 返回 shape: {x_true.shape}, ndim: {x_true.ndim}")
# 确保是4D张量 (B, C, H, W)
if x_true.ndim == 3:
    x_true = x_true.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
elif x_true.ndim == 5:
    # 可能是 (1, 1, C, H, W)，需要squeeze
    x_true = x_true.squeeze(0) if x_true.shape[0] == 1 else x_true[0]
x_true = x_true.to(device)
print(f"处理后 x_true shape: {x_true.shape}")
print(f"测试图像 shape: {x_true.shape}")

# 定义退化场景
scenarios = {}

# 1a. 去模糊场景
print("\n构造去模糊场景...")
try:
    # ★ 修复：deepinv 的 gaussian_blur 只接受 sigma 参数，kernel_size 是自动计算的
    # 使用新的API路径，避免DeprecationWarning
    try:
        from deepinv.physics.functional import gaussian_blur as gauss_blur_func
        # ★ 注意：gaussian_blur 只接受 sigma 参数，不接受 kernel_size
        light_kernel = gauss_blur_func(sigma=(1.5, 1.5))
        heavy_kernel = gauss_blur_func(sigma=(3.0, 3.0))
    except ImportError:
        # 兼容旧版本
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            light_kernel = dinv.physics.blur.gaussian_blur(sigma=(1.5, 1.5))
            heavy_kernel = dinv.physics.blur.gaussian_blur(sigma=(3.0, 3.0))

    # 使用 "circular" padding 确保卷积输出与输入同尺寸
    # 注意：constant/valid padding 会导致输出尺寸变小，无法与原图比较PSNR
    light_blur = Blur(filter=light_kernel, padding="circular", device=device)
    light_blur.set_noise_model(GaussianNoise(sigma=0.01))

    # 重度模糊
    heavy_blur = Blur(filter=heavy_kernel, padding="circular", device=device)
    heavy_blur.set_noise_model(GaussianNoise(sigma=0.05))

    y_light_blur = light_blur(x_true)
    y_heavy_blur = heavy_blur(x_true)

    scenarios['轻度模糊'] = {'physics': light_blur, 'y': y_light_blur}
    scenarios['重度模糊'] = {'physics': heavy_blur, 'y': y_heavy_blur}
    print(f"  轻度模糊: σ_K=1.5, σ=0.01, 观测PSNR={compute_psnr(x_true, y_light_blur):.2f} dB")
    print(f"  重度模糊: σ_K=3.0, σ=0.05, 观测PSNR={compute_psnr(x_true, y_heavy_blur):.2f} dB")
    has_blur_scenario = True
except Exception as e:
    print(f"  模糊场景创建失败: {e}")
    import traceback
    traceback.print_exc()  # ★ 打印详细错误信息，便于调试
    has_blur_scenario = False

# 1b. 超分辨率场景
print("\n构造超分辨率场景...")
try:
    # 注意：filter参数是必需的，否则prox_l2会因Fh=None而失败
    # bicubic是推荐的默认选择，提供抗混叠滤波
    sr2 = Downsampling(factor=2, filter='bicubic', img_size=(3, 256, 256), device=device)
    sr2.set_noise_model(GaussianNoise(sigma=0.01))

    sr4 = Downsampling(factor=4, filter='bicubic', img_size=(3, 256, 256), device=device)
    sr4.set_noise_model(GaussianNoise(sigma=0.01))

    y_sr2 = sr2(x_true)
    y_sr4 = sr4(x_true)
    scenarios['2倍超分'] = {'physics': sr2, 'y': y_sr2}
    scenarios['4倍超分'] = {'physics': sr4, 'y': y_sr4}
    print(f"  2×超分: 观测shape={y_sr2.shape}, PSNR={compute_psnr(x_true, sr2.A_adjoint(y_sr2)):.2f} dB")
    print(f"  4×超分: 观测shape={y_sr4.shape}, PSNR={compute_psnr(x_true, sr4.A_adjoint(y_sr4)):.2f} dB")
    has_sr_scenario = True
except Exception as e:
    print(f"  超分场景创建失败: {e}")
    has_sr_scenario = False

# 1c. 修复场景
print("\n构造修复场景...")
try:
    inp50 = Inpainting(img_size=(3, 256, 256), mask=0.5, device=device)
    inp50.set_noise_model(GaussianNoise(sigma=0.01))

    inp80 = Inpainting(img_size=(3, 256, 256), mask=0.8, device=device)
    inp80.set_noise_model(GaussianNoise(sigma=0.01))

    y_inp50 = inp50(x_true)
    y_inp80 = inp80(x_true)
    scenarios['50%修复'] = {'physics': inp50, 'y': y_inp50}
    scenarios['80%修复'] = {'physics': inp80, 'y': y_inp80}
    print(f"  50%缺失: PSNR={compute_psnr(x_true, y_inp50):.2f} dB")
    print(f"  80%缺失: PSNR={compute_psnr(x_true, y_inp80):.2f} dB")
    has_inp_scenario = True
except Exception as e:
    print(f"  修复场景创建失败: {e}")
    has_inp_scenario = False

# Step 1 可视化
# ★ 动态调整子图数量：根据实际可用的场景数量决定布局
num_scenarios = sum([
    2 if has_blur_scenario else 0,  # 轻度+重度模糊
    1 if has_sr_scenario else 0,    # 4x超分
    2 if has_inp_scenario else 0,   # 50%+80%修复
])
total_images = 1 + num_scenarios  # 1个原始图像 + 退化场景

# 根据图片数量选择合适的布局
if total_images <= 4:
    nrows, ncols = 2, 2
elif total_images <= 6:
    nrows, ncols = 2, 3
else:
    nrows, ncols = 3, 3

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
# 如果只有一个子图，axes不是数组，需要特殊处理
if total_images == 1:
    axes = np.array([[axes]])
elif nrows == 1 or ncols == 1:
    axes = axes.reshape(nrows, ncols)

# 安全地显示图像，处理不同维度的情况
def safe_imshow(ax, img_tensor, title):
    """安全显示图像，自动处理3D/4D/5D张量"""
    if img_tensor.ndim == 5:
        img_tensor = img_tensor[0]  # (1, C, H, W) -> (C, H, W)
    if img_tensor.ndim == 4:
        img_tensor = img_tensor[0]  # (B, C, H, W) -> (C, H, W)
    # 现在是 (C, H, W)，permute to (H, W, C)
    ax.imshow(img_tensor.cpu().permute(1, 2, 0).clamp(0, 1))
    ax.set_title(title, fontsize=13)
    ax.axis('off')

safe_imshow(axes[0, 0], x_true, '原始图像')

vis_items = []
if has_blur_scenario:
    vis_items.append(('轻度模糊', y_light_blur))
    vis_items.append(('重度模糊', y_heavy_blur))
if has_sr_scenario:
    vis_items.append(('4×超分', y_sr4))
if has_inp_scenario:
    vis_items.append(('50%修复', y_inp50))
    vis_items.append(('80%修复', y_inp80))

# 填充退化场景图像
for idx, (title, img) in enumerate(vis_items):
    if idx + 1 < nrows * ncols:  # 确保不超出子图范围
        row, col = (idx + 1) // ncols, (idx + 1) % ncols
        axes[row, col].imshow(img[0].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[row, col].set_title(title, fontsize=12)
        axes[row, col].axis('off')

# 隐藏多余的子图
for idx in range(len(vis_items) + 1, nrows * ncols):
    row, col = idx // ncols, idx % ncols
    axes[row, col].axis('off')

fig.suptitle('Step 1: 三种退化场景', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step1_degradation_scenarios.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n已保存: step1_degradation_scenarios.png")


# ========================================================================
# Step 2: 优化方法求解
# 对应18.3节知识点：Tikhonov/TV正则化
# ========================================================================
print("\n" + "="*70)
print("Step 2: 优化方法求解（Tikhonov / TV）")
print("="*70)

# ★ 加载缓存（如果启用）
skip_step2_scenarios = set()  # ★ 只跳过已有缓存结果的场景
if use_cache and os.path.exists(cache_file):
    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"✓ 检测到缓存文件: {cache_file}")

        # 检查缓存中是否有 Step 2 的结果
        if 'step2_results' in cached_data:
            # ★ 从缓存恢复结果字典结构
            cached_step2 = cached_data['step2_results']
            results = {}
            for scenario_name, methods in cached_step2.items():
                results[scenario_name] = {}
                for method_name, metrics in methods.items():
                    # ★ 确保所有值都是 Python float 类型
                    results[scenario_name][method_name] = {
                        'psnr': float(metrics['psnr']),
                        'ssim': float(metrics.get('ssim', float('nan'))),
                        'time': float(metrics['time']),
                        'img': None  # 缓存中不保存图像，标记为 None
                    }
                skip_step2_scenarios.add(scenario_name)  # ★ 标记已有缓存结果的场景

            cached_s2 = list(skip_step2_scenarios)
            missing_s2 = [s for s in scenarios if s not in skip_step2_scenarios]
            print(f"✓ 从缓存加载了 Step 2 的优化方法结果: {cached_s2}")
            if missing_s2:
                print(f"  需要补算的场景: {missing_s2}")

            # 打印缓存的结果摘要
            for scenario_name, methods in results.items():
                print(f"  [{scenario_name}] 已有 {len(methods)} 个方法的结果")
            print("→ 如需重新计算，请设置 use_cache = False 或删除缓存文件")
            skip_step2 = len(missing_s2) == 0  # ★ 只有所有场景都有缓存时才完全跳过
        else:
            skip_step2 = False
            print("缓存中未找到 Step 2 结果，将执行计算...")
    except Exception as e:
        print(f"⚠ 缓存加载失败: {e}，将重新计算")
        skip_step2 = False
else:
    skip_step2 = False
    if use_cache:
        print("未检测到缓存文件，将执行完整计算...")

if not skip_step2:
    if not skip_step2_scenarios:
        results = {}  # 全部重新计算时才清空

# 对每种场景，用优化方法求解
for scenario_name, scenario_data in scenarios.items():
    # ★ 跳过已有缓存结果的场景
    if scenario_name in skip_step2_scenarios:
        print(f"\n--- 场景: {scenario_name} --- (缓存已加载，跳过)")
        continue
    physics = scenario_data['physics']
    y = scenario_data['y']
    print(f"\n--- 场景: {scenario_name} ---")

    # ★ 对超分场景，用双线性插值代替 A_adjoint 作为初始化
    #   原因：Downsampling 的 A_adjoint 是零填充上采样（极暗），双线性插值能提供更好的初值
    if isinstance(physics, Downsampling):
        x_init = F.interpolate(y, size=x_true.shape[-2:], mode='bilinear', align_corners=False)
    else:
        x_init = physics.A_adjoint(y)
    results[scenario_name] = {}

    # 2a. Tikhonov正则化 (L2正则化)
    try:
        # Tikhonov正则化: min_x ||Ax-y||² + λ||x||²
        # 使用梯度下降迭代求解（deepinv的physics是函数式API，不能直接访问矩阵A）
        lambda_reg = 0.1  # 正则化参数
        n_tik_iter = 100  # 迭代次数

        # 自适应步长: 用power iteration估计||A||²，保证收敛 lr < 2/(||A||²+λ)
        x_tmp = torch.randn_like(x_true)
        for _ in range(10):
            x_tmp = physics.A_adjoint(physics.A(x_tmp))
            norm_tmp = x_tmp.norm()
            if norm_tmp > 0:
                x_tmp = x_tmp / norm_tmp
        op_norm_sq = (physics.A(x_tmp)**2).sum() / (x_tmp**2).sum()
        lr_tik = 1.0 / (op_norm_sq.item() + lambda_reg)

        t_start = time.time()
        x_tikhonov = x_init.clone()
        for _ in range(n_tik_iter):
            # 梯度: ∇(||Ax-y||² + λ||x||²) = A^T(Ax-y) + λx
            grad = physics.A_adjoint(physics.A(x_tikhonov) - y) + lambda_reg * x_tikhonov
            x_tikhonov = x_tikhonov - lr_tik * grad
            x_tikhonov = x_tikhonov.clamp(0, 1)
        t_tikh = time.time() - t_start

        psnr_tikh = compute_psnr(x_true, x_tikhonov)
        ssim_tikh = compute_ssim(x_true, x_tikhonov)
        results[scenario_name]['Tikhonov'] = {
            'psnr': psnr_tikh, 'ssim': ssim_tikh, 'time': t_tikh, 'img': x_tikhonov, 'lambda': lambda_reg
        }
        print(f"  Tikhonov:   PSNR={psnr_tikh:.2f} dB, SSIM={ssim_tikh:.4f}, 耗时={t_tikh:.3f}s, λ={lambda_reg}")
    except Exception as e:
        print(f"  Tikhonov求解失败: {e}")

    # 2b. TV正则化
    # 教学说明：TV正则化求解 min_x ||Ax-y||² + λ||∇x||₁
    #   - 旧版deepinv使用PDIteration（Primal-Dual原始-对偶迭代）
    #   - 新版deepinv重构了优化框架，提供optim_builder/ADMM/PGD等新接口
    #   - 本代码采用"新API优先+手动实现保底"的策略，兼容各版本
    try:
        from deepinv.optim import TVPrior, L2 as DataFidelity_L2

        # TV: min_x ||Ax-y||² + λ||∇x||_1
        lambda_tv = 0.05  # TV正则化参数

        prior_tv = TVPrior(def_crit=1e-3, n_it_max=10)  # 教学参数：减少内层迭代，避免耗时过长
        data_fidelity_tv = DataFidelity_L2()

        # 尝试使用新版deepinv的优化器（优先方案）
        t_start = time.time()

        # 方案1: 尝试使用 optim_builder（如果可用）
        try:
            from deepinv.optim import optim_builder
            optimizer = optim_builder(
                'PGD',  # Proximal Gradient Descent
                data_fidelity=data_fidelity_tv,
                prior=prior_tv,
                params={'stepsize': 0.1, 'lambda': lambda_tv},
                max_iter=30  # 减少外层迭代：30次足以收敛到合理结果
            )
            x_tv = optimizer(y, physics)
            print("  ✓ 使用 optim_builder (PGD)")
        except ImportError:
            # 方案2: 尝试使用 ADMM
            try:
                from deepinv.optim import ADMM
                optimizer = ADMM(
                    data_fidelity=data_fidelity_tv,
                    prior=prior_tv,
                    max_iter=30,
                    params_algo={'lambda': lambda_tv}
                )
                x_tv = optimizer(y, physics)
                print("  ✓ 使用 ADMM 优化器")
            except ImportError:
                # 方案3: 尝试使用 PGD 类
                try:
                    from deepinv.optim import PGD
                    optimizer = PGD(
                        data_fidelity=data_fidelity_tv,
                        prior=prior_tv,
                        max_iter=30,
                        params_algo={'stepsize': 0.1, 'lambda': lambda_tv}
                    )
                    x_tv = optimizer(y, physics)
                    print("  ✓ 使用 PGD 优化器")
                except ImportError:
                    raise ImportError("新版优化器均不可用")

        t_tv = time.time() - t_start
        psnr_tv = compute_psnr(x_true, x_tv)
        ssim_tv = compute_ssim(x_true, x_tv)
        results[scenario_name]['TV正则化'] = {
            'psnr': psnr_tv, 'ssim': ssim_tv, 'time': t_tv, 'img': x_tv, 'lambda': lambda_tv
        }
        print(f"  TV(新优化器): PSNR={psnr_tv:.2f} dB, SSIM={ssim_tv:.4f}, 耗时={t_tv:.3f}s, λ={lambda_tv}")

    except Exception as e:
        print(f"  TV正则化(新优化器)失败: {e}")
        print("  回退到手动TV梯度下降...")
        print("  [教学提示] 手动实现虽然较慢，但能清晰展示TV优化的核心思想")
        # 备用方案：手动梯度下降 + TV近端算子
        try:
            print("  尝试手动TV梯度下降...")
            lambda_tv = 0.05
            lr_tv = 0.01
            n_tv_iter = 100  # 减少迭代次数
            x_tv = x_init.clone().detach().requires_grad_(False)
            t_start = time.time()
            for it in range(n_tv_iter):
                # 数据保真项梯度
                grad_data = physics.A_adjoint(physics.A(x_tv) - y)
                # TV梯度（各向异性TV，零边界Neumann条件）
                #   ∇·sign(∇x) 的离散散度，使用零边界条件：
                #   内部: (sign(∇x[i-1]) - sign(∇x[i]))
                #   边界: 边界外侧的 ∇x 视为 0，所以只保留一个邻居的贡献
                #   即位置 0 处的散度 = -sign(∇x[0])，位置 W-1 处的散度 = sign(∇x[W-2])
                #   此处改用 F.pad+差分 的标准实现，更清晰且与deepinv默认实现一致
                dx = torch.diff(x_tv, dim=-1)  # (B,C,H,W-1)
                dy = torch.diff(x_tv, dim=-2)  # (B,C,H-1,W)
                s_x = torch.sign(dx)
                s_y = torch.sign(dy)
                # 水平方向散度: 用零填充后做差分
                # F.pad(s_x, (1,1)) → 在 W 维两端各补 0，然后 diff 得到 (B,C,H,W)
                tv_grad_x = -torch.diff(F.pad(s_x, (1, 1), mode='constant', value=0), dim=-1)
                # 垂直方向散度
                tv_grad_y = -torch.diff(F.pad(s_y, (1, 1), mode='constant', value=0), dim=-2)
                tv_grad = tv_grad_x + tv_grad_y  # (B,C,H,W)
                x_tv = x_tv - lr_tv * (grad_data + lambda_tv * tv_grad)
                x_tv = x_tv.clamp(0, 1)
                if (it + 1) % 25 == 0:
                    print(f"    TV迭代 {it+1}/{n_tv_iter} 完成")
            t_tv = time.time() - t_start

            psnr_tv = compute_psnr(x_true, x_tv)
            ssim_tv = compute_ssim(x_true, x_tv)
            results[scenario_name]['TV正则化(手动)'] = {
                'psnr': psnr_tv, 'ssim': ssim_tv, 'time': t_tv, 'img': x_tv, 'lambda': lambda_tv
            }
            print(f"  TV手动:     PSNR={psnr_tv:.2f} dB, SSIM={ssim_tv:.4f}, 耗时={t_tv:.3f}s, λ={lambda_tv}")
        except Exception as e2:
            import traceback
            print(f"  手动TV也失败: {e2}")
            traceback.print_exc()

    # 2c. 伴随重建（超分场景用双线性插值，其他场景用A_adjoint）
    try:
        t_start = time.time()
        x_adj = x_init.clone()
        t_adj = time.time() - t_start

        psnr_adj = compute_psnr(x_true, x_adj)
        ssim_adj = compute_ssim(x_true, x_adj)
        results[scenario_name]['伴随重建'] = {
            'psnr': psnr_adj, 'ssim': ssim_adj, 'time': t_adj, 'img': x_adj
        }
        print(f"  伴随重建:   PSNR={psnr_adj:.2f} dB, SSIM={ssim_adj:.4f}, 耗时={t_adj:.3f}s")
    except Exception as e:
        print(f"  伴随重建失败: {e}")

    print("\n优化方法求解完成")

    # ★ 保存 Step 2 结果到缓存
    if use_cache:
        try:
            # ★ 注意：不保存 scenarios（包含不可序列化的 physics 对象）
            # 只保存计算结果（纯数据，可序列化）
            cached_data = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                except:
                    cached_data = {}  # 如果加载失败，从头开始

            # 更新 Step 2 结果（只保存指标，不保存图像张量以减小文件大小）
            step2_data = {}
            for scenario_name, methods in results.items():
                step2_data[scenario_name] = {}
                for method_name, metrics in methods.items():
                    # ★ 关键：确保所有值都是 Python 原生类型，不是 numpy 或 torch 类型
                    psnr_val = float(metrics['psnr']) if hasattr(metrics['psnr'], 'item') else metrics['psnr']
                    ssim_val = float(metrics.get('ssim', float('nan'))) if hasattr(metrics.get('ssim', 0), 'item') else metrics.get('ssim', float('nan'))
                    time_val = float(metrics['time']) if hasattr(metrics['time'], 'item') else metrics['time']

                    step2_data[scenario_name][method_name] = {
                        'psnr': psnr_val,
                        'ssim': ssim_val,
                        'time': time_val
                    }

            cached_data['step2_results'] = step2_data

            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"✓ Step 2 结果已保存到缓存: {cache_file}")
        except Exception as e:
            print(f"⚠ 缓存保存失败: {e}")
else:
    print("\n✓ 使用缓存的优化方法结果，跳过 Step 2 计算")


# ========================================================================
# Step 3: PnP方法求解 —— DPIR
# 对应18.3节知识点：PnP-ADMM / DPIR
# ========================================================================
print("\n" + "="*70)
print("Step 3: PnP方法求解 —— DPIR (DRUNet去噪器)")
print("="*70)

# ★ 检查是否需要执行 Step 3
skip_step3 = False
skip_step3_scenarios = set()  # ★ 只跳过已有缓存结果的场景
if use_cache and os.path.exists(cache_file):
    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        if 'step3_results' in cached_data:
            # 合并 Step 3 的结果到 results
            for scenario_name, methods in cached_data['step3_results'].items():
                if scenario_name not in results:
                    results[scenario_name] = {}
                for method_name, metrics in methods.items():
                    results[scenario_name][method_name] = {
                        'psnr': float(metrics['psnr']),
                        'ssim': float(metrics.get('ssim', float('nan'))),
                        'time': float(metrics['time']),
                        'img': None  # 缓存中不保存图像
                    }
                skip_step3_scenarios.add(scenario_name)  # ★ 标记已有缓存结果的场景
            cached_s3 = list(skip_step3_scenarios)
            missing_s3 = [s for s in scenarios if s not in skip_step3_scenarios]
            print(f"✓ 从缓存加载了 Step 3 的 PnP 方法结果: {cached_s3}")
            if missing_s3:
                print(f"  需要补算的场景: {missing_s3}")
            skip_step3 = len(missing_s3) == 0  # ★ 只有所有场景都有缓存时才完全跳过
        else:
            print("缓存中未找到 Step 3 结果，将执行计算...")
    except Exception as e:
        print(f"⚠ 缓存加载失败: {e}，将重新计算")
else:
    if use_cache:
        print("未检测到缓存文件，将执行完整计算...")

print("""
DPIR算法核心思路:
  将优化问题的近端算子替换为预训练去噪器DRUNet
  通过递减噪声水平表 σ_k 逐步改善重建质量
  优势: 不需要针对特定逆问题训练，开箱即用
""")

# 加载预训练DRUNet去噪器
try:
    from deepinv.models import DRUNet
    denoiser_drunet = DRUNet(pretrained='download').to(device)
    print("DRUNet预训练模型加载成功")
    has_dpir = True
except Exception as e:
    print(f"DRUNet加载失败: {e}")
    has_dpir = False

if has_dpir and not skip_step3:
    # 教学说明：DPIR (Deep Plug-and-Play Image Restoration)
    #   - 核心思想：将优化问题的近端算子替换为预训练去噪器
    #   - 旧版deepinv使用DPIR类封装PnP-ADMM/HQS算法
    #   - 新版deepinv可能重构了接口，本代码采用"新API优先+手动实现保底"
    for scenario_name, scenario_data in scenarios.items():
        # ★ 跳过已有缓存结果的场景
        if scenario_name in skip_step3_scenarios:
            print(f"\n--- 场景: {scenario_name} --- (缓存已加载，跳过)")
            continue
        physics = scenario_data['physics']
        y = scenario_data['y']
        print(f"\n--- 场景: {scenario_name} ---")

        # 根据场景确定噪声水平（DPIR需要此参数来设置迭代参数）
        if '重度' in scenario_name:
            noise_sigma = 0.05
        elif '80%' in scenario_name:
            noise_sigma = 0.01
        else:
            noise_sigma = 0.01

        try:
            from deepinv.optim.optimizers import optim_builder
            from deepinv.optim.prior import PnP
            from deepinv.optim.data_fidelity import L2
            from deepinv.optim.dpir import get_DPIR_params

            # 使用 optim_builder 方式创建 DPIR（更稳定）
            sigma_denoiser, stepsize, max_iter = get_DPIR_params(noise_sigma, device=device)
            params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
            prior = PnP(denoiser=denoiser_drunet)
            data_fidelity = L2()

            t_start = time.time()
            model = optim_builder(
                iteration="HQS",
                prior=prior,
                data_fidelity=data_fidelity,
                early_stop=False,
                max_iter=max_iter,
                verbose=False,
                params_algo=params_algo,
            )
            x_dpir = model(y, physics)
            x_dpir = x_dpir.clamp(0, 1)  # ★ 确保输出在[0,1]范围
            t_dpir = time.time() - t_start

            psnr_dpir = compute_psnr(x_true, x_dpir)
            ssim_dpir = compute_ssim(x_true, x_dpir)
            results[scenario_name]['DPIR(PnP)'] = {
                'psnr': psnr_dpir, 'ssim': ssim_dpir, 'time': t_dpir, 'img': x_dpir
            }
            print(f"  DPIR:  PSNR={psnr_dpir:.2f} dB, SSIM={ssim_dpir:.4f}, 耗时={t_dpir:.3f}s")
        except Exception as e:
            import traceback
            print(f"  DPIR(新API)求解失败: {e}")
            print(f"  [调试信息] physics类型: {type(physics).__name__}")
            print(f"  [调试信息] y shape: {y.shape if hasattr(y, 'shape') else 'N/A'}")
            traceback.print_exc()
            print("  回退到手动PnP-HQS实现...")
            try:
                print("  尝试手动PnP-HQS实现...")

                # PnP-HQS: x_{k+1} = denoiser(prox_{data}(x_k))
                # 教学提示：手动实现清晰展示了PnP的核心思想
                x_pnp = x_init.clone()
                n_iter = 20  # 减少迭代次数，加快演示
                sigma_pnp = noise_sigma * 2  # 初始噪声水平
                step_size = 0.5  # 适中的步长
                t_start = time.time()

                for it in range(n_iter):
                    # 数据保真项梯度步
                    grad = physics.A_adjoint(physics.A(x_pnp) - y)
                    x_pnp = x_pnp - step_size * grad
                    # 去噪步
                    sigma_cur = max(sigma_pnp * (1 - it / n_iter), 0.01)
                    noise_level = torch.tensor([sigma_cur]).to(device)
                    x_pnp = denoiser_drunet(x_pnp, noise_level)
                    x_pnp = x_pnp.clamp(0, 1)  # ★ 确保输出在[0,1]范围
                    if (it + 1) % 5 == 0:
                        print(f"    迭代 {it+1}/{n_iter} 完成")

                t_pnp = time.time() - t_start
                psnr_pnp = compute_psnr(x_true, x_pnp)
                ssim_pnp = compute_ssim(x_true, x_pnp)
                results[scenario_name]['PnP-HQS(手动)'] = {
                    'psnr': psnr_pnp, 'ssim': ssim_pnp, 'time': t_pnp, 'img': x_pnp
                }
                print(f"  PnP-HQS:  PSNR={psnr_pnp:.2f} dB, SSIM={ssim_pnp:.4f}, 耗时={t_pnp:.3f}s")
            except Exception as e2:
                import traceback
                print(f"  手动PnP也失败: {e2}")
                traceback.print_exc()

    # ★ 保存 Step 3 结果到缓存
    if use_cache and not skip_step3:
        try:
            # 读取现有缓存
            cached_data = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                except:
                    cached_data = {}

            # 提取 Step 3 的结果（只包含 PnP 相关方法，只保存指标）
            step3_results = {}
            for scenario_name, methods in results.items():
                pnp_methods = {k: v for k, v in methods.items() if 'DPIR' in k or 'PnP' in k}
                if pnp_methods:
                    step3_results[scenario_name] = {}
                    for method_name, metrics in pnp_methods.items():
                        # ★ 确保所有值都是 Python 原生类型
                        psnr_val = float(metrics['psnr']) if hasattr(metrics['psnr'], 'item') else metrics['psnr']
                        ssim_val = float(metrics.get('ssim', float('nan'))) if hasattr(metrics.get('ssim', 0), 'item') else metrics.get('ssim', float('nan'))
                        time_val = float(metrics['time']) if hasattr(metrics['time'], 'item') else metrics['time']

                        step3_results[scenario_name][method_name] = {
                            'psnr': psnr_val,
                            'ssim': ssim_val,
                            'time': time_val
                        }

            cached_data['step3_results'] = step3_results

            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"✓ Step 3 结果已保存到缓存: {cache_file}")
        except Exception as e:
            print(f"⚠ 缓存保存失败: {e}")
elif skip_step3:
    print("\n✓ 使用缓存的 PnP 方法结果，跳过 Step 3 计算")


# ========================================================================
# Step 4: 扩散方法求解 —— DiffPIR
# 对应18.3节知识点：扩散采样求解逆问题
# ========================================================================
print("\n" + "="*70)
print("Step 4: 扩散方法求解 —— DiffPIR (DiffUNet)")
print("="*70)

# ★ 检查是否需要执行 Step 4
skip_step4 = False
skip_step4_scenarios = set()  # ★ 只跳过已有缓存结果的场景
if use_cache and os.path.exists(cache_file):
    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        if 'step4_results' in cached_data:
            # 合并 Step 4 的结果到 results
            for scenario_name, methods in cached_data['step4_results'].items():
                if scenario_name not in results:
                    results[scenario_name] = {}
                for method_name, metrics in methods.items():
                    results[scenario_name][method_name] = {
                        'psnr': float(metrics['psnr']),
                        'ssim': float(metrics.get('ssim', float('nan'))),
                        'time': float(metrics['time']),
                        'img': None  # 缓存中不保存图像
                    }
                skip_step4_scenarios.add(scenario_name)  # ★ 标记已有缓存结果的场景
            cached_s4 = list(skip_step4_scenarios)
            missing_s4 = [s for s in scenarios if s not in skip_step4_scenarios]
            print(f"✓ 从缓存加载了 Step 4 的扩散方法结果: {cached_s4}")
            if missing_s4:
                print(f"  需要补算的场景: {missing_s4}")
            skip_step4 = len(missing_s4) == 0  # ★ 只有所有场景都有缓存时才完全跳过
        else:
            print("缓存中未找到 Step 4 结果，将执行计算...")
    except Exception as e:
        print(f"⚠ 缓存加载失败: {e}，将重新计算")
else:
    if use_cache:
        print("未检测到缓存文件，将执行完整计算...")

print("""
DiffPIR算法核心思路:
  在扩散采样过程中融入数据保真约束
  每个去噪步骤后，用数据一致性步骤将采样引导向观测一致
  优势: 自然图像先验更强，对重度退化效果更好
  劣势: 采样步数多（可配置100-1000步），推理较慢

★ 当前配置: 采样步数={n_diffusion_steps}，进度条={'开启' if show_progress else '关闭'}
""")


# 加载预训练DiffUNet
try:
    from deepinv.models import DiffUNet
    denoiser_diffunet = DiffUNet(pretrained='download').to(device)
    print("DiffUNet预训练模型加载成功")
    has_diffpir = True
except Exception as e:
    print(f"DiffUNet加载失败: {e}")
    has_diffpir = False

if has_diffpir and not skip_step4:
    # 教学说明：DiffPIR (Diffusion Plug-and-Play Image Restoration)
    #   - 核心思想：将扩散模型作为PnP去噪器，结合数据保真项
    #   - 旧版deepinv使用DiffPIR类封装扩散采样算法
    #   - 新版deepinv可能重构了接口，本代码采用"新API优先+DDRM备选"
    # 选择代表性场景测试扩散方法（较慢，不全测）
    diff_scenarios = list(scenarios.keys())[:4]  # 最多测4个场景（含2倍和4倍超分）
    for scenario_name in diff_scenarios:
        # ★ 跳过已有缓存结果的场景
        if scenario_name in skip_step4_scenarios:
            print(f"\n--- 场景: {scenario_name} --- (缓存已加载，跳过)")
            continue
        scenario_data = scenarios[scenario_name]
        physics = scenario_data['physics']
        y = scenario_data['y']
        print(f"\n--- 场景: {scenario_name} ---")

        try:
            from deepinv.sampling import DiffPIR as DiffPIR_algo
            from deepinv.optim import L2 as DiffL2

            t_start = time.time()
            model_diffpir = DiffPIR_algo(
                model=denoiser_diffunet,
                data_fidelity=DiffL2(),
                max_iter=n_diffusion_steps,
                sigma=0.05,
                zeta=0.1,  # ★ 似然项权重，控制数据保真强度
                lambda_=7.0,  # ★ 噪声调度系数，影响采样步长
                verbose=show_progress,  # ★ 启用进度条
                device=device,
            )
            x_diffpir = model_diffpir(y, physics)
            x_diffpir = x_diffpir.clamp(0, 1)  # ★ 确保输出在[0,1]范围，避免显示过暗或过亮
            t_diffpir = time.time() - t_start

            psnr_diffpir = compute_psnr(x_true, x_diffpir)
            ssim_diffpir = compute_ssim(x_true, x_diffpir)
            results[scenario_name]['DiffPIR(扩散)'] = {
                'psnr': psnr_diffpir, 'ssim': ssim_diffpir, 'time': t_diffpir, 'img': x_diffpir
            }
            print(f"  DiffPIR:  PSNR={psnr_diffpir:.2f} dB, SSIM={ssim_diffpir:.4f}, 耗时={t_diffpir:.3f}s")
        except Exception as e:
            print(f"  DiffPIR(新API)求解失败: {e}")
            # 回退方案：先尝试DDRM，再尝试DPS（不需要SVD，适合Blur场景）
            try:
                print("  回退到DDRM算法...")
                from deepinv.sampling import DDRM
                t_start = time.time()
                model_ddrm = DDRM(
                    denoiser=denoiser_diffunet,
                    sigmas=np.linspace(1, 0, n_diffusion_steps),
                    verbose=show_progress
                )
                x_ddrm = model_ddrm(y, physics)
                x_ddrm = x_ddrm.clamp(0, 1)  # ★ 确保输出在[0,1]范围
                t_ddrm = time.time() - t_start

                psnr_ddrm = compute_psnr(x_true, x_ddrm)
                ssim_ddrm = compute_ssim(x_true, x_ddrm)
                results[scenario_name]['DDRM(扩散)'] = {
                    'psnr': psnr_ddrm, 'ssim': ssim_ddrm, 'time': t_ddrm, 'img': x_ddrm
                }
                print(f"  DDRM:  PSNR={psnr_ddrm:.2f} dB, SSIM={ssim_ddrm:.4f}, 耗时={t_ddrm:.3f}s")
            except Exception as e2:
                print(f"  DDRM也失败: {e2}")
                # ★ DPS (Diffusion Posterior Sampling) 不需要SVD分解，适合Blur等无SVD的场景
                try:
                    print("  回退到DPS算法（无需SVD，适合Blur场景）...")
                    from deepinv.sampling import DPS
                    t_start = time.time()
                    model_dps = DPS(
                        model=denoiser_diffunet,
                        data_fidelity=DiffL2(),
                        max_iter=n_diffusion_steps,
                        verbose=show_progress
                    )
                    x_dps = model_dps(y, physics)
                    x_dps = x_dps.clamp(0, 1)  # ★ 确保输出在[0,1]范围
                    t_dps = time.time() - t_start

                    psnr_dps = compute_psnr(x_true, x_dps)
                    ssim_dps = compute_ssim(x_true, x_dps)
                    results[scenario_name]['DPS(扩散)'] = {
                        'psnr': psnr_dps, 'ssim': ssim_dps, 'time': t_dps, 'img': x_dps
                    }
                    print(f"  DPS:  PSNR={psnr_dps:.2f} dB, SSIM={ssim_dps:.4f}, 耗时={t_dps:.3f}s")
                except Exception as e3:
                    print(f"  DPS也失败: {e3}")
                    print(f"  [提示] 场景 '{scenario_name}' 的所有扩散方法均失败，Step 5可视化中将缺少扩散方法子图")

    # ★ 保存 Step 4 结果到缓存
    if use_cache and not skip_step4:
        try:
            # 读取现有缓存
            cached_data = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                except:
                    cached_data = {}

            # 提取 Step 4 的结果（只包含扩散相关方法，只保存指标）
            step4_results = {}
            for scenario_name, methods in results.items():
                diff_methods = {k: v for k, v in methods.items() if 'DiffPIR' in k or 'DDRM' in k or 'DPS' in k}
                if diff_methods:
                    step4_results[scenario_name] = {}
                    for method_name, metrics in diff_methods.items():
                        # ★ 确保所有值都是 Python 原生类型
                        psnr_val = float(metrics['psnr']) if hasattr(metrics['psnr'], 'item') else metrics['psnr']
                        ssim_val = float(metrics.get('ssim', float('nan'))) if hasattr(metrics.get('ssim', 0), 'item') else metrics.get('ssim', float('nan'))
                        time_val = float(metrics['time']) if hasattr(metrics['time'], 'item') else metrics['time']

                        step4_results[scenario_name][method_name] = {
                            'psnr': psnr_val,
                            'ssim': ssim_val,
                            'time': time_val
                        }

            cached_data['step4_results'] = step4_results

            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"✓ Step 4 结果已保存到缓存: {cache_file}")
        except Exception as e:
            print(f"⚠ 缓存保存失败: {e}")
elif skip_step4:
    print("\n✓ 使用缓存的扩散方法结果，跳过 Step 4 计算")


# ========================================================================
# Step 5: ★ 方法对比与决策指南
# 对应18.3节知识点：方法选择决策树
# ========================================================================
print("\n" + "="*70)
print("Step 5: ★ 方法对比与决策指南")
print("="*70)

# 5a. 汇总表格
print("\n--- 5a. PSNR/SSIM对比表格 ---")
print(f"{'场景':<12} {'方法':<16} {'PSNR(dB)':<10} {'SSIM':<10} {'耗时(s)':<10}")
print("-" * 60)
for scenario_name, methods in results.items():
    for method_name, metrics in methods.items():
        ssim_val = metrics.get('ssim', float('nan'))
        print(f"{scenario_name:<12} {method_name:<16} {metrics['psnr']:<10.2f} {ssim_val:<10.4f} {metrics['time']:<10.3f}")
    print("-" * 60)

# 5b. ★ 决策指南可视化
print("\n--- 5b. ★ 决策指南可视化 ---")

# 找出有足够结果的场景来画图
plot_scenarios = [s for s in results.keys() if len(results[s]) >= 2]

if len(plot_scenarios) >= 1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: PSNR柱状图
    x_pos = np.arange(len(plot_scenarios))
    width = 0.15
    all_methods = set()
    for s in plot_scenarios:
        all_methods.update(results[s].keys())
    all_methods = sorted(all_methods)
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_methods)))

    for i, method in enumerate(all_methods):
        psnrs = []
        for s in plot_scenarios:
            if method in results[s]:
                psnrs.append(results[s][method]['psnr'])
            else:
                psnrs.append(0)
        axes[0].bar(x_pos + i * width, psnrs, width, label=method, color=colors[i])

    axes[0].set_xlabel('退化场景', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('★ 各方法PSNR对比', fontsize=13)
    axes[0].set_xticks(x_pos + width * (len(all_methods) - 1) / 2)
    axes[0].set_xticklabels(plot_scenarios, fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    # 右图: ★ 时间-质量权衡图 (Pareto前沿)
    markers = ['o', 's', '^', 'D', 'v']
    for i, method in enumerate(all_methods):
        times = []
        psnrs = []
        for s in plot_scenarios:
            if method in results[s]:
                times.append(results[s][method]['time'])
                psnrs.append(results[s][method]['psnr'])
        if times:
            axes[1].scatter(times, psnrs, marker=markers[i % len(markers)],
                           s=100, label=method, color=colors[i], zorder=5)
            for j, s in enumerate(plot_scenarios):
                if method in results[s]:
                    axes[1].annotate(s, (times[j], psnrs[j]),
                                    fontsize=8, xytext=(5, 5), textcoords='offset points')

    axes[1].set_xlabel('推理时间 (秒)', fontsize=12)
    axes[1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1].set_title('★ 时间-质量权衡 (Pareto)', fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle('Step 5: ★ 端到端求解策略对比与决策指南', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'step5_method_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: step5_method_comparison.png")

# 5c. 重建结果可视化
print("\n--- 5c. 重建结果可视化 ---")
# ★ 为每个场景都生成重建对比图
for scenario_name, scenario_methods in results.items():
    num_methods = len(scenario_methods)
    if num_methods == 0:
        print(f"  场景 '{scenario_name}' 无结果，跳过")
        continue

    # 动态调整子图数量：真值 + 观测 + 所有方法的结果
    total_subplots = 2 + num_methods
    ncols = min(total_subplots, 5)
    nrows = (total_subplots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if total_subplots == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    axes_flat = axes.flatten()

    # 显示真值
    safe_imshow(axes_flat[0], x_true, '真值')

    # 显示观测图像（如果场景存在）
    if scenario_name in scenarios:
        safe_imshow(axes_flat[1], scenarios[scenario_name]['y'], f'观测\n({scenario_name})')
    else:
        axes_flat[1].text(0.5, 0.5, '观测数据\n不可用', ha='center', va='center', fontsize=12)
        axes_flat[1].axis('off')

    # 显示各方法的重建结果
    for idx, (method_name, metrics) in enumerate(scenario_methods.items()):
        ax_idx = idx + 2
        if ax_idx < len(axes_flat):
            img = metrics.get('img')
            if img is not None:
                safe_imshow(axes_flat[ax_idx], img, f'{method_name}\nPSNR={metrics["psnr"]:.1f}dB, SSIM={metrics.get("ssim", 0):.3f}')
            else:
                # img 为 None（缓存加载）：逐级降级显示
                displayed = False
                if scenario_name in scenarios:
                    physics = scenarios[scenario_name]['physics']
                    y_obs = scenarios[scenario_name]['y']
                    # ★ 对超分场景，A_adjoint 是零填充（极暗），改用双线性插值
                    try:
                        from deepinv.physics import Downsampling
                        if isinstance(physics, Downsampling):
                            import torch.nn.functional as F
                            target_h, target_w = x_true.shape[-2], x_true.shape[-1]
                            adj = F.interpolate(y_obs, size=(target_h, target_w), mode='bilinear', align_corners=False)
                        else:
                            adj = physics.A_adjoint(y_obs)
                        safe_imshow(axes_flat[ax_idx], adj, f'{method_name}\nPSNR={metrics["psnr"]:.1f}dB (伴随)')
                        displayed = True
                    except Exception:
                        pass
                # 尝试直接显示观测图
                if not displayed and scenario_name in scenarios and 'y' in scenarios[scenario_name]:
                    safe_imshow(axes_flat[ax_idx], scenarios[scenario_name]['y'], f'{method_name}\nPSNR={metrics["psnr"]:.1f}dB (观测)')
                    displayed = True
                # 文字占位
                if not displayed:
                    axes_flat[ax_idx].text(0.5, 0.5, f'{method_name}\nPSNR={metrics["psnr"]:.1f}dB\n(图像未保存)',
                                          ha='center', va='center', fontsize=10)
                    axes_flat[ax_idx].axis('off')

    # 隐藏多余的子图
    for idx in range(2 + num_methods, len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.suptitle(f'重建对比: {scenario_name}', fontsize=14)
    plt.tight_layout()
    # 文件名中替换特殊字符
    safe_name = scenario_name.replace('×', 'x').replace('=', '_')
    plt.savefig(os.path.join(SAVE_DIR, f'step5_reconstruction_{safe_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: step5_reconstruction_{safe_name}.png")

if not results:
    print("  ⚠ 没有可用的结果进行可视化")

# 5d. 决策树文字输出
print("""
★ 方法选择决策指南（对应18.3节决策树）:

┌─ 是否需要不确定性量化？
│  ├─ 是 → 扩散方法（DiffPIR/DDRM/DPS），可产生后验样本
│  └─ 否 ↓
│
├─ 退化程度如何？
│  ├─ 轻度（高PSNR观测）→ 优化方法（TV/Tikhonov）或PnP（DPIR）
│  │   优势: 快速、稳定、不需预训练模型
│  │
│  ├─ 中等 → PnP（DPIR/DRUNet）
│  │   优势: 质量好、速度适中、通用性强
│  │
│  └─ 重度（低PSNR观测）→ 扩散方法（DiffPIR/DPS）
│      优势: 更强先验、重建质量更高
│      劣势: 推理慢（需~1000步采样）
│
└─ 时间约束？
   ├─ 严格（<1s）→ 优化方法
   ├─ 中等（1-10s）→ PnP
   └─ 宽松（>10s）→ 扩散方法

★ 扩散采样配置建议：
   - 采样步数 (n_diffusion_steps):
     * 100步：快速预览，适合调试和初步实验
     * 300步：平衡质量和速度，推荐日常使用
     * 1000步：最高质量，适合最终结果生成
   - 进度条 (show_progress):
     * True：显示tqdm进度条，实时了解采样进度
     * False：静默模式，减少输出干扰
   - 当前设置: n_diffusion_steps={n_diffusion_steps}, show_progress={show_progress}
""")


# ========================================================================
# 实验总结
# ========================================================================
print("\n" + "="*70)
print("实验18.4-1 总结")
print("="*70)
print(f"""
本实验对应18.3-18.4节核心知识点：

1. 三种退化场景 ✓
   - 去模糊（轻度σ_K=1.5/重度σ_K=3.0）
   - 超分辨率（4×下采样）
   - 修复（50%/80%缺失）

2. 优化方法 ✓
   - Tikhonov: L2正则化，速度快
   - TV正则化: 保边缘，轻度退化效果好
   - 伴随重建: 零填充基线

3. PnP方法 ✓
   - DPIR: DRUNet去噪器+递减噪声表
   - PnP-HQS手动实现（备用方案）

4. 扩散方法 ✓
   - DiffPIR: DiffUNet+数据一致性 (采样步数={n_diffusion_steps})
   - DDRM: 备选扩散求解器

5. ★ 方法对比与决策 ✓
   - PSNR/时间对比表格
   - 时间-质量权衡图
   - 决策树指南

关键发现:
- 优化方法: 速度快，轻度退化效果好
- PnP方法: 速度与质量平衡最佳
- 扩散方法: 质量最高但推理慢

★ 结果缓存功能:
- 缓存文件: {cache_file if use_cache else '未启用'}
- 缓存状态: {'已启用' if use_cache else '已禁用'}
- 优势: 避免重复计算，加速调试和实验迭代
- 使用: 设置 use_cache = False 可强制重新计算
- 清理: 删除缓存文件即可清除所有缓存结果

所有图像已保存至: {SAVE_DIR}
""")

# 保存数值结果到文件
import json
results_summary = {}
for s_name, methods in results.items():
    results_summary[s_name] = {}
    for m_name, m_data in methods.items():
        results_summary[s_name][m_name] = {
            'psnr': float(round(m_data['psnr'], 2)),  # ★ 转换为 Python float
            'ssim': float(round(m_data.get('ssim', 0), 4)),  # ★ 转换为 Python float
            'time': float(round(m_data['time'], 3))  # ★ 转换为 Python float
        }
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print("数值结果已保存: results_summary.json")
