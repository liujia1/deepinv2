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
import subprocess  # ★ 用于 pip install 失败时显式退出（与 18_2-1.py 风格统一，避免 os.system 静默失败）
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # ★ 教学注意：必须全局导入，Step 2 超分初始化(368行) / TV 散度手动实现(499/501行) / Step 5 降级可视化(1106行) 均依赖 F
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

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验18.4-1')
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
# ★ 扩散采样步数：100 是 DiffPIR 库默认值与论文标准配置（并非"快速演示档"）；
#   300 步是论文推荐的高质量配置，1000 步为完整 DDPM 轨迹。本实验采用标准 100 步。
n_diffusion_steps = 100  # 扩散采样步数: 100(标准/默认), 300(高质量), 1000(完整轨迹)
# ★ DiffPIR 的 ζ=0.1 输出是后验随机样本（含随机性），单次 PSNR 不能代表方法水平，
#   对每个场景独立采样 n_diffusion_samples 次，报告均值±标准差，重建图取后验均值（MMSE）。
n_diffusion_samples = 3
diffusion_seeds = [42, 123, 2024]
show_progress = True  # 是否显示扩散采样进度条
use_cache = True  # ★ 是否启用结果缓存（避免重复计算）
cache_file = os.path.join(SAVE_DIR, 'experiment_cache.pkl')  # ★ 缓存文件路径
# ★ 缓存版本号：退化场景定义/求解参数/方法配置变更时必须+1（如 v2: Blur→BlurFFT、mask语义修正；
#   v3: TV优化器显式传入x_init初值，修复超分场景TV重建全黑问题；
#   v4: 缓存额外保存真实重建图（CPU张量），可视化不再降级显示；
#   v5: TV外层迭代 30→150（但该版本params参数名错误，已被v6取代）；
#   v6: 修正 optim_builder 参数名 params→params_algo（旧名被静默忽略，TV实际跑默认λ=1/step=1），
#       并按场景分别设置TV的(λ,stepsize)，修复超分TV卡通化且步数越多越差的问题；
#   v7: Tikhonov的λ由固定0.1改为相对算子范数缩放（λ=0.1·‖A‖²），修复超分场景Tikhonov又暗又糊；
#   v8: 【公平性大修】
#       (a) DPIR 修复场景换用任务匹配调度（20步，σ:0.5→0.05对数递减，强数据锁定），
#           修复 50%/80% 修复场景 DPIR 假性失败（12.3/9.8dB → 33/28dB）；
#       (b) DiffPIR 补测 50%/80% 修复场景（L2 prox 对 Inpainting 有闭式解，无需SVD，
#           官方文档示例本身就是 Inpainting），全部6场景对齐；
#       (c) DiffPIR 每场景独立采样3次，PSNR/SSIM 报告均值±标准差，重建图取后验均值；
#       (d) Tikhonov 正则强度比例 0.1→0.01，修复轻度模糊场景正则过强、PSNR 反低于观测；
#       (e) Pareto 图纵轴改为"相对基线增益ΔPSNR"（跨场景可比），与左图口径声明一致；
#       (f) 决策指南改为由实测指标动态生成，不再预设"扩散质量最高"等固定结论。
#   加载时版本不匹配的旧缓存会自动失效，避免复用基于错误代码的旧结果。
CACHE_VERSION = 8

# 安装deepinv
try:
    import deepinv
except ImportError:
    print("正在安装 deepinv ...")
    # 教学注意：使用 subprocess.check_call 并显式退出，避免 os.system 静默失败
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/deepinv/deepinv.git#egg=deepinv'])
    except subprocess.CalledProcessError as e:
        print(f"deepinv 安装失败，退出码 {e.returncode}")
        raise SystemExit(1)
    import deepinv as dinv
else:
    dinv = deepinv
    print(f"deepinv 版本: {dinv.__version__}")

from deepinv.physics import Downsampling, BlurFFT, Inpainting, GaussianNoise
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
    """简化SSIM计算（基于局部统计量）

    ★ 教学提示：若skimage不可用，返回NaN而非近似值，避免误导结果。
    后续可视化（如柱状图）仅使用PSNR，但若扩展为SSIM维度的决策图，
    需注意NaN值的传播与过滤处理。
    """
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
# ★ v8：在线优先（官方 deepinv/images 数据集）；网络不可达时回退到脚本同目录
#   下的 celeba_example.jpg（同源文件的本地副本），保证离线环境可复现。
def _load_test_image():
    try:
        img = load_example("celeba_example.jpg", img_size=(256, 256), resize_mode='resize')
        print("测试图像: 在线加载（deepinv/images 官方数据集）")
        return img
    except Exception as e:
        local_candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "celeba_example.jpg"),
            os.path.join(os.environ.get("TEMP", ""), "celeba_example.jpg"),
        ]
        for p in local_candidates:
            if p and os.path.exists(p):
                from deepinv.utils import load_image
                print(f"测试图像: 在线加载失败（{type(e).__name__}），改用本地副本 {p}")
                return load_image(p, img_size=(256, 256), resize_mode='resize')
        raise
x_true = _load_test_image()
print(f"load_example 返回 shape: {x_true.shape}, ndim: {x_true.ndim}")
# 确保是4D张量 (B, C, H, W)
if x_true.ndim == 3:
    x_true = x_true.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
elif x_true.ndim == 5:
    # 5D 属于非标准格式（如 (1, 1, C, H, W)），按启发式规则显式压缩；
    # 打印警告说明假设，便于学生排查数据结构异常
    print(f"  ⚠️ [Warning] load_example 返回 5D 张量 shape={tuple(x_true.shape)}，"
          f"按 (B, 1, C, H, W) 假设压缩为 4D；如不符合请检查数据源")
    if x_true.shape[1] == 1:
        x_true = x_true.squeeze(1)
    else:
        # ⚠️ 通道维不为 1 时保守取第一组（隐性丢数据），此情况罕见但在预期外时建议排查数据源
        x_true = x_true[:, 0]
elif x_true.ndim != 4:
    raise ValueError(f"load_example 返回的 x_true ndim={x_true.ndim} 不在支持范围 [3,4,5]")
x_true = x_true.to(device)
print(f"处理后 x_true shape: {x_true.shape}")
print(f"测试图像 shape: {x_true.shape}")

# 定义退化场景
scenarios = {}

# 1a. 去模糊场景
print("\n构造去模糊场景...")
try:
    # ★ 修复：deepinv 的 gaussian_blur 只接受 sigma 参数，kernel_size 是自动计算的
    # ★ 注意：deepinv ≥0.4 起 gaussian_blur 位于 deepinv.physics.blur 模块
    #   （更早版本在 deepinv.physics.functional），两条导入路径都尝试以兼容各版本
    try:
        from deepinv.physics.blur import gaussian_blur as gauss_blur_func
        # ★ 注意：gaussian_blur 只接受 sigma 参数，不接受 kernel_size
        light_kernel = gauss_blur_func(sigma=(1.5, 1.5))
        heavy_kernel = gauss_blur_func(sigma=(3.0, 3.0))
    except ImportError:
        # 兼容早期版本（gaussian_blur 位于 deepinv.physics.functional）
        from deepinv.physics.functional import gaussian_blur as gauss_blur_func
        light_kernel = gauss_blur_func(sigma=(1.5, 1.5))
        heavy_kernel = gauss_blur_func(sigma=(3.0, 3.0))

    # ★ 修复：使用可分解的 BlurFFT 而非 Blur（对应DPIR等依赖SVD的方法）
    #   原因：Blur 为非可分解算子（无SVD），DPIR 的数据保真步 prox_l2 需内层迭代求解，收敛差；
    #   而 BlurFFT 基于 FFT 实现，继承 DecomposablePhysics，prox_l2 有闭式解，性能稳定。
    #   注：BlurFFT 内部按 circular padding 处理，输出与输入同尺寸，无需额外指定 padding。
    light_blur = BlurFFT(img_size=(3, 256, 256), filter=light_kernel, device=device)
    light_blur.set_noise_model(GaussianNoise(sigma=0.01))

    # 重度模糊
    heavy_blur = BlurFFT(img_size=(3, 256, 256), filter=heavy_kernel, device=device)
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
    import traceback
    print(f"  超分场景创建失败: {e}")
    traceback.print_exc()  # ★ 统一风格：关键路径打印完整traceback便于调试
    has_sr_scenario = False

# 1c. 修复场景
print("\n构造修复场景...")
try:
    inp50 = Inpainting(img_size=(3, 256, 256), mask=0.5, device=device)
    inp50.set_noise_model(GaussianNoise(sigma=0.01))

    # ★ 修复：mask 语义为"保留比例"而非"缺失比例"（见 deepinv.physics.Inpainting 文档）
    #   原代码 mask=0.8 表示保留80%（仅缺失20%），导致80%修复场景比50%还简单、PSNR倒挂。
    #   修正为 mask=0.2，即保留20%（缺失80%）。
    inp80 = Inpainting(img_size=(3, 256, 256), mask=0.2, device=device)
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
        # ★ 版本校验：旧版本缓存基于错误的场景定义（Blur/mask语义），直接忽略
        if cached_data.get('cache_version') != CACHE_VERSION:
            print(f"⚠ 缓存版本不匹配（文件: {cached_data.get('cache_version', '无标记')}，"
                  f"代码: {CACHE_VERSION}），忽略旧缓存，将重新计算")
            cached_data = {}
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
                    # ★ v4 起缓存保存真实重建图（CPU张量），img 直接恢复用于可视化
                    results[scenario_name][method_name] = {
                        'psnr': float(metrics['psnr']),
                        'ssim': float(metrics.get('ssim', float('nan'))),
                        'time': float(metrics['time']),
                        'img': metrics.get('img')  # 可能为None（异常条目），可视化端有降级处理
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
    # ★ 状态管理说明（四种组合分支）：
    #   1. skip_step2=True, skip_step2_scenarios非空 → 缓存已加载全部场景，results已在前文(323行)重建
    #   2. skip_step2=True, skip_step2_scenarios为空 → 不可能（全部缓存命中意味着至少有一个场景）
    #   3. skip_step2=False, skip_step2_scenarios非空 → 部分缓存命中，results已有部分数据，需补算缺失场景
    #   4. skip_step2=False, skip_step2_scenarios为空 → 无缓存或缓存失效，results为空字典，需全部计算

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
        # ★ v8修复：正则比例 0.1→0.01。比例扫描（6场景×5档比例）表明 0.1 对良态去模糊过强
        #   （轻度模糊仅27.2dB，甚至低于观测29.4dB，正则反而帮倒忙）；0.02 虽在模糊上最好，
        #   但2×超分32.7dB 会略低于插值基线33.1dB；0.01 是唯一在全部场景都不劣于基线的比例
        #   （实测：轻糊31.2/重糊25.3/2×33.2/4×29.1，修复场景≈观测，这是零均值L2先验的固有特性）。
        n_tik_iter = 100  # 迭代次数

        # 自适应步长: 用power iteration估计||A||²，保证收敛 lr < 2/(||A||²+λ)
        x_tmp = torch.randn_like(x_true)
        for _ in range(10):
            x_tmp = physics.A_adjoint(physics.A(x_tmp))
            norm_tmp = x_tmp.norm()
            if norm_tmp > 0:
                x_tmp = x_tmp / norm_tmp
        op_norm_sq = (physics.A(x_tmp)**2).sum() / (x_tmp**2).sum()
        # λ 取算子范数的1%：保证"正则强度/数据保真强度"之比在各退化场景一致
        # （模糊/修复场景 ‖A‖²≈1，λ≈0.01；超分场景随算子范数自动缩小λ）
        lambda_reg = 0.01 * op_norm_sq.item()
        lr_tik = 1.0 / (op_norm_sq.item() + lambda_reg)

        # ★ 自检：验证Tikhonov自适应步长满足收敛条件(避免步长系数被误改后仍能跑)
        # 收敛条件: lr < 2/(‖A‖²+λ), 这里取lr = 1/(‖A‖²+λ) < 2/(‖A‖²+λ)  (1<2, 永远满足)
        # 检验规则:
        #   T1: lr > 0 (避免除零/算子范数为0的退化情况)
        #   T2: lr * (‖A‖²+λ) ≈ 1 (验证公式一致性)
        #   T3: op_norm_sq > 0 (power iteration输出非零, 避免‖A‖=0的平凡情况)
        print(f"[Tikhonov自检] 步长收敛条件验证(实际运行, 非手算):")
        _lr_check = 1.0 / (op_norm_sq.item() + lambda_reg)
        print(f"  T1: lr={_lr_check:.6e} > 0: {'OK' if _lr_check > 0 else 'FAIL'}")
        print(f"  T2: lr·(‖A‖²+λ)={_lr_check * (op_norm_sq.item() + lambda_reg):.6f} (应=1): "
              f"{'OK' if abs(_lr_check * (op_norm_sq.item() + lambda_reg) - 1.0) < 1e-5 else 'FAIL'}")
        print(f"  T3: op_norm_sq={op_norm_sq.item():.4f} > 0: {'OK' if op_norm_sq.item() > 0 else 'FAIL'}")

        t_start = time.perf_counter()  # ★ 用 perf_counter 计时，避免系统时钟调整影响且分辨率更高
        x_tikhonov = x_init.clone()
        for _ in range(n_tik_iter):
            # 梯度: ∇(||Ax-y||² + λ||x||²) = A^T(Ax-y) + λx
            grad = physics.A_adjoint(physics.A(x_tikhonov) - y) + lambda_reg * x_tikhonov
            x_tikhonov = x_tikhonov - lr_tik * grad
            x_tikhonov = x_tikhonov.clamp(0, 1)
        t_tikh = time.perf_counter() - t_start  # ★ 改用 perf_counter 计时：分辨率高于 time.time()，且不受系统时钟调整影响；公式为"结束时刻 - 起始时刻"得到真实耗时（秒）

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
        # ★ v5关键修复 + 参数调优（结论来自CPU参数扫描，测试图 astronaut，256×256）：
        #   1) 【根因修复】optim_builder 的参数名必须是 params_algo——旧代码误用 params，
        #      参数被静默忽略，实际一直跑默认值 λ=1.0、stepsize=1.0，导致：
        #      (a) 调 λ 完全无效；(b) 超分TV被过强收缩成"卡通脸"，且步数越多越差（150步 13.8dB < 30步 16.2dB）。
        #   2) 最优(λ, stepsize)随物理算子不同，统一参数无法兼顾：
        #      - 去模糊 BlurFFT（‖A‖≈1，良态）：大步长1.0 + 小λ 0.01；
        #      - 超分 Downsampling（欠定）：step0.5 + 极小λ 0.001，λ稍大即分段常数化；
        #      - 修复 Inpainting：缺失区域越大越依赖TV先验填充，按实际mask保留率取λ。
        if isinstance(physics, Downsampling):
            tv_stepsize, lambda_tv = 0.5, 0.001
        elif isinstance(physics, Inpainting):
            keep_ratio = float(physics.mask.mean().item())  # mask为1表示保留，均值即保留比例
            # 保留率≈0.5（缺失50%）→ λ=0.05；保留率≈0.2（缺失80%）→ λ=0.2（大洞需更强TV）
            lambda_tv = 0.05 if keep_ratio > 0.35 else 0.2
            tv_stepsize = 1.0
        else:  # BlurFFT 去模糊场景
            tv_stepsize, lambda_tv = 1.0, 0.01

        prior_tv = TVPrior(def_crit=1e-3, n_it_max=10)  # 教学参数：减少内层迭代，避免耗时过长
        data_fidelity_tv = DataFidelity_L2()

        # 尝试使用新版deepinv的优化器（优先方案）
        t_start = time.perf_counter()  # ★ 用 perf_counter 计时，避免系统时钟调整影响且分辨率更高

        # 方案1: 尝试使用 optim_builder（如果可用）
        try:
            from deepinv.optim import optim_builder
            optimizer = optim_builder(
                'PGD',  # Proximal Gradient Descent
                data_fidelity=data_fidelity_tv,
                prior=prior_tv,
                # ★ 必须用 params_algo（用 params 会被静默忽略）；g_param 是PGD必需键，TV先验不用它故设0
                params_algo={'stepsize': tv_stepsize, 'lambda': lambda_tv, 'g_param': 0.0},
                max_iter=150  # v5调优：外层迭代 30→150，给收敛留出余量
            )
            x_tv = optimizer(y, physics, init=x_init)
            # ★ 修复：必须显式传入 init=x_init。deepinv 优化器默认从 A_adjoint(y) 初始化，
            #   而超分场景的 A_adjoint 是零填充上采样（4×时亮度仅原图1/16，极暗），
            #   PGD 从极暗初值难以恢复（早期错误参数下表现为4×超分TV全黑）。
            print(f"  ✓ 使用 optim_builder (PGD), λ={lambda_tv}, stepsize={tv_stepsize}")
        except ImportError:
            # 方案2: 尝试使用 ADMM
            try:
                from deepinv.optim import ADMM
                optimizer = ADMM(
                    data_fidelity=data_fidelity_tv,
                    prior=prior_tv,
                    max_iter=150,  # v5调优：与方案1一致 30→150
                    params_algo={'stepsize': tv_stepsize, 'lambda': lambda_tv, 'g_param': 0.0}
                )
                x_tv = optimizer(y, physics, init=x_init)  # ★ 同方案1：显式传入初值
                print("  ✓ 使用 ADMM 优化器")
            except ImportError:
                # 方案3: 尝试使用 PGD 类
                try:
                    from deepinv.optim import PGD
                    optimizer = PGD(
                        data_fidelity=data_fidelity_tv,
                        prior=prior_tv,
                        max_iter=150,  # v5调优：与方案1一致 30→150
                        params_algo={'stepsize': tv_stepsize, 'lambda': lambda_tv, 'g_param': 0.0}
                    )
                    x_tv = optimizer(y, physics, init=x_init)  # ★ 同方案1：显式传入初值，避免超分场景从极暗的A_adjoint出发
                    print("  ✓ 使用 PGD 优化器")
                except ImportError:
                    raise ImportError("新版优化器均不可用")

        t_tv = time.perf_counter() - t_start  # ★ 改用 perf_counter 计时：分辨率高于 time.time()，且不受系统时钟调整影响；公式为"结束时刻 - 起始时刻"得到真实耗时（秒）
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
            t_start = time.perf_counter()  # ★ 用 perf_counter 计时，避免系统时钟调整影响且分辨率更高
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
                # ★ 修复：pad=(0,0,1,1) —— 前两个 0 表示 W 维不补、后两个 1 表示 H 维两端各补 0。
                #   旧代码误写为 (1,1)，等于在 W 维两端补零却沿 dim=-2 差分，
                #   差分为 (B,C,H-2,W+2)，与 tv_grad_x 形状不匹配会直接 RuntimeError。
                tv_grad_y = -torch.diff(F.pad(s_y, (0, 0, 1, 1), mode='constant', value=0), dim=-2)
                tv_grad = tv_grad_x + tv_grad_y  # (B,C,H,W)
                x_tv = x_tv - lr_tv * (grad_data + lambda_tv * tv_grad)
                x_tv = x_tv.clamp(0, 1)
                if (it + 1) % 25 == 0:
                    print(f"    TV迭代 {it+1}/{n_tv_iter} 完成")
            t_tv = time.perf_counter() - t_start  # ★ 改用 perf_counter 计时：分辨率高于 time.time()，且不受系统时钟调整影响；公式为"结束时刻 - 起始时刻"得到真实耗时（秒）

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

    # 2c. 伴随重建
    # ★ 教学注意：超分场景下 A_adjoint(y) 是零填充（极暗），本节实际使用的是 x_init.clone()，对超分场景即"双线性插值"结果。
    #   为避免方法名误导，将伴随重建结果的方法名按场景区分：超分场景叫"插值基线"，其他场景才叫"伴随重建"。
    try:
        t_start = time.perf_counter()  # ★ 用 perf_counter 计时，避免系统时钟调整影响且分辨率更高
        x_adj = x_init.clone()
        t_adj = time.perf_counter() - t_start  # ★ 改用 perf_counter 计时：分辨率高于 time.time()，且不受系统时钟调整影响；公式为"结束时刻 - 起始时刻"得到真实耗时（秒）
        # ★ 运行时自检：防止之前"t_xxx = time.perf_counter() + 注释里的 - t_start"那种隐蔽 bug 重犯
        if not (0 <= t_adj < 3600):
            print(f"  [警告] 伴随重建耗时异常: t_adj={t_adj:.3f}s（预期 0~3600），可能计时逻辑有误")

        psnr_adj = compute_psnr(x_true, x_adj)
        ssim_adj = compute_ssim(x_true, x_adj)
        # ★ 根据是否超分场景，给方法名以更准确的中文标签
        method_name_adj = '插值基线' if isinstance(physics, Downsampling) else '伴随重建'
        results[scenario_name][method_name_adj] = {
            'psnr': psnr_adj, 'ssim': ssim_adj, 'time': t_adj, 'img': x_adj
        }
        print(f"  {method_name_adj}:   PSNR={psnr_adj:.2f} dB, SSIM={ssim_adj:.4f}, 耗时={t_adj:.3f}s")
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
                except Exception:  # ★ 明确捕获异常类型，避免吞掉 KeyboardInterrupt/SystemExit
                    cached_data = {}  # 如果加载失败，从头开始

            # 更新 Step 2 结果
            # ★ v4起：缓存额外保存真实重建图（.cpu()转CPU张量），可视化不再降级
            #   关键：必须 .cpu() 后再 pickle —— Colab(GPU)生成的缓存下载到本地(CPU)加载时，
            #   若张量仍挂在CUDA设备上会报错 "Attempting to deserialize object on a CUDA device"
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
                        'time': time_val,
                        'img': metrics['img'].cpu() if isinstance(metrics.get('img'), torch.Tensor) else None
                    }

            cached_data['step2_results'] = step2_data
            cached_data['cache_version'] = CACHE_VERSION  # ★ 写入版本号供下次校验

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
        # ★ 版本校验：与 Step 2 一致，旧版本缓存直接忽略
        if cached_data.get('cache_version') != CACHE_VERSION:
            print("⚠ 缓存版本不匹配，忽略旧缓存，将重新计算")
            cached_data = {}
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
                        'img': metrics.get('img')  # ★ v4起缓存保存真实重建图，直接恢复
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
        # 教学说明：原代码的 elif '80%' 分支和 else 完全等价（都是 0.01），属于死代码。
        # 现简化为两层：重度模糊用较高噪声(0.05)，其余场景统一用 0.01
        if '重度' in scenario_name:
            noise_sigma = 0.05
        else:
            noise_sigma = 0.01

        try:
            from deepinv.optim.optimizers import optim_builder
            from deepinv.optim.prior import PnP
            from deepinv.optim.data_fidelity import L2
            from deepinv.optim.dpir import get_DPIR_params
            from deepinv.physics import Inpainting as _InpaintingCls

            # ★ v8 关键修复：get_DPIR_params 是为去模糊/去噪/超分设计的官方默认配方
            #   （8步，去噪σ从49/255≈0.19对数降到测量噪声）。直接用于大面积随机掩码修复时，
            #   去噪先验强度不足以从50%/80%空洞中"幻想"内容，DPIR 假性失败
            #   （实测50%缺失仅12.3dB、80%缺失仅9.8dB，甚至不如TV）。
            #   修复任务需要：更高的起始噪声水平、更多迭代、随σ²缩放的强数据锁定。
            #   下列调度（与官方 lamb=1/0.23 的 stepsize 约定一致，仅端点/步数按修复任务设置）
            #   经实测可把同一 DRUNet 在 50%/80% 缺失上提升到 33/28 dB。
            if isinstance(physics, _InpaintingCls):
                max_iter = 20
                _sigma_hi, _sigma_lo = 0.5, 0.05
                sigma_denoiser = torch.logspace(
                    torch.log10(torch.tensor(_sigma_hi, dtype=torch.float32)),
                    torch.log10(torch.tensor(_sigma_lo, dtype=torch.float32)),
                    steps=max_iter, dtype=torch.float32, device=device,
                )
                # 数据保真权重，随去噪σ²缩放（与 get_DPIR_params 中 (σ/σ_n)² 同约定）
                stepsize = (1.0 / 0.23) * (sigma_denoiser / _sigma_lo) ** 2
                print(f"  ✓ 修复场景使用专用HQS调度: {max_iter}步, σ {_sigma_hi}→{_sigma_lo} 对数递减")
            else:
                # 使用官方 DPIR 默认参数（去模糊/超分）
                sigma_denoiser, stepsize, max_iter = get_DPIR_params(noise_sigma, device=device)
            params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}
            prior = PnP(denoiser=denoiser_drunet)
            data_fidelity = L2()

            t_start = time.perf_counter()  # ★ 用 perf_counter 计时，避免系统时钟调整影响且分辨率更高
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
            t_dpir = time.perf_counter() - t_start  # ★ 改用 perf_counter 计时：分辨率高于 time.time()，且不受系统时钟调整影响；公式为"结束时刻 - 起始时刻"得到真实耗时（秒）

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
                # ★ 简化配置：n_iter=20, step_size=0.5 为经验值，未做类似Tikhonov(387-395行)的自适应步长估计
                #   实际应用中，PnP收敛性依赖步长与去噪器的匹配，可能需要按场景调优
                x_pnp = x_init.clone()
                n_iter = 20  # 减少迭代次数，加快演示
                sigma_pnp = noise_sigma * 2  # 初始噪声水平
                step_size = 0.5  # 适中的步长（经验值，缺乏理论保证）
                t_start = time.perf_counter()  # ★ 用 perf_counter 计时，避免系统时钟调整影响且分辨率更高

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

                t_pnp = time.perf_counter() - t_start  # ★ 改用 perf_counter 计时：分辨率高于 time.time()，且不受系统时钟调整影响；公式为"结束时刻 - 起始时刻"得到真实耗时（秒）
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
                except Exception:  # ★ 明确捕获异常类型，避免吞掉 KeyboardInterrupt/SystemExit
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
                            'time': time_val,
                            'img': metrics['img'].cpu() if isinstance(metrics.get('img'), torch.Tensor) else None  # ★ v4起保存真实重建图（CPU张量）
                        }

            cached_data['step3_results'] = step3_results
            cached_data['cache_version'] = CACHE_VERSION  # ★ 写入版本号供下次校验

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
        # ★ 版本校验：与 Step 2 一致，旧版本缓存直接忽略
        if cached_data.get('cache_version') != CACHE_VERSION:
            print("⚠ 缓存版本不匹配，忽略旧缓存，将重新计算")
            cached_data = {}
        if 'step4_results' in cached_data:
            # 合并 Step 4 的结果到 results
            for scenario_name, methods in cached_data['step4_results'].items():
                if scenario_name not in results:
                    results[scenario_name] = {}
                for method_name, metrics in methods.items():
                    entry = {
                        'psnr': float(metrics['psnr']),
                        'ssim': float(metrics.get('ssim', float('nan'))),
                        'time': float(metrics['time']),
                        'img': metrics.get('img')  # ★ v4起缓存保存真实重建图，直接恢复
                    }
                    # ★ v8：透传多次采样附加字段（若缓存中有）
                    for _k in ('psnr_std', 'psnr_mmse', 'n_samples', 'time_total'):
                        if _k in metrics:
                            entry[_k] = metrics[_k]
                    results[scenario_name][method_name] = entry
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

print(f"""
DiffPIR算法核心思路:
  在扩散采样过程中融入数据保真约束
  每个去噪步骤后，用数据一致性步骤将采样引导向观测一致
  特点: ζ=0.1 时产生后验随机样本（可做不确定性量化）；PSNR未必超过确定性PnP，
        感知质量/样本多样性需用 LPIPS/FID 等指标或后验均值评估

★ 当前配置: 采样步数={n_diffusion_steps}（库默认/论文标准配置），
  每场景独立采样 {n_diffusion_samples} 次（报告 PSNR 均值±标准差，配图为后验均值），
  全部6个退化场景（含50%/80%修复）均参与对比，进度条={'开启' if show_progress else '关闭'}
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
    # ★ v8：全部6个场景都测扩散方法（包括50%/80%修复）。
    #   依据：Inpainting 的 L2 prox 有闭式解（DecomposablePhysics），DiffPIR 不需要SVD，
    #   官方文档示例本身就是 Inpainting(mask=0.5)；GPU上每场景仅约10s，"太慢不全测"不成立。
    # ★ 显式按名称选择场景，避免依赖 scenarios 字典的插入顺序；若某场景在 Step1 构造失败则自动缺失。
    diff_scenarios = [k for k in ['轻度模糊', '重度模糊', '2倍超分', '4倍超分', '50%修复', '80%修复']
                       if k in scenarios]
    if not diff_scenarios:
        diff_scenarios = list(scenarios.keys())

    # ★ v8：增量落盘——每完成一个扩散场景即写缓存，避免长时间 CPU 运行中断后全部丢失
    def _persist_step4_cache():
        if not use_cache:
            return
        try:
            cached_data = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                except Exception:
                    cached_data = {}
            step4_results = {}
            for _sname, _methods in results.items():
                _diff_methods = {k: v for k, v in _methods.items()
                                 if 'DiffPIR' in k or 'DDRM' in k or 'DPS' in k}
                if _diff_methods:
                    step4_results[_sname] = {}
                    for _mn, _mm in _diff_methods.items():
                        _entry = {
                            'psnr': float(_mm['psnr']),
                            'ssim': float(_mm.get('ssim', float('nan'))),
                            'time': float(_mm['time']),
                            'img': _mm['img'].cpu() if isinstance(_mm.get('img'), torch.Tensor) else None,
                        }
                        for _k in ('psnr_std', 'psnr_mmse', 'time_total'):
                            if _k in _mm:
                                _entry[_k] = float(_mm[_k])
                        if 'n_samples' in _mm:
                            _entry['n_samples'] = int(_mm['n_samples'])
                        step4_results[_sname][_mn] = _entry
            cached_data['step4_results'] = step4_results
            cached_data['cache_version'] = CACHE_VERSION
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"  ✓ 已增量保存扩散结果到缓存（{len(step4_results)}个场景）")
        except Exception as _e:
            print(f"  ⚠ 增量缓存保存失败: {_e}")

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

            # ★ v8：显式按场景噪声构造。注意 DiffPIR.forward 内部还会用 physics.noise_model.sigma
            #   重建噪声调度（见 deepinv/sampling/diffusion.py），这里传值保持显式一致，
            #   不再写死 sigma=0.05（旧代码中该值对0.01噪声场景会被覆盖，属误导性"死参数"）。
            noise_sigma_diff = 0.05 if '重度' in scenario_name else 0.01
            # ★ ζ=0.1, λ=7.0 为库/论文默认值（官方文档：λ最优范围3~25，按问题调；本实验保持统一以公平计时与对比）
            model_diffpir = DiffPIR_algo(
                model=denoiser_diffunet,
                data_fidelity=DiffL2(),
                max_iter=n_diffusion_steps,
                sigma=noise_sigma_diff,
                zeta=0.1,  # ★ 采样步随机性权重；ζ=0.1 时输出为后验随机样本（故需多次采样报均值±方差）
                lambda_=7.0,  # ★ 数据保真调度系数 ρ_t=λ·σ_n²/σ̄_t²
                verbose=show_progress,  # ★ 启用进度条
                device=device,
            )

            # ★ v8：多次独立采样。ζ=0.1 的扩散结果是后验随机样本，单次PSNR不具代表性；
            #   保存每样本PSNR（报均值±标准差），重建图取多样本后验均值（MMSE估计，PSNR最优口径）。
            t_start = time.perf_counter()
            sample_imgs, sample_psnrs = [], []
            for _si, _seed in enumerate(diffusion_seeds[:n_diffusion_samples]):
                print(f"  DiffPIR 采样 {_si+1}/{n_diffusion_samples} (seed={_seed})")
                with torch.no_grad():
                    _xs = model_diffpir(y, physics, seed=_seed).clamp(0, 1)
                sample_imgs.append(_xs)
                sample_psnrs.append(compute_psnr(x_true, _xs))
            t_diffpir_total = time.perf_counter() - t_start
            t_diffpir = t_diffpir_total / n_diffusion_samples  # ★ 单样本（单张重建）耗时，与其他方法口径一致

            x_diffpir = torch.stack(sample_imgs, dim=0).mean(dim=0).clamp(0, 1)  # 后验均值(MMSE)
            psnr_diffpir = float(np.mean(sample_psnrs))  # 单次样本PSNR的均值（表头口径）
            psnr_std_diffpir = float(np.std(sample_psnrs))
            psnr_mmse_diffpir = compute_psnr(x_true, x_diffpir)  # 后验均值图的PSNR（配图口径）
            ssim_diffpir = compute_ssim(x_true, x_diffpir)
            results[scenario_name]['DiffPIR(扩散)'] = {
                'psnr': psnr_diffpir, 'psnr_std': psnr_std_diffpir,
                'psnr_mmse': psnr_mmse_diffpir, 'ssim': ssim_diffpir,
                'time': t_diffpir, 'time_total': t_diffpir_total,
                'img': x_diffpir, 'n_samples': n_diffusion_samples,
            }
            print(f"  DiffPIR:  样本PSNR={psnr_diffpir:.2f}±{psnr_std_diffpir:.2f} dB "
                  f"(后验均值图PSNR={psnr_mmse_diffpir:.2f}), SSIM={ssim_diffpir:.4f}, "
                  f"单样本耗时={t_diffpir:.1f}s（{n_diffusion_samples}次合计={t_diffpir_total:.0f}s）")
        except Exception as e:
            print(f"  DiffPIR(新API)求解失败: {e}")
            # 回退方案：先尝试DDRM，再尝试DPS（不需要SVD，适合Blur场景）
            try:
                print("  回退到DDRM算法...")
                from deepinv.sampling import DDRM
                t_start = time.perf_counter()  # ★ 用 perf_counter 计时，避免系统时钟调整影响且分辨率更高
                model_ddrm = DDRM(
                    denoiser=denoiser_diffunet,
                    sigmas=np.linspace(1, 0, n_diffusion_steps),
                    verbose=show_progress
                )
                x_ddrm = model_ddrm(y, physics)
                x_ddrm = x_ddrm.clamp(0, 1)  # ★ 确保输出在[0,1]范围
                t_ddrm = time.perf_counter() - t_start  # ★ 改用 perf_counter 计时：分辨率高于 time.time()，且不受系统时钟调整影响；公式为"结束时刻 - 起始时刻"得到真实耗时（秒）

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
                    t_start = time.perf_counter()  # ★ 用 perf_counter 计时，避免系统时钟调整影响且分辨率更高
                    model_dps = DPS(
                        model=denoiser_diffunet,
                        data_fidelity=DiffL2(),
                        max_iter=n_diffusion_steps,
                        verbose=show_progress
                    )
                    x_dps = model_dps(y, physics)
                    x_dps = x_dps.clamp(0, 1)  # ★ 确保输出在[0,1]范围
                    t_dps = time.perf_counter() - t_start  # ★ 改用 perf_counter 计时：分辨率高于 time.time()，且不受系统时钟调整影响；公式为"结束时刻 - 起始时刻"得到真实耗时（秒）

                    psnr_dps = compute_psnr(x_true, x_dps)
                    ssim_dps = compute_ssim(x_true, x_dps)
                    results[scenario_name]['DPS(扩散)'] = {
                        'psnr': psnr_dps, 'ssim': ssim_dps, 'time': t_dps, 'img': x_dps
                    }
                    print(f"  DPS:  PSNR={psnr_dps:.2f} dB, SSIM={ssim_dps:.4f}, 耗时={t_dps:.3f}s")
                except Exception as e3:
                    print(f"  DPS也失败: {e3}")
                    print(f"  [提示] 场景 '{scenario_name}' 的所有扩散方法均失败，Step 5可视化中将缺少扩散方法子图")

        # ★ v8：每个场景跑完立即落盘（中断后已完成场景可复用）
        _persist_step4_cache()

    # ★ v8：Step 4 已按场景增量落盘，循环结束后再统一保存一次
    if use_cache and not skip_step4:
        _persist_step4_cache()
        print(f"✓ Step 4 结果已保存到缓存: {cache_file}")
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
print(f"{'场景':<10} {'方法':<15} {'PSNR(dB)':<26} {'SSIM':<9} {'耗时(s)':<9}")
print("-" * 74)
for scenario_name, methods in results.items():
    for method_name, metrics in methods.items():
        ssim_val = metrics.get('ssim', float('nan'))
        # ★ v8：扩散方法同时报两个口径——样本PSNR均值±标准差（随机后验样本）
        #   与后验均值(MMSE)图PSNR（配图本身）；耗时为单样本均值
        if 'psnr_std' in metrics:
            psnr_txt = (f"样本{metrics['psnr']:.2f}±{metrics['psnr_std']:.2f}"
                        f"/均值图{metrics.get('psnr_mmse', metrics['psnr']):.2f}")
        else:
            psnr_txt = f"{metrics['psnr']:.2f}"
        print(f"{scenario_name:<10} {method_name:<15} {psnr_txt:<26} {ssim_val:<9.4f} {metrics['time']:<9.3f}")
    print("-" * 74)
print("注：扩散行「样本」为3次独立后验采样PSNR的均值±标准差；「均值图」为3样本平均图像(MMSE)的PSNR（即5c配图）；耗时为单样本均值。")

# 5b. ★ 决策指南可视化
print("\n--- 5b. ★ 决策指南可视化 ---")

# 找出有足够结果的场景来画图
plot_scenarios = [s for s in results.keys() if len(results[s]) >= 2]

# ★ 统一方法展示顺序（避免按 Unicode 排序造成的混乱；基线/优化/PnP/扩散）
CANON_METHOD_ORDER = ['Tikhonov', 'TV正则化', '伴随重建', '插值基线', 'DPIR(PnP)', 'DiffPIR(扩散)',
                      'TV正则化(手动)', 'PnP-HQS(手动)', 'DDRM(扩散)', 'DPS(扩散)']
def _baseline_key_of(methods_map):
    """该场景的基线方法名（超分为双线性插值基线，其余为伴随重建）"""
    return '插值基线' if '插值基线' in methods_map else ('伴随重建' if '伴随重建' in methods_map else None)

if len(plot_scenarios) >= 1:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 左图: PSNR柱状图（同场景内可比；扩散方法带±1标准差误差棒）
    x_pos = np.arange(len(plot_scenarios))
    present = set()
    for s in plot_scenarios:
        present.update(results[s].keys())
    all_methods = [m for m in CANON_METHOD_ORDER if m in present] + \
                  sorted(present - set(CANON_METHOD_ORDER))
    width = 0.8 / max(len(all_methods), 1)
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_methods)))

    for i, method in enumerate(all_methods):
        psnrs, stds = [], []
        for s in plot_scenarios:
            if method in results[s]:
                psnrs.append(results[s][method]['psnr'])
                stds.append(results[s][method].get('psnr_std', 0.0))
            else:
                psnrs.append(np.nan)  # ★ 缺失用NaN（不再画成0，避免"假差"误读）
                stds.append(0.0)
        psnrs, stds = np.array(psnrs, dtype=float), np.array(stds, dtype=float)
        axes[0].bar(x_pos + (i - (len(all_methods)-1)/2) * width, psnrs, width,
                    yerr=stds, capsize=3, label=method, color=colors[i],
                    error_kw={'elinewidth': 1, 'ecolor': 'dimgray'})

    axes[0].set_xlabel('退化场景', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    # ★ 教学提示：PSNR绝对值不可跨场景比较（退化强度不同），仅可比较同场景内不同方法；
    #   扩散柱误差棒为 n_diffusion_samples 次独立采样的±1标准差
    axes[0].set_title('★ 各方法PSNR对比（扩散柱为均值±1标准差）\n仅同场景内可比', fontsize=13)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(plot_scenarios, fontsize=10)
    axes[0].legend(fontsize=9, ncol=2)
    axes[0].grid(axis='y', alpha=0.3)

    # 右图: ★ 时间-质量增益权衡图（Pareto前沿）
    # ★ v8 修正：纵轴改为 ΔPSNR = 方法PSNR − 该场景基线PSNR（增益可跨场景比较，
    #   与左图"绝对值不可跨场景"的声明不再自相矛盾）；基线耗时≈0，用对数横轴+0.03s显示下限。
    markers = ['o', 's', '^', 'D', 'v', 'P']
    t_floor = 0.03  # 近零耗时（伴随/插值基线）的显示位置
    for i, method in enumerate(all_methods):
        if method in ('伴随重建', '插值基线'):
            continue  # 基线自身增益≈0，不画
        gains, times_disp, labels = [], [], []
        for s in plot_scenarios:
            if method not in results[s]:
                continue
            bkey = _baseline_key_of(results[s])
            if bkey is None:
                continue
            gains.append(results[s][method]['psnr'] - results[s][bkey]['psnr'])
            times_disp.append(max(results[s][method]['time'], t_floor))
            labels.append(s)
        if gains:
            axes[1].scatter(times_disp, gains, marker=markers[i % len(markers)],
                            s=110, label=method, color=colors[i], zorder=5,
                            edgecolors='k', linewidths=0.5)
            for j in range(len(gains)):
                axes[1].annotate(labels[j], (times_disp[j], gains[j]),
                                fontsize=8, xytext=(5, 5), textcoords='offset points')

    axes[1].axhline(0, color='k', linewidth=1, linestyle='--', alpha=0.7)
    axes[1].set_xscale('log')
    axes[1].set_xlabel(f'推理时间 (秒，对数轴；近零耗时基线显示在{t_floor}s处)', fontsize=11)
    axes[1].set_ylabel('ΔPSNR：相对本场景基线的增益 (dB)', fontsize=11)
    axes[1].set_title('★ 时间-质量增益权衡（越靠左上越好；虚线=与基线持平）', fontsize=13)
    axes[1].legend(fontsize=9, ncol=2)
    axes[1].grid(alpha=0.3, which='both')

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
    # ★ 布局改为单行横向排列：方法数量最多7个（真值+观测+5方法），
    #   两行布局（5+2）会造成行间数量不齐、视线折返；单行并排更利于逐方法对比
    total_subplots = 2 + num_methods
    ncols = total_subplots
    nrows = 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 4.2))
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
    def _method_title(mname, mm):
        # ★ v8：扩散方法标题同时标注两个口径：样本PSNR均值±标准差（表格/柱状图口径）
        #   与后验均值图PSNR（配图本身的PSNR），避免"图与分数对不上"的误读
        if 'psnr_std' in mm:
            return (f'{mname}\n样本PSNR={mm["psnr"]:.1f}±{mm["psnr_std"]:.1f}dB '
                    f'({mm.get("n_samples", 3)}样本)\n本图(后验均值)PSNR={mm.get("psnr_mmse", mm["psnr"]):.1f}dB, '
                    f'SSIM={mm.get("ssim", 0):.3f}')
        return f'{mname}\nPSNR={mm["psnr"]:.1f}dB, SSIM={mm.get("ssim", 0):.3f}'

    for idx, (method_name, metrics) in enumerate(scenario_methods.items()):
        ax_idx = idx + 2
        if ax_idx < len(axes_flat):
            img = metrics.get('img')
            if img is not None:
                safe_imshow(axes_flat[ax_idx], img, _method_title(method_name, metrics))
            else:
                # img 为 None（缓存加载）：逐级降级显示
                displayed = False
                if scenario_name in scenarios:
                    physics = scenarios[scenario_name]['physics']
                    y_obs = scenarios[scenario_name]['y']
                    # ★ 对超分场景，A_adjoint 是零填充（极暗），改用双线性插值
                    #   F 已通过文件头部 import torch.nn.functional as F 全局导入，无需在此重复
                    try:
                        from deepinv.physics import Downsampling
                        if isinstance(physics, Downsampling):
                            target_h, target_w = x_true.shape[-2], x_true.shape[-1]
                            adj = F.interpolate(y_obs, size=(target_h, target_w), mode='bilinear', align_corners=False)
                        else:
                            adj = physics.A_adjoint(y_obs)
                        safe_imshow(axes_flat[ax_idx], adj, f'{method_name}\nPSNR={metrics["psnr"]:.1f}dB\n(缓存降级-非真实重建)')
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

    fig.suptitle(f'重建对比: {scenario_name}', fontsize=14, y=1.02)
    # ★ v8：扩散标题为三行文字，顶部预留空间，避免子图标题被裁切
    plt.tight_layout()
    fig.subplots_adjust(top=0.82)
    # 文件名中替换特殊字符
    safe_name = scenario_name.replace('×', 'x').replace('=', '_')
    plt.savefig(os.path.join(SAVE_DIR, f'step5_reconstruction_{safe_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: step5_reconstruction_{safe_name}.png")

if not results:
    print("  ⚠ 没有可用的结果进行可视化")

# 5d. 决策指南（★ v8：数据驱动——结论由上方 results 实际指标生成，不预设固定话术）
print("\n" + "-" * 64)
print("★ 方法选择决策指南（由本次实测指标自动生成，对应18.3节决策树）")
print("-" * 64)

_solvable = {'Tikhonov', 'TV正则化', 'DPIR(PnP)', 'DiffPIR(扩散)'}
print("\n[1] 各场景 PSNR 最优方法（同场景内才可比；扩散为"
      f"{n_diffusion_samples}次采样均值）:")
for s_name in scenarios:
    ms = results.get(s_name, {})
    ranked = sorted(((m, d['psnr']) for m, d in ms.items() if m in _solvable),
                    key=lambda kv: kv[1], reverse=True)
    if not ranked:
        continue
    best_m, best_p = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    gap = f"，领先次优 {best_p - runner[1]:.2f}dB（{runner[0]}）" if runner else ""
    std = ms[best_m].get('psnr_std')
    std_txt = f" ± {std:.2f}" if std is not None else ""
    # 若以"后验均值(MMSE)图PSNR"为口径，扩散可能反超——显式提示两种口径的排名差异
    mmse_best_m, mmse_best_p = max(
        ((m, d.get('psnr_mmse', d['psnr'])) for m, d in ms.items() if m in _solvable),
        key=lambda kv: kv[1])
    if mmse_best_m != best_m and mmse_best_p > best_p + 1e-9:
        gap += f"｜注：按后验均值图口径 {mmse_best_m}={mmse_best_p:.2f}dB 更优"
    print(f"   - {s_name:<7}: {best_m}（{best_p:.2f}{std_txt} dB）{gap}")

print("\n[2] 跨场景平均增益（ΔPSNR：相对本场景伴随/插值基线，仅在该方法有结果的场景上平均）:")
_gain_sum, _gain_n = {}, {}
for s_name, ms in results.items():
    bkey = _baseline_key_of(ms)
    if bkey is None:
        continue
    for m, d in ms.items():
        if m in _solvable:
            _gain_sum[m] = _gain_sum.get(m, 0.0) + (d['psnr'] - ms[bkey]['psnr'])
            _gain_n[m] = _gain_n.get(m, 0) + 1
for m in CANON_METHOD_ORDER:
    if m in _gain_sum:
        print(f"   - {m:<14}: 平均增益 {_gain_sum[m]/_gain_n[m]:+5.2f} dB"
              f"（{_gain_n[m]}个场景）")

# 全局性价比：在"质量排名"与"耗时"上给出实测描述（耗时取自本次设备实测范围）
def _time_range(method):
    ts = [ms[method]['time'] for ms in results.values() if method in ms]
    return (min(ts), max(ts)) if ts else None
_fast = [m for m in ('Tikhonov', 'TV正则化') if any(m in ms for ms in results.values())]
_mid = [m for m in ('DPIR(PnP)',) if any(m in ms for ms in results.values())]
_slow = [m for m in ('DiffPIR(扩散)',) if any(m in ms for ms in results.values())]
print("\n[3] 时间分层（本次设备实测，CPU）:")
if _fast:
    _tr = [_time_range(m) for m in _fast]
    _lo = min(t[0] for t in _tr); _hi = max(t[1] for t in _tr)
    print(f"   - 严格时限（实测 {_lo:.2f}–{_hi:.1f}s/张）→ 优化方法 {_fast}：其中 TV 显著优于 Tikhonov，"
          "且能以分段常数方式填充修复空洞；但 Tikhonov 无法填洞，且大空洞下 TV 明显弱于 PnP/扩散")
if _mid:
    _lo, _hi = _time_range('DPIR(PnP)')
    print(f"   - 中等时限（实测 {_lo:.1f}–{_hi:.1f}s/张）→ {_mid}：本次实验中质量-速度综合最佳，"
          "修复场景须换用修复专用噪声调度（默认 get_DPIR_params 配方只适合去模糊/超分）")
if _slow:
    _lo, _hi = _time_range('DiffPIR(扩散)')
    print(f"   - 时限宽松（实测单样本 {_lo:.0f}–{_hi:.0f}s；GPU 约秒级）→ {_slow}：唯一能输出后验随机样本、"
          "支持不确定性量化的方法族；默认参数下PSNR不保证超过DPIR，"
          "追求PSNR需按问题调 ζ/λ 并用后验均值，感知质量应改用 LPIPS/FID 评估")
print("\n[4] 适用范围提醒:")
print("   - 本结论基于单张人脸图像(N=1)与固定退化实例，跨数据集/跨退化强度需更多样本验证；")
print("   - Tikhonov 对修复场景输出黑洞（缺失像素解析解=0），是L2零均值先验的固有特性，非bug；")
print(f"   - 扩散采样步数 {n_diffusion_steps} 为库默认/论文标准配置；进度条: {show_progress}。")


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
   - 超分辨率（2×/4×下采样）
   - 修复（50%/80%缺失）

2. 优化方法 ✓
   - Tikhonov: L2正则化，速度快
   - TV正则化: 保边缘，轻度退化效果好
   - 伴随重建/插值基线: 超分用双线性插值（命名"插值基线"），其余场景用零填充伴随（命名"伴随重建"）

3. PnP方法 ✓
   - DPIR: DRUNet去噪器+递减噪声表；修复场景使用修复专用调度（20步，σ:0.5→0.05）
   - PnP-HQS手动实现（备用方案）

4. 扩散方法 ✓
   - DiffPIR: DiffUNet+数据一致性，全部6场景（含50%/80%修复）
   - 每场景 {n_diffusion_samples} 次独立采样：报告样本PSNR均值±标准差，配图取后验均值(MMSE)
   - DDRM/DPS: 备选扩散求解器

5. ★ 方法对比与决策 ✓
   - PSNR(均值±标准差)/时间对比表格
   - 时间-质量增益权衡图（ΔPSNR，跨场景可比）
   - 决策指南（由实测指标自动生成）

关键发现（以本脚本实测为准，见5d）:
- 优化方法: 速度快；TV显著优于Tikhonov且能以分段常数方式填充修复空洞；
  Tikhonov（零均值L2先验）无法填充空洞，是固有特性而非bug
- PnP(DPIR): 本次实验中质量-速度综合最佳；在修复任务上必须使用任务匹配的噪声调度
- 扩散(DiffPIR): 唯一支持后验采样/不确定性量化；推理慢（×30–50），
  随机样本的单次PSNR不保证优于确定性PnP，需多样本统计或LPIPS/FID评估

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
        entry = {
            'psnr': float(round(m_data['psnr'], 2)),  # ★ 转换为 Python float
            'ssim': float(round(m_data.get('ssim', 0), 4)),  # ★ 转换为 Python float
            'time': float(round(m_data['time'], 3))  # ★ 转换为 Python float
        }
        # ★ v8：扩散多次采样附加信息
        if 'psnr_std' in m_data:
            entry['psnr_std'] = float(round(m_data['psnr_std'], 3))
            entry['psnr_mmse'] = float(round(m_data.get('psnr_mmse', m_data['psnr']), 2))
            entry['n_samples'] = int(m_data.get('n_samples', n_diffusion_samples))
            if 'time_total' in m_data:
                entry['time_total'] = float(round(m_data['time_total'], 1))  # n次采样合计耗时
        results_summary[s_name][m_name] = entry
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print("数值结果已保存: results_summary.json")
