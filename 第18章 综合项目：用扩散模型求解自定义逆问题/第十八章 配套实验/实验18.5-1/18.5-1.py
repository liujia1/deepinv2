# -*- coding: utf-8 -*-
"""
实验18.5-1 不确定性量化与后验采样
对应知识点：18.5节（不确定性量化：从点估计到分布推断）

实验内容：
Step 1: 从点估计到分布 —— 单次重建 vs 多次后验采样
Step 2: 后验采样实现 —— PnP-ULA与扩散采样
Step 3: 不确定性地图计算 —— 像素级标准差、经验分位数区间
Step 4: ★ 校准检验 —— 覆盖率与可靠性分析
Step 5: ★ 样本数对不确定性估计的影响

★原创设计：
- Step 4的校准检验：验证经验分位数区间覆盖率（⚠️ 样本数较少时结果仅供参考）
- Step 5的样本数对比：量化样本数对不确定性估计的影响
- 不确定性地图按问题类型(去模糊/超分/修复)分类解读
- 后验样本数 S 在脚本运行时打印, 校准阈值与总结文字均基于实际S动态计算

素材来源：18.5节后验采样代码、deepinv sampling API
运行前提：需GPU（Colab T4即可），需下载预训练模型(DRUNet/DiffUNet)
"""

import os, sys, time, pickle, hashlib, gc
import numpy as np
import torch

# ★ CUDA显存优化：减少碎片，提升长时间运行的稳定性
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 设置非交互式后端（必须在 import matplotlib.pyplot 之前）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import tqdm as _tqdm_module   # ★ 补上: patch 逻辑需要的是 tqdm 模块对象, 而非 tqdm 类

# ★★★ 清理全局状态（Colab 环境下之前运行失败的 cell 可能污染了 tqdm 模块）★★★
# 在导入后立即恢复 tqdm 的原始状态，避免后续所有 tqdm 调用都触发递归错误。
# 这一步是幂等的（多次运行无副作用），且对本地运行无影响（本地每次都是新进程）。
if hasattr(_tqdm_module, 'tqdm') and hasattr(_tqdm_module.tqdm, '__module__'):
    # 如果 tqdm 已经被污染（不是原始的 tqdm 类），重新导入恢复
    if _tqdm_module.tqdm.__module__ != 'tqdm.std':
        import importlib
        importlib.reload(_tqdm_module)
        tqdm = _tqdm_module.tqdm
        print("[状态恢复] tqdm 模块已重置为原始状态")
else:
    # 备用方案：直接从 tqdm.std 导入
    from tqdm.std import tqdm as _original_tqdm
    _tqdm_module.tqdm = _original_tqdm
    tqdm = _original_tqdm
    print("[状态恢复] tqdm 已从 tqdm.std 恢复")

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验18.5-1')
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

# ★ 缓存配置
use_cache = True
cache_file = os.path.join(SAVE_DIR, 'experiment_cache.pkl')
print(f"缓存配置: use_cache={use_cache}")

# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cpu':
    print("[警告] 扩散采样在CPU上会非常慢，强烈建议使用GPU")

# ========================================================================
# ★ 统一配置管理（借鉴测试.txt的CFG字典设计）
# ========================================================================
CFG = {
    # 运行设备
    "device": str(device),

    # 图像尺寸
    "image_size": 128,  # 平衡效率与质量

    # 噪声水平
    "sigma_data": 0.01,  # 观测噪声（y中加性高斯噪声的标准差）
    "sigma_denoiser": 2.0 / 255.0,  # 去噪器噪声（ScorePrior内部Tweedie公式假设）

    # 退火ULA参数（从测试.py移植优化版）
    "sigma_schedule": [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01],  # 退火调度（7级）
    "ula_steps_each_sigma": 50,  # 每个sigma级别的迭代步数（从40增至50，与测试.py一致）
    "lambda_data": 0.05,  # 数据保真项权重（与测试.py一致，数据保真项已归一化）
    "step_size_coeff_annealed": 0.005,  # 步长系数（保留与测试.py一致）
    "ula_noise_scale": 1.0,  # ★ 新增：Langevin噪声尺度（测试.py验证值）
    "ula_burn_in": 10,  # ★ 新增：每级sigma的burn-in步数（MCMC标准做法）

    # 后验采样数量
    "samples": 30,  # 推荐值：可靠的置信区间估计
}

# ★ 从CFG提取常用变量（保持兼容性）
sigma_data = CFG["sigma_data"]
sigma_denoiser = CFG["sigma_denoiser"]
sigma_schedule = CFG["sigma_schedule"]
ula_steps_each_sigma = CFG["ula_steps_each_sigma"]
lambda_data = CFG["lambda_data"]
step_size_coeff_annealed = CFG["step_size_coeff_annealed"]
ula_noise_scale = CFG["ula_noise_scale"]
ula_burn_in = CFG["ula_burn_in"]
IMG_SIZE = CFG["image_size"]
S = CFG["samples"]

# ★ 打印关键参数
print(f"噪声标准差: sigma_data={sigma_data} (观测), sigma_denoiser={sigma_denoiser:.5f} (去噪器)")
print(f"图像尺寸: {IMG_SIZE}x{IMG_SIZE}")
print(f"后验采样数量: S={S}")
print(f"退火ULA: lambda_data={lambda_data}, step_size_coeff={step_size_coeff_annealed}")
print(f"  噪声尺度: ula_noise_scale={ula_noise_scale}, burn_in={ula_burn_in}步/级")

# ========================================================================
# ★ 参数设计说明（静态说明，不影响代码执行）
# ========================================================================
# 本节说明各采样方法的超参数取值依据，供学生参考。
#
# ------ ULA (Unadjusted Langevin Algorithm) 参数 ------
# 退火ULA使用多级噪声调度 [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
#
# ★ 根本局限性与教学平衡说明:
#   DRUNet 的训练噪声范围为 [0, 50/255≈0.196]，而退火调度的前3级
#   (σ=1.0, 0.5, 0.25) 均超出此范围。在这些 σ 上，DRUNet 输出 D(x,σ)
#   不可靠，score = (D-x)/σ² 的质量下降，导致：
#   (a) 单样本 PSNR ≈ 12 dB（仅略高于观测 11.39 dB）——单样本被噪声淹没
#   (b) 后验均值 PSNR ≈ 15.7 dB——30 个独立样本取平均时噪声部分抵消
#   (c) 93% 区间覆盖率 ≈ 98.8%（略保守）——区间偏宽但覆盖了真值
#
#   为什么不增大 lambda_data 来增强数据保真？
#     lambda_data 控制数据保真项 A^T(Ax-y)/sigma_data² 的权重。
#     增大 lambda（如 0.5）会"钉死"链在伪逆附近，导致样本多样性
#     极低（PSNR std ≈ 0.016 dB），后验均值 PSNR 反而下降到 12.4 dB，
#     校准区间过窄（93% 覆盖率仅 72.9%），教学效果更差。
#
#   为什么不缩短 sigma_schedule 到 DRUNet 工作范围？
#     缩到 [0.15, ..., 0.0025] 后，末级 σ 太小使链"冻结"在伪逆附近，
#     同样导致样本多样性不足，校准从保守变为严重不足。
#
#   当前配置 (lambda=0.05, burn_in=10) 是在教学目标下的最佳平衡：
#     - 后验均值 PSNR 15.7 dB 展示了"多样本平均去噪"的效果
#     - 校准曲线虽然偏保守，但区间覆盖了真值，学生能理解UQ的含义
#     - 若需更高单样本 PSNR，应换用 DPS 采样（见代码降级策略）
#
# 每级噪声的步长: step_size = step_size_coeff_annealed * sigma^2
#   - 物理含义：Langevin 动力学的离散化步长
#   - 诊断信号：样本PSNR.std()应>0.5dB，若<0.5dB提示混合不足
#
# ula_steps_each_sigma（每级噪声的迭代步数）
#   - 物理含义：退火ULA在每个sigma级别运行的步数
#   - 取值依据：50步/级 × 7级 = 350总步数（与测试.py一致）
#
# ula_burn_in（每级sigma的burn-in步数）★新增
#   - 物理含义：每级sigma切换时，先跑N步让chain到达平稳分布，再开始采集中间样本
#   - 取值依据：测试.py验证值10步/级，标准MCMC warm-up做法
#   - 效果：直接降低 std/σ 比值，避免初始瞬态进入终态sample
#
# ula_noise_scale（Langevin噪声尺度）★新增
#   - 物理含义：控制Langevin随机项的幅度
#   - 取值依据：测试.py验证值1.0（标准Langevin SDE离散化）
#   - 调整：增大可加宽后验、改进coverage；减小则后验更紧凑
#
# sigma_denoiser（去噪器噪声水平）
#   - 物理含义：ScorePrior内部Tweedie公式所假设的去噪器训练噪声
#   - 取值依据：deepinv官方示例 sigma_denoiser=2/255
#   - 与sigma_data区别：sigma_data是观测噪声，sigma_denoiser是去噪器内部噪声
#
# lambda_data（数据保真项权重）
#   - 物理含义：控制数据保真项 ||y-A(x)||² 的权重
#   - 取值依据：0.05（测试.py验证值，数据保真项已归一化为A^T(Ax-y)/sigma_data²）
#   - 教学平衡：lambda=0.05 让链保留足够多样性，后验均值 PSNR ≈ 15.7 dB，
#     93% 区间覆盖率 ≈ 98.8%（略保守但覆盖真值）；增大 lambda 会牺牲多样性
#
# ------ DPS (Diffusion Posterior Sampling) 参数 ------
# weight（似然梯度权重）: 1.0（deepinv默认值）
# alpha（扩散步长系数）: 1.0（deepinv默认值）
# num_steps（采样步数）: 100（DPS论文推荐值）
# ========================================================================

# 安装deepinv（带版本检查，避免重复安装）
def _check_deepinv_installed():
    """检查 deepinv 是否已安装及其版本。"""
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', 'deepinv'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split(':')[1].strip()
                    return True, version
            return True, 'unknown'
        return False, None
    except Exception:
        return False, None

_dinv_installed, _dinv_version = _check_deepinv_installed()

if _dinv_installed:
    print(f"[安装检查] deepinv 已安装，版本: {_dinv_version}")
    try:
        import deepinv as dinv
        print(f"deepinv 版本: {dinv.__version__}")
    except ImportError as e:
        print(f"[警告] deepinv 已安装但导入失败: {e}")
        print("[警告] 尝试重新安装...")
        _dinv_installed = False

def _install_deepinv():
    """安装 deepinv，添加 --no-deps 避免重复安装依赖，增加超时控制。"""
    import subprocess
    print("正在安装 deepinv ...")
    try:
        # ★ 使用 --no-deps 避免重复安装已有的 torch 等大依赖，加速安装
        # ★ 添加 --quiet 减少输出噪音
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            '--no-deps', '--quiet',
            'git+https://github.com/deepinv/deepinv.git#egg=deepinv'
        ], timeout=300)  # 5分钟超时
        # ★ 安装成功后重新导入
        import deepinv as dinv
        print(f"deepinv 安装成功，版本: {dinv.__version__}")
        return dinv
    except subprocess.TimeoutExpired:
        print("[错误] deepinv 安装超时（>5分钟），网络可能不稳定")
        print("[错误] 建议手动安装: pip install git+https://github.com/deepinv/deepinv.git")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[错误] deepinv 安装失败，错误码: {e.returncode}")
        print(f"[错误] 请手动安装: pip install git+https://github.com/deepinv/deepinv.git")
        sys.exit(1)

if not _dinv_installed:
    dinv = _install_deepinv()

from deepinv.physics import Blur, Inpainting, GaussianNoise
from deepinv.utils import load_example


# ========================================================================
# 辅助函数
# ========================================================================

def clear_gpu():
    """清理GPU显存（借鉴测试.txt）。

    包含三级清理：
    1. gc.collect(): Python垃圾回收
    2. torch.cuda.empty_cache(): PyTorch缓存清理
    3. torch.cuda.ipc_collect(): 进程间通信缓存清理
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def compute_psnr(img1, img2):
    """计算PSNR (dB)。

    注意: 本函数假设图像动态范围为[0,1]，但不执行clamp操作。
    理由:
    1. PSNR应反映真实重建误差，而非人为限制范围后的误差
    2. 后验采样结果可能超出[0,1]（如迭代去噪过程中），clamp会低估真实误差
    3. 可视化时统一clamp(0,1)是为了显示，但PSNR计算应保持原始数值精度

    参数:
        img1, img2: 4D张量 (B, C, H, W)，假设值域[0,1]但允许超出
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


# ========================================================================
# Step 1: 从点估计到分布 —— 单次重建 vs 多次后验采样
# 对应18.5节知识点：点估计的局限性与后验分布的意义
# ========================================================================
print("\n" + "="*70)
print("Step 1: 从点估计到分布")
print("="*70)

print("""
点估计的局限性:
  单次重建 x^ = f(y) 只给出一个解，无法量化可靠性
  对于欠定逆问题（如50%像素缺失的修复——本实验实际配置），不同先验
  可能给出完全不同的解

后验分布的价值:
  p(x|y) 描述了所有与观测一致的解的概率分布
  多次采样 {x^(s)} ~ p(x|y) 可量化:
  - 均值 E[x|y]: 最优点估计
  - 标准差 std[x|y]: 像素级不确定性
  - 分位数: 置信区间
""")

# 加载测试图像
# ★ 使用 IMG_SIZE（小图诊断模式下为64，完整模式下为256）
x_true = load_example("celeba_example.jpg", img_size=(IMG_SIZE, IMG_SIZE), resize_mode='resize')
# 确保是4D张量 (B, C, H, W)
if x_true.ndim == 3:
    x_true = x_true.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
elif x_true.ndim == 5:
    x_true = x_true.squeeze(0) if x_true.shape[0] == 1 else x_true[0]
x_true = x_true.to(device)

# 创建退化模型（修复场景，50%缺失——欠定程度适中）
torch.manual_seed(42)
physics_inp = Inpainting(img_size=(3, IMG_SIZE, IMG_SIZE), mask=0.5, device=device)
physics_inp.set_noise_model(GaussianNoise(sigma=sigma_data))
y_inp = physics_inp(x_true)

print(f"真值 shape: {x_true.shape}")
print(f"观测 shape: {y_inp.shape}")
print(f"观测 PSNR: {compute_psnr(x_true, y_inp):.2f} dB")


# ========================================================================
# ★ 退火ULA采样函数（从测试.py移植优化版）
# ========================================================================
# ★ 全局：tqdm 进度条开关
#  - Jupyter/Colab 环境下 tqdm.auto 会自动选择 notebook widget
#  - 纯脚本/管道/CI 环境下自动降级为 ASCII 进度条
#  - 需要静默运行（如重定向到日志）时，置为 False 即可
try:
    from tqdm.auto import tqdm as _tqdm_auto
except Exception:
    _tqdm_auto = tqdm  # 退化到已导入的普通 tqdm
# 检测 stdout 是否是 tty：管道/CI 环境关闭进度条避免刷屏
_TQDM_ENABLED = sys.stdout.isatty() or bool(getattr(sys, 'ps1', None))
def sample_annealed_ula(y_obs, physics, model, device, sigma_schedule, steps_per_sigma,
                         lambda_data, step_size_coeff, S=1, burn_in_steps=0, noise_scale=1.0,
                         tqdm_enabled=None):
    """
    退火ULA后验采样（含burn-in和数值稳定性保护）

    参数:
        y_obs: 观测 (1, C, H, W)
        physics: 物理算子（Inpainting）
        model: DRUNet去噪器
        device: 运行设备
        sigma_schedule: 退火调度 [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
        steps_per_sigma: 每个sigma级别的步数
        lambda_data: 数据保真权重
        step_size_coeff: 步长系数
        S: 样本数
        burn_in_steps: ★新增 每级sigma的burn-in步数（不收集中间样本）
        noise_scale: ★新增 Langevin噪声尺度
        tqdm_enabled: ★新增 是否显示进度条（None=根据环境自动判断，True/False 强制开关）

    返回:
        (samples, n_diverged_steps) - 元组:
            samples: (S, C, H, W) 后验样本
            n_diverged_steps: 整个采样过程中触发nan_to_num修正的累计步数
    """
    samples = []
    model.eval()  # ★ 设置为eval模式，减少显存占用

    # ★ 数值稳定性监控：累计整轮 ULA 链中触发 nan_to_num 修正的步数
    # 若 n_diverged_steps > 0，说明链在中途曾发散，calibration 与
    # spatial std 是在"被救回的样本"上计算的，可能低估真实发散程度
    n_diverged_steps = 0

    # ★ 解析 tqdm_enabled：None 时按全局开关自动判断
    if tqdm_enabled is None:
        tqdm_enabled = _TQDM_ENABLED

    # ★ 外层 tqdm 进度条：30 个样本（用 disable 参数可一行关闭进度条）
    for sample_id in _tqdm_auto(range(S), desc="[退火ULA] 采样样本", unit="个", ncols=80, disable=not tqdm_enabled):
        # ★ 为每个样本设置不同的随机种子（与DPS/PnP保持一致）
        torch.manual_seed(sample_id * 1000 + 42)

        # 清理显存
        clear_gpu()

        # 初始化：伪逆 + 小噪声（与测试.py一致）
        x = physics.A_dagger(y_obs)
        x = x + 0.01 * torch.randn_like(x)
        x = x.clamp(0, 1)

        # 退火循环（每级 sigma 独立显示内层进度条）
        for sigma in sigma_schedule:
            step_size = step_size_coeff * (sigma ** 2)

            # ★ Burn-in阶段：先让chain到达平稳分布，不收集中间样本
            # 这是MCMC标准的warm-up做法，直接降低 std/σ 比值
            for _ in range(burn_in_steps):
                # 计算score: (D(x,sigma) - x) / sigma^2
                B = x.shape[0]
                # ★ sigma 兼容性处理：构造4D sigma_map (B,1,1,1) 传入 DRUNet。
                # deepinv 的 DRUNet 同时兼容 1D 和 4D sigma 输入，此处使用 4D 形式。
                sigma_1d = torch.full((B,), float(sigma), device=device)
                sigma_map = sigma_1d.view(B, 1, 1, 1)

                with torch.no_grad():
                    denoised = model(x, sigma_map)

                score = (denoised - x) / (sigma ** 2)

                # 数据保真梯度: A^T(Ax - y) / sigma_data^2
                data_grad = physics.A_adjoint(physics.A(x) - y_obs) / (sigma_data ** 2)

                # 后验梯度: score - lambda * data_grad
                grad = score - lambda_data * data_grad

                # Langevin更新（带噪声尺度控制）
                noise = torch.randn_like(x)
                x = x + step_size * grad + noise_scale * torch.sqrt(torch.tensor(2 * step_size, device=device)) * noise

                # ★ 数值稳定性保护
                if not torch.isfinite(x).all():
                    n_diverged_steps += 1
                    print(f"  [ula burn-in] non-finite state at sigma={sigma:.4f}, "
                          f"clamping & resuming (total diverged={n_diverged_steps})")
                    x = torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)

                # ★ 软投影：允许略微越界（借鉴测试.py第819行）
                # 范围(-0.1, 1.1)比测试.py的(-0.5, 1.5)更紧，减少过度保守
                # 既保持Langevin遍历性，又避免收集样本前的clamp(0,1)引入伪方差
                x = x.clamp(-0.1, 1.1)

                # 清理中间变量
                del denoised, score, sigma_map, data_grad, grad, noise

            # ★ 正式采样阶段：内层 tqdm 进度条，每级 sigma 显示
            # leave=False 让内层进度条用完即消失；总迭代数 = steps_per_sigma
            with _tqdm_auto(total=steps_per_sigma,
                      desc=f"  σ={sigma:.4f} 采样",
                      unit="步",
                      ncols=80,
                      leave=False,
                      disable=not tqdm_enabled) as pbar:
                for step in range(steps_per_sigma):
                    # 计算score
                    B = x.shape[0]
                    # ★ 与burn-in阶段保持一致，构造4D sigma_map
                    sigma_1d = torch.full((B,), float(sigma), device=device)
                    sigma_map = sigma_1d.view(B, 1, 1, 1)

                    # ★ 使用no_grad减少显存
                    with torch.no_grad():
                        denoised = model(x, sigma_map)

                    score = (denoised - x) / (sigma ** 2)

                    # 数据保真梯度
                    data_grad = physics.A_adjoint(physics.A(x) - y_obs) / (sigma_data ** 2)

                    # 后验梯度
                    grad = score - lambda_data * data_grad

                    # Langevin更新（带噪声尺度控制）
                    noise = torch.randn_like(x)
                    x = x + step_size * grad + noise_scale * torch.sqrt(torch.tensor(2 * step_size, device=device)) * noise

                    # ★ 数值稳定性保护
                    if not torch.isfinite(x).all():
                        n_diverged_steps += 1
                        print(f"  [ula sample] non-finite state at sigma={sigma:.4f}, "
                              f"clamping & resuming (total diverged={n_diverged_steps})")
                        x = torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)

                    # ★ 软投影：每步都保持遍历性
                    x = x.clamp(-0.1, 1.1)

                    # ★ 定期清理中间变量（仅empty_cache，gc.collect留到每样本结束）
                    del denoised, score, sigma_map, data_grad, grad, noise
                    if step % 20 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # ★ 推进内层进度条
                    pbar.update(1)

        # ★ 收集样本前最终硬约束到[0,1]（保证PSNR/校准计算正确）
        x = x.clamp(0, 1)

        # ★ 样本存储后立即移到CPU并清理显存
        samples.append(x.detach().cpu())
        del x
        clear_gpu()

    samples_tensor = torch.stack(samples)
    # ★ 进度条结束换行（避免"完成"信息接在最后一个进度行后面）
    print()
    print(f"[退火ULA] 完成。n_diverged_steps={n_diverged_steps} (0=无nan_to_num触发)")
    return samples_tensor, n_diverged_steps


# ========================================================================
# Step 2: 后验采样实现
# 对应18.5节知识点：PnP-ULA与扩散采样
# ========================================================================
print("\n" + "="*70)
print("Step 2: 后验采样实现")
print("="*70)

print("""
★ 自动降级策略说明:
  本脚本采用"尝试-失败-降级"策略：
  1. 优先尝试 ULA 采样（最准确的后验采样方法）
  2. 若 ULA 失败，降级到 DPS 扩散采样
  3. 若 DPS 失败，降级到 PnP 近似采样
  4. 若 PnP 失败，降级到伪逆+噪声近似（最粗糙）

  ⚠️ 注意: 这种降级策略是工程容错设计，但会产生"混合样本"：
     - 混合样本的统计同质性较差（不同样本来自不同的后验近似）
     - 混合采样时，统计结果（均值、标准差）可能受方法切换影响
     - 代码会在混合采样时打印警告，并改用按方法分组的可视化

  如果需要严格统计同质性，建议：
  - 固定使用单一方法（删除其他分支）
  - 或增大样本数 S 以降低单方法失败的影响
""")

ula_method = None

# ★ 后验采样数量已在CFG中定义，此处打印确认
print(f"[配置] 后验采样数量 S={S}")

all_samples = []
sample_times = []
# ★ 新增：记录每个样本的来源方法，避免混合采样标签污染
sample_methods = []
# ★ 新增：用于动态生成总结文字的PSNR-vs-S曲线（None表示未计算）
psnr_by_s = None

import traceback

# ★ 缓存超参数指纹（与第4-6章 checkpoint 约定一致）
# 目的：检测"修改 sigma_data/换图/改 S/换设备"后误用旧缓存的隐患。
# 物理算子 mask=0.5 与 Inpainting 构造的 torch.manual_seed(42) 固定，
# 物理算子类型本身由代码写死（Inpainting+50%缺失），因此指纹只需覆盖
# 真正可能变化的维度：观测噪声、图像内容、目标样本数、运行设备。
# 图像内容用整图 MD5(16位) + 统计量(mean/std) 双重把关，避免仅哈希
# 前若干字节的碰撞风险（与第4-6章约定一致：hash + 统计量兜底）。
def _compute_fingerprint():
    """基于当前超参数计算缓存指纹字典，确保改动后自动丢弃旧缓存。

    指纹覆盖所有影响采样结果的超参数：
    - sigma_data: 观测噪声水平（影响 ULA step_size、DPS 似然缩放）
    - sigma_denoiser: 去噪器噪声水平（影响 ScorePrior 的 Tweedie 公式）
    - step_size_coeff: ULA Langevin 步长系数
    - ula_max_iter: ULA 链最大迭代次数
    - IMG_SIZE: 图像尺寸（影响维度和稳定性边界）
    - x_true: 测试图像内容（MD5 + 统计量双重校验）
    - S: 目标样本数
    - device: 运行设备（GPU vs CPU 影响随机数生成）
    """
    arr = x_true.detach().cpu().contiguous().numpy()
    img_hash = hashlib.md5(arr.tobytes()).hexdigest()[:16]  # 整图 MD5 前 16 位
    return {
        'sigma_data': float(sigma_data),
        'sigma_denoiser': float(sigma_denoiser),
        'x_true_shape': tuple(arr.shape),
        'x_true_hash': img_hash,
        'x_true_mean': float(arr.mean()),
        'x_true_std': float(arr.std()),
        'IMG_SIZE': int(IMG_SIZE),
        'S': int(S),
        'device': str(device),
        # ★ 退火ULA参数
        'sigma_schedule': tuple(sigma_schedule),
        'ula_steps_each_sigma': int(ula_steps_each_sigma),
        'lambda_data': float(lambda_data),
        'step_size_coeff_annealed': float(step_size_coeff_annealed),
        'ula_noise_scale': float(ula_noise_scale),  # ★ 补充：Langevin噪声尺度影响采样结果
        'ula_burn_in': int(ula_burn_in),  # ★ 补充：burn-in步数影响采样结果
    }

def _compare_fingerprint(stored_fp, current_fp):
    """逐字段对比指纹，返回 (matched: bool, diff_message: str)。"""
    if stored_fp is None:
        return False, "缓存无 fingerprint 字段（疑似旧版缓存）"
    diffs = []
    for k, v_now in current_fp.items():
        v_old = stored_fp.get(k, '<missing>')
        if v_old != v_now:
            diffs.append(f"  - {k}: 缓存={v_old!r}, 当前={v_now!r}")
    if diffs:
        return False, "指纹不匹配:\n" + "\n".join(diffs)
    return True, "指纹一致"

_current_fp = _compute_fingerprint()

cached_samples = []
cached_methods = []  # ★ 修改：支持样本级方法记录
cached_times = []    # ★ 新增：支持样本级时间记录
if use_cache and os.path.exists(cache_file):
    try:
        if os.path.getsize(cache_file) > 0:
            # ★ 改用 torch.load（对应 torch.save 格式）
            # 兼容旧版 pickle 缓存：先尝试 torch.load，失败回退到 pickle
            cached_data = None
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = torch.load(f, weights_only=False)  # weights_only=False 以加载非张量字段
            except Exception:
                # 回退到旧版 pickle 格式
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    print(f"[缓存] 检测到旧版 pickle 缓存，将自动迁移到 torch.save 格式")
                    # 迁移为新格式
                    try:
                        torch.save(cached_data, cache_file)
                        print(f"[缓存] 旧版缓存已迁移到 torch.save 格式")
                    except Exception as e:
                        print(f"[缓存] 缓存迁移失败（不影响本次运行）: {e}")
                except Exception as e:
                    print(f"[缓存] 旧版 pickle 也无法加载: {e}")
                    cached_data = None
            if cached_data is None:
                raise EOFError("缓存加载失败")
            # ★ 指纹校验：与第4-6章 checkpoint 约定一致
            stored_fp = cached_data.get('fingerprint')
            fp_matched, fp_msg = _compare_fingerprint(stored_fp, _current_fp)
            if not fp_matched:
                print(f"⚠️ [缓存] {fp_msg}")
                print(f"⚠️ [缓存] 丢弃旧缓存，将基于当前 (sigma_data={sigma_data}, "
                      f"image_hash={_current_fp['x_true_hash']}, S={S}) 重新采样")
                try:
                    os.remove(cache_file)
                except OSError:
                    pass
            elif 'samples' in cached_data and len(cached_data['samples']) > 0:
                # ★ 兼容两种格式：torch.save 的 samples 是 (S, 1, C, H, W) 张量
                #   旧版 pickle 的 samples 是 list of (1, C, H, W) 张量
                raw_samples = cached_data['samples']
                if isinstance(raw_samples, torch.Tensor):
                    # ★ torch.save 格式：堆叠张量，转换为 list 保持后续索引访问一致
                    cached_samples = [raw_samples[i] for i in range(raw_samples.shape[0])]
                else:
                    # ★ 旧版 pickle 格式：list of tensors
                    cached_samples = raw_samples
                # ★ 兼容旧缓存：如果没有methods字段，使用method字段
                cached_methods = cached_data.get('methods', [])
                if not cached_methods and 'method' in cached_data:
                    cached_methods = [cached_data['method']] * len(cached_samples)
                # ★ 兼容旧缓存：如果没有times字段，补充为空列表
                cached_times = cached_data.get('times', [])
                if cached_samples and not cached_times:
                    # ★ 旧版缓存不含耗时字段，后续平均耗时统计仅包含本次新增样本
                    print(f"⚠️ 缓存为旧版格式（无times字段），平均耗时统计将仅包含本次新增样本")
                print(f"[缓存] 指纹一致 (sigma_data={sigma_data}, image_hash={_current_fp['x_true_hash']}, S={S})")
                print(f"[缓存] 加载了 {len(cached_samples)} 个样本")
                # ★ 打印方法分布
                method_counts = {}
                for m in cached_methods:
                    method_counts[m] = method_counts.get(m, 0) + 1
                print(f"[缓存] 方法分布: {method_counts}")
                # ★ 降级结果警告：缓存样本并非全部来自 ULA(最优方法)
                # 即便 fingerprint 匹配, 也不应静默复用降级结果, 避免学生因
                # 首次偶发失败而错过本可获得的更优样本。
                # 兼容旧缓存: 没有 best_achieved 字段时按方法分布推断
                best_achieved = cached_data.get('best_achieved', None)
                if best_achieved is None:
                    # 旧版缓存: 检查是否所有方法都是有效后验方法（ULA/退火ULA）
                    # ★ 修复：原代码硬编码 ["ULA"]，无法识别 "退火ULA"
                    valid_ula_methods = {"ULA", "退火ULA"}
                    best_achieved = all(m in valid_ula_methods for m in method_counts.keys())
                if not best_achieved:
                    print(f"⚠️⚠️⚠️ [缓存] 缓存中的样本并非全部来自 ULA (最优方法)")
                    print(f"⚠️⚠️⚠️ [缓存] 方法分布: {method_counts}")
                    print(f"⚠️⚠️⚠️ [缓存] 如本次运行 ULA/DPS 已可正常采样, 建议删除缓存以重新尝试更优方法")
                    print(f"⚠️⚠️⚠️ [缓存]   rm \"{cache_file}\"")
                if len(cached_samples) >= S:
                    print(f"[缓存] 样本数已满足需求，跳过采样")
                    all_samples = cached_samples[:S]
                    sample_methods = cached_methods[:S]
                    sample_times = cached_times[:S] if cached_times else []  # ★ 加载时间记录
                    # ★ 计算最终方法（检查是否混合采样）
                    unique_methods = list(set(sample_methods))
                    if len(unique_methods) == 1:
                        ula_method = unique_methods[0]
                    else:
                        ula_method = "混合采样"
        else:
            print(f"[缓存] 缓存文件为空，将重新采样")
            os.remove(cache_file)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"[缓存] 缓存文件损坏，将重新采样")
        try:
            os.remove(cache_file)
        except:
            pass
    except Exception as e:
        print(f"[缓存] 加载失败: {e}")

def save_samples_to_cache(samples, methods, times=None):
    """保存样本到缓存，包含样本级方法标签、超参数指纹与"是否达到最优方法"标记。

    字段说明:
    - fingerprint: 输入维度指纹（sigma_data/x_true/S/device），与加载侧 _compare_fingerprint 配套
    - best_achieved: bool, 是否所有样本均来自 ULA（最优方法）。
      加载时若为 False, 说明缓存中包含降级结果（DPS/PnP/伪逆）, 即便 fingerprint
      匹配也会打醒目警告，避免学生因首次偶发失败而静默复用质量较差的样本。

    存储格式说明:
    - 改用 torch.save + 堆叠张量, 相比 pickle 列表式存储可减少 80%+ 空间
    - pickle 存储 30 个独立张量时会重复序列化 dtype/stride/storage 元数据
      及 Python 对象包装, 100MB+ 文件实际仅含 5.6MB 有效数据
    - 堆叠后 torch.save 直接写入连续内存块, 同样数据量约 10-15MB
    """
    if not use_cache:
        return None
    try:
        # ★ best_achieved: 仅当所有样本方法标签都是有效ULA方法时为 True
        # "混合采样" 或任何降级方法都会让 best_achieved = False
        # ★ 修复：原代码硬编码 m == "ULA"，无法识别 "退火ULA"
        valid_ula_methods = {"ULA", "退火ULA"}
        best_achieved = bool(methods) and all(m in valid_ula_methods for m in methods)
        # ★ 堆叠为单个张量后再保存，大幅减少序列化开销
        #   保留 list 形式以兼容后续 all_samples[i] 索引访问
        samples_tensor = torch.stack(samples, dim=0)  # (S, 1, C, H, W)
        cached_data = {
            'samples': samples_tensor,  # ★ 改为堆叠后的张量（节省空间）
            'methods': methods,
            'times': times or [],
            'fingerprint': _compute_fingerprint(),  # ★ 与加载侧 _compare_fingerprint 配套
            'best_achieved': best_achieved,
        }
        # ★ 使用 torch.save 替代 pickle.dump
        #   - torch.save 直接序列化张量数据，零 Python 对象包装开销
        #   - 默认使用 pickle 协议但仅打包 tensor 元数据
        #   - 30样本×128²×3 float32 ≈ 5.6MB 原始数据，torch.save 后约 6-8MB
        with open(cache_file, 'wb') as f:
            torch.save(cached_data, f)
        return best_achieved  # ★ 返回质量标记, 供循环结束后统一汇总打印
    except Exception as e:
        print(f"[缓存] 保存失败: {e}")  # 异常时打印（会打断进度条，但异常本身就会中断程序）
        return None

def release_model(model_var_name):
    """释放全局作用域中的模型变量并清理显存。

    参数:
        model_var_name: 全局变量名（字符串），如 'denoiser_ula'

    实现说明:
        使用 globals() 字典检查和删除全局变量，因为函数内部无法通过
        局部作用域访问全局变量。删除后调用 clear_gpu() 立即释放显存。
    """
    if model_var_name in globals():
        del globals()[model_var_name]
        print(f"[显存] 已删除全局变量 {model_var_name}")
    else:
        print(f"[显存] 全局变量 {model_var_name} 不存在，可能已被释放")
    clear_gpu()
    if torch.cuda.is_available():
        print(f"[显存] 当前显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

if len(all_samples) < S:
    start_idx = len(all_samples)
    print(f"需要采样 {S - start_idx} 个样本")

    # ★ 修改：使用 cached_methods 和 cached_times 而非 cached_method
    if cached_samples and cached_methods:
        all_samples = cached_samples.copy()
        sample_methods = cached_methods.copy()
        sample_times = cached_times.copy() if cached_times else []  # ★ 加载已有时间记录

    # ========== 2a. ULA 采样 ==========
    if len(all_samples) < S:
        print("\n" + "="*50)
        print("--- 2a. ULA 后验采样 ---")
        print("="*50)

        denoiser_ula = None
        try:
            from deepinv.models import DRUNet
            print("[退火ULA] 加载 DRUNet...")
            denoiser_ula = DRUNet(pretrained='download').to(device)
            print(f"[退火ULA] DRUNet 加载成功，显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

            # ★ 验证预训练权重是否加载
            total_params = sum(p.numel() for p in denoiser_ula.parameters())
            non_zero_params = sum((p != 0).sum().item() for p in denoiser_ula.parameters())
            print(f"[退火ULA] 总参数: {total_params}, 非零参数: {non_zero_params}")
            if non_zero_params < total_params * 0.5:
                print("[退火ULA] ⚠️ 警告: 超过50%参数为零，可能未正确加载预训练权重！")
            else:
                print("[退火ULA] ✓ 预训练权重加载正常")
        except Exception as e:
            print(f"[退火ULA] DRUNet 加载失败!")
            print(f"[退火ULA] 错误类型: {type(e).__name__}")
            print(f"[退火ULA] 错误信息: {e}")
            print("[退火ULA] 详细堆栈:")
            traceback.print_exc()

        if denoiser_ula is not None:
            try:
                print(f"[退火ULA] 参数配置:")
                print(f"  - sigma_schedule: {sigma_schedule}")
                print(f"  - steps_per_sigma: {ula_steps_each_sigma}")
                print(f"  - lambda_data: {lambda_data}")
                print(f"  - step_size_coeff: {step_size_coeff_annealed}")
                total_steps = len(sigma_schedule) * ula_steps_each_sigma
                print(f"  - 总步数: {total_steps} ({len(sigma_schedule)}级 × {ula_steps_each_sigma}步)")

                start_idx = len(all_samples)
                samples_needed = S - start_idx

                t_start_all = time.time()
                # ★ 修改：适配新的元组返回值 (samples, n_diverged_steps)
                annealed_samples, n_diverged = sample_annealed_ula(
                    y_obs=y_inp,
                    physics=physics_inp,
                    model=denoiser_ula,
                    device=device,
                    sigma_schedule=sigma_schedule,
                    steps_per_sigma=ula_steps_each_sigma,
                    lambda_data=lambda_data,
                    step_size_coeff=step_size_coeff_annealed,
                    S=samples_needed,
                    burn_in_steps=ula_burn_in,  # ★ 新增：burn-in步数
                    noise_scale=ula_noise_scale  # ★ 新增：Langevin噪声尺度
                )
                t_total = time.time() - t_start_all

                # 添加样本到列表
                for i, sample in enumerate(annealed_samples):
                    all_samples.append(sample)
                    sample_methods.append("退火ULA")
                    # 估算每个样本的耗时（平均）
                    sample_times.append(t_total / samples_needed)

                # 计算PSNR（一次性统计，不再逐样本打印）
                sample_psnrs = []
                for i, sample in enumerate(annealed_samples):
                    # ★ 样本在CPU上，需移到GPU再计算PSNR
                    psnr_val = compute_psnr(x_true, sample.to(device))
                    sample_psnrs.append(psnr_val)
                # ★ 汇总报告（用一行展示，避免30行堆叠）
                psnr_min, psnr_max, psnr_mean = min(sample_psnrs), max(sample_psnrs), sum(sample_psnrs) / len(sample_psnrs)
                print(f"[退火ULA] 采样完成! 总耗时: {t_total:.1f}s")
                print(f"[退火ULA] 样本 PSNR 范围: [{psnr_min:.2f}, {psnr_max:.2f}] dB, 均值: {psnr_mean:.2f}dB")
                print(f"[退火ULA] 数值稳定性: n_diverged_steps={n_diverged} (0=无nan_to_num触发)")
                print(f"[缓存] 已保存 {len(all_samples)} 个样本 (全部退火ULA)")
                save_samples_to_cache(all_samples, sample_methods, sample_times)

            except Exception as e:
                print(f"[退火ULA] 采样失败!")
                print(f"[退火ULA] 错误类型: {type(e).__name__}")
                print(f"[退火ULA] 错误信息: {e}")
                print("[退火ULA] 详细堆栈:")
                traceback.print_exc()
            finally:
                release_model('denoiser_ula')

    # ========== 2b. DPS 采样 ==========
    if len(all_samples) < S:
        print("\n" + "="*50)
        print("--- 2b. DPS 扩散采样 ---")
        print("="*50)

        denoiser_dps = None
        try:
            from deepinv.models import DiffUNet
            print("[DPS] 加载 DiffUNet...")
            denoiser_dps = DiffUNet(pretrained='download').to(device)
            print(f"[DPS] DiffUNet 加载成功，显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        except Exception as e:
            print(f"[DPS] DiffUNet 加载失败!")
            print(f"[DPS] 错误类型: {type(e).__name__}")
            print(f"[DPS] 错误信息: {e}")
            print("[DPS] 详细堆栈:")
            traceback.print_exc()

        if denoiser_dps is not None:
            try:
                from deepinv.sampling import DPS

                print("[DPS] 创建 DPS 采样器...")
                n_dps_iter = 100
                # ★ DPS 核心参数（详细说明见文件开头"参数设计说明"区块）
                # weight=1.0: 似然梯度权重，deepinv默认值，对应DPS论文标准设置
                # alpha=1.0: 扩散步长系数，deepinv默认值
                # 噪声水平由 physics_inp.noise_model 提供（sigma_data=0.01）
                # DPS 内部用 1/sigma_data² 缩放似然梯度（已在开头参数验证中打印）
                dps = DPS(
                    denoiser=denoiser_dps,
                    schedule="vp",
                    num_steps=n_dps_iter,
                    weight=1.0,
                    alpha=1.0,
                    verbose=False,
                    device=device,
                )

                print(f"[DPS] 开始采样 (每样本 {n_dps_iter} 步)...")
                start_idx = len(all_samples)
                pbar = tqdm(range(start_idx, S), desc="[DPS] 采样", unit="样本")

                for s in pbar:
                    # ★ 使用差异更大的种子确保样本多样性
                    torch.manual_seed(s * 1000 + 42)
                    t_start = time.time()
                    dps_result = dps(y_inp, physics_inp, seed=s * 1000 + 42)
                    x_sample = dps_result[0] if isinstance(dps_result, tuple) else dps_result
                    t_sample = time.time() - t_start
                    all_samples.append(x_sample.detach().cpu())
                    sample_times.append(t_sample)  # ★ 记录实际耗时
                    sample_methods.append("DPS")  # ★ 记录样本来源方法
                    psnr_val = compute_psnr(x_true, x_sample)
                    pbar.set_postfix({"耗时": f"{t_sample:.1f}s", "PSNR": f"{psnr_val:.1f}dB"})
                    del x_sample, dps_result
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    save_samples_to_cache(all_samples, sample_methods, sample_times)

                if sample_times:
                    cache_best = all(m in {"ULA", "退火ULA"} for m in sample_methods)
                    print(f"[DPS] 采样完成! 平均耗时: {np.mean(sample_times[-(S-start_idx):]):.2f}s")
                    if use_cache:
                        qtag = "全部ULA(最优)" if cache_best else "含降级样本"
                        print(f"[缓存] 已保存 {len(all_samples)} 个样本 (含 fingerprint, {qtag})")

            except ImportError as e:
                print(f"[DPS] 导入失败: {e}")
                print("[DPS] 详细堆栈:")
                traceback.print_exc()
            except Exception as e:
                print(f"[DPS] 采样失败!")
                print(f"[DPS] 错误类型: {type(e).__name__}")
                print(f"[DPS] 错误信息: {e}")
                print("[DPS] 详细堆栈:")
                traceback.print_exc()
            finally:
                release_model('denoiser_dps')

    # ========== 2c. PnP 近似采样 ==========
    if len(all_samples) < S:
        print("\n" + "="*50)
        print("--- 2c. PnP 近似后验采样 ---")
        print("="*50)

        denoiser_pnp = None
        try:
            from deepinv.models import DRUNet
            print("[PnP] 加载 DRUNet...")
            denoiser_pnp = DRUNet(pretrained='download').to(device)
            print(f"[PnP] DRUNet 加载成功，显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        except Exception as e:
            print(f"[PnP] DRUNet 加载失败!")
            print(f"[PnP] 错误类型: {type(e).__name__}")
            print(f"[PnP] 错误信息: {e}")
            print("[PnP] 详细堆栈:")
            traceback.print_exc()

        if denoiser_pnp is not None:
            try:
                print("[PnP] 开始采样...")
                start_idx = len(all_samples)
                pbar_outer = tqdm(range(start_idx, S), desc="[PnP] 采样", unit="样本")
                for s in pbar_outer:
                    t_start = time.time()
                    # ★ 与 ULA/DPS 一致, 使用 s*1000+42 公式
                    torch.manual_seed(s * 1000 + 42)
                    x_pnp = physics_inp.A_adjoint(y_inp) + 0.05 * torch.randn_like(x_true)

                    n_iter = 20
                    pbar_inner = tqdm(range(n_iter), desc=f"  [PnP] 样本 {s+1} 迭代", unit="步",
                                      leave=False)
                    for it in pbar_inner:
                        with torch.no_grad():
                            grad = physics_inp.A_adjoint(physics_inp.A(x_pnp) - y_inp)
                            x_pnp = x_pnp - 0.5 * grad
                            sigma_cur = max(0.1 * (1 - it / n_iter), 0.01)
                            noise_level = torch.tensor([sigma_cur] * x_pnp.shape[0]).to(device)
                            x_pnp = denoiser_pnp(x_pnp, noise_level)
                            del grad, noise_level
                        if (it + 1) % 5 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    pbar_inner.close()

                    t_sample = time.time() - t_start
                    all_samples.append(x_pnp.detach().cpu())
                    sample_times.append(t_sample)  # ★ 记录实际耗时
                    sample_methods.append("PnP近似")  # ★ 记录样本来源方法
                    del x_pnp
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    save_samples_to_cache(all_samples, sample_methods, sample_times)
                    pbar_outer.set_postfix({"耗时": f"{t_sample:.1f}s"})

                if sample_times:
                    cache_best = all(m in {"ULA", "退火ULA"} for m in sample_methods)
                    print(f"[PnP] 采样完成! 平均耗时: {np.mean(sample_times[-(S-start_idx):]):.2f}s")
                    if use_cache:
                        qtag = "全部ULA(最优)" if cache_best else "含降级样本"
                        print(f"[缓存] 已保存 {len(all_samples)} 个样本 (含 fingerprint, {qtag})")

            except Exception as e:
                print(f"[PnP] 采样失败!")
                print(f"[PnP] 错误类型: {type(e).__name__}")
                print(f"[PnP] 错误信息: {e}")
                print("[PnP] 详细堆栈:")
                traceback.print_exc()
            finally:
                release_model('denoiser_pnp')

    # ========== 2d. 伪逆近似 ==========
    if len(all_samples) < S:
        print("\n" + "="*50)
        print("--- 2d. 伪逆+噪声近似 ---")
        print("="*50)

        start_idx = len(all_samples)
        pbar = tqdm(range(start_idx, S), desc="[伪逆] 采样", unit="样本",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        for s in pbar:
            t_start = time.time()
            # ★ 与 ULA/DPS/PnP 一致, 使用 s*1000+42 公式
            torch.manual_seed(s * 1000 + 42)
            x_approx = physics_inp.A_adjoint(y_inp) + 0.02 * torch.randn_like(x_true)
            t_sample = time.time() - t_start
            all_samples.append(x_approx.cpu())
            sample_times.append(t_sample)  # ★ 记录实际耗时(伪逆操作极快)
            sample_methods.append("伪逆+噪声")  # ★ 记录样本来源方法
            # ★ 修复：与 ULA/DPS/PnP 三个分支保持一致, 改为逐样本保存
            #   理由: 伪逆分支虽然极快(无 GPU 显存压力), 但如果未来在该循环
            #   中加入耗时操作(如伪逆后跑一轮快速去噪), 仍需保证中断后能
            #   从缓存中恢复; 同时保持四个分支的代码风格统一, 便于阅读。
            del x_approx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            save_samples_to_cache(all_samples, sample_methods, sample_times)
            pbar.set_postfix({"耗时": f"{t_sample:.1f}s"})

        if use_cache:
            cache_best = all(m in {"ULA", "退火ULA"} for m in sample_methods)
            qtag = "全部ULA(最优)" if cache_best else "含降级样本"
            print(f"[缓存] 已保存 {len(all_samples)} 个样本 (含 fingerprint, {qtag})")

# ★ 计算最终采样方法（检查是否混合采样）
if len(sample_methods) > 0:
    unique_methods = list(set(sample_methods))
    if len(unique_methods) == 1:
        ula_method = unique_methods[0]
    else:
        ula_method = "混合采样"
        # 计算各方法占比
        method_counts = {}
        for m in sample_methods:
            method_counts[m] = method_counts.get(m, 0) + 1
        print(f"\n⚠️ 警告: 采样为混合方法，统计结果可能受不同后验近似方法的影响")
        print(f"   方法分布: {method_counts}")

print(f"\n最终采样方法: {ula_method}, 样本数: {len(all_samples)}")


# ========================================================================
# ★ 根据S动态推导置信水平（供Step3/4/5/总结统一使用）
# ========================================================================
# 经验分位数在样本中对应整数位置 k=1,2,...,S-1
# 双侧coverage level = 2k/S，候选分位点为 2*k/S (k=1..S//2)
# 过滤掉边界值1.0（对应[0%,100%]退化情况）和0（无意义）
# 候选数≤4时全部保留；候选数>4时按索引均匀采样4个，让校准曲线在可用水平上均匀分布
# 典型S的预期输出（运行下方自检可看到实际结果）:
#   S=8  -> [0.250, 0.500, 0.750]   (候选数=3, 全部保留)
#   S=16 -> [0.125, 0.375, 0.625, 0.875]   (候选数=7, 索引[0,2,4,6])
#   S=30 -> [0.067, 0.333, 0.667, 0.933]   (候选数=14, 索引[0,4,9,13])
#   S=64 -> [0.031, 0.344, 0.656, 0.969]   (候选数=31, 索引[0,10,20,30])
def _derive_confidence_levels(n_samples):
    """根据样本数S动态推导用于校准曲线绘制的经验分位数覆盖率列表。

    推导规则:
    1. 候选集 = {2k/S : k=1..S//2} ∩ (0, 1)   排除退化覆盖率
    2. 候选数 ≤ 4: 全部保留
    3. 候选数 > 4: 在候选索引[0, n-1]上均匀取4个点 (i*(n-1)/3, i=0..3)
    4. 候选数 = 0: 退化为单点中位数0.5 (S=1或S=2)

    关键点: 索引均匀 ≠ 值均匀 (因为候选本身在[2/S, (S-1)/S]上等距,
    所以索引均匀也意味着值近似均匀, 但精度受S的离散性影响)。
    """
    _all_levels = [2 * k / n_samples for k in range(1, n_samples // 2 + 1)
                   if 0 < 2 * k / n_samples < 1]
    if len(_all_levels) == 0:
        return [0.5]  # S=1或S=2的退化情况
    if len(_all_levels) <= 4:
        return _all_levels
    # 候选数>4: 按索引均匀采样4个, 覆盖[0, n-1]的完整范围
    n = len(_all_levels)
    indices = [int(round(i * (n - 1) / 3)) for i in range(4)]
    return sorted({_all_levels[i] for i in indices})

confidence_levels = _derive_confidence_levels(S)
# ★ cl_max: S可靠范围内最高置信水平, 贯穿Step3/4/5/JSON/总结
cl_max = max(confidence_levels)
print(f"[校准] 使用置信水平: {[round(cl, 3) for cl in confidence_levels]}, "
      f"主置信水平 cl_max={cl_max:.1%} (基于S={S}动态推导, 候选数={len([2*k/S for k in range(1, S//2 + 1) if 0 < 2*k/S < 1])})")

# ★ 自检: 打印几个典型S值, 方便读者直接验证推导公式的正确性
# (避免注释里写"算例"但实际跑出来对不上的问题)
print(f"[校准自检] 几个典型S值的confidence_levels推导结果(实际运行, 非手算):")
for _S_test in [4, 8, 16, 30, 64]:
    _cls = _derive_confidence_levels(_S_test)
    print(f"  S={_S_test:3d} -> {[round(c, 3) for c in _cls]}")


# ========================================================================
# Step 3: 不确定性地图计算
# 对应18.5节知识点：像素级标准差与置信区间
# ========================================================================
print("\n" + "="*70)
print("Step 3: 不确定性地图计算")
print("="*70)

# 将样本堆叠为张量
samples_tensor = torch.stack(all_samples, dim=0)  # (S, 1, 3, 256, 256)
print(f"样本张量 shape: {samples_tensor.shape}")

# 计算后验统计量
posterior_mean = samples_tensor.mean(dim=0)  # (1, 3, 256, 256)
posterior_std = samples_tensor.std(dim=0)    # (1, 3, 256, 256)
posterior_var = samples_tensor.var(dim=0)

# ★ 主置信区间（基于动态推导的 cl_max，而非硬编码95%）
# S=8 时 cl_max=0.75（即75%区间）；S≥16 时 cl_max 可能更大
# 校准检验、覆盖率统计、可视化均统一使用此区间，避免硬编码95%在S=8时几乎外推的误导
# ★★ 实现细节: torch.quantile 默认走线性插值（type=7, R 默认），
#   与 _derive_confidence_levels 中"分位数对应整数位置 k"的严格顺序统计量
#   推导在数值上接近但不等价；下文 Step4 覆盖率统计的"理论上应≈cl_max"是
#   渐近成立, S=8 时会有几个百分点的偏差, 不视为算法错误。
q_low_cl = samples_tensor.quantile((1 - cl_max) / 2, dim=0)
q_high_cl = samples_tensor.quantile(1 - (1 - cl_max) / 2, dim=0)
ci_width_cl = q_high_cl - q_low_cl

# 统计量
posterior_mean_psnr = compute_psnr(x_true, posterior_mean.to(device))
mean_std = posterior_std.mean().item()
max_std = posterior_std.max().item()
mean_ci_width_cl = ci_width_cl.mean().item()

print(f"后验均值 PSNR: {posterior_mean_psnr:.2f} dB")
print(f"平均像素标准差: {mean_std:.4f}")
print(f"最大像素标准差: {max_std:.4f}")
print(f"平均经验分位数区间宽度(cl={cl_max:.0%}, S={S}):  {mean_ci_width_cl:.4f}")
print(f"⚠️ 注意: S={S} 时区间估计不可靠，建议 S≥30")

# 各样本的PSNR分布
sample_psnrs = [compute_psnr(x_true, s.to(device)) for s in all_samples]
sample_psnr_mean = np.mean(sample_psnrs)  # ★ 样本 PSNR 的均值（直方图用）
print(f"样本PSNR范围: {min(sample_psnrs):.2f} - {max(sample_psnrs):.2f} dB")
print(f"样本PSNR标准差: {np.std(sample_psnrs):.2f} dB")

# ★★★ 运行时自动诊断：检测 ULA 链是否发散 ★★★
# 诊断信号：
#   1. max_std > 0.5: 像素标准差超过图像合法值域 [0,1] 的一半，说明样本间差异过大
#   2. 样本包含 NaN: 数值不稳定
#   3. 样本包含极端值 (abs > 3): 链走向了错误区域
#   4. PSNR < 10 dB: 重建质量过低（本配置下单样本预期≈12dB, 后验均值≈15.7dB）
DIAG_MAX_STD_THRESHOLD = 0.5  # 像素 std 阈值
DIAG_PSNR_MIN_THRESHOLD = 10.0  # PSNR 最低阈值（inpainting 任务）
has_nan = torch.isnan(samples_tensor).any().item()
has_extreme = (samples_tensor.abs() > 3).any().item()
psnr_too_low = min(sample_psnrs) < DIAG_PSNR_MIN_THRESHOLD
std_too_high = max_std > DIAG_MAX_STD_THRESHOLD

if std_too_high or has_nan or has_extreme or psnr_too_low:
    print("\n" + "⚠️" * 30)
    print("⚠️⚠️⚠️ [诊断警告] ULA 链可能已发散，结果不可信！")
    print("⚠️" * 30)
    if std_too_high:
        print(f"  - max_std={max_std:.4f} > {DIAG_MAX_STD_THRESHOLD} (超过图像值域阈值)")
    if has_nan:
        print(f"  - 样本包含 NaN（数值不稳定）")
    if has_extreme:
        print(f"  - 样本包含极端值（abs > 3，链走向错误区域）")
    if psnr_too_low:
        print(f"  - 最小PSNR={min(sample_psnrs):.1f}dB < {DIAG_PSNR_MIN_THRESHOLD}dB（重建质量过低）")
    print("\n建议操作：")
    print(f"  1. 减小 step_size_coeff_annealed（当前={step_size_coeff_annealed}，尝试 0.001 或更小）")
    print(f"  2. 检查 lambda_data 是否合理（当前={lambda_data}）")
    print(f"  3. 检查样本可视化图片，确认是否出现噪声/伪影")
    print("⚠️" * 30 + "\n")
else:
    print(f"[诊断] ✓ 样本质量正常（max_std={max_std:.4f}, PSNR范围=[{min(sample_psnrs):.1f}, {max(sample_psnrs):.1f}]dB）")

# ====== 按方法分组的后验统计量对比（实证验证）======
# 验证"DPS 与 ULA 的不确定性估计是否一致"的理论预期
# 注意：仅比较 ULA 和 DPS（两者构成严格后验采样），
#       PnP近似/伪逆+噪声不参与此验证（见下方说明）
VALID_POSTERIOR_METHODS = {"ULA", "退火ULA", "DPS"}  # 构成严格后验采样的方法
APPROXIMATE_METHODS = {"PnP近似", "伪逆+噪声"}  # 不构成严格后验的方法

# ★ PnP/伪逆不可比的原因说明（与最终总结的 _std_comparable_msg 共享逻辑）
_APPROX_METHOD_NOT_COMPARABLE = {
    "PnP近似": "缺少Langevin逐步加噪, std系统性偏小",
    "伪逆+噪声": "std几乎完全由初始化扰动决定, 与后验语义相去甚远"
}

unique_methods = list(set(sample_methods))
valid_methods = [m for m in unique_methods if m in VALID_POSTERIOR_METHODS]
approx_methods = [m for m in unique_methods if m in APPROXIMATE_METHODS]

if approx_methods:
    print(f"\n[实证验证] 检测到非严格后验方法: {approx_methods}")
    print(f"[实证验证] 这些方法的 posterior_std 不能用于验证'DPS与ULA物理含义一致'：")
    for method in approx_methods:
        print(f"  - {method}: {_APPROX_METHOD_NOT_COMPARABLE.get(method, '原因未知')}")

if len(valid_methods) >= 2:
    print("\n" + "="*60)
    print("★ 跨方法后验统计量对比（实证验证）")
    print("="*60)
    print("理论预期：若 DPS 与 ULA 的 sigma_data 语义一致，")
    print("          两者的 posterior_std 应在相近量级。")
    print("-"*60)
    for method in valid_methods:
        method_indices = [i for i, m in enumerate(sample_methods) if m == method]
        method_samples = [all_samples[i] for i in method_indices]
        if len(method_samples) > 1:
            method_tensor = torch.stack(method_samples, dim=0)
            method_std = method_tensor.std(dim=0).mean().item()
            method_psnrs = [sample_psnrs[i] for i in method_indices]
            print(f"  {method}: n={len(method_samples)}, "
                  f"平均std={method_std:.4f}, "
                  f"PSNR={np.mean(method_psnrs):.1f}±{np.std(method_psnrs):.2f}dB")
        else:
            print(f"  {method}: n=1（无法计算std）")
    print("-"*60)
    # 判断是否一致（仅比较有效后验方法）
    method_stds = []
    for method in valid_methods:
        method_indices = [i for i, m in enumerate(sample_methods) if m == method]
        method_samples = [all_samples[i] for i in method_indices]
        if len(method_samples) > 1:
            method_tensor = torch.stack(method_samples, dim=0)
            method_stds.append(method_tensor.std(dim=0).mean().item())
    if len(method_stds) >= 2:
        std_ratio = max(method_stds) / min(method_stds) if min(method_stds) > 0 else float('inf')
        if std_ratio < 2:
            print(f"✓ ULA vs DPS 的 posterior_std 比值={std_ratio:.1f}（<2），"
                  f"与'物理含义一致'的理论预期吻合")
        else:
            print(f"✗ ULA vs DPS 的 posterior_std 比值={std_ratio:.1f}（≥2），"
                  f"与'物理含义一致'的理论预期不符，需检查超参数")
elif len(valid_methods) == 1:
    print(f"\n[实证验证] 本次运行仅使用单一有效后验方法 ({valid_methods[0]})，"
          f"无法验证'ULA vs DPS 物理含义一致'的理论预期")
    print(f"[实证验证] 平均 posterior_std={mean_std:.4f}（仅作为该方法的参考值）")
else:
    print(f"\n[实证验证] 本次运行未使用有效后验方法（仅 {approx_methods}），"
          f"无法验证'DPS与ULA物理含义一致'的理论预期")

# 可视化: 后验样本 + 均值 + 不确定性
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 4, figure=fig)

# 第一行: 真值 + 4个后验样本
ax00 = fig.add_subplot(gs[0, 0])
ax00.imshow(x_true[0].cpu().permute(1, 2, 0).clamp(0, 1))
ax00.set_title('真值 x', fontsize=12)
ax00.axis('off')

for i in range(min(3, len(all_samples))):
    ax = fig.add_subplot(gs[0, i+1])
    ax.imshow(all_samples[i][0].cpu().permute(1, 2, 0).clamp(0, 1))
    ax.set_title(f'样本 {i+1}\nPSNR={sample_psnrs[i]:.1f}dB', fontsize=10)
    ax.axis('off')

# 第二行: 均值 + 观测 + 标准差地图 + CI宽度
ax10 = fig.add_subplot(gs[1, 0])
ax10.imshow(posterior_mean[0].cpu().permute(1, 2, 0).clamp(0, 1))
ax10.set_title(f'后验均值\nPSNR={posterior_mean_psnr:.1f}dB', fontsize=11)
ax10.axis('off')

ax11 = fig.add_subplot(gs[1, 1])
ax11.imshow(y_inp[0].cpu().permute(1, 2, 0).clamp(0, 1))
ax11.set_title('观测 y (50%修复)', fontsize=11)
ax11.axis('off')

ax12 = fig.add_subplot(gs[1, 2])
# 不确定性地图（灰度，越亮越不确定）
std_map = posterior_std[0].cpu().mean(dim=0).numpy()
# 使用99%分位数作为色彩上限，避免个别极值将颜色范围过度拉伸导致"噪声化"视觉效果
std_vmax = float(np.percentile(std_map, 99))
im12 = ax12.imshow(std_map, cmap='hot', vmin=0, vmax=std_vmax)
ax12.set_title('★ 不确定性地图\n(像素级std)', fontsize=11)
ax12.axis('off')
plt.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04)

ax13 = fig.add_subplot(gs[1, 3])
# ★ 修改：主图展示主置信区间（动态cl_max，S=8时为75%）
ci_map_cl = ci_width_cl[0].cpu().mean(dim=0).numpy()
# 同样使用99%分位数作为色彩上限，避免极值拉伸
ci_vmax = float(np.percentile(ci_map_cl, 99))
im13 = ax13.imshow(ci_map_cl, cmap='hot', vmin=0, vmax=ci_vmax)
ax13.set_title(f'{cl_max:.0%}经验分位数区间宽度\n(S={S}, 可靠范围)', fontsize=10)
ax13.axis('off')
plt.colorbar(im13, ax=ax13, fraction=0.046, pad=0.04)

# 第三行: 真值vs均值误差 + 覆盖图 + PSNR直方图 + 误差分布
ax20 = fig.add_subplot(gs[2, 0])
error_map = (x_true - posterior_mean.to(device)).abs()[0].cpu().mean(dim=0).numpy()
im20 = ax20.imshow(error_map, cmap='hot', vmin=0, vmax=error_map.max())
ax20.set_title('重建误差地图', fontsize=11)
ax20.axis('off')
plt.colorbar(im20, ax=ax20, fraction=0.046, pad=0.04)

# ★ 修改：覆盖率基于主置信区间（动态cl_max）
ax21 = fig.add_subplot(gs[2, 1])
in_ci_cl = ((x_true >= q_low_cl.to(device)) & (x_true <= q_high_cl.to(device))).float()
coverage_map = in_ci_cl[0].cpu().mean(dim=0).numpy()
im21 = ax21.imshow(coverage_map, cmap='RdYlGn', vmin=0, vmax=1)
overall_coverage = in_ci_cl.mean().item()
ax21.set_title(f'★ {cl_max:.0%}经验分位数覆盖图\n(S={S}, 可靠范围, 覆盖率={overall_coverage:.1%})', fontsize=10)
ax21.axis('off')
plt.colorbar(im21, ax=ax21, fraction=0.046, pad=0.04)

# PSNR直方图（★ 显示两条线：样本均值 + 后验均值）
ax22 = fig.add_subplot(gs[2, 2])
ax22.hist(sample_psnrs, bins=max(5, S//2), color='steelblue', edgecolor='white', alpha=0.8)
ax22.axvline(sample_psnr_mean, color='red', linestyle='--', linewidth=2, label=f'样本均值={sample_psnr_mean:.1f}dB')
ax22.axvline(posterior_mean_psnr, color='orange', linestyle=':', linewidth=2, label=f'后验均值={posterior_mean_psnr:.1f}dB')
ax22.set_xlabel('PSNR (dB)', fontsize=10)
ax22.set_ylabel('频次', fontsize=10)
ax22.set_title('样本PSNR分布', fontsize=11)
ax22.legend(fontsize=8, loc='upper right')

# 误差分布
ax23 = fig.add_subplot(gs[2, 3])
errors = (posterior_mean.to(device) - x_true).cpu().numpy().flatten()
ax23.hist(errors, bins=100, density=True, color='steelblue', alpha=0.7)
ax23.axvline(0, color='red', linestyle='--')
ax23.set_xlabel('误差值', fontsize=10)
ax23.set_ylabel('概率密度', fontsize=10)
ax23.set_title('重建误差分布', fontsize=11)

fig.suptitle('Step 1-3: 后验采样与不确定性量化 (50%修复场景)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step1_3_uncertainty_maps.png'), dpi=150, bbox_inches='tight')
plt.close()
print("已保存: step1_3_uncertainty_maps.png")


# ========================================================================
# Step 4: ★ 校准检验
# 对应18.5节知识点：校准检查与覆盖率分析
# ========================================================================
print("\n" + "="*70)
print("Step 4: ★ 校准检验")
print("="*70)

# ★ 共享的校准偏差等级阈值 (Step 4 的逐cl判断与最终总结使用同一套)
#   偏差 < 0.05  → 良好 (✓)
#   偏差 < 0.15  → 存在偏差 (△)
#   偏差 ≥ 0.15  → 严重失配 (✗)
_CALIB_GOOD_THRESH = 0.05
_CALIB_FAIR_THRESH = 0.15

def _calib_grade(dev):
    """根据覆盖率偏差 |coverage - cl| 返回 (等级文字, 标记符号)。"""
    if dev < _CALIB_GOOD_THRESH:
        return "良好", "✓"
    if dev < _CALIB_FAIR_THRESH:
        return "存在偏差", "△"
    return "严重失配", "✗"

print("""
★ 校准检验原理:
  如果后验分布p(x|y)是"正确校准"的，那么:
  - 经验分位数区间应该覆盖对应比例的真值像素
  - 覆盖率 ≈ 名义覆盖率 → 校准良好
  - 覆盖率 > 名义覆盖率 → 过于保守（区间太宽）
  - 覆盖率 < 名义覆盖率 → 过于自信（区间太窄）

  ⚠️ 注意: 本检验是"单图像×像素空间池化"(详见下方说明)
     要获得统计意义严格的95%置信区间校准，需要 S≥30 且多次独立实验
""")

print("""
★ 单图像×像素空间池化 vs 多次独立实验校准:
  本实验的"校准"做法: 在单张真值图像x上, 对每个像素构造经验分位数
  区间, 再把整图256×256×3个像素的覆盖结果pool起来, 统计"被覆盖比例"。
  这与统计教材里"频率学派置信区间覆盖率"(多个独立(x_i, y_i)各自采样、
  各自检查真值是否落入置信区间、统计覆盖频率)有重要区别:
    - 多次独立校准: 各实验独立, 可估计覆盖率方差和置信区间
    - 像素空间池化: 像素高度相关(邻居像素通常共享后验结构),
      pooled coverage的方差估计会失真, 但点估计仍有描述性参考价值
  这是图像UQ文献中的常见做法(逐像素校准曲线),
  但不宜与统计推断意义下的"覆盖率"直接等同。
""")

# ★ confidence_levels 已在 Step3 之前动态推导，此处直接使用
coverages = []

for cl in confidence_levels:
    # ★ 复用: cl_max 在 Step 3 算 overall_coverage 时已做过相同的
    # quantile + 区间内判断, 与本循环体内重新计算结果应逐位一致。
    # 跳过该分支避免重复计算, 防止以后改一处忘改另一处导致的不一致。
    if cl == cl_max:
        coverage = overall_coverage
    else:
        q_low = samples_tensor.quantile((1 - cl) / 2, dim=0).to(device)
        q_high = samples_tensor.quantile(1 - (1 - cl) / 2, dim=0).to(device)
        in_interval = ((x_true >= q_low) & (x_true <= q_high)).float()
        coverage = in_interval.mean().item()
    coverages.append(coverage)
    # ★ 使用共享的三档校准阈值, 与最终总结保持一致
    dev = abs(coverage - cl)
    grade, mark = _calib_grade(dev)
    print(f"  名义覆盖率 {cl:.0%}: 实际覆盖率 = {coverage:.1%}  {mark} 偏差={dev:.1%} ({grade})")

# 校准曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左: 校准曲线
axes[0].plot([0, 1], [0, 1], 'k--', label='完美校准', alpha=0.5)

# ★ 当只有一个置信水平时（S=4），绘制单点+标注；否则绘制曲线
if len(confidence_levels) == 1:
    # 单点情况：绘制点并添加说明
    axes[0].plot(confidence_levels[0], coverages[0], 'ro', markersize=12, label='实际校准')
    axes[0].annotate(f'({confidence_levels[0]:.0%}, {coverages[0]:.1%})',
                     xy=(confidence_levels[0], coverages[0]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=11, color='red',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='red'))
    # ★ 添加警告说明
    axes[0].text(0.5, 0.15,
                 f'⚠️ S={S}时仅能计算{len(confidence_levels)}个置信水平\n'
                 f'需要S≥8才能绘制完整校准曲线',
                 ha='center', va='center', fontsize=10, color='darkorange',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='darkorange'))
else:
    # 多个置信水平：绘制曲线
    axes[0].plot(confidence_levels, coverages, 'ro-', label='实际校准', markersize=8)

axes[0].set_xlabel('名义覆盖率', fontsize=12)
axes[0].set_ylabel('实际覆盖率', fontsize=12)
axes[0].set_title('★ 校准曲线', fontsize=13)
axes[0].legend(fontsize=10, loc='upper left')
axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)

# 右: 按像素强度的覆盖率分析
x_flat = x_true[0].cpu().numpy().flatten()
# ★ 修复：使用 in_ci_cl（主置信区间覆盖率）而非 in_ci_75
in_ci_flat = in_ci_cl[0].cpu().numpy().flatten()
# 按像素强度分组
# ★ 修复：用 np.digitize 替代手写 (>= ) & (<) 掩码，避免最后一个 bin 漏掉
#   强度恰好等于 1.0 的像素（CelebA 人像过曝高光区域常见）的问题。
#   digitize 默认右开 bins[i-1] <= x < bins[i]，恰好 1.0 落到索引 len(bins)
#   的越界位置，需要 clip 到 [0, n_bins-1]。
bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2
n_bins = len(bins) - 1
bin_idx = np.digitize(x_flat, bins) - 1  # 范围 [-1, n_bins]
bin_idx = np.clip(bin_idx, 0, n_bins - 1)  # 钳制到 [0, n_bins-1]
coverage_by_intensity = []
for i in range(n_bins):
    mask = bin_idx == i
    if mask.sum() > 0:
        coverage_by_intensity.append(in_ci_flat[mask].mean())
    else:
        coverage_by_intensity.append(np.nan)

axes[1].bar(bin_centers, coverage_by_intensity, width=0.08, color='steelblue', alpha=0.8)
axes[1].axhline(cl_max, color='red', linestyle='--', label=f'{cl_max:.0%}名义覆盖率')
axes[1].set_xlabel('像素强度', fontsize=12)
axes[1].set_ylabel('实际覆盖率', fontsize=12)
axes[1].set_title('★ 按像素强度的覆盖率', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

fig.suptitle('Step 4: ★ 校准检验', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step4_calibration.png'), dpi=150, bbox_inches='tight')
plt.close()
print("已保存: step4_calibration.png")


# ========================================================================
# Step 5: ★ 样本数对不确定性估计的影响
# 对应18.5节知识点：采样数量与不确定性估计精度
# ========================================================================
print("\n" + "="*70)
print("Step 5: ★ 样本数对不确定性估计的影响")
print("="*70)

print("""
★ 样本数对不确定性估计的影响:
  样本数 S 直接决定后验统计量的估计精度:
  - S过少: 标准差估计偏差大，置信区间不可靠
  - S适中(≥8): 均值和标准差趋于稳定
  - S较多(≥30): 高置信水平分位数也可可靠估计

本步骤对比: S=4 vs S={S} 样本数对不确定性的影响""".format(S=S))

# 减少样本数的不确定性对比
if len(all_samples) >= 4:
    # 用前4个样本
    samples_s4 = torch.stack(all_samples[:4], dim=0)
    mean_s4 = samples_s4.mean(dim=0)
    std_s4 = samples_s4.std(dim=0)
    psnr_s4 = compute_psnr(x_true, mean_s4.to(device))

    # 用全部样本
    psnr_full = compute_psnr(x_true, posterior_mean.to(device))

    print(f"\nS=4 采样:  PSNR={psnr_s4:.2f} dB, 平均std={std_s4.mean():.4f}")
    print(f"S={S} 采样: PSNR={psnr_full:.2f} dB, 平均std={posterior_std.mean():.4f}")

    # 样本数对不确定性的影响
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 上行: S=4
    axes[0, 0].imshow(mean_s4[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0, 0].set_title(f'S=4 后验均值\nPSNR={psnr_s4:.1f}dB', fontsize=11)
    axes[0, 0].axis('off')

    std_s4_map = std_s4[0].cpu().mean(dim=0).numpy()
    im01 = axes[0, 1].imshow(std_s4_map, cmap='hot', vmin=0, vmax=np.percentile(std_s4_map, 98))
    axes[0, 1].set_title(f'S=4 不确定性\n平均std={std_s4.mean():.4f}', fontsize=11)
    axes[0, 1].axis('off')
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046)

    # S=4 vs S=full 差异
    diff_std = (std_s4 - posterior_std).abs()[0].cpu().mean(dim=0).numpy()
    im02 = axes[0, 2].imshow(diff_std, cmap='hot', vmin=0, vmax=np.percentile(diff_std, 98))
    axes[0, 2].set_title(f'|std(S=4) - std(S={S})|', fontsize=11)
    axes[0, 2].axis('off')
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.046)

    # 下行: S=full
    axes[1, 0].imshow(posterior_mean[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[1, 0].set_title(f'S={S} 后验均值\nPSNR={psnr_full:.1f}dB', fontsize=11)
    axes[1, 0].axis('off')

    im11 = axes[1, 1].imshow(std_map, cmap='hot', vmin=0, vmax=np.percentile(std_map, 98))
    axes[1, 1].set_title(f'S={S} 不确定性\n平均std={posterior_std.mean():.4f}', fontsize=11)
    axes[1, 1].axis('off')
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046)

    # 样本数 vs PSNR收敛
    # ★ 注意: 如果为混合采样（如ULA+DPS），曲线跳变反映方法切换而非收敛
    if ula_method != "混合采样":
        # ★ 非混合采样：绘制PSNR收敛曲线
        psnr_by_s = []
        for n in range(1, len(all_samples) + 1):
            mean_n = torch.stack(all_samples[:n], dim=0).mean(dim=0)
            psnr_by_s.append(compute_psnr(x_true, mean_n.to(device)))

        axes[1, 2].plot(range(1, len(all_samples) + 1), psnr_by_s, 'bo-', markersize=6)
        axes[1, 2].set_xlabel('样本数 S', fontsize=11)
        axes[1, 2].set_ylabel('PSNR (dB)', fontsize=11)
        axes[1, 2].set_title(f'★ 后验均值PSNR vs 样本数', fontsize=11)
        axes[1, 2].grid(alpha=0.3)
    else:
        # ★ 混合采样：改为按方法分组的柱状图，避免混淆"样本数"和"方法质量"
        method_psnrs = {}
        for s_idx, (sample, method) in enumerate(zip(all_samples, sample_methods)):
            if method not in method_psnrs:
                method_psnrs[method] = []
            method_psnrs[method].append(compute_psnr(x_true, sample.to(device)))

        methods_list = list(method_psnrs.keys())
        mean_psnrs = [np.mean(method_psnrs[m]) for m in methods_list]
        std_psnrs = [np.std(method_psnrs[m]) for m in methods_list]

        bars = axes[1, 2].bar(range(len(methods_list)), mean_psnrs,
                              yerr=std_psnrs, capsize=5, color='steelblue', alpha=0.8)
        axes[1, 2].set_xticks(range(len(methods_list)))
        axes[1, 2].set_xticklabels(methods_list, rotation=15, ha='right', fontsize=9)
        axes[1, 2].set_ylabel('平均PSNR (dB)', fontsize=11)
        axes[1, 2].set_title('★ 按方法分组的PSNR\n(混合采样)', fontsize=11)
        axes[1, 2].grid(alpha=0.3, axis='y')

        # 在柱状图上标注样本数
        for i, method in enumerate(methods_list):
            count = len(method_psnrs[method])
            axes[1, 2].text(i, mean_psnrs[i] + std_psnrs[i] + 0.5, f'n={count}',
                           ha='center', va='bottom', fontsize=9)

    fig.suptitle('Step 5: ★ 样本数对不确定性估计的影响', fontsize=14)
    # ★ 混合采样警告：S=4 子集不一定具有方法同质性
    # 在"尝试-失败-降级"策略下，前 4 个样本可能全部来自 ULA、后 4 个降级
    # 为 DPS/PnP，导致 S=4 与 S=8 对比实际测量的是"方法差异"而非
    # "样本数效应"，与 PSNR 曲线分支采用同样的同质性提示。
    if ula_method == "混合采样":
        sample_methods_s4 = sample_methods[:4]
        if len(set(sample_methods_s4)) > 1:
            subset_mix_note = f"S=4 子集已混合: {dict((m, sample_methods_s4.count(m)) for m in set(sample_methods_s4))}"
        else:
            subset_mix_note = f"S=4 子集单方法 ({sample_methods_s4[0]}), 全量 S={S} 中混入其他方法"
        fig.text(0.5, -0.02,
                 f"⚠️ {subset_mix_note} — S=4 vs S=full 对比同时混合了'样本数效应'与'方法质量差异'，"
                 f"结论应保守",
                 ha='center', va='top', fontsize=9, color='darkorange', style='italic',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='darkorange'))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'step5_sample_size_effect.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: step5_sample_size_effect.png")
else:
    print("样本数不足4个，跳过样本数对比")

# ★ 不确定性地图按问题类型解读
print("""
★ 不确定性地图解读指南（对应18.5节）:

去模糊:
  - 均匀分布的不确定性 → 噪声主导
  - 边缘区域不确定性高 → 模糊核导致的结构模糊

超分辨率:
  - 高频细节区域不确定性高 → 细节丢失无法恢复
  - 平滑区域不确定性低 → 低频信息保留完好

修复:
  - 缺失区域不确定性高 → 无观测约束
  - 观测区域不确定性低 → 有直接约束
  - 缺失区域边缘 → 不确定性过渡带

通用规律:
  - 不确定性高 ↔ 信息丢失严重
  - 不确定性低 ↔ 有充分观测约束
  - 决策时: 高不确定性区域应谨慎对待
""")


# ========================================================================
# 保存数值结果
# ========================================================================
import json

# ★ 计算方法分布（用于混合采样情况）
method_distribution = {}
for m in sample_methods:
    method_distribution[m] = method_distribution.get(m, 0) + 1

# ★ 观测质量（评估任务难度）
obs_psnr = compute_psnr(x_true, y_inp)

# ★ 样本多样性指标
sample_psnr_std = float(np.std(sample_psnrs))

# ★ 校准状态评估（与Step 4共享_calib_grade）
_coverage_dev = abs(overall_coverage - cl_max)
_calib_grade_text, _ = _calib_grade(_coverage_dev)
_coverage_devs_all = {f"{cl:.4f}": abs(cov - cl) for cl, cov in zip(confidence_levels, coverages)}
_calib_grades_all = {f"{cl:.4f}": _calib_grade(abs(cov - cl))[0] for cl, cov in zip(confidence_levels, coverages)}

# ★ 数值稳定性信息（从n_diverged_steps提取）
# n_diverged_steps可能不在当前作用域，从sample_times中提取总耗时
total_time = sum(sample_times) if sample_times else 0.0
avg_time_per_sample = total_time / len(all_samples) if all_samples else 0.0

# ★ S=4对比结果（从Step 5提取）
s4_comparison = {}
if len(all_samples) >= 4:
    samples_s4 = torch.stack(all_samples[:4], dim=0)
    mean_s4 = samples_s4.mean(dim=0)
    std_s4 = samples_s4.std(dim=0)
    psnr_s4 = compute_psnr(x_true, mean_s4.to(device))
    s4_comparison = {
        'S4_PSNR': round(psnr_s4, 2),
        'S4_平均std': round(std_s4.mean().item(), 4),
        f'S4_vs_S{S}_PSNR差': round(posterior_mean_psnr - psnr_s4, 2),
        f'S4_vs_S{S}_std差': round(mean_std - std_s4.mean().item(), 4)
    }

# ★ 完整的ULA参数配置（实验复现必备）
ula_params = {
    'sigma_schedule': list(sigma_schedule),
    'ula_steps_each_sigma': int(ula_steps_each_sigma),
    'lambda_data': float(lambda_data),
    'step_size_coeff_annealed': float(step_size_coeff_annealed),
    'ula_noise_scale': float(ula_noise_scale),
    'ula_burn_in': int(ula_burn_in),
    'total_steps_per_sample': len(sigma_schedule) * ula_steps_each_sigma,
    'soft_projection_range': '(-0.1, 1.1)'
}

uq_results = {
    '采样方法': ula_method,
    '样本数': S,

    # ===== 实验配置 =====
    '图像尺寸': IMG_SIZE,
    '观测噪声sigma_data': float(sigma_data),
    'ULA参数配置': ula_params,

    # ===== 观测质量 =====
    '观测PSNR': round(obs_psnr, 2),

    # ===== 后验重建质量 =====
    '后验均值PSNR': round(posterior_mean_psnr, 2),
    '样本PSNR范围': [round(min(sample_psnrs), 2), round(max(sample_psnrs), 2)],
    '样本PSNR标准差': round(sample_psnr_std, 4),
    '样本PSNR均值': round(float(np.mean(sample_psnrs)), 2),

    # ===== 不确定性度量 =====
    '平均像素std': round(mean_std, 4),
    '最大像素std': round(max_std, 4),
    f'平均{cl_max:.0%}经验分位数区间宽度': round(ci_width_cl.mean().item(), 4),

    # ===== 校准检验 =====
    '主置信水平': round(cl_max, 4),
    '全部置信水平': [round(cl, 4) for cl in confidence_levels],
    f'{cl_max:.0%}经验分位数区间覆盖率': round(overall_coverage, 4),
    f'{cl_max:.0%}校准状态': _calib_grade_text,
    f'{cl_max:.0%}校准偏差': round(_coverage_dev, 4),
    '校准数据': {str(round(cl, 4)): round(cov, 4) for cl, cov in zip(confidence_levels, coverages)},
    '各置信水平校准偏差': _coverage_devs_all,
    '各置信水平校准状态': _calib_grades_all,

    # ===== 数值稳定性 =====
    '总采样耗时_秒': round(total_time, 1),
    '平均每样本耗时_秒': round(avg_time_per_sample, 2),

    # ===== 样本数影响对比 =====
    **s4_comparison,

    # ===== 方法分布 =====
    '方法分布': method_distribution
}
with open(os.path.join(SAVE_DIR, 'uq_results.json'), 'w', encoding='utf-8') as f:
    json.dump(uq_results, f, ensure_ascii=False, indent=2)
print("数值结果已保存: uq_results.json")


# ========================================================================
# ★ 动态生成总结结论（避免硬编码文字与实际计算结果脱节）
# ========================================================================

# 1. 校准状态（基于主置信区间覆盖率与名义值的偏差）
# ★ 使用与Step 4共享的 _calib_grade 函数, 保证两处判断一致
_coverage_dev = abs(overall_coverage - cl_max)
_calib_grade_text, _ = _calib_grade(_coverage_dev)
_calib_msg = (f"{cl_max:.0%}区间校准{_calib_grade_text}"
              f"（覆盖率={overall_coverage:.1%}，与名义值偏差{_coverage_dev:.1%}）")

# 2. PSNR收敛信息（仅在非混合采样且psnr_by_s已计算时使用）
if ula_method != "混合采样" and psnr_by_s is not None and len(psnr_by_s) > 1:
    _psnr_min = min(psnr_by_s)
    _psnr_max = max(psnr_by_s)
    _psnr_range = _psnr_max - _psnr_min
    if _psnr_range < 0.5:
        _convergence_msg = f"后验均值PSNR随S变化范围较小（{_psnr_min:.2f}~{_psnr_max:.2f} dB），已趋于稳定"
    elif psnr_by_s[-1] >= psnr_by_s[0]:
        _convergence_msg = f"后验均值PSNR随S大致递增（{_psnr_min:.2f}~{_psnr_max:.2f} dB）"
    else:
        _convergence_msg = f"后验均值PSNR随S非单调（{_psnr_min:.2f}~{_psnr_max:.2f} dB）"
elif ula_method == "混合采样":
    _convergence_msg = "混合采样下PSNR-vs-S曲线受方法切换影响，未单独分析S的影响"
else:
    _convergence_msg = f"样本数S={S}过少，未绘制PSNR-vs-S曲线"

# 3. 样本数建议（根据实际校准偏差和S大小动态调整）
if S >= 30 and _coverage_dev < 0.10:
    _sample_advice = f"S={S}已满足推荐阈值，校准检验较为可靠"
elif S >= 30:
    _sample_advice = f"S={S}已满足推荐阈值，但校准偏差较大（{_coverage_dev:.1%}），需检查采样方法"
elif S >= 8:
    _sample_advice = f"S={S}可获得基本不确定性估计，但S≥30才能可靠估计95%置信区间"
else:
    _sample_advice = f"S={S}远低于推荐阈值（建议S≥30以获得可靠的95%置信区间估计）"

# 4. ★ 不确定性数值可比性提示 (覆盖"纯降级方法"场景, 与"混合采样"警告互补)
#   原 Step 3/4/5 的可比性警告主要针对"混合采样" (ULA+DPS+PnP+伪逆 混合),
#   但学生也常使用纯 PnP 近似 或 纯伪逆+噪声 跑完整流程, 此时 posterior_std
#   同样不能与 ULA 的不确定性直接做绝对数值对比:
#     - PnP 近似: 20 步迭代是确定性梯度下降+去噪, 仅在初始化注入 0.05*randn,
#       缺少 Langevin 每步加噪, 不同"样本"可能收敛到彼此接近的点, std 系统性偏小;
#     - 伪逆+噪声: 各样本之间共享同一个伪逆解, 仅以 0.02*randn 扰动, std 几乎
#       完全由初始化噪声决定, 与"后验"语义相去甚远。
#   下面这段提示在最终总结中根据 ula_method 动态生成, 与上面的"混合采样"警告互补。
#   ★ 注意：PnP/伪逆不可比的原因与 Step 3 的 _APPROX_METHOD_NOT_COMPARABLE 保持一致
if ula_method in ("PnP近似", "伪逆+噪声"):
    _std_comparable_msg = (
        f"⚠️ 采样为纯降级方法 ({ula_method}), 该方法不构成严格后验采样 "
        f"({_APPROX_METHOD_NOT_COMPARABLE.get(ula_method, '原因未知')}), "
        f"posterior_std 的绝对数值不能与 ULA 的不确定性直接比较大小, "
        f"本结果仅作'方法可运行'演示"
    )
elif ula_method == "混合采样":
    _std_comparable_msg = (
        f"⚠️ 采样为混合方法 (方法分布: {method_distribution}), "
        f"posterior_std 同时受'样本数'与'方法质量'影响, "
        f"不宜与单一 ULA 跑出的 std 直接比较"
    )
elif ula_method == "DPS":
    _std_comparable_msg = (
        f"DPS 采样的 posterior_std 与 ULA 的 std 在物理含义上一致 "
        f"(均为后验样本统计量), 但 deepinv 的 DPS 默认未做论文推荐的归一化缩放, "
        f"在 sigma_data 较大时 DPS 可能会比 ULA 给出更保守的不确定性"
    )
else:  # ULA 或 None
    _std_comparable_msg = "后验采样来自单一方法 (ULA), posterior_std 数值可作为该方法的不确定性估计"


# ========================================================================
# 实验总结
# ========================================================================
print("\n" + "="*70)
print("实验18.5-1 总结")
print("="*70)
print(f"""
本实验对应18.5节核心知识点：

1. 从点估计到分布 ✓
   - 点估计只给一个解，无法量化可靠性
   - 后验分布p(x|y)描述所有与观测一致的解

2. 后验采样方法 ✓
   - PnP-ULA: Langevin动力学 + DRUNet去噪器
   - DPS: 扩散后验采样
   - 加噪PnP: 近似后备方案

3. 不确定性量化 ✓
   - 后验均值: 最优点估计
   - 像素级标准差: 不确定性地图
   - 经验分位数区间: 覆盖真值的概率范围（⚠️ S<30时仅供参考）

4. ★ 校准检验 ✓
   - 覆盖率 vs 名义覆盖率（⚠️ S<30时结果不准确）
   - 按像素强度的覆盖率分析
   - {_calib_msg}

5. ★ 样本数影响 ✓
   - 样本数S对不确定性估计的影响
   - {_convergence_msg}
   - {_sample_advice}
   - {_std_comparable_msg}

关键发现:
- 采样方法: {ula_method}
- 后验均值PSNR: {posterior_mean_psnr:.2f} dB
- {cl_max:.0%}CI覆盖率: {overall_coverage:.1%}

所有图像已保存至: {SAVE_DIR}
""")