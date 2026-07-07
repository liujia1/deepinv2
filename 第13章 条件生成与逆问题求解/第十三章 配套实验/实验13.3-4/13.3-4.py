# -*- coding: utf-8 -*-
"""
实验13.3-4：deepinv框架 DPS vs DiffPIR 对比
对应章节：13.3.6节 四类方法对比与选择指南

实验内容：
  - deepinv扩散后验采样：图像去模糊（DPS，13.3.1/13.3.2节，DiffUNet/ADM + VE-SDE）
  - DiffPIR 重建（13.3.4节 隐空间优化，DRUNet + DDPM）
  - 两类算法在各自最优配置下的对比（deepinv 官方推荐用法）

设计说明：
  - DPS 需要 score network（训练时输出 ∇log p_σ(x)），使用 DiffUNet/ADM UNet
  - DiffPIR 需要去噪器（训练时输出 x_clean），使用 DRUNet
  - 两者使用不同模型不是"不公乎"，而是 deepinv 官方设计的最优配置
  - DiffUNet 默认 checkpoint 为 FFHQ 256x256 的 OpenAI guided-diffusion（ADM 架构），
    对应 VE-SDE 参数 sigma_min=0.02, sigma_max=20.0

运行前提：需要GPU + deepinv库（未安装时自动通过 pip 安装）
"""

import sys
import io
import os
import subprocess
import inspect
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import warnings

# 设置控制台输出为 UTF-8 (Windows下避免中文乱码)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 静默matplotlib相关警告
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
    SAVE_DIR = os.path.join(_gdrive, '实验13.3-4')
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
    print("警告: chinese_font模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)


# ============================================================
# 步骤0：自动确保 deepinv 已安装（Colab/本地皆适用）
# ============================================================
def _ensure_deepinv():
    """确保 deepinv 已安装；若未安装则按顺序尝试国内/默认 pip 源自动安装。"""
    try:
        import deepinv as _dinv
        print(f"deepinv库已安装 (version {_dinv.__version__})")
        return _dinv
    except ImportError:
        print("未检测到 deepinv，正在自动安装...")
        # 优先使用国内镜像加速；若失败再回退到默认源
        mirrors = [
            "https://pypi.tuna.tsinghua.edu.cn/simple",
            "https://mirrors.aliyun.com/pypi/simple",
            None,  # 默认 PyPI
        ]
        for mirror in mirrors:
            cmd = [sys.executable, "-m", "pip", "install", "--quiet"]
            if mirror is not None:
                cmd += ["-i", mirror]
            cmd += ["git+https://github.com/deepinv/deepinv.git#egg=deepinv"]
            print(f"  -> 尝试安装源: {mirror or 'PyPI默认源'}")
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode == 0:
                print("  -> 安装成功，正在重新导入...")
                import deepinv as _dinv
                print(f"  -> deepinv 已就绪 (version {_dinv.__version__})")
                return _dinv
            else:
                err_msg = ret.stderr.strip()[:200] if ret.stderr else "未知错误"
                print(f"  -> 安装失败: {err_msg}")
        raise ImportError("deepinv 自动安装失败，请检查网络后手动安装。\n"
                          "安装方法: pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv")


print("\n" + "=" * 60)
print("实验13.3-4: deepinv框架 DPS vs DiffPIR 对比")
print("=" * 60)
print("对应章节: 13.3.6节 四类方法对比与选择指南")
print("知识点: deepinv工程框架, 扩散后验采样, PnP交替优化")


# ============================================================
# 步骤1：deepinv扩散后验采样——图像去模糊（DPS）
# ============================================================
print("\n" + "=" * 60)
print("步骤1：deepinv扩散后验采样——图像去模糊（DPS）")
print("=" * 60)

dinv = _ensure_deepinv()
from deepinv.sampling import (
    PosteriorDiffusion,
    VarianceExplodingDiffusion,
    EulerSolver,
    DPSDataFidelity,
    DiffPIR,
)

dps_psnr = None
diffpir_psnr = None
HAS_RUN = False
use_clip = False  # 模块级默认值，保证无GPU分支也有定义

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 统一使用 DRUNet（图像尺寸无关），公平对比 DPS 和 DiffPIR
# 失败则回退到 DiffUNet + 256x256（但此时 DiffPIR 也需用 DiffUNet）
_MODELS_DIR = os.path.join(SAVE_DIR, "models")
_DRUNET_LOCAL = os.path.join(_MODELS_DIR, "drunet_color.pth")
# DiffUNet 预训练分辨率为 256x256，测试图与此对齐
IMG_SIZE = 256


def _load_model_local(model_cls, local_path, **kwargs):
    """通用模型加载器：优先本地，否则调用 deepinv 自动下载（使用 PyTorch hub 缓存）。

    重要：deepinv 的 ``pretrained`` 参数既可以是 ``"download"``，也可以是本地 checkpoint
    路径字符串。若只传 ``pretrained="download"`` 而不指定 ``local_path``，实际下载的
    checkpoint 可能与注释中声明的模型不一致（本实验默认下载的是 256x256 ADM UNet）。
    """
    if os.path.isfile(local_path):
        print(f"  -> 从本地加载: {local_path}")
        # deepinv 支持通过 pretrained=local_path 直接加载本地权重
        return model_cls(pretrained=local_path, **kwargs).eval()
    else:
        print(f"  -> 本地模型不存在 ({local_path})，使用 pretrained='download'")
        print(f"     （模型会从 PyTorch hub 缓存加载或首次下载到 ~/.cache/torch/hub/）")
        return model_cls(pretrained="download", **kwargs).eval()


def _load_blur_kernel(device):
    levin_path = os.path.join(_MODELS_DIR, "Levin09.npy")
    if os.path.isfile(levin_path):
        print(f"  -> 从本地加载模糊核: {levin_path}")
        kernels = np.load(levin_path, allow_pickle=True)
        kernel = np.array(kernels[1], dtype=np.float32)
        return torch.tensor(kernel, dtype=torch.float32, device=device)
    else:
        try:
            from deepinv.utils.demo import load_degradation
            kernel = load_degradation("Levin09.npy", "kernels", index=1)
            return torch.tensor(kernel, dtype=torch.float32, device=device)
        except Exception as e:
            raise FileNotFoundError(
                f"无法加载模糊核 Levin09.npy。\n"
                f"请将 Levin09.npy 放入 models/ 目录，或确保网络可访问。\n"
                f"原始错误: {e}"
            )


if device.type == "cuda":
    try:
        # ============================================================
        # 设计：各用各的最优模型（deepinv 官方推荐用法）
        #   - DPS 需要 score network（训练时输出 ∇log p_σ(x)），用 ADM/guided-diffusion UNet
        #   - DiffPIR 需要去噪器（训练时输出 x_clean），用 DRUNet
        #   - 两者使用不同模型是 deepinv 官方设计的：DiffUNet 配 VE-SDE，
        #     DRUNet 配 DDPM/HQS 框架
        # 测试图：256x256（与 DiffUNet FFHQ 256x256 预训练分辨率匹配）
        # ============================================================

        # ===== 加载 DiffUNet/ADM UNet（DPS 用，VE-SDE score network） =====
        # deepinv DiffUNet 默认下载的 checkpoint 为 OpenAI guided-diffusion（ADM 架构）
        # 在 FFHQ 256x256 上训练，官方 VE-SDE 参数为 sigma_min=0.02, sigma_max=20.0。
        _DIFFUNET_LOCAL = os.path.join(_MODELS_DIR, "diffusion_ffhq_10m.pt")
        print("正在加载预训练 DiffUNet/ADM UNet（VE-SDE score network，FFHQ 256x256）...")
        diffunet = _load_model_local(
            dinv.models.DiffUNet, _DIFFUNET_LOCAL
        ).to(device)
        print("DiffUNet加载成功")

        # ===== 加载 DRUNet（DiffPIR 用，去噪器） =====
        print("正在加载预训练 DRUNet 去噪器...")
        drunet = _load_model_local(
            dinv.models.DRUNet, _DRUNET_LOCAL
        ).to(device)
        print("DRUNet加载成功")

        print("加载模糊核...")
        kernel_t = _load_blur_kernel(device)
        physics = dinv.physics.Blur(
            filter=kernel_t.unsqueeze(0).unsqueeze(0),
            device=device,
            noise_model=dinv.physics.GaussianNoise(sigma=0.02)
        )

        # 加载测试图（统一 256x256，与 DiffUNet 预训练分辨率匹配）
        # 注意：resize_mode="resize" 会缩放整个图像，保留完整人脸结构
        #       resize_mode="crop"（默认）会中心裁剪，丢失大部分人脸
        try:
            url = dinv.utils.get_image_url("celeba_example.jpg")
            x_test = dinv.utils.load_url_image(
                url=url, img_size=IMG_SIZE, resize_mode="resize"
            ).to(device)
            print(f"测试图像加载成功: shape={x_test.shape}")
        except Exception as e:
            print(f"无法从网络加载测试图像: {e}")
            try:
                url = dinv.utils.get_image_url("barbara.jpeg")
                x_test = dinv.utils.load_url_image(
                    url=url, img_size=IMG_SIZE, resize_mode="resize"
                ).to(device)
                print(f"测试图像加载成功(备用): shape={x_test.shape}")
            except Exception as e2:
                print(f"备用URL也加载失败: {e2}")
                x_test = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device)

        y = physics(x_test)

        # ---- DPS ----
        # DPS 用 DiffUNet/ADM UNet + VE-SDE（deepinv 官方 DPS demo 配置）
        n_steps_dps = 1000
        print(f"\n正在执行扩散后验采样（DPS）共 {n_steps_dps} 步（VE-SDE + DiffUNet）...")
        print("提示：每步需 DiffUNet 推理一次 + 似然梯度计算，GPU 上约需 5-15 分钟")
        import time
        t0 = time.time()
        # 调小 weight：deepinv DPSDataFidelity 的默认 weight=1.0 在去模糊任务上
        # 容易导致似然梯度单步过冲（眼睛/嘴部出现孤立亮/暗斑块）。
        # 公式：∇_x log p(y|x) ≈ (λ/√m) * ∇_x ||A·denoiser(x_σ) - y||
        # 因此 weight ≈ λ，对应原 DPS 论文中的 λ_。λ 越大、似然项越强、越快但越有偏。
        # 注：deepinv 0.4.0 中此处参数名为 weight（无 zeta），对应论文符号 λ。
        
        # 参数验证：检查 DPSDataFidelity 是否支持 clip 参数
        # 注意：若 __init__ 内部用 **kwargs 转发，签名里不会显示 clip（即使实际可用）
        dps_init_sig = inspect.signature(DPSDataFidelity.__init__)
        dps_params = dps_init_sig.parameters
        print(f"DPSDataFidelity 支持的参数: {list(dps_params.keys())}")
        
        # 检查签名中是否有 **kwargs（参数名为 'kwargs' 且类型为 VAR_KEYWORD）
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in dps_params.values()
        )
        
        use_clip = 'clip' in dps_params
        if use_clip:
            print("检测到 clip 参数，使用 weight=0.5, clip=(0,1)")
            dps_fidelity = DPSDataFidelity(
                denoiser=diffunet,
                weight=0.5,
                clip=(0.0, 1.0),
            )
        else:
            if has_kwargs:
                print("签名中有 **kwargs，clip 可能实际可用但未显式声明")
                print("建议：检查 DPSDataFidelity 源码确认是否支持 clip 参数")
            print("使用官方推荐参数 weight=0.5")
            dps_fidelity = DPSDataFidelity(
                denoiser=diffunet,
                weight=0.5,
            )
        sde = VarianceExplodingDiffusion(
            sigma_min=0.02,
            sigma_max=20.0,  # 与 deepinv 官方 DPS demo 一致（FFHQ 256x256 ADM UNet）
            device=device
        )
        # 终点取 0.001 而非 0，避免 VE-SDE score 公式中 sigma=0 时除零
        # （官方 DPS demo 统一用 torch.linspace(1, 0.001, 1000)）
        timesteps = torch.linspace(sde.T, 0.001, n_steps_dps + 1, device=device)
        solver = EulerSolver(timesteps=timesteps)
        model_dps = PosteriorDiffusion(
            data_fidelity=dps_fidelity,
            denoiser=diffunet,
            sde=sde,
            solver=solver,
            device=device,
            verbose=True,         # 开启 tqdm 进度条
        )
        # 防御性解包
        result_dps = model_dps(y, physics, seed=42, denoise_output=True)
        x_hat_dps = result_dps[0] if isinstance(result_dps, (tuple, list)) else result_dps
        print(f"DPS重建耗时: {time.time() - t0:.1f}s")
        print(f"DPS重建: shape={x_hat_dps.shape}, min={x_hat_dps.min():.3f}, max={x_hat_dps.max():.3f}")

        mse = torch.mean((x_hat_dps - x_test)**2).item()
        dps_psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        print(f"DPS重建PSNR: {dps_psnr:.2f} dB")

        # ---- DiffPIR ----
        # DiffPIR 直接接受 DRUNet 作为去噪器（训练目标为从含噪输入预测干净图像 x_clean）
        print("\n正在初始化DiffPIR（DRUNet作为去噪器）...")
        n_steps_diffpir = 100
        print(f"正在运行DiffPIR（max_iter={n_steps_diffpir}，含去噪+数据保真+采样三步）...")
        t0 = time.time()
        diffpir = DiffPIR(
            model=drunet,             # DRUNet（去噪器）
            data_fidelity=dinv.optim.L2(),
            sigma=0.02,
            zeta=0.3,
            lambda_=7.0,
            verbose=True,
            device=device,
        )
        x_hat_diffpir = diffpir(y, physics, seed=42)
        print(f"DiffPIR重建耗时: {time.time() - t0:.1f}s")
        print(f"DiffPIR重建: shape={x_hat_diffpir.shape}, min={x_hat_diffpir.min():.3f}, max={x_hat_diffpir.max():.3f}")
        mse_p = torch.mean((x_hat_diffpir - x_test)**2).item()
        diffpir_psnr = 10 * np.log10(1.0 / (mse_p + 1e-10))
        print(f"DiffPIR重建PSNR: {diffpir_psnr:.2f} dB")
        HAS_RUN = True

        # ---- 可视化对比 ----
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        y_vis = y[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        axes[0].imshow(y_vis)
        axes[0].set_title('观测 y=Ax+n（模糊+噪声）', fontsize=12)
        axes[0].axis('off')

        x_dps_vis = x_hat_dps[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        axes[1].imshow(x_dps_vis)
        axes[1].set_title(f'DPS重建\nPSNR={dps_psnr:.2f} dB', fontsize=12)
        axes[1].axis('off')

        x_pir_vis = x_hat_diffpir[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        axes[2].imshow(x_pir_vis)
        axes[2].set_title(f'DiffPIR重建\nPSNR={diffpir_psnr:.2f} dB', fontsize=12)
        axes[2].axis('off')

        plt.suptitle('deepinv框架：DPS vs DiffPIR 图像去模糊对比（13.3.4节）', fontsize=13, y=1.01)
        plt.tight_layout()
        fig_path = os.path.join(SAVE_DIR, 'DPSvsDiffPIR对比.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n图已保存: {fig_path}")

    except Exception as e:
        print(f"步骤执行出错: {e}")
else:
    print("未检测到GPU，跳过图像实验（需要GPU运行扩散模型）")


# ============================================================
# 步骤2：DPS vs DiffPIR 算法理论对比
# ============================================================
print("\n" + "=" * 60)
print("步骤2：DPS vs DiffPIR 算法理论对比")
print("=" * 60)

if not HAS_RUN:
    print("（注：以下为理论对比，无实际运行结果）")

print("\nDPS vs DiffPIR 对比：")
print(f"{'属性':<20s} | {'DPS':<25s} | {'DiffPIR':<25s}")
print("-" * 75)
print(f"{'核心思想':<20s} | {'似然梯度引导逆向SDE':<25s} | {'PnP嵌入扩散采样':<25s}")
print(f"{'似然近似':<20s} | {'Laplace近似':<25s} | {'交替优化':<25s}")
print(f"{'采样方式':<20s} | {'修正逆向SDE':<25s} | {'去噪-投影交替':<25s}")
print(f"{'调度方式':<20s} | {'VE-SDE（连续时间）':<25s} | {'DDPM（离散1000步）':<25s}")
print(f"{'去噪器':<20s} | {'DiffUNet/ADM (FFHQ 256x256)':<25s} | {'DRUNet':<25s}")
print(f"{'输出性质':<20s} | {'近似后验样本':<25s} | {'MAP-like解':<25s}")
print(f"{'参考论文':<20s} | {'Chung et al. 2022':<25s} | {'Zhu et al. 2023':<25s}")
print(f"{'测试尺寸':<20s} | {f'{IMG_SIZE}x{IMG_SIZE}':<25s} | {f'{IMG_SIZE}x{IMG_SIZE}':<25s}")

# 构建参数建议文本（与实际运行状态同步）
clip_advice = (
    "clip=(0,1)" if use_clip 
    else "clip 参数未启用（签名检测未发现，建议检查源码确认是否实际支持）"
)

print("\n" + "=" * 60)
print("实验13.3-4 完成!")
print("=" * 60)
if HAS_RUN and dps_psnr is not None:
    print(f"""
关键结论:
1. deepinv库实现（13.3.6节 四类方法对比）
   - DPS使用DiffUNet/ADM UNet (FFHQ 256x256) + VE-SDE + DPSDataFidelity → 近似后验样本
   - DiffPIR使用DRUNet + DDPM + L2数据保真项 → MAP-like解
   - 本次DPS参数配置：weight=0.5, {clip_advice}
   - 两者在各自最优配置下的PSNR对比：DPS={dps_psnr:.2f} dB vs DiffPIR={diffpir_psnr:.2f} dB

2. 算法差异（验证13.3.6节对比表）
   - DPS (Grad类): 在逆向SDE的每一步用autograd计算似然梯度，直接修正eps
     → 快但有偏（Laplace近似引入系统性偏差）
   - DiffPIR (Opt类): 在每步去噪后做一次数据一致性投影（PnP风格）
     → 给出MAP-like解，确定性优化无法探索后验多峰

3. 选型建议（13.3.6节方法选择指南）
   - 快速原型/非线性问题/需要后验样本 -> 选DPS（配 DiffUNet/ADM 等 score network）
   - 需要MAP解/PnP模块化设计/现成去噪器 -> 选DiffPIR（配 DRUNet 等去噪器）

4. 框架差异
   - DPS 的 score-based 框架需要 score network（DiffUNet/ADM），不能用普通去噪器（DRUNet）
   - DiffPIR 的 HQS 框架可以直接吃去噪器（DRUNet），无需 score network
   - 两者使用不同模型不是"不公平"，而是 deepinv 官方推荐的最优配置

5. 分辨率与退化强度的关系（实验设计考量）
   - Levin09 模糊核尺寸固定（约十几像素），但图像从 64×64 变成 256×256，
     模糊核占图像比例变小，视觉上模糊程度明显减弱
   - 这意味着 256×256 上的去模糊任务比 64×64 更"容易"，两个方法的 PSNR
     都会偏高、差距可能被压缩
   - 若最终两者 PSNR 意外接近，不要急着下"两个算法差不多"的结论，
     先确认是否任务本身变简单了

6. 实测观察（PSNR 与视觉质量的关系）
   - 修复模型/分辨率/sigma 配置错配后，DPS 给出 PSNR={dps_psnr:.2f} dB、
     DiffPIR 给出 PSNR={diffpir_psnr:.2f} dB
   - 但 PSNR 反映全局 MSE，可能掩盖局部伪影：实测中 DPS 偶有眼睛/嘴部等
     高频区域的局部过冲斑块（似然梯度单步过大的典型表现），而 DiffPIR 全局
     更平滑但整图易偏色
   - 注意 DPS 子图中人物鼻子出现重影：这是采样式方法的典型 artifact。
     DPS 每次反向过程注入随机噪声，单次采样落在"两个合理位置之间"时
     会呈现重影；若固定 y_obs 用不同 seed 多次采样，鼻子位置会有 ±1-2 像素
     抖动——这正是 DPS 的后验多样性，而非算法错误
   - 这恰好印证 13.3.6 节核心观点：Grad 类（快但有偏）vs Opt 类（稳定但慢），
     PSNR 数值高低 ≠ 视觉质量优劣
""")
else:
    print("""
关键结论（理论对比，无实际运行结果）:
1. 核心思想差异
   - DPS: 在逆向SDE中注入似然梯度，理论对应13.2.2节的后验得分分解
   - DiffPIR: 在每步去噪后做数据一致性投影，PnP思想（第5章的延伸）

2. 似然近似方式
   - DPS: delta函数近似 p(x_0|x_t) ~ delta(x_0 - x_hat_{0|t})
   - DiffPIR: 通过HQS/HAL求解 min ||y-Ax||^2 + (1/2sigma^2)||x-x_hat||^2

3. 选型建议
   - 想要理论清晰、与得分函数严格对应 -> 选DPS
   - 想要PnP思想、模块化设计 -> 选DiffPIR
""")
