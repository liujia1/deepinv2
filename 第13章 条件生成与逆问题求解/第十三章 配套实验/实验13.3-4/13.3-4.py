# -*- coding: utf-8 -*-
"""
实验13.3-4：deepinv框架 DPS vs DiffPIR 对比
对应章节：13.3.4节 隐空间优化

实验内容：
  - deepinv扩散后验采样：图像去模糊（DPS）
  - DPS vs DiffPIR 重建效果对比
  - 算法层面的理论差异分析

运行前提：需要GPU + deepinv库（未安装时自动通过 pip 安装）
"""

import sys
import io
import os
import subprocess
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
print("对应章节: 13.3.4节 隐空间优化")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

_MODELS_DIR = os.path.join(SAVE_DIR, "models")
_NCSNPP_LOCAL = os.path.join(_MODELS_DIR, "edm-ffhq-64x64-uncond-ve.pt")


def _load_model_local(model_cls, local_path, model_name, **kwargs):
    if os.path.isfile(local_path):
        print(f"  -> 从本地加载: {local_path}")
        model = model_cls(pretrained=None, **kwargs)
        ckpt = torch.load(local_path, map_location=lambda storage, loc: storage, weights_only=False)
        model.load_state_dict(ckpt, strict=True)
        if model_name == "NCSNpp":
            model.precondition_type = "edm"
            model.pixel_std = 0.5
            model._was_trained_on_minus_one_one = True
        model.eval()
        return model
    else:
        print(f"  -> 本地模型不存在 ({local_path})")
        # 显式触发下载并显示进度
        # 解析模型对应的权重 URL（NCSNpp 默认 'edm-ffhq-64x64-uncond-ve'）
        from deepinv.models.utils import get_weights_url
        if model_name == "NCSNpp":
            url_name = "edm-ffhq-64x64-uncond-ve.pt"
        elif model_name == "DiffUNet":
            url_name = "diffusion_ffhq_10m.pt"
        else:
            url_name = None
        if url_name is not None:
            url = get_weights_url(model_name="edm", file_name=url_name)
            print(f"  -> 正在从 HuggingFace 下载预训练权重: {url}")
            print(f"  -> 文件大小约 500MB，请耐心等待（首次下载约需 1-3 分钟）...")
            # 用 tqdm 显示下载进度
            try:
                from tqdm import tqdm
                import urllib.request
                cache_root = torch.hub.get_dir()
                os.makedirs(cache_root, exist_ok=True)
                target = os.path.join(cache_root, url_name)
                if os.path.isfile(target):
                    print(f"  -> 已存在缓存: {target}")
                else:
                    with tqdm(unit="B", unit_scale=True, miniters=1, desc=url_name) as t:
                        def reporthook(block_num, block_size, total_size):
                            if total_size > 0:
                                t.total = total_size
                                t.update(block_size - t.n)
                        urllib.request.urlretrieve(url, target, reporthook=reporthook)
                    print(f"  -> 下载完成: {target}")
            except Exception as e:
                print(f"  -> 进度条下载失败 ({e})，回退到默认方式")
        # 走 deepinv 内部加载（此时会复用上面的缓存，不再重复下载）
        return model_cls(pretrained="download", **kwargs)


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
        print("正在加载预训练模型NCSNpp（VE-SDE得分网络，FFHQ 64x64）...")
        denoiser = _load_model_local(dinv.models.NCSNpp, _NCSNPP_LOCAL, "NCSNpp").to(device)
        print("模型加载成功")

        print("加载模糊核...")
        kernel_t = _load_blur_kernel(device)
        physics = dinv.physics.Blur(
            filter=kernel_t.unsqueeze(0).unsqueeze(0),
            device=device,
            noise_model=dinv.physics.GaussianNoise(sigma=0.02)
        )

        try:
            # 使用 deepinv 内置的 get_image_url 获取有效 URL
            # DiffPIR 官方 demo 建议使用 256x256 图片（DiffUNet 预训练权重基于 256x256 FFHQ）
            url = dinv.utils.get_image_url("celeba_example.jpg")
            x_test = dinv.utils.load_url_image(url=url, img_size=256).to(device)
            print(f"测试图像加载成功: shape={x_test.shape}")
        except Exception as e:
            print(f"无法从网络加载测试图像: {e}")
            try:
                # 尝试第二候选 URL
                url = dinv.utils.get_image_url("barbara.jpeg")
                x_test = dinv.utils.load_url_image(url=url, img_size=256).to(device)
                print(f"测试图像加载成功(备用): shape={x_test.shape}")
            except Exception as e2:
                print(f"备用URL也加载失败: {e2}")
                x_test = torch.rand(1, 3, 256, 256, device=device)

        y = physics(x_test)

        # ---- DPS ----
        n_steps_dps = 250
        print(f"正在执行扩散后验采样（DPS）共 {n_steps_dps} 步（VE-SDE 逆向积分 + 似然梯度修正）...")
        print("提示：每步需 NCSNpp 推理一次 + 似然梯度计算，GPU 上约需 1-3 分钟")
        # 计时辅助
        import time
        t0 = time.time()
        dps_fidelity = DPSDataFidelity(denoiser=denoiser)
        sde = VarianceExplodingDiffusion(
            sigma_min=0.01,
            sigma_max=1346,
            device=device
        )
        timesteps = torch.linspace(sde.T, 0, n_steps_dps + 1, device=device)
        solver = EulerSolver(timesteps=timesteps)
        model_dps = PosteriorDiffusion(
            data_fidelity=dps_fidelity,
            denoiser=denoiser,
            sde=sde,
            solver=solver,
            device=device,
            verbose=True,         # 开启 tqdm 进度条
        )
        x_hat_dps = model_dps(y, physics, seed=42, denoise_output=True)
        print(f"DPS重建耗时: {time.time() - t0:.1f}s")
        print(f"DPS重建: shape={x_hat_dps.shape}, min={x_hat_dps.min():.3f}, max={x_hat_dps.max():.3f}")

        mse = torch.mean((x_hat_dps - x_test)**2).item()
        dps_psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        print(f"DPS重建PSNR: {dps_psnr:.2f} dB")

        # ---- DiffPIR ----
        # DiffPIR 使用 DiffUNet 作为去噪器（与官方 demo_diffpir.py 一致）
        # 预训练权重基于 256x256 FFHQ 训练；与 DPS 使用的 NCSNpp 风格不同（DDPM vs VE-SDE）
        print("\n正在加载预训练DiffUNet去噪模型...")
        diffpir_model = dinv.models.DiffUNet(pretrained="download").to(device)
        print("DiffUNet预训练模型加载成功")

        # DiffPIR 默认 max_iter=100；增加 max_iter 需权衡速度
        n_steps_diffpir = 100
        print(f"正在运行DiffPIR（max_iter={n_steps_diffpir}，含去噪+数据保真+采样三步）...")
        t0 = time.time()
        diffpir = DiffPIR(
            model=diffpir_model,
            data_fidelity=dinv.optim.L2(),
            sigma=0.02,
            zeta=0.3,           # 采样步长权重（与官方 demo 一致）
            lambda_=7.0,        # 数据保真项权重（官方推荐范围 3-25）
            verbose=True,       # 开启 tqdm 进度条
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
        import traceback
        print(f"步骤执行出错: {e}")
        traceback.print_exc()
        print("如果以上报错来自 deepinv API 调用本身，请检查 GPU/模型文件；"
              "如果是 NameError/AttributeError 等，通常是代码本身的问题，请先检查拼写和导入。")
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
print(f"{'参考论文':<20s} | {'Chung et al. 2022':<25s} | {'Zhu et al. 2023':<25s}")
print(f"{'得分模型':<20s} | {'NCSNpp (VE-SDE)':<25s} | {'DiffUNet (DDPM)':<25s}")

print("\n" + "=" * 60)
print("实验13.3-4 完成!")
print("=" * 60)
if HAS_RUN and dps_psnr is not None:
    print(f"""
关键结论:
1. deepinv工程实现（13.3.4节）
   - DPS使用NCSNpp (VE-SDE) + DPSDataFidelity
   - DiffPIR使用DiffUNet (DDPM) + L2数据保真项
   - 两者在同一测试集FFHQ上的PSNR对比：DPS={dps_psnr:.2f} dB vs DiffPIR={diffpir_psnr:.2f} dB

2. 算法差异
   - DPS: 在逆向SDE的每一步用autograd计算似然梯度，直接修正eps
   - DiffPIR: 在每步去噪后做一次数据一致性投影（PnP风格）

3. 选型建议
   - 想要理论清晰、与得分函数严格对应 -> 选DPS
   - 想要PnP思想、模块化设计 -> 选DiffPIR
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
