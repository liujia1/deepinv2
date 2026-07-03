# -*- coding: utf-8 -*-
"""
实验13.3-4：deepinv框架 DPS vs DiffPIR 对比
对应章节：13.3.4节 隐空间优化（DiffPIR风格对比）

素材来源：实验13.6-步骤2+3

★ 原创设计：DPS vs DiffPIR vs DDRM算法对比
  使用deepinv库进行真实图像去模糊，FFHQ数据集 + NCSNpp + DRUNet

实验内容：
  - deepinv扩散后验采样：图像去模糊（DPS）
  - DPS vs DiffPIR 重建效果对比
  - 算法层面的理论差异分析

运行前提：需要GPU + deepinv库 + 预训练模型（自动下载）

安装deepinv：pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import warnings

# ====== 解决中文乱码的核心代码 ======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

import platform
from matplotlib.font_manager import FontManager
def _find_chinese_font():
    candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong'] if platform.system() == 'Windows' else ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import os as _os, re
    for f in fm.ttflist:
        for pat in ['cjk', 'wqy', 'noto.*cjk', 'simhei']:
            if re.search(pat, f.name.lower()):
                return f.name
    return None

_cn_font = _find_chinese_font()
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"[Font] 已检测到中文字体: {_cn_font}")
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()


# ============================================================
# 步骤1：deepinv扩散后验采样——图像去模糊（DPS）
# ============================================================
print("\n" + "=" * 60)
print("步骤1：deepinv扩散后验采样——图像去模糊（DPS）")
print("=" * 60)

try:
    import deepinv as dinv
    from deepinv.sampling import (
        PosteriorDiffusion,
        VarianceExplodingDiffusion,
        EulerSolver,
        DPSDataFidelity,
        DiffPIR,
    )
    HAS_DEEPINV = True
    print("deepinv库已安装")
except ImportError:
    HAS_DEEPINV = False
    print("deepinv库未安装，步骤将跳过")
    print("安装方法：pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv")

dps_psnr = None
diffpir_psnr = None
HAS_RUN = False

if HAS_DEEPINV:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    _MODELS_DIR = os.path.join(SAVE_DIR, "models")
    _NCSNPP_LOCAL = os.path.join(_MODELS_DIR, "edm-ffhq-64x64-uncond-ve.pt")
    _DRUNET_LOCAL = os.path.join(_MODELS_DIR, "drunet_deepinv_color_finetune_22k.pth")

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
            print(f"  -> 本地模型不存在({local_path})，将使用pretrained='download'")
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
                x_test = dinv.utils.load_url_image(
                    url="https://deepinv-data.s3.amazonaws.com/demo_images/face.png",
                    img_size=64
                ).to(device)
                print(f"测试图像加载成功: shape={x_test.shape}")
            except Exception as e:
                print(f"无法从网络加载测试图像: {e}")
                x_test = torch.rand(1, 3, 64, 64, device=device)

            y = physics(x_test)

            # ---- DPS ----
            dps_fidelity = DPSDataFidelity(denoiser=denoiser)
            sde = VarianceExplodingDiffusion(
                sigma_min=0.01,
                sigma_max=1346,
                device=device
            )
            n_steps = 250
            timesteps = torch.linspace(sde.T, 0, n_steps + 1, device=device)
            solver = EulerSolver(timesteps=timesteps)
            model_dps = PosteriorDiffusion(
                data_fidelity=dps_fidelity,
                denoiser=denoiser,
                sde=sde,
                solver=solver,
                device=device,
            )
            print("正在执行扩散后验采样（DPS）...")
            x_hat_dps, _ = model_dps(y, physics, seed=42, denoise_output=True)
            print(f"DPS重建: shape={x_hat_dps.shape}, min={x_hat_dps.min():.3f}, max={x_hat_dps.max():.3f}")

            mse = torch.mean((x_hat_dps - x_test)**2).item()
            dps_psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            print(f"DPS重建PSNR: {dps_psnr:.2f} dB")

            # ---- DiffPIR ----
            print("\n正在运行DiffPIR...")
            diffpir = DiffPIR(
                denoiser=_load_model_local(dinv.models.DRUNet, _DRUNET_LOCAL, "DRUNet").to(device),
                data_fidelity=dinv.optim.L2(),
            )
            x_hat_diffpir = diffpir(y, physics, seed=42)
            mse_p = torch.mean((x_hat_diffpir - x_test)**2).item()
            diffpir_psnr = 10 * np.log10(1.0 / (mse_p + 1e-10))
            print(f"DiffPIR重建PSNR: {diffpir_psnr:.2f} dB")
            HAS_RUN = True

            # ---- 可视化对比 ----
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            y_vis = y[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
            axes[0].imshow(y_vis)
            axes[0].set_title('观测 y=Ax+n\n（模糊+噪声）', fontsize=12)
            axes[0].axis('off')

            x_dps_vis = x_hat_dps[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
            axes[1].imshow(x_dps_vis)
            axes[1].set_title(f'DPS重建\nPSNR={dps_psnr:.2f} dB', fontsize=12)
            axes[1].axis('off')

            x_pir_vis = x_hat_diffpir[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
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
            print("可能原因：GPU内存不足、模型文件损坏等")
            print("请确保：1) 有GPU可用  2) models/目录下有正确的模型文件  3) 至少4GB GPU内存")
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
print(f"{'似然近似':<20s} | {'Laplacian近似':<25s} | {'交替优化':<25s}")
print(f"{'采样方式':<20s} | {'修正逆向SDE':<25s} | {'去噪-投影交替':<25s}")
print(f"{'参考论文':<20s} | {'Chung et al. 2022':<25s} | {'Zhu et al. 2023':<25s}")
print(f"{'得分模型':<20s} | {'NCSNpp (VE-SDE)':<25s} | {'DRUNet':<25s}")

print(f"\n{'='*60}")
print("实验13.3-4 完成!")
print("=" * 60)
if HAS_RUN and dps_psnr is not None:
    print(f"""
关键结论:
1. deepinv工程实现（13.3.4节）
   - DPS使用NCSNpp (VE-SDE) + DPSDataFidelity
   - DiffPIR使用DRUNet + L2数据保真项
   - 两者在同一测试集FFHQ上的PSNR对比：DPS={dps_psnr:.2f} dB vs DiffPIR={diffpir_psnr:.2f} dB

2. 算法差异
   - DPS: 在逆向SDE的每一步用autograd计算似然梯度，直接修正ε
   - DiffPIR: 在每步去噪后做一次数据一致性投影（PnP风格）

3. 选型建议
   - 想要理论清晰、与得分函数严格对应 → 选DPS
   - 想要PnP思想、模块化设计 → 选DiffPIR
""")
else:
    print("""
关键结论（理论对比，无实际运行结果）:
1. 核心思想差异
   - DPS: 在逆向SDE中注入似然梯度，理论对应13.2.2节的后验得分分解
   - DiffPIR: 在每步去噪后做数据一致性投影，PnP思想（第5章的延伸）

2. 似然近似方式
   - DPS: delta函数近似 p(x_0|x_t) ≈ δ(x_0 - x̂_{0|t})
   - DiffPIR: 通过HQS/HAL求解 min ||y-Ax||² + (1/2σ²)||x-x̂||²

3. 选型建议
   - 想要理论清晰、与得分函数严格对应 → 选DPS
   - 想要PnP思想、模块化设计 → 选DiffPIR
""")
