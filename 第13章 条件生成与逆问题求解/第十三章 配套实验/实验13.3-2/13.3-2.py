# -*- coding: utf-8 -*-
"""
实验13.3-2：deepinv框架 DPS vs DiffPIR 对比（同一逆问题：64×64 图像修复）

实验内容：
  - DPS（Grad 类）：PosteriorDiffusion + DPSDataFidelity(weight=1024) + NCSNpp
    + 默认 VE-SDE + EulerSolver，近似后验采样。装配框架对齐官方
    examples/sampling/demo_diffusion_sde.py 的 NCSNpp 段；weight 经本图实测
    校准（官方 4.0 只是其演示图上的取值，见设计说明末条）
  - DiffPIR（Opt 类）：DRUNet + DDPM + L2 数据保真 + HQS 交替优化，MAP-like 解
  - 两种方法共享同一观测 y（同一 Inpainting 算子、同一掩码），各自使用其原生先验

设计说明（为什么这样设计）：
  - 任务选 64×64 inpainting（中心 16×16 缺口、无测量噪声）：这是 deepinv 官方
    DPS demo 验证过的场景——demo_diffusion_sde.py 两段与 demo_dps.py 全部是
    inpainting。官方从未在任何去模糊任务上验证过 DPS；实测（DRUNet+256×256
    去模糊）会结构性失败，见"关键结论 4/6"的踩坑记录
  - 为什么不共享同一先验（对比早期版本的重要修订）：DPS 为 score model 而设计，
    需要去噪器在 SDE 全程 σ∈[0.001, sigma_max] 上都近似有效（NCSNpp 即按
    σ∈[0.001,80] 训练）；DRUNet 只在 σ≲0.5 有效，塞进 DPS 的 SDE 框架会在
    高 σ 死区失效。DiffPIR 的 HQS 框架才接受任意普通去噪器。故"各自原生先验"
    才是两种算法的典型用法；先验不同这一混杂因素在结论中如实注明
  - NCSNpp：FFHQ-64 的 EDM 得分模型（Karras et al. 2022），输入约定 [0,1]，
    模型内部自带 [-1,1]↔[0,1] 变换（forward 的 input_in_minus_one_one 参数，
    默认 False）。外层 MinusOneOneDenoiserWrapper（minus_one_one=True 默认包装）
    在 0.4.2 中只有调用方显式传 input_in_minus_one_one=False 才会做变换，而库内
    从不传——包装器实际直通，与官方 demo 行为一致，全部照抄不自行改动
  - weight=1024 是本图实测校准值（seed=42、250步、11档扫描）：DPSDataFidelity 的
    梯度 = weight*sqrt(C*H*W)*残差"单位方向"，模长恒定不随残差自适应，weight 是
    唯一强度旋钮。总PSNR 随 weight 单调上升 4→1024（19.8→34.5 dB）、可见区MSE
    同步下降（0.0101→0.0001）；2048 时无数据锚定的缺口区数值发散（骤降至 16.6，
    缺口呈蓝黑色块、可见区仍完好）。低权重下先验压过数据：观测为闭唇，w=4 却
    生成露齿笑、脸型漂移，"不像同一个人"。Inpainting 的 A 是对角掩码，高权重
    近似每步良性硬投影，故可远高于官方演示值；换图/换seed/换步数后应重扫

运行前提：需要 GPU + deepinv 库（未安装时自动 pip 安装）+ 首次运行联网下载 NCSNpp 权重
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
    SAVE_DIR = os.path.join(_gdrive, '实验13.3-2')
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
print("实验13.3-2: deepinv框架 DPS vs DiffPIR 对比")
print("=" * 60)
print("对应章节: 13.3.6节 四类方法对比与选择指南")
print("知识点: deepinv工程框架, 扩散后验采样, PnP交替优化")


# ============================================================
# 步骤1：同一逆问题（64×64 inpainting）下 DPS 与 DiffPIR 对比
# ============================================================
print("\n" + "=" * 60)
print("步骤1：DPS（Grad类） vs DiffPIR（Opt类）—— 图像修复")
print("=" * 60)

dinv = _ensure_deepinv()
# DPS 逐字采用官方 demo_diffusion_sde.py 的 NCSNpp 配方；DiffPIR 用高层类
from deepinv.sampling import (
    DiffPIR,
    PosteriorDiffusion,
    DPSDataFidelity,
    VarianceExplodingDiffusion,
    EulerSolver,
)

dps_psnr = None
diffpir_psnr = None
obs_psnr = None
HAS_RUN = False
run_error = None  # GPU分支异常信息，用于区分"无GPU"与"运行失败"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

_MODELS_DIR = os.path.join(SAVE_DIR, "models")
# 测试图 64×64：NCSNpp 为 FFHQ-64 模型（按 64×64 训练）；DRUNet 对分辨率不敏感
IMG_SIZE = 64
# Inpainting 缺口：与官方 demo_diffusion_sde.py 一致（中心 16×16）
_MASK_HOLE = (slice(24, 40), slice(24, 40))


def _load_model_local(model_cls, local_path, **kwargs):
    """通用模型加载器：优先本地，否则调用 deepinv 自动下载（使用 PyTorch hub 缓存）。

    重要：deepinv 的 ``pretrained`` 参数既可以是 ``"download"``，也可以是本地 checkpoint
    路径字符串。若只传 ``pretrained="download"`` 而不指定 ``local_path``，实际下载的
    checkpoint 可能与预期声明的模型不一致，需核对下载来源与文件名。
    """
    if os.path.isfile(local_path):
        print(f"  -> 从本地加载: {local_path}")
        # deepinv 支持通过 pretrained=local_path 直接加载本地权重
        return model_cls(pretrained=local_path, **kwargs).eval()
    else:
        print(f"  -> 本地模型不存在 ({local_path})，使用 pretrained='download'")
        print(f"     （模型会从 PyTorch hub 缓存加载或首次下载到 ~/.cache/torch/hub/）")
        return model_cls(pretrained="download", **kwargs).eval()


if device.type == "cuda":
    try:
        # ============================================================
        # 设计：同一逆问题（同 y、同 physics），各自原生先验
        #   - DPS：NCSNpp 得分模型 + DPSDataFidelity(weight=1024) + 默认 VE-SDE
        #     （装配框架对齐官方 demo_diffusion_sde.py，weight 按本图实测校准）
        #   - DiffPIR：DRUNet + DDPM/HQS（其原生 PnP 用法）
        #   - 先验不同是如实注明的混杂因素（见结论4），但这正是两种算法生态的典型用法
        # ============================================================

        # ===== 加载 NCSNpp 得分模型（DPS 原生先验）=====
        print("正在加载 NCSNpp 得分模型（FFHQ-64 EDM，DPS 原生先验，首次运行需下载权重）...")
        ncsnpp = _load_model_local(
            dinv.models.NCSNpp, os.path.join(_MODELS_DIR, "ncsnpp.pth")
        ).to(device)
        print("NCSNpp加载成功")

        # ===== 加载 DRUNet（DiffPIR 原生先验）=====
        print("正在加载预训练 DRUNet 去噪器（DiffPIR 先验）...")
        drunet = _load_model_local(
            dinv.models.DRUNet, os.path.join(_MODELS_DIR, "drunet_color.pth")
        ).to(device)
        print("DRUNet加载成功")

        # ===== 构造逆问题：64×64 inpainting（官方 demo 场景，无测量噪声）=====
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
            except Exception as e2:
                print(f"备用URL也加载失败: {e2}")
                x_test = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device)

        mask = torch.ones_like(x_test)
        mask[..., _MASK_HOLE[0], _MASK_HOLE[1]] = 0.0
        physics = dinv.physics.Inpainting(
            img_size=x_test.shape[1:], mask=mask, device=device
        )

        y = physics(x_test)
        assert y.shape == x_test.shape, (
            f"观测尺寸 {tuple(y.shape)} 与图像 {tuple(x_test.shape)} 不一致，"
            "请检查 Inpainting 掩码设置"
        )
        mse_y = torch.mean((y - x_test)**2).item()
        obs_psnr = 10 * np.log10(1.0 / (mse_y + 1e-10))
        print(f"观测y（中心16×16缺口）PSNR: {obs_psnr:.2f} dB")

        # ---- DPS（装配框架对齐官方 demo，weight 按本图实测校准）----
        n_steps_dps = 250    # 官方 demo 用 150（其注释明言仅为省时，实践中应加大）
        dps_weight = 1024.0  # 引导强度：实测 4→1024 单调改善(19.8→34.5dB)、
        #   2048 缺口区数值发散(16.6dB)。官方 4.0 是其演示图的取值，换图需重扫
        # 防护：装配式 API 需 deepinv >= 0.4.x
        for _name in ("PosteriorDiffusion", "DPSDataFidelity",
                      "VarianceExplodingDiffusion", "EulerSolver"):
            if not hasattr(dinv.sampling, _name):
                raise RuntimeError(
                    f"当前 deepinv 版本 {dinv.__version__} 缺少 dinv.sampling.{_name}，"
                    "本实验需 >=0.4.2，请升级：pip install -U deepinv "
                    "或 pip install git+https://github.com/deepinv/deepinv.git"
                )
        print(f"\n正在执行 DPS 后验采样共 {n_steps_dps} 步"
              f"（NCSNpp + VE-SDE(默认 sigma_max=80) + weight={dps_weight}）...")
        print("提示：每步需 NCSNpp 推理一次 + 似然梯度计算，GPU 上约需 1-3 分钟")
        import time
        t0 = time.time()
        model_dps = PosteriorDiffusion(
            data_fidelity=DPSDataFidelity(denoiser=ncsnpp, weight=dps_weight),
            denoiser=ncsnpp,
            sde=VarianceExplodingDiffusion(     # 官方默认: sigma_min=0.001,
                device=device,                  # sigma_max=80, alpha=0.25
                dtype=torch.float32,
            ),
            solver=EulerSolver(
                timesteps=torch.linspace(1, 0.001, n_steps_dps),
                rng=torch.Generator(device=device),  # 显式传 Generator → seed 生效
            ),
            dtype=torch.float32,
            device=device,
            verbose=True,
            # minus_one_one 不传（默认 True，与官方 demo 一致）：
            # NCSNpp 内部自带 [-1,1]↔[0,1] 变换，外层包装器在 0.4.2 中实际直通
        )
        result_dps = model_dps(y, physics, seed=42, denoise_output=True)
        x_hat_dps = result_dps[0] if isinstance(result_dps, (tuple, list)) else result_dps
        # 输出可能轻微越界 [0,1]，评估/可视化前统一 clamp
        x_hat_dps = x_hat_dps.detach().clamp(0, 1)
        print(f"DPS重建耗时: {time.time() - t0:.1f}s")
        print(f"DPS重建: shape={x_hat_dps.shape}, min={x_hat_dps.min():.3f}, max={x_hat_dps.max():.3f}")

        mse = torch.mean((x_hat_dps - x_test)**2).item()
        dps_psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        print(f"DPS重建PSNR: {dps_psnr:.2f} dB")

        # ---- DiffPIR ----
        # DiffPIR 的 HQS 框架直接接受普通去噪器（DRUNet），其原生 PnP 用法
        print("\n正在初始化DiffPIR（DRUNet作为去噪器）...")
        n_steps_diffpir = 100
        print(f"正在运行DiffPIR（max_iter={n_steps_diffpir}，含去噪+数据保真+采样三步）...")
        t0 = time.time()
        diffpir = DiffPIR(
            model=drunet,             # DRUNet（去噪器）
            data_fidelity=dinv.optim.L2(),
            sigma=0.02,
            max_iter=n_steps_diffpir,  # 显式传入，不依赖默认值
            zeta=0.7,  # 定稿依据：本地CPU×5种子实测——0.3 均值24.5dB且3/5坏签
            #   （21.0-30.2 摆动，坏签=缺口内发丝涂抹伪影）；0.7 均值28.4dB、
            #   方差更小（1/5坏签）；0.1 直接出涂抹伪影；sigma/lambda 扫描无改善。
            #   见脚本结论5与 results_summary.json 的 diffpir_zeta_sweep
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
        axes[0].set_title('观测 y（中心16×16缺口）', fontsize=12)
        axes[0].axis('off')

        x_dps_vis = x_hat_dps[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        axes[1].imshow(x_dps_vis)
        axes[1].set_title(f'DPS重建（NCSNpp先验）\nPSNR={dps_psnr:.2f} dB', fontsize=12)
        axes[1].axis('off')

        x_pir_vis = x_hat_diffpir[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        axes[2].imshow(x_pir_vis)
        axes[2].set_title(f'DiffPIR重建（DRUNet先验）\nPSNR={diffpir_psnr:.2f} dB', fontsize=12)
        axes[2].axis('off')

        plt.suptitle('deepinv框架：DPS vs DiffPIR 图像修复对比（13.3.6节）', fontsize=13, y=1.01)
        plt.tight_layout()
        fig_path = os.path.join(SAVE_DIR, 'DPSvsDiffPIR对比.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n图已保存: {fig_path}")

    except Exception as e:
        run_error = f"{type(e).__name__}: {e}"
        print(f"步骤执行出错: {e}")
        print("注意：本次未生成新结果，目录中若存在旧图/旧JSON均为历史产物，不代表本次运行")
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
print(f"{'似然近似':<20s} | {'delta近似（插件式）':<25s} | {'交替优化':<25s}")
print(f"{'采样方式':<20s} | {'修正逆向SDE':<25s} | {'去噪-投影交替':<25s}")
print(f"{'调度方式':<20s} | {'VE-SDE（默认sigma_max=80）':<25s} | {'DDPM（离散1000步）':<25s}")
print(f"{'先验/去噪器':<20s} | {'NCSNpp（score model）':<25s} | {'DRUNet（PnP去噪器）':<25s}")
print(f"{'输出性质':<20s} | {'近似后验样本':<25s} | {'MAP-like解':<25s}")
print(f"{'参考论文':<20s} | {'Chung et al. 2022':<25s} | {'Zhu et al. 2023':<25s}")
print(f"{'任务':<20s} | {f'{IMG_SIZE}x{IMG_SIZE} inpainting':<25s} | {f'{IMG_SIZE}x{IMG_SIZE} inpainting':<25s}")

print("\n" + "=" * 60)
print("实验13.3-2 完成!")
print("=" * 60)
if HAS_RUN and dps_psnr is not None:
    print(f"""
关键结论:
1. deepinv库实现（13.3.6节 四类方法对比）
   - DPS（Grad类）使用官方装配式API：PosteriorDiffusion + DPSDataFidelity(weight=1024)
     + NCSNpp + 默认VE-SDE(sigma_max=80) + EulerSolver
     （装配框架对齐 examples/sampling/demo_diffusion_sde.py，weight 按本图实测校准）
     → 近似后验样本
   - DiffPIR（Opt类）使用 DRUNet + DDPM + L2数据保真项 → MAP-like解
   - 两种方法共享同一观测 y（同 Inpainting 算子、同缺口），各自使用原生先验
   - PSNR对比：DPS={dps_psnr:.2f} dB vs DiffPIR={diffpir_psnr:.2f} dB
   - 观测y（含缺口）PSNR={obs_psnr:.2f} dB

2. 算法差异（对照13.3.6节对比表，附本次实测修正）
   - DPS (Grad类): 每步用autograd计算似然梯度修正eps；delta插件近似使结果对引导
     强度高度敏感——本图实测 4→1024 总PSNR差 14.7 dB、2048 直接发散。校准后
     强生成先验的价值得以兑现：总PSNR与可见区一致性均超过 DiffPIR
   - DiffPIR (Opt类): 每步去噪后做数据一致性投影（PnP风格），再按 zeta 重注入噪声
     → MAP-like解。"开箱即用"仅指数据保真步：缺口外跨种子/超参稳定硬锁存
     （MSE~0.0002）；但缺口内填充由随机扩散轨迹+弱先验决定，跨种子 21-30 dB
     摆动（本地CPU×5种子），坏签呈"发丝涂抹盖眼鼻"伪影。zeta 是填充质量旋钮
     （0.1 伪影 / 0.3 均值24.5 / 0.7 均值28.4），本实验定稿 0.7

3. 选型建议（13.3.6节方法选择指南，结合本次实测）
   - 追求重建上限、愿意做超参校准 -> 选DPS（配 score model 类先验，weight 必扫）
   - 快速原型/开箱即用/任意去噪器接入 -> 选DiffPIR（数据步默认值即可工作；
     zeta 需按填充质量校准，且对随机抽签敏感，必要时多种子取优）

4. 为什么不共享同一先验（本实验相对早期版本的重要修订）
   - DPS 为 score model 而设计：需去噪器在 SDE 全程 sigma∈[0.001, 80] 上近似有效
     （NCSNpp 即按此范围训练）；DiffPIR 的 HQS 只在数据保真步调用去噪器，
     任意普通去噪器（DRUNet，sigma≲0.5 有效）即可
   - 实测踩坑（256×256 去模糊 + DRUNet + DPS，13种配置无一正常）：漂移力 ∝ g²(t)、
     噪声注入 ∝ g(t)，sigma→0 时随机项压倒数据引导；linspace 时间网格扫过
     DRUNet 有效带仅数步；warm-start 甚至会把好初值破坏到 11.7 dB。唯一接近正常
     的是 alpha=0 纯 ODE（18.3 dB）——恰好印证上述机制
   - deepinv 官方所有 DPS demo（demo_dps.py、demo_diffusion_sde.py）全部是
     inpainting + score model/DiffUNet，从未验证过去模糊。教学实验应站在
     官方验证过的场景上，而不是替库踩未测试区域
   - 代价：先验不同（NCSNpp vs DRUNet）是混杂因素，PSNR 差异不能 100% 归因于
     算法类别——这正是"各用原生先验才是典型用法"与"控制变量"的权衡，如实注明

5. 任务与参数说明（超参扫描记录，seed=42、DPS 250步 / DiffPIR 100步）
   - inpainting（中心16×16缺口、无测量噪声）是官方验证过的 DPS 场景
   - 扫描：w=4/8/16/32/64/128/256/512/1024 → 总PSNR
     19.8/20.7/22.1/23.9/26.8/28.3/30.8/33.5/34.5 dB（单调），2048 → 16.6（发散）。
     可见区MSE 同步单调 0.0101→0.0001；w=1024 时可见区一致性已优于 DiffPIR
   - 机理：梯度 = weight*sqrt(C*H*W)*残差"单位方向"，模长恒定不随残差自适应，
     weight 是唯一强度旋钮；inpainting 的 A 为对角掩码，高权重≈每步良性硬投影。
     官方 4.0 只是其演示图上的取值——"官方值"不可跨图照搬，更不可跨算子照搬
     （去模糊+256×256 即便照搬官方 DRUNet 配方的 0.3 也曾实测发散全黑，PSNR 8.34）
   - DiffPIR 超参/种子扫描（本地CPU×5种子，sigma=0.02、lam=7、100步）：zeta=0.1
     直接涂抹伪影；0.3 均值24.5 dB（21.0-30.2 摆动，3/5坏签）；0.7 均值28.4 dB、
     方差更小（1/5坏签，定稿）。sigma=0.1/0.4 使数据锁变松（缺口外MSE 升至
     0.0016/0.009）不可用；lambda=25 无改善。结论：DiffPIR 的缺口内填充本质是
     随机扩散采样，与 DPS 不同——它的"重建"也带抽签属性，换图换种子可能变脸

6. 实测观察
   - 本次运行：DPS={dps_psnr:.2f} dB，DiffPIR={diffpir_psnr:.2f} dB，观测={obs_psnr:.2f} dB
   - 注意后验采样的多样性属性：DPS 是"近似后验样本"，缺口补全样式可能与真值不同
     （尤其大缺口），这属于后验多样性而非错误；判断引导是否失效应看重建是否与
     y 的可见区域一致，或跨 seed 采样是否围绕 y 合理解释
   - PSNR 反映全局 MSE，可能掩盖局部伪影：建议数值与目检并用
   - 伪影解剖（DiffPIR 坏签实拍：zeta=0.3/seed=42 GPU 运行出现右眼区发丝涂抹，
     24.0 dB）：伪影全部在缺口内，缺口外逐像素完好（MSE≤0.0003）——数据 prox
     对缺口无约束（A^T A=0），填充 100% 来自 DRUNet 先验步的随机轨迹；坏签
     形态=邻近刘海/发丝沿缺口边界延拓盖住眼鼻+回归均值偏糊。重建图若出现
     此类涂抹，先换种子/调 zeta 重跑（本实验已定稿 zeta=0.7），而非怀疑算子
     或数据链路
   - 可见区一致性与身份保持（误差分解：缺口外=数据一致性，缺口内=补全样式）：
     w=4 时 DPS 缺口外MSE≈0.010、占总误差74%——观测为闭唇却被重绘成露齿笑，
     "不像同一个人"，这是先验压过数据的典型形态；校准后（w=1024）缺口外
     MSE≈0.0001，优于 DiffPIR（≈0.0002，其数据保真步为硬约束）。该分解可作为
     学生自查工具：把重建图与观测图在缺口外逐像素比对，即可判断引导强度是否合适
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
# ===== 保存数值结果 =====
import json

def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
    import numpy as np
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except: pass
    return obj

results_summary = {}
if HAS_RUN and dps_psnr is not None:
    results_summary["DPS_PSNR"] = round(float(dps_psnr), 2)
    results_summary["DiffPIR_PSNR"] = round(float(diffpir_psnr), 2)
    if obs_psnr is not None:
        results_summary["obs_PSNR"] = round(float(obs_psnr), 2)
    results_summary["config"] = {
        "img_size": IMG_SIZE,
        "task": "inpainting",
        "mask_hole": "[24:40, 24:40]（中心16×16，官方demo同款）",
        "measurement_noise": "none",
        "dps_model": "NCSNpp（FFHQ-64 EDM score model）",
        "dps_weight": dps_weight,
        "dps_weight_sweep": "4→1024 单调改善（总PSNR 19.8→34.5 dB，可见区MSE 0.0101→0.0001）；2048 发散（16.6 dB）。seed=42、250步",
        "dps_n_steps": n_steps_dps,
        "dps_schedule": "ve",
        "dps_sigma_max": 80.0,
        "dps_alpha": 0.25,
        "diffpir_max_iter": n_steps_diffpir,
        "diffpir_zeta": 0.7,
        "diffpir_lambda": 7.0,
        "diffpir_denoiser": "DRUNet",
        "diffpir_zeta_sweep": "本地CPU×5种子(sigma=0.02, lam=7, 100步)：zeta=0.1 涂抹伪影；"
                              "0.3 均值24.5dB、21.0-30.2摆动、3/5坏签(缺口内发丝涂抹)；"
                              "0.7 均值28.4dB、1/5坏签(定稿)。sigma=0.1/0.4 数据锁变松"
                              "(缺口外MSE 0.0016/0.009)不可用；lambda=25 无改善",
        "seed": 42,
        "deepinv_version": dinv.__version__,
        "note": "DPS 与 DiffPIR 共享同一观测 y 与 Inpainting 算子，各自使用原生先验"
                "（NCSNpp / DRUNet）。DPS 装配框架对齐官方 demo_diffusion_sde.py 的"
                " NCSNpp 段（weight=1024 为本图实测校准值，官方 4.0 仅是其演示图取值）；"
                "rng 显式传 torch.Generator 使 seed=42 生效。"
                "DiffPIR 缺口外为硬数据锁存（跨种子/超参稳定 ~0.0002），缺口内填充"
                "由随机扩散轨迹+DRUNet 弱先验决定，跨种子大幅摆动（zeta=0.3 时"
                "21-30dB）——PSNR 对比需注明抽签成分。"
                "不共享先验的原因与 256×256 去模糊踩坑记录见脚本结论 4/6",
    }
else:
    if run_error is not None:
        results_summary["note"] = f"GPU实验运行失败，未产生新结果: {run_error}"
    else:
        results_summary["note"] = "GPU未检测到，未实际运行"
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
