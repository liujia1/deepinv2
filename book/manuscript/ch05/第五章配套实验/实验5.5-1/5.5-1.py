# -*- coding: utf-8 -*-
"""
实验5.5-1 PnP-ULA后验采样实现
对应章节：5.5 PnP框架：用去噪器替换先验梯度
知识点：
  - PnP-ULA递推式：X_{m+1} = X_m - δ∇f(X_m) + (δ/ε)[D_ε(X_m)-X_m] + √(2δ)Z
  - Tweedie等式：∇log p_ε(x) = (D_ε(x) - x) / ε
  - 去噪器作为隐式先验
  - 步长约束：δ ≤ 1/(L_f + L_D/ε)

素材来源：
  - Mathematics.../Teaching Unit 2/labs/lab2_PnP_sol.ipynb
    - Cell 3-6: 导入、图像加载、模糊算子、含噪观测
    - Cell 8-10: 似然函数、梯度、Lipschitz常数
    - Cell 11-12: 去噪器加载、算法参数
    - Cell 13: Markov_kernel 函数（PnP-ULA）
    - Cell 14-17: 步长选择、主采样循环

运行前提：
  步骤1-4：CPU 或 GPU 均可运行（PnP-ULA 算法本身不依赖 GPU）
    - sampling_tools/ 已包含在当前目录
    - Pretrained_models/ 会自动下载（首次运行）
    - 仅当缺少 sampling_tools 或预训练模型时，步骤4才会被跳过
    - 若有 GPU 会自动使用以加速（可选，非必需）

链长与耗时（重要）：
  - 默认 maxit=10000、burnin=5%、目标保存 100 个抽稀样本（间隔约100步，
    实际落盘约94个，npz 约25~30MB，供实验5.6系列使用），与上游官方笔记本的
    迭代预算一致（上游保存2000个样本，本实验受落盘体积限制取100个）；
    10000 次在 GPU 上约 9 分钟，在纯 CPU 上约 2.5~3 小时。
  - 10000 次仅够"演示采样流程 + MMSE 复原"：实测后验均值 PSNR 已到约 33 dB
    的渐近平台，但数据失配慢模态的 IAT 长达上千步（ESS 仅约10个量级），
    尚不满足严格的不确定性量化（UQ）；上游明确指出真实 UQ 通常需要
    5×10^5 ~ 10^6 次迭代。请以收敛曲线走平 + IAT/ESS + 劈半诊断为准。
  - 快速冒烟测试（不用于结论）可设置环境变量：
        PNP_MAXIT=200 PNP_NSAMPLES=40 python 5.5-1.py
"""

import math
import torch
import numpy as np
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
    SAVE_DIR = os.path.join(_gdrive, '实验5.5-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
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
# ========================================================

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
# 尽量保证同设备跨次运行可复现（不影响 CPU/GPU 均可运行的性质）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        print(f"  路径: {_gdrive}/实验5.5-1/")
        print("  需要上传的文件:")
        print("    - 5.5-1.py")
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


# ============================================================
# 模型自动下载功能
# ============================================================
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

if not (_has_sampling_tools and _has_model):
    print("\n" + "=" * 60)
    print("运行环境检查")
    print("=" * 60)
    print(f"  device: {device}（CPU/GPU 均可运行）")
    print(f"  sampling_tools可用: {_has_sampling_tools}")
    print(f"  预训练模型可用: {_has_model}")
    print("\n  缺少必要资源（sampling_tools 或预训练模型），将仅运行步骤1-3")
    print("  步骤4需要 sampling_tools 和预训练模型，不强制要求 GPU")
    print("=" * 60)


# ============================================================
# 步骤1：加载图像与模糊算子
# ============================================================
print("\n" + "=" * 60)
print("步骤1：加载图像与模糊算子")
print("=" * 60)

# 加载图像
_cman_path = os.path.join(SAVE_DIR, "cman.png")
if not os.path.exists(_cman_path):
    print(f"[Warning] 未找到图像文件: {_cman_path}")
    print("[Warning] 将使用合成图像进行演示")
    # 创建合成图像
    im = np.zeros((256, 256))
    im[50:100, 50:100] = 1.0
    im[150:200, 150:200] = 0.8
    im[100:150, 50:200] = 0.5
else:
    im = np.array(Image.open(_cman_path))

print(f"图像尺寸: {im.shape}")

# 可视化
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(im, cmap='gray')
plt.title('原始图像 $x$')
plt.axis('off')

if _has_sampling_tools:
    x = torch.Tensor(im/255.).to(device)

    # 模糊算子
    kernel_len = [5, 5]
    size = [im.shape[0], im.shape[1]]
    type_blur = "uniform"
    A, AT, AAT_norm = blur_operators(kernel_len, size, type_blur, device)

    # 含噪观测
    y0 = A(x)
    BSNRdb = 40
    sigma = torch.linalg.matrix_norm(A(x)-torch.mean(A(x)), ord='fro')/math.sqrt(torch.numel(x)*10**(BSNRdb/10))
    y = y0 + sigma * torch.randn_like(x)

    plt.subplot(1, 2, 2)
    plt.imshow(y.cpu().numpy(), cmap='gray')
    plt.title(f'含噪+模糊观测 $y$ (BSNR={BSNRdb}dB)')
    plt.axis('off')

    print(f"模糊核类型: {type_blur}, 尺寸: {kernel_len}")
    print(f"噪声标准差: {sigma.item():.4f}")
    print(f"BSNR: {BSNRdb} dB")
else:
    print("  缺少 sampling_tools，跳过模糊和加噪")
    plt.subplot(1, 2, 2)
    # 创建一个空白图像用于显示提示
    plt.imshow(np.ones_like(im) * 0.9, cmap='gray', vmin=0, vmax=1)
    plt.text(0.5, 0.6, '缺少依赖', 
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=14, fontweight='bold', color='red',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2))
    plt.text(0.5, 0.35, '请安装 sampling_tools\n以生成含噪+模糊观测', 
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=10, color='darkred')
    plt.title('含噪+模糊观测 $y$ (缺少 sampling_tools)')
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_图像与观测.png'), dpi=150)
plt.close()


# ============================================================
# 步骤2-3：似然/梯度、Tweedie/去噪器 —— 统一由共享模块构建
# 问题构建与采样循环的唯一实现在同目录 pnp_ula_common.py，
# 实验5.6-1/5.6-2/5.6-3 均通过该模块引用，不在各脚本中复制。
# ============================================================
prob = None
if _has_sampling_tools and _has_model:
    try:
        from pnp_ula_common import build_problem
        prob = build_problem(device=device, model_path=_model_path)
    except Exception as _build_err:
        print(f"[Error] 构建PnP-ULA问题失败: {_build_err}")
        prob = None

if prob is not None:
    print("\n" + "=" * 60)
    print("步骤2：似然与梯度")
    print("=" * 60)

    print("似然函数: f(x) = ‖y-Ax‖² / (2σ²)")
    print("似然梯度: ∇f(x) = Aᵀ(Ax-y) / σ²")
    print(f"似然梯度Lipschitz常数: L_f = ‖AA‖ / σ² = {float(prob.L_y):.2f}")

    print("\n" + "=" * 60)
    print("步骤3：Tweedie等式与去噪器")
    print("=" * 60)

    print(f"\n去噪器训练噪声水平: ε = (5/255)² = {prob.eps0:.6f}")
    print("\nTweedie等式:")
    print("  ∇log p_ε(x) = (D_ε(x) - x) / ε")
    print("\nPnP-ULA中的先验得分步:")
    print("  (δ/ε)(D_ε(x) - x) = δ · ∇log p_ε(x)")
    print("\n这展示了PnP的核心思想：用去噪器替换先验梯度")


# ============================================================
# 步骤4：PnP-ULA采样（采样循环实现见 pnp_ula_common.run_pnp_ula，禁止复制）
# ============================================================
if prob is not None:
    from pnp_ula_common import run_pnp_ula

    print("\n" + "=" * 60)
    print("步骤4：PnP-ULA采样")
    print("=" * 60)

    # 主采样循环预算
    # maxit=10000、burnin=500（5%，与官方 lab2_PnP_sol 一致），post-burnin 共9500步。
    # 预算依据：本链与实验5.4-2 完全同构（同 cman 图、同 σ/δ、同 RealSN_DnCNN 去噪器），
    # 正式实测后验运行均值 PSNR 在约1200步时仅约30.1 dB，到10000步才到33.1 dB平台；
    # 早期 maxit=200 版曲线末端近线性上升、maxit=1000 版仍未到平台，
    # 均不能声称链已进入平稳分布（见实验5.6-2复核问题2），故对齐官方预算并保留IAT/ESS诊断。
    # 默认 10000（官方预算）；可用环境变量 PNP_MAXIT/PNP_NSAMPLES 缩短做冒烟测试
    maxit = int(os.environ.get("PNP_MAXIT", "10000"))
    burnin = np.int64(maxit * 0.05)
    # 抽稀目标样本数：官方 n_samples=2000（间隔5步），但这里样本要落盘 npz 供5.6系列使用，
    # 2000张256×256未压缩约占0.5GB以上；故取100（间隔100步，实际约94张、约25~30MB）。
    # 抽稀间隔是否≥IAT由下方诊断自动判定并如实打印。
    n_samples = np.int64(os.environ.get("PNP_NSAMPLES", "100"))
    _smoke_run = ("PNP_MAXIT" in os.environ) or ("PNP_NSAMPLES" in os.environ)
    if _smoke_run:
        print("[冒烟测试] 检测到 PNP_MAXIT/PNP_NSAMPLES 环境变量覆盖："
              f"maxit={maxit}, n_samples={n_samples}")
        print("[冒烟测试] 该设置仅用于检查代码能否跑通，产出结果不可用于实验结论")

    # 问题相关变量（供本文件打印/绘图/JSON 使用，均来自共享模块构建的 prob）
    alpha = prob.alpha
    L_net = prob.L_net
    eps = prob.eps0
    sigma = prob.sigma
    L_y = prob.L_y
    x = prob.x
    y = prob.y
    BSNRdb = prob.bsnr_db
    NRMSE, PSNR, SSIM = prob.NRMSE, prob.PSNR, prob.SSIM

    res = run_pnp_ula(prob, maxit=maxit, burnin=int(burnin), n_samples=int(n_samples),
                      desc="PnP-ULA采样")

    delta = res['delta']
    delta_max = res['delta_max']
    print(f"步长: δ = {delta:.6f} (最大允许: {delta_max:.6f})")
    print(f"步长约束: δ ≤ 1/(L_f + L_D/ε)")

    # 解包到与旧版同名的变量（绘图/npz落盘/JSON 代码保持不变）
    burnin = np.int64(res['burnin'])
    half_mark = res['half_mark']
    thinning_step = np.int64(res['thinning_step'])
    post_meanvar = res['post_meanvar']
    post_meanvar_h1 = res['post_meanvar_h1']
    post_meanvar_h2 = res['post_meanvar_h2']
    absfouriercoeff = res['absfouriercoeff']
    MC_X = list(res['mc_samples'])
    trace_fid = list(res['trace_fid'])
    nrmse_values = res['nrmse_values']
    psnr_values = res['psnr_values']
    ssim_values = res['ssim_values']
    iat_fid = res['iat_fid']
    ess_fid = res['ess_fid']
    iat_diag = res['iat_diag']
    n_post = res['n_post']
    last_quarter_gain = res['last_quarter_gain']
    half_sigma_corr = res['half_sigma_corr']
    elapsed = res['elapsed']

    # 显式收敛判定（阈值固定并打印，避免"只报数值不给结论"或"暗示已收敛"）
    _GAIN_PLATFORM_DB = 0.2     # 末1/4段运行均值 PSNR 提升小于此值视为 MMSE 渐近平台
    _HALF_CORR_STATIONARY = 0.9 # 劈半后验标准差空间相关系数达到此值才支持链已平稳
    _ESS_MIN_UQ = 30.0          # 定量 UQ 所需有效样本量的最低门槛（经验值，仅作粗筛）
    mmse_platform = last_quarter_gain < _GAIN_PLATFORM_DB
    half_stationary = half_sigma_corr >= _HALF_CORR_STATIONARY
    iat_truncated = bool(iat_diag['truncated'])
    # 慢模态充分混合需同时满足：Geyer 序列在 lag 上限内完成去相关，且 ESS 达到最低门槛
    uq_mixing_ok = (not iat_truncated) and (ess_fid >= _ESS_MIN_UQ)
    uq_ok = uq_mixing_ok and half_stationary

    print(f"\n收敛诊断（基于数据失配 ||y-AX||^2/(2*sigma^2) 的标量轨迹，不依赖真值）:")
    print(f"  post-burnin 步数: {n_post}")
    if iat_truncated:
        print(f"  IAT（积分自相关时间）: >= {iat_fid:.2f} 步【下界】"
              f"——滞后 {iat_diag['max_lag']} 步内 rho 对始终未转负")
        print(f"  ESS（有效样本量）: <= {ess_fid:.1f}【上界】：慢模态在本链长下未充分混合")
    else:
        print(f"  IAT（积分自相关时间）: {iat_fid:.2f} 步（初始正序列截断于 lag {iat_diag['cutoff_lag']}）")
        print(f"  ESS（有效样本量）: {ess_fid:.1f}")
    if thinning_step >= iat_fid:
        print(f"  抽稀间隔 {thinning_step} 步 >= IAT {iat_fid:.2f}：抽稀样本可近似视为独立")
    else:
        print(f"  抽稀间隔 {thinning_step} 步 < IAT {iat_fid:.2f}：抽稀样本仍有自相关，下游统计结论需谨慎")
    print(f"  前/后半段后验sigma空间相关系数: {half_sigma_corr:.4f}（判平稳阈值 >= {_HALF_CORR_STATIONARY}）")
    print(f"  末1/4段运行均值PSNR提升: {last_quarter_gain:+.3f} dB（判平台阈值 < {_GAIN_PLATFORM_DB} dB）")
    print("\n  结论（MMSE 复原与 UQ 分开判定）:")
    print(f"    - MMSE 后验均值渐近平台: {'是' if mmse_platform else '否'}")
    print(f"    - 劈半平稳性: {'支持' if half_stationary else '不支持（相关系数低于阈值，链可能仍在漂移）'}")
    print(f"    - 慢模态去相关: {'Geyer序列已在lag上限内截断' if not iat_truncated else '否（IAT下界仍超滞后上限）'}"
          f"，ESS={ess_fid:.1f}（定量UQ经验门槛≥{_ESS_MIN_UQ:g}）")
    print(f"    - 定量 UQ 前提（未截断 + ESS达标 + 劈半平稳）: {'满足' if uq_ok else '不满足'}")
    if not (mmse_platform and uq_ok):
        print("    => 当前链长可用于演示 PnP-ULA 采样与 MMSE 复原；")
        print("       后验方差/分位数等定量 UQ 结论需更长链（上游建议 5e5~1e6 次）后重跑。")

    # 结果评估
    print(f"\n采样耗时: {elapsed:.2f} 秒")
    print(f"\n初始指标:")
    print(f"  NRMSE: {NRMSE(x, y):.4f}")
    print(f"  PSNR: {PSNR(x, y):.2f} dB")
    print(f"  SSIM: {SSIM(x, y):.4f}")

    post_mean = post_meanvar.get_mean()
    print(f"\n后验均值指标:")
    print(f"  NRMSE: {NRMSE(x, post_mean):.4f}")
    print(f"  PSNR: {PSNR(x, post_mean):.2f} dB")
    print(f"  SSIM: {SSIM(x, post_mean):.4f}")

    # 可视化 - 需要手动保存两个图表
    # plots函数会创建两个图表，我们需要分别保存
    
    # 创建第一个图表（2行4列）
    post_mean_numpy = post_meanvar.get_mean().detach().cpu().numpy()
    post_var_numpy = post_meanvar.get_var().detach().cpu().numpy()
    
    fig1, axes1 = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
    fig1.tight_layout(pad=.01)
    fig1.suptitle(f'PnP-ULA 后验采样结果（cameraman，BSNR=40dB，maxit={maxit}，烧入={burnin}）：'
                  f'后验均值 PSNR={PSNR(x, post_mean):.2f} dB',
                  fontsize=14)
    
    axes1[0,0].imshow(x.detach().cpu().numpy(), cmap="gray")
    axes1[0,0].set_title('Ground truth image')
    axes1[0,0].axis('off')
    
    axes1[0,1].imshow(y.detach().cpu().numpy(), cmap="gray")
    axes1[0,1].set_title('Blurred noisy image')
    axes1[0,1].axis('off')
    
    axes1[0,2].imshow(post_mean_numpy, cmap="gray")
    axes1[0,2].set_title('x - posterior mean')
    axes1[0,2].axis('off')
    
    axes1[0,3].imshow(post_var_numpy, cmap="gray")
    axes1[0,3].set_title('x - posterior variance')
    axes1[0,3].axis('off')
    
    axes1[1,0].imshow(post_mean_numpy/np.sqrt(post_meanvar.get_var().detach().cpu().numpy()), cmap="gray")
    axes1[1,0].set_title('x - posterior mean/posterior SD')
    axes1[1,0].axis('off')
    
    axes1[1,1].imshow(np.sqrt(post_var_numpy)/post_mean_numpy, cmap="gray")
    axes1[1,1].set_title('x - Coefs of variation')
    axes1[1,1].axis('off')
    
    axes1[1,2].imshow(torch.log(absfouriercoeff.get_mean()).detach().cpu().numpy())
    axes1[1,2].set_title('Mean coefs (log-scale)')
    axes1[1,2].axis('off')
    
    axes1[1,3].imshow(torch.log(absfouriercoeff.get_var()).detach().cpu().numpy())
    axes1[1,3].set_title('Var coefs (log-scale)')
    axes1[1,3].axis('off')
    
    plt.savefig(os.path.join(SAVE_DIR, '步骤4_PnP-ULA结果_图像.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 创建第二个图表（1行3列）
    # 注意：三条曲线画的是“后验运行均值”（welford 在线累积均值）相对真值的指标，
    # 不是单步链样本 X 的指标——运行均值才是 MMSE 估计，其末端斜率趋平支持链已平稳。
    fig2, axes2 = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig2.tight_layout(pad=.01, rect=[0, 0, 1, 0.92])
    _xx = np.arange(len(nrmse_values))[::10]

    axes2[0].plot(_xx, nrmse_values[::10])
    axes2[0].set_title(f'后验运行均值 NRMSE（末值 {nrmse_values[-1]:.4f}）')
    axes2[0].set_xlabel('烧入后累积样本序号')

    axes2[1].plot(_xx, psnr_values[::10])
    axes2[1].set_title(f'后验运行均值 PSNR（末值 {psnr_values[-1]:.2f} dB，'
                       f'末1/4段 {last_quarter_gain:+.3f} dB）')
    axes2[1].set_xlabel('烧入后累积样本序号')

    axes2[2].plot(_xx, ssim_values[::10])
    axes2[2].set_title(f'后验运行均值 SSIM（末值 {ssim_values[-1]:.4f}）')
    axes2[2].set_xlabel('烧入后累积样本序号')

    _verdict_txt = ('MMSE运行均值已到渐近平台；但慢模态未充分混合、劈半相关 %.2f，'
                    '本链长仅用于流程演示与MMSE复原，定量UQ需更长链' % half_sigma_corr)
    if mmse_platform and uq_ok:
        _verdict_txt = '末端趋平 + IAT/ESS与劈半诊断共同支持链已平稳'
    fig2.suptitle('收敛轨迹（曲线为后验运行均值，非单步链样本；观测起点 PSNR=%.2f dB）：%s'
                  % (PSNR(x, y), _verdict_txt),
                  fontsize=13)
    plt.savefig(os.path.join(SAVE_DIR, '步骤4_PnP-ULA结果_曲线.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n可视化结果已保存:")
    print(f"  - {os.path.join(SAVE_DIR, '步骤4_PnP-ULA结果_图像.png')}")
    print(f"  - {os.path.join(SAVE_DIR, '步骤4_PnP-ULA结果_曲线.png')}")

    # 保存采样结果供后续实验使用
    np.savez(os.path.join(SAVE_DIR, 'sampling_results.npz'),
             post_mean=post_mean.cpu().numpy(),
             post_var=post_meanvar.get_var().cpu().numpy(),
             post_mean_h1=post_meanvar_h1.get_mean().cpu().numpy(),
             post_var_h1=post_meanvar_h1.get_var().cpu().numpy(),
             post_mean_h2=post_meanvar_h2.get_mean().cpu().numpy(),
             post_var_h2=post_meanvar_h2.get_var().cpu().numpy(),
             mc_samples=np.array(MC_X),
             # 收敛诊断轨迹：trace_fid 为数据失配（不依赖真值），trace_psnr 为运行均值PSNR
             trace_fid=np.array(trace_fid, dtype=np.float64),
             trace_psnr=np.array(psnr_values, dtype=np.float64),
             chain_config=np.array([maxit, burnin, half_mark, thinning_step], dtype=np.int64),
             x_true=x.cpu().numpy(),
             y_obs=y.cpu().numpy())
    print(f"\n采样结果已保存到: {os.path.join(SAVE_DIR, 'sampling_results.npz')}")
else:
    # 缺少 sampling_tools 或预训练模型时的可视化提示
    print("\n" + "=" * 60)
    print("步骤4：PnP-ULA采样")
    print("=" * 60)
    print("  缺少必要资源（sampling_tools 或预训练模型）")
    print("  无法执行PnP-ULA采样")
    
    # 创建第一个提示图表（2行4列，对应plots函数的第一个图表）
    fig1, axes1 = plt.subplots(2, 4, figsize=(15, 10))
    fig1.suptitle('PnP-ULA采样结果 (缺少依赖)', fontsize=16, fontweight='bold', color='red')
    
    for ax in axes1.flat:
        ax.imshow(np.ones((256, 256)) * 0.9, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    
    # 设置子图标题
    titles_row1 = ['Ground truth image', 'Blurred noisy image', 'x - posterior mean', 'x - posterior variance']
    titles_row2 = ['x - posterior mean/SD', 'x - Coefs of variation', 'Mean coefs (log)', 'Var coefs (log)']
    
    for i, title in enumerate(titles_row1):
        axes1[0, i].set_title(title, fontsize=10)
        axes1[0, i].text(0.5, 0.5, '缺少依赖', 
                        ha='center', va='center', transform=axes1[0, i].transAxes,
                        fontsize=12, color='red', fontweight='bold')
    
    for i, title in enumerate(titles_row2):
        axes1[1, i].set_title(title, fontsize=10)
        axes1[1, i].text(0.5, 0.5, '缺少依赖', 
                        ha='center', va='center', transform=axes1[1, i].transAxes,
                        fontsize=12, color='red', fontweight='bold')
    
    plt.tight_layout()
    # 缺依赖占位图必须与真实结果文件名区分，避免与成功运行的产物混淆/互相覆盖
    plt.savefig(os.path.join(SAVE_DIR, '步骤4_未执行_图像.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 创建第二个提示图表（1行3列，对应plots函数的第二个图表）
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('收敛曲线 (缺少依赖)', fontsize=14, fontweight='bold', color='red')
    
    titles_row3 = ['NRMSE of $X$ vs $x_{gr}$', 'PSNR of $X$ vs $x_{gr}$', 'SSIM of $X$ vs $x_{gr}$']
    
    for i, ax in enumerate(axes2):
        ax.text(0.5, 0.5, '缺少依赖\n才能显示\n收敛曲线', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2))
        ax.set_title(titles_row3[i], fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤4_未执行_曲线.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n提示图表已保存（注意：以下为缺依赖占位图，不是采样结果）:")
    print(f"  - {os.path.join(SAVE_DIR, '步骤4_未执行_图像.png')}")
    print(f"  - {os.path.join(SAVE_DIR, '步骤4_未执行_曲线.png')}")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.5-1 总结")
print("=" * 60)
print("1. PnP-ULA核心递推式:")
print("   X_{m+1} = X_m - δ∇f(X_m) + (δ/ε)[D_ε(X_m)-X_m] + √(2δ)·Z")
print("2. Tweedie等式架起去噪器与先验得分的桥梁:")
print("   ∇log p_ε(x) = (D_ε(x) - x) / ε")
print("3. 步长约束保证收敛:")
print("   δ ≤ 1/(L_f + L_D/ε)")
print("4. 后验采样生成多个样本，是不确定性量化(UQ)的原材料；但默认10000步链的慢模态")
print("   尚未充分混合（见IAT/ESS与劈半诊断），定量UQ结论需更长链（5e5~1e6步），")
print("   实验5.6-1/2/3 中凡依赖样本独立性的统计均会再次显式提示这一点")


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

results_summary = {
    "实验名称": "实验5.5-1 PnP-ULA后验采样实现",
    "环境信息": {
        "GPU可用": bool(_has_gpu),
        "sampling_tools可用": bool(_has_sampling_tools),
        "预训练模型可用": bool(_has_model),
        "device": str(device),
    },
}

# 步骤4的关键变量在条件块内定义，需安全访问
try:
    _step4 = {
        "算法参数": {
            "去噪器噪声水平_eps": round(float(eps), 8),
            "步长_delta": round(float(delta), 8),
            "最大允许步长_delta_max": round(float(delta_max), 8),
            "总迭代次数_maxit": int(maxit),
            "burnin迭代数": int(burnin),
            "采样样本数_n_samples": int(n_samples),
            "BSNR_db": float(BSNRdb),
            "噪声标准差_sigma": round(float(sigma), 8),
        },
        "初始指标": {
            "NRMSE": round(float(NRMSE(x, y)), 6),
            "PSNR_dB": round(float(PSNR(x, y)), 4),
            "SSIM": round(float(SSIM(x, y)), 6),
        },
        "后验均值指标": {
            "NRMSE": round(float(NRMSE(x, post_mean)), 6),
            "PSNR_dB": round(float(PSNR(x, post_mean)), 4),
            "SSIM": round(float(SSIM(x, post_mean)), 6),
        },
        "收敛诊断": {
            "诊断轨迹": "数据失配||y-AX||^2/(2*sigma^2)（不依赖真值x）",
            "post_burnin步数": int(n_post),
            "IAT_积分自相关时间_步": (f">={round(iat_fid, 4)}(下界)" if iat_truncated else round(iat_fid, 4)),
            "ESS_有效样本量": (f"<={round(ess_fid, 2)}(上界)" if iat_truncated else round(ess_fid, 2)),
            "IAT滞后截断": {
                "truncated": bool(iat_truncated),
                "初始正序列截断lag": iat_diag['cutoff_lag'],
                "计算lag上限": int(iat_diag['max_lag']),
                "含义": "truncated=true 表示数据失配轨迹在 lag 上限内 rho 对始终未转负，IAT 仅为下界、ESS 仅为上界，慢模态未充分混合",
            },
            "抽稀间隔_步": int(thinning_step),
            "实际落盘抽稀样本数": int(len(MC_X)),
            "抽稀间隔_ge_IAT": bool(thinning_step >= iat_fid),
            "前后半段后验sigma相关系数": round(half_sigma_corr, 6),
            "劈半平稳阈值": _HALF_CORR_STATIONARY,
            "末1/4段运行均值PSNR提升_dB": round(last_quarter_gain, 4),
            "MMSE平台阈值_dB": _GAIN_PLATFORM_DB,
            "定量UQ最低ESS门槛": _ESS_MIN_UQ,
            "MMSE复原渐近平台": bool(mmse_platform),
            "劈半平稳性支持": bool(half_stationary),
            "慢模态去相关_Geyer未截断": bool(not iat_truncated),
            "慢模态充分混合_支持定量UQ": bool(uq_mixing_ok),
            "定量UQ前提满足": bool(uq_ok),
            "综合结论": ("MMSE 复原演示成立" +
                      ("且后验均值到渐近平台；" if mmse_platform else "且运行均值尚未到平台；") +
                      ("当前链长不支持定量UQ（慢模态未充分混合/ESS不足/劈半未平稳），需5e5~1e6步长链"
                       if not uq_ok
                       else "IAT/ESS与劈半诊断支持当前链已平稳，可做定量UQ")),
            "说明": "IAT由Geyer初始正序列估计；前后半段为链上不同数据段；MMSE平台与UQ可用性是两件事，需分别判定",
        },
        "冒烟测试": bool(_smoke_run),
        "采样耗时_秒": round(float(elapsed), 4),
    }
    results_summary["步骤4_PnP-ULA采样"] = _step4
except (NameError, UnboundLocalError):
    results_summary["步骤4_PnP-ULA采样"] = "未执行（缺少 sampling_tools 或预训练模型；CPU/GPU 均可运行，GPU 仅用于加速）"

results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
