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
  步骤1-3：无特殊要求，CPU即可运行
  步骤4：需要GPU和预训练RealSN-DnCNN模型
    - sampling_tools/ 已包含在当前目录
    - Pretrained_models/ 会自动下载（首次运行）
    - 若无GPU，步骤4将被跳过
"""

import math
import torch
import numpy as np
import time
from tqdm.auto import tqdm
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

if not (_has_gpu and _has_sampling_tools and _has_model):
    print("\n" + "=" * 60)
    print("运行环境检查")
    print("=" * 60)
    print(f"  GPU可用: {_has_gpu}")
    print(f"  sampling_tools可用: {_has_sampling_tools}")
    print(f"  预训练模型可用: {_has_model}")
    print("\n  缺少必要资源，将仅运行步骤1-3（数据准备和算法演示）")
    print("  步骤4需要GPU和预训练模型")
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

if _has_gpu and _has_sampling_tools:
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
    print("  缺少GPU或sampling_tools，跳过模糊和加噪")
    plt.subplot(1, 2, 2)
    # 创建一个空白图像用于显示提示
    plt.imshow(np.ones_like(im) * 0.9, cmap='gray', vmin=0, vmax=1)
    plt.text(0.5, 0.6, '需要GPU运行', 
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=14, fontweight='bold', color='red',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2))
    plt.text(0.5, 0.35, '请使用GPU设备\n以生成含噪+模糊观测', 
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=10, color='darkred')
    plt.title('含噪+模糊观测 $y$ (需要GPU)')
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_图像与观测.png'), dpi=150)
plt.close()


# ============================================================
# 步骤2：似然与梯度
# ============================================================
if _has_gpu and _has_sampling_tools:
    print("\n" + "=" * 60)
    print("步骤2：似然与梯度")
    print("=" * 60)

    # 似然函数
    f = lambda x, A: (torch.linalg.matrix_norm(y-A(x), ord='fro')**2.0)/(2.0*sigma**2)

    # 似然梯度
    gradf = lambda x, A, AT: AT(A(x)-y)/sigma**2

    # Lipschitz常数
    L_y = AAT_norm/(sigma**2)

    print(f"似然函数: $f(x) = \\|y-Ax\\|^2 / (2\\sigma^2)$")
    print(f"似然梯度: $\\nabla f(x) = A^T(Ax-y) / \\sigma^2$")
    print(f"似然梯度Lipschitz常数: $L_f = \\|A^TA\\| / \\sigma^2 = {L_y.item():.2f}$")


# ============================================================
# 步骤3：Tweedie等式与去噪器
# ============================================================
if _has_gpu and _has_sampling_tools and _has_model:
    print("\n" + "=" * 60)
    print("步骤3：Tweedie等式与去噪器")
    print("=" * 60)

    # 加载去噪器
    L_net = 1.0
    model = load_model(_model_path, device)
    denoise = lambda x: (x - model(x[None][None].to(device))[0][0]).detach()

    # Tweedie等式视角
    eps = (5/255)**2
    print(f"\n去噪器训练噪声水平: $\\varepsilon = (5/255)^2 = {eps:.6f}$")
    print("\nTweedie等式:")
    print("  $\\nabla \\log p_\\varepsilon(x) = (D_\\varepsilon(x) - x) / \\varepsilon$")
    print("\nPnP-ULA中的先验得分步:")
    print("  $(\\delta/\\varepsilon)(D_\\varepsilon(x) - x) = \\delta \\cdot \\nabla \\log p_\\varepsilon(x)$")
    print("\n这展示了PnP的核心思想：用去噪器替换先验梯度")


# ============================================================
# 步骤4：PnP-ULA采样
# ============================================================
if _has_gpu and _has_sampling_tools and _has_model:
    print("\n" + "=" * 60)
    print("步骤4：PnP-ULA采样")
    print("=" * 60)

    # 算法参数
    alpha = 1
    max_lambd = 1.0/((2.0*alpha*L_net)/eps+4.0*L_y)
    lambd_frac = 0.99
    lambd = max_lambd*lambd_frac

    C_upper_lim = torch.tensor(1).to(device)
    C_lower_lim = torch.tensor(0).to(device)
    projbox = lambda x: torch.clamp(x, min=C_lower_lim, max=C_upper_lim)

    # PnP-ULA Markov核
    def Markov_kernel(X, delta, projected=True):
        """
        PnP-ULA递推式:
        X_{m+1} = X_m - δ∇f(X_m) + (αδ/ε)(D_ε(X_m)-X_m) + √(2δ)Z

        其中:
        - δ∇f(X_m): 似然梯度步
        - (αδ/ε)(D_ε(X_m)-X_m): 先验得分步（Tweedie替换）
        - √(2δ)Z: 探索噪声
        """
        if projected:
            return projbox(X - delta * gradf(X, A, AT) +
                          alpha*delta/eps*(denoise(X)-X) +
                          math.sqrt(2*delta) * torch.randn_like(X))
        else:
            return X - delta * gradf(X, A, AT) + \
                   alpha*delta/eps*(denoise(X)-X) + \
                   delta/lambd*(projbox(X)-X) + \
                   math.sqrt(2*delta) * torch.randn_like(X)

    # 步长选择
    projected = True
    if projected:
        delta_max = (1.0)/(L_net/eps+L_y)
    else:
        delta_max = (1.0/3.0)/((alpha*L_net)/eps+L_y+1/lambd)
    delta_frac = 0.99
    delta = delta_max*delta_frac

    print(f"步长: $\\delta = {delta:.6f}$ (最大允许: {delta_max:.6f})")
    print(f"步长约束: $\\delta \\leq 1/(L_f + L_D/\\varepsilon)$")

    # 主采样循环
    maxit = 200
    burnin = np.int64(maxit*0.05)
    n_samples = np.int64(40)
    X = y.clone()
    MC_X = []
    thinned_trace_counter = 0
    thinning_step = np.int64(maxit/n_samples)

    nrmse_values = []
    psnr_values = []
    ssim_values = []

    start_time = time.time()
    for i_x in tqdm(range(maxit), desc="PnP-ULA采样"):
        X = Markov_kernel(X, delta, projected=projected)

        if i_x == burnin:
            post_meanvar = welford(X)
            absfouriercoeff = welford(torch.fft.fft2(X).abs())
            count = 0
        elif i_x > burnin:
            post_meanvar.update(X)
            absfouriercoeff.update(torch.fft.fft2(X).abs())

            current_mean = post_meanvar.get_mean()
            nrmse_values.append(NRMSE(x, current_mean))
            psnr_values.append(PSNR(x, current_mean))
            ssim_values.append(SSIM(x, current_mean))

            if count == thinning_step-1:
                MC_X.append(X.detach().cpu().numpy())
                count = 0
            else:
                count += 1

    end_time = time.time()
    elapsed = end_time - start_time

    # 结果评估
    print(f"\n采样耗时: {elapsed:.2f} 秒")
    print(f"\n初始指标:")
    print(f"  NRMSE: {NRMSE(x, y):.4f}")
    print(f"  PSNR: {PSNR(x, y):.2f} dB")
    print(f"  SSIM: {SSIM(x, y):.4f}")

    post_mean = post_meanvar.get_mean()
    print(f"\n后验均值指标:")
    print(f"  NRMSE: {NRMSE(post_mean, x):.4f}")
    print(f"  PSNR: {PSNR(post_mean, x):.2f} dB")
    print(f"  SSIM: {SSIM(x, post_mean):.4f}")

    # 可视化 - 需要手动保存两个图表
    # plots函数会创建两个图表，我们需要分别保存
    
    # 创建第一个图表（2行4列）
    post_mean_numpy = post_meanvar.get_mean().detach().cpu().numpy()
    post_var_numpy = post_meanvar.get_var().detach().cpu().numpy()
    
    fig1, axes1 = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
    fig1.tight_layout(pad=.01)
    
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
    fig2, axes2 = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig2.tight_layout(pad=.01)
    
    axes2[0].plot(np.arange(len(nrmse_values))[::10], nrmse_values[::10], label="-- NRMSE --")
    axes2[0].set_title('NRMSE of $X$ vs $x_{gr}$')
    axes2[0].legend()
    
    axes2[1].plot(np.arange(len(psnr_values))[::10], psnr_values[::10], label="-- PSNR --")
    axes2[1].set_title('PSNR of $X$ vs $x_{gr}$')
    axes2[1].legend()
    
    axes2[2].plot(np.arange(len(ssim_values))[::10], ssim_values[::10], label="-- SSIM --")
    axes2[2].set_title('SSIM of $X$ vs $x_{gr}$')
    axes2[2].legend()
    
    plt.savefig(os.path.join(SAVE_DIR, '步骤4_PnP-ULA结果_曲线.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n可视化结果已保存:")
    print(f"  - {os.path.join(SAVE_DIR, '步骤4_PnP-ULA结果_图像.png')}")
    print(f"  - {os.path.join(SAVE_DIR, '步骤4_PnP-ULA结果_曲线.png')}")

    # 保存采样结果供后续实验使用
    np.savez(os.path.join(SAVE_DIR, 'sampling_results.npz'),
             post_mean=post_mean.cpu().numpy(),
             post_var=post_meanvar.get_var().cpu().numpy(),
             mc_samples=np.array(MC_X),
             x_true=x.cpu().numpy(),
             y_obs=y.cpu().numpy())
    print(f"\n采样结果已保存到: {os.path.join(SAVE_DIR, 'sampling_results.npz')}")
else:
    # 步骤4需要GPU但不可用时的可视化提示
    print("\n" + "=" * 60)
    print("步骤4：PnP-ULA采样")
    print("=" * 60)
    print("  缺少必要资源（GPU、sampling_tools或预训练模型）")
    print("  无法执行PnP-ULA采样")
    
    # 创建第一个提示图表（2行4列，对应plots函数的第一个图表）
    fig1, axes1 = plt.subplots(2, 4, figsize=(15, 10))
    fig1.suptitle('PnP-ULA采样结果 (需要GPU运行)', fontsize=16, fontweight='bold', color='red')
    
    for ax in axes1.flat:
        ax.imshow(np.ones((256, 256)) * 0.9, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    
    # 设置子图标题
    titles_row1 = ['Ground truth image', 'Blurred noisy image', 'x - posterior mean', 'x - posterior variance']
    titles_row2 = ['x - posterior mean/SD', 'x - Coefs of variation', 'Mean coefs (log)', 'Var coefs (log)']
    
    for i, title in enumerate(titles_row1):
        axes1[0, i].set_title(title, fontsize=10)
        axes1[0, i].text(0.5, 0.5, '需要GPU', 
                        ha='center', va='center', transform=axes1[0, i].transAxes,
                        fontsize=12, color='red', fontweight='bold')
    
    for i, title in enumerate(titles_row2):
        axes1[1, i].set_title(title, fontsize=10)
        axes1[1, i].text(0.5, 0.5, '需要GPU', 
                        ha='center', va='center', transform=axes1[1, i].transAxes,
                        fontsize=12, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤4_PnP-ULA结果_图像.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 创建第二个提示图表（1行3列，对应plots函数的第二个图表）
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('收敛曲线 (需要GPU运行)', fontsize=14, fontweight='bold', color='red')
    
    titles_row3 = ['NRMSE of $X$ vs $x_{gr}$', 'PSNR of $X$ vs $x_{gr}$', 'SSIM of $X$ vs $x_{gr}$']
    
    for i, ax in enumerate(axes2):
        ax.text(0.5, 0.5, '需要GPU\n才能显示\n收敛曲线', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2))
        ax.set_title(titles_row3[i], fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤4_PnP-ULA结果_曲线.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n提示图表已保存:")
    print(f"  - {os.path.join(SAVE_DIR, '步骤4_PnP-ULA结果_图像.png')}")
    print(f"  - {os.path.join(SAVE_DIR, '步骤4_PnP-ULA结果_曲线.png')}")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.5-1 总结")
print("=" * 60)
print("1. PnP-ULA核心递推式:")
print("   $X_{m+1} = X_m - \\delta\\nabla f(X_m) + (\\delta/\\varepsilon)[D_\\varepsilon(X_m)-X_m] + \\sqrt{2\\delta}Z$")
print("2. Tweedie等式架起去噪器与先验得分的桥梁:")
print("   $\\nabla \\log p_\\varepsilon(x) = (D_\\varepsilon(x) - x) / \\varepsilon$")
print("3. 步长约束保证收敛:")
print("   $\\delta \\leq 1/(L_f + L_D/\\varepsilon)$")
print("4. 后验采样生成多个样本，可用于不确定性量化（见实验5.6-1/2/3）")
