"""
实验4.7-1 PnP-ULA后验采样与收敛诊断
对应章节：4.7 收敛诊断与不确定性量化
知识点：
  - Burn-in期：丢弃未收敛样本
  - 后验均值/方差：不确定性量化
  - 自相关函数：评估样本质量
  - 有效样本量（ESS）：量化信息量
  - PnP-ULA：用预训练去噪器替代显式先验

素材来源：Mathematics.../Teaching Unit 2/labs/lab2_PnP_sol.ipynb

PnP-ULA核心思想：
  传统ULA需要显式定义势能函数 E(x) = -log p(x|y)
  当先验 p(x) 难以解析表达时（如深度学习先验），可用去噪器 D(x) 隐式定义：
  
  ULA迭代：X_{m+1} = X_m - δ∇f(X_m) + δ/ε(D(X_m) - X_m) + √(2δ) N_m
  
  其中：
  - f(x) = ||y - Ax||²/(2σ²) 是数据保真项
  - D(x) 是预训练去噪器（如DnCNN）
  - ε 是正则化参数
  - D(x) - x 近似 ∇(-log p(x))，即得分函数
  
  这就是"Plug-and-Play"思想：将去噪器"插入"采样算法，无需显式定义先验。
"""

import math
import torch
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验4.7-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
else:
    _chinese_path = '.chinese'
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")

SCRIPT_DIR = SAVE_DIR
sys.path.insert(0, SCRIPT_DIR)

# ══════════════════════════════════════════════════════════
# GPU 检测
# ══════════════════════════════════════════════════════════
_has_gpu = torch.cuda.is_available()
if _has_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Device: {device}")

# Colab环境GPU提示
if _IN_COLAB and not _has_gpu:
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

# ══════════════════════════════════════════════════════════
# 检查运行条件与算法策略
# ══════════════════════════════════════════════════════════
_model_dir = os.path.join(SCRIPT_DIR, 'Pretrained_models')
_model_path = os.path.join(_model_dir, 'RealSN_DnCNN_noise5.pth')
_has_sampling_tools = True
try:
    from sampling_tools import *
except ImportError:
    _has_sampling_tools = False

# 确定算法策略：优先 PnP-ULA（GPU+模型），备选 TV 正则化（CPU/无模型）
_use_pnp = _has_gpu and os.path.exists(_model_path) and _has_sampling_tools

print("\n" + "=" * 60)
print("【环境检查】")
print("=" * 60)
print(f"  GPU可用: {_has_gpu}")
print(f"  sampling_tools: {'可用' if _has_sampling_tools else '不可用'}")
print(f"  预训练模型: {'存在' if os.path.exists(_model_path) else '不存在'}")

if _use_pnp:
    print(f"\n  使用算法: PnP-ULA（GPU + 预训练 DnCNN 去噪器）")
    print(f"  PnP-ULA 是实验的完整实现，将展示深度学习先验的力量。")
else:
    print(f"\n  [警告] 缺少 PnP-ULA 运行条件（GPU 或预训练模型或 sampling_tools）")
    print(f"  将使用 TV 正则化替代方案继续运行。")
    if not _has_gpu:
        print(f"    - 缺少GPU: DnCNN 推理在CPU上极慢，建议启用GPU")
    if not os.path.exists(_model_path):
        print(f"    - 缺少模型: {os.path.dirname(_model_path)}")
        print(f"      可从 https://github.com/uclaopt/Provable_Plug_and_Play/ 下载")
    if not _has_sampling_tools:
        print(f"    - 缺少 sampling_tools 模块")
    print(f"  注意: TV 正则化是传统手工先验，与 PnP 的深度学习先验效果不同。")
    print(f"  获得完整实验体验需要 GPU + 预训练 RealSN-DnCNN 模型。")

# ══════════════════════════════════════════════════════════
# 加载图像
# ══════════════════════════════════════════════════════════
from PIL import Image

im_path = os.path.join(SCRIPT_DIR, "cman.png")
if not os.path.exists(im_path):
    print(f"错误: 图像文件 {im_path} 不存在")
    print("请从参考实验目录复制 cman.png 到当前目录")
    sys.exit(1)

im = np.array(Image.open(im_path))
x = torch.Tensor(im / 255.).to(device)

print("\n" + "=" * 60)
print("【实验设置】")
print("=" * 60)
print(f"图像尺寸: {im.shape[0]} × {im.shape[1]}")
print(f"设备: {device}")
print(f"算法策略: {'PnP-ULA (深度学习去噪器先验)' if _use_pnp else 'ULA + TV 正则化 (手工先验替代)'}")

# ══════════════════════════════════════════════════════════
# 模糊算子（前向模型 A）
# ══════════════════════════════════════════════════════════
kernel_len = [5, 5]
size = [im.shape[0], im.shape[1]]
type_blur = "uniform"
A, AT, AAT_norm = blur_operators(kernel_len, size, type_blur, device)

# ══════════════════════════════════════════════════════════
# 含噪观测
# ══════════════════════════════════════════════════════════
y0 = A(x)
BSNRdb = 40
sigma = torch.linalg.matrix_norm(A(x) - torch.mean(A(x)), ord='fro') / math.sqrt(torch.numel(x) * 10 ** (BSNRdb / 10))
y = y0 + sigma * torch.randn_like(x)

print(f"模糊核: {type_blur} {kernel_len}")
print(f"噪声标准差: σ = {sigma.item():.4f}")
print(f"BSNR: {BSNRdb} dB")

# ══════════════════════════════════════════════════════════
# 似然函数与梯度
# ══════════════════════════════════════════════════════════
f = lambda x, A: (torch.linalg.matrix_norm(y - A(x), ord='fro') ** 2.0) / (2.0 * sigma ** 2)
gradf = lambda x, A, AT: AT(A(x) - y) / sigma ** 2
L_y = AAT_norm / (sigma ** 2)

print(f"似然Lipschitz常数: L_y = {L_y:.4f}")

# ══════════════════════════════════════════════════════════
# 先验项配置
# ══════════════════════════════════════════════════════════
alpha = 1
eps = (5 / 255) ** 2
C_upper_lim = torch.tensor(1).to(device)
C_lower_lim = torch.tensor(0).to(device)
projbox = lambda x: torch.clamp(x, min=C_lower_lim, max=C_upper_lim)

if _use_pnp:
    # ---- PnP-ULA: 深度学习去噪器先验 ----
    print(f"\n[先验配置] PnP-ULA: RealSN-DnCNN 去噪器")
    L_net = 1.0
    model = load_model(_model_path, device)
    denoise = lambda x: (x - model(x[None][None].to(device))[0][0]).detach()
    max_lambd = 1.0 / ((2.0 * alpha * L_net) / eps + 4.0 * L_y)
    lambd = max_lambd * 0.99
    delta_max = (1.0) / (L_net / eps + L_y)
else:
    # ---- TV 正则化: 手工先验（替代方案） ----
    print(f"\n[先验配置] ULA + TV 正则化 (Chambolle 近端算子)")
    print(f"  说明: TV 是传统手工先验，不同于 PnP 的深度学习先验")
    print(f"  参数: delta_max 设为 L_y 的 1/L_y（不含去噪器项）")
    lambd = 1.0 / (2.0 * L_y)  # TV 正则化参数
    delta_max = 1.0 / L_y

delta_frac = 0.99
delta = delta_max * delta_frac

print(f"正则化参数: ε = {eps:.6f}")
print(f"步长: δ = {delta:.6f} (δ_max = {delta_max:.6f})")

# ══════════════════════════════════════════════════════════
# PnP-ULA / TV-ULA Markov核
# ══════════════════════════════════════════════════════════
def Markov_kernel(X, delta):
    """
    PnP-ULA / TV-ULA 迭代（根据 _use_pnp 选择）
    
    PnP-ULA:
      X_{m+1} = X_m - δ∇f(X_m) + δ/ε(D(X_m) - X_m) + √(2δ) N_m
      其中 D(X_m) - X_m 是去噪器残差，近似得分函数 ∇(-log p(X_m))
    
    TV-ULA (替代方案):
      X_{m+1} = X_m - δ∇f(X_m) + δ/λ·prox_TV(X_m) + √(2δ) N_m
      其中 prox_TV 是 Chambolle 近端算子，实现 TV 先验
    """
    noise = math.sqrt(2 * delta) * torch.randn_like(X)
    grad_data = gradf(X, A, AT)
    
    if _use_pnp:
        # PnP 先验: 去噪器残差
        prior_term = alpha * delta / eps * (denoise(X) - X)
    else:
        # TV 先验: Chambolle 近端算子
        prior_term = delta / lambd * (projbox(X) - X)
    
    return X - delta * grad_data + prior_term + noise

# ══════════════════════════════════════════════════════════
# 主采样循环
# ══════════════════════════════════════════════════════════
maxit = 500
burnin = int(maxit * 0.1)
n_samples = int(50)
X = y.clone()
MC_X = []
thinned_trace_counter = 0
thinning_step = max(1, int((maxit - burnin) / n_samples))

nrmse_values = []
psnr_values = []
ssim_values = []

print("\n" + "=" * 60)
print("【采样】" + ("PnP-ULA" if _use_pnp else "TV-ULA (替代方案)"))
print("=" * 60)
print(f"总迭代: {maxit}")
print(f"Burn-in: {burnin} ({burnin/maxit*100:.0f}%)")
print(f"采样数: {n_samples}")

start_time = time.time()
for i_x in range(maxit):
    X = Markov_kernel(X, delta)
    
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
        
        if count == thinning_step - 1:
            MC_X.append(X.detach().cpu().numpy())
            count = 0
        else:
            count += 1
    
    if (i_x + 1) % 100 == 0:
        print(f"  iter {i_x+1}/{maxit}", end="\r")

end_time = time.time()
elapsed = end_time - start_time
print(f"\n  采样完成，耗时 {elapsed:.2f} 秒")

# ══════════════════════════════════════════════════════════
# 结果输出
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【重建质量】")
print("=" * 60)
print(f"初始 NRMSE: {NRMSE(x, y):.4f}")
print(f"初始 PSNR:  {PSNR(x, y):.2f} dB")
print(f"初始 SSIM:  {SSIM(x, y):.4f}")
print()
print(f"结果 NRMSE: {NRMSE(post_meanvar.get_mean(), x):.4f}")
print(f"结果 PSNR:  {PSNR(post_meanvar.get_mean(), x):.2f} dB")
print(f"结果 SSIM:  {SSIM(x, post_meanvar.get_mean()):.4f}")

# ══════════════════════════════════════════════════════════
# 自相关函数与ESS
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【收敛诊断】")
print("=" * 60)

if len(MC_X) > 10:
    MC_X_array = np.array(MC_X)
    print(f"样本链长度: {len(MC_X)}")
    
    try:
        from statsmodels.graphics.tsaplots import plot_acf
        import arviz
        _HAS_STATS = True
    except ImportError:
        _HAS_STATS = False
        print("警告: 需要安装 statsmodels 和 arviz 以计算自相关和ESS")
        print("  pip install statsmodels arviz")
    
    if _HAS_STATS:
        X_chain_vec = MC_X_array.reshape(len(MC_X), -1)
        var_sp = np.var(X_chain_vec, axis=0)
        
        trace_slow = X_chain_vec[:, np.argmax(var_sp)]
        trace_fast = X_chain_vec[:, np.argmin(var_sp)]
        trace_med = X_chain_vec[:, np.argsort(var_sp)[len(var_sp)//2]]
        
        ess_slow = arviz.ess(trace_slow)
        ess_fast = arviz.ess(trace_fast)
        ess_med = arviz.ess(trace_med)
        
        print(f"有效样本量 (ESS):")
        print(f"  最慢分量: {ess_slow:.2f} / {len(MC_X)} ({ess_slow/len(MC_X)*100:.1f}%)")
        print(f"  中速分量: {ess_med:.2f} / {len(MC_X)} ({ess_med/len(MC_X)*100:.1f}%)")
        print(f"  最快分量: {ess_fast:.2f} / {len(MC_X)} ({ess_fast/len(MC_X)*100:.1f}%)")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        nLags = min(50, len(MC_X) - 1)
        
        plot_acf(trace_med, ax=axes[0], lags=nLags, alpha=None)
        axes[0].set_title(rf'中速分量 ACF (ESS={ess_med:.1f})', fontsize=11)
        axes[0].set_xlabel(r'滞后 $k$', fontsize=10)
        axes[0].set_ylabel(r'自相关 $\rho_k$', fontsize=10)
        
        plot_acf(trace_slow, ax=axes[1], lags=nLags, alpha=None)
        axes[1].set_title(rf'最慢分量 ACF (ESS={ess_slow:.1f})', fontsize=11)
        axes[1].set_xlabel(r'滞后 $k$', fontsize=10)
        
        plot_acf(trace_fast, ax=axes[2], lags=nLags, alpha=None)
        axes[2].set_title(rf'最快分量 ACF (ESS={ess_fast:.1f})', fontsize=11)
        axes[2].set_xlabel(r'滞后 $k$', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, '实验4_7-1_自相关函数.png'), dpi=150, bbox_inches='tight')
        print(f"\n自相关图已保存: {os.path.join(SAVE_DIR, '实验4_7-1_自相关函数.png')}")
else:
    print("样本数不足，跳过自相关分析")

# ══════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════
post_mean = post_meanvar.get_mean().detach().cpu().numpy()
post_var = post_meanvar.get_var().detach().cpu().numpy()
post_std = np.sqrt(post_var)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(x.detach().cpu().numpy(), cmap='gray')
axes[0, 0].set_title(r'真值', fontsize=11)
axes[0, 0].axis('off')

axes[0, 1].imshow(y.detach().cpu().numpy(), cmap='gray')
axes[0, 1].set_title(r'含噪观测 $y$', fontsize=11)
axes[0, 1].axis('off')

axes[0, 2].imshow(post_mean, cmap='gray')
if _use_pnp:
    axes[0, 2].set_title(r'后验均值 (MMSE)', fontsize=11)
else:
    axes[0, 2].set_title(r'后验均值 (MMSE, TV 先验)', fontsize=11)
axes[0, 2].axis('off')

axes[0, 3].imshow(post_std, cmap='hot')
axes[0, 3].set_title(r'后验标准差 $\sigma$', fontsize=11)
axes[0, 3].axis('off')

axes[1, 0].imshow(post_mean / (post_std + 1e-8), cmap='gray')
axes[1, 0].set_title(r'信噪比 $\mu/\sigma$', fontsize=11)
axes[1, 0].axis('off')

axes[1, 1].imshow(post_std / (post_mean + 1e-8), cmap='gray')
axes[1, 1].set_title(r'变异系数 $\sigma/\mu$', fontsize=11)
axes[1, 1].axis('off')

if len(psnr_values) > 0:
    axes[1, 2].plot(psnr_values)
    axes[1, 2].set_title(r'PSNR 收敛曲线', fontsize=11)
    axes[1, 2].set_xlabel(r'迭代', fontsize=10)
    axes[1, 2].set_ylabel(r'PSNR (dB)', fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)

if len(ssim_values) > 0:
    axes[1, 3].plot(ssim_values)
    axes[1, 3].set_title(r'SSIM 收敛曲线', fontsize=11)
    axes[1, 3].set_xlabel(r'迭代', fontsize=10)
    axes[1, 3].set_ylabel(r'SSIM', fontsize=10)
    axes[1, 3].grid(True, alpha=0.3)

if _use_pnp:
    fig.suptitle(r'实验4.7-1 PnP-ULA后验采样与收敛诊断', fontsize=14, y=1.02)
else:
    fig.suptitle(r'实验4.7-1 TV-ULA后验采样（替代方案）', fontsize=14, y=1.02, color='orange')
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, '实验4_7-1_PnP-ULA采样.png'), dpi=150, bbox_inches='tight')
print(f"\n结果图已保存: {os.path.join(SAVE_DIR, '实验4_7-1_PnP-ULA采样.png')}")

# ══════════════════════════════════════════════════════════
# 核心发现
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【核心发现】")
print("=" * 60)

if _use_pnp:
    print("1. PnP-ULA核心思想：")
    print("   用预训练去噪器 D(x) 替代显式先验 p(x)")
    print("   迭代：X_{m+1} = X_m - δ∇f(X_m) + δ/ε(D(X_m)-X_m) + √(2δ)N_m")
    print("   其中 D(x)-x 近似得分函数 ∇(-log p(x))")
else:
    print("1. 替代方案说明：")
    print("   当前使用 TV 正则化替代 PnP-ULA（深度学习先验）")
    print("   迭代：X_{m+1} = X_m - delta*grad_f(X_m) + delta/lambd*prox_TV(X_m) + sqrt(2*delta)*N_m")
    print("   TV 是传统手工先验，效果不同于 PnP 的深度学习先验")
    print("   获得完整 PnP-ULA 体验需要 GPU + 预训练 RealSN-DnCNN 模型")
print("")
print("2. Burn-in期：")
print(f"   丢弃前 {burnin} 个样本（{burnin/maxit*100:.0f}%），确保链收敛到平稳分布")
print("")
print("3. 不确定性量化：")
print("   后验标准差图显示：边缘处不确定性高，平坦区域不确定性低")
print("   变异系数图显示：暗区相对不确定性更高")
print("")
print("4. 收敛诊断：")
if _HAS_STATS and len(MC_X) > 10:
    print(f"   ESS/M 比率：{ess_med/len(MC_X)*100:.1f}%（中速分量）")
    print("   ESS > 100 通常足够估计后验均值")
else:
    print("   需要更多样本进行可靠的ESS估计")
