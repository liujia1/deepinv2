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

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")

SCRIPT_DIR = SAVE_DIR
sys.path.insert(0, SCRIPT_DIR)

from sampling_tools import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════
# 1. 加载图像
# ══════════════════════════════════════════════════════════
from PIL import Image

im_path = os.path.join(SCRIPT_DIR, "cman.png")
if not os.path.exists(im_path):
    print(f"错误: 图像文件 {im_path} 不存在")
    print("请从参考实验目录复制 cman.png 到当前目录")
    sys.exit(1)

im = np.array(Image.open(im_path))
x = torch.Tensor(im / 255.).to(device)

print("=" * 60)
print("【实验设置】")
print("=" * 60)
print(f"图像尺寸: {im.shape[0]} × {im.shape[1]}")
print(f"设备: {device}")

# ══════════════════════════════════════════════════════════
# 2. 模糊算子（前向模型 A）
# ══════════════════════════════════════════════════════════
kernel_len = [5, 5]
size = [im.shape[0], im.shape[1]]
type_blur = "uniform"
A, AT, AAT_norm = blur_operators(kernel_len, size, type_blur, device)

# ══════════════════════════════════════════════════════════
# 3. 含噪观测
# ══════════════════════════════════════════════════════════
y0 = A(x)
BSNRdb = 40
sigma = torch.linalg.matrix_norm(A(x) - torch.mean(A(x)), ord='fro') / math.sqrt(torch.numel(x) * 10 ** (BSNRdb / 10))
y = y0 + sigma * torch.randn_like(x)

print(f"模糊核: {type_blur} {kernel_len}")
print(f"噪声标准差: σ = {sigma.item():.4f}")
print(f"BSNR: {BSNRdb} dB")

# ══════════════════════════════════════════════════════════
# 4. 似然函数与梯度
# ══════════════════════════════════════════════════════════
f = lambda x, A: (torch.linalg.matrix_norm(y - A(x), ord='fro') ** 2.0) / (2.0 * sigma ** 2)
gradf = lambda x, A, AT: AT(A(x) - y) / sigma ** 2
L_y = AAT_norm / (sigma ** 2)

print(f"似然Lipschitz常数: L_y = {L_y:.4f}")

# ══════════════════════════════════════════════════════════
# 5. 加载去噪器（PnP核心）
# ══════════════════════════════════════════════════════════
model_path = os.path.join(SCRIPT_DIR, 'Pretrained_models', 'RealSN_DnCNN_noise5.pth')
has_model = os.path.exists(model_path)

if has_model:
    L_net = 1.0
    model = load_model(model_path, device)
    denoise = lambda x: (x - model(x[None][None].to(device))[0][0]).detach()
    print(f"去噪器: RealSN-DnCNN (已加载)")
    use_pnp = True
else:
    print("警告: 预训练模型不存在，将使用简化的TV正则化")
    print(f"  模型路径: {model_path}")
    print("  可从 https://github.com/uclaopt/Provable_Plug_and_Play/ 下载")
    use_pnp = False

# ══════════════════════════════════════════════════════════
# 6. 算法参数
# ══════════════════════════════════════════════════════════
alpha = 1
eps = (5 / 255) ** 2

if use_pnp:
    max_lambd = 1.0 / ((2.0 * alpha * L_net) / eps + 4.0 * L_y)
    lambd_frac = 0.99
    lambd = max_lambd * lambd_frac
    delta_max = (1.0) / (L_net / eps + L_y)
else:
    delta_max = 1.0 / (L_y + 1.0)

delta_frac = 0.99
delta = delta_max * delta_frac

C_upper_lim = torch.tensor(1).to(device)
C_lower_lim = torch.tensor(0).to(device)

print(f"正则化参数: ε = {eps:.6f}")
print(f"步长: δ = {delta:.6f} (δ_max = {delta_max:.6f})")

# ══════════════════════════════════════════════════════════
# 7. PnP-ULA Markov核
# ══════════════════════════════════════════════════════════
projbox = lambda x: torch.clamp(x, min=C_lower_lim, max=C_upper_lim)

def Markov_kernel(X, delta, use_pnp=True):
    """
    PnP-ULA迭代：
    X_{m+1} = X_m - δ∇f(X_m) + δ/ε(D(X_m) - X_m) + √(2δ) N_m
    
    其中：
    - ∇f(X_m) = A^T(A(X_m) - y)/σ² 是似然梯度
    - D(X_m) - X_m 是去噪器残差，近似得分函数 ∇(-log p(X_m))
    - √(2δ) N_m 是扩散噪声
    """
    noise = math.sqrt(2 * delta) * torch.randn_like(X)
    grad_data = gradf(X, A, AT)
    
    if use_pnp:
        prior_term = alpha * delta / eps * (denoise(X) - X)
    else:
        prior_term = delta / lambd * (projbox(X) - X) if 'lambd' in dir() else torch.zeros_like(X)
    
    return X - delta * grad_data + prior_term + noise

# ══════════════════════════════════════════════════════════
# 8. 主采样循环
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
print("【PnP-ULA采样】")
print("=" * 60)
print(f"总迭代: {maxit}")
print(f"Burn-in: {burnin} ({burnin/maxit*100:.0f}%)")
print(f"采样数: {n_samples}")

start_time = time.time()
for i_x in range(maxit):
    X = Markov_kernel(X, delta, use_pnp=use_pnp)
    
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
# 9. 结果输出
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
# 10. 自相关函数与ESS
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
# 11. 可视化
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
axes[0, 2].set_title(r'后验均值 (MMSE)', fontsize=11)
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

fig.suptitle(r'实验4.7-1 PnP-ULA后验采样与收敛诊断', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验4_7-1_PnP-ULA采样.png'), dpi=150, bbox_inches='tight')
print(f"\n结果图已保存: {os.path.join(SAVE_DIR, '实验4_7-1_PnP-ULA采样.png')}")

# ══════════════════════════════════════════════════════════
# 12. 核心发现
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【核心发现】")
print("=" * 60)
print("1. PnP-ULA核心思想：")
print("   用预训练去噪器 D(x) 替代显式先验 p(x)")
print("   迭代：X_{m+1} = X_m - δ∇f(X_m) + δ/ε(D(X_m)-X_m) + √(2δ)N_m")
print("   其中 D(x)-x 近似得分函数 ∇(-log p(x))")
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
