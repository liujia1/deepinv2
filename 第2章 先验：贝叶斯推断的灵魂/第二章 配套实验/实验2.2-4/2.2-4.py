"""
实验2.2-4 经典先验族对比：高斯、Laplace与TV
对应章节：2.2 经典先验族
知识点：高斯先验→Tikhonov；Laplace先验→LASSO；TV先验→ROF模型；三种先验的解形态对比

素材来源：
  - M1 CompImLab25.ipynb: Tikhonov闭式解 + LASSO软阈值
  - 2.2章节: TV先验与ROF模型

修改说明（基于代码评审）：
  1. Chambolle算法重写为标准形式（零边界差分 + 正确对偶更新公式）
  2. 添加了lambda参数量纲和物理意义注释
  3. 添加了LASSO像素域局限性的教学注释
  4. soft_thresh使用更简洁形式
  5. PSNR显式指定data_range=1.0
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio
import os
import sys

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    SAVE_DIR = os.path.join(_gdrive, '实验2.2-4')
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# ─── 中文字体配置 ───
# 不依赖外部 chinese_font 模块：直接在本机已安装的字体中查找可用的中文字体
# （Windows 常见为 SimHei/微软雅黑；Linux 为 WenQuanYi/Noto Sans CJK）。
# 用 rcParams 设置后，图上中文即可正常渲染，数学符号仍走 matplotlib 内置 mathtext。
import matplotlib.font_manager as fm
_cjk_candidates = [
    'SimHei', 'Microsoft YaHei', 'Microsoft YaHei UI', 'WenQuanYi Micro Hei',
    'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
    'Source Han Sans SC', 'STHeiti', 'Arial Unicode MS',
]
_available = {f.name for f in fm.fontManager.ttflist}
_chinese_font = next((name for name in _cjk_candidates if name in _available), None)
if _chinese_font is not None:
    plt.rcParams['font.family'] = _chinese_font
    plt.rcParams['font.sans-serif'] = [_chinese_font]
    plt.rcParams['mathtext.fontset'] = 'stix'   # 数学符号用 stix，减号正常
    plt.rcParams['axes.unicode_minus'] = False  # 普通文本负号用 ASCII '-'
    print(f"已启用中文字体: {_chinese_font}")
else:
    print("警告: 未在本机找到中文字体，中文可能显示为方框。请安装 SimHei/Noto Sans CJK 等中文字体。")

np.random.seed(42)

# ─── 实验参数配置 ───
N_PIXELS    = 128
NOISE_LEVEL = 0.1
DATA_RANGE  = 1.0  # camera() 经 resize 后值域为 [0,1]
TV_ITERS    = 200
# PSNR-lambda曲线扫描参数
N_LAMBDA   = 30       # 扫描点数
TV_ITERS_SWEEP = 100  # 扫描时减少迭代以加速

n = N_PIXELS
x = resize(data.camera(), (n, n))

y = x + NOISE_LEVEL * np.random.randn(n, n)

def soft_thresh(x, l):
    """软阈值函数：Laplace先验的MAP解"""
    return np.sign(x) * np.maximum(np.abs(x) - l, 0)

def tv_denoise_chambolle(y, lam, n_iter=100):
    """
    Chambolle对偶投影算法求解TV去噪（ROF模型）
    
    优化问题: min_x 0.5||x-y||^2 + lam * TV(x)
    
    算法原理 (Chambolle 2004, IPOL 2013):
      将原问题转化为对偶问题，通过对偶变量的梯度投影迭代求解。
      离散梯度采用前向差分（零边界），散度采用后向差分（零边界），
      满足伴随关系 ⟨-div(p), u⟩ = ⟨p, grad u⟩。
    
      最终解: x = y - lam * div(p*)
      其中 p* 是对偶问题在最优点处的解。
    
    数值稳定性:
      时间步长 dt < 1/(2*ndim) = 1/4 (二维)，取 dt=0.249。
    
    边界处理:
      使用零填充边界（而非周期边界），避免非周期性图像（如cameraman）
      在边缘产生振铃伪影。
    
    参考:
      Chambolle, J. Math. Imaging Vision, 20:89-97, 2004
      Duran et al., IPOL, 2013 (https://www.ipol.im/pub/art/2013/61/)
"""
    h, w = y.shape
    # p_row: 行方向(垂直)分量, 沿 axis=0 做后向差分
    # p_col: 列方向(水平)分量, 沿 axis=1 做后向差分
    p_row = np.zeros((h, w))
    p_col = np.zeros((h, w))
    dt = 0.249                 # 时间步长，略小于 1/4 保证收敛
    
    for _ in range(n_iter):
        # ── 散度 div(p): 后向差分（零边界）──
        # div(p)[i,j] = p_row[i,j] - p_row[i-1,j] + p_col[i,j] - p_col[i,j-1]
        # 边界处 p_row[-1,j] = p_col[i,-1] = 0
        div_p = np.zeros((h, w))
        div_p[0, :] = p_row[0, :]                     # i=0: p_row[0,j]
        div_p[1:, :] = p_row[1:, :] - p_row[:-1, :]   # i>=1: 沿行方向(axis=0)
        div_p[:, 0] += p_col[:, 0]                     # j=0: p_col[i,0]
        div_p[:, 1:] += p_col[:, 1:] - p_col[:, :-1]   # j>=1: 沿列方向(axis=1)
        
        # ── 当前原始变量估计 ──
        u = y - lam * div_p
        
        # ── 梯度 ∇u: 前向差分（零边界）──
        # grad_row[i,j] = u[i+1,j] - u[i,j] (i < h-1), 沿行方向(垂直), axis=0
        # grad_col[i,j] = u[i,j+1] - u[i,j] (j < w-1), 沿列方向(水平), axis=1
        grad_row = np.zeros((h, w))
        grad_row[:-1, :] = u[1:, :] - u[:-1, :]
        grad_col = np.zeros((h, w))
        grad_col[:, :-1] = u[:, 1:] - u[:, :-1]
        
        # ── 标准Chambolle对偶更新 ──
        # IPOL论文 (2013) Algorithm 1:
        #   p ← (p + δt·D(div p - λ_paper·y)) / (1 + δt·|D(div p - λ_paper·y)|)
        # 其中 λ_paper (数据项系数) = 1/λ (正则项系数, 即代码中的lam)
        # 代入: D(div p - y/lam) = -(∇u)/lam, 故:
        #   p ← (p - (δt/lam)·∇u) / (1 + (δt/lam)·|∇u|)
        # 等价于梯度上升（对偶最大化），只是展开后出现负号。
        grad_norm = np.sqrt(grad_row**2 + grad_col**2)
        factor = dt / lam
        denom = 1.0 + factor * grad_norm
        p_row = (p_row - factor * grad_row) / denom   # 沿行方向(垂直)分量
        p_col = (p_col - factor * grad_col) / denom   # 沿列方向(水平)分量
    
    # ── 最终重建 ──
    div_p = np.zeros((h, w))
    div_p[0, :] = p_row[0, :]
    div_p[1:, :] = p_row[1:, :] - p_row[:-1, :]
    div_p[:, 0] += p_col[:, 0]
    div_p[:, 1:] += p_col[:, 1:] - p_col[:, :-1]
    
    return y - lam * div_p


# ══════════════════════════════════════════════════════════
# 超参数设置与说明
# ══════════════════════════════════════════════════════════

sigma = NOISE_LEVEL
sigma_x = 1.0
b_laplace = 0.5

lam_tikh = sigma**2 / sigma_x**2      # ~ 0.01
lam_lasso = sigma**2 / b_laplace      # ~ 0.02
lam_tv = 0.15                          # 手动调参

# ── 参数量纲说明 ──
# Tikhonov lam = sigma^2/sigma_x^2: 作用于像素域, ||x||^2 正则项
# LASSO   lam = sigma^2/b:          作用于像素域, ||x||_1 正则项
# TV      lam = 0.15:               作用于梯度域, ||grad x||_1 正则项
# 注意: 不同正则项的量纲不同（像素值 vs 梯度值），
#       lam 的数值大小不可直接对比。
#       此处TV lam 为手动调参结果，可通过PSNR-lambda曲线交叉验证。
#       在实践中，建议对每种方法分别做网格搜索选取最优lam。

x_tikh = y / (1 + lam_tikh)
x_lasso = soft_thresh(y, lam_lasso)
x_tv = tv_denoise_chambolle(y, lam_tv, n_iter=TV_ITERS)

psnr_noisy = peak_signal_noise_ratio(x, y, data_range=DATA_RANGE)
psnr_tikh = peak_signal_noise_ratio(x, x_tikh, data_range=DATA_RANGE)
psnr_lasso = peak_signal_noise_ratio(x, x_lasso, data_range=DATA_RANGE)
psnr_tv = peak_signal_noise_ratio(x, x_tv, data_range=DATA_RANGE)

print("===== 经典先验族对比 =====")
print(f"噪声水平 sigma = {sigma:.4f}")
print(f"\n高斯先验 (假设: 值小):")
print(f"  正则项: ||x||_2^2 -> Tikhonov")
print(f"  lam = sigma^2/sigma_x^2 = {lam_tikh:.4f}")
print(f"  PSNR = {psnr_tikh:.4f} dB")
print(f"\nLaplace先验 (假设: 值稀疏):")
print(f"  正则项: ||x||_1 -> LASSO")
print(f"  lam = sigma^2/b = {lam_lasso:.4f}")
print(f"  PSNR = {psnr_lasso:.4f} dB")
print(f"\nTV先验 (假设: 梯度稀疏):")
print(f"  正则项: ||grad x||_1 -> ROF模型")
print(f"  lam = {lam_tv:.4f} (手动调参)")
print(f"  PSNR = {psnr_tv:.4f} dB")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

axes[0, 0].imshow(x, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(y, cmap='gray')
axes[0, 1].set_title(f'含噪图像\nPSNR={psnr_noisy:.4f}dB')
axes[0, 1].axis('off')

axes[0, 2].imshow(x_tikh, cmap='gray')
axes[0, 2].set_title(f'Tikhonov (高斯先验)\n假设: 值小\nPSNR={psnr_tikh:.4f}dB')
axes[0, 2].axis('off')

axes[1, 0].imshow(x_lasso, cmap='gray')
axes[1, 0].set_title(f'LASSO (Laplace先验)\n假设: 值稀疏\nPSNR={psnr_lasso:.4f}dB')
axes[1, 0].axis('off')

axes[1, 1].imshow(x_tv, cmap='gray')
axes[1, 1].set_title(f'TV去噪 (TV先验)\n假设: 梯度稀疏\nPSNR={psnr_tv:.4f}dB')
axes[1, 1].axis('off')

center = n // 2
axes[1, 2].plot(x[center, :], 'k--', linewidth=1.5, label='真实')
axes[1, 2].plot(x_tikh[center, :], 'b-', linewidth=1, label='Tikhonov')
axes[1, 2].plot(x_lasso[center, :], 'g-', linewidth=1, label='LASSO')
axes[1, 2].plot(x_tv[center, :], 'r-', linewidth=1, label='TV')
axes[1, 2].set_title('中心行剖面对比')
axes[1, 2].legend()
axes[1, 2].set_xlabel('像素索引')

plt.suptitle('经典先验族对比：不同假设→不同正则项→不同解形态', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_三种先验对比.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

t = np.linspace(-2, 2, 400)
axes[0].plot(t, t**2, 'b-', linewidth=2, label=r'$L^2$: $t^2$ (高斯先验)')
axes[0].plot(t, np.abs(t), 'g-', linewidth=2, label=r'$L^1$: $|t|$ (Laplace先验)')
axes[0].set_title('正则项形态对比')
axes[0].legend()
axes[0].set_xlabel('t')
axes[0].set_ylabel('惩罚值')
axes[0].set_ylim(-0.2, 4)
axes[0].grid(True, alpha=0.3)

# PSNR-lambda曲线扫描
# 注意: LASSO在像素域直接应用对自然图像效果有限，此处仅作教学对比
lambdas = np.logspace(-4, 1, N_LAMBDA)
psnr_tikh_list = []
psnr_lasso_list = []
psnr_tv_list = []

for lam in lambdas:
    psnr_tikh_list.append(peak_signal_noise_ratio(x, y / (1 + lam), data_range=DATA_RANGE))
    psnr_lasso_list.append(peak_signal_noise_ratio(x, soft_thresh(y, lam), data_range=DATA_RANGE))
    psnr_tv_list.append(peak_signal_noise_ratio(x, tv_denoise_chambolle(y, lam, n_iter=TV_ITERS_SWEEP), data_range=DATA_RANGE))

axes[1].semilogx(lambdas, psnr_tikh_list, 'b-o', markersize=3, label='Tikhonov')
axes[1].semilogx(lambdas, psnr_lasso_list, 'g-s', markersize=3, label='LASSO')
axes[1].semilogx(lambdas, psnr_tv_list, 'r-^', markersize=3, label='TV')
axes[1].axhline(y=psnr_noisy, color='k', linestyle='--', alpha=0.5, label='含噪')
axes[1].set_xlabel(r'$\lambda$')
axes[1].set_ylabel('PSNR (dB)')
axes[1].set_title(r'不同先验的 PSNR-$\lambda$ 曲线')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

res_tikh = np.abs(x_tikh - x)
res_lasso = np.abs(x_lasso - x)
res_tv = np.abs(x_tv - x)

axes[2].bar(['Tikhonov\n(高斯)', 'LASSO\n(Laplace)', 'TV\n(梯度稀疏)'],
            [np.mean(res_tikh**2), np.mean(res_lasso**2), np.mean(res_tv**2)],
            color=['blue', 'green', 'red'], alpha=0.7)
axes[2].set_ylabel('MSE')
axes[2].set_title('重建误差对比')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_正则项形态与性能对比.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n===== 三种先验的特点总结 =====")
print("\n高斯先验 (Tikhonov):")
print("  - 假设: 图像值小")
print("  - 正则项: ||x||_2^2")
print("  - 优点: 有闭式解, 计算高效")
print("  - 缺点: 过度平滑, 丢失边缘")
print("\nLaplace先验 (LASSO):")
print("  - 假设: 图像值稀疏")
print("  - 正则项: ||x||_1")
print("  - 优点: 促稀疏, 保留显著特征")
print("  - 缺点: 自然图像像素不满足稀疏假设, 直接应用效果有限")
print("          实际应用中通常在小波/DCT变换域使用LASSO")
print("\nTV先验 (ROF模型):")
print("  - 假设: 图像梯度稀疏")
print("  - 正则项: ||grad x||_1")
print("  - 优点: 同时平滑与保边")
print("  - 缺点: 可能产生阶梯效应")
print("\n注: 三种方法的lambda参数量纲不同, PSNR数值仅在同一方法内可比。")

# ===== 保存数值结果 =====
import json
results_summary = {
    'image_size': n,
    'noise_level': float(NOISE_LEVEL),
    'psnr_noisy_dB': float(round(psnr_noisy, 2)),
    'psnr_tikhonov_dB': float(round(psnr_tikh, 2)),
    'psnr_lasso_dB': float(round(psnr_lasso, 2)),
    'psnr_tv_dB': float(round(psnr_tv, 2)),
    'lambda_tikhonov': float(round(lam_tikh, 6)),
    'lambda_lasso': float(round(lam_lasso, 6)),
    'lambda_tv': float(round(lam_tv, 4)),
    'mse_tikhonov': float(round(np.mean((x_tikh - x)**2), 6)),
    'mse_lasso': float(round(np.mean((x_lasso - x)**2), 6)),
    'mse_tv': float(round(np.mean((x_tv - x)**2), 6)),
}

def _to_native(obj):
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

results_summary = {k: _to_native(v) for k, v in results_summary.items()}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")