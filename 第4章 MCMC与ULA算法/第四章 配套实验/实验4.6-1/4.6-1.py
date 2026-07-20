"""
实验4.6-1 优化 vs 采样：MAP vs MMSE（加速梯度下降 vs ILA）
对应章节：4.6 加速采样方法：过松弛与惯性Langevin算法
知识点：
  - ILA迭代格式：X_{m+1} = X_m - γ∇E(X_m) + β(X_m - X_{m-1}) + √(2γ(1-β)) N_m
  - "优化+噪声=采样"的统一规律
  - 动量项β(X_m - X_{m-1})加速收敛
  - 后验均值（MMSE）vs 后验众数（MAP）
  - 后验标准差量化不确定性

素材来源：Mathematics.../Teaching Unit 3/Labs/Lab_2.zip -> opt_vs_sample.ipynb
"""

import numpy as np
import torch
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
    SAVE_DIR = os.path.join(_gdrive, '实验4.6-1')
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

np.random.seed(42)
torch.manual_seed(42)

# ══════════════════════════════════════════════════════════
# 1. 离散梯度算子 Du（Neumann边界）
# ══════════════════════════════════════════════════════════
def Du(u):
    """
    离散梯度算子（前向差分，Neumann边界）
    输入: u [M, N]
    输出: [2, M, N]，第0维是x方向梯度，第1维是y方向梯度
    """
    M, N = u.shape
    Du_out = torch.zeros((2, M, N), dtype=u.dtype, device=u.device)
    Du_out[0, :, :-1] += u[:, 1:] - u[:, :-1]
    Du_out[1, :-1, :] += u[1:, :] - u[:-1, :]
    return Du_out

# ══════════════════════════════════════════════════════════
# 2. 伴随散度算子 DTp（Neumann边界）
# ══════════════════════════════════════════════════════════
def DTp(p):
    """
    伴随散度算子（后向差分，Neumann边界）
    输入: p [2, M, N]
    输出: [M, N]
    """
    C, M, N = p.shape
    DTp_out = torch.zeros((M, N), dtype=p.dtype, device=p.device)
    DTp_out[:, 1:] += p[0, :, :-1]
    DTp_out[:, :-1] -= p[0, :, :-1]
    DTp_out[1:, :] += p[1, :-1, :]
    DTp_out[:-1, :] -= p[1, :-1, :]
    return DTp_out

# ══════════════════════════════════════════════════════════
# 3. 合成图像生成
# ══════════════════════════════════════════════════════════
N = 100
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)

u1 = 0*X + 0.5
u2 = X + Y
u2 -= u2.min()
u2 /= u2.max()

u3 = np.clip(1 - ((1-X)**2 + (1-Y)**2), a_min=0, a_max=None)
u3 -= u3.min()
u3 /= u3.max()

m1 = 0*X
m1[N//8:-N//8, N//8:-N//8] = 1
u = u1.copy()
u[m1==1] = u2[m1==1] * 1.25

m2 = 0*X
m2[(X-1)**2 + (Y-1)**2 <= 1] = 1
u[m2==1] = u3[m2==1]

# ══════════════════════════════════════════════════════════
# 4. 加噪声
# ══════════════════════════════════════════════════════════
sigma = 0.1
f = u + np.random.randn(N, N) * sigma

print("=" * 60)
print("【参数设定】")
print("=" * 60)
print(f"图像尺寸: {N} × {N}")
print(f"噪声标准差: σ = {sigma}")
print(f"TV正则化参数: λ = 10.0")
print(f"动量参数: θ = 0.99")
print(f"迭代次数: {10000}")

# ══════════════════════════════════════════════════════════
# 5. MAP：加速梯度下降（重球法/Heavy-ball momentum）
# ══════════════════════════════════════════════════════════
lamb = 10.0
eps = 1e-03
theta = 0.99
L = lamb * 8 / eps + 1 / sigma**2
tau = 2 / L

f_ = torch.from_numpy(f.copy())
u_map = torch.from_numpy(f.copy())
u_old = torch.from_numpy(f.copy())

maxit = 10000
check = 1000

print("\n" + "=" * 60)
print("【MAP求解】加速梯度下降（重球法）")
print("=" * 60)

for it in range(maxit):
    diff_u = Du(u_map)
    grad_u = lamb * DTp(diff_u / torch.clamp(torch.abs(diff_u), min=eps)) + (u_map - f_) / sigma**2
    
    u_i = u_map + theta * (u_map - u_old)
    u_old = u_map.clone()
    u_map = u_i - tau * grad_u
    
    if it % check == 0:
        TV = lamb * torch.abs(diff_u).sum()
        Dat = torch.sum((u_map - f_)**2 / (2 * sigma**2))
        print(f"  iter {it:5d}: TV + Data = {TV.item() + Dat.item():.6f}", end="\r")

print(f"\n  MAP求解完成")

# ══════════════════════════════════════════════════════════
# 6. MMSE：ILA（惯性Langevin算法）
#    与MAP的唯一区别：多了噪声项 √(2τ(1-θ)) N
# ══════════════════════════════════════════════════════════
u_ila = torch.from_numpy(f.copy())
u_old_ = torch.from_numpy(f.copy())

n_burn_in = 1000
u_sum_ = torch.zeros((N, N))
u_sqr_ = torch.zeros((N, N))

print("\n" + "=" * 60)
print("【MMSE求解】ILA采样")
print("=" * 60)

for it in range(maxit):
    diff_u = Du(u_ila)
    grad_u = lamb * DTp(diff_u / torch.clamp(torch.abs(diff_u), min=eps)) + (u_ila - f_) / sigma**2
    
    u_i = u_ila + theta * (u_ila - u_old_)
    u_old_ = u_ila.clone()
    u_ila = u_i - tau * grad_u + torch.randn((N, N)) * np.sqrt(2 * tau * (1 - theta))
    
    # burn-in: 跳过前期非平稳样本
    if it >= n_burn_in:
        u_sum_ += u_ila
        u_sqr_ += u_ila**2
    
    if it % check == 0:
        TV = lamb * torch.abs(diff_u).sum()
        Dat = torch.sum((u_ila - f_)**2 / (2 * sigma**2))
        print(f"  iter {it:5d}: TV + Data = {TV.item() + Dat.item():.6f}", end="\r")

print(f"\n  ILA采样完成 (burn-in: {n_burn_in} iterations)")

n_samples = maxit - n_burn_in
u_avg_ = u_sum_ / n_samples
u_var_ = u_sqr_ / n_samples - u_avg_**2

# ══════════════════════════════════════════════════════════
# 7. 结果输出
# ══════════════════════════════════════════════════════════
def psnr(x, y):
    mse = np.mean((x - y)**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)

psnr_noisy = psnr(f, u)
psnr_map = psnr(u_map.numpy(), u)
psnr_ila = psnr(u_avg_.numpy(), u)

print("\n" + "=" * 60)
print("【PSNR对比】")
print("=" * 60)
print(f"  含噪图像:     {psnr_noisy:.2f} dB")
print(f"  MAP估计:      {psnr_map:.2f} dB")
print(f"  MMSE估计:     {psnr_ila:.2f} dB")

print("\n" + "=" * 60)
print("【核心发现】")
print("=" * 60)
print("1. ILA迭代格式：")
print("   X_{m+1} = X_m - γ∇E(X_m) + β(X_m - X_{m-1}) + √(2γ(1-β)) N_m")
print("")
print("2. 与MAP的唯一区别：多了噪声项 √(2γ(1-β)) N_m")
print("   → 优化算法 + 噪声 = 采样算法")
print("")
print("3. 动量项 β(X_m - X_{m-1}) 加速收敛，对应优化中的重球法（Heavy-ball momentum）")
print("")
print("4. 后验标准差量化不确定性：边缘处不确定性高，平坦区域不确定性低")

# ══════════════════════════════════════════════════════════
# 8. 可视化
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(u, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title(r'原始图像', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(f, cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title(rf'含噪图像 ($\sigma={sigma}$)', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].axis('off')

axes[1, 0].imshow(u_map.numpy(), cmap='gray', vmin=0, vmax=1)
axes[1, 0].set_title(r'MAP（加速梯度下降）', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(u_avg_.numpy(), cmap='gray', vmin=0, vmax=1)
axes[1, 1].set_title(r'MMSE（ILA后验均值）', fontsize=12)
axes[1, 1].axis('off')

axes[1, 2].imshow(torch.sqrt(torch.clamp(u_var_, min=0)).numpy(), cmap='hot')
axes[1, 2].set_title(r'后验标准差（不确定性）', fontsize=12)
axes[1, 2].axis('off')

fig.suptitle(r'实验4.6-1 优化 vs 采样：MAP vs MMSE', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'exp4_6-1_MAP_vs_MMSE.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图像已保存: {os.path.join(SAVE_DIR, 'exp4_6-1_MAP_vs_MMSE.png')}")

# ══════════════════════════════════════════════════════════
# 保存数值结果到JSON文件
# ══════════════════════════════════════════════════════════
import json

def _to_native(obj):
    """递归将numpy/torch类型转换为Python原生类型，便于JSON序列化"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, 'item') and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (ValueError, RuntimeError, TypeError):
            return obj
    return obj

results_summary = {
    'experiment': '4.6-1',
    'title': '优化 vs 采样：MAP vs MMSE',
    'setup': {
        'image_size': [int(N), int(N)],
        'sigma': float(sigma),
        'lamb': float(lamb),
        'theta': float(theta),
        'maxit': maxit,
        'n_burn_in': n_burn_in,
    },
    'results': {
        'psnr_noisy': float(round(psnr_noisy, 2)),
        'psnr_map': float(round(psnr_map, 2)),
        'psnr_ila': float(round(psnr_ila, 2)),
    }
}

with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(_to_native(results_summary), f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
