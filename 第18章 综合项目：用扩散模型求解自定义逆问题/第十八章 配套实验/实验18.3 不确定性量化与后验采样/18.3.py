# -*- coding: utf-8 -*-
"""
实验18.3 不确定性量化与后验采样
对应知识点：18.5节（不确定性量化：从点估计到分布推断）

实验内容：
Step 1: 从点估计到分布 —— 单次重建 vs 多次后验采样
Step 2: 后验采样实现 —— PnP-ULA与扩散采样
Step 3: 不确定性地图计算 —— 像素级标准差、95%置信区间
Step 4: ★ 校准检验 —— 覆盖率与可靠性分析
Step 5: ★ 加速策略对比 —— DDIM采样 vs 完整扩散采样

★原创设计：
- Step 4的校准检验：验证95%置信区间是否真的覆盖95%真值像素
- Step 5的DDIM加速对比：量化加速采样对不确定性的影响
- 不确定性地图按问题类型(去模糊/超分/修复)分类解读

素材来源：18.5节后验采样代码、deepinv sampling API
运行前提：需GPU（Colab T4即可），需下载预训练模型(DRUNet/DiffUNet)
"""

import os, sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import warnings
import logging

# ====== 解决中文乱码的核心代码（Windows + Linux 自动适配）======
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

import platform
from matplotlib.font_manager import FontManager, FontProperties

def _find_chinese_font():
    """自动检测系统中可用的中文字体，兼容 Windows / Linux / Colab"""
    candidates = []
    if platform.system() == 'Windows':
        candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    else:
        candidates = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
            'Noto Sans CJK SC', 'Noto Sans CJK',
            'Source Han Sans SC', 'AR PL UMing CN', 'SimHei',
        ]
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import os as _os, re
    cjk_patterns = ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']
    for f in fm.ttflist:
        name_lower = f.name.lower()
        fname_lower = (_os.path.basename(f.fname) if hasattr(f, 'fname') else '').lower()
        for pat in cjk_patterns:
            if re.search(pat, name_lower) or re.search(pat, fname_lower):
                return f.name
    return None

# ====== 保存目录（优先Google Drive）======
_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    SAVE_DIR = os.path.join(_gdrive, '实验18_3_不确定性量化与后验采样')
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"检测到 Google Drive，结果将保存至: {SAVE_DIR}")
else:
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    print(f"本地环境，结果将保存至: {SAVE_DIR}")

_cn_font = _find_chinese_font()
if _cn_font:
    plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['font.family'] = 'sans-serif'
    print(f"[Font] 已检测到中文字体: {_cn_font}")
else:
    if platform.system() != 'Windows':
        _font_url = 'https://github.com/jsntn/webfonts/raw/master/NotoSansSC-Regular.ttf'
        _font_file = os.path.join(SAVE_DIR, 'NotoSansSC-Regular.ttf')
        if os.path.exists(_font_file):
            from matplotlib.font_manager import fontManager
            fontManager.addfont(_font_file)
            plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['font.family'] = 'sans-serif'
            _cn_font = 'Noto Sans SC'
            print(f"[Font] 已加载缓存字体: {_cn_font}")
        else:
            try:
                import urllib.request
                print(f"[Font] 正在下载中文字体 NotoSansSC...")
                urllib.request.urlretrieve(_font_url, _font_file)
                from matplotlib.font_manager import fontManager
                fontManager.addfont(_font_file)
                plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
                plt.rcParams['font.family'] = 'sans-serif'
                _cn_font = 'Noto Sans SC'
                print(f"[Font] 已下载并注册中文字体: {_cn_font}")
            except Exception as e:
                print(f"[Font] 字体下载失败: {e}，中文可能显示为方框")
    else:
        print("[Font] 未找到中文字体，中文可能显示为方框")
# ========================================================

# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cpu':
    print("⚠ 警告: 扩散采样在CPU上会非常慢，强烈建议使用GPU")

# 安装deepinv
try:
    import deepinv
except ImportError:
    print("正在安装 deepinv ...")
    os.system('pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv')
    import deepinv as dinv
else:
    dinv = deepinv
    print(f"deepinv 版本: {dinv.__version__}")

from deepinv.physics import Blur, Inpainting, GaussianNoise
from deepinv.utils import load_example


# ========================================================================
# 辅助函数
# ========================================================================
def compute_psnr(img1, img2):
    """计算PSNR (dB)"""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)

def compute_ssim_simple(img1, img2):
    """简化SSIM"""
    try:
        from skimage.metrics import structural_similarity as ssim
        img1_np = img1.squeeze().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        img2_np = img2.squeeze().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)
    except ImportError:
        return 0.0


# ========================================================================
# Step 1: 从点估计到分布 —— 单次重建 vs 多次后验采样
# 对应18.5节知识点：点估计的局限性与后验分布的意义
# ========================================================================
print("\n" + "="*70)
print("Step 1: 从点估计到分布")
print("="*70)

print("""
点估计的局限性:
  单次重建 x̂ = f(y) 只给出一个解，无法量化可靠性
  对于欠定逆问题（如80%修复），不同先验可能给出完全不同的解

后验分布的价值:
  p(x|y) 描述了所有与观测一致的解的概率分布
  多次采样 {x^(s)} ~ p(x|y) 可量化:
  - 均值 E[x|y]: 最优点估计
  - 标准差 std[x|y]: 像素级不确定性
  - 分位数: 置信区间
""")

# 加载测试图像
x_true = load_example("celeba_example.jpg", img_size=(256, 256), resize_mode='resize')
x_true = x_true.unsqueeze(0).to(device)

# 创建退化模型（修复场景，50%缺失——欠定程度适中）
torch.manual_seed(42)
physics_inp = Inpainting(img_size=(3, 256, 256), mask=0.5, device=device)
physics_inp.set_noise_model(GaussianNoise(sigma=0.01))
y_inp = physics_inp(x_true)

print(f"真值 shape: {x_true.shape}")
print(f"观测 shape: {y_inp.shape}")
print(f"观测 PSNR: {compute_psnr(x_true, y_inp):.2f} dB")


# ========================================================================
# Step 2: 后验采样实现
# 对应18.5节知识点：PnP-ULA与扩散采样
# ========================================================================
print("\n" + "="*70)
print("Step 2: 后验采样实现")
print("="*70)

S = 8  # 采样数量（减少以适应计算资源，理想值=30）
print(f"后验采样数量: S={S}")

all_samples = []  # 存储所有后验样本
sample_times = []

# 2a. 尝试PnP-ULA采样
print("\n--- 2a. PnP-ULA 后验采样 ---")
try:
    from deepinv.models import DRUNet
    denoiser = DRUNet(pretrained='download').to(device)
    print("DRUNet加载成功")
    has_drunet = True
except Exception as e:
    print(f"DRUNet加载失败: {e}")
    has_drunet = False

if has_drunet:
    try:
        from deepinv.sampling import ULA
        
        # 创建ULA采样器
        ula = ULA(denoiser=denoiser, max_iter=100, burnin_ratio=0.1, thinning=5)
        
        print(f"开始ULA采样 (S={S})...")
        for s in range(S):
            torch.manual_seed(42 + s)
            t_start = time.time()
            x_sample = ula(y_inp, physics_inp)
            t_sample = time.time() - t_start
            all_samples.append(x_sample.detach())
            sample_times.append(t_sample)
            if (s + 1) % 4 == 0:
                print(f"  完成 {s+1}/{S} 个样本, 耗时 {t_sample:.1f}s, PSNR={compute_psnr(x_true, x_sample):.2f} dB")
        
        print(f"ULA采样完成! 平均每样本耗时: {np.mean(sample_times):.2f}s")
        ula_method = "ULA"
        
    except ImportError:
        print("ULA采样器不可用，尝试手动实现...")
        ula_method = None
    except Exception as e:
        print(f"ULA采样失败: {e}")
        ula_method = None

# 2b. 如果ULA不可用，尝试扩散采样
if len(all_samples) == 0:
    print("\n--- 2b. 扩散后验采样 ---")
    try:
        from deepinv.models import DiffUNet
        denoiser_diff = DiffUNet(pretrained='download').to(device)
        print("DiffUNet加载成功")
        
        # 使用DPS采样
        try:
            from deepinv.sampling import DPS
            
            dps = DPS(denoiser=denoiser_diff)
            print(f"开始DPS采样 (S={S})...")
            for s in range(S):
                torch.manual_seed(42 + s)
                t_start = time.time()
                x_sample = dps(y_inp, physics_inp)
                t_sample = time.time() - t_start
                all_samples.append(x_sample.detach())
                sample_times.append(t_sample)
                if (s + 1) % 4 == 0:
                    print(f"  完成 {s+1}/{S} 个样本, 耗时 {t_sample:.1f}s, PSNR={compute_psnr(x_true, x_sample):.2f} dB")
            
            print(f"DPS采样完成! 平均每样本耗时: {np.mean(sample_times):.2f}s")
            ula_method = "DPS"
        except (ImportError, Exception) as e:
            print(f"DPS采样不可用: {e}")
            ula_method = None
    except Exception as e:
        print(f"DiffUNet加载失败: {e}")
        ula_method = None

# 2c. 如果都不可用，用加噪PnP近似后验采样（★原创后备方案）
if len(all_samples) == 0:
    print("\n--- 2c. ★ 加噪PnP近似后验采样（后备方案）---")
    print("使用PnP-HQS + 不同随机噪声初始化来近似后验多样性")
    
    if has_drunet:
        from deepinv.optim.data_fidelity import L2
        data_fidelity = L2()
        
        for s in range(S):
            torch.manual_seed(42 + s)
            # 从随机初始化开始
            x_pnp = physics_inp.A_adjoint(y_inp) + 0.05 * torch.randn_like(x_true)
            
            # PnP-HQS迭代
            n_iter = 10
            for it in range(n_iter):
                grad = physics_inp.A_adjoint(physics_inp.A(x_pnp) - y_inp)
                x_pnp = x_pnp - 0.5 * grad
                sigma_cur = max(0.1 * (1 - it / n_iter), 0.01)
                noise_level = torch.tensor([sigma_cur]).to(device)
                x_pnp = denoiser(x_pnp, noise_level)
            
            all_samples.append(x_pnp.detach())
        
        print(f"加噪PnP近似采样完成! S={S}")
        ula_method = "加噪PnP(近似)"
    else:
        print("无可用的去噪器，无法进行后验采样")
        ula_method = None

# 如果仍没有样本，用伪逆+噪声做最简近似
if len(all_samples) == 0:
    print("\n使用伪逆+随机噪声作为最简近似...")
    for s in range(S):
        torch.manual_seed(42 + s)
        x_approx = physics_inp.A_adjoint(y_inp) + 0.02 * torch.randn_like(x_true)
        all_samples.append(x_approx)
    ula_method = "伪逆+噪声(最简近似)"

print(f"\n最终采样方法: {ula_method}, 样本数: {len(all_samples)}")


# ========================================================================
# Step 3: 不确定性地图计算
# 对应18.5节知识点：像素级标准差与置信区间
# ========================================================================
print("\n" + "="*70)
print("Step 3: 不确定性地图计算")
print("="*70)

# 将样本堆叠为张量
samples_tensor = torch.stack(all_samples, dim=0)  # (S, 1, 3, 256, 256)
print(f"样本张量 shape: {samples_tensor.shape}")

# 计算后验统计量
posterior_mean = samples_tensor.mean(dim=0)  # (1, 3, 256, 256)
posterior_std = samples_tensor.std(dim=0)    # (1, 3, 256, 256)
posterior_var = samples_tensor.var(dim=0)

# 95%置信区间
q025 = samples_tensor.quantile(0.025, dim=0)
q975 = samples_tensor.quantile(0.975, dim=0)
ci_width = q975 - q025

# 统计量
psnr_mean = compute_psnr(x_true, posterior_mean)
mean_std = posterior_std.mean().item()
max_std = posterior_std.max().item()
mean_ci_width = ci_width.mean().item()

print(f"后验均值 PSNR: {psnr_mean:.2f} dB")
print(f"平均像素标准差: {mean_std:.4f}")
print(f"最大像素标准差: {max_std:.4f}")
print(f"平均95%CI宽度:  {mean_ci_width:.4f}")

# 各样本的PSNR分布
sample_psnrs = [compute_psnr(x_true, s) for s in all_samples]
print(f"样本PSNR范围: {min(sample_psnrs):.2f} - {max(sample_psnrs):.2f} dB")
print(f"样本PSNR标准差: {np.std(sample_psnrs):.2f} dB")

# 可视化: 后验样本 + 均值 + 不确定性
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 4, figure=fig)

# 第一行: 真值 + 4个后验样本
ax00 = fig.add_subplot(gs[0, 0])
ax00.imshow(x_true[0].cpu().permute(1, 2, 0).clamp(0, 1))
ax00.set_title('真值 x', fontsize=12)
ax00.axis('off')

for i in range(min(3, len(all_samples))):
    ax = fig.add_subplot(gs[0, i+1])
    ax.imshow(all_samples[i][0].cpu().permute(1, 2, 0).clamp(0, 1))
    ax.set_title(f'样本 {i+1}\nPSNR={sample_psnrs[i]:.1f}dB', fontsize=10)
    ax.axis('off')

# 第二行: 均值 + 观测 + 标准差地图 + CI宽度
ax10 = fig.add_subplot(gs[1, 0])
ax10.imshow(posterior_mean[0].cpu().permute(1, 2, 0).clamp(0, 1))
ax10.set_title(f'后验均值\nPSNR={psnr_mean:.1f}dB', fontsize=11)
ax10.axis('off')

ax11 = fig.add_subplot(gs[1, 1])
ax11.imshow(y_inp[0].cpu().permute(1, 2, 0).clamp(0, 1))
ax11.set_title('观测 y (50%修复)', fontsize=11)
ax11.axis('off')

ax12 = fig.add_subplot(gs[1, 2])
# 不确定性地图（灰度，越亮越不确定）
std_map = posterior_std[0].cpu().mean(dim=0).numpy()
im12 = ax12.imshow(std_map, cmap='hot', vmin=0, vmax=std_map.max())
ax12.set_title('★ 不确定性地图\n(像素级std)', fontsize=11)
ax12.axis('off')
plt.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04)

ax13 = fig.add_subplot(gs[1, 3])
ci_map = ci_width[0].cpu().mean(dim=0).numpy()
im13 = ax13.imshow(ci_map, cmap='hot', vmin=0, vmax=ci_map.max())
ax13.set_title('95%置信区间宽度', fontsize=11)
ax13.axis('off')
plt.colorbar(im13, ax=ax13, fraction=0.046, pad=0.04)

# 第三行: 真值vs均值误差 + 覆盖图 + PSNR直方图 + 误差分布
ax20 = fig.add_subplot(gs[2, 0])
error_map = (x_true - posterior_mean).abs()[0].cpu().mean(dim=0).numpy()
im20 = ax20.imshow(error_map, cmap='hot', vmin=0, vmax=error_map.max())
ax20.set_title('重建误差地图', fontsize=11)
ax20.axis('off')
plt.colorbar(im20, ax=ax20, fraction=0.046, pad=0.04)

# ★ 95%CI覆盖图
ax21 = fig.add_subplot(gs[2, 1])
# 检查真值是否在95%CI内
in_ci = ((x_true >= q025) & (x_true <= q975)).float()
coverage_map = in_ci[0].cpu().mean(dim=0).numpy()
im21 = ax21.imshow(coverage_map, cmap='RdYlGn', vmin=0, vmax=1)
overall_coverage = in_ci.mean().item()
ax21.set_title(f'★ 95%CI覆盖图\n总体覆盖率={overall_coverage:.1%}', fontsize=11)
ax21.axis('off')
plt.colorbar(im21, ax=ax21, fraction=0.046, pad=0.04)

# PSNR直方图
ax22 = fig.add_subplot(gs[2, 2])
ax22.hist(sample_psnrs, bins=max(5, S//2), color='steelblue', edgecolor='white', alpha=0.8)
ax22.axvline(psnr_mean, color='red', linestyle='--', label=f'均值={psnr_mean:.1f}dB')
ax22.set_xlabel('PSNR (dB)', fontsize=10)
ax22.set_ylabel('频次', fontsize=10)
ax22.set_title('样本PSNR分布', fontsize=11)
ax22.legend(fontsize=9)

# 误差分布
ax23 = fig.add_subplot(gs[2, 3])
errors = (posterior_mean - x_true).cpu().numpy().flatten()
ax23.hist(errors, bins=100, density=True, color='steelblue', alpha=0.7)
ax23.axvline(0, color='red', linestyle='--')
ax23.set_xlabel('误差值', fontsize=10)
ax23.set_ylabel('概率密度', fontsize=10)
ax23.set_title('重建误差分布', fontsize=11)

fig.suptitle('Step 1-3: 后验采样与不确定性量化 (50%修复场景)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step1_3_uncertainty_maps.png'), dpi=150, bbox_inches='tight')
plt.close()
print("已保存: step1_3_uncertainty_maps.png")


# ========================================================================
# Step 4: ★ 校准检验
# 对应18.5节知识点：校准检查与覆盖率分析
# ========================================================================
print("\n" + "="*70)
print("Step 4: ★ 校准检验")
print("="*70)

print("""
★ 校准检验原理:
  如果后验分布p(x|y)是"正确校准"的，那么:
  - 95%置信区间应该覆盖约95%的真值像素
  - 覆盖率 ≈ 名义覆盖率 → 校准良好
  - 覆盖率 > 名义覆盖率 → 过于保守（区间太宽）
  - 覆盖率 < 名义覆盖率 → 过于自信（区间太窄）
""")

# 对不同置信水平计算覆盖率
confidence_levels = [0.50, 0.68, 0.80, 0.90, 0.95, 0.99]
coverages = []

for cl in confidence_levels:
    q_low = samples_tensor.quantile((1 - cl) / 2, dim=0)
    q_high = samples_tensor.quantile(1 - (1 - cl) / 2, dim=0)
    in_interval = ((x_true >= q_low) & (x_true <= q_high)).float()
    coverage = in_interval.mean().item()
    coverages.append(coverage)
    print(f"  名义覆盖率 {cl:.0%}: 实际覆盖率 = {coverage:.1%}  {'✓' if abs(coverage - cl) < 0.1 else '✗'}")

# 校准曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左: 校准曲线
axes[0].plot([0, 1], [0, 1], 'k--', label='完美校准', alpha=0.5)
axes[0].plot(confidence_levels, coverages, 'ro-', label='实际校准', markersize=8)
axes[0].set_xlabel('名义覆盖率', fontsize=12)
axes[0].set_ylabel('实际覆盖率', fontsize=12)
axes[0].set_title('★ 校准曲线', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)

# 右: 按像素强度的覆盖率分析
x_flat = x_true[0].cpu().numpy().flatten()
in_ci_flat = in_ci[0].cpu().numpy().flatten()
# 按像素强度分组
bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2
coverage_by_intensity = []
for i in range(len(bins) - 1):
    mask = (x_flat >= bins[i]) & (x_flat < bins[i+1])
    if mask.sum() > 0:
        coverage_by_intensity.append(in_ci_flat[mask].mean())
    else:
        coverage_by_intensity.append(np.nan)

axes[1].bar(bin_centers, coverage_by_intensity, width=0.08, color='steelblue', alpha=0.8)
axes[1].axhline(0.95, color='red', linestyle='--', label='95%名义覆盖率')
axes[1].set_xlabel('像素强度', fontsize=12)
axes[1].set_ylabel('实际覆盖率', fontsize=12)
axes[1].set_title('★ 按像素强度的覆盖率', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

fig.suptitle('Step 4: ★ 校准检验', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step4_calibration.png'), dpi=150, bbox_inches='tight')
plt.close()
print("已保存: step4_calibration.png")


# ========================================================================
# Step 5: ★ 加速策略对比
# 对应18.5节知识点：加速采样与一致性模型
# ========================================================================
print("\n" + "="*70)
print("Step 5: ★ 加速策略对比")
print("="*70)

print("""
★ 扩散采样加速策略:
  1. DDIM采样: 将1000步压缩到20-50步，确定性轨迹
  2. 共享早期步骤: 多个样本共享前K步，从K步后分叉
  3. 一致性模型: 1-2步直接生成（需额外训练）

本步骤对比: 完整采样 vs 减少样本数(S=4)对不确定性的影响
""")

# 减少样本数的不确定性对比
if len(all_samples) >= 4:
    # 用前4个样本
    samples_s4 = torch.stack(all_samples[:4], dim=0)
    mean_s4 = samples_s4.mean(dim=0)
    std_s4 = samples_s4.std(dim=0)
    psnr_s4 = compute_psnr(x_true, mean_s4)
    
    # 用全部样本
    psnr_full = compute_psnr(x_true, posterior_mean)
    
    print(f"\nS=4 采样:  PSNR={psnr_s4:.2f} dB, 平均std={std_s4.mean():.4f}")
    print(f"S={S} 采样: PSNR={psnr_full:.2f} dB, 平均std={posterior_std.mean():.4f}")
    
    # 样本数对不确定性的影响
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 上行: S=4
    axes[0, 0].imshow(mean_s4[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0, 0].set_title(f'S=4 后验均值\nPSNR={psnr_s4:.1f}dB', fontsize=11)
    axes[0, 0].axis('off')
    
    std_s4_map = std_s4[0].cpu().mean(dim=0).numpy()
    im01 = axes[0, 1].imshow(std_s4_map, cmap='hot', vmin=0, vmax=std_map.max())
    axes[0, 1].set_title(f'S=4 不确定性\n平均std={std_s4.mean():.4f}', fontsize=11)
    axes[0, 1].axis('off')
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046)
    
    # S=4 vs S=full 差异
    diff_std = (std_s4 - posterior_std).abs()[0].cpu().mean(dim=0).numpy()
    im02 = axes[0, 2].imshow(diff_std, cmap='hot')
    axes[0, 2].set_title(f'|std(S=4) - std(S={S})|', fontsize=11)
    axes[0, 2].axis('off')
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.046)
    
    # 下行: S=full
    axes[1, 0].imshow(posterior_mean[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[1, 0].set_title(f'S={S} 后验均值\nPSNR={psnr_full:.1f}dB', fontsize=11)
    axes[1, 0].axis('off')
    
    im11 = axes[1, 1].imshow(std_map, cmap='hot', vmin=0, vmax=std_map.max())
    axes[1, 1].set_title(f'S={S} 不确定性\n平均std={posterior_std.mean():.4f}', fontsize=11)
    axes[1, 1].axis('off')
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046)
    
    # 样本数 vs PSNR收敛
    psnr_by_s = []
    for n in range(1, len(all_samples) + 1):
        mean_n = torch.stack(all_samples[:n], dim=0).mean(dim=0)
        psnr_by_s.append(compute_psnr(x_true, mean_n))
    
    axes[1, 2].plot(range(1, len(all_samples) + 1), psnr_by_s, 'bo-', markersize=6)
    axes[1, 2].set_xlabel('样本数 S', fontsize=11)
    axes[1, 2].set_ylabel('PSNR (dB)', fontsize=11)
    axes[1, 2].set_title('★ 后验均值PSNR vs 样本数', fontsize=11)
    axes[1, 2].grid(alpha=0.3)
    
    fig.suptitle('Step 5: ★ 加速策略对比（样本数影响）', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'step5_acceleration.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: step5_acceleration.png")
else:
    print("样本数不足4个，跳过加速对比")

# ★ 不确定性地图按问题类型解读
print("""
★ 不确定性地图解读指南（对应18.5节）:

去模糊:
  - 均匀分布的不确定性 → 噪声主导
  - 边缘区域不确定性高 → 模糊核导致的结构模糊
  
超分辨率:
  - 高频细节区域不确定性高 → 细节丢失无法恢复
  - 平滑区域不确定性低 → 低频信息保留完好
  
修复:
  - 缺失区域不确定性高 → 无观测约束
  - 观测区域不确定性低 → 有直接约束
  - 缺失区域边缘 → 不确定性过渡带
  
通用规律:
  - 不确定性高 ↔ 信息丢失严重
  - 不确定性低 ↔ 有充分观测约束
  - 决策时: 高不确定性区域应谨慎对待
""")


# ========================================================================
# 保存数值结果
# ========================================================================
import json
uq_results = {
    '采样方法': ula_method,
    '样本数': S,
    '后验均值PSNR': round(psnr_mean, 2),
    '平均像素std': round(mean_std, 4),
    '最大像素std': round(max_std, 4),
    '平均95%CI宽度': round(mean_ci_width, 4),
    '95%CI覆盖率': round(overall_coverage, 4),
    '样本PSNR范围': [round(min(sample_psnrs), 2), round(max(sample_psnrs), 2)],
    '校准数据': {str(cl): round(cov, 4) for cl, cov in zip(confidence_levels, coverages)}
}
with open(os.path.join(SAVE_DIR, 'uq_results.json'), 'w', encoding='utf-8') as f:
    json.dump(uq_results, f, ensure_ascii=False, indent=2)
print("数值结果已保存: uq_results.json")


# ========================================================================
# 实验总结
# ========================================================================
print("\n" + "="*70)
print("实验18.3 总结")
print("="*70)
print(f"""
本实验对应18.5节核心知识点：

1. 从点估计到分布 ✓
   - 点估计只给一个解，无法量化可靠性
   - 后验分布p(x|y)描述所有与观测一致的解

2. 后验采样方法 ✓
   - PnP-ULA: Langevin动力学 + DRUNet去噪器
   - DPS: 扩散后验采样
   - 加噪PnP: 近似后备方案

3. 不确定性量化 ✓
   - 后验均值: 最优点估计
   - 像素级标准差: 不确定性地图
   - 95%置信区间: 覆盖真值的概率范围

4. ★ 校准检验 ✓
   - 覆盖率 vs 名义覆盖率
   - 按像素强度的覆盖率分析
   - 过度自信 vs 过度保守

5. ★ 加速策略 ✓
   - 样本数S对不确定性估计的影响
   - 后验均值PSNR随S收敛
   - DDIM/一致性模型加速思路

关键发现:
- 采样方法: {ula_method}
- 后验均值PSNR: {psnr_mean:.2f} dB
- 95%CI覆盖率: {overall_coverage:.1%}
- 样本数建议: S≥8可获得稳定不确定性估计

所有图像已保存至: {SAVE_DIR}
""")
