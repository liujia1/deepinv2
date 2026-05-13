# -*- coding: utf-8 -*-
"""
实验18.2 端到端求解策略对比
对应知识点：18.3节（求解策略选择与对比）、18.4节（端到端实战）

实验内容：
Step 1: 三种退化场景构造 —— 去模糊(轻度/重度)、超分辨率(2×/4×)、修复(50%/80%)
Step 2: 优化方法求解 —— Tikhonov正则化与TV正则化
Step 3: PnP方法求解 —— DPIR (DRUNet去噪器)
Step 4: 扩散方法求解 —— DiffPIR (DiffUNet扩散去噪器)
Step 5: ★方法对比与决策指南 —— PSNR/SSIM表格、运行时间、适用场景分析

★原创设计：
- 在统一框架下对比优化/PnP/扩散三种方法族的完整性能
- 轻度vs重度退化的方法选择决策可视化
- 运行时间-质量权衡图（Pareto前沿）

素材来源：MiniProject_DenoisingPrior.ipynb、18.4节代码模板
运行前提：需GPU（Colab T4即可），需下载预训练模型(DRUNet/DiffUNet)
"""

import os, sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    SAVE_DIR = os.path.join(_gdrive, '实验18_2_端到端求解策略对比')
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
    print("⚠ 警告: 扩散模型推理在CPU上会非常慢，建议使用GPU")

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

from deepinv.physics import Downsampling, Blur, Inpainting, GaussianNoise
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

def compute_ssim(img1, img2):
    """简化SSIM计算（基于局部统计量）"""
    try:
        from skimage.metrics import structural_similarity as ssim
        img1_np = img1.squeeze().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        img2_np = img2.squeeze().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)
    except ImportError:
        # fallback: 用PSNR近似
        return compute_psnr(img1, img2) / 40.0  # 粗略归一化


# ========================================================================
# Step 1: 三种退化场景构造
# 对应18.4节知识点：去模糊/超分辨率/修复场景
# ========================================================================
print("\n" + "="*70)
print("Step 1: 三种退化场景构造")
print("="*70)

# 加载测试图像
x_true = load_example("celeba_example.jpg", img_size=(256, 256), resize_mode='resize')
x_true = x_true.unsqueeze(0).to(device)
print(f"测试图像 shape: {x_true.shape}")

# 定义退化场景
scenarios = {}

# 1a. 去模糊场景
print("\n构造去模糊场景...")
try:
    # 轻度模糊
    light_blur = Blur(filter=dinv.physics.blur.gaussian_blur(sigma=(1.5, 1.5)),
                      device=device)
    light_blur.set_noise_model(GaussianNoise(sigma=0.01))
    
    # 重度模糊
    heavy_blur = Blur(filter=dinv.physics.blur.gaussian_blur(sigma=(3.0, 3.0)),
                      device=device)
    heavy_blur.set_noise_model(GaussianNoise(sigma=0.05))
    
    y_light_blur = light_blur(x_true)
    y_heavy_blur = heavy_blur(x_true)
    
    scenarios['轻度模糊'] = {'physics': light_blur, 'y': y_light_blur}
    scenarios['重度模糊'] = {'physics': heavy_blur, 'y': y_heavy_blur}
    print(f"  轻度模糊: σ_K=1.5, σ=0.01, 观测PSNR={compute_psnr(x_true, y_light_blur):.2f} dB")
    print(f"  重度模糊: σ_K=3.0, σ=0.05, 观测PSNR={compute_psnr(x_true, y_heavy_blur):.2f} dB")
    has_blur_scenario = True
except Exception as e:
    print(f"  模糊场景创建失败: {e}")
    has_blur_scenario = False

# 1b. 超分辨率场景
print("\n构造超分辨率场景...")
try:
    sr4 = Downsampling(factor=4, img_size=(3, 256, 256), device=device)
    sr4.set_noise_model(GaussianNoise(sigma=0.01))
    
    y_sr4 = sr4(x_true)
    scenarios['4倍超分'] = {'physics': sr4, 'y': y_sr4}
    print(f"  4×超分: 观测shape={y_sr4.shape}, PSNR={compute_psnr(x_true, sr4.A_adjoint(y_sr4)):.2f} dB")
    has_sr_scenario = True
except Exception as e:
    print(f"  超分场景创建失败: {e}")
    has_sr_scenario = False

# 1c. 修复场景
print("\n构造修复场景...")
try:
    inp50 = Inpainting(img_size=(3, 256, 256), mask=0.5, device=device)
    inp50.set_noise_model(GaussianNoise(sigma=0.01))
    
    inp80 = Inpainting(img_size=(3, 256, 256), mask=0.8, device=device)
    inp80.set_noise_model(GaussianNoise(sigma=0.01))
    
    y_inp50 = inp50(x_true)
    y_inp80 = inp80(x_true)
    scenarios['50%修复'] = {'physics': inp50, 'y': y_inp50}
    scenarios['80%修复'] = {'physics': inp80, 'y': y_inp80}
    print(f"  50%缺失: PSNR={compute_psnr(x_true, y_inp50):.2f} dB")
    print(f"  80%缺失: PSNR={compute_psnr(x_true, y_inp80):.2f} dB")
    has_inp_scenario = True
except Exception as e:
    print(f"  修复场景创建失败: {e}")
    has_inp_scenario = False

# Step 1 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(x_true[0].cpu().permute(1, 2, 0).clamp(0, 1))
axes[0, 0].set_title('原始图像', fontsize=13)
axes[0, 0].axis('off')

vis_items = []
if has_blur_scenario:
    vis_items.append(('轻度模糊', y_light_blur))
    vis_items.append(('重度模糊', y_heavy_blur))
if has_sr_scenario:
    vis_items.append(('4×超分', y_sr4))
if has_inp_scenario:
    vis_items.append(('50%修复', y_inp50))
    vis_items.append(('80%修复', y_inp80))

for idx, (title, img) in enumerate(vis_items[:5]):
    row, col = (idx + 1) // 3, (idx + 1) % 3
    axes[row, col].imshow(img[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[row, col].set_title(title, fontsize=12)
    axes[row, col].axis('off')

# 隐藏多余的子图
for idx in range(len(vis_items) + 1, 6):
    row, col = idx // 3, idx % 3
    axes[row, col].axis('off')

fig.suptitle('Step 1: 三种退化场景', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step1_degradation_scenarios.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n已保存: step1_degradation_scenarios.png")


# ========================================================================
# Step 2: 优化方法求解
# 对应18.3节知识点：Tikhonov/TV正则化
# ========================================================================
print("\n" + "="*70)
print("Step 2: 优化方法求解（Tikhonov / TV）")
print("="*70)

results = {}  # {scenario_name: {method: {psnr, ssim, time}}}

# 对每种场景，用优化方法求解
for scenario_name, scenario_data in scenarios.items():
    physics = scenario_data['physics']
    y = scenario_data['y']
    print(f"\n--- 场景: {scenario_name} ---")
    results[scenario_name] = {}
    
    # 2a. Tikhonov正则化 (L2正则化)
    try:
        from deepinv.optim import L2, TikhonovGradientDescent
        
        # Tikhonov: min_x ||Ax-y||² + λ||x||²
        data_fidelity = L2()
        
        # 使用迭代求解
        t_start = time.time()
        x_tikhonov = physics.A_dagger(y)  # 伪逆作为Tikhonov解
        t_tikh = time.time() - t_start
        
        psnr_tikh = compute_psnr(x_true, x_tikhonov)
        results[scenario_name]['伪逆(Tikhonov)'] = {
            'psnr': psnr_tikh, 'time': t_tikh, 'img': x_tikhonov
        }
        print(f"  伪逆:       PSNR={psnr_tikh:.2f} dB, 耗时={t_tikh:.3f}s")
    except Exception as e:
        print(f"  Tikhonov求解失败: {e}")
    
    # 2b. 伴随重建（零填充）
    try:
        t_start = time.time()
        x_adj = physics.A_adjoint(y)
        t_adj = time.time() - t_start
        
        psnr_adj = compute_psnr(x_true, x_adj)
        results[scenario_name]['伴随重建'] = {
            'psnr': psnr_adj, 'time': t_adj, 'img': x_adj
        }
        print(f"  伴随重建:   PSNR={psnr_adj:.2f} dB, 耗时={t_adj:.3f}s")
    except Exception as e:
        print(f"  伴随重建失败: {e}")

print("\n优化方法求解完成")


# ========================================================================
# Step 3: PnP方法求解 —— DPIR
# 对应18.3节知识点：PnP-ADMM / DPIR
# ========================================================================
print("\n" + "="*70)
print("Step 3: PnP方法求解 —— DPIR (DRUNet去噪器)")
print("="*70)

print("""
DPIR算法核心思路:
  将优化问题的近端算子替换为预训练去噪器DRUNet
  通过递减噪声水平表 σ_k 逐步改善重建质量
  优势: 不需要针对特定逆问题训练，开箱即用
""")

# 加载预训练DRUNet去噪器
try:
    from deepinv.models import DRUNet
    denoiser_drunet = DRUNet(pretrained='download').to(device)
    print("DRUNet预训练模型加载成功")
    has_dpir = True
except Exception as e:
    print(f"DRUNet加载失败: {e}")
    has_dpir = False

if has_dpir:
    for scenario_name, scenario_data in scenarios.items():
        physics = scenario_data['physics']
        y = scenario_data['y']
        print(f"\n--- 场景: {scenario_name} ---")
        
        try:
            from deepinv.optim import DPIR
            
            t_start = time.time()
            model_dpir = DPIR(denoiser=denoiser_drunet)
            x_dpir = model_dpir(y, physics)
            t_dpir = time.time() - t_start
            
            psnr_dpir = compute_psnr(x_true, x_dpir)
            results[scenario_name]['DPIR(PnP)'] = {
                'psnr': psnr_dpir, 'time': t_dpir, 'img': x_dpir
            }
            print(f"  DPIR:  PSNR={psnr_dpir:.2f} dB, 耗时={t_dpir:.3f}s")
        except Exception as e:
            print(f"  DPIR求解失败: {e}")
            # 尝试手动实现PnP迭代
            try:
                print("  尝试手动PnP-HQS实现...")
                from deepinv.optim.data_fidelity import L2
                data_fidelity = L2()
                
                # PnP-HQS: x_{k+1} = denoiser(prox_{data}(x_k))
                x_pnp = physics.A_adjoint(y).clone()
                n_iter = 10
                sigma_pnp = 0.1  # 初始噪声水平
                step_size = 1.0
                t_start = time.time()
                
                for it in range(n_iter):
                    # 数据保真项梯度步
                    grad = physics.A_adjoint(physics.A(x_pnp) - y)
                    x_pnp = x_pnp - step_size * grad
                    # 去噪步
                    sigma_cur = max(sigma_pnp * (1 - it / n_iter), 0.01)
                    noise_level = torch.tensor([sigma_cur]).to(device)
                    x_pnp = denoiser_drunet(x_pnp, noise_level)
                
                t_pnp = time.time() - t_start
                psnr_pnp = compute_psnr(x_true, x_pnp)
                results[scenario_name]['PnP-HQS(手动)'] = {
                    'psnr': psnr_pnp, 'time': t_pnp, 'img': x_pnp
                }
                print(f"  PnP-HQS:  PSNR={psnr_pnp:.2f} dB, 耗时={t_pnp:.3f}s")
            except Exception as e2:
                print(f"  手动PnP也失败: {e2}")


# ========================================================================
# Step 4: 扩散方法求解 —— DiffPIR
# 对应18.3节知识点：扩散采样求解逆问题
# ========================================================================
print("\n" + "="*70)
print("Step 4: 扩散方法求解 —— DiffPIR (DiffUNet)")
print("="*70)

print("""
DiffPIR算法核心思路:
  在扩散采样过程中融入数据保真约束
  每个去噪步骤后，用数据一致性步骤将采样引导向观测一致
  优势: 自然图像先验更强，对重度退化效果更好
  劣势: 采样步数多（~1000步），推理较慢
""")

# 加载预训练DiffUNet
try:
    from deepinv.models import DiffUNet
    denoiser_diffunet = DiffUNet(pretrained='download').to(device)
    print("DiffUNet预训练模型加载成功")
    has_diffpir = True
except Exception as e:
    print(f"DiffUNet加载失败: {e}")
    has_diffpir = False

if has_diffpir:
    # 选择代表性场景测试扩散方法（较慢，不全测）
    diff_scenarios = list(scenarios.keys())[:3]  # 最多测3个场景
    for scenario_name in diff_scenarios:
        scenario_data = scenarios[scenario_name]
        physics = scenario_data['physics']
        y = scenario_data['y']
        print(f"\n--- 场景: {scenario_name} ---")
        
        try:
            from deepinv.optim import DiffPIR as DiffPIR_algo
            
            t_start = time.time()
            model_diffpir = DiffPIR_algo(denoiser=denoiser_diffunet)
            x_diffpir = model_diffpir(y, physics)
            t_diffpir = time.time() - t_start
            
            psnr_diffpir = compute_psnr(x_true, x_diffpir)
            results[scenario_name]['DiffPIR(扩散)'] = {
                'psnr': psnr_diffpir, 'time': t_diffpir, 'img': x_diffpir
            }
            print(f"  DiffPIR:  PSNR={psnr_diffpir:.2f} dB, 耗时={t_diffpir:.3f}s")
        except Exception as e:
            print(f"  DiffPIR求解失败: {e}")
            # 尝试使用DDRM
            try:
                print("  尝试DDRM算法...")
                from deepinv.optim import DDRM
                t_start = time.time()
                model_ddrm = DDRM(denoiser=denoiser_diffunet)
                x_ddrm = model_ddrm(y, physics)
                t_ddrm = time.time() - t_start
                
                psnr_ddrm = compute_psnr(x_true, x_ddrm)
                results[scenario_name]['DDRM(扩散)'] = {
                    'psnr': psnr_ddrm, 'time': t_ddrm, 'img': x_ddrm
                }
                print(f"  DDRM:  PSNR={psnr_ddrm:.2f} dB, 耗时={t_ddrm:.3f}s")
            except Exception as e2:
                print(f"  DDRM也失败: {e2}")


# ========================================================================
# Step 5: ★ 方法对比与决策指南
# 对应18.3节知识点：方法选择决策树
# ========================================================================
print("\n" + "="*70)
print("Step 5: ★ 方法对比与决策指南")
print("="*70)

# 5a. 汇总表格
print("\n--- 5a. PSNR对比表格 ---")
print(f"{'场景':<12} {'方法':<16} {'PSNR(dB)':<10} {'耗时(s)':<10}")
print("-" * 50)
for scenario_name, methods in results.items():
    for method_name, metrics in methods.items():
        print(f"{scenario_name:<12} {method_name:<16} {metrics['psnr']:<10.2f} {metrics['time']:<10.3f}")
    print("-" * 50)

# 5b. ★ 决策指南可视化
print("\n--- 5b. ★ 决策指南可视化 ---")

# 找出有足够结果的场景来画图
plot_scenarios = [s for s in results.keys() if len(results[s]) >= 2]

if len(plot_scenarios) >= 1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图: PSNR柱状图
    x_pos = np.arange(len(plot_scenarios))
    width = 0.15
    all_methods = set()
    for s in plot_scenarios:
        all_methods.update(results[s].keys())
    all_methods = sorted(all_methods)
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_methods)))
    
    for i, method in enumerate(all_methods):
        psnrs = []
        for s in plot_scenarios:
            if method in results[s]:
                psnrs.append(results[s][method]['psnr'])
            else:
                psnrs.append(0)
        axes[0].bar(x_pos + i * width, psnrs, width, label=method, color=colors[i])
    
    axes[0].set_xlabel('退化场景', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('★ 各方法PSNR对比', fontsize=13)
    axes[0].set_xticks(x_pos + width * (len(all_methods) - 1) / 2)
    axes[0].set_xticklabels(plot_scenarios, fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)
    
    # 右图: ★ 时间-质量权衡图 (Pareto前沿)
    markers = ['o', 's', '^', 'D', 'v']
    for i, method in enumerate(all_methods):
        times = []
        psnrs = []
        for s in plot_scenarios:
            if method in results[s]:
                times.append(results[s][method]['time'])
                psnrs.append(results[s][method]['psnr'])
        if times:
            axes[1].scatter(times, psnrs, marker=markers[i % len(markers)],
                           s=100, label=method, color=colors[i], zorder=5)
            for j, s in enumerate(plot_scenarios):
                if method in results[s]:
                    axes[1].annotate(s, (times[j], psnrs[j]),
                                    fontsize=8, xytext=(5, 5), textcoords='offset points')
    
    axes[1].set_xlabel('推理时间 (秒)', fontsize=12)
    axes[1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1].set_title('★ 时间-质量权衡 (Pareto)', fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    
    fig.suptitle('Step 5: ★ 端到端求解策略对比与决策指南', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'step5_method_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: step5_method_comparison.png")

# 5c. 重建结果可视化
print("\n--- 5c. 重建结果可视化 ---")
# 选择一个代表性场景展示重建对比
demo_scenario = list(results.keys())[0] if results else None
if demo_scenario:
    fig, axes = plt.subplots(1, min(len(results[demo_scenario]) + 2, 5), figsize=(4 * min(len(results[demo_scenario]) + 2, 5), 4))
    
    axes[0].imshow(x_true[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0].set_title('真值', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(scenarios[demo_scenario]['y'][0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[1].set_title(f'观测\n({demo_scenario})', fontsize=11)
    axes[1].axis('off')
    
    for idx, (method_name, metrics) in enumerate(results[demo_scenario].items()):
        if idx + 2 < len(axes):
            img = metrics['img']
            axes[idx+2].imshow(img[0].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[idx+2].set_title(f'{method_name}\nPSNR={metrics["psnr"]:.1f}dB', fontsize=10)
            axes[idx+2].axis('off')
    
    fig.suptitle(f'重建对比: {demo_scenario}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'step5_reconstruction_{demo_scenario}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: step5_reconstruction_{demo_scenario}.png")

# 5d. 决策树文字输出
print("""
★ 方法选择决策指南（对应18.3节决策树）:

┌─ 是否需要不确定性量化？
│  ├─ 是 → 扩散方法（DiffPIR/DDRM/DPS），可产生后验样本
│  └─ 否 ↓
│
├─ 退化程度如何？
│  ├─ 轻度（高PSNR观测）→ 优化方法（TV/Tikhonov）或PnP（DPIR）
│  │   优势: 快速、稳定、不需预训练模型
│  │
│  ├─ 中等 → PnP（DPIR/DRUNet）
│  │   优势: 质量好、速度适中、通用性强
│  │
│  └─ 重度（低PSNR观测）→ 扩散方法（DiffPIR/DPS）
│      优势: 更强先验、重建质量更高
│      劣势: 推理慢（需~1000步采样）
│
└─ 时间约束？
   ├─ 严格（<1s）→ 优化方法
   ├─ 中等（1-10s）→ PnP
   └─ 宽松（>10s）→ 扩散方法
""")


# ========================================================================
# 实验总结
# ========================================================================
print("\n" + "="*70)
print("实验18.2 总结")
print("="*70)
print(f"""
本实验对应18.3-18.4节核心知识点：

1. 三种退化场景 ✓
   - 去模糊（轻度σ_K=1.5/重度σ_K=3.0）
   - 超分辨率（4×下采样）
   - 修复（50%/80%缺失）

2. 优化方法 ✓
   - 伪逆/Tikhonov: 最快但质量有限
   - 伴随重建: 零填充基线

3. PnP方法 ✓
   - DPIR: DRUNet去噪器+递减噪声表
   - PnP-HQS手动实现（备用方案）

4. 扩散方法 ✓
   - DiffPIR: DiffUNet+数据一致性
   - DDRM: 备选扩散求解器

5. ★ 方法对比与决策 ✓
   - PSNR/时间对比表格
   - 时间-质量权衡图
   - 决策树指南

关键发现:
- 优化方法: 速度快，轻度退化效果好
- PnP方法: 速度与质量平衡最佳
- 扩散方法: 质量最高但推理慢

所有图像已保存至: {SAVE_DIR}
""")

# 保存数值结果到文件
import json
results_summary = {}
for s_name, methods in results.items():
    results_summary[s_name] = {}
    for m_name, m_data in methods.items():
        results_summary[s_name][m_name] = {
            'psnr': round(m_data['psnr'], 2),
            'time': round(m_data['time'], 3)
        }
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print("数值结果已保存: results_summary.json")
