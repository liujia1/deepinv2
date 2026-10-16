# -*- coding: utf-8 -*-
"""
实验10.3-1 从变分下界到去噪目标
对应章节: 10.3 从变分下界到去噪目标

知识点:
  - 三种参数化的等价性: x₀-prediction, ε-prediction, score prediction
  - 均值匹配目标 vs L_simple简化目标
  - VLB权重随时间的变化特性

实验内容:
  步骤1: 三种参数化等价性验证 (x₀/ε/score prediction)
  步骤2: 均值匹配损失 vs L_simple权重分析
  步骤3: 损失权重可视化

本实验不需要GPU，通过数值实验验证10.3节的核心数学结论。
"""

import sys
import io
import os
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
    SAVE_DIR = os.path.join(_gdrive, '实验10.3-1')
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

print("\n" + "="*60)
print("实验10.3-1: 从变分下界到去噪目标")
print("="*60)
print("对应章节: 10.3 从变分下界到去噪目标")
print("使用设备: CPU (本实验不需要GPU)")


# ============================================================
# 噪声调度定义
# ============================================================
T = 1000  # 扩散步数
beta_min, beta_max = 1e-4, 0.02

# 线性噪声调度 (DDPM)
betas = torch.linspace(beta_min, beta_max, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])

print("\n噪声调度: beta从" + str(beta_min) + "线性增至" + str(beta_max) + ", T=" + str(T))


# ============================================================
# 步骤1：三种参数化的等价性验证 —— 10.3节核心
# ============================================================
print("\n" + "="*60)
print("步骤1：三种参数化的等价性验证")
print("="*60)
print("\n[核心思想]")
print("  三种参数化方式在完美预测下给出完全相同的后验均值:")
print("  (1) x_0-prediction: 预测hat_x_0 → 计算mu_theta")
print("  (2) epsilon-prediction: 预测hat_epsilon → 计算mu_theta")
print("  (3) score prediction: 预测hat_s → 计算mu_theta")

# 给定x_0, t, ε，验证三种参数化的一致性
x0_test = torch.randn(3, 2)
eps_test = torch.randn_like(x0_test)

print("\n验证三种参数化对后验均值 tilde_mu_t 的等价性:")
print("    t      真实mu_t      x0-pred      eps-pred       score")
print("-" * 70)

equivalence_errors = {}

for t_idx in [10, 100, 500, 900]:
    # 注意：Python索引从0开始，时间步t从1开始，所以索引为t-1
    idx = t_idx - 1
    ab = alpha_bars[idx]
    a = alphas[idx]
    ab_prev = alpha_bars_prev[idx]
    b = betas[idx]

    # 构造x_t
    x_t = torch.sqrt(ab) * x0_test + torch.sqrt(1 - ab) * eps_test

    # 真实后验均值 mu_t(x_t, x_0)
    mu_true = (torch.sqrt(a) * (1 - ab_prev) / (1 - ab)) * x_t + \
              (torch.sqrt(ab_prev) * (1 - a) / (1 - ab)) * x0_test

    # (1) x₀-prediction: 先由(x_t, eps)恢复x0_hat(前向过程一致性)，再代入后验均值公式
    x0_hat = (x_t - torch.sqrt(1 - ab) * eps_test) / torch.sqrt(ab)
    err_x0rec = (x0_hat - x0_test).abs().max().item()
    mu_x0 = (torch.sqrt(a) * (1 - ab_prev) / (1 - ab)) * x_t + \
            (torch.sqrt(ab_prev) * (1 - a) / (1 - ab)) * x0_hat

    # (2) ε-prediction: eps → mu = (1/sqrt(a))(x_t - (1-a)/sqrt(1-ab)·eps)
    mu_eps = (1.0 / torch.sqrt(a)) * (x_t - (1 - a) / torch.sqrt(1 - ab) * eps_test)

    # (3) score prediction: 直接形式 mu = (x_t + beta_t·s)/sqrt(alpha_t)，其中 s = -eps/sqrt(1-ab)
    score_hat = -eps_test / torch.sqrt(1 - ab)  # 完美预测
    mu_score = (x_t + b * score_hat) / torch.sqrt(a)

    err_x0 = (mu_true - mu_x0).abs().max().item()
    err_eps = (mu_true - mu_eps).abs().max().item()
    err_score = (mu_true - mu_score).abs().max().item()

    equivalence_errors['t_%d' % t_idx] = {
        'x0_recovery': err_x0rec,
        'mu_via_x0': err_x0,
        'mu_via_eps': err_eps,
        'mu_via_score_direct': err_score,
    }

    print(str(t_idx).rjust(5) + "  " + str(round(mu_true[0,0].item(), 8)).rjust(15) + "  " + str(round(mu_x0[0,0].item(), 8)).rjust(15) + "  " + str(round(mu_eps[0,0].item(), 8)).rjust(15) + "  " + str(round(mu_score[0,0].item(), 8)).rjust(15))
    print("       误差: x0恢复=" + "%.3e" % err_x0rec + ", x0路径=" + "%.3e" % err_x0 + ", eps路径=" + "%.3e" % err_eps + ", score直接=" + "%.3e" % err_score)

print("\n→ 三种参数化在完美预测下给出完全相同的后验均值")
print("  这是10.3节的核心结论: x0-prediction ≡ eps-prediction ≡ score prediction")
print("\n三种参数化的转换关系:")
print("  - x0 ↔ epsilon: x0 = (x_t - sqrt(1-ab_t)·eps) / sqrt(ab_t)")
print("  - epsilon ↔ score: s = -eps / sqrt(1-ab_t)")
print("  - score ↔ x0: x0 = (x_t + (1-ab_t)·s) / sqrt(ab_t)")

# 补充验证: 不完美预测下三者不再等价 —— "完美预测"限定词是必要的
print("\n[补充] 不完美预测下的分化: 同一0.01输出误差诱导的后验均值误差")
print("    t      经x0_hat      经eps_hat      经score       (x0/eps)")
print("-" * 62)

imperfect_divergence = {}
d_err = 0.01
for t_idx in [10, 100, 500, 900]:
    idx = t_idx - 1
    ab = alpha_bars[idx]
    a = alphas[idx]
    ab_prev = alpha_bars_prev[idx]
    b = betas[idx]
    e_x0 = (torch.sqrt(ab_prev) * b / (1 - ab) * d_err).item()
    e_eps = (b / (torch.sqrt(a) * torch.sqrt(1 - ab)) * d_err).item()
    e_s = (b / torch.sqrt(a) * d_err).item()
    imperfect_divergence['t_%d' % t_idx] = {'x0': e_x0, 'eps': e_eps, 'score': e_s}
    print(str(t_idx).rjust(5) + "  " + ("%.6f" % e_x0).rjust(12) + "  " + ("%.6f" % e_eps).rjust(12) + "  " + ("%.6f" % e_s).rjust(12) + "  " + ("%.1f" % (e_x0 / e_eps)).rjust(10))

print("→ 不完美预测时三者诱导的均值误差按t依赖的权重分化(小t时x0误差影响大, 大t反之)")
print("  这正是VLB加权与L_simple等权差异的本质, 也是'完美预测'限定词必要性的来源")


# ============================================================
# 步骤2：均值匹配损失 vs L_simple权重分析 —— 10.3节
# ============================================================
print("\n" + "="*60)
print("步骤2：均值匹配损失 vs L_simple权重分析")
print("="*60)
print("\n[核心思想]")
print("  VLB目标中的均值匹配项(Ho et al. 2020):")
print("  L_{t-1} ∝ β_t²/(2σ_t²·α_t(1-ᾱ_t)) · ||ε - ε̂||²")
print("  取σ_t²=β_t(DDPM约定)后，权重简化为 w_t = β_t/(2α_t(1-ᾱ_t))")
print("  L_simple对所有时间步赋等权重1，这是Ho et al. 2020的关键简化")

print("\n不同时间步的VLB权重:")
print("    t       ab_t       VLB权重     L_simple权重   相对t=1")
print("-" * 55)

# 先计算t=1的权重作为基准
weight_t1 = betas[0] / (2 * alphas[0] * (1 - alpha_bars[0]))

for t_idx in [1, 10, 50, 100, 300, 500, 700, 900, 999]:
    # 注意：Python索引从0开始，时间步t从1开始，所以索引为t-1
    idx = t_idx - 1
    ab = alpha_bars[idx]
    a = alphas[idx]
    b = betas[idx]

    # VLB权重系数: 根据Ho et al. 2020 eq.14，使用σ_t²=β_t归一化
    # L_{t-1} ∝ 1/(2σ_t²) · β_t²/(α_t(1-ᾱ_t)) · ||ε-ε_θ||²
    # 取σ_t²=β_t时，权重系数 = β_t/(2α_t(1-ᾱ_t))
    weight_vlb = b / (2 * a * (1 - ab))

    # 计算相对于t=1的权重比值
    ratio_to_t1 = weight_vlb / weight_t1

    print(str(t_idx).rjust(5) + "  " + str(round(ab.item(), 6)).rjust(10) + "  " + str(round(weight_vlb.item(), 6)).rjust(12) + "       1.000000      " + str(round(ratio_to_t1.item(), 4)).rjust(8))

# 精确定位谷值
w_all = (betas / (2 * alphas * (1 - alpha_bars))).numpy()
t_min = int(np.argmin(w_all)) + 1

print("\n关键观察:")
print("  - t小(1-100): 权重随t增大而急剧下降(约50倍)，早期时间步主导VLB目标")
print("  - t中段(t≈" + str(t_min) + "): 权重达到谷值(约" + str(round(float(w_all[t_min - 1]), 5)) + ")，相对t=1下降约100倍，对VLB贡献最小")
print("  - t大(500+): 权重缓慢回升(t=999约" + str(round(float(w_all[-1]), 4)) + ")，但整体仍远小于t=1时的水平")
print("  - L_simple对所有t赋等权重1，这是Ho et al. 2020的关键简化")
print("  - 实际效果: L_simple忽略权重差异，但训练更稳定（10.3节）")


# ============================================================
# 步骤3：ε-prediction的优势分析
# ============================================================
print("\n" + "="*60)
print("步骤3：eps-prediction的优势分析")
print("="*60)
print("\n[为什么DDPM选择eps-prediction]")
print("  1. 目标函数简单: L_simple = ||eps - hat_eps_theta(x_t, t)||^2")
print("  2. 网络输出范围: 噪声eps服从N(0,I)，输出范围有界")
print("  3. 训练稳定性: 无需处理x_0的复杂依赖")
print("  4. 计算效率: 直接预测噪声，一步到位")

# 数值验证: 不同t下的目标函数范围
print("\n不同参数化下的目标函数数值范围:")
print("    t     ||x0-hat_x0||范围   ||eps-hat_eps||范围")
print("-" * 45)

for t_idx in [10, 100, 500, 900]:
    # 注意：Python索引从0开始，时间步t从1开始，所以索引为t-1
    idx = t_idx - 1
    ab = alpha_bars[idx]
    # 假设预测误差为1
    err_eps = 1.0
    # 转换到x0空间的误差
    err_x0 = err_eps / torch.sqrt(ab)

    print(str(t_idx).rjust(5) + "           " + str(round(err_x0.item(), 4)).rjust(15) + "           " + str(round(err_eps, 4)).rjust(15))

print("\n关键发现:")
print("  - t小(ab_t大): 转换系数≈1，两种参数化的误差尺度相当")
print("  - t大(ab_t小): 同样的eps误差映射到x0空间被1/sqrt(ab_t)急剧放大(t=900约60倍)")
print("  → eps-prediction在所有时间步误差尺度一致，x0-prediction在大t数值敏感")


# ============================================================
# 可视化
# ============================================================
print("\n" + "="*60)
print("生成可视化图表...")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) VLB权重 vs L_simple权重
ax = axes[0]
# 修正后的VLB权重公式：使用σ_t²=β_t归一化
weights = betas / (2 * alphas * (1 - alpha_bars))
ax.semilogy(range(1, T + 1), weights.numpy(), 'b-', linewidth=2, label='VLB权重')
ax.axhline(1, color='r', linestyle='--', linewidth=1.5, label='$L_{simple}$权重 (=1)')
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('损失权重 (对数尺度)', fontsize=12)
ax.set_title('(a) VLB损失权重 vs $L_{simple}$', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.annotate('VLB权重: 小t占主导, t≈' + str(t_min) + '谷值后回升', xy=(0.3, 0.8), xycoords='axes fraction',
            fontsize=10, color='blue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#dfe6e9', alpha=0.8))

# (b) 三种参数化的转换系数
ax = axes[1]
t_range = range(1, T + 1)

# eps到x0的转换系数: 1/sqrt(ab_t)
coeff_eps_to_x0 = 1.0 / torch.sqrt(alpha_bars)
# score到eps的转换系数: sqrt(1-ab_t)
coeff_score_to_eps = torch.sqrt(1 - alpha_bars)

ax.plot(t_range, coeff_eps_to_x0.numpy(), 'b-', linewidth=2, label=r'$\epsilon \to x_0$: $1/\sqrt{\bar{\alpha}_t}$')
ax.plot(t_range, coeff_score_to_eps.numpy(), 'g-', linewidth=2, label=r'$s \to \epsilon$: $\sqrt{1-\bar{\alpha}_t}$')
ax.set_xlabel('时间步 $t$', fontsize=12)
ax.set_ylabel('转换系数 (对数尺度)', fontsize=12)
ax.set_title('(b) 三种参数化的转换系数', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
ax.set_ylim(5e-3, 3e2)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, '步骤1_参数化等价性与损失权重.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print("图表已保存: 步骤1_参数化等价性与损失权重.png")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "="*60)
print("实验10.3-1 总结")
print("="*60)
print("\n关键结论:")
print("\n1. 三种参数化等价性 (步骤1)")
print("   - x0-prediction, eps-prediction, score prediction在完美预测下给出相同后验均值")
print("   - 不完美预测下三者按t依赖的权重分化，等价性仅在最优处成立")
print("   - 转换关系:")
print("     x0 = (x_t - sqrt(1-ab_t)·eps) / sqrt(ab_t)")
print("     s = -eps / sqrt(1-ab_t)")
print("\n2. VLB权重分析 (步骤2)")
print("   - VLB权重 w_t = β_t / (2α_t(1-ᾱ_t))  (使用σ_t²=β_t归一化)")
print("   - 权重呈U形: t小急剧下降，t≈" + str(t_min) + "谷值，t大缓慢回升")
print("   - L_simple对所有t赋等权重1")
print("\n3. eps-prediction的优势 (步骤3)")
print("   - 目标函数简单: ||eps - hat_eps||^2")
print("   - 数值范围稳定，所有时间步误差尺度一致")
print("   - DDPM选择eps-prediction作为默认参数化")

print("\n" + "="*60)
print("实验10.3-1 完成!")

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

# VLB权重 (使用σ_t²=β_t归一化)
weights_vlb = (betas / (2 * alphas * (1 - alpha_bars))).numpy()
results_summary = {
    'parameterization_equivalence': {
        'note': '三种参数化在完美预测下给出完全相同的后验均值 (误差为float32精度量级)',
        **equivalence_errors,
    },
    'imperfect_prediction_divergence': {
        'note': '同一0.01输出误差诱导的后验均值误差: 三者随t分化, 等价性仅在完美预测(最优)处成立',
        **imperfect_divergence,
    },
    'VLB_weights': {
        'argmin_t': t_min,
        **{f't_{t}': weights_vlb[t-1]
           for t in [1, 10, 50, 100, 300, 500, 700, 900, 999]},
    },
    'conversion_coefficients': {
        'note': 'eps_to_x0: 1/sqrt(ab_t), score_to_eps: sqrt(1-ab_t)',
        **{f't_{t}': {
            'eps_to_x0': (1.0 / torch.sqrt(alpha_bars[t-1])).item(),
            'score_to_eps': torch.sqrt(1 - alpha_bars[t-1]).item(),
        } for t in [10, 100, 500, 900]}
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")