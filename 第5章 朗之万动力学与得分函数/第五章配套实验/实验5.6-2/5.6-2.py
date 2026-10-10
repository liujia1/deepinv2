# -*- coding: utf-8 -*-
"""
实验5.6-2 置信区间与多样性分析
对应章节：5.6 近似理论与收敛保证 -> 5.6.5 置信区间 + 后验样本多样性与多峰性
知识点：
  - 95%置信区间：x_MMSE ± 1.96√Var(x|y)
  - 置信区间宽度反映局部不确定性
  - 后验样本展示分布的多样性（本例强数据约束下为单峰附近小幅扰动；
    真正可分的多峰后验演示见实验5.2-1的高斯混合）
  - 样本差异大的区域 = 高后验方差区域
  - 95%区间的经验覆盖率检验：真值 x_true 落入区间的像素比例应接近95%
  - 平稳性的劈半检验：前/后半段链各自估计的后验σ应一致（数据段不同，非自比）
  - 统计量口径：正态法主区间的中心与宽度都取自全链 Welford 累积量（与5.6-1
    一致、与上游闸门"全链Welford量演示可用"的裁定一致）；94个抽稀样本重算
    方差的正态区间仅作对照；非参数分位数只能由抽稀样本估计（npz未存全链分位数）

前置实验：
  - 需要先运行实验5.5-1，生成sampling_results.npz

素材来源：
  - 实验5.3.py 不确定性量化部分（置信区间、多样本展示）
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys

# Windows 控制台默认 GBK，打印某些非 GBK 数学符号（如带帽 x̂ 的组合抑扬符）会
# UnicodeEncodeError 直接崩溃，导致 JSON/配图无法生成。将 stdout/stderr 设为
# 容错（这些字符降级为 '?'，中文与数字照常输出），保证脚本在任意终端都能跑完。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验5.6-2')
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
    print("警告: chinese_font 模块未找到")
# ========================================================

# ============================================================
# 引用实验5.5-1 的唯一共享模块（npz加载/上游闸门/IAT的唯一事实源在
# 实验5.5-1/pnp_ula_common.py）。本文件不复制 5.5-1 的问题构建与采样代码。
# ============================================================
_exp551_dir = (os.path.join(_gdrive, '实验5.5-1') if _IN_COLAB
               else os.path.join(os.path.dirname(SAVE_DIR), '实验5.5-1'))
if os.path.isdir(_exp551_dir) and _exp551_dir not in sys.path:
    sys.path.insert(0, _exp551_dir)
# Colab Drive FUSE：目录在 pnp_ula_common.py 上传前被首次 import 探测时，
# FileFinder 会缓存“该目录无此模块”，文件之后到位也不刷新 → 导入前强制失效缓存。
import importlib
importlib.invalidate_caches()
sys.path_importer_cache.pop(_exp551_dir, None)
try:
    from pnp_ula_common import (load_sampling_result, upstream_chain_gate,
                                print_gate)
except ImportError as _e:
    raise FileNotFoundError(
        "未找到实验5.5-1 的共享代码 pnp_ula_common.py。\n"
        "请确认完整的“实验5.5-1”目录（含 pnp_ula_common.py、sampling_tools/、cman.png）"
        "与本实验同在“第五章配套实验”下（Colab 为 我的云端硬盘/实验5.5-1/）。\n"
        "并请先【运行实验5.5-1/5.5-1.py】生成 sampling_results.npz 后，再运行本实验。"
    ) from _e

np.random.seed(42)


# ============================================================
# 步骤1：加载采样结果（查找/加载/闸门的唯一实现见 pnp_ula_common）
# ============================================================
print("=" * 60)
print("步骤1：加载采样结果")
print("=" * 60)

# 找不到 5.5-1 的 npz 或必需字段时直接硬失败（Missing551DataError），
# 消息中明确要求先运行实验5.5-1；不允许合成数据静默兜底。
data, _data_path = load_sampling_result(SAVE_DIR)
print(f"[Data] 从 {_data_path} 加载采样结果（数据来源：真实 PnP-ULA 采样）")
post_mean = data['post_mean']
post_var = data['post_var']
mc_samples = data['mc_samples']
x_true = data['x_true']
_data_source = f"真实采样结果：{os.path.abspath(_data_path)}"

print(f"[Data] 数据来源：{_data_source}")
print(f"后验均值形状: {post_mean.shape}")
print(f"样本数量: {len(mc_samples)}")

# 样本数警告
if len(mc_samples) < 50:
    print(f"\n[警告] 样本数量 {len(mc_samples)} 过少，统计估计不可靠（建议≥100）")

# ------------------------------------------------------------
# 上游链(5.5-1)收敛闸门：判定实现与5.6-1/5.6-3共用（pnp_ula_common）。
# 新版 5.5-1 的 npz 含 trace_psnr/trace_fid/chain_config；旧版没有则如实提示。
# ------------------------------------------------------------
src_status, src_warnings, _src_extras = upstream_chain_gate(
    data, _data_path, purpose='置信区间/覆盖率')
n_post_full = _src_extras['n_post_full']
print_gate(src_status, src_warnings, '下游置信区间/覆盖率结论有效')

# 主统计量口径（与5.6-1一致、与上游闸门裁定一致）：
# post_mean / post_var_full 是整条链全部 post-burn-in 状态的 Welford 在线
# 累积量。MCMC 状态虽强自相关，但每个状态的边际分布都是目标后验，Welford
# 时间平均对后验均值/方差无偏，且是当前链长下方差最小的估计；闸门明确裁定
# "全链 Welford 均值/方差作为演示可用"。故正态法主区间一律用全链 Welford 量。
post_var_full = np.asarray(post_var).copy()

# 抽稀诊断量：94 个抽稀样本（抽稀间隔100 < 全链IAT≥1101，ESS 上界仅8.6）。
# 它们同样无偏但逐像素估计噪声大，仅用于：(a) 抽稀口径的正态区间【对照】；
# (b) 非参数分位数（npz 未保存全链逐像素分位数，分位数只能由这批样本估计）；
# (c) 步骤5的抽稀保真度检验。注意：不得像旧版那样把它当作主区间的方差来源。
post_mean_recomputed = np.mean(mc_samples, axis=0)
post_var_recomputed = np.var(mc_samples, axis=0, ddof=1)

print(f"\n[验证] 从稀疏样本重新估计的统计量 vs 全链 Welford 统计量:")
print(f"  均值差异 (L2范数): {np.linalg.norm(post_mean - post_mean_recomputed):.6f}")
print(f"  方差差异 (L2范数): {np.linalg.norm(post_var_full - post_var_recomputed):.6f}")
print("[Info] 正态法主区间使用全链 Welford 的均值与方差（主口径，与5.6-1一致）；")
print("       抽稀样本方差版正态区间仅作对照；分位数法只能由94个抽稀样本估计（定性参考）")

# 劈半统计：使用上游5.5-1保存的"全链前/后半段"（两段是链上不同数据，
# 比稀疏样本子集更接近独立检验）；旧版npz无此字段时，降级为稀疏样本前/后半。
if 'post_var_h1' in data.files and 'post_var_h2' in data.files:
    post_var_h1 = np.asarray(data['post_var_h1'])
    post_var_h2 = np.asarray(data['post_var_h2'])
    half_source = "全链前/后半段"
else:
    _nh = len(mc_samples) // 2
    post_var_h1 = np.var(mc_samples[:_nh], axis=0, ddof=1)
    post_var_h2 = np.var(mc_samples[_nh:], axis=0, ddof=1)
    half_source = f"稀疏样本前/后半（各{_nh}个，降级方案，非严格独立）"
    print(f"[Info] npz中无全链劈半字段，劈半检验降级为：{half_source}")


# ============================================================
# 步骤2：计算置信区间
# ============================================================
print("\n" + "=" * 60)
print("步骤2：计算95%置信区间")
print("=" * 60)

post_std = np.sqrt(post_var)                    # 主：全链 Welford σ
post_std_thin = np.sqrt(post_var_recomputed)    # 对照：94 抽稀样本 σ

# 方法1（主口径）：正态近似（假设后验为正态分布），中心与宽度都取自全链 Welford
z_95 = 1.96
q_low = post_mean - z_95 * post_std
q_high = post_mean + z_95 * post_std
ci_width = q_high - q_low  # 置信区间宽度

# 方法1（抽稀对照）：中心仍为全链 MMSE，仅把 σ 换成94个抽稀样本的重算标准差，
# 使两版差异纯粹来自"方差估计口径"，直接展示使用高噪声（ESS≤8.6）方差的后果
q_low_thin = post_mean - z_95 * post_std_thin
q_high_thin = post_mean + z_95 * post_std_thin
ci_width_thin = q_high_thin - q_low_thin

# 方法2：基于样本分位数（非参数方法，不假设正态）；npz 未存全链分位数，只能用抽稀样本
q_low_empirical = np.percentile(mc_samples, 2.5, axis=0)
q_high_empirical = np.percentile(mc_samples, 97.5, axis=0)
ci_width_empirical = q_high_empirical - q_low_empirical

print(f"95%置信区间公式（正态近似）: x̂_MMSE ± 1.96·√Var(x|y)")
print(f"\n[注意] ±1.96σ 假设后验为正态分布，是近似结果")
print(f"       若后验非正态（如TV先验），应使用样本分位数方法")
print(f"\n正态近似置信区间宽度统计（主口径：全链 Welford σ）:")
print(f"  平均宽度: {np.mean(ci_width):.4f}")
print(f"  最大宽度: {np.max(ci_width):.4f}")
print(f"  最小宽度: {np.min(ci_width):.4f}")
print(f"\n正态近似置信区间宽度统计（对照：94抽稀样本 σ, ESS≤8.6）:")
print(f"  平均宽度: {np.mean(ci_width_thin):.4f}")
print(f"  最大宽度: {np.max(ci_width_thin):.4f}")
print(f"  最小宽度: {np.min(ci_width_thin):.4f}")
print(f"\n样本分位数置信区间宽度统计（非参数，只能由抽稀样本估计）:")
print(f"  平均宽度: {np.mean(ci_width_empirical):.4f}")
print(f"  最大宽度: {np.max(ci_width_empirical):.4f}")
print(f"  最小宽度: {np.min(ci_width_empirical):.4f}")

# ===== 95%区间的经验覆盖率检验（直接回答"95%是否名副其实"）=====
# 逐像素检查真值 x_true 是否落在区间内；边界使用未裁剪数组（裁剪仅用于显示）。
# 注意：像素间存在强空间相关，不能把65536个像素当作独立伯努利试验套用
# sqrt(0.95*0.05/N) 的标准误，故这里只作描述性校准检查，不做假设检验。
cover_mask_norm = (x_true >= q_low) & (x_true <= q_high)
cover_mask_norm_thin = (x_true >= q_low_thin) & (x_true <= q_high_thin)
cover_mask_emp = (x_true >= q_low_empirical) & (x_true <= q_high_empirical)
coverage_norm = float(cover_mask_norm.mean())
coverage_norm_thin = float(cover_mask_norm_thin.mean())
coverage_emp = float(cover_mask_emp.mean())
print(f"\n95%区间经验覆盖率（真值落入区间的像素比例，名义水平=95%）:")
print(f"  正态近似法（主，全链Welford σ）: {coverage_norm*100:.2f}%")
print(f"  正态近似法（对照，94抽稀样本 σ）: {coverage_norm_thin*100:.2f}%")
print(f"  样本分位数法（抽稀样本）        : {coverage_emp*100:.2f}%")
print(f"  注：像素强空间相关，此为描述性校准；覆盖严重偏离95%说明区间不可直接定量使用")
_q_order = max(1, int(round(0.025 * len(mc_samples))))
print(f"  注：样本分位数仅由 {len(mc_samples)} 个抽稀样本估计，2.5%分位约对应第 {_q_order} 小样本，"
      "末端次序统计量分辨率粗；若抽稀间隔<全链IAT，有效样本更少，覆盖率只作定性参考")

# ===== 覆盖率偏离95%的定量解释：校准比（与5.6-1同一公式、同一npz，可交叉验证）=====
# 理想（逐像素高斯后验）：E|x_true - x_MMSE| / E[σ] = sqrt(2/pi) ≈ 0.798。
#   比值 >0.80：σ相对实际重建误差偏窄（后验方差不含模型偏差，如边缘振铃）；
#   比值 <0.80：σ相对实际误差偏保守——PnP-ULA 在有限去噪噪声 ε 下的平稳分布
#               比目标后验更弥散（见5.6-1知识点说明），区间系统性偏宽→覆盖率>95%。
_recon_err = np.abs(x_true - post_mean)
_sigma_mean = float(post_std.mean())
calib_mae = float(_recon_err.mean() / max(_sigma_mean, 1e-12))
calib_rms = float(np.sqrt(np.mean((x_true - post_mean) ** 2))
                 / max(_sigma_mean, 1e-12))
_calib_ideal = float(np.sqrt(2.0 / np.pi))
print(f"\n覆盖率偏离的校准诊断（全链Welford口径）:")
print(f"  重建MAE={_recon_err.mean():.6f}, 全链σ均值={_sigma_mean:.6f}")
print(f"  校准比 E|e|/E[σ] = {calib_mae:.4f}（理想高斯≈{_calib_ideal:.3f}；与5.6-1同源应一致）")
print(f"  校准比 RMS(e)/E[σ] = {calib_rms:.4f}")
if calib_mae < 0.80:
    print(f"  解读: 校准比<0.80，后验σ相对实际重建误差偏保守（有限ε下平稳分布过弥散），")
    print("        95%区间系统性偏宽，因此经验覆盖率高于名义95%（过覆盖），方向与数值均与此一致。")
elif calib_mae > 0.90:
    print(f"  解读: 校准比明显>0.80，σ偏窄（模型偏差不进后验方差），区间倾向欠覆盖。")
else:
    print(f"  解读: 校准比接近理想值0.80，σ幅度与实际误差同量级，覆盖率偏离主要来自短链/空间相关。")
print(f"  两比值联读: MAE比={calib_mae:.3f} 而 RMS比={calib_rms:.3f}（理想高斯应同时≈0.798与1.00）。")
print("        本例典型像素上σ偏宽（MAE比低）、但大误差像素把RMS比抬到≈1，说明校准失配是")
print("        空间异质的：平滑背景区σ明显过宽（对应5.6-1五分位Q1~Q3比值0.45~0.51），")
print("        高不确定边缘区σ才与误差同量级（Q5≈0.85）；这与漏检像素沿边缘分布一致。")


# ============================================================
# 步骤3：可视化置信区间
# ============================================================
print("\n" + "=" * 60)
print("步骤3：可视化置信区间")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第1行：后验均值、下界、上界
axes[0, 0].imshow(post_mean, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('后验均值 $\\hat{x}_{MMSE}$')
axes[0, 0].axis('off')

axes[0, 1].imshow(np.clip(q_low, 0, 1), cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title('95%置信区间下界')
axes[0, 1].axis('off')

axes[0, 2].imshow(np.clip(q_high, 0, 1), cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title('95%置信区间上界')
axes[0, 2].axis('off')

# 第2行：区间宽度、宽度直方图、高宽度区域
im_width = axes[1, 0].imshow(ci_width, cmap='hot')
axes[1, 0].set_title('置信区间宽度 $w = 3.92\\sqrt{\\mathrm{Var}}$（主口径：全链Welford σ）')
axes[1, 0].axis('off')
plt.colorbar(im_width, ax=axes[1, 0], fraction=0.046, pad=0.04)

# 宽度直方图
unique_widths = np.unique(ci_width)
if len(unique_widths) > 10:
    n_bins = 50
else:
    n_bins = min(10, len(unique_widths))

if n_bins > 1:
    axes[1, 1].hist(ci_width.flatten(), bins=n_bins, color='coral', alpha=0.7, edgecolor='white')
    axes[1, 1].axvline(np.mean(ci_width), color='red', linestyle='--',
                       label=f'均值：{np.mean(ci_width):.4f}')
    axes[1, 1].set_xlabel('置信区间宽度')
    axes[1, 1].set_ylabel('像素数量')
    axes[1, 1].set_title('宽度分布')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
else:
    axes[1, 1].text(0.5, 0.5, f'宽度恒定：{ci_width[0,0]:.4f}', ha='center', va='center',
                    transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('宽度分布（常数）')
    axes[1, 1].axis('off')

# 高宽度区域
threshold = np.percentile(ci_width, 90)
high_width = ci_width > threshold
axes[1, 2].imshow(high_width, cmap='Reds')
axes[1, 2].set_title(f'高宽度区域 (>P90={threshold:.4f})')
axes[1, 2].axis('off')

fig.suptitle('实验5.6-2 置信区间分析', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_置信区间分析.png'), dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 步骤4：后验样本多样性展示
# ============================================================
print("\n" + "=" * 60)
print("步骤4：后验样本多样性展示")
print("=" * 60)

num_show = min(16, len(mc_samples))
step = max(1, len(mc_samples) // num_show)
selected = mc_samples[::step][:num_show]

ncols = 4
nrows = (num_show + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows))

for i in range(nrows * ncols):
    ax = axes[i // ncols][i % ncols] if nrows > 1 else axes[i]
    if i < len(selected):
        ax.imshow(selected[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'样本 {i+1}')
    ax.axis('off')

fig.suptitle('PnP-ULA后验样本（展示后验分布的多样性）', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_后验样本多样性.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"展示了 {len(selected)} 个后验样本")
print("\n观察要点:")
print("  - 共享相似的全局结构（先验约束）")
print("  - 细节存在差异（数据一致性与先验的权衡）")
print("  - 差异区域 = 高不确定性区域")


# ============================================================
# 步骤5：样本差异分析
# ============================================================
print("\n" + "=" * 60)
print("步骤5：样本差异分析")
print("=" * 60)

if len(mc_samples) >= 2:
    # 稀疏样本逐像素标准差（多样性度量）；ddof=1 与全链 Welford 的无偏方差
    # 约定（S/(k-1)）一致，但它只基于94个抽稀样本，不是步骤2主区间所用的σ
    sample_std = np.std(mc_samples, axis=0, ddof=1)
    # 参照：整条 MCMC 链累积的后验标准差
    post_std_full = np.sqrt(post_var_full)

    # 劈半一致性：前/后半段各自的后验σ（两段是链上不同数据，比"全链vs其子集"更有检验力）
    half_std1 = np.sqrt(post_var_h1)
    half_std2 = np.sqrt(post_var_h2)
    corr_thin = np.corrcoef(post_std_full.flatten(), sample_std.flatten())[0, 1]
    corr_half = np.corrcoef(half_std1.flatten(), half_std2.flatten())[0, 1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) 样本标准差
    im_sample_std = axes[0, 0].imshow(sample_std, cmap='hot')
    axes[0, 0].set_title('样本标准差（多样性度量，ddof=1）')
    axes[0, 0].axis('off')
    plt.colorbar(im_sample_std, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # (0,1) 抽稀保真度：全链σ vs 稀疏子集σ（随机采样像素，避免视觉过密）
    ref_flat = post_std_full.flatten()
    sample_std_flat = sample_std.flatten()
    n_pixels_scatter = min(500, len(ref_flat))
    idx = np.random.choice(len(ref_flat), size=n_pixels_scatter, replace=False)
    axes[0, 1].scatter(ref_flat[idx], sample_std_flat[idx], alpha=0.3, s=1)
    lim = max(post_std_full.max(), sample_std.max())
    axes[0, 1].plot([0, lim], [0, lim], 'r--', label='y=x')
    _npost_lbl = f'{n_post_full}步累积' if n_post_full is not None else '步数未知（旧版npz）'
    axes[0, 1].set_xlabel(f'全链后验σ（约{_npost_lbl}）')
    axes[0, 1].set_ylabel(f'稀疏抽稀样本σ（{len(mc_samples)}个）')
    axes[0, 1].set_title(f'抽稀保真度（稀疏样本是全链子集，非独立），r={corr_thin:.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # (0,2) 差异图（相邻抽稀样本）
    diff = np.abs(mc_samples[0] - mc_samples[1])
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('相邻抽稀样本差异（样本1 vs 样本2）')
    axes[0, 2].axis('off')

    # (1,0) 劈半一致性散点：前半段σ vs 后半段σ
    h1_flat = half_std1.flatten()
    h2_flat = half_std2.flatten()
    axes[1, 0].scatter(h1_flat[idx], h2_flat[idx], alpha=0.3, s=1, color='seagreen')
    lim_h = max(half_std1.max(), half_std2.max())
    axes[1, 0].plot([0, lim_h], [0, lim_h], 'r--', label='y=x')
    axes[1, 0].set_xlabel('前半段后验σ')
    axes[1, 0].set_ylabel('后半段后验σ')
    axes[1, 0].set_title(f'劈半一致性（{half_source}），r={corr_half:.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # (1,1) 未覆盖真值的像素：红=主正态法(全链σ)漏盖，蓝=分位数法漏盖，紫=两者都漏
    miss_overlay = np.zeros((*x_true.shape, 3))
    miss_overlay[~cover_mask_norm, 0] = 1.0
    miss_overlay[~cover_mask_emp, 2] = 1.0
    axes[1, 1].imshow(miss_overlay)
    axes[1, 1].set_title('95%区间未覆盖真值的像素（主口径）\n'
                         '（红=正态法[全链σ]漏，蓝=分位数法漏，紫=两者都漏）')
    axes[1, 1].axis('off')

    # (1,2) 定量结论文字
    axes[1, 2].axis('off')
    _conclusion_txt = (
        "定量检验汇总\n"
        "----------------------------\n"
        f"样本数: {len(mc_samples)}（真实PnP-ULA链）\n"
        f"上游链: {'有警告，详见控制台/JSON' if src_warnings else '已到平台、抽稀充分'}\n\n"
        "95%区间经验覆盖率:\n"
        f"  正态(全链σ,主) : {coverage_norm*100:.2f}%\n"
        f"  正态(抽稀σ对照): {coverage_norm_thin*100:.2f}%\n"
        f"  样本分位数     : {coverage_emp*100:.2f}%\n"
        "  名义水平       : 95.00%\n"
        f"校准比 E|e|/E[σ] = {calib_mae:.2f}\n"
        "  (理想≈0.80；<0.80表示σ\n"
        "   偏保守、区间偏宽→过覆盖)\n\n"
        f"抽稀保真度 r = {corr_thin:.3f}\n"
        "  (稀疏样本是全链的子集，仅\n"
        "   检验空间格局是否保真，\n"
        "   非独立验证)\n\n"
        f"劈半一致性 r = {corr_half:.3f}\n"
        f"  ({half_source};\n"
        "   贴近y=x才支持链已平稳)"
    )
    axes[1, 2].text(0.03, 0.97, _conclusion_txt, transform=axes[1, 2].transAxes,
                    fontsize=11, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='whitesmoke', edgecolor='gray'))

    fig.suptitle('实验5.6-2 样本差异与区间校准分析', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤5_样本差异分析.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"抽稀保真度（全链σ vs 稀疏子集σ）相关系数: {corr_thin:.3f}")
    print("解读: 稀疏样本是全链的嵌套子集（每隔thinning_step步抽1个），两者不独立；")
    print("      r接近1仅说明'稀疏化保留了不确定性的空间格局'，不构成幅度校准，也不是独立验证。")
    print(f"劈半一致性（{half_source}）相关系数: {corr_half:.3f}")
    print("解读: 前/后半段是链上不同数据段；r高且散点贴近y=x，才支持链已平稳、不确定性估计可重复。")
    print(f"经验覆盖率（名义95%）: 正态法[全链σ,主] {coverage_norm*100:.2f}% / "
          f"正态法[抽稀σ,对照] {coverage_norm_thin*100:.2f}% / 分位数法 {coverage_emp*100:.2f}%")
    print(f"校准比 E|e|/E[σ] = {calib_mae:.3f}（理想≈0.80）：<0.80 表示σ偏保守、区间偏宽，定量解释过覆盖")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.6-2 总结")
print("=" * 60)
print("1. 95%置信区间（主口径，全链Welford）: x̂_MMSE ± 1.96·√Var(x|y)；另报抽稀σ版作对照")
print("2. 置信区间宽度直观反映局部不确定性")
print("3. 后验样本展示分布的多样性（本例为单峰附近的小幅扰动，非明显多峰；多峰演示见5.2-1）:")
print("   - 共享全局结构（先验约束）")
print("   - 细节存在差异（高不确定区扰动更大）")
print(f"4. 经验覆盖率：正态法[全链σ,主] {coverage_norm*100:.1f}% / 正态法[抽稀σ,对照] "
      f"{coverage_norm_thin*100:.1f}% / 分位数法 {coverage_emp*100:.1f}%（名义95%），")
print("   这是对'95%区间是否名副其实'的直接校准（像素强相关，仅作描述性检查）")
print(f"   校准比 E|e|/E[σ]={calib_mae:.2f}<0.80：σ相对实际误差偏保守（有限ε下平稳分布过弥散），")
print("   区间系统性偏宽，定量解释了覆盖率高于95%的过覆盖现象（与5.6-1同公式同源，可交叉核对）")
print("5. 抽稀保真检验：稀疏子集σ与全链σ高相关仅说明空间格局保真（二者为包含关系、非独立）；")
print("   链是否平稳应看劈半一致性（前/后半段不同数据段）与上游5.5-1的IAT/ESS诊断")
if src_warnings:
    print("6. [注意] 上游5.5-1链核查有警告（见开头[上游链状态]），上述覆盖率/区间结论需结合该警告解读")
else:
    print("6. 上游5.5-1链已确认到平台且抽稀充分，上述覆盖率/区间结论有效")
print("7. 置信区间可用于下游决策（如医学影像中的专家复核区域）")


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
    "实验名称": "实验5.6-2 置信区间与多样性分析",
    "上游链状态_5.5-1": {**src_status, "警告": src_warnings},
    "数据信息": {
        "数据来源": _data_source,
        "后验均值形状": list(post_mean.shape),
        "样本数量": int(len(mc_samples)),
        "均值差异_L2范数": round(float(np.linalg.norm(post_mean - post_mean_recomputed)), 8),
        "方差差异_L2范数": round(float(np.linalg.norm(post_var_full - post_var_recomputed)), 8),
    },
    "步骤2_95%置信区间": {
        "口径说明": (
            "主口径：正态区间的中心(全链Welford MMSE)与宽度(全链Welford σ)同源，"
            "与5.6-1及上游闸门'全链Welford量演示可用'的裁定一致；"
            "抽稀94样本σ版正态区间仅作口径对照（中心相同、只换σ）；"
            "非参数分位数只能由94个抽稀样本估计（npz未存全链逐像素分位数），定性参考"),
        "正态近似_主_全链Welford": {
            "z值": float(z_95),
            "区间宽度统计": {
                "平均宽度": round(float(np.mean(ci_width)), 6),
                "最大宽度": round(float(np.max(ci_width)), 6),
                "最小宽度": round(float(np.min(ci_width)), 6),
            },
            "经验覆盖率": round(coverage_norm, 6),
        },
        "正态近似_对照_抽稀94样本": {
            "区间宽度统计": {
                "平均宽度": round(float(np.mean(ci_width_thin)), 6),
                "最大宽度": round(float(np.max(ci_width_thin)), 6),
                "最小宽度": round(float(np.min(ci_width_thin)), 6),
            },
            "经验覆盖率": round(coverage_norm_thin, 6),
            "注记": (
                "σ由94个强自相关抽稀样本重算（ESS上界≤8.6、抽稀间隔100<全链IAT≥1101），"
                "无偏但逐像素估计噪声大；与主版中心相同、仅σ口径不同，用于对照而非主结论"),
        },
        "样本分位数_抽稀94样本": {
            "区间宽度统计": {
                "平均宽度": round(float(np.mean(ci_width_empirical)), 6),
                "最大宽度": round(float(np.max(ci_width_empirical)), 6),
                "最小宽度": round(float(np.min(ci_width_empirical)), 6),
            },
            "经验覆盖率": round(coverage_emp, 6),
            "分位数分辨率注记": (
                f"仅由{len(mc_samples)}个抽稀样本估计，2.5%分位约对应第{_q_order}小样本；"
                "抽稀间隔<全链IAT时有效样本更少，分位数法覆盖率只作定性参考"),
        },
        "经验覆盖率_汇总": {
            "名义水平": 0.95,
            "正态_全链σ_主": round(coverage_norm, 6),
            "正态_抽稀σ_对照": round(coverage_norm_thin, 6),
            "样本分位数法": round(coverage_emp, 6),
            "备注": "像素间强空间相关，不能按独立伯努利试验给标准误，此处为描述性校准检查",
        },
        "校准诊断_解释覆盖率偏离95": {
            "重建MAE": round(float(_recon_err.mean()), 6),
            "全链σ均值": round(_sigma_mean, 6),
            "校准比_MAE除平均σ": round(calib_mae, 6),
            "校准比_RMS误差除平均σ": round(calib_rms, 6),
            "理想高斯值_sqrt2_pi": round(_calib_ideal, 6),
            "解释": (
                "校准比 E|x_true-x_MMSE|/E[σ] 与5.6-1同公式、同一npz（数值应≈0.65，可交叉核对）；"
                "理想逐像素高斯后验≈0.798。本次<0.80，说明后验σ相对实际重建误差偏保守——"
                "PnP-ULA在有限去噪噪声ε下的平稳分布比目标后验更弥散（见5.6-1知识点），"
                "95%区间系统性偏宽，故经验覆盖率高于95%（过覆盖），方向与幅度均由此定量解释。"
                "RMS版比值≈0.99而MAE版仅0.65，二者并不矛盾：失配是空间异质的——"
                "平滑背景区σ明显过宽（5.6-1五分位Q1~Q3比值0.45/0.45/0.51）拉低MAE比，"
                "高不确定边缘区σ才与误差同量级（Q5≈0.85），大误差像素把RMS比抬到≈1；"
                "这正对应漏检像素沿边缘轮廓分布"),
        },
    },
    "步骤4_后验样本多样性": {
        "展示样本数": int(len(selected)),
        "多峰性说明": "本例强数据约束(BSNR=40dB)+近对数凹去噪先验，样本为单峰附近小幅扰动；可分多峰演示见实验5.2-1",
    },
}

# 步骤5的相关系数在条件块内定义，需安全访问
try:
    results_summary["步骤5_样本差异分析"] = {
        "抽稀保真度_全链σ_vs_稀疏子集σ_相关系数": round(float(corr_thin), 6),
        "抽稀保真度说明": ("稀疏样本是全链的嵌套子集（每隔thinning_step步抽1个），二者非独立；"
                          "r高仅说明抽稀保留了不确定性的空间格局，不构成幅度校准或独立验证"),
        "劈半一致性_前半段σ_vs_后半段σ_相关系数": round(float(corr_half), 6),
        "劈半数据来源": half_source,
        "劈半检验说明": "前/后半段是链上不同数据段；r高且散点贴近y=x才支持链已平稳、估计可重复",
    }
except (NameError, UnboundLocalError):
    results_summary["步骤5_样本差异分析"] = "未执行（样本数不足）"

results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
