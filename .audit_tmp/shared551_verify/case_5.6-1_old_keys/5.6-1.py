# -*- coding: utf-8 -*-
"""
实验5.6-1 后验标准差与不确定性图
对应章节：5.6 近似理论与收敛保证 -> 5.6.5 像素级不确定性：后验方差
知识点：
  - 后验方差估计：Var(x_i) = 1/(M-B) Σ(X_m^(i) - x_MMSE^(i))^2
  - 高方差区域 = 信息丢失区域
  - 低方差区域 = 数据约束强
  - 注意：后验方差只刻画“数据/采样不确定性”，不包含模型偏差；
    PnP 近似后验对系统性误差（如边缘振铃）会给出偏窄的方差，
    因此方差与误差“正相关”，但绝对幅度可能低估实际误差。
  - 短链强自相关时，逐像素方差估计噪声很大（ESS 远小于名义样本数），
    空间聚合后的结论才稳健。

前置实验：
  - 需要先运行实验5.5-1，生成sampling_results.npz
    （npz 中 post_mean/post_var 为整条链经 Welford 在线累积的统计量，
      mc_samples 为其中抽稀保存的样本，用于教学演示与诊断）

素材来源：
  - Mathematics.../Teaching Unit 2/labs/lab2_PnP_sol.ipynb
    - Cell 18-20: 结果评估与可视化
  - 实验5.3.py 不确定性量化部分
"""

import torch
import numpy as np
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
    SAVE_DIR = os.path.join(_gdrive, '实验5.6-1')
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

rng = np.random.default_rng(42)


# ============================================================
# 工具函数
# ============================================================
def rankdata_average(a):
    """numpy 实现的平均秩（结取平均），避免对 scipy 的硬依赖"""
    a = np.asarray(a).ravel()
    order = np.argsort(a, kind='mergesort')
    sa = a[order]
    ranks_sorted = np.empty(len(a), dtype=np.float64)
    # 结边界
    change = np.flatnonzero(np.diff(sa) != 0)
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change + 1, [len(a)]))
    for st, en in zip(starts, ends):
        ranks_sorted[st:en] = 0.5 * (st + en - 1) + 1.0  # 1-based 平均秩
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def ess_initial_positive(samples):
    """
    从（抽稀）链样本估计有效样本量 ESS。
    先对全部像素汇总逐滞后自相关，再用 Geyer initial-positive-sequence
    截断（逐对 rho_{2k-1}+rho_{2k}，遇到非正即停）。
    返回 (ESS, rho数组)。
    """
    M = samples.shape[0]
    X = samples.reshape(M, -1).astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    c0 = np.sum(X * X)
    if c0 == 0:
        return float(M), np.zeros(M - 1)
    rhos = []
    for k in range(1, M):
        ck = np.sum(X[:M - k] * X[k:])
        rhos.append((ck / c0) * (M / (M - k)))
    rhos = np.asarray(rhos)
    pair_sum = 0.0
    for j in range(0, len(rhos) - 1, 2):
        pair = rhos[j] + rhos[j + 1]
        if pair <= 0:
            break
        pair_sum += pair
    return M / (1.0 + 2.0 * pair_sum), rhos


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
post_mean = data['post_mean']      # 整条链 Welford 累积的后验均值（主 MMSE 估计）
post_var = data['post_var']        # 整条链 Welford 累积的后验方差（主方差估计）
mc_samples = data['mc_samples']    # 抽稀保存的链样本（教学/诊断用）
x_true = data['x_true']
y_obs = data['y_obs']

print(f"后验均值形状: {post_mean.shape}")
print(f"后验方差形状: {post_var.shape}")
print(f"抽稀样本数量: {len(mc_samples)}")

# ------------------------------------------------------------
# 上游链(5.5-1)收敛闸门：判定实现与5.6-2/5.6-3共用（pnp_ula_common）。
# 新版 5.5-1 的 npz 落盘了 trace_psnr/trace_fid/chain_config；旧版没有，
# 此时只能提示“无法核查”，不得默认上游已收敛。
# ------------------------------------------------------------
src_status, src_warnings, _src_extras = upstream_chain_gate(
    data, _data_path, purpose='后验方差/校准')
print_gate(src_status, src_warnings, '下游不确定性结论有效')

# ------------------------------------------------------------
# 诊断：抽稀样本重算统计量 vs 全链 Welford 统计量
# 正确解释：mc_samples 是强自相关短链的抽稀状态，与全链统计量
# “共享信息”，所以两者差异很小（远小于 iid 情形的 σ/√m），
# 这不是异常，更不是“差异偏大”。真正需要披露的是 ESS 很小、
# 逐像素方差噪声大。下列数值仅作诊断，主结果一律使用全链 Welford 量。
# ------------------------------------------------------------
post_mean_thin = np.mean(mc_samples, axis=0)
post_var_thin = np.var(mc_samples, axis=0, ddof=1)

mean_diff = np.linalg.norm(post_mean - post_mean_thin)
var_diff = np.linalg.norm(post_var - post_var_thin)
mean_diff_rms = float(np.sqrt(np.mean((post_mean - post_mean_thin) ** 2)))

ess, rhos = ess_initial_positive(mc_samples)
lag1 = float(rhos[0]) if len(rhos) else 0.0
mean_std_thin = float(np.sqrt(post_var_thin).mean())
mc_se_thin = mean_std_thin / np.sqrt(ess)   # 抽稀均值的每像素蒙特卡洛标准误

print("\n[诊断] 全链 Welford 统计量 vs 抽稀样本重算量：")
print(f"  均值差异 L2范数      : {mean_diff:.6f}")
print(f"  均值差异 每像素RMS   : {mean_diff_rms:.6f} "
      f"(占抽稀平均std的 {mean_diff_rms / max(mean_std_thin, 1e-12) * 100:.1f}%)")
print(f"  方差差异 L2范数      : {var_diff:.6f}")
print(f"  抽稀序列 lag1 自相关 : {lag1:.3f}")
print(f"  抽稀序列 ESS        : {ess:.1f} (名义样本数 {len(mc_samples)})")
print(f"  抽稀均值 MC标准误    : {mc_se_thin:.5f}")
print("  解读: 链强自相关(ESS≪名义样本数)使抽稀样本与全链统计量高度共享信息，")
print("        差异很小属正常；但逐像素方差估计噪声大，结论应以空间聚合为主。")


# ============================================================
# 步骤2：计算后验标准差（主结果使用全链 Welford 方差）
# ============================================================
print("\n" + "=" * 60)
print("步骤2：计算后验标准差")
print("=" * 60)

post_var = np.maximum(post_var, 0.0)   # 数值保险
post_std = np.sqrt(post_var)

print(f"后验标准差统计（全链 Welford）:")
print(f"  平均值: {np.mean(post_std):.4f}")
print(f"  最大值: {np.max(post_std):.4f}")
print(f"  最小值: {np.min(post_std):.4f}")
print(f"  中位数: {np.median(post_std):.4f}")


# ============================================================
# 步骤3：可视化不确定性
# ============================================================
print("\n" + "=" * 60)
print("步骤3：可视化不确定性")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 三幅灰度图统一动态范围，便于直接对比
_vmin = float(min(x_true.min(), y_obs.min(), post_mean.min()))
_vmax = float(max(x_true.max(), y_obs.max(), post_mean.max()))

axes[0, 0].imshow(x_true, cmap='gray', vmin=_vmin, vmax=_vmax)
axes[0, 0].set_title('原始图像 $x$')
axes[0, 0].axis('off')

axes[0, 1].imshow(y_obs, cmap='gray', vmin=_vmin, vmax=_vmax)
axes[0, 1].set_title('含噪观测 $y$')
axes[0, 1].axis('off')

axes[0, 2].imshow(post_mean, cmap='gray', vmin=_vmin, vmax=_vmax)
axes[0, 2].set_title('后验均值 $\\hat{x}_{MMSE}$')
axes[0, 2].axis('off')

# 第2行：后验标准差、直方图、高不确定性区域
im_std = axes[1, 0].imshow(post_std, cmap='hot')
axes[1, 0].set_title('后验标准差 $\\sqrt{\\mathrm{Var}(x|y)}$（全链）')
axes[1, 0].axis('off')
plt.colorbar(im_std, ax=axes[1, 0], fraction=0.046, pad=0.04)

# 标准差直方图
axes[1, 1].hist(post_std.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='white')
axes[1, 1].axvline(np.mean(post_std), color='red', linestyle='--',
                   label=f'均值: {np.mean(post_std):.4f}')
axes[1, 1].set_xlabel('后验标准差')
axes[1, 1].set_ylabel('像素数量')
axes[1, 1].set_title('不确定性分布')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# 高不确定性区域标记（超过均值+1倍标准差）
threshold = np.mean(post_std) + np.std(post_std)
high_uncertainty = post_std > threshold
axes[1, 2].imshow(high_uncertainty, cmap='Reds')
axes[1, 2].set_title(f'高不确定性区域 (>{threshold:.4f})')
axes[1, 2].axis('off')

fig.suptitle('实验5.6-1 后验标准差与不确定性图（全链 Welford）', fontsize=14, y=1.02)
fig.text(0.5, 0.005,
         f'注：逐像素标准差含有限 ESS（抽稀序列 ESS≈{ess:.0f}，名义 {len(mc_samples)} 样本）的采样噪声，'
         '仅供定性空间定位；定量结论以步骤4的五分位/区域聚合为准。',
         ha='center', fontsize=10)
plt.tight_layout(rect=[0, 0.025, 1, 1])
plt.savefig(os.path.join(SAVE_DIR, '步骤3_后验标准差与不确定性.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"高不确定性像素占比: {np.sum(high_uncertainty) / high_uncertainty.size * 100:.1f}%")


# ============================================================
# 步骤4：不确定性与重建误差的关系
# ============================================================
print("\n" + "=" * 60)
print("步骤4：不确定性与重建误差的关系")
print("=" * 60)

# 重建误差（相对于全链 MMSE 估计）
error = np.abs(x_true - post_mean)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 误差图
im_err = axes[0].imshow(error, cmap='hot')
axes[0].set_title('重建误差 $|x - \\hat{x}_{MMSE}|$')
axes[0].axis('off')
plt.colorbar(im_err, ax=axes[0], fraction=0.046, pad=0.04)

# 散点图：按不确定性五分位分层抽样（仅为让尾部点可见；相关系数仍用全部像素）
post_std_flat = post_std.flatten()
error_flat = error.flatten()
n_pix = len(post_std_flat)
_qedges = np.percentile(post_std_flat, np.linspace(0, 100, 6))
_qid = np.clip(np.digitize(post_std_flat, _qedges[1:-1]), 0, 4)
n_per_bin = 100
idx_parts = []
for _q in range(5):
    cand = np.flatnonzero(_qid == _q)
    take = min(n_per_bin, len(cand))
    idx_parts.append(rng.choice(cand, size=take, replace=False))
idx = np.concatenate(idx_parts)
axes[1].scatter(post_std_flat[idx], error_flat[idx], alpha=0.3, s=8)
axes[1].set_xlim(0, float(post_std_flat.max()) * 1.02)
axes[1].set_ylim(0, float(error_flat.max()) * 1.05)
axes[1].set_xlabel('不确定性（标准差）')
axes[1].set_ylabel('重建误差')
axes[1].set_title('不确定性 vs 误差（五分位分层抽样展示）')

corr = float(np.corrcoef(post_std_flat, error_flat)[0, 1])
corr_spearman = float(np.corrcoef(rankdata_average(post_std_flat),
                                  rankdata_average(error_flat))[0, 1])
axes[1].annotate(f'Pearson $r$ = {corr:.3f}\nSpearman $r_s$ = {corr_spearman:.3f}\n'
                 f'(全部 {n_pix} 像素)',
                 xy=(0.05, 0.95), xycoords='axes fraction', va='top',
                 fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1].grid(alpha=0.3)

# 分区域统计：_qid 保证每个像素恰好落入一箱（含最大值结）
n_bins = 5
bin_means, bin_std, bin_ratio, bin_labels = [], [], [], []
for i in range(n_bins):
    mask = (_qid == i)
    if np.sum(mask) > 0:
        bin_means.append(float(np.mean(error_flat[mask])))
        bin_std.append(float(np.mean(post_std_flat[mask])))
        # 校准比：理想高斯后验下 E|误差|/std ≈ sqrt(2/pi)≈0.80；
        # 系统性模型偏差不进后验方差时该比值 >1，即方差偏窄/低估误差
        bin_ratio.append(bin_means[-1] / max(bin_std[-1], 1e-12))
        bin_labels.append(f'Q{i+1}')

_bars = axes[2].bar(range(len(bin_means)), bin_means, color='steelblue', alpha=0.7)
# 柱顶标注校准比 MAE/std：理想高斯后验≈0.80，明显>0.80 表示方差偏窄
for _rect, _rr in zip(_bars, bin_ratio):
    axes[2].text(_rect.get_x() + _rect.get_width() / 2, _rect.get_height(),
                 f'{_rr:.2f}', ha='center', va='bottom', fontsize=11)
axes[2].set_ylim(0, float(max(bin_means)) * 1.12)
axes[2].set_xlabel('不确定性分位数')
axes[2].set_ylabel('平均重建误差')
axes[2].set_title('五分位平均误差（柱顶为 MAE/std，理想≈0.80）')
axes[2].set_xticks(range(len(bin_means)))
axes[2].set_xticklabels(bin_labels)
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤4_不确定性与误差关系.png'), dpi=150, bbox_inches='tight')
plt.close()

calib_overall = float(error.mean() / max(post_std.mean(), 1e-12))
print(f"不确定性-误差 Pearson 相关 : {corr:.3f}")
print(f"不确定性-误差 Spearman 秩相关: {corr_spearman:.3f}")
print("五分位校准（平均误差 / 平均后验std，理想≈0.80）:")
for i, (lab, m_, s_, rr) in enumerate(zip(bin_labels, bin_means, bin_std, bin_ratio)):
    print(f"  {lab}: MAE={m_:.5f}, std={s_:.5f}, 比值={rr:.2f}")
print(f"整体 MAE/平均std = {calib_overall:.2f}")
print("解读: 正相关成立，方差可定位“容易出错”的区域；但比值明显>0.8，")
print("      说明后验方差未覆盖系统性模型偏差（边缘振铃等），绝对不确定性偏窄。")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.6-1 总结")
print("=" * 60)
print("1. 后验标准差提供像素级的不确定性度量（主结果用全链 Welford 统计量）")
print("2. 高方差区域对应信息丢失区域（模糊核导致的高频缺失：边缘、纹理）")
print("3. 低方差区域对应数据约束强的区域（平滑背景）")
print("4. 不确定性与重建误差正相关，可定位易错区域；但后验方差只含数据/采样")
print("   不确定性，不含模型偏差，故对系统性误差存在低估（校准比>0.8）")
print(f"5. 抽稀链 ESS≈{ess:.0f}（名义{len(mc_samples)}个样本），逐像素方差噪声大，")
print("   应以空间聚合（分位数、区域均值）后的结论为准")
if src_warnings:
    print("6. [注意] 上游5.5-1链核查有警告（见开头[上游链状态]），本实验结论需结合该警告解读")
else:
    print("6. 上游5.5-1链已确认到平台且抽稀充分，本实验结论有效")


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

quintile_table = [
    {
        "分位": lab,
        "平均重建误差_MAE": round(m_, 6),
        "平均后验标准差": round(s_, 6),
        "校准比_MAE除std": round(rr, 4),
    }
    for lab, m_, s_, rr in zip(bin_labels, bin_means, bin_std, bin_ratio)
]

results_summary = {
    "实验名称": "实验5.6-1 后验标准差与不确定性图",
    "数据来源": "实验5.5-1 的 sampling_results.npz（真实 PnP-ULA 采样；主统计量为全链 Welford 累积）",
    "上游链状态_5.5-1": {**src_status, "警告": src_warnings},
    "数据信息": {
        "后验均值形状": list(post_mean.shape),
        "抽稀样本数量": int(len(mc_samples)),
        "均值差异_L2范数": round(float(mean_diff), 8),
        "均值差异_每像素RMS": round(mean_diff_rms, 8),
        "方差差异_L2范数": round(float(var_diff), 8),
        "抽稀序列_lag1自相关": round(lag1, 6),
        "抽稀序列_有效样本量ESS": round(float(ess), 4),
        "抽稀均值_MC标准误": round(float(mc_se_thin), 6),
    },
    "步骤2_后验标准差统计_全链Welford": {
        "平均值": round(float(np.mean(post_std)), 6),
        "最大值": round(float(np.max(post_std)), 6),
        "最小值": round(float(np.min(post_std)), 6),
        "中位数": round(float(np.median(post_std)), 6),
    },
    "步骤3_不确定性": {
        "高不确定性阈值": round(float(threshold), 6),
        "高不确定性像素占比_百分比": round(float(np.sum(high_uncertainty) / high_uncertainty.size * 100), 4),
    },
    "步骤4_不确定性与误差关系": {
        "Pearson相关系数": round(corr, 6),
        "Spearman秩相关系数": round(corr_spearman, 6),
        "整体校准比_MAE除平均std_理想约0.80": round(calib_overall, 4),
        "五分位校准表": quintile_table,
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
