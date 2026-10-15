"""
经验贝叶斯参数估计：边际似然最大化
对应章节：3.6 收敛性分析与正则化参数选择（经验贝叶斯部分）
知识点：边际似然 p(y|alpha)、经验贝叶斯策略、高斯共轭闭式解、Fisher恒等式

严谨版说明（二次复核后升级）：
  - 早期版本用 100 点粗对数网格，经验贝叶斯(EB)与 PSNR 最优(oracle)恰好舍入到同一格点，
    造成"完全相同"的假象。现用 400 点网格 + 连续峰值细分，如实呈现二者"接近但不同"；
  - 稳健性实验由 30 个种子扩到 300 个种子，并补充：
      * Spearman 秩相关（抗离群点）与 Pearson 相关的 bootstrap 95% 置信区间；
      * 原 30 种子的 Pearson=-0.51 受单个离群种子(seed=62)放大，
        300 种子下稳定在约 -0.35（Spearman 同量级）——EB 逐样本不复现 oracle 的 alpha，
        这是噪声通道的固有机制（噪声偏大时 EB 上调 alpha、MSE 最优却要求更强收缩）；
      * 但重建 PSNR 损失中位数仅约 0.06 dB，EB 稳健逼近的是 oracle 的"性能"而非"参数"；
  - 新增"尺度自适应"对照：信号尺度未知时，固定常数 alpha 会随尺度失配而崩坏
    （尺度 0.1 时损失约 11 dB），而 EB 的 alpha 自动按 c^2 缩放、性能始终贴近 oracle，
    这才是"无需真解的自动选参"真正不可替代的价值。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
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
    SAVE_DIR = os.path.join(_gdrive, '经验贝叶斯参数估计')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    # 修复：字体目录锚定脚本所在目录，保证从任意工作目录运行都能加载
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")

np.random.seed(42)

# ══════════════════════════════════════════════════════════
# 1. 构造小规模问题
# ══════════════════════════════════════════════════════════
n = 50
m = 40
A = np.random.randn(m, n) / np.sqrt(m)

x_true = np.zeros(n)
x_true[5] = 1.0
x_true[15] = 0.8
x_true[30] = -0.5

sigma2 = 0.01
y = A @ x_true + np.sqrt(sigma2) * np.random.randn(m)
peak = np.max(np.abs(x_true))

# ══════════════════════════════════════════════════════════
# 2. 边际似然（高斯共轭闭式解）
# ══════════════════════════════════════════════════════════
def marginal_likelihood(y, A, alpha, sigma2):
    """
    高斯共轭下的边际似然 log p(y|alpha)
    边际协方差: C = sigma^2 * I + alpha * A * A^T

    高斯共轭情形下 p(y|alpha) 有解析闭式解，无需 EM / MCMC / Monte Carlo gradient。

    Fisher恒等式（标准形式）:
        nabla_alpha log p(y|alpha) = E_{p(x|y,alpha)}[nabla_alpha log p(y,x|alpha)]
    当似然 p(y|x) 不依赖 alpha 时（即 sigma^2 固定、alpha 仅出现在先验中），
    nabla_alpha log p(y|x) = 0，上式右端退化为：
        E_{p(x|y,alpha)}[nabla_alpha log p(x|alpha)]
    Fisher恒等式在一般模型中提供梯度的随机近似途径，
    但本实验有闭式边际似然，可直接对 alpha 优化，无需借助Fisher恒等式。
    """
    m = A.shape[0]
    C = sigma2 * np.eye(m) + alpha * A @ A.T
    sign, logdet = np.linalg.slogdet(C)
    C_inv_y = np.linalg.solve(C, y)
    log_ml = -0.5 * m * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * y @ C_inv_y
    return log_ml


def curves_svd(A, y, alphas, x_ref, peak_ref, sigma2=sigma2):
    """
    一次 SVD 向量化计算整条 alpha 网格上的【边际似然】与【重建 PSNR】，
    与 marginal_likelihood / tikhonov_solve 的 dense 闭式解数值一致
    （复核误差 < 1e-13），供多种子批量实验加速。

    令 A = U diag(s) V^T，则
      log p(y|a) = -1/2 sum_i [log(sigma2 + a s_i^2) + z_i^2/(sigma2 + a s_i^2)] - 常数
      x_hat(a)   = V diag( a s_i/(sigma2 + a s_i^2) ) U^T y,  z = U^T y
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    d = s ** 2                                   # AA^T 的非零特征值
    z = U.T @ y
    a = np.asarray(alphas, dtype=float)[:, None]
    den = sigma2 + a * d
    log_ml = (-0.5 * np.sum(np.log(den), axis=1)
              - 0.5 * np.sum(z[None, :] ** 2 / den, axis=1)
              - 0.5 * A.shape[0] * np.log(2 * np.pi))
    filt = (a * s) / den                         # 奇异空间滤波因子
    x_hat = (filt * z[None, :]) @ Vt
    mse = np.mean((x_hat - x_ref[None, :]) ** 2, axis=1)
    psnr = 10 * np.log10(peak_ref ** 2 / mse)
    return log_ml, psnr

# ══════════════════════════════════════════════════════════
# 3. Tikhonov解（用于评估重建质量）
# ══════════════════════════════════════════════════════════
def tikhonov_solve(y, A, alpha, sigma2):
    return np.linalg.solve(
        A.T @ A / sigma2 + np.eye(A.shape[1]) / alpha,
        A.T @ y / sigma2
    )

def psnr_of(y, A, alpha, x_true, sigma2, peak):
    x_hat = tikhonov_solve(y, A, alpha, sigma2)
    mse = np.mean((x_hat - x_true) ** 2)
    return 10 * np.log10(peak ** 2 / mse)

def refine_peak(fun, center, lo=1 / 4, hi=4, npts=4000):
    """在 center 邻域 [center*lo, center*hi] 用细网格求连续极大值点，消除粗网格舍入。"""
    local = np.logspace(np.log10(center * lo), np.log10(center * hi), npts)
    vals = np.array([fun(a) for a in local])
    return local[np.argmax(vals)], vals.max()

# ══════════════════════════════════════════════════════════
# 4. 对比不同参数选择方法（400 点网格 + 连续细分，seed=42 单实例）
# ══════════════════════════════════════════════════════════
alphas = np.logspace(-4, 2, 400)

# (a) 边际似然（经验贝叶斯）
log_ml = np.array([marginal_likelihood(y, A, alpha, sigma2) for alpha in alphas])
alpha_eb = alphas[np.argmax(log_ml)]
alpha_eb_cont, log_ml_max = refine_peak(
    lambda a: marginal_likelihood(y, A, a, sigma2), alpha_eb)

# (b) PSNR（需知真解，oracle 参考）
psnrs = np.array([psnr_of(y, A, alpha, x_true, sigma2, peak) for alpha in alphas])
alpha_opt = alphas[np.argmax(psnrs)]
alpha_opt_cont, psnr_opt = refine_peak(
    lambda a: psnr_of(y, A, a, x_true, sigma2, peak), alpha_opt)
psnr_eb = psnr_of(y, A, alpha_eb, x_true, sigma2, peak)

print("=" * 60)
print("经验贝叶斯参数估计")
print("=" * 60)
print(f"[400点网格]  经验贝叶斯 alpha_hat = {alpha_eb:.4f}   oracle(PSNR最优) = {alpha_opt:.4f}")
print(f"[连续细分]   经验贝叶斯 alpha_hat = {alpha_eb_cont:.4f}   oracle(PSNR最优) = {alpha_opt_cont:.4f}")
print(f"  两者 alpha 相对差 = {100*(alpha_opt_cont/alpha_eb_cont-1):.2f}%  （非零，说明并非'完全相同'）")
print(f"  但重建 PSNR：EB={psnr_eb:.3f} dB, oracle={psnr_opt:.3f} dB, 差={psnr_opt-psnr_eb:.4f} dB")
print(f"  对数边际似然最大值: log p(y|alpha) = {log_ml_max:.2f}")

# ══════════════════════════════════════════════════════════
# 5. 多种子稳健性（300 个种子）：EB 可靠逼近的是 oracle 的"性能"
#    数据生成保持 legacy 随机流（与早期 30 种子版本的前 30 个实例完全一致），
#    评估改用 SVD 向量化；统计量补充 Spearman 与 bootstrap 置信区间。
# ══════════════════════════════════════════════════════════
def rankdata_avg(v):
    """平均秩（处理网格 argmax 造成的并列），无需引入 scipy 依赖。"""
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind='mergesort')
    ranks = np.empty(len(v))
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    return ranks

seeds = list(range(42, 342))
eb_list, opt_list, gap_list = [], [], []
for s in seeds:
    np.random.seed(s)
    A_s = np.random.randn(m, n) / np.sqrt(m)
    y_s = A_s @ x_true + np.sqrt(sigma2) * np.random.randn(m)
    lm, ps = curves_svd(A_s, y_s, alphas, x_true, peak)
    ie, io = np.argmax(lm), np.argmax(ps)
    eb_list.append(alphas[ie])
    opt_list.append(alphas[io])
    gap_list.append(ps.max() - ps[ie])
eb_arr = np.array(eb_list)
opt_arr = np.array(opt_list)
gap_arr = np.array(gap_list)
log_eb, log_opt = np.log10(eb_arr), np.log10(opt_arr)

ratio = eb_arr / opt_arr
within2_mask = (ratio < 2) & (ratio > 0.5)
within2 = int(within2_mask.sum())
pearson = float(np.corrcoef(log_eb, log_opt)[0, 1])
spearman = float(np.corrcoef(rankdata_avg(log_eb), rankdata_avg(log_opt))[0, 1])

# bootstrap 95% 置信区间（独立随机流，不污染数据生成）
_boot_rng = np.random.RandomState(20240101)
_n_boot = 2000
_boots = np.empty(_n_boot)
for _b in range(_n_boot):
    _idx = _boot_rng.randint(0, len(seeds), len(seeds))
    _boots[_b] = np.corrcoef(log_eb[_idx], log_opt[_idx])[0, 1]
pearson_ci = np.percentile(_boots, [2.5, 97.5])

# 小样本对照：仅前 30 个种子（早期版本口径）
pearson_first30 = float(np.corrcoef(log_eb[:30], log_opt[:30])[0, 1])
spearman_first30 = float(
    np.corrcoef(rankdata_avg(log_eb[:30]), rankdata_avg(log_opt[:30]))[0, 1])

worst_i = int(gap_arr.argmax())
big_loss = gap_arr > 0.5

print("\n" + "=" * 60)
print(f"[多种子稳健性 n={len(seeds)}]")
print("=" * 60)
print(f"  alpha 中位数：EB={np.median(eb_arr):.4f}, oracle={np.median(opt_arr):.4f}")
print(f"  EB 落在 oracle 2倍范围内的种子数：{within2}/{len(seeds)} ({100*within2/len(seeds):.0f}%)")
print(f"  log10(alpha) Pearson 相关 = {pearson:.3f}, bootstrap95%CI=[{pearson_ci[0]:.2f},{pearson_ci[1]:.2f}]")
print(f"  log10(alpha) Spearman 秩相关 = {spearman:.3f}（抗离群点）")
print(f"  [小样本对照] 仅前30种子：Pearson={pearson_first30:.3f}, Spearman={spearman_first30:.3f}"
      f"  —— 30 种子点估计被离群点放大")
print(f"  重建 PSNR 差距 (oracle-EB)：均值 {gap_arr.mean():.3f} dB, 中位 {np.median(gap_arr):.3f} dB,"
      f" 最大 {gap_arr.max():.3f} dB")
print(f"  PSNR 损失 >0.5 dB 的种子：{int(big_loss.sum())}/{len(seeds)}"
      f"（最差 seed={seeds[worst_i]}，{gap_arr[worst_i]:.2f} dB）")

# ══════════════════════════════════════════════════════════
# 5b. 尺度自适应对照：信号尺度 c 未知时，EB 才真正不可替代
#     固定 alpha=0.038 只在尺度碰巧匹配(c≈1)时可用；
#     EB 对每个实例独立估计，alpha 应随信号能量按 c^2 缩放。
# ══════════════════════════════════════════════════════════
FIXED_ALPHA = 0.038                  # 尺度 c=1 下的"好常数"
scales = np.array([0.1, 0.3, 1.0, 3.0, 10.0])
scale_seeds = list(range(42, 102))   # 每尺度 60 个实例
wide_grid = np.logspace(-6, 4, 600)

med_eb_c, med_opt_c, gap_eb_c, gap_fix_c = [], [], [], []
for c in scales:
    x_c = c * x_true
    peak_c = c * peak
    aes, aos, geb, gfx = [], [], [], []
    for s in scale_seeds:
        np.random.seed(s)
        A_s = np.random.randn(m, n) / np.sqrt(m)
        y_s = A_s @ x_c + np.sqrt(sigma2) * np.random.randn(m)
        lm, ps = curves_svd(A_s, y_s, wide_grid, x_c, peak_c)
        ie, io = np.argmax(lm), np.argmax(ps)
        aes.append(wide_grid[ie]); aos.append(wide_grid[io])
        geb.append(ps.max() - ps[ie])
        # 固定常数 alpha=0.038（精确值，非网格最近点）
        ps_fixed = curves_svd(A_s, y_s, np.array([FIXED_ALPHA]), x_c, peak_c)[1][0]
        gfx.append(ps.max() - ps_fixed)
    med_eb_c.append(np.median(aes)); med_opt_c.append(np.median(aos))
    gap_eb_c.append(np.mean(geb)); gap_fix_c.append(np.mean(gfx))
med_eb_c, med_opt_c = map(np.array, (med_eb_c, med_opt_c))
gap_eb_c, gap_fix_c = map(np.array, (gap_eb_c, gap_fix_c))

print("\n" + "=" * 60)
print(f"[尺度自适应对照 n={len(scale_seeds)}/尺度]  固定常数 alpha={FIXED_ALPHA}")
print("=" * 60)
print("  尺度c  中位alpha_EB  中位alpha_oracle  EB损失dB  固定常数损失dB")
for k, c in enumerate(scales):
    print(f"  {c:5.1f}  {med_eb_c[k]:12.5f}  {med_opt_c[k]:14.5f}  {gap_eb_c[k]:8.3f}  {gap_fix_c[k]:12.3f}")
print("  => EB 的 alpha 近似随 c^2 缩放；尺度失配时固定常数显著崩坏，EB 始终贴近 oracle")

# ══════════════════════════════════════════════════════════
# 6. 可视化 1：边际似然与重建质量随 alpha 变化（seed=42）
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].semilogx(alphas, log_ml, 'b-', linewidth=2)
axes[0].axvline(alpha_eb, color='r', linestyle='--',
                label=rf'EB: $\hat{{\alpha}}={alpha_eb:.4f}$')
axes[0].axvline(alpha_opt, color='g', linestyle=':',
                label=rf'最优: $\alpha={alpha_opt:.4f}$')
axes[0].set_xlabel(r'正则化参数 $\alpha$')
axes[0].set_ylabel(r'$\log p(y|\alpha)$')
axes[0].set_title(r'边际似然 $\log p(y|\alpha)$')
axes[0].legend()
axes[0].grid(True)

axes[1].semilogx(alphas, psnrs, 'b-', linewidth=2)
axes[1].axvline(alpha_eb, color='r', linestyle='--',
                label=rf'EB: $\hat{{\alpha}}={alpha_eb:.4f}$')
axes[1].axvline(alpha_opt, color='g', linestyle=':',
                label=rf'最优: $\alpha={alpha_opt:.4f}$')
axes[1].set_xlabel(r'正则化参数 $\alpha$')
axes[1].set_ylabel('PSNR (dB)')
axes[1].set_title(r'重建质量 vs $\alpha$')
axes[1].legend()
axes[1].grid(True)

plt.suptitle(r'经验贝叶斯参数估计', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_经验贝叶斯参数估计.png'), dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════
# 7. 可视化 2：多种子稳健性（300 种子 alpha 对照散点 + PSNR 差距直方图）
# ══════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))

lims = [min(eb_arr.min(), opt_arr.min()) * 0.5,
        max(eb_arr.max(), opt_arr.max()) * 2.0]
normal = ~big_loss
axes2[0].loglog(opt_arr[normal], eb_arr[normal], 'o', ms=5, alpha=0.65,
                color='steelblue', label=f'各随机种子（损失≤0.5 dB，{int(normal.sum())}个）')
axes2[0].loglog(opt_arr[big_loss], eb_arr[big_loss], 'o', ms=7, alpha=0.9,
                color='crimson', label=f'PSNR损失>0.5 dB（{int(big_loss.sum())}个）')
axes2[0].plot(lims, lims, 'k--', linewidth=1, label=r'$\alpha_{EB}=\alpha_{oracle}$')
axes2[0].fill_between(lims, [l * 0.5 for l in lims], [l * 2 for l in lims],
                      color='gray', alpha=0.15, label='2× 一致带')
axes2[0].set_xlabel(r'oracle $\alpha$（PSNR 最优，需真解）')
axes2[0].set_ylabel(r'经验贝叶斯 $\hat{\alpha}$（仅需 $y$）')
axes2[0].set_title(rf'多种子：$\alpha$ 对照 (Pearson $r={pearson:.2f}$,'
                   rf' Spearman $\rho={spearman:.2f}$)')
axes2[0].legend(fontsize=8.5, loc='upper left')
axes2[0].grid(True, which='both', alpha=0.3)

axes2[1].hist(gap_arr, bins=np.linspace(0, max(gap_arr.max(), 1.3), 27),
              color='steelblue', edgecolor='white')
axes2[1].axvline(0, color='r', linestyle='--', linewidth=1, label='0 dB（与 oracle 持平）')
axes2[1].axvline(np.median(gap_arr), color='orange', linestyle='-', linewidth=1.5,
                 label=rf'中位数 {np.median(gap_arr):.2f} dB')
axes2[1].set_xlabel(r'重建质量差距 PSNR(oracle) $-$ PSNR(EB) (dB)')
axes2[1].set_ylabel('种子数')
axes2[1].set_title(rf'EB 相对 oracle 的 PSNR 损失（均值 {gap_arr.mean():.2f} dB，'
                   rf'中位 {np.median(gap_arr):.2f} dB）')
axes2[1].legend(fontsize=9)
axes2[1].grid(True, alpha=0.3)

plt.suptitle(rf'经验贝叶斯选参的稳健性（{len(seeds)} 个种子）', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_多种子稳健性.png'), dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════
# 7b. 可视化 3：尺度自适应（alpha 随 c^2 缩放 + EB vs 固定常数性能）
# ══════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5))

c_ref = np.logspace(-1.2, 1.2, 50)
axes3[0].loglog(scales, med_eb_c, 'o-', color='crimson', ms=7,
                label=r'EB $\hat{\alpha}$（中位数）')
axes3[0].loglog(scales, med_opt_c, 's--', color='green', ms=6,
                label=r'oracle $\alpha$（中位数）')
axes3[0].loglog(c_ref, FIXED_ALPHA * c_ref ** 2, 'k:', linewidth=1.2,
                label=rf'理论参考 $\alpha \propto c^2$（锚定 {FIXED_ALPHA}）')
axes3[0].set_xlabel(r'信号尺度 $c$（$x = c\,x_0$，未知）')
axes3[0].set_ylabel(r'选择的正则化参数 $\alpha$')
axes3[0].set_title(r'EB 自动跟踪信号尺度（先验方差 $\propto c^2$）')
axes3[0].legend(fontsize=9)
axes3[0].grid(True, which='both', alpha=0.3)

axes3[1].semilogx(scales, gap_eb_c, 'o-', color='crimson', ms=7, label='经验贝叶斯（自动选参）')
axes3[1].semilogx(scales, gap_fix_c, 's--', color='gray', ms=7,
                  label=rf'固定常数 $\alpha={FIXED_ALPHA}$')
axes3[1].axhline(0.5, color='orange', linestyle=':', linewidth=1, label='0.5 dB 损失线')
axes3[1].set_xlabel(r'信号尺度 $c$')
axes3[1].set_ylabel(r'相对 oracle 的 PSNR 损失 (dB)')
axes3[1].set_title(r'尺度失配：固定常数崩坏，EB 始终贴近 oracle')
axes3[1].legend(fontsize=9)
axes3[1].grid(True, which='both', alpha=0.3)
axes3[1].annotate(f'{gap_fix_c[0]:.1f} dB', xy=(scales[0], gap_fix_c[0]),
                  xytext=(scales[0] * 1.6, gap_fix_c[0] - 0.8), fontsize=9, color='gray')

plt.suptitle(r'经验贝叶斯自动选参不可替代的价值：尺度未知时的自适应', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_尺度自适应.png'), dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════
# 8. 结果输出与教学说明
# ══════════════════════════════════════════════════════════
print(f"\n[核心发现]")
print(f"  1. EB 与 oracle 的 alpha 接近但不同（连续值 {alpha_eb_cont:.4f} vs {alpha_opt_cont:.4f}，"
      f"差 {100*(alpha_opt_cont/alpha_eb_cont-1):.1f}%）；早期 100 点粗网格的'完全相同'是舍入假象")
print(f"  2. 但 EB 选参的重建 PSNR（{psnr_eb:.2f} dB）与 oracle（{psnr_opt:.2f} dB）几乎持平——"
      f"PSNR 峰顶是平台，alpha 的小偏移不损害重建")
print(f"  3. {len(seeds)} 种子：alpha 的 Pearson 相关 {pearson:.2f}（CI[{pearson_ci[0]:.2f},"
      f"{pearson_ci[1]:.2f}]）、Spearman {spearman:.2f}，EB 不复现逐样本最优 alpha；"
      f"但 PSNR 损失中位仅 {np.median(gap_arr):.2f} dB，仅 {int(big_loss.sum())} 个种子 >0.5 dB")
print(f"  4. 尺度自适应：固定常数 alpha={FIXED_ALPHA} 在尺度 0.1 时损失 {gap_fix_c[0]:.1f} dB、"
      f"尺度 10 时损失 {gap_fix_c[-1]:.2f} dB；EB 的 alpha 随 c^2 缩放、各尺度损失均 ≤"
      f"{gap_eb_c.max():.2f} dB")
print(f"  5. 结论：经验贝叶斯无需真解即可把重建送到最优平台附近，且在信号尺度未知时"
      f"自动适配，是可靠的自动选参法")

# ===== 保存数值结果 =====
import json

def _to_native(obj):
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except Exception:
        pass
    return obj

results_summary = {
    'n': n,
    'm': m,
    'sigma2': float(round(sigma2, 4)),
    'grid_points': len(alphas),
    'seed_main': 42,
    'alpha_eb_grid': float(round(alpha_eb, 4)),
    'alpha_opt_grid': float(round(alpha_opt, 4)),
    'alpha_eb_cont': float(round(alpha_eb_cont, 5)),
    'alpha_opt_cont': float(round(alpha_opt_cont, 5)),
    'alpha_rel_gap_pct': float(round(100 * (alpha_opt_cont / alpha_eb_cont - 1), 2)),
    'log_ml_max': float(round(log_ml_max, 2)),
    'psnr_eb': float(round(psnr_eb, 3)),
    'psnr_opt': float(round(psnr_opt, 3)),
    'psnr_gap_db': float(round(psnr_opt - psnr_eb, 4)),
    'multiseed': {
        'n_seeds': len(seeds),
        'median_alpha_eb': float(round(np.median(eb_arr), 4)),
        'median_alpha_opt': float(round(np.median(opt_arr), 4)),
        'within_2x_count': within2,
        'within_2x_pct': float(round(100 * within2 / len(seeds), 1)),
        'log_alpha_pearson': float(round(pearson, 3)),
        'pearson_bootstrap95_ci': [float(round(pearson_ci[0], 3)),
                                  float(round(pearson_ci[1], 3))],
        'n_bootstrap': _n_boot,
        'log_alpha_spearman': float(round(spearman, 3)),
        'pearson_first30_seeds': float(round(pearson_first30, 3)),
        'spearman_first30_seeds': float(round(spearman_first30, 3)),
        'psnr_gap_mean_db': float(round(gap_arr.mean(), 3)),
        'psnr_gap_median_db': float(round(np.median(gap_arr), 3)),
        'psnr_gap_max_db': float(round(gap_arr.max(), 3)),
        'n_seeds_gap_gt_0p5db': int(big_loss.sum()),
        'worst_seed': seeds[worst_i],
    },
    'scale_adaptive': {
        'scales': [float(c) for c in scales],
        'n_seeds_per_scale': len(scale_seeds),
        'fixed_alpha_baseline': FIXED_ALPHA,
        'median_alpha_eb': [float(round(v, 5)) for v in med_eb_c],
        'median_alpha_opt': [float(round(v, 5)) for v in med_opt_c],
        'psnr_gap_eb_mean_db': [float(round(v, 3)) for v in gap_eb_c],
        'psnr_gap_fixed_mean_db': [float(round(v, 3)) for v in gap_fix_c],
    },
}

results_summary = {k: _to_native(v) for k, v in results_summary.items()}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 60}")
print(f"实验完成")
print(f"输出图片: 步骤1_经验贝叶斯参数估计.png, 步骤2_多种子稳健性.png, 步骤3_尺度自适应.png")
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
print(f"{'=' * 60}")
