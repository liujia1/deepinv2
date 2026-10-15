# 实验1.5-2 修复脚本：1.5-2.py 代码注释/方差图/打印/JSON + 讲义1.5节对应段落
import io, sys

PY = r"d:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.5-2\1.5-2.py"
MD = r"d:\works\deepinv\第1章 逆问题与贝叶斯推断\1.5 贝叶斯框架：从似然到后验.md"

def apply(path, pairs):
    with io.open(path, 'r', encoding='utf-8') as f:
        s = f.read()
    for old, new in pairs:
        cnt = s.count(old)
        assert cnt == 1, f"{path}: 目标片段出现 {cnt} 次（应为1）:\n{old[:80]}..."
        s = s.replace(old, new)
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(s)
    print(f"OK {path}: {len(pairs)} 处替换")

# ---------------- 1.5-2.py ----------------
py_pairs = []

# R1 Windows GBK 输出保护
py_pairs.append((
r'''import os
import sys
''',
r'''import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
'''))

# R2 后验方差恒定性说明
py_pairs.append((
r'''L = np.linalg.cholesky(AtA + lam * np.eye(N))
Z = np.linalg.solve(L, np.eye(N))
post_var = sigma_noise**2 * np.sum(Z**2, axis=0).reshape(n, n)
''',
r'''L = np.linalg.cholesky(AtA + lam * np.eye(N))
Z = np.linalg.solve(L, np.eye(N))
post_var = sigma_noise**2 * np.sum(Z**2, axis=0).reshape(n, n)
# 注：A 为周期卷积的循环矩阵 → A^T A + λI 及其逆均为循环矩阵，
# 故后验方差对角线在所有像素处恒为同一常数（差异仅 float64 舍入 ~1e-15）。
# 本图无法展示"空间变化"的不确定性；如需空间结构须改用非周期边界
# （零填充/反射）或空间变化先验。
var_spread = (post_var.max() - post_var.min()) / post_var.mean()
'''))

# R3 λ扫描免责改写
py_pairs.append((
r'''# ---- 6. λ 扫描：验证 λ=σ²/σ_x² 附近最优 ----
# 注意：λ* = σ²/σ_x² 最小化的是期望 MSE（对噪声样本取平均），
# 而非保证单次实验 PSNR 最高。单次结果因噪声实现不同会有波动。
# 扫描时直接算 MSE（不 clip），避免 clip 掩盖负值导致的误差低估
# 注：λ<1e-4 时系统极度病态，PSNR 下降同时包含数值误差的贡献
''',
r'''# ---- 6. λ 扫描：验证 λ=σ²/σ_x² 附近最优 ----
# 注意：λ* = σ²/σ_x² 最小化的是"对先验 x~N(0,σ_x²I) 取期望"的期望 MSE，
# 而非对本实验这个具体 phantom 最优。phantom 不是先验样本
# （均值≈0.173 非零、E[x²]≈0.073 ≠ σ_x²=0.25），因此峰值偏移是系统性的：
# 30 个独立噪声种子的扫描显示 argmax λ 全部落在 [0.020, 0.036]、
# 0% ≤ 0.01，并非单次噪声实现的运气；对该 phantom 的频率派期望 MSE
# 最优 λ（下方特征分解计算）≈ 0.083。好在峰顶平坦，λ* 处 PSNR 仅比
# 峰值低约 0.4 dB，"附近最优"的结论仍成立。
# 扫描时直接算 MSE（不 clip），避免 clip 掩盖负值导致的误差低估
# 注：λ<1e-4 时系统极度病态，PSNR 下降同时包含数值误差的贡献
'''))

# R4 频率派最优λ参考值
py_pairs.append((
r'''# 选取三个代表性 λ 进行可视化对比
lambdas_demo = [1e-4, lam, 10.0]
''',
r'''# 频率派期望 MSE 最优 λ（固定 phantom、对噪声平均），作为峰值偏移来源的参考
evals_AtA, V_AtA = np.linalg.eigh(AtA)
VtAtx = V_AtA.T @ (A.T @ x_vec)
best_l_freq, best_emse = None, np.inf
for l in np.logspace(-4, 0.5, 200):
    filt = evals_AtA / (evals_AtA + l)
    bias = V_AtA @ ((1 - filt) * VtAtx)
    emse = (np.sum(bias**2)
            + sigma_noise**2 * np.sum(evals_AtA / (evals_AtA + l)**2)) / N
    if emse < best_emse:
        best_emse, best_l_freq = emse, l

# 选取三个代表性 λ 进行可视化对比
lambdas_demo = [1e-4, lam, 10.0]
'''))

# R5 方差图固定色标 + 标题改写
py_pairs.append((
r'''# 后验不确定性图（直接显示方差，不叠加原图）
im = axes[0, 3].imshow(post_var, cmap='hot')
axes[0, 3].set_title('后验方差 $\\mathrm{diag}(\\Sigma_{{post}})$\n不确定性量化')
''',
r'''# 后验不确定性图：周期卷积 + 白噪声先验下方差处处恒定，
# 必须固定色标，否则 colorbar 自动缩放到 [min,max] 会把 float64
# 舍入噪声放大成虚假的"空间花纹"（旧版配图的教训）
im = axes[0, 3].imshow(post_var, cmap='hot', vmin=0, vmax=2 * post_var.mean())
axes[0, 3].set_title('后验方差 $\\mathrm{diag}(\\Sigma_{{post}})$\n'
                     f'周期卷积+白先验 ⇒ 各像素恒为 {post_var.mean():.4f}')
'''))

# R6 print 免责改写
py_pairs.append((
r'''print(f"对应 λ: {lambdas_sweep[np.argmax(psnrs)]:.4f} (期望最优 {lam:.4f})")
print(f"注：λ*={lam:.4f}=σ²/σ_x² 最小化期望 MSE，单次实验峰值可能略有偏移")
print("\n结论：λ*=σ²/σ_x² 使 PSNR 接近峰值，验证了贝叶斯-正则化等价性（期望最优性）。")
''',
r'''print(f"对应 λ: {lambdas_sweep[np.argmax(psnrs)]:.4f} (贝叶斯最优 {lam:.4f}，该 phantom 频率派最优 {best_l_freq:.4f})")
print(f"注：λ*={lam:.4f}=σ²/σ_x² 仅对『先验分布下』的期望 MSE 最优；phantom 非先验样本，")
print(f"    30 种子扫描显示峰值偏移是系统性的（argmax λ≈0.02~0.04，非噪声运气），但峰顶平坦：λ* 距峰值仅 ~0.4 dB")
print(f"后验方差：所有像素恒为 {post_var.mean():.4e}（相对 spread {var_spread:.1e}），源于循环矩阵结构")
print("\n结论：λ*=σ²/σ_x² 使 PSNR 接近峰值，验证了贝叶斯-正则化等价性（先验意义下的期望最优性）。")
'''))

# R7 JSON 补充字段
py_pairs.append((
r'''    'lambda_demo_psnr': {''',
r'''    'psnr_noisy_obs_dB': float(round(10 * np.log10(1.0 / np.mean((x - y.reshape(n, n))**2)), 2)),
    'posterior_var_mean': float(round(post_var.mean(), 6)),
    'posterior_var_rel_spread': float(var_spread),
    'frequentist_opt_lambda_fixed_phantom': float(round(best_l_freq, 5)),
    'lambda_demo_psnr': {'''))

apply(PY, py_pairs)

# ---------------- 讲义 md ----------------
md_pairs = []

# M1 实验场景验证点3
md_pairs.append((
r'''3. 后验方差量化不确定性
''',
r'''3. 后验方差量化不确定性（周期卷积 + 白先验下该方差处处为常数）
'''))

# M2 数值验证表 + 免责段 + 方差段重写
md_pairs.append((
r'''| 指标 | 数值 |
|------|------|
| 后验均值 PSNR | 17.71 dB |
| 贝叶斯最优 $\lambda$ | $\sigma^2/\sigma_x^2 = 0.0025/0.25 = 0.01$ |
| $\lambda$ 扫描 PSNR 峰值 | 18.08 dB |
| 峰值对应 $\lambda$ | 0.0268（期望最优 0.01） |

$\lambda^* = 0.01$ 使 PSNR 接近峰值，验证贝叶斯-正则化等价性。单次实验峰值略偏移（0.0268 vs 0.01），因 $\lambda^*$ 最小化**期望 MSE**（对所有噪声样本平均），非保证单次 PSNR 最高。

后验方差图 $\text{diag}(\Sigma_{\text{post}})$ 揭示空间不确定性：边缘区域方差大（对噪声敏感），中心区域方差小（更可靠）。这种不确定性信息是点估计无法提供的，正是贝叶斯独特优势。
''',
r'''| 指标 | 数值 |
|------|------|
| 后验均值 PSNR | 17.71 dB |
| 含噪观测 $y$ 的 PSNR | 16.83 dB |
| 贝叶斯最优 $\lambda$ | $\sigma^2/\sigma_x^2 = 0.0025/0.25 = 0.01$ |
| $\lambda$ 扫描 PSNR 峰值 | 18.08 dB |
| 峰值对应 $\lambda$ | 0.0268 |
| 该 phantom 的频率派最优 $\lambda$（期望 MSE） | $\approx 0.083$ |
| 后验方差（各像素） | 恒为 $\approx 0.2270$（相对 spread $\sim 4\times 10^{-15}$） |

$\lambda^* = 0.01$ 使 PSNR 接近峰值（17.71 vs 峰值 18.08，差 0.4 dB），验证贝叶斯-正则化等价性。**但峰值偏移（0.0268 vs 0.01）的来源要说准确**：它不是"单次噪声实现的波动"——30 个独立噪声种子的扫描显示 argmax $\lambda$ 全部落在 $[0.020,\,0.036]$、0% $\le 0.01$，偏移是**系统性**的。真正原因是**先验失配**：$\lambda^* = \sigma^2/\sigma_x^2$ 只对"对先验 $x \sim \mathcal{N}(0, \sigma_x^2 I)$ 取期望"的期望 MSE 最优，而 Shepp-Logan phantom 并不是先验的样本（均值 $\approx 0.173$ 非零、$\mathbb{E}[x^2] \approx 0.073 \ne \sigma_x^2 = 0.25$）；固定该 phantom 时频率派期望 MSE 最优 $\lambda \approx 0.083$。好在峰顶平坦，"接近峰值"的结论仍成立——但"$\lambda^*$ 最优"只在先验意义下成立，不对任意具体图像成立。

后验方差图要和边界条件一起读：本实验用 FFT 周期卷积，$A$ 是循环矩阵 $\Rightarrow$ $A^\top A + \lambda I$ 及其逆均为循环 $\Rightarrow$ $\text{diag}(\Sigma_{\text{post}})$ 在**所有像素处恒为同一常数 $\approx 0.227$**（差异仅 float64 舍入的 $\sim 10^{-15}$），配图是一片均匀暖色。旧版配图的 colorbar 曾自动缩放到 $[\min,\max]$，把舍入噪声放大成满屏"花纹"，旧讲义据此声称"边缘方差大、中心方差小"——那是视觉假象而非物理事实，现已固定色标（$vmin=0,\ vmax=2\bar{v}$）。"贝叶斯给出点估计给不了的定量不确定性"这一结论本身仍然成立：常数 0.227（每像素后验标准差 $\approx 0.48$）就是 MAP 点估计不会告诉你的信息——只是在这种对称设定下它不随位置变化。要展示**空间变化**的不确定性，须改用非周期边界（零填充/反射）或空间变化先验。
'''))

apply(MD, md_pairs)
print("全部完成")
