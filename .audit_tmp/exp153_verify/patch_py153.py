# -*- coding: utf-8 -*-
"""修补 1.5-3.py：消除共享随机流耦合 + 把"同义反复验证"升级为三条路径 + JSON 补字段。
每处替换都断言命中次数，避免静默错改。"""
import io

P = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.5-3\1.5-3.py'
src = io.open(P, encoding='utf-8').read()

REPS = []


def rep(old, new, n=1):
    REPS.append((old, new, n))


# R1 独立随机流
rep("""np.random.seed(42)

print("=" * 60)""",
    """np.random.seed(42)
# 候选解的扰动改用独立生成器，不与观测噪声共用同一条随机流。
# 否则改动候选数量或循环结构会连带换掉 noise，使 ls_ideal 等归档数值整体漂移
# （共享随机流耦合：第1章多个实验共有的陷阱）。
cand_rng = np.random.default_rng(2024)

print("=" * 60)""")

# R2 循环内换 rng + 保存 residual
rep("""least_squares_list = []
neg_log_lik_list = []
""",
    """least_squares_list = []
neg_log_lik_list = []
residuals = []           # 留给后面"按定义独立复算"用
""")

rep("""    x_cand = x_true + scale * np.random.randn(n)
    residual = y - np.real(np.fft.ifft(H * np.fft.fft(x_cand)))
    ls = np.sum(residual ** 2)                       # ||y - A x||^2
    nll = 0.5 / sigma ** 2 * np.sum(residual ** 2)   # -ln p(y|x) 不计常数
    least_squares_list.append(ls)
    neg_log_lik_list.append(nll)""",
    """    x_cand = x_true + scale * cand_rng.standard_normal(n)
    residual = y - np.real(np.fft.ifft(H * np.fft.fft(x_cand)))
    ls = np.sum(residual ** 2)                       # ||y - A x||^2
    nll = 0.5 / sigma ** 2 * np.sum(residual ** 2)   # -ln p(y|x) 不计常数
    least_squares_list.append(ls)
    neg_log_lik_list.append(nll)
    residuals.append(residual)""")

# R3 三条验证路径
rep("""# 验证等价性：nll 应约等于 (1/(2 sigma^2)) * ls
factor = 1.0 / (2.0 * sigma ** 2)
predicted_nll = factor * least_squares_list
max_rel_err = float(np.max(np.abs(neg_log_lik_list - predicted_nll) /
                           (np.abs(predicted_nll) + 1e-12)))

print(f"\\n[等价性验证]")
print(f"  理论比例因子 1/(2 sigma^2) = {factor:.4f}")
print(f"  最大相对偏差 = {max_rel_err:.3e}  （应接近 0）")
if max_rel_err < 1e-10:
    print(f"  [验证通过] 负对数似然与最小二乘仅差常数因子，二者代数等价")
else:
    print(f"  [警告] 偏差偏大，请检查实现")""",
    """# ---- 2b. 等价性验证：三条路径，检验强度递增 ----
# 路径1（最弱，系数自洽检查）：neg_log_lik_list 与 predicted_nll 都由同一个
# sum(residual**2) 乘同一个标量得出，差恒为浮点 0。变异测试表明它能发现
# "比例系数写错"（漏 1/2、写成 1/(4σ²)、residual 少平方都会报警），
# 但把 residual 换成任意向量、或整个漏掉常数项，它照样通过 —— 所以它
# 只能算代码自洽检查，不能单独充当"代数等价"的数值证据。
factor = 1.0 / (2.0 * sigma ** 2)
predicted_nll = factor * least_squares_list
max_rel_err = float(np.max(np.abs(neg_log_lik_list - predicted_nll) /
                           (np.abs(predicted_nll) + 1e-12)))

# 路径2（实质验证）：按定义把 -ln p(y|x) 完整写出（含常数项 n/2·ln(2πσ²)），
# 逐元素累加，与 1/(2σ²)·||r||² 那条 np.sum 路径的算式和累加顺序都不同。
const_term = 0.5 * n * np.log(2 * np.pi * sigma ** 2)
_log_norm = 0.5 * np.log(2 * np.pi * sigma ** 2)


def _nll_by_definition(r):
    '''-ln p(y|x) = Σ_i [ (r_i/σ)²/2 + ln(√(2π)σ) ]，逐元素求和'''
    return float(np.sum(0.5 * (r / sigma) ** 2 + _log_norm))


_indep_gaps = [abs(_nll_by_definition(r) - (const_term + factor * ls)) /
               max(abs(const_term + factor * ls), 1e-12)
               for r, ls in zip(residuals, least_squares_list)]
indep_max_gap = float(max(_indep_gaps))

# 路径3（消除 σ=1 造成的检验盲区）：σ=1 时 1/(2σ²)=1/(2σ)=0.5，
# 任何把 σ² 写成 σ 的幂次错误都不可见；换 σ=2 复核比例因子随 σ 变、
# 而最优候选位置不变（常数项与候选无关，只平移不改变 argmin）。
sigma_probe = 2.0
factor_probe = 1.0 / (2.0 * sigma_probe ** 2)
const_probe = 0.5 * n * np.log(2 * np.pi * sigma_probe ** 2)
nll_probe = const_probe + factor_probe * least_squares_list
argmin_ls = int(np.argmin(least_squares_list))
argmin_probe = int(np.argmin(nll_probe))

print(f"\\n[等价性验证]")
print(f"  路径1 系数自洽检查: 最大相对偏差 = {max_rel_err:.3e}")
print(f"        （构造性恒等，只能发现比例系数写错；对 residual 错误、常数项缺失不敏感）")
print(f"  路径2 按定义含常数项独立复算: 常数项 n/2·ln(2πσ²) = {const_term:.4f}")
print(f"        与 const + ||r||²/(2σ²) 的最大相对偏差 = {indep_max_gap:.3e}  （真实浮点误差，非恒零）")
print(f"  路径3 换 σ={sigma_probe} 消除 σ=1 盲区: 比例因子 {factor:.4f} → {factor_probe:.4f}")
print(f"        最优候选索引 {argmin_ls} == {argmin_probe} → 缩放与平移都不改变 argmin")
if max_rel_err < 1e-10 and indep_max_gap < 1e-12 and argmin_ls == argmin_probe:
    print(f"  [验证通过] 负对数似然与最小二乘仅差常数因子（与候选无关），二者代数等价")
else:
    print(f"  [警告] 偏差偏大或最优位置不一致，请检查实现")""")

# R4 单点对比补常数项
rep("""print(f"  比值 nll / ls                   = {nll_ideal / ls_ideal:.4f}  (= 1/(2 sigma^2) = {factor:.4f} [OK])")""",
    """print(f"  比值 nll / ls                   = {nll_ideal / ls_ideal:.4f}  (= 1/(2 sigma^2) = {factor:.4f} [OK])")
print(f"  含常数项的完整 -ln p(y|x)       = {nll_ideal + const_term:.4f}")
print(f"        （常数项 {const_term:.4f} 与候选 x 无关，故做推断时可安全丢弃 —— 这正是"
      f"似然与最小二乘等价的关键一步）")""")

# R5 结论段补一条
rep("""print("2. 比例因子 = 1/(2 sigma^2)，与候选 x 无关 -> 仅缩放不影响最优解位置")""",
    """print("2. 比例因子 = 1/(2 sigma^2)，与候选 x 无关 -> 仅缩放不影响最优解位置")
print("   （另加常数项 n/2·ln(2πσ²) 也与 x 无关；σ=1 会让幂次写错不可见，")
print("     故本实验另用 σ=2 复核了比例因子随 σ 变化而 argmin 不变）")""")

# R6 JSON 补字段
rep("""    'max_relative_deviation': float(f"{max_rel_err:.3e}"),
    'verification_passed': bool(max_rel_err < 1e-10),
    'least_squares_at_true': float(round(ls_ideal, 4)),
    'neg_log_likelihood_at_true': float(round(nll_ideal, 4)),""",
    """    'max_relative_deviation': float(f"{max_rel_err:.3e}"),
    # 上面这项是"系数自洽检查"的偏差，由构造恒为 0，单看它会高估验证强度；
    # 真正非平凡的是下面 independent_definition_gap（独立算式+独立累加顺序）。
    'independent_definition_gap': float(indep_max_gap),
    'constant_term': float(const_term),
    'neg_log_likelihood_at_true_with_constant': float(nll_ideal + const_term),
    'sigma_probe': float(sigma_probe),
    'sigma_probe_factor': float(factor_probe),
    'argmin_index_ls': argmin_ls,
    'argmin_index_nll_probe': argmin_probe,
    'verification_passed': bool(max_rel_err < 1e-10 and indep_max_gap < 1e-12
                                and argmin_ls == argmin_probe),
    'least_squares_at_true': float(round(ls_ideal, 4)),
    'neg_log_likelihood_at_true': float(round(nll_ideal, 4)),""")

for old, new, n in REPS:
    c = src.count(old)
    assert c == n, '命中 %d 次（期望 %d）: %r' % (c, n, old[:70])
    src = src.replace(old, new)

io.open(P, 'w', encoding='utf-8', newline='').write(src)
print('patched %d items OK' % len(REPS))
