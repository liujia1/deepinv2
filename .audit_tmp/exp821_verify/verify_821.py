# -*- coding: utf-8 -*-
"""
实验8.2-1 独立复核脚本（不引用原实验代码，全部独立实现）
1) 解析后验 / log p(x) 复核
2) 高精度数值积分(quad)给出 ELBO、KL(q||post)、两种分解各项的"标准答案"
3) gap = log p(x) - ELBO 与 KL(q||p(z|x)) 的恒等性检查
4) 步骤3：代码实际构造的 q_alpha 路径 vs 其注释声称的"真混合分布"路径，
   各自精确 ELBO 是否单调；端点是否正确
5) 蒙特卡罗估计的噪声/偏差分析：多种子、+1e-3 判定容差的脆弱性
6) 步骤2"三种算法机器精度一致"的实际浮点差，以及该结论是否空洞（同一批样本）
7) int(w*N) 采样配比偏差量化
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp

rng = np.random.default_rng(0)

# ---------------- 模型 ----------------
pw = np.array([0.3, 0.7])
pm = np.array([-2.0, 1.0])
pt = np.array([1.0, 1.0])
sig = 0.5
x = 0.5

LN2PI = np.log(2.0 * np.pi)

def ln_norm_scalar(z, m, s):
    return -0.5 * LN2PI - np.log(s) - 0.5 * ((z - m) / s) ** 2

def ln_gmm_scalar(z, w, m, s):
    terms = [np.log(wk) + ln_norm_scalar(z, mk, sk) for wk, mk, sk in zip(w, m, s)]
    out = terms[0]
    for t in terms[1:]:
        out = np.logaddexp(out, t)
    return out

def ln_gmm_vec(z, w, m, s):
    z = np.asarray(z, float)
    out = np.full_like(z, -np.inf)
    for wk, mk, sk in zip(w, m, s):
        out = np.logaddexp(out, np.log(wk) - 0.5 * LN2PI - np.log(sk) - 0.5 * ((z - mk) / sk) ** 2)
    return out

# ---------------- 解析后验 ----------------
s2 = 1.0 / (1.0 / sig**2 + 1.0 / pt**2)            # 两个分量 τ 相同
post_s = float(np.sqrt(s2)[0])
post_m = s2 * (x / sig**2 + pm / pt**2)
marg_std = np.sqrt(sig**2 + pt**2)
log_marg_comp = -0.5 * LN2PI - np.log(marg_std) - 0.5 * ((x - pm) / marg_std)**2
unnorm = pw * np.exp(log_marg_comp)
post_w = unnorm / unnorm.sum()
print("解析后验: w =", post_w, " m =", post_m, " s =", post_s)

log_px = np.log(unnorm.sum())
print("log p(x) =", repr(log_px))

# ---------------- 精确量（quad） ----------------
def Q(func, lo=-30.0, hi=30.0):
    v, e = quad(func, lo, hi, epsabs=1e-13, epsrel=1e-12, limit=500)
    return v

def exact_expectations(qw, qm, qs):
    qdens = lambda z: np.exp(ln_gmm_scalar(z, qw, qm, qs))
    Eq_logpxz = Q(lambda z: qdens(z) * ln_norm_scalar(z, x, sig))
    Eq_logpz  = Q(lambda z: qdens(z) * ln_gmm_scalar(z, pw, pm, pt))
    Eq_logq   = Q(lambda z: qdens(z) * ln_gmm_scalar(z, qw, qm, qs))
    Eq_logpost= Q(lambda z: qdens(z) * ln_gmm_scalar(z, post_w, post_m,
                         np.array([post_s, post_s])))
    elbo = Eq_logpxz + Eq_logpz - Eq_logq
    H = -Eq_logq
    kl_q_prior = Eq_logq - Eq_logpz
    kl_q_post  = Eq_logq - Eq_logpost
    # 闭式 E_q log p(x|z)（高斯混合 q，逐项高斯矩）
    Eq_logpxz_closed = float(np.sum(qw * (-0.5 * LN2PI - np.log(sig)
                               - 0.5 / sig**2 * ((x - qm)**2 + qs**2))))
    return dict(Eq_logpxz=Eq_logpxz, Eq_logpz=Eq_logpz, Eq_logq=Eq_logq,
                elbo=elbo, H=H, kl_q_prior=kl_q_prior, kl_q_post=kl_q_post,
                gap=log_px - elbo, Eq_logpxz_closed=Eq_logpxz_closed)

tests = [
    ("N(0,1)",        np.array([1.0]),       np.array([0.0]), np.array([1.0])),
    ("N(0.5,0.5)",    np.array([1.0]),       np.array([0.5]), np.array([0.5])),
    ("posterior",     post_w, post_m, np.array([post_s, post_s])),
    ("bimodal",       np.array([0.5, 0.5]),  np.array([-1.0, 1.0]), np.array([0.5, 0.5])),
    ("N(5,0.1)",      np.array([1.0]),       np.array([5.0]), np.array([0.1])),
]

print("\n===== 精确积分结果 vs 代码 MC 报告值 =====")
mc_report = {"N(0,1)": (-3.0580, 1.6090), "N(0.5,0.5)": (-1.4695, 0.0205),
             "posterior": (-1.4490, 0.0000), "bimodal": (-3.6178, 2.1688),
             "N(5,0.1)": (-50.9113, 49.4623)}
print(f"{'q':<12}{'ELBO_exact':>14}{'ELBO_MC':>11}{'diff':>9}"
      f"{'KL(q|post)':>13}{'gap_exact':>12}{'gap-KL':>10}")
exact_store = {}
for name, qw, qm, qs in tests:
    r = exact_expectations(qw, qm, qs)
    exact_store[name] = r
    mc_elbo, mc_gap = mc_report[name]
    print(f"{name:<12}{r['elbo']:>14.6f}{mc_elbo:>11.4f}{r['elbo']-mc_elbo:>9.1e}"
          f"{r['kl_q_post']:>13.6f}{r['gap']:>12.6f}{r['gap']-r['kl_q_post']:>10.1e}")

print("\n===== 步骤2 各项精确值（对照教材表） =====")
for name in ["N(0,1)", "N(0.5,0.5)", "posterior"]:
    r = exact_store[name]
    print(f"{name}: E[log joint]={r['Eq_logpxz']+r['Eq_logpz']:.4f}  H={r['H']:.4f}"
          f"  | E[log p(x|z)]={r['Eq_logpxz']:.4f}(闭式{r['Eq_logpxz_closed']:.4f})"
          f"  KL(q||p)={r['kl_q_prior']:.4f}  ELBO={r['elbo']:.6f}")

# 单高斯熵的解析值
print("H(N(0,1)) 解析 =", 0.5 * np.log(2 * np.pi * np.e), " H(N(0.5,0.5)) 解析 =",
      0.5 * np.log(2 * np.pi * np.e * 0.25))

# ---------------- 步骤3 路径 ----------------
print("\n===== 步骤3 路径核查 =====")
alphas = np.linspace(0, 1, 50)

def code_path(alpha):
    qw = np.array([(1 - alpha) * pw[0] + alpha * post_w[0],
                   (1 - alpha) * pw[1] + alpha * post_w[1]])
    qm = np.array([(1 - alpha) * pm[0] + alpha * post_m[0],
                   (1 - alpha) * pm[1] + alpha * post_m[1]])
    qs = np.sqrt(np.array([(1 - alpha) * pt[0]**2 + alpha * post_s**2,
                           (1 - alpha) * pt[1]**2 + alpha * post_s**2]))
    return qw, qm, qs

def true_mixture_path(alpha):
    # (1-alpha) p(z) + alpha p(z|x): 4 个分量
    qw = np.concatenate([(1 - alpha) * pw, alpha * post_w])
    qm = np.concatenate([(1 - alpha) * pm, alpha * post_m])
    qs = np.concatenate([pt, np.array([post_s, post_s])])  # 分量标准差不随权重缩放
    return qw, qm, qs

def elbo_exact_generic(qw, qm, qs):
    qdens = lambda z: np.exp(ln_gmm_vec(np.array([z]), qw, qm, qs)[0])
    return Q(lambda z: qdens(z) * (ln_norm_scalar(z, x, sig)
                 + ln_gmm_scalar(z, pw, pm, pt)
                 - ln_gmm_vec(np.array([z]), qw, qm, qs)[0]))

elbo_code, elbo_true = [], []
for a in alphas:
    elbo_code.append(elbo_exact_generic(*code_path(a)))
    elbo_true.append(elbo_exact_generic(*true_mixture_path(a)))
elbo_code, elbo_true = map(np.array, (elbo_code, elbo_true))

d_code = np.diff(elbo_code)
d_true = np.diff(elbo_true)
print(f"代码实际路径(插值): α=0 ELBO={elbo_code[0]:.6f}, α=1 ELBO={elbo_code[-1]:.6f}, "
      f"严格单调递增: {np.all(d_code > 0)} (最小增量 {d_code.min():.2e})")
print(f"注释声称路径(真混合): α=0 ELBO={elbo_true[0]:.6f}, α=1 ELBO={elbo_true[-1]:.6f}, "
      f"严格单调递增: {np.all(d_true > 0)} (最小增量 {d_true.min():.2e})")
# 两条路径在内部是否相同
mid_c = elbo_exact_generic(*code_path(0.5))
mid_t = elbo_exact_generic(*true_mixture_path(0.5))
print(f"α=0.5: 插值路径 ELBO={mid_c:.6f} vs 真混合 ELBO={mid_t:.6f}  "
      f"-> 注释的分布等式不成立，相差 {mid_c-mid_t:.4f}")
print(f"α=0 精确 gap = {log_px - elbo_code[0]:.4f}（教材/图：4.87, -6.32）")

# ---------------- 复刻原代码 MC ----------------
def mc_elbo(qw, qm, qs, n=100000, seed=42):
    r = np.random.RandomState(seed)
    z = []
    realized = []
    for wk, mk, sk in zip(qw, qm, qs):
        nk = int(wk * n)
        realized.append(nk)
        z.append(r.randn(nk) * sk + mk)
    z = np.concatenate(z)
    log_pxz = -0.5 * LN2PI - np.log(sig) - 0.5 * ((x - z) / sig)**2
    log_pz = ln_gmm_vec(z, pw, pm, pt)
    log_qz = ln_gmm_vec(z, qw, qm, qs)
    return np.mean(log_pxz + log_pz - log_qz), np.array(realized) / sum(realized), z

print("\n===== MC 噪声与判定容差分析（40 个种子, N=100000） =====")
for name in ["N(0,1)", "N(0.5,0.5)", "bimodal", "N(5,0.1)", "posterior"]:
    qw, qm, qs = dict(((n_, t_) for n_, t_ in
                       [("N(0,1)", tests[0][1:]), ("N(0.5,0.5)", tests[1][1:]),
                        ("posterior", (post_w, post_m, np.array([post_s, post_s]))),
                        ("bimodal", tests[3][1:]), ("N(5,0.1)", tests[4][1:])]))[name]
    vals = np.array([mc_elbo(qw, qm, qs, seed=s)[0] for s in range(40)])
    viol = np.mean(vals > log_px + 1e-3)
    print(f"{name:<11} 精确ELBO={exact_store[name]['elbo']:>10.5f}  "
          f"MC均值={vals.mean():.5f} 标准差={vals.std(ddof=1):.5f}  "
          f"范围[{vals.min():.4f},{vals.max():.4f}]  误判✗比例(+1e-3)={viol:.2%}")

# 采样配比偏差
qw = post_w
_, realized, _ = mc_elbo(qw, post_m, np.array([post_s, post_s]))
print("\n后验 q 目标权重 =", qw, " 实际采样配比 =", realized, " 差异 =", realized - qw)

# ---------------- 步骤2 浮点差 ----------------
print("\n===== 步骤2 三种写法的真实浮点差（seed=42, N=100000） =====")
for name, (_, qw, qm, qs) in zip(["N(0,1)", "N(0.5,0.5)", "posterior"], tests[:3]):
    r0 = np.random.RandomState(42)
    z = []
    for wk, mk, sk in zip(qw, qm, qs):
        z.append(r0.randn(int(wk * 100000)) * sk + mk)
    z = np.concatenate(z)
    log_pxz = -0.5 * LN2PI - np.log(sig) - 0.5 * ((x - z) / sig)**2
    log_pz = ln_gmm_vec(z, pw, pm, pt)
    log_qz = ln_gmm_vec(z, qw, qm, qs)
    d1 = np.mean(log_pxz + log_pz) - np.mean(log_qz)
    d2 = np.mean(log_pxz) - (np.mean(log_qz) - np.mean(log_pz))
    d3 = np.mean(log_pxz + log_pz - log_qz)
    print(f"{name}: |d1-d2|={abs(d1-d2):.3e}, |d1-d3|={abs(d1-d3):.3e}, "
          f"|d2-d3|={abs(d2-d3):.3e}; 估计值与精确ELBO偏差={d3-exact_store[name]['elbo']:.3e}")

# 若三种写法各自独立采样，差异量级（说明"机器精度一致"来自共用样本，而非分解本身被独立验证）
print("\n独立采样时分解一/分解二的差异（20 组）:")
qw, qm, qs = tests[1][1:]
diffs = []
for s in range(20):
    def one(seed):
        r0 = np.random.RandomState(seed)
        z = r0.randn(100000) * qs + qm
        log_pxz = -0.5 * LN2PI - np.log(sig) - 0.5 * ((x - z) / sig)**2
        log_pz = ln_gmm_vec(z, pw, pm, pt)
        log_qz = ln_gmm_vec(z, qw, qm, qs)
        return np.mean(log_pxz + log_pz) - np.mean(log_qz), \
               np.mean(log_pxz) - (np.mean(log_qz) - np.mean(log_pz))
    a, b = one(2 * s)
    c, d = one(2 * s + 1)
    diffs.append(a - d)  # 不同样本批
print(f"  独立批次差异: 均值 {np.mean(diffs):.2e}, 标准差 {np.std(diffs):.2e}, "
      f"典型量级 ~1e-2（对比同批样本的机器精度 1e-15）")
