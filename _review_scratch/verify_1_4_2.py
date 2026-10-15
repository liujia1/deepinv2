# -*- coding: utf-8 -*-
"""实验1.4-2 独立复核脚本（只读原始数据，不写实验目录）"""
import json
import numpy as np
from PIL import Image

EXP = r"d:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.4-2"

def load_png(name):
    return np.array(Image.open(EXP + "\\" + name).convert("L")).astype(np.float64) / 255.0

A32 = np.load(EXP + r"\transform_A.npy")
print("A dtype:", A32.dtype, "shape:", A32.shape)
A = A32.astype(np.float64)

xt = load_png("maomi_grayscale.png").ravel()
y = load_png("maomi_mohu.png").ravel()
xr = load_png("random_gray.png").ravel()

raw_xt = np.array(Image.open(EXP + r"\maomi_grayscale.png").convert("L"))
raw_y = np.array(Image.open(EXP + r"\maomi_mohu.png").convert("L"))
raw_xr = np.array(Image.open(EXP + r"\random_gray.png").convert("L"))

with open(EXP + r"\results_summary.json", encoding="utf-8") as f:
    J = json.load(f)

ok = True
def chk(name, got, exp, tol=1e-6):
    global ok
    rel = abs(got - exp) / max(abs(exp), 1e-30)
    good = rel <= tol
    ok = ok and good
    print(f"[{'PASS' if good else 'FAIL'}] {name}: 复算={got:.10g}  json={exp:.10g}  rel={rel:.2e}")

print("\n===== 1. 复算 JSON 全部数值 =====")
U, s, Vt = np.linalg.svd(A)
kappa = s[0] / s[-1]
chk("condition_number", kappa, J["condition_number"], 1e-7)
chk("sigma_max", s[0], J["max_singular_value"], 1e-7)
chk("sigma_min", s[-1], J["min_singular_value"], 1e-7)
chk("log10 kappa", np.log10(kappa), J["log10_condition_number"], 1e-9)

et = np.linalg.norm(A @ xt - y)
er = np.linalg.norm(A @ xr - y)
chk("err_true", et, J["data_fit_error_true"])
chk("err_random", er, J["data_fit_error_random"])
chk("ratio", et / er, J["error_ratio_true_vs_random"])

rng = np.random.default_rng(2026)
N = 500
Xmc = rng.uniform(0, 1, size=(1024, N))
Emc = np.linalg.norm(A @ Xmc - y[:, None], axis=0)
chk("MC median", np.median(Emc), J["mc_fresh_noise_median_error"])
chk("MC p05", np.percentile(Emc, 5), J["mc_fresh_noise_err_p05"])
chk("MC p95", np.percentile(Emc, 95), J["mc_fresh_noise_err_p95"])
chk("MC frac<true", float((Emc < et).mean()), J["mc_fresh_noise_beat_true_fraction"])
chk("MC frac<given", float((Emc < er).mean()), J["given_random_gray_percentile_in_mc"])
print("    MC min =", Emc.min(), " MC max =", Emc.max())

d = xr - xt
Ad = A @ d
chk("||d||", np.linalg.norm(d), J["diff_norm_x_random_vs_true"])
chk("||Ad||", np.linalg.norm(Ad), J["diff_norm_after_A"])
coef = Vt @ d
mask = s < 0.01 * s[0]
chk("n_small", int(mask.sum()), J["n_near_null_directions_sigma_lt_1pct"])
frac = np.sum(coef[mask] ** 2) / np.sum(coef ** 2)
chk("energy frac small", frac, J["diff_energy_frac_in_near_null"], 1e-9)

xls = np.linalg.lstsq(A, y, rcond=None)[0]
chk("ls fit", np.linalg.norm(A @ xls - y), J["lstsq_fit_error"], 1e-3)
chk("ls dev", np.linalg.norm(xls - xt), J["lstsq_deviation_from_true"], 1e-9)
chk("||xt||", np.linalg.norm(xt), J["x_true_norm"], 1e-9)

print("\n===== 2. A 的算子结构鉴定 =====")
print("||A - A^T||_F / ||A||_F =", np.linalg.norm(A - A.T) / np.linalg.norm(A))
print("行和: mean=%.4g std=%.4g min=%.4g max=%.4g" % (A.sum(1).mean(), A.sum(1).std(), A.sum(1).min(), A.sum(1).max()))
print("列和: mean=%.4g std=%.4g min=%.4g max=%.4g" % (A.sum(0).mean(), A.sum(0).std(), A.sum(0).min(), A.sum(0).max()))
print("A 元素: mean=%.3e std=%.3e min=%.3e max=%.3e" % (A.mean(), A.std(), A.min(), A.max()))
print("||A||_F =", np.linalg.norm(A, "fro"), " sum(s^2) =", np.sqrt((s ** 2).sum()))
print("A 中非零(|x|>1e-6)元素比例:", (np.abs(A) > 1e-6).mean())
# 特征值（对称性快速判断）
ev = np.linalg.eigvalsh(A) if np.linalg.norm(A - A.T) / np.linalg.norm(A) < 1e-8 else None
if ev is not None:
    print("对称矩阵；特征值范围:", ev.min(), ev.max(), " 负特征值个数:", (ev < 0).sum())

# 可分离性 A = kron(K,K) 检验：A4d[i,j,k,l] 应 = K[i,k] K[j,l]
A4 = A.reshape(32, 32, 32, 32)
M = A4.transpose(0, 2, 1, 3).reshape(1024, 1024)  # M[(i,k),(j,l)]
ss = np.linalg.svd(M, compute_uv=False)
print("可分离秩检验（kron(K,K) => 秩1）: top5 奇异值 =", ss[:5])
print("  s1/s2 =", ss[0] / ss[1], " rank1 解释能量比例 =", ss[0] ** 2 / (ss ** 2).sum())
# 若秩1，还原 K
K = None
if ss[0] / ss[1] > 100:
    u1 = su[:, 0] * np.sqrt(ss[0])
    K = u1.reshape(32, 32)
    Ak = np.kron(K, K)
    print("  ||A - kron(K,K)||_F/||A||_F =", np.linalg.norm(A - Ak) / np.linalg.norm(A))
    # 符号/缩放二义性
    print("  K 行和 mean=%.4f std=%.4f" % (K.sum(1).mean(), K.sum(1).std()))

# 平移不变性（卷积）检验：A T_p == T_p A
def shift2(v, di, dj):
    X = v.reshape(32, 32)
    return np.roll(np.roll(X, di, 0), dj, 1).ravel()
res = []
for di, dj in [(1, 0), (0, 1), (3, 5)]:
    v = xt
    lhs = A @ shift2(v, di, dj)
    rhs = shift2(A @ v, di, dj)
    res.append(np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs))
print("平移不变性相对误差 (1,0)/(0,1)/(3,5):", res)

# 对图像作用：2D 卷积形式 K X K^T 检验
if K is not None:
    X = xt.reshape(32, 32)
    y2 = (K @ X @ K.T).ravel()
    print("  K X K^T vs A@xt 相对误差:", np.linalg.norm(y2 - A @ xt) / np.linalg.norm(A @ xt))
    print("  K 对称性 ||K-K^T||/||K|| =", np.linalg.norm(K - K.T) / np.linalg.norm(K))
    print("  K 中心行:", np.round(K[16, 12:21], 4))

print("\n===== 3. y 的生成方式 =====")
r = y - A @ xt
print("||y - A xt|| =", np.linalg.norm(r), " 逐像素 std=%.5f max=%.5f" % (r.std(), np.abs(r).max()))
print("残差均值=%.2e" % r.mean())
# 量化噪声基线
q = y - np.round(y * 255) / 255
print("y 量化残差范数 =", np.linalg.norm(q), "（纯量化理论期望 sqrt(n/12)/255 =", np.sqrt(1024 / 12) / 255, ")")
# 若不加噪声：||round(255 Axt)/255 - Axt||
Axt = A @ xt
qonly = np.round(Axt * 255) / 255 - Axt
print("假设 y=量化(Axt) 且 x=量化xt，预期残差范数 =", np.linalg.norm(qonly))
# 残差高斯性粗查：落在 ±1σ/±2σ 比例
z = r / r.std()
for k in (1, 2, 3):
    print(f"  |z|<{k}σ 比例 = {(np.abs(z) < k).mean():.3f}（高斯 {0.683 if k==1 else 0.954 if k==2 else 0.997}）")
print("y 像素取值范围:", raw_y.min(), raw_y.max(), " 唯一值数:", len(np.unique(raw_y)))

print("\n===== 4. random_gray 构造方式取证 =====")
# 4.1 三角不等式等号检验
print("|et - er| =", abs(et - er), " ||Ad|| =", np.linalg.norm(Ad),
      " 两者之差 =", np.linalg.norm(Ad) - abs(et - er))
# 4.2 Ad 是否与残差 e=y-Axt 共线
e = y - A @ xt
alpha = float(Ad @ e / (e @ e))
coll = np.linalg.norm(Ad - alpha * e) / np.linalg.norm(Ad)
print(f"Ad = α e 共线检验: α={alpha:.6f}, 相对偏离={coll:.3e}")
# 4.3 混合假设 xr_raw/255 ≈ clip((1-α)xt + α xls)
blend = (1 - alpha) * xt + alpha * xls
blend_c = np.clip(blend, 0, 1)
print("混合前越界像素比例:", (blend < 0).mean() + (blend > 1).mean(),
      " blend 范围 [%.3f, %.3f]" % (blend.min(), blend.max()))
recon = np.round(blend_c * 255) / 255
exact = (recon == xr).mean()
print("混合+裁剪+8bit量化 与 random_gray.png 完全一致像素比例:", exact)
print("  最大逐像素差异:", np.abs(recon - xr).max())
# 备选：不裁剪
recon0 = np.round(np.clip(blend, 0, 1) * 255) / 255
# 备选纯零空间构造：d 是否恰在 24 维张成空间（扣除量化）
V_small = Vt[mask]
proj_small = V_small.T @ (V_small @ d)
proj_large = d - proj_small
print("d 在小σ子空间投影范数=%.5f, 正交补范数=%.5f" % (np.linalg.norm(proj_small), np.linalg.norm(proj_large)))
print("random_gray 饱和像素(0或255)比例:", ((raw_xr == 0) | (raw_xr == 255)).mean())
print("random_gray 取值唯一数:", len(np.unique(raw_xr)), " 范围:", raw_xr.min(), raw_xr.max())
# d 与 (xls-xt) 的共线关系
h = xls - xt
beta = float(d @ h / (h @ h))
print(f"d = β(xls-xt) 检验: β={beta:.6f}, 余弦={np.linalg.norm(d @ h)/(np.linalg.norm(d)*np.linalg.norm(h)):.6f}")
resid_h = d - beta * h
print("  残差范数=%.5f（量化噪声量级约 %.4f）" % (np.linalg.norm(resid_h), np.sqrt(1024/12)/255))

print("\n===== 5. LS 误差放大的理论核对 =====")
# xls - xt = V diag(1/s) U^T e
err_spec = Vt.T @ (xls - xt)
noise_spec = U.T @ e
ratio_spec = np.divide(err_spec, 1.0 / s * noise_spec + 1e-300)
print("谱域 xls-xt == V Σ^-1 U^T e 最大相对偏差:",
      np.abs(err_spec - (1 / s) * noise_spec).max() / np.abs(err_spec).max())
amp = np.abs(err_spec) * s / max(np.linalg.norm(noise_spec), 1e-30)
print("各方向噪声投影 |u_j^T e| 前5大小方向 j:", np.argsort(np.abs(noise_spec))[-5:])
print("对应 σ:", s[np.argsort(np.abs(noise_spec))[-5:]])
print("数据噪声/信号 ||e||/||y|| =", np.linalg.norm(e) / np.linalg.norm(y))
print("LS 相对误差/数据相对误差（实际放大倍数）=",
      (np.linalg.norm(xls - xt) / np.linalg.norm(xt)) / (np.linalg.norm(e) / np.linalg.norm(y)))
print("κ 上界 =", kappa)

print("\n===== 6. '随机噪声碰巧拟合' 的概率（理论） =====")
from scipy.stats import beta as B
k = int(mask.sum())
# 均匀随机方向在 k 维子空间能量占率 ~ Beta(k/2,(n-k)/2)
F = B(k / 2, (1024 - k) / 2)
for f0 in [0.5, 0.9, 0.99, 0.999]:
    print(f"  P(F >= {f0}) = {1 - F.cdf(f0):.3e}")
# 需要 F 多高才能把 ||Ax-y|| 压到 0.045？粗略：误差² ≈ (1-F)·E||Au||²量级
print("  实测构造样本 F =", frac)

print("\nALL PASS" if ok else "\n存在 FAIL 项")
