# -*- coding: utf-8 -*-
# 实验1.4-2 复核诊断脚本
import numpy as np, os
from PIL import Image

D = r"d:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.4-2"
A = np.load(os.path.join(D, 'transform_A.npy'))
maomi = np.array(Image.open(os.path.join(D, 'maomi_grayscale.png')).convert('L')) / 255.0
mohu = np.array(Image.open(os.path.join(D, 'maomi_mohu.png')).convert('L')) / 255.0
rand = np.array(Image.open(os.path.join(D, 'random_gray.png')).convert('L')) / 255.0

x_true = maomi.ravel(); y = mohu.ravel(); x_rand = rand.ravel()
print("== 基本量 ==")
print("A shape:", A.shape, " A对称?", np.allclose(A, A.T, atol=1e-10))
print("||y|| =", np.linalg.norm(y), " ||A x_true|| =", np.linalg.norm(A @ x_true))
print("A 元素范围: [{:.4e}, {:.4e}]  行和均值: {:.4f}  行和std: {:.4e}".format(
    A.min(), A.max(), A.sum(1).mean(), A.sum(1).std()))
print("A 列和均值: {:.4f}  列和std: {:.4e}".format(A.sum(0).mean(), A.sum(0).std()))

# 是否为卷积型（每行是位移核）：检查第一行与列结构
row0 = A[0]; nz0 = np.where(np.abs(row0) > 1e-12)[0]
print("第0行非零个数:", len(nz0), " 前几个索引:", nz0[:10])
row17 = A[17]; nz17 = np.where(np.abs(row17) > 1e-12)[0]
print("第17行非零个数:", len(nz17), " 索引差示例:", (nz17 - nz0)[:10] if len(nz17)==len(nz0) else "不同")

s = np.linalg.svd(A, compute_uv=False)
print("== 奇异值谱 ==")
print("s[0:6]:", np.array2string(s[:6], precision=4))
print("s[-6:]:", np.array2string(s[-6:], precision=4))
print("kappa:", s[0]/s[-1], " >1e3个数:", (s>1e-3*s[0]).sum(), " >1e-2*s0个数:", (s>0.01*s[0]).sum())
print("||A||_F^2 =", np.sum(s**2))

# 残差 d = y - A x_true
Ax = A @ x_true
d = y - Ax
print("== 观测与真值映射的残差 ==")
print("||y - A x_true|| =", np.linalg.norm(d))
# PNG 8bit 量化假设：round(Ax*255)/255 与 mohu 一致吗？
Axq = np.round(Ax.reshape(32,32)*255)
print("mohu*255 与 round(Ax*255) 的最大/平均偏差(px):", np.abs(mohu*255-Axq).max(), np.abs(mohu*255-Axq).mean())
# 若 mohu = quantize(Ax + eps)
diff_px = mohu*255 - Axq
print("残差按量化级(1 LSB)的std:", diff_px.std(), "level")

q_err = (np.round(Ax*255)/255 - Ax)
print("量化误差 ||q|| =", np.linalg.norm(q_err), "(即纯量化也会产生的误差)")

print("== random_gray 统计 ==")
print("min/max/mean/std:", x_rand.min(), x_rand.max(), x_rand.mean(), x_rand.std())
u = np.unique(np.round(rand*255).astype(int))
print("离散取值个数:", len(u), " 直方图前8:", u[:8], "末8:", u[-8:])

print("== 单次复现 ==")
e_true = np.linalg.norm(Ax - y); e_rand = np.linalg.norm(A @ x_rand - y)
print("e_true =", e_true, " e_rand(给定图) =", e_rand)

print("== 蒙特卡洛：随机噪声是否普遍能打败真图 ==")
rng = np.random.default_rng(2026)
N = 500
errs_u = []; errs_g = []
for i in range(N):
    xr = rng.uniform(0,1,size=1024)
    errs_u.append(np.linalg.norm(A@xr - y))
    xg = np.clip(rng.normal(x_rand.mean(), x_rand.std(), size=1024),0,1)
    errs_g.append(np.linalg.norm(A@xg - y))
errs_u = np.array(errs_u); errs_g = np.array(errs_g)
print("uniform[0,1]随机图: 误差中位数 {:.4f}, 5-95分位 [{:.4f},{:.4f}], 战胜真图比例 {:.1%}".format(
    np.median(errs_u), *np.percentile(errs_u,[5,95]), (errs_u<e_true).mean()))
print("gaussian随机图: 误差中位数 {:.4f}, 5-95分位 [{:.4f},{:.4f}], 战胜真图比例 {:.1%}".format(
    np.median(errs_g), *np.percentile(errs_g,[5,95]), (errs_g<e_true).mean()))
print("给定 random_gray 处于 uniform 分布的分位: {:.1%}".format((errs_u<e_rand).mean()))

# 理论期望: E||A x_r - y||^2 = ||y - A m||^2 + var * ||A||_F^2, m=mean图
m = x_rand.mean()
ybar = np.full(1024, m)  # A @ (m * 1) 
Am = A @ np.full(1024, m)
base = np.linalg.norm(Am - y)**2
var_term = (x_rand.std()**2 if False else (1/12)) * np.sum(s**2)
print("理论 E||Ax_r-y||^2 (uniform) =", base + var_term, " 开方 =", np.sqrt(base+var_term))

print("== 其它对照：常数图等 ==")
rng2 = np.random.default_rng(7)
for name, xv in [("全0.5常数图", np.full(1024,0.5)),
                 ("均值匹配常数图", np.full(1024, maomi.mean())),
                 ("真图+白噪混合 50%", 0.5*x_true+0.5*rng2.uniform(0,1,1024))]:
    e = np.linalg.norm(A@xv - y)
    print(f"{name}: ||Ax-y|| = {e:.4f}")

# 病态程度对“高频损失”的验证：A 近似模糊？看 Ax_true 与 mohu 视觉差异
print("== 似然能否区分：更大随机族中误差 < e_rand ==")
print("e_rand  percentile in uniform MC:", (errs_u < e_rand).mean())
