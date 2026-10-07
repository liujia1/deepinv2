import numpy as np

def norm_pdf(y, mu, var):
    return np.exp(-0.5 * (y - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)

eps = 0.5
mus = np.array([-3.0, 3.0])
w = np.array([0.5, 0.5])
var_p = 1.0 + eps  # 边际分量方差

y = np.linspace(-8, 8, 200001)
# 边际 p_eps 与其解析得分
comps = np.stack([norm_pdf(y, m, var_p) for m in mus])
p_eps = (w[:, None] * comps).sum(0)
dp_eps = (w[:, None] * comps * (-(y - mus[:, None]) / var_p)).sum(0)
score_exact = dp_eps / p_eps

# MMSE 去噪器：分量后验权重加权的条件后验均值 E[x|y,k] = (eps*mu_k + y)/(1+eps)
post_w = (w[:, None] * comps) / p_eps
cond_mean = (eps * mus[:, None] + y[None, :]) / (1 + eps)
D = (post_w * cond_mean).sum(0)

score_tweedie = (D - y) / eps
diff = np.abs(score_exact - score_tweedie)
print("max |score_exact - score_tweedie| =", diff.max())
print("at y=0: D(0) =", np.interp(0.0, y, D), " score_exact(0) =", np.interp(0.0, y, score_exact))
# y=0 附近 p_eps 是极小还是极大
i0 = np.argmin(np.abs(y))
print("p_eps(y=0) =", p_eps[i0], " p_eps(±1) =", p_eps[np.argmin(np.abs(y - 1))], p_eps[np.argmin(np.abs(y + 1))])
print("p_eps(±2.5) =", p_eps[np.argmin(np.abs(y - 2.5))])
# 得分在0附近的符号（左侧应为正=密度上升朝0? 极小则左侧下降方向为负梯度）
for yy in [-1.0, -0.5, 0.5, 1.0]:
    print(f"  score({yy}) = {np.interp(yy, y, score_exact): .4f}")
