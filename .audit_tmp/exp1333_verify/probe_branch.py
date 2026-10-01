# -*- coding: utf-8 -*-
"""探针：验证13.3-3中DDRM频谱域条件化的分支是否退化
1) noise_level = sigma_t*sigma_obs/S_k 在整个t范围内是否会进入"先验主导(>2.0)"或"中间区(0.5~2.0)"
2) 若恒为"强约束"，则DDRM重建 = A伪逆(y_obs)，与x_t/t/先验完全无关
"""
import numpy as np

T = 200
beta_min, beta_max = 1e-4, 0.02
betas = np.linspace(beta_min, beta_max, T)
alpha_bars = np.cumprod(1.0 - betas)

A = np.array([[2.0, 0.1], [1.5, 0.5]])
U, S, Vt = np.linalg.svd(A)
sigma_obs = 0.1

print("奇异值 S =", np.round(S, 4))
print("alpha_bars[-1] =", alpha_bars[-1])

# 遍历所有时间步，记录每个频谱分量落入哪个分支
branch_count = {"strong": 0, "mid": 0, "weak": 0}
max_noise_level = 0.0
for t in range(T):
    sigma_t = np.sqrt(1.0 - alpha_bars[t])
    for k in range(len(S)):
        nl = sigma_t * sigma_obs / S[k]
        max_noise_level = max(max_noise_level, nl)
        if nl < 0.5:
            branch_count["strong"] += 1
        elif nl > 2.0:
            branch_count["weak"] += 1
        else:
            branch_count["mid"] += 1

print("全时间步×全频谱分量的分支命中统计:", branch_count)
print("noise_level 最大值 =", round(max_noise_level, 4), "(>0.5才会离开强约束分支)")

# 验证退化：DDRM重建是否恒等于伪逆解
np.random.seed(0)
max_diff = 0.0
for _ in range(200):
    x0 = np.random.normal(1.0, 1.0)
    x2d = np.array([x0, x0 * 0.5])
    y = A @ x2d + np.random.randn(2) * sigma_obs
    t = np.random.randint(0, T)
    x_t, _ = np.random.randn(2), None  # x_t任意甚至传nan，验证是否影响输出
    # 复制ddrm逻辑的强约束分支（若分支恒强，则输出=Ainv@y）
    sigma_t = np.sqrt(1.0 - alpha_bars[t])
    y_freq = U.T @ y
    x_freq_cond = np.array([y_freq[k] / S[k] for k in range(len(S))])
    x_recon = Vt.T @ x_freq_cond
    pinv = np.linalg.solve(A, y)
    max_diff = max(max_diff, np.linalg.norm(x_recon - pinv))
print("DDRM输出与A^-1 y的最大差异 =", max_diff, "(≈0 即证明退化为纯伪逆，与先验/x_t无关)")

# 理论误差下限：E||A^-1 n||^2 = sigma^2 * tr((A^T A)^-1)
err2 = sigma_obs**2 * np.trace(np.linalg.inv(A.T @ A))
print("伪逆方案的理论RMSE =", round(np.sqrt(err2), 4), "(与讲义DDRM误差0.22~0.27量级对照)")
