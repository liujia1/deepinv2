"""核验 D2(后验||先验) 解析式：小维数下与经验相对方差对比
相对方差定义: Var_prior(w)/E[w]^2, w = p(y|x)/p(y)（归一化权重, E[w]=1）
理论: E[w^2] = exp(D2), 故相对方差 = exp(D2) - 1
"""
import numpy as np


def analytic_D2(Am, yy, sig, sig_x):
    d = Am.shape[0]
    post_prec = np.eye(d) / sig_x ** 2 + Am.T @ Am / sig ** 2
    post_cov = np.linalg.inv(post_prec)
    m = post_cov @ (Am.T @ yy / sig ** 2)
    P = post_prec - np.eye(d) / sig_x ** 2
    _, logdetS = np.linalg.slogdet(post_cov)
    _, logdetP = np.linalg.slogdet(P)
    Pinv = np.linalg.inv(P)
    quad = m @ post_prec @ Pinv @ post_prec @ m - m @ post_prec @ m
    return (-d * np.log(2 * np.pi) - logdetS + 0.5 * d * np.log(2 * np.pi * sig_x ** 2)
            + 0.5 * d * np.log(np.pi) - 0.5 * logdetP + quad)


def analytic_log_py(Am, yy, sig, sig_x):
    d = yy.size
    cov = sig_x ** 2 * (Am @ Am.T) + sig ** 2 * np.eye(d)
    _, ld = np.linalg.slogdet(cov)
    return -0.5 * (d * np.log(2 * np.pi) + ld + yy @ np.linalg.solve(cov, yy))


for (d, sig) in [(1, 0.5), (2, 0.5), (3, 0.3), (5, 0.5)]:
    rng = np.random.default_rng(3)
    xt = rng.standard_normal(d)
    Am = rng.standard_normal((d, d)) / np.sqrt(d)
    yy = Am @ xt + sig * rng.standard_normal(d)
    logpy = analytic_log_py(Am, yy, sig, 1.0)
    M = 2_000_000
    Xs = rng.standard_normal((M, d))
    ll = (-0.5 * np.sum((yy[None, :] - Xs @ Am.T) ** 2, axis=1) / sig ** 2
          - 0.5 * d * np.log(2 * np.pi * sig ** 2))
    w = np.exp(ll - logpy)          # 归一化权重, E[w] = 1
    emp_relvar = w.var(ddof=1) / (w.mean() ** 2)
    D2 = analytic_D2(Am, yy, sig, 1.0)
    print(f"d={d}, σ={sig}: 经验相对方差={emp_relvar:12.4f}  exp(D2)-1={np.exp(D2) - 1:12.4f}  "
          f"D2={D2:8.4f}  匹配={'OK' if abs(emp_relvar - (np.exp(D2) - 1)) < 0.05 * max(1, emp_relvar) else 'FAIL'}")

# 与 KL 的对比（说明二者不同）
for (d, sig) in [(10, 0.1), (50, 0.1)]:
    rng = np.random.default_rng(42)
    xt = rng.standard_normal(d)
    Am = rng.standard_normal((d, d)) / np.sqrt(d)
    yy = Am @ xt + sig * rng.standard_normal(d)
    post_cov = np.linalg.inv(np.eye(d) + Am.T @ Am / sig ** 2)
    m = post_cov @ (Am.T @ yy / sig ** 2)
    _, ld = np.linalg.slogdet(post_cov)
    kl = 0.5 * (np.trace(post_cov) + m @ m - d - ld)
    print(f"d={d}, σ={sig}: KL(后验||先验)={kl:8.2f}   D2(后验||先验)={analytic_D2(Am, yy, sig, 1.0):8.2f}")
