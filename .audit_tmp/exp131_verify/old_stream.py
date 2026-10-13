import numpy as np
# 复现 c411ed5 之前（kappas 不含 1e7）的随机数流，检验 md 表格后三行是否为旧运行残留
np.random.seed(42)
n = 100
x_true = np.random.randn(n)
noise_level = 1e-8
kappas_old = [10.0**k for k in [0, 1, 2, 3, 4, 6, 8, 10, 12]]   # 旧列表（无 1e7）
sv = {k: np.logspace(0, -np.log10(k), n) for k in kappas_old}
print('旧 kappas 列表（无 1e7）下的主循环结果：')
for kappa in kappas_old:
    s = sv[kappa]
    y = s * x_true
    yn = np.linalg.norm(y)
    y_noisy = y + noise_level * np.random.randn(n)
    xr = y_noisy / s
    rel = np.linalg.norm(xr - x_true) / np.linalg.norm(x_true)
    print(f'  kappa={kappa:10.0e}  rel_err={rel:.4e}   theory={kappa*np.linalg.norm(y_noisy-y)/yn:.4e}')
print('\n讲义表格后三行: 1e8 -> 1.06e-1, 1e10 -> 1.56e1, 1e12 -> 1.30e3')
