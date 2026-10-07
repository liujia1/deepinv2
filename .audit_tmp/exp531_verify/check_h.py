import numpy as np
y = np.linspace(-3, 3, 200)
for eps in (0.5,):
    exact = -y / (1 + eps)
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        lp = lambda v: -0.5 * v ** 2 / (1 + eps)
        num = (lp(y + h) - lp(y - h)) / (2 * h)
        print(f"eps={eps} h={h:.0e} max_err={np.max(np.abs(num - exact)):.3e}")
