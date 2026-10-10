"""1D 情形：用数值积分核验 D2 解析式（区分"公式错"与"经验方差估计不准"）"""
import numpy as np
from scipy.integrate import quad

sig_x = 1.0
for (a, sig, y) in [(1.0, 0.5, 0.3), (2.0, 0.5, 0.3), (1.0, 0.2, 0.5)]:
    # 后验 q(x) ∝ N(y; a x, sig^2) N(x; 0, sig_x^2)
    prec = 1 / sig_x ** 2 + a ** 2 / sig ** 2
    S = 1 / prec
    m = a * y / sig ** 2 * S
    # 证据 p(y) = N(y; 0, sig_x^2 a^2 + sig^2)
    py = np.exp(-0.5 * y ** 2 / (sig_x ** 2 * a ** 2 + sig ** 2)) / np.sqrt(2 * np.pi * (sig_x ** 2 * a ** 2 + sig ** 2))

    def q(x):
        return np.exp(-0.5 * (x - m) ** 2 / S) / np.sqrt(2 * np.pi * S)

    def p(x):
        return np.exp(-0.5 * x ** 2 / sig_x ** 2) / np.sqrt(2 * np.pi * sig_x ** 2)

    Ew2, _ = quad(lambda x: q(x) ** 2 / p(x), -20, 20, limit=400)
    # 解析式
    P = prec - 1 / (2 * sig_x ** 2)   # q^2 的精度是 1/S 的两倍, 而 1/p 只贡献 1/(2σx²)
    quad_term = m * prec * (1 / P) * prec * m - m * prec * m
    D2_formula = (-np.log(2 * np.pi) - np.log(S) + 0.5 * np.log(2 * np.pi * sig_x ** 2)
                  + 0.5 * np.log(np.pi) - 0.5 * np.log(P) + quad_term)
    print(f"a={a}, sig={sig}, y={y}: 数值积分 E[w^2]={Ew2:12.5f} -> log={np.log(Ew2):8.4f} | "
          f"解析公式 D2={D2_formula:8.4f} | 差={abs(np.log(Ew2) - D2_formula):.2e}")
