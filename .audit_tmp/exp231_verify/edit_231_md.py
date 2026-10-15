# -*- coding: utf-8 -*-
"""修正讲义 2.3 中实验2.3-1 的 Tikhonov λ 口径与数值（23.68→23.42, λ=0.0299→0.0597, σ_x²口径澄清）"""
import io, sys
p = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.3 先验的质量：MMSE vs MAP估计器.md"
s = io.open(p, encoding="utf-8").read()

pairs = [
    ("其中 $\\sigma_x^2$ 从真实信号方差估计（Oracle 估计，仅用于展示）。",
     "其中 $\\sigma_x^2$ 为**每坐标**信号方差，从真实信号估计（Oracle 估计，仅用于展示）。"
     "Tikhonov 对每个坐标独立收缩 $y_j/(1+\\lambda)$，故 $\\sigma_x^2$ 应取单坐标方差"
     "$\\mathbb{E}[x_j^2]$，而非两坐标之和 $\\mathbb{E}\\|x\\|_2^2$。"),
    ("| 信号方差 | $\\sigma_x^2 = 0.3348$（从真实分布计算） |",
     "| 信号方差 | 每坐标 $\\sigma_x^2 = 0.1674$（两坐标之和 $\\mathbb{E}\\|x\\|_2^2 = 0.3348$） |"),
    ("| Tikhonov 参数 | $\\lambda = \\sigma^2/\\sigma_x^2 = 0.0299$（Oracle 估计） |",
     "| Tikhonov 参数 | $\\lambda = \\sigma^2/\\sigma_x^2 = 0.0597$（Oracle 估计） |"),
    ("| Tikhonov（高斯先验） | $x \\sim \\mathcal{N}(0, \\sigma_x^2 I)$ | 23.68% |",
     "| Tikhonov（高斯先验） | $x \\sim \\mathcal{N}(0, \\sigma_x^2 I)$ | 23.42% |"),
    ("**LASSO 去噪器**（21.60%）优于 Tikhonov（23.68%）",
     "**LASSO 去噪器**（21.60%）优于 Tikhonov（23.42%）"),
    ("**Tikhonov 去噪器**（23.68%）仅比恒等映射（24.32%）略有改善",
     "**Tikhonov 去噪器**（23.42%）仅比恒等映射（24.32%）略有改善"),
]

for old, new in pairs:
    cnt = s.count(old)
    if cnt != 1:
        print(f"[FAIL] 命中 {cnt} 次: {old[:50]}...")
        sys.exit(1)
    s = s.replace(old, new)

io.open(p, "w", encoding="utf-8", newline="").write(s)
# 双向核查：不应再有旧数值；新数值应各出现预期次数
for bad in ["23.68", "0.0299", "0.3348$（从真实分布计算）"]:
    print(f"[check] 残留 '{bad}':", s.count(bad))
for good in ["23.42", "0.0597", "0.1674"]:
    print(f"[check] 新值 '{good}' 出现:", s.count(good))
# LaTeX 完整性抽查
import re
print("[check] sigma_x^2 命令数:", len(re.findall(r"\$\\sigma_x\^2\$", s)))
