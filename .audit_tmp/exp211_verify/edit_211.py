# -*- coding: utf-8 -*-
import io, sys

PY = r"d:/works/deepinv/第2章 先验：贝叶斯推断的灵魂/第二章 配套实验/实验2.1-1/2.1-1.py"
MD = r"d:/works/deepinv/第2章 先验：贝叶斯推断的灵魂/2.1 先验的数学角色：正则化的概率诠释.md"

def load(p):
    with io.open(p, 'r', encoding='utf-8') as f:
        return f.read()

def save(p, s):
    with io.open(p, 'w', encoding='utf-8') as f:
        f.write(s)

def rep(text, old, new, tag):
    cnt = text.count(old)
    if cnt != 1:
        print(f"[FAIL] {tag}: expected 1 occurrence, found {cnt}")
        print("  old snippet:", repr(old[:60]))
        sys.exit(1)
    print(f"[OK] {tag}")
    return text.replace(old, new)

# ---------------- 编辑 2.1-1.py ----------------
py = load(PY)

py = rep(py,
    r"lambda=0.001 (欠正则化)\nPSNR={psnr_demo[0]:.2f}dB\n模糊残留明显",
    r"lambda=0.001 (欠正则化)\nPSNR={psnr_demo[0]:.2f}dB\n噪声放大明显",
    "py-step2标题 模糊残留->噪声放大")

py = rep(py,
    r"# 注：第一行的贝叶斯 lambda (≈0.126) 与最优 lambda (≈0.026) 接近，",
    r"# 注：第一行的贝叶斯 lambda (≈0.126) 偏离最优 lambda (≈0.026) 约 5 倍，",
    "py-步骤4注释 接近->偏离")

py = rep(py,
    r'print("4. 不同 lambda 影响重建: lambda小->模糊残留(欠正则化)，lambda大->过度平滑(过正则化)")',
    r'print("4. 不同 lambda 影响重建: lambda小->噪声放大(欠正则化)，lambda大->过度平滑(过正则化)")',
    "py-结论4 模糊残留->噪声放大")

py = rep(py,
    'print("5. 贝叶斯 lambda 与 PSNR 最优 lambda 接近，验证了贝叶斯框架的正确性")',
    'print("5. 纯去噪下贝叶斯 lambda(0.1262) 偏离最优 lambda(0.0259)，其 PSNR(18.91dB) 甚至低于含噪基线(19.76dB)")' + "\n"
    + 'print("   —— 因零均值/空间白高斯先验与真实图像(均值≈0.51、强相关)不匹配，印证了需要更强先验")',
    "py-结论5 失实->如实")

# 在 results_summary 中补上含噪基线 PSNR，便于审计对比
py = rep(py,
    "    'psnr_bayes_lambda_dB': float(round(psnr_bayes, 2)),",
    "    'psnr_bayes_lambda_dB': float(round(psnr_bayes, 2))," + "\n"
    + "    'psnr_denoise_noisy_input_dB': float(round(psnr_noisy, 2)),",
    "py-json 增加含噪基线字段")

save(PY, py)

# ---------------- 编辑讲义 md ----------------
md = load(MD)

md = rep(md,
    r"- 贝叶斯 $\lambda$ 对应的 PSNR 接近扫描最优值",
    r"- 贝叶斯 $\lambda = 0.1262$ 对应的 PSNR 仅 18.91 dB，**反而低于含噪观测基线（19.76 dB）**；扫描最优 $\lambda = 0.0259$ 也只有 19.88 dB，比基线高约 0.12 dB",
    "md-场景2 bullet1")

md = rep(md,
    r"- 验证了 $\lambda = \sigma^2/\sigma_x^2$ 的概率诠释在去噪问题中同样成立",
    r"- 这说明 $\lambda = \sigma^2/\sigma_x^2$ 只在“$x$ 服从零均值、空间白高斯先验”的理想假设下才最优。真实图像均值非零（$\approx 0.51$）且像素强相关，向 0 收缩引入的偏差压过降噪收益，故贝叶斯 $\lambda$ 明显偏离经验最优——这恰好印证：纯标量 Tikhonov 去噪能力有限，需要 TV / 深度学习等能刻画空间相关性的更强先验",
    "md-场景2 bullet2")

md = rep(md,
    r"| 0.001 | 欠正则化，模糊残留明显 | 7.11 dB |",
    r"| 0.001 | 欠正则化，噪声放大明显 | 7.11 dB |",
    "md-不同lambda表 模糊残留->噪声放大")

md = rep(md,
    r"$\lambda$ 小则更多依赖数据（模糊残留），$\lambda$ 大则更多依赖先验（过度平滑）",
    r"$\lambda$ 小则更多依赖数据（噪声放大），$\lambda$ 大则更多依赖先验（过度平滑）",
    "md-权衡说明 模糊残留->噪声放大")

md = rep(md,
    r"- 不同 $\lambda$ 下重建效果有明显差异：欠正则化模糊残留，过正则化过度平滑",
    r"- 不同 $\lambda$ 下重建效果有明显差异：欠正则化噪声放大，过正则化过度平滑",
    "md-预期结果 模糊残留->噪声放大")

save(MD, md)
print("\n全部替换完成。")
