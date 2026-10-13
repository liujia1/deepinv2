# -*- coding: utf-8 -*-
import io

path = r"d:/works/deepinv/第3章 从MAP到后验探索/3.6 收敛性分析与正则化参数选择.md"
with io.open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# (key_substring, new_line_content)  —— 整行替换，key 须唯一命中一行
repl = [
    ("100 点对数扫描",
     "- $n=50, m=40$；真解 3-稀疏（$x_5=1.0,x_{15}=0.8,x_{30}=-0.5$）；$A$ 随机归一化；$\\sigma^2=0.01$；先验 $\\mathcal{N}(0,\\alpha I)$；$\\alpha$ 400 点对数扫描并在峰值邻域做连续细分；另跑 30 个随机种子做稳健性检验。\n"),
    ("在峰值；与 PSNR 最优",
     "- $\\log p(y|\\alpha)$ 单峰，$\\hat\\alpha$ 在峰值；$\\hat\\alpha$ 与 PSNR 最优 $\\alpha$ 接近（非恒等），但两者给出的重建 PSNR 应几乎持平；多种子下经验贝叶斯稳健地逼近 oracle 的“重建性能”。\n"),
    ("后生成含 2 子图的图",
     "运行 `3.6-2.py` 后生成两张各含 2 子图的图。\n"),
    ("| 经验贝叶斯 | $",
     "| 经验贝叶斯 | $\\hat\\alpha=0.0360$（连续 $0.0356$） | 最大化 $\\log p(y\\|\\alpha)$（无需真解） |\n"),
    ("| PSNR 最优（oracle） |",
     "| PSNR 最优（oracle） | $0.0373$（连续 $0.0370$） | 最大化 PSNR（需真解） |\n"),
    ("对数边际似然最大值 7.46",
     "对数边际似然最大值 7.46。两者选参虽差约 4%，但重建 PSNR 均为 17.53 dB（差 0.0002 dB）。\n\n"
     "多种子稳健性（30 个种子）：EB 与 oracle 的 $\\alpha$ 中位数同为 0.0393，24/30 落在 2× 一致带内，但逐样本 $\\log\\alpha$ 相关性弱（corr$=-0.51$）；"
     "关键在于 EB 相对 oracle 的重建 PSNR 损失均值仅 0.13 dB（最大 1.24 dB）。\n\n"
     "![多种子稳健性](<./第三章 配套实验/实验3.6-2/步骤2_多种子稳健性.png>)\n"),
    ("**EB 与 oracle 高度接近",
     "1. **EB 逼近的是 oracle 的“性能”而非“参数”**：400 点网格下 $\\hat\\alpha=0.0360$ 与 PSNR 最优 $0.0373$ 已不再相同（连续细分 $0.0356$ vs $0.0370$，差约 4%）——早期 100 点粗网格的“完全相同”只是舍入到同一点的假象。真正稳健的结论是：尽管 $\\alpha$ 略有偏移，二者重建 PSNR 几乎持平（17.53 dB，差 0.0002 dB），因为 PSNR 峰顶是平台，$\\alpha$ 的小误差被平台吸收。多种子进一步佐证：EB 不复现逐样本最优 $\\alpha$（corr 弱），但 PSNR 损失均值仅 0.13 dB——**经验贝叶斯无需真解即可把重建送到最优平台附近**，与“预期结果：接近”一致，且不应夸大为恒等；\n"),
]

for key, new in repl:
    idxs = [i for i, ln in enumerate(lines) if key in ln]
    assert len(idxs) == 1, f"key not unique ({len(idxs)}): {key!r}"
    lines[idxs[0]] = new

with io.open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)
print("doc updated OK")
