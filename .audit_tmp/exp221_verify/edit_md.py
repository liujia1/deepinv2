# -*- coding: utf-8 -*-
import io, sys

MD = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.2 经典先验族.md"
with io.open(MD, "r", encoding="utf-8") as f:
    s = f.read()

edits = []

# Edit 1: 实验输入 Sobolev 参数网格
edits.append((
r"| Sobolev 参数 | $\lambda \in \{0.05, 0.1, 0.2\}$ |",
r"| Sobolev 参数 | $\lambda \in \{0.05, 0.2, 0.5, 1.0, 2.0\}$（扩展网格以覆盖 PSNR 峰值） |",
))

# Edit 2a: 预期结果 Tikhonov
edits.append((
r"- Tikhonov 去噪效果受参数选择影响显著：从数据估计的 $\sigma_x$ 更匹配数据特性",
r"- Tikhonov 两种参数选择均因零均值先验假设而表现受限：数据估计的 $\sigma_x$ 正则化偏强、PSNR 甚至低于含噪图；主观 $\sigma_x=1$ 正则化偏弱",
))

# Edit 2b: 预期结果 机器精度
edits.append((
r"- Sobolev 迭代解与频域闭式解数值差异应接近机器精度",
r"- Sobolev 迭代解与频域闭式解数值差异应达到迭代容差精度（$\sim 10^{-12}$）",
))

# Edit 3a: 步骤1 表格 Sobolev 行
edits.append((
r"""| Sobolev | $\lambda = 0.05$ | 21.51 dB | 0.470 |
| Sobolev | $\lambda = 0.10$ | 22.60 dB | 0.504 |
| Sobolev | $\lambda = 0.20$ | 24.02 dB | 0.556 |""",
r"""| Sobolev | $\lambda = 0.05$ | 21.51 dB | 0.470 |
| Sobolev | $\lambda = 0.20$ | 24.02 dB | 0.556 |
| Sobolev | $\lambda = 0.50$ | 25.37 dB | 0.646 |
| Sobolev | $\lambda = 1.00$ | 24.91 dB | 0.703 |
| Sobolev | $\lambda = 2.00$ | 23.41 dB | 0.717 |""",
))

# Edit 3b: 步骤1 叙述（问题D）
edits.append((
r"""Tikhonov 两种参数选择方式展示了先验假设对结果的影响：
- 从数据估计 $\sigma_x = 0.276$：$\lambda = 0.1317$，正则化较强
- 主观先验 $\sigma_x = 1.0$：$\lambda = 0.01$，正则化较弱（因假设值域过大）

对比 PSNR（19.17 vs 20.10 dB）说明：先验假设需与数据特性匹配，不匹配的先验会引入偏差或不足。""",
r"""Tikhonov 两种参数选择方式展示了先验假设对结果的影响：
- 从数据估计 $\sigma_x = 0.276$：$\lambda = 0.1317$，正则化较强，PSNR 掉到 19.17 dB（低于含噪 20.03）
- 主观先验 $\sigma_x = 1.0$：$\lambda = 0.01$，正则化较弱，PSNR 20.10 dB（几乎等于含噪）

关键不在 $\sigma_x$ 估得准不准，而在于**零均值先验 $x\sim\mathcal{N}(0,\sigma_x^2 I)$ 本身与数据不匹配**：camera 图均值约 0.506，把解往 0 收缩必然引入偏差，因此两种 $\lambda$ 下 Tikhonov 的 PSNR 都不升反降（仅 SSIM 略升，因平滑抑制了噪声高频）。这正说明"简单高斯先验"对非零均值图像的局限，也解释了为何需要 Sobolev 这类编码空间结构的先验。

Sobolev 一侧，PSNR 随 $\lambda$ 先升后降：$\lambda=0.5$ 处达峰值 25.37 dB，$\lambda=2.0$ 因过平滑回落到 23.41 dB——扩展后的网格真正演示出了超参数的最优点（旧网格 $\{0.05,0.1,0.2\}$ 只覆盖单调上升支，"最优 $\lambda=0.2$"实为区间边缘而非峰值）。""",
))

# Edit 4a: 步骤3 局部方差表
edits.append((
r"""| 方法 | 局部方差 |
|------|----------|
| 真实 | 0.0234 |
| Tikhonov（数据估计） | 0.0174 |
| Sobolev（$\lambda = 0.05$） | 0.0206 |
| Sobolev（$\lambda = 0.10$） | 0.0190 |
| Sobolev（$\lambda = 0.20$） | 0.0166 |""",
r"""| 方法 | 局部方差 |
|------|----------|
| 真实 | 0.0254 |
| Tikhonov（数据估计） | 0.0265 |
| Sobolev（$\lambda = 0.05$） | 0.0292 |
| Sobolev（$\lambda = 0.20$） | 0.0220 |
| Sobolev（$\lambda = 0.50$） | 0.0162 |
| Sobolev（$\lambda = 1.00$） | 0.0123 |
| Sobolev（$\lambda = 2.00$） | 0.0092 |

小 $\lambda$（0.05/0.2）局部方差**高于**真实值——噪声未充分去除；大 $\lambda$（0.5/1.0/2.0）**低于**真实值——过度平滑压掉了真实结构。注意 PSNR 最优的 $\lambda=0.5$ 对应局部方差 0.0162，已明显低于真实的 0.0254：说明"按 MSE/PSNR 选参"与"保持局部方差（纹理）"并非同一目标，这正是高斯先验过平滑的体现。""",
))

# Edit 5: 步骤4 差异表 + 叙述（问题B）
edits.append((
r"""| $\lambda$ | 最大差异 | 迭代解 PSNR | 闭式解 PSNR |
|-----------|----------|-------------|-------------|
| 0.05 | $2.69 \times 10^{-5}$ | 21.51 dB | 21.51 dB |
| 0.10 | $6.01 \times 10^{-5}$ | 22.60 dB | 22.60 dB |
| 0.20 | $1.88 \times 10^{-4}$ | 24.02 dB | 24.02 dB |

差异量级远小于像素值范围，验证了两种方法的等价性。""",
r"""| $\lambda$ | 最大差异 | 迭代次数 | 迭代解 PSNR | 闭式解 PSNR |
|-----------|----------|----------|-------------|-------------|
| 0.05 | $1.97 \times 10^{-14}$ | 25 | 21.51 dB | 21.51 dB |
| 0.20 | $1.55 \times 10^{-13}$ | 57 | 24.02 dB | 24.02 dB |
| 0.50 | $4.12 \times 10^{-13}$ | 117 | 25.37 dB | 25.37 dB |
| 1.00 | $8.28 \times 10^{-13}$ | 215 | 24.91 dB | 24.91 dB |
| 2.00 | $1.70 \times 10^{-12}$ | 406 | 23.41 dB | 23.41 dB |

差异量级为 $10^{-12}\!\sim\!10^{-14}$（受迭代容差 $10^{-13}$ 控制），远小于像素值范围，验证了两种方法在数值上的等价性。注意：早期版本以"目标函数相对变化"作停止判据，会在解尚未收敛时（仅 6–11 步）提前终止，使差异停留在 $10^{-4}$ 量级；现改用"解的逐像素最大变化 $\max|\Delta x|<$ 容差"判据，迭代解才真正逼近闭式解。""",
))

for i, (old, new) in enumerate(edits):
    cnt = s.count(old)
    if cnt != 1:
        print(f"[FAIL] edit {i+1}: found {cnt} occurrences")
        print(repr(old[:60]))
        sys.exit(1)
    s = s.replace(old, new)
    print(f"[OK] edit {i+1}")

with io.open(MD, "w", encoding="utf-8") as f:
    f.write(s)
print("ALL DONE")
