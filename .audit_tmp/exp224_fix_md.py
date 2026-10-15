# -*- coding: utf-8 -*-
import io

path = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.2 经典先验族.md"
with io.open(path, "r", encoding="utf-8") as f:
    txt = f.read()

edits = [
    # 1) 含噪 PSNR 数值修正：20.03 -> 19.97（与 JSON/图/复算一致）
    ("| 含噪图像 | — | — | — | 20.03 dB |",
     "| 含噪图像 | — | — | — | 19.97 dB |"),
    # 2) LASSO 局限性描述修正：原“大量黑色区域”与图/实测不符
    ("- **LASSO 的局限性**：直接在像素域施加 $\\ell^1$ 正则化，将大量像素值推到零，导致重建图像出现大量黑色区域。这正是 LASSO 需要在变换域（小波、DCT）中使用的原因——自然图像在变换域下才是稀疏的",
     "- **LASSO 的局限性**：在像素域直接施加 $\\ell^1$ 正则化，其软阈值在 $\\lambda=0.02$ 时仅把约 $2.7\\%$ 的暗像素推到零，对其余绝大多数像素只是施加一个不超过 $\\lambda$ 的近常量整体下移，几乎不改变噪声结构——故 PSNR（20.05 dB）与含噪图像（19.97 dB）几乎持平，去噪能力名不副实。根因是自然图像的**像素值本身并不稀疏**（大部分像素显著非零），$\\ell^1$ 找不到可牺牲的稀疏分量。这正是 LASSO 需要在变换域（小波、DCT）中使用的原因——自然图像只有在变换域下才近似稀疏"),
    # 3) LASSO 曲线下降位置描述更精确
    ("- **LASSO（绿色）**：曲线在 $\\lambda$ 较小时快速下降——$\\ell^1$ 软阈值对 $\\lambda$ 极度敏感，稍大就会将过多像素推到零",
     "- **LASSO（绿色）**：曲线在 $\\lambda \\lesssim 10^{-2}$ 前基本平坦（贴合含噪基线），超过约 $10^{-2}$ 后快速跌落至地板——$\\ell^1$ 软阈值对 $\\lambda$ 敏感，$\\lambda$ 稍大就会将过多像素推到零"),
]

for old, new in edits:
    cnt = txt.count(old)
    assert cnt == 1, f"expected 1 match, got {cnt} for: {old[:40]}..."
    txt = txt.replace(old, new)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(txt)

print("OK: 3 edits applied")
