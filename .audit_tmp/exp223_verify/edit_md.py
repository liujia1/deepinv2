# -*- coding: utf-8 -*-
import io, sys

path = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.2 经典先验族.md"
with io.open(path, "r", encoding="utf-8") as f:
    txt = f.read()

pairs = []

# 1. Sobolev 求解方法（实验场景）
pairs.append((
r"""**Sobolev 平滑先验**：假设 $\nabla x \sim \mathcal{N}(0, \sigma_\nabla^2 I)$，正则项为 $\|\nabla x\|_2^2$。MAP 估计通过梯度下降迭代求解。""",
r"""**Sobolev 平滑先验**：假设 $\nabla x \sim \mathcal{N}(0, \sigma_\nabla^2 I)$，正则项为 $\|\nabla x\|_2^2$。MAP 估计在周期边界下可通过 FFT 频域闭式解直接计算。"""
))

# 2. TV 求解方法
pairs.append((
r"""**TV 全变差先验**：假设 $\nabla x \sim \text{Laplace}(0, bI)$，正则项为 $\|\nabla x\|_1$。MAP 估计通过近端梯度下降（PGD）求解，其中近端算子为软阈值操作。""",
r"""**TV 全变差先验**：假设 $\nabla x \sim \text{Laplace}(0, bI)$，正则项为 $\|\nabla x\|_1$。MAP 估计（ROF 模型）通过 Chambolle 原始-对偶投影算法求解。"""
))

# 3. 实验输入表：Sobolev / TV 参数 + 求解方法
pairs.append((
r"""| Sobolev 参数 | $\lambda = 0.15$，控制梯度平滑强度 |
| TV 参数 | $\alpha = 0.15$，控制全变差正则化强度 |
| 求解方法 | Sobolev：梯度下降；TV：近端梯度下降（PGD） |""",
r"""| Sobolev 参数 | $\lambda = 2.5$，调至其 SSIM 最优的去噪工作点 |
| TV 参数 | $\alpha = 0.15$，调至保边与去噪平衡的工作点 |
| 求解方法 | Sobolev：FFT 频域闭式解；TV：Chambolle 原始-对偶算法 |"""
))

# 4. 参数选择说明
pairs.append((
r"""> **参数选择说明**：Sobolev 的 $\lambda$ 与 TV 的 $\alpha$ 量纲不同——$\lambda$ 惩罚 $\|\nabla x\|_2^2$（二次惩罚），$\alpha$ 惩罚 $\|\nabla x\|_1$（线性惩罚）。本实验取相近数值以展示定性差异，非严格最优对比。""",
r"""> **参数选择说明**：Sobolev 的 $\lambda$ 与 TV 的 $\alpha$ 量纲不同——$\lambda$ 惩罚 $\|\nabla x\|_2^2$（二次惩罚），$\alpha$ 惩罚 $\|\nabla x\|_1$（线性惩罚），二者数值不可直接比较。本实验将两者各自调至接近最优的去噪工作点（Sobolev $\lambda=2.5$、TV $\alpha=0.15$），以便在公平的噪声抑制水平下对比其边缘保持行为（Sobolev 磨边 vs TV 保边）。"""
))

# 5. 质量指标表 Sobolev 行
pairs.append((
r"""| Sobolev（smoothness prior） | 23.64 dB | 0.459 |""",
r"""| Sobolev（smoothness prior） | 23.90 dB | 0.695 |"""
))

# 6. 提升叙述
pairs.append((
r"""TV 先验在整体质量指标上显著优于 Sobolev 先验：
- PSNR 提升 3.35 dB（23.64 → 26.99 dB）
- SSIM 提升 0.303（0.459 → 0.762）

这主要归功于 TV 的保边能力：边缘锐利度对 SSIM 贡献显著。Sobolev 的 $\ell^2$ 梯度惩罚对大梯度（边缘处）施加二次代价，导致边缘过度平滑；而 TV 的 $\ell^1$ 惩罚只产生线性代价，边缘得以保留。""",
r"""TV 先验在两项质量指标上均优于 Sobolev 先验：
- PSNR 高 3.09 dB（23.90 → 26.99 dB）
- SSIM 高 0.067（0.695 → 0.762）

这主要归功于 TV 的保边能力。Sobolev 的 $\ell^2$ 梯度惩罚对大梯度（边缘处）施加二次代价，为压制噪声不得不整体平滑，结果边缘被磨糊（见图1第三面板：轮廓发虚、细节丢失）；而 TV 的 $\ell^1$ 惩罚只产生线性代价，边缘得以锐利保留。因此在两者都充分去噪的前提下，TV 指标仍更高。""",
))

# 7. 步骤2 剖面描述
pairs.append((
r"""左图展示了图像中心行的剖面对比，可以清晰看到：
- **Sobolev 剖面**：平滑连续，但边缘处过渡缓慢（过度平滑）
- **TV 剖面**：边缘处跃变陡峭（保边），但在渐变区域出现明显的阶梯状跳跃""",
r"""左图展示了自动检测行（第 50 行）的剖面对比，可以清晰看到：
- **Sobolev 剖面**：整体平滑，但在边缘处过渡缓慢、峰值被明显压低（过度平滑磨边）
- **TV 剖面**：边缘处跃变陡峭、峰值保留更好（保边），但在缓变区段以"平坦段＋跳变"逼近，呈阶梯状""",
))

# 8. 中图 histogram 描述 + Sobolev 28.1% 捏造值
pairs.append((
r"""- **原始图像**：梯度分布连续，边缘处梯度平缓过渡
- **Sobolev 去噪**：梯度分布与原始相似，但整体幅值降低（过度平滑）
- **TV 去噪**：产生大量零梯度（图中 TV 零梯度比例约 37.6%，Sobolev 约 28.1%），验证了 TV 偏好分段常数函数的特性""",
r"""- **原始图像**：梯度分布以低值为主，并带有边缘产生的高梯度长尾
- **Sobolev 去噪**：梯度整体向低值收拢，但分布较 TV 更分散（以平滑方式衰减梯度）
- **TV 去噪**：产生大量近零梯度、分布高度集中在原点（图中 TV 零梯度比例约 37.6%，Sobolev 约 16.3%），验证了 TV 偏好分段常数函数的特性""",
))

miss = []
for old, new in pairs:
    cnt = txt.count(old)
    if cnt != 1:
        miss.append((cnt, old[:40]))
        continue
    txt = txt.replace(old, new)

if miss:
    print("NOT-UNIQUE / NOT-FOUND:")
    for c, s in miss:
        print(f"  count={c}  head={s!r}")
    sys.exit(1)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(txt)
print("all 8 replacements applied OK")
