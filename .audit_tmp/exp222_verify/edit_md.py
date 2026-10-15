# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MD = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\2.2 经典先验族.md"
with open(MD, encoding='utf-8') as f:
    lines = f.readlines()

# (1-based line number, anchor substring that must exist in old line, new content)
edits = [
    (324, "IHT 与 OMP 应恢复出恰好",
     r'- **Part B**：IHT 与 OMP 会输出恰好 $K=10$ 个非零分量（这是硬阈值"只留 top-$K$"与贪心"选 $K$ 个原子"的构造必然，并不代表支撑集完全猜对——真实命中数可能少于 $K$）；重加权 ℓ¹ 非零个数可能少于 $K$（比固定 ℓ¹ 更贴近 ℓ⁰，代价是漏掉被噪声压弱的支撑），其相对误差在当前 seed 下可能最低，但这一"最低"未必跨 seed 普适；三者相对误差预计在 ~0.13–0.21 区间，均远优于随机猜测——验证压缩感知中"稀疏先验 + 欠定测量"的可恢复性。'),
    (385, "都恢复出恰好 10 个非零分量",
     r'- **IHT 与 OMP 输出恰好 10 个非零分量**（= 真支撑大小 $K$）——但这只是算法构造的必然结果：IHT 每步硬阈值只留 top-$K$、OMP 恰好选 $K$ 个原子，非零个数恒等于 $K$，与"是否猜对支撑"无关。真实的支撑命中仅 **7/10**（两者各含 3 个误报、并漏掉 3 个被噪声压弱的真支撑）；相对误差约 0.14~0.21，主要来自这些误报与漏检。'),
    (386, "Reweighted ℓ¹ 相对误差最低",
     r'- **Reweighted ℓ¹ 只恢复出 7 个非零且零误报**——它把被噪声压到接近零的弱系数**进一步压成精确零**，比固定 ℓ¹ 更稀疏，正是上文"重加权 ℓ¹ 比固定 ℓ¹ 更接近真 ℓ⁰ 解"论断的实证，代价是漏掉了 3 个弱支撑。这一"更稀疏、无误报"的特性在多 seed 下稳健成立；而它在本组实验里**相对误差最低（0.1336）只是当前 seed=7 的结果**，换 seed 后与 OMP 各有胜负，并非普适规律。'),
    (395, "三算法都重建出了",
     r'四张茎叶图从左到右：真实稀疏系数、IHT 恢复、重加权 ℓ¹ 恢复、OMP 恢复。在 $M\ll N$ 的欠定条件下，IHT 与 OMP 各输出 $K=10$ 个尖峰（由算法强制保留 top-$K$ 所致），但真实支撑只命中 7 个、含 3 个误报；重加权 ℓ¹ 输出更稀疏（7 个尖峰、零误报），漏掉了最弱的 3 个支撑。与 Part A 中"纯加噪、$A=I$"的情形不同，这里没有逐分量的闭式软阈值解，必须靠迭代/贪心才能在测量矩阵 $A$ 的列空间里找回真支撑——这也是压缩感知比"直接置零"更难也更有意思的地方。'),
    (403, "IHT 收敛快而直接",
     r'半对数坐标下，IHT 与重加权 ℓ¹ 的相对误差都随迭代下降并趋于平稳。需注意两条曲线的横轴计数口径不同：IHT 每个梯度步记一点（共约 200 点），重加权 ℓ¹ 每个外层迭代记一点（共 12 点，而每个外层内部还含至多 300 步 ISTA），因此**不能按横轴长度直接比较二者的收敛快慢**。就趋势而言，IHT 靠显式 ℓ 投影（硬阈值）单调逼近、曲线平滑；重加权 ℓ¹ 初期像普通 ℓ¹，随权重更新逐步"收紧"到更稀疏的解。两条曲线共同说明：ℓ¹ 不是终点，而是逼近 ℓ⁰ 的一个可优化的起点。'),
]

for ln, anchor, new in edits:
    old = lines[ln-1]
    assert anchor in old, f"line {ln} anchor missing: {anchor}"
    assert not old.endswith('\n') or True
    lines[ln-1] = new + '\n'
    print(f"[line {ln}] OK  (anchor found & replaced)")

with open(MD, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("WRITTEN")
