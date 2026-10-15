# -*- coding: utf-8 -*-
import io, sys
MD = r"d:/works/deepinv/第2章 先验：贝叶斯推断的灵魂/2.1 先验的数学角色：正则化的概率诠释.md"

def load(p):
    with io.open(p, 'r', encoding='utf-8') as f: return f.read()
def save(p, s):
    with io.open(p, 'w', encoding='utf-8') as f: f.write(s)
def rep(text, old, new, tag):
    c = text.count(old)
    if c != 1:
        print(f"[FAIL] {tag}: found {c}"); print("  old:", repr(old[:70])); sys.exit(1)
    print(f"[OK] {tag}"); return text.replace(old, new)

md = load(MD)

# 1) 实验目的 2：加"适用边界"
md = rep(md,
    r"2. **验证 $\lambda$ 的概率诠释**：$\lambda = \sigma^2/\sigma_x^2$ 在模糊逆问题和纯去噪中同样适用",
    r"2. **验证 $\lambda$ 的概率诠释及其适用边界**：$\lambda = \sigma^2/\sigma_x^2$ 仅在先验假设成立时给出最优权衡（合成对照），先验误设时会明显偏离（真实图像）",
    "实验目的2")

# 2) 场景2 bullet2 尾部：加 §六 指引 + 新增合成对照 bullet3（含真实换行）
old_b2 = r"故贝叶斯 $\lambda$ 明显偏离经验最优——这恰好印证：纯标量 Tikhonov 去噪能力有限，需要 TV / 深度学习等能刻画空间相关性的更强先验"
new_b2 = (old_b2 + r"（这正是 §六「先验选择的核心问题」的入口）" + "\n"
    + r"- **合成对照（先验假设成立时）**：改用与先验同分布的 iid 高斯“图像” $x\sim\mathcal{N}(0,\sigma_x^2 I)$ 重复扫描，贝叶斯 $\lambda=0.1262$ 与经验最优 $\lambda=0.1269$ 几乎重合，去噪 PSNR（27.54 dB）也优于含噪基线（27.02 dB）——可见 $\lambda=\sigma^2/\sigma_x^2$ 的贝叶斯诠释在其假设成立时精确兑现；真实图像的偏离不是理论之过，而是先验误设之过（见步骤3 左图：品红曲线峰值正落在贝叶斯线上，蓝色曲线峰值偏左、贝叶斯点低于红色基线）")
md = rep(md, old_b2, new_b2, "场景2合成对照bullet")

# 3) 正文 line 238：强断言 -> 条件式 + 呼应实验
md = rep(md,
    r"只要你估得出噪声水平（通常可行），又指定了先验强度（需要领域知识），$\lambda$ 就不再是调参问题，而是概率推论的结果。",
    r"只要你估得出噪声水平（通常可行），又指定了先验强度（需要领域知识），$\lambda$ 就**在你所指定的那组先验假设下**不再是漫无目的的调参，而是概率推论的结果。但这份“确定性”是有前提的：$\lambda=\sigma^2/\sigma_x^2$ 只在先验真的成立（如 $x$ 确为零均值、空间白的高斯）时才最优；先验一旦与真实数据不符，这个“推论出的” $\lambda$ 也会失准——实验 2.1-1 场景2 活生生演示了这一点（真实图像上贝叶斯 $\lambda$ 反而劣于含噪基线，换成符合假设的 iid 高斯数据时它又精确回到最优）。",
    "line238条件化")

save(MD, md)
print("\n讲义编辑完成。")
