# -*- coding: utf-8 -*-
"""修正 6.2 讲义：实验路径链接、步骤1-2/3 表述、步骤4 等价性证据链实测描述"""
import io, sys

PATH = r"d:\works\deepinv\第6章 得分匹配：从去噪中学习得分\6.2 显式得分匹配（ESM）与隐式得分匹配（ISM）.md"

with io.open(PATH, 'r', encoding='utf-8') as f:
    text = f.read()

reps = [
    # 1) 实验目录路径修正（原链接指向不存在的 6.2-1/）
    (r"实验 `6.2-1.py`（`第六章 配套实现/6.2-1/`）",
     r"实验 `6.2-1.py`（`第六章 配套实现/实验6.2-1/`）"),
    (r"**结果**（看图 `./第六章配套实现/6.2-1/步骤1-2_ESM与ISM损失对比.png` 等）：",
     r"**结果**（看图 `./第六章 配套实现/实验6.2-1/步骤1-2_ESM与ISM损失对比.png` 等，数值存档见 `results_summary.json`）："),
    # 2) 步骤1-2：右图"近似水平线"的精确表述（残余斜率=MC误差，总体层面严格常数）
    (r"右图显示 ESM 损失减 ISM 损失近似一条水平线——坐实两者之差 $\mathcal{J}_{\text{ESM}}-\mathcal{J}_{\text{ISM}}=\frac12\mathbb{E}[\|\nabla\log p\|^2]$ 是与 $\theta$ 无关的常数。",
     r"右图是两者之差：全曲线相对起伏仅约 1.2%（均值 0.3634）。纵轴放大后肉眼可见一条轻微下倾的线，那不是理论错了，而是有限样本（$N=5\times10^4$）的蒙特卡洛误差——分部积分恒等式 $\mathbb{E}[x\,\nabla\log p]=-1$ 在样本层面只能近似成立（实测 $\mathbb{E}_{\text{emp}}[x\,\nabla\log p]\approx-0.998$，残余斜率恰为 $-(-0.998+1)\approx-2.2\times10^{-3}$）。在总体层面做数值积分，该差在 $\theta\in\{-1.5,\dots,0.5\}$ 上严格等于常数 $\frac12\mathbb{E}[\|\nabla\log p\|^2]\approx0.3628$——坐实 $\mathcal{J}_{\text{ESM}}-\mathcal{J}_{\text{ISM}}$ 与 $\theta$ 无关。"),
    # 3) 步骤3："高度一致"补充尾部偏差与跨步骤自洽
    (r"两者学到的得分高度一致。",
     r"两者学到的得分在主体区间（$|x|\lesssim4$）高度一致；ISM 在尾部偏差更大（验证损失 $0.0068$ vs ESM $\approx0.000006$）。值得注意的是，ISM 训练损失收敛到 $\approx-0.366$，恰为 ESM 最小值减去常数 $0.363$——与步骤1–2 的常数差关系跨步骤自洽。"),
    # 4) 步骤4：补等价性证据链实测结果（原图含右半图但讲义未描述；标题旧说法与数据矛盾）
    (r"  图解为什么 ISM 在图像上走不通。",
     r"""  图解为什么 ISM 在图像上走不通。

  同一图的右半是**等价性证据链**：取浅（1→8→1）、中（1→32→1）、深（1→64→64→1）三种容量分别按 ESM、ISM 训练 100 轮，ESM–ISM 预测的 RMSE 差异实测约为 $0.31\to0.17\to0.33$，**并非随容量单调减小**（Deep 反而最大，且 ISM 自身的得分 MSE 也非单调：$0.10\to0.02\to0.065$）。这不是矛盾：等价性定理只保证两个损失泛函有相同的最优解（差常数），并不保证两次独立的有限步训练收敛到同一参数；ISM 目标非强制（non-coercive），优化对容量与初始化更敏感。Medium 恰好最接近最优解、Deep 反而偏离更远，正是"理论等价"与"实践收敛"之别的活教材。"""),
]

for i, (old, new) in enumerate(reps, 1):
    n = text.count(old)
    if n != 1:
        print(f"[FAIL] 替换{i}: 匹配次数={n}（应为1）\n  旧文: {old[:60]}...")
        sys.exit(1)
    text = text.replace(old, new)
    print(f"[OK] 替换{i} 完成")

with io.open(PATH, 'w', encoding='utf-8') as f:
    f.write(text)
print("讲义已写入。")
