# -*- coding: utf-8 -*-
"""1) 复制多seed证据JSON到实验目录 2) 讲义步骤4段落升级为多seed稳健性表述"""
import io, sys, shutil, os

EXP = r"d:\works\deepinv\第6章 得分匹配：从去噪中学习得分\第六章 配套实现\实验6.2-1"
src = r"d:\works\deepinv\.audit_tmp\exp621_seedvar.json"
dst = os.path.join(EXP, 'step4_multiseed_seedvar.json')
shutil.copyfile(src, dst)
print("[OK] 已复制 step4_multiseed_seedvar.json 到实验目录")

PATH = r"d:\works\deepinv\第6章 得分匹配：从去噪中学习得分\6.2 显式得分匹配（ESM）与隐式得分匹配（ISM）.md"
with io.open(PATH, 'r', encoding='utf-8') as f:
    text = f.read()

old = r"""（Deep 反而最大，且 ISM 自身的得分 MSE 也非单调：$0.10\to0.02\to0.065$）。这不是矛盾：等价性定理只保证两个损失泛函有相同的最优解（差常数），并不保证两次独立的有限步训练收敛到同一参数；ISM 目标非强制（non-coercive），优化对容量与初始化更敏感。Medium 恰好最接近最优解、Deep 反而偏离更远，正是"理论等价"与"实践收敛"之别的活教材。"""
new = r"""（Deep 反而最大，且 ISM 自身的得分 MSE 也非单调：$0.10\to0.02\to0.065$）。为排除"单 seed 巧合"，我们用 seed $\{42,0,1,2,3\}$ 重跑了同一协议（完整数值见本目录 `step4_multiseed_seedvar.json`）：**diff 随容量单调递减在 5 个 seed 中 0 次成立**；Deep 的 diff 在全部 5 个 seed 中都最大（$0.31\sim0.39$），5-seed 均值下 ISM 得分 MSE 也最差（$0.071\pm0.012$，Medium 最优 $0.023\pm0.004$）；而 ESM 的损失在全部 5 个 seed 中都随容量单调下降（均值 $0.007\to0.001\to0.0005$）——不稳定的来源是 ISM 的优化本身，而非任务或数据。这不是矛盾：等价性定理只保证两个损失泛函有相同的最优解（差常数），并不保证两次独立的有限步训练收敛到同一参数；ISM 目标非强制（non-coercive），优化对容量与初始化更敏感。Medium 恰好最接近最优解、Deep 反而偏离更远，正是"理论等价"与"实践收敛"之别的活教材。"""

n = text.count(old)
if n != 1:
    print(f"[FAIL] 匹配次数={n}")
    sys.exit(1)
text = text.replace(old, new)
with io.open(PATH, 'w', encoding='utf-8') as f:
    f.write(text)
print("[OK] 讲义已更新")
