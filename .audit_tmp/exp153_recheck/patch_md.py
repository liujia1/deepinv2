# -*- coding: utf-8 -*-
"""修补 1.5 讲义中实验1.5-3 的路径3描述：区分'真实交叉检验'与'argmin演示项'。"""
import io, sys
p = r'D:\works\deepinv\第1章 逆问题与贝叶斯推断\1.5 贝叶斯框架：从似然到后验.md'
s = io.open(p, encoding='utf-8').read()

old_row = r'| 路径3 换 $\sigma=2$ 复核 | 比例因子 $0.5\to 0.125$，最优候选索引 $27=27$ |'
new_row = r'| 路径3 换 $\sigma=2$ 复核 | 比例因子 $0.5\to 0.125$；$\sigma=2$ 下两条独立算式复算偏差 $1.28\times 10^{-16}$；最优候选索引 $27=27$（演示项） |'

old_p3 = r'路径3 处理 $\sigma=1$ 造成的盲区：$\sigma=1$ 时 $1/(2\sigma^2)=1/(2\sigma)=0.5$，任何把 $\sigma^2$ 误写成 $\sigma$ 的幂次错误都不可见，故另取 $\sigma=2$ 复核比例因子确实从 $0.5$ 变到 $0.125$，而最优候选索引不变（$27=27$）——常数项与缩放都不改变 $\arg\min$，这才是"最大化似然 $=$ 最小二乘"的实质内容。'
new_p3 = r'路径3 处理 $\sigma=1$ 造成的盲区：$\sigma=1$ 时 $1/(2\sigma^2)=1/(2\sigma)=0.5$，任何把 $\sigma^2$ 误写成 $\sigma$ 的幂次错误都不可见（变异测试已证实），故另取 $\sigma=2$（此时 $\sigma\neq\sigma^2$）把按定义逐元素复算与闭式 $\text{const}+\|r\|^2/(2\sigma^2)$ 交叉比对，偏差 $1.28\times10^{-16}$，比例因子也随 $\sigma$ 从 $0.5$ 变到 $0.125$。附带演示最优候选索引不变（$27=27$）——但要诚实指出：正仿射变换（乘正数、加常数）必然保持 $\arg\min$，该等式恒成立、不构成数值检验；"最大化似然 $=$ 最小二乘"的实质证据是路径2、路径3 两次独立复算，$\arg\min$ 演示只负责直观。'

for tag, o, n in [('row', old_row, new_row), ('p3', old_p3, new_p3)]:
    cnt = s.count(o)
    if cnt != 1:
        print(f'[FAIL] {tag}: found {cnt} occurrences'); sys.exit(1)
    s = s.replace(o, n)
    print(f'[OK] {tag} replaced')

io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('saved')
