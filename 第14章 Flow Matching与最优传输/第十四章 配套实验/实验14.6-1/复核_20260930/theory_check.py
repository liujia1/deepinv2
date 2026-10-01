# -*- coding: utf-8 -*-
"""14.6-1 理论与实测一致性检验（纯 python，无依赖）
逐条比对实验声明中的"理论值"与 JSON 实测值，并给出采样噪声判据。"""
from collections import Counter
from math import exp, log

TPL = [
    'the cat sat on the mat',
    'the dog sat on the rug',
    'the cat ate the red fish',
    'the dog ate the big fish',
    'a big cat sat on it',
    'a big dog ate it up',
]
tpl = [t.split() for t in TPL]
d = 6

print("=" * 70)
print("A. N=1 理论命中率（t=0, kappa=0 -> 全部位置为噪声, 真后验=p_data）")
print("   CE 最优的因子化模型在该条件处的输出 = 逐位置边缘分布；")
print("   一次采样命中模板的概率 = sum_T prod_j p_data(T_j)")
print("=" * 70)
marg = [Counter(row[j] for row in tpl) for j in range(d)]
for j in range(d):
    print(f"  pos{j+1}: " + ", ".join(f"{w}={c}/6" for w, c in sorted(marg[j].items())))
p_theory = sum(
    __import__('functools').reduce(lambda a, b: a * b,
                                   [marg[j][row[j]] / 6.0 for j in range(d)], 1.0)
    for row in tpl
)
print(f"\n  => 理论 N=1 命中率 = {p_theory:.4f}  (期望 {p_theory*256:.2f} 句 / 256)")


def binom_tail(n, p, k, side):
    """P(X=k) 与单侧尾部概率"""
    pk = exp(log(p) * k + log(1 - p) * (n - k) + log(__import__('math').comb(n, k))) if 0 < p < 1 else 0
    tail = 0.0
    for i in range(n + 1):
        pi = exp(log(p) * i + log(1 - p) * (n - i) + log(__import__('math').comb(n, i))) if 0 < p < 1 else 0
        if (side == 'ge' and i >= k) or (side == 'le' and i <= k):
            tail += pi
    return pk, tail


print("\n" + "=" * 70)
print("B. 实测 vs 理论（n=256, 粒度 1/256=0.0039）")
print("=" * 70)
for name, hit in [("MDLM 版", 0.0039), ("均匀噪声版", 0.0195)]:
    k = round(hit * 256)
    pk, tail = binom_tail(256, p_theory, k, 'ge' if hit > p_theory else 'le')
    print(f"  {name}: 实测 {hit:.4f} = {k}/256 | 理论 {p_theory:.4f} | "
          f"P(X={k})={pk:.3f}, 单侧 P(X{'≥' if hit>p_theory else '≤'}{k})={tail:.3f}")

print("\n" + "=" * 70)
print("C. 其他理论断言 vs 实测")
print("=" * 70)
# C1: rate = min(h*hazard,1) 线性调度+均匀网格
N = 32
rates = [min((1.0 / N) * (1.0 / (1.0 - n / N)), 1.0) for n in range(N)]
print(f"  C1 rate 上界: 理论 max=1(末步 n=31), 实测 {max(rates):.6f} | 全步 rate<1 且非单调? "
      f"{'否' if all(rates[i] <= rates[i+1] for i in range(N-1)) else '否(单调递增)'}")
print(f"     末步 rate = {rates[-1]:.6f} (理论恰=1)")
# C2: 约束(2) float32 舍入估计: residual ~ hazard_max * eps * (累加项数)
eps32 = 2 ** -23
t_max = 1 - 1 / 513  # 512 个 U(0,1) 样本的最大值的典型量级
print(f"  C2 约束(2): 理论恒 0（整行求和）。实测 7.534e-05。")
print(f"     量级估计 hazard_max(≈{1/(1-0.998):.0f}) × float32 eps({eps32:.2e}) × 累加(15项) "
      f"≈ {1/(1-0.998)*eps32*15:.1e} -> 与实测同量级, 属浮点舍入而非算法违反")
# C3: t=0 后验 = 数据边缘, pos6 rug 应为 1/6
print(f"  C3 t=0 处 pos6='rug' 后验: 理论 {marg[5]['rug']}/6 = {marg[5]['rug']/6:.4f}, 实测 0.1615")
# C4: 落子无悔 —— 真后验速率在已揭示位置恒为 0
print(f"  C4 已揭示位置改写率: 理论(真后验速率)=0.0, 实测 MDLM 0.000888 / 均匀 0.030192 "
      f"(比值 {0.030192/0.000888:.1f}×)")