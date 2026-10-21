# -*- coding: utf-8 -*-
"""L2 正文同步：14.2 CNF 章节按重跑后 json 更新 7 处。
重点：MMD 方向反转（旧 FM 优 -> 新 CNLL 略优）导致的两处叙事重写。
"""
MD = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\14.2 连续归一化流(CNF).md'
s = open(MD, encoding='utf-8').read()

reps = []

# (1) 预期结果：加速比预期 10-20x -> 与实测对齐
reps.append((
"- Flow Matching训练速度显著快于Neural ODE（加速比约10-20x）",
"- Flow Matching训练速度显著快于Neural ODE（是否仿真ODE决定数量级差异，本机实测约39x；详见步骤1）",
))

# (2) 时间表 + 墙钟波动说明
reps.append((
"""**训练时间对比**：

| 方法 | 平均每epoch时间 | 加速比 |
|------|----------------|--------|
| Neural ODE | ~1.25s | 1.0x |
| Flow Matching | ~0.08s | ~15.8x |
""",
"""**训练时间对比**（json 字段 `avg_time_cnll_per_epoch` / `avg_time_fm_per_epoch` / `speedup`）：

| 方法 | 平均每epoch时间 | 加速比 |
|------|----------------|--------|
| Neural ODE | ~1.68s | 1.0x |
| Flow Matching | ~0.04s | ~38.7x |

> 墙钟绝对值随机器与并行负载波动（同一脚本在不同负载下跑过，加速比在十几倍到近四十倍之间浮动），但"是否仿真 ODE"带来的数量级差异是稳定的。
""",
))

# (3) 抖动 std：数值微调 + 字段指引
reps.append((
"实测反而是**FM的曲线更抖**（后300 epoch去趋势残差std：FM≈0.046 vs CNLL≈0.020）",
"实测反而是**FM的曲线更抖**（后300 epoch一次线性去趋势后的残差std：FM≈0.045 vs CNLL≈0.021，json 字段 `loss_jitter_fm` / `loss_jitter_cnll`）",
))

# (4) 时间-收敛段：40s->22s、624s->841s、熵表述与 100 步重估
reps.append((
"""右图展示时间-收敛效率：Flow Matching全部500个epoch仅需约40秒墙钟时间即到达平台，Neural ODE同等轮数约需624秒——这直接验证14.2.4节"CNF最大似然训练的瓶颈是仿真代价"。另需注意：CNLL的NLL**绝对值系统性偏低**（最终值0.95低于目标分布理论熵下界≈1.12，因为交叉熵不可能低于熵）——这是20步Euler离散化偏差所致，把同一模型用100步重估得NLL≈1.16即回到合理区间。趋势正确，但别把CNLL曲线的纵坐标当真值。""",
"""右图展示时间-收敛效率：Flow Matching全部500个epoch仅需约22秒墙钟时间即到达平台，Neural ODE同等轮数约需841秒——这直接验证14.2.4节"CNF最大似然训练的瓶颈是仿真代价"。另需注意：CNLL的NLL**绝对值系统性偏低**（20步评估的最终值为0.969，低于目标分布的微分熵1.1231 nats——8个等权高斯σ=0.15的微分熵 $H=\\log 8+\\log(2\\pi e\\sigma^2)=1.1231$，而交叉熵不应低于熵）——这是20步Euler离散化偏差所致；把同一模型用100步重估得NLL≈1.173，即回到熵上方。趋势正确，但别把CNLL曲线的纵坐标当真值（对应 json 的 `final_loss_cnll`、`nll_recheck_20step` / `nll_recheck_100step`、`entropy_theory`）。""",
))

# (5) MMD 表 + 结论句（方向反转，重写）
reps.append((
"""| 方法 | MMD vs 真实分布 |
|------|-----------------|
| Neural ODE | ~0.0015 |
| Flow Matching | ~0.0010 |

三幅子图展示：(a)目标分布（8-Gaussian），(b)Neural ODE采样，(c)Flow Matching采样。两种方法都能学会目标分布的8峰结构，FM在MMD上略优。""",
"""| 方法 | MMD vs 真实分布 |
|------|-----------------|
| Neural ODE | 0.00102 |
| Flow Matching | 0.00109 |

三幅子图展示：(a)目标分布（8-Gaussian），(b)Neural ODE采样，(c)Flow Matching采样。两种方法都能学会目标分布的8峰结构；MMD 二者同量级（相差约7%，本次运行 CNLL 反而略低），在采样噪声范围内不足以分出优劣——这与"两者最终目标分布相同"的预期一致（json 字段 `mmd_cnll` / `mmd_fm`）。""",
))

# (6) 曲率表
reps.append((
"""| 方法 | 平均曲率 $\\kappa$ |
|------|------------------|
| Neural ODE | ~0.12 |
| Flow Matching | ~0.35 |""",
"""| 方法 | 平均曲率 $\\kappa$ |
|------|------------------|
| Neural ODE | 0.0466 |
| Flow Matching | 0.3533 |""",
))

# (7) 桥接点脚注（数值更新 + 叙事重写）
reps.append((
"""**MMD vs 桥接点**：统计采样点到最近高斯中心的距离，FM的"桥接点"（距最近中心>0.5）占比实测约4.5%，高于Neural ODE的约1.2%（真实数据自身仅约0.4%）。这说明FM在离散众数意义上更容易产生跨众数弧线。但MMD作为整体分布距离度量，对这几个百分点的桥接离群点不敏感，更多反映整体均值/协方差/形状匹配（FM在这些方面确实更贴近目标）。关键启示：MMD说FM更优，不代表视觉上更"干净"——不同指标衡量的是分布匹配的不同侧面。""",
"""**MMD vs 桥接点**：统计采样点到最近高斯中心的距离，FM的"桥接点"（距最近中心>0.5）占比实测约4.9%，高于Neural ODE的约1.8%（真实数据自身仅约0.45%；json 字段 `bridge_fm_pct` / `bridge_cnll_pct` / `bridge_real_pct`）。这说明FM在离散众数意义上更容易产生跨众数弧线。但MMD作为整体分布距离度量，对这几个百分点的桥接离群点不敏感，更多反映整体均值/协方差/形状匹配。关键启示：**MMD 给出的"谁更优"很不稳定**（本次运行 CNLL 仅以约7%略低，换一次运行可能反转），而桥接点占比稳定地显示"FM 跨众数弧线更多"——评估生成质量时，整体分布指标与局部形态指标必须一起看。""",
))

for i, (o, n) in enumerate(reps, 1):
    c = s.count(o)
    print('c%d' % i, c)
    assert c == 1, 'FAIL at %d' % i
    s = s.replace(o, n)

open(MD, 'w', encoding='utf-8').write(s)
print('OK')
