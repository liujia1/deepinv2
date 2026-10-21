# -*- coding: utf-8 -*-
"""14.2-1 补充修复：
(1) 把正文 L180 引用的「后300 epoch 去趋势残差 std」入库（此前只在正文口头存在）；
(2) 脚本末尾教学脚注的「MMD说FM更优」改为方向不定（重跑后 MMD 已反转为 CNLL 略优）；
(3) 「理论熵下界」措辞精确化为「目标分布微分熵」。
"""
PY = r'h:\deepinv\deepinv2\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.2-1\14.2-1.py'
s = open(PY, encoding='utf-8').read()

# ---- (1) json 汇总前插入去趋势残差 std 计算 ----
o1 = """results_summary = {
    'avg_time_cnll_per_epoch': round(float(np.mean(times_cnll)), 4) if times_cnll else None,"""
n1 = """# 损失曲线抖动指标：后300 epoch 一次线性去趋势后的残差 std。
# 去趋势是为了剥离余弦退火下仍在下行的趋势项，只留"抖动"。
def _loss_jitter(losses, tail=300):
    y = np.asarray(losses, dtype=np.float64)
    if len(y) < tail:
        return None
    seg = y[-tail:]
    x = np.arange(len(seg))
    resid = seg - np.polyval(np.polyfit(x, seg, 1), x)
    return float(resid.std())

jitter_cnll = _loss_jitter(losses_cnll)
jitter_fm = _loss_jitter(losses_fm)

results_summary = {
    'avg_time_cnll_per_epoch': round(float(np.mean(times_cnll)), 4) if times_cnll else None,"""
print('c1', s.count(o1)); assert s.count(o1) == 1
s = s.replace(o1, n1)

# ---- (2) json 增加抖动字段 ----
o2 = """    'entropy_note': 'log 8 + log(2*pi*e*sigma^2), sigma=0.15, 单位 nats',
}"""
n2 = """    'entropy_note': 'log 8 + log(2*pi*e*sigma^2), sigma=0.15, 单位 nats',
    'loss_jitter_cnll': round(jitter_cnll, 4) if jitter_cnll is not None else None,
    'loss_jitter_fm': round(jitter_fm, 4) if jitter_fm is not None else None,
    'loss_jitter_note': '后300 epoch一次线性去趋势后的残差std（抖动指标，非绝对值）',
}"""
print('c2', s.count(o2)); assert s.count(o2) == 1
s = s.replace(o2, n2)

# ---- (3) 脚本末尾脚注：MMD 方向不定 + 术语精确化 ----
o3 = """     但其NLL绝对值有偏：20步Euler离散化会系统性低估NLL（本实验最终值
     低于目标分布的理论熵下界），趋势正确、纵坐标不可当真值。"""
n3 = """     但其NLL绝对值有偏：20步Euler离散化会系统性低估NLL（本实验20步评估
     低于目标分布的微分熵，100步重估即回到熵上方），趋势正确、纵坐标不可当真值。"""
print('c3', s.count(o3)); assert s.count(o3) == 1
s = s.replace(o3, n3)

o4 = """6. 教学脚注：MMD说FM更优，但图2c/图3b中FM轨迹反而有更多"桥接"弧线？
   - 统计采样点到最近高斯中心的距离，FM的"桥接点"（距最近中心>0.5）占比见
     results_summary.json 的 bridge_fm_pct，高于Neural ODE的 bridge_cnll_pct。
     这说明FM在离散众数意义上更容易产生跨众数弧线。
   - 但MMD作为整体分布距离度量，对这几个百分点的桥接离群点不敏感，
     更多反映的是整体均值/协方差/形状的匹配（FM在这些方面确实更贴近目标）。
   - MMD更优不代表视觉上更"干净"——不同指标衡量的是分布匹配的不同侧面，
     这也是14.3.5节引入OT-CFM的动机之一。"""
n4 = """6. 教学脚注：MMD 与桥接点为何可能给出不同的结论？
   - 统计采样点到最近高斯中心的距离，两模型的"桥接点"（距最近中心>0.5）占比见
     results_summary.json 的 bridge_fm_pct / bridge_cnll_pct，FM 明显更高。
     这说明FM在离散众数意义上更容易产生跨众数弧线。
   - 而MMD作为整体分布距离度量，对这几个百分点的桥接离群点不敏感，
     更多反映的是整体均值/协方差/形状的匹配（见 json 的 mmd_cnll / mmd_fm，
     二者同量级，几次运行中谁略低并不稳定）。
   - 所以「MMD 谁更优」与「视觉上谁更干净」可以不一致——不同指标衡量的是
     分布匹配的不同侧面，这也是14.3.5节引入OT-CFM的动机之一。"""
print('c4', s.count(o4)); assert s.count(o4) == 1
s = s.replace(o4, n4)

open(PY, 'w', encoding='utf-8').write(s)
print('OK')
