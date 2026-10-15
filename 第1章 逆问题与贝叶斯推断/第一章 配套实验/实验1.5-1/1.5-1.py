# -*- coding: utf-8 -*-
"""
实验1.5-1 信念如何被数据"洗掉"：Beta-Bernoulli 共轭的贝叶斯更新
对应章节：1.5 贝叶斯框架：从似然到后验（"信念更新叙事"小节）

知识点：
  - 共轭先验让后验更新有解析形式：Beta 先验 + 伯努利似然 -> Beta 后验
  - 数据越多，后验从"又宽又平"变"又尖又锁定"，先验话语权被数据稀释
  - 后验均值向真实参数移动，方差（不确定性）随 n 增大而收拢

实验内容：
  - 硬币正面概率的先验取 Beta(1,1)（均匀，啥也不知道），真实 θ=0.7
  - 逐批观测抛掷结果，用共轭公式 a_n = a_0 + 正面数, b_n = b_0 + 反面数 更新后验
  - 绘制随数据量 {0,2,10,50,500} 演化的后验密度曲线，观察移动与收拢
  - 附加 4 万次蒙特卡洛重演：把"种子 42 这一次抽样的运气"量化下来（正文里那几个
    重演数字必须能从本脚本复现，不能只写在正文里）
  - 图为双面板：左为完整密度轴（看收拢），右为局部放大（看小样本时曲线如何从
    平坦先验慢慢"鼓"起来——线性轴上这一差别只有约 10 个像素，单面板看不见）

素材来源：
  - 绪论 0.10"动手感受一下（二）"内嵌代码段迁移而来，规范化成本章配套实验
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 静默模式，非交互式后端，不弹出 GUI 窗口
import matplotlib.pyplot as plt
from scipy.stats import beta, binom
import os
import sys

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验1.5-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)
# ========================================================

rng = np.random.default_rng(42)
theta_true = 0.7          # 硬币真实的正面概率（你假装不知道）
n_flips_list = [0, 2, 10, 50, 500]   # 逐步增多的数据量
N_MAX = max(n_flips_list) # 抛掷总次数由数据量列表决定，改列表不会被静默截断
results = []
a, b = 1.0, 1.0           # 先验 Beta(a,b) 的参数
prior_mean = a / (a + b)

# 预先生成好 N_MAX 次抛掷，按当前需要的次数截取
# （取前缀 => 各 n 的后验是同一条数据路径上的嵌套更新，这才是"信念随数据演化"）
flips = (rng.random(N_MAX) < theta_true).astype(int)


def posterior_stats(n, heads, a0=a, b0=b):
    """共轭更新。全脚本只在这里算一次均值/方差，避免正文表格与图出自两套实现"""
    a_post = a0 + heads                 # 后验参数 = 先验参数 + 观测计数
    b_post = b0 + (n - heads)
    mean = a_post / (a_post + b_post)
    var = (a_post * b_post) / ((a_post + b_post) ** 2 * (a_post + b_post + 1))
    mode = (a_post - 1) / (a_post + b_post - 2) if (a_post > 1 and b_post > 1) else None
    return a_post, b_post, mean, var, mode

print("=" * 60)
print("实验1.5-1：信念如何被数据'洗掉'（Beta-Bernoulli 共轭）")
print("=" * 60)
print(f"真实正面概率 θ = {theta_true}，先验 Beta(1,1)（均匀，啥也不知道）\n")
print(f"{'数据量 n':>10} | {'正面/反面':>14} | {'后验均值':>12} | {'后验方差':>12}")
print("-" * 58)

for n in n_flips_list:
    heads = int(flips[:n].sum())
    tails = n - heads
    a_post, b_post, mean, var, mode = posterior_stats(n, heads)
    results.append({'n': n, 'heads': heads, 'tails': tails, 'a_post': a_post,
                    'b_post': b_post, 'mean': mean, 'var': var, 'mode': mode,
                    'prior_weight': (a + b) / (a + b + n)})
    print(f"{n:>10} | {heads:>6} / {tails:<6} | {mean:>12.4f} | {var:>12.6f}")

print(f"\n观察：数据从 0 增到 {N_MAX}，后验均值从先验均值 {prior_mean:.4f} 收敛到 "
      f"{results[-1]['mean']:.4f}（真值 {theta_true}），方差收缩 "
      f"{results[0]['var'] / results[-1]['var']:.0f} 倍。")
print("      均值是否逐级上升取决于这一次抽样的运气（见下方重演）；方差才是稳定收缩的量。")

# ---- 画图：随着数据增多，后验从又宽又平 -> 又尖又锁定在 0.7 ----
# 左面板用完整尺度看"收拢"，右面板用局部放大看小样本时曲线如何"鼓起来"
xs = np.linspace(0, 1, 400)
fig, (ax, axz) = plt.subplots(1, 2, figsize=(13.2, 4.2),
                              gridspec_kw={'width_ratios': [1.25, 1]})
for r in results:
    line, = ax.plot(xs, beta.pdf(xs, r['a_post'], r['b_post']), lw=2,
                    label=f"抛 {r['n']} 次（{r['heads']} 正）")
    axz.plot(xs, beta.pdf(xs, r['a_post'], r['b_post']), lw=2, color=line.get_color())
ax.axvline(theta_true, color='k', ls='--', alpha=0.6, label=f"真实概率 {theta_true}")
axz.axvline(theta_true, color='k', ls='--', alpha=0.6)

# 右面板：标出小样本三条曲线在 θ=0.5 处的密度，让"峰钉在 0.5"可读
small = results[:3]
peaks = [beta.pdf(0.5, r['a_post'], r['b_post']) for r in small]
axz.scatter([0.5] * len(small), peaks, s=26, zorder=5, linewidths=0,
            c=[l.get_color() for l in ax.lines[:len(small)]])
axz.text(0.03, 0.05, "θ=0.5 处密度：" + " / ".join(f"{p:.2f}" for p in peaks) +
         "\n（1.00 即平坦先验本身）\n→ 小样本的峰仍钉在 0.5", transform=axz.transAxes,
         fontsize=8, va='bottom')
axz.set_xlim(0.1, 0.9)
axz.set_ylim(0, 3.2)
axz.set_xlabel("硬币正面概率 θ")
axz.set_title("局部放大（密度 0~3.2）")
axz.grid(True, alpha=0.3)

ax.set_xlabel("硬币正面概率 θ")
ax.set_ylabel("后验密度")
ax.set_title("后验随数据增多：移动并收拢，先验被'洗掉'")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_5_1_硬币贝叶斯更新.png'), dpi=150, bbox_inches='tight')  # 仅保存图片，不弹窗
plt.close()

# ---- 蒙特卡洛重演：把"这一次抽样的运气"量化下来（正文引用的重演数字出自这里） ----
N_MC = 40000
MC_SEED = 2026
print("\n" + "=" * 60)
print(f"蒙特卡洛重演：{N_MC} 条独立数据路径（重演种子 {MC_SEED}）")
print("=" * 60)
mc_rng = np.random.default_rng(MC_SEED)
mc_flips = mc_rng.random((N_MC, N_MAX)) < theta_true

mc_replay = {'n_replays': N_MC, 'mc_seed': MC_SEED, 'by_n': []}
print(f"{'数据量 n':>10} | {'均值中位数':>12} | {'P(均值=先验)':>14} | {'解析值':>10} | {'P(均值>真值)':>13}")
print("-" * 70)
for n in n_flips_list[1:]:
    h = mc_flips[:, :n].sum(1)
    m = (a + h) / (a + b + n)
    p_eq = float(np.mean(np.isclose(m, prior_mean)))
    # 均值回到先验均值 0.5 等价于正面数恰为 n/2，这是解析可得的量，不必靠重演估
    p_eq_exact = float(binom.pmf(n // 2, n, theta_true)) if n % 2 == 0 else 0.0
    p_gt = float(np.mean(m > theta_true))
    med = float(np.median(m))
    mc_replay['by_n'].append({'n': n, 'median_mean': med,
                              'prob_mean_equals_prior': p_eq,
                              'prob_mean_equals_prior_exact': p_eq_exact,
                              'prob_mean_exceeds_true': p_gt})
    print(f"{n:>10} | {med:>12.4f} | {p_eq:>14.4f} | {p_eq_exact:>10.4f} | {p_gt:>13.4f}")

# 均值路径的随机性：非单调、以及"冲过真值再回落"有多常见
ms_ck = np.stack([(a + mc_flips[:, :n].sum(1)) / (a + b + n) for n in n_flips_list], 1)
p_nonmono = float(np.mean(np.any(np.diff(ms_ck, axis=1) < 0, axis=1)))
p_over_fall = float(np.mean(np.any(ms_ck > theta_true, axis=1) &
                            np.any(np.diff(ms_ck, axis=1) < 0, axis=1)))
mc_replay['mean_path'] = {'prob_non_monotone': p_nonmono,
                          'prob_overshoot_then_fall': p_over_fall}
print(f"\n均值路径：非单调的比例 = {p_nonmono:.4f}；"
      f"至少一次冲过真值 {theta_true} 后又回落的比例 = {p_over_fall:.4f}")

# 方差单调性：分"检查点尺度"与"逐次更新尺度"两层检验（两者结论不同）
ns = np.array(n_flips_list, dtype=float)
hs = np.stack([mc_flips[:, :n].sum(1) for n in n_flips_list], 1).astype(float)
ap = a + hs
bp = b + (ns[None, :] - hs)
vs = ap * bp / ((ap + bp) ** 2 * (ap + bp + 1))
p_var_dec = float(np.mean(np.all(np.diff(vs, axis=1) < 0, axis=1)))

v_prior = a * b / ((a + b) ** 2 * (a + b + 1))
n_inc = 0
CH = 4000
for s in range(0, N_MC, CH):
    c = mc_flips[s:s + CH].cumsum(1).astype(float)
    nn = np.arange(1, N_MAX + 1)[None, :]
    aa, bbb = a + c, b + nn - c
    vv = np.concatenate([np.full((c.shape[0], 1), v_prior),
                         aa * bbb / ((aa + bbb) ** 2 * (aa + bbb + 1))], axis=1)
    n_inc += int(np.count_nonzero(np.any(np.diff(vv, axis=1) > 0, axis=1)))
p_stepwise_inc = n_inc / N_MC

_, _, _, v_tt, _ = posterior_stats(2, 0)    # 数据"反反" -> Beta(1,3)
_, _, _, v_tth, _ = posterior_stats(3, 1)   # 再来一次正面 -> Beta(2,3)
print(f"\n方差单调性：")
print(f"  在 {n_flips_list} 这几个检查点上严格下降的路径比例 = {p_var_dec:.4f}")
print(f"  逐次更新时至少出现一次方差上升的路径比例 = {p_stepwise_inc:.4f}")
print(f"  解析反例：数据'反反正'时 Beta(1,3) 方差 {v_tt:.5f} -> Beta(2,3) 方差 {v_tth:.5f}（上升）")
print("  => 方差单调是'检查点尺度上几乎必然'，不是逐次更新的定理；先验话语权 2/(2+n) 则严格递减")

mc_replay['variance_monotonicity'] = {
    'prob_decreasing_at_checkpoints': p_var_dec,
    'prob_stepwise_increase_somewhere': p_stepwise_inc,
    'stepwise_counterexample': {'Beta(1,3)_var': v_tt, 'Beta(2,3)_var': v_tth},
}
# 种子 42 这一次"均值停在 0.5"的解析概率（对应正文的"运气"说法）
p_seed42 = {
    'P_first2_exactly_1_head': float(binom.pmf(1, 2, theta_true)),
    'P_next8_exactly_4_heads': float(binom.pmf(4, 8, theta_true)),
    'P_joint': float(binom.pmf(1, 2, theta_true) * binom.pmf(4, 8, theta_true)),
}
print(f"\n种子 42 的\"运气\"：P(前2次恰1正)={p_seed42['P_first2_exactly_1_head']:.4f}，"
      f"P(后8次恰4正)={p_seed42['P_next8_exactly_4_heads']:.4f}，"
      f"联合={p_seed42['P_joint']:.4f}")

# ---- 结论 ----
print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print("1. Beta 先验 + 伯努利似然 -> Beta 后验（共轭），更新只靠计数：a_n=a_0+正面, b_n=b_0+反面")
print("2. 数据越多，后验越窄、越锁向真实值 0.7；最初的均匀先验（Beta(1,1)）不再起作用")
print("3. 这与 1.5 节'世界人口估计（Poisson-Gamma 共轭）'是同一个故事：先验被数据稀释")
print("4. 这就是贝叶斯'信念更新'的全部精髓：数据按量级逐步稀释先验的话语权，并收拢不确定性")
print("5. 但均值走的路径并不单调：小样本时先验权重 2/(2+n) 还很大，均值可能停在先验附近")
print("   甚至冲过真值再回落——这是抽样的固有随机性（见上方重演），随 n 增大按 1/n 消失")

# ===== 保存数值结果 =====
import json
results_summary = {
    'theta_true': float(theta_true),
    'prior': {'dist': 'Beta', 'a': float(a), 'b': float(b)},
    'data_sizes': n_flips_list,
    # 直接复用主循环的 results，不再第二套实现重算一遍
    'posterior': [{k: r[k] for k in ('n', 'heads', 'tails', 'a_post', 'b_post',
                                     'mean', 'var', 'mode', 'prior_weight')}
                  for r in results],
    'seed42_explanation': p_seed42,
    'mc_replay': mc_replay,
}

def _to_native(obj):
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except Exception:
        pass
    return obj

results_summary = {k: _to_native(v) for k, v in results_summary.items()}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"\n数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")