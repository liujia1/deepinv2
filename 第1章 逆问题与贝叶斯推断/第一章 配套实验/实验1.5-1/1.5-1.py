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

素材来源：
  - 绪论 0.10"动手感受一下（二）"内嵌代码段迁移而来，规范化成本章配套实验
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 静默模式，非交互式后端，不弹出 GUI 窗口
import matplotlib.pyplot as plt
from scipy.stats import beta
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
results = []
a, b = 1.0, 1.0           # 先验 Beta(a,b) 的参数

# 预先生成好 500 次抛掷，按当前需要的次数截取
flips = (rng.random(500) < theta_true).astype(int)

print("=" * 60)
print("实验1.5-1：信念如何被数据'洗掉'（Beta-Bernoulli 共轭）")
print("=" * 60)
print(f"真实正面概率 θ = {theta_true}，先验 Beta(1,1)（均匀，啥也不知道）\n")
print(f"{'数据量 n':>10} | {'正面/反面':>14} | {'后验均值':>12} | {'后验方差':>12}")
print("-" * 58)

for n in n_flips_list:
    heads = int(flips[:n].sum())
    tails = n - heads
    a_post = a + heads     # 共轭更新：后验参数 = 先验参数 + 观测计数
    b_post = b + tails
    mean = a_post / (a_post + b_post)
    var = (a_post * b_post) / ((a_post + b_post) ** 2 * (a_post + b_post + 1))
    results.append((n, a_post, b_post, heads))
    print(f"{n:>10} | {heads:>6} / {tails:<6} | {mean:>12.4f} | {var:>12.6f}")

print(f"\n观察：数据从 0 增到 500，后验均值从 0.5 -> 向真实值 0.7 收敛，方差不断收缩。")

# ---- 画图：随着数据增多，后验从又宽又平 -> 又尖又锁定在 0.7 ----
xs = np.linspace(0, 1, 400)
fig, ax = plt.subplots(figsize=(10, 4))
for i, (n, ap, bp, heads) in enumerate(results):
    ax.plot(xs, beta.pdf(xs, ap, bp), lw=2, label=f"抛 {n} 次（{heads} 正）")
ax.axvline(theta_true, color='k', ls='--', alpha=0.6, label=f"真实概率 {theta_true}")
ax.set_xlabel("硬币正面概率 θ")
ax.set_ylabel("后验密度")
ax.set_title("后验随数据增多：移动并收拢，先验被'洗掉'")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_5_1_硬币贝叶斯更新.png'), dpi=150, bbox_inches='tight')  # 仅保存图片，不弹窗
plt.close()

# ---- 结论 ----
print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print("1. Beta 先验 + 伯努利似然 -> Beta 后验（共轭），更新只靠计数：a_n=a_0+正面, b_n=b_0+反面")
print("2. 数据越多，后验越窄、越锁向真实值 0.7；最初的均匀先验（Beta(1,1)）不再起作用")
print("3. 这与 1.5 节'世界人口估计（Poisson-Gamma 共轭）'是同一个故事：先验被数据稀释")
print("4. 这就是贝叶斯'信念更新'的全部精髓：数据按量级逐步稀释先验的话语权，并收拢不确定性")

# ===== 保存数值结果 =====
import json
results_summary = {
    'theta_true': float(theta_true),
    'prior': {'dist': 'Beta', 'a': float(a), 'b': float(b)},
    'data_sizes': n_flips_list,
    'posterior': [],
}
for n in n_flips_list:
    h = int(flips[:n].sum())
    ap, bp = a + h, b + (n - h)
    results_summary['posterior'].append({
        'n': n, 'heads': h,
        'a_post': float(ap), 'b_post': float(bp),
        'mean': float(ap / (ap + bp)),
        'var': float(ap * bp / ((ap + bp) ** 2 * (ap + bp + 1))),
    })

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