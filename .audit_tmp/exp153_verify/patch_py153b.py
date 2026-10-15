# -*- coding: utf-8 -*-
"""修补 1.5-3.py 的左面板：原图把 ‖y−Ax‖² 与 −ln p 画在同一纵轴上，
两者相差 1/(2σ²)=0.5 倍，视觉上完全分开，与图标题"完全重合"及讲义"几乎重合"矛盾。
改为三条线：实测 ls、实测 nll、预测 factor·ls（后者与 nll 严格重合）。"""
import io

P = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.5-3\1.5-3.py'
src = io.open(P, encoding='utf-8').read()

REPS = []


def rep(old, new):
    REPS.append((old, new))


rep("""# 左：最小二乘 vs 负对数似然（按候选排序，两条线应完全重合）
ax[0].plot(least_squares_list, label=r'$\\|y - A x\\|^2$（最小二乘）', linewidth=2)
ax[0].plot(neg_log_lik_list, '--', label=r'$-\\ln p(y|x)$（负对数似然）', linewidth=2)
ax[0].set_xlabel('候选解索引')
ax[0].set_ylabel('目标值')
ax[0].set_title(r'最小二乘 vs 负对数似然（完全重合 = 代数等价）')""",
    """# 左：三条线。注意 ‖y−Ax‖² 与 −ln p 只差因子 1/(2σ²)=0.5，
# 画在同一纵轴上必然分开 2 倍（不是"重合"）；真正该重合的是
# 实测 −ln p（橙虚线）与预测 (1/(2σ²))·‖y−Ax‖²（灰点线）。
ax[0].plot(least_squares_list, label=r'$\\|y - A x\\|^2$（最小二乘）', linewidth=2)
ax[0].plot(neg_log_lik_list, '--', label=r'$-\\ln p(y|x)$（负对数似然，实测）', linewidth=2)
ax[0].plot(factor * least_squares_list, ':', color='gray', linewidth=2.5,
           label=r'预测 $\\frac{1}{2\\sigma^2}\\|y-Ax\\|^2$（与橙虚线重合）')
ax[0].set_xlabel('候选解索引')
ax[0].set_ylabel('目标值')
ax[0].set_title(r'最小二乘与负对数似然：形状一致、相差 $1/(2\\sigma^2)$ 倍缩放' + '\\n' +
                r'（橙虚线与灰点线重合 = 仅差常数因子）')""")

rep("""plt.suptitle(r'实验1.5-3：高斯似然 = 最小二乘（$y = A x + \\varepsilon,\\ \\varepsilon\\sim\\mathcal{N}(0,\\sigma^2 I)$）',
             fontsize=13)""",
    """plt.suptitle(r'实验1.5-3：高斯似然 = 最小二乘（$y = A x + \\varepsilon,\\ \\varepsilon\\sim\\mathcal{N}(0,\\sigma^2 I)$）',
             fontsize=13)
# 图必须由本脚本重跑生成：归档图曾出现"标题编号写成 1.5-2"而脚本已是 1.5-3 的错配，
# 只有重跑并目视才能发现这类图内文字与代码不一致的问题。""")

for old, new in REPS:
    c = src.count(old)
    assert c == 1, '命中 %d 次: %r' % (c, old[:60])
    src = src.replace(old, new)

io.open(P, 'w', encoding='utf-8', newline='').write(src)
print('patched %d items OK' % len(REPS))
