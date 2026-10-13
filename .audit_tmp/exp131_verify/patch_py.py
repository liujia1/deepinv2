# -*- coding: utf-8 -*-
"""对 1.3-1.py 打补丁：修复渲染缺陷 / 陈旧注释 / 硬编码结论 / JSON 精度。
每处替换都断言命中 1 次，避免静默失配。"""
import io, sys

P = r'd:\works\deepinv\第1章 逆问题与贝叶斯推断\第一章 配套实验\实验1.3-1\1.3-1.py'
src = io.open(P, encoding='utf-8').read()
orig = src
REPS = []

# 1. 本地分支：中文字体模块路径改为脚本目录（否则从仓库根目录运行时静默丢字体）
REPS.append((
'''else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()''',
'''else:
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    # 用脚本所在目录拼路径，保证从仓库根目录运行也能找到 .chinese
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')'''))

# 2. 中文字体配置之后：mathtext 字体 + 对数轴刻度格式化
REPS.append((
'''# ========================================================

np.random.seed(42)''',
'''# ========================================================

# ---- 数学符号渲染 ----
# SimHei 缺 U+2212 数学减号与上标数字：默认 LogFormatter 的 \\mathdefault{} 会回退到
# 中文字体，使负指数刻度显示成叉框/豆腐块，故对数轴刻度统一改用纯 mathtext 渲染。
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False

def _pow10_fmt(v, _pos):
    e = int(round(np.log10(v)))
    return '$10$' if e == 1 else f'$10^{{{e}}}$'

np.random.seed(42)'''))

# 3. 记录每 κ 的实际数据相对误差（供 JSON 与讲义披露）
REPS.append((
'''rel_errors = []
theory_bound = []''',
'''rel_errors = []
theory_bound = []
data_rel_noise = []   # 各 κ 下真实的 ‖δy‖/‖y‖（并非恒等于 1e-8，见循环内说明）'''))

REPS.append((
'''    delta_y_norm = np.linalg.norm(y_noisy - y)
    theory_bound.append(kappa * delta_y_norm / y_norm)''',
'''    delta_y_norm = np.linalg.norm(y_noisy - y)
    theory_bound.append(kappa * delta_y_norm / y_norm)
    # 噪声按绝对量 1e-8 注入，而 ‖y‖ 随 σ 衰减变小 → 数据相对误差随 κ 缓慢漂移
    data_rel_noise.append(delta_y_norm / y_norm)'''))

# 4. 左图注释：κ=1/ε 只是量级判据，把实测值写进图里；上标改用 mathtext
REPS.append((
r'''axes[0].axvline(kappa_critical, color='dimgray', linestyle=':', linewidth=2.0, alpha=0.8)
# 在参考线旁添加注释：同时说明公式和含义
axes[0].text(kappa_critical*1.3, 2e-4, 
             f'κ = 1/ε = {kappa_critical:.0e}\n(误差≈1，开始失真)', 
             fontsize=8, color='darkred',''',
r'''axes[0].axvline(kappa_critical, color='dimgray', linestyle=':', linewidth=2.0, alpha=0.8)
# 在参考线旁添加注释：κ=1/ε 是量级判据，实测值直接写进图里，避免"误差≈1"这类与曲线对不上的断言
_err_map = dict(zip(kappas, rel_errors))
_crit_note = (f'(量级判据；此处实测误差 {_err_map[kappa_critical]:.2f})'
              if kappa_critical in _err_map else '(量级判据)')
axes[0].text(kappa_critical*1.3, 2e-4,
             f'κ = 1/ε = {kappa_critical:.0e}\n{_crit_note}',
             fontsize=8, color='darkred','''))

REPS.append((
r'''axes[0].text(kappa_hook*1.15, 1e-2,
             f'开头悬念：κ=10¹⁰\n(10⁻⁸ 误差 → 完全失真)',
             fontsize=8, color='purple',''',
r'''axes[0].text(kappa_hook*1.15, 1e-2,
             '开头悬念：$\\kappa=10^{10}$\n($10^{-8}$ 误差 → 完全失真)',
             fontsize=8, color='purple','''))

# 5. 右图标题：r'' 里的 \n 不会被解释成换行（图上印出字面 "\n"）；并挂上刻度格式化
REPS.append((
r'''axes[1].set_title(r'不同条件数的奇异值分布\n（小 $\sigma_i$ → $1/\sigma_i$ 爆炸 → 噪声放大）',
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)''',
r'''axes[1].set_title(r'不同条件数的奇异值分布' + '\n' +
                  r'（小 $\sigma_i$ → $1/\sigma_i$ 爆炸 → 噪声放大）',
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
# 对数轴刻度用纯 mathtext，避免 SimHei 缺字形把负指数显示成叉框
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(_pow10_fmt))
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(_pow10_fmt))
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(_pow10_fmt))'''))

# 6. 附图：机制表述纠正（正逆用同一 D → 信号零衰减，偏离全部来自被放大的噪声）
REPS.append((
r'''# 用一条含丰富高频细节的波形作待重建信号：高频越密，κ=10^6 的"发钝"越明显''',
r'''# 用一条含丰富高频细节的波形作待重建信号，便于看清"重建偏离"落在哪些索引上
# 注意：本实验正逆用同一个 D，x_hat = x + e/σ 恒成立 → 信号本身零衰减，
# 图上看到的偏离 100% 来自被 1/σ_i 放大的噪声，不是"高频细节被抹平"'''))

REPS.append((
r'''colors = {1e7: 'steelblue', 1e10: 'crimson'}
tags = {1e7: '轻微发钝', 1e10: '一团乱麻'}''',
r'''colors = {1e7: 'steelblue', 1e10: 'crimson'}
# 偏离来自尾部小 σ_i 方向被放大的噪声（信号本身未被平滑），故不再称作"发钝"
tags = {1e7: '仅尾部轻微噪声放大', 1e10: '一团乱麻'}'''))

REPS.append((
r'''ax2.set_xlabel('索引 / 位置', fontsize=11)''',
r'''ax2.set_xlabel('索引 / 位置（对角算子按索引衰减 $\sigma_i$，此处索引不等于频率）', fontsize=11)'''))

# 7. 基线精度注释修正 + 把"丢几位"换算抽成函数（供两个输出分支共用）
REPS.append((
'''baseline_digits = round(-np.log10(noise_level))  # = 6，表示噪声本身的精度''',
'''baseline_digits = round(-np.log10(noise_level))  # = 8，噪声本身给出的精度上限

def _lost_str(k):
    """把实测相对误差换算成"丢了几位有效数字"；误差 ≥1 记为完全失真"""
    e = _err_map[k]
    if e >= 1.0:
        return '完全失真（误差>1）'
    return f'约丢 {max(0, baseline_digits - int(round(-np.log10(e))))} 位有效数字\''''))

# 8. 数值解读：改为读本次实测，避免与上方动态表格自相矛盾（原写 10^6 丢6位，实测丢5位）
REPS.append((
r'''print("\n数值解读（参考1.3节）：")
print("- κ ≈ 10^3：轻度病态，约丢失3位有效数字")
print("- κ ≈ 10^6：中度病态，约丢失6位有效数字")
print("- κ ≈ 10^10：严重病态，几乎无法恢复有意义的解")''',
r'''print("\n数值解读（取自本次实测）：")
print(f"- κ ≈ 10^3：{_lost_str(1e3)}")
print(f"- κ ≈ 10^6：{_lost_str(1e6)}")
print(f"- κ ≈ 10^8：{_lost_str(1e8)}（κ=1/ε 仅为量级判据）")
print(f"- κ ≈ 10^10：{_lost_str(1e10)}")
print(f"- 数据相对误差 ‖δy‖/‖y‖ 随 κ 从 {data_rel_noise[0]:.2e} 漂到 {data_rel_noise[-1]:.2e}"
      f"（噪声按绝对量注入，‖y‖ 随 σ 衰减变小，并非严格恒定 1e-8）")'''))

# 9. Jupyter 分支同步：理论预期保留，但补上本次实测（用字符串拼接，避免 f-string 与 LaTeX 大括号冲突）
REPS.append((
r'''**数值示例**：
- $\\kappa \\approx 10^3$：轻度病态，约丢失3位有效数字
- $\\kappa \\approx 10^6$：中度病态，约丢失6位有效数字
- $\\kappa \\approx 10^{10}$：严重病态，几乎无法恢复有意义的解''',
r'''**数值示例（理论预期 → 本次实测）**：
- $\\kappa \\approx 10^3$：轻度病态，约丢失3位有效数字 → 实测 """'''
    + ''' + _lost_str(1e3) + """'''
    + r'''
- $\\kappa \\approx 10^6$：中度病态，约丢失6位有效数字 → 实测 """'''
    + ''' + _lost_str(1e6) + """'''
    + r'''
- $\\kappa \\approx 10^{10}$：严重病态，几乎无法恢复 → 实测 """'''
    + ''' + _lost_str(1e10) + """'''
    + r'''
（$\\kappa = 1/\\varepsilon = 10^8$ 只是量级判据：该处实测误差 """'''
    + ''' + format(_err_map[1e8], '.2f') + """'''
    + r'''，尚未到 1'''))

# 10. 结论第 3 条：写明上界有多保守（实测/上界），不再含糊说"保守估计"
REPS.append((
r'''print("3. 理论上界 κ·(‖δy‖/‖y‖) 为误差提供了保守估计")''',
r'''_ratios = [e / b for e, b in zip(rel_errors, theory_bound) if b > 0]
print(f"3. 理论上界 κ·(‖δy‖/‖y‖) 很保守：本次实测误差仅为上界的 "
      f"{min(_ratios):.1e}~{max(_ratios):.1e} 倍（最坏情形要求噪声全部落在 σ_min 方向）")'''))

# 11. JSON：全精度 + 补齐附图与派生量
REPS.append((
r'''    'relative_errors': [float(round(e, 6)) for e in rel_errors],
    'theory_bounds': [float(round(b, 6)) for b in theory_bound],
}''',
r'''    # 全精度保存：原先 round(e, 6) 会把 κ≤10^2 的行压成 0.0，归档失去信息
    'relative_errors': [float(e) for e in rel_errors],
    'theory_bounds': [float(b) for b in theory_bound],
    'data_rel_noise': [float(v) for v in data_rel_noise],
    'measured_over_bound_ratio': [float(e / b) if b > 0 else None
                                  for e, b in zip(rel_errors, theory_bound)],
    'kappa_critical_rule': float(kappa_critical),
    'demo_shared_noise': True,
    'demo_relative_errors': {f'{kk:.0e}': float(recons[kk][1]) for kk in demo_kappas},
}'''))

for i, (old, new) in enumerate(REPS, 1):
    cnt = src.count(old)
    if cnt != 1:
        print(f'[FAIL] 第{i}处命中 {cnt} 次：\n---\n{old[:200]}\n---')
        sys.exit(1)
    src = src.replace(old, new)

io.open(P, 'w', encoding='utf-8', newline='').write(src)
print(f'OK 共 {len(REPS)} 处替换，字符数 {len(orig)} -> {len(src)}')
