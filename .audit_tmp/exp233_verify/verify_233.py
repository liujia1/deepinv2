"""复核 实验2.3-3：
1) 逐位复现 results_summary.json 的 sample_curves
2) 数值积分 vs 闭式对照（gaussian / identity / laplace-MAP）
3) 网格截断与步长敏感性（x_grid 范围/密度）
4) 独立求积（scipy.integrate.quad，无穷区间）交叉验证 Student-t 后验均值
5) 补算代码"算了但丢弃"的 Laplace MMSE（后验均值），检查图中 Laplace 曲线实为 MAP
6) 复核讲义/打印中的百分比断言与"谁收缩更温和"的排序断言
"""
import json, os, sys
import numpy as np
from scipy.integrate import quad

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

HERE = r'd:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\第二章 配套实验\实验2.3-3'
sigma, gamma, b, nu = 0.5, 1.0, 0.5, 2.0

y_grid = np.linspace(-3, 3, 400)
x_grid = np.linspace(-6, 6, 2000)

LPS = {
    'identity':  lambda x: np.zeros_like(x),
    'gaussian':  lambda x: -x**2/(2*gamma**2),
    'laplace':   lambda x: -np.abs(x)/b,
    'student_t': lambda x: -((nu+1)/2)*np.log1p(x**2/nu),
}

def post_mean(lp, y, xg):
    w = np.exp(lp(xg) - (y-xg)**2/(2*sigma**2))
    w -= 0.0
    return np.sum(xg*w)/np.sum(w)

def compute(xg, ys=None):
    ys = y_grid if ys is None else ys
    return {k: np.array([post_mean(lp, y, xg) for y in ys]) for k, lp in LPS.items()}

# ---------- 1) 复现主实验（与 2.3-3.py 同网格） ----------
D = compute(x_grid)
closed_g = gamma**2/(gamma**2+sigma**2)*y_grid
closed_l_map = np.sign(y_grid)*np.maximum(np.abs(y_grid)-b*sigma**2, 0.0)

print('== 1) 数值 vs 闭式 偏差 (max |D_num - closed|) ==')
print('  identity  vs y        : %.3e' % np.max(np.abs(D['identity'] - y_grid)))
print('  gaussian  vs 0.8y     : %.3e' % np.max(np.abs(D['gaussian'] - closed_g)))
print('  laplace MMSE vs MAP   : %.4f  (y=0.2 处: MMSE=%.4f vs MAP=%.4f)' % (
    np.max(np.abs(D['laplace'] - closed_l_map)),
    D['laplace'][np.argmin(abs(y_grid-0.2))], closed_l_map[np.argmin(abs(y_grid-0.2))]))
# MAP 的闭式对照：拉普lace 后验众数应等于软阈值
def post_mode(lp, y, xg):
    lg = lp(xg) - (y-xg)**2/(2*sigma**2)
    return xg[np.argmax(lg)]
mode_l = np.array([post_mode(LPS['laplace'], y, np.linspace(-6,6,20001)) for y in y_grid])
print('  laplace 数值众数 vs 软阈值: %.4e (grid 0.0006 量化内)' % np.max(np.abs(mode_l-closed_l_map)))

# ---------- 2) 逐位复现 JSON ----------
with open(os.path.join(HERE, 'results_summary.json'), encoding='utf-8') as f:
    js = json.load(f)
sample_y = js['sample_curves']['y']
print('\n== 2) JSON 逐位复现 ==')
ok = True
for i, yy in enumerate(sample_y):
    iy = int(np.argmin(np.abs(y_grid-yy)))
    row = (js['sample_curves']['gaussian'][i], closed_g[iy],
           js['sample_curves']['laplace'][i], closed_l_map[iy],
           js['sample_curves']['student_t'][i], D['student_t'][iy])
    d1 = abs(row[0]-row[1]); d2 = abs(row[2]-row[3]); d3 = abs(row[4]-row[5])
    ok &= max(d1, d2, d3) < 1e-12
    print('  y=%+5.1f grid=%.6f | G %.6f(Δ%.1e) L %.6f(Δ%.1e) t %.6f(Δ%.1e)'
          % (yy, y_grid[iy], row[1], d1, row[3], d2, row[5], d3))
print('  逐位一致:', ok)

# ---------- 3) 网格敏感性 ----------
print('\n== 3) 网格截断/步长敏感性 (Student-t 与 Laplace MMSE) ==')
cands = {'base[-6,6,2000]': np.linspace(-6,6,2000),
         'wide[-20,20,8000]': np.linspace(-20,20,8000),
         'wide[-40,40,16001]': np.linspace(-40,40,16001),
         'fine[-6,6,12000]': np.linspace(-6,6,12000)}
probe = [-3.0, -1.0, 0.2, 1.0, 3.0]
ref = None
for nm, xg in cands.items():
    vals_t = [post_mean(LPS['student_t'], y, xg) for y in probe]
    vals_l = [post_mean(LPS['laplace'], y, xg) for y in probe]
    if ref is None:
        ref = (vals_t, vals_l)
    print('  %-18s t: %s | laplaceMMSE: %s' % (nm,
          ' '.join('%+.5f' % v for v in vals_t), ' '.join('%+.5f' % v for v in vals_l)))
print('  与 base 最大偏离: t %.2e, laplace %.2e' % (
    max(abs(a-c) for a, c in zip(ref[0], [post_mean(LPS['student_t'], y, cands['wide[-40,40,16001]']) for y in probe])),
    max(abs(a-c) for a, c in zip(ref[1], [post_mean(LPS['laplace'], y, cands['wide[-40,40,16001]']) for y in probe]))))

# ---------- 4) 独立连续求积交叉验证 ----------
print('\n== 4) scipy quad 连续积分交叉验证（非网格） ==')
for y in [-3.0, 0.2, 1.0, 3.0]:
    res = {}
    for k in ['laplace', 'student_t', 'gaussian']:
        f_num = lambda x, k=k: np.exp(LPS[k](np.array([x]))[0] - (y-x)**2/(2*sigma**2)) * x
        f_den = lambda x, k=k: np.exp(LPS[k](np.array([x]))[0] - (y-x)**2/(2*sigma**2))
        num, _ = quad(f_num, -np.inf, np.inf, limit=400, epsabs=1e-13, epsrel=1e-11)
        den, _ = quad(f_den, -np.inf, np.inf, limit=400, epsabs=1e-13, epsrel=1e-11)
        res[k] = num/den
    print('  y=%+5.1f quad: G %.6f L %.6f t %.6f | 网格: G %.6f L %.6f t %.6f' % (
        y, res['gaussian'], res['laplace'], res['student_t'],
        post_mean(LPS['gaussian'], y, x_grid), post_mean(LPS['laplace'], y, x_grid),
        post_mean(LPS['student_t'], y, x_grid)))

# ---------- 5) 收缩强弱排序：三种 MMSE + Laplace MAP ----------
print('\n== 5) 收缩排序核查（相对压缩 % = (1-D/y)*100，及绝对残差 |y-D|）==')
yg = np.linspace(0.05, 3.0, 26)
Dl_mmse = np.array([post_mean(LPS['laplace'], y, x_grid) for y in yg])
Dt = np.array([post_mean(LPS['student_t'], y, x_grid) for y in yg])
Dg = gamma**2/(gamma**2+sigma**2)*yg
Dl_map = np.sign(yg)*np.maximum(np.abs(yg)-b*sigma**2, 0.0)
print('  %-6s %-8s %-8s %-8s %-8s %-8s %-8s' % ('y', 'G%缩', 'Lmap%缩', 't%缩', 'Lmmse%缩', '|y-t|', '|y-Lmap|'))
for i in range(0, len(yg), 5):
    y = yg[i]
    print('  %-6.2f %-8.1f %-8.1f %-8.1f %-8.1f %-8.3f %-8.3f' % (
        y, (1-Dg[i]/y)*100, (1-Dl_map[i]/y)*100, (1-Dt[i]/y)*100, (1-Dl_mmse[i]/y)*100,
        abs(y-Dt[i]), abs(y-Dl_map[i])))
# 交叉点：t 与 Laplace(MAP) 谁的残差更小
gap_t = yg - Dt
gap_lmap = b*sigma**2*np.ones_like(yg)
cross = np.where(gap_t < gap_lmap)[0]
print('  |y-D_t| < b·σ²(=%.3f) 的最小 y: %s' % (b*sigma**2, ('%.3f' % yg[cross[0]]) if len(cross) else '在 y≤3 范围内不存在'))
gap_lmmse = yg - Dl_mmse
cross2 = np.where(gap_t < gap_lmmse)[0]
print('  MMSE 口径下 |y-D_t| < |y-D_laplaceMMSE| 的最小 y: %s' % (('%.3f' % yg[cross2[0]]) if len(cross2) else '在 y≤3 范围内不存在'))
print('  y→0 局部斜率: G %.3f  L(MAP近似) %.3f  L(MMSE) %.3f  t %.3f' % (
    Dg[0]/yg[0], Dl_map[0]/yg[0], Dl_mmse[0]/yg[0], Dt[0]/yg[0]))

# ---------- 6) 讲义百分比断言 ----------
print('\n== 6) 讲义/打印 断言数值核对 ==')
for target in (0.2,):
    iy = int(np.argmin(np.abs(y_grid-target)))
    yv = y_grid[iy]
    print('  网格最近 y=%.6f (脚本口径) : L(MAP) 压制 %.1f%% | t 压制 %.1f%% | G 压制 %.1f%%'
          % (yv, (1-closed_l_map[iy]/yv)*100, (1-D['student_t'][iy]/yv)*100, (1-closed_g[iy]/yv)*100))
    lm = post_mean(LPS['laplace'], yv, x_grid)
    print('  同点 若用 Laplace 真 MMSE: D=%.5f → 压制 %.1f%%（与 t 的 %.1f%% 比较）'
          % (lm, (1-lm/yv)*100, (1-D['student_t'][iy]/yv)*100))
    print('  若按 y=0.2 精确值口径     : L(MAP) %.1f%% | t %.1f%%'
          % ((1-post_mean(LPS['laplace'], 0.2, np.linspace(-6,6,120001))/0.2)*100,
             (1-post_mean(LPS['student_t'], 0.2, np.linspace(-6,6,120001))/0.2)*100))
