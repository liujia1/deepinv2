"""量化修正口径：Laplace 软阈值正确参数 = σ²/b（代码误用 b·σ²），
并给出各先验 MMSE / MAP 的对照表与交叉点，供改写讲义与配图使用。"""
import sys
import numpy as np
from scipy.integrate import quad

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sigma, gamma, b, nu = 0.5, 1.0, 0.5, 2.0
thr_correct = sigma**2 / b
thr_code = b * sigma**2
print('阈值口径: 代码 b·σ² = %.4f  |  正确 σ²/b = %.4f  (Laplace(0,b): p∝exp(-|x|/b))' % (thr_code, thr_correct))

xg = np.linspace(-40, 40, 40001)
lp = {'gaussian': lambda x: -x**2/(2*gamma**2),
      'laplace': lambda x: -np.abs(x)/b,
      'student_t': lambda x: -((nu+1)/2)*np.log1p(x**2/nu)}

def pm(key, y):
    w = np.exp(lp[key](xg) - (y-xg)**2/(2*sigma**2))
    return np.sum(xg*w)/np.sum(w)

def pmode(key, y):
    return xg[np.argmax(lp[key](xg) - (y-xg)**2/(2*sigma**2))]

ys = [0.2, 0.3, 0.5, 1.0, 2.0, 3.0]
print('\n y     G(MMSE)  L(MMSE)  L(MAP=ST σ²/b)  t(MMSE)   |  gap: G     Lmmse  Lmap   t')
for y in ys:
    g = gamma**2/(gamma**2+sigma**2)*y
    lm, lo, tm = pm('laplace', y), pmode('laplace', y), pm('student_t', y)
    print(' %-5.2f  %+.4f   %+.4f   %+.4f (mode %+.4f)  %+.4f  |  %.4f  %.4f  %.4f  %.4f'
          % (y, g, lm, max(y-thr_correct, 0), lo, tm, y-g, y-lm, y-lo, y-tm))
    # 校验数值众数 == 闭式软阈值(σ²/b)
    assert abs(lo - max(y-thr_correct, 0)) < 1e-3, (lo, y)

print('\n近零 (y=0.203008 网格点) 压制百分比：')
yv = 0.2030075187969925
vals = {'gaussian': gamma**2/(gamma**2+sigma**2)*yv,
        'laplace_MAP(σ²/b)': max(yv-thr_correct, 0.0),
        'laplace_MAP(代码 b·σ²)': yv-thr_code,
        'laplace_MMSE': pm('laplace', yv),
        'student_t_MMSE': pm('student_t', yv)}
for k, v in vals.items():
    print('  %-22s D=%+.5f  压制 %.1f%%' % (k, v, (1-v/yv)*100))

print('\n斜率（y→0⁺，D/y 极限）与交叉点：')
fine = np.linspace(1e-3, 3.0, 3000)
G = gamma**2/(gamma**2+sigma**2)*fine
T = np.array([pm('student_t', y) for y in fine])
L = np.array([pm('laplace', y) for y in fine])
gapG, gapT, gapL = fine-G, fine-T, fine-L
def first_below(a, c):
    idx = np.where(a < c)[0]
    return fine[idx[0]] if len(idx) else None
print('  y→0 斜率: G %.4f  t %.4f  L(MMSE) %.4f' % (G[0]/fine[0], T[0]/fine[0], L[0]/fine[0]))
print('  t 的绝对收缩量 |y-D| 小于高斯的起点 y ≈ %s' % first_below(gapT, gapG))
print('  t 的绝对收缩量 小于 Laplace-MMSE 的起点 y ≈ %s' % first_below(gapT, gapL))
print('  t 的相对压制%% 小于高斯(20%%) 的起点 y ≈ %s' % first_below((1-T/fine)*100, 20.0))
print('  t 的绝对收缩量 小于 Laplace-MAP(σ²/b=%.3f) 的起点 y ≈ %s' % (thr_correct, first_below(gapT, thr_correct)))
print('\n最大绝对收缩量（t）: %.4f @ y=%.2f' % (gapT.max(), fine[np.argmax(gapT)]))
