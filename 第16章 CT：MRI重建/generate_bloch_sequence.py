# -*- coding: utf-8 -*-
"""
临时脚本：生成 Bloch 脉冲序列示意图（SE / GRE / EPI 三序列对比 + k-space 填充轨迹）
纯几何示意图，不依赖实验数据 / GPU，秒级出图。
"""
import os
import sys
import io
import warnings
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['font.size'] = 10

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

C = {
    'rf':     '#D32F2F',
    'slice':  '#1976D2',
    'phase':  '#388E3C',
    'freq':   '#F57C00',
    'signal': '#7B1FA2',
    'te':     '#E91E63',
    'tr':     '#607D8B',
}

def gauss(t, center, width, amp):
    return amp * np.exp(-((t - center) / width) ** 2)

def trapz(t, center, flat, rise, amp):
    half = flat / 2 + rise
    x = np.abs(t - center)
    out = np.zeros_like(t, dtype=float)
    mf = x <= flat / 2
    mr = (x > flat / 2) & (x <= half)
    out[mf] = amp
    out[mr] = amp * (half - x[mr]) / rise
    return out

t = np.linspace(0, 100, 2000)

BANDS = [('RF', 4, C['rf']), ('$G_z$ 选层', 3, C['slice']),
         ('$G_y$ 相位', 2, C['phase']), ('$G_x$ 读出', 1, C['freq']),
         ('信号 / ADC', 0, C['signal'])]

def new_timing_ax(fig, row):
    ax = fig.add_subplot(gs[row, 0])
    ax.set_ylim(-0.7, 6.1)
    ax.set_yticks([b[1] for b in BANDS])
    ax.set_yticklabels([b[0] for b in BANDS], fontsize=10)
    ax.set_xlim(0, 100)
    for y in [b[1] for b in BANDS]:
        ax.axhline(y, color='#BBBBBB', lw=0.6, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def fill_band(ax, y, sig, color, alpha=0.6):
    ax.fill_between(t, y, y + sig, color=color, alpha=alpha, lw=0)
    ax.plot(t, y + sig, color=color, lw=1.2)

def draw_SE(ax):
    t90, t180, techo = 8, 29, 50
    fill_band(ax, 4, gauss(t, t90, 2, 0.9), C['rf'])
    fill_band(ax, 4, gauss(t, t180, 3, 1.7), C['rf'])
    ax.annotate('90°', (t90, 4.95), ha='center', fontsize=9, color=C['rf'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.9), zorder=10)
    ax.annotate('180°', (t180, 5.75), ha='center', fontsize=9, color=C['rf'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.9), zorder=10)
    gz = (gauss(t, t90, 4, 0.7) + gauss(t, t180, 4, 0.7)
          - gauss(t, t90 + 5, 3, 0.35) - gauss(t, t180 + 5, 3, 0.35))
    fill_band(ax, 3, gz, C['slice'])
    gy = trapz(t, (t90 + t180) / 2, 8, 3, 0.6)
    fill_band(ax, 2, gy, C['phase'])
    ax.annotate('相位编码\n(每 TR 变强度)', (18, 3.25), ha='center', fontsize=8, color=C['phase'],
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9), zorder=10)
    gx = -gauss(t, techo - 6, 3, 0.35) + gauss(t, techo, 8, 0.7)
    fill_band(ax, 1, gx, C['freq'])
    ax.axvspan(techo - 5, techo + 5, color=C['freq'], alpha=0.12)
    fid = gauss(t, t90 + 2, 6, 0.5) * np.cos(1.2 * (t - t90)) * (t > t90)
    echo = gauss(t, techo, 6, 0.85) * np.cos(0.6 * (t - techo))
    sig = fid + echo
    fill_band(ax, 0, sig, C['signal'], alpha=0.45)
    ax.axvline(techo, color=C['te'], ls='--', lw=1.4, alpha=0.8)
    ax.annotate('', (t90, -0.45), (techo, -0.45),
                arrowprops=dict(arrowstyle='<->', color=C['te'], lw=1.3))
    ax.text((t90 + techo) / 2, -0.5, 'TE', ha='center', fontsize=9, color=C['te'])
    ax.annotate('', (t90, -0.6), (96, -0.6),
                arrowprops=dict(arrowstyle='<->', color=C['tr'], lw=1.3))
    ax.text((t90 + 96) / 2, -0.65, 'TR', ha='center', fontsize=9, color=C['tr'])

def draw_GRE(ax):
    t90, techo = 8, 28
    fill_band(ax, 4, gauss(t, t90, 2.5, 0.45), C['rf'])
    ax.annotate('小角度\n(如 30°)', (t90, 4.55), ha='center', fontsize=8, color=C['rf'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9), zorder=10)
    gz = gauss(t, t90, 4, 0.7) - gauss(t, t90 + 5, 3, 0.35)
    fill_band(ax, 3, gz, C['slice'])
    gy = trapz(t, (t90 + techo) / 2, 8, 3, 0.6)
    fill_band(ax, 2, gy, C['phase'])
    gx = gauss(t, techo, 7, 1.0)
    fill_band(ax, 1, gx, C['freq'])
    ax.axvspan(techo - 4, techo + 4, color=C['freq'], alpha=0.12)
    sig = gauss(t, techo, 6, 0.8) * np.cos(0.7 * (t - techo)) * (t > t90)
    fill_band(ax, 0, sig, C['signal'], alpha=0.45)
    ax.axvline(techo, color=C['te'], ls='--', lw=1.4, alpha=0.8)
    ax.annotate('', (t90, -0.45), (techo, -0.45),
                arrowprops=dict(arrowstyle='<->', color=C['te'], lw=1.3))
    ax.text((t90 + techo) / 2, -0.5, 'TE', ha='center', fontsize=9, color=C['te'])
    ax.annotate('', (t90, -0.6), (96, -0.6),
                arrowprops=dict(arrowstyle='<->', color=C['tr'], lw=1.3))
    ax.text((t90 + 96) / 2, -0.65, 'TR(短)', ha='center', fontsize=9, color=C['tr'])

def draw_EPI(ax):
    t90 = 5
    n = 8
    t0, sp, flat, rise = 15, 10, 5, 1.0
    amp = 0.7
    fill_band(ax, 4, gauss(t, t90, 2, 0.9), C['rf'])
    ax.annotate('90°', (t90, 4.95), ha='center', fontsize=9, color=C['rf'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.9), zorder=10)
    gz = gauss(t, t90, 4, 0.7) - gauss(t, t90 + 5, 3, 0.35)
    fill_band(ax, 3, gz, C['slice'])
    for i in range(n - 1):
        c = t0 + (i + 0.5) * sp
        fill_band(ax, 2, trapz(t, c, 1.2, 0.5, 0.5), C['phase'])
    gx = np.zeros_like(t)
    for i in range(n):
        c = t0 + i * sp
        gx += (-1) ** i * trapz(t, c, flat, rise, amp)
    fill_band(ax, 1, gx, C['freq'])
    ax.axvspan(t0 - rise, t0 + (n - 1) * sp + rise, color=C['freq'], alpha=0.10)
    ax.annotate('ADC 连续采样', (t0 + n * sp / 2, 1.9), ha='center', fontsize=8, color=C['freq'],
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9), zorder=10)
    # 物理修正：EPI 是梯度回波链，所有回波同号（正值），振幅按 T2* 指数衰减
    sig = np.zeros_like(t)
    for i in range(n):
        c = t0 + i * sp
        t2star_decay = np.exp(-i * 0.12)
        sig += 0.6 * t2star_decay * gauss(t, c, 2.5, 1.0)
    sig *= (t > t90)
    fill_band(ax, 0, sig, C['signal'], alpha=0.45)
    # 注释放在信号 lobe 上方（y=0.85），加白底边框确保不与信号/Gx lobe 视觉冲突
    ax.text(50, 0.85, '一个 TR 内填完整幅 k-space', ha='center',
            fontsize=9, color=C['signal'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C['signal'], lw=0.6, alpha=0.92), zorder=10)

def draw_kspace(ax, kind):
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.set_xlabel('$k_x$', fontsize=10)
    ax.set_ylabel('$k_y$', fontsize=10)
    for v in np.linspace(-1, 1, 9):
        ax.axvline(v, color='#E0E0E0', lw=0.6)
        ax.axhline(v, color='#E0E0E0', lw=0.6)
    if kind in ('SE', 'GRE'):
        # 7 条线对称分布，中间一条在 ky=0，作为"当前读出行"
        ys = np.linspace(-0.8, 0.8, 7)
        # 用不同饱和度区分第 1 TR（亮）→ 第 7 TR（淡），可视化"顺序采集"过程
        for idx, yv in enumerate(ys):
            alpha = 0.95 - 0.1 * idx  # 0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35
            ax.plot([-1, 1], [yv, yv], color=C['freq'], lw=1.6, alpha=alpha)
            ax.annotate('', (-1, yv), (1, yv),
                        arrowprops=dict(arrowstyle='->', color=C['freq'], lw=1.2, alpha=alpha))
            # 在每条线的右端标注 TR 序号
            ax.text(1.05, yv, f'TR{idx+1}', ha='left', va='center',
                    fontsize=7, color=C['freq'], alpha=alpha)
        # "当前读出行"对齐中间那条线（ky=0，即第 4 条，idx=3）
        hy = ys[3]  # = 0.0
        ax.plot([-1, 1], [hy, hy], color=C['te'], lw=2.6)
        ax.annotate('当前读出行', (0, hy + 0.12), ha='center', fontsize=8, color=C['te'])
        # 起点 (绿) = 第 1 TR 左端；终点 (红) = 第 7 TR 右端
        ax.scatter(-1, ys[0], color='green', zorder=6, s=40, edgecolors='white', linewidths=0.5)
        # 文字放在圆点右侧，避免被 xlim 左侧裁切
        ax.text(-0.92, ys[0], '起点 (TR1)', ha='left', va='center',
                fontsize=7.5, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))
        ax.scatter(1, ys[-1], color='red', zorder=6, s=40, edgecolors='white', linewidths=0.5)
        ax.text(0.92, ys[-1], '终点 (TR7)', ha='right', va='center',
                fontsize=7.5, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))
        txt = '每次 TR 仅填 1 行\n（多激发 / 慢）' if kind == 'SE' \
              else '小角度 · 短 TR\n每次 TR 填 1 行'
        ax.text(0, -1.08, txt, ha='center', fontsize=8.5, color=C['slice'])
    else:
        Ny, Nx = 16, 60
        xs = np.linspace(-1, 1, Nx)
        path = []
        for i in range(Ny):
            yv = -1 + 2 * i / (Ny - 1)
            if i % 2 == 0:
                seg = list(zip(xs, [yv] * Nx))
            else:
                seg = list(zip(xs[::-1], [yv] * Nx))
            path += seg
        px, py = zip(*path)
        ax.plot(px, py, color=C['te'], lw=1.4)
        # 起点 (i=0 偶数行左端) = (-1, -1) = 左下；终点 (i=15 奇数行左端) = (-1, 1) = 左上
        # 两者都在 kx=-1 边（EPI 之字形特性），文字统一放在圆点右侧避免 xlim 裁切
        ax.scatter(px[0], py[0], color='green', zorder=6, s=45, edgecolors='white', linewidths=0.5)
        ax.text(px[0] + 0.08, py[0], '起点', ha='left', va='center',
                fontsize=8, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
        ax.scatter(px[-1], py[-1], color='red', zorder=6, s=45, edgecolors='white', linewidths=0.5)
        ax.text(px[-1] + 0.08, py[-1], '终点', ha='left', va='center',
                fontsize=8, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
        # 在前几行标注"之字形来回"的方向（放大箭头）
        for i in [0, 1, 2]:
            yv = -1 + 2 * i / (Ny - 1)
            direction = '→' if i % 2 == 0 else '←'
            ax.text(0, yv + (0.06 if i % 2 == 0 else -0.06), direction,
                    ha='center', va='center', fontsize=14, color=C['te'], fontweight='bold')
        ax.text(0, -1.08, '单次激发：一个 TR 填满整幅\n（之字形梯度回波链）',
                ha='center', fontsize=8.5, color=C['te'])

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, width_ratios=[3, 1.3],
                      hspace=0.35, wspace=0.18,
                      left=0.07, right=0.97, top=0.93, bottom=0.05)

rows = [('（a）自旋回波  Spin Echo (SE)', 'SE'),
        ('（b）梯度回波  Gradient Echo / FLASH', 'GRE'),
        ('（c）平面回波  EPI（单次激发）', 'EPI')]

for r, (title, kind) in enumerate(rows):
    ax = new_timing_ax(fig, r)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=8)
    if kind == 'SE':
        draw_SE(ax)
    elif kind == 'GRE':
        draw_GRE(ax)
    else:
        draw_EPI(ax)
    if r == 2:
        ax.set_xlabel('时间 $t$', fontsize=11)
    ax_ksp = fig.add_subplot(gs[r, 1])
    # 简化子图标题，避免与主标题重复
    ax_ksp.set_title(f'{kind} 采集', fontsize=10, fontweight='bold')
    draw_kspace(ax_ksp, kind)

fig.suptitle('MRI 脉冲序列与 k-space 填充轨迹对比\n'
             'Bloch 序列示意：RF 激发 → 梯度编码 → 信号采集',
             fontsize=15, fontweight='bold', y=0.975)

out = os.path.join(SAVE_DIR, 'Bloch序列与kspace轨迹.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'✓ 已保存: {out}')
