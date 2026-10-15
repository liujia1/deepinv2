# -*- coding: utf-8 -*-
import io, re, py_compile, sys

p = r"d:\works\deepinv\第2章 先验：贝叶斯推断的灵魂\第二章 配套实验\实验2.2-3\2.2-3.py"
txt = io.open(p, encoding="utf-8").read()
orig = txt

# ---- Edit A: 移除图1中把锐边误标为"阶梯效应"的箭头与文字 ----
txt, nA = re.subn(r"axes\[3\]\.annotate\(.*?'阶梯效应'.*?ha='center'\)\n", "", txt, flags=re.DOTALL)

# ---- Edit B: 用"合成斜坡+阶跃"重构步骤2图（三面板：真实剖面 + 合成全信号 + 渐变区放大）----
start_marker = "stair_x, stair_y = staircase_pos"
end_marker = ("plt.savefig(os.path.join(SAVE_DIR, '步骤2_阶梯效应分析.png'), "
              "dpi=150, bbox_inches='tight')\nplt.close()\n")
i = txt.index(start_marker)
j = txt.index(end_marker) + len(end_marker)

new_B = r'''# ===== 合成一维信号：无歧义地展示阶梯效应 =====
def sobolev_1d(y, lam):
    """一维 Sobolev 去噪闭式解（周期边界）：X = Y / (1 - lam*2(cos f - 1))"""
    m = len(y)
    f = np.fft.fftfreq(m) * 2 * np.pi
    lap = 2 * (np.cos(f) - 1)
    return np.real(np.fft.ifft(np.fft.fft(y) / (1 - lam * lap)))

m = 128
tt = np.arange(m)
x_true = np.zeros(m)
x_true[:32] = 0.20
x_true[32:80] = 0.20 + 0.55 * (tt[32:80] - 32) / (80 - 32)   # 线性渐变区(ramp)
x_true[80:96] = 0.75
x_true[96:] = 0.30                                            # 阶跃边缘(step)
rng_synth = np.random.default_rng(0)
x_noisy = x_true + rng_synth.normal(0, sigma, m)

lam_synth = 3.0
x_sob = sobolev_1d(x_noisy, lam_synth)
x_tv = denoise_tv_chambolle(x_noisy, weight=alpha_tv)

ramp = slice(32, 80)
step = slice(88, 104)
def _mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))
step_mse_tv = _mse(x_tv[step], x_true[step])
step_mse_sob = _mse(x_sob[step], x_true[step])
# 阶梯效应客观度量：真实斜坡应有恒定非零斜率，TV 却把大部分斜率压成零(平台)
_tv_d = np.diff(x_tv[ramp]); _sob_d = np.diff(x_sob[ramp])
ramp_flat_frac_tv = float(np.mean(np.abs(_tv_d) < 0.005))
ramp_flat_frac_sob = float(np.mean(np.abs(_sob_d) < 0.005))

print("\n【合成信号：阶梯效应定量验证】")
print(f"  阶跃边缘 MSE: TV={step_mse_tv:.5f}  Sobolev={step_mse_sob:.5f}  (TV 保边更优)")
print(f"  渐变区平台占比(近零梯度): TV={ramp_flat_frac_tv*100:.1f}%  Sobolev={ramp_flat_frac_sob*100:.1f}%  (TV 阶梯效应)")

# ===== 步骤2图：真实剖面 + 合成信号阶梯效应 =====
stair_x, stair_y = staircase_pos
row = stair_y

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(camera[row, :], color=COLOR_ORIGINAL, linewidth=2, label='真实信号')
axes[0].plot(camera_noisy[row, :], color=COLOR_NOISY, linewidth=0.5, alpha=0.5, label='含噪')
axes[0].plot(camera_sob[row, :], color=COLOR_SOBOLEV, linewidth=1.5, label='Sobolev (边缘模糊)')
axes[0].plot(camera_tv[row, :], color=COLOR_TV, linewidth=1.5, label='TV (边缘锐利)')
axes[0].axvline(x=stair_x, color='red', linestyle='--', linewidth=1, alpha=0.5, label='检测位置')
axes[0].set_title(f'真实图像第{row}行剖面\n(Sobolev 磨边 vs TV 保边)')
axes[0].legend(fontsize=8)
axes[0].set_xlabel('像素索引')
axes[0].set_ylabel('灰度值')
axes[0].grid(True, alpha=0.3)

axes[1].plot(tt, x_true, color=COLOR_ORIGINAL, linewidth=2, label='真实信号')
axes[1].plot(tt, x_noisy, color=COLOR_NOISY, linewidth=0.5, alpha=0.5, label='含噪')
axes[1].plot(tt, x_sob, color=COLOR_SOBOLEV, linewidth=1.5, label=f'Sobolev (λ={lam_synth})')
axes[1].plot(tt, x_tv, color=COLOR_TV, linewidth=1.5, label=f'TV (α={alpha_tv})')
axes[1].axvspan(32, 80, color='gold', alpha=0.15)
axes[1].text(56, 0.90, '渐变区', ha='center', fontsize=9, color='darkgoldenrod')
axes[1].axvline(x=96, color='green', linestyle=':', linewidth=1.2)
axes[1].text(96, 0.55, ' 阶跃', color='green', fontsize=9)
axes[1].set_title('合成斜坡+阶跃：全信号对比\n(阶跃处 TV 更陡、Sobolev 被磨平)')
axes[1].legend(fontsize=8)
axes[1].set_xlabel('像素索引')
axes[1].set_ylabel('信号值')
axes[1].grid(True, alpha=0.3)

zr = slice(30, 82)
axes[2].plot(tt[zr], x_true[zr], color=COLOR_ORIGINAL, linewidth=2, label='真实(线性斜坡)')
axes[2].plot(tt[zr], x_sob[zr], color=COLOR_SOBOLEV, linewidth=1.5, label='Sobolev(平滑跟随)')
axes[2].plot(tt[zr], x_tv[zr], color=COLOR_TV, linewidth=1.5, label='TV(阶梯效应)')
axes[2].set_title(f'渐变区放大：TV 阶梯效应\n(TV平台占比{ramp_flat_frac_tv*100:.0f}% vs Sobolev {ramp_flat_frac_sob*100:.0f}%)')
axes[2].legend(fontsize=8)
axes[2].set_xlabel('像素索引')
axes[2].set_ylabel('信号值')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_阶梯效应分析.png'), dpi=150, bbox_inches='tight')
plt.close()
'''
txt = txt[:i] + new_B + txt[j:]

# ---- Edit C: 合成指标写入 results_summary ----
old_c = "    'sobolev_zero_gradient_pct': float(round(sob_zero_grad, 2)),\n}"
new_c = ("    'sobolev_zero_gradient_pct': float(round(sob_zero_grad, 2)),\n"
         "    'synth_sobolev_lambda': float(lam_synth),\n"
         "    'synth_step_mse_tv': float(round(step_mse_tv, 5)),\n"
         "    'synth_step_mse_sobolev': float(round(step_mse_sob, 5)),\n"
         "    'synth_ramp_flat_frac_tv': float(round(ramp_flat_frac_tv, 3)),\n"
         "    'synth_ramp_flat_frac_sobolev': float(round(ramp_flat_frac_sob, 3)),\n"
         "}")
assert txt.count(old_c) == 1, "EditC anchor not unique"
txt = txt.replace(old_c, new_c)

io.open(p, "w", encoding="utf-8").write(txt)
print("EditA arrow removed:", nA)
# syntax check
py_compile.compile(p, doraise=True)
print("py_compile OK; file bytes", len(txt), "(was", len(orig), ")")
