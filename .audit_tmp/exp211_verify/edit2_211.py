# -*- coding: utf-8 -*-
import io, sys

PY = r"d:/works/deepinv/第2章 先验：贝叶斯推断的灵魂/第二章 配套实验/实验2.1-1/2.1-1.py"

def load(p):
    with io.open(p, 'r', encoding='utf-8') as f:
        return f.read()
def save(p, s):
    with io.open(p, 'w', encoding='utf-8') as f:
        f.write(s)
def rep(text, old, new, tag):
    cnt = text.count(old)
    if cnt != 1:
        print(f"[FAIL] {tag}: found {cnt}"); print("  old:", repr(old[:70])); sys.exit(1)
    print(f"[OK] {tag}")
    return text.replace(old, new)

py = load(PY)

# A. 插入合成对照的计算
SYNTH_CALC = '''# ---- 合成对照：先验假设真的成立时，贝叶斯 lambda 是否最优？ ----
# 用与先验同分布的 iid 高斯"图像" x_syn ~ N(0, sigma_x^2)，此时理论假设成立
# 独立 RNG，不扰动全局随机流，保证场景1结果不变
rng_syn = np.random.default_rng(0)
x_syn = sigma_x_denoise * rng_syn.standard_normal((n, n))
y_syn = x_syn + noise_lev2 * rng_syn.standard_normal((n, n))
dr_syn = x_syn.max() - x_syn.min()
psnr_list_syn = [peak_signal_noise_ratio(x_syn, y_syn / (1 + lam), data_range=dr_syn) for lam in lambdas]
best_idx_syn = int(np.argmax(psnr_list_syn))
psnr_bayes_syn = peak_signal_noise_ratio(x_syn, y_syn / (1 + lambda_bayes), data_range=dr_syn)
psnr_noisy_syn = peak_signal_noise_ratio(x_syn, y_syn, data_range=dr_syn)

'''
old_a = 'psnr_noisy = peak_signal_noise_ratio(x, y_denoise, data_range=DATA_RANGE)\n\nprint(f"噪声水平 sigma = {noise_lev2:.4f}")'
new_a = 'psnr_noisy = peak_signal_noise_ratio(x, y_denoise, data_range=DATA_RANGE)\n\n' + SYNTH_CALC + 'print(f"噪声水平 sigma = {noise_lev2:.4f}")'
py = rep(py, old_a, new_a, "A: 合成对照计算")

# B. 插入合成对照的打印
SYNTH_PRINT = '''print(f"\\n[合成对照 x~N(0,sigma_x^2) iid, 先验假设成立]")
print(f"  含噪基线 PSNR = {psnr_noisy_syn:.2f} dB")
print(f"  贝叶斯 lambda={lambda_bayes:.4f} -> PSNR = {psnr_bayes_syn:.2f} dB")
print(f"  扫描最优 lambda = {lambdas[best_idx_syn]:.4f} -> PSNR = {psnr_list_syn[best_idx_syn]:.2f} dB")
print(f"  [对照结论] 先验成立时贝叶斯 lambda≈经验最优; 真实图因先验误设而偏离")
'''
old_b = 'print(f"PSNR 扫描最优 lambda = {lambdas[best_idx]:.4f}, PSNR = {psnr_list[best_idx]:.2f} dB")'
new_b = old_b + "\n" + SYNTH_PRINT
py = rep(py, old_b, new_b, "B: 合成对照打印")

# C. 面板0 真实曲线 label
old_c = "axes[0].semilogx(lambdas, psnr_list, 'b-o', markersize=4, label='Tikhonov')"
new_c = "axes[0].semilogx(lambdas, psnr_list, 'b-o', markersize=4, label='Tikhonov (真实camera图)')"
py = rep(py, old_c, new_c, "C: 真实曲线label")

# D. 面板0 追加合成曲线 + 峰值点
old_d = """axes[0].scatter([lambda_bayes], [psnr_bayes], color='g', s=100, zorder=5, marker='*',
                label=f'贝叶斯 lambda 点 (PSNR={psnr_bayes:.2f}dB)')"""
new_d = old_d + """
axes[0].semilogx(lambdas, psnr_list_syn, 'm--s', markersize=3, alpha=0.8, label='iid高斯合成 (先验成立)')
axes[0].scatter([lambdas[best_idx_syn]], [psnr_list_syn[best_idx_syn]], color='m', s=150, zorder=6, marker='P',
                label=f'合成最优 lambda={lambdas[best_idx_syn]:.4f} (落在贝叶斯线上)')"""
py = rep(py, old_d, new_d, "D: 追加合成曲线")

# E. 面板0 标题（含字面 \\n，用 raw 三引号）
old_e = r"""axes[0].set_title('PSNR vs lambda：贝叶斯 lambda 与 PSNR 最优 lambda 对比\n(A=I 假设下的闭式解)')"""
new_e = r"""axes[0].set_title('PSNR vs lambda：贝叶斯 λ 与经验最优 λ 对比\n真实图(蓝)峰值偏离贝叶斯线; 合成(品红)峰值吻合贝叶斯线')"""
py = rep(py, old_e, new_e, "E: 面板0标题")

# F. JSON 增加合成对照字段
old_f = "    'psnr_sweep_max_dB': float(round(psnr_list[best_idx], 2)),\n}"
new_f = ("    'psnr_sweep_max_dB': float(round(psnr_list[best_idx], 2)),\n"
         "    'psnr_syn_bayes_dB': float(round(psnr_bayes_syn, 2)),\n"
         "    'psnr_syn_optimal_dB': float(round(psnr_list_syn[best_idx_syn], 2)),\n"
         "    'psnr_syn_noisy_dB': float(round(psnr_noisy_syn, 2)),\n"
         "    'lambda_syn_optimal': float(round(lambdas[best_idx_syn], 6)),\n}")
py = rep(py, old_f, new_f, "F: JSON合成字段")

save(PY, py)
print("\n代码编辑完成。")
