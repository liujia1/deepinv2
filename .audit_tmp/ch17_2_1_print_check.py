# -*- coding: utf-8 -*-
"""抽取 17.2-1.py L1720-1739 的打印段，用真实结果数值跑一遍，确认输出正确"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

psnr_n2n, psnr_n2nb, psnr_hfsup = 26.28744763520912, 19.648783962074166, 22.04675040510387
n2b_gap = psnr_n2n - psnr_n2nb
gap_repr = psnr_n2n - psnr_hfsup
gap_target = psnr_hfsup - psnr_n2nb

print(f"  3. N2B 比 N2N 低 {n2b_gap:.2f}dB，由'半分辨率监督对照'实测拆分：")
print(f"     · 表示/分辨率代价 PSNR(N2N)−PSNR(对照) = {gap_repr:+.2f} dB")
print(f"       （半分辨率2通道子图 vs 全分辨率，属结构性代价）")
print(f"     · 伪配对目标代价   PSNR(对照)−PSNR(N2B) = {gap_target:+.2f} dB")
print(f"       （目标用互补位置的带噪像素，而非同位置的干净像素）")
if gap_repr > gap_target + 0.5:
    main_cause = "半分辨率表示(i)"
elif gap_target > gap_repr + 0.5:
    main_cause = "伪配对目标位置错位(ii)：目标取相邻像素而非同位置干净像素"
else:
    main_cause = "(i)与(ii)相当"
print(f"     → 主导因素：{main_cause}")
print(f'     注：该拆分是"先表示、后目标"的单路径可加分解，两项之和恒等于总差距，')
print(f"         但未分离二者的交互项（例如分辨率代价在目标带噪时可能略有不同）。")
print(f"     注（伪配对目标为何有偏）：噪声独立零均值只给出 E[y₂|y₁]=E[x₂|y₁]，")
print(f"         而该位置本应给出 x₁，且 E[x₂|y₁]≠x₁——所以伪配对目标对 x **有偏**。")
print(f"         相邻像素的信号相关性正是该偏差的控制因素（线性近似下 E[x₂|y₁] 相当于")
print(f"         把 E[x₁|y₁] 按相关系数ρ收缩）：ρ↑→偏差↓；ρ→0 时网络无法从 y₁ 推断 x₂，")
print(f"         只能输出局部统计均值。（本实验数据实测相邻ρ≈0.87，偏差受控，N2B 才可用；")
print(f"         这也是它仍不及同位置 N2N 的根因。）")
print()
print(f"[自检] gap_repr+gap_target = {gap_repr + gap_target:.2f} dB == n2b_gap = {n2b_gap:.2f} dB")
