# -*- coding: utf-8 -*-
"""验证附录17C 修改后的两条论断：
   C节：E[TᵀAᵀAT] = E[TᵀAᵀTA] + E[TᵀAᵀD]，且非等变时 TᵀAᵀTA ≠ AᵀA
        （等变算例 TᵀAᵀTA = AᵀA）
   D节：固定掩码 MRI 关于平移严格等变：A = M F，A T = T_freq A；
        等价地图像域算子 A_img = F⁻¹ M F 满足 A_img T = T A_img，与掩码形状无关
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

rng = np.random.default_rng(0)
N = 8
n = N * N
F1 = np.fft.fft(np.eye(N)) / np.sqrt(N)      # unitary 1D DFT
F = np.kron(F1, F1)                           # 2D DFT
Fh = F.conj().T
Ty = np.roll(np.eye(N), 1, axis=0)
Tx = np.roll(np.eye(N), 1, axis=1)
T = np.kron(Ty, Tx)                           # 循环平移（正交）
assert np.allclose(T.T @ T, np.eye(n))

print('=== C 节：代数恒等式 ===')
# 附录中的写法 A T_g = T_g A 隐含 A 为 n×n（inpainting 的掩码、MRI 的 F⁻¹MF 都是）
A = rng.standard_normal((n, n)) / np.sqrt(n)  # 随机非等变算子
D = A @ T - T @ A
lhs = T.T @ A.T @ A @ T
rhs = T.T @ A.T @ T @ A + T.T @ A.T @ D
print(f'  |lhs - rhs|_max                = {np.abs(lhs - rhs).max():.2e}  (应为0)')
first = T.T @ A.T @ T @ A
print(f'  |TᵀAᵀTA - AᵀA|_max            = {np.abs(first - A.T @ A).max():.2e}  '
      f'(非等变时应显著>0)')

# 等变算例：循环卷积 A = Fh K F（K 对角），与所有平移可交换
K = np.diag(rng.standard_normal(n) * (rng.random(n) < 0.5))
A_eq = Fh @ K @ F
D_eq = A_eq @ T - T @ A_eq
first_eq = T.T @ A_eq.T @ T @ A_eq
print(f'  等变算例: |D|_max              = {np.abs(D_eq).max():.2e}  (应为0)')
print(f'  等变算例: |TᵀAᵀTA - AᵀA|_max   = {np.abs(first_eq - A_eq.T @ A_eq).max():.2e}  (应为0)')

print('\n=== D 节：MRI 严格平移等变 ===')
Tfreq = F @ T @ Fh
print(f'  |M_freq - diag|_max            = '
      f'{np.abs(Tfreq - np.diag(np.diag(Tfreq))).max():.2e}  (相位调制应为对角阵)')
for trial in range(5):
    mask = (rng.random(n) < 0.5).astype(float)
    if trial == 0:
        mask[:] = 0; mask[::7] = 1.0        # 结构化稀疏掩码
    if trial == 1:
        mask[:] = 1; mask[::3] = 0.0        # 另一种图案
    Ms = np.diag(mask)
    A_fwd = Ms @ F                            # 图像 → k 空间
    err_fwd = np.abs(A_fwd @ T - Tfreq @ A_fwd).max()
    A_img = Fh @ Ms @ F                       # 图像 → 图像（17.6-1 的 mri_A）
    err_img = np.abs(A_img @ T - T @ A_img).max()
    print(f'  mask#{trial} 保留{int(mask.sum()):3d}/{n}: '
          f'|MF T - T_freq MF|_max = {err_fwd:.2e}, '
          f'|A_img T - T A_img|_max = {err_img:.2e}')
print('  => 与掩码形状无关，严格等变（数值零）')

# inpainting 对照：图像域像素掩码一般不交换
Mpix = np.diag((rng.random(n) < 0.5).astype(float))
print(f'\n  对照 inpainting: |M_pix T - T M_pix|_max = '
      f'{np.abs(Mpix @ T - T @ Mpix).max():.2e}  (显著>0 → 不等变，EI可用)')
