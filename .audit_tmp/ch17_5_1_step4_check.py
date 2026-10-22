# -*- coding: utf-8 -*-
"""复现 17.5-1 Step 4 的覆盖率/秩统计（不训练），验证：
   (1) 经验覆盖 ≈ 1-(1-p)^G，rank_ratio 与覆盖率完全相等（对角0/1下同一量）
   (2) 但 G=1 时实例 rank(Σ_g AᵀA)/n 只有 50%，
       而期望矩阵 rank(E[AᵀA]) = n = 1024（满秩）——两者不是同一个量，
       所以把 Step 4 叫"满秩条件验证"是口径错位。
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch

H = W = 32
n = H * W
p = 0.5
n_keep = int(n * p)


def make_masks(B, G):
    scores = torch.rand(B, G, n)
    idx = scores.argsort(dim=-1)[:, :, -n_keep:]
    m = torch.zeros(B, G, n)
    m.scatter_(-1, idx, 1.0)
    return m


print(f'{"G":>3} {"经验覆盖":>9} {"理论覆盖":>9} {"经验rank":>9} {"全覆比例":>9} {"理论p^G":>8}')
T = 200
for G in [1, 2, 3, 4, 6, 8, 12, 16]:
    cov = rank = allc = 0.0
    for _ in range(T):
        m = make_masks(1, G)[0]          # (G, n)
        s = m.sum(0)                     # 每个像素被几个算子覆盖
        cov += (s > 0).float().mean().item()
        rank += (s > 0).float().mean().item()   # rank(Σ_g A_gᵀA_g)/n，对角下即非零元比例
        allc += (s == G).float().mean().item()
    print(f'{G:>3} {cov/T:>9.4f} {1-(1-p)**G:>9.4f} {rank/T:>9.4f} {allc/T:>9.4f} {p**G:>8.4f}')

# 期望矩阵：E[AᵀA] = diag(E[m]) = pI（解析成立）
# 注意：不能把掩码矩阵 M 的 MᵀM 当成 AᵀA——那是掩码向量的外积和（非对角≈p²）。
# A 是“选择矩阵”（每行一个1），AᵀA 严格为对角阵 diag(m)。
m1 = make_masks(1, 1)[0, 0]              # (n,)
idx = (m1 > 0).nonzero().squeeze(1)
A = torch.zeros(len(idx), n)
A[torch.arange(len(idx)), idx] = 1.0
A2 = A.T @ A
print(f'\n单掩码：AᵀA 非对角 max = {(A2 - torch.diag(A2.diag())).abs().max():.2e}（理论=0）')
print(f'        AᵀA 对角 == 掩码 m ? {torch.equal(A2.diag(), m1)}')
Mm = make_masks(2000, 1)[:, 0].mean(0)
print(f'        E[m] 实测 min/max = {Mm.min():.4f} / {Mm.max():.4f}（理论=p={p}）')
print(f'解析：E[AᵀA] = diag(p) = pI（精确），rank = n = {n}  -> G=1 时期望意义下已满秩')
print(f'而 G=1 的实例 rank(Σ_g A_gᵀA)/n = {p:.2f}（覆盖率）')
print('=> "期望满秩条件"与"单实例联合覆盖度"是两个不同的量。')
