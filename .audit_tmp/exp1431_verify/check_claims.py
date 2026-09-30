# -*- coding: utf-8 -*-
"""
针对实验14.3-1若干"免责/论断"的快速独立检验（迭代次数与误差均为确定性量，不受CPU负载影响）
  T1: 固定tol=1e-6下收敛迭代次数随n的增长(封顶10000); 同时记录 n²预算公式基准
      (ε=0.01:2000次, ε=0.1:500次) 处的代价误差 -> 检验"固定预算大n不收敛"与"预算随n²缩放"是否必要
  T2: n=500/1000 时 Sinkhorn 峰值内存 -> 检验"500x500内存紧张"
  T3: 步骤3扫描 ε<0.01 预算5000 是否截断 -> 检验图3(b)"平台非截断产物"与讲义"需10000+次"
  T4: n=50, ε=0.01 代价轨迹 -> 检验"约2000次进入稳态"
"""
import sys, io, time, ctypes
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment

def sample_gm(n, n_modes=4, radius=2.0, std=0.2, seed=None):
    rng = np.random.RandomState(42) if seed is None else np.random.RandomState(seed)
    angles = np.linspace(0, 2*np.pi, n_modes, endpoint=False)
    centers = np.stack([radius*np.cos(angles), radius*np.sin(angles)], axis=1)
    return centers[rng.randint(0, n_modes, n)] + std*rng.randn(n, 2), centers

def cost_matrix(s, t):
    return np.sum((s[:, None, :] - t[None, :, :])**2, axis=-1)

def sinkhorn(C, reg, max_iter, tol=1e-6, track=None):
    """返回 (最终代价, 实际迭代数, 是否收敛, [(it,cost)...])"""
    n, m = C.shape
    log_K = -C / reg
    la, lb = -np.log(n)*np.ones(n), -np.log(m)*np.ones(m)
    lu, lv = np.zeros(n), np.zeros(m)
    traj = []
    converged = False
    it = 0
    for it in range(1, max_iter+1):
        lu0, lv0 = lu.copy(), lv.copy()
        lu = la - logsumexp(log_K + lv[None, :], axis=1)
        lv = lb - logsumexp(log_K + lu[:, None], axis=0)
        if track and it % track == 0:
            lp = lu[:, None] + lv[None, :] + log_K
            P = np.exp(lp - lp.max()); P /= P.sum()
            traj.append((it, float(np.sum(C*P))))
        if max(np.max(np.abs(lu-lu0)), np.max(np.abs(lv-lv0))) < tol:
            converged = True
            break
    lp = lu[:, None] + lv[None, :] + log_K
    P = np.exp(lp - lp.max()); P /= P.sum()
    return float(np.sum(C*P)), it, converged, traj

class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_size_t), ("PageFaultCount", ctypes.c_size_t),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t)]
def peak_rss_mb():
    h = ctypes.windll.kernel32.GetCurrentProcess()
    p = PMC(); p.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(p), p.cb)
    return p.PeakWorkingSetSize/1e6

print("="*72); print("T1/T4: 收敛迭代数 vs n (tol=1e-6, 封顶10000) + 固定基准预算处的误差"); print("="*72)
print("说明: 讲义n²预算公式取 ε=0.01:2000·(n/50)², ε=0.1:500·(n/50)²; 此处对比'不缩放'是否也可行")
for n in [50, 100, 200, 300]:
    src, _ = sample_gm(n, seed=42); tgt, _ = sample_gm(n, seed=123)
    C = cost_matrix(src, tgt)
    r, c_ = linear_sum_assignment(C); ch = C[r, c_].mean()
    for reg, base in [(0.01, 2000), (0.1, 500)]:
        cf, it, conv, traj = sinkhorn(C, reg, max_iter=10000, track=100)
        # 固定基准预算(不随n²缩放)处的代价
        at_base = min(traj, key=lambda x: abs(x[0]-base)) if traj else (base, float('nan'))
        err_f = abs(cf-ch)/ch*100
        err_b = abs(at_base[1]-ch)/ch*100
        print(f"  n={n:>4} eps={reg}: 收敛it={it:>6} {'[收敛]' if conv else '[封顶未收敛]'} 最终误差={err_f:5.2f}% | "
              f"固定预算{base}次处误差={err_b:5.2f}% | n²缩放预算={int(base*(n/50)**2)}次")
        if n == 50 and reg == 0.01:
            print("    T4 代价轨迹(n=50,eps=0.01): " + ", ".join(f"{i}:{v:.4f}" for i, v in traj[::5]))

print(); print("="*72); print("T2: n=500/1000 内存占用 (float64, 含代价矩阵+迭代临时量)"); print("="*72)
for n in [500, 1000]:
    src, _ = sample_gm(n, seed=42); tgt, _ = sample_gm(n, seed=123)
    base_rss = peak_rss_mb()
    C = cost_matrix(src, tgt)
    cf, it, conv, _ = sinkhorn(C, 0.1, max_iter=2000)
    print(f"  n={n:>5}: 2000次迭代峰值RSS增量 = {peak_rss_mb()-base_rss:8.1f} MB "
          f"(理论: C与log_K各 {n*n*8/1e6:.1f} MB)")

print(); print("="*72); print("T3: 步骤3扫描 n=100, 预算5000(讲义) vs 50000, ε<0.01"); print("="*72)
n = 100
src, _ = sample_gm(n, seed=42); tgt, _ = sample_gm(n, seed=123)
C = cost_matrix(src, tgt)
r, c_ = linear_sum_assignment(C); ch = C[r, c_].mean()
for reg in [0.001, 0.0046, 0.0076]:
    c1, it1, cv1, _ = sinkhorn(C, reg, max_iter=5000)
    c2, it2, cv2, _ = sinkhorn(C, reg, max_iter=50000)
    print(f"  eps={reg:<7}: 预算5000 -> it={it1:>5}{'[截断]' if not cv1 else '[收敛]'} err={abs(c1-ch)/ch*100:6.1f}% | "
          f"预算50000 -> it={it2:>5}{'[截断]' if not cv2 else '[收敛]'} err={abs(c2-ch)/ch*100:6.1f}%")
print("完成")
