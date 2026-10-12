"""
复核脚本 3.4-1：
1) 复现 results_summary.json 的四个非扩张性 ratio（seed=42）
2) 核对四个近端算子闭式定义是否正确（对 prox 定义做数值 argmin 验证）
3) 构造显式反例，裁决"硬阈值不保证非扩张"是固有属性还是规避托辞
4) 多 seed 检验硬阈值经验 ratio 是否会 >1（"✅"是否稳健）
"""
import numpy as np

lam = 1.0

def soft_threshold(v, lam):
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)

def hard_threshold(v, lam):
    tau = np.sqrt(2 * lam)
    return v * (np.abs(v) >= tau).astype(float)

def shrinkage(v, lam):
    return v / (1 + lam)

def projection_nonneg(v):
    return np.maximum(v, 0)

# ---------- 1) 复现 JSON ratio ----------
np.random.seed(42)
n_test = 10000
v1 = np.random.randn(n_test)
v2 = np.random.randn(n_test)
dist_v = np.linalg.norm(v1 - v2)
fns = {
    "l2_shrink": lambda v: shrinkage(v, lam),
    "l1_soft": lambda v: soft_threshold(v, lam),
    "l0_hard": lambda v: hard_threshold(v, lam),
    "proj": lambda v: projection_nonneg(v),
}
print("=== 1) 复现非扩张性 ratio (seed=42) ===")
json_expect = {"l2_shrink": 0.5, "l1_soft": 0.391706, "l0_hard": 0.760203, "proj": 0.58598}
for name, fn in fns.items():
    r = np.linalg.norm(fn(v1) - fn(v2)) / dist_v
    print(f"  {name}: ratio={r:.6f}  (JSON={json_expect[name]})  match={abs(r-json_expect[name])<1e-5}")

# ---------- 2) 用数值 argmin 验证闭式 prox 定义 ----------
# prox_{lam*g}(v) = argmin_x 0.5*(x-v)^2 + lam*g(x)
print("\n=== 2) 闭式 prox 定义数值验证 (在一维网格上 argmin) ===")
def numeric_prox(v_scalar, g, lam, lo=-8, hi=8, n=200001):
    xs = np.linspace(lo, hi, n)
    obj = 0.5 * (xs - v_scalar) ** 2 + lam * g(xs)
    return xs[np.argmin(obj)]

tests = [
    ("l2^2 g=0.5x^2", lambda x: 0.5 * x ** 2, shrinkage, 2.3),
    ("l1  g=|x|",     lambda x: np.abs(x),     soft_threshold, 2.3),
    ("l1  g=|x|",     lambda x: np.abs(x),     soft_threshold, 0.4),
    ("l0  g=|x|!=0",  lambda x: (x != 0).astype(float), hard_threshold, 2.3),
    ("l0  g=|x|!=0",  lambda x: (x != 0).astype(float), hard_threshold, 1.0),
]
for gname, g, closed, vv in tests:
    num = numeric_prox(vv, g, lam)
    clo = closed(np.array([vv]), lam)[0]
    print(f"  {gname:16s} v={vv:5.2f}  numeric={num:8.4f}  closed={clo:8.4f}  ok={abs(num-clo)<0.02}")

# ---------- 3) 显式反例：硬阈值违反非扩张 ----------
print("\n=== 3) 硬阈值非扩张性反例 (标量, 跨越阈值 tau=sqrt(2)) ===")
tau = np.sqrt(2 * lam)
for eps in [1e-1, 1e-2, 1e-3, 1e-6]:
    a = tau + eps   # 保留原值
    b = tau - eps   # 归零
    ratio = abs(hard_threshold(np.array([a]), lam)[0] - hard_threshold(np.array([b]), lam)[0]) / abs(a - b)
    print(f"  eps={eps:8.1e}  |H(a)-H(b)|/|a-b| = {ratio:12.3f}  (>1 违反非扩张: {ratio>1})")

# ---------- 4) 多 seed 检验硬阈值经验 ratio 是否会 >1 ----------
print("\n=== 4) 多 seed 下高维随机向量对硬阈值 ratio (看是否稳健<=1) ===")
viol = 0
vals = []
for seed in range(200):
    np.random.seed(seed)
    a = np.random.randn(n_test)
    b = np.random.randn(n_test)
    r = np.linalg.norm(hard_threshold(a, lam) - hard_threshold(b, lam)) / np.linalg.norm(a - b)
    vals.append(r)
    if r > 1.0:
        viol += 1
vals = np.array(vals)
print(f"  200 seeds: min={vals.min():.4f} max={vals.max():.4f} mean={vals.mean():.4f}")
print(f"  ratio>1 的次数: {viol}/200")

# 附加：把维度降低, 看硬阈值是否更容易 >1
print("\n  降维检验 (每 seed 用较少分量, 更容易跨阈值):")
for dim in [10000, 100, 10, 2]:
    viol_d = 0
    for seed in range(200):
        np.random.seed(seed)
        a = np.random.randn(dim)
        b = np.random.randn(dim)
        r = np.linalg.norm(hard_threshold(a, lam) - hard_threshold(b, lam)) / np.linalg.norm(a - b)
        if r > 1.0:
            viol_d += 1
    print(f"    dim={dim:6d}: ratio>1 次数 {viol_d}/200")
