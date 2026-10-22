"""审计用: 复用 15.1-1.py 的类定义, 复现步骤2(显存)与步骤3(梯度流)的测量。

目的:
  1) 验证 f03 修复(先记基线再取峰值增量)后 Full/Linear 比值;
  2) 验证 md 中 1.5x / 6e6x / 3.5e-3 / 5.4e-3 / 4.1e-8 这些数字是否真实可复现;
  3) 产出 attn_memory_MB / attn_memory_analytic_MB / grad_flow_by_layer 补入 results_summary.json。
"""
import ast
import json
import math
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

SRC = pathlib.Path(
    r"h:\deepinv\deepinv2\第15章 扩散模型的架构实践：UNet → DiT"
    r"\第十五章 配套实验\实验15.1-1\15.1-1.py"
)
tree = ast.parse(SRC.read_text(encoding="utf-8"))

WANT = {"Attention", "LinearAttention", "PreNorm", "_PreNormBlock", "_PostNormBlock"}
nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in WANT]
print("extracted classes:", [n.name for n in nodes])
assert len(nodes) == len(WANT), "类提取不完整"

ns = {"torch": torch, "nn": nn, "F": F, "math": math, "rearrange": rearrange}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<extract>", "exec"), ns)
Attention = ns["Attention"]
LinearAttention = ns["LinearAttention"]
_PreNormBlock = ns["_PreNormBlock"]
_PostNormBlock = ns["_PostNormBlock"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")


def measure(mod, x_s, legacy):
    mod.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    y = mod(x_s)
    y.sum().backward()
    peak = torch.cuda.max_memory_allocated()
    return (peak if legacy else peak - base) / 1024 ** 2


# ---------------- 步骤2: 显存 ----------------
print("\n=== 步骤2 显存 (MB) ===")
attn_full = Attention(64).to(device)
attn_linear = LinearAttention(64).to(device)

rows, ana_rows = {}, {}
for size in [16, 32, 64]:
    x_s = torch.randn(1, 64, size, size, device=device)
    N = size * size
    row, ana = {}, {}
    for tag, mod in [("full", attn_full), ("linear", attn_linear)]:
        n_bytes = (4 * 4 * N * N) if tag == "full" else (4 * N * 128)
        ana[tag] = round(n_bytes / 1024 ** 2, 4)
        legacy = measure(mod, x_s, legacy=True)
        fixed = measure(mod, x_s, legacy=False)
        row[tag] = round(fixed, 3)
        print(
            f"  {size}x{size} {tag:6s}: legacy(含基线)={legacy:8.3f}  "
            f"fixed(增量)={fixed:8.3f}  analytic={ana[tag]:8.4f}"
        )
    r = row["full"] / row["linear"] if row["linear"] else float("inf")
    ra = ana["full"] / ana["linear"]
    row["unit"] = "MB"
    row["measured_on_gpu"] = True
    ana["unit"] = "MB"
    ana["note"] = "仅计单份主导中间量, 不含反传额外保存; 与实测口径不同"
    rows[str(size)] = row
    ana_rows[str(size)] = ana
    print(f"  {size}x{size} 实测比={r:.2f}x  解析比={ra:.2f}x")

# ---------------- 步骤3: 梯度流 ----------------
print("\n=== 步骤3 梯度流 (12 层, 固定CPU·float32) ===")
DEPTH_GF = 12
grad_flow = {}
gf_device = torch.device("cpu")
for tag, cls in [("PreNorm", _PreNormBlock), ("PostNorm", _PostNormBlock)]:
    torch.manual_seed(0)
    layers = nn.ModuleList([cls(64) for _ in range(DEPTH_GF)]).to(gf_device)
    h = torch.randn(2, 64, 16, 16, device=gf_device)
    for l in layers:
        h = l(h)
    h.pow(2).mean().backward()
    norms = [float(l.norm.weight.grad.norm()) for l in layers]
    grad_flow[tag] = norms
    print(
        f"  {tag}: 输入侧(第1层)={norms[0]:.3e}  输出侧(第12层)={norms[-1]:.3e}  "
        f"输出/输入={norms[-1] / max(norms[0], 1e-12):.3f}"
    )

gf = {k: [float(f"{x:.6e}") for x in v] for k, v in grad_flow.items()}
out = {"attn_memory_MB": rows, "attn_memory_analytic_MB": ana_rows, "grad_flow_by_layer": gf}
dst = pathlib.Path(__file__).with_name("exp151_measure_out.json")
dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved:", dst)

SUMMARY = SRC.parent / "results_summary.json"
old = json.loads(SUMMARY.read_text(encoding="utf-8"))
merged = {
    "total_params": old["total_params"],
    "final_loss": old["final_loss"],
    "num_epochs": old["num_epochs"],
    "attn_memory_MB": rows,
    "attn_memory_analytic_MB": ana_rows,
    "grad_flow_by_layer": gf,
    "note": (
        "attn_memory_MB=GPU峰值增量实测; attn_memory_analytic_MB=单份中间量解析值; "
        "grad_flow_by_layer=CPU float32 seed0; 训练指标沿用未重训的旧值"
    ),
}
SUMMARY.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
print("merged into:", SUMMARY)
