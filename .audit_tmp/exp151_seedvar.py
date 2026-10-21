"""审计用: 检查梯度流比值的 seed / device 敏感性, 判定 md 数字是否需要更新。"""
import ast
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
ns = {"torch": torch, "nn": nn, "F": F, "math": math, "rearrange": rearrange}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "<x>", "exec"), ns)
_PreNormBlock = ns["_PreNormBlock"]
_PostNormBlock = ns["_PostNormBlock"]

DEPTH = 12


def run(cls, dev, seed):
    torch.manual_seed(seed)
    layers = nn.ModuleList([cls(64) for _ in range(DEPTH)]).to(dev)
    h = torch.randn(2, 64, 16, 16, device=dev)
    for l in layers:
        h = l(h)
    h.pow(2).mean().backward()
    return [float(l.norm.weight.grad.norm()) for l in layers]


for dev in [torch.device("cpu"), torch.device("cuda")]:
    print(f"\n--- device={dev} ---")
    for seed in [0, 1, 2, 42]:
        pn = run(_PreNormBlock, dev, seed)
        pon = run(_PostNormBlock, dev, seed)
        print(
            f"seed={seed:2d} PreNorm 1st={pn[0]:.3e} last={pn[-1]:.3e} last/1st={pn[-1]/pn[0]:.2f} | "
            f"PostNorm 1st={pon[0]:.3e} last={pon[-1]:.3e} last/1st={pon[-1]/max(pon[0],1e-30):.2e}"
        )
