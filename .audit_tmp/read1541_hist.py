"""读取 15.4-1 重跑后的 loss_history, 输出 md 表格所需轮次"""
import pathlib
import torch

D = pathlib.Path(
    r"h:\deepinv\deepinv2\第15章 扩散模型的架构实践：UNet → DiT"
    r"\第十五章 配套实验\实验15.4-1"
)
files = {
    "unet": "unet_step2_checkpoint.pth",
    "dit": "dit_step2_checkpoint.pth",
    "zero": "dit_zero_step3_checkpoint.pth",
    "nozero": "dit_nozero_step3_checkpoint.pth",
}
hist = {}
for k, f in files.items():
    c = torch.load(D / f, map_location="cpu", weights_only=False)
    h = c["loss_history"]
    hist[k] = h
    print(f"{k:7s} epochs={len(h):2d} is_final={c['is_final']} last={h[-1]:.6f}")
    print("   ", ", ".join(f"{v:.4f}" for v in h))

print("\n=== md 表格取值 ===")
for k in ("unet", "dit"):
    h = hist[k]
    print(f"{k}: ep10={h[9]:.4f} ep20={h[19]:.4f} ep30={h[29]:.4f}")
for k in ("zero", "nozero"):
    h = hist[k]
    print(f"{k}: ep10={h[9]:.4f} ep20={h[19]:.4f}")
