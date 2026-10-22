"""冒烟测试: 从 15.5-1.py 提取 sample_ddpm, 验证共享噪声机制
1) 同一模型两次调用输出必须完全一致(确定性)
2) 两个不同模型调用输出不同, 但使用同一噪声轨迹
3) 传入的 init_noise / noise_seq 不被修改
"""
import ast
import pathlib

import torch

SRC = pathlib.Path(
    r"h:\deepinv\deepinv2\第15章 扩散模型的架构实践：UNet → DiT"
    r"\第十五章 配套实验\实验15.5-1\15.5-1.py"
)
tree = ast.parse(SRC.read_text(encoding="utf-8"))
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sample_ddpm")
fn_src = ast.get_source_segment(SRC.read_text(encoding="utf-8"), fn)

T = 5
device = torch.device("cpu")
betas = torch.linspace(1e-4, 0.02, T)
alpha_bars = torch.cumprod(1 - betas, 0)

ns = {"torch": torch, "T": T, "device": device, "betas": betas, "alpha_bars": alpha_bars}
exec(fn_src, ns)
sample_ddpm = ns["sample_ddpm"]


class Dummy(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x, t):
        return x * self.scale


g = torch.Generator(device=device).manual_seed(0)
init_noise = torch.randn(2, 1, 28, 28, device=device, generator=g)
noise_seq = [torch.randn(2, 1, 28, 28, device=device, generator=g) for _ in range(T)]
init_copy = init_noise.clone()
seq_copy = [z.clone() for z in noise_seq]

a1 = sample_ddpm(Dummy(0.1), 2, init_noise, noise_seq)
a2 = sample_ddpm(Dummy(0.1), 2, init_noise, noise_seq)
b = sample_ddpm(Dummy(0.5), 2, init_noise, noise_seq)

print("同模型两次输出一致:", torch.equal(a1, a2))
print("不同模型输出不同:", not torch.equal(a1, b))
print("init_noise 未被修改:", torch.equal(init_noise, init_copy))
print("noise_seq 未被修改:", all(torch.equal(x, y) for x, y in zip(noise_seq, seq_copy)))
print("默认参数(不传噪声)仍可运行:", tuple(sample_ddpm(Dummy(0.1), 2).shape))
