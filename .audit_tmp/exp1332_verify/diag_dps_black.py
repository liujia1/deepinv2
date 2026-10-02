# -*- coding: utf-8 -*-
"""13.3-2 DPS 重跑变黑诊断（CPU）

用户用新配置 VE(sigma_max=10, alpha=0.75) + w=0.3 + 250步 在 256x256 + Levin 模糊核
上重跑，DPS 子图全黑 (PSNR=8.34)。

本脚本验证猜想：guidance 梯度进入 SDE drift 时乘了 g(t)^2 ∝ sigma^2，
在 sigma=10 的起点被放大 ~1840 倍；对 inpainting（官方 demo 调参场景）方向正确
尚可存活，对强模糊 A^T 的病态梯度 + 256x256 下 sqrt(m) 归一化改变梯度分布，
早期轨迹被推离数据区 -> 发散到负值 -> clamp 后全黑。

步骤1: 单点检查——x_T ~ N(0, 10^2) 时 DRUNet 输出统计 + DPSDataFidelity.grad 量级
       并估算前几步每步位移 |dt * (1+alpha)/2 * g^2 * grad|
步骤2: 用 get_trajectory=True 跑短轨迹(60步, 同 schedule)，打印每步 min/max/mean
"""
import time
import numpy as np
import torch
import deepinv as dinv
from deepinv.sampling import (
    PosteriorDiffusion, VarianceExplodingDiffusion, DPSDataFidelity, EulerSolver,
)

dev = torch.device("cpu")
torch.manual_seed(42)
IMG = 256

# ---- 测试图与模糊核（尽量与正式脚本一致） ----
try:
    url = dinv.utils.get_image_url("celeba_example.jpg")
    x = dinv.utils.load_url_image(url=url, img_size=IMG, resize_mode="resize").to(dev)
    print("celeba 测试图加载成功", tuple(x.shape))
except Exception as e:
    print(f"celeba 加载失败({e})，用随机平滑图代替")
    xx = np.random.RandomState(0).rand(IMG, IMG)
    from scipy.ndimage import gaussian_filter
    xx = gaussian_filter(xx, 8)
    xx = (xx - xx.min()) / (xx.max() - xx.min())
    x = torch.tensor(xx, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(dev)

try:
    from deepinv.utils.demo import load_degradation
    kern = load_degradation("Levin09.npy", "kernels", index=1)
    kernel_t = torch.tensor(np.array(kern, dtype=np.float32))
    print("Levin09[1] 加载成功, shape", kernel_t.shape)
except Exception as e:
    print(f"Levin 加载失败({e})，改用 9x9 sigma=2 高斯核近似")
    kernel_t = torch.tensor(
        [[np.exp(-((i - 4) ** 2 + (j - 4) ** 2) / (2 * 2.0 ** 2)) for j in range(9)]
         for i in range(9)], dtype=torch.float32)
    kernel_t = kernel_t / kernel_t.sum()

physics = dinv.physics.Blur(filter=kernel_t.unsqueeze(0).unsqueeze(0), device=dev,
                            padding="circular",
                            noise_model=dinv.physics.GaussianNoise(sigma=0.02))
torch.manual_seed(42)
y = physics(x)
psnr = lambda a, b: 10 * np.log10(1 / (torch.mean((a - b) ** 2).item() + 1e-12))
print(f"观测 y PSNR={psnr(y.clamp(0,1), x):.2f} dB")

drunet = dinv.models.DRUNet(pretrained="download").to(dev).eval()

# ================= 步骤1：单点检查 =================
print("\n===== 单点检查（起点 x_T ~ N(0,10^2), sigma=10）=====")
sigma_max, alpha, n_steps = 10.0, 0.75, 250
sde_probe = VarianceExplodingDiffusion(sigma_max=sigma_max, alpha=alpha,
                                       device=dev, dtype=torch.float32)
t1 = time.time()
rng = torch.Generator(device=dev).manual_seed(42)
x_T = torch.randn(x.shape, generator=rng) * sigma_max
print(f"x_T: min={x_T.min():.2f} max={x_T.max():.2f} mean={x_T.mean():.3f}")
with torch.no_grad():
    d_out = drunet(x_T, torch.tensor(10.0))
print(f"DRUNet(x_T, sigma=10): min={d_out.min():.3f} max={d_out.max():.3f} "
      f"mean={d_out.mean():.3f} std={d_out.std():.3f}  [{time.time()-t1:.1f}s]")

fid = DPSDataFidelity(denoiser=drunet, weight=0.3)
t1 = time.time()
g = fid.grad(x_T.clone(), y, physics=physics, sigma=torch.tensor(10.0))
print(f"guidance grad(|sigma=10|): mean|g|={g.abs().mean():.4f} max|g|={g.abs().max():.4f}  [{time.time()-t1:.1f}s]")

# 每步位移估计: dt * (1+alpha)/2 * g(t)^2 * |grad|
dt = (1.0 - 0.001) / (n_steps - 1)
sig = sde_probe.sigma_t(torch.tensor(1.0))
sig_p = sde_probe.sigma_prime_t(torch.tensor(1.0))
g2 = 2 * sig * sig_p  # forward_diffusion^2, VE: scale=1
step_disp = dt * (1 + alpha) / 2 * g2 * g.abs().mean()
print(f"sigma(t=1)={sig:.2f}, g(t)^2={g2:.1f}, dt={dt:.4f}")
print(f"=> 起点处 guidance 每步位移估计 ≈ {step_disp:.3f}（对比：图像合法范围 [0,1]）")

# 再测小 sigma 下同样量级（看 guidance 是否随 sigma^2 放大）
for s in (3.0, 1.0, 0.3):
    xs = x * 1.0 + torch.randn(x.shape) * s
    gs = fid.grad(xs.clone(), y, physics=physics, sigma=torch.tensor(s))
    t_ = float(torch.logsumexp(torch.zeros(1), 0))  # dummy
    sigp_ = s * (np.log(sigma_max) - np.log(0.001))
    g2_ = 2 * s * sigp_
    disp = dt * (1 + alpha) / 2 * g2_ * gs.abs().mean()
    print(f"  sigma={s:4.1f}: mean|grad|={gs.abs().mean():.4f} -> 每步位移≈{disp:.4f}")

# ================= 步骤2：短轨迹 =================
print("\n===== 短轨迹复现（同配置, 60步, 观察是否漂移变黑）=====")
n_try = 60
solver = EulerSolver(timesteps=torch.linspace(1, 0.001, n_try),
                     rng=torch.Generator(device=dev))
model = PosteriorDiffusion(
    data_fidelity=DPSDataFidelity(denoiser=drunet, weight=0.3),
    denoiser=drunet,
    sde=VarianceExplodingDiffusion(sigma_max=sigma_max, alpha=alpha,
                                   device=dev, dtype=torch.float32),
    solver=solver, dtype=torch.float32, device=dev, verbose=False,
    minus_one_one=False,
)
t0 = time.time()
out, traj = model(y, physics, seed=42, get_trajectory=True)
print(f"完成 {n_try} 步, 耗时 {time.time()-t0:.0f}s")
print(f"traj shape={tuple(traj.shape)}")
idx = list(range(0, n_try, max(1, n_try // 10))) + [n_try - 1]
for i in sorted(set(idx)):
    xt = traj[i]
    print(f"  step {i:3d}: min={xt.min():8.3f} max={xt.max():8.3f} "
          f"mean={xt.mean():8.3f} std={xt.std():7.3f}")
fin = out.detach().clamp(0, 1)
print(f"最终输出(clamp后): mean={fin.mean():.4f}, PSNR vs x = {psnr(fin, x):.2f} dB")
torch.save({"out": out.detach(), "x": x, "y": y},
           r"D:\works\deepinv\.audit_tmp\exp1332_black_diag_traj.pt")
print("已保存 traj 结果到 exp1332_black_diag_traj.pt")
