import sys, torch
sys.path.insert(0, r"D:\works\deepinv\第4章 MCMC与ULA算法\第四章 配套实验\实验4.7-1")
from sampling_tools.load_model import load_model
p = r"D:\works\deepinv\第4章 MCMC与ULA算法\第四章 配套实验\实验4.7-1\Pretrained_models\RealSN_DnCNN_noise5.pth"
for dv in [torch.device("cpu"), "cpu", torch.device("cuda")]:
    m = load_model(p, dv)
    x = torch.rand(256,256)
    out = (x - m(x[None][None])[0][0]).detach()
    print(f"device={dv!r:24s} -> load OK, denoise out.shape={tuple(out.shape)}")
print("ALL OK")
