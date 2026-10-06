# -*- coding: utf-8 -*-
"""估算CPU上hvae一轮训练的耗时(含额外prior网络)"""
import time, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

class Net(nn.Module):
    def __init__(s, din, dh, dout):
        super().__init__()
        s.f = nn.Linear(din, dh); s.m = nn.Linear(dh, dout); s.v = nn.Linear(dh, dout)
    def forward(s, x):
        h = F.relu(s.f(x)); return s.m(h), s.v(h)

e1 = Net(784, 400, 20); e2 = Net(20, 200, 10); d1 = Net(10, 200, 20)
p1 = Net(10, 200, 20)  # 摊销先验网络 z2 -> (mu_p, logvar_p)
d0f = nn.Linear(20, 400); d0o = nn.Linear(400, 784)
params = list(e1.parameters())+list(e2.parameters())+list(d1.parameters())+list(p1.parameters())+[d0f.weight,d0f.bias,d0o.weight,d0o.bias]
opt = torch.optim.Adam(params, lr=1e-3)
x = torch.rand(128, 784)

def step():
    mu1, lv1 = e1(x); z1 = mu1 + torch.randn_like(mu1)*torch.exp(.5*lv1)
    mu2, lv2 = e2(z1); z2 = mu2 + torch.randn_like(mu2)*torch.exp(.5*lv2)
    mup, lvpp = d1(z2)  # p(z1|z2) 解析高斯参数
    kl1 = 0.5*(torch.exp(lv1-lvpp) + (mu1-mup)**2 - (lv1-lvpp) - 1).sum()
    kl2 = -0.5*torch.sum(1 + lv2 - mu2.pow(2) - lv2.exp())
    xr = torch.sigmoid(d0o(F.relu(d0f(z1))))
    recon = F.binary_cross_entropy(xr, x, reduction='sum')
    loss = recon + kl1 + kl2
    opt.zero_grad(); loss.backward(); opt.step()

t = time.time()
for _ in range(50):
    step()
dt = (time.time()-t)/50
print(f"每batch {dt*1000:.1f} ms -> 一轮469 batch ≈ {dt*469:.1f} s -> 20轮 ≈ {dt*469*20/60:.1f} min")
