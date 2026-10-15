import numpy as np, torch
def ct(size=64):
    x = np.zeros((1, size, size)); c = size // 2
    for i in range(size):
        for j in range(size):
            if abs(i-c) < size//4 and abs(j-c) < size//4: x[0,i,j] = 1.0
            elif abs(i-c) < size//3 and abs(j-c) < size//3: x[0,i,j] = 0.5
    return torch.tensor(x, dtype=torch.float32)
def kb(size=5, sigma=1.0):
    a = np.arange(size)-size//2; k1 = np.exp(-a**2/(2*sigma**2)); k2 = np.outer(k1, k1)
    return torch.tensor(k2/k2.sum(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
cv = lambda x, k: torch.nn.functional.conv2d(x, k, padding=k.shape[-1]//2)
ps = lambda x, t: 10*np.log10(1.0/max(torch.mean((x-t)**2).item(), 1e-12))
xt = ct(64); ker = kb(5, 1.0); A = lambda x: cv(x, ker); AT = lambda x: cv(x, ker.flip(-1).flip(-2))
def den(x, s):
    x4 = x.unsqueeze(0)
    xp = torch.nn.functional.pad(x4, (1,1,1,1), mode='reflect')
    p = torch.stack([xp[:,:,1+di:1+di+64,1+dj:1+dj+64] for di in [-1,0,1] for dj in [-1,0,1]])
    w = torch.exp(-(p-x4)**2/(2*s**2)); w = w/(w.sum(0, keepdim=True)+1e-10)
    return (w*p).sum(0).squeeze(0)
def pnp(y, step, sd, ni):
    x = AT(y).clone()
    for _ in range(ni): x = den(x - step*AT(A(x)-y), sd)
    return x
for seed in [42, 0, 1, 7, 123]:
    g = torch.Generator().manual_seed(seed)
    y = A(xt) + 0.05*torch.randn(xt.shape, generator=g)
    pl = ps(AT(y), xt); pp = ps(den(AT(y), 0.2), xt); pi = ps(pnp(y, 0.5, 0.05, 50), xt)
    print(f'seed={seed:<4} 线性={pl:.2f} 单次PnP={pp:.2f} 迭代PnP-GD={pi:.2f}')
