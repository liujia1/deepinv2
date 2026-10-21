# -*- coding: utf-8 -*-
"""核查实验14.2-2：y=0 处 cINN 后验是否真的双峰
思路: 加载已训练权重 -> 采 20000 个 y=0 样本 -> 与解析后验比较
      (a) x2 的边缘分布形状(双峰性指标: |x2|<1 的质量、峰谷比)
      (b) 与解析后验样本的能量距离/W2
"""
import numpy as np, torch, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.chinese'))

# 复刻模型结构
class Net1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1,64), torch.nn.Tanh(),
                                       torch.nn.Linear(64,64), torch.nn.Tanh(),
                                       torch.nn.Linear(64,2))
    def forward(self, y):
        out = self.mlp(y)
        return 1.5*torch.tanh(out[:,0:1]), out[:,1:2]

class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2,64), torch.nn.Tanh(),
                                       torch.nn.Linear(64,64), torch.nn.Tanh(),
                                       torch.nn.Linear(64,2))
    def forward(self, y, u2):
        out = self.mlp(torch.cat([y,u2],dim=1))
        return 1.5*torch.tanh(out[:,0:1]), out[:,1:2]

class CINN(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.net1=Net1(); self.net2=Net2()
    def inverse(self, z, y):
        z1, z2 = z[:,0:1], z[:,1:2]
        u2 = z2
        s2,t2 = self.net2(y,u2); u1 = (z1-t2)*torch.exp(-s2)
        s1,t1 = self.net1(y); x2 = (u2-t1)*torch.exp(-s1)
        return torch.cat([u1,x2],dim=1)

m = CINN()
m.load_state_dict(torch.load('cinn_weights.pt', map_location='cpu'))
m.eval()

N = 20000
torch.manual_seed(0)
with torch.no_grad():
    z = torch.randn(N,2)
    y = torch.zeros(N,1)
    xs = m.inverse(z,y).numpy()

# 解析后验 y=0
sig_p, sig_obs = 1.0, 0.25
v = 1/(1/sig_p**2 + 1/sig_obs**2)
mu1, mu2 = np.array([-2.5,-2.5]), np.array([2.5,2.5])
marg = sig_p**2 + sig_obs**2
w1 = 0.5*np.exp(-0.5*0**2/marg); w2 = 0.5*np.exp(-0.5*0**2/marg)
w1, w2 = w1/(w1+w2), w2/(w1+w2)
mp1 = v*(mu1[0]/sig_p**2 + 0/marg); mp2 = v*(mu2[0]/sig_p**2 + 0/marg)
print(f"解析: 分量均值 x1={mp1:.3f} / {mp2:.3f}, x2={mu1[1]},{mu2[1]}; 后验 sd_x1={np.sqrt(v):.3f}, sd_x2=1.0")

rng = np.random.default_rng(0)
n1 = rng.binomial(N, 0.5)
ana = np.zeros((N,2))
ana[:n1,0] = rng.normal(mp1, np.sqrt(v), n1); ana[:n1,1] = rng.normal(-2.5, 1.0, n1)
ana[n1:,0] = rng.normal(mp2, np.sqrt(v), N-n1); ana[n1:,1] = rng.normal(2.5, 1.0, N-n1)

print("\n[关键指标] y=0 处 x2 的分布")
print(f"  |x2|<1.0 的质量: cINN={np.mean(np.abs(xs[:,1])<1.0):.3f}  解析={np.mean(np.abs(ana[:,1])<1.0):.3f}")
print(f"  |x2|<1.5 的质量: cINN={np.mean(np.abs(xs[:,1])<1.5):.3f}  解析={np.mean(np.abs(ana[:,1])<1.5):.3f}")
print(f"  x2 的偏度/峰度: cINN=({float(((xs[:,1]-xs[:,1].mean())**3).mean()/xs[:,1].std()**3):.3f},"
      f"{float(((xs[:,1]-xs[:,1].mean())**4).mean()/xs[:,1].std()**4):.3f})  "
      f"解析=({float(((ana[:,1]-ana[:,1].mean())**3).mean()/ana[:,1].std()**3):.3f},"
      f"{float(((ana[:,1]-ana[:,1].mean())**4).mean()/ana[:,1].std()**4):.3f})")
print(f"  x2 标准差: cINN={xs[:,1].std():.3f}  解析={ana[:,1].std():.3f}")

# 双峰性: 用 1D KDE 的峰谷比 (|x2|=0 处密度 / x2=±2.5 处密度)
grid = np.linspace(-5,5,401)
def kde(vals, grid, bw=0.25):
    vals = vals[:4000]
    d = (grid[:,None]-vals[None,:])/bw
    return np.exp(-0.5*d**2).sum(1)/(len(vals)*bw*np.sqrt(2*np.pi))
kc = kde(xs[:,1], grid); ka = kde(ana[:,1], grid)
i0 = np.argmin(np.abs(grid)); ipk = np.argmin(np.abs(grid-2.5))
print(f"  KDE(x2=0)/KDE(x2=2.5): cINN={kc[i0]/kc[ipk]:.3f}  解析={ka[i0]/ka[ipk]:.3f}   (解析=0.135≈两峰间隔过远)")

# 二维能量距离(近似, 用最近邻估计) 与分坐标 W2
from scipy.stats import wasserstein_distance
print(f"\n  坐标 W2: x1={wasserstein_distance(xs[:,0],ana[:,0]):.4f}  x2={wasserstein_distance(xs[:,1],ana[:,1]):.4f}")

# 用真正的双峰标签看跨峰样本
print("\n  按最近峰归属(x2 符号)后, 各峰内部 |x2-峰心| 均值:")
for sgn in (-1,1):
    mask = np.sign(xs[:,1])==sgn
    print(f"   峰 {sgn*2.5:+.1f}: 占比 {mask.mean():.3f}, x2 均值 {xs[mask,1].mean():+.3f}, sd {xs[mask,1].std():.3f}")
