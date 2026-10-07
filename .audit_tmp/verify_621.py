# -*- coding: utf-8 -*-
"""实验6.2-1 复核脚本：
1) 复算步骤1-2：ESM/ISM 最优theta、损失差曲线（是否真为水平线）
2) 加载 checkpoint_esm/ism_final.pth，复算步骤3验证损失（对照图标签 0.0000/0.0068）
3) 重跑步骤4等价性证据链（3种容量 x ESM/ISM），核对"容量越大差异越小"是否成立
"""
import numpy as np
import torch, os, sys

SAVE_DIR = r"d:\works\deepinv\第6章 得分匹配：从去噪中学习得分\第六章 配套实现\实验6.2-1"
device = torch.device('cpu')

# ---------- 复现步骤1-2 ----------
np.random.seed(42)
torch.manual_seed(42)

def gm_score(x):
    p1 = np.exp(-0.5 * (x + 2)**2) / np.sqrt(2*np.pi)
    p2 = np.exp(-0.5 * (x - 2)**2) / np.sqrt(2*np.pi)
    p = 0.5 * p1 + 0.5 * p2
    grad_p1 = -(x + 2) * p1
    grad_p2 = -(x - 2) * p2
    return (0.5 * grad_p1 + 0.5 * grad_p2) / p

N_samples = 50000
x_samples = np.concatenate([
    np.random.randn(N_samples // 2) - 2,
    np.random.randn(N_samples // 2) + 2
])

thetas = np.linspace(-1.5, 0.5, 100)
def esm_loss(theta):
    return 0.5 * np.mean((theta * x_samples - gm_score(x_samples))**2)
esm_losses = np.array([esm_loss(t) for t in thetas])
E_x2 = np.mean(x_samples**2)
ism_losses = thetas + 0.5 * thetas**2 * E_x2
best_esm = thetas[np.argmin(esm_losses)]
best_ism = thetas[np.argmin(ism_losses)]
print(f"[步骤1-2] E[x^2]={E_x2:.6f}  解析theta*={-1/E_x2:.6f}")
print(f"[步骤1-2] ESM argmin theta={best_esm:.4f}, ISM argmin theta={best_ism:.4f} (网格步长{thetas[1]-thetas[0]:.4f})")
diff = esm_losses - ism_losses
slope, intercept = np.polyfit(thetas, diff, 1)
print(f"[步骤1-2] ESM-ISM差: mean={diff.mean():.6f}, min={diff.min():.6f}, max={diff.max():.6f}, 斜率={slope:.3e}, 相对起伏={(diff.max()-diff.min())/diff.mean()*100:.3f}%")
# 蒙特卡洛解释: E_emp[x*s*] 与 -1 的偏差
print(f"[步骤1-2] E_emp[x*score]={np.mean(x_samples*gm_score(x_samples)):.6f} (分部积分理论值 -1)")
# 总体层面(数值积分)验证差为常数
from numpy.polynomial.hermite_e import *  # not used, do grid integration instead
xg = np.linspace(-15, 15, 2000001)
def p_pdf(x):
    return 0.5*np.exp(-0.5*(x+2)**2)/np.sqrt(2*np.pi) + 0.5*np.exp(-0.5*(x-2)**2)/np.sqrt(2*np.pi)
def score_pop(x):
    return gm_score(x)
dx = xg[1]-xg[0]
pop_diff = []
for t in [-1.5, -0.75, -0.2, 0.0, 0.5]:
    esm_pop = np.trapezoid(0.5*(t*xg - score_pop(xg))**2 * p_pdf(xg), dx=dx)
    ism_pop = np.trapezoid(t + 0.5*(t*xg)**2 * p_pdf(xg), dx=dx)
    pop_diff.append((t, esm_pop - ism_pop))
print("[步骤1-2] 总体层面 ESM-ISM (应为常数=0.5E[|score|^2]):", [(f"{t:.2f}", f"{d:.6f}") for t, d in pop_diff])

# ---------- 复现步骤3的数据顺序（RNG流必须与原脚本一致） ----------
N_TRAIN, N_VAL = 20000, 5000
x_train = np.concatenate([
    np.random.randn(N_TRAIN // 2) - 2,
    np.random.randn(N_TRAIN // 2) + 2
]).astype(np.float32)
x_val = np.concatenate([
    np.random.randn(N_VAL // 2) - 2,
    np.random.randn(N_VAL // 2) + 2
]).astype(np.float32)
true_score_val = gm_score(x_val).astype(np.float32)
x_val_t = torch.tensor(x_val).unsqueeze(1)
true_score_val_t = torch.tensor(true_score_val).unsqueeze(1)

class ScoreNet1D(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim), torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1))
    def forward(self, x):
        return self.net(x)

for method, expect in [('esm', 0.0000), ('ism', 0.0068)]:
    ck = torch.load(os.path.join(SAVE_DIR, f'checkpoint_{method}_final.pth'),
                    map_location=device, weights_only=False)
    m = ScoreNet1D(64)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    with torch.no_grad():
        v = 0.5 * torch.mean((m(x_val_t) - true_score_val_t)**2).item()
    tl, vl = ck.get('train_losses', []), ck.get('val_losses', [])
    print(f"[步骤3] {method.upper()}: ckpt epoch={ck['epoch']}, len(train)={len(tl)}, "
          f"train_loss[-1]={tl[-1]:.6f}, val[-1]={vl[-1]:.6f}, 复算val(ESM metric)={v:.6f} (图标签≈{expect})")

# ---------- 重跑步骤4等价性证据链 ----------
x_grid = np.linspace(-6, 6, 200)
true_score = gm_score(x_grid)
true_score_train = gm_score(x_train).astype(np.float32)
x_train_t = torch.tensor(x_train).unsqueeze(1)
true_score_train_t = torch.tensor(true_score_train).unsqueeze(1)
x_grid_t = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(1)
BATCH_SIZE, LR = 256, 1e-3

class ScoreNetFlex(torch.nn.Module):
    def __init__(self, hidden_dim=64, n_hidden=2):
        super().__init__()
        layers = [torch.nn.Linear(1, hidden_dim), torch.nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Tanh()]
        layers += [torch.nn.Linear(hidden_dim, 1)]
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

print("\n[步骤4] 重跑等价性证据链 (CPU, 100 epochs, seed=42)...")
configs = [('Shallow(1->8->1)', 8, 1), ('Medium(1->32->1)', 32, 1), ('Deep(1->64->64->1)', 64, 2)]
results = []
for name, hidden, n_layers in configs:
    np.random.seed(42); torch.manual_seed(42)
    model_e = ScoreNetFlex(hidden, n_layers)
    opt_e = torch.optim.Adam(model_e.parameters(), lr=LR)
    for _ in range(100):
        perm = np.random.permutation(N_TRAIN)
        for start in range(0, N_TRAIN, BATCH_SIZE):
            idx = perm[start:min(start+BATCH_SIZE, N_TRAIN)]
            opt_e.zero_grad()
            loss = 0.5*torch.mean((model_e(x_train_t[idx]) - true_score_train_t[idx])**2)
            loss.backward(); opt_e.step()
    model_e.eval()
    with torch.no_grad():
        esm_pred = model_e(x_grid_t).numpy().flatten()
    esm_l = 0.5*np.mean((esm_pred - true_score)**2)

    np.random.seed(42); torch.manual_seed(42)
    model_i = ScoreNetFlex(hidden, n_layers)
    opt_i = torch.optim.Adam(model_i.parameters(), lr=LR)
    for _ in range(100):
        perm = np.random.permutation(N_TRAIN)
        for start in range(0, N_TRAIN, BATCH_SIZE):
            idx = perm[start:min(start+BATCH_SIZE, N_TRAIN)]
            xb = x_train_t[idx].clone().requires_grad_(True)
            opt_i.zero_grad()
            s = model_i(xb)
            tr = torch.autograd.grad(s.sum(), xb, create_graph=True)[0]
            loss = torch.mean(tr + 0.5*s**2)
            loss.backward(); opt_i.step()
    model_i.eval()
    with torch.no_grad():
        ism_pred = model_i(x_grid_t).numpy().flatten()
    ism_l = 0.5*np.mean((ism_pred - true_score)**2)
    d = np.sqrt(np.mean((esm_pred - ism_pred)**2))
    results.append((name, esm_l, ism_l, d))
    print(f"  {name}: ESM loss={esm_l:.6f}, ISM loss={ism_l:.6f}, ESM-ISM diff(RMSE)={d:.6f}")

diffs = [r[3] for r in results]
mono = diffs[0] >= diffs[1] >= diffs[2]
print(f"[步骤4] diff序列={np.round(diffs,4).tolist()}  单调递减('容量越大差异越小')成立? {mono}")
