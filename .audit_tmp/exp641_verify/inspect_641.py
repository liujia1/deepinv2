# -*- coding: utf-8 -*-
"""独立核查: 读 checkpoint, 复算步骤4相关系数/MSE、步骤5 ISM精确损失、步骤3 ESM/ISM积分口径"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cpu')

ck = torch.load(os.path.join(HERE, 'ssm_checkpoint.pth'), map_location=device)
print('== checkpoint 字段 ==')
for k, v in ck.items():
    if k == 'train_losses':
        arr = np.array(v)
        print(f'  train_losses: len={len(arr)}, first={arr[0]:.6f}, last={arr[-1]:.6f}, min={arr.min():.6f}, max={arr.max():.6f}')
    elif k == 'model_state_dict':
        print('  model_state_dict keys:', list(v.keys()))
    elif k == 'optimizer_state_dict':
        print('  optimizer: yes')
    else:
        print(f'  {k}: {v}')

# ---- 复算步骤4: 学习得分 vs 真实得分 ----
sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location('ssm_mod', os.path.join(HERE, '6.4-1.py'))

# 不执行整个脚本(会重训), 手动重建网络
class ScoreNet1D(torch.nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x.unsqueeze(-1)).squeeze(-1)

def gm1d_score(x):
    p1 = np.exp(-0.5 * (x + 2)**2) / np.sqrt(2 * np.pi)
    p2 = np.exp(-0.5 * (x - 2)**2) / np.sqrt(2 * np.pi)
    p = 0.5 * p1 + 0.5 * p2
    grad_p1 = -(x + 2) * p1
    grad_p2 = -(x - 2) * p2
    return (0.5 * grad_p1 + 0.5 * grad_p2) / p

model = ScoreNet1D()
model.load_state_dict(ck['model_state_dict'])
model.eval()

x_eval = np.linspace(-6, 6, 200)
with torch.no_grad():
    learned = model(torch.tensor(x_eval, dtype=torch.float32)).numpy()
true_s = gm1d_score(x_eval)
corr = np.corrcoef(learned, true_s)[0, 1]
mse = np.mean((learned - true_s) ** 2)
print(f'\n== 步骤4 复算 ==\n  corr = {corr:.4f}, mse = {mse:.6f}  (图标题应为 0.9995)')

# ---- 复算步骤5: ISM 精确损失 (同一 RNG 种子 123 的样本) ----
rng_ism = np.random.RandomState(123)
N = 1000
x_is = np.concatenate([rng_ism.randn(N // 2) - 2, rng_ism.randn(N // 2) + 2])
x_is_t = torch.tensor(x_is, dtype=torch.float32, requires_grad=True)
s_pred = model(x_is_t)
ds = torch.autograd.grad(s_pred.sum(), x_is_t)[0].detach().numpy()
s_c = s_pred.detach().numpy()
ism_exact = (ds + 0.5 * s_c ** 2).mean()
print(f'== 步骤5 复算 ==\n  ism_exact = {ism_exact:.6f}  (图中红虚线应为 -0.3484)')

# ---- 步骤3: 讲义/print 的 ESM/ISM 积分口径检查 ----
x_grid = np.linspace(-5, 5, 200)
h = 1e-4
true_score = gm1d_score(x_grid)
ds_dx = (gm1d_score(x_grid + h) - gm1d_score(x_grid - h)) / (2 * h)
_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
esm_value = _trapz(0.5 * true_score ** 2, x_grid)          # 代码所称 "ESM 积分"
ism_value = _trapz(ds_dx + 0.5 * true_score ** 2, x_grid)  # 代码所称 "ISM 积分"
# 正确的 E_p 加权版本 (p 为 GMM 密度)
p1 = np.exp(-0.5 * (x_grid + 2) ** 2) / np.sqrt(2 * np.pi)
p2 = np.exp(-0.5 * (x_grid - 2) ** 2) / np.sqrt(2 * np.pi)
p = 0.5 * p1 + 0.5 * p2
ism_proper = _trapz(p * (ds_dx + 0.5 * true_score ** 2), x_grid)
esm_proper = 0.0  # s_theta = s 时 ESM 真值
print('== 步骤3 口径检查 ==')
print(f'  代码 esm_value = ∫0.5‖s‖²dx (均匀测度) = {esm_value:.6f}  <- print 却称 "理论值 0"')
print(f'  代码 ism_value = ∫(ds/dx + 0.5s²)dx (均匀测度) = {ism_value:.6f}')
print(f'  正确 ISM = E_p[Tr(∇s)+0.5‖s‖²] ≈ {ism_proper:.6f} (理论 = -0.5·E_p‖s‖²)')
p_s2 = _trapz(p * true_score ** 2, x_grid)
print(f'  -0.5·E_p‖s² = {-0.5 * p_s2:.6f}')

# ---- 步骤1: 多次试验检验 1/√M 收敛 (原图单次误差曲线是否足以支撑结论) ----
np.random.seed(42)
d = 20
A = np.random.randn(d, d)
A = A @ A.T / d + np.eye(d)
tr = np.trace(A)
print(f'\n== 步骤1 复算 ==\n  true_trace = {tr:.6f} (图注应为 38.41)')
theory_var_g = 2 * np.linalg.norm(A, 'fro') ** 2
theory_var_r = theory_var_g - 2 * np.sum(np.diag(A) ** 2)
print(f'  理论方差: 高斯 {theory_var_g:.3f}, Rademacher {theory_var_r:.3f} (图: 179.020 / 28.370)')

Ms = [1, 5, 10, 50, 100, 500, 1000]
print('  M | 高斯SE理论 | 高斯RMSE(2000次) | Rad SE理论 | Rad RMSE(2000次)')
rng = np.random.RandomState(7)
for M in Ms:
    ntr = 2000 if M <= 100 else 400
    V_g = rng.randn(ntr, M, d)
    est_g = np.einsum('nmi,ij,nmj->nm', V_g, A, V_g).mean(axis=1)
    V_r = rng.choice([-1.0, 1.0], size=(ntr, M, d))
    est_r = np.einsum('nmi,ij,nmj->nm', V_r, A, V_r).mean(axis=1)
    print(f'  {M:5d} | {np.sqrt(theory_var_g/M):10.4f} | {np.sqrt(np.mean((est_g-tr)**2)):14.4f} '
          f'| {np.sqrt(theory_var_r/M):11.4f} | {np.sqrt(np.mean((est_r-tr)**2)):14.4f}')
