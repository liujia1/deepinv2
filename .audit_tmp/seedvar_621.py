# -*- coding: utf-8 -*-
"""实验6.2-1 步骤4 多seed稳健性测试：
数据管线与归档一致（seed42 生成 x_samples->x_train），仅改变模型初始化与minibatch顺序的seed。
验证：ESM-ISM diff 随容量非单调、ISM 对初始化敏感 是否跨 seed 成立。
"""
import numpy as np
import torch

device = torch.device('cpu')

np.random.seed(42)
torch.manual_seed(42)

def gm_score(x):
    p1 = np.exp(-0.5 * (x + 2)**2) / np.sqrt(2*np.pi)
    p2 = np.exp(-0.5 * (x - 2)**2) / np.sqrt(2*np.pi)
    p = 0.5 * p1 + 0.5 * p2
    grad_p1 = -(x + 2) * p1
    grad_p2 = -(x - 2) * p2
    return (0.5 * grad_p1 + 0.5 * grad_p2) / p

# 复现原脚本 RNG 流：x_samples(50000) -> x_train(20000)
N_samples = 50000
_ = np.concatenate([np.random.randn(N_samples // 2) - 2, np.random.randn(N_samples // 2) + 2])
N_TRAIN = 20000
x_train = np.concatenate([
    np.random.randn(N_TRAIN // 2) - 2,
    np.random.randn(N_TRAIN // 2) + 2
]).astype(np.float32)

x_grid = np.linspace(-6, 6, 200)
true_score = gm_score(x_grid)
true_score_train = gm_score(x_train).astype(np.float32)
x_train_t = torch.tensor(x_train).unsqueeze(1)
true_score_train_t = torch.tensor(true_score_train).unsqueeze(1)
x_grid_t = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(1)

BATCH_SIZE, LR, EPOCHS = 256, 1e-3, 100

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

def train_esm(seed, hidden, n_layers):
    np.random.seed(seed); torch.manual_seed(seed)
    m = ScoreNetFlex(hidden, n_layers)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    for _ in range(EPOCHS):
        perm = np.random.permutation(N_TRAIN)
        for start in range(0, N_TRAIN, BATCH_SIZE):
            idx = perm[start:min(start + BATCH_SIZE, N_TRAIN)]
            opt.zero_grad()
            loss = 0.5 * torch.mean((m(x_train_t[idx]) - true_score_train_t[idx])**2)
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(x_grid_t).numpy().flatten()

def train_ism(seed, hidden, n_layers):
    np.random.seed(seed); torch.manual_seed(seed)
    m = ScoreNetFlex(hidden, n_layers)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    for _ in range(EPOCHS):
        perm = np.random.permutation(N_TRAIN)
        for start in range(0, N_TRAIN, BATCH_SIZE):
            idx = perm[start:min(start + BATCH_SIZE, N_TRAIN)]
            xb = x_train_t[idx].clone().requires_grad_(True)
            opt.zero_grad()
            s = m(xb)
            tr = torch.autograd.grad(s.sum(), xb, create_graph=True)[0]
            loss = torch.mean(tr + 0.5 * s**2)
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(x_grid_t).numpy().flatten()

configs = [('Shallow', 8, 1), ('Medium', 32, 1), ('Deep', 64, 2)]
seeds = [42, 0, 1, 2, 3]

print(f"{'seed':>5s} | {'config':>8s} | {'ESM loss':>9s} | {'ISM loss':>9s} | {'diff(RMSE)':>10s}")
print("-" * 55)
per_seed = {}
for seed in seeds:
    esm_ls, ism_ls, diffs = [], [], []
    for name, hidden, n_layers in configs:
        ep = train_esm(seed, hidden, n_layers)
        ip = train_ism(seed, hidden, n_layers)
        e = 0.5 * np.mean((ep - true_score)**2)
        i = 0.5 * np.mean((ip - true_score)**2)
        d = np.sqrt(np.mean((ep - ip)**2))
        esm_ls.append(e); ism_ls.append(i); diffs.append(d)
        print(f"{seed:5d} | {name:>8s} | {e:9.4f} | {i:9.4f} | {d:10.4f}", flush=True)
    mono = bool(np.all(np.diff(diffs) < 0))
    per_seed[seed] = (esm_ls, ism_ls, diffs, mono)
    print(f"      diff序列={['%.4f' % x for x in diffs]}  单调递减? {'是' if mono else '否'}\n", flush=True)

print("=" * 60)
print("汇总：")
n_mono = sum(1 for v in per_seed.values() if v[3])
print(f"- '容量越大差异越小'单调成立: {n_mono}/{len(seeds)} 个seed")
ism_mat = np.array([v[1] for v in per_seed.values()])   # (n_seed, n_cfg)
esm_mat = np.array([v[0] for v in per_seed.values()])
diff_mat = np.array([v[2] for v in per_seed.values()])
for j, (name, _, _) in enumerate(configs):
    print(f"- {name:>8s}: ESM loss {esm_mat[:,j].mean():.4f}±{esm_mat[:,j].std():.4f}, "
          f"ISM loss {ism_mat[:,j].mean():.4f}±{ism_mat[:,j].std():.4f}, "
          f"diff {diff_mat[:,j].mean():.4f}±{diff_mat[:,j].std():.4f}")
rel_cv_esm = (esm_mat.std(axis=0) / esm_mat.mean(axis=0)).mean()
rel_cv_ism = (ism_mat.std(axis=0) / ism_mat.mean(axis=0)).mean()
print(f"- seed间相对波动(变异系数均值): ESM={rel_cv_esm:.2%}, ISM={rel_cv_ism:.2%}")

import json, io
out = {str(s): {'esm_loss': [float(x) for x in v[0]], 'ism_loss': [float(x) for x in v[1]],
                'diff': [float(x) for x in v[2]], 'monotone': bool(v[3])}
       for s, v in per_seed.items()}
with io.open(r"d:\works\deepinv\.audit_tmp\exp621_seedvar.json", 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("已存档 exp621_seedvar.json")
