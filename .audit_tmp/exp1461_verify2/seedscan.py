# -*- coding: utf-8 -*-
"""
14.6-1 免责内容裁决实验:
  正文声称: "玩具语料太简单, 两版命中率相近(均匀版甚至略优), 命中率不构成 p_init 的卖点"
  本脚本裁决该现象是固有属性还是撰写者规避:
    A. 原任务 x 3个训练seed x 4个采样seed: 均匀版优势是否系统存在(跨训练/采样种子)
    B. 加难任务(30模板, 更宽词表, 边缘分布更平) x 2个训练seed: 命中率是否脱离饱和区,
       两版差距如何变化(若 MDLM 在难任务追平/反超, 则"任务太简单"的说法成立)
"""
import sys, io, os, json
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
import torch
import torch.nn as nn

DEVICE_LOG = []

def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32) * (-np.log(10000.0) / (half - 1)))
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ToySeqDenoiser(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=64, n_layers=2, n_heads=2, ff_dim=128, time_dim=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.time_dim = time_dim
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
                                         batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x, t):
        tf = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        h = self.tok_emb(x) + self.pos_emb.weight[None, :, :] + tf[:, None, :]
        return self.head(self.encoder(h))

class Task:
    def __init__(self, templates, words, name):
        self.name = name
        self.templates = templates
        self.words = words
        self.seq_len = len(templates[0].split())
        w2id = {w: i for i, w in enumerate(words)}
        self.TPL = np.array([[w2id[w] for w in s.split()] for s in templates])
        self.TPL_SET = {tuple(r) for r in self.TPL}
        self.N_T = len(templates)
        self.MASK_ID = len(words)
        self.V_MASK = len(words) + 1
        self.V_UNIF = len(words)

def make_original():
    WORDS = ['the', 'a', 'cat', 'dog', 'big', 'red', 'sat', 'ate', 'on', 'mat', 'rug', 'fish', 'it', 'up']
    T = ['the cat sat on the mat', 'the dog sat on the rug', 'the cat ate the red fish',
         'the dog ate the big fish', 'a big cat sat on it', 'a big dog ate it up']
    return Task(T, WORDS, 'original')

def make_hard():
    D = ['the', 'a']; N1 = ['cat', 'dog', 'fish', 'bird', 'bug']
    V = ['sat', 'ate', 'saw', 'chased']; P = ['on', 'under', 'near']
    A = ['big', 'red', 'lazy']; N2 = ['mat', 'rug', 'bone', 'tree', 'log']
    rng = np.random.RandomState(0)
    words = sorted(set(D + N1 + V + P + A + N2))
    tpls = set()
    while len(tpls) < 30:
        s = f"{rng.choice(D)} {rng.choice(N1)} {rng.choice(V)} {rng.choice(P)} {rng.choice(D)} {rng.choice(N2)}"
        tpls.add(s)
    return Task(sorted(tpls), words, 'hard30')

KAPPA = lambda t: t

def train_dfm(task, kind, n_steps=3000, batch=128, lr=3e-4):
    vocab = task.V_MASK if kind == 'mask' else task.V_UNIF
    model = ToySeqDenoiser(vocab, task.seq_len)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last = None
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, task.N_T, (batch,))
        x0 = torch.from_numpy(task.TPL[idx.numpy()])
        t = torch.rand(batch)
        keep = torch.rand(batch, task.seq_len) < KAPPA(t)[:, None]
        if kind == 'mask':
            noise = torch.full_like(x0, task.MASK_ID)
        else:
            noise = torch.randint(0, task.V_UNIF, x0.shape)
        x_t = torch.where(keep, x0, noise)
        logits = model(x_t, t)
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), x0.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        last = loss.item()
    return model, last

@torch.no_grad()
def sample_euler(task, model, kind, n_steps=32, n_samples=512):
    vocab = task.V_MASK if kind == 'mask' else task.V_UNIF
    model.eval()
    if kind == 'mask':
        x = torch.full((n_samples, task.seq_len), task.MASK_ID, dtype=torch.long)
    else:
        x = torch.randint(0, task.V_UNIF, (n_samples, task.seq_len), dtype=torch.long)
    h = 1.0 / n_steps
    n_rw = n_exp = 0
    noise_id = task.MASK_ID if kind == 'mask' else -1
    for n in range(n_steps):
        t_n = n / n_steps
        rate = min(h / (1.0 - t_n), 1.0)
        probs = torch.softmax(model(x, torch.full((n_samples,), t_n)), dim=-1)
        flip = torch.rand(x.shape) < rate
        n_exp += int((x != noise_id).sum())
        new = torch.multinomial(probs.reshape(-1, vocab), 1).reshape(x.shape)
        n_rw += int(((x != new) & flip & (x != noise_id)).sum())
        x = torch.where(flip, new, x)
    return x, n_rw, n_exp

def hit_rate(task, ids):
    return sum(1 for r in ids.numpy() if tuple(r) in task.TPL_SET) / len(ids)

def run(task, train_seeds, samp_seeds, n_steps_list=(8, 32, 64)):
    out = []
    for ts in train_seeds:
        np.random.seed(ts); torch.manual_seed(ts)
        m_mlm, l_mlm = train_dfm(task, 'mask')
        np.random.seed(ts); torch.manual_seed(ts)
        m_unif, l_unif = train_dfm(task, 'uniform')
        for ss in samp_seeds:
            row = {'task': task.name, 'train_seed': ts, 'samp_seed': ss,
                   'loss_mlm': round(l_mlm, 4), 'loss_unif': round(l_unif, 4)}
            for N in n_steps_list:
                torch.manual_seed(ss)
                fx, rw, ex = sample_euler(task, m_mlm, 'mask', n_steps=N)
                hm = hit_rate(task, fx); rm = rw / max(ex, 1)
                torch.manual_seed(ss)
                fu, rw2, ex2 = sample_euler(task, m_unif, 'uniform', n_steps=N)
                hu = hit_rate(task, fu); ru = rw2 / max(ex2, 1)
                row[f'N{N}'] = {'hit_mlm': round(hm, 4), 'hit_unif': round(hu, 4),
                                'diff': round(hu - hm, 4),
                                'rw_mlm': round(rm, 5), 'rw_unif': round(ru, 5)}
            out.append(row)
            print(f"[{task.name}] ts={ts} ss={ss} loss({l_mlm:.3f}/{l_unif:.3f}) " +
                  " ".join(f"N{N}: {row[f'N{N}']['hit_mlm']:.3f} vs {row[f'N{N}']['hit_unif']:.3f}" for N in n_steps_list),
                  flush=True)
    return out

if __name__ == '__main__':
    import time
    t0 = time.time()
    results = []
    print("=== A. 原任务: 3 训练seed x 2 采样seed ===", flush=True)
    results += run(make_original(), [42, 7, 123], [2025, 101])
    print("=== B. 加难任务(30模板/19词): 2 训练seed x 2 采样seed ===", flush=True)
    results += run(make_hard(), [42, 7], [2025, 101])
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seedscan_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"完成, 用时 {(time.time()-t0)/60:.1f} 分钟")
