# -*- coding: utf-8 -*-
"""
实验14.6-1 复核补充: C块——随机30序列(纯记忆任务), 真正脱离饱和区的困难语料
语法共享的30模板语料仍饱和(~0.88), 本块用30条随机6词序列(14词表中采样, 固定种子去重),
使命中率脱离饱和区后, 检验书中"任务变难后 MDLM 反超"的声明。
模型/训练/采样与 14.6-1.py 一致; 训练后保存 checkpoint 以防中断。
"""
import sys, io, os, json, time
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
import torch
import torch.nn as nn

OUT = os.path.dirname(os.path.abspath(__file__))

def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32) * (-np.log(10000.0) / (half - 1)))
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ToySeqDenoiser(nn.Module):
    def __init__(self, vocab_size, seq_len=6, d_model=64, n_layers=2, n_heads=2, ff_dim=128, time_dim=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.time_dim = time_dim
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, t):
        time_feat = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        h = self.tok_emb(x) + self.pos_emb.weight[None, :, :] + time_feat[:, None, :]
        h = self.encoder(h)
        return self.head(h)

def train_dfm(tpl_ids, vocab_size, mask_id, n_steps=3000, batch_size=128, lr=3e-4):
    seq_len = tpl_ids.shape[1]
    n_tpl = tpl_ids.shape[0]
    model = ToySeqDenoiser(vocab_size, seq_len=seq_len)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x0_all = torch.from_numpy(tpl_ids)
    model.train()
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_tpl, (batch_size,))
        x0 = x0_all[idx]
        t = torch.rand(batch_size)
        keep = torch.rand(batch_size, seq_len) < t[:, None]
        if mask_id is not None:
            noise_tok = torch.full_like(x0, mask_id)
        else:
            noise_tok = torch.randint(0, vocab_size, x0.shape)
        x_t = torch.where(keep, x0, noise_tok)
        logits = model(x_t, t)
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), x0.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, float(loss.item())

@torch.no_grad()
def sample_euler(model, mask_id, vocab_size, seq_len=6, n_steps=32, n_samples=256):
    if mask_id is not None:
        x = torch.full((n_samples, seq_len), mask_id, dtype=torch.long)
    else:
        x = torch.randint(0, vocab_size, (n_samples, seq_len), dtype=torch.long)
    h = 1.0 / n_steps
    n_rewrite = n_exposed = 0
    for n in range(n_steps):
        t_n = n / n_steps
        rate = min(h * (1.0 / (1.0 - t_n)), 1.0)
        t_batch = torch.full((n_samples,), t_n)
        probs = torch.softmax(model(x, t_batch), dim=-1)
        flip = torch.rand(x.shape) < rate
        new_tok = torch.multinomial(probs.reshape(-1, vocab_size), 1).reshape(x.shape)
        if mask_id is None:
            n_exposed += x.numel()
            n_rewrite += int(((x != new_tok) & flip).sum())
        else:
            n_exposed += int((x != mask_id).sum())
            n_rewrite += int(((x != new_tok) & flip & (x != mask_id)).sum())
        x = torch.where(flip, new_tok, x)
    return x, n_rewrite, n_exposed

def hit_rate(ids, tpl_set):
    return sum(1 for row in ids.numpy() if tuple(row) in tpl_set) / len(ids)

# ---- 随机30序列语料 ----
WORDS6 = ['the', 'a', 'cat', 'dog', 'big', 'red', 'sat', 'ate', 'on',
          'mat', 'rug', 'fish', 'it', 'up']
_rng = np.random.RandomState(20260930)
_seen, TPLR = set(), []
while len(TPLR) < 30:
    s = ' '.join(_rng.choice(WORDS6, 6))
    if s not in _seen:
        _seen.add(s); TPLR.append(s)
assert all(len(t.split()) == 6 for t in TPLR)
print(f"[语料] 随机30序列构造完成 (seed=20260930), 示例:")
for t in TPLR[:4]:
    print(f"    {t}")

TRAIN_SEEDS = [42, 7, 123]
SAMPLE_SEEDS = [2025, 2026]
word2id = {w: i for i, w in enumerate(WORDS6)}
tpl_ids = np.array([[word2id[w] for w in s.split()] for s in TPLR])
tpl_set = {tuple(r) for r in tpl_ids}
n_real = len(WORDS6)
mask_id = n_real

rows = []
for ts in TRAIN_SEEDS:
    ckpt = os.path.join(OUT, f'ckpt_rand30_s{ts}.pt')
    t0 = time.time()
    if os.path.exists(ckpt):
        blob = torch.load(ckpt, weights_only=True)
        m_mlm, m_unif = ToySeqDenoiser(n_real + 1), ToySeqDenoiser(n_real)
        m_mlm.load_state_dict(blob['mlm']); m_unif.load_state_dict(blob['unif'])
        loss_mlm, loss_unif = blob['loss_mlm'], blob['loss_unif']
        print(f"  [C_rand30] 训练种子 {ts} 从缓存加载")
    else:
        torch.manual_seed(ts)
        m_mlm, loss_mlm = train_dfm(tpl_ids, n_real + 1, mask_id)
        m_unif, loss_unif = train_dfm(tpl_ids, n_real, None)
        torch.save({'mlm': m_mlm.state_dict(), 'unif': m_unif.state_dict(),
                    'loss_mlm': loss_mlm, 'loss_unif': loss_unif}, ckpt)
        print(f"  [C_rand30] 训练种子 {ts} 完成 ({time.time()-t0:.0f}s): CE mlm={loss_mlm:.4f} unif={loss_unif:.4f}")
    m_mlm.eval(); m_unif.eval()
    for ss in SAMPLE_SEEDS:
        torch.manual_seed(ss)
        fx_m, rw_m, ex_m = sample_euler(m_mlm, mask_id, n_real + 1)
        fx_u, rw_u, ex_u = sample_euler(m_unif, None, n_real)
        hm, hu = hit_rate(fx_m, tpl_set), hit_rate(fx_u, tpl_set)
        rows.append(dict(train_seed=ts, sample_seed=ss,
                         hit_mlm=round(hm, 4), hit_unif=round(hu, 4),
                         diff_mlm_minus_unif=round(hm - hu, 4),
                         rewrite_mlm=round(rw_m / max(ex_m, 1), 6),
                         rewrite_unif=round(rw_u / max(ex_u, 1), 6)))
        print(f"    采样种子 {ss}: MDLM命中={hm:.4f} 改写率={rw_m/max(ex_m,1):.4f} | "
              f"均匀命中={hu:.4f} 改写率={rw_u/max(ex_u,1):.4f} | MDLM-均匀={hm-hu:+.4f}")
n_win = sum(1 for r in rows if r['hit_mlm'] > r['hit_unif'])
n_tie = sum(1 for r in rows if r['hit_mlm'] == r['hit_unif'])
diffs = [r['diff_mlm_minus_unif'] for r in rows]
print(f"  [C_rand30] 汇总: MDLM胜 {n_win}/{len(rows)} (平 {n_tie}), "
      f"MDLM-均匀差值范围 [{min(diffs):+.4f}, {max(diffs):+.4f}]")

with open(os.path.join(OUT, 'verify_results_rand30.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(name='C_rand30', n_templates=30, corpus=TPLR, rows=rows,
                   mlm_wins=n_win, ties=n_tie, diff_min=min(diffs), diff_max=max(diffs)),
              f, ensure_ascii=False, indent=2)
print("结果已保存: verify_results_rand30.json")
