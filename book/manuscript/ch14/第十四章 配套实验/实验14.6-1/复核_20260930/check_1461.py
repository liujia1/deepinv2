# -*- coding: utf-8 -*-
"""
实验14.6-1 复核脚本——验证书中"复核裁决"的多种子声明是否真实可复现
待验证声明:
  A(6模板): 均匀版命中率在 3训练种子×2采样种子 下稳定略优于 MDLM(优势 +0.02~+0.14)
  B(30模板, 困难语料): MDLM 在大多数种子组合下反超 (书中数字: 12组中11组, N=32: 0.69 vs 0.55)
附带: 复现原实验 seed=42 主对比(0.9375/0.9766)与最终损失(0.202177/0.56018), 确认本地环境一致性

模型/训练/采样代码与 14.6-1.py 逐行一致(仅去掉绘图与Colab分支)。
30模板语料构造规则(确定性, 可复现):
  基于原玩具语法扩展: det{the,a} × subj{cat,dog} × {sat+on the X(4个), ate+the Y fish(4个)} = 32
  加上 det{the,a} × subj{big cat,big dog} × {sat on it, ate it up} = 8, 共40个合法6词句,
  用固定种子 RandomState(20260930) 无放回抽 30 个作为语料。新增实词: bed sofa small yellow。
"""
import sys, io, os, json, time
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
import torch
import torch.nn as nn

OUT = os.path.dirname(os.path.abspath(__file__))

# ================= 与 14.6-1.py 完全一致的部分 =================
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
    """与原脚本一致: t~U(0,1), kappa=t, keep< kappa_t, 全位置CE"""
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
        kappa_t = t[:, None]
        keep = torch.rand(batch_size, seq_len) < kappa_t
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
    """与原脚本一致: rate=min(h*hazard,1), hazard=1/(1-t_n); mask版只统计已揭示位置的改写"""
    if mask_id is not None:
        x = torch.full((n_samples, seq_len), mask_id, dtype=torch.long)
    else:
        x = torch.randint(0, vocab_size, (n_samples, seq_len), dtype=torch.long)
    h = 1.0 / n_steps
    n_rewrite = n_exposed = 0
    for n in range(n_steps):
        t_n = n / n_steps
        hazard = 1.0 / (1.0 - t_n)
        rate = min(h * hazard, 1.0)
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

# ================= 语料 =================
TPL6 = [
    'the cat sat on the mat', 'the dog sat on the rug',
    'the cat ate the red fish', 'the dog ate the big fish',
    'a big cat sat on it', 'a big dog ate it up',
]
WORDS6 = ['the', 'a', 'cat', 'dog', 'big', 'red', 'sat', 'ate', 'on',
          'mat', 'rug', 'fish', 'it', 'up']

# 30模板语料: 40个合法句按固定规则枚举, 固定种子抽30
_grammar = []
for det in ['the', 'a']:
    for subj in ['cat', 'dog']:
        for obj in ['on the mat', 'on the rug', 'on the bed', 'on the sofa']:
            _grammar.append(f'{det} {subj} sat {obj}')
        for obj in ['the red fish', 'the big fish', 'the small fish', 'the yellow fish']:
            _grammar.append(f'{det} {subj} ate {obj}')
    for subj in ['big cat', 'big dog']:
        _grammar.append(f'{det} {subj} sat on it')
        _grammar.append(f'{det} {subj} ate it up')
assert len(_grammar) == 40 and len(set(_grammar)) == 40
_sel = np.random.RandomState(20260930).choice(40, 30, replace=False)
TPL30 = sorted(_grammar[i] for i in _sel)
WORDS30 = sorted({w for t in TPL30 for w in t.split()})
assert all(len(t.split()) == 6 for t in TPL30)
print(f"[语料] 30模板语料构造完成: 40个合法句中抽30 (seed=20260930), 实词数 {len(WORDS30)}")
for t in TPL30[:5]:
    print(f"    {t}")
print("    ...")

TRAIN_SEEDS = [42, 7, 123]
SAMPLE_SEEDS = [2025, 2026]
N_STEPS = 32
N_SAMPLES = 256

def run_block(name, tpl_list, words):
    word2id = {w: i for i, w in enumerate(words)}
    tpl_ids = np.array([[word2id[w] for w in s.split()] for s in tpl_list])
    tpl_set = {tuple(r) for r in tpl_ids}
    n_real = len(words)
    mask_id = n_real          # mask版: V = n_real+1, [MASK]=n_real
    rows = []
    for ts in TRAIN_SEEDS:
        t0 = time.time()
        torch.manual_seed(ts)
        m_mlm, loss_mlm = train_dfm(tpl_ids, n_real + 1, mask_id)
        m_unif, loss_unif = train_dfm(tpl_ids, n_real, None)
        dt = time.time() - t0
        print(f"  [{name}] 训练种子 {ts} 完成 ({dt:.0f}s): CE mlm={loss_mlm:.4f} unif={loss_unif:.4f}")
        for ss in SAMPLE_SEEDS:
            # 与原脚本协议一致: 一次播种, 先 mlm 后 unif(共享随机流)
            torch.manual_seed(ss)
            fx_m, rw_m, ex_m = sample_euler(m_mlm, mask_id, n_real + 1, n_steps=N_STEPS, n_samples=N_SAMPLES)
            fx_u, rw_u, ex_u = sample_euler(m_unif, None, n_real, n_steps=N_STEPS, n_samples=N_SAMPLES)
            hm, hu = hit_rate(fx_m, tpl_set), hit_rate(fx_u, tpl_set)
            rr_m, rr_u = rw_m / max(ex_m, 1), rw_u / max(ex_u, 1)
            rows.append(dict(train_seed=ts, sample_seed=ss,
                             hit_mlm=round(hm, 4), hit_unif=round(hu, 4),
                             diff_unif_minus_mlm=round(hu - hm, 4),
                             rewrite_mlm=round(rr_m, 6), rewrite_unif=round(rr_u, 6),
                             loss_mlm=round(loss_mlm, 6), loss_unif=round(loss_unif, 6)))
            print(f"    采样种子 {ss}: MDLM命中={hm:.4f} 改写率={rr_m:.4f} | 均匀命中={hu:.4f} 改写率={rr_u:.4f} | 均匀-MDLM={hu-hm:+.4f}")
    n_win = sum(1 for r in rows if r['hit_mlm'] > r['hit_unif'])
    n_tie = sum(1 for r in rows if r['hit_mlm'] == r['hit_unif'])
    diffs = [r['diff_unif_minus_mlm'] for r in rows]
    print(f"  [{name}] 汇总: MDLM胜 {n_win}/{len(rows)} (平 {n_tie}), 均匀-MDLM差值范围 [{min(diffs):+.4f}, {max(diffs):+.4f}]")
    return dict(name=name, n_templates=len(tpl_list), rows=rows,
                mlm_wins=n_win, ties=n_tie,
                diff_min=min(diffs), diff_max=max(diffs))

results = {'meta': dict(torch=torch.__version__, n_samples=N_SAMPLES, n_steps=N_STEPS,
                        train_seeds=TRAIN_SEEDS, sample_seeds=SAMPLE_SEEDS,
                        tpl30_corpus=TPL30, tpl30_words=WORDS30)}
print("\n===== 块A: 原版6模板语料 =====")
results['A_6tpl'] = run_block('A_6tpl', TPL6, WORDS6)
print("\n===== 块B: 30模板困难语料 =====")
results['B_30tpl'] = run_block('B_30tpl', TPL30, WORDS30)

# N=1 理论值(6模板): 全MASK、真后验=逐位置边缘分布、逐位置独立采样时的精确命中率
tpl_arr = np.array([[WORDS6.index(w) for w in t.split()] for t in TPL6])
p_theory = 0.0
for i in range(len(TPL6)):
    p = 1.0
    for j in range(6):
        p *= float((tpl_arr[:, j] == tpl_arr[i, j]).sum()) / len(TPL6)
    p_theory += p
results['N1_theory_hit_6tpl'] = round(p_theory, 4)
print(f"\n[理论] 6模板、N=1(真后验逐位置独立采样)的精确命中率 = {p_theory:.4f} (实测原实验 0.0039 -> 模型在t≈0过自信, 非算法错误)")

with open(os.path.join(OUT, 'verify_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存: {os.path.join(OUT, 'verify_results.json')}")
