# -*- coding: utf-8 -*-
"""14.6-1 理论-实测一致性探针

问题：JSON 里 N=1（单步、t=0）的实测命中率 MDLM=0.0039(1/256)、均匀=0.0195(5/256)，
      而"因子化最优模型在 t=0 处 = 逐位置边缘分布"的理论命中率是 0.0057(期望 1.45/256)。
      MLDM 完全落在采样噪声内；均匀版 5/256 属 ~2% 尾部，需要判定是
      (a) 采样噪声，还是 (b) 训练出的均匀版模型本身偏离了理想边缘分布。

做法：精确复刻原脚本随机流训练两个模型 -> 在 t=0 处直接读取网络后验
      - 闭式命中率：对给定 x_t，一次采样命中模板的概率 = Σ_T Π_j p_θ(T_j | x_t, t=0)
      - 理想值同为 Σ_T Π_j p_marginal(T_j) = 0.0057
      - 并用多个种子各采 256 句，看"该模型"的经验命中率分布
"""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

WORDS = ['the', 'a', 'cat', 'dog', 'big', 'red', 'sat', 'ate', 'on',
         'mat', 'rug', 'fish', 'it', 'up']
MASK_ID = len(WORDS)          # 14
V_MASK, V_UNIF = 15, 14
TEMPLATES = [
    'the cat sat on the mat', 'the dog sat on the rug',
    'the cat ate the red fish', 'the dog ate the big fish',
    'a big cat sat on it', 'a big dog ate it up',
]
SEQ_LEN = 6
TEMPLATE_IDS = np.array([[WORDS.index(w) for w in s.split()] for s in TEMPLATES])
N_TEMPLATES = len(TEMPLATES)
TEMPLATE_SET = {tuple(r) for r in TEMPLATE_IDS}

# 理想（数据）逐位置边缘分布
p_ideal = []
for j in range(SEQ_LEN):
    c = Counter(int(r[j]) for r in TEMPLATE_IDS)
    p_ideal.append({k: v / N_TEMPLATES for k, v in c.items()})
p_ideal_hit = sum(float(np.prod([p_ideal[j].get(int(r[j]), 0.0) for j in range(SEQ_LEN)])) for r in TEMPLATE_IDS)
print(f"[理想] 因子化最优模型 N=1 命中率 = {p_ideal_hit:.5f} (期望 {p_ideal_hit*256:.2f}/256)")


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32) * (-np.log(10000.0) / (half - 1)))
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ToySeqDenoiser(nn.Module):
    def __init__(self, vocab_size, seq_len=SEQ_LEN, d_model=64, n_layers=2,
                 n_heads=2, ff_dim=128, time_dim=32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.time_dim = time_dim
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
                                       batch_first=True, dropout=0.0), num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, t):
        tf = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        h = self.tok_emb(x) + self.pos_emb.weight[None, :, :] + tf[:, None, :]
        return self.head(self.encoder(h))


def train(p_kind, n_steps=3000, batch_size=128, lr=3e-4):
    vocab = V_MASK if p_kind == 'mask' else V_UNIF
    model = ToySeqDenoiser(vocab)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, N_TEMPLATES, (batch_size,))
        x0 = torch.from_numpy(TEMPLATE_IDS[idx.numpy()])
        t = torch.rand(batch_size)
        keep = torch.rand(batch_size, SEQ_LEN) < t[:, None]
        if p_kind == 'mask':
            noise = torch.full_like(x0, MASK_ID)
        else:
            noise = torch.randint(0, V_UNIF, x0.shape)
        x_t = torch.where(keep, x0, noise)
        logits = model(x_t, t)
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), x0.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model, loss.item()


# ---- 精确复刻原脚本随机流 ----
np.random.seed(42); torch.manual_seed(42)
_ = ToySeqDenoiser(V_MASK)                    # line 161 的 N_PARAMS 哑模型
m_mlm, l_mlm = train('mask')
m_unif, l_unif = train('uniform')
print(f"[复刻] 末步CE: MLM={l_mlm:.4f} (JSON 0.202177) | 均匀={l_unif:.4f} (JSON 0.560180)")

hits = []
for tmpl in TEMPLATE_IDS:
    hits.append(tmpl)


def closed_form_hit(probs, tpl_ids):
    """probs: (B,d,V) -> 每样本一次采样命中任一模板的概率，返回 (B,)"""
    B = probs.shape[0]
    out = torch.zeros(B)
    for r in tpl_ids:
        p = torch.ones(B)
        for j in range(SEQ_LEN):
            p = p * probs[:, j, int(r[j])]
        out = out + p
    return out


# ===== MDLM: t=0, x = 全 [MASK] =====
m_mlm.eval()
with torch.no_grad():
    x0_mlm = torch.full((1, SEQ_LEN), MASK_ID, dtype=torch.long)
    p_mlm = torch.softmax(m_mlm(x0_mlm, torch.zeros(1)), dim=-1)[0]     # (d,V)
cf_mlm = float(closed_form_hit(p_mlm[None], TEMPLATE_IDS)[0])
print(f"\n[MDLM] t=0 处网络后验 -> 闭式 N=1 命中率 = {cf_mlm:.5f} (理想 {p_ideal_hit:.5f})")
for j in range(SEQ_LEN):
    top = torch.topk(p_mlm[j], 3)
    s = ", ".join(f"{WORDS[int(i)]}={float(v):.3f}" for v, i in zip(top.values, top.indices))
    ideal_s = ", ".join(f"{WORDS[k]}={v:.3f}" for k, v in sorted(p_ideal[j].items(), key=lambda kv: -kv[1]))
    print(f"   pos{j+1}: 网络 {s}  | 理想 {ideal_s}")

# ===== 均匀版: t=0, x ~ Unif =====
m_unif.eval()
with torch.no_grad():
    torch.manual_seed(11)
    x0_unif = torch.randint(0, V_UNIF, (4096, SEQ_LEN), dtype=torch.long)
    p_unif = torch.softmax(m_unif(x0_unif, torch.zeros(4096)), dim=-1)  # (B,d,V)
cf_unif = float(closed_form_hit(p_unif, TEMPLATE_IDS).mean())
print(f"\n[均匀版] t=0 处 4096 个随机起点 -> 平均闭式 N=1 命中率 = {cf_unif:.5f} (理想 {p_ideal_hit:.5f})")
p_avg = p_unif.mean(0)
for j in range(SEQ_LEN):
    top = torch.topk(p_avg[j], 3)
    s = ", ".join(f"{WORDS[int(i)]}={float(v):.3f}" for v, i in zip(top.values, top.indices))
    ideal_s = ", ".join(f"{WORDS[k]}={v:.3f}" for k, v in sorted(p_ideal[j].items(), key=lambda kv: -kv[1]))
    print(f"   pos{j+1}: 网络(对起点平均) {s}  | 理想 {ideal_s}")


# ===== 该模型下的经验命中率（多组 256 句）=====
def empirical(model, kind, seeds=(2027, 1, 2, 3, 4)):
    res = []
    vocab = V_MASK if kind == 'mask' else V_UNIF
    for sd in seeds:
        torch.manual_seed(sd)
        if kind == 'mask':
            x = torch.full((256, SEQ_LEN), MASK_ID, dtype=torch.long)
        else:
            x = torch.randint(0, V_UNIF, (256, SEQ_LEN), dtype=torch.long)
        with torch.no_grad():
            pr = torch.softmax(model(x, torch.zeros(256)), dim=-1)
        x = torch.multinomial(pr.reshape(-1, vocab), 1).reshape(x.shape)
        h = sum(1 for row in x.numpy() if tuple(row) in TEMPLATE_SET) / 256
        res.append(h)
    return res

print(f"\n[经验] MDLM  t=0 单步采样 256 句 × 5 种子: {empirical(m_mlm, 'mask')}   (JSON 实测 0.0039)")
print(f"[经验] 均匀版 t=0 单步采样 256 句 × 5 种子: {empirical(m_unif, 'uniform')}   (JSON 实测 0.0195)")
print(f"\n[结论] 均匀版闭式 {cf_unif:.5f} vs 理想 {p_ideal_hit:.5f}: "
      f"{'模型本身偏离理想边缘分布(与采样噪声无关)' if cf_unif > 1.6*p_ideal_hit else '模型≈理想边缘分布, 差异属采样噪声'}")