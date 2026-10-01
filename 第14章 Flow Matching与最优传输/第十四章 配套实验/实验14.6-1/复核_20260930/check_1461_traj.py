# -*- coding: utf-8 -*-
"""
验证 14.6-1.py 当前代码与已存 results_summary.json 的揭示轨迹是否同代:
精确复刻原脚本随机流(seed=42 -> N_PARAMS哑模型实例化 -> 训练MDLM -> seed=2025 -> 单样本轨迹),
对比 JSON 中 reveal_trajectory(8快照) 与 reveal_details(6位置)。
"""
import sys, io, os, json
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
import torch
import torch.nn as nn

EXP_DIR = r"d:\works\deepinv\第14章 Flow Matching与最优传输\第十四章 配套实验\实验14.6-1"

WORDS = ['the', 'a', 'cat', 'dog', 'big', 'red', 'sat', 'ate', 'on',
         'mat', 'rug', 'fish', 'it', 'up']
MASK_TOKEN = '[MASK]'
VOCAB_MLM = WORDS + [MASK_TOKEN]
MASK_ID = len(WORDS)
V_MASK = len(VOCAB_MLM)
TEMPLATES = [
    'the cat sat on the mat', 'the dog sat on the rug',
    'the cat ate the red fish', 'the dog ate the big fish',
    'a big cat sat on it', 'a big dog ate it up',
]
word2id = {w: i for i, w in enumerate(WORDS)}
id2word = {i: w for i, w in enumerate(WORDS)}
SEQ_LEN = 6
TEMPLATE_IDS = np.array([[word2id[w] for w in s.split()] for s in TEMPLATES])

def decode(ids):
    return ' '.join(MASK_TOKEN if i == MASK_ID else id2word[int(i)] for i in ids)

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

# ===== 精确复刻原脚本随机流 =====
np.random.seed(42)
torch.manual_seed(42)
_ = ToySeqDenoiser(V_MASK)          # 原脚本 line 161: N_PARAMS 哑模型(消耗初始化随机流)
model = ToySeqDenoiser(V_MASK)      # train_dfm('mask') 内创建
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
model.train()
for step in range(1, 3001):
    idx = torch.randint(0, 6, (128,))
    x0 = torch.from_numpy(TEMPLATE_IDS[idx.numpy()])
    t = torch.rand(128)
    kappa_t = t[:, None]
    keep = torch.rand(128, SEQ_LEN) < kappa_t
    noise_tok = torch.full_like(x0, MASK_ID)
    x_t = torch.where(keep, x0, noise_tok)
    logits = model(x_t, t)
    loss = nn.functional.cross_entropy(logits.reshape(-1, V_MASK), x0.reshape(-1))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
model.eval()
print(f"训练完成, 末步CE={loss.item():.4f} (JSON final_loss_mlm=0.202177)")
_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ckpt_mlm_s42.pt')
torch.save(model.state_dict(), _ckpt)

torch.manual_seed(2025)
N_TRAJ = 32
with torch.no_grad():
    x = torch.full((1, SEQ_LEN), MASK_ID, dtype=torch.long)
    traj = [x.clone()]
    probs_rec = []
    for n in range(N_TRAJ):
        t_n = n / N_TRAJ
        rate = min((1.0 / N_TRAJ) * (1.0 / (1.0 - t_n)), 1.0)
        t_batch = torch.full((1,), t_n)
        probs = torch.softmax(model(x, t_batch), dim=-1)
        probs_rec.append(probs)
        flip = torch.rand(x.shape) < rate
        new_tok = torch.multinomial(probs.reshape(-1, V_MASK), 1).reshape(x.shape)
        x = torch.where(flip, new_tok, x)
        traj.append(x.clone())
traj = torch.stack(traj)             # (33,1,d)
probs_rec = torch.stack(probs_rec)   # (32,1,d,V)

snap_idx = [0, 2, 4, 6, 8, 12, 18, 32]
traj1 = traj[:, 0, :]
my_traj_snap = [decode(traj1[si]) for si in snap_idx]
final_ids = traj1[-1]
reveal_step = torch.full((SEQ_LEN,), N_TRAJ, dtype=torch.long)
for j in range(SEQ_LEN):
    trans = (traj1[:-1, j] == MASK_ID) & (traj1[1:, j] != MASK_ID)
    if trans.any():
        reveal_step[j] = int(trans.nonzero()[0])
probs1 = probs_rec[:, 0, :]
conf = probs1.gather(2, final_ids[None, :, None].expand(probs1.shape[0], -1, -1)).squeeze(-1)
my_details = [dict(pos=j + 1, token=id2word[int(final_ids[j])],
                   t_reveal=round(int(reveal_step[j]) / N_TRAJ, 4),
                   conf_at_reveal=round(float(conf[int(reveal_step[j]), j]), 4))
              for j in range(SEQ_LEN)]

with open(os.path.join(EXP_DIR, 'results_summary.json'), encoding='utf-8') as f:
    ref = json.load(f)

print("\n快照对比 (复现 vs JSON):")
ok = True
for a, b in zip(my_traj_snap, ref['reveal_trajectory']):
    mark = 'OK' if a == b else 'MISMATCH'
    if a != b: ok = False
    print(f"  [{mark}] {a}" + ("" if a == b else f"  <->  {b}"))
print("揭示详情对比:")
for a, b in zip(my_details, ref['reveal_details']):
    same = (a['pos'] == b['pos'] and a['token'] == b['token']
            and abs(a['t_reveal'] - b['t_reveal']) < 1e-9
            and abs(a['conf_at_reveal'] - b['conf_at_reveal']) < 5e-4)
    if not same: ok = False
    print(f"  [{'OK' if same else 'MISMATCH'}] {a}  <->  {b}")
print("\n结论:", "当前脚本与已存 JSON 同代(随机流逐位一致)" if ok else "当前脚本与已存 JSON 不同代(需重跑或修正声明)")
