# -*- coding: utf-8 -*-
"""
实验14.6-1 玩具级MDLM——离散Flow Matching的CTMC实现
对应章节: 14.6 离散扩散与掩码语言模型：从向量场到速率矩阵

知识点:
  - CTMC 与速率矩阵: 离散状态空间中没有 dX/dt，"驱动力"是跳变速率 Q_t(y|x)
  - 两条速率约束: 非对角 >= 0、对角 = 负离开总率（14.6.1）
  - 因子化混合路径: 每位置以 kappa_t 保留干净 token、以 1-kappa_t 换噪声（14.6.2）
  - 离散边际化技巧 + DFM 损失: 学后验 p_{1|t}，交叉熵训练，免仿真（14.6.3）
  - MDLM: 把 [MASK] 当噪声，生成 = 从全 [MASK] 逐步揭示（14.6.4）
  - softmax 参数化自动满足两条速率约束 + 欧拉步合法性: h·kdot/(1-kappa) <= 1

实验内容:
  - 玩具语言: 6 个固定长度(6词)模板句, 小词表 V=15(14 实词 + [MASK])
  - 训练两个 DFM 模型: MDLM 版(p_init=全[MASK]) 与 均匀噪声版(p_init=Unif)
  - 欧拉步采样: p~ = (1-h·h_t)·delta + h·h_t·p_theta, 等价于"以 rate 概率按后验重采"
  - 对比: 揭示热图+置信度可视化 / 模板命中率(生成质量) / 步数扫描 / 改写率
  - 数值验证: softmax 版 vs 原始参数版(不过softmax)的速率约束违反量

本实验不需要GPU，CPU几分钟可跑完。
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import logging
import warnings

# 静默 matplotlib 相关警告
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*U\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", message=".*Glyph.*")
warnings.filterwarnings("ignore", message=".*cmap.*")

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验14.6-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(_chinese_path, exist_ok=True)

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

np.random.seed(42)
import torch
torch.manual_seed(42)
import torch.nn as nn

# ============================================================
# 步骤0：玩具语言定义——6 个模板句，固定长度 d=6
# ============================================================
# 14 个实词 + [MASK]。语料只有这 6 个模板句，因此"生成质量"可以严格定量:
# 生成句与某个模板句逐词完全一致 => 合法(命中模板); 否则非法。
WORDS = ['the', 'a', 'cat', 'dog', 'big', 'red', 'sat', 'ate', 'on',
         'mat', 'rug', 'fish', 'it', 'up']
MASK_TOKEN = '[MASK]'
VOCAB_MLM = WORDS + [MASK_TOKEN]          # MDLM 版词表: 14 实词 + [MASK] = 15
MASK_ID = len(WORDS)                      # [MASK] 的 id = 14
VOCAB_UNIF = WORDS                        # 均匀噪声版词表: 只有 14 实词(无 [MASK])
V_MASK, V_UNIF = len(VOCAB_MLM), len(VOCAB_UNIF)

TEMPLATES = [
    'the cat sat on the mat',
    'the dog sat on the rug',
    'the cat ate the red fish',
    'the dog ate the big fish',
    'a big cat sat on it',
    'a big dog ate it up',
]
word2id = {w: i for i, w in enumerate(WORDS)}
id2word = {i: w for i, w in enumerate(WORDS)}

SEQ_LEN = 6
TEMPLATE_IDS = np.array([[word2id[w] for w in s.split()] for s in TEMPLATES])  # (6, 6)
TEMPLATE_SET = {tuple(row) for row in TEMPLATE_IDS}
N_TEMPLATES = len(TEMPLATES)

def decode(ids):
    """token id 序列 -> 可读字符串（[MASK] 显示为 ____）"""
    toks = [MASK_TOKEN if i == MASK_ID else id2word[int(i)] for i in ids]
    return ' '.join(toks)

# 自检: 模板长度、词表覆盖
assert all(len(t.split()) == SEQ_LEN for t in TEMPLATES), "模板句必须都是 6 词"
_used = sorted({w for t in TEMPLATES for w in t.split()})
assert _used == sorted(WORDS), "词表应恰好覆盖模板中出现的词"
print(f"[自检] 玩具语言: {N_TEMPLATES} 个模板句, 词表 {V_UNIF} 实词 + [MASK] = {V_MASK}, 序列长度 d={SEQ_LEN}")
for t in TEMPLATES:
    print(f"    {t}")

# ============================================================
# 步骤1：微型 Transformer 去噪分类器 p_{1|t}^theta
# ============================================================
def sinusoidal_embedding(t, dim):
    """正弦时间嵌入: t (B,) -> (B, dim)，标准 Transformer 位置编码公式"""
    half = dim // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32) * (-np.log(10000.0) / (half - 1)))
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ToySeqDenoiser(nn.Module):
    """逐位置去噪分类器: 输入含噪序列 x_t 与时间 t, 输出每位置 V 维 logits

    结构: 词嵌入 + 位置嵌入 + 正弦时间嵌入(MLP) -> 2层 TransformerEncoder -> 线性头
    网络输出经 softmax 即 p_{1|t}^theta(.|x_t) (14.6.3)
    """
    def __init__(self, vocab_size, seq_len=SEQ_LEN, d_model=64,
                 n_layers=2, n_heads=2, ff_dim=128, time_dim=32):
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
        """x: (B, d) int, t: (B,) float in [0,1] -> logits (B, d, V)"""
        time_feat = self.time_mlp(sinusoidal_embedding(t, self.time_dim))  # (B, d_model)
        h = self.tok_emb(x) + self.pos_emb.weight[None, :, :] + time_feat[:, None, :]
        h = self.encoder(h)
        return self.head(h)

N_PARAMS = sum(p.numel() for p in ToySeqDenoiser(V_MASK).parameters())
print(f"\n[模型] 微型 Transformer 去噪分类器: 参数量 {N_PARAMS:,}（2层, d_model=64）")


# ============================================================
# 步骤2：DFM 免仿真训练（14.6.3 训练算法的单步采样版）
# ============================================================
KAPPA = lambda t: t  # 线性调度 kappa_t = t: kappa_0=0, kappa_1=1, kdot=1
# hazard = kdot/(1-kappa) = 1/(1-t)（正文14.6.3: 风险率, t->1 时发散）

def train_dfm(p_init_kind, n_steps=3000, batch_size=128, lr=3e-4):
    """DFM 训练。p_init_kind: 'mask'(MDLM) 或 'uniform'(D3PM风格)

    每步: 采 x0 与 t -> 按因子化混合路径加噪 -> 全位置交叉熵(正文公式原样,
    已揭示位置贡献近零损失, 学习信号集中在噪声位置——14.6.4 的观察)
    """
    vocab_size = V_MASK if p_init_kind == 'mask' else V_UNIF
    model = ToySeqDenoiser(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for step in range(1, n_steps + 1):
        # 1. 采干净序列 x0 ~ p_data（此处 p_data = 6 个模板句的均匀分布）
        idx = torch.randint(0, N_TEMPLATES, (batch_size,))
        x0 = torch.from_numpy(TEMPLATE_IDS[idx.numpy()])          # (B, d)

        # 2. 采时间 t ~ U(0,1) 并计算调度
        t = torch.rand(batch_size)                                 # (B,)
        kappa_t = KAPPA(t)[:, None]                                # (B, 1)

        # 3. 因子化混合路径加噪: 每位置独立, 以 kappa_t 保留干净 token
        keep = torch.rand(batch_size, SEQ_LEN) < kappa_t           # (B, d)
        if p_init_kind == 'mask':
            noise_tok = torch.full_like(x0, MASK_ID)               # p_init = delta_[MASK]
        else:
            noise_tok = torch.randint(0, V_UNIF, x0.shape)         # p_init = Unif(实词表)
        x_t = torch.where(keep, x0, noise_tok)

        # 4. 网络前向 + 全位置交叉熵（L_DFM 的蒙特卡洛估计）
        logits = model(x_t, t)                                     # (B, d, V)
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size),
                                           x0.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 250 == 0 or step == 1:
            losses.append((step, loss.item()))
            if step % 1000 == 0:
                print(f"  [{p_init_kind:>7}] step {step}/{n_steps}  CE loss = {loss.item():.4f}")
    return model, losses


print("\n训练 MDLM 版(p_init = 全[MASK])...")
model_mlm, losses_mlm = train_dfm('mask')
print("训练 均匀噪声版(p_init = Unif)...")
model_unif, losses_unif = train_dfm('uniform')


# ============================================================
# 步骤3：欧拉步采样器（14.6.1 欧拉步的离散实现）
# ============================================================
# 连续 ODE 欧拉步: x_{t+h} = x_t + h·v_t(x_t)
# CTMC 欧拉步:    p~(y|x) = delta(y=x) + h·Q_t(y|x)
#                          = (1 - h·h_t)·delta(y=x) + h·h_t·p_theta(y|x)   [代入边际速率矩阵]
# 其中 h_t = kdot/(1-kappa) 是 hazard。关键观察:
#   p~ 是 delta 与 p_theta 的凸组合  <=>  h·h_t <= 1
#   -> "以概率 rate = h·h_t 按网络后验重采, 以概率 1-rate 保持"
# 线性调度 kappa_t = t + 均匀时间网格 t_n = n/N 时, 最大 h·h_t = (1/N)·(N/1) = 1
#   -> 恰好合法, 无需任何截断（正文 14.6.3 "采样实现中在 t 逼近 1 时截断步长"的落地）

@torch.no_grad()
def sample_euler(model, p_init_kind, n_steps=32, n_samples=256,
                 lock_revealed=False, record_traj=False):
    """欧拉步采样（CTMC 模拟）

    lock_revealed: True 时已揭示(非噪声)位置锁定不动——MDLM 的"只揭示不回改"实现。
                   False 时严格按边际速率矩阵欧拉步, 已揭示位置也允许小概率改写。
    record_traj:   记录完整轨迹 (n_steps+1, n_samples, d) 与逐步后验 (n_steps, n_samples, d, V),
                   用于揭示热图与置信度可视化。
    返回 (final_ids, n_rewrite, n_exposed, traj 或 None)，traj=(轨迹张量, 后验张量)
      n_rewrite: 已揭示 token 被改写的总次数; n_exposed: 已揭示 token 的暴露步·位总数
    """
    noise_id = MASK_ID if p_init_kind == 'mask' else -1  # 均匀版无噪声标记, 改写率统计全部位置
    vocab_size = V_MASK if p_init_kind == 'mask' else V_UNIF  # 两版词表大小不同(15 vs 14)
    model.eval()
    # 初始分布 p_init
    if p_init_kind == 'mask':
        x = torch.full((n_samples, SEQ_LEN), MASK_ID, dtype=torch.long)
    else:
        x = torch.randint(0, V_UNIF, (n_samples, SEQ_LEN), dtype=torch.long)

    traj = [x.clone()] if record_traj else None
    probs_rec = [] if record_traj else None
    h = 1.0 / n_steps
    n_rewrite, n_exposed = 0, 0
    for n in range(n_steps):
        t_n = n / n_steps                       # 当前时间(0 起步)
        hazard = 1.0 / (1.0 - t_n)              # 线性调度下的风险率 kdot/(1-kappa)
        rate = min(h * hazard, 1.0)             # = h·h_t, 线性调度下最大恰为 1(末步)
        t_batch = torch.full((n_samples,), t_n)
        probs = torch.softmax(model(x, t_batch), dim=-1)           # (B, d, V) 后验
        if record_traj:
            probs_rec.append(probs.cpu())
        # 每位置独立掷硬币: rate 概率按 p_theta 重采, 1-rate 概率保持
        flip = torch.rand(x.shape) < rate
        if not lock_revealed:
            n_exposed += int((x != noise_id).sum())
        new_tok = torch.multinomial(probs.reshape(-1, vocab_size), 1).reshape(x.shape)
        if not lock_revealed:
            n_rewrite += int(((x != new_tok) & flip & (x != noise_id)).sum())
        if lock_revealed:
            flip &= (x == noise_id)             # 已揭示位置锁定(MDLM 实践版)
        x = torch.where(flip, new_tok, x)
        if record_traj:
            traj.append(x.clone())
    traj = (torch.stack(traj), torch.stack(probs_rec)) if record_traj else None
    return x, n_rewrite, n_exposed, traj

# 合法性自检: 线性调度+均匀网格下 rate 最大值
_n = 32
_rates = [min((1.0/_n) * (1.0/(1.0 - n/_n)), 1.0) for n in range(_n)]
print(f"\n[自检] 欧拉步合法性: N={_n} 步均匀网格, rate = h·h_t ∈ [{min(_rates):.4f}, {max(_rates):.4f}]")
print(f"       最大 rate = {max(_rates):.4f} <= 1  ->  p~ 恒为凸组合, 欧拉步无截断即可采样")

# ============================================================
# 步骤4：揭示轨迹可视化——MDLM 从全 [MASK] 逐步揭示（14.6.4 表格的动态版）
# 文字快照由控制台 print 承担; 图只画 print 无法表达的信息:
#   (a) 时间×位置揭示热图: 色块=token, 单调性(落子无悔)与揭示顺序一眼可见
#   (b) 揭示置信度: 掩码位置在逐步更新中对"最终 token"的后验概率演化,
#       展示"上下文积累→置信度上升→被揭示"的机制(网络在"看上下文填空")
# ============================================================
print("\n步骤4: 采样一条揭示轨迹（MDLM, N=32 步欧拉步）...")
torch.manual_seed(2025)  # 固定轨迹采样种子, 保证图可复现（2024 落入尾部: 直到 t=0.56 仍 0/6 揭示, 不具代表性）
_, _, _, (traj_all, probs_all) = sample_euler(model_mlm, 'mask', n_steps=32, n_samples=1,
                                              lock_revealed=False, record_traj=True)
traj = traj_all[:, 0, :]      # (33, d) 含起点与每步更新后的状态
probs_rec = probs_all[:, 0, :]  # (32, d, V) 每步更新前的后验
snap_idx = [0, 2, 4, 6, 8, 12, 18, 32]  # 8 个快照时间步(仅用于 print 与 JSON)
print("  揭示轨迹(部分快照):")
for si in snap_idx:
    print(f"    t={si/32:.3f}  {decode(traj[si])}")

# --- 提取每个位置的揭示时刻与揭示置信度 ---
N_TRAJ = 32
final_ids = traj[-1]                                    # (d,)
reveal_step = torch.full((SEQ_LEN,), N_TRAJ, dtype=torch.long)  # 无揭示则记为末步
for j in range(SEQ_LEN):
    trans = (traj[:-1, j] == MASK_ID) & (traj[1:, j] != MASK_ID)  # MASK -> 实词 的跳变步
    if trans.any():
        reveal_step[j] = int(trans.nonzero()[0])        # 第 n 步更新后揭示 => 后验取自第 n 步(更新前)
# conf[n, j] = 第 n 步(更新前, 位置仍为 MASK 时)网络给"该位置最终 token"的后验概率
# 揭示发生在第 rs 步更新: traj[rs]=MASK, traj[rs+1]=实词, 落子用的后验正是 conf[rs]
conf = probs_rec.gather(2, final_ids[None, :, None].expand(probs_rec.shape[0], -1, -1)).squeeze(-1)  # (32, d)
print("  各位置揭示详情: 位置 | token | 揭示步 t | 揭示时置信度 p_θ")
for j in range(SEQ_LEN):
    rs = int(reveal_step[j])
    print(f"    pos {j+1}  {id2word[int(final_ids[j])]:>4}  t={rs/N_TRAJ:.3f}  conf={conf[rs, j]:.3f}")

# --- 图: (a) 揭示热图 + (b) 揭示置信度 ---
from matplotlib.colors import ListedColormap, BoundaryNorm
uniq_tokens = [MASK_ID] + sorted({int(t) for t in final_ids})   # 本轨迹出现的 token(含 MASK)
palette = ['#d9d9d9'] + [plt.cm.tab10(i / 10.0) for i in range(len(uniq_tokens) - 1)]
cmap_traj = ListedColormap(palette)
norm_traj = BoundaryNorm(np.arange(len(uniq_tokens) + 1) - 0.5, len(uniq_tokens))
tok2clr = {tok: i for i, tok in enumerate(uniq_tokens)}

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), width_ratios=[1.15, 1])
# (a) 时间×位置揭示热图: 行=位置, 列=欧拉步 n(状态为第 n 步更新后)
ax = axes[0]
grid = torch.stack([traj[n] for n in range(N_TRAJ + 1)], dim=1).numpy()      # (d, 33)
grid_clr = np.vectorize(lambda t: tok2clr[int(t)])(grid)
ax.imshow(grid_clr, cmap=cmap_traj, norm=norm_traj, aspect='auto',
          extent=[0, N_TRAJ, SEQ_LEN - 0.5, -0.5])
for j in range(SEQ_LEN):                                      # 在揭示时刻标注 token 名
    rs = int(reveal_step[j])
    ax.text(rs + 0.55, j, id2word[int(final_ids[j])], va='center', ha='left',
            fontsize=9, fontweight='bold', color='#333333')
ax.axvline(N_TRAJ - 0.5, color='k', lw=0.8, ls=':')
ax.set_yticks(range(SEQ_LEN))
ax.set_yticklabels([f'pos {j+1}' for j in range(SEQ_LEN)], fontsize=9)
ax.set_xticks([0, 8, 16, 24, 32])
ax.set_xticklabels([f'{k/32:.2f}' for k in [0, 8, 16, 24, 32]], fontsize=9)
ax.set_xlabel('时间 $t$（欧拉步 $n/32$）', fontsize=10)
ax.set_title('(a) 揭示热图: 色块=token, 灰=[MASK], 一旦揭示不回改', fontsize=11)
# (b) 揭示置信度: 掩码期间位置对"最终 token"的后验演化, 星号=被揭示时刻
ax = axes[1]
for j in range(SEQ_LEN):
    rs = int(reveal_step[j])
    ts = np.arange(rs + 1) / N_TRAJ                          # 揭示前各步(含落子那一步, 更新前仍为 MASK)
    ax.plot(ts, conf[:rs + 1, j], 'o-', ms=3.5, lw=1.2,
            color=palette[tok2clr[int(final_ids[j])]],
            label=f"pos {j+1}: {id2word[int(final_ids[j])]} ($t$={rs/N_TRAJ:.2f} 揭示)")
    ax.scatter([rs / N_TRAJ], [conf[rs, j]], marker='*', s=130, zorder=5,
               color=palette[tok2clr[int(final_ids[j])]], edgecolor='k', linewidth=0.5)
ax.set_xlabel('时间 $t$', fontsize=10)
ax.set_ylabel('掩码期间后验 $p_{1|t}^{\\theta}$（对最终 token）', fontsize=10)
ax.set_title('(b) 揭示置信度: 上下文积累→后验上升→星号处落子', fontsize=11)
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right')
fig.suptitle('实验14.6-1 步骤4: MDLM 生成 = 从全 [MASK] 逐步揭示（欧拉步 CTMC 模拟, $N=32$, seed=2025）',
             fontsize=12, y=1.00)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path_traj = os.path.join(SAVE_DIR, '步骤4_揭示热图与置信度.png')
plt.savefig(fig_path_traj, dpi=150, bbox_inches='tight')
plt.close()
print(f"图已保存: {fig_path_traj}")

# ============================================================
# 步骤5：均匀噪声 vs [MASK] 对比（14.6.4 对比表的定量版）
# ============================================================
def template_hit_rate(ids):
    """模板命中率: 生成句与 6 个模板之一完全一致的比例（生成质量指标）"""
    hit = sum(1 for row in ids.numpy() if tuple(row) in TEMPLATE_SET)
    return hit / len(ids)

N_STEPS_MAIN = 32
N_SAMPLES_EVAL = 256
print(f"\n步骤5: 两个模型各采 {N_SAMPLES_EVAL} 句（N={N_STEPS_MAIN} 步欧拉步）对比...")
torch.manual_seed(2025)
final_mlm, rw_mlm, exp_mlm, _ = sample_euler(model_mlm, 'mask', n_steps=N_STEPS_MAIN,
                                             n_samples=N_SAMPLES_EVAL, lock_revealed=False)
final_unif, rw_unif, exp_unif, _ = sample_euler(model_unif, 'uniform', n_steps=N_STEPS_MAIN,
                                                n_samples=N_SAMPLES_EVAL, lock_revealed=False)
hit_mlm = template_hit_rate(final_mlm)
hit_unif = template_hit_rate(final_unif)
rewrite_mlm = rw_mlm / max(exp_mlm, 1)
rewrite_unif = rw_unif / max(exp_unif, 1)
print(f"  MDLM   版: 模板命中率 {hit_mlm:.3f}, 已揭示token改写率 {rewrite_mlm:.4f}")
print(f"  均匀噪声版: 模板命中率 {hit_unif:.3f}, 已揭示token改写率 {rewrite_unif:.4f}")
print("  (均匀版每位置都可被反复改写; MDLM 版已揭示位置的后验集中在自身, 改写率应显著更低)")

# 步数扫描: 欧拉步数 N 对模板命中率的影响
STEPS_SWEEP = [1, 2, 4, 8, 16, 32, 64]
hit_mlm_sweep, hit_unif_sweep = [], []
print(f"\n  步数扫描（每点 {N_SAMPLES_EVAL} 句）:")
print(f"  {'N步':>4} | {'MDLM命中率':>10} | {'均匀版命中率':>11}")
for n_s in STEPS_SWEEP:
    torch.manual_seed(2026 + n_s)  # 每个步数独立播种, 保证可复现
    fx, _, _, _ = sample_euler(model_mlm, 'mask', n_steps=n_s, n_samples=N_SAMPLES_EVAL)
    hu, _, _, _ = sample_euler(model_unif, 'uniform', n_steps=n_s, n_samples=N_SAMPLES_EVAL)
    hm, huu = template_hit_rate(fx), template_hit_rate(hu)
    hit_mlm_sweep.append(hm); hit_unif_sweep.append(huu)
    print(f"  {n_s:>4} | {hm:>10.3f} | {huu:>11.3f}")

# 生成样本示例: 纯文字快照改由控制台 print 承担(图不排文字)
print("\n  生成样本示例（各 4 句, N=32）:")
print("    MDLM 版:")
for r in final_mlm.numpy()[:4]:
    print(f"      {decode(r)}")
print("    均匀噪声版:")
for r in final_unif.numpy()[:4]:
    print(f"      {decode(r)}")

# 模板覆盖统计: 每个模板被命中的句数——揭示是否模式坍塌, 是聚合命中率(b)看不到的信息
def template_counts(ids):
    arr = ids.numpy()
    return np.array([int((arr == tpl).all(axis=1).sum()) for tpl in TEMPLATE_IDS])
cnt_mlm = template_counts(final_mlm)
cnt_unif = template_counts(final_unif)
print("  模板编号: " + ", ".join(f'T{j+1}={TEMPLATES[j]}' for j in range(N_TEMPLATES)))
print(f"  MDLM 版各模板命中数:  {cnt_mlm.tolist()}")
print(f"  均匀版各模板命中数: {cnt_unif.tolist()}")

# 图: 对比总览
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
# (a) 模板覆盖: 各模板命中次数分组柱状(观察模式坍塌/覆盖均衡, print 难以直观呈现)
ax = axes[0][0]
xpos = np.arange(N_TEMPLATES); w = 0.38
ax.bar(xpos - w/2, cnt_mlm, w, color='#2a7de1', alpha=0.85, label='MDLM版(全[MASK])')
ax.bar(xpos + w/2, cnt_unif, w, color='#e1812c', alpha=0.85, label='均匀噪声版')
ax.set_xticks(xpos); ax.set_xticklabels([f'T{j+1}' for j in range(N_TEMPLATES)])
ax.set_xlabel('训练模板', fontsize=11)
ax.set_ylabel(f'命中句数（共 {N_SAMPLES_EVAL} 句）', fontsize=11)
ax.set_title('(a) 模板覆盖: 各模板命中次数（观察是否模式坍塌）', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
# (b) 模板命中率柱状
ax = axes[0][1]
bars = ax.bar(['MDLM版\n(全[MASK])', '均匀噪声版'],
              [hit_mlm, hit_unif], color=['#2a7de1', '#e1812c'], alpha=0.8, width=0.5)
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3, axis='y')
ax.set_ylabel(f'模板命中率（{N_SAMPLES_EVAL} 句, $N={N_STEPS_MAIN}$）', fontsize=11)
ax.set_title('(b) 生成质量: 命中 6 个训练模板的比例', fontsize=12)
for b, v in zip(bars, [hit_mlm, hit_unif]):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
# (c) 步数扫描
ax = axes[1][0]
ax.plot(STEPS_SWEEP, hit_mlm_sweep, 'o-', color='#2a7de1', label='MDLM版(全[MASK])')
ax.plot(STEPS_SWEEP, hit_unif_sweep, 's--', color='#e1812c', label='均匀噪声版')
ax.set_xscale('log', base=2); ax.set_xticks(STEPS_SWEEP); ax.set_xticklabels(STEPS_SWEEP)
ax.set_xlabel('欧拉步数 $N$（生成预算）', fontsize=11)
ax.set_ylabel('模板命中率', fontsize=11)
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=10)
ax.set_title('(c) 步数扫描: 生成质量随欧拉步数的变化', fontsize=12)
# (d) 改写率
ax = axes[1][1]
bars = ax.bar(['MDLM版\n(全[MASK])', '均匀噪声版'],
              [rewrite_mlm, rewrite_unif], color=['#2a7de1', '#e1812c'], alpha=0.8, width=0.5)
ax.grid(alpha=0.3, axis='y')
ax.set_ylabel('已揭示 token 改写率（$N=32$）', fontsize=11)
ax.set_title('(d) 轨迹形态: 揭示式 vs 改写式', fontsize=12)
for b, v in zip(bars, [rewrite_mlm, rewrite_unif]):
    ax.text(b.get_x() + b.get_width()/2, v + max(rewrite_mlm, rewrite_unif)*0.02 + 0.001,
            f'{v:.4f}', ha='center', fontweight='bold')
plt.suptitle('实验14.6-1 步骤5: 均匀噪声 vs [MASK]——$p_{\\mathrm{init}}$ 选择的定量对比', fontsize=13, y=1.00)
plt.tight_layout()
fig_path_cmp = os.path.join(SAVE_DIR, '步骤5_均匀vs掩码对比.png')
plt.savefig(fig_path_cmp, dpi=150, bbox_inches='tight')
plt.close()
print(f"图已保存: {fig_path_cmp}")

# ============================================================
# 步骤6：速率约束数值验证——softmax 参数化为何"免费合法"（14.6.3 红利）
# ============================================================
# 构造边际速率矩阵 Q_t(v,j|x_t) = h_t·(p_theta(v|x_t) - delta_{x_t,j}(v))
# 两条约束(14.6.1):
#   (1) 非对角(v != x_t,j) >= 0   <- softmax 输出 p_theta >= 0, 自动满足
#   (2) 对角 = -sum_{v != x_t,j} 非对角   <- 由构造恒等(减去同一个 delta 项)
# 对照组: "原始参数版"——网络直接输出未归一化分数当速率(不过 softmax),
#         展示两条约束如何被破坏, 以及欧拉步 p~ 如何随之非法。
print("\n步骤6: 速率约束与欧拉步合法性数值验证...")
torch.manual_seed(2027)
idx = torch.randint(0, N_TEMPLATES, (512,))
x0 = torch.from_numpy(TEMPLATE_IDS[idx.numpy()])
t = torch.rand(512)
kappa_t = KAPPA(t)[:, None]
keep = torch.rand(512, SEQ_LEN) < kappa_t
x_t = torch.where(keep, x0, torch.full_like(x0, MASK_ID))

model_mlm.eval()
with torch.no_grad():
    probs = torch.softmax(model_mlm(x_t, t), dim=-1)            # (B, d, V) 后验
    raw = model_mlm(x_t, t)                                      # 未过 softmax 的原始输出
delta = nn.functional.one_hot(x_t, V_MASK).float()               # delta_{x_t}(v)
hazard = (1.0 / (1.0 - t))[:, None, None]                        # (B,1,1)
Q_soft = hazard * (probs - delta)                                # softmax 版速率矩阵
Q_raw = hazard * (raw - delta)                                   # 原始参数版(野路子)

off_diag = ~nn.functional.one_hot(x_t, V_MASK).bool()            # 非对角掩码(v != x_t,j)
# 约束(1): 非对角非负
viol1_soft = float(torch.clamp(-Q_soft[off_diag], min=0).max())   # 最大负值(违反量)
viol1_raw = float(torch.clamp(-Q_raw[off_diag], min=0).max())
frac_neg_raw = float((Q_raw[off_diag] < 0).float().mean())        # 原始版非对角为负的比例
# 约束(2): 对角 = -Σ非对角  等价于  整行(对角+全部非对角)之和 = 0
# 直接对最后一维(V)求和即可; 若写成 Q[diag]+非对角和, 布尔掩码索引会把对角
# 展平成 1D(长度 B·d), 与 (B,d) 的非对角和无法广播而报错
viol2_soft = float(Q_soft.sum(-1).abs().max())
viol2_raw = float(Q_raw.sum(-1).abs().max())
# 欧拉步合法性: 末步 rate=1 处 p~ 的最小值与行和
rate_max = _rates[-1]  # 末步 rate(=1)
p_tilde_soft = (1 - rate_max) * delta + rate_max * probs
p_tilde_raw = delta + rate_max * Q_raw                           # 原始版欧拉步
min_pt_soft = float(p_tilde_soft.min())
rowsum_raw = float((p_tilde_raw.sum(-1) - 1).abs().max())        # 行和偏离 1
min_pt_raw = float(p_tilde_raw.min())
print(f"  约束(1) 非对角 >= 0    : softmax版违反量 {viol1_soft:.1e}(构造性恒 0: hazard>0 且 softmax>=0) | 原始版最大违反 {viol1_raw:.3f}(负速率比例 {frac_neg_raw:.1%})")
print(f"  约束(2) 对角守恒       : softmax版最大误差 {viol2_soft:.2e} | 原始版最大误差 {viol2_raw:.3f}")
print(f"  欧拉步 p~(末步 rate=1) : softmax版 min(p~) = {min_pt_soft:.2e} >= 0 且行和=1 合法 | 原始版 min = {min_pt_raw:.3f}, 行和最大偏离 {rowsum_raw:.3f} -> 非法")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
ax = axes[0]
bars = ax.bar(['softmax版\n(本书做法)', '原始参数版\n(不过softmax)'],
              [max(viol1_soft, 1e-16), viol1_raw], color=['#2a9d4a', '#d1495b'],
              alpha=0.85, width=0.5, log=True)
ax.set_ylabel('约束(1) 最大违反量: 非对角 $<0$（对数轴）', fontsize=10)
ax.set_title('(a) 离席速率非负', fontsize=12); ax.grid(alpha=0.3, axis='y')
for b, v, note in zip(bars, [viol1_soft, viol1_raw],
                      [f'{viol1_soft:.1e}\n（构造性恒 0）', f'{viol1_raw:.3f}\n({frac_neg_raw:.0%}速率$<0$)']):
    ax.text(b.get_x() + b.get_width()/2, max(v, 1e-16) * 1.8, note, ha='center', fontsize=9, fontweight='bold')  # 对数轴上 y=0 会把画布撑爆, 钳到下限
ax = axes[1]
bars = ax.bar(['softmax版', '原始参数版'], [max(viol2_soft, 1e-16), viol2_raw],
              color=['#2a9d4a', '#d1495b'], alpha=0.85, width=0.5, log=True)
ax.set_ylabel('约束(2) 最大误差: |对角+$\\Sigma$非对角|（对数轴）', fontsize=10)
ax.set_title('(b) 对角 = 负离开总率', fontsize=12); ax.grid(alpha=0.3, axis='y')
for b, v in zip(bars, [viol2_soft, viol2_raw]):
    ax.text(b.get_x() + b.get_width()/2, max(v, 1e-16) * 1.8, f'{v:.1e}', ha='center', fontsize=9, fontweight='bold')
ax = axes[2]
bars = ax.bar(['softmax版', '原始参数版'], [min_pt_soft, min_pt_raw],
              color=['#2a9d4a', '#d1495b'], alpha=0.85, width=0.5)
ax.axhline(0, color='k', linewidth=0.8)
ax.set_ylabel('欧拉步分布 $\\tilde{p}$ 的最小元素（末步 rate$=1$）', fontsize=10)
ax.set_title('(c) 欧拉步合法性: $\\tilde{p} \\geq 0$ 且行和$=1$', fontsize=12); ax.grid(alpha=0.3, axis='y')
for b, v, note in zip(bars, [min_pt_soft, min_pt_raw],
                      [f'{min_pt_soft:.1e}\n合法', f'{min_pt_raw:.3f}\n行和偏离{rowsum_raw:.3f}']):
    # 近 0 的正值柱若把标注放在 v+0.02 会超出轴顶压住标题, 改放到零线下方
    y_txt = v + 0.02 if v < 0 else -0.08 * abs(min_pt_raw)
    ax.text(b.get_x() + b.get_width()/2, y_txt, note, ha='center', fontsize=9, fontweight='bold')
plt.suptitle('实验14.6-1 步骤6: softmax 参数化自动满足两条速率约束——"学概率分布"的红利', fontsize=13, y=1.02)
plt.tight_layout()
fig_path_constraint = os.path.join(SAVE_DIR, '步骤6_速率约束验证.png')
plt.savefig(fig_path_constraint, dpi=150, bbox_inches='tight')
plt.close()
print(f"图已保存: {fig_path_constraint}")

# ============================================================
# 步骤7：训练损失曲线
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot([s for s, _ in losses_mlm], [l for _, l in losses_mlm], '-', color='#2a7de1', label='MDLM版 ($p_{\\mathrm{init}}$=全[MASK])')
ax.plot([s for s, _ in losses_unif], [l for _, l in losses_unif], '--', color='#e1812c', label='均匀噪声版 ($p_{\\mathrm{init}}$=Unif)')
ax.set_xlabel('训练步数', fontsize=11); ax.set_ylabel('DFM 损失（逐位置交叉熵）', fontsize=11)
ax.grid(alpha=0.3); ax.legend(fontsize=10)
ax.set_title('实验14.6-1 步骤7: DFM 训练收敛曲线（免仿真, 单步监督）', fontsize=12)
plt.tight_layout()
fig_path_loss = os.path.join(SAVE_DIR, '步骤7_训练损失.png')
plt.savefig(fig_path_loss, dpi=150, bbox_inches='tight')
plt.close()
print(f"图已保存: {fig_path_loss}")
print(f"  最终损失: MDLM版 {losses_mlm[-1][1]:.4f} | 均匀版 {losses_unif[-1][1]:.4f}")

# ============================================================
# 保存数值结果
# ============================================================
import json

def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    return obj

results_summary = {
    'toy_language': {
        'n_templates': N_TEMPLATES, 'seq_len': SEQ_LEN,
        'vocab_size_mlm': V_MASK, 'vocab_size_unif': V_UNIF,
        'templates': TEMPLATES,
    },
    'model_params': int(N_PARAMS),
    'schedule': 'kappa_t = t (线性), hazard = 1/(1-t)',
    'final_loss_mlm': round(float(losses_mlm[-1][1]), 6),
    'final_loss_unif': round(float(losses_unif[-1][1]), 6),
    'euler_validity': {
        'n_steps': 32, 'max_rate': round(float(max(_rates)), 4),
        'note': '线性调度+均匀网格下 rate=h*h_t 最大恰为1, 欧拉步分布恒为凸组合',
    },
    'reveal_trajectory': [decode(traj[si]) for si in snap_idx],
    'reveal_details': [
        {'pos': int(j + 1), 'token': id2word[int(final_ids[j])],
         't_reveal': round(int(reveal_step[j]) / N_TRAJ, 4),
         'conf_at_reveal': round(float(conf[int(reveal_step[j]), j]), 4)}
        for j in range(SEQ_LEN)
    ],
    'comparison_N32': {
        'template_hit_mlm': round(float(hit_mlm), 4),
        'template_hit_unif': round(float(hit_unif), 4),
        'rewrite_rate_mlm': round(float(rewrite_mlm), 6),
        'rewrite_rate_unif': round(float(rewrite_unif), 6),
    },
    'template_coverage': {
        'templates': TEMPLATES,
        'count_mlm': cnt_mlm.tolist(),
        'count_unif': cnt_unif.tolist(),
    },
    'steps_sweep': {
        'n_steps_list': STEPS_SWEEP,
        'hit_mlm': [round(float(v), 4) for v in hit_mlm_sweep],
        'hit_unif': [round(float(v), 4) for v in hit_unif_sweep],
    },
    'rate_constraint_check': {
        'viol1_nondiag_softmax': float(f"{viol1_soft:.3e}"),
        'viol1_nondiag_raw': round(float(viol1_raw), 6),
        'frac_neg_rate_raw': round(float(frac_neg_raw), 6),
        'viol2_diag_softmax': float(f"{viol2_soft:.3e}"),
        'viol2_diag_raw': round(float(viol2_raw), 6),
        'euler_min_ptilde_softmax': float(f"{min_pt_soft:.3e}"),
        'euler_rowsum_dev_raw': round(float(rowsum_raw), 6),
    },
}
results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"\n数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")

print("\n" + "=" * 62)
print("实验14.6-1 总结")
print("=" * 62)
print(f"""1. 免仿真训练（14.6.3）: DFM 损失 = 逐位置交叉熵, 单步监督即收敛
   (最终 CE: MDLM版 {losses_mlm[-1][1]:.4f} / 均匀版 {losses_unif[-1][1]:.4f}), 无需模拟任何 CTMC 轨迹

2. 生成 = CTMC 欧拉步（14.6.1）: p~ = (1-rate)·delta + rate·p_theta
   线性调度 + 均匀网格下 rate = h·h_t 最大恰为 1 -> 恒合法, 无需截断

3. 均匀噪声 vs [MASK]（14.6.4 对比表的定量版, N={N_STEPS_MAIN}）:
   - 模板命中率: MDLM {hit_mlm:.3f} vs 均匀 {hit_unif:.3f}
   - 已揭示 token 改写率: MDLM {rewrite_mlm:.4f} vs 均匀 {rewrite_unif:.4f}
   - MDLM = "从全[MASK]揭示", 均匀版 = "渐进改写"——轨迹形态肉眼可见不同

4. softmax 参数化自动合法（14.6.3 红利）:
   约束(1) 违反量 {viol1_soft:.1e}（构造性恒 0: hazard>0 且 softmax>=0）; 而不过 softmax 的原始参数
   有 {frac_neg_raw:.0%} 的"速率"为负、行和偏离 1 -> 欧拉步非法
   => "学概率分布而不是学速率"的设计在数值上兑现红利""")
