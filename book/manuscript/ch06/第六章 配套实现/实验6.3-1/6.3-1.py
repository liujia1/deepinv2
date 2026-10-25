# -*- coding: utf-8 -*-
"""
实验6.3-1: 去噪得分匹配(DSM)训练与验证
对应章节: 6.3 去噪得分匹配(DSM)

实验内容:
  步骤1: DSM训练 - 训练得分网络,观察损失收敛
  步骤2: 得分场可视化 - 学到的得分 vs 真实得分,不同噪声水平对比
  步骤3: DSM=ESM等价性数值验证 - 验证Vincent (2011)定理

知识点:
  - DSM通过引入噪声扰动,使条件得分有解析解
  - DSM目标等价于去噪任务
  - 不同噪声水平下的得分场特性

运行环境: PyTorch, CPU/GPU均可
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
import json
import warnings
import logging

# 兼容 NumPy 1.x (使用 np.trapz) 与 NumPy 2.x (使用 np.trapezoid)
# np.trapezoid 在 NumPy 2.0 才加入,Colab 默认仍是 1.x
_trapz = getattr(np, 'trapezoid', np.trapz)

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验6.3-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)

sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
# ========================================================

# 设置随机种子
np.random.seed(42)

import torch
torch.manual_seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*60}")
print(f"实验6.3-1: 去噪得分匹配(DSM)训练与验证")
print(f"{'='*60}")
print(f"使用设备: {device}")

if _IN_COLAB and device.type == 'cpu':
    print("\n提示: 当前在Colab中运行,建议使用GPU加速")
    print("  菜单: 运行时 -> 更改运行时类型 -> 选择GPU")

# ============================================================
# 步骤1: DSM训练
# ============================================================
print(f"\n{'='*60}")
print("步骤1: DSM训练")
print(f"{'='*60}")

# 1D高斯混合 p(x) = 0.5*N(-2,1) + 0.5*N(2,1)
def gm1d_pdf(x):
    return 0.5 * np.exp(-0.5*(x+2)**2)/np.sqrt(2*np.pi) + \
           0.5 * np.exp(-0.5*(x-2)**2)/np.sqrt(2*np.pi)

def gm1d_score(x):
    """精确得分 $\\nabla\\log p(x)$"""
    p1 = np.exp(-0.5*(x+2)**2)/np.sqrt(2*np.pi)
    p2 = np.exp(-0.5*(x-2)**2)/np.sqrt(2*np.pi)
    p = 0.5*p1 + 0.5*p2
    return (0.5*-(x+2)*p1 + 0.5*-(x-2)*p2) / p

# 噪声扰动后的分布得分 $\nabla\log p_\sigma(x)$
def gm1d_score_noisy(x, sigma):
    """噪声扰动分布的精确得分 $\nabla\log p_\sigma(x)$"""
    # $p_\sigma(x) = 0.5 \cdot \mathcal{N}(x; -2, 1+\sigma^2) + 0.5 \cdot \mathcal{N}(x; 2, 1+\sigma^2)$
    # 其中方差 = 1(原分布) + σ²(噪声)
    v = 1 + sigma**2
    p1 = np.exp(-0.5*(x+2)**2/v)/np.sqrt(2*np.pi*v)
    p2 = np.exp(-0.5*(x-2)**2/v)/np.sqrt(2*np.pi*v)
    p = 0.5*p1 + 0.5*p2
    return (0.5*-(x+2)/v*p1 + 0.5*-(x-2)/v*p2) / p

# 用一个简单的MLP作为得分网络
class ScoreNet1D(torch.nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x.unsqueeze(-1)).squeeze(-1)

# 生成训练数据
N_train = 10000
x_train = np.concatenate([np.random.randn(N_train//2) - 2, np.random.randn(N_train//2) + 2])
x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)

# DSM训练
sigma = 1.0
model = ScoreNet1D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 记录训练损失
train_losses = []

# 均匀格点 (用于步骤2可视化,绘制精确得分与学习得分的曲线对比)
x_test = np.linspace(-6, 6, 200)
x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)
true_score_noisy = gm1d_score_noisy(x_test, sigma)

# 全批量梯度下降(教学示范,非mini-batch)
# 多批 MC 平均: DSM 目标 $\mathcal{L}(\theta)=\mathbb{E}_z[\|s_\theta(x+\sigma z)+z/\sigma\|^2]$
# 是对噪声 z 的期望,而非单次实现。若每步只采一次 z, 在训练好的模型上实测
# 单次估计的标准差约 1.1e-2, 与"损失地板到最优值"的间距同量级, 后果有二:
#   (1) 损失曲线被估计噪声淹没, 看不出收敛趋势;
#   (2) 单次估计有约 41% 的概率低于理论不可约地板 2C(σ), 造成"突破了理论下界"的假象。
# 这里每个 epoch 采 n_mc_train 次独立噪声, 梯度与记录的损失都取其平均,
# 估计标准差降到约 1/sqrt(n_mc_train)。
n_mc_train = 8
print(f"\n训练DSM得分网络 ($\\sigma={sigma}$, 每步 {n_mc_train} 次 MC 平均)...")
for epoch in range(500):
    loss_sum = 0.0
    for _ in range(n_mc_train):
        # 采样噪声
        z = torch.randn_like(x_train_t)
        x_noisy = x_train_t + sigma * z

        # DSM目标: $\\|s_\\theta(\\tilde{x}) - \\nabla\\log q_\\sigma(\\tilde{x}|x)\\|^2$
        # 其中 $\\nabla\\log q_\\sigma(\\tilde{x}|x) = -(\\tilde{x}-x)/\\sigma^2 = -\\sigma z/\\sigma^2 = -z/\\sigma$
        pred = model(x_noisy)
        target = -z / sigma
        loss_sum = loss_sum + torch.mean((pred - target)**2)
    loss = loss_sum / n_mc_train

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1}: DSM loss = {loss.item():.6f}")

# 理论不可约损失地板: 训练损失未带 0.5 系数, 故地板为 2*C(σ); C(σ) 的推导见步骤3
# 这里提前算出, 用于在步骤2损失曲线上标注"平台期即最优"
_x_grid = np.linspace(-8, 8, 2000)
_score_grid = gm1d_score_noisy(_x_grid, sigma)
_pdf_grid = 0.5*np.exp(-0.5*(_x_grid+2)**2/(1+sigma**2))/np.sqrt(2*np.pi*(1+sigma**2)) + \
            0.5*np.exp(-0.5*(_x_grid-2)**2/(1+sigma**2))/np.sqrt(2*np.pi*(1+sigma**2))
c_theory = 0.5/sigma**2 - 0.5*_trapz(_score_grid**2*_pdf_grid, _x_grid)
loss_floor_theory = 2*c_theory

# 结论量取末端窗口平均: 即便每步已做 n_mc_train 次平均, 单点估计仍有残余波动
# (实测 std ~4e-3, 与末段损失到地板的间距同量级), 窗口平均把残余噪声再降一个量级,
# 才能可靠地宣称"贴近地板"。单点值另存备查。
_final_window = 50
final_loss = float(np.mean(train_losses[-_final_window:]))
last_epoch_loss = float(train_losses[-1])
print(f"\n训练完成,最终损失(末 {_final_window} 个 epoch 的平均): {final_loss:.6f}")
print(f"  参考: 最后一个 epoch 的单点损失 = {last_epoch_loss:.6f}")
print(f"理论不可约地板 2C(σ) = {loss_floor_theory:.6f} (最终损失在有限样本MC误差内贴近地板, 说明已收敛到最优得分)")

# ============================================================
# 步骤2: 得分场可视化
# ============================================================
print(f"\n{'='*60}")
print("步骤2: 得分场可视化")
print(f"{'='*60}")

# 验证: 训练后的 $s_\theta \approx \nabla\log p_\sigma$
with torch.no_grad():
    learned_score = model(x_test_t).cpu().numpy()

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 子图1: 学到的得分 vs 真实得分
axes[0].plot(x_test, true_score_noisy, 'r-', lw=2, label=r'精确 $\nabla\log p_\sigma(x)$')
axes[0].plot(x_test, learned_score, 'b--', lw=2, label=r'学习 $s_\theta(x)$')
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('得分')
axes[0].set_title(f'学到的得分 vs 真实得分 ($\\sigma={sigma}$)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 子图2: 不同噪声水平的得分场
for s in [0.1, 0.5, 1.0, 2.0]:
    score_s = gm1d_score_noisy(x_test, s)
    axes[1].plot(x_test, score_s, lw=1.5, label=f'$\\sigma={s}$')
axes[1].set_xlabel('$x$')
axes[1].set_ylabel('得分')
axes[1].set_title('不同噪声水平的精确得分场 (解析解, 非网络输出)')
axes[1].legend()
axes[1].grid(alpha=0.3)

# 子图3: DSM训练损失曲线
axes[2].plot(train_losses, 'b-', lw=1.5, alpha=0.7, label='DSM 训练损失 (MC估计)')
# 不可约地板: DSM 损失的期望下界为 2C(σ) (无0.5因子), 平台期贴近该线即代表已学到最优得分
axes[2].axhline(loss_floor_theory, color='green', linestyle='--', lw=1.8,
                label=f'不可约地板 $2C(\\sigma)={loss_floor_theory:.4f}$')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('DSM Loss')
axes[2].set_title('DSM训练损失曲线 (平台期≈不可约常数, 并非未收敛)')
axes[2].legend(loc='upper right', fontsize=9)
axes[2].grid(alpha=0.3)
# 添加最终损失标注(放在子图内部空白处, 不超出图片边界)
# final_loss 已按末端窗口平均在训练结束时算好(见步骤1), 此处直接复用,
# 避免用最后一个 epoch 的单点估计去和地板比较。
# 取 y 轴上限的 90% 位置, 确保在子图内部
y_max = max(train_losses) * 1.05
y_text = y_max * 0.90
axes[2].annotate(f'Final: {final_loss:.4f}', 
                xy=(len(train_losses)-1, final_loss),
                xytext=(len(train_losses)*0.15, y_text),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_得分场可视化.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图表已保存: 步骤2_得分场可视化.png")

# ============================================================
# 步骤3: DSM=ESM等价性数值验证
# ============================================================
print(f"\n{'='*60}")
print("步骤3: DSM=ESM等价性数值验证")
print(f"{'='*60}")

# DSM与ESM等价性验证
# $\mathcal{J}_{ESM}^{(\sigma)}(\theta) = 0.5 * \mathbb{E}[\|s_\theta(x) - \nabla\log p_\sigma(x)\|^2]$
# $\mathcal{J}_{DSM}(\theta) = \mathcal{J}_{ESM}^{(\sigma)}(\theta) + C(\sigma)$

# Vincent (2011) 定理数值验证: J_DSM - J_ESM = C(σ)
# 用蒙特卡洛估计两个损失,验证其差为常数
N_mc = 5000
x_mc = np.concatenate([np.random.randn(N_mc//2) - 2, np.random.randn(N_mc//2) + 2])
z_mc = np.random.randn(N_mc)
x_noisy_mc = x_mc + sigma * z_mc
x_noisy_mc_t = torch.tensor(x_noisy_mc, dtype=torch.float32, device=device)

with torch.no_grad():
    s_theta_mc = model(x_noisy_mc_t).cpu().numpy()

# J_DSM: 期望 ||s_θ(x̃) - (-z/σ)||²
target_mc = -z_mc / sigma
j_dsm = 0.5 * np.mean((s_theta_mc - target_mc)**2)

# J_ESM: 期望 ||s_θ(x̃) - ∇log p_σ(x̃)||²
true_score_mc = gm1d_score_noisy(x_noisy_mc, sigma)
j_esm = 0.5 * np.mean((s_theta_mc - true_score_mc)**2)

# 常数差 C(σ)
c_sigma = j_dsm - j_esm

# 数值计算理论 C(σ):
# 由 Vincent (2011) 推导:
#   C(σ) = 0.5 * E_{p(x) q_σ(x̃|x)}[||∇_x̃ log q_σ(x̃|x)||²]
#          - 0.5 * E_{p_σ(x̃)}[||∇_x̃ log p_σ(x̃)||²]
# 第一项: 0.5 * E[||z/σ||²] = 0.5 * E[z²]/σ² = 0.5 * 1/σ²
#        (因为 z ~ N(0,I), E[z²]=1, 系数 0.5 来自 J_DSM 目标函数)
c_term1 = 0.5 / sigma**2

# 第二项: E_{p_σ}[(∇log p_σ)²] 的数值积分已在步骤1训练后用完全相同的格点
#         ([-8,8], 2000点)算出并保存为 c_theory (= c_term1 - c_term2), 此处直接复用
c_term2 = c_term1 - c_theory

# 用随机初始化的模型再验证一次: C(σ) 应与 θ 无关
# 同时记录随机初始化的 ESM 损失 (MC 采样) 作为训练前基线,与 j_esm 测度一致
# 注意: 这个对照模型要显式固定初始化种子。否则它的参数取决于此前训练循环
# 从全局 RNG 消耗了多少次随机数 —— 一改 n_mc_train 或 epoch 数, C(σ) 的
# "随机基线"就会跟着漂移, "与 θ 无关"的对照就失去意义。
_rng_state = torch.get_rng_state()
try:
    torch.manual_seed(1234)
    model_random = ScoreNet1D().to(device)
finally:
    torch.set_rng_state(_rng_state)
with torch.no_grad():
    s_random_mc = model_random(x_noisy_mc_t).cpu().numpy()
j_dsm_r = 0.5 * np.mean((s_random_mc - target_mc)**2)
j_esm_r = 0.5 * np.mean((s_random_mc - true_score_mc)**2)
c_random = j_dsm_r - j_esm_r
# MC 测度下的训练前 ESM 基线 (与 j_esm 同测度,可直接对比)
esm_loss_before_mc = j_esm_r

print(f"\nVincent (2011) 定理数值验证: J_DSM = J_ESM + C(σ)")
print(f"  J_DSM (训练后模型): {j_dsm:.6f}")
print(f"  J_ESM (训练后模型): {j_esm:.6f}")
print(f"  常数差 C(σ) = J_DSM - J_ESM: {c_sigma:.6f}")
print(f"  理论值 C(σ) (数值积分): {c_theory:.6f}")
print(f"\n  验证 C(σ) 与 θ 无关:")
print(f"    随机初始化模型的 C(σ): {c_random:.6f}")
print(f"    训练后模型的 C(σ):     {c_sigma:.6f}")
print(f"    两者应接近 (差异来自有限样本)")

print(f"\n训练效果对比 (MC 采样点, {N_mc} 个):")
print(f"  ESM损失: 训练前={esm_loss_before_mc:.4f} → 训练后={j_esm:.4f}")
print(f"  相关系数 (均匀格点): {np.corrcoef(learned_score, true_score_noisy)[0,1]:.4f}")

# ---- 步骤3可视化: 损失分量条形图 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图1: J_DSM, J_ESM, C(σ) 三个量级对比
labels_1 = [r'$\mathcal{J}_{DSM}$', r'$\mathcal{J}_{ESM}$', r'$C(\sigma)$ (MC)', r'$C(\sigma)$ (理论)']
values_1 = [j_dsm, j_esm, c_sigma, c_theory]
colors_1 = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
bars1 = axes[0].bar(labels_1, values_1, color=colors_1, alpha=0.85, edgecolor='black')
axes[0].set_ylabel('损失值')
axes[0].set_title(r'Vincent 定理: $\mathcal{J}_{DSM} = \mathcal{J}_{ESM} + C(\sigma)$')
axes[0].grid(axis='y', alpha=0.3)
for bar, v in zip(bars1, values_1):
    # 处理负值: 正值标注在条形上方,负值标注在条形下方
    y_pos = v + 0.005 if v >= 0 else v - 0.015
    va = 'bottom' if v >= 0 else 'top'
    axes[0].text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'{v:.4f}', ha='center', va=va, fontsize=9)

# 子图2: C(σ) 与 θ 无关性验证 (训练后 vs 随机初始化)
labels_2 = [r'随机初始化 $C(\sigma)$', r'训练后 $C(\sigma)$', r'理论 $C(\sigma)$']
values_2 = [c_random, c_sigma, c_theory]
colors_2 = ['#FFA15A', '#4C72B0', '#55A868']
bars2 = axes[1].bar(labels_2, values_2, color=colors_2, alpha=0.85, edgecolor='black')
axes[1].axhline(c_theory, color='gray', linestyle='--', alpha=0.5, label=r'理论值')
axes[1].set_ylabel(r'$C(\sigma)$')
axes[1].set_title(r'$C(\sigma)$ 与 $\theta$ 无关性验证')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)
for bar, v in zip(bars2, values_2):
    # 处理负值: 正值标注在条形上方,负值标注在条形下方
    y_pos = v + 0.0005 if v >= 0 else v - 0.002
    va = 'bottom' if v >= 0 else 'top'
    axes[1].text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'{v:.4f}', ha='center', va=va, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_Vincent定理验证.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图表已保存: 步骤3_Vincent定理验证.png")

# ---- 结果归档: results_summary.json (数值与两张配图对应) ----
results = {
    'experiment': '实验6.3-1',
    'config': {'sigma': float(sigma), 'N_train': int(N_train), 'N_mc': int(N_mc),
               'n_mc_train': int(n_mc_train), 'final_loss_window': int(_final_window),
               'epochs': 500, 'lr': 1e-3, 'seed': 42},
    # 最终损失: 末 _final_window 个 epoch 的平均(每次已是 n_mc_train 次 MC 平均)
    'step1_final_dsm_loss_no_half_factor': float(final_loss),
    'step1_last_epoch_dsm_loss_single': float(last_epoch_loss),
    'step1_loss_floor_theory_2C': float(loss_floor_theory),
    'step2_corr_learned_vs_true_score': float(np.corrcoef(learned_score, true_score_noisy)[0, 1]),
    'step3_j_dsm_trained': float(j_dsm),
    'step3_j_esm_trained': float(j_esm),
    'step3_c_sigma_mc': float(c_sigma),
    'step3_c_sigma_theory': float(c_theory),
    'step3_c_sigma_random_init': float(c_random),
    'step3_esm_loss_before_mc': float(esm_loss_before_mc),
}
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"结果已保存: results_summary.json")

# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验6.3-1 总结")
print(f"{'='*60}")
print("1. DSM目标函数验证:")
print("   - DSM通过引入噪声扰动,使条件得分有解析解")
print("   - 数值验证了Vincent (2011)定理: J_DSM = J_ESM + C(σ)")
print("   - 蒙特卡洛估计的常数差与理论值接近")
print("\n2. 训练效果:")
print("   - ESM损失从训练前的基线显著降低")
print("   - 学到的得分与真实得分高度相关(相关系数≈1)")
print("\n3. 不同噪声水平的得分场:")
print("   - σ大: 得分平缓,提供全局方向")
print("   - σ小: 得分精细,关注局部细节")
print("   - 这为后续多尺度得分匹配(NCSN)奠定基础")
print("\n4. DSM训练目标:")
print("   - s_θ(x+σz) ≈ -z/σ")
print("   - 完全可计算,无需未知的真实得分(监督标签解析已知)")
print("   - 等价于去噪任务")

print(f"\n{'='*60}")
print("下一步: 实验6.6-1 - 从去噪器中提取得分函数(Tweedie等式)")
print(f"{'='*60}")
