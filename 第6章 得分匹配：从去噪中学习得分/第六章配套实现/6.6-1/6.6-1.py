# -*- coding: utf-8 -*-
"""
实验6.6-1: 从去噪器中提取得分函数(Tweedie等式实践)
对应章节: 6.6 去噪器作为得分估计器: 实践与架构

实验内容:
  步骤1: DSM训练去噪器 - 训练一个去噪网络
  步骤2: Tweedie等式实践 - 从去噪器提取得分函数
  步骤3: 去噪效果验证 - 测试去噪性能与误差分布

知识点:
  - Tweedie等式: $\\nabla\\log p_\\sigma(x) = (D_\\sigma(x) - x) / \\sigma^2$
  - 去噪器与得分函数的等价关系
  - 从去噪器到得分估计器的转换

运行环境: PyTorch, CPU/GPU均可
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys
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
    SAVE_DIR = os.path.join(_gdrive, '实验6.6-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
else:
    try:
        SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SAVE_DIR = os.getcwd()
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')

os.makedirs(_chinese_path, exist_ok=True)

# 在Colab或本地首次运行时自动创建chinese_font.py
_chinese_font_path = os.path.join(_chinese_path, 'chinese_font.py')
if not os.path.exists(_chinese_font_path):
    print("正在创建中文字体配置模块...")
    _chinese_font_code = '''# -*- coding: utf-8 -*-
"""
中文显示支持模块 - 兼容 Windows / Linux / Colab
"""
import os
import sys
import platform
import warnings
import logging
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*U\\\\+2212.*")
warnings.filterwarnings("ignore", message=".*glyph.*")
plt.rcParams['axes.unicode_minus'] = False

def _find_chinese_font():
    candidates = []
    if platform.system() == 'Windows':
        candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    else:
        candidates = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK', 'Source Han Sans SC', 'AR PL UMing CN', 'SimHei']
    fm = FontManager()
    available = set(f.name for f in fm.ttflist)
    for font in candidates:
        if font in available:
            return font
    import re
    cjk_patterns = ['cjk', 'wqy', 'noto.*cjk', 'wenquan', 'chinese', 'simhei']
    for f in fm.ttflist:
        name_lower = f.name.lower()
        fname_lower = (os.path.basename(f.fname) if hasattr(f, 'fname') else '').lower()
        for pat in cjk_patterns:
            if re.search(pat, name_lower) or re.search(pat, fname_lower):
                return f.name
    return None

def setup_chinese_font(save_dir=None):
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    _cn_font = _find_chinese_font()
    if _cn_font:
        plt.rcParams['font.sans-serif'] = [_cn_font] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['font.family'] = 'sans-serif'
        print(f"[Font] 已检测到中文字体: {_cn_font}")
        return _cn_font
    if platform.system() != 'Windows':
        _font_url = 'https://github.com/jsntn/webfonts/raw/master/NotoSansSC-Regular.ttf'
        _font_file = os.path.join(save_dir, 'NotoSansSC-Regular.ttf')
        if os.path.exists(_font_file):
            from matplotlib.font_manager import fontManager
            fontManager.addfont(_font_file)
            plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['font.family'] = 'sans-serif'
            print(f"[Font] 已加载缓存字体: Noto Sans SC")
            return 'Noto Sans SC'
        else:
            try:
                import urllib.request
                print(f"[Font] 正在下载中文字体 NotoSansSC...")
                urllib.request.urlretrieve(_font_url, _font_file)
                from matplotlib.font_manager import fontManager
                fontManager.addfont(_font_file)
                plt.rcParams['font.sans-serif'] = ['Noto Sans SC'] + plt.rcParams.get('font.sans-serif', [])
                plt.rcParams['font.family'] = 'sans-serif'
                print(f"[Font] 已下载并注册中文字体: Noto Sans SC")
                return 'Noto Sans SC'
            except Exception as e:
                print(f"[Font] 字体下载失败: {e}")
    else:
        print("[Font] 未找到中文字体")
    return None

__all__ = ['setup_chinese_font']
'''
    with open(_chinese_font_path, 'w', encoding='utf-8') as f:
        f.write(_chinese_font_code)
    print(f"[Font] 已创建字体配置模块: {_chinese_font_path}")

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
print(f"实验6.6-1: 从去噪器中提取得分函数(Tweedie等式实践)")
print(f"{'='*60}")
print(f"使用设备: {device}")

if _IN_COLAB and device.type == 'cpu':
    print("\n提示: 当前在Colab中运行,建议使用GPU加速")
    print("  菜单: 运行时 -> 更改运行时类型 -> 选择GPU")

# Checkpoint路径
CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'denoiser_checkpoint.pth')

# ============================================================
# 步骤1: DSM训练去噪器
# ============================================================
print(f"\n{'='*60}")
print("步骤1: DSM训练去噪器")
print(f"{'='*60}")

# 1D高斯混合 p(x) = 0.5*N(-2,1) + 0.5*N(2,1)

# 噪声扰动后的分布得分 $\\nabla\\log p_\\sigma(x)$
def gm1d_score_noisy(x, sigma):
    """噪声扰动分布的精确得分 $\\nabla\\log p_\\sigma(x)$"""
    v = 1 + sigma**2
    p1 = np.exp(-0.5*(x+2)**2/v)/np.sqrt(2*np.pi*v)
    p2 = np.exp(-0.5*(x-2)**2/v)/np.sqrt(2*np.pi*v)
    p = 0.5*p1 + 0.5*p2
    return (0.5*-(x+2)/v*p1 + 0.5*-(x-2)/v*p2) / p

# 去噪器网络: 输入含噪样本, 输出预测的干净样本
class DenoiserNet(torch.nn.Module):
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

# 训练配置
sigma = 1.0
model = DenoiserNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 检查是否存在已训练好的模型
# 注: 当前代码未绘制训练损失曲线,故未保存 train_losses 到 checkpoint
# 若后续扩展需要训练曲线,可在 checkpoint 中添加 'train_losses' 字段
start_epoch = 0
is_final = False

if os.path.exists(CHECKPOINT_PATH):
    print(f"\n检测到已保存的模型: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    # 检查是否为最终训练完成的模型
    if checkpoint.get('is_final', False):
        print(f"✓ 这是最终训练完成的模型,直接加载,跳过训练过程")
        print(f"  训练轮数: {checkpoint['epoch']+1}")
        print(f"  最终损失: {checkpoint['loss']:.6f}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        is_final = True
    else:
        print(f"检测到未完成的训练,从第 {checkpoint['epoch']+1} 轮继续")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

# 训练去噪器
if not is_final:
    print(f"\n训练去噪器网络 ($\\sigma={sigma}$)...")
    N_epochs = 500
    
    for epoch in range(start_epoch, N_epochs):
        # 采样噪声
        z = torch.randn_like(x_train_t)
        x_noisy = x_train_t + sigma * z
        
        # 去噪目标: E[||D_σ(x̃) - x||²]，等价于 DSM 目标（见Vincent 2011）
        # 通过 Tweedie 等式可从去噪器 D_σ 恢复得分: s = (D_σ(x̃) - x̃) / σ²
        pred = model(x_noisy)
        target = x_train_t  # 目标是预测干净样本
        loss = torch.mean((pred - target)**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{N_epochs}: Denoiser loss = {loss.item():.6f}")
            
            # 保存checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'is_final': False
            }, CHECKPOINT_PATH)
    
    # 保存最终模型
    torch.save({
        'epoch': N_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'is_final': True
    }, CHECKPOINT_PATH)
    print(f"\n✓ 训练完成,模型已保存: {CHECKPOINT_PATH}")

# ============================================================
# 步骤2: Tweedie等式实践 - 从去噪器提取得分函数
# ============================================================
print(f"\n{'='*60}")
print("步骤2: Tweedie等式实践 - 从去噪器提取得分函数")
print(f"{'='*60}")

print("\nTweedie等式:")
print("  $\\nabla\\log p_\\sigma(\\tilde{x}) = \\frac{D_\\sigma(\\tilde{x}) - \\tilde{x}}{\\sigma^2}$")
print("\n其中:")
print("  - $D_\\sigma(\\tilde{x})$: 去噪器输出(预测的干净样本)")
print("  - $\\tilde{x}$: 含噪输入")
print("  - $\\sigma$: 噪声水平")
print("  - $\\nabla\\log p_\\sigma(\\tilde{x})$: 噪声扰动分布的得分")

# 用于可视化和评估的有序网格点（避免随机噪声导致曲线混乱）
x_grid = np.linspace(-6, 6, 200)
true_score_grid = gm1d_score_noisy(x_grid, sigma)

# 通过 Tweedie 等式从去噪器计算得分
with torch.no_grad():
    x_grid_t = torch.tensor(x_grid, dtype=torch.float32, device=device)
    denoised_grid = model(x_grid_t).cpu().numpy()
    # Tweedie: score = (D(x) - x) / sigma^2
    learned_score_grid = (denoised_grid - x_grid) / sigma**2

# 评估（使用与可视化相同的网格点）
corr = np.corrcoef(learned_score_grid, true_score_grid)[0, 1]
mse = np.mean((learned_score_grid - true_score_grid)**2)

print(f"\n得分估计质量评估:")
print(f"  相关系数: {corr:.4f}")
print(f"  MSE: {mse:.6f}")

# ============================================================
# 步骤3: 去噪效果验证
# ============================================================
print(f"\n{'='*60}")
print("步骤3: 去噪效果验证")
print(f"{'='*60}")

# 从 p(x) 中采样更多测试点（避免 x=0 处两个高斯分量中间点的不确定性）
np.random.seed(123)  # 固定种子保证可复现
N_demo = 50
x_clean_test = np.concatenate([np.random.randn(N_demo//2) - 2, np.random.randn(N_demo//2) + 2])
# 演示噪声 std=1.0，与去噪器训练的 sigma=1.0 保持一致
# 这样 Tweedie 公式 score = (D_σ(x̃) - x̃) / σ² 才正确应用
z_noise = np.random.randn(N_demo)
x_noisy_demo = x_clean_test + sigma * z_noise

# 去噪
with torch.no_grad():
    x_noisy_t = torch.tensor(x_noisy_demo, dtype=torch.float32, device=device)
    denoised_demo = model(x_noisy_t).cpu().numpy()

# 通过Tweedie等式提取得分
score_demo = (denoised_demo - x_noisy_demo) / sigma**2

# 打印前10个样本作为示例
print(f"\n去噪效果 ($\\sigma={sigma}$, 展示前10个样本):")
print(f"{'原始x':>8s} | {'含噪x̃':>8s} | {'去噪D_σ':>8s} | {'得分s_θ':>8s} | {'误差':>8s}")
print("-" * 60)
for i in range(min(10, len(x_clean_test))):
    err = denoised_demo[i] - x_clean_test[i]
    print(f"{x_clean_test[i]:8.3f} | {x_noisy_demo[i]:8.3f} | {denoised_demo[i]:8.3f} | {score_demo[i]:8.3f} | {err:8.3f}")

print("\nTweedie等式验证:")
print("  去噪器: $D_\\sigma(\\tilde{x}) = \\tilde{x} + \\sigma^2 \\cdot s_\\theta(\\tilde{x})$")
print("  得分函数: $s_\\theta(\\tilde{x}) = \\frac{D_\\sigma(\\tilde{x}) - \\tilde{x}}{\\sigma^2}$")
print("  这正是Tweedie等式 $\\nabla\\log p_\\sigma(\\tilde{x}) = \\frac{D_\\sigma(\\tilde{x}) - \\tilde{x}}{\\sigma^2}$ 的实践体现")

# ============================================================
# 可视化
# ============================================================
print(f"\n{'='*60}")
print("生成可视化图表...")
print(f"{'='*60}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: 去噪器输出 vs 真实干净样本
axes[0, 0].scatter(x_noisy_demo, x_clean_test, c='blue', s=50, alpha=0.6, label='真实干净样本')
axes[0, 0].scatter(x_noisy_demo, denoised_demo, c='red', s=50, alpha=0.6, marker='x', label='去噪器输出')
axes[0, 0].plot([-6, 6], [-6, 6], 'k--', alpha=0.3, label='理想去噪')
axes[0, 0].set_xlabel('含噪输入 $\\tilde{x}$')
axes[0, 0].set_ylabel('输出')
axes[0, 0].set_title(f'去噪效果 ($\\sigma={sigma}$)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 子图2: 学习到的得分 vs 真实得分（使用有序网格，避免随机噪声导致曲线混乱）
axes[0, 1].plot(x_grid, true_score_grid, 'r-', lw=2, label=r'真实 $\nabla\log p_\sigma(x)$')
axes[0, 1].plot(x_grid, learned_score_grid, 'b--', lw=2, label=r'学习 $s_\theta(x)$ (Tweedie)')
axes[0, 1].set_xlabel('$\\tilde{x}$')
axes[0, 1].set_ylabel('得分')
axes[0, 1].set_title(f'Tweedie等式: 从去噪器提取得分 (相关系数={corr:.3f})')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 子图3: 去噪误差分布（条形图）
errors = denoised_demo - x_clean_test
axes[1, 0].bar(range(len(errors)), errors, color='orange', alpha=0.7)
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].set_xlabel('样本索引')
axes[1, 0].set_ylabel('去噪误差')
axes[1, 0].set_title(f'去噪误差条形图 (平均绝对误差={np.mean(np.abs(errors)):.3f})')
axes[1, 0].grid(alpha=0.3, axis='y')

# 子图4: 去噪误差直方图（与子图3形成对比，展示统计分布）
axes[1, 1].hist(errors, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='零误差线')
axes[1, 1].axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=2, 
                   alpha=0.7, label=f'均值={np.mean(errors):.3f}')
axes[1, 1].set_xlabel('去噪误差')
axes[1, 1].set_ylabel('频数')
axes[1, 1].set_title(f'去噪误差分布直方图 (标准差={np.std(errors):.3f})')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1-3_去噪器与得分函数.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图表已保存: 步骤1-3_去噪器与得分函数.png")

# ============================================================
# 实验总结
# ============================================================
print(f"\n{'='*60}")
print("实验6.6-1 总结")
print(f"{'='*60}")
print("1. Tweedie等式实践:")
print("   - 训练去噪器: $D_\\sigma(\\tilde{x}) \\approx x$")
print("   - 提取得分: $\\nabla\\log p_\\sigma(\\tilde{x}) = \\frac{D_\\sigma(\\tilde{x}) - \\tilde{x}}{\\sigma^2}$")
print("   - 去噪器与得分函数通过Tweedie等式等价转换")
print("\n2. 去噪效果验证:")
print(f"   - 相关系数: {corr:.4f}")
print(f"   - 平均去噪误差: {np.mean(np.abs(errors)):.4f}")
print("   - Tweedie等式成功将去噪器转换为得分估计器")
print("\n3. 实践意义:")
print("   - 去噪是更容易学习的任务(有明确的监督信号)")
print("   - 通过Tweedie等式,可以间接获得得分函数")
print("   - 这是现代扩散模型(Score-based Models)的核心思想")

print(f"\n{'='*60}")
print("第六章配套实验完成!")
print(f"{'='*60}")
print("实验列表:")
print("  6.1-1: 归一化常数困境与得分匹配动机")
print("  6.2-1: ESM与ISM的验证")
print("  6.3-1: 去噪得分匹配(DSM)训练与验证")
print("  6.4-1: Hutchinson迹估计与切片得分匹配(SSM)")
print("  6.5-1: 多尺度得分匹配与退火Langevin采样")
print("  6.6-1: 从去噪器中提取得分函数(Tweedie等式实践)")
