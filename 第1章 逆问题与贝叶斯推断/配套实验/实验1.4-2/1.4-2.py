import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# ====== 中文字体配置（兼容本地和 Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    _chinese_path = os.path.join(_gdrive, '实验1.4-2', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验1.4-2')
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()

sys.path.insert(0, _chinese_path)
from chinese_font import setup_chinese_font
setup_chinese_font(save_dir=_chinese_path)
# ========================================================

# ---- 1. 加载数据 ----
# 检查必要文件是否存在
required_files = ['transform_A.npy', 'maomi_grayscale.png', 'maomi_mohu.png', 'random_gray.png']
missing_files = [f for f in required_files if not os.path.exists(os.path.join(SAVE_DIR, f))]

if missing_files:
    print("=" * 60)
    print("缺少以下数据文件:")
    for f in missing_files:
        print(f"  - {f}")
    print("\n请在 Google Colab 中运行以下命令下载数据:")
    print("  !mkdir -p /content/drive/MyDrive/实验1.4-2")
    print("  !wget -O /content/drive/MyDrive/实验1.4-2/transform_A.npy <下载链接>")
    print("  !wget -O /content/drive/MyDrive/实验1.4-2/maomi_grayscale.png <下载链接>")
    print("  !wget -O /content/drive/MyDrive/实验1.4-2/maomi_mohu.png <下载链接>")
    print("  !wget -O /content/drive/MyDrive/实验1.4-2/random_gray.png <下载链接>")
    print("=" * 60)
    raise FileNotFoundError(f"缺少数据文件: {missing_files}")

# 加载变换矩阵 A 
# A 是 1024×1024 矩阵，用于对 32×32 图像的向量化形式（1024 维）进行线性变换
A = np.load(os.path.join(SAVE_DIR, 'transform_A.npy'))
print(f"变换矩阵 A 的形状: {A.shape}")

# 计算 A 的条件数（使用 SVD）
U, s, Vt = np.linalg.svd(A)
condition_number = s[0] / s[-1]
print(f"\nA 的奇异值分析：")
print(f"  最大奇异值 σ_max: {s[0]:.4e}")
print(f"  最小奇异值 σ_min: {s[-1]:.4e}")
print(f"  条件数 κ(A) = σ_max/σ_min: {condition_number:.4e}")
print(f"  log₁₀(κ): {np.log10(condition_number):.2f}")

if condition_number > 1e10:
    print(f"\n⚠️ A 是高度病态矩阵（条件数 > 10¹⁰）")
    print("   这意味着 A 严重损失信息，逆问题极度不稳定")
elif condition_number > 1e6:
    print(f"\n⚠️ A 是严重病态矩阵（条件数 > 10⁶）")
    print("   这意味着 A 显著损失信息，逆问题很不稳定")
elif condition_number > 1e3:
    print(f"\nℹ️ A 是轻度病态矩阵（条件数 > 10³）")
    print("   这意味着 A 有一定信息损失，逆问题较不稳定")
else:
    print(f"\nℹ️ A 的条件数较小，不适定性不明显")

# 加载图像
maomi_gray = np.array(Image.open(os.path.join(SAVE_DIR, 'maomi_grayscale.png')).convert('L')) / 255.0
maomi_mohu = np.array(Image.open(os.path.join(SAVE_DIR, 'maomi_mohu.png')).convert('L')) / 255.0
random_gray = np.array(Image.open(os.path.join(SAVE_DIR, 'random_gray.png')).convert('L')) / 255.0

print(f"图像形状: {maomi_gray.shape}")

# ---- 2. 向量化图像 ----
x_true = maomi_gray.ravel()
y_obs = maomi_mohu.ravel()
x_random = random_gray.ravel()

# ---- 3. 计算似然（数据拟合误差）----
# 似然函数 p(y|x) ∝ exp(-||Ax - y||₂² / 2σ²)
# 数据拟合误差越小，似然越大

Ax_true = A @ x_true
Ax_random = A @ x_random

error_true = np.linalg.norm(Ax_true - y_obs)
error_random = np.linalg.norm(Ax_random - y_obs)

print("\n" + "="*60)
print("似然的关键洞察：似然只关心数据拟合，不关心 x 是否合理")
print("="*60)
print("\n数据拟合误差 ||Ax - y||_2 (L2范数):")
print(f"  真实图像 x_true:  {error_true:.4f}")
print(f"  随机噪声 x_random: {error_random:.4f}")
print(f"\n比值: {error_true / error_random:.2f} 倍")

if error_random < error_true:
    print("\n⚠️ 随机噪声的数据拟合误差更小！")
    print("   这意味着在似然意义下，随机噪声比真实图像'更合理'")
    print("   但随机噪声显然不是一张有意义的图像")
    print(f"\n💡 原因分析：")
    print(f"   - A 的条件数 κ(A) = {condition_number:.2e}，属于{'高度' if condition_number > 1e10 else '严重' if condition_number > 1e6 else '轻度'}病态")
    print(f"   - A 严重损失高频信息，使得多个不同的 x 产生相似的 Ax")
    print(f"   - 随机噪声恰好'运气好'，其模糊结果与观测 y 的误差更小")
    print(f"   - 这正是 1.3 节讨论的'病态性导致不稳定性'的体现")
else:
    print("\n✓ 真实图像的数据拟合误差更小")
    print(f"\n💡 说明：")
    print(f"   - A 的条件数 κ(A) = {condition_number:.2e}")
    print(f"   - 虽然 A 是病态的，但本次随机噪声未能在数据拟合上超越真实图像")
    print(f"   - 这并不影响结论：似然无法区分'有意义'和'无意义'的解")
    print(f"   - 可以尝试多次随机噪声，或增大 A 的病态程度")
    print(f"   - 关键点是：两者的误差可能相当，说明似然的局限性")

# ---- 4. 可视化 ----
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# 第一行：原始图像
axes[0, 0].imshow(maomi_gray, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('真实图像 $x_{true}$\n（有意义的猫咪图像）')
axes[0, 0].axis('off')

axes[0, 1].imshow(maomi_mohu, cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title('观测数据 $y$\n（模糊的猫咪图像）')
axes[0, 1].axis('off')

axes[0, 2].imshow(random_gray, cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title('随机噪声 $x_{random}$\n（无意义的噪声图像）')
axes[0, 2].axis('off')

# 第二行：变换结果和矩阵A可视化
axes[1, 0].imshow(Ax_true.reshape(32, 32), cmap='gray')
axes[1, 0].set_title(f'$A \\cdot x_{{true}}$\n$||Ax - y||_2 = {error_true:.4f}$')
axes[1, 0].axis('off')

# 中间：可视化矩阵 A（将 1024x1024 缩小到 32x32，与其他子图保持一致）
from scipy.ndimage import zoom
# 使用双线性插值将 1024x1024 缩小到 32x32（缩放因子 32/1024 = 1/32）
A_resized = zoom(A, 32/1024, order=1)  # order=1 表示双线性插值
im = axes[1, 1].imshow(A_resized, cmap='RdBu_r', aspect='auto')
axes[1, 1].set_title(f'变换矩阵 $A$（1024×1024 缩放到 32×32）\n$\\kappa(A) = {condition_number:.2e}$')
axes[1, 1].axis('off')
# 添加颜色条
cbar = plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
cbar.set_label('矩阵元素值', fontsize=8)

axes[1, 2].imshow(Ax_random.reshape(32, 32), cmap='gray')
axes[1, 2].set_title(f'$A \\cdot x_{{random}}$\n$||Ax - y||_2 = {error_random:.4f}$')
axes[1, 2].axis('off')

plt.suptitle('似然的局限：数据拟合好 ≠ 图像有意义\n' + 
             ('随机噪声的数据拟合误差更小，但它显然不是真实图像' if error_random < error_true else ''),
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '实验1_4_2_似然的局限.png'), dpi=150, bbox_inches='tight')
plt.show()

# ---- 5. 结论 ----
print("\n" + "="*60)
print("结论")
print("="*60)
print("似然函数 p(y|x) 只衡量'数据拟合程度'，即 ||Ax - y||_2 的大小。")
print("它无法区分：")
print("  - 一个与数据拟合好的'有意义'图像（如真实猫咪）")
print("  - 一个与数据拟合好的'无意义'图像（如随机噪声）")
print("\n这正是不适定性的后果：")
print("  算子 A 的信息损失使得多个不同的 x 能产生相似的 Ax")
print("  仅靠似然无法从中选出'正确'的 x")
print("\n解决方案：引入先验 p(x)")
print("  先验编码了我们对'什么样的图像更合理'的知识")
print("  后验 = 似然 × 先验，同时考虑数据拟合和图像合理性")
