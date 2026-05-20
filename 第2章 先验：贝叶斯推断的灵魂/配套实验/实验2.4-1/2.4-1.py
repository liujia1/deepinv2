"""
实验2.4-1 Deep Image Prior：网络结构作为隐式先验
对应章节：2.4 从显式先验到隐式先验
知识点：DIP概念；网络结构即先验；早停防止过拟合；隐式正则化

素材来源：
  - examples/optimization/demo_dip.py
  - 2.4章节: Deep Image Prior
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import sys

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.4-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.4-1')
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
else:
    _chinese_path = '.chinese'
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, _chinese_path)
try:
    from chinese_font import setup_chinese_font
    setup_chinese_font(save_dir=_chinese_path)
except ImportError:
    print("警告: chinese_font 模块未找到，中文字体可能无法正常显示")
    print("请确保 .chinese 文件夹已上传到 Google Drive 的正确位置")

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvDecoder(nn.Module):
    """简单的卷积解码器网络"""
    def __init__(self, img_size, in_size=(2, 2), channels=64):
        super().__init__()
        self.img_size = img_size
        self.in_size = in_size
        self.channels = channels
        
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, img_size[0], kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.layers(z)

def create_test_image(size=64):
    """创建测试图像"""
    x = np.zeros((1, size, size))
    center = size // 2
    radius = size // 4
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 < radius**2:
                x[0, i, j] = 1.0
    return torch.tensor(x, dtype=torch.float32)

def create_mask(img_size, mask_ratio=0.5):
    """创建随机掩码"""
    mask = torch.rand(1, 1, *img_size) > mask_ratio
    return mask.float()

img_size = (1, 64, 64)
x_true = create_test_image(64).to(device)

mask = create_mask((64, 64), mask_ratio=0.5).to(device)
physics = lambda x: x * mask

noise_level = 0.1
noise = noise_level * torch.randn_like(x_true)
y = physics(x_true) + noise * mask

channels = 64
in_size = (2, 2)
decoder = ConvDecoder(img_size, in_size, channels).to(device)

z = torch.randn(1, channels, *in_size, device=device)

iterations = 500
lr = 1e-2
optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

losses = []
psnrs = []

print("===== Deep Image Prior：网络结构作为隐式先验 =====")
print(f"\n实验设定:")
print(f"  问题: 图像修复 (Inpainting)")
print(f"  掩码比例: 50%")
print(f"  噪声水平: {noise_level}")
print(f"  迭代次数: {iterations}")
print(f"\nDIP核心思想:")
print(f"  优化目标: min_θ ||y - A·f_θ(z)||²")
print(f"  f_θ: 卷积解码器网络")
print(f"  z: 随机输入 (固定)")
print(f"  网络结构本身起到正则化作用")

for i in range(iterations):
    optimizer.zero_grad()
    x_pred = decoder(z)
    loss = torch.mean((physics(x_pred) - y)**2)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    with torch.no_grad():
        mse = torch.mean((x_pred - x_true)**2).item()
        psnr = 10 * np.log10(1.0 / mse)
        psnrs.append(psnr)
    
    if (i + 1) % 100 == 0:
        print(f"  迭代 {i+1}: Loss = {loss.item():.6f}, PSNR = {psnr:.2f} dB")

best_iter = np.argmax(psnrs)
best_psnr = psnrs[best_iter]

print(f"\n结果:")
print(f"  最佳PSNR: {best_psnr:.2f} dB (迭代 {best_iter+1})")
print(f"  最终PSNR: {psnrs[-1]:.2f} dB")
print(f"\n关键观察:")
print(f"  早停很重要: 过度优化会导致过拟合噪声")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(x_true.cpu().squeeze(), cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(y.cpu().squeeze(), cmap='gray')
axes[0, 1].set_title(f'观测图像 (50%缺失+噪声)')
axes[0, 1].axis('off')

axes[0, 2].imshow(decoder(z).detach().cpu().squeeze(), cmap='gray')
axes[0, 2].set_title(f'DIP重建\nPSNR={psnrs[-1]:.2f}dB')
axes[0, 2].axis('off')

axes[1, 0].plot(losses)
axes[1, 0].set_xlabel('迭代次数')
axes[1, 0].set_ylabel('损失')
axes[1, 0].set_title('损失曲线')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(psnrs)
axes[1, 1].axvline(x=best_iter, color='r', linestyle='--', label=f'最佳: 迭代{best_iter+1}')
axes[1, 1].set_xlabel('迭代次数')
axes[1, 1].set_ylabel('PSNR (dB)')
axes[1, 1].set_title('PSNR曲线 (早停的重要性)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

iterations_show = [50, 200, 400]
for idx, it in enumerate(iterations_show):
    if it < len(psnrs):
        pass
axes[1, 2].text(0.5, 0.7, 'DIP核心洞察', fontsize=14, ha='center', fontweight='bold')
axes[1, 2].text(0.5, 0.5, '网络结构 = 隐式先验', fontsize=12, ha='center')
axes[1, 2].text(0.5, 0.35, '无需训练数据', fontsize=12, ha='center')
axes[1, 2].text(0.5, 0.2, '早停防止过拟合', fontsize=12, ha='center')
axes[1, 2].axis('off')

plt.suptitle('Deep Image Prior: 网络结构作为隐式先验', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_DIP实验.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n===== 2.4章节核心结论 =====")
print("\n1. DIP的定义:")
print("   min_θ ||y - A·f_θ(z)||²")
print("   f_θ: 卷积解码器, z: 随机输入(固定)")
print("\n2. 为什么DIP有效?")
print("   - 卷积网络结构偏好自然图像")
print("   - 网络先学习图像结构,再拟合噪声")
print("   - 结构本身起到隐式正则化作用")
print("\n3. 早停的重要性:")
print("   - 继续优化会过拟合噪声")
print("   - 最佳结果通常在中间迭代")
print("\n4. DIP的特点:")
print("   - 无需训练数据 (单图像优化)")
print("   - 适用于各种逆问题")
print("   - 计算成本高 (每次都要优化)")
print("\n5. 与显式先验的对比:")
print("   显式先验: 手工指定 p(x) 或 R(x)")
print("   DIP: 网络结构隐式编码先验")
