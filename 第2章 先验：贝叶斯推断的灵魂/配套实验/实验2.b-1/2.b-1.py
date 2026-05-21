"""
实验2.b-1 TGV：TV的改进
对应章节：附录2B TGV：TV的改进
知识点：阶梯效应根源；TGV定义；TGV vs TV对比；分段仿射 vs 分段常数

素材来源：
  - 2.5.py: TV去噪与阶梯效应展示
  - 附录2B: TGV理论
"""

import numpy as np
import os
import sys
import warnings

# ====== 静默模式配置 ======
SILENT_MODE = True  # True: 不弹窗、不显示警告；False: 正常交互模式

if SILENT_MODE:
    import matplotlib
    matplotlib.use('Agg')
    warnings.filterwarnings('ignore')
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
else:
    import matplotlib

import matplotlib.pyplot as plt
from skimage import data
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio

_gdrive = '/content/drive/MyDrive'
if os.path.isdir(_gdrive):
    _chinese_path = os.path.join(_gdrive, '实验2.b-1', '.chinese')
    SAVE_DIR = os.path.join(_gdrive, '实验2.b-1')
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

np.random.seed(42)

def tgv_denoise_primal_dual(y, alpha0, alpha1, n_iter=200, tau=0.1, sigma=0.1):
    """
    TGV去噪的原始-对偶方法实现
    
    TGV模型: min_{x,w} 0.5||x-y||^2 + alpha1*||Dx-w||_1 + alpha0*||Ew||_1
    
    其中:
    - x: 图像
    - w: 辅助向量场（梯度的仿射分量逼近）
    - D: 梯度算子
    - E: 对称梯度算子
    
    参数:
        y: 含噪图像
        alpha0: 二阶正则化参数
        alpha1: 一阶正则化参数
        n_iter: 迭代次数
        tau, sigma: 步长参数
    """
    h, w = y.shape
    x = y.copy()
    w_h = np.zeros((h, w))
    w_v = np.zeros((h, w))
    p_h = np.zeros((h, w))
    p_v = np.zeros((h, w))
    q_1 = np.zeros((h, w))
    q_2 = np.zeros((h, w))
    q_3 = np.zeros((h, w))
    
    theta = 1.0
    x_bar = x.copy()
    w_h_bar = w_h.copy()
    w_v_bar = w_v.copy()
    
    for k in range(n_iter):
        Dx_h = np.roll(x_bar, -1, axis=1) - x_bar
        Dx_v = np.roll(x_bar, -1, axis=0) - x_bar
        
        p_h = p_h + sigma * (Dx_h - w_h_bar)
        p_v = p_v + sigma * (Dx_v - w_v_bar)
        
        norm_p = np.sqrt(p_h**2 + p_v**2)
        p_h = p_h / np.maximum(1, norm_p / alpha1)
        p_v = p_v / np.maximum(1, norm_p / alpha1)
        
        Ew_1 = np.roll(w_h_bar, -1, axis=1) - w_h_bar
        Ew_2 = np.roll(w_v_bar, -1, axis=0) - w_v_bar
        Ew_3 = 0.5 * (np.roll(w_h_bar, -1, axis=0) - w_h_bar + np.roll(w_v_bar, -1, axis=1) - w_v_bar)
        
        q_1 = q_1 + sigma * Ew_1
        q_2 = q_2 + sigma * Ew_2
        q_3 = q_3 + sigma * Ew_3
        
        norm_q = np.sqrt(q_1**2 + q_2**2 + 2*q_3**2)
        q_1 = q_1 / np.maximum(1, norm_q / alpha0)
        q_2 = q_2 / np.maximum(1, norm_q / alpha0)
        q_3 = q_3 / np.maximum(1, norm_q / alpha0)
        
        div_p = np.roll(p_h, 1, axis=1) - p_h + np.roll(p_v, 1, axis=0) - p_v
        div_q_1 = np.roll(q_1, 1, axis=1) - q_1
        div_q_2 = np.roll(q_2, 1, axis=0) - q_2
        div_q_3_h = 0.5 * (np.roll(q_3, 1, axis=0) - q_3)
        div_q_3_v = 0.5 * (np.roll(q_3, 1, axis=1) - q_3)
        
        x_new = (x - tau * (-div_p + x_bar - y)) / (1 + tau)
        w_h_new = w_h - tau * (p_h - div_q_1 - div_q_3_h)
        w_v_new = w_v - tau * (p_v - div_q_2 - div_q_3_v)
        
        x_bar = x_new + theta * (x_new - x)
        w_h_bar = w_h_new + theta * (w_h_new - w_h)
        w_v_bar = w_v_new + theta * (w_v_new - w_v)
        
        x = x_new
        w_h = w_h_new
        w_v = w_v_new
    
    return x

n = 128
camera = resize(data.camera(), (n, n))

sigma = 0.1
camera_noisy = random_noise(camera, mode='gaussian', var=sigma**2)

alpha_tv = 0.2
camera_tv = denoise_tv_chambolle(camera_noisy, weight=alpha_tv)

alpha0 = 0.3
alpha1 = 0.15
camera_tgv = tgv_denoise_primal_dual(camera_noisy, alpha0, alpha1, n_iter=300)

psnr_noisy = peak_signal_noise_ratio(camera, camera_noisy)
psnr_tv = peak_signal_noise_ratio(camera, camera_tv)
psnr_tgv = peak_signal_noise_ratio(camera, camera_tgv)

print("===== TGV：TV的改进 =====")
print(f"\n噪声水平 σ = {sigma}")
print(f"\n去噪结果对比:")
print(f"  含噪图像: PSNR = {psnr_noisy:.2f} dB")
print(f"  TV去噪: PSNR = {psnr_tv:.2f} dB")
print(f"  TGV去噪: PSNR = {psnr_tgv:.2f} dB")
print(f"\nTGV参数:")
print(f"  α₀ (二阶正则化) = {alpha0}")
print(f"  α₁ (一阶正则化) = {alpha1}")
print(f"  α₀/α₁ = {alpha0/alpha1:.2f} (建议约2)")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(camera_noisy, cmap='gray')
axes[0].set_title(f'含噪图像\nPSNR={psnr_noisy:.2f}dB')
axes[0].axis('off')

axes[1].imshow(camera, cmap='gray')
axes[1].set_title('原始图像')
axes[1].axis('off')

axes[2].imshow(camera_tv, cmap='gray')
axes[2].set_title(f'TV去噪\n边缘锐利，渐变区阶梯化\nPSNR={psnr_tv:.2f}dB')
axes[2].axis('off')

axes[3].imshow(camera_tgv, cmap='gray')
axes[3].set_title(f'TGV去噪\n渐变区更自然\nPSNR={psnr_tgv:.2f}dB')
axes[3].axis('off')

plt.suptitle('TV vs TGV：阶梯效应的改善', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤1_TV_vs_TGV对比.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()

row = n // 2
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(camera[row, :], 'k-', linewidth=2, label='真实信号')
axes[0].plot(camera_tv[row, :], 'r-', linewidth=1.5, label='TV（阶梯效应）')
axes[0].plot(camera_tgv[row, :], 'g-', linewidth=1.5, label='TGV（更平滑）')
axes[0].set_title('中心行剖面对比')
axes[0].legend()
axes[0].set_xlabel('像素索引')
axes[0].set_ylabel('灰度值')
axes[0].grid(True, alpha=0.3)

grad_region = slice(n//4, n//2)
axes[1].plot(camera[row, grad_region], 'k-', linewidth=2, label='真实（渐变区域）')
axes[1].plot(camera_tv[row, grad_region], 'r-', linewidth=1.5, label='TV（阶梯效应）')
axes[1].plot(camera_tgv[row, grad_region], 'g-', linewidth=1.5, label='TGV（渐变自然）')
axes[1].set_title('渐变区域放大：TGV改善阶梯效应')
axes[1].legend()
axes[1].set_xlabel('像素索引')
axes[1].set_ylabel('灰度值')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤2_阶梯效应改善分析.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

methods = ['含噪', 'TV', 'TGV']
psnrs = [psnr_noisy, psnr_tv, psnr_tgv]
colors = ['gray', 'red', 'green']

axes[0].bar(methods, psnrs, color=colors, alpha=0.7)
axes[0].set_ylabel('PSNR (dB)')
axes[0].set_title('去噪性能对比')
axes[0].set_ylim([0, max(psnrs) + 5])
for i, (m, p) in enumerate(zip(methods, psnrs)):
    axes[0].text(i, p + 0.5, f'{p:.2f}', ha='center', fontsize=10)

table_data = [
    ['惩罚项', '||∇x||₁', 'α₁||∇x-w||₁ + α₀||Ew||₁'],
    ['偏好解', '分段常数', '分段仿射'],
    ['渐变区域', '阶梯效应', '自然渐变'],
    ['边缘保持', '能', '能'],
    ['参数数量', '1个 (α)', '2个 (α₀, α₁)'],
]
axes[1].axis('off')
table = axes[1].table(cellText=table_data,
                      colLabels=['特性', 'TV', 'TGV'],
                      loc='center',
                      cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)
axes[1].set_title('TV vs TGV 特性对比', fontsize=12, pad=20)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '步骤3_性能与特性对比.png'), dpi=150, bbox_inches='tight')
if not SILENT_MODE:
    plt.show()

print("\n===== 附录2B 核心结论 =====")
print("\n1. 阶梯效应根源:")
print("   TV只使用一阶导数，偏好分段常数函数")
print("   渐变区域被近似为一系列阶梯")
print("\n2. TGV改进思路:")
print("   引入二阶导数信息，允许分段仿射函数")
print("   TGV = min_w [α₁||∇x-w||₁ + α₀||Ew||₁]")
print("\n3. TV与TGV的关系:")
print("   当α₀→∞时，TGV退化为TV")
print("   TV是TGV的特例")
print("\n4. 应用场景:")
print("   PET/MRI重建、图像修复等渐变区域重要的场景")
print("\n5. 局限性:")
print("   TGV仍是显式先验，参数需手工调节")
print("   更根本的改进需要走向隐式先验（2.4节）")
