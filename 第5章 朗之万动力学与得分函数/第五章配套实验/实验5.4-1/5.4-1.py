# -*- coding: utf-8 -*-
"""
实验5.4-1 近端算子：Moreau包络的一步梯度
对应章节：5.4 MAP与MMSE的结构对偶性
  - 近端算子 = Moreau包络的一步梯度（第3小节）
  - 验证TV近端算子作为MAP去噪器的实现

知识点：
  - 近端算子定义：prox_{λR}(y) = argmin_x {R(x) + ||x-y||^2/(2λ)}
  - Moreau包络梯度：∇R̂_λ(y) = (y - prox_{λR}(y)) / λ
  - 近端算子 = 沿Moreau包络梯度走一步：prox = y - λ∇R̂_λ(y)

运行前提：
  仅需CPU，无需GPU和预训练模型
  需要 sampling_tools/chambolle_prox_TV.py（已包含在当前目录）

本实验对应5.4节第3小节"近端算子 = Moreau包络的一步梯度"。
拆分自原实验5.4-1的步骤1。
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import os
import sys

# ====== 中文字体配置（兼容本地和Google Colab）======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验5.4-1')
    _chinese_path = os.path.join(SAVE_DIR, '.chinese')
    os.makedirs(_chinese_path, exist_ok=True)
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

np.random.seed(42)
torch.manual_seed(42)

# 导入sampling_tools模块
_local_sampling_tools = os.path.join(SAVE_DIR, 'sampling_tools')
if os.path.exists(_local_sampling_tools):
    sys.path.insert(0, SAVE_DIR)
    try:
        from sampling_tools import chambolle_prox_TV
        _has_sampling_tools = True
    except ImportError as e:
        print(f"警告: sampling_tools 导入失败: {e}")
        _has_sampling_tools = False
else:
    _has_sampling_tools = False
    if _IN_COLAB:
        print("\n" + "=" * 60)
        print("Colab环境提示")
        print("=" * 60)
        print("  sampling_tools 模块未找到")
        print(f"  请确保已将整个实验目录上传到 Google Drive:")
        print(f"  路径: {_gdrive}/实验5.4-1/")
        print("  需要上传的文件:")
        print("    - 5.4-1.py")
        print("    - sampling_tools/ (整个目录)")
        print("    - .chinese/ (可选，会自动创建)")
        print("=" * 60)
    else:
        print("警告: sampling_tools 模块未找到")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ============================================================
# 步骤1：TV近端算子（显式先验）演示
# 使用 sampling_tools/chambolle_prox_TV.py
# ============================================================
print("\n" + "=" * 60)
print("步骤1：TV近端算子（显式先验）演示")
print("=" * 60)

if not _has_sampling_tools:
    print("警告: sampling_tools 模块未找到，跳过步骤1")
    print("  请确保 sampling_tools/ 目录存在且包含 chambolle_prox_TV.py")
else:
    # 创建测试图像（简单形状）
    test_image = np.zeros((64, 64))
    test_image[10:20, 10:20] = 1.0  # 正方形
    test_image[40:50, 40:50] = 0.8  # 另一个正方形
    test_image_t = torch.from_numpy(test_image).float().to(device)

    # 添加噪声
    noisy_image_t = test_image_t + 0.2 * torch.randn_like(test_image_t)

    # 测试不同lam值的TV近端算子
    lam_values = [0.01, 0.05, 0.1, 0.5]

    plt.figure(figsize=(15, 4))

    plt.subplot(1, len(lam_values)+2, 1)
    plt.imshow(test_image, cmap='gray', vmin=0, vmax=1)
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(1, len(lam_values)+2, 2)
    plt.imshow(noisy_image_t.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    plt.title('含噪图像')
    plt.axis('off')

    for i, lam in enumerate(lam_values):
        result = chambolle_prox_TV(noisy_image_t, device, {'lambda': lam, 'maxiter': 200})
        # 设备一致性检查
        assert result.device == noisy_image_t.device, f"设备不一致: result在{result.device}, 输入在{noisy_image_t.device}"
        plt.subplot(1, len(lam_values)+2, i+3)
        plt.imshow(result.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.title(r'TV近端 ($\lambda$={})'.format(lam))
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤1_TV近端算子.png'), dpi=150)
    plt.close()

    print("TV近端算子说明：")
    print(r"  $\lambda$小（0.01）：近端算子接近输入，TV正则化弱，保留更多噪声")
    print(r"  $\lambda$大（0.5）：近端算子趋向于常数，TV正则化强，图像被过度平滑")
    print(r"  适当$\lambda$：平衡去噪与保真度")

# ============================================================
# 步骤2：Moreau包络梯度验证
# 验证：prox = y - λ∇R̂_λ(y)，即近端算子 = 沿Moreau包络梯度走一步
# ============================================================
print("\n" + "=" * 60)
print("步骤2：验证 Moreau包络梯度关系")
print("=" * 60)

if not _has_sampling_tools:
    print("警告: sampling_tools 模块未找到，跳过步骤2")
else:
    lam_tv = 0.1

    # 计算近端算子
    prox_result = chambolle_prox_TV(noisy_image_t, device, {'lambda': lam_tv, 'maxiter': 200})
    
    # 理论梯度：(y - prox) / λ
    gradient_theory = (noisy_image_t - prox_result) / lam_tv

    # ====== 用有限差分独立估计Moreau包络的梯度 ======
    # Moreau包络：R̂_λ(y) = R(prox(y)) + ||y - prox(y)||^2 / (2λ)
    # 其中 R(x) = ||x||_TV 是TV范数
    
    def compute_tv_norm(x):
        """计算TV范数（各向异性）"""
        dx = torch.diff(x, dim=0)
        dy = torch.diff(x, dim=1)
        return torch.sum(torch.abs(dx)) + torch.sum(torch.abs(dy))
    
    def compute_moreau_envelope(y, lam):
        """计算Moreau包络值 R̂_λ(y) = R(prox(y)) + ||y - prox(y)||^2 / (2λ)"""
        prox_y = chambolle_prox_TV(y, device, {'lambda': lam, 'maxiter': 200})
        residual = y - prox_y
        # TV范数项：R(prox(y)) = ||prox(y)||_TV
        tv_term = compute_tv_norm(prox_y)
        # 保真项：||y - prox(y)||^2 / (2λ)
        fidelity_term = 0.5 * torch.sum(residual ** 2) / lam
        return tv_term + fidelity_term
    
    # 使用有限差分估计梯度（只对部分像素采样，避免计算量过大）
    eps = 1e-4  # 有限差分步长
    n_samples = 100  # 采样像素数
    H, W = noisy_image_t.shape
    
    # 随机选择采样像素
    torch.manual_seed(42)
    sample_indices = torch.randint(0, H * W, (n_samples,))
    sample_h = sample_indices // W
    sample_w = sample_indices % W
    
    # 计算有限差分梯度
    gradient_fd = torch.zeros_like(noisy_image_t)
    errors = []
    
    print(f"  使用有限差分验证梯度关系（采样 {n_samples} 个像素）...")
    print(f"  有限差分步长: ε = {eps}")
    
    for i in range(n_samples):
        h, w = sample_h[i].item(), sample_w[i].item()
        
        # 创建扰动向量
        e_i = torch.zeros_like(noisy_image_t)
        e_i[h, w] = 1.0
        
        # 中心差分：(R̂(y + εe_i) - R̂(y - εe_i)) / (2ε)
        moreau_plus = compute_moreau_envelope(noisy_image_t + eps * e_i, lam_tv)
        moreau_minus = compute_moreau_envelope(noisy_image_t - eps * e_i, lam_tv)
        grad_fd_i = (moreau_plus - moreau_minus) / (2 * eps)
        
        gradient_fd[h, w] = grad_fd_i
        
        # 计算误差
        error_i = torch.abs(grad_fd_i - gradient_theory[h, w]).item()
        errors.append(error_i)
    
    # 统计误差
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    rel_error = mean_error / (torch.mean(torch.abs(gradient_theory)).item() + 1e-8)
    
    print(rf"  测试 $\lambda = {lam_tv}$")
    print(f"  理论梯度: (y - prox) / λ")
    print(f"  有限差分梯度: [R̂(y+εe_i) - R̂(y-εe_i)] / (2ε)")
    print(f"  平均绝对误差: {mean_error:.4e}")
    print(f"  最大绝对误差: {max_error:.4e}")
    print(f"  相对误差: {rel_error:.2%}")
    
    is_valid = rel_error < 0.1  # 相对误差 < 10% 则认为验证通过
    print(f"  验证结果：{is_valid}（相对误差 < 10% 则成立）")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(noisy_image_t.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(r'含噪图像 $y$')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(prox_result.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(r'$\mathrm{prox}_{\lambda\mathrm{TV}}(y)$' + '\n(MAP方向)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow((noisy_image_t - prox_result).cpu().numpy(), cmap='RdBu_r')
    axes[0, 2].set_title(r'$y - \mathrm{prox}(y)$' + '\n' + r'$= \lambda\cdot\nabla \hat{R}_\lambda(y)$')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(gradient_theory.cpu().numpy(), cmap='RdBu_r')
    axes[1, 0].set_title(r'理论梯度 $\nabla \hat{R}_\lambda(y)$' + '\n' + r'$= (y - \mathrm{prox})/\lambda$')
    axes[1, 0].axis('off')

    # 有限差分梯度（只在采样点有值）
    axes[1, 1].imshow(gradient_fd.cpu().numpy(), cmap='RdBu_r')
    axes[1, 1].set_title(r'有限差分梯度' + '\n' + r'$(R̂(y+εe_i) - R̂(y-εe_i))/(2ε)$')
    axes[1, 1].axis('off')

    # 梯度误差图
    error_map = torch.abs(gradient_fd - gradient_theory).cpu().numpy()
    axes[1, 2].imshow(error_map, cmap='hot')
    axes[1, 2].set_title(f'梯度误差图\n平均误差: {mean_error:.2e}')
    axes[1, 2].axis('off')

    fig.suptitle('Moreau包络梯度验证：有限差分 vs 理论公式', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '步骤2_Moreau包络梯度验证.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  上图说明：")
    print(r"    上排：含噪图像、近端算子结果、残差（λ倍梯度）")
    print(r"    下排：理论梯度、有限差分梯度、梯度误差图")
    print(r"    验证：有限差分独立估计的梯度与理论公式 (y-prox)/λ 一致")


# ============================================================
# 实验总结
# ============================================================
print("\n" + "=" * 60)
print("实验5.4-1 总结")
print("=" * 60)
print("1. 近端算子定义：")
print(r"   $\mathrm{prox}_{\lambda R}(y) = \arg\min_x \{R(x) + \frac{\|x-y\|^2}{2\lambda}\}$")
print("\n2. Moreau包络梯度：")
print(r"   $\nabla \hat{R}_\lambda(y) = \frac{y - \mathrm{prox}_{\lambda R}(y)}{\lambda}$")
print("\n3. 梯度步解读：")
print(r"   $\mathrm{prox}_{\lambda R}(y) = y - \lambda\,\nabla\hat{R}_\lambda(y)$")
print(r"   近端算子 = 从 $y$ 出发，沿Moreau包络梯度走一步，步长为 $\lambda$")
print(r"   这是MAP方向的'一步梯度下降'")
print("\n4. TV近端算子参数敏感性：")
print(r"   $\lambda$小：弱正则化，保留更多细节和噪声")
print(r"   $\lambda$大：强正则化，趋向常数，过度平滑")
print("\n下一步：加载去噪器（学习去噪器），实现MMSE方向的'一步'（见拆分实验5.4-2）")


# ===== 保存数值结果 =====
import json

def _to_native(obj):
    """递归转换numpy/torch类型为Python原生类型"""
    import numpy as np
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return _to_native(obj.tolist())
    try:
        import torch
        if isinstance(obj, torch.Tensor): return _to_native(obj.detach().cpu().tolist())
    except: pass
    return obj

results_summary = {
    "实验名称": "实验5.4-1 近端算子：Moreau包络的一步梯度",
    "环境信息": {
        "sampling_tools可用": bool(_has_sampling_tools),
        "device": str(device),
    },
}

# 步骤1：TV近端算子
if _has_sampling_tools:
    results_summary["步骤1_TV近端算子"] = {
        "lambda测试值": list(lam_values),
    }

# 步骤2：Moreau包络梯度验证（变量在else块内定义，需安全访问）
try:
    _step2 = {
        "lambda_tv": round(float(lam_tv), 6),
        "平均绝对误差": round(float(mean_error), 8),
        "最大绝对误差": round(float(max_error), 8),
        "相对误差": round(float(rel_error), 6),
        "验证通过": bool(is_valid),
        "采样像素数": int(n_samples),
        "有限差分步长": float(eps),
    }
    results_summary["步骤2_Moreau包络梯度验证"] = _step2
except (NameError, UnboundLocalError):
    results_summary["步骤2_Moreau包络梯度验证"] = "未执行（缺少sampling_tools）"

results_summary = _to_native(results_summary)
with open(os.path.join(SAVE_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"数值结果已保存: {os.path.join(SAVE_DIR, 'results_summary.json')}")
