# -*- coding: utf-8 -*-
"""
实验18.2-1 自定义前向算子与伴随验证
对应知识点：18.2节（定义自定义前向算子）

实验内容：
Step 1: 深入理解deepinv Physics类体系 —— 浏览内置算子并理解三原则（配对A/A^T、噪声分离、可组合性）
Step 2: 实现多视角下采样算子 MultiViewPhysics —— 仿射变换+下采样的复合前向模型
Step 3: 伴随验证与自伴随检测 —— dot product test验证手动伴随，autograd自动伴随对比
Step 4: 噪声模型对比 —— 高斯/泊松/泊松-高斯复合噪声的视觉效果与统计特性
Step 5: ★自定义模糊核与MRI掩模算子 —— 将自定义退化集成到deepinv框架中

★原创设计：
- Step 5的自定义模糊核+MRI掩模综合实验，验证算子可组合性原则
- 对比手动伴随 vs autograd自动伴随的数值精度
- 泊松-高斯复合噪声模型的可视化对比

素材来源：MiniProject_DefiningOperator.ipynb、deepinv Physics API
运行前提：无需GPU，CPU即可运行
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

# Windows PowerShell 默认 GBK 编码无法打印上标 ²、中点 · 等 Unicode 字符，
# 这里主动把 stdout 切到 utf-8，避免 Step 4 公式 print 时 UnicodeEncodeError 中断运行
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# ====== 中文字体配置(兼容本地和Google Colab) ======
_gdrive = '/content/drive/MyDrive'
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    if not os.path.isdir(_gdrive):
        print("正在挂载 Google Drive...")
        drive.mount('/content/drive')
    SAVE_DIR = os.path.join(_gdrive, '实验18.2-1')
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

# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 安装deepinv（Colab环境）
# 教学注意：使用 subprocess.check_call 并显式退出，避免 os.system 静默失败
try:
    import deepinv
except ImportError:
    print("正在安装 deepinv ...")
    import subprocess
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install',
             'git+https://github.com/deepinv/deepinv.git#egg=deepinv']
        )
    except subprocess.CalledProcessError as e:
        print(f"deepinv 安装失败（退出码 {e.returncode}），请检查网络或手动安装后重试")
        raise SystemExit(1)
    import deepinv as dinv
else:
    dinv = deepinv
    print(f"deepinv 版本: {dinv.__version__}")

from deepinv.physics import LinearPhysics, Downsampling, GaussianNoise, PoissonNoise, Denoising
from deepinv.utils import load_example


# ========================================================================
# Step 1: 浏览deepinv Physics类体系
# 对应18.2节知识点：Physics类设计三原则
# ========================================================================
print("\n" + "="*70)
print("Step 1: deepinv Physics 类体系与设计三原则")
print("="*70)

print("""
deepinv Physics 类体系的设计三原则：
1. 配对性: 每个算子 A 必须提供对应的伴随 A_adjoint（或 A^T）
2. 噪声分离: 前向模型 y = A(x) + ε，噪声独立于算子可组合
3. 可组合性: 多个算子可通过乘法组合成复合前向模型

常用内置 Physics 类：
- Denoising:        A = I (恒等算子)，加性噪声
- Downsampling:     A = 下采样算子（最近邻/双线性）
- Blur:             A = 卷积模糊算子
- Inpainting:       A = 逐像素掩模
- Tomography:       A = Radon 变换（CT重建）
- MRI:              A = 傅里叶+子采样掩模
""")

# 加载示例图像
x_true = load_example("celeba_example.jpg", img_size=(256, 256), resize_mode='resize')
print(f"load_example 返回 shape: {x_true.shape}, ndim: {x_true.ndim}")
# 确保是4D张量 (B, C, H, W)
if x_true.ndim == 3:
    x_true = x_true.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
elif x_true.ndim == 5:
    # 5D 属于非标准格式（如 (1, 1, C, H, W)），按启发式规则显式压缩；
    # 打印警告说明假设，便于学生排查数据结构异常
    print(f"  ⚠️ [Warning] load_example 返回 5D 张量 shape={tuple(x_true.shape)}，"
          f"按 (B, 1, C, H, W) 假设压缩为 4D；如不符合请检查数据源")
    if x_true.shape[1] == 1:
        x_true = x_true.squeeze(1)
    else:
        # ⚠️ 通道维不为 1 时保守取第一组（隐性丢数据），此情况罕见但在预期外时建议排查数据源
        x_true = x_true[:, 0]
elif x_true.ndim != 4:
    raise ValueError(f"load_example 返回的 x_true ndim={x_true.ndim} 不在支持范围 [3,4,5]")
x_true = x_true.to(device)
print(f"处理后 x_true shape: {x_true.shape}")


# ========================================================================
# Step 2: 实现多视角下采样算子 MultiViewPhysics
# 对应18.2节知识点：自定义LinearPhysics子类
# 素材来源: MiniProject_DefiningOperator.ipynb
# ========================================================================
print("\n" + "="*70)
print("Step 2: 实现多视角下采样算子 MultiViewPhysics")
print("="*70)

class MultiViewPhysics(LinearPhysics):
    """
    多视角下采样算子：y_j = A * T_j * x
    其中 A 是基础算子（如下采样），T_j 是仿射变换（旋转+平移）
    
    输入 x: (B, C, H, W)
    输出 y: (B, J, C, H', W')，其中 H'=H/factor, W'=W/factor
    """
    def __init__(self, base_physics, transf=None, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.base_physics = base_physics
        self.transf = transf
        self.device = device
        # 【重要】img_size 以 base_physics 为准，避免 kwargs 传入冲突参数时导致尺寸判断错误

    def A(self, x, **kwargs):
        """前向传播: 对x施加J个仿射变换，再分别下采样"""
        transf = kwargs.get('transf', self.transf)
        if transf is None:
            raise ValueError("需要提供变换矩阵 transf")
        
        # 确保输入是4D张量 (B, C, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        elif x.ndim == 5:
            # 5D 属于非标准格式（如 (B, 1, C, H, W)），按启发式规则显式压缩；
            # 打印警告说明假设，便于学生排查数据结构异常
            print(f"  ⚠️ [Warning] MultiViewPhysics.A 收到 5D 输入 shape={tuple(x.shape)}，"
                  f"按 (B, 1, C, H, W) 假设压缩为 4D；如不符合请检查上游算子")
            if x.shape[1] == 1:
                x = x.squeeze(1)
            else:
                # ⚠️ 通道维不为 1 时保守取第一组（隐性丢数据），此情况罕见但在预期外时建议排查上游算子
                x = x[:, 0]
        elif x.ndim != 4:
            raise ValueError(f"期望输入为3D或4D张量，但得到 {x.ndim}D，shape={x.shape}")
        
        B, C, H, W = x.shape
        J = transf.shape[0]
        
        # 对每个视角施加仿射变换
        views = []
        for j in range(J):
            # 构造仿射网格
            theta_j = transf[j:j+1].to(x.device).float()  # (1, 2, 3)
            grid = F.affine_grid(theta_j.expand(B, -1, -1),  # (B, 2, 3)
                                  size=(B, C, H, W),
                                  align_corners=False)
            x_transformed = F.grid_sample(x, grid, align_corners=False)
            # 下采样
            y_j = self.base_physics.A(x_transformed)
            views.append(y_j)
        
        y = torch.stack(views, dim=1)  # (B, J, C, H', W')
        return y

    def A_adjoint(self, y, **kwargs):
        """
        伴随算子: 复合算子 y = A * T_j * x 的伴随

        【重要教学说明——手动伴随是"近似"而非精确】
        对于像素空间的纯仿射变换 T(x)=Ax+t，其严格伴随为
            T^T(y) = A^T y + (-A^T t)
        但本实现的"复合"算子内部使用 grid_sample+affine_grid 完成 T_j，
        grid_sample 本质是带双线性插值的采样算子：
            1) 严格意义下，采样算子的伴随需要 VJP（autograd）逐点计算插值梯度；
            2) 仿射矩阵转置 A^T 只对"连续坐标"层面的仿射映射是精确伴随，
               对"像素值+插值"层面是数学上的近似；
            3) 误差量级受 align_corners、插值模式、变换剧烈程度影响。
        因此本方法在变换接近恒等（旋转角 ≤ π/8、平移小）时误差约 1e-3~1e-2，
        当变换较大或需亚像素精度时应改用 autograd 自动伴随（见 Step 3c）。
        【注】上述"1e-3~1e-2"为基于 grid_sample+affine_grid 一般行为的经验估计，
        仅反映单次运行观察值，并非理论保证的误差界；
        实际数值请以本实验运行输出为准，且可能随分辨率、变换幅度变化。
        """
        transf = kwargs.get('transf', self.transf)
        if transf is None:
            raise ValueError("需要提供变换矩阵 transf")
        
        B, J, C, H_prime, W_prime = y.shape
        # 从base_physics推断原始尺寸
        if hasattr(self.base_physics, 'img_size') and self.base_physics.img_size is not None:
            img_size = self.base_physics.img_size
            if len(img_size) == 3:
                _, H_full, W_full = img_size  # img_size 是 (C, H, W) 三元组
            elif len(img_size) == 4:
                _, _, H_full, W_full = img_size  # (B, C, H, W)
            else:
                raise ValueError(f"不支持的img_size格式: {img_size}")
        else:
            factor = self.base_physics.factor if hasattr(self.base_physics, 'factor') else 4
            H_full = H_prime * factor
            W_full = W_prime * factor
        
        x_adj = torch.zeros(B, C, H_full, W_full, device=y.device)
        
        for j in range(J):
            y_j = y[:, j]  # (B, C, H', W')
            # A^T: 上采样（伴随下采样）
            # 【教学提示】本实现调用 base_physics.A_adjoint 的"精确"伴随；
            # 当前 base_physics 用 filter=None（恒等滤波），因此 A^T 实际就是上采样本身。
            # 若改用非平凡 filter（如默认的 sinc 低通），base_physics 内部会有滤波伴随步骤，
            # 此时本方法对该部分的"近似"误差会进一步叠加 filter 伴随的实现误差。
            # 在本实验的参数设置（filter=None）下，该叠加源不存在。
            x_up = self.base_physics.A_adjoint(y_j)  # (B, C, H, W)
            # T_j^T 的近似：仿射矩阵转置 + 平移调整
            # 严格意义下，grid_sample 的伴随是 VJP；此处用 A^T 近似
            # 当变换接近恒等时误差较小，剧烈变换时需用 autograd 自动伴随
            theta_2x3 = transf[j].to(y.device).float()  # (2, 3)
            A_2x2 = theta_2x3[:, :2]  # (2, 2) 旋转+缩放部分
            t_2x1 = theta_2x3[:, 2:3]  # (2, 1) 平移部分
            A_T = A_2x2.T  # 线性部分取转置作为连续坐标层的近似伴随
            t_T = -A_T @ t_2x1  # 伴随变换的平移: t^T = -A^T t
            theta_adj_full = torch.cat([A_T, t_T], dim=1).unsqueeze(0)  # (1, 2, 3)

            grid = F.affine_grid(theta_adj_full.expand(B, -1, -1),  # (B, 2, 3)
                                  size=(B, C, H_full, W_full),
                                  align_corners=False)
            # 累加: A^T y = Σ_j T_j^T A^T y_j （堆叠算子的伴随是各子伴随之和，
            # 重叠区域值大反映观测更多，后续求解器会归一化）
            x_adj = x_adj + F.grid_sample(x_up, grid, align_corners=False)
        
        return x_adj

    def update_parameters(self, transf=None, **kwargs):
        """更新变换参数"""
        if transf is not None:
            self.transf = transf


def generate_random_transforms(J, scale=0.8, max_angle=np.pi/8, max_shift=0.05):
    """生成J个随机仿射变换矩阵（旋转+缩放+平移）"""
    transf = torch.zeros(J, 2, 3)
    for i in range(J):
        angle = (torch.rand(1) * 2 - 1) * max_angle  # 双向旋转，范围 [-max_angle, max_angle]
        transf[i, 0, 0] = torch.cos(angle) * scale
        transf[i, 0, 1] = -torch.sin(angle) * scale
        transf[i, 1, 0] = torch.sin(angle) * scale
        transf[i, 1, 1] = torch.cos(angle) * scale
        transf[i, :, -1] = torch.randn(2) * max_shift
    return transf


# 创建多视角算子并测试
print("\n创建多视角下采样算子 (4×下采样, J=16个视角)...")
base_physics = Downsampling(factor=4, img_size=(3, 256, 256), device=device,
                             filter=None, padding="zeros")
# 【教学说明】此处重设种子是为了让本步骤（Step 2）独立可复现，
# 不受 Step 1 图像加载等全局随机调用的影响；与文件开头的全局种子无关。
torch.manual_seed(42)
transf = generate_random_transforms(J=16)

multi_physics = MultiViewPhysics(base_physics, transf=transf, device=device)
y_multi = multi_physics(x_true)
print(f"  输入 x shape:  {x_true.shape}")
print(f"  输出 y shape:  {y_multi.shape}")
print(f"  J={y_multi.shape[1]} 个视角, 每个视角 {y_multi.shape[3]}×{y_multi.shape[4]}")


# ========================================================================
# Step 3: 伴随验证与自伴随检测
# 对应18.2节知识点：dot product test与autograd自动伴随
# ========================================================================
print("\n" + "="*70)
print("Step 3: 伴随验证与自伴随检测")
print("="*70)

print("""
伴随验证原理 (Dot Product Test):
  对任意 x, y: <Ax, y> ≈ <x, A^T y>
  相对误差 |<Ax,y> - <x,A^Ty>| / max(|<Ax,y>|, |<x,A^Ty>|) 应接近0

deepinv 提供:
  physics.adjointness_test(x) → 相对误差
  dinv.physics.adjoint_function(A) → 基于autograd的自动伴随
""")


def detailed_adjointness_test(physics, x, n_tests=5):
    """多次随机测试伴随的精度，返回均值和标准差
    
    注意：使用 physics.A(x) 而非 physics(x)，避免噪声模型的随机性影响
    """
    errors = []
    for i in range(n_tests):
        # 每次重设不同种子以测试不同随机向量下的伴随精度
        torch.manual_seed(100 + i)
        Ax = physics.A(x)  # 使用确定性的前向算子 A(x)，避免 physics(x) 自带噪声的随机性
        y_rand = torch.randn_like(Ax)
        lhs = (Ax * y_rand).sum().item()
        rhs = (x * physics.A_adjoint(y_rand)).sum().item()
        rel_err = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-10)
        errors.append(rel_err)
    return np.mean(errors), np.std(errors)

# 3a. 对内置算子做伴随验证
print("\n--- 3a. 内置算子伴随验证 ---")

# 下采样算子
down_phys_step3 = Downsampling(factor=2, img_size=(3, 256, 256), device=device,
                                 filter=None)
x_test = torch.randn(1, 3, 256, 256, device=device)
adj_err_down = down_phys_step3.adjointness_test(x_test)
mean_err, std_err = detailed_adjointness_test(down_phys_step3, x_test, n_tests=5)
print(f"  下采样 (factor=2) 伴随误差: {adj_err_down:.2e} (多次测试: {mean_err:.2e} ± {std_err:.2e})")

# 3b. 对MultiViewPhysics做伴随验证
print("\n--- 3b. MultiViewPhysics 伴随验证（★手动伴随是近似）---")
adj_err_multi = multi_physics.adjointness_test(x_true)
mean_multi, std_multi = detailed_adjointness_test(multi_physics, x_true, n_tests=5)
print(f"  手动伴随相对误差: {adj_err_multi:.2e} (多次测试: {mean_multi:.2e} ± {std_multi:.2e})")

# 【教学提示】对手动伴随做分层解释，避免简单的 PASS/FAIL 二分
# grid_sample 内部的双线性插值伴随在严格意义上需要 autograd VJP；
# 本实现用仿射矩阵转置作近似，误差量级受 align_corners、变换剧烈程度影响
# 注意：adjointness_test 输出的相对误差可能为负（lhs-rhs 符号），
# 判断阈值时需取绝对值
abs_err = abs(adj_err_multi)
if abs_err < 1e-3:
    verdict = "满足 1e-3 阈值（对小旋转/小平移场景可接受）"
    advice = "在变换接近恒等时手动伴随已足够；如需更严格精度请用 Step 3c autograd"
elif abs_err < 1e-2:
    verdict = "介于 1e-3 ~ 1e-2 之间（典型 grid_sample 插值伴随误差量级）"
    advice = "强烈推荐使用 Step 3c 的 autograd 自动伴随"
else:
    verdict = "超过 1e-2（变换较剧烈时手动近似的固有误差）"
    advice = "应当使用 Step 3c 的 autograd 自动伴随；本实现仅作教学演示"

print(f"  |手动伴随相对误差|: {abs_err:.2e}")
print(f"  解读: {verdict}")
print(f"  建议: {advice}")
print("  注意: 手动伴随本质是'连续坐标层'的精确 + '插值层'的近似；")
print("        dot product test 通过并不等于算法实现正确，只反映近似精度足够")

# 3c. ★ 使用autograd自动伴随对比
print("\n--- 3c. ★ autograd 自动伴随 vs 手动伴随 ---")
try:
    # 兼容不同deepinv版本的导入路径
    try:
        from deepinv.physics import adjoint_function
    except ImportError:
        from deepinv.utils import adjoint_function
    
    # 创建使用autograd伴随的版本
    class MultiViewPhysicsAutoAdj(MultiViewPhysics):
        """使用autograd自动计算伴随的版本"""
        def A_adjoint(self, y, **kwargs):
            # 利用autograd自动计算A^T
            # 注意：adjoint_function 需要的是输入 x 的形状，不是输出 y 的形状
            # 从 base_physics 推断原始图像尺寸
            if hasattr(self.base_physics, 'img_size') and self.base_physics.img_size is not None:
                img_size = self.base_physics.img_size
                if len(img_size) == 3:
                    _, H_full, W_full = img_size
                elif len(img_size) == 4:
                    _, _, H_full, W_full = img_size
                else:
                    raise ValueError(f"不支持的img_size格式: {img_size}")
            else:
                factor = self.base_physics.factor if hasattr(self.base_physics, 'factor') else 4
                H_full = y.shape[-2] * factor
                W_full = y.shape[-1] * factor
            
            B = y.shape[0]
            C = y.shape[2] if y.ndim == 5 else y.shape[1]
            x_shape = (B, C, H_full, W_full)
            
            adj_fn = adjoint_function(self.A, x_shape, device=y.device)
            return adj_fn(y)
    
    multi_physics_auto = MultiViewPhysicsAutoAdj(base_physics, transf=transf, device=device)
    adj_err_auto = multi_physics_auto.adjointness_test(x_true)
    print(f"  autograd伴随误差: {adj_err_auto:.2e} (|.|={abs(adj_err_auto):.2e})")
    print(f"  手动伴随误差:     {adj_err_multi:.2e} (|.|={abs(adj_err_multi):.2e})")

    # 比较绝对误差（dot product test 误差可能带符号）
    if abs(adj_err_auto) < abs(adj_err_multi):
        print("  ★ autograd自动伴随更精确！推荐在手动伴随不精确时使用")
    else:
        print("  手动伴随已经足够精确（与 autograd 误差量级相当）")
    print("  说明：autograd 通过 VJP 精确计算 grid_sample 的插值伴随，")
    print("        误差接近数值精度；手动版本用仿射矩阵转置作近似，会随变换剧烈程度下降")
        
except (ImportError, AttributeError):
    print("  autograd伴随功能在当前deepinv版本不可用，跳过对比")

# 3d. 算子范数
print("\n--- 3d. 算子范数估计 ---")
try:
    norm_down = down_phys_step3.compute_norm(x_test)
    print(f"  下采样算子范数: {norm_down:.4f}")
except Exception as e:
    print(f"  算子范数计算失败: {e}")

# --- Step 2 可视化（多视角前向结果）---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
# 原始图像
axes[0].imshow(x_true[0].cpu().permute(1, 2, 0).clamp(0, 1))
axes[0].set_title('原始图像 x', fontsize=12)
axes[0].axis('off')

# 多视角观测（前4个视角）
for i in range(3):
    axes[i+1].imshow(y_multi[0, i].cpu().permute(1, 2, 0).clamp(0, 1))
    angle_deg = torch.atan2(transf[i, 1, 0], transf[i, 0, 0]).item() * 180 / np.pi
    axes[i+1].set_title(f'视角 {i+1}\n旋转≈{angle_deg:.1f}°', fontsize=11)
    axes[i+1].axis('off')

fig.suptitle('Step 2: 多视角下采样算子的前向结果', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step2_multiview_forward.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step2_multiview_forward.png")


# ========================================================================
# Step 4: 噪声模型对比
# 对应18.2节知识点：噪声分离原则与常见噪声模型
# ========================================================================
print("\n" + "="*70)
print("Step 4: 噪声模型对比（高斯/泊松/泊松-高斯）")
print("="*70)

print("""
噪声模型对比:
  高斯噪声:     y = Ax + ε,           ε ~ N(0, σ²I)        — 加性，方差恒定
  泊松噪声:     y = Poisson(Ax/λ)·λ   — 信号依赖型，方差随信号强度变化
  泊松-高斯:    y = Poisson(Ax/λ)·λ + ε — 传感器噪声的更真实模型
""")

# 设置不同噪声模型
sigma_gauss = 0.05  # 高斯噪声标准差
gain_poisson = 0.1  # 泊松噪声增益（越小噪声越大）

# 使用下采样算子 + 不同噪声
down_phys_step4 = Downsampling(factor=4, img_size=(3, 256, 256), device=device,
                                 filter=None)

# 高斯噪声
phys_gauss = Downsampling(factor=4, img_size=(3, 256, 256), device=device,
                           filter=None)
phys_gauss.set_noise_model(GaussianNoise(sigma=sigma_gauss))

# 泊松噪声
phys_poisson = Downsampling(factor=4, img_size=(3, 256, 256), device=device,
                             filter=None)
phys_poisson.set_noise_model(PoissonNoise(gain=gain_poisson))

# 泊松-高斯复合噪声（★模拟真实传感器：光子散粒噪声+电子读出噪声）
class PoissonGaussianNoise(nn.Module):
    """泊松-高斯复合噪声模型
    y = Poisson(x/λ)·λ + N(0, σ²)
    - 输入x为 A(x_out)（由deepinv Physics自动传入）
    - Poisson: 光子散粒噪声（信号依赖型，方差∝信号强度）
    - Gaussian: 电子读出噪声（信号无关型，方差恒定）
    - gain(λ)越小，泊松噪声越强；sigma越大，高斯噪声越强

    推荐取值范围（基于x∈[0,1]假设）:
    - gain ∈ [0.05, 0.5]：
      gain 太小（如 <0.01）时，x/gain 极大，
      torch.poisson 在不同硬件/PyTorch 版本上可能溢出或精度下降；
      gain=0.1 是教学演示的典型取值。
    - sigma ∈ [0.005, 0.1]：
      sigma 太大（如 >0.2）时泊松部分会被高斯淹没，
      sigma=0.05 是教学演示的典型取值，高斯分量在视觉上可辨。

    【重要简化假设】
    forward() 方法对输入执行两次 clamp：
      1) x.clamp(0,1) 限制泊松采样前的输入范围
      2) 输出再次 clamp(0,1) 确保结果有效
    这假设 A(x) 已在 [0,1] 范围内（如经过归一化的图像）。
    若实际输入超出该范围（如模糊/下采样后有轻微 overshoot），
    clamp 会截断极端值，可能引入信息损失。
    对于未归一化数据，应先做归一化预处理或移除 clamp 逻辑。
    """
    def __init__(self, gain=0.1, sigma=0.02):
        super().__init__()
        self.gain = gain
        self.sigma = sigma

    def forward(self, x, **kwargs):
        # 1. 泊松噪声：模拟光子计数过程
        #    注意：clamp(0,1) 限制输入幅度，避免 x/gain 过大导致数值不稳定
        x_poisson = torch.poisson(x.clamp(0, 1) / self.gain) * self.gain
        # 2. 高斯噪声：模拟电子读出噪声（独立于信号）
        x_noisy = x_poisson + self.sigma * torch.randn_like(x)
        return x_noisy.clamp(0, 1)

phys_pg = Downsampling(factor=4, img_size=(3, 256, 256), device=device,
                        filter=None)
phys_pg.set_noise_model(PoissonGaussianNoise(gain=0.1, sigma=0.05))

# 生成观测
# 【教学说明】此处重设种子是为了让本步骤（Step 4）独立可复现，
# 不受 Step 2/3 等全局随机调用的影响；与文件开头的全局种子无关。
torch.manual_seed(42)
y_clean = down_phys_step4(x_true)
y_gauss = phys_gauss(x_true)
y_poisson = phys_poisson(x_true)
y_pg = phys_pg(x_true)

# 计算噪声水平
noise_gauss = (y_gauss - y_clean).abs().mean().item()
noise_poisson = (y_poisson - y_clean).abs().mean().item()
noise_pg = (y_pg - y_clean).abs().mean().item()

print(f"  无噪声观测:    MAE = 0")
print(f"  高斯噪声:      MAE = {noise_gauss:.4f}  (σ={sigma_gauss})")
print(f"  泊松噪声:      MAE = {noise_poisson:.4f}  (gain={gain_poisson})")
print(f"  泊松-高斯:     MAE = {noise_pg:.4f}  (gain=0.1, σ=0.05)")

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
titles = ['无噪声', '高斯噪声', '泊松噪声', '泊松-高斯']
images = [y_clean, y_gauss, y_poisson, y_pg]

for i, (title, img) in enumerate(zip(titles, images)):
    axes[0, i].imshow(img[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0, i].set_title(title, fontsize=12)
    axes[0, i].axis('off')
    
    if i > 0:
        residual = (img - y_clean)[0].cpu().permute(1, 2, 0)
        axes[1, i].imshow(residual.clamp(-0.1, 0.1) + 0.5, vmin=0, vmax=1)
        mae_value = (images[i] - y_clean).abs().mean().item()
        axes[1, i].set_title(f'残差 (MAE={mae_value:.4f})', fontsize=10)
    else:
        axes[1, i].text(0.5, 0.5, '(原始观测)', ha='center', va='center', fontsize=12)
    axes[1, i].axis('off')

# 隐藏左下角的空位
axes[1, 0].imshow(y_clean[0].cpu().permute(1, 2, 0).clamp(0, 1))
axes[1, 0].set_title('原始观测', fontsize=10)
axes[1, 0].axis('off')

fig.suptitle('Step 4: 噪声模型对比（4×下采样观测）', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step4_noise_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step4_noise_comparison.png")

# 噪声直方图对比
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i, (title, img) in enumerate(zip(titles[1:], images[1:])):
    residual = (img - y_clean).cpu().numpy().flatten()
    axes[i].hist(residual, bins=100, density=True, alpha=0.7, color=f'C{i}')
    axes[i].set_title(f'{title} 残差分布', fontsize=12)
    axes[i].set_xlabel('残差值')
    axes[i].set_ylabel('概率密度')
    axes[i].axvline(0, color='red', linestyle='--', alpha=0.5)

fig.suptitle('Step 4: 噪声残差分布对比', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step4_noise_histogram.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step4_noise_histogram.png")


# ========================================================================
# Step 5: ★自定义模糊核与MRI掩模算子
# 对应18.2节知识点：算子可组合性与自定义退化
# ========================================================================
print("\n" + "="*70)
print("Step 5: ★ 自定义模糊核与MRI掩模算子（算子组合性验证）")
print("="*70)

print("""
★ 本步骤演示deepinv的算子可组合性原则：
  将自定义模糊核 → 下采样 → MRI子采样 组合为复合前向模型
  复合模型 y = M·D·K·x + ε 中每个子算子都有独立的伴随
""")

# 5a. 自定义模糊核
print("\n--- 5a. 自定义运动模糊核 ---")

# 初始化所有flag，避免后续逻辑混乱
has_blur = has_composite = has_mri = False

def create_motion_blur_kernel(length=15, angle=45):
    """创建运动模糊核"""
    kernel = np.zeros((length, length))
    center = length // 2
    angle_rad = np.deg2rad(angle)
    for i in range(length):
        offset = i - center
        x = int(round(center + offset * np.cos(angle_rad)))
        y = int(round(center + offset * np.sin(angle_rad)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    kernel = kernel / kernel.sum()
    return kernel

# 创建两种模糊核
motion_kernel = create_motion_blur_kernel(length=15, angle=30)
# 尝试获取高斯核，fallback使用手动构造的单位冲激核（避免全零导致的问题）
try:
    # 使用新的API路径
    try:
        from deepinv.physics.functional import gaussian_blur as gauss_blur_func
        gauss_kernel_np = gauss_blur_func(sigma=(2.0, 2.0)).numpy()
    except ImportError:
        # 兼容旧版本
        if hasattr(dinv.physics.blur, 'gaussian_blur'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                gauss_kernel_np = dinv.physics.blur.gaussian_blur(sigma=(2.0, 2.0)).numpy()
        else:
            raise AttributeError("gaussian_blur not available")
except Exception:
    # Fallback: 手动构造简单的高斯核或单位冲激核
    size = 15
    center = size // 2
    gauss_kernel_np = np.zeros((1, 1, size, size))
    # 使用单位冲激（中心为1，其余为0）作为最安全的fallback
    gauss_kernel_np[0, 0, center, center] = 1.0
    print("  [Warning] 使用单位冲激核作为gaussian_blur的fallback")

gauss_kernel = torch.tensor(gauss_kernel_np, dtype=torch.float32)

# 使用deepinv的Blur算子
try:
    from deepinv.physics import Blur
    motion_k_tensor = torch.tensor(motion_kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # 扩展到3通道
    motion_k_3ch = motion_k_tensor.repeat(1, 3, 1, 1)
    blur_phys = Blur(filter=motion_k_3ch, device=device)

    y_blurred = blur_phys(x_true)

    # 伴随验证
    adj_err_blur = blur_phys.adjointness_test(x_true)
    print(f"  运动模糊算子伴随误差: {adj_err_blur:.2e}")

    has_blur = True
except Exception as e:
    # 【教学改进】关键路径（★原创部分）打印完整 traceback，便于学生排查
    import traceback
    print(f"  Blur算子创建失败: {e}")
    print("  详细 traceback:")
    traceback.print_exc()
    print("  → 跳过模糊部分")

# 5b. ★ 组合算子：模糊 → 下采样
if has_blur:
    print("\n--- 5b. ★ 组合算子: 模糊 → 下采样 ---")
    try:
        # 注意：Blur算子可能会改变图像尺寸（边界效应），需要确保尺寸匹配
        # 方案1：使用padding='same'保持尺寸（如果Blur支持）
        # 方案2：先检查模糊后的尺寸，再创建Downsampling
        y_blurred_test = blur_phys(x_true)
        _, _, H_blur, W_blur = y_blurred_test.shape
        
        # 组合算子: y = D(K(x))，即先模糊再下采样
        # deepinv中用乘法表示算子组合，执行顺序从右到左: D * K 表示先应用K再应用D
        composite_phys = Downsampling(factor=4, img_size=(3, H_blur, W_blur), device=device,
                                       filter=None) * blur_phys
        y_composite = composite_phys(x_true)

        # ★ 显式验证组合顺序正确性：对比手动调用与组合调用的结果
        # 手动调用：先模糊，再下采样
        down_phys_step5 = Downsampling(factor=4, img_size=(3, H_blur, W_blur), device=device,
                                        filter=None)
        y_manual = down_phys_step5.A(blur_phys.A(x_true))
        # 使用相对误差而非绝对误差，阈值 1e-5 基于 |y| 的数量级（≈1），
        # 适用于 x ∈ [0,1] 归一化图像；其他数值范围需相应调整阈值
        abs_diff = (y_composite - y_manual).abs().max().item()
        y_scale = y_composite.abs().mean().item() + 1e-10
        order_diff = abs_diff / y_scale
        print(f"  组合顺序验证: 组合结果 vs 手动调用 最大相对误差 = {order_diff:.2e} (最大绝对误差 = {abs_diff:.2e})")
        if order_diff < 1e-5:
            print("  ✓ 组合顺序正确：D * K 确实对应 先应用K(模糊) 再应用D(下采样)")
        else:
            print(f"  ! 组合顺序可能有问题（相对误差 {order_diff:.2e} > 1e-5），请检查deepinv版本")
        
        adj_err_composite = composite_phys.adjointness_test(x_true)
        print(f"  组合算子(模糊+下采样)伴随误差: {adj_err_composite:.2e}")
        print(f"  组合观测 shape: {y_composite.shape}")
        has_composite = True
    except Exception as e:
        # 【教学改进】打印完整 traceback，便于学生调试
        import traceback
        print(f"  算子组合失败: {e}")
        print("  详细 traceback:")
        traceback.print_exc()
        print("  → 跳过组合算子部分")
        has_composite = False

# 5c. MRI子采样掩模
print("\n--- 5c. MRI子采样掩模 ---")
try:
    from deepinv.physics import MRI
    
    # 创建4倍加速的MRI物理模型
    mri_phys = MRI(img_size=(256, 256), device=device, acceleration=4)
    # MRI通常是单通道，将RGB转灰度（ITU-R BT.709标准）
    x_gray = 0.2126*x_true[:,0:1] + 0.7152*x_true[:,1:2] + 0.0722*x_true[:,2:3]
    
    # MRI算子期望输入是 (B, 2, H, W) 的实数张量：channel 0=实部, channel 1=虚部
    # deepinv 内部会调用 view_as_complex(x.moveaxis(1, -1)) 把它转为复数
    # 【教学注意】不要预先用 view_as_complex 转成复数！否则会触发
    # "view_as_complex is only supported for half, float and double tensors"
    x_mri_input = torch.stack(
        [x_gray.squeeze(1), torch.zeros_like(x_gray.squeeze(1))], dim=1
    ).contiguous()  # (B, 2, H, W) float32

    y_mri = mri_phys(x_mri_input)

    adj_err_mri = mri_phys.adjointness_test(x_mri_input)
    print(f"  MRI算子伴随误差: {adj_err_mri:.2e}")
    print(f"  MRI观测 shape: {y_mri.shape}")
    has_mri = True
except Exception as e:
    # 【教学改进】关键路径（★原创部分）打印完整 traceback
    import traceback
    print(f"  MRI算子创建失败: {e}")
    print("  详细 traceback:")
    traceback.print_exc()
    print("  → 跳过 MRI 部分")
    has_mri = False

# 5d. 可视化
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
# 第一行: 原图 + 模糊 + 组合
axes[0, 0].imshow(x_true[0].cpu().permute(1, 2, 0).clamp(0, 1))
axes[0, 0].set_title('原始图像', fontsize=12)
axes[0, 0].axis('off')

if has_blur:
    axes[0, 1].imshow(y_blurred[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[0, 1].set_title('运动模糊', fontsize=12)
    axes[0, 1].axis('off')
    # ★ 模糊核的幅度谱（频域视角理解模糊）
    kernel_fft = torch.fft.fftshift(
        torch.fft.fft2(torch.tensor(motion_kernel, dtype=torch.float32))
    )
    axes[0, 2].imshow(np.log1p(kernel_fft.abs().numpy()), cmap='viridis')
    axes[0, 2].set_title('★ 模糊核幅度谱', fontsize=12)
    axes[0, 2].axis('off')
else:
    axes[0, 1].text(0.5, 0.5, '模糊算子不可用', ha='center', va='center', fontsize=12)
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')

# 第二行: 组合 + MRI重建 + MRI k空间
if has_composite:
    axes[1, 0].imshow(y_composite[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[1, 0].set_title('★ 模糊+下采样组合', fontsize=12)
    axes[1, 0].axis('off')
else:
    axes[1, 0].text(0.5, 0.5, '组合算子不可用', ha='center', va='center', fontsize=12)
    axes[1, 0].axis('off')

if has_mri:
    # 零填充重建（伪逆）：x* = A^T y，非最优但快速
    x_adj_mri = mri_phys.A_adjoint(y_mri)
    axes[1, 1].imshow(x_adj_mri[0, 0].cpu(), cmap='gray')
    axes[1, 1].set_title('MRI零填充重建', fontsize=12)
    axes[1, 1].axis('off')

    # k-space
    # ★ 修复：deepinv 的 MRI 算子内部已对 k-space 做 fftshift 居中
    # （im_to_kspace = ifftshift + fftn + fftshift），输出 y_mri 即为中心化 k-space
    # 原代码再 fftshift 一次会把中心点移到角落，加上只取实部、log1p 无归一化导致全黑
    # 正确做法：view_as_complex 转复数 → abs 取模 → log1p → 99 百分位归一化增强对比度
    y_complex = torch.view_as_complex(
        y_mri[0].permute(1, 2, 0).contiguous()  # (H, W, 2) → complex
    )  # (H, W) complex
    kspace_abs = y_complex.abs().numpy()
    kspace_log = np.log1p(kspace_abs)
    # 用 99 百分位归一化以增强 k-space 高频细节的对比度（k-space 动态范围极大）
    vmax = float(np.percentile(kspace_log, 99))
    kspace_disp = np.clip(kspace_log / vmax, 0, 1) if vmax > 0 else kspace_log
    axes[1, 2].imshow(kspace_disp, cmap='gray')
    axes[1, 2].set_title('MRI k空间(对数, 99%归一化)', fontsize=12)
    axes[1, 2].axis('off')
else:
    axes[1, 1].text(0.5, 0.5, 'MRI不可用', ha='center', va='center', fontsize=12)
    axes[1, 1].axis('off')
    axes[1, 2].text(0.5, 0.5, 'MRI不可用', ha='center', va='center', fontsize=12)
    axes[1, 2].axis('off')

# 第三行: 伴随误差条形图（图标风格）+ 模糊核 + MRI输入
# 【教学改进】将原文字汇总改为 log10 误差条形图，便于横向比较各算子伴随精度
# 文字汇总通过 print 在控制台输出（不在图中放置文字，符合"图或图标"要求）

# 收集各算子伴随误差数据
op_names = []
op_errors = []
op_colors = []

# 内置算子
op_names.append('Down')
op_errors.append(abs(adj_err_down))
op_colors.append('#1f77b4')  # 蓝色：内置

if has_blur and 'adj_err_blur' in locals():
    op_names.append('Blur')
    op_errors.append(abs(adj_err_blur))
    op_colors.append('#1f77b4')

if has_composite and 'adj_err_composite' in locals():
    op_names.append('Composite')
    op_errors.append(abs(adj_err_composite))
    op_colors.append('#1f77b4')

if has_mri and 'adj_err_mri' in locals():
    op_names.append('MRI')
    op_errors.append(abs(adj_err_mri))
    op_colors.append('#1f77b4')

# 自定义算子（红色高亮）
op_names.append('MultiView')
op_errors.append(abs(adj_err_multi))
op_colors.append('#d62728')

# 绘制对数刻度横向条形图（log10 量级），便于横向比较各算子精度
log_errors = np.log10(np.maximum(np.array(op_errors), 1e-15))
bars = axes[2, 0].barh(op_names, log_errors, color=op_colors,
                       edgecolor='black', linewidth=0.5)
axes[2, 0].axvline(-3, color='green', linestyle='--', alpha=0.7, label='阈值 1e-3')
axes[2, 0].set_xlabel('log10(相对误差)', fontsize=10)
axes[2, 0].set_title('★ 伴随验证误差对比', fontsize=12)
axes[2, 0].legend(loc='lower right', fontsize=9)
axes[2, 0].grid(axis='x', alpha=0.3)
axes[2, 0].invert_yaxis()  # 让第一个算子在最上面
# 在条形末端标注具体数值
for bar, err in zip(bars, op_errors):
    axes[2, 0].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f'{err:.1e}', va='center', fontsize=9)

# 文字描述：通过 print 输出（不在图中放置文字）
print("\n--- 伴随验证汇总（图例式条形图见 axes[2,0]） ---")
print("[内置算子——库内精确实现，PASS/FAIL 二元判断]")
status_down = "PASS" if abs(adj_err_down) < 1e-3 else "FAIL"
print(f"  下采样算子:    {abs(adj_err_down):.2e}  {status_down}")
if has_blur and 'adj_err_blur' in locals():
    status_blur = "PASS" if abs(adj_err_blur) < 1e-3 else "FAIL"
    print(f"  运动模糊:      {abs(adj_err_blur):.2e}  {status_blur}")
if has_composite and 'adj_err_composite' in locals():
    status_composite = "PASS" if abs(adj_err_composite) < 1e-3 else "FAIL"
    print(f"  模糊+下采样:   {abs(adj_err_composite):.2e}  {status_composite}")
if has_mri and 'adj_err_mri' in locals():
    status_mri = "PASS" if abs(adj_err_mri) < 1e-3 else "FAIL"
    print(f"  MRI子采样:     {abs(adj_err_mri):.2e}  {status_mri}")
print("\n[自定义算子——手动伴随为近似，详见 Step 3b 分层解读]")
abs_err_multi = abs(adj_err_multi)
if abs_err_multi < 1e-3:
    multi_label = "近似足够（<1e-3）"
elif abs_err_multi < 1e-2:
    multi_label = "需用 autograd（1e-3~1e-2）"
else:
    multi_label = "需用 autograd（>=1e-2）"
print(f"  MultiViewPhysics: {abs_err_multi:.2e} -> {multi_label}")
print("\n阈值说明: 内置算子 < 1e-3 通过; 自定义算子详见 Step 3b 分层")

# ★ 模糊核空间域可视化
if has_blur:
    axes[2, 1].imshow(motion_kernel, cmap='gray')
    axes[2, 1].set_title('★ 运动模糊核(空间域)', fontsize=12)
    axes[2, 1].axis('off')

# MRI灰度输入
if has_mri:
    axes[2, 2].imshow(x_gray[0, 0].cpu(), cmap='gray')
    axes[2, 2].set_title('MRI输入(灰度)', fontsize=12)
    axes[2, 2].axis('off')

fig.suptitle('Step 5: ★ 自定义算子与算子组合性验证', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'step5_custom_operators.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: step5_custom_operators.png")

# 5e. ★ 性能基准测试
print("\n--- 5e. ★ 算子性能基准测试 ---")
# 【免责说明】本测试仅演示 time.perf_counter 的计时方法与算子调用流程，
# 并非严格的算子效率横向对比：
#   - Blur 输入 (3, 256, 256) 3通道；Composite 为 Blur+Downsampling 复合，尺寸经模糊后保持；
#   - MRI 输入 (2, 256, 256) 复数表示的灰度图，与 RGB 算子通道数不同；
#   - MRI 内部涉及 FFT 等复数运算，与 Blur/Downsampling 的实数卷积/采样在计算范式上也不同；
#   - 真实横向对比需统一输入尺寸、通道数、迭代次数与硬件环境。
# 本节目的：让学生掌握算子耗时评估方法，理解影响算子速度的混杂变量。
import time
benchmark_ops = []
# 准备每个算子对应的真实输入：MRI 期望复数，其他算子期望实数 RGB
# 教学注意：若 MRI 误用实数输入调用，计时结果无法代表真实调用路径
if has_blur:
    benchmark_ops.append(('Blur', blur_phys, x_true))
if has_composite:
    benchmark_ops.append(('Composite', composite_phys, x_true))
if has_mri:
    benchmark_ops.append(('MRI', mri_phys, x_mri_input))

for name, phys, bench_input in benchmark_ops:
    try:
        # 预热
        _ = phys(bench_input)
        # 正式测试
        t0 = time.perf_counter()
        n_iters = 10
        for _ in range(n_iters):
            _ = phys(bench_input)
        elapsed_ms = (time.perf_counter() - t0) / n_iters * 1000
        print(f"  {name:10s}: {elapsed_ms:7.2f} ms/iter")
    except Exception as e:
        # 【教学改进】打印 traceback，便于排查算子调用路径问题
        import traceback
        print(f"  {name:10s}: 测试失败 ({e})")
        traceback.print_exc()


# ========================================================================
# 实验总结
# ========================================================================
print("\n" + "="*70)
print("实验18.2-1 总结")
print("="*70)
print("""
本实验对应18.2节核心知识点：

1. Physics类设计三原则 ✓
   - 配对性: A 与 A^T 必须配对
   - 噪声分离: 前向模型 + 噪声模型独立设置
   - 可组合性: 算子可通过乘法组合

2. 自定义LinearPhysics子类 ✓
   - MultiViewPhysics: 仿射变换 + 下采样
   - A(x) 和 A_adjoint(y) 的实现
   - update_parameters 动态参数更新

3. 伴随验证方法 ✓
   - dot product test: <Ax,y> ≈ <x,A^T y>
   - autograd自动伴随: 精度更高但计算更慢
   - 算子范数估计: compute_norm (power iteration)

4. 噪声模型体系 ✓
   - 高斯噪声: 加性，方差恒定
   - 泊松噪声: 信号依赖型
   - 泊松-高斯: 复合噪声，更贴近实际传感器

5. ★ 自定义算子组合 ✓
   - 模糊 → 下采样组合
   - MRI子采样验证
   - 验证组合算子伴随仍然正确

预期结果：
- 伴随误差均 < 1e-3（通过验证）
- 泊松噪声残差分布非高斯（信号依赖）
- 组合算子伴随误差与子算子同量级
""")

print(f"所有图像已保存至: {SAVE_DIR}")