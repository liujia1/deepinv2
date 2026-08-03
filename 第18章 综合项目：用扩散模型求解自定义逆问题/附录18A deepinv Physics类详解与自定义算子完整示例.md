# 附录18A deepinv Physics类详解与自定义算子完整示例

> 定位：18.2 节讲了 Physics 类的设计哲学和"为什么要成对写 $A$ 与 $A^\top$"。这一附录是"工具箱说明书"——把 `LinearPhysics` 的每个方法、每个常用内置类、以及 18.2 里那个多视角下采样算子的**完整可跑代码**摊开给你。想直接 copy 改，看这里。

> 提示：本附录是 API 参考，公式与代码为主、叙述为辅，可跳过不影响主线。

## LinearPhysics类的方法详解

`deepinv.physics.LinearPhysics`是定义线性前向算子的基类。以下是其核心方法的详细说明。

### 构造函数

```python
class LinearPhysics(nn.Module):
    def __init__(self, A=None, A_adjoint=None, noise_model=None, **kwargs):
        """
        参数:
            A: 前向算子函数，签名 A(x, **kwargs) -> y
            A_adjoint: 伴随算子函数，签名 A_adjoint(y, **kwargs) -> z
            noise_model: 噪声模型，None表示无噪声
        """
```

### 核心方法

#### A(x) —— 前向运算

```python
def A(self, x, **kwargs):
    """计算 y = A(x)

    参数:
        x: 输入图像, shape (batch, C, H, W)
    返回:
        y: 测量数据, shape 取决于A
    """
```

#### A_adjoint(y) —— 伴随运算

```python
def A_adjoint(self, y, **kwargs):
    """计算 z = A^T(y)

    参数:
        y: 测量数据
    返回:
        z: 伴随运算结果, shape 与x相同
    """
```

#### A_dagger(y) —— 伪逆运算

```python
def A_dagger(self, y, solver='lsqr', **kwargs):
    """计算 x_hat = A^dagger(y)（最小二乘解）

    参数:
        y: 测量数据
        solver: 求解器选择
            - 'lsqr': LSQR迭代求解
            - 'cg': 共轭梯度法
    返回:
        x_hat: 伪逆解
    """
```

#### adjointness_test(x) —— 伴随验证

```python
def adjointness_test(self, x):
    """验证伴随算子的正确性

    测试 <A(x), y> == <x, A_adjoint(y)> 是否成立

    参数:
        x: 测试输入
    返回:
        相对误差 |<Ax,y> - <x,A^T y>| / |<Ax,y>|
    """
```

#### compute_norm(x) —— 算子范数

```python
def compute_norm(self, x, max_iter=100, tol=1e-6):
    """计算算子的L2范数 ||A||_2（通过幂迭代法）

    参数:
        x: 初始化向量
        max_iter: 最大迭代次数
        tol: 收敛容差
    返回:
        算子范数的估计值
    """
```

## 常用内置Physics类

### Denoising —— 去噪（A=I）

```python
physics = dinv.physics.Denoising()
physics.set_noise_model(dinv.physics.GaussianNoise(sigma=0.1))
```

### Blur —— 卷积模糊

```python
# 高斯模糊
kernel = dinv.physics.blur.gaussian_blur(sigma=(3.0, 3.0))
physics = dinv.physics.Blur(img_size=(3, 256, 256), filter=kernel)

# 运动模糊
kernel = dinv.physics.blur.motion_blur(length=15, angle=30)
physics = dinv.physics.Blur(img_size=(3, 256, 256), filter=kernel)

# FFT加速的模糊（大核推荐）
physics = dinv.physics.BlurFFT(img_size=(3, 256, 256), filter=kernel)
```

### Downsampling —— 下采样（超分辨率）

```python
# 4倍下采样
physics = dinv.physics.Downsampling(
    factor=4,
    img_size=(3, 256, 256),
    filter="gaussian"  # 抗混叠滤波器
)
```

### Inpainting —— 图像修复

```python
# 随机80%像素缺失
physics = dinv.physics.Inpainting(
    img_size=(3, 256, 256),
    mask=0.20,  # 保留20%像素
    device=device
)

# 自定义掩码
mask = torch.zeros(1, 1, 256, 256)
mask[:, :, 100:200, 50:150] = 1  # 仅保留一个矩形区域
physics = dinv.physics.Inpainting(img_size=(3, 256, 256), mask=mask)
```

### Tomography —— CT Radon变换

```python
physics = dinv.physics.Tomography(
    img_width=256,
    num_angles=30,      # 投影角度数
    noise_model=dinv.physics.GaussianNoise(sigma=0.05)
)
```

## 自定义多视角算子完整代码

以下是MiniProject_DefiningOperator中多视角下采样算子的完整实现。它是 18.2 节 `MultiViewPhysics` 的"完整可跑版"——注意看 `A_adjoint` 里我们**没有手推仿射的伴随**，而是直接交给 `adjoint_function` 用自动微分兜底，这正是 18.2.4 说的"正确性 >> 速度"：

```python
import torch
import deepinv as dinv

class MultiViewPhysics(dinv.physics.LinearPhysics):
    """
    多视角下采样前向算子

    前向模型: y_j = A * T_j * x, j = 1, ..., J

    其中:
        A: 基础前向算子（如下采样）
        T_j: 仿射变换（旋转+平移+缩放）
        J: 视角数量
    """

    def __init__(self, base_physics, transf=None, device='cpu', **kwargs):
        """
        参数:
            base_physics: 基础前向算子（如Downsampling）
            transf: 仿射变换矩阵, shape (J, 2, 3)
            device: 计算设备
        """
        super().__init__(**kwargs)
        self.base_physics = base_physics
        self.transf = transf
        self.device = device

    def A(self, x, **kwargs):
        """前向运算: y_j = A * T_j * x"""
        if self.transf is None:
            return self.base_physics.A(x)

        J = self.transf.shape[0]
        y_list = []
        for j in range(J):
            # 步骤1: 应用仿射变换 T_j
            grid = torch.nn.functional.affine_grid(
                self.transf[j:j+1].unsqueeze(0).expand(x.shape[0], -1, -1, -1).reshape(-1, 2, 3),
                x.shape,
                align_corners=False
            )
            x_transformed = torch.nn.functional.grid_sample(
                x, grid, mode='bilinear', align_corners=False
            )
            # 步骤2: 应用基础算子 A
            y_j = self.base_physics.A(x_transformed)
            y_list.append(y_j)

        return torch.stack(y_list, dim=1)  # shape: (batch, J, C, H', W')

    def A_adjoint(self, y, **kwargs):
        """伴随运算: x = sum_j T_j^T * A^T * y_j

        注意：仿射变换的伴随算子涉及逆变换和散布操作，
        手动实现容易出错。推荐使用deepinv的adjoint_function
        通过自动微分确保正确性。
        """
        if self.transf is None:
            return self.base_physics.A_adjoint(y)

        # 初始化输出
        # 注意：尺寸需要从测量y恢复到原始图像尺寸
        # 这里简化处理，实际应根据base_physics的参数计算
        ref = self.base_physics.A_adjoint(y[:, 0, ...])
        x = torch.zeros_like(ref)
        J = y.shape[1]

        for j in range(J):
            # 步骤1: 基础算子的伴随 A^T
            z_j = self.base_physics.A_adjoint(y[:, j, ...])
            # 步骤2: 仿射变换的伴随——使用自动微分计算
            # 手动实现需要逆仿射矩阵+scatter操作，容易出错
            # 推荐使用 adjoint_function 通过 vJP 确保正确性
            from deepinv.physics import adjoint_function

            def apply_Tj(x_in):
                grid = torch.nn.functional.affine_grid(
                    self.transf[j:j+1].expand(x_in.shape[0], -1, -1).reshape(-1, 2, 3),
                    x_in.shape,
                    align_corners=False
                )
                return torch.nn.functional.grid_sample(
                    x_in, grid, mode='bilinear', align_corners=False
                )

            adj_fn = adjoint_function(apply_Tj, z_j.shape)
            x_j = adj_fn(z_j)
            x = x + x_j

        return x

    def update_parameters(self, transf=None, **kwargs):
        """更新仿射变换参数"""
        if transf is not None:
            self.transf = transf
```

### 使用示例

```python
# 定义基础算子（4倍下采样）
base_physics = dinv.physics.Downsampling(factor=4, img_size=(3, 256, 256))
base_physics.set_noise_model(dinv.physics.GaussianNoise(sigma=0.05))

# 生成随机仿射变换
J = 16
transf = torch.zeros(J, 2, 3)
scale = 0.8
for i in range(J):
    angle = torch.rand(1) * torch.pi / 8  # 随机旋转角度
    transf[i, 0, 0] = torch.cos(angle) * scale
    transf[i, 0, 1] = -torch.sin(angle) * scale
    transf[i, 1, 0] = torch.sin(angle) * scale
    transf[i, 1, 1] = torch.cos(angle) * scale
    transf[i, :, -1] = torch.randn(2) * 0.05  # 随机平移

# 创建多视角算子
physics = MultiViewPhysics(base_physics, transf=transf)

# 验证伴随算子
x = torch.randn(1, 3, 256, 256)
print(f"伴随验证: {physics.adjointness_test(x)}")  # 应接近0
print(f"算子范数: {physics.compute_norm(x)}")

# 生成测量数据
x_true = dinv.utils.load_example("celeba_example.jpg", img_size=(256, 256))
y = physics(x_true)

# 伴随重建（初始估计）
x_adj = physics.A_adjoint(y)

# 伪逆重建（最小二乘）
x_pinv = physics.A_dagger(y, solver='lsqr')
```

## 伴随算子的自动计算

如果手动实现伴随算子困难，可以使用deepinv提供的自动伴随函数：

```python
from deepinv.physics import adjoint_function

# 为任意前向函数自动计算伴随
def my_forward(x):
    """自定义前向运算"""
    # ... 任意PyTorch运算 ...
    return y

# 创建自动伴随函数
auto_adjoint = adjoint_function(my_forward, input_shape=(1, 3, 256, 256))

# 使用
z = auto_adjoint(y)  # 等价于 A^T y
```

**原理**：自动伴随利用PyTorch的自动微分系统，通过向量-Jacobian乘积（vJP）计算$A^\top y$。具体地，对$A(x)$关于$x$求方向导数：

$$A^\top y = \frac{\partial}{\partial x}\langle A(x), y\rangle$$

**优势**：无需手动推导和实现伴随算子
**劣势**：计算效率略低于手动实现（约慢10-50%，取决于算子复杂度）

> 小贴士：把这份代码当成"备料间"就好——主线读 18.1→18.5 时用到的 `MultiViewPhysics`、`Blur`、`Downsampling` 都在这里能找到完整可跑版。写完自己的算子，记得先跑一遍 `adjointness_test` 再进求解器，这是 18.2 反复强调的"公证"步骤。
