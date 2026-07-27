# 第16章 CT/MRI重建 — 提纲

## 本章定位

**叙事角色**：第16章是全书从理论到实践的落地——前13章完成了"逆问题→贝叶斯→采样→得分→扩散"的理论闭环，第14章拓展了Flow Matching前沿，第15章解决了"用什么架构"的实践问题。本章回答：**这些理论在真实医学成像中如何落地？**CT和MRI是两个最典型的医学逆问题，它们的正向模型有明确的物理基础，不适定性有深刻的数学刻画，而扩散先验重建则是全书核心论点——"扩散模型是逆问题自然发展的终点"——在真实场景中的验证。

**核心论点**：**从物理到先验的统一**——CT和MRI看似是两种完全不同的成像模态，但从逆问题的视角看，它们共享同一结构：物理正向模型→不适定性→先验正则化→学习型重建→扩散先验。这条从物理到先验的路径，正是全书从第1章到第13章推理链的缩影。

**与前章衔接**：第15章介绍了去噪器架构（UNet→DiT），本章将使用这些架构进行重建；第13章建立了"条件扩散=逆问题求解"的理论框架，本章将其在CT/MRI上落地实践。

**与后章衔接**：第17章的自监督学习和等变架构将解决"如何在没有干净数据时训练重建网络"的问题；第18章的综合项目将整合本章的CT/MRI算子与扩散模型。

## 叙事弧

```
CT物理与重建（16.1）→ CT不适定性与正则化（16.2）→ MRI物理与重建（16.3）→
学习型方法（16.4）→ 扩散先验重建（16.5）→ 小结与统一（16.6）
```

抽象模式：**物理建模→数学分析→深度学习→扩散先验**——每一步都是前一步局限性的自然回应，与全书整体叙事弧一致。

---

## 章节结构

### 16.0 本章导读

- 核心观点摘要
- 叙事弧图示
- 各节导航

---

### 16.1 CT成像：从X射线到sinogram

> **核心观点**：CT的正向模型——Radon变换——将三维物体的线积分映射为sinogram，其数学结构由Beer-Lambert衰减定律和Fourier切片定理严格刻画。理解正向模型是理解CT逆问题的前提。

#### 16.1.1 Beer-Lambert定律：X射线衰减的物理

- **X射线与物质的相互作用**：光电吸收、康普顿散射、瑞利散射
- **Beer-Lambert定律推导**：
  - 离散形式：$I(x+\delta x) = I(x) - \mu(x) I(x) \delta x$
  - 连续ODE：$\frac{dI}{dx} = -\mu(x) I$
  - 解：$I(x) = I_0 \exp\left(-\int_0^x \mu(x') dx'\right)$
  - 线积分：$P_l = -\ln\frac{I_1}{I_0} = \int_l \mu(x) dx$
- **物理含义**：测量值是对衰减系数的线积分

#### 16.1.2 Radon变换与sinogram

- **2D Radon变换定义**：
  $$\mathcal{R}f(\theta, s) = \int_{l(\theta,s)} f(x) dx = \int_{-\infty}^{\infty} f(s\cos\theta - t\sin\theta,\; s\sin\theta + t\cos\theta)\, dt$$
  其中 $l(\theta,s) = \{x \mid x_1\cos\theta + x_2\sin\theta = s\}$
- **sinogram**：Radon变换关于$(\theta, s)$的图像表示
  - 为何叫"sinogram"：物体的点在sinogram中呈现正弦曲线
  - Shepp-Logan phantom的sinogram示例
- **离散化**：像素化测量模型——构造测量矩阵$A$
  - 从连续Radon变换到离散系统$y = Ax + \epsilon$
  - $A$的维度：$n_{\text{rays}} \times n_{\text{pixels}}$

#### 16.1.3 Fourier切片定理

- **定理陈述**：1D傅里叶变换（关于$s$）的投影 = 2D傅里叶变换的径向切片
  $$\widehat{\mathcal{R}_\theta f}(\omega) = \hat{f}(\omega\cos\theta, \omega\sin\theta)$$
- **证明思路**：变量替换 + Fubini定理（详见附录16A）
- **核心洞察**：投影数据包含了频域的径向信息——但不均匀，这是后续不适定性的根源之一

#### 16.1.4 滤波反投影（FBP）

- **反投影**：Radon变换的伴随$\mathcal{R}^*$（不是逆）
  $$\mathcal{R}^* g(x) = \int_0^{\pi} g(\theta, x_1\cos\theta + x_2\sin\theta) d\theta$$
- **$\mathcal{R}^*\mathcal{R}$的平滑效应**：反投影≠逆投影，产生模糊
- **FBP公式**：先滤波再反投影
  $$f = \mathcal{R}^* \mathcal{F}^{-1}[|\omega| \cdot \widehat{\mathcal{R}_\theta f}(\omega)]$$
  - 斜坡滤波器$|\omega|$补偿高频衰减
  - 实际滤波器选择：ramp、cosine、Hann窗
- **FBP的局限**：
  - 需要完整的投影角度
  - 对噪声敏感（高频放大）
  - 有限角度下产生严重伪影
- **新增 §16.1.5 多能量 CT 与物质分离**（Bologna 冬季学校 Ex2）：单能→能谱；μ 分解为 R 种基础物质体积分数 α_r(x)μ_r(E) → 能谱耦合到同一组 α_r 可分离材料；material-separating regularizer 把能谱耦合变结构约束（呼应第17章零空间填充）；规模大用 IPM+PCG 求解 KKT

FBP在全角、低噪声条件下工作良好，但当投影角度不完整或噪声较大时，CT重建变得不适定——这正是第1章讨论的逆问题一般性困难在CT中的具体表现。

**来源**：Siltanen Day2 P37-47（Beer-Lambert）；Siltanen Day2 P48-97（像素化模型）；IP_and_Im_Lectures-master tomography.md（Radon/FBP/Fourier切片定理完整推导）；IP_and_Im_Lectures-master fourier_sampling.md

---

### 16.2 CT的不适定性与正则化重建

> **核心观点**：CT逆问题的不适定性有两种截然不同的表现：全角CT是温和不适定的（奇异值~1/n衰减），而有限角CT是严重不适定的（奇异值指数衰减）。波前集理论从几何角度解释了有限角为何丢失信息。正则化方法的选择必须匹配不适定性的程度。

#### 16.2.1 全角CT的奇异性分析

- **Radon变换作为紧算子**：$\mathcal{R}$是$L^2 \to L^2$的紧算子
- **奇异值衰减**：全角CT的奇异值$\sigma_n \sim 1/n$（多项式衰减）
  - 与经典算子对比：紧算子SVD衰减率决定不适定程度
  - $\sigma_n \sim 1/n$意味着温和不适定——正则化可有效恢复
- **全角CT正则化重建**：
  - Tikhonov正则化：$\min_x \|Ax - y\|^2 + \lambda\|x\|^2$
  - TV正则化：$\min_x \|Ax - y\|^2 + \lambda\|Dx\|_1$
  - 小波/剪切波稀疏正则化

#### 16.2.2 有限角CT：奇异值的指数衰减

- **有限角问题定义**：仅在$\theta \in [\theta_{\min}, \theta_{\max}]$范围内有投影
- **SVD揭示的病态性**：
  - 全角：$\sigma_n \sim 1/n$（多项式衰减）
  - 有限角：$\sigma_n \sim e^{-cn}$（指数衰减）
  - 即使缺失很小的角度区间，奇异值也指数衰减
- **有限角sinogram的结构**
- **应用场景**：乳腺断层成像（tomosynthesis）、工业焊接检测

#### 16.2.3 波前集理论：为什么有限角丢失信息

- **波前集**的定义：函数的奇异性位置+方向
  - $WF(f) \subset \mathbb{R}^n \times S^{n-1}$：点×方向
- **有限角CT的可见性**：
  - 全角CT恢复所有方向（完整波前集）
  - 有限角CT只恢复"可见"方向的波前集
  - 法线方向与测量角度垂直的边缘"不可见"
- **FBP恢复的是稳定部分**——可见波前集
- **TV能恢复更多**——利用全变差的结构先验补充不可见边缘
- **剪切波学习不可见边缘**（Bubba et al. 2019）：数据驱动的不可见信息恢复

#### 16.2.4 正则化重建：从Tikhonov到TV到剪切波

- **正则化方法谱系**：
  - Tikhonov → 过度平滑，丢失边缘
  - TV → 保持边缘，但产生阶梯效应
  - 小波稀疏（$B_{11}^1$）→ 多尺度稀疏表示
  - 剪切波稀疏 → 方向性稀疏表示，特别适合有限角
  - TGV（广义全变差）→ 避免TV阶梯效应
- **方法选择与不适定程度的对应**：
  - 温和不适定（全角+高SNR）→ Tikhonov/TV
  - 中度不适定（稀疏投影）→ 小波/TV
  - 严重不适定（有限角）→ 剪切波/学习型方法
- **Helsinki Tomography Challenge 2022**：有限角CT竞赛的实践启示

CT的正则化重建展示了"先验选择决定重建质量"这一核心原则。现在转向另一种医学成像模态——MRI，它的正向模型与CT完全不同（傅里叶采样而非线积分），但不适定性的挑战同样存在，只是形式不同。

**来源**：Siltanen Day3A P28-54（有限角SVD与不适定性）；Siltanen Day3A P55-58（波前集理论）；Siltanen Day3A P63（剪切波学习不可见边缘）；Siltanen Day2 P125-162（CT正则化方法）；Siltanen Day3A P70-80（Helsinki挑战赛）

---

### 16.3 MRI成像：从自旋到k-space

> **核心观点**：MRI的正向模型是傅里叶采样——k-space数据是图像的傅里叶变换。MRI的不适定性不来自算子本身（傅里叶变换是良态的），而来自加速采集的愿望：用更少的k-space样本重建图像，这正是压缩感知和深度学习方法的用武之地。

#### 16.3.1 MRI物理基础：Bloch方程与信号生成

- **核磁共振原理**：
  - 质子在磁场中的进动
  - 磁化矢量$\mathbf{m} = (m_x, m_y, m_z)$
- **Bloch方程**：
  $$\frac{d\mathbf{m}}{dt} = \gamma \mathbf{m} \times \mathbf{B} - \begin{pmatrix} m_x/T_2 \\ m_y/T_2 \\ (m_z - m_0)/T_1 \end{pmatrix}$$
  - 旋转项（Larmor进动）+ 弛豫项（$T_1$纵向，$T_2$横向）
  - 详见附录16B
- **对比度机制**：$T_1$加权、$T_2$加权——同一解剖结构可产生不同对比度图像
- **从物理到信号**：RF激发产生横向磁化→梯度编码空间信息→Faraday感应定律测量信号

#### 16.3.2 k-space采样与傅里叶重建

- **k-space变量**：
  $$\mathbf{k}(t) = \frac{\gamma}{2\pi} \int_0^t \mathbf{g}(\tau) d\tau$$
- **信号方程**：
  $$s(\mathbf{k}) = \int u(\mathbf{r}) e^{-2\pi i \mathbf{k} \cdot \mathbf{r}} d\mathbf{r}$$
  - **关键洞察**：$s(\mathbf{k})$就是图像$u$的傅里叶变换
- **重建**：$u = \mathcal{F}^{-1} s$（逆FFT）
- **k-space轨迹**：
  - Cartesian轨迹：逐行采样，FFT直接重建
  - 非Cartesian轨迹：径向、螺旋——需要非均匀FFT或正则化
  - 非Cartesian的病态性：SVD衰减，需要正则化

#### 16.3.3 欠采样掩码与零填充重建

- **加速采集的动机**：3D MRI扫描时间~分钟级，减少k-space样本可缩短时间
- **欠采样掩码**$\Omega$：
  - Cartesian欠采样：跳过k-space行
  - 随机欠采样：伪随机可变密度
  - 等间距欠采样：产生混叠伪影（aliasing）
- **零填充重建**：缺失k-space数据置零→逆FFT
  - 混叠伪影的数学解释：欠采样等价于频域与梳状函数的乘积→空间域的周期性叠加
- **Nyquist准则**：采样间距$\leq 1/L$（$L$为视野大小）

#### 16.3.4 加速采集：并行成像与压缩感知MRI

- **并行成像（Parallel Imaging）**：
  - 多线圈信号模型：$s^p = F C^p u$（$C^p$为线圈灵敏度图）
  - 联合重建：$\min_u \|s - \tilde{F}u\|^2 + \lambda R(u)$
  - 加速因子2-5倍，临床几乎标配
- **压缩感知MRI**：
  - 核心条件：稀疏性+不相干性
  - 稀疏变换：小波$\|Wu\|_1$、TV $\|Du\|_1$
  - 不相干采样：伪随机可变密度（k-space中心密集）
  - 优化问题：$\min_u \|F_\Omega u - s\|^2 + \lambda\|Wu\|_1$
  - 临床加速约40%
- **并行成像+压缩感知的组合**

经典的CT正则化和MRI压缩感知方法都依赖手工设计的先验（TV、小波稀疏等），这与第2章讨论的"显式先验天花板"完全对应。自然地，下一步是用学习型方法替代手工先验。

**来源**：IP_and_Im_Lectures-master magnetic_resonance_imaging.md（MRI物理/k-space/并行成像/压缩感知完整材料）；Lustig et al. (2008) Compressed Sensing MRI

---

### 16.4 学习型重建方法

> **核心观点**：学习型重建方法用数据驱动的网络替代手工设计的先验或算法模块，代表了从"设计先验"到"学习先验"的范式转移。三种主流范式——端到端学习、算法展开、采样模式学习——分别替换了重建流程的不同部分。

#### 16.4.1 UNet端到端重建

- **后处理范式**：
  - 先用FBP/零填充得到初步重建$\hat{x}_0$
  - 再用UNet修正：$\hat{x} = \text{UNet}(\hat{x}_0)$
- **训练**：配对数据$(\hat{x}_0^{(i)}, x_{\text{true}}^{(i)})$，MSE损失
- **优势**：简单、快速（单次前向传播）
- **局限**：
  - 依赖初步重建的质量
  - 可能产生幻觉（网络学到的先验与测量不一致）
  - 不保证测量一致性
  - 需要配对训练数据（见正文"延伸"框：第17章等变成像/SURE 可绕过配对数据，附 FastMRI 4× 加速 PSNR 对比 35.73 vs 监督 36.63）

#### 16.4.2 算法展开：Learned Gradient Descent

- **从迭代算法到展开网络**：
  - 经典迭代：$x_{k+1} = x_k - \alpha(A^T(Ax_k - y) + \nabla R(x_k))$
  - 展开为网络：每步迭代=一个网络层，近端算子=可学习模块
- **Learned Gradient Descent (LGD)**：
  $$x_{k+1} = x_k - \alpha_k A^T(Ax_k - y) + G_k(x_k)$$
  其中$G_k$是可学习的梯度修正模块（UNet/ResNet）
- **与经典方法的对应**：
  - ISTA展开 → LISTA
  - ADMM展开 → Learned ADMM
  - 原始-对偶展开 → Learned Primal-Dual (Adler & Öktem 2018)
- **优势**：
  - 保留迭代结构，测量一致性强
  - 可学习步长和正则化
- **局限**：训练需要配对数据；泛化性受限

#### 16.4.3 学习MRI采样模式

- **问题**：给定加速因子，如何选择k-space采样模式？
- **经典方法**：手工设计（可变密度随机采样）
- **学习方法**：
  - 将采样掩码$\Omega$参数化
  - 端到端训练：采样模式→重建→损失→梯度反传
  - 离散优化（Gumbel-Softmax等）
- **与源条件的联系**：采样模式的选择等价于选择正向算子$A$的行——影响正则化解的误差界

学习型方法用数据替代了手工先验，但仍有两个根本局限：(1) 训练需要配对数据；(2) 重建结果是点估计，无法量化不确定性。而扩散先验方法同时解决了这两个问题——这正是第13章"条件扩散=逆问题求解"在医学成像中的落地。

**来源**：Bologna_UNet_example.ipynb；Bologna_LGS_example.ipynb；Adler & Öktem (2018) Learned Primal-Dual；Benning L2（学习采样模式）

---

### 16.5 扩散先验重建

> **核心观点**：扩散先验重建是全书核心论点在医学成像中的落地——用预训练的扩散模型作为隐式先验，通过条件采样求解CT/MRI逆问题。无需配对训练数据、可量化不确定性、零样本迁移到新任务——这三重优势（第13章13.6节）在医学成像中尤为宝贵。

#### 16.5.1 从传统先验到扩散先验：回到第13章

- **先验演化回顾**：
  - 显式先验（第2章）：TV、小波稀疏 → 表达能力有限
  - 学习型先验（16.4）：UNet/展开 → 需要配对数据、无不确定性量化
  - 扩散先验（第13章）：隐式先验 + 条件采样 → 无需配对数据 + 不确定性量化
- **条件扩散采样的统一框架**（回顾第13章）：
  $$\nabla_{x_t} \log p(x_t|y) = \underbrace{\nabla_{x_t} \log p(x_t)}_{\text{扩散先验}} + \underbrace{\nabla_{x_t} \log p(y|x_t)}_{\text{似然得分}}$$

#### 16.5.2 DiffPIR for CT/MRI

- **DiffPIR算法回顾**（第13章13.3.4节）：
  - 隐空间优化+扩散去噪交替
  - 每步：预测$\hat{x}_0$→数据一致性步→去噪步
- **CT上的DiffPIR**：
  - 正向算子：$A$ = Radon变换（或ASTRA实现的离散版）
  - 数据一致性步：$x_t \leftarrow x_t - \zeta A^T(A\hat{x}_0 - y)$
- **MRI上的DiffPIR**：
  - 正向算子：$A = M_\Omega F$（欠采样傅里叶算子）
  - 数据一致性步更简单：$x_t \leftarrow x_t - \zeta F^T M_\Omega^T(F\hat{x}_0 - y)$

#### 16.5.3 DDRM与DPS for CT/MRI

- **DDRM**（第13章13.3.1节）：
  - 基于SVD分解的方法——适用于线性问题
  - CT：SVD可由ASTRA计算，但大规模时计算代价高
  - MRI：$A = M_\Omega F$，SVD退化为对角化（特别高效）
- **DPS**（第13章13.3.2节）：
  - Laplace近似：$p(y|x_t) \approx p(y|\hat{x}_{0|t})$
  - CT：$\nabla_{x_t} \log p(y|x_t) \approx \zeta A^T(y - A\hat{x}_{0|t})$
  - MRI：$\nabla_{x_t} \log p(y|x_t) \approx \zeta F^T M_\Omega^T(y - M_\Omega F\hat{x}_{0|t})$
  - 缩放因子$\zeta$的选择

#### 16.5.4 方法对比与选择指南

- **对比维度**：
  | 方法 | 是否需要训练 | 不确定性量化 | 非线性问题 | 计算代价 |
  |------|------------|------------|-----------|---------|
  | FBP/零填充 | 否 | 否 | 否 | 低 |
  | 正则化(TV/CS) | 否 | 否 | 否 | 中 |
  | UNet端到端 | 是（监督） | 否 | 是 | 低 |
  | 展开网络 | 是（监督） | 否 | 是 | 中 |
  | DiffPIR | 否（预训练扩散） | 是 | 是 | 高 |
  | DDRM | 否（预训练扩散） | 是 | 仅线性 | 高 |
  | DPS | 否（预训练扩散） | 是 | 是 | 高 |
- **选择指南**：
  - 高SNR全角CT → FBP即可
  - 稀疏投影CT → 正则化/展开网络
  - 有限角CT → 扩散先验
  - 快速MRI → 并行成像+压缩感知
  - 极端欠采样MRI → 扩散先验
  - 需要不确定性量化 → 扩散先验（多次采样）

从FBP到扩散先验，CT/MRI重建方法的演化完美映射了全书的叙事弧——从显式先验到隐式先验、从点估计到后验采样、从手工设计到数据驱动。

**来源**：Chung et al. (2023) DPS；DDRM (Kawar et al. 2022)；DiffPIR (Zhu et al. 2023)；deepinv demo_ddrm/diffpir；MiniProject_DenoisingPrior

---

### 16.6 本章小结：从物理到先验的统一

- **CT/MRI的统一逆问题视角**：
  - CT：Radon变换→线积分→频域径向采样
  - MRI：傅里叶变换→k-space采样
  - 统一：$y = Ax + \epsilon$，不同的$A$，相同的逆问题框架
- **先验演化的统一**：
  - 物理先验（FBP/SVD）→ 正则化先验（TV/小波/CS）→ 学习型先验（UNet/展开）→ 扩散先验
  - 每一步都是前一步局限性的回应
- **与全书论点的呼应**：
  - "扩散模型是逆问题自然发展的终点"在医学成像中得到验证
  - 扩散先验的三重优势：任意复杂先验、不确定性量化、零样本迁移
- **展望**：自监督训练（第17章）将解决医学成像中配对数据稀缺的问题

---

### 附录16A Radon变换与Fourier切片定理的严格证明

> 定位：16.1.3节给出了定理陈述和直觉，严格证明放入附录以保持主线流畅。

- Radon变换的函数空间性质
- Fourier切片定理的完整证明
- FBP公式的推导

### 附录16B Bloch方程的完整推导

> 定位：16.3.1节简述了Bloch方程，完整推导放入附录。

- Bloch方程的矩阵形式
- 旋转矩阵的解析解
- 弛豫项的解
- 稳态信号的推导

### 附录16C 波前集理论与有限角CT的数学框架

> 定位：16.2.3节介绍了波前集的直觉和结论，数学框架放入附录。

- 波前集的严格定义（基于局部傅里叶变换）
- 有限角Radon变换的微局部性质
- Quinto定理：有限角CT恢复的波前集
- 剪切波与不可见波前集的恢复

### 附录16D PET与发射断层成像

> 定位：PET是CT的重要变体（发射而非透射），但与主线叙事关系较远，放入附录。

- PET成像原理：正电子湮灭→511keV光子对
- PGET：被动伽马发射断层成像
- 发射断层的非线性正模型
- Levenberg-Marquardt迭代重建

---

## 素材来源映射

| 节 | 核心来源 | 补充来源 |
|---|---|---|
| 16.1 | IP_and_Im_Lectures-master tomography.md；Siltanen Day2 P37-97 | Siltanen Day1 exercises（2×2像素断层） |
| 16.2 | Siltanen Day3A P28-63；Siltanen Day2 P125-162 | Davison 1983；Natterer 1986；Frikel & Quinto 2013 |
| 16.3 | IP_and_Im_Lectures-master magnetic_resonance_imaging.md | Lustig et al. (2008) CS-MRI |
| 16.4 | Bologna_UNet_example.ipynb；Bologna_LGS_example.ipynb | Adler & Öktem (2018)；Benning L2 |
| 16.5 | Chung et al. (2023)；deepinv demo_ddrm/diffpir | MiniProject_DenoisingPrior |
| 附录16A | IP_and_Im_Lectures-master tomography.md L177-210 | Siltanen Day3A P51 |
| 附录16B | IP_and_Im_Lectures-master magnetic_resonance_imaging.md L42-115 | Bloch (1946) |
| 附录16C | Siltanen Day3A P55-63 | Greenleaf & Uhlmann 1989；Quinto 1993 |
| 附录16D | Siltanen Day3B P1-54 | IAEA PGET Challenge |

## 章节逻辑流

```
16.1 CT物理与正向模型 ──→ 16.2 CT不适定性与正则化 ──→ 16.3 MRI物理与k-space
       │                        │                          │
       │ Radon变换               │ 显式先验的天花板           │ 傅里叶采样
       │ FBP                     │ TV/小波/剪切波             │ 欠采样+CS
       ▼                        ▼                          ▼
       └────────────────────→ 16.4 学习型重建 ←─────────────┘
                                │
                                │ 数据驱动先验
                                │ 但缺少不确定性量化
                                ▼
                           16.5 扩散先验重建
                                │
                                │ 条件扩散=逆问题求解
                                │ 不确定性量化+零样本
                                ▼
                           16.6 小结：物理→先验的统一
```

## 缺失素材清单

| 素材 | 用途 | 紧急度 | 建议来源 |
|------|------|--------|---------|
| CT/MRI重建结果对比图（FBP vs TV vs UNet vs 扩散） | 16.5.4对比表的可视化 | ⭐⭐⭐ | 自行生成或从论文获取 |
| sinogram示意图 | 16.1.2 sinogram可视化 | ⭐⭐⭐ | skimage Shepp-Logan phantom |
| FBP vs 反投影对比图 | 16.1.4 FBP效果展示 | ⭐⭐⭐ | skimage iradon |
| 有限角CT SVD衰减对比图 | 16.2.2 全角vs有限角 | ⭐⭐⭐ | Siltanen Day3A P44 |
| 波前集示意图 | 16.2.3 可见/不可见边缘 | ⭐⭐ | Siltanen Day3A P57 |
| k-space轨迹图 | 16.3.2 Cartesian/径向/螺旋 | ⭐⭐ | IP_and_Im_Lectures-master MRI |
| 欠采样掩码与零填充重建对比图 | 16.3.3 欠采样效果 | ⭐⭐⭐ | 自行生成 |
| MRI并行成像示意图 | 16.3.4 多线圈重建 | ⭐⭐ | IP_and_Im_Lectures-master MRI |
| 压缩感知MRI对比图 | 16.3.4 CS-MRI效果 | ⭐⭐ | Lustig et al. (2008) |
| UNet端到端CT重建结果图 | 16.4.1 | ⭐⭐ | Bologna_UNet_example |
| Learned Gradient Descent结果图 | 16.4.2 | ⭐⭐ | Bologna_LGS_example |
| DiffPIR/DDRM/DPS CT/MRI重建结果对比图 | 16.5.2-16.5.3 | ⭐⭐⭐ | deepinv demos或论文 |
