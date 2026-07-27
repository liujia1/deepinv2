# 附录13B DDRM算法的SVD推导

本附录给出DDRM（Denoising Diffusion Restoration Models）算法基于SVD分解的严格推导。DDRM利用线性算子的SVD结构，将向量值逆问题解耦为标量逆问题，在频谱域上实现精确的条件化。

## 1 线性算子A的SVD分解

### 1.1 SVD分解

设线性算子 $A \in \mathbb{R}^{m \times n}$，其奇异值分解（SVD）为：

$$A = U\Sigma V^\top$$

其中：
- $U \in \mathbb{R}^{m \times m}$ 是左奇异向量矩阵，满足 $U^\top U = UU^\top = I_m$
- $V \in \mathbb{R}^{n \times n}$ 是右奇异向量矩阵，满足 $V^\top V = VV^\top = I_n$
- $\Sigma \in \mathbb{R}^{m \times n}$ 是对角矩阵，对角元素为奇异值 $\{s_i\}_{i=1}^{\min(m,n)}$，满足 $s_1 \geq s_2 \geq \cdots \geq s_r > 0 = s_{r+1} = \cdots$，其中 $r = \text{rank}(A)$

### 1.2 变换到频谱域

定义频谱域变量：

$$\bar{y} = U^\top y \in \mathbb{R}^m, \quad \bar{x} = V^\top x \in \mathbb{R}^n, \quad \bar{\epsilon} = U^\top \epsilon \in \mathbb{R}^m$$

由于 $U$ 和 $V$ 是正交矩阵，上述变换是保距的（isometry）：$\|\bar{y}\| = \|y\|$，$\|\bar{x}\| = \|x\|$。

在频谱域下，线性测量模型 $y = Ax + \epsilon_y$（$\epsilon_y \sim \mathcal{N}(0, \sigma_y^2 I)$）变为：

$$\bar{y} = U^\top(Ax + \epsilon_y) = U^\top U\Sigma V^\top x + U^\top\epsilon_y = \Sigma\bar{x} + \bar{\epsilon}$$

由于 $U$ 是正交矩阵且 $\epsilon_y \sim \mathcal{N}(0, \sigma_y^2 I)$，有 $\bar{\epsilon} = U^\top\epsilon_y \sim \mathcal{N}(0, \sigma_y^2 I)$——噪声在频谱域下仍然是独立同分布的高斯噪声。

### 1.3 解耦为标量问题

将频谱域方程逐分量写出：

$$\bar{y}_i = s_i\bar{x}_i + \sigma_y\bar{\epsilon}_i, \quad i = 1, \ldots, \min(m, n)$$

对于 $i > r$（即 $s_i = 0$），方程退化为 $\bar{y}_i = \sigma_y\bar{\epsilon}_i$——观测不包含关于 $\bar{x}_i$ 的任何信息，这些分量完全由先验决定。

**关键洞察**：SVD将向量值逆问题 $y = Ax + \epsilon_y$ **解耦为 $n$ 个独立的标量逆问题**——每个频谱分量 $\bar{x}_i$ 可以独立地根据 $\bar{y}_i$ 和 $s_i$ 进行推断。这是DDRM精确条件化的数学基础。

## 2 频谱域上的条件化

### 2.1 条件分布的因子化

由于频谱域解耦，后验分布因子化为：

$$p(\bar{x}|y) = p(\bar{x}|\bar{y}) \propto p(\bar{y}|\bar{x})\,p(\bar{x})$$

在独立高斯噪声假设下，似然函数因子化：

$$p(\bar{y}|\bar{x}) = \prod_{i=1}^{m} p(\bar{y}_i|\bar{x}_i) = \prod_{i=1}^{m} \mathcal{N}(\bar{y}_i; s_i\bar{x}_i, \sigma_y^2)$$

然而，先验 $p(\bar{x})$ 一般**不能**因子化——因为图像的频谱分量之间存在复杂的相关性（这正是自然图像先验的复杂之处）。这意味着我们不能简单地对每个 $\bar{x}_i$ 独立采样然后组合。

### 2.2 DDRM的策略：在扩散采样中注入频谱域条件

DDRM的策略不是直接对因子化的后验采样，而是在扩散逆向采样的每一步中，对频谱域的每个分量独立施加条件约束。具体而言，DDRM将逆向采样步骤中产生的含噪状态 $x_t$ 变换到频谱域 $\bar{x}_t = V^\top x_t$，对每个频谱分量 $\bar{x}_t^i$ 独立施加条件化，再变换回原始域。

这一策略的可行性依赖于一个关键事实：**扩散过程的噪声是各向同性的**——在VP-SDE和VE-SDE中，加入的噪声是 $\sigma_t\epsilon$，$\epsilon \sim \mathcal{N}(0, I)$。各向同性噪声在正交变换下保持不变，因此频谱域中的扩散过程与原始域中的扩散过程具有相同的结构。这使得我们可以安全地在频谱域中进行条件化操作。

## 3 DDRM的逐分量条件化公式

### 3.1 条件化准则的推导

DDRM的核心是：在逆向采样的每一步 $t \to t-1$，对每个频谱分量 $\bar{x}_t^i$，根据该分量的信噪比选择不同的条件化策略。

考虑时刻 $t$ 的含噪状态 $x_t$。在VP-SDE下，其与干净图像 $x_0$ 的关系为 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$。变换到频谱域：

$$\bar{x}_t^i = \sqrt{\bar\alpha_t}\,\bar{x}_0^i + \sqrt{1-\bar\alpha_t}\,\bar{\epsilon}_t^i$$

其中 $\bar{\epsilon}_t^i \sim \mathcal{N}(0, 1)$。在频谱域中，测量方程为 $\bar{y}_i = s_i\bar{x}_0^i + \sigma_y\bar{\epsilon}_i$。

DDRM的关键判断是：对于每个频谱分量，**当前扩散噪声水平与测量噪声水平的相对大小**决定了应该信任先验还是测量。

### 3.2 三种情况的条件化

定义扩散噪声方差为 $\sigma_t^2 = 1 - \bar\alpha_t$（VP-SDE下），测量噪声在频谱域中的有效方差为 $\sigma_y^2/s_i^2$（注意到 $s_i\bar{x}_0^i$ 的测量噪声为 $\sigma_y\bar{\epsilon}_i$，归一化到 $\bar{x}_0^i$ 后有效方差为 $\sigma_y^2/s_i^2$）。

**情况一：$s_i = 0$（零空间分量）**

当 $s_i = 0$ 时，观测 $\bar{y}_i$ 不包含关于 $\bar{x}_0^i$ 的任何信息——这个分量完全处于算子 $A$ 的零空间中。此时只能依赖先验：

$$p(\bar{x}_{t-1}^i | x_t, y) = \mathcal{N}(\bar{x}_{t-1}^i;\, \hat{\bar{x}}_{0|t}^i,\, \sigma_{t-1}^2)$$

其中 $\hat{\bar{x}}_{0|t}^i$ 是先验后验均值（由Tweedie估计给出），$\sigma_{t-1}^2$ 是下一步的扩散噪声方差。

**情况二：$\sigma_t < \sigma_y/s_i$（测量噪声主导）**

当 $\sigma_t < \sigma_y/s_i$ 时，当前扩散噪声水平低于测量噪声的有效水平——意味着扩散过程已经去除了大部分噪声，图像在该频谱分量上的估计精度**高于**测量能够提供的精度。此时，信任测量反而引入更多噪声，因此仍应依赖先验：

$$p(\bar{x}_{t-1}^i | x_t, y) = \mathcal{N}(\bar{x}_{t-1}^i;\, \hat{\bar{x}}_{0|t}^i,\, \sigma_{t-1}^2)$$

与情况一相同的先验驱动采样。

**情况三：$\sigma_t \geq \sigma_y/s_i$（扩散噪声主导）**

当 $\sigma_t \geq \sigma_y/s_i$ 时，当前扩散噪声水平不低于测量噪声的有效水平——意味着测量提供的信息比当前含噪状态的估计更可靠。此时应信任测量，用测量信息条件化：

$$p(\bar{x}_{t-1}^i | x_t, y) = \mathcal{N}\!\left(\bar{x}_{t-1}^i;\, \frac{\bar{y}_i}{s_i},\, \frac{\sigma_y^2}{s_i^2} - \sigma_t^2\right)$$

**推导**：这个条件分布可以严格推导。在频谱域中，$\bar{x}_0^i$ 的后验分布为：

$$p(\bar{x}_0^i | \bar{y}_i) \propto p(\bar{y}_i | \bar{x}_0^i)\,p(\bar{x}_0^i)$$

忽略先验 $p(\bar{x}_0^i)$ 的影响（在扩散噪声主导时，似然的信息量更大），仅考虑似然 $p(\bar{y}_i | \bar{x}_0^i) = \mathcal{N}(\bar{y}_i; s_i\bar{x}_0^i, \sigma_y^2)$，对 $\bar{x}_0^i$ 的最大似然估计为 $\bar{y}_i/s_i$。

在时刻 $t-1$，含噪状态 $\bar{x}_{t-1}^i$ 的分布为 $\mathcal{N}(\sqrt{\bar\alpha_{t-1}}\bar{x}_0^i, \sigma_{t-1}^2)$（先验驱动），或条件化后为关于 $\bar{y}_i/s_i$ 的高斯。两种信息源的结合（先验扩散步骤 + 测量条件化）给出方差为 $\sigma_y^2/s_i^2 - \sigma_t^2$——这是测量信息中扣除已被扩散过程解释的部分后的剩余方差。

### 3.3 物理解释：从"信任先验"到"信任测量"的过渡

DDRM的三种情况揭示了一个优美的物理图像：**随着逆向采样从高噪声到低噪声推进，每个频谱分量经历一个从"信任先验"到"信任测量"的过渡**。

具体地，固定频谱分量 $i$（固定 $s_i$），随着 $t$ 从 $T$ 减小到 $0$：

1. **高噪声阶段**（$t$ 大，$\sigma_t$ 大）：$\sigma_t \geq \sigma_y/s_i$，扩散噪声主导→信任测量，使用 $\bar{y}_i/s_i$ 作为条件均值
2. **低噪声阶段**（$t$ 小，$\sigma_t$ 小）：$\sigma_t < \sigma_y/s_i$，测量噪声主导→信任先验，使用Tweedie估计

过渡时间点 $t_i^*$ 满足 $\sigma_{t_i^*} = \sigma_y/s_i$，即：

$$t_i^* = \arg\min_t \{t : \sigma_t \geq \sigma_y/s_i\}$$

**奇异值 $s_i$ 越大的分量，过渡越早**——因为大奇异值意味着该分量在测量中被更好地保留，测量信息更早变得可靠。这解释了为什么DDRM能自然地处理不同频谱分量的信噪比差异——条件化策略自动适应每个分量的信息量。

**奇异值 $s_i = 0$ 的分量（零空间），始终信任先验**——因为测量不提供任何关于这些分量的信息，只能依赖先验知识来推断。

## 4 DDRM with $\eta$ parameter

### 4.1 随机性控制参数 $\eta$

DDRM引入了一个随机性控制参数 $\eta \in (0, 1]$，其角色类似于DDIM（第7章7.4节）中的 $\eta$——控制逆向采样过程的随机性。

在DDRM的逐分量条件化中，$\eta$ 控制条件分布的方差：

- 当 $\eta = 1$ 时，使用完整的条件方差（上述三种情况中的方差），实现完全随机采样——后验的多样性被完整保留
- 当 $\eta \to 0$ 时，条件分布的方差趋于零，采样过程趋于确定性——每次运行产生相同的结果

### 4.2 $\eta$ 的数学表达

DDRM中带 $\eta$ 的条件化公式为：

**情况三**（$\sigma_t \geq \sigma_y/s_i$）：

$$p(\bar{x}_{t-1}^i | x_t, y) = \mathcal{N}\!\left(\bar{x}_{t-1}^i;\, \frac{\bar{y}_i}{s_i},\, \eta^2\left(\frac{\sigma_y^2}{s_i^2} - \sigma_t^2\right)\right)$$

**情况一和二**（先验驱动）：

$$p(\bar{x}_{t-1}^i | x_t, y) = \mathcal{N}\!\left(\bar{x}_{t-1}^i;\, \hat{\bar{x}}_{0|t}^i,\, \eta^2\sigma_{t-1}^2\right)$$

### 4.3 $\eta$ 与采样质量的关系

$\eta$ 的选择影响采样结果的多样性和鲁棒性：

- **$\eta = 1$**：完全随机采样，后验的多个峰都能被探索到。适合不确定性量化，但单次采样的重建质量可能不够好
- **$\eta \in [0.5, 0.85]$**：实践中常用的范围，在多样性和重建质量之间取得平衡。与DDIM中 $\eta = 0.5$ 或 $0.75$ 的经验一致
- **$\eta \to 0$**：确定性采样，类似于条件ODE求解。单次采样的重建质量最高，但丧失了不确定性量化能力——每次运行产生相同的结果

## 5 DDRM vs DPS在线性问题下的对比

### 5.1 数学结构的对比

| 维度 | DDRM | DPS |
|---|---|---|
| 条件化方式 | 频谱域逐分量精确条件化 | 原始域全局近似条件化 |
| 似然得分计算 | SVD解耦后直接计算 $\bar{y}_i/s_i$ | Tweedie估计 + $A^\top(y - A\hat{x}_0)/\sigma_y^2$ |
| 近似来源 | 无（在线性问题下精确） | delta函数近似 + Jacobian省略 |
| 对 $A$ 的要求 | 必须能计算SVD | 任意（只需前向 $A$ 和反传 $A^\top$） |
| 零空间处理 | 自然分离（$s_i = 0$ 分量纯先验） | 隐式（通过先验得分驱动） |
| 计算代价 | SVD预计算 + 频谱域条件化 | 每步一次额外前向/反传 |

### 5.2 精度对比

**在线性问题下**，DDRM的条件化是精确的——频谱域解耦后每个标量子问题的条件分布可以直接计算，无需任何近似。相比之下，DPS引入了delta函数近似和Jacobian省略，理论上不如DDRM精确。

实验上，在线性逆问题（如高斯去模糊、超分辨率、inpainting）上，DDRM通常获得比DPS更高的PSNR和更稳定的重建结果——特别是在测量噪声较低的场景下，DDRM的精确条件化优势更明显。

### 5.3 适用范围对比

**DPS的适用范围更广**：DPS不要求算子 $A$ 有SVD分解——它只需要能够计算 $A\hat{x}_{0|t}$（前向传播）和 $A^\top(y - A\hat{x}_{0|t})$（反传）。这使得DPS可以处理非线性算子 $A$（如相位恢复、非线性散射等），而DDRM仅适用于线性算子。

**DDRM的精度更高**：在线性问题下，DDRM利用SVD实现了频谱域的精确条件化，避免了DPS的近似误差。此外，DDRM的条件化策略自动适应每个频谱分量的信噪比——大奇异值分量更早被测量约束，小奇异值分量更依赖先验——这种自适应机制在理论上更合理。

### 5.4 计算代价对比

**DDRM的预计算代价**：需要计算 $A$ 的SVD分解，对于大尺寸算子（如 $A \in \mathbb{R}^{m \times n}$，$n$ 为图像像素数），SVD的计算复杂度为 $O(\min(m,n)^2 \max(m,n))$。对于某些特定算子（如模糊算子的循环卷积结构），可以利用FFT加速SVD计算。但一般而言，SVD的计算代价限制了DDRM在大规模问题上的应用。

**DPS的逐步代价**：每步需要一次 $A$ 的前向传播和一次 $A^\top$ 的反传。对于具有快速算法的算子（如CT的Radon变换、模糊的FFT卷积），这些操作的计算量远小于SVD。

**总结**：DDM适合中等规模、线性算子有快速SVD的场景；DPS适合大规模或非线性算子的场景。两者在线性问题上互补，选择取决于算子结构和问题规模。

### 5.5 统一视角

从后验得分分解的角度看，DDRM和DPS的差异可以统一理解：

$$\nabla_{x_t}\log p(x_t|y) = \underbrace{\nabla_{x_t}\log p(x_t)}_{\text{先验得分（两者相同）}} + \underbrace{\nabla_{x_t}\log p(y|x_t)}_{\text{似然得分（近似策略不同）}}$$

- **DDRM**：通过SVD在频谱域精确计算 $\nabla_{x_t}\log p(y|x_t)$，但仅限线性 $A$
- **DPS**：通过Tweedie近似计算 $\nabla_{x_t}\log p(y|x_t)$，但适用于任意 $A$

两者的先验得分完全相同（使用同一个预训练扩散模型），差异仅在于似然得分的计算方式——精确但受限 vs. 近似但通用。这一对比再次印证了13.3.6节的核心论点：**所有条件扩散采样方法本质上都在近似 $\nabla\log p(y|x_t)$，差异在于近似策略与精度-效率权衡**。

## 附录小结

DDRM算法的SVD推导揭示了线性逆问题中条件扩散采样的一个特殊结构：**SVD将向量值问题解耦为标量问题，使得精确条件化成为可能**。

DDRM的核心公式总结：

| 条件 | 条件均值 | 条件方差 | 物理含义 |
|---|---|---|---|
| $s_i = 0$ | $\hat{\bar{x}}_{0\|t}^i$（Tweedie估计） | $\eta^2\sigma_{t-1}^2$ | 零空间：纯先验 |
| $\sigma_t < \sigma_y/s_i$ | $\hat{\bar{x}}_{0\|t}^i$（Tweedie估计） | $\eta^2\sigma_{t-1}^2$ | 测量噪声主导：信任先验 |
| $\sigma_t \geq \sigma_y/s_i$ | $\bar{y}_i/s_i$ | $\eta^2(\sigma_y^2/s_i^2 - \sigma_t^2)$ | 扩散噪声主导：信任测量 |

DDRM的局限与DPS的互补性构成了第13章方法选择的重要维度：当算子线性且SVD可得时，DDRM提供更精确的条件化；当算子非线性或SVD不可得时，DPS提供更灵活的近似方案。

**来源**：Kawar et al. (2022) "Denoising Diffusion Restoration Models"；Chung et al. (2209.14687) §3.1；第7章7.2-7.4节VP-SDE与DDIM
