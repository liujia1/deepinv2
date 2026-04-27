# 附录11B 连续时间VLB与VDM

> 定位：为11.4节提供连续时间VLB的补充材料。Kingma (2021) 的Variational Diffusion Models (VDM) 将离散时间VLB推广到连续时间，允许噪声调度作为变分参数被优化。

## 从离散VLB到连续VLB

### 离散VLB回顾

第10-11章推导的离散时间VLB：

$$L_\text{VLB} = \underbrace{D_\text{KL}(q(x_T|x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^T \underbrace{\mathbb{E}_q[D_\text{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))]}_{L_{t-1}} - \underbrace{\mathbb{E}_q[\log p_\theta(x_0|x_1)]}_{\text{负}L_0}$$

当 $T \to \infty$，$\Delta t = 1/T \to 0$，离散求和趋于连续积分。

### 连续时间VLB的形式

在连续时间 $t \in [0, 1]$ 下，VLB变为：

$$L_\text{VLB}^\text{cont} = \underbrace{D_\text{KL}(q(x_1|x_0) \| p(x_1))}_{L_1} + \int_0^1 \mathbb{E}_q\left[\frac{1}{2}\left\|\frac{\mu_\theta(x_t, t) - \tilde\mu_t(x_t, x_0)}{\sigma_t}\right\|^2\right]dt - \underbrace{\mathbb{E}_q[\log p_\theta(x_0|x_{\epsilon})]}_{L_0}$$

其中 $x_\epsilon$ 表示 $t \to 0^+$ 时的含噪状态。

## VDM的核心贡献

### 噪声调度作为变分参数

Kingma (2021) 的关键洞察：**噪声调度 $\beta(t)$ 不必是固定超参数——它可以是变分参数，与 $\theta$ 一起被优化**。

在离散DDPM中，$\beta_1, \ldots, \beta_T$ 是预设值（如线性调度或余弦调度）。这些值的选择对训练效果有显著影响，但需要手动调参。

VDM将 $\beta(t)$ 参数化为可微函数（如用单调神经网络或参数化的分段函数），通过梯度下降同时优化 $\theta$ 和 $\beta(t)$。

### 噪声调度的优化方向

VDM的实验发现最优噪声调度是高度非线性的：

- **更多时间步分配在"困难"区域**：信噪比变化剧烈的区域需要更精细的时间步
- **更少时间步分配在"容易"区域**：信噪比变化缓慢的区域可以用更大的时间步
- 最优调度在视觉上更接近余弦调度而非线性调度——验证了Nichol & Dhariwal (2021) 的经验观察

### 与SDE视角的对比

第7章的SDE框架中，漂移系数 $f(x,t)$ 和扩散系数 $g(t)$ 是固定选择（VE-SDE或VP-SDE）。VDM将这些系数视为可优化对象，提供了更灵活的框架。

SDE视角与VDM视角的关系：
- SDE视角：固定正向过程 $f, g$，学习得分函数 $s_\theta$
- VDM视角：同时优化正向过程（噪声调度）和逆向过程（网络参数）

### VDM的三种等价参数化

Kingma (2021) 证明：在连续时间下，三种参数化（噪声预测、x₀预测、得分预测）也是等价的。

连续时间下ε预测VLB的权重与得分匹配的权重一致——这进一步强化了第12章等价性的结论。具体地，连续时间VLB的最优权重恰好等于DSM的最优权重 $\lambda(\sigma)$。

## VDM的实践意义

### 自适应噪声调度

VDM允许噪声调度自动适应数据集——无需手动调参。不同数据集（如CIFAR-10 vs ImageNet）的最优噪声调度可能不同，VDM可以自动找到适合的调度。

### 似然优化

VDM的连续时间VLB可以直接用于优化数据似然——这对于需要精确概率模型的应用（如数据压缩、异常检测）非常重要。

### 与其他生成模型的统一

VDM的框架可以统一：
- **DDPM**：固定噪声调度 + 简化VLB
- **Score-SDE**：固定SDE + 得分匹配
- **VDM**：可优化噪声调度 + 连续时间VLB

三者是VDM框架在不同约束条件下的特例。

## 局限性

- VDM的优化涉及同时对 $\theta$ 和噪声调度求梯度，计算代价高于固定调度的DDPM
- 噪声调度的优化可能导致过拟合——需要正则化
- 连续时间VLB的理论分析比离散情况更复杂

**来源**：Kingma et al. (2021) VDM; Kingma & Gao (2023) Understanding Diffusion Objectives; 2406.08929v2 Sec 2.3
