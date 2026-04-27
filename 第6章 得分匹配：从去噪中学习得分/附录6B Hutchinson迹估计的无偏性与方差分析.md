# 附录6B Hutchinson迹估计的无偏性与方差分析

> 定位：为6.4节提供Hutchinson迹估计的数学基础。证明无偏性，分析方差，比较不同采样分布的效率。

## 无偏性证明

**定理（Hutchinson迹估计）**：设 $A \in \mathbb{R}^{d \times d}$ 是任意方阵，$v \in \mathbb{R}^d$ 是满足 $\mathbb{E}[vv^T] = I$ 的随机向量，则

$$\text{Tr}(A) = \mathbb{E}[v^T A v]$$

**证明**：

$$\mathbb{E}[v^T A v] = \mathbb{E}[\text{Tr}(v^T A v)]$$

因为 $v^T A v$ 是标量，其迹等于自身。

$$= \mathbb{E}[\text{Tr}(A v v^T)]$$

利用迹的循环性质：$\text{Tr}(BC) = \text{Tr}(CB)$，取 $B = A$，$C = vv^T$。

$$= \text{Tr}(\mathbb{E}[A v v^T])$$

交换期望与迹（迹是线性运算）。

$$= \text{Tr}(A\,\mathbb{E}[vv^T])$$

$A$ 与 $v$ 无关，提取出来。

$$= \text{Tr}(A \cdot I) = \text{Tr}(A)$$

由假设 $\mathbb{E}[vv^T] = I$。$\square$

## M次蒙特卡罗估计

给定 $M$ 个独立同分布的随机向量 $v_1, \ldots, v_M \sim p_v$，迹的无偏估计为：

$$\widehat{\text{Tr}}_M(A) = \frac{1}{M}\sum_{j=1}^M v_j^T A v_j$$

### 无偏性

$$\mathbb{E}[\widehat{\text{Tr}}_M(A)] = \frac{1}{M}\sum_{j=1}^M \mathbb{E}[v_j^T A v_j] = \frac{1}{M}\cdot M\cdot\text{Tr}(A) = \text{Tr}(A)$$

### 方差

由于 $v_1, \ldots, v_M$ 独立同分布：

$$\text{Var}(\widehat{\text{Tr}}_M) = \frac{\text{Var}(v^T A v)}{M}$$

$M$ 越大方差越小，但计算量正比于 $M$。

## 方差分析：不同采样分布

### 高斯分布

设 $v \sim \mathcal{N}(0, I_d)$，则 $v^T A v$ 的方差为：

$$\text{Var}(v^T A v) = 2\|A\|_F^2$$

其中 $\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$ 是Frobenius范数。

**推导概要**：利用高斯随机向量的四阶矩公式 $\mathbb{E}[v_i v_j v_k v_l] = \delta_{ij}\delta_{kl} + \delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk}$，计算 $\mathbb{E}[(v^T A v)^2]$ 和 $(\mathbb{E}[v^T A v])^2 = (\text{Tr}(A))^2$，两者之差即为方差。

### Rademacher分布

设 $v \in \{-1, +1\}^d$，各分量独立等概率取 $\pm 1$。此时 $\mathbb{E}[v_i v_j] = \delta_{ij}$（满足 $\mathbb{E}[vv^T] = I$），且 $\mathbb{E}[v_i^4] = 1$。

$$\text{Var}(v^T A v) = 2\|A\|_F^2 - 2\sum_{i=1}^d A_{ii}^2 = 2\left(\|A\|_F^2 - \|A \circ I\|_F^2\right)$$

其中 $\circ$ 表示Hadamard积（逐元素乘），$I$ 是单位矩阵。

**关键对比**：Rademacher的方差严格小于高斯的方差，除非 $A$ 是对角矩阵：

$$\text{Var}_{\text{Rademacher}}(v^T A v) = \text{Var}_{\text{Gauss}}(v^T A v) - 2\sum_i A_{ii}^2 \leq \text{Var}_{\text{Gauss}}(v^T A v)$$

因此**Rademacher分布更优**——同样的 $M$ 下方差更低。

## 在得分匹配中的应用

### Jacobian迹的估计

在SSM中，$A = \nabla_x s_\theta(x)$ 是得分网络的Jacobian矩阵。我们需要估计：

$$\text{Tr}(\nabla_x s_\theta(x)) = \mathbb{E}_v[v^T\nabla_x s_\theta(x)\,v]$$

### 前向自动微分的实现

$v^T\nabla_x s_\theta(x)\,v$ 不需要构造完整Jacobian矩阵，可以通过前向自动微分高效计算：

1. 定义方向导数函数：$g(t) = v^T s_\theta(x + t v)$
2. 计算 $g'(0) = v^T\nabla_x s_\theta(x)\,v$

在PyTorch等框架中，这可以通过 `torch.autograd.functional.jvp`（Jacobian-Vector Product）实现，仅需一次前向传播加一次前向自动微分。

### 计算复杂度

| 方法 | 每次迭代的计算量 | 精度 |
|---|---|---|
| 精确计算Jacobian迹 | $O(d)$ 次前向传播 | 精确 |
| Hutchinson $M=1$ | $O(1)$ 前向传播 + 1次JVP | 无偏，方差大 |
| Hutchinson $M=10$ | $O(10)$ JVP | 无偏，方差中等 |

实践中，$M = 1$ 即可获得无偏估计（虽然方差较大），多次平均可降低方差。SSM的训练通常取 $M = 1$，因为训练过程本身的随机性（小批量采样、SGD噪声）已经能平滑单次估计的方差。

**来源**：Hutchinson (1990) "A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines"; Song et al. (2019) "Sliced Score Matching"
