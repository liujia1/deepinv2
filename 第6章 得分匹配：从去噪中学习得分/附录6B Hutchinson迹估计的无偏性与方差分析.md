# 附录6B Hutchinson 迹估计的无偏性与方差分析

> 定位：给 6.4 节的 Hutchinson 迹估计补上数学基础——证明它无偏，分析方差，并比较不同采样分布（高斯 vs Rademacher）谁更高效。这些细节放附录，主线才能专注讲 SSM 的思想。

## 为什么需要这个分析

6.4 节我们说：用随机向量 $v$ 做 $v^TAv$ 的期望，就能无偏地估出迹 $\mathrm{Tr}(A)$，从而让 SSM 不必构造完整 Jacobian。但"无偏"只是及格线——你一定还想知道：**这个估计有多抖？用哪种随机向量更稳？** 本附录回答这两点：先证无偏性，再比较高斯与 Rademacher 两种采样的方差，结论是 Rademacher 几乎总更优。

> 下面的证明偏技术，**可跳过**；记住"无偏、方差随 $M$ 按 $1/M$ 降、Rademacher 比高斯更稳"即可。

## 无偏性证明

**定理（Hutchinson 1990）**：设 $A\in\mathbb R^{d\times d}$ 任意方阵，$v$ 满足 $\mathbb{E}[vv^T]=I$，则

$$\mathrm{Tr}(A)=\mathbb{E}[v^T A v].$$

**证明**（只用迹的两条性质：标量迹等于自身、循环性质 $\mathrm{Tr}(BC)=\mathrm{Tr}(CB)$）：

$$\mathbb{E}[v^T A v]=\mathbb{E}[\mathrm{Tr}(v^T A v)]=\mathbb{E}[\mathrm{Tr}(A v v^T)]=\mathrm{Tr}(A\,\mathbb{E}[vv^T])=\mathrm{Tr}(A\cdot I)=\mathrm{Tr}(A). \quad\square$$

## M 次蒙特卡罗估计

给定 $M$ 个独立同分布样本 $v_1,\dots,v_M\sim p_v$，无偏估计为

$$\widehat{\mathrm{Tr}}_M(A)=\frac{1}{M}\sum_{j=1}^M v_j^T A v_j.$$

**无偏性**：$\mathbb{E}[\widehat{\mathrm{Tr}}_M]=\mathrm{Tr}(A)$。

**方差**：因独立同分布，$\mathrm{Var}(\widehat{\mathrm{Tr}}_M)=\mathrm{Var}(v^T A v)/M$。$M$ 越大方差越小，但计算量正比于 $M$。

## 方差分析：不同采样分布

**高斯** $v\sim\mathcal N(0,I)$：方差

$$\mathrm{Var}(v^T A v)=2\|A\|_F^2,$$

其中 $\|A\|_F=\sqrt{\sum_{i,j}A_{ij}^2}$ 是 Frobenius 范数。推导用高斯四阶矩 $\mathbb{E}[v_i v_j v_k v_l]=\delta_{ij}\delta_{kl}+\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk}$。

**Rademacher** $v\in\{-1,+1\}^d$，各分量独立等概率取 $\pm1$。此时 $\mathbb{E}[v_i v_j]=\delta_{ij}$，$\mathbb{E}[v_i^4]=1$，方差

$$\mathrm{Var}(v^T A v)=2\|A\|_F^2-2\sum_{i=1}^d A_{ii}^2=2\big(\|A\|_F^2-\|A\circ I\|_F^2\big),$$

其中 $\circ$ 是逐元素乘（Hadamard 积）。

**关键对比**：Rademacher 方差严格小于高斯，除非 $A$ 是对角阵：

$$\mathrm{Var}_{\text{Rademacher}}-\mathrm{Var}_{\text{Gauss}}=-2\sum_i A_{ii}^2\le0.$$

所以**实践里推荐 Rademacher**——同样 $M$ 下方差更低。

## 在得分匹配中的应用

SSM 里 $A=\nabla_x s_\theta(x)$ 是得分网络的 Jacobian。要估 $\mathrm{Tr}(\nabla_x s_\theta)=\mathbb{E}_v[v^T\nabla_x s_\theta\,v]$。

**前向自动微分实现**：不需构造完整 Jacobian。定义 $g(t)=v^T s_\theta(x+tv)$，则 $g'(0)=v^T\nabla_x s_\theta(x)\,v$。在 PyTorch 里用 `torch.autograd.functional.jvp`（Jacobian–Vector Product），一次前向传播加一次前向自动微分即可。

**计算复杂度对比**：

| 方法 | 每次迭代成本 | 精度 |
|---|---|---|
| 精确 Jacobian 迹 | $O(d)$ 次前向 | 精确 |
| Hutchinson $M=1$ | $O(1)$ 前向 + 1 次 JVP | 无偏，方差大 |
| Hutchinson $M=10$ | $O(10)$ JVP | 无偏，方差中 |

实践中 $M=1$ 就能拿无偏估计（方差大点无所谓），多次平均再降方差。SSM 训练常取 $M=1$，因为训练本身的小批量采样和 SGD 噪声已经能平滑单次估计的方差。

**来源**：Hutchinson (1990); Song et al. (2019) Sliced Score Matching
