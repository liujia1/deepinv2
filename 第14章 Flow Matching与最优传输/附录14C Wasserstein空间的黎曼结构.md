# 附录14C Wasserstein空间的黎曼结构

> **定位**：对理解OT与生成模型的关系有深层价值，但涉及黎曼几何，超出本书主要读者群体的背景。本附录提供概念性介绍，严格的数学处理参见Villani (2008) 和Ambrosio et al. (2008)。

## 概率分布空间上的黎曼结构

### 动机

14.1节定义了Wasserstein距离 $W_2$，它赋予概率分布空间 $\mathcal{P}_2(\mathbb{R}^d)$ 以度量结构。一个深刻的问题是：这个度量空间是否具有更丰富的几何结构——例如黎曼结构？答案是肯定的。

### Wasserstein空间的切空间

在黎曼几何中，切空间描述了流形上某一点的"无穷小邻域"。在Wasserstein空间 $\mathcal{P}_2(\mathbb{R}^d)$ 中，概率分布 $\mu$ 处的"切向量"是一个向量场 $v: \mathbb{R}^d \to \mathbb{R}^d$，它描述了分布 $\mu$ 的一个无穷小扰动。

形式化地，设 $\mu_t$ 是一条在Wasserstein空间中的光滑路径，$\mu_0 = \mu$。由连续性方程，存在向量场 $v_0$ 使得：

$$\frac{\partial \mu_t}{\partial t}\bigg|_{t=0} + \nabla \cdot (\mu\,v_0) = 0$$

向量场 $v_0$ 就是 $\mu$ 处沿路径 $\mu_t$ 方向的"切向量"。

### 黎曼度量

Wasserstein空间上的黎曼度量由切向量的 $L^2$ 内积定义：对两个切向量 $v_1, v_2$，

$$\langle v_1, v_2 \rangle_\mu = \int \langle v_1(x), v_2(x) \rangle\,d\mu(x)$$

这个内积诱导的范数 $\|v\|_\mu^2 = \int \|v(x)\|^2 d\mu(x)$ 恰好是传输代价——Wasserstein距离的"微分"正是这个范数的积分。

### Otto calculus

Otto (2001) 发展了一套在Wasserstein空间上的微分计算框架，称为**Otto calculus**。其核心思想是将概率分布空间视为一个无限维黎曼流形，传统的黎曼几何操作（梯度、Hessian、测地线等）都有对应的Wasserstein版本。

**Wasserstein梯度**：在Wasserstein空间中，泛函 $\mathcal{F}(\mu)$ 的梯度为：

$$\text{grad}_W \mathcal{F}(\mu) = \nabla \frac{\delta \mathcal{F}}{\delta \mu}$$

其中 $\frac{\delta \mathcal{F}}{\delta \mu}$ 是 $\mathcal{F}$ 的一阶变分（函数导数）。

### 与Langevin动力学的联系

Otto calculus的一个重要应用是理解Langevin动力学。考虑KL散度 $\text{KL}(\mu \| p)$ 作为Wasserstein空间上的泛函：

$$\mathcal{F}(\mu) = \text{KL}(\mu \| p) = \int \mu \log\frac{\mu}{p}$$

其Wasserstein梯度为：

$$\text{grad}_W \text{KL}(\mu \| p) = \nabla\log\frac{\mu}{p} = \nabla\log\mu - \nabla\log p$$

Wasserstein梯度流为：

$$\frac{\partial \mu_t}{\partial t} = \nabla \cdot (\mu_t\,\text{grad}_W \text{KL}(\mu_t \| p)) = \nabla \cdot (\mu_t\,\nabla\log\mu_t) - \nabla \cdot (\mu_t\,\nabla\log p)$$

$$= \Delta\mu_t - \nabla \cdot (\mu_t\,\nabla\log p)$$

这正是Fokker-Planck方程——Langevin SDE $\frac{dx}{dt} = \nabla\log p(x) + \sqrt{2}\,dw$ 的概率密度演化方程。

**含义**：Langevin动力学是KL散度在Wasserstein空间上的梯度流 + 噪声。

- **梯度流部分**（$\nabla \cdot (\mu_t\,\nabla\log p)$）：驱动分布向 $p$ 移动
- **噪声部分**（$\Delta\mu_t$）：防止分布坍缩到众数

这个视角深化了第4-5章的讨论——Langevin动力学不仅是"得分驱动采样"，更是Wasserstein空间中的"梯度下降 + 正则化"。

### 与Flow Matching的联系

从Otto calculus的视角看，Flow Matching训练的是一个Wasserstein空间中的传输映射——从基础分布 $q$ 到数据分布 $p$ 的测地线（OT映射）或近似测地线。

- **OT-CFM**：直接逼近Wasserstein测地线——OT映射是最短路径
- **Rectified Flow + Reflow**：迭代逼近Wasserstein测地线——每次Reflow减少与测地线的距离
- **扩散模型**：走的是一条非测地线路径——由SDE结构决定，路径弯曲

这三者都可以在Wasserstein空间的黎曼框架下统一理解——区别仅在于所走的路径是否是测地线。

## 参考文献

- Ambrosio, L., Gigli, N., & Savaré, G. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures*. Birkhäuser.
- Otto, F. (2001). The geometry of dissipative evolution equations: the porous medium equation. *Communications in Partial Differential Equations*, 26(1-2), 101-174.
- Villani, C. (2008). *Optimal Transport: Old and New*. Springer.

**来源**：Villani (2008) Ch. 7-8; Ambrosio et al. (2008) Ch. 8; Otto (2001); Santambrogio (2015) Ch. 7
