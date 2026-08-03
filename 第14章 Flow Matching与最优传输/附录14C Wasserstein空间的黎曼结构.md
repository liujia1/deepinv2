# 附录14C Wasserstein空间的黎曼结构

> **定位**：对理解 OT 与生成模型的关系有深层价值，但涉及黎曼几何，超出本书主要读者群体的背景。本附录提供概念性介绍，严格数学处理参见 Villani (2008) 和 Ambrosio et al. (2008)。想"看懂主线"的读者可以只读加粗的三句话。

## 概率分布空间上的黎曼结构

### 动机（因果：我们要给分布空间一个"几何"）

14.1 节定义了 Wasserstein 距离 $W_2$，它给概率分布空间 $\mathcal{P}_2(\mathbb{R}^d)$ 赋予了度量结构。一个深刻的问题：这个度量空间有没有更丰富的几何——比如流形上的"切空间""测地线"？答案是肯定的，这就是 **Wasserstein 空间** 的黎曼结构。

### Wasserstein 空间的切空间

在黎曼几何里，切空间描述流形上某点的"无穷小邻域"。在 Wasserstein 空间里，分布 $\mu$ 处的"切向量"是一个向量场 $v:\mathbb{R}^d\to\mathbb{R}^d$，描述 $\mu$ 的一个无穷小扰动。

形式化地，设 $\mu_t$ 是 Wasserstein 空间里一条光滑路径，$\mu_0=\mu$。由连续性方程，存在向量场 $v_0$ 使

$$\frac{\partial \mu_t}{\partial t}\bigg|_{t=0} + \nabla \cdot (\mu\,v_0) = 0$$

这个 $v_0$ 就是 $\mu$ 处沿路径方向的"切向量"。**白话：在分布空间里"挪一小步"，等价于在整个空间里定义一个向量场。**

### 黎曼度量

Wasserstein 空间上的黎曼度量由切向量的 $L^2$ 内积定义：对切向量 $v_1,v_2$，

$$\langle v_1, v_2 \rangle_\mu = \int \langle v_1(x), v_2(x) \rangle\,d\mu(x)$$

诱导的范数 $\|v\|_\mu^2=\int\|v(x)\|^2d\mu(x)$ **恰好就是传输代价**——Wasserstein 距离的"微分"正是这个范数的积分。这把"搬运距离"和"几何长度"统一了。

### Otto calculus（白话：在分布空间里做微积分）

Otto (2001) 发展了一套在 Wasserstein 空间上的微分计算，叫 **Otto calculus**——把概率分布空间看成无限维黎曼流形，梯度、Hessian、测地线都有对应版本。Wasserstein 梯度：

$$\text{grad}_W \mathcal{F}(\mu) = \nabla \frac{\delta \mathcal{F}}{\delta \mu}$$

其中 $\frac{\delta\mathcal{F}}{\delta\mu}$ 是泛函 $\mathcal{F}$ 的一阶变分（函数导数）。

### 与 Langevin 动力学的联系（因果：把前面章节串起来）

把 KL 散度 $\mathcal{F}(\mu)=\text{KL}(\mu\|p)=\int\mu\log\frac{\mu}{p}$ 看作 Wasserstein 空间上的泛函，其 Wasserstein 梯度为

$$\text{grad}_W \text{KL}(\mu\|p) = \nabla\log\frac{\mu}{p} = \nabla\log\mu - \nabla\log p$$

Wasserstein 梯度流：

$$\frac{\partial \mu_t}{\partial t} = \nabla\cdot(\mu_t\,\text{grad}_W\text{KL}) = \nabla\cdot(\mu_t\nabla\log\mu_t) - \nabla\cdot(\mu_t\nabla\log p) = \Delta\mu_t - \nabla\cdot(\mu_t\nabla\log p)$$

这正是 Fokker-Planck 方程——Langevin SDE $\frac{dx}{dt}=\nabla\log p(x)+\sqrt{2}\,dw$ 的概率密度演化。**含义：Langevin 动力学 = KL 散度在 Wasserstein 空间上的梯度流 + 噪声。** 梯度流部分把分布推向 $p$，噪声部分防止坍缩到众数。这深化了第4-5章的讨论：Langevin 不仅是"得分驱动采样"，更是 Wasserstein 空间里的"梯度下降 + 正则化"。

### 与 Flow Matching 的联系（因果收束）

从 Otto calculus 看，Flow Matching 训练的是 Wasserstein 空间里的一条传输映射——从基础分布 $q$ 到数据分布 $p$ 的测地线（OT 映射）或近似测地线：
- **OT-CFM**：直接逼近 Wasserstein 测地线（最短路径）；
- **Rectified Flow + Reflow**：迭代逼近测地线（每轮 Reflow 减少与测地线距离）；
- **扩散模型**：走非测地线路径（由 SDE 结构决定，弯曲）。

三者都能在 Wasserstein 空间的黎曼框架下统一理解——区别仅在于所走路径是不是测地线。

## 参考文献

- Ambrosio, L., Gigli, N., & Savaré, G. (2008). *Gradient Flows in Metric Spaces and in the Space of Probability Measures*. Birkhäuser.
- Otto, F. (2001). The geometry of dissipative evolution equations: the porous medium equation. *Communications in Partial Differential Equations*, 26(1-2), 101-174.
- Villani, C. (2008). *Optimal Transport: Old and New*. Springer.

**来源**：Villani (2008) Ch. 7-8; Ambrosio et al. (2008) Ch. 8; Otto (2001); Santambrogio (2015) Ch. 7
