# 附录16C 波前集理论与有限角CT的数学框架

> 定位：16.2.3节介绍了波前集的直觉和结论，本附录提供波前集的严格定义和有限角CT的微局部性质。

## C.1 波前集的严格定义

**局部傅里叶变换**：设$u \in \mathcal{D}'(\mathbb{R}^n)$是一个分布，$\phi \in C_c^\infty(\mathbb{R}^n)$是一个在$x_0$附近为1的截断函数。$u$在$x_0$附近的局部傅里叶变换定义为：

$$\widehat{\phi u}(\xi) = \int u(x) \phi(x) e^{-i x \cdot \xi} dx$$

**定义（波前集）**：点$(x_0, \xi_0) \in \mathbb{R}^n \times (\mathbb{R}^n \setminus \{0\})$**不属于**$u$的波前集$WF(u)$，如果存在$x_0$的邻域$U$和$\xi_0$的锥形邻域$V$，使得对任意在$U$中为1的截断函数$\phi$，$\widehat{\phi u}(\xi)$在$V$中快速衰减（即$|\widehat{\phi u}(\xi)| \leq C_N (1+|\xi|)^{-N}$对任意$N$成立）。

直觉：$(x_0, \xi_0) \notin WF(u)$意味着$u$在$x_0$处沿方向$\xi_0$是"光滑的"——局部傅里叶变换在$\xi_0$方向快速衰减。反之，$(x_0, \xi_0) \in WF(u)$意味着$u$在$x_0$处沿方向$\xi_0$有"奇异性"。

**例子**：
- 设$u = \delta_{(a,b)}$（点源），则$WF(u) = \{(a,b, \xi) : \xi \neq 0\}$——点源在所有方向都是奇异的
- 设$u = H(x_1)$（Heaviside函数，$x_1 = 0$处的阶跃），则$WF(u) = \{(0, x_2, \xi_1, 0) : \xi_1 \neq 0, x_2 \in \mathbb{R}\}$——阶跃边缘只在法线方向（$x_1$方向）奇异，在切线方向（$x_2$方向）光滑

## C.2 Radon变换的微局部性质

**定理**（Quinto 1993）：设$u \in \mathcal{E}'(\mathbb{R}^2)$（紧支撑分布），则Radon变换$\mathcal{R}$保持波前集的如下关系：

$$(x_0, \xi_0) \in WF(u) \iff \text{存在}\theta_0\text{使得}(\theta_0, s_0, \xi_0 \cdot \theta_0^\perp) \in WF(\mathcal{R}u)$$

其中$s_0 = x_0 \cdot \theta_0$，$\theta_0^\perp$是垂直于$\theta_0$的单位向量。

关键含义：Radon变换将图像空间的波前集"投影"到sinogram空间的波前集——每个$(x_0, \xi_0)$对对应sinogram中的一个特定位置和方向。

## C.3 有限角CT的可见性定理

**定理**（Greenleaf & Uhlmann 1989；Frikel & Quinto 2013）：设测量角度范围为$\Theta \subset [0, \pi)$，则从有限角度数据$\mathcal{R}_\Theta u$可以稳定恢复的波前集为：

$$WF_{\text{visible}}(u) = \{(x, \xi) \in WF(u) : \xi/|\xi| = (\cos\theta, \sin\theta), \theta \in \Theta\}$$

即：**只有法线方向落在测量角度范围内的边缘才可被稳定恢复**。

**推论**：有限角CT的不可见波前集为：

$$WF_{\text{invisible}}(u) = \{(x, \xi) \in WF(u) : \xi/|\xi| \notin \Theta\}$$

不可见波前集对应的边缘信息无法被任何线性方法或凸正则化方法稳定恢复。

## C.4 剪切波与不可见波前集的恢复

**剪切波**（shearlet）系统$\{\psi_{j,k,m}\}$是$\mathbb{R}^2$上的方向敏感多尺度表示，具有以下最优性质：

- 对于分片光滑函数$u$，剪切波系数的衰减率精确刻画了$WF(u)$
- 具体地，大系数对应$WF(u)$中的元素，系数的方向和尺度与奇异性位置和方向对应

Bubba et al. (2019)的核心思路：
1. 将剪切波系数分为"可见"和"不可见"两组（对应$WF_{\text{visible}}$和$WF_{\text{invisible}}$）
2. 可见系数可从有限角度数据稳定恢复
3. 不可见系数通过学习（从训练数据中建立的统计模型）估计
4. 合并两组系数得到完整重建

这是数据驱动方法突破传统数学限制的典型例子——数学分析告诉我们"线性方法无法恢复不可见边缘"，但学习型方法可以通过数据中建立的先验知识来"猜测"不可见边缘。

**来源**：Greenleaf & Uhlmann (1989)；Quinto (1993)；Frikel & Quinto (2013)；Bubba, Kutyniok, Lassas, Maerz, Samek, Siltanen & Srinivasan (2019) Inverse Problems；Natterer (1986) The Mathematics of Computerized Tomography
