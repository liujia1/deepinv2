# 附录5A Fokker-Planck方程与Langevin扩散的平稳分布

> 定位：为5.1节提供Langevin SDE平稳分布的严格证明。Fokker-Planck方程是连接SDE与概率密度演化的核心工具，其推导涉及随机分析的基础知识，放在附录以保持主线叙事的流畅。

## Fokker-Planck方程的推导

### 从Itô公式出发

考虑Langevin SDE：

$$dX_t = s(X_t)\,dt + \sqrt{2}\,dW_t, \quad X_0 = x_0$$

其中 $s(x) = \nabla\log p(x)$ 是得分函数，$W_t$ 是标准布朗运动。

设 $\rho_t(x)$ 为 $X_t$ 的概率密度函数。我们的目标是推导 $\rho_t$ 满足的偏微分方程——Fokker-Planck方程。

### Itô公式的应用

对任意测试函数 $\phi \in C_c^\infty(\mathbb{R}^n)$（无穷可微、紧支撑），由Itô公式：

$$d\phi(X_t) = \nabla\phi(X_t)^T\,dX_t + \frac{1}{2}\text{tr}\left[\nabla^2\phi(X_t)\,d\langle X\rangle_t\right]$$

代入Langevin SDE：

$$d\phi(X_t) = \nabla\phi(X_t)^T s(X_t)\,dt + \sqrt{2}\,\nabla\phi(X_t)^T dW_t + \frac{1}{2}\cdot 2\,\Delta\phi(X_t)\,dt$$

其中 $d\langle X\rangle_t = 2\,dt$（扩散系数 $\sqrt{2}$ 的二次变差为 $2\,dt$）。

整理得：

$$d\phi(X_t) = \left[\nabla\phi(X_t)^T s(X_t) + \Delta\phi(X_t)\right]dt + \sqrt{2}\,\nabla\phi(X_t)^T dW_t$$

### 取期望

两边取期望，布朗运动项的期望为零（Itô积分的鞅性质）：

$$\frac{d}{dt}\mathbb{E}[\phi(X_t)] = \mathbb{E}[\nabla\phi(X_t) \cdot s(X_t)] + \mathbb{E}[\Delta\phi(X_t)]$$

用密度函数 $\rho_t$ 表达期望：

$$\frac{d}{dt}\int\phi(x)\,\rho_t(x)\,dx = \int\nabla\phi(x) \cdot s(x)\,\rho_t(x)\,dx + \int\Delta\phi(x)\,\rho_t(x)\,dx$$

### 分部积分

对右边两项分别进行分部积分，将导数从 $\phi$ 转移到 $\rho_t$：

**第一项**（散度定理）：

$$\int\nabla\phi(x) \cdot s(x)\,\rho_t(x)\,dx = -\int\phi(x)\,\nabla\cdot[s(x)\,\rho_t(x)]\,dx$$

**第二项**（两次分部积分）：

$$\int\Delta\phi(x)\,\rho_t(x)\,dx = \int\phi(x)\,\Delta\rho_t(x)\,dx$$

### 得到Fokker-Planck方程

代入整理：

$$\int\phi(x)\,\frac{\partial\rho_t}{\partial t}\,dx = \int\phi(x)\left\{-\nabla\cdot[s(x)\,\rho_t] + \Delta\rho_t\right\}dx$$

由于 $\phi$ 是任意的，被积函数必须相等：

$$\boxed{\frac{\partial \rho_t}{\partial t} = -\nabla\cdot[s(x)\,\rho_t] + \Delta\rho_t}$$

这就是**Fokker-Planck方程**（也称前向Kolmogorov方程），描述了Langevin SDE驱动的概率密度的时间演化。

## 验证 $p(x)$ 是平稳分布

### 平稳分布的定义

平稳分布 $\rho^*$ 满足 $\partial\rho^*/\partial t = 0$，即：

$$0 = -\nabla\cdot[s(x)\,\rho^*] + \Delta\rho^*$$

### 代入 $\rho^* = p(x)$

将 $\rho^* = p(x)$ 代入Fokker-Planck方程的右边：

$$-\nabla\cdot[s(x)\,p(x)] + \Delta p(x)$$

关键观察：$s(x)\,p(x) = \nabla\log p(x) \cdot p(x) = \nabla p(x)$，因此：

$$-\nabla\cdot[\nabla p(x)] + \Delta p(x) = -\Delta p(x) + \Delta p(x) = 0 \quad \checkmark$$

**$p(x)$ 确实是Langevin SDE的平稳分布。**

### 物理解释的严格化

Fokker-Planck方程可以重写为"概率流"的形式：

$$\frac{\partial\rho_t}{\partial t} = -\nabla\cdot J(x, t)$$

其中概率流 $J$ 定义为：

$$J(x, t) = s(x)\,\rho_t(x) - \nabla\rho_t(x) = \rho_t(x)\left[s(x) - \frac{\nabla\rho_t}{\rho_t}\right] = \rho_t(x)\left[\nabla\log p(x) - \nabla\log\rho_t(x)\right]$$

在平稳态 $\rho_t = p$ 时：

$$J(x) = p(x)[\nabla\log p(x) - \nabla\log p(x)] = 0$$

平稳态的**概率流为零**——粒子虽然在运动，但净流量为零，形成动态平衡。这就是细致平衡（detailed balance）的含义：每一点的流入量等于流出量。

## 指数收敛的证明（强对数凹情形）

### Bakry-Émery理论

当势能 $U(x) = -\log p(x)$ 满足Bakry-Émery曲率条件：

$$\text{Hess}\,U(x) \succeq m\,I, \quad m > 0$$

即 $U$ 是 $m$-强凸的。

### Wasserstein-2距离的指数衰减

**定理**：在 $m$-强对数凹条件下，Langevin扩散在Wasserstein-2距离下指数收敛：

$$W_2(\rho_t, p) \leq e^{-mt}\,W_2(\rho_0, p)$$

**证明思路**：
1. 利用Wasserstein-2距离的对偶表示
2. 通过耦合方法：设 $(X_t, Y_t)$ 是两个Langevin过程，分别从 $\rho_0$ 和 $p$ 出发
3. 耦合SDE：$dX_t = s(X_t)\,dt + \sqrt{2}\,dW_t$，$dY_t = s(Y_t)\,dt + \sqrt{2}\,dW_t$（**共享同一布朗运动**）
4. 令 $\Delta_t = X_t - Y_t$，则 $d\Delta_t = [s(X_t) - s(Y_t)]\,dt$
5. 由强凸性：$\langle s(X_t) - s(Y_t), X_t - Y_t\rangle \leq -m\|X_t - Y_t\|^2$
6. 因此 $\frac{d}{dt}\mathbb{E}[\|\Delta_t\|^2] \leq -2m\,\mathbb{E}[\|\Delta_t\|^2]$
7. 解得 $\mathbb{E}[\|\Delta_t\|^2] \leq e^{-2mt}\,\mathbb{E}[\|\Delta_0\|^2]$
8. 取下确界（inf over couplings）得 $W_2^2(\rho_t, p) \leq e^{-2mt}\,W_2^2(\rho_0, p)$，即 $W_2(\rho_t, p) \leq e^{-mt}\,W_2(\rho_0, p)$

### 收敛速率的物理含义

收敛速率 $2m$ 由势能的强凸性决定：
- $m$ 大（碗很陡）→ 收敛快——粒子被强力拉回
- $m$ 小（碗很平）→ 收敛慢——粒子自由游荡
- $m = 0$（非凸）→ 可能不指数收敛——粒子可能陷入局部极值

这个结果为实践中的收敛性评估提供了理论基准：如果我们能估计后验的强凸参数 $m$，就能预测Langevin采样的收敛时间。

**来源**：Risken (1996); Bakry, Gentil & Ledoux (2014); Villani (2009)
