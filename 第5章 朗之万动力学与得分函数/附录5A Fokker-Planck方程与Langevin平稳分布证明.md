# 附录5A Fokker-Planck 方程与 Langevin 平稳分布证明

> 定位：为 5.1 节提供 Langevin SDE 平稳分布的严格证明。Fokker-Planck 方程是连接 SDE 与概率密度演化的核心工具，推导涉及随机分析基础，放附录以保持主线叙事流畅。

**你会在本附录看到**：为什么一个用得分函数当漂移力的随机过程，跑久了概率分布恰好停在目标后验上。证明分两块——先推 Fokker-Planck 方程（密度怎么随时间变），再代入验证后验是平稳分布，最后给强对数凹下的指数收敛速率。

---

## Fokker-Planck 方程的推导

### 从 Itô 公式出发

考虑 Langevin SDE：

$$dX_t = s(X_t)\,dt + \sqrt{2}\,dW_t, \quad X_0 = x_0$$

其中 $s(x)=\nabla\log p(x)$ 是得分函数，$W_t$ 是标准布朗运动。设 $\rho_t(x)$ 是 $X_t$ 的概率密度。目标是推导 $\rho_t$ 满足的偏微分方程——Fokker-Planck 方程。

### Itô 公式的应用

对任意测试函数 $\phi\in C_c^\infty(\mathbb{R}^n)$（无穷可微、紧支撑），由 Itô 公式：

$$d\phi(X_t) = \nabla\phi(X_t)^T\,dX_t + \frac{1}{2}\text{tr}\left[\nabla^2\phi(X_t)\,d\langle X\rangle_t\right]$$

代入 Langevin SDE：

$$d\phi(X_t) = \nabla\phi(X_t)^T s(X_t)\,dt + \sqrt{2}\,\nabla\phi(X_t)^T dW_t + \frac{1}{2}\cdot 2\,\Delta\phi(X_t)\,dt$$

其中 $d\langle X\rangle_t = 2\,dt$（扩散系数 $\sqrt{2}$ 的二次变差为 $2\,dt$）。整理得：

$$d\phi(X_t) = \left[\nabla\phi(X_t)^T s(X_t) + \Delta\phi(X_t)\right]dt + \sqrt{2}\,\nabla\phi(X_t)^T dW_t$$

### 取期望

两边取期望，布朗运动项期望为零（Itô 积分的鞅性质）：

$$\frac{d}{dt}\mathbb{E}[\phi(X_t)] = \mathbb{E}[\nabla\phi(X_t) \cdot s(X_t)] + \mathbb{E}[\Delta\phi(X_t)]$$

用密度 $\rho_t$ 写期望：

$$\frac{d}{dt}\int\phi(x)\,\rho_t(x)\,dx = \int\nabla\phi(x) \cdot s(x)\,\rho_t(x)\,dx + \int\Delta\phi(x)\,\rho_t(x)\,dx$$

### 分部积分

对右边两项分别分部积分，把导数从 $\phi$ 转移到 $\rho_t$：

**第一项**（散度定理）：

$$\int\nabla\phi(x) \cdot s(x)\,\rho_t(x)\,dx = -\int\phi(x)\,\nabla\cdot[s(x)\,\rho_t(x)]\,dx$$

**第二项**（两次分部积分）：

$$\int\Delta\phi(x)\,\rho_t(x)\,dx = \int\phi(x)\,\Delta\rho_t(x)\,dx$$

### 得到 Fokker-Planck 方程

代入整理：

$$\int\phi(x)\,\frac{\partial\rho_t}{\partial t}\,dx = \int\phi(x)\left\{-\nabla\cdot[s(x)\,\rho_t] + \Delta\rho_t\right\}dx$$

由于 $\phi$ 任意，被积函数必相等：

$$\boxed{\frac{\partial \rho_t}{\partial t} = -\nabla\cdot[s(x)\,\rho_t] + \Delta\rho_t}$$

这就是 **Fokker-Planck 方程**（前向 Kolmogorov 方程），描述 Langevin SDE 驱动的概率密度随时间演化。

---

## 验证 $p(x)$ 是平稳分布

### 平稳分布的定义

平稳分布 $\rho^*$ 满足 $\partial\rho^*/\partial t = 0$：

$$0 = -\nabla\cdot[s(x)\,\rho^*] + \Delta\rho^*$$

### 代入 $\rho^* = p(x)$

$$-\nabla\cdot[s(x)\,p(x)] + \Delta p(x)$$

关键观察：$s(x)\,p(x)=\nabla\log p(x)\cdot p(x)=\nabla p(x)$，**于是**：

$$-\nabla\cdot[\nabla p(x)] + \Delta p(x) = -\Delta p(x) + \Delta p(x) = 0 \quad \checkmark$$

**$p(x)$ 确实是 Langevin SDE 的平稳分布。**

### 物理解释的严格化

Fokker-Planck 方程可写成"概率流"形式：

$$\frac{\partial\rho_t}{\partial t} = -\nabla\cdot J(x, t)$$

其中概率流 $J = s(x)\,\rho_t(x) - \nabla\rho_t(x) = \rho_t(x)\left[s(x) - \frac{\nabla\rho_t}{\rho_t}\right] = \rho_t(x)\left[\nabla\log p(x) - \nabla\log\rho_t(x)\right]$。

平稳态 $\rho_t=p$ 时：

$$J(x) = p(x)[\nabla\log p(x) - \nabla\log p(x)] = 0$$

平稳态**概率流为零**——粒子虽在动，但净流量为零，形成动态平衡。这正是细致平衡（detailed balance）：每点流入量等于流出量。

---

## 指数收敛的证明（强对数凹情形）

### Bakry-Émery 理论

当势能 $U(x)=-\log p(x)$ 满足 Bakry-Émery 曲率条件：

$$\text{Hess}\,U(x) \succeq m\,I, \quad m>0$$

即 $U$ 是 $m$-强凸。

### Wasserstein-2 距离的指数衰减

**定理**：在 $m$-强对数凹条件下，Langevin 扩散在 Wasserstein-2 距离下指数收敛：

$$W_2(\rho_t, p) \leq e^{-mt}\,W_2(\rho_0, p)$$

**证明思路**：

1. 用 Wasserstein-2 距离的对偶表示；
2. 耦合方法：设 $(X_t,Y_t)$ 是两个 Langevin 过程，分别从 $\rho_0$ 和 $p$ 出发；
3. 耦合 SDE：$dX_t=s(X_t)\,dt+\sqrt{2}\,dW_t$，$dY_t=s(Y_t)\,dt+\sqrt{2}\,dW_t$（**共享同一布朗运动**）；
4. 令 $\Delta_t=X_t-Y_t$，则 $d\Delta_t=[s(X_t)-s(Y_t)]\,dt$；
5. 由强凸性：$\langle s(X_t)-s(Y_t), X_t-Y_t\rangle \le -m\|X_t-Y_t\|^2$；
6. 故 $\frac{d}{dt}\mathbb{E}[\|\Delta_t\|^2] \le -2m\,\mathbb{E}[\|\Delta_t\|^2]$；
7. 解得 $\mathbb{E}[\|\Delta_t\|^2] \le e^{-2mt}\,\mathbb{E}[\|\Delta_0\|^2]$；
8. 取下确界得 $W_2^2(\rho_t,p)\le e^{-2mt}\,W_2^2(\rho_0,p)$，即 $W_2(\rho_t,p)\le e^{-mt}\,W_2(\rho_0,p)$。

### 收敛速率的物理含义

速率 $2m$ 由势能强凸性决定：

- $m$ 大（碗很陡）→ 收敛快——粒子被强力拉回；
- $m$ 小（碗很平）→ 收敛慢——粒子自由游荡；
- $m=0$（非凸）→ 可能不指数收敛——粒子可能陷局部极值。

这给实践里收敛性评估提供理论基准：能估出后验强凸参数 $m$，就能预测 Langevin 采样收敛时间。

**来源**：Risken (1996); Bakry, Gentil & Ledoux (2014); Villani (2009)
