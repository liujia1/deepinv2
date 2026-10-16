# 附录5A Fokker-Planck 方程与 Langevin 平稳分布证明

> 定位：为 5.1 节提供 Langevin SDE 平稳分布的严格证明。Fokker-Planck 方程是连接 SDE 与概率密度演化的核心工具，推导涉及随机分析基础，放附录以保持主线叙事流畅。

**你会在本附录看到**：为什么一个用得分函数当漂移力的随机过程，跑久了概率分布恰好停在目标后验上。证明分两块——先推 Fokker-Planck 方程（密度怎么随时间变），再代入验证后验是平稳分布，最后给强对数凹下的指数收敛速率。

在踏进推导之前，先想想我们正在为什么拼出一幅统计图景。5.1 节断言 Langevin SDE 能以任意初始分布出发、最终收敛到目标后验，但这句断言不能靠虔诚，而要由一个偏微分方程来落地：我们得先知道"密度随时间如何流动"（Fokker-Planck 方程），才能检验"停在哪里不再流动"（平稳分布）。下面的每一行分部积分与鞅性质，都是在替这个"流动 → 静止"的图景钉上数学的骨架。

---

## Fokker-Planck 方程的推导

### 从 Itô 公式出发

考虑 Langevin SDE：

$$dX_t = s(X_t)\,dt + \sqrt{2}\,dW_t, \quad X_0 = x_0$$

其中 $s(x)=\nabla\log p(x)$ 是得分函数，$W_t$ 是标准布朗运动。设 $\rho_t(x)$ 是 $X_t$ 的概率密度。目标是推导 $\rho_t$ 满足的偏微分方程——Fokker-Planck 方程。

逐项把你现在看到的东西和物理图像对上：第一项 $s(X_t)\,dt$ 是漂移——沿着得分（密度增长方向）把粒子向前推的确定性力；第二项 $\sqrt{2}\,dW_t$ 是扩散——布朗运动带来的随机扰动。正是这两股力的角力，决定了密度 $\rho_t$ 会被揉成什么形状、朝哪个方向流。

### Itô 公式的应用

对任意测试函数 $\phi\in C_c^\infty(\mathbb{R}^n)$（无穷可微、紧支撑），由 Itô 公式：

$$d\phi(X_t) = \nabla\phi(X_t)^T\,dX_t + \frac{1}{2}\text{tr}\left[\nabla^2\phi(X_t)\,d\langle X\rangle_t\right]$$

代入 Langevin SDE：

$$d\phi(X_t) = \nabla\phi(X_t)^T s(X_t)\,dt + \sqrt{2}\,\nabla\phi(X_t)^T dW_t + \frac{1}{2}\cdot 2\,\Delta\phi(X_t)\,dt$$

其中 $d\langle X\rangle_t = 2\,dt$（扩散系数 $\sqrt{2}$ 的二次变差为 $2\,dt$）。整理得：

$$d\phi(X_t) = \left[\nabla\phi(X_t)^T s(X_t) + \Delta\phi(X_t)\right]dt + \sqrt{2}\,\nabla\phi(X_t)^T dW_t$$

这里的试探函数 $\phi$ 不过是我们的"探针"：它不关心某个具体粒子的轨迹，而关心"任意光滑观测在时间微元里的期望变化"——这样我们就能绕过难以直接处理的随机路径，转而写下一个关于密度的确定性方程。

### 取期望

两边取期望，布朗运动项期望为零（Itô 积分的鞅性质）：

$$\frac{d}{dt}\mathbb{E}[\phi(X_t)] = \mathbb{E}[\nabla\phi(X_t) \cdot s(X_t)] + \mathbb{E}[\Delta\phi(X_t)]$$

用密度 $\rho_t$ 写期望：

$$\frac{d}{dt}\int\phi(x)\,\rho_t(x)\,dx = \int\nabla\phi(x) \cdot s(x)\,\rho_t(x)\,dx + \int\Delta\phi(x)\,\rho_t(x)\,dx$$

取期望这一刀砍掉的是随机性：布朗项期望为零，意味着从"单个粒子的随机轨迹"跃迁到"群体的密度演化"时，我们只需关心漂移项与扩散项对密度的一阶与二阶贡献。

### 分部积分

对右边两项分别分部积分，把导数从 $\phi$ 转移到 $\rho_t$：

**第一项**（散度定理）：

$$\int\nabla\phi(x) \cdot s(x)\,\rho_t(x)\,dx = -\int\phi(x)\,\nabla\cdot[s(x)\,\rho_t(x)]\,dx$$

**第二项**（两次分部积分）：

$$\int\Delta\phi(x)\,\rho_t(x)\,dx = \int\phi(x)\,\Delta\rho_t(x)\,dx$$

关键的一步是把梯度从试探函数身上"搬到"密度身上：这相当于把流量的净变化（散度）与曲率的净贡献（拉普拉斯）从探针转移到密度函数本身，是我们能从"关于 $\phi$ 的恒等式"提炼出"关于 $\rho_t$ 的方程"的前提。

### 得到 Fokker-Planck 方程

代入整理：

$$\int\phi(x)\,\frac{\partial\rho_t}{\partial t}\,dx = \int\phi(x)\left\{-\nabla\cdot[s(x)\,\rho_t] + \Delta\rho_t\right\}dx$$

由于 $\phi$ 任意，被积函数必相等：

$$\boxed{\frac{\partial \rho_t}{\partial t} = -\nabla\cdot[s(x)\,\rho_t] + \Delta\rho_t}$$

这就是 **Fokker-Planck 方程**（前向 Kolmogorov 方程），描述 Langevin SDE 驱动的概率密度随时间演化。

$\phi$ 的任意性换来的这条等式，是把"许多粒子分散每个角落的概率质量"用一个场论式方程全局刻画的收尾：左侧是密度的时间变化率，右侧第一项是漂移把质量卷走的散度，第二项是扩散把质量抹平的拉普拉斯。谁大谁小，决定了此刻密度是正在被拉拢还是正在变平。

---

## 验证 $p(x)$ 是平稳分布

### 平稳分布的定义

平稳分布 $\rho^*$ 满足 $\partial\rho^*/\partial t = 0$：

$$0 = -\nabla\cdot[s(x)\,\rho^*] + \Delta\rho^*$$

平稳在这里的含义要读清楚："不再随时间变化"并不等于"所有粒子都停下来"，而是"流入某个区域的概率质量恰好等于流出的质量"——是一种动态下的收支平衡。

### 代入 $\rho^* = p(x)$

$$-\nabla\cdot[s(x)\,p(x)] + \Delta p(x)$$

关键观察：$s(x)\,p(x)=\nabla\log p(x)\cdot p(x)=\nabla p(x)$，**于是**：

$$-\nabla\cdot[\nabla p(x)] + \Delta p(x) = -\Delta p(x) + \Delta p(x) = 0 \quad \checkmark$$

**$p(x)$ 确实是 Langevin SDE 的平稳分布。**

这一行等式是整节推导的"报偿时刻"：当我们让漂移力 $s(x)$ 恰好等于得分函数 $\nabla\log p(x)$ 时，漂移项 $s\,p=\nabla p$ 的散度与扩散项成对相消。换句话说，得分函数不是被我们对概率的直觉"硬凑"成漂移力的，而是千古不变的统计平衡——只要漂移与得分成对，均衡就自动成立。

### 物理解释的严格化

Fokker-Planck 方程可写成"概率流"形式：

$$\frac{\partial\rho_t}{\partial t} = -\nabla\cdot J(x, t)$$

其中概率流 $J = s(x)\,\rho_t(x) - \nabla\rho_t(x) = \rho_t(x)\left[s(x) - \frac{\nabla\rho_t}{\rho_t}\right] = \rho_t(x)\left[\nabla\log p(x) - \nabla\log\rho_t(x)\right]$。

平稳态 $\rho_t=p$ 时：

$$J(x) = p(x)[\nabla\log p(x) - \nabla\log p(x)] = 0$$

平稳态**概率流为零**——粒子虽在动，但净流量为零，形成动态平衡。这正是细致平衡（detailed balance）：每点流入量等于流出量。

从更深层次看，"概率流为零"这句话道出了平稳态的双重意味：它既刻画了采样器在平稳态的统计行为（样本分布不再漂移），也刻画了智能在知识达到稳态时的模样——系统仍在不断尝试（粒子在动），但整体的信仰分布已经稳定（净流为零）。这恰恰呼应"后验采样相对点估计"的哲学：真正成熟的推断不是停止探索，而是在充分探索后保持信念结构的稳定。

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

证明的引擎是第 4–7 步：把"分布间距离"的问题转化为"同一布朗运动驱动下两个粒子相距多远"的问题。共享布朗运动让随机部分相消，只剩得分函数的差在推动 $\Delta_t$；而第 5 步的强凸性不等式 $\langle s(X_t)-s(Y_t), X_t-Y_t\rangle\le -m\|\Delta_t\|^2$ 正是"地势下凹、指向同一谷底"的代数化身——它保证两个粒子只会相互靠拢、共同奔向公共的谷底，靠拢速度由曲率 $m$ 决定。

### 收敛速率的物理含义

速率 $2m$ 由势能强凸性决定：

- $m$ 大（碗很陡）→ 收敛快——粒子被强力拉回；
- $m$ 小（碗很平）→ 收敛慢——粒子自由游荡；
- $m=0$（非凸）→ 可能不指数收敛——粒子可能陷局部极值。

这给实践里收敛性评估提供理论基准：能估出后验强凸参数 $m$，就能预测 Langevin 采样收敛时间。

这里的 $m$ 是可以被直觉把握的量：它是"谷底置信"的强度——$m$ 越大，意味着后验在它的峰附近隆起得越陡、越"有主见"，粒子一旦走偏就会被迅速拉回，收敛自然快；$m$ 越小，后验越是平坦到模棱两可，粒子东游西荡，收敛自然慢。这恰好呼应整部书的主题：一个结构清晰（强凸）的后验，比一个犹豫不决（平坦）的后验更容易达成收敛的共识。

**来源**：Risken (1996); Bakry, Gentil & Ledoux (2014); Villani (2009)