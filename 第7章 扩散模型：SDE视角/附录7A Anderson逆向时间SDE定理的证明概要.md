# 附录7A Anderson逆向时间SDE定理的证明概要

> 定位：为7.3节提供Anderson (1982) 逆向时间SDE定理的证明概要。完整证明需要随机分析的工具（Kunita-Watanabe分解等），这里给出核心思路和关键步骤。

## 先说人话：这个定理在证明什么？

7.3节我们直接用了 Anderson 定理："正向 SDE 倒着走，还是一个 SDE，而且漂移项只要在原漂移上**减掉一个得分修正** $-g^2\nabla\log p_t$ 就行。" 你可能会问：凭什么？这附录就是把"凭什么"讲清楚。核心直觉只有一句：**概率密度随时间怎么变（Fokker-Planck 方程）和"粒子按什么方程走（SDE）"是一一对应的**——所以我们只要把正向的密度演化"反着时间"重写一遍，再翻译回 SDE，就自动得到了逆向方程。

## 定理陈述

**Anderson (1982) 逆向时间SDE定理**：设正向SDE

$$dx = f(x,t)\,dt + g(t)\,dw, \quad t \in [0, T]$$

其中 $f: \mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ 是漂移系数，$g: [0,T] \to \mathbb{R}_+$ 是扩散系数，$w$ 是标准布朗运动。则逆向时间过程 $y(s) = x(T-s)$ 满足

$$dy = \left[f(y, T-s) - g(T-s)^2\,\nabla_y\log p_{T-s}(y)\right]\,ds + g(T-s)\,d\bar{w}$$

其中 $p_t(y)$ 是 $x(t)$ 在时刻 $t$ 的边际密度，$\bar{w}$ 是另一标准布朗运动。

## 证明思路（四步，密度→方程→识别→换回时间）

### 第一步：建立逆向时间的密度演化方程

令 $q_s(y) = p_{T-s}(y)$——逆向时间的边际密度。由正向SDE的Fokker-Planck方程（描述密度如何随时间变化）：

$$\frac{\partial p_t}{\partial t} = -\nabla\cdot[f(x,t)\,p_t] + \frac{1}{2}g(t)^2\,\Delta p_t$$

对 $q_s(y) = p_{T-s}(y)$ 求导（链式法则，时间反了所以多个负号）：

$$\frac{\partial q_s}{\partial s} = -\frac{\partial p_{T-s}}{\partial t} = \nabla\cdot[f(y,T-s)\,q_s] - \frac{1}{2}g(T-s)^2\,\Delta q_s$$

### 第二步：把扩散项重组

将 $\Delta q_s$ 写成 $\nabla\cdot[\nabla q_s]$，并利用 $\nabla q_s = q_s \nabla\log q_s$（得分就是对数密度的梯度）：

$$\frac{\partial q_s}{\partial s} = \nabla\cdot\left[f(y,T-s)\,q_s - \frac{1}{2}g(T-s)^2\,q_s\,\nabla\log q_s\right]$$

$$= -\nabla\cdot\left[\left(-f(y,T-s) + \frac{1}{2}g(T-s)^2\,\nabla\log q_s\right)q_s\right]$$

### 第三步：识别SDE的漂移项

密度演化方程 $\frac{\partial q_s}{\partial s} = -\nabla\cdot[b(y,s)\,q_s] + \frac{1}{2}\nabla\cdot[\sigma^2(s)\nabla q_s]$ 对应 SDE $dy = b(y,s)\,ds + \sigma(s)\,d\bar{w}$。把第二步结果对照，识别：

$$b(y,s) = f(y,T-s) - g(T-s)^2\,\nabla\log q_s(y) = f(y,T-s) - g(T-s)^2\,\nabla\log p_{T-s}(y)$$

$$\sigma(s) = g(T-s)$$

因此逆向SDE为：

$$dy = \left[f(y,T-s) - g(T-s)^2\,\nabla\log p_{T-s}(y)\right]\,ds + g(T-s)\,d\bar{w}$$

### 第四步：变回原始时间变量

令 $t = T-s$，$x(t) = y(T-t) = x(t)$，$d\bar{w}$ 为逆向布朗运动，得：

$$dx = \left[f(x,t) - g(t)^2\,\nabla\log p_t(x)\right]\,dt + g(t)\,d\bar{w} \quad \checkmark$$

证毕。注意关键：我们得到的漂移项正好是"原漂移 **减** 得分修正"——这就是为什么 7.3 节说"改符号、加得分"。

## 特殊情况：Langevin SDE（帮你核对直觉）

正向 Langevin SDE：$dx = \nabla\log p(x)\,dt + \sqrt{2}\,dW$，$f=\nabla\log p$，$g=\sqrt{2}$。

逆向 SDE：

$$dx = \left[\nabla\log p(x) - 2\nabla\log p(x)\right]\,dt + \sqrt{2}\,d\bar{w} = -\nabla\log p(x)\,dt + \sqrt{2}\,d\bar{w}$$

逆向 Langevin 的漂移项**反号**——从"爬坡"（推向高概率区）变"下坡"（推向低概率区）。但因为时间方向也反转了，逆向过程的效果仍是：粒子从初始分布出发，经逆向 Langevin 演化后收敛到 $p(x)$。这条小练习也印证了 7.3 节的画面：得分函数就是"指南针"，时间倒流时它照样把你领回家。

## 得分修正项的直觉

逆向 SDE 里的得分修正项 $-g(t)^2\nabla\log p_t(x)$ 这样理解：

- 正向 SDE 把概率分布"抹平"——信息逐步丢失；
- 逆向过程需"恢复"信息——得分函数 $\nabla\log p_t(x)$ 指向概率增长方向；
- $-g(t)^2$ 的系数确保修正强度与正向噪声强度匹配——噪声越强，修正越大。

**来源**：Anderson (1982); Winkler (2021); Risken (1996); Song et al. (2021)
