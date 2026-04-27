# 附录7A Anderson逆向时间SDE定理的证明概要

> 定位：为7.3节提供Anderson (1982) 逆向时间SDE定理的证明概要。完整证明需要随机分析的工具（Kunita-Watanabe分解等），这里给出核心思路和关键步骤。

## 定理陈述

**Anderson (1982) 逆向时间SDE定理**：设正向SDE

$$dx = f(x,t)\,dt + g(t)\,dw, \quad t \in [0, T]$$

其中 $f: \mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ 是漂移系数，$g: [0,T] \to \mathbb{R}_+$ 是扩散系数，$w$ 是标准布朗运动。则逆向时间过程 $y(s) = x(T-s)$ 满足

$$dy = \left[f(y, T-s) - g(T-s)^2\,\nabla_y\log p_{T-s}(y)\right]\,ds + g(T-s)\,d\bar{w}$$

其中 $p_t(y)$ 是 $x(t)$ 在时刻 $t$ 的边际密度，$\bar{w}$ 是另一标准布朗运动。

## 证明思路

### 第一步：建立逆向时间的密度演化方程

令 $q_s(y) = p_{T-s}(y)$——逆向时间的边际密度。由正向SDE的Fokker-Planck方程：

$$\frac{\partial p_t}{\partial t} = -\nabla\cdot[f(x,t)\,p_t] + \frac{1}{2}g(t)^2\,\Delta p_t$$

对 $q_s(y) = p_{T-s}(y)$ 求导：

$$\frac{\partial q_s}{\partial s} = -\frac{\partial p_{T-s}}{\partial t} = \nabla\cdot[f(y,T-s)\,q_s] - \frac{1}{2}g(T-s)^2\,\Delta q_s$$

### 第二步：将扩散项重组

将 $\Delta q_s$ 写为 $\nabla\cdot[\nabla q_s]$，并利用 $\nabla q_s = q_s \nabla\log q_s$：

$$\frac{\partial q_s}{\partial s} = \nabla\cdot\left[f(y,T-s)\,q_s - \frac{1}{2}g(T-s)^2\,q_s\,\nabla\log q_s\right]$$

$$= -\nabla\cdot\left[\left(-f(y,T-s) + \frac{1}{2}g(T-s)^2\,\nabla\log q_s\right)q_s\right]$$

### 第三步：识别SDE的漂移项

密度演化方程 $\frac{\partial q_s}{\partial s} = -\nabla\cdot[b(y,s)\,q_s] + \frac{1}{2}\nabla\cdot[\sigma^2(s)\nabla q_s]$ 对应SDE $dy = b(y,s)\,ds + \sigma(s)\,d\bar{w}$。

将第二步的结果与此对照，识别：

$$b(y,s) = f(y,T-s) - g(T-s)^2\,\nabla\log q_s(y) = f(y,T-s) - g(T-s)^2\,\nabla\log p_{T-s}(y)$$

$$\sigma(s) = g(T-s)$$

因此逆向SDE为：

$$dy = \left[f(y,T-s) - g(T-s)^2\,\nabla\log p_{T-s}(y)\right]\,ds + g(T-s)\,d\bar{w}$$

### 第四步：变回原始时间变量

令 $t = T-s$，$x(t) = y(T-t) = x(t)$，$d\bar{w}$ 为逆向布朗运动，得：

$$dx = \left[f(x,t) - g(t)^2\,\nabla\log p_t(x)\right]\,dt + g(t)\,d\bar{w} \quad \checkmark$$

## 特殊情况：Langevin SDE

正向Langevin SDE：$dx = \nabla\log p(x)\,dt + \sqrt{2}\,dW$，$f = \nabla\log p$，$g = \sqrt{2}$。

逆向SDE：

$$dx = \left[\nabla\log p(x) - 2\nabla\log p(x)\right]\,dt + \sqrt{2}\,d\bar{w} = -\nabla\log p(x)\,dt + \sqrt{2}\,d\bar{w}$$

逆向Langevin的漂移项反号——从"爬坡"（推向高概率区域）变为"下坡"（推向低概率区域）。但由于时间方向也反转了，逆向过程的效果是：粒子从初始分布出发，经过逆向Langevin演化后，仍然收敛到 $p(x)$。

## 得分修正项的直觉

逆向SDE中的得分修正项 $-g(t)^2\nabla\log p_t(x)$ 可以这样理解：

- 正向SDE将概率分布"抹平"——信息逐步丢失
- 逆向过程需要"恢复"信息——得分函数 $\nabla\log p_t(x)$ 指向概率增长的方向
- $-g(t)^2$ 的系数确保修正的强度与正向噪声的强度匹配——噪声越强，修正越大

**来源**：Anderson (1982); Winkler (2021); Risken (1996); Song et al. (2021)
