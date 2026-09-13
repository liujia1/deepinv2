# 附录9A 重参数化技巧的完整推导：REINFORCE 与它的方差之痛

> 本附录推导 REINFORCE（得分函数）梯度估计器，分析它的方差性质，并和正文 9.2 节的重参数化估计器做定量对比。读这一节，你会明白为什么 VAE 里"能用重参数化就别用 REINFORCE"——不是它错，而是它方差太大。这个代价决定了 VAE 这类深度生成模型的工程形态，也解释了重参数化为何成为事实标准。

## 为什么需要 REINFORCE？先理解它的位置

正文 9.2 讲：重参数化把随机性外移到 $\epsilon$，梯度顺滑通过。但它要求分布能写成 $z=g_\phi(\epsilon,x)$ 且可微。碰到**离散隐变量**或**不可重参数化的分布**，这招失效。REINFORCE（也叫得分函数估计器）是更通用的后备方案：只要你能从 $q_\phi$ 采样、且知道它的密度，就能估梯度——代价是高方差。从思想定位上看，重参数化与 REINFORCE 是"同一条路的两条岔口"：它们都试图把随机节点处的梯度搬出来，只是一个靠"换一个可微的抽样方式"，一个靠"对密度取对数求导"。

**本附录目标**：
- 推得 REINFORCE 估计器并证明它无偏；
- 看清热方差爆炸的根源；
- 定量对比它和重参数化的方差差距。

---

## 得分函数估计器的推导

### 问题设定

给定分布 $q_\phi(z|x)$ 和函数 $f(z)$，要算：

$$\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)] = \nabla_\phi \int q_\phi(z|x) f(z) dz$$

因为 $q_\phi$ 依赖 $\phi$，梯度和期望不能直接交换。于是问题变成：怎么把"对 $\phi$ 求导"移进期望号里？

### 推导（对数导数技巧）

借一个恒等式 $\nabla_\phi q_\phi = q_\phi \nabla_\phi \log q_\phi$（它就是"先取对数再求导"的连锁结果）：

$$\nabla_\phi \int q_\phi(z|x) f(z) dz = \int \nabla_\phi q_\phi(z|x) f(z) dz = \int q_\phi(z|x) \nabla_\phi \log q_\phi(z|x) f(z) dz$$

于是得到 REINFORCE 估计器：

$$\boxed{\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)] = \mathbb{E}_{q_\phi(z|x)}[f(z) \nabla_\phi \log q_\phi(z|x)]}$$

蒙特卡罗估计：

$$\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)] \approx \frac{1}{L}\sum_{l=1}^{L} f(z^{(l)}) \nabla_\phi \log q_\phi(z^{(l)}|x), \quad z^{(l)} \sim q_\phi(z|x)$$

### 对高斯编码器的具体形式

当 $q_\phi(z|x)=\mathcal{N}(z|\mu_\phi(x),\text{diag}(\sigma_\phi^2(x)))$ 时：

$$\log q_\phi(z|x) = -\frac{1}{2}\sum_{j=1}^{d_z}\left[\log\sigma_j^2 + \frac{(z_j - \mu_j)^2}{\sigma_j^2}\right] + \text{const}$$

对 $\mu_j$ 和 $\log\sigma_j^2$ 求梯度（"得分"）：

$$\frac{\partial \log q_\phi}{\partial \mu_j} = \frac{z_j - \mu_j}{\sigma_j^2}, \qquad \frac{\partial \log q_\phi}{\partial \log\sigma_j^2} = \frac{1}{2}\left[\frac{(z_j - \mu_j)^2}{\sigma_j^2} - 1\right]$$

代回 REINFORCE：

$$\nabla_{\mu_j} \mathbb{E}_{q_\phi}[f(z)] \approx \frac{1}{L}\sum_{l=1}^{L} f(z^{(l)}) \frac{z_j^{(l)} - \mu_j}{\sigma_j^2}$$

$$\nabla_{\log\sigma_j^2} \mathbb{E}_{q_\phi}[f(z)] \approx \frac{1}{2L}\sum_{l=1}^{L} f(z^{(l)}) \left[\frac{(z_j^{(l)} - \mu_j)^2}{\sigma_j^2} - 1\right]$$

---

## 无偏性证明（平均来看它是对的）

**命题**：REINFORCE 估计器无偏。

**证明**：

$$\mathbb{E}_{q_\phi(z|x)}[f(z) \nabla_\phi \log q_\phi(z|x)] = \int q_\phi(z|x) f(z) \frac{\nabla_\phi q_\phi(z|x)}{q_\phi(z|x)} dz = \int f(z) \nabla_\phi q_\phi(z|x) dz$$

$$= \nabla_\phi \int q_\phi(z|x) f(z) dz = \nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)]$$

交换积分与梯度需要正则性条件（$q_\phi$ 可微、$f$ 有界等），实战通常满足。$\square$

> 白话：无偏只保证"平均正确"，不保证"单次稳定"。下面看它到底稳不稳。

---

## 方差分析：问题出在"两个随机量相乘"

### REINFORCE 的方差

$$\text{Var}[f(z) \nabla_\phi \log q_\phi(z|x)] = \mathbb{E}[\|f(z) \nabla_\phi \log q_\phi(z|x)\|^2] - \|\mathbb{E}[f(z) \nabla_\phi \log q_\phi(z|x)]\|^2$$

方差来自三层叠加：
1. **$f(z)$ 的波动**；
2. **$\nabla_\phi\log q_\phi$ 的波动**（得分方向随 $z$ 变）；
3. **两者乘积的波动**——乘积的方差通常远大于各自方差之和，这是 REINFORCE 方差大的根源。

对高斯编码器，$\nabla_{\mu_j}\log q_\phi=(z_j-\mu_j)/\sigma_j^2$ 量级是 $O(1/\sigma_j)$。**$\sigma_j$ 越小，得分方向波动越剧烈，方差越大**。

### 重参数化的方差

$$\text{Var}[\nabla_\phi f(g_\phi(\epsilon, x))] = \mathbb{E}[\|\nabla_\phi f(g_\phi(\epsilon, x))\|^2] - \|\mathbb{E}[\nabla_\phi f(g_\phi(\epsilon, x))]\|^2$$

方差的唯一来源是 $\epsilon$ 的采样。因为 $g_\phi$ 是确定性的，给定 $\epsilon$，梯度就确定了——没有"乘积放大"。

### 定量对比（最简一维例子）

取 $q_\phi(z)=\mathcal{N}(\mu,\sigma^2)$，$f(z)=z^2$。

**REINFORCE**：

$$\nabla_\mu \mathbb{E}_{q_\phi}[z^2] \approx \frac{1}{L}\sum_{l=1}^{L} (z^{(l)})^2 \frac{z^{(l)} - \mu}{\sigma^2}$$

方差 $\propto O(\sigma^2 + \mu^2\sigma^{-2})$——$\sigma$ 小时方差极大。

**重参数化**：

$$z = \mu + \sigma\epsilon,\quad f(z)=(\mu+\sigma\epsilon)^2,\quad \nabla_\mu f = 2(\mu+\sigma\epsilon)$$

方差 $\propto O(\sigma^2)$——随 $\sigma$ 减小而减小。

**结论**：在编码器比较"确信"（$\sigma$ 小）的区域，重参数化的方差优势最明显。正文 9.2 的实验 9.2-1 正是用这个例子把差距画了出来。

---

## 方差缩减技术（万一只能用 REINFORCE）

### Baseline（基线）

用 $f(z)-b$ 替换 $f(z)$，$b$ 与 $z$ 无关：

$$\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{q_\phi}[(f(z) - b) \nabla_\phi \log q_\phi(z|x)]$$

无偏性不变（因为 $\mathbb{E}_{q_\phi}[\nabla_\phi\log q_\phi]=\nabla_\phi\mathbb{E}_{q_\phi}[1]=0$），但方差可降。最优基线 $b^*=\mathbb{E}[f(z)\|\nabla_\phi\log q_\phi\|^2]/\mathbb{E}[\|\nabla_\phi\log q_\phi\|^2]$，实践常用 $f(z)$ 的运行平均近似。

### Control Variate（控制变量）

引入和 $f(z)$ 相关、但梯度已知的函数 $h(z)$：

$$\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{q_\phi}[f(z) \nabla_\phi \log q_\phi] - c\cdot(\mathbb{E}_{q_\phi}[h(z)\nabla_\phi \log q_\phi] - \nabla_\phi \mathbb{E}_{q_\phi}[h(z)])$$

选好 $c$ 和 $h$ 能降方差，但设计好控制变量需要领域知识。

### 对比总结

| 方法 | 无偏性 | 方差 | 计算代价 | 适用条件 |
|---|---|---|---|---|
| REINFORCE | 是 | 高 | 需 $\nabla_\phi \log q_\phi$ | 任意 $q_\phi$ |
| REINFORCE + baseline | 是 | 中 | 同上 | 任意 $q_\phi$ |
| 重参数化 | 是 | 低 | 需通过 $g_\phi$ 反向传播 | $q_\phi$ 可重参数化 |

**实践建议**：VAE 常用的对角高斯编码器，优先用重参数化；只有隐变量离散或不可重参数化时才上 REINFORCE。

## 收尾：为什么这件事重要

REINFORCE 不是错的——它无偏、通用、连离散分布都能用。但它"两个随机量相乘"的结构，让它在编码器很确信（$\sigma$ 小）时方差爆炸，而那恰恰是 VAE 训练后期最常见的状态。重参数化正是绕开了"相乘"，把随机性锁在 $\epsilon$ 里，让梯度只来自确定性的 $g_\phi$。所以结论很朴素：**只要能重参数化，就别用 REINFORCE**——这正是 VAE 工程实践里那把省下的算力。

从更深层次看，这个"方差之痛"并非单纯的工程细节：它决定了深度生成模型里"随机性该放在哪"。重参数化把随机性当作"可被微分的噪声"，让梯度的方向不再被采样噪声淹没，VAE 才得以稳定地学到隐空间结构——而这份"让噪声可微分"的设计哲学，也正是扩散模型把整个生成过程写成一条加噪-去噪、处处可反向传播链路的思想先声。