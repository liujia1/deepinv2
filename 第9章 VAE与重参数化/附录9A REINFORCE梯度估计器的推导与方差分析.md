# 附录9A REINFORCE梯度估计器的推导与方差分析

本附录推导REINFORCE（得分函数）梯度估计器，分析其方差性质，并与重参数化梯度估计器进行对比。

## 得分函数估计器的推导

### 问题设定

给定概率分布 $q_\phi(z|x)$ 和函数 $f(z)$，我们需要计算梯度：

$$\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)] = \nabla_\phi \int q_\phi(z|x) f(z) dz$$

由于 $q_\phi$ 依赖 $\phi$，梯度与期望不可直接交换。

### 推导

利用恒等式 $\nabla_\phi q_\phi = q_\phi \nabla_\phi \log q_\phi$（对数导数技巧）：

$$\nabla_\phi \int q_\phi(z|x) f(z) dz = \int \nabla_\phi q_\phi(z|x) f(z) dz = \int q_\phi(z|x) \nabla_\phi \log q_\phi(z|x) f(z) dz$$

因此：

$$\boxed{\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)] = \mathbb{E}_{q_\phi(z|x)}[f(z) \nabla_\phi \log q_\phi(z|x)]}$$

蒙特卡罗估计：

$$\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)] \approx \frac{1}{L}\sum_{l=1}^{L} f(z^{(l)}) \nabla_\phi \log q_\phi(z^{(l)}|x), \quad z^{(l)} \sim q_\phi(z|x)$$

### 对高斯编码器的具体形式

当 $q_\phi(z|x) = \mathcal{N}(z | \mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$ 时：

$$\log q_\phi(z|x) = -\frac{1}{2}\sum_{j=1}^{d_z}\left[\log\sigma_j^2 + \frac{(z_j - \mu_j)^2}{\sigma_j^2}\right] + \text{const}$$

对 $\mu_j$ 和 $\log\sigma_j^2$ 的梯度：

$$\frac{\partial \log q_\phi}{\partial \mu_j} = \frac{z_j - \mu_j}{\sigma_j^2}$$

$$\frac{\partial \log q_\phi}{\partial \log\sigma_j^2} = \frac{1}{2}\left[\frac{(z_j - \mu_j)^2}{\sigma_j^2} - 1\right]$$

代入REINFORCE估计器：

$$\nabla_{\mu_j} \mathbb{E}_{q_\phi}[f(z)] \approx \frac{1}{L}\sum_{l=1}^{L} f(z^{(l)}) \frac{z_j^{(l)} - \mu_j}{\sigma_j^2}$$

$$\nabla_{\log\sigma_j^2} \mathbb{E}_{q_\phi}[f(z)] \approx \frac{1}{2L}\sum_{l=1}^{L} f(z^{(l)}) \left[\frac{(z_j^{(l)} - \mu_j)^2}{\sigma_j^2} - 1\right]$$

## 无偏性证明

**命题**：REINFORCE估计器是无偏的。

**证明**：

$$\mathbb{E}_{q_\phi(z|x)}[f(z) \nabla_\phi \log q_\phi(z|x)] = \int q_\phi(z|x) f(z) \nabla_\phi \log q_\phi(z|x) dz$$

$$= \int q_\phi(z|x) f(z) \frac{\nabla_\phi q_\phi(z|x)}{q_\phi(z|x)} dz = \int f(z) \nabla_\phi q_\phi(z|x) dz$$

$$= \nabla_\phi \int q_\phi(z|x) f(z) dz = \nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)]$$

其中交换积分与梯度需要正则性条件（$q_\phi$ 可微，$f$ 有界等），在实践中通常满足。$\square$

## 方差分析

### REINFORCE的方差

REINFORCE估计器的方差为：

$$\text{Var}[f(z) \nabla_\phi \log q_\phi(z|x)] = \mathbb{E}[\|f(z) \nabla_\phi \log q_\phi(z|x)\|^2] - \|\mathbb{E}[f(z) \nabla_\phi \log q_\phi(z|x)]\|^2$$

方差的主要来源：

1. **$f(z)$ 的波动**：函数值随 $z$ 变化
2. **$\nabla_\phi \log q_\phi(z|x)$ 的波动**：得分方向随 $z$ 变化
3. **两者乘积的波动**：乘积的方差通常远大于各自方差的乘积

对于高斯编码器，$\nabla_{\mu_j}\log q_\phi = (z_j - \mu_j)/\sigma_j^2$ 的量级为 $O(1/\sigma_j)$。当 $\sigma_j$ 较小时，得分方向波动剧烈，导致方差极大。

### 重参数化的方差

重参数化估计器的方差为：

$$\text{Var}[\nabla_\phi f(g_\phi(\epsilon, x))] = \mathbb{E}[\|\nabla_\phi f(g_\phi(\epsilon, x))\|^2] - \|\mathbb{E}[\nabla_\phi f(g_\phi(\epsilon, x))]\|^2$$

方差的唯一来源是 $\epsilon$ 的采样。由于 $g_\phi$ 是确定性的，给定 $\epsilon$，梯度是确定的。方差仅来自不同 $\epsilon$ 采样导致的梯度变化。

### 定量对比

考虑最简单的一维例子：$q_\phi(z) = \mathcal{N}(\mu, \sigma^2)$，$f(z) = z^2$。

**REINFORCE**：

$$\nabla_\mu \mathbb{E}_{q_\phi}[z^2] \approx \frac{1}{L}\sum_{l=1}^{L} (z^{(l)})^2 \frac{z^{(l)} - \mu}{\sigma^2}$$

方差 $\propto O(\sigma^2 + \mu^2\sigma^{-2})$——当 $\sigma$ 小时方差极大。

**重参数化**：

$$z = \mu + \sigma\epsilon, \quad f(z) = (\mu + \sigma\epsilon)^2$$

$$\nabla_\mu f = 2(\mu + \sigma\epsilon)$$

方差 $\propto O(\sigma^2)$——随 $\sigma$ 减小而减小。

**结论**：在 $\sigma$ 较小的区域（即编码器较确信时），重参数化的方差优势更为明显。

## 方差缩减技术

### Baseline

用 $f(z) - b$ 替换 $f(z)$，其中 $b$ 与 $z$ 无关：

$$\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{q_\phi}[(f(z) - b) \nabla_\phi \log q_\phi(z|x)]$$

无偏性不变（因为 $\mathbb{E}_{q_\phi}[\nabla_\phi \log q_\phi] = \nabla_\phi \mathbb{E}_{q_\phi}[1] = 0$），但方差可以降低。

最优基线 $b^* = \mathbb{E}_{q_\phi}[f(z) \|\nabla_\phi \log q_\phi\|^2] / \mathbb{E}_{q_\phi}[\|\nabla_\phi \log q_\phi\|^2]$，实践中通常用 $f(z)$ 的运行平均近似。

### Control Variate

引入与 $f(z)$ 相关的函数 $h(z)$，其梯度已知：

$$\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{q_\phi}[f(z) \nabla_\phi \log q_\phi] - c \cdot (\mathbb{E}_{q_\phi}[h(z) \nabla_\phi \log q_\phi] - \nabla_\phi \mathbb{E}_{q_\phi}[h(z)])$$

选择合适的 $c$ 和 $h$ 可以降低方差，但设计好的控制变量需要领域知识。

### 对比总结

| 方法 | 无偏性 | 方差 | 计算代价 | 适用条件 |
|---|---|---|---|---|
| REINFORCE | 是 | 高 | 需 $\nabla_\phi \log q_\phi$ | 任意 $q_\phi$ |
| REINFORCE + baseline | 是 | 中 | 同上 | 任意 $q_\phi$ |
| 重参数化 | 是 | 低 | 需通过 $g_\phi$ 反向传播 | $q_\phi$ 可重参数化 |

**实践建议**：对于VAE中常用的对角高斯编码器，优先使用重参数化。仅在隐变量为离散或不可重参数化的情况下使用REINFORCE。
