# 附录9B 高斯KL散度闭式解推导

本附录推导两个高斯分布之间的KL散度闭式公式，特别关注VAE中常见的对角高斯编码器与标准正态先验之间的KL散度。

## 一般高斯分布间的KL散度

### 定理

设 $q = \mathcal{N}(\mu_0, \Sigma_0)$ 和 $p = \mathcal{N}(\mu_1, \Sigma_1)$ 为 $d$ 维高斯分布，则：

$$D_{\text{KL}}(q \| p) = \frac{1}{2}\left[\text{tr}(\Sigma_1^{-1}\Sigma_0) - d + (\mu_1 - \mu_0)^\top \Sigma_1^{-1}(\mu_1 - \mu_0) + \ln\frac{\det\Sigma_1}{\det\Sigma_0}\right]$$

### 推导

KL散度的定义为：

$$D_{\text{KL}}(q \| p) = \mathbb{E}_{q}[\log q(z) - \log p(z)]$$

高斯分布的对数密度为：

$$\log \mathcal{N}(z | \mu, \Sigma) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log\det\Sigma - \frac{1}{2}(z-\mu)^\top\Sigma^{-1}(z-\mu)$$

因此：

$$\log q(z) - \log p(z) = -\frac{1}{2}\log\frac{\det\Sigma_0}{\det\Sigma_1} - \frac{1}{2}(z-\mu_0)^\top\Sigma_0^{-1}(z-\mu_0) + \frac{1}{2}(z-\mu_1)^\top\Sigma_1^{-1}(z-\mu_1)$$

取期望 $\mathbb{E}_{z \sim q}[\cdot]$：

**第一项**：$-\frac{1}{2}\log\frac{\det\Sigma_0}{\det\Sigma_1}$（与 $z$ 无关）

**第二项**：$\mathbb{E}_{q}[(z-\mu_0)^\top\Sigma_0^{-1}(z-\mu_0)] = \text{tr}(\Sigma_0^{-1}\mathbb{E}[(z-\mu_0)(z-\mu_0)^\top]) = \text{tr}(\Sigma_0^{-1}\Sigma_0) = d$

**第三项**：利用 $z - \mu_1 = (z - \mu_0) + (\mu_0 - \mu_1)$，展开：

$$\mathbb{E}_{q}[(z-\mu_1)^\top\Sigma_1^{-1}(z-\mu_1)] = \mathbb{E}_{q}[(z-\mu_0 + \mu_0 - \mu_1)^\top\Sigma_1^{-1}(z-\mu_0 + \mu_0 - \mu_1)]$$

$$= \mathbb{E}_{q}[(z-\mu_0)^\top\Sigma_1^{-1}(z-\mu_0)] + (\mu_0 - \mu_1)^\top\Sigma_1^{-1}(\mu_0 - \mu_1)$$

交叉项为0（因为 $\mathbb{E}[z-\mu_0] = 0$）。

第一部分：

$$\mathbb{E}_{q}[(z-\mu_0)^\top\Sigma_1^{-1}(z-\mu_0)] = \text{tr}(\Sigma_1^{-1}\mathbb{E}[(z-\mu_0)(z-\mu_0)^\top]) = \text{tr}(\Sigma_1^{-1}\Sigma_0)$$

合并所有项：

$$D_{\text{KL}}(q \| p) = -\frac{1}{2}\log\frac{\det\Sigma_0}{\det\Sigma_1} - \frac{d}{2} + \frac{1}{2}\text{tr}(\Sigma_1^{-1}\Sigma_0) + \frac{1}{2}(\mu_1 - \mu_0)^\top\Sigma_1^{-1}(\mu_1 - \mu_0)$$

整理得到：

$$\boxed{D_{\text{KL}}(q \| p) = \frac{1}{2}\left[\text{tr}(\Sigma_1^{-1}\Sigma_0) - d + (\mu_1 - \mu_0)^\top\Sigma_1^{-1}(\mu_1 - \mu_0) + \ln\frac{\det\Sigma_1}{\det\Sigma_0}\right]}$$

## VAE中的简化情形

在VAE中，$q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$，$p(z) = \mathcal{N}(0, I)$。即：

- $\mu_0 = \mu_\phi(x)$，$\mu_1 = 0$
- $\Sigma_0 = \text{diag}(\sigma_\phi^2(x))$，$\Sigma_1 = I$

逐项计算：

- $\text{tr}(\Sigma_1^{-1}\Sigma_0) = \text{tr}(\text{diag}(\sigma_\phi^2)) = \sum_{j=1}^{d_z} \sigma_j^2$
- $d = d_z$
- $(\mu_1 - \mu_0)^\top\Sigma_1^{-1}(\mu_1 - \mu_0) = \|\mu_\phi(x)\|^2 = \sum_{j=1}^{d_z}\mu_j^2$
- $\ln\frac{\det\Sigma_1}{\det\Sigma_0} = -\ln\det(\text{diag}(\sigma_\phi^2)) = -\sum_{j=1}^{d_z}\ln\sigma_j^2$

合并：

$$D_{\text{KL}}(q_\phi(z|x) \| p(z)) = \frac{1}{2}\sum_{j=1}^{d_z}\left[\sigma_j^2 - 1 + \mu_j^2 - \ln\sigma_j^2\right]$$

$$\boxed{D_{\text{KL}}(q_\phi(z|x) \| p(z)) = \frac{1}{2}\sum_{j=1}^{d_z}\left(\mu_j^2 + \sigma_j^2 - \ln\sigma_j^2 - 1\right)}$$

## 逐维度分解

KL散度可以逐维度分解，每个维度的贡献独立计算：

$$D_{\text{KL},j} = \frac{1}{2}\left(\mu_j^2 + \sigma_j^2 - \ln\sigma_j^2 - 1\right)$$

这一性质来自对角协方差假设——各维度独立，KL散度可加。

**各项含义**：

| 项 | 含义 |
|---|---|
| $\mu_j^2$ | 均值偏离0的惩罚 |
| $\sigma_j^2$ | 方差偏离1（偏大）的惩罚 |
| $-\ln\sigma_j^2$ | 方差偏离1（偏小）的惩罚 |
| $-1$ | 常数，保证 $\mu_j=0, \sigma_j^2=1$ 时 $D_{\text{KL},j}=0$ |

**验证**：当 $q_\phi(z_j|x) = \mathcal{N}(0, 1) = p(z_j)$ 时，$\mu_j = 0$，$\sigma_j^2 = 1$：

$$D_{\text{KL},j} = \frac{1}{2}(0 + 1 - 0 - 1) = 0 \quad \checkmark$$

## 数值稳定实现

在实践中，编码器输出的是 $\log\sigma_j^2$ 而非 $\sigma_j^2$，这样做有几个好处。用 `logvar` $= \log\sigma_j^2$ 表示：

$$D_{\text{KL},j} = \frac{1}{2}\left(\mu_j^2 + e^{\text{logvar}_j} - \text{logvar}_j - 1\right)$$

PyTorch实现：

```python
def kl_divergence(mu, logvar):
    """
    计算高斯编码器与标准正态先验之间的KL散度

    参数:
        mu: 编码器输出的均值, shape (batch_size, latent_dim)
        logvar: 编码器输出的对数方差, shape (batch_size, latent_dim)

    返回:
        KL散度, shape (batch_size,)
    """
    # 逐维度计算: 0.5 * (μ² + exp(logσ²) - logσ² - 1)
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
    # 对所有维度求和
    return kl_per_dim.sum(dim=-1)
```

**数值稳定性注意**：

1. 避免直接计算 $\sigma_j^2 = e^{\text{logvar}_j}$ 然后计算 $\sigma_j^2 - \log\sigma_j^2$——这在 `logvar` 很大或很小时可能溢出。上面的实现直接用 `logvar.exp() - logvar` 是安全的。

2. 当 `logvar` 非常负（$\sigma_j^2$ 接近0）时，$e^{\text{logvar}_j}$ 接近0，$-\text{logvar}_j$ 很大，KL散度趋于 $\infty$——这是正确的：方差趋于0意味着编码器变得极其确定，偏离了先验的"扩散"要求。

3. 实践中，可以对 `logvar` 做裁剪（如限制在 $[-10, 10]$）以避免极端情况。
