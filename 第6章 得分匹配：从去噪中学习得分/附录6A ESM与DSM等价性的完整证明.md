# 附录6A ESM与DSM等价性的完整证明

> 定位：为6.3节提供Vincent (2011) 定理 $\mathcal{J}_{\text{DSM}}(\theta) = \mathcal{J}_{\text{ESM}}(\theta) + C$ 的完整数学证明。证明涉及条件期望的分解和交叉项的化简，放在附录以保持主线叙事的流畅。

## 前提与记号

设：

- $p(x)$：数据分布
- $q_\sigma(\tilde{x}|x) = \mathcal{N}(\tilde{x}|x, \sigma^2 I)$：条件噪声分布
- $q_\sigma(\tilde{x}) = \int p(x)\,q_\sigma(\tilde{x}|x)\,dx$：噪声扰动边际分布
- $q_\sigma(x|\tilde{x}) = \frac{p(x)\,q_\sigma(\tilde{x}|x)}{q_\sigma(\tilde{x})}$：后验分布

两个目标函数定义在噪声扰动分布 $q_\sigma$ 上：

**ESM目标**（在 $q_\sigma$ 上）：

$$\mathcal{J}_{\text{ESM}}(\theta) = \frac{1}{2}\mathbb{E}_{q_\sigma(\tilde{x})}\left[\left\|s_\theta(\tilde{x}) - \nabla_{\tilde{x}}\log q_\sigma(\tilde{x})\right\|^2\right]$$

**DSM目标**：

$$\mathcal{J}_{\text{DSM}}(\theta) = \frac{1}{2}\mathbb{E}_{q_\sigma(\tilde{x}, x)}\left[\left\|s_\theta(\tilde{x}) - \nabla_{\tilde{x}}\log q_\sigma(\tilde{x}|x)\right\|^2\right]$$

## 第一步：展开ESM目标

$$\mathcal{J}_{\text{ESM}}(\theta) = \frac{1}{2}\mathbb{E}_{q_\sigma(\tilde{x})}\left[\|s_\theta(\tilde{x})\|^2 - 2\,s_\theta(\tilde{x})^T\nabla\log q_\sigma(\tilde{x}) + \|\nabla\log q_\sigma(\tilde{x})\|^2\right]$$

$$= \frac{1}{2}\mathbb{E}_{q_\sigma(\tilde{x})}\left[\|s_\theta(\tilde{x})\|^2\right] - \mathbb{E}_{q_\sigma(\tilde{x})}\left[s_\theta(\tilde{x})^T\nabla\log q_\sigma(\tilde{x})\right] + C_1$$

其中 $C_1 = \frac{1}{2}\mathbb{E}_{q_\sigma(\tilde{x})}\left[\|\nabla\log q_\sigma(\tilde{x})\|^2\right]$ 与 $\theta$ 无关。

## 第二步：展开DSM目标

$$\mathcal{J}_{\text{DSM}}(\theta) = \frac{1}{2}\mathbb{E}_{q_\sigma(\tilde{x}, x)}\left[\|s_\theta(\tilde{x})\|^2 - 2\,s_\theta(\tilde{x})^T\nabla\log q_\sigma(\tilde{x}|x) + \|\nabla\log q_\sigma(\tilde{x}|x)\|^2\right]$$

$$= \frac{1}{2}\mathbb{E}_{q_\sigma(\tilde{x})}\left[\|s_\theta(\tilde{x})\|^2\right] - \mathbb{E}_{q_\sigma(\tilde{x}, x)}\left[s_\theta(\tilde{x})^T\nabla\log q_\sigma(\tilde{x}|x)\right] + C_2$$

其中 $C_2 = \frac{1}{2}\mathbb{E}_{q_\sigma(\tilde{x}, x)}\left[\|\nabla\log q_\sigma(\tilde{x}|x)\|^2\right]$ 与 $\theta$ 无关。

第一项相同，因此关键在于比较**交叉项**。

## 第三步：化简交叉项

ESM的交叉项：

$$\mathbb{E}_{q_\sigma(\tilde{x})}\left[s_\theta(\tilde{x})^T\nabla\log q_\sigma(\tilde{x})\right]$$

利用 $q_\sigma(\tilde{x}) = \int p(x)\,q_\sigma(\tilde{x}|x)\,dx$，将边际期望转化为联合期望：

$$\mathbb{E}_{q_\sigma(\tilde{x})}\left[s_\theta(\tilde{x})^T\nabla\log q_\sigma(\tilde{x})\right] = \int s_\theta(\tilde{x})^T\nabla\log q_\sigma(\tilde{x})\,q_\sigma(\tilde{x})\,d\tilde{x}$$

$$= \int s_\theta(\tilde{x})^T\nabla q_\sigma(\tilde{x})\,d\tilde{x}$$

将 $q_\sigma(\tilde{x}) = \int p(x)\,q_\sigma(\tilde{x}|x)\,dx$ 代入，交换积分顺序：

$$= \int p(x)\left[\int s_\theta(\tilde{x})^T\nabla_{\tilde{x}} q_\sigma(\tilde{x}|x)\,d\tilde{x}\right]dx$$

利用 $\nabla\log q_\sigma(\tilde{x}|x) = \nabla q_\sigma(\tilde{x}|x)/q_\sigma(\tilde{x}|x)$，即 $\nabla q_\sigma(\tilde{x}|x) = \nabla\log q_\sigma(\tilde{x}|x)\cdot q_\sigma(\tilde{x}|x)$：

$$= \int p(x)\left[\int s_\theta(\tilde{x})^T\nabla\log q_\sigma(\tilde{x}|x)\,q_\sigma(\tilde{x}|x)\,d\tilde{x}\right]dx$$

$$= \mathbb{E}_{q_\sigma(\tilde{x}, x)}\left[s_\theta(\tilde{x})^T\nabla\log q_\sigma(\tilde{x}|x)\right]$$

**ESM的交叉项 = DSM的交叉项！**

## 第四步：得出等价性

$$\mathcal{J}_{\text{DSM}}(\theta) - \mathcal{J}_{\text{ESM}}(\theta) = C_2 - C_1 = C(\sigma)$$

其中：

$$C(\sigma) = \frac{1}{2}\mathbb{E}_{q_\sigma(\tilde{x}, x)}\left[\|\nabla\log q_\sigma(\tilde{x}|x)\|^2\right] - \frac{1}{2}\mathbb{E}_{q_\sigma(\tilde{x})}\left[\|\nabla\log q_\sigma(\tilde{x})\|^2\right]$$

$C(\sigma)$ 仅依赖于噪声水平 $\sigma$ 和数据分布 $p(x)$，与网络参数 $\theta$ 无关。

因此，$\nabla_\theta \mathcal{J}_{\text{DSM}}(\theta) = \nabla_\theta \mathcal{J}_{\text{ESM}}(\theta)$——两者的梯度完全相同，优化轨迹一致。$\square$

## 补充说明

1. **为什么交叉项相等**：直觉上，$\nabla\log q_\sigma(\tilde{x})$ 是边际分布的得分，而 $\nabla\log q_\sigma(\tilde{x}|x)$ 是条件分布的得分。条件得分的期望（对 $x$ 取条件期望）恰好等于边际得分——这可以看作Tweedie等式的一种推广。

2. **$C(\sigma)$ 的含义**：$C(\sigma) = C_2 - C_1$ 是条件得分范数期望与边际得分范数期望之差。由Jensen不等式，$C_1 \leq C_2$（边际得分范数 ≤ 条件得分范数的期望），因此 $C(\sigma) \geq 0$。

3. **$\sigma \to 0$ 的极限**：当 $\sigma \to 0$ 时，$q_\sigma(\tilde{x}) \to p(\tilde{x})$，$q_\sigma(\tilde{x}|x) \to \delta(\tilde{x} - x)$，$C(\sigma) \to 0$。此时DSM与ESM完全一致。

**来源**：Vincent (2011) "A Connection Between Score Matching and Denoising Autoencoders"; Tutorial_Diffusion_Imaging_Vision Theorem 3.4
