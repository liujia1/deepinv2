# 附录10A 层级ELBO的完整推导

> 定位：为10.1节提供层级ELBO的详细代数推导。

---

## 起点：对数似然分解

从对数似然出发，引入变分分布 $q(\mathbf{z}_{1:L}|\mathbf{x})$：

$$\log p(\mathbf{x}) = \log \int p(\mathbf{x}, \mathbf{z}_{1:L})\,d\mathbf{z}_{1:L}$$

乘以 $q(\mathbf{z}_{1:L}|\mathbf{x}) / q(\mathbf{z}_{1:L}|\mathbf{x}) = 1$：

$$\log p(\mathbf{x}) = \log \mathbb{E}_{q(\mathbf{z}_{1:L}|\mathbf{x})}\left[\frac{p(\mathbf{x}, \mathbf{z}_{1:L})}{q(\mathbf{z}_{1:L}|\mathbf{x})}\right]$$

由Jensen不等式（$\log$ 为凹函数）：

$$\log p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z}_{1:L}|\mathbf{x})}\left[\log \frac{p(\mathbf{x}, \mathbf{z}_{1:L})}{q(\mathbf{z}_{1:L}|\mathbf{x})}\right] =: \text{ELBO}(\mathbf{x})$$

等号成立当且仅当 $q(\mathbf{z}_{1:L}|\mathbf{x}) = p(\mathbf{z}_{1:L}|\mathbf{x})$。

---

## 展开联合分布

### 生成联合分布

利用马尔可夫结构，联合生成分布展开为：

$$p(\mathbf{x}, \mathbf{z}_{1:L}) = p(\mathbf{z}_L) \prod_{l=1}^{L-1} p(\mathbf{z}_l|\mathbf{z}_{l+1}) \cdot p(\mathbf{x}|\mathbf{z}_1)$$

取对数：

$$\log p(\mathbf{x}, \mathbf{z}_{1:L}) = \log p(\mathbf{z}_L) + \sum_{l=1}^{L-1} \log p(\mathbf{z}_l|\mathbf{z}_{l+1}) + \log p(\mathbf{x}|\mathbf{z}_1)$$

### 推断联合分布

$$q(\mathbf{z}_{1:L}|\mathbf{x}) = \prod_{l=1}^{L} q(\mathbf{z}_l|\mathbf{z}_{l-1})$$

取对数：

$$\log q(\mathbf{z}_{1:L}|\mathbf{x}) = \sum_{l=1}^{L} \log q(\mathbf{z}_l|\mathbf{z}_{l-1})$$

---

## 代入ELBO

$$\text{ELBO}(\mathbf{x}) = \mathbb{E}_{q(\mathbf{z}_{1:L}|\mathbf{x})}\left[\log p(\mathbf{x}|\mathbf{z}_1) + \sum_{l=1}^{L-1} \log p(\mathbf{z}_l|\mathbf{z}_{l+1}) + \log p(\mathbf{z}_L) - \sum_{l=1}^{L} \log q(\mathbf{z}_l|\mathbf{z}_{l-1})\right]$$

重新组织，将 $q$ 和 $p$ 配对：

$$= \mathbb{E}_{q}[\log p(\mathbf{x}|\mathbf{z}_1)] + \sum_{l=1}^{L-1} \mathbb{E}_{q}\left[\log \frac{p(\mathbf{z}_l|\mathbf{z}_{l+1})}{q(\mathbf{z}_l|\mathbf{z}_{l-1})}\right] + \mathbb{E}_{q}\left[\log \frac{p(\mathbf{z}_L)}{q(\mathbf{z}_L|\mathbf{z}_{L-1})}\right]$$

---

## 转化为KL散度

对于 $l = 1, \ldots, L-1$ 的每一项：

$$\mathbb{E}_{q}\left[\log \frac{p(\mathbf{z}_l|\mathbf{z}_{l+1})}{q(\mathbf{z}_l|\mathbf{z}_{l-1})}\right]$$

这并非直接的KL散度，因为 $q$ 的条件变量是 $\mathbf{z}_{l-1}$（推断方向），$p$ 的条件变量是 $\mathbf{z}_{l+1}$（生成方向），两者条件不同。

为了得到KL散度形式，需要在适当的边际分布上取期望。具体地，对第 $l$ 层：

$$\mathbb{E}_{q(\mathbf{z}_{l-1}, \mathbf{z}_{l+1}|\mathbf{x})}\left[D_{\text{KL}}(q(\mathbf{z}_l|\mathbf{z}_{l-1}) \| p(\mathbf{z}_l|\mathbf{z}_{l+1}))\right]$$

$$= \mathbb{E}_{q(\mathbf{z}_{l-1}, \mathbf{z}_{l+1}|\mathbf{x})}\left[\int q(\mathbf{z}_l|\mathbf{z}_{l-1}) \log \frac{q(\mathbf{z}_l|\mathbf{z}_{l-1})}{p(\mathbf{z}_l|\mathbf{z}_{l+1})} d\mathbf{z}_l\right]$$

类似地，第 $L$ 层（顶层）：

$$\mathbb{E}_{q(\mathbf{z}_{L-1}|\mathbf{x})}\left[D_{\text{KL}}(q(\mathbf{z}_L|\mathbf{z}_{L-1}) \| p(\mathbf{z}_L))\right]$$

---

## 最终形式

综合以上推导：

$$\text{ELBO}(\mathbf{x}) = \mathbb{E}_{q(\mathbf{z}_{1:L}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z}_1)] - \sum_{l=1}^{L-1} \mathbb{E}_{q(\mathbf{z}_{l-1}, \mathbf{z}_{l+1}|\mathbf{x})}\left[D_{\text{KL}}(q(\mathbf{z}_l|\mathbf{z}_{l-1}) \| p(\mathbf{z}_l|\mathbf{z}_{l+1}))\right] - \mathbb{E}_{q(\mathbf{z}_{L-1}|\mathbf{x})}\left[D_{\text{KL}}(q(\mathbf{z}_L|\mathbf{z}_{L-1}) \| p(\mathbf{z}_L))\right]$$

合并为统一形式（约定 $p(\mathbf{z}_L|\mathbf{z}_{L+1}) = p(\mathbf{z}_L)$，$\mathbf{z}_0 = \mathbf{x}$）：

$$\boxed{\text{ELBO}(\mathbf{x}) = \mathbb{E}_{q}[\log p(\mathbf{x}|\mathbf{z}_1)] - \sum_{l=1}^{L} \mathbb{E}_{q}\left[D_{\text{KL}}(q(\mathbf{z}_l|\mathbf{z}_{l-1}) \| p(\mathbf{z}_l|\mathbf{z}_{l+1}))\right]}$$

---

## 与单层ELBO的一致性验证

当 $L = 1$ 时，只有一个潜变量 $\mathbf{z}_1$：

$$\text{ELBO}(\mathbf{x}) = \mathbb{E}_{q(\mathbf{z}_1|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z}_1)] - D_{\text{KL}}(q(\mathbf{z}_1|\mathbf{x}) \| p(\mathbf{z}_1))$$

这正是第8章的标准单层ELBO——重建项减去KL项。层级ELBO是单层ELBO的自然推广。$\blacksquare$

---

## 扩散模型标记下的等价形式

将 $\mathbf{z}_l$ 替换为 $\mathbf{x}_t$，$\mathbf{z}_0 = \mathbf{x}$ 替换为 $\mathbf{x}_0$，层级ELBO变为：

$$\text{ELBO}(\mathbf{x}_0) = \mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)}[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)] - \sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{t-1}, \mathbf{x}_{t+1}|\mathbf{x}_0)}\left[D_{\text{KL}}(q(\mathbf{x}_t|\mathbf{x}_{t-1}) \| p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1}))\right] - D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_{T-1}) \| p(\mathbf{x}_T))$$

通过10.2.3节的贝叶斯反转技巧，可以将其转化为更实用的形式（Theorem 2.4 in Tutorial on Diffusion Models）：

$$\text{ELBO}(\mathbf{x}_0) = \mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)}[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)] - D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T)) - \sum_{t=2}^{T} \mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)}\left[D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))\right]$$

**来源**：Tutorial on Diffusion Models for Imaging and Vision Theorem 2.3-2.4; Ho et al. (2020) DDPM Appendix A
