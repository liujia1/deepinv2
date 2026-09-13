# 附录 10A 层级 ELBO 的完整推导

> 定位：给 10.1 节"层级 ELBO 推导"补上详细代数。如果你只想用结论，可以跳过本附录——记住那行框起来的公式就够了。本附录只做一件事：把"单层 ELBO 推广到多层"的每一步补齐，并验证 $L=1$ 时退化回标准单层 ELBO。从更深层次看，这份推导正是本书变分路径的承重墙——后面把每个 $z_l$ 换成时间步 $x_t$ 时，10.2 节的三项 VLB 就从这堵墙里自然长出来；把它走一遍，你就理解了为什么扩散的训练目标"从头到尾都是 KL 散度在排队"。

---

## 起点：对数似然分解

我们想给数据 $x$ 的对数似然 $\log p(x)$ 找一个"可计算的下界"。办法是引入一个变分分布（variational distribution，我们故意选的简单分布，用来近似难算的真实后验）$q(z_{1:L}|x)$，在积分里乘一个 $q/q=1$：

$$\log p(\mathbf{x}) = \log \int p(\mathbf{x}, \mathbf{z}_{1:L})\,d\mathbf{z}_{1:L}
= \log \mathbb{E}_{q(\mathbf{z}_{1:L}|\mathbf{x})}\left[\frac{p(\mathbf{x}, \mathbf{z}_{1:L})}{q(\mathbf{z}_{1:L}|\mathbf{x})}\right]$$

因为 $\log$ 是凹函数，由 Jensen 不等式（Jensen's inequality，凹函数把期望的 log 放大成 log 的期望之上界），把 log 挪进期望里面会得到一个**下界**：

$$\log p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z}_{1:L}|\mathbf{x})}\left[\log \frac{p(\mathbf{x}, \mathbf{z}_{1:L})}{q(\mathbf{z}_{1:L}|\mathbf{x})}\right] =: \text{ELBO}(\mathbf{x})$$

等号成立当且仅当 $q(z_{1:L}|x)=p(z_{1:L}|x)$（近似分布正好等于真实后验）。这个下界就是 ELBO（evidence lower bound，证据下界）。

---

## 展开联合分布

### 生成联合分布

利用 10.1 节的马尔可夫结构，联合生成分布展开成链条：

$$p(\mathbf{x}, \mathbf{z}_{1:L}) = p(\mathbf{z}_L) \prod_{l=1}^{L-1} p(\mathbf{z}_l|\mathbf{z}_{l+1}) \cdot p(\mathbf{x}|\mathbf{z}_1)$$

取对数（乘积变求和）：

$$\log p(\mathbf{x}, \mathbf{z}_{1:L}) = \log p(\mathbf{z}_L) + \sum_{l=1}^{L-1} \log p(\mathbf{z}_l|\mathbf{z}_{l+1}) + \log p(\mathbf{x}|\mathbf{z}_1)$$

### 推断联合分布

$$q(\mathbf{z}_{1:L}|\mathbf{x}) = \prod_{l=1}^{L} q(\mathbf{z}_l|\mathbf{z}_{l-1}),\quad (\mathbf{z}_0\equiv\mathbf{x})$$

取对数：

$$\log q(\mathbf{z}_{1:L}|\mathbf{x}) = \sum_{l=1}^{L} \log q(\mathbf{z}_l|\mathbf{z}_{l-1})$$

---

## 代入 ELBO

把上面两项代回 ELBO 定义，把 $p$ 的部分和 $q$ 的部分配对：

$$\text{ELBO}(\mathbf{x}) = \mathbb{E}_{q}\left[\log p(\mathbf{x}|\mathbf{z}_1) + \sum_{l=1}^{L-1} \log p(\mathbf{z}_l|\mathbf{z}_{l+1}) + \log p(\mathbf{z}_L) - \sum_{l=1}^{L} \log q(\mathbf{z}_l|\mathbf{z}_{l-1})\right]$$

重新组织（把每层的 $p$ 和对应的 $q$ 凑一对）：

$$= \mathbb{E}_{q}[\log p(\mathbf{x}|\mathbf{z}_1)] + \sum_{l=1}^{L-1} \mathbb{E}_{q}\left[\log \frac{p(\mathbf{z}_l|\mathbf{z}_{l+1})}{q(\mathbf{z}_l|\mathbf{z}_{l-1})}\right] + \mathbb{E}_{q}\left[\log \frac{p(\mathbf{z}_L)}{q(\mathbf{z}_L|\mathbf{z}_{L-1})}\right]$$

---

## 转化为 KL 散度

这里有个小坎：对 $l=1,\ldots,L-1$，上面那一项里 $p$ 的条件变量是 $z_{l+1}$（生成方向，自顶向下），而 $q$ 的条件变量是 $z_{l-1}$（推断方向，自底向上）——两边条件不同，不能直接认作 KL 散度。

KL 散度（Kullback–Leibler divergence，衡量两个分布差异、永远 ≥0 的量）要求同一个条件下的两个分布之比。所以我们改在"恰当的边际"上取期望。对第 $l$ 层，在 $z_{l-1},z_{l+1}$ 的边际上：

$$\mathbb{E}_{q(\mathbf{z}_{l-1}, \mathbf{z}_{l+1}|\mathbf{x})}\left[D_{\text{KL}}(q(\mathbf{z}_l|\mathbf{z}_{l-1}) \| p(\mathbf{z}_l|\mathbf{z}_{l+1}))\right]
= \mathbb{E}_{q(\mathbf{z}_{l-1}, \mathbf{z}_{l+1}|\mathbf{x})}\left[\int q(\mathbf{z}_l|\mathbf{z}_{l-1}) \log \frac{q(\mathbf{z}_l|\mathbf{z}_{l-1})}{p(\mathbf{z}_l|\mathbf{z}_{l+1})} d\mathbf{z}_l\right]$$

这一改写恰好等于前面那一项。同理，第 $L$ 层（顶层，生成侧无条件、只有先验）：

$$\mathbb{E}_{q(\mathbf{z}_{L-1}|\mathbf{x})}\left[D_{\text{KL}}(q(\mathbf{z}_L|\mathbf{z}_{L-1}) \| p(\mathbf{z}_L))\right]$$

---

## 最终形式

把以上拼起来，并约定 $p(z_L|z_{L+1})\equiv p(z_L)$、$\mathbf{z}_0=\mathbf{x}$，得到正文里那行框起来的公式：

$$\boxed{\text{ELBO}(\mathbf{x}) = \mathbb{E}_{q(\mathbf{z}_{1:L}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z}_1)] - \sum_{l=1}^{L} \mathbb{E}_{q(\mathbf{z}_{l-1}, \mathbf{z}_{l+1}|\mathbf{x})}\left[D_{\text{KL}}(q(\mathbf{z}_l|\mathbf{z}_{l-1}) \| p(\mathbf{z}_l|\mathbf{z}_{l+1}))\right]}$$

直觉复述一遍：第一项是"重建项"（让解码器还原数据）；后面那一长串求和，**每一层一个 KL 项**，逼着推断链（自底向上）和生成链（自顶向下）在每一层对齐。

---

## 与单层 ELBO 的一致性验证

把 $L=1$ 代进去，只剩一个潜变量 $z_1$：

$$\text{ELBO}(\mathbf{x}) = \mathbb{E}_{q(\mathbf{z}_1|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z}_1)] - D_{\text{KL}}(q(\mathbf{z}_1|\mathbf{x}) \| p(\mathbf{z}_1))$$

这正是第8章的标准单层 ELBO——重建项减 KL 项。所以**层级 ELBO 是单层 ELBO 的自然推广**，不是新东西。$\blacksquare$

---

## 扩散模型记号下的等价形式

把 $z_l$ 换成 $x_t$、$z_0=x$ 换成 $x_0$，层级 ELBO 变成：

$$\text{ELBO}(\mathbf{x}_0) = \mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)}[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)] - \sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{t-1}, \mathbf{x}_{t+1}|\mathbf{x}_0)}\left[D_{\text{KL}}(q(\mathbf{x}_t|\mathbf{x}_{t-1}) \| p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1}))\right] - D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_{T-1}) \| p(\mathbf{x}_T))$$

再用 10.2 节的贝叶斯反转技巧（把"正向相邻"的条件翻成"带 $x_0$ 的后验"），可化成更实用的 VLB 形式（Tutorial on Diffusion Models 的 Theorem 2.4）：

$$\text{ELBO}(\mathbf{x}_0) = \mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)}[\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)] - D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T)) - \sum_{t=2}^{T} \mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_0)}\left[D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))\right]$$

这正是 10.2 节 VLB 三项分解的来源。

收个尾，把这条推导放回全章的位置：我们从"单层 ELBO 除以一个 $q/q=1$"出发，一路走到"扩散版的三项 VLB"，中间没有引入任何新的目标函数，只是把同一套变分工具搬到了更长的链条上。这一转变意味着，扩散模型的可训练目标不是另起炉灶的发明，而是贝叶斯变分推断在层级结构下的忠实翻版——只要愿意把链拉长、把分布设为高斯，扩散的损失就会自动从推导中浮现。

**来源**：Tutorial on Diffusion Models for Imaging and Vision Theorem 2.3-2.4; Ho et al. (2020) DDPM Appendix A
