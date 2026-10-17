# 附录6A ESM 与 DSM 等价性的完整证明

> 定位：给 6.3 节的 Vincent (2011) 定理 $\mathcal{J}_{\text{DSM}}(\theta)=\mathcal{J}_{\text{ESM}}(\theta)+C$ 补上完整的数学证明。证明其实就是把两边展开、比交叉项——核心 trick 是"边际得分的条件期望等于条件得分"。放附录是为了不让主线叙事断节奏。

## 为什么需要这个证明

6.3 节我们说：**DSM 训出来的网络，和"在加噪分布上做 ESM"训出来的网络，是同一个**。这句话是整章的支柱——它保证了"去噪即学得分"不只是直觉，而是有数学保证的。可凭什么一个"拿含噪图当输入、预测 $-\mathbf{z}/\sigma$"的目标，能等价于"拟合真实得分"？本附录把这件事钉死。

**一句话直觉**：两边展开后，只有交叉项不同；而交叉项里，"对加噪图求边际得分"与"给定干净图求条件得分、再对干净图平均"其实是同一件事（这就是 Tweedie 等式的精神）。所以两个目标只差一个与网络参数无关的常数 $C$——优化它们得到的解完全相同。这一结论至关重要：它意味着"去噪即学得分"不是巧合，而是必然。

> 下面的推导是主线之外的内容，**可以跳过**；结论（差一个常数、最优解相同）才是关键。

## 前提与记号

设：

- $p(\mathbf{x})$：数据分布；
- $q_\sigma(\tilde \mathbf{x}|\mathbf{x})=\mathcal \mathcal{N}(\tilde \mathbf{x}|\mathbf{x},\sigma^2 I)$：条件噪声分布；
- $q_\sigma(\tilde \mathbf{x})=\int p(\mathbf{x})\,q_\sigma(\tilde \mathbf{x}|\mathbf{x})\,dx$：噪声扰动后的边际分布；
- $q_\sigma(\mathbf{x}|\tilde \mathbf{x})=\dfrac{p(\mathbf{x})\,q_\sigma(\tilde \mathbf{x}|\mathbf{x})}{q_\sigma(\tilde \mathbf{x})}$：后验分布。

两个目标都建立在**加噪分布** $q_\sigma$ 上：

**ESM 目标**（在 $q_\sigma$ 上）：

$$\mathcal{J}_{\text{ESM}}(\theta)=\frac{1}{2}\,\mathbb{E}_{q_\sigma(\tilde \mathbf{x})}\Big[\big\|s_\theta(\tilde \mathbf{x})-\nabla_{\tilde \mathbf{x}}\log q_\sigma(\tilde \mathbf{x})\big\|^2\Big].$$

**DSM 目标**：

$$\mathcal{J}_{\text{DSM}}(\theta)=\frac{1}{2}\,\mathbb{E}_{q_\sigma(\tilde \mathbf{x},\mathbf{x})}\Big[\big\|s_\theta(\tilde \mathbf{x})-\nabla_{\tilde \mathbf{x}}\log q_\sigma(\tilde \mathbf{x}|\mathbf{x})\big\|^2\Big].$$

两者的差别一眼可见：ESM 用边际得分 $\nabla\log q_\sigma(\tilde \mathbf{x})$ 作监督，而 DSM 用条件得分 $\nabla\log q_\sigma(\tilde \mathbf{x}|\mathbf{x})$ 作监督——后者是闭式已知的。所谓等价性，就是要证明这种"监督信号的替换"不改变优化结果。

---

## 第一步：展开 ESM 目标

$$\mathcal{J}_{\text{ESM}}(\theta)=\frac{1}{2}\,\mathbb{E}_{q_\sigma(\tilde \mathbf{x})}\Big[\|s_\theta\|^2-2\,s_\theta^T\nabla\log q_\sigma+\|\nabla\log q_\sigma\|^2\Big].$$

$$=\frac{1}{2}\,\mathbb{E}_{q_\sigma(\tilde \mathbf{x})}\big[\|s_\theta\|^2\big]-\mathbb{E}_{q_\sigma(\tilde \mathbf{x})}\big[s_\theta^T\nabla\log q_\sigma\big]+C_1,$$

其中 $C_1=\frac{1}{2}\mathbb{E}_{q_\sigma(\tilde \mathbf{x})}[\|\nabla\log q_\sigma\|^2]$ 与 $\theta$ 无关。

---

## 第二步：展开 DSM 目标

$$\mathcal{J}_{\text{DSM}}(\theta)=\frac{1}{2}\,\mathbb{E}_{q_\sigma(\tilde \mathbf{x},\mathbf{x})}\Big[\|s_\theta\|^2-2\,s_\theta^T\nabla\log q_\sigma(\tilde \mathbf{x}|\mathbf{x})+\|\nabla\log q_\sigma(\tilde \mathbf{x}|\mathbf{x})\|^2\Big].$$

$$=\frac{1}{2}\,\mathbb{E}_{q_\sigma(\tilde \mathbf{x})}\big[\|s_\theta\|^2\big]-\mathbb{E}_{q_\sigma(\tilde \mathbf{x},\mathbf{x})}\big[s_\theta^T\nabla\log q_\sigma(\tilde \mathbf{x}|\mathbf{x})\big]+C_2,$$

其中 $C_2=\frac{1}{2}\mathbb{E}_{q_\sigma(\tilde \mathbf{x},\mathbf{x})}[\|\nabla\log q_\sigma(\tilde \mathbf{x}|\mathbf{x})\|^2]$ 与 $\theta$ 无关。

第一项两边相同（注意 $\mathbb{E}_{q_\sigma(\tilde \mathbf{x})}=\mathbb{E}_{q_\sigma(\tilde \mathbf{x},\mathbf{x})}$，因为后者对 $\mathbf{x}$ 的边际就是前者），所以**关键就在比交叉项**。

---

## 第三步：化简交叉项

ESM 的交叉项：

$$\mathbb{E}_{q_\sigma(\tilde \mathbf{x})}\big[s_\theta^T\nabla\log q_\sigma(\tilde \mathbf{x})\big].$$

利用 $q_\sigma(\tilde \mathbf{x})=\int p(\mathbf{x})\,q_\sigma(\tilde \mathbf{x}|\mathbf{x})\,dx$，把边际期望改写成联合期望。先写 $\nabla\log q_\sigma=\nabla q_\sigma/q_\sigma$，于是

$$\mathbb{E}_{q_\sigma(\tilde \mathbf{x})}\big[s_\theta^T\nabla\log q_\sigma\big]=\int s_\theta^T\nabla q_\sigma\,d\tilde \mathbf{x}
=\int p(\mathbf{x})\Big[\int s_\theta^T\nabla_{\tilde \mathbf{x}} q_\sigma(\tilde \mathbf{x}|\mathbf{x})\,d\tilde \mathbf{x}\Big]dx.$$

再用 $\nabla_{\tilde \mathbf{x}} q_\sigma(\tilde \mathbf{x}|\mathbf{x})=\nabla\log q_\sigma(\tilde \mathbf{x}|\mathbf{x})\cdot q_\sigma(\tilde \mathbf{x}|\mathbf{x})$ 代回：

$$=\int p(\mathbf{x})\Big[\int s_\theta^T\nabla\log q_\sigma(\tilde \mathbf{x}|\mathbf{x})\,q_\sigma(\tilde \mathbf{x}|\mathbf{x})\,d\tilde \mathbf{x}\Big]dx
=\mathbb{E}_{q_\sigma(\tilde \mathbf{x},\mathbf{x})}\big[s_\theta^T\nabla\log q_\sigma(\tilde \mathbf{x}|\mathbf{x})\big].$$

**ESM 的交叉项 = DSM 的交叉项！** 这就是整件事的命门——也再次印证了我们开头的那句直觉：对 $\mathbf{x}$ 平均后的边际得分，与被直接当做监督的条件得分，在期望意义上是同一回事。

---

## 第四步：得出等价性

$$\mathcal{J}_{\text{DSM}}(\theta)-\mathcal{J}_{\text{ESM}}(\theta)=C_2-C_1=C(\sigma),$$

其中

$$C(\sigma)=\frac{1}{2}\mathbb{E}_{q_\sigma(\tilde \mathbf{x},\mathbf{x})}\big[\|\nabla\log q_\sigma(\tilde \mathbf{x}|\mathbf{x})\|^2\big]-\frac{1}{2}\mathbb{E}_{q_\sigma(\tilde \mathbf{x})}\big[\|\nabla\log q_\sigma(\tilde \mathbf{x})\|^2\big].$$

$C(\sigma)$ 只依赖噪声水平 $\sigma$ 和数据分布 $p(\mathbf{x})$，与网络参数 $\theta$ 无关。因此 $\nabla_\theta\mathcal{J}_{\text{DSM}}=\nabla_\theta\mathcal{J}_{\text{ESM}}$——两目标梯度完全相同，优化轨迹一致。$\square$

---

## 补充说明

1. **为什么交叉项相等**：直觉上，边际得分 $\nabla\log q_\sigma(\tilde \mathbf{x})$ 是"对 $\mathbf{x}$ 平均之后"的得分；条件得分 $\nabla\log q_\sigma(\tilde \mathbf{x}|\mathbf{x})$ 是"给定 $\mathbf{x}$"的得分。对 $\mathbf{x}$ 取条件期望，恰好回到边际得分——这其实是 Tweedie 等式的一种推广。
2. **$C(\sigma)$ 的符号**：由 Jensen 不等式，边际得分范数 $\le$ 条件得分范数的期望，故 $C(\sigma)\ge0$。
3. **$\sigma\to0$ 的极限**：此时 $q_\sigma(\tilde \mathbf{x})\to p(\tilde \mathbf{x})$，$q_\sigma(\tilde \mathbf{x}|\mathbf{x})\to\delta(\tilde \mathbf{x}-\mathbf{x})$，$C(\sigma)\to0$，DSM 与 ESM 完全一致。这一极限也解释了 6.3 节"$\sigma$ 足够小时 $s_{\theta^*}\approx\nabla\log p$"的说法，并为 6.5 节"$\sigma$ 太小会有低密度区问题"埋下伏笔。

**来源**：Vincent (2011); Tutorial_Diffusion_Imaging_Vision Theorem 3.4