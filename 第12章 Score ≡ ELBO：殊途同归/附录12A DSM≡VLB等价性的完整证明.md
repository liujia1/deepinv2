# 附录12A DSM ≡ VLB 等价性的完整证明

本附录给出 DSM 损失与变分下界等价性的完整证明。正文 12.4 节是这里的精简版，这里把每一步补细——如果你只想相信结论，跳过本附录完全不影响理解全书；如果你想亲手验证"两个目标真的只差一个常数"，下面四块把每一步钉死。

---

## A.1 前置结果：ESM 与 DSM 的等价性

**定理**（Vincent 2011）：对噪声扰动分布 $p_\sigma(\tilde{x}) = \int p(x)\mathcal{N}(\tilde{x}|x, \sigma^2 I)dx$，有

$$\mathcal{J}_\text{DSM}(\theta) = \mathcal{J}_\text{ESM}^{(\sigma)}(\theta) + C(\sigma)$$

其中：

$$\mathcal{J}_\text{ESM}^{(\sigma)}(\theta) = \frac{1}{2}\mathbb{E}_{p_\sigma(\tilde{x})}\left[\|s_\theta(\tilde{x}) - \nabla\log p_\sigma(\tilde{x})\|^2\right]$$

$$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\mathbb{E}_{p(x)}\mathbb{E}_\epsilon\left[\left\|s_\theta(x+\sigma\epsilon) - \left(-\frac{\epsilon}{\sigma}\right)\right\|^2\right]$$

$C(\sigma)$ 与 $\theta$ 无关。翻译一下：DSM 的最优解，同时就是含噪分布上 ESM 的最优解——我们不必知道真实得分，用加噪游戏就能学出真得分。

**证明概要**：

展开 $\mathcal{J}_\text{ESM}^{(\sigma)}$：

$$\mathcal{J}_\text{ESM}^{(\sigma)}(\theta) = \frac{1}{2}\mathbb{E}_{p_\sigma}\left[\|s_\theta\|^2\right] - \mathbb{E}_{p_\sigma}\left[s_\theta^\top\nabla\log p_\sigma\right] + \frac{1}{2}\mathbb{E}_{p_\sigma}\left[\|\nabla\log p_\sigma\|^2\right]$$

展开 $\mathcal{J}_\text{DSM}$：

$$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\mathbb{E}_{p_\sigma}\left[\|s_\theta\|^2\right] - \mathbb{E}_{p(x)}\mathbb{E}_\epsilon\left[s_\theta^\top\left(-\frac{\epsilon}{\sigma}\right)\right] + \frac{1}{2}\mathbb{E}_{p(x)}\mathbb{E}_\epsilon\left[\left\|\frac{\epsilon}{\sigma}\right\|^2\right]$$

关键在交叉项：利用 $\nabla_{\tilde{x}}\log q_\sigma(\tilde{x}|x) = -\epsilon/\sigma$ 和分部积分，可证

$$\mathbb{E}_{p_\sigma}\left[s_\theta^\top\nabla\log p_\sigma\right] = \mathbb{E}_{p(x)}\mathbb{E}_\epsilon\left[s_\theta^\top\left(-\frac{\epsilon}{\sigma}\right)\right]$$

两个目标的交叉项相等，差异只在常数项 $C(\sigma) = \frac{1}{2}\mathbb{E}_{p_\sigma}[\|\nabla\log p_\sigma\|^2] - \frac{d}{2\sigma^2}$。 $\square$

---

## A.2 VP-SDE 下的 ε 预测 VLB 展开

**设定**：VP-SDE 前向过程 $q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$，等价地 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$。

**VLB 分解**：

$$L_\text{VLB} = D_\text{KL}(q(x_T|x_0)\|p(x_T)) + \sum_{t=2}^T \mathbb{E}_q[D_\text{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))] - \mathbb{E}_q[\log p_\theta(x_0|x_1)]$$

**步骤 1：正向过程后验**

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(\tilde\mu_t, \tilde\beta_t I)$$

其中

$$\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$

$$\tilde\beta_t = \frac{(1-\bar\alpha_{t-1})\beta_t}{1-\bar\alpha_t}$$

**步骤 2：ε 预测参数化的后验均值**

利用 $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\epsilon)/\sqrt{\bar\alpha_t}$：

$$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right)$$

**步骤 3：逆向过程均值的 ε 预测参数化**

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t)\right)$$

**步骤 4：KL 散度化简**

由于 $q$ 和 $p_\theta$ 都是高斯且方差相同（$\tilde\beta_t I$），KL 散度退化为：

$$D_\text{KL}(q\|p_\theta) = \frac{1}{2\tilde\beta_t}\|\tilde\mu_t - \mu_\theta\|^2$$

代入步骤 2 和 3：

$$\tilde\mu_t - \mu_\theta = \frac{1}{\sqrt{\alpha_t}} \cdot \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}(\epsilon - \epsilon_\theta)$$

因此

$$D_\text{KL}(q\|p_\theta) = \frac{\beta_t^2}{2\tilde\beta_t\alpha_t(1-\bar\alpha_t)}\|\epsilon - \epsilon_\theta\|^2$$

**步骤 5：合并**

$$L_\text{VLB}(\theta) = \sum_{t=1}^T \frac{\beta_t^2}{2\tilde\beta_t\alpha_t(1-\bar\alpha_t)}\,\mathbb{E}_{x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right] + C$$

$C$ 含 $L_T$（常数）及各时间步常数项。

---

## A.3 DSM 在 VP-SDE 记号下的展开

**多尺度 DSM 的 ε 预测形式**（VE-SDE 设定）：

$$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\sum_{i=1}^L \frac{\lambda(\sigma_i)}{\sigma_i^2}\,\mathbb{E}_{x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_0+\sigma_i\epsilon, \sigma_i)\|^2\right]$$

**转换到 VP-SDE 记号**：令 $\sigma_i \leftrightarrow \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$，含噪输入统一为 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$：

$$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\sum_{t=1}^T \frac{\lambda(\sigma_t)}{\sigma_t^2}\,\mathbb{E}_{x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

---

## A.4 权重匹配与等价性

**比较 VLB 和 DSM**：

$$L_\text{VLB}(\theta) = \sum_{t=1}^T \underbrace{\frac{\beta_t^2}{2\tilde\beta_t\alpha_t(1-\bar\alpha_t)}}_{w_t}\,\mathbb{E}\left[\|\epsilon - \epsilon_\theta\|^2\right] + C$$

$$\mathcal{J}_\text{DSM}(\theta) = \sum_{t=1}^T \underbrace{\frac{\lambda(\sigma_t)}{2\sigma_t^2}}_{\text{DSM 权重}}\,\mathbb{E}\left[\|\epsilon - \epsilon_\theta\|^2\right]$$

**令两者权重相等**：

$$\frac{\lambda(\sigma_t)}{2\sigma_t^2} = \frac{\beta_t^2}{2\tilde\beta_t\alpha_t(1-\bar\alpha_t)}$$

解出：

$$\lambda(\sigma_t) = \sigma_t^2 \cdot \frac{\beta_t^2}{\tilde\beta_t\alpha_t(1-\bar\alpha_t)}$$

把 $\sigma_t^2 = (1-\bar\alpha_t)/\bar\alpha_t$ 和 $\tilde\beta_t = (1-\bar\alpha_{t-1})\beta_t/(1-\bar\alpha_t)$ 代入：

$$\lambda(\sigma_t) = \frac{(1-\bar\alpha_t)}{\bar\alpha_t} \cdot \frac{\beta_t^2}{\frac{(1-\bar\alpha_{t-1})\beta_t}{1-\bar\alpha_t} \cdot \alpha_t \cdot (1-\bar\alpha_t)} = \frac{(1-\bar\alpha_t)^2\beta_t}{\bar\alpha_t(1-\bar\alpha_{t-1})\alpha_t(1-\bar\alpha_t)}$$

利用 $1-\bar\alpha_{t-1} = 1 - \bar\alpha_t/\alpha_t = (\alpha_t - \bar\alpha_t)/\alpha_t$：

$$\lambda(\sigma_t) = \frac{(1-\bar\alpha_t)\beta_t}{\bar\alpha_t(\alpha_t - \bar\alpha_t)} = \frac{(1-\bar\alpha_t)\beta_t}{\bar\alpha_t\alpha_t(1-\bar\alpha_t/\alpha_t)}$$

进一步用 $\beta_t = 1-\alpha_t$ 和 $1-\bar\alpha_t/\alpha_t = (1-\bar\alpha_{t-1})$：

$$\lambda(\sigma_t) = \frac{(1-\bar\alpha_t)(1-\alpha_t)}{\bar\alpha_t(1-\bar\alpha_{t-1})}$$

在此权重选择下：

$$\boxed{\mathcal{J}_\text{DSM}(\theta) = L_\text{VLB}(\theta) + \text{const}}$$

两个目标的最优解完全相同。 $\blacksquare$

---

## A.5 简化目标的等价性

当 $\lambda(\sigma) = \sigma^2$ 时，DSM 权重：

$$\frac{\lambda(\sigma_t)}{2\sigma_t^2} = \frac{\sigma_t^2}{2\sigma_t^2} = \frac{1}{2}$$

简化 VLB 的权重为 1（或 $1/T$，取决于采样方式）。两者仅差缩放因子，最优解一致。

这对应 DDPM 论文 Eq. (14) 与 NCSN 训练目标的等价性——Ho et al. (2020) 的核心发现之一。
