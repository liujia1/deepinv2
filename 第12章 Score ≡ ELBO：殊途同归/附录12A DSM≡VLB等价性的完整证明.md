# 附录12A DSM ≡ VLB等价性的完整证明

本附录给出DSM损失与变分下界等价性的完整证明。证明分为四个部分：从ESM等价性出发，经过ε预测参数化转换，逐步展开VLB各项，最后进行权重匹配。

## A.1 前置结果：ESM与DSM的等价性

**定理**（Vincent 2011）：对噪声扰动分布 $p_\sigma(\tilde{x}) = \int p(x)\mathcal{N}(\tilde{x}|x, \sigma^2 I)dx$，有

$$\mathcal{J}_\text{DSM}(\theta) = \mathcal{J}_\text{ESM}^{(\sigma)}(\theta) + C(\sigma)$$

其中：

$$\mathcal{J}_\text{ESM}^{(\sigma)}(\theta) = \frac{1}{2}\mathbb{E}_{p_\sigma(\tilde{x})}\left[\|s_\theta(\tilde{x}) - \nabla\log p_\sigma(\tilde{x})\|^2\right]$$

$$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\mathbb{E}_{p(x)}\mathbb{E}_\epsilon\left[\left\|s_\theta(x+\sigma\epsilon) - \left(-\frac{\epsilon}{\sigma}\right)\right\|^2\right]$$

$C(\sigma)$ 与 $\theta$ 无关。

**证明概要**：

展开 $\mathcal{J}_\text{ESM}^{(\sigma)}$：

$$\mathcal{J}_\text{ESM}^{(\sigma)}(\theta) = \frac{1}{2}\mathbb{E}_{p_\sigma}\left[\|s_\theta\|^2\right] - \mathbb{E}_{p_\sigma}\left[s_\theta^\top\nabla\log p_\sigma\right] + \frac{1}{2}\mathbb{E}_{p_\sigma}\left[\|\nabla\log p_\sigma\|^2\right]$$

展开 $\mathcal{J}_\text{DSM}$：

$$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\mathbb{E}_{p_\sigma}\left[\|s_\theta\|^2\right] - \mathbb{E}_{p(x)}\mathbb{E}_\epsilon\left[s_\theta^\top\left(-\frac{\epsilon}{\sigma}\right)\right] + \frac{1}{2}\mathbb{E}_{p(x)}\mathbb{E}_\epsilon\left[\left\|\frac{\epsilon}{\sigma}\right\|^2\right]$$

关键是交叉项：利用 $\nabla_{\tilde{x}}\log q_\sigma(\tilde{x}|x) = -\epsilon/\sigma$ 和分部积分，可以证明：

$$\mathbb{E}_{p_\sigma}\left[s_\theta^\top\nabla\log p_\sigma\right] = \mathbb{E}_{p(x)}\mathbb{E}_\epsilon\left[s_\theta^\top\left(-\frac{\epsilon}{\sigma}\right)\right]$$

因此两个目标的交叉项相等，差异仅在常数项 $C(\sigma) = \frac{1}{2}\mathbb{E}_{p_\sigma}[\|\nabla\log p_\sigma\|^2] - \frac{d}{2\sigma^2}$。 $\square$

## A.2 VP-SDE下的ε预测VLB展开

**设定**：VP-SDE前向过程 $q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$，等价地 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$。

**VLB分解**：

$$L_\text{VLB} = D_\text{KL}(q(x_T|x_0)\|p(x_T)) + \sum_{t=2}^T \mathbb{E}_q[D_\text{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))] - \mathbb{E}_q[\log p_\theta(x_0|x_1)]$$

**步骤1：正向过程后验**

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(\tilde\mu_t, \tilde\beta_t I)$$

其中：

$$\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$

$$\tilde\beta_t = \frac{(1-\bar\alpha_{t-1})\beta_t}{1-\bar\alpha_t}$$

**步骤2：ε预测参数化的后验均值**

利用 $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\epsilon)/\sqrt{\bar\alpha_t}$：

$$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right)$$

**步骤3：逆向过程均值的ε预测参数化**

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t)\right)$$

**步骤4：KL散度化简**

由于 $q$ 和 $p_\theta$ 都是高斯且方差相同（$\tilde\beta_t I$），KL散度退化为：

$$D_\text{KL}(q\|p_\theta) = \frac{1}{2\tilde\beta_t}\|\tilde\mu_t - \mu_\theta\|^2$$

代入步骤2和步骤3：

$$\tilde\mu_t - \mu_\theta = \frac{1}{\sqrt{\alpha_t}} \cdot \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}(\epsilon - \epsilon_\theta)$$

因此：

$$D_\text{KL}(q\|p_\theta) = \frac{\beta_t^2}{2\tilde\beta_t\alpha_t(1-\bar\alpha_t)}\|\epsilon - \epsilon_\theta\|^2$$

**步骤5：合并**

$$L_\text{VLB}(\theta) = \sum_{t=1}^T \frac{\beta_t^2}{2\tilde\beta_t\alpha_t(1-\bar\alpha_t)}\,\mathbb{E}_{x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right] + C$$

其中 $C$ 包含 $L_T$（常数）和各时间步的常数项。

## A.3 DSM在VP-SDE记号下的展开

**多尺度DSM的ε预测形式**（VE-SDE设定）：

$$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\sum_{i=1}^L \frac{\lambda(\sigma_i)}{\sigma_i^2}\,\mathbb{E}_{x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_0+\sigma_i\epsilon, \sigma_i)\|^2\right]$$

**转换到VP-SDE记号**：令 $\sigma_i \leftrightarrow \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$，含噪输入统一为 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$：

$$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\sum_{t=1}^T \frac{\lambda(\sigma_t)}{\sigma_t^2}\,\mathbb{E}_{x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

## A.4 权重匹配与等价性

**比较VLB和DSM**：

$$L_\text{VLB}(\theta) = \sum_{t=1}^T \underbrace{\frac{\beta_t^2}{2\tilde\beta_t\alpha_t(1-\bar\alpha_t)}}_{w_t}\,\mathbb{E}\left[\|\epsilon - \epsilon_\theta\|^2\right] + C$$

$$\mathcal{J}_\text{DSM}(\theta) = \sum_{t=1}^T \underbrace{\frac{\lambda(\sigma_t)}{2\sigma_t^2}}_{\text{DSM权重}}\,\mathbb{E}\left[\|\epsilon - \epsilon_\theta\|^2\right]$$

**令两者权重相等**：

$$\frac{\lambda(\sigma_t)}{2\sigma_t^2} = \frac{\beta_t^2}{2\tilde\beta_t\alpha_t(1-\bar\alpha_t)}$$

解出：

$$\lambda(\sigma_t) = \sigma_t^2 \cdot \frac{\beta_t^2}{\tilde\beta_t\alpha_t(1-\bar\alpha_t)}$$

将 $\sigma_t^2 = (1-\bar\alpha_t)/\bar\alpha_t$ 和 $\tilde\beta_t = (1-\bar\alpha_{t-1})\beta_t/(1-\bar\alpha_t)$ 代入：

$$\lambda(\sigma_t) = \frac{(1-\bar\alpha_t)}{\bar\alpha_t} \cdot \frac{\beta_t^2}{\frac{(1-\bar\alpha_{t-1})\beta_t}{1-\bar\alpha_t} \cdot \alpha_t \cdot (1-\bar\alpha_t)} = \frac{(1-\bar\alpha_t)^2\beta_t}{\bar\alpha_t(1-\bar\alpha_{t-1})\alpha_t(1-\bar\alpha_t)}$$

利用 $1-\bar\alpha_{t-1} = 1 - \bar\alpha_t/\alpha_t = (\alpha_t - \bar\alpha_t)/\alpha_t$：

$$\lambda(\sigma_t) = \frac{(1-\bar\alpha_t)\beta_t}{\bar\alpha_t(\alpha_t - \bar\alpha_t)} = \frac{(1-\bar\alpha_t)\beta_t}{\bar\alpha_t\alpha_t(1-\bar\alpha_t/\alpha_t)}$$

进一步利用 $\beta_t = 1-\alpha_t$ 和 $1-\bar\alpha_t/\alpha_t = (\alpha_t - \bar\alpha_t)/\alpha_t = (1-\bar\alpha_{t-1})$：

$$\lambda(\sigma_t) = \frac{(1-\bar\alpha_t)(1-\alpha_t)}{\bar\alpha_t(1-\bar\alpha_{t-1})}$$

在此权重选择下：

$$\boxed{\mathcal{J}_\text{DSM}(\theta) = L_\text{VLB}(\theta) + \text{const}}$$

两个目标的最优解完全相同。 $\blacksquare$

## A.5 简化目标的等价性

当 $\lambda(\sigma) = \sigma^2$ 时，DSM权重：

$$\frac{\lambda(\sigma_t)}{2\sigma_t^2} = \frac{\sigma_t^2}{2\sigma_t^2} = \frac{1}{2}$$

简化VLB的权重为1（或 $1/T$，取决于采样方式）。两者仅差缩放因子，最优解一致。

这对应DDPM论文Eq. (14) 与NCSN训练目标的等价性——Ho et al. (2020) 的核心发现之一。
