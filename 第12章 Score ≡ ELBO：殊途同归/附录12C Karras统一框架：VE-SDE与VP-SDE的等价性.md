# 附录12C Karras et al. (2022) 统一框架：VE-SDE与VP-SDE的等价性

本附录介绍Karras et al. (2022) 提出的统一参数化框架，展示VE-SDE和VP-SDE是该框架的不同参数化选择，从而在更基础的层面上理解DSM ≡ VLB等价性为何不依赖于特定的SDE形式。

## C.1 统一参数化

### 一般加噪过程

Karras et al. (2022) 将扩散模型的加噪过程统一为：

$$x_t = s(t)\,x_0 + s(t)\,\sigma(t)\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中：
- $s(t)$：信号缩放因子（$s(t) > 0$）
- $\sigma(t)$：噪声水平（$\sigma(t) > 0$，关于 $t$ 单调递增）

边际分布为：

$$q(x_t|x_0) = \mathcal{N}(x_t | s(t)\,x_0,\, s^2(t)\sigma^2(t)\,I)$$

### VE-SDE和VP-SDE作为特例

**VE-SDE（NCSN/SMLD）**：

$$s(t) = 1, \quad \sigma(t) = \sigma_{\min}\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t$$

加噪过程：$x_t = x_0 + \sigma(t)\epsilon$

信号幅度不变，噪声叠加增长。

**VP-SDE（DDPM）**：

$$s(t) = \sqrt{\bar\alpha(t)}, \quad \sigma(t) = \sqrt{\frac{1-\bar\alpha(t)}{\bar\alpha(t)}}$$

加噪过程：$x_t = \sqrt{\bar\alpha(t)}\,x_0 + \sqrt{1-\bar\alpha(t)}\,\epsilon$

信号逐渐衰减，噪声相对增长。

### 等价的推理过程

Karras et al. (2022) 的关键观察是：**推理过程（从 $x_t$ 估计 $x_0$）仅取决于信噪比 $\text{SNR}(t) = 1/\sigma^2(t)$，而不取决于信号缩放 $s(t)$**。

由Tweedie等式，MMSE估计为：

$$\hat{x}_0(x_t) = \mathbb{E}[x_0|x_t] = \frac{x_t + \sigma^2(t)\nabla_{x_t}\log p_t(x_t)}{s(t)}$$

去噪器的输出（预测 $x_0$）仅通过 $1/s(t)$ 受信号缩放影响——这是简单的归一化，不改变估计的本质。

## C.2 统一框架下的Tweedie等式

在统一参数化下，Tweedie等式的一般形式为：

$$\nabla_{x_t}\log p_t(x_t) = \frac{s(t)\,\mathbb{E}[x_0|x_t] - x_t}{s^2(t)\sigma^2(t)}$$

即：

$$\nabla_{x_t}\log p_t(x_t) = \frac{\hat{x}_0(x_t) - x_t/s(t)}{s(t)\sigma^2(t)}$$

在ε预测参数化下（$\hat{x}_0 = (x_t - s(t)\sigma(t)\epsilon_\theta)/(s(t))$）：

$$\nabla_{x_t}\log p_t(x_t) = -\frac{\epsilon_\theta(x_t, t)}{s(t)\sigma(t)}$$

### VE-SDE特例

$s(t)=1$：$\nabla\log p_t(x_t) = -\epsilon_\theta/\sigma(t)$

### VP-SDE特例

$s(t)=\sqrt{\bar\alpha_t}$，$s(t)\sigma(t)=\sqrt{1-\bar\alpha_t}$：

$$\nabla\log p_t(x_t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}$$

## C.3 统一框架下的训练目标

### 统一DSM目标

在统一参数化下，DSM目标为：

$$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\int \frac{\lambda(t)}{s^2(t)\sigma^2(t)}\,\mathbb{E}_{x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]dt$$

### 统一VLB目标

连续时间VLB的一致性积分在统一参数化下为：

$$L_\text{VLB}(\theta) = \int w(t)\,\mathbb{E}_{x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]dt$$

其中 $w(t)$ 由 $(s(t), \sigma(t))$ 的动力学方程确定。

### 等价性不受参数化影响

由于两种SDE形式（VE和VP）仅是统一框架的不同参数化选择，且推理过程仅取决于SNR $\sigma(t)$，DSM ≡ VLB的等价性不依赖于SDE的具体形式——它在统一框架下自然成立。

选择VE-SDE还是VP-SDE（或其他参数化）只影响权重 $w(t)$ 和 $\lambda(t)/\sigma^2(t)$ 的具体表达式，不影响等价性本身。这进一步验证了等价性的必然性——它不是某种特定参数化的巧合，而是概率分布"梯度信息"与"函数值信息"等价性的体现。

## C.4 从VE-SDE到VP-SDE的权重映射

给定VE-SDE下的权重 $\lambda_\text{VE}(\sigma)$，可以通过SNR对应映射到VP-SDE下的权重：

1. 建立SNR对应：$\sigma_\text{VE} \leftrightarrow \sigma_\text{VP}(t) = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$
2. 将 $\lambda_\text{VE}(\sigma)$ 中的 $\sigma$ 替换为 $\sigma_\text{VP}(t)$
3. 考虑信号缩放因子 $s(t)$ 的影响

当 $\lambda_\text{VE}(\sigma) = \sigma^2$（NCSN默认）时，在VP-SDE下对应 $\lambda_\text{VP}(t) = \sigma_\text{VP}^2(t) = (1-\bar\alpha_t)/\bar\alpha_t$，此时DSM权重为：

$$\frac{\lambda_\text{VP}(t)}{s^2(t)\sigma_\text{VP}^2(t)} = \frac{(1-\bar\alpha_t)/\bar\alpha_t}{\bar\alpha_t \cdot (1-\bar\alpha_t)/\bar\alpha_t} = \frac{1}{\bar\alpha_t}$$

这与VP-SDE下简化VLB的权重一致（均匀权重，仅差 $\bar\alpha_t$ 的缩放，当 $\bar\alpha_t \approx 1$ 时近似为1）。
