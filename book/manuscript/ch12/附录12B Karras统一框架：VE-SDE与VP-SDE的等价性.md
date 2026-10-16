# 附录12B Karras et al. (2022) 统一框架：VE-SDE 与 VP-SDE 的等价性

本附录介绍 Karras et al. (2022) 的统一参数化框架，展示 VE-SDE 和 VP-SDE 是该框架的不同参数化选择，从而在更基础的层面上说明：DSM ≡ VLB 的等价性不依赖于特定的 SDE 形式。正文 12.3 节观察 2 和 12.5 节都用到了它。一句话定位：如果说 12.4 证明了"两条路径在同一坐标系下等价"，那么本附录展示的是——换不换坐标系都不影响这个等价。

---

## B.1 统一参数化

### 一般加噪过程

Karras et al. (2022) 把扩散模型的加噪过程统一为：

$$x_t = s(t)\,x_0 + s(t)\,\sigma(t)\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中：
- $s(t)$：信号缩放因子（$s(t) > 0$）
- $\sigma(t)$：噪声水平（$\sigma(t) > 0$，关于 $t$ 单调递增）

边际分布为：

$$q(x_t|x_0) = \mathcal{N}(x_t | s(t)\,x_0,\, s^2(t)\sigma^2(t)\,I)$$

### VE-SDE 和 VP-SDE 作为特例

**VE-SDE（NCSN/SMLD）**：

$$s(t) = 1, \quad \sigma(t) = \sigma_{\min}\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t$$

加噪过程：$x_t = x_0 + \sigma(t)\epsilon$——信号幅度不变，噪声叠加增长。

**VP-SDE（DDPM）**：

$$s(t) = \sqrt{\bar\alpha(t)}, \quad \sigma(t) = \sqrt{\frac{1-\bar\alpha(t)}{\bar\alpha(t)}}$$

加噪过程：$x_t = \sqrt{\bar\alpha(t)}\,x_0 + \sqrt{1-\bar\alpha(t)}\,\epsilon$——信号逐渐衰减，噪声相对增长。

### 等价的推理过程

Karras et al. (2022) 的关键观察：**推理过程（从 $x_t$ 估计 $x_0$）只取决于信噪比 $\text{SNR}(t) = 1/\sigma^2(t)$，而不取决于信号缩放 $s(t)$**。

由 Tweedie 等式，MMSE 估计为：

$$\hat{x}_0(x_t) = \mathbb{E}[x_0|x_t] = \frac{x_t + \sigma^2(t)\nabla_{x_t}\log p_t(x_t)}{s(t)}$$

去噪器的输出（预测 $x_0$）只通过 $1/s(t)$ 受信号缩放影响——这是简单归一化，不改变估计本质。

> **从更深层次看**，这一观察把"哪种加噪方案更好"这个工程问题彻底降维了：只要信噪比曲线相同，再怎么换信号缩放 $s(t)$，模型推理的信息都不变。这一转变意味着，VE 与 VP 的差别最终只是一个"读数单位"问题，而非本质差别。

---

## B.2 统一框架下的 Tweedie 等式

在统一参数化下，Tweedie 等式的一般形式为：

$$\nabla_{x_t}\log p_t(x_t) = \frac{s(t)\,\mathbb{E}[x_0|x_t] - x_t}{s^2(t)\sigma^2(t)} = \frac{\hat{x}_0(x_t) - x_t/s(t)}{s(t)\sigma^2(t)}$$

在 ε 预测参数化下（$\hat{x}_0 = (x_t - s(t)\sigma(t)\epsilon_\theta)/(s(t))$）：

$$\nabla_{x_t}\log p_t(x_t) = -\frac{\epsilon_\theta(x_t, t)}{s(t)\sigma(t)}$$

### VE-SDE 特例
$s(t)=1$：$\nabla\log p_t(x_t) = -\epsilon_\theta/\sigma(t)$

### VP-SDE 特例
$s(t)=\sqrt{\bar\alpha_t}$，$s(t)\sigma(t)=\sqrt{1-\bar\alpha_t}$：
$$\nabla\log p_t(x_t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}$$

---

## B.3 统一框架下的训练目标

### 统一 DSM 目标

$$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\int \frac{\lambda(t)}{s^2(t)\sigma^2(t)}\,\mathbb{E}_{x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]dt$$

注意分母 $s^2(t)\sigma^2(t)$ 正是扰动方差（B.1 中 $q(x_t|x_0)$ 的方差），它统一了 12.4 里"VE 用 $\sigma^2$、VP 用 $1-\bar\alpha$"两种分母——因为 VP 下 $s^2\sigma^2 = \bar\alpha \cdot (1-\bar\alpha)/\bar\alpha = 1-\bar\alpha$。

### 统一 VLB 目标

连续时间 VLB 的一致性积分在统一参数化下为：

$$L_\text{VLB}(\theta) = \int w(t)\,\mathbb{E}_{x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]dt$$

$w(t)$ 由 $(s(t), \sigma(t))$ 的动力学方程确定。

### 等价性不受参数化影响

由于两种 SDE 形式（VE 和 VP）只是统一框架的不同参数化选择，且推理过程只取决于 SNR $\sigma(t)$，DSM ≡ VLB 的等价性不依赖于 SDE 的具体形式——它在统一框架下自然成立。选 VE-SDE 还是 VP-SDE（或其他参数化）只影响权重 $w(t)$ 和 $\lambda(t)/\sigma^2(t)$ 的具体表达式，不影响等价性本身。这进一步验证等价性的必然性——它不是某种特定参数化的巧合，而是概率分布"梯度信息"与"函数值信息"等价性的体现。

> **这一转变意味着**：12.4 在 VP-SDE 下证明的等价性，本可以放进任何参数化下重演一遍。换句话说，等价性不是"VP-SDE 的专利"，而是"得分函数与似然下界是同一对象两面"这一更基本事实的必然推论——具体用哪套记号，只是这一事实的一次显影。

---

## B.4 从 VE-SDE 到 VP-SDE 的权重映射

给定 VE-SDE 下的权重 $\lambda_\text{VE}(\sigma)$，可通过 SNR 对应映射到 VP-SDE 下的权重：

1. 建立 SNR 对应：$\sigma_\text{VE} \leftrightarrow \sigma_\text{VP}(t) = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$
2. 把 $\lambda_\text{VE}(\sigma)$ 里的 $\sigma$ 替换成 $\sigma_\text{VP}(t)$
3. 考虑信号缩放因子 $s(t)$ 的影响

当 $\lambda_\text{VE}(\sigma) = \sigma^2$（NCSN 默认）时，在 VP-SDE 下对应 $\lambda_\text{VP}(t) = \sigma_\text{VP}^2(t) = (1-\bar\alpha_t)/\bar\alpha_t$，此时 DSM 权重为：

$$\frac{\lambda_\text{VP}(t)}{s^2(t)\sigma_\text{VP}^2(t)} = \frac{(1-\bar\alpha_t)/\bar\alpha_t}{\bar\alpha_t \cdot (1-\bar\alpha_t)/\bar\alpha_t} = \frac{1}{\bar\alpha_t}$$

这与 VP-SDE 下简化 VLB 的权重一致（均匀权重，仅差 $\bar\alpha_t$ 的缩放，当 $\bar\alpha_t \approx 1$ 时近似为 1）。