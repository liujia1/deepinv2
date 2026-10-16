# 附录9B 重参数化技巧的多元推广：高斯 KL 散度的闭式解

> 正文 9.3 用到了一个关键公式：对角高斯编码器 $q_\phi(z|x)$ 与标准正态先验 $p(z)=\mathcal{N}(0,I)$ 之间的 KL 散度有闭式解 $D_{\text{KL}}=\frac12\sum_j(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1)$。本附录把它从最一般的高斯 KL 一路推到这个简化式，并讲清楚每一项的含义和实现时的数值坑。

## 先建立直觉：KL 衡量"两个分布差多少"

KL 散度（Kullback–Leibler 散度）$D_{\text{KL}}(q\|p)$ 回答一个问题："如果我用 $q$ 当真相、$p$ 当近似，平均会多丢多少信息？"它不对称、且非负，等于 0 当且仅当两分布相同。VAE 里，我们希望编码器分布 $q_\phi(z|x)$ 别和先验 $p(z)$ 差太远——于是 KL 成了天然的"纪律罚分"。换句话说，这个闭式解不是一堆公式变形，而是把"解码器该忠实还原、同时编码器得守纪律"这一整场比赛，落地成一串能直接被反向传播求出梯度的解析项。

**本附录目标**：
- 从一般高斯 KL 公式推出对角高斯 vs 标准正态的简化式；
- 弄懂闭式解里每一项在"罚什么"；
- 记住实现时的数值稳定写法。

---

## 一般高斯分布间的 KL 散度

### 定理

设 $q=\mathcal{N}(\mu_0,\Sigma_0)$、$p=\mathcal{N}(\mu_1,\Sigma_1)$ 是 $d$ 维高斯，则：

$$D_{\text{KL}}(q \| p) = \frac{1}{2}\left[\text{tr}(\Sigma_1^{-1}\Sigma_0) - d + (\mu_1 - \mu_0)^\top \Sigma_1^{-1}(\mu_1 - \mu_0) + \ln\frac{\det\Sigma_1}{\det\Sigma_0}\right]$$

### 推导

KL 定义：

$$D_{\text{KL}}(q \| p) = \mathbb{E}_{q}[\log q(z) - \log p(z)]$$

高斯对数密度：

$$\log \mathcal{N}(z | \mu, \Sigma) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log\det\Sigma - \frac{1}{2}(z-\mu)^\top\Sigma^{-1}(z-\mu)$$

相减：

$$\log q(z) - \log p(z) = -\frac{1}{2}\log\frac{\det\Sigma_0}{\det\Sigma_1} - \frac{1}{2}(z-\mu_0)^\top\Sigma_0^{-1}(z-\mu_0) + \frac{1}{2}(z-\mu_1)^\top\Sigma_1^{-1}(z-\mu_1)$$

对 $z\sim q$ 取期望，逐项算：

**第一项**（与 $z$ 无关）：$-\frac{1}{2}\log\frac{\det\Sigma_0}{\det\Sigma_1}$。

**第二项**：$\mathbb{E}_q[(z-\mu_0)^\top\Sigma_0^{-1}(z-\mu_0)] = \text{tr}(\Sigma_0^{-1}\,\mathbb{E}[(z-\mu_0)(z-\mu_0)^\top]) = \text{tr}(\Sigma_0^{-1}\Sigma_0) = d$。

**第三项**：把 $z-\mu_1 = (z-\mu_0)+(\mu_0-\mu_1)$ 展开。交叉项因 $\mathbb{E}[z-\mu_0]=0$ 而消失，剩：

$$\mathbb{E}_q[(z-\mu_1)^\top\Sigma_1^{-1}(z-\mu_1)] = \text{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_0-\mu_1)^\top\Sigma_1^{-1}(\mu_0-\mu_1)$$

合并所有项并整理，得到上面的定理公式。$\square$

---

## VAE 中的简化情形：对角高斯 vs 标准正态

VAE 里 $q_\phi(z|x)=\mathcal{N}(\mu_\phi(x),\text{diag}(\sigma_\phi^2(x))$，$p(z)=\mathcal{N}(0,I)$。即 $\mu_0=\mu_\phi(x),\ \mu_1=0,\ \Sigma_0=\text{diag}(\sigma_\phi^2),\ \Sigma_1=I$。逐项代入定理：

- $\text{tr}(\Sigma_1^{-1}\Sigma_0)=\text{tr}(\text{diag}(\sigma_\phi^2))=\sum_j\sigma_j^2$
- $d = d_z$
- $(\mu_1-\mu_0)^\top\Sigma_1^{-1}(\mu_1-\mu_0)=\|\mu_\phi(x)\|^2=\sum_j\mu_j^2$
- $\ln\frac{\det\Sigma_1}{\det\Sigma_0}=-\ln\det(\text{diag}(\sigma_\phi^2))=-\sum_j\ln\sigma_j^2$

合并：

$$D_{\text{KL}}(q_\phi(z|x) \| p(z)) = \frac{1}{2}\sum_{j=1}^{d_z}\left[\sigma_j^2 - 1 + \mu_j^2 - \ln\sigma_j^2\right]$$

$$\boxed{D_{\text{KL}}(q_\phi(z|x) \| p(z)) = \frac{1}{2}\sum_{j=1}^{d_z}\left(\mu_j^2 + \sigma_j^2 - \ln\sigma_j^2 - 1\right)}$$

---

## 逐维度分解：为什么能一项项加

因为对角协方差假设，隐变量各维度独立，KL 可加：

$$D_{\text{KL},j} = \frac{1}{2}\left(\mu_j^2 + \sigma_j^2 - \ln\sigma_j^2 - 1\right)$$

| 项 | 含义（在罚什么） |
|---|---|
| $\mu_j^2$ | 均值偏离 0 的惩罚（"编码别太偏"） |
| $\sigma_j^2$ | 方差偏大（偏离 1）的惩罚（"编码别太散"） |
| $-\ln\sigma_j^2$ | 方差偏小（偏离 1）的惩罚（"编码也别太窄"） |
| $-1$ | 常数，保证 $\mu_j=0,\sigma_j^2=1$ 时 $D_{\text{KL},j}=0$ |

**验证**：当 $q_\phi(z_j|x)=\mathcal{N}(0,1)=p(z_j)$ 时，$\mu_j=0,\sigma_j^2=1$：

$$D_{\text{KL},j}=\frac12(0+1-0-1)=0 \quad \checkmark$$

---

## 数值稳定实现（实战必看）

实践里编码器输出的是 $\log\sigma_j^2$ 而非 $\sigma_j^2$（正文 9.1 讲过原因）。记 `logvar`$=\log\sigma_j^2$：

$$D_{\text{KL},j} = \frac{1}{2}\left(\mu_j^2 + e^{\text{logvar}_j} - \text{logvar}_j - 1\right)$$

PyTorch 实现：

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
1. 别先算 $\sigma_j^2=e^{\text{logvar}_j}$ 再算 $\sigma_j^2-\log\sigma_j^2$——`logvar` 很大或很小时可能溢出。上面直接用 `logvar.exp() - logvar` 是安全的。
2. `logvar` 很负（$\sigma_j^2\to0$）时，$e^{\text{logvar}_j}\to0$、$-\text{logvar}_j$ 很大，KL 趋于 $\infty$——这正确：方差趋 0 表示编码器把所有 $z$ 都压到一个确定值，也就完全偏离了先验"应该还是标准正态"的要求。
3. 实践中可对 `logvar` 做裁剪（如 $[-10,10]$）避免极端值。

## 收尾：为什么这件事重要

这个看似平凡的闭式解，是 VAE 能"便宜地"训练的关键：它把本该用蒙特卡罗积分去估的 KL 项，变成了一个不用采样的、可微的解析表达式，于是 KL 正则可以零成本塞进损失里。而把每一项拆开看（$\mu^2$ 罚偏、$\sigma^2$ 罚散、$-\log\sigma^2$ 罚窄），你就真正读懂了"KL 正则到底在雕刻隐空间的什么"——这正是 9.3 节那场重建-KL 拉扯的数学地基。

从更深层次看，这个闭式解揭示了先验与后验的一场"拔河"：先验 $p(z)=\mathcal{N}(0,I)$ 是一个固定的"守门员"，它不许编码器放肆地增大方差或偏离原点；而闭式解把这场拔河变成一行可微表达式，让深度生成模型能端到端地学着在"忠实重建"与"守纪律的隐空间"之间找到平衡点。这份"用解析可微的形式承载先验约束"的做法，也正是 VAE 作为学习型隐式先验区别于手工正则的关键所在。