# 附录 11B 连续时间 VLB 与 VDM

> 到现在为止，我们的扩散过程都是"离散的"——固定 $T$ 步，一步一步走。但你想过没有：如果步数 $T$ 趋向无穷、每步趋近于零，会发生什么？答案是 VLB 变成一条积分曲线，而我们一直当"超参数"死定死的噪声调度 $\beta_t$，居然也能拿来一起训练。这一节快速补上这个连续时间视角（Kingma 2021 的 VDM）。

## 从离散 VLB 到连续 VLB

第10–11章的离散时间 VLB：

$$L_\text{VLB} = \underbrace{D_\text{KL}(q(x_T|x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^T \underbrace{\mathbb{E}_q[D_\text{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))]}_{L_{t-1}} - \underbrace{\mathbb{E}_q[\log p_\theta(x_0|x_1)]}_{\text{负}L_0}$$

当 $T\to\infty$、$\Delta t=1/T\to 0$，求和趋于积分。在连续时间 $t\in[0,1]$ 下：

$$L_\text{VLB}^\text{cont} = \underbrace{D_\text{KL}(q(x_1|x_0) \| p(x_1))}_{L_1} + \int_0^1 \mathbb{E}_q\left[\frac{1}{2}\left\|\frac{\mu_\theta(x_t,t) - \tilde\mu_t(x_t,x_0)}{\sigma_t}\right\|^2\right]dt - \underbrace{\mathbb{E}_q[\log p_\theta(x_0|x_{\epsilon})]}_{L_0}$$

其中 $x_\epsilon$ 是 $t\to 0^+$ 时的含噪状态。形式没变，只是"求和"换成了"沿时间的积分"。

## VDM 的核心贡献：噪声调度也能学

**关键洞察**：噪声调度 $\beta(t)$（每一步掺多少噪声）不必是预设超参——它可以是变分参数，和 $\theta$ 一起用梯度下降优化。

离散 DDPM 里 $\beta_1,\dots,\beta_T$ 是手动选的（线性或余弦），对效果影响大却要调参。VDM 把 $\beta(t)$ 参数化成可微函数（如单调网络），同时优化 $\theta$ 和 $\beta(t)$。

VDM 发现最优调度高度非线性：更多步分配给"信噪比变化剧烈"的困难区，少步给"变化慢"的容易区——视觉上更接近余弦调度，印证了 Nichol & Dhariwal (2021) 的经验观察。

## 与 SDE 视角的对照

第7章 SDE 框架里，漂移系数 $f$ 和扩散系数 $g$ 是固定选择（VE 或 VP）。VDM 把它们当可优化对象：

- **SDE 视角**：固定正向过程 $f,g$，学得分 $s_\theta$；
- **VDM 视角**：同时优化正向过程（噪声调度）和逆向过程（网络参数）。

而且 VDM 证明：连续时间下三种参数化（噪声/ x₀ / 得分预测）仍等价，且 ε 预测 VLB 的最优权重恰好等于 DSM 的最优权重 $\lambda(\sigma)$——再次强化第12章等价性的结论。

> **与第12章的关系**：此处"连续时间 VLB 权重 ≡ DSM 权重"正是第 12 章 12.5 节在连续时间视角下设的统一结论，其逐项推导在附录 12B。本节强调"VDM 还能连噪声调度一起训练"这一额外收获，不重复等价性证明——完整证明见 12.4 与附录 12B。

## 实践意义与局限

**意义**：自适应噪声调度免手动调参，不同数据集自动找到合适调度；连续 VLB 可直接优化数据似然（压缩、异常检测有用）；VDM 统一了 DDPM（固定调度+简化 VLB）、Score-SDE（固定 SDE+得分匹配）、VDM（可优调度+连续 VLB）三者，后两者只是 VDM 加约束的特例。

**局限**：同时优化 $\theta$ 和调度，计算比固定调度的 DDPM 贵；调度优化可能过拟合，需正则化；连续时间理论分析更复杂。

**来源**：Kingma et al. (2021) VDM; Kingma & Gao (2023) Understanding Diffusion Objectives; 2406.08929v2 Sec 2.3
