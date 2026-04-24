# 附录4A 马尔可夫链收敛性与Gibbs分布

> 本附录补充4.2节和4.5节所需的马尔可夫链理论基础，以及Gibbs分布的数学性质。

## A4.1 马尔可夫链的收敛性

### 遍历定理

MCMC方法的理论基础是马尔可夫链的**遍历定理**（ergodic theorem）：

**定理**（遍历定理）：设 $(X_m)_{m \geq 1}$ 是以 $\mu$ 为平稳分布的不可约、非周期、Harris常返马尔可夫链，则对任意可积函数 $h$：

$$\frac{1}{M}\sum_{m=1}^M h(X_m) \xrightarrow{\text{a.s.}} \int h(x) \, \mu(\mathrm{d}x), \quad M \to \infty$$

这个定理是MCMC方法的理论保证：即使样本不是独立的，样本均值仍然收敛到期望值。

### 不可约性、非周期性与常返性

遍历定理的三个条件：

1. **不可约性**（irreducibility）：链可以从任何状态到达任何其他状态（以正概率）——保证链能遍历整个支撑集
2. **非周期性**（aperiodicity）：链不会在几个状态子集之间周期性振荡——保证收敛到平稳分布（非周期振荡）
3. **Harris常返性**（Harris recurrence）：链会无限次访问每个正概率区域——保证大数定律成立

对MH算法和ULA，在温和条件下这三个性质都满足。

### 收敛速率

遍历定理保证了收敛，但不给出速率。收敛速率取决于谱间隙（spectral gap）：

- **几何遍历**（geometrically ergodic）：$\|K^m(x, \cdot) - \mu\|_{\text{TV}} \leq C(x)r^m$，$r < 1$——最快的实用收敛速率
- **多项式遍历**：收敛速率为 $O(1/m^\alpha)$——慢收敛

ULA在强对数凹条件下是几何遍历的，收敛速率为 $O(e^{-cm})$。

---

## A4.2 细致平衡与可逆性

### 可逆性

细致平衡条件（4.2节）等价于马尔可夫核 $K$ 关于 $\mu$ **可逆**（reversible）：

$$\int_A K(B|x) \, \mu(\mathrm{d}x) = \int_B K(A|x) \, \mu(\mathrm{d}x), \quad \forall A, B$$

可逆性是一个比平稳性更强的条件：

$$\text{可逆性} \implies \text{平稳性} \implies \text{遍历性（在不可约非周期条件下）}$$

### 不可逆MCMC

不满足细致平衡但仍有正确平稳分布的MCMC算法称为**不可逆MCMC**。理论上，不可逆MCMC可以有更小的自相关和更快的收敛——因为打破细致平衡允许"定向流"。

ILA（4.6节）在一定意义上是"部分不可逆"的——动量项引入了非对称的转移方向。欠阻尼Langevin SDE的连续极限也不满足细致平衡（它在 $(x, v)$ 空间上有定向流），但仍有正确的平稳分布。

---

## A4.3 Gibbs分布与指数族

### Gibbs分布

**Gibbs分布**（又称Boltzmann分布）是统计物理和概率论中的基本分布族：

$$p(x) = \frac{1}{Z} \exp(-E(x))$$

其中 $E(x)$ 是**能量函数**，$Z = \int \exp(-E(x)) \mathrm{d}x$ 是配分函数（归一化常数）。

贝叶斯后验是Gibbs分布的特例：$E(x) = -\log p(y|x) - \log p(x)$，$Z = p(y)$。

### 高斯Gibbs分布

当能量函数为二次型 $E(x) = \frac{1}{2}(x - \mu)^\top Q(x - \mu)$ 时，Gibbs分布为高斯：

$$p(x) = \mathcal{N}(x \mid \mu, Q^{-1})$$

其中 $Q$ 是精度矩阵（precision matrix，协方差矩阵的逆）。

高斯Gibbs分布的条件分布也是高斯——这是Gibbs采样在高斯模型上高效的根本原因。

### 二元高斯的条件分布

设 $x = (x_1, x_2) \sim \mathcal{N}(\mu, Q^{-1})$，精度矩阵 $Q = \begin{pmatrix} Q_{11} & Q_{12} \\ Q_{21} & Q_{22} \end{pmatrix}$，则：

$$p(x_1 | x_2) = \mathcal{N}(x_1 \mid \mu_{1|2}, \, Q_{11}^{-1})$$

其中条件均值和方差为：

$$\mu_{1|2} = \mu_1 - Q_{11}^{-1}Q_{12}(x_2 - \mu_2), \quad \Sigma_{1|2} = Q_{11}^{-1}$$

注意：条件方差 $Q_{11}^{-1}$ 不依赖 $x_2$——这是高斯分布的特殊性质，使得Gibbs采样中的条件分布容易计算。

---

## A4.4 广义逆高斯分布（GIG）

Gibbs采样在TV先验下需要从GIG分布中采样（4.5节）。这里给出GIG的定义和采样方法。

### 定义

**广义逆高斯分布**（Generalized Inverse Gaussian, GIG）的概率密度为：

$$p(z) = \frac{(a/b)^{p/2}}{2K_p(\sqrt{ab})} z^{p-1} \exp\left(-\frac{1}{2}(az + b/z)\right), \quad z > 0$$

其中 $K_p$ 是修正Bessel函数，参数 $a > 0$，$b > 0$，$p \in \mathbb{R}$。

### TV先验下的GIG

在TV先验的GLM框架中，辅助变量 $z_j$ 的条件分布为：

$$z_j | (Kx)_j = t \sim \text{GIG}(\lambda^2, t^2, 1/2)$$

即 $a = \lambda^2$，$b = t^2$，$p = 1/2$。

### GIG的采样

GIG分布的采样有成熟的算法：

- 当 $p = 1/2$ 时（TV先验的情况），GIG退化为**逆高斯分布**（Inverse Gaussian），有高效的采样方法
- 一般情况可用接受-拒绝方法（Hörmann & Leydold, 2014）

逆高斯分布 $\text{IG}(\mu, \lambda)$ 的采样算法：

1. 生成 $V \sim \chi^2(1)$（标准卡方分布）
2. 计算 $Y = \mu + \frac{\mu^2 V}{2\lambda} - \frac{\mu}{2\lambda}\sqrt{4\mu\lambda V + \mu^2 V^2}$
3. 生成 $U \sim \text{Uniform}(0, 1)$
4. 若 $U \leq \frac{\mu}{\mu + Y}$，返回 $Y$；否则返回 $\frac{\mu^2}{Y}$

---

## A4.5 MCMC实践检查清单

将本章的内容整合为一个实践检查清单，供读者在实际应用中参考：

1. **选择算法**：光滑后验 → ULA；不可微 → MYULA；有条件结构 → Gibbs；需无偏 → MALA
2. **选择步长**：$\delta \leq 1/L$；过大则链发散，过小则移动慢；可用试错法调参
3. **选择 $\lambda$（MYULA）**：权衡近似误差与收敛速度；典型值 $\lambda \in [0.01, 1]$
4. **Burn-in**：从轨迹图判断，丢弃前期未收敛样本
5. **收敛诊断**：自相关函数快速衰减？ESS足够大？
6. **多链对比**：从不同初始值出发，检查是否收敛到同一区域
7. **不确定性量化**：用采样结果构造置信区间和可信区域
8. **结果验证**：采样结果的后验均值与MAP估计是否一致？（不一致可能表明后验偏斜或多峰）

---

## 参考文献

- Adler, R. J. (1981). *The Geometry of Random Fields*. Wiley.
- Cotter, S. L., Roberts, G. O., Stuart, A. M., & White, D. (2013). MCMC methods for functions: modifying old algorithms to make them faster. *Statistical Science*, 28(3), 424-446.
- Durmus, A., & Moulines, E. (2019). High-dimensional Bayesian inference via unadjusted Langevin algorithms. *Bernoulli*, 25(4A), 2854-2882.
- Durmus, A., Moulines, E., & Pereyra, M. (2018). Efficient Bayesian computation by proximal Markov chain Monte Carlo: when Langevin meets Moreau. *SIAM Journal on Imaging Sciences*, 11(1), 473-506.
- Falk, T., Habring, A., & Pock, T. (2025). Langevin meets Gibbs: Sampling via overrelaxation. *Preprint*.
- Fox, C., & Parker, A. (2017). Accelerated Gibbs sampling of normal distributions using matrix splittings and polynomials. *Journal of Computational and Graphical Statistics*, 26(2), 343-355.
- Frankel, S. (1950). Convergence rates of iterative treatments of partial differential equations. *Mathematical Tables and Other Aids to Computation*, 4(30), 65-75.
- Geman, S., & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 6(6), 721-741.
- Geman, D., & Reynolds, G. (1992). Constrained restoration and the recovery of discontinuities. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 14(3), 367-383.
- Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. *Biometrika*, 57(1), 97-109.
- Kuric, I., Zach, C., Habring, A., Unser, M., & Pock, T. (2025). Gaussian latent machines. *Preprint*.
- Pereyra, M. (2015). Proximal Markov chain Monte Carlo algorithms. *Statistics and Computing*, 26(4), 745-760.
- Pereyra, M., Mieles, L. V., & Zygalakis, K. C. (2020). Accelerating proximal Markov chain Monte Carlo by using an explicit stabilized method. *SIAM Journal on Imaging Sciences*, 13(2), 905-935.
- Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, 4(5), 1-17.
- Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. *The Annals of Applied Probability*, 7(1), 110-120.
- Roth, S., & Black, M. J. (2005). Fields of experts: A framework for learning image priors. *CVPR 2005*.
