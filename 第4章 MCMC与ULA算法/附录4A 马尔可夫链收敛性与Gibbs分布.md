# 附录4A 马尔可夫链收敛性与Gibbs分布

> 本附录补充 4.2 节和 4.5 节用到的马尔可夫链理论基础，以及 Gibbs 分布的数学性质。当正文说"链跑久了就收敛到目标后验"时，这里的定理给它兜底。

## A4.1 马尔可夫链的收敛性

### 遍历定理

MCMC 的理论基石是马尔可夫链的**遍历定理**（ergodic theorem）：

**定理**（遍历定理）：设 $(X_m)_{m\ge1}$ 是以 $\mu$ 为平稳分布的不可约、非周期、Harris 常返马尔可夫链，则对任意可积函数 $h$：
$$\frac{1}{M}\sum_{m=1}^M h(X_m) \xrightarrow{\text{a.s.}} \int h(x)\,\mu(\mathrm{d}x), \quad M\to\infty$$
这就是 MCMC 的理论保证：即使样本不独立，样本均值照样收敛到期望。

### 不可约性、非周期性与常返性

遍历定理的三个条件：
1. **不可约性**（irreducibility）：链能从任何状态到任何状态（以正概率）——保证能逛遍整个支撑集；
2. **非周期性**（aperiodicity）：链不会在几个子集间周期振荡——保证收敛到平稳分布（而非周期摆动）；
3. **Harris 常返性**（Harris recurrence）：链会无限次访问每个正概率区域——保证大数定律成立。

对 MH 和 ULA，温和条件下这三条都满足。

### 收敛速率

遍历定理只保证收敛、不给速率。速率由**谱间隙**（spectral gap）决定：
- **几何遍历**（geometrically ergodic）：$\|K^m(x,\cdot)-\mu\|_{\text{TV}}\le C(x)r^m,\ r<1$——最快的实用速率；
- **多项式遍历**：速率 $O(1/m^\alpha)$——慢。

ULA 在强对数凹条件下是几何遍历的，速率 $O(e^{-cm})$。

---

## A4.2 细致平衡与可逆性

### 可逆性

4.2 节的细致平衡条件等价于马尔可夫核 $K$ 关于 $\mu$ **可逆**（reversible）：
$$\int_A K(B|x)\,\mu(\mathrm{d}x) = \int_B K(A|x)\,\mu(\mathrm{d}x), \quad \forall A,B$$
可逆性比平稳性更强：
$$\text{可逆性} \implies \text{平稳性} \implies \text{遍历性（在不可约非周期条件下）}$$

### 不可逆 MCMC

不满足细致平衡但仍有正确平稳分布的，叫**不可逆 MCMC**。理论上它可有更小的自相关、更快收敛——打破细致平衡允许"定向流"。ILA（4.6 节）在某种意义上"部分不可逆"：动量项引入了非对称转移方向。欠阻尼 Langevin SDE 的连续极限也不满足细致平衡（在 $(x,v)$ 空间有定向流），但平稳分布仍正确。

---

## A4.3 Gibbs分布与指数族

### Gibbs分布

**Gibbs 分布**（又称 Boltzmann 分布）是统计物理与概率论的基本分布族：
$$p(x) = \frac{1}{Z}\exp(-E(x))$$
$E(x)$ 是**能量函数**，$Z=\int\exp(-E(x))\,\mathrm{d}x$ 是配分函数（归一化常数）。贝叶斯后验是它的特例：$E(x)=-\log p(y|x)-\log p(x)$，$Z=p(y)$。

### 高斯 Gibbs分布

当能量是二次型 $E(x)=\frac{1}{2}(x-\mu)^\top Q(x-\mu)$，Gibbs 分布就是高斯：
$$p(x)=\mathcal{N}(x\mid\mu, Q^{-1})$$
$Q$ 是精度矩阵（precision matrix，协方差的逆）。高斯 Gibbs 的**条件分布也是高斯**——这是 Gibbs 采样在高斯模型上高效的根本原因。

### 二元高斯的条件分布

设 $x=(x_1,x_2)\sim\mathcal{N}(\mu,Q^{-1})$，精度矩阵 $Q=\begin{pmatrix}Q_{11}&Q_{12}\\Q_{21}&Q_{22}\end{pmatrix}$，则
$$p(x_1|x_2)=\mathcal{N}(x_1\mid\mu_{1|2},\ Q_{11}^{-1}),\quad \mu_{1|2}=\mu_1-Q_{11}^{-1}Q_{12}(x_2-\mu_2)$$
注意条件方差 $Q_{11}^{-1}$ **不依赖** $x_2$——这是高斯的特殊性质，让 Gibbs 采样里的条件分布好算。

---

## A4.4 广义逆高斯分布（GIG）

Gibbs 在 TV 先验下要从 GIG 抽样（4.5 节）。这里给定义和抽样法。

### 定义

**广义逆高斯分布**（Generalized Inverse Gaussian, GIG）密度为：
$$p(z) = \frac{(a/b)^{p/2}}{2K_p(\sqrt{ab})}\,z^{p-1}\exp\left(-\frac{1}{2}(az+b/z)\right),\quad z>0$$
$K_p$ 是修正 Bessel 函数，参数 $a>0,\ b>0,\ p\in\mathbb{R}$。

### TV 先验下的 GIG

在 TV 先验的 GLM 框架里，辅助变量 $z_j$ 的条件分布是：
$$z_j\mid (Kx)_j=t \sim \text{GIG}(\lambda^2, t^2, 1/2)$$
即 $a=\lambda^2,\ b=t^2,\ p=1/2$。

### GIG 的抽样

- $p=1/2$（TV 的情况）时，GIG 退化成**逆高斯分布**（Inverse Gaussian），有高效抽样法；
- 一般情形可用接受-拒绝法（Hörmann & Leydold, 2014）。

逆高斯 $\text{IG}(\mu,\lambda)$ 抽样算法：
1. 生成 $V\sim\chi^2(1)$（标准卡方）
2. 算 $Y=\mu+\frac{\mu^2 V}{2\lambda}-\frac{\mu}{2\lambda}\sqrt{4\mu\lambda V+\mu^2 V^2}$
3. 生成 $U\sim\text{Uniform}(0,1)$
4. 若 $U\le\frac{\mu}{\mu+Y}$，返回 $Y$；否则返回 $\frac{\mu^2}{Y}$

---

## A4.5 MCMC实践检查清单

把本章内容压成一个实战清单，随手备查：

1. **选算法**：光滑后验 → ULA；不可微 → MYULA；有条件结构 → Gibbs；需无偏 → MALA
2. **选步长**：$\delta\le1/L$；过大发散、过小慢；可试错调参
3. **选 $\lambda$（MYULA）**：权衡近似误差与收敛速度；典型 $\lambda\in[0.01,1]$
4. **Burn-in**：从轨迹图判断，丢弃未收敛样本
5. **收敛诊断**：自相关函数快速衰减？ESS 足够大？
6. **多链对比**：不同起点出发，看是否汇合到同一区域
7. **不确定性量化**：用样本构造置信区间和可信区域
8. **结果验证**：后验均值与 MAP 是否一致？（不一致可能说明后验偏斜或多峰）

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
- Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. *Biometrika*, 57(1), 97-110.
- Kuric, I., Zach, C., Habring, A., Unser, M., & Pock, T. (2025). Gaussian latent machines. *Preprint*.
- Pereyra, M. (2015). Proximal Markov chain Monte Carlo algorithms. *Statistics and Computing*, 26(4), 745-760.
- Pereyra, M., Mieles, L. V., & Zygalakis, K. C. (2020). Accelerating proximal Markov chain Monte Carlo by using an explicit stabilized method. *SIAM Journal on Imaging Sciences*, 13(2), 905-935.
- Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, 4(5), 1-17.
- Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. *The Annals of Applied Probability*, 7(1), 110-120.
- Roth, S., & Black, M. J. (2005). Fields of experts: A framework for learning image priors. *CVPR 2005*.
