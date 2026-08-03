# 附录 17A SURE 的无偏性证明

> 定位：17.3.2 节给了 SURE 公式和直觉，这里补上严格证明。**如果你只关心"SURE 为什么能只用 $y$ 无偏估计监督风险"，读 A、B 两节即可**（这两节是本章论证的核心，建议通读）；C 节讲最优解为何是 Tweedie 公式；D 节是 Poisson-Gaussian 推广。数学细节可按需跳读，不影响主文理解。

**本附录路线图**：
- **A 节**：证明 Stein 引理——把"网络追噪声的偏差"换成只依赖 $y$ 的散度；
- **B 节**：用 Stein 引理推出 SURE 期望等于监督风险加常数（无偏性）；
- **C 节**：对 SURE 做变分，得到最优解恰是 Tweedie 公式；
- **D 节**：把 SURE 推广到 Poisson-Gaussian 噪声。

## A. Stein 引理

**引理**（Stein 1981）：设 $y|x \sim \mathcal{N}(x, \sigma^2 I_n)$（高斯噪声），函数 $f: \mathbb{R}^n \to \mathbb{R}^n$ 弱可微，且 $\mathbb{E}[\|f(y)\|^2] < \infty$，则

$$\mathbb{E}[\varepsilon^\top f(y)] = \sigma^2 \mathbb{E}\left[\sum_{i=1}^n \frac{\partial f_i}{\partial y_i}(y)\right]$$

其中 $\varepsilon = y - x$。

### 证明

目标：对每个分量 $i$，证 $\mathbb{E}[\varepsilon_i f_i(y)] = \sigma^2 \mathbb{E}\left[\frac{\partial f_i}{\partial y_i}(y)\right]$。

**第一步**：条件期望。给定 $x$，$y \sim \mathcal{N}(x, \sigma^2 I_n)$，于是 $y_i \sim \mathcal{N}(x_i, \sigma^2)$，$\varepsilon_i = y_i - x_i \sim \mathcal{N}(0, \sigma^2)$。

**第二步**：对 $y_i$ 分部积分。

$$\mathbb{E}[\varepsilon_i f_i(y) | x] = \int_{-\infty}^{\infty} (y_i - x_i) f_i(y) \frac{1}{\sqrt{2\pi\sigma^2}} e^{-(y_i - x_i)^2/(2\sigma^2)} dy_i$$

令 $t = y_i - x_i$：

$$= \int_{-\infty}^{\infty} t \cdot f_i(y) \frac{1}{\sqrt{2\pi\sigma^2}} e^{-t^2/(2\sigma^2)} dt$$

注意到 $t e^{-t^2/(2\sigma^2)} = -\sigma^2 \frac{d}{dt}e^{-t^2/(2\sigma^2)}$，分部积分得：

$$= \sigma^2 \int_{-\infty}^{\infty} \frac{\partial f_i}{\partial y_i}(y) \frac{1}{\sqrt{2\pi\sigma^2}} e^{-t^2/(2\sigma^2)} dt = \sigma^2 \mathbb{E}\left[\frac{\partial f_i}{\partial y_i}(y) \bigg| x\right]$$

**第三步**：对 $x$ 取期望，并对所有分量求和，即得 Stein 引理。$\blacksquare$

> **可跳过细节**：上面三步严格建立了"网络输出和噪声的协方差 = $\sigma^2 \times$ 散度"。读不懂分部积分完全不影响主文——只要记住这条引理把"追噪声的偏差"翻译成了"只依赖 $y$ 的散度项"即可。

## B. SURE 无偏性

**定理**：设 $y|x \sim \mathcal{N}(x, \sigma^2 I_n)$，则

$$\mathbb{E}_y[\mathcal{L}_{\text{SURE}}(y, f)] = \mathbb{E}_{x,y}\|x - f(y)\|^2 + n\sigma^2$$

### 证明

展开监督风险（把 $x - f(y)$ 写成 $(y - f(y)) - (y - x)$）：

$$R_{\text{SUP}}(f) = \mathbb{E}_{x,y}\|(y - f(y)) - (y - x)\|^2$$

$$= \mathbb{E}_y\|y - f(y)\|^2 - 2\mathbb{E}_{x,y}[(y - x)^\top(y - f(y))] + \mathbb{E}\|y - x\|^2$$

$$= \mathbb{E}_y\|y - f(y)\|^2 - 2\mathbb{E}[\varepsilon^\top(y - f(y))] + n\sigma^2$$

$$= \mathbb{E}_y\|y - f(y)\|^2 - 2\mathbb{E}[\varepsilon^\top y] + 2\mathbb{E}[\varepsilon^\top f(y)] + n\sigma^2$$

由于 $\mathbb{E}[\varepsilon^\top y] = \mathbb{E}[\varepsilon^\top(x + \varepsilon)] = 0 + n\sigma^2 = n\sigma^2$，并套用 Stein 引理 $\mathbb{E}[\varepsilon^\top f(y)] = \sigma^2 \mathbb{E}[\text{div}\, f(y)]$：

$$R_{\text{SUP}}(f) = \mathbb{E}_y\|y - f(y)\|^2 - 2n\sigma^2 + 2\sigma^2\mathbb{E}[\text{div}\, f(y)] + n\sigma^2$$

$$= \mathbb{E}_y\left[\|y - f(y)\|^2 + 2\sigma^2 \text{div}\, f(y)\right] - n\sigma^2 = \mathbb{E}_y[\mathcal{L}_{\text{SURE}}(y, f)] - n\sigma^2$$

因此 $\mathbb{E}_y[\mathcal{L}_{\text{SURE}}(y, f)] = R_{\text{SUP}}(f) + n\sigma^2$。$\blacksquare$

## C. SURE 的最优解为 Tweedie 公式

对 SURE 做变分：给 $f$ 加微小扰动 $f \to f + \delta g$，令变分为零：

$$\frac{\delta}{\delta f}\mathbb{E}_y[\mathcal{L}_{\text{SURE}}] = 0$$

展开变分：

$$\mathbb{E}_y\left[-2(y - f(y)) + 2\sigma^2 \frac{\partial g}{\partial y}\right]_{\delta \to 0} = 0$$

对任意扰动 $g$，用分部积分消去 $g$ 的导数项，得到 Euler–Lagrange 方程：

$$f^*(y) = y + \sigma^2 \nabla_y \log p_y(y)$$

这正是 Tweedie 公式——噪声分布 $p_y$ 下的 MMSE 去噪器。$\blacksquare$

> **可跳过细节**：这段变分推导说明 SURE 损失的最优解就是 Tweedie 公式，从而把自监督去噪和全书得分函数主线连起来。只关心结论的读者可记：SURE 训出的最优去噪器 = Tweedie 给出的 MMSE 估计器。

## D. 推广到 Poisson-Gaussian 噪声

Hudson (1978) 把 Stein 引理推广到指数族噪声。对 Poisson-Gaussian 噪声 $y \sim \text{Poisson}(x) + \mathcal{N}(0, \sigma^2 I)$，SURE 形式为：

$$\mathcal{L}_{\text{PG-SURE}}(y, f) = \|y - f(y)\|^2 + 2\text{Cov}(y, f(y)) - \text{Var}(y)$$

协方差项可用 Monte Carlo 近似。详见 Hudson (1978) 与 Tachella et al. (ICLR 2025)。
