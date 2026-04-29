# 附录17A SURE的无偏性证明

> 定位：17.3.2节给出了SURE公式和直觉，本附录提供严格证明。

## A. Stein引理

**引理**（Stein 1981）：设$y|x \sim \mathcal{N}(x, \sigma^2 I_n)$，$f: \mathbb{R}^n \to \mathbb{R}^n$是弱可微函数，且$\mathbb{E}[\|f(y)\|^2] < \infty$，则

$$\mathbb{E}[\varepsilon^\top f(y)] = \sigma^2 \mathbb{E}\left[\sum_{i=1}^n \frac{\partial f_i}{\partial y_i}(y)\right]$$

其中$\varepsilon = y - x$。

### 证明

对每个分量$i$，需要证明$\mathbb{E}[\varepsilon_i f_i(y)] = \sigma^2 \mathbb{E}\left[\frac{\partial f_i}{\partial y_i}(y)\right]$。

**第一步**：利用条件期望。给定$x$，$y \sim \mathcal{N}(x, \sigma^2 I_n)$，$y_i \sim \mathcal{N}(x_i, \sigma^2)$，$\varepsilon_i = y_i - x_i \sim \mathcal{N}(0, \sigma^2)$。

**第二步**：分部积分。对$y_i$进行分部积分：

$$\mathbb{E}[\varepsilon_i f_i(y) | x] = \int_{-\infty}^{\infty} (y_i - x_i) f_i(y) \frac{1}{\sqrt{2\pi\sigma^2}} e^{-(y_i - x_i)^2/(2\sigma^2)} dy_i$$

令$t = y_i - x_i$：

$$= \int_{-\infty}^{\infty} t \cdot f_i(y) \frac{1}{\sqrt{2\pi\sigma^2}} e^{-t^2/(2\sigma^2)} dt$$

注意到$t e^{-t^2/(2\sigma^2)} = -\sigma^2 \frac{d}{dt}e^{-t^2/(2\sigma^2)}$，分部积分得：

$$= \sigma^2 \int_{-\infty}^{\infty} \frac{\partial f_i}{\partial y_i}(y) \frac{1}{\sqrt{2\pi\sigma^2}} e^{-t^2/(2\sigma^2)} dt$$

$$= \sigma^2 \mathbb{E}\left[\frac{\partial f_i}{\partial y_i}(y) \bigg| x\right]$$

**第三步**：取期望（对$x$）：

$$\mathbb{E}[\varepsilon_i f_i(y)] = \sigma^2 \mathbb{E}\left[\frac{\partial f_i}{\partial y_i}(y)\right]$$

对所有分量求和即得Stein引理。$\blacksquare$

## B. SURE无偏性

**定理**：设$y|x \sim \mathcal{N}(x, \sigma^2 I_n)$，则

$$\mathbb{E}_y[\mathcal{L}_{\text{SURE}}(y, f)] = \mathbb{E}_{x,y}\|x - f(y)\|^2 + n\sigma^2$$

### 证明

展开监督风险：

$$R_{\text{SUP}}(f) = \mathbb{E}_{x,y}\|x - f(y)\|^2 = \mathbb{E}_{x,y}\|(y - f(y)) - (y - x)\|^2$$

$$= \mathbb{E}_y\|y - f(y)\|^2 - 2\mathbb{E}_{x,y}[(y - x)^\top(y - f(y))] + \mathbb{E}\|y - x\|^2$$

$$= \mathbb{E}_y\|y - f(y)\|^2 - 2\mathbb{E}[\varepsilon^\top(y - f(y))] + n\sigma^2$$

$$= \mathbb{E}_y\|y - f(y)\|^2 - 2\mathbb{E}[\varepsilon^\top y] + 2\mathbb{E}[\varepsilon^\top f(y)] + n\sigma^2$$

由于$\mathbb{E}[\varepsilon^\top y] = \mathbb{E}[\varepsilon^\top(x + \varepsilon)] = 0 + n\sigma^2 = n\sigma^2$，以及Stein引理$\mathbb{E}[\varepsilon^\top f(y)] = \sigma^2 \mathbb{E}[\text{div}\, f(y)]$：

$$R_{\text{SUP}}(f) = \mathbb{E}_y\|y - f(y)\|^2 - 2n\sigma^2 + 2\sigma^2\mathbb{E}[\text{div}\, f(y)] + n\sigma^2$$

$$= \mathbb{E}_y\left[\|y - f(y)\|^2 + 2\sigma^2 \text{div}\, f(y)\right] - n\sigma^2$$

$$= \mathbb{E}_y[\mathcal{L}_{\text{SURE}}(y, f)] - n\sigma^2$$

因此$\mathbb{E}_y[\mathcal{L}_{\text{SURE}}(y, f)] = R_{\text{SUP}}(f) + n\sigma^2$。$\blacksquare$

## C. SURE的最优解为Tweedie公式

对SURE做变分，考虑函数扰动$f \to f + \delta g$，令变分为零：

$$\frac{\delta}{\delta f}\mathbb{E}_y[\mathcal{L}_{\text{SURE}}] = 0$$

展开变分：

$$\mathbb{E}_y\left[-2(y - f(y)) + 2\sigma^2 \frac{\partial g}{\partial y}\right]_{\delta \to 0} = 0$$

对任意扰动$g$，利用分部积分消去$g$的导数项，得到Euler-Lagrange方程：

$$f^*(y) = y + \sigma^2 \nabla_y \log p_y(y)$$

这正是Tweedie公式——噪声分布$p_y$下的MMSE去噪器。$\blacksquare$

## D. 推广到Poisson-Gaussian噪声

Hudson (1978) 将Stein引理推广到指数族噪声。对于Poisson-Gaussian噪声$y \sim \text{Poisson}(x) + \mathcal{N}(0, \sigma^2 I)$，SURE的形式为：

$$\mathcal{L}_{\text{PG-SURE}}(y, f) = \|y - f(y)\|^2 + 2\text{Cov}(y, f(y)) - \text{Var}(y)$$

其中协方差项可用Monte Carlo近似。具体推导见Hudson (1978) 和Tachella et al. (ICLR 2025)。
