# 附录 17B R2R 与 SURE 等价性的推导

> 定位：17.3.5 节给了 R2R 公式和"渐近等价 SURE"的陈述，这里补全推导。重点看 A 节（为什么两个重加噪副本条件独立）和 C 节（为什么 $\alpha \to 0$ 时 R2R 退化成 SURE）。

整个推导的枢纽只有一个词：**独立性**。R2R 用"一加一减"的重加噪，把单帧观测拆成两路条件独立的噪声副本——正是独立性，使得两路副本之间的 MSE 期望恰好分解出"信号拟合项 + 散度修正项"的结构，躲开了朴素损失的系统性低估；而 $\alpha \to 0$ 的极限只是让"重排后的观测"退回原始观测，从而与 SURE 会师。

## A. $y_a$ 和 $y_b$ 的条件独立性

**命题**：设 $y = x + \varepsilon$，$\varepsilon \sim \mathcal{N}(0, \sigma^2 I_n)$。定义 $y_a = y + \alpha\omega$，$y_b = y - \omega/\alpha$，其中 $\omega \sim \mathcal{N}(0, \sigma^2 I_n)$ 且独立于 $\varepsilon$。则 $y_a, y_b$ 在给定 $x$ 时条件独立。

这一步是 R2R 全部魔法的源头：只要两路噪声"互不通气"，它们之间的误差才可能成为干净信号差异的化身。

### 证明

**第一步**：写出两者的噪声分量。

$$y_a = x + \underbrace{(\varepsilon + \alpha\omega)}_{\varepsilon_a}, \qquad y_b = x + \underbrace{(\varepsilon - \omega/\alpha)}_{\varepsilon_b}$$

**第二步**：算条件协方差。

$$\text{Cov}(\varepsilon_a, \varepsilon_b | x) = \mathbb{E}[(\varepsilon + \alpha\omega)(\varepsilon - \omega/\alpha)^\top | x]$$

$$= \mathbb{E}[\varepsilon\varepsilon^\top] - \frac{1}{\alpha}\mathbb{E}[\varepsilon\omega^\top] + \alpha\mathbb{E}[\omega\varepsilon^\top] - \mathbb{E}[\omega\omega^\top] = \sigma^2 I - 0 + 0 - \sigma^2 I = 0$$

（用 $\varepsilon, \omega$ 独立，故 $\mathbb{E}[\varepsilon\omega^\top] = 0$。）

**第三步**：$\varepsilon_a, \varepsilon_b$ 是联合高斯的（线性变换保持高斯性），协方差为零即条件独立。$\blacksquare$

---

## B. R2R 损失的展开

**R2R 损失**：

$$\mathcal{L}_{\text{R2R}}(y, f, \alpha) = \mathbb{E}_\omega\|y_b - f(y_a)\|^2$$

展开：

$$= \mathbb{E}_\omega\left[\|y_b\|^2 - 2y_b^\top f(y_a) + \|f(y_a)\|^2\right]$$

代入 $y_b = y - \omega/\alpha$ 并对 $\omega$ 取期望：

$$= \mathbb{E}_\omega\left[\|y - \omega/\alpha\|^2\right] - 2\mathbb{E}_\omega[y_b^\top f(y_a)] + \mathbb{E}_\omega[\|f(y_a)\|^2]$$

第一项 $= \|y\|^2 + n\sigma^2/\alpha^2$（与 $f$ 无关）。关键在第二项 $\mathbb{E}_\omega[y_b^\top f(y_a)]$——它同时含着 $y_b$（随机）与 $f(y_a)$（对随机输入的响应），两者的纠缠方式决定了 R2R 的命运。

---

## C. 渐近等价性 $\lim_{\alpha \to 0}\mathcal{L}_{\text{R2R}} = \mathcal{L}_{\text{SURE}}$

**定理**：在 $f$ 连续可微等正则性条件下，$\lim_{\alpha \to 0}\mathcal{L}_{\text{R2R}}(y, f, \alpha) = \mathcal{L}_{\text{SURE}}(y, f)$。

直觉先行的说法是：$\alpha \to 0$ 时 $y_a$ 几乎就是 $y$，而 $y_b$ 是叠加在 $y$ 上、幅度被 $1/\alpha$ 放大的"正交扰动"——扰动被放大，反而把它携带的一阶信息精确地"顶"到了台面上。下面把这句话算出来。

### 证明思路

**第一步**：$\alpha \to 0$ 时 $y_a = y + \alpha\omega \to y$，故 $f(y_a) \approx f(y) + \alpha J_f(y)\omega$，其中 $J_f = \partial f/\partial y$ 是 Jacobian（导数矩阵）。

**第二步**：$y_b = y - \omega/\alpha$，于是

$$y_b^\top f(y_a) \approx (y - \omega/\alpha)^\top(f(y) + \alpha J_f(y)\omega) = y^\top f(y) + \alpha y^\top J_f(y)\omega - \frac{1}{\alpha}\omega^\top f(y) - \omega^\top J_f(y)\omega$$

**第三步**：对 $\omega$ 取期望（用 $\mathbb{E}[\omega]=0$，$\mathbb{E}[\omega\omega^\top]=\sigma^2 I$）：

$$\mathbb{E}_\omega[y_b^\top f(y_a)] \approx y^\top f(y) - \sigma^2\text{tr}(J_f(y))$$

**第四步**：代回 B 节的展开，消去常数项，得

$$\mathcal{L}_{\text{R2R}} \approx \|y - f(y)\|^2 + 2\sigma^2\text{tr}(J_f(y)) = \mathcal{L}_{\text{SURE}}(y, f) \qquad \blacksquare$$

---

## D. Generalized R2R 的扩展

Monroy, Bacca & Tachella (CVPR 2025) 把 R2R 推广到非高斯噪声：

- **Poisson 噪声**：从 $\text{Poisson}(y)$ 重采样 $y'$，构造 $y_a = y + y'$，$y_b = y - y'/\alpha$（用 Poisson 可加性）；
- **Gamma 噪声**：用 Gamma 分布的缩放性质造副本；
- **Binomial 噪声**：用二项分布分解性质造副本。

**核心洞察**：任何有已知采样结构的噪声模型，都能造独立重加噪副本，从而替代 SURE 里那个散度计算。这与 A 节的结论一脉相承——独立副本一旦造出，散度就不再是必需品；变化的只是"怎么造出这对副本"，不变的是"用独立副本换掉散度"这条设计路线。
