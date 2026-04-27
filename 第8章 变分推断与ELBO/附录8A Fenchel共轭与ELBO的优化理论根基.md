# 附录8A Fenchel共轭与ELBO的优化理论根基

> 定位：为8.4节"变分推断与正则化的统一视角"提供Fenchel共轭的数学基础，建立ELBO与凸分析的联系。

## 凸共轭（Fenchel共轭）定义

设 $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ 是正常凸函数（proper convex function），其**凸共轭**（Fenchel conjugate）定义为：

$$f^*(y) = \sup_{x \in \mathbb{R}^n} [\langle y, x \rangle - f(x)]$$

凸共轭 $f^*$ 总是凸函数（无论 $f$ 是否凸），因为它是仿射函数族 $\langle y, x \rangle - f(x)$ 关于 $y$ 的上确界，而仿射函数族的上确界是凸函数。

### 几何意义

$f^*(y)$ 的几何含义是：斜率为 $y$ 的超平面 $\langle y, x \rangle - c$ 与 $f(x)$ 的最大垂直距离。具体来说，$f^*(y)$ 是最大的 $c$ 使得 $\langle y, x \rangle - c \leq f(x)$ 对所有 $x$ 成立——即 $f^*(y)$ 是 $f$ 的"最优线性下界"的截距。

### 经典例子

1. **二次函数**：$f(x) = \frac{1}{2}\|x\|^2$ → $f^*(y) = \frac{1}{2}\|y\|^2$（自共轭）
   - 推导：$\sup_x [\langle y, x \rangle - \frac{1}{2}\|x\|^2] = \frac{1}{2}\|y\|^2$（在 $x = y$ 处取最大值）

2. **指数函数**：$f(x) = e^x$ → $f^*(y) = y\log y - y$（$y > 0$），$f^*(0) = 0$
   - 推导：$\sup_x [yx - e^x]$ 在 $x = \log y$ 处取最大值（当 $y > 0$）

3. **L1范数**：$f(x) = \|x\|_1$ → $f^*(y) = \mathbb{I}_{\|y\|_\infty \leq 1}$（示性函数：$y$ 在无穷范数单位球内时为0，否则为 $+\infty$）
   - 推导：$\sup_x [\langle y, x \rangle - \|x\|_1]$，当 $\|y\|_\infty > 1$ 时可以取到 $+\infty$

4. **负对数**：$f(x) = -\log x$（$x > 0$）→ $f^*(y) = -1 - \log(-y)$（$y < 0$）
   - 这个例子将在下面连接到ELBO时用到

## Fenchel-Young不等式

**定理**（Fenchel-Young不等式）：对任意 $x, y$，

$$f(x) + f^*(y) \geq \langle x, y \rangle$$

**证明**：由凸共轭的定义 $f^*(y) = \sup_{x'}[\langle y, x'\rangle - f(x')] \geq \langle y, x\rangle - f(x)$，移项即得。

**取等条件**：$f(x) + f^*(y) = \langle x, y \rangle$ 当且仅当 $y \in \partial f(x)$（$y$ 是 $f$ 在 $x$ 处的次梯度）。

Fenchel-Young不等式可以改写为：

$$f(x) \geq \langle x, y \rangle - f^*(y)$$

这意味着 $f(x)$ 被"线性化"下界估计——$f^*(y)$ 给出了线性下界 $\langle x, y \rangle - f^*(y)$ 的最优截距。

## 从Fenchel-Young到ELBO

现在，我们建立Fenchel共轭与ELBO的联系。关键观察是：$\log p(x) = \log \int p(x,z)dz$ 的变分表示与Fenchel-Young不等式有相同的数学结构。

### 函数空间的Fenchel共轭

将Fenchel共轭从有限维推广到函数空间。定义泛函：

$$F[q] = \mathbb{E}_{q(z)}[\log q(z)] = -H(q)$$

（$F$ 是负熵，$\mathbb{R}$-值泛函，定义在概率分布空间上。）

其对偶泛函：

$$F^*[g] = \sup_q \left[\mathbb{E}_{q(z)}[g(z)] - F[q]\right] = \sup_q \left[\mathbb{E}_{q(z)}[g(z)] + H(q)\right]$$

对于 $g(z) = \log p(x,z)$：

$$F^*[\log p(x,\cdot)] = \sup_q \left[\mathbb{E}_{q(z)}[\log p(x,z)] + H(q)\right] = \sup_q \text{ELBO}(q)$$

由Fenchel-Young不等式：

$F[q] + F^*[\log p(x,\cdot)] \geq \mathbb{E}_{q(z)}[\log p(x,z)]$

而当 $q = p(z|x)$ 取等时：

$\sup_{q'} \text{ELBO}(q') = \log p(x)$

因此：

$$\log p(x) = F^*[\log p(x,\cdot)] = \sup_q \text{ELBO}(q)$$

**ELBO是 $\log p(x)$ 的Fenchel对偶表示**——通过凸共轭获得的下界。变分间隙 $\text{KL}(q\|p(z|x))$ 对应Fenchel-Young间隙。

### 更直接的推导

更直接地，可以将ELBO的推导看作Jensen不等式的应用，而Jensen不等式本身是Fenchel-Young不等式在凸函数情形的特例。

$\log$ 是凸函数的负数（$\log$ 是凹函数），对凹函数 $g$ 的Jensen不等式为：

$$\mathbb{E}[g(X)] \leq g(\mathbb{E}[X])$$

反过来，对凸函数 $f = -g$：

$$\mathbb{E}[f(X)] \geq f(\mathbb{E}[X])$$

这恰好是Fenchel-Young不等式在仿射下界取极限的结果。因此，**Jensen不等式推导ELBO是Fenchel-Young不等式的特例**——两者都来自凸分析的同一个根基。

## 变分正则化的凸分析视角

Fenchel共轭也为变分正则化提供了统一的凸分析视角。

### Tikhonov正则化的对偶

考虑Tikhonov正则化问题：

$$\min_x \frac{1}{2}\|Ax - y\|^2 + \frac{\lambda}{2}\|x\|^2$$

由Fenchel对偶理论（LectureNotes2020_v2 Ch4, Theorem 4.1.43），其对偶问题为：

$$\max_p \left[-\frac{1}{2}\|A^*p\|^2 - \frac{1}{2\lambda}\|p - y/\lambda\|^2 + \frac{1}{2\lambda}\|y\|^2\right]$$

原始-对偶间隙 = 原始值 - 对偶值 ≥ 0（弱对偶性）。当问题满足强对偶条件时，间隙为零。

### 概率框架 vs 优化框架的对偶

| 概率框架 | 优化框架 |
|---|---|
| $\log p(x)$ | 原始问题最优值 |
| ELBO | 对偶问题值 |
| KL间隙 | 对偶间隙 |
| $q = p(z\|x)$ 时间隙为零 | 强对偶时间隙为零 |

两种框架共享同一套凸分析工具——Fenchel共轭是统一的数学语言。这解释了为什么ELBO与正则化有相同的"数据+正则"结构：**它们都是同一个凸优化问题的不同表达**。

**来源**：Benning L2 P23-25; LectureNotes2020_v2 Ch4 (Fenchel conjugate, lines 1832-1979); Rockafellar (1970) Convex Analysis; Borwein & Lewis (2010) Convex Analysis and Nonlinear Optimization
