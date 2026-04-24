# 附录3A 截断SVD正则化

> 定位：与Tikhonov并列的经典谱正则化方法，供对比参考。不参与主线叙事。

## 截断SVD

3.3节讨论了Tikhonov正则化的SVD表示：

$$\hat{x}_\lambda = \sum_{i=1}^n \frac{\sigma_i}{\sigma_i^2 + \lambda}\langle y, u_i\rangle v_i$$

其中滤波函数 $\frac{\sigma_i}{\sigma_i^2 + \lambda}$ 对所有奇异值做**平滑衰减**——小奇异值被大幅压制，但不完全归零。

**截断SVD**（Truncated SVD, TSVD）采用另一种策略——**硬截断**：

$$\hat{x}_r = \sum_{i=1}^r \frac{1}{\sigma_i}\langle y, u_i\rangle v_i$$

保留前 $r$ 个奇异值分量，完全丢弃后面的。截断点 $r$ 扮演正则化参数的角色。

## 滤波函数对比

将两种方法统一到谱滤波框架下：

| 方法 | 滤波函数 $\varphi_i$ | 行为 |
|---|---|---|
| 最小二乘 | $\frac{1}{\sigma_i}$ | 无衰减，小 $\sigma_i$ 噪声放大 |
| Tikhonov | $\frac{\sigma_i}{\sigma_i^2 + \lambda}$ | 平滑衰减，渐进归零 |
| 截断SVD | $\frac{1}{\sigma_i}\mathbf{1}_{i \leq r}$ | 硬截断，保留前 $r$ 个 |

关键区别：

- **Tikhonov滤波**是连续的——$\sigma_i$ 越小，衰减越多，但从不完全归零。这使得Tikhonov解是数据的连续函数，保证了稳定性
- **截断SVD滤波**是离散的——$i \leq r$ 时完全保留，$i > r$ 时完全丢弃。截断点处的突变可能导致伪影（如Gibbs现象）

## 正则化参数的等价性

截断SVD的截断点 $r$ 与Tikhonov的 $\lambda$ 扮演等价的角色：

- $r$ 小（或 $\lambda$ 大）→ 更多分量被压制→更强正则化→偏差大、方差小
- $r$ 大（或 $\lambda$ 小）→ 更多分量被保留→更弱正则化→偏差小、方差大

两者的选择都面临偏差-方差的权衡。Morozov偏差原理同样适用于截断SVD：选择 $r$ 使 $\|A\hat{x}_r - y^\delta\| \approx \delta$。

## 谱正则化的统一视角

更一般地，可以定义一族**谱正则化方法**，每种方法对应不同的滤波函数 $\varphi(\sigma, \alpha)$：

$$\hat{x}_\alpha = \sum_i \varphi(\sigma_i, \alpha) \frac{\langle y, u_i\rangle}{\sigma_i} v_i$$

滤波函数需要满足：
- $\varphi(\sigma, \alpha) \to 1$ 当 $\alpha \to 0$（无正则化→最小二乘）
- $\varphi(\sigma, \alpha) \to 0$ 当 $\sigma \to 0$（压制小奇异值→稳定化）

不同正则化方法 = 不同的滤波函数族。Tikhonov和截断SVD是最基本的两种；更高级的滤波函数（如Tikhonov高阶、Landweber滤波）可以突破Tikhonov的饱和效应，达到更高的收敛阶。

截断SVD的一个优势是**资格无限**（qualification $\nu_0 = \infty$）——对任意光滑度的源条件都能达到最优收敛率，不受Tikhonov饱和效应（$\nu_0 = 2$）的限制。这使得截断SVD在理论上更优，但实践中的硬截断伪影限制了它的应用。
