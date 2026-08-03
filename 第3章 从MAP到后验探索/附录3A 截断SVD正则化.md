# 附录3A 截断SVD正则化

> *"Tikhonov 把小奇异值慢慢压下去，那我直接砍掉不行吗？"* —— 截断 SVD 就是这么"简单粗暴"的兄弟方法。

**定位**：与 Tikhonov 并列的经典谱正则化方法，供对比参考。不参与主线叙事。

---

## 截断SVD

3.3 节给了 Tikhonov 的 SVD 表示：

$$\hat{x}_\lambda = \sum_{i=1}^n \frac{\sigma_i}{\sigma_i^2 + \lambda}\langle y, u_i\rangle v_i$$

它的滤波函数 $\frac{\sigma_i}{\sigma_i^2+\lambda}$ 对所有奇异值做**平滑衰减**——小奇异值被大幅压，但不完全归零。

**截断SVD（TSVD）**换一种策略——**硬截断**：

$$\hat{x}_r = \sum_{i=1}^r \frac{1}{\sigma_i}\langle y, u_i\rangle v_i$$

只保留前 $r$ 个奇异值分量，后面的全扔。截断点 $r$ 扮演正则化参数的角色。

---

## 滤波函数对比

统一到谱滤波框架下看：

| 方法 | 滤波函数 $\varphi_i$ | 行为 |
|---|---|---|
| 最小二乘 | $\frac{1}{\sigma_i}$ | 无衰减，小 $\sigma_i$ 噪声放大 |
| Tikhonov | $\frac{\sigma_i}{\sigma_i^2 + \lambda}$ | 平滑衰减，渐进归零 |
| 截断SVD | $\frac{1}{\sigma_i}\mathbf{1}_{i \leq r}$ | 硬截断，留前 $r$ 个 |

关键区别：

- **Tikhonov 滤波**连续——$\sigma_i$ 越小衰减越多，但从不完全归零，解是数据的连续函数，保证稳定；
- **截断SVD 滤波**离散——$i\leq r$ 全留，$i>r$ 全扔。截断处突变可能带来伪影（如 Gibbs 现象）。

---

## 正则化参数的等价性

截断点 $r$ 与 Tikhonov 的 $\lambda$ 等价：

- $r$ 小（或 $\lambda$ 大）→ 更多分量被压 → 更强正则化 → 偏差大、方差小；
- $r$ 大（或 $\lambda$ 小）→ 更多分量留 → 更弱正则化 → 偏差小、方差大。

都面临偏差-方差权衡。Morozov 偏差原理同样适用 TSVD：选 $r$ 使 $\|A\hat{x}_r-y^\delta\|\approx\delta$。

---

## 谱正则化的统一视角

更一般地，可定义一族**谱正则化方法**，每种对应不同滤波函数 $\varphi(\sigma,\alpha)$：

$$\hat{x}_\alpha = \sum_i \varphi(\sigma_i, \alpha) \frac{\langle y, u_i\rangle}{\sigma_i} v_i$$

滤波函数需满足：

- $\varphi(\sigma,\alpha)\to 1$ 当 $\alpha\to 0$（无正则化→最小二乘）；
- $\varphi(\sigma,\alpha)\to 0$ 当 $\sigma\to 0$（压小奇异值→稳定化）。

不同正则化 = 不同滤波函数族。Tikhonov 和截断 SVD 是最基本的两种；更高级滤波（Tikhonov 高阶、Landweber 滤波）可突破 Tikhonov 饱和效应，达更高收敛阶。

截断 SVD 的优势是**资格无限**（qualification $\nu_0=\infty$）——对任意光滑度源条件都达最优收敛率，不受 Tikhonov 饱和（$\nu_0=2$）限制，理论上更优；但实践中硬截断伪影限制了应用。
