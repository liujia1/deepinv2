# 附录8C KL散度的性质与变分推断的信息论视角

> 定位：为 8.2–8.3 节补充 KL 散度的性质，并从信息论（编码长度）视角重新看 ELBO。8.3 节那道最亮的 aha——零强迫 vs 零避免——本质上就是 KL 不对称性的两个后果；这里把它的数学性质坐实。读完你会发现：最大化 ELBO 等价于"用最短的码把数据说清楚"——和最小描述长度原则一脉相承。

## KL散度的定义与性质

KL 散度（Kullback-Leibler divergence）衡量两个分布"差多远"。对连续分布 $q,p$：

$$\text{KL}(q \| p) = \int q(x) \log \frac{q(x)}{p(x)} dx = \mathbb{E}_{q(x)}\left[\log \frac{q(x)}{p(x)}\right]$$

### 基本性质

1. **非负性**：$\text{KL}(q\|p) \geq 0$，等号当且仅当 $q=p$（几乎处处）
   - 证明：Jensen 不等式，$\text{KL} = -\mathbb{E}_q[\log(p/q)] \geq -\log\mathbb{E}_q[p/q] = -\log 1 = 0$
2. **非对称性**：$\text{KL}(q\|p) \neq \text{KL}(p\|q)$——所以它**不是真正的距离**，没有对称性
3. **不三角不等式**：一般 $\text{KL}(q\|p) \not\leq \text{KL}(q\|r)+\text{KL}(r\|p)$——进一步说明它不是度量（metric）
4. **凸性**：关于 $(q,p)$ 联合凸——这保了一部分 ELBO 的凸结构
5. **与交叉熵**：$\text{KL}(q\|p) = H(q,p) - H(q)$，其中 $H(q,p)=-\int q\log p$ 是交叉熵，$H(q)$ 是熵
6. **与互信息**：$I(X;Y) = \text{KL}(p(x,y)\|p(x)p(y))$——互信息就是联合分布与边际乘积的 KL

## 前向KL vs 逆向KL

非对称性导致两种近似模式（8.4 节已用图展示，这里补性质）：

### 前向KL：$\text{KL}(p\|q)$
$$\text{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} dx$$
- **零避免（zero-avoiding）**：$q$ 在 $p>0$ 处必须 $>0$，否则 $\log(p/q)\to\infty$
- 倾向让 $q$ **覆盖** $p$ 所有模态（均值寻求，mean-seeking）——像"广播"，别漏听众

### 逆向KL：$\text{KL}(q\|p)$
$$\text{KL}(q \| p) = \int q(x) \log \frac{q(x)}{p(x)} dx$$
- **零强迫（zero-forcing）**：$q$ 在 $p=0$ 处不能有质量
- 倾向让 $q$ **锁住** $p$ 的一个模态（模态寻求，mode-seeking）——像"狙击"，瞄准一个目标

```
前向KL最小化 (KL(p||q)):           逆向KL最小化 (KL(q||p)):
                                     
  p(x) (真实分布,双峰)               p(x) (真实分布,双峰)
   ╱╲      ╱╲                        ╱╲      ╱╲
  ╱  ╲    ╱  ╲                      ╱  ╲    ╱  ╲
 ╱    ╲  ╱    ╲                    ╱    ╲  ╱    ╲
╱      ╲╱      ╲                  ╱      ╲╱      ╲
────────────────────               ────────────────────
      ╱──────╲                      ╱╲
     ╱  q(x)  ╲                    ╱  ╲  q(x)
    ╱ (覆盖两峰) ╲                 ╱    ╲ (锁定一峰)
```

### 变分推断选逆向KL

1. **计算可行**：只需从 $q$ 采样（ELBO 里的 $\mathbb{E}_q[\cdot]$），不用从算不出的真后验 $p(z|x)$ 采样；
2. **模型正确**：$q$ 不会在真后验为零处乱放质量，不给出"不可能的解"。

局限：多模态后验下只抓一个峰（逆问题若有多解，会变脸只给一个）。

## ELBO的信息论解释

### 编码视角

回忆"重建+正则"：

$$\text{ELBO} = \mathbb{E}_{q(z)}[\log p(x|z)] - \text{KL}(q(z) \| p(z))$$

- **重建项** $-\mathbb{E}_q[\log p(x|z)]$：给定 $z$ 后用最优编码描述 $x$ 所需比特数（"从 $z$ 重建 $x$ 还差多少信息"）；
- **KL 项** $\text{KL}(q\|p)$：以先验 $p(z)$ 为参考，描述 $q(z)$ 相对先验的额外比特数（"编码 $z$ 比先验多花多少"）。

所以**最大化 ELBO = 最小化总编码长度**——和**最小描述长度（MDL）**原则一致：最好的模型是最能压缩数据的模型。这一转变意味着，"用最简洁的编码把数据说清楚"正是全章"信念刻画从采样转向优化"的落点之一——优化不再只是数值手段，而成了让隐变量表示达到信息效率的自然语言。

### 与率失真理论的联系

| ELBO | 率失真理论 |
|---|---|
| 重建项 $-\mathbb{E}_q[\log p(x\|z)]$ | 失真 $D$（重建误差） |
| KL 项 $\text{KL}(q\|p)$ | 码率 $R$（编码代价） |
| 最大化 ELBO | 率失真优化 $R+\lambda D$ |

$\beta$-VAE（Higgins et al., 2017）把目标写成 $\mathbb{E}_q[\log p(x|z)] - \beta\cdot\text{KL}(q\|p)$，$\beta>1$ 加大对码率的惩罚，逼 $z$ 学更紧凑表示——正是率失真里"加重码率权重"的做法。

### ELBO与互信息

重建项还能连到互信息：

$$\mathbb{E}_{q(z)}[\log p(x|z)] = \mathbb{E}_{q(z,x)}[\log p(x|z)] \geq I_q(X;Z) - H(X)$$

最大化重建项间接最大化了 $X,Z$ 的互信息——即 $z$ 应尽量多携带 $x$ 的信息。这和第 9 章 VAE"编码器要保留信息"的直觉一致。

**来源**：Cover & Thomas (2006) Elements of Information Theory; Minka (2005) Divergence measures and message passing; Higgins et al. (2017) $\beta$-VAE; Alemi et al. (2017) Fixing a Broken ELBO
