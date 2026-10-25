# 附录8A Fenchel共轭与ELBO的优化理论根基

> 定位：为 8.4 节"变分推断与正则化的统一视角"提供 Fenchel 共轭的数学基础，建立 ELBO 与凸分析的联系。读完你会明白：ELBO 并非随意构造的下界，它是凸优化里"对偶问题"的自然产物——也是 8.3 节零强迫/零避免 aha 背后"为什么变分推断选逆向 KL 就天然可算"的统一解释框架。

## 凸共轭（Fenchel共轭）定义

设 $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ 是正常凸函数（proper convex function，定义域非空、向下凸），其**凸共轭**（Fenchel conjugate）定义为：

$$f^*(\mathbf{y}) = \sup_{x \in \mathbb{R}^n} [\langle \mathbf{y}, \mathbf{x} \rangle - f(\mathbf{x})]$$

凸共轭 $f^*$ 永远凸（无论 $f$ 是否凸），因为它是仿射函数族 $\langle \mathbf{y}, \mathbf{x}\rangle - f(\mathbf{x})$ 关于 $\mathbf{y}$ 的上确界，而上确界保凸。

先停下来建立直觉：为什么要给一个函数做这种"共轭"操作？因为很多关于函数的问题（求极值、求下界、写对偶），换个视角看同一件事会简单得多。凸共轭就是把一个函数重新包装成"以斜率为自变量"的另一种描述——$\log p(\mathbf{x})$ 与正则化的统一，正是靠这种"换个自变量"的视角翻转才显现出来的。

### 几何意义

$f^*(\mathbf{y})$ 是：斜率为 $\mathbf{y}$ 的直线 $\langle \mathbf{y}, \mathbf{x}\rangle - c$ 能与 $f(\mathbf{x})$ 贴合时，最大的那个截距 $c$。换句话说，$f^*(\mathbf{y})$ 是 $f$ 的"最优线性下界"的截距——这已经预告了它和"下界"的亲缘关系。

### 经典例子

1. **二次函数**：$f(\mathbf{x})=\frac{1}{2}\|\mathbf{x}\|^2 \to f^*(\mathbf{y})=\frac{1}{2}\|\mathbf{y}\|^2$（自共轭，自己共轭自己）
   - 推导：$\sup_x[\langle \mathbf{y},\mathbf{x}\rangle - \frac{1}{2}\|\mathbf{x}\|^2]$ 在 $\mathbf{x}=\mathbf{y}$ 处取最大，得 $\frac{1}{2}\|\mathbf{y}\|^2$
2. **指数函数**：$f(\mathbf{x})=e^\mathbf{x} \to f^*(\mathbf{y})=\mathbf{y}\log \mathbf{y} - \mathbf{y}\ (\mathbf{y}>0)$，$f^*(0)=0$
3. **L1 范数**：$f(\mathbf{x})=\|\mathbf{x}\|_1 \to f^*(\mathbf{y})=\mathbb{I}_{\|\mathbf{y}\|_\infty\leq 1}$（示性函数：在单位球内为 0，否则 $+\infty$）
4. **负对数**：$f(\mathbf{x})=-\log \mathbf{x}\ (\mathbf{x}>0) \to f^*(\mathbf{y}) = -1-\log(-\mathbf{y})\ (\mathbf{y}<0)$ —— 下面连 ELBO 时会用到

## Fenchel-Young不等式

**定理**：对任意 $\mathbf{x}, \mathbf{y}$，
$$f(\mathbf{x}) + f^*(\mathbf{y}) \geq \langle \mathbf{x}, \mathbf{y} \rangle$$

**证明**：由定义 $f^*(\mathbf{y})=\sup_{x'}[\langle \mathbf{y},\mathbf{x}'\rangle - f(\mathbf{x}')] \geq \langle \mathbf{y},\mathbf{x}\rangle - f(\mathbf{x})$，移项即得。

**取等条件**：$f(\mathbf{x})+f^*(\mathbf{y})=\langle \mathbf{x},\mathbf{y}\rangle$ 当且仅当 $\mathbf{y} \in \partial f(\mathbf{x})$（$\mathbf{y}$ 是 $f$ 在 $\mathbf{x}$ 处的次梯度，subgradient，凸函数在不可导点处的广义梯度）。

改写为 $f(\mathbf{x}) \geq \langle \mathbf{x}, \mathbf{y}\rangle - f^*(\mathbf{y})$：即 $f(\mathbf{x})$ 被一个"线性函数"下界估计，$f^*(\mathbf{y})$ 给出这个下界的最优截距。

## 从Fenchel-Young到ELBO

关键观察：$\log p(\mathbf{x})=\log\int p(\mathbf{x},\mathbf{z})dz$ 的变分表示，和 Fenchel-Young 不等式结构相同。

### 函数空间的 Fenchel 共轭

把 Fenchel 共轭从有限维推广到"分布空间"。定义泛函：

$$F[q] = \mathbb{E}_{q(\mathbf{z})}[\log q(\mathbf{z})] = -H(q)$$

（$F$ 是负熵，定义在概率分布空间上。）其对偶泛函：

$$F^*[g] = \sup_q \big[\mathbb{E}_{q(\mathbf{z})}[g(\mathbf{z})] - F[q]\big] = \sup_q \big[\mathbb{E}_{q(\mathbf{z})}[g(\mathbf{z})] + H(q)\big]$$

令 $g(\mathbf{z})=\log p(\mathbf{x},\mathbf{z})$，则：

$$F^*[\log p(\mathbf{x},\cdot)] = \sup_q \big[\mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x},\mathbf{z})] + H(q)\big] = \sup_q \text{ELBO}(q)$$

由 Fenchel-Young 不等式，当 $q=p(\mathbf{z}|\mathbf{x})$ 取等时：

$$\sup_{q'} \text{ELBO}(q') = \log p(\mathbf{x})$$

因此：

$$\boxed{\log p(\mathbf{x}) = F^*[\log p(\mathbf{x},\cdot)] = \sup_q \text{ELBO}(q)}$$

**ELBO 是 $\log p(\mathbf{x})$ 的 Fenchel 对偶表示**——通过凸共轭获得的下界。变分间隙 $\text{KL}(q\|p(\mathbf{z}|\mathbf{x}))$ 对应 Fenchel-Young 间隙。

到这里，停一下体会这条等式的分量：它把"最大化 ELBO"与"$\log p(\mathbf{x})$ 这个数"之间，安上了一把通用数学语言——对偶。这也正是 8.4 节"统一不是比喻"的真正底气：ELBO 与 $p(\mathbf{x})$ 的关系，和凸优化里原始-对偶的关系，是同一件事。

### 更直接的视角

ELBO 的 Jensen 推导，本身就是 Fenchel-Young 在凹函数情形的特例：$\log$ 是凹函数，对凹函数 $g$ 的 Jensen 是 $\mathbb{E}[g(X)]\leq g(\mathbb{E}[X])$；对凸函数 $f=-g$ 则是 $\mathbb{E}[f(X)]\geq f(\mathbb{E}[X])$——正是 Fenchel-Young 取极限的样子。所以 **Jensen 推导 ELBO 是 Fenchel-Young 的特例**，都来自凸分析同一根基。这一转变意味着，我们在 8.2 节靠 Jensen 得到的结果，并不是孤立的小技巧，而是一条高阶法则（凸对偶）在大分布空间里的投影。

---

## 实验 8A-1：Fenchel共轭的数值验证与ELBO的凸共轭视角

理论推导之后，一个自然的问题是：**Fenchel共轭在数值上如何计算？经典函数的共轭公式是否可靠？ELBO的Fenchel对偶表示是否有实际意义？** 实验 `8A-1.py` 先数值验证经典函数的共轭，再在高斯混合模型上核验 $\sup_q \text{ELBO}(q) = \log p(\mathbf{x})$ 这条恒等式，并用**两个对照配置**把"变分间隙 = $D_{\text{KL}}(q^*\|p(\mathbf{z}|\mathbf{x}))$ = Fenchel-Young间隙"算到小数点后第六位。

### 实验场景

实验分两步：

1. **步骤1：Fenchel共轭的数值验证**——对四个经典函数 $(\mathbf{x}^2/2,\ |\mathbf{x}|,\ -\log \mathbf{x},\ e^\mathbf{x})$ 在指定 $\mathbf{y}$ 处用网格搜索计算 $f^*(\mathbf{y}) = \sup_x [xy - f(\mathbf{x})]$，与理论公式对比。实现细节：格点数取**奇数** 10001，使对称区间必含 $\mathbf{x}=0$；否则 $|\mathbf{x}|$ 这种在 $\mathbf{x}=0$ 处取 $\sup$ 的函数会凭空多出 $\sim 5\times 10^{-4}$ 的格点伪影（早期版本正是如此）。

2. **步骤2：ELBO的Fenchel对偶表示（单峰/双峰后验对照）**——似然 $p(\mathbf{x}|\mathbf{z})=\mathcal{N}(\mathbf{x};\mathbf{z},\sigma_{obs}^2)$，先验为双分量高斯混合，变分族取单高斯 $q(\mathbf{z})=\mathcal{N}(\mu_q,\sigma_q^2)$。用"公共随机数(CRN) + 粗网格初值 + Nelder-Mead"最大化 MC 估计的 ELBO，再用两条**独立**途径复核：Gauss-Hermite 积分算精确 ELBO、梯形积分直接算 $D_{\text{KL}}(q^*\|p(\mathbf{z}|\mathbf{x}))$。两个配置的区别在后验形态：

   - **配置A（原设定）**：$p(\mathbf{z})=0.3\,\mathcal{N}(-2,1)+0.7\,\mathcal{N}(1,1)$，$\sigma_{obs}=0.5$，$\mathbf{x}=0.5$；
   - **配置B（强分离 + 紧似然）**：$p(\mathbf{z})=0.3\,\mathcal{N}(-2.5,0.5^2)+0.7\,\mathcal{N}(2.5,0.5^2)$，$\sigma_{obs}=0.4$，$\mathbf{x}=0$。

### 实验目的

1. **验证Fenchel共轭的数值计算**：网格搜索得到的 $f^*(\mathbf{y})$ 与理论公式吻合，实测最大误差 $1.1\times10^{-6}$（格点离散误差量级）。
2. **理解经典函数的共轭公式**：二次函数（自共轭）、绝对值（示性函数，注意其定义域限制）、负对数（与熵相关）、指数函数（与KL散度相关）。
3. **验证ELBO的下界性质与间隙身份**：$\sup_q \text{ELBO}(q) \le \log p(\mathbf{x})$，且间隙数值上等于 $D_{\text{KL}}(q^*\|p(\mathbf{z}|\mathbf{x}))$。
4. **看清"族受限"到底何时才产生大间隙**：先验双峰并不等于后验双峰，间隙量级由后验的真实形态决定。
5. **连接凸分析与变分推断**：ELBO的"重建+正则"结构有凸分析的数学根基——Fenchel共轭是统一的数学语言。

### 实验结果

#### 步骤1：Fenchel共轭的数值验证

![实验8A-1 步骤1：Fenchel共轭的几何意义](<./第八章 配套实验/实验8A-1/步骤1_fenchel共轭几何.png>)

图中每条红色虚线都是相应 $f$ 的**最优线性下界** $\ell(\mathbf{x})=\langle \mathbf{y},\mathbf{x}\rangle-f^*(\mathbf{y})$，切点满足 $\mathbf{y} \in \partial f(\mathbf{x}^*)$（次梯度取等条件）；右下角板则显示 $|\mathbf{y}|>1$ 时根本不存在这样的支撑线。

| 函数 $f(\mathbf{x})$ | $f^*(\mathbf{y})$ 理论公式 | 理论值 | 数值结果 | 误差 |
|:------------|:------------------|:------:|:--------:|:-----|
| $\mathbf{x}^2/2\ (\mathbf{y}=2)$ | $\mathbf{y}^2/2$ | 2.000000 | 2.000000 | $0$（$\mathbf{x}^*=2.000$） |
| $\|\mathbf{x}\|\ (\mathbf{y}=0.5)$ | $0$（$\|\mathbf{y}\|\le1$） | 0 | 0.000000 | $0$（$\mathbf{x}^*=0$） |
| $\|\mathbf{x}\|\ (\mathbf{y}=1.5)$ | $+\infty$（$\|\mathbf{y}\|>1$） | $+\infty$ | 5.000000（截断） | N/A（有限域截断） |
| $-\log \mathbf{x}\ (\mathbf{y}=-1.5)$ | $-1-\log(-\mathbf{y})$ | -1.405465 | -1.405466 | $1.1\times10^{-6}$ |
| $e^\mathbf{x}\ (\mathbf{y}=2)$ | $\mathbf{y}\log \mathbf{y}-\mathbf{y}$ | -0.613706 | -0.613706 | $1.2\times10^{-7}$ |

**验证小结**：

1. **有限共轭情形**：$\mathbf{x}^2/2$、$-\log \mathbf{x}$、$e^\mathbf{x}$ 与 $\|\mathbf{x}\|(|\mathbf{y}|\le1)$ 的数值解与理论值吻合，实测最大误差 $1.11\times10^{-6}$。注意 $e^\mathbf{x}$ 在 $\mathbf{y}=2$ 处的共轭值是**负的**（$2\log 2-2=-0.614$），$-\log \mathbf{x}$ 在 $\mathbf{y}=-1.5$ 处是 $-1.405$ 而非 $-0.405$——手算时最容易在这两处丢掉常数项。
2. **示性函数情形（固有属性，非实现缺陷）**：$|\mathbf{y}|\le1$ 时数值结果为 $0$（正确）；$|\mathbf{y}|>1$ 时理论值为 $+\infty$，有限区间上的网格搜索只能返回端点处的有限值。代码不拿这个有限值去和 $+\infty$ 比误差，而是改用**扩容实验**实证：

   | `x_range` 上界 | 10 | 100 | 1000 | 10000 |
   |:---|:---|:---|:---|:---|
   | 数值 $\sup$ | 5.0 | 50.0 | 500.0 | 5000.0 |
   | 解析预测 $hi\cdot(\|\mathbf{y}\|-1)$ | 5.0 | 50.0 | 500.0 | 5000.0 |

   数值解随区间单调增长、永不收敛，且逐点等于 $hi\cdot(|\mathbf{y}|-1)$——这正是 $+\infty$ 的正确数值表现。无法用有限域搜索触及 $+\infty$ 是**工具的定义域限制**（共轭的自然定义域 $\mathrm{dom}\,f^*=[-1,1]$ 之外），不是数值方法与理论的矛盾。

#### 步骤2：ELBO的Fenchel对偶表示

![实验8A-1 步骤2：变分间隙与ELBO地形](<./第八章 配套实验/实验8A-1/步骤2_变分间隙与elbo地形.png>)

**证据与后验形态**（$\log p(\mathbf{x})$ 用数值积分与混合边际闭式两法独立计算，配置A 相差 $2.2\times10^{-16}$，配置B 相差 $0$）：

| 配置 | $\log p(\mathbf{x})$ | 后验分量权重 | 后验分量均值 | 分量 std | 间距 vs $2\sigma$ | 实测局部极大 |
|:-----|:-----------:|:------------:|:------------:|:--------:|:-----------------:|:-------------|
| A | -1.449043 | [0.0374, 0.9626] | [0.000, 0.600] | 0.4472 | 0.600 ≤ 0.894 → **单峰** | 1 个 @ $\mathbf{z}=0.591$ |
| B | -8.095091 | [0.3000, 0.7000] | [-0.976, 0.976] | 0.3123 | 1.951 > 0.625 → **双峰** | 2 个 @ $\mathbf{z}=\pm0.976$ |

**单高斯变分族的优化结果**（MC-CRN 求解，解析 ELBO 与 KL 独立复核）：

| 配置 | 最优 $q^*$ | ELBO(MC) | ELBO(解析) | 变分间隙（按解析） | $D_{\text{KL}}(q^*\|p(\mathbf{z}|\mathbf{x}))$ 直积分 | 两法之差 |
|:-----|:-----------|:---------:|:-----------:|:--------:|:--------------------------------:|:--------:|
| A | $\mathcal{N}(0.5775, 0.4607^2)$ | -1.449505 | -1.449523 | 0.000480 | 0.000480 | $1.8\times10^{-16}$ |
| B | $\mathcal{N}(0.9737, 0.3156^2)$ | -8.449862 | -8.450043 | 0.354952 | 0.354952 | $2.1\times10^{-9}$ |

> **口径说明**：最后两列（变分间隙、直积分）都对齐**解析** ELBO，所以“两法之差”只反映积分误差（$10^{-16}$、$10^{-9}$）。若改用 MC 列算间隙，A/B 分别是 $0.000462$ / $0.354771$，比解析值小 $1.75\times10^{-5}$ / $1.81\times10^{-4}$——即观察 5 所说的 CRN 过拟合方向。引用时应以解析列为准。

配置B 还存在一个**次优盆地**（右图青色星号）：$q_{\text{次优}}=\mathcal{N}(-0.9704, 0.3201^2)$，ELBO $=-9.294462$，间隙 $1.199371$，比全局最优低 $0.8444$。

**核心观察**：

1. **ELBO 作为下界**：两配置均满足单高斯族上的 $\sup_q \text{ELBO}(q) \le \log p(\mathbf{x})$（A：$-1.449523 \le -1.449043$；B：$-8.450043 \le -8.095091$），且从未出现"MC 估出的 ELBO 超过 $\log p(\mathbf{x})$"的越界。
2. **间隙 = Fenchel-Young间隙 = KL，数值闭合**：用与 ELBO 完全无关的梯形积分直接算 $D_{\text{KL}}(q^*\|p(\mathbf{z}|\mathbf{x}))$，与 $\log p(\mathbf{x})-\text{ELBO}(q^*)$ 吻合到 $10^{-16}$（A）与 $10^{-9}$（B）。这把"变分间隙对应 Fenchel-Young 间隙"从类比变成了可验证的等式。
3. **先验双峰 ≠ 后验双峰**：配置A 的先验是双峰，但 $\mathbf{x}=0.5$ 的观测把后验压成**密度上的单峰**（分量间距 $0.60 < 2\sigma_{\text{post}} = 0.89$，实测只有 1 个局部极大），于是单高斯族几乎达成强对偶，间隙仅 $4.8\times10^{-4}$，且这点残余只来自后验的轻微左偏（左分量占 $3.7\%$ 质量）。这提醒我们：**"受限族必然留下大间隙"不能由"先验是混合"直接推出，必须落到后验的真实形态上验证**。
4. **配置B 才是教科书式的族受限间隙**：后验真正双峰，单高斯只能"二选一"（左图灰色区域即被放弃的左峰质量），代价约为 $-\log \pi_{\max} = -\log 0.7 = 0.3567$，实测间隙 $0.3550$（残差 $0.0017$ 是 $q^*$ 与所锁分量之间的高斯形状失配）；次优盆地比全局最优低 $0.8444$，与质量比预测 $\log(\pi_{\max}/\pi_{\min}) = \log(0.7/0.3) = 0.8473$ 吻合。这正是 8.3 节"逆向 KL 零强迫/mode-seeking"的**定量价格标签**：放弃一个模式，就要付 $\log$ 质量比那么多个 nat。
5. **MC-CRN 的诚实说明**：CRN 把随机目标变成确定性目标，但"在固定的 20 万样本上再做优化"仍有轻微过拟合，使 MC 报告的间隙比解析间隙略小（A 差 $1.75\times10^{-5}$，B 差 $1.81\times10^{-4}$）。量级不影响任何结论，但引用 MC 数值时应以解析复核值为准。
6. **凸分析与变分推断的统一**：$\log p(\mathbf{x}) = F^*[\log p(\mathbf{x},\cdot)] = \sup_q \text{ELBO}(q)$ 来自 Fenchel 共轭定义本身（一次共轭），而非双重共轭（Fenchel-Moreau）；当变分族 $\mathcal{Q}$ 足够大（包含 $p(\mathbf{z}|\mathbf{x})$）时间隙为零（强对偶）。

**产出物**：`步骤1_fenchel共轭几何.png`、`步骤2_变分间隙与elbo地形.png`、`results_summary.json`（含全部实测数值，seed=42 可逐位复现）。

---

## 变分正则化的凸分析视角

Fenchel 共轭也给变分正则化统一视角。

### Tikhonov正则化的对偶

$$\min_x \frac{1}{2}\|Ax - \mathbf{y}\|^2 + \frac{\lambda}{2}\|\mathbf{x}\|^2$$

由 Fenchel 对偶理论，其对偶问题为：

$$\max_p \left[-\frac{1}{2}\|A^*p\|^2 - \frac{1}{2\lambda}\|p - \mathbf{y}/\lambda\|^2 + \frac{1}{2\lambda}\|\mathbf{y}\|^2\right]$$

原始-对偶间隙 = 原始值 - 对偶值 $\geq 0$（弱对偶）。强对偶时间隙为零。

### 概率框架 vs 优化框架的对偶

| 概率框架 | 优化框架 |
|---|---|
| $\log p(\mathbf{x})$ | 原始问题最优值 |
| ELBO | 对偶问题值 |
| KL间隙 | 对偶间隙 |
| $q = p(\mathbf{z}|\mathbf{x})$ 时间隙为零 | 强对偶时间隙为零 |

两者共享同一套凸分析工具——Fenchel 共轭是统一数学语言。这解释了为何 ELBO 与正则化有相同"数据+正则"结构：**它们是同一个凸优化问题的不同表达**。这一转变意味着，第 2 章的 MAP 正则化作图和本章的 ELBO 变分推断，并不是两条不相干的支流，而是一条凸分析大河的两岸。

**来源**：Benning L2 P23-25; LectureNotes2020_v2 Ch4 (Fenchel conjugate, lines 1832-1979); @rockafellar1970convex Convex Analysis; @borwein2010convex Convex Analysis and Nonlinear Optimization