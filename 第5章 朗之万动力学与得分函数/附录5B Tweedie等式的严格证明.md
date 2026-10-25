# 附录5B Tweedie 等式的严格证明

> 定位：为 5.3 节提供 Tweedie 等式的完整数学推导。证明虽不复杂，但需要仔细的积分交换和分部积分，放附录以免打断主线概念流。

**你会在本附录看到**：5.3 节那个反直觉等式 $\nabla\log p_\varepsilon(\mathbf{y})=(D_\varepsilon^*(\mathbf{y})-\mathbf{y})/\varepsilon$ 是怎么从边际密度的定义一步步推出来的，以及它在 Robbins 经验贝叶斯框架下的更一般形式。

在翻开证明之前，先交代这几行演算之所以值得逐字符核对的理由。5.3 节把结论摆在了台面上，但一个"反直觉却正确"的等式，最怕的是凭着直觉相信、凭着马虎验证。下面的每一步都负责一个环节的严谨性：步骤 1–2 保证"我们操作的对象（边际密度）确实如定义所示"，步骤 3–4 保证"那条正好抵消神秘系数的梯度确实存在且合法"，步骤 5 则是全片的高潮——把积分识破为条件期望（即 MMSE 去噪器）。整篇读下来，你会看清 Tweedie 等式并非巧合的因式分解，而是后验均值被边际密度的对数梯度"藏起来"又"引出来"的一次精准归还。

---

## 一般形式的 Tweedie 等式

### 定理陈述

**定理（Tweedie 等式）**：设 $X\sim p(\mathbf{x})$ 为 $\mathbb{R}^n$ 上随机变量，$Y=X+\sqrt{\varepsilon}\,Z$，其中 $Z\sim\mathcal{N}(0,I_n)$ 与 $X$ 独立，$\varepsilon>0$。则 $Y$ 的边际密度 $p_\varepsilon(\mathbf{y})=(p*\mathcal{N}(0,\varepsilon I))(\mathbf{y})$，且：

$$\nabla_y \log p_\varepsilon(\mathbf{y}) = \frac{D_\varepsilon^*(\mathbf{y}) - \mathbf{y}}{\varepsilon}$$

其中 $D_\varepsilon^*(\mathbf{y})=\mathbb{E}[X|Y=\mathbf{y}]$ 是 MMSE 去噪器。

先在符号上把对象钉死，防止后面的常用陷阱：等式左边的得分属于**噪声扰动分布** $p_\varepsilon=(p*\mathcal{N}(0,\varepsilon I))$（先验与噪声的卷积），绝不是先验 $p$ 本身；等式右边的系数是噪声方差 $\varepsilon$（即 $\sqrt{\varepsilon}$ 的平方），不是 $\sqrt{\varepsilon}$。这一对象与系数上的约定，与 5.3 节保持一致，是整个等式成立与否的命门。

---

## 完整证明

### 步骤1：写出边际密度的积分表达式

$Y$ 的边际密度是 $X$ 和 $Z$ 联合密度的边际化：

$$p_\varepsilon(\mathbf{y}) = \int p(\mathbf{x})\,\mathcal{N}(\mathbf{y}\mid\mathbf{x}, \varepsilon I)\,dx = \int\!\!\int p(\mathbf{x})\,\mathcal{N}(\mathbf{z};0,I)\,\delta(\mathbf{y}-\mathbf{x}-\sqrt{\varepsilon}\,\mathbf{z})\,d\mathbf{x}\,d\mathbf{z}$$

展开高斯密度：

$$p_\varepsilon(\mathbf{y}) = \int p(\mathbf{x})\,(2\pi\varepsilon)^{-n/2}\exp\left(-\frac{\|\mathbf{y} - \mathbf{x}\|^2}{2\varepsilon}\right)dx$$

第一步确立出发点是边际化：$Y$ 是"干净的 $X$ 加上噪声"，所以它的密度是先将 $X$ 关于自身的分布平均化、再把高斯核 $\mathcal{N}(\mathbf{y}|\mathbf{x},\varepsilon I)$ 卷入——卷积正是这种"模糊化"的代数表达。

### 步骤2：对 $\mathbf{y}$ 求梯度

对 $\mathbf{y}$ 求梯度，在正则性条件下可交换积分与微分：

$$\nabla_y p_\varepsilon(\mathbf{y}) = \int p(\mathbf{x})\,\nabla_y\left[\mathcal{N}(\mathbf{y}|\mathbf{x}, \varepsilon I)\right]dx$$

这里"积分与求导可交换"是一条需要正则性托底但常被默认成立的步骤：只要交掉的积分收敛得足够好（高斯卷积通常满足），我们就可以合法地把梯度放进积分号里，这一合法性由高斯核的快速衰减自动背书。

### 步骤3：计算高斯密度的梯度

$$\nabla_y \mathcal{N}(\mathbf{y}|\mathbf{x}, \varepsilon I) = (2\pi\varepsilon)^{-n/2}\exp\left(-\frac{\|\mathbf{y}-\mathbf{x}\|^2}{2\varepsilon}\right)\cdot\left(-\frac{y-\mathbf{x}}{\varepsilon}\right) = \frac{x - \mathbf{y}}{\varepsilon}\,\mathcal{N}(\mathbf{y}|\mathbf{x}, \varepsilon I)$$

关键的一步藏着核心的"巧合"来源：指数函数的导数会把 $- (\mathbf{y}-\mathbf{x})/\varepsilon$ 拿到眼前，恰好多出一个 $\varepsilon^{-1}(\mathbf{x}-\mathbf{y})$ 因子。这一因子并不多余——它正是后面"除以噪声方差"那一刀的原型，也是高斯密度（光滑而指数尾）所独有的恩赐。

### 步骤4：代入并提取因子

代入步骤2：

$$\nabla_y p_\varepsilon(\mathbf{y}) = \int p(\mathbf{x})\,\frac{x - \mathbf{y}}{\varepsilon}\,\mathcal{N}(\mathbf{y}|\mathbf{x}, \varepsilon I)\,dx$$

拆开 $(\mathbf{x}-\mathbf{y})/\varepsilon = \mathbf{x}/\varepsilon - \mathbf{y}/\varepsilon$：

$$= \frac{1}{\varepsilon}\int \mathbf{x}\,p(\mathbf{x})\,\mathcal{N}(\mathbf{y}|\mathbf{x}, \varepsilon I)\,dx - \frac{y}{\varepsilon}\int p(\mathbf{x})\,\mathcal{N}(\mathbf{y}|\mathbf{x}, \varepsilon I)\,dx$$

这一步纯属代数整理，却把两件事分清了：左边是"带 $\mathbf{x}$ 的加权积分"（指向均值方向），右边是"不带 $\mathbf{x}$ 的纯概率质量"（即 $\mathbf{y}$ 的边际密度 $\varepsilon^{-1}y\,p_\varepsilon(\mathbf{y})$）。Tweedie 等式本质就是前者被"识别"为后验均值的过程。

### 步骤5：识别条件期望

第一个积分：

$$\int \mathbf{x}\,p(\mathbf{x})\,\mathcal{N}(\mathbf{y}|\mathbf{x}, \varepsilon I)\,dx = \int \mathbf{x}\,\frac{p(\mathbf{x})\,\mathcal{N}(\mathbf{y}|\mathbf{x}, \varepsilon I)}{p_\varepsilon(\mathbf{y})}\,dx \cdot p_\varepsilon(\mathbf{y}) = D_\varepsilon^*(\mathbf{y})\,p_\varepsilon(\mathbf{y})$$

因为 $\frac{p(\mathbf{x})\mathcal{N}(\mathbf{y}|\mathbf{x},\varepsilon I)}{p_\varepsilon(\mathbf{y})}=p(\mathbf{x}|\mathbf{y})$ 是后验密度，$\int \mathbf{x}\,p(\mathbf{x}|\mathbf{y})\,dx=\mathbb{E}[X|Y=\mathbf{y}]=D_\varepsilon^*(\mathbf{y})$。

第二个积分：

$$\int p(\mathbf{x})\,\mathcal{N}(\mathbf{y}|\mathbf{x}, \varepsilon I)\,dx = p_\varepsilon(\mathbf{y})$$

这是边际密度定义。

第5步是整篇证明的心脏：我们看到 $\frac{p(\mathbf{x})\mathcal{N}(\mathbf{y}|\mathbf{x},\varepsilon I)}{p_\varepsilon(\mathbf{y})}$ 恰好就是贝叶斯后验密度 $p(\mathbf{x}|\mathbf{y})$（分子是联合密度，分母是边际密度），于是"带 $\mathbf{x}$ 的积分"不再是抽象的泛函，而是后验均值 $\mathbb{E}[X|Y=\mathbf{y}]$——正是 MMSE 去噪器 $D_\varepsilon^*(\mathbf{y})$。边际密度的梯度里，藏着一个去噪器，这既是 5.3 节那句"去噪器给出得分"的源头，也是整个等式在直觉上最出人意料、却又最顺理成章的一跳。

### 步骤6：整理得到 Tweedie 等式

代入步骤5：

$$\nabla_y p_\varepsilon(\mathbf{y}) = \frac{1}{\varepsilon}\left[D_\varepsilon^*(\mathbf{y})\,p_\varepsilon(\mathbf{y}) - \mathbf{y}\,p_\varepsilon(\mathbf{y})\right] = \frac{p_\varepsilon(\mathbf{y})}{\varepsilon}\left[D_\varepsilon^*(\mathbf{y}) - \mathbf{y}\right]$$

两边除以 $p_\varepsilon(\mathbf{y})$：

$$\frac{\nabla_y p_\varepsilon(\mathbf{y})}{p_\varepsilon(\mathbf{y})} = \nabla_y\log p_\varepsilon(\mathbf{y}) = \frac{D_\varepsilon^*(\mathbf{y}) - \mathbf{y}}{\varepsilon} \quad \blacksquare$$

最后这一刀，把"$\nabla p_\varepsilon$"转写为"$\nabla\log p_\varepsilon=\nabla p_\varepsilon/p_\varepsilon$"，正是全书反复强调的得分函数的绝招——用"对数梯度"消去归一化常数。除以 $p_\varepsilon$ 不只是形式上的化简，它让等式两端都不再依赖任何求不出的积分常数，从而可以仅凭去噪器的输出 $D_\varepsilon^*(\mathbf{y})$ 直接计算得分。

### 证明的关键点

1. **积分与微分的交换**：在 $p_\varepsilon$ 和 $\nabla p_\varepsilon$ 可积条件下合法；高斯卷积通常满足。
2. **高斯密度的梯度**：$\nabla_y\mathcal{N}(\mathbf{y}|\mathbf{x},\varepsilon I)=(\mathbf{x}-\mathbf{y})/\varepsilon\cdot\mathcal{N}(\mathbf{y}|\mathbf{x},\varepsilon I)$——梯度指向均值 $\mathbf{x}$，模长正比于 $(\mathbf{x}-\mathbf{y})/\varepsilon$。
3. **条件期望的识别**：$\int \mathbf{x}\,p(\mathbf{x})\mathcal{N}(\mathbf{y}|\mathbf{x},\varepsilon I)\,dx = D_\varepsilon^*(\mathbf{y})\,p_\varepsilon(\mathbf{y})$——正是 MMSE 去噪器定义。

从更深层次看，这三条关键点的分工其实刻画了一条认知闭环：可交换性保证"我们能算"，高斯梯度给出"算出来的形"，条件期望识别则揭示"算出来的原来是去噪器"。Tweedie 等式之所以能搭起"去噪"与"采样"的桥，正因为这套证明把"对密度的求导"翻译成了"对数据的取均值"——统计计算的新旧语言在这里完成了无缝互译。

---

## 推广：标量 Tweedie 等式[@efron2011tweedie]

### 一维指数族形式

设 $Y|\theta\sim h(y)\,e^{\theta y-\psi(\theta)}$（$\theta$ 为自然参数，$h$ 为基测度），$\theta\sim p(\theta)$。边际：

$$p(y) = \int h(y)\,e^{\theta y - \psi(\theta)}\,p(\theta)\,d\theta \equiv h(y)\,Z(y)$$

**标量 Tweedie 等式**：

$$\mathbb{E}[\theta|Y = y] = \frac{d}{dy}\log p(y) - \frac{d}{dy}\log h(y)$$

（推导一句话：$Z'(y)/Z(y)=\mathbb{E}[\theta|Y=y]$，而 $\log p=\log h+\log Z$，移项即得。）

### 高斯情形的特殊化

对高斯噪声 $Y|X\sim\mathcal{N}(X,\varepsilon)$：自然参数 $\theta=X/\varepsilon$，$\psi(\theta)=\varepsilon\theta^2/2$，基测度 $h(y)=e^{-y^2/(2\varepsilon)}/\sqrt{2\pi\varepsilon}$，故 $\frac{d}{dy}\log h(y)=-y/\varepsilon$。代入：

$$\mathbb{E}[\theta|Y = y] = \frac{d}{dy}\log p_\varepsilon(y) + \frac{y}{\varepsilon}$$

两边乘 $\varepsilon$，并注意 $\mathbb{E}[\theta|Y=y]=\mathbb{E}[X|Y=y]/\varepsilon$：

$$\mathbb{E}[X|Y = y] = y + \varepsilon\,\frac{d}{dy}\log p_\varepsilon(y)$$

整理得 $\frac{d}{dy}\log p_\varepsilon(y) = \frac{\mathbb{E}[X|Y=y]-y}{\varepsilon}$，与向量形式一致——标量形式是一维特例。

反过来看，高斯情形只是一般指数族 Tweedie 等式的一个特例，这说明该等式并非高斯噪声的孤例：只要似然属于自然参数指数族，后验均值与边际密度梯度之间就有同构的联系，高斯噪声只是其中基测度导数 $\frac{d}{dy}\log h(y)=-y/\varepsilon$ 为线性、形式最干净的一员。

### 与经验贝叶斯的联系

@robbins1956robayes 最初动机是**经验贝叶斯**：不指定先验 $p(\mathbf{x})$，仅从观测 $\{y_i\}$ 估后验均值 $\mathbb{E}[X|Y=\mathbf{y}]$。Tweedie 把后验均值和边际密度梯度联系起来：$D_\varepsilon^*(\mathbf{y})=\mathbf{y}+\varepsilon\,\nabla\log p_\varepsilon(\mathbf{y})$。若能从数据估 $\nabla\log p_\varepsilon(\mathbf{y})$（这正是得分匹配做的），就能得后验均值——这就是经验贝叶斯思想。

这一转变意味着，Tweedie 等式在统计史上的出身——经验贝叶斯——与它在本书的用途（从去噪器到得分）其实共享同一灵魂：既不写出 $p(\mathbf{x})$，也不计算归一化常数 $Z$，只从"数据或去噪器中能得到的量"出发，反推出后验均值。于是"绕过 $Z$"这句话，既是统计推断的古老智慧，也在现代去噪器身上获得了新的载体。
