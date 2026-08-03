# 附录5B Tweedie 等式的严格证明

> 定位：为 5.3 节提供 Tweedie 等式的完整数学推导。证明虽不复杂，但需要仔细的积分交换和分部积分，放附录以免打断主线概念流。

**你会在本附录看到**：5.3 节那个反直觉等式 $\nabla\log p_\varepsilon(y)=(D_\varepsilon^*(y)-y)/\varepsilon$ 是怎么从边际密度的定义一步步推出来的，以及它在 Robbins 经验贝叶斯框架下的更一般形式。

---

## 一般形式的 Tweedie 等式

### 定理陈述

**定理（Tweedie 等式）**：设 $X\sim p(x)$ 为 $\mathbb{R}^n$ 上随机变量，$Y=X+\sqrt{\varepsilon}\,Z$，其中 $Z\sim\mathcal{N}(0,I_n)$ 与 $X$ 独立，$\varepsilon>0$。则 $Y$ 的边际密度 $p_\varepsilon(y)=(p*\mathcal{N}(0,\varepsilon I))(y)$，且：

$$\nabla_y \log p_\varepsilon(y) = \frac{D_\varepsilon^*(y) - y}{\varepsilon}$$

其中 $D_\varepsilon^*(y)=\mathbb{E}[X|Y=y]$ 是 MMSE 去噪器。

---

## 完整证明

### 步骤1：写出边际密度的积分表达式

$Y$ 的边际密度是 $X$ 和 $Z$ 联合密度的边际化：

$$p_\varepsilon(y) = \int p(x)\,p(z)\,dz = \int p(x)\,\mathcal{N}(y|x, \varepsilon I)\,dx$$

展开高斯密度：

$$p_\varepsilon(y) = \int p(x)\,(2\pi\varepsilon)^{-n/2}\exp\left(-\frac{\|y - x\|^2}{2\varepsilon}\right)dx$$

### 步骤2：对 $y$ 求梯度

对 $y$ 求梯度，在正则性条件下可交换积分与微分：

$$\nabla_y p_\varepsilon(y) = \int p(x)\,\nabla_y\left[\mathcal{N}(y|x, \varepsilon I)\right]dx$$

### 步骤3：计算高斯密度的梯度

$$\nabla_y \mathcal{N}(y|x, \varepsilon I) = (2\pi\varepsilon)^{-n/2}\exp\left(-\frac{\|y-x\|^2}{2\varepsilon}\right)\cdot\left(-\frac{y-x}{\varepsilon}\right) = \frac{x - y}{\varepsilon}\,\mathcal{N}(y|x, \varepsilon I)$$

### 步骤4：代入并提取因子

代入步骤2：

$$\nabla_y p_\varepsilon(y) = \int p(x)\,\frac{x - y}{\varepsilon}\,\mathcal{N}(y|x, \varepsilon I)\,dx$$

拆开 $(x-y)/\varepsilon = x/\varepsilon - y/\varepsilon$：

$$= \frac{1}{\varepsilon}\int x\,p(x)\,\mathcal{N}(y|x, \varepsilon I)\,dx - \frac{y}{\varepsilon}\int p(x)\,\mathcal{N}(y|x, \varepsilon I)\,dx$$

### 步骤5：识别条件期望

第一个积分：

$$\int x\,p(x)\,\mathcal{N}(y|x, \varepsilon I)\,dx = \int x\,\frac{p(x)\,\mathcal{N}(y|x, \varepsilon I)}{p_\varepsilon(y)}\,dx \cdot p_\varepsilon(y) = D_\varepsilon^*(y)\,p_\varepsilon(y)$$

因为 $\frac{p(x)\mathcal{N}(y|x,\varepsilon I)}{p_\varepsilon(y)}=p(x|y)$ 是后验密度，$\int x\,p(x|y)\,dx=\mathbb{E}[X|Y=y]=D_\varepsilon^*(y)$。

第二个积分：

$$\int p(x)\,\mathcal{N}(y|x, \varepsilon I)\,dx = p_\varepsilon(y)$$

这是边际密度定义。

### 步骤6：整理得到 Tweedie 等式

代入步骤5：

$$\nabla_y p_\varepsilon(y) = \frac{1}{\varepsilon}\left[D_\varepsilon^*(y)\,p_\varepsilon(y) - y\,p_\varepsilon(y)\right] = \frac{p_\varepsilon(y)}{\varepsilon}\left[D_\varepsilon^*(y) - y\right]$$

两边除以 $p_\varepsilon(y)$：

$$\frac{\nabla_y p_\varepsilon(y)}{p_\varepsilon(y)} = \nabla_y\log p_\varepsilon(y) = \frac{D_\varepsilon^*(y) - y}{\varepsilon} \quad \blacksquare$$

### 证明的关键点

1. **积分与微分的交换**：在 $p_\varepsilon$ 和 $\nabla p_\varepsilon$ 可积条件下合法；高斯卷积通常满足。
2. **高斯密度的梯度**：$\nabla_y\mathcal{N}(y|x,\varepsilon I)=(x-y)/\varepsilon\cdot\mathcal{N}(y|x,\varepsilon I)$——梯度指向均值 $x$，模长正比于 $(x-y)/\varepsilon$。
3. **条件期望的识别**：$\int x\,p(x)\mathcal{N}(y|x,\varepsilon I)\,dx = D_\varepsilon^*(y)\,p_\varepsilon(y)$——正是 MMSE 去噪器定义。

---

## 推广：标量 Tweedie 等式（Efron, 2011）

### 一维指数族形式

设 $Y|X\sim e^{XY-\psi(X)}$（自然参数 $X$ 的指数族），$X\sim p(X)$。边际：

$$p(y) = \int e^{xy - \psi(x)}\,p(x)\,dx$$

**标量 Tweedie 等式**：

$$\mathbb{E}[X|Y = y] = \frac{d}{dy}\log p(y) + \psi''(\mathbb{E}[X|Y = y])$$

### 高斯情形的特殊化

对高斯噪声 $Y|X\sim\mathcal{N}(X,\varepsilon)$：自然参数 $X/\varepsilon$，$\psi(x)=x^2/(2\varepsilon)$，$\psi''(x)=1/\varepsilon$。代入：

$$\mathbb{E}[X|Y = y] = \frac{d}{dy}\log p_\varepsilon(y) + y$$

整理得 $\frac{d}{dy}\log p_\varepsilon(y) = \frac{\mathbb{E}[X|Y=y]-y}{\varepsilon}$，与向量形式一致——标量形式是一维特例。

### 与经验贝叶斯的联系

Robbins (1956) 最初动机是**经验贝叶斯**：不指定先验 $p(x)$，仅从观测 $\{y_i\}$ 估后验均值 $\mathbb{E}[X|Y=y]$。Tweedie 把后验均值和边际密度梯度联系起来：$D_\varepsilon^*(y)=y+\varepsilon\,\nabla\log p_\varepsilon(y)$。若能从数据估 $\nabla\log p_\varepsilon(y)$（这正是得分匹配做的），就能得后验均值——这就是经验贝叶斯思想。

**来源**：Robbins (1956); Miyasawa (1961); Efron (2011)
