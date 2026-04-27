# 附录5B Tweedie等式的严格证明

> 定位：为5.3节提供Tweedie等式的完整数学推导。证明虽不复杂，但需要仔细的积分交换和分部积分操作，放在附录以避免打断主线的概念流。

## 一般形式的Tweedie等式

### 定理陈述

**定理（Tweedie等式）**：设 $X \sim p(x)$ 为 $\mathbb{R}^n$ 上的随机变量，$Y = X + \sqrt{\varepsilon}\,Z$，其中 $Z \sim \mathcal{N}(0, I_n)$ 与 $X$ 独立，$\varepsilon > 0$。则 $Y$ 的边际密度为 $p_\varepsilon(y) = (p * \mathcal{N}(0, \varepsilon I))(y)$，且：

$$\nabla_y \log p_\varepsilon(y) = \frac{D_\varepsilon^*(y) - y}{\varepsilon}$$

其中 $D_\varepsilon^*(y) = \mathbb{E}[X|Y = y]$ 是MMSE去噪器。

## 完整证明

### 步骤1：写出边际密度的积分表达式

$Y$ 的边际密度是 $X$ 和 $Z$ 的联合密度的边际化：

$$p_\varepsilon(y) = \int p(x)\,p(z)\,dz = \int p(x)\,\mathcal{N}(y|x, \varepsilon I)\,dx$$

展开高斯密度：

$$p_\varepsilon(y) = \int p(x)\,(2\pi\varepsilon)^{-n/2}\exp\left(-\frac{\|y - x\|^2}{2\varepsilon}\right)dx$$

### 步骤2：对 $y$ 求梯度

对 $y$ 求梯度，可以交换积分与微分（在正则性条件下）：

$$\nabla_y p_\varepsilon(y) = \int p(x)\,\nabla_y\left[\mathcal{N}(y|x, \varepsilon I)\right]dx$$

### 步骤3：计算高斯密度的梯度

高斯密度的梯度为：

$$\nabla_y \mathcal{N}(y|x, \varepsilon I) = \nabla_y\left[(2\pi\varepsilon)^{-n/2}\exp\left(-\frac{\|y - x\|^2}{2\varepsilon}\right)\right]$$

$$= (2\pi\varepsilon)^{-n/2}\exp\left(-\frac{\|y-x\|^2}{2\varepsilon}\right)\cdot\left(-\frac{y-x}{\varepsilon}\right)$$

$$= -\frac{y - x}{\varepsilon}\,\mathcal{N}(y|x, \varepsilon I) = \frac{x - y}{\varepsilon}\,\mathcal{N}(y|x, \varepsilon I)$$

### 步骤4：代入并提取因子

将步骤3的结果代入步骤2：

$$\nabla_y p_\varepsilon(y) = \int p(x)\,\frac{x - y}{\varepsilon}\,\mathcal{N}(y|x, \varepsilon I)\,dx$$

将 $(x - y)/\varepsilon$ 拆开为 $x/\varepsilon - y/\varepsilon$：

$$= \frac{1}{\varepsilon}\int x\,p(x)\,\mathcal{N}(y|x, \varepsilon I)\,dx - \frac{y}{\varepsilon}\int p(x)\,\mathcal{N}(y|x, \varepsilon I)\,dx$$

### 步骤5：识别条件期望

第一个积分：

$$\int x\,p(x)\,\mathcal{N}(y|x, \varepsilon I)\,dx = \int x\,\frac{p(x)\,\mathcal{N}(y|x, \varepsilon I)}{p_\varepsilon(y)}\,dx \cdot p_\varepsilon(y) = D_\varepsilon^*(y)\,p_\varepsilon(y)$$

因为 $\frac{p(x)\,\mathcal{N}(y|x, \varepsilon I)}{p_\varepsilon(y)} = p(x|y)$ 是后验密度，$\int x\,p(x|y)\,dx = \mathbb{E}[X|Y=y] = D_\varepsilon^*(y)$。

第二个积分：

$$\int p(x)\,\mathcal{N}(y|x, \varepsilon I)\,dx = p_\varepsilon(y)$$

这是边际密度的定义。

### 步骤6：整理得到Tweedie等式

代入步骤5的结果：

$$\nabla_y p_\varepsilon(y) = \frac{1}{\varepsilon}\left[D_\varepsilon^*(y)\,p_\varepsilon(y) - y\,p_\varepsilon(y)\right] = \frac{p_\varepsilon(y)}{\varepsilon}\left[D_\varepsilon^*(y) - y\right]$$

两边除以 $p_\varepsilon(y)$：

$$\frac{\nabla_y p_\varepsilon(y)}{p_\varepsilon(y)} = \nabla_y\log p_\varepsilon(y) = \frac{D_\varepsilon^*(y) - y}{\varepsilon} \quad \blacksquare$$

### 证明的关键点

1. **积分与微分的交换**：在 $p_\varepsilon$ 和 $\nabla p_\varepsilon$ 可积的条件下，交换是合法的。对于高斯卷积，这通常成立。
2. **高斯密度的梯度**：$\nabla_y\mathcal{N}(y|x, \varepsilon I) = (x-y)/\varepsilon \cdot \mathcal{N}(y|x, \varepsilon I)$——梯度方向指向均值 $x$，模长与 $(x-y)/\varepsilon$ 成正比。
3. **条件期望的识别**：$\int x\,p(x)\,\mathcal{N}(y|x, \varepsilon I)\,dx = D_\varepsilon^*(y)\,p_\varepsilon(y)$——这是MMSE去噪器的定义。

## 推广：标量Tweedie等式（Efron, 2011）

### 一维指数族形式

Tweedie等式有更一般的标量形式，适用于一维指数族分布。

设 $Y|X \sim e^{XY - \psi(X)}$（自然参数为 $X$ 的指数族），$X \sim p(X)$。则边际分布：

$$p(y) = \int e^{xy - \psi(x)}\,p(x)\,dx$$

**标量Tweedie等式**：

$$\mathbb{E}[X|Y = y] = \frac{d}{dy}\log p(y) + \psi''(\mathbb{E}[X|Y = y])$$

### 高斯情形的特殊化

对于高斯噪声模型 $Y|X \sim \mathcal{N}(X, \varepsilon)$：
- 自然参数：$X/\varepsilon$
- $\psi(x) = x^2/(2\varepsilon)$
- $\psi''(x) = 1/\varepsilon$

代入标量Tweedie等式：

$$\mathbb{E}[X|Y = y] = \frac{d}{dy}\log p_\varepsilon(y) + y$$

整理得：

$$\frac{d}{dy}\log p_\varepsilon(y) = \frac{\mathbb{E}[X|Y=y] - y}{\varepsilon}$$

这与向量形式的Tweedie等式一致——标量形式是向量形式在一维的特例。

### 与经验贝叶斯的联系

Robbins (1956) 最初的动机是**经验贝叶斯**：在不指定先验 $p(x)$ 的情况下，仅从观测数据 $\{y_i\}$ 中估计后验均值 $\mathbb{E}[X|Y = y]$。

Tweedie等式将后验均值与边际密度的梯度联系起来：

$$D_\varepsilon^*(y) = y + \varepsilon\,\nabla\log p_\varepsilon(y)$$

如果可以从数据中估计 $\nabla\log p_\varepsilon(y)$（这正是得分匹配做的事），就可以得到后验均值——这就是经验贝叶斯的思想。

**来源**：Robbins (1956); Miyasawa (1961); Efron (2011)
