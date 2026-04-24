# 附录3B Moreau包络与近端算子的性质

> 定位：3.4节近端算子的数学性质补充，供对优化理论有深入兴趣的读者参考。

## Moreau包络：不可微函数的光滑化

3.4节定义了近端算子 $\text{prox}_{\lambda g}(v) = \arg\min_x \{\frac{1}{2}\|x-v\|^2 + \lambda g(x)\}$。现在考虑近端问题的最优值——**Moreau包络**（Moreau envelope）：

$$g^\lambda(x) = \inf_y \left\{g(y) + \frac{1}{2\lambda}\|y - x\|^2\right\}$$

Moreau包络也称为**Moreau-Yoshida正则化**。它是 $g$ 的一个光滑近似——将不可微的 $g$ "磨光"为可微的 $g^\lambda$。

### 核心性质

1. **光滑性**：若 $g$ 是凸的且正常的，则 $g^\lambda$ 是 $\frac{1}{\lambda}$-光滑的（即梯度Lipschitz常数为 $\frac{1}{\lambda}$）。这意味着即使 $g$ 不可微（如 $\|x\|_1$），$g^\lambda$ 也是处处可微的。

2. **相同极小点**：$g^\lambda$ 和 $g$ 有相同的极小点集——$\arg\min g = \arg\min g^\lambda$。光滑化不改变最优解的位置，只是让优化路径更平滑。

3. **梯度公式**：

$$\nabla g^\lambda(x) = \frac{1}{\lambda}\big(x - \text{prox}_{\lambda g}(x)\big)$$

Moreau包络的梯度由近端算子给出——近端算子不仅提供迭代点，还提供光滑近似函数的梯度信息。

4. **单调逼近**：当 $\lambda \to 0$ 时，$g^\lambda(x) \to g(x)$（逐点收敛）。$\lambda$ 越小，光滑近似越接近原函数，但光滑性越弱（Lipschitz常数越大）。

## Moreau恒等式

**Moreau恒等式**建立了近端算子与其共轭近端算子的关系：

$$\text{prox}_{\lambda g}(x) + \lambda \, \text{prox}_{g^*/\lambda}(x/\lambda) = x$$

其中 $g^*$ 是 $g$ 的Fenchel共轭。

这个恒等式有两层含义：

1. **计算意义**：若 $g$ 的近端算子难以计算但 $g^*$ 的近端算子容易，可以通过Moreau恒等式间接计算——这正是3.5节Chambolle-Pock算法的思路
2. **对偶意义**：近端算子将 $x$ 分解为两部分——原始近端 $\text{prox}_{\lambda g}(x)$ 和对偶近端 $\lambda\,\text{prox}_{g^*/\lambda}(x/\lambda)$，两者互补

### 应用于L1范数

对 $g(x) = \|x\|_1$，$g^*(y) = \iota_{\|\cdot\|_\infty \leq 1}(y)$（无穷范数球的指示函数）。Moreau恒等式给出：

$$\text{prox}_{\lambda\|\cdot\|_1}(x) + \lambda \, \text{proj}_{[-1,1]}(x/\lambda) = x$$

即软阈值 $\mathcal{S}_\lambda(x) = x - \lambda \, \text{proj}_{[-1,1]}(x/\lambda)$——软阈值可以理解为从 $x$ 中减去对偶投影。

## 与第4章ULA的联系

Moreau包络的一个重要应用是构造**MYULA**（Moreau-Yoshida Unadjusted Langevin Algorithm）。

Langevin采样（第4章）需要后验的梯度 $\nabla_x \ln p(x|y)$。当后验含不可微先验时（如Laplace先验→L1正则项），梯度不存在。MYULA的策略是：用Moreau包络 $g^\lambda$ 替换不可微的 $g$，使得后验变为光滑的近似后验：

$$\tilde{p}_\lambda(x|y) \propto p(y|x) \exp(-g^\lambda(x))$$

这个近似后验处处可微，Langevin采样可以直接应用。当 $\lambda \to 0$ 时，近似后验收敛到真实后验——MYULA在光滑性和逼近精度之间通过 $\lambda$ 取得平衡。

这是近端方法与采样方法的一个重要交汇点：**近端算子不仅是优化工具，也是连接优化与采样的桥梁**。
