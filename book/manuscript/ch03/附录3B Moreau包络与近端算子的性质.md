# 附录3B Moreau包络与近端算子的性质

> *"不可微的函数，能不能被'磨光'成可微的？"* —— Moreau 包络就是干这个的"砂纸"。

**定位**：3.4 节近端算子的数学性质补充，供对优化理论有深入兴趣的读者参考。

---

## Moreau包络：不可微函数的光滑化

3.4 定义了近端算子 $\text{prox}_{\lambda g}(v)=\arg\min_x\{\frac12\|x-v\|^2+\lambda g(x)\}$。现在看近端问题的最优值——**Moreau 包络**（Moreau-Yoshida 正则化）：

$$g^\lambda(x) = \inf_y \left\{g(y) + \frac{1}{2\lambda}\|y - x\|^2\right\}$$

它是 $g$ 的光滑近似——把不可微的 $g$"磨光"成可微的 $g^\lambda$。直观理解：$g^\lambda(x)$ 是在 $x$ 附近"加权"地取 $g$ 的最小值，权重随距 $x$ 的距离衰减；$\lambda$ 越大，越多的远处信息被纳入平均，曲线越平滑，但离原函数越远。

### 核心性质

1. **光滑性**：若 $g$ 凸且正常，则 $g^\lambda$ 是 $\frac{1}{\lambda}$-光滑的（$\nabla$ 的 Lipschitz 常数为 $\frac{1}{\lambda}$）。即使 $g$ 不可微（如 $\|x\|_1$），$g^\lambda$ 处处可微。
2. **相同极小点**：$g^\lambda$ 与 $g$ 极小点集相同——$\arg\min g=\arg\min g^\lambda$。磨光不改变最优解位置，只让优化路径更平滑。
3. **梯度公式**：
   $$\nabla g^\lambda(x) = \frac{1}{\lambda}\big(x - \text{prox}_{\lambda g}(x)\big)$$
   包络的梯度由近端算子给出——近端算子不仅给迭代点，还给出光滑近似函数的梯度。这正是"prox 是 Moreau 包络的梯度"这一衔接 3.2 节光滑化与 3.4 节近端方法的关键事实。
4. **单调逼近**：$\lambda\to 0$ 时 $g^\lambda(x)\to g(x)$（逐点）。$\lambda$ 越小越接近原函数，但光滑性越弱。

---

## Moreau恒等式

**Moreau 恒等式**建立近端算子与其共轭近端算子的关系：

$$\text{prox}_{\lambda g}(x) + \lambda \, \text{prox}_{g^*/\lambda}(x/\lambda) = x$$

$g^*$ 是 $g$ 的 Fenchel 共轭。两层含义：

1. **计算意义**：若 $g$ 的近端难算但 $g^*$ 的近端易算，可经 Moreau 恒等式间接算——正是 3.5 节 Chambolle-Pock 的思路；
2. **对偶意义**：近端算子把 $x$ 拆成两部分——原始近端 $\text{prox}_{\lambda g}(x)$ 与对偶近端 $\lambda\,\text{prox}_{g^*/\lambda}(x/\lambda)$，互补。

### 应用于 L1 范数

对 $g(x)=\|x\|_1$，$g^*(y)=\iota_{\|\cdot\|_\infty\leq 1}(y)$。Moreau 恒等式给：

$$\text{prox}_{\lambda\|\cdot\|_1}(x) + \lambda \, \text{proj}_{[-1,1]}(x/\lambda) = x$$

即软阈值 $\mathcal{S}_\lambda(x) = x - \lambda\,\text{proj}_{[-1,1]}(x/\lambda)$——软阈值可理解为从 $x$ 中减去对偶投影。

---

## 与第4章 ULA 的联系

Moreau 包络的重要应用是构造 **MYULA（Moreau-Yoshida Unadjusted Langevin Algorithm）**。

Langevin 采样（第4章）需要后验梯度 $\nabla_x\ln p(x|y)$。当后验含不可微先验（如 Laplace→L1），梯度不存在。MYULA 用 Moreau 包络 $g^\lambda$ 替换不可微的 $g$，使后验变光滑近似：

$$\tilde{p}_\lambda(x|y) \propto p(y|x) \exp(-g^\lambda(x))$$

这近似后验处处可微，Langevin 可直接用。$\lambda\to 0$ 时近似后验收敛到真实后验——MYULA 在光滑性与逼近精度间用 $\lambda$ 取平衡。

从更深层次看，这揭示了一个贯穿全书的交汇点：近端算子与 Moreau 包络并不只是优化工具，它们还为"从优化迈入采样"提供了桥梁。无论是 3.4 用 prox 求解不可微正则，还是第4章用 Moreau 包络构造可微后验近似，背后是同一个"磨光不可微"的思想——**这是近端方法与采样方法的重要交汇点：近端算子不仅是优化工具，也是连接优化与采样的桥梁。**