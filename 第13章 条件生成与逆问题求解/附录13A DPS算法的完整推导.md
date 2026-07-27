# 附录13A DPS算法的完整推导

本附录给出DPS（Diffusion Posterior Sampling）算法的严格数学推导，从后验得分分解出发，经过Jensen近似、链式法则、Tweedie等式，最终得到VP-SDE和VE-SDE下的具体算法形式。

## 1 从后验得分分解出发

DPS的理论起点是13.2.2节的后验得分分解定理。给定观测 $y$，条件得分可分解为先验得分与似然得分之和：

$$\boxed{\nabla_{x_t}\log p(x_t|y) = \nabla_{x_t}\log p(x_t) + \nabla_{x_t}\log p(y|x_t)}$$

这两项的角色截然不同：

- **第一项** $\nabla_{x_t}\log p(x_t)$：先验得分，由预训练扩散模型直接提供。通过去噪得分匹配（第6章）或等价的变分下界训练（第11-12章），我们有 $s_\theta(x_t, t) \approx \nabla_{x_t}\log p_t(x_t)$。**此项无需任何额外处理。**

- **第二项** $\nabla_{x_t}\log p(y|x_t)$：似然得分，编码了观测 $y$ 提供的约束信息。此项**不可直接计算**，是DPS需要近似的核心对象。

DPS的全部工作集中在如何近似 $\nabla_{x_t}\log p(y|x_t)$。下面的推导将从其不可解性出发，逐步给出近似方案。

## 2 似然得分的目标展开

### 2.1 似然函数的积分表示

由全概率公式，$p(y|x_t)$ 可以展开为对 $x_0$ 的边际化积分：

$$p(y|x_t) = \int p(y|x_0)\,p(x_0|x_t)\,dx_0 = \mathbb{E}_{x_0 \sim p(x_0|x_t)}\bigl[p(y|x_0)\bigr]$$

这一展开的物理含义是：含噪状态 $x_t$ 对观测 $y$ 的似然，需要通过对所有可能的干净图像 $x_0$ 求期望来获得——因为 $x_t$ 并不直接决定 $y$，而是通过"去噪到 $x_0$ → 经由正向模型映射到 $y$"这条间接路径产生关联。

### 2.2 不可解性的分析

$p(y|x_t)$ 的不可解性源于两个因素：

**因素一：后验 $p(x_0|x_t)$ 无闭式表达。** 尽管正向转移核 $p(x_t|x_0)$ 是简单的高斯分布（由扩散过程的结构保证），但逆向核 $p(x_0|x_t) = p(x_t|x_0)p(x_0)/p(x_t)$ 依赖于数据分布 $p(x_0)$，一般没有闭式表达。

**因素二：即使 $p(y|x_0)$ 是简单的，积分仍不可解。** 以线性高斯测量 $y = Ax_0 + n$，$n \sim \mathcal{N}(0, \sigma_y^2 I)$ 为例，似然函数 $p(y|x_0) = \mathcal{N}(y; Ax_0, \sigma_y^2 I)$ 是简单的高斯分布。但被 $p(x_0|x_t)$ 加权积分后，结果一般无法化简为闭式。

**根本原因**：$x_t$ 与 $y$ 之间的因果链 $x_t \to x_0 \to y$ 引入了对 $x_0$ 的边际化，而 $p(x_0|x_t)$ 的复杂性使得这一边际化不可解。

### 2.3 近似策略的出发点

既然精确计算 $p(y|x_t)$ 不可行，DPS的近似策略是：**用一个可计算的近似替代 $p(x_0|x_t)$，从而将积分化简为单点估计。**

这一策略的关键在于：$p(x_0|x_t)$ 虽然没有闭式表达，但其后验均值 $\hat{x}_{0|t} = \mathbb{E}[x_0|x_t]$ 有闭式表达——这正是Tweedie等式提供的（13.2.3节）。DPS选择用后验均值处的delta函数近似整个后验分布：

$$p(x_0|x_t) \approx \delta(x_0 - \hat{x}_{0|t})$$

下面的推导将严格论证这一近似的来源和合理性。

## 3 Jensen近似的严格推导

### 3.1 Step 1：Jensen不等式给出下界

由Jensen不等式，对于凸函数 $\log(\cdot)$，对数的期望不超过期望的对数：

$$\log p(y|x_t) = \log \mathbb{E}_{x_0 \sim p(x_0|x_t)}\bigl[p(y|x_0)\bigr] \geq \mathbb{E}_{x_0 \sim p(x_0|x_t)}\bigl[\log p(y|x_0)\bigr]$$

即：

$$\log p(y|x_t) \geq \int \log p(y|x_0)\,p(x_0|x_t)\,dx_0$$

Jensen不等式取等号的条件是 $p(y|x_0)$ 在 $p(x_0|x_t)$ 的支集上为常数——这在一般情况下不成立，因此下界是严格的。

### 3.2 Step 2：DPS的点估计近似

DPS并未直接使用Jensen下界。取而代之，它将积分替换为在后验均值处的单点估计：

$$p(y|x_t) = \mathbb{E}_{x_0 \sim p(x_0|x_t)}\bigl[p(y|x_0)\bigr] \approx p(y|\hat{x}_{0|t})$$

其中 $\hat{x}_{0|t} = \mathbb{E}[x_0|x_t]$ 是Tweedie估计。这一近似等价于将后验分布替换为delta函数：

$$p(x_0|x_t) \approx \delta(x_0 - \hat{x}_{0|t})$$

代入积分：

$$p(y|x_t) = \int p(y|x_0)\,p(x_0|x_t)\,dx_0 \approx \int p(y|x_0)\,\delta(x_0 - \hat{x}_{0|t})\,dx_0 = p(y|\hat{x}_{0|t})$$

**近似解读**：DPS假设在给定 $x_t$ 的条件下，$x_0$ 的全部概率质量集中在后验均值 $\hat{x}_{0|t}$ 处。这是一种最粗糙的近似——丢弃了 $p(x_0|x_t)$ 的所有高阶矩（方差、偏度等），仅保留了一阶矩（均值）。

### 3.3 Step 3：近似与Jensen下界的关系

需要强调的是，**DPS的近似 $p(y|x_t) \approx p(y|\hat{x}_{0|t})$ 并不等价于Jensen下界**。两者的关系如下：

- **Jensen下界**：$\log p(y|x_t) \geq \mathbb{E}_{x_0}[\log p(y|x_0)]$ ——用 $\log$ 的凸性给出下界
- **DPS近似**：$\log p(y|x_t) \approx \log p(y|\hat{x}_{0|t})$ ——用delta函数替换整个积分

两者的区别在于：Jensen下界对 $p(y|x_0)$ 取对数后积分（先取对数再积分），而DPS近似是先积分再取对数（保持了 $\log p(y|x_t)$ 的结构，但将积分替换为单点估计）。DPS近似不是Jensen下界，也不一定是下界——它是一个单独的近似，其误差取决于 $p(y|x_0)$ 在 $p(x_0|x_t)$ 支集上的变化幅度。

### 3.4 近似误差的界与讨论

对于高斯测量模型 $y = Ax_0 + n$，$n \sim \mathcal{N}(0, \sigma_y^2 I)$，似然函数为：

$$p(y|x_0) = \frac{1}{(2\pi\sigma_y^2)^{m/2}} \exp\left(-\frac{\|y - Ax_0\|^2}{2\sigma_y^2}\right)$$

DPS近似为 $p(y|x_t) \approx p(y|\hat{x}_{0|t})$，其误差可以定性分析：

**当 $p(x_0|x_t)$ 高度集中时**（即后验方差 $\text{Var}[x_0|x_t]$ 很小），delta函数近似是合理的——因为 $p(y|x_0)$ 在 $p(x_0|x_t)$ 的支集上近似为常数。在扩散过程的低噪声阶段（$t$ 较小），$\hat{x}_{0|t}$ 趋近于真实 $x_0$，后验方差很小，近似精度较高。

**当 $p(x_0|x_t)$ 高度弥散时**（即后验方差 $\text{Var}[x_0|x_t]$ 很大），delta函数近似可能严重偏离真实值——因为 $p(y|x_0)$ 在支集上的变化显著。在扩散过程的高噪声阶段（$t$ 较大），$\hat{x}_{0|t}$ 仅是一个非常粗略的估计，后验方差很大，近似精度较低。

**高斯测量下的可控性**：对于线性高斯测量，$p(y|x_0)$ 是 $x_0$ 的二次函数（对数似然是二次的），因此 $p(y|\hat{x}_{0|t})$ 与 $\mathbb{E}_{x_0}[p(y|x_0)]$ 的差异可以通过 $A$ 和 $\text{Var}[x_0|x_t]$ 的函数来界定。具体地，利用高斯积分的性质：

$$\mathbb{E}_{x_0 \sim p(x_0|x_t)}\bigl[\|y - Ax_0\|^2\bigr] = \|y - A\hat{x}_{0|t}\|^2 + \text{tr}(A\,\text{Var}[x_0|x_t]\,A^\top)$$

因此DPS近似将重建误差低估了 $\text{tr}(A\,\text{Var}[x_0|x_t]\,A^\top)$ 这一项——这等价于忽略了后验不确定性对测量一致性的影响。这一误差项在后验方差较小时可忽略，但在高噪声阶段可能显著——这也解释了DPS在高噪声阶段表现不佳的实验现象（13.3.2节）。

## 4 梯度计算的链式法则

### 4.1 似然得分近似的链式法则

在DPS近似 $p(y|x_t) \approx p(y|\hat{x}_{0|t})$ 下，似然得分变为：

$$\nabla_{x_t}\log p(y|x_t) \approx \nabla_{x_t}\log p(y|\hat{x}_{0|t})$$

由于 $\hat{x}_{0|t}$ 是 $x_t$ 的函数（通过Tweedie等式），由链式法则：

$$\boxed{\nabla_{x_t}\log p(y|\hat{x}_{0|t}) = \left(\nabla_{x_0}\log p(y|x_0)\Big|_{x_0 = \hat{x}_{0|t}}\right)^\top \cdot \nabla_{x_t}\hat{x}_{0|t}}$$

其中 $\nabla_{x_t}\hat{x}_{0|t} \in \mathbb{R}^{n \times n}$ 是Tweedie估计关于含噪状态的Jacobian矩阵。

### 4.2 高斯噪声模型下的显式表达

对于线性高斯测量模型 $y = Ax_0 + n$，$n \sim \mathcal{N}(0, \sigma_y^2 I)$，似然函数为：

$$p(y|x_0) \propto \exp\left(-\frac{\|y - Ax_0\|^2}{2\sigma_y^2}\right)$$

其对数似然关于 $x_0$ 的梯度为：

$$\nabla_{x_0}\log p(y|x_0) = \frac{A^\top(y - Ax_0)}{\sigma_y^2}$$

代入链式法则：

$$\nabla_{x_t}\log p(y|\hat{x}_{0|t}) = \frac{1}{\sigma_y^2}\left(A^\top(y - A\hat{x}_{0|t})\right)^\top \cdot \nabla_{x_t}\hat{x}_{0|t}$$

展开内积：

$$\boxed{\nabla_{x_t}\log p(y|\hat{x}_{0|t}) = \frac{1}{\sigma_y^2}\,(\nabla_{x_t}\hat{x}_{0|t})^\top \cdot A^\top(y - A\hat{x}_{0|t})}$$

这是DPS似然得分近似的完整表达式。它包含两个部分：

- **数据一致性梯度** $\frac{A^\top(y - A\hat{x}_{0|t})}{\sigma_y^2}$：这是关于 $\hat{x}_{0|t}$ 的标准最小二乘梯度，指向使测量残差减小的方向。它在测量空间中定义，与具体的扩散模型无关。

- **Jacobian矩阵** $\nabla_{x_t}\hat{x}_{0|t}$：这是Tweedie估计关于含噪状态的敏感度矩阵，描述了"含噪状态 $x_t$ 的微小变化如何传播到去噪估计 $\hat{x}_{0|t}$"。它将测量空间中的梯度回传到含噪状态空间。

### 4.3 DPS简化：Jacobian的省略

在实践中，Jacobian矩阵 $\nabla_{x_t}\hat{x}_{0|t}$ 是一个 $n \times n$ 的矩阵（$n$ 是图像的像素数），直接计算代价极高。DPS做出了关键的简化：**省略Jacobian矩阵**，将似然得分近似为：

$$\nabla_{x_t}\log p(y|\hat{x}_{0|t}) \approx \frac{1}{\sigma_y^2}\,A^\top(y - A\hat{x}_{0|t})$$

更精确地说，DPS引入一个缩放因子 $\zeta$ 来补偿Jacobian省略的影响：

$$\boxed{\nabla_{x_t}\log p(y|x_t) \approx \zeta \cdot \frac{A^\top(y - A\hat{x}_{0|t})}{\sigma_y^2}}$$

**Jacobian省略的合理性论证**：

1. **量级吸收**：Jacobian矩阵 $\nabla_{x_t}\hat{x}_{0|t}$ 的作用本质上是将 $x_0$ 空间的梯度重新缩放到 $x_t$ 空间。这个重新缩放的效果可以用一个标量因子 $\zeta$ 近似——将Jacobian的"平均缩放效果"吸收到 $\zeta$ 中。

2. **方向保持**：在许多情况下，$\nabla_{x_t}\hat{x}_{0|t}$ 并不显著改变梯度的方向——它主要影响梯度的大小。因此，省略Jacobian后保留梯度方向，用 $\zeta$ 调节大小，是一个合理的折中。

3. **实践经验**：Chung et al. (2023) 的实验表明，选择适当的 $\zeta$ 后，省略Jacobian的DPS在多种逆问题上的性能与完整计算Jacobian的版本相当——这说明 $\zeta$ 确实能够有效补偿Jacobian省略的影响。

**$\zeta$ 的选择**：$\zeta$ 是一个超参数，通常 $\zeta \in [0.1, 1.0]$（13.4.3节详细讨论了 $\zeta$ 的选择策略）。$\zeta$ 过大导致过拟合测量噪声，$\zeta$ 过小导致测量一致性不足——它扮演着与经典正则化参数 $\lambda$ 相同的角色。

## 5 VP-SDE下的具体形式

### 5.1 VP-SDE的正向过程

VP-SDE（Variance Preserving SDE，第7章7.2节）的正向过程为：

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\bar\alpha_t = \prod_{s=1}^{t}(1 - \beta_s)$ 是累积乘积，$\beta_t$ 是噪声调度。正向转移核为：

$$p(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)$$

### 5.2 VP-SDE下的Tweedie估计

由Tweedie等式（13.2.3节），VP-SDE下的后验均值为：

$$\hat{x}_{0|t} = \mathbb{E}[x_0|x_t] = \frac{x_t + (1-\bar\alpha_t)\,\nabla_{x_t}\log p_t(x_t)}{\sqrt{\bar\alpha_t}}$$

利用 $\epsilon$-预测参数化 $\nabla_{x_t}\log p_t(x_t) = -\epsilon_\theta(x_t, t)/\sqrt{1-\bar\alpha_t}$（第11章11.3节），Tweedie估计等价地表示为：

$$\boxed{\hat{x}_{0|t} = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}}$$

这是VP-SDE下DPS算法使用的去噪估计公式。

也可以使用得分函数 $s_\theta(x_t, t) \approx \nabla_{x_t}\log p_t(x_t)$ 直接表达：

$$\hat{x}_{0|t} = \frac{x_t + (1-\bar\alpha_t)\,s_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$$

两种参数化完全等价——区别仅在于网络输出的物理含义：$\epsilon_\theta$ 预测噪声，$s_\theta$ 预测得分。

### 5.3 VP-SDE下DPS的完整算法

将Tweedie估计代入似然得分近似公式，VP-SDE下DPS的条件得分为：

$$\nabla_{x_t}\log p(x_t|y) \approx s_\theta(x_t, t) + \zeta \cdot \frac{A^\top\!\left(y - A\hat{x}_{0|t}\right)}{\sigma_y^2}$$

其中：

$$\hat{x}_{0|t} = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$$

**VP-SDE下DPS的采样步骤**（从 $t = T$ 到 $t = 0$）：

**Step 1**：计算先验得分和去噪估计

$$\epsilon_t = \epsilon_\theta(x_t, t), \quad \hat{x}_{0|t} = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_t}{\sqrt{\bar\alpha_t}}$$

**Step 2**：计算似然梯度修正项

$$g_t = \zeta \cdot \frac{A^\top(y - A\hat{x}_{0|t})}{\sigma_y^2}$$

**Step 3**：执行条件逆向SDE的一步（以DDPM离散化为例）

$$x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_t\right) - \beta_t\,g_t + \sigma_t\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\sigma_t$ 是DDPM的随机性参数。

**关键观察**：DPS的额外计算量仅为一次前向传播 $A\hat{x}_{0|t}$ 和一次反传 $A^\top(y - A\hat{x}_{0|t})$——对于线性算子 $A$，这通常是轻量的（例如对于CT重建，$A$ 是Radon变换，$A^\top$ 是反Radon变换）。这使得DPS的计算效率远高于需要完整Jacobian的方法。

## 6 VE-SDE下的具体形式

### 6.1 VE-SDE的正向过程

VE-SDE（Variance Exploding SDE，第7章7.2节）的正向过程为：

$$x_t = x_0 + \sigma_t\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\sigma_t$ 是随时间单调递增的噪声标准差，满足 $\sigma_0 \approx 0$（几乎无噪声）和 $\sigma_T \gg 1$（几乎纯噪声）。正向转移核为：

$$p(x_t|x_0) = \mathcal{N}(x_t; x_0, \sigma_t^2 I)$$

### 6.2 VE-SDE下的Tweedie估计

对于VE-SDE，$s_t = 1$，$\sigma_t^2 = \sigma_t^2$。由Tweedie等式的一般形式 $\mathbb{E}[x_0|x_t] = (x_t + \sigma_t^2\,\nabla_{x_t}\log p_t(x_t))/s_t$，代入 $s_t = 1$：

$$\boxed{\hat{x}_{0|t} = x_t + \sigma_t^2\,s_\theta(x_t, \sigma_t)}$$

其中 $s_\theta(x_t, \sigma_t) \approx \nabla_{x_t}\log p_t(x_t)$ 是得分网络。在VE-SDE下，习惯用噪声水平 $\sigma_t$ 作为条件输入而非时间步 $t$。

### 6.3 VE-SDE下DPS的完整算法

VE-SDE下DPS的条件得分为：

$$\nabla_{x_t}\log p(x_t|y) \approx s_\theta(x_t, \sigma_t) + \zeta \cdot \frac{A^\top\!\left(y - A\hat{x}_{0|t}\right)}{\sigma_y^2}$$

其中：

$$\hat{x}_{0|t} = x_t + \sigma_t^2\,s_\theta(x_t, \sigma_t)$$

**VE-SDE下DPS的采样步骤**（从 $t = T$ 到 $t = 0$）：

**Step 1**：计算先验得分和去噪估计

$$s_t = s_\theta(x_t, \sigma_t), \quad \hat{x}_{0|t} = x_t + \sigma_t^2\,s_t$$

**Step 2**：计算似然梯度修正项

$$g_t = \zeta \cdot \frac{A^\top(y - A\hat{x}_{0|t})}{\sigma_y^2}$$

**Step 3**：执行条件逆向SDE的一步（以Euler-Maruyama离散化为例）

$$x_{t-1} = x_t + \left(\sigma_t^2\,s_t + \sigma_t^2\,g_t\right)\frac{\sigma_{t-1}^2 - \sigma_t^2}{\sigma_t^2} + \sqrt{\sigma_t^2 - \sigma_{t-1}^2}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 6.4 VP-SDE与VE-SDE下DPS的对比

两种SDE设定下DPS的核心逻辑完全一致——差异仅体现在Tweedie估计的具体形式和逆向SDE的离散化方式上：

| 维度 | VP-SDE | VE-SDE |
|---|---|---|
| 正向过程 | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ | $x_t = x_0 + \sigma_t\epsilon$ |
| Tweedie估计 | $\hat{x}_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta)/\sqrt{\bar\alpha_t}$ | $\hat{x}_0 = x_t + \sigma_t^2\,s_\theta$ |
| 似然梯度 | $\zeta A^\top(y - A\hat{x}_0)/\sigma_y^2$ | $\zeta A^\top(y - A\hat{x}_0)/\sigma_y^2$ |
| 噪声调度 | $\beta_t$（通常较小，如0.0001→0.02） | $\sigma_t$（通常几何级数，如0.01→1348） |

似然梯度修正项 $g_t = \zeta \cdot A^\top(y - A\hat{x}_{0|t})/\sigma_y^2$ 在两种设定下**完全相同**——因为DPS的似然梯度只依赖于去噪估计 $\hat{x}_{0|t}$ 和测量模型 $(A, \sigma_y)$，与具体的SDE形式无关。这正是后验得分分解的模块化优势：先验得分和似然得分的计算完全解耦，更换SDE只需调整先验得分的计算方式。

## 附录小结

DPS算法的完整推导路径可总结为以下关键步骤：

$$\underbrace{\nabla\log p(x_t|y) = \nabla\log p(x_t) + \nabla\log p(y|x_t)}_{\text{后验得分分解}} \xrightarrow{p(y|x_t) \approx p(y|\hat{x}_{0|t})} \underbrace{\nabla\log p(x_t|y) \approx s_\theta + \zeta \cdot \frac{A^\top(y - A\hat{x}_0)}{\sigma_y^2}}_{\text{DPS近似}}$$

每一步引入的近似及其后果：

| 步骤 | 近似 | 引入的误差 | 误差控制 |
|---|---|---|---|
| $p(x_0\|x_t) \approx \delta(x_0 - \hat{x}_{0\|t})$ | delta函数替代积分 | 忽略后验方差 | 低噪声阶段误差小 |
| $\nabla_{x_t}\hat{x}_{0\|t} \approx \zeta \cdot I$ | Jacobian省略 | 方向/大小偏差 | $\zeta$ 补偿量级 |

DPS的简洁性来自于这两步近似的大胆简化——用delta函数替代不可解积分，用标量因子替代Jacobian矩阵。这些近似使得DPS在计算上极为高效（每步仅需一次额外的前向传播和反传），但也引入了理论上的不严格性——这正是ΠGDM（各向同性高斯近似）和DOC（精确反传）等改进方法的动机。

**来源**：Chung et al. (2023) "Diffusion Posterior Sampling for General Noisy Inverse Problems"；Chung et al. (2209.14687) §3.2 公式(31)-(33)；第7章7.2-7.3节VP-SDE/VE-SDE与逆向SDE；第5章5.3节Tweedie等式
