# 附录13A DPS 算法的完整推导

> 这一节是 13.2~13.3 节 DPS 算法的"厨房后台"——把每一步推导摊开给你看。如果你只想用 DPS，跳到 13.3.2 的伪代码就够了；但如果你想搞清楚"为什么 delta 近似是合理的""Jacobian 到底是省还是不省"，这里把账算到底。

本附录给出 DPS（Diffusion Posterior Sampling）算法的严格数学推导，从后验得分分解出发，经 Jensen 近似、链式法则、Tweedie 等式，最终得 VP-SDE 和 VE-SDE 下具体算法形式。

---

## 1 从后验得分分解出发

DPS 理论起点是 13.2.2 节后验得分分解定理。给定观测 `y`，条件得分可拆成先验得分 + 似然得分：

$$\boxed{\nabla_{x_t}\log p(x_t|y) = \nabla_{x_t}\log p(x_t) + \nabla_{x_t}\log p(y|x_t)}$$

两项角色截然不同：

- **第一项** `∇_{x_t}log p(x_t)`：先验得分，由预训练扩散模型直接给。通过去噪得分匹配（第6章）或等价变分下界训练（第11~12章），有 `s_θ(x_t, t) ≈ ∇_{x_t}log p_t(x_t)`。**此项无需任何额外处理。**
- **第二项** `∇_{x_t}log p(y|x_t)`：似然得分，编码观测 `y` 约束信息。此项**算不出**，是 DPS 需近似的核心对象。

> **为什么 DPS 的全部功夫都花在第二项上？** 因为第一项我们已经在第4~12章"白嫖"到了——预训练扩散模型就是个先验得分机器。真正的硬骨头是第二项：它要把"观测 `y` 的约束"翻译成一个能算的梯度。下面推导的每一步，都是在想办法把 `∇log p(y|x_t)` 这个"算不出"的量，替换成一个"算得出"的近似。

DPS 全部工作集中在怎么近似 `∇_{x_t}log p(y|x_t)`。下面推导从不可解性出发，逐步给近似方案。

---

## 2 似然得分的目标展开

### 2.1 似然函数的积分表示

由全概率公式，`p(y|x_t)` 可展开为对 `x_0` 的边际化积分：

$$p(y|x_t) = \int p(y|x_0)\,p(x_0|x_t)\,dx_0 = \mathbb{E}_{x_0 \sim p(x_0|x_t)}\bigl[p(y|x_0)\bigr]$$

物理含义：含噪状态 `x_t` 对观测 `y` 的似然，需通过对所有可能干净图 `x_0` 求期望获得——因 `x_t` 不直接决定 `y`，而是通过"去噪到 `x_0` → 经正向模型映到 `y`"这条间接路径产生关联。

### 2.2 不可解性的分析

`p(y|x_t)` 不可解源于两因素：

**因素一**：后验 `p(x_0|x_t)` 无闭式。虽正向转移核 `p(x_t|x_0)` 是简单高斯（扩散结构保证），但逆向核 `p(x_0|x_t) = p(x_t|x_0)p(x_0)/p(x_t)` 依赖数据分布 `p(x_0)`，一般没闭式。

**因素二**：即使 `p(y|x_0)` 简单，积分仍不可解。以线性高斯测量 `y = Ax_0 + n`，`n ~ N(0, σ_y²I)` 为例，似然 `p(y|x_0) = N(y; Ax_0, σ_y²I)` 是简单高斯。但被 `p(x_0|x_t)` 加权积分后，结果一般化不开。

**根本原因**：`x_t` 与 `y` 间因果链 `x_t → x_0 → y` 引入对 `x_0` 的边际化，而 `p(x_0|x_t)` 复杂性使这边际化不可解。

### 2.3 近似策略的出发点

既然精确算 `p(y|x_t)` 不可行，DPS 近似策略：**用一个可计算近似替代 `p(x_0|x_t)`，从而把积分化简成单点估计。**

关键点：`p(x_0|x_t)` 虽无闭式，但其后验均值 `x̂_{0|t} = E[x_0|x_t]` 有闭式——这正是 Tweedie 等式提供（13.2.3节）。DPS 选在后验均值处用 delta 函数近似整个后验分布：

$$p(x_0|x_t) \approx \delta(x_0 - \hat{x}_{0|t})$$

下面推导严格论证这近似来源和合理性。

---

## 3 Jensen 近似的严格推导（可跳过）

### 3.1 Step 1：Jensen 不等式给下界

由 Jensen 不等式，对凸函数 `log(·)`，对数期望不超过期望对数：

$$\log p(y|x_t) = \log \mathbb{E}_{x_0 \sim p(x_0|x_t)}\bigl[p(y|x_0)\bigr] \geq \mathbb{E}_{x_0 \sim p(x_0|x_t)}\bigl[\log p(y|x_0)\bigr]$$

即：

$$\log p(y|x_t) \geq \int \log p(y|x_0)\,p(x_0|x_t)\,dx_0$$

Jensen 取等号条件是 `p(y|x_0)` 在 `p(x_0|x_t)` 支集上为常数——一般不成立，故下界严格。

### 3.2 Step 2：DPS 的点估计近似

DPS 未直接用 Jensen 下界。取而代之，把积分换成后验均值处单点估计：

$$p(y|x_t) = \mathbb{E}_{x_0 \sim p(x_0|x_t)}\bigl[p(y|x_0)\bigr] \approx p(y|\hat{x}_{0|t})$$

`x̂_{0|t} = E[x_0|x_t]` 是 Tweedie 估计。这近似等价于把后验分布换成 delta 函数：

$$p(x_0|x_t) \approx \delta(x_0 - \hat{x}_{0|t})$$

代入积分：

$$p(y|x_t) = \int p(y|x_0)\,p(x_0|x_t)\,dx_0 \approx \int p(y|x_0)\,\delta(x_0 - \hat{x}_{0|t})\,dx_0 = p(y|\hat{x}_{0|t})$$

**近似解读**：DPS 假设给定 `x_t` 时 `x_0` 全部概率质量集中在后验均值 `x̂_{0|t}` 处。这是最粗近似——丢了 `p(x_0|x_t)` 所有高阶矩（方差、偏度等），只留一阶矩（均值）。

### 3.3 Step 3：近似与 Jensen 下界的关系

需强调：**DPS 近似 `p(y|x_t) ≈ p(y|x̂_{0|t})` 不等价于 Jensen 下界**。两者关系：
- **Jensen 下界**：`log p(y|x_t) ≥ E_{x_0}[log p(y|x_0)]`——用 `log` 凸性给下界
- **DPS 近似**：`log p(y|x_t) ≈ log p(y|x̂_{0|t})`——用 delta 函数替换整个积分

区别：Jensen 下界对 `p(y|x_0)` 取对数后积分（先取对数再积分），DPS 近似先积分再取对数（保 `log p(y|x_t)` 结构，但积分换单点估计）。DPS 近似非 Jensen 下界，也不一定是下界——它是单独近似，误差取决于 `p(y|x_0)` 在 `p(x_0|x_t)` 支集上变化幅度。

### 3.4 近似误差的界与讨论

对高斯测量 `y = Ax_0 + n`，`n ~ N(0, σ_y²I)`，似然：

$$p(y|x_0) = \frac{1}{(2\pi\sigma_y^2)^{m/2}} \exp\left(-\frac{\|y - Ax_0\|^2}{2\sigma_y^2}\right)$$

DPS 近似 `p(y|x_t) ≈ p(y|x̂_{0|t})`，误差可定性分析：

**当 `p(x_0|x_t)` 高度集中时**（后验方差 `Var[x_0|x_t]` 很小），delta 近似合理——因 `p(y|x_0)` 在 `p(x_0|x_t)` 支集上近似常数。扩散低噪声阶段（`t` 小），`x̂_{0|t}` 趋近真 `x_0`，后验方差小，近似精度高。

**当 `p(x_0|x_t)` 高度弥散时**（后验方差 `Var[x_0|x_t]` 很大），delta 近似可能严重偏离——因 `p(y|x_0)` 在支集上变化显著。扩散高噪声阶段（`t` 大），`x̂_{0|t}` 仅粗略估计，后验方差大，近似精度低。

**高斯测量下可控性**：线性高斯测量 `p(y|x_0)` 是 `x_0` 二次函数（对数似然二次），故 `p(y|x̂_{0|t})` 与 `E_{x_0}[p(y|x_0)]` 差异可用 `A` 和 `Var[x_0|x_t]` 函数界定。利用高斯积分性质：

$$\mathbb{E}_{x_0 \sim p(x_0|x_t)}\bigl[\|y - Ax_0\|^2\bigr] = \|y - A\hat{x}_{0|t}\|^2 + \text{tr}(A\,\text{Var}[x_0|x_t]\,A^\top)$$

故 DPS 近似把重建误差低估了 `tr(A Var[x_0|x_t] A^⊤)` 这一项——等价忽略后验不确定性对测量一致性的影响。这误差项在后验方差小时可忽略，高噪声阶段可能显著——也解释 DPS 在高噪声阶段表现不佳的实验现象（13.3.2节）。

---

## 4 梯度计算的链式法则

### 4.1 似然得分近似的链式法则

DPS 近似 `p(y|x_t) ≈ p(y|x̂_{0|t})` 下，似然得分变：

$$\nabla_{x_t}\log p(y|x_t) \approx \nabla_{x_t}\log p(y|\hat{x}_{0|t})$$

因 `x̂_{0|t}` 是 `x_t` 函数（经 Tweedie 等式），由链式法则：

$$\boxed{\nabla_{x_t}\log p(y|\hat{x}_{0|t}) = \left(\nabla_{x_0}\log p(y|x_0)\Big|_{x_0 = \hat{x}_{0|t}}\right)^\top \cdot \nabla_{x_t}\hat{x}_{0|t}}$$

`∇_{x_t}x̂_{0|t} ∈ R^{n×n}` 是 Tweedie 估计关于含噪状态的 Jacobian 矩阵。

### 4.2 高斯噪声模型下显式表达

对线性高斯测量 `y = Ax_0 + n`，`n ~ N(0, σ_y²I)`，似然：

$$p(y|x_0) \propto \exp\left(-\frac{\|y - Ax_0\|^2}{2\sigma_y^2}\right)$$

其对数似然关于 `x_0` 梯度：

$$\nabla_{x_0}\log p(y|x_0) = \frac{A^\top(y - Ax_0)}{\sigma_y^2}$$

代入链式法则：

$$\nabla_{x_t}\log p(y|\hat{x}_{0|t}) = \frac{1}{\sigma_y^2}\left(A^\top(y - A\hat{x}_{0|t})\right)^\top \cdot \nabla_{x_t}\hat{x}_{0|t}$$

展开内积：

$$\boxed{\nabla_{x_t}\log p(y|\hat{x}_{0|t}) = \frac{1}{\sigma_y^2}\,(\nabla_{x_t}\hat{x}_{0|t})^\top \cdot A^\top(y - A\hat{x}_{0|t})}$$

这是 DPS 似然得分近似完整表达式。它含两部分：
- **数据一致性梯度** `A^⊤(y - A(x̂_{0|t}))/σ_y²`：关于 `x̂_{0|t}` 标准最小二乘梯度，指向使测量残差减小方向。在测量空间定义，与具体扩散模型无关。
- **Jacobian 矩阵** `∇_{x_t}x̂_{0|t}`：Tweedie 估计关于含噪状态敏感度矩阵，描述"含噪状态 `x_t` 微小变化如何传播到去噪估计 `x̂_{0|t}`"。它把测量空间梯度回传到含噪状态空间。

### 4.3 DPS 简化：Jacobian 的省略

实践 Jacobian 矩阵 `∇_{x_t}x̂_{0|t}` 是 `n×n` 矩阵（`n` 图像像素数），直接计算代价极高。DPS 做关键简化：**省略 Jacobian 矩阵**，把似然得分近似为：

$$\nabla_{x_t}\log p(y|\hat{x}_{0|t}) \approx \frac{1}{\sigma_y^2}\,A^\top(y - A\hat{x}_{0|t})$$

更精确，DPS 引入缩放因子 `ζ` 补偿 Jacobian 省略影响：

$$\boxed{\nabla_{x_t}\log p(y|x_t) \approx \zeta \cdot \frac{A^\top(y - A\hat{x}_{0|t})}{\sigma_y^2}}$$

**Jacobian 省略合理性论证**：
1. **量级吸收**：Jacobian `∇_{x_t}x̂_{0|t}` 本质是把 `x_0` 空间梯度重新缩放到 `x_t` 空间。这重新缩放效果可用标量因子 `ζ` 近似——把 Jacobian"平均缩放效果"吸收进 `ζ`。
2. **方向保持**：许多情况 `∇_{x_t}x̂_{0|t}` 不显著改变梯度方向——主要影响大小。故省略 Jacobian 保留梯度方向，用 `ζ` 调大小，是合理折中。
3. **实践经验**：Chung et al. (2023) 实验表明，选适当 `ζ` 后，省略 Jacobian 的 DPS 在多种逆问题上性能与完整算 Jacobian 版本相当——说明 `ζ` 能有效补偿 Jacobian 省略影响。

**`ζ` 选择**：`ζ` 是超参数，通常 `ζ ∈ [0.1, 1.0]`（13.4.3节详讨 `ζ` 策略）。`ζ` 过大过拟合测量噪声，`ζ` 过小测量一致性不足——扮演与经典正则化参数 `λ` 相同角色。

---

## 5 VP-SDE 下的具体形式

### 5.1 VP-SDE 的正向过程

VP-SDE（Variance Preserving SDE，第7章 7.2）正向过程：

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

`ᾱ_t = ∏_{s=1}^{t}(1 - β_s)` 是累积乘积，`β_t` 是噪声调度。正向转移核：

$$p(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)$$

### 5.2 VP-SDE 下的 Tweedie 估计

由 Tweedie 等式（13.2.3节），VP-SDE 下后验均值：

$$\hat{x}_{0|t} = \mathbb{E}[x_0|x_t] = \frac{x_t + (1-\bar\alpha_t)\,\nabla_{x_t}\log p_t(x_t)}{\sqrt{\bar\alpha_t}}$$

用 ε 预测参数化 `∇_{x_t}log p_t(x_t) = -ε_θ(x_t, t)/√(1-ᾱ_t)`（第11章 11.3），Tweedie 估计等价写成：

$$\boxed{\hat{x}_{0|t} = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}}$$

这是 VP-SDE 下 DPS 算法用的去噪估计公式。

也可用得分函数 `s_θ(x_t, t) ≈ ∇_{x_t}log p_t(x_t)` 直接表达：

$$\hat{x}_{0|t} = \frac{x_t + (1-\bar\alpha_t)\,s_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$$

两种参数化完全等价——只差网络输出物理含义：`ε_θ` 预测噪声，`s_θ` 预测得分。

### 5.3 VP-SDE 下 DPS 的完整算法

把 Tweedie 估计代入似然得分近似，VP-SDE 下 DPS 条件得分：

$$\nabla_{x_t}\log p(x_t|y) \approx s_\theta(x_t, t) + \zeta \cdot \frac{A^\top\!\left(y - A\hat{x}_{0|t}\right)}{\sigma_y^2}$$

其中：

$$\hat{x}_{0|t} = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$$

**VP-SDE 下 DPS 采样步骤**（从 `t = T` 到 `t = 0`）：

**Step 1**：算先验得分和去噪估计

$$\epsilon_t = \epsilon_\theta(x_t, t), \quad \hat{x}_{0|t} = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_t}{\sqrt{\bar\alpha_t}}$$

**Step 2**：算似然梯度修正项

$$g_t = \zeta \cdot \frac{A^\top(y - A\hat{x}_{0|t})}{\sigma_y^2}$$

**Step 3**：执行条件逆向 SDE 一步（以 DDPM 离散化为例）

$$x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_t\right) - \beta_t\,g_t + \sigma_t\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

`σ_t` 是 DDPM 随机性参数。

**关键观察**：DPS 额外计算量仅一次前向传播 `A(x̂_{0|t})` 和一次反传 `A^⊤(y - A(x̂_{0|t}))`——对线性算子 `A` 通常轻量（如 CT 重建 `A` 是 Radon 变换，`A^⊤` 是反 Radon 变换）。这使 DPS 计算效率远高于需完整 Jacobian 的方法。

---

## 6 VE-SDE 下的具体形式

### 6.1 VE-SDE 的正向过程

VE-SDE（Variance Exploding SDE，第7章 7.2）正向过程：

$$x_t = x_0 + \sigma_t\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

`σ_t` 是随时间单调递增噪声标准差，满足 `σ_0 ≈ 0`（几乎无噪声）、`σ_T ≫ 1`（几乎纯噪声）。正向转移核：

$$p(x_t|x_0) = \mathcal{N}(x_t; x_0, \sigma_t^2 I)$$

### 6.2 VE-SDE 下的 Tweedie 估计

对 VE-SDE，`s_t = 1`，`σ_t² = σ_t²`。由 Tweedie 等式一般形式 `E[x_0|x_t] = (x_t + σ_t²∇_{x_t}log p_t(x_t))/s_t`，代入 `s_t = 1`：

$$\boxed{\hat{x}_{0|t} = x_t + \sigma_t^2\,s_\theta(x_t, \sigma_t)}$$

`s_θ(x_t, σ_t) ≈ ∇_{x_t}log p_t(x_t)` 是得分网络。VE-SDE 下习惯用噪声水平 `σ_t` 作条件输入而非时间步 `t`。

### 6.3 VE-SDE 下 DPS 的完整算法

VE-SDE 下 DPS 条件得分：

$$\nabla_{x_t}\log p(x_t|y) \approx s_\theta(x_t, \sigma_t) + \zeta \cdot \frac{A^\top\!\left(y - A\hat{x}_{0|t}\right)}{\sigma_y^2}$$

其中：

$$\hat{x}_{0|t} = x_t + \sigma_t^2\,s_\theta(x_t, \sigma_t)$$

**VE-SDE 下 DPS 采样步骤**（从 `t = T` 到 `t = 0`）：

**Step 1**：算先验得分和去噪估计

$$s_t = s_\theta(x_t, \sigma_t), \quad \hat{x}_{0|t} = x_t + \sigma_t^2\,s_t$$

**Step 2**：算似然梯度修正项

$$g_t = \zeta \cdot \frac{A^\top(y - A\hat{x}_{0|t})}{\sigma_y^2}$$

**Step 3**：执行条件逆向 SDE 一步（以 Euler-Maruyama 离散化为例）

$$x_{t-1} = x_t + \left(\sigma_t^2\,s_t + \sigma_t^2\,g_t\right)\frac{\sigma_{t-1}^2 - \sigma_t^2}{\sigma_t^2} + \sqrt{\sigma_t^2 - \sigma_{t-1}^2}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 6.4 VP-SDE 与 VE-SDE 下 DPS 对比

两种 SDE 设定下 DPS 核心逻辑完全一致——差异只在 Tweedie 估计具体形式和逆向 SDE 离散化方式：

| 维度 | VP-SDE | VE-SDE |
|---|---|---|
| 正向过程 | `x_t = √ᾱ_t x_0 + √(1-ᾱ_t)ε` | `x_t = x_0 + σ_t ε` |
| Tweedie估计 | `x̂_0 = (x_t - √(1-ᾱ_t)ε_θ)/√ᾱ_t` | `x̂_0 = x_t + σ_t² s_θ` |
| 似然梯度 | `ζ A^⊤(y - A(x̂_0))/σ_y²` | `ζ A^⊤(y - A(x̂_0))/σ_y²` |
| 噪声调度 | `β_t`（通常较小，如0.0001→0.02） | `σ_t`（通常几何级数，如0.01→1348） |

似然梯度修正项 `g_t = ζ·A^⊤(y - A(x̂_{0|t}))/σ_y²` 两种设定下**完全相同**——因 DPS 似然梯度只依赖去噪估计 `x̂_{0|t}` 和测量模型 `(A, σ_y)`，与具体 SDE 形式无关。这正是后验得分分解模块化优势：先验得分和似然得分计算完全解耦，换 SDE 只需调先验得分计算方式。

---

## 附录小结

DPS 算法完整推导路径可总结为关键步骤：

$$\underbrace{\nabla\log p(x_t|y) = \nabla\log p(x_t) + \nabla\log p(y|x_t)}_{\text{后验得分分解}} \xrightarrow{p(y|x_t) \approx p(y|\hat{x}_{0|t})} \underbrace{\nabla\log p(x_t|y) \approx s_\theta + \zeta \cdot \frac{A^\top(y - A\hat{x}_0)}{\sigma_y^2}}_{\text{DPS近似}}$$

每步引入近似及后果：

| 步骤 | 近似 | 引入误差 | 误差控制 |
|---|---|---|---|
| `p(x_0\|x_t) ≈ δ(x_0 - x̂_{0\|t})` | delta函数替代积分 | 忽略后验方差 | 低噪声阶段误差小 |
| `∇_{x_t}x̂_{0\|t} ≈ ζ·I` | Jacobian省略 | 方向/大小偏差 | `ζ` 补偿量级 |

DPS 简洁性来自这两步近似的大胆简化——用 delta 替代不可解积分，用标量因子替代 Jacobian。这些近似使 DPS 计算极高效（每步仅一次额外前向传播和反传），但也引入理论不严格——这正是 ΠGDM（各向同性高斯近似）和 DOC（精确反传）等改进方法动机。

**来源**：Chung et al. (2023) "Diffusion Posterior Sampling for General Noisy Inverse Problems"；Chung et al. (2209.14687) §3.2 公式(31)-(33)；第7章7.2-7.3节VP-SDE/VE-SDE与逆向SDE；第5章5.3节Tweedie等式
