# 第12章 Score ≡ ELBO：殊途同归 — 提纲

## 本章定位

第12章是全书**理论高潮**——两条独立发展的路径（采样路径与变分路径）在此交汇。采样路径的终点是"通过DSM训练去噪器学习得分函数"，变分路径的终点是"通过VLB训练去噪器最大化数据似然"。本章证明：**DSM损失 ≡ 变分下界（VLB）**——两个目标在数学上等价，仅差时间权重。这意味着：从贝叶斯后验采样需求出发，经由得分匹配走向扩散模型（采样路径），与从生成建模需求出发，经由变分推断走向扩散模型（变分路径），殊途同归。

**核心论点**：DSM损失与VLB的等价性不是巧合——得分函数 $\nabla\log p(x)$ 是概率分布的"梯度信息"，而 $\log p(x)$ 是"概率值信息"，梯度与函数值是同一对象的不同表示。准确估计得分 = 对数据赋予高概率，二者等价是数学上的必然。

---

## 叙事弧

```
两条路径的回顾 → 结构对比（发现一致性）→ 直觉解释（为何必然）→ 形式化证明（数学严格性）
→ 连续时间统一（理论深化）→ 实践意义（落地应用）→ 叙事汇合（全书论点验证）
```

---

## 章节结构

### 12.0 本章导读：两条路径的交汇时刻

**核心观点**：从第1章到第11章，我们走了两条路——采样路径（第4-7章）从逆问题的后验采样需求出发，经由Langevin动力学、得分函数、Tweedie等式、得分匹配，走向扩散模型；变分路径（第8-11章）从生成建模的最大似然需求出发，经由ELBO、VAE、层级VAE，走向变分扩散模型。两条路径各自独立发展，但训练出来的扩散模型几乎一样——去噪器网络、加噪过程、采样过程都高度一致。这暗示：两条路径在数学上是等价的。

- 两条路径的时间线对照图
- 章节导航：回顾→对比→证明→统一→实践→汇合

**来源**：全书叙事；book_plan.md 核心论点

---

### 12.1 采样路径回顾：从Tweedie到DSM损失

**核心观点**：采样路径的核心链条是"去噪器→得分函数→采样"——Tweedie等式将去噪器与得分函数桥接，DSM将去噪器训练与得分匹配等价，多尺度得分匹配将单一噪声扩展为噪声调度，最终走向扩散SDE。本节回顾这条路径的关键数学结果，为等价性证明做准备。

- **得分函数与Tweedie等式**（第5章核心结果）
  - 得分函数 $s(x) = \nabla\log p(x)$：概率密度的梯度方向
  - Tweedie等式：$\nabla\log p_\sigma(x) = (D_\sigma(x) - x)/\sigma^2$
  - 核心洞见：去噪器残差 = 得分函数（差一个缩放因子）
- **DSM目标函数**（第6章核心结果）
  - ESM→ISM→DSM的演化链
  - DSM目标：$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\mathbb{E}_{p(x)}\mathbb{E}_\epsilon\left[\|s_\theta(x+\sigma\epsilon) + \epsilon/\sigma\|^2\right]$
  - Vincent (2011) 等价性：$\mathcal{J}_\text{DSM} = \mathcal{J}_\text{ESM}^{(\sigma)} + C$
- **ε预测参数化**（第6章6.6节）
  - $s_\theta = -\epsilon_\theta/\sigma$：得分预测↔噪声预测的转换
  - DSM的ε预测形式：$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\mathbb{E}_\sigma\mathbb{E}_{x_0}\mathbb{E}_\epsilon\left[\frac{\lambda(\sigma)}{\sigma^2}\|\epsilon - \epsilon_\theta(x_0+\sigma\epsilon, \sigma)\|^2\right]$
- **多尺度得分匹配与噪声调度**（第6章6.5节→第7章）
  - 从单一σ到噪声调度 $\{\sigma_i\}_{i=1}^L$
  - NCSN/VE-SDE：$x_t = x_0 + \sigma_t\epsilon$
  - VP-SDE/DDPM：$x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$
  - 采样路径的终点：**训练去噪器 = 学习得分函数 = 驱动逆向扩散采样**

**来源**：第5章5.3 Tweedie等式；第6章6.3 DSM、6.6 参数化、6.5 多尺度；第7章7.2-7.3 SDE

---

### 12.2 变分路径回顾：从ELBO到VLB

**核心观点**：变分路径的核心链条是"ELBO→VAE→层级VAE→VLB"——ELBO将似然最大化转化为变分下界最大化，VAE引入重参数化技巧，层级VAE增加潜变量层次，扩散模型是无限层层级VAE的极限。本节回顾这条路径的关键数学结果，聚焦于ε预测VLB的形式，为等价性证明做准备。

- **ELBO与变分推断**（第8章核心结果）
  - $\log p(x) \geq \text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_\text{KL}(q(z|x)\|p(z))$
  - ELBO最大化 = KL最小化 = 变分推断
  - ELBO的两种分解：重建+正则化 / 联合熵+后验匹配
- **VAE→层级VAE→扩散**（第9-10章演化链）
  - VAE：单层编码-解码，重参数化技巧
  - 层级VAE：多层潜变量，逐层ELBO分解
  - 扩散模型 = 层级VAE的极限：$T\to\infty$，编码器固定为加噪过程
- **VLB分解与ε预测形式**（第11章核心结果）
  - VLB三分解：$L_T$（先验匹配，常数）+ $L_{t-1}$（一致性项，核心训练项）+ $L_0$（重建项）
  - 正向过程后验：$q(x_{t-1}|x_t,x_0) = \mathcal{N}(\tilde\mu_t, \sigma_t^2 I)$
  - 一致性项简化：$L_{t-1} = \frac{w_t}{2}\|\tilde\mu_t - \mu_\theta\|^2 \propto w_t\|\epsilon - \epsilon_\theta(x_t,t)\|^2$
  - ε预测VLB：$L_\text{VLB} = \sum_{t=1}^T w_t\,\mathbb{E}_{x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t,t)\|^2\right]$
  - 权重 $w_t = \frac{\beta_t^2}{2\tilde\beta_t\alpha_t(1-\bar\alpha_t)}$，$\tilde\beta_t = \frac{(1-\bar\alpha_{t-1})\beta_t}{1-\bar\alpha_t}$
- **简化VLB**（第11章11.4节）
  - DDPM发现：丢弃权重 $w_t$，简化为 $L_\text{simple} = \mathbb{E}_{t,x_0,\epsilon}[\|\epsilon - \epsilon_\theta(x_t,t)\|^2]$
  - 简化VLB在实践中效果更好（重点训练大噪声步的去噪）
- **变分路径的终点：训练去噪器 = 最大化ELBO = 最大化数据似然下界**

**来源**：第8章8.2 ELBO推导；第9章VAE；第10章层级VAE；第11章11.1-11.4 VLB与参数化

---

### 12.3 结构对比：殊途为何同归？

**核心观点**：将12.1节和12.2节的结果并排对比，可以清晰看到：DSM损失与ε预测VLB的核心结构完全一致——都是"预测添加的噪声ε"的MSE损失，差异仅在于含噪输入的形式和时间权重。这种结构相似性暗示了深层的数学等价性。

- **两个损失函数的并排对比**

  | 维度 | DSM损失（采样路径） | ε预测VLB（变分路径） |
  |---|---|---|
  | 噪声变量 | $\sigma$（噪声水平） | $t$（时间步，对应 $\sqrt{1-\bar\alpha_t}$） |
  | 含噪输入 | $x_0 + \sigma\epsilon$（VE-SDE） | $\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$（VP-SDE） |
  | 预测目标 | $\epsilon$（噪声） | $\epsilon$（噪声） |
  | 核心结构 | $\|\epsilon - \epsilon_\theta(\cdot)\|^2$ | $\|\epsilon - \epsilon_\theta(\cdot)\|^2$ |
  | 时间权重 | $\lambda(\sigma)/\sigma^2$ | $w_t = \frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$ |
  | 噪声采样 | $\sigma \sim p(\sigma)$ | $t \sim \mathcal{U}\{1,T\}$ |

- **关键观察**
  - 两个损失的核心都是 $\|\epsilon - \epsilon_\theta(\text{含噪输入}, \text{条件})\|^2$——预测噪声
  - 差异仅有两处：
    1. 含噪输入的形式（VE vs VP的噪声注入方式不同，但可统一）
    2. 时间权重（$\lambda(\sigma)/\sigma^2$ vs $w_t$）
  - 当 $\lambda(\sigma) = \sigma^2$（NCSN默认），DSM权重退化为1 → 简化VLB
  - 当 $\lambda(\sigma)/\sigma^2 = w_t$，DSM与完整VLB权重一致 → **DSM ≡ VLB**

- **直觉解释：等价性为何必然？**
  - 得分匹配优化"得分估计的准确性"：让模型知道概率分布的梯度指向
  - VLB优化"数据对数似然的下界"：让模型对数据赋予高概率
  - 得分函数 $\nabla\log p(x)$ 正是"概率增长最快方向"
  - 准确估计得分 = 对数据赋予高概率——二者是同一函数的不同表示
  - 类比：知道梯度→积分得函数值；知道函数值→微分得梯度；信息量等价

- **从Tweedie等式看等价性的根源**
  - Tweedie等式 $\nabla\log p_\sigma(x) = (D_\sigma(x) - x)/\sigma^2$ 桥接了去噪器与得分函数
  - DSM通过Tweedie从去噪器提取得分 → 采样路径的核心
  - VLB通过ELBO从去噪器最大化似然 → 变分路径的核心
  - 两条路径的交汇点正是去噪器——它同时承载了得分信息和似然信息

**来源**：第6章6.3 DSM、6.6 参数化；第11章11.5 交汇预告；Ho et al. (2020) DDPM；Song & Ermon (2019) NCSN

---

### 12.4 DSM ≡ VLB：等价性的形式化证明

**核心观点**：本节给出DSM损失与变分下界等价性的严格数学证明。核心论证链：ε预测VLB的核心项 $\propto w_t\|\epsilon - \epsilon_\theta\|^2$；DSM的核心项 $\propto (\lambda(\sigma)/\sigma^2)\|\epsilon - \epsilon_\theta\|^2$；当权重匹配时二者完全等价。证明分三步走：统一记号→逐项对应→权重等价。

- **第一步：统一记号与噪声注入**
  - VE-SDE (NCSN/SMLD) 的加噪：$\tilde x = x_0 + \sigma_t\epsilon$
  - VP-SDE (DDPM) 的加噪：$x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$
  - 二者可以通过Karras et al. (2022) 的统一框架相互转换
  - 统一表示：$x_t = s(t)x_0 + s(t)\sigma(t)\epsilon$，其中 $s(t)$ 为缩放因子，$\sigma(t)$ 为噪声水平
  - VE-SDE：$s(t)=1, \sigma(t)=\sigma_t$；VP-SDE：$s(t)=\sqrt{\bar\alpha_t}, \sigma(t)=\sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$

- **第二步：ε预测VLB的逐项展开**
  - 从第11章的VLB分解出发：
    $$L_\text{VLB} = \underbrace{L_T}_{\text{常数}} + \sum_{t=2}^T \underbrace{L_{t-1}}_{\text{训练项}} + \underbrace{L_0}_{\text{可合并}}$$
  - 每个训练项的ε预测形式（第11章定理）：
    $$L_{t-1} = \mathbb{E}_{x_0,\epsilon}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\|\epsilon - \epsilon_\theta(x_t,t)\|^2\right] + C_t$$
  - 合并后（第11章11.3节）：
    $$L_\text{VLB} = \sum_{t=1}^T w_t\,\mathbb{E}_{x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t,t)\|^2\right] + \text{const}$$

- **第三步：DSM损失的逐项展开**
  - 从第6章的DSM目标出发（多尺度版本）：
    $$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\sum_{i=1}^L \lambda(\sigma_i)\,\mathbb{E}_{x_0}\mathbb{E}_\epsilon\left[\left\|s_\theta(x_0+\sigma_i\epsilon,\sigma_i) + \frac{\epsilon}{\sigma_i}\right\|^2\right]$$
  - 转换为ε预测参数化（$s_\theta = -\epsilon_\theta/\sigma$）：
    $$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\sum_{i=1}^L \frac{\lambda(\sigma_i)}{\sigma_i^2}\,\mathbb{E}_{x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_0+\sigma_i\epsilon,\sigma_i)\|^2\right]$$

- **第四步：权重匹配与等价性**
  - 比较两个目标：
    - VLB：$\sum_t w_t\|\epsilon - \epsilon_\theta\|^2$，$w_t = \frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$
    - DSM：$\sum_i \frac{\lambda(\sigma_i)}{2\sigma_i^2}\|\epsilon - \epsilon_\theta\|^2$
  - **关键等式**：选择 $\lambda(\sigma_i)$ 使得 $\frac{\lambda(\sigma_i)}{2\sigma_i^2} = w_t$（其中 $\sigma_i$ 与 $t$ 对应）
  - 此时 $\mathcal{J}_\text{DSM}(\theta) = L_\text{VLB}(\theta) + \text{const}$
  - **定理（DSM ≡ VLB）**：在VP-SDE设定下，选择 $\lambda(t)$ 使得 $\frac{\lambda(t)}{2(1-\bar\alpha_t)} = w_t$，即 $\lambda(t) = \frac{\beta_t(1-\bar\alpha_t)}{(1-\bar\alpha_{t-1})\alpha_t}$ 时，DSM损失与VLB仅差常数，因此最优解完全一致。

- **简化目标下的等价性**
  - 简化VLB：$L_\text{simple} = \mathbb{E}_{t,x_0,\epsilon}[\|\epsilon - \epsilon_\theta(x_t,t)\|^2]$（丢弃 $w_t$）
  - 简化DSM：$\lambda(\sigma) = \sigma^2$ 时，权重退化为1
  - **简化VLB ≡ 简化DSM**（均匀权重下完全一致）
  - 这正是DDPM (Ho et al. 2020) 与NCSN (Song & Ermon 2019) 的训练目标等价

- **三种参数化下的等价性**
  - ε预测：$\|\epsilon - \epsilon_\theta\|^2$（两条路径的共同语言）
  - $x_0$预测：$\|x_0 - \hat{x}_\theta\|^2$（通过Tweedie等价转换）
  - 得分预测：$\|s + s_\theta\|^2$（通过Tweedie缩放等价转换）
  - 三种参数化是同一目标的不同坐标表示

**来源**：Ho et al. (2020) DDPM Eq. 12, 14；Song & Ermon (2019) NCSN；Luo (2022) 统一视角；Song et al. (2021) SDE；Kingma (2021) VDM

---

### 12.5 连续时间视角的统一

**核心观点**：离散时间的等价性在连续时间极限下获得更深刻的形式——SDE框架（采样路径）与VDM框架（变分路径）在连续时间下完全统一。连续时间VLB的权重与连续时间DSM的权重天然一致，进一步验证了等价性的必然性。

- **SDE形式（采样路径的连续极限）**
  - 正向SDE：$dx = f(x,t)dt + g(t)dw$
  - 逆向SDE：$dx = [f - g^2\nabla\log p_t]dt + g\,d\bar{w}$
  - 得分匹配训练：学习 $\nabla\log p_t(x)$

- **VDM形式（变分路径的连续极限）**
  - 连续时间ELBO（Kingma 2021）：
    $$\log p(x_0) \geq \mathbb{E}_q\left[\log\frac{p(x_0|x_\epsilon)}{q(x_\epsilon|x_0)}\right] + \int_0^1 \mathbb{E}_q\left[\frac{1}{2}\left\|\frac{\mu_\theta - \tilde\mu_t}{\sigma_t}\right\|^2\right]dt$$
  - 连续时间下的权重与DSM权重一致

- **等价性的连续时间表述**
  - 离散→连续：$\sigma_t^2 \to \sigma^2(t)$，$\bar\alpha_t \to \bar\alpha(t)$，$\beta_t \to \beta(t)$
  - 连续时间VLB的权重自然对应连续时间DSM的权重
  - Kingma (2021) 证明：三种参数化在连续时间下等价

- **概率流ODE：两条路径的第三种交汇**
  - PF-ODE从Fokker-Planck方程推导，将随机逆向SDE转化为确定性ODE
  - PF-ODE既可从SDE框架推导（第7章7.4），也可从变分框架推导
  - DDIM = PF-ODE的离散化
  - 三种采样方式的统一：随机（逆向SDE）→ 半随机（DDPM）→ 确定性（PF-ODE/DDIM）

- **三方统一**
  - DDPM：固定噪声调度 + 简化VLB
  - Score-SDE：固定SDE + 得分匹配
  - VDM：可优化噪声调度 + 连续时间VLB
  - 三者是同一框架在不同约束条件下的特例

**来源**：Song et al. (2021) SDE Sec 4; Kingma (2021) VDM；第7章7.4 PF-ODE；第11章附录11B

---

### 12.6 实践意义与训练目标选择

**核心观点**：DSM ≡ VLB的等价性不仅是理论上的优美结论，更有直接的实践意义——它意味着同一个扩散模型可以从两种视角理解和训练，而训练目标的选择（简化VLB vs 完整VLB vs 自定义权重）影响模型的生成质量和似然性能。

- **同一扩散模型的两种训练视角**
  - 采样视角：我在学习数据分布的得分函数，用来驱动逆向扩散采样
  - 变分视角：我在最大化数据的对数似然下界，训练一个潜变量生成模型
  - 两种视角对应同一个训练算法、同一个网络架构、同一个采样过程

- **ε预测：两条路径的公共语言**
  - ε预测是DSM与VLB的共同参数化
  - 它也是实际工程中最常用的参数化方式
  - ε预测的物理直觉：预测"添加了什么噪声"，比预测"原始信号是什么"更容易学习

- **训练目标选择指南**
  - **简化VLB / 简化DSM**（$\lambda=\sigma^2$，均匀权重）
    - 优点：实现简单，实践中生成质量最好
    - 缺点：不是严格的最大似然，codelength次优
    - 适用：追求生成质量（FID/IS）的场景
  - **完整VLB / 加权DSM**（$\lambda$使得权重匹配VLB）
    - 优点：严格的最大似然下界，codelength最优
    - 缺点：权重随时间剧烈变化，小t项被过度强调，训练不稳定
    - 适用：追求似然/压缩的场景
  - **自定义权重**（如Importance Sampling）
    - 对训练过程中方差大的时间步赋予更高权重
    - 在完整VLB和简化VLB之间取得平衡
    - 适用：需要兼顾质量和似然的场景

- **噪声调度与时间加权的实践影响**
  - 噪声调度的选择（线性/余弦/SNR-感知）影响权重分布
  - 时间步采样策略（均匀 vs 重要性采样）影响训练效率
  - Improved DDPM：学习方差参数，兼顾质量和似然

**来源**：Ho et al. (2020) DDPM Sec 3.4；Kingma (2021) VDM；Nichol & Dhariwal (2021) Improved DDPM；Karras et al. (2022) 统一框架

---

### 12.7 全书叙事的汇合

**核心观点**：DSM ≡ VLB的等价性验证了全书核心论点——扩散模型是逆问题自然发展的产物。两条路径殊途同归不是偶然，而是贝叶斯框架内在逻辑的必然结果：逆问题需要后验采样 → 采样需要得分 → 得分可从去噪器提取（Tweedie） → 去噪器可通过得分匹配训练 → 多步得分驱动采样即扩散模型。变分路径提供了从最大似然出发的等价论证，印证了"逆问题的需求"与"生成建模的独立探索"在数学上殊途同归。

- **两条路径的时间线对照**
  - 采样路径（第4-7章）：ULA → Langevin → Score → Tweedie → DSM → Diffusion(SDE)
  - 变分路径（第8-11章）：ELBO → VAE → 层级VAE → VLB → Diffusion(VLB)
  - 汇合点（第12章）：Score ≡ ELBO → DSM ≡ VLB

- **核心论点的验证**
  - 论点：扩散模型是逆问题自然发展的终点
  - 验证：逆问题的后验采样需求 → 驱动每一步方法演化 → 最终走向扩散模型
  - 互补验证：从最大似然出发的变分路径也走向扩散模型
  - 两条独立路径的汇合证明：扩散模型不是特例，而是贝叶斯框架的必然终点

- **"殊途同归"的深层原因**
  - 贝叶斯框架中，$\log p(x)$ 与 $\nabla\log p(x)$ 是同一分布的两种等价描述
  - VLB优化 $\log p(x)$ 的下界；DSM优化 $\nabla\log p(x)$ 的估计精度
  - 函数值与梯度的信息等价性，决定了两条路径必然等价
  - 这是更一般的数学原理：优化一个函数 vs 优化其梯度，在适当条件下等价

- **展望第13章：从等价性到条件生成**
  - 等价性建立了无条作扩散模型的统一理论
  - 第13章将等价性扩展到条件扩散：$p(x|y)$
  - 条件扩散采样 = 逆问题求解 → 完成全书闭环

**来源**：全书叙事；book_plan.md 核心论点

---

## 附录

### 附录12A 连续时间得分匹配与变分下界的等价性推导

- 连续时间ESM目标函数
- 连续时间VLB的积分形式
- 变分法推导权重等价
- Kingma (2021) 定理的证明概要

### 附录12B Karras et al. (2022) 统一框架：VE-SDE与VP-SDE的等价性

- 统一参数化：$x_t = s(t)x_0 + s(t)\sigma(t)\epsilon$
- VE-SDE和VP-SDE在不同 $(s,\sigma)$ 下的特例
- 统一框架下的权重表达式
- 从VE-SDE到VP-SDE的权重映射

---

## 素材来源映射

| 节 | 核心素材 | 补充来源 |
|---|---|---|
| 12.0 | 全书叙事 | book_plan.md |
| 12.1 | 第5章5.3、第6章6.3/6.5/6.6、第7章7.2-7.3 | Song & Ermon (2019) NCSN |
| 12.2 | 第8章8.2、第11章11.1-11.4 | Tutorial_Diffusion Theorem 2.3-2.7 |
| 12.3 | 第11章11.5、第6章6.3/6.6 | Ho et al. (2020) DDPM |
| 12.4 | Ho et al. (2020) Eq.12,14; Luo (2022) | Tutorial_Diffusion Sec 2-3; 2406.08929v2 |
| 12.5 | 第7章7.4、第11章附录11B | Song et al. (2021) SDE; Kingma (2021) VDM |
| 12.6 | Ho et al. (2020) Sec 3.4 | Nichol & Dhariwal (2021); Karras et al. (2022) |
| 12.7 | 全书叙事 | book_plan.md |

---

## 缺失素材清单

| 素材 | 用途 | 状态 | 替代方案 |
|---|---|---|---|
| Luo (2022) "Understanding Diffusion Models: A Unified Perspective" 全文 | 12.4等价性证明的主要参考 | ❌ 未找到全文 | 使用Tutorial_Diffusion和DDPM论文的内容推导 |
| Song & Kingma 统一视角的原始推导 | 12.4-12.5权重匹配的细节 | ❌ 未找到 | 从Song et al. (2021) SDE论文和Kingma (2021) VDM论文分别推导 |
| Kingma & Gao (2023) "Understanding diffusion objectives as the ELBO with simple data augmentation" | 12.5连续时间等价性 | ❌ 未找到 | 使用附录11B已有内容 |
| 实验对比图（简化VLB vs 完整VLB vs DSM的FID/codelength对比） | 12.6实践意义 | ❌ 无 | 参考Ho et al. (2020) Table 1的文字描述 |
| 两条路径时间线对照图 | 12.0/12.7 | ❌ 需绘制 | 文字描述+ASCII图 |
