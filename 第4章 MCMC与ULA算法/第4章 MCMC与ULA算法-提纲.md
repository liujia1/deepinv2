# 第4章 MCMC与ULA算法 — 提纲

> **章节定位**：承接第3章末尾的分叉点，走"从后验中采样"的路径。第3章告诉我们MAP只是后验众数，丢失了不确定性信息——我们需要从后验分布中**采样**，而非仅仅优化。本章从Monte Carlo积分出发，逐步建立MCMC采样工具箱：Metropolis-Hastings是通用框架，ULA是高效特例（利用梯度信息），MYULA将ULA扩展到不可微先验，Gibbs采样提供了另一条路径（利用条件分布结构），加速方法（过松弛、ILA）则提升采样效率。本章的终点是：读者能够从任意后验分布中高效采样，并用采样结果进行不确定性量化——这是采样路径的起点，后续章节将从ULA走向Langevin SDE（第5章）、得分匹配（第6章）和扩散模型（第7章）。

> **叙事主线**：Monte Carlo积分：为什么需要采样(4.1) → Metropolis-Hastings：通用MCMC框架(4.2) → ULA：利用梯度的高效采样(4.3) → MYULA：不可微先验的ULA(4.4) → Gibbs采样：利用条件结构的采样(4.5) → 加速采样：过松弛与惯性Langevin(4.6) → 收敛诊断与不确定性量化(4.7)

---

## 4.0 本章导读：从优化到采样——后验分布的探索

第3章的结尾留下了一个根本性问题：MAP只给出后验众数，而众数≠典型。高维空间中众数可能远离均值，多峰后验中众数只代表一个峰。不确定性量化、多解识别、条件生成——这些任务都需要后验分布的完整信息，而非单点估计。

从后验中提取完整信息的途径是**采样**：生成一组样本 $X_1, \ldots, X_M \sim p(x|y)$，然后用样本均值近似后验期望：

$$\mathbb{E}[h(x)|y] \approx \frac{1}{M}\sum_{m=1}^M h(X_m)$$

这个简单的想法——**Monte Carlo积分**——是整个采样路径的基石。问题在于：如何从无法直接采样的高维后验分布中生成样本？这正是本章的核心问题。

**Monte Carlo积分**（4.1）：后验期望的近似需要样本，但后验分布通常无法直接采样——这正是Monte Carlo方法面临的根本挑战。大数定律保证了样本均值的收敛性，但独立性假设在高维后验中难以满足。

**Metropolis-Hastings算法**（4.2）：MCMC的核心思想是构造一条马尔可夫链，使其平稳分布恰好是目标后验。Metropolis-Hastings通过"提议-接受/拒绝"机制实现这一点——只需知道后验密度（到归一化常数），无需直接采样。细致平衡条件是正确性的数学保证。

**ULA：利用梯度的高效采样**（4.3）：MH的随机游走提议在高维空间效率极低。ULA利用后验的梯度信息构造"有方向的"提议——Langevin扩散的Euler离散化。ULA有偏（不满足细致平衡），但收敛速度远快于随机游走MH——这是采样路径中"梯度信息"的首次登场，为第5章的Langevin SDE和第7章的扩散模型埋下伏笔。

**MYULA：不可微先验的ULA**（4.4）：图像逆问题的后验常含不可微项（TV范数、指示函数），ULA的梯度要求被违反。MYULA用Moreau-Yoshida包络光滑化不可微项，在可微的近似分布上运行ULA——近似误差可控，收敛性有非渐近保证。

**Gibbs采样：利用条件结构的采样**（4.5）：与ULA利用梯度信息不同，Gibbs采样利用后验的条件分布结构。通过引入辅助变量，将复杂分布分解为简单的条件分布——Gibbs是"有方向的随机游走"的另一种形式。GLM（Gaussian Latent Machine）框架展示了Gibbs采样在TV先验下的优雅实现：Gibbs采样就是"带噪声的交替最小化"。

**加速采样**（4.6）：ULA的收敛速率可以通过过松弛和惯性机制加速——正如Nesterov加速之于梯度下降。过松弛Gibbs减小样本的自相关，惯性Langevin算法（ILA）引入动量项，其连续极限是欠阻尼Langevin SDE——优化与采样的结构对偶在此达到高潮。

**收敛诊断与不确定性量化**（4.7）：MCMC的输出是一条马尔可夫链，如何判断它是否已经收敛？Burn-in期、自相关函数、有效样本量（ESS）是三个核心诊断工具。收敛之后，采样结果如何用于不确定性量化？后验可信区域、HPD区间、像素级置信区间——从采样到决策的最后一步。

```
Monte Carlo(4.1) → MH算法(4.2) → ULA/梯度采样(4.3) → MYULA/不可微(4.4) → Gibbs/条件结构(4.5) → 加速方法(4.6) → 诊断与UQ(4.7)
```

本章结束时，读者将拥有后验采样的完整工具箱——从通用框架（MH）到高效特例（ULA/MYULA），从梯度驱动（ULA）到结构驱动（Gibbs），从基本方法到加速方法。更重要的是，读者将理解采样的目的不只是"生成样本"，而是**从后验中提取完整信息**——期望、方差、可信区域、假设检验。下一章将从ULA的离散迭代走向Langevin SDE的连续极限，揭示得分函数的核心角色。

---

## 4.1 Monte Carlo方法：从积分到采样

**核心观点**：后验推断的核心计算任务是求期望 $\mathbb{E}[h(x)|y] = \int h(x)\,p(x|y)\,dx$——高维积分无法解析计算，Monte Carlo方法用随机样本近似积分，是大数定律的直接应用。但"独立采样"在高维后验中通常不可行，这正是MCMC方法的出发点。

- **后验推断的核心计算任务**
  - 期望：$\mathbb{E}[x|y]$（MMSE估计）、$\mathbb{E}[x^2|y]$（方差）、$\mathbb{E}[\mathbf{1}_A(x)|y]$（概率）
  - 统一形式：$\mathbb{E}[h(x)|y] = \int h(x)\,p(x|y)\,dx$——高维积分
  - 高维积分的困难：维数诅咒——确定性数值积分在 $d > 10$ 时已不可行

- **Monte Carlo估计**
  - 给定独立同分布样本 $X_1, \ldots, X_M \sim p(x|y)$：
    $$\hat{I}_M = \frac{1}{M}\sum_{m=1}^M h(X_m) \xrightarrow{M \to \infty} \mathbb{E}[h(x)|y]$$
  - 收敛速率：$O(1/\sqrt{M})$（大数定律 + 中心极限定理）——**与维数 $d$ 无关**
  - 这是Monte Carlo方法的核心优势：收敛速率不依赖维数

- **根本困难：如何从后验中采样？**
  - 独立采样需要知道后验的归一化常数 $p(y) = \int p(y|x)p(x)\,dx$——通常不可解
  - 即使知道归一化常数，高维分布的直接采样（逆CDF、拒绝采样）也效率极低
  - **出路**：不要求独立采样，构造一条**马尔可夫链**，使其平稳分布为 $p(x|y)$——MCMC

- **📌 侧栏：Monte Carlo与优化的对比**
  - 优化（第3章）：从后验中提取一个点（MAP）→ 计算 $\arg\max_x p(x|y)$
  - Monte Carlo：从后验中提取分布信息 → 计算 $\mathbb{E}[h(x)|y]$
  - 优化只需梯度，Monte Carlo需要样本——但样本的获取比梯度困难得多
  - 两者的收敛速率对比：优化 $O(1/k^2)$（FISTA）vs Monte Carlo $O(1/\sqrt{M})$——但Monte Carlo提供的是分布信息，不是单点

**来源**：Pereyra L1 P30-35; LectureNotes2020_v2 Section 8.2.1; invprobs_v2 Ch12

> **过渡**：Monte Carlo方法的瓶颈是"如何从后验中采样"。MCMC方法通过构造马尔可夫链巧妙地绕过了直接采样的困难——Metropolis-Hastings算法是MCMC的经典框架。

---

## 4.2 Metropolis-Hastings算法

**核心观点**：Metropolis-Hastings（MH）算法通过"提议-接受/拒绝"机制构造一条以 $p(x|y)$ 为平稳分布的马尔可夫链。它的核心优势是：只需要后验密度到归一化常数（$p(x|y) \propto p(y|x)p(x)$），无需计算 $p(y)$——这恰好是贝叶斯框架下最自然的信息。

- **马尔可夫链与平稳分布**
  - 马尔可夫链定义：$X_{m+1}|X_m \sim K(\cdot|X_m)$，转移只依赖当前状态
  - 平稳分布：若 $\mu K = \mu$（$\mu$ 经一步转移后不变），则 $\mu$ 是平稳分布
  - 目标：构造 $K$ 使得 $p(x|y)$ 是 $K$ 的平稳分布

- **细致平衡条件**
  - 定义：$K$ 关于 $p(x|y)$ 满足细致平衡，若
    $$p(x|y)\,K(x'|x) = p(x'|y)\,K(x|x')$$
  - 细致平衡 $\Rightarrow$ 平稳分布（充分条件，非必要）
  - 直觉：从 $x$ 转移到 $x'$ 的"流量"等于从 $x'$ 转移到 $x$ 的"流量"——平衡

- **Metropolis-Hastings算法**
  - 提议核 $Q(x'|x)$：从当前状态 $x$ 提议新状态 $x'$
  - 接受概率：
    $$\alpha(x, x') = \min\left\{1, \frac{p(x'|y)\,Q(x|x')}{p(x|y)\,Q(x'|x)}\right\}$$
  - 算法步骤：
    1. 从 $Q(\cdot|X_m)$ 采样提议 $X^*$
    2. 以概率 $\alpha(X_m, X^*)$ 接受：$X_{m+1} = X^*$；否则拒绝：$X_{m+1} = X_m$
  - MH转移核：$K_{\text{MH}}(x'|x) = Q(x'|x)\alpha(x, x') + \delta_x(x')\left[1 - \int Q(x''|x)\alpha(x, x'')\,dx''\right]$

- **MH的正确性证明**
  - 命题：$K_{\text{MH}}$ 关于 $p(x|y)$ 满足细致平衡
  - 证明核心：分情况讨论接受和拒绝部分，验证等式成立
  - 关键观察：$p(x|y)$ 中的归一化常数 $p(y)$ 在比值 $\frac{p(x'|y)}{p(x|y)}$ 中被约去——**MH不需要归一化常数**

- **三种经典提议核**
  - **独立采样器**：$Q(x'|x) = \rho(x')$，提议与当前状态无关
    - 接受概率 $\alpha = \min\{1, \frac{p(x'|y)/\rho(x')}{p(x|y)/\rho(x)}\}$
    - 适合后验有好的近似（如Laplace近似）时
  - **随机游走MH**：$Q(x'|x) = \rho(x'-x)$，对称提议
    - 接受概率简化为 $\alpha = \min\{1, \frac{p(x'|y)}{p(x|y)}\}$——不依赖提议分布
    - 高维空间效率极低：接受率随维数指数衰减
  - **pCN-MCMC（预条件Crank-Nicolson）**：$Q(x'|x) = \mathcal{N}(\sqrt{1-\beta^2}\,x, \beta^2 C)$
    - 接受概率只依赖似然比 $\frac{p(y|x')}{p(y|x)}$——与先验无关
    - 高维/无穷维中特别重要：接受率不随维数下降
    - **来源**：LectureNotes2020_v2 Example 8.2.7; Cotter et al. (2013)

- **📌 侧栏：MH的物理直觉——"试错法"**
  - MH像一个谨慎的探险者：先试探性地走一步（提议），然后评估这一步是否"更好"（后验比值）
  - 更好的步 → 一定接受；更差的步 → 以一定概率接受（避免陷入局部）
  - 关键：即使接受"更差"的步，长远来看链会收敛到正确的分布

**来源**：Pereyra L1 P35-40; LectureNotes2020_v2 Section 8.2 (lines 5041-5235); Pock L2 P28

> **过渡**：MH是通用框架，但其随机游走提议在高维空间效率极低——就像蒙眼随机行走，每步都很小。能否利用后验的**梯度信息**，让每一步都"朝着正确的方向"走？这就是ULA的思想——Langevin扩散的离散化。

---

## 4.3 ULA：Langevin采样的Euler离散

**核心观点**：ULA（Unadjusted Langevin Algorithm）利用后验的梯度 $\nabla\log p(x|y)$ 构造有方向的采样步骤——它是Langevin扩散过程的Euler-Maruyama离散化。ULA有偏（不满足细致平衡），但在正则性条件下收敛速度远快于随机游走MH——这是采样路径中"梯度信息"的首次登场。

- **从MH到ULA：为什么需要梯度信息？**
  - 随机游走MH在高维空间的困难：提议方向随机 → 大部分提议被拒绝 → 链移动极慢
  - 自然想法：如果能沿着 $\nabla\log p(x|y)$ 的方向提议，每一步都"朝着高概率区域"走
  - $\nabla\log p(x|y)$ 的含义：后验密度增长最快的方向——"最陡上升方向"

- **Langevin扩散与ULA**
  - Langevin扩散：$dX_t = \nabla\log p(X_t|y)\,dt + \sqrt{2}\,dW_t$
    - 漂移项 $\nabla\log p(x|y)dt$：沿梯度方向移动（确定性）
    - 扩散项 $\sqrt{2}\,dW_t$：布朗运动（随机性）
    - 理论结果：$X_t$ 的分布当 $t \to \infty$ 时收敛到 $p(x|y)$（第5章详细证明）
  - Euler-Maruyama离散化 → ULA：
    $$X_{m+1} = X_m + \delta\,\nabla\log p(X_m|y) + \sqrt{2\delta}\,Z_{m+1}, \quad Z_{m+1} \sim \mathcal{N}(0, I_n)$$
  - 两步解读：先沿梯度走一步（$\delta\nabla\log p$），再加高斯噪声（$\sqrt{2\delta}\,Z$）

- **ULA的后验梯度计算**
  - 势能函数：$U(x) = -\log p(x|y) = -\log p(y|x) - \log p(x) + \text{const}$
  - $\nabla\log p(x|y) = -\nabla U(x) = -\nabla(-\log p(y|x)) - \nabla(-\log p(x))$
  - 高斯似然 + 高斯先验：$\nabla\log p(x|y) = -\frac{1}{\sigma^2}A^T(Ax-y) - \frac{1}{\sigma_x^2}x$（梯度下降的梯度！）
  - ULA = 梯度下降 + 噪声 → 优化与采样的统一视角

- **步长选择与收敛性**
  - 步长条件：$\delta \leq 1/L$，其中 $L$ 是 $\nabla\log p(x|y)$ 的Lipschitz常数
    - 与梯度下降的步长条件 $τ \leq 1/L$ 完全一致（第3章3.2节）
    - 步长太大 → 离散误差大 → 链发散
    - 步长太小 → 移动太慢 → 收敛慢
  - ULA的偏差：ULA不满足细致平衡，样本分布不精确等于 $p(x|y)$
    - 偏差来源：Euler离散化引入的误差
    - 偏差量级：$\delta$ 越小偏差越小，但收敛越慢
  - ULA的非渐近收敛界：在强对数凹条件下，$\|p_m - p\|_{\text{TV}} \leq C\,\delta\,m$（Durmus & Moulines, 2019）

- **MH vs ULA：有偏但高效**
  - MH：无偏（满足细致平衡），但高维空间接受率低、收敛慢
  - ULA：有偏（不满足细致平衡），但利用梯度信息，收敛速度远快于MH
  - 实际选择：
    - 维数低/后验简单 → MH足够
    - 维数高/后验光滑 → ULA显著更优
    - 需要无偏采样 → MALA（Metropolis-Adjusted Langevin Algorithm）：ULA + MH校正步
  - 📌 **侧栏：MALA——ULA与MH的折中**
    - MALA = ULA提议 + MH接受/拒绝步
    - 既有梯度的方向性，又有MH的无偏性
    - 代价：每步需要额外计算接受概率
    - 实践中，当ULA偏差可接受时（图像逆问题中常如此），ULA优于MALA——计算效率更高

- **📌 侧栏：ULA = 梯度下降 + 噪声**
  - 梯度下降（第3章）：$x_{k+1} = x_k - \tau\nabla f(x_k)$——确定性，收敛到众数
  - ULA（本章）：$X_{m+1} = X_m + \delta\nabla\log p(X_m|y) + \sqrt{2\delta}\,Z_{m+1}$——随机性，收敛到分布
  - 当步长 $\delta \to 0$、噪声项 $\sqrt{2\delta}\,Z \to 0$，ULA退化为梯度下降
  - 噪声是"探索"的代价——没有噪声，只有"开发"（众数）；有噪声，才有"探索"（分布）
  - 这一洞见将在第5章的"扩散模型=高温Langevin"中再次出现

**来源**：Pereyra L1 P40-50; Pock L2 P14-17; lab1_ULA_sol; Durmus & Moulines (2019)

> **过渡**：ULA要求 $\nabla\log p(x|y)$ 存在且Lipschitz连续——但图像逆问题的后验常含不可微项（TV范数、L1范数、指示函数），ULA的梯度要求被违反。MYULA通过Moreau-Yoshida包络光滑化不可微项，巧妙地将ULA扩展到不可微情形。

---

## 4.4 MYULA：近端ULA

**核心观点**：图像逆问题的后验 $p(x|y) \propto \exp\{-f(x) - g(x)\}$ 中，$g(x)$（如TV范数）通常不可微。MYULA用Moreau-Yoshida包络 $g_\lambda$ 光滑化 $g$，在可微的近似分布 $p_\lambda(x|y)$ 上运行ULA——近似误差由 $\lambda$ 控制，收敛性有非渐近保证。

- **不可微后验的挑战**
  - 典型后验结构：$p(x|y) \propto \exp\{-f(x) - g(x)\}$
    - $f(x) = \frac{1}{2\sigma^2}\|y - Ax\|^2$：光滑（$L_f$-Lipschitz梯度）
    - $g(x) = \lambda\|Bx\|_1 + \iota_S(x)$：不可微（TV范数、指示函数）
  - 问题：$\nabla g(x)$ 不存在 → $\nabla\log p(x|y)$ 不存在 → ULA不可用
  - 解决思路：光滑化 $g$，使近似后的 $p_\lambda$ 可微

- **Moreau-Yoshida包络**
  - 定义：$g_\lambda(x) = \inf_u \left\{g(u) + \frac{1}{2\lambda}\|u - x\|^2\right\}$
  - 关键性质：
    1. $g_\lambda$ 定义了一个真密度：$p_\lambda(x|y) = \frac{\exp\{-f(x) - g_\lambda(x)\}}{\int \exp\{-f(x) - g_\lambda(x)\}dx}$
    2. 对数凹性与可微性：$p_\lambda$ 是对数凹的，且 $p_\lambda \in C^1$（即使 $p$ 不可微）
    3. 梯度表达式：$\nabla g_\lambda(x) = \frac{x - \text{prox}_{\lambda g}(x)}{\lambda}$
    4. Lipschitz常数：$\nabla\log p_\lambda$ 是 $L$-Lipschitz的，$L \leq L_f + 1/\lambda$
  - 近端算子回顾（第3章3.4节）：$\text{prox}_{\lambda g}(x) = \arg\min_u \{g(u) + \frac{1}{2\lambda}\|u - x\|^2\}$

- **MYULA算法**
  - 在 $p_\lambda$ 上运行ULA：
    $$X_{m+1} = X_m + \delta\,\nabla\log p_\lambda(X_m|y) + \sqrt{2\delta}\,Z_{m+1}$$
  - 代入梯度表达式：
    $$X_{m+1} = \left(1 - \frac{\delta}{\lambda}\right)X_m - \delta\,\nabla f(X_m) + \frac{\delta}{\lambda}\,\text{prox}_{\lambda g}(X_m) + \sqrt{2\delta}\,Z_{m+1}$$
  - 三步解读：
    1. 沿光滑梯度走一步：$-\delta\,\nabla f(X_m)$
    2. 近端校正（替代不可微梯度）：$\frac{\delta}{\lambda}[\text{prox}_{\lambda g}(X_m) - X_m]$
    3. 加高斯噪声：$\sqrt{2\delta}\,Z_{m+1}$

- **MYULA的收敛性**
  - 近似误差：$\|p_\lambda - p\|_{\text{TV}} \leq \lambda\,L_g^2$（$g$ 是 $L_g$-Lipschitz时）
  - 非渐近误差界（Durmus et al., 2018, Theorem 3.1）：
    - 对 $\delta < \delta_\lambda^{\max} = (L_f + 1/\lambda)^{-1}$，存在 $\delta_\epsilon$ 和 $M_\epsilon$ 使得
    $$\|\delta_{x_0}\,Q_\delta^M - p\|_{\text{TV}} < \epsilon + \lambda\,L_g^2$$
    - $Q_\delta^M$ 是MYULA迭代 $M$ 步后的核
    - 若 $f + g$ 在球外强凸，则 $M_\epsilon = O(d\log d)$——**维数对数依赖**
  - 权衡：$\lambda$ 小 → 近似误差小，但 $L = L_f + 1/\lambda$ 大 → 步长 $\delta$ 小 → 收敛慢
  - $\lambda$ 大 → $L$ 小 → 步长大 → 收敛快，但近似误差大

- **📌 侧栏：MYULA与近端梯度下降的对偶**
  - 近端梯度下降（第3章3.4节）：$x_{k+1} = \text{prox}_{\tau g}(x_k - \tau\nabla f(x_k))$
  - MYULA：$X_{m+1} = (1 - \delta/\lambda)X_m - \delta\nabla f(X_m) + (\delta/\lambda)\text{prox}_{\lambda g}(X_m) + \sqrt{2\delta}\,Z_{m+1}$
  - 当 $\lambda = \tau$ 且 $\delta = \tau$ 时，MYULA ≈ 近端梯度下降 + 噪声
  - 优化与采样的结构对偶：近端梯度下降 → MYULA，正如梯度下降 → ULA

**来源**：Pereyra L1 P43-44; Pereyra L3 P9-11; Durmus et al. (2018); Pereyra (2015)

> **过渡**：ULA/MYULA利用后验的梯度信息构造采样步骤。另一种利用后验结构的方式是Gibbs采样——不利用梯度，而是利用条件分布的结构。通过引入辅助变量，将复杂后验分解为简单的条件分布，Gibbs采样在特定结构下极为高效。

---

## 4.5 Gibbs采样：利用条件结构的采样

**核心观点**：Gibbs采样不利用梯度信息，而是利用后验的条件分布结构——通过引入辅助变量，将复杂分布分解为简单的条件分布，然后交替采样。GLM（Gaussian Latent Machine）框架展示了Gibbs采样在TV先验下的优雅实现，也揭示了采样与优化之间深层对偶：Gibbs = 带噪声的交替最小化。

- **Gibbs采样的基本思想**
  - 考虑联合分布 $p(x, z)$，目标是边缘分布 $p(x) = \int p(x, z)\,dz$
  - Gibbs采样交替从条件分布中采样：
    $$x^{k+1} \sim p_{X|Z=z^k}, \quad z^{k+1} \sim p_{Z|X=x^{k+1}}$$
  - Gibbs采样是MH的特例：提议核为条件分布时，接受概率恒为1
  - 优势：不需要调节步长或提议分布——条件分布就是最优提议

- **从半二次最小化到GLM**
  - 半二次最小化回顾（优化视角）：
    - 表达 $f(x) = \min_z q(x, z)$，$q$ 关于 $x$ 是二次的
    - 交替最小化：$x^{k+1} = \arg\min_x q(x, z^k)$，$z^{k+1} = \arg\min_z q(x^{k+1}, z)$
    - TV的例子：$|t| = \min_{z > 0} \frac{t^2}{2z} + \frac{z}{2}$（Geman & Reynolds, 1992）
  - 在负对数域中，最小化变成soft-min：
    $$\lambda|t| = -\log\int \exp(-q(t, z))\,dz$$
    其中 $q(t, z)$ 关于 $t$ 是二次的
  - 从优化到采样的转化：$\min$ → $\text{softmin}$，交替最小化 → Gibbs采样

- **Gaussian Latent Machine（GLM）**
  - PoE（Product of Experts）先验：$p_X(x) \propto \prod_{j=1}^m \phi_j((Kx)_j)$
  - GLM提升表示：$p_{X,Z}(x, z) = \underbrace{\mathcal{N}(Kx|\tilde{\mu}(z), \tilde{\Sigma}(z))}_{p_{X|Z=z}(x)} \cdot \prod_{j=1}^m p_j(z_j)$
  - 条件分布：
    - $p_{X|Z}$：多元高斯 $\mathcal{N}(x|\mu(z), \Sigma(z))$——可以直接采样
    - $p_{Z|X}$：$m$ 个独立的一元分布——可以逐个采样
  - Gibbs采样在GLM上交替采样 $x$ 和 $z$，每一步都有闭式解

- **TV先验的GLM实现**
  - TV先验的Laplace因子：$\exp(-\lambda|t|)$ 是高斯尺度混合
    $$\frac{\lambda}{2}\exp(-\lambda|t|) = \int_{\mathbb{R}^+} \frac{1}{\sqrt{2\pi z}}\exp\left(-\frac{t^2}{2z}\right)\frac{\lambda^2}{2}\exp\left(-\frac{\lambda^2 z}{2}\right)dz$$
  - 条件分布 $p_{Z_j|X=t}$：广义逆高斯分布 GIG$(\lambda^2, t^2, 1/2)$——可以高效采样
  - Gibbs采样算法：
    - $x^{k+1} \sim \mathcal{N}(x|0, (K^T\text{diag}(z^k)^{-1}K)^{-1})$（多元高斯）
    - $z_j^{k+1} \sim \text{GIG}(\lambda^2, (Kx^{k+1})_j^2, 1/2)$，$j = 1, \ldots, m$（独立一元）
  - 扩展到后验采样：加入高斯似然项，仅修改第一步的高斯分布

- **Gibbs = 带噪声的交替最小化**
  - 高斯分布的Gibbs采样 vs Gauss-Seidel迭代：
    - Gibbs：$X_1^{k+1} \sim \mathcal{N}(\mu_1 - Q_{11}^{-1}Q_{12}(X_2^k - \mu_2), Q_{11}^{-1})$
    - Gauss-Seidel：$x_1^{k+1} = \mu_1 - Q_{11}^{-1}Q_{12}(x_2^k - \mu_2)$
    - **Gibbs就是带噪声的Gauss-Seidel！**
  - 结构对偶总结：

    | 优化 | 采样 |
    |---|---|
    | 交替最小化 | Gibbs采样 |
    | Gauss-Seidel | 带噪声的Gauss-Seidel |
    | 半二次最小化 | GLM |
    | 收敛到众数 | 收敛到分布 |

  - 📌 **侧栏：优化→采样的转化律**
    - $\min \to \text{softmin}$：优化是最小化，采样是soft-minimum（带指数权重的平均）
    - 梯度下降 $\to$ ULA：加噪声
    - 近端梯度下降 $\to$ MYULA：加噪声
    - 交替最小化 $\to$ Gibbs：加噪声
    - 统一规律：优化算法 + 噪声 = 采样算法——噪声将"收敛到点"变为"收敛到分布"

**来源**：Pock L2 P25-33; Geman & Reynolds (1992); Kuric, Zach, Habring, Unser & Pock (2025); Geman & Geman (1984)

> **过渡**：Gibbs采样利用条件分布结构，ULA利用梯度信息——两种策略各有优势。能否进一步加速采样？正如优化中的Nesterov加速之于梯度下降，采样中也有对应的加速机制——过松弛与惯性Langevin算法。

---

## 4.6 加速采样方法：过松弛与惯性Langevin算法

**核心观点**：采样算法的加速与优化算法的加速存在深层的结构对偶——过松弛Gibbs对应SOR加速，惯性Langevin算法（ILA）对应动量法/重球法。ILA的连续极限是欠阻尼Langevin SDE，其收敛速率优于标准Langevin扩散。这些加速方法在实践中可以显著减少MCMC的自相关，提高有效样本量。

- **过松弛Gibbs采样**
  - SOR加速（优化视角）：Gauss-Seidel加速为
    $$x_1^{k+1} = (1-\omega)x_1^k + \omega\,Q_{11}^{-1}(b_1 - Q_{12}x_2^k), \quad \omega \in (0, 2)$$
  - 过松弛Gibbs（采样视角，Adler 1981）：
    $$X_1^{k+1} \sim \mathcal{N}((1-\omega)X_1^k + \omega(\mu_1 - Q_{11}^{-1}Q_{12}(X_2^k - \mu_2)),\;\omega(2-\omega)Q_{11}^{-1})$$
  - 关键：均值移动 + **协方差缩放 $\omega(2-\omega)$**——保证正确的平稳分布
  - $\omega \to 2$：负自相关——样本在均值两侧交替跳跃，加速探索
  - 收敛速率（Fox & Parker, 2017）：匹配SOR的加速率

- **惯性Langevin算法（ILA）**
  - "Langevin meets Gibbs"（Falk, Habring & Pock, 2025）：ULA可以看作二块Gibbs算法
    - 噪声分裂：$X_{m+1} = X_m - \tau\nabla E(X_m) + \sqrt{\tau_1}N_1 + \sqrt{\tau_2}N_2$，$\tau_1 + \tau_2 = 2\tau$
    - 等价Gibbs：$Y_{m+1} \sim \mathcal{N}(X_m - \tau\nabla E(X_m), \tau_1 I)$，$X_{m+1} \sim \mathcal{N}(Y_{m+1}, \tau_2 I)$
  - 对ULA施加过松弛 → ILA：
    $$X_{m+1} = X_m - \gamma\,\nabla E(X_m) + \beta(X_m - X_{m-1}) + \sqrt{2\gamma(1-\beta)}\,N_m$$
    其中 $\gamma = \omega^2\tau$，$\beta = (1-\omega)^2$
  - ILA = 重球法 + 噪声：
    - 动量项 $\beta(X_m - X_{m-1})$：保持运动方向，加速穿越平坦区域
    - 噪声项 $\sqrt{2\gamma(1-\beta)}\,N_m$：保持探索能力——**噪声缩放至关重要**
  - $\beta \to 1$：显著加速，但偏差增大（可用MH校正步弥补）

- **欠阻尼Langevin SDE**
  - ILA的连续极限：欠阻尼Langevin方程
    $$\begin{cases} dV_t = (-\delta V_t - \nabla E(X_t))\,dt + \sqrt{2\delta}\,dW_t \\ dX_t = V_t\,dt \end{cases}$$
  - 平稳分布：$\pi(x, v) \propto \exp\left(-E(x) - \frac{\|v\|^2}{2}\right)$
    - 边缘化速度变量 $v$ 后，$x$ 的边缘分布恰为目标分布 $p(x) \propto \exp(-E(x))$
  - 收敛性定理：若 $E$ 是强凸且Lipschitz梯度，ILA的样本分布在 $W_2$ 距离下收敛到欠阻尼Langevin SDE的平稳分布
  - 与第7章的联系：欠阻尼Langevin SDE是扩散模型中二阶SDE的特例

- **📌 侧栏：优化加速 → 采样加速的完整对偶表**

  | 优化方法 | 采样方法 | 加速机制 |
  |---|---|---|
  | 梯度下降 | ULA | 加噪声 |
  | 近端梯度下降 | MYULA | 加噪声 |
  | 交替最小化 | Gibbs采样 | 加噪声 |
  | SOR/Gauss-Seidel加速 | 过松弛Gibbs | 协方差缩放 |
  | 动量法/重球法 | ILA | 动量+噪声缩放 |
  | Nesterov加速 | ? | 第7章：扩散模型 |

  统一规律：优化中的加速技巧 → 采样中的对应方法——第7章将展示扩散模型是"终极加速"

**来源**：Pock L2 P32-42; Adler (1981); Fox & Parker (2017); Falk, Habring & Pock (2025); Polyak (1964)

> **过渡**：我们已建立了MCMC采样的完整方法工具箱——从通用框架（MH）到高效特例（ULA/MYULA），从梯度驱动到结构驱动（Gibbs），从基本方法到加速方法。但MCMC的输出是一条马尔可夫链——如何判断它是否收敛？收敛后如何用采样结果进行不确定性量化？

---

## 4.7 收敛诊断与不确定性量化

**核心观点**：MCMC的输出是一条马尔可夫链，不是独立样本。收敛诊断判断链是否已到达平稳分布，自相关和有效样本量评估采样效率，不确定性量化将采样结果转化为决策依据——从"生成样本"到"提取信息"的最后一步。

- **Burn-in期**
  - 马尔可夫链需要一段时间才能"忘记"初始值，到达平稳分布
  - Burn-in：丢弃前 $B$ 个样本，只使用 $X_{B+1}, \ldots, X_M$
  - $B$ 的选择：观察链的轨迹图（trace plot），从趋势稳定处开始

- **自相关函数**
  - MCMC样本自相关：$\rho_k = \text{Corr}(h(X_m), h(X_{m+k}))$
  - 自相关衰减慢 → 样本间依赖强 → 有效信息量少
  - 理想情况：$\rho_k$ 快速衰减到0（如ILA的负自相关反而有益）
  - 自相关函数图：直观评估MCMC质量的核心工具

- **有效样本量（ESS）**
  - 定义：$\text{ESS} = \frac{M}{1 + 2\sum_{k=1}^K \rho_k}$
  - 含义：$M$ 个相关样本 ≈ ESS个独立样本的信息量
  - ESS/M 比率：衡量采样效率——越大越好
  - 实践准则：ESS > 100 通常足够估计后验均值；ESS > 1000 用于尾部概率

- **不确定性量化**
  - **后验可信区域**：$C_\alpha$ 满足 $P(x \in C_\alpha | y) = 1 - \alpha$
  - **HPD区域**（Highest Posterior Density）：最小的可信区域
    $$C_\alpha^{\text{HPD}} = \{x : p(x|y) \geq \gamma_\alpha\}$$
    其中 $\gamma_\alpha$ 使得 $P(x \in C_\alpha^{\text{HPD}}|y) = 1-\alpha$
  - **像素级置信区间**：对每个像素 $i$，用采样结果的分位数 $[q_{0.025}(x_i), q_{0.975}(x_i)]$ 构造95%置信区间
  - **假设检验**：检验图像中是否存在特定结构（如边缘、异常值）
    - 用HPD区域判断：若零假设值不在HPD区域内，则拒绝
    - **来源**：Pereyra L1 P980-1022

- **📌 侧栏：MCMC实践检查清单**
  1. 选择算法：光滑后验 → ULA；不可微 → MYULA；有条件结构 → Gibbs
  2. 选择步长：$\delta \leq 1/L$；过大则链发散，过小则移动慢
  3. Burn-in：从轨迹图判断，丢弃前期未收敛样本
  4. 收敛诊断：自相关函数快速衰减？ESS足够大？
  5. 多链对比：从不同初始值出发，检查是否收敛到同一区域
  6. 不确定性量化：用采样结果构造置信区间和可信区域

**来源**：lab1_ULA_sol (部分实践); Pereyra L1 P980-1022, P1192-1199

---

## 本章逻辑流总结

```
4.1 Monte Carlo积分：为什么需要采样
      │
      │ "如何从后验中采样？"
      ▼
4.2 Metropolis-Hastings：通用MCMC框架
      │
      │ "随机游走太慢，需要方向"
      ▼
4.3 ULA：利用梯度的高效采样
      │
      │ "梯度要求后验可微，不可微怎么办？"
      ▼
4.4 MYULA：Moreau包络光滑化不可微项
      │
      │ "除了梯度，还能利用什么结构？"
      ▼
4.5 Gibbs采样：利用条件分布结构（GLM框架）
      │
      │ "能否进一步加速？"
      ▼
4.6 加速方法：过松弛与惯性Langevin（ILA）
      │
      │ "链收敛了吗？采样结果怎么用？"
      ▼
4.7 收敛诊断与不确定性量化
      │
      ├──→ ULA的连续极限是什么？→ 第5章（Langevin SDE与得分函数）
      └──→ 采样路径的进一步发展 → 第6章（得分匹配）→ 第7章（扩散模型）
```

**核心叙事**：第3章告诉我们MAP只是后验众数，丢失了不确定性——本章走采样路径，从后验中提取完整信息。Monte Carlo积分是理论基础，MH是通用框架，ULA利用梯度信息实现高效采样（梯度下降+噪声=ULA），MYULA将ULA扩展到不可微情形（近端梯度下降+噪声=MYULA），Gibbs采样利用条件分布结构提供另一条路径（交替最小化+噪声=Gibbs），加速方法将优化中的加速技巧迁移到采样（动量法+噪声=ILA）。收敛诊断确保采样的可靠性，不确定性量化将采样结果转化为决策依据。本章的核心洞见是：**优化算法 + 噪声 = 采样算法**——噪声将"收敛到点"变为"收敛到分布"，而第5章将从ULA的离散迭代走向Langevin SDE的连续极限，揭示得分函数的核心角色。

---

## 材料覆盖状态

| 子主题 | 来源 | 状态 |
|---|---|---|
| Monte Carlo积分 | Pereyra L1 P30-35; LectureNotes2020_v2 Sec 8.2.1 | ✅ |
| 马尔可夫链与平稳分布 | LectureNotes2020_v2 Sec 8.2 | ✅ |
| MH算法与细致平衡 | Pereyra L1 P35-40; LectureNotes2020_v2 Sec 8.2 | ✅ |
| pCN-MCMC | LectureNotes2020_v2 Example 8.2.7; Cotter et al. (2013) | ✅ |
| ULA递推式与Euler离散化 | Pereyra L1 P40-50; Pock L2 P14-17; lab1_ULA_sol | ✅ |
| ULA步长δ≤1/L条件 | Pereyra L1 P45-48; lab1_ULA_sol | ✅ |
| MH vs ULA对比 | Pereyra L1 P48-50; Pock L2 P15-17 | ✅ |
| MALA侧栏 | Pereyra L1 P48-50 | ✅ |
| MYULA（Moreau包络） | Pereyra L1 P43-44; Pereyra L3 P9-11 | ✅ |
| MYULA收敛性（Durmus et al. 2018） | Pereyra L1 P46-47 | ✅ |
| Gibbs采样基本思想 | Pock L2 P28; Geman & Geman (1984) | ✅ |
| 半二次最小化→GLM | Pock L2 P25-26; Kuric et al. (2025) | ✅ |
| GLM与TV先验的Gibbs实现 | Pock L2 P29-33 | ✅ |
| Gibbs vs Gauss-Seidel对偶 | Pock L2 P36 | ✅ |
| 过松弛Gibbs | Pock L2 P37-38; Adler (1981); Fox & Parker (2017) | ✅ |
| ULA的Gibbs解读 | Pock L2 P39; Falk, Habring & Pock (2025) | ✅ |
| 惯性Langevin算法（ILA） | Pock L2 P40-41 | ✅ |
| 欠阻尼Langevin SDE | Pock L2 P42 | ✅ |
| MCMC收敛诊断（burn-in, ACF, ESS） | lab1_ULA_sol (部分) | 🟡 实践有，理论少 |
| 不确定性量化（可信区域、HPD） | Pereyra L1 P980-1022 | ✅ |
