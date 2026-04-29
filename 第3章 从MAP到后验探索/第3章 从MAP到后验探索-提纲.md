# 第3章 从MAP到后验探索 — 提纲

> **章节定位**：承接第2章，回答"如何从后验 p(x|y) 中提取解"。第2章末尾给出了两条路径：点估计（MAP）与分布采样（MCMC）。本章走第一条路——MAP估计的优化求解，按"先验类型→优化困难→求解算法"的主线，从光滑到不可微再到复合结构，逐步建立完整的优化工具箱。同时在本章结尾回到分叉点，揭示点估计的局限，为Part II的采样路径和Part III的变分路径打开接口。

> **叙事主线**：MAP=后验众数=优化问题(3.1) → 优化基础：凸性、光滑性与梯度下降(3.2) → 最简实例：Tikhonov闭式解与迭代解(3.3) → 不可微先验的挑战：近端算子与ISTA/FISTA(3.4) → TV正则化与原始-对偶算法(3.5) → 收敛性分析与正则化参数选择(3.6) → 从MAP到后验：点估计的局限与分叉(3.7)

---

## 3.0 本章导读：MAP——从后验众数到优化工具箱

上一章的结尾，后验分布 $p(x|y)$ 的形式已经完全确定——似然（第1章）乘以先验（第2章）。但确定形式不等于解决问题：高维空间中的后验分布无法直接计算，我们必须从中**提取**有用的信息。

最自然的提取方式是：找后验最"高"的那个点——后验众数，即MAP估计。取负对数后，MAP估计等价于一个优化问题：

$$\hat{x}_{\text{MAP}} = \arg\min_x \underbrace{-\ln p(y|x)}_{\text{数据项 } D(x)} + \underbrace{-\ln p(x)}_{\text{正则项 } R(x)}$$

第2章已经建立了"先验→正则项"的对应关系，现在进入求解阶段。不同的先验带来不同的正则项，不同的正则项带来不同的优化困难，不同的优化困难需要不同的求解算法——本章的主线就在这条因果链上展开：

**MAP=优化问题**（3.1）：MAP估计将"从后验中找最可能的x"转化为最小化后验能量。这是贝叶斯推断与正则化优化之间最直接的桥梁，也是本章的逻辑起点。

**优化需要什么条件**（3.2）：目标函数的凸性保证解的唯一性，光滑性决定梯度信息是否可用，强制性保证解的存在性。梯度下降是最基本的迭代框架，其收敛性由Lipschitz条件和步长选择保证。

**最简实例：光滑+凸**（3.3）：高斯先验+高斯似然是唯一的共轭情形——后验为高斯，MAP=MMSE，有闭式解。Tikhonov正则化是理解更复杂方法的锚点：闭式解给出理论直觉，迭代解提供数值框架。

**不可微怎么办**（3.4）：Laplace先验（L1正则化）的目标函数含不可微项，梯度下降失效。近端算子是处理不可微凸函数的基本工具——ISTA将梯度下降推广到"光滑+不可微"的复合结构，FISTA用Nesterov加速将收敛速率从 $O(1/k)$ 提升到 $O(1/k^2)$。

**TV的复合结构怎么办**（3.5）：TV正则化 $\|\nabla x\|_1$ 的梯度算子 $\nabla$ 使近端算子没有闭式解。通过Fenchel对偶转化为鞍点问题，Chambolle-Pock算法高效求解——这是处理"复合不可微"问题的标准工具。

**收敛有保障吗？参数怎么选**（3.6）：Bregman距离衡量收敛质量，源条件连接收敛速率与问题的正则性，经验贝叶斯将"人为调参"转化为"数据驱动"——收敛性分析是MAP估计可靠性的数学根基。

**MAP够了吗**（3.7）：MAP只给出后验众数，但众数≠典型。高维空间中众数可能远离均值，多峰后验中众数只代表一个峰。不确定性量化、多解识别、条件生成——这些任务都需要后验分布的完整信息，而非单点估计。两条路径从分叉点出发：采样探索后验（第4章），变分近似后验（第8章）。

```
MAP=优化(3.1) → 优化基础(3.2) → Tikhonov光滑实例(3.3) → 近端方法/不可微(3.4) → 原始-对偶/TV(3.5) → 收敛与参数(3.6) → 分叉：MAP→后验(3.7)
```

本章结束时，读者将拥有MAP求解的完整工具箱，也将理解它的局限——点估计只是后验的一个点，后验的完整信息需要采样或近似。接下来的两条路径各有侧重：第4章走采样路线（MCMC→ULA），第8章走变分路线（ELBO→VAE）——两条路终将在扩散模型处汇合。

---

## 3.1 MAP估计：从后验众数到优化问题

**核心观点**：MAP估计将"从后验中找最可能的x"转化为一个优化问题——最小化后验能量。这是贝叶斯框架与正则化方法之间最直接的桥梁，也是全书"优化视角"的起点。

- **MAP的定义与概率意义**
  - $\hat{x}_{\text{MAP}} = \arg\max_x p(x|y) = \arg\min_x [-\ln p(y|x) - \ln p(x)]$
  - 在Bregman散度损失下的最优性（Pereyra P28）：MAP估计器最小化期望Bregman散度 $\mathbb{E}[D_\phi(\hat{x}, x)]$，而MMSE估计器最小化对称Bregman散度——两者是同一后验下不同的"问法"
  - 回顾第2章结论：MAP看众数，MMSE看均值；高斯后验时两者一致，非高斯时产生分歧

- **从概率到优化：后验能量最小化**
  - $-\ln p(x|y) = \underbrace{-\ln p(y|x)}_{\text{数据项 } D(x)} + \underbrace{-\ln p(x)}_{\text{正则项 } R(x)} + \text{const}$
  - MAP = $\arg\min_x D(x) + \lambda R(x)$——第2章建立的贝叶斯→变分对应，现在进入求解阶段
  - 不同先验→不同正则项→不同优化困难→不同算法，形成本章后续各节：

  | 先验 | 正则项 | 优化困难 | 求解算法 |
  |---|---|---|---|
  | 高斯 | $\|x\|^2$ | 光滑凸 | 梯度下降 / 闭式解 |
  | Laplace | $\|x\|_1$ | 不可微 | 近端方法（ISTA/FISTA） |
  | TV | $\|\nabla x\|_1$ | 复合不可微 | 原始-对偶（Chambolle-Pock） |

- **MAP的适用范围与局限**
  - 优势：只需优化，不需要积分——计算友好
  - 局限：只给出众数，丢失后验的形状信息（多峰？偏斜？方差？）
  - 这一局限将在3.7节展开，引出采样路径

**来源**：Pereyra L1 P12, P24-28; invprobs_v2 Ch12; MIVAcourse_opt1 P11

MAP估计在数学上等价于一个优化问题。而优化问题的"好不好解"，取决于目标函数的性质——这就需要先建立优化基础。

---

## 3.2 优化基础：凸性、光滑性与梯度下降

**核心观点**：MAP优化问题的"好不好解"，取决于目标函数的凸性（是否有唯一解）和光滑性（梯度是否可用）。梯度下降是最基本的迭代求解框架，其收敛性由Lipschitz条件和步长选择保证。

- **凸性：解的唯一性保障**
  - 凸函数定义与一阶判定条件：$f(y) \geq f(x) + \nabla f(x)^T(y-x)$
  - 强凸性：$f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2$ → 唯一极小点 → 线性收敛
  - MAP目标函数的凸性：数据项（$\|Ax-y\|^2$，凸）+ 正则项（取决于先验）→ 凸性组合

- **光滑性：梯度信息的可用性**
  - L-光滑（Lipschitz梯度）的定义：$\|\nabla f(x) - \nabla f(y)\| \leq L\|x-y\|$
  - 对L²数据项 $\frac{1}{2}\|Ax-y\|^2$：$L = \lambda_{\max}(A^TA)$
  - 光滑→梯度下降可用；非光滑→需要近端方法（3.4节）

- **梯度下降算法**
  - 迭代格式：$x_{k+1} = x_k - \tau \nabla f(x_k)$
  - 步长选择：$\tau \leq 1/L$（保证函数值单调下降）
  - 收敛速率：$O(1/k)$（凸，$f(x_k) - f^* \leq \frac{L\|x_0-x^*\|^2}{2k}$）；线性收敛率（强凸）

- **强制性：解的存在性保障**
  - 强制函数定义：$\lim_{\|x\|\to\infty} f(x) = +\infty$
  - MAP目标函数的强制性：数据项 + 正则项共同保证（正则项起关键作用——若无正则项，数据项可能不强制）

- **📌 梯度下降的物理直觉**（已融入3.2节正文，以blockquote标注形式呈现）
  - 梯度下降 = 沿最陡下降方向走一步
  - 步长太大 → 越过极小点 → 震荡甚至发散
  - 步长太小 → 收敛极慢
  - 最优步长 $\tau = 1/L$ 的几何含义

**来源**：Calatroni P49-51; MIVAcourse_opt1 P19-55; invprobs_v2 Ch11

有了优化基础，我们从最简单的MAP实例开始——高斯先验下的Tikhonov正则化，它有闭式解。

---

## 3.3 Tikhonov正则化：闭式解与迭代解

**核心观点**：高斯先验+高斯似然是唯一的共轭情形（第2章已预告），后验为高斯→MAP=MMSE→有闭式解。Tikhonov正则化是最基本的正则化方法，其闭式解和迭代解分别为理解更复杂方法提供了锚点。

- **MAP→Tikhonov的推导回顾**
  - 高斯似然 $p(y|x) = \mathcal{N}(Ax, \sigma^2 I)$ + 高斯先验 $p(x) = \mathcal{N}(0, \sigma_x^2 I)$
  - MAP目标：$\min_x \frac{1}{2\sigma^2}\|Ax-y\|^2 + \frac{1}{2\sigma_x^2}\|x\|^2$
  - 即 Tikhonov：$\min_x \frac{1}{2}\|Ax-y\|^2 + \frac{\lambda}{2}\|x\|^2$，$\lambda = \sigma^2/\sigma_x^2$

- **闭式解：SVD域直接求解**
  - 正规方程：$(A^TA + \lambda I)\hat{x} = A^Ty$
  - SVD表示：$\hat{x}_\lambda = \sum_i \frac{\sigma_i}{\sigma_i^2 + \lambda}\langle y, u_i\rangle v_i$
  - 滤波解释：Tikhonov = 低通滤波器，$\lambda$控制截止频率
  - $\lambda \to 0$：趋向最小二乘（过拟合）；$\lambda \to \infty$：趋向零（过正则化）
  - DFT域加速：当 $A$ 为卷积算子时，利用循环矩阵的DFT对角化，闭式解在频域逐元素计算

- **迭代解：梯度下降**
  - 梯度：$\nabla f(x) = A^T(Ax-y) + \lambda x$
  - 迭代：$x_{k+1} = x_k - \tau(A^T(Ax_k-y) + \lambda x_k)$
  - Landweber迭代（$\lambda=0$的特例）及其半收敛性：先逼近真解，后被噪声主导——正则化的迭代视角

- **偏差-方差分解**
  - 总误差 = 偏差（正则化引入的偏差）+ 方差（噪声放大的方差）
  - $\lambda$小→偏差小但方差大；$\lambda$大→方差小但偏差大
  - 最优 $\lambda$ 在偏差与方差之间取得平衡

- **正则化参数 $\lambda$ 的概率含义**
  - $\lambda = \sigma^2/\sigma_x^2$ = 噪声方差/先验方差 = 信噪比的倒数
  - 噪声大 → $\sigma^2$大 → $\lambda$大 → 更强正则化（贝叶斯解释：噪声大时更依赖先验）
  - 正则化参数不是任意常数——它有明确的概率含义

**来源**：Calatroni P53; Benning L1 P75-83; invprobs_v2 Ch6; invprobs_v2 Ch11; XR05

Tikhonov的闭式解依赖于高斯先验的可微性。一旦换用Laplace先验（L1正则化），目标函数不再处处可微——梯度下降失效，近端方法出场。

---

## 3.4 近端方法：不可微先验的求解策略

**核心观点**：Laplace先验和稀疏正验化的目标函数含不可微项，梯度下降无法直接使用。近端算子是处理不可微凸函数的基本工具，ISTA/FISTA将梯度下降推广到"光滑+不可微"的复合结构。

- **不可微的挑战：为什么梯度下降不够用**
  - L1范数 $\|x\|_1$ 在 $x_i=0$ 处不可微——无法计算梯度
  - TV范数 $\|\nabla x\|_1$ 同样不可微
  - 次梯度：推广梯度概念——$g \in \partial f(x)$ 满足 $f(y) \geq f(x) + g^T(y-x)$
  - 次梯度≠梯度：次梯度不唯一，不能直接用于梯度下降

- **近端算子：不可微函数的"梯度下降替代"**
  - 定义：$\text{prox}_{\lambda g}(v) = \arg\min_x \left\{\frac{1}{2}\|x-v\|^2 + \lambda g(x)\right\}$
  - 直觉：在梯度下降一步之后，再做一个"近端校正"——靠近极小点，但不过分远离当前位置
  - 关键性质：唯一性（强凸保证）、非扩张性、与次梯度的关系（$\text{prox}_{\lambda g} = (I + \lambda\partial g)^{-1}$）

- **经典近端算子**

  | 函数 $g(x)$ | 近端算子 $\text{prox}_{\lambda g}(v)$ | 名称 |
  |---|---|---|
  | $\frac{1}{2}\|x\|^2$ | $\frac{v}{1+\lambda}$ | 收缩 |
  | $\|x\|_1$ | $\text{sign}(v)\max(|v|-\lambda, 0)$ | 软阈值 |
  | $\|x\|_0$ | $v \cdot \mathbf{1}_{\|v\|\geq\sqrt{2\lambda}}$ | 硬阈值 |
  | $\iota_C(x)$ | $\text{proj}_C(v)$ | 投影 |

  - 软阈值的几何理解：将小于 $\lambda$ 的分量归零，大于 $\lambda$ 的分量向零收缩 $\lambda$——L1近端 = 稀疏化

- **ISTA：近端梯度下降**
  - 问题形式：$\min_x f(x) + g(x)$，$f$光滑凸、$g$下半连续凸且易近端
  - 迭代：$x_{k+1} = \text{prox}_{\tau g}(x_k - \tau \nabla f(x_k))$
  - 两步解读：先沿光滑部分走一步梯度下降，再做一次近端校正
  - 收敛速率：$O(1/k)$（凸）；线性收敛（强凸）

- **FISTA：Nesterov加速**
  - 动量项：$y_{k+1} = x_{k+1} + \frac{t_k-1}{t_{k+1}}(x_{k+1} - x_k)$，其中 $t_{k+1} = \frac{1+\sqrt{1+4t_k^2}}{2}$
  - 收敛速率：$O(1/k^2)$——一阶方法的最优速率（Nesterov下界）
  - ISTA vs FISTA收敛曲线对比：FISTA在迭代初期即显著领先
  - V-FISTA（强凸变体）：常数动量，线性收敛率

- **迭代硬阈值（IHT）与重加权L1**
  - IHT：L0问题的近端算子（硬阈值），贪婪稀疏选择——保留最大的 $k$ 个分量
  - 重加权L1：用迭代重加权策略逼近L0——每次求解一个加权L1问题，权重由上次解决定
  - L1的偏差问题：L1正则化倾向于压缩所有非零系数（shrinking bias），重加权L1可缓解

**来源**：MIVAcourse_opt2 P6-52; numerical_optimisation P426-520; PHD_MIVA lab; MIVAcourse_opt3 P20-28

近端方法解决了L1稀疏正则化的计算，但TV正则化带来了新的困难：$\text{TV}(x) = \|\nabla x\|_1$ 中梯度算子 $\nabla$ 使近端算子不再有闭式解。原始-对偶算法绕过了这一障碍。

---

## 3.5 TV正则化与原始-对偶算法

**核心观点**：TV正则化是最常用的图像先验，但其复合结构（$\|\nabla x\|_1$）使近端算子无法直接计算。通过Fenchel对偶，将原始问题转化为鞍点问题，Chambolle-Pock算法高效求解。

- **TV正则化的MAP形式**
  - ROF模型（去噪）：$\min_x \frac{1}{2}\|x-y\|^2 + \lambda\text{TV}(x)$
  - 去模糊/CT重建：$\min_x \frac{1}{2}\|Ax-y\|^2 + \lambda\text{TV}(x)$
  - 为什么近端方法失效：$\text{prox}_{\lambda\|\nabla\cdot\|_1}(v) = \arg\min_x \frac{1}{2}\|x-v\|^2 + \lambda\|\nabla x\|_1$ 无闭式解（$\nabla$不可逆，不像正交变换那样可以分离计算）

- **Fenchel对偶与鞍点问题**
  - 共轭函数定义：$g^*(y) = \sup_x \{\langle y, x\rangle - g(x)\}$
  - Fenchel-Rockafellar对偶：$\min_x f(x) + g(Kx) \Longleftrightarrow \min_x \max_y \langle Kx, y\rangle + f(x) - g^*(y)$
  - TV的对偶形式：$g^* = $ 指示函数（对偶变量有界的指示）→ 对偶变量有约束
  - 直觉：原始问题在"图像空间"求解，对偶问题在"梯度空间"求解，两者互为补充

- **Chambolle-Pock算法（原始-对偶混合梯度）**
  - 鞍点形式：$\min_x \max_y \langle Kx, y\rangle + f(x) - g^*(y)$
  - 迭代格式：
    - $x_{k+1} = \text{prox}_{\tau f}(x_k - \tau K^T y_k)$
    - $\bar{x}_{k+1} = 2x_{k+1} - x_k$（外推）
    - $y_{k+1} = \text{prox}_{\sigma g^*}(y_k + \sigma K\bar{x}_{k+1})$
  - 步长条件：$\tau\sigma\|K\|^2 < 1$
  - 应用于TV去模糊与CT重建的实例

- **ADMM：交替方向乘子法**（简要介绍）
  - 增广拉格朗日：$\mathcal{L}_\rho(x, z, u) = f(x) + g(z) + \langle u, Kx-z\rangle + \frac{\rho}{2}\|Kx-z\|^2$
  - 交替更新：$x$步 → $z$步 → 乘子 $u$步
  - ADMM vs Chambolle-Pock：ADMM更适合有自然分裂的问题（如TV+L1复合），Chambolle-Pock更适合标准鞍点结构

- **PnP——近端算子与去噪器的桥梁**（已升格为3.5节内独立小节）
  - 近端算子的另一种解读：$\text{prox}_{\lambda R}(v)$ 是以 $v$ 为观测、$R$ 为正则项的去噪器
  - PnP（Plug-and-Play）思想：用学习到的去噪器替换近端算子
  - ADMM-PnP：将ADMM中的 $z$步替换为外部去噪器
  - 这为第5章的Tweedie等式和PnP-ULA埋下伏笔——近端算子→去噪器→得分函数的递进

**来源**：Benning L1 P2442-2562; numerical_optimisation P522-655; MIVAcourse_opt2 P730-745; MIVAcourse_opt3 P289-307; XR09_TV; Gondzio L6

至此我们已建立了MAP求解的完整工具箱——梯度下降（光滑）、近端方法（L1稀疏）、原始-对偶（TV）。但算法收敛到什么？收敛有多快？参数如何选择？这些问题的回答需要收敛性分析。

---

## 3.6 收敛性分析与正则化参数选择

**核心观点**：优化算法的收敛性保障是MAP估计可靠性的数学根基。Bregman距离是衡量收敛质量的核心工具，源条件连接了收敛速率与问题的正则性。经验贝叶斯将"人为调参"转化为"数据驱动"——正则化参数的选择不再是黑箱。

- **Bregman距离：收敛质量的度量**
  - 定义：$D_J^p(u_1, u_2) = J(u_1) - J(u_2) - \langle p, u_1-u_2\rangle$，$p \in \partial J(u_2)$
  - 关键性质：非负（$D_J^p \geq 0$，$J$凸时）；非对称
  - 特例：$J(x) = \frac{1}{2}\|x\|^2$ → Bregman距离退化为欧氏距离；$J(x) = x\ln x - x$ → Bregman距离 = KL散度
  - 对称Bregman距离：$D_J^{\text{symm}}(u,v) = \langle q-p, v-u\rangle$，$p \in \partial J(u), q \in \partial J(v)$

- **源条件与收敛速率**
  - 源条件：存在 $v^\dagger \in V$ 使得 $A^Tv^\dagger \in \partial J(u^\dagger)$
  - 物理含义：源条件刻画了真解 $u^\dagger$ 与正则项 $J$ 的"兼容程度"——$J$的假设越贴近真解的结构，源条件越容易满足
  - 带源条件的收敛速率：$D_J^{\text{symm}}(u_\alpha, u^\dagger) \leq \|v^\dagger\| \cdot \delta$，其中 $\alpha(\delta) \propto \delta$
  - 无源条件：收敛速率显著下降——源条件不是技术假设，而是问题可解性的度量

- **迭代正则化的半收敛性**
  - Landweber迭代的典型行为：前期逼近真解，后期被噪声主导
  - 早停 = 隐式正则化：迭代次数 $k$ 扮演正则化参数的角色
  - Morozov偏差原理：$\|Ax_k - y^\delta\| \approx \delta$ 时停止

- **经验贝叶斯：从数据估计正则化参数**
  - 边际似然：$p(y|\lambda) = \int p(y|x)p(x|\lambda)\,dx$——$\lambda$的似然函数
  - 经验贝叶斯策略：$\hat{\lambda} = \arg\max_\lambda p(y|\lambda)$
  - Fisher恒等式与随机近似：用梯度信息迭代更新 $\lambda$
  - 从"手动调参"到"自动估计"——经验贝叶斯让正则化参数的选择有了概率论根基
  - 正则化参数的两种视角：频率学派（交叉验证/L曲线） vs 贝叶斯学派（边际似然最大化）

**来源**：Benning L1 P3576-3732; Benning L2 P1326-1446; invprobs_v2 Ch11; Pereyra L1 P58-84; Unit2_exercise

至此，MAP估计的完整求解框架已经建立——从优化基础到各类算法，从收敛保障到参数选择。然而MAP只是后验的一个点——众数。后验的完整信息（不确定性、多峰性、方差）被优化过程完全丢弃了。

---

## 3.7 从MAP到后验：点估计的局限与分叉

**核心观点**：MAP给出后验众数，但众数≠典型。高维空间中，众数可能远离均值；多峰后验中，众数只代表一个峰。不确定性量化、多解识别、条件生成——这些任务都需要后验分布的完整信息，而非单点估计。

- **众数≠典型：高维空间的反直觉**
  - 高维高斯的众数 vs 均值 vs 典型集（invprobs_v2 Example 12.2-12.3）
  - "MAP估计可能最可能，但不一定典型"——维度诅咒
  - 具体算例：$n$维标准高斯 $\mathcal{N}(0, I_n)$ 的众数在原点，但典型样本 $\|x\| \approx \sqrt{n}$ 远离原点

- **MAP丢失了什么**
  - **不确定性**：后验方差 → 置信区间 → 重建的可靠性——MAP无法提供
  - **多峰性**：多解逆问题中（如有限角CT），MAP只给一个解——其他物理上合理的解被忽略
  - **后验的形状信息**：偏斜、重尾等统计特征在优化过程中被完全丢弃
  - 误差分解（回顾第2章）：MAP的误差 = 不可约误差 + 优化误差，但缺少采样误差的估计

- **从"求点"到"求分布"的视角跃迁**
  - MAP问："最可能的x是什么？" → 优化
  - 贝叶斯问："给定y，x的一切可能性是什么？" → 采样/近似
  - 这不是对MAP的否定，而是从"点估计"到"分布估计"的自然升级

- **分叉点：两条路径**
  - **路径一（采样·主线）**：用MCMC/ULA从后验中采样 → 求期望、估计方差 → Part II（第4-7章）
    - 这是逆问题需求自然驱动的路径：MAP丢弃不确定性→需要后验采样→MCMC→Langevin→得分→扩散
  - **路径二（近似·副线）**：用变分分布 $q_\phi(x)$ 近似后验 → ELBO → Part III（第8-11章）
    - 这是由生成建模社区独立发展的路径，提供从优化角度理解扩散的互补视角
  - 两条路径从不同角度回答同一个问题：**如何处理无法直接计算的高维后验分布？**

- **两条路径的终点**
  - 采样路径（主线）：MCMC → ULA → Langevin → Score Matching → Diffusion(SDE)——逆问题需求驱动的自然演化
  - 变分路径（副线）：ELBO → VAE → 层级VAE → Diffusion(VLB)——生成建模的独立推导
  - 终点都是扩散模型——第12章殊途同归
  - 第13章完成闭环：条件扩散采样 = 逆问题求解

- **PnP——第三条路？**（已升格为3.7节内独立小节）
  - 在MAP框架内，用学习到的去噪器替换近端算子 → PnP-ADMM / PnP-FISTA
  - PnP绕过了显式先验的局限，但仍在优化框架内——它是"增强的MAP"，不是后验采样
  - 真正的后验探索需要第4章的采样方法

**来源**：Pereyra L1 P30+; Pock L2 P18-24; invprobs_v2 Ch12; 成书/绪论 0.4

---

## 附录 3A 截断SVD正则化

> 定位：与Tikhonov并列的经典谱正则化方法，供对比参考。不参与主线叙事。

- 截断SVD：$\hat{x}_r = \sum_{i=1}^r \frac{1}{\sigma_i}\langle y, u_i\rangle v_i$
- 与Tikhonov的对比：硬截断 vs 软衰减
  - Tikhonov滤波：$\frac{\sigma_i}{\sigma_i^2 + \lambda}$（平滑衰减）
  - 截断SVD滤波：$\frac{1}{\sigma_i}\mathbf{1}_{i \leq r}$（硬截断）
- 截断点 $r$ 的选择与正则化参数 $\lambda$ 的等价性
- 谱正则化的统一视角：不同正则化方法 = 不同的滤波函数族

**来源**：Benning L1 P40-70; invprobs_v2 Ch5-6; BunnyTomo3

---

## 附录 3B Moreau包络与近端算子的性质

> 定位：3.4节近端算子的数学性质补充，供对优化理论有深入兴趣的读者参考。

- Moreau包络：$g^\lambda(x) = \inf_y \{g(y) + \frac{1}{2\lambda}\|y-x\|^2\}$——$g$的光滑化
- Moreau包络的性质：$\lambda$-光滑，与$g$有相同的极小点，梯度 $\nabla g^\lambda(x) = \frac{1}{\lambda}(x - \text{prox}_{\lambda g}(x))$
- Moreau恒等式：$\text{prox}_{\lambda g}(x) + \lambda\,\text{prox}_{g^*/\lambda}(x/\lambda) = x$
- 与第4章ULA的联系：Moreau包络用于构造MYULA（Moreau-Yoshida正则化的ULA）

**来源**：MIVAcourse_opt2 P534-553; PEREYRA Lectures 1&2 P830-901

---

## 本章逻辑流总结

```
3.1 MAP=后验众数=优化问题
      │
      │ "优化需要什么条件？"
      ▼
3.2 优化基础：凸性、光滑性、梯度下降
      │
      │ "最简实例：光滑+凸"
      ▼
3.3 Tikhonov：闭式解与迭代解（高斯先验）
      │
      │ "不可微先验怎么办？"
      ▼
3.4 近端方法：ISTA/FISTA（Laplace/L1先验）
      │
      │ "TV的复合结构怎么办？"
      ▼
3.5 原始-对偶：Chambolle-Pock（TV先验）
      │
      │ "收敛有保障吗？参数怎么选？"
      ▼
3.6 收敛性分析 + 经验贝叶斯
      │
      │ "MAP够了吗？"
      ▼
3.7 从MAP到后验：点估计的局限与分叉点
      │
      ├──→ 如何从后验采样？→ 第4章（MCMC/ULA）→ Part II 采样路径
      └──→ 如何近似后验？→ 第8章（变分推断/ELBO）→ Part III 变分路径
```

**核心叙事**：MAP将贝叶斯推断转化为优化问题，优化工具箱按"光滑→不可微→复合"的困难递进：梯度下降→近端方法→原始-对偶。Tikhonov给出闭式锚点，ISTA/FISTA处理稀疏性，Chambolle-Pock解决TV的复合结构。Bregman距离和源条件为收敛提供保障，经验贝叶斯让参数选择告别手动调参。但MAP只是后验的一个点——众数≠典型，点估计丢失不确定性。两条路径从分叉点出发：采样探索后验（Part II），变分近似后验（Part III），殊途同归于扩散模型。

---

## 材料覆盖状态

| 子主题 | 来源 | 状态 |
|---|---|---|
| MAP估计与贝叶斯决策理论 | Pereyra L1 P12, P24-25; invprobs_v2 Ch12 | ✅ |
| 凸性/Lipschitz/强制性 | Calatroni P49-51; MIVAcourse_opt1 P19-40 | ✅ |
| 梯度下降与步长选择 | MIVAcourse_opt1 P41-55; Calatroni P51-53 | ✅ |
| Tikhonov闭式解(SVD/DFT) | Calatroni P53; Benning L1 P75-83; invprobs_v2 Ch6; XR05 | ✅ |
| Landweber迭代与半收敛性 | invprobs_v2 Ch11; variational_formulations P406-415 | ✅ |
| 偏差-方差分解 | invprobs_v2 Ch6; discrete_ip_regularization P387 | ✅ |
| 次梯度 | MIVAcourse_opt2 P80-135 | ✅ |
| 近端算子定义与计算 | MIVAcourse_opt2 P409-723; numerical_optimisation P455-520 | ✅ |
| ISTA/FISTA | MIVAcourse_opt2 P820-1270; PHD_MIVA lab | ✅ |
| 迭代硬阈值(IHT) | MIVAcourse_opt3 P20-22 | ✅ |
| 重加权L1 | MIVAcourse_opt3 P28+ | ✅ |
| Fenchel对偶 | numerical_optimisation P522-600; MIVAcourse_opt2 P730-745 | ✅ |
| TV正则化与Chambolle-Pock | Benning L1 P2442-2562; XR09_TV; tomo_tv | ✅ |
| ADMM算法 | numerical_optimisation P631-655; Gondzio L6 | 🟡 缺完整推导 |
| PnP框架 | PEREYRA Lecture 3 P624+ | ✅ |
| Bregman距离与误差估计 | Benning L1 P3576-3732; Benning L2 P1326-1446 | ✅ |
| 源条件与收敛速率 | Benning L2 P10-20, P33-38 | ✅ |
| 迭代正则化与偏差原理 | invprobs_v2 Ch11 | ✅ |
| 截断SVD正则化 | Benning L1 P40-70; BunnyTomo3 | ✅ |
| 经验贝叶斯参数估计 | Pereyra L1 P58-84; Unit2_exercise | ✅ |
| Moreau包络 | MIVAcourse_opt2 P534-553; PEREYRA L1&2 P830-901 | ✅ |
| 从MAP到后验的动机 | Pereyra L1 P30+; Pock L2 P18-24 | ✅ |
