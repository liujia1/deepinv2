# 第14章 Flow Matching与最优传输 — 提纲

## 本章定位

第14章是全书叙事从"经典框架"向"前沿发展"的跃迁——前13章完成了从逆问题到扩散模型的完整理论构建（贝叶斯→采样→得分→扩散→统一→条件生成），本章引入Flow Matching这一全新生成框架，打破了扩散模型对随机微分方程和得分函数的依赖，开辟了"学向量场"的新路径。核心论点是：**Flow Matching是扩散模型的范式升级——从学习得分函数到学习向量场，从SDE采样到ODE传输，从曲线路径到直线路径**。最优传输（OT）为向量场的学习提供了最优的耦合策略，使得Flow Matching不仅理论上更简洁，实践上也更高效（更少步数即可生成高质量样本）。

**核心论点**：Flow Matching通过回归向量场而非得分函数来训练生成模型，条件Flow Matching定理保证了无需模拟ODE即可训练，最优传输耦合提供了更直的传输路径。Rectified Flow将直线插值与Reflow过程结合，实现了从多步到少步甚至单步生成的蒸馏。SD3/Flux证明了Flow Matching + DiT的组合在工业级生成任务中的SOTA性能。

**与前章衔接**：第7章建立了扩散模型的SDE框架，概率流ODE（PF-ODE）是Flow Matching的直接前身——Flow Matching可以看作"跳过SDE，直接学ODE"的思路。第12章证明了Score ≡ ELBO，统一了得分路径和变分路径；本章引入第三条路径——向量场回归路径，三条路径在ODE层面汇合。第13章的条件扩散采样框架在Flow Matching中同样成立，只需将得分函数替换为向量场。

**与后章衔接**：14.5节的SD3/Flux = Rectified Flow + DiT连接到第15章的架构实践（UNet → DiT），为架构演进提供了算法动机。

---

## 叙事弧

```
最优传输基础（为何OT？）→ 连续归一化流（CNF的困境）→
Flow Matching（突破：学向量场）→ Rectified Flow（直线路径+Reflow）→
前沿组合（SD3/Flux）
```

理解动机→建立基础→核心突破→工程优化→实践落地

---

## 章节结构

### 14.0 本章导读：从得分到向量场——范式跃迁

**核心观点**：前13章的生成模型都以得分函数 $\nabla\log p_t(x)$ 为核心——无论是Langevin采样、逆向SDE还是概率流ODE，得分函数都是"引擎"。本章提出一个根本性的问题：**我们能否绕过得分函数，直接学习驱动力本身——向量场？** 这个思路的转变，从"学得分"到"学向量场"，开启了一个全新的生成框架：Flow Matching。

- 全书生成路径回顾：得分路径（第5-7章）、变分路径（第8-12章），两者在Score ≡ ELBO统一
- 第三条路径的直觉：与其先定义加噪过程再学逆向，不如直接学一个"传输"过程——从噪声到数据的ODE
- Flow Matching的核心优势：无需模拟ODE即可训练（仿真自由）、路径更直（更少步数）、框架更统一（OT自然融入）
- 章节导航：OT基础→CNF困境→Flow Matching→Rectified Flow→前沿组合

**来源**：全书叙事；book_plan.md 核心论点

---

### 14.1 最优传输基础

**核心观点**：最优传输（Optimal Transport, OT）研究如何以最小代价将一个概率分布"搬运"到另一个概率分布。Monge问题寻找一个确定性传输映射，Kantorovich松弛将其推广为概率耦合的优化问题。Wasserstein距离是OT代价的最优值，它赋予概率分布空间以几何结构。OT为生成模型提供了最优的"传输蓝图"——沿着OT路径传输，路径最短、效率最高。

#### 14.1.1 Monge问题：寻找最优传输映射

- **Monge问题的形式化**
  - 给定源分布 $p_0$ 和目标分布 $p_1$，寻找传输映射 $T: \mathbb{R}^d \to \mathbb{R}^d$
  - 使得 $T_\# p_0 = p_1$（推前条件：$p_1 = p_0 \circ T^{-1}$）
  - 最小化传输代价：$\min_T \int c(x, T(x))\,dp_0(x)$
  - 最常见的代价函数：$c(x, y) = \|x - y\|^2$（平方欧氏距离）
- **Monge问题的困难**
  - 约束 $T_\# p_0 = p_1$ 是无限维的非线性约束
  - 传输映射 $T$ 可能不存在（如 $p_0$ 是点质量、$p_1$ 不是）
  - 即使存在，优化问题高度非凸
- **物理直觉**：Monge问题像是"最聪明的搬家公司"——把一堆沙子从形状A搬到形状B，每次移动一整铲，寻找最省力的搬运方案

#### 14.1.2 Kantorovich松弛：从确定性映射到概率耦合

- **Kantorovich问题的形式化**
  - 放松Monge的确定性约束，允许"分拆搬运"——一个源点的沙子可以分散到多个目标点
  - 耦合（联合分布）$\gamma \in \Pi(p_0, p_1)$：边际分布分别为 $p_0$ 和 $p_1$ 的联合分布
  - 优化目标：
    $$\min_{\gamma \in \Pi(p_0, p_1)} \int \|x_0 - x_1\|^2\,d\gamma(x_0, x_1)$$
  - $\Pi(p_0, p_1)$：所有以 $p_0, p_1$ 为边际的联合分布的集合
- **Kantorovich松弛的优势**
  - 始终有解（$\Pi(p_0, p_1)$ 非空且紧）
  - 线性规划问题（凸优化）
  - Monge映射是Kantorovich最优耦合的特例（当耦合退化为确定性映射时）
- **独立耦合 vs 最优耦合**
  - 独立耦合：$\gamma_{\text{ind}} = p_0 \otimes p_1$，源和目标独立采样
  - 最优耦合：$\gamma^*$ 最小化传输代价，源和目标之间存在相关性
  - **对生成模型的意义**：独立耦合→弯曲路径，最优耦合→直线路径（14.3节将详细展开）
- **离散情形**：Kantorovich问题退化为线性规划
  - 给定 $n$ 个源点和 $m$ 个目标点，代价矩阵 $C_{ij} = \|x_i - y_j\|^2$
  - 耦合矩阵 $\Gamma$：$\Gamma_{ij} \geq 0$，行和 = $p_0(x_i)$，列和 = $p_1(y_j)$
  - $\min_{\Gamma} \sum_{ij} C_{ij}\Gamma_{ij}$

#### 14.1.3 Kantorovich对偶

- **对偶问题**
  - 原问题：$\min_{\gamma \in \Pi} \int c\,d\gamma$
  - 对偶问题：$\max_{\phi, \psi} \int \phi\,dp_0 + \int \psi\,dp_1$，约束 $\phi(x) + \psi(y) \leq c(x, y)$
  - 对偶变量 $\phi, \psi$ 称为Kantorovich势（Kantorovich potentials）
  - 强对偶性：最优原值 = 最优对偶值
- **对偶的直觉**：对偶问题是"定价问题"——给每个源点和目标点定价，使得"买入+卖出"的总利润最大，但任何一对点的价差不超过运输成本
- **c-变换与1-Wasserstein对偶**：$W_1$ 距离的对偶形式是1-Lipschitz函数空间上的上确界（Kantorovich-Rubinstein对偶）

#### 14.1.4 Wasserstein距离

- **定义**
  - $p$-Wasserstein距离：
    $$W_p(p_0, p_1) = \left(\min_{\gamma \in \Pi(p_0, p_1)} \int \|x_0 - x_1\|^p\,d\gamma(x_0, x_1)\right)^{1/p}$$
  - 最常用：$W_1$（Earth Mover's Distance）和 $W_2$（平方Wasserstein距离）
- **Wasserstein距离 vs 其他分布距离**
  | 距离 | 定义 | 拓扑 | 度量性质 | 对生成模型的意义 |
  |---|---|---|---|---|
  | KL散度 | $\int p\log(p/q)$ | 强（绝对连续要求） | 非对称，非度量 | VAE/ELBO的训练目标 |
  | $W_1$ | 最优传输代价 | 弱（支持分布间距离） | 对称，度量 | WGAN的训练目标 |
  | $W_2$ | 最优传输代价的平方根 | 弱 | 对称，度量 | OT-FM的路径优化 |
  | MMD | 核空间距离 | 弱 | 对称，半度量 | GAN评估 |
- **$W_2$ 的关键性质**
  - 度量性质：非负、对称、三角不等式
  - 弱拓扑：$W_2$ 收敛弱于KL散度收敛，但更实用（能度量不重叠分布间的距离）
  - **为什么 $W_2$ 对生成模型重要**：$W_2$ 赋予概率分布空间以黎曼几何结构（Wasserstein空间），传输映射是Wasserstein空间中的"测地线"
- **Wasserstein空间与测地线**
  - 测地线（McCann插值）：给定最优耦合 $\gamma^*$，中间分布为
    $$p_t = ((1-t)\text{id} + tT)_\# p_0$$
  - 这是Wasserstein空间中连接 $p_0$ 和 $p_1$ 的"最短路径"
  - **关键意义**：OT测地线给出的是直线路径 $x_t = (1-t)x_0 + tT(x_0)$，即McCann插值——这正是OT-CFM和Rectified Flow的数学基础

> **过渡**：最优传输给出了分布之间"最短路径"的数学描述——Wasserstein测地线。但如何用神经网络实现这条最短路径？连续归一化流（CNF）提供了一种基于ODE的框架，但训练效率低下。Flow Matching正是对CNF训练方式的革新。

**来源**：Villani (2008) "Optimal Transport: Old and New"; Santambrogio (2015) "Optimal Transport for Applied Mathematicians"; Peyré & Cuturi (2019) "Computational Optimal Transport"; winter_school/JGondzio-Lecture5.md (离散OT)

---

### 14.2 连续归一化流（CNF）

**核心观点**：归一化流（Normalizing Flow）通过可逆变换将简单分布映射到复杂分布。离散NF需要计算Jacobian行列式，限制了网络架构的自由度；连续NF（CNF）将离散变换推向连续极限，用Neural ODE描述，只需计算Jacobian的迹（瞬时变量替换公式），释放了架构设计自由度。然而，CNF的最大瓶颈是训练时需要求解ODE——既慢又可能不稳定。Flow Matching的核心突破正是：**无需模拟ODE即可训练CNF**。

#### 14.2.1 离散归一化流回顾

- **基本思想**：通过一系列可逆变换 $f = f_K \circ \cdots \circ f_1$，将基础分布 $q(z_0)$ 映射到目标分布 $p(x)$
  - 前向：$z_K = f(z_0)$，从噪声到数据
  - 逆向：$z_0 = f^{-1}(z_K)$，从数据到噪声
- **变量替换公式**：
  $$\log p(z_K) = \log q(z_0) - \sum_{k=1}^{K}\log\left|\det\frac{\partial f_k}{\partial z_{k-1}}\right|$$
- **离散NF的局限**
  - Jacobian行列式计算代价高（$O(d^3)$），除非限制架构（如planar/radial flow、RealNVP、Glow）
  - 架构受限：必须设计行列式"可计算"的可逆变换
  - 深度固定：$K$ 步变换，无法自适应调整计算精度
- **与第8-9章的联系**：VAE用不可逆编码器+解码器，牺牲精确似然计算换取架构自由度；NF保持可逆性，以架构限制为代价获得精确似然。CNF是两者的"最佳结合"——可逆+架构自由。

#### 14.2.2 Neural ODE与连续时间变换

- **从离散到连续**：将 $K$ 步离散变换推向连续极限，$K \to \infty$，$\Delta t \to 0$
  - 离散变换：$z_{k+1} = z_k + f_\theta(z_k, t_k)\Delta t$
  - 连续极限：$\frac{dz(t)}{dt} = v_\theta(z(t), t)$——**Neural ODE**（Chen et al., 2018）
- **Neural ODE的核心特性**
  - ODE求解器自适应选择求值点（不再固定步数）
  - 前向与逆向使用同一动力学（自动可逆，无需存储中间状态）
  - 内存效率：伴随方法（adjoint method）实现 $O(1)$ 内存的梯度计算
- **Neural ODE的解**
  - 给定初始值 $z(0) = z_0$，通过ODE求解器得到 $z(t)$
  - 记号：$z(t) = \text{ODESolve}(v_\theta, z_0, 0, t)$
  - 推前映射：$z(t)$ 是由向量场 $v_\theta$ 定义的流（flow）

#### 14.2.3 瞬时变量替换公式

- **定理（Chen et al., 2018, Theorem 1）**：设 $z(t)$ 由 $\frac{dz}{dt} = v_\theta(z, t)$ 定义，则
  $$\frac{\partial \log p(z(t))}{\partial t} = -\nabla \cdot v_\theta(z(t), t) = -\text{tr}\left(\frac{\partial v_\theta}{\partial z}\right)$$
- **与离散NF的对应**
  - 离散：$\log\left|\det\frac{\partial f}{\partial z}\right|$ → 连续：$\int_0^T \text{tr}\left(\frac{\partial v_\theta}{\partial z}\right)dt$
  - 行列式 → 迹：这是连续极限带来的核心简化！$\det \to \text{tr}$ 将计算复杂度从 $O(d^3)$ 降至 $O(d)$（配合Hutchinson估计器）
- **似然计算**
  $$\log p(z(T)) = \log p(z(0)) - \int_0^T \text{tr}\left(\frac{\partial v_\theta(z(t), t)}{\partial z}\right)dt$$
  - FFJORD（Grathwohl et al., 2019）：用Hutchinson无偏估计器近似迹
    $$\text{tr}\left(\frac{\partial v_\theta}{\partial z}\right) \approx \epsilon^T \frac{\partial v_\theta}{\partial z}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

#### 14.2.4 CNF的训练瓶颈：仿真代价

- **最大似然训练**
  - 目标：$\max_\theta \mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]$
  - 需要：对每个训练样本 $x$，求解ODE从 $t=0$ 到 $t=T$，计算迹积分
  - 计算代价：每个样本需要一次完整ODE求解 + 多次向量场评估
- **训练瓶颈**
  1. **ODE求解代价**：训练的每一步都需要求解ODE——计算密集且可能数值不稳定
  2. **迹估计方差**：Hutchinson估计器引入额外方差，影响训练稳定性
  3. **刚性ODE**：训练后期向量场可能变化剧烈，ODE求解器需要更多步数
- **与扩散模型的对比**
  | 特性 | CNF（最大似然训练） | 扩散模型（DSM训练） |
  |---|---|---|
  | 训练时是否需要模拟 | ✅ 需要求解ODE | ❌ 不需要（DSM是单步损失） |
  | 似然计算 | 精确（需迹估计） | 精确（需K积分，第8章） |
  | 采样方式 | ODE求解 | SDE或ODE求解 |
  | 路径形态 | 自由（由 $v_\theta$ 决定） | 固定（由SDE结构决定） |
- **关键洞察**：扩散模型之所以训练高效，是因为它绕过了"模拟ODE"这一步——DSM损失只需单步预测。Flow Matching的核心思想正是将这一优势移植到CNF：**用回归损失训练向量场，无需模拟ODE**。

> **过渡**：CNF的架构自由度是优势，但训练时的仿真代价是致命瓶颈。扩散模型通过DSM训练绕过了仿真——那么，能否用类似思路训练CNF？答案是Flow Matching：直接回归向量场，无需模拟ODE。

**来源**：Chen et al. (2018) "Neural Ordinary Differential Equations"; Grathwohl et al. (2019) "FFJORD"; Rezende & Mohamed (2015) "Variational Inference with Normalizing Flows"

---

### 14.3 Flow Matching

**核心观点**：Flow Matching的核心突破是**条件Flow Matching定理**——回归条件向量场的损失函数与回归边际向量场的损失函数具有完全相同的梯度，而条件向量场是闭式可计算的。这意味着：**训练CNF无需模拟ODE，无需计算迹，只需在采样对 $(x_0, x_1)$ 上回归向量场**。选择不同的耦合 $\Pi$ 和条件路径 $p_t(x|x_1)$，可以得到不同的Flow Matching变体。OT-CFM选择最优传输耦合，产生更直的路径和更高质量的样本。DDIM是Flow Matching在扩散耦合下的特例——这揭示了扩散模型与Flow Matching的深层统一。

#### 14.3.1 向量场与流ODE：从点到分布的传输

- **流ODE**：给定时间相关的向量场 $v = \{v_t\}_{t \in [0,1]}$，定义ODE
  $$\frac{dx_t}{dt} = v_t(x_t), \quad x_1 \sim q$$
  - 从初始分布 $q$ 出发，沿向量场 $v_t$ 传输到目标分布 $p$
  - 记号：$x_t = \text{RunFlow}(v, x_1, t)$
- **推前分布**：向量场 $v$ 将初始分布 $q$ 推前为目标分布 $p$
  $$q \xrightarrow{v} p$$
  - 等价写法：$p_0 = \text{RunFlow}(v, \cdot, 0)_\# q$
- **生成模型的目标**：学习向量场 $v^*$，使得 $q \xrightarrow{v^*} p_{\text{data}}$
- **与概率流ODE的关系**（第7章7.4节）
  - PF-ODE：$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2\nabla\log p_t(x)$——向量场由得分函数决定
  - Flow ODE：$\frac{dx_t}{dt} = v_t(x_t)$——向量场直接参数化，不依赖得分函数
  - **关键区别**：PF-ODE的向量场是"间接"的（需先学得分函数），Flow ODE的向量场是"直接"的（直接回归学习）

#### 14.3.2 点wise流与边际流

- **点wise流**（Pointwise Flow）：连接两个点 $x_0$ 和 $x_1$ 的向量场
  - $v_t^{[x_1, x_0]}$：将 $x_1$ 传输到 $x_0$ 的向量场
  - 满足：从 $x_1$ 出发，沿 $v^{[x_1, x_0]}$ 积分到达 $x_0$
  - 点wise流不唯一：两点之间可以走曲线，也可以走直线
  - **最简单的选择——直线插值**：
    $$x_t = (1-t)x_1 + t\,x_0, \quad v_t^{[x_1, x_0]}(x_t) = x_0 - x_1$$
    - 速度恒定、路径最短——这就是Rectified Flow的起点
- **边际流**（Marginal Flow）：将点wise流"聚合"为传输整个分布的向量场
  - 给定耦合 $\Pi_{q,p}$（源-目标对的联合分布），所有点wise流的加权平均
  - **边际向量场公式**：
    $$v_t^*(x_t) = \mathbb{E}_{x_0, x_1 | x_t}[v_t^{[x_1, x_0]}(x_t) \mid x_t]$$
  - 直觉：在位置 $x_t$ 处，边际速度 = 所有可能的粒子速度的概率加权平均
  - **边际概率路径**：
    $$p_t(x) = \int p_t(x|x_1)\,q(x_1)\,dx_1$$
    - 即条件路径的边际化
- **物理直觉**：点wise流是单个粒子的轨迹，边际流是整个粒子群的"平均速度场"——正如气象学中单个气团的轨迹 vs 整个风场的平均速度

#### 14.3.3 条件Flow Matching：突破仿真瓶颈

- **Flow Matching（FM）目标**——直接回归边际向量场（不可计算）：
  $$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, p_t(x)}\left[\|v_\theta(x, t) - v_t^*(x)\|^2\right]$$
  - 不可计算的原因：$v_t^*(x)$ 需要对 $p(x_0, x_1 | x_t)$ 做期望——而 $p(x_0, x_1 | x_t)$ 本身需要已知向量场才能计算（鸡生蛋问题）
- **条件Flow Matching（CFM）目标**——回归条件向量场（可计算）：
  $$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p_t(x|x_1)}\left[\|v_\theta(x, t) - v_t(x|x_1)\|^2\right]$$
  - 条件向量场 $v_t(x|x_1)$ 是闭式可计算的——它就是连接 $x_1$ 和 $x_0$ 的点wise流的速度
  - 条件路径 $p_t(x|x_1)$ 也是闭式的（如高斯路径）
- **核心定理（Lipman et al., 2023, Theorem 1 & 2）**：CFM损失与FM损失关于 $\theta$ 的梯度完全相同：
  $$\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)$$
  - **含义**：最小化CFM损失 = 最小化FM损失——但我们只需要计算CFM损失！
  - **类比**：这与DSM定理（第6章）如出一辙——DSM证明回归去噪器等价于回归得分函数；CFM证明回归条件向量场等价于回归边际向量场
  - 证明思路：$\nabla_\theta \mathcal{L}_{\text{FM}}$ 中的不可计算项 $v_t^*$ 在取期望后被消去，与 $\nabla_\theta \mathcal{L}_{\text{CFM}}$ 中可计算的 $v_t(\cdot|x_1)$ 产生相同梯度（详细证明见附录14B）
- **CFM的训练算法**
  ```
  输入：数据集 {x_0^(i)}，基础分布 q，条件路径 p_t(x|x_1)，条件向量场 v_t(x|x_1)
  重复：
    1. 采样 x_0 ~ p_data, x_1 ~ q
    2. 采样 t ~ U(0,1)
    3. 采样 x ~ p_t(x|x_1)
    4. 计算损失 L = ||v_θ(x, t) - v_t(x|x_1)||^2
    5. 更新 θ ← θ - η∇_θ L
  ```
- **CFM的采样算法**
  ```
  输入：训练好的向量场 v_θ
  1. 采样 x_1 ~ q（如标准高斯）
  2. 求解 ODE: dx_t/dt = v_θ(x_t, t)，从 t=1 到 t=0
  3. 输出 x_0
  ```
- **与扩散模型训练的对比**
  | 特性 | 扩散模型（DDPM/DSM） | Flow Matching（CFM） |
  |---|---|---|
  | 训练目标 | 回归噪声 $\epsilon$ 或得分 $s$ | 回归向量场 $v$ |
  | 条件信息 | 噪声水平 $t$ + 含噪数据 $x_t$ | 时间 $t$ + 插值点 $x_t$ |
  | 是否需要模拟 | ❌ 不需要 | ❌ 不需要 |
  | 采样方式 | SDE或ODE | ODE |
  | 路径灵活性 | 受SDE结构约束 | 自由选择条件路径 |

#### 14.3.4 高斯条件路径与条件向量场

- **高斯条件路径**（Lipman et al., 2023）
  $$p_t(x|x_1) = \mathcal{N}(x; \mu_t(x_1), \sigma_t(x_1)^2 I)$$
  - 均值路径：$\mu_t(x_1) = (1-t)\mu + t\,x_1$，从基础分布均值 $\mu$ 到数据点 $x_1$
  - 方差路径：$\sigma_t(x_1)$ 从 $\sigma_1$ 到 $\sigma_0$（通常 $\sigma_0 = 0$，即到达数据点时无噪声）
- **条件向量场**（Theorem 3, Lipman et al., 2023）
  $$v_t(x|x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)}(x - \mu_t(x_1)) + \mu_t'(x_1)$$
  - 这是生成高斯路径 $p_t(x|x_1)$ 的唯一向量场
  - 当 $\sigma_t \to 0$（确定性路径）时，条件向量场退化为直线速度 $v_t = x_1 - x_0$

#### 14.3.5 OT-CFM：最优传输条件流

- **动机**：独立耦合 $\Pi = q \otimes p$ 导致路径交叉——源点 $x_1$ 和目标点 $x_0$ 随机配对，传输路径可能绕远。最优耦合使路径更直、更短。
- **OT-CFM的核心修改**：将独立耦合替换为最优传输耦合
  - 独立耦合：$\Pi_{\text{ind}} = q \otimes p$，$x_1$ 和 $x_0$ 独立采样
  - OT耦合：$\Pi_{\text{OT}} = \arg\min_{\gamma \in \Pi(q,p)} \int \|x_0 - x_1\|^2 d\gamma$——Wasserstein最优耦合
- **OT-CFM的训练**
  - 采样 $(x_0, x_1) \sim \Pi_{\text{OT}}$（而非独立采样）
  - 其余与CFM完全相同
  - **Minibatch OT**（Pooladian et al., 2023; Tong et al., 2024）：在每个minibatch内求解OT问题
    - 复杂度：$O(n^3)$（Sinkhorn加速至 $O(n^2)$）
    - 实践效果：路径更直，FID更低，收敛更快
- **McCann插值与直线路径**
  - OT耦合下的条件路径：$x_t = (1-t)x_1 + t\,x_0$（McCann插值/直线插值）
  - 条件向量场：$v_t(x|x_0, x_1) = x_0 - x_1$（常数速度！）
  - **直觉**：OT耦合使得每个源点配对到最近的目标点，路径自然更短更直
- **独立耦合 vs OT耦合的几何对比**
  - 独立耦合：路径交叉缠绕，需要更多ODE步数才能准确追踪
  - OT耦合：路径近乎平行，少数ODE步数即可准确追踪
  - 这就是Flow Matching"更快采样"的几何根源

#### 14.3.6 DDIM即Flow Matching：与第7章的统一

- **核心等价性**（Nakkiran et al., 2024, Claim 4）：DDIM采样器等价于Flow Matching的一种特殊形式
  - DDIM = Flow Matching with 扩散耦合（diffusion coupling）+ 特定的点wise流
  - 具体地：DDIM的连续极限是向量场 $v_t(x_t) = \frac{1}{2t}\mathbb{E}[x_0 - x_t | x_t]$
  - 经过时间重参数化后，与扩散耦合下的Flow Matching一致
- **意义**：DDIM不是扩散模型独有的采样器——它是Flow Matching框架的一个特例
  - 这解释了为什么DDIM是确定性采样器（因为Flow Matching本质上就是ODE）
  - 这也解释了为什么DDIM比DDPM更"直"（Flow Matching追求直线路径）
- **三种采样器的统一视角**
  | 采样器 | 框架 | 随机性 | 路径 |
  |---|---|---|---|
  | DDPM | 逆向SDE | 随机 | 弯曲 |
  | DDIM | Flow Matching（扩散耦合） | 确定性 | 较直 |
  | OT-CFM | Flow Matching（OT耦合） | 确定性 | 最直 |

> **过渡**：Flow Matching通过OT耦合获得了更直的路径，但实践中OT耦合的精确计算代价高昂。Rectified Flow从另一个角度解决路径弯曲问题：通过直线插值+Reflow迭代，逐步拉直路径，无需显式计算OT耦合。

**来源**：Lipman et al. (2023) "Flow Matching for Generative Modeling"; Nakkiran et al. (2024) "Step-by-Step Diffusion" §4; Pooladian et al. (2023) "Multisample Flow Matching"; Tong et al. (2024) "Conditional Flow Matching"

---

### 14.4 Rectified Flow

**核心观点**：Rectified Flow（Liu et al., 2023）是Flow Matching的一个具体实例化——选择直线插值作为条件路径，速度场恒为 $v = x_0 - x_1$。其核心贡献是Reflow过程：通过迭代"重新配对"，将弯曲的ODE轨迹逐步拉直。直线化带来了巨大的实践优势——越直的路径，Euler离散化误差越小，所需的采样步数越少。极限情况下，完全笔直的路径只需一步Euler步即可精确求解，实现单步生成。

#### 14.4.1 直线插值与Rectified Flow的训练

- **Rectified Flow的基本设定**
  - 条件路径：$x_t = (1-t)x_1 + t\,x_0$（直线插值）
  - 条件向量场：$v_t(x_t | x_0, x_1) = x_0 - x_1$（常数速度）
  - 训练目标：
    $$\mathcal{L}_{\text{RF}}(\theta) = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - (x_0 - x_1)\|^2\right]$$
  - 其中 $x_0 \sim p_{\text{data}}$，$x_1 \sim q$（标准高斯），$x_t = (1-t)x_1 + t\,x_0$
- **与扩散模型训练的对比**
  - 扩散模型：回归噪声 $\epsilon_\theta(x_t, t) \approx \epsilon$，其中 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$
  - Rectified Flow：回归速度 $v_\theta(x_t, t) \approx x_0 - x_1$
  - 关键区别：扩散模型的插值是弯曲的（由 $\bar\alpha_t$ 调度决定曲率），Rectified Flow的插值是直线的

#### 14.4.2 路径交叉与弯曲的根源

- **为什么初始配对的路径是弯曲的？**
  - 虽然每个条件路径 $(x_0, x_1)$ 都是直线，但边际向量场 $v_t^*(x_t)$ 在不同条件路径的交叉点处需要"折中"
  - 路径交叉（trajectory crossing）：两条直线路径在空间中相交，ODE求解器在交叉点处需要选择方向
  - 边际向量场在交叉点处"平均"了不同粒子的速度，导致非直线运动
- **直觉**：想象多辆车从不同起点出发，沿直线驶向不同终点——如果两条路线交叉，交叉处的交通规则（平均速度）使得实际行驶路线弯曲
- **直线度度量**（Straightness）：
  $$S(Z) = \mathbb{E}\left[\int_0^1 \|\dot{Z}_t - (Z_0 - Z_1)\|^2 dt\right]$$
  - $S = 0$：完全笔直（单步Euler步精确）
  - $S > 0$：存在弯曲，需要多步ODE求解
  - 弯曲越严重（$S$ 越大），离散化误差越大，所需步数越多

#### 14.4.3 Reflow：迭代直线化

- **Reflow的核心思想**：用当前flow的端点重新配对，生成更直的新flow
  - 第1轮：$Z_1 = \text{RectFlow}((X_0, X_1))$，其中 $X_0 \sim p_{\text{data}}$，$X_1 \sim q$
  - 第2轮（Reflow）：$Z_2 = \text{RectFlow}((Z_1^0, Z_1^1))$
    - $Z_1^0$：第1轮flow的终点（生成样本）
    - $Z_1^1$：第1轮flow的起点（噪声样本）
    - 用 $(Z_1^1, Z_1^0)$ 作为新的训练对
  - 第k轮：$Z_k = \text{RectFlow}((Z_{k-1}^1, Z_{k-1}^0))$
- **Reflow的直线化效果**
  - 每轮Reflow都减少路径交叉——"绕线"被逐步解开
  - 直线度 $S$ 单调递减（理论上）
  - 传输代价 $\mathbb{E}[\|Z_1 - Z_0\|^2]$ 也单调递减
- **Reflow与最优传输的联系**
  - Reflow的极限是OT映射——无限次Reflow后，路径完全不交叉，等价于OT测地线
  - 这解释了为什么Reflow能同时降低直线度和传输代价
- **Reflow的实践价值**
  - 1-Rectified Flow：多步采样（与普通FM类似）
  - 2-Rectified Flow（1次Reflow）：少步采样已有很好质量
  - 3-Rectified Flow及以上：接近单步生成
  - Reflow本质上是**蒸馏**——用多步模型的输出训练少步/单步模型

#### 14.4.4 与扩散模型的全面对比

| 维度 | 扩散模型 | Flow Matching | Rectified Flow |
|---|---|---|---|
| 训练目标 | 回归噪声 $\epsilon$ / 得分 $s$ | 回归向量场 $v$ | 回归速度 $x_0 - x_1$ |
| 正向过程 | SDE（加噪） | 无（直接定义条件路径） | 无（直线插值） |
| 逆向过程 | SDE或ODE | ODE | ODE |
| 路径形态 | 由噪声调度决定（弯曲） | 由条件路径决定 | 直线 |
| 加速策略 | DDIM/DPMSolver | OT耦合 | Reflow |
| 极限少步 | 4-8步（质量下降） | 4-8步（OT-CFM较好） | 1步（Reflow后） |
| 理论优雅度 | SDE框架深刻 | ODE框架简洁 | 直线直觉清晰 |

> **过渡**：Rectified Flow通过直线化和Reflow实现了高效的少步生成，但在工业级文本到图像生成中，算法只是拼图的一半——架构是另一半。SD3/Flux将Rectified Flow与DiT架构结合，实现了SOTA性能。

**来源**：Liu et al. (2023) "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"; Liu et al. (2024) "InstaFlow"

---

### 14.5 前沿组合：SD3/Flux = Rectified Flow + DiT

**核心观点**：Stable Diffusion 3（Esser et al., 2024）和Flux是工业级Flow Matching的成功验证——它们将Rectified Flow的训练目标与Diffusion Transformer（DiT）架构结合，在文本到图像生成中达到SOTA性能。关键创新包括：Logit-Normal时间采样（给中间时间步更多权重）、多模态DiT（MMDiT，图文双模态交互）、以及在大规模数据上的缩放规律。

#### 14.5.1 SD3的核心设计

- **Rectified Flow训练目标**
  - SD3采用Rectified Flow的直线插值路径
  - 时间步均匀采样 $t \sim U(0,1)$，插值 $x_t = (1-t)x_1 + t\,x_0$
- **Logit-Normal时间采样**（SD3的关键改进）
  - 问题：均匀采样对所有时间步同等对待，但中间时间步的预测更困难（信号和噪声量级相当）
  - 改进：用Logit-Normal分布采样时间步，使中间时间步的概率更高
    $$\pi_{\ln}(t) = \frac{1}{\sigma\sqrt{2\pi}} \cdot \frac{1}{t(1-t)} \exp\left(-\frac{(\text{logit}(t) - m)^2}{2\sigma^2}\right)$$
  - 其中 $\text{logit}(t) = \log(t/(1-t))$
  - 参数 $m$ 控制重心偏移，$\sigma$ 控制集中程度
  - 效果：在少步采样（4步、8步）下FID显著优于均匀采样
- **与其他扩散轨迹的对比**
  - SD3论文在60种扩散轨迹（LDM、EDM、ADM等）上做了系统对比
  - Rectified Flow + Logit-Normal采样在所有设置下优于或等于其他轨迹

#### 14.5.2 MMDiT架构：双模态Diffusion Transformer

- **从DiT到MMDiT**
  - DiT（Peebles & Xie, 2023）：用Transformer替代UNet，patch化输入，adaLN-Zero调制
  - MMDiT（Multi-Modal DiT）：扩展DiT以同时处理图像和文本两种模态
- **MMDiT的关键设计**
  - 双权重流：图像token和文本token使用独立的权重集
  - 双向信息流：图像token和文本token在Transformer层中交互
  - 序列构建：图像patch序列 + 文本token序列 → 拼接 → Transformer处理
  - 时间和条件调制：通过adaLN-Zero注入时间步 $t$ 和文本条件信息
- **与UNet的对比**
  - UNet：CNN架构，局部感受野，归纳偏置强
  - DiT/MMDiT：Transformer架构，全局感受野，缩放性好
  - SD3结论：DiT在参数量增大时遵循可预测的缩放规律，验证损失下降与生成质量提升正相关

#### 14.5.3 Flux与后续发展

- **Flux**（Black Forest Labs, 2024）
  - 基于Rectified Flow + 改进的DiT架构
  - 在SD3基础上进一步优化训练策略和架构设计
  - 支持高分辨率图像生成（1024×1024及以上）
- **Flow Matching在工业界的验证**
  - Rectified Flow已从学术概念走向工业部署
  - 证明了"直线路径 + Transformer架构"的组合在大规模生成任务中的有效性
  - 预示着生成模型的未来方向：从SDE到ODE，从弯曲到直线，从CNN到Transformer

#### 14.5.4 回顾与展望

- **本章的叙事线回顾**
  - OT基础→CNF困境→Flow Matching突破→Rectified Flow优化→SD3/Flux落地
  - 核心转变：得分函数 → 向量场，SDE → ODE，弯曲路径 → 直线路径
- **Flow Matching与扩散模型的统一视角**
  - 扩散模型是Flow Matching的特例（特定的条件路径+耦合选择）
  - Flow Matching是更一般的框架，包容扩散模型和Rectified Flow
  - 两条路径（得分/向量场）在ODE层面汇合——都生成概率流
- **与第15章的衔接**：14.5节涉及的DiT/MMDiT架构，将在第15章详细剖析其设计原理和缩放规律

**来源**：Esser et al. (2024) "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"; Peebles & Xie (2023) "Scalable Diffusion Models with Transformers"; Black Forest Labs (2024) Flux

---

## 附录

### 附录14A 连续性方程与边际向量场

> 定位：14.3节中"边际向量场生成边际概率路径"的严格证明需要连续性方程。此证明对理解FM的数学基础重要，但涉及较多PDE知识，放入正文会打断叙事节奏。

- 连续性方程（Continuity Equation）：向量场 $v_t$ 生成概率路径 $p_t$ 的充要条件
  $$\frac{\partial p_t(x)}{\partial t} + \nabla \cdot (p_t(x)\,v_t(x)) = 0$$
- 证明：从ODE $\frac{dx_t}{dt} = v_t(x_t)$ 和变量替换公式出发
- 边际向量场公式 $v_t^* = \mathbb{E}[v_t^{[x_1,x_0]} | x_t]$ 满足连续性方程的证明

### 附录14B 条件Flow Matching定理的完整证明

> 定位：CFM定理是Flow Matching的核心理论结果，证明需要仔细的积分运算，放入附录保持正文简洁。

- 定理陈述：$\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)$
- 证明步骤：
  1. 展开 $\mathcal{L}_{\text{FM}}$ 和 $\mathcal{L}_{\text{CFM}}$ 的梯度
  2. 利用条件期望的塔性质（tower property）建立联系
  3. 证明两个梯度中的不可计算项恰好对消
- 与DSM定理证明的类比（第6章）

### 附录14C Wasserstein空间的黎曼结构

> 定位：对理解OT与生成模型的关系有深层价值，但涉及黎曼几何，超出本书主要读者群体的背景。

- Wasserstein空间的切空间与黎曼度量
- Otto calculus：Wasserstein梯度流
- 与Langevin动力学的联系：Langevin SDE是KL散度的Wasserstein梯度流+噪声

---

## 素材来源映射

| 节 | 核心素材 | 补充来源 |
|---|---|---|
| 14.0 | 全书叙事 | book_plan.md |
| 14.1.1-14.1.2 | Villani (2008); Santambrogio (2015) | winter_school/JGondzio-Lecture5.md |
| 14.1.3 | Villani (2008) Ch.5 | Peyré & Cuturi (2019) |
| 14.1.4 | Villani (2008); Santambrogio (2015) | — |
| 14.2.1 | Rezende & Mohamed (2015) | 第8章变分推断 |
| 14.2.2 | Chen et al. (2018) Neural ODE | — |
| 14.2.3 | Chen et al. (2018) Theorem 1; Grathwohl et al. (2019) | — |
| 14.2.4 | FFJORD; 扩散模型对比 | 第7章7.4节PF-ODE |
| 14.3.1 | 2406.08929v2 §4.1 | 第7章7.4节 |
| 14.3.2 | 2406.08929v2 §4.2-4.3 | — |
| 14.3.3 | Lipman et al. (2023) Theorem 1-2; 2406.08929v2 §4.5 | 第6章DSM定理 |
| 14.3.4 | Lipman et al. (2023) Theorem 3 | — |
| 14.3.5 | Pooladian et al. (2023); Tong et al. (2024) | 2406.08929v2 §4.4 |
| 14.3.6 | 2406.08929v2 §4.6 | 第7章7.4-7.5节DDIM |
| 14.4.1 | Liu et al. (2023) | 2406.08929v2 §4.4 |
| 14.4.2 | Liu et al. (2023) §2 | — |
| 14.4.3 | Liu et al. (2023) §3 Reflow | — |
| 14.4.4 | 全章综合 | — |
| 14.5.1 | Esser et al. (2024) SD3 | — |
| 14.5.2 | Esser et al. (2024); Peebles & Xie (2023) DiT | — |
| 14.5.3 | Black Forest Labs (2024) Flux | — |
| 14.5.4 | 全章综合 | — |

---

## 章节逻辑流

```
14.1 最优传输基础（为何OT？路径最优性）
      │
      │ "OT给出了最优路径的数学描述，如何实现？"
      ▼
14.2 连续归一化流（CNF的ODE框架）
      │
      │ "CNF架构自由但训练需仿真，能否绕过？"
      ▼
14.3 Flow Matching（突破：CFM定理，无需仿真训练CNF）
      │
      │ "FM路径更直，但初始配对仍可能交叉，如何拉直？"
      ▼
14.4 Rectified Flow（直线插值 + Reflow拉直 → 少步/单步生成）
      │
      │ "算法就绪，架构如何配合？"
      ▼
14.5 SD3/Flux = Rectified Flow + DiT（工业级验证与SOTA）
```

---

## 缺失素材清单

| 素材 | 用途 | 紧急程度 | 状态 |
|---|---|---|---|
| 最优传输2D可视化（Monge映射/OT耦合/独立耦合对比） | 14.1 几何直觉 | ⭐⭐ 中 | ❌ 待补充 |
| Wasserstein测地线/McCann插值动画 | 14.1.4 测地线直觉 | ⭐⭐ 中 | ❌ 待补充 |
| 离散NF vs CNF对比图（流形变形可视化） | 14.2.1-14.2.2 架构对比 | ⭐ 低 | ❌ 待补充 |
| Flow Matching训练/采样伪代码流程图 | 14.3.3 算法示意 | ⭐⭐ 中 | ❌ 待补充 |
| 独立耦合 vs OT耦合路径可视化 | 14.3.5 路径对比 | ⭐⭐⭐ 高 | ❌ 待补充 |
| Reflow逐轮拉直可视化（1-RF → 2-RF → 3-RF） | 14.4.3 直线化效果 | ⭐⭐⭐ 高 | ❌ 待补充 |
| 扩散模型 vs FM vs RF 的轨迹对比图 | 14.4.4 全面对比 | ⭐⭐⭐ 高 | ❌ 待补充 |
| SD3/MMDiT架构图 | 14.5.2 架构说明 | ⭐⭐ 中 | ❌ 待补充 |
| Logit-Normal采样 vs 均匀采样FID对比数据 | 14.5.1 时间采样效果 | ⭐ 低 | ❌ 待补充 |
| Rectified Flow CIFAR-10定量结果（FID/IS） | 14.4 实验验证 | ⭐ 低 | ❌ 待补充 |
