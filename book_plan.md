# 《图像生成：从贝叶斯到扩散》写作规划

---

## 一、核心思想

将图像逆问题求解与图像生成技术视为同一本质问题的不同表达。**逆问题求解 = 条件图像生成**。通过贝叶斯推断、得分匹配、扩散采样等框架实现理论统一与实践贯通。

**核心论点**：扩散模型是逆问题自然发展的产物。逆问题需要后验采样 → 采样需要得分函数 → 得分函数可以从去噪器中提取（Tweedie）→ 去噪器可通过得分匹配训练 → 多步得分驱动采样即扩散模型。这条链有明确的因果逻辑：逆问题的需求驱动了每一步方法演化。变分路径（VAE → 层级VAE → 扩散）是另一条独立发展但殊途同归的路线，提供了互补的数学视角。

---

## 二、书名策略

主标题"图像生成"为流量入口（市场导向），内容内核为逆问题与生成模型的理论统一性（学术导向）。

- 首选书名：**《图像生成：从贝叶斯到扩散》**
- 副标题可选：——采样与变分两条路径

**市场定位**：面向大众市场的技术读物，书名以"图像生成"等高流量关键词为核心降低认知门槛、提升传播性与销售潜力。扩散模型/图像生成是搜索热词，"逆问题"对大众太学术。策略：**用生成的外壳，讲逆问题的灵魂**——内容不变，换入口。

---

## 三、双路径统一架构

```
           ┌── 采样路径（主线·逆问题驱动）：ULA → Langevin → Score → Diffusion (SDE视角)
贝叶斯框架 ─┤
           └── 变分路径（副线·生成建模驱动）：ELBO → VAE → 层级VAE → Diffusion (似然视角)
```

两条路径均始于贝叶斯框架，终于扩散模型的数学等价性（**Score Matching损失 ≡ 变分下界**），构成全书核心理论主干。第12章"殊途同归"为全书理论高潮，第13章"条件生成=逆问题求解"为全书论点闭环——条件扩散采样回到第1章的逆问题，完成从逆问题出发、经由漫长理论旅程、回到逆问题的完整闭环。

**采样路径是主线**：它直接由逆问题的后验采样需求驱动——MAP丢弃不确定性→需要后验采样→MCMC效率低→Langevin利用梯度→得分函数需要学习→得分匹配→多步扩散。每一步都是逆问题局限性的自然回应。**变分路径是副线**：它由纯生成建模社区独立发展，提供从优化角度理解扩散模型的互补视角。第12章揭示两条路径的数学等价性，既是理论高潮，也印证了"逆问题的自然发展"与"生成建模的独立探索"殊途同归。

---

## 四、关键理论桥梁

1. **显式先验的天花板 → 学习型先验的必然性**（第2章）：真实图像分布远比高斯/Laplace/TV复杂，显式先验的表达能力有根本局限——这驱动了从手工先验到数据驱动先验的转变
2. **PnP去噪器 → 得分函数估计器**（Tweedie等式）：去噪即学习先验梯度——绕过归一化常数Z的计算，是"隐式先验"的关键实现
3. **ULA朗之万采样 → 扩散模型**：多步扩散过程是Langevin的推广——优化+噪声=采样，多步采样=扩散
4. **"离散→连续→再离散"螺旋**：Langevin(离散迭代) → 扩散SDE(连续方程) → Euler-Maruyama(再离散化) → DDPM(特定离散方案)
5. **VAE编码/解码 → 扩散加噪/去噪**：扩散 = 无穷层级的层级VAE——变分路径的独立推导
6. **条件生成 = 逆问题求解**：条件扩散采样完成第1章到第13章的闭环——全书核心论点的落地
7. **后验得分分解**（第13章新增）：∇log p(x_t|y) = ∇log p(x_t) + ∇log p(y|x_t)，第一项由扩散模型提供，第二项的近似是所有扩散逆问题方法的核心差异

---

## 五、Transformer/DiT 的定位

DiT 本质是将扩散模型的去噪网络从 UNet 替换为 Transformer，扩散数学框架不变，属于**架构选择而非新理论**。纳入实践章节（第15章），不与 Score/变分争理论地位。

具体替换内容：
- **Patchify**：把图像切成 patch 序列（同 ViT）
- **Transformer Blocks**：替换 UNet 的卷积残差块
- **adaLN-Zero**：自适应 LayerNorm 注入时间步条件

本书聚焦扩散模型，不涉及自回归生成（VQGAN+Transformer）和掩码生成（MaskGIT）。

---

## 六、微分方程离散化的归属

SDE/ODE 的数值离散化（Euler-Maruyama 等）归属**第7章扩散 SDE 视角**，不属于 Flow Matching。Flow Matching 是独立生成框架（学向量场 vs 学得分函数），归属第14章。

---

## 七、全书18章完整架构

```
Part I   贝叶斯基石
           第1章   逆问题与贝叶斯推断
           第2章   先验：贝叶斯推断的灵魂
           第3章   从MAP到后验探索

Part II  采样路径（主线·逆问题驱动）
           第4章   MCMC与ULA算法
                  【逆问题动机】MAP丢弃不确定性 → 需要从后验采样
           第5章   朗之万动力学与得分函数
                  【逆问题动机】标准MCMC在高维图像上效率极低 → 需要利用后验梯度（得分函数）
           第6章   得分匹配：从去噪中学习得分
                  【逆问题动机】得分函数无法直接计算 → 需要从数据中学习 → 得分匹配
           第7章   扩散模型：SDE视角（含离散化、概率流ODE、DDPM）
                  【逆问题动机】单步Langevin对复杂分布采样不足 → 需要多时间步扩散 → 扩散模型

Part III 变分路径（副线·生成建模驱动）
           第8章   变分推断与ELBO
                  【定位】另一条通向扩散的独立路径：从优化角度近似后验
           第9章   VAE与重参数化
           第10章  层级VAE与扩散的变分推导
           第11章  扩散模型：变分视角

Part IV  统一
           第12章  Score ≡ ELBO：殊途同归
           第13章  条件生成与逆问题求解（全书高潮·闭环）
                  【定位】扩散模型回到逆问题——全书核心论点的落脚点。
                  条件扩散采样 = 后验采样 = 逆问题求解，完成第1章到第13章的完整闭环。

Part V   前沿
           第14章  Flow Matching与最优传输

Part VI  实践与应用
           第15章  扩散模型的架构实践：UNet → DiT
           第16章  CT/MRI重建
           第17章  自监督学习与等变架构
           第18章  综合项目：用扩散模型求解自定义逆问题
```

### 各章节要点

#### 第1章 逆问题与贝叶斯推断

```
1.1  什么是逆问题？从观察到因果的推理
1.2  正向模型：线性算子y=Ax
     - 卷积模糊、下采样、Radon变换
1.3  不适定性：为什么逆问题难？
     - Hadamard适定性条件；病态性；inverse crime
1.4  噪声建模与似然函数
     - 高斯噪声 → L2数据项
     - Poisson噪声 → KL散度数据项（Calatroni P36-37）
     - 脉冲/Laplace噪声 → L1数据项（Calatroni P37-38）
1.5  贝叶斯框架：从似然到后验
     - 贝叶斯定理p(x|y)∝p(y|x)p(x)
     - 后验=似然×先验：信息融合的数学表达
```

#### 第2章 先验：贝叶斯推断的灵魂

```
2.1  先验的数学角色：正则化的概率诠释
     - -ln后验=数据项+正则项（Calatroni P29-44完整对应表）
2.2  经典先验族
     - 高斯先验 → Tikhonov正则化（及其推广：梯度高斯→Sobolev平滑）
     - Laplace先验 → L1/稀疏正则化
     - TV先验 → 全变差正则化（阶梯效应的局限与TGV改进思路见附录2B）
2.3  先验的质量：MMSE vs MAP估计器
     - MMSE估计器（后验均值）vs MAP估计器（后验众数）
     - 贝叶斯去噪器：去噪器=先验p(x)下的MMSE估计器（→第5章Tweedie等式的铺垫）
     - 共轭先验：高斯似然+高斯先验→高斯后验（→第3章闭式解的理论来源）
     - 误差分解五层次——不可约/近似/采样/优化/训练误差（Ratti P49-57，已升格为2.3节内独立小节）
2.4  从显式先验到隐式先验
     - 学习型先验：从手工设计到数据驱动
```

#### 第3章 从MAP到后验探索

```
3.1  MAP估计：后验众数求解
     - 贝叶斯决策理论：MAP在Bregman散度损失下的最优性（Pereyra P28）
3.2  优化基础
     - 凸性、Lipschitz梯度、强制性的理论前提（Calatroni P49-51）
     - 梯度下降：步长选择与收敛条件
3.3  Tikhonov正则化
     - 迭代求解（梯度下降）
     - 闭式解：DFT域直接求解（Calatroni P53）
3.4  稀疏优化与近端方法
     - ISTA/FISTA：近端梯度下降+Nesterov加速
     - 迭代硬阈值(IHT)：L0的近端算子（MIVA opt3）
     - 重加权L1：L0的连续松弛（MIVA opt3）
3.5  TV正则化与原始-对偶算法
     - Chambolle-Pock算法
     - ADMM：交替方向乘子法（Gondzio L6）
3.6  收敛性分析基础
     - Bregman距离与误差估计（Benning L2）
     - 源条件与收敛速率
3.7  从MAP到后验：为什么要探索后验分布？
     - MAP只是众数，丢失不确定性信息
     - 分叉点：采样还是近似？
```

#### 第4章 MCMC与ULA算法

```
4.1  Monte Carlo方法：从积分到采样
4.2  Metropolis-Hastings算法
     - 接受-拒绝准则；细致平衡条件
4.3  ULA：Langevin采样的Euler离散
     - ULA递推式；步长δ≤1/L
     - MH vs ULA：有偏但高效
4.4  MYULA：近端ULA
     - Moreau包络近似不可微势能（Pereyra P43-44）
     - 近端算子→平滑梯度的理论保证
4.5  加速采样方法（可选/进阶）
     - 过松弛采样（Pock L2）
     - 惯性Langevin算法(ILA)与欠阻尼Langevin（Pock L2）
4.6  MCMC收敛诊断
     - Burn-in；自相关函数；有效样本量(ESS)
```

#### 第5章 朗之万动力学与得分函数

```
5.1  从MCMC到Langevin SDE
     - 离散迭代→连续SDE的推导
     - Langevin方程：dx=∇log p(x)dt+√2dW
     - Fokker-Planck方程与平稳分布证明
5.2  得分函数：对数概率的梯度
     - ∇_x log p(x)的几何含义
     - 后验得分分解：似然得分+先验得分
     - 噪声扰动分布的得分函数
5.3  Tweedie等式：从去噪器到得分函数（从5.2独立出来）
     - Tweedie等式：∇log p_ε(x)=(D_ε(x)-x)/ε
     - 去噪=得分估计的等价性
     - 从Tweedie到PnP-ULA的计算链
5.4  MAP与MMSE的结构对偶性（Pock L2）
     - 近端算子=Moreau包络的一步梯度（MAP方向）
     - 去噪器=软下卷积的一步梯度（MMSE方向）
     - 温度参数：MAP=零温度的MMSE
5.5  PnP框架：用去噪器替换先验梯度
     - 近端算子prox_λR→去噪器D_ε
     - PnP-ULA后验采样与不确定性量化
     - PnP-ADMM优化视角
5.6  近似理论与收敛保证（原5.5，前移并扩展）
     - p_λ的性质：真密度、log-concave、C^1
     - 近似误差界‖p_λ-p‖_TV≤λL_g²
     - PnP-ULA收敛性（Laumont et al. 2021）
```

#### 第6章 得分匹配：从去噪中学习得分

```
6.1  为什么需要学习得分？
     - 得分函数不可直接计算（需要归一化常数Z）
     - 得分函数的关键优势：无需归一化常数
6.2  显式得分匹配（ESM）与隐式得分匹配（ISM）
     - ESM目标函数与不可行性
     - ISM：分部积分消去∇log p(x)
     - ISM的计算瓶颈：Jacobian迹
6.3  去噪得分匹配（DSM）
     - DSM目标函数推导
     - DSM=ESM+常数等价性（Vincent 2011）
     - 去噪=得分匹配的等价性（呼应Tweedie）
6.4  切片得分匹配（SSM）与Hutchinson迹估计
     - 避免Jacobian精确计算的随机估计
     - ESM/ISM/DSM/SSM四种方法关系与对比
6.5  多尺度得分匹配：从单一噪声到噪声条件网络
     - 低密度区域问题与流形假设
     - NCSN训练与噪声调度
     - 退火Langevin动力学
     - NCSN→扩散模型的桥梁
6.6  去噪器作为得分估计器：实践与架构
     - DRUNet条件噪声水平设计
     - 三种参数化：ε预测/s预测/x₀预测
     - Tweedie等式在实践中的角色
6.7  用学习到的得分驱动采样
     - 退火Langevin动力学（生成）
     - PnP-ULA（逆问题求解）
     - 学习先验 vs 手工先验对比
```

#### 第7章 扩散模型：SDE视角

```
7.1  从Langevin到扩散：连续时间推广
7.2  正向SDE：从数据到噪声的连续过程
7.3  逆向SDE：从噪声到数据的采样过程
7.4  概率流ODE：随机采样的确定性等价
7.5  数值离散化：从连续方程到可执行算法
     - Euler-Maruyama 方法
     - DDPM 作为 SDE 的离散特例
     - SMLD 作为 VE-SDE 的离散特例
     - 采样器选择：DDPM vs DDIM vs 概率流ODE
7.6  实践：用扩散SDE实现图像生成
附录7A Anderson逆向时间SDE定理的证明概要
附录7B VE-SDE与VP-SDE的推理等价性
```

#### 第8章 变分推断与ELBO

```
8.1  为什么需要变分推断？
     - 采样路径的成就与局限
     - 真实后验不可解的普遍性
     - 变分推断的核心思想：从采样到优化
     - 两条路径的分野与互补
8.2  ELBO推导：证据下界
     - Jensen不等式推导
     - KL散度分解：log p(x) = ELBO + KL(q‖p(z|x))
     - ELBO两种等价分解：联合-熵 / 重建+正则
     - ELBO最大化 = KL最小化
8.3  变分推断作为优化问题
     - 变分族q的选择与近似-效率权衡
     - 平均场近似（Mean-Field Approximation）
     - 坐标上升变分推断（CAVI）
     - 变分间隙
     - 变分推断 vs MCMC对比
8.4  变分推断与正则化的统一视角
     - ELBO = 重建项 + KL正则项
     - 回顾第2章：先验 = 正则化
     - 两层对应：点估计 vs 分布估计；确定性正则 vs 概率正则
     - Fenchel共轭与变分下界的优化根基（简述，详见附录8A）
     - 变分正则化框架
8.5  从变分推断到生成模型：路线图
     - 逐样本推断 vs 摊推断（Amortized Inference）
     - 生成模型的变分框架
     - 变分路径路线图：ELBO → VAE → 层级VAE → 变分扩散
附录8A Fenchel共轭与ELBO的优化理论根基
附录8B 平均场近似的闭式推导（CAVI算法推导）
附录8C KL散度的性质与变分推断的信息论视角
```

#### 第9章 VAE与重参数化

```
9.1  VAE架构：编码器-解码器
     - 识别模型q_φ(z|x)与生成模型p_θ(x|z)
9.2  重参数化技巧
     - 梯度穿过随机节点
     - REINFORCE vs 重参数化：方差对比
9.3  ELBO训练与KL正则化
     - 重建项+KL项的权衡
     - β-VAE与解纠缠
9.4  四种学习设定（Ratti P18-21）
     - 监督学习（配对数据）
     - 自监督（已知噪声模型的干净数据）
     - 无监督-x（Tweedie等式学习先验）
     - 无监督-y（仅有噪声数据）
9.5  过参数化与双重下降（Ratti P29-30，可选/进阶）
     - 经典U形偏差-方差曲线 vs 现代双重下降
```

#### 第10章 层级VAE与扩散的变分推导

```
10.0 本章导读：从一步编码到逐步加噪
10.1 从VAE到层级VAE
     - 单层VAE的局限
     - 层级潜变量z_1,...,z_L的引入
     - 马尔可夫推断链
     - 层级ELBO推导
10.2 扩散过程的变分下界推导
     - 高斯编码器=加噪过程
     - 噪声调度与直接采样
     - VLB三项KL分解
     - 前向过程后验q(x_{t-1}|x_t,x_0)的闭式解
10.3 从变分下界到去噪目标
     - KL散度项的化简
     - 逆向过程的参数化（x₀预测 vs ε预测）
     - 简化训练目标L_simple
     - 变分视角与得分视角的初步对应（→第12章）
10.4 层级VAE→扩散的极限
     - L→∞：从离散步到连续过程
     - 正向SDE：编码链的连续极限
     - 逆向SDE：解码链的连续极限
     - 统一视角：编码/解码↔加噪/去噪
附录10A 层级ELBO的完整推导
附录10B 前向过程后验的闭式推导
```

#### 第11章 扩散模型：变分视角

```
11.0 本章导读：从层级VAE的训练目标到扩散实践
11.1 VLB分解与正向过程后验
     - VLB三项分解：L_T + ΣL_{t-1} + L_0
     - 正向过程后验 q(x_{t-1}|x_t, x_0) 的闭式推导
     - 三项的物理意义与训练角色
11.2 一致性项化简：从KL散度到均值匹配
     - L_{t-1} = KL(q||p_θ) 的高斯闭式解
     - 均值匹配目标 ‖μ̃_t - μ_θ‖²
     - 从均值匹配到可训练目标
11.3 三种参数化：ε预测、得分预测与x₀预测
     - 噪声预测ε_θ参数化（DDPM）
     - x₀预测参数化（去噪参数化）
     - 得分预测s_θ参数化（SMLD/NCSN）
     - 三种参数化的数学等价性与训练稳定性差异
11.4 简化VLB与DDPM训练
     - Ho et al. 简化目标 L_simple
     - 丢弃时间权重的动机与效果
     - L_simple vs L_VLB 的实验对比
     - 学习方差：Improved DDPM (Nichol & Dhariwal 2021)
11.5 两条路径的交汇预告
     - VLB训练目标与DSM损失的结构相似性
     - ε预测VLB = 加权DSM
     - 预告第12章：Score ≡ ELBO的等价性证明
附录11A L_0项的离散解码器推导
附录11B 连续时间VLB与VDM (Kingma 2021)
```

#### 第12章 Score ≡ ELBO：殊途同归

```
12.1 采样路径回顾：Score Matching损失
12.2 变分路径回顾：变分下界
12.3 等价性证明
     - DSM损失≡VLB的形式化推导
     - 直观解释+形式化证明双写
12.4 实践意义
     - 同一扩散模型的两种训练视角
     - 训练目标选择指南
```

#### 第13章 条件生成与逆问题求解（全书高潮·闭环）

```
13.1 闭环：从逆问题出发，回到逆问题
     - 回顾全书推理链：逆问题 → 贝叶斯 → 采样 → 得分 → 扩散
     - 关键等式：条件扩散采样 p(x|y) = 逆问题求解
     - 扩散模型的独特优势：无需显式先验、无需归一化常数、零样本迁移

13.2 后验得分分解：条件化的理论基础
     13.2.1 条件逆向SDE的推导
       - 从无条件逆向SDE出发：dx = [f(x,t) - g(t)²∇log p_t(x)]dt + g(t)d𝑤̄
       - 引入条件y：dx = [f(x,t) - g(t)²∇log p_t(x|y)]dt + g(t)d𝑤̄_y
       - 条件逆向SDE与无条件逆向SDE的形式统一
     13.2.2 后验得分分解定理
       - ∇log p(x_t|y) = ∇log p(x_t) + ∇log p(y|x_t)
       - 第一项：无条件扩散模型提供（已学到的先验得分）
       - 第二项：似然得分——所有方法的差异在于如何近似这一项
       - 证明思路：贝叶斯定理在得分函数层面的应用
       - 来源：Chung et al. (2508.01975) 公式(3)
     13.2.3 似然得分 ∇log p(y|x_t) 的计算挑战
       - 为什么不可直接计算：p(y|x_t) = ∫ p(y|x₀)p(x₀|x_t)dx₀，积分不可解
       - 直观理解：这一项将观测y"拉回"到与测量一致的轨迹上
       - 与Tweedie等式的联系：x̂₀ = E[x₀|x_t] 通过Tweedie从得分函数提取
       - 形成闭环：得分→去噪→一致性梯度→修正得分

13.3 近似方法分类与DPS深度剖析
     13.3.1 第一类：显式近似（Laplace近似族）
       * Score-ALD：最简单的投影近似
       * DDRM：基于SVD分解的线性问题专用方法
       * DPS近似：p(y|x_t) ≈ p(y|x̂_{0|t})（Jensen近似/delta函数近似）
       * ΠGDM：用各向同性高斯替代delta函数
       * 近似精度递进：delta函数(DPS) → 各向同性高斯(ΠGDM) → 完整协方差(Moment Matching)
     13.3.2 DPS深度剖析
       * 核心思想：用Laplace近似 p(y|x₀)，在 x̂₀ = E[x₀|x_t] 处线性化
       * 近似公式：∇log p_t(y|x_t) ≈ ∇log p(y|x̂₀) · ∂x̂₀/∂x_t
       * DPS算法伪代码：
         for t in reversed(range(T)):
             1. 预测干净图像：x̂₀ = predict_x0(x_t, t)   # Tweedie或ε预测器反推
             2. 计算数据一致性梯度：∇_l = Aᵀ(y - Ax̂₀) / σ_y²
             3. 修正得分：corrected = s_θ(x_t,t) + ζ·∇_l   # ζ为缩放因子
             4. Euler-Maruyama步进：x_{t-1} = x_t + f·dt - g²·corrected·dt + g·√dt·z
       * 缩放因子ζ的作用：控制数据一致性强度，经验值ζ∈[0.1,1.0]
       * 优势：简单、通用、无需重新训练、支持非线性逆问题
       * 局限：Laplace近似在高噪声下不准确
       * 与Tweedie等式的闭环：得分→去噪估计x̂₀→一致性梯度→修正得分
     13.3.3 第二类：变分推断（RED-Diff, VIDS）
       * 用参数化分布近似后验 p(x_0|y)，优化ELBO
     13.3.4 第三类：隐空间优化/CSGM类（DiffPIR, DDS, DMPlug）
       * 通过反向传播优化初始噪声z
       * Gutha et al. MAP-GA：MAP估计视角 + 一致性模型重参数化
     13.3.5 第四类：渐近精确方法（MCMC/SMC, DreamSampler）
       * 目标是从真实后验采样，计算代价高
     13.3.6 四类方法的对比与选择指南

13.4 引导采样：从分类器引导到逆问题引导
     13.4.1 Classifier Guidance与DPS的数学统一与关键差异
       - 数学结构相似：均基于条件得分分解 s = s_uncond + ∇log p(condition|x)
       - 关键差异：
         * Classifier Guidance：∇log p(c|x_t)，需额外训练噪声分类器，用于类别/文本条件生成
         * DPS：∇log p(y|x_t)，利用Tweedie估计+测量模型，无需额外训练，用于逆问题求解
       - DPS是classifier guidance在逆问题领域的自然延伸
       - 引导强度对比：Classifier Guidance的w ↔ DPS的ζ
     13.4.2 Classifier-Free Guidance (CFG)
       - CFG核心公式：s_θ(x,y) = s_θ(x) + w·(s_θ(x,y) - s_θ(x))
       - 逆问题中的CFG前沿：训练时随机丢弃测量y，推理时用引导权重控制一致性
       - 优势：避免显式计算∇log p(y|x)，更稳定
     13.4.3 引导权重w与质量-多样性权衡

13.5 扩散最优控制（进阶）
     - 控制论视角：逆向扩散过程作为最优控制问题
     - 与DPS的联系：DPS可视为最优控制的近似解
     - 来源：NeurIPS 2024 "Solving Inverse Problems via Diffusion Optimal Control"

13.6 闭环：回到第1章的逆问题
     - 从贝叶斯框架到条件扩散的完整路径
     - 扩散模型为何能超越传统方法：任意复杂先验、不确定性量化、零样本迁移
     - 全书核心论点的落地：diffusion是逆问题自然发展的终点
```

#### 第14章 Flow Matching与最优传输

```
14.1 最优传输基础
     - Monge问题与Kantorovich松弛
     - Wasserstein距离
14.2 连续归一化流(CNF)
     - 向量场与流ODE
14.3 Flow Matching
     - 条件Flow Matching目标
     - OT-CFM：最优传输条件流
14.4 Rectified Flow
     - 直线插值 vs 扩散路径
     - 与扩散模型的对比
14.5 前沿组合：SD3/Flux = Rectified Flow + DiT
```

#### 第15章 扩散模型的架构实践：UNet → DiT

```
15.1  去噪器架构演进：CNN → UNet → DiT
15.2  DiT的关键设计：Patchify、adaLN-Zero
15.3  扩散 + Transformer的SOTA组合
      （SD3/Flux = Rectified Flow + DiT）
```

#### 第16章 CT/MRI重建

```
16.1 CT重建基础
     - Beer-Lambert定律（X射线衰减物理）
     - Radon变换与sinogram
     - Fourier切片定理与滤波反投影(FBP)
16.2 不适定性与有限角CT
     - 奇异值衰减：全角~1/n vs 有限角指数衰减
     - 波前集理论：为什么有限角丢失信息
16.3 MRI重建基础
     - k-space采样与傅里叶算子
     - 欠采样掩码与零填充重建
     - 压缩感知MRI基础
16.4 学习型重建方法
     - UNet端到端重建
     - Learned Gradient Descent迭代重建
     - 学习MRI采样模式（Benning L2）
16.5 扩散先验重建
     - DiffPIR for CT/MRI
```

#### 第17章 自监督学习与等变架构

```
17.1 为什么自监督？
     - 无需干净数据的训练动机
17.2 自监督去噪方法族
     - Noise2Self / Noise2Void（盲点网络）
     - Noise2Noise（配对噪声数据）
     - SURE（Stein无偏风险估计）
     - R2R（Recorrupted-to-Recorrupted）
17.3 等变架构
     - 等变性的数学定义
     - 测量一致性损失
     - 物理约束融入网络
17.4 四种学习设定的统一视角（回顾Ratti P18-21）
```

#### 第18章 综合项目：用扩散模型求解自定义逆问题

```
18.1 定义自定义前向算子
     - deepinv Physics类设计
     - 伴随算子验证
18.2 扩散模型求解自定义逆问题
     - 全书知识整合流程
18.3 不确定性量化
     - 多次后验采样→像素级置信区间
18.4 拓展方向
     - 赫尔辛基断层成像挑战赛（有限角CT竞赛）
     - 自定义逆问题的扩散求解
```

---

## 八、材料覆盖评估

### 各 Part 覆盖率

```
Part I   █████████████████░░░  85%  贝叶斯基石充足，部分优化/决策理论需补
Part II  ██████████████░░░░░░  70%  ULA/Langevin充足，得分匹配与扩散SDE理论缺口大
Part III █████░░░░░░░░░░░░░░░  25%  仅有周边基础，核心推导几乎全部缺失
Part IV  ██████░░░░░░░░░░░░░░  30%  两条路径端点有材料，统一论证需新写
Part V   █░░░░░░░░░░░░░░░░░░░   5%  Flow Matching无任何源材料
Part VI  ███████████████░░░░░  75%  实践素材充足，MRI与DiT为缺口
```

### 逐章主题级覆盖详情

> 状态说明：✅ 有直接源材料 | 🟡 有间接/部分材料 | ❌ 无材料需新写

#### 第1章 逆问题与贝叶斯推断 — 覆盖率 92%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| 逆问题定义与Hadamard适定性 | Pereyra L1 P1-5; Benning L1 P1-30 | ✅ |
| 正向模型y=Ax（卷积/下采样/Radon） | Calatroni P1-13; Siltanen D2 P28-70; MiniProject_DefiningOperator | ✅ |
| Beer-Lambert定律 | Siltanen D2 P55-70 | ✅ |
| 高斯噪声→L2数据项 | Calatroni P14-17; Pereyra L1 P8 | ✅ |
| Poisson噪声→KL散度数据项 | Calatroni P36-37 | ✅ |
| 脉冲/Laplace噪声→L1数据项 | Calatroni P37-38 | ✅ |
| 似然函数推导 | Pereyra L1 P8; Calatroni P26-28 | ✅ |
| 不适定性与病态性 | Benning L1 P1-30; Pereyra L1 P5 | ✅ |
| inverse crime | Siltanen D2 (XR02代码) | 🟡 代码有,理论解释少 |
| 贝叶斯定理p(x\|y)∝p(y\|x)p(x) | Pereyra L1 P7-10; Calatroni P26-28 | ✅ |
| 图像质量度量(MSE/PSNR/SSIM) | Calatroni P18-24; CompImLab25 Part 1 | ✅ |

#### 第2章 先验：贝叶斯推断的灵魂 — 覆盖率 83%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| -ln后验=数据项+正则项（完整对应表） | Calatroni P29-44; Pock L2 P7-13 | ✅ |
| 高斯先验→Tikhonov | Calatroni P29-34; Benning L1 P75-83 | ✅ |
| Laplace先验→L1/稀疏 | Calatroni P35; Benning L1 P138-139 | ✅ |
| 梯度高斯先验→Sobolev平滑 | Calatroni P43 | ✅ |
| TV先验→全变差 | Siltanen D2; Pock L3 P1-9; Benning L1 P139+ | ✅ |
| TGV先验 | Siltanen D2 P11; Pock L3 (简要) | 🟡 仅提及,无理论 |
| Fields of Experts (FoE)先验 | Pock L1 P6-20 | ✅ |
| 最大熵先验 | Pock L2 P7-8 | ✅ |
| MMSE vs MAP估计器 | Ratti P10-13; Pock L2 P7-13 | ✅ |
| 五类误差分解 | Ratti P49-57 | ✅ |
| 从显式先验到隐式先验 | Pock L1 P20-56; Pereyra L3 P36-52 | ✅ |
| 贝叶斯去噪器 | Ratti P1-13 | ✅ |

#### 第3章 从MAP到后验探索 — 覆盖率 88%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| MAP估计与贝叶斯决策理论 | Pereyra L1 P12, P24-25 | ✅ |
| 凸性/Lipschitz/强制性 | Calatroni P49-51; opt1 P19-40 | ✅ |
| 梯度下降与步长选择 | opt1 P41-55; Calatroni P51-53 | ✅ |
| Tikhonov迭代+闭式解(DFT) | Calatroni P53; Benning L1 P75-83; XR05 | ✅ |
| ISTA/FISTA | opt2 P21+; PHD_MIVA lab | ✅ |
| 迭代硬阈值(IHT) | opt3 P20-22 | ✅ |
| 重加权L1 | opt3 P28+ | ✅ |
| TV正则化与Chambolle-Pock | XR09_TV; tomo_tv; Siltanen D2 P46-48 | ✅ |
| ADMM算法 | Gondzio L6 | 🟡 仅提及,缺完整推导 |
| Bregman距离与误差估计 | Benning L2 P68-71; Benning L1 P170+ | ✅ |
| 源条件与收敛速率 | Benning L2 P10-20, P33-38 | ✅ |
| 近端算子定义与计算 | opt2 P9-20; Calatroni P41-44; proximal.m | ✅ |
| 截断SVD正则化 | Benning L1 P40-70; BunnyTomo3 | ✅ |
| 经验贝叶斯参数估计 | Pereyra L1 P58-84; Unit2_exercise | ✅ |
| 从MAP到后验的动机 | Pereyra L1 P30+; Pock L2 P18-24 | ✅ |

#### 第4章 MCMC与ULA算法 — 覆盖率 91%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| Monte Carlo积分 | Pereyra L1 P30-35 | ✅ |
| Metropolis-Hastings算法 | Pereyra L1 P35-40 | ✅ |
| ULA递推式与Euler离散化 | Pereyra L1 P40-50; Pock L2 P14-17; lab1_ULA_sol | ✅ |
| 步长δ≤1/L条件 | Pereyra L1 P45-48; lab1_ULA_sol | ✅ |
| MH vs ULA对比 | Pereyra L1 P48-50; Pock L2 P15-17 | ✅ |
| MYULA（Moreau包络） | Pereyra L1 P43-44; Pereyra L3 P9-11 | ✅ |
| Gibbs采样 | Pock L2 P27-31 | ✅ |
| 过松弛/惯性Langevin | Pock L2 P32-33 | ✅ |
| 半二次最小化→GLM | Pock L2 P25-26 | ✅ |
| MCMC收敛诊断 | lab1_ULA_sol (部分) | 🟡 实践有,理论少 |

#### 第5章 朗之万动力学与得分函数 — 覆盖率 100%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| 从MCMC到Langevin SDE | Pereyra L1 P40-50; Pock L2 P14 | ✅ |
| Langevin方程dx=∇log p dt+√2dW | Pereyra L1 P45-48; Pock L2 P14 | ✅ |
| 得分函数几何含义 | Pereyra L1 P58-60 | ✅ |
| Tweedie等式 | Pock L2 P10-13; Pereyra L3 P12-21; Ratti P9-13 | ✅ |
| MAP/MMSE结构对偶(Moreau vs 软下卷积) | Pock L2 P7-13, P18-20 | ✅ |
| PnP框架 | Pereyra L3 P9-11; Calatroni P41-44; lab2_PnP_sol | ✅ |
| PnP-ULA后验采样与不确定性量化 | Pereyra L3 P22-35; lab2_PnP_sol | ✅ |
| Moreau-Yoshida近似理论 | Pereyra L1 P43-44; Pereyra L3 P9-11 | ✅ |

#### 第6章 得分匹配：从去噪中学习得分 — 覆盖率 75%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| 得分函数不可计算（需Z） | Pereyra L1 P58-60 | ✅ |
| 得分函数关键优势：无需Z | 第5章5.2节 | ✅ |
| ESM目标函数与不可行性 | Tutorial_Diffusion Sec 3.3; Hyvärinen (2005) | ✅ |
| ISM分部积分与Jacobian迹 | Tutorial_Diffusion Sec 3.3; Hyvärinen (2005) | ✅ |
| DSM目标函数推导 | Tutorial_Diffusion Sec 3.3; Vincent (2011) | ✅ |
| DSM=ESM+常数等价性证明 | Tutorial_Diffusion Theorem 3.4; Vincent (2011) | ✅ |
| 去噪=得分匹配等价性 | 第5章Tweedie等式; Tutorial_Diffusion | ✅ |
| 三种参数化（ε/s/x₀预测） | Tutorial_Diffusion; 2508.01975v1 | ✅ |
| SSM与Hutchinson迹估计 | Song et al. (2019) SSM; Hutchinson (1990) | ✅ |
| ESM/ISM/DSM/SSM关系 | Tutorial_Diffusion Sec 3.3 | ✅ |
| 多尺度得分匹配(NCSN) | Song & Ermon (2019); Tutorial_Diffusion | ✅ |
| 低密度区域与流形假设 | Song & Ermon (2019) | ✅ |
| 退火Langevin动力学 | Song & Ermon (2019) | ✅ |
| DRUNet架构 | Zhang et al. (2021); MiniProject_DenoisingPrior | ✅ |
| Tweedie等式实践连接 | Pock L2 P10-13; 第5章5.3节 | ✅ |
| PnP-ULA实验 | Pereyra L3 P22-35; lab2_PnP_sol | ✅ |
| 学习先验vs手工先验对比 | lab2_PnP_sol; MiniProject_DenoisingPrior | ✅ |

#### 第7章 扩散模型：SDE视角 — 覆盖率 40%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| 从Langevin到扩散：连续时间推广 | Pock L2 P14 (概念); Pereyra L1 P50 | 🟡 概念有,完整推广缺 |
| 正向SDE与逆向SDE | Pereyra L1 P50-55; Pereyra L3 P7-8 | 🟡 有基础但缺完整推导 |
| 概率流ODE | — | ❌ |
| Euler-Maruyama离散化 | lab1_ULA_sol (ULA类似) | 🟡 ULA有,扩散离散化缺 |
| DDPM作为SDE离散特例 | — | ❌ |
| 采样器选择：DDPM vs DDIM vs ODE | — | ❌ |
| "扩散在绝对零度"概念 | Pock L2 P14 (温度→0) | ✅ |
| deepinv扩散SDE demo | demo_diffusion_sde | ✅ |

#### 第8章 变分推断与ELBO — 覆盖率 29%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| 为什么需要变分推断？ | Pereyra L1 P6 (隐含) | 🟡 动机弱 |
| ELBO推导：Jensen不等式+KL散度 | — | ❌ |
| log p(x) = ELBO + KL(q\|p) | — | ❌ |
| Fenchel共轭与变分下界 | Benning L2 P23-25; opt3 | 🟡 凸共轭有,ELBO连接缺 |
| 变分正则化框架 | Benning L1 P130-175 | ✅ |
| 变分族选择：平均场近似 | — | ❌ |
| 变分间隙 | — | ❌ |

#### 第9章 VAE与重参数化 — 覆盖率 44%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| VAE架构：编码器-解码器 | — | ❌ |
| 重参数化技巧 | — | ❌ |
| REINFORCE vs 重参数化 | — | ❌ |
| ELBO训练与KL正则化 | — | ❌ |
| β-VAE与解纠缠 | — | ❌ |
| 四种学习设定 | Ratti P18-21 | ✅ |
| 神经网络架构(MLP/CNN/UNet/ViT) | Ratti P31-48 | ✅ |
| 通用近似定理 | Ratti P33 | ✅ |
| 偏差-方差与双重下降 | Ratti P26-30 | ✅ |

#### 第10章 层级VAE与扩散的变分推导 — 覆盖率 85%（已写完，原14%）

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| 从VAE到层级VAE | Tutorial_Diffusion Sec 1-2; Kingma & Welling (2014) | ✅ |
| 马尔可夫推断链 | Tutorial_Diffusion Sec 2.1; 2406.08929v2 Sec 5 | ✅ |
| 层级ELBO推导 | Tutorial_Diffusion Theorem 2.3; DDPM Appendix A | ✅ |
| 高斯编码器=加噪过程 | Tutorial_Diffusion Sec 2.1; 2508.01975v1 Sec 2.2 | ✅ |
| 噪声调度与直接采样 | DDPM (Ho et al. 2020); Tutorial_Diffusion | ✅ |
| VLB三项分解 | Tutorial_Diffusion Theorem 2.3-2.4; DDPM Eq.5 | ✅ |
| 前向后验闭式解 | Tutorial_Diffusion Theorem 2.5; DDPM Appendix B | ✅ |
| x₀预测参数化 | Tutorial_Diffusion Sec 2.4; Kingma et al. (2021) | ✅ |
| ε预测参数化 | Tutorial_Diffusion Sec 2.5; DDPM | ✅ |
| 简化训练目标L_simple | DDPM (Ho et al. 2020) | ✅ |
| 变分与得分对应 | 2406.08929v2 Sec 5; 第6章DSM | ✅ |
| L→∞连续极限 | Tutorial_Diffusion Sec 4; Song et al. (2021) | 🟡 概念有，严格推导待补 |
| VP-SDE连续极限推导 | Song et al. (2021) Score-SDE | 🟡 需从SDE视角补充细节 |
| 编码/解码↔加噪/去噪对应表 | 2406.08929v2; VDM论文概念 | ✅ |
| 贝叶斯反转推导 | DDPM Appendix A | 🟡 待补充到附录 |
| 图示与可视化 | — | ❌ 缺少图示 |

#### 第11章 扩散模型：变分视角 — 覆盖率 0%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| VLB分解：L_T + L_{t-1} + L_0 | — | ❌ |
| 三种参数化：ε_θ vs s_θ vs x₀预测 | — | ❌ |
| 参数化对训练稳定性的影响 | — | ❌ |
| 简化VLB与DDPM训练 | — | ❌ |
| DDPM训练目标作为VLB简化 | — | ❌ |

#### 第12章 Score ≡ ELBO：殊途同归 — 覆盖率 17%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| 采样路径回顾：Score Matching损失 | 第6章Tweedie identity | 🟡 有Tweedie缺DSM |
| 变分路径回顾：变分下界 | 第8-11章(均缺失) | ❌ |
| DSM损失≡VLB等价性证明 | — | ❌ |
| 直观解释+形式化证明 | — | ❌ |
| 两种训练视角对比 | — | ❌ |
| 训练目标选择指南 | — | ❌ |

#### 第13章 条件生成与逆问题求解 — 覆盖率 70%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| 闭环叙事：逆问题→扩散→回到逆问题 | 第1-12章完整链 | ✅ |
| 条件逆向SDE推导 | Chung et al. (2508.01975) §2 | ✅ |
| 后验得分分解 ∇log p(x_t\|y) = ∇log p(x_t) + ∇log p(y\|x_t) | Chung et al. (2508.01975) 公式(3) | ✅ |
| 似然得分∇log p(y\|x_t)的计算挑战与Tweedie闭环 | Chung et al. (2508.01975) Theorem 1 | ✅ |
| DPS深度剖析：Laplace近似推导+算法伪代码 | Chung et al. (2508.01975) §3.2 | ✅ |
| DPS缩放因子ζ与实践技巧 | Chung et al. (2508.01975) §3.2 | ✅ |
| DPS近似链（delta→高斯→完整协方差） | Chung et al. (2508.01975) §3.2 | ✅ |
| 四类方法分类（显式近似/变分/CSGM/渐近精确） | Daras et al. Survey (2410.00083) | ✅ |
| MAP-GA算法（MAP估计+一致性模型） | Gutha et al. WACV 2025 | ✅ |
| Classifier Guidance与DPS的统一与差异 | diffusion-tutorials 06-classifier-guidance.ipynb + Chung et al. | 🟡 |
| Classifier-Free Guidance | diffusion-tutorials 07-classifier-free-guidance.ipynb | 🟡 |
| 引导权重w与质量-多样性权衡 | — | ❌ |
| 扩散最优控制视角 | NeurIPS 2024 | ✅ |
| DDRM算法 | deepinv demo_ddrm | ✅ |
| DiffPIR算法 | deepinv demo_diffpir | ✅ |
| DPS算法实现（建议新增demo_dps） | deepinv（待实现） | ❌ |
| 贝叶斯假设检验 | Pereyra L1 P30, P51-53 | 🟡 |

#### 第14章 Flow Matching与最优传输 — 覆盖率 0%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| 最优传输基础(Monge/Kantorovich) | — | ❌ |
| Wasserstein距离 | — | ❌ |
| 连续归一化流(CNF) | — | ❌ |
| Flow Matching / 条件Flow Matching | — | ❌ |
| OT-CFM | — | ❌ |
| Rectified Flow | — | ❌ |
| 与扩散模型对比 | — | ❌ |
| SD3/Flux = Rectified Flow + DiT | — | ❌ |

#### 第15章 扩散模型的架构实践：UNet → DiT — 覆盖率 62%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| UNet架构(编码器-解码器+skip) | Ratti P36-37; Bologna_UNet_example | ✅ |
| UNet去噪器训练 | CompImLab25 Part 3 | ✅ |
| UNet端到端CT重建 | Bologna_UNet_example | ✅ |
| ViT概念(单页) | Ratti P38 | 🟡 仅概念,无细节 |
| Restormer(仅提及) | Ratti P38 | 🟡 仅名称 |
| Transformer架构细节(MHA/MLP块) | — | ❌ |
| DiT架构(Patchify/adaLN-Zero) | — | ❌ |
| 时间步嵌入机制 | — | ❌ |
| 训练最佳实践 | Ratti P44-48 | ✅ |
| 反向传播与SGD | Ratti P40-43 | ✅ |

#### 第16章 CT/MRI重建 — 覆盖率 75%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| Beer-Lambert定律 | Siltanen D2 P37-47 | ✅ |
| 像素化测量模型(矩阵A) | Siltanen D2 P48-97 | ✅ |
| 反投影(A^T) | Siltanen D2 P98-103 | ✅ |
| Fourier切片定理(含证明) | Siltanen D3A P51 | ✅ |
| 滤波反投影(FBP) | Siltanen D2 P5-7 | ✅ |
| 有限角CT与不适定性 | Siltanen D3A P28-54 | ✅ |
| 波前集理论(可见vs不可见边缘) | Siltanen D3A P55-58 | ✅ |
| 剪切波学习不可见边缘 | Siltanen D3A P63 | ✅ |
| CT正则化(Tikhonov/TV/小波/剪切波) | Siltanen D2 P125-162 | ✅ |
| 发射断层(PET/PGET) | Siltanen D3B P1-27 | ✅ |
| Helsinki挑战赛 | Siltanen D3A P70-80 | ✅ |
| ASTRA工具箱(代码) | astra_operators_example | ✅ |
| UNet端到端CT(代码) | Bologna_UNet_example | ✅ |
| Learned Gradient Descent(代码) | Bologna_LGS_example | ✅ |
| **MRI k-space采样** | — | ❌ |
| **MRI正向模型(傅里叶算子)** | — | ❌ |
| **MRI欠采样掩码** | — | ❌ |
| **压缩感知MRI** | — | ❌ |
| **学习MRI采样模式** | Benning L2(提及,无源文件) | ❌ |
| 扩散先验CT重建 | MiniProject_DenoisingPrior(去模糊) | 🟡 需迁移到CT算子 |

#### 第17章 自监督学习与等变架构 — 覆盖率 80%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| 自监督学习动机 | Tachella P1-9 | ✅ |
| SURE(Stein无偏风险估计) | Tachella P12-13 | ✅ |
| UNSURE(未知噪声水平) | Tachella P17-19 | ✅ |
| Noise2Void/Noise2Self | Tachella P15-16 | ✅ |
| Recorrupted2Recorrupted(R2R) | Tachella P13, P41 | ✅ |
| 等变成像(EI)原理 | Tachella P28-30 | ✅ |
| EI损失与伪代码 | Tachella P31 | ✅ |
| 算子-等变性表(平移/旋转/CT/MRI) | Tachella P29 | ✅ |
| 四种学习设定分类 | Ratti P18-21 | ✅ |
| Cryo-EM自监督去噪(代码) | MiniProject_Self_Supervised | ✅ |
| deepinv自监督损失API | MiniProject_Self_Supervised Step3 | ✅ |
| Noise2Noise专门处理 | Ratti P37仅提及 | 🟡 无专门材料 |
| EI代码实现 | MiniProject "Going Further"仅提及 | 🟡 无代码 |

#### 第18章 综合项目 — 覆盖率 65%

| 子主题 | 可用来源 | 状态 |
|---|---|---|
| 自定义前向算子(MultiViewPhysics) | MiniProject_DefiningOperator | ✅ |
| 伴随算子自动计算与验证 | MiniProject_DefiningOperator | ✅ |
| 扩散模型求解逆问题(去模糊) | MiniProject_DenoisingPrior | ✅ |
| PnP vs 扩散对比 | MiniProject_DenoisingPrior | ✅ |
| 不确定性量化方法 | lab1_ULA_sol + 扩散采样 | 🟡 方法有,需组合 |
| 端到端自定义逆问题流程 | — | ❌ 需设计整合 |

### 缺失内容汇总

#### 完全缺失（需从零写的核心内容）

| 章 | 缺失核心内容 | 建议参考文献 |
|---|---|---|
| 第6章 | ESM/DSM/SSM完整推导与训练代码 | Hyvärinen 2005; Vincent 2011 |
| 第7章 | 正向/逆向SDE推导；概率流ODE；DDPM=SDE离散特例；采样器理论 | Song et al. 2021 (Score-SDE) |
| 第8章 | ELBO推导；Jensen→KL分解；平均场近似；变分间隙 | Bishop PRML Ch10; Blei et al. 2017 |
| 第9章 | VAE架构；重参数化技巧；ELBO训练；β-VAE | Kingma & Welling 2014; Rezende et al. 2014 |
| 第10章 | 层级VAE全链；层级ELBO推导；高斯编码器=加噪 | Kingma et al. 2021 (VDM); Sønderby 2016 |
| 第11章 | VLB分解；三种参数化；简化VLB | Ho et al. 2020 (DDPM); Kingma 2021 |
| 第12章 | DSM≡VLB等价性证明 | Luo 2022综述; Song & Kingma |
| 第14章 | 全部（OT/Wasserstein/CNF/FM/RF） | Lipman et al. 2023; Liu et al. 2023 |
| 第16章 | MRI全部（k-space/欠采样/压缩感知） | Lustig et al. 2008; deepinv MRI |

#### 部分缺失（需扩展的关键内容）

| 章 | 缺失内容 | 可扩展的基座 |
|---|---|---|
| 第1章 | inverse crime理论解释 | XR02代码已有实践 |
| 第3章 | ADMM完整推导 | Gondzio L6有提及 |
| 第4章 | MCMC收敛诊断理论 | lab1_ULA_sol有实践 |
| 第6章 | DSM理论推导（从Tweedie推出） | Pock L2有Tweedie |
| 第7章 | Langevin→扩散SDE完整推广 | Pock L2有概念 |
| 第13章 | 条件扩散/引导采样理论 | MiniProject有DDRM/DiffPIR代码 |
| 第15章 | Transformer/DiT架构细节 | Ratti P38有ViT概念 |
| 第16章 | 扩散先验迁移到CT | MiniProject有去模糊版本 |
| 第17章 | Noise2Noise专门材料; EI代码 | Tachella有理论 |
| 第18章 | 端到端整合流程设计 | 两个MiniProject可组合 |

### 资源清单（按类型）

#### PDF讲座（已转为.md，共35个）

> 新增补充论文见下方"补充论文"小节。

| 来源 | 文件 | 页数 | 关键内容 |
|---|---|---|---|
| Unit 1 | Pre_course_I_Calatroni | 59 | 正向模型/噪声/贝叶斯推导/正则化=先验/近端=PnP前传 |
| Unit 1 | Pre_course_II_Ratti | 59 | 回归视角/监督自监督/NN架构(CNN/UNet/**ViT**)/训练/学习误差分解 |
| Unit 2 | PEREYRA - Lectures 1 & 2 | 84 | 贝叶斯推断/Monte Carlo/**stochastic diffusion processes** |
| Unit 2 | PEREYRA - Lecture 3 | 53 | 扩散过程/不确定性量化/模型选择 |
| Unit 3 | lecture_1 | 82 | 学习先验工具箱/**explicit diffusion models**/Gaussian mixture priors |
| Unit 3 | lecture_2 | 47 | 最大熵先验/生成先验 |
| Unit 3 | lecture_3 | 54 | 先验学习深度内容 |
| Unit 5 | Benning - Lecture 1 | 270 | 正则化理论（大型讲座） |
| Unit 5 | Benning - Lecture 2 | 71 | 正则化→机器学习过渡 |
| Unit 5 | BENNING - Lecture 3 | 97 | 机器学习融合 |
| Winter School | Siltanen Day1 | 87 | 断层成像基础/Radon变换 |
| Winter School | Siltanen Day1 exercises | 75 | 断层成像练习 |
| Winter School | Siltanen Day2 | 174 | 正则化方法/Tikhonov/TV |
| Winter School | Siltanen Day3A | 82 | 高级重建/Besov小波 |
| Winter School | Siltanen Day3B | 56 | 总结/应用 |
| Winter School | MIVAcourse_opt1 | 65 | 凸分析基础/梯度下降/近端梯度 |
| Winter School | MIVAcourse_opt2 | 71 | 加速策略/Nesterov/FISTA |
| Winter School | MIVAcourse_opt3 | 57 | 稀疏优化/l0/l1/ISTA |
| Winter School | JGondzio-Lectures1and2 | 46 | 内点法/LP |
| Winter School | JGondzio-Lectures3and4 | 64 | QP/SOCP/SDP/ADMM |
| Winter School | JGondzio-Lecture5 | 40 | 稀疏近似/ADMM |
| Winter School | JGondzio-Lecture6 | 30 | 大规模优化 |
| Seminar | Tachella | 43 | 逆问题/自监督/等变架构 |
| Seminar | DiStefano | 43 | 应用报告 |
| Seminar | Toschi_Franchini | 49 | 应用报告 |

#### .m代码（42个）— Winter School 断层成像与优化

| 文件 | 功能 | 对应章节 |
|---|---|---|
| XR01_matrix_comp | 构建Radon测量矩阵A | 第1章(正向模型) |
| XR02_data_comp | 生成含噪sinogram(含/不含inverse crime) | 第1章(噪声建模) |
| XR03_naive_comp | 朴素逆(A\m，无正则化) | 第1章(不适定性) |
| XR04_SVD_comp | 计算A的SVD分解 | 第1章(奇异性) |
| BunnyTomo2_SVD_comp | 小尺寸Bunny phantom SVD | 第1章 |
| BunnyTomo3_truncSVD_comp | 截断SVD正则化重建 | 第3章(正则化) |
| XR05_Tikhonov_comp | Tikhonov正则化(共轭梯度) | 第3章 |
| XR09_TV_comp | 全变差重建(原始-对偶) | 第3章(TV) |
| XR10_B111_comp | Besov B111正则化(ISTA+小波软阈值) | 第3章(稀疏先验) |
| tomo_tv | TV原始-对偶算法核心 | 第3章(优化) |
| proximal | 近端算子求解 | 第5-6章(近端=PnP前身) |
| Smu / Smu_wavelet_oper | 软阈值函数 | 第3章(稀疏) |
| wavetrans2D / wavetrans2D_inv | 2D小波正/逆变换 | 第3章(多尺度) |
| Complex_wavelet_wavefront | 复小波与波前集恢复 | 第1章 |
| dxm/dxp/dym/dyp + _ad | 差分算子及伴随 | 第3章(TV梯度) |
| tomosynthesis_demo | 有限角断层成像 | 第16章(CT重建) |
| tomo2x2_TV_comp_bruteforce | 2×2 TV暴力搜索(教学) | 第3章 |
| tomo2x2_TV_comp_quadprog | 2×2 TV二次规划(教学) | 第3章 |

#### 补充论文（from inverse problem to diffusion文件夹）

| 文件 | 主题 | 对应章节 | 与book_plan的关系 |
|---|---|---|---|
| Daras et al. Survey (2410.00083) | 扩散逆问题方法全面综述，四类分类框架 | 第13章 | **核心参考**：四类方法分类直接用于第13章结构 |
| Chung et al. (2508.01975) | 扩散逆问题书章，Tweedie/后验得分分解/DPS近似链 | 第5-7章、第13章 | **核心参考**：数学推导最详细，Tweedie和DPS链 |
| Gutha et al. WACV 2025 (2407.20784) | MAP估计视角+一致性模型重参数化 | 第13章 | MAP-GA算法纳入第13章第三类方法 |
| NeurIPS 2024 最优控制 | 扩散最优控制视角求解逆问题 | 第13章 | 控制论视角，第13章进阶内容 |
| NeurIPS 2023 图逆问题 | 图上源定位的扩散模型 | 第13章 | 特定领域，可选提及 |
| Jin & Rundell 2015 | 物理反常扩散（分数阶PDE）的逆问题 | — | ⚠️ 此文"扩散"指物理扩散，与生成式扩散无关，不建议纳入正文 |
| Sato report | 综述报告 | 待确认 | 待评估 |

> ⚠️ **关于Jin & Rundell 2015**：此文的"扩散"指物理中的反常扩散过程（分数阶微分方程），与本书的生成式扩散模型是完全不同的概念。不建议纳入正文，以免造成术语混淆。

#### .ipynb代码（13个）

| 文件 | 功能 | 对应章节 |
|---|---|---|
| CompImLab25 | 图像处理+变分去噪+训练UNet去噪器 | 第1-3章、第15章 |
| MiniProject_DefiningOperator | deepinv多视角线性算子 | 第1章、第18章 |
| MiniProject_DenoisingPrior | **PnP vs 扩散模型对比去模糊** | 第6-7章、第13章 |
| MiniProject_Self_Supervised | Cryo-EM自监督去噪 | 第17章 |
| lab1_ULA + lab1_ULA_sol | ULA算法理论+**完整解答** | 第4章 |
| lab2_PnP + lab2_PnP_sol | PnP先验+Tweedie等式+**完整解答** | 第5-6章 |
| astra_operators_example | ASTRA断层成像算子 | 第16章 |
| Bologna_UNet_example | UNet断层成像重建 | 第15-16章 |
| Bologna_LGS_example | Learned Gradient Descent断层成像 | 第16章 |
| Unit2_exercise | 贝叶斯推断+经验贝叶斯+近端梯度+MAP | 第2-3章、第5章 |
| PHD_MIVA_winter_school_lab | ISTA/FISTA稀疏图像重建 | 第3章 |

---

## 九、三大核心缺口（需从零写）

| 缺口 | 范围 | 性质 | 建议来源 |
|---|---|---|---|
| VAE → 层级VAE → 变分扩散 | 第8-11章 | 有贝叶斯+变分+生成先验铺垫，需补中间推导链 | Kingma & Welling原始论文；Kingma VDM论文；Bishop PRML Ch10 |
| Score ≡ VLB 等价性证明 | 第12章 | 两条路径材料已有，需"缝合" | Song & Kingma统一视角；Luo 2022综述 |
| Flow Matching | 第14章 | 完全新内容 | Lipman et al. Flow Matching论文；Liu et al. Rectified Flow |

### 建议补充资源

| 缺失内容 | 对应章节 | 建议来源 |
|---|---|---|
| DSM/SSM训练实践 | 第6章 | 从现有PnP lab扩展，加一个score matching训练notebook |
| 条件扩散/引导采样 | 第13章 | Dhariwal & Nichol (classifier guidance)；Ho & Salimans (classifier-free)；**Daras et al. Survey提供方法分类框架**；**Chung et al.提供DPS近似链推导** |
| DiT架构 | 第15章 | Peebles & Xie DiT论文 + 官方代码 |
| 扩散模型SDE完整理论 | 第7章 | Song et al. Score-Based SDE论文 |

---

## 十、各章实战练习规划

> 标注说明：✅ 有现成素材可直接用 | 🔄 有素材需改造/扩展 | 🆕 需从零新写

### Part I 贝叶斯基石

#### 第1章 逆问题与贝叶斯推断

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验1.1 | 数字图像读取、显示与质量评估（MSE/PSNR/SSIM） | 数字图像表示；质量度量 | CompImLab25 Part 1 | ✅ |
| 实验1.2 | 构建正向模型：卷积模糊+下采样+掩码 | 线性正向模型y=Ax；常见退化算子 | CompImLab25 + MiniProject_DefiningOperator | ✅ |
| 实验1.3 | 噪声建模：高斯/Poisson/脉冲噪声模拟 | 噪声概率模型；似然函数 | CompImLab25 + Calatroni P14-17 | ✅ |
| 实验1.4 | 构建Radon测量矩阵与sinogram生成 | Radon变换；断层成像正向模型 | XR01_matrix_comp + XR02_data_comp（Python重写） | 🔄 .m→Python |
| 实验1.5 | 朴素逆重建与不适定性观察 | 不适定性；逆问题的病态性；inverse crime | XR03_naive_comp（Python重写） | 🔄 .m→Python |

#### 第2章 先验：贝叶斯推断的灵魂

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验2.1 | 贝叶斯去噪：先验=正则化的数值验证（高斯先验→Tikhonov，Laplace先验→L1） | 贝叶斯定理；先验→正则化对应关系；-ln后验=数据项+正则项 | Calatroni P29-44 + CompImLab25 | ✅ |
| 实验2.2 | 观察正则化参数λ对重建的影响 | 正则化参数λ=噪声方差/先验方差；过拟合vs过正则化 | CompImLab25（变分去噪调参） | ✅ |
| 实验2.3 | Tweedie's formula直觉：去噪器与得分函数的关系 | Tweedie等式；∇log p(x)与去噪器的联系 | Ratti P9-13（2D去噪回归示例） | 🔄 需扩展 |
| 实验2.4 | 贝叶斯去噪器 vs 变分去噪器 vs 恒等映射：误差对比 | MMSE估计器；MAP估计器；先验质量对重建的影响 | Ratti P10-13（relative error对比） | 🔄 需扩展 |

#### 第3章 从MAP到后验探索

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验3.1 | 梯度下降求解Tikhonov正则化 | 梯度下降算法；Lipschitz条件；步长选择 | XR05_Tikhonov_comp（Python重写） | 🔄 .m→Python |
| 实验3.2 | ISTA/FISTA求解稀疏重建 | 近端梯度下降；软阈值；Nesterov加速 | PHD_MIVA_winter_school_lab | ✅ |
| 实验3.3 | TV正则化重建（原始-对偶算法） | TV正则化；原始-对偶分裂；Chambolle-Pock算法 | tomo_tv + XR09_TV_comp（Python重写） | 🔄 .m→Python |
| 实验3.4 | 截断SVD vs Tikhonov vs TV：正则化方法对比 | 不同先验的重建效果对比；正则化方法选择 | BunnyTomo3 + XR05 + XR09（Python重写） | 🔄 .m→Python |
| 实验3.5 | 近端算子计算与性质 | 近端算子定义与计算；近端=PnP的基础 | proximal.m + Unit2_exercise | ✅ |
| 实验3.6 | 经验贝叶斯：从数据自动估计正则化参数θ | 经验贝叶斯；Fisher Identity；边际似然优化 | Unit2_exercise（Fisher Identity + 随机近似） | ✅ |

---

### Part II 采样路径

#### 第4章 MCMC与ULA算法

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验4.1 | Metropolis-Hastings采样（1D高斯混合） | MCMC基本思想；接受-拒绝准则；细致平衡条件；Metropolis-Hastings算法 | 🆕 新写（1D教学案例） | 🆕 新写 |
| 实验4.2 | 1D高斯分布ULA采样 | ULA递推式；Euler离散化Langevin SDE；MH vs ULA对比 | lab1_ULA_sol（ULA_gauss函数） | ✅ |
| 实验4.3 | ULA步长δ对收敛的影响 | 步长选择；δ≤1/L条件；收敛性与偏差 | lab1_ULA_sol（实验δ=0.1,0.5,1.0） | ✅ |
| 实验4.4 | 2D图像ULA后验采样（去卷积） | 高维ULA；后验分布采样；势能函数U=-log后验 | lab1_ULA_sol（2D实验部分） | ✅ |
| 实验4.5 | MCMC收敛诊断：自相关与有效样本量 | MCMC收敛诊断；burn-in；自相关函数；ESS | lab1_ULA扩展 | 🔄 需补充 |

#### 第5章 朗之万动力学与得分函数

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验5.1 | 从MCMC到朗之万：Langevin SDE的推导与验证 | 朗之万SDE从MCMC的连续化推导；Langevin方程dx=∇log p(x)dt+√2dW；ULA作为Langevin的Euler离散 | lab1_ULA_sol（回顾ULA）+ Unit 2 Pereyra理论 | 🔄 需扩展 |
| 实验5.2 | Tweedie等式验证：去噪器→得分函数 | Tweedie等式∇log p_ε(x)=(D_ε(x)-x)/ε；得分函数 | lab2_PnP_sol（PnP ULA递推） | ✅ |
| 实验5.3 | PnP-ULA后验采样与不确定性量化 | PnP框架；用去噪器替换先验梯度；后验采样→不确定性量化 | lab2_PnP_sol（去卷积PnP采样） | ✅ |
| 实验5.4 | 近端算子 vs 学习去噪器：PnP中的先验替换 | 近端算子prox_λR→去噪器D_ε；显式先验→隐式先验 | proximal.m思想 + lab2_PnP | 🔄 需扩展 |

#### 第6章 得分匹配：从去噪中学习得分

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验6.1 | 训练一个UNet去噪器（DRUNet） | 去噪器作为得分估计器；条件噪声水平训练 | CompImLab25 Part 3 + MiniProject_DenoisingPrior | ✅ |
| 实验6.2 | 去噪得分匹配（DSM）：从去噪器提取得分 | DSM目标函数；得分匹配与去噪的等价性；s_θ≈∇log p；SSM（切片得分匹配）与Hutchinson迹估计简介 | 基于CompImLab25训练的去噪器 | 🆕 新写 |
| 实验6.3 | 用学习到的得分驱动PnP-ULA采样 | 学习得分→PnP采样；与手工先验对比 | lab2_PnP + 实验6.1的去噪器 | 🔄 组合 |

#### 第7章 扩散模型：SDE视角

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验7.1 | 从Langevin到扩散：增加时间维度的连续化推广 | Langevin→扩散SDE的连续时间推广；多时间步噪声调度；连续极限β(t)调度 | 🆕 新写（1D对比：单步Langevin vs 多步扩散） | 🆕 新写 |
| 实验7.2 | DDPM正向加噪过程：不同时间步的噪声水平 | 噪声调度α_t, ᾱ_t；信噪比随时间变化 | 🆕 新写（简单，基于DDPM公式） | 🆕 新写 |
| 实验7.3 | DDPM反向去噪采样 | 反向SDE；Euler-Maruyama离散化；去噪采样循环 | deepinv库 demo_diffusion_sde | ✅ |
| 实验7.4 | SDE采样 vs 概率流ODE vs DDIM：质量与速度对比 | 概率流ODE；确定性采样；DDIM加速；采样器权衡 | deepinv库（多个采样器对比） | 🔄 需扩展 |
| 实验7.5 | PnP-ULA vs 扩散模型：同一去卷积问题的对比 | 采样路径的终点：Langevin→扩散的自然升级 | MiniProject_DenoisingPrior | ✅ |

---

### Part III 变分路径

#### 第8章 变分推断与ELBO

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验8.1 | 手动计算1D高斯混合模型的ELBO | ELBO定义；Jensen不等式；KL散度 | 🆕 新写（教学用简化案例） | 🆕 新写 |
| 实验8.2 | Fenchel共轭计算练习 | Fenchel共轭；凸共轭与变分下界的关系 | MIVAcourse_opt1 理论 + 编程 | 🔄 需设计 |
| 实验8.3 | 变分推断与真实后验的对比（1D案例） | 变分族q的选择；平均场近似；变分间隙 | 🆕 新写 | 🆕 新写 |

#### 第9章 VAE与重参数化

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验9.1 | 实现简单VAE（MNIST） | 编码器-解码器架构；ELBO训练；重参数化技巧 | 🆕 新写（PyTorch，可参考CompImLab25训练流程） | 🆕 新写 |
| 实验9.2 | 重参数化技巧的数值验证 | 梯度穿过随机节点；REINFORCE vs 重参数化；方差对比 | 🆕 新写 | 🆕 新写 |
| 实验9.3 | VAE隐空间可视化与插值 | 隐空间结构；KL正则化；插值生成 | 🆕 新写 | 🆕 新写 |

#### 第10章 层级VAE与扩散的变分推导

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验10.1 | 实现层级VAE（2-3层） | 层级潜变量；马尔可夫推断链；层级ELBO推导；扩散过程的变分下界推导 | 🆕 新写（在实验9.1基础上扩展） | 🆕 新写 |
| 实验10.2 | 层级VAE与扩散加噪的类比：观察L→∞的极限 | 层级VAE→扩散的极限关系；高斯编码器=加噪过程；变分下界→扩散训练目标；扩散的变分推导 | 🆕 新写 | 🆕 新写 |

#### 第11章 扩散模型：变分视角

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验11.1 | 变分下界（VLB）的数值计算 | VLB分解；重建项+先验匹配项；去噪损失与VLB的关系 | 🆕 新写 | 🆕 新写 |
| 实验11.2 | 简化VLB训练扩散模型 | 简化VLB；噪声预测参数化；DDPM训练目标作为VLB的简化 | 🆕 新写（简化版VDM） | 🆕 新写 |

---

### Part IV 统一

#### 第12章 Score ≡ ELBO：殊途同归

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验12.1 | 数值验证：DSM损失 = VLB（1D案例） | Score Matching损失≡变分下界的数值验证；殊途同归 | 🆕 新写（结合实验6.2和实验11.1的结果） | 🆕 新写 |
| 实验12.2 | 采样路径 vs 变分路径：同一扩散模型的两种训练方式对比 | SDE训练目标 vs VLB训练目标；等价性的实践验证 | 🆕 新写 | 🆕 新写 |

#### 第13章 条件生成与逆问题求解

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验13.1 | 条件扩散采样：用扩散模型求解图像去模糊 | 条件扩散；逆问题=条件生成；后验采样 | MiniProject_DenoisingPrior扩展 | 🔄 需扩展 |
| 实验13.2 | Classifier-free guidance实现 | 引导采样；分类器引导 vs 无分类器引导；引导权重w | 🆕 新写 | 🆕 新写 |
| 实验13.3 | DDRM / DiffPIR算法实践 | 扩散逆问题求解算法；DDRM；DiffPIR | deepinv库（demo_ddrm + demo_diffpir） | ✅ |

---

### Part V 前沿

#### 第14章 Flow Matching与最优传输

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验14.1 | 2D点云Flow Matching（教学演示） | 最优传输问题形式化；Wasserstein距离；Monge vs Kantorovich形式；向量场学习；条件Flow Matching；OT-CFM | 🆕 新写 | 🆕 新写 |
| 实验14.2 | Rectified Flow图像生成 | Rectified Flow；直线插值 vs 扩散路径；与扩散的对比 | 🆕 新写 | 🆕 新写 |

---

### Part VI 实践与应用

#### 第15章 扩散模型的架构实践：UNet → DiT

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验15.1 | UNet去噪器实现与训练 | UNet架构；编码器-解码器+skip connections；时间步嵌入 | CompImLab25 + Bologna_UNet_example | ✅ |
| 实验15.2 | DiT架构实现（Patchify + adaLN-Zero + Transformer） | DiT；Patchify；adaLN-Zero条件注入；Transformer去噪器 | 🆕 新写（参考Peebles & Xie官方代码） | 🆕 新写 |
| 实验15.3 | UNet vs DiT去噪器性能对比 | 架构选择对扩散性能的影响；参数量/速度/质量权衡 | 实验15.1 + 实验15.2 组合 | 🔄 组合 |

#### 第16章 CT/MRI重建

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验16.1 | ASTRA断层成像算子与FBP重建 | Radon变换；滤波反投影；ASTRA工具箱 | astra_operators_example | ✅ |
| 实验16.2 | MRI k-space采样与零填充重建 | MRI正向模型（傅里叶采样）；k-space欠采样掩码；零填充重建；压缩感知MRI基础 | deepinv库MRI算子（sigpy/sigchem参考） | 🆕 新写 |
| 实验16.3 | UNet端到端CT重建 | 端到端学习重建；监督训练；post-processing | Bologna_UNet_example | ✅ |
| 实验16.4 | Learned Gradient Descent迭代重建 | 学习型迭代重建；算法展开；unrolled optimization | Bologna_LGS_example | ✅ |
| 实验16.5 | 扩散先验CT重建 | 扩散模型作为CT重建先验；DiffPIR for CT | MiniProject_DenoisingPrior方法迁移到CT | 🔄 需扩展 |

#### 第17章 自监督学习与等变架构

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验17.1 | Cryo-EM自监督去噪 | 自监督学习；无需干净数据的训练；Noise2Self | MiniProject_Self_Supervised | ✅ |
| 实验17.2 | Noise2Noise训练 | Noise2Noise原理；配对噪声数据训练；等价于MSE监督 | deepinv库自监督模块 | 🔄 需扩展 |
| 实验17.3 | 等变架构与测量一致性 | 等变性；测量一致性损失；物理约束融入网络 | Seminar Tachella理论 + deepinv | 🔄 需设计 |

#### 第18章 综合项目：用扩散模型求解自定义逆问题

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验18.1 | 定义自定义前向算子（多视角/超分辨/修复） | 自定义Physics类；前向模型设计 | MiniProject_DefiningOperator | ✅ |
| 实验18.2 | 扩散模型求解自定义逆问题（端到端） | 全书知识整合：贝叶斯框架→扩散采样→条件生成=逆问题求解 | 全书技能整合 | 🔄 需设计 |
| 实验18.3 | 不确定性量化：多次采样生成置信区间 | 后验采样→不确定性；像素级置信区间；多次采样统计 | lab1_ULA_sol方法 + 扩散采样 | 🔄 组合 |

---

### 练习统计

| 状态 | 数量 | 占比 |
|---|---|---|
| ✅ 有现成素材 | 20 | 36% |
| 🔄 有素材需改造/扩展 | 18 | 32% |
| 🆕 需从零新写 | 18 | 32% |
| **总计** | **56** | 100% |

### 新写练习重点清单（18个）

| 优先级 | 练习 | 对应知识点 | 难度 | 依赖 |
|---|---|---|---|---|
| ⭐⭐⭐ | 实验6.2 DSM：从去噪器提取得分 | DSM目标函数；得分匹配与去噪的等价性 | 中 | 实验6.1去噪器 |
| ⭐⭐⭐ | 实验9.1 实现简单VAE(MNIST) | 编码器-解码器架构；ELBO训练；重参数化 | 中 | PyTorch基础 |
| ⭐⭐⭐ | 实验10.1 层级VAE | 层级潜变量；马尔可夫推断链；层级ELBO | 中高 | 实验9.1 |
| ⭐⭐⭐ | 实验11.2 简化VLB训练扩散 | 简化VLB；噪声预测参数化；DDPM训练=VLB简化 | 高 | 实验10.1 |
| ⭐⭐⭐ | 实验12.1 DSM≡VLB数值验证 | Score Matching≡变分下界；殊途同归的数值验证 | 高 | 实验6.2+11.1 |
| ⭐⭐ | 实验4.1 Metropolis-Hastings采样 | MCMC；接受-拒绝准则；细致平衡条件 | 低 | 无 |
| ⭐⭐ | 实验7.1 从Langevin到扩散 | Langevin→扩散SDE连续时间推广；多时间步噪声调度 | 中 | 实验5.1+4.2 |
| ⭐⭐ | 实验8.1 1D高斯混合ELBO | ELBO定义；Jensen不等式；KL散度 | 低 | 无 |
| ⭐⭐ | 实验8.3 变分推断1D案例 | 变分族q；平均场近似；变分间隙 | 低 | 实验8.1 |
| ⭐⭐ | 实验9.2 重参数化数值验证 | 梯度穿过随机节点；REINFORCE vs 重参数化 | 低 | 实验9.1 |
| ⭐⭐ | 实验9.3 VAE隐空间可视化 | 隐空间结构；KL正则化；插值生成 | 低 | 实验9.1 |
| ⭐⭐ | 实验13.2 Classifier-free guidance | 引导采样；无分类器引导；引导权重w | 中高 | 实验7.3扩散模型 |
| ⭐ | 实验7.2 DDPM正向加噪 | 噪声调度α_t, ᾱ_t；信噪比随时间变化 | 低 | 无 |
| ⭐ | 实验10.2 层级VAE→扩散极限 | 层级VAE→扩散极限；高斯编码器=加噪 | 中 | 实验10.1 |
| ⭐ | 实验11.1 VLB数值计算 | VLB分解；重建项+先验匹配项 | 中 | 实验9.1 |
| ⭐ | 实验14.1 2D点云Flow Matching | 最优传输；Wasserstein距离；向量场学习；OT-CFM | 中 | 无 |
| ⭐ | 实验14.2 Rectified Flow图像生成 | Rectified Flow；直线插值 vs 扩散路径 | 高 | 实验14.1 |
| ⭐ | 实验16.2 MRI k-space采样与重建 | MRI正向模型；k-space欠采样；压缩感知MRI | 中 | 实验16.1 |

---

## 十一、写作要点

1. **Part II每一章必须保持"逆问题动机"不断线**：每章开头明确回答"逆问题的什么局限性推动了本章方法的出现"。读者应始终记得"我们为什么需要这个方法"，而非仅仅看到技术推导
2. **第2章埋下"显式先验天花板"的伏笔**：在2.4节强调真实图像分布的复杂性远超高斯/Laplace/TV——这驱使先验从手工走向数据驱动，是全书演化的第一推动力
3. **第6章与第9章互相呼应**：采样路径学到第6章（去噪=学习得分），变分路径学到第9章（VAE编码=隐空间映射），两者都是"用神经网络逼近概率分布"，但手段不同
4. **第12章等价性证明放正文不放附录**：这是两条路径的数学收网时刻，要用直观推导+形式化证明双写
5. **每条路径内部要有"实践锚点"**：
   - 采样路径：第4章实现ULA，第6章实现去噪得分匹配
   - 变分路径：第9章实现VAE，第10章实现简化版VDM
6. **第3章结尾设分叉点**：提出核心问题"如何处理复杂后验？"，引出两个方向：采样还是近似？**明确采样路径为主线**
7. **第13章是全书核心论点的落脚点**：条件扩散 = 逆问题求解，完成从第1章到第13章的完整闭环。扩散模型能超越传统方法的三重优势——任意复杂先验、不确定性量化、零样本迁移——应在此明确总结
8. **变分路径在Part III开头明确定位**："我们刚看到逆问题的需要如何推动了从Langevin到扩散的演化。有趣的是，还有一条完全不同的路也通向扩散模型——变分推断。"

---

## 十二、整体叙事逻辑图

```
Part I   贝叶斯基石
          │  ┌─────────────────────────────────────────────────┐
          │  │核心论点：逆问题的需求如何一步步驱动方法演化？    │
          │  │显式先验有限 → 需要学习型先验                     │
          │  │MAP丢弃不确定性 → 需要后验采样                    │
          │  └─────────────────────────────────────────────────┘
          │
          ├────────────────────────────┐
          ▼                            ▼
Part II  采样路径（主线）          Part III 变分路径（副线）
         逆问题驱动                  生成建模驱动
         ULA→Langevin→Score       ELBO→VAE→层级VAE
          │                            │
          ▼                            ▼
       Diffusion(SDE)             Diffusion(VLB)
          │                            │
          └──────────┬─────────────────┘
                     ▼
Part IV   统一：Score ≡ ELBO
          条件生成 = 逆问题求解 ←── 全书高潮·闭环第1章
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
Part V   Flow      Part VI    实践
         Matching   架构应用    项目
```

---

## 十三、环境与工具依赖

### 核心工具链

| 工具 | 用途 | 涉及章节 | 安装方式 |
|---|---|---|---|
| Python ≥3.9 | 编程语言 | 全书 | — |
| PyTorch ≥2.0 | 深度学习框架 | 第6,9-15章 | `pip install torch torchvision` |
| deepinv | 逆问题与扩散模型库 | 第1,4-7,13,16-18章 | `pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv` |
| ASTRA Toolbox | CT断层成像算子 | 第1,16章 | `pip install astra_toolbox` |
| NumPy / SciPy | 数值计算 | 第1-8章 | `pip install numpy scipy` |
| Matplotlib | 可视化 | 全书 | `pip install matplotlib` |

### deepinv 关键API索引

| 模块 | 类/函数 | 对应练习 |
|---|---|---|
| `deepinv.models` | `DRUNet`, `DiffUNet`, `NCSNpp` | 实验6.1, 7.3, 13.3 |
| `deepinv.physics` | `Denoising`, `GaussianNoise`, `Downsampling`, `LinearPhysics`, `Tomography` | 实验1.2, 16.1, 18.1 |
| `deepinv.physics.blur` | `gaussian_blur` | 实验1.2, 13.1 |
| `deepinv.optim` | `DPIR` | 实验16.5 |
| `deepinv.loss` | `SplittingLoss`, `Neighbor2Neighbor`, `SureGaussianLoss`, `R2RLoss`, `EILoss` | 实验17.1-17.3 |
| `deepinv.datasets` | `HDF5Dataset`, `generate_dataset` | 实验17.1 |
| `deepinv.Trainer` | 训练编排器 | 实验17.1 |
| `deepinv.utils` | `plot`, `load_example`, `get_freer_gpu` | 全书 |

### 预训练模型

| 模型 | 权重来源 | 用途 |
|---|---|---|
| DRUNet | `huggingface.co/deepinv/drunet` | 通用去噪器/得分估计 |
| DiffUNet | `huggingface.co/deepinv/diffunet` | 扩散去噪 |
| NCSNpp | `huggingface.co/deepinv/edm` | Score-SDE模型 |

---

## 十四、讲座材料中待整合的内容要点

> 以下为讲座中发现的、应纳入对应章节但book_plan尚未体现的重要知识点。

### 高优先级（直接影响章节完整性）

| 内容 | 来源 | 建议纳入章节 | 说明 |
|---|---|---|---|
| Poisson噪声的KL散度数据项 | Calatroni P36-37 | 第1章 | 非高斯噪声的数据项推导，CT/PET成像的核心 |
| Laplace/脉冲噪声的L1数据项 | Calatroni P37-38 | 第1章 | 与L2数据项对称，补完噪声→数据项映射 |
| 贝叶斯→变分完整对应表 | Calatroni P29-44 | 第2章 | 噪声模型→似然→数据项，先验→正则项，参数→超参数的系统映射表 |
| Tikhonov闭式解（DFT域） | Calatroni P53 | 第3章 | 迭代法之外的直接求解法，教学对比用 |
| 贝叶斯决策理论 | Pereyra P24-25 | 第3章 | MAP/MMSE最优性的理论根基 |
| ADMM算法 | Gondzio L6 | 第3章 | 与原始-对偶并列的重要优化方法 |
| Moreau-Yoshida近似理论 | Pereyra P43-44 | 第5章 | MYULA的理论保证，近似误差界 |
| MAP/MMSE结构对偶（Moreau vs 软下卷积） | Pock L2 | 第5章 | 近端=Moreau一步梯度(MAP)，Tweedie=软下卷积一步梯度(MMSE) |
| VLB三种参数化（ε/s/x₀） | — | 第11章 | 噪声预测vs得分预测vs直接预测，训练稳定性差异 |
| Fourier切片定理 | Siltanen Day3A | 第16章 | FBP的理论根基，CT重建必讲 |
| Beer-Lambert定律 | Siltanen Day2 | 第16章 | CT成像物理基础 |
| 有限角CT与波前集理论 | Siltanen Day3A | 第16章 | 为什么有限角丢失信息的数学解释 |

### 中优先级（增强深度与前沿性）

| 内容 | 来源 | 建议纳入章节 | 说明 |
|---|---|---|---|
| TGV（广义全变差）正则化 | Siltanen Day2, Pock L3 | 第2/3章 | TV的进阶版，避免阶梯效应 |
| 四种学习设定分类法 | Ratti P18-21 | 第9/17章 | 监督/自监督/无监督-x/无监督-y的系统分类 |
| Bregman距离与误差估计 | Benning L2 | 第3章 | 正则化解与真解的误差分析 |
| 欠阻尼Langevin/惯性Langevin | Pock L2 | 第4章 | 采样加速方法，与FISTA对称 |
| "扩散在绝对零度"概念 | Pock L2 | 第7章 | 温度参数T→0时PnP-近端→PnP-扩散的桥梁 |
| Fields of Experts (FoE)先验 | Pock L2 | 第2章 | 学习型卷积先验模型 |
| 过参数化与双重下降 | Ratti P29-30 | 第9/15章 | 现代深度学习的重要现象 |
| Restormer架构 | Ratti P38 | 第15章 | ViT→DiT之间的桥梁架构 |
| 学习MRI采样模式 | Benning L2 | 第16章 | k-space最优欠采样，deepinv+源条件方法 |
| IHT（迭代硬阈值） | MIVA opt3 | 第3章 | L0的直接算法，与软阈值(ISTA)对比 |
| 压缩感知与RIP | MIVA opt3 | 第3/16章 | 稀疏恢复的理论保证 |

### 低优先级（可选择性纳入或作为延伸阅读）

| 内容 | 来源 | 建议纳入章节 | 说明 |
|---|---|---|---|
| 贝叶斯假设检验 | Pereyra P30,51-53 | 第3/13章 | 结构检测的形式化框架 |
| 贝叶斯模型选择/平均 | Pereyra P31-33 | 第3章 | 多模型竞争下的推理 |
| Gibbs采样 | Pock L2 | 第4章 | 与MH并列的经典MCMC方法 |
| 半二次最小化→GLM | Pock L2 | 第4/5章 | 结构化先验的可采栾示例 |
| 双层优化学习正则化 | Pock L3, Benning L2 | 第15/16章 | 学习正则化器参数的前沿方法 |
| Gamma收敛理论 | Pock L3 | 第3章 | 离散→连续正则化的数学保证 |
| 发射断层成像(PET) | Siltanen Day3B | 第16章 | 非线性CT，Levenberg-Marquardt方法 |
| 通用近似定理 | Ratti P33 | 第9章 | NN表达能力的形式化保证 |
| 训练最佳实践 | Ratti P44-48 | 第15章 | 初始化/调度/早停/验证 |
| Helsinki Tomography Challenge | Siltanen Seminar | 第18章 | 可作为综合项目素材 |

---

## 十五、补充资源清单

> 以下为book_plan.md原资源清单中遗漏的文件。

### 遗漏的.ipynb文件

| 文件 | 位置 | 功能 | 对应章节 |
|---|---|---|---|
| Lab_1.zip | Unit 3 Labs/ | Unit 3先验学习实验 | 第6章 |
| Lab_2.zip | Unit 3 Labs/ | Unit 3先验学习实验 | 第6章 |

### 遗漏的.pdf/.md文件

| 文件 | 位置 | 功能 | 对应章节 |
|---|---|---|---|
| Bologna_exercise.pdf | Unit 6/ | 博洛尼亚课程期末练习 | 第18章 |
| Seminar_Roffilli.md | Seminars/ | 应用报告 | 第16/18章 |
| Seminar_Vezzali.ppsx | Seminars/ | 应用报告 | 第18章 |

### 遗漏的.zip文件

| 文件 | 位置 | 功能 | 对应章节 |
|---|---|---|---|
| lab2_PnP.zip | Unit 2 labs/ | PnP实验补充数据 | 第5章 |
| Bologna_summerschool_share_Hauptmann.zip | Unit 4/ | 迭代重建补充材料 | 第16章 |

### 竞赛与开放数据集

| 资源 | 说明 | 对应章节 |
|---|---|---|
| Helsinki Tomography Challenge 2022 | 有限角CT竞赛，含评分标准(MCC) | 第18章 |
| Helsinki Deblur Challenge 2021 | 图像去模糊竞赛 | 第18章 |
| fips.fi/dataset.php | 开放X射线数据集 | 第16章 |

---

## 十六、参考文献策略

### 核心参考文献（全书引用）

| 文献 | 用途 | 涉及章节 |
|---|---|---|
| Song et al. "Score-Based Generative Modeling through SDEs" ICLR 2021 | 采样路径+SDE理论基础 | 第4-7,12章 |
| Kingma & Welling "Auto-Encoding Variational Bayes" ICLR 2014 | VAE原始论文 | 第9章 |
| Kingma et al. "Variational Diffusion Models" NeurIPS 2021 | 变分扩散推导 | 第10-12章 |
| Ho et al. "Denoising Diffusion Probabilistic Models" NeurIPS 2020 | DDPM原始论文 | 第7,11章 |
| Dhariwal & Nichol "Diffusion Models Beat GANs on Image Synthesis" NeurIPS 2021 | Classifier Guidance | 第13章 |
| Ho & Salimans "Classifier-Free Diffusion Guidance" 2022 | Classifier-Free Guidance | 第13章 |
| Lipman et al. "Flow Matching for Generative Modeling" ICLR 2023 | Flow Matching | 第14章 |
| Liu et al. "Flow Straight and Fast" ICLR 2023 | Rectified Flow | 第14章 |
| Luo "Understanding Diffusion Models: A Unified Perspective" 2022 | Score≡VLB统一综述 | 第12章 |
| Bishop "Pattern Recognition and Machine Learning" Ch10 | 变分推断经典教材 | 第8-10章 |

### 按章节补充文献

| 章节 | 补充文献 |
|---|---|
| 第1章 | Siltanen & Mueller "Linear and Nonlinear Inverse Problems with Practical Applications" |
| 第2-3章 | Benning & Burger "Modern Regularization Methods for Inverse Problems" |
| 第4-5章 | Durmus & Moulines "High-dimensional Bayesian inference via ULA" |
| 第6章 | Hyvärinen "Estimation of Non-Normalized Statistical Models by Score Matching" |
| 第15章 | Peebles & Xie "Scalable Diffusion Models with Transformers" (DiT); Zamir et al. "Restormer" |
| 第16章 | Lustig et al. "Compressed Sensing MRI"; Benning "Learning Optimal Sampling" |
| 第17章 | Batson & Royer "Noise2Self"; Krull et al. "Noise2Void"; Tachella "Equivariant Imaging" |
