# ⚠ 待终校清单——全部清零 ✅（原 80 → 29 → 第一组 11 ✅ → 第二组 14 ✅ + 1废弃 + 1拆分 → 第三组 bubba ✅ → wu→Trippe 张冠李戴纠正 → **0**；2026-10-21 复核补记：holden2022langevin 平反恢复合并）

> ✅ **第一组（11 个 `*unknown`）已于 2026-10-20 全部经作者确认 + OpenAlex 联网坐实并回填 bib。**
> ✅ **第二组（17 个会场/卷页键）已于同日经作者裁决 + 联网核查回填**（含拆分 laumont 为两条、废弃 chung2025survey、纠正 kobler/frikel/greenleaf/tran/tachella 等多处作者/venue 错误；其中 holden2022langevin 的"废弃"已于 2026-10-21 平反——经 arXiv:2103.10182 坐实与 holden2022unknown 为同一篇 SIAM J. Imaging Sci. 15(2):892–924 (2022) 正式版，正文两处引用合并至 `holden2022datadriven`）。
> ✅ **第三组 bubba2019learning** 已于同日修正：首名 Giorgia→**Tatiana A.**、篇名 Photoacoustic→**shearlet/limited angle CT**、完整 7 人名单。
> ✅ **wu2024smcdiff** 最终确认为正文张冠李戴：13.3#630 "Wu et al. (2024)" 实为 **Trippe et al. (2022)** (arXiv:2206.04119)。bib 已回填正确作者/篇名，citekey 保留不改。Phase 5 须将正文作者引用一并修正。


> 用途：本清单原追踪 `references.bib` 中 **80** 个 `note={⚠}` 条目，现已 **全部清零（0 个活跃 ⚠）**。
> 每条附「正文首见位置 + 该处上下文摘录 + 待确认要点」，便于对照原文一次性判定并回填 bib。
> 判定后：坐实 → 删除该条 bib 的 `note={⚠…}`（或改 `note={已联网坐实…}`）；发现字段错 → 直接改正 title/journal/volume/pages/author/year。
> 说明：摘录行号可能因编辑略有漂移，以「定位文件 + 作者姓氏」检索为准；若标 ⛔ 未在定位处命中，请全文检索该姓氏。


## 一、身份存疑（`*unknown` 键）（11 键）✅ 已确认回填（⚠ → ✓）

### `bellec2021unknown`
- 正文写法：**Bellec et al**（年：2021）
- 定位：`ch17/17.7#104` → `book/manuscript/ch17/17.7 本章小结：从数据困境到自监督统一.md`
  - 上下文摘录：
    > - **非线性逆问题**：本章方法主要针对线性正向模型 $\mathbf{y} = Ax + \varepsilon$，非线性的自监督训练更具挑战；
    > - **复杂噪声模型**：空间相关噪声、信号依赖噪声（如 Poisson）的自监督方法仍在发展中；
    > - **与扩散模型深度整合**：用 SURE 训扩散模型（Daras et al. 2024）是新方向；
    > - **自监督的不确定性量化**：SURE 本身可估重建误差（SURE4SURE, Bellec et al. 2021），但系统性的自监督不确定性框架仍待完善。
    > 
    > ---
- 待确认要点（bib note 摘录）：⚠待补：正文 17.7#104 "Bellec et al. (2021)" 所指不明，须作者确认}

### `gozcu2018unknown`
- 正文写法：**Gozcu et al**（年：2018）
- 定位：`ch16/16.4#107` → `book/manuscript/ch16/16.4 学习型重建方法.md`
  - 上下文摘录：
    > **与源条件的联系**：回顾第 3 章 3.6 的源条件，正则化解的误差界和正向算子 $A$ 的性质相关。MRI 里 $A = M_\Omega F$，所以采样模式 $\Omega$ 直接决定重建误差理论上界——学习 $\Omega$ 等价于在加速约束下最小化误差上界。
    > 
    > **当前进展**：LOUPE（Bahadir et al. 2020，连续松弛 + Gumbel-Softmax 学 1D Cartesian 掩码）、Joint Optimization（Gozcu et al. 2018，联合优化采样与重建）、自适应采样（按当前不确定性动态选下一个采样点，类似主动学习）。
    > 
    > ## 本节小结
- 待确认要点（bib note 摘录）：⚠待补：正文 16.4#107 "Gozcu et al. (2018)"（联合优化采样与重建）；上列为候选题名，第二作者/确切出处（arXiv/会场）须核},

### `holden2022unknown`
- 正文写法：**Holden et al**（年：2022）
- 定位：`ch09/9.4#212` → `book/manuscript/ch09/9.4 VAE作为逆问题的学习先验.md`
  - 上下文摘录：
    > ---
    > 
    > **来源**：Pereyra Lecture 3 P33-36; Ratti Pre-course II P18-21; Holden et al. (2022); Ongie et al. (2020) Deep priors for inverse problems
    > 
    > ## 收尾：为什么这件事重要
- 待确认要点（bib note 摘录）：⚠待补：正文 9.4#212 "Holden et al. (2022)"；与 2.4#293 Holden, Pereyra & Zygalakis 疑不同篇，须核}

### `holderrieth2024unknown`
- 正文写法：**Holderrieth et al**（年：2024）
- 定位：`ch14/14.6#348` → `book/manuscript/ch14/14.6 离散扩散与掩码语言模型：从向量场到速率矩阵.md`
  - 上下文摘录：
    > - 连续时间离散扩散的奠基：Campbell et al., *A Continuous Time Framework for Discrete Denoising Models* (2022)；Austin et al., *Structured Denoising Diffusion Models in Discrete State-Spaces*（D3PM, NeurIPS 2021）
    > - 离散 Flow Matching 与本节配方：Gat et al., *Discrete Flow Matching* (NeurIPS 2024)
    > - 掩码扩散语言模型：Sahoo et al., *Simple and Effective Masked Diffusion Language Models* (NeurIPS 2024)；Lou et al., *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*（SEDD, ICML 2024 最佳论文）
    > - 统一框架：Holderrieth et al., *Generator Matching: Generative modeling with arbitrary Markov processes* (2024)
- 待确认要点（bib note 摘录）：正文 14.6#348 "Holderrieth et al., Generator Matching"；候选为 arXiv 'Generator Matching with Trajectory Balance'（或 ICLR 2025），确切篇名/会场/作者待核实}

### `kumar2018unknown`
- 正文写法：**Kumar et al**（年：2018）
- 定位：`ch09/9.3#274` → `book/manuscript/ch09/9.3 ELBO训练与KL正则化.md`
  - 上下文摘录：
    > - **Beta-VAE 度量**（Higgins et al., 2017）：在受控生成数据集上检查隐变量与生成因素的线性可分性；
    > - **Factor-VAE 度量**（Kim & Mnih, 2018）：用分类器评估隐变量维度与生成因素的对应；
    > - **SAP 分数**（Kumar et al., 2018）：基于平均可分性。
    > 
    > 研究里常用，实战中多靠可视化定性看。
- 待确认要点（bib note 摘录）：⚠待补：正文 9.3#274 "Kumar et al. (2018)" 所指不明，须作者确认},

### `narnhofer2019unknown`
- 正文写法：**Narnhofer et al**（年：2019）
- 定位：`ch02/2.4#248` → `book/manuscript/ch02/2.4 从显式先验到隐式先验.md`
  - 上下文摘录：
    > $$\hat{z} = \arg\min_{z \in \mathbb{R}^p} \frac{1}{2}\|AG_\theta(\mathbf{z}) - \mathbf{y}\|_2^2, \quad \text{s.t.} \;\|\mathbf{z}\| \leq \sqrt{p}$$
    > 
    > 重建图像为 $\hat{x} = G_\theta(\hat{z})$。Bora 等人（2017）和 Narnhofer 等人（2019）将这一方法成功用于 MRI 重建等实际问题。
    > 
    > 但 GAN 先验有根本性局限：**生成器定义的分布是隐式的**——无法计算 $p_G(\mathbf{x})$ 的值，也无法直接对后验 $p(\mathbf{x}|\mathbf{y})$ 采样。GAN 只能提供 MAP 类型的点估计，无法量化不确定性。此外，GAN 训练中的**模式坍缩**（mode collapse）意味着生成分布可能只覆盖真实分布的一部分，导致重建多样性不足。
- 待确认要点（bib note 摘录）：⚠待补：正文 2.4#248 "Narnhofer et al. (2019)"（隐式先验语境）篇名/作者须核}

### `pang2021unknown`
- 正文写法：**Pang et al**（年：2021）
- 定位：`ch17/17.3#121` → `book/manuscript/ch17/17.3 SURE：Stein无偏风险估计与R2R.md`
  - 上下文摘录：
    > ## 17.3.5 R2R：避免散度计算的 SURE 替代
    > 
    > SURE 有个实用痛点：散度项要 Monte Carlo 近似或自动微分，计算贵，还对 $f$ 的可微性有要求。R2R（Recorrupted-to-Recorrupted，Pang et al. 2021）提供了**完全绕开散度**的替代。
    > 
    > **为什么需要 R2R**：既然散度是 SURE 的痛点，那能不能换一条路，连散度都不算就得到等价的损失？R2R 的答案是"重新加噪"。
- 待确认要点（bib note 摘录）：⚠待补：正文 17.3#121 "Pang et al. (2021)"（R2R 语境）篇名须核}

### `tachella2025unknown`
- 正文写法：**Tachella**（年：2025）
- 定位：`ch16/16.4#40` → `book/manuscript/ch16/16.4 学习型重建方法.md`
  - 上下文摘录：
    > - **需要配对训练数据**：干净的配对 $(\mathbf{x},\mathbf{y})$ 在医学里极难获得
    > 
    > > **延伸：摆脱配对数据——等变成像的自监督重建（见第 17 章）**
    > > 上面"需要配对训练数据"是端到端学习与展开网络的根本瓶颈。第 17 章给一条**完全绕过配对数据**的路——等变成像（Equivariant Imaging, EI）。动机：多数逆问题算子 $A$ 只是其零空间里的成分不可观测；若数据分布有某种对称性（旋转、平移、缩放），就能用该对称性"虚拟"出多个观测来约束零空间方向。Tachella (2025) 在加速 MRI（4× 加速、高斯噪声）上的数字：
    > >
    > > | 方法 | 是否需配对数据 | 噪声假设 | PSNR (dB) |
    > > |---|---|---|---|
- 待确认要点（bib note 摘录）：⚠待补：正文 16.4#40 单作者 "Tachella (2025)"；首作者名应为 Juli{\'a}n（上一版误作 'J{\'e}r{\'e}my' 已更正）；若为讲座则属非正式来源应剔除，否则补条目}
- ✅ **2026-10-21 合并**：经作者裁决为非讲座，坐实为 UNSURE 正式文献（Tachella, Davies \& Jacques, ICLR 2025, arXiv:2409.01985）——**与 17.4#301 的 `tachella2025unsure` 为同一篇**，本键已删除，映射表 #207/#208 两行统一指向 `tachella2025unsure`。

### `terris2022unknown`
- 正文写法：**Terris et al**（年：2022）
- 定位：`ch05/5.5#249` → `book/manuscript/ch05/5.5 PnP框架：用去噪器替换先验梯度.md`
  - 上下文摘录：
    > - **Sreehari et al. (2016)**：把 PnP-ADMM 扩展到更多逆问题（CT、超分辨）；
    > - **Laumont et al. (2021)**：建立 PnP-ULA 收敛性理论——证明在 Lipschitz 去噪器条件下，PnP-ULA 收敛到后验的邻域；
    > - **Ryu et al. (2019)**：给出 PnP-ADMM / PnP-FISTA 收敛条件——去噪器需是"压缩映射"；
    > - **Terris et al. (2022)**：PnP-ULA 在医学成像中的不确定性量化应用。
    > 
    > PnP 的核心洞见是**去噪器即先验**——去噪器隐式定义先验分布，Tweedie 把这一隐式先验翻译成显式得分函数。这让 PnP 既能使用数据驱动的强大先验，又保持采样框架的概率推断能力。
- 待确认要点（bib note 摘录）：⚠待补：正文 5.5#249 "Terris et al. (2022)"（PnP/算子学习）篇名须核}

### `winkler2021unknown`
- 正文写法：**Winkler**（年：2021）
- 定位：`ch07/7A#79` → `book/manuscript/ch07/附录7A Anderson逆向时间SDE定理的证明概要.md`
  - 上下文摘录：
    > - 逆向过程需"恢复"信息——得分函数 $\nabla\log p_t(\mathbf{x})$ 指向概率增长方向；
    > - $-g(t)^2$ 的系数确保修正强度与正向噪声强度匹配——噪声越强，修正越大。
    > 
    > **来源**：Anderson (1982); Winkler (2021); Risken (1996); Song et al. (2021)
- 待确认要点（bib note 摘录）：⚠待补：正文 7A#79 "Winkler (2021)"（Anderson 逆时 SDE 证明概要）出处不明，须确认}

### `zhang2021unknown`
- 正文写法：**Zhang**（年：2021）
- 定位：`ch06/6.6#34` → `book/manuscript/ch06/6.6 去噪器作为得分估计器：实践与架构.md`
  - 上下文摘录：
    > **UNet**（Ronneberger 2015）是图像去噪/得分的事实标准：编码器逐步下采样提多尺度特征，解码器上采样恢复分辨率，跳跃连接把编码器特征直送解码器对应层以保空间细节。它同时抓全局结构和局部纹理，参数效率还高（层次结构详见第15章）。
    > 
    > **DRUNet**（Zhang 2021）在 UNet 上加了**噪声水平条件输入**，成为多尺度得分估计的理想架构。做法：把标量 $\sigma$ 经一个小 MLP 映射成特征向量 $e_\sigma$，再注入每个残差块（相加或用 adaLN 调制）。
    > 
    > ```
    > 输入: x（含噪图）, σ（噪声水平）
- 定位：`ch05/5.5#159` → `book/manuscript/ch05/5.5 PnP框架：用去噪器替换先验梯度.md`
  - 上下文摘录：
    > ### 深度学习去噪器
    > 
    > - **DnCNN**（Zhang et al., 2017）：残差学习，盲去噪（不指定噪声水平）；
    > - **DRUNet**（Zhang et al., 2021）：条件去噪网络，把噪声水平 $\varepsilon$ 作为额外输入；
    >   - 关键优势：**同一网络处理不同噪声水平** → 对应不同尺度的得分函数；
    >   - 这让 DRUNet 成为 PnP-ULA 和多时间步扩散模型（第 7 章）的理想去噪器。
- 待确认要点（bib note 摘录）：⚠ 仅剩 title={⚠待核}；正文 6.6#34 "Zhang (2021)" 与 5.5#159 "Zhang et al. (2021)"（DRUNet 语境）——疑指 *Multi-Grade Residual Learning*（CVPR 2021）同一篇，须坐实具体篇名/会场后回填 title。

## 二、会场 / 年份 / 卷页待确认（题名多已坐实，余字段待终校）（17 键）

### `bahadir2020loupe`
- 正文写法：**Bahadir et al**（年：2020）
- 定位：`ch16/16.4#107` → `book/manuscript/ch16/16.4 学习型重建方法.md`
  - 上下文摘录：
    > **与源条件的联系**：回顾第 3 章 3.6 的源条件，正则化解的误差界和正向算子 $A$ 的性质相关。MRI 里 $A = M_\Omega F$，所以采样模式 $\Omega$ 直接决定重建误差理论上界——学习 $\Omega$ 等价于在加速约束下最小化误差上界。
    > 
    > **当前进展**：LOUPE（Bahadir et al. 2020，连续松弛 + Gumbel-Softmax 学 1D Cartesian 掩码）、Joint Optimization（Gozcu et al. 2018，联合优化采样与重建）、自适应采样（按当前不确定性动态选下一个采样点，类似主动学习）。
    > 
    > ## 本节小结
- 待确认要点（bib note 摘录）：⚠待补：正文 16.4#107 标注为 'LOUPE'，实为 Bahadir et al. 2020 TCIA（连续松弛/Gumbel-Softmax 学 1D Cartesian 掩码）；非 Adler-{\"O}ktem 2018，作者名单请补全}

### `borwein2010convex`
- 正文写法：**Borwein & Lewis**（年：2010）
- 定位：`ch08/8A#183` → `book/manuscript/ch08/附录8A Fenchel共轭与ELBO的优化理论根基.md`
  - 上下文摘录：
    > 两者共享同一套凸分析工具——Fenchel 共轭是统一数学语言。这解释了为何 ELBO 与正则化有相同"数据+正则"结构：**它们是同一个凸优化问题的不同表达**。这一转变意味着，第 2 章的 MAP 正则化作图和本章的 ELBO 变分推断，并不是两条不相干的支流，而是一条凸分析大河的两岸。
    > 
    > **来源**：Benning L2 P23-25; LectureNotes2020_v2 Ch4 (Fenchel conjugate, lines 1832-1979); Rockafellar (1970) Convex Analysis; Borwein & Lewis (2010) Convex Analysis and Nonlinear Optimization
- 待确认要点（bib note 摘录）：⚠待补：正文附录8A 作 "Borwein & Lewis (2010)"（Fenchel 共轭），具体书目请核对（二人有多种教材）},

### `chen2014foe`
- 正文写法：**Chen, Ranftl & Pock**（年：2014）
- 定位：`ch02/2.4#107` → `book/manuscript/ch02/2.4 从显式先验到隐式先验.md`
  - 上下文摘录：
    > 其中 $\phi_k(t) = \text{diag}(\ldots, \rho_k'(t_{ij}), \ldots)$ 是势函数导数。引入拉格朗日乘子 $p$ 即可解出 $\theta$ 的梯度——这就是隐微分的核心思想。
    > 
    > Chen, Ranftl 和 Pock（2014）用此方法在约 200 张图像的库上学习了 80 个 $9 \times 9$ 滤波器及对应的势函数参数，共 6480 个参数。学出的滤波器呈现出多种方向与频率的特征检测模式——水平、垂直、对角线的边缘检测器，不同尺度的纹理检测器。这些滤波器不是人为设计的，而是从数据中自动涌现的，验证了 FoE "让数据说话"的理念。
    > 
    > ## 实验 2.4-1：FoE 半学习先验的去噪验证
- 待确认要点：title/作者已坐实（Chen, Ranftl & Pock, *A Statistical Approach to Image Denoising by Prior Learning from Patch Representations*, 2014）；⚠ 仅剩 `booktitle={⚠待核（Energy Minimization / Scale-Space 系列）}`——须核确切会场（EMMCVPR 2013 / SSVM 2013 系列？）。

### `chung2025survey`
- 正文写法：**Chung et al**（年：2025）
- 定位：`ch12/12.7#158` → `book/manuscript/ch12/12.7 全书叙事的汇合.md`
  - 上下文摘录：
    > 第13章在建立"条件扩散采样 = 逆问题求解"时，实际上复现了与本章同构的"双路径"结构——构成全书核心论点的**第二重闭环**：
    > 
    > - **得分路径（后验得分分解）**：Chung et al. (2025) 把条件采样拆为
    >   $$\nabla\log p(\mathbf{x}|\mathbf{y}) = \nabla\log p(\mathbf{x}) + \nabla\log p(\mathbf{y}|\mathbf{x})$$
    >   第一项是无条件先验得分（本章已学），第二项由具体正向算子 $A$ 提供。DPS、DDRM 等显式近似方法都走这条路。Chung et al. 还给出与之平行的"变分视角"（从联合分布 $p(\mathbf{x},\mathbf{y})$ 出发推导），与本书第8–11章的变分路径遥相呼应。
    > - **控制路径（最优控制）**：Li & Pereira (NeurIPS 2024) 把同一步采样重述为一个离散最优控制问题——把数据一致性当作控制代价，把去噪得分当作动力学。其优雅结论是：**所需的条件得分函数直接等于价值函数 $V(\mathbf{x}_t,t)$ 对其状态 $\mathbf{x}_t$ 的雅可比** $\nabla_{\mathbf{x}_t}V$。
- 待确认要点（bib note 摘录）：⚠待补：正文 12.7#158 与 13.3#742 "Chung et al. (2025)"；与 Daras 2410.00083 综述区分，确切篇名未联网坐实，待作者核实}

### `frikel2013deconv`
- 正文写法：**Frikel & Quinto**（年：2013）
- 定位：`ch16/16C#33` → `book/manuscript/ch16/附录16C 波前集理论与有限角CT的数学框架.md`
  - 上下文摘录：
    > ## C.3 有限角CT的可见性定理
    > 
    > **定理**（Greenleaf & Uhlmann 1989；Frikel & Quinto 2013）：设测量角度范围为 $\Theta \subset [0,\pi)$，则从有限角度数据 $\mathcal{R}_\Theta u$ 能稳定恢复的波前集为：
    > 
    > $$WF_{\text{visible}}(u) = \{(\mathbf{x},\xi) \in WF(u) : \xi/|\xi| = (\cos\theta,\sin\theta),\ \theta \in \Theta\}$$
- 待确认要点（bib note 摘录）：⚠待补：正文 16C 作 "Frikel \& Quinto (2013)"（有限角/伪微分去卷积）；首作者为 Johannes Frikel（上一版误作 Ingrid 已更正）；合作的 Quinto 为 Eric Todd Quinto（上一版误作 'Erica L.' 已更正）；其 2013 有两篇常被引：{Characterization and Reduction of Artifacts in Limited Angle Tomography}, Inverse Problems 29:125007 (2013)（已联网坐实）与 {Partial-Data Deconvolution by Regularization under a Total Variation Assumption}（会场未坐实），按正文去卷积语境候选后者，确切篇名/卷页请核对},

### `greenleaf1989visibility`
- 正文写法：**Greenleaf & Uhlmann**（年：1989）
- 定位：`ch16/16C#33` → `book/manuscript/ch16/附录16C 波前集理论与有限角CT的数学框架.md`
  - 上下文摘录：
    > ## C.3 有限角CT的可见性定理
    > 
    > **定理**（Greenleaf & Uhlmann 1989；Frikel & Quinto 2013）：设测量角度范围为 $\Theta \subset [0,\pi)$，则从有限角度数据 $\mathcal{R}_\Theta u$ 能稳定恢复的波前集为：
    > 
    > $$WF_{\text{visible}}(u) = \{(\mathbf{x},\xi) \in WF(u) : \xi/|\xi| = (\cos\theta,\sin\theta),\ \theta \in \Theta\}$$
- 待确认要点（bib note 摘录）：⚠待补：正文 16C 作 "Greenleaf & Uhlmann (1989)"，卷页请核对},

### `gutha2025mapga`
- 正文写法：**Gutha et al**（年：2025）
- 定位：`ch13/13.3#539` → `book/manuscript/ch13/13.3 近似方法分类与DPS深度剖析.md`
  - 上下文摘录：
    > DMPlug 本质是把扩散模型当**参数化生成器** `G_θ`，把条件生成变成生成器隐空间搜索——和 GAN 隐空间优化（如 Image2StyleGAN）思路完全一致。
    > 
    > ### MAP-GA：基于一致性模型重参数化的 MAP 估计（Gutha et al., WACV 2025）
    > 
    > MAP-GA 从 MAP 视角重看隐空间优化，用**一致性模型（Consistency Model）`f_θ`** 对扩散采样映射 `G_θ` 重参数化，使 MAP 优化既高效又有严格动机。
- 待确认要点（bib note 摘录）：⚠待补：正文 13.3#539 "Gutha et al. (WACV 2025) MAP-GA"；会场按正文取 WACV 2025；确切篇名/作者未能联网坐实，待核实}

### `habring2025discretization`
- 正文写法：**Habring et al**（年：2025）
- 定位：`ch05/5.7#56` → `book/manuscript/ch05/5.7 本章小结：去噪器即隐式先验即得分估计器.md`
  - 上下文摘录：
    > **一句话总结全章**：把整章压缩成一句话——**你不需要知道"自然图像长什么样"的公式，只要有一个会去噪的网络，Tweedie 等式就能把它翻译成得分函数，进而驱动采样——这就是"隐式先验"思想的核心所在。**
    > 
    > **来源**：本章综合；Pereyra L1/L3；Pock L2；Ratti；Habring et al. (2025)。
- 待确认要点（bib note 摘录）：⚠待补：正文 5.7#56 / 7.5#146 "Habring et al. (2025)"（数值离散化语境）；与 Habring, Falk, Zach & Pock 2025 是否同篇须核}

### `holden2022langevin`
- 正文写法：**Holden, Pereyra & Zygalakis**（年：2022）
- 定位：`ch02/2.4#293` → `book/manuscript/ch02/2.4 从显式先验到隐式先验.md`
  - 上下文摘录：
    > 将 $(\mathbf{z}|\mathbf{y})$ 通过 $\nu_\theta$ 推送，即得到图像空间的后验 $(\mathbf{x}|\mathbf{y})$。Holden, Pereyra 和 Zygalakis（2022）证明此后验在 Hadamard 意义下适定，且具有限矩——为生成式先验的贝叶斯推断提供了理论基础。
- 待确认要点（bib note 摘录）：⚠待补：正文 2.4#293 "Holden, Pereyra & Zygalakis (2022)"；篇名须核}
- ✅ **2026-10-21 平反恢复**：初判"废弃"有误。经 arXiv API（`all:Holden AND all:Zygalakis` → arXiv:2103.10182）与 OpenAlex（W4283643017）坐实——**Matthew Holden, Marcelo Pereyra, Konstantinos C. Zygalakis, "Bayesian Imaging with Data-Driven Priors Encoded by Neural Networks", SIAM Journal on Imaging Sciences 15(2):892–924, 2022**, doi:10.1137/21m1406313（arXiv 2021 预印本的正式版）。该文摘要"证明后验分布及后验矩的存在性与适定性"与正文断言逐字对应，且同节后续正是其 MNIST 实验；正文年份（2022）与正式版一致。**处理**：与 9.4#212 的 holden2022unknown 合并为同一键 `holden2022datadriven`，映射表 #152/#153 两行均指向该键；正文无需改动。

### `hudson1978sure`
- 正文写法：**Hudson**（年：1978）
- 定位：`ch17/17A#97` → `book/manuscript/ch17/附录17A SURE的无偏性证明.md`
  - 上下文摘录：
    > 最后把地基拓宽：Stein 的"翻译"并非高斯独有，指数族噪声都配有各自的版本，散度修正项随之换成对应噪声统计量。
    > 
    > Hudson (1978) 把 Stein 引理推广到指数族噪声。对 Poisson-Gaussian 噪声 $\mathbf{y} \sim \text{Poisson}(\mathbf{x}) + \mathcal{N}(0, \sigma^2 I)$，SURE 形式为：
    > 
    > $$\mathcal{L}_{\text{PG-SURE}}(\mathbf{y}, f) = \|\mathbf{y} - f(\mathbf{y})\|^2 + 2\text{Cov}(\mathbf{y}, f(\mathbf{y})) - \text{Var}(\mathbf{y})$$
- 待确认要点（bib note 摘录）：⚠待补：正文作 "Hudson (1978)"，请确认是否指此文（SURE 的无偏估计）},

### `kobler2020vargauss`
- 正文写法：**Kobler, Effland, Kunisch & Pock**（年：2020）
- 定位：`ch02/2.4#204` → `book/manuscript/ch02/2.4 从显式先验到隐式先验.md`
  - 上下文摘录：
    > 此外，FoE 的双层优化训练计算量大（Chen 等人报告训练耗时两周），且学出的正则项通常非凸，给优化带来额外困难。
    > 
    > 作为 FoE 的进一步发展，Kobler, Effland, Kunisch 和 Pock（2020）提出 **Total Deep Variation（TDV）**——基于 CNN 的学习型正则项：
    > 
    > $$\text{TDV}(\mathbf{x}, \theta) = \sum_{i,j} r_{i,j}(\mathbf{x}, \theta)$$
- 待确认要点（bib note 摘录）：⚠待补：正文 2.4#204 "Kobler, Effland, Kunisch & Pock (2020)"；注意同名 'The Variational Gaussian Approximation Revisited' 实为 Opper & Archambeau (Neural Computation 2009) 而非本文；Kobler 等 2020 应为 deep image prior 相关论文，确切篇名/会场须作者确认}

### `laumont2022pnpdm`
- 正文写法：**Laumont et al**（年：2021）
- 定位：`ch03/3.7#182` → `book/manuscript/ch03/3.7 从MAP到后验：点估计的局限与分叉.md`
  - 上下文摘录：
    > 不过 PnP 与采样有深层联系：近端算子=MAP 去噪器→学习去噪器→得分函数，这条链在第5章 Tweedie 等式闭合，PnP-ULA 把 PnP 从优化推广到采样。
    > 
    > > **澄清：PnP 有两种形态（Pereyra, Lecture 3）**
    > > 3.5 见到的是 **PnP-ADMM/FISTA**：替换近端算子，仍在**优化框架**内，给"增强的 MAP"（点估计，无不确定性）。但 PnP 还有第二种真正做后验采样的形态（Laumont et al. 2021）：
    > > - **PnP-ULA**：把去噪器 $D_\varepsilon$ 当得分代理，加朗之万更新：
    > >   $$X_{k+1}=X_k+\delta_k\nabla\log p(\mathbf{y}|X_k)+\frac{\delta_k}{\varepsilon}\big(D_\varepsilon(X_k)-X_k\big)+\frac{\delta_k}{\lambda}\big(\Pi_C(X_k)-X_k\big)+\sqrt{2\delta_k}\,Z_{k+1}$$
    > > - **PnP-SGD**：去掉投影项，仅留去噪器得分与噪声。
- 定位：`ch13/13.3#601` → `book/manuscript/ch13/13.3 近似方法分类与DPS深度剖析.md`
  - 上下文摘录：
    > ### PnP-DM：Gibbs 采样 + 扩散去噪
    > 
    > PnP-DM（Laumont et al., 2022）把经典 Gibbs 采样与扩散去噪结合，交替更新两个条件分布逼近后验：
    > 
    > **似然步**：固定图像 `x_0`，从似然约束分布采中间变量 `z`：
- 待确认要点（bib note 摘录）：⚠待核：正文 Laumont et al. (2021, 3.7#182) 与 (2022, 13.3#601) 合并；题名/四作者已联网坐实（Beyond Minimization，后发 SIAM J. Imaging Sci. 2023），唯确切 arXiv 号与卷页未坐实；若 13.3#601 实指另一篇 'Diffusion Plug-and-Play'（作者组合不同）请另立条目核实}

### `pereyra2019proximal`
- 正文写法：**Pereyra**（年：2019）
- 定位：`ch03/3.1#69` → `book/manuscript/ch03/3.1 MAP估计：从后验众数到优化问题.md`
  - 上下文摘录：
    > 它衡量 $\phi$ 在 $u$ 处的值，和它用 $\mathbf{x}$ 点的一阶近似去估的差——非负（凸性保证），但一般**不对称**。
    > 
    > 有个非常漂亮的结论（Pereyra, 2019）：
    > 
    > - **MAP** 是在 Bregman 散度损失下的贝叶斯最优估计器：
    >   $$\hat{x}_{\text{MAP}} = \arg\min_{u} \, \mathbb{E}\big[D_{\phi}(u, \mathbf{x}) \,\big|\, \mathbf{y}\big]$$
- 待确认要点（bib note 摘录）：⚠待补：正文 3.1#69 "Pereyra (2019)"（MAP/近端）；与 Durmus 2018 区分，具体篇/年须核}

### `tachella2023me`
- 正文写法：**Tachella et al**（年：2023）
- 定位：`ch17/17.5#182` → `book/manuscript/ch17/17.5 等变成像：从不完整测量中学习.md`
  - 上下文摘录：
    > - 例子：循环卷积关于循环平移等变；图像修复掩码（周期性的）关于平移等变；Radon 变换关于旋转等变。
    > 
    > ### 关键定理（Tachella et al. 2023, JMLR）
    > 
    > 等变成像的满秩条件要求 $A$ **不是** $G$-等变的：
- 待确认要点（bib note 摘录）：⚠待补：正文 17.5 "Tachella et al. (2023, JMLR)"；未能联网坐实同名 JMLR 2023 论文；首作者应为 Juli{\'a}n Tachella（上一版误作 'J{\'e}r{\'e}my' 已更正，OpenAlex 坐实其 Chen--Tachella--Davies Equivariant Imaging 线）；候选：Chen, Tachella \& Davies, {Equivariant Imaging: Learning Beyond the Range Space} (ICCV 2021) 或 Tachella, Chen \& Davies, {Sensing Theorems for Unsupervised Learning in Linear Inverse Problems} (2023)；篇名/会场须作者确认}

### `tran2021tutorialvb`
- 正文写法：**Tran et al**（年：2021）
- 定位：`ch08/8.5#113` → `book/manuscript/ch08/8.5 从变分推断到生成模型：路线图.md`
  - 上下文摘录：
    > 这还提醒我们全书的深层叙事：所谓"变分路径"并不只是一堆新算法的堆叠，而是"从观测推断未知"这条古老主线在"用优化刻画分布"这一新姿势下的延续。到了第 12 章，它跟"从分布中采样"的采样路径在同一目标上相遇，恰好证明了两者在数学上是一体的。
    > 
    > **来源**：Kingma & Welling (2014); Rezende et al. (2014); Kingma et al. (2021) Variational Diffusion Models; Tran et al. (2021) A Tutorial on Variational Bayes; 第1-7章叙事
    > 
    > ---
- 待确认要点（bib note 摘录）：⚠待补：正文 8.5#113 "Tran et al. (2021)"；作者/venue(JASA) 已核实，online-first 2021、正式卷期约为 JASA 118(542) (2023)；卷期页码请最终核对}

### `wu2024smcdiff`
- 正文写法：**Wu et al**（年：2024）
- 定位：`ch13/13.3#630` → `book/manuscript/ch13/13.3 近似方法分类与DPS深度剖析.md`
  - 上下文摘录：
    > SMC 理论保证：**粒子数 `N → ∞` 时，粒子集经验分布收敛到真实后验 `p(x_{t-1}|y)`**。这是渐近精确方法的核心优势——计算资源越多精度越高。
    > 
    > MCGDiff（Cardoso et al., 2023）是 SMC 在条件扩散的具体实现，用预训练扩散逆向核作提议分布，用观测 `y` 算粒子权重。SMCDiff（Wu et al., 2024）进一步改进提议分布设计，使粒子传播更高效。
    > 
    > ### DPS 与渐近精确方法的精度-效率权衡
- 待确认要点（bib note 摘录）：⚠待补：正文 13.3#630 "Wu et al. (2024) SMCDiff"；候选篇名如上，会场/作者未能联网坐实，待作者核实}

### `xu2025dmap`
- 正文写法：**Xu et al**（年：2025）
- 定位：`ch13/13.3#745` → `book/manuscript/ch13/13.3 近似方法分类与DPS深度剖析.md`
  - 上下文摘录：
    > 截至2025年中，文献在本书核心方法之上涌现多条主线（Chung et al., 2025 系统梳理）：
    > 
    > **DPS 家族的改进**。经典 DPS 被指出更像 MAP 估计而非后验采样器，多种改进旨在更接近后验或提升质量：
    > - **DMAP**（Xu et al., 2025）：多步梯度使 DPS 行为更接近 MAP 近似，提升性能；
    > - **DPPS**（Wu et al., 2024）：每步去噪生成多个候选，仅留数据一致性最大的，降方差；
    > - **MPGD**（He et al., 2024）：借自编码器把 DPS 梯度投影到数据流形上；
    > - **DSG**（Yang et al., 2024）：施加球约束使步进保持在含噪流形上。
- 待确认要点（bib note 摘录）：⚠待补：正文 13.3#745 "Xu et al. (2025) DMAP"；候选为 Tongda Xu 等 'Rethinking Diffusion Posterior Sampling'（ICLR 2025）但未能坐实其算法即称 DMAP，篇名/会场/作者待作者核实}

## 三、作者名单待补全（1 键）

### `bubba2019learning`
- 正文写法：**Bubba et al**（年：2019）
- 定位：`ch16/16.2#85` → `book/manuscript/ch16/16.2 CT的不适定性与正则化重建.md`
  - 上下文摘录：
    > > 从更深层次看，"能恢复什么"由测量与先验共同决定：测量决定了哪些波前集方向**被触碰过**，先验决定了这些触碰如何被解读。只要先验仍只约束"可见"方向，不可见方向的缺憾就无从弥补——除非先验本身携带了数据之外的信息。这正是下一段剪切波"学习"不可见边缘的真正含义，也提前预告了 16.4、16.5 更大规模的"用数据补全先验"。
    > 
    > **剪切波学习不可见边缘**：剪切波（shearlet，一种方向敏感的多尺度表示）能把系数拆成"可见"和"不可见"两组。Bubba et al. (2019) 发现：可见部分能从数据稳定恢复，不可见部分可以**从训练数据里学出来**（数据驱动的统计模型去"猜"）。把猜到的不可见边缘和 FBP 的可见边缘合并，结果比任何纯重建方法都好。这就是"从显式先验到学习型先验"在 CT 里的具体演出——和第 2 章 2.4 完全同构。
    > 
    > ## 16.2.4 正则化谱系：选什么得看不适定有多重
- 待确认要点（bib note 摘录）：⚠待补：作者名单请按原文补全（16.2#85）}
---

## 2026-10-21 复核修订日志（citekey 规范化 + 误判平反）

> 背景：对盘复核 Phase 3 交付物时发现 ① holden2022langevin 废弃存疑、② `validate_bib.py` 不存在、③ 映射表计数口径不一、④ 15 个 `*unknown` 占位键名。处理如下：

1. **holden2022langevin 平反**（原"废弃"有误）：经 arXiv:2103.10182 + OpenAlex W4283643017 坐实为 Holden, Pereyra \& Zygalakis, *Bayesian Imaging with Data-Driven Priors Encoded by Neural Networks*, SIAM J. Imaging Sci. 15(2):892--924 (2022)。与 holden2022unknown 为同一篇，合并为 `holden2022datadriven`；正文 2.4#293 与 9.4#212 两处引用无需改动。
2. **tachella2025unknown 合并**：与 17.4#301 的 `tachella2025unsure` 为同一篇 UNSURE（ICLR 2025），bib 条目删除，映射表 #207/#208 统一指向该键。
3. **新增 `book/tools/validate_bib.py`**：bib 解析/重复键/花括号配平/citekey 规范/必填字段/活跃⚠/映射表与 CSV 双向覆盖五项校验；首跑即抓获并修复 `venkatakrishnan2013red` 类型标错（@article → @inproceedings，GlobalSIP 2013）。现全部通过。
4. **14 个 `*unknown` 键规范更名**（内容不变，仅键名；映射表同步）：

   | 旧键 | 新键 | | 旧键 | 新键 |
   |---|---|---|---|---|
   | fox2017unknown | fox2017accelerated | | bellec2021unknown | bellec2021sure |
   | gozcu2018unknown | gozcu2018compressive | | pang2021unknown | pang2021r2r |
   | kumar2018unknown | kumar2018disentangled | | winkler2021unknown | winkler2021reverse |
   | belthangady2019unknown | belthangady2019fluorescence | | zhang2021unknown | zhang2021pnp |
   | narnhofer2019unknown | narnhofer2019inversegan | | terris2022unknown | terris2022radio |
   | holderrieth2024unknown | holderrieth2024genmatch | | monroy2025unknown | monroy2025gr2r |
   | （holden2022unknown | holden2022datadriven） | | terris2025unknown | terris2025ram |

5. **总键数 197 → 196**；README、缺口报告、映射表计数口径已统一。

6. **正文两处张冠李戴已提前修正**（不等 Phase 5，写作源与成书树同步修改）：
   - 13.3#630 及 #815/#944 两处来源行的 "Wu et al. (2024) SMCDiff" → "Trippe et al. (2022)"（共 3 处；bib 键 `wu2024smcdiff` → `trippe2022smcdiff`，mapping 新增 #212 行）；
   - 12.7#158 与 13.3#742 的 "Chung et al. (2025)" → "Daras et al. (2024)"（原 chung2025survey redirect 落地，mapping #198 行消除、编号留空）。
   - ✅ **DPPS 已坐实（同日后续）**：13.3#746 "DPPS（Wu et al., 2024）" 经作者提供线索 + OpenAlex 核实为 **Hongjie Wu 等 9 人，*Diffusion Posterior Proximal Sampling for Image Restoration*, ACM MM 2024, pp. 214–223**（doi:10.1145/3664647.3681556，arXiv:2402.16907；候选数按信噪比自适应的采样选择策略，与正文描述一致）——该处引用本就正确，新增 bib 键 `wu2024dpps`，mapping #196 转 ✓，⚠ 待核清零，正文无需改动。
   - 扫描重跑：仍 **211 条 / 602 次**（+Trippe 2022 ×3、−Chung 2025 ×2、Wu 2024 余 1、Daras 2024 ×8）；`validate_bib.py` 全通过（0 警告）。
