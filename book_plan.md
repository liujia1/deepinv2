# 《图像生成：从贝叶斯到扩散》写作规划

---

## 一、核心思想

将图像逆问题求解与图像生成技术视为同一本质问题的不同表达。**逆问题求解 = 条件图像生成**。通过贝叶斯推断、得分匹配、扩散采样等框架实现理论统一与实践贯通。

---

## 二、书名策略

主标题"图像生成"为流量入口（市场导向），内容内核为逆问题与生成模型的理论统一性（学术导向）。

- 首选书名：**《图像生成：从贝叶斯到扩散》**
- 副标题可选：——采样与变分两条路径

**市场定位**：面向大众市场的技术读物，书名以"图像生成"等高流量关键词为核心降低认知门槛、提升传播性与销售潜力。扩散模型/图像生成是搜索热词，"逆问题"对大众太学术。策略：**用生成的外壳，讲逆问题的灵魂**——内容不变，换入口。

---

## 三、双路径统一架构

```
           ┌── 采样路径：ULA → Langevin → Score → Diffusion (SDE视角)
贝叶斯框架 ─┤
           └── 变分路径：ELBO → VAE → 层级VAE → Diffusion (似然视角)
```

两条路径均始于贝叶斯框架，终于扩散模型的数学等价性（**Score Matching损失 ≡ 变分下界**），构成全书核心理论主干。第12章"殊途同归"为全书高潮。

---

## 四、关键理论桥梁

1. **PnP去噪器 → 得分函数估计器**（Tweedie等式）：去噪即学习先验梯度
2. **ULA朗之万采样 → 扩散模型**：多步扩散过程是Langevin的推广
3. **"离散→连续→再离散"螺旋**：Langevin(离散迭代) → 扩散SDE(连续方程) → Euler-Maruyama(再离散化) → DDPM(特定离散方案)
4. **VAE编码/解码 → 扩散加噪/去噪**：扩散 = 无穷层级的层级VAE
5. **条件生成 = 逆问题求解**：条件扩散采样完成第1章到第13章的闭环

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

Part II  采样路径
           第4章   MCMC与ULA算法
           第5章   朗之万动力学与得分函数
           第6章   得分匹配：从去噪中学习得分
           第7章   扩散模型：SDE视角（含离散化、概率流ODE、DDPM）

Part III 变分路径
           第8章   变分推断与ELBO
           第9章   VAE与重参数化
           第10章  层级VAE与扩散的变分推导
           第11章  扩散模型：变分视角

Part IV  统一
           第12章  Score ≡ ELBO：殊途同归
           第13章  条件生成与逆问题求解

Part V   前沿
           第14章  Flow Matching与最优传输

Part VI  实践与应用
           第15章  扩散模型的架构实践：UNet → DiT
           第16章  CT/MRI重建
           第17章  自监督学习与等变架构
           第18章  综合项目：用扩散模型求解自定义逆问题
```

### 各章节要点

#### 第7章 扩散模型：SDE视角

```
7.1  从Langevin到扩散：连续时间推广
7.2  正向SDE与逆向SDE
7.3  概率流ODE：随机采样的确定性等价
7.4  数值离散化：从连续方程到可执行算法
     - Euler-Maruyama 方法
     - DDPM 作为 SDE 的离散特例
     - 采样器选择：DDPM vs DDIM vs 概率流ODE
7.5  实践：用离散化SDE实现图像生成
```

#### 第15章 扩散模型的架构实践：UNet → DiT

```
15.1  去噪器架构演进：CNN → UNet → DiT
15.2  DiT的关键设计：Patchify、adaLN-Zero
15.3  扩散 + Transformer的SOTA组合
      （SD3/Flux = Rectified Flow + DiT）
```

---

## 八、材料覆盖评估

### 各 Part 覆盖率

```
Part I   ████████████████████  90%  充足（.m代码完整覆盖经典方法链）
Part II  ██████████████████░░  85%  前半充足，后半需扩展（lab有完整解答版）
Part III ██████████░░░░░░░░░░  45%  有理论基础，需大量推导和代码
Part IV  ████████████░░░░░░░░  50%  两条路径材料已有，统一论证需新写
Part V   ████░░░░░░░░░░░░░░░░  15%  基本需新写
Part VI  ██████████████████░░  85%  实践素材充足（.m + .ipynb）
```

### 逐章材料支撑详情

| 章 | 可支撑的资料 | 覆盖度 |
|---|---|---|
| 第1章 逆问题与贝叶斯推断 | Unit 2讲座(L1&2 84+53页)、Unit 5 Benning讲座(270页)、Winter School Siltanen断层成像(Day1-3 87+174+82+56页+7个.m代码)、Unit 1预课程(Calatroni 59页：正向模型/噪声建模/贝叶斯推导)、Unit 2 exercise(经验贝叶斯) | ✅ 充足 |
| 第2章 先验：贝叶斯推断的灵魂 | Unit 3先验讲座3讲(82+47+54页)、Unit 1 Calatroni(贝叶斯→变分对应表、先验=正则化推导)、Ratti(贝叶斯去噪器/学习误差分解)、Unit 2 exercise(后验推导+MAP+近端算子) | ✅ 充足 |
| 第3章 从MAP到后验探索 | Winter School优化(FISTA/ISTA lab)、Benning正则化3讲(270+71+97页)、.m完整方法链(XR03朴素逆→XR04 SVD→BunnyTomo3截断SVD→XR05 Tikhonov→XR09 TV→XR10 Besov小波ISTA)、MIVA优化lab(ISTA/FISTA)、proximal.m(近端算子)、Unit 1(梯度下降) | ✅ 充足 |
| 第4章 MCMC与ULA | lab1_ULA完整notebook+**解答版lab1_ULA_sol**(1D高斯采样+2D图像实现)、Unit 2讲座Pereyra(Monte Carlo integration) | ✅ 充足 |
| 第5章 朗之万动力学与得分 | lab2_PnP+**解答版lab2_PnP_sol**(Tweedie等式完整推导+ULA采样实现)、ULA lab、Unit 2"stochastic diffusion processes"、Unit 1 Calatroni(近端算子=去噪器，PnP前传) | ✅ 充足 |
| 第6章 得分匹配 | PnP lab的Tweedie identity、Unit 3"explicit diffusion models"(82页含扩散模型内容)、MiniProject_DenoisingPrior(PnP vs 扩散对比)、Ratti(Tweedie's formula用于无监督先验学习) | 🟡 需扩展：缺DSM/SSM正式推导和训练代码 |
| 第7章 扩散SDE视角 | Unit 2 Pereyra"stochastic diffusion processes"(84+53页)、Unit 3"explicit diffusion models based on products of 1D Gaussian mixture models"、deepinv库(diffusion_sde/demo_ddrm/demo_diffpir) | 🟡 需扩展：缺SDE→ODE→DDPM完整推导链 |
| 第8章 变分推断与ELBO | Unit 5 Benning正则化理论(变分框架270页)、Winter School凸分析(Fenchel共轭65+71页)、Unit 2 Pereyra贝叶斯决策理论 | 🟡 有基础：需自行推导ELBO→VAE桥梁 |
| 第9章 VAE与重参数化 | Unit 1 Ratti(神经网络架构：MLP/CNN/UNet/**ViT**、训练流程、偏差-方差)、Unit 3"learning generative priors"(最大熵原理) | 🟡 有基础：VAE架构和重参数化需新写 |
| 第10章 层级VAE | Unit 3"Gaussian mixture models"提示层级结构 | 🟠 弱支撑：需大量新内容 |
| 第11章 扩散变分视角 | Unit 3"explicit diffusion models"、已有SDE视角可对照 | 🟠 弱支撑：Kingma VDM推导需新写 |
| 第12章 Score ≡ ELBO | 已有Tweedie identity(PnP lab)+变分下界自然汇合 | 🟡 需论证：等价性证明需自行组织 |
| 第13章 条件生成与逆问题 | MiniProject_DenoisingPrior(**PnP vs 扩散模型完整对比实践**)、Unit 4 learned iterative reconstruction(LGS notebook)、astra实践 | 🟡 需扩展：条件扩散引导需新写 |
| 第14章 Flow Matching | Winter School内点法讲座(Gondzio 46+64页)中ADMM/最优传输仅提及 | 🔴 基本需新写 |
| 第15章 架构实践：UNet→DiT | Bologna_UNet_example(UNet断层成像)、CompImLab25(训练UNet去噪器完整流程)、Unit 1 Ratti(**ViT/Restormer架构介绍**) | 🟡 UNet充足，ViT有铺垫，DiT细节需新写 |
| 第16章 CT/MRI重建 | astra_operators_example(ASTRA断层成像)、Bologna_UNet_example(UNet重建)、Bologna_LGS_example(Learned Gradient Descent)、.m代码(XR系列完整断层重建链) | ✅ 充足 |
| 第17章 自监督学习 | MiniProject_Self_Supervised(**Cryo-EM真实数据自监督去噪**，deepinv库，Noise2Noise等) | ✅ 充足 |
| 第18章 综合项目 | MiniProject_DefiningOperator(多视角自定义算子)、MiniProject_DenoisingPrior(扩散求解逆问题)、Unit 6评估材料 | 🟡 有框架，需设计 |

### 资源清单（按类型）

#### PDF讲座（已转为.md，共35个）

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
| 条件扩散/引导采样 | 第13章 | Dhariwal & Nichol (classifier guidance)；Ho & Salimans (classifier-free) |
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
| 实验4.1 | 1D高斯分布ULA采样 | ULA递推式；Euler离散化Langevin SDE | lab1_ULA_sol（ULA_gauss函数） | ✅ |
| 实验4.2 | ULA步长δ对收敛的影响 | 步长选择；δ≤1/L条件；收敛性与偏差 | lab1_ULA_sol（实验δ=0.1,0.5,1.0） | ✅ |
| 实验4.3 | 2D图像ULA后验采样（去卷积） | 高维ULA；后验分布采样；势能函数U=-log后验 | lab1_ULA_sol（2D实验部分） | ✅ |
| 实验4.4 | MCMC收敛诊断：自相关与有效样本量 | MCMC收敛诊断；burn-in；自相关函数；ESS | lab1_ULA扩展 | 🔄 需补充 |

#### 第5章 朗之万动力学与得分函数

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验5.1 | Tweedie等式验证：去噪器→得分函数 | Tweedie等式∇log p_ε(x)=(D_ε(x)-x)/ε；得分函数 | lab2_PnP_sol（PnP ULA递推） | ✅ |
| 实验5.2 | PnP-ULA后验采样与不确定性量化 | PnP框架；用去噪器替换先验梯度；后验采样→不确定性量化 | lab2_PnP_sol（去卷积PnP采样） | ✅ |
| 实验5.3 | 近端算子 vs 学习去噪器：PnP中的先验替换 | 近端算子prox_λR→去噪器D_ε；显式先验→隐式先验 | proximal.m思想 + lab2_PnP | 🔄 需扩展 |

#### 第6章 得分匹配：从去噪中学习得分

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验6.1 | 训练一个UNet去噪器（DRUNet） | 去噪器作为得分估计器；条件噪声水平训练 | CompImLab25 Part 3 + MiniProject_DenoisingPrior | ✅ |
| 实验6.2 | 去噪得分匹配（DSM）：从去噪器提取得分 | DSM目标函数；得分匹配与去噪的等价性；s_θ≈∇log p | 基于CompImLab25训练的去噪器 | 🆕 新写 |
| 实验6.3 | 用学习到的得分驱动PnP-ULA采样 | 学习得分→PnP采样；与手工先验对比 | lab2_PnP + 实验6.1的去噪器 | 🔄 组合 |

#### 第7章 扩散模型：SDE视角

| 练习 | 内容 | 对应知识点 | 素材来源 | 状态 |
|---|---|---|---|---|
| 实验7.1 | DDPM正向加噪过程：不同时间步的噪声水平 | 噪声调度α_t, ᾱ_t；信噪比随时间变化 | 🆕 新写（简单，基于DDPM公式） | 🆕 新写 |
| 实验7.2 | DDPM反向去噪采样 | 反向SDE；Euler-Maruyama离散化；去噪采样循环 | deepinv库 demo_diffusion_sde | ✅ |
| 实验7.3 | SDE采样 vs 概率流ODE vs DDIM：质量与速度对比 | 概率流ODE；确定性采样；DDIM加速；采样器权衡 | deepinv库（多个采样器对比） | 🔄 需扩展 |
| 实验7.4 | PnP-ULA vs 扩散模型：同一去卷积问题的对比 | 采样路径的终点：Langevin→扩散的自然升级 | MiniProject_DenoisingPrior | ✅ |

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
| 实验10.1 | 实现层级VAE（2-3层） | 层级潜变量；马尔可夫推断链；层级ELBO | 🆕 新写（在实验9.1基础上扩展） | 🆕 新写 |
| 实验10.2 | 层级VAE与扩散加噪的类比：观察L→∞的极限 | 层级VAE→扩散的极限关系；高斯编码器=加噪过程 | 🆕 新写 | 🆕 新写 |

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
| 实验14.1 | 2D点云Flow Matching（教学演示） | 向量场学习；条件Flow Matching；OT-CFM | 🆕 新写 | 🆕 新写 |
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
| 实验16.2 | UNet端到端CT重建 | 端到端学习重建；监督训练；post-processing | Bologna_UNet_example | ✅ |
| 实验16.3 | Learned Gradient Descent迭代重建 | 学习型迭代重建；算法展开；unrolled optimization | Bologna_LGS_example | ✅ |
| 实验16.4 | 扩散先验CT重建 | 扩散模型作为CT重建先验；DiffPIR for CT | MiniProject_DenoisingPrior方法迁移到CT | 🔄 需扩展 |

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
| ✅ 有现成素材 | 20 | 38% |
| 🔄 有素材需改造/扩展 | 17 | 33% |
| 🆕 需从零新写 | 15 | 29% |
| **总计** | **52** | 100% |

### 新写练习重点清单（15个）

| 优先级 | 练习 | 对应知识点 | 难度 | 依赖 |
|---|---|---|---|---|
| ⭐⭐⭐ | 实验6.2 DSM：从去噪器提取得分 | DSM目标函数；得分匹配与去噪的等价性 | 中 | 实验6.1去噪器 |
| ⭐⭐⭐ | 实验9.1 实现简单VAE(MNIST) | 编码器-解码器架构；ELBO训练；重参数化 | 中 | PyTorch基础 |
| ⭐⭐⭐ | 实验10.1 层级VAE | 层级潜变量；马尔可夫推断链；层级ELBO | 中高 | 实验9.1 |
| ⭐⭐⭐ | 实验11.2 简化VLB训练扩散 | 简化VLB；噪声预测参数化；DDPM训练=VLB简化 | 高 | 实验10.1 |
| ⭐⭐⭐ | 实验12.1 DSM≡VLB数值验证 | Score Matching≡变分下界；殊途同归的数值验证 | 高 | 实验6.2+11.1 |
| ⭐⭐ | 实验8.1 1D高斯混合ELBO | ELBO定义；Jensen不等式；KL散度 | 低 | 无 |
| ⭐⭐ | 实验8.3 变分推断1D案例 | 变分族q；平均场近似；变分间隙 | 低 | 实验8.1 |
| ⭐⭐ | 实验9.2 重参数化数值验证 | 梯度穿过随机节点；REINFORCE vs 重参数化 | 低 | 实验9.1 |
| ⭐⭐ | 实验9.3 VAE隐空间可视化 | 隐空间结构；KL正则化；插值生成 | 低 | 实验9.1 |
| ⭐⭐ | 实验13.2 Classifier-free guidance | 引导采样；无分类器引导；引导权重w | 中高 | 实验7.2扩散模型 |
| ⭐ | 实验7.1 DDPM正向加噪 | 噪声调度α_t, ᾱ_t；信噪比随时间变化 | 低 | 无 |
| ⭐ | 实验10.2 层级VAE→扩散极限 | 层级VAE→扩散极限；高斯编码器=加噪 | 中 | 实验10.1 |
| ⭐ | 实验11.1 VLB数值计算 | VLB分解；重建项+先验匹配项 | 中 | 实验9.1 |
| ⭐ | 实验14.1 2D点云Flow Matching | 向量场学习；条件Flow Matching；OT-CFM | 中 | 无 |
| ⭐ | 实验14.2 Rectified Flow图像生成 | Rectified Flow；直线插值 vs 扩散路径 | 高 | 实验14.1 |

---

## 十一、写作要点

1. **第6章与第9章互相呼应**：采样路径学到第6章（去噪=学习得分），变分路径学到第9章（VAE编码=隐空间映射），两者都是"用神经网络逼近概率分布"，但手段不同
2. **第12章等价性证明放正文不放附录**：这是全书最精彩的"收网"时刻，要用直观推导+形式化证明双写
3. **每条路径内部要有"实践锚点"**：
   - 采样路径：第4章实现ULA，第6章实现去噪得分匹配
   - 变分路径：第9章实现VAE，第10章实现简化版VDM
4. **第3章结尾设分叉点**：提出核心问题"如何处理复杂后验？"，引出两个方向：采样还是近似？
5. **第13章完成闭环**：回到第1章的逆问题，但现在有了扩散这个强大的后验采样器

---

## 十二、整体叙事逻辑图

```
Part I   贝叶斯基石
          │
          ├────────────────────────────┐
          ▼                            ▼
Part II  采样路径                  Part III 变分路径
         ULA→Langevin→Score       ELBO→VAE→层级VAE
          │                            │
          ▼                            ▼
       Diffusion(SDE)             Diffusion(VLB)
          │                            │
          └──────────┬─────────────────┘
                     ▼
Part IV   统一：Score ≡ ELBO
          条件生成 = 逆问题求解 ←── 闭环第1章
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
Part V   Flow      Part VI    实践
         Matching   架构应用    项目
```
