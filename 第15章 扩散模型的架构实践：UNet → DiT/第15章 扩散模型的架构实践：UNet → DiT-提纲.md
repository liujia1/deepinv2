# 第15章 扩散模型的架构实践：UNet → DiT — 提纲

## 本章定位

第15章是全书从理论到实践的桥梁——前14章完成了从逆问题到扩散模型再到Flow Matching的完整理论构建，读者已经理解了扩散模型"学什么"（得分函数/向量场）和"怎么学"（DSM/VLB/CFM），但尚未系统回答"用什么网络去学"这一关键工程问题。本章聚焦去噪器的架构设计：从经典CNN到UNet，再到DiT，系统梳理扩散模型去噪网络的架构演进。核心论点是：**架构即先验——去噪器的架构选择直接影响所学隐式先验的质量，进而决定生成和逆问题求解的效果**。

**核心论点**：扩散模型的去噪器 $s_\theta(x_t, t)$ 或 $\epsilon_\theta(x_t, t)$ 不仅是一个函数逼近器——它的架构选择决定了所学隐式先验的表达能力。UNet通过编码器-解码器+跳跃连接实现了多尺度特征提取，是扩散模型前十年的标准架构；DiT用Transformer替代CNN，以全局自注意力突破局部感受野的限制，在大规模数据和参数下展现出可预测的缩放规律。时间步嵌入机制是扩散去噪器区别于普通去噪器的关键设计——它让同一个网络在不同噪声水平下表现出不同的去噪策略。

**与前章衔接**：第14章14.5节简要介绍了SD3/Flux = Rectified Flow + DiT的组合，但未展开DiT的架构细节。本章从架构角度深入剖析UNet和DiT的设计原理，回答14.5节留下的核心问题："为什么Transformer架构比UNet更适合Flow Matching？"同时，第6章6.6节提到DRUNet条件噪声水平设计和三种参数化（ε/s/x₀预测），本章补充这些设计的网络层面实现细节。

**与后章衔接**：第16-18章的实践应用将使用本章介绍的架构。理解UNet和DiT的设计原理，是正确实现扩散模型求解逆问题的基础。

---

## 叙事弧

```
经典架构（CNN/UNet如何去噪？）→ 时间步条件（如何感知噪声？）→
Transformer革命（ViT→DiT如何替换UNet？）→ 架构选择（UNet vs DiT？）→
训练实践（如何训练好一个去噪器？）
```

理解现状→补充关键机制→架构演进→实践选择→工程落地

---

## 章节结构

### 15.0 本章导读：架构即先验——去噪器的设计如何影响生成质量

**核心观点**：前14章回答了扩散模型"学什么"和"怎么学"——得分函数、向量场、DSM、VLB、CFM。但有一个隐含的前提始终未被展开：**用什么网络去学？** 去噪器 $D_\theta(x_t, t)$ 的架构选择不是一个工程细节——它决定了所学隐式先验 $p_\theta(x)$ 的表达能力。一个容量不足或设计不合理的去噪器，无论训练目标多么精确，都无法学到高质量的先验。本章系统回答这个被理论章节略过的实践问题。

- 去噪器的双重身份：函数逼近器 + 隐式先验的载体
- 从第5-6章的回顾：Tweedie等式将去噪器与得分函数等价，DSM将训练去噪器与学习得分等价——但去噪器本身长什么样？
- 架构演进的三个阶段：CNN → UNet → DiT
- 章节导航：经典架构→时间步嵌入→Transformer革命→架构选择→训练实践

**来源**：全书叙事；book_plan.md 第五章定位

---

### 15.1 去噪器的经典架构：CNN与UNet

**核心观点**：去噪器的核心任务是映射含噪图像 $x_t$ 到干净估计 $\hat{x}_0$（或噪声估计 $\hat{\epsilon}$）。CNN利用卷积的局部感受野和参数共享，天然适合图像处理；UNet在CNN基础上引入编码器-解码器结构和跳跃连接，实现了多尺度特征的提取与融合，成为扩散模型第一个十年的标准去噪架构。

#### 15.1.1 CNN去噪器：DnCNN

- **图像去噪作为回归问题**
  - 输入：含噪图像 $y = x + \epsilon$，输出：去噪估计 $\hat{x} = D_\theta(y)$
  - 训练目标：$\min_\theta \mathbb{E}[\|D_\theta(y) - x\|^2]$（监督学习）
  - 回顾第9章9.4节四种学习设定：这里属于"自监督"设定（已知噪声模型的干净数据）
- **DnCNN架构**（Zhang et al., 2017）
  - 核心设计：堆叠"Conv + BN + ReLU"残差块
  - 残差学习：网络预测噪声 $\hat{\epsilon} = D_\theta(y)$，去噪结果 $\hat{x} = y - \hat{\epsilon}$
  - 为什么残差学习更有效：噪声通常比信号更"简单"（接近高频），学习噪声比学习干净图像更容易
  - 感受野堆叠：17层Conv(3×3) → 有效感受野约35×35
- **CNN的局限**
  - 感受野有限：局部卷积核无法捕获长程依赖
  - 单尺度处理：所有卷积在同一分辨率上操作，缺乏多尺度信息
  - 对大尺度结构建模不足：去噪效果好，但对图像全局结构的理解有限

#### 15.1.2 UNet：编码器-解码器+跳跃连接

- **UNet的诞生**（Ronneberger et al., 2015）
  - 原始动机：医学图像分割——需要精确的像素级输出
  - 核心思想：编码器提取多尺度特征，解码器逐步恢复分辨率，跳跃连接传递细节
- **编码器路径**
  - 逐层下采样：分辨率减半，通道数加倍
  - 每层：Conv → BN → ReLU → Conv → BN → ReLU → MaxPool
  - 功能：提取从细粒度到粗粒度的多尺度特征
- **解码器路径**
  - 逐层上采样：分辨率加倍，通道数减半
  - 每层：Upsample（或ConvTranspose2d）→ Conv → BN → ReLU → Conv → BN → ReLU
  - 功能：从粗粒度特征逐步恢复到像素级输出
- **跳跃连接（Skip Connections）**
  - 编码器第 $l$ 层的特征图 → 拼接（concatenate）到解码器第 $l$ 层
  - 为什么重要：解码器在恢复分辨率时容易丢失空间细节，跳跃连接将编码器的高分辨率特征直接传递过来
  - 直觉：编码器是"分析师"（提取抽象特征），解码器是"重建师"（恢复空间细节），跳跃连接是"分析师给重建师递便签"
- **UNet的关键优势**
  - 多尺度特征融合：同时利用细粒度和粗粒度信息
  - 梯度流通畅：跳跃连接缓解深层网络的梯度消失问题
  - 参数高效：编码器-解码器的对称设计使得参数量可控

#### 15.1.3 UNet在扩散模型中的角色

- **从去噪器到扩散去噪器**
  - 回顾第6章6.6节：DRUNet——条件噪声水平的UNet去噪器
  - 普通UNet vs 扩散UNet：关键区别在于时间步条件的注入（15.2节详述）
  - 扩散UNet的输入：$(x_t, t)$，输出：$\hat{\epsilon}_\theta(x_t, t)$ 或 $\hat{x}_\theta(x_t, t)$
- **DDPM中的UNet**（Ho et al., 2020）
  - 架构：基于OpenAI的Guided Diffusion UNet
  - 关键改进：注意力机制在低分辨率层引入、时间步嵌入注入
  - 训练：噪声预测参数化 $\epsilon_\theta(x_t, t)$
- **Score-SDE中的NCSNpp**（Song et al., 2021）
  - 架构：连续时间的UNet变体
  - 关键改进：连续时间条件、多尺度噪声水平的统一处理
- **UNet作为扩散去噪器的实践代码**（参考CompImLab25 Part 3 + Bologna_UNet_example）
  - UNetMini架构示意：编码器（1→32→64）+ 解码器（64→32→1）
  - 训练流程：数据加载 → 加噪 → 前向传播 → 计算MSE损失 → 反向传播 → 更新参数
  - 典型超参数：Adam优化器，lr=1e-3，20-50 epochs

UNet之所以能成为扩散模型的标准去噪架构，不仅因为其编码器-解码器+跳跃连接的设计，更因为一个被我们在前14章中反复使用但从未展开的关键机制——**时间步嵌入**。它让同一个网络在不同噪声水平下表现出不同的去噪策略。

**来源**：Zhang et al. (2017) DnCNN; Ronneberger et al. (2015) UNet; Ho et al. (2020) DDPM; Song et al. (2021) Score-SDE; Ratti P35-37 (UNet去噪示例); CompImLab25 Part 3; Bologna_UNet_example

---

### 15.2 时间步嵌入：让网络感知噪声水平

**核心观点**：扩散模型的去噪器与普通去噪器的根本区别在于时间步条件——同一个网络需要根据当前噪声水平 $t$ 调整去噪策略。时间步嵌入通过正弦位置编码将标量 $t$ 映射为高维向量，再通过各种注入方式（加法、缩放-偏移、adaLN-Zero）调制网络特征。时间步嵌入的质量直接影响去噪器在不同噪声水平下的表现。

#### 15.2.1 为什么需要时间步条件？

- **扩散去噪器的特殊需求**
  - 普通去噪器：固定噪声水平，$D_\theta(y) \approx x$
  - 扩散去噪器：噪声水平随时间变化，$D_\theta(x_t, t)$ 必须根据 $t$ 调整行为
  - 直觉：$t$ 接近0时（高噪声），需要"大刀阔斧"地去除噪声；$t$ 接近 $T$ 时（低噪声），需要"精雕细琢"地保留细节
- **回顾全书中的时间步依赖**
  - 第6章6.6节：DRUNet的条件噪声水平设计
  - 第7章：不同时间步 $t$ 的得分函数 $\nabla\log p_t(x_t)$ 不同
  - 第11章11.3节：三种参数化（$\epsilon$/$s$/$x_0$预测）都与时间步相关
- **无条件 vs 条件去噪器**
  - 无条件：一个噪声水平训练一个去噪器——代价高、不灵活
  - 条件：一个去噪器处理所有噪声水平——高效、统一

#### 15.2.2 正弦位置编码

- **从NLP到扩散模型**
  - Transformer的位置编码（Vaswani et al., 2017）：将位置索引编码为高维向量
  - 扩散模型的时间步编码：将连续时间步 $t \in [0, T]$ 编码为高维向量
- **正弦位置编码公式**
  $$\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)$$
  - $d$：嵌入维度，$i$：维度索引
  - 每个维度对应不同的频率，形成多尺度的时间步表示
- **为什么用正弦编码？**
  - 频率递减：低频捕获大范围时间变化，高频捕获精细时间差异
  - 相对位置可线性表达：$\text{PE}(t+\Delta t) = M(\Delta t) \cdot \text{PE}(t)$（旋转矩阵）
  - 无需学习：正弦编码是固定的，不增加训练参数
  - 数学性质详见附录15B
- **从正弦编码到时间步嵌入**
  - 正弦编码 $\text{PE}(t) \in \mathbb{R}^d$ → MLP → 时间步嵌入 $e(t) \in \mathbb{R}^d$
  - MLP的作用：将固定编码映射为可学习的、任务相关的时间步表示
  - DDPM的实现：$\text{Embedding}(t) = \text{MLP}(\text{PE}(t))$

#### 15.2.3 条件注入方式：从加法到调制

- **方式一：加法注入**
  - 将时间步嵌入 $e(t)$ 加到网络某一层的特征上
  - $h' = h + e(t)$
  - 优点：简单；缺点：调制能力弱，仅改变特征的偏移
- **方式二：缩放-偏移注入（FiLM）**
  - 从时间步嵌入生成缩放因子 $\gamma(t)$ 和偏移因子 $\beta(t)$
  - $h' = \gamma(t) \odot h + \beta(t)$
  - 优点：同时调制特征的尺度和偏移，表达力更强
  - FiLM（Perez et al., 2018）：Feature-wise Linear Modulation
- **方式三：adaLN-Zero**
  - 自适应Layer Normalization + Zero初始化
  - $h' = (1 + \gamma(t)) \odot \text{LN}(h) + \beta(t)$
  - 关键：初始化时 $\gamma(t) = 0, \beta(t) = 0$，即初始状态 $h' = \text{LN}(h)$
  - Zero初始化的意义：训练初期网络如同无条件模型，时间步条件的调制从零逐步增强——训练更稳定
  - 这是DiT的核心条件注入方式（15.3节将详细展开）
- **三种注入方式对比**

  | 方式 | 公式 | 参数量 | 调制能力 | 训练稳定性 |
  |---|---|---|---|---|
  | 加法 | $h + e(t)$ | 最少 | 弱 | 一般 |
  | FiLM | $\gamma(t) \odot h + \beta(t)$ | 中等 | 中等 | 较好 |
  | adaLN-Zero | $(1+\gamma(t)) \odot \text{LN}(h) + \beta(t)$ | 中等 | 强 | 最好 |

时间步嵌入解决了"如何让网络感知噪声水平"的问题，但UNet的CNN架构本身有一个根本性局限：局部感受野。无论堆叠多少层卷积，每个像素的"视野"始终有限。而Transformer的自注意力机制提供了一种完全不同的信息聚合方式——每个位置可以直接"看到"整幅图像的所有位置，这一特性使得Transformer成为扩散模型去噪器的下一代架构选择。

**来源**：Vaswani et al. (2017) Attention Is All You Need; Perez et al. (2018) FiLM; Peebles & Xie (2023) DiT; Ho et al. (2020) DDPM; Dhariwal & Nichol (2021) ADM

---

### 15.3 从ViT到DiT：Transformer接管扩散模型

**核心观点**：Vision Transformer（ViT）将图像分割为patch序列并用自注意力处理，证明Transformer可以在图像任务上与CNN竞争。Diffusion Transformer（DiT）将ViT的思路移植到扩散去噪器：Patchify将含噪图像转化为token序列，adaLN-Zero注入时间步条件，Transformer Block处理全局依赖。DiT的核心优势是缩放性——参数量增加时，生成质量遵循可预测的提升曲线。

#### 15.3.1 Transformer基础：自注意力机制

- **自注意力（Self-Attention）**
  - 输入：序列 $X \in \mathbb{R}^{n \times d}$（$n$ 个token，每个 $d$ 维）
  - 查询-键-值机制：
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  - $Q = XW_Q, K = XW_K, V = XW_V$，$W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$
  - 直觉：每个token通过"查询"找到最相关的其他token（通过"键"匹配），然后从"值"中聚合信息
- **多头注意力（Multi-Head Attention, MHA）**
  - 并行执行 $h$ 个注意力头，每个头在不同的子空间上计算
  - 拼接所有头的输出后线性投影
  - 优势：不同头关注不同类型的关系（局部/全局/语义/空间）
- **Transformer Block**
  - 标准结构：LayerNorm → MHA → 残差连接 → LayerNorm → MLP → 残差连接
  - MLP：两层全连接，中间有GELU激活，隐层维度通常为 $4d$
  - 详见附录15A
- **Transformer vs CNN的核心区别**
  - CNN：局部连接（卷积核），权重共享（平移等变性），层次化感受野
  - Transformer：全局连接（自注意力），动态权重（输入依赖），单层即全局

#### 15.3.2 Vision Transformer：图像→Patch序列

- **ViT的核心思想**（Dosovitskiy et al., 2021）
  - 图像不像文本那样天然是序列——ViT通过"Patchify"将图像转化为序列
  - 将 $H \times W \times C$ 图像分割为 $N = (H/p)(W/p)$ 个 $p \times p \times C$ 的patch
  - 每个patch通过线性投影映射为 $d$ 维token
  - 加上可学习的位置编码（保留空间信息）
- **Patchify详解**
  - 输入图像 $x \in \mathbb{R}^{H \times W \times C}$
  - 重排为patch序列 $x_p \in \mathbb{R}^{N \times (p^2 C)}$，$N = HW/p^2$
  - 线性投影：$z_0 = [x_p^1 E; x_p^2 E; \ldots; x_p^N E] + E_{\text{pos}}$
  - $E \in \mathbb{R}^{(p^2 C) \times d}$：patch嵌入矩阵
  - $E_{\text{pos}} \in \mathbb{R}^{N \times d}$：位置编码
- **ViT的缩放规律**
  - 数据量小时：ViT不如CNN（缺乏归纳偏置）
  - 数据量大时：ViT超越CNN（归纳偏置不再是瓶颈，全局注意力的优势显现）
  - 关键洞察：**归纳偏置是"小数据的朋友，大数据的敌人"**
- **从ViT到图像恢复：Restormer**
  - Restormer（Zamir et al., 2022）：为图像恢复任务设计的Transformer变体
  - 关键改进：跨头注意力（transMHA）减少计算量、多尺度特征融合
  - 桥梁地位：Restormer证明了Transformer在像素级图像任务上的可行性，是ViT→DiT之间的关键中间站

#### 15.3.3 DiT：用Transformer替换UNet

- **DiT的核心问题**（Peebles & Xie, 2023）
  - UNet的CNN架构是否不可替代？
  - Transformer能否在扩散去噪任务上与UNet竞争？
- **DiT的架构设计**
  - **Patchify**：与ViT相同，将含噪图像 $x_t$ 分割为patch序列
    - patch大小 $p$ 的选择：$p=2$（高分辨率token，计算量大）vs $p=16$（低分辨率token，计算量小）
    - DiT的patch大小通常为2或4（扩散模型需要保留足够的空间细节）
  - **DiT Block**：在标准Transformer Block基础上加入时间步条件
    - 标准Block：LN → MHA → 残差 → LN → MLP → 残差
    - DiT Block：adaLN-Zero调制 → MHA → 残差 → adaLN-Zero调制 → MLP → 残差
    - 时间步嵌入通过adaLN-Zero调制LayerNorm的缩放和偏移参数
  - **adaLN-Zero的完整实现**
    - 输入：时间步嵌入 $e(t)$ 和条件嵌入 $e(c)$（如类别标签）
    - 从 $[e(t); e(c)]$ 通过线性层预测6个调制参数：$(\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2)$
    - MHA前：$h' = (1 + \gamma_1) \odot \text{LN}(h) + \beta_1$
    - MHA后：$h'' = h + \alpha_1 \odot \text{MHA}(h')$（$\alpha_1$ 初始化为0）
    - MLP前：$h''' = (1 + \gamma_2) \odot \text{LN}(h'') + \beta_2$
    - MLP后：$h'''' = h'' + \alpha_2 \odot \text{MLP}(h''')$（$\alpha_2$ 初始化为0）
    - Zero初始化确保训练初期Block为恒等映射
  - **输出层**：DiT最后一层的输出token序列通过"Unpatchify"（Patchify的逆操作）还原为图像
- **DiT的缩放配置**
  - DiT-S/2, DiT-B/2, DiT-L/2, DiT-XL/2：从小到大的配置
  - 参数量：33M → 130M → 458M → 675M
  - patch大小固定为2，通过调整Transformer的深度（层数）和宽度（隐藏维度）缩放
- **DiT的缩放规律**
  - Gflops与FID强负相关：计算量越大，生成质量越好
  - 参数量与FID的幂律关系：$\text{FID} \propto (\text{params})^{-\alpha}$
  - 意义：可以通过小规模实验预测大规模模型的性能

DiT在ImageNet类别条件生成上证明了Transformer可以替代UNet。但对于扩散模型的实践者而言，更实际的问题是：在具体任务中，应该选择UNet还是DiT？两种架构的设计哲学有何本质区别？

**来源**：Dosovitskiy et al. (2021) ViT; Peebles & Xie (2023) DiT; Zamir et al. (2022) Restormer; Vaswani et al. (2017) Transformer; Ratti P38 (ViT概念)

---

### 15.4 UNet vs DiT：架构选择的艺术

**核心观点**：UNet和DiT代表了两种截然不同的设计哲学——CNN的归纳偏置 vs Transformer的数据驱动学习。UNet在小数据和中等规模下依然有优势（归纳偏置是"小数据的朋友"），DiT在大数据和大规模下展现可预测的缩放优势。对于逆问题求解，UNet在PnP框架中更成熟；DiT在Flow Matching框架中更契合。架构选择应基于数据量、计算预算和任务需求。

#### 15.4.1 设计哲学对比

- **UNet的设计哲学：归纳偏置驱动**
  - 卷积的平移等变性：适合图像的局部结构
  - 层次化感受野：编码器逐层提取从局部到全局的特征
  - 跳跃连接：显式传递多尺度信息
  - 偏置的利弊：小数据下高效（无需从头学习局部结构），大数据下受限（归纳偏置可能不是最优的）
- **DiT的设计哲学：数据驱动学习**
  - 全局自注意力：无先验的空间结构假设，从数据中学习最优的信息聚合方式
  - Patchify：最少的图像先验（仅分割为patch，不假设局部平移等变性）
  - adaLN-Zero：最灵活的条件注入（从零开始学习条件调制）
  - 偏置的利弊：小数据下低效（需要从头学习局部结构），大数据下灵活（不受先验限制）
- **核心洞察**：归纳偏置 vs 数据驱动的权衡
  - **归纳偏置是"小数据的朋友，大数据的敌人"**
  - 当数据量不足以学到好的归纳偏置时，手工设计的偏置优于数据驱动的学习
  - 当数据量足够时，数据驱动学习可以发现比手工设计更优的模式

#### 15.4.2 性能对比与缩放规律

- **DiT vs UNet的定量对比**（Peebles & Xie, 2023; Esser et al., 2024）

  | 模型 | 参数量 | Gflops | FID (ImageNet 256×256) | 采样步数 |
  |---|---|---|---|---|
  | ADM-UNet (Dhariwal & Nichol, 2021) | ~554M | — | 4.59 | 250 |
  | DiT-XL/2 | 675M | 119 | 2.27 | 250 |
  | SD3-MMDiT | 2B+ | — | SOTA | 4-50 |
  | Flux | 12B+ | — | SOTA | 4-50 |

- **缩放规律对比**
  - UNet：参数量增加时性能提升逐渐饱和（CNN架构的归纳偏置成为瓶颈）
  - DiT：参数量增加时性能遵循幂律提升（尚未观测到饱和）
  - SD3的验证：从0.4B到8B参数，验证损失持续下降，FID持续改善
- **计算效率对比**
  - UNet：推理速度快（卷积运算高效），但缩放性差
  - DiT：推理速度较慢（自注意力 $O(n^2)$ 复杂度），但缩放性好
  - 实践折中：DiT在低分辨率特征图上使用自注意力，避免过高的计算代价
- **为何Transformer更适合Flow Matching？**（回答14.5节留下的问题）
  - Flow Matching追求直线路径——一步传输需要全局信息
  - UNet的局部感受野适合逐步扩散的弯曲路径——每步只需局部修正
  - DiT的全局注意力适合少步/单步Flow Matching——一步需要"看到"整个传输路径
  - 实验验证：SD3在4步采样下，DiT + Rectified Flow 远优于 UNet + DDPM

#### 15.4.3 何时选择UNet？何时选择DiT？

- **选择UNet的场景**
  - 数据量有限（< 100K图像）：UNet的归纳偏置提供更好的先验
  - 计算预算有限：UNet推理更快、训练更便宜
  - 传统PnP框架（第5章）：UNet去噪器在PnP-ULA中更成熟
  - 特定领域（医学成像、科学计算）：数据量通常不大，UNet更实用
- **选择DiT的场景**
  - 大规模数据（> 1M图像）：DiT的缩放规律保证性能提升
  - 少步生成需求：DiT + Flow Matching在4-8步即可生成高质量图像
  - 文本条件生成：MMDiT的双模态设计天然支持图文交互
  - 工业级应用：可预测的缩放规律降低试错成本
- **混合架构的可能性**
  - UNet的自注意力变体：在UNet的某些层引入自注意力（如DDPM-UNet）
  - DiT的多尺度变体：在低分辨率层使用Transformer，高分辨率层使用卷积

无论选择UNet还是DiT，一个训练不当的去噪器都无法发挥架构的优势。训练最佳实践是连接"好的架构设计"和"好的生成质量"的最后一英里。

**来源**：Peebles & Xie (2023) DiT; Esser et al. (2024) SD3; Dhariwal & Nichol (2021) ADM; Ho et al. (2020) DDPM

---

### 15.5 训练最佳实践

**核心观点**：好的架构只是起点，好的训练才能让架构发挥潜力。本节汇总扩散去噪器训练中的关键实践：优化器选择（Adam及其变体）、学习率调度（warmup + cosine decay）、初始化策略（Xavier/He初始化、adaLN-Zero的零初始化）、早停与验证（防止过拟合）、以及过参数化与双重下降现象（理解"更大模型不总是更好"的微妙之处）。

#### 15.5.1 优化器与学习率调度

- **优化器选择**
  - SGD + Momentum：简单但需要精心调参
  - Adam（Kingma & Ba, 2015）：自适应学习率，扩散模型训练的标准选择
    - $\theta \leftarrow \theta - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$
    - 一阶矩估计 $m$（动量）+ 二阶矩估计 $v$（自适应步长）
  - AdamW（Loshchilov & Hutter, 2019）：解耦权重衰减，大模型训练的标配
  - 实践建议：扩散去噪器训练首选AdamW，$\beta_1=0.9, \beta_2=0.999, \text{weight\_decay}=0.01$
- **学习率调度**
  - Warmup：训练初期用小学习率逐步增大，避免早期梯度不稳定
    - 典型设置：前5000步线性从0增至峰值lr
  - Cosine Decay：训练后期学习率按余弦曲线衰减
    - $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$
  - 组合策略：Warmup + Cosine Decay是扩散模型训练的标准调度

#### 15.5.2 初始化策略

- **权重初始化的重要性**
  - 初始化决定训练起点：好的初始化→梯度流通畅→训练高效
  - 坏的初始化→梯度消失/爆炸→训练失败
- **经典初始化方法**
  - Xavier初始化（Glorot & Bengio, 2010）：$W \sim \mathcal{N}(0, 2/(d_{\text{in}} + d_{\text{out}}))$
    - 适用于tanh/sigmoid激活
  - He初始化（He et al., 2015）：$W \sim \mathcal{N}(0, 2/d_{\text{in}})$
    - 适用于ReLU激活，扩散模型UNet的默认选择
- **特殊初始化：adaLN-Zero**
  - 回顾15.2.3节：adaLN-Zero的 $\gamma, \beta, \alpha$ 初始化为0
  - 效果：训练初期DiT Block近似为恒等映射，时间步条件从零逐步增强
  - 训练更稳定、收敛更快——这是DiT论文的关键发现之一

#### 15.5.3 早停、验证与超参数调优

- **早停（Early Stopping）**
  - 监控验证集损失，当连续N个epoch不再下降时停止训练
  - 防止过拟合：训练损失继续下降但验证损失上升
  - 扩散模型的特殊性：训练损失（简化VLB）与生成质量（FID）不完全一致——但仍是最实用的监控指标
- **验证集的角色**
  - 不参与训练的独立数据集
  - 用于：(1) 监控过拟合，(2) 调整超参数（学习率、batch size、网络深度/宽度）
  - 实践建议：训练/验证/测试 = 90%/5%/5%
- **关键超参数**
  - 网络深度（层数）：深→表达力强但训练难
  - 隐藏维度（通道数）：宽→参数多但可能过拟合
  - Batch size：大→梯度估计准但内存高；小→正则化效果但噪声大
  - 学习率：与batch size相关，线性缩放规则 $\eta \propto \text{batch\_size}$

#### 15.5.4 过参数化与双重下降

- **经典偏差-方差权衡**
  - 模型容量不足→欠拟合（高偏差）
  - 模型容量过大→过拟合（高方差）
  - 最优点：适度容量
- **双重下降现象**（Belkin et al., 2019; Nakkiran et al., 2021）
  - 当模型容量远超"经典最优点"时，测试误差再次下降
  - 形成双U形曲线：欠拟合 → 过拟合 → 二次下降
  - 现代深度学习的普遍现象：大模型（GPT、DiT）远在二次下降区域
- **对扩散去噪器的意义**
  - 扩散去噪器通常是过参数化的（参数量远大于训练样本量×像素数）
  - 但测试误差并不一定高——这解释了为什么大模型仍然泛化良好
  - 实践启示：不要害怕过参数化，但需要充分训练（足够的epoch和数据）
- **回顾第9章9.5节**
  - 此内容在第9章已简要提及，本节从训练实践角度补充

本章系统梳理了去噪器的架构演进（CNN→UNet→DiT）、关键设计（时间步嵌入）、架构选择（UNet vs DiT）和训练实践。下一章将把这些架构知识应用到具体的逆问题求解中——CT和MRI重建。

**来源**：Kingma & Ba (2015) Adam; Loshchilov & Hutter (2019) AdamW; Glorot & Bengio (2010) Xavier初始化; He et al. (2015) He初始化; Belkin et al. (2019) Double Descent; Nakkiran et al. (2021) Deep Double Descent; Ratti P40-48 (训练最佳实践); Ratti P29-30 (过参数化与双重下降)

---

## 附录

### 附录15A Transformer架构：从注意力到MLP

> 定位：15.3.1节简述了自注意力机制，但Transformer Block的完整结构（LayerNorm、残差连接、MLP、位置编码）对理解DiT至关重要。本附录提供完整的架构细节，供需要深入理解的读者参考。

- 多头注意力的详细计算流程
- 前馈网络（MLP）的结构：两层线性变换 + GELU激活
- 残差连接与LayerNorm：Pre-LN vs Post-LN
- 可学习位置编码 vs 正弦位置编码
- Transformer Block的数据流图

### 附录15B 正弦位置编码的数学性质

> 定位：15.2.2节介绍了正弦位置编码的公式，但其数学性质（相对位置可线性表达、频率递减的多尺度表示）对理解为何选择正弦编码有深层价值。

- 正弦编码的旋转矩阵性质：$\text{PE}(t + \Delta t)$ 可由 $\text{PE}(t)$ 的线性变换表示
- 多尺度频率分析：不同维度对应不同频率，等价于多尺度滤波器组
- 与傅里叶特征的联系：正弦编码是随机傅里叶特征的特例
- 从固定编码到可学习编码的过渡

---

## 素材来源映射

| 节 | 核心素材 | 补充来源 |
|---|---|---|
| 15.0 | 全书叙事 | book_plan.md |
| 15.1.1 | Zhang et al. (2017) DnCNN; Ratti P34-35 | — |
| 15.1.2 | Ronneberger et al. (2015) UNet; Ratti P36-37 | CompImLab25 Part 3 |
| 15.1.3 | Ho et al. (2020) DDPM; Song et al. (2021) Score-SDE | Bologna_UNet_example; 第6章6.6节 |
| 15.2.1 | 第6章6.6节; 第7章; 第11章11.3节 | — |
| 15.2.2 | Vaswani et al. (2017) Transformer | — |
| 15.2.3 | Perez et al. (2018) FiLM; Peebles & Xie (2023) DiT | Ho et al. (2020) DDPM |
| 15.3.1 | Vaswani et al. (2017) Transformer | 附录15A |
| 15.3.2 | Dosovitskiy et al. (2021) ViT; Ratti P38 | Zamir et al. (2022) Restormer |
| 15.3.3 | Peebles & Xie (2023) DiT | Esser et al. (2024) SD3; 第14章14.5节 |
| 15.4.1 | 设计哲学综合 | Dosovitskiy et al. (2021); Peebles & Xie (2023) |
| 15.4.2 | Peebles & Xie (2023) DiT; Esser et al. (2024) SD3 | Dhariwal & Nichol (2021) ADM |
| 15.4.3 | 实践经验综合 | 全书架构使用经验 |
| 15.5.1 | Kingma & Ba (2015) Adam; Ratti P41-43 | Loshchilov & Hutter (2019) AdamW |
| 15.5.2 | Ratti P45; Glorot & Bengio (2010); He et al. (2015) | Peebles & Xie (2023) adaLN-Zero |
| 15.5.3 | Ratti P47-48 | — |
| 15.5.4 | Belkin et al. (2019); Ratti P29-30; 第9章9.5节 | Nakkiran et al. (2021) |

---

## 章节逻辑流

```
15.1 经典架构（CNN/UNet如何去噪？）
      │
      │ "UNet能成为扩散标准架构，不仅因为编码器-解码器设计，
      │   更因为时间步条件注入——网络如何感知噪声？"
      ▼
15.2 时间步嵌入（正弦编码 + 条件注入方式）
      │
      │ "时间步条件解决了感知问题，但CNN的局部感受野限制仍在——
      │   Transformer能否突破？"
      ▼
15.3 ViT→DiT（Transformer替换UNet）
      │
      │ "两种架构各有优势，如何选择？"
      ▼
15.4 UNet vs DiT（设计哲学 + 性能对比 + 选择指南）
      │
      │ "无论选哪种架构，训练都是最后一英里"
      ▼
15.5 训练最佳实践（优化器 + 初始化 + 早停 + 过参数化）
```

---

## 缺失素材清单

| 素材 | 用途 | 紧急程度 | 状态 | 建议来源 |
|---|---|---|---|---|
| UNet架构图（编码器-解码器+跳跃连接） | 15.1.2 架构说明 | ⭐⭐⭐ 高 | ❌ 待补充 | Ronneberger et al. (2015) 原图或重绘 |
| DnCNN架构图 | 15.1.1 架构说明 | ⭐⭐ 中 | ❌ 待补充 | Zhang et al. (2017) 原图 |
| DDPM-UNet架构图（含时间步嵌入） | 15.1.3 架构说明 | ⭐⭐⭐ 高 | ❌ 待补充 | Ho et al. (2020) Figure 2 |
| 时间步嵌入流程图（PE→MLP→注入） | 15.2.2-15.2.3 机制说明 | ⭐⭐⭐ 高 | ❌ 待补充 | 自绘 |
| adaLN-Zero vs FiLM vs 加法注入对比图 | 15.2.3 方式对比 | ⭐⭐ 中 | ❌ 待补充 | 自绘 |
| Transformer Block结构图 | 15.3.1 架构说明 | ⭐⭐⭐ 高 | ❌ 待补充 | Vaswani et al. (2017) 原图 |
| ViT Patchify示意图 | 15.3.2 Patch化说明 | ⭐⭐⭐ 高 | ❌ 待补充 | Dosovitskiy et al. (2021) Figure 1 |
| DiT Block架构图（含adaLN-Zero） | 15.3.3 架构说明 | ⭐⭐⭐ 高 | ❌ 待补充 | Peebles & Xie (2023) Figure 2 |
| DiT缩放规律图（Gflops vs FID） | 15.4.2 定量对比 | ⭐⭐ 中 | ❌ 待补充 | Peebles & Xie (2023) Figure 3 |
| UNet vs DiT感受野对比图 | 15.4.1 设计哲学 | ⭐⭐ 中 | ❌ 待补充 | 自绘 |
| 过参数化双重下降曲线 | 15.5.4 现象说明 | ⭐ 低 | ❌ 待补充 | Nakkiran et al. (2021) |
| Restormer架构图 | 15.3.2 桥梁架构 | ⭐ 低 | ❌ 待补充 | Zamir et al. (2022) |
| 学习率调度曲线（Warmup+Cosine） | 15.5.1 调度说明 | ⭐ 低 | ❌ 待补充 | 自绘 |
