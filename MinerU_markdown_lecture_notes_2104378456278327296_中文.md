## 流匹配与扩散模型导论

Peter Holderrieth 和 Ezra Erives 

网站: https://difusion.csail.mit.edu/ 

1 引言 3
1.1 概述 3
1.2 课程结构 4
1.3 生成建模即采样 4
2 流模型与扩散模型 7
2.1 流模型 7
2.2 扩散模型 10
3 流匹配 14
3.1 条件与边缘概率路径 14
3.2 条件与边缘向量场 16
3.3 学习边缘向量场 19
4 得分函数与得分匹配 25
4.1 条件与边缘得分函数 25
4.2 用 SDE 采样 27
4.3 得分匹配 31
5 引导: 如何基于提示条件生成 34
5.1 朴素引导 34
5.2 无分类器引导 35
6 构建大规模图像或视频生成器 41
6.1 神经网络架构 41
6.2 在潜空间中工作: (变分)自编码器 46
6.3 案例研究: Stable Diffusion 3 与 Meta Movie Gen 52
7 离散扩散模型: 用扩散构建语言模型 54
7.1 连续时间马尔可夫链 (CTMC) 模型 54
7.2 训练 CTMC 模型 59
8 参考文献 66
A 概率论回顾 70
A.1 随机向量 70
A.2 条件密度与条件期望 70 

B Fokker-Planck 方程的证明 72  
C 连续时间马尔可夫链的存在性与唯一性 74  
D 关于 VAE 的补充视角 77  
E 扩散模型文献导读 81 

## 1 引言

从数据制造噪声很容易；从噪声制造数据就是生成建模。 

Song 等人 [43] 

## 1.1 概述

近年来，我们都见证了人工智能（AI）领域一场巨大的革命。像 Nano Banana 或 Stable Difusion 3 这样的图像生成器可以在多种风格下生成照片级真实感和艺术感图像，像 Meta 的 VEO-3 这样的视频模型可以生成极其逼真的电影片段，像 ChatGPT 这样的大语言模型可以对文本提示生成看似人类水平的回复。这场革命的核心是 AI 系统的一种新能力：生成对象的能力。以往的 AI 系统主要用于预测，而新一代 AI 系统具有创造性：它们会“做梦”或基于用户给定的输入创造出新的对象。这样的生成式 AI 系统正是近期这场 AI 革命的核心。 

本课程的目的是教你两种使用最广泛的生成式 AI 算法：去噪扩散模型 [45] 和流匹配 [25, 27, 1, 26]。这些模型是当今最好的图像、音频和视频生成模型（例如 Nano Banana、FLUX 或 VEO-3）的骨干，并且最近成为科学应用（例如蛋白质结构，AlphaFold3 就是一个扩散模型）中最先进的技术。毫无疑问，理解这些模型是一项极其有用的技能。 

所有这些生成式模型都通过反复将噪声转化为数据来生成对象。这种从噪声到数据的演化由常微分方程或随机微分方程（ODE/SDE）的模拟来驱动。流匹配和去噪扩散模型是一族技术，使我们能够利用深度神经网络大规模地构造、训练和模拟这些 ODE/SDE。虽然这些模型实现起来相当简单，但 SDE 的数理性质可能使这些模型难以理解。在本文中，我们提供一份自包含的介绍，讲解微分方程相关的必要数学工具，使你能够系统性地理解这些模型。然后我们逐步讲解当今最先进的图像和视频生成器的完整技术栈。除应用广泛之外，我们认为流模型与扩散模型的内在理论本身就十分优美。因此，最重要的是，我们希望这门课程能给你带来许多乐趣。 

注 1（附加资源） 

虽然这些讲义是自包含的，但我们鼓励你使用以下两种附加资源： 

1. 课堂录像：以讲课形式带你逐节讲解每个部分。 

2. 实验：带你从零实现一个自己的扩散模型。我们强烈建议你“动手实践”并编写代码。 

你可以在课程网站上找到这些资源：https://difusion.csail.mit.edu/。 

## 1.2 课程结构

我们简要概述本文档的内容。 

• 第 1 节，生成建模作为采样：我们形式化“生成”一张图像、视频、蛋白质等的含义。我们会把例如“如何生成一张狗的图像？”这样的问题转化为更精确的从一个概率分布采样的问题。 

• 第 2 节，流与扩散模型：我们讲解生成的机制。正如课程名字所暗示的，这个机制由模拟常微分方程与随机微分方程组成。我们提供微分方程的入门讲解，并说明如何用它们构造生成模型。 

• 第 3 节，流匹配：接下来我们解释并推导流匹配，这是一种简单且可扩展的算法，是所有前述大规模生成模型（如 Stable Difusion、Nano Banana 或 SORA）的核心。 

• 第 4 节，得分匹配：我们研究得分函数以及如何通过得分匹配学习它们。这不仅是扩散模型的训练算法，还解锁了 SDE 采样与引导。 

• 第 5 节，引导：我们学习如何让我们的样本依赖于某个提示（例如“一张猫的图片”），以及如何通过无分类器引导强制样本符合这样的提示。 

• 第 6 节，潜空间与神经网络架构：我们讨论如何构建 Nano Banana 这样的大规模图像和视频生成器。这包括常见的神经网络架构、如何在潜空间中构造模型，以及对各最先进的模型的综述。 

• 第 7 节（可选），离散扩散模型：我们学习如何将扩散模型的原理从欧氏空间推广到语言等离散数据上。这使得我们可以基于扩散模型的原理构建大语言模型。 

所需基础。由于本主题的数理性，我们建议读者具备一定的数学成熟度，尤其对概率论有一些熟悉。因此，我们在附录 A 中加入了一个简短的概率论回顾。如果其中某些概念对你陌生，请不要担心。 

## 1.3 生成建模作为采样

我们首先思考可能遇到的各种数据类型，或称数据模态，并讨论如何对其进行数值表示： 

1. 图像：考虑一张尺寸为 H × W 像素的图像，其中 H 表示高度，W 表示宽度，每个像素有三个颜色通道（RGB）。对每个像素的每个颜色通道，我们给定一个属于 R 的强度值。因此，一张图像可以表示为 ${ \boldsymbol { z } } \in \mathbb { R } ^ { H \times W \times 3 }$ 中的一个元素 

2. 视频：视频只是随时间排列的图像序列。若有 T 个时间点或帧，视频可表示为 $z \in \mathbb { R } ^ { T \times H \times W \times 3 }$ 中的一个元素 

3. 分子结构：一种朴素的方式是用矩阵 $\boldsymbol { z } = ( z ^ { 1 } , \dots , z ^ { N } ) \in \mathbb { R } ^ { 3 \times N }$ 表示一个分子的结构，其中 N 是分子中原子的数目，每个 $z ^ { i } \in \mathbb { R } ^ { 3 }$ 描述一个原子的位置。当然，还有其他更精细的方式来表示分子。 

在上述所有例子中，我们要生成的对象都可以数学上表示为向量（必要时经过展平）。因此，在本文档中，我们约定： 

关键思想 1（对象作为向量） 

我们将被生成的对象视为向量 $z \in \mathbb { R } ^ { d }$ 

上述的一个明显例外是文本数据，它通常被语言模型（如 ChatGPT）建模为离散对象。虽然连续数据 $z \in \mathbb { R } ^ { d }$ 是我们的主要关注对象，我们也在第 7 节研究文本生成。 

生成即采样。让我们定义“生成”的含义。例如，假设我们想生成一张狗的图像。显然，有许多可能的狗的图像都会令我们满意。尤其不存在唯一的“最好”的狗图像。而是存在一系列图像，它们与“像狗”的符合程度各有高低。在机器学习中，人们通常用图像空间上的概率分布来体现这种多样性。我们称这样的分布为数据分布，记作 $p _ { \mathrm { d a t a } }$ 。从数学上可以把 $p _ { \mathrm { d a t a } }$ 看作概率密度，即一个函数 $p _ { \mathrm { d a t a } } : \mathbb { R } ^ { d }  \mathbb { R } _ { \geq 0 }$ ，它给每个可能的对象 $z \in \mathbb { R } ^ { d }$ 指派一个似然度 $p _ { \mathrm { d a t a } } ( z ) ~ \ge ~ 0$ 。在狗图像的例子里，这个分布会给更像狗的图像 z 更高的似然度 $p _ { \mathrm { d a t a } } ( z )$ 。因此，“图像/视频/分子有多符合要求”这种主观说法，被“它在数据分布 $p _ { \mathrm { d a t a } }$ 下有多可能”所替代。有了这个基础，我们可以把生成的任务数学化地表述为从（未知的）分布 $p _ { \mathrm { d a t a } }$ 中采样： 

关键思想 2（生成即采样） 

生成对象 z 建模为从数据分布中采样 $z \sim p _ { \mathrm { d a t a } } .$ 

生成模型是一种让我们能从 $p _ { \mathrm { d a t a } }$ 中生成样本的机器学习模型。在机器学习中，我们训练模型需要数据。在生成建模中，我们通常假设可以访问从 $p _ { \mathrm { d a t a } } .$ 独立采样的有限个样本，它们共同作为真实分布的替代。 

关键思想 3（数据集） 

数据集由有限个样本 $z _ { 1 } , . . . , z _ { N } \sim p _ { \mathrm { d a t a } } .$ 组成 

对于图像，我们可以通过汇编互联网上公开可用的图像来构造数据集。对于视频，我们同样可以利用 YouTube。对于蛋白质结构，像 RCSB 蛋白质数据库（PDB）这样的来源提供了数十万个实验解析的结构。随着数据集规模变得非常大，它对底层分布 $p _ { \mathrm { d a t a } }$ 的表征也变得越来越好。 

引导/条件生成。在许多情况下，我们希望在给定数据 y 的条件下生成对象。例如，我们可能想生成一张以 $y=$ “一条狗跑下覆满积雪的小山，背景是群山” 为条件的图像。我们可以把它重新表述为从条件分布中采样： 

关键思想 4（引导生成） 

引导生成涉及从 $z \sim p _ { \mathrm { d a t a } } ( \cdot | y )$ 采样，其中 y 是条件变量。 

我们把 $p _ { \mathrm { d a t a } } ( \cdot | y )$ 称为引导数据分布。引导式生成任务通常涉及学习对任意的、而非固定的 $y .$ 进行条件化。沿用前面的例子，我们也许想改为条件于另一个文本提示，例如 $y=$ “一张吹生日蜡烛的猫的照片级真实感图像”。因此，我们寻求一个单一模型，可以被条件于这类 $y .$ 中的任意选择。事实证明，无条件生成的技术很容易推广到条件情形。因此，在前 3 节中，我们几乎只关注无条件情形（但要记住，我们的最终目标是条件生成）。 

生成模型。抽象地说，生成模型是一种从 $z \sim p _ { \mathrm { d a t a } }$ 返回样本的算法（或至少近似）。若 $p _ { \mathrm { d a t a } }$ 是狗图像的分布，该算法就返回随机狗图像。在本课程中，我们将聚焦于使用流模型或扩散模型构造生成模型，因为它们是当前的最先进水平。不过要注意，许多其他生成模型已经被开发出来（未来可能还会有更多被发现）。 

## 摘要 2（生成即采样）

我们总结本节的结论： 

1. 在本文中，我们主要考虑生成表示为向量 $z \in \mathbb { R } ^ { d }$ 的对象（如图像、视频和分子结构）的任务。 

2. 生成是这样一个任务：在训练期间访问样本数据集 $z _ { 1 } , \dots , z _ { N } \sim p _ { \mathrm { d a t a } }$ ，从而生成来自概率分布 $p _ { \mathrm { d a t a } }$ 的样本。 

3. 引导生成假设我们把分布条件化于标签 y，希望在训练期间访问成对数据 $( z _ { 1 } , y ) \dots , ( z _ { N } , y )$ ，从而从 $p _ { \mathrm { d a t a } } ( \cdot | y )$ 采样。 

4. 我们的目标是构造一个生成模型，即一个在训练后能从 $p _ { \mathrm { d a t a } }$ 返回样本的模型。 

## 2 流模型与扩散模型

在前一节中，我们把生成建模形式化为从数据分布 $p _ { \mathrm { d a t a } }$ 中采样。我们还形式化了目标：构造一个生成模型，即一个能返回样本 $z \sim p _ { \mathrm { d a t a } }$ 的算法。在本节中，我们描述生成模型如何被构造为对一个适当设计的微分方程的模拟。例如，流匹配和扩散模型分别涉及模拟常微分方程（ODE）和随机微分方程（SDE）。本节的任务是定义并构造这些生成模型，它们将贯穿讲义其余部分。具体而言，我们首先定义 ODE 和 SDE，并讨论它们的模拟。其次，我们描述如何用深度神经网络对 ODE/SDE 进行参数化。由此我们得到流模型与扩散模型的定义以及从这类模型中采样的基本算法。在后续章节中，我们再探索如何训练这些模型。 

## 2.1 流模型

我们先定义常微分方程（ODE）。ODE 的解由一条轨迹定义，即形如 

$$
X: [ 0, 1 ] \to \mathbb {R} ^ {d}, t \mapsto X _ {t},
$$

的函数，它把时间 t 映射到空间 $\mathbb { R } ^ { d }$ 中的某个位置。每个 ODE 由一个向量场 u 定义，即形如 

$$
u: \mathbb {R} ^ {d} \times [ 0, 1 ] \to \mathbb {R} ^ {d}, (x, t) \mapsto u _ {t} (x),
$$

的函数，也就是说，对每个时间 t 和位置 x，我们得到一个向量 $u _ { t } ( x ) \in \mathbb { R } ^ { d }$ ，指定空间中的速度（见图 1）。ODE 对轨迹施加一个条件：我们要一条“沿着”向量场 $u _ { t } .$ 的轨迹，从点 $x _ { 0 }$ 出发。这样的轨迹可以被形式化为方程的解： 

$$
\frac {\mathrm{d}}{\mathrm{d} t} X _ {t} = u _ {t} (X _ {t})
$$

$$
\blacktriangleright \mathrm{ODE}\tag{1a}
$$

$$
X _ {0} = x _ {0}
$$

$$
\blacktriangleright \text{初始条件}\tag{1b}
$$

方程 (1a) 要求 $X _ { t }$ 的导数由 $u _ { t }$ 给出的方向指定。方程 (1b) 要求我们在 $t = 0$ 时从 $x _ { 0 }$ 出发。我们现在可以问：若我们在 $t = 0$ 时从 $X _ { 0 } = x _ { 0 }$ 出发，那么时间 t 时我们在哪里（即 $X _ { t }$ 是什么）？这个问题由一个称为流的函数回答，它是该 ODE 的一个解 

$$
\psi : \mathbb {R} ^ {d} \times [ 0, 1 ] \to \mathbb {R} ^ {d}, (x _ {0}, t) \mapsto \psi_ {t} (x _ {0})\tag{2a}
$$

$$
\frac {\mathrm{d}}{\mathrm{d} t} \psi_ {t} (x _ {0}) = u _ {t} (\psi_ {t} (x _ {0}))
$$

$$
\blacktriangleright \text{流 ODE}\tag{2b}
$$

$$
\psi_ {0} (x _ {0}) = x _ {0}
$$

$$
\blacktriangleright \text{流的初始条件}\tag{2c}
$$

对给定初值 $X _ { 0 } = x _ { 0 }$ ，ODE 的轨迹可通过 $X _ { t } = \psi _ { t } ( X _ { 0 } )$ 得到。因此，向量场、ODE 与流直观上是同一对象的三种描述：向量场定义 ODE，其解就是流。如同每个方程一样，我们应该问自己：ODE 的解存在吗？若存在，唯一吗？数学中的一个基本结果对这两个问题都给出了肯定的回答（“yes!”）——只要我们对 $u_t$ 施加较弱的假设。

> **译注**：流 $\psi_t$ 可以理解为“一整族轨迹的打包”：它把每个可能的出发点 $x_0$ 同时映到时刻 $t$ 的位置，相当于一次解出所有初值问题。也可以把 $\psi_t$ 想象成 $t$ 时刻对整个空间的一次“变形”——图 1 中方格网格被扭曲正是这个含义。当 $u_t$ 足够光滑时，不同出发点不会被挤压到同一点，这正是定理 3 说 $\psi_t$ 是微分同胚的原因。

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/712beec7219106f2b9ffa82528673f384325f28694e0c278763b6e652c943779.jpg)



图 1: 流 $\psi _ { t } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$（红色方格网格）由速度场 $u _ { t } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$（用蓝色箭头可视化）定义，它规定了在所有位置上的瞬时运动（此处 $d = 2 )$ 。我们展示了三个不同时刻 t。可以看到，流是一个微分同胚映射，它“扭曲”空间。图引自 [26]。


## 定理 3（流的存在性与唯一性）

若 $u : \mathbb { R } ^ { d } \times [ 0 , 1 ] \to \mathbb { R } ^ { d }$ 连续可微且导数有界，则 (2) 中的 ODE 有唯一解，由流 $\psi _ { t }$ 给出。此时 $\psi _ { t }$ 对所有 t 都是微分同胚，即 $\psi _ { t }$ 连续可微且其逆 $\psi _ { t } ^ { - 1 }$ 也连续可微 

注意，流存在性与唯一性所要求的假设在机器学习中几乎总是满足的，因为我们用神经网络参数化 $u _ { t } ( x )$ ，而它们的导数总有界。因此，定理 3 对你不必担忧，反而应该视为好消息：在我们要关注的这些情形中，流存在，且是 ODE 的唯一解。证明可参见 [32, 9]。 

## 例 4（线性向量场）

让我们考虑一个简单的向量场 $u _ { t } ( x )$ 例子，它是 $x ,$ 的简单线性函数，即 $\theta > 0$ 时 $u _ { t } ( x ) = - \theta x$ 。则函数 

$$
\psi_ {t} (x _ {0}) = \exp (- \theta t) x _ {0}\tag{3}
$$

定义了流 ψ，它求解方程 (2) 中的 ODE。你可以自行验证：检查 $\psi _ { 0 } ( x _ { 0 } ) = x _ { 0 }$ 并计算 

$$
\frac {\mathrm{d}}{\mathrm{d} t} \psi_ {t} (x _ {0}) \stackrel {(3)} {=} \frac {\mathrm{d}}{\mathrm{d} t} \left(\exp (- \theta t) x _ {0}\right) \stackrel {(i)} {=} - \theta \exp (- \theta t) x _ {0} \stackrel {(3)} {=} - \theta \psi_ {t} (x _ {0}) = u _ {t} (\psi_ {t} (x _ {0})),
$$

其中 (i) 步使用了链式法则。在图 3 中，我们可视化了这种形式随时间指数收敛到 0 的流。 

模拟 ODE。一般而言，若 $u _ { t }$ 不像前面例子中那么简单，则无法显式计算流 $\psi _ { t }$ 。在这种情况下，人们使用数值方法模拟 ODE。幸运的是，这是数值分析中一个经典且研究充分的话题，存在大量强有力方法 [21]。最简单直观的方法是欧拉法。在欧拉法中，我们从 $X _ { 0 } = x _ { 0 }$ 初始化，并按 

$$
X _ {t + h} = X _ {t} + h u _ { t} (X _ {t}) \quad (t = 0, h, 2 h, 3 h, \ldots , 1 - h)\tag{4}
$$

更新，其中 $h = n ^ { - 1 } > 0$ 是步长，$n \in \mathbb { N }$ 是模拟步数。就本课程而言，欧拉法足够用了。为了让你体验一种更复杂的方法，我们考虑按如下更新规则定义的 Heun 法 

$$
\begin{array}{r l} {X _ {t + h} ^ { \prime } = X _ {t} + h u _ {t} (X _ {t})} & {\blacktriangleright \text{新状态的初始猜测（同欧拉步）}} \\ {X _ {t + h} = X _ {t} + \frac {h}{2} (u _ {t} (X _ {t}) + u _ {t + h} (X _ {t + h} ^ { \prime}))} & {\blacktriangleright \text{用 } u \text{ 在当前状态与猜测状态处的平均值更新}} \end{array}
$$

直观地说，Heun 法是这样做的：它先对下一步可能是什么给出一个初步猜测 $X _ { t + h } ^ { \prime }$ ，再通过更新猜测来修正最初的方向。 

流模型。我们现在可以通过让向量场成为神经网络向量场 $u _ { t } ^ { \theta }$ 来用 ODE 构造生成模型。这里我们暂时只是说 $u _ { t } ^ { \theta }$ 是带参数 θ 的参数化函数 $u _ { t } ^ { \theta } : \mathbb { R } ^ { d } \times [ 0 , 1 ]  \mathbb { R } ^ { d }$ 。稍后我们会讨论具体的神经网络架构选择。请记住我们的目标是生成来自分布 $p _ { \mathrm { d a t a } }$ 的样本 $z \sim p _ { \mathrm { d a t a } }$ 。特别地，这些样本必须是随机的。但 ODE 本身并不随机，而是完全确定性的。为了注入一些随机性，我们只需让初值 $X _ { 0 }$ 随机。具体地，我们选择一个初始分布 $p _ { \mathrm { i n i t } }$ 。多数情形下，我们设 $p _ { \mathrm { i n i t } } = \mathcal { N } ( 0 , I _ { d } )$ ，即一个简单的标准高斯分布。最重要的是，无论选什么分布，它都必须是推理时可以轻松采样的。流模型则由如下 ODE 描述 

$$
\begin{array}{c} {X _ {0} \sim p _ {\mathrm{init}}} \\ {\frac {\mathrm{d}}{\mathrm{d} t} X _ {t} = u _ {t} ^ { \theta } (X _ {t})} \end{array}
$$

▶ 随机初始化 

我们的目标是让轨迹的终点 $X _ { 1 }$ 具有分布 $p _ { \mathrm { d a t a } }$ ，即 

$$
X _ {1} \sim p _ {\mathrm{data}} \quad \Leftrightarrow \quad \psi_ {1} ^ {\theta} (X _ {0}) \sim p _ {\mathrm{data}}
$$

其中 $\psi _ { t } ^ { \theta }$ 描述由 $u _ { t } ^ { \theta } .$ 诱导的流。但请注意：虽然称为流模型，神经网络参数化的是向量场，而非流。要计算流，我们需要模拟 ODE。在算法 1 中，我们总结了从流模型中采样的流程。

> **译注**：初学者常困惑：ODE 完全确定，流模型的随机性从何而来？答案是随机性完全来自初始分布 $p_{\mathrm{init}}$——同一个 $X_0$ 永远给出同一条轨迹，抽到不同的 $X_0$ 才得到不同样本。生成模型的本质由此显现：把一个容易采样的简单分布（如标准高斯），通过确定性的可逆变形“搬运”成复杂的数据分布。

算法 1 用欧拉法从流模型中采样
输入: 神经网络向量场 $u_{t}^{\theta}$ ，步数 n
1: 设 t = 0
2: 设步长 $h = \frac{1}{n}$ 3: 抽取样本 $X_{0} \sim p_{init}$ 4: 对 $i = 1, \ldots, n$ 执行
5: $X_{t+h} = X_{t} + hu_{t}^{\theta}(X_{t})$ 6: 更新 $t \leftarrow t + h$ 7: 结束循环
8: 返回 $X_{1}$ 

## 2.2 扩散模型

随机微分方程（SDE）在 ODE 的确定性轨迹之上添加了随机轨迹。随机轨迹通常称为随机过程 $( X _ { t } ) _ { 0 \leq t \leq 1 }$ ，由 

$X _ { t }$ 对每个 $0 \leq t \leq 1$ 都是一个随机变量 

$X : [ 0 , 1 ] \to \mathbb { R } ^ { d } , \quad t \mapsto X _ { t }$ 对 X 的每次抽取都是一条随机轨迹 

给出。特别地，当我们两次模拟同一个随机过程时，由于动力学被设计为随机的，我们可能得到不同结果。 

布朗运动。SDE 是通过布朗运动构造的——布朗运动是源自物理扩散过程研究的一个基本随机过程。你可以把布朗运动看作连续的随机游走。 

让我们定义它：布朗运动 $W = ( W _ { t } ) _ { 0 \leq t \leq 1 }$ 是一个随机 

过程，满足 $W _ { 0 } = 0$ ，轨迹 $t \mapsto W _ { t }$ 连续，且下列两个条件成立： 

1. 正态增量：对所有 $0 \leq s <$ $t ,$ 有 $W _ { t } { - } W _ { s } \sim \mathcal { N } ( 0 , ( t { - } s ) I _ { d } )$ ，即增量服从高斯分布，其方差随时间线性增长（$I _ { d }$ 是单位矩阵）。 

2. 独立增量：对任意 $0 \leq t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { n } =$ 1，增量 $W _ { t _ { 1 } } { - } W _ { t _ { 0 } } , \ldots , W _ { t _ { n } } { - } W _ { t _ { n - 1 } }$ 是独立随机变量。 

布朗运动也被称为维纳过程，这就是我们用 "$"W"$" 标记它的原因。<sup>1</sup> 我们可以用步长 $h > 0$ 通过设 $W _ { 0 } = 0$ 并更新来近似模拟布朗运动 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/82683eb85f30f72eff5a1b1fb65ea07e9803722281a053e861d378569ea8383e.jpg)


$$
W _ {t + h} = W _ {t} + \sqrt {h} \epsilon_ {t}, \quad \epsilon_ {t} \sim \mathcal {N} (0, I _ {d}) \quad (t = 0, h, 2 h, \dots , 1 - h)\tag{5}
$$


图 2: 一维（$d = 1$）布朗运动 $W _ { t }$ 的示例轨迹，用方程 (5) 模拟得到。


在图 2 中，我们绘制了几条布朗运

动的示例轨迹。布朗运动之于随机过程的研究，正如高斯分布之于概率分布研究。从金融到统计物理再到流行病学，布朗运动的研究在机器学习之外有广泛应用。例如，在金融中，布朗运动被用来建模复杂金融产品的价格。即便只作为一个数学构造，布朗运动也引人入胜：例如，布朗运动的路径是连续的（因此你可以一笔不抬地画出来），但它们的长度是无限的（所以你永远也画不完）。

> **译注**：式 (5) 中噪声乘的是 $\sqrt{h}$ 而非 $h$，这是布朗运动最反直觉之处：独立高斯增量相加时方差相加，要使 $t$ 时刻的方差正比于 $t$，每步增量的标准差就必须正比于 $\sqrt{h}$。若误写成 $h\,\epsilon$，各步噪声会更快地相互抵消，模拟出的路径几乎不动。这也解释了式 (9) Euler–Maruyama 方法中同样的 $\sqrt{h}$ 因子。

从 ODE 到 SDE。SDE 的思想是在 ODE 的确定性动力学上加上由布朗运动驱动的随机动力学。由于一切都是随机的，我们不能再像方程 (1a) 那样取导数。因此，我们需要找到一个不使用导数的 ODE 等价形式。为此，我们把 ODE 的轨迹 $( X _ { t } ) _ { 0 \leq t \leq 1 }$ 改写如下： 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/c82b1c93857a6b2e09a5991c06cecde7ac4267b3842831c923c5fd60f82ec652.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/d6f37b1ccc64cb9cfedc18d7e182d43c36b09e94296142f2402a77d6bc2076e9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/eb382a62a8b124818e6cd9c37b761a5520ec670aa28307812a226b56517ddb9b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/dcafa7c2b00b5cd93537e22fcfbac00abd2dd4649e19ce58089326e07e504628.jpg)



图 3: Ornstein-Uhlenbeck 过程（方程 (8)）在 $d = 1$ 维、$\theta = 0 . 2 5$ 且不同 σ 取值（从左到右递增）下的示意。当 $\sigma = 0$ 时，我们得到一个流（平滑、确定性轨迹），并随 $t \to \infty$ 收敛到原点。当 $\sigma > 0$ 时我们得到随机路径，它们随 $t \to \infty$ 收敛到高斯分布 $\textstyle { \mathcal { N } } ( 0 , { \frac { \sigma ^ { 2 } } { 2 \theta } } )$


$$
\begin{array}{r l r l} & {\frac {\mathrm{d}}{\mathrm{d} t} X _ {t} = u _ {t} (X _ {t})} & & {\blacktriangleright \text {用导数表达}} \\ {\stackrel {(i)} {\Leftrightarrow}} & {\frac {1}{h} \left(X _ {t + h} - X _ {t}\right) = u _ {t} (X _ {t}) + R _ {t} (h)} \\ & {\Leftrightarrow \quad X _ {t + h} = X _ {t} + h u _ {t} (X _ {t}) + h R _ {t} (h)} & & {\blacktriangleright \text {用无穷小更新表达}} \end{array}
$$

其中 $R _ { t } ( h )$ 描述当 $h$ 很小时可忽略的函数，即满足 $\lim_{h\to 0} R _ { t } ( h ) = 0$ ，而 (i) 中我们只是使用了导数的定义。上述推导只不过重新陈述了已知的结论：ODE 的轨迹 $( X _ { t } ) _ { 0 \leq t \leq 1 }$ 在每个时间步沿方向 $\boldsymbol { u } _ { t } ( \boldsymbol { X } _ { t } )$ 走一小步。我们现在可以修改最后一个方程使其随机化：SDE 的轨迹 $( X _ { t } ) _ { 0 \leq t \leq 1 }$ 在每个时间步沿方向 $\boldsymbol { u } _ { t } ( \boldsymbol { X } _ { t } )$ 走一小步，外加来自布朗运动的一些贡献：

$$
X _ {t + h} = X _ {t} + \underbrace {h u _ {t} (X _ {t})} _ {\text {确定性项}} + \sigma_ {t} \underbrace {(W _ {t + h} - W _ {t})} _ {\text {随机项}} + \underbrace {h R _ {t} (h)} _ {\text {误差项}}\tag{6}
$$

其中 $\sigma _ { t } \geq 0$ 描述扩散系数，$R _ { t } ( h )$ 描述随机误差项，其标准差 $\mathbb { E } [ \| R _ { t } ( h ) \| ^ { 2 } ] ^ { 1 / 2 } \to 0$ 随 $h  0$ 趋于零。上述描述了一个随机微分方程（SDE）。通常用如下符号记号表示它： 

$$
\mathrm{d} X _ {t} = u _ {t} (X _ {t}) \mathrm{d} t + \sigma_ {t} \mathrm{d} W _ {t}
$$

$$
\blacktriangleright \mathrm{SDE}\tag{7a}
$$

$$
X _ {0} = x _ {0}
$$

$$
\blacktriangleright \text{初始条件}\tag{7b}
$$

但要始终记住，上面 $" \mathrm { d } X _ { t } "$ 记号只是方程 (6) 的一种纯粹非正式记法。不幸的是，SDE 不再有流映射 $\phi _ { t }$ 。这是因为 $X _ { t }$ 的值不再完全由 $X _ { 0 } \sim p _ { \mathrm { i n i t } }$ 决定，因为演化本身是随机的。尽管如此，与 ODE 类似，我们有： 

## 定理 5（SDE 解的存在性与唯一性）

若 $u : \mathbb { R } ^ { d } \times [ 0 , 1 ] \to \mathbb { R } ^ { d }$ 连续可微且导数有界，且 $\sigma _ { t }$ 连续，则 (7) 中的 SDE 有解，由唯一随机过程 $( X _ { t } ) _ { 0 \leq t \leq 1 }$ 给出，满足方程 (6)。 

若这是一门随机微积分课，我们会花好几节课用完整的数学严谨性证明这个定理并构造 SDE，即从基本原理构造布朗运动，并通过随机积分构造过程 $X _ { t }$ 。由于本课程聚焦于机器学习，关于更技术性的处理参见 [29]。最后注意：每个 ODE 也是一个 SDE——只不过其扩散系数恒为零（$\sigma _ { t } = 0$）。因此，在本课程余下的部分中，当我们谈论 SDE 时，把 ODE 视为一个特例。

> **译注**：布朗运动的路径处处连续却处处不可微，因此 “$\mathrm{d}X_t/\mathrm{d}t$” 对 SDE 严格来说并不存在，$\mathrm{d}X_t$ 记号只是式 (6) 离散更新的速记。更重要的观念转变是：SDE 没有“流映射”——相同的初值每次模拟会走出不同的轨迹。于是对 SDE 我们不再追踪单条轨迹，而是关心 $X_t$ 的分布随时间如何演化——这正是后文“概率路径”概念的伏笔。

## 例 6（Ornstein-Uhlenbeck 过程）

让我们考虑常数扩散系数 $\sigma _ { t } = \sigma \geq 0$，以及常数线性漂移 $u _ { t } ( x ) = - \theta x$（其中 $\theta > 0$），得到 SDE 

$$
\mathrm{d} X _ { t } = - \theta X _ { t } \mathrm{d} t + \sigma \mathrm{d} W _ { t }.\tag{8}
$$

上述 SDE 的解 $( X _ { t } ) _ { 0 \leq t \leq 1 }$ 被称为 Ornstein-Uhlenbeck (OU) 过程。我们在图 3 中对其进行了可视化。向量场 $- \theta x$ 把过程推回中心 0（因为漂移总是指向与当前位置相反的方向），而扩散系数 σ 则不断加入噪声。如果我们模拟到 $t \to \infty$ ，这个过程收敛到高斯分布 ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } / ( 2 \theta ) )$ 。注意当 $\sigma = 0$ 时，我们得到的是一个具有线性向量场的流，如方程 (3) 中研究的那样。 

模拟 SDE。到目前为止若你觉得 SDE 的抽象定义难以理解，别担心。对 SDE 更直观的理解方式是回答这个问题：我们可能如何模拟一个 SDE？最简单的方案叫 Euler-Maruyama 法，它之于 SDE 正如欧拉法之于 ODE。用 Euler-Maruyama 法，我们从 $X _ { 0 } = x _ { 0 }$ 初始化，并按 

$$
X _ {t + h} = X _ {t} + h u _ {t} (X _ {t}) + \sqrt {h} \sigma_ {t} \epsilon_ {t}, \qquad \epsilon_ {t} \sim \mathcal {N} (0, I _ {d})\tag{9}
$$

迭代更新，其中 $h = n ^ { - 1 } > 0$ 是步长超参数，$n \in \mathbb { N } .$ 。换句话说，用 Euler-Maruyama 法模拟时，我们沿 $\boldsymbol { u } _ { t } ( \boldsymbol { X } _ { t } )$ 方向走一小步，同时加入一点由 $\sqrt { h } \sigma _ { t }$ 缩放的少量高斯噪声。在本课程（以及配套实验）中模拟 SDE 时，我们通常坚持使用 Euler-Maruyama 法。 

扩散模型。我们现在可以像处理 ODE 那样通过 SDE 构造生成模型。记住我们的目标是把简单分布 $p _ { \mathrm { i n i t } }$ 转化为复杂分布 $p _ { \mathrm { d a t a } }$ 。与 ODE 一样，从 $X _ { 0 } \sim p _ { \mathrm { i n i t } }$ 随机初始化模拟 SDE 是这种转化的自然选择。要参数化这个 SDE，我们只需参数化其核心成分——向量场 $u _ { t }$ ——用神经 

算法 2 从扩散模型中采样（Euler-Maruyama 法）

输入: 神经网络 $u_{t}^{\theta}$ ，步数 n，扩散系数 $\sigma_{t}$ 1: 设 t = 0

2: 设步长 $h = \frac{1}{n}$ 3: 抽取样本 $X_{0} \sim p_{init}$ 4: 对 $i = 1, \ldots, n$ 执行

5: 抽取样本 $\epsilon \sim \mathcal{N}(0, I_{d})$ 6: $X_{t+h} = X_{t} + hu_{t}^{\theta}(X_{t}) + \sigma_{t}\sqrt{h}\epsilon$ 7: 更新 $t \leftarrow t + h$ 8: 结束循环

9: 返回 $X_{1}$ 

网络 $u _ { t } ^ { \theta }$ 。因此，扩散模型由 

$$
\begin{array}{c} {X _ {0} \sim p _ {\mathrm{init}}} \\ {\mathrm{d} X _ {t} = u _ {t} ^ { \theta } (X _ {t}) \mathrm{d} t + \sigma_ {t} \mathrm{d} W _ {t}} \end{array}
$$

给出。在算法 2 中，我们描述了用 Euler-Maruyama 法从扩散模型中采样的流程。我们把本节的结论总结如下。 

## 摘要 7（SDE 生成模型）

在本文档中，扩散模型由一个带参数 θ、参数化向量场的神经网络 $u _ { t } ^ { \theta }$ 和一个固定的扩散系数 $\sigma _ { t }$ 组成： 

神经网络: $u ^ { \theta } : \mathbb { R } ^ { d } \times [ 0 , 1 ] \to \mathbb { R } ^ { d } , ~ ( x , t ) \mapsto u _ { t } ^ { \theta } ( x )$ ，带参数 θ 

固定: $\sigma _ { t } : [ 0 , 1 ] \to [ 0 , \infty ) , \ t \mapsto \sigma _ { t }$ 

要从我们的 SDE 模型（即生成对象）获得样本，流程如下： 

$$
X _ {0} \sim p _ {\mathrm{init}}
$$

▶ 用简单分布初始化，例如高斯 

模拟: $\mathrm { d } X _ { t } = u _ { t } ^ { \theta } ( X _ { t } ) \mathrm { d } t + \sigma _ { t } \mathrm { d } W _ { t }$ 

目标: $X _ { 1 } \sim p _ { \mathrm { d a t a } }$ 

▶ 从 0 到 1 模拟 SDE 

▶ 目标是让 $X _ { 1 }$ 具有分布 $p _ { \mathrm { d a t a } }$ 

$\sigma _ { t } = 0$ 的扩散模型就是一个流模型。 

## 3 流匹配

在上一节中，我们把流模型和扩散模型构造为由神经网络向量场 $u _ { t } ^ { \theta }$ 参数化的生成模型。然而我们还没有讨论如何训练它们，即如何优化参数 θ，使生成模型返回合理的东西（例如好看的图像或令人兴奋的视频）。接下来我们讨论流匹配 [25, 1, 27]，这是一种训练 $u _ { t } ^ { \theta }$ 的简单、可扩展的算法，并代表当前的最先进水平。 

在本节中，我们限定于流模型，即我们有一个神经网络 $u _ { t } ^ { \theta }$ ，并通过模拟 ODE 从生成模型中获得样本 

$$
X _ {0} \sim p _ {\mathrm{init}}, \quad \mathrm{d} X _ {t} = u _ {t} ^ { \theta } (X _ {t}) \mathrm{d} t
$$

(流模型) 

(10) 

并以 $t = 1$ 时的终点 $X _ { 1 }$ 作为样本。如前所述，我们的目标是 $X _ { 1 }$ 服从数据分布 $p _ { \mathrm { d a t a } } ,$ 即 $X _ { 1 } \sim p _ { \mathrm { d a t a } }$ 。因此，"如何训练" 神经网络的问题实际上是下面这个问题：我们如何优化 θ，使得模拟方程 (10) 中的流模型得到来自数据分布的样本 $X _ { 1 } \sim p _ { \mathrm { d a t a } } ?$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/7a85c7b372fe3b6b668b43014a43536393fb43af193ce24c6b5a66d7a6ac81ad.jpg)



图 4: 通过高斯条件概率路径，对一组图像进行从噪声到数据的逐渐插值。注意每张图像是维数 $d = 3 2 \times 3 2$ 的数据点，因此我们绘制的是概率路径上的单个样本；而图 5 中我们把分布画成二维直方图。


## 3.1 条件与边缘概率路径

流匹配的第一步是指定一条概率路径。直观地，概率路径指定了噪声 $p _ { \mathrm { i n i t } }$ 与数据 $p _ { \mathrm { d a t a } }$ 之间的逐渐插值（见图 4）。但我们为什么需要它呢？请记住我们期望的 ODE 轨迹在 $t = 0$ 时满足 $X _ { 0 } \sim p _ { \mathrm { i n i t } }$ ，在 $t = 1$ 时满足 $X _ { 1 } \sim p _ { \mathrm { d a t a } }$ 。但起点与终点之间 $0 < t < 1$ 的时间呢？事实上，对于中间时刻发生什么，我们有一定的选择自由度，而这正是概率路径在数学上所形式化的内容。 

在下文中，对数据点 $z \in \mathbb { R } ^ { d }$ ，我们用 $\delta _ { z }$ 表示狄拉克 delta "分布"。这是你能想象的最简单的分布：从 $\delta _ { z }$ 采样总是返回 z（即它是确定性的）。条件（插值）概率路径是 $\mathbb { R } ^ { d }$ 上的一组分布 $p _ { t } ( x | z )$ ，满足： 

$$
p _ {0} (\cdot | z) = p _ {\mathrm{init}}, \quad p _ {1} (\cdot | z) = \delta_ {z} \quad \mathrm{forall} z \in \mathbb {R} ^ {d}.\tag{11}
$$

换言之，条件概率路径把初始分布 $p _ { \mathrm { i n i t } }$ 逐渐转化为单个数据点（例如见图 4）。你可以把概率路径看作分布空间中的一条轨迹。 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/3404dd4cd447cb9a5d80421fcfb0d2f7b1ff6841ae3b8db7f48d0ac6031f3c98.jpg)



图 5: 条件（上）与边缘（下）概率路径的示意。这里我们绘制的是高斯概率路径，取 $\alpha _ { t } = t , \beta _ { t } = 1 - t .$ 。条件概率路径插值高斯分布 $p _ { \mathrm { i n i t } } = \mathcal { N } ( 0 , I _ { d } )$ 与单数据点 z 的 $\delta _ { z }$ 。边缘概率路径插值一个高斯分布和数据分布 $p _ { \mathrm { d a t a } }$（此处 $p _ { \mathrm { d a t a } }$ 是维度 $d = 2$ 的一个玩具分布，用棋盘图案表示。）


每个条件概率路径 $p _ { t } ( x | z )$ 都诱导出一条边缘概率路径 $p _ { t } ( x )$ ，定义为先从数据分布中采样一个数据点 $z \sim p _ { \mathrm { d a t a } }$ ，再从 $p _ { t } ( \cdot | z )$ 中采样得到的分布 

$$
z \sim p _ {\mathrm{data}}, \quad x \sim p _ {t} (\cdot | z) \quad \Rightarrow x \sim p _ {t}
$$

▶ 从边缘路径中采样 

$$
p _ {t} (x) = \int p _ {t} (x | z) p _ {\mathrm{data}} (z) \mathrm{d} z\tag{12}
$$

$$
\blacktriangleright \text{边缘路径的密度}\tag{13}
$$

注意我们知道如何从 $p _ { t }$ 采样，但无法计算密度值 $p _ { t } ( x )$，因为该积分难解（intractable）——也就是说，我们可以按方程 (12) 采样，却算不出方程 (13) 中的密度。请自行验证：由于方程 (11) 中 $p _ { t } ( \cdot | z )$ 的条件，边缘概率路径 $p _ { t }$ 在 $p _ { \mathrm { i n i t } }$ 和 $p _ { \mathrm { d a t a } } .$ 之间插值。 

$$
p _ {0} = p _ {\mathrm{init}} \quad \mathrm{and} \quad p _ {1} = p _ {\mathrm{data}}.
$$

$$
\blacktriangleright \text{噪声-数据插值}\tag{14}
$$

概率路径中最重要（远超其他）的例子是高斯概率路径——因此我们强烈建议认真阅读下一个例子。 

## 例 8（高斯条件概率路径）

一种特别流行的概率路径是高斯概率路径。这是大多数最先进模型使用的概率路径。设 $\alpha _ { t } , \beta _ { t }$ 为噪声调度器：两个连续可微、单调函数，满足 $\alpha _ { 0 } = \beta _ { 1 } = 0$ 且 $\alpha _ { 1 } = \beta _ { 0 } = 1$ 。我们这样定义条件概率路径 

$$
p _ {t} (\cdot | z) = \mathcal {N} (\alpha_ {t} z, \beta_ {t} ^ {2} I _ {d})
$$

$$
\blacktriangleright \text{高斯条件路径}\tag{15}
$$

由我们对 $\alpha _ { t }$ 和 $\beta _ { t }$ 施加的条件，它满足 

$$
p _ {0} (\cdot | z) = \mathcal {N} (\alpha_ {0} z, \beta_ {0} ^ {2} I _ {d}) = \mathcal {N} (0, I _ {d}), \quad \mathrm{and} \quad p _ {1} (\cdot | z) = \mathcal {N} (\alpha_ {1} z, \beta_ {1} ^ {2} I _ {d}) = \delta_ {z},
$$

其中我们用到方差为零、均值 z 的正态分布就是 $\delta _ { z }$ 。因此，这种 $p _ { t } ( x | z )$ 的选择满足 $p _ { \mathrm { i n i t } } = \mathcal { N } ( 0 , I _ { d } )$ 下的方程 (11)，因此是一个有效的条件插值路径。在图 4 中，我们展示了它应用于图像的情形。我们可以把从边缘路径 $p _ { t }$ 采样表述为： 

$$
z \sim p _ {\mathrm{data}}, \epsilon \sim p _ {\mathrm{init}} = \mathcal {N} (0, I _{d }) \Rightarrow x = \alpha_ {t} z + \beta_ {t} \epsilon \sim p _ {t} \quad \blacktriangleright \text {从边缘高斯路径中采样} \tag {16}
$$

直观地说，上述过程在较小的 t 时加入更多噪声，直到时间 $t = 0$ ，那时只剩噪声。在图 5 中，我们绘制了这种插值路径的一个例子。 

## 3.2 条件与边缘向量场

概率路径 $( p _ { t } ) _ { 0 \leq t \leq 1 }$ 规定了轨迹上的点 $X _ { t }$ 应该具有什么分布 $X _ { t } \sim p _ { t }$ 。在这里，这只是我们"希望"成立的情形。但我们怎样才能找到一个向量场，使得轨迹 $X _ { t }$ 跟随概率路径？流匹配显式地构造了这样一个向量场——"边缘向量场"——我们在本节中讲解它。 

对每个数据点 $z \in \mathbb { R } ^ { d }$ ，我们用 $u _ { t } ^ { \mathrm { t a r g e t } } ( \cdot | z )$ 表示一个条件向量场。它可以是任何使得对应 ODE 给出条件概率路径 $p _ { t } ( \cdot | z )$ 的向量场，即满足 

$$
X _ {0} \sim p _ {\mathrm{init}}, \quad \frac {\mathrm{d}}{\mathrm{d} t} X _ {t} = u _ {t} ^ {\mathrm{target}} (X _ {t} | z) \Rightarrow X _ {t} \sim p _ {t} (\cdot | z) (0 \leq t \leq 1).\tag{17}
$$

我们常常可以靠手算（即自己做点代数）解析地找到条件向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( \cdot | z )$ 。我们将以贯穿全文的高斯概率路径为例，在例 10 中推导其条件向量场 $u _ { t } ( x | z )$，以此说明。 

乍一看，条件向量场似乎无用，因为 ODE 的所有终点 $X _ { 1 }$ 都会坍缩为 $X _ { 1 } \ = \ z _ { \cdot }$ ，即我们只是在重新生成已知的数据点 z。然而，条件向量场是构造向量场的基本构件，该向量场能生成真正来自 $p _ { \mathrm { d a t a } }$ 的样本： 

## 定理 9（边缘化技巧）

设 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ 是条件向量场（方程 (17)）。那么按如下定义的边缘向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x )$ 

$$
u _ {t} ^ {\mathrm{target}} (x) = \int u _ {t} ^ {\mathrm{target}} (x | z) \frac {p _ {t} (x | z) p _ {\mathrm{data}} (z)}{p _ {t} (x)} \mathrm{d} z,\tag{18}
$$

跟随边缘概率路径，即 

$$
X _ {0} \sim p _ {\mathrm{init}}, \quad \frac {\mathrm{d}}{\mathrm{d} t} X _ {t} = u _ {t} ^ {\mathrm{target}} (X _ {t}) \quad \Rightarrow \quad X _ {t} \sim p _ {t} \quad (0 \leq t \leq 1).\tag{19}
$$

特别地，对该 ODE 有 $X _ { 1 } \sim p _ { \mathrm { d a t a } }$ ，因此我们可以说，$u _ { t } ^ { \mathrm { t a r g e t } }$ “把噪声 $p _ { \mathrm { i n i t } }$ 转化为数据 $p _ { \mathrm { d a t a } }$”

> **译注**（边缘化技巧为何成立）：单个条件向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ 只会把所有初始噪声点都送往同一个数据点 $z$，直接使用它只能“复读”训练数据。边缘化技巧的思路是：在每个位置 $x$ 处，先按后验 $p _ { t } ( x | z ) \, p _ { \mathrm { d a t a } } ( z) / p _ { t } ( x)$ 衡量“这个含噪样本 $x$ 有多大可能来自各数据点 $z$”，再把对应的条件速度加权平均。可以证明，这个平均速度恰好是让边缘分布 $p _ { t }$ 按连续性方程演化所需的速度，因此其 ODE 能把 $p _ { \mathrm { i n i t } }$ 变换为 $p _ { \mathrm { d a t a } }$。 

 
学到的边缘 ODE 的轨迹


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/a62477ba0dc2ffecdf7ab2cf2bbeee893720385380918a580db38a934866c591.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/406038fffa6a90b9dce1abf6d7620b217e674e35ba883f7a62351df73b2a0760.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/76ef6645860f2d4cd1ad2b8352e260e4584dbe3bc720af4e8a9434facdc4b63f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/eeb7676aad2de64a78f6dfc98388ebc5d698c024a1eea0d5869588d82e44885f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/71a2b0b9dad56935cf1d0c03113f54d6faef993bc95d1ae757c6a0b31389362a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/77bcd2b19bc979d5ea2ed885fe51b2afdb6014b77203a867a6a524cb7401a6f6.jpg)



图 6: 定理 9 的示意。用 ODE 模拟概率路径。数据分布 $p _ { \mathrm { d a t a } }$ 用蓝色背景。高斯 $p _ { \mathrm { i n i t } }$ 用红色背景。上排：条件概率路径。左：来自条件路径 $p _ { t } ( \cdot | z )$ 的 ground truth 样本。中：随时间变化的 ODE 样本。右：用方程 (20) 中 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ 模拟 ODE 得到的轨迹。下排：模拟边缘概率路径。左：来自 $p _ { t }$ 的 ground truth 样本。中：随时间变化的 ODE 样本。右：用边缘向量场 $u _ { t } ^ { \operatorname { f l o w } } ( x )$ 模拟 ODE 得到的轨迹。可以看到，条件向量场跟随条件概率路径，边缘向量场跟随边缘概率路径。


## 例 10（高斯概率路径的目标 ODE）

如前，设 $p _ { t } ( \cdot | z ) = \mathcal N ( \alpha _ { t } z , \beta _ { t } ^ { 2 } I _ { d } )$ 为噪声调度器 $\alpha _ { t } , \beta _ { t }$（见方程 (15)）。令 $\dot { \alpha } _ { t } = \partial _ { t } \alpha _ { t }$ 且 $\dot { \boldsymbol { \beta } } _ { t } = \partial _ { t } \boldsymbol { \beta } _ { t }$ 分别表示 $\alpha _ { t }$ 与 $\beta _ { t }$ 的时间导数。这里我们要证明由 

$$
u _ {t} ^ {\mathrm{target}} (x | z) = \left(\dot {\alpha} _ {t} - \frac {\dot {\beta} _ {t}}{\beta_ {t}} \alpha_ {t}\right) z + \frac {\dot {\beta} _ {t}}{\beta_ {t}} x\tag{20}
$$

给出的条件高斯向量场在定理 9 的意义下是有效的条件向量场模型：若 $X _ { 0 } \sim { \mathcal { N } } ( 0 , I _ { d } )$ ，其 ODE 轨迹 $X _ { t }$ 满足 $X _ { t } \sim p _ { t } ( \cdot | z ) = \mathcal N ( \alpha _ { t } z , \beta _ { t } ^ { 2 } I _ { d } )$ 。在图 6 中，我们将来自条件概率路径（ground truth）的样本与此流模拟 ODE 轨迹的样本进行视觉比较来验证这一点。可以看到两者分布一致。我们现在证明这一点。 

证明。我们首先通过定义构造条件流模型 $\psi _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ 

$$
\psi_ {t} ^ {\mathrm{target}} (x | z) = \alpha_ {t} z + \beta_ {t} x.\tag{21}
$$

若 $X _ { t }$ 是 $\psi _ { t } ^ { \mathrm { t a r g e t } } ( \cdot | z )$ 的 ODE 轨迹且 $X _ { 0 } \sim p _ { \mathrm { i n i t } } = \mathcal { N } ( 0 , I _ { d } )$ ，则由定义 

$$
X _ {t} = \psi_ {t} ^ {\mathrm{target}} (X _ {0} | z) = \alpha_ {t} z + \beta_ {t} X _ {0} \sim \mathcal {N} (\alpha_ {t} z, \beta^ {2} I _ {d}) = p _ {t} (\cdot | z).
$$

我们得出轨迹的分布就像条件概率路径（即方程 (17) 成立）。剩下的是从 $\psi _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ 中提取向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ 。按流的定义（方程 (2b)），有 

$$
\frac {\mathrm{d}}{\mathrm{d} t} \psi_ {t} ^ {\mathrm{target}} (x | z) = u _ {t} ^ {\mathrm{target}} (\psi_ {t} ^ {\mathrm{target}} (x | z) | z) \quad \text { for   all } x, z \in \mathbb {R} ^ {d}
$$

$$
\stackrel {(i)} {\Leftrightarrow} \dot {\alpha} _ {t} z + \dot {\beta} _ {t} x = u _ {t} ^ {\mathrm{target}} (\alpha_ {t} z + \beta_ {t} x | z) \quad \text {for all} x, z \in \mathbb {R} ^ {d}\tag{ii}
$$

$$
\dot {\alpha} _ {t} z + \dot {\beta} _ {t} \left(\frac {x - \alpha_ {t} z}{\beta_ {t}}\right) = u _ {t} ^ {\mathrm{target}} (x | z) \quad \mathrm{forall} x, z \in \mathbb {R} ^ {d}\tag{iii}
$$

$$
\left(\dot {\alpha} _ {t} - \frac {\dot {\beta} _ {t}}{\beta_ {t}} \alpha_ {t}\right) z + \frac {\dot {\beta} _ {t}}{\beta_ {t}} x = u _ {t} ^ {\mathrm{target}} (x | z) \quad \text { for   all } x, z \in \mathbb {R} ^ {d}
$$

其中 (i) 使用了 $\psi _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ 的定义（方程 (21)），(ii) 我们做重参数化 $x  ( x - \alpha _ { t } z ) / \beta _ { t }$ ，(iii) 我们只做了一些代数。注意最后这个方程就是我们方程 (20) 中定义的条件高斯向量场。这就证明了该论断。<sup>a</sup> □ 

<sup>a</sup>也可以通过把它代入本节稍后介绍的连续性方程来验证。

> **译注**（如何手推条件向量场）：例 10 的推导思路是“从答案出发”——先构造显然满足要求的条件流 $\psi _ { t } ^ { \mathrm { t a r g e t } } ( x | z ) = \alpha _ { t } z + \beta _ { t } x$（线性变换把 $\mathcal { N } ( 0 , I _ { d } )$ 精确地变为 $\mathcal { N } ( \alpha _ { t } z , \beta _ { t } ^ { 2 } I _ { d } )$），再对 $t$ 求导，反解出向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$。这是手推条件向量场的标准套路：先猜一个能把初始分布精确“搬运”到目标分布的显式变换，再对时间求导提取速度场。 

见定理 9 的图 6 示意。让我们对边缘向量场获得一些直觉。统计学中的贝叶斯法则说，下面这个项描述一个后验分布 

$$
\frac {p _ {t} (x | z) p _ {\mathrm{data}} (z)}{p _ {t} (x)} = \text {"给定含噪数据 x 时数据点 z 的后验"}
$$

其中 $p _ { \mathrm { d a t a } } ( z )$ 是先验分布。边缘向量场则只是一个平均：对每个可能的数据点 z，取速度 $u _ { t } ( x | z ) - \mathrm { i . e }$ ，即把我们带到 z 的方向——然后按我们认为 x 来自 z 的置信度加权这个速度。对所有数据点取平均，就得到边缘向量场。 

本节余下部分将使这个直觉严格化并证明定理 9。作为主要数学工具，我们将使用连续性方程——一个在数学与物理学中的基本方程。定义散度算子 div 为 

$$
\operatorname{div} (v _ {t}) (x) = \sum_ {i = 1} ^ {d} \frac {\partial}{\partial x _ {i}} v _ {t} ^ { i } (x)\tag{22}
$$

其中 $v _ { t } ^ { i }$ 是 $v _ { t }$ 的第 i 个坐标。 

## 定理 11（连续性方程）

考虑一个向量场为 $u _ { t } ^ { \mathrm { t a r g e t } }$ 且 $X _ { 0 } \sim p _ { \mathrm { i n i t } } = p _ { 0 }$ 的流模型。则对所有 $0 \leq t \leq 1$ 有 $X _ { t } \sim p _ { t }$ 当且仅当 

$$
\partial_ {t} p _ {t} (x) = - \mathrm{div} (p _ {t} u _ {t} ^ {\mathrm{target}}) (x) \quad \text { for   all } x \in \mathbb {R} ^ {d}, 0 \leq t \leq 1,\tag{23}
$$

其中 $\begin{array} { r } { \partial _ { t } p _ { t } ( x ) = \frac { \mathrm { d } } { \mathrm { d } t } p _ { t } ( x ) } \end{array}$ 表示 $p _ { t } ( x )$ 的时间导数。方程 23 称为连续性方程。 

对于偏好数学的读者，我们在附录 B 给出连续性方程的一个自包含证明。在进入正题之前，让我们先试着直观理解连续性方程。左端 $\partial _ { t } p _ { t } ( x )$ 描述 $x$ 处的概率 $p _ { t } ( x )$ 随时间变化多少。直观地，这个变化应该对应概率质量的净流入。对流模型而言，粒子 $X _ { t }$ 沿向量场 $u _ { t } ^ { \mathrm { t a r g e t } }$ 前进。正如你可能从物理中回忆的，散度度量向量场从某点的净流出量，因此负散度度量净流入量。再乘以当前停留在 $x$ 处的总概率质量，即知 $- \mathrm { d i v } ( p _ { t } u _ { t } )$ 度量概率质量的总流入。由于概率质量守恒（总是积分为 1），方程的左右两边应该相等！现在我们开始证明定理 9 的边缘化技巧。

> **译注**（连续性方程的物理直觉）：连续性方程是概率质量的“守恒定律”，与流体力学中的质量守恒方程同构：散度为正表示该处概率质量净流出，为负表示净流入；总概率恒等于 1，概率质量不生不灭、只被向量场搬运，因此方程两端必然相等。 

定理 9 的证明。由定理 11，我们要证明按方程 (18) 定义的边缘向量场 $u _ { t } ^ { \mathrm { t a r g e t } }$ 满足连续性方程。我们可以直接计算： 

$$
\begin{array}{r l} \partial_ {t} p _ {t} (x) & \stackrel {(i)} {=} \partial_ {t} \int p _ {t} (x | z) p _ {\mathrm{data}} (z) \mathrm{d} z = \int \partial_ {t} p _ {t} (x | z) p _ {\mathrm{data}} (z) \mathrm{d} z \\ & \stackrel {(i i)} {=} \int - \mathrm{div} (p _ {t} (\cdot | z) u _ {t} ^ {\mathrm{target}} (\cdot | z)) (x) p _ {\mathrm{data}} (z) \mathrm{d} z \\ & \stackrel {(i i i)} {=} - \mathrm{div} \left(\int p _ {t} (x | z) u _ {t} ^ {\mathrm{target}} (x | z) p _ {\mathrm{data}} (z) \mathrm{d} z\right) \\ & \stackrel {(i v)} {=} - \mathrm{div} \left(p _ {t} (x) \int u _ {t} ^ {\mathrm{target}} (x | z) \frac {p _ {t} (x | z) p _ {\mathrm{data}} (z)}{p _ {t} (x)} \mathrm{d} z\right) (x) \\ & \stackrel {(v)} {=} - \mathrm{div} \left(p _ {t} u _ {t} ^ {\mathrm{target}}\right) (x), \end{array}
$$

其中 (i) 使用方程 (12) 中 $p _ { t } ( x )$ 的定义，(ii) 使用条件概率路径 $p _ { t } ( \cdot | z )$ 的连续性方程，(iii) 用方程 (22) 交换积分与散度算子，(iv) 乘以并除以 $p _ { t } ( x )$ ，(v) 使用方程 (18)。上述等式链的开头与结尾表明 $u _ { t } ^ { \mathrm { t a r g e t } }$ 满足连续性方程。由定理 11，这足以推出方程 (19)，证明完毕。 □ 

## 3.3 学习边缘向量场

现在，我们准备好描述训练算法了。流匹配的目标是训练神经网络 $u _ { t } ^ { \theta }$ ，使其等于边缘向量场 $u _ { t } ^ { \mathrm { t a r g e t } }$ 。若此成立，由定理 9 我们知道终点 $X _ { 1 } \sim p _ { \mathrm { d a t a } }$ 具有我们所期望的分布。在下文中，我们用 $\mathrm { U n i f } = \mathrm { U n i f } _ { [ 0 , 1 ] }$ 表示区间 [0, 1] 上的均匀分布，用 E 表示随机变量的期望。得到 $u _ { t } ^ { \theta } \approx u _ { t } ^ { \mathrm { t a r g e t } }$ 的直观方式是使用均方误差，即使用如下定义的流匹配损失 

$$
\mathcal {L} _ {\mathrm{FM}} (\theta) = \mathbb {E} _ {t \sim \mathrm{Unif}, x \sim p _ {t}} [ \| u _ {t} ^ { \theta } (x) - u _ {t} ^ {\mathrm{target}} (x) \| ^ {2} ]\tag{24}
$$

$$
\stackrel {(i)} {=} \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, x \sim p _ {t} (\cdot | z)} [ \| u _ {t} ^ { \theta } (x) - u _ {t} ^ {\mathrm{target}} (x) \| ^ {2} ],\tag{25}
$$

其中 $\begin{array} { r } { p _ { t } ( x ) = \int p _ { t } ( x | z ) p _ { \mathrm { d a t a } } ( z ) \mathrm { d } z } \end{array}$ 是边缘概率路径，(i) 中我们使用方程 (12) 给出的采样过程。直观地，这个损失在说：第一步，随机抽取一个时间 $t \in [ 0 , 1 ]$ 。第二步，从数据集中随机抽取一个点 z，从 $p _ { t } ( \cdot | z )$（例如加些噪声）采样，并计算 $u _ { t } ^ { \theta } ( x )$ 。最后，计算神经网络输出与边缘向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x )$ 之间的均方误差。可惜事情到此还没完。虽然由定理 9 我们知道 $u _ { t } ^ { \mathrm { t a r g e t } }$ 的公式，但由于该积分不可解析求解，我们无法高效计算它。相反，我们要利用条件向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ 可以解析计算这一事实。为此，我们定义条件流匹配损失 

$$
\mathcal {L} _ {\mathrm{CFM}} (\theta) = \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, x \sim p _ {t} (\cdot | z)} [ \| u _ {t} ^ { \theta } (x) - u _ {t} ^ {\mathrm{target}} (x | z) \| ^ {2} ].\tag{26}
$$

注意与方程 (24) 的区别：我们使用条件向量场 $\boldsymbol u _ { t } ^ { \mathrm { t a r g e t } } ( \boldsymbol x | \boldsymbol z )$ 替代边缘向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x )$ 。既然 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ 有解析公式，我们就可以轻松最小化上述损失。但等等，若我们关心的是边缘向量场，对条件向量场做回归又有什么意义？事实证明，通过显式地对可解析计算的条件向量场做回归，我们隐式地对不可解析求解的边缘向量场做回归。下面这个结果把这个直觉精确化。 

## 定理 12

边缘流匹配损失与条件流匹配损失只差一个常数。即 

$$
\mathcal {L} _ {\mathrm{FM}} (\theta) = \mathcal {L} _ {\mathrm{CFM}} (\theta) + C,
$$

其中 C 与 θ 无关。因此它们的梯度相同： 

$$
\nabla_ {\theta} \mathcal {L} _ {\mathrm{FM}} (\theta) = \nabla_ {\theta} \mathcal {L} _ {\mathrm{CFM}} (\theta).
$$

于是，例如用随机梯度下降（SGD）最小化 ${ \mathcal { L } } _ { \mathrm { C F M } } ( \theta )$ 等价于用同样方式最小化 ${ \mathcal { L } } _ { \mathrm { F M } } ( \theta )$ 。特别地，对 ${ \mathcal { L } } _ { \mathsf { C F M } } ( \theta )$ 的极小化点 $\theta ^ { * }$ ，有 $u _ { t } ^ { \theta ^ { * } } = u _ { t } ^ { \tt t a r g e t }$ 成立，即神经网络将等于边缘向量场（假设参数化具有无穷表达能力）。 

直接证明。证明通过把均方误差展开成三部分并消去常数： 

$$
\begin{array}{r l} & {\mathcal {L} _ {\mathrm{FM}} (\theta) \overset {(i)} {=} \mathbb {E} _ {t \sim \mathrm{Unif}, x \sim p _ {t}} [ \| u _ {t} ^ { \theta } (x) - u _ {t} ^ {\mathrm{target}} (x) \| ^ {2} ]} \\ & {\overset {(i i)} {=} \mathbb {E} _ {t \sim \mathrm{Unif}, x \sim p _ {t}} [ \| u _ {t} ^ { \theta } (x) \| ^ {2} - 2 u _ {t} ^ { \theta } (x) ^ {T} u _ {t} ^ {\mathrm{target}} (x) + \| u _ {t} ^ {\mathrm{target}} (x) \| ^ {2} ]} \\ & {\overset {(i i i)} {=} \mathbb {E} _ {t \sim \mathrm{Unif}, x \sim p _ {t}} \left[ \| u _ {t} ^ { \theta } (x) \| ^ {2} \right] - 2 \mathbb {E} _ {t \sim \mathrm{Unif}, x \sim p _ {t}} [ u _ {t} ^ { \theta } (x) ^ {T} u _ {t} ^ {\mathrm{target}} (x) ] + \underbrace {\mathbb {E} _ {t \sim \mathrm{Unif} _ {[ 0 , 1 ] , x \sim p _ {t}}} [ \| u _ {t} ^ {\mathrm{target}} (x) \| ^ {2} ]} _ {=: C _ {1}}} \\ & {\overset {(i v)} {=} \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, x \sim p _ {t} (\cdot | z)} [ \| u _ {t} ^ { \theta } (x) \| ^ {2} ] - 2 \mathbb {E} _ {t \sim \mathrm{Unif}, x \sim p _ {t}} [ u _ {t} ^ { \theta } (x) ^ {T} u _ {t} ^ {\mathrm{target}} (x) ] + C _ {1}} \end{array}
$$

其中 (i) 由定义成立，(ii) 使用公式 $\| a - b \| ^ { 2 } = \| a \| ^ { 2 } - 2 a ^ { T } b + \| b \| ^ { 2 }$ ，(iii) 定义常数 $C _ { 1 }$ ，(iv) 使用方程 (12) 给出的 $p _ { t }$ 的采样过程。让我们重新表达第二项： 

$$
\begin{array}{r l} & {\mathbb {E} _ {t \sim \mathrm{Unif}, x \sim p _ {t}} [ u _ {t} ^ { \theta } (x) ^ {T} u _ {t} ^ {\mathrm{target}} (x) ] \overset {(i)} {=} \int_ {0} ^ {1} \int p _ {t} (x) u _ {t} ^ { \theta } (x) ^ {T} u _ {t} ^ {\mathrm{target}} (x) \mathrm{d} x \mathrm{d} t} \\ & {\overset {(i i)} {=} \int_ {0} ^ {1} \int p _ {t} (x) u _ {t} ^ { \theta } (x) ^ {T} \left[ \int u _ {t} ^ {\mathrm{target}} (x | z) \frac {p _ {t} (x | z) p _ {\mathrm{data}} (z)}{p _ {t} (x)} \mathrm{d} z \right] \mathrm{d} x \mathrm{d} t} \\ & {\overset {(i i i)} {=} \int_ {0} ^ {1} \int \int u _ {t} ^ { \theta } (x) ^ {T} u _ {t} ^ {\mathrm{target}} (x | z) p _ {t} (x | z) p _ {\mathrm{data}} (z) \mathrm{d} z \mathrm{d} x \mathrm{d} t} \\ & {\overset {(i v)} {=} \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, x \sim p _ {t} (\cdot | z)} [ u _ {t} ^ { \theta } (x) ^ {T} u _ {t} ^ {\mathrm{target}} (x | z) ]} \end{array}
$$

其中 (i) 把期望写成积分，(ii) 使用方程 (18)，(iii) 使用积分是线性这一事实，(iv) 把积分写回期望。注意这正是证明的关键一步：等式开头使用的是边缘向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x )$ ，而结尾使用的是条件向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ 。把它代回 $\mathcal { L } _ { \mathrm { F M } }$ 的方程，得到： 

$$
\begin{array}{r l} & {\mathcal {L} _ {\mathrm{FM}} (\theta) \overset {(i)} {=} \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, x \sim p _ {t} (\cdot | z)} [ \| u _ {t} ^ { \theta } (x) \| ^ {2} ] - 2 \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, x \sim p _ {t} (\cdot | z)} [ u _ {t} ^ { \theta } (x) ^ {T} u _ {t} ^ {\mathrm{target}} (x | z) ] + C _ {1}} \\ & {\overset {(i i)} {=} \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, x \sim p _ {t} (\cdot | z)} [ \| u _ {t} ^ { \theta } (x) \| ^ {2} - 2 u _ {t} ^ { \theta } (x) ^ {T} u _ {t} ^ {\mathrm{target}} (x | z) + \| u _ {t} ^ {\mathrm{target}} (x | z) \| ^ {2} - \| u _ {t} ^ {\mathrm{target}} (x | z) \| ^ {2} ] + C _ {1}} \\ & {\overset {(i i i)} {=} \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, x \sim p _ {t} (\cdot | z)} [ \| u _ {t} ^ { \theta } (x) - u _ {t} ^ {\mathrm{target}} (x | z) \| ^ {2} ] + \underbrace {\mathbb {E} _ {t \sim \mathrm{Unif} , z \sim p _ {\mathrm{data}} , x \sim p _ {t} (\cdot | z)} [ - \| u _ {t} ^ {\mathrm{target}} (x | z) \| ^ {2} ]} _ {C _ {2}} + C _ {1}} \\ & {\overset {(i v)} {=} \mathcal {L} _ {\mathrm{CFM}} (\theta) + \underbrace {C _ {2} + C _ {1}} _ {=: C}} \end{array}
$$

其中 (i) 代入了推导出的方程，(ii) 加了又减了同一个值，(iii) 再次使用公式 $\| a - b \| ^ { 2 } = \| a \| ^ { 2 } - 2 a ^ { T } b + \| b \| ^ { 2 }$ ，(iv) 定义了关于 θ 的常数。这完成了证明。

> **译注**（为何只需优化条件损失）：把均方误差展开后，唯一含参数 $\theta$ 的交叉项可借助式 (18) 与式 (12) 的采样过程改写为与 $\mathcal { L } _ { \mathrm { C F M } }$ 完全相同的条件期望，两个损失只相差一个与 $\theta$ 无关的常数；对 $\theta$ 求梯度时该常数自动消失，因此最小化 $\mathcal { L } _ { \mathrm { C F M } }$ 等价于最小化 $\mathcal { L } _ { \mathrm { F M } }$——“显式回归条件向量场＝隐式回归边缘向量场”。 

因此，流匹配训练由最小化条件流匹配损失构成。训练过程总结于算法 3，并在图 7 中可视化。注意这个算法有几个显著特点：首先，我们训练时实际上从不模拟任何 ODE。人们把算法的这个特性称为"无模拟"（simulation-free）。这使得训练极其廉价，因为你不必在训练时展开 ODE 的轨迹（这会花很多步）。其次，训练是一个简单的回归目标——我们只是对 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ 做回归。所以它毕竟与监督学习没有太大不同。最后，算法极其简单——很难想到更简单的训练目标了。这一切使流匹配成为大规模机器学习模型极其有吸引力的方法。一旦 $u _ { t } ^ { \theta }$ 训练好，我们可模拟流模型 

$$
\mathrm{d} X _ {t} = u _ {t} ^ { \theta } (X _ {t}) \mathrm{d} t, \qquad X _ {0} \sim p _ {\mathrm{init}}\tag{27}
$$

例如通过算法 1，得到样本 $X _ { 1 } \sim p _ { \mathrm { d a t a } }$ 。整个流程在文献中被称为流匹配 [25, 27, 1, 26]。让我们把条件流匹配损失实例化到高斯概率路径上： 

算法 3 流匹配训练过程（针对高斯 CondOT 路径 $p_t(x|z) = \mathcal{N}(tz, (1-t)^2)$）
输入: 来自 $z \sim p_{\text{data}}$ 的数据集、神经网络 $u_t^\theta$ 1: 对每个 mini-batch 数据
2: 从数据集采样一个数据例子 $z$。
3: 采样随机时间 $t \sim \text{Unif}_{[0,1]}$。
4: 采样噪声 $\epsilon \sim \mathcal{N}(0, I_d)$ 5: 设 $x = tz + (1-t)\epsilon$（一般情形: $x \sim p_t(\cdot | z)$）
6: 计算损失 $\mathcal{L}(\theta) = \|u_t^\theta(x) - (z-\epsilon)\|^2$（一般情形: $= \|u_t^\theta(x) - u_t^{\text{target}}(x|z)\|^2$）
7: 更新 $\theta \leftarrow \text{grad\_update}(\mathcal{L}(\theta))$。
8: 结束循环 

## 例 13（针对高斯条件概率路径的流匹配）

让我们回到高斯概率路径 $p _ { t } ( \cdot | z ) = \mathcal N ( \alpha _ { t } z ; \beta _ { t } ^ { 2 } I _ { d } )$ 的例子，其中我们可以按 

$$
\epsilon \sim \mathcal {N} (0, I _ {d}) \Rightarrow x _ {t} = \alpha_ {t} z + \beta_ {t} \epsilon \sim \mathcal {N} (\alpha_ {t} z, \beta_ {t} ^ {2} I _ {d}) = p _ {t} (\cdot | z).\tag{28}
$$

从条件路径采样。如我们在方程 (20) 中推导的，条件向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ 由 

$$
u _ {t} ^ {\mathrm{target}} (x | z) = \left(\dot {\alpha} _ {t} - \frac {\dot {\beta} _ {t}}{\beta_ {t}} \alpha_ {t}\right) z + \frac {\dot {\beta} _ {t}}{\beta_ {t}} x,\tag{29}
$$

其中 $\dot { \alpha } _ { t } = \partial _ { t } \alpha _ { t }$ 且 $\dot { \boldsymbol { \beta } } _ { t } = \partial _ { t } \boldsymbol { \beta } _ { t }$ 分别是时间导数。把这个公式代入，条件流匹配损失可写为 

$$
\mathcal {L} _ {\mathrm{CFM}} (\theta) = \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, x \sim \mathcal {N} (\alpha_ {t} z, \beta_ {t} ^ {2} I _ {d})} [ \| u _ {t} ^ { \theta } (x) - \left(\dot {\alpha} _ {t} - \frac {\dot {\beta} _ {t}}{\beta_ {t}} \alpha_ {t}\right) z - \frac {\dot {\beta} _ {t}}{\beta_ {t}} x \| ^ {2} ]\tag{30}
$$

$$
\stackrel {(i)} {=} \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, \epsilon \sim \mathcal {N} (0, I _ {d})} [ \| u _ {t} ^ { \theta } (\alpha_ {t} z + \beta_ {t} \epsilon) - (\dot {\alpha} _ {t} z + \dot {\beta} _ {t} \epsilon) \| ^ {2} ]\tag{31}
$$

其中 (i) 代入方程 (28) 并用 $\alpha _ { t } z + \beta _ { t } \epsilon$ 替代 x。注意 $\mathcal { L } _ { \mathrm { C F M } }$ 的简洁性：我们采样一个数据点 z，采样一些噪声 ϵ，然后取均方误差。对于特例 $\alpha _ { t } = t ,$ 且 $\beta _ { t } = 1 - t$ 让它更加具体。相应的概率 $p _ { t } ( x | z ) = \mathcal { N } ( t z , ( 1 - t ) ^ { 2 } )$ 有时被称为 (高斯) CondOT 概率路径。此时 $\dot { \alpha } _ { t } = 1 , \dot { \beta } _ { t } = - 1$ ，因此 

$$
\mathcal {L} _ {\mathrm{cfm}} (\theta) = \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, \epsilon \sim \mathcal {N} (0, I _ {d})} [ \| u _ {t} ^ { \theta } (t z + (1 - t) \epsilon) - (z - \epsilon) \| ^ {2} ]
$$

许多著名的最先进模型都用这种简单却有效的过程训练，例如 Stable Difusion 3、Meta 的 Movie Gen Video，以及可能还有许多其他专有模型。在图 7 中，我们在一个简单例子中对其进行了可视化，算法 3 总结了训练过程。 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/726d2e2535ccc1af0b4d7c2396956c8d75a524fbe270cd8a3cb484efea7e29db.jpg)



图 7: 用高斯 CondOT 概率路径展示定理 12：从一个训练好的流匹配模型模拟 ODE。数据分布是棋盘图案（右上）。上排：来自 ground truth 边缘概率路径 $p _ { t } ( x )$ 的直方图。下排：来自流匹配模型样本的直方图。可以看到，训练后（在训练误差范围内）上排与下排一致。该模型使用算法 3 训练。


让我们总结本节的结论。 

## 摘要 14（流匹配）

流匹配训练由学习边缘向量场 $u _ { t } ^ { \mathsf { t a r g e t } }$ 构成。要构造它，我们选择一条满足 $p _ { 0 } ( \cdot | z ) = p _ { \mathrm { i n i t } } , p _ { 1 } ( \cdot | z ) = \delta _ { z }$ 的条件概率路径 $p _ { t } ( x | z )$ 。接着，我们找到条件向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ ，使其对应流 $\psi _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ 满足 

$$
X _ {0} \sim p _ {\mathrm{init}} \Rightarrow X _ {t} = \psi_ {t} ^ {\mathrm{target}} (X _ {0} | z) \sim p _ {t} (\cdot | z),
$$

或者等价地，使 $u _ { t } ^ { \mathrm { t a r g e t } }$ 满足连续性方程。那么按 

$$
u _ {t} ^ {\mathrm{target}} (x) = \int u _ {t} ^ {\mathrm{target}} (x | z) \frac {p _ {t} (x | z) p _ {\mathrm{data}} (z)}{p _ {t} (x)} \mathrm{d} z,\tag{32}
$$

定义的边缘向量场跟随边缘概率路径，即 

$$
X _ {0} \sim p _ {\mathrm{init}}, \quad \mathrm{d} X _ {t} = u _ {t} ^ {\mathrm{target}} (X _ {t}) \mathrm{d} t \Rightarrow X _ {t} \sim p _ {t} \quad (0 \leq t \leq 1).\tag{33}
$$

特别地，对该 ODE 有 $X _ { 1 } \sim p _ { \mathrm { d a t a } }$ ，使 $u _ { t } ^ { \mathrm { t a r g e t } }$ "把噪声转化为数据"，正如期望的那样。要学习它，我们最小化条件流匹配损失 

$$
\mathcal {L} _ {\mathrm{CFM}} (\theta) = \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, x \sim p _ {t} (\cdot | z)} [ \| u _ {t} ^ { \theta } (x) - u _ {t} ^ {\mathrm{target}} (x | z) \| ^ {2} ].\tag{34}
$$

最广泛使用的例子是高斯概率路径。对这种情况，公式变为： 

$$
p _ {t} (x | z) = \mathcal {N} (x; \alpha_ {t} z, \beta_ {t} ^ {2} I _ {d})\tag{35}
$$

$$
u _ {t} ^ { \mathrm{flow}} (x | z) = \left(\dot {\alpha} _ {t} - \frac {\dot {\beta} _ {t}}{\beta_ {t}} \alpha_ {t}\right) z + \frac {\dot {\beta} _ {t}}{\beta_ {t}} x\tag{36}
$$

$$
\mathcal {L} _ {\mathrm{CFM}} (\theta) = \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, \epsilon \sim \mathcal {N} (0, I _ {d})} [ \| u _ {t} ^ { \theta } (\alpha_ {t} z + \beta_ {t} \epsilon) - (\dot {\alpha} _ {t} z + \dot { \beta } _ {t} \epsilon) \| ^ {2} ]\tag{37}
$$

其中噪声调度器 $\alpha _ { t } , \beta _ { t } \in \mathbb { R } .$ 是连续可微、单调函数，我们选其使得 $\alpha _ { 0 } = \beta _ { 1 } = 0 \ \alpha _ { 1 } = \beta _ { 0 } = 1 \ ( \text{例如} \ \alpha _ { t } = t , \beta _ { t } = 1 - t )$" 

## 4 得分函数与得分匹配

在上一节中，我们展示了如何用流匹配训练流模型。本节将讨论扩散模型，并演示如何用得分匹配来训练它们。


## 4.1 条件与边缘得分函数

到目前为止，我们研究所关注的核心对象是一个向量场 $u _ { t } ( x )$ 。扩散模型 [45, 44] 采取了一种聚焦于得分函数的不同视角。因此，在本节中，我们将用得分函数的语言重新表述我们所学的内容——从而提供一个全新的视角。设 $q ( x )$ 为任意概率分布。则 $q$ 的得分函数定义为 log q(x)，即 $q$ 的对数似然关于 x 的梯度。得分具有直观的含义： log q(x) 是对数似然的最速上升方向。图 8 对此作了图示。

> **译注**（得分的直觉）：得分 $\nabla \log q ( x )$ 是对数密度上升最快的方向，可理解为指向高概率区的“指南针”。对高斯路径 $p _ { t } ( x | z ) = \mathcal { N } ( \alpha _ { t } z , \beta _ { t } ^ { 2 } I _ { d } )$，得分 $- ( x - \alpha _ { t } z ) / \beta _ { t } ^ { 2 }$ 的方向由含噪样本指向去噪均值 $\alpha _ { t } z$，因此“学得分”与“学去噪”本质上是同一件事；且噪声水平 $\beta _ { t }$ 越大，得分的幅值越小。



q(x)



r <sup>log</sup> <sup>q(x)</sup>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/a69d40179f35d62381fd86cbe3c7f3b78915a54e0311b05f6a72709e37cbfa74.jpg)



图 8：得分函数 $\nabla \log q ( x )$ 的示意：左图为一般概率分布 q(x)，右图将其得分（得分场）绘制为黑色箭头。


让我们回到第 3 节中条件概率路径 $p _ { t } ( x | z )$ 与边缘概率路径 $p _ { t } ( x )$ 的设定。那么我们可以等价地定义条件得分函数为 $\nabla \log p _ { t } ( x | z )$ ，边缘得分函数为 $\nabla \log p _ { t } ( x )$ 。与方程 (18) 类似，边缘得分可以通过条件得分函数 $\nabla \log p _ { t } ( x | z )$ 表示为


$$
\nabla \log p _ {t} (x) = \int \nabla \log p _ {t} (x | z) \frac {p _ {t} (x | z) p _ {\mathrm{data}} (z)}{p _ {t} (x)} \mathrm{d} z.\tag{38}
$$

因此，条件得分与边缘得分之间的关系，与条件向量场和边缘向量场之间的关系是类似的。注意，我们可以通过下式证明方程 (38)


$$
\nabla \log p _ {t} (x) = \frac {\nabla p _ {t} (x)}{p _ {t} (x)} = \frac {\nabla \int p _ {t} (x | z) p _ {\mathrm{data}} (z) \mathrm{d} z}{p _ {t} (x)} = \frac {\int \nabla p _ {t} (x | z) p _ {\mathrm{data}} (z) \mathrm{d} z}{p _ {t} (x)} = \int \nabla \log p _ {t} (x | z) \frac {p _ {t} (x | z) p _ {\mathrm{data}} (z)}{p _ {t} (x)} \mathrm{d} z,\tag{39}
$$

其中两次用到链式法则以及规则 $\partial _ { y } \log y = 1 / y$ 。

例 15（高斯概率路径的得分函数。 ）

对高斯路径 $p _ { t } ( \boldsymbol { x } | \boldsymbol { z } ) = \mathcal { N } ( \boldsymbol { x } ; \alpha _ { t } \boldsymbol { z } , \beta _ { t } ^ { 2 } I _ { d } )$ ，我们可以利用高斯概率密度的形式（见方程 (97)）得到


$$
\nabla \log p _ {t} (x | z) = \nabla \log \mathcal {N} (x; \alpha_ {t} z, \beta_ {t} ^ {2} I _ {d}) = - \frac {x - \alpha_ {t} z}{\beta_ {t} ^ {2}}.\tag{40}
$$

注意高斯概率路径的得分函数是 x 与 z 的线性函数。条件向量场 $\boldsymbol u _ { t } ( \boldsymbol x | \boldsymbol z )$ 也是如此（见方程 (20)）。因此两者之间可以互相转换，如下一个命题所示。

## 命题 1（高斯概率路径的转换公式）

对高斯概率路径 $p _ { t } ( x | z ) = \mathcal N ( \alpha _ { t } z , \beta _ { t } ^ { 2 } I _ { d } )$ ，条件（对应地，边缘）向量场与条件（对应地，边缘）得分由如下恒等式相关联


$$
u _ {t} ^ {\mathrm{target}} (x | z) = a _ {t} \nabla \log p _ {t} (x | z) + b _ {t} x, \quad a _ {t} = \left(\beta_ {t} ^ {2} \frac {\dot {\alpha} _ {t}}{\alpha_ {t}} - \dot {\beta} _ {t} \beta_ {t}\right), \quad b _ {t} = \frac {\dot {\alpha} _ {t}}{\alpha_ {t}}\tag{41}
$$


$$
u _ {t} ^ {\mathrm{target}} (x) = a _ {t} \nabla \log p _ {t} (x) + b _ {t} x.\tag{42}
$$

特别地，我们注意到条件（对应地，边缘）向量场可以从条件（对应地，边缘）得分恢复，反之亦然。

证明。对条件向量场与条件得分，我们可以推得：

$$
u _ {t} ^ {\mathrm{target}} (x | z) = \left(\dot {\alpha} _ {t} - \frac {\dot {\beta} _ {t}}{\beta_ {t}} \alpha_ {t}\right) z + \frac {\dot {\beta} _ {t}}{\beta_ {t}} x \stackrel {(i)} {=} \left(\beta_ {t} ^ {2} \frac {\dot {\alpha} _ {t}}{\alpha_ {t}} - \dot {\beta} _ {t} \beta_ {t}\right) \left(\frac {\alpha_ {t} z - x}{\beta_ {t} ^ {2}}\right) + \frac {\dot {\alpha} _ {t}}{\alpha_ {t}} x = \left(\beta_ {t} ^ {2} \frac {\dot {\alpha} _ {t}}{\alpha_ {t}} - \dot {\beta} _ {t} \beta_ {t}\right) \nabla \log p _ {t} (x | z) + \frac {\dot {\alpha} _ {t}}{\alpha_ {t}} x
$$

其中 (i) 只是做了一些代数运算。通过取积分，对边缘流向量场与边缘得分函数，同样的恒等式也成立：

$$
\begin{array}{r} u ^ {\mathrm{target}} (x) = \int u _ {t} ^ {\mathrm{target}} (x | z) \frac {p _ {t} (x | z) p _ {\mathrm{data}} (z)}{p _ {t} (x)} \mathrm{d} z = \int [ a _ {t} \nabla \log p _ {t} (x | z) + b _ {t} x ] \frac {p _ {t} (x | z) p _ {\mathrm{data}} (z)}{p _ {t} (x)} \mathrm{d} z \\ \stackrel {(i)} {=} a _ {t} \nabla \log p _ {t} (x) + b _ {t} x \end{array}
$$

其中 (i) 使用了方程 (38) 以及后验密度积分为 1 的事实。

命题 1 引人注目之处在于：一旦我们学会了 $u _ { t } ^ { \mathrm { t a r g e t } }$ ，我们就同时学会了得分函数 $\nabla \log p _ { t } ( x )$ ，反之亦然。因此，许多扩散模型改为用神经网络来学习得分函数 $\nabla \log p _ { t } ( x )$ 。我们将在 4.3 节讨论这一点。

## 注 16（得分的重参数化）

方程 (41) 中高斯概率路径的重参数化公式之所以成立，是因为等式两边（条件向量场与条件得分）都是 x 和 z 的线性函数。一旦我们边缘化（边缘向量场与边缘得分），两边都只是后验均值 $\mathbb { E } _ { z | x } \left[ z \right]$ 的一个线性重参数化。因此，任何能够恢复 $\mathbb { E } _ { z \mid x } \left[ z \right]$ 的量都可以反过来用于恢复无条件向量场与得分。此外，从数值/训练稳定性的角度看，这样做甚至可能是更可取的。一个常见的选择是后验均值本身，它通常被称为去噪器。形式上，我们定义条件去噪器与边缘去噪器为

$$
D _ {t} (x | z) = z, \quad D _ {t} (x) = \int z \frac {p _ {t} (x | z) p _ {\mathrm{data}} (z)}{p _ {t} (x)} \mathrm{d} z \stackrel {(i)} {=} \frac {1}{\dot {\alpha} _ {t} \beta_ {t} - \alpha_ {t} \dot {\beta} _ {t}} (\beta_ {t} u _ {t} ^ {\mathrm{target}} (x _ {t}) - \dot {\beta} _ {t} x _ {t}).\tag{43}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/d0b0469b3d2988354a17b37c5bd8bf43d1a78346c5c53254d49b4cf4381b052b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/657940404a0805468fe99c6b2aa715410fe2ace1b3c0dfb74f2b727b51179ed1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/b369477dc180b49c223167ccf6d6901fb7a2f6dab7aaae84d1b055fc4a9683b6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/6b7cbf155e8c8edf2bc90a02852dffefdb243a5f220076d83919d18a51394465.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/526f91e8b1547958f83b19b365ab00954eb6c2402a7a158ba1803dbb29da860b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/2b9f63229eb5540435952bd8f36ca3a576f23fc38fea448f2083899df33cdc2e.jpg)



图 9：定理 17 的示意：用 SDE 模拟概率路径。本图重复了图 6 中的绘图，改用方程 (44) 的 SDE 采样。数据分布 $p _ { \mathrm { d a t a } }$ 用蓝色背景。高斯 $p _ { \mathrm { i n i t } }$ 用红色背景。上排：条件路径。下排：边缘概率路径。可以看到，该 SDE 把来自 $p _ { \mathrm { i n i t } }$ 的样本传输为来自 $\delta _ { z }$ 的样本（对条件路径而言）以及来自 $p _ { \mathrm { d a t a } }$ 的样本（对边缘路径而言）。


这里，(i) 来自与命题 1 等价的推导。去噪器有一个非常直观的解释：它是在给定含噪数据 $x$ 下干净数据 z 的期望值。$^{ a}$ 人们之所以常常把这类模型称为去噪扩散模型，是因为学习 $D _ { t }$ 与学习 $u _ { t } ^ { \mathrm { t a r g e t } }$ 在理论上是等价的。 

<sup>a</sup>思考题：去噪器是否总会输出一个“干净”的数据点？为什么（或为什么不）？这可能取决于什么？ 

## 4.2 用 SDE 采样

到目前为止，我们已经展示了如何通过边缘向量场 $u _ { t } ^ { \mathrm { t a r g e t } }$ 构造一条遵循期望概率路径 $p _ { t }$ 的 ODE 轨迹 $X _ { t }$ 。但这种方法局限于流模型。那么扩散模型呢？借助得分函数，我们现在把该结果推广到 SDE。 

## 定理 17（SDE 推广技巧）

按前面一样定义条件向量场与边缘向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ 与 $u _ { t } ^ { \mathrm { t a r g e t } } ( x )$ 。那么对任意扩散系数 $\sigma _ { t } \geq 0 .$ ，我们可以通过在原 ODE 的动力学上添加如下随机动力学来构造一个 SDE： 

$$
\begin{array}{r l r} & & {X _ {0} \sim p _ {\mathrm{init}}, \qquad \mathrm{d} X _ {t} = u _ {t} ^ {\mathrm{target}} (X _ {t}) \mathrm{d} t + \frac {\sigma_ {t} ^ {2}}{2} \nabla \log p _ {t} (X _ {t}) \mathrm{d} t + \sigma_ {t} \mathrm{d} W _ {t}} \\ & & {\qquad = \Big [ u _ {t} ^ {\mathrm{target}} (X _ {t}) + \frac {\sigma_ {t} ^ {2}}{2} \nabla \log p _ {t} (X _ {t}) \Big ] \mathrm{d} t + \sigma_ {t} \mathrm{d} W _ {t}} \\ & {\Rightarrow} & {X _ {t} \sim p _ {t} \quad (0 \leq t \leq 1).} \end{array}\tag{44}
$$

(45) 

特别地，对该 SDE 有 $X _ { 1 } \sim p _ { \mathrm { d a t a } }$ 。我们注意到，该随机动力学与 Langevin 动力学关系密切，可以理解为在保持边缘分布 $p _ { t }$ 不变的同时注入噪声。我们会在注 20 中简要讨论 Langevin 动力学。

> **译注**（定理 17 的关键：加噪声、补漂移）：若只在 ODE 上附加噪声项 $\sigma _ { t } \mathrm { d } W _ { t }$，分布会被逐渐“抹糊”；定理 17 额外添加的漂移项 $\frac { \sigma _ { t } ^ { 2 } } { 2 } \nabla \log p _ { t } ( X _ { t } )$ 恰好把被扩散抹开的概率“推回去”，使边缘分布 $p _ { t }$ 保持不变。因此扩散系数 $\sigma _ { t }$ 在训练完成后可以任选，相当于调节采样过程的随机性。 

我们在图 9 中展示了定理 17 所描述的动力学。可以看到，轨迹现在呈锯齿状，体现了 SDE 演化的随机性。但如定理 17 所建立的，边缘分布 $p _ { t }$ 保持不变。注意上述结果之所以引人注目，是因为即使已经训练好网络之后，我们仍可以任意选择扩散系数 $\sigma _ { t } \geq 0$ 。理论上，定理 17 对 $\sigma _ { t }$ 的任意选择都成立。但在实践中，我们同时受训练误差（神经网络并不能完美近似边缘向量场与得分）与模拟误差（例如当 $\sigma _ { t } \gg 0 .$ 时，算法 2 中需要取小得不可行的步长）的影响。实践中，对一个固定的已训练模型，存在一个最优的 $\sigma _ { t } \geq 0$ ，可以经验地确定 [23, 1, 28].<sup>2</sup> 

对高斯概率路径而言，通过学习边缘向量场，我们就能免费得到得分函数。 

## 例 18（高斯 SDE 推广技巧）

由命题 1，对高斯概率路径，我们可以完全用得分函数来表达定理 17 中的 SDE： 

$$
\begin{array}{c} X _ {0} \sim p _ {\mathrm{init}}, \quad \mathrm{d} X _ {t} = \left[ \left(a _ {t} + \frac {\sigma_ {t} ^ {2}}{2}\right) \nabla \log p _ {t} (X _ {t}) + b _ {t} X _ {t} \right] \mathrm{d} t + \sigma_ {t} \mathrm{d} W _ {t} \\ \Rightarrow X _ {t} \sim p _ {t} \quad (0 \leq t \leq 1) \end{array}\tag{46}
$$

(47) 

其中 $a _ { t } , b _ { t }$ 的定义与命题 1 中相同。 

在本节余下部分，我们将通过 Fokker-Planck 方程证明定理 17，该方程把 ODE 的连续性方程推广到了 SDE。为此，我们先通过下式定义拉普拉斯算子 $\Delta$ 

$$
\Delta w _ {t} (x) = \sum_ {i = 1} ^ {d} \frac {\partial^ {2}}{\partial x _ {i} ^ {2}} w _ {t} (x) = \mathrm{div} (\nabla w _ {t}) (x),\tag{48}
$$

作用于标量场 $\boldsymbol { w } _ { t } : \mathbb { R } ^ { d }  \mathbb { R }$ 上

## 定理 19（Fokker-Planck 方程）

设 $p _ { t }$ 为一条概率路径，并考虑 SDE 

$$
X _ {0} \sim p _ {\mathrm{init}}, \quad \mathrm{d} X _ {t} = u _ {t} (X _ {t}) \mathrm{d} t + \sigma_ {t} \mathrm{d} W _ {t}.
$$

则 $X _ { t }$ 对所有 $0 \leq t \leq 1$ 具有分布 $p _ { t }$ ，当且仅当 Fokker-Planck 方程成立： 

$$
\partial_ {t} p _ {t} (x) = - \mathrm{div} (p _ {t} u _ {t}) (x) + \frac {\sigma_ {t} ^ {2}}{2} \Delta p _ {t} (x) \quad \mathrm{forall} x \in \mathbb {R} ^ {d}, 0 \leq t \leq 1,\tag{49}
$$

Fokker-Planck 方程的一个自包含证明见 B 节。注意当 $\sigma _ { t } = 0$ 时，可由 Fokker-Planck 方程恢复定理 11。额外的拉普拉斯算子项 $\Delta { p } _ { t }$ 一开始可能不太好理解。熟悉物理的读者会注意到，同样的项也出现在热方程中（热方程事实上是 Fokker-Planck 方程的特例）。热量会通过介质扩散。我们同样也添加了一个扩散过程（不是物理意义上的，而是数学意义上的），因此就添加了这一额外的拉普拉斯算子项。现在让我们用 Fokker-Planck 方程帮助证明定理 17。

> **译注**（Fokker-Planck 与连续性方程）：Fokker-Planck 方程是连续性方程向随机过程的推广：右端 $- \mathrm { d i v } ( p _ { t } u _ { t } )$ 对应确定性的概率“搬运”，新增的 $\frac { \sigma _ { t } ^ { 2 } } { 2 } \Delta p _ { t }$ 对应随机扩散对分布的“抹平”（与热方程同形）；当 $\sigma _ { t } = 0$ 时方程退化为第 3 章的连续性方程。 

定理 17 的证明。由定理 19，我们需要证明方程 (44) 所定义的 SDE 对 $p _ { t }$ 满足 Fokker-Planck 方程。我们可以通过直接计算来做： 

$$
\begin{array}{r l} & {\partial_ {t} p _ {t} (x) \stackrel {(i)} {=} - \mathrm{div} (p _ {t} u _ {t} ^ {\mathrm{target}}) (x)} \\ & {\stackrel {(i i)} {=} - \mathrm{div} (p _ {t} u _ {t} ^ {\mathrm{target}}) (x) - \frac {\sigma_ {t} ^ {2}}{2} \Delta p _ {t} (x) + \frac {\sigma_ {t} ^ {2}}{2} \Delta p _ {t} (x)} \\ & {\stackrel {(i i i)} {=} - \mathrm{div} (p _ {t} u _ {t} ^ {\mathrm{target}}) (x) - \mathrm{div} (\frac {\sigma_ {t} ^ {2}}{2} \nabla p _ {t}) (x) + \frac {\sigma_ {t} ^ {2}}{2} \Delta p _ {t} (x)} \\ & {\stackrel {(i v)} {=} - \mathrm{div} (p _ {t} u _ {t} ^ {\mathrm{target}}) (x) - \mathrm{div} (p _ {t} \left[ \frac {\sigma_ {t} ^ {2}}{2} \nabla \log p _ {t} \right]) (x) + \frac {\sigma_ {t} ^ {2}}{2} \Delta p _ {t} (x)} \\ & {\stackrel {(v)} {=} - \mathrm{div} \left(p _ {t} \left[ u _ {t} ^ {\mathrm{target}} + \frac {\sigma_ {t} ^ {2}}{2} \nabla \log p _ {t} \right]\right) (x) + \frac {\sigma_ {t} ^ {2}}{2} \Delta p _ {t} (x),} \end{array}
$$

其中 (i) 使用了定理 11，(ii) 加减了同一项，(iii) 使用了拉普拉斯算子的定义（方程 (48)），(iv) 使用了 log $\begin{array} { r } { p _ { t } = \frac { \nabla p _ { t } } { p _ { t } } } \end{array}$ ，(v) 使用了散度算子的线性性。上述推导表明，方程 (44) 定义的 SDE 对 $p _ { t }$ 满足 Fokker-Planck 方程。由定理 19，这意味着 $X _ { t } \sim p _ { t }$ 对 $0 \leq t \leq 1$ 成立，正如期望的。 □ 

## 注 20（可选：Langevin 动力学）

上述构造有一个著名的特例，即当概率路径为常数时，亦即 $p _ { t } = p$ 对某个固定分布 p 成立。此时我们取 $u _ { t } ^ { \mathrm { t a r g e t } } = 0$ ，得到 SDE 

$$
\mathrm{d} X _ {t} = \frac {\sigma_ {t} ^ {2}}{2} \nabla \log p (X _ {t}) \mathrm{d} t + \sigma_ {t} d W _ {t},\tag{50}
$$

它通常被称为 Langevin 动力学。$p _ { t }$ 为常数意味着 $\partial _ { t } p _ { t } ( x ) = 0$ 。由定理 17 可直接推出，这些动力学对定理 17 中的静态路径 $p _ { t } = p$ 满足 Fokker-Planck 方程。因此，我们可以得出 p 是 Langevin 动力学的平稳分布： 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/d34e2ad4e721288577bf8950711d611746184cb9576c5e58741981cfc2240b40.jpg)



图 10：上排：在方程 (50) 给出的 Langevin 动力学下演化的粒子，其中 $p ( x )$ 取为具有 5 个模态的高斯混合。下排：对上排所示同样样本做的核密度估计。可以看到，样本的分布收敛到平衡分布 p（蓝色背景）。


$$
X _ {0} \sim p \Rightarrow X _ {t } \sim p (t \geq 0).
$$

与许多马尔可夫过程一样，在相当一般条件下，这些动力学收敛到平稳分布 p。也就是说，如果我们改取 $X _ { 0 } \sim p ^ { \prime } \neq p ,$ 使得 $X _ { t } \sim p _ { t } ^ { \prime } ,$ 则在温和条件下 $p _ { t } \to p .$ 这一事实使 Langevin 动力学极其有用，它相应地成为例如分子动力学模拟，以及贝叶斯统计与自然科学中许多其他马尔可夫链蒙特卡洛（MCMC）方法的基础。特别地，当 p 为高斯分布时，Langevin 动力学的特例即为 Ornstein-Uhlenbeck 过程，它构成了扩散模型最早期表述的基础。

> **译注**（Langevin 动力学的直觉）：Langevin 动力学中，漂移项沿得分方向“爬坡”（把样本推向高概率区），噪声项则不断注入随机性；两者平衡时 p 成为平稳分布。无论初值如何，动力学在温和条件下都收敛到 p——这正是 MCMC 采样器与扩散模型反向过程的理论基础。 

## 注 21（可选：GLASS Flows，用 ODE 的随机演化）

与 ODE 相比，SDE 采样的显著特性在于演化变成了随机的，即初始点 $X _ { 0 }$ 不再完全决定 $t > 0$ 时的 $X _ { t }$ 。或许出人意料的是，通过一个称为 GLASS Flows [20] 的简单采样技巧，也可以纯粹通过 ODE 得到同样的随机转移。这使得我们既可以利用 SDE 的随机性（例如通过搜索算法），又保持 ODE 的高效性。 

## 4.3 得分匹配

剩下还需说明：我们如何学习边缘得分函数 log $p _ { t } ( x )$ 。当然，对高斯概率路径，我们只需按命题 1 对 $u _ { t } ^ { \mathrm { t a r g e t } } ( x )$ 做变换即可。但一般情形呢？事实证明，我们也可以直接学习边缘得分函数。为近似边缘得分 log $p _ { t } ,$ 我们使用一个神经网络，称为得分网络 $s _ { t } ^ { \theta } : \mathbb { R } ^ { d } \times [ 0 , 1 ] \to \mathbb { R } ^ { d }$ 。与之前一样，我们可以设计得分匹配损失与去噪得分匹配损失： 

$$
\begin{array}{r l r l} \mathcal {L} _ {\mathrm{SM}} (\theta) = \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, x \sim p _ {t} (\cdot | z)} \left[ \left| \left| s _ {t} ^ {\theta} (x) - \nabla \log p _ {t} (x) \right| \right| ^ {2} \right] & & \blacktriangleright \mathrm{scorematchingloss} \\ \mathcal {L} _ {\mathrm{CSM}} (\theta) = \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, x \sim p _ {t} (\cdot | z)} \left[ \left| \left| s _ {t} ^ {\theta} (x) - \nabla \log p _ {t} (x | z) \right| \right| ^ {2} \right] & & \blacktriangleright \mathrm{conditionalscorematchingloss} \end{array}
$$

其中区别同样是使用边缘得分 log $p _ { t } ( x )$ 与使用条件得分 log $p _ { t } ( x | z )$ 。与之前一样，我们理想上想最小化得分匹配损失，但我们做不到，因为我们不知道 $\nabla \log p _ { t } ( x )$ 。但与前次类似，去噪得分匹配损失是一个可实际计算的替代方案： 

## 定理 22

得分匹配损失与去噪得分匹配损失仅相差一个常数： 

$$
\mathcal {L} _ {\mathrm{SM}} (\theta) = \mathcal {L} _ {\mathrm{CSM}} (\theta) + C,
$$

其中 C 独立于参数 θ。因此，它们的梯度一致： 

$$
\nabla_ {\theta} \mathcal {L} _ {\mathrm{SM}} (\theta) = \nabla_ {\theta} \mathcal {L} _ {\mathrm{CSM}} (\theta).
$$

特别地，对极小化点 $\theta ^ { * }$ ，将成立 $s _ { t } ^ { \theta ^ { * } } = \nabla \log p _ { t }$ 

证明。注意 log $p _ { t }$ 的公式（方程 (38)）与 $u _ { t } ^ { \mathrm { t a r g e t } }$ 的公式（方程 (18)）形式相同。因此，证明与定理 12 的证明相同，只需把 $u _ { t } ^ { \mathrm { t a r g e t } }$ 替换为 log p<sub>t</sub>。 □ 

## 例 23（去噪扩散模型：高斯概率路径的得分匹配）

让我们把去噪得分匹配损失实例化到 $p _ { t } ( x | z ) = \mathcal N ( \alpha _ { t } z , \beta _ { t } ^ { 2 } I _ { d } )$ 的情形。如我们在方程 (40) 中推导的，条件得分 $\nabla \log p _ { t } ( x | z )$ 具有公式 

$$
\nabla \log p _ {t} (x | z) = - \frac {x - \alpha_ {t} z}{\beta_ {t} ^ {2}}.\tag{51}
$$

代入该公式，条件得分匹配损失变为： 

$\mathcal{L}_{\mathrm{CSM}}(\theta) = \mathbb{E}_{t\sim \mathrm{Unif},z\sim p_{\mathrm{data}},x\sim p_t(\cdot |z)}\left[\left\| s_t^\theta (x) + \frac{x - \alpha_tz}{\beta_t^2}\right\| ^2\right]$ $\stackrel {(i)}{=}\mathbb{E}_{t\sim \mathrm{Unif},z\sim p_{\mathrm{data}},\epsilon \sim \mathcal{N}(0,I_d)}\left[\left\| s_t^\theta (\alpha_tz + \beta_t\epsilon) + \frac{\epsilon}{\beta_t}\right\| ^2\right]$ $= \mathbb{E}_{t\sim \mathrm{Unif},z\sim p_{\mathrm{data}},\epsilon \sim \mathcal{N}(0,I_d)}\left[\frac{1}{\beta_t^2}\left\| \beta_ts_t^\theta (\alpha _t z + \beta_t\epsilon) + \epsilon \right\| ^2\right]$ 

其中 (i) 代入了方程 (28) 并用 $\alpha _ { t } z + \beta _ { t } \epsilon$ 替代 x。注意网络 $s _ { t } ^ { \theta }$ 本质上学习的是预测用于污染数据样本 z 的噪声。这解释了为什么上述训练损失被称为去噪得分匹配。很快人们就意识到，当 $\beta _ { t } \approx 0$ 接近零时，上述损失在数值上不稳定（即去噪得分匹配只有在你加入足够多的噪声时才行得通）。在关于去噪扩散模型的一些早期工作（见 *Denoising Diffusion Probabilistic Models*，[17]）中，人们提出：去掉损失中的常数 $\frac { 1 } { \beta _ { t } ^ { 2 } }$，并按下式把 $s _ { t } ^ { \theta }$ 重参数化为噪声预测网络 $\epsilon _ { t } ^ { \theta } : \mathbb { R } ^ { d } \times [ 0 , 1 ] \to \mathbb { R } ^ { d }$： 

$$
- \beta_ {t} s _ {t} ^ {\theta} (x) = \epsilon_ {t} ^ {\theta} (x) \quad \Rightarrow \quad \mathcal {L} _ {\mathrm{DDPM}} (\theta) = \mathbb {E} _ {t \sim \mathrm{Unif}, z \sim p _ {\mathrm{data}}, \epsilon \sim \mathcal {N} (0, I _ {d})} \left[ \| \epsilon_ {t} ^ {\theta} (\alpha_ {t} z + \beta_ {t} \epsilon) - \epsilon \| ^ {2} \right]
$$

与之前一样，网络 $\epsilon _ { t } ^ { \theta }$ 本质上学习的是预测用于污染数据样本 z 的噪声。在算法 4 中，我们总结了训练过程。 

算法 4 高斯概率路径的得分匹配训练过程

输入（Require）：样本数据集 $z \sim p_{data}$ ，得分网络 $s_{t}^{\theta}$ 或噪声预测器 $\epsilon_{t}^{\theta}$ 1: 对每个 mini-batch 数据执行
2: 从数据集中抽取一个数据样本 z。
3: 抽取一个随机时间 $t \sim Unif_{[0,1]}$ 。
4: 抽取噪声 $\epsilon \sim \mathcal{N}(0, I_{d})$ 5: 设 $x_{t} = \alpha_{t}z + \beta_{t}\epsilon$ （一般情形：$x_{t} \sim p_{t}(\cdot|z)$ ）
6: 计算损失 $\mathcal{L}(\theta) = \|s_{t}^{\theta}(x_{t}) + \frac{\epsilon}{\beta_{t}}\|^{2}$ （一般情形：$= \|s_{t}^{\theta}(x_{t}) - \nabla \log p_{t}(x_{t}|z)\|^{2}$ ）
或者：$\mathcal{L}(\theta) = \|e_{t}^{\theta}(x_{t}) - \epsilon\|^{2}$ 7: 通过在 $\mathcal{L}(\theta)$ 上梯度下降更新模型参数 $\theta$ 。
8: 结束循环

让我们总结本节的结论：

摘要 24（得分函数、得分匹配与随机采样）

设 $p_{t}(x|z), p_{t}(x)$ 为条件概率路径与边缘概率路径。条件得分函数由 $\nabla \log p_{t}(x|z)$ 给出，边缘得分函数由 $\nabla \log p_{t}(x)$ 给出。对每个扩散系数 $\sigma_{t} \geq 0$ ，

下述 SDE 的轨迹遵循概率路径： 

$$
X _ {0} \sim p _ {\mathrm{init}}, \quad \mathrm{d} X _ {t} = \left[ u _ {t} ^ {\mathrm{target}} (X _ {t}) + \frac {\sigma_ {t} ^ {2}}{2} \nabla \log p _ {t} (X _ {t}) \right] \mathrm{d} t + \sigma_ {t} \mathrm{d} W _ {t}\tag{52}
$$


$$
\Rightarrow X _ {t } \sim p _ {t} \quad (0 \leq t \leq 1),\tag{53}
$$

其中 $u _ { t } ^ { \mathrm { t a r g e t } } ( x )$ 为如前所述的边缘向量场（见方程 (18)）。 

得分匹配。为学习边缘得分函数 log $p _ { t } ( x )$ ，我们可以使用得分网络 $s _ { t } ^ { \theta }$ 并通过去噪得分匹配训练它 

$$
\mathcal {L} _ {\mathrm{CSM}} (\theta) = \mathbb {E} _ {z \sim p _ {\mathrm{data}}, t \sim \mathrm{Unif}, x \sim p _ {t} (\cdot | z)} [ \| s _ {t} ^ {\theta} (x) - \nabla \log p _ {t} (x | z) \| ^ {2} ] \qquad \text{（去噪得分匹配损失）}\tag{54}
$$

高斯概率路径。对——最重要的——高斯概率路径 $p _ { t } ( x | z ) ~ =$ $\mathcal { N } ( x ; \alpha _ { t } z , \beta _ { t } ^ { 2 } I _ { d } )$ 情形，没有必要分别训练 $s _ { t } ^ { \theta }$ 与 $u _ { t } ^ { \theta }$ ，因为我们可以用如下公式在两者之间转换： 

$$
u _ {t} ^ {\theta} (x) = a _ {t} s _ {t} ^ {\theta} (x) + b _ {t} x, \quad a _ {t} = \left(\beta_ {t} ^ {2} \frac {\dot {\alpha} _ {t}}{\alpha_ {t}} - \dot {\beta} _ {t} \beta_ {t}\right), b _ {t} = \frac {\dot {\alpha} _ {t}}{\alpha_ {t}}
$$

训练完成后，我们可以模拟如下 SDE 

$$
X _ {0} \sim p _ {\mathrm{init}}, \quad \mathrm{d} X _ {t} = \left[ \left(1 + \frac {\sigma_ {t} ^ {2}}{2 a _ {t}}\right) u _ {t} ^ {\theta} (X _ {t}) - \frac {\sigma_ {t} ^ { 2} b _ {t}}{2 a _ {t}} X _ {t} \right] \mathrm{d} t + \sigma_ {t} \mathrm{d} W _ {t}\tag{55}
$$


$$
= \left[ \left(a _ {t} + \frac {\sigma_ {t} ^ {2}}{2}\right) s _ {t} ^ {\theta} (X _ {t}) + b _ {t} X _ {t} \right] \mathrm{d} t + \sigma_ {t} \mathrm{d} W _ {t}\tag{56}
$$

对任意扩散系数 $\sigma _ { t } \geq 0$ ，得到近似样本 $X _ { 1 } \sim p _ { \mathrm { d a t a } }$ 。最优的 $\sigma _ { t } \geq 0$ 可以经验地找到 

## 5 引导：如何基于提示进行条件生成

到目前为止，我们考虑的生成模型都是无引导的，例如一个图像模型只会生成某个图像。数学上，这意味着我们的模型返回的是来自无条件数据分布 $p _ { \mathrm { d a t a } } ( z )$ 的样本。然而，在大多数情况下，我们的目标不仅仅是生成任意对象，而是基于某些附加信息生成对象。换言之，我们想引导模型生成某一类的对象。例如，可以想象一个生成图像的生成模型，它接收一个文本提示 y，然后生成一张与文本提示 y 相匹配的图像 x。如第 1 节所述，这意味着我们希望从 $p _ { \mathrm { d a t a } } ( z | y )$ 采样，即以 y 为条件的引导数据分布。我们将在本节讨论这一点。 

## 注 25（术语）

为了避免与用“条件”一词来指对 $z \sim p _ { \mathrm { d a t a } }$ 条件化（条件概率路径/向量场）时的符号与术语冲突，我们将使用 guided（引导的）一词专门指对 y（例如文本提示）的条件化。 

## 5.1 朴素引导

首先，我们讨论构建一个引导生成模型的“标准”方式。简短的回答是：我们只需在训练和推理时把输入提示 y 提供给网络，然后一切照旧。我们在下面对此进行形式化。我们设想条件变量或提示 y 驻留在一个空间 $\mathcal{Y}$ 中。当 y 对应一个文本提示时，例如，$\mathcal{Y}$ 是所有文本的空间。当 y 对应某个离散类别标签时，$\mathcal{Y}$ 是离散的。我们对 $\mathcal{Y}$ 不做任何约束。 

我们定义一个引导扩散模型由一个由某个神经网络参数化的引导向量场 $u _ { t } ^ { \theta } ( \cdot | y )$ 与一个时间依赖的扩散系数 $\sigma _ { t }$ 共同构成，二者由

神经网络给出：$u ^ { \theta } : \mathbb { R } ^ { d } \times \mathcal { Y } \times [ 0 , 1 ]  \mathbb { R } ^ { d } , ~ ( x , y , t ) \mapsto u _ {t} ^ {\theta} ( x | y )$ 

固定的： 

$$
\sigma_ {t}: [ 0, 1 ] \to [ 0, \infty), t \mapsto \sigma_ {t}
$$

注意与摘要 7 的区别：我们额外用输入 $y \in \mathcal{Y}$ 来引导 $u _ { t } ^ { \theta }$。对任意这样的 $y \in \mathcal{Y}$，可以从这样的模型中按如下方式生成样本： 

初始化： 

$$
X _ {0} \sim p _ {\mathrm{init}}
$$

▶ 用简单分布（如高斯分布）初始化 

模拟： 

$$
\mathrm{d} X _ {t} = u _ {t} ^ {\theta} (X _ {t} | y) \mathrm{d} t + \sigma_ {t} \mathrm{d} W _ {t}
$$

▶ 从 t = 0 到 t = 1 模拟 SDE。 

目标： 

$$
X _ {1} \sim p _ {\mathrm{data}} (\cdot | y)
$$

▶ 目标是让 $X _ { 1 }$ 像 $p _ { \mathrm { d a t a } } ( \cdot | y )$ 一样分布 

当 $\sigma _ { t } = 0$ 时，我们说这样的模型是一个引导流模型。在以下部分，我们把范围限制在流匹配与流模型上以使表述更简洁，但一切都同样适用于一般情形。 

接下来，我们讨论：如何训练一个引导流模型 $u _ { t } ^ { \theta } ( x | y )$？一个简单的小技巧是固定我们对 $y$ 的选择，并把我们的数据分布取为 $p _ { \mathrm { d a t a } } ( x | y )$ 。那么我们就恢复了与之前一样的无引导生成问题，可以相应地用条件流匹配目标构造生成模型，即

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/004294c8be67cedd7f42de9d1d8682437985256db5c028a1e734e3bc65b76980.jpg)



图 11：基于提示/类别 y =“corgi dog”的图像生成。左：用朴素引导生成的样本——图像与提示匹配得不好。右：用分类器引导且 $w = 4$ 生成的样本。如所示，无分类器引导改善了对提示的遵从性。该图取自 [18]。


$$
\mathbb {E} _ {z \sim p _ {\mathrm{data}} (\cdot | y), x \sim p _ {t} (\cdot | z)} \| u _ {t} ^ {\theta} (x | y) - u _ {t} ^ {\mathrm{target}} (x | z) \| ^ {2}.\tag{57}
$$

注意标签 y 并不影响条件概率路径 $p _ { t } ( \cdot | z )$ 或条件向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | z )$ （尽管原则上我们可以让它依赖于 y）。把期望展开到所有这些 $y$ 的选择上，我们从而得到一个引导的条件流匹配目标 

$$
\mathcal {L} _ {\mathrm{CFM}} ^ {\mathrm{guided}} (\theta) = \mathbb {E} _ {(z, y) \sim p _ {\mathrm{data}} (z, y), t \sim \mathrm{Unif} [ 0, 1 ], x \sim p _ {t} (\cdot | z)} \| u _ {t} ^ {\theta} (x | y) - u _ {t} ^ {\mathrm{target}} (x | z) \| ^ {2}.\tag{58}
$$

方程 (58) 的引导目标与方程 (26) 的无引导目标之间的主要区别之一是，这里我们采样的是 $( z , y ) \sim p _ { \mathrm { d a t a } }$ 而不仅仅是 $z \sim p _ { \mathrm { d a t a } }$ 。原因是我们的数据分布在原则上现在是一个联合分布，例如同时覆盖图像 z 与文本提示 y。在实践中，这意味着方程 (58) 的一个 PyTorch 实现会涉及一个同时返回 z 与 y 的批次的 dataloader。 

## 5.2 无分类器引导

理论上，朴素引导应当能够忠实地生成 $p _ { \mathrm { d a t a } } ( \cdot | y )$ 的样本。然而，很快就通过经验发现，用这种过程生成的图像样本与期望的标签 y 匹配得不够好（见图 11）。这可能有很多原因：模型可能欠拟合（即我们实际上没有学到真正的边缘向量场），或者我们的数据可能不完美（例如来自万维网的文本-图像对有很多错误）。因此，为了真正生成与提示更匹配的样本，我们必须找到一种方式来人为地强化提示变量 y。实现这一点的主要技术被称为无分类器引导（classifier-free guidance），它被广泛用于最先进扩散模型中，我们将在下面讨论它。 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/a49552b6707e3dc97225f727c9a4f054db05916dcb315efa0046535ccedfa345.jpg)



图 12：分类器引导与无分类器引导的示意。分类器引导把引导向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | y )$ 分解为与分类器 log $p _ { t } ( y | x )$ 的梯度之和，并以引导尺度 $w > 1$ 放大分类器项。无分类器引导放大了两个向量场之间的差，从而在不需单独训练一个分类器模型的情况下达到同样的效果。


分类器引导。为简单起见，这里我们聚焦高斯概率路径的情形。回忆方程 (15) 中，高斯条件概率路径由 $p _ { t } ( \cdot | z ) = \mathcal N ( \alpha _ { t } z , \beta _ { t } ^ { 2 } I _ { d } )$ 给出，其中噪声调度器 $\alpha _ { t }$ 与 $\beta _ { t }$ 连续可微、单调，并满足 $\alpha _ { 0 } = \beta _ { 1 } = 0$ 与 $\alpha _ { 1 } = \beta _ { 0 } = 1$ 。进一步，回忆我们可以用命题 1 把引导向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | y )$ 用引导得分函数 $\nabla \log p _ { t } ( x | y )$ 改写成如下形式 

$$
u _ {t} ^ {\mathrm{target}} (x | y) = a _ {t} \nabla \log p _ {t} (x | y) + b _ {t} x,\tag{59}
$$

接下来，注意到 $p _ { t } ( x | y )$ 是一个条件密度。因此，我们可以用贝叶斯公式把引导得分改写为 

$$
p _ {t} (x | y) = \frac {p _ {t} (x) p _ {t} (y | x)}{p _ {t} (y)}\tag{60}
$$


$$
\nabla \log p _ {t} (x | y) = \nabla \log \left(\frac {p _ {t} (x) p _ {t} (y | x)}{p _ {t} (y)}\right) = \nabla \log p _ {t} (x) + \nabla \log p _ {t} (y | x),\tag{61}
$$

其中我们用到了梯度是关于变量 $x$ 求取的，因此 $\nabla \log p _ { t } ( y ) = 0$。于是我们可以改写为 

$$
u _ {t} ^ {\mathrm{target}} (x | y) = b _ {t} x + a _ {t} (\nabla \log p _ {t} (x) + \nabla \log p _ {t} (y | x)) = u _ {t} ^ {\mathrm{target}} (x) + a _ {t} \nabla \log p _ {t} (y | x).
$$

注意上述方程的形状：引导向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | y )$ 是无引导向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x )$ 加上引导变量 y 的似然 $p _ { t } ( y | x )$ 的梯度之和。当人们观察到其图像 x 与提示 y 的匹配不够好时，放大 $\nabla \log p _ { t } ( y | x )$ 项的贡献是一个自然的想法，得到 

$$
\tilde {u} _ {t} (x | y) = u _ {t} ^ {\mathrm{target}} (x) + w a _ {t} \nabla \log p _ {t} (y | x), \quad (\text { 分类器引导 })\tag{62}
$$

其中 $w > 1$ 被称为引导尺度。我们如何学习 log $p _ { t } ( y | x )$ 项？注意它可被视为一种对含噪数据的分类器（即它给出在给定 $x$ 时 $y$ 的对数似然）。因此我们可以简单地通过监督学习来学习它。这就导致了分类器引导 [11, 43]（示意见图 12）。分类器引导在很大程度上已被无分类器引导所取代，这就是为什么我们在这里不再进一步讨论它。不过，它是无分类器引导的基础，我们将在下面看到。最后注意，这是一个启发式：当 $w \neq 1$ 时，有 $\tilde { u } _ { t } ( x | y ) \neq u _ { t } ^ { \mathrm { t a r g e t } } ( x | y )$，也就是说它并不是“真正的”引导向量场。

> **译注**（隐式分类器的观点）：$\nabla \log p _ { t } ( y | x )$ 可理解为“判断含噪样本 $x$ 有多符合提示 $y$”的分类器的对数似然梯度；将其放大 $w > 1$ 倍后，$\tilde { u } _ { t } ( x | y )$ 不再是任何真实分布所对应的向量场，因此引导只是一种启发式，而非精确采样。 

无分类器引导。虽然分类器引导在原理上是可行的，但它带来一些困难：首先，我们需要与一个流/扩散模型一起训练一个分类器——因此我们有 2 个网络而不是 1 个。其次，如果 y 是高维的，例如是文本提示而不仅仅是一个类别，那么 $p _ { t } ( y | x )$ 可能非常难以学习，梯度 $\nabla \log p _ { t } ( y | x )$ 也难以获得。因此，无分类器引导 [18] 被引入。无分类器引导在不需训练单独分类器的情况下，产生与分类器引导理论等价的效果。 

为此，我们可以再次应用等式 

$$
\nabla \log p _ {t} (x | y) = \nabla \log p _ {t} (x) + \nabla \log p _ {t} (y | x)
$$

得到 

$$
\begin{array}{r l} & {\tilde {u} _ {t} (x | y) = u _ {t} ^ {\mathrm{target}} (x) + w a _ {t} \nabla \log p _ {t} (y | x)} \\ & {\qquad = u _ {t} ^ {\mathrm{target}} (x) + w a _ {t} (\nabla \log p _ {t} (x | y) - \nabla \log p _ {t} (x))} \\ & {\qquad = u _ {t} ^ {\mathrm{target}} (x) - (w b _ {t} x + w a _ {t} \nabla \log p _ {t} (x)) + (w b _ {t} x + w a _ {t} \nabla \log p _ {t} (x | y))} \\ & {\qquad = (1 - w) u _ {t} ^ {\mathrm{target}} (x) + w u _ {t} ^ {\mathrm{target}} (x | y).} \end{array}
$$

因此我们可以把放大后的引导向量场 $\tilde { u } _ { t } ( \boldsymbol { x } | \boldsymbol { y } )$ 表示为无引导向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x )$ 与引导向量场 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | y )$ 的线性组合。那么思路可以是既训练一个无引导的 $u _ { t } ^ { \mathrm { t a r g e t } } ( x )$ （例如使用方程 (26)），又训练一个引导的 $u _ { t } ^ { \mathrm { t a r g e t } } ( x | y )$ （例如使用方程 (58)），然后在推理时把它们组合得到 $\tilde { u } _ { t } ( x | y )$ 。“但等等！”你可能会问，“我们岂不是需要训练两个模型！？”事实上，我们可以在一个模型中同时训练两者：我们可以在标签集合中加入一个新的附加 ∅ 标签，表示无条件化。然后我们可以把 $u _ { t } ^ { \mathrm { t a r g e t } } ( x ) = u _ { t } ^ { \mathrm { t a r g e t } } ( x | \varnothing )$。这样一来，我们就不需要训练一个单独的模型来强化某个假想分类器的效果。这种把条件模型与无条件模型合在一起训练（随后再强化条件化）的方法被称为无分类器引导（CFG）[18]（示意见图 12）。

> **译注**（符号更正）：原文 OCR 将“无条件”空标签 $\varnothing$ 误识别为 $\mathcal { D }$（下文式 (64) 附近又误作 $\mathcal { O }$），译文已统一更正为 $\varnothing$，与式 (64) 一致。

> **译注**（CFG 的几何直觉）：把 $\tilde { u } _ { t } ( x | y ) = ( 1 - w ) u _ { t } ^ { \mathrm { t a r g e t } } ( x ) + w u _ { t } ^ { \mathrm { t a r g e t } } ( x | y )$ 改写为 $\tilde { u } _ { t } ( x | y ) = u _ { t } ^ { \mathrm { t a r g e t } } ( x | y ) + ( w - 1 ) \bigl( u _ { t } ^ { \mathrm { t a r g e t } } ( x | y ) - u _ { t } ^ { \mathrm { t a r g e t } } ( x ) \bigr)$，可见它沿“条件场减无条件场”的方向外推 $w - 1$ 倍：$w$ 越大，对提示的遵从越强，但样本多样性下降，甚至可能出现饱和等伪影——这是保真度与提示匹配度之间的权衡。 

注 26（一般概率路径的推导） 

$$
\tilde {u} _ {t} (x | y) = (1 - w) u _ {t} ^ {\mathrm{target}} (x) + w u _ {t} ^ {\mathrm{target}} (x | y),
$$

对任意选择的概率路径都成立，而不仅限于高斯路径。当 $w = 1$ 时，可以很容易验证 $\tilde { u } _ { t } ( x | y ) = u _ { t } ^ { \mathrm { t a r g e t } } ( x | y )$ 我们用高斯路径做的推导只是为了说明这一构造背后的直觉，特别是放大假想“分类器”$\nabla \log p _ { t } ( y | x )$ 的贡献。 

训练与无分类器引导。我们现在必须修改方程 (58) 的引导条件流匹配目标，以考虑 $y = \emptyset$ 的可能性。难点在于采样 $( z , y ) \sim p _ { \mathrm { d a t a } } ,$ 时我们永远得不到 $y = \emptyset$ 。因此我们必须人为地引入 $y = \emptyset$ 的可能性。为此，我们定义某个超参数 η 为丢弃原始标签 $y$ 并以 $\varnothing$ 替换它的概率。我们从而得到 CFG 条件流匹配训练目标 

$$
\mathcal {L} _ {\mathrm{CFM}} ^ {\mathrm{CFG}} (\theta) = \mathbb {E} _ {\square} \| u _ {t} ^ {\theta} (x | y) - u _ {t} ^ {\mathrm{target}} (x | z) \| ^ {2}\tag{63}
$$
$$
\Box = (z, y) \sim p _ {\mathrm{data}} (z, y),   t \sim \operatorname{Unif} [ 0, 1 ],   x \sim p _ {t} (\cdot | z), \text { 以概率 } \eta \text { 将 } y \text { 替换为 } \varnothing\tag{64}
$$

> **译注**（丢弃标签的作用）：以概率 $\eta$ 把 $y$ 替换为 $\varnothing$，可让同一个神经网络兼任条件模型与无条件模型；$\eta$ 通常取 $0.1\text{–}0.2$——太小则无条件分支训练不足，太大则浪费成对标注数据。

算法 5 高斯概率路径 $p_t(x|z) = \mathcal{N}(x; \alpha_t z, \beta_t^2 I_d)$ 的无分类器引导训练
输入（Require）：成对数据集 $(z, y) \sim p_{\text{data}}$，神经网络 $u_t^\theta$ 1: 对每个 mini-batch 数据执行
2: 从数据集中抽取一个数据样本 $(z, y)$ 。
3: 抽取一个随机时间 $t \sim \text{Unif}_{[0,1]}$ 。
4: 抽取噪声 $\epsilon \sim \mathcal{N}(0, I_d)$ 5: 设 $x = \alpha_t z + \beta_t \epsilon$ 6: 以概率 η 丢弃标签：$y \leftarrow \varnothing$ 7: 计算损失 $\mathcal{L}(\theta) = \|u_t^\theta(x|y) - (\dot{\alpha}_t z + \dot{\beta}_t \epsilon)\|^2$ 8: 通过在 $\mathcal{L}(\theta)$ 上梯度下降更新模型参数 $\theta$ 。
9: 结束循环

> **译注**（第 6 步的 $p$ 系原文笔误）：算法第 6 步原文写作“以概率 $p$ 丢弃标签”，但正文定义的丢弃概率超参数为 $\eta$（见式 (64)），译文已径改为 $\eta$。

我们在下面总结我们的发现。 

摘要 27（流模型的无分类器引导） 给定无引导边缘向量场 $u_{t}^{\mathrm{target}}(x|\emptyset)$ 、引导边缘向量场 $u_{t}^{\mathrm{target}}(x|y)$ ，以及引导尺度 $w > 1$ ，我们由 $\tilde{u}_{t}(x|y) = (1 - w)u_{t}^{\mathrm{target}}(x|\emptyset) + wu_{t}^{\mathrm{target}}(x|y).$ 定义无分类器引导向量场 $\tilde{u}_{t}(x|y)$ 。(65) 通过使用同一个神经网络来近似 $u_{t}^{\mathrm{target}}(x|\emptyset)$ 与 $u_{t}^{\mathrm{target}}(x|y)$ ，我们可以利用下述 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/7893998120f7128bd5e0077f5f580e7acd8eb0488ef6a0e102b22ee9c10689d6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/120d4b9c9c76f57823fb98a8e49169705616b69633de3ff1478151604f3dcde0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/546d84d2d4945e30cd54256e3b14860f98b4222450836f967f21e0599e71f3a0.jpg)



图 13：在 MNIST 手写数字数据集上、在不同引导尺度下应用无分类器引导的效果。左：引导尺度设为 $w = 1 . 0 .$ 中：引导尺度设为 $w = 2 . 0$ 。右：引导尺度设为 $w = 4 . 0$ 。你会在实验三中自己生成类似的图像！


无分类器引导 CFM（CFG-CFM）目标，由 $\mathcal{L}_{\mathrm{CFM}}^{\mathrm{CFG}}(\theta)=\mathbb{E}_{\Box}\|u_{t}^{\theta}(x|y)-u_{t}^{\mathrm{target}}(x|z)\|^{2}$ (66) $\Box=(z,y)\sim p_{\mathrm{data}}(z,y), t\sim\mathrm{Unif}[0,1], x\sim p_{t}(\cdot|z),\text{以概率 } \eta \text{ 将 } y \text{ 替换为 } \varnothing$ (67) 给出。 

用通俗的话来说，$\mathcal{L}_{\mathrm{CFM}}^{\mathrm{CFG}}$ 可由下式近似：

$$
\widehat{\mathcal{L}_{\mathrm{CFM}}^{\mathrm{CFG}}(\theta)} = \|u_{t}^{\theta}(x|y) - u_{t}^{\mathrm{target}}(x|z)\|^{2}
$$

▶ 从数据分布采样 $(z,y)$。

▶ 在 $[0,1]$ 上均匀采样 t。

▶ 从条件概率路径 $p_{t}(x|z)$ 采样 x。

▶ 以概率 $\eta$ 把 y 替换为 $\varnothing$ 。

▶ 使模型对条件向量场回归。

在推理时，对固定选择的 y，我们可以通过如下方式采样

初始化：$X_{0}\sim p_{\mathrm{init}}(x)$ 模拟：$\mathrm{d}X_{t}=\tilde{u}_{t}^{\theta}(X_{t}|y)\mathrm{d}t$ ▶ 用简单分布（如高斯分布）初始化

▶ 从 t=0 到 t=1 模拟 ODE。

样本：$X_{1}$ ▶ 目标是让 $X_{1}$ 遵从引导变量 y。 

注意 $X _ { 1 }$ 的分布在 $w > 1$ 时不再一定与 $p _ { \mathrm { d a t a } } ( \cdot | y )$ 精确对齐。 
然而经验上，它对条件变量的对齐更好。因此无分类器引导实际上是一种启发式方法，主要由其出色的经验结果来论证。事实上，几乎任何你看到的 AI 生成的图像或视频都大量依赖引导尺度 $w \geq 4$ 的无分类器引导。在图 11 中，我们在 128x128 的 ImageNet 上展示了基于类别的无分类器引导，如 [18] 所示。类似地，在图 13 中，我们可视化了在对 MNIST 手写数字数据集采样时应用不同引导尺度 w 的无分类器引导的效果。

## 注 28（扩散模型的引导）

把上述讨论从流模型推广到扩散模型是直接的。只需把 $u _ { t } ^ { \theta } ( x | y )$ 替换为 $\tilde { u } _ { t } ^ { \theta } ( x | y )$ ，并按第 4 节讨论的那样用 SDE 采样即可。

## 6 构建大规模图像或视频生成器

在前面的章节中，我们学习了如何训练流匹配或扩散模型以从分布 $p _ { \mathrm { d a t a } } ( x | y )$ 中采样。这个配方是通用的，可应用于多种不同的数据类型与应用。在本节中，我们深入考察大规模图像与视频生成这一特定情形，其中包括 FLUX 2.0、Stable Difusion 3、Nano Banana 与 VEO-3 或 Meta Movie Gen Video 等知名模型。最后，我们将在实验中把所学内容应用于从零构建我们自己的这类模型！本节大体按如下方式组织：

1. 神经网络架构：我们首先讨论原始条件输入——包括时间 $t$ 与引导变量 $y _ { \mathrm { r a w } }$（即离散类别标签或原始文本）——如何被转换（嵌入）为模型 $u _ { t } ^ { \theta } ( x | y )$ 本身可消化的向量形式。然后我们讨论 $u _ { t } ^ { \theta } ( x | y )$ 的流行架构选择，包括 U-Net 与扩散 Transformer。

2. 潜空间：我们讨论变分自编码器，它允许在低维潜空间中进行生成建模，从而实现超高分辨率图像生成。

3. 案例研究：最后，我们将深入考察上面提到的两个最先进图像与视频模型——Stable Difusion 与 Meta MovieGen——让你体验一下在大规模系统中实际是如何做的。

## 6.1 神经网络架构

让我们首先把注意力转向针对图像类模态（例如图像与视频）的流模型与扩散模型的可扩展神经网络架构设计。具体而言，我们将探索（引导）向量场 $u _ { t } ^ { \theta } ( x | y )$ 的任务（参数为 θ）在实践中是如何实现的。注意该神经网络必须有 3 个输入：一个向量 $\boldsymbol { x } \in \mathbb { R } ^ { d }$ 、一个条件变量 $y \in \mathcal { V }$ 与一个时间值 $t \in [ 0 , 1 ]$ ，以及一个输出，即向量 $u _ { t } ^ { \theta } ( x | y ) \in \mathbb { R } ^ { d }$ 。对于低维分布（例如前几节中见过的玩具示例分布），把 $u _ { t } ^ { \theta } ( x | y )$ 参数化为多层感知机（MLP，又称全连接神经网络）就足够了。也就是说，在这个简单设定中，$u _ { t } ^ { \theta } ( x | y )$ 的一次前向传播将把我们的输入 x、y 与 $t$ 拼接起来，并送入 MLP。但对于复杂的高维分布，例如图像、视频与蛋白质上的分布，MLP 很可能不够用，通常需要使用特殊的、面向应用的架构。在剩下的这一小节中，我们将考虑图像（以及推广到视频）的情形。首先，我们将考虑原始条件信息——时间 t 与条件变量 y——如何被嵌入为实际模型可消化的向量形式。其次，我们将考虑此类模型的两种常见架构选择：U-Net [38, 17, 22, 11] 与扩散 Transformer（DiT）[12, 30, 28]。

## 6.1.1 嵌入条件变量

嵌入时间。对于简单的玩具示例模型，把 t 的原始值拼接到输入就足以训练一个表现合理的网络。在实践中，标量时间通常用傅里叶特征嵌入到更高维的空间中，使模型能够更忠实地捕捉高频时间依赖 [46]。具体地，

特征化由下式给出


$$
\mathrm{TimeEmb} (t) = \sqrt {\frac {2}{d}} \left[ \cos (2 \pi w _ {1} t) \quad \dots \quad \cos (2 \pi w _ {d / 2} t) \quad \sin (2 \pi w _ {1} t) \quad \dots \quad \sin (2 \pi w _ {d / 2} t) \right] ^ {T},\tag{68}
$$

其中频率 $w _ { i }$ 按如下方式设定


$$
w _ {i} = w _ {\min} \left(\frac {w _ {\max}}{w _ {\min}}\right) ^ {\frac {i - 1}{d / 2 - 1}}, \qquad i = 1, \ldots , d / 2.\tag{69}
$$

TimeEmb 的这一选择是标准做法，但其精确形式并非严格必需。上面的做法只是获得一个范数归一的 $d$ 维嵌入的便捷方式，即 $\| { \mathrm { T i m e E m b } } ( t ) \| = 1$（因为 $\sin ^ { 2 } + \cos ^ { 2 } = 1$）。 

嵌入类别标签。当 $y _ { \mathrm { r a w } } \in \mathcal { Y } \triangleq \{ 0 , \dots , N \}$ 只是一个类别标签时，最简单的做法通常为 $y _ { \mathrm { r a w } }$ 的 $N + 1$ 个可能取值分别学习一个嵌入向量，并把 $y$ 设为该嵌入向量。这些嵌入的参数应被视为包含在 $u _ { t } ^ { \theta } ( x | y )$ 的参数中，因此在训练中一并学习。

嵌入文本输入。当 $y _ { \mathrm { r a w } }$ 是文本提示时，情况更复杂，方法大多依赖冻结的预训练模型。这类模型被训练为把离散的文本输入嵌入为捕捉相关信息的连续向量。其中一个模型被称为 CLIP（Contrastive Language-Image Pretraining）。CLIP 被训练为学习图像与文本提示的共享嵌入空间，使用一种鼓励图像嵌入靠近其对应提示嵌入、远离其他图像与提示嵌入的训练损失 [34]。因此，我们可以取 $y = \mathrm { C L I P } ( y _ { \mathrm { r a w } } ) \in \mathbb { R } ^ { d _ { \mathrm { C L I P } } }$ 为冻结的预训练 CLIP 模型产生的嵌入。在某些情况下，把整个序列压缩为单一表示可能并不可取。这时，可以考虑用预训练的 Transformer 额外嵌入提示以获得嵌入序列。在条件化时组合多个这样的预训练嵌入也很常见，以便同时获得每个模型的好处 [14, 33]。就我们的目的而言，可以简单假设应用这样的模型后，提示嵌入的形状为


$$
\operatorname{PromptEmbed} (y _ {\mathrm{raw}}) \in \mathbb {R} ^ {S \times k}
$$

## 6.1.2 扩散 Transformer

在深入这些架构的细节之前，让我们回忆一下引言中提到的：一张图像就是一个向量 $x \in \mathbb { R } ^ { C _ { \mathrm { i m a g e } } \times H \times W }$ 。这里 $C _ { \mathrm { i m a g e } }$ 表示通道数（RGB 图像通常有 $C _ { \mathrm { i m a g e } } = 3$ 个颜色通道），H 与 W 分别表示图像在像素意义下的高与宽。一个特别突出的架构类是所谓的扩散 Transformer（DiT）及其变体，它们使用注意力机制来构建网络 [49, 30, 28]。扩散 Transformer 有不同"风味"。我们在这里解释一种通用设计，但要注意，DiT 的具体实例化可能因模型与应用而异。在本节余下部分，我们用 d 表示隐维度，L 表示 Transformer 层数，h 表示每层的注意力头数。扩散 Transformer 基于视觉 Transformer（ViT），其主要思想本质上是：把图像切分为 patch，嵌入 patch 以获得 token 序列，然后通过标准注意力处理这些 token [13]。最后应用一次去 patch 化（depatchification）操作来恢复正确形状的图像。初始的 patch 化操作只是对图像张量 $\boldsymbol { x } \in \mathbb { R } ^ { C \times H \times W }$ 的一个重新组织：

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/8efcb2ee65aeb40d9432bc819934cd8cc2f10ec49e783d49c1abdc0d73baaba6.jpg)



图 14：左：扩散 Transformer 架构概览，取自 [30]。右：对比式 CLIP 损失的示意图，其中学习了一个共享的图文嵌入空间，取自 [34]。


$$
\operatorname{Patchify} (x) \in \mathbb {R} ^ {N \times C ^ {\prime}}
$$

其中 $C ^ { \prime } = C P ^ { 2 } , N = ( H / P ) \cdot ( W / P )$ ，P 为 patch 大小。接下来，我们对输出施加一个线性变换，得到最终的 patch 嵌入


$$
\operatorname{PatchEmb} (x) = \operatorname{Patchify} (x) W \in \mathbb {R} ^ {N \times d}
$$

其中 $W \in \mathbb { R } ^ { C ^ { \prime } \times d }$ 是可学习的权重矩阵。扩散 Transformer 的输入随后是时间嵌入、提示嵌入与 patch 化的图像张量，由下式给出（见 6.1.1 节）：


$$
\begin{array}{r l} & {\tilde {t} = \mathrm{TimeEmb} (t) \in \mathbb {R} ^ {d}} \\ & {\tilde {y} = \mathrm{PromptEmb} (y) \in \mathbb {R} ^ {S \times d}} \\ & {\tilde {x} _ {0} = \mathrm{PatchEmb} (x) \in \mathbb {R} ^ {N \times d}} \end{array}
$$

注意，所有元素现在都具有 Transformer 期望的隐维度。扩散 Transformer 随后通过 DiTBlock 中的 Transformer 层对 $\tilde { x } _ { i }$ 进行迭代更新（对 $i = 0 , \cdots , L - 1$ ）（详见注 29）：


$$
\tilde {x} _ {i + 1} = \mathrm{DiTBlock} (\tilde {x} _ {i}, \tilde {t}, \tilde {y}) \in \mathbb {R} ^ {N \times d} (i = 0, \ldots , L - 1).\tag{70}
$$

其中 L 是层数。最后，一个去 patch 化（depatchification）操作把 DiT 的输出映射回期望的输出形状：


$$
u = \mathrm{Depatchify} (\tilde {x} _ {L} \tilde {W}) \in \mathbb {R} ^ { C \times H \times W},
$$

其中 $\tilde { W } \in \mathbb { R } ^ { d \times C ^ { \prime } }$。最后的张量 u 作为模型的输出，即预测的速度 $u _ { t } ^ { \theta } ( x | y )$。

> **译注**（原文记号照录说明）：原文此处写作 $\tilde { x } _ { N }$ 并称 "N is the number of layers"；但上文以 N 记词元数（$\tilde { x } _ { 0 } \in \mathbb { R } ^ { N \times d }$）、以 L 记层数（$i = 0 , \ldots , L - 1$），原文疑为笔误，译文已按上下文统一改用 L。

## 注 29（DiT Block）

为完整性起见，我们给出单个 DiT 层的简短数学描述。虽然我们试图包含足够细节以让读者对 DiT 模型族有总体理解，但我们提醒读者，这些选择强调关键算法设计而非架构细节。现在，设 $\boldsymbol { x } \in \mathbb { R } ^ { N \times d }$ 表示当前的 patch token 序列（这里 $x = \tilde { x } _ { i }$），设 $y \in \mathbb { R } ^ { S \times d }$ 表示嵌入后的引导变量（这里 $y = \tilde { y }$）。那么，典型的 DiT block 用以下方式更新 x：(i) 对 patch 的自注意力，(ii) 对提示的交叉注意力，以及 (iii) 通过自适应归一化（AdaLN）做时间条件化。

缩放点积注意力。给定查询 $Q \in \mathbb { R } ^ { N \times d _ { h } }$ 、键 $K \in \mathbb { R } ^ { M \times d _ { h } }$ 与值 $V \in \mathbb { R } ^ { M \times d _ { h } }$ ，


$$
\operatorname{Attn} (Q, K, V) = \operatorname{softmax} \left(\frac {Q K ^ {\top}}{\sqrt {d _ {h}}}\right) V \in \mathbb {R} ^ {N \times d _ {h}},
$$

其中 softmax 按行应用。

多头注意力。设 h 表示头数，$\begin{array} { r } { d _ { h } = \frac { d } { h } } \end{array}$ 为每头维度。对每个头 $h \in \{ 1 , \dots , n _ { \mathrm { h e a d s } } \}$ ，学习投影矩阵 $W _ { Q } ^ { ( h ) } , W _ { K } ^ { ( h ) } , W _ { V } ^ { ( h ) } \in \mathbb { R } ^ { k \times d _ { h } }$ 。定义


$$
\mathrm{head} _ {h} (x, z) = \mathrm{Attn} \bigl (x W _ {Q} ^ {(h)}, z W _ {K} ^ {(h)}, z W _ {V} ^ {(h)} \bigr),
$$

其中源序列 z 为


$$
z = x \quad (\text { 对 patch 的自注意力 }), \qquad z = y \quad (\text { 对提示的交叉注意力 }).
$$

拼接各头并施加输出投影 $W _ { O } \in \mathbb { R } ^ { d \times d }$ 


$$
\text { MultiHeadattention } (x, z) = \text { Concat } \bigl (\text { head } _ {1} (x, z), \ldots , \text { head } _ {h} (x, z) \bigr) W _ {O} \in \mathbb {R} ^ {N \times d}.
$$

通过自适应归一化做时间条件化。设 $\widetilde t \in \mathbb { R } ^ { d }$ 为时间步嵌入。DiT 中的一个标准选择是用 $\tilde { t }$ 产生按通道的缩放/平移参数来调制归一化后的激活 [31]。具体地，设 $g :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { 2 d } }$ 为 MLP，并设


$$
(\gamma , \beta) = g (\tilde {t}),
$$

其中 $\gamma , \beta \in \mathbb { R } ^ { d }$ （或根据实现，对不同子层如注意力与 MLP 使用单独的 $( \gamma , \beta )$ 对）。给定 token 矩阵 $\boldsymbol { x } \in \mathbb { R } ^ { N \times d }$ 与归一化算子 $\operatorname { Norm } ( \cdot )$（例如 LayerNorm），

定义带调制的归一化


$$
\operatorname{AdaNorm} _ {\tilde {t}} (x) = (1 + \gamma) \odot \operatorname{Norm} (x) + \beta ,
$$

其中 $\odot$ 表示在 token 维度上带广播的逐元素乘法。

> **译注**：AdaLN（自适应层归一化）的机制：时间步 $t$ 先经时间嵌入得到 $\tilde { t }$，再由小 MLP $g$ 映射出逐通道的缩放与平移参数 $( \gamma , \beta )$；每个 token 先做 $\operatorname { Norm } ( x )$，再按 $( 1 + \gamma ) \odot \operatorname { Norm } ( x ) + \beta$ 调制。这样 $t$ 的信息以逐通道门控的方式注入每一层，主干中无需再拼接时间向量。另：原文此处作 $\operatorname { Norm } ( H )$，疑为笔误，按上下文应为 $\operatorname { Norm } ( x )$（归一化的对象是 token 矩阵 $x$）。

把它们组合起来。组合后的操作，也就是 DiTBlock，由下式给出。


$$
\begin{array}{r l} & x \gets x + g _ {\mathrm{self}} (\tilde {t}) \odot \mathrm{MultiHeadattention} \bigl (\mathrm{AdaNorm} _ {\tilde {t}} (x), \mathrm{AdaNorm} _ {\tilde {t}} (x) \bigr) \\ & x \gets x + g _ {\mathrm{cross}} (\tilde {t}) \mathrm{MultiHeadattention} \bigl (\mathrm{AdaNorm} _ {\tilde {t}} (x), y \bigr) \\ & x \gets x + g _ {\mathrm{MLP}} (\tilde {t}) \mathrm{MLP} \bigl (\mathrm{AdaNorm} _ {\tilde {t}} (x) \bigr), \end{array}
$$

其中 MLP 是逐位置的前馈网络，$g _ { \cdot }$（即 $g _ { \mathrm { s e l f } } , g _ { \mathrm { c r o s s } } , g _ { \mathrm { M L P } }$）是可学习的门控参数。输出 $\boldsymbol { x } \in \mathbb { R } ^ { N \times d }$ 成为下一层的 patch-token 序列（在我们的记号中为 $\tilde { x } _ { i + 1 }$）。最后，我们注意到类别条件化的 DiT（例如实验中所实现的那个）通常更简单，用基于时间与类别的 AdaNorm 条件化来取代交叉注意力层。 

## 6.1.3 U-Net

U-Net 架构 [38] 是 DiT 架构的一种替代架构，是一种特定类型的卷积神经网络。它最初为图像分割而设计，其关键特性是输入与输出都具有图像形状（通道数可能不同）。这使它非常适合参数化向量场 $x \mapsto u _ { t } ^ { \theta } ( x | y )$ ，因为对固定的 $y , t$ ，其输入具有图像形状，输出同样具有图像形状。相应地，U-Net 在关于扩散模型的大量早期文献中得到了广泛使用 [17, 22, 11]。一个 U-Net 由一系列编码器 $\mathcal { E } _ { i }$ 与相应的解码器序列 $\mathcal { D } _ { i }$ 组成，中间还有一个潜处理块，我们称之为 midcoder。<sup>3</sup> 作为例子，让我们走一遍一张图像 $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { 3 \times 2 5 6 \times 2 5 6 }$ （我们取 $\left( C _ { \mathrm { i n p u t } } , H , W \right) = \left( 3 , 2 5 6 , 2 5 6 \right)$）在被 U-Net 处理时所走的路径：


$$
x _ {t} ^ {\mathrm{input}} \in \mathbb {R} ^ {3 \times 2 5 6 \times 2 5 6} \quad \blacktriangleright \text{ U-Net 的输入}
$$

$x _ { t } ^ { \mathrm { l a t e n t } } = \mathcal { E } ( x _ { t } ^ { \mathrm { i n p u t } } ) \in \mathbb { R } ^ { 5 1 2 \times 3 2 \times 3 2 }$ ▶ 通过编码器得到潜变量。 

$$
x _ {t} ^ {\mathrm{mid}} = \mathcal {M} (x _ {t} ^ {\mathrm{latent}}) \in \mathbb {R} ^ {5 1 2 \times 3 2 \times 3 2}
$$

▶ 把潜变量送入 midcoder。 

$$
x _ {t} ^ {\mathrm{output}} = \mathcal {D} (x _ {t} ^ {\mathrm{mid}}) \in \mathbb {R} ^ {3 \times 2 5 6 \times 2 5 6}
$$

▶ 通过解码器得到输出。 

注意，当输入通过编码器时，表示中的通道数增加，而图像的高与宽减小。编码器与解码器通常都由一系列卷积层构成（中间夹有激活函数、池化操作等）。上面没有展示的两点是：第一，输入 $x _ { t } ^ { \mathrm { { i n p u t } } } \in \mathbb { R } ^ { 3 \times 2 5 6 \times 2 5 6 }$ 通常先送入一个初始预编码块以增加通道数，然后再送入第一个编码器块。第二，编码器与解码器通常通过残差连接相连。完整图示见图 15。从高层看，大多数 U-Net 都涉及上面描述的某种变体。但上述某些设计选择在实际实现中可能有所不同。特别地，上面我们选择了一个纯卷积架构，而实践中也常见在编码器与解码器中加入注意力层。U-Net 得名于其编码器与解码器形成的 "U" 形形状（见图 15）。 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/c8aa8d92f985029d561689e64d021b499275cf98bae7a242d8ef069e6854c912.jpg)



图 15：一个简化的 U-Net 架构（像这样的架构在 2025 版本本课程实验 03 中使用了）。 

## 6.2 在潜空间中工作：（变分）自编码器

到目前为止，我们一直在数据空间 $\mathbb { R } ^ { d }$ 上操作。然而，随着分辨率不断提高，直接在该空间建模的成本很快变得难以承受。例如，一张带三个 RGB 颜色通道的 $1 0 2 4 \times 1 0 2 4$ 图像对应总维度 $d = H \cdot W \cdot 3 \approx 3 \times 10 ^ { 6 }$！ 注意对视频而言维度会进一步增加，因为一切都随帧数 T 缩放。正如你可以想象的，在这样的空间上训练很快就不可行。与图像分类不同（其低维输出允许收窄卷积堆叠），我们的基于流的建模方法要求输出 $u _ { t } ^ { \theta } ( x ) \in \mathbb { R } ^ { d }$ 与输入一样大。因此关键问题是：我们如何在合理的内存与计算预算内对高维图像建模？ 

## 6.2.1 标准自编码器

这个问题的一个自然答案在于压缩：例如，真实的图像空间可能恰好位于高维图像空间中一个低得多的流形附近。更具体地，我们可以考虑一个编码器 $\mu _ { \phi } : \mathbb { R } ^ { d }  \mathbb { R } ^ { k }$ 与某个解码器 $\mu _ { \theta } : \mathbb { R } ^ { k }  \mathbb { R } ^ { d }$ ，它们共同把原始图像 x $\in \mathbb { R } ^ { d }$ 映射为潜变量 $z \in \mathbb { R } ^ { k }$ 并反向映射回来。维度 k 通常被选得远小于 d。对于图像，例如 $d = 3 \times 1 0 2 4 \times 1 0 2 4$ ，下采样到例如 $\begin{array} { r } { k = 3 \times \frac { 1 0 2 4 } { 1 6 } \times \frac { 1 0 2 4 } { 1 6 } } \end{array}$ 并不罕见。合起来，$\mu _ { \phi }$ 与 $\mu _ { \theta }$ 被称为自编码器。理想地，$\mu _ { \phi }$ 与 $\mu _ { \theta }$ 被选为取得高重构质量，换言之，使 $\mu _ { \theta } ( \mu _ { \phi } ( x ) )$ 在平均意义上接近 x。相应地，自编码器通常用重构损失训练 

$$
\mathcal {L} _ {\mathrm{Recon}} (\phi , \theta ) = \mathbb {E} _ {x \sim p _ {\mathrm{data}}} \left[ \| \mu_ {\theta} (\mu_ {\phi} (x)) - x \| ^ {2} \right].
$$

它度量原始数据点 x 与其重构 $\mu _ { \theta } ( \mu _ { \phi } ( x ) )$ 之间的平方误差。 

对生成建模的友好性。不幸的是，上述重构损失不足以训练一个"好"的自编码器。回忆我们的最终目标是在潜空间中训练生成模型，目标为潜分布 $p _ { \mathrm { l a t e n t } } ( z )$ ，由 $z = \mu _ { \phi } ( x ) , x \sim p _ { \mathrm { d a t a } }$ 给出。$p _ { \mathrm { d a t a } } ( x )$ 的生成模型可以通过把潜生成模型的输出送入解码器 $\mu _ { \theta }$ 来实现。按目前的形式，自编码器存在一个微妙的问题：我们对 $p _ { \mathrm { l a t e n t } } ( z )$ 几乎没有控制，因此几乎没有保证 $p _ { \mathrm { l a t e n t } } ( z )$ 足够"良好行为"（即规整、简单、类高斯），足以让我们在其上训练这样的生成模型。虽然把数据变换到潜空间可能压缩了它，但我们可能把数据分布 $p _ { \mathrm { d a t a } }$ 变成了一个很难学习的分布 $p _ { \mathrm { l a t e n t } }$ 。因此问题是：我们如何确保潜分布 $p _ { \mathrm { l a t e n t } }$ 仍然性质良好且易于学习？为了允许对潜分布做更显式的正则化，我们现在要把自编码器的概念重新表述在一个更一般的概率框架中，从而得到变分自编码器的概念。 

## 6.2.2 变分自编码器

变分自编码器（VAE）是在（确定性）标准自编码器的基础上，放松编码器与解码器必须为确定性函数这一约束而得到的。具体地，让我们考虑一个参数为 $\phi$ 的编码器 $q _ { \phi } ( z | x )$ 与参数为 θ 的解码器 $p _ { \theta } ( x | z )$ 。最常见的选择是取 

$$
q _ {\phi} (z | x) = \mathcal {N} (z; \mu_ {\phi} (x), \mathrm{diag} (\sigma_ {\phi} ^ {2} (x))), p _ {\theta} (x | z) = \mathcal {N} (x; \mu_ {\theta} (z), \sigma_ {\theta} ^ {2} (z) I _ {d})\tag{71}
$$

其中 $\mu _ { \phi } ( x ) \in \mathbb { R } ^ { k } , \sigma _ { \phi } ^ { 2 } ( x ) \in \mathbb { R } _ { > 0 } ^ { k } , \mu _ { \theta } ( z ) \in \mathbb { R } ^ { d }$ 与 $\sigma _ { \theta } ^ { 2 } ( z ) \in \mathbb { R } _ { \ge 0 }$ 被参数化为神经网络，diag 表示对角矩阵。要编码或解码一个变量，我们采样 

$$
\begin{array}{l} z \sim q _ {\phi} (\cdot | x) \\ x \sim p _ {\theta} (\cdot | z) \end{array}
$$

（编码）

（解码）

最后，我们注意到当 $\sigma _ { \phi } ( x ) = 0$ 与 $\sigma _ { \theta } ( x ) = 0$ 恒成立时，便退化为标准自编码器。让我们看看重构损失是什么样子。一个自然的目标函数是如下： 

$$
\mathcal {L} _ {\mathrm{VAE-Recon}} (\phi , \theta) = - \mathbb {E} _ {x \sim p _ {\mathrm{data}} (x), z \sim q _ {\phi} (\cdot | x)} \left[ \log p _ {\theta} (x | z) \right]
$$

(72) 

注意这两处变化：不再是确定性编码，我们现在采样 $z \sim q _ { \phi } ( z | x )$ 。进一步，我们现在取 x 在解码下的负对数似然，即损失实际上在问：如果对原始数据点 x 先编码再解码，它出现的可能性有多大——由于事情现在变得随机，我们考虑了所有可能的编码/解码。对高斯情形，该重构损失变为： 

$$
\mathcal {L} _ {\mathrm{VAE-Recon}} (\phi , \theta) = \mathbb {E} _ {x \sim p _ {\mathrm{data}} (x), z \sim q _ {\phi} (z | x)} \left[ \frac {1}{2 \sigma_ {\theta} ^ {2} (z)} \| x - \mu_ {\theta} (z) \| ^ {2} + \frac {d}{2} \log \sigma_ {\theta} ^ {2} (z) \right] + \mathrm{const}\tag{73}
$$

其中用到了正态分布的密度（见方程 (97)）。因此，VAE 重构损失与标准 AE 重构损失差别不大，我们只需考虑所有可能的编码 $z \sim q _ { \phi } ( \cdot | x )$ 即可。依赖于解码器方差的第二项控制重构精度与预测不确定性之间的权衡。许多实现（包括实验中的那个）把 $\sigma _ { \phi } ( x )$ 与 $\sigma _ { \theta } ( z )$ 固定为学习到的标量常数（即分别独立于 x 与 z），从而在方差学习时避免病态行为并保持数值稳定。因此，这种情况下的 VAE 重构损失除了编码的随机性与常数外，本质上就是标准自编码器重构损失： 

$$
\mathcal {L} _ {\mathrm{VAE-Recon}} (\phi , \theta) = \mathbb {E} _ {x \sim p _ {\mathrm{data}} (x), z \sim q _ {\phi} (z | x)} \left[ \frac {1}{2 \sigma_ {\theta} ^ {2}} \| x - \mu_ {\theta} (z) \| ^ {2} \right] + \mathrm{const}\tag{74}
$$

> **译注**（式 (73) 从何而来）：该式是把高斯密度（方程 (97)）代入 $-\log p _ { \theta } ( x | z )$ 得到的：密度指数部分给出加权平方误差项 $\| x - \mu _ { \theta } ( z ) \| ^ { 2 } / ( 2 \sigma _ { \theta } ^ { 2 } ( z ) )$，归一化常数给出 $( d / 2 ) \log \sigma _ { \theta } ^ { 2 } ( z )$ 项；若把方差固定为常数，损失即退化为普通均方误差（MSE）。

让我们重新审视目标：我们想为我们的数据分布 $p _ { \mathrm { d a t a } } ( x )$ 创建一个编码，使得把它映射到潜空间后，分布变得"良好"或易于学习。为此，我们现在引入一个关于潜变量 z 的先验分布 $p _ { \mathrm { p r i o r } } ( z )$ 。就我们的目的而言，我们取 $p _ { \mathrm { p r i o r } } = \mathcal { N } ( 0 , I _ { k } )$ 为各向同性高斯。这个先验分布 $p _ { \mathrm { p r i o r } }$ 的选择实际上代表了潜分布"理想"应该长什么样子。高斯分布将非常易于学习，因此满足我们获得"可训练"潜分布的目标。核心思想因此是正则化我们的编码器，确保编码后的数据分布尽可能接近 $p _ { \mathrm { p r i o r } }$，我们通过如下辅助损失实现： 

$$
\mathcal {L} _ {\mathrm{VAE-Prior}} (\phi ) = \mathbb {E} _ {x \sim p _ {\mathrm{data}} (x)} \left[ D _ {\mathrm{KL}} (q _ {\phi} (\cdot | x) \| p _ {\mathrm{prior}}) \right],\tag{75}
$$

其中 $D _ { K L }$ 是 Kullback-Leibler（KL）散度。KL 散度是度量两个概率分布差异的一种基本方式。详细解释它超出了本文范围，但我们会在注 30 中给读者一个简短背景作为提醒。这里定义的损失 ${ \mathcal { L } } _ { \mathrm { V A E - P r i o r } }$ 非常直观：我们希望编码分布对任意数据点 x 看起来都像高斯分布。如果对所有的 $x$ 都这样做，那么我们自然会期望潜分布也像高斯。 

## 注 30（KL 散度背景）

对两个概率密度 $q , p$ ，Kullback-Leibler 散度（KL 散度）定义为 

$$
D _ {\mathrm{KL}} (q (x) \parallel p (x)) = \int q (x) \log \frac {q (x)}{p (x)} = \mathbb {E} _ {X \sim q} \left[ \log \frac {q (X)}{p (X)} \right].
$$

KL 散度是分布间差异（不相似度）的标准度量。特别地，KL 散度满足下列有用性质： 

$$
D _ {\mathrm{KL}} (q (x) \parallel p (x)) \geq 0,\tag{76}
$$

$$
D _ {\mathrm{KL}} (q (x) \parallel p (x)) = 0 \quad \Leftrightarrow \quad q = p.\tag{77}
$$

即它总是非负的，且当且仅当两个概率分布重合时为零。 

为定义变分自编码器的损失函数，我们现在把重构损失与先验损失用一个参数权重 $\beta \geq 0$ 组合起来，得到 VAE 训练目标 

$$
\mathcal {L} _ {\mathrm{VAE}} (\phi , \theta ) = \mathcal {L} _ {\mathrm{VAE-Recon}} (\phi , \theta ) + \beta \mathcal {L} _ {\mathrm{VAE-Prior}} (\phi)\tag{78}
$$

$$
= - \mathbb {E} _ {x \sim p _ {\mathrm{data}} (x), z \sim q _ {\phi} (z | x)} [ \log p _ {\theta} (x \mid z) ] + \beta \mathbb {E} _ {x \sim p _ {\mathrm{data}} (x)} [ D _ {K L} (q _ {\phi} (\cdot | x) | | p _ {\mathrm{prior}}) ]\tag{79}
$$

其中第一项强制潜变量能够高效地解码回数据，第二项强制我们的潜分布接近高斯。参数 $\beta$ 控制每一项的强度。为让这个损失更具体，让我们推导高斯情形的 KL 散度： 

## 例 31（各向同性高斯之间的 KL 散度）

设 $\boldsymbol { q } ( \boldsymbol { x } ) = \mathcal { N } ( \boldsymbol { x } ; \mu _ { q } , \mathrm { d i a g } ( \sigma _ { q } ^ { 2 } ) )$ 与 $p ( x ) = \mathcal { N } ( x ; \mu _ { p } , \mathrm { d i a g } ( \sigma _ { p } ^ { 2 } ) )$ 为具有对角协方差矩阵的高斯，$\sigma _ { q } , \sigma _ { p } \in \mathbb { R } _ { \geq 0 } ^ { d }$，且 $x \in \mathbb { R } ^ { d }$ 。那么 

$$
D _ {\mathrm{KL}} (q \parallel p) = \frac {1}{2} \left(\mathcal {K} \left(\frac {\sigma_ {q} ^ {2}}{\sigma_ {p} ^ {2}}\right) + \frac {\| \mu_ {q} - \mu_ {p} \| ^ {2}}{\sigma_ {p} ^ {2}}\right), \quad \text { 其中 } \mathcal {K} (\alpha) = \sum_ {i = 1} ^ {d} \alpha_ {i} - \log \alpha_ {i} - 1.\tag{80}
$$

上面的表达式是直观的：若均值与方差重合，则 $D _ { \mathrm { K L } } ( q \parallel p ) = 0$ 。进一步，它随均值向量之间的平方误差 $\| \mu _ { q } - \mu _ { p } \| ^ { 2 }$ 增大。最后，函数 $\mathcal { K } ( \alpha )$ 在 $\alpha = 1$ 处有唯一极小值，因此 $D _ { \mathrm { K L } } ( q \parallel p )$ 在 $\sigma _ { q } = \sigma _ { p }$ 时最小。 

> **译注**（KL 闭式解的读法）：式 (80) 中的 $\mathcal { K } ( \alpha ) = \sum _ { i } ( \alpha _ { i } - \log \alpha _ { i } - 1 )$ 逐项满足 $\alpha - \log \alpha - 1 \geq 0$，且仅在 $\alpha = 1$ 时取等；因此 $D _ { \mathrm { K L } } \geq 0$ 且当且仅当两分布均值、方差重合时为零。均值差项 $\| \mu _ { q } - \mu _ { p } \| ^ { 2 } / ( 2 \sigma _ { p } ^ { 2 } )$ 以 $p$ 的方差加权：$p$ 越“宽”，同样的均值差代价越小。取先验为 $\mathcal { N } ( 0 , I _ { k } )$ 时（见式 (82)），损失化为 $\tfrac { 1 } { 2 } \mathcal { K } ( \sigma _ { \phi } ^ { 2 } ) + \tfrac { 1 } { 2 } \| \mu _ { \phi } \| ^ { 2 }$，即“把均值拉向 0、把方差拉向 1”。

证明。我们对 $d = 1$ 做证明（对 $d > 1$ 的证明是类似的，只需对每个维度求和）。给定正态分布的密度，我们知道（见方程 (97)）： 

$$
\log q (x) = - \frac {1}{2} \log (2 \pi \sigma_ {q} ^ {2}) - \frac {1}{2 \sigma_ {q} ^ {2}} \| x - \mu_ {q} \| ^ {2}, \quad \log p (x) = - \frac {1}{2} \log (2 \pi \sigma_ {p} ^ {2}) - \frac {1}{2 \sigma_ {p} ^ {2}} \| x - \mu_ {p} \| ^ {2}
$$

那么 

$$
D _ {\mathrm{KL}} (q \| p) = \mathbb {E} _ {x \sim q} \big [ \log q (x) - \log p (x) \big ] = \frac {1}{2} \log \frac {\sigma_ {p} ^ {2}}{\sigma_ {q} ^ {2}} + \frac {1}{2 \sigma_ {p} ^ { 2}} \mathbb {E} _ {q} \big [ \| x - \mu_ {p} \| ^ {2} \big ] - \frac {1}{2 \sigma_ {q} ^ { 2}} \mathbb {E} _ {q} \big [ \| x - \mu_ {q} \| ^ {2} \big ].\tag{81}
$$

对 $x \sim \mathcal N ( \mu _ { q } , \sigma _ { q } ^ { 2 } I )$ 有 

$$
\mathbb {E} _ {q} \big [ \| x - \mu_ {q} \| ^ {2} \big ] = \mathrm{tr} (\sigma_ {q} ^ {2} I) = \sigma_ {q} ^ {2}.
$$

结合 $x - \mu _ { p } = ( x - \mu _ { q } ) + ( \mu _ { q } - \mu _ { p } )$ 与 $\mathbb { E } _ { q } [ x - \mu _ { q } ] = 0$ ，我们得到 

$$
\mathbb {E} _ {q} \big [ \| x - \mu_ {p} \| ^ {2} \big ] = \mathbb {E} _ {q} \big [ \| x - \mu_ {q} \| ^ {2} \big ] + \| \mu_ {q} - \mu_ {p} \| ^ {2} = \sigma_ {q} ^ {2} + \| \mu_ {q} - \mu_ {p} \| ^ {2}.
$$

代入方程 (81) 即得 (80)。 

让我们现在假设编码器为高斯形式。那么我们得到： 

$$
\mathcal {L} _ {\mathrm{VAE-Prior}} (\phi) = \mathbb {E} _ {x \sim p _ {\mathrm{data}} (x)} \left[ D _ {\mathrm{KL}} \left(q _ {\phi} (\cdot | x) \| \mathcal {N} \left(0, I _ {k}\right)\right) \right] = \mathbb {E} \left[ \frac {1}{2} \mathcal {K} \left(\sigma_ {\phi} ^ {2} (x)\right) + \frac {1}{2} \| \mu_ {\phi} (x) \| ^ {2} \right]\tag{82}
$$

这个损失是直观的：均值 $\mu _ { \phi } ( x )$ 因偏离零而受罚，方差因偏离 1 而受罚。作为 VAE 的总损失，我们得到 

$$
\begin{array}{r l} & {\mathcal {L} _ {\mathrm{VAE}} (\phi , \theta)} \\ & {= \mathcal {L} _ {\mathrm{VAE-Recon}} (\phi , \theta ) + \beta \mathcal {L} _ {\mathrm{VAE-Prior}} (\phi)} \\ & {= \mathbb {E} _ {x \sim p _ {\mathrm{data}} (x), z \sim q _ {\phi} (z | x)} \left[ \underbrace {\frac {1}{2 \sigma_ {\theta} ^ {2} (z)} \| x - \mu_ {\theta} (z) \| ^ {2}} _ {\text{重构误差}} + \underbrace {\frac {d}{2} \log \sigma_ {\theta} ^ {2} (z)} _ {\text{解码器置信度}} + \underbrace {\frac {\beta}{2} \mathcal {K} \left(\sigma_ {\phi} ^ {2} (x)\right)} _ {\text{使潜方差为 1}} + \underbrace {\frac {\beta}{2} \| \mu_ {\phi} (x) \| ^ {2}} _ {\text{使潜均值为 0}} \right]} \end{array}\tag{83}
$$

上述损失函数的四项非常直观：第一项就是重构误差。第二项误差描述解码器的不确定性：更小的方差使解码器更"自信"，但同时也更强烈地惩罚重构误差。进一步，我们希望潜方差为 1、潜均值为 0——强制潜空间中的分布接近高斯。 

训练 VAE。剩下要讨论的是我们如何最小化 VAE 损失 ${ \mathcal { L } } _ { \mathrm { V A E } } ( \phi , \theta )$ 。该损失的问题是，到目前为止我们取期望的分布 $\left( q _ { \phi } ( z | x ) \right)$ 仍依赖于参数 $\phi$。然而，我们可以应用所谓的重参数化技巧来重写它。具体地，对 

$$
q _ {\phi} (z | x) = \mathcal {N} (z; \mu_ {\phi} (x), \sigma_ {\phi} ^ {2} (x) I _ {k})
$$

我们可以用下式得到样本 

$$
\epsilon \sim \mathcal {N} (0, I _ {k}), \quad z = \mu_ {\phi} (x) + \sigma_ {\phi} (x) \epsilon \quad \Rightarrow \quad z \sim q _ {\phi} (\cdot | x)
$$

> **译注**（重参数化技巧为何关键）：未重参数化前，期望的采样分布 $q _ { \phi } ( z | x )$ 本身依赖 $\phi$，梯度无法直接回传。重参数化把随机性“外包”给 $\epsilon \sim \mathcal { N } ( 0 , I _ { k } )$，写成 $z = \mu _ { \phi } ( x ) + \sigma _ { \phi } ( x ) \odot \epsilon$，使 $z$ 成为 $\phi$ 的确定函数；梯度得以穿过 $z$ 回传到 $\mu _ { \phi } , \sigma _ { \phi }$，从而可用标准 SGD 端到端训练。

注意在这个方程中，噪声/随机性的唯一来源是 ϵ，其分布独立于 $\phi$。因此，我们可以把损失重写为： 

$$
\mathcal {L} _ {\mathrm{VAE}} (\phi , \theta ) = \mathbb {E} _ {x \sim p _ {\mathrm{data}} (x), \epsilon \sim \mathcal {N} (0, I _ {k})} \left[ \frac {1}{2 \sigma_ {\theta} ^ {2} (z)} \| x - \mu_ {\theta} (\mu_ {\phi} (x) + \sigma_ {\phi} (x) \epsilon) \| ^ {2} + \frac {d}{2} \log \sigma_ {\theta} ^ {2} (z) + \frac {\beta}{2} \mathcal {K} \left(\sigma_ {\phi} ^ {2} (x)\right) + \frac {\beta}{2} \| \mu_ {\phi} (x) \| ^ {2} \right]
$$

重参数化之后，随机性只来自 $\epsilon \sim \mathcal { N } ( 0 , I _ { k } )$ ，其分布不依赖于 $\phi$。因此，我们可以用深度学习的标准工具最小化这个损失。为更进一步简化，我们可以把 $\sigma _ { \theta } ^ { 2 } ( z )$ 再次取为处处常数 $\sigma ^ { 2 }$ ，得到： 

$$
\mathcal {L} _ {\mathrm{VAE}} (\phi , \theta ) = \mathbb {E} _ {x \sim p _ {\mathrm{data}} (x), \epsilon \sim \mathcal {N} (0, I _ {k})} \left[ \frac {1}{2 \sigma^ {2}} \| x - \mu_ {\theta} (\mu_ {\phi} (x) + \sigma_ {\phi} (x) \epsilon) \| ^ {2} + \frac {\beta}{2} \mathcal {K} \left(\sigma_ {\phi} ^ {2} (x)\right) + \frac {\beta}{2} \| \mu_ {\phi} (x) \| ^ {2} \right]
$$

在算法 6 中，我们总结了 VAE 的训练过程。 

实践说明。我们这里发展的构造展示了自编码器设计的原则。当然，在实践中人们可能会加入更多损失项或其他约束。因此，我们最后给出关于自编码器的实践说明： 

1. 选择 β（与 KL 热身）。大的 $\beta$ 强制潜变量更接近先验，但可能损害重构并引发后验坍塌（编码器忽略 x 并输出 $q _ { \phi } ( z | x ) \approx \mathcal { N } ( 0 , I _ { k } ) )$ 。一种常见的稳定化是 KL 热身：从 $\beta = 0$ 开始，在前几个 epoch 中逐渐增大到目标值。然而，在所有现代自编码器中，$\beta$ 值都非常小，即 $\beta \ll 1$。 

算法 6 β-VAE 训练过程（固定方差的高斯解码器 $p_{\theta}(x|z) = \mathcal{N}(x; \mu_{\theta}(z), \tilde{\sigma}^2 I_d)$）

输入（Require）：样本数据集 $x \sim p_{\text{data}}$ ，编码器网络 $(\mu_{\phi}(x), \log \sigma_{\phi}^2(x))$ ，解码器网络 $\mu_{\theta}(z)$ ，潜维 $k$ ，常数 $\beta \geq 0, \sigma^2 > 0$  
1: 对每个 mini-batch $\{x_i\}_{i=1}^B$ 执行  
2: 编码每个 $x_i$ ：$\mu_i \leftarrow \mu_{\phi}(x_i)$ ，$\log \sigma_i^2 \leftarrow \log \sigma_{\phi}^2(x_i)$  
3: 采样噪声 $\epsilon_i \sim \mathcal{N}(0, I_k)$  
4: 重参数化：$z_i \leftarrow \mu_i + \sigma_i \odot \epsilon_i$ （其中 $\sigma_i = \exp(\frac{1}{2} \log \sigma_i^2)$）  
5: 解码均值：$\hat{x}_i \leftarrow \mu_\theta(z_i)$  
6: 重构损失：$\mathcal{L}_{\text{recon}} \leftarrow \frac{1}{B} \sum_{i=1}^{B} \frac{1}{2\tilde{\sigma}^2} \| x_i - \hat{x}_i \|^2$  
7: 对先验 $p_{\text{prior}}(z) = \mathcal{N}(0, I_k)$ 的 KL 损失：$\mathcal{L}_{\text{KL}} \leftarrow \frac{1}{B} \sum_{i=1}^{B} \frac{1}{2} \sum_{j=1}^{k} (\mu_{i,j}^2 + \sigma_{i,j}^2 - \log \sigma_{i,j}^2 - 1)$  
8: 总损失：$\mathcal{L} \leftarrow \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}}$  
9: 更新 $(\phi, \theta) \leftarrow \text{grad\_update}(\mathcal{L})$  
10: 结束循环

2. 解码器方差。学习高斯解码器方差 $\sigma _ { \theta } ^ { 2 }$ 在数值上很微妙，除非正则化否则可能导致退化解。为稳定，许多实现把 $p _ { \theta } ( x | z ) = \mathcal { N } ( x ; \mu _ { \theta } ( z ) , \sigma ^ { 2 } I _ { d } )$ 固定为常数 $\sigma ^ { 2 }$ ，这使得重构项与均方误差成正比（不计常数）。 

3. 超越像素 MSE 的重构损失。对图像而言，逐像素的高斯似然（均方误差）往往产生过于平滑的重构。实践中，人们加入感知损失（用预训练网络在特征空间计算的损失）来提升锐度与语义保真度。 

4. 对抗与混合目标。为进一步提升视觉真实感，可以把 VAE 目标与对抗损失（VAE-GAN 风格）结合，在解码样本上使用判别器。这通常使输出更锐利，但会引入额外的优化不稳定性与额外超参数。 

## 注 32（在潜空间中工作）

要训练一个潜生成模型，我们只需遵循现有的训练配方，但直接在潜空间中工作。在训练时，我们在 $x \sim p _ { \mathrm { d a t a } }$ 的条件下从 $q _ { \phi } ( z | x )$ 中采样；在推理时，我们从潜扩散或流模型中采样 z，然后用 $x = \mu _ { \mathrm { m e a n } } ( z )$ 解码（注意我们取均值而非随机采样，以避免噪声诱导的伪影）。直观地说，训练良好的自编码器可以被视为滤除高频或语义上无意义的细节，使生成模型能够"聚焦"于重要的、感知相关的特征 [36]。在本文写作之时，几乎所有图像与视频生成的最先进方法都遵循所谓的潜扩散范式，即在自编码器的潜空间中训练一个流或扩散模型 [36, 48]。但需要注意：在训练扩散模型之前，也需要先训练自编码器。至关重要的是，性能现在也依赖于自编码器把图像压缩到潜空间并恢复美观图像的能力。 

我们在 D 节中提供了关于 VAE 的进一步讨论。 

## 6.3 案例研究：Stable Difusion 3 与 Meta Movie Gen

我们通过简要考察两个大规模生成模型来结束本节：用于图像生成的 Stable Difusion 3 与 Meta 的用于视频生成的 Movie Gen Video [14, 33]。正如你将看到的，这些模型使用了我们在本文描述的技术，外加额外的架构增强，以扩展规模并容纳富结构化的条件模态，例如基于文本的输入。 

## 6.3.1 Stable Difusion 3

Stable Difusion 是一系列最先进图像生成模型。这些模型中最早的一批使用大规模潜扩散模型进行图像生成。如果你还没有试过，我们强烈推荐你在线亲自测试（https://stability.ai/news/stable-difusion-3）。 

Stable Difusion 3 使用了与本文所研究的相同的条件流匹配目标（见算法 4）。$^{ 4}$ 如他们论文中所述，他们广泛测试了各种流与扩散替代方案，发现流匹配表现最好。在训练上，它使用了如上述的无分类器引导训练（伴随类别标签丢弃）。进一步，Stable Difusion 3 遵循了第 6.1 节概述的方法，在预训练自编码器的潜空间中训练。训练一个好的自编码器是第一篇 Stable Difusion 论文的一大贡献。 

为增强文本条件化，Stable Difusion 3 使用了 3 种不同类型的文本嵌入（包括 CLIP 嵌入，以及 Google 的 T5-XXL [35] 预训练编码器实例产生的序列输出，类似于 [3, 39] 中采用的方法）。CLIP 嵌入提供输入文本的粗粒度、整体性嵌入，而 T5 嵌入提供细粒度层面的上下文，使模型能够关注条件文本的特定元素。为容纳这些序列上下文嵌入，作者提出把扩散 Transformer 扩展为不仅关注图像 patch，也关注文本嵌入，从而把条件化能力从 DiT 最初提出的基于类别的方案扩展到序列上下文嵌入。所提出的这个修改版 DiT 被称为多模态 DiT（MM-DiT），示意见图 16。他们最终的、最大的模型有 80 亿参数。在采样上，他们使用 50 步（即必须评估网络 50 次），采用欧拉模拟方案与 2.0-5.0 之间的无分类器引导权重。 

## 6.3.2 Meta Movie Gen Video

接下来，我们讨论 Meta 的视频生成器 Movie Gen Video（https://ai.meta.com/research/movie-gen/）。由于数据不是图像而是视频，数据 x 位于空间 $\mathbb { R } ^ { T \times C \times H \times W }$ ，其中 T 代表新的时间维度（即帧数）。正如我们将看到的，在这个视频设定中做出的许多设计选择都可以视为把已有技术（例如自编码器、扩散 Transformer 等）从图像设定适配来处理这个额外时间维度。 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/e5ceb12b6e1bc56aab875adbc737fb69fda60396352cb67f2f3770ca82a7fe5f.jpg)



图 16：[14] 中提出的多模态扩散 Transformer（MM-DiT）架构。该图也取自 [14]。 


Movie Gen Video 使用条件流匹配目标，并采用与前文相同的直线调度器 $\alpha _ { t } = t , \sigma _ { t } = 1 - t$ 。与 Stable Difusion 3 类似，Movie Gen Video 也在冻结的预训练自编码器的潜空间中工作。注意，为降低内存消耗而使用自编码器对视频比对图像更重要——这就是为什么目前大多数视频生成器在生成视频长度上相当有限。具体地，作者提出通过引入一个时间自编码器（TAE）来处理新增的时间维度，它把原始视频 $x _ { t } ^ { \prime } \in \mathbb { R } ^ { T ^ { \prime } \times 3 \times H \times W }$ 映射为潜变量 $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { T \times C \times H \times W }$ ，其中 $\begin{array} { r } { \frac { T ^ { \prime } } { T } = \frac { H ^ { \prime } } { H } = \frac { W ^ { \prime } } { W } = 8 \ [ \mathrm { 3 3 } ] }$ 为容纳长视频，提出了一个时间分块（temporal tiling）过程：把视频切分为片段，每个片段单独编码，潜变量拼接起来 [33]。模型本身——即 $u _ { t } ^ { \theta } ( x _ { t } )$ ——由一个 DiT 类骨干网络给出，其中 $x _ { t }$ 沿时间与空间维度做 patch 化。图像 patch 随后通过一个 Transformer 处理，该 Transformer 同时使用图像 patch 之间的自注意力与语言模型嵌入的交叉注意力，类似于 Stable Difusion 3 使用的 MM-DiT。在文本条件化上，Movie Gen Video 使用三种文本嵌入：用于细粒度基于文本推理的 UL2 嵌入 [47]、用于关注字符级细节（例如提示明确要求某段文本出现）的 ByT5 嵌入 [50]、以及在共享图文嵌入空间中训练的 MetaCLIP 嵌入 [24, 33]。他们最终的、最大的模型有 300 亿参数。如需更详细、更全面的处理，我们鼓励读者查看 Movie Gen 技术报告本身 [33]。 




## 7 离散扩散模型：用扩散构建语言模型

在前面的章节中，我们探讨了欧氏空间 $\mathbb { R } ^ { d }$ 上的流模型与扩散模型，它们作为生成模型，允许我们生成由向量 $z \in \mathbb { R } ^ { d }$ 表示的数据点。然而，并非所有数据都天然地适合建模为欧氏空间 $\mathbb { R } ^ { d }$ 中的点。许多数据类型，例如文本或 DNA，更自然地被视为离散状态空间 S 中的元素。最重要的是，语言由离散的词元序列组成，而我们要对它们进行建模。我们如何把流模型和扩散模型应用于这类数据？事实上，前几章学到的原理同样适用于这类数据类型。所得到的模型在机器学习文献中被称为离散扩散模型 [5, 16]。但重要的是要意识到，并不存在数学意义上的扩散过程（SDE 在离散状态空间中并不存在）。我们用连续时间马尔可夫链（CTMC）来替代 ODE/SDE。下面，我们将解释 CTMC 模型（见第 7.1 节）以及如何去学习它们（见第 7.2 节），从而让我们能够运用流模型与扩散模型的原理来构建大语言模型（LLM）。

## 7.1 连续时间马尔可夫链（CTMC）模型

在本节中，我们介绍连续时间马尔可夫链（CTMC）。你可以把 CTMC 理解为 SDE 的离散类比，我们可以用它来构建生成离散状态的神经网络模型。此外，我们将引入 CTMC 模型，即利用 CTMC 生成文本等离散序列的神经网络模型。

让我们先刻画我们的状态空间 S。设 $\mathcal { V } = \{ v _ { 1 } , \cdots , v _ { V } \}$ 为词表。状态空间由 $S \ = \ \mathcal { V } ^ { d }$ 给出，其中 $d \in \mathbb { N }$ 是序列长度，$V ~ \in ~ \mathbb { N }$ 是词表大小。对于语言而言，$\{ v _ { 1 } , \cdots , v _ { V } \}$ 可以是字母表或一组离散词元的枚举，而 S 则表示长度为 $d$ 的序列（或句子）集合。对于 DNA，$\{ v _ { 1 } , \cdots , v _ { V } \}$ 可以是全部 4 种 DNA 碱基，而 S 则是所有长度为 d 的 DNA 序列。

接下来，设 $X _ { t }$ 为 $S$ 上的一个随机过程，即 S 中的一条随机轨迹 $X : [ 0 , 1 ]  S , t \mapsto X _ { t }$ 。我们要求 $X _ { t }$ 是一个马尔可夫过程，即一个没有记忆的过程。具体来说，这意味着以下条件成立

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/1b6537bd71a5e83fb754d621c98beccb697ea7c6662219e8c4d3be9057d7ea2b.jpg)



图 17：状态空间为 $S = \{ S _ { 1 } , S _ { 2 } , S _ { 3 } \}$ （序列长度 $d = 1 )$ 的 CTMC 轨迹示意图。该图改编自 [5]。


$$
\underbrace {p \left(X _ {t + h} \mid X _ {t} , X _ {t _ {1}} , \cdots , X _ {t _ {k}}\right)} = \underbrace {p \left(X _ {t + h} \mid X _ {t}\right)} \quad (\text { for   all } 0 <   h, 0 \leq t _ {1} <   t _ {2} <   \dots <   t _ {k} <   t)
$$

换言之，未来事件的概率只取决于当前状态——过去对未来的发生不再有任何影响。注意，ODE/SDE 虽然不在离散状态空间上，但同样是马尔可夫过程。这里 X 位于一个离散空间上，因此被称为马尔可夫链，更具体地说是连续时间马尔可夫链（CTMC）。量 $p _ { t + h | t } ( X _ { t + h } | X _ { t } )$ 是转移概率，它们与马尔可夫链的初始分布 $X _ { 0 } \sim p _ { 0 }$ 一起完全决定了该 CTMC。因此，当我们说 CTMC 时，你也可以直接把它理解为转移概率 $p _ { t + h | t } ( X _ { t + h } | X _ { t } )$ 。

接下来，让我们推导向量场在离散设定下的类比物。由于我们处于离散设定中，状态之间只能跳跃（或切换）——我们不能再像指定 ODE 时那样"朝某个方向前进"了。因此，我们定义一个速率矩阵 $Q _ { t } ( y | x )$ ，它实际上总结了从状态 $x \in S$ 跳跃（或切换）到状态 $y \in S$ 的速率。形式上，速率矩阵 $Q _ { t }$ 由一个有界函数（关于时间连续）给出

$$
Q: S \times S \times [ 0, 1 ] \to \mathbb {R}, \quad (x, y, t) \mapsto Q _ {t} (y | x)\tag{84}
$$

其中 $Q _ { t } ( y | x )$ 描述从 x 切换到 y 的速率，使得

$$
(1) \text {   Outgoing   rates   are   positives:   } Q _ {t} (y | x) \geq 0 \quad \text { whenever   } x \neq y\tag{85}
$$

$$
\text {(2) Rate staying equals negative outgoing rate:} Q _ {t} (x | x) = - \sum_ {y \neq x} Q _ {t} (y | x) \quad \text { for   all } x\tag{86}
$$

这两个条件都很直观：第一个条件说，从 x 切换到另一个状态 $y \neq x$ 的速率只能是非负的（不切换对应于 0，因此速率小于 0 没有意义）。第二个条件说，停留在 x 的速率 $Q _ { t } ( x | x )$ 应当与离开 x 的速率相抵消——它本质上是一个一致性条件，说明你要么停留在 x，要么离开（没有第三种选择）。注意这些条件特别意味着 $Q _ { t } ( x | x ) \leq 0$ 。因此，$Q _ { t } ( y | x )$ 是一个矩阵，其对角线元素全部非正，而非对角线元素全部非负。

> **译注**（速率矩阵＝离散世界的“向量场”）：速率矩阵是向量场在离散状态空间中的对应物：对小的时间步长 $h$ ，$Q _ { t } ( y | x ) h$ 近似给出 $h$ 内从 $x$ 跳到 $y$ 的概率；对角元非正（每行和为 0）正是“要么离开、要么留下”的概率守恒要求。后续所有“回归目标向量场”的推理，都可平行地替换为“回归速率矩阵”。

现在我们定义了微分方程的类比物，即 CTMC "遵循"速率矩阵的条件。其基本思想是 X 的分布或演化应当遵循速率矩阵 $Q _ { t }$ 。换句话说，我们要求转移概率满足

$$
\frac {\mathrm{d}}{\mathrm{d} h} p _ {t + h | t} (X _ {t + h} = y | X _ {t} = x) _ {| h = 0} = Q _ {t} (y | x) \quad \text { for   all } x, y \in S, 0 \leq t\tag{87}
$$

左边是从 $x$ 切换到 $y$ 的概率的无穷小变化速率。我们施加的条件是：这些概率的变化方式应当由速率矩阵规定。让我们简要检查一下要求这些条件是否合理，即如果我们简单地按方程 (87) 设定 $Q _ { t } ( y | x )$ ，它是否是一个合法的速率矩阵？当 $h = 0$ 时，从 x 切换到 y = x 的概率为零（因为时间没有流逝），即对所有 $y \neq x$ 有 $p _ { t | t } ( y | x ) = 0$ 。因此，我们知道该导数必须非负，即只要 $y \neq x$ 就有 $Q _ { t } ( y | x ) \geq 0$ 。这验证了方程 (85) 中的第一个条件成立。此外，我们知道

$$
\sum_ {y \neq x} Q _ {t} (y | x) = \sum_ {y \neq x} \frac {\mathrm{d}}{\mathrm{d} h} p (X _ {t + h} = y | X _ {t} = x) _ {| h = 0} = \frac {\mathrm{d}}{\mathrm{d} h} \sum_ {y \neq x} p (X _ {t + h} = y | X _ {t} = x) _ {| h = 0} = \frac {\mathrm{d}}{\mathrm{d} h} (1 - p (X _ {t + h} = x | X _ {t} = x))
$$

其中我们用到了概率之和为 1 的事实。这验证了方程 (86)。这检查了每个 CTMC 至少有一个满足方程 (87) 的速率矩阵。但反过来呢——如果我们指定了 $Q _ { t }$ ，是否存在对应的 CTMC？如果存在，它是否唯一？确实如此。

## 定理 33（CTMC 的存在性与唯一性）

对任意速率矩阵 $Q _ { t }$ （有界且关于时间 $t$ 连续），存在唯一的马尔可夫链 $X _ { t }$ （即唯一的转移概率集合 $p _ { t + h | t } ( y | x )$ ）使得方程 (87) 成立。

对于感兴趣的读者，我们在 C 节给出了一个自包含的证明。该定理的核心结论是：就机器学习目的而言，我们可以构造速率矩阵 $Q _ { t }$（例如用神经网络），并假设存在唯一马尔可夫链与之对应。

## 例 34（跳跃速率相等的两状态 CTMC）

设 $S = \{ a , b \}$ 并考虑一个时齐的 $\mathrm { C T M C } \left( X _ { t } \right) _ { t \geq 0 }$ ，它在两个状态之间以恒定速率 $\lambda > 0 \mathrm { { ; } }$ 相互切换：

$$
Q = \begin{array}{c c c} & a & b \\ \hline a & - \lambda & \lambda \\ b & \lambda & - \lambda \end{array} .
$$

那么时间增量 $h \geq 0$ 上的转移概率也关于时间 t 恒定，且由

$$
\left( \begin{array}{c c} p (X _ {t + h} = a | X _ {t} = a) & p (X _ {t + h} = a | X _ {t} = b) \\ p (X _ {t + h} = b | X _ {t} = a) & p (X _ {t + h} = b | X _ {t} = b) \end{array} \right) = \frac {1}{2} \left( \begin{array}{c c} 1 + e ^ {- 2 \lambda h } & 1 - e ^ {- 2 \lambda h } \\ 1 - e ^ {- 2 \lambda h } & 1 + e ^ {- 2 \lambda h } \end{array} \right).
$$

可以手工验证方程 (87) 成立，即这些转移概率确实是对应该速率矩阵的正确转移概率。事实上，这些速率非常直观：该链以瞬时速率 $\lambda .$ 持续翻转。指数项 $e ^ { - 2 \lambda h }$ 刻画了初始状态的记忆如何衰减。当时间趋于无穷，即 $h \to \infty$ 时，有

$$
P (h) \to \left( \begin{array}{c c} \frac {1}{2} & \frac {1}{2} \\ \frac {1}{2} & \frac {1}{2} \end{array} \right),
$$

因此该链忘记了它的起点，并以概率 $1 / 2$ 处于 a 或 b 。切换速率 $\lambda > 0$ 越高，收敛越快。

CTMC 的模拟。接下来，让我们思考如何模拟一条 CTMC 的轨迹。设 $h > 0$ 为步长，$p _ { \mathrm { i n i t } }$ 为 $S$ 上的初始分布，例如 $p _ { \mathrm { i n i t } } = \mathrm { U n i f } _ { S }$ 是 $S$ 上的均匀分布。那么我们可以迭代地模拟它：设 $X _ { 0 } \sim p _ { \mathrm { i n i t } }$ 并设置

$$
X _ {t + h} \sim p _ {t + h | t } (\cdot | X _ {t})
$$

如果知道 $p _ { t + h | t } ( \cdot | X _ { t } )$ ，这就可以实现。然而，对于最简单 CTMC 之外的所有情况，我们通常并不知道转移核的闭式形式，只能访问速率矩阵 $Q _ { t }$ 。尽管如此，由方程 (87) 有：

$$
p _ {t + h | t} (X _ {t + h} = y | X _ {t} = x) = p _ {t | t} (X _ {t} = y | X _ {t} = x) + h Q _ {t} (y | x) + R _ {t} (h) = 1 _ {y = x} + h Q _ {t} (y | x) + R _ {t} (h)
$$

其中 $R _ { t } ( h )$ 是我们在 h 较小时可以忽略的误差项。因此，对于较小的 $h$ ，我们可以设

$$
p _ {t + h | t} (X _ {t + h} = y | X _ {t} = x) \approx 1 _ {y = x} + h Q _ {t} (y | x) =: \tilde {p} _ {t + h | t} (y | x)
$$

由我们对速率矩阵施加的条件可以验证，$\tilde { p } _ { t + h | t } ( y | x )$ 在 h 较小时确实是一个合法的概率分布。因此，我们可以近似地通过

$$
X _ {t + h} \sim \tilde {p} _ {t + h | t} (\cdot | x) = (1 _ {y = x} + h Q _ {t} (y | x)) _ {y \in S}\tag{88}
$$

采样下一个点。由于上面只是一个离散分布，我们可以用标准方法轻松地从中采样。这是一种模拟 CTMC 的简单方法。

CTMC 模型。接下来，让我们定义如何用神经网络参数化一个 CTMC。CTMC 模型（或离散扩散模型）由 $S$ 上的初始分布 $p _ { \mathrm { i n i t } }$ 和一个参数为 θ 的神经网络 $Q _ { t } ^ { \theta }$ 给出，使得对每个输入 $x \in S$ ，模型返回速率矩阵的一整列

$$
x \mapsto \{Q _ {t} ^ {\theta} (y | x) \} _ {y \in S}
$$

我们要求模型返回一整列，因为模拟 CTMC（方程 (88)）需要它，即采样下一个状态。

上述模型的一个复杂之处在于 $S$ 可能非常大。特别是，$| S | = V ^ { d }$ ，其中 V 是词表大小，d 是序列长度。这种指数级增长使得在内存中存储速率矩阵的整一列基本上是不可能的——$\{ Q _ { t } ^ { \theta } ( y | x ) \} _ { y \in S }$ 永远无法在计算机中表示。因此，我们必须约束模型。具体来说，几乎所有 CTMC 模型都是因子化的（见图 18），这实际上是一种稀疏性约束。具体地，一个因子化的 CTMC 模型由一个 CTMC 模型 $Q _ { t } ^ { \theta }$ 给出，使得对所有 $y = ( y _ { 1 } , \cdot \cdot \cdot , y _ { d } ) , x = ( x _ { 1 } , \cdot \cdot \cdot , x _ { d } ) \in S = \mathcal { V } ^ { d }$ 有

当 $y$ 与 $x$ 在多于一个位置 $i$ 上不同（即 $y _ { i } \neq x _ { i }$）时，$Q _ { t } ^ { \theta } ( y | x ) = 0$

我们将所有与 x 至多在一个词元上不同的 $y$ 称为 x 的邻居 $N ( x )$ 。这样的因子化 CTMC 模型可以写为

$$
x \mapsto \{Q _ {t} ^ {\theta} (y | x) \} _ {y \in N (x)} = \left( \begin{array}{c c} Q _ {t} ^ {\theta} (v _ {1}, 1 | x) & \dots Q _ {t} ^ {\theta} (v _ {V}, 1 | x) \\ \dots \\ Q _ {t} ^ {\theta} (v _ {1}, d | x) & \dots Q _ {t} ^ {\theta} (v _ {V}, d | x) \end{array} \right)
$$

其中 $Q _ { t } ( y | x ) = Q _ { t } ^ { \theta } ( v _ { i } , j | x )$ 现在给出从 $\boldsymbol { x } = \left( x _ { 1 } , \cdots , x _ { d } \right)$ 跳到通过将第 j 个元素替换为 $v _ { i }$ 所得到的 x 的邻居的速率，即 $y = ( x _ { 1 } , \cdot \cdot \cdot , x _ { j - 1 } , v _ { i } , x _ { j + 1 } , \cdot \cdot \cdot , x _ { d } )$ 。每一行对应每个位置 $i = 1 , \cdots , d$ 的一个速率矩阵，即我们要求

$$
Q _ {t} ^ {\theta} (v, i | x) \geq 0 \text {   if   } v \neq x _ {i}, Q _ {t} (x _ {i}, i | x) = - \sum_ {v \neq x _ {i}} Q _ {t} ^ {\theta} (v, i | x)
$$

我们可以轻松地在神经网络输出上强制这些条件，例如可以使用一个在序列长度 d 上、输出维度为 V 的 transformer 模型。还要注意，因子化的速率矩阵使得输出形状为 $d \times V$ ——这个尺寸随维度线性增长（而非指数增长）。

模拟 CTMC 模型。要从一个 CTMC 模型中采样，我们采样 $X _ { 0 } \sim p _ { \mathrm { i n i t } }$ 并执行一个迭代过程，按方程 (88) 采样下一个状态。我们在算法 7 中给出了一个算法。如

$$
\text { General   Rate   Matrix } \quad Q _ {t} (y | x) \xrightarrow {y} x
$$

$$
\text { Factorized   Rate   Matrix } \quad y \rightarrow Q _ {t} (y | x) \quad x \rightarrow \left(\begin{array}{c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c c}\end{array}\right)
$$

图 18：因子化 CTMC 模型的示意图。因子化的 CTMC 只有当起点和终点在至多一个维度上（这里 d = 2）不同时才有非零速率（$Q _ { t } ( y | x ) \neq 0$ ）。该图取自 [26]。

所示，对于因子化的 CTMC 模型，可以使用并行的逐词元 Euler 近似，其中每个词元在较小步长 $h > 0$ 时独立更新。该近似与完整的 CTMC Euler 步在 h 的一阶意义上一致，但允许对多个词元进行 $O ( h ^ { 2 } )$ 概率的同时更新。

算法 7 从因子化 CTMC 模型中采样

输入（Require）：速率网络 $Q_t^\theta$（因子化）、初始分布 $p_{\text{init}}$ 、步数 $n$  
1: 设 $t \leftarrow 0$ ，步长 $h \leftarrow \frac{1}{n}$  
2: 抽取样本 $X_0 \sim p_{\text{init}}$ ，其中 $X_0 = (X_0^{(1)}, \ldots, X_0^{(d)}) \in \mathcal{V}^d$  
3: 对 $i = 1, \ldots, n$ 执行  
4: 计算因子化跳跃速率 $\{q_j(v)\}_{j=1..d, v \in \mathcal{V}} \leftarrow Q_t^\theta(\cdot | X_t)$  
5: 对 $j = 1, \ldots, d$ （并行）执行  
6: $x \leftarrow X_t^{(j)}$ （位置 $j$ 处的当前词元）  
7: 按下式定义逐位置的 Euler 转移概率 $\tilde{p}_{j,t}(\cdot | X_t^{(j)} = x)$ ：$\tilde{p}_{j,t}(v | x) = \begin{cases} h q_j(v), & v \neq x, \\ 1 - h \sum_{v' \in \mathcal{V} \setminus \{x\}} q_j(v'), & v = x. \end{cases}$  
8: 采样 $X_{t+h}^{(j)} \sim \text{CATEGORICAL}(\{\tilde{p}_{j,t}(v | x)\}_{v \in \mathcal{V}})$  
9: 结束循环  
10: 设 $t \leftarrow t + h$  
11: 结束循环  
12: 返回 $X_1$

## 7.2 训练 CTMC 模型

下面我们讨论如何学习 CTMC 模型。其原理与流匹配相同：(1) 我们构造一条在噪声和数据之间插值的概率路径。(2) 我们推导条件速率矩阵和边缘速率矩阵。(3) 我们以无模拟（simulation-free）的方式学习边缘速率矩阵。现在我们一步一步讲解这个配方。

在本节中，数据分布 $p _ { \mathrm { d a t a } }$ 是 $S$ 上由概率质量函数刻画的分布。具体地，$p _ { \mathrm { d a t a } } : S  \mathbb { R } _ { \ge 0 } , z \mapsto p _ { \mathrm { d a t a } } ( z )$ 且 $\begin{array} { r } { \sum _ { z \in S } p _ { \mathrm { d a t a } } ( z ) = 1 } \end{array}$ 。我们不知道 $p _ { \mathrm { d a t a } }$ ，但训练时可以通过数据集访问样本 $z \sim p _ { \mathrm { d a t a } }$ 。例如，整个万维网上的所有文本。我们的目标是学习生成样本 $z \sim p _ { \mathrm { d a t a } }$ 。我们的目标是训练 CTMC 模型 $Q _ { t } ^ { \theta }$ 使得

$$
X _ {0} \sim p _ {\mathrm{init}}, X _ {t} \mathrm{CTMCof} Q _ {t} ^ {\theta } \Rightarrow X _ {1} \sim p _ {\mathrm{data}}
$$

所以你也许会意识到，这与欧氏情形 $\mathbb { R } ^ { d }$ （见第 2、3 节）并无不同，只不过我们用 CTMC 模型替代了流/扩散模型。

## 7.2.1 条件与边缘概率路径

我们定义 $\delta _ { z } ( x )$ 为满足当 $x \neq z$ 时 $\delta _ { z } ( x ) = 0$ 、当 $x = z$ 时 $\delta _ { z } ( x ) = 1$ 的函数。一个（离散）条件概率路径由分布族 $p _ { t } ( x | z )$ 给出，其中 $x , z \in S$ 且 $0 \leq t \leq 1$ ，满足

$$
p _ {0} (\cdot | z) = p _ {\mathrm{init}}, \quad p _ {1} (\cdot | z) = \delta_ {z}
$$

与欧氏情形类似，离散条件概率路径在一个不依赖于 $z$ 的分布和一个全部质量集中在 $z$ 上的分布之间插值。一个（离散）边缘概率路径则由

$$
p _ {t} (x) = \sum_ {z \in S} p _ {t} (x | z) p _ {\mathrm{data}} (z)
$$

给出。可以很容易地验证边缘概率路径在"噪声"和数据之间插值：

$$
p _ {0} = p _ {\mathrm{init}}, \quad p _ {1} = p _ {\mathrm{data}}\tag{89}
$$

## 例 35（因子化混合路径（逐词元独立加噪））

设 $S = \mathcal { V } ^ { d }$ 且设 $\begin{array} { r } { p _ { \mathrm { i n i t } } ( x ) = \prod _ { j = 1 } ^ { d } p _ { \mathrm { i n i t } } ^ { ( j ) } ( x _ { j } ) } \end{array}$ 为一个因子化的初始分布。固定一个调度器 $0 \le \kappa _ { t } \le 1$ 使得 $\kappa _ { 0 } = 0 , \kappa _ { 1 } = 1$ 且 $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } \dot { \kappa } _ { t } \geq 0 . } \end{array}$ 。定义条件路径为

$$
p _ {t} (x | z) = \prod_ {j = 1 } ^ { d } \Big [ (1 - \kappa_ {t}) p _ {\mathrm{init}} ^ {(j)} (x _ {j}) + \kappa_ {t} \delta_ {z _ {j}} (x _ {j}) \Big ].
$$

等价地，可以通过独立同分布地抽取掩码 $m _ { j } = 0 , 1$ 和噪声 $\xi _ { j } \sim p _ { \mathrm { i n i t } } ^ { ( j ) }$ 来采样 $x \sim p _ { t } ( \cdot \mid z )$ ，然后设置

$$
\begin{array}{l} m _ {j} \sim \mathrm{Bernoulli} (\kappa_ {t}), \quad \xi_ {j} \sim p _ {\mathrm{init}} ^ {(j)} \\ x _ {j} = m _ {j}   z _ {j} + (1 - m _ {j})   \xi_ {j}, \qquad j = 1, \ldots , d \\ x = (x _ {1}, \dots , x _ {d}) \end{array}
$$

我们称上述为因子化混合路径。上述过程实际上在序列的每个位置以概率 $1 - \kappa _ { t }$ 独立地"摧毁"第 j 个词元，即当 $t = 0$ 时 $1 - \kappa _ { t } = 1$ ，所有信息都被摧毁；而当 $t = 1$ 时 $1 - \kappa _ { t } = 0$ ，没有信息被摧毁。注意这与例 8 中的高斯概率路径类似，因为信息随着由调度器 $\kappa _ { t }$ 决定的速度被逐渐摧毁。然而，它与高斯概率路径也不同：因子化混合路径不移动/传输概率质量（在离散空间中不存在方向）——它只是让一个分布淡入，另一个淡出。

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/e259690b69eb0075f0de0ace974083416940d5b6f2d33e2b346c8eb5cb3dcbbd.jpg)



图 19：$d = 2$ 时离散概率路径的示意图。顶行：在初始分布与狄拉克分布之间插值的条件概率路径。底行：在初始分布与数据分布（这里是棋盘格图案）之间的插值。注意它与图 5 的相似之处和不同之处：这里概率路径是"瞬移"的（我们给初始分布降权、给终端分布升权）。


## 7.2.2 条件与边缘速率矩阵

作为下一步，我们现在将构造离散流匹配的训练目标。首先，我们构造一个条件速率矩阵——它是流匹配中条件向量场的类比物。对每个数据点 $z \in S$ ，设 $Q _ { t } ^ { z } ( y | x )$ 为一个速率矩阵。若它满足

$$
X _ {0} \sim p _ {\mathrm{init}}, X _ {t} \mathrm{CTMCof} Q _ {t} ^ {z} \Rightarrow X _ {t} \sim p _ {t} (\cdot | z)
$$

则我们称其为条件速率矩阵。换言之，条件速率矩阵是指由其驱动的 CTMC "遵循"条件概率路径的速率矩阵。条件速率矩阵作为构建边缘速率矩阵的模块，而边缘速率矩阵则遵循边缘概率路径：

## 定理 36（离散边缘化技巧）

由下式定义边缘速率矩阵

$$
Q _ {t} (y | x) = \sum_ {z \in S} Q _ {t} ^ {z} (y | x) \frac {p _ {t} (x | z) p _ {\mathrm{data}} (z)}{p _ {t} (x)} = \sum_ {z \in S} Q _ {t} ^ {z} (y | x) p _ {1 | t} (z | x) \quad \text {where} p _ {1 | t} (z | x) := \frac {p _ {t} (x | z) p _ {\mathrm{data}} (z)}{p _ {t} (x)}\tag{90}
$$

是一个合法的速率矩阵，并满足以下条件：

$$
X _ {0} \sim p _ {\mathrm{init}}, X _ {t} \mathrm{CTMCof} Q _ {t } \Rightarrow X _ {t} \sim p _ {t}
$$

特别地，由方程 (89) 得 $X _ { 1 } \sim p _ { \mathrm { d a t a } }$ ，即边缘速率矩阵的 CTMC 将噪声转化为数据。

要证明这一论断，我们需要 CTMC 的一个基本方程，即所谓的 Kolmogorov 前向方程（KFE）：

## 命题 2（Kolmogorov 前向方程）

设 $( p _ { t } ) _ { 0 \leq t \leq 1 }$ 为 $S$ 上的一族分布。进一步，设 $X _ { t }$ 为具有矩阵 $Q _ { t }$ 和初始分布 $p _ { 0 }$ 的 CTMC。则 $X _ { t } \sim p _ { t }$ 对所有 $0 \leq t \leq 1$ 成立，当且仅当 Kolmogorov 前向方程（KFE）成立：

$$
\frac {\mathrm{d}}{\mathrm{d} t} p _ {t} (x) = \sum_ {y \in S} Q _ {t} (x | y) p _ {t} (y)
$$

KFE 的证明。为证明 KFE 是必要的，假设 $p _ { t } ( x )$ 是 CTMC 的真实边缘分布，即对每个 $0 \leq t \leq 1$ 有 $X _ { t } \sim p _ { t }$ 。那么我们可以计算：

$$
\begin{array}{r l} \frac {\mathrm{d}}{\mathrm{d} t} p _ {t} (x) & \stackrel {(i)} {=} \frac {\mathrm{d}}{\mathrm{d} h _ {| h = 0}} p _ {t + h} (x) \\ & \stackrel {(i i)} {=} \frac {\mathrm{d}}{\mathrm{d} h _ {| h = 0}} \sum_ {y} p _ {t + h | t} (x | y) p _ {t} (y) \\ & \stackrel {(i i i)} {=} \sum_ {y} \frac {\mathrm{d}}{\mathrm{d} h _ {| h = 0}} p _ {t + h | t} (x | y) p _ {t} (y) \\ & \stackrel {(i v)} {=} \sum_ {y} Q _ {t} (x | y) p _ {t} (y) \end{array}
$$

其中在 (i) 中我们简单使用了一个时间偏移，在 (ii) 中我们使用了转移概率的定义，在 (iii) 中我们对求和与求导交换了顺序，在 (iv) 中我们使用了速率矩阵的定义（见方程 (87)）。

接下来，为证明 KFE 是充分的，我们可以把 KFE 改写为矩阵形式：

$$
\frac {\mathrm{d}}{\mathrm{d} t} p _ {t } = Q _ {t } p _ {t }
$$

其中在这个方程中我们把 $p _ { t } = ( p _ { t } ( x ) ) _ { x \in S }$ 看作向量，把 $Q _ { t } = ( Q _ { t } ( y | x ) ) _ { x , y \in S }$ 看作矩阵。注意上述是一个在向量空间 $\mathbb { R } ^ { S }$ 上的线性 ODE。它的初始条件由定理中所述的 $p _ { 0 }$ 固定。因此，如果另一组边缘分布 $q _ { t }$ 也满足这个方程，由 ODE 的唯一性（见定理 3），我们可以得出 $q _ { t } = p _ { t }$ 。这表明 KFE 也是充分的。□

> **译注**（KFE 的地位）：Kolmogorov 前向方程是 Fokker–Planck 方程/连续性方程在离散状态空间中的类比：把分布看作向量 $p _ { t }$ ，时间导数等于速率矩阵作用于分布向量（$\tfrac { \mathrm { d } } { \mathrm { d } t } p _ { t } = Q _ { t } p _ { t }$ ，一个线性 ODE）；它既是验证“CTMC 遵循给定概率路径”的工具，也是定理 36 中把条件速率矩阵平均成边缘速率矩阵的关键。

定理 36 的证明。利用 KFE，只需证明按定理（见方程 (90)）定义边缘速率矩阵满足 KFE：

$$
\begin{array}{r l} & {\frac {\mathrm{d}}{\mathrm{d} t} p _ {t} (x) \stackrel {(i)} {=} \frac {\mathrm{d}}{\mathrm{d} t} \sum_ {z \in S} p _ {t} (x | z) p _ {\mathrm{data}} (z)} \\ & {\stackrel {(i i)} {=} \sum_ {z \in S} \frac {\mathrm{d}}{\mathrm{d} t} p _ {t} (x | z) p _ {\mathrm{data}} (z)} \\ & {\stackrel {(i i i)} {=} \sum_ {z \in S} \left[ \sum_ {y \in S} Q _ {t} ^ {z} (x | y) p _ {t} (y | z) \right] p _ {\mathrm{data}} (z)} \\ & {\stackrel {(i v)} {=} \sum_ {y \in S} p _ {t} (y) \left[ \sum_ {z \in S} Q _ {t} ^ {z} (x | y) \frac {p _ {t} (y | z) p _ {\mathrm{data}} (z)}{p _ {t} (y)} \right]} \\ & {\stackrel {(v)} {=} \sum_ {y \in S} p _ {t} (y) Q _ {t} (x | y)} \end{array}
$$

其中 (i) 由边缘概率路径的定义得到，在 (ii) 中我们对求和与求导交换了顺序，在 (iii) 中我们对条件速率矩阵使用了 KFE，在 (iv) 中我们乘以并除以 $p _ { t } ( y )$ ，在 (v) 中我们使用了边缘速率矩阵 $Q _ { t } ( y | x )$ 的定义。这表明 KFE 被满足。结论由命题 2 得出。□

现在让我们推导因子化混合路径的一个具体条件速率矩阵的例子。

例 37（因子化混合路径的条件速率矩阵）

设 $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } \kappa _ { t } = \dot { \kappa } _ { t } } \end{array}$ 。因子化混合路径具有一个因子化的条件速率矩阵，由

$$
\begin{array}{c} Q _ {t} ^ {z} (y | x) = (Q _ {t} ^ {z} (v _ {i}, j | x _ {j})) _ {v _ {i}, j} \\ Q _ {t} ^ {z} (v _ {i}, j | x _ {j}) = \frac {\dot {\kappa} _ {t}}{1 - \kappa_ {t}} (\delta_ {z _ {j}} (v _ {i}) - \delta_ {x _ {j}} (v _ {i})) \\ = \frac {\dot {\kappa} _ {t}}{1 - \kappa_ {t}} \left\{ \begin{array}{l l} 0 & \text {if} x _ {j} = z _ {j} \\ 1 & \text {if} v _ {i} = z _ {j}, x _ {j} \neq z _ {j} \\ 0 & \text {if} v _ {i} \neq z _ {j}, x _ {j} \neq z _ {j} \\ - 1 & \text {if} v _ {i} = x _ {j}, x _ {j} \neq z _ {j} \end{array} \right. \end{array}
$$

给出。

注意这是一个非常简单的速率矩阵：它只允许跳到 $z ^ { j }$ ——即任一词元位置 $j$ 被更新时，必须跳成终端数据点 $\boldsymbol { z } = ( z _ { 1 } , \cdots , z _ { d } )$ 在该位置的词元值；且仅当当前位置尚不等于 $z ^ { j }$ 时才跳。

证明。我们注意到因子化混合路径完全因子化为独立分量，所建议的条件速率矩阵也是如此。因此，不失一般性，我们可以假设 $d = 1$ 。我们只需逐维做计算即可。于是，我们可以推导：

$$
\begin{array}{r l} & {\frac {\mathrm{d}}{\mathrm{d} t} p _ {t} (x | z) \stackrel {(i)} {=} \frac {\mathrm{d}}{\mathrm{d} t} \left[ (1 - \kappa_ {t}) p _ {\mathrm{init}} (x) + \kappa_ {t} \delta_ {z} (x) \right]} \\ & {\quad \stackrel {(i i)} {=} \dot {\kappa} _ {t } \delta_ {z} (x) - \dot {\kappa} _ {t } p _ {\mathrm{init}} (x)} \\ & {\quad \stackrel {(i i i)} {=} \frac {\dot {\kappa} _ {t}}{1 - \kappa_ {t}} (\delta_ {z} (x) - [ (1 - \kappa_ {t}) p _ {\mathrm{init}} (x) + \kappa_ {t} \delta_ {z} (x) ])} \\ & {\quad \stackrel {(i v)} {=} \frac {\dot {\kappa} _ {t}}{1 - \kappa_ {t}} (\delta_ {z} (x) - p _ {t} (x | z))} \\ & {\quad \stackrel {(v)} {=} \frac {\dot {\kappa} _ {t}}{1 - \kappa_ {t}} \delta_ {z} (x)   (1 - p _ {t} (x | z)) + \frac {\dot {\kappa} _ {t}}{1 - \kappa_ {t}} (\delta_ {z} (x) - 1) p _ {t} (x | z)} \\ & {\quad \stackrel {(v i)} {=} \sum_ {y \neq x} \frac {\dot {\kappa} _ {t}}{1 - \kappa_ {t}} \delta_ {z} (x) p _ {t} (y | z) + \frac {\dot {\kappa} _ {t}}{1 - \kappa_ {t}} (\delta_ {z} (x) - 1) p _ {t} (x | z)} \\ & {\quad \stackrel {(v i i)} {=} \sum_ {y \neq x} Q _ {t} ^ {z} (x | y) p _ {t} (y | z) + Q _ {t} ^ {z} (x | x) p _ {t} (x | z)} \\ & {\quad \stackrel {(v i i i)} {=} \sum_ {y \in S} Q _ {t} ^ {z} (x | y) p _ {t} (y | z)} \end{array}
$$

其中 (i) 使用了因子化混合路径在 $d = 1$ 时的定义，(ii) 通过对 $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } \kappa _ { t } = \dot { \kappa } _ { t } } \end{array}$ 取导数并代入得到，(iii) 由简单代数推得，(iv) 由因子化混合路径的定义得到，(v) 由简单代数推得，(vi) 由定义及 $\begin{array} { r } { \sum _ { y \in S } p _ { t } ( y | z ) = 1 } \end{array}$ 的事实推得，(vii) 由速率矩阵的定义得到，(viii) 由简单代数推得。上述表明 KFE 被满足，因此结论得证。□

## 7.2.3 学习边缘速率矩阵

在本节中，我们推导训练 CTMC 模型的基本算法。由定理 36，训练 CTMC 模型 $Q _ { t } ^ { \theta } ( y | x )$ 可以通过学习边缘速率矩阵来实现。

在本节中，我们现在只考虑因子化混合路径（见例 35），因为这是迄今为止大多数离散扩散/流匹配模型所使用的路径。在这种情况下，边缘速率矩阵具有一个非常直观的形式：

## 定理 38（因子化混合路径的边缘化技巧）

因子化混合路径的边缘速率矩阵是因子化的，并具有形式

$$
Q _ {t} (v _ {i}, j | x) = \frac {\dot {\kappa} _ {t}}{1 - \kappa_ {t}} (p _ {1 | t} (z _ {j} = v _ {i} | x) - \delta_ {x _ {j}} (v _ {i}))
$$

其中 $p _ { 1 | t } ( z _ { j } = v _ { i } | x )$ 是在给定完整加噪序列 x 的条件下第 j 个位置（序列中第 j 个词元）等于 $v _ { i }$ 的条件概率。

证明。边缘速率矩阵由

$$
Q _ {t} (y | x) = \sum_ {z \in S} Q _ {t} ^ {z} (y | x) p _ {1 | t} (z | x)\tag{91}
$$

给出。现在，每当 y 和 x 不是邻居（即在多于一个词元上不同）时，对每个 z 都有 $Q _ { t } ^ { z } ( y | x ) = 0$ 。因此在这种情况下也有 $Q _ { t } ( y | x ) = 0$ 。这表明边缘速率矩阵也是因子化的。于是有

$$
Q _ {t} (v _ {i}, j | x) = \sum_ {z \in S} Q _ {t} ^ {z} (v _ {i}, j | x) p _ {1 | t} (z | x)\tag{92}
$$

$$
\stackrel {(i)} {=} \sum_ {z \in S} \frac {\dot {\kappa} _ {t}}{1 - \kappa_ {t}} (\delta_ {z _ {j}} (v _ {i}) - \delta_ {x _ {j}} (v _ {i})) p _ {1 | t} (z | x)\tag{93}
$$

$$
\stackrel {(i i)} {=} \frac {\dot {\kappa} _ {t}}{1 - \kappa_ {t}} \left(\sum_ {z \in S} \delta_ {z _ {j}} (v _ {i}) p _ {1 | t} (z | x) - \delta_ {x _ {j}} (v _ {i})\right)\tag{94}
$$

$$
\stackrel {(i i i)} {=} \frac {\dot {\kappa} _ {t}}{1 - \kappa_ {t}} \left(p _ {1 | t} (z _ {j} = v _ {i} | x) - \delta_ {x _ {j}} (v _ {i})\right)\tag{95}
$$

其中 (i) 由条件速率矩阵的公式（见例 37）得到，(ii) 由 $\begin{array} { r } { \sum _ { z \in S } p _ { 1 | t } ( z | x ) = 1 } \end{array}$ 的事实得到，(iii) 由边缘化得到。证明完毕。□

> **译注**（离散流匹配＝逐词元分类）：定理 38 把“回归问题”换成了“分类问题”：网络逐位置预测后验 $p _ { 1 | t } ( z _ { j } = v _ { i } | x )$（softmax 头），乘以调度因子 $\dot { \kappa } _ { t } / ( 1 - \kappa _ { t } )$ 即得边缘速率矩阵，训练损失随之退化为逐词元交叉熵。这正是离散扩散模型能用标准 transformer 训练的原因。

值得注意的是：边缘速率矩阵本质上是对概率 $p _ { 1 | t } ( z _ { j } = v _ { i } | x )$ 的重新参数化。这实际上就是在每个词元位置 $j = 1 , \dots , d$ 上学习一个分类器。换句话说，我们可以简单地定义一个去噪概率网络为

$$
p _ {1 | t} ^ {\theta}: \underbrace {x} _ {\text { network   input }} \mapsto \underbrace {(p _ {1 | t} ^ {\theta} (z _ {j} = v _ {i} | x)) _ {j = 1 , \ldots , d , v _ {i} \in \mathcal {V}}} _ {\text { network   output }}
$$

注意网络输出形状为 $d \times V$ 。可以通过简单的 softmax 层得到每个词元位置的概率。网络本身可以是标准的序列到序列网络，例如 transformer 就可以（见第 6.1.2 节）。

由于这只是在每个位置 $j$ 上的分类器，我们可以通过对每个 $j = 1 , \cdots , d$ 使用交叉熵损失来训练这样的网络。这就导出了由

$$
\mathcal {L} _ {\mathrm{DFM}} (\theta) = \mathbb {E} _ {z \sim p _ {\mathrm{data}}, t \sim \mathrm{Unif} _ {[ 0, 1 ]}, x \sim p _ {t} (\cdot | z)} \left[ \sum_ {j = 1} ^ { d } - \log p _ {1 | t} ^ {\theta} (z _ {j} | x) \right]
$$

给出的离散流匹配（Discrete Flow Matching）损失。

值得注意的是：要训练一个生成模型，我们所要做的只是在每个位置 $j$ 上训练一个分类器模型。与连续流匹配简化为简单回归（见第 3 节）类似，离散流匹配和离散扩散模型简化为简单的分类训练。我们在算法 8 中总结了训练算法。训练完成后，可以通过算法 7 进行采样。

## 例 39（掩码扩散语言模型）

上述方法的一个特例是掩码扩散语言模型（MDLM）。MDLM 的思想是我们可以把词元词表 $\mathcal { V } = \{ v _ { 1 } , \cdots , v _ { V } \}$ 扩展，加入一个新词元 [mask]，表示该词元缺失（或被掩码）。具体地，我们设 $\mathcal { V } = \{ v _ { 1 } , \cdot \cdot \cdot , v _ { V } , [ \mathrm { m a s k } ] \}$ 且初始点就是 $[ \mathrm { m a s k } ] ^ { d }$ ，即全部为掩码的序列。形式上，这意味着在上述框架中设置 $p _ { \mathrm { i n i t } } = \delta _ { [ \mathrm { m a s k } ] ^ { d } }$ 。采样过程在图 20 中图示。

> **译注**（MDLM 与 BERT 的关系）：掩码扩散语言模型可视为 BERT 式 [MASK] 预训练的“生成化”：加噪即以 $1 - \kappa _ { t }$ 的概率把词元替换为 [MASK]，训练目标与 MLM 的交叉熵一致；区别在采样——不是预测一次就结束，而是沿算法 7 迭代、逐步揭示位置并解掩码，从而实现并行式文本生成。

算法 8 训练因子化 CTMC 模型（离散扩散）

输入（Require）：词元序列数据集 $z \sim p_{data}$ ，其中 $z = (z_1, \ldots, z_d) \in \mathcal{V}^d$ ；

初始（噪声）词元边缘分布 $p_{init}^{(j)}$ 在 V 上；调度 $\kappa_t \in [0, 1]$ ；

后验网络 $f_\theta$ 返回 V 上逐位置的 logits；优化器 OPT

1: 对每次训练迭代执行

2: 采样数据点 $z \sim p_{data}$ 3: 采样时间 $t \sim \text{Unif}[0, 1]$ 并计算 $\kappa \leftarrow \kappa_t$ 4: 采样加噪状态 $x \sim p_t(\cdot | z)$（因子化混合路径）：

5: 对 $j = 1, \ldots, d$ （并行）执行

6: 采样掩码 $m_j \sim \text{Bernoulli}(\kappa)$ 7: 采样噪声词元 $\xi_j \sim p_{init}^{(j)}$ 8: 设 $x_j \leftarrow m_j z_j + (1 - m_j) \xi_j$ 9: 结束循环

10: $x \leftarrow (x_1, \ldots, x_d)$ 11: 通过网络的 logits 预测终端词元后验：$\ell_j(\cdot) \leftarrow f_\theta(x, t)_j \Rightarrow p_{1|t}^\theta(v | x)_j = \text{Softmax}(\ell_j)(v)$ 12: 离散流匹配损失（z 的逐词元 NLL）：$\mathcal{L}_{\text{DFM}}(\theta) \leftarrow \sum_{j=1}^d \left[ -\log p_{1|t}^\theta(z_j | x)_j \right]$ 13: 更新参数：$\theta \leftarrow \text{OPT.STEP}\big(\nabla_\theta\mathcal{L}_{\text{DFM}}(\theta)\big)$ 14: 结束循环

t = 0

[MASK]    [MASK]    [MASK]    [MASK]    [MASK]    [MASK]    [MASK]

t = 0.25

[MASK]    [MASK]    [MASK]    on    [MASK]    [MASK]    [MASK]

t = 0.75

[MASK]    cat    [MASK]    on    the    mat    [MASK]

t = 1

The    cat    sat    on    the    mat    . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/c4ba83674d54654c7ea8d63a06928a2be6c12dbe8b8790464b4f7fe9ab745ee8.jpg)



图 20：掩码扩散语言模型轨迹的示意图。


这便完成了一个完整的 CTMC 模型训练与采样流程，它允许我们生成诸如文本这样的离散序列。当前最先进的离散扩散模型 [4] 使用了本工作描述的配方，采用在 web 规模数据上训练的神经网络（通常是 transformer）。

## 注 40（生成器匹配）

你可能会好奇，为什么流/扩散模型的原理能够如此无缝地迁移到离散状态空间。事实上，流匹配的原理并非流甚至 CTMC 所独有。相反，它们是构建基于马尔可夫过程的生成模型的一般性学习原理。这一思想导向了生成器匹配（Generator Matching）框架 [19]，该框架把离散的与连续的流模型和扩散模型扩展并统一到一起。生成器是向量场 $u _ { t }$ 与速率矩阵 $Q _ { t }$ 的统一推广；基于马尔可夫过程与生成器，可为任意数据模态和状态空间构建模型。例如，你可以为光滑流形 [8, 10] 构建模型（例如几何数据）、混合状态空间（例如文本与图像的联合生成）[6]，以及其他马尔可夫过程，如跳跃过程 [19, 7]。

## 8 References

[1] Michael S Albergo, Nicholas M Bofi, and Eric Vanden-Eijnden. “Stochastic interpolants: A unifying framework for flows and difusions”. In: arXiv preprint arXiv:2303.08797 (2023). 

[2] Brian DO Anderson. “Reverse-time difusion equation models”. In: Stochastic Processes and their Applications 12.3 (1982), pp. 313–326. 

[3] Yogesh Balaji et al. eDif-I: Text-to-Image Difusion Models with an Ensemble of Expert Denoisers. 2023. arXiv: 2211.01324 [cs.CV]. url: https://arxiv.org/abs/2211.01324. 

[4] Tiwei Bie et al. “Llada2. 0: Scaling up difusion language models to 100b”. In: arXiv preprint arXiv:2512.15745 (2025). 

[5] Andrew Campbell et al. “A continuous time framework for discrete denoising models”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 28266–28279. 

[6] Andrew Campbell et al. “Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design”. In: arXiv preprint arXiv:2402.04997 (2024). 

[7] Andrew Campbell et al. “Trans-dimensional generative modeling via jump difusion models”. In: Advances in Neural Information Processing Systems 36 (2023), pp. 42217–42257. 

[8] Ricky TQ Chen and Yaron Lipman. “Flow matching on general geometries”. In: arXiv preprint arXiv:2302.03660 (2023). 

[9] Earl A Coddington, Norman Levinson, and T Teichmann. Theory of ordinary diferential equations. 1956. 

[10] Valentin De Bortoli et al. “Riemannian score-based generative modelling”. In: Advances in neural information processing systems 35 (2022), pp. 2406–2422. 

[11] Prafulla Dhariwal and Alex Nichol. Difusion Models Beat GANs on Image Synthesis. 2021. arXiv: 2105.05233 [cs.LG]. url: https://arxiv.org/abs/2105.05233. 

[12] Alexey Dosovitskiy. “An image is worth 16x16 words: Transformers for image recognition at scale”. In: arXiv preprint arXiv:2010.11929 (2020). 

[13] Alexey Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. 2021. arXiv: 2010.11929 [cs.CV]. url: https://arxiv.org/abs/2010.11929. 

[14] Patrick Esser et al. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. 2024. arXiv: 2403.03206 [cs.CV]. url: https://arxiv.org/abs/2403.03206. 

[15] Lawrence C Evans. Partial diferential equations. Vol. 19. American Mathematical Society, 2022. 

[16] Itai Gat et al. “Discrete flow matching”. In: Advances in Neural Information Processing Systems 37 (2024), pp. 133345–133385. 

[17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. “Denoising difusion probabilistic models”. In: Advances in neural information processing systems 33 (2020), pp. 6840–6851. 

[18] Jonathan Ho and Tim Salimans. Classifier-Free Difusion Guidance. 2022. arXiv: 2207.12598 [cs.LG]. url: https://arxiv.org/abs/2207.12598. 

[19] Peter Holderrieth et al. “Generator matching: Generative modeling with arbitrary markov processes”. In: arXiv preprint arXiv:2410.20587 (2024). 

[20] Peter Holderrieth et al. “GLASS Flows: Transition Sampling for Alignment of Flow and Difusion Models”. In: arXiv preprint arXiv:2509.25170 (2025). 

[21] Arieh Iserles. A first course in the numerical analysis of diferential equations. Cambridge university press, 2009. 

[22] Alexia Jolicoeur-Martineau et al. “Adversarial score matching and improved sampling for image generation”. In: arXiv preprint arXiv:2009.05475 (2020). 

[23] Tero Karras et al. “Elucidating the design space of difusion-based generative models”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 26565–26577. 

[24] Samuel Lavoie et al. Modeling Caption Diversity in Contrastive Vision-Language Pretraining. 2024. arXiv: 2405.00740 [cs.CV]. url: https://arxiv.org/abs/2405.00740. 

[25] Yaron Lipman et al. “Flow matching for generative modeling”. In: arXiv preprint arXiv:2210.02747 (2022). 

[26] Yaron Lipman et al. “Flow Matching Guide and Code”. In: arXiv preprint arXiv:2412.06264 (2024). 

[27] Xingchao Liu, Chengyue Gong, and Qiang Liu. “Flow straight and fast: Learning to generate and transfer data with rectified flow”. In: arXiv preprint arXiv:2209.03003 (2022). 

[28] Nanye Ma et al. “Sit: Exploring flow and difusion-based generative models with scalable interpolant transformers”. In: arXiv preprint arXiv:2401.08740 (2024). 

[29] Xuerong Mao. Stochastic diferential equations and applications. Elsevier, 2007. 

[30] William Peebles and Saining Xie. Scalable Difusion Models with Transformers. 2023. arXiv: 2212 . 09748 [cs.CV]. url: https://arxiv.org/abs/2212.09748. 

[31] Ethan Perez et al. “Film: Visual reasoning with a general conditioning layer”. In: Proceedings of the AAAI conference on artificial intelligence. Vol. 32. 1. 2018. 

[32] Lawrence Perko. Diferential equations and dynamical systems. Vol. 7. Springer Science & Business Media, 2013. 

[33] Adam Polyak et al. Movie Gen: A Cast of Media Foundation Models. 2024. arXiv: 2410.13720 [cs.CV]. url: https://arxiv.org/abs/2410.13720. 

[34] Alec Radford et al. Learning Transferable Visual Models From Natural Language Supervision. 2021. arXiv: 2103.00020 [cs.CV]. url: https://arxiv.org/abs/2103.00020. 

[35] Colin Rafel et al. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. 2023. arXiv: 1910.10683 [cs.LG]. url: https://arxiv.org/abs/1910.10683. 

[36] Robin Rombach et al. High-Resolution Image Synthesis with Latent Difusion Models. 2022. arXiv: 2112.10752 [cs.CV]. url: https://arxiv.org/abs/2112.10752. 


[37] Robin Rombach et al. “High-resolution image synthesis with latent difusion models”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022, pp. 10684–10695. 




[38] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. “U-net: Convolutional networks for biomedical image segmentation”. In: Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer. 2015, pp. 234– 241. 




[39] Chitwan Saharia et al. Photorealistic Text-to-Image Difusion Models with Deep Language Understanding. 2022. arXiv: 2205.11487 [cs.CV]. url: https://arxiv.org/abs/2205.11487. 




[40] Simo Särkkä and Arno Solin. Applied stochastic diferential equations. Vol. 10. Cambridge University Press, 2019. 




[41] Jascha Sohl-Dickstein et al. “Deep unsupervised learning using nonequilibrium thermodynamics”. In: International conference on machine learning. PMLR. 2015, pp. 2256–2265. 




[42] Yang Song and Stefano Ermon. “Generative modeling by estimating gradients of the data distribution”. In: Advances in neural information processing systems 32 (2019). 




[43] Yang Song et al. Score-Based Generative Modeling through Stochastic Diferential Equations. 2021. arXiv: 2011.13456 [cs.LG]. url: https://arxiv.org/abs/2011.13456. 




[44] Yang Song et al. “Score-Based Generative Modeling through Stochastic Diferential Equations”. In: International Conference on Learning Representations (ICLR). 2021. 




[45] Yang Song et al. “Score-based generative modeling through stochastic diferential equations”. In: arXiv preprint arXiv:2011.13456 (2020). 




[46] Matthew Tancik et al. Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. 2020. arXiv: 2006.10739 [cs.CV]. url: https://arxiv.org/abs/2006.10739. 




[47] Yi Tay et al. UL2: Unifying Language Learning Paradigms. 2023. arXiv: 2205.05131 [cs.CL]. url: https: //arxiv.org/abs/2205.05131. 




[48] Arash Vahdat, Karsten Kreis, and Jan Kautz. “Score-based generative modeling in latent space”. In: Advances in neural information processing systems 34 (2021), pp. 11287–11302. 




[49] Ashish Vaswani et al. Attention Is All You Need. 2023. arXiv: 1706.03762 [cs.CL]. url: https://arxiv.org/ abs/1706.03762. 




[50] Linting Xue et al. ByT5: Towards a token-free future with pre-trained byte-to-byte models. 2022. arXiv: 2105. 13626 [cs.CL]. url: https://arxiv.org/abs/2105.13626. 




[51] Jingfeng Yao, Bin Yang, and Xinggang Wang. “Reconstruction vs. generation: Taming optimization dilemma in latent difusion models”. In: Proceedings of the Computer Vision and Pattern Recognition Conference. 2025, pp. 15703–15712. 

## A 概率论回顾

我们给出概率论中一些基本概念的简要概述。本节部分引自 [26]。

## A.1 随机向量

考虑 d 维欧几里得空间中的数据 $x = ( x ^ { 1 } , \ldots , x ^ { d } ) \in \mathbb { R } ^ { d }$，其标准欧几里得内积为 $\begin{array} { r } { \langle x , y \rangle = \sum _ { i = 1 } ^ { d } x ^ { i } y ^ { i } } \end{array}$，范数为 $\| x \| = { \sqrt { \langle x , x \rangle } }$ 。我们考虑取值于 $\mathbb { R } ^ { d }$ 的随机变量（RV），其具有连续的（概率）密度函数（PDF），该 PDF 定义为一个连续函数 $p _ { X } : \mathbb { R } ^ { d }  \mathbb { R } _ { \geq 0 }$，赋予事件 A 的概率为


$$
\mathbb {P} (X \in A) = \int_ {A} p _ {X} (x) \mathrm{d} x,\tag{96}
$$


其中 $\begin{array} { r } { \int p _ { X } ( x ) \mathrm { d } x = 1 } \end{array}$ 。按惯例，在整个空间上积分时省略积分区间 $\textstyle ( \int \equiv \int _ { \mathbb { R } ^ { d } } )$ 。为保持记号简洁，我们将随机变量 $X _ { t }$ 的 PDF $p _ { X _ { t } }$ 简记为 $p _ { t }$ 。我们用记号 $X \sim p$ 或 $X \sim p ( X )$ 表示 X 服从分布 $p .$ 生成建模中一种常见的 PDF 是 d 维各向同性高斯分布：


$$
\mathcal {N} (x; \mu , \sigma^ {2} I) = (2 \pi \sigma^ {2}) ^ {- \frac {d}{2}} \exp \left(- \frac {\| x - \mu \| _ {2} ^ {2}}{2 \sigma^ {2}}\right),\tag{97}
$$


其中 $\mu \in \mathbb { R } ^ { d }$ 和 $\sigma \in \mathbb { R } _ { > 0 }$ 分别表示该分布的均值和标准差。

随机变量的期望是最小二乘意义下与 X 最接近的常向量：


$$
\mathbb {E} \left[ X \right] = \underset {z \in \mathbb {R} ^ {d}} {\arg \min} \int \left\| x - z \right\| ^ {2} p _ {X} (x) \mathrm{d} x = \int x p _ {X} (x) \mathrm{d} x.\tag{98}
$$


一个计算随机变量函数期望的有用工具是"无意识统计学家定律"（law of the unconscious statistician）：


$$
\mathbb {E} \left[ f (X) \right] = \int f (x) p _ {X} (x) \mathrm{d} x.\tag{99}
$$


必要时，我们会用 $\mathbb { E } _ { X } f ( X )$ 标出期望下的随机变量。

## A.2 条件密度与条件期望

给定两个随机变量 X、$Y \in \mathbb { R } ^ { d }$ ，它们的联合 PDF $p _ { X , Y } ( x , y )$ 的边缘分布为


$$
\int p _ {X, Y} (x, y) \mathrm{d} y = p _ {X} (x) \text { and } \int p _ {X, Y} (x, y) \mathrm{d} x = p _ {Y} (y).\tag{100}
$$


图 21 给出了两个 RV 在 R $( d = 1 )$ 中联合 PDF 的示意。条件 PDF $p _ { X | Y }$ 描述了在条件事件 $Y = y$（其密度 $p _ { Y } ( y ) > 0$）下随机变量 X 的 PDF：


$$
p _ {X | Y} (x | y) := \frac {p _ {X , Y} (x , y)}{p _ {Y} (y)},\tag{101}
$$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/49f461867205418df15d3347329b4978eb12be47320b667398aa281b70bf0fe3.jpg)


图 21：联合 PDF $p _ { X , Y }$（以阴影表示）及其边缘分布 $p _ { X }$ 和 $p _ { Y }$（以黑线表示）。图引自 [26]


条件 PDF $p _ { Y \mid X }$ 也类似。贝叶斯公式用 $p _ { X | Y }$ 表示条件 PDF $p _ { Y \mid X }$：


$$
p _ {Y | X} (y | x) = \frac {p _ {X | Y} (x | y) p _ {Y} (y)}{p _ {X} (x)},\tag{102}
$$


当 $p _ { X } ( x ) > 0$ 时成立。

条件期望 $\mathbb { E } \left[ X | Y \right]$ 是最小二乘意义下对 X 的最佳逼近函数 $g _ { \star } ( Y )$：


$$
\begin{array}{l} g _ {\star} := \underset {g: \mathbb {R} ^ {d} \to \mathbb {R} ^ {d}} {\arg \min} \mathbb {E} \left[ \| X - g (Y) \| ^ {2} \right] = \underset {g: \mathbb {R} ^ {d} \to \mathbb {R} ^ {d}} {\arg \min} \int \| x - g (y) \| ^ {2} p _ {X, Y} (x, y) \mathrm{d} x \mathrm{d} y \\ = \underset {g: \mathbb {R} ^ {d} \to \mathbb {R} ^ {d}} {\arg \min} \int \left[ \int \| x - g (y) \| ^ {2} p _ {X | Y} (x | y) \mathrm{d} x \right] p _ {Y} (y) \mathrm{d} y. \end{array}\tag{103}
$$


对满足 $p _ { Y } ( y ) > 0$ 的 $\boldsymbol { y } \in \mathbb { R } ^ { d }$ ，条件期望函数因此为


$$
\mathbb {E} \left[ X | Y = y \right] := g _ {\star} (y) = \int x p _ {X | Y} (x | y) \mathrm{d} x,\tag{104}
$$


其中第二个等式类似于公式 (98)，由在 $Y = y$ 处取公式 (103) 内层方括号的极小化元得到。将 g<sub>⋆</sub> 与随机变量 ${ \cal Y } ,$ 复合，得到


$$
\mathbb {E} \left[ X | Y \right] := g _ {\star} (Y),\tag{105}
$$


它是一个 $\mathbb { R } ^ { d }$ 中的随机变量。令人困惑的是，$\mathbb { E } \left[ X | Y = y \right]$ 和 $\mathbb { E } \left[ X | Y \right]$ 两者都常被称为条件期望，但它们是不同的对象。具体而言，$\mathbb { E } \left[ X | Y = y \right]$ 是一个函数 $\mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ ，而 $\mathbb { E } \left[ X | Y \right]$ 是一个取值于 $\mathbb { R } ^ { d }$ 的随机变量。为了区分这两个概念，我们在讨论中将使用本文引入的记号。

塔性质（tower property）是一条有用的性质，可简化涉及两个随机变量 X 与 Y 的条件期望的推导：


$$
\mathbb {E} \left[ \mathbb {E} \left[ X | Y \right] \right] = \mathbb {E} \left[ X \right]\tag{106}
$$


由于 $\mathbb { E } \left[ X | Y \right]$ 本身是一个 RV，是 RV Y 的函数，因此外层期望计算的是 $\mathbb { E } \left[ X | Y \right]$ 的期望。可以用上述部分定义验证塔性质：


$$
\begin{array}{c} \mathbb {E} \left[ \mathbb {E} \left[ X | Y \right] \right] = \int \left(\int x p _ {X | Y} (x | y) \mathrm{d} x\right) p _ {Y} (y) \mathrm{d} y \\ \stackrel {{(1 0 1)}} {{=}} \int \int x p _ {X, Y} (x, y) \mathrm{d} x \mathrm{d} y \\ \stackrel {{(1 0 0)}} {{=}} \int x p _ {X} (x) \mathrm{d} x = \mathbb {E} \left[ X \right]. \end{array}
$$


最后，考虑一个涉及两个 RV $f ( X , Y )$ 和 $Y _ { z }$ 的有用性质，其中 X 和 Y 是任意两个 RV。利用"无意识统计学家定律"与 (104)，我们得到恒等式


$$
\mathbb {E} \left[ f (X, Y) | Y = y \right] = \int f (x, y) p _ {X | Y} (x | y) \mathrm{d} x.\tag{107}
$$


## B Fokker-Planck 方程的证明

在本节中，我们给出 Fokker-Planck 方程的一个自包含证明，其中连续性方程是其特例（定理 11）。我们强调，理解本文其余内容并不必需这一节，且它在数学上更为进阶。如果你希望了解 Fokker-Planck 方程的来龙去脉，那么这一节就是为你准备的。

## 定理 41（Fokker-Planck 方程）

设 $p _ { t }$ 为一条概率路径且 $p _ { 0 } = p _ { \mathrm { i n i t } }$ ，考虑如下 SDE


$$
X _ {0} \sim p _ {\mathrm{init}}, \quad \mathrm{d} X _ {t} = u _ {t} (X _ {t}) \mathrm{d} t + \sigma_ {t} \mathrm{d} W _ {t}.
$$


当且仅当 Fokker-Planck 方程成立时，$X _ { t }$ 对所有的 $0 \leq t \leq 1$ 具有分布 $p _ { t }$ ：


$$
\partial_ {t} p _ {t} (x) = - \mathrm{div} (p _ {t} u _ {t}) (x) + \frac {\sigma_ {t} ^ {2}}{2} \Delta p _ {t} (x) \quad \mathrm{forall} x \in \mathbb {R} ^ {d}, 0 \leq t \leq 1,\tag{108}
$$


我们先证明 Fokker-Planck 方程是必要条件，即若 $X _ { t } \sim p _ { t }$ ，则 Fokker-Planck 方程成立。证明的技巧是使用测试函数 $f$ ，即无穷可微（"光滑"）且仅在有限域内非零（紧支）的函数 $f : \mathbb { R } ^ { d }  \mathbb { R }$ 。我们利用如下事实：对任意可积函数 $g _ { 1 } , g _ { 2 } : \mathbb { R } ^ { d } $ R 有


$$
g _ {1} (x) = g _ {2} (x) \text {for all} x \in \mathbb {R} ^ {d} \quad \Leftrightarrow \quad \int f (x) g _ {1} (x) \mathrm{d} x = \int f (x) g _ {2} (x) \mathrm{d} x \text {for all test functions} f\tag{109}
$$


换言之，我们可以将逐点相等表述为取积分后的相等。测试函数的有用之处在于它们是光滑的，即我们可以取梯度和高阶导数。特别地，对任意测试函数 $f _ { 1 } , f _ { 2 }$ ，我们可以使用分部积分：


$$
\int f _ {1} (x) \frac {\partial}{\partial x _ {i}} f _ {2} (x) \mathrm{d} x = - \int f _ {2} (x) \frac {\partial}{\partial x _ {i}} f _ {1} (x) \mathrm{d} x\tag{110}
$$


条件是 $f _ { 1 } , f _ { 2 }$ 及其乘积 $f _ { 1 } \cdot f _ { 2 }$ 可积。利用这一点与散度和拉普拉斯算子的定义（见公式 (22)），我们得到恒等式：


$$
\int \nabla f _ {1} ^ {T} (x) f _ {2} (x) \mathrm{d} x = - \int f _ {1} (x) \operatorname{div} \left(f _ {2}\right) (x) \mathrm{d} x \quad \left(f _ {1}: \mathbb {R} ^ {d} \rightarrow \mathbb {R}, f _ {2}: \mathbb {R} ^ {d} \rightarrow \mathbb {R} ^ {d}\right)\tag{111}
$$


$$
\int f _ {1} (x) \Delta f _ {2} (x) \mathrm{d} x = \int f _ {2} (x) \Delta f _ {1} (x) \mathrm{d} x \quad (f _ {1}: \mathbb {R} ^ {d} \to \mathbb {R}, f _ {2}: \mathbb {R} ^ {d} \to \mathbb {R})\tag{112}
$$


现在进入证明。我们使用 SDE 轨迹的随机更新，如公式 (6) 所示：


$$
X _ {t + h} = X _ {t} + h u _ {t} (X _ {t}) + \sigma_ {t} (W _ {t + h} - W _ {t}) + h R _ {t} (h)\tag{113}
$$


$$
\approx X _ {t} + h u _ {t} (X _ {t}) + \sigma_ {t} (W _ {t + h} - W _ {t})\tag{114}
$$


为简洁起见，我们暂时忽略误差项 $R _ { t } ( h )$，因为我们反正会令 $h  0$ 。然后我们可以做如下计算：


$$
\begin{array}{r l} & f (X _ {t + h}) - f (X _ {t}) \stackrel {(1 1 4)} {=} f (X _ {t} + h u _ {t} (X _ {t}) + \sigma_ {t} (W _ {t + h} - W _ {t})) - f (X _ {t}) \\ & \stackrel {(i)} {=} \nabla f (X _ {t}) ^ {T} \left(h u _ {t} (X _ {t}) + \sigma_ {t} (W _ {t + h} - W _ {t}))\right) \\ & \qquad + \frac {1}{2} \left(h u _ {t} (X _ {t}) + \sigma_ {t} (W _ {t + h} - W _ {t}))\right) ^ {T} \nabla^ {2} f (X _ {t}) \left(h u _ {t} (X _ {t}) + \sigma_ {t} (W _ {t + h} - W _ {t}))\right) \\ & \stackrel {(i i)} {=} h \nabla f (X _ {t}) ^ {T} u _ {t} (X _ {t}) + \sigma_ {t} \nabla f (X _ {t}) ^ {T} (W _ {t + h} - W _ {t}) \\ & \qquad + \frac {1}{2} h ^ {2} u _ {t} (X _ {t}) ^ {T} \nabla^ {2} f (X _ {t}) u _ {t} (X _ {t}) + h \sigma_ {t} u _ {t} (X _ {t}) ^ {T} \nabla^ {2} f (X _ {t}) (W _ {t + h} - W _ {t}) + \\ & \qquad + \frac {1}{2} \sigma_ {t} ^ {2} (W _ {t + h} - W _ {t}) ^ {T} \nabla^ {2} f (X _ {t}) (W _ {t + h} - W _ {t}) \end{array}
$$


其中 (i) 使用了对 $f$ 在 $X _ { t }$ 处的二阶泰勒近似，(ii) 使用了 Hessian $\nabla ^ { 2 } f$ 是对称矩阵这一事实。注意 $\mathbb { E } [ W _ { t + h } - W _ { t } | X _ { t } ] = 0$ 且 $W _ { t + h } - W _ { t } | X _ { t } \sim { \mathcal { N } } ( 0 , h I _ { d } )$ 。因此


$$
\begin{array}{r l} & {\mathbb {E} [ f (X _ {t + h}) - f (X _ {t}) | X _ {t} ]} \\ & {= h \nabla f (X _ {t}) ^ {T} u _ {t} (X _ {t}) + \frac {1}{2} h ^ {2} u _ {t} (X _ {t}) ^ {T} \nabla^ {2} f (X _ {t}) u _ {t} (X _ {t}) + \frac {h}{2} \sigma_ {t} ^ {2} \mathbb {E} _ {\epsilon_ {t} \sim \mathcal {N} (0, I _ {d})} [ \epsilon_ {t} ^ {T} \nabla^ {2} f (X _ {t}) \epsilon_ {t} ]} \\ & {\stackrel {(i)} {=} h \nabla f (X _ {t}) ^ {T} u _ {t} (X _ {t}) + \frac {1}{2} h ^ {2} u _ {t} (X _ {t}) ^ {T} \nabla^ {2} f (X _ {t}) u _ {t} (X _ {t}) + \frac {h}{2} \sigma_ {t} ^ {2} {\mathrm{trace}} (\nabla^ {2} f (X _ {t}))} \\ & {\stackrel {(i i)} {=} h \nabla f (X _ {t}) ^ {T} u _ {t} (X _ {t}) + \frac {1}{2} h ^ {2} u _ {t} (X _ {t}) ^ {T} \nabla^ {2} f (X _ {t}) u _ {t} (X _ {t}) + \frac {h}{2} \sigma_ {t} ^ {\mathrm{2}} \Delta f (X _ {t})} \end{array}
$$


其中 (i) 使用了 $\mathbb { E } _ { \epsilon _ { t } \sim \mathcal { N } ( 0 , I _ { d } ) } [ \epsilon _ { t } ^ { T } A \epsilon _ { t } ] = \operatorname { t r a c e } ( A )$ ，(ii) 使用了拉普拉斯算子和 Hessian 矩阵的定义。由此我们得到


$$
\begin{array}{l} \partial_ {t} \mathbb {E} [ f (X _ {t}) ] \\ = \lim _ {h \to 0} \frac {1}{h} \mathbb {E} [ f (X _ {t + h}) - f (X _ {t}) ] \\ = \lim _ {h \to 0} \frac {1}{h} \mathbb {E} [ \mathbb {E} [ f (X _ {t + h}) - f (X _ {t}) | X _ {t} ] ] \\ = \mathbb {E} [ \lim _ {h \to 0} \frac {1}{h} \left(h \nabla f (X _ {t}) ^ {T} u _ {t} (X _ {t}) + \frac {1}{2} h ^ {2} u _ {t} (X _ {t}) ^ {T} \nabla^ {2} f (X _ {t}) u _ {t} (X _ {t}) + \frac {h}{2} \sigma_ {t} ^ {2} \Delta f (X _ {t})\right) ] \\ = \mathbb {E} [ \nabla f (X _ {t}) ^ {T} u _ {t} (X _ {t}) + \frac {1}{2} \sigma_ {t} ^ {2} \Delta f (X _ {t}) ] \\ \stackrel {(i)} {=} \int \nabla f (x) ^ {T} u _ {t} (x) p _ {t} (x) \mathrm{d} x + \int \frac {1}{2} \sigma_ {t} ^ {2} \Delta f (x) p _ {t} (x) \mathrm{d} x \\ \stackrel {(i i)} {=} - \int f (x) \mathrm{div} (u _ {t} p _ {t}) (x) \mathrm{d} x + \int \frac {1}{2} \sigma_ {t} ^ {2} f (x) \Delta p _ {t} (x) \mathrm{d} x \\ = \int f (x) \left(- \mathrm{div} (u _ {t} p _ {t}) (x) + \frac {1}{2} \sigma_ {t} ^ {2} \Delta p _ {t} (x)\right) \mathrm{d} x \end{array}
$$


其中 (i) 使用了假设 $p _ { t }$ 为 $X _ { t }$ 的分布，(ii) 使用了公式 (111) 和公式 (112) 。注意，要使用这一点，我们需要乘积 $p _ { t } ( x ) u _ { t } ( x )$ 可积，即


$$
\int p _ {t} (x) \| u _ {t} (x) \| \mathrm{d} x <   \infty
$$


注意，在机器学习中（由于数据有界以及数值精度限制导致函数有界）该条件几乎总是成立。因此有


$$
\partial_ {t} \mathbb {E} [ f (X _ {t}) ] = \int f (x) \left(- \operatorname{div} \left(p _ {t} u _ {t}\right) (x) + \frac {\sigma_ {t} ^ {2}}{2} \Delta p _ {t} (x)\right) \mathrm{d} x \quad (\text { for   all } f \text { and } 0 \leq t \leq 1)\tag{115}
$$


$$
\stackrel {(i)} {\Leftrightarrow} \partial_ {t} \int f (x) p _ {t} (x) \mathrm{d} x = \int f (x) \left(- \operatorname{div} (p _ {t} u _ {t}) (x) + \frac {\sigma_ {t} ^ {2}}{2} \Delta p _ {t} (x)\right) \mathrm{d} x \quad (\text {for all} f \text {and} 0 \leq t \leq 1)\tag{116}
$$


$$
\stackrel {(i i)} {\Leftrightarrow} \quad \int f (x) \partial_ {t} p _ {t} (x) \mathrm{d} x = \int f (x) \left(- \operatorname{div} (p _ {t} u _ {t}) (x) + \frac {\sigma_ {t} ^ {2}}{2} \Delta p _ {t} (x)\right) \mathrm{d} x \quad (\text { for   all } f \text { and } 0 \leq t \leq 1)\tag{117}
$$


$$
\stackrel {(i i i)} {\Leftrightarrow} \partial_ {t} p _ {t} (x) = - \operatorname{div} (p _ {t} u _ {t}) (x) + \frac {\sigma_ {t} ^ {2}}{2} \Delta p _ {t} (x) \quad (\text {for all} x \in \mathbb {R} ^ {d}, 0 \leq t \leq 1)\tag{118}
$$


其中 (i) 使用了假设 $X _ { t } \sim p _ { t }$ ，(ii) 交换了导数与积分，(iii) 使用了公式 (109) 。这完成了 Fokker-Planck 方程是必要条件的证明。

最后，我们解释为什么它也是充分条件。Fokker-Planck 方程是一个偏微分方程（PDE）。更具体地说，它是一个所谓的抛物型偏微分方程。与定理 $^ { 3 , }$ 类似，此类微分方程在给定初始条件时具有唯一解（例如见 [15, 第 7 章]）。现在，若公式 (108) 对 $p _ { t }$ 成立，我们上面已证明它对 $X _ { t } \ ( \mathrm { i . e . } \ X _ { t } \sim q _ { t } )$ 的真实分布 $q _ { t }$ 也必须成立——换言之，$p _ { t }$ 和 $q _ { t }$ 都是该抛物型 PDE 的解。此外，我们知道初始条件相同，即由插值概率路径的构造，$p _ { 0 } = q _ { 0 } = p _ { \mathrm { i n i t } }$ 。因此，由微分方程解的唯一性，我们知道 $p _ { t } = q _ { t }$ 对所有的 $0 \leq t \leq 1$ 成立——即 $X _ { t } \sim q _ { t } = p _ { t }$ ，这正是我们要证明的。

> **译注**（本证明的"弱形式"策略）：证明的总体思路是与任意光滑紧支撑测试函数 $f$ 作积分内积，把导数通过分部积分"转移"到 $f$ 上（公式 (110)(111)(112)），这样等式只需在"积分意义"下成立，从而只要求 $p _ { t } u _ { t }$ 可积等温和条件；再用公式 (109) 把"积分相等"翻译回"逐点相等"。充分性则依赖抛物型 PDE 的标准唯一性结果：给定初始条件解唯一，而 $p _ { t }$ 与 $X _ { t }$ 的真实分布 $q _ { t }$ 满足同一方程与同一初值，故必重合。

## C 连续时间马尔可夫链的存在性与唯一性

我们在本节中证明定理 33。

证明。唯一性：我们需要证明满足公式 (87) 的转移核 $p _ { t ^ { \prime } | t } ( X _ { t ^ { \prime } } = y | X _ { t } = x )$ 至多只有一个。作为第一步，我们注意到公式 (87) 蕴含


$$
\frac {\mathrm{d}}{\mathrm{d} t ^ {\prime}} p _ {t ^ {\prime} | t} (X _ {t ^ {\prime}} = y | X _ {t} = x)\tag{119}
$$


$$
= \frac {\mathrm{d}}{\mathrm{d} h} p _ {t ^ {\prime} + h | t} (X _ {t ^ {\prime} + h} = y | X _ {t} = x) _ {| h = 0}\tag{120}
$$


$$
= \frac {\mathrm{d}}{\mathrm{d} h} \left[ \sum_ {z \in S} p _ {t ^ {\prime} + h | t ^ {\prime}} (X _ {t ^ {\prime} + h} = y | X _ {t ^ {\prime}} = z) p _ {t ^ {\prime} | t} (X _ {t ^ {\prime}} = z | X _ {t} = x) \right] _ {| h = 0}\tag{121}
$$


$$
= \sum_ {z \in S} Q _ {t ^ {\prime}} (y | z) p _ {t ^ {\prime} | t} (X _ {t ^ {\prime}} = z | X _ {t} = x)\tag{122}
$$


对固定的 $x , t$ ，可将 $t ^ { \prime } \mapsto p _ { t ^ { \prime } | t } ( X _ { t ^ { \prime } } = y | X _ { t } = x )$ 视为一个向量值函数，上式即为该函数的一个线性 ODE（事实上是 Kolmogorov 前向方程，参见命题 2），且具有已知的初始条件，即 $p _ { t | t } ( X _ { t } = y | X _ { t } = x ) = \delta _ { y } ( x )$ 。我们知道，每个线性 ODE 都有唯一解（参见定理 3），因此 $p _ { t ^ { \prime } | t } ( X _ { t ^ { \prime } } = y | X _ { t } = x )$ 也必须是唯一的。

存在性：反之，每个线性 ODE 都有解，即我们知道对每个 $x , t$ 都存在 $p _ { t ^ { \prime } | t } ( X _ { t ^ { \prime } } =$ y $| X _ { t } = x )$ 使得


$$
p _ {t | t} (X _ {t} = y | X _ {t} = x) = \delta_ {y} (x)\tag{123}
$$


$$
\frac {\mathrm{d}}{\mathrm{d} t ^ {\prime}} p _ {t ^ { \prime } | t} (X _ {t ^ {\prime}} = y | X _ {t} = x) = \sum_ {z \in S} Q _ {t ^ { \prime}} (y | z) p _ {t ^ { \prime } | t} (X _ {t ^ { \prime}} = z | X _ {t} = x)\tag{124}
$$


当 $t ^ { \prime } = t$ 时，这特别蕴含公式 (87) 。还需证明此时 $p _ { t ^ { \prime } | t } ( X _ { t ^ { \prime } } = y | X _ { t } = x )$ 是合法的转移核，即以下 3 个性质必须成立：


$$
\sum_ {y \in S} p _ {t ^ { \prime } | t} (X _ {t ^ { \prime}} = y | X _ {t} = x) = 1\tag{125}
$$


$$
p _ {t ^ { \prime } | t} (X _ {t ^ { \prime}} = y | X _ {t} = x) \geq 0\tag{126}
$$


$$
\sum_ {z \in S} p _ {t _ {2} | t _ {1}} (X _ {t _ {2}} = y | X _ {t _ {1}} = z) p _ {t _ {1} | t _ {0}} (X _ {t _ {1}} = z | X _ {t _ {0}} = x) = p _ {t _ {2} | t _ {0}} (y | x)\tag{127}
$$


对于第一个性质，可以观察到由公式 (123) 它在 $t ^ { \prime } = t$ 时成立，且


$$
\frac {\mathrm{d}}{\mathrm{d} t ^ { \prime}} \sum_ {y \in S} p _ {t ^ { \prime } | t} (X _ {t ^ { \prime}} = y | X _ {t} = x)\tag{128}
$$


$$
= \sum_ {y \in S} \frac {\mathrm{d}}{\mathrm{d} t ^ { \prime}} p _ {t ^ { \prime } | t} (X _ {t ^ { \prime}} = y | X _ {t} = x)\tag{129}
$$


$$
= \sum_ {z \in S} \left[ \sum_ {y \in S} Q _ {t ^ { \prime}} (y | z) \right] p _ {t ^ { \prime } | t} (X _ {t ^ { \prime}} = z | X _ {t} = x)\tag{130}
$$


(131)


其中我们使用了速率矩阵各列之和为 0 这一事实。对于第二个性质，注意在时刻 $t ^ { \prime } = t$ 时它成立。此外，只要 $p _ { t ^ { \prime } | t } ( X _ { t ^ { \prime } } = y | X _ { t } = x ) = 0$ ，就必有


$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t ^ { \prime}} p _ {t ^ { \prime } | t} (X _ {t ^ { \prime}} = y | X _ {t} = x) = \sum_ {z \neq y} \underbrace {Q _ {t ^ { \prime}} (y | z)} _ {\geq 0} p _ {t ^ { \prime } | t} (X _ {t ^ { \prime}} = z | X _ {t} = x) \\ \qquad \qquad \qquad \qquad \qquad \qquad \geq 0 \end{array}
$$


因此，只要 $p _ { t ^ { \prime } | t } ( X _ { t ^ { \prime } } = y | X _ { t } = x ) = 0$ ，它只会增加。因此，$p _ { t ^ { \prime } | t } ( X _ { t ^ { \prime } } = y | X _ { t } = x )$ 永远不会为负。

要证明第三个性质，定义 $q _ { t _ { 2 } | t _ { 0 } } ( y | x )$ 为


$$
q _ {t _ {2} | t _ {0}} (y | x) = \sum_ {z \in S} p _ {t _ {2} | t _ {1}} (X _ {t _ {2}} = y | X _ {t _ {1}} = z) p _ {t _ {1} | t _ {0}} (X _ {t _ {1}} = z | X _ {t _ {0}} = x)
$$


那么我们知道


$$
q _ {t _ {2} = t _ {1} | t _ {0}} (y | x) = \sum_ {z \in S} \delta_ {y} (z) p _ {t _ {1} | t _ {0}} (X _ {t _ {1}} = z | X _ {t _ {0}} = x) = p _ {t _ {1} | t _ {0}} (X _ {t _ {1}} = y | X _ {t _ {0}} = x)
$$


并且


$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t _ {2}} q _ {t _ {2} | t _ {0}} (y | x) = \sum_ {z \in S} \frac {\mathrm{d}}{\mathrm{d} t _ {2}} p _ {t _ {2} | t _ {1}} (X _ {t _ {2}} = y | X _ {t _ {1}} = z) p _ {t _ {1} | t _ {0}} (X _ {t _ {1}} = z | X _ {t _ {0}} = x) \\ \qquad = \sum_ {z \in S} \sum_ {\tilde {z} \in S} Q _ {t _ {2}} (y | \tilde {z}) p _ {t _ {2} | t _ {1}} (X _ {t _ {2}} = \tilde {z} | X _ {t _ {1}} = z) p _ {t _ {1} | t _ {0}} (X _ {t _ {1}} = z | X _ {t _ {0}} = x) \\ \qquad = \sum_ {\tilde {z} \in S} Q _ {t _ {2}} (y | \tilde {z}) \left[ \sum_ {z \in S} p _ {t _ {2} | t _ {1}} (X _ {t _ {2}} = \tilde {z} | X _ {t _ {1}} = z) p _ {t _ {1} | t _ {0}} (X _ {t _ {1}} = z | X _ {t _ {0}} = x) \right] \\ \qquad = \sum_ {\tilde {z} \in S} Q _ {t _ {2}} (y | \tilde {z}) q _ {t _ {2} | t _ {0}} (\tilde {z} | x) \end{array}
$$


这表明 $p _ { t _ { 2 } | t _ { 0 } } ( z | x )$ 和 $q _ { t _ { 2 } | t _ { 0 } } ( z | x )$ 满足同一个 ODE。因此必有


$$
\sum_ {z \in S} p _ {t _ {2} | t _ {1}} (X _ {t _ {2}} = y | X _ {t _ {1}} = z) p _ {t _ {1} | t _ {0}} (X _ {t _ {1}} = z | X _ {t _ {0}} = x) = q _ {t _ {2} | t _ {0}} (y | x) = p _ {t _ {2} | t _ {0}} (y | x)
$$


这就证明了第三个性质。因此 $p _ { t ^ { \prime } | t } ( y | x )$ 确实是满足公式 (87) 的转移核。证明完毕。 □ 

> **译注**（生成元（速率矩阵）与前向方程）：生成元即速率矩阵 $Q _ { t } ( y | z )$ ，刻画单位时间内从状态 $z$ 跳到状态 $y$ 的速率，各列之和为 0（概率守恒）。Kolmogorov 前向方程是关于转移概率向量的线性 ODE，由线性 ODE 基本理论即得存在唯一——正文的唯一性/存在性证明正是沿此展开；随后验证的三个性质（归一性、非负性、Chapman–Kolmogorov 方程）正是一个合法的马尔可夫转移核所需满足的条件。

## D 关于 VAE 的补充视角

本节展开讨论正文中对 VAE 的处理，并对公式 (83) 的总 VAE 损失给出一个变分推导。首先注意到，编码器和解码器都给出 x 与隐变量 z 的联合分布，即


$$
\begin{array}{l} q _ {\phi} (x, z) = p _ {\mathrm{data}} (x) q _ {\phi} (\cdot | x) \\ p _ {\theta} (x, z) = p _ {\theta} (x | z) p _ {\mathrm{prior}} (z) \end{array}
$$


因此，我们可以把 VAE 的训练理解为学习 $\phi$ 和 $\theta$ ，使得编码器与解码器的联合分布合理地相似。我们可以通过隐变量与数据的联合分布的 KL 散度来实现这一点：


$$
\begin{array}{r l} & D _ {\mathrm{KL}} (q _ {\phi} (x, z) \parallel p _ {\theta} (x, z)) = D _ {\mathrm{KL}} (p _ {\mathrm{data}} (x) q _ {\phi} (z \mid x) \parallel p _ {\theta} (x \mid z) p _ {\mathrm{prior}} (z)) \\ & \qquad = \mathbb {E} _ {\blacksquare} \left[ \log \left(\frac {p _ {\mathrm{data}} (x) q _ {\phi} (z \mid x)}{p _ {\theta} (x \mid z) p _ {\mathrm{prior}} (z)}\right) \right] \\ & \qquad = \mathbb {E} _ {\blacksquare} \left[ \log p _ {\mathrm{data}} (x) \right] + \mathbb {E} _ {\blacksquare} \left[ \log \left(\frac {q _ {\phi} (z \mid x)}{p _ {\mathrm{prior}} (z)}\right) \right] - \mathbb {E} _ {\blacksquare} \left[ \log p _ {\theta} (x \mid z) \right] \\ & \qquad = x \sim p _ {\mathrm{data}} (x) z \sim q _ {\phi} (z | x). \end{array}\tag{132}
$$


现在依次考察剩余的三个项。首先，我们发现


$$
\mathbb {E} _ {\blacksquare} \left[ \log p _ {\mathrm{data}} (x) \right] = \mathbb {E} _ {x \sim p _ {\mathrm{data}} (x)} \left[ \log p _ {\mathrm{data}} (x) \right] = C,\tag{133}
$$


其中 C 是某个与 $\phi$ 和 θ 无关的常数。其次，我们发现


$$
\mathbb {E} _ {\blacksquare} \left[ \log \left(\frac {q _ {\phi} (z \mid x)}{p _ {\text { prior }} (z)}\right) \right] = \mathbb {E} _ {x \sim p _ {\text { data }} (x)} \left[ D _ {\text { KL }} (q _ {\phi} (z \mid x) \parallel p _ {\text { prior }} (z)) \right]\tag{134}
$$


促使 $q _ { \phi } ( z \mid x )$ 逼近先验 $p _ { \mathrm { p r i o r } } ( z )$ 。最后，我们发现


$$
- \mathbb {E} _ {x \sim p _ {\mathrm{data}} (x) z \sim q _ {\phi} (z | x)} \left[ \log p _ {\theta} (x \mid z) \right]\tag{135}
$$


对应平均负对数似然，从而用于最小化重构损失。忽略常数项，我们把先验惩罚项和重构项合并，得到 VAE 损失实际上就是联合数据空间与隐空间中的 KL 散度：


$$
\begin{array}{c} \mathcal {L} _ {\mathrm{VAE}} (\phi , \theta) = \underbrace {\mathbb {E} _ {x \sim p _ {\mathrm{data}} (x)} \left[ D _ {\mathrm{KL}} (q _ {\phi} (z \mid x) \parallel p _ {\mathrm{prior}} (z)) \right]} _ {\text {prior enforcement loss}} - \underbrace {\mathbb {E} _ {x \sim p _ {\mathrm{data}} (x) z \sim q _ {\phi} (z | x)} \left[ \log p _ {\theta} (x \mid z) \right]} _ {\text {reconstruction loss}} \\ = D _ {\mathrm{KL}} (q _ {\phi} (x, z) \parallel p _ {\theta} (x, z)) + \text {const} \end{array}\tag{136}
$$


(137)


因此，我们可以把 VAE 解释为隐空间与图像联合空间中的 KL 散度。

VAE 作为生成模型。现在我们解释如何把 VAE 解释为生成模型。我们可以通过设定 $z \sim p _ { \mathrm { p r i o r } } = \mathcal { N } ( 0 , I _ { k } )$ 并从解码器采样 $x \sim p _ { \theta } ( \cdot | z )$ 来生成样本。由此得到的


分布为：


$$
p _ {\theta} (x) = \int_ {z} p _ {\theta} (x | z) p _ {\mathrm{prior}} (z) \mathrm{d} z
$$


现在我们想证明 VAE 学会了近似地从 $p _ { \theta }$ 中采样。为证明这一点，我们需要如下结果：

## 命题 3（链式法则）

设 $q ( x , z ) , p ( x , z )$ 为两个变量 $x \in \mathbb { R } ^ { l _ { 1 } } , z \in \mathbb { R } ^ { l _ { 2 } }$ 上的分布。则成立：


$$
D _ {\mathrm{KL}} (q (z, x) \parallel p (z, x)) = D _ {\mathrm{KL}} (q (x) \parallel p (x)) + \mathbb {E} _ {x \sim q} \left[ D _ {\mathrm{KL}} (q (z | x) \parallel p (z | x)) \right].
$$


特别地，由于第二项由公式 (76) 可知非负，我们得到数据处理不等式


$$
D _ {\mathrm{KL}} (q (x) \parallel p (x)) \leq D _ {\mathrm{KL}} (q (z, x) \parallel p (z, x)).\tag{138}
$$


证明。


$$
\begin{array}{r l} & D _ {\mathrm{KL}} (q (z, x) \parallel p (z, x)) = \mathbb {E} _ {q} \left[ \log \frac {q (z , x)}{p (z , x)} \right] \\ & \qquad = \mathbb {E} _ {(x, z) \sim q} \left[ \log \frac {q (z | x)}{p (z | x)} \frac {q (x)}{p (x)} \right] \\ & \qquad = \mathbb {E} _ {(x, z) \sim q} \left[ \log \frac {q (z | x)}{p (z | x)} \right] + \mathbb {E} _ {x \sim q} \left[ \log \frac {q (x)}{p (x)} \right] \\ & \qquad = D _ {\mathrm{KL}} (q (x) \parallel p (x)) + \mathbb {E} _ {x \sim q} \left[ D _ {\mathrm{KL}} (q (z | x) \parallel p (z | x)) \right] \end{array}
$$


其中反复使用了 KL 散度的定义。

由命题 3，我们现在可以证明


$$
\mathcal {L} _ {\mathrm{VAE}} (\phi , \theta) = D _ {\mathrm{KL}} (q _ {\phi} (x, z) \parallel p _ {\theta} (x, z)) + \mathrm{const} \geq D _ {\mathrm{KL}} (p _ {\mathrm{data}} (x) \parallel p _ {\theta} (x)) + \mathrm{const}\tag{139}
$$


其中我们使用了 $q _ { \phi } ( x , z )$ 的 x-边缘分布为 $p _ { \mathrm { d a t a } }$ 这一事实。换言之，VAE 损失最小化的是数据分布 $p _ { \mathrm { d a t a } }$ 与 VAE 生成分布之间 KL 散度的一个上界。因此，我们可以把 VAE 本身就视为一种生成模型。同样地，我们可以证明


$$
\mathcal {L} _ {\mathrm{VAE}} (\phi , \theta) = D _ {\mathrm{KL}} (q _ {\phi} (x, z) \parallel p _ {\theta} (x, z)) + \mathrm{const} \geq D _ {\mathrm{KL}} (q _ {\phi} (z) \parallel p _ {\mathrm{prior}} (z)) + \mathrm{const}\tag{140}
$$


换言之，VAE 目标最小化的是隐分布与先验之间 KL 散度的一个上界。

为何不停留在 VAE？根据上述讨论，VAE 本身就可以实现为生成模型，编码器仅仅是为了辅助训练一个互补的解码器，后者将高斯分布转化为所需的数据分布。样本可以通过采样 $z \sim p _ { \mathrm { p r i o r } }$ 然后 $x \sim p _ { \theta } ( x | z )$ 得到。那么我们为何仍坚持在学习的隐空间中训练一个独立的生成模型？答案与公式 (139) 和公式 (140) 左右两侧的所谓摊销间隙（amortization gap）有关，它恰好对应于信息处理不等式中的间隙。当且仅当 $q _ { \phi } ( z | x ) = p _ { \theta } ( z | x )$ 时该间隙为零，此时编码器表示真实后验。因此，例如 $D _ { \mathrm { K L } } ( q _ { \phi } ( x , z ) \parallel p _ { \theta } ( x , z ) )$ 最小化蕴含 $D _ { \mathrm { K L } } ( q _ { \phi } ( z ) \parallel p _ { \mathrm { p r i o r } } ( z ) )$ 最小化（参见公式 (140)），但前者下降并不意味着后者同步下降。因此，在训练结束时，$D _ { \mathrm { K L } } ( q _ { \phi } ( x , z ) \parallel p _ { \theta } ( x , z ) )$ 与摊销间隙


$$
D _ {\mathrm{KL}} (q _ {\phi} (x, z) \parallel p _ {\theta} (x, z)) - D _ {\mathrm{KL}} (q _ {\phi} (z) \parallel p _ {\mathrm{prior}} (z))\tag{141}
$$


都未被完全最小化，故 $q _ { \phi } ( z ) \neq p _ { \mathrm { p r i o r } } ( z )$ 。最后注意，训练过程中，解码器学会从 $q _ { \phi } ( z )$ 而非 $p _ { \mathrm { p r i o r } } ( z )$ 重构，因此在推理时切换到从 $p _ { \mathrm { p r i o r } } ( z )$ 重构会偏离训练时的分布。然而在实践中，这种失配是特性而非缺陷。实践表明，流模型和扩散模型总体上比实现 VAE 解码器所使用的卷积堆栈能力更强，因此将部分生成复杂性分给隐生成模型是合理的。我们稍后会在讨论中回到这一思路。此外，超出本讲义范围的还有：扩散模型与流模型的变分表述将这些模型族本身视为 VAE。

> **译注**（摊销间隙）：命题 3 把联合 KL 拆成"边缘 KL ＋条件 KL 的期望"，因此 VAE 损失只是数据分布与生成分布之间差距的上界；两界之差正是这里所说的摊销间隙（amortization gap），仅当编码器表示真实后验 $q _ { \phi } ( z | x ) = p _ { \theta } ( z | x )$ 时才为零。这正是为什么训练结束时通常 $q _ { \phi } ( z ) \neq p _ { \mathrm { p r i o r } } ( z )$ ：联合 KL 下降并不带动边缘 KL 同步下降，而解码器训练时只见过 $q _ { \phi } ( z )$ ，推理时改从先验采样便超出其训练分布。

证据下界。重新整理公式 (132) 中的各项，可以得到各种互补的视角。其中之一即所谓的证据下界（ELBO），我们按如下方式提取它。注意到对固定的 x


$$
\begin{array}{r l} & {\mathbb {E} _ {z \sim q _ {\phi} (z | x)} \left[ \log \left(\frac {q _ {\phi} (z \mid x)}{p _ {\theta} (x \mid z) p _ {\mathrm{prior}} (z)}\right) \right] = \mathbb {E} _ {z \sim q _ {\phi} (z | x)} \left[ \log \left(\frac {q _ {\phi} (z \mid x)}{p _ {\theta} (z \mid x)}\right) \right] - \log p _ {\theta} (x)} \\ & {\qquad = D _ {\mathrm{KL}} (q _ {\phi} (z \mid x) \parallel p _ {\theta} (z \mid x)) - \log p _ {\theta} (x)} \end{array}\tag{142}
$$


其中第一个等式由


$$
p _ {\theta} (z \mid x) = \frac {p _ {\theta} (x \mid z) p _ {\mathrm{prior}} (z)}{p _ {\theta} (x)}.
$$


得到。于是可重排公式 (142) 得到


$$
\mathbb {E} _ {z \sim q _ {\phi} (z | x)} \left[ \log \left(\frac {p _ {\theta} (x \mid z) p _ {\mathrm{prior}} (z)}{q _ {\phi} (z \mid x)}\right) \right] + D _ {\mathrm{KL}} (q _ {\phi} (z \mid x) \parallel p _ {\theta} (z \mid x)) = \log p _ {\theta} (x),\tag{143}
$$


由此可得


$$
\underbrace {\mathbb {E} _ {z \sim q _ {\phi} (z | x)} \left[ \log \left(\frac {p _ {\theta} (x \mid z) p _ {\mathrm{prior}} (z)}{q _ {\phi} (z \mid x)}\right) \right]} _ {\triangleq \operatorname{ELBO} (x; \phi , \theta)} \leq \underbrace {\log p _ {\theta} (x)} _ {\text {evidence}}.\tag{144}
$$


因此左侧通常被称为证据下界，即 ELBO 。现在我们可以用 ELBO 改写公式 (136) 中的 $\mathcal { L } _ { \mathrm { V A E } }$ ：


$$
\begin{array}{l} \mathcal {L} _ {\mathrm{VAE}} = D _ {\mathrm{KL}} (q _ {\phi} (x, z) \parallel p _ {\theta} (x, z)) + \text {const} \\ \quad = \mathbb {E} _ {x \sim p _ {\mathrm{data}}} \mathbb {E} _ {z \sim q _ {\phi} (z | x)} \left[ \log \left(\frac {p _ {\mathrm{data}} (x) q _ {\phi} (z \mid x)}{p _ {\theta} (x \mid z) p _ {\mathrm{prior}} (z)}\right) \right] + \text {const} \\ \quad = \mathbb {E} _ {x \sim p _ {\mathrm{data}}} \left[ \log p _ {\mathrm{data}} (x) - \operatorname{ELBO} (x; \phi , \theta) \right] + \text {const} \\ \quad = - \mathbb {E} _ {x \sim p _ {\mathrm{data}}} \left[ \operatorname{ELBO} (x; \phi , \theta) \right] \underbrace {- H (p _ {\mathrm{data}}) + \text {const}} _ {\text {const}} \\ \quad = - \mathbb {E} _ {x \sim p _ {\mathrm{data}}} \left[ \operatorname{ELBO} (x; \phi , \theta) \right] + \text {const} \end{array}\tag{145}
$$


于是原始的 VAE 目标可以看作就是试图最大化期望 ELBO 。最后，我们来考虑当 VAE 被完美训练时的情形。

## 注 42（当 $q _ { \phi } ( x , z ) \approx p _ { \theta } ( x , z )$ 时会发生什么？）


首先注意，用于训练隐生成模型的采样分布由边缘分布给出


$$
q _ {\phi} (z) = \int_ {x} q _ {\phi} (z | x) p _ {\mathrm{data}} (x) \mathrm{d} x.
$$


若 $q _ { \phi } ( x , z ) = p _ { \theta } ( x , z )$ ，则特别有


$$
q _ {\phi} (z) = p _ {\theta} (z) = p _ {\mathrm{prior}} (z).
$$


因此，$q _ { \phi } ( x , z ) \approx p _ { \theta } ( x , z )$ 蕴含对隐采样分布的正则化。其次，$q _ { \phi } ( x , z ) \approx p _ { \theta } ( x , z )$ 意味着变分近似 $p _ { \theta } ( x \mid z ) \approx q _ { \phi } ( x \mid z )$ 良好，进而意味着重构误差低。

## 注 43（VAE 的"变分"之处何在？）

为何不能直接取 $q _ { \phi } ( \cdot \mid x ) = p _ { \theta } ( \cdot \mid x )$ ，从而保证 $q _ { \phi } ( x , z ) = p _ { \theta } ( x , z ) = 0$？原因是，虽然我们已知似然 $p _ { \theta } ( x \mid z )$ ，但后验


$$
p _ {\theta} (z \mid x) = \frac {p _ {\theta} (x \mid z) p _ {\mathrm{prior}} (z)}{p _ {\theta} (x)}
$$


通常是不可处理（intractable）的，因为我们无法显式计算似然 $p _ { \theta } ( x )$ 。因此 VAE 中"变分"一词的存在，是因为 $q _ { \phi } ( \cdot \mid x )$ 充当不可处理后验 $p _ { \theta } ( \cdot \mid x )$ 的替代物，或变分近似。

重构与生成。给定编码器 $q _ { \phi } ( z | x )$ 、解码器 $p _ { \theta } ( x | z )$ 以及训练以从 $q _ { \phi } ( z )$ 采样的隐生成模型 $r _ { \psi }$ ，我们可以考虑以下两个生成模型


$$
\begin{array}{r l r} & {r _ {\psi , \theta} ^ {\mathrm{recon}} (x _ {\mathrm{out}}) = \int_ {z, x _ {\mathrm{in}}} p _ {\theta} (x _ {\mathrm{out}} \mid z) q _ {\phi} (z \mid x _ {\mathrm{data}}) p _ {\mathrm{data}} (x _ {\mathrm{data}}) \mathrm{d} z \mathrm{d} x _ {\mathrm{in}}} & {\qquad \mathrm{(reconstruction~sampler)}} \\ & {r _ {\psi , \phi} ^ {\mathrm{gen}} (x _ {\mathrm{out}}) = \int_ {z _ {\mathrm{gen}}} p _ {\theta} (x _ {\mathrm{out}} | z _ {\mathrm{gen}}) r _ {\psi} (z _ {\mathrm{gen}}) \mathrm{d} z _ {\mathrm{gen}}} & {\qquad \mathrm{(generative~sampler)}} \end{array}
$$


换言之，重构采样器从 $x _ { \mathrm { d a t a } } \in p _ { \mathrm { d a t a } }$ 出发，编码到 $z$ ，再解码到 $x _ { \mathrm { o u t } }$ ；而生成采样器则从生成模型的 $z _ { \mathrm { g e n } } \in r _ { \psi }$ 出发，再经过解码器。通过计算这两个采样器分布到 $p _ { \mathrm { d a t a } }$ 的 Fréchet Inception 距离（FID），我们得到重构-FID（rFID）和生成-FID（gFID）。也可以考虑用平均失真（重构的均方根误差）来度量重构采样器的质量，尽管这样的度量对生成采样器没有意义。事实证明，重构采样器的质量与生成采样器的质量之间存在天然的张力。低 rFID（高质量的重构采样器）通常意味着隐空间中信息损失小，即隐分布 $q _ { \phi } ( z )$ 在很大程度上类似于 $p _ { \mathrm { d a t a } }$ ，从而学习隐生成模型的任务可能更加困难，提高 $\mathrm { g F I D }$ 。反之，高 rFID 通常意味着信息损失大，隐分布 $q _ { \phi } ( z )$ 更易学习，从而降低 gFID。这一现象在图 22 中可视化。

分工。重构-生成采样器之间的权衡迫使我们考虑信息损失应如何在自编码器与隐生成模型 $r _ { \psi }$ 之间分配。直觉上，$r _ { \psi }$ 通过某个学习到的向量场 $u _ { t } ^ { \psi } ( z _ { t } )$ 将标准高斯分布输运到 $q _ { \phi } ( z ) \approx p _ { \mathrm { p r i o r } }$ ，之后解码器 $p _ { \theta } ( x | z )$ 将 $q _ { \phi } ( z )$ 输运到 $p _ { \mathrm { d a t a } }$ 。现在我们（不严格地）将率定义为隐分布 $q _ { \phi } ( z )$ 与 $p _ { \mathrm { p r i o r } } ( z )$ 的匹配程度，并由此延伸，生成任务分摊给隐生成模型的程度。<sup>5</sup>这种分工可以通过绘制率与失真之间的帕累托前沿来可视化，如图 22 所示。具体而言，当率较高时失真较低，反之亦然，这为前文关于重构与生成采样器质量的讨论提供了第二种视角。我们将讨论归结为以下洞见。

## 直觉 44（分工）

图 22 的关键洞见是：在帕累托前沿的"拐点"处存在最优的分工，在该点我们得到低率（高压缩率！）而失真不高的状态。换言之，这样的点对应一种压缩水平，它同时降低了底层生成模型训练的难度，并保持了合理的重构质量。

> **译注**（重读图 22 的"率–失真"曲线）：这里"率"度量 $q _ { \phi } ( z )$ 接近先验 $p _ { \mathrm { p r i o r } } ( z )$ 的程度，即生成任务被"外包"给隐生成模型的比例；"失真"对应重构质量。二者构成帕累托前沿：率越高失真越低，反之亦然；最优分工位于前沿的"拐点"——在不显著牺牲重构质量的前提下获得高压缩率。

## E 扩散模型文献导读

文献中围绕扩散模型和流匹配存在一整个模型家族。当你阅读这些论文时，你很可能会发现与我们这门课不同的（但等价的）呈现方式。这使得阅读这些论文有时略显困惑。为此，我们希望简要概述各种框架及其差异，并将其置于历史背景中。理解本文其余内容并不需要这些，但其意图是为你阅读文献提供辅助。

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-23/970b8c7f-2444-4605-a6d9-e04583c6620c/69261afca77843b450ef91724b309476563387330c9e36540d2862f053644eb0.jpg)


图 22：右：$\mathrm { g F I D }$ 与 rFID 之间的权衡，图引自 [51]。这里 f 表示下采样因子，d 表示隐通道维度。右：失真（重构质量）对率，引自 [17, 37] 。我们注意到，这条特定曲线是用 DDPM（本身是一种 VAE）生成的。尽管失真与率计算中的某些技术细节可能不同于本教材中不严格的定义，但整体直觉保持不变。

> **译注**：本题注第一处"右："按英文原文 "Right:" 直译，但从上下文看应作 "Left:"（原文疑为笔误），指的是 gFID–rFID 权衡的左图。

离散时间与连续时间。最初的去噪扩散模型论文 [41, 42, 17] 没有使用 SDE，而是在离散时间构造马尔可夫链，即时间步为 $t = 0 , 1 , 2 , 3 , \ldots$ 。至今，你会在文献中发现大量工作采用这种离散时间形式。虽然这种构造因简单而有吸引力，但时间离散方法的缺点在于它迫使你在训练前就选定一种时间离散化。此外，损失函数需要通过证据下界（ELBO）来近似——顾名思义，它只是我们真正想要最小化的损失的一个下界。后来 Song 等人 [45] 表明，这些构造本质上是对时间连续 SDE 的近似。此外，在连续时间情形下 ELBO 损失变得紧致（即不再是下界）（例如注意定理 12 与定理 22 是等式而非下界——在离散时间情形则不然）。这使得 SDE 构造流行起来，因为它被认为在数学上"更干净"，并且可以通过 ODE/SDE 采样器在训练后控制模拟误差。但重要的是要注意，两种模型使用相同的损失，本质上并无不同。

"前向过程"与概率路径。第一代去噪扩散模型 [41, 42, 17, 45] 没有使用概率路径这个术语，而是通过所谓的前向过程（forward process）构造数据点 $z \in \mathbb { R } ^ { d }$ 的加噪程序。这是一种形如


$$
\bar {X} _ {0} = z, \quad \mathrm{d} \bar {X} _ {t} = u _ {t} ^ {\mathrm{forw}} (\bar {X} _ {t}) \mathrm{d} t + \sigma_ {t} ^ {\mathrm{forw}} \mathrm{d} \bar {W} _ {t}\tag{146}
$$


的 SDE。其思想是，在抽取一个数据点 $z \sim p _ { \mathrm { d a t a } }$ 后，模拟前向过程，从而损坏或"加噪"数据。前向过程被设计成使得当 $t \to \infty$ 时其分布收敛到高斯分布 $\mathcal { N } ( 0 , I _ { d } )$ 。换言之，当 $T \gg 0$ 时，近似有 $\bar { X } _ { T } \sim \mathcal { N } ( 0 , I _ { d } )$ 。注意，这本质上对应于一条概率路径：${ \bar { X } } _ { t }$ 在给定 $\bar { X } _ { 0 } = z$ 下的条件分布是条件概率路径 $\bar { p } _ { t } ( \cdot | z )$ ，而 ${ \bar { X } } _ { t }$ 在 $z \sim p _ { \mathrm { d a t a } }$ 下的边缘分布对应于边缘概率路径 $\bar { p } _ { t } . { } ^ { 6 }$ 然而注意，在这种构造中，为避免模拟 SDE，我们需要以闭式知道 $X _ { t } | X _ { 0 } = z$ 的分布来训练模型。这本质上将向量场 $u _ { t } ^ { \mathrm { f o r w } }$ 限制为那种我们闭式知道 $\bar { X } _ { t } | \bar { X } _ { 0 } = z$ 分布的向量场。因此，在扩散模型文献中，前向过程的向量场总取仿射形式，即 $u _ { t } ^ { \mathrm { f o r w } } ( x ) = a _ { t } x$，其中 $a _ { t }$ 为某个连续函数。对于这个选择，我们可以使用条件分布的已知公式 [40, 44, 23]：


$$
\bar {X} _ {t} | \bar {X} _ {0} = z \sim \mathcal {N} (\alpha_ {t} z, \beta_ {t} ^ {2} I), \quad \alpha_ {t} = \exp \left(\int_ {0} ^ {t} a _ {r} \mathrm{d} r\right), \quad \beta_ {t} ^ {2} = \alpha_ {t} ^ {2} \int_ {0} ^ {t} \frac {(\sigma_ {r} ^ {\mathrm{forw}}) ^ {2}}{\alpha_ {r} ^ {2}} d r
$$


注意这些只是高斯概率路径。因此可以说前向过程是构造（高斯）概率路径的一种特定方式。概率路径这一术语由流匹配 [25] 引入，既简化了构造，同时使其更具一般性：第一，扩散模型的"前向过程"实际上从未被模拟（训练时仅从 $\bar { p } _ { t } ( \cdot | z )$ 抽取样本）。第二，前向过程仅在 $t \to \infty$ 时收敛（即我们永远无法在有限时间内到达 $p _ { \mathrm { i n i t } }$）。因此，本文选择使用概率路径。

时间反转与求解 Fokker-Planck 方程。扩散模型的原始表述并未通过 Fokker-Planck 方程（或连续性方程）构造训练目标 $u _ { t } ^ { \mathrm { t a r g e t } }$ 或  log p<sub>t</sub>，而是通过前向过程的时间反转 [2] 。时间反转 $( X _ { t } ) _ { 0 \leq t \leq T }$ 是一个轨迹分布相同但在时间上反转的 SDE，即


$$
\mathbb {P} [ \bar {X} _ {t _ {1}} \in A _ {1}, \dots , \bar {X} _ {t _ {n}} \in A _ {n} ] = \mathbb {P} [ X _ {T - t _ {1}} \in A _ {1}, \dots , X _ {T - t _ {n}} \in A _ {n} ]\tag{147}
$$


$$
\text {   for   all   } 0 \leq t _ {1}, \ldots , t _ {n} \leq T, \text {   and   } A _ {1}, \ldots , A _ {n} \subset S\tag{148}
$$


如 Anderson [2] 所示，通过如下 SDE 可以得到满足上述条件的时间反转：


$$
\mathrm{d} X _ {t} = \left[ - u _ {t} (X _ {t}) + \sigma_ {t} ^ {2} \nabla \log p _ {t} (X _ {t}) \right] \mathrm{d} t + \sigma_ {t} \mathrm{d} W _ {t}, \quad u _ {t} (x) = u _ {T - t} ^ {\mathrm{forw}} (x), \sigma_ {t} = \bar {\sigma} _ {T - t}
$$


由于 $u _ { t } ( X _ { t } ) = a _ { t } X _ { t }$ ，上述对应于命题 1 中我们推导的训练目标的某个特定实例（由于使用了不同的时间约定，这并不显而易见。推导参见例如 [26]）。然而，就生成建模而言，我们通常只使用马尔可夫过程的终点 $X _ { 1 }$（例如作为生成的图像），而丢弃较早的时间点。因此，马尔可夫过程是"真正的"时间反转还是沿着一条概率路径，对许多应用并不重要。因此，使用时间反转并非必需，且常常导致次优结果，例如概率流 ODE 通常更好 [23, 28]。扩散模型中不同于时间反转的所有采样方式，都要再次依赖使用 Fokker-Planck 方程。我们希望通过这能说明，为何如今许多人直接通过


Fokker-Planck 方程构造训练目标——由 [25, 27, 1] 开创，并为本课程所采用。

流匹配 [25] 与随机插值 [1] 。我们呈现的框架与流匹配和随机插值（SI）的框架最为接近。正如我们所见，流匹配限制于流。事实上，流匹配的一项关键创新在于表明，无需通过前向过程和 SDE 构造，仅靠流模型就可以以可扩展的方式训练。由于这一限制，你应当记住，从流匹配模型采样将是确定性的（只有初始 $X _ { 0 } \sim p _ { \mathrm { i n i t } }$ 是随机的）。随机插值同时包含了纯流以及我们这里使用的通过"朗之万动力学"的 SDE 推广（参见定理 17）。随机插值的名称来源于一个插值函数 $I ( t , x , z )$，它旨在对两个分布进行插值。用我们这里的术语来说，这对应于构造条件和边缘概率路径的一种不同但（主要）等价的方式。流匹配和随机插值相对扩散模型的优势在于其简洁性和一般性：它们的训练框架非常简单，但同时允许你从任意分布 $p _ { \mathrm { i n i t } }$ 到任意分布 $p _ { \mathrm { d a t a } }$ ——而去噪扩散模型只对高斯初始分布和高斯概率路径起作用。这为生成建模开辟了新的可能性，我们将在本课程后文简要涉及。

## 摘要 45（扩散模型的其他表述）

文献中流行的扩散模型的其他表述，通常涉及以下要素的某种组合：

1. 离散时间：通过离散时间马尔可夫链对 SDE 的近似经常被使用。

2. 反转的时间约定：常见做法是采用反转的时间约定，其中 $t = 0$ 对应 p<sub>data</sub>（而非此处 t = 0 对应 $p _ { \mathrm { i n i t } } )$

3. 前向过程：前向过程（或加噪过程）是构造（高斯）概率路径的方式。

4. 通过时间反转的训练目标：训练目标也可以通过 SDE 的时间反转来构造。这是本文所呈现构造的一个特定实例（带有反转的时间约定）。 


