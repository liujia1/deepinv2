# 第7章 扩散模型：SDE视角 — 提纲

> **章节定位**：承接第6章的多尺度得分匹配——NCSN通过离散噪声调度和退火Langevin动力学实现了初步的图像生成，但它本质上是离散的"跳跃式"框架。本章将NCSN/DDPM的离散加噪-去噪过程推广为连续时间的随机微分方程（SDE），建立扩散模型的SDE统一视角。Song et al. (2021) 的Score-SDE框架揭示：NCSN（VE-SDE）和DDPM（VP-SDE）是同一连续框架的两个离散特例，逆向SDE提供了从噪声生成图像的一般采样方程，概率流ODE则揭示了随机采样的确定性等价。本章完成采样路径的关键跃迁——从"多步Langevin"到"连续扩散过程"，为后续变分路径（第8-12章）的汇合奠定基础。

> **逆问题动机**：第5-6章建立了"去噪器→得分函数→PnP采样"的逆问题求解框架，但PnP-ULA受限于单一噪声水平，难以处理复杂多尺度分布。扩散SDE提供了自然的解决方案：多时间步噪声调度对应不同尺度的先验知识，逆向SDE采样等价于从后验分布中采样——**扩散采样 = 连续化的后验采样**。这一认识将在第13章的条件扩散中完成闭环。

> **叙事主线**：为什么需要连续时间(7.1) → 正向SDE：噪声如何加入(7.2) → 逆向SDE：如何从噪声恢复(7.3) → 概率流ODE：确定性等价(7.4) → 数值离散化：如何计算(7.5) → 实践(7.6)

> **与前章衔接**：第6章末尾指出NCSN的离散噪声调度是扩散模型的前身——7.1节将这一离散框架推广为连续SDE。6.5节的退火Langevin动力学是7.3节逆向VE-SDE的离散版前身。6.6节的三种参数化（ε预测/s预测/x₀预测）在7.5节的离散化中再次出现。第5章的Langevin SDE是7.2节正向SDE的特例（固定噪声、无漂移）。

> **与后章衔接**：7.2节的正向SDE（加噪过程）是第10章层级VAE中高斯编码器的连续极限。7.3节的逆向SDE提供了扩散采样的SDE推导，而第11章将从变分下界（VLB）角度给出等价推导。7.4节的PF-ODE与第14章的Flow Sharing和连续正规化流有深层联系。本章建立的SDE框架将在第12章与变分路径汇合（DSM≡VLB），在第13章扩展为条件扩散采样。

---

## 7.0 本章导读：从离散迭代到连续扩散——SDE统一框架

第6章建立了得分匹配的完整体系——DSM训练去噪器等价于学习得分函数，NCSN的多噪声水平训练架起了通往扩散模型的桥梁。然而，NCSN和DDPM仍然是离散框架：噪声水平是有限的几个，采样步骤是"跳跃式"切换噪声水平。一个自然的问题浮现：**能否将这些离散过程统一到一个连续时间的数学框架中？**

本章从这个问题出发，走一条"推广→建立→等价→计算→实践"的路径：

**从Langevin到扩散**（7.1）：回顾单噪声水平的Langevin SDE，分析NCSN退火Langevin的"跳跃式"局限，提出连续时间推广的动机。引入"离散→连续→重离散化"的认知螺旋——先将离散过程推向连续极限获得SDE框架，再从SDE出发选择最优离散化方案。

**正向SDE**（7.2）：建立一般正向SDE框架 $dx = f(x,t)\,dt + g(t)\,dw$，展示两种重要特例：VE-SDE（方差爆炸，SMLD/NCSN的连续极限）和VP-SDE（方差保留，DDPM的连续极限）。两者对应不同的漂移系数 $f$ 和扩散系数 $g$，但共享同一逆向SDE的数学结构。

**逆向SDE**（7.3）：Anderson (1982) 的逆向时间SDE定理提供了从噪声生成数据的一般方程——$dx = [f(x,t) - g(t)^2\nabla\log p_t(x)]\,dt + g(t)\,d\bar{w}$。得分函数 $\nabla\log p_t(x)$ 再次成为核心——正向过程"抹平"概率，逆向过程用得分函数"恢复"概率。VE-SDE和VP-SDE的逆向方程分别对应SMLD采样和DDPM采样。

**概率流ODE**（7.4）：Song et al. (2020) 证明，每个逆向SDE都有一个确定性等价——概率流ODE：$dx/dt = f(x,t) - \frac{1}{2}g(t)^2\nabla\log p_t(x)$。ODE与SDE在每个时刻有相同的边际分布，但轨迹完全不同——SDE是"布朗运动"，ODE是"流线"。DDIM是PF-ODE的离散化，正如DDPM是逆向SDE的离散化。

**数值离散化**（7.5）：连续方程需要数值求解。Euler-Maruyama是SDE的基本离散化方法，其在VP-SDE逆向方程上的应用恰好恢复DDPM迭代——这是Score-SDE框架的核心统一结论。PF-ODE的Euler离散化恢复DDIM。高阶求解器（RK4、DPM++）提供更少步数的加速采样。三种采样器（DDPM/DDIM/PF-ODE）的对比揭示了随机性与确定性的权衡。

**实践**（7.6）：从理论到工程——噪声调度设计、时间条件得分网络的训练、完整采样流程、实验对比。

```
Langevin局限(7.1) → 正向SDE(7.2) → 逆向SDE(7.3) → 概率流ODE(7.4) → 离散化(7.5) → 实践(7.6)
```

本章结束时，读者将理解：扩散模型可以用统一的SDE框架描述——正向SDE加噪、逆向SDE去噪、PF-ODE提供确定性等价。DDPM和NCSN不再是独立的方法，而是同一SDE框架的不同离散化方案。这一统一视角不仅深化了理论理解，更为实践中的采样器选择和噪声调度设计提供了系统化的指导。本章的核心洞见是：**扩散模型 = 正向SDE（定义加噪） + 学习得分函数（驱动逆向） + 数值离散化（实现采样）**——三位一体，完成了采样路径从"多步Langevin"到"连续扩散过程"的跃迁。

---

## 7.1 从Langevin到扩散：连续时间推广

**核心观点**：第5章的Langevin SDE以单一固定得分函数为驱动力，第6章的NCSN通过多噪声水平的退火Langevin实现了初步的图像生成，但本质上是"跳跃式"离散框架。本节回顾Langevin SDE的结构，分析其作为生成模型的局限（固定噪声水平、单尺度得分），展示NCSN/退火Langevin向连续时间推广的路径——将离散噪声调度参数化为连续时间函数，将条件得分网络推广为时间条件得分网络。"离散→连续→重离散化"的认知螺旋是本章的方法论核心。

- **回顾：Langevin SDE与ULA**
  - ULA递推式：$X_{m+1} = X_m + \delta\,\nabla\log p(X_m|y) + \sqrt{2\delta}\,Z_{m+1}$（第4章4.3节）
  - 连续极限：$dX_t = \nabla\log p(X_t)\,dt + \sqrt{2}\,dW_t$（第5章5.1节）
  - Langevin SDE的特点：单一得分函数 $\nabla\log p(x)$、固定噪声系数 $\sqrt{2}$
  - 作为采样器的成功：收敛到目标分布 $p(x)$（Fokker-Planck保证）
  - 作为生成模型的局限：需要知道 $\nabla\log p(x)$，且仅有一个"尺度"

- **NCSN退火Langevin：离散多尺度尝试**
  - 第6章6.5节回顾：NCSN使用多个噪声水平 $\sigma_1 > \sigma_2 > \cdots > \sigma_L$
  - 退火Langevin：在每个噪声水平运行Langevin，然后"跳"到下一个噪声水平
  - 成功之处：多尺度得分覆盖了从粗糙到精细的分布特征
  - 局限之处：
    1. **"跳跃式"切换**：噪声水平的切换是离散的，不是连续变化
    2. **步数分配不均**：每个噪声水平的步数需要手动设定
    3. **框架不统一**：NCSN（SMLD）和DDPM看起来是不同的方法
  - 关键问题：能否将离散噪声调度推广为连续时间函数？

- **连续时间推广的核心思想**
  - 将离散噪声水平 $\{\sigma_i\}$ 参数化为连续时间函数 $\sigma(t)$，$t \in [0, 1]$
  - 将条件得分 $s_\theta(x, \sigma_i)$ 推广为时间条件得分 $s_\theta(x, t)$
  - 退火Langevin的多步迭代 → 连续时间的SDE
  - 直觉：从"阶梯式"降温到"斜坡式"降温——更光滑、更高效

- **"离散→连续→重离散化"的认知螺旋**
  - 第一步（离散→连续）：将NCSN/DDPM的离散迭代推向连续极限，获得SDE
  - 第二步（连续框架）：在SDE框架中，逆向SDE和PF-ODE的推导是"免费"的
  - 第三步（连续→重离散化）：从SDE出发，选择最优的离散化方案
  - 为什么这个螺旋有价值？
    - SDE框架统一了NCSN和DDPM（它们是同一SDE的不同离散化）
    - SDE框架自动提供逆向方程（不需要单独推导）
    - 可以选择更优的数值求解器（不一定用Euler-Maruyama）
  - 这一螺旋是本章的方法论核心

**来源**：Song et al. (2021) Score-SDE; 2406.08929v2 Sec 2.4; 第5章5.1节; 第6章6.5节

> **过渡**：连续时间推广的动机已经明确——将离散噪声调度参数化为连续时间函数。那么，连续时间的加噪过程具体是什么样的数学方程？正向SDE给出了答案。

---

## 7.2 正向SDE：从数据到噪声的连续过程

**核心观点**：正向SDE描述了从数据到噪声的连续加噪过程：$dx = f(x,t)\,dt + g(t)\,dw$，其中 $f(x,t)$ 是漂移系数（确定性趋势），$g(t)$ 是扩散系数（噪声强度）。两种最重要的特例是VE-SDE（$f=0$, $g(t)=\sqrt{d[\sigma(t)^2]/dt}$，SMLD/NCSN的连续极限）和VP-SDE（$f(x,t)=-\beta(t)x/2$, $g(t)=\sqrt{\beta(t)}$，DDPM的连续极限）。两者的名称反映了不同的方差行为：VE-SDE下 $\text{Var}[x_t] \to \infty$，VP-SDE下 $\text{Var}[x_t]$ 保持有界。

- **一般正向SDE框架**
  - 定义：$dx = f(x,t)\,dt + g(t)\,dw$
  - 漂移系数 $f(x,t)$：确定性力，控制信号的趋势
  - 扩散系数 $g(t)$：随机噪声的强度
  - 初始条件：$x(0) = x_0 \sim p_{\text{data}}(x)$
  - 时间范围：$t \in [0, 1]$，$t=0$ 对应干净数据，$t=1$ 对应纯噪声
  - 边际分布：$p_t(x)$ 表示 $x(t)$ 在时刻 $t$ 的概率密度

- **VE-SDE：方差爆炸（SMLD/NCSN的连续极限）**
  - 离散前身：NCSN正向过程 $x_i = x_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2}\,z_{i-1}$
  - 连续极限：令 $\Delta t = 1/N$，$\sigma_i \to \sigma(t)$
  - 推导：$x(t+\Delta t) = x(t) + \sqrt{\sigma(t+\Delta t)^2 - \sigma(t)^2}\,z(t)$
    - 当 $\Delta t \to 0$：$\sqrt{\sigma(t+\Delta t)^2 - \sigma(t)^2} \approx \sqrt{\frac{d[\sigma(t)^2]}{dt}\,\Delta t}$
  - **VE-SDE**：
    $$\boxed{dx = \sqrt{\frac{d[\sigma(t)^2]}{dt}}\,dw}$$
  - 漂移系数 $f = 0$（纯噪声添加，无确定性趋势）
  - 扩散系数 $g(t) = \sqrt{d[\sigma(t)^2]/dt}$
  - 为什么叫"方差爆炸"：$\text{Var}[x_t] = \sigma(t)^2 \to \infty$（当 $\sigma(t) \to \infty$）
  - 转移核：$p_{0t}(x_t|x_0) = \mathcal{N}(x_t | x_0, \sigma(t)^2 I)$——加性高斯噪声

- **VP-SDE：方差保留（DDPM的连续极限）**
  - 离散前身：DDPM正向过程 $x_i = \sqrt{1-\beta_i}\,x_{i-1} + \sqrt{\beta_i}\,z_{i-1}$
  - 连续极限：令 $\Delta t = 1/N$，$\beta_i \to \beta(t)\Delta t$
  - 推导：
    - $x(t+\Delta t) = \sqrt{1-\beta(t)\Delta t}\,x(t) + \sqrt{\beta(t)\Delta t}\,z(t)$
    - $\approx (1-\frac{\beta(t)}{2}\Delta t)\,x(t) + \sqrt{\beta(t)\Delta t}\,z(t)$
    - 当 $\Delta t \to 0$
  - **VP-SDE**：
    $$\boxed{dx = -\frac{\beta(t)}{2}\,x\,dt + \sqrt{\beta(t)}\,dw}$$
  - 漂移系数 $f(x,t) = -\frac{\beta(t)}{2}x$（向原点收缩的确定性力）
  - 扩散系数 $g(t) = \sqrt{\beta(t)}$
  - 为什么叫"方差保留"：$\text{Var}[x_t] = 1 - e^{-\int_0^t\beta(s)ds} \leq 1$（假设 $\text{Var}[x_0]=1$）
  - 转移核：$p_{0t}(x_t|x_0) = \mathcal{N}(x_t | \sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)$，其中 $\bar\alpha_t = e^{-\int_0^t\beta(s)ds}$
  - VP-SDE的收缩漂移补偿了噪声添加，使方差保持有界

- **VE-SDE vs VP-SDE：对比与联系**
  | 性质 | VE-SDE | VP-SDE |
  |---|---|---|
  | 漂移 $f(x,t)$ | $0$ | $-\frac{\beta(t)}{2}x$ |
  | 扩散 $g(t)$ | $\sqrt{d[\sigma(t)^2]/dt}$ | $\sqrt{\beta(t)}$ |
  | 离散前身 | SMLD/NCSN | DDPM |
  | $\text{Var}[x_t]$ | $\sigma(t)^2$（趋于无穷） | $1-e^{-\int\beta}$（有界） |
  | 信号缩放 | 无（$x_t \approx x_0 + \text{noise}$） | 有（$x_t \approx \sqrt{\bar\alpha_t}x_0 + \text{noise}$） |
  | 终态 $t=1$ | 近似纯噪声（大方差） | 近似标准高斯（有界方差） |
  - 统一视角：Karras et al. (2022) 表明VE-SDE和VP-SDE可以通过信号缩放 $s(t)$ 统一
  - 推理等价性（Kawar et al.）：两种SDE的逆向推理过程是等价的

- **正向过程的闭式解**
  - VE-SDE：$x_t = x_0 + \sigma(t)\,\epsilon$，$\epsilon \sim \mathcal{N}(0, I)$——纯加性噪声
  - VP-SDE：$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$——缩放信号 + 加性噪声
  - 闭式解的意义：训练时可以直接采样任意时刻 $t$ 的含噪数据，不需要逐步迭代
  - 信噪比（SNR）随时间的变化：
    - VE-SDE：$\text{SNR}(t) = \|x_0\|^2 / \sigma(t)^2$——单调递减
    - VP-SDE：$\text{SNR}(t) = \bar\alpha_t / (1-\bar\alpha_t)$——单调递减

**来源**：Song et al. (2021) Score-SDE; Tutorial_Diffusion_Imaging_Vision Sec 4.3; 2406.08929v2 Sec 2.4; 2508.01975v1 Sec 3.1

> **过渡**：正向SDE描述了数据如何被噪声逐步"淹没"。但我们的目标是从噪声恢复数据——逆向SDE提供了这条"回家之路"。

---

## 7.3 逆向SDE：从噪声到数据的采样过程

**核心观点**：Anderson (1982) 的逆向时间SDE定理提供了从噪声生成数据的一般方程——$dx = [f(x,t) - g(t)^2\nabla_x\log p_t(x)]\,dt + g(t)\,d\bar{w}$，其中 $\bar{w}$ 是逆向时间的布朗运动。得分函数 $\nabla\log p_t(x)$ 在逆向SDE中扮演核心角色——它提供了"恢复"概率分布的驱动力。正向过程"抹平"概率分布（信息丢失），逆向过程用得分函数"重建"概率分布（信息恢复）。VE-SDE和VP-SDE的逆向方程分别恢复SMLD采样和DDPM采样。

- **Anderson逆向时间SDE定理**
  - 定理陈述：对正向SDE $dx = f(x,t)\,dt + g(t)\,dw$，其逆向时间过程满足
    $$\boxed{dx = \left[f(x,t) - g(t)^2\,\nabla_x\log p_t(x)\right]\,dt + g(t)\,d\bar{w}}$$
  - $\bar{w}$ 是逆向时间的布朗运动
  - 与正向SDE的关键区别：
    1. 漂移项增加了 $-g(t)^2\nabla\log p_t(x)$——得分修正项
    2. 布朗运动方向反转：$d\bar{w}$（时间倒流）
  - 得分函数的角色：$\nabla\log p_t(x)$ 指向概率增长最快的方向——在逆向过程中，它"指导"粒子从低概率（纯噪声）回到高概率（数据分布）
  - 证明概要见附录7A

- **逆向VE-SDE：SMLD采样的连续形式**
  - 代入 $f=0$, $g(t) = \sqrt{d[\sigma(t)^2]/dt}$：
    $$dx = -\frac{d[\sigma(t)^2]}{dt}\,\nabla_x\log p_t(x)\,dt + \sqrt{\frac{d[\sigma(t)^2]}{dt}}\,d\bar{w}$$
  - 离散化恢复SMLD采样（7.5节将详细推导）
  - 直觉：纯得分驱动——没有漂移修正，完全依赖得分函数"引导"粒子回到数据

- **逆向VP-SDE：DDPM采样的连续形式**
  - 代入 $f(x,t) = -\beta(t)x/2$, $g(t) = \sqrt{\beta(t)}$：
    $$dx = -\beta(t)\left[\frac{x}{2} + \nabla_x\log p_t(x)\right]\,dt + \sqrt{\beta(t)}\,d\bar{w}$$
  - 离散化恢复DDPM采样（7.5节将详细推导）
  - 直觉：漂移修正 + 得分驱动——收缩力 $-\beta x/2$ 将信号拉回原点，得分力 $\nabla\log p_t$ 将信号拉向数据

- **得分函数与条件期望的联系**
  - 由Tweedie等式（第5章5.3节），对于VE-SDE：
    $$\nabla_x\log p_t(x) = \frac{\mathbb{E}[x_0|x_t] - x_t}{\sigma(t)^2}$$
  - 代入逆向VE-SDE：
    $$dx = \frac{d[\sigma(t)^2]/dt}{\sigma(t)^2}\left[x_t - \mathbb{E}[x_0|x_t]\right]\,dt + \sqrt{\frac{d[\sigma(t)^2]}{dt}}\,d\bar{w}$$
  - 逆向SDE的漂移力指向 $\mathbb{E}[x_0|x_t]$（MMSE估计方向）——与Tweedie等式呼应
  - 对于VP-SDE：
    $$\nabla_{x_t}\log p_t(x_t) = \frac{\mathbb{E}[\sqrt{\bar\alpha_t}x_0|x_t] - x_t}{1-\bar\alpha_t}$$
    - 代入逆向VP-SDE后同样可表示为条件期望的形式
  - 统一洞见：**逆向SDE的漂移力 = 指向MMSE估计的方向 + 随机探索**

- **逆向SDE的直觉理解**
  - 物理类比：正向SDE像"墨水滴入水中"——信息逐渐扩散消失；逆向SDE像"墨水从水中凝聚"——信息逐渐聚合恢复
  - 得分函数像"指南针"：在每个含噪状态 $x_t$ 处，得分函数指向"原始数据最可能在哪"
  - 布朗运动像"随机搜索"：防止所有粒子坍缩到同一点
  - 两者结合实现从纯噪声到数据分布的采样

- **逆向SDE与Langevin动力学的对比**
  - Langevin SDE：$dx = \nabla\log p(x)\,dt + \sqrt{2}\,dW$（固定得分、固定噪声系数）
  - 逆向扩散SDE：$dx = [f(x,t) - g(t)^2\nabla\log p_t(x)]\,dt + g(t)\,d\bar{w}$（时间条件得分、时间条件噪声）
  - Langevin是逆向扩散SDE的特例：$f=0$, $g(t) = \sqrt{2}$, $p_t \equiv p$
  - 核心区别：扩散SDE的得分函数随时间变化——从"粗糙得分"（大噪声时）到"精细得分"（小噪声时），形成了自然的多尺度采样策略

**来源**：Anderson (1982); Song et al. (2021) Score-SDE; Tutorial_Diffusion_Imaging_Vision Sec 4.3; 2406.08929v2 Sec 2.4

> **过渡**：逆向SDE是随机过程——每次采样轨迹都不同。是否存在一种确定性方法，能从相同的起点到达相同的终点？概率流ODE给出了肯定的回答。

---

## 7.4 概率流ODE：随机采样的确定性等价

**核心观点**：Song et al. (2020) 证明，每个逆向SDE都有一个确定性等价——概率流ODE（Probability Flow ODE）：$dx/dt = f(x,t) - \frac{1}{2}g(t)^2\nabla_x\log p_t(x)$。ODE与SDE在每个时刻有相同的边际分布 $p_t(x)$，但单条轨迹的演化方式完全不同——SDE描述"粒子的布朗运动"，ODE描述"气体的流线"。DDIM是PF-ODE的离散化特例，正如DDPM是逆向SDE的离散化特例。PF-ODE为扩散模型打开了确定性采样、似然计算和潜在空间操作的大门。

- **从SDE到ODE：去除随机性**
  - 逆向SDE：$dx = [f - g^2\nabla\log p_t]\,dt + g\,d\bar{w}$——包含随机项 $g\,d\bar{w}$
  - 关键观察：随机项的效果可以"吸收"到确定性漂移中
  - 由Fokker-Planck方程分析（附录5A的推广）：
    - SDE的Fokker-Planck方程：$\frac{\partial p_t}{\partial t} = -\nabla\cdot[(f-g^2\nabla\log p_t)p_t] + \frac{1}{2}\nabla\cdot[g^2\nabla p_t]$
    - 化简后：$\frac{\partial p_t}{\partial t} = -\nabla\cdot[\tilde{f}(x,t)\,p_t]$，其中 $\tilde{f}(x,t) = f(x,t) - \frac{1}{2}g(t)^2\nabla\log p_t(x)$
    - 这是连续性方程的形式！对应确定性ODE：$dx/dt = \tilde{f}(x,t)$
  - **概率流ODE**：
    $$\boxed{\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2\,\nabla_x\log p_t(x)}$$

- **PF-ODE的等价性证明**
  - 定理：PF-ODE与正向SDE在每个时刻 $t$ 有相同的边际分布 $p_t(x)$
  - 证明思路（通过Fokker-Planck方程）：
    1. 正向SDE的Fokker-Planck方程：$\frac{\partial p_t}{\partial t} = -\nabla\cdot(fp_t) + \frac{1}{2}\nabla\cdot[g^2\nabla p_t]$
    2. PF-ODE的连续性方程：$\frac{\partial p_t}{\partial t} = -\nabla\cdot(\tilde{f}p_t)$
    3. 代入 $\tilde{f} = f - \frac{1}{2}g^2\nabla\log p_t$：
       $$-\nabla\cdot(\tilde{f}p_t) = -\nabla\cdot(fp_t) + \frac{1}{2}\nabla\cdot[g^2\nabla\log p_t \cdot p_t]$$
    4. 利用 $\nabla\log p_t \cdot p_t = \nabla p_t$，得 $-\nabla\cdot(\tilde{f}p_t) = -\nabla\cdot(fp_t) + \frac{1}{2}\nabla\cdot[g^2\nabla p_t]$
    5. 与正向SDE的Fokker-Planck方程完全相同 ✓

- **PF-ODE的物理直觉**
  - SDE描述的是个体粒子的布朗运动——充满随机性
  - PF-ODE描述的是气体流线——确定性的速度场
  - 类比：想象一条河——SDE描述水中一片树叶的轨迹（随机漂移），PF-ODE描述水流的流线（确定性的速度场）
  - 边际分布等价：大量树叶的分布 = 大量流线的分布——"殊途同归"
  - 单条轨迹不同：同一片树叶的SDE轨迹和ODE轨迹可能完全不同

- **VE-SDE和VP-SDE的PF-ODE**
  - VE-SDE的PF-ODE：$\frac{dx}{dt} = -\frac{1}{2}\frac{d[\sigma(t)^2]}{dt}\nabla_x\log p_t(x)$
    - 代入Tweedie等式：$\frac{dx}{dt} = -\frac{1}{2\sigma(t)^2}\frac{d[\sigma(t)^2]}{dt}[\mathbb{E}[x_0|x_t] - x_t]$
  - VP-SDE的PF-ODE：$\frac{dx}{dt} = -\frac{\beta(t)}{2}x - \frac{\beta(t)}{2}\nabla_x\log p_t(x) = -\frac{\beta(t)}{2}\left[x + \nabla_x\log p_t(x)\right]$
    - 代入Tweedie等式：$\frac{dx}{dt} = -\frac{\beta(t)}{2}\left[\frac{1}{\bar\alpha_t}\mathbb{E}[x_0|x_t]\right]$

- **DDIM = PF-ODE的离散化**
  - PF-ODE是连续方程，需要离散化求解
  - Euler离散化：$x_{t-\Delta t} = x_t + \tilde{f}(x_t, t)\,\Delta t$
  - 对于简单扩散（VE-SDE, $f=0$）：
    $$x_{t-\Delta t} = x_t - \frac{\Delta t}{2}\sigma_q^2\nabla\log p_t(x_t) = x_t + \frac{\Delta t}{2t}[\mathbb{E}[x_0|x_t] - x_t]$$
  - 当 $\Delta t \to 0$，$\lambda = \sigma_t/(\sigma_{t-\Delta t} + \sigma_t) \to 1/2$
  - 恰好恢复DDIM更新（2406.08929v2 Eq. 33-35）
  - **核心结论**：DDPM = 逆向SDE的Euler-Maruyama离散化；DDIM = PF-ODE的Euler离散化

- **PF-ODE的实践意义**
  - **确定性采样**：同一噪声起点 $x_1$，ODE给出确定性输出——便于复现和插值
  - **似然计算**：PF-ODE定义了从噪声到数据的可逆变换——通过变量替换公式可计算精确似然
  - **潜在空间操作**：ODE将数据映射到噪声空间，可在噪声空间做插值、编辑
  - **灵活求解器**：ODE可以用任意数值求解器（RK4、DPM++等），不限于Euler方法
  - **连续正规化流**：PF-ODE定义了一个连续时间的正规化流——第14章将进一步展开

- **DDPM vs DDIM：随机性与确定性的对比**
  - DDPM（随机采样）：每次采样轨迹不同，同一起点可能产生不同输出
  - DDIM（确定性采样）：同一噪声起点确定性地映射到同一输出
  - DDPM更强但更慢：随机性提供更好的多样性，但需要更多步数
  - DDIM更快但可控：确定性使少步采样成为可能，但多样性受限
  - 实践中的选择：生成任务用DDPM（多样性），逆问题用DDIM/ODE（确定性、可控性）

**来源**：Song et al. (2020); 2406.08929v2 Sec 3.5; 2508.01975v1 Sec 3.1; Tutorial_Diffusion_Imaging_Vision Sec 4.4

> **过渡**：逆向SDE和PF-ODE提供了从噪声生成数据的连续方程，但计算机只能执行离散迭代。如何将连续方程离散化为可执行的算法？数值离散化方法给出了答案。

---

## 7.5 数值离散化：从连续方程到可执行算法

**核心观点**：连续时间的SDE和ODE需要数值离散化才能在计算机上执行。Euler-Maruyama方法是SDE的基本离散化方案，其在逆向VP-SDE上的应用恰好恢复DDPM迭代——这是Score-SDE框架的核心统一结论。类似地，逆向VE-SDE的离散化恢复SMLD采样，PF-ODE的离散化恢复DDIM。高阶数值求解器（RK4、DPM++）提供更少步数的加速采样。三种采样器（DDPM/DDIM/PF-ODE）的对比揭示了随机性与确定性的根本权衡。

- **Euler-Maruyama方法：SDE的基本离散化**
  - 一般SDE $dx = a(x,t)\,dt + b(x,t)\,dw$ 的Euler-Maruyama离散化：
    $$x_{t-\Delta t} = x_t - a(x_t, t)\,\Delta t + b(x_t, t)\,\sqrt{\Delta t}\,z$$
    其中 $z \sim \mathcal{N}(0, I)$
  - 与第4章4.3节的联系：ULA = Langevin SDE的Euler-Maruyama离散化
  - 精度：$O(\Delta t)$ 强收敛阶
  - 更高阶方法：Milstein方法（$O(\Delta t)$ 强收敛阶，更好的常数），但实现更复杂

- **DDPM = 逆向VP-SDE的Euler-Maruyama离散化**
  - 逆向VP-SDE：$dx = -\beta(t)[x/2 + \nabla\log p_t(x)]\,dt + \sqrt{\beta(t)}\,d\bar{w}$
  - Euler-Maruyama离散化：
    $$x_{t-\Delta t} = x_t + \beta(t)\Delta t\left[\frac{x_t}{2} + \nabla\log p_t(x_t)\right] + \sqrt{\beta(t)\Delta t}\,z$$
  - 令 $\beta_i = \beta(t)\Delta t$，$\Delta t = 1/N$：
    $$x_{i-1} \approx \frac{1}{\sqrt{1-\beta_i}}\left[x_i + \frac{\beta_i}{2}\nabla\log p_i(x_i)\right] + \sqrt{\beta_i}\,z_i$$
  - 用得分网络替换 $\nabla\log p_i(x_i) = s_\theta(x_i, i)$，即得DDPM迭代——**DDPM是逆向VP-SDE的Euler-Maruyama离散化**
  - 这是一个深刻的统一结论：DDPM不再是孤立的算法，而是SDE求解器的一个实例

- **SMLD采样 = 逆向VE-SDE的Euler-Maruyama离散化**
  - 逆向VE-SDE：$dx = -\frac{d[\sigma(t)^2]}{dt}\nabla\log p_t(x)\,dt + \sqrt{\frac{d[\sigma(t)^2]}{dt}}\,d\bar{w}$
  - 令 $\alpha_i = \sigma_i^2 - \sigma_{i-1}^2$（对应 $\frac{d[\sigma(t)^2]}{dt}\Delta t$）：
    $$x_{i-1} = x_i + \alpha_i\nabla\log p_i(x_i) + \sqrt{\alpha_i}\,z_i$$
  - 恰好是SMLD/退火Langevin的采样迭代——**SMLD是逆向VE-SDE的Euler-Maruyama离散化**

- **DDIM = PF-ODE的Euler离散化**
  - 7.4节已推导：PF-ODE的Euler离散化恢复DDIM
  - 三种采样的统一关系：

  | 采样器 | 连续方程 | 离散化方法 | 特性 |
  |---|---|---|---|
  | DDPM | 逆向VP-SDE | Euler-Maruyama | 随机 |
  | SMLD | 逆向VE-SDE | Euler-Maruyama | 随机 |
  | DDIM | PF-ODE | Euler | 确定性 |

- **高阶求解器与加速采样**
  - Euler-Maruyama仅有一阶精度——步数多、速度慢
  - Runge-Kutta方法（用于PF-ODE）：
    - RK4：4阶精度，每步4次网络评估
    - 在步数相同时，RK4比Euler精度高得多
  - DPM-Solver / DPM++（Lu et al. 2022）：
    - 专为扩散模型设计的ODE求解器
    - 利用扩散ODE的特殊结构（指数衰减的噪声调度）
    - 20步即可达到DDPM 1000步的质量
  - 随机高阶方法：
    - 将Euler-Maruyama替换为高阶SDE求解器
    - 理论上精度更高，但实践中提升有限（因为得分估计误差是主要瓶颈）

- **采样器选择：DDPM vs DDIM vs PF-ODE**
  - **DDPM（随机采样）**：
    - 优势：多样性好、对得分估计误差鲁棒
    - 劣势：步数多（通常1000步）、不可复现
    - 适用场景：无条件生成、需要多样性的任务
  - **DDIM（确定性采样）**：
    - 优势：步数少（20-50步）、可复现、潜在空间操作
    - 劣势：多样性受限、对得分估计误差敏感
    - 适用场景：图像编辑、逆问题求解、需要可控性的任务
  - **PF-ODE + 高阶求解器**：
    - 优势：步数最少（10-20步）、精确似然计算
    - 劣势：需要更复杂的求解器实现
    - 适用场景：需要精确似然或极少步数的场景
  - **核心权衡**：随机性 ↔ 确定性 ↔ 计算效率
    - 更多随机性 → 更好的多样性、鲁棒性
    - 更少随机性 → 更快的速度、更好的可控性

- **温度参数与采样控制**
  - 回顾第5章5.4节的温度视角：MAP是"零温度"的MMSE
  - 在SDE框架中，可以引入温度参数 $\eta \in [0, 1]$ 控制随机性：
    $$dx = \left[f - \eta\,g^2\nabla\log p_t\right]\,dt + \sqrt{\eta}\,g\,d\bar{w}$$
  - $\eta = 1$：完整逆向SDE（DDPM）
  - $\eta = 0$：PF-ODE（DDIM）
  - $0 < \eta < 1$：部分随机——在多样性和确定性之间插值
  - 这一温度视角统一了DDPM和DDIM——它们不是两种不同的方法，而是同一连续框架的两种极端

**来源**：Song et al. (2021); Tutorial_Diffusion_Imaging_Vision Sec 4.3-4.4; 2406.08929v2 Sec 2.4, 3.5; Kloeden & Platen (2011); Lu et al. (2022) DPM-Solver

> **过渡**：理论框架已经完备——正向SDE定义加噪，逆向SDE/PF-ODE定义去噪，数值离散化实现采样。现在需要将这些理论落地为工程实践。

---

## 7.6 实践：用扩散SDE实现图像生成

**核心观点**：将SDE框架从理论转化为实践需要解决三个工程问题：如何设计噪声调度、如何训练时间条件得分网络、如何选择和配置采样器。本节给出完整的实践流程，并通过实验对比展示不同设计的效应。关键洞察：噪声调度设计等价于选择正向SDE的系数 $f$ 和 $g$，训练目标等价于第6章的DSM加上时间条件，采样流程等价于求解逆向SDE或PF-ODE。

- **噪声调度设计**
  - 噪声调度的角色：决定正向SDE如何从数据过渡到噪声
  - VE-SDE的噪声调度：
    - 常用选择：$\sigma(t) = \sigma_{\max}^{t}\sigma_{\min}^{1-t}$（几何插值）
    - $\sigma_{\min}$：最小噪声（保留细节），$\sigma_{\max}$：最大噪声（覆盖全局）
    - 实践值：$\sigma_{\min} \approx 0.01$, $\sigma_{\max} \approx 50$（CIFAR-10）
  - VP-SDE的噪声调度：
    - 常用选择：线性调度 $\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})$
    - 实践值：$\beta_{\min} = 0.1$, $\beta_{\max} = 20$（CIFAR-10）
    - 改进：余弦调度（Nichol & Dhariwal 2021）：$\bar\alpha_t = \frac{f(t)}{f(0)}$，$f(t) = \cos^2\frac{\pi(t+s)}{2(1+s)}$
  - 噪声调度对生成质量的影响：
    - 调度太慢（噪声增加不够快）→ 正向过程不充分 → 逆向采样困难
    - 调度太快（噪声增加过快）→ 训练信号太弱 → 得分估计不准
    - 余弦调度的优势：信噪比变化更均匀，避免线性调度的"过早饱和"

- **时间条件得分网络的训练**
  - 网络架构：DRUNet（第6章6.6节）+ 时间条件输入
  - 训练目标：第6章DSM的时间条件版本
    - VE-SDE训练：
      $$\mathcal{J}(\theta) = \frac{1}{2}\mathbb{E}_{t}\mathbb{E}_{x_0}\mathbb{E}_{\epsilon}\left[\left\|s_\theta(x_t, t) + \frac{\epsilon}{\sigma(t)}\right\|^2\right\|g(t)\|^2\lambda(t)$$
    - VP-SDE训练：
      $$\mathcal{J}(\theta) = \frac{1}{2}\mathbb{E}_{t}\mathbb{E}_{x_0}\mathbb{E}_{\epsilon}\left[\left\|\epsilon_\theta(x_t, t) - \epsilon\right\|^2\right]$$
    - 两种训练目标在数学上等价（通过Tweedie等式互推）
  - 时间嵌入：将连续时间 $t$ 通过正弦位置编码映射为高维向量
  - 时间采样策略：训练时 $t \sim \mathcal{U}(0, 1)$——均匀采样保证所有时间步被充分训练
  - 与第6章DSM的对应：6.5节的多噪声水平训练 → 本章的连续时间训练

- **采样流程**
  - 输入：训练好的得分网络 $s_\theta(x, t)$（或噪声预测网络 $\epsilon_\theta(x, t)$）
  - VE-SDE采样流程：
    1. 采样 $x_1 \sim \mathcal{N}(0, \sigma_{\max}^2 I)$
    2. 选择求解器（Euler-Maruyama / 高阶SDE求解器）
    3. 从 $t=1$ 到 $t=0$ 求解逆向VE-SDE
    4. 输出 $x_0$ 作为生成样本
  - VP-SDE采样流程：
    1. 采样 $x_1 \sim \mathcal{N}(0, I)$
    2. 选择求解器（DDPM / DDIM / DPM-Solver）
    3. 从 $t=1$ 到 $t=0$ 求解逆向VP-SDE或PF-ODE
    4. 输出 $x_0$ 作为生成样本

- **实验对比**
  - VE-SDE vs VP-SDE的生成质量（FID指标）
  - 不同噪声调度的效应（线性 vs 余弦 vs 自适应）
  - 不同采样器的步数-质量曲线（DDPM 1000步 vs DDIM 50步 vs DPM-Solver 20步）
  - 采样轨迹可视化：从纯噪声到清晰图像的逐步去噪过程
  - 与第6章退火Langevin的对比：连续化带来的质量提升

- **从PnP-ULA到扩散SDE：逆问题视角的回顾**
  - 第5章PnP-ULA：单一噪声水平的得分驱动采样
  - 第6章退火Langevin：多噪声水平的离散采样
  - 本章扩散SDE：连续时间的统一框架
  - 演化路径：PnP-ULA → 退火Langevin → 扩散SDE
  - 核心进步：从"单温度"到"多温度"到"连续温度调度"
  - 预告第13章：条件扩散采样 = 扩散SDE + 似然约束 = 逆问题求解

**来源**：Song et al. (2021); Ho et al. (2020); Nichol & Dhariwal (2021); deepinv demo_diffusion_sde; 2406.08929v2

> **过渡**：本章建立了扩散模型的SDE统一视角——正向SDE加噪、逆向SDE去噪、PF-ODE确定性采样、数值离散化实现。第8章将切换到变分路径——从ELBO推导扩散模型的训练目标，最终在第12章证明得分匹配与变分下界的等价性。

---

## 附录7A Anderson逆向时间SDE定理的证明概要

> 定位：为7.3节提供Anderson (1982) 逆向时间SDE定理的证明概要。完整证明需要随机分析的工具（Kunita-Watanabe分解等），这里给出核心思路和关键步骤。

- **定理陈述**
  - 正向SDE：$dx = f(x,t)\,dt + g(t)\,dw$，$t \in [0, T]$
  - 则逆向时间过程 $y(s) = x(T-s)$ 满足
    $$dy = \left[f(y, T-s) - g(T-s)^2\,\nabla_y\log p_{T-s}(y)\right]\,ds + g(T-s)\,d\bar{w}$$

- **证明思路**
  - 利用Fokker-Planck方程分析正向过程的密度演化
  - 逆向时间的密度演化：$q_s(y) = p_{T-s}(y)$
  - 对 $q_s(y)$ 建立"逆向Fokker-Planck方程"
  - 由Fokker-Planck与SDE的对应关系，得到逆向SDE
  - 关键步骤：逆向漂移项的得分修正 $-g^2\nabla\log p_t$ 来自密度的时空梯度关系

- **特殊情况：Langevin SDE**
  - 正向：$dx = \nabla\log p(x)\,dt + \sqrt{2}\,dW$（平稳分布为 $p$）
  - 逆向：$dx = [\nabla\log p(x) - 2\nabla\log p(x)]\,dt + \sqrt{2}\,d\bar{w} = -\nabla\log p(x)\,dt + \sqrt{2}\,d\bar{w}$
  - 逆向Langevin的漂移项反号——从"爬坡"变为"下坡"

**来源**：Anderson (1982); Winkler (2021); Risken (1996)

---

## 附录7B VE-SDE与VP-SDE的推理等价性

> 定位：为7.2节的VE-SDE与VP-SDE对比提供补充材料。Kawar et al. 的观察表明，两种SDE在推理（采样）阶段是等价的，差异主要在训练阶段。

- **等价性的直观解释**
  - VE-SDE：$x_t = x_0 + \sigma(t)\epsilon$（加性噪声，无信号缩放）
  - VP-SDE：$x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$（缩放信号 + 加性噪声）
  - 令 $\hat{x}_t = x_t / \sqrt{\bar\alpha_t}$（重缩放），则 $\hat{x}_t = x_0 + \hat{\sigma}(t)\epsilon$
  - 重缩放后的VP-SDE在形式上等同于VE-SDE

- **推理等价性的含义**
  - 如果训练好的模型质量相同，用VE-SDE还是VP-SDE推理不影响结果
  - 但训练动态不同：VP-SDE的信号缩放帮助稳定训练
  - 实践中，VP-SDE更常用于图像生成，VE-SDE更常用于逆问题

- **Karras et al. (2022) 的统一框架**
  - 引入信号缩放 $s(t)$ 和噪声调度 $\sigma(t)$ 作为独立设计选择
  - 统一SDE：$dx = \frac{\dot{s}(t)}{s(t)}x\,dt + s(t)\sqrt{2\dot{\sigma}(t)\sigma(t)}\,dw$
  - VE-SDE：$s(t) \equiv 1$；VP-SDE：$s(t) = e^{-\frac{1}{2}\int_0^t\beta(\tau)d\tau}$

**来源**：Kawar et al. (2022); Karras et al. (2022); Song et al. (2021)

---

## 本章逻辑流总结

```
7.1 从Langevin到扩散（离散→连续的动机）
      │
      │ "连续时间的加噪过程是什么？"
      ▼
7.2 正向SDE（VE-SDE + VP-SDE）
      │
      │ "如何从噪声恢复数据？"
      ▼
7.3 逆向SDE（Anderson定理 + 得分驱动）
      │
      │ "有确定性版本吗？"
      ▼
7.4 概率流ODE（确定性等价 + DDIM）
      │
      │ "如何计算？"
      ▼
7.5 数值离散化（DDPM=SDE离散 + 采样器对比）
      │
      │ "如何实践？"
      ▼
7.6 实践（噪声调度 + 训练 + 采样 + 实验）
      │
      ├──→ 变分路径 → 第8章（ELBO → VAE → 层级VAE）
      └──→ 条件扩散 → 第13章（逆问题闭环）
```

**核心叙事**：第6章将得分匹配从单一噪声水平扩展到多噪声水平（NCSN），架起了通往扩散模型的桥梁。本章将这座桥梁推到连续时间——NCSN的离散噪声调度变为VE-SDE，DDPM的离散迭代变为VP-SDE，两者统一在一般SDE框架 $dx = f(x,t)\,dt + g(t)\,dw$ 之下。Anderson的逆向时间SDE定理提供了从噪声生成数据的一般方程，得分函数 $\nabla\log p_t(x)$ 再次成为核心驱动力。概率流ODE揭示了随机采样的确定性等价——DDPM和DDIM不再是对立的方法，而是同一连续框架的两种离散化。数值离散化将理论落地为算法——DDPM = 逆向SDE的Euler-Maruyama，DDIM = PF-ODE的Euler。本章的核心洞见是：**扩散模型 = 正向SDE（定义加噪）+ 学习得分函数（驱动逆向）+ 数值离散化（实现采样）**——三位一体，完成了采样路径从"多步Langevin"到"连续扩散过程"的关键跃迁。

---

## 材料覆盖状态

| 子主题 | 来源 | 状态 |
|---|---|---|
| 从Langevin到扩散SDE的连续推广 | Pock L2 P14; 2406.08929v2 Sec 2.4 | ✅ |
| "离散→连续→重离散化"螺旋 | book_plan; 成书绪论 | ✅ |
| 一般正向SDE框架 | Tutorial_Diffusion Sec 4.3; 2508.01975v1 | ✅ |
| VE-SDE推导 | Tutorial_Diffusion Theorem 4.3; Song et al. (2021) | ✅ |
| VP-SDE推导 | Tutorial_Diffusion Theorem 4.1; Song et al. (2021) | ✅ |
| VE-SDE vs VP-SDE对比 | Tutorial_Diffusion Sec 4.3; Karras et al. (2022) | ✅ |
| 正向过程闭式解 | 2406.08929v2; Tutorial_Diffusion | ✅ |
| Anderson逆向SDE定理 | Anderson (1982); 2406.08929v2; 2508.01975v1 | ✅ |
| 逆向VE-SDE | Tutorial_Diffusion Theorem 4.4 | ✅ |
| 逆向VP-SDE | Tutorial_Diffusion Theorem 4.2 | ✅ |
| 得分与条件期望的联系 | 2406.08929v2 Eq 28; Tweedie等式 | ✅ |
| 概率流ODE推导 | 2406.08929v2 Sec 3.5; Song et al. (2020) | ✅ |
| PF-ODE等价性证明 | 2406.08929v2; Fokker-Planck分析 | ✅ |
| DDIM = PF-ODE离散化 | 2406.08929v2 Sec 3.5 | ✅ |
| Euler-Maruyama方法 | 第4章4.3节; Kloeden & Platen | ✅ |
| DDPM = 逆向VP-SDE离散化 | Tutorial_Diffusion Sec 4.3; 2406.08929v2 | ✅ |
| SMLD = 逆向VE-SDE离散化 | Tutorial_Diffusion Sec 4.3 | ✅ |
| 高阶求解器（DPM++等） | Lu et al. (2022) | 🟡 需补充细节 |
| 噪声调度设计 | Nichol & Dhariwal (2021); Ho et al. (2020) | ✅ |
| 时间条件得分网络训练 | Song et al. (2021); 第6章6.6节 | ✅ |
| 实验对比数据 | deepinv demo_diffusion_sde; 文献 | 🟡 需补充实验图 |
| 温度参数与采样控制 | Pock L2 P14; Habring et al. (2025) | ✅ |
| VE/VP推理等价性 | Kawar et al. (2022); Karras et al. (2022) | ✅ |
