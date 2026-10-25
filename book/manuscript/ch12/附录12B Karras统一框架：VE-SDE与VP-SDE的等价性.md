# 附录12B Karras et al. (2022) 统一框架：VE-SDE 与 VP-SDE 的等价性

本附录介绍 Karras et al. (2022) 的统一参数化框架，展示 VE-SDE 和 VP-SDE 是该框架的不同参数化选择，从而在更基础的层面上说明：DSM ≡ VLB 的等价性不依赖于特定的 SDE 形式。正文 12.3 节观察 2 和 12.5 节都用到了它。一句话定位：如果说 12.4 证明了"两条路径在同一坐标系下等价"，那么本附录展示的是——换不换坐标系都不影响这个等价。

---

## B.1 统一参数化

### 一般加噪过程

Karras et al. (2022) 把扩散模型的加噪过程统一为：

$$\mathbf{x}_t = s(t)\,\mathbf{x}_0 + s(t)\,\sigma(t)\,\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

其中：
- $s(t)$：信号缩放因子（$s(t) > 0$）
- $\sigma(t)$：噪声水平（$\sigma(t) > 0$，关于 $t$ 单调递增）

边际分布为：

$$q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t | s(t)\,\mathbf{x}_0,\, s^2(t)\sigma^2(t)\,I)$$

### VE-SDE 和 VP-SDE 作为特例

**VE-SDE（NCSN/SMLD）**：

$$s(t) = 1, \quad \sigma(t) = \sigma_{\min}\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t$$

加噪过程：$\mathbf{x}_t = \mathbf{x}_0 + \sigma(t)\varepsilon$——信号幅度不变，噪声叠加增长。

**VP-SDE（DDPM）**：

$$s(t) = \sqrt{\bar\alpha(t)}, \quad \sigma(t) = \sqrt{\frac{1-\bar\alpha(t)}{\bar\alpha(t)}}$$

加噪过程：$\mathbf{x}_t = \sqrt{\bar\alpha(t)}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha(t)}\,\varepsilon$——信号逐渐衰减，噪声相对增长。

### 等价的推理过程

Karras et al. (2022) 的关键观察：**推理过程（从 $\mathbf{x}_t$ 估计 $\mathbf{x}_0$）只取决于信噪比 $\text{SNR}(t) = 1/\sigma^2(t)$，而不取决于信号缩放 $s(t)$**。

由 Tweedie 等式，MMSE 估计为：

$$\hat{x}_0(\mathbf{x}_t) = \mathbb{E}[\mathbf{x}_0|\mathbf{x}_t] = \frac{\mathbf{x}_t + \sigma^2(t)\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)}{s(t)}$$

去噪器的输出（预测 $\mathbf{x}_0$）只通过 $1/s(t)$ 受信号缩放影响——这是简单归一化，不改变估计本质。

> **从更深层次看**，这一观察把"哪种加噪方案更好"这个工程问题彻底降维了：只要信噪比曲线相同，再怎么换信号缩放 $s(t)$，模型推理的信息都不变。这一转变意味着，VE 与 VP 的差别最终只是一个"读数单位"问题，而非本质差别。

---

## B.2 统一框架下的 Tweedie 等式

在统一参数化下，Tweedie 等式的一般形式为：

$$\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t) = \frac{s(t)\,\mathbb{E}[\mathbf{x}_0|\mathbf{x}_t] - \mathbf{x}_t}{s^2(t)\sigma^2(t)} = \frac{\hat{x}_0(\mathbf{x}_t) - \mathbf{x}_t/s(t)}{s(t)\sigma^2(t)}$$

在 ε 预测参数化下（$\hat{x}_0 = (\mathbf{x}_t - s(t)\sigma(t)\varepsilon_\theta)/(s(t))$）：

$$\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t) = -\frac{\varepsilon_\theta(\mathbf{x}_t, t)}{s(t)\sigma(t)}$$

### VE-SDE 特例
$s(t)=1$：$\nabla\log p_t(\mathbf{x}_t) = -\varepsilon_\theta/\sigma(t)$

### VP-SDE 特例
$s(t)=\sqrt{\bar\alpha_t}$，$s(t)\sigma(t)=\sqrt{1-\bar\alpha_t}$：
$$\nabla\log p_t(\mathbf{x}_t) = -\frac{\varepsilon_\theta(\mathbf{x}_t, t)}{\sqrt{1-\bar\alpha_t}}$$

---

## B.3 统一框架下的训练目标

### 统一 DSM 目标

$$\mathcal{J}_\text{DSM}(\theta) = \frac{1}{2}\int \frac{\lambda(t)}{s^2(t)\sigma^2(t)}\,\mathbb{E}_{\mathbf{x}_0,\varepsilon}\left[\|\varepsilon - \varepsilon_\theta(\mathbf{x}_t, t)\|^2\right]dt$$

注意分母 $s^2(t)\sigma^2(t)$ 正是扰动方差（B.1 中 $q(\mathbf{x}_t|\mathbf{x}_0)$ 的方差），它统一了 12.4 里"VE 用 $\sigma^2$、VP 用 $1-\bar\alpha$"两种分母——因为 VP 下 $s^2\sigma^2 = \bar\alpha \cdot (1-\bar\alpha)/\bar\alpha = 1-\bar\alpha$。

### 统一 VLB 目标

连续时间 VLB 的一致性积分在统一参数化下为：

$$L_\text{VLB}(\theta) = \int w(t)\,\mathbb{E}_{\mathbf{x}_0,\varepsilon}\left[\|\varepsilon - \varepsilon_\theta(\mathbf{x}_t, t)\|^2\right]dt$$

$w(t)$ 由 $(s(t), \sigma(t))$ 的动力学方程确定。

### 等价性不受参数化影响

由于两种 SDE 形式（VE 和 VP）只是统一框架的不同参数化选择，且推理过程只取决于 SNR $\sigma(t)$，DSM ≡ VLB 的等价性不依赖于 SDE 的具体形式——它在统一框架下自然成立。选 VE-SDE 还是 VP-SDE（或其他参数化）只影响权重 $w(t)$ 和 $\lambda(t)/\sigma^2(t)$ 的具体表达式，不影响等价性本身。这进一步验证等价性的必然性——它不是某种特定参数化的巧合，而是概率分布"梯度信息"与"函数值信息"等价性的体现。

> **这一转变意味着**：12.4 在 VP-SDE 下证明的等价性，本可以放进任何参数化下重演一遍。换句话说，等价性不是"VP-SDE 的专利"，而是"得分函数与似然下界是同一对象两面"这一更基本事实的必然推论——具体用哪套记号，只是这一事实的一次显影。

---

## B.4 从 VE-SDE 到 VP-SDE 的权重映射

给定 VE-SDE 下的权重 $\lambda_\text{VE}(\sigma)$，可通过 SNR 对应映射到 VP-SDE 下的权重：

1. 建立 SNR 对应：$\sigma_\text{VE} \leftrightarrow \sigma_\text{VP}(t) = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$
2. 把 $\lambda_\text{VE}(\sigma)$ 里的 $\sigma$ 替换成 $\sigma_\text{VP}(t)$
3. **把信号缩放乘回去**：$\lambda_\text{VE}$ 的自变量是扰动噪声的标准差，在统一记法下它等于 $s(t)\sigma(t)$（B.1 中 $q(\mathbf{x}_t|\mathbf{x}_0)$ 的方差为 $s^2\sigma^2$），因此映射到 VP 后应写成 $\lambda_\text{VP}(t)=s^2(t)\sigma_\text{VP}^2(t)$

第 3 步最容易被漏掉。当 $\lambda_\text{VE}(\sigma)=\sigma^2$（NCSN 默认）时：

$$\lambda_\text{VP}(t) = s^2(t)\sigma_\text{VP}^2(t) = \bar\alpha_t\cdot\frac{1-\bar\alpha_t}{\bar\alpha_t} = 1-\bar\alpha_t$$

此时 DSM 权重（含 B.3 目标前的 $1/2$）为：

$$\frac{1}{2}\cdot\frac{\lambda_\text{VP}(t)}{s^2(t)\sigma_\text{VP}^2(t)} = \frac{1-\bar\alpha_t}{2(1-\bar\alpha_t)} = \frac{1}{2}$$

**它对任意 $t$ 精确等于 $1/2$**，与 VP-SDE 下简化 VLB 的均匀权重逐点相同（仅差常数因子 2），不是“$\bar\alpha_t\approx1$ 时的近似”。若漏掉第 3 步，会得到 $1/\bar\alpha_t$，只能靠 $\bar\alpha_t\approx1$ 勉强近似——这正是把一个精确恒等式误读成近似的典型来源（参见 12.4 “简化目标下的等价性”）。

---

## B.5 EDM 最优网络预处理（preconditioning）

> 本节回应附录 7B 的迁移说明：7.2 与附录 7B 只给出"VE/VP 等价"的结论与直觉，此处补上 Karras et al. (2022) 最优设计 $c_{\text{skip}}/c_{\text{out}}/c_{\text{in}}/c_{\text{noise}}$ 的逐项推导。这也是实验 7.7-1 代码中同名变量的公式出处。

### 为什么不让网络直接学去噪器

B.1 已说明推理只取决于噪声水平 $\sigma$。但若把去噪器直接实现为网络 $D_\theta(\mathbf{x},\sigma)$ 来训练并不理想——含噪输入 $\mathbf{x} = \mathbf{x}_0 + \sigma\,\varepsilon$ 的数值幅度随 $\sigma$ 剧烈变化（$\sigma$ 越大输入越"爆"），直接回归会使梯度尺度失衡。Karras et al. 的做法是训练一个"良态"网络 $F_\theta$，再用一组只依赖 $\sigma$ 的缩放函数把它组装成去噪器：

$$D_\theta(\mathbf{x};\sigma) = c_{\text{skip}}(\sigma)\,\mathbf{x} + c_{\text{out}}(\sigma)\,F_\theta\!\big(c_{\text{in}}(\sigma)\,\mathbf{x},\; c_{\text{noise}}(\sigma)\big)$$

四个系数各司其职：$c_{\text{in}}$ 把输入归一到单位方差，$c_{\text{out}}$ 缩放输出，$c_{\text{skip}}$ 是一条跳连、让网络在"预测 $\mathbf{x}_0$"与"预测噪声"之间自动插值，$c_{\text{noise}}$ 把噪声水平映射成网络的条件输入。

### 从第一性原理导出最优设计

记数据的标准差为 $\sigma_{\text{data}}$（对归一化到 $[-1,1]$ 的图像约 $0.5$）。把上式代入 B.3 的统一 DSM 目标，网络 $F_\theta$ 的**有效训练目标**与**有效权重**为：

$$F_{\text{target}}(\mathbf{x}_0,\varepsilon;\sigma) = \frac{1}{c_{\text{out}}(\sigma)}\big(\mathbf{x}_0 - c_{\text{skip}}(\sigma)\,(\mathbf{x}_0 + \sigma\,\varepsilon)\big), \qquad w(\sigma) = \lambda(\sigma)\,c_{\text{out}}(\sigma)^2$$

Karras et al. 用四条原则唯一定出各系数：

1. **输入单位方差** $\Rightarrow c_{\text{in}}(\sigma) = 1/\sqrt{\sigma^2 + \sigma_{\text{data}}^2}$；
2. **有效目标单位方差** $\Rightarrow c_{\text{out}}(\sigma)^2 = (1-c_{\text{skip}})^2\,\sigma_{\text{data}}^2 + c_{\text{skip}}^2\,\sigma^2$；
3. **最小化 $c_{\text{out}}$（尽量少放大网络误差）**：对上式关于 $c_{\text{skip}}$ 求导置零，得 $c_{\text{skip}}(\sigma) = \sigma_{\text{data}}^2/(\sigma^2 + \sigma_{\text{data}}^2)$，回代得 $c_{\text{out}}(\sigma) = \sigma\,\sigma_{\text{data}}/\sqrt{\sigma^2 + \sigma_{\text{data}}^2}$；
4. **有效权重对 $\sigma$ 均匀**（$w(\sigma)=1$）$\Rightarrow \lambda(\sigma) = 1/c_{\text{out}}(\sigma)^2 = (\sigma^2 + \sigma_{\text{data}}^2)/(\sigma\,\sigma_{\text{data}})^2$。

条件输入 $c_{\text{noise}}$ 不受最优性约束，经验取 $c_{\text{noise}}(\sigma) = \tfrac{1}{4}\ln\sigma$。汇总即 EDM 论文 Table 1 的 "Ours" 列：

$$c_{\text{in}}(\sigma) = \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}},\quad c_{\text{skip}}(\sigma) = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2},\quad c_{\text{out}}(\sigma) = \frac{\sigma\,\sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}},\quad c_{\text{noise}}(\sigma) = \tfrac{1}{4}\ln\sigma$$

### 与 VE / VP 的关系：一套设计"一记两通"

上述推导写在 VE 规范（$s(t)=1$，$\mathbf{x} = \mathbf{x}_0 + \sigma\,\varepsilon$）下，四个系数**只依赖 $\sigma$**。回到 B.1 的核心观察——推理只取决于噪声水平 $\sigma(t)$、与信号缩放 $s(t)$ 无关——对 VP-SDE 只需把 $\sigma$ 换成 $\sigma_{\text{VP}}(t) = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$，整套最优设计原样适用；两种参数化下网络输出动态范围的差异，由 $c_{\text{in}}, c_{\text{out}}$ 自动吸收。

这说明"VE 与 VP 是同一框架的两种参数化"不仅体现在推理层面（B.1–B.2），也落地到**训练实现**层面：连最优网络预处理都共享同一组关于 $\sigma$ 的闭式公式。至此，正向 SDE 与转移核（B.1）、得分与训练目标（B.2–B.3）、权重映射（B.4）与最优预处理（本节）五个环节，都被统一在同一组 $(s(t), \sigma(t))$ 参数化之下——附录 7B 遗留的"表面差异"被彻底消解。