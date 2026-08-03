# 附录7B VE-SDE与VP-SDE的推理等价性

> 定位：为7.2节的VE-SDE与VP-SDE对比提供补充材料。Kawar et al. 的观察表明，两种SDE在推理（采样）阶段是等价的，差异主要在训练阶段。Karras et al. (2022) 的统一框架进一步消除了两种SDE的表面差异。

## 一个"长得不一样但其实是一回事"的故事

7.2节我们看到 VE-SDE（方差爆炸，不缩放信号）和 VP-SDE（方差保留，缩放信号）的公式差很多。直觉上你会以为它们是两种本质不同的加噪方式。这个附录要告诉你：**在"从噪声生成数据"这件事上，两者是等价的**——只要你把一个做一下重缩放，它就变成另一个。差别主要藏在"怎么训练"里，而不在"怎么采样"里。

## 等价性的直观解释：重缩放一下就一样了

**VE-SDE**：$x_t = x_0 + \sigma(t)\,\epsilon$

含噪信号 = 原始信号 + 加性噪声。信号不缩放，噪声方差 $\sigma(t)^2$ 随时间增长。

**VP-SDE**：$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$

含噪信号 = 缩放信号 + 加性噪声。信号被 $\sqrt{\bar\alpha_t}$ 缩放，总方差保持有界。

**关键观察**：对 VP-SDE 的含噪信号做重缩放。令 $\hat{x}_t = x_t / \sqrt{\bar\alpha_t}$：

$$\hat{x}_t = x_0 + \hat{\sigma}(t)\,\epsilon$$

其中 $\hat{\sigma}(t) = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$。重缩放后的 VP-SDE 在形式上**完全等同于 VE-SDE**——都是纯加性噪声模型。只是"噪声尺度"换了个名字。

## 推理等价性的形式化

推理等价性意味着：给定相同的得分函数估计器，VE-SDE 采样和 VP-SDE 采样产生的样本分布相同（在重缩放意义下）。

具体来说，设 VE-SDE 的得分网络为 $s_\theta^{(\text{VE})}(x, t)$，VP-SDE 的得分网络为 $s_\theta^{(\text{VP})}(x, t)$。若它们在各自框架下被正确训练，则：

- VE-SDE 的逆向采样：从 $\mathcal{N}(0, \sigma_{\max}^2 I)$ 出发，解逆向 VE-SDE；
- VP-SDE 的逆向采样：从 $\mathcal{N}(0, I)$ 出发，解逆向 VP-SDE。

两者的生成质量（FID、IS 等指标）在统计意义上相当——差异不显著。

## 等价性的含义

### 训练阶段的差异（这才是真区别）

尽管推理等价，两种 SDE 在**训练**阶段有显著差异：

- **VP-SDE 训练更稳定**：信号缩放 $\sqrt{\bar\alpha_t}$ 让含噪信号 $x_t$ 的数值范围在不同时间步大致一致，利于梯度稳定；
- **VE-SDE 训练需多注意**：大方差时 $x_t$ 数值范围很大，需适当加权函数 $\lambda(t)=\sigma(t)^2$ 来平衡不同时间步的损失。

### 实践中的选择

- **图像生成**：VP-SDE 更常用（余弦调度稳定性优势）；
- **逆问题求解**：VE-SDE 更常用（加性噪声模型与逆问题观测模型更一致，便于似然计算）；
- **统一框架**：Karras et al. (2022) 的 EDM 框架消除了选择的必要性——在统一框架下设计 $s(t)$ 和 $\sigma(t)$。

## Karras et al. (2022) 的统一框架：把两种"方言"并成一门口语

Karras et al. 在"Elucidating the Design Space of Diffusion-Based Generative Models"中提出统一框架，把 VE-SDE 和 VP-SDE 视为同一框架的两种参数化。

统一 SDE 引入两个设计选择：

- **信号缩放** $s(t)$：控制信号的幅度；
- **噪声调度** $\sigma(t)$：控制噪声的方差。

统一正向 SDE：

$$dx = \frac{\dot{s}(t)}{s(t)}x\,dt + s(t)\sqrt{2\dot{\sigma}(t)\sigma(t)}\,dw$$

转移核：

$$p_{0t}(x_t|x_0) = \mathcal{N}(x_t | s(t)x_0, s(t)^2\sigma(t)^2 I)$$

- **VE-SDE**：$s(t) \equiv 1$（无缩放），$\sigma(t)$ 自由选择；
- **VP-SDE**：$s(t) = e^{-\frac{1}{2}\int_0^t\beta(\tau)\,d\tau}$，$\sigma(t) = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}/s(t)$。

Karras et al. 进一步推荐了一种"最优"设计：

- $s(t) = 1$（无缩放，避免信号幅度变化）；
- $\sigma(t)$ 选使 $\dot{\sigma}/\sigma$ 近似常数（几何调度）；
- 去噪器输出 $D_\theta(x_t, \sigma(t)) = c_{\text{skip}}x_t + c_{\text{out}}F_\theta(c_{\text{in}}x_t, c_{\text{noise}}\sigma(t))$

其中 $c_{\text{skip}}, c_{\text{out}}, c_{\text{in}}, c_{\text{noise}}$ 是精心设计的系数，保证训练目标的单位方差。

**核心洞见**：VE-SDE 和 VP-SDE 不是根本不同的方法——它们只是信号缩放和噪声调度的不同选择。统一框架消除了表面差异，让设计者专注于真正重要的选择：**信噪比曲线的形状**。

**来源**：Kawar et al. (2022); Karras et al. (2022); Song et al. (2021)
