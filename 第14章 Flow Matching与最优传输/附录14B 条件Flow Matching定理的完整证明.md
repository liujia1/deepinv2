# 附录14B 条件Flow Matching定理的完整证明

> **定位**：CFM定理是Flow Matching的核心理论结果，证明需要仔细的积分运算，放入附录保持正文简洁。

## 定理陈述

**条件Flow Matching定理**（Lipman et al., 2023, Theorem 1 & 2）：设 $v_t(x|x_0)$ 是生成条件概率路径 $p_t(x|x_0)$ 的向量场，$v_t^*(x)$ 是对应的边际向量场。定义两个损失函数：

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, p_t(x)}\left[\|v_\theta(x, t) - v_t^*(x)\|^2\right]$$

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\, p_{\text{data}}(x_0),\, p_t(x|x_0)}\left[\|v_\theta(x, t) - v_t(x|x_0)\|^2\right]$$

则：

$$\boxed{\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)}$$

## 证明

### 第一步：展开FM损失的梯度

$$\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathbb{E}_{t, p_t(x)}\left[\|v_\theta - v_t^*\|^2\right]$$

$$= 2\,\mathbb{E}_{t, p_t(x)}\left[(v_\theta - v_t^*)\,\nabla_\theta v_\theta\right]$$

$$= 2\,\mathbb{E}_{t, p_t(x)}\left[v_\theta\,\nabla_\theta v_\theta\right] - 2\,\mathbb{E}_{t, p_t(x)}\left[v_t^*\,\nabla_\theta v_\theta\right]$$

### 第二步：展开CFM损失的梯度

$$\nabla_\theta \mathcal{L}_{\text{CFM}}(\theta) = \nabla_\theta \mathbb{E}_{t, q(x_0), p_t(x|x_0)}\left[\|v_\theta - v_t(\cdot|x_0)\|^2\right]$$

$$= 2\,\mathbb{E}_{t, q(x_0), p_t(x|x_0)}\left[(v_\theta - v_t(\cdot|x_0))\,\nabla_\theta v_\theta\right]$$

$$= 2\,\mathbb{E}_{t, q(x_0), p_t(x|x_0)}\left[v_\theta\,\nabla_\theta v_\theta\right] - 2\,\mathbb{E}_{t, q(x_0), p_t(x|x_0)}\left[v_t(\cdot|x_0)\,\nabla_\theta v_\theta\right]$$

### 第三步：比较两个梯度

两个梯度的第一项相同（都与 $v_\theta$ 有关），关键在于第二项。

**FM梯度的第二项**：

$$-2\,\mathbb{E}_{t, p_t(x)}\left[v_t^*(x)\,\nabla_\theta v_\theta(x, t)\right]$$

$$= -2\,\mathbb{E}_{t, p_t(x)}\left[\mathbb{E}_{x_0|x}[v_t(x|x_0)]\,\nabla_\theta v_\theta(x, t)\right]$$

**CFM梯度的第二项**：

$$-2\,\mathbb{E}_{t, p_{\text{data}}(x_0), p_t(x|x_0)}\left[v_t(x|x_0)\,\nabla_\theta v_\theta(x, t)\right]$$

### 第四步：证明两个第二项相等

由条件期望的塔性质（tower property / law of total expectation）：

$$\mathbb{E}_{t, p_t(x)}\left[\mathbb{E}_{x_0|x}[v_t(x|x_0)]\,\nabla_\theta v_\theta(x, t)\right] = \mathbb{E}_{t, p_{\text{data}}(x_0), p_t(x|x_0)}\left[v_t(x|x_0)\,\nabla_\theta v_\theta(x, t)\right]$$

这是因为：左边先对 $x$ 积分再对 $x_0|x$ 取条件期望，等价于右边先对 $(x_0, x)$ 联合积分。具体地：

$$\mathbb{E}_{p_t(x)}\left[\mathbb{E}_{x_0|x}[v_t(x|x_0)]\,\nabla_\theta v_\theta(x, t)\right] = \int p_t(x)\,\mathbb{E}_{x_0|x}[v_t(x|x_0)]\,\nabla_\theta v_\theta(x, t)\,dx$$

$$= \int \left(\int p_t(x|x_0)\,p_{\text{data}}(x_0)\,v_t(x|x_0)\,dx_0\right) \nabla_\theta v_\theta(x, t)\,dx$$

$$= \int p_{\text{data}}(x_0)\,p_t(x|x_0)\,v_t(x|x_0)\,\nabla_\theta v_\theta(x, t)\,dx\,dx_0$$

$$= \mathbb{E}_{p_{\text{data}}(x_0), p_t(x|x_0)}\left[v_t(x|x_0)\,\nabla_\theta v_\theta(x, t)\right]$$

### 第五步：结论

两个梯度的第一项和第二项分别相等，因此：

$$\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)$$

$\blacksquare$

## 与DSM定理的类比

将CFM定理的证明与第6章的DSM定理证明对比：

| 步骤 | DSM定理 | CFM定理 |
|---|---|---|
| 不可计算目标 | $\mathcal{L}_{\text{ESM}}$（显式得分匹配） | $\mathcal{L}_{\text{FM}}$（Flow Matching） |
| 可计算替代 | $\mathcal{L}_{\text{DSM}}$（去噪得分匹配） | $\mathcal{L}_{\text{CFM}}$（条件FM） |
| 不可计算项 | $\nabla\log p_t(x)$（真实得分） | $v_t^*(x)$（边际向量场） |
| 可计算项 | $\nabla\log p_t(x\|x_0)$（条件得分） | $v_t(x\|x_0)$（条件向量场） |
| 消去机制 | 分部积分 + 塔性质 | 塔性质 |
| 结论 | $\nabla_\theta \mathcal{L}_{\text{ESM}} = \nabla_\theta \mathcal{L}_{\text{DSM}}$ | $\nabla_\theta \mathcal{L}_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{CFM}}$ |

两个证明的核心逻辑一致——通过"条件化"将不可计算的期望转化为可计算的条件期望，利用塔性质证明梯度等价。这正是生成模型中"条件化技巧"的统一模式：**在训练时条件化，在推理时边际化**。

## 更一般的CFM定理

上述定理可以推广到更一般的条件变量 $z$（不仅仅是 $x_0$）。设 $q(z)$ 是条件变量的分布，$p_t(x|z)$ 是条件概率路径，$v_t(x|z)$ 是条件向量场。定义：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(z), p_t(x|z)}\left[\|v_\theta(x, t) - v_t(x|z)\|^2\right]$$

则 $\nabla_\theta \mathcal{L}_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{CFM}}$ 仍然成立。这个推广允许：

- $z = x_0$：标准的CFM（条件化在数据点上）
- $z = (x_0, x_1)$：双向条件FM（同时条件化在源和目标上，用于OT-CFM）
- $z = y$（标签/文本）：条件生成FM（用于文本到图像等条件生成任务）

**来源**：Lipman et al. (2023) "Flow Matching for Generative Modeling" Theorem 1-2 & Appendix B; Tong et al. (2024) "Conditional Flow Matching" Theorem 2.2
