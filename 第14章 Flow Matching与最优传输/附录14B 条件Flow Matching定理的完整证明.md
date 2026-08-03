# 附录14B 条件Flow Matching定理的完整证明

> **定位**：CFM定理是Flow Matching的核心理论结果，证明需要仔细的积分运算，放入附录保持正文简洁。如果你接受"梯度等价"这个结论、不想看积分，可以直接跳到末尾"与DSM定理的类比"——那里用一张表点明它和第6章的同构关系。

## 定理陈述

**条件Flow Matching定理**（Lipman et al., 2023, Theorem 1 & 2）：设 $v_t(x|x_0)$ 生成条件概率路径 $p_t(x|x_0)$，$v_t^*(x)$ 是对应边际向量场。定义两个损失：

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, p_t(x)}\left[\|v_\theta(x, t) - v_t^*(x)\|^2\right]$$

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\, p_{\text{data}}(x_0),\, p_t(x|x_0)}\left[\|v_\theta(x, t) - v_t(x|x_0)\|^2\right]$$

则：

$$\boxed{\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)}$$

**白话翻译**：虽然 FM 损失（回归边际向量场）不可算，CFM 损失（回归条件向量场）可算，但两者关于网络参数 $\theta$ 的梯度完全一样。所以你只优化 CFM，就等于在优化 FM——这正是"无需仿真即可训练"的数学根基。

## 证明

### 第一步：展开 FM 损失的梯度

$$\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathbb{E}_{t, p_t(x)}\left[\|v_\theta - v_t^*\|^2\right]$$

$$= 2\,\mathbb{E}_{t, p_t(x)}\left[(v_\theta - v_t^*)\,\nabla_\theta v_\theta\right] = 2\,\mathbb{E}_{t, p_t(x)}\left[v_\theta\,\nabla_\theta v_\theta\right] - 2\,\mathbb{E}_{t, p_t(x)}\left[v_t^*\,\nabla_\theta v_\theta\right]$$

### 第二步：展开 CFM 损失的梯度

$$\nabla_\theta \mathcal{L}_{\text{CFM}}(\theta) = \nabla_\theta \mathbb{E}_{t, q(x_0), p_t(x|x_0)}\left[\|v_\theta - v_t(\cdot|x_0)\|^2\right]$$

$$= 2\,\mathbb{E}_{t, q(x_0), p_t(x|x_0)}\left[(v_\theta - v_t(\cdot|x_0))\,\nabla_\theta v_\theta\right]$$

$$= 2\,\mathbb{E}_{t, q(x_0), p_t(x|x_0)}\left[v_\theta\,\nabla_\theta v_\theta\right] - 2\,\mathbb{E}_{t, q(x_0), p_t(x|x_0)}\left[v_t(\cdot|x_0)\,\nabla_\theta v_\theta\right]$$

### 第三步：比较——关键在第二项

两个梯度第一项相同（都含 $v_\theta\nabla_\theta v_\theta$），只需看第二项：
- FM 第二项：$-2\,\mathbb{E}_{t, p_t(x)}\left[v_t^*(x)\,\nabla_\theta v_\theta(x,t)\right] = -2\,\mathbb{E}_{t, p_t(x)}\left[\mathbb{E}_{x_0|x}[v_t(x|x_0)]\,\nabla_\theta v_\theta(x,t)\right]$
- CFM 第二项：$-2\,\mathbb{E}_{t, p_{\text{data}}(x_0), p_t(x|x_0)}\left[v_t(x|x_0)\,\nabla_\theta v_\theta(x,t)\right]$

### 第四步：两项相等（塔性质 / law of total expectation）

$$\mathbb{E}_{t, p_t(x)}\left[\mathbb{E}_{x_0|x}[v_t(x|x_0)]\,\nabla_\theta v_\theta(x, t)\right] = \mathbb{E}_{t, p_{\text{data}}(x_0), p_t(x|x_0)}\left[v_t(x|x_0)\,\nabla_\theta v_\theta(x, t)\right]$$

展开看就清楚了：

$$\int p_t(x)\,\mathbb{E}_{x_0|x}[v_t(x|x_0)]\,\nabla_\theta v_\theta(x,t)\,dx = \int\!\left(\int p_t(x|x_0)p_{\text{data}}(x_0)v_t(x|x_0)dx_0\right)\nabla_\theta v_\theta(x,t)\,dx$$

$$= \int p_{\text{data}}(x_0)\,p_t(x|x_0)\,v_t(x|x_0)\,\nabla_\theta v_\theta(x,t)\,dx\,dx_0 = \mathbb{E}_{p_{\text{data}}(x_0), p_t(x|x_0)}\left[v_t(x|x_0)\,\nabla_\theta v_\theta(x,t)\right]$$

### 第五步：结论

两项分别相等，故 $\nabla_\theta \mathcal{L}_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{CFM}}$。$\blacksquare$

## 与 DSM 定理的类比（因果：同一套"条件化"把戏）

| 步骤 | DSM定理（第6章） | CFM定理（本章） |
|---|---|---|
| 不可计算目标 | $\mathcal{L}_{\text{ESM}}$（显式得分匹配） | $\mathcal{L}_{\text{FM}}$（Flow Matching） |
| 可计算替代 | $\mathcal{L}_{\text{DSM}}$（去噪得分匹配） | $\mathcal{L}_{\text{CFM}}$（条件FM） |
| 不可计算项 | $\nabla\log p_t(x)$（真实得分） | $v_t^*(x)$（边际向量场） |
| 可计算项 | $\nabla\log p_t(x\|x_0)$（条件得分） | $v_t(x\|x_0)$（条件向量场） |
| 消去机制 | 分部积分 + 塔性质 | 塔性质 |
| 结论 | 梯度相等 | 梯度相等 |

两个证明核心逻辑一致——通过"条件化"把不可算期望变成可算条件期望，靠塔性质证明梯度等价。这正是生成模型的统一模式：**训练时条件化，推理时边际化**。

## 更一般的 CFM 定理

上述定理可推广到更一般的条件变量 $z$（不限于 $x_0$）。设 $q(z)$ 是条件变量分布，$p_t(x|z)$ 是条件路径，$v_t(x|z)$ 是条件向量场，则 $\nabla_\theta\mathcal{L}_{\text{FM}}=\nabla_\theta\mathcal{L}_{\text{CFM}}$ 仍成立。这允许：
- $z=x_0$：标准 CFM；
- $z=(x_0,x_1)$：双向条件 FM（用于 OT-CFM）；
- $z=y$（标签/文本）：条件生成 FM（文生图等）。

**来源**：Lipman et al. (2023) "Flow Matching for Generative Modeling" Theorem 1-2 & Appendix B; Tong et al. (2024) "Conditional Flow Matching" Theorem 2.2
