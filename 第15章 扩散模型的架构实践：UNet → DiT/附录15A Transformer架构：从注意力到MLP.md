# 附录15A Transformer 架构：从注意力到 MLP

> **定位**：15.3.1 节我们讲了自注意力的直觉，但 Transformer Block 的完整零件（LayerNorm、残差连接、MLP、位置编码）对看懂 DiT 至关重要。这份附录把零件拆开摆齐，供想深入的人备查。

## 多头注意力的详细计算流程

输入序列 $X\in\mathbb{R}^{n\times d}$（$n$ 个 token，每个 $d$ 维）。多头注意力（Multi-Head Attention，白话：并行跑多个“注意力头”，各看不同子空间再拼起来）分三步：

**第一步**：线性投影出查询、键、值（每个头一套权重）
$$Q^i = X W_Q^i,\quad K^i = X W_K^i,\quad V^i = X W_V^i,\quad i=1,\ldots,h$$
其中 $W_Q^i,W_K^i\in\mathbb{R}^{d\times d_k}$，$W_V^i\in\mathbb{R}^{d\times d_v}$，$d_k=d_v=d/h$。

**第二步**：每个头独立算缩放点积注意力
$$\text{head}_i = \text{softmax}\left(\frac{Q^i (K^i)^T}{\sqrt{d_k}}\right) V^i$$
注意力矩阵 $A^i=\text{softmax}(Q^i(K^i)^T/\sqrt{d_k})\in\mathbb{R}^{n\times n}$ 表示第 $i$ 个头里每对 token 的注意力权重。

**第三步**：拼接所有头再线性投影
$$\text{MHA}(X) = [\text{head}_1;\ldots;\text{head}_h]\,W_O,\quad W_O\in\mathbb{R}^{hd_v\times d}$$

**计算复杂度**：$O(n^2 d + nd^2)$——$n^2 d$ 来自注意力矩阵，$nd^2$ 来自线性投影。当 $n>d$ 时注意力是瓶颈，$n<d$ 时线性投影是瓶颈。这正是高分辨率图像上自注意力贵的原因。

## 前馈网络（MLP）

Transformer Block 里夹在注意力之间的前馈网络是个两层全连接：
$$\text{MLP}(x) = W_2\cdot\text{GELU}(W_1 x + b_1) + b_2$$
- $W_1\in\mathbb{R}^{d_{\text{ff}}\times d}$，$W_2\in\mathbb{R}^{d\times d_{\text{ff}}}$；
- 隐层维度 $d_{\text{ff}}=4d$（标准）；
- GELU 激活：$\text{GELU}(x)=x\cdot\Phi(x)$，$\Phi$ 是标准正态分布的 CDF（累积分布函数）。

**GELU vs ReLU**：GELU 在 0 附近平滑过渡（ReLU 有不可导点）；GELU 对负值不完全截断（小负值仍有输出），保留更多信息；Transformer 里默认用 GELU。

## 残差连接与 LayerNorm：Pre-LN vs Post-LN

Transformer Block 的残差（residual，白话：把子层输出加回原输入，让信息有“高速公路”）和归一化（LayerNorm，白话：把一层特征归一化成零均值单位方差）有两种常见顺序：

**Post-LN**（原始 Transformer）：
$$h' = \text{LN}(h + \text{MHA}(h)),\quad h'' = \text{LN}(h' + \text{MLP}(h'))$$

**Pre-LN**（GPT-2 及以后默认）：
$$h' = h + \text{MHA}(\text{LN}(h)),\quad h'' = h' + \text{MLP}(\text{LN}(h'))$$

**对比**：Pre-LN 更稳（归一化在子层前，喂给 MHA/MLP 的特征始终归一化）；Post-LN 训得好时可能更快，但需更小心 warmup。DiT 选 Pre-LN——对要注入时间步条件的扩散去噪器，稳定性更重要。

## 可学习位置编码 vs 正弦位置编码

**可学习位置编码**（ViT）：把 $E_{\text{pos}}\in\mathbb{R}^{N\times d}$ 当可训练参数。优点能学任务特化的位置表示；缺点不能泛化到没见过的序列长度。

**正弦位置编码**（原始 Transformer）：固定的 $\sin/\cos$，不可学。优点理论上能外推到任意长度；缺点不是任务特化的。

**DiT 的选择**：可学习位置编码——扩散去噪器在固定分辨率上训和推，不需要外推，可学习更灵活。

## Transformer Block 的完整数据流

```
输入 h ∈ R^{n×d}
      │
      ├──────→ + ←─────────────────────┐
      │        │                        │
      │   LayerNorm                     │
      │        │                        │
      │   MHA(Q, K, V)                  │
      │        │                        │
      │        ├──────→ * α₁ ←────────┐ │
      │        │           ↑           │ │
      │        │        adaLN-Zero     │ │
      │        │           ↑           │ │
      │        │     条件嵌入 e(t)+e(c) │ │
      │        │                       │ │
      ├──────→ + ←────────────────────┘ │
      │        │                        │
      │   LayerNorm                     │
      │        │                        │
      │   MLP                            │
      │        │                        │
      │        ├──────→ * α₂ ←────────┐ │
      │        │           ↑           │ │
      │        │        adaLN-Zero     │ │
      │        │           ↑           │ │
      │        │     条件嵌入 e(t)+e(c) │ │
      │        │                       │ │
      ├──────→ + ←────────────────────┘ │
      │                                  │
      ▼                                  │
    输出 h'' ∈ R^{n×d}                   │
```

**来源**：Vaswani et al. (2017) "Attention Is All You Need"; Dosovitskiy et al. (2021) "An Image Is Worth 16x16 Words"; Peebles & Xie (2023) "Scalable Diffusion Models with Transformers"
