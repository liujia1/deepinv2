# 附录15A Transformer架构：从注意力到MLP

> **定位**：15.3.1节简述了自注意力机制，但Transformer Block的完整结构（LayerNorm、残差连接、MLP、位置编码）对理解DiT至关重要。本附录提供完整的架构细节，供需要深入理解的读者参考。

## 多头注意力的详细计算流程

给定输入序列 $X \in \mathbb{R}^{n \times d}$（$n$ 个token，每个 $d$ 维），多头注意力的计算步骤如下：

**第一步**：线性投影生成查询、键、值

$$Q^i = X W_Q^i, \quad K^i = X W_K^i, \quad V^i = X W_V^i, \quad i = 1, \ldots, h$$

其中 $W_Q^i, W_K^i \in \mathbb{R}^{d \times d_k}$，$W_V^i \in \mathbb{R}^{d \times d_v}$，$d_k = d_v = d/h$。

**第二步**：每个头独立计算缩放点积注意力

$$\text{head}_i = \text{softmax}\left(\frac{Q^i (K^i)^T}{\sqrt{d_k}}\right) V^i$$

注意力矩阵 $A^i = \text{softmax}(Q^i (K^i)^T / \sqrt{d_k}) \in \mathbb{R}^{n \times n}$ 表示第 $i$ 个头中每对token之间的注意力权重。

**第三步**：拼接所有头的输出并线性投影

$$\text{MHA}(X) = [\text{head}_1; \ldots; \text{head}_h] W_O$$

其中 $W_O \in \mathbb{R}^{hd_v \times d}$，$[\cdot;\cdot]$ 表示拼接。

**计算复杂度**：$O(n^2 d + nd^2)$——$n^2 d$ 来自注意力矩阵的计算，$nd^2$ 来自线性投影。当 $n > d$ 时，注意力计算成为瓶颈；当 $n < d$ 时，线性投影成为瓶颈。

## 前馈网络（MLP）

Transformer Block中的前馈网络是一个两层全连接网络：

$$\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 \cdot x + b_1) + b_2$$

- $W_1 \in \mathbb{R}^{d_{\text{ff}} \times d}$，$W_2 \in \mathbb{R}^{d \times d_{\text{ff}}}$
- 隐层维度 $d_{\text{ff}} = 4d$（标准设置）
- GELU激活：$\text{GELU}(x) = x \cdot \Phi(x)$，$\Phi$ 为标准正态分布的CDF

**GELU vs ReLU**：

- GELU在 $x = 0$ 附近平滑过渡，ReLU有不可导点
- GELU对负值不完全截断（小的负值仍有非零输出），保留了更多信息
- 实践中，GELU在Transformer中表现优于ReLU，是默认选择

## 残差连接与LayerNorm：Pre-LN vs Post-LN

Transformer Block的残差连接和归一化有两种常见顺序：

**Post-LN**（原始Transformer，Vaswani et al., 2017）：

$$h' = \text{LN}(h + \text{MHA}(h))$$
$$h'' = \text{LN}(h' + \text{MLP}(h'))$$

**Pre-LN**（GPT-2及后续工作）：

$$h' = h + \text{MHA}(\text{LN}(h))$$
$$h'' = h' + \text{MLP}(\text{LN}(h'))$$

**Pre-LN vs Post-LN**：

- **训练稳定性**：Pre-LN更稳定——归一化在子层之前执行，输入到MHA/MLP的特征始终是归一化的；Post-LN中归一化在残差连接之后，梯度需要穿过归一化层，可能导致训练不稳定
- **收敛速度**：Post-LN在训练良好时可能收敛更快，但需要更仔细的warmup和学习率调整
- **DiT的选择**：Pre-LN——稳定性更重要，尤其是对于需要注入时间步条件的扩散去噪器

## 可学习位置编码 vs 正弦位置编码

**可学习位置编码**（ViT, Dosovitskiy et al., 2021）：

- 将位置编码 $E_{\text{pos}} \in \mathbb{R}^{N \times d}$ 作为可训练参数
- 优点：可以学习任务特定的位置表示
- 缺点：无法泛化到训练时未见过的序列长度

**正弦位置编码**（原始Transformer, Vaswani et al., 2017）：

- 固定的正弦/余弦函数，不可学习
- 优点：可以外推到任意序列长度（理论上）
- 缺点：不是任务特定的

**DiT的选择**：可学习位置编码——扩散去噪器在固定分辨率上训练和推理，不需要外推到新长度，可学习编码更灵活。

## Transformer Block的完整数据流

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
