# 附录16D PET与发射断层成像

> 定位：PET是CT的重要变体（发射而非透射），但与主线叙事关系较远，放入附录。PET的重建方法与CT类似，但其正模型的非线性特性带来了额外挑战。

## D.1 PET成像原理

正电子发射断层成像（Positron Emission Tomography, PET）的工作原理：

1. **放射性示踪剂注入**：将含有正电子发射核素（如${}^{18}$F-FDG）的示踪剂注入体内
2. **正电子湮灭**：正电子与体内电子湮灭，产生两个方向相反的511keV光子
3. **符合探测**：两个相对的探测器同时探测到光子，确定一条"响应线"（line of response）
4. **投影数据**：每条响应线上的计数正比于沿线的放射性活度积分

与CT的区别：
- CT：X射线源在外部→测量衰减系数（透射成像）
- PET：放射源在内部→测量放射性活度分布（发射成像）

## D.2 发射断层的正模型

PET的正模型比CT更复杂，因为发射的光子在体内也会被衰减。

**简化模型**（忽略衰减和散射）：

$$y_i \sim \text{Poisson}\left(\sum_j A_{ij} x_j + b_i\right)$$

其中$A$是系统矩阵（类似CT的Radon矩阵），$x$是放射性活度分布，$b_i$是背景计数。

**完整模型**（包含衰减）：

$$y_i \sim \text{Poisson}\left(\sum_j A_{ij} e^{-\sum_k l_{ik} \mu_k} x_j + b_i\right)$$

其中$\mu_k$是衰减系数，$l_{ik}$是射线$i$穿过像素$k$的长度，指数项是衰减校正。**注意**：衰减项使得正模型变为非线性的。

## D.3 PET重建方法

### 经典方法：MLEM

最大似然期望最大化（Maximum Likelihood Expectation Maximization, MLEM）是PET重建的经典方法：

$$x_j^{(k+1)} = \frac{x_j^{(k)}}{\sum_i A_{ij}} \sum_i \frac{A_{ij} y_i}{\sum_{j'} A_{ij'} x_{j'}^{(k)} + b_i}$$

MLEM收敛到最大似然解，但收敛速度慢（通常需要数十到数百次迭代）。

### 加速方法：OSEM

有序子集期望最大化（Ordered Subset EM, OSEM）将投影数据分为$S$个子集，每个子集做一次MLEM更新：

$$x_j^{(s+1)} = \frac{x_j^{(s)}}{\sum_{i \in \text{subset}} A_{ij}} \sum_{i \in \text{subset}} \frac{A_{ij} y_i}{\sum_{j'} A_{ij'} x_{j'}^{(s)} + b_i}$$

一次完整迭代包含$S$次子集更新，收敛速度约加速$S$倍。

### PGET：被动伽马发射断层成像

被动伽马发射断层成像（Passive Gamma Emission Tomography, PGET）用于核燃料检测：

- 目标：检测核燃料棒的内部结构（无损检测）
- 正模型：非线性（包含衰减和几何校正）
- 重建方法：Levenberg-Marquardt迭代 + Tikhonov/TV正则化
- IAEA PGET Challenge：银牌方案展示了正则化方法的有效性

## D.4 PET与扩散先验

PET的Poisson似然使得扩散先验方法的应用比CT/MRI更复杂——需要将Poisson似然得分$\nabla\log p(y|x_t)$近似到扩散采样框架中。这是当前的研究前沿之一。

**来源**：Siltanen Day3B P1-54（PET/PGET完整材料）；Shepp & Vardi (1982) MLEM原始论文；Hudson & Larkin (1994) OSEM
