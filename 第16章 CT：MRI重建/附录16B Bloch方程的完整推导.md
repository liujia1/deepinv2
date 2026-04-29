# 附录16B Bloch方程的完整推导

> 定位：16.3.1节简述了Bloch方程和MRI信号生成过程，本附录提供Bloch方程的完整推导和稳态信号的解析解。

## B.1 Bloch方程的矩阵形式

磁化矢量$\mathbf{m} = (m_x, m_y, m_z)$在静磁场$\mathbf{B}_0 = (0, 0, B_0)$中的动力学由Bloch方程描述：

$$\frac{d\mathbf{m}}{dt} = \gamma \mathbf{m} \times \mathbf{B} - \begin{pmatrix} m_x/T_2 \\ m_y/T_2 \\ (m_z - m_0)/T_1 \end{pmatrix}$$

可以写成紧凑的矩阵形式：

$$\frac{d\mathbf{m}}{dt} = A(t)\mathbf{m} + \mathbf{d}$$

其中：

$$A(t) = \begin{pmatrix} -1/T_2 & \gamma B_z & -\gamma B_y \\ -\gamma B_z & -1/T_2 & \gamma B_x \\ \gamma B_y & -\gamma B_x & -1/T_1 \end{pmatrix}, \quad \mathbf{d} = \begin{pmatrix} 0 \\ 0 \\ m_0/T_1 \end{pmatrix}$$

## B.2 旋转部分的解析解

在没有弛豫项的情况下（$T_1, T_2 \to \infty$），磁化矢量绕$z$轴做Larmor进动：

$$\mathbf{m}(t) = R_z(\omega_0 t) \mathbf{m}(0)$$

其中$\omega_0 = \gamma B_0$是Larmor频率，$R_z(\phi)$是绕$z$轴旋转$\phi$角的旋转矩阵：

$$R_z(\phi) = \begin{pmatrix} \cos\phi & -\sin\phi & 0 \\ \sin\phi & \cos\phi & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

## B.3 弛豫部分的解

考虑横向弛豫（$T_2$效应）和纵向恢复（$T_1$效应）：

**横向弛豫**：

$$m_x(t) = m_x(0) e^{-t/T_2}, \quad m_y(t) = m_y(0) e^{-t/T_2}$$

**纵向恢复**：

$$m_z(t) = m_0 + (m_z(0) - m_0) e^{-t/T_1}$$

关键观察：$T_1$恢复是向平衡值$m_0$的指数趋近，$T_2$衰减是向零的指数衰减。对于生物组织，通常$T_1 > T_2$（例如脑组织$T_1 \sim 600\text{ms}$，$T_2 \sim 100\text{ms}$）。

## B.4 稳态信号

在稳态（$\frac{d\mathbf{m}}{dt} = 0$）下求解Bloch方程，设横向场$B_y$为RF激发产生的场：

$$m_{xy}^{ss} = \frac{T_2 \gamma B_y}{T_1 T_2 (\gamma B_y)^2 + 1} \cdot m_0$$

这个稳态信号的大小取决于$B_y$——通过调节$B_y$的大小，可以改变不同组织之间的信号对比度，这就是MRI对比度机制的数学基础。

## B.5 从Bloch方程到信号方程

在梯度场$B_z = \mathbf{g}(t) \cdot \mathbf{r}$的作用下，不同位置的质子以不同频率进动。横向磁化为：

$$m_{xy}(t, \mathbf{r}) = u(\mathbf{r}) \exp\left(-i\gamma \int_0^t \mathbf{g}(\tau) \cdot \mathbf{r} \, d\tau\right)$$

测量信号为所有位置的横向磁化的体积分（Faraday感应定律）：

$$s(t) = \int m_{xy}(t, \mathbf{r}) \, d\mathbf{r} = \int u(\mathbf{r}) \exp\left(-i\gamma \int_0^t \mathbf{g}(\tau) \cdot \mathbf{r} \, d\tau\right) d\mathbf{r}$$

定义k-space变量$\mathbf{k}(t) = \frac{\gamma}{2\pi} \int_0^t \mathbf{g}(\tau) d\tau$，则：

$$s(\mathbf{k}) = \int u(\mathbf{r}) e^{-2\pi i \mathbf{k} \cdot \mathbf{r}} d\mathbf{r} = \hat{u}(\mathbf{k})$$

这就得到了16.3.2节的关键结论：**MRI信号是图像的傅里叶变换**。$\blacksquare$

**来源**：IP_and_Im_Lectures-master magnetic_resonance_imaging.md L42-297；Bloch (1946) Nuclear Induction
