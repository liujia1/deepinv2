# 附录16A Radon变换与Fourier切片定理的严格证明

> 定位：16.1.3节给出了Fourier切片定理的陈述和证明思路，本附录提供完整严格的证明以及FBP公式的推导。

## A.1 Radon变换的函数空间性质

**定义**：设$u \in L^2(\Omega)$，$\Omega = \{x \in \mathbb{R}^2 : |x| \leq 1\}$，Radon变换定义为：

$$\mathcal{R}u(\theta, s) = \int_{\ell(\theta,s)} u(x) \, dx = \int_{-\infty}^{\infty} u(s\cos\theta - t\sin\theta, \; s\sin\theta + t\cos\theta) \, dt$$

其中$\theta \in [0, \pi)$，$s \in [-1, 1]$。

**命题**：$\mathcal{R}: L^2(\Omega) \to L^2(S^1 \times [-1,1])$是有界线性算子。

证明概要：由Schwarz不等式，$|\mathcal{R}u(\theta,s)|^2 \leq \int_{\ell(\theta,s)} |u(x)|^2 dx \cdot |\ell(\theta,s) \cap \Omega|$。对$(\theta, s)$积分即得有界性。$\blacksquare$

## A.2 Fourier切片定理的完整证明

**定理（Fourier切片定理）**：设$u \in L^1(\mathbb{R}^2) \cap L^2(\mathbb{R}^2)$，则

$$\widehat{\mathcal{R}_\theta u}(\omega) = \hat{u}(\omega\cos\theta, \; \omega\sin\theta)$$

其中$\widehat{\mathcal{R}_\theta u}(\omega) = \int_{-\infty}^{\infty} \mathcal{R}_\theta u(s) e^{-2\pi i \omega s} ds$是1D傅里叶变换，$\hat{u}(k_1, k_2) = \int_{\mathbb{R}^2} u(x_1, x_2) e^{-2\pi i (x_1 k_1 + x_2 k_2)} dx_1 dx_2$是2D傅里叶变换。

**证明**：

$$\widehat{\mathcal{R}_\theta u}(\omega) = \int_{-\infty}^{\infty} \mathcal{R}_\theta u(s) \, e^{-2\pi i \omega s} \, ds$$

展开Radon变换的定义：

$$= \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} u(s\cos\theta - t\sin\theta, \; s\sin\theta + t\cos\theta) \, dt \; e^{-2\pi i \omega s} \, ds$$

做变量替换：令$x_1 = s\cos\theta - t\sin\theta$，$x_2 = s\sin\theta + t\cos\theta$。这是一个旋转变换，Jacobian行列式为1，且$s = x_1\cos\theta + x_2\sin\theta$。

$$= \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} u(x_1, x_2) \, e^{-2\pi i \omega(x_1\cos\theta + x_2\sin\theta)} \, dx_1 dx_2$$

令$k_1 = \omega\cos\theta$，$k_2 = \omega\sin\theta$：

$$= \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} u(x_1, x_2) \, e^{-2\pi i (x_1 k_1 + x_2 k_2)} \, dx_1 dx_2 = \hat{u}(k_1, k_2) = \hat{u}(\omega\cos\theta, \omega\sin\theta) \qquad \blacksquare$$

## A.3 滤波反投影（FBP）公式的推导

**目标**：从Fourier切片定理推导FBP公式$u = \mathcal{R}^* \mathcal{F}^{-1}[|\omega| \cdot \widehat{\mathcal{R}_\theta u}]$。

**推导**：

由2D逆傅里叶变换，在极坐标$(r, \theta)$下表示频率变量$\mathbf{k} = (k_1, k_2) = (\omega\cos\theta, \omega\sin\theta)$：

$$u(x) = \int_{\mathbb{R}^2} \hat{u}(\mathbf{k}) e^{2\pi i \mathbf{k} \cdot \mathbf{x}} d\mathbf{k} = \int_0^{\pi} \int_{-\infty}^{\infty} \hat{u}(\omega\cos\theta, \omega\sin\theta) \, e^{2\pi i \omega(x_1\cos\theta + x_2\sin\theta)} \, |\omega| \, d\omega \, d\theta$$

注意极坐标变换的Jacobian为$|\omega|$（不是$\omega$，因为$\omega$可取负值）。

由Fourier切片定理，$\hat{u}(\omega\cos\theta, \omega\sin\theta) = \widehat{\mathcal{R}_\theta u}(\omega)$，代入得：

$$u(x) = \int_0^{\pi} \left[\int_{-\infty}^{\infty} |\omega| \cdot \widehat{\mathcal{R}_\theta u}(\omega) \, e^{2\pi i \omega s} d\omega\right]_{s = x_1\cos\theta + x_2\sin\theta} d\theta$$

内层积分恰好是$\mathcal{F}^{-1}[|\omega| \cdot \widehat{\mathcal{R}_\theta u}](s)$，即斜坡滤波后的投影。外层积分是反投影。因此：

$$\boxed{u(x) = \int_0^{\pi} \mathcal{F}^{-1}\left[|\omega| \cdot \widehat{\mathcal{R}_\theta u}(\omega)\right](x_1\cos\theta + x_2\sin\theta) \, d\theta = \mathcal{R}^* \mathcal{F}^{-1}[|\omega| \cdot \widehat{\mathcal{R}_\theta u}]}$$

这就是FBP公式。$\blacksquare$

**来源**：IP_and_Im_Lectures-master tomography.md L177-296；Siltanen Day3A P51（Fourier切片定理证明）；Natterer (1986) The Mathematics of Computerized Tomography
