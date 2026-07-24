# Self-Supervised Learning for Imaging Inverse Problems

Julián Tachella, CNRS, École Normale Supérieure de Lyon 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/a285c6284adaa5e9b5efc4faeae9c5c5f7a5c032dcceddb9c450734700f84500.jpg)


## Linear Inverse Problems

Goal: recover signal ?? from ?? 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/dbec48ff0c3a81bd8ce2f62f34d62b84171d362ec2fffefec8446ef420f64fd9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/cae085712a148f861d7453d1edfe15e9183537f401711ec24f7678a0aa46e477.jpg)


## Examples

## Image denoising

• ?? = identity 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/7e83f47466f83ad5dc0686caa3f611370e6afca2dc06005f1939b3589442e82f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/6a242f457398f7cbcbd09f14f34a9b899c6bed76e4bc29c21d01236c8614c62f.jpg)



??


## Image inpainting

• ?? = diagonal matrix with 1’s and 0s. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/8ec9af993a6d3abe3f3d4da5644542344fb97ab1cf7b33783fc7ac9977784c3c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/2e476d383a5e33c46f5c96f6aaac3d6c6fce2dcb7cebc80222e6880d870d72ea.jpg)



??


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/502df2a61623fabb71b7e9fa434bf74262a064d2fb3a828c7926b671bd2c050a.jpg)



??


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/279c56c8543e2888d1a9e3a66c7436fc1a8ced7ecf6f26c1fae05971c9453b15.jpg)



??


Magnetic resonance imaging 

• ?? = subset of Fourier modes (?? − space) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/c8ba0d244598c6ffb9c99c31fbb5a2943abbf88a5b77f1e0f57971c67b201e77.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/54b5eb259bd9eebb5bb0b4b2d4fd7a96cdc235e8fc51b4f395e4c73de0b68c45.jpg)



??


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/0117c27183de99cc1ba197ab3c86f5fea2a53ffa18e93e0e6792c1fd7175261e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/c94ffa192d8be7614277c386cc244d853ffedecb8b164b18feabdf566ec22a04.jpg)



??


Computed tomography 

• ?? = 1D projections (sinograms) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/03aef4c412ad5de8dbb0d227475a5f4a8282176baee9b38b2cf78d286dd4ee4b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/71b7abdfcb1beb3d47b8a169356d4856361b765e21a30e295a89db09ad406d71.jpg)


?? 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/86ffd96632d86b07b07b8c77bb879d7bb90348e74fef7273f6745f84ad18e15d.jpg)



??


## Regularised reconstruction

Idea: define a regularisation $\rho ( x )$ promoting plausible reconstructions 

$$
\widehat {\pmb {x}} = \underset {\pmb {x}} {\mathrm{argmin}} \left| | \pmb {y} - A \pmb {x} | \right| ^ {2} + \rho (\pmb {x})
$$

Examples: total-variation, sparsity, etc. 

Disadvantages: hard to define a good $\rho ( x )$ in real world problems, loose with respect to the true signal distribution 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/3d21483c81cc17e576e1b0510f50d8923c95bc9093daf476fd61899b2893c18c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/4cee97be626e2f37ad84f7642d28506cb211a3530f46fd6f6c4dd5ab31666aad.jpg)


## Learning approach

Idea: use training pairs of signals and measurements to directly learn the inversion function 

$$
\underset {f} {\operatorname{argmin}} \mathbb {E} _ {x, y} \| x - f (y) \| ^ {2}
$$


supervised dataset


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/79968ff3c62325319bddff05bb9b3d9c460c0e9fad280801824529c302442c5d.jpg)



input


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/2e4640c18151ada0d8808ba90ddf7de3c54653700137d0b9983ecfb72c77827a.jpg)



target


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/ca2a4a262773c2b92e0bc76b2a5842e153902be9e82ba1fe6885358dda4ca74a.jpg)



input


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/2cbf296682a971dbdbbae041e3d4a1fd4aca8de452415368f74e9cefa8089f90.jpg)



target


## Learning approach

## Advantages:

• State-of-the-art reconstructions 

• Once trained, ?? is easy to evaluate 

## fastMRI

Accelerating MR Imaging with AI 

Ground-truth 

Total variation (28.2 dB) 

Deep network (34.5 dB) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/fe2ce63bdccdd7e79a694986327edc4e0ed00622fd09cc705dccef2b21e81c13.jpg)



x8 accelerated MRI [Zbontar et al., 2019]


## Learning approach

Main disadvantage: Obtaining training signals $x _ { i }$ can be expensive or impossible. 

• Medical and scientific imaging 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/6b0bccdaba500de78ccd3a9e0e056c04929b143d1b7ab320118f7365d57b018e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/019f703eabb71c8946baee266a1c67e41c3bc6a0a81232ebeac36e5da01121c7.jpg)


• Distribution shift [Belthangady & Royer, 2019] 

<table><tr><td>Training datasets</td><td>Witenagemot</td><td>Degraded image</td></tr><tr><td>abcdefghijklmnopqrstuvwxyz...</td><td>Witenagemot</td><td rowspan="3">Restored images</td></tr><tr><td>abcdefghijklmnopqrstuvwxyz...</td><td>Witynagemot</td></tr><tr><td>中文王国...</td><td>Witienagemot</td></tr><tr><td>a中b文c...</td><td>Witenagemot</td><td rowspan="2">Ground truth</td></tr><tr><td>Old english word</td><td>Witenagemot</td></tr></table>

## AI for Knowledge Discovery

## Black hole picture captured for first time in space breakthrough

## Guardian

DeepMind uncovers structure of 20om proteins in scientific leap forward 

## Outline

How can we learn $f$ from measurement $\{ { y } _ { i } \}$ data alone? 

1. Noisy: $y = x + \epsilon$ 

2. Incomplete and noisy: $y = A x + \epsilon$ 

Goal: build a self-supervised loss $\mathcal { L } _ { \mathrm { S E L F } }$ such that 

$$
\mathbb {E} _ {\boldsymbol {y}} \mathcal {L} _ {\mathrm{SELF}} (\boldsymbol {y}, f) = \mathbb {E} _ {\boldsymbol {x}, \boldsymbol {y}} \mathcal {L} _ {\mathrm{SUP}} (\boldsymbol {x}, \boldsymbol {y}, f) + \mathrm{const.}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/d2bccc0b631c47276c5520071d125acd57d5da40ec8189dc837d4ec4db2b1626.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/a81eb6ecfb2c88fecd80e26facffea935fb9ea39fdec0de5319a239b4fd9f8cf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/fdc956f0fb016e41379c7756cd047d617d9cb4af99a0af08673f6a63a8136699.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/e1d4f64c0166937aab9745db9cac4e7eb988dbe9ca6006cba66f6d3add402844.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/7767532bc20a6322f7f2453fd7a66e8fc8f64af49c6819c9484fb52c49ebbb8f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/61f761fa3937128965ca7ad630091bbf6972a35b8e40fde82ba93b88d9f4bc39.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/f8cda2ff5a617188055ea435d93fcc1bf46b60c319a58b2a1218461dfdb81222.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/2bd19d1472e6c231a07b361a2020a7e134a092621baba8c670282ada8e264a65.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/b0afd9b39162a11a4a575b8e37532e836f6fe1492c20fb809c358b5ae451ad4c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/5f771765ac0aadd9ca038021a824e988d085c697f6f7e9152dbbd1e5153db8e2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/64d44c531e4460ea9543e64aa63b3b59e657ff0626e38846b471624c453a1870.jpg)


## How can we learn ?? from measurement $\{ { y } _ { i } \}$ data alone?

## Example

• Cryo-Electron microscopy images 

• Extremely low SNR 

• Noise distribution is unknown 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/5c59c96fa65ab58076f41e3d227c0a7abedbcff7f060c9d50a98ede63918ca09.jpg)


## Self-Supervised Risk Estimators

Goal: build self-supervised loss $\mathcal { L } _ { \mathrm { S E L F } } |$ 

• Unbiased estimator: $\mathbb { E } _ { y } \mathcal { L } _ { \mathrm { S E L F } } ( y , f ) \propto \mathbb { E } _ { x , y } | | x - f ( y ) | | ^ { 2 } = \mathbb { E } \{ x | y \}$ 

• Same global minimum: argmin $\mathbb { E } _ { y } \mathcal { L } _ { \mathrm { S E L F } } ( y , f ) = \underset { f } { \mathrm { a r g m i n } } \mathbb { E } _ { x , y } | | x - f ( y ) | | ^ { 2 } = \mathbb { E } \{ x | y \}$ 

• Same minima constrained set: argmin ?? $\bar { f } \in \mathcal { F }$ $\mathbf { \ \ } _ { y } \mathcal { L } _ { \mathrm { S E L F } } ( \mathbf { \boldsymbol { y } } , f ) = \arg \operatorname* { m i n } _ { f \in \mathcal { F } } \mathbb { E } _ { x , y } | | \boldsymbol { x } - f ( \mathbf { \boldsymbol { y } } ) | | ^ { 2 } \neq \mathbb { E } \{ \boldsymbol { x } | \boldsymbol { y } \}$ 

What is the best we can do? MMSE estimator 

## Stein’s Unbiased Risk Estimator

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/144f90a937a88c5eadb1048ac19e74f3bca30a877076a39da17c732cd40936fc.jpg)



Measurement Degrees of freedom [Efron, 2004] consistency


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/c649e73db1f796f08aff0e28a89b5792dad53ef9e2a006bc92c644c2bcc096d6.jpg)


• Stein’s lemma: Let $y | x \sim \mathcal { N } \big ( x , I \sigma ^ { 2 } \big )$ , then $\mathbb { E } _ { y } \mathcal { L } _ { \mathrm { S U R E } } ( { \boldsymbol y } , { \boldsymbol f } ) \propto \mathbb { E } _ { x , y } | | { \boldsymbol x } - { \boldsymbol f } ( { \boldsymbol y } ) | | ^ { 2 }$ 

• Extensions for Poisson, Poisson-Gaussian [Hudson, 1978], 

• MMSE estimator $f ^ { * } ( y ) = \mathbb { E } \{ x | y \}$ 

## Efficient SURE

1) Monte Carlo SURE: approx. divergence as [Ramani et al., 2007] 

$$
\sum_ {i} \frac {\delta f _ {i}}{\delta y _ {i}} (\pmb {y}) \approx \frac {\pmb {\omega} ^ {\top}}{\alpha} \left(f (\pmb {y}) - f (\pmb {y} + \pmb {\omega} \alpha)\right)
$$

$$
\begin{array}{l} \boldsymbol {\omega} \sim \mathcal {N} (\mathbf {0}, I \sigma^ {2}) \\ \text {and} \alpha \in \mathbb {R} \end{array}
$$

2) Autodiff SURE: use auto-diff [Soltanayev, 2020] 

$$
\sum_ {i} \frac {\delta f _ {i}}{\delta y _ {i}} (\pmb {y}) \approx \pmb {\omega} ^ {\top} \left(\frac {\delta f}{\delta \pmb {y}} \pmb {\omega}\right)
$$

3) Recorrupted2Recorrupted [Pang et al., 2021], [Monroy, Bacca & T., CVPR 2025] 

$$
\mathcal {L} _ {\mathrm{R2R}} (\pmb {y}, f, \alpha) = \mathbb {E} _ {\pmb {\omega}} | | \pmb {y} + \alpha \pmb {\omega} - f (\pmb {y} - \pmb {\omega} / \alpha) | | ^ {2}
$$

## Tweedie’s Formula

The solution to SURE is Tweedie’s Formula 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/262fb05e858cddbf6611156c5de2220fc1640d00743ae15d5df54820d2ed6754.jpg)


$$
\begin{array}{r l} & {\underset {f} {\min} \mathbb {E} _ {y} | | y - f (y) | | ^ {2} + 2 \sigma^ {2} \sum_ {i} \frac {\delta f _ {i}}{\delta y _ {i}} (y)} \\ & {\underset {f} {\min} \mathbb {E} _ {y} | | y - f (y) | | ^ {2} - 2 \sigma^ {2} \sum_ {i} f _ {i} (y) \frac {\delta \log p _ {y} (y)}{\delta y _ {i}}} \\ & {\underset {f} {\min} \mathbb {E} _ {y} | | f (y) - y - \sigma^ {2} \nabla \log p _ {y} (y) | | ^ {2}} \\ & {\quad \Rightarrow f (y) = y + \sigma^ {2} \nabla \log p _ {y} (y)} \end{array}
$$

• Noise2Score [Kim and Ye, 2021] learns $\nabla \log p _ { y } ( y )$ from noisy data + denoises with Tweedie. 

• Key formula behind diffusion models, which can be trained self-supervised [Daras et al., 2024] 

## Cross-Validation Methods

What happens if we only know that $p ( \pmb { y } | \pmb { x } ) = \prod p ( y _ { i } | x _ { i } ) ?$ 

Idea: if $f _ { i }$ doesn’t depend on $y _ { i }$ we cannot overfit the noise! 

Noise2Void [Krull et al., 2019], Noise2Self [Batson, 2019], Neighbor2Neighbor [Huang, 2023] 

• During training flip centre pixel 

• Computes loss only on flipped pixels 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/d4b74aa27502500f02a59dde454ea4ff464e1167668415381ead99e835233d48.jpg)


Blind spot networks [Laine et al., 2019] 

• Convolutional architecture that doesn’t ‘see’ centre 

• pixel by construction 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/31be8cc72ad7049e8e9209115f169c28d1a459a3cdad61611160fdc6d0b12cba.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/6c6c6bca77cf6f6e0e48da3bf422c8002ed0fa63d538207298e2893d02bcea97.jpg)


## Cross-Validation Methods

• We can write these losses as 

$$
\boxed {\min _ {f} | | \mathbf {y} - f (\mathbf {y}) | | ^ {2} \mathrm{subjectto} \frac {\delta f _ {i}}{\delta y _ {i}} (\mathbf {y}) = 0 \forall i, \forall \mathbf {y}}
$$

• SURE’s perspective: 

$$
\min _ {f} | | \mathbf {y} - f (\mathbf {y}) | | ^ {2} + 2 \sigma^ {2} \sum_ {i} \frac {\delta f _ {i}}{\delta y _ {i}} (\mathbf {y})
$$

• These methods are not MMSE optimal! 

## Are We Missing Something?

<table><tr><td>SURE, R2R</td><td>???</td><td>Noise2Self, Noise2Inverse, BlindSpot Nets</td></tr><tr><td>y = x + σε</td><td>unknown σ</td><td>unknown σ</td></tr><tr><td>y = x + σε</td><td>unknown σ</td><td>unknown σ, γ</td></tr><tr><td>y = x + σ ∘ ε</td><td rowspan="2">unknown σ, γ</td><td>y|x ∼ ∏i=1npi(yi|xi), E y|x=x</td></tr><tr><td>unknown σ</td><td>unknown pi</td></tr></table>

## UNSURE

• SURE’s perspective: 

$$
\mathcal {L} _ {\mathrm{UNSURE}} (\pmb {y}, f) = | | \pmb {y} - f (\pmb {y}) | | ^ {2} + 2 \sigma^ {2} \sum_ {i} ^ {\delta f _ {i}} \delta y _ {i} (\pmb {y})
$$

• Impose zero-expected divergence [Tachella et al., ICLR25] 

$$
\mathcal {L} _ {\mathrm{UNSURE}} (\pmb {y}, f) = | | \pmb {y} - f (\pmb {y}) | | ^ {2} \mathrm{subjectto} \mathbb {E} _ {\pmb {y}} \sum_ {i} \frac {\delta f _ {i}}{\delta y _ {i}} (\pmb {y}) = 0
$$

• In practice, we use Lagrange multipliers 

$$
\min _ {f} \max _ {\eta} \mathbb {E} _ {\boldsymbol {y}} | | \boldsymbol {y} - f (\boldsymbol {y}) | | ^ {2} + 2 \eta \sum_ {i} \frac {\delta f _ {i}}{\delta y _ {i}} (\boldsymbol {y})
$$

## UNSURE

• Closed-for solution (si ilar to Tweedie’s for ula) 

$$
\begin{array}{r l r} & & {\underset {f} {\min} \underset {\eta} {\max} \mathbb {E} _ {\mathbf {y}} | | \mathbf {y} - f (\mathbf {y}) | | ^ {2} + 2 \eta \sum_ {i} \frac {\delta f _ {i}}{\delta y _ {i}} (\mathbf {y})} \\ & {\Longrightarrow} & {f ^ {\mathrm{ZED}} (\mathbf {y}) = \mathbf {y} + \hat {\eta} \nabla \log p _ {\mathbf {y}} (\mathbf {y}) \qquad \hat {\eta} = \left(\frac {1}{n} \mathbb {E} _ {\mathbf {y}} | | \nabla \log p _ {\mathbf {y}} (\mathbf {y}) | | ^ {2}\right) ^ {- 1}} \end{array}
$$

• Expected error 

$$
\frac {1}{n} \mathbb {E} _ {x, y} | | f ^ {\mathrm{ZED}} (\pmb {y}) - \pmb {x} | | = \sigma^ {2} \left(\frac {1}{1 - \frac {\mathrm{MMSE}}{\sigma^ {2}}} - 1\right) \approx \mathrm{MMSE} + \frac {\mathrm{MMSE} ^ {2}}{\sigma^ {2}}
$$

• UNSURE can be extended to unknown noise covariance and Poisson Gaussian noise 

## Denoising Experiments

• MNIST dataset (28x28 grayscale) 

• Isotropic Gaussian noise 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/9c799a223c66eb362814139e6fafe8dcc1d5fca1260749822054e245efa422be.jpg)



--- supervised  unsure  unsure via score



neighbor2neighbor  R2R σ = 0.2  SURE σ = 0.2


## Poisson-Gaussian UNSURE

## Real data experiments

• Cryo electron microscopy images 

• Extremely low SNR 

• Approx. Poisson-Gaussian noise 


Measurement


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/1a2af7828180ea0a57d12253a31a8f166f576764a89f50b6117b9d5d96e5bbf6.jpg)



PG-UNSURE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/03d5aa338e0ff75ab81139c1838ff6176dc626144417ce6339f8705b80393957.jpg)


## Outline

How can we learn $f$ from measurement $\{ { y } _ { i } \}$ data alone? 

1. Noisy: $y = x + \epsilon$ 

2. Incomplete and noisy: $y = A x + \epsilon$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/33729b32c0c356689d9c6d57d7888d6a9aafa904325ff8c372139685b13462c8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/0c0b3c614ec2981674388b18582079919e1380c9634cd1f2132938c910530937.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/80695b4b4c5ecc889815b8fa29c5112f940b9afedf34a2c6dc4a549e482ea6ae.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/bd67de4bec1c8820705dec3df008bd341d770e4b890598a9aad98eecf4375121.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/418080401ce8c927d827489e5836cf72b4f2cdb48e12d2034a2fbf3af7579511.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/e50bf452bc53f88172d2cb6f0c76232026307cd02459aeb37b8a4be0959cc2a2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/30d50e6602105c7af03bbe6b1d92e41417d25b35e170c89a11c737d772a9dff0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/3dee23fd3f74a27826e315b1d0bc49a0a34a77677d77a1168461f8f8aa15671c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/c165b1c7fe9c8c01f31a8fcbd069e0c8f5ccd62f783ac12154e71ec9e4b70e78.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/e5602d7e5872e2719f27729f7735bd6458b40316ba9aef7e7623026979dfccb2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/b501a0627872fc24f07f1ae1f4b542db08e5f14cdb49dd400b4f86eab463795b.jpg)


## Incomplete Measurements?

For $A \neq I ,$ most estimators can be adapted to approximate 

$$
\mathbb {E} _ {\boldsymbol {x}, \boldsymbol {y}} | | A (\boldsymbol {x} - f (\boldsymbol {y})) | | ^ {2}
$$

In this $\mathsf { c a s e } ,$ the risk does not penalise $f ( y )$ in the nullspace of $A !$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/9059b76f0cbd0f5e5b3ea9ae833346e3f0b2c683e3d0a6b0b398f9a5b16da6ba.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/e862db1272dbf622d3446f2e3af8d87f556a2fb6f47c4335e35d81d244379403.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/5a399c1935c40fe8deb3bc00ba02f14b8c378186ffe0eed45badd8263a1474ea.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/35b4601dc06681883e0274a559e0d7450f57a13787ffb29b29589f5253724fab.jpg)


## Learning from Measurements

How to learn from only ??? 

• Access multiple operators $y _ { i } = A _ { g _ { i } } x _ { i }$ with $g \in \{ 1 , \ldots , G \}$ 

• Each $A _ { g }$ with different nullspace 

• Offers the possibility for learning using multiple measurement operators 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/30fdfcb16ac918ad0b41d87cb6fb2bc025adf8c0e808f7d22efbca0847ae117d.jpg)


## Necessary Condition

Intuition: we need that the operators $A _ { 1 } , A _ { 2 } , \ldots A _ { G }$ cover the whole ambient space [Tachella et al., 2023, JMLR]. 

Proposition: Learning reconstruction mapping ?? from observed measurements possible only if 

$$
\mathrm{rank} \bigl (\mathbb {E} _ {g} A _ {g} ^ {\top} A _ {g} \bigr) = n
$$

and thus, if $m \geq n / G$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/c8c19897ac2dd1186173d329e6ca46d31564fbc69a7314d5f8401819c3950dbf.jpg)


## Multi Operator Imaging

Multi Operator Imaging (MOI) [Tachella et al., 2022, NeurIPS] 

$$
\mathcal {L} _ {\mathrm{MOI}} (\pmb {y}, f) = \underbrace {\left| \left| \pmb {y} - A _ {g} f (\pmb {y} , A _ {g}) \right| \right| ^ {2}} + \sum_ {s} \underbrace {\left| | f (A _ {s} \widehat {\pmb {x}} , A _ {s}) - \widehat {\pmb {x}} | \right| ^ {2}} \quad \text {with} \widehat {\pmb {x}} = f (\pmb {y}, A _ {g})
$$

Can be replaced by SURE, UNSURE, etc. 

Enforces $f \left( A _ { g } \mathbf { x } , A _ { g } \right) \approx f ( A _ { s } \mathbf { x } , A _ { s } )$ 

## Inpainting Experiments

• U-Net network 

• CelebA dataset 

$G = 4 0$ different $A _ { g }$ inpainting masks 

Measurements ?? 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/18bc903718f20fe2ae834a5f0aa3c6c97d5e590e3f717025ed55ff8d5de5c20d.jpg)



Signal ??


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/0f5b0b8e750c9d3397a1cb86a0b7c35ffccf9835e57338a32e171d813f437fce.jpg)



Supervised


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/94e027498cd41e65266c2c85744b0767713ddee4a40aa9491fadcfd06602baf4.jpg)



MOI


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/ac749978aadeef41765483ff1d38595f8238b3d9bc851e38868e7a750e6ab6f7.jpg)


## Symmetry Prior

Equivariant Imaging [Chen, Tachella and Davies, ICCV 2021] 

For all $g \in G$ we have 

$$
\mathbf {y} = A \mathbf {x} = \overbrace {A T _ {g} T _ {g} ^ {- 1} \mathbf {x}} ^ {\mathbf {x} ^ {\prime}} = A _ {g} \mathbf {x} ^ {\prime}
$$

• We get multiple virtual operators $\left\{ A _ { g } \right\} _ { g \in G }$ ‘for free’! 

• Each $A T _ { g }$ might have a different nullspace 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/ffdfc7bfdb69b0c030d57a20e15ba609e552308b386fed039074574eff05ef86.jpg)


## (Non)-Equivariant Operators

Theorem [T. et al., 2023]: The full rank condition requires that ?? is not equivariant: $A A T _ { g } \ne T _ { g } A A$ 

$$
\operatorname{rank} \bigl (\mathbb {E} _ {g} T _ {g} ^ {\top} A ^ {\top} A T _ {g} \bigr) = \operatorname{rank} \bigl (A ^ {\top} (\mathbb {E} _ {g} \tilde {T} _ {g} ^ {\top} \tilde {T} _ {g}) A \bigr) = \operatorname{rank} \bigl (A ^ {\top} A \bigr) = m <   n
$$

## Equivariant Imaging

How can we enforce equivariance in practice? 

Idea: we should have $f { \big ( } A T _ { g } x { \big ) } = T _ { g } f ( A x )$ , i.e. ?? ∘ ?? should be ??-equivariant 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/922ea2ee4c2e09628797dad972f68b11ee3ad38ac23550b14e11e6ff964526b5.jpg)


## Equivariant Imaging

We can leverage invariance of $p ( { \pmb x } )$ to transformations $T _ { g }$ to learn in the nullspace 

$$
\mathcal {L} _ {E I} (\pmb {y}, f) = \mathbb {E} _ {g} | | T _ {g} \widehat {\pmb {x}} - f (A T _ {g} \widehat {\pmb {x}}) | | ^ {2}
$$

where ${ \widehat { \pmb x } } = f ( \pmb y )$ is used as reference 

Robust Equivariant Imaging++ [Chen, Tachella & Davies 2022, CVPR] enforces equivariance of ?? ∘ ?? 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/d65bbed21df3bc9ed3e45c0390e43045ff012c4e7b6f6985070b7eb2ba0d27bd.jpg)


Handles noisy measurements of unknown noise level 

## Experiments

• FastMRI dataset 

• Gaussian noise 

<table><tr><td>Method</td><td>CV + EI</td><td>EI(assumes σ = 0)</td><td>UNSURE + EI(unknown σ)</td><td>SURE + EI(assumes σ = 0.05)</td><td>Supervised</td></tr><tr><td>PSNR [dB]</td><td>33.25 ± 1.14</td><td>34.32 ± 0.91</td><td>35.73 ± 1.45</td><td>28.05 ± 4.73</td><td>36.63 ± 1.38</td></tr></table>


Ground-truth


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/8dab84a99836cebdc929601127aa187842995045c85d15c4572d6c8771fb856a.jpg)



Measurement


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/f619656022577df062f0811e1a11ea595f5784e220c72d9e6550304710f2dc1a.jpg)



Noise2Inverse


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/3fbe9c61e6830c26074300c0e196141871ea48e0b115bffc1291340369a4b0e1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/ace16d9c874772b2fc66b5f32e2ab0bd6e8e6dfa89d070a5f105f5365477d7e5.jpg)



El σ= 0



UNSURE+EI


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/fc1289f07ab49e1da5389cfb7ca4d75b1440029872a0e92cf75a8852a075b67d.jpg)



SURE+EI $\sigma = 0 . 0 5$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/fdece9f578e21bf7a777e6b541f1394bf976ea74764b25f1e0550f3850922606.jpg)



Supervised


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/d5ea9051e5ce04efe89c39962c7351e9e26b19a6d7d54950a6ab6374999a5072.jpg)


## Experiments

• Operator ?? is isotropic blur with Gaussian noise 

• Dataset is approximately scale invariant 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/8f57a3e6a68443470b3b0507ed3c86dd30f1fe259f6404bdeb98530c9da1659e.jpg)


## Finetuning

• Reconstruct Anything Model: general model solving many inverse problems [Terris et al., 2025] 

easure ent 

Motion blur 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/627923f9c0f7107e44c5ae7ac4955022a813a7dd49ee8448fb603237a4d08ca1.jpg)



Gaussian tomography


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/f4a871e08856aedb640a5715dd6134ce7bc92c50370445545384ecf769ee04b2.jpg)



Singlecoil MRI


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/92ed316479375b33e15af02bacf7710be9ef887cfca046bedbdf80d8e9ff436d.jpg)



Poisson tomography


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/6f0c32ed5a1262079335373f13a01d383b8cfabd977f8053bc10edaa8ccf8006.jpg)



Multicoil MRI


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/0ce4b909be447093e9aa18ac46fc61a8bd140af3aab742af72c168c15eb71dc3.jpg)



RAM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/2acbc1698a009e7ad7ee338be85fa3f475f5f2df6d5793d4a2cfddd0913a3121.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/44d108e94cfd4b2b22890b2e19ffe350d1d9f26c55ed6f6b00522d1933ec388e.jpg)



In-distribution degradation and images


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/2645c5edbc0acbd07445b29df3ebf85153b4a8c44122788ee087fdfb811e2332.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/c6cd0ac34ffbdf25b62540cea4b4ba262286ecf64a001cd30a9f2c853e9cbf9b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/ccec31d8c4a9e67ec2c1b7b585904d46f94dfef4bfc6827390d63e2f2ab473fb.jpg)



Out-of-distribution degradation



Zero-shot performance


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/e7c69f6e3d5fcb93bb4c13b7831dc2babb82d74968d2033779678e0afd65e178.jpg)


## Finetuning

• The model can be finetuned with self-supervised losses on up to a single ?? ?? = 1 • Finetuning can be done in a few seconds 

Table 5. Self-supervised finetuning time in seconds. 

## References

Slides and codes of a recent 3-hour tutorial can be found here: 

https://tachella.github.io/blog/selfsuptutorial/ 

## Deep Inverse

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/788f9dd1b0bd8d113e2fa6c914b35df28f9e65500c9c0ccbc7f30dad53b7605f.jpg)


## Section Navigation

Basics 

Optimization 

Plug-and-Play 

Sampling 

Unfolded V 

Patch Priors V 

Self-Supervised Learning 

Adversarial Learning 

Advanced V 

Quickstart Examples User Guide API Finding Help More 

> Examples 

## Examples

All the examples have a download link at the end. You can load the example's notebook on Google Colab and run them by adding the line 

pip instal1 git+https://github.com/deepinv/deepinv.git#egg=deepinv 

to the top of the notebook (e.g., as in here). 

## Basics

## Deep Inverse

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/82cf8ad4a1b7dbd0f00a097f1246cc22b34110295ee1916a954a0fdb02ca865a.jpg)


## 1-bit Compressed Sensing

$y = \mathrm { s i g n } ( A x )$ with Gaussian ?? and 20% undersampling ratio 

• Dataset is approximately translation invariant 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/f4a34820621940a7f38e367412b6a0d84aa63348f2745290ac826298447330b8.jpg)



[Tachella & Jacques, TMLR 2023]


## Correlated Noise Experiments

• DIV2K dataset (320x320 RGB) 

• Spatially correlated Gaussian noise 3x3 pixels 

<table><tr><td>UNSURE(unknown Σ)</td><td>SURE(known Σ)</td><td>Supervised</td></tr><tr><td>28.72 ± 1.03</td><td>29.77 ± 1.22</td><td>29.91 ± 1.26</td></tr></table>

<table><tr><td>Kernel size η</td><td>1 × 1</td><td>3 × 3</td><td>5 × 5</td></tr><tr><td>PSNR [dB]</td><td>23.62</td><td>28.72 ± 1.03</td><td>27.38 ± 0.88</td></tr></table>


Ground-truth


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/c0315b278142349cdd31a498440213aa0bc2dc38f6258af6253321669d5f5316.jpg)



Measurement


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/69618900db07d7925ebf457dce3fc13b62416ecc986d886adbb013835f19c683.jpg)



UNSURE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/3606359992c94e6c73c0579114615605374d8854a96480a39d650a306ff1cdbb.jpg)



SURE known∑


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/f5acfb8b48e1072c704aacf6b7ab382e686f153d14ca67bc17833acb424bf234.jpg)



Supervised


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/5fe78c3f36f956a4c523b995a65d34e9f788c188f6d0f9eac5387493b70a1c98.jpg)


## Efficient SURE

Recorrupted2Recorrupted [Pang et al., 2021], Noisier2Noise [Moran et al., 2020], etc. 

Proposition: Let $\pmb { y } \sim \mathcal { N } \big ( \pmb { x } , \pmb { I } \sigma ^ { 2 } \big )$ and define 

$$
\begin{array}{r} \mathbf {y} _ {a} = \mathbf {y} + \alpha \boldsymbol {\omega} \\ \mathbf {y} _ {b} = \mathbf {y} - \boldsymbol {\omega} / \alpha \end{array}
$$

where ${ \pmb { \omega } } \sim \mathcal { N } \big ( { \bf 0 } , I \sigma ^ { 2 } \big )$ and $\alpha \in \mathbb { R }$ , then $y _ { a }$ and $y _ { b }$ are independent random variables (fixed ??). 

$$
\mathcal {L} _ {\mathrm{R2R}} (\pmb {y}, f, \alpha) = \mathbb {E} _ {\pmb {\omega}} | | \pmb {y} _ {b} - f (\pmb {y} _ {a}) | | ^ {2}
$$

Generalized Recorrupted2Recorrupted [Monroy, Bacca & Tachella, CVPR 2025] 

• Equivalent to SURE in the limit: lim $\mid _ { _ { \mathrm { \scriptsize ~ R 2 R } } } ( \boldsymbol { y } , \boldsymbol { f } , \alpha ) = \mathcal { L } _ { \mathrm { S U R E } } ( \boldsymbol { y } , \boldsymbol { f } )$ ??→ 

• Extensions to Poisson, Gamma and Binomial 

## Uncertainty Quantification

Can we measure the uncertainty of the reconstructions? 

Self-supervised losses can also be used for uncertainty quantification! 

• SURE can be used to assess reconstruction error in denoising 

• SURE4SURE [Bellec et al., 2021] gives error variance estimates. 

• EI loss can be seen as a bootstrapping technique 

[T. & Pereyra, 2024] with well calibrated uncertainty estimates 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/c3a6a3a1-9247-4711-b328-f93dabb461a5/960261002d7fcda2dad715f41c8ce948bdc4cce1662df60c16368616475c787b.jpg)


## Implementation

We parameterize $f$ as a deep neural network, small $\alpha > 0 , \omega \sim \mathcal { N } ( \mathbf { 0 } , I )$ 

1) UNSURE: Solve Lagrangian problem, approximating divergence as [Ramani et al., 2007] 

$$
\operatorname{tr} (\Sigma \frac {\delta f}{\delta \mathbf {y}}) \approx \frac {(\Sigma \pmb {\omega}) ^ {\top}}{\alpha} \left(f (\mathbf {y}) - f (\mathbf {y} + \pmb {\omega} \alpha)\right)
$$

2) UNSURE via score: Learn score $s ( y ) \approx \nabla \log p _ { y } ( y )$ via 

$$
\arg \min _ {s} \mathbb {E} _ {\boldsymbol {y}, \boldsymbol {\omega}} | | \boldsymbol {\omega} - \alpha s (\boldsymbol {y} + \boldsymbol {\omega} \alpha) | | ^ {2}
$$

and use $f ( y ) = y + \Sigma _ { \widehat { \pmb { \eta } } } s ( \pmb { y } )$ to denoise at test time. 