# Lecture 3: Models and algorithms for $\ell _ { 2 } \mathrm { - } \ell _ { 0 }$ optimisation problems

Luca Calatroni CR CNRS, Laboratoire I3S CNRS, UCA, Inria SAM, France 

MIVA ERASMUS BIP PhD winter school Advanced methods for mathematical image analysis University of Bologna, IT January 18-20 2022 

## Table of contents

1. Introduction 

2. $\ell _ { 2 } - \ell _ { 0 }$ minimisation 

3. Algorithms for $\ell _ { 2 } - \ell _ { 0 }$ minimisation Iterative Hard Thresholding Greedy algorithms 

4. Continuous relaxations Exactness Iteratively reweighted algorithms 

5. Application to super-resolution microscopy 

## Why `<sub>0</sub>?

Many problems in signal/image processing are concerned with sparse recovery: compressed sensing, variable selection, source separation, learning... 

$$
d = A x + n
$$

$d \in \mathbb { R } ^ { m }$ : observed data (signal processing notation) 

$x \in \mathbb { R } ^ { n }$ unknown solution to be estimated 

$A \in \mathbb { R } ^ { m \times n }$ observation matrix, 

• Few observations y and large explicative unknown variables x, with $m \ll n .$ Undertermined system! A is ill-conditioned, noise is present. 

• Regularisation: assume the signal is sparse by considering $\ell _ { 1 } { \mathrm { - n o r m } }$ or $\ell _ { 0 }$ pseudo-norm constraints: 

$$
\| x \| _ {1} \leq K,
$$

$$
\| x \| _ {0} \leq K
$$

with $\begin{array} { r } { \| { \boldsymbol x } \| _ { 0 } : = \# \left\{ { \boldsymbol x } _ { i } , \ i = 1 , \ldots , n : { \boldsymbol x } _ { i } \neq 0 \right\} = \sum _ { i = 1 } ^ { n } | { \boldsymbol x } _ { i } | _ { 0 } } \end{array}$ , with 

$$
| z | _ {0} = \left\{ \begin{array}{l l} 1 & \quad \text { if } x \neq 0 \\ 0 & \quad \text { if } x = 0 \end{array} \right.
$$

## Dictionary representation in imaging

Image are heterogeneous signals, with smooth (homogeneous) areas, edges, texture,.. 

Take $d \in \mathbb { R } ^ { m }$ be a patch of an image or a signal 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/5d9ed3c8c649fd7f5b290c8a89486e2a1181319df77a137492c7da954eb4561a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/fe797854627aa45856efbba03a8e5673c47f77de359dac2261aaf9cf44e6b952.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/61231d1ca2e3f065b51a162bf93af0fd3c99410fe4fa48f54f7e5e25bd5ffc90.jpg)


Each $d$ is represented by given waveforms whose shape matches the image structure. Standard choices of $a _ { j }$ vectors come from Haar, smooth wavelets, sine/cosine transform... 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/8b67a286bfd4a770d790b6e7aa2a7f8f72dab9a1b0c13d36d41b21b002202383.jpg)


Take $A = [ a _ { 1 } , . . . , a _ { n } ] \in \mathbb { R } ^ { m \times n }$ to be a set of normalised (basis) vectors. 

## Dictionary representation in imaging

• Such A is a redundant dictionary (sequence of representative waveforms) 

• The dictionary A is adapted to the signal d if d can be represented by a few number of vectors $a _ { j }$ (atoms) of $A ,$ , that is $d \approx A x$ with x sparse, that is 

$$
\| x \| _ {0} \leq K, \quad K <   <   n
$$

d 

$$
\left[ \begin{array}{c} \mathbb {I} \\ \hline \end{array} \right] = \left[ \begin{array}{c c c c} \hline & & & \\ a _ {i 1} & & a _ {i 2} & a _ {i 3} \\ \hline \end{array} \right] \left[ \begin{array}{c} x _ {l} \\ \hline x _ {2} \\ x _ {3} \\ \hline \end{array} \right] = x _ {l} \left[ \begin{array}{c} \mathbb {I} \\ \hline \end{array} \right] + x _ {2} \left[ \begin{array}{c} \mathbb {I} \\ \hline \end{array} \right] + x _ {3} \left[ \begin{array}{c} \mathbb {I} \\ \hline \end{array} \right] + \dots
$$

## Examples in signal/image processing

## Examples

• signal is a sum of spikes, modelled by a sum of Dirac $\textstyle \sum _ { r = 1 } ^ { K } x _ { r } \delta _ { t _ { r } }$ . 

• acquisition system is modelled as a convolution with a Gaussian function: 

$$
d (\cdot) = h * \sum_ {r = 1} ^ {K} x _ {r} \delta_ {t _ {r}} = \sum_ {r = 1} ^ {K} x _ {r} h (\cdot - t _ {r}).
$$

Assume that the Dirac locations $t _ { r }$ are on a regular grid indexed by $i = 1 , . . . n$ 

$$
\left[ \begin{array}{c} \framebox {\|} \\ \framebox {\|} \end{array} \right] = \left[ \begin{array}{c} \framebox {\|} \\ \framebox {\|} \end{array} \right] \left[ \begin{array}{c c c} \underline {{x _ {1}}} & - - t _ {1} & + \\ \underline {{x _ {2}}} & - - t _ {2} & \\ \underline {{x _ {3}}} & - - t _ {3} & \end{array} \right] \left[ \begin{array}{c} \framebox {\|} \\ \framebox {\|} \end{array} \right]
$$

$$
\mathrm{d} = \mathrm{A} \quad \mathrm{x} + \mathrm{n}
$$

• 1D example: Channel estimation in communications, ... 

• 2D example: Single Molecule Localisation in super-resolution microscopy 

## Single Molecule Localisation in super-resolution microscopy I

SMLM idea 

Modelling: for $t \in \{ 1 , \ldots , T \}$ , given a blurry, undersampled and noisy image $d _ { t } \in \mathbb { R } ^ { m }$ , consider the problem: 

find sparse 

s.t. 

$$
d _ {t} = A x _ {t} + n _ {t}
$$

$$
\forall t \in \{1, \dots , T \}
$$

$A : = S H \in \mathbb { R } ^ { m \times n }$ with $H \in \mathbb { R } ^ { n \times n }$ convolution and $S \in \mathbb { R } ^ { m \times n }$ undersampling , $n = L m , L > 1$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/01492218819637d06aa12f45bb61fc770b228eacf8a490f6fb658857d97e9e07.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/333af95ef0659974c633ff9fbd978055234a5bf01f3f0b9b3e0df52c3f567ed0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/affe3c53d0e6ba8231aee6a2846c780611e856a6cd02b6be407b6a8ecec868e7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/48729bf9351eee962635eb597b9c66b96bae97410687e8b780913fcad77dc0b8.jpg)


## Single Molecule Localisation in super-resolution microscopy II

Regularisation approach: look for sparse solutions at each time $t \in \{ 1 , \ldots , T \}$ 

$$
x _ {t} ^ {*} \in \arg \min _ {x} \frac {1}{2} \| A x - d _ {t} \| ^ {2} + \lambda \| x \| _ {0} + \iota_ {x \geq 0} (x), \quad \lambda > 0
$$

Final reconstruction obtained simply by $\begin{array} { r } { x = \sum _ { i = 1 } ^ { T } x _ { t } ^ { * } } \end{array}$ (Gazagnes, Soubies, Blanc-F´eraud, Schaub, ’15, Lazzaretti, Calatroni, Estatico, ’21) 

$\ell _ { 2 } { - } \ell _ { 0 }$ minimisation 

## ` -` minimisation

## $\ell _ { 2 } - \ell _ { 0 }$ : problem forms

For $A \in \mathbb { R } ^ { m \times n } , ~ m \leq n$ consider the following formulations: 

• Exact recovery: 

$$
\hat {x} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} \| x \| _ {0} \text {   subject   to   } A x = d
$$

• Approximation problem in constrained forms $( \epsilon > 0 , K > 0 )$ 

$$
\hat {x} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} \frac {1}{2} \| A x - d \| _ {2} ^ {2} \text {   subject   to   } \| x \| _ {0} \leq K
$$

$$
\hat {x} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} \| x \| _ {0} \text {   subject   to   } \| A x - d \| _ {2} ^ {2} \leq \epsilon
$$

• Approximation problem in penalised form $( \lambda > 0 )$ 

$$
\hat {x} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} G _ {\ell_ {0}} (x) := \frac {1}{2} \| A x - d \| _ {2} ^ {2} + \lambda \| x \| _ {0}
$$

non-continuous, non-convex and NP-hard optimisation problem (Natarajan, ’95, Davies et al., ’97): a solution cannot be verified in polynomial time w.r.t the dimension of the problem 

• Non equivalent formulations 

• Existence of optimal solutions and relations between formulations in Nikolova, ’16 

• Very active field of research in signal and image processing, and in statistics. 

## How people do: $y _ { 2 } = 1 3$ minimisation

A popular way to deal with this problem consists in considering the $\ell _ { 1 } { \mathrm { - n o r m } }$ instead 

$\ell_2 - \ell_1$ problem formulations Constrained formulation $(K > 0)$ : $\hat{x} \in \arg \min_{x \in \mathbb{R}^n} \|Ax - d\|_2^2$ subject to $\|x\|_1 \leq K$ Penalised formulation $(\lambda > 0)$ : $\hat{x} \in \arg \min_{x \in \mathbb{R}^n} \|Ax - d\|_2^2 + \lambda \|x\|_1$ 

• Easier optimization problems: convex and continuous (but non smooth) → available solvers (see previous courses)! 

• The two formulations are equivalent 

• Under some conditions involving A, solving these problems allows to find a solution of the $\ell _ { 2 } { - } \ell _ { 0 }$ problem (Cand`es, Romberg, Tao, ’05) 

• They are known as Basis Pursuit De-Noising (BPDN) Chen et al., ’98, or LASSO (Tibshirani, ’96) problems, respectively. 

Standard example in $\mathbb { R } ^ { 2 }$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/9faf99614300e04533118f1b3c8a6b5b1011a3cdd7a5da42e80525a57773cb44.jpg)



Level lines of $\| A x - d \| _ { 2 } ^ { 2 }$


Standard example in $\mathbb { R } ^ { 2 }$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/9ad421b2d1732202e86f231b95df087848c30cb7f1bd5e0f46de8700a27d3831.jpg)



Level lines of $\| A x - d \| _ { 2 } ^ { 2 }$ with $\ell _ { 2 }$ constraint $\| x \| _ { 2 } \leq K \  \ ( x _ { 1 } , x _ { 2 } ) \neq ( 0 , 0 )$


Standard example in $\mathbb { R } ^ { 2 }$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/974b044757fde5b2adc1a39a447120aa0345f6eb9d8c4fa7ea6ca9cebc77e347.jpg)


Level lines of $\| A x - d \| _ { 2 } ^ { 2 }$ with $\ell _ { 1 }$ constraint $\| x \| _ { 1 } \le K \ \to \ x _ { 1 } = 0$ 

## Sparsity through sof-thresholding

Recall that in 1D: 

$$
\hat {x} = \underset {x \in \mathbb {R}} {\arg \min} \left\{\frac {1}{2} (d - x) ^ {2} + \lambda | x | \right\} = \operatorname{prox} _ {\lambda |. |} (d)
$$

is reached at 

$$
\hat {x} = \mathcal {T} _ {\lambda} (d) = \left\{ \begin{array}{l l} d - \text {sign} (d) \lambda & \text {if} | d | > \lambda \\ 0 & \text {if} | d | \leq \lambda \end{array} \right.
$$

By, separability, this is then used for defining $\mathsf { p r o x } _ { \lambda \parallel \cdot \parallel _ { 1 } } ( \cdot )$ 

. . . many zeros! 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/de2e9864ff1d9289d629cd42bc68fbc7f35013f2d43d5da3197b4a25edae4a6c.jpg)


Note: using $\ell _ { 2 }$ norm we get instead 

$$
\hat {x} = \underset {x \in \mathbb {R}} {\arg \min} \left\{\frac {1}{2} (d - x) ^ {2} + \lambda x ^ {2} \right\}.
$$

$\begin{array} { r } { \hat { x } = \frac { d } { 1 + 2 \lambda } } \end{array}$ which is diferent from 0 as soon as $d \neq 0$ 

## Algorithmic advantages in solving $y _ { 2 } = 1 3$ problems

You now know how to solve the problem: 

$$
\underset {x} {\arg \min} \frac {1}{2} \| A x - d \| ^ {2} + \lambda \| x \| _ {1}, \quad \lambda > 0
$$

• ISTA (Combettes, Wajs, ’05) 

• FISTA (Beck, Teboulle, ’09) 

• If A is positive definite → strongly convex problem, hence V-FISTA can be used (Beck, ’17) 

For analysis approaches, i.e. when sparsity is assumed w.r.t. to some basis $W \in \mathbb { R } ^ { N \times n }$ (gradient, wavelets. . . ) 

$$
\arg \min _ {x} \frac {1}{2} \| A x - d \| ^ {2} + \lambda \| W x \| _ {1}, \quad \lambda > 0
$$

you can use, e.g., ADMM (Glowinski, Marroco, ’75, Boyd et al, ’11). 

## So. . . why just not solving $1 4 2 - 1 6 8 = 1 8 8$

## Compressed Sensing Theory

• A sparse signal $( \| x \| _ { 0 } \leq K )$ can be exactly reconstructed by solving the constrained $\ell _ { 1 }$ problem when Restricted Isometry Property (RIP) of matrix A (Donoho et al., Cand`es et al. ’06) 

• Roughly speaking A satisfies the RIP if $A ^ { T } A \approx I d$ 

• Under RIP conditions on $A , \ell _ { 0 }$ can be replaced by $\ell _ { 1 }$ 

• Otherwise (frequent cases in inverse problems), the two optimisation problems give diferent solutions. 

$\ell _ { 1 }$ promotes sparsity but introduces biases, since in correspondence of the actua non-zeros the magnitude is lowered. 

$\ell _ { 0 }$ better promotes sparsity than $\ell _ { 1 }$ in the general case. 

Algorithms for $\ell _ { 2 } { - } \ell _ { 0 }$ minimisation 

Algorithms for $\ell _ { 2 } - \ell _ { 0 }$ minimisation 

Iterative Hard Thresholding 

$\arg \min_{x\in \mathbb{R}^n}\frac{1}{2}\| Ax - d\| _2^2 +\lambda \| x\| _0$ - $\frac{1}{2}\| Ax - d\|^2$ is $L$ -smooth $(L = \| A\| ^2)$ - The proximal operator of $\| \cdot \| _0$ is the hard thresholding operator 

Algorithm: Iterative hard thresholding (IHT)

Input: $x_{0} \in R^{n}, \tau \in (0, \frac{1}{L})$ .

for $k \geq 0$ do $x_{k+1} = \text{prox}_{\tau\lambda\|\cdot\|_{0}} \left( x_{k} - \tau A^{T}(Ax_{k} - d) \right)$ $= \mathcal{H}_{\sqrt{2\lambda\tau}} \left( x_{k} - \tau A^{T}(Ax_{k} - d) \right)$ end for

- IHT converges to a critical point (in Blumensath, Davies, '09 for $\tau = 1$ and $\|A\| < 1$ , in Attouch et al., '13 general FB-type result)
- As always for non convex problems, initialisation is crucial! One good idea is to initialise with the solution of $\arg\min_{x \in R^{n}} \frac{1}{2} \|Ax - d\|_{2}^{2} + \lambda \|x\|_{1} \rightarrow \text{computed by FISTA}$ 

## Non-convex proximal gradient: iterative hard thresholding

Consider the penalised form of the problem: 

$$
\underset {x \in \mathbb {R} ^ {n}} {\arg \min} G _ {\ell_ {0}} (x) := \frac {1}{2} \| A x - d \| _ {2} ^ {2} + \lambda \| x \| _ {0},
$$

Introduce the surrogate function for al $z \in \mathbb { R } ^ { n }$ 

$$
C _ {\ell_ {0}} ^ {S} (x, z) := \frac {1}{2} \| A x - d \| _ {2} ^ {2} + \lambda \| x \| _ {0} - \frac {1}{2} \| A x - A z \| _ {2} ^ {2} + \| x - z \| _ {2} ^ {2}
$$

It can be shown that if $\| A \| < 1$ , then $C _ { \ell _ { 0 } } ^ { S } ( x , z )$ majorises $G _ { \ell _ { 0 } } ( x )$ 

$$
G _ {\ell_ {0}} (x) \leq C _ {\ell_ {0}} ^ {S} (x, z), \quad \forall z \in \mathbb {R} ^ {n}.
$$

Note, moreover, that $G _ { \ell _ { 0 } } ( x ) = C _ { \ell _ { 0 } } ^ { S } ( x , x )$ . We can thus optimise $C _ { \ell _ { 0 } } ^ { S } ( x , z )$ with respect to x. We can rewrite: 

$$
C _ {\ell_ {0}} ^ {S} (x, z) = \frac {1}{2} \sum_ {i = 1} ^ {n} \left(x _ {i} ^ {2} - 2 x _ {i} \left(z _ {i} + a _ {i} ^ {T} d - a _ {i} ^ {T} A z\right) + \lambda | x _ {i} | _ {0}\right) + \frac {1}{2} \left(\| d \| ^ {2} + \| z \| ^ {2} - \| A z \| ^ {2}\right)
$$

By treating the case $x _ { i } = 0$ and $x _ { i } \neq 0$ separately and comparing we get: 

$$
x = \mathcal {H} _ {\sqrt {2 \lambda}} (z - A ^ {T} (A z - d)), \quad \forall z
$$

IHT obtained by setting $z = x _ { k }$ and $x = x _ { k + 1 }$ 

Greedy algorithms 

## Greedy algorithms

Greedy algorithms: matching pursuit (MP) (Mallat et al., ’93), Orthogonal MP (Pati et al., ’93), Orthogonal Least Squares (OLS, Chen et al., ’89), Bayesian OMP (Herzen et al., ’10), Single Best Replacement (Soussen et al, ’11). 

## Matching Pursuit

$d \in \mathbb { R } ^ { m }$ is the signal to represent with a limited number of $K \ll n$ of atoms of dictionary $A \in \mathbb { R } ^ { m \times n }$ , i.e. of columns $a _ { j }$ of $A , i = 1 , \dotsc , n .$ 

$$
\left[ \begin{array}{c} d \\ \hline \end{array} \right] = \left[ \begin{array}{c c c} & A \\ a _ {1 1} & a _ {1 2} & a _ {1 3} \end{array} \right] \left[ \begin{array}{c} x \\ \hline x _ {1} \\ x _ {2} \\ x _ {3} \end{array} \right] = x _ {1} \left[ \begin{array}{c} \hline \\ a _ {1 1} \end{array} \right] + x _ {2} \left[ \begin{array}{c} \hline \\ a _ {1 2} \end{array} \right] + x _ {3} \left[ \begin{array}{c} \hline \\ a _ {1 3} \end{array} \right] + \dots
$$

MP considers the constrained formulation: 

$$
\underset {x \in \mathbb {R} ^ {n}} {\arg \min} \| A x - d \| ^ {2}, \quad \text { subject   to } \quad \| x \| _ {0} \leq K
$$

and try to add one component at a time. 

## Matching pursuit: main ideas

Assumption: A has unit column norms, i.e. $\| a _ { i } \| = 1$ for all $i = 1 , \ldots , n .$ 

Algorithm: Matching pursuit

Input: A s.t. $\|a_{i}\|=1$ , d, $K\ll n$ .

Initialise: $r_{0}=d$ , $\sigma_{0}=\emptyset$ , $x_{0}=0$ .

while $\#\sigma_{k}\leq K$ do $i_{k}=\arg\max_{j\in\{1,\ldots,n\}}|\langle r_{k},a_{j}\rangle|$ $\sigma_{k+1}=\sigma_{k}\cup\{i_{k}\}$ $x_{k+1}=x_{n}+\langle a_{i_{k}},r_{k}\rangle e_{i_{k}}$ $r_{k+1}=r_{k}-\langle r_{k},a_{i_{k}}\rangle a_{i_{k}}$ end while 

• The quantity $\| r _ { k } \|$ converges exponentially to 0 (Mallat et al, ’93) 

• In Gribonval et al., ’96, a diferent correlation function (not $| \langle \cdot , \cdot \rangle | )$ is considered. 

## Orthogonal Matching Pursuit

OMP idea (Pati et al. ’93, Tropp, $^ { \prime } 0 4 )$ : at each iteration of MP optimally estimate the intensity values having the current support fixed by solving 

$$
x _ {k + 1} = \arg
$$

$$
\left\| A x - d \right\| ^ {2}
$$

$$
x \in \mathbb {R} ^ {n}
$$

$$
x _ {i} = 0 \forall i \notin \omega := \sigma (x _ {k}) \cup i _ {k + 1}
$$

Algorithm: Orthogonal matching pursuit

Input: A s.t. $\|a_{i}\|=1$ , d, $K\ll n$ .

Initialise: $r_{0}=d$ , $\sigma_{0}=\emptyset$ , $x_{0}=0$ .

while $\#\sigma_{k}\leq K$ do $i_{k}=\arg\max_{j\in\{1,\ldots,n\}}|\langle r_{k},a_{j}\rangle|$ $\sigma_{k+1}=\sigma_{k}\cup\{i_{k}\}$ $x_{k+1}=\arg\min_{x\in\mathbb{R}^{n}}\|Ax-d\|^{2}$ , subject to $x_{i}=0\forall i\notin\sigma(x_{k+1})$ $r_{k+1}=d-Ax_{k+1}$ end while 

$$
k \geq 0
$$

$$
A)
$$

The main idea of the other existing greedy algorithms is that at each iteration one component is: 

• added 

• removed 

• replaced 

The more complex is the strategy, the best is the solution, but the largest is the computing time. . . 

Continuous relaxations 

Think of a diferent idea for solving the problem: 

$$
\frac {1}{2} \| A x - d \| ^ {2} + \lambda \| x \| _ {0} \quad \Longrightarrow \quad \frac {1}{2} \| A x - d \| ^ {2} + \sum_ {i = 1} ^ {n} \phi_ {i} (x _ {i})
$$

Idea: use continuous and separable functions $\phi _ { j } \mathopen { } \mathclose \bgroup \left( x _ { j } \aftergroup \egroup \right)$ (convex and non-convex). 

$\ell _ { 1 }$ norm: LASSO (Tibshirani, ’96), Basis Pursuit (Chen, ’98), Compressed Sensing (Donoho, ’06, Cand`es et al., ’06) 

• Adaptive LASSO (Zou, ’06) 

• Exponential approximation (Mangasarian, ’96) 

• Log-sum penalty (Cand`es, ’08) 

• Smoothly Clipped Absolute Deviation (SCAD) (Fan, Liu, ’01) and Minimax Concave Penalty (MCP) (Zhang, ’10 

$\ell _ { p }$ “norms”, $p < 1$ (Chartrand, ’07, Foucart, Lai, ’09) 

• Beautiful review (Soubies, Blanc-F´eraud, Aubert, ’17) Which approximation should we use? 

## Continuous relaxation idea

Think of a diferent idea for solving the problem: 

$$
\frac {1}{2} \| A x - d \| ^ {2} + \lambda \| x \| _ {0} \quad \Longrightarrow \quad \frac {1}{2} \| A x - d \| ^ {2} + \sum_ {i = 1} ^ {n} \phi_ {i} (x _ {i})
$$

Idea: use continuous and separable functions $\phi _ { j } \mathopen { } \mathclose \bgroup \left( x _ { j } \aftergroup \egroup \right)$ (convex and non-convex) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/f82a586e8cafbe09e172399a1295bf9ce86ab61c84c302bff9f16899fcf2e6f5.jpg)



Which approximation should we use?


Think of a diferent idea for solving the problem: 

$$
\frac {1}{2} \| A x - d \| ^ {2} + \lambda \| x \| _ {0} \quad \Longrightarrow \quad \frac {1}{2} \| A x - d \| ^ {2} + \sum_ {i = 1} ^ {n} \phi_ {i} (x _ {i})
$$

Idea: use continuous and separable functions $\phi _ { j } \mathopen { } \mathclose \bgroup \left( x _ { j } \aftergroup \egroup \right)$ (convex and non-convex) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/6e630ef4b26c505c39d80ce43e322b712278514aadb5787e333658961e7480a1.jpg)



Which approximation should we use?


# Continuous relaxations

Exactness 

## What is a good relaxation?

$$
G _ {\ell_ {0}} (x) = \frac {1}{2} \| A x - d \| ^ {2} + \lambda \| x \| _ {0} \quad \Longrightarrow \quad \tilde {G} (x) := \frac {1}{2} \| A x - d \| ^ {2} + \sum_ {i = 1} ^ {n} \phi_ {i} (x _ {i})
$$

## Good (exact) relaxation

• $G _ { \ell _ { 0 } } ( x )$ and $\tilde { G } ( x )$ have the same global minimisers: 

$$
\underset {x \in \mathbb {R} ^ {n}} {\arg \min} G _ {\ell_ {0}} (x) = \underset {x \in \mathbb {R} ^ {n}} {\arg \min} \tilde {G} (x),
$$

$$
(g l o b a l)\tag{P1}
$$

$\tilde { G } ( x )$ has “less” local minimisers than $G _ { \ell _ { 0 } } ( x )$ : 

$$
x ^ {*} \text {   minimiser   of   } \tilde {G} \Rightarrow x ^ {*} \text {   minimiser   of   } G _ {\ell_ {0}}\tag{P2}
$$

## The continuous exact $i \langle j |$ relaxation (CEL0) penalty

In Soubies, Aubert, Blanc-F´eraud, ’15-’17 a particular choice of $\phi : \mathbb { R } \to \mathbb { R } _ { + }$ is studied. By convex conjugation, the penalty removing most of the local minimisers is: 

$$
\phi_ {C E L 0} (\| a _ {i} \|, \lambda , x) = \lambda - \frac {\| a _ {i} \| ^ {2}}{2} \left(| x | - \frac {\sqrt {2 \lambda}}{\| a _ {i} \|}\right) ^ {2} \mathbf {1} _ {\left\{| x | \leq \frac {\sqrt {2 \lambda}}{\| a _ {i} \|} \right\}}
$$

where $\mathbf { 1 } _ { C } ( x ) = 1 \mathrm { ~ i f ~ } x \in C$ and $\mathbf { 1 } _ { C } ( x ) = 0$ otherwise. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/a01424c9a7c4e036f8a61cca4e7d226c0285a8f0acbf67502cf2ac29e6c92e9a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/ddba00e119be0c0b479fecc54fd68b21b3848ffd5e14937dca7ffe605e105c7d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/71bee55fd722013fe14e59de857e3507ce51769d954d446e3ab1ac95e37e4c3d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/ac265827ef5efcfa89125b3b14391ce811af3c4c0b2af39b9d053bb849736632.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/87fe4f147a0336359da9c88e3dec00e908271198de440ffc9ccacfbda5341264.jpg)



Examples of penalties for which (P1) (top) or (P1) and (P2) (bottom) hold for $a = 0 . 5 ,$ , λ = 1 and $d = 1 . 8$ in the 1D case.


$$
\boxed {G _ {C E L 0} (x) := \frac {1}{2} \| A x - d \| ^ {2} + \underbrace {\sum_ {i = 1} ^ {n} \phi_ {C E L 0} (\| a _ {i} \| , \lambda , x _ {i})} _ {\Phi_ {C E L 0} :=}}
$$

where: $\begin{array} { r } { \phi _ { C E L 0 } ( \| a _ { i } \| , \lambda , x ) = \lambda - \frac { \| a _ { i } \| ^ { 2 } } { 2 } \left( | x | - \frac { \sqrt { 2 \lambda } } { \| a _ { i } \| } \right) ^ { 2 } \mathbf { 1 } _ { \left\{ | x | \leq \frac { \sqrt { 2 \lambda } } { \| a _ { i } \| } \right\} } } \end{array}$ 

Properties of $G _ { C E L 0 }$ : 

• Inferior limit of all functions satisfying (P1) and (P2) 

• Convex envelope of $G _ { \ell _ { 0 } }$ if A diagonal or $A ^ { T } A = s \mathsf { I d } , s > 0$ 

• Continuous 

• Non convex for general operators A 

• Convexity w.r.t. each component $x _ { i } , i = 1 , \ldots , n$ 

Thanks to its continuity we can resort to nonsmooth, nonconvex algorithms such as, e.g., forward-backward and majorisation-minimisation (MM) algorithms (e.g., iterative reweighted $\ell _ { 1 }$ Ochs et al., ’15). 

## Understanding the relaxation

1D example: $\begin{array} { r } { G _ { \ell _ { 0 } } ( x ) : = \frac { 1 } { 2 } ( a x - y ) ^ { 2 } + \lambda | x | _ { 0 } \mathrm { ~ f o r ~ } a , \lambda > 0 } \end{array}$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/8043e8ea2b5d575fdecf6b71b495e6ef809c9894b5a54a80bd8bb0adf36d167c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/9644d27a8d0b1ecb6d6712118f4300c076ac674a107f12740c2d8aa03b472f27.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/513b6d1546da4e3fc0ddddd9868ccfb24024aeca524402a6c828df00a7d3a95d.jpg)



Blue lines: plots of $G _ { \ell _ { 0 } }$ for diferent values of d (note discontinuity in $x = 0 )$ . Red lines: plots of $G _ { C E L 0 }$ (convex biconjugate).


In 1D $G _ { C E L 0 }$ is always a convex function, in the multi-dimensional case it depends on the operator A. Generally, it is non-convex with convex 1D restrictions. 

## Forward-backward splitting for $y _ { 2 } = \textcircled { 9 } \textcircled { 1 } \textcircled { 1 }$

Iterate for $k \geq 0$ and $\tau \in ( 0 , \frac { 1 } { \Vert A \Vert ^ { 2 } } )$ 

$$
x _ {k + 1} \in \operatorname{prox} _ {\tau \Phi_ {C E L 0}} \left(x _ {k} - \tau A ^ {T} (A x _ {k} - d)\right)
$$

where, by separability, we can look at the prox of the 1D components: 

$$
\operatorname{prox} _ {\tau \phi_ {C E L 0} (a, \lambda ; \cdot)} (u) = \left\{ \begin{array}{l l} \operatorname{sign} (u) \min \left(| u |, (| u | - \sqrt {2 \lambda} \tau a) _ {+} / (1 - a ^ {2} \tau)\right) & \text {if} a ^ {2} \tau <   1 \\ u \mathbf {1} _ {| u | > \sqrt {2 \tau \lambda}} + \{0, u \} \mathbf {1} _ {| u | = \sqrt {2 \tau \lambda}} & \text {if} a ^ {2} \tau \geq 1 \end{array} \right.
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/bccedebcb28bcbcaa141d5612953f620e26fe42ba4cbecb8c6b87145f8907b48.jpg)


Dependence of $\phi _ { C E L 0 }$ on $a = \left| \left| a _ { i } \right| \right|$ at component $u = x _ { i }$ 

Convergence to a critical point under Kurdyka- Lojaseiwicz (KL) property (Attouch et al, ’13) 

Iteratively reweighted algorithms 

$$
\min _ {x \in \mathbb {R} ^ {n}} F (x) := f (x) + g (x)
$$

for g proper, l.s.c. and bounded from below but generally non-convex 

## Majorisation-minimisation technique

Construct a sequence of easier (convex) functions majorising $F$ and minimise them to simplify the problem. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/43b84e5bfdf55c12415cd1e9fc0cf5c296a1fb36946fb9af5dafc45804d5c9a0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/09e71e78fe66070a27b560edca4b198ddcd5c256411ecd9d9d7c45e967cd176b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/334cd68a4eab787e247ded1ebded93f21f28027484a6fc1f7a04d00030b221fa.jpg)



Minimisation of a non-convex function (red) using MM techniques. Non-convexity induced by $g ( x ) = \log ( 1 + 2 | x | )$ . Majorant functions in blue.


```txt
Majorisation-minimisation technique 
```

$\min_{x\in\mathbb{R}^{n}} F(x):=f(x)+g(x)$ 

for g proper, l.s.c. and bounded from below but generally non-convex 

Construct a sequence of easier (convex) functions majorising $F$ and minimise them to simplify the problem. 

Pseudocode: general idea for MM algorithms
Input: $x_{0} \in R^{n}$ .
while not converging do
    Build a majorising function $M_{x_{k}} : R^{n} \to R$ such that:
    - $\forall x \in \mathbb{R}^{n}: F(x) \leq M_{x_{k}}(x)$ - $F(x_{k}) = M_{x_{k}}(x_{k})$ - $M_{x_{k}}(x_{k}) \in \Gamma_{0}(\mathbb{R}^{n})$ Define $x_{k+1} \in \arg\min_{x} M_{x_{k}}(x)$ end while 

## MM approaches

Several approaches for building such functions: 

• Iterative least-squares (IRLS) (Daubechies et al. ’10, Gorodnitsky, Rao, ’97): 

$$
M _ {x _ {k}} (x) = \sum (w _ {x _ {k}}) _ {i} x _ {i} ^ {2}
$$

• MM approaches for inverse problems (Chouzenoux et al., $^ { \prime } 1 0 \dots )$ 

• Iterative reweighted $\ell _ { 1 }$ algorithms: better suited to construct majorants of functions which are not suficiently smooth of the form: 

$$
F (x) = \frac {1}{2} \| A x - d \| ^ {2} + \sum \phi (| x _ {i} |)
$$

with $\phi : \mathbb { R } _ { + } \to \mathbb { R }$ continuous, concave and non-decreasing (Ochs et al, ’15.) 

Algorithm: IR $\ell_{1}$ (Ochs et al, '15)

Input: $x_{0} \in R^{n}$ .

while not converging do $(w_{x_{k}})_{i} \in \partial^{+}\phi_{i}(|(x_{k})_{i}|)$ $x_{k+1} \in \arg\min_{x} \frac{1}{2} \|Ax - d\|^{2} + \sum_{i=1}^{n}(w_{x_{k}})_{i}|x_{i}| \to solve with FISTA$ end while 

Weights can be computed in an explicit form: 

$$
(w _ {x _ {k}}) _ {i} := \left\{ \begin{array}{l l} \sqrt {2 \lambda} \| a _ {i} \| - \| a _ {i} \| ^ {2} | (x _ {k}) _ {i} | & \quad \text { if } 0 \leq | (x _ {k}) _ {i} | <   \sqrt {2 \lambda} / \| a _ {i} \| \\ 0 & \quad \| (x _ {k}) _ {i} | \geq \sqrt {2 \lambda} / \| a _ {i} \| \end{array} \right.
$$

Convergence of $\mathsf { I R } \ell _ { 1 }$ to critical points can be proved for general class of functions satisfying the so-called Kurdyka- Lojasiewicz property (Ochs et al, ’15). 

Application to super-resolution microscopy 

## Super-resolution microscopy

Spatial resolution is limited by light difraction phenomena. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/cd3725dcc769798ba2841b2198e23fc58db3edc378d1848f6f482ca9c96e6cfb.jpg)


Rayleigh criterion 

$$
d = \frac {0 . 6 1 \lambda}{N A} \approx 2 0 0 n m
$$

$\lambda \colon$ emission wavelength 

$N A { : }$ microscope numerical aperture 

Point Spread Function: Gaussian, Airy disk. . . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/be45353ff9bfce25c636c3456c7480aba56fbed5ef5004e872f4279a5aa06365.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/069f32786ffd0823401827f1f48aadb8eae572eb00038040cb734995af92ba41.jpg)



Resolved


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/688b2031fe48f17412d7f0a94ce252e0df958aefc7613d78ba6a63aecba96465.jpg)



Unresolved



Rayleigh Criterion


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/0983c9025b80fa055115aad73e2ca590abd9a56d79e4ab13cb4eb74053886b87.jpg)


## Super-resolution microscopy

Spatial resolution is limited by light difraction phenomena. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/9cebc279f6637e3d8ef0bcc03a8c42752e1d36f27a5fa263f4aa3dc36c2ceb46.jpg)



Rayleigh criterion


$$
d = \frac {0 . 6 1 \lambda}{N A} \approx 2 0 0 n m
$$

• λ: emission wavelength 

• NA: microscope numerical aperture 

Point Spread Function: Gaussian, Airy disk. . . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/b98d5682359e4c89da69cc01f054a3db947f83043d87721ea552b67d66d49abd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/d530fb24cb0e301f5debef9b9c6bd7089651f13523c6dedd200b6e29e8d14744.jpg)



FWHM = 228nm


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/1c29a986ebaf819e891f80ed478439630dac1688cd5823a8449db4941641853b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/81612879598afd9eb55e5d4fb929ac641cb2d86cb3e3b66d455d6361ff2f07f4.jpg)


## Discrete mathematical modelling

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/9cad31e0fa02d809fbfdde123465dbb5a992c969e3af36c370b0b4932871d900.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/29af89971618709cc9d6e99d4f97628e5691b9d4d9af70a698bb97a0d0478352.jpg)


## Image formation model

$$
\boldsymbol {Y} = \mathcal {P} (M _ {q} (H (\boldsymbol {X})) + \boldsymbol {B}) + \boldsymbol {N}
$$

• $\boldsymbol { Y } \in \mathbb { R } ^ { N \times N }$ : LR acquisition 

• $\pmb { x } \in \mathbb { R } ^ { L \times L }$ : HR image (L = qN, q ∈ <sup>N</sup>) 

• $\mathcal { P } ( \cdot ) \colon$ Poisson r.v. 

• $M _ { q } \in \mathbb { R } ^ { N \times L }$ : down-sampling matrix 

• $H \in \mathbb { R } ^ { N \times N }$ : convolution matrix 

• N: additive white Gaussian noise 

• B: background 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/56b43b16b195abe06a86dc73f4ec631e93c60a6c67ef60cefc94a8849cf38471.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/c619bca6d0c3584836a9394b7dcbf4f34785484dbca196e37a1dc3d2b98e88ae.jpg)


$$
q = 4
$$

## State-of-the-art methods in fluorescence microscopy

## Key idea

In microscopy imaging, the experimental setup and the sample preparation can be used to ‘sparsify’ the measurements. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/1dffb23d727a186b52e39280ac10378288d033708be3f6b37b9b7e3bddcd7aa2.jpg)



Fluorescence microscopy


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/671dd75748e6cf7e634f336e624b084ce2e72cf2ad8aa708536c4dd0e011f05a.jpg)



Absorption/emission diagram


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/3c308294a6133c8806012f2f6b486efcd68acfe87d88feb01f342c0ce278ddd9.jpg)



Fluorescent molecule



Nobel prize in chemistry in 2008.


## State-of-the-art methods in fluorescence microscopy

## Key idea

In microscopy imaging, the experimental setup and the sample preparation can be used to ‘sparsify’ the measurements. 

Example: Single Molecule Localization Microscopy (Betzig, Zhuang, Hess, ’06, Rust, Bates, Zhuang, ’06) 

- Specific fluorescent molecules activating with low probability in a sequential way 

- Improved sparsity! 

http://zeiss-campus.magnet. fsu.edu/ 

$$
\mathbf {y} _ {t} = \mathcal {P} (\Psi \mathbf {x} _ {t} + \mathbf {b}) + \mathbf {n} _ {t}, \quad \Psi := M _ {q} H, \quad \mathbf {n} _ {t} \sim \mathcal {N} (0, \sigma^ {2} I d), \quad \bar {\mathbf {y}} := \sum_ {t = 1} ^ {T} \mathbf {y} _ {t} / T
$$

To incorporate signal-dependence (modelling Poisson photon counting) in Lazzaretti, Calatroni, Estatico, ’21 we considered a weighted $\ell _ { 2 }$ fidelity term. 

Weighted- $\ell _ { 2 } - \ell _ { 0 }$ problem 

$$
\boldsymbol {x} ^ {*} \in \underset {\boldsymbol {x} \in \mathbb {R} ^ {L ^ {2}}} {\arg \min} \left\{G _ {w \ell_ {0}} (\boldsymbol {x}) := \frac {1}{2} \sum_ {j = 1} ^ {N ^ {2}} \frac {((\Psi \boldsymbol {x}) _ {j} - y _ {j} - b _ {j}) ^ {2}}{y _ {j} + b _ {j}} + \lambda \| \boldsymbol {x} \| _ {0} + \iota_ {\geq 0} (\boldsymbol {x}) \right\}, \quad \lambda > 0
$$

Theorem 

• If $\Psi ^ { \tau } W \Psi = D ^ { 2 }$ with $D = \mathsf { d i a g } ( \| \psi _ { i } \| _ { W } ) \in \mathbb { R } ^ { L ^ { 2 } \times L ^ { 2 } }$ , then $G _ { \mathrm { w C E L } 0 } = G _ { \mathrm { w } \ell _ { 0 } } ^ { \ast \ast }$ 

• arg min $G _ { \mathrm { w C E L 0 } } = \mathsf { a r g }$ min $G _ { w \ell _ { 0 } }$ (same global minimisers) 

• x minimiser of $G _ { \mathrm { w C E L 0 } } \Rightarrow x$ minimiser of $G _ { w \ell _ { 0 } }$ (less local minimisers). 

+ Minimisation with $\mathsf { I R } \ell _ { 1 }$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/0285be7db9fb36daf6b6627d0d53ccb187c65d782fc18bf946ac300a1764ac47.jpg)



GT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/fe952613772dfe27d995d3d2ea60646406cce8d9d11e1c8a45f174808b2697a6.jpg)



One frame


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/9b4a20a5e16fd90174ca7d20604368c76d6c2dd1e43fc069e446a9ad1bc35ec4.jpg)



y


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/4d83f6449c078d27396bfdad9e5b93fa4689ed0c2258c771d7f7c2e25486713a.jpg)



CEL0


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/791a18005863ee781b454bb04a392b904b4c0c48b6d414de2b95279a10111ac1.jpg)



wCEL0


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/d6440c69d05c3460b6406a13c2cde38a38c84639ec5586838610b990f34a080f.jpg)



DeepStorm


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/de93be412854fbe347d649ca2f9bc89b4ce155aca983503ce863bbde9539fe14.jpg)



GT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/227a80fb662a8caebeff6e434d8ca839670123ab9b186a9da69665773a698269.jpg)



One frame


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/5532106fa81ca00308d1b78f22940fd88d69f1e36b842a91b88d2d543ffedc90.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/b964ad029c08417aec5ed386586044f2ae559c51725274c79248cdc6c1ae596c.jpg)



CEL0


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/77c7a54568f62a007edac075680c54220449290251b320a940d5d3bc195fe521.jpg)



wCEL0


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/07af4a44180d01b480dce21703df368c20e753901a0bbec40763fe2d5aa51a4b.jpg)



DeepStorm


We focused on models and algorithms tackling the $\ell _ { 2 } - \ell _ { 0 }$ minimisation problem. 

• NP-hardness is avoided by alternative formulations 

• Greedy approaches provide interesting results, at the price of increased complexity 

• Continuous relaxations (both convex and non-convex) ease the problem 

• CEL0 is the “best” (liminf) continuous, non-convex relaxation, and it is exact. 

• A MM strategy such as $\mathsf { I R } \ell _ { 1 }$ can be used. Fast convex optimisation is here essential for solving inner problems with high precision. 

• Application areas are vast: inverse problems in imaging, vision, variable selection in machine learning. . . 

## Interested in a PostDoc (or PhD) in optimisation?

## ANR AGENCE NATIONALE DE LA RECHERCHE

Task-adaptive bilevel learning of flexible statistical models for imaging and vision (2023-2027) 

• 2-year post-doctoral position (open) 

• 1 PhD position (from October 2023) 

## Announcement II: SSVM 2023

• What? IX conference on Scale Space and Variational Methods in Compute Vision (SSVM). 

• Where? Hotel Flamingo, Santa Margherita di Pula, Sardegna, IT. 

• When? May 21-25 2023 

• Who? Giunta Gruppo UMI MIVA + G. Rodriguez (local organiser) 

• Why Oral + poster session of selected papers (published in Springer LNCS) 

Website: SSVM 2023 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/e822a964f3669929846bca8d9a3ad62b0970f319472cfd4b04bb6ca8a0c0e0de.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/f1ee26c9-06f6-4ae7-ab77-c280eff6e183/50844c7d3246281035f2d50e4a78ff5f87ac2e48b6a3832ac3f94382bb9016e6.jpg)



NEW DEADLINE for submissions: January 30 2023


## Questions?

calatroni@i3s.unice.fr 