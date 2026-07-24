## Lecture 1: Convex smooth optimisation

Luca Calatroni CR CNRS, Laboratoire I3S CNRS, UCA, Inria SAM, France 

MIVA ERASMUS BIP PhD winter school Advanced methods for mathematical image analysis University of Bologna, IT January 18-20 2022 

## Table of contents

## 1. Introduction

2. Notation, preliminaries & basic notions Convexity, strong convexity Lower semi-continuity & coercivity Diferentiability and L-smoothness 

3. Smooth optimisation algorithms Gradient descent algorithm Convergence proof under PL condition Motivation for accelerated algorithms 

4. Accelerated smooth optimisation algorithms Nesterov acceleration of GD 

<table><tr><td></td><td>WEDNESDAY 18/01</td><td>THURSDAY 19/01</td><td>FRIDAY 20/01</td></tr><tr><td>08:00</td><td></td><td></td><td></td></tr><tr><td>09:00</td><td></td><td></td><td></td></tr><tr><td>10:00</td><td></td><td></td><td></td></tr><tr><td>11:00</td><td></td><td></td><td></td></tr><tr><td>12:00</td><td></td><td></td><td></td></tr><tr><td>13:00</td><td>Lunch</td><td>Lunch</td><td>Lunch</td></tr><tr><td>14:30</td><td>Comp. Imaging Lab</td><td>EXERCISES</td><td>Comp. Imaging LAB</td></tr><tr><td>15:30</td><td>LAB</td><td>LAB</td><td>EXERCISES</td></tr><tr><td>16:30</td><td>SEMINAR Automotive</td><td>SEMINAR Industrial</td><td>SEMINAR Health</td></tr><tr><td>17:30</td><td colspan="3"></td></tr><tr><td></td><td>Prof. L. Calatroni</td><td rowspan="2" colspan="2">Social Dinner</td></tr><tr><td></td><td>Prof. O. Öktem</td></tr></table>

## Introduction

## Motivation

Goal: providing theoretical $\&$ practical tools (i.e. algorithms) for solving 

$$
\min _ {x \in \mathbb {R} ^ {n}} F (x)
$$

for a functional $F : \mathbb { R } ^ { n }  \overline { { \mathbb { R } } }$ with suitable properties. 

$F$ is smooth → gradient descent $\&$ variants (this lecture) 

$F : = f + g ,$ f smooth $\& \ g$ non-smooth → proximal-gradient algorithms $\&$ variants (next lecture) 

$F : = f + \| x \| _ { 0 }$ with $f$ smooth → which algorithms? (last lecture) 

Such minimisation problems often appears in many contexts: 

Inverse problems in signal/image processing: image reconstruction, variable/parameter selection, compressed sensing. . . . 

• Statistical/machine learning: empirical risk minimisation, regression. . . 

• Optimisation per se: analysis/implementation of fast algorithms for solving large-scale problems. . . 

## Framework: optimisation for inverse problems in imaging

$$
y \in \mathbb {R} ^ {m}, A \in \mathbb {R} ^ {m \times n}
$$

$$
x \in \mathbb {R} ^ {n}
$$

$$
y = \mathcal {T} (A x)
$$

where $m \leq n$ and $\mathcal { T } : \mathbb { R } ^ { m }  \mathbb { R } ^ { m }$ models noise degradation. 

• Image restoration (denoising, deconvolution, super-resolution) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/06b48f127a5f97fe25cdfde8dd34c8ec6662f6f46788bb00f1883f05c855cea2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/4fc1bab3777edb5f909b6476d916c080b8cbf4e873c5e856b6b2ebf5ec5b1887.jpg)


Acquisition (Convolution + Noise) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/a53b9156549341aa2a09c0b74dde5f031e6b528f008b1ea14bd1392746c58b3c.jpg)


## Framework: optimisation for inverse problems in imaging

Given $y \in \mathbb { R } ^ { m } , A \in \mathbb { R } ^ { m \times n }$ find $\boldsymbol { x } \in \mathbb { R } ^ { n }$ s.t. $y = { \mathcal { T } } ( A x )$ 

where $m \leq n$ and $\mathcal { T } : \mathbb { R } ^ { m }  \mathbb { R } ^ { m }$ models noise degradation. 

• Image restoration (denoising, deconvolution, super-resolution) 

• Image reconstruction (e.g., medical imaging) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/4dd2f4a0e09f9383e6478fdd49b4f75a3cfe8e835354ad32e568ec88078b33d2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/ee5f04f075fe9b74f676a2d6535022bb423cd6d12317bbb7d87673689778926c.jpg)


## Framework: optimisation for inverse problems in imaging

Given $y \in \mathbb { R } ^ { m } , A \in \mathbb { R } ^ { m \times n }$ find $\boldsymbol { x } \in \mathbb { R } ^ { n }$ s.t. $y = { \mathcal { T } } ( A x )$ 

where $m \leq n$ and $\mathcal { T } : \mathbb { R } ^ { m }  \mathbb { R } ^ { m }$ models noise degradation. 

• Image restoration (denoising, deconvolution, super-resolution) 

• Image reconstruction (e.g., medical imaging) 

• Dictionary representation (data analysis, vision) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/c9a82c9979a86d989d6d3ffae8b0d69a95d9c91ab2cdfe986897874fd68384c2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/f09962d79a3ea6cc5d2b4c0f6ce81d5fbe1db0074908ab4ff14c7b6819ba5c05.jpg)



Test Example


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/ce461047a34f706585428d3d7c287aaf6e56b576bb226f914cdcb0d7f6ee5237.jpg)


## Framework: optimisation for inverse problems in imaging

$$
\text { Given   } y \in \mathbb {R} ^ {m}, A \in \mathbb {R} ^ {m \times n} \quad \text { find } \quad x \in \mathbb {R} ^ {n} \quad \text { s.t. } \quad y = \mathcal {T} (A x)
$$

where $m \leq n$ and $\mathcal { T } : \mathbb { R } ^ { m }  \mathbb { R } ^ { m }$ models noise degradation. 

• Image restoration (denoising, deconvolution, super-resolution) 

• Image reconstruction (e.g., medical imaging) 

• Dictionary representation (data analysis, vision) 

. . . “naive inversion” not possible for $y = A x + n , n \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } | \mathsf { d } )$ 

$$
x = A ^ {- 1} (y - n)
$$

## Bad positioning of inverse filtering

$$
y = A x + n
$$

Inverse filtering approach: 

$$
x = A ^ {- 1} y = A ^ {- 1} (A x + n) = x + A ^ {- 1} n
$$

Amplification of the noise if $A ^ { - 1 }$ is bad conditioned! Need of regularisation! Find an estimate $\mathbb { R } ^ { n } \ni x ^ { * } \approx x$ by solving 

$$
x ^ {*} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} F (x) := f (x) + g (x)
$$

• f is the data fidelity term, it relates to noise statistics 

$\boldsymbol { g }$ is the regularisation term, it encodes a priori information expected on the desired solution 

## Variational regularisation: Bayesian motivation

Following a Bayesian/MAP approach consider: 

$$
P (y | A x; \theta_ {f}) \quad (\text { likelihood }), \qquad P (x; \theta_ {g}) \quad (\text { prior })
$$

with $\theta _ { f } , \theta _ { g } > 0$ hyperparameters of the distributions. By Bayes’ theorem: 

$$
\begin{array}{l} x ^ {*} \in \underset {x} {\arg \max} P (x | y) = \underset {x} {\arg \max} \frac {P (y | A x ; \theta_ {f}) P (x ; \theta_ {g})}{P (y)} \\ \Leftrightarrow x ^ {*} \in \underset {x} {\arg \min} - \ln (P (x | y)) = \underset {x} {\arg \min} - \ln (P (y | A x; \theta_ {f})) - \ln (P (x; \theta_ {g})) + \underline {{\ln (P (y))}} \end{array}
$$

Now, if $P ( x ; \theta _ { g } ) = e ^ { - \theta _ { g } g ( x ) }$ and $P ( y | A x ; \theta _ { f } ) = e ^ { - \theta _ { f } f ( x ) }$ , then: 

$$
x ^ {*} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} f (x) + \lambda g (x), \qquad \lambda := \theta_ {g} / \theta_ {f}
$$

Note: incorporate the parameter α in either of the two functions, e.g. $g ( x ) : = \lambda g ( x )$ 

## Exemplar problems: smooth optimisation

$$
y = A x + b
$$

• Generalised Tikhonov $\textstyle n \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } | \ d )$ (Gaussian noise) and assume x is smooth in some sense (e.g., in terms of an operator $\boldsymbol { L } \in \mathbb { R } ^ { N \times n } )$ 

$$
x ^ {*} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} \frac {1}{2} \| A x - y \| ^ {2} + \lambda \| L x \| ^ {2}
$$

Examples: $L = \mathsf { I d } \in \mathbb { R } ^ { n \times n } , L = D \in \mathbb { R } ^ { 2 n \times n }$ (discrete gradient) . . . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/5a15ddcd6b447e55ef6aadaa8b98015587458e5c5d6e9b761fdcac4ffa33617c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/41c68f1eba4e5aeaaa44772665fca89e259b054c467c70ec3e40565e2d121862.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/a01e66b39cb7dafab583aca590d8ea58ca5a2b0ee1b393f64f0133b192ec65cd.jpg)



Parameter selection for $\ell _ { 2 } - \ell _ { 2 }$ single-image super-resolution, $A = S H$ , where S is a decimation operator (Pragliola, Calatroni, Lanza, Sgallari, ’21-’22)


## Exemplar problems: non-smooth optimisation

Assume for simplicity additive white Gaussian noise $\begin{array} { r } {  f ( x ) = \frac { 1 } { 2 } \| A x - y \| ^ { 2 } } \end{array}$ 

• Sparsity (Donoho et al., Cand`es, Romberg, Tao, ’06): sparse recovery: 

$$
x ^ {*} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min}   \frac {1}{2} \| A x - y \| ^ {2} + \lambda \| x \| _ {1}
$$

Analysis approach: sparse representation of x in some overcomplete basis (e.g., wavelets, Mallat, ’89) represented by $W \in \mathbb { R } ^ { N \times n }$ 

$$
x ^ {*} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} \frac {1}{2} \| A x - y \| ^ {2} + \lambda \| W x \| _ {1}
$$

• Total variation reconstruction: “few gradients” for removing noise oscillation and preserving edges (Rudin, Osher, Fatemi, ’92): 

$$
x ^ {*} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} \frac {1}{2} \| A x - y \| ^ {2} + \lambda \| D x \| _ {2, 1}
$$

with $\| D x \| _ { 2 , 1 } = \sum _ { i = 1 } ^ { n } \sqrt { ( D _ { h } x ) _ { i } ^ { 2 } + ( D _ { v } x ) _ { i } ^ { 2 } }$ and Dx is the discrete image gradient. 

## Exemplar problems: non-smooth optimisation (continuation)

It helps in dealing with admissibility constraints: 

$$
x ^ {*} \in \underset {x \in C} {\arg \min} \frac {1}{2} \| A x - y \| ^ {2}
$$

with $\begin{array} { r } { C : = \bigcap _ { m = 1 } ^ { M } C _ { m } } \end{array}$ and $C _ { m } \subset \mathbb { R } ^ { n }$ 

• Non-negativity constraint: $x \geq 0 , C : = \{ x \geq 0 \}$ 

• Box constraint: $x \in [ a , b ] = : C$ 

• . . . 

How to encode it into a variational formulation? 

Using the indicator function $\iota : \mathbb { R } ^ { n }  \{ 0 , + \infty \}$ 

$$
\iota_ {C _ {m}} (x) := \left\{ \begin{array}{l l} 0 & \quad \text { if } x \in C _ {m} \\ + \infty & \quad \text { if } x \notin C _ {m} \end{array} \right.
$$

$$
x ^ {*} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} \frac {1}{2} \| A x - y \| ^ {2} + \sum_ {m = 1} ^ {M} \iota_ {C _ {m}} (x)
$$

## Exemplar problems: $1 5 2 \%$ optimisation

Arising, e.g., in sparse dictionary representation problems 

$$
y = A x + n
$$

where $y \in \mathbb { R } ^ { m } , A \in \mathbb { R } ^ { m \times n }$ and $x \in \mathbb { R } ^ { n }$ and $m \ll n$ . Undetermined system! 

To minimise the number of entries of solutions, the natural choice is to consider: 

$$
x ^ {*} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} \frac {1}{2} \| A x - y \| ^ {2} + \lambda \| x \| _ {0} \quad \text { or } \quad x ^ {*} \in \underset {x: \| x \| _ {0} \leq K} {\arg \min} \frac {1}{2} \| A x - y \| ^ {2}
$$

$$
\| x \| _ {0} := \# \left\{x _ {i}, i = 1, \dots , N: x _ {i} \neq 0 \right\}
$$

Some standard reference books/surveys: 



R. Tyller Rockafeller, Convex Analysis, Princeton University Press, 1970. 





S. Boyd, L. Vandenberghe, Convex Optimization, Cambridge University Press, 2004. 





N. Parikh, S. Boyd, Proximal Algorithms, Foundations and Trends in Optimization, 2013. 





A. Beck, First-order methods in optimization, Volume 25, MOS-SIAM series on Optimization, 2017. 





A. Chambolle, T. Pock, An introduction to continuous optimization for imaging, Acta Numerica, 2016 





S. Salzo, S. Villa, Proximal Gradient Methods for Machine Learning and Imaging, Handbook on Harmonic and Applied Analysis, Applied and Numerical Harmonic Analysis, 2021. 



$$
x ^ {*} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} F (x) := f (x) + g (x)
$$

Often the solution $x ^ { * }$ cannot be expressed in closed form. We consider eficient iterative solvers for its computation (especially in large scale context!) 

• Avoid inversion $A ^ { - 1 } \left( 1 \ll m \leq n \right)$ 

• How to exploit the mathematical structure of the functions involved? 

• How to handle constraints? 

• How to speed up the eficiency of a first-order algorithm? 

• What can be said in the non-convex case? 

Notation, preliminaries & basic notions 

## Notation

$( X , \langle v , w \rangle ) = ( \mathbb R ^ { n } , v ^ { T } w )$ with Euclidean norm $\| \cdot \|$ as reference Hilbert space. Extensions to general Hilbert setting straightforward. 

• <sup>R</sup> := <sup>R</sup> ∪ {+∞}, <sup>R</sup> := {α ∈ <sup>R</sup> : α ≥ 0}, <sup>R</sup> := {α ∈ <sup>R</sup> : α > 0} 

• Closed ball of radius $\delta > 0$ in $x \in X$ : 

$$
B _ {\delta} (x) = \{y \in X: \| y - x \| \leq \delta \}
$$

• Convex set $C \subset X$ 

$$
(\forall x, y \in C) \quad \forall \alpha \in [ 0, 1 ] \quad \alpha x + (1 - \alpha) y \in C
$$

• Epigraph of a function $f : \mathbb { R } \to { \overline { { \mathbb { R } } } } ;$ 

$$
\operatorname{epi} (f) = \{(x, t) \in X \times \mathbb {R}: f (x) \leq t \}
$$

## Proper functions

Minimal property to have well-defined minimisation problems. 

Definition (proper function)  
A function $F: \mathbb{R}^n \to \overline{\mathbb{R}}$ is said proper iff $\exists x \in \mathbb{R}^n$ such that $F(x) \neq +\infty$ .  
We define $\mathcal{P} := \{F : \mathbb{R}^n \to \overline{\mathbb{R}} : F \text{ is proper}\}$ and $\operatorname{dom}(F) := \{x \in \mathbb{R}^n : F(x) < +\infty\}$ 

Clearly, $F \in { \mathcal { P } } \Leftrightarrow \mathsf { d o m } ( F ) \neq \emptyset$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/5d2c829179e92463dae75c71d8f2b7cabca7699e1247407b95f0acf5b480db13.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/134c5c38b1b7a135c2a66cddc26f2a07e0d315be884c4b11f452e50d44993c02.jpg)


## Global/local minimisers

For $F \in { \mathcal { P } }$ , recall: 

• global minimiser: $x ^ { * } \in \mathbb { R } ^ { n } \colon F ( x ^ { * } ) \leq F ( x )$ for every $x \in \mathbb { R } ^ { n }$ 

• local minimiser: $x ^ { \ast } \in \mathbb { R } ^ { n }$ : there exists $\delta > 0$ and a neighbourhood $B _ { \delta } ( x ^ { * } )$ such that $F ( x ^ { * } ) \leq F ( x )$ for every $x \in B _ { \delta } ( x ^ { * } )$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/5519176b2a205206a64e77ce7dfe7eb5ae4adbea6160782f5236fa9ebb79932d.jpg)


$$
\min _ {x \in \mathbb {R} ^ {n}} F (x) \quad \text { VS } \quad \underset {x \in \mathbb {R} ^ {n}} {\arg \min} F (x)
$$

For $F \in { \mathcal { P } }$ , recall: 

• global minimiser: $x ^ { * } \in \mathbb { R } ^ { n } \colon F ( x ^ { * } ) \leq F ( x )$ for every $\ b { x } \in \mathbb { R } ^ { n }$ 

• local minimiser: $x ^ { \ast } \in \mathbb { R } ^ { n }$ : there exists $\delta > 0$ and a neighbourhood $B _ { \delta } ( x ^ { * } )$ such that $F ( x ^ { * } ) \leq F ( x )$ for every $\boldsymbol { x } \in B _ { \delta } ( \boldsymbol { x } ^ { * } )$ . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/f6736809bfff13f12de1229153ce381e2933608972e121a4c80d69e82934600c.jpg)


$$
\min _ {x \in \mathbb {R} ^ {n}} F (x) \quad \text { VS } \quad \underset {x \in \mathbb {R} ^ {n}} {\arg \min} F (x)
$$

## Definition (set of minimisers)

The set of (local, global) minimisers of F is denoted by: 

arg min $F = \{ x ^ { * } \in \mathbb { R } ^ { n } : x ^ { * }$ is a minimiser of $F \} \subset \mathbb { R } ^ { n }$ 

Empty? Singleton? (it depends on $F )$ 

Notation, preliminaries & basic notions 

Convexity, strong convexity 

## Convex functions

## Definition (convex function)

$F \in { \mathcal { P } }$ is said to be convex if: 

$$
(\forall x, y \in \mathbb {R} ^ {n}), \quad (\forall \alpha \in [ 0, 1 ]), \quad F (\alpha x + (1 - \alpha) y) \leq \alpha F (x) + (1 - \alpha) F (y).
$$

Moreover, F is strictly convex if the inequality holds when $x , y \in \mathsf { d o m } ( F ) , \ x \neq y$ and $\alpha \in ( 0 , 1 )$ . We say that $G : \mathbb { R } ^ { n }  [ - \infty , + \infty )$ is concave is $F = - G$ is convex. If a function is not convex nor concave we say that is non-convex. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/eb2578e7073fb2e7a00e78d1ba266333b65bbbbebac90ff2ac57268dc05807e9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/563a2e5c0c8f8171cfbf59b75b2903d221d8b64d0d59289e84a0a344d2f6bffe.jpg)



Convex/concave function


## Convex functions

## Definition (convex function)

$F \in { \mathcal { P } }$ is said to be convex if: 

$$
(\forall x, y \in \mathbb {R} ^ {n}), \quad (\forall \alpha \in [ 0, 1 ]), \quad F (\alpha x + (1 - \alpha) y) \leq \alpha F (x) + (1 - \alpha) F (y).
$$

Moreover, F is strictly convex if the inequality holds when $x , y \in \mathsf { d o m } ( F ) , \ x \neq y$ and $\alpha \in ( 0 , 1 )$ . We say that $G : \mathbb { R } ^ { n }  [ - \infty , + \infty )$ is concave is $F = - G$ is convex. If a function is not convex nor concave we say that is non-convex. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/6be86308d9779b0ed2431f15a0c2520d5e628d9a073be2e236944666cfecc145.jpg)



Convex VS. strictly convex functions


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/36e9b1bb75e169d340da201792775f6f5baf39c800e217ca26dd77fb98050d4a.jpg)


## Convex functions

## Definition (convex function)

$F \in { \mathcal { P } }$ is said to be convex if: 

$$
(\forall x, y \in \mathbb {R} ^ {n}), \quad (\forall \alpha \in [ 0, 1 ]), \quad F (\alpha x + (1 - \alpha) y) \leq \alpha F (x) + (1 - \alpha) F (y).
$$

Moreover, F is strictly convex if the inequality holds when $x , y \in \mathsf { d o m } ( F ) , \ x \neq y$ and $\alpha \in ( 0 , 1 )$ . We say that $G : \mathbb { R } ^ { n }  [ - \infty , + \infty )$ is concave is $F = - G$ is convex. If a function is not convex nor concave we say that is non-convex. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/a272f47e73fc1afc7fd5f0eafbedd32b323184fe58127fdb6626cf0a8756648e.jpg)



Convex VS. non-convex function


## Convex functions

## Definition (convex function)

$F \in { \mathcal { P } }$ is said to be convex if: 

$$
(\forall x, y \in \mathbb {R} ^ {n}), \quad (\forall \alpha \in [ 0, 1 ]), \quad F (\alpha x + (1 - \alpha) y) \leq \alpha F (x) + (1 - \alpha) F (y).
$$

Moreover, F is strictly convex if the inequality holds when $x , y \in \mathsf { d o m } ( F ) , \ x \neq y$ and $\alpha \in ( 0 , 1 )$ . We say that $G : \mathbb { R } ^ { n }  [ - \infty , + \infty )$ is concave is $F = - G$ is convex. If a function is not convex nor concave we say that is non-convex. 

## Examples:

• $F ( x ) = \left\| x \right\|$ is convex 

$$
\| \alpha x + (1 - \alpha) y \| \leq \| \alpha x \| + \| (1 - \alpha) y \| = \alpha \| x \| + (1 - \alpha) \| y \| \quad \forall x, y \in \mathbb {R} ^ {n}
$$

$F ( x ) = \| x \| ^ { 2 }$ is strictly convex 

$F ( x ) = \| x \| _ { p } , p \in [ 1 , + \infty )$ are convex 

Epigraph 

## Proposition (epigraph of convex functions is convex set)

Let $F \in { \mathcal { P } }$ . Then F is convex if and only if ep $( F )$ is a convex set. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/1ce7b4949cac2dbc6aa4f46e2dc60e00d2346fe3404d4c4ecbde31f8b01b00d6.jpg)



Epigraph


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/963b780bf779e061f69c1447887879fa0db622dd4928755d244c5d529f713c66.jpg)


## Proposition (operations with convex functions)

Let $f$ and $\boldsymbol { g }$ be two convex functions and let $\beta \in \mathbb { R } _ { + + }$ . Then, the sum $f + g$ is a convex function and the function $\beta f$ is a convex function. 

## Strong convexity

## Definition (strongly convex function)

$F \in { \mathcal { P } }$ is said to be strongly convex of parameter $\mu > 0$ if $\forall x , y \in \mathbb { R } ^ { n }$ and $\forall \alpha \in [ 0 , 1 ]$ 

$$
F (\alpha x + (1 - \alpha) y) \leq \alpha F (x) + (1 - \alpha) F (y) - \frac {\mu}{2} (1 - \alpha) \alpha \| x - y \| ^ {2}
$$

## Proposition (characteristion of strongly convex functions)

$F \in { \mathcal { P } }$ is µ-strongly convex if and only if $\begin{array} { r } { G ( \cdot ) : = F ( \cdot ) - \frac { \mu } { 2 } \| \cdot \| ^ { 2 } } \end{array}$ is convex. 

## Proposition (growth condition around minimisers)

If $F \in { \mathcal { P } }$ is µ-strongly convex and $x ^ { * } \in \mathsf { a r g m i n } _ { x } F ( x )$ , then: 

$$
F (x) - F (x ^ {*}) \geq \frac {\mu}{2} \| x - x ^ {*} \| ^ {2}, \quad \forall x \in X.
$$

## Strong convexity

## Definition (strongly convex function)

$F \in { \mathcal { P } }$ is said to be strongly convex of parameter $\mu > 0$ if $\forall x , y \in \mathbb { R } ^ { n }$ and $\forall \alpha \in [ 0 , 1 ]$ 

$$
F (\alpha x + (1 - \alpha) y) \leq \alpha F (x) + (1 - \alpha) F (y) - \frac {\mu}{2} (1 - \alpha) \alpha \| x - y \| ^ {2}
$$

## Proposition (characteristion of strongly convex functions)

$F \in { \mathcal { P } }$ is µ-strongly convex if and only if $\begin{array} { r } { G ( \cdot ) : = F ( \cdot ) - \frac { \mu } { 2 } \| \cdot \| ^ { 2 } } \end{array}$ is convex. 

## Proposition (growth condition around minimisers)

If $F \in { \mathcal { P } }$ is µ-strongly convex and $x ^ { * } \in \mathsf { a r g m i n } _ { x } F ( x )$ , then: 

$$
F (x) - F (x ^ {*}) \geq \frac {\mu}{2} \| x - x ^ {*} \| ^ {2}, \quad \forall x \in X.
$$

strong convexity ⇒ strict convexity ⇒ convexity 

Counterexample (strict convexity $\nRightarrow$ strong convexity): $F : \mathbb { R } \to \overline { { \mathbb { R } } } , F ( x ) = e ^ { x }$ 

Notation, preliminaries & basic notions 

Lower semi-continuity & coercivity 

## Lower semi-continuity

## Definition (lower semi-continuity)

Let $F \in { \mathcal { P } }$ . We say that F is lower semi-continuous (l.s.c.) at the point $x \in \mathbb { R } ^ { n }$ if 

$$
F (x) \leq \operatorname * {l i m i n f} _ {y \to x} F (y).
$$

Equivalently, for every sequence $( x _ { k } ) _ { k \in \mathbb { N } }$ with $x _ { k } \to x ;$ 

$$
F (x) \leq \liminf _ {k \to + \infty} F (x _ {k}) \left(= \lim _ {k \to + \infty} \inf \left\{F (x _ {j}): j \geq k \right\}\right).
$$

If F is l.s.c. at every $x \in \mathbb { R } ^ { n }$ , we say that the function is l.s.c. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/07f37659b232f24f2cb8181fa0db38886df279a2418eb899a626b2e9a4557c22.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/e7a1cc638006e7aa475220127725f6afbbd0dff3926384ebbfc4bc73128ace59.jpg)



Left: lower l.s.c. Right: where the function is lower l.s.c.?


## Examples of l.s.c. functions

• The functions 

$$
F (x) = \left\{ \begin{array}{l l} 0 & \text { if } x \leq 0 \\ 1 & \text { if } x > 0 \end{array} \right., \qquad F (x) = \lceil x \rceil = \min \left\{k \in \mathbb {Z}: x \leq k \right\}
$$

are l.s.c. (but not continuous). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/0a282b15772fd7931893c4409b03982780ec243eae3b3d2c5526a6f8d9b8df05.jpg)



F (x) = dxe


• All continuous functions $( 1 . 5 . mathsf { c } + \mathsf { u } . 5 . \mathsf { c } . )$ 

## Coercivity

How to ensure that the minimum is not attained at “extreme points” of the domain? 

Definition (coercivity) 

Let $F \in { \mathcal { P } }$ . We say that F is coercive if 

$$
\lim _ {\| x \| \to + \infty} F (x) = + \infty .
$$

## Examples:

$F : \mathbb { R } \to \mathbb { R } _ { + } , F ( x ) = e ^ { x }$ is not coercive, but $F : \mathbb { R } \to \mathbb { R } _ { + } , F ( x ) = e ^ { | x | }$ is. 

$F : \mathbb { R } ^ { 2 } \to \mathbb { R } _ { + } , F ( x , y ) = x ^ { 2 } + y ^ { 2 }$ is coercive. 

$F : \mathbb { R } ^ { 2 } \to \mathbb { R } _ { + } , F ( x , y ) = x ^ { 2 } - 2 x y + x ^ { 2 } = ( x - y ) ^ { 2 }$ is not coercive. Why? 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/54cd4cdf88e0a8c37167bc5fa14a32339eaf7bbd1144852c9934e7ab6bb4e055.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/b3fcf162bd6bb025b6b91f57dab2d6699088322537ccc727dcc8a7e3d12cb600.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/95882afa37163e46a9c6034fe2585899fb3bc1e45efbd027217d0ecad76dfafa.jpg)


## Existence of minimisers

## Theorem (existence of minimisers)

If F is proper, l.s.c. and coercive, then the set of minimisers of F is non-empty and compact. 

Note: generalises the Bolzano-Weirestrass theorem holding for problems 

$$
\min _ {x \in C} F (x)
$$

for compact $C \subset \mathbb { R } ^ { n }$ s.t. $C \cap { \mathsf { d o m } } ( F ) \neq \emptyset$ and continuous $F$ 

## Theorem (existence of minimisers)

If F is proper, l.s.c. and coercive, then the set of minimisers of F is non-empty and compact. 

Note: generalises the Bolzano-Weirestrass theorem holding for problems 

$$
\min _ {x \in C} F (x)
$$

for compact $C \subset \mathbb { R } ^ { n }$ s.t. $C \cap { \mathsf { d o m } } ( F ) \neq \emptyset$ and continuous $F$ . 

## Theorem (convex case)

If F is proper, coercive and convex, then every local minimiser is a global minimiser. 

## Definition $\left( \Gamma _ { 0 } \left( \mathbb { R } ^ { n } \right) \right)$

$\Gamma _ { 0 } ( X ) : = \{ F : X \to \overline { { \mathbb { R } } }$ : F is proper, convex and l.s.c. 

Remark: $F \in \Gamma _ { 0 } ( X ) \not \Rightarrow F$ admits a minimiser. Take e.g. $F ( x ) = - \log x , x > 0$ and $F ( x ) = + \infty , x \leq 0 .$ . . no coercivity guaranteed! 

So far, only existence of minimisers. How to guarantee uniqueness? 

Theorem (existence+uniqueness of minimisers) 

If F is proper, l.s.c., coercive and strictly convex, then F admits a unique minimiser. 

Equivalently, arg min $\boldsymbol { F } = \{ \boldsymbol { x } ^ { * } \}$ , a singleton. 

Remark: as strong convexity implies strict convexity, the same holds. 

Notation, preliminaries & basic notions 

How to provide a characterisation of the minimisers of a function f in terms of a suitable notion of $\sqrt [ [object Object] ] { \sqrt { \mathbf { \Lambda } } } f " ?$ 

Definition (Gâteaux differentiability)

Let $f \in \mathcal{P}$ and let $x \in \text{dom}(f)$ . For $v \in \mathbb{R}^n$ , we denote the directional derivative in $x$ along the direction $v$ as the limit $f'(x; v) = f'(x)[v] := \lim_{t \to 0^+} \frac{f(x + tv) - f(x)}{t},$ when it exists. If there exists $w \in \mathbb{R}^n$ such that: $(\forall v \in \mathbb{R}^n) \quad f'(x)[v] = \langle w, v \rangle,$ then we say that $f$ is Gâteaux differentiable in $x$ and denote by $\nabla f(x) = w$ the Gâteaux derivative (or, simply, the gradient) of $f$ at $x$ . 

## Optimality conditions and relations with convexity

## Theorem (Fermat’s rule)

Let $f \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ be diferentiable at point $x ^ { * }$ . Then: 

$$
x ^ {*} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} f (x) \quad \Longleftrightarrow \quad \nabla f (x ^ {*}) = 0.
$$

## Optimality conditions and relations with convexity

## Theorem (Fermat’s rule)

Let $f \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ be diferentiable at point $x ^ { * }$ . Then: 

$$
x ^ {*} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} f (x) \quad \Longleftrightarrow \quad \nabla f (x ^ {*}) = 0.
$$

## Proposition (Diferentiability and convexity)

Let $f \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ . Suppose that f is diferentiable on dom(f ). Then the following statements are equivalent: 

1. f is convex; 

2. $\forall x , y \in \mathsf { d o m } ( f ) , f ( y ) \geq f ( x ) + \langle \nabla f ( x ) , y - x \rangle ;$ 

$$
\forall x, y \in \operatorname{dom} (f), \langle \nabla f (x) - \nabla f (y), x - y \rangle \geq 0.
$$

## Diferentiability and strong convexity

## Corollary (Diferentiability and strong convexity)

Let $f \in \Gamma_0(\mathbb{R}^n)$ and $\mu > 0$ . Suppose that $f$ is differentiable on $\operatorname{dom}(f)$ . Then the following statements are equivalent: 

1. f is µ-strongly convex; 

$$
2. \forall x, y \in \operatorname{dom} (f), f (y) \geq f (x) + \langle \nabla f (x), y - x \rangle + \frac {\mu}{2} \| y - x \| ^ {2};
$$

$$
3. \forall x, y \in \operatorname{dom} (f), \langle \nabla f (x) - \nabla f (y), x - y \rangle \geq \mu \| x - y \| ^ {2}.
$$

Example: let $\begin{array} { r } { f ( x ) = \frac { 1 } { 2 } \| A x - y \| ^ { 2 } } \end{array}$ , for $A \in \mathbb { R } ^ { m \times n }$ positive definite, $y \in \mathbb { R } ^ { m }$ . Then: 

$$
\nabla f (x) = A ^ {T} (A x - y).
$$

Since $A ^ { T } A$ is positive definite $( \mathsf { i . e . , ~ } \lambda _ { \mathsf { m i n } } : = \lambda _ { \mathsf { m i n } } ( A ^ { T } A ) > 0 )$ , then: 

$$
(\forall x, y \in \mathbb {R} ^ {n}) \quad \langle \nabla f (x) - \nabla f (y), x - y \rangle = \left\langle A ^ {T} A (x - y), x - y \right\rangle \geq \lambda_ {\min} \| x - y \| ^ {2},
$$

hence f is $\lambda _ { \mathrm { { m i n } ^ { - } \mathsf { { s t r o n g l y } } } }$ convex. 

Remark: from condition 3., if $x ^ { \ast } \in$ arg min $f ( x )$ , then for all $x \in \mathsf { d o m } ( f )$ 

$$
\langle \nabla f (x) - 0, x - x ^ {*} \rangle \geq \mu \| x - x ^ {*} \| ^ {2} \quad \Rightarrow \quad \boxed {\mu \| x - x ^ {*} \| \leq \| \nabla f (x) \|}
$$

## Polyak- Lojasiewicz condition

## Proposition (Polyak- Lojasiewicz condition)

Let $f \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ and let $\mu > 0$ Suppose that f is diferentiable on dom(f ), that f is $\mu -$ strongly convex and that there exists $x ^ { \ast } \in$ arg min $f ( x )$ . Then: 

$$
(\forall x \in \operatorname{dom} (f)) \quad \boxed {f (x) - \min _ {x} f (x) \leq \frac {1}{2 \mu} \| \nabla f (x) \| ^ {2}}\tag{*}
$$

Proof. 

$$
\begin{array}{l} \underset {y \in \operatorname{dom} (f)} {\min} f (y) \geq \underset {y \in \operatorname{dom} (f)} {\min} \left(f (x) + \langle \nabla f (x), y - x \rangle + \frac {\mu}{2} \| y - x \| ^ {2}\right) \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qend{array}
$$

• “Gradient grows as a quadratic function as we increase $f "$ . Important condition for achieving fast convergence rates! 

$( * )$ holds also for non-strongly convex functions $( \mathsf { e . g . } , \mathsf { \Omega } _ { 2 } ^ { 1 } \| A x - y \| ^ { 2 }$ for A not positive definite) 

## Lipschitz smoothness (L-smoothness)

In the framework of first-order optimisation methods, it’s important to provide conditions on the growth of functions considered. 

Definition (L-smoothness) Let $f \in \Gamma_0(\mathbb{R}^n)$ be differentiable. We say that $f$ is an L-smooth function with constant $L \geq 0$ iff: $\exists L \geq 0 : \forall x, y \in \mathbb{R}^n \quad \| \nabla f(x) - \nabla f(y) \| \leq L \| x - y \|$ . 

Remark: For $\begin{array} { r } { f ( x ) = \frac { 1 } { 2 } \| A x - y \| _ { 2 } ^ { 2 } } \end{array}$ , you can check $L = \| A ^ { T } A \| \leq \| A \| ^ { 2 }$ 

## Theorem (characterisation of L-smooth functions)

Let $f : \mathbb { R } ^ { n } \to \mathbb { R }$ a convex diferentiable function and let $L > 0$ The following statements are equivalent: 

1. f is L-smooth 

2. (descent lemma) 

$$
(\forall x, y \in \mathbb {R} ^ {n}) f (y) - f (x) - \langle \nabla f (x), y - x \rangle \leq \frac {L}{2} \| x - y \| ^ {2}
$$

3. 

$$
\left(\forall x, y \in \mathbb {R} ^ {n}\right) \frac {1}{2 L} \| f (x) - f (y) \| ^ {2} \leq f (y) - f (x) - \langle \nabla f (x), y - x \rangle
$$

4. 

$$
\left(\forall x, y \in \mathbb {R} ^ {n}\right) \frac {1}{L} \| f (x) - f (y) \| ^ {2} \leq \langle \nabla f (x) - \nabla f (y), x - y \rangle
$$

5. 

$$
(\forall x, y \in \mathbb {R} ^ {n}) \quad \langle \nabla f (x) - \nabla f (y), x - y \rangle \leq L \| x - y \| ^ {2}
$$

6. $\frac { L } { 2 } \parallel \cdot \parallel ^ { 2 } - f ( \cdot )$ is convex. 

## Comparing smoothness and strong convexity

• f is L-smooth if and only if: 

$$
(\forall x, y \in \mathbb {R} ^ {n}) f (y) \leq f (x) + \langle \nabla f (x), y - x \rangle + \frac {L}{2} \| x - y \| ^ {2}
$$

• f is µ-strongly convex if and only if: 

$$
\forall x, y \in \operatorname{dom} (f), \quad f (y) \geq f (x) + \langle \nabla f (x), y - x \rangle + \frac {\mu}{2} \| y - x \| ^ {2}
$$

It can be proved that if f is a $C ^ { 2 }$ function there holds: 

$$
\mu \mathrm{Id} \preceq \nabla^ {2} f (x) \preceq L \mathrm{Id}, \quad \text {   for   all   } x
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/74ab4c12b99e7fae75d17f057549eeb498420627acca036156493556803e88c9.jpg)


# Smooth optimisation algorithms

# Smooth optimisation algorithms

Gradient descent 

## Gradient descent

Gradient descent (GD) algorithm: ubiquitous in many applications fo minimising (non-)convex, diferentiable and proper functions $f : \mathbb { R } ^ { n } \to { \overline { { \mathbb { R } } } }$ 

Algorithm: Gradient Descent (GD) algorithm

Input: $\tau\in(0,\frac{2}{L}),x^{0}\in\mathbb{R}^{n}$ .

for $k\geq0$ do $x_{k+1}=x_{k}-\tau\nabla f(x_{k})$ end for 

Choice of $\underline { { \tau } } :$ important to guarantee convergence (need to be suficiently small), it relates to $L$ (∼growth of $f )$ Example: minimise $f ( x ) = x ^ { 2 } / 2 .$ . GD iteration: $x _ { k + 1 } = ( 1 - \tau ) x _ { k }$ , convergence for. . . ? 

• Convexity assumption: no dependence on $x _ { 0 }$ 

• Stopping criterion: relative error $\left\| x _ { k + 1 } - x _ { k } \right\| \leq$ tol or gradient check $\| \nabla f ( x _ { k + 1 } ) \| \leq \mathtt { t o l }$ (approaching 0). 

## Understanding the step-size upper bound

## Lemma

For all $k \geq 0$ , there holds: 

$$
\tau \left(1 - \frac {\tau L}{2}\right) \| f (x _ {k}) \| ^ {2} \leq f (x _ {k}) - f (x _ {k + 1}).
$$

Thus, if $\begin{array} { r } { \tau < \frac { 2 } { L } } \end{array}$ , then $f { \bigl ( } x _ { k + 1 } { \bigr ) } \leq f { \bigl ( } x _ { k } { \bigr ) }$ , i.e. the GD algorithm is descending. 

Proof. Since $x _ { k + 1 } - x _ { k } = - \tau \nabla f ( x _ { k } )$ , then by the characterisation 2. of L-smoothness we have: 

$$
f (x _ {k + 1}) \leq f (x _ {k}) - \tau \langle \nabla f (x _ {k}), \nabla f (x _ {k}) \rangle + \frac {L}{2} \tau^ {2} \| \nabla f (x _ {k}) \| ^ {2},
$$

so the thesis follows. 

## Convergence of GD algorithm

## Theorem (convergence of GD)

Let $\left( \boldsymbol { x } _ { k } \right) _ { k }$ the sequence of iterates generated by GD. Then, if $\tau \in ( 0 , 2 / L )$ there holds: 

$$
f (x _ {k}) - f (x ^ {*}) \leq \frac {\| x ^ {0} - x ^ {*} \| ^ {2}}{2 \tau k} = O \left(\frac {1}{k}\right)
$$

## Lemma (progress bounds)

For GD iterations with $\tau = 1 / L$ there holds: 

$$
f (x _ {k + 1}) \leq f (x _ {k}) - \frac {1}{2 L} \| \nabla f (x _ {k}) \| ^ {2}
$$

Proof. Using $\begin{array} { r } { x _ { k + 1 } - x _ { k } = - \frac { 1 } { L } \nabla f ( x _ { k } ) } \end{array}$ we can apply the characterisation 2. to get: 

$$
\begin{array}{c} f (x _ {k + 1}) \leq f (x _ {k}) - \frac {1}{L} \| \nabla f (x _ {k}) \| ^ {2} + \frac {L}{2} \| \frac {1}{L} \nabla f (x _ {k}) \| ^ {2} \\ \leq f (x _ {k}) - \frac {1}{2 L} \| \nabla f (x _ {k}) \| ^ {2}. \end{array}\tag{1}
$$

We can use this progress bound to show improved rates under Polyak- Lojasiewicz condition (in particular, strongly convex functions). 

Smooth optimisation algorithms 

Convergence proof under PL condition 

## Linear convergence of GD under PL condition

## Theorem (linear convergence of GD under PL)

Let $\left( \boldsymbol { x } _ { k } \right) _ { k }$ the sequence of iterates generated by GD. Then, if $\tau = 1 / L$ there holds: 

$$
f (x _ {k}) - f (x ^ {*}) \leq \left(1 - \frac {\mu}{L}\right) ^ {k} (f (x _ {0}) - f (x ^ {*})),
$$

where, notice, $0 < \mu \le L$ . 

Proof. Use the Lemma (progress bound) and the PL inequality: 

$$
f (x _ {k + 1}) \leq f (x _ {k}) - \frac {1}{2 L} \| \nabla f (x _ {k}) \| ^ {2} \leq f (x _ {k}) - \frac {\mu}{L} (f (x _ {k}) - f (x ^ {*})).
$$

Subtracting $f ( x ^ { * } )$ from both sides we get: 

$$
f (x _ {k + 1}) - f (x ^ {*}) \leq \left(1 - \frac {\mu}{L}\right) (f (x _ {k}) - f (x ^ {*})).
$$

Applying this recursively gives the thesis since: 

$$
f (x _ {k + 1}) - f (x ^ {*}) \leq \left(1 - \frac {\mu}{L}\right) (f (x _ {k}) - f (x ^ {*})) \leq \left(1 - \frac {\mu}{L}\right) ^ {2} (f (x _ {k - 1}) - f (x ^ {*}))
$$

$$
\leq \dots \leq \left(1 - \frac {\mu}{L}\right) ^ {k} (f (x _ {0}) - f (x ^ {*})).
$$

To show $0 < \mu \le L$ , since by descent lemma we have that for all $v \in \mathbb { R } ^ { n }$ 

$$
f (x ^ {*}) \leq f (v) - \frac {1}{2 L} \| \nabla f (v) \| ^ {2}.
$$

Combining PL with this inequality we get: 

$$
\frac {1}{2 \mu} \| \nabla f (v) \| ^ {2} \geq f (v) - f (x ^ {*}) \geq \frac {1}{2 L} \| \nabla f (v) \| ^ {2} \quad \forall v \in \mathbb {R} ^ {n} \Rightarrow \mu \leq L
$$

Do we practically see this gain in known problems? 

$$
f (x) = \frac {1}{2} \| A x - y \| ^ {2} + \frac {\lambda}{2} \| x \| ^ {2}, \quad \lambda > 0
$$

f is λ-strongly convex. Convergence factor of the theorem: 

$$
\frac {\mu}{L} = \frac {\min \left\{\operatorname{eig} (A ^ {T} A) \right\} + \lambda}{\max \left\{\operatorname{eig} (A ^ {T} A) \right\} + \lambda}
$$

• If $\lambda \gg 1$ , then $\left( 1 - { \frac { \mu } { L } } \right) \to 0$ hence faster convergence 

• If $L \gg \mu \ ( \ \mathsf { \Omega } ^ { \ast } \mathsf { s m a l l ^ { \prime \prime } } \ \mathsf { P L } )$ , then this rate is not very informative, so in practice we observe the rate $O ( 1 / k )$ 

• The quantity $L / \mu$ is called the condition number of $f$ (relates with the condition number of matrix $\nabla ^ { 2 } f$ when f is $C ^ { 2 } )$ 

Smooth optimisation algorithms 

Motivation for accelerated algorithms 

. . . back to standard GD iteration and $O ( 1 / k )$ convergence rate. 

. . . back to standard GD iteration and $O ( 1 / k )$ convergence rate. 

## Theorem (worst-case bounds<sup>1</sup>)

For $x _ { 0 } \in \mathbb { R } ^ { n } , \ L > 0$ and $\begin{array} { r } { 1 < k \le \frac { 1 } { 2 } ( n - 1 ) } \end{array}$ , there exists a convex, L-smooth function f s.t. for any first-order algorithm: 

$$
f (x _ {k}) - f (x ^ {*}) \geq \frac {3 L \| x _ {0} - x ^ {*} \| ^ {2}}{3 2 (k + 1) ^ {2}} = O \left(\frac {1}{(k + 1) ^ {2}}\right).
$$

It would be somehow ‘optimal’ finding convergence rates close to such lowe (inevitable) bound. . . 

How to fill the gap between $O ( 1 / k )$ and $O ( 1 / ( k + 1 ) ^ { 2 } )$ for convex functions? 

Accelerated smooth optimisation algorithms 

Nesterov acceleration of GD 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/9d92f2b5d27f2a2371a30d51d799e4b1cf2b340f2b3563fe4dc57119f0472a03.jpg)


## Accelerated gradient descent

Idea: add inertia to “shift” the sequence of iterates. 

Algorithm: Accelerated Gradient Descent (AGD) algorithm $^{2}$ Input: $x_{0} = x^{-1} \in R^{n}, \tau \in \left(0, \frac{1}{L}\right], t_{0} = 0.$ for $k \geq 0$ do $t_{k+1} = \frac{1 + \sqrt{1 + 4t_{k}^{2}}}{2}$ $y_{k+1} = x_{k} + \frac{t_{k} - 1}{t_{k+1}}(x_{k} - x_{k-1})$ $x_{k+1} = y_{k+1} - \tau \nabla f(y_{k+1})$ end for 

## Lemma (behaviour of the sequence $\left( t _ { k } \right) )$

Let $t _ { 0 }$ and the sequence $t _ { k }$ be defined by: 

$$
t _ {k + 1} = \frac {1 + \sqrt {1 + 4 t _ {k} ^ {2}}}{2}.
$$

Then $t _ { k } \geq { \frac { k + 2 } { 2 } }$ for all $k \geq 0$ In particular, $t _ { k } \to \infty$ 

Proof: by induction. For $k = 0$ we have $t _ { 0 } \geq 1$ . Suppose that the claim holds for some $k ,$ meaning that $t _ { k } \geq \frac { k + 2 } { 2 }$ . Want to show: 

$$
t _ {k + 1} \geq \frac {k + 1 + 2}{2} = \frac {k + 3}{2}.
$$

Using recursion and $2 t _ { k } \geq k + 2$ (induction) 

$$
t _ {k + 1} = \frac {1 + \sqrt {1 + 4 t _ {k} ^ {2}}}{2} \geq \frac {1 + \sqrt {1 + (k + 2) ^ {2}}}{2} \geq \frac {1 + \sqrt {(k + 2) ^ {2}}}{2} = \frac {k + 3}{2}.
$$

Remark: any sequence $\left( t _ { k } \right) _ { k }$ satisfying $t _ { k + 1 } ^ { 2 } - t _ { k + 1 } \leq t _ { k } ^ { 2 } , k \geq 0$ works (Chambolle, Dossal, 2015). 

## Theorem (convergence of $\mathsf { A G D } ) ^ { 3 }$

Let $\left( x _ { k } \right) _ { k }$ the sequence of iterates generated by AGD. Then, there holds: 

$$
f (x _ {k}) - f (x ^ {*}) \leq \frac {2 \| x ^ {0} - x ^ {*} \| ^ {2}}{\tau (k + 1) ^ {2}}.
$$

Get faster, at $\begin{array} { r } { O \left( \frac { 1 } { ( k + 1 ) ^ { 2 } } \right) } \end{array}$ to a reasonably accurate approximation of $x ^ { * }$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2ca212a3-c552-4ff0-90cd-ec927a5c7892/45edcc190f812c99ec104f485b0f075a1b1107532665cd0b43331edc96d180f2.jpg)


. . . proof is quite technical. You’ll see this in the case of non-smooth problems tomorrow. 

How many iterations are needed for such algorithms to achieve ε-accuracy, i.e. 

$$
\boxed {f (x _ {k}) - f (x ^ {*}) \leq \varepsilon}
$$

• GD: for all $k \geq 0$ such that $k \geq \lceil C / \varepsilon \rceil$ 

$\mathsf { A G D }$ : for all $k \geq 0$ such that $k \geq \lceil C / \sqrt { \varepsilon } - 1 \rceil$ 

$G \mathsf { D } + \mathsf { P } \mathsf { L }$ : for all $k \geq 0$ such that $k \geq \lceil C \log \left( 1 / \varepsilon \right) \rceil$ 

## Conclusions

We focus on convex, smooth optimisation problems arising in applications (e.g., imaging inverse problems). 

• We revised basic notions for having well-posedness of the underlying problem 

• We considered GD as a reference first-order algorithm 

• We commented on the improved speed achieved by GD whenever the underlying function enjoys further regularity (PL + strong convexity) 

• We discussed Nesterov acceleration for improving convergence speed in convex cases 

How to explore analogous ideas in the structured smooth+non-smooth setting? 

## Questions?

calatroni@i3s.unice.fr 