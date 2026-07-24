## Regularisation Theory What is a regularisation and why do we regularise?

Martin Benning 

University College London 

Erasmus+ International PhD Summer School 2025 

Mathematics and Machine Learning for Image Analysis University of Bologna 

9 June 2025 

Regularisation Theory – What is a regularisation and why do we regularise? 

Regularisation Theory – What is a regularisation and why do we regularise? 

Learning optimal sampling strategies for Magnetic Resonance Imaging (MRI) 

Regularisation Theory – What is a regularisation and why do we regularise? 

Learning optimal sampling strategies for Magnetic Resonance Imaging (MRI) 

Lifted training and inversion of neural networks 

## Focus of First Lecture(s)

## Regularisation theory

## What is a regularisation and why do we regularise?

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/85689468ff0094939102f72c767255bb9dc8d2a065f9d178b892411bfeca9abf.jpg)


## Contents

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/934258ab423a4c5e8fd1e564996e422e49d9dcb16a27308d7891dc95228dd48e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/1c2a5455708293ff174076fdb4e994c752cf5fc4f0dc4c8997adfe6a2dcd5b9c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/fa2caa5e0b1f15cea67fb3cebe88191664dfdc5357b69c338d2331c021cce766.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/5e21255becaf2935c69b4f8381bfb8d860ff8a98597ec8e86a3a5848744bb234.jpg)


1 Introduction to Inverse Problems
2 Fundamental Concepts in Regularisation
3 Selecting Solutions
4 Convergent Regularisation Methods
5 Variational Regularisation Methods
6 Convergence Analysis: Error Estimates
7 Iterative Regularisation Methods
8 Data-Driven Regularisation: Spectral Methods
9 Outlook and Open Questions 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/9608b89706fd6bd8fc8b18901b276d4681c486091f290322ac15777731c87215.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/401412c09a539f1f6eac161a73d93083e10e95e6845076b70fcb5c3c4b4d831c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/9355833b236c01a2f6ecc3a4691f6cb0db3aa88cf00d6ed41ad14d144751cf30.jpg)


## Introduction to Inverse Problems

## Introduction: What is an Inverse Problem?

## General Form

Mathematically, an inverse problem can be described as solving the operator equation: 

$$
\mathrm{Ku} = \mathrm{f}
$$

for u, where 

## Introduction: What is an Inverse Problem?

## General Form

Mathematically, an inverse problem can be described as solving the operator equation: 

$$
\mathrm{Ku} = \mathrm{f}
$$

## for u, where

u ∈ U is the unknown quantity we want to determine (e.g., an image, a function). 

## Introduction: What is an Inverse Problem?

## General Form

Mathematically, an inverse problem can be described as solving the operator equation: 

$$
\mathrm{Ku} = \mathrm{f}
$$

## for u, where

u ∈ U is the unknown quantity we want to determine (e.g., an image, a function). 

f ∈ V is the given measurement data. 

## Introduction: What is an Inverse Problem?

## General Form

Mathematically, an inverse problem can be described as solving the operator equation: 

$$
\mathrm{Ku} = \mathrm{f}
$$

for u, where 

u ∈ U is the unknown quantity we want to determine (e.g., an image, a function). 

f ∈ V is the given measurement data. 

K : U → V is an operator mapping from a Banach space U to a Banach space V. 

## Introduction: What is an Inverse Problem?

## General Form

Mathematically, an inverse problem can be described as solving the operator equation: 

$$
\mathrm{Ku} = \mathrm{f}
$$

## for u, where

u ∈ U is the unknown quantity we want to determine (e.g., an image, a function). 

f ∈ V is the given measurement data. 

K : U → V is an operator mapping from a Banach space U to a Banach space V. 

This operator K models the forward process (how u generates f). 

## Introduction: What is an Inverse Problem?

## General Form

Mathematically, an inverse problem can be described as solving the operator equation: 

$$
\mathrm{Ku} = \mathrm{f}
$$

## for u, where

u ∈ U is the unknown quantity we want to determine (e.g., an image, a function). 

f ∈ V is the given measurement data. 

K : U → V is an operator mapping from a Banach space U to a Banach space V. 

This operator K models the forward process (how u generates f). 

Some useful references: [5, 14, 1] 

## Inverse Problems: A Visual Illustration

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/8d78de19bf3e183d181468b7ae566a00e774fd4bcfa07365b51a84a895e9f221.jpg)


## The Shadow Image Problem

Forward Problem: Given 3D hand shapes, compute 2D shadows 

## Inverse Problems: A Visual Illustration

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/f5d0c3e67230858321895e3fa219df912e494253033dcf87d892a6e08317df81.jpg)


## The Shadow Image Problem

Forward Problem: Given 3D hand shapes, compute 2D shadows 

Inverse Problem: Given only the 2D shadow silhouettes, determine the original 3D object 

## Inverse Problems: A Visual Illustration

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/01c384a01815f6e8ec39ec43689f69b3a0d4a865ffb08d018d6d6f979e623628.jpg)


## The Shadow Image Problem

Forward Problem: Given 3D hand shapes, compute 2D shadows 

Inverse Problem: Given only the 2D shadow silhouettes, determine the original 3D object 

Multiple different 3D objects can produce same shadow (non-uniqueness) 

## The Challenge: Well-Posedness vs. Ill-Posedness

## Conditions for Well-Posedness

Most practical inverse problems are ill-posed in the sense of Hadamard [7, 8] and John [9]. A problem is well-posed if it satisfies 

1 Existence: A solution u exists for all $\mathsf { f } \in \mathcal { V }$ 

2 Uniqueness: The solution is unique. 

3 Stability (Continuity): The solution u depends continuously on the data f. Small changes in f lead to small changes in u. 

If any of these conditions are violated, the problem is ill-posed. Stability is often the most problematic condition in practice. 

## Example: The Inverse Problem of Differentiation

## Problem Setup

Consider the task of finding a function $\mathfrak { u } ( { \boldsymbol { x } } )$ given its integral f(y). If we assume $\mathsf { f } ( 0 ) = 0$ we want to find $\mathrm { u } = \mathrm { f } ^ { \prime }$ . This can be formulated as solving ${ \mathrm { k u } } = { \mathrm { f } }$ where the operator K is integration, i.e. 

$$
(K u) (y) = \int_ {0} ^ {y} u (x) d x = f (y).
$$

Here, $\mathsf { K } : \mathsf { C } ( [ 0 , 1 ] ) \to \{ \mathsf { g } \in \mathsf { C } ^ { 1 } ( [ 0 , 1 ] ) | \mathsf { g } ( 0 ) = 0 \}$ (or suitable $\mathrm { L } ^ { \mathfrak { p } }$ spaces, e.g., $\operatorname { L } ^ { 2 } ( [ 0 , 1 ] ) \to \operatorname { L } ^ { 2 } ( [ 0 , 1 ] ) )$ 

## Example: The Inverse Problem of Differentiation

## Problem Setup

Consider the task of finding a function $\mathfrak { u } ( { \boldsymbol { x } } )$ given its integral f(y). If we assume $\mathsf { f } ( 0 ) = 0$ we want to find $\mathrm { u } = \mathrm { f } ^ { \prime }$ . This can be formulated as solving ${ \mathrm { k u } } = { \mathrm { f } }$ where the operator K is integration, i.e. 

$$
(K u) (y) = \int_ {0} ^ {y} u (x) d x = f (y).
$$

Here, $\mathsf { K } : \mathsf { C } ( [ 0 , 1 ] ) \to \{ \mathsf { g } \in \mathsf { C } ^ { 1 } ( [ 0 , 1 ] ) | \mathsf { g } ( 0 ) = 0 \}$ (or suitable $\mathrm { L } ^ { \mathfrak { p } }$ spaces, e.g., $\operatorname { L } ^ { 2 } ( [ 0 , 1 ] ) \to \operatorname { L } ^ { 2 } ( [ 0 , 1 ] ) )$ 

## Goal

Our goal is to recover u (the derivative) from f (the integral). This is the inverse problem of differentiation. 

## Ill-Posedness of Differentiation: Setup

Suppose instead of the exact data f, we observe noisy data $\mathsf { f } ^ { \delta } = \mathsf { f } + \mathsf { n } ^ { \delta }$ , where $\mathfrak { n } ^ { \delta }$ is some noise term. We are interested in $\mathbf { u } ^ { \delta } = ( \mathsf { f } ^ { \delta } ) ^ { \prime } = \mathsf { f } ^ { \prime } + ( \mathsf { n } ^ { \delta } ) ^ { \prime } = \mathbf { u } + ( \mathsf { n } ^ { \delta } ) ^ { \prime }$ 

## Ill-Posedness of Differentiation: Setup

Suppose instead of the exact data f, we observe noisy data $\mathsf { f } ^ { \delta } = \mathsf { f } + \mathsf { n } ^ { \delta }$ , where $\mathfrak { n } ^ { \delta }$ is some noise term. We are interested in $\mathbf { u } ^ { \delta } = ( \mathsf { f } ^ { \delta } ) ^ { \prime } = \mathsf { f } ^ { \prime } + ( \mathsf { n } ^ { \delta } ) ^ { \prime } = \mathbf { u } + ( \mathsf { n } ^ { \delta } ) ^ { \prime }$ 

## A Perturbation Example

Consider the sequence of noise functions $\mathfrak { n } ^ { \delta } \in \mathrm { L } ^ { \infty } ( [ 0 , 1 ] )$ : 

$$
n ^ {\delta} (x) := \delta \sin \left(\frac {k x}{\delta}\right)
$$

for a fixed but arbitrary number ${ \gtrsim } { > } 0$ . 

The noise in the data can be made arbitrarily small: 

$$
\left\| n ^ {\delta} \right\| _ {L ^ {\infty} ([ 0, 1 ])} = \delta \rightarrow 0 \quad \text { as } \delta \rightarrow 0
$$

## Ill-Posedness of Differentiation: Analysis

## A Perturbation Example (continued)

However, the derivative of this noise is: 

$$
\left(n ^ {\delta}\right) ^ {\prime} (x) = k \cos \left(\frac {k x}{\delta}\right)
$$

The error in the reconstructed derivative $\mathrm { u } ^ { \delta }$ is: 

$$
\left\| u - u ^ {\delta} \right\| _ {L ^ {\infty} ([ 0, 1 ])} = \left\| (n ^ {\delta}) ^ {\prime} \right\| _ {L ^ {\infty} ([ 0, 1 ])} = k
$$

## Ill-Posedness of Differentiation: Analysis

## A Perturbation Example (continued)

However, the derivative of this noise is: 

$$
\left(n ^ {\delta}\right) ^ {\prime} (x) = k \cos \left(\frac {k x}{\delta}\right)
$$

The error in the reconstructed derivative $\mathrm { u } ^ { \delta }$ is: 

$$
\left\| u - u ^ {\delta} \right\| _ {L ^ {\infty} ([ 0, 1 ])} = \left\| (n ^ {\delta}) ^ {\prime} \right\| _ {L ^ {\infty} ([ 0, 1 ])} = k
$$

## Conclusion on Ill-Posedness

Despite noise becoming arbitrarily small $( \left\| \mathsf { n } ^ { \delta } \right\| _ { \mathrm { L } ^ { \infty } } \to 0 )$ , the error in $\mathrm { u } ^ { \delta }$ remains k. Therefore, $\mathrm { u } ^ { \delta }$ does not depend continuously on $\mathsf { f } ^ { \delta }$ in the ${ \mathrm { L } } ^ { \infty }$ norm. The problem is ill-posed. 

## The Operator’s ”Blueprint”: Singular Value Decomposition (SVD)

Many inverse problems involve compact linear operators K : U → V between Hilbert spaces. The SVD provides a fundamental way to understand the action of such operators. 

## The Operator’s ”Blueprint”: Singular Value Decomposition (SVD)

Many inverse problems involve compact linear operators K : U → V between Hilbert spaces. The SVD provides a fundamental way to understand the action of such operators. 

## Singular Value Decomposition (SVD) of K

For a compact operator K, there exist: 

Singular values: A sequence $\sigma _ { 1 } \geqslant \sigma _ { 2 } \geqslant \cdots > 0$ . These are positive numbers that typically decay towards zero $( \sigma _ { \mathrm { j } }  0 )$ if the range of K is infinite-dimensional. 

## The Operator’s ”Blueprint”: Singular Value Decomposition (SVD)

Many inverse problems involve compact linear operators K : U → V between Hilbert spaces. The SVD provides a fundamental way to understand the action of such operators. 

## Singular Value Decomposition (SVD) of K

For a compact operator K, there exist: 

Singular values: A sequence $\sigma _ { 1 } \geqslant \sigma _ { 2 } \geqslant \cdots > 0$ . These are positive numbers that typically decay towards zero $( \sigma _ { \mathrm { j } }  0 )$ if the range of K is infinite-dimensional. 

Orthonormal sets (or bases for relevant subspaces): 

■ $\{ \mathfrak { u } _ { \mathrm { j } } \} _ { \mathrm { j } \in \mathbb { N } }$ in U (input space elements, form an orthonormal basis of $\mathcal { N } ( \mathsf { K } ) ^ { \perp } )$ . 

■ $\{ \boldsymbol { \nu } _ { \mathrm { j } } \} _ { \mathrm { j \in N } }$ in V (output space elements, form an orthonormal basis of $\mathcal { R } ( \mathsf { K } ) )$ . 

## SVD: How K Acts

The SVD components are linked by these key relationships: 

■ ${ \sf K u _ { j } } = { \sigma } _ { \mathrm { j } } { \nu } _ { \mathrm { j } }$ (The operator K maps $\mathfrak { u } _ { \mathrm { j } }$ to $\nu _ { \mathrm { j } }$ , scaled/attenuated by $\sigma _ { \mathrm { j } } )$ . 

■ $\mathsf { K } ^ { \ast } \boldsymbol { \nu } _ { \mathrm { j } } = \sigma _ { \mathrm { j } } \mathrm { u } _ { \mathrm { j } }$ (The adjoint operator $\mathsf { K } ^ { * }$ maps $\nu _ { \mathrm { j } }$ back to $\mathfrak { u } _ { \mathrm { j } }$ , also scaled by $\sigma _ { \mathrm { j } } )$ . 

## SVD: How K Acts

The SVD components are linked by these key relationships: 

■ ${ \sf K u _ { j } } = { \sigma } _ { \mathrm { j } } { \nu } _ { \mathrm { j } }$ (The operator K maps $\mathfrak { u } _ { \mathrm { j } }$ to $\nu _ { \mathrm { j } }$ , scaled/attenuated by $\sigma _ { \mathrm { j } } )$ . 

■ $\mathsf { K } ^ { \ast } \boldsymbol { \nu } _ { \mathrm { j } } = \sigma _ { \mathrm { j } } \mathrm { u } _ { \mathrm { j } }$ (The adjoint operator $\mathsf { K } ^ { * }$ maps $\nu _ { \mathrm { j } }$ back to $\mathfrak { u } _ { \mathrm { j } }$ , also scaled by $\sigma _ { \mathrm { j } } )$ . 

## Representation of K’s Action

Any $w \in \mathcal { U }$ can be represented using the $\mathfrak { u } _ { \mathrm { j } }$ , and K acts on w as 

$$
K w = \sum_ {j = 1} ^ {\infty} \sigma_ {j} \underbrace {\left\langle w , u _ {j} \right\rangle_ {\mathcal {U}}} _ {\text { coeff   of   } w \text {   along   } u _ {j}} v _ {j}
$$

| {z }<sub>coeff.</sub> <sub>of</sub> <sub>w</sub> <sub>along</sub> <sub>uj</sub> 

## SVD: How K Acts

■ ${ \sf K u _ { j } } = { \sigma } _ { \mathrm { j } } { \nu } _ { \mathrm { j } }$ (The operator K maps $\nu _ { \mathrm { j } }$ to $\nu _ { \mathrm { j } }$ , scaled/attenuated by $\sigma _ { \mathrm { j } } )$ . 

■ $\mathsf { K } ^ { \ast } \boldsymbol { \nu } _ { \mathrm { j } } = \sigma _ { \mathrm { j } } \mathrm { u } _ { \mathrm { j } }$ (The adjoint operator $\mathsf { K } ^ { * }$ maps $\nu _ { \mathrm { j } }$ back to $\mathfrak { u } _ { \mathrm { j } }$ , also scaled by $\sigma _ { \mathrm { j } } )$ . 

## Representation of K’s Action

Any $w \in \mathcal { U }$ can be represented using the $\mathfrak { u } _ { \mathrm { j } }$ , and K acts on w as 

$$
K w = \sum_ {j = 1} ^ {\infty} \sigma_ {j}
$$

$$
\underbrace {\langle w , u _ {j} \rangle_ {\mathcal {U}}}
$$

$$
\nu_ {j}
$$

| {z }<sub>coeff.</sub> <sub>of</sub> <sub>w</sub> <sub>along</sub> $\nu _ { \mathrm { j } }$ 

## SVD and Ill-Posedness

Decay of singular values $\sigma _ { \mathrm { j } }  0$ is primary source of ill-posedness for compact operators. To find u from ${ \mathrm { k u } } = { \mathrm { f } }$ , one might think of $\mathrm { u } \approx \sum \frac { 1 } { \sigma _ { \mathrm { j } } } \left. \mathrm { f } , \nu _ { \mathrm { j } } \right. \mathrm { u } _ { \mathrm { j } }$ $\sigma _ { \mathrm { j } }$ become very small, any noise in $\left. \mathbf { f } , \nu _ { \mathrm { j } } \right.$ gets greatly amplified. 

## SVD for Integration: The Forward Operator

We revisit the inverse problem of differentiation. The forward operator $\mathsf K : \mathsf L ^ { 2 } ( [ 0 , 1 ] ) \to \mathsf L ^ { 2 } ( [ 0 , 1 ] )$ is the integration operator 

$$
(K u) (y) = \int_ {0} ^ {y} u (x) d x.
$$

## SVD for Integration: The Forward Operator

We revisit the inverse problem of differentiation. The forward operator $\mathsf K : \mathsf L ^ { 2 } ( [ 0 , 1 ] ) \to \mathsf L ^ { 2 } ( [ 0 , 1 ] )$ is the integration operator 

$$
(K u) (y) = \int_ {0} ^ {y} u (x) d x.
$$

This can be written as an integral operator with kernel $\mathsf { k } ( \mathsf { x } , \mathsf { y } )$ , i.e. 

$$
(K u) (y) = \int_ {0} ^ {1} k (x, y) u (x) d x, \quad \text { where } k (x, y) = \left\{ \begin{array}{l l} 1 & \text { if } x \leqslant y \\ 0 & \text { if } x > y \end{array} \right..
$$

## SVD for Integration: The Forward Operator

We revisit the inverse problem of differentiation. The forward operator $\mathsf K : \mathsf L ^ { 2 } ( [ 0 , 1 ] ) \to \mathsf L ^ { 2 } ( [ 0 , 1 ] )$ is the integration operator 

$$
(K u) (y) = \int_ {0} ^ {y} u (x) d x.
$$

This can be written as an integral operator with kernel $\mathsf { k } ( \mathsf { x } , \mathsf { y } )$ , i.e. 

$$
(K u) (y) = \int_ {0} ^ {1} k (x, y) u (x) d x, \quad \text { where } k (x, y) = \left\{ \begin{array}{l l} 1 & \text { if } x \leqslant y \\ 0 & \text { if } x > y \end{array} \right..
$$

This operator K is compact. Our goal is to find its SVD. 

## Computing the Adjoint Operator $\mathsf { K } ^ { * }$

The adjoint operator $\mathsf { K } ^ { * }$ is defined by $\langle \mathsf { K u } , \nu \rangle _ { \mathsf { L } ^ { 2 } ( [ 0 , 1 ] ) } = \langle \mathsf { u } , \mathsf { K } ^ { * } \nu \rangle _ { \mathsf { L } ^ { 2 } ( [ 0 , 1 ] ) }$ . We observe 

$$
\begin{array}{l} \langle K u, v \rangle = \int_ {0} ^ {1} \left(\int_ {0} ^ {y} u (x) d x\right) v (y) d y \\ = \int_ {0} ^ {1} \int_ {0} ^ {1} k (x, y) u (x) v (y) d x d y \\ = \int_ {0} ^ {1} u (x) \left(\int_ {x} ^ {1} v (y) d y\right) d x \end{array}
$$

## Computing the Adjoint Operator $\mathsf { K } ^ { * }$

The adjoint operator $\mathsf { K } ^ { * }$ is defined by $\langle \mathsf { K u } , \nu \rangle _ { \mathsf { L } ^ { 2 } ( [ 0 , 1 ] ) } = \langle \mathsf { u } , \mathsf { K } ^ { * } \nu \rangle _ { \mathsf { L } ^ { 2 } ( [ 0 , 1 ] ) }$ . We observe 

$$
\begin{array}{l} \langle K u, v \rangle = \int_ {0} ^ {1} \left(\int_ {0} ^ {y} u (x) d x\right) v (y) d y \\ = \int_ {0} ^ {1} \int_ {0} ^ {1} k (x, y) u (x) v (y) d x d y \\ = \int_ {0} ^ {1} u (x) \left(\int_ {x} ^ {1} v (y) d y\right) d x \end{array}
$$

Thus, the adjoint operator $\mathsf { K } ^ { \ast } : \mathsf { L } ^ { 2 } ( [ 0 , 1 ] ) \to \mathsf { L } ^ { 2 } ( [ 0 , 1 ] )$ is given by 

$$
(K ^ {*} v) (x) = \int_ {x} ^ {1} v (y) d y.
$$

## The Operator $\mathsf { K } ^ { * } \mathsf { K }$ and Eigenvalue Problem Next, we form the operator ${ \sf K } ^ { * } { \sf K } ,$ , i.e.

$$
\begin{array}{l} (K ^ {*} K u) (x) = K ^ {*} ((K u) (\cdot)) (x) \\ \qquad = \int_ {x} ^ {1} \left(\int_ {0} ^ {y} u (z) d z\right) d y. \end{array}
$$

## The Operator $\mathsf { K } ^ { * } \mathsf { K }$ and Eigenvalue Problem

Next, we form the operator ${ \sf K } ^ { * } { \sf K } ,$ , i.e. 

$$
\begin{array}{l} (K ^ {*} K u) (x) = K ^ {*} ((K u) (\cdot)) (x) \\ \qquad = \int_ {x} ^ {1} \left(\int_ {0} ^ {y} u (z) d z\right) d y. \end{array}
$$

We seek eigenvalues $\lambda = \sigma ^ { 2 } > 0$ and eigenfunctions $\mathfrak { u } \in \mathrm { L } ^ { 2 } ( [ 0 , 1 ] )$ for ${ \sf K } ^ { * } { \sf K } .$ , i.e. 

$$
\mathrm{K} ^ {*} \mathrm{Ku} = \lambda u.
$$

## The Operator $\mathsf { K } ^ { * } \mathsf { K }$ and Eigenvalue Problem Next, we form the operator ${ \sf K } ^ { * } { \sf K } ,$ , i.e.

$$
\begin{array}{l} (K ^ {*} K u) (x) = K ^ {*} ((K u) (\cdot)) (x) \\ \qquad = \int_ {x} ^ {1} \left(\int_ {0} ^ {y} u (z) d z\right) d y. \end{array}
$$

We seek eigenvalues $\lambda = \sigma ^ { 2 } > 0$ and eigenfunctions $\mathfrak { u } \in \mathrm { L } ^ { 2 } ( [ 0 , 1 ] )$ for ${ \sf K } ^ { * } { \sf K } ,$ i.e. 

$$
\mathrm{K} ^ {*} \mathrm{Ku} = \lambda u.
$$

This leads to the solving the integral equation 

$$
\int_ {x} ^ {1} \left(\int_ {0} ^ {y} u (z) d z\right) d y = \lambda u (x).
$$

## Deriving the ODE and Boundary Conditions

Differentiating with respect to x (using Leibniz integral rule) yields 

$$
\lambda u ^ {\prime} (x) = - \int_ {0} ^ {x} u (z) d z.
$$

From this, setting $x = 0$ , we get $\mathrm { u } ^ { \prime } ( 0 ) = 0$ 

## Deriving the ODE and Boundary Conditions

Differentiating with respect to x (using Leibniz integral rule) yields 

$$
\lambda u ^ {\prime} (x) = - \int_ {0} ^ {x} u (z) d z.
$$

From this, setting $x = 0$ , we get $\mathrm { u } ^ { \prime } ( 0 ) = 0$ 

Differentiating again with respect to x gives $\lambda \mathrm { u } ^ { \prime \prime } ( x ) = - \mathrm { u } ( x )$ . This leads to the following Ordinary Differential Equation (ODE): 

$$
\lambda u ^ {\prime \prime} (x) + u (x) = 0
$$

## Deriving the ODE and Boundary Conditions

Differentiating with respect to x (using Leibniz integral rule) yields 

$$
\lambda u ^ {\prime} (x) = - \int_ {0} ^ {x} u (z) d z.
$$

From this, setting $x = 0$ , we get $\mathrm { u } ^ { \prime } ( 0 ) = 0$ 

Differentiating again with respect to x gives $\lambda \mathrm { u } ^ { \prime \prime } ( x ) = - \mathrm { u } ( x )$ . This leads to the following Ordinary Differential Equation (ODE): 

$$
\lambda u ^ {\prime \prime} (x) + u (x) = 0
$$

From the integral equation for $\lambda \mathfrak { u } ( \mathfrak { x } )$ , if we set $x = 1$ , the outer integral vanishes, i.e. $\lambda \mathfrak { u } ( 1 ) = 0$ . Since we seek $\lambda > 0$ , we must have $\mathfrak { u } ( 1 ) = 0$ 

## Deriving the ODE and Boundary Conditions

Differentiating with respect to x (using Leibniz integral rule) yields 

$$
\lambda u ^ {\prime} (x) = - \int_ {0} ^ {x} u (z) d z.
$$

From this, setting $x = 0$ , we get $\mathrm { u } ^ { \prime } ( 0 ) = 0$ 

Differentiating again with respect to x gives $\lambda \mathrm { u } ^ { \prime \prime } ( x ) = - \mathrm { u } ( x )$ . This leads to the following Ordinary Differential Equation (ODE): 

$$
\lambda u ^ {\prime \prime} (x) + u (x) = 0
$$

From the integral equation for $\lambda \mathfrak { u } ( \mathfrak { x } )$ , if we set x = 1, the outer integral vanishes, i.e. $\lambda \mathfrak { u } ( 1 ) = 0$ . Since we seek $\lambda > 0$ , we must have $\mathfrak { u } ( 1 ) = 0$ 

## Summary of ODE Problem

We need to solve $\lambda { \mathfrak { u } } ^ { \prime \prime } ( x ) + { \mathfrak { u } } ( x ) = 0$ with boundary conditions $\mathrm { u } ^ { \prime } ( 0 ) = 0$ and $\mathfrak { u } ( 1 ) = 0$ 

## Singular Values $\sigma _ { \mathrm { j } }$ and Orthonormal Functions $\mathfrak { u } _ { \mathrm { j } }$

The general solution to $\begin{array} { r } { \mathfrak { u } ^ { \prime \prime } ( x ) + \frac { 1 } { \lambda } \mathfrak { u } ( x ) = 0 \mathrm { i } \mathtt { s } \mathfrak { u } ( x ) = c _ { 1 } \mathtt { s i n } ( x / \sqrt { \lambda } ) + c _ { 2 } \mathtt { c o s } ( x / \sqrt { \lambda } ) } \end{array}$ . Let $\sigma : = { \sqrt { \lambda } } .$ 

A $\begin{array} { r } { \mathrm { p p l y ~ u ^ { \prime } ( 0 ) } = 0 \colon \mathrm { u ^ { \prime } ( x ) } = \frac { \mathfrak { c } _ { 1 } } { \sigma } \cos ( \mathrm { x / \sigma } ) - \frac { \mathfrak { c } _ { 2 } } { \sigma } \sin ( \mathrm { x / \sigma } ) \cdot \mathrm { u ^ { \prime } ( 0 ) } = \frac { \mathfrak { c } _ { 1 } } { \sigma } = 0 \implies \mathfrak { c } _ { 1 } = 0 . \mathrm { S o } , } \end{array}$ $\mathsf { u } ( \mathsf { x } ) = \mathsf { c } _ { 2 } \mathsf { c o s } ( \mathsf { x } / \sigma )$ 

## Singular Values $\sigma _ { \mathrm { j } }$ and Orthonormal Functions $\mu _ { \mathrm { j } }$

The general solution to $\begin{array} { r } { \mathfrak { u } ^ { \prime \prime } ( x ) + \frac { 1 } { \lambda } \mathfrak { u } ( x ) = 0 \mathrm { i } \mathtt { s } \mathfrak { u } ( x ) = c _ { 1 } \mathtt { s i n } ( x / \sqrt { \lambda } ) + c _ { 2 } \mathtt { c o s } ( x / \sqrt { \lambda } ) } \end{array}$ . Let $\sigma : = { \sqrt { \lambda } } .$ 

Apply $\begin{array} { r } { \mathrm { ~  ~ u ~ } ^ { \prime } ( 0 ) = 0 \colon \mathrm {  ~ u ~ } ^ { \prime } ( x ) = \frac { c _ { 1 } } { \sigma } \cos ( x / \sigma ) - \frac { c _ { 2 } } { \sigma } \sin ( x / \sigma ) . \mathrm { ~  ~ u ~ } ^ { \prime } ( 0 ) = \frac { c _ { 1 } } { \sigma } = 0 \Longrightarrow \mathrm { ~  ~ c _ { 1 } = 0 . ~ } \mathrm { S o } , } \end{array}$ $\mathsf { u } ( \mathsf { x } ) = \mathsf { c } _ { 2 } \mathsf { c o s } ( \mathsf { x } / \sigma )$ 

■ $\mathsf { A p p } | \mathsf { y } \mathrm { u } ( 1 ) = 0 { \mathrm { : ~ } } \mathsf { c } _ { 2 } \mathsf { c o s } ( 1 / \sigma ) = 0$ . For non-trivial solutions $( \mathrm { c } _ { 2 } \ne 0 )$ , we require cos(1/σ) = 0. This means $1 / \sigma = ( \mathrm { j } - \textstyle { \frac { 1 } { 2 } } ) \pi \mathsf { f o r } \mathrm { j } \in \mathbb { N } .$ 

## Singular Values $\sigma _ { \mathrm { j } }$ and Orthonormal Functions $\mu _ { \mathrm { j } }$

The general solution to $\begin{array} { r } { \mathfrak { u } ^ { \prime \prime } ( x ) + \frac { 1 } { \lambda } \mathfrak { u } ( x ) = 0 \mathrm { i } \mathtt { s } \mathfrak { u } ( x ) = c _ { 1 } \mathtt { s i n } ( x / \sqrt { \lambda } ) + c _ { 2 } \mathtt { c o s } ( x / \sqrt { \lambda } ) } \end{array}$ . Let $\sigma : = { \sqrt { \lambda } } .$ 

A $\begin{array} { r } { \mathsf { s p p l y } \mathsf { u } ^ { \prime } ( 0 ) = 0 { \boldsymbol { : } } \mathsf { u } ^ { \prime } ( { \boldsymbol { x } } ) = \frac { \mathtt { c } _ { 1 } } { \sigma } \mathsf { c o s } \big ( { \boldsymbol { x } } / \sigma \big ) - \frac { \mathtt { c } _ { 2 } } { \sigma } \mathsf { s i n } \big ( { \boldsymbol { x } } / \sigma \big ) . \mathsf { u } ^ { \prime } ( 0 ) = \frac { \mathtt { c } _ { 1 } } { \sigma } = 0 \implies \mathsf { c } _ { 1 } = 0 . \mathsf { S o } , } \end{array}$ $\mathsf { u } ( \mathsf { x } ) = \mathsf { c } _ { 2 } \mathsf { c o s } ( \mathsf { x } / \sigma )$ 

■ $\mathsf { A p p } | \mathsf { y } \mathrm { u } ( 1 ) = 0 { \mathrm { : ~ } } \mathsf { c } _ { 2 } \mathsf { c o s } ( 1 / \sigma ) = 0$ . For non-trivial solutions $( \mathrm { c } _ { 2 } \ne 0 )$ , we require cos(1/σ) = 0. This means $1 / \sigma = ( \mathrm { j } - \textstyle { \frac { 1 } { 2 } } ) \pi \mathsf { f o r } \mathrm { j } \in \mathbb { N } .$ 

The singular values are $\begin{array} { r } { \sigma _ { \mathfrak { j } } = \frac { 1 } { ( \mathtt { j } - \frac { 1 } { 2 } ) \pi } = \frac { 2 } { ( 2 \mathtt { j } - 1 ) \pi } \mathrm { ~ f o r ~ \mathtt { j } \in \mathbb { N } . } } \end{array}$ 

## Singular Values $\sigma _ { \mathrm { j } }$ and Orthonormal Functions $\mu _ { \mathrm { j } }$

The general solution to $\begin{array} { r } { \mathfrak { u } ^ { \prime \prime } ( x ) + \frac { 1 } { \lambda } \mathfrak { u } ( x ) = 0 \mathrm { i } \mathtt { s } \mathfrak { u } ( x ) = c _ { 1 } \mathtt { s i n } ( x / \sqrt { \lambda } ) + c _ { 2 } \mathtt { c o s } ( x / \sqrt { \lambda } ) } \end{array}$ . Let $\sigma : = { \sqrt { \lambda } } .$ 

Apply $\begin{array} { r } { \mathbf { u } ^ { \prime } ( 0 ) = 0 { : } \mathbf { u } ^ { \prime } ( x ) = \frac { c _ { 1 } } { \sigma } \cos ( x / \sigma ) - \frac { c _ { 2 } } { \sigma } \sin ( x / \sigma ) { . } ~ \mathbf { u } ^ { \prime } ( 0 ) = \frac { c _ { 1 } } { \sigma } = 0 \implies c _ { 1 } = 0 . ~ \mathrm { S } 0 , } \end{array}$ $\mathsf { u } ( \mathsf { x } ) = \mathsf { c } _ { 2 } \mathsf { c o s } ( \mathsf { x } / \sigma )$ 

■ $\mathsf { A p p } | \mathsf { y } \mathrm { u } ( 1 ) = 0 { \mathrm { : ~ } } \mathsf { c } _ { 2 } \mathsf { c o s } ( 1 / \sigma ) = 0$ . For non-trivial solutions $( \mathrm { c } _ { 2 } \ne 0 )$ , we require cos $( 1 / \sigma ) = 0$ . This means $1 / \sigma = ( \mathrm { j } - \textstyle { \frac { 1 } { 2 } } ) \pi \mathsf { f o r } \mathrm { j } \in \mathbb { N } .$ 

The singular values are $\begin{array} { r } { \sigma _ { \mathrm { j } } = \frac { 1 } { ( \mathrm { j } - \frac { 1 } { 2 } ) \pi } = \frac { 2 } { ( 2 \mathrm { j } - 1 ) \pi } \ : \mathsf { f o r } \ : \mathrm { j } \in \mathbb { N } } \end{array}$ . The corresponding normalised eigenfunctions $\mathrm { u _ { j } } ( \boldsymbol { x } )$ (after choosing $c _ { 2 } = { \sqrt { 2 } }$ for normalisation) are 

$$
u _ {j} (x) = \sqrt {2} \cos \left(\left(j - \frac {1}{2}\right) \pi x\right)
$$

## Computing Orthonormal Functions $\nu _ { \mathrm { j } }$

The singular functions $\nu _ { \mathrm { j } }$ are obtained via $\begin{array} { r } { \nu _ { \mathrm { j } } = \frac { 1 } { \sigma _ { \mathrm { j } } } \mathsf { K u } _ { \mathrm { j } } . } \end{array}$ , i.e. 

$$
\begin{array}{l} (K u _ {j}) (x) = \int_ {0} ^ {x} u _ {j} (y) d y = \int_ {0} ^ {x} \sqrt {2} \cos \left(\left(j - \frac {1}{2}\right) \pi y\right) d y, \\ \qquad = \sqrt {2} \left[ \frac {\sin ((j - \frac {1}{2}) \pi y)}{(j - \frac {1}{2}) \pi} \right] _ {0} ^ {x} = \sqrt {2} \frac {\sin ((j - \frac {1}{2}) \pi x)}{(j - \frac {1}{2}) \pi}, \\ \qquad = \sigma_ {j} \sqrt {2} \sin \left(\left(j - \frac {1}{2}\right) \pi x\right). \end{array}
$$

## Computing Orthonormal Functions $\nu _ { \mathrm { j } }$

The singular functions $\nu _ { \mathrm { j } }$ are obtained via $\begin{array} { r } { \nu _ { \mathrm { j } } = \frac { 1 } { \sigma _ { \mathrm { j } } } \mathsf { K u } _ { \mathrm { j } } . } \end{array}$ , i.e. 

$$
\begin{array}{l} (K u _ {j}) (x) = \int_ {0} ^ {x} u _ {j} (y) d y = \int_ {0} ^ {x} \sqrt {2} \cos \left(\left(j - \frac {1}{2}\right) \pi y\right) d y, \\ \qquad = \sqrt {2} \left[ \frac {\sin ((j - \frac {1}{2}) \pi y)}{(j - \frac {1}{2}) \pi} \right] _ {0} ^ {x} = \sqrt {2} \frac {\sin ((j - \frac {1}{2}) \pi x)}{(j - \frac {1}{2}) \pi}, \\ \qquad = \sigma_ {j} \sqrt {2} \sin \left(\left(j - \frac {1}{2}\right) \pi x\right). \end{array}
$$

Therefore, 

$$
v _ {j} (x) = \frac {1}{\sigma_ {j}} (K u _ {j}) (x) = \sqrt {2} \sin \left(\left(j - \frac {1}{2}\right) \pi x\right)
$$

## SVD Summary for Integration Operator

For the integration operator $\begin{array} { r } { \mathsf { K } : \mathsf { L } ^ { 2 } ( [ 0 , 1 ] ) \to \mathsf { L } ^ { 2 } ( [ 0 , 1 ] ) , ( \mathsf { K } \mathsf { u } ) ( \mathsf { y } ) = \int _ { 0 } ^ { \mathsf { y } } \mathsf { u } ( x ) \mathrm { d } x \mathrm { . } } \end{array}$ 

Singular values: $\sigma _ { \mathrm { j } } = 2 / ( ( 2 \mathrm { j } - 1 ) \pi ) \ \mathsf { f o r \ \mathrm { j } \in N } .$ 

## SVD Summary for Integration Operator

For the integration operator $\begin{array} { r } { \mathsf { K } : \mathsf { L } ^ { 2 } ( [ 0 , 1 ] ) \to \mathsf { L } ^ { 2 } ( [ 0 , 1 ] ) , ( \mathsf { K } \mathsf { u } ) ( \mathsf { y } ) = \int _ { 0 } ^ { \mathsf { y } } \mathsf { u } ( x ) \mathrm { d } x \mathrm { . } } \end{array}$ 

Singular values: $\sigma _ { \mathrm { j } } = 2 / ( ( 2 \mathrm { j } - 1 ) \pi ) \ \mathsf { f o r \ \mathrm { j } } \in \mathbb { N } .$ 

Orthonormal functions $( \mathrm { u _ { j } } ) \mathrm { : \ u _ { j } ( } \mathrm { x } ) = \sqrt { 2 } \cos \left( \left( \mathrm { j } - \textstyle { \frac { 1 } { 2 } } \right) \pi \mathrm { x } \right)$ 

## SVD Summary for Integration Operator

For the integration operator $\begin{array} { r } { \mathsf { K } : \mathsf { L } ^ { 2 } ( [ 0 , 1 ] ) \to \mathsf { L } ^ { 2 } ( [ 0 , 1 ] ) , ( \mathsf { K } \mathsf { u } ) ( \mathsf { y } ) = \int _ { 0 } ^ { \mathsf { y } } \mathsf { u } ( x ) \mathrm { d } x \mathrm { . } } \end{array}$ 

Singular values: $\sigma _ { \mathrm { j } } = 2 / ( ( 2 \mathrm { j } - 1 ) \pi ) \ \mathsf { f o r \ \mathrm { j } } \in \mathbb { N } .$ 

Orthonormal functions $( \mathrm { u _ { j } } ) \mathrm { : \ u _ { j } ( } \mathrm { x } ) = \sqrt { 2 } \cos \left( \left( \mathrm { j } - \textstyle { \frac { 1 } { 2 } } \right) \pi \mathrm { x } \right)$ 

Orthonormal functions $( \nu _ { \mathrm { j } } ) \colon \nu _ { \mathrm { j } } ( x ) = { \sqrt { 2 } } \sin \left( \left( \mathrm { j } - { \textstyle { \frac { 1 } { 2 } } } \right) \pi x \right)$ 

## SVD Summary for Integration Operator

For the integration operator $\begin{array} { r } { \mathsf { K } : \mathsf { L } ^ { 2 } ( [ 0 , 1 ] ) \to \mathsf { L } ^ { 2 } ( [ 0 , 1 ] ) , ( \mathsf { K } \mathsf { u } ) ( \mathsf { y } ) = \int _ { 0 } ^ { \mathsf { y } } \mathsf { u } ( x ) \mathrm { d } x \mathrm { . } } \end{array}$ 

Singular values: $\begin{array} { r } { \sigma _ { \mathrm { j } } = 2 / ( ( 2 \mathrm { j } - 1 ) \pi ) \mathrm { f o r ~ \mathrm { j } \in \mathbb { N } } . } \end{array}$ 

Orthonormal functions $( \mathrm { u _ { j } } ) \mathrm { : \ u _ { j } ( } \mathrm { x } ) = \sqrt { 2 } \cos \left( \left( \mathrm { j } - \textstyle { \frac { 1 } { 2 } } \right) \pi \mathrm { x } \right)$ 

Orthonormal functions $( \nu _ { \mathrm { j } } ) \colon \nu _ { \mathrm { j } } ( x ) = { \sqrt { 2 } } \sin \left( \left( \mathrm { j } - { \textstyle { \frac { 1 } { 2 } } } \right) \pi x \right)$ 

## Expansion of Kw

$$
\begin{array}{l} (K w) (x) = \sum_ {j = 1} ^ {\infty} \frac {2}{(2 j - 1) \pi} \left(\int_ {0} ^ {1} w (s) \sqrt {2} \cos \left(\left(j - \frac {1}{2}\right) \pi s\right) d s\right) \sqrt {2} \sin \left(\left(j - \frac {1}{2}\right) \pi x\right) \\ = \sum_ {j = 1} ^ {\infty} \frac {4}{(2 j - 1) \pi} \left(\int_ {0} ^ {1} w (s) \cos \left(\left(j - \frac {1}{2}\right) \pi s\right) d s\right) \sin \left(\left(j - \frac {1}{2}\right) \pi x\right) \end{array}
$$

## Fundamental Concepts in Regularisation

## Motivation for Regularisation

## Why Regularisation?

Since many inverse problems Ku = f are ill-posed (especially regarding stability), direct inversion or naive solutions are often highly sensitive to noise in the data f. 

## Motivation for Regularisation

## Why Regularisation?

Since many inverse problems Ku = f are ill-posed (especially regarding stability), direct inversion or naive solutions are often highly sensitive to noise in the data f. 

Regularisation methods aim to find stable approximate solutions by incorporating prio knowledge or preferences about the solution u. 

## Motivation for Regularisation

## Why Regularisation?

Since many inverse problems Ku = f are ill-posed (especially regarding stability), direct inversion or naive solutions are often highly sensitive to noise in the data f. 

Regularisation methods aim to find stable approximate solutions by incorporating prio knowledge or preferences about the solution u. 

We need a formal framework to define what constitutes a ”good” regularisation. 

## General Regularisation Operators: Definition

Let U, V be metric spaces. For $x \in X , S \subset X .$ , define $\mathrm { d } ( x , S ) : = \mathsf { i n f } _ { \nu \in S } \mathrm { d } ( x , \nu )$ 

## Regularisation Operator [1]

Set-valued operators $\mathtt { R } _ { \alpha } : \mathcal { V } \Longrightarrow \mathcal { U }$ (parameterised by $\alpha \in \mathcal { A } \subset \mathbb { R } ^ { \mathsf { m } } )$ ) are called regularisation operators if for each fixed $\alpha \in A$ and for all $\mathsf { f } ^ { \delta } \in \mathcal { V }$ and sequences $\mathsf { f } ^ { \delta _ { \mathrm { n } } } \in \mathcal { V }$ converging to $\mathsf { f } ^ { \delta } ( \mathsf { i . e . , d } _ { \mathcal { V } } ( \mathsf { f } ^ { \delta _ { \mathrm { n } } } , \mathsf { f } ^ { \delta } )    0 )$ , we have 

$$
\emptyset \neq \left\{u \in \mathcal {U}   \middle |   \operatorname * {l i m s u p} _ {k \to \infty} d _ {\mathcal {U}} (u, R _ {\alpha} (f ^ {\delta_ {k}})) = 0 \right\} \subset R _ {\alpha} (f ^ {\delta})
$$

The set in the middle is the Kuratowski limit inferior of the sequence of sets $\mathsf { R } _ { \alpha } ( \mathsf { f } ^ { \delta _ { \mathrm { n } } } )$ 

## General Regularisation Operators: Definition

Let U, V be metric spaces. For $x \in X , S \subset X ,$ , define $\mathrm { d } ( x , S ) : = \mathsf { i n f } _ { \nu \in S } \mathrm { d } ( x , \nu )$ 

## Regularisation Operator [1]

Set-valued operators $\mathtt { R } _ { \alpha } : \mathcal { V } \Longrightarrow \mathcal { U }$ (parameterised by $\alpha \in \mathcal { A } \subset \mathbb { R } ^ { \mathsf { m } } )$ ) are called regularisation operators if for each fixed $\alpha \in A$ and for all $\mathsf { f } ^ { \delta } \in \mathcal { V }$ and sequences $\mathsf { f } ^ { \delta _ { \mathrm { n } } } \in \mathcal { V }$ converging to $\mathsf { f } ^ { \delta } ( \mathsf { i . e . , d } _ { \mathcal { V } } ( \mathsf { f } ^ { \delta _ { \mathrm { n } } } , \mathsf { f } ^ { \delta } )    0 )$ , we have 

$$
\emptyset \neq \left\{u \in \mathcal {U}   \middle |   \operatorname * {l i m s u p} _ {k \to \infty} d _ {\mathcal {U}} (u, R _ {\alpha} (f ^ {\delta_ {k}})) = 0 \right\} \subset R _ {\alpha} (f ^ {\delta})
$$

The set in the middle is the Kuratowski limit inferior of the sequence of sets $\mathsf { R } _ { \alpha } ( \mathsf { f } ^ { \delta _ { \mathrm { n } } } )$ 

## Regularisation Method

A regularisation operator $\mathtt { R } _ { \alpha }$ together with a parameter choice strategy $\alpha _ { \mathtt { C h o i c e } } : ( 0 , \delta _ { 0 } ) \times \mathcal { V }  \mathtt { A }$ , denoted $\alpha ( \delta , \mathsf { f } ^ { \delta } )$ , forms a regularisation method. 

## Regularisation Operators: Single-Valued Case

## Simplification for Single-Valued Operators

If $\mathbb { R } _ { \alpha } : \mathcal { V }  \mathcal { U }$ is a single-valued operator for each $\alpha \in A$ then for $\mathtt { R } _ { \alpha }$ to be a regularisation operator the previous condition simplifies to $\mathsf { R } _ { \alpha }$ being continuous on V. 

## Regularisation Operators: Single-Valued Case

## Simplification for Single-Valued Operators

If $\mathbb { R } _ { \alpha } : \mathcal { V }  \mathcal { U }$ is a single-valued operator for each $\alpha \in A$ then for $\mathtt { R } _ { \alpha }$ to be a regularisation operator the previous condition simplifies to $\mathsf { R } _ { \alpha }$ being continuous on V.That is, if $\mathsf { f } ^ { \delta _ { \mathrm { n } } } \to \mathsf { f } ^ { \delta }$ in V, then $\mathsf { R } _ { \alpha } ( \mathsf { f } ^ { \delta _ { \mathrm { n } } } ) \to \mathsf { R } _ { \alpha } ( \mathsf { f } ^ { \delta } )$ in U. 

## Regularisation Operators: Single-Valued Case

## Simplification for Single-Valued Operators

If $\mathbb { R } _ { \alpha } : \mathcal { V }  \mathcal { U }$ is a single-valued operator for each $\alpha \in A$ then for $\mathtt { R } _ { \alpha }$ to be a regularisation operator the previous condition simplifies to $\mathsf { R } _ { \alpha }$ being continuous on V.That is, if $\mathsf { f } ^ { \delta _ { \mathrm { n } } } \to \mathsf { f } ^ { \delta }$ in V, then $\mathsf { R } _ { \alpha } ( \mathsf { f } ^ { \delta _ { \mathrm { n } } } ) \to \mathsf { R } _ { \alpha } ( \mathsf { f } ^ { \delta } )$ in U. 

Verification: If $\mathtt { R } _ { \alpha }$ is continuous and $\mathfrak { u } _ { \mathfrak { n } } = \mathbb { R } _ { \alpha } ( \mathsf { f } ^ { \delta _ { \mathfrak { n } } } ) \to \mathbb { R } _ { \alpha } ( \mathsf { f } ^ { \delta } ) = \mathfrak { u }$ , then $\{ \nu | \mathsf { l i m } _ { \mathbf { k } } \| \nu - \mathsf { u } _ { \mathbf { k } } \| _ { \mathcal { U } } = 0 \} = \{ \mathrm { u } \}$ . The stability condition becomes $\emptyset \neq \{ \mathrm { u } \} \subset \{ \mathrm { u } \}$ , which is true. 

## Example: Spectral Regularisation Operators

We will now examine a common family of regularisation operators known as spectral regularisation operators. 

## General Form of Spectral Regularisation

For a compact linear operator $\mathsf { K } : \mathcal { U } \to \mathcal { V }$ between Hilbert spaces with SVD $\{ ( \sigma _ { \mathrm { j } } , \psi _ { \mathrm { j } } , \nu _ { \mathrm { j } } ) \}$ spectral regularisation operators $\mathbb { R } _ { \propto } : \mathcal { V }  \mathcal { U }$ are defined by 

$$
R _ {\alpha} f = \sum_ {j = 1} ^ {\infty} g _ {\alpha} (\sigma_ {j}) \left\langle f, v _ {j} \right\rangle_ {\mathcal {V}} u _ {j}
$$

## Example: Spectral Regularisation Operators

We will now examine a common family of regularisation operators known as spectral regularisation operators. 

## General Form of Spectral Regularisation

For a compact linear operator $\mathsf { K } : \mathcal { U } \to \mathcal { V }$ between Hilbert spaces with SVD $\{ ( \sigma _ { \mathrm { j } } , \psi _ { \mathrm { j } } , \nu _ { \mathrm { j } } ) \}$ spectral regularisation operators $\mathbb { R } _ { \propto } : \mathcal { V }  \mathcal { U }$ are defined by 

$$
R _ {\alpha} f = \sum_ {j = 1} ^ {\infty} g _ {\alpha} (\sigma_ {j}) \left\langle f, v _ {j} \right\rangle_ {\mathcal {V}} u _ {j}
$$

$\alpha > 0$ is the regularisation parameter. 

## Example: Spectral Regularisation Operators

We will now examine a common family of regularisation operators known as spectral regularisation operators. 

## General Form of Spectral Regularisation

For a compact linear operator $\mathsf { K } : \mathcal { U } \to \mathcal { V }$ between Hilbert spaces with SVD $\{ ( \sigma _ { \mathrm { j } } , \psi _ { \mathrm { j } } , \nu _ { \mathrm { j } } ) \}$ spectral regularisation operators $\mathbb { R } _ { \propto } : \mathcal { V }  \mathcal { U }$ are defined by 

$$
R _ {\alpha} f = \sum_ {j = 1} ^ {\infty} g _ {\alpha} (\sigma_ {j}) \left\langle f, v _ {j} \right\rangle_ {\mathcal {V}} u _ {j}
$$

■ $\alpha > 0$ is the regularisation parameter. 

$9 \alpha : \mathbb { R } _ { > 0 }  \mathbb { R } _ { \geqslant 0 }$ are known as filter functions. They modify the way singular values contribute to the reconstruction. 

## Example: Spectral Regularisation Operators

We will now examine a common family of regularisation operators known as spectral regularisation operators. 

## General Form of Spectral Regularisation

For a compact linear operator $\mathsf { K } : \mathcal { U } \to \mathcal { V }$ between Hilbert spaces with SVD $\{ ( \sigma _ { \mathrm { j } } , \psi _ { \mathrm { j } } , \nu _ { \mathrm { j } } ) \}$ spectral regularisation operators $\mathbb { R } _ { \propto } : \mathcal { V }  \mathcal { U }$ are defined by 

$$
R _ {\alpha} f = \sum_ {j = 1} ^ {\infty} g _ {\alpha} (\sigma_ {j}) \left\langle f, v _ {j} \right\rangle_ {\mathcal {V}} u _ {j}
$$

■ $\alpha > 0$ is the regularisation parameter. 

$9 \alpha : \mathbb { R } _ { > 0 }  \mathbb { R } _ { \geqslant 0 }$ are known as filter functions. They modify the way singular values contribute to the reconstruction. 

These operators $\mathtt { R } _ { \alpha }$ are linear and single-valued. 

## Spectral $\mathtt { R } _ { \alpha }$ as Regularisation Operators

Recall from the previous slide: for a single-valued operator $\mathbb { R } _ { \alpha } : \mathcal { V }  \mathcal { U }$ to be a regularisation operator, it must be continuous on V for each fixed α. 

## Spectral $\mathtt { R } _ { \alpha }$ as Regularisation Operators

Recall from the previous slide: for a single-valued operator $\mathbb { R } _ { \alpha } : \mathcal { V }  \mathcal { U }$ to be a regularisation operator, it must be continuous on V for each fixed $\alpha .$ 

## Condition for Continuity of Spectral $\scriptstyle { \mathbb { B } }$

A spectral regularisation operator $\mathtt { R } _ { \alpha }$ is continuous if its associated filter function ${ \mathfrak { g } } _ { \alpha } ( \sigma )$ is bounded for fixed $\alpha ,$ i.e. 

$$
\sup _ {\sigma > 0} | g _ {\alpha} (\sigma) | \leqslant C _ {\alpha} <   \infty .
$$

If this holds, $\mathsf { R } _ { \alpha }$ is a bounded linear operator, because 

$$
\| R _ {\alpha} f \| _ {\mathcal {U}} ^ {2} = \sum_ {j = 1} ^ {\infty} | g _ {\alpha} (\sigma_ {j}) | ^ {2} | \left\langle f, v _ {j} \right\rangle_ {\mathcal {V}} | ^ {2} \leqslant C _ {\alpha} ^ {2} \sum_ {j = 1} ^ {\infty} | \left\langle f, v _ {j} \right\rangle_ {\mathcal {V}} | ^ {2} \leqslant C _ {\alpha} ^ {2} \| f \| _ {\mathcal {V}} ^ {2}.
$$

Thus, $\| \mathsf { R } _ { \alpha } \| _ { \mathcal { L } ( \mathcal { V } , \mathcal { U } ) } \leqslant C _ { \alpha }$ , which implies that $\mathtt { R } _ { \alpha }$ is continuous. 

## Spectral Regularisation Methods

If a spectral operator $\mathtt { R } _ { \alpha }$ is continuous for a fixed α (due to its filter function ${ \mathfrak { g } } _ { \propto } ( \sigma )$ being bounded by $C _ { \alpha } )$ , it qualifies as a regularisation operator. 

## Spectral Regularisation Methods

If a spectral operator $\mathtt { R } _ { \alpha }$ is continuous for a fixed α (due to its filter function ${ \mathfrak { g } } _ { \alpha } ( \sigma )$ being bounded by $C _ { \alpha } )$ , it qualifies as a regularisation operator. 

When such a regularisation operator $\mathtt { R } _ { \alpha }$ is combined with a parameter choice strategy $\alpha ( \delta , \mathsf { f } ^ { \delta } )$ , it forms a spectral regularisation method. 

## Spectral Regularisation Methods

If a spectral operator $\mathtt { R } _ { \alpha }$ is continuous for a fixed α (due to its filter function ${ \mathfrak { g } } _ { \alpha } ( \sigma )$ being bounded by $C _ { \alpha } )$ , it qualifies as a regularisation operator. 

When such a regularisation operator $\mathtt { R } _ { \alpha }$ is combined with a parameter choice strategy $\alpha ( \delta , \mathsf { f } ^ { \delta } )$ , it forms a spectral regularisation method. 

## Example: Tikhonov Regularisation [17, 16, 15]

The filter function for Tikhonov regularisation is ${ \mathfrak { g } } _ { \alpha } ( \sigma ) = \sigma / ( \sigma ^ { 2 } + \alpha )$ . 

For any fixed $\alpha > 0$ , this function is bounded: 

$$
\sup _ {\sigma > 0} \left| \frac {\sigma}{\sigma^ {2} + \alpha} \right| = \frac {1}{2 \sqrt {\alpha}} =: C _ {\alpha} <   \infty
$$

Hence, for each $\alpha > 0$ , the operator ${ \sf R } _ { \alpha }$ is a continuous regularisation operator. 

Paired with $\alpha ( \delta , { \sf f } ^ { \delta } ) , ( { \sf R } _ { \alpha } , \alpha ( \delta , { \sf f } ^ { \delta } ) )$ forms the Tikhonov regularisation method. 

## Tikhonov Regularisation in Spectral Form

## Tikhonov Filter Function

Applied to our integration operator $( \mathsf { K u } ) ( \mathsf { y } ) = \int _ { 0 } ^ { \mathsf { y } } \mathsf { u } ( \mathsf { x } ) \mathrm { d } \mathsf { x }$ , we have 

$$
(R _ {\alpha} f) (x) = \sum_ {j = 1} ^ {\infty} \frac {4 (2 j - 1) ^ {2} \pi^ {2}}{4 + \alpha (2 j - 1) ^ {2} \pi^ {2}} \left(\int_ {0} ^ {1} f (s) \sin \left(\left(j - \frac {1}{2}\right) \pi x\right) d s\right) \cos \left(\left(j - \frac {1}{2}\right) \pi x\right)
$$

## Tikhonov Regularisation in Spectral Form

## Tikhonov Filter Function

Applied to our integration operator $( \mathsf { K u } ) ( \mathsf { y } ) = \int _ { 0 } ^ { \mathsf { y } } \mathsf { u } ( \mathsf { x } ) \mathrm { d } \mathsf { x }$ , we have 

$$
(R _ {\alpha} f) (x) = \sum_ {j = 1} ^ {\infty} \frac {4 (2 j - 1) ^ {2} \pi^ {2}}{4 + \alpha (2 j - 1) ^ {2} \pi^ {2}} \left(\int_ {0} ^ {1} f (s) \sin \left(\left(j - \frac {1}{2}\right) \pi x\right) d s\right) \cos \left(\left(j - \frac {1}{2}\right) \pi x\right)
$$

## General Tikhonov Form

Note: for any bounded linear operator $\mathsf { K } : \mathcal { U } \to \mathcal { V }$ between Hilbert spaces, Tikhonov regularisation can be written as 

$$
R _ {\alpha} (f ^ {\delta}) = (K ^ {*} K + \alpha I) ^ {- 1} K ^ {*} f ^ {\delta}
$$

This form doesn’t require knowledge of the SVD and applies generally. 

## Tikhonov as an Optimisation Problem

## Variational Formulation

The Tikhonov regularised solution $\ R _ { \alpha } ( { \sf f ^ { \delta } } )$ is also the unique minimiser of 

$$
R _ {\alpha} (f ^ {\delta}) = \underset {u \in \mathcal {U}} {\arg \min} \left\{\frac {1}{2} \| K u - f ^ {\delta} \| _ {\mathcal {V}} ^ {2} + \frac {\alpha}{2} \| u \| _ {\mathcal {U}} ^ {2} \right\}.
$$

## Tikhonov as an Optimisation Problem

## Variational Formulation

The Tikhonov regularised solution $\ R _ { \alpha } ( { \sf f ^ { \delta } } )$ is also the unique minimiser of 

$$
R _ {\alpha} (f ^ {\delta}) = \underset {u \in \mathcal {U}} {\arg \min} \left\{\frac {1}{2} \| K u - f ^ {\delta} \| _ {\mathcal {V}} ^ {2} + \frac {\alpha}{2} \| u \| _ {\mathcal {U}} ^ {2} \right\}.
$$

## Interpretation

Data fidelity term: $\begin{array} { r } { \frac 1 2 \lVert \mathsf { K u } - \mathsf { f } ^ { \delta } \rVert _ { \mathcal { V } } ^ { 2 } } \end{array}$ ensures the solution fits the observed data. 

## Tikhonov as an Optimisation Problem

## Variational Formulation

The Tikhonov regularised solution $\ R _ { \alpha } ( { \sf f ^ { \delta } } )$ is also the unique minimiser of 

$$
R _ {\alpha} (f ^ {\delta}) = \underset {u \in \mathcal {U}} {\arg \min} \left\{\frac {1}{2} \| K u - f ^ {\delta} \| _ {\mathcal {V}} ^ {2} + \frac {\alpha}{2} \| u \| _ {\mathcal {U}} ^ {2} \right\}.
$$

## Interpretation

Data fidelity term: $\begin{array} { r } { \frac 1 2 \lVert \mathsf { K u } - \mathsf { f } ^ { \delta } \rVert _ { \mathcal { V } } ^ { 2 } } \end{array}$ ensures the solution fits the observed data. 

Regularisation term: $\frac { \alpha } { 2 } \lVert \mathbf { u } \rVert _ { \mathcal { U } } ^ { 2 }$ penalises large solution norms. 

## Tikhonov as an Optimisation Problem

## Variational Formulation

The Tikhonov regularised solution $\ R _ { \alpha } ( { \sf f ^ { \delta } } )$ is also the unique minimiser of 

$$
R _ {\alpha} (f ^ {\delta}) = \underset {u \in \mathcal {U}} {\arg \min} \left\{\frac {1}{2} \| K u - f ^ {\delta} \| _ {\mathcal {V}} ^ {2} + \frac {\alpha}{2} \| u \| _ {\mathcal {U}} ^ {2} \right\}.
$$

## Interpretation

Data fidelity term: $\begin{array} { r } { \frac 1 2 \lVert \mathsf { K u } - \mathsf { f } ^ { \delta } \rVert _ { \mathcal { V } } ^ { 2 } } \end{array}$ ensures the solution fits the observed data. 

Regularisation term: $\frac { \alpha } { 2 } \lVert \mathbf { u } \rVert _ { \mathcal { U } } ^ { 2 }$ penalises large solution norms. 

Balance: Parameter $\alpha > 0$ controls trade-off between data fidelity and regularity. 

## Tikhonov as an Optimisation Problem

## Variational Formulation

The Tikhonov regularised solution $\ R _ { \alpha } ( { \sf f ^ { \delta } } )$ is also the unique minimiser of 

$$
R _ {\alpha} (f ^ {\delta}) = \underset {u \in \mathcal {U}} {\arg \min} \left\{\frac {1}{2} \| K u - f ^ {\delta} \| _ {\mathcal {V}} ^ {2} + \frac {\alpha}{2} \| u \| _ {\mathcal {U}} ^ {2} \right\}.
$$

## Interpretation

Data fidelity term: $\begin{array} { r } { \frac 1 2 \lVert \mathsf { K u } - \mathsf { f } ^ { \delta } \rVert _ { \mathcal { V } } ^ { 2 } } \end{array}$ ensures the solution fits the observed data. 

Regularisation term: $\frac { \alpha } { 2 } \lVert \mathbf { u } \rVert _ { \mathcal { U } } ^ { 2 }$ penalises large solution norms. 

Balance: Parameter $\alpha > 0$ controls trade-off between data fidelity and regularity. 

This variational approach is equivalent to both expression $( \mathsf { K } ^ { * } \mathsf { K } + \alpha \mathrm { I } ) ^ { - 1 } \mathsf { K } ^ { * } \mathsf { f } ^ { \delta }$ and spectral form shown on previous slide. 

## Example: Neural Network

## Pre-trained Neural Network

Consider a pre-trained feed-forward neural network $\mathcal { N } _ { \theta } : \mathcal { V } \to \mathcal { U } ,$ i.e. 

$$
R _ {\theta} \left(f ^ {\delta}\right) := \mathcal {N} _ {\theta} \left(f ^ {\delta}\right) = W _ {L} \sigma_ {L - 1} \left(\dots W _ {2} \sigma_ {1} \left(W _ {1} f ^ {\delta} + b _ {1}\right) + b _ {2} \dots\right) + b _ {L}
$$

The parameters $\boldsymbol { \Theta } = \{ ( W _ { \mathrm { j } } , \boldsymbol { \mathrm { b } } _ { \mathrm { j } } ) _ { \mathrm { j } = 1 } ^ { \mathrm { L } } \}$ are fixed. $\mathsf { R } _ { \Theta }$ is single-valued. 

## Example: Neural Network

## Pre-trained Neural Network

Consider a pre-trained feed-forward neural network $\mathcal { N } _ { \theta } : \mathcal { V } \to \mathcal { U } ,$ i.e. 

$$
R _ {\theta} (f ^ {\delta}) := \mathcal {N} _ {\theta} (f ^ {\delta}) = W _ {L} \sigma_ {L - 1} (\dots W _ {2} \sigma_ {1} (W _ {1} f ^ {\delta} + b _ {1}) + b _ {2} \dots) + b _ {L}
$$

The parameters $\boldsymbol { \Theta } = \{ ( W _ { \mathrm { j } } , \boldsymbol { \mathrm { b } } _ { \mathrm { j } } ) _ { \mathrm { j } = 1 } ^ { \mathrm { L } } \}$ are fixed. $\mathsf { R } _ { \Theta }$ is single-valued. 

## Verification as a Regularisation Operator

The parameter α in the definition corresponds to the fixed set of weights θ. 

Each layer (affine transformation $W _ { \mathrm { j } } ( \cdot ) + \boldsymbol { \mathrm { b } } _ { \mathrm { j } }$ and activation $\sigma _ { \mathrm { j } } )$ is typically continuous. Standard activation functions (ReLU, sigmoid, tanh, etc.) are continuous. 

## Example: Neural Network

## Pre-trained Neural Network

Consider a pre-trained feed-forward neural network $\mathcal { N } _ { \theta } : \mathcal { V } \to \mathcal { U } ,$ i.e. 

$$
R _ {\theta} (f ^ {\delta}) := \mathcal {N} _ {\theta} (f ^ {\delta}) = W _ {L} \sigma_ {L - 1} (\dots W _ {2} \sigma_ {1} (W _ {1} f ^ {\delta} + b _ {1}) + b _ {2} \dots) + b _ {L}
$$

The parameters $\boldsymbol { \Theta } = \{ ( W _ { \mathrm { j } } , \boldsymbol { \mathrm { b } } _ { \mathrm { j } } ) _ { \mathrm { j } = 1 } ^ { \mathrm { L } } \}$ are fixed. $\mathsf { R } _ { \Theta }$ is single-valued. 

## Verification as a Regularisation Operator

The parameter α in the definition corresponds to the fixed set of weights θ. 

Each layer (affine transformation $W _ { \mathrm { j } } ( \cdot ) + \boldsymbol { \mathrm { b } } _ { \mathrm { j } }$ and activation $\sigma _ { \mathrm { j } } )$ is typically continuous. Standard activation functions (ReLU, sigmoid, tanh, etc.) are continuous. 

A finite composition of continuous functions is continuous. 

## Pre-trained Neural Network

Consider a pre-trained feed-forward neural network $\mathcal { N } _ { \theta } : \mathcal { V } \to \mathcal { U } ,$ i.e. 

$$
R _ {\theta} (f ^ {\delta}) := \mathcal {N} _ {\theta} (f ^ {\delta}) = W _ {L} \sigma_ {L - 1} (\dots W _ {2} \sigma_ {1} (W _ {1} f ^ {\delta} + b _ {1}) + b _ {2} \dots) + b _ {L}
$$

The parameters $\boldsymbol { \Theta } = \{ ( W _ { \mathrm { j } } , \boldsymbol { \mathrm { b } } _ { \mathrm { j } } ) _ { \mathrm { j } = 1 } ^ { \mathrm { L } } \}$ are fixed. $\mathsf { R } _ { \Theta }$ is single-valued. 

## Verification as a Regularisation Operator

The parameter α in the definition corresponds to the fixed set of weights θ. 

Each layer (affine transformation $W _ { \mathrm { j } } ( \cdot ) + \boldsymbol { \mathrm { b } } _ { \mathrm { j } }$ and activation $\sigma _ { \mathrm { j } } )$ is typically continuous. Standard activation functions (ReLU, sigmoid, tanh, etc.) are continuous. 

A finite composition of continuous functions is continuous. 

Thus, the neural network $\operatorname { \mathcal { N } } _ { \theta }$ is a continuous function from V to U. 

## Pre-trained Neural Network

Consider a pre-trained feed-forward neural network $\mathcal { N } _ { \theta } : \mathcal { V } \to \mathcal { U } ,$ i.e. 

$$
R _ {\theta} (f ^ {\delta}) := \mathcal {N} _ {\theta} (f ^ {\delta}) = W _ {L} \sigma_ {L - 1} (\dots W _ {2} \sigma_ {1} (W _ {1} f ^ {\delta} + b _ {1}) + b _ {2} \dots) + b _ {L}
$$

The parameters $\boldsymbol { \Theta } = \{ ( W _ { \mathrm { j } } , \boldsymbol { \mathrm { b } } _ { \mathrm { j } } ) _ { \mathrm { j } = 1 } ^ { \mathrm { L } } \}$ are fixed. $\mathsf { R } _ { \Theta }$ is single-valued. 

## Verification as a Regularisation Operator

The parameter α in the definition corresponds to the fixed set of weights θ. 

A finite composition of continuous functions is continuous. 

Thus, the neural network $\operatorname { \mathcal { N } } _ { \theta }$ is a continuous function from V to U (assuming standard topologies, e.g., if $\mathcal { V } = \mathbb { R } ^ { \mathrm { m } } , \mathcal { U } = \mathbb { R } ^ { \mathrm { n } } )$ 

## Pre-trained Neural Network

Consider a pre-trained feed-forward neural network $\mathcal { N } _ { \theta } : \mathcal { V } \to \mathcal { U } ,$ i.e. 

$$
R _ {\theta} (f ^ {\delta}) := \mathcal {N} _ {\theta} (f ^ {\delta}) = W _ {L} \sigma_ {L - 1} (\dots W _ {2} \sigma_ {1} (W _ {1} f ^ {\delta} + b _ {1}) + b _ {2} \dots) + b _ {L}
$$

The parameters $\boldsymbol { \Theta } = \{ ( W _ { \mathrm { j } } , \boldsymbol { \mathrm { b } } _ { \mathrm { j } } ) _ { \mathrm { j } = 1 } ^ { \mathrm { L } } \}$ are fixed. $\mathsf { R } _ { \Theta }$ is single-valued. 

## Verification as a Regularisation Operator

The parameter α in the definition corresponds to the fixed set of weights θ. 

A finite composition of continuous functions is continuous. 

Thus, the neural network $\operatorname { \mathcal { N } } _ { \theta }$ is a continuous function from V to U (assuming standard topologies, e.g., if $\mathcal { V } = \mathbb { R } ^ { \mathrm { m } } , \mathcal { U } = \mathbb { R } ^ { \mathrm { n } } )$ 

Therefore, $\mathsf { R } _ { \theta } ( \mathsf { f } ^ { \delta } ) = \mathcal { N } _ { \theta } ( \mathsf { f } ^ { \delta } )$ (for fixed pre-trained θ) is a regularisation operator. 

## Beyond Stability: The Need for Convergence

We’ve established that regularisation operators $\mathtt { R } _ { \alpha }$ provide stable processing of data for a fixed parameter α. A regularisation method $\left( \mathsf { R } _ { \alpha } , \alpha ( \delta , \mathsf { f } ^ { \delta } ) \right)$ ) then uses a rule to select α. 

## Beyond Stability: The Need for Convergence

We’ve established that regularisation operators $\mathtt { R } _ { \alpha }$ provide stable processing of data for a fixed parameter α. A regularisation method $\left( \mathsf { R } _ { \alpha } , \alpha ( \delta , \mathsf { f } ^ { \delta } ) \right)$ ) then uses a rule to select α. 

## Why This Isn’t the Whole Story

Stability for each fixed α is crucial, but it doesn’t guarantee that our method produces solutions that are close to the true underlying solution of Ku = f. 

## Beyond Stability: The Need for Convergence

We’ve established that regularisation operators $\mathtt { R } _ { \alpha }$ provide stable processing of data for a fixed parameter α. A regularisation method $\left( \mathsf { R } _ { \alpha } , \alpha ( \delta , \mathsf { f } ^ { \delta } ) \right)$ ) then uses a rule to select α. 

## Why This Isn’t the Whole Story

Stability for each fixed α is crucial, but it doesn’t guarantee that our method produces solutions that are close to the true underlying solution of Ku = f. 

As the noise δ in our data $\mathsf { f } ^ { \delta }$ diminishes, we expect our regularisation method to yield solutions that improve and approach this true solution. 

## Beyond Stability: The Need for Convergence

We’ve established that regularisation operators $\mathtt { R } _ { \alpha }$ provide stable processing of data for a fixed parameter α. A regularisation method $\left( \mathsf { R } _ { \alpha } , \alpha ( \delta , \mathsf { f } ^ { \delta } ) \right)$ ) then uses a rule to select α. 

## Why This Isn’t the Whole Story

Stability for each fixed α is crucial, but it doesn’t guarantee that our method produces solutions that are close to the true underlying solution of Ku = f. 

As the noise δ in our data $\mathsf { f } ^ { \delta }$ diminishes, we expect our regularisation method to yield solutions that improve and approach this true solution. 

This requires that the parameter choice $\alpha ( \delta , { \sf f } ^ { \delta } )$ adapts appropriately (e.g., $\alpha  0 )$ , and that $\mathsf { R } _ { \alpha ( \delta , \mathsf { f } ^ { \delta } ) }$ indeed approximates the correct inverse operation in this limit. 

## Beyond Stability: The Need for Convergence

## Why This Isn’t the Whole Story

Stability for each fixed α is crucial, but it doesn’t guarantee that our method produces solutions that are close to the true underlying solution of ${ \mathrm { k u } } = { \mathrm { f } }$ 

As the noise δ in our data $\mathsf { f } ^ { \delta }$ diminishes, we expect our regularisation method to yield solutions that improve and approach this true solution. 

This requires that the parameter choice $\alpha ( \delta , { \sf f } ^ { \delta } )$ adapts appropriately (e.g., α → 0), and that $\mathsf { R } _ { \alpha ( \delta , \mathsf { f } ^ { \delta } ) }$ indeed approximates the correct inverse operation in this limit. 

## The Next Step: Defining the Target

Before we can formally discuss convergence of a regularisation method to the true solution, we need to define what this true solution is. This leads us to consider concepts like best approximate solutions and selection operators. 

## Selecting Solutions

## Best Approximate Solutions and Selection Operators

## Best Approximate Solution

Given an error measure F : $\mathcal { V } \times \mathcal { V }  \mathbb { R } _ { + } \cup \{ + \infty \}$ , we call $\hat { \mathbf { u } } \in \mathcal { U }$ a best approximate solution of ${ \mathrm { k u } } = { \mathrm { f } }$ with respect to F if 

$$
F (K \hat {u}, f) \leqslant F (K u, f) \quad \text { for   all } u \in \mathcal {U}
$$

## Best Approximate Solutions and Selection Operators

## Best Approximate Solution

Given an error measure F : $\mathcal { V } \times \mathcal { V }  \mathbb { R } _ { + } \cup \{ + \infty \}$ , we call $\hat { \mathbf { u } } \in \mathcal { U }$ a best approximate solution of ${ \mathrm { k u } } = { \mathrm { f } }$ with respect to F if 

$$
F (K \hat {u}, f) \leqslant F (K u, f) \quad \text { for   all } u \in \mathcal {U}
$$

## Selection Operator

A multivalued operator $\mathcal { S } : \mathcal { R } ( \mathsf { K } ) \Longrightarrow \mathcal { U }$ is called a selection operator if $\mathcal { S } ( \mathsf { K u } ) \subset \{ \mathfrak { u } \} + \mathcal { N } ( \mathsf { K } )$ for all $\mathrm { \mathfrak { u } } \in \mathrm { \mathfrak { U } }$ 

## Best Approximate Solutions and Selection Operators

## Best Approximate Solution

Given an error measure F : $\mathcal { V } \times \mathcal { V }  \mathbb { R } _ { + } \cup \{ + \infty \}$ , we call $\hat { \mathbf { u } } \in \mathcal { U }$ a best approximate solution of ${ \mathrm { k u } } = { \mathrm { f } }$ with respect to F if 

$$
F (K \hat {u}, f) \leqslant F (K u, f) \quad \text { for   all } u \in \mathcal {U}
$$

## Selection Operator

A multivalued operator $\mathcal { S } : \mathcal { R } ( \mathsf { K } ) \Longrightarrow \mathcal { U }$ is called a selection operator if $\mathcal { S } ( \mathsf { K u } ) \subset \{ \mathfrak { u } \} + \mathcal { N } ( \mathsf { K } )$ for all $\mathrm { \mathfrak { u } } \in \mathrm { \mathfrak { U } }$ . A best approximate solution ˆu is called prior selected solution if $\hat { \mathbf { u } } \in \mathcal { S } ( \mathsf { K } \hat { \mathbf { u } } )$ 

Often, $\mathcal { S } ( { \sf f ^ { \prime } } )$ selects solutions from the set of best approximate solutions for data $\mathsf { f } ^ { \prime }$ by minimising a secondary (regularisation) functional. 

## Examples of Selection Operators

Let $\mathsf { K } : \mathsf { L } ^ { 2 } ( \Omega ) \to \mathsf { L } ^ { 2 } ( \Sigma )$ or similar Hilbert spaces. 

Selection via Exact Fit (Minimum Norm) 

$$
\text { If } F (K u, f) = \chi_ {= 0} (f - K u) = \left\{ \begin{array}{l l} 0 & K u = f \\ \infty & \text { else } \end{array} \right..
$$

## Examples of Selection Operators

Let $\mathsf { K } : \mathsf { L } ^ { 2 } ( \Omega ) \to \mathsf { L } ^ { 2 } ( \Sigma )$ or similar Hilbert spaces. 

## Selection via Exact Fit (Minimum Norm)

$\begin{array} { r l } & { \mathsf { I f } \mathsf { F } ( \mathsf { K u , f } ) = \mathsf { \chi } _ { = 0 } ( \mathsf { f } - \mathsf { K u } ) = \left\{ 0 \quad \mathsf { K u = f } \right. } \\ & { \mathsf { S o l u t i o n s } \left\{ \mathsf { u } \in \mathsf { L } ^ { 2 } ( \Omega ) \mid \mathsf { K u = f } \right\} . } \end{array}$ The best approximate solutions are exact 

## Examples of Selection Operators

Let $\mathsf { K } : \mathsf { L } ^ { 2 } ( \Omega ) \to \mathsf { L } ^ { 2 } ( \Sigma )$ or similar Hilbert spaces. 

## Selection via Exact Fit (Minimum Norm)

$1 \mathsf { f } \mathsf { F } ( \mathsf { K } \mathrm { u } , \mathsf { f } ) = \chi _ { = 0 } ( \mathsf { f } - \mathsf { K } \mathrm { u } ) = \left\{ 0 \begin{array} { l l } { \mathsf { K } \mathrm { u } = \mathsf { f } } \\ { \infty } & { \mathsf { e } \mathsf { l } \mathsf { s } \mathsf { e } } \end{array} \right.$ The best approximate solutions are exact solutions $\{ \mathfrak { u } \in \mathrm { L } ^ { 2 } ( \Omega ) \ | \ \mathsf { K } \mathfrak { u } = \mathfrak { f } \}$ . A selection operator can be defined as 

$$
\mathcal {S} (f) = \underset {u \in L ^ {2} (\Omega)} {\arg \min} \left\{\| u \| _ {L ^ {2} (\Omega)} \mid K u = f \right\}
$$

This yields 

$$
\mathcal {S} (f) = \left\{ \begin{array}{l l} \{\mathrm{K} ^ {\dagger} f \} & \text { if } f \in \mathcal {R} (\mathrm{K}) \\ \emptyset & \text { if } f \notin \mathcal {R} (\mathrm{K}) \end{array} \right.,
$$

where $\mathsf { K } ^ { \dagger }$ is the Moore-Penrose pseudo-inverse of K. 

## Examples of Selection Operators (continued)

## Selection via Least Squares (Minimum Norm)

Let $\begin{array} { r } { \mathsf { F } ( \mathsf { K } \mathrm { u } , \mathsf { f } ) = \frac { 1 } { 2 } \left\| \mathsf { K } \mathrm { u } - \mathsf { f } \right\| _ { \mathsf { L } ^ { 2 } ( \Sigma ) } ^ { 2 } } \end{array}$ . 

## Examples of Selection Operators (continued)

## Selection via Least Squares (Minimum Norm)

Let $\begin{array} { r } { \mathsf { F } ( \mathsf { K } \mathrm { u } , \mathsf { f } ) = \frac { 1 } { 2 } \left\| \mathsf { K } \mathrm { u } - \mathsf { f } \right\| _ { \mathsf { L } ^ { 2 } ( \Sigma ) } ^ { 2 } } \end{array}$ . Best approximate solutions are least-squares solutions, satisfying ${ \sf K } ^ { * } { \sf K } { \sf u } = { \sf K } ^ { * } { \sf f }$ 

## Examples of Selection Operators (continued)

## Selection via Least Squares (Minimum Norm)

Let $\begin{array} { r } { \mathsf { F } ( \mathsf { K } \mathrm { u } , \mathsf { f } ) = \frac { 1 } { 2 } \left\| \mathsf { K } \mathrm { u } - \mathsf { f } \right\| _ { \mathsf { L } ^ { 2 } ( \Sigma ) } ^ { 2 } } \end{array}$ . Best approximate solutions are least-squares solutions, satisfying ${ \sf K } ^ { * } { \sf K } { \sf u } = { \sf K } ^ { * } { \sf f }$ . A selection operator can be defiend as 

$$
\mathcal {S} (f) = \underset {u \in L ^ {2} (\Omega)} {\arg \min} \left\{\| u \| _ {L ^ {2} (\Omega)} \mid K ^ {*} K u = K ^ {*} f \right\}
$$

## Examples of Selection Operators (continued)

## Selection via Least Squares (Minimum Norm)

Let $\begin{array} { r } { \mathsf { F } ( \mathsf { K } \mathrm { u } , \mathsf { f } ) = \frac { 1 } { 2 } \left\| \mathsf { K } \mathrm { u } - \mathsf { f } \right\| _ { \mathsf { L } ^ { 2 } ( \Sigma ) } ^ { 2 } } \end{array}$ . Best approximate solutions are least-squares solutions, satisfying ${ \sf K } ^ { * } { \sf K } { \sf u } = { \sf K } ^ { * } { \sf f }$ . A selection operator can be defiend as 

$$
\mathcal {S} (f) = \underset {u \in L ^ {2} (\Omega)} {\arg \min} \left\{\| u \| _ {L ^ {2} (\Omega)} \mid K ^ {*} K u = K ^ {*} f \right\}
$$

This yields 

$$
\mathcal {S} (f) = \left\{ \begin{array}{l l} \{\mathsf {K} ^ {\dagger} f \} & \text {   if   } f \in \mathcal {D} (\mathsf {K} ^ {\dagger}) = \mathcal {R} (\mathsf {K}) \oplus \mathcal {R} (\mathsf {K}) ^ {\top} \\ \emptyset & \text {   if   } f \in \overline {{\mathcal {R} (\mathsf {K})}} \setminus \mathcal {R} (\mathsf {K}) \end{array} \right.
$$

where $\mathsf { K } ^ { \dagger }$ is the Moore-Penrose pseudo-inverse of $\mathsf { K } .$ . 

## How to Compute Selection Operators

Let K : U → V for Banach spaces U and V. 

## Selection via Exact Fit (J-minimising solution)

Suppose we choose $\mathsf { F } ( \mathsf { K u } , \mathsf { f } ) = \mathsf { X } _ { = 0 } ( \mathsf { f } - \mathsf { K u } )$ as earlier. 

## How to Compute Selection Operators

Let K : U → V for Banach spaces U and V. 

## Selection via Exact Fit (J-minimising solution)

Suppose we choose $\mathsf { F } ( \mathsf { K u } , \mathsf { f } ) = \mathsf { X } _ { = 0 } ( \mathsf { f } - \mathsf { K u } )$ as earlier. The best approximate solutions are exact solutions $\{ \mathfrak { u } \in \mathfrak { U } \mid \mathsf { K } \mathfrak { u } = \mathfrak { f } \}$ 

## How to Compute Selection Operators

Let K : U → V for Banach spaces U and V. 

## Selection via Exact Fit (J-minimising solution)

Suppose we choose $\mathsf { F } ( \mathsf { K u } , \mathsf { f } ) = \mathsf { X } _ { = 0 } ( \mathsf { f } - \mathsf { K u } )$ as earlier. The best approximate solutions are exact solutions $\{ \mathfrak { u } \in \mathfrak { U } \mid \mathsf { K } \mathfrak { u } = \mathfrak { f } \}$ . A selection operator can be defined as 

$$
\mathcal {S} (f) = \underset {u \in \mathcal {U}} {\arg \min} \{J (u) \mid K u = f \}
$$

for a proper, lower semi-continuous and convex functional J : $\mathcal { U } \to \mathbb { R } _ { + } \cup \{ + \infty \}$ 

## How to Compute Selection Operators

Let K : U → V for Banach spaces U and V. 

## Selection via Exact Fit (J-minimising solution)

Suppose we choose $\mathsf { F } ( \mathsf { K u } , \mathsf { f } ) = \mathsf { X } _ { = 0 } ( \mathsf { f } - \mathsf { K u } )$ as earlier. The best approximate solutions are exact solutions $\{ \mathfrak { u } \in \mathfrak { U } \mid \mathsf { K } \mathfrak { u } = \mathfrak { f } \}$ . A selection operator can be defined as 

$$
\mathcal {S} (f) = \underset {u \in \mathcal {U}} {\arg \min} \{J (u) \mid K u = f \}
$$

for a proper, lower semi-continuous and convex functional J $: \mathbb { U } \to \mathbb { R } _ { + } \cup \{ + \infty \}$ . In primal-dual form, the corresponding saddle point problem reads 

$$
\inf _ {u \in \mathcal {U}} \sup _ {v \in \mathcal {V}} J (u) + \langle v, f - K u \rangle .
$$

## How to Compute Selection Operators (continued)

Let K : U → V for Banach spaces U and V. 

## Selection via Exact Fit (J-minimising solution)

In primal-dual form, the corresponding saddle point problem reads 

$$
\inf _ {u \in \mathcal {U}} \sup _ {v \in \mathcal {V}} J (u) + \langle v, f - K u \rangle .
$$

The first-order optimality conditions are 

$$
K ^ {*} v ^ {\dagger} \in \partial J (u ^ {\dagger})
$$

source condition 

$$
\mathsf {K u} ^ {\dagger} = \mathsf {f}
$$

consistency condition 

## How to Compute Selection Operators (continued)

Let K : U → V for Banach spaces U and V. 

## Selection via Exact Fit (J-minimising solution)

In primal-dual form, the corresponding saddle point problem reads 

$$
\inf _ {u \in \mathcal {U}} \sup _ {v \in \mathcal {V}} J (u) + \langle v, f - K u \rangle .
$$

The first-order optimality conditions are 

$$
K ^ {*} v ^ {\dagger} \in \partial J (u ^ {\dagger})
$$

source condition 

$$
\mathsf {K u} ^ {\dagger} = \mathsf {f}
$$

consistency condition 

How do we compute $\mathfrak { u } ^ { \dagger }$ and $\nu ^ { \dagger }$ numerically? 

## How to Compute Selection Operators (continued)

Let K : U → V for Banach spaces U and V. 

## Selection via Exact Fit (J-minimising solution)

In primal-dual form, the corresponding saddle point problem reads 

$$
\inf _ {u \in \mathcal {U}} \sup _ {v \in \mathcal {V}} J (u) + \langle v, f - K u \rangle .
$$

One option: Primal-Dual Hybrid Gradient / Chambolle-Pock algorithm [18, 13, 6, 3, 4], i.e. 

$$
u ^ {k + 1} = \operatorname{prox} _ {\tau J} \left(u ^ {k} + \tau K ^ {*} v ^ {k}\right),
$$

$$
v ^ {k + 1} = v ^ {k} - \sigma \left(K (2 u ^ {k + 1} - u ^ {k}) - f\right),
$$

for parameters τ, $\sigma > 0$ with τσ $< 1 / \Vert \mathsf { K } \Vert ^ { 2 }$ 

## How to Compute Selection Operators (continued)

Let K : U → V for Banach spaces U and V. 

## Selection via Exact Fit (J-minimising solution)

In primal-dual form, the corresponding saddle point problem reads 

$$
\inf _ {u \in \mathcal {U}} \sup _ {v \in \mathcal {V}} J (u) + \langle v, f - K u \rangle .
$$

One option: Primal-Dual Hybrid Gradient / Chambolle-Pock algorithm [18, 13, 6, 3, 4], i.e. 

$$
u ^ {k + 1} = \operatorname{prox} _ {\tau J} \left(u ^ {k} + \tau K ^ {*} v ^ {k}\right),
$$

$$
v ^ {k + 1} = v ^ {k} - \sigma \left(K (2 u ^ {k + 1} - u ^ {k}) - f\right),
$$

with τσ $< 1 / \vert \vert \mathsf { K } \vert \vert ^ { 2 }$ . You will compute J-minimising solutions in your first lab work! 

## How to Compute Selection Operators (continued)

Example: Let K = I be the identity operator and J = TV the (isotropic) total variation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/1c18789edb0da6be0bd29555fa7eb1d96fd6c4681f9f160df890bfebebc60d87.jpg)



$\mathsf { f } = \mathrm { u } ^ { \dagger }$ (ground truth)


## How to Compute Selection Operators (continued)

Example: Let K = I be the identity operator and J = TV the (isotropic) total variation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/8ec1765e73e1a788cce9c6e1361dcfa62c4d24300de3e4025a22145102f387c9.jpg)



f = u<sup>†</sup> (ground truth)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/1a5da82ccb88c4042eda0e93e4f1df4454c7ead6af4b606744c3f7acc7b6a938.jpg)



S(u<sup>†</sup>)


## How to Compute Selection Operators (continued)

Example: Let K = I be the identity operator and J = TV the (isotropic) total variation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/e5cdd48dfeda21473da6861d3ec5924fa655cb99f38652dae9f249235fcd0c14.jpg)



$\mathsf { f } = \mathrm { u } ^ { \dagger }$ (ground truth)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/4aa7b39b1b3a6d853364226bbb10f352ecae0926baf2abc4abdcec0258ade639.jpg)



S(u<sup>†</sup>)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/b6358245129f15735af6931991b72ecd0b943f4fec4e6a6abcb0696d730409ec.jpg)



$\nu ^ { \dagger }$ (source condition element)


## How to Compute Selection Operators (continued)

Example: Let ${ \sf K } = \cdot * { \sf h }$ be a motion blur operator and J = TV the (isotropic) total variation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/4b9d8f0db82b9f779cc16db3622aa58d6bc263c4eb627afb1e42846950b07af4.jpg)



$\mathsf { f } = \mathsf { u } ^ { \dagger } * \mathsf { h }$ (blurred image)


## How to Compute Selection Operators (continued)

Example: Let ${ \sf K } = \cdot * { \sf h }$ be a motion blur operator and J = TV the (isotropic) total variation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/3705d6ed9c5aa62a860b2fdef56dcfbb4f2fd63f14c108e357f85681fb48506c.jpg)



$\mathsf { f } = \mathsf { u } ^ { \dagger } * \mathsf { h }$ (blurred image)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/d991941430f33495955bec287a85c17a8ced3f96dd9690d765f4a28b9cfe5b88.jpg)



S(f)


## How to Compute Selection Operators (continued)

Example: Let ${ \sf K } = \cdot * { \sf h }$ be a motion blur operator and $\boldsymbol { \mathrm { J } } = \mathsf { T V }$ the (isotropic) total variation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/a98116551e6c0861499286f8c47e29e5d1972546f7bc5a2d19c9018e25aa39a5.jpg)



$\mathsf { f } = \mathsf { u } ^ { \dagger } * \mathsf { h }$ (blurred image)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/309dbf98f75950971cbce0f3b2dbbf43cbd02ae4649bfdd956e3809a805fdc1a.jpg)



S(f)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/a4ddddb8d45702d27165bc3f0e40bed3a7dc78d65d402162dc5dcba5cfe00590.jpg)



$\nu ^ { \dagger }$ (source condition element)


## Convergent Regularisation Methods

## Convergent Regularisation Method: Definition

Recall: A regularisation method consists of regularisation operators $\mathtt { R } _ { \alpha }$ and a parameter choice strategy $\alpha _ { \sf c h o i c e } ( \delta , { \sf f } ^ { \delta } )$ . 

## Convergent Regularisation Method: Definition

Recall: A regularisation method consists of regularisation operators $\mathtt { R } _ { \alpha }$ and a parameter choice strategy $\alpha _ { \sf c h o i c e } ( \delta , { \sf f } ^ { \delta } )$ 

## Convergent Regularisation Method [1]

A regularisation method is called convergent if for any ”exact” data f (for which a set of desired solutions S(f) ⊂ U is well-defined), any sequence of noise levels $\delta _ { \mathfrak { n } } \to 0$ , data $\mathsf { f } ^ { \delta _ { \mathrm { n } } }$ satisfying $\mathsf { F } ( \mathsf { f } , \mathsf { f } ^ { \delta _ { \mathsf { n } } } ) \leqslant \delta _ { \mathsf { n } }$ and parameter choice strategy $\alpha _ { \mathfrak { n } } = \alpha _ { \mathtt { c h o i c e } } ( \delta _ { \mathfrak { n } } , \mathsf { f } ^ { \delta _ { \mathfrak { n } } } )$ we have 

$$
\emptyset \neq \left\{x \in \mathcal {U}, \left| \operatorname * {l i m s u p} _ {n \to \infty} d _ {\mathcal {U}} (x, R _ {\alpha_ {n}} (f ^ {\delta_ {k}})) = 0 \right. \right\} \subset \mathcal {S} (f)
$$

This means the Kuratowski limit inferior of $\mathsf { R } _ { \alpha _ { \mathrm { n } } } \bigl ( \mathsf { f } ^ { \delta _ { \mathrm { n } } } \bigr )$ must be non-empty and contained in the set of desired solutions S(f). 

## Convergent Regularisation Method: Definition

Recall: A regularisation method consists of regularisation operators $\mathtt { R } _ { \alpha }$ and a parameter choice strategy $\alpha _ { \sf c h o i c e } ( \delta , { \sf f } ^ { \delta } )$ . If $\mathtt { R } _ { \alpha }$ is single-valued, the previous definition simplifies to: 

## Convergent Regularisation Method (Single-Valued Operators)

A regularisation method is called convergent if for any ”exact” data f (for which a set of desired solutions S(f) ⊂ U is well-defined), any sequence of noise levels $\delta _ { \mathfrak { n } } \to 0$ , data $\mathsf { f } ^ { \delta _ { \mathrm { n } } }$ satisfying $\mathsf { F } ( \mathsf { f } , \mathsf { f } ^ { \delta _ { \mathsf { n } } } ) \leqslant \delta _ { \mathsf { n } }$ and parameter choice strategy $\alpha _ { \mathfrak { n } } = \alpha _ { \mathtt { c h o i c e } } ( \delta _ { \mathfrak { n } } , \mathsf { f } ^ { \delta _ { \mathfrak { n } } } )$ we have 

$$
\lim _ {n \to \infty} R _ {\alpha_ {n}} (f ^ {\delta_ {n}}) = u ^ {*} \quad \text { where } u ^ {*} \in \mathcal {S} (f).
$$

This means that the sequence of solutions $\mathsf { R } _ { \alpha _ { \mathrm { n } } } \big ( \mathsf { f } ^ { \delta _ { \mathrm { n } } } \big )$ must converge to some solution $\mu ^ { \ast }$ and $\mathfrak { u } ^ { * } \in \mathcal { S } ( \mathfrak { f } )$ 

## Example: Tikhonov Regularisation

Let us revisit Tikhonov regularisation. Recall that the filter functions ${ \mathfrak { g } } _ { \alpha } ( \sigma )$ are defined as 

$$
g _ {\alpha} (\sigma) = \frac {\sigma}{\sigma^ {2} + \alpha}.
$$

## Example: Tikhonov Regularisation

Let us revisit Tikhonov regularisation. Recall that the filter functions ${ \mathfrak { g } } _ { \alpha } ( \sigma )$ are defined as 

$$
g _ {\alpha} (\sigma) = \frac {\sigma}{\sigma^ {2} + \alpha}.
$$

The Tikhonov regularisation operator $\mathbb { R } _ { \alpha } : \mathcal { V }  \mathcal { U }$ is then defined as 

$$
R _ {\alpha} f = \sum_ {j = 1} ^ {\infty} \frac {\sigma_ {j}}{\sigma_ {j} ^ {2} + \alpha} \left\langle f, v _ {j} \right\rangle_ {\mathcal {V}} u _ {j}.
$$

## Example: Tikhonov Regularisation

Let us revisit Tikhonov regularisation. Recall that the filter functions ${ \mathfrak { g } } _ { \alpha } ( \sigma )$ are defined as 

$$
g _ {\alpha} (\sigma) = \frac {\sigma}{\sigma^ {2} + \alpha}.
$$

The Tikhonov regularisation operator $\mathbb { R } _ { \alpha } : \mathcal { V }  \mathcal { U }$ is then defined as 

$$
R _ {\alpha} f = \sum_ {j = 1} ^ {\infty} \frac {\sigma_ {j}}{\sigma_ {j} ^ {2} + \alpha} \left\langle f, v _ {j} \right\rangle_ {\mathcal {V}} u _ {j}.
$$

## Key Property for Fixed $\mathfrak { Q } \geqslant \mathfrak { Q }$

For any fixed $\alpha > 0$ , the filter function ${ \mathfrak { g } } _ { \alpha } ( \sigma )$ is bounded, i.e. 

$$
\sup _ {\sigma > 0} \left| \frac {\sigma}{\sigma^ {2} + \alpha} \right| = \frac {1}{2 \sqrt {\alpha}} <   \infty
$$

Thus, $\mathtt { R } _ { \alpha }$ is a bounded (and therefore continuous) linear operator for fixed $\alpha > 0$ 

## Tikhonov as a Regularisation Operator

We recall that Tikhonov regularisation fits the definition of a regularisation operator. 

## Recall: Regularisation Operator (Single-Valued Case)

An operator $\mathtt { R } _ { \alpha }$ : V → U is a regularisation operator if it is continuous on V for each fixed α. 

## Tikhonov as a Regularisation Operator

We recall that Tikhonov regularisation fits the definition of a regularisation operator. 

## Recall: Regularisation Operator (Single-Valued Case)

An operator $\begin{array} { r } { \mathsf { R } _ { \alpha } : \mathcal { V } \to } \end{array}$ U is a regularisation operator if it is continuous on V for each fixed α. 

## Tikhonov is a Regularisation Operator

For any fixed $\alpha > 0$ , the Tikhonov filter function $\textstyle { \mathfrak { g } } _ { \alpha } ( \sigma ) = { \frac { \sigma } { \sigma ^ { 2 } + \alpha } }$ is bounded, as $\begin{array} { r } { \mathsf { s u p } _ { \sigma > 0 } | g _ { \alpha } ( \sigma ) | = \frac { 1 } { 2 \sqrt { \alpha } } = : \mathbb { C } _ { \alpha } < \infty . } \end{array}$ 

This ensures that the operator $\begin{array} { r } { \mathsf { R } _ { \alpha } \mathsf { f } = \sum _ { \mathrm { i } = 1 } ^ { \infty } \mathsf { g } _ { \alpha } ( \sigma _ { \mathrm { j } } ) \left. \mathsf { f } , \nu _ { \mathrm { j } } \right. _ { \mathcal { V } } \boldsymbol { \mathrm { u } } _ { \mathrm { j } } } \end{array}$ is a bounded linear operator, because $\| \mathsf { R } _ { \alpha } \mathsf { f } \| _ { \mathcal { U } } \leqslant \mathsf { C } _ { \alpha } \| \mathsf { f } \| _ { \mathcal { V } }$ 

Bounded linear operators are continuous. 

Hence, for each fixed $\alpha > 0 , \mathtt { R } _ { \alpha }$ is a regularisation operator. 

## Tikhonov as a Convergent Regularisation Method

Let $\mathrm { { u } ^ { \dag } = K ^ { \dag } f }$ be the desired (minimum-norm least-squares) solution for exact data $\mathsf { f } \in \mathcal { D } ( \mathsf { K } ^ { \dagger } )$ . So $\mathcal { S } ( \mathsf { f } ) = \{ \mathsf { K } ^ { \dagger } \mathsf { f } \}$ 

## Recall: Convergent Regularisation Method (Single-Valued)

$\left( \mathsf { R } _ { \alpha } , \alpha _ { \mathsf { c h o i c e } } \right)$ is convergent if for noise $\delta _ { \mathfrak { n } } \to 0$ and data $\mathsf { f } ^ { \delta _ { \mathrm { n } } }$ (with $\mathsf { F } ( \mathsf { f } , \mathsf { f } ^ { \delta _ { \mathrm { n } } } ) \leqslant \delta _ { \mathrm { n } } )$ , setting $\alpha _ { \mathfrak { n } } = \alpha _ { \mathtt { c h o i c e } } ( \delta _ { \mathfrak { n } } , \mathsf { f } ^ { \delta _ { \mathfrak { n } } } )$ , we have $\begin{array} { r } { \operatorname* { l i m } _ { \mathfrak { n }  \infty } \mathbb { R } _ { \alpha _ { \mathfrak { n } } } ( \mathsf { f } ^ { \delta _ { \mathfrak { n } } } ) = \mathsf { K } ^ { \dagger } \mathsf { f } } \end{array}$ 

## Tikhonov as a Convergent Regularisation Method

Let $\mathrm { { u } ^ { \dag } = K ^ { \dag } f }$ be the desired (minimum-norm least-squares) solution for exact data $\mathsf { f } \in \mathcal { D } ( \mathsf { K } ^ { \dagger } )$ . So $\mathcal { S } ( \mathsf { f } ) = \{ \mathsf { K } ^ { \dagger } \mathsf { f } \}$ 

## Recall: Convergent Regularisation Method (Single-Valued)

$\left( \mathsf { R } _ { \alpha } , \alpha _ { \mathsf { c h o i c e } } \right)$ is convergent if for noise $\delta _ { \mathfrak { n } } \to 0$ and data $\mathsf { f } ^ { \delta _ { \mathrm { n } } }$ (with $\mathsf { F } ( \mathsf { f } , \mathsf { f } ^ { \delta _ { \mathrm { n } } } ) \leqslant \delta _ { \mathrm { n } } )$ , setting $\alpha _ { \mathfrak { n } } = \alpha _ { \mathtt { c h o i c e } } ( \delta _ { \mathfrak { n } } , \mathsf { f } ^ { \delta _ { \mathfrak { n } } } )$ , we have $\begin{array} { r } { \operatorname* { l i m } _ { \mathfrak { n }  \infty } \mathsf { R } _ { \alpha _ { \mathfrak { n } } } ( \mathsf { f } ^ { \delta _ { \mathfrak { n } } } ) = \mathsf { K } ^ { \dagger } \mathsf { f } . } \end{array}$ 

## Convergence of Tikhonov Regularisation

For Tikhonov regularisation, the filter $\begin{array} { r } { 9 \alpha ( \sigma _ { \mathrm { j } } ) = \frac { \sigma _ { \mathrm { j } } } { \sigma _ { \mathrm { j } } ^ { 2 } + \alpha } } \end{array}$ satisfies 

lim $\begin{array} { r } { { } | \alpha \to 0 \mathrm { g } _ { \alpha } ( \sigma _ { \mathrm { j } } ) = | \mathsf { i m } _ { \alpha \to 0 } \frac { \sigma _ { \mathrm { j } } } { \sigma _ { \mathrm { j } } ^ { 2 } + \alpha } = \frac { 1 } { \sigma _ { \mathrm { j } } } } \end{array}$ for $\sigma _ { \mathrm { j } } > 0$ This is a key condition for approximating $\mathsf { K } ^ { \dagger }$ 

## Tikhonov as a Convergent Regularisation Method

## Convergence of Tikhonov Regularisation (continued)

Consider an a-priori parameter choice $\alpha ( \delta )$ that depends on the noise level δ. The error is 

$$
\left\| R _ {\alpha (\delta)} f ^ {\delta} - K ^ {\dagger} f \right\| _ {\mathcal {U}} \leqslant \left\| R _ {\alpha (\delta)} (f ^ {\delta} - f) \right\| _ {\mathcal {U}} + \left\| R _ {\alpha (\delta)} f - K ^ {\dagger} f \right\| _ {\mathcal {U}}.
$$

## Tikhonov as a Convergent Regularisation Method

## Convergence of Tikhonov Regularisation (continued)

Consider an a-priori parameter choice $\alpha ( \delta )$ that depends on the noise level δ. The error is 

$$
\begin{array}{c} \left\| R _ {\alpha (\delta)} f ^ {\delta} - K ^ {\dagger} f \right\| _ {\mathcal {U}} \leqslant \left\| R _ {\alpha (\delta)} (f ^ {\delta} - f) \right\| _ {\mathcal {U}} + \left\| R _ {\alpha (\delta)} f - K ^ {\dagger} f \right\| _ {\mathcal {U}}. \\ \blacksquare \left\| R _ {\alpha (\delta)} (f ^ {\delta} - f) \right\| _ {\mathcal {U}} \leqslant \left\| R _ {\alpha (\delta)} \right\| _ {\mathcal {L} (\mathcal {V}, \mathcal {U})} \left\| f ^ {\delta} - f \right\| _ {\mathcal {V}} \leqslant \frac {1}{2 \sqrt {\alpha (\delta)}} \delta . \end{array}
$$

## Tikhonov as a Convergent Regularisation Method

## Convergence of Tikhonov Regularisation (continued)

Consider an a-priori parameter choice $\alpha ( \delta )$ that depends on the noise level δ. The error is 

$$
\left\| R _ {\alpha (\delta)} f ^ {\delta} - K ^ {\dagger} f \right\| _ {\mathcal {U}} \leqslant \left\| R _ {\alpha (\delta)} (f ^ {\delta} - f) \right\| _ {\mathcal {U}} + \left\| R _ {\alpha (\delta)} f - K ^ {\dagger} f \right\| _ {\mathcal {U}}.
$$

$$
\left\| R _ {\alpha (\delta)} (f ^ {\delta} - f) \right\| _ {\mathcal {U}} \leqslant \left\| R _ {\alpha (\delta)} \right\| _ {\mathcal {L} (\mathcal {V}, \mathcal {U})} \left\| f ^ {\delta} - f \right\| _ {\mathcal {V}} \leqslant \frac {1}{2 \sqrt {\alpha (\delta)}} \delta .
$$

$$
\begin{array}{l} \left\| R _ {\alpha (\delta)} f - K ^ {\dagger} f \right\| _ {\mathcal {U}} = \left\| \sum_ {j = 1} ^ {\infty} \left(\frac {\sigma_ {j}}{\sigma_ {j} ^ {2} + \alpha (\delta)} - \frac {1}{\sigma_ {j}}\right) \left\langle f, v _ {j} \right\rangle_ {\mathcal {V}} u _ {j} \right\| _ {\mathcal {U}} \to 0 \text {   as   } \alpha (\delta) \to 0 \text {   if   } \\ f \in \mathcal {D} (K ^ {\dagger}). \end{array}
$$

## Tikhonov as a Convergent Regularisation Method

## Convergence of Tikhonov Regularisation (continued)

Consider an a-priori parameter choice $\alpha ( \delta )$ that depends on the noise level δ. The error is 

$$
\left\| R _ {\alpha (\delta)} f ^ {\delta} - K ^ {\dagger} f \right\| _ {\mathcal {U}} \leqslant \left\| R _ {\alpha (\delta)} (f ^ {\delta} - f) \right\| _ {\mathcal {U}} + \left\| R _ {\alpha (\delta)} f - K ^ {\dagger} f \right\| _ {\mathcal {U}}.
$$

$$
\left\| R _ {\alpha (\delta)} (f ^ {\delta} - f) \right\| _ {\mathcal {U}} \leqslant \left\| R _ {\alpha (\delta)} \right\| _ {\mathcal {L} (\mathcal {V}, \mathcal {U})} \left\| f ^ {\delta} - f \right\| _ {\mathcal {V}} \leqslant \frac {1}{2 \sqrt {\alpha (\delta)}} \delta .
$$

$$
\begin{array}{l} \left\| R _ {\alpha (\delta)} f - K ^ {\dagger} f \right\| _ {\mathcal {U}} = \left\| \sum_ {j = 1} ^ {\infty} \left(\frac {\sigma_ {j}}{\sigma_ {j} ^ {2} + \alpha (\delta)} - \frac {1}{\sigma_ {j}}\right) \left\langle f, v _ {j} \right\rangle_ {\mathcal {V}} u _ {j} \right\| _ {\mathcal {U}} \to 0 \text {   as   } \alpha (\delta) \to 0 \text {   if   } \\ f \in \mathcal {D} (K ^ {\dagger}). \end{array}
$$

Thus, if $\alpha ( \delta ) \to 0 \mathsf { A N D } \delta / \sqrt { \alpha ( \delta ) } \to 0$ as $\delta  0$ , then $\mathsf { R } _ { \alpha ( \delta ) } \mathsf { f } ^ { \delta } \to \mathsf { K } ^ { \dagger } \mathsf { f }$ . Tikhonov regularisation with such an $\alpha ( \delta )$ is a convergent regularisation method. 

## Variational Regularisation Methods

## Variational Regularisation: A Major Class of Methods

A significant and widely used approach to constructing regularisation operators is through variational regularisation. 

## Variational Regularisation Operator

The (potentially set-valued) operator $\mathbb { R } _ { \alpha } : \mathcal { V } \Longrightarrow \mathcal { U }$ defined as 

$$
R _ {\alpha} (f ^ {\delta}) := \underset {u \in \mathcal {U}} {\arg \min} \{F (K u, f ^ {\delta}) + J _ {\alpha} (u) \}
$$

is said to be a variational regularisation. Here $\alpha \in A$ are the regularisation parameter(s). 

## Variational Regularisation: A Major Class of Methods

A significant and widely used approach to constructing regularisation operators is through variational regularisation. 

## Variational Regularisation Operator

The (potentially set-valued) operator $\mathbb { R } _ { \alpha } : \mathcal { V } \Longrightarrow \mathcal { U }$ defined as 

$$
R _ {\alpha} (f ^ {\delta}) := \underset {u \in \mathcal {U}} {\arg \min} \{F (K u, f ^ {\delta}) + J _ {\alpha} (u) \}
$$

is said to be a variational regularisation. Here $\alpha \in A$ are the regularisation parameter(s). 

## We have

A data fidelity term, $\mathsf { F } ( \mathsf { K u } , \mathsf { f } ^ { \delta } )$ , measuring how well $\mathsf { K u }$ fits the observed data $\mathsf { f } ^ { \delta }$ . 

A (parameterised) regularisation term (or penalty term), $\mathrm { J } _ { \alpha } ( \mathfrak { u } )$ , which incorporates prior knowledge about the desired solution u (e.g., smoothness, sparsity). 

## Variational Regularisation: Key Theoretical Assumptions

## Assumption (Based on [1, Assumption 5.4])

Let $\ b { \mathcal { U } } = \ b { Z } ^ { * }$ for some normed space Z, and let the weak-star topology on U be metrisable on bounded sets. Moreover assume 

$\mathsf { K } = \mathsf { L } ^ { * }$ for a bounded linear operator L : V → Z. 

## Variational Regularisation: Key Theoretical Assumptions

## Assumption (Based on [1, Assumption 5.4])

Let $\ b { \mathcal { U } } = \ b { Z } ^ { * }$ for some normed space Z, and let the weak-star topology on U be metrisable on bounded sets. Moreover assume 

$\mathsf { K } = \mathsf { L } ^ { * }$ for a bounded linear operator $\mathrm { L } : \mathcal { V } \to \mathrm { Z } \quad$ 

■ $J _ { \alpha } ( \cdot ) = \mathsf { H } _ { \alpha } ^ { * }$ for some proper functional $\mathsf { H } _ { \alpha } : Z \to \mathbb { R } \cup \{ + \infty \}$ , and $\textstyle \int _ { \alpha } ( \cdot )$ non-negative. 

## Variational Regularisation: Key Theoretical Assumptions

## Assumption (Based on [1, Assumption 5.4])

Let $\ b { \mathcal { U } } = \ b { Z } ^ { * }$ for some normed space Z, and let the weak-star topology on U be metrisable on bounded sets. Moreover assume 

$\mathsf { K } = \mathsf { L } ^ { * }$ for a bounded linear operator $\mathrm { L } : \mathcal { V } \to Z$ . 

$J _ { \alpha } ( \cdot ) = \mathsf { H } _ { \alpha } ^ { * }$ for some proper functional $\mathsf { H } _ { \alpha } : Z \to \mathbb { R } \cup \{ + \infty \}$ , and $\textstyle \int _ { \alpha } ( \cdot )$ non-negative. 

F is proper, non-negative, convex functional in first argument, and continuous in second argument; for every ${ \mathfrak { g } } \in { \mathcal { V } }$ there exists $\mathrm { \mathfrak { u } } \in \mathrm { \mathfrak { U } }$ such that $\mathbb { F } ( \mathbb { K } \mathfrak { u } , \mathfrak { g } ) + \ J _ { \alpha } ( \mathfrak { u } ) < \infty$ 

## Variational Regularisation: Key Theoretical Assumptions

## Assumption (Based on [1, Assumption 5.4])

Let $\ b { \mathcal { U } } = \ b { Z } ^ { * }$ for some normed space Z, and let the weak-star topology on U be metrisable on bounded sets. Moreover assume 

$\mathsf { K } = \mathsf { L } ^ { * }$ for a bounded linear operator $\mathrm { L } : \mathcal { V } \to \mathrm { Z } \quad$ 

■ $J _ { \alpha } ( \cdot ) = \mathsf { H } _ { \alpha } ^ { * }$ for some proper functional $\mathsf { H } _ { \alpha } : Z \to \mathbb { R } \cup \{ + \infty \}$ , and $\textstyle \int _ { \alpha } ( \cdot )$ non-negative. 

F is proper, non-negative, convex functional in first argument, and continuous in second argument; for every ${ \mathfrak { g } } \in { \mathcal { V } }$ there exists $\mathrm { \mathfrak { u } } \in \mathrm { \mathfrak { U } }$ such that $\mathbb { F } ( \mathbb { K } \mathfrak { u } , \mathfrak { g } ) + \ J _ { \alpha } ( \mathfrak { u } ) < \infty$ 

For each ${ \mathfrak { g } } \in { \mathcal { V } }$ and $\alpha \in A$ , there exists a constant ${ \mathfrak { c } } = { \mathfrak { c } } ( { \mathfrak { a } } , { \mathfrak { b } } , \| { \mathfrak { g } } \| _ { \mathcal { V } } )$ , which depends monotonically non-decreasingly on all its arguments, such that 

$$
\| u \| _ {\mathcal {U}} \leqslant c \quad \text { if } F (K u, g) \leqslant a \text { and } J _ {\alpha} (u) \leqslant b.
$$

## Variational Regularisation: Operator Stability Under those assumptions we can guarantee

## Well-Posedness of $\mathbb { B }$

For every $\mathsf { f } ^ { \delta } \in \mathcal { V }$ and $\alpha \in A$ , the set of minimisers $\ R _ { \alpha } ( { \sf f ^ { \delta } } )$ is non-empty. If $\mathsf { F } ( \mathsf { K u } , \mathsf { f } ^ { \delta } ) + \mathsf { J } ( \mathsf { u } , \alpha )$ ) is strictly convex, the minimiser is unique. 

## Variational Regularisation: Operator Stability Under those assumptions we can guarantee

## Well-Posedness of $\mathbb { B }$

For every $\mathsf { f } ^ { \delta } \in \mathcal { V }$ and $\alpha \in A$ , the set of minimisers $\ R _ { \alpha } ( { \sf f ^ { \delta } } )$ is non-empty. If $\mathsf { F } ( \mathsf { K u } , \mathsf { f } ^ { \delta } ) + \mathsf { J } ( \mathsf { u } , \alpha )$ is strictly convex, the minimiser is unique. 

## Stability of $\mathbb { B } _ { 0 . 0 } ^ { } \mathbb { B } _ { 1 } ^ { \ast }$ (Operator Property)

If F is continuous w.r.t. its second variable, and $\mathsf { f } ^ { \delta _ { \mathrm { n } } } \to \mathsf { f } ^ { \delta }$ in V: 

Any sequence $\mathfrak { u } _ { \mathfrak { n } } \in \mathfrak { R } _ { \alpha } ( \mathfrak { f } ^ { \delta _ { \mathfrak { n } } } )$ ) possesses a (weakly, weak-*, or strongly, depending on space and functional properties) convergent subsequence $\mathfrak { u } _ { \mathfrak { n } _ { \mathrm { k } } } \to \mathfrak { u } ^ { * }$ 

Crucially, this limit point $\mu ^ { \ast }$ is itself a minimiser for the limit data: $\mathfrak { u } ^ { * } \in \mathbb { R } _ { \alpha } ( \mathfrak { f } ^ { \delta } )$ 

This implies $\emptyset \neq \{ \mathfrak { u } \in \mathcal { U }$ | lim $\mathsf { s u p } _ { \mathsf { k } \to \infty } \mathsf { d } _ { \mathcal { U } } ( \mathsf { u } , \mathsf { R } _ { \alpha } ( \mathsf { f } ^ { \delta _ { \mathsf { k } } } ) ) = 0 \} \subset \mathsf { R } _ { \alpha } ( \mathsf { f } ^ { \delta } )$ . 

## Tikhonov Regularisation (Revisited)

$$
\begin{array}{l} R _ {\alpha} (f ^ {\delta}) = \underset {u \in \mathcal {U}} {\arg \min} \left\{\frac {1}{2} \left\| K u - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2} + \frac {\alpha}{2} \left\| u \right\| _ {\mathcal {U}} ^ {2} \right\}, \\ = (K ^ {*} K + \alpha I) ^ {- 1} K ^ {*} f ^ {\delta}. \end{array}
$$

Here $\begin{array} { r } { \mathsf { F } ( \mathsf { K u } , \mathsf { f } ^ { \delta } ) = \frac 1 2 \left\| \mathsf { K u } - \mathsf { f } ^ { \delta } \right\| _ { \mathcal { V } } ^ { 2 } } \end{array}$ and $\begin{array} { r } { \mathrm { J } _ { \alpha } ( \mathrm { u } ) = \frac { \alpha } { 2 } \left. \mathrm { u } \right. _ { \mathcal { U } } ^ { 2 } } \end{array}$ . This fits the variational framework and is a convergent method with appropriate $\alpha _ { \sf c h o i c e } ( \delta , { \sf f } ^ { \delta } )$ 

## Variational Regularisation: Examples

## LASSO (Least Absolute Shrinkage and Selection Operator)

$$
R _ {\alpha} (f ^ {\delta}) = \underset {u \in \mathbb {R} ^ {n}} {\arg \min} \left\{\frac {1}{2} \left\| K u - f ^ {\delta} \right\| _ {\mathbb {R} ^ {m}} ^ {2} + \alpha \| u \| _ {\ell^ {1}} \right\}
$$

Here $\textstyle \int _ { \alpha } ( \boldsymbol { \mu } ) = \alpha \left\| \boldsymbol { \ u } \right\| _ { \ell ^ { 1 } } = \alpha \sum _ { \mathrm { i } } ^ { \mathrm { n } } | \boldsymbol { \mu } _ { \mathrm { i } } |$ . Promotes sparse solutions. Its convergence analysis relies on convexity and properties of the $\ell ^ { 1 } - \mathsf { n o r m }$ 

## LASSO (Least Absolute Shrinkage and Selection Operator)

$$
R _ {\alpha} (f ^ {\delta}) = \underset {u \in \mathbb {R} ^ {n}} {\arg \min} \left\{\frac {1}{2} \left\| K u - f ^ {\delta} \right\| _ {\mathbb {R} ^ {m}} ^ {2} + \alpha \| u \| _ {\ell^ {1}} \right\}
$$

Here $\textstyle \int _ { \alpha } ( \boldsymbol { \mu } ) = \alpha \left\| \boldsymbol { \ u } \right\| _ { \ell ^ { 1 } } = \alpha \sum _ { \mathrm { i } } ^ { \mathrm { n } } | \boldsymbol { \mu } _ { \mathrm { i } } |$ . Promotes sparse solutions. Its convergence analysis relies on convexity and properties of the $\ell ^ { 1 } - \mathsf { n o r m }$ 

## Total Variation (TV) Regularisation (e.g., for Images with sharp edges)

$$
R _ {\alpha} (f ^ {\delta}) = \underset {u} {\arg \min} \left\{\frac {1}{2} \left\| K u - f ^ {\delta} \right\| ^ {2} + \alpha T V (u) \right\}
$$

where $\mathsf { T V } ( \mathsf { u } ) = \mathsf { s u p } _ { \varphi \in \{ \Phi \in C _ { 0 } ^ { \infty } ( \Omega ; \mathbb { R } ^ { n } ) | | \| \Phi ( \mathsf { x } ) \| _ { 2 } \leqslant 1 \} } \int _ { \Omega } \mathsf { u } ( \mathsf { x } ) \mathsf { d i v } \varphi ( \mathsf { x } ) \mathrm { d } \mathsf { x }$ . 

## Convergence Analysis: Error Estimates

## Error Estimates: Motivation for Deeper Analysis

Once we establish that a regularisation method is convergent (e.g., $\mathsf { R } _ { \alpha _ { \mathrm { n } } } \mathopen { } \mathclose \bgroup ( \mathsf { f } ^ { \delta _ { \mathrm { n } } } \aftergroup \egroup )  \mathrm { u } ^ { * }$ for $\mathfrak { u } ^ { * } \in \mathcal { S } ( \mathfrak { f } )$ as noise $\delta _ { \mathfrak { n } } \to 0$ in the single-valued cases), further important questions arise: 

How good is the approximation? We need a way to measure the error between the regularised solution $\mathsf { R } _ { \alpha } ( \mathsf { f } ^ { \delta } )$ and the desired true solution u<sup>†</sup>. 

## Error Estimates: Motivation for Deeper Analysis

Once we establish that a regularisation method is convergent (e.g., $\mathsf { R } _ { \alpha _ { \mathrm { n } } } \mathopen { } \mathclose \bgroup ( \mathsf { f } ^ { \delta _ { \mathrm { n } } } \aftergroup \egroup )  \mathrm { u } ^ { * }$ for $\mathfrak { u } ^ { * } \in \mathcal { S } ( \mathfrak { f } )$ as noise $\delta _ { \mathfrak { n } } \to 0$ in the single-valued cases), further important questions arise: 

How good is the approximation? We need a way to measure the error between the regularised solution $\mathsf { R } _ { \alpha } ( \mathsf { f } ^ { \delta } )$ and the desired true solution u<sup>†</sup>. 

How fast does the error decrease as the noise level δ vanishes? This refers to the rate of convergence. 

## Error Estimates: Motivation for Deeper Analysis

Once we establish that a regularisation method is convergent (e.g., $\mathsf { R } _ { \alpha _ { \mathrm { n } } } \mathopen { } \mathclose \bgroup ( \mathsf { f } ^ { \delta _ { \mathrm { n } } } \aftergroup \egroup )  \mathrm { u } ^ { * }$ for $\mathfrak { u } ^ { * } \in \mathcal { S } ( \mathfrak { f } )$ as noise $\delta _ { \mathfrak { n } } \to 0$ in the single-valued cases), further important questions arise: 

How good is the approximation? We need a way to measure the error between the regularised solution $\mathsf { R } _ { \alpha } ( \mathsf { f } ^ { \delta } )$ and the desired true solution u<sup>†</sup>. 

How fast does the error decrease as the noise level δ vanishes? This refers to the rate of convergence. 

To address these, we introduce an error measure D : $\mathcal { U } \times \mathcal { U }  \mathbb { R } _ { + } \cup \{ + \infty \}$ in the solution space U. This D is not necessarily a norm (e.g., it could be a Bregman distance). 

## D-Convergence of a Regularisation Method

## D-convergent

Let D : $\mathcal { U } \times \mathcal { U }  \mathbb { R } _ { + } \cup \{ + \infty \}$ be an error measure. Let $\mathfrak { u } ^ { \dagger } \in \mathcal { S } ( \mathfrak { f } )$ be a desired inverse problem solution corresponding to exact data f. A regularisation method (consisting of operators $\mathtt { R } _ { \alpha }$ and parameter choice $\alpha _ { \sf c h o i c e } ( \delta , { \sf f } ^ { \delta } ) )$ is called D-convergent if for any $\mathfrak { u } _ { \alpha ( \delta , \mathfrak { f } ^ { \delta } ) } ^ { \delta } \in \mathrm { R } _ { \alpha ( \delta , \mathfrak { f } ^ { \delta } ) } ( \mathfrak { f } ^ { \delta } )$ we observe 

$$
\lim _ {\delta \rightarrow 0} \sup \left\{D (u _ {\alpha (\delta , f ^ {\delta})} ^ {\delta}, u ^ {\dagger}) \mid f ^ {\delta} \in \mathcal {V}, F (f, f ^ {\delta}) \leqslant \delta \right\} = 0.
$$

## D-Convergence of a Regularisation Method

## D-convergent

Let D : $\mathcal { U } \times \mathcal { U }  \mathbb { R } _ { + } \cup \{ + \infty \}$ be an error measure. Let $\mathfrak { u } ^ { \dagger } \in \mathcal { S } ( \mathfrak { f } )$ be a desired inverse problem solution corresponding to exact data f. A regularisation method (consisting of operators $\mathtt { R } _ { \alpha }$ and parameter choice $\alpha _ { \sf c h o i c e } ( \delta , { \sf f } ^ { \delta } ) )$ is called D-convergent if for any $\mathfrak { u } _ { \alpha ( \delta , \mathfrak { f } ^ { \delta } ) } ^ { \delta } \in \mathrm { R } _ { \alpha ( \delta , \mathfrak { f } ^ { \delta } ) } ( \mathfrak { f } ^ { \delta } )$ we observe 

$$
\lim _ {\delta \rightarrow 0} \sup \left\{D (u _ {\alpha (\delta , f ^ {\delta})} ^ {\delta}, u ^ {\dagger}) \mid f ^ {\delta} \in \mathcal {V}, F (f, f ^ {\delta}) \leqslant \delta \right\} = 0.
$$

$\mathsf { F } \big ( \mathsf { f } , \mathsf { f } ^ { \delta } \big ) \leqslant \delta$ models the noise in the data $\mathsf { f } ^ { \delta }$ 

## D-Convergence of a Regularisation Method

## D-convergent

Let D : $\mathcal { U } \times \mathcal { U }  \mathbb { R } _ { + } \cup \{ + \infty \}$ be an error measure. Let $\mathfrak { u } ^ { \dagger } \in \mathcal { S } ( \mathfrak { f } )$ be a desired inverse problem solution corresponding to exact data f. A regularisation method (consisting of operators $\mathtt { R } _ { \alpha }$ and parameter choice $\alpha _ { \sf c h o i c e } ( \delta , { \sf f } ^ { \delta } ) )$ is called D-convergent if for any $\mathfrak { u } _ { \alpha ( \delta , \mathfrak { f } ^ { \delta } ) } ^ { \delta } \in \mathrm { R } _ { \alpha ( \delta , \mathfrak { f } ^ { \delta } ) } ( \mathfrak { f } ^ { \delta } )$ we observe 

$$
\lim _ {\delta \rightarrow 0} \sup \left\{D (u _ {\alpha (\delta , f ^ {\delta})} ^ {\delta}, u ^ {\dagger}) \mid f ^ {\delta} \in \mathcal {V}, F (f, f ^ {\delta}) \leqslant \delta \right\} = 0.
$$

$\mathsf { F } \big ( \mathsf { f } , \mathsf { f } ^ { \delta } \big ) \leqslant \delta$ models the noise in the data $\mathsf { f } ^ { \delta }$ 

This means the maximum error (measured by D) between any obtained regularised solution and the true solution $\mathfrak { u } ^ { \dagger }$ vanishes as the noise level δ goes to zero. 

## Convergence Rates of a Regularisation Method

To discuss specific rates, we often restrict the ”true” solution $\mathrm { \mathfrak { u } } ^ { \dagger }$ to a smoothness class $\mathcal { M } _ { v } \subset \mathcal { U }$ , where $v > 0$ measures the degree of smoothness or regularity. 

## Convergence Rates of a Regularisation Method

To discuss specific rates, we often restrict the ”true” solution $\mathrm { \mathfrak { u } } ^ { \dagger }$ to a smoothness class $\mathcal { M } _ { v } \subset \mathcal { U }$ , where $v > 0$ measures the degree of smoothness or regularity. 

## Convergent at Order ν

A regularisation method is called convergent at order ν on a set ${ \mathcal { M } } _ { v } \ { \dot { \mathsf { H } } } ,$ for all f such that $\mathsf { K u } ^ { \dagger } = \mathsf { f }$ for some $\mathfrak { u } ^ { \dag } \in \mathfrak { M } _ { v }$ there exists a constant $C _ {  v } > 0$ such that for all data $\mathsf { f } ^ { \delta }$ satisfying $\mathsf { F } \big ( \mathsf { f } , \mathsf { f } ^ { \delta } \big ) \leqslant \delta$ 

$$
\sup _ {u \in R _ {\alpha (\delta , f ^ {\delta})} (f ^ {\delta})} D (u, u ^ {\dagger}) \leqslant C _ {\nu} \delta^ {\nu}
$$

## Convergence Rates of a Regularisation Method

To discuss specific rates, we often restrict the ”true” solution $\mathrm { \mathfrak { u } } ^ { \dagger }$ to a smoothness class $\mathcal { M } _ { v } \subset \mathcal { U }$ , where $v > 0$ measures the degree of smoothness or regularity. 

## Convergent at Order ν

A regularisation method is called convergent at order ν on a set ${ \mathcal { M } } _ { v } \ { \dot { \mathsf { H } } } ,$ for all f such that $\mathsf { K u } ^ { \dagger } = \mathsf { f }$ for some $\mathfrak { u } ^ { \dag } \in \mathfrak { M } _ { v }$ there exists a constant $C _ {  v } > 0$ such that for all data $\mathsf { f } ^ { \delta }$ satisfying $\begin{array} { r } { \mathsf { F } \big ( \mathsf { f } , \mathsf { f } ^ { \delta } \big ) \leqslant \delta \qquad } \end{array}$ 

$$
\sup _ {u \in R _ {\alpha (\delta , f ^ {\delta})} (f ^ {\delta})} D (u, u ^ {\dagger}) \leqslant C _ {\nu} \delta^ {\nu}
$$

■ $\mathsf { I f } \ \mathsf { R } _ { \alpha ( \delta , \mathsf { f } ^ { \delta } ) } \mathopen { } \mathclose \bgroup \left( \mathsf { f } ^ { \delta } \aftergroup \egroup \right)$ is single-valued, this simplifies to $\mathrm { D } \big ( \mathsf { R } _ { \alpha ( \delta , \mathsf { f } ^ { \delta } ) } ( \mathsf { f } ^ { \delta } ) , \mathsf { u } ^ { \dag } \big ) \leqslant \mathsf { C } _ { \mathsf { v } } \delta ^ { \mathsf { v } }$ . 

## Convergence Rates of a Regularisation Method

To discuss specific rates, we often restrict the ”true” solution $\mathrm { \mathfrak { u } } ^ { \dagger }$ to a smoothness class $\mathcal { M } _ { v } \subset \mathcal { U }$ , where $v > 0$ measures the degree of smoothness or regularity. 

## Convergent at Order ν

A regularisation method is called convergent at order ν on a set ${ \mathcal { M } } _ { v } \ { \dot { \mathsf { H } } } ,$ for all f such that $\mathsf { K u } ^ { \dagger } = \mathsf { f }$ for some $\mathfrak { u } ^ { \dag } \in \mathfrak { M } _ { v }$ there exists a constant $C _ {  v } > 0$ such that for all data $\mathsf { f } ^ { \delta }$ satisfying $\begin{array} { r } { \mathsf { F } \big ( \mathsf { f } , \mathsf { f } ^ { \delta } \big ) \leqslant \delta \qquad } \end{array}$ 

$$
\sup _ {u \in R _ {\alpha (\delta , f ^ {\delta})} (f ^ {\delta})} D (u, u ^ {\dagger}) \leqslant C _ {\nu} \delta^ {\nu}
$$

If $\mathsf { R } _ { \alpha ( \delta , \mathsf { f } ^ { \delta } ) } \mathopen { } \mathclose \bgroup \left( \mathsf { f } ^ { \delta } \aftergroup \egroup \right)$ is single-valued, this simplifies to $\mathrm { D } \big ( \mathsf { R } _ { \alpha ( \delta , \mathsf { f } ^ { \delta } ) } ( \mathsf { f } ^ { \delta } ) , \mathsf { u } ^ { \dag } \big ) \leqslant \mathsf { C } _ { \mathsf { v } } \delta ^ { \mathsf { v } }$ . 

This provides a quantitative estimate on how fast the error decreases as a function of the noise level $\delta ,$ for solutions $\mathrm { { u } ^ { \dag } }$ possessing sufficient regularity $( \mathsf { i } . \mathsf { e } . , \mathsf { u } ^ { \dagger } \in \mathfrak { M } _ { v } )$ 

## Tikhonov: Convergence Rate

We use the definition of ”Convergent at Order ν” with error measure $\mathrm { D } ( \mathfrak { u } , \nu ) = \| \mathfrak { u } - \nu \| _ { \mathcal { U } }$ 

## Smoothness Assumption (Source Condition)

Assume the true solution $u^{\dagger} = K^{\dagger}f$ possesses a certain "smoothness". A common assumption is $u^{\dagger} \in \mathcal{R}((K^{*}K)^{\mu})$ for some $\mu > 0$ , i.e., $u^{\dagger} = (K^{*}K)^{\mu}w = \sum_{j=1}^{\infty} \sigma_{j}^{2\mu} \langle w, u_{j} \rangle_{\mathcal{U}} u_{j}$ , for some $w \in \mathcal{U}$ . 

## Tikhonov: Convergence Rate

We use the definition of ”Convergent at Order ν” with error measure $\mathrm { D } ( \mathfrak { u } , \nu ) = \| \mathfrak { u } - \nu \| _ { \mathcal { U } }$ 

## Smoothness Assumption (Source Condition)

Assume the true solution $u^{\dagger} = K^{\dagger}f$ possesses a certain "smoothness". A common assumption is $u^{\dagger} \in \mathcal{R}((K^{*}K)^{\mu})$ for some $\mu > 0$ , i.e., $u^{\dagger} = (K^{*}K)^{\mu}w = \sum_{j=1}^{\infty} \sigma_{j}^{2\mu} \left\langle w, u_{j} \right\rangle_{\mathcal{U}} u_{j}$ , for some $w \in \mathcal{U}$ . 

## Error Bound and Rate for Tikhonov

Under this smoothness assumption $\left( \mu > 0 \right)$ , for $\mathfrak { u } _ { \alpha } ^ { \delta } = \ R _ { \alpha } \mathsf { f } ^ { \delta }$ (Tikhonov), the error can be bounded. The two main error components behave as: 

Data error propagation: $\left\| \mathbf { R } _ { \alpha } ( \mathsf { f } ^ { \delta } - \mathsf { f } ) \right\| _ { \mathcal { U } } \approx \mathcal { O } ( \delta / \sqrt { \alpha } )$ . 

## Tikhonov: Convergence Rate

We use the definition of ”Convergent at Order ν” with error measure $\mathrm { D } ( \mathfrak { u } , \nu ) = \| \mathfrak { u } - \nu \| _ { \mathcal { U } }$ 

## Smoothness Assumption (Source Condition)

Assume the true solution $\mathfrak { u } ^ { \dagger } = \mathfrak { K } ^ { \dagger } \mathfrak { f }$ possesses a certain ”smoothness”. A common assumption is $\mathfrak { u } ^ { \dag } \in \mathcal { R } ( ( \mathsf { K } ^ { \ast } \mathsf { K } ) ^ { \flat } )$ for some $\mu > 0 ;$ , i.e., $\begin{array} { r } { \mathbf { \boldsymbol { \mathsf { u } } } ^ { \dagger } = ( \mathbf { \boldsymbol { K } } ^ { * } \mathbf { \boldsymbol { K } } ) ^ { \mu } \boldsymbol { \mathcal { W } } = \sum _ { \mathrm { j = 1 } } ^ { \infty } \mathbf { \boldsymbol { \sigma } } _ { \mathrm { j } } ^ { 2 \mu } \left. \boldsymbol { w } , \mathbf { \boldsymbol { u } } _ { \mathrm { j } } \right. _ { \mathcal { U } } \mathbf { \boldsymbol { u } } _ { \mathrm { j } } } \end{array}$ , for some $w \in \mathcal { U }$ 

## Error Bound and Rate for Tikhonov

Under this smoothness assumption $\left( \mu > 0 \right)$ , for $\mathfrak { u } _ { \alpha } ^ { \delta } = \ R _ { \alpha } \mathsf { f } ^ { \delta }$ (Tikhonov), the error can be bounded. The two main error components behave as: 

Data error propagation: $\left\| \mathbf { R } _ { \alpha } ( \mathsf { f } ^ { \delta } - \mathsf { f } ) \right\| _ { \mathcal { U } } \approx \mathcal { O } ( \delta / \sqrt { \alpha } )$ . 

Approximation error: $\left\| \mathsf { R } _ { \alpha } \mathsf { f } - \mathsf { K } ^ { \dagger } \mathsf { f } \right\| _ { \mathcal { U } } \approx \mathcal { O } ( \alpha ^ { \mu } )$ for $\mathfrak { u } ^ { \dagger } \in \mathcal { R } ( ( \mathsf { K } ^ { \ast } \mathsf { K } ) ^ { \mu } )$ 

## Tikhonov: Convergence Rate

We use the definition of ”Convergent at Order ν” with error measure $\mathrm { D } ( \mathfrak { u } , \nu ) = \| \mathfrak { u } - \nu \| _ { \mathcal { U } }$ 

## Error Bound and Rate for Tikhonov

Under this smoothness assumption $\left( \mu > 0 \right)$ , for $\mathfrak { u } _ { \alpha } ^ { \delta } = \ R _ { \alpha } \mathsf { f } ^ { \delta }$ (Tikhonov), the error can be bounded. The two main error components behave as: 

Data error propagation: $\left\| \mathbf { R } _ { \alpha } ( \mathsf { f } ^ { \delta } - \mathsf { f } ) \right\| _ { \mathcal { U } } \approx \mathcal { O } ( \delta / \sqrt { \alpha } )$ . 

Approximation error: $\left\| \mathsf { R } _ { \alpha } \mathsf { f } - \mathsf { K } ^ { \dagger } \mathsf { f } \right\| _ { \mathcal { U } } \approx \mathcal { O } ( \alpha ^ { \mu } )$ for $\mathfrak { u } ^ { \dagger } \in \mathcal { R } ( ( \mathsf { K } ^ { \ast } \mathsf { K } ) ^ { \mu } )$ 

Balancing these terms by choosing $\propto ( \delta ) \sim \delta ^ { 2 / ( 2 \mu + 1 ) }$ gives 

$$
\left\| R _ {\alpha (\delta)} f ^ {\delta} - u ^ {\dagger} \right\| _ {\mathcal {U}} \leqslant C \delta^ {\frac {2 \mu}{2 \mu + 1}}.
$$

Hence, ${ \sf R } _ { \alpha ( \delta ) }$ is convergent at order $\pmb { \nu } = \pmb { 2 \mu } / ( \pmb { 2 \mu } + \pmb { 1 } )$ for solutions $\mathfrak { u } ^ { \dagger } \in \mathcal { R } ( ( \mathsf { K } ^ { \ast } \mathsf { K } ) ^ { \mu } )$ $\mu \in \left( 0 , 1 \right]$ . For $\mu = 1$ , the rate is $\mathcal { O } \big ( \delta ^ { 2 / 3 } \big )$ . If $\mathfrak { u } ^ { \dagger } \in \mathcal { R } ( \mathsf { K } ^ { \ast } ) ( \mathsf { i } . \mathsf { e } . \mu = 1 / 2 )$ , the rate is $\Theta ( \sqrt { \delta } )$ 

Beyond Norms: Bregman Distances for Error Measurement The error measure D(u, v) in convergence analysis need not be a norm. Bregman distances [2, 11] offer powerful alternatives for analysing variational regularisations. 

## Beyond Norms: Bregman Distances for Error Measurement

The error measure $\scriptstyle \mathrm { \mathrm { D } } ( \boldsymbol { \mathrm { u } } , \boldsymbol { \nu } )$ in convergence analysis need not be a norm. Bregman distances [2, 11] offer powerful alternatives for analysing variational regularisations. 

## Bregman Distance

Let J : $\mathcal { U } \to \mathbb { R } \cup \{ + \infty \}$ be a proper, convex and lower semi-continuous functional. For $\mathfrak { u } _ { 1 } , \mathfrak { u } _ { 2 } \in \mathcal { U }$ and ${ \mathfrak { p } } _ { 2 } \in \mathfrak { d } J ( \mathfrak { u } _ { 2 } )$ (a subgradient of J at $\mu _ { 2 } )$ , the Bregman distance is defined as 

$$
D _ {J} ^ {p _ {2}} (u _ {1}, u _ {2}) = J (u _ {1}) - J (u _ {2}) - \left\langle p _ {2}, u _ {1} - u _ {2} \right\rangle .
$$

## Beyond Norms: Bregman Distances for Error Measurement

The error measure $\scriptstyle \mathrm { \mathrm { D } } ( \boldsymbol { \mathrm { u } } , \boldsymbol { \nu } )$ in convergence analysis need not be a norm. Bregman distances [2, 11] offer powerful alternatives for analysing variational regularisations. 

## Bregman Distance

Let J : $\mathcal { U } \to \mathbb { R } \cup \{ + \infty \}$ be a proper, convex and lower semi-continuous functional. For $\mathfrak { u } _ { 1 } , \mathfrak { u } _ { 2 } \in \mathcal { U }$ and ${ \mathfrak { p } } _ { 2 } \in \mathfrak { d } J ( \mathfrak { u } _ { 2 } )$ (a subgradient of J at $\mu _ { 2 } )$ , the Bregman distance is defined as 

$$
D _ {J} ^ {p _ {2}} (u _ {1}, u _ {2}) = J (u _ {1}) - J (u _ {2}) - \left\langle p _ {2}, u _ {1} - u _ {2} \right\rangle .
$$

## Key Properties:

Non-negative: $\mathrm { D } _ { \mathrm { J } } ^ { \mathrm { p } _ { 2 } } \big ( \mathfrak { u } _ { 1 } , \mathfrak { u } _ { 2 } \big ) \geqslant 0$ 

## Beyond Norms: Bregman Distances for Error Measurement

The error measure $\scriptstyle \mathrm { \mathrm { D } } ( \boldsymbol { \mathrm { u } } , \boldsymbol { \nu } )$ in convergence analysis need not be a norm. Bregman distances [2, 11] offer powerful alternatives for analysing variational regularisations. 

## Bregman Distance

Let J $: \mathcal { U } \to \mathbb { R } \cup \{ + \infty \}$ be a proper, convex and lower semi-continuous functional. For $\mathfrak { u } _ { 1 } , \mathfrak { u } _ { 2 } \in \mathcal { U }$ and ${ \mathfrak { p } } _ { 2 } \in \mathfrak { d } J ( \mathfrak { u } _ { 2 } )$ (a subgradient of J at $\mu _ { 2 } )$ , the Bregman distance is defined as 

$$
D _ {J} ^ {p _ {2}} (u _ {1}, u _ {2}) = J (u _ {1}) - J (u _ {2}) - \left\langle p _ {2}, u _ {1} - u _ {2} \right\rangle .
$$

## Key Properties:

Non-negative: $\mathrm { D } _ { \mathrm { J } } ^ { \mathsf { p } _ { 2 } } ( \mathfrak { u } _ { 1 } , \mathfrak { u } _ { 2 } ) \geqslant 0$ 

Generally non-symmetric: $\mathrm { D } _ { \mathrm { J } } ^ { \mathsf { p } _ { 2 } } ( \mathfrak { u } _ { 1 } , \mathfrak { u } _ { 2 } ) \neq \mathrm { D } _ { \mathrm { J } } ^ { \mathsf { p } _ { 1 } } ( \mathfrak { u } _ { 2 } , \mathfrak { u } _ { 1 } )$ 

## Beyond Norms: Bregman Distances for Error Measurement

The error measure $\scriptstyle \mathrm { \mathrm { D } } ( \boldsymbol { \mathrm { u } } , \boldsymbol { \nu } )$ in convergence analysis need not be a norm. Bregman distances [2, 11] offer powerful alternatives for analysing variational regularisations. 

## Bregman Distance

Let J : $\mathcal { U } \to \mathbb { R } \cup \{ + \infty \}$ be a proper, convex and lower semi-continuous functional. For $\mathfrak { u } _ { 1 } , \mathfrak { u } _ { 2 } \in \mathcal { U }$ and ${ \mathfrak { p } } _ { 2 } \in \mathfrak { d } J ( \mathfrak { u } _ { 2 } )$ (a subgradient of J at $\mu _ { 2 } )$ , the Bregman distance is defined as 

$$
D _ {J} ^ {p _ {2}} (u _ {1}, u _ {2}) = J (u _ {1}) - J (u _ {2}) - \left\langle p _ {2}, u _ {1} - u _ {2} \right\rangle .
$$

## Key Properties:

Non-negative: $\mathrm { D } _ { \mathrm { J } } ^ { \mathsf { p } _ { 2 } } ( \mathfrak { u } _ { 1 } , \mathfrak { u } _ { 2 } ) \geqslant 0$ 

Generally non-symmetric: $\mathrm { D } _ { \mathrm { J } } ^ { \mathsf { p } _ { 2 } } ( \mathfrak { u } _ { 1 } , \mathfrak { u } _ { 2 } ) \neq \mathrm { D } _ { \mathrm { J } } ^ { \mathsf { p } _ { 1 } } ( \mathfrak { u } _ { 2 } , \mathfrak { u } _ { 1 } )$ 

Symmetric special case: If $\begin{array} { r } { J ( \mathrm { u } ) = \frac { 1 } { 2 } \left. \mathrm { u } \right. ^ { 2 } } \end{array}$ , then $\begin{array} { r } { \mathsf { D } _ { \mathrm { J } } ^ { \mathsf { u } _ { 2 } } ( \mathsf { u } _ { 1 } , \mathsf { u } _ { 2 } ) = \frac { 1 } { 2 } \left\| \mathsf { u } _ { 1 } - \mathsf { u } _ { 2 } \right\| ^ { 2 } } \end{array}$ 

## Visualising Bregman Distance: $\mathrm { J } ( \mathrm { u } ) = \mathrm { u } \log ( \mathrm { u } ) - \mathrm { u }$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/cd051db5-f6e3-4d63-8bee-b9956127e06a/fdd11ebbb7584c80b7e330100efc5a0332d910cb28921d5346d58697bdebdcb8.jpg)


## Interpretation

Function: Convex for $\mathrm {  ~ u ~ } > 0$ 

Derivative: $\mathrm { J } ^ { \prime } ( \mathrm { u } ) = \mathsf { l o g } ( \mathrm { u } )$ 

Subgradient: $\mathfrak { p } _ { 2 } = \log ( \mathfrak { u } _ { 2 } )$ at point $\mathfrak { u } _ { 2 }$ 

## Bregman Distance

$$
D _ {J} ^ {\log (u _ {2})} \left(u _ {1}, u _ {2}\right) = u _ {1} \log \left(\frac {u _ {1}}{u _ {2}}\right) + u _ {2} - u _ {1}
$$

This is the Kullback-Leibler divergence between $\mathbf { u } _ { 1 }$ and $\mathfrak { u } _ { 2 }$ 

## Symmetrised Bregman Distances

To obtain a symmetric measure, one can define the symmetrised Bregman distance: 

## Symmetrised Bregman Distance

Given $\mathfrak { p } \in \mathfrak { d } \mathrm { J } ( \mathfrak { u } )$ and ${ \mathfrak { q } } \in \mathfrak { d } J ( \nu )$ we define 

$$
D _ {J} ^ {\text { symm }} (u, v) = D _ {J} ^ {p} (v, u) + D _ {J} ^ {q} (u, v)
$$

$$
\begin{array}{c} = J (v) - J (u) - \langle p, v - u \rangle + J (u) - J (v) - \langle q, u - v \rangle \\ = \langle q - p, v - u \rangle \geqslant 0. \end{array}
$$

## Symmetrised Bregman Distances

To obtain a symmetric measure, one can define the symmetrised Bregman distance: 

## Symmetrised Bregman Distance

Given $\mathfrak { p } \in \mathfrak { d } \mathrm { J } ( \mathfrak { u } )$ and ${ \mathfrak { q } } \in \mathfrak { d } J ( \nu )$ we define 

$$
D _ {J} ^ {\text { symm }} (u, v) = D _ {J} ^ {p} (v, u) + D _ {J} ^ {q} (u, v)
$$

$$
\begin{array}{c} = J (v) - J (u) - \langle p, v - u \rangle + J (u) - J (v) - \langle q, u - v \rangle \\ = \langle q - p, v - u \rangle \geqslant 0. \end{array}
$$

## Relevance

Symmetrised Bregman distances naturally appear in error estimates for variational regularisation methods, particularly when source conditions are involved. 

## Recap: Error Estimates & Bregman Distances

We are interested in error estimates for variational regularisation with scalar parameter, i.e. 

$$
R _ {\alpha} (f ^ {\delta}) = u _ {\alpha} ^ {\delta} \in \underset {u \in \mathcal {U}} {\arg \min} \left\{\frac {1}{2} \left\| K u - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2} + \alpha J (u) \right\}
$$

Let $\mathrm { { u } ^ { \dag } }$ be the true solution with exact data $\mathsf { f } = \mathsf { K } \mathsf { u } ^ { \dagger }$ 

## Recap: Error Estimates & Bregman Distances

We are interested in error estimates for variational regularisation with scalar parameter, i.e. 

$$
R _ {\alpha} (f ^ {\delta}) = u _ {\alpha} ^ {\delta} \in \underset {u \in \mathcal {U}} {\arg \min} \left\{\frac {1}{2} \left\| K u - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2} + \alpha J (u) \right\}
$$

Let $\mathrm { { u } ^ { \dag } }$ be the true solution with exact data $\mathsf { f } = \mathsf { K } \mathsf { u } ^ { \dagger }$ 

## Key Ingredients

Optimality Condition for $\mathfrak { u } _ { \alpha } ^ { \delta }$ : there exists $\mathfrak { p } _ { \alpha } \in \partial \mathrm { J } ( \mathfrak { u } _ { \alpha } ^ { \delta } )$ such that 

$$
K ^ {*} (K u _ {\alpha} ^ {\delta} - f ^ {\delta}) + \alpha p _ {\alpha} = 0.
$$

## Recap: Error Estimates & Bregman Distances

We are interested in error estimates for variational regularisation with scalar parameter, i.e. 

$$
R _ {\alpha} (f ^ {\delta}) = u _ {\alpha} ^ {\delta} \in \underset {u \in \mathcal {U}} {\arg \min} \left\{\frac {1}{2} \left\| K u - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2} + \alpha J (u) \right\}
$$

Let $\mathrm { { u } ^ { \dag } }$ be the true solution with exact data $\mathsf { f } = \mathsf { K } \mathsf { u } ^ { \dagger }$ 

## Key Ingredients

Optimality Condition for $\mathfrak { u } _ { \alpha } ^ { \delta }$ : there exists $\mathfrak { p } _ { \alpha } \in \partial \mathrm { J } ( \mathfrak { u } _ { \alpha } ^ { \delta } )$ such that 

$$
K ^ {*} (K u _ {\alpha} ^ {\delta} - f ^ {\delta}) + \alpha p _ {\alpha} = 0.
$$

Source Condition for $\mathrm { { u } ^ { \dag } }$ (assumed to hold for $\mathfrak { u } ^ { \dagger } )$ : there exists $\nu \in \mathcal V$ such that 

$$
K ^ {*} v \in \partial J (u ^ {\dagger}).
$$

## Deriving Error Estimates: Step 1 (Main Equation)

1 From the optimality condition we have $\mathsf { K } ^ { * } ( \mathsf { K u } _ { \alpha } ^ { \delta } - \mathsf { f } ^ { \delta } ) + \alpha \mathsf { p } _ { \alpha } = 0$ 

## Deriving Error Estimates: Step 1 (Main Equation)

1 From the optimality condition we have $\mathsf { K } ^ { * } ( \mathsf { K u } _ { \alpha } ^ { \delta } - \mathsf { f } ^ { \delta } ) + \alpha \mathsf { p } _ { \alpha } = 0$ 

2 Subtracting the source condition element v yields 

$$
K ^ {*} (K u _ {\alpha} ^ {\delta} - f ^ {\delta}) + \alpha (p _ {\alpha} - K ^ {*} v) = - \alpha K ^ {*} v
$$

## Deriving Error Estimates: Step 1 (Main Equation)

1 From the optimality condition we have $\mathsf { K } ^ { * } ( \mathsf { K u } _ { \alpha } ^ { \delta } - \mathsf { f } ^ { \delta } ) + \alpha \mathsf { p } _ { \alpha } = 0$ 

2 Subtracting the source condition element v yields 

$$
K ^ {*} (K u _ {\alpha} ^ {\delta} - f ^ {\delta}) + \alpha (p _ {\alpha} - K ^ {*} v) = - \alpha K ^ {*} v
$$

3 Take the dual product with $\mathfrak { u } _ { \alpha } ^ { \delta } - \mathfrak { u } ^ { \dag }$ leads to 

$$
\begin{array}{r} \left\langle K u _ {\alpha} ^ {\delta} - f ^ {\delta}, K (u _ {\alpha} ^ {\delta} - u ^ {\dagger}) \right\rangle_ {\mathcal {V}} + \alpha \left\langle p _ {\alpha} - K ^ {*} v, u _ {\alpha} ^ {\delta} - u ^ {\dagger} \right\rangle_ {\mathcal {U}} = - \alpha \left\langle K ^ {*} v, u _ {\alpha} ^ {\delta} - u ^ {\dagger} \right\rangle_ {\mathcal {U}} \\ = - \alpha \left\langle v, K (u _ {\alpha} ^ {\delta} - u ^ {\dagger}) \right\rangle_ {\mathcal {V}} \end{array}
$$

## Deriving Error Estimates: Step 1 (Main Equation)

1 From the optimality condition we have $\mathsf { K } ^ { * } ( \mathsf { K u } _ { \alpha } ^ { \delta } - \mathsf { f } ^ { \delta } ) + \alpha \mathsf { p } _ { \alpha } = 0$ 

2 Subtracting the source condition element v yields 

$$
K ^ {*} (K u _ {\alpha} ^ {\delta} - f ^ {\delta}) + \alpha (p _ {\alpha} - K ^ {*} v) = - \alpha K ^ {*} v
$$

3 Take the dual product with $\mathfrak { u } _ { \alpha } ^ { \delta } - \mathfrak { u } ^ { \dag }$ leads to 

$$
\begin{array}{r l} \left\langle K u _ {\alpha} ^ {\delta} - f ^ {\delta}, K (u _ {\alpha} ^ {\delta} - u ^ {\dagger}) \right\rangle_ {\mathcal {V}} + \alpha \left\langle p _ {\alpha} - K ^ {*} v, u _ {\alpha} ^ {\delta} - u ^ {\dagger} \right\rangle_ {\mathcal {U}} & = - \alpha \left\langle K ^ {*} v, u _ {\alpha} ^ {\delta} - u ^ {\dagger} \right\rangle_ {\mathcal {U}} \\ & = - \alpha \left\langle v, K (u _ {\alpha} ^ {\delta} - u ^ {\dagger}) \right\rangle_ {\mathcal {V}} \end{array}
$$

4 Recognising that $\big \langle \mathtt { p } _ { \alpha } - \mathtt { K } ^ { \ast } \nu , \mathtt { u } _ { \alpha } ^ { \delta } - \mathtt { u } ^ { \dag } \big \rangle _ { \mathcal { U } } = \mathsf { D } _ { \mathtt { I } } ^ { \mathtt { s y m m } } ( \mathtt { u } _ { \alpha } ^ { \delta } , \mathtt { u } ^ { \dag } )$ , where the specific subgradients $\mathfrak { p } _ { \alpha } \in \partial  J ( \mathfrak { u } _ { \alpha } ^ { \delta } )$ and $\mathsf { K } ^ { \ast } \nu \in \partial \mathrm { J } ( \mathrm { u } ^ { \dag } )$ ) are used 

## Deriving Error Estimates: Step 1 (Main Equation)

1 From the optimality condition we have $\mathsf { K } ^ { * } ( \mathsf { K u } _ { \alpha } ^ { \delta } - \mathsf { f } ^ { \delta } ) + \alpha \mathsf { p } _ { \alpha } = 0$ 

2 Subtracting the source condition element v yields 

$$
K ^ {*} (K u _ {\alpha} ^ {\delta} - f ^ {\delta}) + \alpha (p _ {\alpha} - K ^ {*} v) = - \alpha K ^ {*} v
$$

3 Take the dual product with $\mathfrak { u } _ { \alpha } ^ { \delta } - \mathfrak { u } ^ { \dag }$ leads to 

$$
\begin{array}{c} \left\langle K u _ {\alpha} ^ {\delta} - f ^ {\delta}, K (u _ {\alpha} ^ {\delta} - u ^ {\dagger}) \right\rangle_ {\mathcal {V}} + \alpha \left\langle p _ {\alpha} - K ^ {*} v, u _ {\alpha} ^ {\delta} - u ^ {\dagger} \right\rangle_ {\mathcal {U}} = - \alpha \left\langle K ^ {*} v, u _ {\alpha} ^ {\delta} - u ^ {\dagger} \right\rangle_ {\mathcal {U}} \\ = - \alpha \left\langle v, K (u _ {\alpha} ^ {\delta} - u ^ {\dagger}) \right\rangle_ {\mathcal {V}} \end{array}
$$

4 Recognising that $\big \langle \mathtt { p } _ { \alpha } - \mathtt { K } ^ { \ast } \nu , \mathtt { u } _ { \alpha } ^ { \delta } - \mathtt { u } ^ { \dag } \big \rangle _ { \mathcal { U } } = \mathsf { D } _ { \mathtt { I } } ^ { \mathtt { s y m m } } ( \mathtt { u } _ { \alpha } ^ { \delta } , \mathtt { u } ^ { \dag } )$ , where the specific subgradients $\mathfrak { p } _ { \alpha } \in \partial  J ( \mathfrak { u } _ { \alpha } ^ { \delta } )$ and $\mathsf { K } ^ { \ast } \nu \in \partial \mathrm { J } ( \mathrm { u } ^ { \dag } )$ are used 

5 Hence, with $\mathsf { f } = \mathsf { K } \mathsf { u } ^ { \dagger }$ , the previous equation becomes 

$$
\left\langle K u _ {\alpha} ^ {\delta} - f ^ {\delta}, K u _ {\alpha} ^ {\delta} - f \right\rangle_ {\mathcal {V}} + \alpha D _ {J} ^ {\text {Symm}} (u _ {\alpha} ^ {\delta}, u ^ {\dagger}) = \alpha \left\langle v, f - K u _ {\alpha} ^ {\delta} \right\rangle_ {\mathcal {V}}
$$

## Deriving Error Estimates: Step 2 (Using Identities)

From the previous slide we have the equation 

$$
\left\langle K u _ {\alpha} ^ {\delta} - f ^ {\delta}, K u _ {\alpha} ^ {\delta} - f \right\rangle_ {\mathcal {V}} + \alpha D _ {J} ^ {\text { symm }} (u _ {\alpha} ^ {\delta}, u ^ {\dagger}) = \alpha \left\langle v, f - K u _ {\alpha} ^ {\delta} \right\rangle_ {\mathcal {V}}
$$

## Deriving Error Estimates: Step 2 (Using Identities)

From the previous slide we have the equation 

$$
\left\langle K u _ {\alpha} ^ {\delta} - f ^ {\delta}, K u _ {\alpha} ^ {\delta} - f \right\rangle_ {\mathcal {V}} + \alpha D _ {J} ^ {\text { symm }} (u _ {\alpha} ^ {\delta}, u ^ {\dagger}) = \alpha \left\langle v, f - K u _ {\alpha} ^ {\delta} \right\rangle_ {\mathcal {V}}
$$

We use two standard algebraic identities for inner products: 

1 $\begin{array} { r } { \langle \mathbf { a } - \mathbf { b } , \mathbf { a } - \mathbf { c } \rangle _ { \mathcal { V } } = \frac { 1 } { 2 } \| \mathbf { a } - \mathbf { c } \| _ { \mathcal { V } } ^ { 2 } + \frac { 1 } { 2 } \| \mathbf { a } - \mathbf { b } \| _ { \mathcal { V } } ^ { 2 } - \frac { 1 } { 2 } \| \mathbf { c } - \mathbf { b } \| _ { \mathcal { V } } ^ { 2 } } \end{array}$ . Applying this to the first term with $\mathfrak { a } = \mathsf { K } \mathfrak { u } _ { \alpha } ^ { \delta } , \mathfrak { b } = \mathsf { f } ^ { \delta } , \mathsf { c } = \mathfrak { f }$ yields 

$$
\left\langle K u _ {\alpha} ^ {\delta} - f ^ {\delta}, K u _ {\alpha} ^ {\delta} - f \right\rangle_ {\mathcal {V}} = \frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f \right\| _ {\mathcal {V}} ^ {2} + \frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2} - \frac {1}{2} \left\| f - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2}.
$$

## Deriving Error Estimates: Step 2 (Using Identities)

From the previous slide we have the equation 

$$
\left\langle K u _ {\alpha} ^ {\delta} - f ^ {\delta}, K u _ {\alpha} ^ {\delta} - f \right\rangle_ {\mathcal {V}} + \alpha D _ {J} ^ {\text { symm }} (u _ {\alpha} ^ {\delta}, u ^ {\dagger}) = \alpha \left\langle v, f - K u _ {\alpha} ^ {\delta} \right\rangle_ {\mathcal {V}}
$$

We use two standard algebraic identities for inner products: 

1 $\begin{array} { r } { \langle \mathbf { a } - \mathbf { b } , \mathbf { a } - \mathbf { c } \rangle _ { \mathcal { V } } = \frac { 1 } { 2 } \| \mathbf { a } - \mathbf { c } \| _ { \mathcal { V } } ^ { 2 } + \frac { 1 } { 2 } \| \mathbf { a } - \mathbf { b } \| _ { \mathcal { V } } ^ { 2 } - \frac { 1 } { 2 } \| \mathbf { c } - \mathbf { b } \| _ { \mathcal { V } } ^ { 2 } } \end{array}$ . Applying this to the first term with $\mathfrak { a } = \mathsf { K } \mathfrak { u } _ { \alpha } ^ { \delta } , \mathfrak { b } = \mathsf { f } ^ { \delta } , \mathsf { c } = \mathfrak { f }$ yields 

$$
\left\langle K u _ {\alpha} ^ {\delta} - f ^ {\delta}, K u _ {\alpha} ^ {\delta} - f \right\rangle_ {\mathcal {V}} = \frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f \right\| _ {\mathcal {V}} ^ {2} + \frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2} - \frac {1}{2} \left\| f - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2}.
$$

2 $\begin{array} { r } { \langle { \bf x } , { \bf y } \rangle _ { \mathcal { V } } = \frac { 1 } { 2 } \| { \bf x } \| _ { \mathcal { V } } ^ { 2 } + \frac { 1 } { 2 } \| { \bf y } \| _ { \mathcal { V } } ^ { 2 } - \frac { 1 } { 2 } \| { \bf x } - { \bf y } \| _ { \mathcal { V } } ^ { 2 } } \end{array}$ . Applying this to the right-hand-side yields 

$$
\alpha \left\langle v, f - K u _ {\alpha} ^ {\delta} \right\rangle_ {\mathcal {V}} = \frac {\alpha^ {2}}{2} \left\| v \right\| _ {\mathcal {V}} ^ {2} + \frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f \right\| _ {\mathcal {V}} ^ {2} - \frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f + \alpha v \right\| _ {\mathcal {V}} ^ {2}.
$$

## Deriving Error Estimates: Step 3 (Combining)

Substituting the identities into the main equation from Step 1 leads to 

$$
\begin{array}{l} \left(\frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f \right\| _ {\mathcal {V}} ^ {2} + \frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2} - \frac {1}{2} \left\| f - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2}\right) + \alpha D _ {J} ^ {\text {symm}} (u _ {\alpha} ^ {\delta}, u ^ {\dagger}) \\ = \left(\frac {\alpha^ {2}}{2} \left\| v \right\| _ {\mathcal {V}} ^ {2} + \frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f \right\| _ {\mathcal {V}} ^ {2} - \frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f + \alpha v \right\| _ {\mathcal {V}} ^ {2}\right) \end{array}
$$

## Deriving Error Estimates: Step 3 (Combining)

Substituting the identities into the main equation from Step 1 leads to 

$$
\begin{array}{l} \left(\frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f \right\| _ {\mathcal {V}} ^ {2} + \frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2} - \frac {1}{2} \left\| f - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2}\right) + \alpha D _ {J} ^ {\text {symm}} (u _ {\alpha} ^ {\delta}, u ^ {\dagger}) \\ = \left(\frac {\alpha^ {2}}{2} \left\| v \right\| _ {\mathcal {V}} ^ {2} + \frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f \right\| _ {\mathcal {V}} ^ {2} - \frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f + \alpha v \right\| _ {\mathcal {V}} ^ {2}\right) \end{array}
$$

The term $\begin{array} { r } { \frac { 1 } { 2 } \left. \mathsf { K u } _ { \alpha } ^ { \delta } - \mathsf { f } \right. _ { \mathcal { V } } ^ { 2 } } \end{array}$ cancels on both sides. Rearranging leaves us with 

$$
\frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f + \alpha v \right\| _ {\mathcal {V}} ^ {2} + \frac {1}{2} \left\| K u _ {\alpha} ^ {\delta} - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2} + \alpha D _ {J} ^ {\mathrm{symm}} (u _ {\alpha} ^ {\delta}, u ^ {\dagger}) = \frac {1}{2} \left\| f - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2} + \frac {\alpha^ {2}}{2} \left\| v \right\| _ {\mathcal {V}} ^ {2}
$$

## Deriving Error Estimates: Step 4 (Bounding)

## The Error Bound

Since the first two terms on the left-hand-side are non-negative, we observe 

$$
\alpha D _ {J} ^ {\text {symm}} (u _ {\alpha} ^ {\delta}, u ^ {\dagger}) \leqslant \frac {1}{2} \left\| f - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2} + \frac {\alpha^ {2}}{2} \left\| v \right\| _ {\mathcal {V}} ^ {2}
$$

## Deriving Error Estimates: Step 4 (Bounding)

## The Error Bound

Since the first two terms on the left-hand-side are non-negative, we observe 

$$
\alpha D _ {J} ^ {\text {symm}} (u _ {\alpha} ^ {\delta}, u ^ {\dagger}) \leqslant \frac {1}{2} \left\| f - f ^ {\delta} \right\| _ {\mathcal {V}} ^ {2} + \frac {\alpha^ {2}}{2} \left\| v \right\| _ {\mathcal {V}} ^ {2}
$$

With the noise estimate $\left. \mathbf { f } - \mathbf { f } ^ { \delta } \right. _ { \mathcal { V } } \leqslant \delta$ we further obtain 

$$
D _ {J} ^ {\text { symm }} (u _ {\alpha} ^ {\delta}, u ^ {\dagger}) \leqslant \frac {\delta^ {2}}{2 \alpha} + \frac {\alpha}{2} \| v \| _ {V} ^ {2}
$$

This is a common form of error estimate for variational regularisation (for quadratic fidelity terms). 

## D-Convergence and Rates from Bregman Estimate

We have derived $\begin{array} { r } { \mathrm { D } _ { \mathrm { ~ J ~ } } ^ { \mathsf { s y m m } } ( \mathfrak { u } _ { \alpha } ^ { \delta } , \mathfrak { u } ^ { \dagger } ) \leqslant \frac { \delta ^ { 2 } } { 2 \alpha } + \frac { \alpha } { 2 } \| \nu \| _ { \mathcal { V } } ^ { 2 } } \end{array}$ . Using $\mathsf { D } ( \cdot , \cdot ) = \mathsf { D } _ { \jmath } ^ { \mathsf { s y m m } } ( \cdot , \cdot )$ as our error measure, we can achieve D-convergence. 

## D-Convergence and Rates from Bregman Estimate

We have derived $\begin{array} { r } { \mathrm { D } _ { \mathrm { ~ J ~ } } ^ { \mathsf { s y m m } } ( \mathfrak { u } _ { \alpha } ^ { \delta } , \mathfrak { u } ^ { \dagger } ) \leqslant \frac { \delta ^ { 2 } } { 2 \alpha } + \frac { \alpha } { 2 } \| \nu \| _ { \mathcal { V } } ^ { 2 } } \end{array}$ . Using $\mathsf { D } ( \cdot , \cdot ) = \mathsf { D } _ { \jmath } ^ { \mathsf { s y m m } } ( \cdot , \cdot )$ as our error measure, we can achieve D-convergence. 

## Achieving D-Convergence and Rates

D-Convergence: If the parameter choice strategy ensures $\alpha ( \delta )  0$ and $\delta ^ { 2 } / \alpha ( \delta )  0$ as $\delta  0$ , then the right-hand-side converges to zero. This implies $\mathrm { D } _ { \mathrm { J } } ^ { \mathsf { s y m m } } ( \mathfrak { u } _ { \alpha ( \delta ) } ^ { \delta } , \mathfrak { u } ^ { \dagger } ) \to 0$ , making the method D-convergent. 

## D-Convergence and Rates from Bregman Estimate

We have derived $\begin{array} { r } { \mathrm { D } _ { \mathrm { ~ J ~ } } ^ { \mathsf { s y m m } } ( \mathfrak { u } _ { \alpha } ^ { \delta } , \mathfrak { u } ^ { \dagger } ) \leqslant \frac { \delta ^ { 2 } } { 2 \alpha } + \frac { \alpha } { 2 } \| \nu \| _ { \mathcal { V } } ^ { 2 } } \end{array}$ . Using $\mathsf { D } ( \cdot , \cdot ) = \mathsf { D } _ { \jmath } ^ { \mathsf { s y m m } } ( \cdot , \cdot )$ as our error measure, we can achieve D-convergence. 

## Achieving D-Convergence and Rates

D-Convergence: If the parameter choice strategy ensures $\alpha ( \delta )  0$ and $\delta ^ { 2 } / \alpha ( \delta )  0$ as $\delta  0$ , then the right-hand-side converges to zero. This implies $\mathrm { D } _ { \mathrm { J } } ^ { \mathsf { s y m m } } ( \mathfrak { u } _ { \alpha ( \delta ) } ^ { \delta } , \mathfrak { u } ^ { \dagger } ) \to 0$ , making the method D-convergent. 

Convergence Rate (Example: Order $v = 1 )$ : To optimise the bound, choose $\begin{array} { r } { \alpha ( \delta ) = \alpha ( \delta ) = \frac { \delta } { \| \nu \| _ { \mathcal { V } } } } \end{array}$ , which yields 

$$
D _ {J} ^ {\text { symm }} (u _ {\alpha (\delta)} ^ {\delta}, u ^ {\dagger}) \leqslant \| v \| _ {\mathcal {V}} \delta
$$

This shows convergence at order ν = 1 with rate constant ${ \mathrm { C } } _ { 1 } = \| \nu \| _ { \mathcal { V } }$ . 

## Iterative Regularisation Methods

## Iterative Regularisation: Motivation

We’ve explored variational methods of the form: 

$$
u _ {\alpha} ^ {\delta} \in R _ {\alpha} (f ^ {\delta}) = \underset {u \in \mathcal {U}} {\arg \min} \left\{F (K u, f ^ {\delta}) + \alpha J (u) \right\}
$$

## Iterative Regularisation: Motivation

We’ve explored variational methods of the form: 

$$
u _ {\alpha} ^ {\delta} \in R _ {\alpha} (f ^ {\delta}) = \underset {u \in \mathcal {U}} {\arg \min} \left\{F (K u, f ^ {\delta}) + \alpha J (u) \right\}
$$

How do we compute these regularised solutions $\mathfrak { u } _ { \alpha } ^ { \delta ~ ? }$ Often by using iterative optimisation algorithms. 

## Iterative Regularisation: Motivation

We’ve explored variational methods of the form: 

$$
u _ {\alpha} ^ {\delta} \in R _ {\alpha} (f ^ {\delta}) = \underset {u \in \mathcal {U}} {\arg \min} \left\{F (K u, f ^ {\delta}) + \alpha J (u) \right\}
$$

How do we compute these regularised solutions $\mathfrak { u } _ { \alpha } ^ { \delta ~ ? }$ Often by using iterative optimisation algorithms. 

Question: Can an iterative algorithm itself, when applied to $\mathsf { f } ^ { \delta }$ and stopped early, act as a regularisation method? 

## Iterative Regularisation: Motivation

We’ve explored variational methods of the form: 

$$
u _ {\alpha} ^ {\delta} \in R _ {\alpha} (f ^ {\delta}) = \underset {u \in \mathcal {U}} {\arg \min} \left\{F (K u, f ^ {\delta}) + \alpha J (u) \right\}
$$

How do we compute these regularised solutions $\mathfrak { u } _ { \alpha } ^ { \delta ~ ? }$ Often by using iterative optimisation algorithms. 

Question: Can an iterative algorithm itself, when applied to $\mathsf { f } ^ { \delta }$ and stopped early, act as a regularisation method? 

Answer: Yes! This is the core idea of iterative regularisation. The number of iterations ${ \boldsymbol { \mathrm { k } } } ^ { * }$ becomes the regularisation parameter. 

## Iterative Regularisation: Motivation

We’ve explored variational methods of the form: 

$$
u _ {\alpha} ^ {\delta} \in R _ {\alpha} (f ^ {\delta}) = \underset {u \in \mathcal {U}} {\arg \min} \left\{F (K u, f ^ {\delta}) + \alpha J (u) \right\}
$$

How do we compute these regularised solutions $\mathfrak { u } _ { \alpha } ^ { \delta ~ ? }$ Often by using iterative optimisation algorithms. 

Question: Can an iterative algorithm itself, when applied to $\mathsf { f } ^ { \delta }$ and stopped early, act as a regularisation method? 

Answer: Yes! This is the core idea of iterative regularisation. The number of iterations ${ \boldsymbol { \mathrm { k } } } ^ { * }$ becomes the regularisation parameter. 

## Example: PDHG for J-Minimising Solutions with Noisy Data

We’ll examine the PDHG algorithm for finding a J-minimising solution to ${ \mathrm { k u } } = { \mathrm { f } }$ , even when we only have noisy data $\mathsf { f } ^ { \delta }$ (cf. [12]). 

## PDHG for J-Minimising Solutions with Noisy Data

Recall the problem of finding a J-minimising solution: 

$$
\inf _ {u \in \mathcal {U}} J (u) \quad \text { subject   to } \quad K u = f
$$

## PDHG for J-Minimising Solutions with Noisy Data

Recall the problem of finding a J-minimising solution: 

$$
\inf _ {u \in \mathcal {U}} J (u) \quad \text { subject   to } \quad K u = f
$$

Let $\mathfrak { u } ^ { \dag } \in \mathcal { U }$ be such a solution, and $\boldsymbol \nu ^ { \dagger } \in \mathcal { V }$ be a corresponding dual variable (source condition element) such that $\mathsf { K } ^ { * } \nu ^ { \dagger } \in \partial \mathrm { J } ( \mathrm { u } ^ { \dagger } )$ ) and $\mathsf { K u } ^ { \dagger } = \mathsf { f }$ 

## PDHG for J-Minimising Solutions with Noisy Data

Recall the problem of finding a J-minimising solution: 

$$
\inf _ {u \in \mathcal {U}} J (u) \quad \text { subject   to } \quad K u = f
$$

Let $\mathfrak { u } ^ { \dag } \in \mathcal { U }$ be such a solution, and $\boldsymbol \nu ^ { \dagger } \in \mathcal { V }$ be a corresponding dual variable (source condition element) such that $\mathsf { K } ^ { * } \nu ^ { \dagger } \in \partial \mathrm { J } ( \mathrm { u } ^ { \dagger } )$ ) and $\mathsf { K } \mathsf { u } ^ { \dagger } = \mathsf { f }$ 

## Algorithm with Noisy Data $\mathbb { 1 } ^ { \sharp ( \bullet ) }$

The PDHG algorithm for this problem, using noisy data $\mathsf { f } ^ { \delta }$ where $\left. \mathbf { f } - \mathbf { f } ^ { \delta } \right. _ { \mathcal { V } } \leqslant \delta$ reads 

$$
u ^ {k + 1} = \operatorname{prox} _ {\tau J} \left(u ^ {k} + \tau K ^ {*} v ^ {k}\right)
$$

$$
v ^ {k + 1} = v ^ {k} - \sigma \left(K (2 u ^ {k + 1} - u ^ {k}) - f ^ {\delta}\right)
$$

with $\tau , \sigma > 0$ such that τσ $< 1 / \| \mathsf { K } \| ^ { 2 }$ . We assume $\mathfrak { u } ^ { 0 } = 0 , \nu ^ { 0 } = 0 .$ 

## PDHG for J-Minimising Solutions with Noisy Data

## Algorithm with Noisy Data $\mathbb { 1 } ^ { \sharp ( \bullet ) }$

The PDHG algorithm for this problem, using noisy data $\mathsf { f } ^ { \delta }$ where $\left. \mathbf { f } - \mathbf { f } ^ { \delta } \right. _ { \mathcal { V } } \leqslant \delta$ reads 

$$
u ^ {k + 1} = \operatorname{prox} _ {\tau J} \left(u ^ {k} + \tau K ^ {*} v ^ {k}\right)
$$

$$
v ^ {k + 1} = v ^ {k} - \sigma \left(K (2 u ^ {k + 1} - u ^ {k}) - f ^ {\delta}\right)
$$

with $\tau , \sigma > 0$ such that $\tau { \sigma } < 1 / \| \mathsf { K } \| ^ { 2 }$ . We assume $\mathfrak { u } ^ { 0 } = 0 , \nu ^ { 0 } = 0$ 

Goal: Show that producing $\overline { { \mathfrak { u } } } ^ { \mathrm { k } }$ (e.g., Cesaro mean of` $\mathrm { \Omega } _ { \mathrm { u } } \mathrm { j }$ for $j = 1 , \ldots , \operatorname { k } )$ with $\boldsymbol { \mathrm { k } } = \boldsymbol { \mathrm { k } } ^ { * } ( \delta )$ ) (an early stopping rule) makes this a convergent regularisation method. 

## Convergence Analysis: Key Ingredients

Let $w ^ { \mathrm { k } } = ( \mathfrak { u } ^ { \mathrm { k } } , \nu ^ { \mathrm { k } } )$ and $\boldsymbol { w } ^ { \dagger } = ( \boldsymbol { \mathsf { u } } ^ { \dagger } , \boldsymbol { \nu } ^ { \dagger } )$ ). The analysis relies on properties of the algorithm and saddle-point conditions. 

## Convergence Analysis: Key Ingredients

Let $w ^ { \mathrm { k } } = ( \mathfrak { u } ^ { \mathrm { k } } , \nu ^ { \mathrm { k } } )$ and $\boldsymbol { w } ^ { \dagger } = ( \boldsymbol { \mathsf { u } } ^ { \dagger } , \boldsymbol { \nu } ^ { \dagger } )$ . The analysis relies on properties of the algorithm and saddle-point conditions. 

Define a symmetric operator $M : \mathcal { U } \times \mathcal { V } \to \mathcal { U } \times \mathcal { V }$ and its associated M-norm 

$$
M := \left( \begin{array}{c c} \frac {1}{\tau} I & - K ^ {*} \\ - K & \frac {1}{\sigma} I \end{array} \right), \quad \| w \| _ {M} ^ {2} := \langle M w, w \rangle  .
$$

Note: for $\tau { \sigma } < 1 / \| \mathsf { K } \| ^ { 2 }$ , M is (symmetric) positive definite. 

## Convergence Analysis: Key Ingredients

Let $w ^ { \mathrm { k } } = ( \mathfrak { u } ^ { \mathrm { k } } , \nu ^ { \mathrm { k } } )$ and $\boldsymbol { w } ^ { \dagger } = ( \boldsymbol { \mathsf { u } } ^ { \dagger } , \boldsymbol { \nu } ^ { \dagger } )$ ). The analysis relies on properties of the algorithm and saddle-point conditions. 

Define a symmetric operator $M : \mathcal { U } \times \mathcal { V } \to \mathcal { U } \times \mathcal { V }$ and its associated M-norm 

$$
M := \left( \begin{array}{c c} \frac {1}{\tau} I & - K ^ {*} \\ - K & \frac {1}{\sigma} I \end{array} \right), \quad \| w \| _ {M} ^ {2} := \langle M w, w \rangle  .
$$

Note: for $\tau { \sigma } < 1 / \| \mathsf { K } \| ^ { 2 }$ , M is (symmetric) positive definite. 

Then, the updates of the PDHG algorithm satisfy the equivalent system of optimality conditions 

$$
\binom{0}{0} \in \binom{\partial J (u ^ {k}) - K ^ {*} v ^ {k}}{K u ^ {k} - f ^ {\delta}} + M (w ^ {k} - w ^ {k - 1}).
$$

## Convergence Analysis: Key Ingredients

Let $w ^ { \mathrm { k } } = ( \mathfrak { u } ^ { \mathrm { k } } , \nu ^ { \mathrm { k } } )$ and $\boldsymbol { w } ^ { \dagger } = ( \boldsymbol { \mathsf { u } } ^ { \dagger } , \boldsymbol { \nu } ^ { \dagger } )$ ). The analysis relies on properties of the algorithm and saddle-point conditions. 

Then, the updates of the PDHG algorithm satisfy the equivalent system of optimality conditions 

$$
\binom{0}{0} \in \binom{\partial J (u ^ {k}) - K ^ {*} v ^ {k}}{K u ^ {k} - f ^ {\delta}} + M (w ^ {k} - w ^ {k - 1})  .
$$

The J-minimising solution $\mathrm { \mathfrak { u } } ^ { \dagger }$ and corresponding source condition element $\nu ^ { \dagger }$ satisfy 

$$
\binom{0}{0} \in \binom{\partial J (u ^ {\dagger}) - K ^ {*} v ^ {\dagger}}{K u ^ {\dagger} - f}  .
$$

## Convergence Analysis: Key Ingredients

Subtracting one condition from the other yields 

$$
\binom{0}{0} \in \binom{\partial J (u ^ {k}) - \partial J (u ^ {\dagger}) - K ^ {*} (v ^ {k} - v ^ {\dagger})}{K u ^ {k} - f ^ {\delta} + f - K u ^ {\dagger}} + M (w ^ {k} - w ^ {k - 1})  .
$$

## Convergence Analysis: Key Ingredients

Subtracting one condition from the other yields 

$$
\binom{0}{0} \in \binom{\partial J (u ^ {k}) - \partial J (u ^ {\dagger}) - K ^ {*} (v ^ {k} - v ^ {\dagger})}{K u ^ {k} - f ^ {\delta} + f - K u ^ {\dagger}} + M (w ^ {k} - w ^ {k - 1})  .
$$

Taking the dual product with $w ^ { \boldsymbol { \mathrm { k } } } - w ^ { \dagger }$ , we obtain 

$$
0 = D _ {J} ^ {\text { symm }} (u ^ {k}, u ^ {\dagger}) + \langle f - f ^ {\delta}, v ^ {k} - v ^ {\dagger} \rangle + \langle M (w ^ {k} - w ^ {k - 1}), w ^ {k} - w ^ {\dagger} \rangle ,
$$

## Convergence Analysis: Key Ingredients

Subtracting one condition from the other yields 

$$
\binom{0}{0} \in \binom{\partial J (u ^ {k}) - \partial J (u ^ {\dagger}) - K ^ {*} (v ^ {k} - v ^ {\dagger})}{K u ^ {k} - f ^ {\delta} + f - K u ^ {\dagger}} + M (w ^ {k} - w ^ {k - 1})  .
$$

Taking the dual product with $w ^ { \boldsymbol { \mathrm { k } } } - w ^ { \dagger }$ , we obtain 

$$
0 = D _ {J} ^ {\text { symm }} (u ^ {k}, u ^ {\dagger}) + \langle f - f ^ {\delta}, v ^ {k} - v ^ {\dagger} \rangle + \langle M (w ^ {k} - w ^ {k - 1}), w ^ {k} - w ^ {\dagger} \rangle ,
$$

respectively 

$$
0 \leqslant D _ {J} ^ {\text { symm }} (u ^ {k}, u ^ {\dagger}) = \langle M (w ^ {k - 1} - w ^ {k}), w ^ {k} - w ^ {\dagger} \rangle + \langle f ^ {\delta} - f, v ^ {k} - v ^ {\dagger} \rangle .
$$

Convergence Analysis: Key Ingredients Hence, if we define $\tilde { w } : = \ M ^ { - 1 } \left( \ O _ { \mathsf { f } ^ { \delta } } ^ { \bar { 0 } } \right)$ , we estimate 

$$
0 \leqslant \left\langle M \left(w ^ {k - 1} - w ^ {k}\right), w ^ {k} - w ^ {\dagger} \right\rangle + \left\langle f ^ {\delta} - f, v ^ {k} - v ^ {\dagger} \right\rangle ,
$$

Convergence Analysis: Key Ingredients Hence, if we define $\tilde { w } : = \ M ^ { - 1 } \left( \ O _ { \mathsf { f } ^ { \delta } } ^ { \bar { 0 } } \right)$ , we estimate 

$$
\begin{array}{l} 0 \leqslant \left\langle M \left(w ^ {k - 1} - w ^ {k}\right), w ^ {k} - w ^ {\dagger} \right\rangle + \left\langle f ^ {\delta} - f, v ^ {k} - v ^ {\dagger} \right\rangle , \\ = \left\langle M \left(w ^ {k - 1} - w ^ {k}\right), w ^ {k} - w ^ {\dagger} \right\rangle + \left\langle \binom {0} {f ^ {\delta}} - \binom {0} {f}, w ^ {k} - w ^ {\dagger} \right\rangle , \end{array}
$$

## Convergence Analysis: Key Ingredients Hence, if we define $\tilde { w } : = \ M ^ { - 1 } \left( \ O _ { \mathsf { f } ^ { \delta } } ^ { \bar { 0 } } \right)$ , we estimate

$$
\begin{array}{l} 0 \leqslant \left\langle M \left(w ^ {k - 1} - w ^ {k}\right), w ^ {k} - w ^ {\dagger} \right\rangle + \left\langle f ^ {\delta} - f, v ^ {k} - v ^ {\dagger} \right\rangle , \\ = \left\langle M \left(w ^ {k - 1} - w ^ {k}\right), w ^ {k} - w ^ {\dagger} \right\rangle + \left\langle \binom {0} {f ^ {\delta}} - \binom {0} {f}, w ^ {k} - w ^ {\dagger} \right\rangle , \\ = \left\langle M \left(w ^ {k - 1} - w ^ {k}\right), w ^ {k} - w ^ {\dagger} \right\rangle + \left\langle M \left(M ^ {- 1} \left(\binom {0} {f ^ {\delta}} - \binom {0} {f}\right)\right), w ^ {k} - w ^ {\dagger} \right\rangle , \end{array}
$$