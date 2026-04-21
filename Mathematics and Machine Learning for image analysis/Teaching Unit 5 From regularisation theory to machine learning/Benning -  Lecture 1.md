# Benning -  Lecture 1.pdf

## 第1页

Regularisation Theory
What is a regularisation and why do we regularise?
Martin Benning
University College London
Erasmus+ International PhD Summer School 2025
Mathematics and Machine Learning for Image Analysis
University of Bologna
9 June 2025
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 1 ∞


---

## 第2页

Regularisation Theory – What is a regularisation and why do we regularise?
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 2 ∞


---

## 第3页

Regularisation Theory – What is a regularisation and why do we regularise?
Learning optimal sampling strategies for Magnetic Resonance Imaging (MRI)
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 2 ∞


---

## 第4页

Regularisation Theory – What is a regularisation and why do we regularise?
Learning optimal sampling strategies for Magnetic Resonance Imaging (MRI)
Lifted training and inversion of neural networks
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 2 ∞


---

## 第5页

Focus of First Lecture(s)
Regularisation theory
What is a regularisation and why do we regularise?
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 3 ∞


---

## 第6页

Contents
1
Introduction to Inverse Problems
2
Fundamental Concepts in Regularisation
3
Selecting Solutions
4
Convergent Regularisation Methods
5
Variational Regularisation Methods
6
Convergence Analysis: Error Estimates
7
Iterative Regularisation Methods
8
Data-Driven Regularisation: Spectral Methods
9
Outlook and Open Questions
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 4 ∞


---

## 第7页

Introduction to Inverse Problems
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 5 ∞


---

## 第8页

Introduction: What is an Inverse Problem?
General Form
Mathematically, an inverse problem can be described as solving the operator equation:
Ku = f
for u, where
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 6 ∞


---

## 第9页

Introduction: What is an Inverse Problem?
General Form
Mathematically, an inverse problem can be described as solving the operator equation:
Ku = f
for u, where
u ∈U is the unknown quantity we want to determine (e.g., an image, a function).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 6 ∞


---

## 第10页

Introduction: What is an Inverse Problem?
General Form
Mathematically, an inverse problem can be described as solving the operator equation:
Ku = f
for u, where
u ∈U is the unknown quantity we want to determine (e.g., an image, a function).
f ∈V is the given measurement data.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 6 ∞


---

## 第11页

Introduction: What is an Inverse Problem?
General Form
Mathematically, an inverse problem can be described as solving the operator equation:
Ku = f
for u, where
u ∈U is the unknown quantity we want to determine (e.g., an image, a function).
f ∈V is the given measurement data.
K : U →V is an operator mapping from a Banach space U to a Banach space V.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 6 ∞


---

## 第12页

Introduction: What is an Inverse Problem?
General Form
Mathematically, an inverse problem can be described as solving the operator equation:
Ku = f
for u, where
u ∈U is the unknown quantity we want to determine (e.g., an image, a function).
f ∈V is the given measurement data.
K : U →V is an operator mapping from a Banach space U to a Banach space V.
This operator K models the forward process (how u generates f).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 6 ∞


---

## 第13页

Introduction: What is an Inverse Problem?
General Form
Mathematically, an inverse problem can be described as solving the operator equation:
Ku = f
for u, where
u ∈U is the unknown quantity we want to determine (e.g., an image, a function).
f ∈V is the given measurement data.
K : U →V is an operator mapping from a Banach space U to a Banach space V.
This operator K models the forward process (how u generates f).
Some useful references: [5, 14, 1]
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 6 ∞


---

## 第14页

Inverse Problems: A Visual Illustration
The Shadow Image Problem
Forward Problem: Given 3D hand
shapes, compute 2D shadows
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 7 ∞


---

## 第15页

Inverse Problems: A Visual Illustration
The Shadow Image Problem
Forward Problem: Given 3D hand
shapes, compute 2D shadows
Inverse Problem: Given only the 2D
shadow silhouettes, determine the
original 3D object
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 7 ∞


---

## 第16页

Inverse Problems: A Visual Illustration
The Shadow Image Problem
Forward Problem: Given 3D hand
shapes, compute 2D shadows
Inverse Problem: Given only the 2D
shadow silhouettes, determine the
original 3D object
Multiple different 3D objects can
produce same shadow
(non-uniqueness)
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 7 ∞


---

## 第17页

The Challenge: Well-Posedness vs. Ill-Posedness
Conditions for Well-Posedness
Most practical inverse problems are ill-posed in the sense of Hadamard [7, 8] and John [9].
A problem is well-posed if it satisfies
1 Existence: A solution u exists for all f ∈V.
2 Uniqueness: The solution is unique.
3 Stability (Continuity): The solution u depends continuously on the data f. Small
changes in f lead to small changes in u.
If any of these conditions are violated, the problem is ill-posed. Stability is often the most
problematic condition in practice.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 8 ∞


---

## 第18页

Example: The Inverse Problem of Differentiation
Problem Setup
Consider the task of finding a function u(x) given its integral f(y). If we assume f(0) = 0,
we want to find u = f′. This can be formulated as solving Ku = f where the operator K is
integration, i.e.
(Ku)(y) =
Z y
0
u(x) dx = f(y) .
Here, K : C([0, 1]) →{g ∈C1([0, 1]) | g(0) = 0} (or suitable Lp spaces, e.g.,
L2([0, 1]) →L2([0, 1])).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 9 ∞


---

## 第19页

Example: The Inverse Problem of Differentiation
Problem Setup
Consider the task of finding a function u(x) given its integral f(y). If we assume f(0) = 0,
we want to find u = f′. This can be formulated as solving Ku = f where the operator K is
integration, i.e.
(Ku)(y) =
Z y
0
u(x) dx = f(y) .
Here, K : C([0, 1]) →{g ∈C1([0, 1]) | g(0) = 0} (or suitable Lp spaces, e.g.,
L2([0, 1]) →L2([0, 1])).
Goal
Our goal is to recover u (the derivative) from f (the integral). This is the inverse problem of
differentiation.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 9 ∞


---

## 第20页

Ill-Posedness of Differentiation: Setup
Suppose instead of the exact data f, we observe noisy data fδ = f + nδ, where nδ is some
noise term. We are interested in uδ = (fδ)′ = f′ + (nδ)′ = u + (nδ)′.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 10 ∞


---

## 第21页

Ill-Posedness of Differentiation: Setup
Suppose instead of the exact data f, we observe noisy data fδ = f + nδ, where nδ is some
noise term. We are interested in uδ = (fδ)′ = f′ + (nδ)′ = u + (nδ)′.
A Perturbation Example
Consider the sequence of noise functions nδ ∈L∞([0, 1]):
nδ(x) := δ sin
kx
δ

for a fixed but arbitrary number k > 0.
The noise in the data can be made arbitrarily small:
nδ
L∞([0,1]) = δ →0
as δ →0
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 10 ∞


---

## 第22页

Ill-Posedness of Differentiation: Analysis
A Perturbation Example (continued)
However, the derivative of this noise is:
(nδ)′(x) = k cos
kx
δ

The error in the reconstructed derivative uδ is:
u −uδ
L∞([0,1]) =
(nδ)′
L∞([0,1]) = k
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 11 ∞


---

## 第23页

Ill-Posedness of Differentiation: Analysis
A Perturbation Example (continued)
However, the derivative of this noise is:
(nδ)′(x) = k cos
kx
δ

The error in the reconstructed derivative uδ is:
u −uδ
L∞([0,1]) =
(nδ)′
L∞([0,1]) = k
Conclusion on Ill-Posedness
Despite noise becoming arbitrarily small (
nδ
L∞→0), the error in uδ remains k.
Therefore, uδ does not depend continuously on fδ in the L∞norm. The problem is ill-posed.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 11 ∞


---

## 第24页

The Operator’s ”Blueprint”: Singular Value Decomposition (SVD)
Many inverse problems involve compact linear operators K : U →V between Hilbert spaces.
The SVD provides a fundamental way to understand the action of such operators.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 12 ∞


---

## 第25页

The Operator’s ”Blueprint”: Singular Value Decomposition (SVD)
Many inverse problems involve compact linear operators K : U →V between Hilbert spaces.
The SVD provides a fundamental way to understand the action of such operators.
Singular Value Decomposition (SVD) of K
For a compact operator K, there exist:
Singular values: A sequence σ1 ⩾σ2 ⩾· · · > 0. These are positive numbers that
typically decay towards zero (σj →0) if the range of K is infinite-dimensional.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 12 ∞


---

## 第26页

The Operator’s ”Blueprint”: Singular Value Decomposition (SVD)
Many inverse problems involve compact linear operators K : U →V between Hilbert spaces.
The SVD provides a fundamental way to understand the action of such operators.
Singular Value Decomposition (SVD) of K
For a compact operator K, there exist:
Singular values: A sequence σ1 ⩾σ2 ⩾· · · > 0. These are positive numbers that
typically decay towards zero (σj →0) if the range of K is infinite-dimensional.
Orthonormal sets (or bases for relevant subspaces):
{uj}j∈N in U (input space elements, form an orthonormal basis of N(K)⊥).
{vj}j∈N in V (output space elements, form an orthonormal basis of R(K)).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 12 ∞


---

## 第27页

SVD: How K Acts
The SVD components are linked by these key relationships:
Kuj = σjvj (The operator K maps uj to vj, scaled/attenuated by σj).
K∗vj = σjuj (The adjoint operator K∗maps vj back to uj, also scaled by σj).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 13 ∞


---

## 第28页

SVD: How K Acts
The SVD components are linked by these key relationships:
Kuj = σjvj (The operator K maps uj to vj, scaled/attenuated by σj).
K∗vj = σjuj (The adjoint operator K∗maps vj back to uj, also scaled by σj).
Representation of K’s Action
Any w ∈U can be represented using the uj, and K acts on w as
Kw =
∞
X
j=1
σj

w, uj

U
|
{z
}
coeff. of w along uj
vj
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 13 ∞


---

## 第29页

SVD: How K Acts
Kuj = σjvj (The operator K maps uj to vj, scaled/attenuated by σj).
K∗vj = σjuj (The adjoint operator K∗maps vj back to uj, also scaled by σj).
Representation of K’s Action
Any w ∈U can be represented using the uj, and K acts on w as
Kw =
∞
X
j=1
σj

w, uj

U
|
{z
}
coeff. of w along uj
vj
SVD and Ill-Posedness
Decay of singular values σj →0 is primary source of ill-posedness for compact operators.
To find u from Ku = f, one might think of u ≈P 1
σj

f, vj

uj. If σj become very small, any
noise in

f, vj

gets greatly amplified.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 13 ∞


---

## 第30页

SVD for Integration: The Forward Operator
We revisit the inverse problem of differentiation. The forward operator
K : L2([0, 1]) →L2([0, 1]) is the integration operator
(Ku)(y) =
Z y
0
u(x)dx .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 14 ∞


---

## 第31页

SVD for Integration: The Forward Operator
We revisit the inverse problem of differentiation. The forward operator
K : L2([0, 1]) →L2([0, 1]) is the integration operator
(Ku)(y) =
Z y
0
u(x)dx .
This can be written as an integral operator with kernel k(x, y), i.e.
(Ku)(y) =
Z 1
0
k(x, y)u(x)dx,
where k(x, y) =

1
if x ⩽y
0
if x > y
.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 14 ∞


---

## 第32页

SVD for Integration: The Forward Operator
We revisit the inverse problem of differentiation. The forward operator
K : L2([0, 1]) →L2([0, 1]) is the integration operator
(Ku)(y) =
Z y
0
u(x)dx .
This can be written as an integral operator with kernel k(x, y), i.e.
(Ku)(y) =
Z 1
0
k(x, y)u(x)dx,
where k(x, y) =

1
if x ⩽y
0
if x > y
.
This operator K is compact. Our goal is to find its SVD.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 14 ∞


---

## 第33页

Computing the Adjoint Operator K∗
The adjoint operator K∗is defined by ⟨Ku, v⟩L2([0,1]) = ⟨u, K∗v⟩L2([0,1]). We observe
⟨Ku, v⟩=
Z 1
0
Z y
0
u(x)dx

v(y)dy
=
Z 1
0
Z 1
0
k(x, y)u(x)v(y)dxdy
=
Z 1
0
u(x)
Z 1
x
v(y)dy

dx
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 15 ∞


---

## 第34页

Computing the Adjoint Operator K∗
The adjoint operator K∗is defined by ⟨Ku, v⟩L2([0,1]) = ⟨u, K∗v⟩L2([0,1]). We observe
⟨Ku, v⟩=
Z 1
0
Z y
0
u(x)dx

v(y)dy
=
Z 1
0
Z 1
0
k(x, y)u(x)v(y)dxdy
=
Z 1
0
u(x)
Z 1
x
v(y)dy

dx
Thus, the adjoint operator K∗: L2([0, 1]) →L2([0, 1]) is given by
(K∗v)(x) =
Z 1
x
v(y)dy .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 15 ∞


---

## 第35页

The Operator K∗K and Eigenvalue Problem
Next, we form the operator K∗K, i.e.
(K∗Ku)(x) = K∗((Ku)(·)) (x)
=
Z 1
x
Z y
0
u(z)dz

dy .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 16 ∞


---

## 第36页

The Operator K∗K and Eigenvalue Problem
Next, we form the operator K∗K, i.e.
(K∗Ku)(x) = K∗((Ku)(·)) (x)
=
Z 1
x
Z y
0
u(z)dz

dy .
We seek eigenvalues λ = σ2 > 0 and eigenfunctions u ∈L2([0, 1]) for K∗K, i.e.
K∗Ku = λu .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 16 ∞


---

## 第37页

The Operator K∗K and Eigenvalue Problem
Next, we form the operator K∗K, i.e.
(K∗Ku)(x) = K∗((Ku)(·)) (x)
=
Z 1
x
Z y
0
u(z)dz

dy .
We seek eigenvalues λ = σ2 > 0 and eigenfunctions u ∈L2([0, 1]) for K∗K, i.e.
K∗Ku = λu .
This leads to the solving the integral equation
Z 1
x
Z y
0
u(z)dz

dy = λu(x) .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 16 ∞


---

## 第38页

Deriving the ODE and Boundary Conditions
Differentiating with respect to x (using Leibniz integral rule) yields
λu′(x) = −
Z x
0
u(z)dz .
From this, setting x = 0, we get u′(0) = 0.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 17 ∞


---

## 第39页

Deriving the ODE and Boundary Conditions
Differentiating with respect to x (using Leibniz integral rule) yields
λu′(x) = −
Z x
0
u(z)dz .
From this, setting x = 0, we get u′(0) = 0.
Differentiating again with respect to x gives λu′′(x) = −u(x). This leads to the
following Ordinary Differential Equation (ODE):
λu′′(x) + u(x) = 0
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 17 ∞


---

## 第40页

Deriving the ODE and Boundary Conditions
Differentiating with respect to x (using Leibniz integral rule) yields
λu′(x) = −
Z x
0
u(z)dz .
From this, setting x = 0, we get u′(0) = 0.
Differentiating again with respect to x gives λu′′(x) = −u(x). This leads to the
following Ordinary Differential Equation (ODE):
λu′′(x) + u(x) = 0
From the integral equation for λu(x), if we set x = 1, the outer integral vanishes, i.e.
λu(1) = 0. Since we seek λ > 0, we must have u(1) = 0.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 17 ∞


---

## 第41页

Deriving the ODE and Boundary Conditions
Differentiating with respect to x (using Leibniz integral rule) yields
λu′(x) = −
Z x
0
u(z)dz .
From this, setting x = 0, we get u′(0) = 0.
Differentiating again with respect to x gives λu′′(x) = −u(x). This leads to the
following Ordinary Differential Equation (ODE):
λu′′(x) + u(x) = 0
From the integral equation for λu(x), if we set x = 1, the outer integral vanishes, i.e.
λu(1) = 0. Since we seek λ > 0, we must have u(1) = 0.
Summary of ODE Problem
We need to solve λu′′(x) + u(x) = 0 with boundary conditions u′(0) = 0 and u(1) = 0.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 17 ∞


---

## 第42页

Singular Values σj and Orthonormal Functions uj
The general solution to u′′(x) + 1
λu(x) = 0 is u(x) = c1 sin(x/
√
λ) + c2 cos(x/
√
λ). Let
σ :=
√
λ.
Apply u′(0) = 0: u′(x) = c1
σ cos(x/σ) −c2
σ sin(x/σ). u′(0) = c1
σ = 0 =⇒c1 = 0. So,
u(x) = c2 cos(x/σ).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 18 ∞


---

## 第43页

Singular Values σj and Orthonormal Functions uj
The general solution to u′′(x) + 1
λu(x) = 0 is u(x) = c1 sin(x/
√
λ) + c2 cos(x/
√
λ). Let
σ :=
√
λ.
Apply u′(0) = 0: u′(x) = c1
σ cos(x/σ) −c2
σ sin(x/σ). u′(0) = c1
σ = 0 =⇒c1 = 0. So,
u(x) = c2 cos(x/σ).
Apply u(1) = 0: c2 cos(1/σ) = 0. For non-trivial solutions (c2 ̸= 0), we require
cos(1/σ) = 0. This means 1/σ = (j −1
2)π for j ∈N.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 18 ∞


---

## 第44页

Singular Values σj and Orthonormal Functions uj
The general solution to u′′(x) + 1
λu(x) = 0 is u(x) = c1 sin(x/
√
λ) + c2 cos(x/
√
λ). Let
σ :=
√
λ.
Apply u′(0) = 0: u′(x) = c1
σ cos(x/σ) −c2
σ sin(x/σ). u′(0) = c1
σ = 0 =⇒c1 = 0. So,
u(x) = c2 cos(x/σ).
Apply u(1) = 0: c2 cos(1/σ) = 0. For non-trivial solutions (c2 ̸= 0), we require
cos(1/σ) = 0. This means 1/σ = (j −1
2)π for j ∈N.
The singular values are σj =
1
(j−1
2 )π =
2
(2j−1)π for j ∈N.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 18 ∞


---

## 第45页

Singular Values σj and Orthonormal Functions uj
The general solution to u′′(x) + 1
λu(x) = 0 is u(x) = c1 sin(x/
√
λ) + c2 cos(x/
√
λ). Let
σ :=
√
λ.
Apply u′(0) = 0: u′(x) = c1
σ cos(x/σ) −c2
σ sin(x/σ). u′(0) = c1
σ = 0 =⇒c1 = 0. So,
u(x) = c2 cos(x/σ).
Apply u(1) = 0: c2 cos(1/σ) = 0. For non-trivial solutions (c2 ̸= 0), we require
cos(1/σ) = 0. This means 1/σ = (j −1
2)π for j ∈N.
The singular values are σj =
1
(j−1
2 )π =
2
(2j−1)π for j ∈N. The corresponding normalised
eigenfunctions uj(x) (after choosing c2 =
√
2 for normalisation) are
uj(x) =
√
2 cos

j −1
2

πx

.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 18 ∞


---

## 第46页

Computing Orthonormal Functions vj
The singular functions vj are obtained via vj =
1
σj Kuj, i.e.
(Kuj)(x) =
Z x
0
uj(y)dy =
Z x
0
√
2 cos

j −1
2

πy

dy ,
=
√
2
"
sin((j −1
2)πy)
(j −1
2)π
#x
0
=
√
2sin((j −1
2)πx)
(j −1
2)π
,
= σj
√
2 sin

j −1
2

πx

.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 19 ∞


---

## 第47页

Computing Orthonormal Functions vj
The singular functions vj are obtained via vj =
1
σj Kuj, i.e.
(Kuj)(x) =
Z x
0
uj(y)dy =
Z x
0
√
2 cos

j −1
2

πy

dy ,
=
√
2
"
sin((j −1
2)πy)
(j −1
2)π
#x
0
=
√
2sin((j −1
2)πx)
(j −1
2)π
,
= σj
√
2 sin

j −1
2

πx

.
Therefore,
vj(x) = 1
σj
(Kuj)(x) =
√
2 sin

j −1
2

πx

.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 19 ∞


---

## 第48页

SVD Summary for Integration Operator
For the integration operator K : L2([0, 1]) →L2([0, 1]), (Ku)(y) =
Ry
0 u(x)dx:
Singular values: σj = 2/((2j −1)π) for j ∈N.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 20 ∞


---

## 第49页

SVD Summary for Integration Operator
For the integration operator K : L2([0, 1]) →L2([0, 1]), (Ku)(y) =
Ry
0 u(x)dx:
Singular values: σj = 2/((2j −1)π) for j ∈N.
Orthonormal functions (uj): uj(x) =
√
2 cos
  j −1
2

πx

.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 20 ∞


---

## 第50页

SVD Summary for Integration Operator
For the integration operator K : L2([0, 1]) →L2([0, 1]), (Ku)(y) =
Ry
0 u(x)dx:
Singular values: σj = 2/((2j −1)π) for j ∈N.
Orthonormal functions (uj): uj(x) =
√
2 cos
  j −1
2

πx

.
Orthonormal functions (vj): vj(x) =
√
2 sin
  j −1
2

πx

.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 20 ∞


---

## 第51页

SVD Summary for Integration Operator
For the integration operator K : L2([0, 1]) →L2([0, 1]), (Ku)(y) =
Ry
0 u(x)dx:
Singular values: σj = 2/((2j −1)π) for j ∈N.
Orthonormal functions (uj): uj(x) =
√
2 cos
  j −1
2

πx

.
Orthonormal functions (vj): vj(x) =
√
2 sin
  j −1
2

πx

.
Expansion of Kw
(Kw)(x) =
∞
X
j=1
2
(2j −1)π
Z 1
0
w(s)
√
2 cos

j −1
2

πs

ds
 √
2 sin

j −1
2

πx

=
∞
X
j=1
4
(2j −1)π
Z 1
0
w(s) cos

j −1
2

πs

ds

sin

j −1
2

πx

Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 20 ∞


---

## 第52页

Fundamental Concepts in Regularisation
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 21 ∞


---

## 第53页

Motivation for Regularisation
Why Regularisation?
Since many inverse problems Ku = f are ill-posed (especially regarding stability), direct
inversion or naive solutions are often highly sensitive to noise in the data f.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 22 ∞


---

## 第54页

Motivation for Regularisation
Why Regularisation?
Since many inverse problems Ku = f are ill-posed (especially regarding stability), direct
inversion or naive solutions are often highly sensitive to noise in the data f.
Regularisation methods aim to find stable approximate solutions by incorporating prior
knowledge or preferences about the solution u.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 22 ∞


---

## 第55页

Motivation for Regularisation
Why Regularisation?
Since many inverse problems Ku = f are ill-posed (especially regarding stability), direct
inversion or naive solutions are often highly sensitive to noise in the data f.
Regularisation methods aim to find stable approximate solutions by incorporating prior
knowledge or preferences about the solution u.
We need a formal framework to define what constitutes a ”good” regularisation.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 22 ∞


---

## 第56页

General Regularisation Operators: Definition
Let U, V be metric spaces. For x ∈X, S ⊂X, define d(x, S) := infv∈S d(x, v).
Regularisation Operator [1]
Set-valued operators Rα : V ⇒U (parameterised by α ∈A ⊂Rm) are called
regularisation operators if for each fixed α ∈A and for all fδ ∈V and sequences fδn ∈V
converging to fδ (i.e., dV(fδn, fδ) →0), we have
∅̸=

u ∈U
 lim sup
k→∞
dU(u, Rα(fδk)) = 0

⊂Rα(fδ)
The set in the middle is the Kuratowski limit inferior of the sequence of sets Rα(fδn).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 23 ∞


---

## 第57页

General Regularisation Operators: Definition
Let U, V be metric spaces. For x ∈X, S ⊂X, define d(x, S) := infv∈S d(x, v).
Regularisation Operator [1]
Set-valued operators Rα : V ⇒U (parameterised by α ∈A ⊂Rm) are called
regularisation operators if for each fixed α ∈A and for all fδ ∈V and sequences fδn ∈V
converging to fδ (i.e., dV(fδn, fδ) →0), we have
∅̸=

u ∈U
 lim sup
k→∞
dU(u, Rα(fδk)) = 0

⊂Rα(fδ)
The set in the middle is the Kuratowski limit inferior of the sequence of sets Rα(fδn).
Regularisation Method
A regularisation operator Rα together with a parameter choice strategy
αchoice : (0, δ0) × V →A, denoted α(δ, fδ), forms a regularisation method.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 23 ∞


---

## 第58页

Regularisation Operators: Single-Valued Case
Simplification for Single-Valued Operators
If Rα : V →U is a single-valued operator for each α ∈A then for Rα to be a regularisation
operator the previous condition simplifies to Rα being continuous on V.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 24 ∞


---

## 第59页

Regularisation Operators: Single-Valued Case
Simplification for Single-Valued Operators
If Rα : V →U is a single-valued operator for each α ∈A then for Rα to be a regularisation
operator the previous condition simplifies to Rα being continuous on V.That is, if fδn →fδ
in V, then Rα(fδn) →Rα(fδ) in U.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 24 ∞


---

## 第60页

Regularisation Operators: Single-Valued Case
Simplification for Single-Valued Operators
If Rα : V →U is a single-valued operator for each α ∈A then for Rα to be a regularisation
operator the previous condition simplifies to Rα being continuous on V.That is, if fδn →fδ
in V, then Rα(fδn) →Rα(fδ) in U.
Verification: If Rα is continuous and un = Rα(fδn) →Rα(fδ) = u, then
{v | limk ∥v −uk∥U = 0} = {u}. The stability condition becomes ∅̸= {u} ⊂{u}, which is true.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 24 ∞


---

## 第61页

Example: Spectral Regularisation Operators
We will now examine a common family of regularisation operators known as spectral
regularisation operators.
General Form of Spectral Regularisation
For a compact linear operator K : U →V between Hilbert spaces with SVD {(σj, uj, vj)},
spectral regularisation operators Rα : V →U are defined by
Rαf =
∞
X
j=1
gα(σj)

f, vj

V uj
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 25 ∞


---

## 第62页

Example: Spectral Regularisation Operators
We will now examine a common family of regularisation operators known as spectral
regularisation operators.
General Form of Spectral Regularisation
For a compact linear operator K : U →V between Hilbert spaces with SVD {(σj, uj, vj)},
spectral regularisation operators Rα : V →U are defined by
Rαf =
∞
X
j=1
gα(σj)

f, vj

V uj
α > 0 is the regularisation parameter.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 25 ∞


---

## 第63页

Example: Spectral Regularisation Operators
We will now examine a common family of regularisation operators known as spectral
regularisation operators.
General Form of Spectral Regularisation
For a compact linear operator K : U →V between Hilbert spaces with SVD {(σj, uj, vj)},
spectral regularisation operators Rα : V →U are defined by
Rαf =
∞
X
j=1
gα(σj)

f, vj

V uj
α > 0 is the regularisation parameter.
gα : R>0 →R⩾0 are known as filter functions. They modify the way singular values
contribute to the reconstruction.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 25 ∞


---

## 第64页

Example: Spectral Regularisation Operators
We will now examine a common family of regularisation operators known as spectral
regularisation operators.
General Form of Spectral Regularisation
For a compact linear operator K : U →V between Hilbert spaces with SVD {(σj, uj, vj)},
spectral regularisation operators Rα : V →U are defined by
Rαf =
∞
X
j=1
gα(σj)

f, vj

V uj
α > 0 is the regularisation parameter.
gα : R>0 →R⩾0 are known as filter functions. They modify the way singular values
contribute to the reconstruction.
These operators Rα are linear and single-valued.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 25 ∞


---

## 第65页

Spectral Rα as Regularisation Operators
Recall from the previous slide: for a single-valued operator Rα : V →U to be a
regularisation operator, it must be continuous on V for each fixed α.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 26 ∞


---

## 第66页

Spectral Rα as Regularisation Operators
Recall from the previous slide: for a single-valued operator Rα : V →U to be a
regularisation operator, it must be continuous on V for each fixed α.
Condition for Continuity of Spectral Rα
A spectral regularisation operator Rα is continuous if its associated filter function gα(σ) is
bounded for fixed α, i.e.
sup
σ>0
|gα(σ)| ⩽Cα < ∞.
If this holds, Rα is a bounded linear operator, because
∥Rαf∥2
U =
∞
X
j=1
|gα(σj)|2|

f, vj

V |2 ⩽C2
α
∞
X
j=1
|

f, vj

V |2 ⩽C2
α ∥f∥2
V .
Thus, ∥Rα∥L(V,U) ⩽Cα, which implies that Rα is continuous.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 26 ∞


---

## 第67页

Spectral Regularisation Methods
If a spectral operator Rα is continuous for a fixed α (due to its filter function gα(σ) being
bounded by Cα), it qualifies as a regularisation operator.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 27 ∞


---

## 第68页

Spectral Regularisation Methods
If a spectral operator Rα is continuous for a fixed α (due to its filter function gα(σ) being
bounded by Cα), it qualifies as a regularisation operator.
When such a regularisation operator Rα is combined with a parameter choice strategy
α(δ, fδ), it forms a spectral regularisation method.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 27 ∞


---

## 第69页

Spectral Regularisation Methods
If a spectral operator Rα is continuous for a fixed α (due to its filter function gα(σ) being
bounded by Cα), it qualifies as a regularisation operator.
When such a regularisation operator Rα is combined with a parameter choice strategy
α(δ, fδ), it forms a spectral regularisation method.
Example: Tikhonov Regularisation [17, 16, 15]
The filter function for Tikhonov regularisation is gα(σ) = σ/(σ2 + α).
For any fixed α > 0, this function is bounded:
sup
σ>0

σ
σ2 + α
 =
1
2√α =: Cα < ∞
Hence, for each α > 0, the operator Rα is a continuous regularisation operator.
Paired with α(δ, fδ), (Rα, α(δ, fδ)) forms the Tikhonov regularisation method.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 27 ∞


---

## 第70页

Tikhonov Regularisation in Spectral Form
Tikhonov Filter Function
Applied to our integration operator (Ku)(y) =
Ry
0 u(x)dx, we have
(Rαf)(x) =
∞
X
j=1
4(2j −1)2π2
4 + α(2j −1)2π2
Z 1
0
f(s) sin

j −1
2

πx

ds

cos

j −1
2

πx

.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 28 ∞


---

## 第71页

Tikhonov Regularisation in Spectral Form
Tikhonov Filter Function
Applied to our integration operator (Ku)(y) =
Ry
0 u(x)dx, we have
(Rαf)(x) =
∞
X
j=1
4(2j −1)2π2
4 + α(2j −1)2π2
Z 1
0
f(s) sin

j −1
2

πx

ds

cos

j −1
2

πx

.
General Tikhonov Form
Note: for any bounded linear operator K : U →V between Hilbert spaces, Tikhonov
regularisation can be written as
Rα(fδ) = (K∗K + αI)−1K∗fδ
This form doesn’t require knowledge of the SVD and applies generally.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 28 ∞


---

## 第72页

Tikhonov as an Optimisation Problem
Variational Formulation
The Tikhonov regularised solution Rα(fδ) is also the unique minimiser of
Rα(fδ) = arg min
u∈U
1
2∥Ku −fδ∥2
V + α
2 ∥u∥2
U

.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 29 ∞


---

## 第73页

Tikhonov as an Optimisation Problem
Variational Formulation
The Tikhonov regularised solution Rα(fδ) is also the unique minimiser of
Rα(fδ) = arg min
u∈U
1
2∥Ku −fδ∥2
V + α
2 ∥u∥2
U

.
Interpretation
Data fidelity term: 1
2∥Ku −fδ∥2
V ensures the solution fits the observed data.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 29 ∞


---

## 第74页

Tikhonov as an Optimisation Problem
Variational Formulation
The Tikhonov regularised solution Rα(fδ) is also the unique minimiser of
Rα(fδ) = arg min
u∈U
1
2∥Ku −fδ∥2
V + α
2 ∥u∥2
U

.
Interpretation
Data fidelity term: 1
2∥Ku −fδ∥2
V ensures the solution fits the observed data.
Regularisation term: α
2 ∥u∥2
U penalises large solution norms.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 29 ∞


---

## 第75页

Tikhonov as an Optimisation Problem
Variational Formulation
The Tikhonov regularised solution Rα(fδ) is also the unique minimiser of
Rα(fδ) = arg min
u∈U
1
2∥Ku −fδ∥2
V + α
2 ∥u∥2
U

.
Interpretation
Data fidelity term: 1
2∥Ku −fδ∥2
V ensures the solution fits the observed data.
Regularisation term: α
2 ∥u∥2
U penalises large solution norms.
Balance: Parameter α > 0 controls trade-off between data fidelity and regularity.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 29 ∞


---

## 第76页

Tikhonov as an Optimisation Problem
Variational Formulation
The Tikhonov regularised solution Rα(fδ) is also the unique minimiser of
Rα(fδ) = arg min
u∈U
1
2∥Ku −fδ∥2
V + α
2 ∥u∥2
U

.
Interpretation
Data fidelity term: 1
2∥Ku −fδ∥2
V ensures the solution fits the observed data.
Regularisation term: α
2 ∥u∥2
U penalises large solution norms.
Balance: Parameter α > 0 controls trade-off between data fidelity and regularity.
This variational approach is equivalent to both expression (K∗K + αI)−1K∗fδ and
spectral form shown on previous slide.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 29 ∞


---

## 第77页

Example: Neural Network
Pre-trained Neural Network
Consider a pre-trained feed-forward neural network Nθ : V →U, i.e.
Rθ(fδ) := Nθ(fδ) = WLσL−1(. . . W2σ1(W1fδ + b1) + b2 . . . ) + bL
The parameters θ = {(Wj, bj)L
j=1} are fixed. Rθ is single-valued.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 30 ∞


---

## 第78页

Example: Neural Network
Pre-trained Neural Network
Consider a pre-trained feed-forward neural network Nθ : V →U, i.e.
Rθ(fδ) := Nθ(fδ) = WLσL−1(. . . W2σ1(W1fδ + b1) + b2 . . . ) + bL
The parameters θ = {(Wj, bj)L
j=1} are fixed. Rθ is single-valued.
Verification as a Regularisation Operator
The parameter α in the definition corresponds to the fixed set of weights θ.
Each layer (affine transformation Wj(·) + bj and activation σj) is typically continuous.
Standard activation functions (ReLU, sigmoid, tanh, etc.) are continuous.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 30 ∞


---

## 第79页

Example: Neural Network
Pre-trained Neural Network
Consider a pre-trained feed-forward neural network Nθ : V →U, i.e.
Rθ(fδ) := Nθ(fδ) = WLσL−1(. . . W2σ1(W1fδ + b1) + b2 . . . ) + bL
The parameters θ = {(Wj, bj)L
j=1} are fixed. Rθ is single-valued.
Verification as a Regularisation Operator
The parameter α in the definition corresponds to the fixed set of weights θ.
Each layer (affine transformation Wj(·) + bj and activation σj) is typically continuous.
Standard activation functions (ReLU, sigmoid, tanh, etc.) are continuous.
A finite composition of continuous functions is continuous.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 30 ∞


---

## 第80页

Example: Neural Network
Pre-trained Neural Network
Consider a pre-trained feed-forward neural network Nθ : V →U, i.e.
Rθ(fδ) := Nθ(fδ) = WLσL−1(. . . W2σ1(W1fδ + b1) + b2 . . . ) + bL
The parameters θ = {(Wj, bj)L
j=1} are fixed. Rθ is single-valued.
Verification as a Regularisation Operator
The parameter α in the definition corresponds to the fixed set of weights θ.
Each layer (affine transformation Wj(·) + bj and activation σj) is typically continuous.
Standard activation functions (ReLU, sigmoid, tanh, etc.) are continuous.
A finite composition of continuous functions is continuous.
Thus, the neural network Nθ is a continuous function from V to U.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 30 ∞


---

## 第81页

Example: Neural Network
Pre-trained Neural Network
Consider a pre-trained feed-forward neural network Nθ : V →U, i.e.
Rθ(fδ) := Nθ(fδ) = WLσL−1(. . . W2σ1(W1fδ + b1) + b2 . . . ) + bL
The parameters θ = {(Wj, bj)L
j=1} are fixed. Rθ is single-valued.
Verification as a Regularisation Operator
The parameter α in the definition corresponds to the fixed set of weights θ.
A finite composition of continuous functions is continuous.
Thus, the neural network Nθ is a continuous function from V to U (assuming standard
topologies, e.g., if V = Rm, U = Rn).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 30 ∞


---

## 第82页

Example: Neural Network
Pre-trained Neural Network
Consider a pre-trained feed-forward neural network Nθ : V →U, i.e.
Rθ(fδ) := Nθ(fδ) = WLσL−1(. . . W2σ1(W1fδ + b1) + b2 . . . ) + bL
The parameters θ = {(Wj, bj)L
j=1} are fixed. Rθ is single-valued.
Verification as a Regularisation Operator
The parameter α in the definition corresponds to the fixed set of weights θ.
A finite composition of continuous functions is continuous.
Thus, the neural network Nθ is a continuous function from V to U (assuming standard
topologies, e.g., if V = Rm, U = Rn).
Therefore, Rθ(fδ) = Nθ(fδ) (for fixed pre-trained θ) is a regularisation operator.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 30 ∞


---

## 第83页

Beyond Stability: The Need for Convergence
We’ve established that regularisation operators Rα provide stable processing of data for a
fixed parameter α. A regularisation method (Rα, α(δ, fδ)) then uses a rule to select α.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 31 ∞


---

## 第84页

Beyond Stability: The Need for Convergence
We’ve established that regularisation operators Rα provide stable processing of data for a
fixed parameter α. A regularisation method (Rα, α(δ, fδ)) then uses a rule to select α.
Why This Isn’t the Whole Story
Stability for each fixed α is crucial, but it doesn’t guarantee that our method produces
solutions that are close to the true underlying solution of Ku = f.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 31 ∞


---

## 第85页

Beyond Stability: The Need for Convergence
We’ve established that regularisation operators Rα provide stable processing of data for a
fixed parameter α. A regularisation method (Rα, α(δ, fδ)) then uses a rule to select α.
Why This Isn’t the Whole Story
Stability for each fixed α is crucial, but it doesn’t guarantee that our method produces
solutions that are close to the true underlying solution of Ku = f.
As the noise δ in our data fδ diminishes, we expect our regularisation method to yield
solutions that improve and approach this true solution.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 31 ∞


---

## 第86页

Beyond Stability: The Need for Convergence
We’ve established that regularisation operators Rα provide stable processing of data for a
fixed parameter α. A regularisation method (Rα, α(δ, fδ)) then uses a rule to select α.
Why This Isn’t the Whole Story
Stability for each fixed α is crucial, but it doesn’t guarantee that our method produces
solutions that are close to the true underlying solution of Ku = f.
As the noise δ in our data fδ diminishes, we expect our regularisation method to yield
solutions that improve and approach this true solution.
This requires that the parameter choice α(δ, fδ) adapts appropriately (e.g., α →0), and
that Rα(δ,fδ) indeed approximates the correct inverse operation in this limit.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 31 ∞


---

## 第87页

Beyond Stability: The Need for Convergence
Why This Isn’t the Whole Story
Stability for each fixed α is crucial, but it doesn’t guarantee that our method produces
solutions that are close to the true underlying solution of Ku = f.
As the noise δ in our data fδ diminishes, we expect our regularisation method to yield
solutions that improve and approach this true solution.
This requires that the parameter choice α(δ, fδ) adapts appropriately (e.g., α →0), and
that Rα(δ,fδ) indeed approximates the correct inverse operation in this limit.
The Next Step: Defining the Target
Before we can formally discuss convergence of a regularisation method to the true solution,
we need to define what this true solution is. This leads us to consider concepts like best
approximate solutions and selection operators.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 31 ∞


---

## 第88页

Selecting Solutions
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 32 ∞


---

## 第89页

Best Approximate Solutions and Selection Operators
Best Approximate Solution
Given an error measure F : V × V →R+ ∪{+∞}, we call ˆu ∈U a best approximate
solution of Ku = f with respect to F if
F(K ˆu, f) ⩽F(Ku, f)
for all u ∈U
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 33 ∞


---

## 第90页

Best Approximate Solutions and Selection Operators
Best Approximate Solution
Given an error measure F : V × V →R+ ∪{+∞}, we call ˆu ∈U a best approximate
solution of Ku = f with respect to F if
F(K ˆu, f) ⩽F(Ku, f)
for all u ∈U
Selection Operator
A multivalued operator S : R(K) ⇒U is called a selection operator if S(Ku) ⊂{u} + N(K)
for all u ∈U.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 33 ∞


---

## 第91页

Best Approximate Solutions and Selection Operators
Best Approximate Solution
Given an error measure F : V × V →R+ ∪{+∞}, we call ˆu ∈U a best approximate
solution of Ku = f with respect to F if
F(K ˆu, f) ⩽F(Ku, f)
for all u ∈U
Selection Operator
A multivalued operator S : R(K) ⇒U is called a selection operator if S(Ku) ⊂{u} + N(K)
for all u ∈U. A best approximate solution ˆu is called prior selected solution if ˆu ∈S(K ˆu).
Often, S(f′) selects solutions from the set of best approximate solutions for data f′ by
minimising a secondary (regularisation) functional.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 33 ∞


---

## 第92页

Examples of Selection Operators
Let K : L2(Ω) →L2(Σ) or similar Hilbert spaces.
Selection via Exact Fit (Minimum Norm)
If F(Ku, f) = χ=0(f −Ku) =

0
Ku = f
∞
else
.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 34 ∞


---

## 第93页

Examples of Selection Operators
Let K : L2(Ω) →L2(Σ) or similar Hilbert spaces.
Selection via Exact Fit (Minimum Norm)
If F(Ku, f) = χ=0(f −Ku) =

0
Ku = f
∞
else
. The best approximate solutions are exact
solutions {u ∈L2(Ω) | Ku = f}.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 34 ∞


---

## 第94页

Examples of Selection Operators
Let K : L2(Ω) →L2(Σ) or similar Hilbert spaces.
Selection via Exact Fit (Minimum Norm)
If F(Ku, f) = χ=0(f −Ku) =

0
Ku = f
∞
else
. The best approximate solutions are exact
solutions {u ∈L2(Ω) | Ku = f}. A selection operator can be defined as
S(f) = arg min
u∈L2(Ω)

∥u∥L2(Ω) | Ku = f

This yields
S(f) =

{K†f}
if f ∈R(K)
∅
if f /∈R(K)
,
where K† is the Moore-Penrose pseudo-inverse of K.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 34 ∞


---

## 第95页

Examples of Selection Operators (continued)
Selection via Least Squares (Minimum Norm)
Let F(Ku, f) = 1
2 ∥Ku −f∥2
L2(Σ).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 35 ∞


---

## 第96页

Examples of Selection Operators (continued)
Selection via Least Squares (Minimum Norm)
Let F(Ku, f) = 1
2 ∥Ku −f∥2
L2(Σ). Best approximate solutions are least-squares solutions,
satisfying K∗Ku = K∗f.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 35 ∞


---

## 第97页

Examples of Selection Operators (continued)
Selection via Least Squares (Minimum Norm)
Let F(Ku, f) = 1
2 ∥Ku −f∥2
L2(Σ). Best approximate solutions are least-squares solutions,
satisfying K∗Ku = K∗f. A selection operator can be defiend as
S(f) = arg min
u∈L2(Ω)

∥u∥L2(Ω) | K∗Ku = K∗f

Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 35 ∞


---

## 第98页

Examples of Selection Operators (continued)
Selection via Least Squares (Minimum Norm)
Let F(Ku, f) = 1
2 ∥Ku −f∥2
L2(Σ). Best approximate solutions are least-squares solutions,
satisfying K∗Ku = K∗f. A selection operator can be defiend as
S(f) = arg min
u∈L2(Ω)

∥u∥L2(Ω) | K∗Ku = K∗f

This yields
S(f) =

{K†f}
if f ∈D(K†) = R(K) ⊕R(K)⊤
∅
if f ∈R(K) \ R(K)
,
where K† is the Moore-Penrose pseudo-inverse of K.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 35 ∞


---

## 第99页

How to Compute Selection Operators
Let K : U →V for Banach spaces U and V.
Selection via Exact Fit (J-minimising solution)
Suppose we choose F(Ku, f) = χ=0(f −Ku) as earlier.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 36 ∞


---

## 第100页

How to Compute Selection Operators
Let K : U →V for Banach spaces U and V.
Selection via Exact Fit (J-minimising solution)
Suppose we choose F(Ku, f) = χ=0(f −Ku) as earlier. The best approximate solutions are
exact solutions {u ∈U | Ku = f}.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 36 ∞


---

## 第101页

How to Compute Selection Operators
Let K : U →V for Banach spaces U and V.
Selection via Exact Fit (J-minimising solution)
Suppose we choose F(Ku, f) = χ=0(f −Ku) as earlier. The best approximate solutions are
exact solutions {u ∈U | Ku = f}. A selection operator can be defined as
S(f) = arg min
u∈U
{J(u) | Ku = f}
for a proper, lower semi-continuous and convex functional J : U →R+ ∪{+∞}.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 36 ∞


---

## 第102页

How to Compute Selection Operators
Let K : U →V for Banach spaces U and V.
Selection via Exact Fit (J-minimising solution)
Suppose we choose F(Ku, f) = χ=0(f −Ku) as earlier. The best approximate solutions are
exact solutions {u ∈U | Ku = f}. A selection operator can be defined as
S(f) = arg min
u∈U
{J(u) | Ku = f}
for a proper, lower semi-continuous and convex functional J : U →R+ ∪{+∞}. In
primal-dual form, the corresponding saddle point problem reads
inf
u∈U sup
v∈V
J(u) + ⟨v, f −Ku⟩.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 36 ∞


---

## 第103页

How to Compute Selection Operators (continued)
Let K : U →V for Banach spaces U and V.
Selection via Exact Fit (J-minimising solution)
In primal-dual form, the corresponding saddle point problem reads
inf
u∈U sup
v∈V
J(u) + ⟨v, f −Ku⟩.
The first-order optimality conditions are
K∗v† ∈∂J(u†)
source condition
Ku† = f
consistency condition
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 37 ∞


---

## 第104页

How to Compute Selection Operators (continued)
Let K : U →V for Banach spaces U and V.
Selection via Exact Fit (J-minimising solution)
In primal-dual form, the corresponding saddle point problem reads
inf
u∈U sup
v∈V
J(u) + ⟨v, f −Ku⟩.
The first-order optimality conditions are
K∗v† ∈∂J(u†)
source condition
Ku† = f
consistency condition
How do we compute u† and v† numerically?
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 37 ∞


---

## 第105页

How to Compute Selection Operators (continued)
Let K : U →V for Banach spaces U and V.
Selection via Exact Fit (J-minimising solution)
In primal-dual form, the corresponding saddle point problem reads
inf
u∈U sup
v∈V
J(u) + ⟨v, f −Ku⟩.
One option: Primal-Dual Hybrid Gradient / Chambolle-Pock algorithm [18, 13, 6, 3, 4], i.e.
uk+1 = proxτJ
 uk + τK∗vk
,
vk+1 = vk −σ
 K(2uk+1 −uk) −f

,
for parameters τ, σ > 0 with τσ < 1/∥K∥2.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 38 ∞


---

## 第106页

How to Compute Selection Operators (continued)
Let K : U →V for Banach spaces U and V.
Selection via Exact Fit (J-minimising solution)
In primal-dual form, the corresponding saddle point problem reads
inf
u∈U sup
v∈V
J(u) + ⟨v, f −Ku⟩.
One option: Primal-Dual Hybrid Gradient / Chambolle-Pock algorithm [18, 13, 6, 3, 4], i.e.
uk+1 = proxτJ
 uk + τK∗vk
,
vk+1 = vk −σ
 K(2uk+1 −uk) −f

,
with τσ < 1/∥K∥2. You will compute J-minimising solutions in your first lab work!
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 38 ∞


---

## 第107页

How to Compute Selection Operators (continued)
Example: Let K = I be the identity operator and J = TV the (isotropic) total variation.
f = u† (ground truth)
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 39 ∞


---

## 第108页

How to Compute Selection Operators (continued)
Example: Let K = I be the identity operator and J = TV the (isotropic) total variation.
f = u† (ground truth)
S(u†)
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 39 ∞


---

## 第109页

How to Compute Selection Operators (continued)
Example: Let K = I be the identity operator and J = TV the (isotropic) total variation.
f = u† (ground truth)
S(u†)
v† (source condition element)
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 39 ∞


---

## 第110页

How to Compute Selection Operators (continued)
Example: Let K = · ∗h be a motion blur operator and J = TV the (isotropic) total variation.
f = u† ∗h (blurred image)
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 40 ∞


---

## 第111页

How to Compute Selection Operators (continued)
Example: Let K = · ∗h be a motion blur operator and J = TV the (isotropic) total variation.
f = u† ∗h (blurred image)
S(f)
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 40 ∞


---

## 第112页

How to Compute Selection Operators (continued)
Example: Let K = · ∗h be a motion blur operator and J = TV the (isotropic) total variation.
f = u† ∗h (blurred image)
S(f)
v† (source condition element)
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 40 ∞


---

## 第113页

Convergent Regularisation Methods
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 41 ∞


---

## 第114页

Convergent Regularisation Method: Definition
Recall: A regularisation method consists of regularisation operators Rα and a parameter
choice strategy αchoice(δ, fδ).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 42 ∞


---

## 第115页

Convergent Regularisation Method: Definition
Recall: A regularisation method consists of regularisation operators Rα and a parameter
choice strategy αchoice(δ, fδ).
Convergent Regularisation Method [1]
A regularisation method is called convergent if for any ”exact” data f (for which a set of
desired solutions S(f) ⊂U is well-defined), any sequence of noise levels δn →0, data fδn
satisfying F(f, fδn) ⩽δn and parameter choice strategy αn = αchoice(δn, fδn) we have
∅̸=

x ∈U,
lim sup
n→∞dU(x, Rαn(fδk)) = 0

⊂S(f)
This means the Kuratowski limit inferior of Rαn(fδn) must be non-empty and contained in the
set of desired solutions S(f).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 42 ∞


---

## 第116页

Convergent Regularisation Method: Definition
Recall: A regularisation method consists of regularisation operators Rα and a parameter
choice strategy αchoice(δ, fδ). If Rα is single-valued, the previous definition simplifies to:
Convergent Regularisation Method (Single-Valued Operators)
A regularisation method is called convergent if for any ”exact” data f (for which a set of
desired solutions S(f) ⊂U is well-defined), any sequence of noise levels δn →0, data fδn
satisfying F(f, fδn) ⩽δn and parameter choice strategy αn = αchoice(δn, fδn) we have
lim
n→∞Rαn(fδn) = u∗
where u∗∈S(f) .
This means that the sequence of solutions Rαn(fδn) must converge to some solution u∗,
and u∗∈S(f).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 43 ∞


---

## 第117页

Example: Tikhonov Regularisation
Let us revisit Tikhonov regularisation. Recall that the filter functions gα(σ) are defined as
gα(σ) =
σ
σ2 + α .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 44 ∞


---

## 第118页

Example: Tikhonov Regularisation
Let us revisit Tikhonov regularisation. Recall that the filter functions gα(σ) are defined as
gα(σ) =
σ
σ2 + α .
The Tikhonov regularisation operator Rα : V →U is then defined as
Rαf =
∞
X
j=1
σj
σ2
j + α

f, vj

V uj .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 44 ∞


---

## 第119页

Example: Tikhonov Regularisation
Let us revisit Tikhonov regularisation. Recall that the filter functions gα(σ) are defined as
gα(σ) =
σ
σ2 + α .
The Tikhonov regularisation operator Rα : V →U is then defined as
Rαf =
∞
X
j=1
σj
σ2
j + α

f, vj

V uj .
Key Property for Fixed α > 0
For any fixed α > 0, the filter function gα(σ) is bounded, i.e.
sup
σ>0

σ
σ2 + α
 =
1
2√α < ∞
Thus, Rα is a bounded (and therefore continuous) linear operator for fixed α > 0.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 44 ∞


---

## 第120页

Tikhonov as a Regularisation Operator
We recall that Tikhonov regularisation fits the definition of a regularisation operator.
Recall: Regularisation Operator (Single-Valued Case)
An operator Rα : V →U is a regularisation operator if it is continuous on V for each fixed α.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 45 ∞


---

## 第121页

Tikhonov as a Regularisation Operator
We recall that Tikhonov regularisation fits the definition of a regularisation operator.
Recall: Regularisation Operator (Single-Valued Case)
An operator Rα : V →U is a regularisation operator if it is continuous on V for each fixed α.
Tikhonov is a Regularisation Operator
For any fixed α > 0, the Tikhonov filter function gα(σ) =
σ
σ2+α is bounded, as
supσ>0 |gα(σ)| =
1
2√α =: Cα < ∞.
This ensures that the operator Rαf = P∞
j=1 gα(σj)

f, vj

V uj is a bounded linear
operator, because ∥Rαf∥U ⩽Cα ∥f∥V.
Bounded linear operators are continuous.
Hence, for each fixed α > 0, Rα is a regularisation operator.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 45 ∞


---

## 第122页

Tikhonov as a Convergent Regularisation Method
Let u† = K†f be the desired (minimum-norm least-squares) solution for exact data
f ∈D(K†). So S(f) = {K†f}.
Recall: Convergent Regularisation Method (Single-Valued)
(Rα, αchoice) is convergent if for noise δn →0 and data fδn (with F(f, fδn) ⩽δn), setting
αn = αchoice(δn, fδn), we have limn→∞Rαn(fδn) = K†f.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 46 ∞


---

## 第123页

Tikhonov as a Convergent Regularisation Method
Let u† = K†f be the desired (minimum-norm least-squares) solution for exact data
f ∈D(K†). So S(f) = {K†f}.
Recall: Convergent Regularisation Method (Single-Valued)
(Rα, αchoice) is convergent if for noise δn →0 and data fδn (with F(f, fδn) ⩽δn), setting
αn = αchoice(δn, fδn), we have limn→∞Rαn(fδn) = K†f.
Convergence of Tikhonov Regularisation
For Tikhonov regularisation, the filter gα(σj) =
σj
σ2
j+α satisfies
limα→0 gα(σj) = limα→0
σj
σ2
j+α =
1
σj for σj > 0. This is a key condition for
approximating K†.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 46 ∞


---

## 第124页

Tikhonov as a Convergent Regularisation Method
Convergence of Tikhonov Regularisation (continued)
Consider an a-priori parameter choice α(δ) that depends on the noise level δ. The error is
Rα(δ)fδ −K†f

U ⩽
Rα(δ)(fδ −f)

U +
Rα(δ)f −K†f

U .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 47 ∞


---

## 第125页

Tikhonov as a Convergent Regularisation Method
Convergence of Tikhonov Regularisation (continued)
Consider an a-priori parameter choice α(δ) that depends on the noise level δ. The error is
Rα(δ)fδ −K†f

U ⩽
Rα(δ)(fδ −f)

U +
Rα(δ)f −K†f

U .
Rα(δ)(fδ −f)

U ⩽
Rα(δ)

L(V,U)
fδ −f

V ⩽
1
2√
α(δ)δ.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 47 ∞


---

## 第126页

Tikhonov as a Convergent Regularisation Method
Convergence of Tikhonov Regularisation (continued)
Consider an a-priori parameter choice α(δ) that depends on the noise level δ. The error is
Rα(δ)fδ −K†f

U ⩽
Rα(δ)(fδ −f)

U +
Rα(δ)f −K†f

U .
Rα(δ)(fδ −f)

U ⩽
Rα(δ)

L(V,U)
fδ −f

V ⩽
1
2√
α(δ)δ.
Rα(δ)f −K†f

U =

P∞
j=1

σj
σ2
j+α(δ) −1
σj
 
f, vj

V uj

U
→0 as α(δ) →0 if
f ∈D(K†).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 47 ∞


---

## 第127页

Tikhonov as a Convergent Regularisation Method
Convergence of Tikhonov Regularisation (continued)
Consider an a-priori parameter choice α(δ) that depends on the noise level δ. The error is
Rα(δ)fδ −K†f

U ⩽
Rα(δ)(fδ −f)

U +
Rα(δ)f −K†f

U .
Rα(δ)(fδ −f)

U ⩽
Rα(δ)

L(V,U)
fδ −f

V ⩽
1
2√
α(δ)δ.
Rα(δ)f −K†f

U =

P∞
j=1

σj
σ2
j+α(δ) −1
σj
 
f, vj

V uj

U
→0 as α(δ) →0 if
f ∈D(K†).
Thus, if α(δ) →0 AND δ/
p
α(δ) →0 as δ →0, then Rα(δ)fδ →K†f. Tikhonov
regularisation with such an α(δ) is a convergent regularisation method.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 47 ∞


---

## 第128页

Variational Regularisation Methods
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 48 ∞


---

## 第129页

Variational Regularisation: A Major Class of Methods
A significant and widely used approach to constructing regularisation operators is through
variational regularisation.
Variational Regularisation Operator
The (potentially set-valued) operator Rα : V ⇒U defined as
Rα(fδ) := arg min
u∈U
{F(Ku, fδ) + Jα(u)}
is said to be a variational regularisation. Here α ∈A are the regularisation parameter(s).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 49 ∞


---

## 第130页

Variational Regularisation: A Major Class of Methods
A significant and widely used approach to constructing regularisation operators is through
variational regularisation.
Variational Regularisation Operator
The (potentially set-valued) operator Rα : V ⇒U defined as
Rα(fδ) := arg min
u∈U
{F(Ku, fδ) + Jα(u)}
is said to be a variational regularisation. Here α ∈A are the regularisation parameter(s).
We have
A data fidelity term, F(Ku, fδ), measuring how well Ku fits the observed data fδ.
A (parameterised) regularisation term (or penalty term), Jα(u), which incorporates
prior knowledge about the desired solution u (e.g., smoothness, sparsity).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 49 ∞


---

## 第131页

Variational Regularisation: Key Theoretical Assumptions
Assumption (Based on [1, Assumption 5.4])
Let U = Z∗for some normed space Z, and let the weak-star topology on U be metrisable on
bounded sets. Moreover assume
K = L∗for a bounded linear operator L : V →Z.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 50 ∞


---

## 第132页

Variational Regularisation: Key Theoretical Assumptions
Assumption (Based on [1, Assumption 5.4])
Let U = Z∗for some normed space Z, and let the weak-star topology on U be metrisable on
bounded sets. Moreover assume
K = L∗for a bounded linear operator L : V →Z.
Jα(·) = H∗
α for some proper functional Hα : Z →R ∪{+∞}, and Jα(·) non-negative.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 50 ∞


---

## 第133页

Variational Regularisation: Key Theoretical Assumptions
Assumption (Based on [1, Assumption 5.4])
Let U = Z∗for some normed space Z, and let the weak-star topology on U be metrisable on
bounded sets. Moreover assume
K = L∗for a bounded linear operator L : V →Z.
Jα(·) = H∗
α for some proper functional Hα : Z →R ∪{+∞}, and Jα(·) non-negative.
F is proper, non-negative, convex functional in first argument, and continuous in second
argument; for every g ∈V there exists u ∈U such that F(Ku, g) + Jα(u) < ∞.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 50 ∞


---

## 第134页

Variational Regularisation: Key Theoretical Assumptions
Assumption (Based on [1, Assumption 5.4])
Let U = Z∗for some normed space Z, and let the weak-star topology on U be metrisable on
bounded sets. Moreover assume
K = L∗for a bounded linear operator L : V →Z.
Jα(·) = H∗
α for some proper functional Hα : Z →R ∪{+∞}, and Jα(·) non-negative.
F is proper, non-negative, convex functional in first argument, and continuous in second
argument; for every g ∈V there exists u ∈U such that F(Ku, g) + Jα(u) < ∞.
For each g ∈V and α ∈A, there exists a constant c = c(a, b, ∥g∥V), which depends
monotonically non-decreasingly on all its arguments, such that
∥u∥U ⩽c
if F(Ku, g) ⩽a and Jα(u) ⩽b.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 50 ∞


---

## 第135页

Variational Regularisation: Operator Stability
Under those assumptions we can guarantee
Well-Posedness of Rα(fδ)
For every fδ ∈V and α ∈A, the set of minimisers Rα(fδ) is non-empty. If
F(Ku, fδ) + J(u, α) is strictly convex, the minimiser is unique.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 51 ∞


---

## 第136页

Variational Regularisation: Operator Stability
Under those assumptions we can guarantee
Well-Posedness of Rα(fδ)
For every fδ ∈V and α ∈A, the set of minimisers Rα(fδ) is non-empty. If
F(Ku, fδ) + J(u, α) is strictly convex, the minimiser is unique.
Stability of Rα(fδ) (Operator Property)
If F is continuous w.r.t. its second variable, and fδn →fδ in V:
Any sequence un ∈Rα(fδn) possesses a (weakly, weak-*, or strongly, depending on
space and functional properties) convergent subsequence unk →u∗.
Crucially, this limit point u∗is itself a minimiser for the limit data: u∗∈Rα(fδ).
This implies ∅̸= {u ∈U | lim supk→∞dU(u, Rα(fδk)) = 0} ⊂Rα(fδ).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 51 ∞


---

## 第137页

Variational Regularisation: Examples
Tikhonov Regularisation (Revisited)
Rα(fδ) = arg min
u∈U
1
2
Ku −fδ2
V + α
2 ∥u∥2
U

,
= (K∗K + αI)−1K∗fδ .
Here F(Ku, fδ) = 1
2
Ku −fδ2
V and Jα(u) = α
2 ∥u∥2
U. This fits the variational framework
and is a convergent method with appropriate αchoice(δ, fδ).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 52 ∞


---

## 第138页

Variational Regularisation: Examples
LASSO (Least Absolute Shrinkage and Selection Operator)
Rα(fδ) = arg min
u∈Rn
1
2
Ku −fδ2
Rm + α ∥u∥ℓ1

Here Jα(u) = α ∥u∥ℓ1 = α Pn
i |ui|. Promotes sparse solutions. Its convergence analysis
relies on convexity and properties of the ℓ1-norm.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 53 ∞


---

## 第139页

Variational Regularisation: Examples
LASSO (Least Absolute Shrinkage and Selection Operator)
Rα(fδ) = arg min
u∈Rn
1
2
Ku −fδ2
Rm + α ∥u∥ℓ1

Here Jα(u) = α ∥u∥ℓ1 = α Pn
i |ui|. Promotes sparse solutions. Its convergence analysis
relies on convexity and properties of the ℓ1-norm.
Total Variation (TV) Regularisation (e.g., for Images with sharp edges)
Rα(fδ) = arg min
u
1
2
Ku −fδ2 + αTV(u)

where TV(u) = supφ∈{ϕ∈C∞
0 (Ω;Rn)|∥ϕ(x)∥2⩽1}
R
Ωu(x) divφ(x) dx.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 53 ∞


---

## 第140页

Convergence Analysis: Error Estimates
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 54 ∞


---

## 第141页

Error Estimates: Motivation for Deeper Analysis
Once we establish that a regularisation method is convergent (e.g., Rαn(fδn) →u∗for
u∗∈S(f) as noise δn →0 in the single-valued cases), further important questions arise:
How good is the approximation? We need a way to measure the error between the
regularised solution Rα(fδ) and the desired true solution u†.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 55 ∞


---

## 第142页

Error Estimates: Motivation for Deeper Analysis
Once we establish that a regularisation method is convergent (e.g., Rαn(fδn) →u∗for
u∗∈S(f) as noise δn →0 in the single-valued cases), further important questions arise:
How good is the approximation? We need a way to measure the error between the
regularised solution Rα(fδ) and the desired true solution u†.
How fast does the error decrease as the noise level δ vanishes? This refers to the
rate of convergence.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 55 ∞


---

## 第143页

Error Estimates: Motivation for Deeper Analysis
Once we establish that a regularisation method is convergent (e.g., Rαn(fδn) →u∗for
u∗∈S(f) as noise δn →0 in the single-valued cases), further important questions arise:
How good is the approximation? We need a way to measure the error between the
regularised solution Rα(fδ) and the desired true solution u†.
How fast does the error decrease as the noise level δ vanishes? This refers to the
rate of convergence.
To address these, we introduce an error measure D : U × U →R+ ∪{+∞} in the solution
space U. This D is not necessarily a norm (e.g., it could be a Bregman distance).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 55 ∞


---

## 第144页

D-Convergence of a Regularisation Method
D-convergent
Let D : U × U →R+ ∪{+∞} be an error measure. Let u† ∈S(f) be a desired inverse
problem solution corresponding to exact data f. A regularisation method (consisting of
operators Rα and parameter choice αchoice(δ, fδ)) is called D-convergent if for any
uδ
α(δ,fδ) ∈Rα(δ,fδ)(fδ) we observe
lim
δ→0 sup

D(uδ
α(δ,fδ), u†) | fδ ∈V, F(f, fδ) ⩽δ

= 0.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 56 ∞


---

## 第145页

D-Convergence of a Regularisation Method
D-convergent
Let D : U × U →R+ ∪{+∞} be an error measure. Let u† ∈S(f) be a desired inverse
problem solution corresponding to exact data f. A regularisation method (consisting of
operators Rα and parameter choice αchoice(δ, fδ)) is called D-convergent if for any
uδ
α(δ,fδ) ∈Rα(δ,fδ)(fδ) we observe
lim
δ→0 sup

D(uδ
α(δ,fδ), u†) | fδ ∈V, F(f, fδ) ⩽δ

= 0.
F(f, fδ) ⩽δ models the noise in the data fδ.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 56 ∞


---

## 第146页

D-Convergence of a Regularisation Method
D-convergent
Let D : U × U →R+ ∪{+∞} be an error measure. Let u† ∈S(f) be a desired inverse
problem solution corresponding to exact data f. A regularisation method (consisting of
operators Rα and parameter choice αchoice(δ, fδ)) is called D-convergent if for any
uδ
α(δ,fδ) ∈Rα(δ,fδ)(fδ) we observe
lim
δ→0 sup

D(uδ
α(δ,fδ), u†) | fδ ∈V, F(f, fδ) ⩽δ

= 0.
F(f, fδ) ⩽δ models the noise in the data fδ.
This means the maximum error (measured by D) between any obtained regularised
solution and the true solution u† vanishes as the noise level δ goes to zero.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 56 ∞


---

## 第147页

Convergence Rates of a Regularisation Method
To discuss specific rates, we often restrict the ”true” solution u† to a smoothness class
Mν ⊂U, where ν > 0 measures the degree of smoothness or regularity.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 57 ∞


---

## 第148页

Convergence Rates of a Regularisation Method
To discuss specific rates, we often restrict the ”true” solution u† to a smoothness class
Mν ⊂U, where ν > 0 measures the degree of smoothness or regularity.
Convergent at Order ν
A regularisation method is called convergent at order ν on a set Mν if, for all f such that
Ku† = f for some u† ∈Mν, there exists a constant Cν > 0 such that for all data fδ
satisfying F(f, fδ) ⩽δ:
sup
u∈Rα(δ,fδ)(fδ)
D(u, u†) ⩽Cνδν
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 57 ∞


---

## 第149页

Convergence Rates of a Regularisation Method
To discuss specific rates, we often restrict the ”true” solution u† to a smoothness class
Mν ⊂U, where ν > 0 measures the degree of smoothness or regularity.
Convergent at Order ν
A regularisation method is called convergent at order ν on a set Mν if, for all f such that
Ku† = f for some u† ∈Mν, there exists a constant Cν > 0 such that for all data fδ
satisfying F(f, fδ) ⩽δ:
sup
u∈Rα(δ,fδ)(fδ)
D(u, u†) ⩽Cνδν
If Rα(δ,fδ)(fδ) is single-valued, this simplifies to D(Rα(δ,fδ)(fδ), u†) ⩽Cνδν.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 57 ∞


---

## 第150页

Convergence Rates of a Regularisation Method
To discuss specific rates, we often restrict the ”true” solution u† to a smoothness class
Mν ⊂U, where ν > 0 measures the degree of smoothness or regularity.
Convergent at Order ν
A regularisation method is called convergent at order ν on a set Mν if, for all f such that
Ku† = f for some u† ∈Mν, there exists a constant Cν > 0 such that for all data fδ
satisfying F(f, fδ) ⩽δ:
sup
u∈Rα(δ,fδ)(fδ)
D(u, u†) ⩽Cνδν
If Rα(δ,fδ)(fδ) is single-valued, this simplifies to D(Rα(δ,fδ)(fδ), u†) ⩽Cνδν.
This provides a quantitative estimate on how fast the error decreases as a function of
the noise level δ, for solutions u† possessing sufficient regularity (i.e., u† ∈Mν).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 57 ∞


---

## 第151页

Tikhonov: Convergence Rate
We use the definition of ”Convergent at Order ν” with error measure D(u, v) = ∥u −v∥U.
Smoothness Assumption (Source Condition)
Assume the true solution u† = K†f possesses a certain ”smoothness”. A common
assumption is u† ∈R((K∗K)µ) for some µ > 0, i.e.,
u† = (K∗K)µw = P∞
j=1 σ2µ
j

w, uj

U uj, for some w ∈U.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 58 ∞


---

## 第152页

Tikhonov: Convergence Rate
We use the definition of ”Convergent at Order ν” with error measure D(u, v) = ∥u −v∥U.
Smoothness Assumption (Source Condition)
Assume the true solution u† = K†f possesses a certain ”smoothness”. A common
assumption is u† ∈R((K∗K)µ) for some µ > 0, i.e.,
u† = (K∗K)µw = P∞
j=1 σ2µ
j

w, uj

U uj, for some w ∈U.
Error Bound and Rate for Tikhonov
Under this smoothness assumption (µ > 0), for uδ
α = Rαfδ (Tikhonov), the error can be
bounded. The two main error components behave as:
Data error propagation:
Rα(fδ −f)

U ≈O(δ/√α).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 58 ∞


---

## 第153页

Tikhonov: Convergence Rate
We use the definition of ”Convergent at Order ν” with error measure D(u, v) = ∥u −v∥U.
Smoothness Assumption (Source Condition)
Assume the true solution u† = K†f possesses a certain ”smoothness”. A common
assumption is u† ∈R((K∗K)µ) for some µ > 0, i.e.,
u† = (K∗K)µw = P∞
j=1 σ2µ
j

w, uj

U uj, for some w ∈U.
Error Bound and Rate for Tikhonov
Under this smoothness assumption (µ > 0), for uδ
α = Rαfδ (Tikhonov), the error can be
bounded. The two main error components behave as:
Data error propagation:
Rα(fδ −f)

U ≈O(δ/√α).
Approximation error:
Rαf −K†f

U ≈O(αµ) for u† ∈R((K∗K)µ).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 58 ∞


---

## 第154页

Tikhonov: Convergence Rate
We use the definition of ”Convergent at Order ν” with error measure D(u, v) = ∥u −v∥U.
Error Bound and Rate for Tikhonov
Under this smoothness assumption (µ > 0), for uδ
α = Rαfδ (Tikhonov), the error can be
bounded. The two main error components behave as:
Data error propagation:
Rα(fδ −f)

U ≈O(δ/√α).
Approximation error:
Rαf −K†f

U ≈O(αµ) for u† ∈R((K∗K)µ).
Balancing these terms by choosing α(δ) ∼δ2/(2µ+1) gives
Rα(δ)fδ −u†
U ⩽Cδ
2µ
2µ+1 .
Hence, Rα(δ) is convergent at order ν = 2µ/(2µ + 1) for solutions u† ∈R((K∗K)µ),
µ ∈(0, 1]. For µ = 1, the rate is O(δ2/3). If u† ∈R(K∗) (i.e. µ = 1/2), the rate is O(
√
δ).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 58 ∞


---

## 第155页

Beyond Norms: Bregman Distances for Error Measurement
The error measure D(u, v) in convergence analysis need not be a norm. Bregman
distances [2, 11] offer powerful alternatives for analysing variational regularisations.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 59 ∞


---

## 第156页

Beyond Norms: Bregman Distances for Error Measurement
The error measure D(u, v) in convergence analysis need not be a norm. Bregman
distances [2, 11] offer powerful alternatives for analysing variational regularisations.
Bregman Distance
Let J : U →R ∪{+∞} be a proper, convex and lower semi-continuous functional. For
u1, u2 ∈U and p2 ∈∂J(u2) (a subgradient of J at u2), the Bregman distance is defined as
Dp2
J (u1, u2) = J(u1) −J(u2) −⟨p2, u1 −u2⟩.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 59 ∞


---

## 第157页

Beyond Norms: Bregman Distances for Error Measurement
The error measure D(u, v) in convergence analysis need not be a norm. Bregman
distances [2, 11] offer powerful alternatives for analysing variational regularisations.
Bregman Distance
Let J : U →R ∪{+∞} be a proper, convex and lower semi-continuous functional. For
u1, u2 ∈U and p2 ∈∂J(u2) (a subgradient of J at u2), the Bregman distance is defined as
Dp2
J (u1, u2) = J(u1) −J(u2) −⟨p2, u1 −u2⟩.
Key Properties:
Non-negative: Dp2
J (u1, u2) ⩾0
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 59 ∞


---

## 第158页

Beyond Norms: Bregman Distances for Error Measurement
The error measure D(u, v) in convergence analysis need not be a norm. Bregman
distances [2, 11] offer powerful alternatives for analysing variational regularisations.
Bregman Distance
Let J : U →R ∪{+∞} be a proper, convex and lower semi-continuous functional. For
u1, u2 ∈U and p2 ∈∂J(u2) (a subgradient of J at u2), the Bregman distance is defined as
Dp2
J (u1, u2) = J(u1) −J(u2) −⟨p2, u1 −u2⟩.
Key Properties:
Non-negative: Dp2
J (u1, u2) ⩾0
Generally non-symmetric: Dp2
J (u1, u2) ̸= Dp1
J (u2, u1)
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 59 ∞


---

## 第159页

Beyond Norms: Bregman Distances for Error Measurement
The error measure D(u, v) in convergence analysis need not be a norm. Bregman
distances [2, 11] offer powerful alternatives for analysing variational regularisations.
Bregman Distance
Let J : U →R ∪{+∞} be a proper, convex and lower semi-continuous functional. For
u1, u2 ∈U and p2 ∈∂J(u2) (a subgradient of J at u2), the Bregman distance is defined as
Dp2
J (u1, u2) = J(u1) −J(u2) −⟨p2, u1 −u2⟩.
Key Properties:
Non-negative: Dp2
J (u1, u2) ⩾0
Generally non-symmetric: Dp2
J (u1, u2) ̸= Dp1
J (u2, u1)
Symmetric special case: If J(u) = 1
2 ∥u∥2, then Du2
J (u1, u2) = 1
2 ∥u1 −u2∥2
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 59 ∞


---

## 第160页

Visualising Bregman Distance: J(u) = u log(u) −u
u
J(u)
(u2, J(u2))
(u1, J(u1))
DJ
J(u) = u log(u) −u
Tangent at u2
u2
u1
Interpretation
Function: Convex for u > 0
Derivative: J′(u) = log(u)
Subgradient: p2 = log(u2) at point u2
Bregman Distance
Dlog(u2)
J
(u1, u2) = u1 log
u1
u2

+ u2 −u1
This is the Kullback-Leibler divergence
between u1 and u2.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 60 ∞


---

## 第161页

Symmetrised Bregman Distances
To obtain a symmetric measure, one can define the symmetrised Bregman distance:
Symmetrised Bregman Distance
Given p ∈∂J(u) and q ∈∂J(v) we define
Dsymm
J
(u, v) = Dp
J (v, u) + Dq
J (u, v)
= J(v) −J(u) −⟨p, v −u⟩+ J(u) −J(v) −⟨q, u −v⟩
= ⟨q −p, v −u⟩⩾0 .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 61 ∞


---

## 第162页

Symmetrised Bregman Distances
To obtain a symmetric measure, one can define the symmetrised Bregman distance:
Symmetrised Bregman Distance
Given p ∈∂J(u) and q ∈∂J(v) we define
Dsymm
J
(u, v) = Dp
J (v, u) + Dq
J (u, v)
= J(v) −J(u) −⟨p, v −u⟩+ J(u) −J(v) −⟨q, u −v⟩
= ⟨q −p, v −u⟩⩾0 .
Relevance
Symmetrised Bregman distances naturally appear in error estimates for variational
regularisation methods, particularly when source conditions are involved.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 61 ∞


---

## 第163页

Recap: Error Estimates & Bregman Distances
We are interested in error estimates for variational regularisation with scalar parameter, i.e.
Rα(fδ) = uδ
α ∈arg min
u∈U
1
2
Ku −fδ2
V + αJ(u)

Let u† be the true solution with exact data f = Ku†.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 62 ∞


---

## 第164页

Recap: Error Estimates & Bregman Distances
We are interested in error estimates for variational regularisation with scalar parameter, i.e.
Rα(fδ) = uδ
α ∈arg min
u∈U
1
2
Ku −fδ2
V + αJ(u)

Let u† be the true solution with exact data f = Ku†.
Key Ingredients
Optimality Condition for uδ
α: there exists pα ∈∂J(uδ
α) such that
K∗(Kuδ
α −fδ) + αpα = 0 .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 62 ∞


---

## 第165页

Recap: Error Estimates & Bregman Distances
We are interested in error estimates for variational regularisation with scalar parameter, i.e.
Rα(fδ) = uδ
α ∈arg min
u∈U
1
2
Ku −fδ2
V + αJ(u)

Let u† be the true solution with exact data f = Ku†.
Key Ingredients
Optimality Condition for uδ
α: there exists pα ∈∂J(uδ
α) such that
K∗(Kuδ
α −fδ) + αpα = 0 .
Source Condition for u† (assumed to hold for u†): there exists v ∈V such that
K∗v ∈∂J(u†) .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 62 ∞


---

## 第166页

Deriving Error Estimates: Step 1 (Main Equation)
1 From the optimality condition we have K∗(Kuδ
α −fδ) + αpα = 0.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 63 ∞


---

## 第167页

Deriving Error Estimates: Step 1 (Main Equation)
1 From the optimality condition we have K∗(Kuδ
α −fδ) + αpα = 0.
2 Subtracting the source condition element v yields
K∗(Kuδ
α −fδ) + α(pα −K∗v) = −αK∗v
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 63 ∞


---

## 第168页

Deriving Error Estimates: Step 1 (Main Equation)
1 From the optimality condition we have K∗(Kuδ
α −fδ) + αpα = 0.
2 Subtracting the source condition element v yields
K∗(Kuδ
α −fδ) + α(pα −K∗v) = −αK∗v
3 Take the dual product with uδ
α −u† leads to
D
Kuδ
α −fδ, K(uδ
α −u†)
E
V + α
D
pα −K∗v, uδ
α −u†E
U = −α
D
K∗v, uδ
α −u†E
U
= −α
D
v, K(uδ
α −u†)
E
V
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 63 ∞


---

## 第169页

Deriving Error Estimates: Step 1 (Main Equation)
1 From the optimality condition we have K∗(Kuδ
α −fδ) + αpα = 0.
2 Subtracting the source condition element v yields
K∗(Kuδ
α −fδ) + α(pα −K∗v) = −αK∗v
3 Take the dual product with uδ
α −u† leads to
D
Kuδ
α −fδ, K(uδ
α −u†)
E
V + α
D
pα −K∗v, uδ
α −u†E
U = −α
D
K∗v, uδ
α −u†E
U
= −α
D
v, K(uδ
α −u†)
E
V
4 Recognising that

pα −K∗v, uδ
α −u†
U = Dsymm
J
(uδ
α, u†), where the specific
subgradients pα ∈∂J(uδ
α) and K∗v ∈∂J(u†) are used
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 63 ∞


---

## 第170页

Deriving Error Estimates: Step 1 (Main Equation)
1 From the optimality condition we have K∗(Kuδ
α −fδ) + αpα = 0.
2 Subtracting the source condition element v yields
K∗(Kuδ
α −fδ) + α(pα −K∗v) = −αK∗v
3 Take the dual product with uδ
α −u† leads to
D
Kuδ
α −fδ, K(uδ
α −u†)
E
V + α
D
pα −K∗v, uδ
α −u†E
U = −α
D
K∗v, uδ
α −u†E
U
= −α
D
v, K(uδ
α −u†)
E
V
4 Recognising that

pα −K∗v, uδ
α −u†
U = Dsymm
J
(uδ
α, u†), where the specific
subgradients pα ∈∂J(uδ
α) and K∗v ∈∂J(u†) are used
5 Hence, with f = Ku†, the previous equation becomes

Kuδ
α −fδ, Kuδ
α −f

V + αDSymm
J
(uδ
α, u†) = α

v, f −Kuδ
α

V
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 63 ∞


---

## 第171页

Deriving Error Estimates: Step 2 (Using Identities)
From the previous slide we have the equation

Kuδ
α −fδ, Kuδ
α −f

V + αDsymm
J
(uδ
α, u†) = α

v, f −Kuδ
α

V
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 64 ∞


---

## 第172页

Deriving Error Estimates: Step 2 (Using Identities)
From the previous slide we have the equation

Kuδ
α −fδ, Kuδ
α −f

V + αDsymm
J
(uδ
α, u†) = α

v, f −Kuδ
α

V
We use two standard algebraic identities for inner products:
1 ⟨a −b, a −c⟩V = 1
2 ∥a −c∥2
V + 1
2 ∥a −b∥2
V −1
2 ∥c −b∥2
V. Applying this to the first
term with a = Kuδ
α, b = fδ, c = f yields

Kuδ
α −fδ, Kuδ
α −f

V = 1
2
Kuδ
α −f
2
V + 1
2
Kuδ
α −fδ2
V −1
2
f −fδ2
V .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 64 ∞


---

## 第173页

Deriving Error Estimates: Step 2 (Using Identities)
From the previous slide we have the equation

Kuδ
α −fδ, Kuδ
α −f

V + αDsymm
J
(uδ
α, u†) = α

v, f −Kuδ
α

V
We use two standard algebraic identities for inner products:
1 ⟨a −b, a −c⟩V = 1
2 ∥a −c∥2
V + 1
2 ∥a −b∥2
V −1
2 ∥c −b∥2
V. Applying this to the first
term with a = Kuδ
α, b = fδ, c = f yields

Kuδ
α −fδ, Kuδ
α −f

V = 1
2
Kuδ
α −f
2
V + 1
2
Kuδ
α −fδ2
V −1
2
f −fδ2
V .
2 ⟨x, y⟩V = 1
2 ∥x∥2
V + 1
2 ∥y∥2
V −1
2 ∥x −y∥2
V. Applying this to the right-hand-side yields
α

v, f −Kuδ
α

V = α2
2 ∥v∥2
V + 1
2
Kuδ
α −f
2
V −1
2
Kuδ
α −f + αv
2
V .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 64 ∞


---

## 第174页

Deriving Error Estimates: Step 3 (Combining)
Substituting the identities into the main equation from Step 1 leads to
1
2
Kuδ
α −f
2
V + 1
2
Kuδ
α −fδ2
V −1
2
f −fδ2
V

+ αDsymm
J
(uδ
α, u†)
=
α2
2 ∥v∥2
V + 1
2
Kuδ
α −f
2
V −1
2
Kuδ
α −f + αv
2
V

Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 65 ∞


---

## 第175页

Deriving Error Estimates: Step 3 (Combining)
Substituting the identities into the main equation from Step 1 leads to
1
2
Kuδ
α −f
2
V + 1
2
Kuδ
α −fδ2
V −1
2
f −fδ2
V

+ αDsymm
J
(uδ
α, u†)
=
α2
2 ∥v∥2
V + 1
2
Kuδ
α −f
2
V −1
2
Kuδ
α −f + αv
2
V

The term 1
2
Kuδ
α −f
2
V cancels on both sides. Rearranging leaves us with
1
2
Kuδ
α −f + αv
2
V + 1
2
Kuδ
α −fδ2
V + αDsymm
J
(uδ
α, u†) = 1
2
f −fδ2
V + α2
2 ∥v∥2
V
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 65 ∞


---

## 第176页

Deriving Error Estimates: Step 4 (Bounding)
The Error Bound
Since the first two terms on the left-hand-side are non-negative, we observe
αDsymm
J
(uδ
α, u†) ⩽1
2
f −fδ2
V + α2
2 ∥v∥2
V
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 66 ∞


---

## 第177页

Deriving Error Estimates: Step 4 (Bounding)
The Error Bound
Since the first two terms on the left-hand-side are non-negative, we observe
αDsymm
J
(uδ
α, u†) ⩽1
2
f −fδ2
V + α2
2 ∥v∥2
V
With the noise estimate
f −fδ
V ⩽δ we further obtain
Dsymm
J
(uδ
α, u†) ⩽δ2
2α + α
2 ∥v∥2
V
This is a common form of error estimate for variational regularisation (for quadratic fidelity
terms).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 66 ∞


---

## 第178页

D-Convergence and Rates from Bregman Estimate
We have derived Dsymm
J
(uδ
α, u†) ⩽δ2
2α + α
2 ∥v∥2
V. Using D(·, ·) = Dsymm
J
(·, ·) as our error
measure, we can achieve D-convergence.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 67 ∞


---

## 第179页

D-Convergence and Rates from Bregman Estimate
We have derived Dsymm
J
(uδ
α, u†) ⩽δ2
2α + α
2 ∥v∥2
V. Using D(·, ·) = Dsymm
J
(·, ·) as our error
measure, we can achieve D-convergence.
Achieving D-Convergence and Rates
D-Convergence: If the parameter choice strategy ensures α(δ) →0 and δ2/α(δ) →0
as δ →0, then the right-hand-side converges to zero. This implies
Dsymm
J
(uδ
α(δ), u†) →0, making the method D-convergent.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 67 ∞


---

## 第180页

D-Convergence and Rates from Bregman Estimate
We have derived Dsymm
J
(uδ
α, u†) ⩽δ2
2α + α
2 ∥v∥2
V. Using D(·, ·) = Dsymm
J
(·, ·) as our error
measure, we can achieve D-convergence.
Achieving D-Convergence and Rates
D-Convergence: If the parameter choice strategy ensures α(δ) →0 and δ2/α(δ) →0
as δ →0, then the right-hand-side converges to zero. This implies
Dsymm
J
(uδ
α(δ), u†) →0, making the method D-convergent.
Convergence Rate (Example: Order ν = 1): To optimise the bound, choose
α(δ) = α(δ) =
δ
∥v∥V , which yields
Dsymm
J
(uδ
α(δ), u†) ⩽∥v∥V δ
This shows convergence at order ν = 1 with rate constant C1 = ∥v∥V.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 67 ∞


---

## 第181页

Iterative Regularisation Methods
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 68 ∞


---

## 第182页

Iterative Regularisation: Motivation
We’ve explored variational methods of the form:
uδ
α ∈Rα(fδ) = arg min
u∈U

F(Ku, fδ) + αJ(u)
	
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 69 ∞


---

## 第183页

Iterative Regularisation: Motivation
We’ve explored variational methods of the form:
uδ
α ∈Rα(fδ) = arg min
u∈U

F(Ku, fδ) + αJ(u)
	
How do we compute these regularised solutions uδ
α? Often by using iterative
optimisation algorithms.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 69 ∞


---

## 第184页

Iterative Regularisation: Motivation
We’ve explored variational methods of the form:
uδ
α ∈Rα(fδ) = arg min
u∈U

F(Ku, fδ) + αJ(u)
	
How do we compute these regularised solutions uδ
α? Often by using iterative
optimisation algorithms.
Question: Can an iterative algorithm itself, when applied to fδ and stopped early, act
as a regularisation method?
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 69 ∞


---

## 第185页

Iterative Regularisation: Motivation
We’ve explored variational methods of the form:
uδ
α ∈Rα(fδ) = arg min
u∈U

F(Ku, fδ) + αJ(u)
	
How do we compute these regularised solutions uδ
α? Often by using iterative
optimisation algorithms.
Question: Can an iterative algorithm itself, when applied to fδ and stopped early, act
as a regularisation method?
Answer: Yes! This is the core idea of iterative regularisation. The number of iterations
k∗becomes the regularisation parameter.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 69 ∞


---

## 第186页

Iterative Regularisation: Motivation
We’ve explored variational methods of the form:
uδ
α ∈Rα(fδ) = arg min
u∈U

F(Ku, fδ) + αJ(u)
	
How do we compute these regularised solutions uδ
α? Often by using iterative
optimisation algorithms.
Question: Can an iterative algorithm itself, when applied to fδ and stopped early, act
as a regularisation method?
Answer: Yes! This is the core idea of iterative regularisation. The number of iterations
k∗becomes the regularisation parameter.
Example: PDHG for J-Minimising Solutions with Noisy Data
We’ll examine the PDHG algorithm for finding a J-minimising solution to Ku = f, even when
we only have noisy data fδ (cf. [12]).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 69 ∞


---

## 第187页

PDHG for J-Minimising Solutions with Noisy Data
Recall the problem of finding a J-minimising solution:
inf
u∈U J(u)
subject to
Ku = f
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 70 ∞


---

## 第188页

PDHG for J-Minimising Solutions with Noisy Data
Recall the problem of finding a J-minimising solution:
inf
u∈U J(u)
subject to
Ku = f
Let u† ∈U be such a solution, and v† ∈V be a corresponding dual variable (source
condition element) such that K∗v† ∈∂J(u†) and Ku† = f.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 70 ∞


---

## 第189页

PDHG for J-Minimising Solutions with Noisy Data
Recall the problem of finding a J-minimising solution:
inf
u∈U J(u)
subject to
Ku = f
Let u† ∈U be such a solution, and v† ∈V be a corresponding dual variable (source
condition element) such that K∗v† ∈∂J(u†) and Ku† = f.
Algorithm with Noisy Data fδ
The PDHG algorithm for this problem, using noisy data fδ where
f −fδ
V ⩽δ reads
uk+1 = proxτJ
 uk + τK∗vk
vk+1 = vk −σ
 K(2uk+1 −uk) −fδ
with τ, σ > 0 such that τσ < 1/ ∥K∥2. We assume u0 = 0, v0 = 0.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 70 ∞


---

## 第190页

PDHG for J-Minimising Solutions with Noisy Data
Algorithm with Noisy Data fδ
The PDHG algorithm for this problem, using noisy data fδ where
f −fδ
V ⩽δ reads
uk+1 = proxτJ
 uk + τK∗vk
vk+1 = vk −σ
 K(2uk+1 −uk) −fδ
with τ, σ > 0 such that τσ < 1/ ∥K∥2. We assume u0 = 0, v0 = 0.
Goal: Show that producing uk (e.g., Ces`aro mean of uj for j = 1, . . . , k) with k = k∗(δ) (an
early stopping rule) makes this a convergent regularisation method.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 70 ∞


---

## 第191页

Convergence Analysis: Key Ingredients
Let wk = (uk, vk) and w† = (u†, v†). The analysis relies on properties of the algorithm and
saddle-point conditions.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 71 ∞


---

## 第192页

Convergence Analysis: Key Ingredients
Let wk = (uk, vk) and w† = (u†, v†). The analysis relies on properties of the algorithm and
saddle-point conditions.
Define a symmetric operator M : U × V →U × V and its associated M-norm
M :=
 1
τI
−K∗
−K
1
σI

,
∥w∥2
M := ⟨Mw, w⟩.
Note: for τσ < 1/ ∥K∥2, M is (symmetric) positive definite.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 71 ∞


---

## 第193页

Convergence Analysis: Key Ingredients
Let wk = (uk, vk) and w† = (u†, v†). The analysis relies on properties of the algorithm and
saddle-point conditions.
Define a symmetric operator M : U × V →U × V and its associated M-norm
M :=
 1
τI
−K∗
−K
1
σI

,
∥w∥2
M := ⟨Mw, w⟩.
Note: for τσ < 1/ ∥K∥2, M is (symmetric) positive definite.
Then, the updates of the PDHG algorithm satisfy the equivalent system of optimality
conditions

0
0

∈
∂J(uk) −K∗vk
Kuk −fδ

+ M(wk −wk−1) .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 71 ∞


---

## 第194页

Convergence Analysis: Key Ingredients
Let wk = (uk, vk) and w† = (u†, v†). The analysis relies on properties of the algorithm and
saddle-point conditions.
Then, the updates of the PDHG algorithm satisfy the equivalent system of optimality
conditions

0
0

∈
∂J(uk) −K∗vk
Kuk −fδ

+ M(wk −wk−1) .
The J-minimising solution u† and corresponding source condition element v† satisfy

0
0

∈
∂J(u†) −K∗v†
Ku† −f

.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 72 ∞


---

## 第195页

Convergence Analysis: Key Ingredients
Subtracting one condition from the other yields

0
0

∈
∂J(uk) −∂J(u†) −K∗(vk −v†)
Kuk −fδ + f −Ku†

+ M(wk −wk−1) .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 73 ∞


---

## 第196页

Convergence Analysis: Key Ingredients
Subtracting one condition from the other yields

0
0

∈
∂J(uk) −∂J(u†) −K∗(vk −v†)
Kuk −fδ + f −Ku†

+ M(wk −wk−1) .
Taking the dual product with wk −w†, we obtain
0 = Dsymm
J
(uk, u†) + ⟨f −fδ, vk −v†⟩+ ⟨M(wk −wk−1), wk −w†⟩,
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 73 ∞


---

## 第197页

Convergence Analysis: Key Ingredients
Subtracting one condition from the other yields

0
0

∈
∂J(uk) −∂J(u†) −K∗(vk −v†)
Kuk −fδ + f −Ku†

+ M(wk −wk−1) .
Taking the dual product with wk −w†, we obtain
0 = Dsymm
J
(uk, u†) + ⟨f −fδ, vk −v†⟩+ ⟨M(wk −wk−1), wk −w†⟩,
respectively
0 ⩽Dsymm
J
(uk, u†) = ⟨M(wk−1 −wk), wk −w†⟩+ ⟨fδ −f, vk −v†⟩.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 73 ∞


---

## 第198页

Convergence Analysis: Key Ingredients
Hence, if we define ˜w := M−1

0
fδ

, we estimate
0 ⩽
D
M
 wk−1 −wk
, wk −w†E
+ ⟨fδ −f, vk −v†⟩,
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 74 ∞


---

## 第199页

Convergence Analysis: Key Ingredients
Hence, if we define ˜w := M−1

0
fδ

, we estimate
0 ⩽
D
M
 wk−1 −wk
, wk −w†E
+ ⟨fδ −f, vk −v†⟩,
=
D
M
 wk−1 −wk
, wk −w†E
+

0
fδ

−

0
f

, wk −w†

,
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 74 ∞


---

## 第200页

Convergence Analysis: Key Ingredients
Hence, if we define ˜w := M−1

0
fδ

, we estimate
0 ⩽
D
M
 wk−1 −wk
, wk −w†E
+ ⟨fδ −f, vk −v†⟩,
=
D
M
 wk−1 −wk
, wk −w†E
+

0
fδ

−

0
f

, wk −w†

,
=
D
M
 wk−1 −wk
, wk −w†E
+

M

M−1

0
fδ

−

0
f

, wk −w†

,
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 74 ∞


---

## 第201页

Convergence Analysis: Key Ingredients
Hence, if we define ˜w := M−1

0
fδ

, we estimate
0 ⩽
D
M
 wk−1 −wk
, wk −w†E
+ ⟨fδ −f, vk −v†⟩,
=
D
M
 wk−1 −wk
, wk −w†E
+

0
fδ

−

0
f

, wk −w†

,
=
D
M
 wk−1 −wk
, wk −w†E
+

M

M−1

0
fδ

−

0
f

, wk −w†

,
⩽
D
M
 wk−1 −wk
, wk −w†E
+
M−1

0
fδ

−

0
f

M
wk −w†
M ,
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 74 ∞


---

## 第202页

Convergence Analysis: Key Ingredients
Hence, if we define ˜w := M−1

0
fδ

, we estimate
0 ⩽
D
M
 wk−1 −wk
, wk −w†E
+ ⟨fδ −f, vk −v†⟩,
=
D
M
 wk−1 −wk
, wk −w†E
+

0
fδ

−

0
f

, wk −w†

,
=
D
M
 wk−1 −wk
, wk −w†E
+

M

M−1

0
fδ

−

0
f

, wk −w†

,
⩽
D
M
 wk−1 −wk
, wk −w†E
+
M−1

0
fδ

−

0
f

M
wk −w†
M ,
⩽
D
M
 wk−1 −wk
, wk −w†E
+
δ
p
∥M∥
wk −w†
M .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 74 ∞


---

## 第203页

Convergence Analysis: Key Ingredients
This means we have verified
D
M
 wk −wk−1
, wk −w†E
⩽
δ
p
∥M∥
wk −w†
M .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 75 ∞


---

## 第204页

Convergence Analysis: Key Ingredients
This means we have verified
D
M
 wk −wk−1
, wk −w†E
⩽
δ
p
∥M∥
wk −w†
M .
M-Norm Estimate
For a given positive definite operator M and corresponding inner product and norm, the
inequality ⟨M(wk −wk−1), wk −w†⟩⩽
δ
√
∥M∥∥wk −w†∥M implies
∥wk −w†∥M ⩽∥wk−1 −w†∥M + δ/
p
∥M∥.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 75 ∞


---

## 第205页

Convergence Analysis: Key Ingredients
This means we have verified
D
M
 wk −wk−1
, wk −w†E
⩽
δ
p
∥M∥
wk −w†
M .
M-Norm Estimate
For a given positive definite operator M and corresponding inner product and norm, the
inequality ⟨M(wk −wk−1), wk −w†⟩⩽
δ
√
∥M∥∥wk −w†∥M implies
∥wk −w†∥M ⩽∥wk−1 −w†∥M + δ/
p
∥M∥.
Method of Induction
∥wk −w†∥M ⩽∥w†∥M + k δ/
p
∥M∥.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 75 ∞


---

## 第206页

Convergence Analysis: Summing the Estimates
Together with the three-point identity
D
M
 wk−1 −wk
, wk −w†E
= 1
2∥wk−1 −w†∥2
M −1
2∥wk −w†∥2
M −1
2∥wk−1 −wk∥2
M ,
we estimate with prevous result
Dsymm
J
(uk, u†) + 1
2∥wk −wk−1∥2
M
⩽1
2∥wk−1 −w†∥2
M −1
2∥wk −w†∥2
M + δ∥w†∥M
p
∥M∥
+ kδ2
∥M∥.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 76 ∞


---

## 第207页

Convergence Analysis: Summing the Estimates
Together with the three-point identity
D
M
 wk−1 −wk
, wk −w†E
= 1
2∥wk−1 −w†∥2
M −1
2∥wk −w†∥2
M −1
2∥wk−1 −wk∥2
M ,
we estimate with prevous result
Dsymm
J
(uk, u†) + 1
2∥wk −wk−1∥2
M
⩽1
2∥wk−1 −w†∥2
M −1
2∥wk −w†∥2
M + δ∥w†∥M
p
∥M∥
+ kδ2
∥M∥.
Summing up from 1 to k then yields
k
X
j=1

Dsymm
J
(uj, u†) + 1
2∥wj −wj−1∥2
M

⩽1
2∥w†∥2
M + kδ∥w†∥M
p
∥M∥
+ k(k + 1)δ2
2∥M∥
.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 76 ∞


---

## 第208页

Convergence Rate via Early Stopping
Let uk = 1
k
Pk
j=1 uj be the Cesaro mean of the primal iterates. Since J is convex,
Dsymm
J
(·, u†) is also convex in its first argument. By Jensen’s inequality:
Dsymm
J
(uk, u†) ⩽1
k
k
X
j=1
Dsymm
J
(uj, u†)
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 77 ∞


---

## 第209页

Convergence Rate via Early Stopping
Let uk = 1
k
Pk
j=1 uj be the Cesaro mean of the primal iterates. Since J is convex,
Dsymm
J
(·, u†) is also convex in its first argument. By Jensen’s inequality:
Dsymm
J
(uk, u†) ⩽1
k
k
X
j=1
Dsymm
J
(uj, u†)
Substituting the bound from the previous slide leaves us with
kDsymm
J
(uk, u†) ⩽1
2∥w†∥2
M + kδ∥w†∥M
p
∥M∥
+ k(k + 1)δ2
2∥M∥
,
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 77 ∞


---

## 第210页

Convergence Rate via Early Stopping
Let uk = 1
k
Pk
j=1 uj be the Cesaro mean of the primal iterates. Since J is convex,
Dsymm
J
(·, u†) is also convex in its first argument. By Jensen’s inequality:
Dsymm
J
(uk, u†) ⩽1
k
k
X
j=1
Dsymm
J
(uj, u†)
Substituting the bound from the previous slide leaves us with
kDsymm
J
(uk, u†) ⩽1
2∥w†∥2
M + kδ∥w†∥M
p
∥M∥
+ k(k + 1)δ2
2∥M∥
,
and dividing by k yields
Dsymm
J
(uk, u†) ⩽∥w†∥2
M
2k
+ δ∥w†∥M
p
∥M∥
+ (k + 1)δ2
2∥M∥
.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 77 ∞


---

## 第211页

Order of Convergence for PDHG
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 78 ∞


---

## 第212页

Order of Convergence for PDHG
Stopping Rule (Regularisation Parameter Choice)
Choose number of iterations k = k∗(δ) such that k∗(δ) ∼1/δ, e.g. k∗(δ) =
√
∥M∥∥w†∥M
δ

.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 78 ∞


---

## 第213页

Order of Convergence for PDHG
Stopping Rule (Regularisation Parameter Choice)
Choose number of iterations k = k∗(δ) such that k∗(δ) ∼1/δ, e.g. k∗(δ) =
√
∥M∥∥w†∥M
δ

.
Convergence Result
With an appropriate stopping rule k∗(δ) ∼1/δ, the Condat-Vu algorithm is D-convergent
with ν = 1 with respect to the symmetrised Bregman distance, i.e.
Dsymm
J
(uk∗(δ), u†) = O(δ) .
This means the iterative scheme for computing the selection operator, with fδ and an early
stopping rule k∗(δ), is a convergent regularisation method.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 78 ∞


---

## 第214页

Iterative Regularisation: Summary
Iterative algorithms used to compute selection operators can themselves be
regularisers if stopped appropriately.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 79 ∞


---

## 第215页

Iterative Regularisation: Summary
Iterative algorithms used to compute selection operators can themselves be
regularisers if stopped appropriately.
The number of iterations k acts as the regularisation parameter.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 79 ∞


---

## 第216页

Iterative Regularisation: Summary
Iterative algorithms used to compute selection operators can themselves be
regularisers if stopped appropriately.
The number of iterations k acts as the regularisation parameter.
We showed for the PDHG algorithm (for inf J(u) s.t. Ku = f):
Using noisy data fδ.
Stopping at k∗(δ) ∼1/δ iterations.
Yields a solution uk∗(δ) (e.g., Cesaro mean).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 79 ∞


---

## 第217页

Iterative Regularisation: Summary
Iterative algorithms used to compute selection operators can themselves be
regularisers if stopped appropriately.
The number of iterations k acts as the regularisation parameter.
We showed for the PDHG algorithm (for inf J(u) s.t. Ku = f):
Using noisy data fδ.
Stopping at k∗(δ) ∼1/δ iterations.
Yields a solution uk∗(δ) (e.g., Cesaro mean).
This constitutes a convergent regularisation method with rate O(δ) for
Dsymm
J
(uk∗(δ), u†).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 79 ∞


---

## 第218页

Iterative Regularisation: Summary
Iterative algorithms used to compute selection operators can themselves be
regularisers if stopped appropriately.
The number of iterations k acts as the regularisation parameter.
We showed for the PDHG algorithm (for inf J(u) s.t. Ku = f):
Using noisy data fδ.
Stopping at k∗(δ) ∼1/δ iterations.
Yields a solution uk∗(δ) (e.g., Cesaro mean).
This constitutes a convergent regularisation method with rate O(δ) for
Dsymm
J
(uk∗(δ), u†).
This links iterative optimisation directly to regularisation theory. Many other iterative
methods (Landweber, CG, etc.) also exhibit regularising properties.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 79 ∞


---

## 第219页

Data-Driven Regularisation: Spectral Methods
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 80 ∞


---

## 第220页

Data-Driven Spectral Regularisation: Motivation
We’ve seen spectral regularisation methods Rαf = P∞
j=1 gα(σj)

f, vj

V uj, where the filter
function gα(σj) is pre-defined (e.g., Tikhonov, TSVD).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 81 ∞


---

## 第221页

Data-Driven Spectral Regularisation: Motivation
We’ve seen spectral regularisation methods Rαf = P∞
j=1 gα(σj)

f, vj

V uj, where the filter
function gα(σj) is pre-defined (e.g., Tikhonov, TSVD).
A New Perspective: Learning the Filter
What if we could learn an optimal filter function directly from data?
Suppose we have a training dataset of true solutions u and corresponding noisy
measurements fδ = Ku + noise.
We can aim to find a filter g(σj) that minimises the expected reconstruction error over
this dataset.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 81 ∞


---

## 第222页

Data-Driven Spectral Regularisation: Motivation
We’ve seen spectral regularisation methods Rαf = P∞
j=1 gα(σj)

f, vj

V uj, where the filter
function gα(σj) is pre-defined (e.g., Tikhonov, TSVD).
A New Perspective: Learning the Filter
What if we could learn an optimal filter function directly from data?
Suppose we have a training dataset of true solutions u and corresponding noisy
measurements fδ = Ku + noise.
We can aim to find a filter g(σj) that minimises the expected reconstruction error over
this dataset.
This leads to data-driven spectral regularisation, where the filter itself is shaped by the
statistical properties of the signal and noise.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 81 ∞


---

## 第223页

Data-Driven Spectral Regularisation: Problem Setup
Let K : U →V be a compact linear operator between Hilbert spaces with singular system
{(σj, uj, vj)}. We consider measurements fδ = Ku + v, where u is the true signal and v is
additive noise.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 82 ∞


---

## 第224页

Data-Driven Spectral Regularisation: Problem Setup
Let K : U →V be a compact linear operator between Hilbert spaces with singular system
{(σj, uj, vj)}. We consider measurements fδ = Ku + v, where u is the true signal and v is
additive noise.
Objective
We seek to find filter coefficients gj := g(σj) for the spectral regularisation operator
R(fδ; g) =
∞
X
j=1
gj

fδ, vj

V uj
by minimising the expected squared error
min
{gj} Eu,v
hu −R(fδ; g)
2
U
i
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 82 ∞


---

## 第225页

Data-Driven Spectral Regularisation: Problem Setup
Objective
We seek to find filter coefficients gj := g(σj) for the spectral regularisation operator
R(fδ; g) =
∞
X
j=1
gj

fδ, vj

V uj
by minimising the expected squared error
min
{gj} Eu,v
hu −R(fδ; g)
2
U
i
Assumptions for Derivation
The noise v has zero mean, i.e. Ev[v] = 0.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 82 ∞


---

## 第226页

Data-Driven Spectral Regularisation: Problem Setup
Objective
We seek to find filter coefficients gj := g(σj) for the spectral regularisation operator
R(fδ; g) =
∞
X
j=1
gj

fδ, vj

V uj
by minimising the expected squared error
min
{gj} Eu,v
hu −R(fδ; g)
2
U
i
Assumptions for Derivation
The noise v has zero mean, i.e. Ev[v] = 0.
The noise v is statistically independent of the signal u.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 82 ∞


---

## 第227页

Data-Driven Spectral Regularisation: Problem Setup
Objective
min
{gj} Eu,v
hu −R(fδ; g)
2
U
i
Assumptions for Derivation
The noise v has zero mean, i.e. Ev[v] = 0.
The noise v is statistically independent of the signal u.
The noise components

v, vj

are uncorrelated with signal components ⟨u, uk⟩.
Specifically, Eu,v

u, uj
 
v, vj

= Eu

u, uj

Ev

v, vj

= 0 if Ev

v, vj

= 0.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 82 ∞


---

## 第228页

Data-Driven Spectral Regularisation: Optimal Filter
Under the stated assumptions, the expected squared error can be decomposed:
Eu,v
hu −R(fδ; g)
2
U
i
= Eu
h
∥u0∥2
U
i
+
∞
X
j=1

(1 −σjgj)2Eu
h
u, uj
2
U
i
+ g2
jEv
h
v, vj
2
V
i
where u0 is the component of u in N(K).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 83 ∞


---

## 第229页

Data-Driven Spectral Regularisation: Optimal Filter
Under the stated assumptions, the expected squared error can be decomposed:
Eu,v
hu −R(fδ; g)
2
U
i
= Eu
h
∥u0∥2
U
i
+
∞
X
j=1

(1 −σjgj)2Eu
h
u, uj
2
U
i
+ g2
jEv
h
v, vj
2
V
i
where u0 is the component of u in N(K). Let us define
Signal power per component: Πj := Eu
h
u, uj
2
U
i
Noise power per component: ∆j := Ev
h
v, vj
2
V
i
(depends on noise level δ)
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 83 ∞


---

## 第230页

Data-Driven Spectral Regularisation: Optimal Filter
Under the stated assumptions, the expected squared error can be decomposed:
Eu,v
hu −R(fδ; g)
2
U
i
= Eu
h
∥u0∥2
U
i
+
∞
X
j=1

(1 −σjgj)2Eu
h
u, uj
2
U
i
+ g2
jEv
h
v, vj
2
V
i
where u0 is the component of u in N(K). Let us define
Signal power per component: Πj := Eu
h
u, uj
2
U
i
Noise power per component: ∆j := Ev
h
v, vj
2
V
i
(depends on noise level δ)
The expression to minimise (ignoring the u0 term as it’s independent of gj) is
P
j

(1 −σjgj)2Πj + g2
j∆j

.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 83 ∞


---

## 第231页

Data-Driven Spectral Regularisation: Optimal Filter
Under the stated assumptions, the expected squared error can be decomposed:
Eu,v
hu −R(fδ; g)
2
U
i
= Eu
h
∥u0∥2
U
i
+
∞
X
j=1

(1 −σjgj)2Eu
h
u, uj
2
U
i
+ g2
jEv
h
v, vj
2
V
i
where u0 is the component of u in N(K). Let us define
Signal power per component: Πj := Eu
h
u, uj
2
U
i
Noise power per component: ∆j := Ev
h
v, vj
2
V
i
(depends on noise level δ)
The expression to minimise (ignoring the u0 term as it’s independent of gj) is
P
j

(1 −σjgj)2Πj + g2
j∆j

. Minimising with respect to each gj (by setting ∂/∂gj = 0)
yields the optimal filter coefficients
gj =
σjΠj
σ2
jΠj + ∆j
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 83 ∞


---

## 第232页

Data-Driven Spectral Regularisation: Filter Properties
The optimal learned filter is gj =
σjΠj
σ2
jΠj+∆j .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 84 ∞


---

## 第233页

Data-Driven Spectral Regularisation: Filter Properties
The optimal learned filter is gj =
σjΠj
σ2
jΠj+∆j .
Resemblance to Tikhonov Regularisation
If Πj > 0, we can rewrite this as
gj =
σj
σ2
j + (∆j/Πj)
This is identical in form to the Tikhonov filter gα(σj) =
σj
σ2
j+α, but with an adaptive,
data-driven regularisation parameter αj = ∆j/Πj for each singular component j.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 84 ∞


---

## 第234页

Data-Driven Spectral Regularisation: Filter Properties
The optimal learned filter is gj =
σjΠj
σ2
jΠj+∆j .
Resemblance to Tikhonov Regularisation
If Πj > 0, we can rewrite this as
gj =
σj
σ2
j + (∆j/Πj)
This is identical in form to the Tikhonov filter gα(σj) =
σj
σ2
j+α, but with an adaptive,
data-driven regularisation parameter αj = ∆j/Πj for each singular component j.
If signal power Πj is high relative to noise power ∆j, then αj is small, and gj ≈1/σj.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 84 ∞


---

## 第235页

Data-Driven Spectral Regularisation: Filter Properties
The optimal learned filter is gj =
σjΠj
σ2
jΠj+∆j .
Resemblance to Tikhonov Regularisation
If Πj > 0, we can rewrite this as
gj =
σj
σ2
j + (∆j/Πj)
This is identical in form to the Tikhonov filter gα(σj) =
σj
σ2
j+α, but with an adaptive,
data-driven regularisation parameter αj = ∆j/Πj for each singular component j.
If signal power Πj is high relative to noise power ∆j, then αj is small, and gj ≈1/σj.
If signal power Πj is low relative to noise power ∆j, then αj is large, and gj is small.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 84 ∞


---

## 第236页

Data-Driven Spectral Regularisation: Filter Properties
The optimal learned filter is gj =
σjΠj
σ2
jΠj+∆j .
Resemblance to Tikhonov Regularisation
If Πj > 0, we can rewrite this as
gj =
σj
σ2
j + (∆j/Πj)
This is identical in form to the Tikhonov filter gα(σj) =
σj
σ2
j+α, but with an adaptive,
data-driven regularisation parameter αj = ∆j/Πj for each singular component j.
If signal power Πj is high relative to noise power ∆j, then αj is small, and gj ≈1/σj.
If signal power Πj is low relative to noise power ∆j, then αj is large, and gj is small.
The operator is R(fδ; g) = P∞
j=1 gj

fδ, vj

uj.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 84 ∞


---

## 第237页

Learned Spectral Method as a Regularisation Operator
Let statistics Πj and ∆j(δ) (for given overall noise level δ) be fixed. This defines filter
coefficients gj(δ). The operator is Rg(δ)(f) = P
j gj(δ)

f, vj

uj.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 85 ∞


---

## 第238页

Learned Spectral Method as a Regularisation Operator
Let statistics Πj and ∆j(δ) (for given overall noise level δ) be fixed. This defines filter
coefficients gj(δ). The operator is Rg(δ)(f) = P
j gj(δ)

f, vj

uj. For Rg(δ) to be a
regularisation operator, it must be continuous for this fixed g(δ). Since it is linear, this
means it must be bounded.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 85 ∞


---

## 第239页

Learned Spectral Method as a Regularisation Operator
Let statistics Πj and ∆j(δ) (for given overall noise level δ) be fixed. This defines filter
coefficients gj(δ). The operator is Rg(δ)(f) = P
j gj(δ)

f, vj

uj. For Rg(δ) to be a
regularisation operator, it must be continuous for this fixed g(δ). Since it is linear, this
means it must be bounded.
Boundedness [10, Lemma 1]
Assume there exist c > 0 and j0 ∈N such that for all j ⩾j0 and for a given δ > 0 we have
∆j(δ) ⩾cδ2Πj .
(This means noise doesn’t decay faster than signal for high frequencies, which is typical).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 85 ∞


---

## 第240页

Learned Spectral Method as a Regularisation Operator
Let statistics Πj and ∆j(δ) (for given overall noise level δ) be fixed. This defines filter
coefficients gj(δ). The operator is Rg(δ)(f) = P
j gj(δ)

f, vj

uj. For Rg(δ) to be a
regularisation operator, it must be continuous for this fixed g(δ). Since it is linear, this
means it must be bounded.
Boundedness [10, Lemma 1]
Assume there exist c > 0 and j0 ∈N such that for all j ⩾j0 and for a given δ > 0 we have
∆j(δ) ⩾cδ2Πj .
(This means noise doesn’t decay faster than signal for high frequencies, which is typical).
Under this assumption, it can be shown that
sup
j
|gj(δ)| ⩽max
 1
σj0
,
1
2√cδ

< ∞.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 85 ∞


---

## 第241页

Learned Spectral Method as a Regularisation Operator
For Rg(δ) to be a regularisation operator, it must be continuous for this fixed g(δ). Since it
is linear, this means it must be bounded.
Boundedness [10, Lemma 1]
Assume there exist c > 0 and j0 ∈N such that for all j ⩾j0 and for a given δ > 0 we have
∆j(δ) ⩾cδ2Πj .
(This means noise doesn’t decay faster than signal for high frequencies, which is typical).
Under this assumption, it can be shown that
sup
j
|gj(δ)| ⩽max
 1
σj0
,
1
2√cδ

< ∞.
Thus, Rg(δ) is a bounded linear operator for fixed δ > 0, and therefore a regularisation
operator.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 85 ∞


---

## 第242页

Convergence of Learned Spectral Regularisation (Mean-Squared Error)
A regularisation method (Rα, αchoice) is convergent if Rα(δ,fδ)(fδ) approaches a true
solution u† as noise δ →0.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 86 ∞


---

## 第243页

Convergence of Learned Spectral Regularisation (Mean-Squared Error)
A regularisation method (Rα, αchoice) is convergent if Rα(δ,fδ)(fδ) approaches a true
solution u† as noise δ →0.
For the data-driven spectral regularisation, the ”parameter choice” is implicit; the filter
g(δ) is determined by the (statistical) noise level δ via ∆j(δ).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 86 ∞


---

## 第244页

Convergence of Learned Spectral Regularisation (Mean-Squared Error)
A regularisation method (Rα, αchoice) is convergent if Rα(δ,fδ)(fδ) approaches a true
solution u† as noise δ →0.
For the data-driven spectral regularisation, the ”parameter choice” is implicit; the filter
g(δ) is determined by the (statistical) noise level δ via ∆j(δ).
Convergence for this method is typically established in a mean-squared error (MSE)
sense.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 86 ∞


---

## 第245页

Convergence of Learned Spectral Regularisation (Mean-Squared Error)
Convergence for this method is typically established in a mean-squared error (MSE) sense.
Convergence in MSE [10, Theorem 1]
Let u ∈N(K)⊥. Assume that for any j where Πj = Eu[

u, uj
2] = 0, we also have

u, uj

= 0. Let fδ = Ku + v where v is random noise with Ev[

v, vj
2] = ∆j(δ), and
∆j(δ) →0 as δ →0. Then, the expected squared error converges to zero, i.e.
lim
δ→0 Ev
hu −R(fδ; g(δ))
2
U
i
= 0 .
Furthermore, the expected error over the data distribution also converges, i.e.
lim
δ→0

Eu,v
hu −R(fδ; g(δ))
2
U
i
−Eu
h
∥u0∥2
U
i
= 0 .
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 87 ∞


---

## 第246页

Convergence of Learned Spectral Regularisation (Mean-Squared Error)
Convergence in MSE [10, Theorem 1]
Let u ∈N(K)⊥. Assume that for any j where Πj = Eu[

u, uj
2] = 0, we also have

u, uj

= 0. Let fδ = Ku + v where v is random noise with Ev[

v, vj
2] = ∆j(δ), and
∆j(δ) →0 as δ →0. Then, the expected squared error converges to zero, i.e.
lim
δ→0 Ev
hu −R(fδ; g(δ))
2
U
i
= 0 .
Furthermore, the expected error over the data distribution also converges, i.e.
lim
δ→0

Eu,v
hu −R(fδ; g(δ))
2
U
i
−Eu
h
∥u0∥2
U
i
= 0 .
This means the method is convergent in the sense that, on average, the reconstructions
approach the true solution as the noise level diminishes.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 87 ∞


---

## 第247页

Data-Driven Spectral Regularisation: Convergence Rates
For classical spectral methods like Tikhonov or TSVD, convergence rates of the form
Rα(δ)fδ −u†
U = O(δν) can often be established under specific source conditions on u†.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 88 ∞


---

## 第248页

Data-Driven Spectral Regularisation: Convergence Rates
For classical spectral methods like Tikhonov or TSVD, convergence rates of the form
Rα(δ)fδ −u†
U = O(δν) can often be established under specific source conditions on u†.
Rates for Data-Driven Filters
Deriving explicit O(δν) rates for the deterministic error
R(fδ; g(δ)) −u†
U for the
data-driven filter gj(δ) =
σjΠj
σ2
jΠj+∆j(δ) is more involved.
The rate depends on:
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 88 ∞


---

## 第249页

Data-Driven Spectral Regularisation: Convergence Rates
For classical spectral methods like Tikhonov or TSVD, convergence rates of the form
Rα(δ)fδ −u†
U = O(δν) can often be established under specific source conditions on u†.
Rates for Data-Driven Filters
Deriving explicit O(δν) rates for the deterministic error
R(fδ; g(δ)) −u†
U for the
data-driven filter gj(δ) =
σjΠj
σ2
jΠj+∆j(δ) is more involved.
The rate depends on:
The decay rate of the true signal power Πj (related to source conditions).
O
thi
A
t
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 88 ∞


---

## 第250页

Data-Driven Spectral Regularisation: Convergence Rates
For classical spectral methods like Tikhonov or TSVD, convergence rates of the form
Rα(δ)fδ −u†
U = O(δν) can often be established under specific source conditions on u†.
Rates for Data-Driven Filters
Deriving explicit O(δν) rates for the deterministic error
R(fδ; g(δ)) −u†
U for the
data-driven filter gj(δ) =
σjΠj
σ2
jΠj+∆j(δ) is more involved.
The rate depends on:
The decay rate of the true signal power Πj (related to source conditions).
The behavior of the noise power ∆j(δ) as a function of δ and j.
O
thi
A
t
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 88 ∞


---

## 第251页

Data-Driven Spectral Regularisation: Convergence Rates
For classical spectral methods like Tikhonov or TSVD, convergence rates of the form
Rα(δ)fδ −u†
U = O(δν) can often be established under specific source conditions on u†.
Rates for Data-Driven Filters
Deriving explicit O(δν) rates for the deterministic error
R(fδ; g(δ)) −u†
U for the
data-driven filter gj(δ) =
σjΠj
σ2
jΠj+∆j(δ) is more involved.
The rate depends on:
The decay rate of the true signal power Πj (related to source conditions).
The behavior of the noise power ∆j(δ) as a function of δ and j.
The convergence proof for the MSE (previous slide) demonstrates convergence to zero,
but a specific power of δ for the MSE is not immediately extracted without further
assumptions on Πj and ∆j(δ).
O
thi
A
t
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 88 ∞


---

## 第252页

Summary: Key Principles of Regularisation
The Challenge: Inverse problems (Ku = f) are typically ill-posed. Direct solutions are
highly sensitive to noise in data f.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 89 ∞


---

## 第253页

Summary: Key Principles of Regularisation
The Challenge: Inverse problems (Ku = f) are typically ill-posed. Direct solutions are
highly sensitive to noise in data f.
Regularisation Strategy:
Stabilise the problem by incorporating prior knowledge, controlled by parameters α ∈A.
Aim for methods Rα(fδ) that are:
Stable: Continuous for fixed α.
Convergent: Approach a true solution as noise δ →0 (with α appropriately).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 89 ∞


---

## 第254页

Summary: Key Principles of Regularisation
The Challenge: Inverse problems (Ku = f) are typically ill-posed. Direct solutions are
highly sensitive to noise in data f.
Regularisation Strategy:
Stabilise the problem by incorporating prior knowledge, controlled by parameters α ∈A.
Aim for methods Rα(fδ) that are:
Stable: Continuous for fixed α.
Convergent: Approach a true solution as noise δ →0 (with α appropriately).
Common Approaches:
Variational Methods: Minimise data fidelity F(Ku, fδ) plus penalty Jα(u) (e.g., Tikhonov,
TV).
Iterative Methods: Early stopping of algorithms; iteration count is α.
Spectral Methods: Filter singular values (e.g., Tikhonov); can be data-driven.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 89 ∞


---

## 第255页

Summary: Key Principles of Regularisation
The Challenge: Inverse problems (Ku = f) are typically ill-posed. Direct solutions are
highly sensitive to noise in data f.
Regularisation Strategy:
Stabilise the problem by incorporating prior knowledge, controlled by parameters α ∈A.
Aim for methods Rα(fδ) that are:
Stable: Continuous for fixed α.
Convergent: Approach a true solution as noise δ →0 (with α appropriately).
Common Approaches:
Variational Methods: Minimise data fidelity F(Ku, fδ) plus penalty Jα(u) (e.g., Tikhonov,
TV).
Iterative Methods: Early stopping of algorithms; iteration count is α.
Spectral Methods: Filter singular values (e.g., Tikhonov); can be data-driven.
Analysis Goals: Prove stability and convergence, and establish error rates (e.g., O(δν)
using norms or Bregman distances).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 89 ∞


---

## 第256页

Outlook and Open Questions
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 90 ∞


---

## 第257页

Outlook: Some Open Problems
There are quite a few open problems in the context of determinstic regularisation theory, e.g.
Most of the presented theory works for linear K, but what about nonlinear K? What
conditions are needed? Tangential cone condition, or something less restrictive?
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 91 ∞


---

## 第258页

Outlook: Some Open Problems
There are quite a few open problems in the context of determinstic regularisation theory, e.g.
Most of the presented theory works for linear K, but what about nonlinear K? What
conditions are needed? Tangential cone condition, or something less restrictive?
We have error estimates for variational models of the form
Rα(fδ) ∈arg min
u∈U

F(Ku, fδ) + Jα(u)
	
,
but we have no convergence rates for their iterative counterparts, i.e.
uk+1 ∈arg min
u∈U

F(Ku, fδ) + Dpk
J (u, uk)

,
pk+1 = pk −K∗∂1F(Kuk+1, fδ) ,
except for F(Ku, fδ) = 1
2∥Ku −fδ∥2
V (same for inverse scale space methods).
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 91 ∞


---

## 第259页

Outlook: Some Open Problems
Most of the presented theory works for linear K, but what about nonlinear K? What
conditions are needed? Tangential cone condition, or something less restrictive?
We have error estimates for variational models of the form
Rα(fδ) ∈arg min
u∈U

F(Ku, fδ) + Jα(u)
	
,
but we have no convergence rates for their iterative counterparts, i.e.
uk+1 ∈arg min
u∈U

F(Ku, fδ) + Dpk
J (u, uk)

,
pk+1 = pk −K∗∂1F(Kuk+1, fδ) ,
except for F(Ku, fδ) = 1
2∥Ku −fδ∥2
V (same for inverse scale space methods).
No convergence rates for coordinate descent-, Kaczmarz or incremental methods,
which probably would have closest resemblence to neural network architectures.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 91 ∞


---

## 第260页

Outlook: Some Open Problems
What about non-convex F, or non-convex Jα?
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 92 ∞


---

## 第261页

Outlook: Some Open Problems
What about non-convex F, or non-convex Jα?
What about general operators Rα? When do these form convergent regularisation
methods?
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 92 ∞


---

## 第262页

Outlook: Some Open Problems
What about non-convex F, or non-convex Jα?
What about general operators Rα? When do these form convergent regularisation
methods?
What about general selection operators? For example, would it make sense to train a
neural network SΘ with SΘ(Ku) ⊂{u} + N(K), and simultaniously train a network RΨ
on noisy data fδ, such that limΨ→Θ RΨ(Ku) = SΘ(Ku)?
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 92 ∞


---

## 第263页

Outlook: Some Open Problems
What about non-convex F, or non-convex Jα?
What about general operators Rα? When do these form convergent regularisation
methods?
What about general selection operators? For example, would it make sense to train a
neural network SΘ with SΘ(Ku) ⊂{u} + N(K), and simultaniously train a network RΨ
on noisy data fδ, such that limΨ→Θ RΨ(Ku) = SΘ(Ku)?
Extensions to statistical regularisations?
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 92 ∞


---

## 第264页

Outlook: Some Open Problems
What about non-convex F, or non-convex Jα?
What about general operators Rα? When do these form convergent regularisation
methods?
What about general selection operators? For example, would it make sense to train a
neural network SΘ with SΘ(Ku) ⊂{u} + N(K), and simultaniously train a network RΨ
on noisy data fδ, such that limΨ→Θ RΨ(Ku) = SΘ(Ku)?
Extensions to statistical regularisations?
Extensions to Bayesian regularisation methods?
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 92 ∞


---

## 第265页

Outlook: Some Open Problems
What about non-convex F, or non-convex Jα?
What about general operators Rα? When do these form convergent regularisation
methods?
What about general selection operators? For example, would it make sense to train a
neural network SΘ with SΘ(Ku) ⊂{u} + N(K), and simultaniously train a network RΨ
on noisy data fδ, such that limΨ→Θ RΨ(Ku) = SΘ(Ku)?
Extensions to statistical regularisations?
Extensions to Bayesian regularisation methods?
And many more open questions (discussion?)
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 92 ∞


---

## 第266页

References I
Martin Benning and Martin Burger.
Modern regularization methods for inverse problems.
Acta Numerica, 27:1–111, 2018.
L.M. Bregman.
The relaxation method for finding the common point of convex sets and its application to the solution of problems in
convex programming.
USSR Comp. Math. Math. Phys., 7:200–217, 1967.
A. Chambolle and T. Pock.
A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging.
Journal of Mathematical Imaging and Vision, 40(1):120–145, 2011.
Antonin Chambolle and Thomas Pock.
An introduction to continuous optimization for imaging.
Acta Numerica, 25:161–319, 2016.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 93 ∞


---

## 第267页

References II
Heinz Werner Engl, Martin Hanke, and Andreas Neubauer.
Regularization of inverse problems, volume 375.
Springer Science & Business Media, 1996.
Ernie Esser, Xiaoqun Zhang, and Tony F Chan.
A general framework for a class of first order primal-dual algorithms for convex optimization in imaging science.
SIAM Journal on Imaging Sciences, 3(4):1015–1046, 2010.
J. Hadamard.
Sur les probl`emes aux d´eriv´ees partielles et leur signification physique.
Princeton university bulletin, pages 49–52, 1902.
J. Hadamard.
Lectures on cauchy’s problem in linear partial differential equations, yale univ.
Press. New Haven, 1923.
Fritz John.
Continuous dependence on data for solutions of partial differential equations with a prescribed bound.
Communications on pure and applied mathematics, 13(4):551–585, 1960.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 94 ∞


---

## 第268页

References III
Samira Kabri, Alexander Auras, Danilo Riccio, Hartmut Bauermeister, Martin Benning, Michael Moeller, and Martin
Burger.
Convergent Data-Driven Regularizations for CT Reconstruction.
Communications on Applied Mathematics and Computation, 6(2):1342–1368, June 2024.
Krzysztof C Kiwiel.
Proximal minimization methods with generalized bregman functions.
SIAM journal on control and optimization, 35(4):1142–1168, 1997.
Cesare Molinari, Mathurin Massias, Lorenzo Rosasco, and Silvia Villa.
Iterative regularization for low complexity regularizers.
Numerische Mathematik, 156(2):641–689, 2024.
Thomas Pock, Daniel Cremers, Horst Bischof, and Antonin Chambolle.
An algorithm for minimizing the Mumford-Shah functional.
In Computer Vision, 2009 IEEE 12th International Conference on, pages 1133–1140. IEEE, 2009.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 95 ∞


---

## 第269页

References IV
Otmar Scherzer, Markus Grasmair, Harald Grossauer, Markus Haltmeier, and Frank Lenzen.
Variational methods in imaging, volume 167.
Springer, 2009.
Andrei Nikolaevich Tikhonov.
On the stability of the functional optimization problem.
USSR Computational Mathematics and Mathematical Physics, 6(4):28–33, 1966.
Andrey Tikhonov.
Solution of incorrectly formulated problems and the regularization method.
Soviet Meth. Dokl., 4:1035–1038, 1963.
Andrey Nikolayevich Tikhonov.
On the stability of inverse problems.
In Dokl. Akad. Nauk SSSR, volume 39, pages 195–198, 1943.
Mingqiang Zhu and Tony Chan.
An efficient primal-dual hybrid gradient algorithm for total variation image restoration.
UCLA CAM Report, 34, 2008.
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 96 ∞


---

## 第270页

Questions?
Thank You!
Martin Benning (University College London)
Regularisation Theory
9 June 2025
Slide 97 ∞


---

