# MIVAcourse_opt1.pdf

## 第1页

Lecture 1: Convex smooth optimisation
Luca Calatroni
CR CNRS, Laboratoire I3S
CNRS, UCA, Inria SAM, France
MIVA ERASMUS BIP PhD winter school
Advanced methods for mathematical image analysis
University of Bologna, IT
January 18-20 2022


---

## 第2页

Table of contents
1. Introduction
2. Notation, preliminaries & basic notions
Convexity, strong convexity
Lower semi-continuity & coercivity
Diﬀerentiability and L-smoothness
3. Smooth optimisation algorithms
Gradient descent algorithm
Convergence proof under PL condition
Motivation for accelerated algorithms
4. Accelerated smooth optimisation algorithms
Nesterov acceleration of GD
1


---

## 第3页

Schedule
2


---

## 第4页

Introduction


---

## 第5页

Motivation
Goal: providing theoretical & practical tools (i.e. algorithms) for solving
min
x∈Rn F(x)
for a functional F : Rn →R with suitable properties.
• F is smooth →gradient descent & variants (this lecture)
• F := f + g, f smooth & g non-smooth →proximal-gradient algorithms &
variants (next lecture)
• F := f + ∥x∥0 with f smooth →which algorithms? (last lecture)
Such minimisation problems often appears in many contexts:
•
Inverse problems in signal/image processing: image reconstruction,
variable/parameter selection, compressed sensing. . . .
• Statistical/machine learning: empirical risk minimisation, regression. . .
• Optimisation per se: analysis/implementation of fast algorithms for solving
large-scale problems. . .
3


---

## 第6页

Framework: optimisation for inverse problems in imaging
Given y ∈Rm, A ∈Rm×n
ﬁnd
x ∈Rn
s.t.
y = T (Ax)
where m ≤n and T : Rm →Rm models noise degradation.
• Image restoration (denoising, deconvolution, super-resolution)
4


---

## 第7页

Framework: optimisation for inverse problems in imaging
Given y ∈Rm, A ∈Rm×n
ﬁnd
x ∈Rn
s.t.
y = T (Ax)
where m ≤n and T : Rm →Rm models noise degradation.
• Image restoration (denoising, deconvolution, super-resolution)
• Image reconstruction (e.g., medical imaging)
4


---

## 第8页

Framework: optimisation for inverse problems in imaging
Given y ∈Rm, A ∈Rm×n
ﬁnd
x ∈Rn
s.t.
y = T (Ax)
where m ≤n and T : Rm →Rm models noise degradation.
• Image restoration (denoising, deconvolution, super-resolution)
• Image reconstruction (e.g., medical imaging)
• Dictionary representation (data analysis, vision)
4


---

## 第9页

Framework: optimisation for inverse problems in imaging
Given y ∈Rm, A ∈Rm×n
ﬁnd
x ∈Rn
s.t.
y = T (Ax)
where m ≤n and T : Rm →Rm models noise degradation.
• Image restoration (denoising, deconvolution, super-resolution)
• Image reconstruction (e.g., medical imaging)
• Dictionary representation (data analysis, vision)
. . . “naive inversion” not possible for y = Ax + n, n ∼N(0, σ2Id):
(((((((
x = A−1(y −n)
4


---

## 第10页

Bad positioning of inverse ﬁltering
y = Ax + n
Inverse ﬁltering approach:
x = A−1y = A−1(Ax + n) = x + A−1n
Ampliﬁcation of the noise if A−1 is bad conditioned! Need of regularisation!
Find an estimate Rn ∋x∗≈x by solving
x∗∈arg min
x∈Rn
F(x) := f (x) + g(x)
• f is the data ﬁdelity term, it relates to noise statistics
• g is the regularisation term, it encodes a priori information expected on
the desired solution
5


---

## 第11页

Variational regularisation: Bayesian motivation
Following a Bayesian/MAP approach consider:
P(y|Ax; θf )
(likelihood),
P(x; θg)
(prior)
with θf , θg > 0 hyperparameters of the distributions. By Bayes’ theorem:
x∗∈arg max
x
P(x|y) = arg max
x
P(y|Ax; θf )P(x; θg)
P(y)
⇔x∗∈arg min
x
−ln(P(x|y)) = arg min
x
−ln(P(y|Ax; θf )) −ln(P(x; θg)) +
ln(P(y))
Now, if P(x; θg) = e−θg g(x) and P(y|Ax; θf ) = e−θf f (x), then:
x∗∈arg min
x∈Rn
f (x) + λg(x),
λ := θg/θf
Note: incorporate the parameter α in either of the two functions, e.g. g(x) := λg(x).
6


---

## 第12页

Exemplar problems: smooth optimisation
y = Ax + b
• Generalised Tikhonov n ∼N(0, σ2Id) (Gaussian noise) and assume x is
smooth in some sense (e.g., in terms of an operator L ∈RN×n)
x∗∈arg min
x∈Rn
1
2∥Ax −y∥2 + λ∥Lx∥2
Examples: L = Id ∈Rn×n, L = D ∈R2n×n (discrete gradient) . . .
Parameter selection for ℓ2-ℓ2 single-image super-resolution, A = SH, where S is a
decimation operator (Pragliola, Calatroni, Lanza, Sgallari, ’21-’22)
7


---

## 第13页

Exemplar problems: non-smooth optimisation
Assume for simplicity additive white Gaussian noise →f (x) = 1
2 ∥Ax −y∥2
• Sparsity (Donoho et al., Cand`es, Romberg, Tao, ’06): sparse recovery:
x∗∈arg min
x∈Rn
1
2 ∥Ax −y∥2 + λ∥x∥1
Analysis approach: sparse representation of x in some overcomplete basis (e.g.,
wavelets, Mallat, ’89) represented by W ∈RN×n
x∗∈arg min
x∈Rn
1
2 ∥Ax −y∥2 + λ∥Wx∥1
• Total variation reconstruction: “few gradients” for removing noise oscillations
and preserving edges (Rudin, Osher, Fatemi, ’92):
x∗∈arg min
x∈Rn
1
2 ∥Ax −y∥2 + λ∥Dx∥2,1
with ∥Dx∥2,1 =
n
X
i=1
q
(Dhx)2
i + (Dvx)2
i and Dx is the discrete image gradient.
8


---

## 第14页

Exemplar problems: non-smooth optimisation (continuation)
It helps in dealing with admissibility constraints:
x∗∈arg min
x∈C
1
2 ∥Ax −y∥2
with C := TM
m=1 Cm and Cm ⊂Rn.
• Non-negativity constraint: x ≥0, C := {x ≥0}.
• Box constraint: x ∈[a, b] =: C
• . . .
How to encode it into a variational formulation?
Using the indicator function ι : Rn →{0, +∞}
ιCm(x) :=
(
0
if x ∈Cm
+∞
if x /∈Cm
x∗∈arg min
x∈Rn
1
2 ∥Ax −y∥2 +
M
X
m=1
ιCm(x)
9


---

## 第15页

Exemplar problems: ℓ2-ℓ0 optimisation
Arising, e.g., in sparse dictionary representation problems
y = Ax + n
where y ∈Rm, A ∈Rm×n and x ∈Rn and m ≪n. Undetermined system!
To minimise the number of entries of solutions, the natural choice is to consider:
x∗∈arg min
x∈Rn
1
2 ∥Ax −y∥2 + λ∥x∥0
or
x∗∈arg min
x:∥x∥0≤K
1
2 ∥Ax −y∥2
∥x∥0 := # {xi, i = 1, . . . , N : xi ̸= 0}
Molecule localisation in
super-resolution microscopy.
10


---

## 第16页

References
Some standard reference books/surveys:
R. Tyller Rockafeller, Convex Analysis, Princeton University Press, 1970.
S. Boyd, L. Vandenberghe, Convex Optimization, Cambridge University
Press, 2004.
N. Parikh, S. Boyd, Proximal Algorithms, Foundations and Trends in
Optimization, 2013.
A. Beck, First-order methods in optimization, Volume 25, MOS-SIAM
series on Optimization, 2017.
A. Chambolle, T. Pock, An introduction to continuous optimization for
imaging, Acta Numerica, 2016
S. Salzo, S. Villa, Proximal Gradient Methods for Machine Learning and
Imaging, Handbook on Harmonic and Applied Analysis, Applied and
Numerical Harmonic Analysis, 2021.
11


---

## 第17页

Back to the abstract problem
x∗∈arg min
x∈Rn
F(x) := f (x) + g(x)
Often the solution x∗cannot be expressed in closed form. We consider eﬃcient
iterative solvers for its computation (especially in large scale context!)
• Avoid inversion A−1 (1 ≪m ≤n)
• How to exploit the mathematical structure of the functions involved?
• How to handle constraints?
• How to speed up the eﬃciency of a ﬁrst-order algorithm?
• What can be said in the non-convex case?
12


---

## 第18页

Notation, preliminaries & basic notions


---

## 第19页

Notation
• (X, ⟨v, w⟩) = (Rn, vT w) with Euclidean norm ∥· ∥as reference Hilbert space.
Extensions to general Hilbert setting straightforward.
• R := R ∪{+∞}, R+ := {α ∈R : α ≥0}, R++ := {α ∈R : α > 0}
• Closed ball of radius δ > 0 in x ∈X:
Bδ(x) = {y ∈X : ∥y −x∥≤δ}
• Convex set C ⊂X
(∀x, y ∈C)
∀α ∈[0, 1]
αx + (1 −α)y ∈C
• Epigraph of a function f : R →R:
epi(f ) = {(x, t) ∈X × R : f (x) ≤t}
13


---

## 第20页

Proper functions
Minimal property to have well-deﬁned minimisation problems.
Deﬁnition (proper function)
A function F : Rn →R is said proper iﬀ
∃x ∈Rn
such that
F(x) ̸= +∞.
We deﬁne P :=

F : Rn →R : F is proper
	
and
dom(F) := {x ∈Rn : F(x) < +∞}
Clearly, F ∈P ⇔dom(F) ̸= ∅.
14


---

## 第21页

Global/local minimisers
For F ∈P, recall:
• global minimiser: x∗∈Rn: F(x∗) ≤F(x) for every x ∈Rn.
• local minimiser: x∗∈Rn: there exists δ > 0 and a neighbourhood Bδ(x∗) such
that F(x∗) ≤F(x) for every x ∈Bδ(x∗).
min
x∈Rn F(x)
VS
arg min
x∈Rn
F(x)
15


---

## 第22页

Global/local minimisers
For F ∈P, recall:
• global minimiser: x∗∈Rn: F(x∗) ≤F(x) for every x ∈Rn.
• local minimiser: x∗∈Rn: there exists δ > 0 and a neighbourhood Bδ(x∗) such
that F(x∗) ≤F(x) for every x ∈Bδ(x∗).
min
x∈Rn F(x)
VS
arg min
x∈Rn
F(x)
Deﬁnition (set of minimisers)
The set of (local, global) minimisers of F is denoted by:
arg min F = {x∗∈Rn : x∗is a minimiser of F} ⊂Rn
Empty? Singleton? (it depends on F)
15


---

## 第23页

Notation, preliminaries & basic notions
Convexity, strong convexity


---

## 第24页

Convex functions
Deﬁnition (convex function)
F ∈P is said to be convex if:
(∀x, y ∈Rn),
(∀α ∈[0, 1]),
F(αx + (1 −α)y) ≤αF(x) + (1 −α)F(y).
Moreover, F is strictly convex if the inequality holds when x, y ∈dom(F), x ̸= y
and α ∈(0, 1). We say that G : Rn →[−∞, +∞) is concave is F = −G is convex.
If a function is not convex nor concave we say that is non-convex.
Convex/concave function
16


---

## 第25页

Convex functions
Deﬁnition (convex function)
F ∈P is said to be convex if:
(∀x, y ∈Rn),
(∀α ∈[0, 1]),
F(αx + (1 −α)y) ≤αF(x) + (1 −α)F(y).
Moreover, F is strictly convex if the inequality holds when x, y ∈dom(F), x ̸= y
and α ∈(0, 1). We say that G : Rn →[−∞, +∞) is concave is F = −G is convex.
If a function is not convex nor concave we say that is non-convex.
Convex VS. strictly convex functions
16


---

## 第26页

Convex functions
Deﬁnition (convex function)
F ∈P is said to be convex if:
(∀x, y ∈Rn),
(∀α ∈[0, 1]),
F(αx + (1 −α)y) ≤αF(x) + (1 −α)F(y).
Moreover, F is strictly convex if the inequality holds when x, y ∈dom(F), x ̸= y
and α ∈(0, 1). We say that G : Rn →[−∞, +∞) is concave is F = −G is convex.
If a function is not convex nor concave we say that is non-convex.
Convex VS. non-convex function
16


---

## 第27页

Convex functions
Deﬁnition (convex function)
F ∈P is said to be convex if:
(∀x, y ∈Rn),
(∀α ∈[0, 1]),
F(αx + (1 −α)y) ≤αF(x) + (1 −α)F(y).
Moreover, F is strictly convex if the inequality holds when x, y ∈dom(F), x ̸= y
and α ∈(0, 1). We say that G : Rn →[−∞, +∞) is concave is F = −G is convex.
If a function is not convex nor concave we say that is non-convex.
Examples:
• F(x) = ∥x∥is convex
∥αx + (1 −α)y∥≤∥αx∥+ ∥(1 −α)y∥= α∥x∥+ (1 −α)∥y∥
∀x, y ∈Rn
• F(x) = ∥x∥2 is strictly convex
• F(x) = ∥x∥p, p ∈[1, +∞) are convex
16


---

## 第28页

Useful properties
Proposition (epigraph of convex functions is convex set)
Let F ∈P. Then F is convex if and only if epi(F) is a convex set.
Proposition (operations with convex functions)
Let f and g be two convex functions and let β ∈R++. Then, the sum f + g
is a convex function and the function βf is a convex function.
17


---

## 第29页

Strong convexity
Deﬁnition (strongly convex function)
F ∈P is said to be strongly convex of parameter µ > 0 iﬀ∀x, y ∈Rn and
∀α ∈[0, 1]:
F(αx + (1 −α)y) ≤αF(x) + (1 −α)F(y)−µ
2 (1 −α)α∥x −y∥2
Proposition (characteristion of strongly convex functions)
F ∈P is µ-strongly convex if and only if G(·) := F(·) −µ
2 ∥· ∥2 is convex.
Proposition (growth condition around minimisers)
If F ∈P is µ-strongly convex and x∗∈arg minx F(x), then:
F(x) −F(x∗) ≥µ
2 ∥x −x∗∥2,
∀x ∈X.
18


---

## 第30页

Strong convexity
Deﬁnition (strongly convex function)
F ∈P is said to be strongly convex of parameter µ > 0 iﬀ∀x, y ∈Rn and
∀α ∈[0, 1]:
F(αx + (1 −α)y) ≤αF(x) + (1 −α)F(y)−µ
2 (1 −α)α∥x −y∥2
Proposition (characteristion of strongly convex functions)
F ∈P is µ-strongly convex if and only if G(·) := F(·) −µ
2 ∥· ∥2 is convex.
Proposition (growth condition around minimisers)
If F ∈P is µ-strongly convex and x∗∈arg minx F(x), then:
F(x) −F(x∗) ≥µ
2 ∥x −x∗∥2,
∀x ∈X.
strong convexity ⇒strict convexity ⇒convexity
Counterexample (strict convexity ̸⇒strong convexity): F : R →R, F(x) = ex.
18


---

## 第31页

Notation, preliminaries & basic notions
Lower semi-continuity & coercivity


---

## 第32页

Lower semi-continuity
Deﬁnition (lower semi-continuity)
Let F ∈P. We say that F is lower semi-continuous (l.s.c.) at the point x ∈Rn iﬀ
F(x) ≤lim inf
y→x F(y).
Equivalently, for every sequence (xk)k∈N with xk →x:
F(x) ≤lim inf
k→+∞F(xk)

=
lim
k→+∞inf

F(xj) : j ≥k
	
.
If F is l.s.c. at every x ∈Rn, we say that the function is l.s.c.
Left: lower l.s.c. Right: where the function is lower l.s.c.?
19


---

## 第33页

Examples of l.s.c. functions
• The functions
F(x) =



0
if x ≤0
1
if x > 0
,
F(x) = ⌈x⌉= min {k ∈Z : x ≤k}
are l.s.c. (but not continuous).
F(x) = ⌈x⌉
• All continuous functions (l.s.c + u.s.c.).
20


---

## 第34页

Coercivity
How to ensure that the minimum is not attained at “extreme points” of the domain?
Deﬁnition (coercivity)
Let F ∈P. We say that F is coercive iﬀ
lim
∥x∥→+∞F(x) = +∞.
Examples:
• F : R →R+, F(x) = ex is not coercive, but F : R →R+, F(x) = e|x| is.
• F : R2 →R+, F(x, y) = x2 + y2 is coercive.
• F : R2 →R+, F(x, y) = x2 −2xy + x2 = (x −y)2 is not coercive. Why?
21


---

## 第35页

Existence of minimisers
Theorem (existence of minimisers)
If F is proper, l.s.c. and coercive, then the set of minimisers of F is non-empty and
compact.
Note: generalises the Bolzano-Weirestrass theorem holding for problems
min
x∈C F(x)
for compact C ⊂Rn s.t. C ∩dom(F) ̸= ∅and continuous F.
22


---

## 第36页

Existence of minimisers
Theorem (existence of minimisers)
If F is proper, l.s.c. and coercive, then the set of minimisers of F is non-empty and
compact.
Note: generalises the Bolzano-Weirestrass theorem holding for problems
min
x∈C F(x)
for compact C ⊂Rn s.t. C ∩dom(F) ̸= ∅and continuous F.
Theorem (convex case)
If F is proper, coercive and convex, then every local minimiser is a global minimiser.
Deﬁnition (Γ0(Rn))
Γ0(X) :=

F : X →R : F is proper, convex and l.s.c.
	
Remark: F ∈Γ0(X) ̸⇒F admits a minimiser. Take e.g. F(x) = −log x, x > 0 and
F(x) = +∞, x ≤0. . . no coercivity guaranteed!
22


---

## 第37页

Uniqueness of minimisers
So far, only existence of minimisers. How to guarantee uniqueness?
Theorem (existence+uniqueness of minimisers)
If F is proper, l.s.c., coercive and strictly convex, then F admits a unique
minimiser.
Equivalently, arg min F = {x∗}, a singleton.
Remark: as strong convexity implies strict convexity, the same holds.
23


---

## 第38页

Notation, preliminaries & basic notions
Diﬀerentiability and L-smoothness


---

## 第39页

Gˆateaux diﬀerentiability
How to provide a characterisation of the minimisers of a function f in terms of
a suitable notion of “∇f ”?
Deﬁnition (Gˆateaux diﬀerentiability)
Let f ∈P and let x ∈dom(f ). For v ∈Rn, we denote the directional
derivative in x along the direction v as the limit
f ′(x; v) = f ′(x)[v] := lim
t→0+
f (x + tv) −f (x)
t
,
when it exists. If there exists w ∈Rn such that:
(∀v ∈Rn)
f ′(x)[v] = ⟨w, v⟩,
then we say that f is Gˆateaux diﬀerentiable in x and denote by ∇f (x) = w
the Gˆateaux derivative (or, simply, the gradient) of f at x.
24


---

## 第40页

Optimality conditions and relations with convexity
Theorem (Fermat’s rule)
Let f ∈Γ0(Rn) be diﬀerentiable at point x∗. Then:
x∗∈arg min
x∈Rn
f (x)
⇐⇒
∇f (x∗) = 0.
25


---

## 第41页

Optimality conditions and relations with convexity
Theorem (Fermat’s rule)
Let f ∈Γ0(Rn) be diﬀerentiable at point x∗. Then:
x∗∈arg min
x∈Rn
f (x)
⇐⇒
∇f (x∗) = 0.
Proposition (Diﬀerentiability and convexity)
Let f ∈Γ0(Rn). Suppose that f is diﬀerentiable on dom(f ). Then the
following statements are equivalent:
1. f is convex;
2. ∀x, y ∈dom(f ), f (y) ≥f (x) + ⟨∇f (x), y −x⟩;
3. ∀x, y ∈dom(f ), ⟨∇f (x) −∇f (y), x −y⟩≥0.
25


---

## 第42页

Diﬀerentiability and strong convexity
Corollary (Diﬀerentiability and strong convexity)
Let f ∈Γ0(Rn) and µ > 0. Suppose that f is diﬀerentiable on dom(f ). Then the
following statements are equivalent:
1. f is µ-strongly convex;
2. ∀x, y ∈dom(f ), f (y) ≥f (x) + ⟨∇f (x), y −x⟩+ µ
2 ∥y −x∥2;
3. ∀x, y ∈dom(f ), ⟨∇f (x) −∇f (y), x −y⟩≥µ∥x −y∥2.
Example: let f (x) = 1
2 ∥Ax −y∥2, for A ∈Rm×n positive deﬁnite, y ∈Rm. Then:
∇f (x) = AT (Ax −y).
Since AT A is positive deﬁnite (i.e., λmin := λmin(AT A) > 0), then:
(∀x, y ∈Rn)
⟨∇f (x) −∇f (y), x −y⟩= ⟨AT A(x −y), x −y⟩≥λmin∥x −y∥2,
hence f is λmin-strongly convex.
Remark: from condition 3., if x∗∈arg min f (x), then for all x ∈dom(f ):
⟨∇f (x) −0, x −x∗⟩≥µ∥x −x∗∥2
⇒
µ∥x −x∗∥≤∥∇f (x)∥
26


---

## 第43页

Polyak- Lojasiewicz condition
Proposition (Polyak- Lojasiewicz condition)
Let f ∈Γ0(Rn) and let µ > 0. Suppose that f is diﬀerentiable on dom(f ), that f is
µ-strongly convex and that there exists x∗∈arg min f (x). Then:
(∀x ∈dom(f ))
f (x) −min
x
f (x) ≤1
2µ ∥∇f (x)∥2
(*)
Proof.
min
y∈dom(f ) f (y) ≥
min
y∈dom(f )

f (x) + ⟨∇f (x), y −x⟩+ µ
2 ∥y −x∥2
≥f (x) + 1
2µ
min
y∈dom(f )


∥µ(y −x) + ∇f (x)∥2
|
{z
}
≥0
−∥∇f (x)∥2



≥f (x) −1
2µ ∥f (x)∥2.
• “Gradient grows as a quadratic function as we increase f ”. Important condition
for achieving fast convergence rates!
• (∗) holds also for non-strongly convex functions (e.g., 1
2 ∥Ax −y∥2 for A not
positive deﬁnite)
27


---

## 第44页

Lipschitz smoothness (L-smoothness)
In the framework of ﬁrst-order optimisation methods, it’s important to provide
conditions on the growth of functions considered.
Deﬁnition (L-smoothness)
Let f ∈Γ0(Rn) be diﬀerentiable. We say that f is an L-smooth function with
constant L ≥0 iﬀ:
∃L ≥0 : ∀x, y ∈Rn
∥∇f (x) −∇f (y)∥≤L∥x −y∥.
Remark: For f (x) = 1
2∥Ax −y∥2
2, you can check L = ∥ATA∥≤∥A∥2.
28


---

## 第45页

Characterisations of L-smoothness
Theorem (characterisation of L-smooth functions)
Let f : Rn →R a convex diﬀerentiable function and let L > 0. The following
statements are equivalent:
1. f is L-smooth
2. (descent lemma)
(∀x, y ∈Rn)
f (y) −f (x) −⟨∇f (x), y −x⟩≤L
2 ∥x −y∥2
3.
(∀x, y ∈Rn)
1
2L ∥f (x) −f (y)∥2 ≤f (y) −f (x) −⟨∇f (x), y −x⟩
4.
(∀x, y ∈Rn)
1
L ∥f (x) −f (y)∥2 ≤⟨∇f (x) −∇f (y), x −y⟩
5.
(∀x, y ∈Rn)
⟨∇f (x) −∇f (y), x −y⟩≤L∥x −y∥2
6.
L
2 ∥· ∥2 −f (·) is convex.
29


---

## 第46页

Comparing smoothness and strong convexity
• f is L-smooth if and only if:
(∀x, y ∈Rn)
f (y) ≤f (x) + ⟨∇f (x), y −x⟩+ L
2 ∥x −y∥2
• f is µ-strongly convex if and only if:
∀x, y ∈dom(f ),
f (y) ≥f (x) + ⟨∇f (x), y −x⟩+ µ
2 ∥y −x∥2
It can be proved that if f is a C 2 function there holds:
µId ⪯∇2f (x) ⪯LId,
for all x
30


---

## 第47页

Smooth optimisation algorithms


---

## 第48页

Smooth optimisation algorithms
Gradient descent


---

## 第49页

Gradient descent
Gradient descent (GD) algorithm: ubiquitous in many applications for
minimising (non-)convex, diﬀerentiable and proper functions f : Rn →R
Algorithm: Gradient Descent (GD) algorithm
Input: τ ∈
 0, 2
L

, x0 ∈Rn.
for k ≥0 do
xk+1 = xk −τ∇f (xk)
end for
• Choice of τ: important to guarantee convergence (need to be suﬃciently
small), it relates to L (∼growth of f ).
Example: minimise f (x) = x2/2. GD iteration: xk+1 = (1 −τ)xk, convergence for. . . ?
• Convexity assumption: no dependence on x0.
• Stopping criterion: relative error ∥xk+1 −xk∥≤tol or gradient check
∥∇f (xk+1)∥≤tol (approaching 0).
31


---

## 第50页

Understanding the step-size upper bound
Lemma
For all k ≥0, there holds:
τ

1 −τL
2

∥f (xk)∥2 ≤f (xk) −f (xk+1).
Thus, if τ < 2
L, then f (xk+1) ≤f (xk), i.e. the GD algorithm is descending.
Proof. Since xk+1 −xk = −τ∇f (xk), then by the characterisation 2. of
L-smoothness we have:
f (xk+1) ≤f (xk) −τ⟨∇f (xk), ∇f (xk)⟩+ L
2 τ 2∥∇f (xk)∥2,
so the thesis follows.
32


---

## 第51页

Convergence of GD algorithm
Theorem (convergence of GD)
Let (xk)k the sequence of iterates generated by GD. Then, if τ ∈(0, 2/L) there
holds:
f (xk) −f (x∗) ≤∥x0 −x∗∥2
2τk
= O
 1
k

Lemma (progress bounds)
For GD iterations with τ = 1/L there holds:
f (xk+1) ≤f (xk) −1
2L ∥∇f (xk)∥2
Proof. Using xk+1 −xk = −1
L ∇f (xk) we can apply the characterisation 2. to get:
f (xk+1) ≤f (xk) −1
L ∥∇f (xk)∥2 + L
2 ∥1
L ∇f (xk)∥2
≤f (xk) −1
2L ∥∇f (xk)∥2.
(1)
We can use this progress bound to show improved rates under Polyak- Lojasiewicz
condition (in particular, strongly convex functions).
33


---

## 第52页

Smooth optimisation algorithms
Convergence proof under PL condition


---

## 第53页

Linear convergence of GD under PL condition
Theorem (linear convergence of GD under PL)
Let (xk)k the sequence of iterates generated by GD. Then, if τ = 1/L there holds:
f (xk) −f (x∗) ≤

1 −µ
L
k
(f (x0) −f (x∗)),
where, notice, 0 < µ ≤L.
Proof. Use the Lemma (progress bound) and the PL inequality:
f (xk+1) ≤f (xk )−
1
2L
∥∇f (xk )∥2 ≤f (xk ) −
µ
L
(f (xk ) −f (x∗)).
Subtracting f (x∗) from both sides we get:
f (xk+1) −f (x∗) ≤

1 −
µ
L

(f (xk ) −f (x∗)).
Applying this recursively gives the thesis since:
f (xk+1) −f (x∗) ≤

1 −
µ
L

(f (xk ) −f (x∗)) ≤

1 −
µ
L
2
(f (xk−1) −f (x∗))
≤. . . ≤

1 −
µ
L
k
(f (x0) −f (x∗)).
To show 0 < µ ≤L, since by descent lemma we have that for all v ∈Rn:
f (x∗) ≤f (v) −
1
2L
∥∇f (v)∥2.
Combining PL with this inequality we get:
1
2µ
∥∇f (v)∥2 ≥f (v) −f (x∗) ≥
1
2L
∥∇f (v)∥2
∀v ∈Rn ⇒µ ≤L
34


---

## 第54页

A practical example
Do we practically see this gain in known problems?
f (x) = 1
2∥Ax −y∥2 + λ
2 ∥x∥2,
λ > 0
f is λ-strongly convex. Convergence factor of the theorem:
µ
L = min

eig(ATA)
	
+ λ
max {eig(ATA)} + λ
• If λ ≫1, then
 1 −µ
L

→0 hence faster convergence
• If L ≫µ (“small” PL), then this rate is not very informative, so in
practice we observe the rate O(1/k).
• The quantity L/µ is called the condition number of f (relates with the
condition number of matrix ∇2f when f is C 2).
35


---

## 第55页

Smooth optimisation algorithms
Motivation for accelerated algorithms


---

## 第56页

Lower bounds for smooth optimisation
. . . back to standard GD iteration and O(1/k) convergence rate.
1Nesterov, 2004, adapted from Chambolle-Pock, 2016
36


---

## 第57页

Lower bounds for smooth optimisation
. . . back to standard GD iteration and O(1/k) convergence rate.
Theorem (worst-case bounds1)
For x0 ∈Rn, L > 0 and 1 < k ≤1
2 (n −1), there exists a convex, L-smooth function
f s.t. for any ﬁrst-order algorithm:
f (xk) −f (x∗) ≥3L∥x0 −x∗∥2
32(k + 1)2
= O

1
(k + 1)2

.
It would be somehow ‘optimal’ ﬁnding convergence rates close to such lower
(inevitable) bound. . .
How to ﬁll the gap between O(1/k) and O(1/(k + 1)2) for convex functions?
1Nesterov, 2004, adapted from Chambolle-Pock, 2016
36


---

## 第58页

Accelerated smooth optimisation
algorithms


---

## 第59页

Accelerated smooth optimisation
algorithms
Nesterov acceleration of GD


---

## 第60页

Accelerated gradient descent
Idea: add inertia to “shift” the sequence of iterates.
xk−1
xk
xk+1
−∇f (xk )
xk−1
xk
xk+1
yk+1
−∇f (yk+1)
Algorithm: Accelerated Gradient Descent (AGD) algorithm 2
Input: x0 = x−1 ∈Rn, τ ∈
 0, 1
L

, t0 = 0.
for k ≥0 do
tk+1 =
1 +
q
1 + 4t2
k
2
yk+1 = xk + tk −1
tk+1
(xk −xk−1)
xk+1 = yk+1 −τ∇f (yk+1)
end for
2Nesterov, 1983
37


---

## 第61页

A note on the sequence
Lemma (behaviour of the sequence (tk))
Let t0 and the sequence tk be deﬁned by:
tk+1 = 1 +
p
1 + 4t2
k
2
.
Then tk ≥k+2
2
for all k ≥0. In particular, tk →∞.
Proof: by induction.
For k = 0 we have t0 ≥1. Suppose that the claim holds for
some k, meaning that tk ≥k+2
2 . Want to show:
tk+1 ≥k + 1 + 2
2
= k + 3
2
.
Using recursion and 2tk ≥k + 2 (induction)
tk+1 =
1 +
q
1 + 4t2
k
2
≥1 +
p
1 + (k + 2)2
2
≥1 +
p
(k + 2)2
2
= k + 3
2
.
Remark: any sequence (tk)k satisfying t2
k+1 −tk+1 ≤t2
k, k ≥0 works (Chambolle,
Dossal, 2015).
38


---

## 第62页

Accelerated convergence result
Theorem (convergence of AGD)3
Let (xk)k the sequence of iterates generated by AGD. Then, there holds:
f (xk) −f (x∗) ≤2∥x0 −x∗∥2
τ(k + 1)2 .
Get faster, at O

1
(k+1)2

to a reasonably accurate approximation of x∗.
. . . proof is quite technical. You’ll see this in the case of non-smooth problems
tomorrow.
3Nesterov, 2004, Chambolle-Pock, 2016
39


---

## 第63页

Accuracy view point
How many iterations are needed for such algorithms to achieve ε-accuracy, i.e.
f (xk) −f (x∗) ≤ε
• GD: for all k ≥0 such that k ≥⌈C/ε⌉
• AGD: for all k ≥0 such that k ≥⌈C/√ε −1⌉
• GD + PL: for all k ≥0 such that k ≥⌈C log (1/ε)⌉
40


---

## 第64页

Conclusions
We focus on convex, smooth optimisation problems arising in applications
(e.g., imaging inverse problems).
• We revised basic notions for having well-posedness of the underlying
problem
• We considered GD as a reference ﬁrst-order algorithm
• We commented on the improved speed achieved by GD whenever the
underlying function enjoys further regularity (PL + strong convexity)
• We discussed Nesterov acceleration for improving convergence speed in
convex cases
How to explore analogous ideas in the structured smooth+non-smooth setting?
41


---

## 第65页

Questions?
calatroni@i3s.unice.fr
41


---

