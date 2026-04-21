# MIVAcourse_opt3.pdf

## 第1页

Lecture 3: Models and algorithms for ℓ2-ℓ0 optimisation problems
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
2. ℓ2-ℓ0 minimisation
3. Algorithms for ℓ2-ℓ0 minimisation
Iterative Hard Thresholding
Greedy algorithms
4. Continuous relaxations
Exactness
Iteratively reweighted algorithms
5. Application to super-resolution microscopy
1


---

## 第3页

Introduction


---

## 第4页

Why ℓ0?
Many problems in signal/image processing are concerned with sparse recovery:
compressed sensing, variable selection, source separation, learning...
d = Ax + n
- d ∈Rm: observed data (signal processing notation)
- x ∈Rn unknown solution to be estimated
- A ∈Rm×n observation matrix,
• Few observations y and large explicative unknown variables x, with m ≪n.
Undertermined system! A is ill-conditioned, noise is present.
• Regularisation: assume the signal is sparse by considering ℓ1-norm or ℓ0
pseudo-norm constraints:
∥x∥1 ≤K,
∥x∥0 ≤K
with ∥x∥0 := # {xi, i = 1, . . . , n : xi ̸= 0} = Pn
i=1 |xi|0, with
|z|0 =
(
1
if x ̸= 0
0
if x = 0
NB: ℓ0-norm is NOT a norm as ∥λx∥0 = ∥x∥0 ̸= λ∥x∥0.
2


---

## 第5页

Dictionary representation in imaging
Image are heterogeneous signals, with smooth (homogeneous) areas, edges, texture,...
Take d ∈Rm be a patch of an image or a signal
Each d is represented by given waveforms whose shape matches the image structure.
Standard choices of ai vectors come from Haar, smooth wavelets, sine/cosine
transform...
Take A = [a1, ..., an] ∈Rm×n to be a set of normalised (basis) vectors.
3


---

## 第6页

Dictionary representation in imaging
• Such A is a redundant dictionary (sequence of representative waveforms)
• The dictionary A is adapted to the signal d if d can be represented by a few
number of vectors ai (atoms) of A, that is d ≈Ax with x sparse, that is
∥x∥0 ≤K,
K << n
4


---

## 第7页

Examples in signal/image processing
Examples
• signal is a sum of spikes, modelled by a sum of Dirac PK
r=1 xrδtr .
• acquisition system is modelled as a convolution with a Gaussian function:
d(·) = h ∗PK
r=1 xrδtr = PK
r=1 xrh(· −tr).
Assume that the Dirac locations tr are on a regular grid indexed by i = 1, ...n
• 1D example: Channel estimation in communications, ...
• 2D example: Single Molecule Localisation in super-resolution microscopy
5


---

## 第8页

Single Molecule Localisation in super-resolution microscopy I
SMLM idea
Modelling: for t ∈{1, . . . , T}, given a blurry, undersampled and noisy image
dt ∈Rm, consider the problem:
ﬁnd sparse
xt
s.t.
dt = Axt + nt,
∀t ∈{1, . . . , T}
A := SH ∈Rm×n with H ∈Rn×n convolution and S ∈Rm×n undersampling , n = Lm, L > 1.
6


---

## 第9页

Single Molecule Localisation in super-resolution microscopy II
Regularisation approach: look for sparse solutions at each time t ∈{1, . . . , T}
x∗
t ∈arg min
x
1
2 ∥Ax −dt∥2 + λ∥x∥0 + ιx≥0(x),
λ > 0
Final reconstruction obtained simply by x = PT
i=1 x∗
t (Gazagnes, Soubies,
Blanc-F´eraud, Schaub, ’15, Lazzaretti, Calatroni, Estatico, ’21)
7


---

## 第10页

ℓ2-ℓ0 minimisation


---

## 第11页

ℓ2-ℓ0 minimisation
ℓ2-ℓ0: problem forms
For A ∈Rm×n, m ≤n consider the following formulations:
• Exact recovery:
ˆx ∈arg min
x∈Rn
∥x∥0 subject to Ax = d
• Approximation problem in constrained forms (ϵ > 0,K > 0)
ˆx ∈arg min
x∈Rn
1
2 ∥Ax −d∥2
2 subject to ∥x∥0 ≤K
ˆx ∈arg min
x∈Rn
∥x∥0 subject to ∥Ax −d∥2
2 ≤ϵ
• Approximation problem in penalised form (λ > 0)
ˆx ∈arg min
x∈Rn
Gℓ0(x) := 1
2 ∥Ax −d∥2
2 + λ∥x∥0
• non-continuous, non-convex and NP-hard optimisation problem (Natarajan, ’95, Davies et
al., ’97): a solution cannot be veriﬁed in polynomial time w.r.t the dimension of the problem
• Non equivalent formulations
• Existence of optimal solutions and relations between formulations in Nikolova, ’16
• Very active ﬁeld of research in signal and image processing, and in statistics.
8


---

## 第12页

How people do: ℓ2-ℓ1 minimisation
A popular way to deal with this problem consists in considering the ℓ1-norm instead
ℓ2-ℓ1 problem formulations
• Constrained formulation (K > 0):
ˆx ∈arg min
x∈Rn
∥Ax −d∥2
2 subject to ∥x∥1 ≤K
• Penalised formulation (λ > 0):
ˆx ∈arg min
x∈Rn
∥Ax −d∥2
2 + λ∥x∥1
• Easier optimization problems: convex and continuous (but non smooth) →
available solvers (see previous courses)!
• The two formulations are equivalent
• Under some conditions involving A, solving these problems allows to ﬁnd a
solution of the ℓ2-ℓ0 problem (Cand`es, Romberg, Tao, ’05)
• They are known as Basis Pursuit De-Noising (BPDN) Chen et al., ’98, or
LASSO (Tibshirani, ’96) problems, respectively.
9


---

## 第13页

ℓ1 norm promotes sparsity
Standard example in R2.
Level lines of ∥Ax −d∥2
2.
10


---

## 第14页

ℓ1 norm promotes sparsity
Standard example in R2.
Level lines of ∥Ax −d∥2
2 with ℓ2 constraint ∥x∥2 ≤K →(x1, x2) ̸= (0, 0).
10


---

## 第15页

ℓ1 norm promotes sparsity
Standard example in R2.
Level lines of ∥Ax −d∥2
2 with ℓ1 constraint ∥x∥1 ≤K →x1 = 0.
10


---

## 第16页

Sparsity through sof-thresholding
Recall that in 1D:
ˆx = arg min
x∈R
1
2 (d −x)2 + λ|x|

= proxλ|·|(d)
is reached at
ˆx = Tλ(d) =
 d −sign(d)λ
if |d| > λ
0
if |d| ≤λ
By, separability, this is then used for deﬁning proxλ∥·∥1(·).
. . . many zeros!
Note: using ℓ2 norm we get instead
ˆx = arg min
x∈R
1
2 (d −x)2 + λx2

.
ˆx =
d
1+2λ which is diﬀerent from 0 as soon as d ̸= 0.
11


---

## 第17页

Algorithmic advantages in solving ℓ2-ℓ1 problems
You now know how to solve the problem:
arg min
x
1
2 ∥Ax −d∥2 + λ∥x∥1,
λ > 0
• ISTA (Combettes, Wajs, ’05)
• FISTA (Beck, Teboulle, ’09)
• If A is positive deﬁnite →strongly convex problem, hence V-FISTA can be used
(Beck, ’17)
For analysis approaches, i.e. when sparsity is assumed w.r.t. to some basis W ∈RN×n
(gradient, wavelets. . . )
arg min
x
1
2 ∥Ax −d∥2 + λ∥Wx∥1,
λ > 0
you can use, e.g., ADMM (Glowinski, Marroco, ’75, Boyd et al, ’11).
12


---

## 第18页

So. . . why just not solving ℓ2-ℓ1?
Compressed Sensing Theory
• A sparse signal (∥x∥0 ≤K) can be exactly reconstructed by solving the
constrained ℓ1 problem when Restricted Isometry Property (RIP) of matrix A
(Donoho et al., Cand`es et al. ’06)
• Roughly speaking A satisﬁes the RIP if AT A ≈Id.
• Under RIP conditions on A, ℓ0 can be replaced by ℓ1.
• Otherwise (frequent cases in inverse problems), the two optimisation problems
give diﬀerent solutions.
• ℓ1 promotes sparsity but introduces biases, since in correspondence of the actual
non-zeros the magnitude is lowered.
• ℓ0 better promotes sparsity than ℓ1 in the general case.
13


---

## 第19页

Algorithms for ℓ2-ℓ0 minimisation


---

## 第20页

Algorithms for ℓ2-ℓ0 minimisation
Iterative Hard Thresholding


---

## 第21页

Non-convex proximal gradient: iterative hard thresholding
Consider the penalised form of the problem:
arg min
x∈Rn
1
2 ∥Ax −d∥2
2 + λ∥x∥0
•
1
2 ∥Ax −d∥2 is L-smooth (L = ∥A∥2)
• The proximal operator of ∥· ∥0 is the hard thresholding operator
Algorithm: Iterative hard thresholding (IHT)
Input: x0 ∈Rn, τ ∈
 0, 1
L

.
for k ≥0 do
xk+1 = proxτλ∥·∥0

xk −τAT (Axk −d)

= H√
2λτ

xk −τAT (Axk −d)

end for
• IHT converges to a critical point (in Blumensath, Davies, ’09 for τ = 1 and ∥A∥< 1, in Attouch et al., ’13
general FB-type result)
• As always for non convex problems, initialisation is crucial! One good idea is to initialise with the solution of
arg min
x∈Rn
1
2
∥Ax −d∥2
2 + λ∥x∥1
→computed by FISTA
14


---

## 第22页

IHT: ideas
arg min
x∈Rn
Gℓ0(x) := 1
2 ∥Ax −d∥2
2 + λ∥x∥0,
Introduce the surrogate function for all z ∈Rn:
C S
ℓ0(x, z) := 1
2 ∥Ax −d∥2
2 + λ∥x∥0 −1
2 ∥Ax −Az∥2
2 + ∥x −z∥2
2
It can be shown that if ∥A∥< 1, then C S
ℓ0(x, z) majorises Gℓ0(x):
Gℓ0(x) ≤C S
ℓ0(x, z),
∀z ∈Rn.
Note, moreover, that Gℓ0(x) = C S
ℓ0(x, x). We can thus optimise C S
ℓ0(x, z) with
respect to x. We can rewrite:
C S
ℓ0(x, z) = 1
2
n
X
i=1

x2
i −2xi

zi + aT
i d −aT
i Az

+ λ|xi|0

+ 1
2
 ∥d∥2 + ∥z∥2 −∥Az∥2
By treating the case xi = 0 and xi ̸= 0 separately and comparing we get:
x = H√
2λ(z −AT (Az −d)),
∀z
IHT obtained by setting z = xk and x = xk+1.
15


---

## 第23页

Algorithms for ℓ2-ℓ0 minimisation
Greedy algorithms


---

## 第24页

Greedy algorithms
Greedy algorithms: matching pursuit (MP) (Mallat et al., ’93), Orthogonal MP (Pati
et al., ’93), Orthogonal Least Squares (OLS, Chen et al., ’89), Bayesian OMP (Herzen
et al., ’10), Single Best Replacement (Soussen et al, ’11).
Matching Pursuit
d ∈Rm is the signal to represent with a limited number of K ≪n of atoms of
dictionary A ∈Rm×n, i.e. of columns ai of A, i = 1, . . . , n.
MP considers the constrained formulation:
arg min
x∈Rn
∥Ax −d∥2,
subject to
∥x∥0 ≤K
and try to add one component at a time.
16


---

## 第25页

Matching pursuit: main ideas
Assumption: A has unit column norms, i.e. ∥ai∥= 1 for all i = 1, . . . , n.
Algorithm: Matching pursuit
Input: A s.t. ∥ai∥= 1, d, K ≪n.
Initialise: r0 = d, σ0 = ∅, x0 = 0.
while #σk ≤K do
ik = arg max
j∈{1,...,n}
|⟨rk, aj⟩|
σk+1 = σk ∪{ik}
xk+1 = xn + ⟨aik , rk⟩eik
rk+1 = rk −⟨rk, aik ⟩aik
end while
• The quantity ∥rk∥converges exponentially to 0 (Mallat et al, ’93)
• In Gribonval et al., ’96, a diﬀerent correlation function (not |⟨·, ·⟩|) is considered.
17


---

## 第26页

Orthogonal Matching Pursuit
OMP idea (Pati et al. ’93, Tropp, ’04): at each iteration of MP optimally estimate
the intensity values having the current support ﬁxed by solving
xk+1 = arg min
x∈Rn
∥Ax −d∥2,
subject to xi = 0 ∀i /∈ω := σ(xk) ∪ik+1
Algorithm: Orthogonal matching pursuit
Input: A s.t. ∥ai∥= 1, d, K ≪n.
Initialise: r0 = d, σ0 = ∅, x0 = 0.
while #σk ≤K do
ik = arg max
j∈{1,...,n}
|⟨rk, aj⟩|
σk+1 = σk ∪{ik}
xk+1 = arg min
x∈Rn
∥Ax −d∥2,
subject to xi = 0 ∀i /∈σ(xk+1)
rk+1 = d −Axk+1
end while
• “Orthogonal” as by deﬁnition at each k ≥0 the residual belongs to the
orthogonal space of the current support
• Convergence in n iterations at most (new component at each iteration)
• Exact sparse recovery results (under some conditions on A) (Tropp, ’04)
18


---

## 第27页

Further greedy algorithms
The main idea of the other existing greedy algorithms is that at each iteration
one component is:
• added
• removed
• replaced
The more complex is the strategy, the best is the solution, but the largest is the
computing time. . .
19


---

## 第28页

Continuous relaxations


---

## 第29页

Continuous relaxation idea
Think of a diﬀerent idea for solving the problem:
1
2 ∥Ax −d∥2 + λ∥x∥0
=⇒
1
2 ∥Ax −d∥2 +
n
X
i=1
φi(xi)
Idea: use continuous and separable functions φi(xi) (convex and non-convex).
• ℓ1 norm: LASSO (Tibshirani, ’96), Basis Pursuit (Chen, ’98), Compressed
Sensing (Donoho, ’06, Cand`es et al., ’06)
• Adaptive LASSO (Zou, ’06)
• Exponential approximation (Mangasarian, ’96)
• Log-sum penalty (Cand`es, ’08)
• Smoothly Clipped Absolute Deviation (SCAD) (Fan, Liu, ’01) and Minimax
Concave Penalty (MCP) (Zhang, ’10
• ℓp “norms”, p < 1 (Chartrand, ’07, Foucart, Lai, ’09)
• Beautiful review (Soubies, Blanc-F´eraud, Aubert, ’17)
Which approximation should we use?
20


---

## 第30页

Continuous relaxation idea
Think of a diﬀerent idea for solving the problem:
1
2 ∥Ax −d∥2 + λ∥x∥0
=⇒
1
2 ∥Ax −d∥2 +
n
X
i=1
φi(xi)
Idea: use continuous and separable functions φi(xi) (convex and non-convex).
−2
0
2
0
0.5
1
1.5
ℓ0
ℓ1
Cap-ℓ1
ℓ0.5
Log-Sum
SCAD
MCP
Exp
Which approximation should we use?
20


---

## 第31页

Continuous relaxation idea
Think of a diﬀerent idea for solving the problem:
1
2 ∥Ax −d∥2 + λ∥x∥0
=⇒
1
2 ∥Ax −d∥2 +
n
X
i=1
φi(xi)
Idea: use continuous and separable functions φi(xi) (convex and non-convex).
Thresholding on R+
Which approximation should we use?
20


---

## 第32页

Continuous relaxations
Exactness


---

## 第33页

What is a good relaxation?
Gℓ0(x) = 1
2 ∥Ax −d∥2 + λ∥x∥0
=⇒
˜G(x) := 1
2 ∥Ax −d∥2 +
n
X
i=1
φi(xi)
Good (exact) relaxation
• Gℓ0(x) and ˜G(x) have the same global minimisers:
arg min
x∈Rn
Gℓ0(x) = arg min
x∈Rn
˜G(x),
(global)
(P1)
•
˜G(x) has “less” local minimisers than Gℓ0(x):
x∗minimiser of ˜G ⇒x∗minimiser of Gℓ0
(P2)
21


---

## 第34页

The continuous exact ℓ0 relaxation (CEL0) penalty
In Soubies, Aubert, Blanc-F´eraud, ’15-’17 a particular choice of φ : R →R+ is studied.
By convex conjugation, the penalty removing most of the local minimisers is:
φCEL0(∥ai∥, λ, x) = λ −∥ai∥2
2
 
|x| −
√
2λ
∥ai∥
!2
1n
|x|≤
√
2λ
∥ai ∥
o
where 1C (x) = 1 if x ∈C and 1C (x) = 0 otherwise.
x
φ(x)
−
√
2λ
∥ai ∥
√
2λ
∥ai ∥
φCEL0
β−
β+
22


---

## 第35页

Good relaxations: examples
-3
-2
-1
0
1
2
3
0
0.2
0.4
0.6
0.8
1
ΦCEL0
λθ =
√
2λa
λθ = 2
√
2λa
λθ = 3
√
2λa
λθ = 4
√
2λa
λθ = 5
√
2λa
Capped-ℓ1, Zhang, ’09
-3
-2
-1
0
1
2
3
0
0.2
0.4
0.6
0.8
1
ΦCEL0
γ = 2 + ε
γ = 2.25
γ = 2.5
γ = 2.75
γ = 1/a2 −1
SCAD, Fan, Li, ’01
-3
-2
-1
0
1
2
3
0
0.2
0.4
0.6
0.8
1
ΦCEL0
γ = 0.9/a2
γ = 0.7/a2
γ = 0.5/a2
γ = 0.3/a2
γ = 0.1/a2
MCP, Zhang, ’01
-3
-2
-1
0
1
2
3
0
0.2
0.4
0.6
0.8
1
ΦCEL0
θ = θ0
θ = 1.1θ0
θ = 1.2θ0
θ = 1.3θ0
θ = 1.4θ0
Truncated-ℓp
Examples of penalties for which (P1) (top) or (P1) and (P2) (bottom) hold for
a = 0.5, λ = 1 and d = 1.8 in the 1D case.
23


---

## 第36页

The CEL0 relaxation
GCEL0(x) := 1
2 ∥Ax −d∥2 +
n
X
i=1
φCEL0(∥ai∥, λ, xi)
|
{z
}
ΦCEL0:=
where: φCEL0(∥ai∥, λ, x) = λ −∥ai ∥2
2

|x| −
√
2λ
∥ai ∥
2
1n
|x|≤
√
2λ
∥ai ∥
o
Properties of GCEL0:
• Inferior limit of all functions satisfying (P1) and (P2)
• Convex envelope of Gℓ0 if A diagonal or AT A = sId, s > 0
• Continuous
• Non convex for general operators A
• Convexity w.r.t. each component xi, i = 1, . . . , n
Thanks to its continuity we can resort to nonsmooth, nonconvex algorithms such as,
e.g., forward-backward and majorisation-minimisation (MM) algorithms (e.g., iterative
reweighted ℓ1 Ochs et al., ’15).
24


---

## 第37页

Understanding the relaxation
1D example: Gℓ0(x) := 1
2 (ax −y)2 + λ|x|0 for a, λ > 0.
Blue lines: plots of Gℓ0 for diﬀerent values of d (note discontinuity in x = 0). Red
lines: plots of GCEL0 (convex biconjugate).
In 1D GCEL0 is always a convex function, in the multi-dimensional case it depends on
the operator A. Generally, it is non-convex with convex 1D restrictions.
25


---

## 第38页

Forward-backward splitting for ℓ2-CEL0
Iterate for k ≥0 and τ ∈(0,
1
∥A∥2 )
xk+1 ∈proxτΦCEL0

xk −τAT (Axk −d)

where, by separability, we can look at the prox of the 1D components:
proxτφCEL0(a,λ;·)(u) =
(
sign(u) min

|u|, (|u| −
√
2λτa)+/(1 −a2τ)

if a2τ < 1
u1|u|>
√
2τλ + {0, u}1|u|=
√
2τλ
if a2τ ≥1
−5
−4
−3
−2
−1
0
1
2
3
4
5
−5
−4
−3
−2
−1
0
1
2
3
4
5
 
 
L0
L1
MCP
Dependence of φCEL0 on a = ∥ai∥at component u = xi.
Convergence to a critical point under Kurdyka- Lojaseiwicz (KL) property (Attouch et al, ’13).
26


---

## 第39页

Continuous relaxations
Iteratively reweighted algorithms


---

## 第40页

Key idea
min
x∈Rn F(x) := f (x) + g(x)
for g proper, l.s.c. and bounded from below but generally non-convex
Majorisation-minimisation technique
Construct a sequence of easier (convex) functions majorising F and minimise
them to simplify the problem.
Minimisation of a non-convex function (red) using MM techniques. Non-convexity
induced by g(x) = log(1 + 2|x|). Majorant functions in blue.
27


---

## 第41页

Key idea
min
x∈Rn F(x) := f (x) + g(x)
for g proper, l.s.c. and bounded from below but generally non-convex
Majorisation-minimisation technique
Construct a sequence of easier (convex) functions majorising F and minimise
them to simplify the problem.
Pseudocode: general idea for MM algorithms
Input: x0 ∈Rn.
while not converging do
Build a majorising function Mxk : Rn →R such that:
• ∀x ∈Rn: F(x) ≤Mxk (x)
• F(xk) = Mxk (xk)
• Mxk (xk) ∈Γ0(Rn)
Deﬁne xk+1 ∈arg minx Mxk (x)
end while
27


---

## 第42页

MM approaches
Several approaches for building such functions:
• Iterative least-squares (IRLS) (Daubechies et al. ’10, Gorodnitsky, Rao, ’97):
Mxk (x) =
X
(wxk )ix2
i
• MM approaches for inverse problems (Chouzenoux et al., ’10 -. . . )
• Iterative reweighted ℓ1 algorithms: better suited to construct majorants of
functions which are not suﬃciently smooth of the form:
F(x) = 1
2 ∥Ax −d∥2 +
X
φ(|xi|)
with φ : R+ →R continuous, concave and non-decreasing (Ochs et al, ’15.)
Algorithm: IRℓ1 (Ochs et al, ’15)
Input: x0 ∈Rn.
while not converging do
(wxk )i ∈∂+φi(|(xk)i|)
xk+1 ∈arg minx
1
2 ∥Ax −d∥2 + Pn
i=1(wxk )i|xi| →solve with FISTA
end while
∂+φi(|(xk)i|) extends the notion of subdiﬀerentials to the non-convex case (Clarke,
’90, Rockfellar, Wets, ’09)
28


---

## 第43页

IRℓ1 for GCEL0 minimisation
Weights can be computed in an explicit form:
(wxk )i :=
(√
2λ∥ai∥−∥ai∥2|(xk)i|
if 0 ≤|(xk)i| <
√
2λ/∥ai∥
0
∥(xk)i| ≥
√
2λ/∥ai∥
Convergence of IRℓ1 to critical points can be proved for general class of functions
satisfying the so-called Kurdyka- Lojasiewicz property (Ochs et al, ’15).
29


---

## 第44页

Application to super-resolution
microscopy


---

## 第45页

Super-resolution microscopy
Spatial resolution is limited by light diﬀraction phenomena.
Point Spread Function: Gaussian, Airy disk. . .
Rayleigh criterion
d =
0.61λ
NA
≈200nm
• λ: emission wavelength
• NA: microscope numerical aperture
30


---

## 第46页

Super-resolution microscopy
Spatial resolution is limited by light diﬀraction phenomena.
Point Spread Function: Gaussian, Airy disk. . .
Rayleigh criterion
d =
0.61λ
NA
≈200nm
• λ: emission wavelength
• NA: microscope numerical aperture
Resolvable VS. non-resolvable line proﬁles
30


---

## 第47页

Discrete mathematical modelling
X
Y
Image formation model
Y = P(Mq(H(X)) + B) + N
• Y ∈RN×N: LR acquisition
• X ∈RL×L: HR image (L = qN, q ∈N)
• P(·): Poisson r.v.
• Mq ∈RN×L: down-sampling matrix
• H ∈RN×N: convolution matrix
• N: additive white Gaussian noise
• B: background
q = 4
31


---

## 第48页

State-of-the-art methods in ﬂuorescence microscopy
Key idea
In microscopy imaging, the experimental setup and the sample preparation can be
used to ‘sparsify’ the measurements.
Fluorescence microscopy
Absorption/emission diagram
Fluorescent molecules
Nobel prize in chemistry in 2008.
32


---

## 第49页

State-of-the-art methods in ﬂuorescence microscopy
Key idea
In microscopy imaging, the experimental setup and the sample preparation can be
used to ‘sparsify’ the measurements.
Example: Single Molecule Localization Microscopy
(Betzig, Zhuang, Hess, ’06, Rust, Bates, Zhuang, ’06)
- Speciﬁc ﬂuorescent molecules activating with low
probability in a sequential way
- Improved sparsity!
http://zeiss-campus.magnet.
fsu.edu/
32


---

## 第50页

Spoiler
yt = P(Ψxt + b) + nt,
Ψ := MqH,
nt ∼N (0, σ2Id),
¯y :=
T
X
t=1
yt/T
33


---

## 第51页

Weighted CEL0
To incorporate signal-dependence (modelling Poisson photon counting) in Lazzaretti,
Calatroni, Estatico, ’21 we considered a weighted ℓ2 ﬁdelity term.
Weighted-ℓ2-ℓ0 problem
x∗∈arg min
x∈RL2


Gwℓ0(x) := 1
2
N2
X
j=1
((Ψx)j −yj −bj)2
yj + bj
+ λ∥x∥0 + ι≥0(x)


,
λ > 0
Theorem
• If ΨT W Ψ = D2 with D = diag(∥ψi ∥W ) ∈RL2×L2, then GwCEL0 = G ∗∗
wℓ0.
• arg min GwCEL0 = arg min Gwℓ0 (same global minimisers)
• x minimiser of GwCEL0 ⇒x minimiser of Gwℓ0 (less local minimisers).
+ Minimisation with IRℓ1.
34


---

## 第52页

Zoom on a detail
35


---

## 第53页

Zoom on a detail
35


---

## 第54页

Conclusions
We focused on models and algorithms tackling the ℓ2-ℓ0 minimisation problem.
• NP-hardness is avoided by alternative formulations
• Greedy approaches provide interesting results, at the price of increased complexity
• Continuous relaxations (both convex and non-convex) ease the problem
• CEL0 is the “best” (liminf) continuous, non-convex relaxation, and it is exact.
• A MM strategy such as IRℓ1 can be used. Fast convex optimisation is here
essential for solving inner problems with high precision.
• Application areas are vast: inverse problems in imaging, vision, variable selection
in machine learning. . .
36


---

## 第55页

Interested in a PostDoc (or PhD) in optimisation?
Task-adaptive bilevel learning of ﬂexible statistical models for imaging and vision
(2023-2027)
• 2-year post-doctoral position (open)
• 1 PhD position (from October 2023)
37


---

## 第56页

Announcement II: SSVM 2023
• What? IX conference on Scale Space and Variational Methods in Computer
Vision (SSVM).
• Where? Hotel Flamingo, Santa Margherita di Pula, Sardegna, IT.
• When? May 21-25 2023
• Who? Giunta Gruppo UMI MIVA + G. Rodriguez (local organiser)
• Why Oral + poster session of selected papers (published in Springer LNCS)
Website: SSVM 2023
NEW DEADLINE for submissions: January 30 2023
38


---

## 第57页

Questions?
calatroni@i3s.unice.fr
38


---

