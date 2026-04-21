# JGondzio-Lecture5.pdf

## 第1页

J. Gondzio
L5: Sparse Approximations with IPMs
School of Mathematics
T
H
E
U
N
I
V
E
R
S
I
T
Y
O
F
E
D
I
N
B
U
R
G
H
Applications of IPMs:
From Sparse Approximations
to Discrete Optimal Transport
Jacek Gondzio
Email: J.Gondzio@ed.ac.uk
URL: http://www.maths.ed.ac.uk/~gondzio
Bologna, January 2023
1


---

## 第2页

J. Gondzio
L5: Sparse Approximations with IPMs
Outline
• Motivation: sparsity, a desired feature
−→for example, ℓ1-regularized least squares (LASSO)
• 1st-order vs 2nd-order methods
• Inexact Newton method
– How much of Hessian information is needed?
– Iterative methods with suitable preconditioners
−→Newton Conjugte Gradients
−→(Inexact) Interior Point Methods
• Applications
• Conclusions
Bologna, January 2023
2


---

## 第3页

J. Gondzio
L5: Sparse Approximations with IPMs
Sparse Approximations
• Statistics: Estimate x from observations
• Machine Learning: Classiﬁcations, SVMs, etc
• Inverse Problems
• Wavelet-based signal/image reconstruction & restoration
• Compressed Sensing (Signal Processing)
Such problems lead to some dense, often structured, possibly very
large optimization instances (LP, QP or NLP):
minx f(x) + τ1∥x∥1 + τ2∥Lx∥1
s.t.
Ax = b.
Cutting-edge optimization techniques are needed!
Plethora of highly specialised 1st-order methods exist.
Work of Yu. Nesterov, S. Wright and an army of followers.
Bologna, January 2023
3


---

## 第4页

J. Gondzio
L5: Sparse Approximations with IPMs
1st-order methods vs 2nd-order methods
The 2nd-order methods are sometimes criticised as unsuitable:
“computing/using the 2nd-order information is too expensive”.
An unfounded criticism based on an unfair comparison:
specialised 1st-order methods compared with
general (of-the-shelf) 2nd-order methods.
The 1st-order methods have clear drawbacks:
• they struggle with accuracy, and
• they work only for trivial, well conditioned problems.
The specialised 2nd-order methods
overcome these drawbacks and are very competitive.
This talk will demonstrate why.
Bologna, January 2023
4


---

## 第5页

J. Gondzio
L5: Sparse Approximations with IPMs
ℓ1-regularization
min
x
f(x) + τ∥x∥1.
Think of LASSO:
min
x
∥Ax −b∥2
2 + τ∥x∥1.
Unconstrained optimization
⇒easy
Serious Issue: nondiﬀerentiability of
∥.∥1
Two possible tricks:
• Splitting x = u −v with u, v ≥0
• Smoothing with pseudo-Huber approximation
replaces ∥x∥1 with ψµ(x) = Pn
i=1(
q
µ2 + x2
i −µ)
Bologna, January 2023
5


---

## 第6页

J. Gondzio
L5: Sparse Approximations with IPMs
Huber:
Bologna, January 2023
6


---

## 第7页

J. Gondzio
L5: Sparse Approximations with IPMs
Continuation
Embed inexact Newton Method into a homotopy approach:
• Inequalities u ≥0, v ≥0
−→use IPM
replace z ≥0 with −µ logz and drive µ to zero.
• Pseudo-Huber regression
−→use continuation
replace |xi| with µ(
r
1+x2
i
µ2 −1) and drive µ to zero.
Questions:
• How?
• Theory?
• Practice?
Bologna, January 2023
7


---

## 第8页

J. Gondzio
L5: Sparse Approximations with IPMs
How: Use approximate Hessian
Use 2nd-order information (Newton direction).
But, do not waste time on computing exact direction.
Use Inexact Newton Method
Dembo, Eisenstat and Steihaug,
Inexact Newton Methods,
SIAM J. on Numerical Analysis 19 (1982) 400–408.
Bellavia, Inexact Interior Point Method,
Journal of Optimization Theory and Appls 96 (1998) 109–121.
Bologna, January 2023
8


---

## 第9页

J. Gondzio
L5: Sparse Approximations with IPMs
Inexact Newton Method
Replace an exact Newton direction
∇2f(x)∆x = −∇f(x)
with an inexact one:
∇2f(x)∆x = −∇f(x) + r,
where the error r is small: ∥r∥≤η∥∇f(x)∥, η ∈(0, 1).
Use iterative methods of linear algebra:
• Continuation →Newton CG
• IPMs →Inexact IPM →Iterative schemes for KKT systems
Bologna, January 2023
9


---

## 第10页

J. Gondzio
L5: Sparse Approximations with IPMs
IMPs: Theorem: Suppose the feasible IPM for QP is used.
If the method operates in the small neighbourhood
N2(θ) := {(x, y, s) ∈F0 : ∥XSe −µe∥2 ≤θµ}
and uses the inexact Newton direction with
η = 0.3, then it
converges in at most
K = O(√n ln(1/ǫ))
iterations.
If the method operates in the symmetric neighbourhood
NS(γ) := {(x, y, s) ∈F0 : γµ ≤xisi ≤(1/γ)µ}
and uses the inexact Newton direction with η = 0.05, then it
converges in at most
K = O(n ln(1/ǫ))
iterations.
Gondzio, Convergence Analysis of an Inexact Feasible IPM for Convex Quadratic Programming,
SIAM Journal on Optimization 23 (2013) No 3, pp. 1510-1527.
Bologna, January 2023
10


---

## 第11页

J. Gondzio
L5: Sparse Approximations with IPMs
Continuation: Compressed Sensing Case
Replace
min
x
f(x) = τ∥W Tx∥1 + 1
2∥Ax −b∥2
2,
−→xτ
with
min
x
fµ(x) = τψµ(W Tx) + 1
2∥Ax −b∥2
2,
−→xτ,µ
Solve approximately a family of problems for a (short) decreasing
sequence of µ’s: µ0 > µ1 > µ2 · · ·
Theorem (Brief description)
There exists a ˜µ such that ∀µ ≤˜µ the diﬀerence of the two solutions
satisﬁes
∥xτ,µ −xτ∥2 = O(µ1/2)
∀τ, µ.
Primal-Dual Newton Conjugate Gradient Method:
Fountoulakis and Gondzio, A Second-order Method for Strongly Convex ℓ1-regularization Problems,
Mathematical Programming, 156 (2016) 189–219.
Dassios, Fountoulakis and Gondzio, A Preconditioner for a Primal-Dual Newton Conjugate Gradient
Method for Compressed Sensing Problems, SIAM J on Scientiﬁc Computing, 37 (2015) A2783–A2812.
Bologna, January 2023
11


---

## 第12页

J. Gondzio
L5: Sparse Approximations with IPMs
Examples
Bologna, January 2023
12


---

## 第13页

J. Gondzio
L5: Sparse Approximations with IPMs
Examples of ℓ1-regularization
• Compressed Sensing
with K. Fountoulakis and P. Zhlobich
min
x
τ∥x∥1 + 1
2∥Ax −b∥2
2,
A ∈Rm×n
• Compressed Sensing (Coherent and Redundant Dict.)
with I. Dassios and K. Fountoulakis
min
x
τ∥W ∗x∥1 + 1
2∥Ax −b∥2
2,
W ∈Cn×l, A ∈Rm×n
think of Total Variation
• Big Data optimization (Machine Learning), LASSO
with K. Fountoulakis
Bologna, January 2023
13


---

## 第14页

J. Gondzio
L5: Sparse Approximations with IPMs
Example 1: Compressed Sensing
with K. Fountoulakis and P. Zhlobich
Large dense quadratic optimization problem:
min
x
τ∥x∥1 + 1
2∥Ax −b∥2
2,
where A ∈Rm×n is a very special matrix.
Fountoulakis, Gondzio, Zhlobich
Matrix-free IPM for Compressed Sensing Problems,
Mathematical Programming Computation 6 (2014), pp. 1–31.
Dassios, Fountoulakis, Gondzio
A Preconditioner for a Primal-Dual Newton Conjugate Gradient Method for Compressed Sensing Problems,
SIAM J on Scientiﬁc Computing 37 (2015) A2783–A2812.
Software available at http://www.maths.ed.ac.uk/ERGO/
Bologna, January 2023
14


---

## 第15页

J. Gondzio
L5: Sparse Approximations with IPMs
Restricted Isometry Property (RIP)
• rows of A are orthogonal to each other (A is built of a subset
of rows of an othonormal matrix U ∈Rn×n)
AAT = Im.
• small subsets of columns of A are nearly-orthogonal to each
other: Restricted Isometry Property (RIP)
∥¯AT ¯A −m
n Ik∥≤δk ∈(0, 1).
Cand`es, Romberg and Tao,
Stable Signal Recovery from
Incomplete and Inaccurate Measurements,
Comm on Pure and Applied Mathematics 59 (2006) 1207-1233.
Bologna, January 2023
15


---

## 第16页

J. Gondzio
L5: Sparse Approximations with IPMs
Restricted Isometry Property
Matrix ¯A ∈Rm×k (k ≪n) is built of a subset of columns
of A ∈Rm×n.
A =
−→
¯A =
¯AT ¯A =
= 
≈m
n Ik.
This yields a very well conditioned optimization problem.
Bologna, January 2023
16


---

## 第17页

J. Gondzio
L5: Sparse Approximations with IPMs
Problem Reformulation
min
x
τ∥x∥1 + 1
2∥Ax −b∥2
2
Replace x = x+ −x−to be able to use |x| = x+ + x−.
Use |xi| = zi + zi+n to replace ∥x∥1 with ∥x∥1 = 1T
2nz.
(Increases problem dimension from n to 2n.)
min
z≥0 cTz + 1
2zTQz,
where
Q =

AT
−AT

[ A −A ] =

ATA −ATA
−ATA
ATA

∈R2n×2n
Bologna, January 2023
17


---

## 第18页

J. Gondzio
L5: Sparse Approximations with IPMs
Preconditioner
Approximate
M =

ATA −ATA
−ATA
ATA

+
"
Θ−1
1
Θ−1
2
#
with
P = m
n

In −In
−In
In

+
"
Θ−1
1
Θ−1
2
#
.
We expect (optimal partition):
• k entries of Θ−1 →0,
k ≪2n,
• 2n −k entries of Θ−1 →∞.
Bologna, January 2023
18


---

## 第19页

J. Gondzio
L5: Sparse Approximations with IPMs
Spectral Properties of P−1M
Theorem
• Exactly n eigenvalues of P−1M are 1.
• The remaining n eigenvalues satisfy
|λ(P−1M) −1| ≤δk +
n
mδkL,
where δk is the RIP-constant, and
L is a threshold of “large” (Θ1 + Θ2)−1.
Fountoulakis, Gondzio, Zhlobich
Matrix-free IPM for Compressed Sensing Problems,
Mathematical Programming Computation 6 (2014), pp. 1–31.
Bologna, January 2023
19


---

## 第20页

J. Gondzio
L5: Sparse Approximations with IPMs
Preconditioning
2
4
6
8
10
12
14
16
18
20
22
10
0
10
1
10
2
10
3
 Number of CG/PCG call
 MatVecs
 Matrix−vector products per CG/PCG call
 
 
Unpreconditioned CG
Preconditioned CG
2
4
6
8
10
12
14
16
18
20
22
10
−6
10
−4
10
−2
10
0
10
2
10
4
10
6
10
8
 number of CG/PCG call
 magnitude of λ(M)/λ(P−1M)
 Spread of λ(M)/λ(P−1M) per call of CG/PCG
 
 
λ(M)
λ(P−1M)
−→good clustering of eigenvalues
mf-IPM compares favourably with NestA on easy probs
(NestA: Becker, Bobin and Cand´es).
Bologna, January 2023
20


---

## 第21页

J. Gondzio
L5: Sparse Approximations with IPMs
Example 2: Simple test for ℓ1-regularization
min
x
τ∥x∥1 + ∥Ax −b∥2
2
Special matrix given in SVD form A = UΣV T, where U and V are
products of Givens rotations. The user controls:
• the condition number κ(A),
• the sparsity of matrix A.
Matlab generator:
https://www.maths.ed.ac.uk/ERGO/trillion/
Fountoulakis and Gondzio
Performance of First- and Second-Order Methods for ℓ1-regularized Least Squares Problems,
Computational Optimization and Applications 65 (2016) 605–635.
Bologna, January 2023
21


---

## 第22页

J. Gondzio
L5: Sparse Approximations with IPMs
Excessive Computational Tests (4 mths of CPU)
• FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
• PCDM (Parallel Coordinate Descent Method)
• PSSgb (Projected Scaled Subgradient, Gafni-Bertsekas)
• pdNCG (primal-dual Newton Conjugate Gradient)
The 1st order methods:
• work well if the condition number κ(A) ≤102,
• struggle when κ(A) ≥103,
• stall when κ(A) ≥104.
The 2nd order method (pdNCG, diagonal preconditioner):
• works well if the condition number κ(A) ≤106.
Bologna, January 2023
22


---

## 第23页

J. Gondzio
L5: Sparse Approximations with IPMs
Let us go big: a trillion (240 ≈1012) variables
n (billions) Processors
Memory (TB)
time (s)
1
64
0.192
1923
4
256
0.768
1968
16
1024
3.072
1986
64
4096
12.288
1970
256
16384
49.152
1990
1,024
65536
196.608
2006
ARCHER (ranked 25 on top500.com, 11 March 2015)
Linpack Performance (Rmax) 1,642.54 TFlop/s
Theoretical Peak (Rpeak)
2,550.53 TFlop/s
Fountoulakis and Gondzio
Performance of First- and Second-Order Methods for ℓ1-regularized Least Squares Problems,
Computational Optimization and Applications 65 (2016) 605–635.
Bologna, January 2023
23


---

## 第24页

J. Gondzio
L5: Sparse Approximations with IPMs
More Examples of Sparse Approximations
• Sparse Approximations with IPMs
→ℓ1-regularized problems,
work with V. De Simone, D. di Seraﬁno,
S. Pougkakiotis, M. Viola
• Discrete Optimal Transport with IPMs
→large, but highly structured,
work with F. Zanetti
Bologna, January 2023
24


---

## 第25页

J. Gondzio
L5: Sparse Approximations with IPMs
More Sparse Approximations: Use IPMs
Problems of the form
min f(x) + τ1∥x∥1 + τ2∥Lx∥1
s.t.
Ax = b.
• Sparse portfolio selection
comparison with Split Bregman method
• Classiﬁcation models for funct’l Magnetic Resonance Imaging
comparison with FISTA and ADMM
• TV-based Poisson Image Restoration
comparison with PDAL
• Linear Classiﬁcation via Regularized Logistic Regression
comparison with newGLMNET and ADMM
De Simone, di Seraﬁno, Gondzio, Pougkakiotis, Viola,
Sparse Approximations with Interior Point Methods,
SIAM Review 64 (2022) pp. 954–988.
https://arxiv.org/abs/2102.13608
Bologna, January 2023
25


---

## 第26页

J. Gondzio
L5: Sparse Approximations with IPMs
Example 3: Binary Classiﬁcation of fMRI Data
min
w
1
2s ∥Dw −ˆy∥2 + τ1 ∥w∥1 + τ2 ∥Lw∥1
where: τ1, τ2 > 0,
∥Lw∥1 is a discrete anisotropic TV of w,
and L = [LTx LTy LTz ]T ∈Rl×q are the ﬁrst-order forward ﬁnite
diﬀerences in x, y, z.
Baldassarre, Pontil & Moura˜o-Miranda,
Sparsity Is Better with Stability: Combining Accuracy and Stability for Model Selection in Brain Decoding,
Frontiers of Neuroscience 2017. https://doi.org/10.3389/fnins.2017.00062
Bologna, January 2023
26


---

## 第27页

J. Gondzio
L5: Sparse Approximations with IPMs
Classiﬁcation models for fMRI
Comparison of IPM, FISTA and ADMM (opt tol 10−5). We report:
• classiﬁcation accuracy (ACC),
• corrected pairwise overlap (CORR OVR);
measures the “stability” of each voxel selection,
• solution density (DEN).
Algorithm
τ1 = τ2
ACC
CORR OVR
DEN
IP-PMM
10−2
86.16 ± 7.11
43.47 ± 9.09
20.56 ± 6.63
5 · 10−2
84.90 ± 4.80
62.70 ± 10.39
3.77 ± 0.84
10−1
82.29 ± 6.22
82.60 ± 9.24
2.49 ± 0.34
FISTA
10−2
86.90 ± 5.01
5.43 ± 0.43
88.97 ± 0.71
5 · 10−2
84.15 ± 5.92
65.50 ± 2.68
19.36 ± 0.86
10−1
81.62 ± 7.58
80.44 ± 5.72
5.14 ± 0.44
ADMM
10−2
86.46 ± 6.91
0.03 ± 0.01
98.70 ± 0.03
5 · 10−2
85.57 ± 5.37
0.15 ± 0.04
97.97 ± 0.05
10−1
82.07 ± 6.51
0.26 ± 0.13
97.50 ± 0.19
We want: ACC and CORR OVR close to 100, and small DEN.
Bologna, January 2023
27


---

## 第28页

J. Gondzio
L5: Sparse Approximations with IPMs
Classiﬁcation models for fMRI (cont’d)
Performance comparison in terms of elapsed time:
Evolution of ACC, DEN and CORR OVR with time;
IP-PMM (left) and FISTA (right).
We report average measures with 95% conﬁdence intervals.
Bologna, January 2023
28


---

## 第29页

J. Gondzio
L5: Sparse Approximations with IPMs
Optimal Transport
Signiﬁcant research interest:
Gaspard Monge (1781)
Leonid Kantorovich (1942) Nobel Prize in 1975
Alessio Figalli (2008) Fields Medal in 2018
Good reading:
F. Santambrogio,
Optimal Transport for Applied Mathematicians, Birkhauser Basel,
2016.
G. Peyr´e and M. Cuturi,
Computational
Optimal
Transport: With Applications to Data
Science. Foundations and Trends in Machine Learning 11 (2019)
No 5-6, pp. 355–607.
Bologna, January 2023
29


---

## 第30页

J. Gondzio
L5: Sparse Approximations with IPMs
Example 4: Discrete Optimal Transport
Kantorovich formulation of the discrete Optimal Transport problem:
given a starting vector a ∈Rm
+ and a ﬁnal vector b ∈Rn+,
such that P aj = P bj, ﬁnd a coupling matrix P inside the set
U(a, b) =
n
P ∈Rm×n
+
, Pen = a, PTem = b
o
that is optimal with respect to a certain cost matrix C ∈Rm×n
+
;
i.e. ﬁnd the solution of the following optimization problem
min
P∈U(a,b)
X
i,j
CijPij.
Move the mass in the conﬁguration a into the conﬁguration b.
Bologna, January 2023
30


---

## 第31页

J. Gondzio
L5: Sparse Approximations with IPMs
Discrete Optimal Transport (cont’d)
We can rewrite the optimization problem as a standard LP:
min
p∈Rmn
cTp
s.t.

eTn ⊗Im
In ⊗eTm

p =

a
b

= f,
p ≥0,
where ⊗denotes the Kronecker product,
c ∈Rmn and p ∈Rmn are the vectorized versions of C and P,
respectively, c = vec(C) and p = vec(P).
LP with m + n constraints and m × n variables.
Zanetti and Gondzio,
A Sparse Interior Point Method for Linear Programs arising in Discrete Optimal Transport,
(submitted 22 Jun 2022, revised 6 Dec 2022).
https://arxiv.org/abs/2206.11009
Bologna, January 2023
31


---

## 第32页

J. Gondzio
L5: Sparse Approximations with IPMs
Small OT Example
Move the mass in the red conﬁguration into the blue conﬁguration.
Right ﬁgure: the corresponding bipartite graph. →Sparse solution!
Bologna, January 2023
32


---

## 第33页

J. Gondzio
L5: Sparse Approximations with IPMs
IPM Specialized for Discrete OT Problems
• Ignore “long” matrix A
−→use column-generation-type approach
• Work with expected “sparse” solution set
−→do not update all variables x
• Use simplex-type pricing mechanism
−→update dual slacks only for a subset of variables x
• Simplify normal equations
−→replace PN
j=1 θjAjAT
j with P
j∈S θjAjAT
j ,
where S is a likely “sparse” solution set
• Precondition Cholesky matrix of the normal equations
−→keep it sparse at all times
Bologna, January 2023
33


---

## 第34页

J. Gondzio
L5: Sparse Approximations with IPMs
Test examples from DOTmark collection
Class 1
Class 2
Class 3
Class 4
Class 5
Class 6
Class 7
Class 8
Class 9
Class 10
For the resolution r, the LP has 2r2 constraints and r4 variables.
For r =
32:
2,048 constraints and
1 million variables;
For r =
64:
8,192 constraints and 16.8 million variables;
For r = 128:
32,768 constraints and 268.4 million variables;
For r = 256: 131,072 constraints and 4.295 billion variables.
Bologna, January 2023
34


---

## 第35页

J. Gondzio
L5: Sparse Approximations with IPMs
Discrete Optimal Transport (cont’d)
DOTmark test collection:
Schrieber, Schuhmacher, and Gottschlich,
DOTmark - A Benchmark for Discrete Optimal Transport, IEEE Access, 5 (2017), pp. 271–282.
Softwares compared:
• Cuturi,
Sinkhorn distances: Lightspeed computation of optimal transport,
Proc. NIPS, (2013), pp. 2292–2300.
• Gottschlich and Schuhmacher,
The Shortlist Method for Fast Computation of the Earth Mover’s Distance and Finding Optimal
Solutions to Transportation Problems,
PLoS ONE, 9 (2014), p. e110214.
• Merigot,
A Multiscale Approach to Optimal Transport,
Computer Graphics Forum, 30 (2011), pp. 1583–1592.
• Network Simplex Method, IBM ILOG CPLEX.
https://www.ibm.com/analytics/cplex-optimizer.
• Kovacs,
Minimum-cost ﬂow algorithms: An experimental evaluation, OMS, 30(1):94–127.
https://lemon.cs.elte.hu/trac/lemon.
Bologna, January 2023
35


---

## 第36页

J. Gondzio
L5: Sparse Approximations with IPMs
Comparison: SparseIPM vs Cplex Network
Res = 32 × 32
Res = 64 × 64
Class Iter IPM t Cplex t
RWE Iter IPM t Cplex t
RWE
1
11.4
0.35
0.62 1.2e-07 14.4
2.18
20.92 5.5e-08
2
11.7
0.39
0.60 1.4e-07 18.1
3.46
20.64 4.5e-08
3
15.9
0.59
0.61 2.4e-08 26.8
6.02
20.83 2.1e-08
4
20.3
0.85
0.57 2.0e-08 38.4
9.69
20.69 2.1e-08
5
25.6
1.16
0.61 1.4e-08 40.8
10.78
21.84 1.6e-08
6
18.8
0.72
0.64 3.3e-08 36.2
9.04
23.25 1.3e-08
7
30.8
1.47
0.57 3.8e-08 72.2
39.11
21.80 2.3e-08
8
17.4
0.65
0.58 3.8e-08 52.5
21.69
18.55 8.7e-08
9
14.9
0.52
0.60 2.8e-08 25.0
5.24
21.27 1.4e-08
10
22.4
0.92
0.62 2.0e-08 40.8
10.48
18.33 2.1e-08
Bologna, January 2023
36


---

## 第37页

J. Gondzio
L5: Sparse Approximations with IPMs
CPU time of SparseIPM (1−norm, 128 pixels)
m = 2r2
n = r4
Bologna, January 2023
37


---

## 第38页

J. Gondzio
L5: Sparse Approximations with IPMs
Comparison: Scalability of three solvers
SparseIPM for Discrete OT
Cplex (Simplex Method for Network Problems)
LEMON (Specialized Network Algorithm)
Bologna, January 2023
38


---

## 第39页

J. Gondzio
L5: Sparse Approximations with IPMs
Overarching Feature of IPMs
They possess an unequalled ability to identify
the “essential subspace”
in which the optimal solution is hidden.
Bologna, January 2023
39


---

## 第40页

J. Gondzio
L5: Sparse Approximations with IPMs
Conclusions
2nd-order methods for optimization:
• employ inexact Newton method
• rely on preconditioners
• enjoy matrix-free implementation
Trick:
• ﬁnd the “essential subspace” and
• exploit it to simplify the linear algebra
– works in IPMs for LP
– works in Newton CG for ℓ1-regularization
Simple, reliable test example for ℓ1-regularization:
http://www.maths.ed.ac.uk/ERGO/trillion/
Bologna, January 2023
40


---

