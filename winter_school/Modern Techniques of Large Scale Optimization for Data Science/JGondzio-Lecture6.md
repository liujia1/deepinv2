# JGondzio-Lecture6.pdf

## 第1页

J. Gondzio
L6: ADMM
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
Alternating Direction
Method of Multipliers (ADMM)
Jacek Gondzio
Email: J.Gondzio@ed.ac.uk
URL: http://www.maths.ed.ac.uk/~gondzio
Bologna, January 2023
1


---

## 第2页

J. Gondzio
L6: ADMM
Alternating Direction Method of Multipliers
(ADMM)
• Exploits Duality
• Has inexpensive iterations
• Suitable for problems with loosely coupled variables
• Numerous applications:
– machine learning/statistics (large data sets),
– image processing,
– decentralized optimization.
Bologna, January 2023
2


---

## 第3页

J. Gondzio
L6: ADMM
Dual Decomposition
Consider equality constrained optimization problem
min
f(x)
s.t.
Ax = b
where x∈Rn, f : Rn7→R, A∈Rm×n, b∈Rm. Usually m≤n.
In this lecture f is convex. (In general it does not have to be.)
We associate Lagrange multipliers y ∈Rm with equality constraints
Ax = b and write the Lagrangian:
L(x, y) = f(x) + yT(Ax −b),
dual function:
LD(y) = inf
x L(x, y)
and dual problem:
max
y
LD(y).
Having found the solution of the dual at ˆy, we recover
ˆx = argminxL(x, ˆy).
Bologna, January 2023
3


---

## 第4页

J. Gondzio
L6: ADMM
Dual Ascent Method
Apply a (simple) gradient method for the dual problem, i.e. make
steps in direction of ∇LD(y):
yk+1 = yk + αk∇LD(yk).
Observe that
∇LD(yk) = A˜x −b,
where ˜x = argminxL(x, yk).
Dual Ascent Method:
repeat until optimality is reached:
xk+1 = argminxL(x, yk)
minimize in x
yk+1 = yk + αk(Axk+1 −b) update Lagrange multipliers y
Theory:
Strong assumptions are required for such a simple method to work.
Bologna, January 2023
4


---

## 第5页

J. Gondzio
L6: ADMM
Dual Decomposition and Separable Objective
Suppose the objective function is separable:
f(x) = f1(x1) + f2(x2) + · · · + fp(xp),
x = (x1, x2, . . . , xp).
Rewrite the problem in the following form:
min
f1(x1) + f2(x2) + · · · + fp(xp)
s.t.
A1x1 + A2x2 + · · · + Apxp = b
and observe that the Lagrangian is separable in x:
L(x, y) = L1(x1, y) + L2(x2, y) + · · · + Lp(xp, y) −yTb,
where Li(xi, y) = fi(xi) + yTAixi, i = 1, 2, . . . , p.
Hence the minimization in x may be split into p separate tasks:
xk+1
i
= argminxiLi(xi, yk), i = 1, 2, . . . , p
which do not depend on each other, and may be executed in parallel.
Bologna, January 2023
5


---

## 第6页

J. Gondzio
L6: ADMM
Dual Decomposition in Separable Case
Dual Ascent Method (Separable Case):
repeat until optimality is reached:
xk+1
i
= argminxiLi(xi, yk),
i = 1, 2, . . . , p,
yk+1 = yk + αk(Pp
i=1 Aixk+1
i
−b)
‘Decomposition’ because we decompose a large problem into pieces.
Two widely used decomposition schemes rely on such a framework:
• Dantzig-Wolfe Decomposition (1960)
• Benders Decomposition (1961)
−→essential tools for solving combinatorial optimization problems.
They have a weakness though: they may be slow.
Bologna, January 2023
6


---

## 第7页

J. Gondzio
L6: ADMM
Method of Multiplers
An identiﬁable weakness of Dual Decomposition is the diﬃculty to
satisfy the constraint Ax = b. This may be addressed by giving this
constraint a more prominent role and adding to the Lagrangian the
quadratic penalty of the constraint violation.
Deﬁne the Augmented Lagrangian:
Lρ(x, y) = f(x) + yT(Ax −b) + ρ
2∥(Ax −b)∥2,
where ρ is a weight of the penalty.
This gives the Method of Multipliers (Hestenes, 1969, Powell,
1969):
repeat until optimality is reached:
xk+1 = argminxLρ(x, yk)
yk+1 = yk + ρ(Axk+1 −b)
The method is similar to the dual ascent.
It minimizes Lρ(x, yk) instead of L(x, yk)
and uses a ﬁxed dual update stepsize ρ instead of α.
Bologna, January 2023
7


---

## 第8页

J. Gondzio
L6: ADMM
Convergence of the Method of Multiplers
If the objective function in the optimization problem
min f(x)
s.t. Ax = b
is diﬀerentiable, then the optimality conditions are:
∇f(ˆx) + AT ˆy = 0
dual feasibility
Aˆx = b
primal feasibility
Observe that since xk+1 minimizes Lρ(x, yk), the dual update
yk+1 = yk + ρ(Axk+1 −b)
ensures that (xk+1, yk+1) is dual feasible. Indeed:
0 = ∇xLρ(xk+1, yk) = ∇xf(xk+1) + AT(yk + ρ(Axk+1 −b))
= ∇xf(xk+1) + ATyk+1.
However, the primal feasibility is attained only in the limit
Axk+1 −b →0.
Bologna, January 2023
8


---

## 第9页

J. Gondzio
L6: ADMM
Method of Multiplers vs Dual Decomposition
Method of Multiplers:
• converges under more relaxed assumptions
(f can be nondiﬀerentiable)
• deals better with the primal feasibility Ax −b
(presence of ∥Ax −b∥2 in the Augmented Lagrangian helps)
but
• the quadratic penalty ∥Ax −b∥2 in Lρ destroys separability
→cannot be used in decomposition.
Alternating Direction Method of Multipliers
(ADMM)
ADMM oﬀers a compromise:
• enjoys some of the beneﬁts of the method of multipliers
• is well-suited to decomposition
Gabay and Mercier (1976), Glowinski and Marrocco (1975).
Bologna, January 2023
9


---

## 第10页

J. Gondzio
L6: ADMM
Alternating Direction Method of Multipliers
(ADMM)
Consider a problem in the following form:
min
f1(x1) + f2(x2)
s.t.
A1x1 + A2x2 = b,
where f1 : Rn1 7→R, and f2 : Rn2 7→R are convex functions (do
not have to be diﬀerentiable), Ai ∈Rm×ni, i = 1, 2, b ∈Rm.
Observe that the objective is separable, but the constraint links x1
and x2.
Write down the associated Augmented Lagrangian:
Lρ(x1, x2, y) = f1(x1) + f2(x2) + yT(A1x1+A2x2−b)
(1)
+ρ
2∥(A1x1+A2x2−b)∥2.
Bologna, January 2023
10


---

## 第11页

J. Gondzio
L6: ADMM
Alternating Direction Method of Multipliers
(ADMM)
repeat until optimality is reached:
xk+1
1
= argminx1 Lρ(x1, xk
2, yk)
minimize in x1
xk+1
2
= argminx2 Lρ(xk+1
1
, x2, yk)
minimize in x2
yk+1 = yk + ρ(A1xk+1
1
+ A2xk+1
2
−b)
update multipliers y
Observe that the optimization in x1 uses the “old” xk
2, but
the optimization in x2 uses already the “new” (updated) xk+1
1
.
Bologna, January 2023
11


---

## 第12页

J. Gondzio
L6: ADMM
Convergence of the ADMM If the functions f1 and
f2 in the objective are diﬀerentiable, then the optimality conditions
are:
∇f1(ˆx1) + AT
1 ˆy = 0
1st dual feasibility
∇f2(ˆx2) + AT
2 ˆy = 0
2nd dual feasibility
A1ˆx1 + A2ˆx2 = b
primal feasibility
Note that since xk+1
2
minimizes Lρ(xk+1
1
, x2, yk), the dual update
yk+1 = yk + ρ(Axk+1 −b)
guarantees that (xk+1
1
, xk+1
2
, yk+1) satisﬁes the 2nd dual feasibility
constraint. Indeed:
0 = ∇x2f2(xk+1
2
) + AT
2 yk + ρAT
2 (A1xk+1
1
+ A2xk+1
2
−b)
= ∇x2f2(xk+1
2
) + AT
2 (yk + ρ(Axk+1 −b))
= ∇x2f2(xk+1
2
) + AT
2 yk+1.
However, the 1st dual feasibility and primal feasibility are attained
only in the limit (at convergence).
Bologna, January 2023
12


---

## 第13页

J. Gondzio
L6: ADMM
Convergence of the ADMM
Consider a problem in the following form:
min F(x1, x2) = f1(x1) + f2(x2)
s.t.
A1x1 + A2x2 = b,
where f1 : Rn1 7→R, and f2 : Rn2 7→R are convex functions (do
not have to be diﬀerentiable), Ai ∈Rm×ni, i = 1, 2, b ∈Rm.
Bologna, January 2023
13


---

## 第14页

J. Gondzio
L6: ADMM
Convergence of the ADMM
Theorem. Suppose f1 and f2 are closed convex functions, and γ
is any constant which satisﬁes γ > 2∥ˆy∥2. Then
F(xt
1, xt
2) −F(ˆx1, ˆx2) ≤
∥x0
2−ˆx2∥2
ρAT
2 A2
+ (γ+∥y0∥2)2/ρ
2(t+1)
∥A1xt
1 + A2xt
2 −b∥2 ≤
∥x0
2−ˆx2∥2
ρAT
2 A2
+ (γ+∥y0∥2)2/ρ
γ(t+1)
,
where xt
1 :=
1
t+1
Pt+1
k=1 xk
1, xt
2 :=
1
t+1
Pt+1
k=1 xk
2,
and for C ⪰0, we deﬁne ∥u∥2
C := uTC u.
Theoretical convergence of ADMM is slow:
O(1/t) convergence rate and O(1/ǫ) iteration complexity.
(To compare: IPMs enjoy O(log(1/ǫ)) iteration complexity.)
Bologna, January 2023
14


---

## 第15页

J. Gondzio
L6: ADMM
ADMM: From 2 blocks to p blocks
Consider a problem in the following form:
min
Pp
i=1 fi(xi)
s.t.
Pp
i=1 Aixi = b,
where fi : Rni 7→R, i = 1, 2, . . . , p, are convex functions (do not
have to be diﬀerentiable), Ai ∈Rm×ni, i = 1, 2, . . . , p, b ∈Rm.
Observe that the objective is separable, but the constraint links all
the variables xi.
Write down the associated Augmented Lagrangian:
Lρ(x1, x2, . . . , xp,y) =
p
X
i=1
fi(xi) + yT(
p
X
i=1
Aixi−b)
+ρ
2∥(
p
X
i=1
Aixi−b)∥2.
Bologna, January 2023
15


---

## 第16页

J. Gondzio
L6: ADMM
ADMM: From 2 blocks to p blocks
Multiple block version of ADMM:
repeat until optimality is reached:
xk+1
1
= argminx1 Lρ(x1, xk
2, . . . , xkp, yk)
minimize in x1
xk+1
2
= argminx2 Lρ(xk+1
1
, x2, . . . , xkp, yk)
minimize in x2
...
...
xk+1
p
= argminxp Lρ(xk+1
1
, xk+1
2
, . . . , xp, yk)
minimize in xp
yk+1 = yk + ρ(Pp
i=1 Aixk+1
i
−b)
update
y
Bologna, January 2023
16


---

## 第17页

J. Gondzio
L6: ADMM
Comments on Convergence
While (under suitable assumptions) the 2-block ADMM is proved
to converge, the p-block version does not have to converge, see:
C. Chen, B. He, Y. Ye, X. Yuan, “The direct extension of
ADMM for multi-block convex minimization problems is not neces-
sarily convergent”, Mathematical Prog A 155 (2016) pp. 57–79.
Example with null objective:
min
0
s.t.
A1x1 + A2x2 + A3x3 = 0,
where
A1 =
" 1
1
1
#
,
A2 =
" 1
1
2
#
,
A3 =
" 1
2
2
#
.
Observe that A = [A1, A2, A3] is nonsingular. Since the right-hand-
side b = 0, the feasible set contains a single element ˆx1 = ˆx2 = ˆx3 =
0. Since the objective is null, the optimal Lagrange multiplier ˆy = 0.
The 3-block ADMM is divergent for this problem.
Bologna, January 2023
17


---

## 第18页

J. Gondzio
L6: ADMM
Applications
ADMM is particularly attractive when the independent minimiza-
tions in xi are signiﬁcantly easier than the minimization of the ag-
gregate objective Pp
i=1 fi(xi).
Sometimes a non-separable problem is converted to a separable one,
just to be able to apply ADMM because of its attractive features,
namely, its ability to make independent optimizations in xi.
Example: Consider a non-separable problem
min f1(x) + f2(x)
s.t.
Ax = b,
in which both functions f1 and f2 depend on the same variable x.
We create a copy of variable x and rewrite the above problem as:
min f1(x) + f2(z)
s.t.
Ax = b
x −z = 0
in a form suitable for ADMM.
Bologna, January 2023
18


---

## 第19页

J. Gondzio
L6: ADMM
Example: ADMM for ℓ1-regularized Least Squares
Recall ℓ1-regularized least squares
min τ∥x∥1 + 1
2∥Ax −b∥2
2,
where A ∈Rm×n, b ∈Rm. Usually m ≥n (and often m ≫n).
This problem may be cast in a form suitable for ADMM:
min τ∥z∥1 + 1
2∥Ax −b∥2
2
s.t.
x −z = 0.
Write down the associated Augmented Lagrangian:
Lρ(x, z, y) = τ∥z∥1 + 1
2∥Ax−b∥2
2 + yT(x−z) + ρ
2∥x−z∥2
2.
Bologna, January 2023
19


---

## 第20页

J. Gondzio
L6: ADMM
Example: ADMM for ℓ1-regularized Least Squares
With such Augmented Lagrangian:
Lρ(x, z, y) = τ∥z∥1 + 1
2∥Ax−b∥2
2 + yT(x−z) + ρ
2∥x−z∥2
2.
Minimization in x exploits the diﬀerentiability of Lρ in x:
∇xLρ(x, z, y) = AT(Ax −b) + ρ(x −z) + y = 0,
which gives
x = (ATA + ρI)−1(ATb + ρz −y).
Minimization in z requires:
min
z

τ∥z∥1 + ρ
2∥z −x −y/ρ∥2
2

,
and is perfectly separable into n independent coordinates:
min
zi

τ|zi| + ρ
2(zi −xi −yi/ρ)2
,
i = 1, 2, . . . n.
Bologna, January 2023
20


---

## 第21页

J. Gondzio
L6: ADMM
Soft Thresholding
In ℓ1-regularized least squares (and in many other applications)
there is a need to perform a one-dimensional update of zi:
z+
i := argminzi

τ|zi| + ρ
2(zi −u)2
,
Although the ﬁrst term is not diﬀerentiable because it involves the
absolute value, we can easily compute a closed-form solution:
z+
i := Sτ/ρ(u),
where the soft thresholding operator S is deﬁned as:
Sτ/ρ(u) =



u −τ/ρ if
u > τ/ρ
0
if |u| ≤τ/ρ
u + τ/ρ if
u < −τ/ρ,
or equivalently:
Sτ/ρ(u) = (u −τ/ρ)+ −(−u −τ/ρ)+,
where v+ := max(v, 0).
Bologna, January 2023
21


---

## 第22页

J. Gondzio
L6: ADMM
Example: ADMM for ℓ1-regularized Least Squares
ℓ1-regularized least squares problem cast in a form suitable for
ADMM:
min τ∥z∥1 + 1
2∥Ax −b∥2
2
s.t.
x −z = 0.
where A ∈Rm×n, b ∈Rm. Usually m ≥n (and often m ≫n).
ADMM:
repeat until optimality is reached:
xk+1 = (ATA + ρI)−1(ATb + ρzk −yk)
zk+1 = Sτ/ρ(xk+1 + yk/ρ)
yk+1 = yk + ρ(xk+1 −zk+1),
where Sτ/ρ(.) is the soft thresholding operator:
Sτ/ρ(u) = (u −τ/ρ)+ −(−u −τ/ρ)+.
The optimization in z is split component-wise and enjoys a trivial
solution.
Bologna, January 2023
22


---

## 第23页

J. Gondzio
L6: ADMM
Example: ADMM for Consensus optimization
Consider a non-separable problem
min Pp
i=1 fi(x)
in which all functions fi, i = 1, 2, . . . p depend on the same variable
x. In machine learning, fi might be the loss function for the i-th
block of training data.
We create p copies of variable x, call them xi, add new constraints
xi = z, ∀i, and then rewrite the above problem as:
min Pp
i=1 fi(xi)
s.t.
xi −z = 0,
i = 1, 2, . . . , p
in a form suitable for ADMM. In this problem:
xi are the local variables, z is a global variable and the constraint
xi −z = 0 forces all (independent) sub-problems to agree on a
common value z, i.e., to reach a consensus.
Bologna, January 2023
23


---

## 第24页

J. Gondzio
L6: ADMM
Example: ADMM for Consensus optimization
(cont’d)
Write down the associated Augmented Lagrangian:
Lρ(x1,x2, ...,xp,z,y1,y2, ...,yp)=
p
X
i=1

fi(xi)+yT
i (xi−z) + ρ
2∥xi−z∥2
.
ADMM:
repeat until optimality is reached:
xk+1
i
= argminxi

fi(xi) + (yk
i )T(xi−zk) + ρ
2∥xi−zk∥2
, i = 1..p
zk+1 = 1
p
Pp
i=1(xk+1
i
+ yk
i /ρ)
yk+1
i
= yk
i + ρ(xk+1
i
−zk+1),
i = 1..p
Observe that averaging is performed in the update of z.
Bologna, January 2023
24


---

## 第25页

J. Gondzio
L6: ADMM
Example: ADMM for QP
Consider a convex quadratic programming problem
min
1
2xTH x + cTx
s.t.
Ax = b,
where x=

x1
x2

∈Rn, xi∈Rni, n1+n2=n, H=

H11 HT
21
H21 H22

∈Rn×n
is a symmetric pos. deﬁnite matrix, A = [A1,A2]∈Rm×n, b∈Rm.
Write down the associated Augmented Lagrangian:
Lρ(x1,x2,y) = 1
2xTH x+cTx+yT(Ax−b)+ρ
2∥(Ax−b)∥2
= 1
2
h
xT
1 , xT
2
i "
H11 + ρAT
1 A1 HT
21 + ρAT
1 A2
H21 + ρAT
2 A1 H22 + ρAT
2 A2
# 
x1
x2

+
+(c1+AT
1 y+ρAT
1 b)Tx1 + (c2+AT
2 y+ρAT
2 b)Tx2 +
+ρ
2bTb −bTy.
Bologna, January 2023
25


---

## 第26页

J. Gondzio
L6: ADMM
Example: ADMM for QP (continued)
Recall the general ADMM:
repeat until optimality is reached:
xk+1
1
= argminx1 Lρ(x1, xk
2, yk)
minimize in x1
xk+1
2
= argminx2 Lρ(xk+1
1
, x2, yk)
minimize in x2
yk+1 = yk + ρ(A1xk+1
1
+ A2xk+1
2
−b)
update multipliers y
For convex QP the ﬁrst two tasks have closed form solutions
∇x1Lρ=(H11+ρAT
1 A1)x1+(HT
21+ρAT
1 A2)x2+(c1+AT
1 y+ρAT
1 b) = 0
∇x2Lρ=(H21+ρAT
2 A1)x1+(H22+ρAT
2 A2)x2+(c2+AT
2 y+ρAT
2 b) = 0
hence ADMM for QP repeats the following steps:
xk+1
1
= −(H11+ρAT
1 A1)−1
(HT
21+ρAT
1 A2)xk
2+(c1+AT
1 y+ρAT
1 b)

xk+1
2
= −(H22+ρAT
2 A2)−1
(H21+ρAT
2 A1)xk+1
1
+(c2+AT
2 y+ρAT
2 b)

yk+1 = yk + ρ(A1xk+1
1
+ A2xk+1
2
−b)
Bologna, January 2023
26


---

## 第27页

J. Gondzio
L6: ADMM
Relation between ADMM and Gauss-Seidel
Consider a large system of linear equations Qx = r which involves
a positive deﬁnite matrix Q that is decomposed into p × p blocks:


Q11 Q12 · · · Q1p
Q21 Q22 · · · Q2p
...
...
...
...
Qp1 Qp2 · · · Qpp




x1
x2
...
xp

=


r1
r2
...
rp


(blocks may have diﬀerent sizes).
Gauss-Seidel Method repeats the following steps:
xk+1
1
= Q−1
11 (r1 −Q12xk
2−. . .−Q1pxkp)
xk+1
2
= Q−1
22 (r2 −Q21xk+1
1
−Q23xk
3 −. . .−Q2pxkp)
...
...
xk+1
p
= Q−1
pp (rp −Qp1xk+1
1
−Qp2xk+1
2
−. . .−Qp,p−1xk+1
p−1).
S. Cipolla, J. Gondzio, ADMM and inexact ALM: the QP case.
https://www.maths.ed.ac.uk/~gondzio/reports/ADMMandIALM.html
Bologna, January 2023
27


---

## 第28页

J. Gondzio
L6: ADMM
(Block) Gauss-Seidel Method
Consider the following splitting of the matrix


Q11 Q12 · · · Q1p
Q21 Q22 · · · Q2p
...
...
...
...
Qp1 Qp2 · · · Qpp

=


Q11
0
· · ·
0
Q21 Q22 · · ·
0
...
...
...
...
Qp1 Qp2 · · · Qpp


|
{z
}
L
+


0 Q12 · · · Q1p
0
0
· · · Q2p
...
...
...
...
0
0
· · ·
0


|
{z
}
U
,
and rearrange the equation
Qx = (L + U)x = r
⇔
Lx = r −Ux
⇔
x = L−1(r −Ux).
Gauss-Seidel Method is a ﬁxed point iteration:
xk+1 = L−1(r −Uxk).
Gauss-Seidel iteration overwrites the approximate solution with the
new value as soon as it is computed:
xk+1
i
= Q−1
ii (ri −P
j<i Qijxk+1
j
−P
j>i Qijxk
j),
ADMM for QP acts as a Gauss-Seidel iteration.
Bologna, January 2023
28


---

## 第29页

J. Gondzio
L6: ADMM
Final Remarks
Alternating Direction Method of Multipliers (ADMM)
• is suitable for problems with loosely coupled variables
• has inexpensive iterations
hence is attractive for very large scale optimization
• may be slow, but
is often suﬃciently fast when appropriately tuned
• has numerous applications due to its ‘decoupling’ ability:
– machine learning/statistics (large data sets),
– image processing,
– decentralized optimization.
Bologna, January 2023
29


---

## 第30页

J. Gondzio
L6: ADMM
Modern Techniques
of Large Scale Optimization
for Data Science
Thank you for your attention!
Bologna, January 2023
30


---

