# JGondzio-Lectures1and2.pdf

## 第1页

J. Gondzio
L1&2: IPMs for LP
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
Interior Point Methods
for Linear Programming:
Motivation & Theory
Jacek Gondzio
Email: J.Gondzio@ed.ac.uk
URL: http://www.maths.ed.ac.uk/~gondzio
Bologna, January 2023
1


---

## 第2页

J. Gondzio
L1&2: IPMs for LP
Outline
• IPM for LP: Motivation
– complementarity conditions
– ﬁrst order optimality conditions
– central trajectory
– primal-dual framework
• Polynomial Complexity of IPM
– Newton method
– short step path-following method
– polynomial complexity proof
Bologna, January 2023
2


---

## 第3页

J. Gondzio
L1&2: IPMs for LP
Building Blocks of the IPM
What do we need to derive the Interior Point Method?
• duality theory:
Lagrangian function;
ﬁrst order optimality conditions.
• logarithmic barriers.
• Newton method.
Bologna, January 2023
3


---

## 第4页

J. Gondzio
L1&2: IPMs for LP
Primal-Dual Pair of Linear Programs
Primal
Dual
min
cTx
max
bTy
s.t.
Ax = b,
s.t.
ATy + s = c,
x ≥0;
s ≥0.
Lagrangian
L(x, y) = cTx −yT(Ax −b) −sTx.
Optimality Conditions
Ax = b,
ATy + s = c,
XSe = 0,
( i.e., xj · sj = 0
∀j),
(x, s) ≥0,
X =diag{x1, · · · , xn}, S =diag{s1, · · · , sn}, e=(1, · · · , 1)∈Rn.
Bologna, January 2023
4


---

## 第5页

J. Gondzio
L1&2: IPMs for LP
Logarithmic barrier
−ln xj
“replaces” the inequality
xj ≥0 .
x
−ln x
1
Observe that
min e−Pn
j=1 ln xj
⇐⇒
max
n
Y
j=1
xj
The minimization of −Pn
j=1 ln xj is equivalent to the maximization
of the product of distances from all hyperplanes deﬁning the positive
orthant: it prevents all xj from approaching zero.
Bologna, January 2023
5


---

## 第6页

J. Gondzio
L1&2: IPMs for LP
Logarithmic barrier
Replace the primal LP
min
cTx
s.t.
Ax
= b,
x ≥0,
with the primal barrier program
min cTx −µ
nP
j=1
ln xj
s.t.
Ax = b.
Lagrangian:
L(x, y, µ) = cTx −yT(Ax −b) −µ
n
X
j=1
ln xj.
Bologna, January 2023
6


---

## 第7页

J. Gondzio
L1&2: IPMs for LP
Conditions for a stationary point of the Lagrangian
∇xL(x, y, µ) = c −ATy −µX−1e = 0
∇yL(x, y, µ) =
Ax −b = 0,
where X−1 = diag{x−1
1 , x−1
2 , · · · , x−1
n }.
Let us denote
s = µX−1e,
i.e.
XSe = µe.
The First Order Optimality Conditions are:
Ax = b,
ATy + s = c,
XSe = µe,
(x, s) > 0.
Bologna, January 2023
7


---

## 第8页

J. Gondzio
L1&2: IPMs for LP
Central Trajectory
The ﬁrst order optimality conditions for the barrier problem
Ax = b,
ATy + s = c,
XSe = µe,
(x, s) ≥0
approximate the ﬁrst order optimality conditions for the LP
Ax = b,
ATy + s = c,
XSe = 0,
(x, s) ≥0
more and more closely as µ goes to zero.
Bologna, January 2023
8


---

## 第9页

J. Gondzio
L1&2: IPMs for LP
Central Trajectory
Parameter µ controls the distance to optimality.
cTx−bTy = cTx−xTATy = xT(c−ATy) = xTs = nµ.
Analytic centre (µ-centre): a (unique) point
(x(µ), y(µ), s(µ)),
x(µ) > 0, s(µ) > 0
that satisﬁes FOC.
The path
{(x(µ), y(µ), s(µ)) : µ > 0}
is called the primal-dual central trajectory.
Bologna, January 2023
9


---

## 第10页

J. Gondzio
L1&2: IPMs for LP
Newton Method
is used to ﬁnd a stationary point of the barrier problem.
Recall how to use Newton Method to ﬁnd a root of a nonlinear
equation
f(x) = 0.
A tangent line
z −f(xk) = ∇f(xk) · (x −xk)
is a local approximation of the graph of the function f(x).
Substituting z = 0 deﬁnes a new point
xk+1 = xk −(∇f(xk))−1f(xk).
Bologna, January 2023
10


---

## 第11页

J. Gondzio
L1&2: IPMs for LP
Newton Method
x
f(x)
xk xk+1 xk+2
f(x     )
k+2
f(x     )
k+1
f(x  )
k
k
z
k
k
z-f(x  ) =    f(x  )(x-x  )
xk+1 = xk −(∇f(xk))−1f(xk).
Bologna, January 2023
11


---

## 第12页

J. Gondzio
L1&2: IPMs for LP
Apply Newton Method to the FOC
The ﬁrst order optimality conditions for the barrier problem form a
large system of nonlinear equations
f(x, y, s) = 0,
where f : R2n+m 7→R2n+m is a mapping deﬁned as follows:
f(x, y, s) =


Ax −b
ATy + s −c
XSe −µe

.
Actually, the ﬁrst two terms of it are linear; only the last one,
corresponding to the complementarity condition, is nonlinear.
Bologna, January 2023
12


---

## 第13页

J. Gondzio
L1&2: IPMs for LP
Newton Method (cont’d)
Note that
∇f(x, y, s) =


A
0
0
0 AT I
S
0
X

.
Thus, for a given point (x, y, s) we ﬁnd the Newton direction
(∆x, ∆y, ∆s) by solving the system of linear equations:


A
0
0
0 AT I
S
0
X

·
" ∆x
∆y
∆s
#
=


b −Ax
c −ATy −s
µe −XSe

.
Bologna, January 2023
13


---

## 第14页

J. Gondzio
L1&2: IPMs for LP
Interior-Point Framework
The logarithmic barrier
−ln xj
“replaces” the inequality
xj ≥0.
We derive the ﬁrst order optimality conditions for the primal
barrier problem:
Ax = b,
ATy + s = c,
XSe = µe,
and apply Newton method to solve this system of (nonlinear)
equations.
Actually, we ﬁx the barrier parameter µ and make only one (damped)
Newton step towards the solution of FOC. We do not solve the cur-
rent FOC exactly. Instead, we immediately reduce the barrier pa-
rameter µ (to ensure progress towards optimality) and repeat the
process.
Bologna, January 2023
14


---

## 第15页

J. Gondzio
L1&2: IPMs for LP
Interior Point Algorithm
Initialize
k = 0
(x0, y0, s0) ∈F0
µ0 = 1
n · (x0)Ts0
α0 = 0.9995
Repeat until optimality
k = k + 1
µk = σµk−1, where σ ∈(0, 1)
∆= (∆x, ∆y, ∆s) = Newton direction towards µ-centre
Ratio test:
αP := max {α > 0 : x + α∆x ≥0},
αD := max {α > 0 : s + α∆s ≥0}.
Make step:
xk+1 = xk + α0αP∆x,
yk+1 = yk + α0αD∆y,
sk+1 = sk + α0αD∆s.
Bologna, January 2023
15


---

## 第16页

J. Gondzio
L1&2: IPMs for LP
Notations
X = diag{x1, x2, · · · , xn} =


x1
x2 ...
xn

.
e = (1, 1, · · · , 1) ∈Rn, X−1 = diag{x−1
1 , x−1
2 , · · · , x−1
n }.
An equation
XSe = µe,
is equivalent to
xjsj = µ,
∀j = 1, 2, · · · , n.
Bologna, January 2023
16


---

## 第17页

J. Gondzio
L1&2: IPMs for LP
Notations(cont’d)
Primal feasible set P = {x ∈Rn | Ax = b, x ≥0}.
Primal strictly feasible set P0 = {x ∈Rn | Ax = b, x > 0}.
Dual feasible set D={y ∈Rm, s ∈Rn | ATy + s = c, s ≥0}.
Dual strictly feasible set D0 = {y ∈Rm, s ∈Rn | ATy + s =
c, s>0}.
Primal-dual feasible set
F
= {(x, y, s) | Ax = b, ATy + s = c, (x, s) ≥0}.
Primal-dual strictly feasible set
F0 = {(x, y, s) | Ax = b, ATy + s = c, (x, s) > 0}.
Bologna, January 2023
17


---

## 第18页

J. Gondzio
L1&2: IPMs for LP
Path-Following Algorithm
The analysis given in this lecture comes from the book of
Steve Wright: Primal-Dual Interior-Point Methods,
SIAM Philadelphia, 1997.
We analyze a feasible interior-point algorithm with the following
properties:
• all its iterates are feasible and stay in a close neighbourhood
of the central path;
• the iterates follow the central path towards optimality;
• systematic (though slow) reduction of duality gap is ensured.
This algorithm is called
the short-step path-following method.
Indeed, it makes very slow progress (short-steps) to optimality.
Bologna, January 2023
18


---

## 第19页

J. Gondzio
L1&2: IPMs for LP
Central Path Neighbourhood
Assume a primal-dual strictly feasible solution (x, y, s) ∈F0 lying
in a neighbourhood of the central path is given; namely (x, y, s)
satisﬁes:
Ax = b,
ATy + s = c,
XSe ≈µe.
We deﬁne a θ-neighbourhood of the central path N2(θ), a set of
primal-dual strictly feasible solutions (x, y, s) ∈F0 that satisfy:
∥XSe −µe∥≤θµ,
where θ ∈(0, 1) and the barrier µ satisﬁes:
xTs = nµ.
Hence N2(θ) = {(x, y, s) ∈F0 | ∥XSe −µe∥≤θµ}.
Bologna, January 2023
19


---

## 第20页

J. Gondzio
L1&2: IPMs for LP
Central Path Neighbourhood
2 θ
N  (   ) neighbourhoodof the central path
Bologna, January 2023
20


---

## 第21页

J. Gondzio
L1&2: IPMs for LP
Progress towards optimality
Assume a primal-dual strictly feasible solution (x, y, s) ∈N2(θ) for
some θ ∈(0, 1) is given.
Interior point algorithm tries to move from this point to another
one that also belongs to a θ-neighbourhood of the central path but
corresponds to a smaller µ. The required reduction of µ is small:
µk+1 = σµk,
where
σ = 1 −β/√n,
for some β ∈(0, 1).
This is a short-step method:
It makes short steps to optimality.
Bologna, January 2023
21


---

## 第22页

J. Gondzio
L1&2: IPMs for LP
Progress towards optimality
Given a new µ-centre, interior point algorithm computes Newton
direction: 

A
0
0
0 AT I
S
0
X

·
" ∆x
∆y
∆s
#
=
"
0
0
σµe −XSe
#
,
and makes step in this direction.
Magic numbers (will be explained later):
θ = 0.1
and
β = 0.1.
θ controls the proximity to the central path;
β controls the progress to optimality.
Bologna, January 2023
22


---

## 第23页

J. Gondzio
L1&2: IPMs for LP
How to prove the O(√n) complexity result
We will prove the following:
• full step in Newton direction is feasible;
• the new iterate
(xk+1, yk+1, sk+1)=(xk, yk, sk)+(∆xk, ∆yk, ∆sk)
belongs to the θ-neighbourhood of the new µ-centre
(with µk+1 = σµk);
• duality gap is reduced 1 −β/√n times.
Bologna, January 2023
23


---

## 第24页

J. Gondzio
L1&2: IPMs for LP
O(√n) complexity result
Note that since at one iteration duality gap is reduced 1 −β/√n
times, after √n iterations the reduction achieves:
(1 −β/√n)
√n ≈e−β.
After C · √n iterations, the reduction is e−Cβ.
For suﬃciently
large constant C the reduction can thus be arbitrarily large (i.e. the
duality gap can become arbitrarily small).
Hence this algorithm has complexity O(√n).
This should be understood as follows:
“after the number of iterations proportional to √n
the algorithm solves the problem”.
Bologna, January 2023
24


---

## 第25页

J. Gondzio
L1&2: IPMs for LP
Worst-Case Complexity Result
Bologna, January 2023
25


---

## 第26页

J. Gondzio
L1&2: IPMs for LP
Technical Results
Lemma 1
Newton direction (∆x, ∆y, ∆s) deﬁned by the equation system


A
0
0
0 AT I
S
0
X

·
" ∆x
∆y
∆s
#
=
"
0
0
σµe −XSe
#
,
(1)
satisﬁes:
∆xT∆s = 0.
Proof:
From the ﬁrst two equations in (1) we get
A∆x = 0
and
∆s = −AT∆y.
Hence
∆xT∆s = ∆xT · (−AT∆y) = −∆yT · (A∆x) = 0.
Bologna, January 2023
26


---

## 第27页

J. Gondzio
L1&2: IPMs for LP
Technical Results (cont’d)
Lemma 2
Let (∆x, ∆y, ∆s) be the Newton direction that solves the system
(1). The new iterate
(¯x, ¯y, ¯s) = (x, y, s) + (∆x, ∆y, ∆s)
satisﬁes
¯xT ¯s = n¯µ,
where
¯µ = σµ.
Bologna, January 2023
27


---

## 第28页

J. Gondzio
L1&2: IPMs for LP
Proof: From the third equation in (1) we get
S∆x + X∆s = −XSe + σµe.
By summing the n components of this equation we obtain
eT(S∆x+X∆s) = sT∆x+xT∆s = −eTXSe+σµeTe
= −xTs + nσµ = −xTs · (1 −σ).
Thus
¯xT ¯s = (x + ∆x)T(s + ∆s)
= xTs + (sT∆x + xT∆s) + (∆x)T∆s
= xTs + (σ −1)xTs + 0 = σxTs,
which is equivalent to:
n¯µ = σnµ.
Bologna, January 2023
28


---

## 第29页

J. Gondzio
L1&2: IPMs for LP
Reminder: Norms of the vector x ∈Rn.
∥x∥
= (
nP
j=1
x2
j)1/2
∥x∥∞=
max
j∈{1..n} |xj|
∥x∥1
=
nP
j=1
|xj|
For any x ∈Rn:
∥x∥∞≤
∥x∥1
∥x∥1
≤
n·∥x∥∞
∥x∥∞≤
∥x∥
∥x∥
≤√n·∥x∥∞
∥x∥
≤
∥x∥1
∥x∥1
≤√n·∥x∥
Bologna, January 2023
29


---

## 第30页

J. Gondzio
L1&2: IPMs for LP
Reminder: Triangle Inequality
For any vectors p, q and r and for any norm ∥.∥
∥p −q∥≤∥p −r∥+ ∥r −q∥.
The relation between algebraic and geometric means.
For any scalars a and b such that ab ≥0:
p
|ab| ≤1
2 · |a + b|.
Bologna, January 2023
30


---

## 第31页

J. Gondzio
L1&2: IPMs for LP
Technical Result (algebra)
Lemma 3 Let u and v be any two vectors in Rn such that uTv ≥0.
Then
∥UV e∥≤2−3/2∥u + v∥2,
where U =diag{u1, · · · , un}, V =diag{v1, · · · , vn}.
Proof: Let us partition all products ujvj into positive and negative
ones:
P = {j | ujvj ≥0}
and
M = {j | ujvj < 0} :
0≤uTv=
X
j∈P
ujvj +
X
j∈M
ujvj =
X
j∈P
|ujvj| −
X
j∈M
|ujvj|.
Bologna, January 2023
31


---

## 第32页

J. Gondzio
L1&2: IPMs for LP
Proof (cont’d)
We can now write
∥UV e∥= (∥[ujvj]j∈P∥2 + ∥[ujvj]j∈M∥2)1/2
≤(∥[ujvj]j∈P∥2
1 + ∥[ujvj]j∈M∥2
1)1/2
≤(2∥[ujvj]j∈P∥2
1)1/2
≤
√
2∥[1
4(uj + vj)2]j∈P∥1
= 2−3/2 X
j∈P
(uj + vj)2
≤2−3/2
n
X
j=1
(uj + vj)2
= 2−3/2∥u + v∥2,
as requested.
Bologna, January 2023
32


---

## 第33页

J. Gondzio
L1&2: IPMs for LP
IPM Technical Results (cont’d)
Lemma 4
If (x, y, s) ∈N2(θ) for some θ ∈(0, 1), then
(1 −θ)µ ≤xjsj ≤(1 + θ)µ
∀j.
In other words,
min
j∈{1..n}xjsj ≥(1 −θ)µ,
max
j∈{1..n}xjsj ≤(1 + θ)µ.
Bologna, January 2023
33


---

## 第34页

J. Gondzio
L1&2: IPMs for LP
Proof:
Since ∥x∥∞≤∥x∥, from the deﬁnition of N2(θ),
N2(θ) = {(x, y, s) ∈F0 | ∥XSe −µe∥≤θµ},
we conclude
∥XSe −µe∥∞≤∥XSe −µe∥≤θµ.
Hence
|xjsj −µ| ≤θµ
∀j,
which is equivalent to
−θµ ≤xjsj −µ ≤θµ
∀j.
Thus
(1 −θ)µ ≤xjsj ≤(1 + θ)µ
∀j.
Bologna, January 2023
34


---

## 第35页

J. Gondzio
L1&2: IPMs for LP
IPM Technical Results (cont’d)
Lemma 5
If (x, y, s) ∈N2(θ) for some θ ∈(0, 1), then
∥XSe −σµe∥2 ≤θ2µ2 + (1 −σ)2µ2n.
Proof:
Note ﬁrst that
eT(XSe −µe) = xTs −µeTe = nµ −nµ = 0.
Therefore
∥XSe −σµe∥2
= ∥(XSe−µe) + (1−σ)µe∥2
= ∥XSe−µe∥2+2(1−σ)µeT(XSe−µe)+(1−σ)2µ2eTe
≤θ2µ2 + (1−σ)2µ2n.
Bologna, January 2023
35


---

## 第36页

J. Gondzio
L1&2: IPMs for LP
IPM Technical Results (cont’d)
Lemma 6
If (x, y, s) ∈N2(θ) for some θ ∈(0, 1), then
∥∆X∆Se∥≤θ2 + n(1−σ)2
23/2(1 −θ)
µ.
Proof: 3rd equation in the Newton system gives
S∆x + X∆s = −XSe + σµe.
Having multiplied it with (XS)−1/2, we obtain
X−1/2S1/2∆x+X1/2S−1/2∆s=(XS)−1/2(−XSe+σµe).
Bologna, January 2023
36


---

## 第37页

J. Gondzio
L1&2: IPMs for LP
Proof (cont’d)
Deﬁne u=X−1/2S1/2∆x and v=X1/2S−1/2∆s and observe that
(by Lemma 1) uTv = ∆xT∆s = 0. Now apply Lemma 3:
∥∆X∆Se∥= ∥(X−1/2S1/2∆X)(X1/2S−1/2∆S)e∥
≤2−3/2∥X−1/2S1/2∆x+X1/2S−1/2∆s∥2
= 2−3/2∥X−1/2S−1/2(−XSe + σµe)∥2
= 2−3/2 nP
j=1
(−xjsj+σµ)2
xjsj
≤2−3/2∥XSe−σµe∥2
minj xjsj
≤θ2+n(1−σ)2
23/2(1−θ) µ
(by Lemmas 4 and 5).
Bologna, January 2023
37


---

## 第38页

J. Gondzio
L1&2: IPMs for LP
Magic Numbers
We have previously set two parameters for the short-step path-
following method:
θ ∈[0.05, 0.1]
and
β ∈[0.05, 0.1].
Now it’s time to justify this particular choice.
Both θ and β have to be small to make sure that a full step in the
Newton direction does not take the new iterate outside the neigh-
bourhood N2(θ).
θ controls the proximity to the central path;
β controls the progress to optimality.
Bologna, January 2023
38


---

## 第39页

J. Gondzio
L1&2: IPMs for LP
Magic numbers choice lemma
Lemma 7 If θ ∈[0.05, 0.1] and β ∈[0.05, 0.1], then
θ2 + n(1−σ)2
23/2(1 −θ)
≤σθ.
Proof:
Recall that
σ = 1 −β/√n.
Hence
n(1−σ)2 = β2
and for any β ∈[0.05, 0.1] (for any n ≥1)
σ ≥0.9.
Substituting θ ∈[0.05, 0.1] and β ∈[0.05, 0.1], we obtain
θ2+n(1−σ)2
23/2(1 −θ)
= 0.12 + 0.12
23/2 · 0.9
≤0.02≤0.9 · 0.1≤σθ.
Bologna, January 2023
39


---

## 第40页

J. Gondzio
L1&2: IPMs for LP
Full Newton step in N2(θ)
Lemma 8
Suppose (x, y, s) ∈N2(θ) and (∆x, ∆y, ∆s) is the
Newton direction computed from the system (1). Then the new
iterate
(¯x, ¯y, ¯s) = (x, y, s) + (∆x, ∆y, ∆s)
satisﬁes (¯x, ¯y, ¯s) ∈N2(θ), i.e. ∥¯X ¯Se −¯µe∥≤θ¯µ.
Proof: From Lemma 2, the new iterate (¯x, ¯y, ¯s) satisﬁes
¯xT ¯s = n¯µ = nσµ,
so we have to prove that ∥¯X ¯Se −¯µe∥≤θ¯µ.
For a given component j ∈{1..n}, we have
¯xj¯sj −¯µ = (xj + ∆xj)(sj + ∆sj) −¯µ
= xjsj + (sj∆xj + xj∆sj) + ∆xj∆sj−¯µ
= xjsj + (−xjsj + σµ) + ∆xj∆sj −σµ
= ∆xj∆sj.
Bologna, January 2023
40


---

## 第41页

J. Gondzio
L1&2: IPMs for LP
Proof (cont’d)
Thus, from Lemmas 6 and 7, we get
∥¯X ¯Se −¯µe∥= ∥∆X∆Se∥
≤θ2+n(1−σ)2
23/2(1−θ) µ
≤σθµ
= θ¯µ.
Bologna, January 2023
41


---

## 第42页

J. Gondzio
L1&2: IPMs for LP
A property of log function
Lemma 9 For all δ > −1:
ln(1 + δ) ≤δ.
Proof:
Consider the function
f(δ) = δ −ln(1 + δ)
and its derivative
f
′(δ) = 1 −
1
1 + δ =
δ
1 + δ.
Obviously f
′(δ) < 0 for δ ∈(−1, 0) and f
′(δ) > 0 for δ ∈(0, ∞).
Hence f(.) has a minimum at δ = 0. We ﬁnd that f(δ = 0) = 0.
Consequently, for any δ ∈(−1, ∞), f(δ) ≥0, i.e.
δ −ln(1 + δ) ≥0.
Bologna, January 2023
42


---

## 第43页

J. Gondzio
L1&2: IPMs for LP
O(√n) Complexity Result
Theorem 10
Given ǫ > 0, suppose that a feasible starting point (x0, y0, s0) ∈
N2(0.1) satisﬁes
(x0)Ts0 = nµ0, where µ0 ≤1/ǫκ,
for some positive constant κ. Then there exists an index K with
K = O(√n ln(1/ǫ)) such that
µk ≤ǫ,
∀k ≥K.
Bologna, January 2023
43


---

## 第44页

J. Gondzio
L1&2: IPMs for LP
O(√n) Complexity Result
Proof: From Lemma 2, µk+1 = σµk. Having taken logarithms of
both sides of this equality we obtain
ln µk+1 = ln σ + ln µk.
By repeatedly applying this formula and using µ0 ≤1/ǫκ, we get
ln µk = k ln σ + ln µ0 ≤k ln(1 −β/√n) + κ ln(1/ǫ).
From Lemma 9 we have ln(1−β/√n)≤−β/√n. Thus
ln µk ≤k(−β/√n) + κ ln(1/ǫ).
To satisfy µk ≤ǫ, we need:
k(−β/√n) + κ ln(1/ǫ) ≤ln ǫ.
This inequality holds for any k ≥K, where
K = κ + 1
β
· √n · ln(1/ǫ).
Bologna, January 2023
44


---

## 第45页

J. Gondzio
L1&2: IPMs for LP
Polynomial Complexity Result
Main ingredients of the polynomial complexity result for the short-
step path-following algorithm:
Stay close to the central path:
all iterates stay in the N2(θ) neighbourhood of the central path.
Make (slow) progress towards optimality:
reduce systematically duality gap
µk+1 = σµk,
where
σ = 1 −β/√n,
for some β ∈(0, 1).
Bologna, January 2023
45


---

## 第46页

J. Gondzio
L1&2: IPMs for LP
Reading about IPMs
S. Wright
Primal-Dual Interior-Point Methods, SIAM Philadelphia, 1997.
Gondzio
Interior point methods 25 years later,
European J. of Operational Research 218 (2012) 587–601.
http://www.maths.ed.ac.uk/~gondzio/reports/ipmXXV.html
Gondzio and Grothey
Direct solution of linear systems of size 109 arising in optimiza-
tion with interior point methods, in: Parallel Processing and Ap-
plied Mathematics PPAM 2005, R. Wyrzykowski, J. Dongarra,
N. Meyer and J. Wasniewski (eds.), Lecture Notes in Computer
Science, 3911, Springer-Verlag, Berlin, 2006, pp 513–525.
OOPS: Object-Oriented Parallel Solver
http://www.maths.ed.ac.uk/~gondzio/parallel/solver.html
Bologna, January 2023
46


---

