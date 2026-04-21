# JGondzio-Lectures3and4.pdf

## 第1页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
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
IPMs for Convex Optimization:
QP, NLP, SOCP and SDP
Jacek Gondzio
Email: J.Gondzio@ed.ac.uk
URL: http://www.maths.ed.ac.uk/~gondzio
Bologna, January 2023
1


---

## 第2页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Outline
• IPM for Quadratic and Nonlinear Programming
– quadratic forms, NLP notation
– duality, Lagrangian, ﬁrst order optimality conditions
– primal-dual framework
• Self-concordant barrier
• Second-Order Cone Programming
– example cones
– example SOCP problems
– logarithmic barrier function
• Semideﬁnite Programming
– background (linear matrix inequalities)
– example SDP problems
– logarithmic barrier function
• Final Comments
Bologna, January 2023
2


---

## 第3页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
IPM for Convex QP
Bologna, January 2023
3


---

## 第4页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Convex Quadratic Programs
Def. A matrix Q ∈Rn×n is positive semideﬁnite if xTQx ≥0 for
any x ̸= 0. We write Q ⪰0.
The quadratic function
f(x) = xTQ x
is convex if and only if the matrix Q is positive deﬁnite.
In such case the quadratic programming problem
min cTx + 1
2xTQ x
s.t.
Ax = b,
x ≥0,
is well deﬁned.
If there exists a feasible solution to it,
then there exists an optimal solution.
Bologna, January 2023
4


---

## 第5页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
QP with IPMs
Apply the usual procedure:
• replace inequalities with log barriers;
• form the Lagrangian;
• write the ﬁrst order optimality conditions;
• apply Newton method to them.
Replace the primal QP
min cTx + 1
2xTQ x
s.t.
Ax = b,
x ≥0,
with the primal barrier QP
min cTx + 1
2xTQ x −
nP
j=1
ln xj
s.t.
Ax = b.
Bologna, January 2023
5


---

## 第6页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
First Order Optimality Conditions
Consider the primal barrier quadratic program
min cTx + 1
2xTQ x −µ
nP
j=1
ln xj
s.t.
Ax = b,
where µ ≥0 is a barrier parameter.
Write out the Lagrangian
L(x, y, µ) = cTx + 1
2xTQ x −yT(Ax −b) −µ
n
X
j=1
ln xj,
Bologna, January 2023
6


---

## 第7页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
First Order Optimality Conditions (cont’d)
The conditions for a stationary point of the Lagrangian:
L(x, y, µ) = cTx + 1
2xTQ x −yT(Ax −b) −µ
n
X
j=1
ln xj,
are
∇xL(x, y, µ) = c −ATy −µX−1e + Qx = 0
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
Ax
= b,
ATy + s −Qx = c,
XSe
= µe.
Bologna, January 2023
7


---

## 第8页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Apply Newton Method to the FOC
The ﬁrst order optimality conditions for the barrier problem form a
large system of nonlinear equations
F(x, y, s) = 0,
where F : R2n+m 7→R2n+m is an application deﬁned as follows:
F(x, y, s) =


Ax
−b
ATy + s −Qx −c
XSe
−µe

.
Actually, the ﬁrst two terms of it are linear; only the last one,
corresponding to the complementarity condition, is nonlinear.
Note that
∇F(x, y, s) =


A
0
0
−Q AT I
S
0
X

.
Bologna, January 2023
8


---

## 第9页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Newton Method for the FOC (cont’d)
Thus, for a given point (x, y, s)
we ﬁnd the Newton direction (∆x, ∆y, ∆s)
by solving the system of linear equations:


A
0
0
−Q AT I
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
c −ATy −s + Qx
µe −XSe

.
Bologna, January 2023
9


---

## 第10页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Interior-Point QP Algorithm
Initialize
k = 0,
(x0, y0, s0) ∈F0,
µ0 = 1
n ·(x0)Ts0,
α0 = 0.9995
Repeat until optimality
k = k + 1
µk = σµk−1, where σ ∈(0, 1)
∆= Newton direction towards µ-center
Ratio test:
αP := max {α > 0 : x + α∆x ≥0},
αD := max {α > 0 : s + α∆s ≥0}.
Make step:
xk+1 = xk + α0αP∆x,
yk+1 = yk + α0αD∆y,
sk+1 = sk + α0αD∆s.
Bologna, January 2023
10


---

## 第11页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
From LP to QP
QP problem
min cTx + 1
2xTQ x
s.t.
Ax = b,
x ≥0.
First order conditions (for barrier problem)
Ax
= b,
ATy + s−Qx = c,
XSe
= µe.
Bologna, January 2023
11


---

## 第12页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
IPMs for Convex NLP
Bologna, January 2023
12


---

## 第13页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Convex Nonlinear Optimization
Consider the nonlinear optimization problem
min f(x)
s.t.
g(x) ≤0,
where x ∈Rn, and f : Rn 7→R and g : Rn 7→Rm are convex,
twice diﬀerentiable.
Assumptions:
f and g are convex
⇒If there exists a local minimum then it is a global one.
f and g are twice diﬀerentiable
⇒We can use the second order Taylor approximations.
Some additional (technical) conditions
⇒We need them to prove that the point which satisﬁes the ﬁrst
order optimality conditions is the optimum. We won’t use them in
this course.
Bologna, January 2023
13


---

## 第14页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Nonlinear Optimization with IPMs
Nonlinear Optimization via QPs:
Sequential Quadratic Programming (SQP).
Repeat until optimality:
• approximate NLP (locally) with a QP;
• solve (approximately) the QP.
Nonlinear Optimization with IPMs:
works similarly to SQP scheme.
However, the (local) QP approximations are not solved to optimal-
ity. Instead, only one step in the Newton direction corresponding to
a given QP approximation is made and the new QP approximation
is computed.
Bologna, January 2023
14


---

## 第15页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
NLP Notation
Consider the nonlinear optimization problem
min f(x)
s.t.
g(x) ≤0,
where x ∈Rn, and f : Rn 7→R and g : Rn 7→Rm are convex,
twice diﬀerentiable.
The vector-valued function g : Rn 7→Rm has a derivative
A(x) = ∇g(x) =
∂gi
∂xj

i=1..m, j=1..n
∈Rm×n
which is called the Jacobian of g.
Bologna, January 2023
15


---

## 第16页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
NLP Notation (cont’d)
The Lagrangian associated with the NLP is:
L(x, y) = f(x) + yTg(x),
where y ∈Rm, y ≥0 are Lagrange multipliers (dual variables).
The ﬁrst derivatives of the Lagrangian:
∇xL(x, y) = ∇f(x) + ∇g(x)Ty
∇yL(x, y) = g(x).
The Hessian of the Lagrangian, Q(x, y) ∈Rn×n:
Q(x, y) = ∇2xxL(x, y) = ∇2f(x) +
m
X
i=1
yi∇2gi(x).
Bologna, January 2023
16


---

## 第17页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Convexity in NLP
Lemma 2: If f : Rn 7→R and g : Rn 7→Rm are convex, twice
diﬀerentiable, then the Hessian of the Lagrangian
Q(x, y) = ∇2f(x) +
m
X
i=1
yi∇2gi(x)
is positive semideﬁnite for any x and any y ≥0. If f is strictly
convex, then Q(x, y) is positive deﬁnite for any x and any y ≥0.
Proof: The convexity of f implies that ∇2f(x) is positive semidef-
inite for any x. Similarly, the convexity of g implies that for all
i = 1, 2, ..., m, ∇2gi(x) is positive semideﬁnite for any x. Since
yi ≥0 for all i = 1, 2, ..., m and Q(x, y) is the sum of positive
semideﬁnite matrices, we have that Q(x, y) is positive semideﬁnite.
If f is strictly convex, then ∇2f(x) is positive deﬁnite and so is
Q(x, y).
Bologna, January 2023
17


---

## 第18页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
IPM for NLP
Add slack variables to nonlinear inequalities:
min
f(x)
s.t.
g(x) + z = 0
z
≥0,
where z ∈Rm. Replace inequality z ≥0 with the logarithmic
barrier:
min f(x) −µ
m
P
i=1
ln zi
s.t.
g(x) + z
= 0.
Write out the Lagrangian
L(x, y, z, µ) = f(x) + yT(g(x) + z) −µ
m
X
i=1
ln zi,
Bologna, January 2023
18


---

## 第19页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
IPM for NLP
For the Lagrangian
L(x, y, z, µ) = f(x) + yT(g(x) + z) −µ
m
X
i=1
ln zi,
write the conditions for a stationary point
∇xL(x, y, z, µ) = ∇f(x) + ∇g(x)Ty = 0
∇yL(x, y, z, µ) =
g(x) + z = 0
∇zL(x, y, z, µ) =
y −µZ−1e = 0,
where Z−1 = diag{z−1
1 , z−1
2 , · · · , z−1
m }.
The First Order Optimality Conditions are:
∇f(x) + ∇g(x)Ty = 0,
g(x) + z = 0,
Y Ze = µe.
Bologna, January 2023
19


---

## 第20页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Newton Method for the FOC
The ﬁrst order optimality conditions for the barrier problem form a
large system of nonlinear equations
F(x, y, z) = 0,
where F : Rn+2m 7→Rn+2m is an application deﬁned as follows:
F(x, y, z) =

∇f(x) + ∇g(x)Ty
g(x) + z
Y Ze −µe

.
Note that all three terms of it are nonlinear.
(In LP and QP the ﬁrst two terms were linear.)
Bologna, January 2023
20


---

## 第21页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Newton Method for the FOC
Observe that
∇F(x, y, z) =

Q(x, y) A(x)T 0
A(x)
0
I
0
Z
Y

,
where A(x) is the Jacobian of g
and Q(x, y) is the Hessian of L.
They are deﬁned as follows:
A(x) = ∇g(x)
∈Rm×n
Q(x, y) = ∇2f(x) +
m
P
i=1
yi∇2gi(x) ∈Rn×n
Bologna, January 2023
21


---

## 第22页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Newton Method (cont’d)
For a given point (x, y, z) we ﬁnd the Newton direction (∆x, ∆y, ∆z)
by solving the system of linear equations:

Q(x, y) A(x)T 0
A(x)
0
I
0
Z
Y


" ∆x
∆y
∆z
#
=

−∇f(x) −A(x)Ty
−g(x) −z
µe −Y Ze

.
Using the third equation we eliminate
∆z = µY −1e −Ze −ZY −1∆y,
from the second equation and get

Q(x, y) A(x)T
A(x)
−ZY −1
 
∆x
∆y

=

−∇f(x) −A(x)Ty
−g(x) −µY −1e

.
Bologna, January 2023
22


---

## 第23页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Interior-Point NLP Algorithm
Initialize
k = 0
(x0, y0, z0) such that y0 > 0 and z0 > 0,
µ0 = 1
m · (y0)Tz0
Repeat until optimality
k = k + 1
µk = σµk−1, where σ ∈(0, 1)
Compute A(x) and Q(x, y)
∆= Newton direction towards µ-center
Ratio test:
α1 := max {α > 0 : y + α∆y ≥0},
α2 := max {α > 0 : z + α∆z ≥0}.
Choose the step: (use trust region or line search) α≤min{α1, α2}.
Make step:
xk+1 = xk + α∆x,
yk+1 = yk + α∆y,
zk+1 = zk + α∆z.
Bologna, January 2023
23


---

## 第24页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
From QP to NLP
Newton direction for QP

−Q AT I
A
0
0
S
0
X


" ∆x
∆y
∆s
#
=
" ξd
ξp
ξµ
#
.
Augmented system for QP

−Q −SX−1 AT
A
0
 
∆x
∆y

=

ξd −X−1ξµ
ξp

.
Bologna, January 2023
24


---

## 第25页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
From QP to NLP
Newton direction for NLP

Q(x, y) A(x)T 0
A(x)
0
I
0
Z
Y


" ∆x
∆y
∆z
#
=

−∇f(x) −A(x)Ty
−g(x) −z
µe −Y Ze

.
Augmented system for NLP

Q(x, y) A(x)T
A(x) −ZY −1
 
∆x
∆y

=

−∇f(x)−A(x)Ty
−g(x)−µY −1e

.
Conclusion:
NLP is a natural extension of QP.
Bologna, January 2023
25


---

## 第26页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Newton Method
and Self-concordant Barriers
Bologna, January 2023
26


---

## 第27页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Another View of Newton M. for Optimization
Newton Method for Optimization
Let f : Rn 7→R be a twice continuously diﬀerentiable function.
Suppose we build a quadratic model ˜f of f around a given point
xk, i.e., we deﬁne ∆x = x −xk and write:
˜f(x) = f(xk) + ∇f(xk)T∆x + 1
2∆xT∇2f(xk)∆x
Now we optimize the model ˜f instead of optimizing f.
A minimum (or, more generally, a stationary point) of the quadratic
model satisﬁes:
∇˜f(x) = ∇f(xk) + ∇2f(xk)∆x = 0,
i.e.
∆x = x −xk = −(∇2f(xk))−1∇f(xk),
which reduces to the usual equation:
xk+1 = xk −(∇2f(xk))−1∇f(xk).
Bologna, January 2023
27


---

## 第28页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Self-concordant Functions
There is a nice property of the function that is responsible for a
good behaviour of the Newton method.
Def Let C ∈Rn be an open nonempty convex set.
Let f : C 7→R be a three times continuously diﬀerentiable convex
function.
A function f is called self-concordant if there exists a constant
p > 0 such that
|∇3f(x)[h, h, h]| ≤2p−1/2(∇2f(x)[h, h])3/2,
∀x ∈C, ∀h : x + h ∈C.
(We then say that f is p-self-concordant).
Note that a self-concordant function is always well approximated by
the quadratic model because the error of such an approximation can
be bounded by the 3/2 power of ∇2f(x)[h, h].
Bologna, January 2023
28


---

## 第29页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Self-concordant Barriers
Lemma
The barrier function −log x is self-concordant on R+.
Proof Consider f(x) = −log x.
Compute
f
′(x) = −x−1, f
′′(x) = x−2 and f
′′′(x) = −2x−3
and check that the self-concordance condition is satisﬁed for p = 1.
Lemma
The barrier function 1/xα, with α ∈(0, ∞) is not self-concordant
on R+.
Lemma
The barrier function e1/x is not self-concordant on R+.
Use self-concordant barriers in optimization
Bologna, January 2023
29


---

## 第30页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Second-Order Cone Programming (SOCP)
Bologna, January 2023
30


---

## 第31页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Cones: Background
Def. A set K ∈Rn is called a cone if for any x ∈K and for any
λ ≥0, λx ∈K.
Convex Cone:
x
x
1
2
x3
Example:
K = {x ∈Rn : x2
1 ≥
n
X
j=2
x2
j, x1 ≥0}.
Bologna, January 2023
31


---

## 第32页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Example: Three Cones
R+:
R+ = {x ∈R : x ≥0}.
Quadratic Cone:
Kq = {x ∈Rn : x2
1 ≥
n
X
j=2
x2
j, x1 ≥0}.
Rotated Quadratic Cone:
Kr = {x ∈Rn : 2x1x2 ≥
n
X
j=3
x2
j, x1, x2 ≥0}.
Bologna, January 2023
32


---

## 第33页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Matrix Representation of Cones
Each of the three most common cones has a matrix representation
using orthogonal matrices T and/or Q.
(Orthogonal matrix: QTQ = I).
Quadratic Cone Kq. Deﬁne
Q =


1
−1
−1 ...
−1


and write:
Kq = {x ∈Rn : xTQx ≥0, x1 ≥0}.
Example:
x2
1 ≥x2
2 + x2
3 + · · · + x2n.
Bologna, January 2023
33


---

## 第34页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Matrix Representation of Cones (cont’d)
Rotated Quadratic Cone Kr. Deﬁne
Q =


0 1
1 0
−1 ...
−1


and write:
Kr = {x ∈Rn : xTQx ≥0, x1, x2 ≥0}.
Example:
2x1x2 ≥x2
3 + x2
4 + · · · + x2n.
Bologna, January 2023
34


---

## 第35页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Matrix Representation of Cones (cont’d)
Consider a linear transformation T : R2 7→R2:
T2 =


1
√
2
1
√
2
1
√
2 −1
√
2

.
It corresponds to a rotation by π/4. Indeed, write:

z
y

= T2

u
v

that is
z = u + v
√
2 ,
y = u −v
√
2
to get
2yz = u2 −v2.
Bologna, January 2023
35


---

## 第36页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Matrix Representation of Cones (cont’d)
Now, deﬁne
T =


1
√
2
1
√
2
1
√
2 −1
√
2
1 ...
1


and observe that the rotated quadratic cone satisﬁes
Tx ∈Kr
iﬀ
x ∈Kq.
Bologna, January 2023
36


---

## 第37页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Example: Conic constraint
Consider a constraint:
1
2∥x∥2 + aTx ≤b.
Observe that g(x) = 1
2xTx+aTx−b is convex hence the constraint
deﬁnes a convex set.
The constraint may be reformulated as an intersection of an aﬃne
(linear) constraint and a quadratic one:
aTx + z = b
y = 1
∥x∥2 ≤2yz, y, z ≥0.
Bologna, January 2023
37


---

## 第38页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Example: Conic constraint (cont’d)
Now, substitute:
z = u + v
√
2 ,
y = u −v
√
2
to get
aTx + u + v
√
2
= b
u −v =
√
2
∥x∥2 + v2 ≤u2.
Bologna, January 2023
38


---

## 第39页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Dual Cone
Let K ∈Rn be a cone.
Def. The set:
K∗:= {s ∈Rn : sTx ≥0, ∀x ∈K}
is called the dual cone.
Def. The set:
KP := {s ∈Rn : sTx ≤0, ∀x ∈K}
is called the polar cone (Fig below).
K
K P
Bologna, January 2023
39


---

## 第40页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Conic Optimization
Consider an optimization problem:
min
cTx
s.t.
Ax = b,
x ∈K,
where K is a convex closed cone.
We assume that
K = K1 × K2 × · · · × Kk,
that is, cone K is a product of several individual cones each of which
is one of the three cones deﬁned earlier.
Bologna, January 2023
40


---

## 第41页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Primal and Dual SOCPs
Consider a primal SOCP
min
cTx
s.t.
Ax = b,
x ∈K,
where K is a convex closed cone.
The associated dual SOCP
max
bTy
s.t.
ATy + s = c,
s ∈K∗.
Weak Duality:
If (x, y, s) is a primal-dual feasible solution, then
cTx −bTy = xTs ≥0.
Bologna, January 2023
41


---

## 第42页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
IPM for Conic Optimization
Conic Optimization problems can be solved in polynomial time with
IPMs.
Consider a quadratic cone
Kq = {(x, t) : x ∈Rn−1, t ∈R, t2 ≥∥x∥2, t ≥0},
and deﬁne the (convex) logarithmic barrier function for this
cone f : Rn 7→R
f(x, t) =

−ln(t2 −∥x∥2) if ∥x∥< t
+∞
otherwise.
Theorem:
f(x, t) is a self-concordant barrier on Kq.
Exercise: Prove it in case n = 2.
Bologna, January 2023
42


---

## 第43页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Semideﬁnite Programming (SDP)
Bologna, January 2023
43


---

## 第44页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
SDP: Background
Def. A matrix H ∈Rn×n is positive semideﬁnite if xTHx ≥0
for any x ̸= 0. We write H ⪰0.
Def. A matrix H ∈Rn×n is positive deﬁnite if xTHx > 0 for any
x ̸= 0. We write H ≻0.
We denote with SRn×n (SRn×n
+
) the set of symmetric and sym-
metric positive semideﬁnite matrices.
Let U, V ∈SRn×n. We deﬁne the inner product between U and
V as U • V = trace(UTV ), where trace(H) = Pn
i=1 hii.
The associated norm is the Frobenius norm,
written ∥U∥F = (U • U)1/2 (or just ∥U∥).
Bologna, January 2023
44


---

## 第45页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Linear Matrix Inequalities
Def. Linear Matrix Inequalities
Let U, V ∈SRn×n.
We write U ⪰V
iﬀU −V ⪰0.
We write U ≻V
iﬀU −V ≻0.
We write U ⪯V
iﬀU −V ⪯0.
We write U ≺V
iﬀU −V ≺0.
Bologna, January 2023
45


---

## 第46页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Properties
1. If P ∈Rm×n and Q ∈Rn×m, then trace(PQ) = trace(QP).
2. If U, V ∈SRn×n, and Q ∈Rn×n is orthogonal (i.e. QTQ = I),
then U • V = (QTUQ) • (QTV Q).
More generally, if P is nonsingular, then
U • V = (PUP T) • (P −TV P −1).
3. Every U ∈SRn×n can be written as U = QΛQT, where Q is
orthogonal and Λ is diagonal. Then UQ = QΛ.
In other words the columns of Q are the eigenvectors, and the diag-
onal entries of Λ the corresponding eigenvalues of U.
4. If U ∈SRn×n and U = QΛQT, then
trace(U) = trace(Λ) = P
i λi.
Bologna, January 2023
46


---

## 第47页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Properties (cont’d)
5. For U∈SRn×n, the following are equivalent:
(i) U ⪰0 (U ≻0)
(ii) xTUx ≥0, ∀x∈Rn (xTUx>0, ∀0 ̸= x∈Rn).
(iii) If U = QΛQT, then Λ ⪰0 (Λ ≻0).
(iv) U = P TP for some matrix P (U = P TP for some square
nonsingular matrix P).
6. Every U ∈SRn×n has a square root U1/2 ∈SRn×n.
Proof: From Property 5 (ii) we get U = QΛQT.
Take U1/2 = QΛ1/2QT, where Λ1/2 is the diagonal matrix whose
diagonal contains the (nonnegative) square roots of the eigenvalues
of U, and verify that U1/2U1/2 = U.
Bologna, January 2023
47


---

## 第48页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Properties (cont’d)
7. Suppose
U =

A BT
B C

,
where A and C are symmetric and A ≻0.
Then U ⪰0 (U ≻0)
iﬀ
C −BA−1BT ⪰0 (≻0).
The matrix C −BA−1BT is called the Schur complement of A in
U.
Proof: follows easily from the factorization:

A BT
B C

=

I
0
BA−1 I
  A
0
0
C −BA−1BT
 
I A−1BT
0
I

.
8. If U ∈SRn×n and x ∈Rn, then xTUx = U • xxT.
Bologna, January 2023
48


---

## 第49页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Primal-Dual Pair of SDPs
Primal
Dual
min
C • X
max
bTy
s.t.
Ai • X = bi, i = 1..m
s.t.
Pm
i=1 yiAi+S = C,
X ⪰0;
S ⪰0,
where Ai ∈SRn×n, b ∈Rm, C ∈SRn×n are given;
and X, S ∈SRn×n, y ∈Rm are the variables.
Simpliﬁed notation:
Primal
Dual
min C • X
max
bTy
s.t.
AX = b,
s.t.
A∗y+S = C,
X ⪰0;
S ⪰0.
Bologna, January 2023
49


---

## 第50页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Theorem: Weak Duality in SDP
If X is feasible in the primal and (y, S) in the dual, then
C • X −bTy = X • S ≥0.
Proof:
C • X −bTy = (
m
X
i=1
yiAi + S) • X −bTy
=
m
X
i=1
(Ai • X) yi + S • X −bTy
= S • X = X • S.
Further, since X is positive semideﬁnite, it has a square root X1/2
(Property 6), and so
X • S = trace(XS)=trace(X1/2X1/2S)=trace(X1/2SX1/2) ≥0.
We use Property 1 and the fact that S and X1/2 are positive
semideﬁnite, hence X1/2SX1/2 is positive semideﬁnite and its trace
is nonnegative.
Bologna, January 2023
50


---

## 第51页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
SDP Example 1: Minimize the Max. Eigenvalue
We wish to choose x ∈Rk to minimize the maximum eigenvalue of
A(x)=A0+x1A1+. . .+xkAk, where Ai ∈Rn×n and Ai = AT
i .
Observe that
λmax(A(x)) ≤t
if and only if
λmax(A(x) −tI) ≤0
⇐⇒
λmin(tI −A(x)) ≥0.
This holds iﬀ
tI −A(x) ⪰0.
So we get the SDP in the dual form:
max −t
s.t.
tI −A(x) ⪰0,
where the variable is y := (t, x).
Bologna, January 2023
51


---

## 第52页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Logarithmic Barrier Function
Deﬁne the logarithmic barrier function for the cone SRn×n
+
of positive deﬁnite matrices.
f : SRn×n
+
7→R
f(X) =

−ln det X if X ≻0
+∞
otherwise.
Let us evaluate its derivatives.
Let X ≻0, H ∈SRn×n. Then
f(X + αH) = −ln det[X(I + αX−1H)]
= −ln det X −ln(1 + αtrace(X−1H) + O(α2))
= f(X) −αX−1 • H + O(α2),
so that f′(X) = −X−1 and Df(X)[H] = −X−1 • H.
Bologna, January 2023
52


---

## 第53页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Logarithmic Barrier Function (cont’d)
Similarly
f′(X + αH) = −[X(I + αX−1H)]−1
= −[I−αX−1H + O(α2)]X−1
= f′(X) + αX−1HX−1 + O(α2),
so that f′′(X)[H] = X−1HX−1
and D2f(X)[H, G] = X−1HX−1 • G.
Finally,
f′′′(X)[H, G] = −X−1HX−1GX−1 −X−1GX−1HX−1.
Bologna, January 2023
53


---

## 第54页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Logarithmic Barrier Function (cont’d)
Theorem: f(X) = −ln det X is a convex barrier for SRn×n
+
.
Proof: Deﬁne φ(α) = f(X +αH). We know that f is convex if,
for every X ∈SRn×n
+
and every H ∈SRn×n, φ(α) is convex in
α.
Consider a set of α such that X+αH ≻0. On this set
φ′′(α) = D2f(¯X)[H, H] = ¯X−1H ¯X−1 • H,
where ¯X = X+αH.
Since ¯X ≻0, so is V = ¯X−1/2 (Property 6), and
φ′′(α) = V 2HV 2 • H = trace(V 2HV 2H)
= trace((V HV )(V HV )) = ∥V HV ∥2
F ≥0.
So φ is convex.
When X ≻0 approaches a singular matrix, its determinant ap-
proaches zero and f(X) →∞.
Bologna, January 2023
54


---

## 第55页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Solving SDPs with IPMs
Replace the primal SDP
min C • X
s.t.
AX = b,
X ⪰0,
with the primal barrier SDP
min C • X + µf(X)
s.t.
AX = b,
(with a barrier parameter µ ≥0).
Formulate the Lagrangian
L(X, y, S) = C • X + µf(X) −yT(AX −b),
with y ∈Rm, and write the ﬁrst order conditions (FOC) for a
stationary point of L:
C + µf′(X) −A∗y = 0.
Bologna, January 2023
55


---

## 第56页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Solving SDPs with IPMs (cont’d)
Use f(X) = −ln det(X) and f′(X) = −X−1.
Therefore the FOC become:
C −µX−1 −A∗y = 0.
Denote S = µX−1, i.e., XS = µI.
For a positive deﬁnite matrix X its inverse is also positive deﬁnite.
The FOC now become:
AX = b,
A∗y + S = C,
XS = µI,
with X ≻0 and S ≻0.
Then apply Newton method to the FOC.
Bologna, January 2023
56


---

## 第57页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
The Rank Minimization Problem
min rank(X)
s.t.
A(X) = b
X ∈Rn×n is the unknown and the linear map A : Rn×n →Rm
and the vector b ∈Rm are given.
• NP-hard problem
• Applications: matrix completion
(Netﬂix problem, triangulation from incomplete data),
nonnegative factorization, control and system theory,
image compression.
Bologna, January 2023
57


---

## 第58页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
A Rank Minimization Heuristic
min rank(X)
s.t.
A(X) = b
⇒
|{z}
heuristic
min ∥X∥∗
s.t.
A(X) = b
where ∥·∥∗denotes the nuclear norm (the sum of singular values).
• Convex optimization problem
• Special case:
if X=diag(x), the problem reduces to ℓ1-norm minimization:
min card(x)
s.t.
Ax = b
⇒
|{z}
heuristic
min ∥x∥1
s.t.
Ax = b
Bologna, January 2023
58


---

## 第59页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
SDP formulation
Primal-dual convex formulation (heuristic)
min ∥X∥∗
s.t.
A(X) = b
max bTy
s.t.
∥A∗(y)∥≤1
Primal-dual SDP formulation
min 1
2(Tr(W1) + Tr(W2))
s.t.
W1 X
XT W2

⪰0
A(X) = b
max bTy
s.t.

Im
A∗(y)
A∗(y)T
In

⪰0
where y ∈Rm, W1, W2 ∈Rn×n,
A∗: Rm →Rn×n is the adjoint of A,
∥· ∥denotes the operator norm (the maximum singular value).
Bologna, January 2023
59


---

## 第60页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Netﬂix Problem (Matrix Completion)
Recommender systems
The Netﬂix Prize ($1M): In 2006, Netﬂix held the ﬁrst Netﬂix
Prize competition to ﬁnd a better program to predict user prefer-
ences and beat its existing Netﬂix movie recommendation system
by at least 10%.
• Given 100 million ratings on a scale of 1 to 5,
predict 3 million ratings to highest accuracy
• 17770 total movies x 480189 total users
⇒over 8 billion total ratings
B =
Bij known for black cells, unknown for white
Row index: movie, Column index: user
Find low-rank W such that W ≈B.
Bologna, January 2023
60


---

## 第61页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
The Matrix Completion Problem
A small number of entries of a matrix B ∈R ˆm×ˆn is known:
all entries Bi,j, with (i, j) ∈Ω, where |Ω| = m ≪ˆmˆn.
Find an approximation W ∈R ˆm×ˆn of B such that:
• W has small rank, and
• W and B agree on Ω.
Matrix Completion Problem
min rank(W)
s.t.
Wij = Bij,
∀(i, j) ∈Ω.
Bologna, January 2023
61


---

## 第62页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
SDP Relaxation of Matrix Completion
SDP Relaxation
min 1
2(Tr(W1) + Tr(W2)) ↔C • X
s.t.
 W1 W
W T W2

⪰0
↔X ⪰0
Wij = Bij (i, j) ∈Ω↔Al • X = bl
W ∈R ˆm×ˆn, W1 ∈R ˆm× ˆm, W2 ∈Rˆn×ˆn unknowns, Bij, (i, j) ∈Ω
given
• C = In, X =
 W1 W
W T W2

∈Rn×n, with n = ( ˆm + ˆn).
• Al = 1
2

0
Θij
(Θij)T
0

, l = 1, . . . , m, where for each (i, j) ∈Ω
Θij ∈R ˆm×ˆn: (Θij)st =

1 if (s, t) = (i, j)
0 else
(Al of rank 2).
Bologna, January 2023
62


---

## 第63页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Logarithmic Barrier Function
for the cone SRn×n
+
of positive deﬁnite matrices, f : SRn×n
+
7→R
f(X) =

−ln det X if X ≻0
+∞
otherwise.
LP:
Replace x ≥0
with −µ Pn
j=1 ln xj.
SDP: Replace X ⪰0 with −µ Pn
j=1 ln λj = −µ ln(Qn
j=1 λj).
Nesterov and Nemirovskii,
Interior Point Polynomial Algorithms in Convex Programming:
Theory and Applications, SIAM, Philadelphia, 1994.
Lemma
The barrier function f(X) is self-concordant on SRn×n
+
.
Bologna, January 2023
63


---

## 第64页

J. Gondzio
L3&4: IPMs for QP, NLP, SOCP, SDP
Interior Point Methods:
• Logarithmic barrier functions for LP, QP, SOCP and SDP
Self-concordant barriers
→polynomial complexity (predictable behaviour)
• Uniﬁed view of optimization
→from LP via QP to NLP, SOCP, SDP
• Eﬃciency
– good for SOCP
– problematic for SDP because solving the problem of size
n involves linear algebra operations in dimension n2
→and this requires n6 ﬂops!
Use IPMs in your research!
Bologna, January 2023
64


---

