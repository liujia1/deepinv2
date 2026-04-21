# Siltanen_Bologna2023_Day1exercises_share.pdf

## 第1页

X-ray tomography minicourse:
Monday exercises
Samuli Siltanen
PhD Winter School
Advanced methods for mathematical image analysis
Bologna, January 23, 2023


---

## 第2页

Outline
Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise
Tomography with 1×2 pixels: ill-posedness
Exercises, collected


---

## 第3页

8 (= 2 + 6)
2
6
2
7
X-ray
source
•
-
X-ray camera


---

## 第4页

8 (= 2 + 6)
9 (= 2 + 7)
2
6
2
7
•
-
•
-


---

## 第5页

4
13
2
6
2
7
•
?
•
?


---

## 第6页

4
13
8
9
2
6
2
7
•
?
•
?
•
-
•
-


---

## 第7页

4
13
8
9
?
?
?
?
•
?
•
?
•
-
•
-


---

## 第8页

4
13
8
9
2
6
2
7
4
13
8
9
-1
9
5
4


---

## 第9页

4
13
8
9
2
6
2
7
4
13
8
9
-1
9
5
4


---

## 第10页

4
13
8
9
2
6
2
7
4
13
8
9
-1
9
5
4
8
9
4
4
0
9


---

## 第11页

0
0
0
0
2 -2
-2
2
•
?
•
?
•
-
•
-


---

## 第12页

4
13
8
9
2
6
2
7 +
0
0
0
0
2 -2
-2
2
4
13
8
9
4
4
0
9
=


---

## 第13页

Outline
Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise
Tomography with 1×2 pixels: ill-posedness
Exercises, collected


---

## 第14页

Each data point gives rise to one row in the
measurement matrix
8
x1 x3
x2 x4
•
-


1
0
1
0 



x1
x2
x3
x4

=


8 



---

## 第15页

Each data point gives rise to one row in the
measurement matrix
8
9
x1 x3
x2 x4
•
-
•
-


1
0
1
0
0
1
0
1




x1
x2
x3
x4

=


8
9




---

## 第16页

Each data point gives rise to one row in the
measurement matrix
4
8
9
x1 x3
x2 x4
•
?
•
-
•
-


1
0
1
0
0
1
0
1
1
1
0
0




x1
x2
x3
x4

=


8
9
4




---

## 第17页

Each data point gives rise to one row in the
measurement matrix
4
13
8
9
x1 x3
x2 x4
•
?
•
?
•
-
•
-


1
0
1
0
0
1
0
1
1
1
0
0
0
0
1
1




x1
x2
x3
x4

=


8
9
4
13


Ax = m


---

## 第18页

Outline
Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise
Tomography with 1×2 pixels: ill-posedness
Exercises, collected


---

## 第19页

First 2×2 exercise
Determine the kernel of the measurement matrix
A =


1
0
1
0
0
1
0
1
1
1
0
0
0
0
1
1

.
How is the kernel related to “ghosts”, or objects that are nontrivial
but give zero measurement?


---

## 第20页

Outline
Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise
Tomography with 1×2 pixels: ill-posedness
Exercises, collected


---

## 第21页

Let’s study the two penalties used in
regularization. We focus on three examples
2
6
2
7
Original patient
3
3
3
3
Flat candidate
Wrong data,
good “tissue type”
4
4
0
9
Spooky candidate
Correct data,
bad “tissue type”


---

## 第22页

Calculate data penalty for the original phantom
(8 −8)2
2
6
2
7
-


---

## 第23页

Calculate data penalty for the original phantom
(8 −8)2
(9 −9)2
2
6
2
7
Data penalty: (8 −8)2 + (9 −9)2
-
-


---

## 第24页

Calculate data penalty for the original phantom
(4 −4)2 (13 −13)2
(8 −8)2
(9 −9)2
2
6
2
7
Data penalty: (8 −8)2 + (9 −9)2 + (4 −4)2 + (13 −13)2 = 0.
•
?
•
?
-
-


---

## 第25页

Calculate prior penalty for the original phantom
2
6
2
7
Prior penalty: |2 −6|


---

## 第26页

Calculate prior penalty for the original phantom
2
6
2
7
Prior penalty: |2 −6| + |2 −7|


---

## 第27页

Calculate prior penalty for the original phantom
2
6
2
7
Prior penalty: |2 −6| + |2 −7| + |2 −2|


---

## 第28页

Calculate prior penalty for the original phantom
2
6
2
7
Prior penalty: |2 −6| + |2 −7| + |2 −2| + |6 −7| = 4 + 5 + 0 + 1 = 10.


---

## 第29页

Total penalty is the sum of data&prior penalties
2
6
2
7
data penalty
0
+ prior penalty
10
= total penalty e10


---

## 第30页

Data penalty for ﬂat candidate
Data penalty: 22 + 32 + 22 + 72 = 4 + 9 + 4 + 49 = 66.
3
3
3
3
(6 −4)2 (6 −13)2
(6 −8)2
(6 −9)2


---

## 第31页

Prior penalty for ﬂat candidate
Prior penalty: |3 −3| + |3 −3| + |3 −3| + |3 −3| = 0.
3
3
3
3


---

## 第32页

Total penalty for ﬂat candidate
3
3
3
3
data penalty
66
+ prior penalty
0
= total penalty e66


---

## 第33页

Data penalty for spooky candidate
Data penalty: (8 −8)2 + (9 −9)2 + (4 −4)2 + (13 −13)2 = 0.
4
4
0
9
(4 −4)2 (13 −13)2
(8 −8)2
(9 −9)2


---

## 第34页

Prior penalty for spooky candidate
Prior penalty: |4 −4| + |0 −9| + |4 −0| + |4 −9| = 0 + 9 + 4 + 5 = 18.
4
4
0
9


---

## 第35页

Comparison of the three candidates
2
6
2
7
data penalty
0
+ prior penalty
10
= total penalty e10
Original patient
3
3
3
3
data penalty
66
+ prior penalty
0
= total penalty e66
Flat candidate
4
4
0
9
data penalty
0
+ prior penalty
18
= total penalty e18
Spooky candidate


---

## 第36页

In practice we do not have three candidates.
We need a general reconstruction algorithm
4
13
8
9
x1 x3
x2 x4
Find numbers x1 ≥0, x2 ≥0, x3 ≥0 and
x4 ≥0 such that the sum of these two
penalties is as small as possible:
Data penalty: (x1 +x3 −8)2 +(x2 +x4 −9)2
+(x1+x2−4)2+(x3+x4−13)2
Prior penalty: |x1 −x3| + |x2 −x4|
+ |x1 −x2| + |x3 −x4|
This method is called (anisotropic) total variation regularization.
•
?
•
?
-
-


---

## 第37页

The minimizer of the TV penalty functional has
two “internal organs”, as does the original
2
6
2
7
data penalty
0
+ prior penalty
10
= total penalty e10
Original patient
21
4 61
4
21
4
61
4
data penalty
1
+ prior penalty
8
= total penalty e9
TV minimizer


---

## 第38页

Outline
Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise
Tomography with 1×2 pixels: ill-posedness
Exercises, collected


---

## 第39页

Second 2×2 exercise, slide 1/2
Assume that we know three pixel values and look for the
fourth one, called x.
4
13
8
9
2
6
2
x
•
?
•
?
•
-
•
-


---

## 第40页

Second 2×2 exercise, slide 2/2
Take α = 1. Write down the total variation penalty functional in
the form
ex = arg min
x∈R
{f (x)}.
▶Give the formula for f .
▶Plot f (x).
▶At what points does f fail to be diﬀerentiable?
▶Find the minimizing argument ex ∈R approximately. You can
either use brute-force forking or apply an optimization method.


---

## 第41页

Outline
Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise
Tomography with 1×2 pixels: ill-posedness
Exercises, collected


---

## 第42页

Recall the matrix measurement model
4
13
8
9
x1 x3
x2 x4
•
?
•
?
•
-
•
-


1
0
1
0
0
1
0
1
1
1
0
0
0
0
1
1




x1
x2
x3
x4

=


8
9
4
13


Ax = m


---

## 第43页

We can now formulate (anisotropic) total
variation regularization mathematically
xTV = arg min
x∈R4 {∥Ax −m∥2
2 + ∥LHx∥1 + ∥LVx∥1}


---

## 第44页

Writing the prior penalty in matrix form:
construction of the horizontal diﬀerence matrix LH
x1 x3
x2 x4
 1
0
−1
0 


x1
x2
x3
x4

=
x1 −x3



---

## 第45页

Writing the prior penalty in matrix form:
construction of the horizontal diﬀerence matrix LH
x1 x3
x2 x4
 1
0
−1
0
0
1
0
−1

|
{z
}
LH


x1
x2
x3
x4

=
x1 −x3
x2 −x4



---

## 第46页

Writing the prior penalty in matrix form:
construction of the vertical diﬀerence matrix LV
x1 x3
x2 x4
 1
−1
0
0 


x1
x2
x3
x4

=
x1 −x2



---

## 第47页

Writing the prior penalty in matrix form:
construction of the vertical diﬀerence matrix LV
x1 x3
x2 x4
 1
−1
0
0
0
0
1
−1

|
{z
}
LV


x1
x2
x3
x4

=
x1 −x2
x3 −x4



---

## 第48页

Matrix formulation of the anisotropic
total variation prior penalty
Our minimization problem:
xTV = arg min
x∈R4 {∥Ax −m∥2
2 + ∥LHx∥1 + ∥LV x∥1}
Recall that for a vector y ∈Rn we have
∥y∥1 = |y1| + |y2| + · · · + |yn|.
Therefore, the prior penalty can be written as (why? check!)
∥LHx∥1 + ∥LVx∥1 = |x1 −x3| + |x2 −x4|
+|x1 −x2| + |x3 −x4|.


---

## 第49页

Reformulation as a quadratic problem
We want to minimize the non-quadratic functional
∥Ax −m∥2
2 + ∥LHx∥1 + ∥LV x∥1
over non-negative image vectors x ∈R4. This task can be
converted into minimizing the quadratic functional
1
2zTQz + cTz
over non-negative z ∈R12 with equality constraints Ez = b.


---

## 第50页

Rewriting the TV regularization using the trick of
non-negative vectors
Write the horizontal and vertical diﬀerences in the form
LHx = u+
H −u−
H
and
LVx = u+
V −u−
V ,
using non-negative vectors u±
H , u±
V ∈R2.
Then TV regularization is equivalent to minimizing
xTATAx −2xTATm +
1
1
T
(u+
H + u−
H + u+
V + u−
V ),
over non-negative vectors x ∈R4 (why? check!).


---

## 第51页

Reduction of TV regularization to the quadratic
problem arg min
z∈R12
+
1
2zTQz + cTz
	
with Ez = b
So we aim to minimize 1
2zTQz + cTz with
z =


x
u+
H
u−
H
u+
V
u−
V


∈R12
+ ,
Q =


2ATA
0
. . .
0
0
0
. . .
0
...
...
...
0
. . .
0

,
c =


−2ATm
1
...
1

.


---

## 第52页

Explicit form of the equality constraint, slide 1/2
The equality constraint Ez = b is needed for enforcing the
identities LHx −u+
H + u−
H = 0 and LVx −u+
V + u−
V = 0.
Since
z =


x
u+
H
u−
H
u+
V
u−
V


∈R12,
we have
Ez =
 LH
−I
I
0
0
LV
0
0
−I
I

z = 0.


---

## 第53页

Explicit form of the equality constraint, slide 2/2
Finally we get
E =


1
0
−1
0
−1
0
1
0
0
0
0
0
0
1
0
−1
0
−1
0
1
0
0
0
0
1
−1
0
0
0
0
0
0
−1
0
1
0
0
0
1
−1
0
0
0
0
0
−1
0
1


and
b =


0
0
0
0

.


---

## 第54页

Implementation in Matlab: preliminaries
% Construct the 2x2 pixel image target as a vertical vector
target = [2;2;6;7];
% Construct the measurement matrix
A = [1 0 1 0; 0 1 0 1; 1 1 0 0; 0 0 1 1];
% Compute an ideal X-ray measurement
m = A*target;
% Record the size of the unknown. Here M*M=n,
% since the unknown is a MxM pixel image.
n = 4; M = 2;


---

## 第55页

Regularization parameter
We are actually considering the TV regularization problem in a
restricted form. In general it is advisable to solve
xTV(α) = arg min
x∈R2×2
+
{∥Ax −m∥2
2 + α∥LHx∥1 + α∥LV x∥1},
where α > 0 is a regularization parameter. However, for now we
keep α = 1 and write the following in Matlab:
% Regularization parameter
alpha = 1;


---

## 第56页

% Construct prior matrices
LH = [1 0 -1 0; 0 1 0 -1];
LV = [1 -1 0 0; 0 0 1 -1];
% Construct the quadratic optimization problem matrix
Q = zeros(n+4*M*(M-1));
Q(1:n,1:n) = 2*A.’*A;
% Construct the vector h of the linear term
c = alpha*ones(n+4*M*(M-1),1);
c(1:n) = -2*(A.’)*m(:);


---

## 第57页

% Construct input arguments for quadprog.m
Z = zeros(M*(M-1));
Aeq = [[LH,-eye(M*(M-1)),eye(M*(M-1)),Z,Z];...
[LV,Z,Z,-eye(M*(M-1)),eye(M*(M-1))]];
beq = zeros(2*M*(M-1),1);
lb = zeros(n+4*M*(M-1),1);
ub = Inf(5*n,1);
AA = -eye(n+4*M*(M-1));
AA(1:n,1:n) = zeros(n,n);
iniguess = zeros(n+4*M*(M-1),1);
b = [repmat(10,n,1);zeros(4*M*(M-1),1)];
QPopt = optimset(’quadprog’);
QPopt = optimset(QPopt,’Algorithm’,...
’interior-point-convex’, ’Display’,’iter’);


---

## 第58页

% Compute reconstruction using quadprog
z = quadprog(Q,c,AA,b,Aeq,beq,lb,ub,iniguess,QPopt);
% Pick out the reconstructed image
recn = z(1:n);
% Show the reconstruction in image format
reshape(recn,M,M)


---

## 第59页

21
4 61
4
21
4
61
4
data penalty
1
+ prior penalty
8
= total penalty e9
Total variation
regularization
>> reshape(recn,M,M))
ans =
2.2500 6.2500
2.2500 6.2500


---

## 第60页

Outline
Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise
Tomography with 1×2 pixels: ill-posedness
Exercises, collected


---

## 第61页

Third 2×2 exercise
Run the computation of the previous slide using the Matlab routine
tomo2x2_TV_comp_quadprog.m in the Git repository
https://github.com/ssiltane/BolognaWinterSchool2023
Note that you need the Optimization Toolbox.
Then repeat the computation with several values of regularization
parameter α > 0. What choice of α > 0 gives the smallest
diﬀerence (measured in standard Euclidean norm of R4) between
the true target and the regularized solution? Give the optimal α
with the accuracy of two correct digits after the decimal point.


---

## 第62页

Outline
Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise
Tomography with 1×2 pixels: ill-posedness
Exercises, collected


---

## 第63页

The ﬁrst X-ray in our measurement travels
horizontally
x1
x2
•
-


---

## 第64页

Second X-ray in the measurement has slope 1/2.
Note the geometric parameter 0 < h < 1/2
x1
x2
•
*
| {z }
h


---

## 第65页

First 1×2 exercise: construct measurement matrix
Assuming that the side length of pixel is one, write down the 2×2
matrix Ah modelling the measurement. (Some of the matrix
elements may depend on h > 0.)
▶Show that Ah is invertible for any 0 < h < 1/2.
▶What happens to det(Ah) when h →0? Why?
▶We assume everywhere else that 0 < h < 1/2. However, in
this problem we step outside that assumption a bit. Is Ah
invertible when 1/2 ≤h < 1? How about the case h ≥1?


---

## 第66页

Naive inversion
The measurement model is
Ah
x1
x2

=
m1
m2

,
or Ahx = m in short. Now assume that we have noisy data
em = Ahx + ε.
Here ε ∈R2 is a random noise vector. If Ah is invertible, we can
attempt naive inversion A−1
h
em. In the next exercise you will analyse
this idea.


---

## 第67页

Second 1×2 exercise: ill-posedness of naive
inversion
Naive reconstruction is an approximation of the unknown x, as we
can see by this calculation:
A−1
h
em = A−1
h (Ahx + ε) = x + A−1
h ε.
So we can bound the error by
∥A−1
h ε∥R2 ≤∥A−1
h ∥R2→R2∥ε∥R2,
where ∥A−1
h ∥R2→R2 is the operator norm of Ah.
▶Compute the eigenvalues λ1(h) > 0 and λ2(h) > 0 of the
matrix AT
h Ah numerically for a sequence of h values
approaching zero. The numbers sj(h) =
p
λj(h) are called
singular values of Ah. We order them so that s1 ≥s2.
▶Now ∥A−1
h ∥R2→R2 = 1/s2(h). What happens to ∥A−1
h ∥R2→R2
when h →0? What does that mean for the error bound?


---

## 第68页

Outline
Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise
Tomography with 1×2 pixels: ill-posedness
Exercises, collected


---

## 第69页

First 2×2 exercise
Determine the kernel of the measurement matrix
A =


1
0
1
0
0
1
0
1
1
1
0
0
0
0
1
1

.
How is the kernel related to “ghosts”, or objects that are nontrivial
but give zero measurement?


---

## 第70页

Second 2×2 exercise, slide 1/2
Assume that we know three pixel values and look for the
fourth one, called x.
4
13
8
9
2
6
2
x
•
?
•
?
•
-
•
-


---

## 第71页

Second 2×2 exercise, slide 2/2
Take α = 1. Write down the total variation penalty functional in
the form
ex = arg min
x∈R
{f (x)}.
▶Give the formula for f .
▶Plot f (x).
▶At what points does f fail to be diﬀerentiable?
▶Find the minimizing argument ex ∈R approximately. You can
either use brute-force forking or apply an optimization method.


---

## 第72页

Third 2×2 exercise
Run the computation of the previous slide using the Matlab routine
tomo2x2_TV_comp_quadprog.m in the Git repository
https://github.com/ssiltane/BolognaWinterSchool2023
Note that you need the Optimization Toolbox.
Then repeat the computation with several values of regularization
parameter α > 0. What choice of α > 0 gives the smallest
diﬀerence (measured in standard Euclidean norm of R4) between
the true target and the regularized solution? Give the optimal α
with the accuracy of two correct digits after the decimal point.


---

## 第73页

First 1×2 exercise: construct measurement matrix
Assuming that the side length of pixel is one, write down the 2×2
matrix Ah modelling the measurement. (Some of the matrix
elements may depend on h > 0.)
▶Show that Ah is invertible for any 0 < h < 1/2.
▶What happens to det(Ah) when h →0? Why?
▶We assume everywhere else that 0 < h < 1/2. However, in
this problem we step outside that assumption a bit. Is Ah
invertible when 1/2 ≤h < 1? How about the case h ≥1?


---

## 第74页

Second 1×2 exercise: ill-posedness of naive
inversion
Naive reconstruction is an approximation of the unknown x, as we
can see by this calculation:
A−1
h
em = A−1
h (Ahx + ε) = x + A−1
h ε.
So we can bound the error by
∥A−1
h ε∥R2 ≤∥A−1
h ∥R2→R2∥ε∥R2,
where ∥A−1
h ∥R2→R2 is the operator norm of Ah.
▶Compute the eigenvalues λ1(h) > 0 and λ2(h) > 0 of the
matrix AT
h Ah numerically for a sequence of h values
approaching zero. The numbers sj(h) =
p
λj(h) are called
singular values of Ah. We order them so that s1 ≥s2.
▶Now ∥A−1
h ∥R2→R2 = 1/s2(h). What happens to ∥A−1
h ∥R2→R2
when h →0? What does that mean for the error bound?


---

