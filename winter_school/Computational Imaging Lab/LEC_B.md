# LEC_B.pdf

## 第1页

alessandro.lanza2@unibo.it
PhD Winter School 2023:
ADVANCED METHODS for 
MATHEMATICAL IMAGE ANALYSIS
Computational Imaging Lab (B)
mail:
Alessandro Lanza


---

## 第2页

Alternating Direction Method of Multipliers 
(ADMM)  
(two- and three-blocks)
2


---

## 第3页

From:  Boyd et al., Distributed Optimization and Statistical Learning via the Alternating 
Direction Method of Multipliers, Foundations and Trends in Machine Learning, 2010
3
ADMM
( file “01_ADMM_GENERAL.pdf” in “PAPERS_REPOSITORY” folder )


---

## 第4页

From:  Boyd et al., Distributed Optimization and Statistical Learning via the Alternating 
Direction Method of Multipliers, Foundations and Trends in Machine Learning, 2010
4
ADMM
( file “01_ADMM_GENERAL.pdf” in “PAPERS_REPOSITORY” folder )


---

## 第5页

From:  Boyd et al., Distributed Optimization and Statistical Learning via the Alternating 
Direction Method of Multipliers, Foundations and Trends in Machine Learning, 2010
5
ADMM
( file “01_ADMM_GENERAL.pdf” in “PAPERS_REPOSITORY” folder )


---

## 第6页

6
Prototypical “two-blocks” optimization problem solved by ADMM
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=
, 
, 
, 
, 
n
m
q n
q m
q
x
y
B
C
c







ADMM  (two-blocks)


---

## 第7页

7




:
,  :
  closed, proper and convex
n
m
f
g
→
+
→
+
Main assumption:
ADMM  (two-blocks)
Prototypical “two-blocks” optimization problem solved by ADMM
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=
, 
, 
, 
, 
n
m
q n
q m
q
x
y
B
C
c









---

## 第8页

8




:
,  :
  closed, proper and convex
n
m
f
g
→
+
→
+
Main assumption:
●Convex constraints can be dealt with (indicator functions of closed convex sets)
●For non-convex problems, convergence is yet an open issue
ADMM  (two-blocks)
Prototypical “two-blocks” optimization problem solved by ADMM
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=
, 
, 
, 
, 
n
m
q n
q m
q
x
y
B
C
c









---

## 第9页

9




:
,  :
  closed, proper and convex
n
m
f
g
→
+
→
+
Main assumption:
●Convex constraints can be dealt with (indicator functions of closed convex sets)
●Multi-blocks problems can also be dealt with (convergence guaranteed under
●For non-convex problems, convergence is yet an open issue
ADMM  (two-blocks)
Prototypical “two-blocks” optimization problem solved by ADMM
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=
, 
, 
, 
, 
n
m
q n
q m
q
x
y
B
C
c







more restrictive assumptions)


---

## 第10页

10
“three-blocks” optimization problem solved by ADMM
 minimize   ( )
( )
( )
subject to   
f x
g y
h z
Bx
Cy
Ez
c
+
+
+
+
=
, 
, 
, 
, 
, 
, 
n
m
p
q n
q m
q p
q
x
y
z
B
C
E
c










ADMM


---

## 第11页

11
“two-blocks” 
problem solved 
by ADMM
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=
, 
,  
, 
n
m
q
q n
q m
x
y
c
B
C







ADMM  (two-blocks)


---

## 第12页

12
The associated augmented Lagrangian function:
2
2
( , ; )
( )
( )
,
2
L
x y
f x
g y
Bx
Cy
c
Bx
Cy
c




=
+
+
+
−
+
+
−
penalty parameter
vector of Lagrange multipliers

++

q

ADMM  (two-blocks)
“two-blocks” 
problem solved 
by ADMM
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=
, 
,  
, 
n
m
q
q n
q m
x
y
c
B
C









---

## 第13页

13
Solution of the original minimization problem by seeking for saddle points of Lβ :


(
)
(
)
(
)


*
*
*
*
*
*
*
*
*
*
*
*
find
,
,
   s.t.    
,
,
,
,
, ,
                                    
,
,
n
m
q
x
y
L
x
y
L
x
y
L
x y
x
y














ADMM  (two-blocks)
The associated augmented Lagrangian function:
2
2
( , ; )
( )
( )
,
2
L
x y
f x
g y
Bx
Cy
c
Bx
Cy
c




=
+
+
+
−
+
+
−
“two-blocks” 
problem solved 
by ADMM
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=
, 
,  
, 
n
m
q
q n
q m
x
y
c
B
C









---

## 第14页

14
(
1)
( )
( )
(
1)
(
1)
( )
(
1)
( )
(
1)
(
1)
arg min
( ,
;
)
arg min
(
, ;
)
(
)
n
m
k
k
k
x
k
k
k
y
k
k
k
k
x
L
x y
y
L
x
y
Bx
Cy
c







+

+
+

+
+
+
=
=
=
+
+
−
Primal
descent
(alternating)
Dual
ascent
ADMM iterative algorithm (compute a saddle-point of Lβ):
ADMM  (two-blocks)
The associated augmented Lagrangian function:
2
2
( , ; )
( )
( )
,
2
L
x y
f x
g y
Bx
Cy
c
Bx
Cy
c




=
+
+
+
−
+
+
−
“two-blocks” 
problem solved 
by ADMM
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=
, 
,  
, 
n
m
q
q n
q m
x
y
c
B
C









---

## 第15页

15
(
1)
( )
( )
(
1)
(
1)
( )
(
1)
( )
(
1)
(
1)
arg min
( ,
;
)
arg min
(
, ;
)
(
)
n
m
k
k
k
x
k
k
k
y
k
k
k
k
x
L
x y
y
L
x
y
c
Bx
Cy







+

+
+

+
+
+
=
=
=
−
−
−
Primal
descent
(alternating)
Dual
ascent
The associated augmented Lagrangian function:
2
2
( , ; )
( )
( )
,
)
2
L
x y
f x
g y
c
Bx
Cy
c
Bx
Cy




=
+
−
−
−
+
−
−
ADMM iterative algorithm (compute a saddle-point of Lβ):
EQUIVALENT
(we use this)
ADMM  (two-blocks)
“two-blocks” 
problem solved 
by ADMM
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=
, 
,  
, 
n
m
q
q n
q m
x
y
c
B
C









---

## 第16页

16
 minimize   ( )
( )
( )
subject to   
f x
g y
h z
Bx
Cy
Ez
c
+
+
+
+
=
“three-blocks” 
problem solved 
by ADMM
, 
, 
, 
, 
, 
n
m
p
q
q n
q m
q p
x
y
z
c
B
C
E










ADMM  (three-blocks)


---

## 第17页

17
The associated augmented Lagrangian function:
2
2
( , , ; )
( )
( )
,
2
L
x y z
f x
g y
c
Bx
Cy
Ez
c
Bx
Cy
Ez




=
+
−
−
−
−
+
−
−
−
ADMM  (three-blocks)
 minimize   ( )
( )
( )
subject to   
f x
g y
h z
Bx
Cy
Ez
c
+
+
+
+
=
“three-blocks” 
problem solved 
by ADMM
, 
, 
, 
, 
, 
n
m
p
q
q n
q m
q p
x
y
z
c
B
C
E












---

## 第18页

18
Solution of the original minimization problem by seeking for saddle points of Lβ :


(
)
(
)
(
)


*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
find
,
,
,
   s.t.    
,
,
,
,
,
,
, , ,
                                    
,
,
,
n
m
p
q
x
y
z
L
x
y
z
L
x
y
z
L
x y z
x
y
z















ADMM  (three-blocks)
The associated augmented Lagrangian function:
2
2
( , , ; )
( )
( )
,
2
L
x y z
f x
g y
c
Bx
Cy
Ez
c
Bx
Cy
Ez




=
+
−
−
−
−
+
−
−
−
 minimize   ( )
( )
( )
subject to   
f x
g y
h z
Bx
Cy
Ez
c
+
+
+
+
=
“three-blocks” 
problem solved 
by ADMM
, 
, 
, 
, 
, 
n
m
p
q
q n
q m
q p
x
y
z
c
B
C
E












---

## 第19页

19
ADMM  (three-blocks)
(
1)
( )
( )
( )
(
1)
(
1)
( )
( )
(
1)
(
1)
(
1)
( )
(
1)
( )
(
1)
(
1)
(
1)
arg min
( ,
,
;
)
arg min
(
, ,
;
)
arg min
(
,
, ;
)
(
)
n
m
p
k
k
k
k
x
k
k
k
k
y
k
k
k
k
z
k
k
k
k
k
x
L
x y
z
y
L
x
y z
z
L
x
y
z
c
Bx
Cy
Ez









+

+
+

+
+
+

+
+
+
+
=
=
=
=
−
−
−
−
Primal
descent
(alternating)
Dual
ascent
ADMM iterative algorithm (compute a saddle-point of Lβ):
The associated augmented Lagrangian function:
2
2
( , , ; )
( )
( )
,
2
L
x y z
f x
g y
c
Bx
Cy
Ez
c
Bx
Cy
Ez




=
+
−
−
−
−
+
−
−
−
 minimize   ( )
( )
( )
subject to   
f x
g y
h z
Bx
Cy
Ez
c
+
+
+
+
=
“three-blocks” 
problem solved 
by ADMM
, 
, 
, 
, 
, 
n
m
p
q
q n
q m
q p
x
y
z
c
B
C
E












---

## 第20页

20
ADMM  convergence
●Convergence of ADMM in the two- and three-blocks cases has been studied
in different manners
●For the two-blocks case, you can refer to the standard proof given in
●For the three-blocks case (for a special situation, with a generalization
on the usage of different penalty parameters β for different constraints),
Boyd et al., Distributed Optimization and Statistical Learning via the Alternating Direction 
Method of Multipliers, Foundations and Trends in Machine Learning, 2010
( file “01_ADMM_GENERAL.pdf” in “PAPERS_REPOSITORY” folder )
C. Wu et al., Augmented Lagrangian Method for Total Variation Restoration with 
Non-quadratic Fidelity, Inverse Problems and Imaging, 2011
( file “02_TV_NQ_FIDELITY_ADMM.pdf” in “PAPERS_REPOSITORY” folder )
refer to the proof in


---

## 第21页

Alternating Direction Method of Multipliers 
(ADMM)  
for the numerical solution of the
(unconstrained)
TV-L2 , TIK-L1 , TV-L1 models
21


---

## 第22页

ADMM (two-blocks)  
for the numerical solution of the
(unconstrained)
TV-L2 model
22


---

## 第23页

The unconstrained convex non-smooth model:
ADMM  for TV-L2 model


2
2
*
*
2
2
,
1
,
arg min
( , )
 subject to (s.t.)
2
d
d
d
i
u
t
i
u t
G u t
Au
b
t
t
Du



=


=
=
−
+
=





Variables splitting:
(
)
(
)
,
2
,
,
h
h i
h
h
i
i
v i
v
v
v
i
D u
t
t
D u
t
Du
t
t
t
D u
D u








=

=
=



















Split (linearly constrained) model:
(
)
(
)
2
2
2
*
2
1
arg min
( )
2
d
d
h
v
i
i
u
i
u
J u
Au
b
D u
D u


=


=
=
−
+
+





23


---

## 第24页

Split (linearly constrained) model:


2
2
*
*
2
2
,
1
,
arg min
( , )
s.t.
2
d
d
d
i
u
t
i
u t
G u t
Au
b
t
t
Du



=


=
=
−
+
=





ADMM  for TV-L2 model
24


---

## 第25页

Split (linearly constrained) model:


2
2
*
*
2
2
,
1
,
arg min
( , )
s.t.
2
d
d
d
i
u
t
i
u t
G u t
Au
b
t
t
Du



=


=
=
−
+
=





ADMM  for TV-L2 model
25




2
*
*
2
2
,
,
arg min
( , )
( )
( )
s.t.
(
)
0
d
d
d
d
u
t
u t
G u t
f u
g t
Du
I
t


=
=
+
+ −
=
It can be equivalently rewritten as:
2
2
2
1
with    ( )
,   ( )
,   both closed, proper, convex
2
d
i
i
f u
Au
b
g t
t

=
=
−
= 


---

## 第26页

Split (linearly constrained) model:


2
2
*
*
2
2
,
1
,
arg min
( , )
s.t.
2
d
d
d
i
u
t
i
u t
G u t
Au
b
t
t
Du



=


=
=
−
+
=





ADMM  for TV-L2 model
26




2
*
*
2
2
,
,
arg min
( , )
( )
( )
s.t.
(
)
0
d
d
d
d
u
t
u t
G u t
f u
g t
Du
I
t


=
=
+
+ −
=
It can be equivalently rewritten as:
2
2
2
1
with    ( )
,   ( )
,   both closed, proper, convex
2
d
i
i
f u
Au
b
g t
t

=
=
−
= 
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=
Is it a standard     
“two-blocks” problem?


---

## 第27页

Split (linearly constrained) model:


2
2
*
*
2
2
,
1
,
arg min
( , )
s.t.
2
d
d
d
i
u
t
i
u t
G u t
Au
b
t
t
Du



=


=
=
−
+
=





ADMM  for TV-L2 model
27




2
*
*
2
2
,
,
arg min
( , )
( )
( )
s.t.
(
)
0
d
d
d
d
u
t
u t
G u t
f u
g t
Du
I
t


=
=
+
+ −
=
It can be equivalently rewritten as:
2
2
2
1
with    ( )
,   ( )
,   both closed, proper, convex
2
d
i
i
f u
Au
b
g t
t

=
=
−
= 
Is it a standard     
“two-blocks” problem?
2
2
,
,
,
,
0
d
d
x
u y
t B
D C
I
c
=
=
=
= −
=
YES!
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=


---

## 第28页

The augmented Lagrangian function:
Solution image as the saddle point of the above function, that is:
0

penalty parameter
vector of Lagrange 
multipliers
2
2
2
2
2
1
( , ; )
,
2
2
d
i
i
L u t
Au
b
t
t
Du
t
Du




=
=
−
+
−
−
+
−

Split (linearly constrained) model:


2
2
*
*
2
2
,
1
,
arg min
( , )
s.t.
2
d
d
d
i
u
t
i
u t
G u t
Au
b
t
t
Du



=


=
=
−
+
=





ADMM  for TV-L2 model
2
h
d
v





=





28


(
)
(
)
(
)
*
*
*
*
*
*
*
*
*
2
2
find
, ,
   s.t.    
, ,
, ,
, ,
                                    ( , , )
d
d
d
u t
L u t
L u t
L u t
u t













---

## 第29页

ADMM-based iterative algorithm (split optimization sub-problems):
2
(
1)
( )
( )
(
1)
(
1)
( )
(
1)
( )
(
1)
(
1)
arg min (
, ;
)
arg min ( ,
;
)
(
)
d
d
k
k
k
t R
k
k
k
u R
k
k
k
k
t
L u
t
u
L u t
t
Du





+

+
+

+
+
+
=
=
=
−
−
Primal
descent
(alternating)
Dual
ascent
s.p.d. linear
system
closed-form
closed-form
proximal map
Augmented Lagrangian function:
… saddle point ??
2
2
2
2
2
1
( , ; )
,
2
2
d
i
i
L u t
Au
b
t
t
Du
t
Du




=
=
−
+
−
−
+
−

ADMM  for TV-L2 model
29


---

## 第30页

Subproblem for primal variable t :
Primal descent: subproblem for t
Augmented Lagrangian function:
2
2
2
2
2
1
( , ; )
,
2
2
d
i
i
L u t
Au
b
t
t
Du
t
Du




=
=
−
+
−
−
+
−

2
(
1)
( )
( )
arg min
(
, ;
)
d
k
k
k
t
t
L u
t 
+

=
2
2
( )
( )
( )
2
2
1
arg min
,
2
d
d
k
k
k
i
t
i
t
t
Du
t
Du



=


=
−
−
+
−





… drop the terms which do not depend on t …
2
2
( )
( )
( )
( )
2
2
2
1
1
arg min
,   
2
d
d
k
k
k
k
d
i
t
i
t
t
q
q
Du




=


=
+
−
=
+






30


---

## 第31页

… the cost function is the sum of d “separate” bivariate functions, in fact:
(
)
2
2
(
1)
( )
( )
( )
( )
2
2
2
1
1
arg min
,
,
2
d
d
k
k
k
k
k
i
i
i
i
i
i
t
i
t
t
t
q
q
Du



+

=




=
+
−
=
+










… the problem thus reduces to d independent bivariate minimizations:
2
2
(
1)
( )
2
2
arg min
,
1,
,
2
i
k
k
i
i
i
i
t
t
t
t
q
i
d

+



=
+
−
=




… which all admit closed-form solution (see Proposition 1 in the next slide):
( )
(
1)
( )
2
( )
2
1
max
,0
,
1,
,
k
k
k
i
i
i
k
i
q
t
q
i
d
q

+


=
−
=




Primal descent: subproblem for t
1,
,
i
d
=
linear computational
complexity O(d)
31


---

## 第32页

*
2
2
1
max
,0
v
x
v
v



=
−




Proposition 1 ( proximal map of ||x||2 )
Let                               be given constants. Then, the optimization problem
is strongly convex and admits the unique solution given by the following 
“shrinkage”  (or soft-thresholding) operator:
2
2
2
*
2
2
prox
( )
arg min
(x; )
2
x
x
x
v
v
x
x
v






=
=
=
+
−




(where  0 · 0 / 0 = 0 is assumed)
Primal descent: subproblem for t
2
, v

++


32


---

## 第33页

Subproblem for primal variable u:
Primal descent: subproblem for u
Augmented Lagrangian function:
2
2
2
2
2
1
( , ; )
,
2
2
d
i
i
L u t
Au
b
t
t
Du
t
Du




=
=
−
+
−
−
+
−

(
1)
(
1)
( )
arg min
( ,
;
)
d
k
k
k
u
u
L u t

+
+

=
2
2
( )
(
1)
2
2
arg min
( )
,
2
2
d
k
k
u
Z u
Au
b
Du
t
Du



+



=
=
−
+
+
−




… drop the terms which do not depend on u …
The function Z is quadratic in the optimization variable u, hence its global
minimizers (if there exists one) are to be sought among its stationary points:
(
1)
  set of solutions of :     
( )
0
k
d
u
Z u
+ 

=
33


---

## 第34页

Primal descent: subproblem for u
The coefficient matrix is almost identical to the one previously obtained for
… after some simple algebraic manipulations:
(
1)
( )
1
( )
0   
  
T
T
T
k
k
T
d
Z u
D D
A A u
D
t
A b






+





=

+
=
−
+








the TIK-L2 model (see…): it is symmetric, positive definite, and has full rank,
hence the linear system admits a unique solution giving the new iterate u(k+1)
Assuming periodic/reflective/anti-reflective boundary conditions for u,
the linear system can be solved (like for TIK-L2 case) by 2D DFT/DCT/DST.
By using 2D FFT/FCT/FST implementations   
computational
complexity O(d log d)
34
RESTORATION:


---

## 第35页

Primal descent: subproblem for u
The coefficient matrix is almost identical to the one previously obtained for
… after some simple algebraic manipulations:
(
1)
( )
1
( )
0   
  
T
T
T
k
k
T
d
Z u
D D
A A u
D
t
A b






+





=

+
=
−
+








the TIK-L2 model (see…): it is symmetric, positive definite, and has full rank,
hence the linear system admits a unique solution giving the new iterate u(k+1)
the linear system can be solved (like for TIK-L2 case) by iterative (P)CG
35
INPAINTING:


---

## 第36页

ADMM (two-blocks)
for the numerical solution of the
(unconstrained)
TIK-L1 model
36


---

## 第37页

The unconstrained convex non-smooth model:
ADMM  for TIK-L1 model


2
*
*
1
2
,
1
,
arg min
( , )
 s.t.  
2
d
d
u R
r R
u r
G u r
r
Du
r
Au
b





=
=
+
=
−




Variables splitting:
d
r
Au
b
=
−

Split (linearly constrained) model:
2
*
1
2
1
arg min
( )
2
d
u
u
J u
Au
b
Du




=
=
−
+




… residue image
(noise image estimate)
37


---

## 第38页

Split (linearly constrained) model:
38


2
*
*
1
2
,
1
,
arg min
( , )
 s.t.  
2
d
d
u R
r R
u r
G u r
r
Du
r
Au
b





=
=
+
=
−




ADMM  for TIK-L1 model


---

## 第39页

Split (linearly constrained) model:
39




*
*
,
,
arg min
( , )
( )
( )
s.t.
(
)
d
d
d
u R
r R
u r
G u r
f u
g r
Au
I r
b


=
=
+
+ −
=
It can be equivalently rewritten as:
2
2
1
1
with    ( )
,   ( )
,   both closed, proper, convex
2
f u
Du
g r
r

=
=


2
*
*
1
2
,
1
,
arg min
( , )
 s.t.  
2
d
d
u R
r R
u r
G u r
r
Du
r
Au
b





=
=
+
=
−




ADMM  for TIK-L1 model


---

## 第40页

40
Is it a standard     
“two-blocks” problem?
Split (linearly constrained) model:




*
*
,
,
arg min
( , )
( )
( )
s.t.
(
)
d
d
d
u R
r R
u r
G u r
f u
g r
Au
I r
b


=
=
+
+ −
=
It can be equivalently rewritten as:
2
2
1
1
with    ( )
,   ( )
,   both closed, proper, convex
2
f u
Du
g r
r

=
=


2
*
*
1
2
,
1
,
arg min
( , )
 s.t.  
2
d
d
u R
r R
u r
G u r
r
Du
r
Au
b





=
=
+
=
−




ADMM  for TIK-L1 model
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=


---

## 第41页

41
Is it a standard     
“two-blocks” problem?
,
,
,
,
d
x
u y
r B
A C
I
c
b
=
=
=
= −
=
YES!
Split (linearly constrained) model:




*
*
,
,
arg min
( , )
( )
( )
s.t.
(
)
d
d
d
u R
r R
u r
G u r
f u
g r
Au
I r
b


=
=
+
+ −
=
It can be equivalently rewritten as:
2
2
1
1
with    ( )
,   ( )
,   both closed, proper, convex
2
f u
Du
g r
r

=
=


2
*
*
1
2
,
1
,
arg min
( , )
 s.t.  
2
d
d
u R
r R
u r
G u r
r
Du
r
Au
b





=
=
+
=
−




ADMM  for TIK-L1 model
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=


---

## 第42页

The augmented Lagrangian function:
Solution image as the saddle point of the above function, that is:
0

penalty parameter
vector of Lagrange 
multipliers
(
)
(
)
2
2
1
2
2
1
( , ; )
,
2
2
L u r
r
Du
r
Au
b
r
Au
b




=
+
−
−
−
+
−
−
Split (linearly constrained) model:


(
)
(
)
(
)
*
*
*
*
*
*
*
*
*
find
,
,
   s.t.    
,
,
,
,
, ,
                                    
( , , )
d
d
d
u r
L u r
L u r
L u r
u r











d

ADMM  for TIK-L1 model


2
*
*
1
2
,
1
,
arg min
( , )
 s.t.  
2
d
d
u R
r R
u r
G u r
r
Du
r
Au
b





=
=
+
=
−




42


---

## 第43页

ADMM-based iterative algorithm (split optimization sub-problems):
(
1)
( )
( )
(
1)
(
1)
( )
(
1)
( )
(
1)
(
1)
arg min (
,
;
)
arg min ( ,
;
)
(
(
))
d
d
k
k
k
r R
k
k
k
u R
k
k
k
k
r
L u
r
u
L u r
r
Au
b





+

+
+

+
+
+
=
=
=
−
−
−
Primal
descent
(alternating)
Dual
ascent
s.p.d. linear
system
closed-form
closed-form
proximal map
Augmented Lagrangian function:
… saddle point ??
ADMM  for TIK-L1 model
(
)
(
)
2
2
1
2
2
1
( , ; )
,
2
2
L u r
r
Du
r
Au
b
r
Au
b




=
+
−
−
−
+
−
−
43


---

## 第44页

Subproblem for primal variable r :
Primal descent: subproblem for r
Augmented Lagrangian function:
(
1)
( )
( )
arg min
(
,
;
)
d
k
k
k
r
r
L u
r 
+

=
(
)
(
)
2
( )
( )
( )
1
2
arg min
,
2
d
k
k
k
r
r
r
Au
b
r
Au
b






=
−
−
−
+
−
−




… drop the terms which do not depend on r …
2
( )
1
2
arg min
,
2
d
k
r
r
r
q




=
+
−




(
)
(
)
2
2
1
2
2
1
( , ; )
,
2
2
L u r
r
Du
r
Au
b
r
Au
b




=
+
−
−
−
+
−
−
( )
( )
( )
1
k
k
k
d
q
Au
b


=
−
+

/



++
=

44


---

## 第45页

… the cost function is the sum of d “separate” univariate functions, in fact:
(
)
(
)
2
(
1)
( )
( )
( )
( )
1
1
arg min
,
,
2
d
d
k
k
k
k
k
i
i
i
i
i
i
r
i
r
r
r
q
q
Au
b



+

=




=
+
−
=
−
+










… the problem thus reduces to d independent univariate minimizations:
(
)
2
(
1)
( )
arg min
,
1,
,
2
i
k
k
i
i
i
i
r
r
r
r
q
i
d

+



=
+
−
=




… which all admit closed-form solution (see Proposition 2 in the next slide):
Primal descent: subproblem for r
1,
,
i
d
=
linear computational
complexity O(d)
45
(
)
(
1)
( )
( )
1
 sign
max
,0
,
1,
,
k
k
k
i
i
i
r
q
q
i
d

+


=
−
=






---

## 第46页

Proposition 2 ( proximal map of |x| )
Let                               be given constants. Then, the optimization problem
is strongly convex and admits the unique solution given by the following 
“shrinkage”  (or soft-thresholding) operator:
(
)
2
*
prox ( )
arg min
(x; )
2
x
x
x
v
v
x
x
v






=
=
=
+
−




Primal descent: subproblem for t
, v

++


46
*
1
sign( ) max
,0
x
v
v



=

−




where    sign( )
1 for 
0,  +1 for 
0,  0 for 
0
v
v
v
v
= −


=


---

## 第47页

Subproblem for primal variable u:
Primal descent: subproblem for u
(
1)
(
1)
( )
arg min
( ,
;
)
d
k
k
k
u
u
L u r

+
+

=
(
)
2
2
( )
(
1)
2
2
1
arg min
( )
,
2
2
d
k
k
u
Z u
Du
Au
r
Au
b


+



=
=
+
+
−
−




… drop the terms which do not depend on u …
The function Z is quadratic in the optimization variable u, hence its global
minimizers (if there exists one) are to be sought among its stationary points:
(
1)
  set of solutions of :     
( )
0
k
d
u
Z u
+ 

=
Augmented Lagrangian function:
(
)
(
)
2
2
1
2
2
1
( , ; )
,
2
2
L u r
r
Du
r
Au
b
r
Au
b




=
+
−
−
−
+
−
−
47


---

## 第48页

Primal descent: subproblem for u
The coefficient matrix is almost identical to the one previously obtained for
… after some simple algebraic manipulations:
(
1)
( )
1
1
( )
0   
  
T
T
T
k
k
d
Z u
D D
A A u
A
r
b



+





=

+
=
−
+








the TIK-L2 and TV-L2 models (see…): it is  s. p. d. and has full rank, hence
the linear system admits a unique solution giving the new iterate u(k+1)
48
Assuming periodic/reflective/anti-reflective boundary conditions for u,
the linear system can be solved (like for TIK-L2 case) by 2D DFT/DCT/DST.
By using 2D FFT/FCT/FST implementations   
computational
complexity O(d log d)
RESTORATION:


---

## 第49页

Primal descent: subproblem for u
The coefficient matrix is almost identical to the one previously obtained for
… after some simple algebraic manipulations:
(
1)
( )
1
1
( )
0   
  
T
T
T
k
k
d
Z u
D D
A A u
A
r
b



+





=

+
=
−
+








the TIK-L2 and TV-L2 models (see…): it is  s. p. d. and has full rank, hence
the linear system admits a unique solution giving the new iterate u(k+1)
49
the linear system can be solved (like for TIK-L2 case) by iterative (P)CG
INPAINTING:


---

## 第50页

ADMM (three-blocks → two-blocks)
for the numerical solution of the
(unconstrained)
TV-L1 model
50


---

## 第51页

The unconstrained convex non-smooth model:
ADMM  for TV-L1 model
Variables splitting:
Split (linearly constrained) model:
(
)
(
)
2
2
*
( )
( )
1
1
arg min
( )
d
d
h
v
i
i
u
i
u
J u
Au
b
D
u
D u


=



=
−
+
+





d
r
Au
b
=
−



2
*
*
*
1
2
,
,
1
, ,
arg min
( , , )
s.t.
,
d
d
d
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b



=



=
+
=
=
−





51
(
)
(
)
,
2
,
,
h
h i
h
h
i
i
v i
v
v
v
i
D u
t
t
D u
t
Du
t
t
t
D u
D u








=

=
=





















---

## 第52页

52
Split (linearly constrained) model:


2
*
*
*
1
2
,
,
1
, ,
arg min
( , , )
s.t.
,
d
d
d
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b



=



=
+
=
=
−





ADMM  for TV-L1 model


---

## 第53页

Split (linearly constrained) model:
53




2
*
*
*
,
,
2
2
2
2
, ,
arg min
( , , )
( )
( )
( )
0
0
s.t.
0
d
d
u r
t
d
d d
d
d
d
d
u t r
G u t r
f u
g t
h r
I
D u
t
r
I
A
b





=
+
+
−








+
+
=








−








It can be equivalently rewritten as:
all closed, proper, convex
with
2
1
1
( )
0, ( )
, ( )
,
d
i
i
f u
g t
t
h r
r

=
=
=
=



2
*
*
*
1
2
,
,
1
, ,
arg min
( , , )
s.t.
,
d
d
d
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b



=



=
+
=
=
−





ADMM  for TV-L1 model


---

## 第54页

54
Is it a standard     
“three-blocks” problem?
 minimize   ( )
( )
( )
subject to   
f x
g y
h z
Bx
Cy
Ez
c
+
+
+
+
=
Split (linearly constrained) model:




2
*
*
*
,
,
2
2
2
2
, ,
arg min
( , , )
( )
( )
( )
0
0
s.t.
0
d
d
u r
t
d
d d
d
d
d
d
u t r
G u t r
f u
g t
h r
I
D u
t
r
I
A
b





=
+
+
−








+
+
=








−








It can be equivalently rewritten as:
all closed, proper, convex
with
2
1
1
( )
0, ( )
, ( )
,
d
i
i
f u
g t
t
h r
r

=
=
=
=



2
*
*
*
1
2
,
,
1
, ,
arg min
( , , )
s.t.
,
d
d
d
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b



=



=
+
=
=
−





ADMM  for TV-L1 model


---

## 第55页

55
2
2
2
2
                        
,
,
,
0
0
,
,
,
0
d
d d
d
d
d
d
x
u y
t z
r
I
D
B
C
E
c
I
A
b


=
=
=
−








=
=
=
=








−








YES!
ADMM  for TV-L1 model
Is it a standard     
“three-blocks” problem?
 minimize   ( )
( )
( )
subject to   
f x
g y
h z
Bx
Cy
Ez
c
+
+
+
+
=




2
*
*
*
,
,
2
2
2
2
, ,
arg min
( , , )
( )
( )
( )
0
0
s.t.
0
d
d
u r
t
d
d d
d
d
d
d
u t r
G u t r
f u
g t
h r
I
D u
t
r
I
A
b





=
+
+
−








+
+
=








−








It can be equivalently rewritten as:
all closed, proper, convex
with
( )
2
1
( )
0, ( )
, ( )
( ),
d
i
B
i
f u
g t
t
h r
r


=
=
=
=



2
*
*
*
1
2
,
,
1
, ,
arg min
( , , )
s.t.
,
d
d
d
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b



=



=
+
=
=
−





Split (linearly constrained) model:


---

## 第56页

56
           
,
( ; ),
( )
...,
...,
...,
...
x
u y
t r
g y
B
C
c
=
=
=
=
=
=
YES!
ADMM  for TV-L1 model
Can it also be seen as a 
standard “two-blocks” 
problem?
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=




2
*
*
*
,
,
2
2
2
2
, ,
arg min
( , , )
( )
( )
( )
0
0
s.t.
0
d
d
u r
t
d
d d
d
d
d
d
u t r
G u t r
f u
g t
h r
I
D u
t
r
I
A
b





=
+
+
−








+
+
=








−








It can be equivalently rewritten as:
all closed, proper, convex
with
( )
2
1
( )
0, ( )
, ( )
( ),
d
i
B
i
f u
g t
t
h r
r


=
=
=
=



2
*
*
*
1
2
,
,
1
, ,
arg min
( , , )
s.t.
,
d
d
d
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b



=



=
+
=
=
−





Split (linearly constrained) model:


---

## 第57页

The augmented Lagrangian function:
2
1
2
2
1
( , , ; )
,
2
d
i
i
L u t r
r
t
c
Bu
Ct
Er
c
Bu
Ct
Er




=
=
+
−
−
−
−
+
−
−
−

ADMM  for TV-L1 model
57


2
*
*
*
1
2
,
,
1
, ,
arg min
( , , )
s.t.
,
d
d
d
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b



=



=
+
=
=
−





Split (linearly constrained) model:
2
2
2
2
2
2
2
1
2
1
2
2
2
0
0
0
0
,
0
0
2
d
d
d d
d
d d
d
d
i
i
d
d
d
d
d
d
I
I
D
D
r
t
u
t
r
u
t
r
I
I
b
A
b
A





=


−
−
















=
+
−
−
−
−
+
−
−
−
















−
−



















---

## 第58页

The augmented Lagrangian function:
2
1
2
2
1
( , , ; )
,
2
d
i
i
L u t r
r
t
c
Bu
Ct
Er
c
Bu
Ct
Er




=
=
+
−
−
−
−
+
−
−
−

ADMM  for TV-L1 model
58


2
*
*
*
1
2
,
,
1
, ,
arg min
( , , )
s.t.
,
d
d
d
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b



=



=
+
=
=
−





Split (linearly constrained) model:
2
3
,
,
,
t
d
d
d
t
r
r















2
2
2
2
2
2
2
1
2
1
2
2
2
0
0
0
0
,
0
0
2
d
d
d d
d
d d
d
d
i
i
d
d
d
d
d
d
I
I
D
D
r
t
u
t
r
u
t
r
I
I
b
A
b
A





=


−
−
















=
+
−
−
−
−
+
−
−
−
















−
−



















---

## 第59页

The augmented Lagrangian function:
2
1
2
2
1
( , , ; )
,
2
d
i
i
L u t r
r
t
c
Bu
Ct
Er
c
Bu
Ct
Er




=
=
+
−
−
−
−
+
−
−
−

ADMM  for TV-L1 model
59


2
*
*
*
1
2
,
,
1
, ,
arg min
( , , )
s.t.
,
d
d
d
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b



=



=
+
=
=
−





Split (linearly constrained) model:
2
2
2
2
2
2
2
1
2
1
2
2
2
0
0
0
0
,
0
0
2
d
d
d d
d
d d
d
d
i
i
d
d
d
d
d
d
I
I
D
D
r
t
u
t
r
u
t
r
I
I
b
A
b
A





=


−
−
















=
+
−
−
−
−
+
−
−
−
















−
−

















2
2
1
2
2
2
1
( , , ;
,
)
,
,
(
)
(
)
2
2
d
t
r
i
t
r
i
L u t r
r
t
t
Du
t
Du
r
Au
b
r
Au
b






=
=
+
−
−
+
−
−
−
−
+
−
−



---

## 第60页

The augmented Lagrangian function:
2
1
2
2
1
( , , ; )
,
2
d
i
i
L u t r
r
t
c
Bu
Ct
Er
c
Bu
Ct
Er




=
=
+
−
−
−
−
+
−
−
−

ADMM  for TV-L1 model
60


2
*
*
*
1
2
,
,
1
, ,
arg min
( , , )
s.t.
,
d
d
d
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b



=



=
+
=
=
−





Split (linearly constrained) model:
2
2
2
2
2
2
2
1
2
1
2
2
2
0
0
0
0
,
0
0
2
d
d
d d
d
d d
d
d
i
i
d
d
d
d
d
d
I
I
D
D
r
t
u
t
r
u
t
r
I
I
b
A
b
A





=


−
−
















=
+
−
−
−
−
+
−
−
−
















−
−

















2
2
1
2
2
2
1
( , , ;
,
)
,
,
(
)
(
)
2
2
d
t
r
i
t
r
i
L u t r
r
t
t
Du
t
Du
r
Au
b
r
Au
b






=
=
+
−
−
+
−
−
−
−
+
−
−

●It is like if we “Lagrange-augment the two constraints t = Du, r = Au-b separately”


---

## 第61页

The augmented Lagrangian function:
2
1
2
2
1
( , , ; )
,
2
d
i
i
L u t r
r
t
c
Bu
Ct
Er
c
Bu
Ct
Er




=
=
+
−
−
−
−
+
−
−
−

ADMM  for TV-L1 model
61


2
*
*
*
1
2
,
,
1
, ,
arg min
( , , )
s.t.
,
d
d
d
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b



=



=
+
=
=
−





Split (linearly constrained) model:
2
2
2
2
2
2
2
1
2
1
2
2
2
0
0
0
0
,
0
0
2
d
d
d d
d
d d
d
d
i
i
d
d
d
d
d
d
I
I
D
D
r
t
u
t
r
u
t
r
I
I
b
A
b
A





=


−
−
















=
+
−
−
−
−
+
−
−
−
















−
−

















2
2
1
2
2
2
1
( , , ;
,
)
,
,
(
)
(
)
2
2
d
t
r
t
r
i
t
r
i
L u t r
r
t
t
Du
t
Du
r
Au
b
r
Au
b






=
=
+
−
−
+
−
−
−
−
+
−
−

●It is possible (and we do it) to use two different β values for the two constraints
●It is like if we “Lagrange-augment the two constraints t = Du, r = Au-b separately”


---

## 第62页

The augmented Lagrangian function:
2
1
2
2
1
( , , ; )
,
2
d
i
i
L u t r
r
t
c
Bu
Ct
Er
c
Bu
Ct
Er




=
=
+
−
−
−
−
+
−
−
−

ADMM  for TV-L1 model
62


2
*
*
*
1
2
,
,
1
, ,
arg min
( , , )
s.t.
,
d
d
d
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b



=



=
+
=
=
−





Split (linearly constrained) model:
2
2
2
2
2
2
2
1
2
1
2
2
2
0
0
0
0
,
0
0
2
d
d
d d
d
d d
d
d
i
i
d
d
d
d
d
d
I
I
D
D
r
t
u
t
r
u
t
r
I
I
b
A
b
A





=


−
−
















=
+
−
−
−
−
+
−
−
−
















−
−

















2
2
1
2
2
2
1
( , , ;
,
)
,
,
(
)
(
)
2
2
d
t
r
t
r
i
t
r
i
L u t r
r
t
t
Du
t
Du
r
Au
b
r
Au
b






=
=
+
−
−
+
−
−
−
−
+
−
−

●It is possible (and we do it) to use two different β values for the two constraints
●Convergence of this “three-blocks ADMM” with different β values has been proved in
C. Wu et al., Augmented Lagrangian Method for Total Variation Restoration with Non-quadratic 
Fidelity, Inverse Problems and Imaging, 2011 (file “02_TV_NQ_FIDELITY_ADMM” in REPOSITORY)
●It is like if we “Lagrange-augment the two constraints t = Du, r = Au-b separately”


---

## 第63页

The augmented Lagrangian function:
Solution image as the saddle point of the above function, that is:
2
1
2
2
1
( , , ;
,
)
,
2
d
t
t
r
i
t
i
L u t r
r
t
t
Du
t
Du




=
=
+
−
−
+
−

Split (linearly constrained) model:
ADMM  for TV-L1 model


2
*
*
*
1
2
,
,
1
, ,
arg min
( , , )
s.t.
,
d
d
d
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b



=



=
+
=
=
−





2
2
,
(
)
(
)
2
r
r r
Au
b
r
Au
b


−
−
−
+
−
−
like for TIK-L1
like for TV-L2
63


(
)
(
)
(
)
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
2
2
find
, ,
,
,
   s.t.    
, ,
,
,
, ,
,
,
, , ,
,
                                              ( , , ,
,
)
t
r
t
r
t
r
t
r
d
d
d
d
d
t
r
u t r
L u t r
L u t r
L u t r
u t r



















---

## 第64页

ADMM-based iterative algorithm (split optimization sub-problems):
Primal
descent
(alternating)
Dual
ascent
s.p.d. linear
system
closed-form
closed-form
proximal map
Saddle point of the Augmented Lagrangian function?
closed-form
proximal map
ADMM  for TV-L1 model
64
2
(
1)
( )
( )
( )
( )
(
1)
( )
(
1)
( )
( )
(
1)
(
1)
(
1)
( )
( )
(
1)
( )
(
1)
(
1)
(
1)
( )
(
1)
(
1)
arg min (
, ,
;
,
)
arg min (
,
, ;
,
)
arg min ( ,
,
;
,
)
(
)
(
(
d
d
d
k
k
k
k
k
t
r
t
k
k
k
k
k
t
r
r
k
k
k
k
k
t
r
u
k
k
k
k
t
t
t
k
k
k
k
r
r
r
t
L u
t r
r
L u
t
r
u
L u t
r
t
Du
r
Au












+

+
+

+
+
+

+
+
+
+
+
+
=
=
=
=
−
−
=
−
−
))
b
−


---

## 第65页

Subproblem for primal variable t :
Primal descent: subproblem for t
Augmented Lagrangian function:
2
(
1)
( )
( )
( )
( )
arg min
(
, ,
;
,
)
d
k
k
k
k
k
t
r
t
t
L u
t r


+

=
… exactly the same as for TV-L2 …
2
1
2
2
1
( , , ;
,
)
,
2
d
t
t
r
i
t
i
L u t r
r
t
t
Du
t
Du




=
=
+
−
−
+
−

2
2
,
(
)
(
)
2
r
r r
Au
b
r
Au
b


−
−
−
+
−
−
… closed-form solution (see Proposition 1) of d independent 2D problems:
(
)
2
2
(
1)
( )
( )
( )
( )
2
,
2
2
1
arg min
,   
,
2
i
k
k
k
k
k
t
i
i
i
i
i
t i
i
t
t
t
t
t
q
q
Du



+





=
+
−
=
+









65
( )
( )
2
( )
2
1
max
,0
,
1,
,
k
k
i
i
k
t
i
q
q
i
d
q



=
−
=




linear computational
complexity O(d)


---

## 第66页

Subproblem for primal variable r :
Primal descent: subproblem for r
Augmented Lagrangian function:
2
1
2
2
1
( , , ;
,
)
,
2
d
t
t
r
i
t
i
L u t r
r
t
t
Du
t
Du




=
=
+
−
−
+
−

2
2
,
(
)
(
)
2
r
r r
Au
b
r
Au
b


−
−
−
+
−
−
(
1)
( )
(
1)
( )
( )
arg min
(
,
, ;
,
)
d
k
k
k
k
k
t
r
r
r
L u
t
r 

+
+

=
… exactly the same as for TIK-L1 …
… closed-form solution (see Proposition 2) of d independent 1D problems:
linear computational
(
)
(
)
2
(
1)
( )
( )
( )
( )
,
1
arg min
,
,
2
i
k
k
k
k
k
i
i
i
i
i
r i
i
r
r
r
r
r
w
w
Au
b



+





=
+
−
=
−
+









/
r



=
66
(
)
( )
( )
1
 sign
max
,0
,
1,
,
k
k
i
i
w
w
i
d



=
−
=




complexity O(d)


---

## 第67页

Subproblem for primal variable u:
Primal descent: subproblem for u
(
1)
(
1)
(
1)
( )
( )
arg min
( ,
,
;
,
)
d
k
k
k
k
k
t
r
u
u
L u t
r


+
+
+

=
(
)
2
2
( )
(
1)
( )
(
1)
2
2
arg min
( )
,
,
2
2
d
k
k
k
k
t
r
t
r
u
Z u
Du
t
Du
Au
r
Au
b




+
+



=
=
+
−
+
+
−
−




… drop the terms independent of  u …
The function Z is quadratic in the optimization variable u, hence its global
minimizers (if there exists one) are to be sought among its stationary points:
(
1)
  set of solutions of :     
( )
0
k
d
u
Z u
+ 

=
Augmented Lagrangian function:
2
1
2
2
1
( , , ;
,
)
,
2
d
t
t
r
i
t
i
L u t r
r
t
t
Du
t
Du




=
=
+
−
−
+
−

2
2
,
(
)
(
)
2
r
r r
Au
b
r
Au
b


−
−
−
+
−
−
67


---

## 第68页

Primal descent: subproblem for u
The coefficient matrix is almost identical to the one previously obtained for
… after some simple algebraic manipulations:
(
1)
( )
(
1)
( )
1
1
( )
0   
  
T
T
T
k
k
T
k
k
r
r
d
t
r
t
t
t
r
Z u
D D
A A u
D
t
A
r
b








+
+







=

+
=
−
+
−
+












the TIK-L2 , TV-L2 and TIK-L1 models (see…): it is  s. p. d. and has full rank,
hence the linear system admits a unique solution giving the new iterate u(k+1)
68
Assuming periodic/reflective/anti-reflective boundary conditions for u,
the linear system can be solved (like for TIK-L2 case) by 2D DFT/DCT/DST.
By using 2D FFT/FCT/FST implementations   
computational
complexity O(d log d)
RESTORATION:


---

## 第69页

Primal descent: subproblem for u
The coefficient matrix is almost identical to the one previously obtained for
… after some simple algebraic manipulations:
(
1)
( )
(
1)
( )
1
1
( )
0   
  
T
T
T
k
k
T
k
k
r
r
d
t
r
t
t
t
r
Z u
D D
A A u
D
t
A
r
b








+
+







=

+
=
−
+
−
+












the TIK-L2 , TV-L2 and TIK-L1 models (see…): it is  s. p. d. and has full rank,
hence the linear system admits a unique solution giving the new iterate u(k+1)
69
the linear system can be solved (like for TIK-L2 case) by iterative (P)CG
INPAINTING:


---

## 第70页

ADMM (three-blocks → two-blocks)
for the numerical solution of the
(“discrepancy”-constrained)
TV-L2 model
70


---

## 第71页

• Discrepancy Principle (DP) definition: for any variational model of the form:
Discrepancy-constrained Variational Models 
(for L2 fidelity terms: additive white Gaussian noise)
( )
2
*
2
arg min
( ; )
( )
,  with    any regularizer,
2
d
u
u
J u
R u
Au
b
R






=
=
+
−




71
and with:
d d
A
K

=

blurring matrix (for image restoration)
d d
A
S

=

inpainting matrix (for image inpainting)


---

## 第72页

• Discrepancy Principle (DP) definition: for any variational model of the form:
Discrepancy-constrained Variational Models 
(for L2 fidelity terms: additive white Gaussian noise)
( )
2
*
2
arg min
( ; )
( )
,  with    any regularizer,
2
d
u
u
J u
R u
Au
b
R






=
=
+
−




72
choose μ such that the solution u*(μ) satisfies the dicrepancy constraint:


2
*
2
2
ˆ
ˆ
( )
 ( )
:
, 
( )
, 
1, 
 est. of noise stdv
d
n
n
u
u
Au
b
d











−

=
=
• DP rationale: we aim at solutions u*(μ) which are as near as possible to the sought
2
2
2
2
2
2
 
 
 
 
( ) with 
1
true
true
true
n
b
Au
n
Au
b
n
Au
b
n
d



=
+

−
=

−
=

=
=
clean image, utrue, which, according to the linear degradation model, satisfies:
Hence, by DP we impose that the residual of u*(μ) has the same variance of noise
• DP usefulness: tuning (by hand) the best (in terms of obtained restoration results)
regularization parameter μ of unconstrained models can be a long and tedious task.
If we are able to compute a good estimate      of noise standard deviation, then 
ˆn

imposing directly the discrepancy constraint – i.e., using discrepancy-constrained
variational models – allows to obtain in one shot a good restoration!


---

## 第73页

The Unconstrained (U) and (discrepancy) Constrained (C) TV-L2 models:
ADMM  for discrepancy-constrained TV-L2 model
( )
2
*
2
arg min
( ; )
TV( )
2
d
U
U
u
u
J
u
u
Au
b






=
=
+
−




( )




2
*
2
( )
2
arg min
( ; )
TV( )
( ) ,  ( )
:
d
d
C
C
u
u
J
u
u
u
u
Au
b







=
=
+
=

−

73
… seen in previous slides
Discrepancy 
set / constraint
hyper-ellipsoidal 
shape


---

## 第74页

The Unconstrained (U) and (discrepancy) Constrained (C) TV-L2 models:
ADMM  for discrepancy-constrained TV-L2 model
( )
2
*
2
arg min
( ; )
TV( )
2
d
U
U
u
u
J
u
u
Au
b






=
=
+
−




( )




2
*
2
( )
2
arg min
( ; )
TV( )
( ) ,  ( )
:
d
d
C
C
u
u
J
u
u
u
u
Au
b







=
=
+
=

−

74
0
where the indicator function 
( ) of a set  is defined as  
( )
 ,
S
S
if x
S
x
S
x
if x
S




= +


… seen in previous slides
such that the constrained model above can also be equivalently written as
( )


2
*
2
2
( )
arg min TV( ),  
( )
:
d
C
u
u
u
u
Au
b





=
=

−

For our purposes, it is convenient to consider the form with the indicator function!
discrepancy set
discrepancy constraint


---

## 第75页

The Unconstrained (U) and (discrepancy) Constrained (C) TV-L2 models:
ADMM  for discrepancy-constrained TV-L2 model
( )
2
*
2
arg min
( ; )
TV( )
2
d
U
U
u
u
J
u
u
Au
b






=
=
+
−




( )




2
*
2
( )
2
arg min
( ; )
TV( )
( ) ,  ( )
:
d
d
C
C
u
u
J
u
u
u
u
Au
b







=
=
+
=

−

75
The constrained model can be equivalently and usefully rewritten (by rewriting in
… seen in previous slides
an equivalent form the discrepancy constraint) as follows:
( )




2
*
2
( )
2
arg min
( ; )
TV( )
(
) ,  ( )
:
d
d
C
C
B
u
u
J
u
u
Au
b
B
x
x







=
=
+
−
=


2
where  ( ) is the  ball of  
 with center the origin and radius 
d
B
l




---

## 第76页

The Unconstrained (U) and (discrepancy) Constrained (C) TV-L2 models:
ADMM  for discrepancy-constrained TV-L2 model
( )
2
*
2
arg min
( ; )
TV( )
2
d
U
U
u
u
J
u
u
Au
b






=
=
+
−




( )




2
*
2
( )
2
arg min
( ; )
TV( )
(
) ,  ( )
:
d
d
C
C
B
u
u
J
u
u
Au
b
B
x
x







=
=
+
−
=


76
… seen in previous slides
If we know (or we are able to estimate) the noise standard deviation 
, then the
n

above constrained model (unlike the unconstrained one) allows us to automatically
*
obtain a good-quality solution 
( ) by selecting 
, with 
1, as in this
C
n
u
d





=
way we are imposing that the standard deviation of the solution residual image
*
*
( )
( )
  is approximately equal to the noise standard deviation 
 ...
C
C
n
r
Ku
b



=
−
... this is called the   DISCREPANCY PRINCIPLE
Motivation for the constrained model (numerically more challenging):


---

## 第77页

The Unconstrained (U) and (discrepancy) Constrained (C) TV-L2 models:
ADMM  for discrepancy-constrained TV-L2 model
( )
2
*
2
arg min
( ; )
TV( )
2
d
U
U
u
u
J
u
u
Au
b






=
=
+
−




( )




2
*
2
( )
2
arg min
( ; )
TV( )
(
) ,  ( )
:
d
d
C
C
B
u
u
J
u
u
Au
b
B
x
x







=
=
+
−
=


77
… seen in previous slides
“Equivalence” of the unconstrained and constrained models


---

## 第78页

The (discrepancy) constrained convex non-smooth model:
(
)
(
)
(
)
2
2
*
( )
1
arg min
( ; )
,  with :
d
d
B
h
v
i
i
u
i
u
J u
Au
b
D u
D u




=


=
=
−
+
+







2
2
2
( )
:
,
d
B
x
x


=


Variables splitting:
Split (linearly constrained) model:
ADMM  for discrepancy-constrained TV-L2 model
78


2
*
*
*
( )
2
,
,
1
, ,
arg min
( , , )
( )
s.t.
,
d
d
d
B
i
u r
t
i
u t r
G u t r
r
t
t
Du




=



=
+
=





r
Au
b
=
−
d
r
Au
b
=
−

(
)
(
)
,
2
,
,
h
h i
h
h
i
i
v i
v
v
v
i
D u
t
t
D u
t
Du
t
t
t
D u
D u








=

=
=



















2 ball in 
 with center the origin and radius 
d
l



---

## 第79页

79
Split (linearly constrained) model:
ADMM  for discrepancy-constrained TV-L2 model


2
*
*
*
( )
2
,
,
1
, ,
arg min
( , , )
( )
s.t.
,
d
d
d
B
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b




=



=
+
=
=
−







---

## 第80页

80




2
*
*
*
,
,
2
2
2
2
, ,
arg min
( , , )
( )
( )
( )
0
0
s.t.
0
d
d
u r
t
d
d d
d
d
d
d
u t r
G u t r
f u
g t
h r
I
D u
t
r
I
A
b





=
+
+
−








+
+
=








−








It can be equivalently rewritten as:
all closed, proper, convex
with
( )
2
1
( )
0, ( )
, ( )
( ),
d
i
B
i
f u
g t
t
h r
r


=
=
=
=

Split (linearly constrained) model:
ADMM  for discrepancy-constrained TV-L2 model


2
*
*
*
( )
2
,
,
1
, ,
arg min
( , , )
( )
s.t.
,
d
d
d
B
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b




=



=
+
=
=
−







---

## 第81页

81
Is it a standard     
“three-blocks” problem?
 minimize   ( )
( )
( )
subject to   
f x
g y
h z
Bx
Cy
Ez
c
+
+
+
+
=




2
*
*
*
,
,
2
2
2
2
, ,
arg min
( , , )
( )
( )
( )
0
0
s.t.
0
d
d
u r
t
d
d d
d
d
d
d
u t r
G u t r
f u
g t
h r
I
D u
t
r
I
A
b





=
+
+
−








+
+
=








−








It can be equivalently rewritten as:
all closed, proper, convex
with
( )
2
1
( )
0, ( )
, ( )
( ),
d
i
B
i
f u
g t
t
h r
r


=
=
=
=

Split (linearly constrained) model:
ADMM  for discrepancy-constrained TV-L2 model


2
*
*
*
( )
2
,
,
1
, ,
arg min
( , , )
( )
s.t.
,
d
d
d
B
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b




=



=
+
=
=
−







---

## 第82页

82
2
2
2
2
                        
,
,
,
0
0
,
,
,
0
d
d d
d
d
d
d
x
u y
t z
r
I
D
B
C
E
c
I
A
b


=
=
=
−








=
=
=
=








−








YES!
Is it a standard     
“three-blocks” problem?
 minimize   ( )
( )
( )
subject to   
f x
g y
h z
Bx
Cy
Ez
c
+
+
+
+
=




2
*
*
*
,
,
2
2
2
2
, ,
arg min
( , , )
( )
( )
( )
0
0
s.t.
0
d
d
u r
t
d
d d
d
d
d
d
u t r
G u t r
f u
g t
h r
I
D u
t
r
I
A
b





=
+
+
−








+
+
=








−








It can be equivalently rewritten as:
all closed, proper, convex
with
( )
2
1
( )
0, ( )
, ( )
( ),
d
i
B
i
f u
g t
t
h r
r


=
=
=
=

Split (linearly constrained) model:
ADMM  for discrepancy-constrained TV-L2 model


2
*
*
*
( )
2
,
,
1
, ,
arg min
( , , )
( )
s.t.
,
d
d
d
B
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b




=



=
+
=
=
−







---

## 第83页

83




2
*
*
*
,
,
2
2
2
2
, ,
arg min
( , , )
( )
( )
( )
0
0
s.t.
0
d
d
u r
t
d
d d
d
d
d
d
u t r
G u t r
f u
g t
h r
I
D u
t
r
I
A
b





=
+
+
−








+
+
=








−








It can be equivalently rewritten as:
all closed, proper, convex
with
( )
2
1
( )
0, ( )
, ( )
( ),
d
i
B
i
f u
g t
t
h r
r


=
=
=
=

Split (linearly constrained) model:
ADMM  for discrepancy-constrained TV-L2 model


2
*
*
*
( )
2
,
,
1
, ,
arg min
( , , )
( )
s.t.
,
d
d
d
B
i
u r
t
i
u t r
G u t r
r
t
t
Du r
Au
b




=



=
+
=
=
−





           
,
( ; ),
( )
...,
...,
...,
...
x
u y
t r
g y
B
C
c
=
=
=
=
=
=
YES!
Can it also be seen as a 
standard “two-blocks” 
problem?
 minimize   ( )
( )
subject to   
f x
g y
Bx
Cy
c
+
+
=


---

## 第84页

The augmented Lagrangian function:
Restored image as the saddle point of the above function, that is:
2
( )
2
2
1
( , , ;
,
)
( )
,
2
d
t
t
r
B
i
t
i
L u t r
r
t
t
Du
t
Du





=
=
+
−
−
+
−

Split (linearly constrained) model:
2
2
,
(
)
(
)
2
r
r r
Au
b
r
Au
b


−
−
−
+
−
−
like for TIK-L1
like for TV-L2
ADMM  for discrepancy-constrained TV-L2 model


2
*
*
*
( )
2
,
,
1
, ,
arg min
( , , )
( )
s.t.
,
d
d
d
B
i
u r
t
i
u t r
G u t r
r
t
t
Du




=



=
+
=





r
Au
b
=
−
like for TV-L1
84


(
)
(
)
(
)
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
2
2
find
, ,
,
,
   s.t.    
, ,
,
,
, ,
,
,
, , ,
,
                                              ( , , ,
,
)
r
t
r
r
r
d
d
d
d
d
t
r
u t r
L u t r
L u t r
L u t r
u t r
















---

## 第85页

2
(
1)
( )
( )
( )
( )
(
1)
( )
(
1)
( )
( )
(
1)
(
1)
(
1)
( )
( )
(
1)
( )
(
1)
(
1)
(
1)
( )
(
1)
(
1)
arg min (
, ,
;
,
)
arg min (
,
, ;
,
)
arg min ( ,
,
;
,
)
(
)
(
(
d
d
d
k
k
k
k
k
t
r
t
k
k
k
k
k
t
r
r
k
k
k
k
k
t
r
u
k
k
k
k
t
t
t
k
k
k
k
r
r
r
t
L u
t r
r
L u
t
r
u
L u t
r
t
Du
r
Au












+

+
+

+
+
+

+
+
+
+
+
+
=
=
=
=
−
−
=
−
−
))
b
−
ADMM-based iterative algorithm (split optimization sub-problems):
Primal
descent
(alternating)
Dual
ascent
s.p.d. linear
system
closed-form
closed-form
proximal map
Saddle point of the Augmented Lagrangian function?
Euclidean
projection
ADMM  for discrepancy-constrained TV-L2 model
85


---

## 第86页

Subproblem for primal variable t :
Primal descent: subproblem for t
Augmented Lagrangian function:
2
(
1)
( )
( )
( )
( )
arg min
(
, ,
;
,
)
d
k
k
k
k
k
t
r
t
t
L u
t r


+

=
… exactly the same as for TV-L1,2 …
… closed-form solution (see Proposition 1) of d independent 2D problems:
(
)
2
2
(
1)
( )
( )
( )
( )
2
,
2
2
1
arg min
,   
,
2
i
k
k
k
k
k
t
i
i
i
i
i
t i
i
t
t
t
t
t
q
q
Du



+





=
+
−
=
+









2
( )
2
2
1
( , , ;
,
)
( )
,
2
d
t
t
r
B
i
t
i
L u t r
r
t
t
Du
t
Du





=
=
+
−
−
+
−

2
2
,
(
)
(
)
2
r
r r
Au
b
r
Au
b


−
−
−
+
−
−
86
( )
( )
2
( )
2
1
max
,0
,
1,
,
k
k
i
i
k
t
i
q
q
i
d
q



=
−
=




linear computational
complexity O(d)


---

## 第87页

Subproblem for primal variable r :
Primal descent: subproblem for r
Augmented Lagrangian function:
(
1)
( )
(
1)
( )
( )
arg min
(
,
, ;
,
)
d
k
k
k
k
k
t
r
r
r
L u
t
r 

+
+

=
… drop the terms independent of r …
(
)
(
)
2
( )
( )
( )
( )
2
arg min
( )
,
2
d
r
k
k
k
B
r
r
r
r
Au
b
r
Au
b







=
−
−
−
+
−
−




2
( )
( )
2
arg min
( )
,
2
d
r
k
B
r
r
r
w






=
+
−




( )
( )
( )
1
k
k
k
d
r
r
w
KAu
b


=
−
+

87
(
)
( )
( )
( )
2
( )
arg min
k
k
B
r
B
r
w
w



=
−
= 
Euclidean (orthogonal) projection of
vector w(k) onto the l2 ball of radius δ
2
( )
2
2
1
( , , ;
,
)
( )
,
2
d
t
t
r
B
i
t
i
L u t r
r
t
t
Du
t
Du





=
=
+
−
−
+
−

2
2
,
(
)
(
)
2
r
r r
Au
b
r
Au
b


−
−
−
+
−
−


---

## 第88页

Primal descent: subproblem for r
Euclidean (orthogonal) projection of vector w(k) onto the l2 ball of radius δ :
(
)


(
1)
( )
( )
( )
2
2
( )
arg min
,  with   ( )
:
k
k
k
d
B
r
B
r
r
w
w
B
x
x




+

=
−
= 
=


( )
( )
2
(
1)
( )
( )
( )
2
2
if   
if   
k
k
k
k
k
k
w
w
r
w
w
w



+



= 



The l2 ball is a convex (compact) set, hence the projection r (k+1) exists and is unique, 
and admits a simple closed-form expression :
B(δ)
(1)
(2)
w(k)
=r (k+1)
w(k)
r (k+1)
δ
88
(
1)
( )
( )
2
min
, 1
k
k
k
r
w
w

+




=






linear computational
complexity O(d)


---

## 第89页

Subproblem for primal variable u:
Primal descent: subproblem for u
(
1)
(
1)
(
1)
( )
( )
arg min
( ,
,
;
,
)
d
k
k
k
k
k
t
r
u
u
L u t
r


+
+
+

=
(
)
2
2
( )
(
1)
( )
(
1)
2
2
arg min
( )
,
,
2
2
d
k
k
k
k
t
r
t
r
u
Z u
Du
t
Du
Au
r
Au
b




+
+



=
=
+
−
+
+
−
−




The function Z is quadratic in the optimization variable u, hence its global
minimizers (if there exists one) are to be sought among its stationary points:
(
1)
  set of solutions of :     
( )
0
k
d
u
Z u
+ 

=
Augmented Lagrangian function:
… exactly the same as for TV-L1 …
89
2
( )
2
2
1
( , , ;
,
)
( )
,
2
d
t
t
r
B
i
t
i
L u t r
r
t
t
Du
t
Du





=
=
+
−
−
+
−

2
2
,
(
)
(
)
2
r
r r
Au
b
r
Au
b


−
−
−
+
−
−


---

## 第90页

Primal descent: subproblem for u
The linear system is exactly the same as the one in the subproblem for u
… after some simple algebraic manipulations:
(
1)
( )
(
1)
( )
1
1
( )
0   
  
T
T
T
k
k
T
k
k
r
r
d
t
r
t
t
t
r
Z u
D D
A A u
D
t
A
r
b








+
+







=

+
=
−
+
−
+












for model TV-L1 :  it admits a unique solution giving the new iterate u(k+1)
90
Assuming periodic/reflective/anti-reflective boundary conditions for u,
the linear system can be solved (like for TIK-L2 case) by 2D DFT/DCT/DST.
By using 2D FFT/FCT/FST implementations   
computational
complexity O(d log d)
RESTORATION:


---

## 第91页

Primal descent: subproblem for u
The linear system is exactly the same as the one in the subproblem for u
… after some simple algebraic manipulations:
(
1)
( )
(
1)
( )
1
1
( )
0   
  
T
T
T
k
k
T
k
k
r
r
d
t
r
t
t
t
r
Z u
D D
A A u
D
t
A
r
b








+
+







=

+
=
−
+
−
+












for model TV-L1 :  it admits a unique solution giving the new iterate u(k+1)
91
the linear system can be solved (like for TIK-L2 case) by iterative (P)CG
INPAINTING:


---

