# Benning - Lecture 2.pdf

## 第1页

Faculty of Engineering, Department of Computer Science
Erasmus+ International PhD Summer School 2025 
Mathematics and Machine Learning for Image Analysis 
University of Bologna 
10 June 2025


---

## 第2页

This is joint work with
Danilo Riccio 
Queen Mary University of London
Acknowledgements:
Tatiana Bubba 
Università degli Studi di Ferrara
Luca Ratti 
Università di Bologna


---

## 第3页

Learning optimal sampling strategies for MRI
• Error estimates in inverse problems 
• Computing source condition elements 
• Learning variational regularisations with optimal error 
estimates 
• Learning the sampling pattern in MRI 
• Conclusions & outlook


---

## 第4页

Motivation
Magnetic resonance imaging
© Wikimedia commons
Simplified image reconstruction process
ℱ−1
(ℱu)pq =
1
nxny
ny−1
∑
l=0
nx−1
∑
j=0
ulj e−i 2πpl
ny e−i 2πqj
nx


---

## 第5页

Motivation
Magnetic resonance imaging
© Wikimedia commons
(ℱu)pq =
1
nxny
ny−1
∑
l=0
nx−1
∑
j=0
ulj e−i 2πpl
ny e−i 2πqj
nx
Simplified image reconstruction process
Goal: speed up acquisition process


---

## 第6页

Motivation
Magnetic resonance imaging
© Wikimedia commons
(ℱu)pq =
1
nxny
ny−1
∑
l=0
nx−1
∑
j=0
ulj e−i 2πpl
ny e−i 2πqj
nx
Simplified image formation process
Goal: speed up acquisition process
Sℱ
Sampling operator


---

## 第7页

Motivation
from*
*Sherry, F., MB, De los Reyes, J. C., Graves, M. J., Maierhofer, G., Williams, G., Schönlieb, C.-B. & Ehrhardt, M. J. 
(2020). Learning the sampling pattern for MRI. IEEE Transactions on Medical Imaging, 39(12), 4310-4321.


---

## 第8页

Motivation
from*
min
S,α ∥RS,α(f δ) −u†∥2 + ℛ(S)
subject to
RS,α(f δ) ∈arg min
u {
1
2∥Sℱu −f δ∥2 + αJ(u)}
Our approach 
back then:
Complex bilevel 
optimisation approach
Today’s talk:  
simple convex  
minimisation problem(s)
*Sherry, F., MB, De los Reyes, J. C., Graves, M. J., Maierhofer, G., Williams, G., Schönlieb, C.-B. & Ehrhardt, M. J. 
(2020). Learning the sampling pattern for MRI. IEEE Transactions on Medical Imaging, 39(12), 4310-4321.


---

## 第9页

Motivation
Many other relevant works:
Gözcü et al. 2018, 2019 , Sanchez et al. 2019, Weiss et al. 2019, Bahadir et 
al. 2019, Jin, Unser, Yi 2019, Gossard, Gournay, Weiss 2022, Bakker et al. 
2022, etc.
from*
*Sherry, F., MB, De los Reyes, J. C., Graves, M. J., Maierhofer, G., Williams, G., Schönlieb, C.-B. & Ehrhardt, M. J. 
(2020). Learning the sampling pattern for MRI. IEEE Transactions on Medical Imaging, 39(12), 4310-4321.


---

## 第10页

Error estimates in inverse problems


---

## 第11页

Error estimates in inverse problems
If we approximate the solution of an inverse problem 
 with 
regularisations 
, we ideally want convergent regularisations
Ku† = f
Rα
We would like to establish
D (u†, Rα(f δ)) ≤C δ
such that
lim
δ→0 sup {D (u†, Rα(f δ))
 for f = Ku†  and  f δ  with  ∥f −f δ∥≤δ} = 0
for some distance measure 
 (not necessarily a metric)
D
Engl, H. W., Hanke, M., & Neubauer, A. (1996). Regularization of inverse problems (Vol. 375). Springer Science & Business Media. 
Scherzer, O., Grasmair, M., Grossauer, H., Haltmeier, M., & Lenzen, F. (2009). Variational methods in imaging. 
MB, & Burger, M. (2018). Modern regularization methods for inverse problems. Acta Numerica, 27, 1-111.


---

## 第12页

Error estimates in inverse problems
If we approximate the solution of an inverse problem 
 with 
regularisations 
, we ideally want convergent regularisations
Ku† = f
Rα
We would like to establish
D (u†, Rα(f δ)) ≤C δ
such that
lim
δ→0 sup {D (u†, Rα(f δ))
 for f = Ku†  and  f δ  with  ∥f −f δ∥≤δ} = 0
for some distance measure 
 (not necessarily a metric)
D
Engl, H. W., Hanke, M., & Neubauer, A. (1996). Regularization of inverse problems (Vol. 375). Springer Science & Business Media. 
Scherzer, O., Grasmair, M., Grossauer, H., Haltmeier, M., & Lenzen, F. (2009). Variational methods in imaging. 
MB, & Burger, M. (2018). Modern regularization methods for inverse problems. Acta Numerica, 27, 1-111.
Making this small subject to 
constraints is a surrogate to the 
previous upper level problem 


---

## 第13页

We would like to establish
D (u†, Rα(f δ)) ≤C δ
Example: variational regularisation
Rα: f δ ↦uα ∈arg min
u∈X {
1
2∥Ku −f δ∥2
Y + αJ(u)}
Assume source (or range) condition:
∃v ∈Y:
∈∂J(u†)
K*v
= {p | J(v) −J(u†) −⟨p, v −u†⟩≥0, ∀v}
Error estimates in inverse problems


---

## 第14页

We would like to establish
D (u†, Rα(f δ)) ≤C δ
Example: variational regularisation
Rα: f δ ↦uα ∈arg min
u∈X {
1
2∥Ku −f δ∥2
Y + αJ(u)}
Assume source (or range) condition:
∃v ∈Y:
K*v
Error estimates in inverse problems
= ∇J(u†)


---

## 第15页

We would like to establish
D (u†, Rα(f δ)) ≤C δ
Example: variational regularisation
Rα: f δ ↦uα ∈arg min
u∈X {
1
2∥Ku −f δ∥2
Y + αJ(u)}
Assume source (or range) condition:
Error estimates in inverse problems
∃v ∈Y:
∈∂J(u†)
K*v


---

## 第16页

We would like to establish
D (u†, Rα(f δ)) ≤C δ
Example: variational regularisation
Assume source (or range) condition:
∃v ∈Y:
(equivalent to 
 with 
 )
u† ∈arg min
u∈X {
1
2 ∥Ku −gα∥2
Y + αJ(u)}
gα = αv + Ku† = αv + f
∈∂J(u†)
K*v
Error estimates in inverse problems
Rα: f δ ↦uα ∈arg min
u∈X {
1
2∥Ku −f δ∥2
Y + αJ(u)}


---

## 第17页

We would like to establish
D (u†, Rα(f δ)) ≤C δ
Example: variational regularisation
Assume source (or range) condition:
∃v ∈Y:
∈∂J(u†)
K*v
Error estimates in inverse problems
Rα: f δ ↦uα ∈arg min
u∈X {
1
2∥Ku −f δ∥2
Y + αJ(u)}
Then one can prove
D (Rα(δ)(f δ), u†) ≤∥v∥Y δ
for α(δ) = δ/∥v∥Y
where 
 is a suitable (symmetrised) Bregman distance/divergence w.r.t 
D
J


---

## 第18页

Motivation
This presentation in a nutshell
Example:
we want to study this quantity 
u†
v
J(u) = TV(u)
*To be defined 
later, if you 
haven’t seen this 
before
*
K = I
D (Rα(δ)(f δ), u†) ≤∥v∥Y δ


---

## 第19页

Motivation
This presentation in a nutshell
Example:
we want to study this quantity 
u†
v
= α
+
gα
D (Rα(δ)(f δ), u†) ≤∥v∥Y δ


---

## 第20页

Rα
Motivation
This presentation in a nutshell
Example:
u†
gα
=
Ideally, we want
v =
to have small norm ∥v∥Y
D (Rα(δ)(f δ), u†) ≤∥v∥Y δ
How do we compute  in order to estimate (or even control) 
 ?
v
∥v∥Y


---

## 第21页

Computing source condition elements


---

## 第22页

Computing source condition elements
Approach 1: we can modify the source condition 
∈∂
K*v
J (u†)


---

## 第23页

u†+
∈∂
K*v
J (u†)
(
1
2 ∥⋅∥2 + )
to
Now we use the Fenchel-Young equality:
p ∈∂F(u)
⟺
F(u) + F⋆(p) = ⟨u, p⟩
where 
 denotes the convex conjugate of , i.e.
F⋆
F
F⋆(p) := sup
u
⟨u, p⟩−F(u)
Werner Fenchel, Convex cones, sets, and functions. Princeton University, 1953 
Theorem 23.5,  Ralph Tyrell Rockafellar, Convex analysis, Princeton university press, 1970
Computing source condition elements
Approach 1: we can modify the source condition 


---

## 第24页

Proof:
p ∈∂F(u) ⟺
p ∈{q|F(v) ≥F(u) + ⟨q, v −u⟩, ∀v}
⟺
⟨p, u⟩−F(u) ≥⟨p, v⟩−F(v)
∀v
⟺
⟨p, u⟩−F(u) = F⋆(p)
⟺
F(u) + F⋆(p) = ⟨u, p⟩
p ∈∂F(u)
⟺
F(u) + F⋆(p) = ⟨u, p⟩
where 
 denotes the convex conjugate of , i.e.
F⋆
F
F⋆(p) := sup
u
⟨u, p⟩−F(u)
Now we use the Fenchel-Young equality:
Werner Fenchel, Convex cones, sets, and functions. Princeton University, 1953 
Theorem 23.5,  Ralph Tyrell Rockafellar, Convex analysis, Princeton university press, 1970
Computing source condition elements


---

## 第25页

∈∂
K*v
J (u†)
(
1
2 ∥⋅∥2 + )
to
This implies
(
1
2∥⋅∥2 + J)(u†) + (
1
2 ∥⋅∥2 + J)
⋆
(u† + K*v) = ⟨u†, u† + K*v⟩
u†+
p ∈∂F(u)
⟺
F(u) + F⋆(p) = ⟨u, p⟩
Now we use the Fenchel-Young equality:
Computing source condition elements
Werner Fenchel, Convex cones, sets, and functions. Princeton University, 1953 
Theorem 23.5,  Ralph Tyrell Rockafellar, Convex analysis, Princeton university press, 1970
Approach 1: we can modify the source condition 


---

## 第26页

Instead of enforcing a strict equality, we can define
(
1
2∥⋅∥2 + J)(u†) + (
1
2 ∥⋅∥2 + J)
⋆
(u† + K*v)
⟨u†, u† + K*v⟩
GJ (v) :=
−
Computing source condition elements
MB, Tatiana A. Bubba, Luca Ratti, and Danilo Riccio. "Trust your source: quantifying source condition 
elements for variational regularisation methods." IMA Journal of Applied Mathematics (2024): hxae008.


---

## 第27页

Instead of enforcing a strict equality, we can define
(
1
2∥⋅∥2 + J)(u†) + (
1
2 ∥⋅∥2 + J)
⋆
(u† + K*v)
⟨u†, u† + K*v⟩
GJ (v) :=
−
Proposition: the gradient 
 of 
 exists and reads 
∇GJ
GJ
.
∇GJ(v) = K proxJ(u† + K*v) −Ku†
Computing source condition elements
Here 
 denotes the proximal map, i.e.
prox
proxF: Z →Z,
proxF(z) := arg min
u∈Z {
1
2∥u −z∥2
Z + F(u)}
MB, Tatiana A. Bubba, Luca Ratti, and Danilo Riccio. "Trust your source: quantifying source condition 
elements for variational regularisation methods." IMA Journal of Applied Mathematics (2024): hxae008.


---

## 第28页

(
1
2∥⋅∥2 + J)(u†) + (
1
2 ∥⋅∥2 + J)
⋆
(u† + K*v)
⟨u†, u† + K*v⟩
GJ (v) :=
−
Instead of enforcing a strict equality, we can define
Because of the Fréchet-differentiability of 
, we can solve
GJ
̂v = arg min
v GJ(v)
for instance via gradient descent, i.e
vk+1 = vk −τK (proxJ (u† + K*vk) −u†)
which is globally convergent for 
 
τ ≤1/∥K∥2
Computing source condition elements
MB, Tatiana A. Bubba, Luca Ratti, and Danilo Riccio. "Trust your source: quantifying source condition 
elements for variational regularisation methods." IMA Journal of Applied Mathematics (2024): hxae008.


---

## 第29页

Extension to more general functions and range conditions
We can also consider composite functionals
J(u) = H(Au + b)
Example:
TV(u) =
ny−1
∑
i=1
nx−1
∑
j=1
u(i+1)j −uij
2
+ ui(j+1) −uij
2
 with
J = TV: ℝny×nx →ℝ
We obtain this for 
,  
 with
b = 0 A: ℝny×nx →ℝ(ny−1)×(nx−1)×2
(Au)ijp = {
u(i+1)j −uij
p = 1
ui(j+1) −uij
p = 2 ,
and 
 with
H: ℝ(ny−1)×(nx−1)×2 →ℝ
H(q) =
ny−1
∑
i=1
nx−1
∑
j=1
qij1
2
+ qij2
2
.
MB, Tatiana A. Bubba, Luca Ratti, and Danilo Riccio. "Trust your source: quantifying source condition 
elements for variational regularisation methods." IMA Journal of Applied Mathematics (2024): hxae008.


---

## 第30页

J(u) = H(Au + b)
u† ∈arg min
u∈X {
1
2 ∥Ku −gα∥2 + αJ(u)} ,
Consider the range condition
respectively it’s optimality condition
K*(Ku† −gα) + αA*q† = 0
for 
.
q† ∈∂H(Au† + b)
Extension to more general functions and range conditions
MB, Tatiana A. Bubba, Luca Ratti, and Danilo Riccio. "Trust your source: quantifying source condition 
elements for variational regularisation methods." IMA Journal of Applied Mathematics (2024): hxae008.
We can also consider composite functionals


---

## 第31页

Consider the range condition
K*(Ku† −gα) + αA*q† = 0
for 
.
q† ∈∂H(Au† + b)
Multiplying by 
 and using 
 then yields
1/α
gα = αv + Ku†
K*v = A*q†
q† ∈∂H(Au† + b)
Extension to more general functions and range conditions
MB, Tatiana A. Bubba, Luca Ratti, and Danilo Riccio. "Trust your source: quantifying source condition 
elements for variational regularisation methods." IMA Journal of Applied Mathematics (2024): hxae008.


---

## 第32页

Multiplying by 
 and using 
 then yields
1/α
gα = αv + Ku†
K*v = A*q†
q† ∈∂H(Au† + b)
Similar to before, we relax these problems by introducing 
 with
GH: Y →ℝ
EH(v, q†) = 1
2 ∥K*v −A*q†∥2 + GH(q†)
GH(q) = (
1
2∥⋅∥2 + H)(Au† + b) + (
1
2 ∥⋅∥2 + H)
⋆
(q + Au† + b) −⟨Au† + b, Au† + b + q⟩
and minimise
Extension to more general functions and range conditions
MB, Tatiana A. Bubba, Luca Ratti, and Danilo Riccio. "Trust your source: quantifying source condition 
elements for variational regularisation methods." IMA Journal of Applied Mathematics (2024): hxae008.


---

## 第33页

Computing source condition elements
Approach 2: we can formulate the -minimising solution of 
, i.e.
J
Ku = f
min
u J(u)
subject to
Ku
as a primal-dual problem
min
u sup
v
J(u) + ⟨v, f −Ku⟩
with saddle point 
 and optimality conditions
(u†, v†)
K*v† ∈∂J(u†)
Ku†
source condition
= f
= f


---

## 第34页

H(
)
Computing source condition elements
Approach 2: we can formulate the -minimising solution of 
, i.e.
J
Ku = f
min
u J(u)
as a primal-dual problem
min
u sup
v
J(u) + ⟨v, f −Ku⟩
with saddle point 
 and optimality conditions
(u†, v†)
K*v† ∈∂J(u†)
source condition
−H⋆(v)
Ku†
f −
∈∂H⋆(v†)
Ku
f−
+


---

## 第35页

α∥
∥Y
Computing source condition elements
Approach 2: we can formulate the -minimising solution of 
, i.e.
J
Ku = f
min
u J(u)
as a primal-dual problem
min
u sup
v
J(u) + ⟨v, f −Ku⟩
with saddle point 
 and optimality conditions
(u†, v†)
K*v† ∈∂J(u†)
source condition
−χ∥⋅∥Y≤α(v)
Ku†
f −
∈∂χ∥⋅∥Y≤α(v†)
Ku
f−
+


---

## 第36页

α∥
∥Y
Computing source condition elements
Approach 2: we can formulate the -minimising solution of 
, i.e.
J
Ku = f
min
u J(u)
Ku
f−
+
Example: 
, 
K = I J(u) = TV(u)
f
u∞
 with 
v∞
∥v∞∥≈801.88


---

## 第37页

α∥
∥Y
Computing source condition elements
Approach 2: we can formulate the -minimising solution of 
, i.e.
J
Ku = f
min
u J(u)
Ku
f−
+
Example: 
, 
K = I J(u) = TV(u)
f
uα
 with 
vα
∥vα∥= α = 100


---

## 第38页

H(
)
Computing source condition elements
Approach 2: we can formulate the -minimising solution of 
, i.e.
J
Ku = f
min
u J(u)
as a primal-dual problem
min
u sup
v
J(u) + ⟨v, f −Ku⟩−H⋆(v)
Ku
f−
+
Can (for example) be solved with primal-dual hybrid gradient* method, i.e.
uk+1 = proxτJ (uk + τK*vk)
vk+1 = proxσH⋆(vk −σ (K(2uk+1 −uk) −f))
*(cf. Chambolle, A., & Pock, T. (2016). An introduction to continuous optimization for imaging. Acta Numerica, 25, 161-319.
τσ < 1/∥K∥2


---

## 第39页

Numerical results


---

## 第40页

Numerical results
Inverse problem
Sℱu† = f
Subsampling operator
(ℱu)pq =
1
nxny
ny−1
∑
l=0
nx−1
∑
j=0
ulj e−i 2πpl
ny e−i 2πqj
nx
with discrete 2D Fourier transform 
Regularisation function:
(Au)ijp = {
u(i+1)j −uij
p = 1
ui(j+1) −uij
p = 2 , H(q) =
ny−1
∑
i=1
nx−1
∑
j=1
qij1
2
+ qij2
2
.
J(u) = H(Au)
with
(2D discretised isotropic total variation)


---

## 第41页

Inverse problem
Sℱu† = f
Subsampling operator
vk+1 = vk −τSℱ(ℱ−1S⊤vk −A⊤qk) ,
Source condition
Algorithm
qk+1 = qk −σ (A (A⊤qk −ℱ−1S⊤vk+1) + prox∥⋅∥2,1 (Au† + qk) −Au†) .
ℱ−1S⊤v ∈∂TV(u†)
Numerical results


---

## 第42页

Numerical results
S = I
(a) ℱ−1vK
ℱ−1vK = A⊤qK


---

## 第43页

S = I
(a) ℱ−1vK
∥vK∥≈101.78
Numerical results


---

## 第44页

Numerical results


---

## 第45页

Numerical results
ℱ−1STvK = A⊤qK
∥vK∥≈72.79


---

## 第46页

*Sanity check computed with 
PDHG method (cf. Chambolle, A., 
& Pock, T. (2016). An 
introduction to continuous 
optimization for imaging. Acta 
Numerica, 25, 161-319.)
*


---

## 第47页

Numerical results


---

## 第48页

Numerical results


---

## 第49页

Numerical results
ℱ−1STvK = A⊤qK
∥vK∥≈255.15


---

## 第50页

*Sanity check computed with 
PDHG method (cf. Chambolle, A., 
& Pock, T. (2016). An 
introduction to continuous 
optimization for imaging. Acta 
Numerica, 25, 161-319.)
*


---

## 第51页

Learning variational regularisations 
with optimal error estimates


---

## 第52页

We could approximate a bilevel problem with trainable regularisation function
min
Θ
1
2s
s
∑
i=1
∥u(Θ) −u†∥2
X + βR(Θ)
subject to
u(Θ) ∈arg min
u {
1
2 ∥Ku −f δ∥2
Y + J(u, Θ)}
and check if learned regularisation function 
 improves norm of source 
condition elements
J( ⋅, Θ)
Approximate lower level problem with finite number of PDHG iterations
R(Θ) = 0
J(u, A) =
n
∑
j=1
m
∑
l=1
(Au)jl
K = I
Choose
(Lower level problem)


---

## 第53页

Approximate lower level problem with finite number of PDHG iterations
R(Θ) = 0
J(u, A) =
n
∑
j=1
m
∑
l=1
(Au)jl
K = I
Choose
Au =
h1
⋮
hm
* u
Example: convolution
Samples 
u†
i =
= f δ
i


---

## 第54页

Approximate lower level problem with finite number of PDHG iterations
R(Θ) = 0
J(u, A) =
n
∑
j=1
m
∑
l=1
(Au)jl
K = I
Choose
Au =
h1
⋮
hm
* u
Example: convolution
Optimal kernel (
): 
m = 8


---

## 第55页

Approximate lower level problem with finite number of PDHG iterations
R(Θ) = 0
J(u, A) =
n
∑
j=1
m
∑
l=1
(Au)jl
K = I
Choose
Au =
h1
⋮
hm
* u
Example: convolution
Source condition 
computation
∥v∥≈8.59


---

## 第56页

Approximate lower level problem with finite number of PDHG iterations
R(Θ) = 0
TV(u) =
n
∑
j=1
m
∑
l=1
(∇u)jl
K = I
Choose
Source condition 
computation
∥v∥≈17.376
Comparison with total variation


---

## 第57页

Learning the sampling pattern in MRI


---

## 第58页

Numerical results
Back to the source condition
ℱ−1S⊤v ∈∂TV(u†)
Idea: define
˜v: = S⊤v ∈ℂny×nx
and estimate sparse  instead of  by solving
˜v
v
min
˜v,q†
1
2 ∥ℱ−1˜v −A⊤q†∥2 + G∥⋅∥2,1(q†) + β∥˜v∥1
e.g. via PALM*, i.e.
˜vk+1 = proxβ∥⋅∥1 (˜vk −τ (˜vk −ℱA⊤qk)) ,
qk+1 = qk −σ (A (A⊤qk −ℱ−1˜vk+1) + prox∥⋅∥2,1 (qk + Au†) −Au†) .
*Bolte, J., Sabach, S., & Teboulle, M. (2014). Proximal alternating linearized minimization for 
nonconvex and nonsmooth problems. Mathematical Programming, 146(1-2), 459-494.


---

## 第59页

Numerical results


---

## 第60页

β = 0.1
*
*F. Sherry et al. (2020). Learning the sampling pattern for MRI. IEEE 
Transactions on Medical Imaging, 39(12), 4310-4321.


---

## 第61页

Sherry et al.


---

## 第62页

Numerical results


---

## 第63页

Numerical results
β = 0.24


---

## 第64页

Numerical results


---

## 第65页

Conclusions & Outlook


---

## 第66页

Conclusions & outlook
Conclusions:
Outlook:
•
reformulated source and range conditions as the solution of convex 
minimisation problems 
•
provided iterative algorithms for their numerical approximation 
•
made an attempt at supervised learning of regularisation functions with 
optimal error constants
we have
•
Higher-order or variational source conditions, gen. eigenfunctions 
•
Optimal sampling in various domains (e.g. MRI, single-pixel camera) 
•
Supervised learning for more general operator correction 
•
Use framework in other contexts (e.g., training deep neural networks)


---

## 第67页

Thank you for your attention!
Acknowledgements:
IMA Journal of 
Applied Mathematics
hxae008


---

## 第68页

Example: variational regularisation
Rα: Y ⇉X,
Rα: f ↦uα ∈arg min
u∈X {
1
2∥Ku −f∥2
Y + αJ(u)}
We can then show the following (well-known) estimate:
Optimality condition 
 if and only if
uα ∈Rα(fδ)
∃pα ∈∂J(uα) :
= 0
Assume source condition:
K*(Kuα −f δ) + αpα
∃v ∈Y:
∈∂J(u†)
K*v
Burger, M., Resmerita, E., & He, L. (2007). Error estimation for Bregman iterations and inverse scale space methods in image restoration. 
Computing, 81(2-3), 109-135. 
Scherzer, O., Grasmair, M., Grossauer, H., Haltmeier, M., & Lenzen, F. (2009). Variational methods in imaging. 
MB, & Burger, M. (2018). Modern regularization methods for inverse problems. Acta Numerica, 27, 1-111.
Error estimates in inverse problems


---

## 第69页

=
K*(Kuα −f δ) + αpα
K*v
−α
K*v
−α
Dual product with 
:
uα −u†
⟨Kuα −f δ, Kuα −f⟩Y
=Dpα
J (u†,uα) + DK*v
J
(uα,u†)
where
Dp
J (v, u) = J(v) −J(u) −⟨p, v −u⟩
p ∈∂J(u)
for
is the Bregman distance (divergence) w.r.t.  for arguments 
J
v, u
Dsymm
J
(u, v) = Dp
J (v, u) + Dq
J (u, v)
is a symmetric Bregman distance 
+α⟨uα −u†, pα −K*v⟩* = −α⟨K*v, uα −u†⟩*
Bregman, L. M. (1967). The relaxation method of finding the common point of convex sets and its application to the solution of problems in 
convex programming. USSR computational mathematics and mathematical physics, 7(3), 200-217. 
Kiwiel, K. C. (1997). Proximal minimization methods with generalized Bregman functions. SIAM journal on control and optimization, 35(4), 
1142-1168.
Error estimates in inverse problems


---

## 第70页

⟨Kuα −f δ, Kuα −f⟩Y = 1
2 ∥Kuα −f∥2
Y + 1
2∥Kuα −f δ∥2
Y −1
2 ∥f −f δ∥2
Y
Hence, 
1
2 ∥Kuα −f∥2
Y + 1
2 ∥Kuα −f δ∥2
Y + αDsymm
J
(uα, u†) = 1
2∥f −f δ∥2
Y −α⟨v, Kuα −f⟩Y
Using the identity
⟨αv, f −Kuα⟩Y = α2
2 ∥v∥2
Y + 1
2 ∥Kuα −f∥2
Y −1
2∥αv −f + Kuα∥2
Y
then yields
1
2∥Kuα −f + αv∥2
Y + 1
2∥Kuα −f δ∥2
Y + αDsymm
J
(uα, u†) = 1
2∥f −f δ∥2
Y + α2
2 ∥v∥2
Y
Error estimates in inverse problems


---

## 第71页

1
2∥Kuα −f + αv∥2
Y + 1
2 ∥Kuα −f δ∥2
Y + αDsymm
J
(uα, u†) = 1
2∥f −f δ∥2
Y + α2
2 ∥v∥2
Y
Dividing by  and using 
 yields the estimate
α
∥f −fδ∥≤δ
1
2α∥Kuα −f + αv∥2
Y + 1
2α∥Kuα −f δ∥2
Y + Dsymm
J
(uα, u†) ≤α
2 ∥v∥2
Y + δ2
2α
Hence, if we choose 
, we obtain
α(δ) = δ/∥v∥Y
Dsymm
J
(uα(δ), u†) ≤∥v∥Y δ
How do we compute  in order to estimate (or even control) 
 ?
v
∥v∥Y
Error estimates in inverse problems


---

