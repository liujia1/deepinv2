# lecture_2.pdf

## 第1页

ERASMUS+ INTERNATIONAL PHD SUMMER SCHOOL 2025
Mathematics and Machine Learning for image analysis
Lecture 2 - Optimization vs. Sampling
Thomas Pock
Institute of Visual Computing
Graz University of Technology
University of Bologna, June 3-6 2025
1 / 33


---

## 第2页

Bayesian inverse problems
▶Bayesian inference
pX|Y (x|y) = pY |X(y|x)pX(x)
pY (y)
▶Can be seen as logic with uncertainty.
▶The likelihood pY |X is often known due to the
image formation process
pY |X(y|x) ∝exp
 
−∥A(x) −y∥2
2σ2
!
▶The prior pX is usually unknown and should be
learned from data.
2 / 33


---

## 第3页

Bayesian inverse problems
▶Bayesian inference
pX|Y (x|y) = pY |X(y|x)pX(x)
pY (y)
▶Can be seen as logic with uncertainty.
▶The likelihood pY |X is often known due to the
image formation process
pY |X(y|x) ∝exp
 
−∥A(x) −y∥2
2σ2
!
▶The prior pX is usually unknown and should be
learned from data.
2 / 33


---

## 第4页

Bayesian inverse problems
▶Bayesian inference
pX|Y (x|y) = pY |X(y|x)pX(x)
pY (y)
▶Can be seen as logic with uncertainty.
▶The likelihood pY |X is often known due to the
image formation process
pY |X(y|x) ∝exp
 
−∥A(x) −y∥2
2σ2
!
▶The prior pX is usually unknown and should be
learned from data.
2 / 33


---

## 第5页

Bayesian inverse problems
▶Bayesian inference
pX|Y (x|y) = pY |X(y|x)pX(x)
pY (y)
▶Can be seen as logic with uncertainty.
▶The likelihood pY |X is often known due to the
image formation process
pY |X(y|x) ∝exp
 
−∥A(x) −y∥2
2σ2
!
▶The prior pX is usually unknown and should be
learned from data.
2 / 33


---

## 第6页

Bayesian inverse problems
▶Bayesian inference
pX|Y (x|y) = pY |X(y|x)pX(x)
pY (y)
▶Can be seen as logic with uncertainty.
▶The likelihood pY |X is often known due to the
image formation process
pY |X(y|x) ∝exp
 
−∥A(x) −y∥2
2σ2
!
▶The prior pX is usually unknown and should be
learned from data.
2 / 33


---

## 第7页

Bayesian point estimators
▶There are a number of Bayesian estimators that are used to compute point estimates from the
posterior
ˆx(y) ∈argmin
z
Z
X
ℓ(x, z)pX|Y =y(x) dx,
depending of the choice of the loss function ℓ(x, z).
3 / 33


---

## 第8页

Bayesian point estimators
▶There are a number of Bayesian estimators that are used to compute point estimates from the
posterior
ˆx(y) ∈argmin
z
Z
X
ℓ(x, z)pX|Y =y(x) dx,
depending of the choice of the loss function ℓ(x, z).
▶The 0-1 loss leads to the maximum a-posteriori (MAP) estimate
ˆx(y) = argmax
x
pX|Y =y(x) = argmin
x
−log pX|Y =y(x)
3 / 33


---

## 第9页

Bayesian point estimators
▶There are a number of Bayesian estimators that are used to compute point estimates from the
posterior
ˆx(y) ∈argmin
z
Z
X
ℓ(x, z)pX|Y =y(x) dx,
depending of the choice of the loss function ℓ(x, z).
▶The 0-1 loss leads to the maximum a-posteriori (MAP) estimate
ˆx(y) = argmax
x
pX|Y =y(x) = argmin
x
−log pX|Y =y(x)
▶The squared loss leads to the posterior expectation, or minimum mean squared estimate (MMSE)
¯x(y) =
Z
X
x pX|Y =y(x) dx = EX|Y =y [x]
3 / 33


---

## 第10页

Example: Total variation regularized Gaussian image denoising
▶Let us assume we have a prior and likelihood that take the form of a Gibbs distribution
[Boltzmann 1868], [Gibbs 1889]
pX(x) ∝exp(−R(x)),
pY |X(y|x) ∝exp (−D(x, y)) ,
with R the regularizer and D the data fidelity term, which are here given by
R(x) = λ
X
i,j
|xi+1,j −xi,j| + |xi,j+1 −xi,j|,
D(x, y) = ∥x −y∥2
2σ2
▶In summary, the negative log posterior is given by
−log pX|Y (x|y) ∝E(x) := R(x) + D(x, y)
4 / 33


---

## 第11页

Regularizer vs. prior
The behavior of the total variation as a regularizer and a prior can be very different.
▶It is well-known that using R(x) as a
regularization term (here in 1D) leads to
piecewise constant signals [Chambolle, Lions,
’97].
0.0
0.2
0.4
0.6
0.8
1.0
0.0
0.2
0.4
0.6
0.8
1.0
1.2
noisy signal
TV solution
▶This is in contrast to actual samples
x ∼pX(x) which can be obtained for 1D TV
in linear time via a Levy process [Bohra, et al.,
’23]
0
200
400
600
800
1000
0.0
0.2
0.4
0.6
0.8
1.0
x
exp(
TV(x))
5 / 33


---

## 第12页

MAP estimation
▶MAP estimation (minimization)
ˆx = argmin
x∈X
R(x) +
1
2σ2 ∥x −y∥2
2 .
▶The optimal solution is given by the proximal map [Moreau ’62]
ˆx = (I + σ2∂R)−1(y) := proxσ2R(y),
which can be written as one gradient step on the
infimal-convolution of R with a quadratic function
∥·∥2
2
2σ2 (called Moreau envelope)
ˆx = y −σ2∇ˆRσ2(y),
ˆRσ2 = (R□
∥·∥2
2
2σ2 ),
also with a nice representation of its gradient
∇ˆRσ2(y) = y −proxσ2R(y)
σ2
.
6 / 33


---

## 第13页

MMSE estimation
▶MMSE estimation (expectation)
¯x = 1
Z
Z
X
x exp

−R(x) −
1
2σ2 ∥x −y∥2
dx,
¯x = E[x|y],
which can be written in the form of one gradient step on the soft-infimal-convolution of
exp(−R) with a Gaussian exp (−
∥·∥2
2
2σ2 )
¯x = y −σ2∇¯Rσ2(y),
¯Rσ2 = −log

exp(−R) ∗exp (−
∥·∥2
2
2σ2 )

.
▶This formula is known as Tweedie’s formula [Robbins ’56], [Miyasawa ’60].
▶The negative gradients −∇¯Rσ2 are also known as score functions in denoising diffusion models at
diffusion time t = σ2/2.
7 / 33


---

## 第14页

Diffusion at absolute zero
▶Let’s introduce a temperature parameter T > 0
¯RT
σ2(y) = −T log


Z
exp

−R(x) −
∥x−y∥2
2
2σ2
T

dx


▶One can show that as T →0+, the soft-infimal convolution ¯RT
σ2(y) converges to the infimal
convolution ˆRσ2(y).
▶As a consequence, the “Tweedie-prox” becomes a “Moreau-prox”.
▶This gives rise to a “Diffusion at absolute zero” [Habring, Falk, Zach, P. ’25], as an alternative to
the standard denoising diffusion models, where the score can be computed based on the proximal
map.
8 / 33


---

## 第15页

Algorithms
▶In practice, we usually do not have direct access to the gradients of the (soft-)
infimal-convolutions to compute the MAP or MMSE in one step.
▶The previous formulas can only be applied to image denoising.
▶One needs some more general (iterative algorithms)
9 / 33


---

## 第16页

Algorithms
▶In practice, we usually do not have direct access to the gradients of the (soft-)
infimal-convolutions to compute the MAP or MMSE in one step.
▶The previous formulas can only be applied to image denoising.
▶One needs some more general (iterative algorithms)
▶The most basic optimization algorithm to compute the MAP estimate is gradient descent (GD).
xk+1 = xk −τ∇E(xk)
where τ > 0 is the step size.
9 / 33


---

## 第17页

Algorithms
▶In practice, we usually do not have direct access to the gradients of the (soft-)
infimal-convolutions to compute the MAP or MMSE in one step.
▶The previous formulas can only be applied to image denoising.
▶One needs some more general (iterative algorithms)
▶The most basic optimization algorithm to compute the MAP estimate is gradient descent (GD).
xk+1 = xk −τ∇E(xk)
where τ > 0 is the step size.
▶The sampling analogue is the unadjusted Langevin algorithm (ULA)
X k+1 = X k −τ∇E(X k) +
√
2τNk
where τ is the step size and Nk is a vector of standard i.i.d Gaussian noise.
9 / 33


---

## 第18页

Globalization strategies
▶Sufficient decrease condition [Armijo ’66]. Set τ > 0 such that
E(xk) −E(xk+1) −στ
∇E(xk)

2
≥0,
where σ ∈(0, 1).
10 / 33


---

## 第19页

Globalization strategies
▶Sufficient decrease condition [Armijo ’66]. Set τ > 0 such that
E(xk) −E(xk+1) −στ
∇E(xk)

2
≥0,
where σ ∈(0, 1).
▶The Metropolis-Hastings rule [Metropolis et al. ’53][Hastings ’70]: Accept xk+1 if
min
(
0, E(xk) −E(xk+1) −τ
∇E(xk+ 1
2 )

2
−
√
2τ
2

∇E(xk+ 1
2 ), Nk
)
> log u,
where u ∼U[0, 1] and ∇E(xk+ 1
2 ) = (∇E(xk) + ∇E(xk+1))/2 is the average gradient.
10 / 33


---

## 第20页

MAP vs. MMSE
▶Application of GD and ULA to total variation regularized image denoising:
R(x) = λ
X
i,j
|xi+1,j −xi,j|ε + |xi,j+1 −xi,j|ε,
D(x, y) =
1
2σ2 ∥x −y∥2 ,
using ε = 10−3, λMAP = 10, λMMSE = 20, σ = 0.1, step size τ = 2/L.
MAP solution
MMSE solution
11 / 33


---

## 第21页

Discussion
▶Optimization: A lot of research has gone into optimization in order to develop faster algorithms:
12 / 33


---

## 第22页

Discussion
▶Optimization: A lot of research has gone into optimization in order to develop faster algorithms:
▶Some milestones: Interior point methods, pre-conditioning, half-quadratic optimization, duality,
accelerated gradient methods, block-coordinate descent, proximal methods, primal-dual methods,
dynamic programming, etc.
12 / 33


---

## 第23页

Discussion
▶Optimization: A lot of research has gone into optimization in order to develop faster algorithms:
▶Some milestones: Interior point methods, pre-conditioning, half-quadratic optimization, duality,
accelerated gradient methods, block-coordinate descent, proximal methods, primal-dual methods,
dynamic programming, etc.
▶Sampling: Despite clear structural and algorithmic similarity, the progress in developing faster
sampling algorithms seems to be much slower.
12 / 33


---

## 第24页

Discussion
▶Optimization: A lot of research has gone into optimization in order to develop faster algorithms:
▶Some milestones: Interior point methods, pre-conditioning, half-quadratic optimization, duality,
accelerated gradient methods, block-coordinate descent, proximal methods, primal-dual methods,
dynamic programming, etc.
▶Sampling: Despite clear structural and algorithmic similarity, the progress in developing faster
sampling algorithms seems to be much slower.
▶Question: Can we leverage ideas from optimization to develop faster sampling algorithms?
12 / 33


---

## 第25页

Half-quadratic minimization
▶A fruitful idea in optimization has always been lifting the objective function to a high-dimensional
space where the new representation is expected to offer a better structure.
▶Let us consider a popular technique called half-quadratic minimization [Geman, Reynolds, ’92],
which expresses a function f (x) as
f (x) = min
z
q(x, z)
where q is quadratic in x.
▶The lifting represents the function f (x) as the infimum over a family of quadratic functions.
▶Minimizing f (x) is replaced by alternating minimization with respect to x and z.
xk+1 ∈argmin
x
q(x, zk),
zk+1 = argmin
z
q(xk+1, z).
▶Many variants exist (multiplicative, additive) and there are also relations to the convex conjugate
[Nikolova, Ng ’05].
13 / 33


---

## 第26页

Minimum envelope
▶The total variation regularization term from the image denoising application can be written as a
half-quadratic minimization problem [Chambolle, Lions ’97]
▶It is based on rewriting the absolute function as
R(x) = λ
X
i,j
|xi+1,j −xi,j| + |xi,j+1 −xi,j|,
|t| = min
z>0
|t|2
2z + z
2,
which is quadratic in x and simple in z with z∗(t) = |t|.
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
0.0
0.2
0.4
0.6
0.8
1.0
|x|
14 / 33


---

## 第27页

Example
▶For the TV denoising example the half-quadratic minimization algorithm converges within a few
iterations.
▶Can we develop a similar algorithm for sampling?
15 / 33


---

## 第28页

Gibbs sampling
▶Consider the following lifting of the prior pX with latent variables z:
pX(x) =
Z
pX,Z(x, z) dz,
where pX,Z is expected to have a better structure for sampling.
▶The sampling analogue to alternating minimization is Gibbs sampling [Geman, Geman, ’84]
xk+1 ∼pX|Z=zk ,
zk+1 ∼pZ|X=xk+1,
which alternates sampling from the conditional distributions.
▶Gibbs sampling can be shown to be a particular instance of the Metropolis-Hastings algorithm
[Metropolis et al. ’53][Hastings ’70].
16 / 33


---

## 第29页

Gaussian latent machine (GLM)
▶Assume the following Product of Experts (PoE) prior model [Hinton ’99]
pX(x) ∝
m
Y
j=1
ϕj((Kx)j),
where K : Rn →Rm is a linear operator and ϕj : R →R+ are 1D factors.
▶In case K is a convolutional operator, the model is equivalent to the Fields of Experts (FoE) prior
model [Roth, Black ’05].
▶In [Kuric, Zach, Habring, Unser, P. ’25] we show that a PoE prior admits the following lifted
representation in the form of a Gaussian latent machine (GLM)
pX,Z(x, z) ∝
m
Y
j=1
N((Kx)j|µj(zj), σ2
j (zj))pj(zj) = N(Kx|˜µ(z), ˜Σ(z))
|
{z
}
pX|Z=z (x)
·
m
Y
j=1
pj(zj),
where ˜µ(z) = (˜µ1(z1), ..., ˜µm(zm) and ˜Σ(z) = diag
 ˜σ2
1(z1), ..., ˜σ2
m(zm)

.
17 / 33


---

## 第30页

The conditional distributions
▶For Gibbs sampling we need to have access to the conditional distributions.
▶The conditional distribution pX|Z is simply given by the multivariate Gaussian
pX|Z(x, z) = N(Kx|˜µ(z), ˜Σ(z)) = N(x|µ(z), Σ(z))
where
µ(z) = (K ⊤˜Σ(z)−1K)−1)K ⊤˜Σ(z)−1˜µ(z),
Σ(z) = (K ⊤˜Σ(z)−1K)−1.
▶The conditional distribution pX|Z decomposes into m independent univariate distributions
pZ|X =
m
Y
j=1
pZi |X,
pZj |X=x(zj) ∝N((Kx)j|µj(zj), σ2
j (zj)) · pj(zj),
where pj basically depends on the factors ϕj.
▶In many situations of practical interest, e.g. 1D Gaussian mixture models, there are closed form
solutions for the univariate distributions pZi |X
18 / 33


---

## 第31页

Back to TV
▶Consider again the 2D total variation image prior
pX(x) ∝
Y
i,j
exp(−λ|xi+1,j −xi,j|) exp(−λ|xi,j+1 −xi,j|).
▶Next, we consider a lifted representation of 1D Laplacian factors ϕ(t) = exp(−λ|t|) as a Gaussian
scale mixture
λ
2 exp(−λ|t|) =
Z
R+
1
√
2πz
exp

−t2
2z
 λ2
2 exp

−λ2 z
2

dz
▶From the Gaussian component, one directly sees that µ(z) = 0 and σ2(z) = z.
▶The 1D conditional distribution pZi |X=t is given by a generalized inverse Gaussian
pZi |X=t ∝z−1
2 exp

−
t2
z + λ2z

/2

,
from which sampling is “relatively” easy.
19 / 33


---

## 第32页

Soft half-quadratic minimization
▶In the negative log-domain it turns out that we obtain a soft-minimum over quadratic functions
λ|t| = −log





Z
R+
2
√
2πz
exp




−
 t2
2z + λ2z
2

|
{z
}
q(t,z)




dz




,
where q(t, z) is indeed quadratic in t.
1.00
0.75
0.50
0.25
0.00
0.25
0.50
0.75
1.00
0.0
0.2
0.4
0.6
0.8
1.0
|x|
20 / 33


---

## 第33页

Gibbs sampling on the Gaussian latent machine
▶The Gibbs sampler to sample from pX alternates sampling from a multivariate Gaussian and m
independent generalized inverse Gaussians
(
xk+1 ∼N(x|0, (K ⊤diag
 zk
1 , ..., zk
m
−1 K)−1),
zk+1
j
∼GIG(λ2, (Kxk+1)2
j , 1
2), j = 1...m
▶Sampling from the prior pX(x) can be easily extended to sampling from the posterior
pX|Y = pY |X · pX by combining with a Gaussian likelihood term for denoising
pY |X=x(y) = N(y|x, σ2 · I)
which only modifies the Gaussian in the first step of the Gibbs algorithm.
(
xk+1 ∼N(x|0, (K ⊤diag
 zk
1 , ..., zk
m
−1 K)−1) · N(y|x, σ2 · I),
zk+1
j
∼GIG(λ2, (Kxk+1)2
j , 1
2), i = 1...m
21 / 33


---

## 第34页

Sampling from the TV denoising posterior
▶The MMSE computed from the samples of the GLM converges in a few iterations.
22 / 33


---

## 第35页

Sampling from a Gaussian vs. quadratic minimization
▶Consider the following multivariate Gaussian distribution
πX(x) ∝exp
 −1
2(x −µ)⊤Q(x −µ)

,
▶Sampling: Generate i.i.d. samples from the distribution
▶Optimization: Find the mode of the distribution that is minimizing the quadratic function
min
x
1
2x⊤Qx −b⊤x.
▶In optimization we assume that we only have access to b = Qµ.
23 / 33


---

## 第36页

Gibbs vs. Gauss-Seidel
▶Let us assume the following two-block structure
µ =
µ1
µ2

,
Q =
Q11
Q12
Q21
Q22

,
b =
b1
b2

=
Q11µ1 + Q12µ2
Q21µ1 + Q22µ2

▶A Gibbs sampling algorithm that takes advantage of the two-block structure is given by
(
X k+1
1
∼N(µ1 −Q−1
11 Q12(X k
2 −µ2), Q−1
11 ),
X k+1
2
∼N(µ2 −Q−1
22 Q21(X k+1
1
−µ1), Q−1
22 ),
▶A Gauss-Seidel (alternating minimization) algorithm takes the form
(
xk+1
1
= Q−1
11 (b1 −Q12xk
2 ) = µ1 −Q−1
11 Q12(xk
2 −µ2),
xk+1
2
= Q−1
22 (b2 −Q21xk+1
1
) = µ2 −Q−1
22 Q21(xk
1 −µ1),
▶Gibbs is just noisy Gauss-Seidel!
24 / 33


---

## 第37页

Overrelaxation
▶It is well known that the Gauss-Seidel algorithm can be accelerated using the method of
successive overrelaxation (SOR) [Frankel ’50]
(
xk+1
1
= (1 −ω)xk
1 + ωQ−1
11 (b1 −Q12xk
2 ),
xk+1
2
= (1 −ω)xk
2 + ωQ−1
22 (b2 −Q21xk+1
1
),
where ω ∈(0, 2) is the relaxation parameter.
25 / 33


---

## 第38页

Overrelaxation
▶It is well known that the Gauss-Seidel algorithm can be accelerated using the method of
successive overrelaxation (SOR) [Frankel ’50]
(
xk+1
1
= (1 −ω)xk
1 + ωQ−1
11 (b1 −Q12xk
2 ),
xk+1
2
= (1 −ω)xk
2 + ωQ−1
22 (b2 −Q21xk+1
1
),
where ω ∈(0, 2) is the relaxation parameter.
▶[Adler ’81] observed that the Gibbs sampling algorithm can also be overrelaxed
(
X k+1
1
∼N((1 −ω)X k
1 + ω
 µ1 −Q−1
11 Q12(X k
2 −µ2)

, ω(2 −ω)Q−1
11 ),
X k+1
2
∼N((1 −ω)X k
2 + ω
 µ2 −Q−1
22 Q21(X k+1
1
−µ1)

, ω(2 −ω)Q−1
22 ),
where it is crucial to rescale the covariance matrix by the factor ω(2 −ω) in order to ensure the
correct covariance of X k.
▶[Fox and Parker ’17] proved a convergence rate, matching that of SOR.
▶Values of ω →2 result in negative autocorrelation of successive samples.
25 / 33


---

## 第39页

Langevin meets Gibbs
▶Let’s consider again the unadjusted Langevin algorithm (ULA)
X k+1 = X k −τ∇E(X k) +
√
2τNk ⇐⇒X k+1 ∼N(X k −τ∇E(X k), 2τ · I)
▶Using the following splitting trick [Falk, Habring, P. ’25]
X k+1 = X k −τ∇E(X k) + √τ1Nk
1
|
{z
}
Y k+1
+√τ2Nk
1 ,
τ1 + τ2 = 2τ,
we can interprete ULA as a two-block Gibbs algorithm
(
Y k+1 ∼N(X k −τ∇E(X k), τ1 · I),
X k+1 ∼N(Y k+1, τ2 · I),
which however is not a true Gibbs algorithm because of a lack of a joint distribution in X and Y .
26 / 33


---

## 第40页

Inertial Langevin algorithm (ILA)
▶Now let’s apply the overrelaxation to the obtained Gibbs sampling scheme
(
Y k+1 ∼N((1 −ω)Y k + ω
 X k −τ∇E(X k)

, ω(2 −ω)τ1 · I),
X k+1 ∼N((1 −ω)X k + ωY k+1, ω(2 −ω)τ2 · I),
▶The scheme can be reduced back to a single-variable scheme
X k+1 = X k −γ∇E(X k) + β(X k −X k−1) +
p
2γ(1 −β)Nk,
where we have defined γ = ω2τ and β = (1 −ω)2.
▶The obatined scheme, which we term Inertial Langevin Algorithm (ILA) is nothing than the
sampling analogue of the heavy ball algorithm [Polyak ’64].
▶It is crucial to scale the noise by the factor
p
2γ(1 −β).
27 / 33


---

## 第41页

Comparison
▶The ground truth solution x∗
MMSE is computed using GLM.
▶Choosing an inertial parameter β →1 leads to a significant acceleration.
▶A small bias remains but can be accounted for using a Metropolis-Hastings acceptance test.
100
101
102
103
104
Iterations
100
101
102
xMMSE
x *
MMSE
2
= 0.0
= 0.5
= 0.9
= 0.99
28 / 33


---

## 第42页

Underdamped Langevin
▶The discrete ILA scheme can be shown to be a discretization of the underdamped Lagevin SDE
(
d ¯Vt
=
 −δ ¯Vt −∇E( ¯Xt)

dt +
√
2δ dWt.
d ¯Xt
= ¯Vt dt,
with friction parameter δ > 0, which is known to have a stationary distribution
π(x, v) ∝exp

−

E(x) + ∥v∥2
2

▶For a proof of convergence, we consider a slightly different parametrization of the ILA scheme,
which allows to control the discretization error with required order.
▶Theorem: Assume E is strongly convex with Lipschitz continuous gradient, then the sample
distribution of the time-discrete ILA scheme conveges to the stationary distribution of the
underdamped Langevin SDE in W2 distance as ∆t →0.
29 / 33


---

## 第43页

Conclusion
▶There is an ongoing transition from pure optimization algorithms to more general sampling
algorithms.
▶Boils down to a difference between min and softmin, where the latter is more general.
▶Many tricks that work well in optimization can be transformed to sampling.
▶Half-quadratic minimization ⇝Gaussian latent machine (GLM)
▶Inertial/accelerated gradient descent ⇝Inertial Langevin algorithm (ILA)
▶Allows to perform Bayesian inference and uncertainty quantification for high-dimensional problems.
▶For details and convergence rates see our recent preprints (will be online soon)
30 / 33


---

## 第44页

Conclusion
▶There is an ongoing transition from pure optimization algorithms to more general sampling
algorithms.
▶Boils down to a difference between min and softmin, where the latter is more general.
▶Many tricks that work well in optimization can be transformed to sampling.
▶Half-quadratic minimization ⇝Gaussian latent machine (GLM)
▶Inertial/accelerated gradient descent ⇝Inertial Langevin algorithm (ILA)
▶Allows to perform Bayesian inference and uncertainty quantification for high-dimensional problems.
▶For details and convergence rates see our recent preprints (will be online soon)
Thanks for listening!
30 / 33


---

## 第45页

Appendix: Sampling from the multivariate Gaussian in the GLM
▶In the GLM, we need to sample from a multivariate Gaussian with negative log density (up to
constants)
1
2xT

K ⊤diag

zk
1 , ..., zk
m
−1
K

x +
1
2σ2 ∥x −y∥2 ,
where z are the latent variables.
▶This quadratic function be rewritten in the standard form (up to constants)
1
2(x −µ)T 
A⊤A

(x −µ)
with
A =
"
diag
 zk
1 , ..., zk
m
−1
2 K
1
σ · I
#
,
µ =

A⊤A
−1
y/σ2.
31 / 33


---

## 第46页

Sampling from a Gaussian with precision A⊤A
▶The task is to sample from a Gaussian with negative log density
1
2(x −µ)T 
A⊤A

(x −µ)
▶A sample x is computed from a standard Gaussian sample z ∼N(0, I) as
x = µ +

A⊤A
−1
A⊤z.
▶A direct computation shows that
cov[x] =

A⊤A
−1
A⊤
 
A⊤A
−1
A⊤
⊤
=

A⊤A
−1
32 / 33


---

## 第47页

Putting together
▶Consider the previous equation
x = µ +

A⊤A
−1
A⊤z
which when inserting the formula for them mean µ gives
x =

A⊤A
−1
y/σ2 +

A⊤A
−1
A⊤z.
▶Multiplying from the left with A⊤A we obtain the linear system of equations

A⊤A

x = y/σ2 + A⊤z,
which can be solved using an iterative solver such as CG.
33 / 33


---

