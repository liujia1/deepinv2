# lecture_1.pdf

## 第1页

ERASMUS+ INTERNATIONAL PHD SUMMER SCHOOL 2025
Mathematics and Machine Learning for image analysis
Lecture 1 - Inverse Problems and Learning
Thomas Pock
Institute of Visual Computing
Graz University of Technology
University of Bologna, June 3-6 2025
1 / 56


---

## 第2页

Outline of the lecture
Lecture 1 Inverse Problems and Learning Inverse problems in imaging, image statistics, prior and likelihood
modeling, total variation, Fields-of-experts, bilevel optimization, variational networks, applications.
Lab 1 Bilevel optimization for learning a FoE model
Lecture 2 Optimization vs. Sampling: Bayesian inverse problems, MAP vs. MMSE, gradient descent vs.
Langevin, Half-quadratic minimization vs. Gaussian latent machine, underdamped Langevin vs.
accelerated gradient descent.
Lab 2 Optimization vs. sampling for TV based image denoising
Lecture 3 Learning Optimal Discretizations The total variation, various discretizations,
gamma-convergence, Piggy-back primal-dual method, extensions
All material can be found under the link
https://drive.google.com/drive/folders/14mKyGDQwjfJQl_Kp_lZzxb6FcgFgEw6M?usp=sharing
2 / 56


---

## 第3页

Inverse problems in imaging
As the main topic of this session, we are interested in solving ill-posed inverse problems of the form
Ax ≈y
where A is the linear forward operator and y are the measurements.
Example CT:
Ax = y ⇐⇒y(r, θ) =
Z ∞
−∞
x(rθ + sθ⊥) ds
Measurements y
Reconstructed image x
3 / 56


---

## 第4页

Inverse problems in imaging
As the main topic of this session, we are interested in solving ill-posed inverse problems of the form
Ax ≈y
where A is the linear forward operator and y are the measurements.
Example CT:
Ax = y ⇐⇒y(r, θ) =
Z ∞
−∞
x(rθ + sθ⊥) ds
Measurements y
Reconstructed image x
Least squares solution:
min
x
1
2 ∥Ax −y∥2 =⇒xk+1 = xk −tkA∗(Axk −y)
3 / 56


---

## 第5页

Example: Least squares
Measurements y (sinogram)
4 / 56


---

## 第6页

Example: Least squares
Measurements y (sinogram)
Least squares solution x
4 / 56


---

## 第7页

Regularization
▶The common remedy in inverse problems is to regularize the solution.
min
x
1
2 ∥Ax −y∥2
2 + R(x).
▶There are two popular regularization approaches:
5 / 56


---

## 第8页

Regularization
▶The common remedy in inverse problems is to regularize the solution.
min
x
1
2 ∥Ax −y∥2
2 + R(x).
▶There are two popular regularization approaches:
▶Analysis-based regularization:
min
x
λ ∥∇x∥2,1 + 1
2 ∥x −y∥2
2 ,
(e.g. Total Variation)
where ∇is a finite differences operator.
▶Synthesis-based regularization:
5 / 56


---

## 第9页

Regularization
▶The common remedy in inverse problems is to regularize the solution.
min
x
1
2 ∥Ax −y∥2
2 + R(x).
▶There are two popular regularization approaches:
▶Analysis-based regularization:
min
x
λ ∥∇x∥2,1 + 1
2 ∥x −y∥2
2 ,
(e.g. Total Variation)
where ∇is a finite differences operator.
▶Synthesis-based regularization:
min
z
λ ∥z∥1 + 1
2 ∥Ψ∗z −y∥2
2 ,
(e.g. lasso)
where Ψ is a suitable linear operator, e.g. a dictionary or a wavelet transform.
Assumptions on the sparsity of the solution can lead in some cases to exact recovery [Cand`es,
Romberg, Tao ’06], [Donoho ’06].
5 / 56


---

## 第10页

Example: TV regularization
Measurements y (sinogram)
6 / 56


---

## 第11页

Example: TV regularization
Measurements y (sinogram)
TV regularized solution x
6 / 56


---

## 第12页

Relation to Bayes’ theorem
min
x∈X

D(x, y) + R(x)

7 / 56


---

## 第13页

Relation to Bayes’ theorem
min
x∈X

D(x, y) + R(x)

Data fidelity term:
▶Penalize deviation from the measurements y using forward operator A:
For example: D(x, y) = 1
2 ∥Ax −y∥2
2
7 / 56


---

## 第14页

Relation to Bayes’ theorem
min
x∈X

D(x, y) + R(x)

Data fidelity term:
▶Penalize deviation from the measurements y using forward operator A:
For example: D(x, y) = 1
2 ∥Ax −y∥2
2
Regularizer:
▶Impose prior knowledge on the solution:
For example: R(x) = ∥∇x∥2,1
7 / 56


---

## 第15页

Relation to Bayes’ theorem
min
x∈X

D(x, y) + R(x)

Data fidelity term:
▶Penalize deviation from the measurements y using forward operator A:
For example: D(x, y) = 1
2 ∥Ax −y∥2
2
Regularizer:
▶Impose prior knowledge on the solution:
For example: R(x) = ∥∇x∥2,1
Statistical Interpretation:
▶Maximum a-posteriori (MAP) estimate on Bayes’ theorem
bx = argmax
x

p(x|y) = p(y|x)p(x)
p(y)

= argmin
x

−log p(y|x) + −log p(x)

7 / 56


---

## 第16页

Relation to Bayes’ theorem
min
x∈X

D(x, y) + R(x)

Data fidelity term:
▶Penalize deviation from the measurements y using forward operator A:
For example: D(x, y) = 1
2 ∥Ax −y∥2
2
▶Negative log-likelihood
Regularizer:
▶Impose prior knowledge on the solution:
For example: R(x) = ∥∇x∥2,1
▶Negative log-prior
Statistical Interpretation:
▶Maximum a-posteriori (MAP) estimate on Bayes’ theorem
bx = argmax
x

p(x|y) = p(y|x)p(x)
p(y)

= argmin
x

−log p(y|x) + −log p(x)

7 / 56


---

## 第17页

Example: Image denoising
▶Let us consider the following classical energy-based formulation of denoising an image y that has
been corrupted with Gaussian noise.
E(x|y) := R(x) + 1
2 ∥x −y∥2 ⇐⇒p(x|y) = 1
Z exp (−E(x|y)/T) ,
where we assume here that R is some convex energy, e.g. the total variation T > 0 is some
scaling parameter and Z is the normalization constant.
▶This energy-based formulation leads to the following two approaches:
▶1. Computing the state of minimal energy:
ˆx = min
x
R(x) + 1
2 ∥x −y∥2 .
▶The function y 7→(R□1
2 ∥·∥2)(y) = minx R(x) + 1
2 ∥x −y∥2 is the infimal convolution of R with a
quadratic function, called Moreau envelope.
▶The first order optimality condition yields
0
∈
∂R(ˆx) + ˆx −y
y
∈
(I + ∂R)(ˆx)
ˆx
=
(I + ∂R)−1(y)
ˆx
=
proxR(y),
▶This is the celebrated proximal map [Rockafellar’76] which is the foundation of the famous
proximal algorithm.
8 / 56


---

## 第18页

Example: Image denoising
▶2. Computing the posterior expectation:
¯x = EX|Y [x|y] = 1
Z
Z
x exp (−E(x|y)/T) ,
where Z =
R
X exp (−E(x|y)/T) dx is the normalization constant.
▶The posterior expectation is also obtained as minimum least squares estimator minimizing the
expected quadratic loss
¯x = EX|Y [x|y] ∈argmin
x′
EX[
x′ −x
2 |y]
▶Computing the expectation is intractable in general but has an elegant representation by means of
the so-called Tweedie’s formula.
▶Tweedie’s formula is one of the important bricks in score-matching based learning algorithms and
in recent diffusion models.
9 / 56


---

## 第19页

Interlude: Derivation of Tweedie’s formula (1)
▶Assume, we have some data represented by a random variable X ∈Rd, with density pX(x) 1.
▶Moreover, let us assume that our observations Y ∈Rd are noisy variants of the random variable
X, where the noise is i.i.d additive Gaussian noise with variance σ2.
gσ2(x) =
1
√
2πσ2 exp
 
−∥x∥2
2σ2
!
.
▶The likelihood can be conveniently written as
pY |X(y|x) = gσ2(x −y) =
1
√
2πσ2 exp
 
−∥x −y∥2
2σ2
!
.
▶The aim of Tweedie’s formula is now to find a simple formula for computing the posterior
expectation
EX|Y [x|y] =
Z
x pX|Y (x|y) dx =
Z
x pY |X(y|x)pX(x)
pY (y)
1This is already quite a strong assumption, as the respective probability distribution is assumed to be absolutely continuous
with respect to the Lebesgue measure.
10 / 56


---

## 第20页

Interlude: Derivation of Tweedie’s formula (2)
▶We start by expanding the expression for the density of the random variable Y
pY (y) =
Z
p(x, y) dx =
Z
pY |X(y|x)pX(x) dx =
Z
gσ2(x −y)pX(x) dx.
▶Now taking the derivative wrt to y on both sides
∇pY (y)
=
Z
∇gσ2(x −y)pX(x) dx = 1
σ2
Z
gσ2(x −y)pX(x)(x −y) dx
=
1
σ2
Z
gσ2(x −y)pX(x)x dx −1
σ2
Z
gσ2(x −y)pX(x)y dx
=
pY (y)
σ2
Z pY |X(y|x)pX(x)
pY (y)
x dx −y
σ2
Z
pY |X(y|x)pX(x) dx
=
pY (y)
σ2
Z
x pX|Y (x|y) dx −pY (y)
σ2
y = pY (y)
σ2
 EX|Y [x|y] −y

▶Now observing that ∇pY (y)
pY (y) = ∇log pY (y):
σ2 ∇pY (y)
pY (y)
= σ2∇log pY (y) = EX|Y [x|y] −y.
▶Finally, we obtain Tweedie’s formula
EX|Y [x] = y + σ2∇log pY (y).
11 / 56


---

## 第21页

Expectation vs. Minimization
0
100
200
300
400
500
Marginalization/Expectation
0
100
200
300
400
500
Minimization
▶Example using total variation regularization and a quadratic data fidelity term.
▶In 1D, both results can be computed efficiently using dynamic programming.
▶The minimization leads to the well-known staircasing effect.
▶The computation of the expectation leads to more natural results.
12 / 56


---

## 第22页

Min versus softmin
▶MAP estimation (minimization)
ˆx = argmin
x∈X
R(x) +
1
2τ ∥x −y∥2
2 ,
ˆx = proxτR(y),
which can be written as one gradient step on the
infimal-convolution of R with a quadratic function
∥·∥2
2
2τ
ˆx = y −τ∇ˆRτ(y),
ˆRτ = (R□
∥·∥2
2
2τ ).
13 / 56


---

## 第23页

Min versus softmin
▶MAP estimation (minimization)
ˆx = argmin
x∈X
R(x) +
1
2τ ∥x −y∥2
2 ,
ˆx = proxτR(y),
which can be written as one gradient step on the
infimal-convolution of R with a quadratic function
∥·∥2
2
2τ
ˆx = y −τ∇ˆRτ(y),
ˆRτ = (R□
∥·∥2
2
2τ ).
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
which can be written in the form of one gradient step on the softmin-convolution of exp(−R)
with a Gaussian exp (−
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
This formula is known as Tweedie’s formula [Robbins ’56], [Miyasawa ’60].
13 / 56


---

## 第24页

Learning regularization functions
▶It is nowadays clear that the simple handcrafted regularizers must be improved.
▶Recent trend to apply learning techniques to the field of inverse problems.
▶There are two two fundamentally different learning approaches:
14 / 56


---

## 第25页

Learning regularization functions
▶It is nowadays clear that the simple handcrafted regularizers must be improved.
▶Recent trend to apply learning techniques to the field of inverse problems.
▶There are two two fundamentally different learning approaches:
▶Discriminative (task-driven) learning (learning the posterior)
Directly learn: pθ(x|y) ∼exp(−Rθ(x) −Dθ(x, y))
▶Generative (task-agnostic) learning (learning the prior)
Learn: pθ(x) ∼exp(−Rθ(x)) and use in p(x|y) ∼exp(−Rθ(x) −D(x, y))
14 / 56


---

## 第26页

A deep learning approach to inverse problems: AUTOMAP
Image reconstruction by domain-transform manifold learning [Zhu et al. ’18]
▶Does it really make sense to ignore the knowledge of the forward operator?
15 / 56


---

## 第27页

Synthesis-based approaches
▶A popular synthesis-based approach is to express the unknown signal x by a highly parametrized
and in general nonlinear “generator”:
x = Gθ(z),
where z is usually of lower dimension as compared to x.
▶The low-dimensional parametrization (architecture) imposes an implicit regularization on the
solution.
▶The solution of the least squares problem is then given by
min
z,θ
1
2 ∥AGθ(z) −y∥2
2 +R(z),
▶Deep image prior: [Ulyanov, Vedaldi, Lempitsky ’17]: z is taken at random and the minimization
is taken over the parameter vector θ.
▶GAN-based approaches: [Bora et al. ’17][Narnhofer et al. ’19]: θ is obtained from a pre-training
using a GAN-based approach and the minimization is taken over z.
▶Conditional GAN for efficient posterior sampling [Bendel, Ahmad, Schniter ’22].
16 / 56


---

## 第28页

Interlude: Generative adversarial networks
▶Generative adversarial networks (GANs) are artificial neural networks that learn to generate data
similar to the samples of a training dataset.
▶A GAN consists of two parts: a generator and a discriminator.
▶The generator takes in a random vector sampled from a latent distribution and generates an
artificial sample.
▶The discriminator is trained to distinguish between the generated samples and real samples from
the training dataset.
▶Both components of the GAN are trained together in a process where the generator tries to
produce samples that are hard to discriminate, and the discriminator tries to correctly identify
whether each sample is real or generated.
▶Formulated as a two-player minimax game.
▶GANs have been used to create realistic images, music, and even text.
17 / 56


---

## 第29页

The generator and discriminator
▶The generator is a map (usually parametrized by a neural network) that transforms a sample from
a latent distribution (typically a uniform distribution or a Gaussian) to the data distribution
x = G(z), z ∼PZ,
hence the generator G generates a distribution PG that can be seen as the push-forward G♯PZ of
the latent distribution PZ.
▶Note that in contrast to normalizing flows, we do not make any assumptions on the invertability
of the map G.
▶As a result, we will not be able to directly evaluate the density value pG(x).
▶The discriminator takes the role of a classifier (confirm logistic regression)
y = D(x), y ∈[0, 1],
where y denotes the probability for x coming from the data rather than being generated.
▶It is again parametrized by means of a neural network with a sigmoid activation function at its tail.
18 / 56


---

## 第30页

The GAN objective
▶The GAN objective consist of maximizing with respecto to D, the log-probability of assigning the
correct label to both the real and generated samples.
▶Simultaneously on tries to minimize with respect to G the probability of correctly classifying the
generated samples.
▶This leads to the following minimax game:
min
G max
D
L(G, D) := Ex∼PX [log D(x)]
|
{z
}
log-likelihood on the data
+ Ez∼PZ [log(1 −D(G(z))]
|
{z
}
log-likelihood on the generated samples
▶The GAN objective represents a non-convex saddle-point problem which can be optimized by
alternating gradient descent in G and gradient ascent in D.
▶The training procedure can be very delicate in practice.
19 / 56


---

## 第31页

Optimal solution of the discriminator
Proposition
For fixed generator G, the optimal discriminator D∗is given by
D∗(x) =
pX(x)
pX(x) + pG(x).
Proof.
L(G, D)
=
Z
X
log(D(x)) dPX(x) +
Z
Z
log(1 −D(G(z))) dPZ(z)
=
Z
X
log(D(x)) dPX(x) +
Z
X
log(1 −D(x)) dPG(x)
=
Z
X
 dPX(x)
dP(x) log(D(x)) + dPG(x)
dP(x) log(1 −D(x))

dP(x),
where P = PX + PG is some base measure and we define pX(x) =
dPX (x)
dP(x) , pG(x) =
dPG (x)
dP(x) , as the
Radon-Nikodym derivatives. The result follows from computing the maximizer of the function
x 7→a log x + b log(1 −x), which is given by x∗= a/(a + b).
20 / 56


---

## 第32页

Expression of the global optimum
Using the previous result, the GAN objective can now be written as a minimization problem
min
G L(G, D∗) := Ex∼PX

log
pX(x)
pX(x) + pG(x)

+ Ex∼PG

log
pG(x)
pX(x) + pG(x)

.
Theorem
The global minimum of L(G, D∗) is achieved iff PG = PX with value −log 4.
Proof.
For PG = PX one has D∗(x) = 1
2, so that the value of the objective becomes
L(G, D∗) = log 1
2 + log 1
2 = −log 4. On the other hand, Jensen inequality shows that this value
coincides with the lower bound
Ex∼PX

log
pX(x)
pX(x) + pG(x)

+ Ex∼PG

log
pG(x)
pX(x) + pG(x)

=
Ex∼PX

−log pX(x) + pG(x)
pX(x)

+ Ex∼PG

−log pX(x) + pG(x)
pG(x)

≥
−log Ex∼PX
pX(x) + pG(x)
pX(x)

−log Ex∼PG
pX(x) + pG(x)
pG(x)

=
−log(1 + 1) −log(1 + 1) = −2 log 2 = −log 4.
21 / 56


---

## 第33页

Relation to the Jensen-Shannon divergence
▶Using the previous results, the GAN objective with optimal discriminator can be written as
L(G, D∗)
=
Ex∼PX

log
pX(x)
pX(x) + pG(x)

+ Ex∼PG

log
pG(x)
pX(x) + pG(x)

=
−log 4 + Ex∼PX

log
pX(x)
(pX(x) + pG(x))/2

+ Ex∼PG

log
pG(x)
(pX(x) + pG(x))/2

=
−log 4 + KL (pX||(pX + pG)/2) + KL (pG||(pX + pG)/2)
=
−log 4 + 2JS(pX, pG),
where
1 ≥JS(p||q) = 1
2KL(p||(p + q)/2) + 1
2KL(q||(p + q)/2) ≥0,
is the Jensen-Shannon, divergence, which is a symmetrized form of the KL divergence.
22 / 56


---

## 第34页

Application: GANs for MRI reconstruction
Latent Vector z
Fully Connected 
Batch Normalization
ReLU
Convolution 3x3
c512 r5x5
Generator
b  tanh
Re
Im
Discriminator
Re
Im
Mag
Fully Connected 
Linear
D(u)
Convolution 3x3
LReLU
[Narnhofer et al. ’19]
23 / 56


---

## 第35页

Sampled Magnitude Images
Generated Samples
Source Samples
Figure: Samples created by the trained Generator (left) alongside samples from the source data (right).
24 / 56


---

## 第36页

Advantages of the GAN-based regularizer
▶Training independent of imaging modalities
▶No corrupted data needed in training
▶Can address multiple inverse problems
▶In MRI: Denoising, Different Acceleration Factors, Sampling patterns...
Figure: Possible Sampling Masks. Cartesian Random(left), Radial(middle), Spiral(right).
25 / 56


---

## 第37页

Reconstrucion
▶We aim at finding the latent vector z ∈Rd that yields the estimate Ax = AGθ(z) which is closest
to y within the range of the generator.
ˆz = argmin
z∈Rd
1
2∥AGθ(z) −y∥2,
s.t. ∥z∥≤
√
d,
▶However, this yields an image in the range space of the generator that fails to sufficiently
reconstruct the target image on a high level of detail.
▶We address this issue by adapting the implicit prior towards the observed data:
(ˆz, ˆθ) =
argmin
(z,θ)∈Rd ×Rl
1
2∥AGθ(z) −y∥2,
s.t. ∥z∥≤
√
d.
▶The optimization is done using the iPALM algorithm [P., Sabach ’16] algorithm.
26 / 56


---

## 第38页

Inverse GANs for accelerated MRI reconstruction
27 / 56


---

## 第39页

Bilevel optimization [Haber, Tenorio ’03], [Samuel, Tappen ’09], [Crockett, Fessler ’21], ...
▶We now turn to a different learning approach.
▶Given training data (yi)n
i=1 together with ground truth solutions (x∗
i )n
i=1.
▶Bilevel optimization aims at solving the following problem
min
θ
1
n
n
X
i=1
ℓ(x∗
i , ˆxi),
s.t. ˆxi ∈argmin
x
Rθ(x) + 1
2 ∥x −yi∥2 .
▶If Eθ is strongly convex and twice continuously differentiable in x then one can make use of
implicit differentiation to compute gradients.
▶A simple but effective choice for a parametrized regularizer is given by the Fields of Experts model.
28 / 56


---

## 第40页

The Fields of Experts model
▶The statistics of natural images tells us that a good match for the
response of any zero-mean filter is obtained for the student’s-t
distribution t 7→log(1 + |t|2/µ2) [Huang and Mumford ’99].
▶Let us consider the following nonconvex model [Roth, Black ’09],
[Samuel, Tappen ’09], called the “Fields of Experts” model:
Rθ(x) =
NK
X
k=1
X
i,j
ρk((Kkx)i,j),
where Kk are linear operators implementing 2D convolutions with
small filters fk, that is fk ∗x ⇔Kkx, and ρk(t) = αk log(1 + |t|2).
▶All learnable parameters are summarized in the parameter vector
θ = (αk, fk).
0.75
0.50
0.25
0.00
0.25
0.50
0.75
0.0
0.2
0.4
0.6
0.8
1.0
marginal
|x|2
|x|
|x|
log(1 + |x|2/
2)
29 / 56


---

## 第41页

Interlude: Implicit differentiation
▶In order to compute gradients of the loss function with respect to θ, we replace the lower-level
optimization problem by its first-order optimality condition (assuming n = 1 and dropping the
index):
NK
X
k=1
K ∗
k ϕk(Kkx) + x −y = 0,
ϕk(t) = diag(..., ρ′
k(tij), ...),
where K ∗
k denotes the adjoint filter and consider the Lagrangian functional
L(x, θ, p) = ∥x −x∗∥2 + J(θ) + (
NK
X
k=1
K ∗
k ϕk(Kkx) + x −y)Tp,
where p is a vector of Lagrange multipliers and J(θ) is a regularization term acting on the
parameter vector.
▶Assuming the existence of a regular local minimum in (x, θ), we can invoke the classical Lagrange
multiplier theorem, which guarantees the existence of multipliers p such that:


(PNK
k=1 K ∗
k Dϕk(Kkx)Kk + I)p + x −x∗
DθJ(θ) + (Dθ
PNK
k=1 K ∗
k ϕk(Kkx))p
PNK
k=1 K ∗
k ϕk(Kkx) + x −y

= 0,
where the first line is the gradient of the Lagrangian with respect to x, the second line is the
gradient of the Lagrangian with respect to θ and the last line is the gradient of the Lagrangian
with repect to p.
30 / 56


---

## 第42页

Implicit differentiation
▶For fixed θ, the system can be reduced by first solving the lower level problem (last equation) for
ˆx, that is
NK
X
k=1
K ∗
k ϕk(Kk ˆx) + ˆx −y = 0,
then one can solve for ˆp by solving the linear system
ˆp =
 NK
X
k=1
K ∗
k Dϕk(Kk ˆx)Kk + I
!−1
(x∗−ˆx) ,
and finally the gradient of the loss function with respect to θ is given by
∂θL(θ) = DθJ(θ) + (Dθ
NK
X
k=1
K ∗
k ϕk(Kk ˆx))
 NK
X
k=1
K ∗
k Dϕk(Kk ˆx)Kk + I
!−1
(x∗−ˆx),
which is nothing else then implicit differentiation.
▶The loss function can then be minimized using any gradient-based optimization algorithm.
31 / 56


---

## 第43页

The learned filters and functions
▶In [Chen, Ranftl, P. ’14] we learned 80 filters of size 9 × 9 plus function parameters →6480
parameters on a database of ∼200 images using bilevel optimization
▶... two weeks later (using Matlab) ...
32 / 56


---

## 第44页

The learned filters and functions
▶In [Chen, Ranftl, P. ’14] we learned 80 filters of size 9 × 9 plus function parameters →6480
parameters on a database of ∼200 images using bilevel optimization
▶... two weeks later (using Matlab) ...
(5.21,0.33)
(4.83,0.03)
(4.77,0.02)
(4.73,0.01)
(4.65,0.02)
(4.50,0.42)
(4.29,0.01)
(3.96,0.24)
(3.40,0.23)
(2.71,0.69)
(5.03,0.22)
(4.82,0.27)
(4.77,0.05)
(4.73,0.02)
(4.65,0.23)
(4.48,0.10)
(4.17,0.34)
(3.94,0.50)
(3.24,0.70)
(2.59,0.44)
(4.96,0.29)
(4.81,0.25)
(4.75,0.02)
(4.71,0.01)
(4.63,0.02)
(4.46,0.10)
(4.09,0.14)
(3.89,0.44)
(3.22,0.59)
(2.59,0.39)
(4.88,0.13)
(4.81,0.07)
(4.75,0.13)
(4.71,0.03)
(4.61,0.01)
(4.42,0.01)
(4.03,0.29)
(3.72,0.60)
(3.15,0.43)
(2.37,0.63)
(4.87,0.22)
(4.81,0.08)
(4.75,0.25)
(4.70,0.13)
(4.60,0.10)
(4.39,0.03)
(4.02,0.25)
(3.64,0.32)
(3.09,0.45)
(2.15,1.17)
(4.84,0.01)
(4.81,0.02)
(4.74,0.02)
(4.68,0.23)
(4.56,0.02)
(4.34,0.01)
(4.00,0.41)
(3.58,0.27)
(2.90,0.59)
(2.14,0.78)
(4.83,0.13)
(4.80,0.05)
(4.74,0.18)
(4.68,0.20)
(4.53,0.01)
(4.32,0.34)
(3.99,0.27)
(3.53,0.23)
(2.88,0.24)
(1.90,0.79)
(4.83,0.02)
(4.78,0.06)
(4.73,0.02)
(4.68,0.01)
(4.51,0.19)
(4.32,0.23)
(3.97,0.13)
(3.41,0.29)
(2.74,0.58)
(1.51,0.56)
32 / 56


---

## 第45页

Denoising results for σ = 25
Original image
Noisy image
33 / 56


---

## 第46页

Denoising results for σ = 25
Original image
TV denoised
33 / 56


---

## 第47页

Denoising results for σ = 25
Original image
FoE prior
33 / 56


---

## 第48页

Unrolling [Gregor, LeCun ’10][Domke ’12][Chen, Wei, P. ’15], ...
▶A computational attractive alternative to bilevel learning is unrolling.
▶Find the parameter vector θ by replacing a minimizer in the bilevel problem by the result of
“unrolling” K iterations of an iterative solver.
▶Parameters can be time-evolving (“TNRD”) [Chen, Wei, P. ’15] or kept constant (“variational
network”) [Kobler et al. ’17].
▶Has become very popular as its provides a link to deep learning [He et al. ’15].
▶Accelerated MRI reconstruction [Hammernik, P., Knoll, et al. ’18, ...]
1
34 / 56


---

## 第49页

Unrolling [Gregor, LeCun ’10][Domke ’12][Chen, Wei, P. ’15], ...
▶A computational attractive alternative to bilevel learning is unrolling.
▶Find the parameter vector θ by replacing a minimizer in the bilevel problem by the result of
“unrolling” K iterations of an iterative solver.
▶Parameters can be time-evolving (“TNRD”) [Chen, Wei, P. ’15] or kept constant (“variational
network”) [Kobler et al. ’17].
▶Has become very popular as its provides a link to deep learning [He et al. ’15].
▶Accelerated MRI reconstruction [Hammernik, P., Knoll, et al. ’18, ...]
5
34 / 56


---

## 第50页

Unrolling [Gregor, LeCun ’10][Domke ’12][Chen, Wei, P. ’15], ...
▶A computational attractive alternative to bilevel learning is unrolling.
▶Find the parameter vector θ by replacing a minimizer in the bilevel problem by the result of
“unrolling” K iterations of an iterative solver.
▶Parameters can be time-evolving (“TNRD”) [Chen, Wei, P. ’15] or kept constant (“variational
network”) [Kobler et al. ’17].
▶Has become very popular as its provides a link to deep learning [He et al. ’15].
▶Accelerated MRI reconstruction [Hammernik, P., Knoll, et al. ’18, ...]
10
34 / 56


---

## 第51页

Optimal-control interpretation
▶In [Effland, Kobler, Kunisch, P. ’19], we consider learning via unrolling as an optimal control
problem in a continuous-time setting.
▶The discrete iteration becomes an ODE.







min
T,θ
1
2n
n
X
i=1
∥xi(t) −x∗
i ∥2
s.t.
d
dt xi(t) + ∇Eθ(xi(t), yi) = 0,
t ∈[0, T].
▶Allows to learn the optimal stopping time T as a continuous parameter
▶Interestingly, the possibility of learning the stopping time always leads to an early stopping
behavior.
xi(∞)
xi(T)
x∗
i
xi(0)
35 / 56


---

## 第52页

Total deep variation
▶In [Kobler, Effland, Kunisch, P. ’20] we
proposed a learned regularizer, called Total
Deep Variation (TDV).
▶Given by the sum of pixelwise deep
variation r(x, θ)i,j, i.e.
TDV (x, θ) =
X
i,j
r(x, θ)i,j
▶r(x, θ) = Ψ(wN(Kx)), where
▶K matrix representation of learned 3 × 3
convolution kernel with 32 feature channels
and zero-mean constraint
▶N multiscale convolutional neural network
(right),
▶w learned 1 × 1 convolution kernel,
▶Ψ potential function
r(푥, 휃)
퐾
N
푤
Ψ
Bl1
Bl2
Bl3
R2
1,1
R2
2,1
R2
3,1
R2
2,2
R2
1,2
+
+
+
+
퐾2
3,1,1
Φ
퐾2
3,1,2
+
+
downsampling
upsampling
addition
36 / 56


---

## 第53页

Example
Original image
Noisy image, σ = 25, (PSNR = 20dB)
37 / 56


---

## 第54页

Example
Original image
TV, (PSNR = 29.91dB)
37 / 56


---

## 第55页

Example
Original image
TDV-VN denoised, (PSNR = 32.81dB)
37 / 56


---

## 第56页

Plug-and-play (PnP) and regularization-by-denoising (RED) priors
▶A very popular data-driven approach [Venkatakrishnan et al. ’13 (PnP)], [Romano, M. Elad, and P.
Milanfar ’17 (RED)], which is losely related to generative learning.
▶Consider the following forward-backward splitting algorithm to solve linear inverse problems
xk+1 = proxtk R

xk −tkA∗(Axk −b)

.
38 / 56


---

## 第57页

Plug-and-play (PnP) and regularization-by-denoising (RED) priors
▶A very popular data-driven approach [Venkatakrishnan et al. ’13 (PnP)], [Romano, M. Elad, and P.
Milanfar ’17 (RED)], which is losely related to generative learning.
▶Consider the following forward-backward splitting algorithm to solve linear inverse problems
xk+1 = proxtk R

xk −tkA∗(Axk −b)

.
▶We have already seen that the proximal map can be seen as a “denoiser”.
▶This enables the application of a complete “zoo” of handcrafted (e.g. BM3D) or learned denoisers
(e.g. DnCNN), replacing the proximal map.
▶Convergence is only guaranteed under certain conditions, e.g. if the denoiser represents a firmly
non-expansive operator, see the recent PhD thesis [Terris ’21].
38 / 56


---

## 第58页

Results – undersampled CT
Variational CT reconstruction (A: undersampled Radon transform)
min
x
TDV (x, θ) + Rλ
2 ∥ARx −y∥2
2,
where θ was learned for Gaussian denoising and λ is a hand-chosen balance parameter.
39 / 56


---

## 第59页

Results – accelerated multi-coil MRI
Variational MRI reconstruction of individual coil images.
min
x
TDV (|xi|, θ),
s.t. MRFxi = yi, i = 1, ..., NC
where θ was learned for Gaussian denoising. The final image is reconstructed by computing the ℓ2
norm over the individual images xi.
40 / 56


---

## 第60页

Nonlinear eigenmode analysis of TDV
▶In the linear case, eigenvectors x ∈Rn and associated eigenvalues λ ∈R of a spd matrix
A ∈Rn×n satisfy
Ax = λx ⇐⇒∇
1
2x⊤Ax

= λx
▶In the non-linear case we can search for eigenfunctions (eigenimages) x such that
∇TDV (x, θ) = λ(x)x.
▶Nonlinear eigenimages can be computed by solving the non-linear least squares problem
min
x
1
2∥∇TDV (x, θ) −λ(x)x∥2
2,
with the generalized Rayleigh quotient defining the eigenvalues, given by
λ(x) = ⟨∇TDV (x, θ), x⟩
∥x∥2
2
.
▶Then, a gradient descent with the iterate xk being an eigenfunction gives
xk+1 = xk −h∇TDV (xk) = (1 −hλ(xk))xk.
41 / 56


---

## 第61页

Some computed eigenimages
Original images
42 / 56


---

## 第62页

Some computed eigenimages
Denoising eigenimages
42 / 56


---

## 第63页

Robustness against adversarial attacks
▶Robustness against adversarial attacks of TDV (Gaussian denoising, σ = 25)
▶x denotes a ground truth image patch, ξ ∼N(0, σ2I) Gaussian noise, y = x + ξ
▶Adversarial attack a for ϵ > 0 computed via
max
a:∥a∥2≤ϵ ∥xK(y + a) −x∥2
2
▶Adversarial attacks for TDV are natural image structures!
43 / 56


---

## 第64页

Empirical generalization error
▶The generalization error is defined as the (absolute) difference between the expected loss and the
empirical loss
Egen = |Eexp −Eemp|,
Eexp = Ex,ξ[ℓ(xK(x + ξ), x)],
Eemp = 1
N
N
X
n=1
ℓ(xK
n (xn + ξn), xn)
▶It can be upper bounded by looking for error of the “worst case” image example
|Eexp −Eemp| ≤Eworst −Eemp
▶It turns out that the error scales (almost linearly) with the value of TDV .
44 / 56


---

## 第65页

Empirical generalization error
▶In order to characterize the worst-case images to reasonable images, we bound their value of TDV
within quantiles q ∈[0, 1] of the empirical data:
Eworst =
max
TDV (xq)≤Vq ℓ(xK(xq + ξ), xq)
▶It turns out that the generalization error is in the range of 5-10 dB.
45 / 56


---

## 第66页

Learning the data fidelity term
▶In practice, the distribution of the noise ξ is usually unknown, hence the choice of the data fidelity
term is less clear.
▶In recent work [Pinetz, Kobler, P., Effland ’21] we propose to learn the data fidelity term
D(x, y) =
X
i
dϑ((Ax)i, yi)
by directly learning its proximal map.
▶In order to learn a “true” proximal map we constrain it to be 1-Lipschitz and monotone.
▶We consider different structural restrictions of the data fidelity term: Fr´echet metric and a
generalized divergence.
46 / 56


---

## 第67页

Example
Noisy input, 50% s&p noise, PSNR=7.84
47 / 56


---

## 第68页

Example
ℓ2 data fidelity term, PSNR=19.91
47 / 56


---

## 第69页

Example
Fr´echet metric, PSNR=24.66
47 / 56


---

## 第70页

Example
Generalized divergence, PSNR=28.14
47 / 56


---

## 第71页

Example
Ground truth
47 / 56


---

## 第72页

Learned proximal maps and functions
Fr´echet metric
0.0
0.5
1.0
Ax
0.5
1.0
z
0.0
0.5
1.0
Ax
0.5
1.0
z
proximal map proxdϑ
function dϑ
... learned a ℓ1 data fidelity term
48 / 56


---

## 第73页

Learned proximal maps and functions
Generalized divergence
0.0
0.5
1.0
Ax
0.5
1.0
z
0.0
0.5
1.0
Ax
0.5
1.0
z
proximal map proxdϑ
function dϑ
... learned a {0, 1} inpainting mask
48 / 56


---

## 第74页

Learning a regularizer for CT reconstruction
▶The regularization energy takes the form of a neural network [Zach, Kobler, P. ’21]
Rθ(x) = CNNθ(x) : Rd →R
128
Nf
64
2Nf
32
4Nf
16
8Nf
8
12Nf
4
16Nf
1
1
Rθ(x)
▶Nf = 48, 12.179.905 Parameters
▶Unsupervised training via maximum likelihood learning on CT images of size 128 × 128.
49 / 56


---

## 第75页

Interlude: Maximum likelihood learning
▶The energy Rθ(x) = CNNθ(x) can be associated with the following probability density function
pθ(x) =
exp (−CNNθ(x))
Z
X
exp (−CNNθ(x)) dx
▶The aim of maximimum likelhood learning is to minimize on a given data set DX the negative
log-likelihood function
min
θ ML(θ) := Ex∼DX [CNNθ(x)] + log
Z
X
exp (−CNNθ(x)) dx

.
▶For learning one needs to compute gradients of ML(θ) with respect to θ.
▶They are computed as
∇θML(θ)
=
Ex∼DX [∇θCNNθ(x)] +
R
X exp (−CNNθ(x))(−∇θCNNθ(x)) dx
R
X exp (−CNNθ(x)) dx
=
Ex∼DX [∇θCNNθ(x)] −Ex∼Pθ[∇θCNNθ(x)]
▶The gradient is given by the expected gradient of the CNN on the data set minus the expected
gradient of the CNN under the current model.
▶The first term is usually approximated by averaging the gradient on the data set.
▶Unfortunately, the second term is intractable and needs to be approximated using sampling
techniques.
50 / 56


---

## 第76页

Sampling
▶After learning the energy-based model, we can approximately “draw” samples from the
corresponding prior by running the unadjusted Langevin algorithm (ULA)
xk+1 = xk −τ k∇Rθ(x) +
√
2τ kξ,
ξ ∼N(0, 1).
▶The Markov chain {xk} generated by ULA converges to a stationary distribution Pτ
θ and
converges to Pθ for τ k →0 and P
k τ k = ∞.
▶Conversely, for τ k > 0, the obtained distribution contains some bias and hence remains
unadjusted.
51 / 56


---

## 第77页

Application to limited-angle tomography
FBP
SART
TV
Ours
Reference
FBP
SART
TV
Ours
19.05
27.72
29.67
34.21
52 / 56


---

## 第78页

53 / 56


---

## 第79页

54 / 56


---

## 第80页

Posterior Sampling — Limited-Angle (θ ∈[0, π
2 ])
f ∼DP
Ef ∼DP [f ]
Vf ∼DP [f ]
55 / 56


---

## 第81页

Summary & open problems
Summary:
▶Data-driven approaches for solving inverse problems in imaging.
▶Learning a regularization terms plays a major role.
▶Most natural framework is given through energy-based (variational) models.
▶Several possibilities for learning (GANs, bilevel, unrolling, maximum likelihood)
▶Leads to state-of-the-art results on a variety of problems
56 / 56


---

## 第82页

Summary & open problems
Summary:
▶Data-driven approaches for solving inverse problems in imaging.
▶Learning a regularization terms plays a major role.
▶Most natural framework is given through energy-based (variational) models.
▶Several possibilities for learning (GANs, bilevel, unrolling, maximum likelihood)
▶Leads to state-of-the-art results on a variety of problems
Open problems:
▶Performance gap between generative and discriminative (task-specific) learning.
▶Generalization bounds hard to derive and often not relevant in practice.
▶Sampling algorithms still lack behind in speed as compared to optimization algorithms.
▶Uncertainty quantification based on the posterior variance.
56 / 56


---

