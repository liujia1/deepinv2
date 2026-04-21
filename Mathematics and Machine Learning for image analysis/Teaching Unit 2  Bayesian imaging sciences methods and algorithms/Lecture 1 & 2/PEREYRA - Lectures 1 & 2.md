# PEREYRA - Lectures 1 & 2.pdf

## 第1页

Bayesian inference in imaging inverse problems - Part 1
Marcelo Pereyra
Heriot-Watt University & Maxwell Institute for Mathematical Sciences
June 2025, Bologna, Italy.
M. Pereyra
Bayesian imaging methods
0 / 83


---

## 第2页

Outline
1
Bayesian modelling in imaging inverse problems
2
Bayesian inference
3
Bayesian computation based on SDEs and proximal optimisation
4
Empirical Bayes MAP estimation with unknown regularisation parameters
5
Conclusion
M. Pereyra
Bayesian imaging methods
1 / 83


---

## 第3页

Forward imaging problem
True scene
Imaging device
Observed image
M. Pereyra
Bayesian imaging methods
2 / 83


---

## 第4页

Inverse imaging problem
True scene
Imaging device
Observed image
Restored image
M. Pereyra
Bayesian imaging methods
3 / 83


---

## 第5页

Imaging inverse problems
We are interested in an unknown image x⋆∈Rd.
We measure y ∈Y, related to x⋆by some mathematical model.
For example, in many imaging problems
y = Ax⋆+ w,
for some operator A that is poorly conditioned or rank deficient,
and an unknown perturbation or “noise” w.
The recovery of x⋆from y is usually not well posed. Additional
information is required in order to deliver meaningful solutions.
M. Pereyra
Bayesian imaging methods
4 / 83


---

## 第6页

Mathematical imaging frameworks
There are three main mathematical and computational frameworks for
inference in imaging inverse problems:
1
Mathematical analysis
2
Bayesian statistics.
3
Machine learning.
These frameworks have complementary strengths and weaknesses.
Our aim is to develop a unifying framework of theory, methods, and
algorithms that inherits the benefits of each approach.
M. Pereyra
Bayesian imaging methods
5 / 83


---

## 第7页

The Bayesian statistical approach
Model x⋆as a realisation of a r.v. x on Rd. Use the distribution of x
to regularise the problem and promote expected properties.
The observation y is a realisation of a r.v. (y∣x = x⋆).
Inferences about x⋆from y are derived from the joint distribution of
(x,y) - specified via the decomposition p(x,y) = p(y∣x)p(x).
M. Pereyra
Bayesian imaging methods
6 / 83


---

## 第8页

The Bayesian framework
The decomposition p(x,y) = p(y∣x)p(x) has two key ingredients:
The likelihood function: the conditional distribution p(y∣x) that models
the data observation process (forward model).
The prior function: the marginal distribution p(x) = ∫p(x,y)dy that
models our knowledge about the solution x.
For example, for y = Ax + w, with w ∼N(0,σ2I), we have
y ∼N(Ax,σ2I),
or equivalently
p(y∣x) ∝exp{−∥y −Ax∥2
2/2σ2}.
M. Pereyra
Bayesian imaging methods
7 / 83


---

## 第9页

Prior distribution
For this tutorial, we assume a prior distribution of the form:
p(x) =
1
Z(θ)e−θ⊺ψ(x)1Ω(x),
for some statistic ψ ∶Rd →Rm, θ ∈Rm, and constraint set Ω⊂Rd.
Often ψ and Ωare convex on Rd and p(x) is log-concave.
The normalising constant Z(θ) is given by
Z(θ) = ∫Ωe−θ⊺ψ(x)dx ,
so ∫Ωp(x)dx = 1. This will play a key role in model selection techniques.
The statistic ψ can be assumption-driven (e.g., a sparsity promoting
norm), purely data-driven, or a combination of both.
M. Pereyra
Bayesian imaging methods
8 / 83


---

## 第10页

Posterior distribution
We base our inferences on the posterior distribution p(x∣y).
We derive p(x∣y) from the likelihood p(y∣x) and the prior p(x) by using
p(x∣y) = p(y∣x)p(x)
p(y)
where p(y) = ∫p(y∣x)p(x)dx measures model-fit-to-data.
The conditional p(x∣y) models our beliefs about x after observing y = y.
In this first tutorial, we consider that p(x∣y) is log-concave; i.e.,
p(x∣y) = exp{−ϕ(x)}/∫exp{−ϕ(x)}dx ,
where ϕ(x) is a convex function on Rd.
M. Pereyra
Bayesian imaging methods
9 / 83


---

## 第11页

Illustrative example: astronomical image reconstruction
Recover x ∈Rd from low-dimensional degraded observation
y = MFx + w,
where F is the continuous Fourier transform, M ∈Cm×d is a measurement
operator, Ψ is a wavelet basis, and w ∼N(0,σ2Im). We use the model
p(x∣y) ∝exp(−∥y −MFx∥2/2σ2 −θ∥Ψx∥1)1Rn+(x).
(1)
x⋆
y ∼N(MFx⋆,σ2I)
y
Figure: Radio-interferometric measurements of the W28 supernova.
M. Pereyra
Bayesian imaging methods
10 / 83


---

## 第12页

Maximum-a-posteriori (MAP) estimation
The predominant Bayesian approach in imaging is MAP estimation
ˆxMAP = argmax
x∈Rd
p(x∣y),
= argmin
x∈Rd
ϕ(x).
(2)
When p(x∣y) is log-concave, ˆxMAP is a convex optimisation problem. We
are usually able to solve convex problems very efficiently (see Chambolle
and Pock (2016))..
See, e.g., Chambolle and Pock (2016) for more details.
M. Pereyra
Bayesian imaging methods
11 / 83


---

## 第13页

MAP estimation by proximal optimisation
To compute ˆxMAP we could use a proximal splitting algorithm. Let
f (x) = ∥y −MFx∥2/2σ2 ,
and
g(x) = θ∥Ψx∥1 + −log 1Rn+(x),
where f and g are l.s.c. convex on Rd, and f is Lf -Lipschitz differentiable.
For example, we could use a proximal gradient iteration
xm+1 = prox
L−1
f
g {xm + L−1
f ∇f (xm)},
converges to ˆxMAP at rate O(1/m), with poss. acceleration to O(1/m2).
Definition For λ > 0, the λ-proximal operator of a convex l.s.c. function g
is defined as (Moreau, 1962)
proxλ
g(x) ≜argmin
u∈RN
g(u) + 1
2λ∣∣u −x∣∣2.
M. Pereyra
Bayesian imaging methods
12 / 83


---

## 第14页

Illustrative example: astronomical image reconstruction
Recover x ∈Rd from low-dimensional degraded observation
y = MFx + w,
where F is the continuous Fourier transform, M ∈Cm×d is a measurement
operator, Ψ is a wavelet basis, and w ∼N(0,σ2Im). We use the model
p(x∣y) ∝exp(−∥y −MFx∥2/2σ2 −θ∥Ψx∥1)1Rn+(x).
(3)
y
ˆxMAP
Figure: Radio-interferometric image reconstruction of the W28 supernova.
M. Pereyra
Bayesian imaging methods
13 / 83


---

## 第15页

Summary
Modern convex optimisation can compute ˆx very efficiently...
With parallelised and distributed algorithms...
With theoretical convergence guarantees..
And GPU implementations...
With data-driven flavours based on input convex neural networks..
Also non-convex extensions...
So the problem is quite solved, right?
M. Pereyra
Bayesian imaging methods
14 / 83


---

## 第16页

Not really...
M. Pereyra
Bayesian imaging methods
15 / 83


---

## 第17页

Elephant 1: what is the uncertainty about ˆx?
∣y∣
ˆxMAP
How confident are we about all these structures in the image?
What is the error in their intensity, position, spectral properties?
Using ˆxMAP to derive physical quantities? what error bars should we put...
M. Pereyra
Bayesian imaging methods
16 / 83


---

## 第18页

Illustrative example: magnetic resonance imaging
We use very similar techniques to produce magnetic resonance images...
ˆx
ˆx (zoom)
Figure: Magnetic resonance imaging of brain lession.
How can we quantify our uncertainty about the brain lesion in the image?
M. Pereyra
Bayesian imaging methods
17 / 83


---

## 第19页

Illustrative example: magnetic resonance imaging
What about this other solution to the problem, with no lesion?
ˆx′
ˆx′ (zoom)
Figure: Magnetic resonance imaging of brain lession.
Do we have any arguments to reject this solution?
M. Pereyra
Bayesian imaging methods
18 / 83


---

## 第20页

Elephant 1: what is the uncertainty about ˆx?
Another example related to sparse super-resolution in live-cell microscopy
y
ˆxMAP
ˆxMAP (zoom)
Figure: Live-cell microscopy data (Zhu et al., 2012).
The image is sharpened to enhance molecule position measurements, but
what is the precision of the procedure?
M. Pereyra
Bayesian imaging methods
19 / 83


---

## 第21页

Elephant 2: multiple competing models
Two imaging scientists often formulate different models/cost functions to
recover x
ˆx1 = argmin
x∈Rd
∥y −A1x∥2
2 + θ1h1(x),
ˆx2 = argmin
x∈Rd
∥y −A2x∥2
2 + θ2h2(x),
(4)
How can we compare them without ground truth available?
Can we use several models simultaneously?
M. Pereyra
Bayesian imaging methods
20 / 83


---

## 第22页

Elephant 3: partially unknown models
Some of the model parameters might also be unknown; e.g., θ ∈R+ in
ˆx1 = argmin
x∈Rd
∥y −A1x∥2
2 + θh1(x).
(5)
Then θ parametrises a class of models for y →x.
How can we select θ without using ground truth?
Could we use all models simultaneously?
M. Pereyra
Bayesian imaging methods
21 / 83


---

## 第23页

Outline
1
Bayesian modelling in imaging inverse problems
2
Bayesian inference
3
Bayesian computation based on SDEs and proximal optimisation
4
Empirical Bayes MAP estimation with unknown regularisation parameters
5
Conclusion
M. Pereyra
Bayesian imaging methods
22 / 83


---

## 第24页

Bayesian decision theory
Given the following elements defining a decision problem:
1 Decision space ∆
2 Loss function L(δ,x) ∶∆× Rd →R quantifying the loss (or profit)
related to taking action δ ∈∆when the truth is x ∈Rd.
3 A model p(x) representing probabilities for x.
What is the optimal decision δ∗∈∆when x is unknown?
M. Pereyra
Bayesian imaging methods
23 / 83


---

## 第25页

Bayesian decision theory
Given the following elements defining a decision problem:
1 Decision space ∆
2 Loss function L(δ,x) ∶∆× Rd →R quantifying the loss (or profit)
related to taking action δ ∈∆when the truth is x ∈Rd.
3 A probability model p(x) representing knowledge about x.
According to Bayesian decision theory (Robert, 2001), the optimal decision
under uncertainty is
δ∗= argmin
δ∈∆
E{L(δ,x)∣y} = argmin
δ∈∆
∫L(δ,x)p(x)dx.
M. Pereyra
Bayesian imaging methods
24 / 83


---

## 第26页

Bayesian point estimators
Bayesian point estimators arise from the decision ”what point ˆx ∈Rd
summarises x∣y best?”. The optimal decision under uncertainty is
ˆxL = argmin
u∈Rd
E{L(u,x)∣y} = argmin
u∈Rd
∫L(u,x)p(x∣y)dx
where the loss L(u,x) measures the “dissimilarity” between u and x.
General desiderata:
1 L(u,x) ≥0, ∀u,x ∈Rd ,
2 L(u,x) = 0 ⇐⇒u = x ,
3 L strictly convex w.r.t. its first argument (for estimator uniqueness).
M. Pereyra
Bayesian imaging methods
25 / 83


---

## 第27页

Bayesian point estimators - MMSE estimation
Example: the squared Euclidean distance L(u,x) = ∥u −x∥2 defines the
so-called minimum mean squared error estimator.
ˆxMMSE = argmin
u∈Rd
∫∥u −x∥2
2 p(x∣y)dx .
By differentiating w.r.t. to u and equating to zero we obtain that
∫(ˆxMMSE −x)p(x∣y)dx = 0 Ô⇒ˆxMMSE ∫p(x∣y)dx = ∫xp(x∣y)dx ,
hence ˆxMMSE = E{x∣y} (recall that ∫p(x∣y)dx = 1).
M. Pereyra
Bayesian imaging methods
26 / 83


---

## 第28页

What about the MAP estimator?
Assume that p(x∣y) ∝exp{−ϕy(x)} is log-concave, i.e., ϕy convex on Rd.
The Bayesian estimator that minimises the Bregman divergence loss
L(u,x) = Dϕ(u,x) ≜ϕ(u) −ϕ(x) −∇ϕ(x)(u −x), is the MAP estimator
ˆxL = argmin
u∈Rd
E{L(u,x)∣y} = argmin
u∈Rd
∫Dϕ(u,x)p(x∣y)dx
= argmax
x∈Rd
p(x∣y),
= argmin
x∈Rd
−ϕy(x),
= ˆxMAP
(6)
Interestingly, the “dual” estimator is ˆxMMSE, i.e.,
ˆxL = argmin
u∈Rd
E{L(x,x)∣y} = argmin
u∈Rd
∫Dϕ(x,u))p(x∣y)dx
= ˆxMMSE
(7)
See Pereyra (2019) for proof and details.
M. Pereyra
Bayesian imaging methods
27 / 83


---

## 第29页

Posterior credible regions
Where does the posterior probability mass of x lie?
A set Cα is a posterior credible region of confidence level (1 −α)% if
P[x ∈Cα∣y] = 1 −α.
For any α ∈(0,1) there are infinitely many regions of the parameter space
that verify this property.
The highest posterior density (HPD) region is decision-theoretically
optimal in a compactness sense
C ∗
α = {x ∶ϕ(x) ≤γα}
with γα ∈R chosen such that ∫C ∗
α p(x∣y)dx = 1 −α holds.
M. Pereyra
Bayesian imaging methods
28 / 83


---

## 第30页

Hypothesis testing
Hypothesis test split the solution space in two meaningful regions, e.g.,
H0 ∶x ∈S
H1 ∶x /∈S
where S ⊂Rd contains all solutions with some characteristic of interest.
We can then assess the degree of support for H0 vs. H1 by computing
P(H0∣y) = ∫S p(x∣y)dx,
P(H1∣y) = 1 −P(H0∣y).
We can also reject H0 in favour of H1 with significance α ∈[0,1] if
P(H0∣y) ≤α.
M. Pereyra
Bayesian imaging methods
29 / 83


---

## 第31页

Bayesian model selection
The Bayesian framework provides theory for comparing models objectively.
Given K alternative models {Mj}K
j=1 with posterior densities
Mj ∶
pj(x∣y) = pj(y∣x)pj(x))/pj(y),
we compute the (marginal) posterior probability of each model, i.e.,
p(Mj∣y) ∝p(y∣Mj)p(Mj)
(8)
where p(y∣Mj) ≜pj(y) = ∫pj(y∣x)pj(x)dx measures model-fit-to-data.
We then select for our inferences the “best” model, i.e.,
M∗= argmax
j∈{1,...,K}
p(Mj∣y).
M. Pereyra
Bayesian imaging methods
30 / 83


---

## 第32页

Bayesian model calibration
Alternatively, given a continuous class of models {Mθ,θ ∈θ} with
Mθ ∶
p(x∣y,θ) = p(y∣x,θ)p(x∣θ)
p(y∣θ)
,
we compute the (marginal) posterior
p(θ∣y) = p(y∣θ)p(θ)/p(y)
(9)
where p(y∣θ) = ∫p(y∣x,θ)p(x∣θ)dx measures model-fit-to-data.
We then calibrate our model with the “best” value of θ, i.e.,
ˆθMAP = argmax
θ∈Θ
p(θ∣y).
M. Pereyra
Bayesian imaging methods
31 / 83


---

## 第33页

Bayesian model averaging
We can also use all models simultaneously!
Given K alternative models {Mj}K
j=1 with posterior densities
Mj ∶
pj(x∣y) = pj(y∣x)pj(x))/pj(y),
we marginalise w.r.t. the model selector j, i.e.,
p(x∣y) =
K
∑
j=1
p(x,Mj∣y) =
K
∑
j=1
p(x∣y,Mj)p(Mj∣y)
(10)
where the posterior probabilities p(Mj∣y) control the relative importance
of each model.
M. Pereyra
Bayesian imaging methods
32 / 83


---

## 第34页

Bayesian model calibration
Similarly, given a continuous class of models {Mθ,θ ∈Θ} with
Mθ ∶
p(x∣y,θ) = p(y∣x,θ)p(x∣θ)
p(y∣θ)
,
and a prior p(θ), we marginalise θ
p(x∣y) = ∫Θ p(x,θ∣y)dθ,
= ∫Θ p(x∣y,θ)p(θ∣y)dθ
where again p(θ∣y) controls the relative contribution of each model.
M. Pereyra
Bayesian imaging methods
33 / 83


---

## 第35页

My interpretation of log-concave prior distributions
Log-concave priors regularise the inverse problem by promoting solutions
for which ψ(x) is close to its expectation E(ψ∣θ), controlled by θ ∈Rp.
Formally, when ψ is convex we have concentration of probability mass on
the typical set (see Bobkov and Madiman (2011))
P{∥ψ(x) −E(ψ∣θ)∥> η∣θ} < 3exp{−η2d/16},
∀η ∈(0,2)
(11)
Moreover, by differentiating Z(θ) and using Leibniz integral rule
E(ψ(x)∣θ) = ∫Ωψ(x)p(x)dx = −∇θ log Z(θ),
(12)
hence p(x∣θ) softly constrains ψ(x) ≈−∇θ log Z(θ) when d is large.
Z(θ) is strongly log-concave, hence ∇θ log Z(θ) spans Rp (think duality).
M. Pereyra
Bayesian imaging methods
34 / 83


---

## 第36页

My interpretation of log-concave prior distributions
For example, priors of the form
p(x) ∝e−θ∥Bx∥† ,
for some basis or dictionary W ∈Rd×p and norm ∥⋅∥†, are encoding
E(∥Bx∥†∣θ) = d
θ .
See Pereyra et al. (2015); Fernandez-Vidal and Pereyra (2018) for more
details and other examples.
M. Pereyra
Bayesian imaging methods
35 / 83


---

## 第37页

Summary
The Bayesian statistical paradigm provides a power mathematical
framework to solve imaging problems...
It allows deriving optimal estimators for x..
As well as quantifying the uncertainty in the solutions delivered...
It supports hypothesis tests to inform decisions and conclusions...
And allows operating with partially unknown models...
And with several competing models...
So the problem is quite solved, right?
M. Pereyra
Bayesian imaging methods
36 / 83


---

## 第38页

Not really...
How do we compute all these probabilities?
M. Pereyra
Bayesian imaging methods
37 / 83


---

## 第39页

Outline
1
Bayesian modelling in imaging inverse problems
2
Bayesian inference
3
Bayesian computation based on SDEs and proximal optimisation
4
Empirical Bayes MAP estimation with unknown regularisation parameters
5
Conclusion
M. Pereyra
Bayesian imaging methods
38 / 83


---

## 第40页

Inference by Markov chain Monte Carlo integration
Monte Carlo integration
Given a set of samples X1,...,XM distributed according to p(x∣y), we
approximate posterior expectations and probabilities
1
M
M
∑
m=1
h(Xm) →E{h(x)∣y},
as M →∞
Markov chain Monte Carlo:
Construct a Markov kernel Xm+1∣Xm ∼K(⋅∣Xm) such that the Markov
chain X1,...,XM has p(x∣y) as stationary distribution.
MCMC simulation in high-dimensional spaces is very challenging.
M. Pereyra
Bayesian imaging methods
39 / 83


---

## 第41页

Unadjusted Langevin algorithm
Suppose for now that p(x∣y) ∈C1. Then, we can generate samples by
mimicking a Langevin diffusion process that converges to p(x∣y) as t →∞,
X ∶
dX t = ∇log p (X t∣y)dt +
√
2dWt,
0 ≤t ≤T,
X(0) = x0.
where W is the Brownian motion on Rd.
Because solving X t exactly is generally not possible, we use an Euler
Maruyama approximation and obtain the “unadjusted Langevin algorithm”
ULA ∶Xm+1 = Xm + δ∇log p(Xm∣y) +
√
2δZm+1, Zm+1 ∼N(0,In)
ULA is remarkably efficient when p(x∣y) is sufficiently regular.
Unfortunately, imaging models often violate these regularity conditions.
M. Pereyra
Bayesian imaging methods
40 / 83


---

## 第42页

Non-smooth models
Without loss of generality, suppose that
p(x∣y) ∝exp{−f (x) −g(x)}
(13)
where f (x) and g(x) are l.s.c. convex functions from Rd →(−∞,+∞], f
is Lf -Lipschitz differentiable, and g ∉C1.
For example,
f (x) =
1
2σ2 ∥y −Ax∥2
2,
g(x) = α∥Bx∥† + 1S(x),
for some linear operators A, B, norm ∥⋅∥†, and convex set S.
Unfortunately, such non-models are beyond the scope of ULA.
Idea: Regularise p(x∣y) to enable efficient Langevin sampling.
M. Pereyra
Bayesian imaging methods
41 / 83


---

## 第43页

Approximation of p(x∣y)
Moreau-Yoshida approximation of p(x∣y) (Pereyra, 2015):
Let λ > 0. We propose to approximate p(x∣y) with the density
pλ(x∣y) =
exp[−f (x) −gλ(x)]
∫Rd exp[−f (x) −gλ(x)]dx ,
where gλ is the Moreau-Yoshida envelope of g given by
gλ(x) = inf
u∈Rd{g(u) + (2λ)−1∥u −x∥2
2},
and where λ controls the approximation error involved.
M. Pereyra
Bayesian imaging methods
42 / 83


---

## 第44页

Moreau-Yoshida approximations
Key properties (Pereyra, 2015; Durmus et al., 2018):
1 ∀λ > 0, pλ defines a proper density of a probability measure on Rd.
2 Convexity and differentiability:
pλ is log-concave on Rd.
pλ ∈C1 even if p not differentiable, with
∇log pλ(x∣y) = −∇f (x) + {proxλ
g (x) −x}/λ,
and proxλ
g (x) = argminu∈RN g(u) + 1
2λ∣∣u −x∣∣2.
∇log pλ is Lipchitz continuous with constant L ≤Lf + λ−1.
3 Approximation error between pλ(x∣y) and p(x∣y):
limλ→0 ∥pλ −p∥TV = 0.
If g is Lg-Lipchitz, then ∥pλ −p∥TV ≤λL2
g.
M. Pereyra
Bayesian imaging methods
43 / 83


---

## 第45页

Illustration
Examples of Moreau-Yoshida approximations:
p(x) ∝exp(−∣x∣)
p(x) ∝exp(−x4)
p(x) ∝1[−0.5,0.5](x)
Figure: True densities (solid blue) and approximations (dashed red).
M. Pereyra
Bayesian imaging methods
44 / 83


---

## 第46页

Proximal ULA
We approximate X with the “regularised” auxiliary Langevin diffusion
X λ ∶
dX λ
t = ∇log pλ (X λ
t ∣y)dt +
√
2dWt,
0 ≤t ≤T,
X λ(0) = x0,
which targets pλ(x∣y). Remark: we can make X λ arbitrarily close to X.
Finally, an Euler Maruyama discretisation of X λ leads to the
(Moreau-Yoshida regularised) proximal ULA
MYULA ∶
Xm+1 = (1 −δ
λ)Xm −δ∇f {Xm} + δ
λ proxλ
g{Xm} +
√
2δZm+1,
where we used that ∇gλ(x) = {x −proxλ
g(x)}/λ.
M. Pereyra
Bayesian imaging methods
45 / 83


---

## 第47页

Convergence results
Non-asymptotic estimation error bound
Theorem 3.1 (Durmus et al. (2018))
Let δmax
λ
= (L1 + 1/λ)−1. Assume that g is Lipchitz continuous. Then,
there exist δϵ ∈(0,δmax
λ
] and Mϵ ∈N such that ∀δ < δϵ and ∀M ≥Mϵ
∥δx0QM
δ −p∥TV < ϵ + λL2
g,
where QM
δ
is the kernel assoc. with M iterations of MYULA with step δ.
Note 1: δϵ and Mϵ are explicit and tractable. If f + g is strongly convex
outside some ball, then Mϵ scales with order O(d log(d)). See Durmus
et al. (2018) for other convergence results.
M. Pereyra
Bayesian imaging methods
46 / 83


---

## 第48页

Illustrative example 1
Three toy models:
p(x) ∝exp(−∣x∣)
p(x) ∝exp(−x4)
p(x) ∝1[−0.5,0.5](x)
Figure: True densities (blue) and MC approximations (red histogram).
M. Pereyra
Bayesian imaging methods
47 / 83


---

## 第49页

Illustrative example 2: radio-interferometric imaging
Astro-imaging experiment with redundant wavelet frame (Cai et al., 2017).
ˆxpenMLE (y)
ˆxMMSE = E(x∣y)
credible intervals (scale 10 × 10)
ˆxpenMLE (y)
ˆxMMSE = E(x∣y)
credible intervals (scale 10 × 10)
3C2888 and M31 radio galaxies (size 256 × 256 pixels).
M. Pereyra
Bayesian imaging methods
48 / 83


---

## 第50页

Bayesian Uncertainty quantification
Where does the posterior probability mass of x lie?
A set Cα is a posterior credible region of confidence level (1 −α)% if
P[x ∈Cα∣y] = 1 −α.
The highest posterior density (HPD) region is decision-theoretically
optimal (Robert, 2001)
C ∗
α = {x ∶ϕ(x) ≤γα}
with γα ∈R chosen such that ∫C ∗
α p(x∣y)dx = 1 −α holds.
Given a set of samples X1,...,XM distributed according to p(x∣y),
we can estimate γα from (sample) quantiles of ϕ(X1),...,ϕ(XM).
Alternatively, we can compute a bound γα ≤˜γα(ˆxMAP) analytically by
using probability concentration results. See Pereyra (2016).
M. Pereyra
Bayesian imaging methods
49 / 83


---

## 第51页

Bayesian Uncertainty quantification
Bayesian hypothesis test for specific image structures (e.g., lesions)
H0 ∶The structure of interest is ABSENT in the true image
H1 ∶The structure of interest is PRESENT in the true image
The null hypothesis H0 is rejected with significance α if
P(H0∣y) ≤α.
Theorem (Repetti et al., 2018)
Let S denote the region of Rd associated with H0, containing all images
without the structure of interest. Then
S ∩Cα = ∅Ô⇒P(H0∣y) ≤α .
If in addition S is convex, then checking S ∩Cα = ∅is a convex problem
min
¯x, x∈Rd ∥¯x −x∥2
2
s.t.
¯x ∈Cα ,
x ∈S .
M. Pereyra
Bayesian imaging methods
50 / 83


---

## 第52页

Uncertainty quantification in MRI imaging
ˆxMAP
¯x ∈̃
C0.01
x ∈S
ˆxMAP (zoom)
¯x ∈̃
C0.01 (zoom)
x ∈S (zoom)
MRI experiment: test images ¯x = x, hence we fail to reject H0 and conclude that
there is little evidence to support the observed structure.
M. Pereyra
Bayesian imaging methods
51 / 83


---

## 第53页

Uncertainty quantification in MRI imaging
ˆxMAP
¯x ∈C0.01
x ∈S0
ˆxMAP (zoom)
¯x ∈C0.01 (zoom)
x ∈S0 (zoom)
MRI experiment: test images ¯x ≠x, hence we reject H0 and conclude that there is
significant evidence in favour of the observed structure.
M. Pereyra
Bayesian imaging methods
52 / 83


---

## 第54页

Uncertainty quantification in radio-interferometric imaging
Quantification of minimum energy of different energy structures, at level
(1 −α) = 0.99, as the number of measurements T = dim(y)/2 increases.
∣y∣
ˆxMAP (T = 200)
ρα, energy ratio preserved at α = 0.01
Figure: Analysis of 3 structures in the W28 supernova RI image.
Note: energy ratio calculated as
ρα =
∥¯x −x∥2
∥xMAP −˜xMAP∥2
where ¯x,x are computed with α = 0.01, and ˜xMAP is a modified version of xMAP
where the structure of interest has been carefully removed from the image.
M. Pereyra
Bayesian imaging methods
53 / 83


---

## 第55页

Bayesian Model Selection
The Bayesian framework provides theory for comparing models objectively.
Given K alternative models {Mj}K
j=1 with posterior densities
Mj ∶
pj(x∣y) = pj(y∣x)pj(x))/pj(y),
we compute the (marginal) posterior probability of each model, i.e.,
p(Mj∣y) ∝p(y∣Mj)p(Mj)
(14)
where p(y∣Mj) ≜pj(y) = ∫pj(y∣x)pj(x)dx measures model-fit-to-data.
We then select for our inferences the “best” model, i.e.,
M∗= argmax
j∈{1,...,K}
p(Mj∣y).
M. Pereyra
Bayesian imaging methods
54 / 83


---

## 第56页

Experiment setup
We degrade the Boat image of size 256 × 256 pixels with a 5 × 5 uniform
blur operator A∗and Gaussian noise w ∼N(0,σ2IN) with σ = 0.5.
y = A∗x + w
We consider four alternative models to estimate x, given by
Mj ∶
pj(x∣y) ∝exp[−(∥y −Ajx∥2/2σ2) −βjϕj(x)]
(15)
with fixed hyper-parameters σ and β, and where:
M1: A1 is the correct blur operator and ϕj(x) = TV (x).
M2: A2 is a mildly misspecified blur operator and ϕj(x) = TV (x).
M3: A3 is the correct blur operator and ϕj(x) = ∥Ψx∥1.
M4: A4 is a mildly misspecified blur operator and ϕj(x) = ∥Ψx∥1.
where Ψ is a wavelet frame and TV (x) = ∥∇dx∥1−2 is the total-variation
pseudo-norm. The βj are adjusted automatically (see model calibration).
M. Pereyra
Bayesian imaging methods
55 / 83


---

## 第57页

Monte Carlo strategy
To perform model selection we use MYULA to approximate the posterior
probabilities p(Mj∣y) for j = 1,2,3,4 by Monte Carlo integration.
For each model we generate n = 105 samples {X j
k}n
k=1 ∼p(x∣y,Mj) and
use the truncated harmonic mean estimator
p(y∣Mj) ≈(
n
∑
k=1
1S⋆(X M
k )
p(X M
k ,y∣Mj))
−1
vol(S⋆),
j = {1,2,3,4}
(16)
where S⋆is a union of HPDs of p(x∣y,Mj), also estimated from {X j
k}n
k=1.
Computing time approx. 30 minutes per model.
MYULA can also be used within a nested sampling scheme that provides a
faster and more reliable estimation of p(y∣Mj). See Cai et al. (2022).
M. Pereyra
Bayesian imaging methods
56 / 83


---

## 第58页

Numerical results
We obtain that p(M1∣y) ≈0.68 and p(M3∣y) ≈0.27 with the correct blur
are the best models, p(M2∣y) < 0.05 and p(M4∣y) < 0.01 perform poorly.
y
M1
ˆxMAP (PSNR 34.1dB)
p(M1∣y) ≈0.68
M3
ˆxMAP (PSNR 32.9dB)
p(M3∣y) ≈0.27
Figure: MAP estimation results for the Boat image deblurring experiment. (Note:
error w.r.t. “exact” probabilities from Px-MALA approx. 0.5%.)
M. Pereyra
Bayesian imaging methods
57 / 83


---

## 第59页

Numerical results
MYULA and Px-MALA efficiency comparison:
(a)
(b)
Figure: (a) Convergence of the chains to the typical set of x∣y under model M1,
(b) chain autocorrelation function (ACF).)
M. Pereyra
Bayesian imaging methods
58 / 83


---

## 第60页

Warning: I rarely use MYULA (or any other ULA)!
ULA samplers are based on a simple Euler-Maruyama approximation of the
Langevin diffusion that behaves similarly to gradient descent optimisation.
These samplers are useful for introducing ideas, however, they have been
superseded by accelerated samplers, see, e.g., Pereyra et al. (2020).
M. Pereyra
Bayesian imaging methods
59 / 83


---

## 第61页

Outline
1
Bayesian modelling in imaging inverse problems
2
Bayesian inference
3
Bayesian computation based on SDEs and proximal optimisation
4
Empirical Bayes MAP estimation with unknown regularisation parameters
5
Conclusion
M. Pereyra
Bayesian imaging methods
60 / 83


---

## 第62页

Problem statement
Consider the class of Bayesian models
p(x∣y,θ) = p(y∣x)p(x∣θ)
p(y∣θ)
,
parametrised by a regularisation parameter θ ∈Θ. For example,
p(x∣θ) =
1
Z(θ) exp{−θψ(x)},
p(y∣x) ∝exp{−fy(x)},
with fy and ψ convex l.s.c. functions, and fy L-Lipschitz differentiable.
We assume that p(x∣θ) is proper, i.e.,
Z(θ) = ∫Rd exp{−θψ(x)}dx < ∞,
with Z(θ) unknown and generally intractable.
M. Pereyra
Bayesian imaging methods
61 / 83


---

## 第63页

Maximum-a-posteriori estimation
Recall that when θ is fixed, the posterior p(x∣y,θ) is log-concave and
ˆxMAP = argmin
x∈Rd
fy(x) + θψ(x)
is a convex optimisation problem that can be often solved efficiently.
For example, one can consider the proximal gradient algorithm
xm+1 = proxL−1
ψ {xm + L−1∇fy(xm)}
to iteratively compute ˆxMAP.
However, θ is often unknown, significantly complicating the problem.
M. Pereyra
Bayesian imaging methods
62 / 83


---

## 第64页

Regularisation parameter MLE
We have two option to tackle θ. We first consider an empirical Bayes
approach based on the MLE
ˆθ = argmax
θ∈Θ
p(y∣θ),
= argmax
θ∈Θ
∫Rd p(y,x∣θ)dx ,
which we solve efficiently by using a stochastic gradient algorithm driven
by two proximal MCMC kernels (see Fernandez-Vidal and Pereyra (2018)).
Given ˆθ, we then straightforwardly compute
ˆxMAP = argmin
x∈Rd
fy(x) + ˆθψ(x).
(17)
M. Pereyra
Bayesian imaging methods
63 / 83


---

## 第65页

Projected gradient algorithm
Assume that Θ is convex, and that ˆθ is the only root of ∇θ log p(y∣θ) in Θ.
Then ˆθ is also the unique solution of the fixed-point equation
θ = PΘ [θ + δ∇θ log p(y∣θ)] .
where PΘ is the projection operator on Θ and δ > 0.
If ∇log p(y∣θ) was tractable, we could compute ˆθ iteratively by using
θ(t+1) = PΘ [θ(t) + δt∇θ log p(y∣θ(t))],
with sequence δt = αt−β, α > 0, β ∈[1/2,1].
However, ∇log p(y∣θ) is “doubly” intractable...
M. Pereyra
Bayesian imaging methods
64 / 83


---

## 第66页

Stochastic projected gradient algorithm
To circumvent the intractability of ∇θ log p(y∣θ) we use Fisher’s identity
∇θ log p(y∣θ) = Ex∣y,θ{∇θ log p(x,y∣θ)},
= −Ex∣y,θ{ψ(x) + ∇θ log Z(θ)},
together with the identity
∇θ log Z(θ) = −Ex∣θ{ψ(x)},
to obtain ∇θ log p(y∣θ) = Ex∣θ{ψ(x)} −Ex∣y,θ{ψ(x)}.
This leads to the equivalent fixed-point equation
θ = PΘ (θ + δEx∣θ{ψ(x)} −δEx∣y,θ{ψ(x)}),
(18)
which we solve by using a stochastic approximation algorithm.
M. Pereyra
Bayesian imaging methods
65 / 83


---

## 第67页

SAPG algorithm driven by MCMC kernels
Initialisation x(0),u(0) ∈Rd, θ(0) ∈Θ, δt = δ0t−0.8.
for t = 0 to n
1. MCMC update x(t+1) ∼Mx∣y,θ(t)(⋅∣x(t)) targeting p(x∣y,θ(t))
2. MCMC update u(t+1) ∼Kx∣θ(t)(⋅∣u(t)) targeting p(x∣θ(t))
3. Stoch. grad. update
θ(t+1) = PΘ [θ(t) + δtψ(u(t+1)) −δtψ(x(t+1))].
end for
Output The iterates θ(t) →ˆθ as n →∞.
M. Pereyra
Bayesian imaging methods
66 / 83


---

## 第68页

SAPG algorithm driven MCMC kernels
Initialisation x(0),u(0) ∈Rd, θ(0) ∈Θ, δt = δ0t−0.8, λ = 1/L, γ = 1/4L.
for t = 0 to n
1. Coupled Proximal MCMC updates: generate z(t+1) ∼N(0,Id)
x(t+1) = (1 −γ
λ)x(t) −γ∇fy (x(t)) + γ
λproxθλ
ψ (x(t)) +
√
2γz(t+1) ,
u(t+1) = (1 −γ
λ)u(t) + γ
λproxθλ
ψ (u(t)) +
√
2γz(t+1) ,
2. Stochastic gradient update
θ(t+1) = PΘ [θ(t) + δtψ(u(t+1)) −δtψ(x(t+1))].
end for
Output Averaged estimator ¯θ = n−1 ∑n
t=1 θ(t+1) converges approx. to ˆθ.
M. Pereyra
Bayesian imaging methods
67 / 83


---

## 第69页

Illustrative example: hyperspectral unmixing
We seek to recover fractional abundances x from the mixed noisy
spectral signatures y measured for every pixel.
M. Pereyra
Bayesian imaging methods
68 / 83


---

## 第70页

Empirical Bayesian MAP estimation
We consider the prior of Iordache et al. (IEEE TGRS, 2012)
p(x∣θ) ∝exp{θTV TV (x) + θL1 ∥x∥1}
s.t.x ≥0,
and use the SAPG scheme to estimate θTV and θL1 by MMLE.
Figure: Evolution of the iterates associated to θTV and θL1.
M. Pereyra
Bayesian imaging methods
69 / 83


---

## 第71页

Empirical Bayesian MAP estimation
Given ˆθTV and ˆθL1, we compute the MAP solution by using the convex
optimisation algorithm SUNSAL:
M. Pereyra
Bayesian imaging methods
70 / 83


---

## 第72页

Hierarchical Bayesian treatment of unknown θ
Option 2: hierarchical Bayesian inference also allows estimating x without
specifying the value of θ.
We incorporate θ to the model by assigning it an hyper-prior p(θ).
The extended model is
p(x,θ∣y) = p(y∣x)p(x∣θ)p(θ)/p(y),
∝exp{−fy(x) −θψ(x) −log p(θ)}
Z(θ)
,
(19)
but Z(θ) = ∫Rd exp{−θψ(x)}dx is typically intractable!
If we had access to Z(θ) we could either estimate x and θ jointly, or
alternatively marginalise θ followed by MAP inference on x.
M. Pereyra
Bayesian imaging methods
71 / 83


---

## 第73页

Idea: Use MYULA to estimate E[ψ(x)∣θ] over a θ-grid, and then
approximate log Z(θ) by using the identity
d
dθ log Z(θ) = E[ψ(x)∣θ].
Figure: Monte Carlo approximations of E[ψ(x)∣θ]/d Vs θ for 4 widely used prior
distributions and for θ ∈[10−3,102]. Surprise: they all coincide!
M. Pereyra
Bayesian imaging methods
72 / 83


---

## 第74页

Main theoretical result
Definition[k-homogeneity] The regulariser ψ is a k-homogeneous function
if ∃k ∈R+ such that
ψ(ηx) = ηkψ(x),
∀x ∈Rd,∀η > 0.
(20)
Theorem[Pereyra et al. (2015)] Suppose that ψ, the sufficient statistic of
p(x∣θ), is k-homogenous. Then the normalisation factor has the form
Z(θ) = Z(1)θ−d/k,
with (generally intractable) constant Z(1) independent of θ.
Note: This result holds for all norms (e.g., ℓ1, ℓ2, total-variation, nuclear,
etc.), composite norms (e.g., ℓ1 −ℓ2), and compositions of norms with
linear operators (e.g., analysis terms of the form ∥Ψx∥1)!
M. Pereyra
Bayesian imaging methods
73 / 83


---

## 第75页

Marginal maximum-a-posteriori estimation of x
Knowledge of Z(θ) enables (for example) marginal MAP estimation of x
ˆx†
MAP = argmax
x∈Rd
∫
∞
0
p(x,θ∣y)dθ,
= argmin
x∈Rd
fy(x) + (d/k + α)log{ψ(x) + β},
(21)
where we have used the hyper-prior θ ∼Gamma(α,β).
We can compute ˆx† efficiently by majorisation-minimisation optimisation
x(t) = argmin
x∈Rd
fy(x) + θ(t−1)ψ(x),
θ(t) =
d/k + α
ψ(x(t)) + β .
(22)
which is also an expectation-maximisation algorithm.
M. Pereyra
Bayesian imaging methods
74 / 83


---

## 第76页

Illustrative example: hyperspectral unmixing
We use the hyperspectral unmixing example to develop an intuition for the
strengths and drawbacks of the EB and HB approaches.
EB performs remarkably well in low SNR (high noise) regimes, close to the
oracle performance. The benefits of HB kick-in in high SNR regimes.
M. Pereyra
Bayesian imaging methods
75 / 83


---

## 第77页

Illustrative example: image deconvolution with a TV prior
SNR=20dB
SNR=30
SNR=40
MSE
Time (min)
MSE
Time (min)
MSE
Time (min)
Best
23.29
21.39
19.06
Emp. Bayes
23.50
0.86
21.46
0.85
19.24
0.85
Hier. Bayes
25.07
0.58
22.84
1.27
19.84
3.27
Disc. Princp.
23.73
21.87
19.78
SUGAR
24.44
3.92
24.24
4.50
24.21
4.81
Original
Degraded
x
EB
x
HB
x
DP
x
SUG
X
X
X
SNR=20
SNR=30
SNR=40
X
Min MSE
Empirical B.
Disc. Prin.
Hierarchical B.
SUGAR
10-4
0.001
0.010
0.100
1
θ
20
30
40
50
60
70
MSE(θ)
Image:flinstones
M. Pereyra
Bayesian imaging methods
76 / 83


---

## 第78页

Denoising with Total Generalized Variation
We consider TGV 2
θ (u) =
inf
v∈BD(Ω) θ1 ∫Ω∣∇u −v∣+ θ2 ∫Ω∣ε(v)∣with k = 2.
Figure: Goldhill image (Original-Degraded-Estimated MAP), SNR=12dB.
M. Pereyra
Bayesian imaging methods
77 / 83


---

## 第79页

Denoising with Total Generalized Variation
M. Pereyra
Bayesian imaging methods
78 / 83


---

## 第80页

Evolution of θ through iterations starting from different initial values:
θinit=10
θinit=0.1
θinit=40
M. Pereyra
Bayesian imaging methods
79 / 83


---

## 第81页

Outline
1
Bayesian modelling in imaging inverse problems
2
Bayesian inference
3
Bayesian computation based on SDEs and proximal optimisation
4
Empirical Bayes MAP estimation with unknown regularisation parameters
5
Conclusion
M. Pereyra
Bayesian imaging methods
80 / 83


---

## 第82页

Concluding remarks
There are three main frameworks to solve imaging inverse problems -
mathematical analysis, Bayesian statistics, and machine learning -
with complementary strengths and drawbacks.
This tutorial focused on Bayesian imaging methods that leverage a
range of ideas and techniques from mathematical analysis to perform
computations efficiently.
We explored integrating modern stochastic and optimisation
approaches to construct proximal MCMC methods and stochastic
proximal gradient algorithms.
The tutorials of tomorrow will focus on methods at that also
incorporate a significant machine learning component.
M. Pereyra
Bayesian imaging methods
81 / 83


---

## 第83页

Bibliography:
Bobkov, S. and Madiman, M. (2011). Concentration of the information in data with
log-concave distributions. Ann. Probab., 39(4):1528–1543.
Cai, X., J.D., M., and M., P. (2022). Proximal nested sampling for high-dimensional
Bayesian model selection. Statistics and Computing, 37(87).
Cai, X., Pereyra, M., and McEwen, J. D. (2017). Uncertainty quantification for radio
interferometric imaging II: MAP estimation. ArXiv e-prints.
Chambolle, A. and Pock, T. (2016). An introduction to continuous optimization for
imaging. Acta Numerica, 25:161–319.
Durmus, A., Moulines, E., and Pereyra, M. (2018). Efficient Bayesian computation by
proximal Markov chain Monte Carlo: when Langevin meets Moreau. SIAM J. Imaging
Sci., 11(1):473–506.
Fernandez-Vidal, A. and Pereyra, M. (2018). Maximum likelihood estimation of
regularisation parameters. In Proc. IEEE ICIP 2018.
Moreau, J.-J. (1962). Fonctions convexes duales et points proximaux dans un espace
Hilbertien. C. R. Acad. Sci. Paris S´er. A Math., 255:2897–2899.
Pereyra, M. (2015). Proximal Markov chain Monte Carlo algorithms. Statistics and
Computing. open access paper, http://dx.doi.org/10.1007/s11222-015-9567-4.
Pereyra, M. (2016). Maximum-a-posteriori estimation with bayesian confidence regions.
SIAM J. Imaging Sci., 6(3):1665–1688.
M. Pereyra
Bayesian imaging methods
82 / 83


---

## 第84页

Pereyra, M. (2019). Revisiting maximum-a-posteriori estimation in log-concave models.
SIAM J. Imaging Sci., 12(1):650–670.
Pereyra, M., Bioucas-Dias, J., and Figueiredo, M. (2015). Maximum-a-posteriori
estimation with unknown regularisation parameters. In Proc. Europ. Signal Process.
Conf. (EUSIPCO) 2015.
Pereyra, M., Mieles, L. V., and Zygalakis, K. C. (2020). Accelerating proximal markov
chain monte carlo by using an explicit stabilized method. SIAM Journal on Imaging
Sciences, 13(2):905–935.
Repetti, A., Pereyra, M., and Wiaux, Y. (2018). Scalable Bayesian uncertainty
quantification in imaging inverse problems via convex optimisation. ArXiv e-prints.
Robert, C. P. (2001). The Bayesian Choice (second edition). Springer Verlag, New-York.
Zhu, L., Zhang, W., Elnatan, D., and Huang, B. (2012). Faster STORM using
compressed sensing. Nat. Meth., 9(7):721–723.
M. Pereyra
Bayesian imaging methods
83 / 83


---

