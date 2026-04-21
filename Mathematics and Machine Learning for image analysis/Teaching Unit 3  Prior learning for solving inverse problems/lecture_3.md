# lecture_3.pdf

## 第1页

ERASMUS+ INTERNATIONAL PHD SUMMER SCHOOL 2025
Mathematics and Machine Learning for image analysis
Lecture 3 - Learning Optimal Discretizations
Thomas Pock
Institute of Visual Computing
Graz University of Technology
University of Bologna, June 3-6 2025
1 / 47


---

## 第2页

Edges
▶Edges are among the most important features in images
▶Image understanding relies on abstract discontinuity information
▶Most successful image descriptors are based on intensity gradients
▶First layers in deep convolutional networks represent edge detectors
(a)
(b)
(c)
(d)
(e)
2 / 47


---

## 第3页

Edge statistics of natural images
▶Randomly extracted 15M image patches of
size 2 × 2 from a natural image data set.
▶Compute finite differences in horizontal and
vertical direction.
▶Yields a heavy tailed distribution ⇝most
gradients are zero ⇝sparse gradients.
(Du)1
(Du)2
(Dx)1
−1.0
−0.5
0.0
0.5
1.0
(Dx)2
−1.0
−0.5
0.0
0.5
1.0
−15
−10
−5
0
(Du)1
−1.0
−0.5
0.0
0.5
1.0
(Du)2
−1.0
−0.5
0.0
0.5
1.0
−15
−10
−5
0
3 / 47


---

## 第4页

The total variation
▶The log-statistics of Du looks like an upside-down ice-cream cone.
▶A simple fit to the negative log-statistics is given by the ℓ2 norm, leading to the total variation:
TV(u) = ∥Du∥2,1 =
X
i,j
p
(ui+1,j −ui,j)2 + (ui,j+1 −ui,j)2.
▶Has been introduced in [Rudin, Osher, Fatemi ’92], [Chambolle, Lions ’97], ...
−log p(|Du|2)
|Du|2
4 / 47


---

## 第5页

The ROF model
▶The discrete ROF model [Rudin, Osher, Fatemi ’92] is defined as the following minimization
problem
min
u λ ∥Du∥2,1 + 1
2 ∥u −g∥2 , λ > 0
▶Defines ”the” prototypical variational model in mathematical image processing.
▶Gives a good tradeoff between simplicity of the model and denoising quality.
▶Allows for discontinuities (edges) in the image.
▶It is a convex lower-semicontinuous function.
▶It also has a nice geometric interpretation ⇝minimal surfaces.
5 / 47


---

## 第6页

Advanced applications
(a) Denoising
(b) Deblurring
(c) MRI
(d) Motion
(e) Stereo
(f) Segmentation
6 / 47


---

## 第7页

Image inpainting
▶For most practical problems, the standard discrete total variation gives sufficiently good results.
▶However, on free discontinuity problems such as image inpainting, the standard discretization
yields strong artifacts.
Inpainting of straight discontinuities
7 / 47


---

## 第8页

Image inpainting
▶For most practical problems, the standard discrete total variation gives sufficiently good results.
▶However, on free discontinuity problems such as image inpainting, the standard discretization
yields strong artifacts.
80.366
26.937
86.226
27.024
80.967
23.615
22.899
23.299
81.726
27.021
86.223
26.989
80.357
23.733
23.190
23.609
Inpainting of straight discontinuities
7 / 47


---

## 第9页

Advanced free discontinuities problems
Convexification of the Mumford-Shah functional [P., Cremers, Bischof, Chambolle ’09]:
Convexification of Euler’s elastica [Chambolle, P. ’19]
(data from J. Weickert)
Here, the discretization can make a difference between “working” and “not working”.
8 / 47


---

## 第10页

Related work
Finding a good general discretization of the total variation is far from being trivial and hence many
approaches have been proposed:
▶Non-standard finite differences for anisotropic diffusion [Weickert, Welk, Wichert ’13]
▶Graph-based / MRFs / crystalline energies [Boykov, Kolmogorov ’03], [Chambolle ’05]
▶Upwind discretization [Chambolle, Levine, Lucier ’11]
▶Shannon TV [Abergel, Moisan ’17]
▶Conforming P1 finite elements [Bartels ’12]
▶Non-conforming P1 (Crouzeix-Raviart) finite elements [Chambolle, P. 18]
▶Duality based discretization using H(div)-conforming Raviart-Thomas (RT0) vector fields
[Herrmann, Herzog, Schmidt, Vidal, Wachsmuth ’18], [Caillaud, Chambolle ’20]
▶Approximate Raviart-Thomas [Hinterm¨uller, Rautenberg, Hahn ’14], [Condat ’17]
9 / 47


---

## 第11页

General setting
ui,j
ui,j+1
ui+1,j
p1
i+ 1
2 ,j
p2
i,j+ 1
2
▶We introduce the finite differences operator Du = (D1u, D2u) with
(
(D1u)i+ 1
2 ,j = ui+1,j −ui,j
i = 1, . . . , M −1, j = 1, . . . , N,
(D2u)i,j+ 1
2 = ui,j+1 −ui,j
i = 1, . . . , M, j = 1, . . . , N −1.
▶The total variation is defined via duality as
TV (u) := sup

⟨p, Du⟩Y : ∥F p∥Z
∗≤1
	
where p = (p1, p2) are the dual variables and F = (F 1, ..., F L) are convolutional interpolation
kernels defined as
(F lp)i,j =
(F l,1p1)i,j
(F l,2p2)i,j

=
 Pν
m,n=−ν ξl
m,np1
i+ 1
2 −m,j−n
Pν
m,n=−ν ηl
m,np2
i−m,j+ 1
2 −n
!
▶The primal form has the structure of a sparse coding problem
TV (u) =
min
q: F ∗q=Du ∥q∥Z,
where F ∗can be interpreted as a convolutional dictionary.
10 / 47


---

## 第12页

Example: Forward differences
ui,j
ui,j+1
ui+1,j
p1
i+ 1
2 ,j
p2
i,j+ 1
2
▶Interpolation kernels (Nearest neighbor interpolation):
(Fp)i,j =
 
p1
i+ 1
2 ,j
p2
i,j+ 1
2
!
.
Interpolation kernels F
▶The Z-norm is given by
∥z∥Z =
X
i,j
q
(z1
i+ 1
2 ,j)2 + (z2
i,j+ 1
2 )2
11 / 47


---

## 第13页

Example: Raviart-Thomas
ui,j
ui,j+1
ui+1,j
p1
i+ 1
2 ,j
p2
i,j+ 1
2
▶Interpolation kernels (Nearest neighbor interpolation):
(F 1p)i−1
2 ,j−1
2 =
 
p1
i−1
2 ,j
p2
i,j−1
2
!
, (F 2p)i−1
2 ,j+ 1
2 =
 
p1
i−1
2 ,j
p2
i,j+ 1
2
!
,
(F 3p)i+ 1
2 ,j−1
2 =
 
p1
i+ 1
2 ,j
p2
i,j−1
2
!
,
(F 4p)i+ 1
2 ,j+ 1
2 =
 
p1
i+ 1
2 ,j
p2
i,j+ 1
2
!
.
Interpolation kernels F
▶The Z-norm is given by
(z1, z2, z3, z4)

Z :=
X
i,j
|z1
i−1
2 ,j−1
2 |2 + |z2
i−1
2 ,j+ 1
2 |2 + |z3
i+ 1
2 ,j−1
2 |2 + |z4
i+ 1
2 ,j+ 1
2 |2
12 / 47


---

## 第14页

Example: Condat’s discretization
ui,j
ui,j+1
ui+1,j
p1
i+ 1
2 ,j
p2
i,j+ 1
2
▶Interpolation kernels (bilinear interpolation):
(F 1p)i,j =



p1
i−1
2 ,j+p1
i+ 1
2 ,j
2
p2
i,j−1
2
+p2
i,j+ 1
2
2


,
(F 2p)i+ 1
2 ,j =


p1
i+ 1
2 ,j
p2
i,j−1
2
+p2
i,j+ 1
2
+p2
i+1,j−1
2
+p2
i+1,j+ 1
2
4

,
(F 3p)i,j+ 1
2 =


p1
i−1
2 ,j+p1
i+ 1
2 ,j+p1
i−1
2 ,j+1+p1
i+ 1
2 ,j+1
4
p2
i,j+ 1
2


Interpolation kernels F
▶The Z-norm is given by
∥(z1, z2, z3)∥Z :=
X
i,j
|z1
i,j|2 + |z2
i+ 1
2 ,j|2 + |z3
i,j+ 1
2 |2
13 / 47


---

## 第15页

Comparison
Input
14 / 47


---

## 第16页

Comparison
80.366
26.937
86.226
27.024
80.967
23.615
22.899
23.299
81.726
27.021
86.223
26.989
80.357
23.733
23.190
23.609
Forward differences
14 / 47


---

## 第17页

Comparison
69.822
23.605
22.407
23.602
69.940
23.615
22.620
23.605
69.797
23.615
22.423
23.614
69.809
23.617
22.825
23.602
Raviart-Thomas
14 / 47


---

## 第18页

Comparison
71.509
40.505
36.030
41.699
71.537
41.714
35.853
40.273
71.465
40.055
35.890
38.027
71.447
39.842
51.409
41.768
Condat
14 / 47


---

## 第19页

Consistency result
▶We define a family of discrete total variations for pixels of size ε × ε:
TVε(u) = min

ε2∥q∥Zε : F ∗
εq = Dεu
	
= sup
n
ε2⟨p, Dεu⟩Yε : ∥F εp∥∗
Z ≤1
o
Theorem
Assume the supports and the weights of the convolutions defining F ε are uniformly bounded that is
X
m,n
ξl
m,n =
X
m,n
ηl
m,n = 1 ⇐⇒F l,1, F l,2 ∈CΣ=1.
Then TVε Γ-converges to
TV (u) :=
(
|Du|(Ω)
if u ∈BV (Ω) ,
+∞
else.
As long as the filter coefficients sum up to one and are uniformly bounded, we are having a consistent
discretization of the total variation ⇝learning.
15 / 47


---

## 第20页

Class of total variation minimization problems
▶We consider the following class of total variation minimization problems
min
Du=F ∗q λ ∥q∥Z + G(u, g),
with a saddle-point formulation
min
u,q max
p
⟨Du −F ∗q, p⟩+ λ ∥q∥Z + G(u, g)
▶Can be applied to a large class of inverse problems in imaging such as denoising, inpainting,
segmentation, ....
▶We need access to the proximal maps of ∥·∥Z and G(·, g).
16 / 47


---

## 第21页

Supervised learning
▶Assume we have given training data (gs, ts), s = 1, ..., S.
▶We consider the following bilevel optimization problem:
min
F
L(F ) + R(F ),
u∗
s ∈arg min
u,q max
p
⟨Du −F ∗q, p⟩+ λ ∥q∥Z + G(u, gs),
s = 1, . . . , S
▶L(F ) is a convex and differentiable loss function
L(F ) =
1
MNS
S
X
s=1
ℓ(u∗
s(F ), ts),
that measures the error between the targets ts and the solutions u∗
s, here ℓ(u, t) = 1
2 ∥u −t∥2
2.
▶R(F ) can be used to impose the constraints on the filters F .
R(F ) = δ(CΣ=1)L,2(F ) =
L
X
l=1
δCΣ=1(F l,1) + δCΣ=1(F l,2)
▶For gradient-based learning, we need to compute the derivatives of the loss function with respect
to the linear operator F .
17 / 47


---

## 第22页

Interlude: Derivatives of saddle-points
▶We consider the following class of saddle-point problems
min
x∈X max
y∈Y ⟨Kx, y⟩+ g(x) −f ∗(y),
with corresponding primal and dual problems
min
x∈X f(Kx) + g(x) ⇐⇒max
y∈Y −f ∗(y) −g∗(−K∗y)
▶We assume that the problem has, for a given linear operator K, a unique saddle point (ˆx, ˆy)
characterized by
(
Kˆx −∂f ∗(ˆy) ∋0
K∗ˆy + ∂g(ˆx) ∋0
▶Then we consider that we have given a convex loss function
L(K) = ℓ(ˆx(K), ˆy(K)).
▶We are interested in the gradient of the loss with respect to the linear operator K.
▶We will derive a formula based on a standard sensitivity analysis.
▶Denote by ˆxs = ˆx + sξs, and ˆys = ˆy + sηs the solution of the saddle-point problem perturbed by
a small variation K + sL, |s| ≪1 of the linear operator.
18 / 47


---

## 第23页

▶Substituting (ˆxs, ˆys) into the optimality system yields
(
Kˆx + s(Kξs + Lˆxs) −[∂f ∗(ˆy) + (
R s
0 D2f ∗(ˆy + tηs)dt)ηs] = 0,
K∗ˆy + s(K∗ηs + L∗ˆys) + [∂g(ˆx) + (
R s
0 D2g(ˆx + tξs)dt)ξs] = 0.
▶Again making use of the optimality condition and dividing by s gives
(
Kξs + Lˆxs −
  1
s
R s
0 D2f ∗(ˆy + tηs)dt

ηs = 0,
K∗ηs + L∗ˆys +
  1
s
R s
0 D2g(ˆx + tξs)dt

ξs = 0.
▶Passing to the limit s →0, one obtains the linear system in (ξ, η)
(
Kξ + Lˆx −D2f ∗(ˆy)η = 0,
K∗η + L∗ˆy + D2g(ˆx)ξ = 0.
⇐⇒
ξ
η

=
D2g(ˆx)
K∗
−K
D2f ∗(ˆy)
−1 −L∗ˆy
Lˆx

▶The directional derivative is then given by
L′(K; L) =

∇ℓ(ˆx, ˆy),
ξ
η

=
*
∇ℓ(ˆx, ˆy),
D2g(ˆx)
K∗
−K
D2f ∗(ˆy)
−1 −L∗ˆy
Lˆx
+
=
*D2g(ˆx)
K∗
−K
D2f ∗(ˆy)
−1
∇ℓ(ˆx, ˆy)
|
{z
}
=

−ˆ
X
ˆY


,
−L∗ˆy
Lˆx
+
=
D
ˆ
X, L∗ˆy
E
+
D
ˆY , Lˆx
E
,
where ˆ
X and ˆY being the adjoint variables.
19 / 47


---

## 第24页

▶Interestingly, the adjoint variables are themselves solutions of the quadratic saddle-point problem
min
X max
Y
⟨KX, Y ⟩+ 1
2

D2g(ˆx)X, X

−1
2

D2f ∗(ˆy)Y , Y

+

∇ℓ(ˆx, ˆy),
X
Y

▶This brings up the idea of running in parallel, a primary primal-dual algorithm solving the
lower-level problem and a secondary primal-dual algorithm that solves the adjoint saddle-point
problem.
▶Such algorithmic scheme is denoted in the AD literature as “piggyback” algorithm [Griewank,
Faure ’03].
▶Note that the secondary primal-dual algorithm depends on the solution of the primary primal-dual
algorithm and hence must be analyzed as an algorithm with (summable) errors.
▶The final gradient is then given by
L′(K; L) =
D
ˆ
X, L∗ˆy
E
+
D
ˆY , Lˆx
E
⇐⇒∇L(K) = ˆ
X ⊗ˆy + ˆx ⊗ˆY ,
which we usually compute via automatic differentiation of the scalar products, in order to respect
the structure and boundary conditions of the linear operator K (which can be complicated).
20 / 47


---

## 第25页

Piggyback primal-dual algorithm
▶The primary primal-dual algorithm is given by





xk+1 = (I + τ∇g))−1(xk −τK∗yk)
¯xk+1 = xk+1 + θ(xk+1 −xk)
yk+1 = (I + σ∇f ∗)−1(yk + σK¯xk+1).
▶The secondary primal-dual algorithm is given by





Xk+1 = ∇proxτg(xk −τK∗yk) · (Xk −τ(K∗Y k + ∇xℓ(xk, yk)))
¯
Xk+1 = Xk+1 + θ(Xk+1 −Xk)
Y k+1 = ∇proxσf∗(yk + σK¯xk+1) · (Y k + σ(K ¯
Xk+1 + ∇yℓ(xk, yk))),
where we have used the fact that ∇proxτg(x) = (I + τD2g(proxτg(x)))−1.
Theorem ([Bogensperger, Chambolle, P. ’22])
Assume that f ∗, g are strongly convex and f, g∗are locally C2,α then the piggyback primal-dual
algorithm converges linearly.
21 / 47


---

## 第26页

Learning for inpainting
▶We train on 64 images of size 64 × 64 with directions uniformly sampled between [0, 2π] and we
include random subpixel shifts.
▶We train on a training set and evaluate on a test set.
▶We experiment with different numbers of filters and different symmetry constraints for the filters.
(a) Input images gs
(b) Target images ts
22 / 47


---

## 第27页

Results
L = 2
L = 2 (s)
L = 3
L = 3 (s)
L = 4 (s)
L = 8 (s)
Data
FD
RT
CD
L = 2
L = 2 (s)
L = 3
L = 3 (s)
L = 4 (s)
L = 8 (s)
Train
135
195
6.69
1.26
1.22
1.19
1.27
0.85
0.77
Test
134
194
6.33
1.63
1.45
1.29
1.29
0.87
0.82
Table: 105 × the mean squared error (MSE) of handcrafted and learned filters evaluated on both the training
and test data.
Note that transpose symmetry is almost automatically learned!
23 / 47


---

## 第28页

Filter: L = 8 (s)
98.52
45.31
46.26
47.71
50.09
49.82
52.75
50.33
49.95
50.71
47.36
47.14
48.97
46.14
47.34
45.31
97.63
46.36
47.67
47.50
49.33
49.96
52.11
50.76
50.62
50.05
46.53
44.75
47.83
47.66
46.17
46.56
98.38
45.21
47.33
47.65
49.48
50.22
52.70
51.18
51.15
50.22
46.80
47.22
48.71
47.59
47.23
45.51
98.44
46.58
47.62
48.73
49.46
49.83
51.10
51.28
48.87
50.20
46.51
45.19
48.56
47.30
45.38
45.07
24 / 47


---

## 第29页

Comparison
0
2
3
2
2
s
20
25
30
35
40
45
50
55
60
PSNR
FD
RT
CD
L = 2, n = 3 (s)
L = 3, n = 3 (s)
L = 4, n = 3 (s)
L = 8, n = 3 (s)
Target
FD
RT
CD
L = 2 (s)
L = 3 (s)
L = 4 (s)
L = 8 (s)
25 / 47


---

## 第30页

Learning for disk regularization
▶We train on 64 images with binary disks of various radii and subpixel shifted centers.
▶The ground truth solutions can be computed with an explicit formula.
▶We train on a training set and evaluate on a test set.
▶We experiment with different numbers of filters and different symmetry constraints for the filters.
(a) Input images gs
(b) Target images ts
26 / 47


---

## 第31页

Results
L = 2
L = 2 (s)
L = 3
L = 3 (s)
L = 4 (s)
L = 8 (s)
Data
FD
RT
CD
L = 2
L = 2 (s)
L = 3
L = 3 (s)
L = 4 (s)
L = 8 (s)
Train
22.28
1.36
2.33
2.10
2.10
1.62
1.63
0.73
0.48
Test
22.36
1.32
2.30
2.10
2.10
1.60
1.60
0.72
0.47
Table: 105 times the mean squared error (MSE) of handcrafted and learned filters for the disk denoising problem.
Observe that L = 4 (s) is very close to RT on cubic meshes!
27 / 47


---

## 第32页

Comparison
0.3
0.4
0.5
0.6
0.7
rs
20
25
30
35
40
45
50
55
60
PSNR
FD
RT
CD
L = 2 (s)
L = 3 (s)
L = 4 (s)
L = 8 (s)
Target
FD
RT
CD
L = 2 (s)
L = 3 (s)
L = 4 (s)
L = 8 (s)
28 / 47


---

## 第33页

Natural image denoising
▶We extract 64 patches of size 64 × 64 from a natural image database.
▶The input images gs contain 5% Gaussian noise.
▶We learn both the filter weights and the regularization parameter λ by projecting on the set of
filters with sum equals λ > 0.
(a) Input images gs
(b) Target images ts
29 / 47


---

## 第34页

Results
L = 8 (s), 2 × 2
L = 40 (s), 6 × 6
L = 40, 6 × 6
Data
FD
RT
CD
L = 8 (s)
L = 40 (s)
L = 40
Train
5.05
5.33
4.87
4.58
4.31
4.22
Test
4.72
5.05
4.51
4.28
4.10
4.13
Table: 104× the mean squared error (MSE) of handcrafted and learned filters for natural image denoising.
30 / 47


---

## 第35页

Comparison
26
28
30
32
34
36
CD
26
28
30
32
34
36
L = 40 (s)
(a) Training set
26
28
30
32
34
36
CD
26
28
30
32
34
36
L = 40 (s)
(b) Test set
Target t
Input g
CD, PSNR=28.98
L = 40 (s), PSNR=29.77
(c) Example from the test set
31 / 47


---

## 第36页

Crossover experiments
▶How well do the learned filters generalize to other tasks?
▶We compare the filters L = 8 (s) which gave good results on all tasks.
Learning task
Handcrafted
Line
Disk
Natural
CD
Evaluation task
Line
0.82
243.55
50.71
6.33
Disk
1.88
0.47
4.08
2.30
Natural
48.68
49.65
42.80
45.10
▶The filters learned for inpainting generalize best, but there is no universal best discretization.
32 / 47


---

## 第37页

Extension to 3D
▶Very recently, we have extended the learning of the discrete total variation to 3D
▶We learn 4 sets of 3D filters on 3D minimal surface problems for which closed form solutions are
available
0
5
10
15
20
25
30
0
10
20
30
40
50
60
10
20
30
40
50
60
Catenoid
3D filters
33 / 47


---

## 第38页

Computing the Schwarz P surface
▶After learning, we can compute high-accuracy minimal surfaces for which no closed for solution is
available.
▶A well-known example is the Schwarz P surface
0
20
40
60
80
100
120
0
20
40
60
80
100
120
0
20
40
60
80
100
120
34 / 47


---

## 第39页

Total generalized variation
▶In recent work [Bogensperger, Chambolle, Effland, P. ’23] we have extended the framework to the
secons order total generalized variation [Bredies, Kunsich, P. ’10]
TGV2
α(u) = sup
p
 Z
Ω
u div2 p dx : p ∈C∞(Ω, Sym2×2), ∥p∥∞≤α0, ∥div p∥∞≤α1

,
▶In contrast to the total variation, the second order TGV can reconstruct piecewise affine images.
▶In [Hosseini, Bredies ’22] a Condat-like discretization was proposed, which served as the starting
point for our work.
35 / 47


---

## 第40页

Example
(a) Noisy
(b) TV
(c) TGV2
(d) Graph of TV
(e) Graph of TGV2
36 / 47


---

## 第41页

Discrete model
▶The second-order TGV discretization in its primal form is given by
TGV (u) = min
w α1∥Du −w∥+ α0∥Ew∥,
where D is again the finite differences operator Du = ((Du)1, (Du)2), where
(Du)1
i+ 1
2 ,j = 1
h(ui+1,j −ui,j)
i ≤M −1, j ≤N,
(Du)2
i,j+ 1
2 = 1
h(ui,j+1 −ui,j)
i ≤M, j ≤N −1.
and E is the symmetrized vectorial gradient operator given by Ew =
(Ew)1
(Ew)2
(Ew)2
(Ew)3

with
(Ew)1
i+1,j = 1
h(w1
i+ 3
2 ,j −w1
i+ 1
2 ,j)
i ≤M −1, j ≤N,
(Ew)2
i+ 1
2 ,j+ 1
2 =
1
2h(w1
i+ 1
2 ,j+1 −w1
i+ 1
2 ,j + w2
i+1,j+ 1
2 −w2
i,j+ 1
2 )
i ≤M −1, j ≤N −1,
(Ew)3
i,j+1 = 1
h(w2
i,j+ 3
2 −w2
i,j+ 1
2 )
i ≤M, j ≤N −1.
▶Now, introducing again interpolation operators L and K, applying some “convexity magic” the
model becomes
TGV (u) = sup
p ⟨u, div2 p⟩, s.t. ∥L div p∥∗
Z ≤α1, ∥Kp∥∗
Z ≤α0.
37 / 47


---

## 第42页

Pixel grid
ui,j+1
ui+1,j
ui+1,j+1
ui,j
w1
i+1/2,j
w1
i+3/2,j
w1
i+1/2,j+1
w1
i+3/2,j+1
w2
i,j+1/2
w2
i+1,j+1/2
w2
i,j+3/2
w2
i+1,j+3/2
p1
i+1,j
p1
i+2,j
p1
i+1,j+1, p3
i+1,j+1 p1
i+2,j+1
p3
i,j+1
p3
i,j+2
p3
i+1,j+2
p2
i+1/2,j+1/2 p2
i+3/2,j+1/2
p2
i+1/2,j+3/2 p2
i+3/2,j+3/2
38 / 47


---

## 第43页

Γ-convergence
The discrete TGV model is given by
TGV2
α,h(uh) =
min
wh,vh
K,vh
L

h2α1∥vh
L∥Z + h2α0∥vh
K∥Z : L∗
hvh
L = Dhuh −wh, K∗
hvh
K = Ehwh

= sup
ph

h2⟨div2
h ph, uh⟩: ∥Lh divh ph∥∗
Z ≤α1, ∥Khph∥∗
Z ≤α0

.
Theorem
We consider the setting where u is affine plus periodic with period 1 in R2, and w is 1-periodic. Then,
for interpolation operators K and L that have local support and bounded filter coefficients,
TGV2
α,h(uh) Γ-converges to TGV2
α(u).
39 / 47


---

## 第44页

Quantitative results for image denoising
Table: Quantitative comparison of natural image denoising of the test set with 5% and 10% Gaussian noise for
different handcrafted and learned discretizations.
5% Gaussian noise
10% Gaussian noise
PSNR
MSE ·10−2
SSIM
PSNR
MSE·10−2
SSIM
Corrupted f
26.04
0.2490
0.7885
20.02
0.9959
0.5382
TV
30.14
0.1049
0.9249
26.52
0.2445
0.8497
TGV
30.2
0.1043
0.9257
26.56
0.2431
0.8512
Handcrafted Disc. nK=1, nL=3
30.24
0.1046
0.9267
26.69
0.2394
0.8553
Handcrafted Disc. nK=4, nL=4
30.29
0.1030
0.9278
26.71
0.2370
0.8565
Learned Disc. nK=1, nL=3, 3 × 3
30.52
0.0935
0.9274
26.95
0.2172
0.8596
Learned Disc. nK=4, nL=4, 3 × 3
30.66
0.0906
0.9298
27.06
0.2123
0.8620
Learned Disc. nK=8, nL=8, 7 × 7
30.74
0.0896
0.9314
27.14
0.2090
0.8649
Learned Disc.
nK=8, nL=8, 7 × 7,
sym.
30.72
0.0898
0.9311
27.15
0.2089
0.8649
Learned Disc. nK=10, nL=10, 7 × 7
30.73
0.0896
0.9313
27.17
0.2081
0.8657
Learned Disc. nK=16, nL=16, 7 × 7
30.77
0.0891
0.9319
27.16
0.2087
0.8654
Learned Disc. nK=16, nL=16, 7 × 7,
sym.
30.77
0.0890
0.9320
27.18
0.2074
0.8659
40 / 47


---

## 第45页

Learned filters
learned ﬁlters K with nK=16
learned ﬁlters L with nL=16
Figure: Learned 7 × 7 filters using nL=16 and nK=16 for denoising (10% Gaussian noise). The row of a
depicted filter denotes the component of the respective vector/tensor field that it acts upon, whereas the
column refers to the specific filter r or l (with r = 1, · · · , nK and l = 1, · · · , nL.)
41 / 47


---

## 第46页

Example results
corrupted
standard TGV
handcrafted
learned
ground truth
Figure: Sample reconstructions from natural test images (10% Gaussian noise) comparing the standard TGV,
the handcrafted discretization scheme with nK=1, nL=3, and learned filters using nK=16, nL=16. For
completeness, the ground truth images are also shown.
42 / 47


---

## 第47页

Beyond total variation regularization
▶In [Bogensperger, Chambolle, P. ’22], we applied the same learning framework to the shearlet
transform [Kanghui, Kutyniok, Labate ’06].
▶A shearlet at scale j and shearing k is defined as.
ψd
j,k =
h
Sk
 (pj ∗Wj)↑2j/2 ∗1 hj/2

∗1 ¯hj/2
i
↓2j/2,
which essentially is constructed from a 1D low-pass filter h1 and anisotropic 2D filter P.
Additionally we also learn the importance weight λj,k of each shearlet.
▶Shearlets provide a multiscale framework similar to wavelets but better suited for encoding
anisotropic features necessary for an efficient sparse representations of cartoon-like images.
▶We tried to further optimize the shearlets using the piggy-back primal dual algorithm based on
smooth-ℓ1-regularized image denoising model
min
u ∥K(θ)u∥1,ε + 1
2 ∥u −z∥2
2 ,
where θ is a placeholder for all learnable parameters and ∥·∥1,ε refers to a C2,1 approximation of
the ℓ1 norm.
▶Experiments are carried out on both natural images and synthetic piecewise affine images and we
experimented with different settings of the smoothness parameter ε.
43 / 47


---

## 第48页

Learned shearlets
44 / 47


---

## 第49页

Learned shearlets
44 / 47


---

## 第50页

Image denoising
45 / 47


---

## 第51页

Image denoising
45 / 47


---

## 第52页

Quantitative evaluation
46 / 47


---

## 第53页

Conclusion
▶We proposed learning optimized finite differences discretizations of the total variation.
▶The learning is constraint to a class of consistent discretizations which Γ-converge to the
continuous total variation.
▶We proposed a piggy-back primal-dual algorithm for computing derivatives.
▶Symmetry constraints on the filters give better generalizations.
▶The learned discretizations give significant improvements when optimized for certain applications
but no best universal discretization could be learned.
▶The learning framework has been extended to 3D TV and more complex regularization operators
such as TGV and shearlets.
47 / 47


---

## 第54页

Conclusion
▶We proposed learning optimized finite differences discretizations of the total variation.
▶The learning is constraint to a class of consistent discretizations which Γ-converge to the
continuous total variation.
▶We proposed a piggy-back primal-dual algorithm for computing derivatives.
▶Symmetry constraints on the filters give better generalizations.
▶The learned discretizations give significant improvements when optimized for certain applications
but no best universal discretization could be learned.
▶The learning framework has been extended to 3D TV and more complex regularization operators
such as TGV and shearlets.
Thank you for your attention!
47 / 47


---

