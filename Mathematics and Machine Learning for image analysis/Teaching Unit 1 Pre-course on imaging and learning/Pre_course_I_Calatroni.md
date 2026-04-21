# Pre_course_I_Calatroni.pdf

## 第1页

Pre-course: imaging and learning, part I 
(images, noise, Bayesian/variational formulation, optimisation)
Co-funded by the European Union (ERC, MALIN, 10117133). Views and opinions expressed are however those of the author only and do not necessarily reflect those of the European Union or 
the European Research Council Executive Agency. Neither the EU nor the granting authority can be held responsible for them.
Luca Calatroni 
MaLGa Center, Department of Computer Science, University of Genoa, Italy
Mathematics & Machine Learning for image analysis 
ERASMUS+ International PhD Summer School 
Bologna, June 3-11 2025


---

## 第2页

Meet the instructors (Luca’s) & program
2
- Digital images  
- Modelling noise 
- Quality measures (MSE, PSNR, SSIM) 
- Image denoising as a toy ‘inverse’ problem 
- Bayesian formulation 
- MAP estimators 
- Regularisation  
- Bits of optimisation: GD, proximal operator
Luca Calatroni 
Part I (14:30-15:45)
Luca Ratti 
Part II (16:00-17:15)
- Image denoising as a regression problem 
- Supervised/unsupervised learning 
- Neural networks 
- Bias-variance tradeoff 
- Over-parametrisation 
- NNs for imaging: CNN, U-NET 
- Training a NN: backpropagation, batches, 
optimisers.. 
- NumPy/Matplotlib/Pytorch (tomorrow)


---

## 第3页

Digital images
3
Digital images are discrete representations of the continuous world we live in.
Sampling: allows to represent a continuous image 
into a finite (pixel) grid.
Quantisation: assigns a grey-level describing 
average brightness at each pixel.


---

## 第4页

Digital images
3
Digital images are discrete representations of the continuous world we live in.
Sampling: allows to represent a continuous image 
into a finite (pixel) grid.
Quantisation: assigns a grey-level describing 
average brightness at each pixel.
n1
n2
Ω = {1,…, n1} × {1,…, n2}
image domain


---

## 第5页

Digital images
3
Digital images are discrete representations of the continuous world we live in.
Sampling: allows to represent a continuous image 
into a finite (pixel) grid.
Quantisation: assigns a grey-level describing 
average brightness at each pixel.
n1
n2
Ω = {1,…, n1} × {1,…, n2}
image domain
X : Ω →{0,…,255}
X = (xi,j) ∈{0,…,255}
n1×n2


---

## 第6页

Digital images
3
Digital images are discrete representations of the continuous world we live in.
Sampling: allows to represent a continuous image 
into a finite (pixel) grid.
Quantisation: assigns a grey-level describing 
average brightness at each pixel.
n1
n2
Ω = {1,…, n1} × {1,…, n2}
image domain
X : Ω →{0,…,255}
X = (xi,j) ∈{0,…,255}
n1×n2
normalisation, 
adjustments..
X ∈[0,1]n1×n2
X ∈ℝn1×n2


---

## 第7页

Digital images
3
Digital images are discrete representations of the continuous world we live in.
Sampling: allows to represent a continuous image 
into a finite (pixel) grid.
Quantisation: assigns a grey-level describing 
average brightness at each pixel.
n1
n2
Ω = {1,…, n1} × {1,…, n2}
image domain
X : Ω →{0,…,255}
X = (xi,j) ∈{0,…,255}
n1×n2
Upon vectorisation of the 2D image 
, consider a vector 
, with 
X
x ∈ℝn
n = n1n2 .
normalisation, 
adjustments..
X ∈[0,1]n1×n2
X ∈ℝn1×n2


---

## 第8页

Grayscale/RGB images
4
Natural scenes are not grayscale. Color is a combination of Red-Green-Blue channels.
The higher the intensity in the individual 
channel, the more represented is the color.
xi,j = (ri,j, gi,j, bi,j) ∈ℝ3
Hence, for RGB images: 
x ∈ℝn×3
- Other color spaces are possible (CMYK, HSV..) 
- Often, color channels are processed separately. 
or
ℝ3n


---

## 第9页

Grayscale/RGB images
4
Natural scenes are not grayscale. Color is a combination of Red-Green-Blue channels.
The higher the intensity in the individual 
channel, the more represented is the color.
xi,j = (ri,j, gi,j, bi,j) ∈ℝ3
Hence, for RGB images: 
x ∈ℝn×3
- Other color spaces are possible (CMYK, HSV..) 
- Often, color channels are processed separately. 
Assume  is grayscale (extension is trivial)
x
or
ℝ3n


---

## 第10页

Modelling degradation processes
5
Images are matrices/vectors. How to model acquisition processes?
x ∈ℝn
y ∈ℝm
(observed)
(unknown)
y = Ax
linear input ( )-output ( ) relation
x
y
A ∈ℝm×n


---

## 第11页

Modelling degradation processes
5
Images are matrices/vectors. How to model acquisition processes?
x ∈ℝn
y ∈ℝm
(observed)
(unknown)
y = Ax
linear input ( )-output ( ) relation
x
y
A ∈ℝm×n
- Convolution: Ax ↔h * X,  is a kernel
h


---

## 第12页

Modelling degradation processes
5
Images are matrices/vectors. How to model acquisition processes?
x ∈ℝn
y ∈ℝm
(observed)
(unknown)
y = Ax
linear input ( )-output ( ) relation
x
y
A ∈ℝm×n
- Convolution: Ax ↔h * X,  is a kernel
h
- Masking: 
A = M ∈{0,1}m×n


---

## 第13页

Modelling degradation processes
5
Images are matrices/vectors. How to model acquisition processes?
x ∈ℝn
y ∈ℝm
(observed)
(unknown)
y = Ax
linear input ( )-output ( ) relation
x
y
A ∈ℝm×n
- Convolution: Ax ↔h * X,  is a kernel
h
- Fourier transform + Masking: 
A ⋅= Mℱ( ⋅)
- Masking: 
A = M ∈{0,1}m×n


---

## 第14页

Modelling noise
6
y = Noise(Ax)
Acquisitions are never perfect. Interferences, errors, faults may happen.
 codifies instrumental errors.  
Noise : ℝm →ℝm
In the following: 
. How to model noise?  
A = I ∈ℝn×n


---

## 第15页

Modelling noise
6
y = Noise(Ax)
Acquisitions are never perfect. Interferences, errors, faults may happen.
 codifies instrumental errors.  
Noise : ℝm →ℝm
In the following: 
. How to model noise?  
A = I ∈ℝn×n
Gaussian noise: 
Noise(x) = x + ε,
ε ∼𝒩(0, σ2I)
Mostly used due to CLT.  
Models signal-independent electronic noise


---

## 第16页

Modelling noise
6
y = Noise(Ax)
Acquisitions are never perfect. Interferences, errors, faults may happen.
 codifies instrumental errors.  
Noise : ℝm →ℝm
In the following: 
. How to model noise?  
A = I ∈ℝn×n
Gaussian noise: 
Noise(x) = x + ε,
ε ∼𝒩(0, σ2I)
Mostly used due to CLT.  
Models signal-independent electronic noise
Poisson noise:
Noise(x) = Pois(x+β),
x ∈ℝn
≥0, β ∈ℝn
>0
Used in low-photon imaging. 
Astronomical, microscopy imaging.
Bertero, Boccacci, ‘98


---

## 第17页

Modelling noise
6
y = Noise(Ax)
Acquisitions are never perfect. Interferences, errors, faults may happen.
 codifies instrumental errors.  
Noise : ℝm →ℝm
In the following: 
. How to model noise?  
A = I ∈ℝn×n
Gaussian noise: 
Noise(x) = x + ε,
ε ∼𝒩(0, σ2I)
Mostly used due to CLT.  
Models signal-independent electronic noise
Impulsive noise: 
Noise(x) = (1 −s) ⊙x + s ⊙c
ci = ℬ(1/2), si = ℬ(p), p ∈[0,1]
Used to describe faulty detectors  
and/or long time exposures under bad lighting.
Poisson noise:
Noise(x) = Pois(x+β),
x ∈ℝn
≥0, β ∈ℝn
>0
Used in low-photon imaging. 
Astronomical, microscopy imaging.
Bertero, Boccacci, ‘98


---

## 第18页

Quality metrics: MSE, SNR, PSNR
7
How to assess image quality between two images using pixel information?


---

## 第19页

Quality metrics: MSE, SNR, PSNR
7
How to assess image quality between two images using pixel information?
MSE(y, x) = 1
n
n
∑
i=1
|yi −xi|2 = 1
n1
1
n2
n1
∑
i1=1
n2
∑
i2=1
|yi1,i2 −xi1,i2|2


---

## 第20页

Quality metrics: MSE, SNR, PSNR
7
How to assess image quality between two images using pixel information?
MSE(y, x) = 1
n
n
∑
i=1
|yi −xi|2 = 1
n1
1
n2
n1
∑
i1=1
n2
∑
i2=1
|yi1,i2 −xi1,i2|2
SNR(y, x) = 10 log10 (
∥x∥2
∥x −y∥2)
noise ε


---

## 第21页

Quality metrics: MSE, SNR, PSNR
7
How to assess image quality between two images using pixel information?
MSE(y, x) = 1
n
n
∑
i=1
|yi −xi|2 = 1
n1
1
n2
n1
∑
i1=1
n2
∑
i2=1
|yi1,i2 −xi1,i2|2
PSNR(y, x) = 10 log10 (
MAX2
MSE(y, x) )
where 
 is the highest possible value (e.g., 
 
MAX
255 or 1)
SNR(y, x) = 10 log10 (
∥x∥2
∥x −y∥2)
noise ε


---

## 第22页

Quality metrics: MSE, SNR, PSNR
7
How to assess image quality between two images using pixel information?
MSE(y, x) = 1
n
n
∑
i=1
|yi −xi|2 = 1
n1
1
n2
n1
∑
i1=1
n2
∑
i2=1
|yi1,i2 −xi1,i2|2
PSNR(y, x) = 10 log10 (
MAX2
MSE(y, x) )
where 
 is the highest possible value (e.g., 
 
MAX
255 or 1)
Is it a good quality metric for natural images?!
Sensitive to pixel variations
SNR(y, x) = 10 log10 (
∥x∥2
∥x −y∥2)
noise ε


---

## 第23页

Quality metrics: SSIM
8
SSIM(x, y) =
(2μxμy + C1)(2σxy + C2)
(μ2x + μ2y + C1)(σ2x + σ2y + C2) ∈[0,1]
- Based on image statistics: mean, variance, covariance  
+ constant 
 stabilising the division.
C1, C2
- Typically performed on small image patches + averaging
Wang, Bovik, Sheikh, Simoncelli, ‘04


---

## 第24页

Quality metrics: SSIM
8
SSIM(x, y) =
(2μxμy + C1)(2σxy + C2)
(μ2x + μ2y + C1)(σ2x + σ2y + C2) ∈[0,1]
- Based on image statistics: mean, variance, covariance  
+ constant 
 stabilising the division.
C1, C2
- Typically performed on small image patches + averaging
RANGE
IDEAL
GOOD
MSE
SNR
PSNR
SSIM
[0,MAX2]
0
[0, + ∞)
1
(−∞, + ∞)
+∞
[0,1]
+∞
> 0.9
> 30
> 30
< 100
(for images in [0,255])
All such metrics are supervised. 
They depend on ground-truth .x
Wang, Bovik, Sheikh, Simoncelli, ‘04


---

## 第25页

Image denoising as a ‘toy’ inverse problem
9
Given 
, find 
 such that:
y ∈ℝn
x ∈ℝn
y = x + ε,
ε ∼𝒩(0, σ2I)
- “Inverse problem” (operator to invert is 
) 
- Still challenging: the noise realisation is unknown 
- If noise is high (  is big), image content can be lost
A = I
σ
x ∈[0,1]3n
σ = 0.002
σ = 0.05
σ = 0.3
MSE
PSNR/SSIM
How to model this problem in  
mathematical terms?


---

## 第26页

Bayesian formulation
10
Idea: model also  and  as realisation of random variables 
 + conditional laws.
x
y
𝒳∼π𝒳, 𝒴∼π𝒴


---

## 第27页

Bayesian formulation
10
Idea: model also  and  as realisation of random variables 
 + conditional laws.
x
y
𝒳∼π𝒳, 𝒴∼π𝒴
π𝒳|𝒴(x|y)
π𝒴|𝒳(y|x)
π𝒴(y)
π𝒳(x)
Posterior distribution: what we would like to maximise.
Likelihood: function describing the probability of observing the data given a choice of x
Evidence term: normally neglected, does not depend on x
Prior: prior assumptions on the unknown quantity x


---

## 第28页

Bayesian formulation
10
Idea: model also  and  as realisation of random variables 
 + conditional laws.
x
y
𝒳∼π𝒳, 𝒴∼π𝒴
π𝒳|𝒴(x|y)
π𝒴|𝒳(y|x)
π𝒴(y)
π𝒳(x)
Posterior distribution: what we would like to maximise.
Likelihood: function describing the probability of observing the data given a choice of x
Evidence term: normally neglected, does not depend on x
Prior: prior assumptions on the unknown quantity x
Bayes’ Theorem
π𝒳|𝒴(x|y) =
π𝒴|𝒳(y|x)π𝒳(x)
π𝒴(y)
Under Gaussian noise assumption:
π𝒴|𝒳(y|x) = πE(ε = y −x)


---

## 第29页

Bayesian formulation: priors, score functions
11
x* ∈argmaxx π𝒳|𝒴(x|y) = argmaxx
π𝒴|𝒳(y|x)π𝒳(x)
π𝒴(y)
= argmaxx π𝒴|𝒳(y|x)π𝒳(x)
By taking the negative logarithm:
x* ∈argminx −ln (π𝒳|𝒴(x|y)) = argminx −ln (π𝒴|𝒳(y|x)π𝒳(x)) = argminx −ln (π𝒴|𝒳(y|x)) −ln (π𝒳(x))
max. log-likelihood
prior


---

## 第30页

Bayesian formulation: priors, score functions
11
x* ∈argmaxx π𝒳|𝒴(x|y) = argmaxx
π𝒴|𝒳(y|x)π𝒳(x)
π𝒴(y)
= argmaxx π𝒴|𝒳(y|x)π𝒳(x)
By taking the negative logarithm:
x* ∈argminx −ln (π𝒳|𝒴(x|y)) = argminx −ln (π𝒴|𝒳(y|x)π𝒳(x)) = argminx −ln (π𝒴|𝒳(y|x)) −ln (π𝒳(x))
max. log-likelihood
prior
Example:
π𝒴|𝒳(y|x) = πE(y −x),
π𝒳(x) = 𝒩(μx, Σx)
x* ∈argminx −ln
n
∏
i=1
1
2πσ2ε
exp (−|yi −xi|2
2σ2ε
) −ln (
1
(2π)n/2|Σx|1/2 exp (−1
2 (x −μx)⊤Σx
−1(x −μx)))
—-> towards optimisation problem!
= argminx
1
2σ2ε
∥y −x∥2 + 1
2∥x −μx∥2
Σ−1
x
+ neglecting constants


---

## 第31页

From a statistical to a variational perspective: optimisation
12
argminx J(x) := −ln (π𝒴|𝒳(y|x)) −ln (π𝒳(x))
Whenever likelihood + prior functionals belong to a log-concave exponential family we end up  
with convex optimisation problems for a functional J : ℝn →ℝ≥0 ∪{+∞}


---

## 第32页

From a statistical to a variational perspective: optimisation
12
argminx J(x) := −ln (π𝒴|𝒳(y|x)) −ln (π𝒳(x))
Whenever likelihood + prior functionals belong to a log-concave exponential family we end up  
with convex optimisation problems for a functional J : ℝn →ℝ≥0 ∪{+∞}
We are going to see a very simple algorithm for minimising  based on gradients. Note:
J
∇J(x*) = −∇ln (π𝒴|𝒳(y|x*)) −∇ln (π𝒳(x*)) = 0
 when looking for MAP estimators.


---

## 第33页

Bayesian model
Optimisation 
approach
Noise 
information
Likelihood
Data-term
A-priori 
information
Prior
Regularisation
Parameters
Hyperparameters
Regularisation 
parameters
From a statistical to a variational perspective: optimisation
12
argminx J(x) := −ln (π𝒴|𝒳(y|x)) −ln (π𝒳(x))
Whenever likelihood + prior functionals belong to a log-concave exponential family we end up  
with convex optimisation problems for a functional J : ℝn →ℝ≥0 ∪{+∞}
We are going to see a very simple algorithm for minimising  based on gradients. Note:
J
∇J(x*) = −∇ln (π𝒴|𝒳(y|x*)) −∇ln (π𝒳(x*)) = 0
 when looking for MAP estimators.


---

## 第34页

Bayesian model
Optimisation 
approach
Noise 
information
Likelihood
Data-term
A-priori 
information
Prior
Regularisation
Parameters
Hyperparameters
Regularisation 
parameters
From a statistical to a variational perspective: optimisation
12
argminx J(x) := −ln (π𝒴|𝒳(y|x)) −ln (π𝒳(x))
Whenever likelihood + prior functionals belong to a log-concave exponential family we end up  
with convex optimisation problems for a functional J : ℝn →ℝ≥0 ∪{+∞}
We are going to see a very simple algorithm for minimising  based on gradients. Note:
J
∇J(x*) = −∇ln (π𝒴|𝒳(y|x*)) −∇ln (π𝒳(x*)) = 0
 when looking for MAP estimators.
How to choose good noise and image models?


---

## 第35页

Data terms
13
y = Noise(x)
π𝒴|𝒳(y|x)
−ln
D(y, x)
Tailored distance function with 
observation related to noise assumptions?
+ constants


---

## 第36页

Data terms
13
y = Noise(x)
π𝒴|𝒳(y|x)
−ln
D(y, x)
Gaussian noise: 
Noise(x) = x + ε,
ε ∼𝒩(0, σ2
εI)
D(y, x) = D(y −x) =
1
2σ2ε
∥y −x∥2
2
Tailored distance function with 
observation related to noise assumptions?
+ constants


---

## 第37页

Data terms
13
y = Noise(x)
π𝒴|𝒳(y|x)
−ln
D(y, x)
Gaussian noise: 
Noise(x) = x + ε,
ε ∼𝒩(0, σ2
εI)
D(y, x) = D(y −x) =
1
2σ2ε
∥y −x∥2
2
Poisson noise:
Noise(x) = Pois(x+β),
x ∈ℝn
≥0, β ∈ℝn
>0
D(y, x) = D(y, x + β) = KL(y, x + β) =
n
∑
i=1
xi + βi −yi ln (xi + βi)
Tailored distance function with 
observation related to noise assumptions?
+ constants


---

## 第38页

Data terms
13
y = Noise(x)
π𝒴|𝒳(y|x)
−ln
D(y, x)
Gaussian noise: 
Noise(x) = x + ε,
ε ∼𝒩(0, σ2
εI)
D(y, x) = D(y −x) =
1
2σ2ε
∥y −x∥2
2
Poisson noise:
Noise(x) = Pois(x+β),
x ∈ℝn
≥0, β ∈ℝn
>0
D(y, x) = D(y, x + β) = KL(y, x + β) =
n
∑
i=1
xi + βi −yi ln (xi + βi)
Tailored distance function with 
observation related to noise assumptions?
Impulsive noise: 
Noise(x) = (1 −s) ⊙x + s ⊙c
ci = ℬ(1/2), si = ℬ(p), p ∈[0,1]
D(y, x) = D(y −x) = 1
τε
∥y −x∥1
noise “sparsity” (the residual is 0 only in few pixels)
≈x + ε,
ε ∼ℒ(0, τεI)
+ constants


---

## 第39页

Regularisation terms
14
π𝒳(x)
−ln
R(x)
codify a-priori information
+ constants


---

## 第40页

Regularisation terms
14
π𝒳(x)
−ln
R(x)
codify a-priori information
Regularity around the mean:
π𝒳(x) = 𝒩(μx, Σx)
R(x) = 1
2 ∥x −μx∥2
Σx
−1
+ constants


---

## 第41页

Regularisation terms
14
π𝒳(x)
−ln
R(x)
codify a-priori information
Regularity around the mean:
π𝒳(x) = 𝒩(μx, Σx)
R(x) = 1
2 ∥x −μx∥2
Σx
−1
Sparsity: 
π𝒳(x) = ℒ(0, τI)
R(x) = 1
τ ∥x∥1
+ constants


---

## 第42页

Regularisation terms
14
π𝒳(x)
−ln
R(x)
codify a-priori information
Regularity around the mean:
π𝒳(x) = 𝒩(μx, Σx)
R(x) = 1
2 ∥x −μx∥2
Σx
−1
qi = ∥(∇x)i∥2
Sparsity: 
π𝒳(x) = ℒ(0, τI)
R(x) = 1
τ ∥x∥1
+ constants


---

## 第43页

Regularisation terms
14
π𝒳(x)
−ln
R(x)
codify a-priori information
Regularity around the mean:
π𝒳(x) = 𝒩(μx, Σx)
R(x) = 1
2 ∥x −μx∥2
Σx
−1
qi = ∥(∇x)i∥2
Sparsity: 
π𝒳(x) = ℒ(0, τI)
R(x) = 1
τ ∥x∥1
Smoothness:
π𝒳(x) = π𝒬(q) = 𝒩(0, σ2
qI)
R(x) =
1
2σ2q
∥∇x∥2
2
+ constants


---

## 第44页

Regularisation terms
14
π𝒳(x)
−ln
R(x)
codify a-priori information
Regularity around the mean:
π𝒳(x) = 𝒩(μx, Σx)
R(x) = 1
2 ∥x −μx∥2
Σx
−1
qi = ∥(∇x)i∥2
Sparsity: 
π𝒳(x) = ℒ(0, τI)
R(x) = 1
τ ∥x∥1
Piece-wise constancy:
π𝒳(x) = π𝒬(q) = ℒ(0, τI)
R(x) = 1
τ TV(x) = 1
τ ∥∇x∥2,1
Smoothness:
π𝒳(x) = π𝒬(q) = 𝒩(0, σ2
qI)
R(x) =
1
2σ2q
∥∇x∥2
2
+ constants


---

## 第45页

Regularisation terms
14
π𝒳(x)
−ln
R(x)
codify a-priori information
Regularity around the mean:
π𝒳(x) = 𝒩(μx, Σx)
R(x) = 1
2 ∥x −μx∥2
Σx
−1
qi = ∥(∇x)i∥2
Sparsity: 
π𝒳(x) = ℒ(0, τI)
R(x) = 1
τ ∥x∥1
Piece-wise constancy:
π𝒳(x) = π𝒬(q) = ℒ(0, τI)
R(x) = 1
τ TV(x) = 1
τ ∥∇x∥2,1
Smoothness:
π𝒳(x) = π𝒬(q) = 𝒩(0, σ2
qI)
R(x) =
1
2σ2q
∥∇x∥2
2
Tikhonov
Sparse reg./ 
Compressed sensing
Tikhonov/ 
Sobolev
Total Variation
+ constants


---

## 第46页

The regularisation parameter
15
π𝒴|𝒳(y|x) = πE(y −x),
π𝒳(x) = 𝒩(0, σ2
xI)
Example:
argminx
1
2σ2ε
∥y −x∥2 + 1
2σ2x
∥x∥2
2 = argminx ∥y −x∥2 + σ2
ε
σ2x
∥x∥2 = argminx ∥y −x∥2 +λ∥x∥2
Ratio between noise level (can be estimated) and image statistical features (hard to estimate).


---

## 第47页

The regularisation parameter
15
π𝒴|𝒳(y|x) = πE(y −x),
π𝒳(x) = 𝒩(0, σ2
xI)
Example:
argminx
1
2σ2ε
∥y −x∥2 + 1
2σ2x
∥x∥2
2 = argminx ∥y −x∥2 + σ2
ε
σ2x
∥x∥2 = argminx ∥y −x∥2 +λ∥x∥2
Ratio between noise level (can be estimated) and image statistical features (hard to estimate).
More in general, the regularisation parameter 
 weights data fit against regularisation.
λ > 0
argminx D(y, x)+λR(x)
- Small : low regularisation, trust in the data, noise overfit
λ
- High : high regularisation, need to regularise the data, artefacts induced by 
λ
R
λ →+ ∞


---

## 第48页

Examples
16
y = Noise(x)
Gaussian noise + piece-wise constant image: reference model for many applications.
Poisson noise + sparse signal: used in microscopy/astronomical imaging
Gaussian noise + Tikhonov-type regularisation: often used when 
 for applications
A ≠I
argminx
1
2 ∥y −x∥2
2 + λTV(x)
argminx≥0 KL(y, x + β) + λ∥x∥1
argminx
1
2 ∥y −x∥2
2 + λ
2 ∥Lx∥2
2
L ∈ℝd×n


---

## 第49页

Solving the problem: nods on optimisation
17
argminx J(x) := D(y, x) + λR(x)
variational formulation of the image denoising problem
J : ℝn →ℝ≥0 ∪{+∞}
 is -smooth, i.e. has -Lipschitz (Gâteaux) gradient:
J
L
L
∃L > 0 :
∥∇J(x1) −∇J(x2)∥2 ≤L∥x1 −x2∥2
is proper
dom(J) = {x : J(x) < + ∞} ≠∅
 is convex:
J
(∀x1, x2 ∈ℝn),
(∀α ∈[0,1]) :
J(αx1 + (1 −α)x2) ≤αJ(x1) + (1 −α)J(x2)


---

## 第50页

Solving the problem: nods on optimisation
17
argminx J(x) := D(y, x) + λR(x)
variational formulation of the image denoising problem
J : ℝn →ℝ≥0 ∪{+∞}
 is -smooth, i.e. has -Lipschitz (Gâteaux) gradient:
J
L
L
∃L > 0 :
∥∇J(x1) −∇J(x2)∥2 ≤L∥x1 −x2∥2
is proper
dom(J) = {x : J(x) < + ∞} ≠∅
 is convex:
J
(∀x1, x2 ∈ℝn),
(∀α ∈[0,1]) :
J(αx1 + (1 −α)x2) ≤αJ(x1) + (1 −α)J(x2)
⟺
J(x1) ≤J(x2) + ⟨∇J(x2), x2 −x1⟩+ L
2 ∥x2 −x1∥2
2
under 
convexity


---

## 第51页

Solving the problem: nods on optimisation
18
argminx J(x) := D(y, x) + λR(x)
 is proper, convex, -smooth and coercive.
J
L
Theorem
There exists a minimiser for . All local minimisers are global minimisers.  
For all 
, there holds 
.
J
x* ∈Argminx J(x)
∇J(x*) = 0
lim
∥x∥→+∞J(x) = + ∞


---

## 第52页

Solving the problem: nods on optimisation
18
argminx J(x) := D(y, x) + λR(x)
 is proper, convex, -smooth and coercive.
J
L
Theorem
There exists a minimiser for . All local minimisers are global minimisers.  
For all 
, there holds 
.
J
x* ∈Argminx J(x)
∇J(x*) = 0
lim
∥x∥→+∞J(x) = + ∞
Algorithm (gradient descent): for 
, 
: 
x0 ∈dom(J), τ ∈(0, 2
L) k ≥0
xk+1 = xk −τ∇J(xk)
Theorem
There holds 
 and for the function values:   
. 
xk →x*
J(xk) −J(x*) ≤∥x0 −x*∥2
2τk


---

## 第53页

Example on Tikhonov regularisation
19
argminx J(x) := 1
2 ∥y −x∥2
2 + λ∥Lx∥2
2,
L ∈ℝd×n
Remark: the problem is quadratic, it can be solved by looking at the optimality condition:
(x* −y) + λLTLx* = 0
⇒
(I + λLTL)x* = y
and solving the linear system, e.g., using DFT. Also, faster iterative methods exploiting further 
regularity (strong convexity, 
  can be employed.
C2)
L ∈{I, ∇, ∇2}
Hansen, Nagy, O’leary, ’06, Nesterov, ’83
Examples:


---

## 第54页

Example on Tikhonov regularisation
19
argminx J(x) := 1
2 ∥y −x∥2
2 + λ∥Lx∥2
2,
L ∈ℝd×n
Remark: the problem is quadratic, it can be solved by looking at the optimality condition:
(x* −y) + λLTLx* = 0
⇒
(I + λLTL)x* = y
and solving the linear system, e.g., using DFT. Also, faster iterative methods exploiting further 
regularity (strong convexity, 
  can be employed.
C2)
∇J(x) = (x −y) + λLTLx,
L = 1 + λ∥LTL∥*,
x0 ∈ℝn,
τ ∈(0,2/L)
xk+1 = xk −τ ((xk −y) + λLTLxk)
while not converging
end
L ∈{I, ∇, ∇2}
Hansen, Nagy, O’leary, ’06, Nesterov, ’83
Examples:


---

## 第55页

Image denoisers and proximal operators
20
For general (possibly non-smooth) regularisation functionals  
, note that:
R : ℝn →ℝ≥0 ∪{+∞}
argminx
1
2 ∥y −x∥2 + λR(x) = proxλR(y)
where 
 is single-valued if  is convex and multi-valued (multiple minimisers) 
otherwise. 
proxλR : ℝn ⇉ℝn
R


---

## 第56页

Image denoisers and proximal operators
20
For general (possibly non-smooth) regularisation functionals  
, note that:
R : ℝn →ℝ≥0 ∪{+∞}
argminx
1
2 ∥y −x∥2 + λR(x) = proxλR(y)
where 
 is single-valued if  is convex and multi-valued (multiple minimisers) 
otherwise. 
proxλR : ℝn ⇉ℝn
R
Examples:
 is convex and closed. 
R(x) = ιC(x), C
 
R(x) = ∥x∥1
 
R(x) = TV(x)
Non-smooth regularisation functionals 
 not defined.
∇R
proxιC(y) = PC(y)
proxλ∥⋅∥1(y) = ST(y; λ)
proxλ TV(⋅)(y)?


---

## 第57页

Image denoisers and proximal operators
20
For general (possibly non-smooth) regularisation functionals  
, note that:
R : ℝn →ℝ≥0 ∪{+∞}
argminx
1
2 ∥y −x∥2 + λR(x) = proxλR(y)
where 
 is single-valued if  is convex and multi-valued (multiple minimisers) 
otherwise. 
proxλR : ℝn ⇉ℝn
R
Proximal operators are widely used as implicit variants of gradients for non-smooth optimisation:
xk+1 = proxτλR (xk −τ∇xD(y, Axk))
Gradient-descent on data term + denoising
This observation stands at the very basis of Plug & Play approaches where proxτλR
𝒟ς
Kamilov, Bouman, Buzzard, Wuhlberg, ‘23
Examples:
 is convex and closed. 
R(x) = ιC(x), C
 
R(x) = ∥x∥1
 
R(x) = TV(x)
Non-smooth regularisation functionals 
 not defined.
∇R
proxιC(y) = PC(y)
proxλ∥⋅∥1(y) = ST(y; λ)
proxλ TV(⋅)(y)?


---

## 第58页

References
21
A. Beck, First-order methods in optimization, Volume 25, MOS-SIAM series on Optimization, 2017.
A. Chambolle, T. Pock, An introduction to continuous optimization for imaging, Acta Numerica, 2016
M. Pragliola, L. Calatroni, A. Lanza, F. Sgallari, On and beyond Total Variation in imaging: the role of space 
variance, SIAM Review, 65 (3), (2023). 
Tony F. Chan, J. Shen, Image Processing and Analysis: variational, PDE, wavelet and stochastic methods, 
SIAM, 2005.
S. Arridge, P. Maas, O. Öktem, C.B. Schönlieb, Solving inverse problems using data-driven models, 
Acta Numerica, 2019
J. Kaipio, E. Somersalo, Statistical and computational inverse problems, Springer, 2005.


---

## 第59页

luca.calatroni@unige.it
Thanks for your attention!


---

