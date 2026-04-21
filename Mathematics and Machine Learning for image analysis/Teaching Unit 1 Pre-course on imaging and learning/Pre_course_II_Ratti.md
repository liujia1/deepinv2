# Pre_course_II_Ratti.pdf

## 第1页

Pre-course, part II:
black-box machine learning for image denoising
Luca Ratti
Department of Mathematics, Università degli studi di Bologna,
luca.ratti5@unibo.it
ERASMUS+ International PhD Summer School 2025
Mathematics and Machine Learning for image analysis
University of Bologna
3rd June 2025


---

## 第2页

A statistical learning perspective on
image denoising


---

## 第3页

Statistical learning - a supervised regression problem
On a completely different note...
[Cucker, Smale, On the mathematical foundations of learning, 2002]
A supervised regression problem
Given two random variables x on X and y on Y
find a function f: Y →X s.t. f(y) ≈x
▶(ideal setting) knowing the joint distribution πy,x ∼(y, x);
▶(real setting) knowing a sample {(yi, xi)}N
i=1 ∼i.i.d πy,x;
Luca Ratti
PhD Summer School 2025 - Pre-course part II
1/17


---

## 第4页

Statistical learning - a supervised regression problem
On a completely different note...
[Cucker, Smale, On the mathematical foundations of learning, 2002]
A supervised regression problem
Given two random variables x on X and y on Y
find a function f: Y →X s.t. f(y) ≈x
▶(ideal setting) knowing the joint distribution πy,x ∼(y, x);
▶(real setting) knowing a sample {(yi, xi)}N
i=1 ∼i.i.d πy,x;
1) Consider a loss function, e.g. ℓ(x′, x) = ∥x′ −x∥2
X.
2) Introduce F: hypothesis space, a set of functions from Y to X.
3.id) Minimize the Expected Loss: ⇝min
f∈F E(x,y)∼πxy
h
ℓ(f(y), x)
i
3.re) Minimize the Empirical Risk: ⇝min
f∈F
1
N
N
X
i=1
ℓ(f(yi), xi)
Luca Ratti
PhD Summer School 2025 - Pre-course part II
1/17


---

## 第5页

How we usually see regression...
X = R,
Y = R,
x = sin(y) + ε,
y ∼U([0, 5]),
ε ∼N(0, 0.22)
Luca Ratti
PhD Summer School 2025 - Pre-course part II
2/17


---

## 第6页

How we usually see regression...
X = R,
Y = R,
F: polynomials of degree 1
Luca Ratti
PhD Summer School 2025 - Pre-course part II
2/17


---

## 第7页

How we usually see regression...
X = R,
Y = R,
F: polynomials of degree 5
Luca Ratti
PhD Summer School 2025 - Pre-course part II
2/17


---

## 第8页

How we usually see regression...
X = R,
Y = R,
F: polynomials of degree 10
Luca Ratti
PhD Summer School 2025 - Pre-course part II
2/17


---

## 第9页

... but denoising can be seen as a regression task!
Learned reconstruction for inverse problems:
suppose that y = Ax + ε (forward model): f(y) ≈x ⇝find f ≈A−1.
Denoising: let X = Y and A = Id, i.e.: y = x + ε.
Why not using simply f = A−1 = Id?
Luca Ratti
PhD Summer School 2025 - Pre-course part II
3/17


---

## 第10页

... but denoising can be seen as a regression task!
Learned reconstruction for inverse problems:
suppose that y = Ax + ε (forward model): f(y) ≈x ⇝find f ≈A−1.
Denoising: let X = Y and A = Id, i.e.: y = x + ε.
Why not using simply f = A−1 = Id?
The simplest example:
▶X = R2;
▶ε ∼N(0, 0.12 Id);
▶x : uniformly distributed on
{0} × [−1, 1] ∪[−1, 1] × {0};
▶y = x + ε.
▶f = Id
relative error: 20.70%
Luca Ratti
PhD Summer School 2025 - Pre-course part II
3/17


---

## 第11页

... but denoising can be seen as a regression task!
Learned reconstruction for inverse problems:
suppose that y = Ax + ε (forward model): f(y) ≈x ⇝find f ≈A−1.
Denoising: let X = Y and A = Id, i.e.: y = x + ε.
Why not using simply f = A−1 = Id?
The simplest example:
▶X = R2;
▶ε ∼N(0, 0.12 Id);
▶x : uniformly distributed on
{0} × [−1, 1] ∪[−1, 1] × {0};
▶y = x + ε.
▶Variational denoiser: Tikhonov
fT(y)=arg min
x
n 1
2∥x −y∥2
2 + λ
2 ∥x∥2
2
o
=
1
1 + λy
relative error: 19.21%
Luca Ratti
PhD Summer School 2025 - Pre-course part II
3/17


---

## 第12页

... but denoising can be seen as a regression task!
Learned reconstruction for inverse problems:
suppose that y = Ax + ε (forward model): f(y) ≈x ⇝find f ≈A−1.
Denoising: let X = Y and A = Id, i.e.: y = x + ε.
Why not using simply f = A−1 = Id?
The simplest example:
▶X = R2;
▶ε ∼N(0, 0.12 Id);
▶x : uniformly distributed on
{0} × [−1, 1] ∪[−1, 1] × {0};
▶y = x + ε.
▶Variational denoiser: Lasso
fL(y) = proxλ∥·∥1(y) = ST(y; λ)
[fL(y)]j =
(
sign(yj)(|yj| −λ) if |yj| > λ
0 if |yj| ≤λ
relative error: 18.41%
Luca Ratti
PhD Summer School 2025 - Pre-course part II
3/17


---

## 第13页

... but denoising can be seen as a regression task!
Learned reconstruction for inverse problems:
suppose that y = Ax + ε (forward model): f(y) ≈x ⇝find f ≈A−1.
Denoising: let X = Y and A = Id, i.e.: y = x + ε.
Why not using simply f = A−1 = Id?
The simplest example:
▶X = R2;
▶ε ∼N(0, 0.12 Id);
▶x : uniformly distributed on
{0} × [−1, 1] ∪[−1, 1] × {0};
▶y = x + ε.
▶Learned denoiser (ideal case)
fB = . . . - Bayes denoiser
relative error: 16.56%
Luca Ratti
PhD Summer School 2025 - Pre-course part II
3/17


---

## 第14页

Image Denoising as a regression problem
Setup:
▶Each image is a vector x ∈Rn, where n = height × width × channels
▶Noisy image: y = x + ε, Clean image: x, Noise: ε
▶Goal: learn f : Rn →Rn such that f(y) ≈x
Loss Function:
ℓ(f(y), x) = MSE(f(y), x) = 1
n
n
X
j=1
([f(y)]j −[x]j)2 = 1
n∥f(y) −x∥2
2
Luca Ratti
PhD Summer School 2025 - Pre-course part II
4/17


---

## 第15页

Audience-driven slide
In your experience,
what are key ingredients of statistical learning?
Luca Ratti
PhD Summer School 2025 - Pre-course part II
5/17


---

## 第16页

Audience-driven slide
In your experience,
what are key ingredients of statistical learning?
My personal list
A. a training dataset;
B. a loss function;
C. a (parametric) hypothesis class;
D. an optimization algorithm.
Luca Ratti
PhD Summer School 2025 - Pre-course part II
5/17


---

## 第17页

A statistical learning perspective on
image denoising
A. Training datasets


---

## 第18页

Supervised datasets and beyond
1) Supervised setting
A dataset of paired noisy and clean images {(yi, xi)}N
i=1 is available.
▶empirical loss minimization.
Luca Ratti
PhD Summer School 2025 - Pre-course part II
6/17


---

## 第19页

Supervised datasets and beyond
1) Supervised setting
A dataset of paired noisy and clean images {(yi, xi)}N
i=1 is available.
▶empirical loss minimization.
2) Self-supervised setting
A dataset of clean images {(xi)}N
i=1 is available + the noise model is known
(e.g., y = x + ε, ε ∼N(0, σ2I))
▶sample {εi}N
i=1 ∼i.i.d. πε, define yi = xi + εi;
▶empirical loss minimization.
Luca Ratti
PhD Summer School 2025 - Pre-course part II
6/17


---

## 第20页

Supervised datasets and beyond
1) Supervised setting
A dataset of paired noisy and clean images {(yi, xi)}N
i=1 is available.
▶empirical loss minimization.
2) Self-supervised setting
A dataset of clean images {(xi)}N
i=1 is available + the noise model is known
(e.g., y = x + ε, ε ∼N(0, σ2I))
▶sample {εi}N
i=1 ∼i.i.d. πε, define yi = xi + εi;
▶empirical loss minimization.
3) Unsupervised setting - case x
A dataset of clean images {(xi)}N
i=1 is available.
▶learn the prior distribution πx (e.g. via Tweedie’s formula);
▶use a model-adaptive algorithm to identify the noise model.
Luca Ratti
PhD Summer School 2025 - Pre-course part II
6/17


---

## 第21页

Supervised datasets and beyond
1) Supervised setting
A dataset of paired noisy and clean images {(yi, xi)}N
i=1 is available.
▶empirical loss minimization.
2) Self-supervised setting
A dataset of clean images {(xi)}N
i=1 is available + the noise model is known
(e.g., y = x + ε, ε ∼N(0, σ2I))
▶sample {εi}N
i=1 ∼i.i.d. πε, define yi = xi + εi;
▶empirical loss minimization.
3) Unsupervised setting - case x
A dataset of clean images {(xi)}N
i=1 is available.
▶learn the prior distribution πx (e.g. via Tweedie’s formula);
▶use a model-adaptive algorithm to identify the noise model.
4) Unsupervised setting - case y
A dataset of noisy images {(yi)}N
i=1 is available.
▶ad-hoc techniques (when available).
Luca Ratti
PhD Summer School 2025 - Pre-course part II
6/17


---

## 第22页

A statistical learning perspective on
image denoising
B. Loss function


---

## 第23页

Loss minimization and regularization
▶Loss function, ℓ: measures the quality of a single denoised image.
Es.: ℓ(x′, x) = ∥x′ −x∥2
2,
ℓ(x′, x) = PSNR(x’, x).
▶Expected loss, L: measures the quality of a denoiser f, using the full
knowledge of πy,x (ideal).
Es. L(f) = E(y,x)∼π(y,x)[ℓ(f(y), x)].
▶Empirical risk, ˆL: measures the quality of a denoiser f, using a
supervised dataset.
Es. ˆL(f) =
1
m
PN
i=1 ℓ(f(yi), xi).
Luca Ratti
PhD Summer School 2025 - Pre-course part II
7/17


---

## 第24页

Loss minimization and regularization
▶Loss function, ℓ: measures the quality of a single denoised image.
Es.: ℓ(x′, x) = ∥x′ −x∥2
2,
ℓ(x′, x) = PSNR(x’, x).
▶Expected loss, L: measures the quality of a denoiser f, using the full
knowledge of πy,x (ideal).
Es. L(f) = E(y,x)∼π(y,x)[ℓ(f(y), x)].
▶Empirical risk, ˆL: measures the quality of a denoiser f, using a
supervised dataset.
Es. ˆL(f) =
1
m
PN
i=1 ℓ(f(yi), xi).
The risk of overfitting
Minimizing ˆL among all possible f: Y →X
is not a good idea:
▶the result f might work poorly on new
data ym+1, ym+2, . . .
▶the result f might be unstable (∥f∥≫1)
Luca Ratti
PhD Summer School 2025 - Pre-course part II
7/17


---

## 第25页

Loss minimization and regularization
▶Loss function, ℓ: measures the quality of a single denoised image.
Es.: ℓ(x′, x) = ∥x′ −x∥2
2,
ℓ(x′, x) = PSNR(x’, x).
▶Expected loss, L: measures the quality of a denoiser f, using the full
knowledge of πy,x (ideal).
Es. L(f) = E(y,x)∼π(y,x)[ℓ(f(y), x)].
▶Empirical risk, ˆL: measures the quality of a denoiser f, using a
supervised dataset.
Es. ˆL(f) =
1
m
PN
i=1 ℓ(f(yi), xi).
The risk of overfitting
Minimizing ˆL among all possible f: Y →X
is not a good idea:
▶the result f might work poorly on new
data ym+1, ym+2, . . .
▶the result f might be unstable (∥f∥≫1)
Explicit regularization
One possible solution: minimize ˆL(f) + R(f).
Ex. R(f)=∥f∥H, H a suitable function space (e.g. a RKHS⇝kernel methods).
Luca Ratti
PhD Summer School 2025 - Pre-course part II
7/17


---

## 第26页

A statistical learning perspective on
image denoising
C. Parametric hypothesis classes


---

## 第27页

Implicit regularization via parametric hypothesis spaces
Idea: restricting the space F in which optimizing ˆL induces an implicit
regularization. Ex: F = polynomials of degree 10, 5, 1.
Luca Ratti
PhD Summer School 2025 - Pre-course part II
8/17


---

## 第28页

Implicit regularization via parametric hypothesis spaces
Idea: restricting the space F in which optimizing ˆL induces an implicit
regularization. Ex: F = polynomials of degree 10, 5, 1.
Parametric hypothesis spaces
Spaces of functions depending on (p) parameters:
F = {fθ : Y →X,
θ ∈Θ ∼= Rp}.
Examples (X = Y = R):
▶polynomials: fθ(y) = θ1 + θ2y + . . . + θpyp−1;
▶linear splines: fθ(y) = θ1 + θ2χ[θ3,+∞)(y) + . . . θp−1χ[θp,+∞)(y);
▶Neural Networks, e.g. fθ(y) = θ1σ(θ2y) + . . . + θp−1σ(θpy).
Luca Ratti
PhD Summer School 2025 - Pre-course part II
8/17


---

## 第29页

A bias-variance tradeoff
A variation of a picture from [M. Belkina, D. Hsuc, S. Maa, S. Mandal,
Reconciling modern machine-learning practice and the classical
bias–variance trade-off, 2019] - part 1
Number of parameters p
Risk
Test risk
Training risk
under-fitting
over-fitting
Luca Ratti
PhD Summer School 2025 - Pre-course part II
9/17


---

## 第30页

A bias-variance tradeoff
A variation of a picture from [M. Belkina, D. Hsuc, S. Maa, S. Mandal,
Reconciling modern machine-learning practice and the classical
bias–variance trade-off, 2019] - part 2
Number of parameters p
Risk
interpolation threshold
Test risk
Training risk
under-parameterized
over-parameterized
Luca Ratti
PhD Summer School 2025 - Pre-course part II
9/17


---

## 第31页

Neural Networks: very expressive parametric functions from Rn →Rn
Neural Networks: a working expression
Given a nonlinear function σ : R →R and its element-wise version s.t.
[σ(x)]j = σ([x]j), given Wl ∈Rnl−1×nl and bl ∈Rnl, l = 1, . . . , L, let
fθ(x) = WL+1σ(WL · · · σ(W1x + b1) · · · + bL) + bL+1.
The learnable parameters are θ = {W1, . . . , WL, b1, . . . , bL}.
Luca Ratti
PhD Summer School 2025 - Pre-course part II
10/17


---

## 第32页

Neural Networks: very expressive parametric functions from Rn →Rn
Neural Networks: a working expression
Given a nonlinear function σ : R →R and its element-wise version s.t.
[σ(x)]j = σ([x]j), given Wl ∈Rnl−1×nl and bl ∈Rnl, l = 1, . . . , L, let
fθ(x) = WL+1σ(WL · · · σ(W1x + b1) · · · + bL) + bL+1.
The learnable parameters are θ = {W1, . . . , WL, b1, . . . , bL}.
Examples:
▶Multilayer Perceptron (MLP): fully connected layers (full matrices Wl),
σ = ReLU, tanh - target: vector data.
▶Convolutional Neural Network (CNN): convolutional layers (Wlx = Kl ∗X)
- target: image data.
Luca Ratti
PhD Summer School 2025 - Pre-course part II
10/17


---

## 第33页

Neural Networks: very expressive parametric functions from Rn →Rn
Neural Networks: a working expression
Given a nonlinear function σ : R →R and its element-wise version s.t.
[σ(x)]j = σ([x]j), given Wl ∈Rnl−1×nl and bl ∈Rnl, l = 1, . . . , L, let
fθ(x) = WL+1σ(WL · · · σ(W1x + b1) · · · + bL) + bL+1.
The learnable parameters are θ = {W1, . . . , WL, b1, . . . , bL}.
Examples:
▶Multilayer Perceptron (MLP): fully connected layers (full matrices Wl),
σ = ReLU, tanh - target: vector data.
▶Convolutional Neural Network (CNN): convolutional layers (Wlx = Kl ∗X)
- target: image data.
Theorem (Universal Approximation):
Under reasonable assumptions on σ, any continuous function
f : [0, 1]n →Rn can be approximated arbitrarily well (in the L∞norm) by a
NN with L = 1, provided that W1 ∈Rn1×n and W2 ∈Rn×n1, and n1 is
sufficiently large. (Cybenko, 1989; Hornik, 1991)
Variants: arbitrary width, higher regularity, avoid the curse of dimensionality.
Luca Ratti
PhD Summer School 2025 - Pre-course part II
10/17


---

## 第34页

Network architectures for images: Convolutional Neural Networks (CNNs)
Key Idea: exploit spatial locality and translational invariance using local
linear operations.
▶Layers apply convolutions: X 7→σ(K ∗X + b);
▶Common operations: ReLU, pooling, normalization;
▶Weight sharing between layers (parameter reduction).
Credits: Wikipedia, by Aphex34
Luca Ratti
PhD Summer School 2025 - Pre-course part II
11/17


---

## 第35页

Network architectures for images: Convolutional Neural Networks (CNNs)
Key Idea: exploit spatial locality and translational invariance using local
linear operations.
▶Layers apply convolutions: X 7→σ(K ∗X + b);
▶Common operations: ReLU, pooling, normalization;
▶Weight sharing between layers (parameter reduction).
Credits: [Zhang et al., 2017]
Image denoising example:
▶DnCNN (Zhang et al., 2017): deep CNN trained to remove additive
Gaussian noise.
Warning: input dimensions!
In these examples, the input of the network is not a vectorized image
x ∈Rn but a tensor X of size: height × width × channels.
Luca Ratti
PhD Summer School 2025 - Pre-course part II
11/17


---

## 第36页

Network architectures for images: U-Nets
Key Idea: extract and process local features with symmetric skip
connections.
▶An encoder branch extracts
coarse features (convolution +
downsampling)
▶A decoder branch reconstructs
full-resolution output
(upsampling)
▶Skip connections copy feature
maps to enhance spatial details
Credits: [Ronneberger, Fischer, Brox, 15]
Luca Ratti
PhD Summer School 2025 - Pre-course part II
12/17


---

## 第37页

Network architectures for images: U-Nets
Key Idea: extract and process local features with symmetric skip
connections.
▶An encoder branch extracts
coarse features (convolution +
downsampling)
▶A decoder branch reconstructs
full-resolution output
(upsampling)
▶Skip connections copy feature
maps to enhance spatial details
Credits: [Ronneberger, Fischer, Brox, 15]
Image denoising example:
▶U-Net variants are widely used in medical image denoising, see e.g.
Noise2Noise (Lehtinen et al., 2018), DRUNET (Devalla et al., 2018), ...
Luca Ratti
PhD Summer School 2025 - Pre-course part II
12/17


---

## 第38页

Network architectures for images: Vision Transformers
Key Idea: Process images as sequences of patches with self-attention,
removing convolutional inductive bias.
▶Image is split into fixed-size patches (e.g., 16 × 16)
▶Each patch is embedded into a vector (via linear projection)
▶Transformer encoder applies global attention across all patches.
https://www.pinecone.io/learn/series/image-search/vision-transformers/
Image denoising example:
▶Restormer (Zamir et al., 2022): uses self-attention over multi-resolution
image representations; can be applied to unsupervised settings.
Luca Ratti
PhD Summer School 2025 - Pre-course part II
13/17


---

## 第39页

A statistical learning perspective on
image denoising
D. Optimization algorithms


---

## 第40页

How to train your network - part I
Training a network: the process of finding θ that minimizes ˆ
L(θ) = ˆL(fθ).
Luca Ratti
PhD Summer School 2025 - Pre-course part II
14/17


---

## 第41页

How to train your network - part I
Training a network: the process of finding θ that minimizes ˆ
L(θ) = ˆL(fθ).
An optimizer is a numerical algorithm to do so. Key features:
1. first-order schemes: leverages derivatives of ˆ
L in θ;
2. stochastic optimization: it exploits random batches to reduce
computations.
Luca Ratti
PhD Summer School 2025 - Pre-course part II
14/17


---

## 第42页

How to train your network - part I
Training a network: the process of finding θ that minimizes ˆ
L(θ) = ˆL(fθ).
An optimizer is a numerical algorithm to do so. Key features:
1. first-order schemes: leverages derivatives of ˆ
L in θ;
2. stochastic optimization: it exploits random batches to reduce
computations.
Backpropagation: just an intuition
▶Exploit the compositional expression of fθ: repeated chain rule.
▶Use automated differentiation and zero-order methods.
▶Efficiency: proportional to the forward pass (computing fθ)
Luca Ratti
PhD Summer School 2025 - Pre-course part II
14/17


---

## 第43页

How to train your network - part I
Training a network: the process of finding θ that minimizes ˆ
L(θ) = ˆL(fθ).
An optimizer is a numerical algorithm to do so. Key features:
1. first-order schemes: leverages derivatives of ˆ
L in θ;
2. stochastic optimization: it exploits random batches to reduce
computations.
Backpropagation: just an intuition
▶Exploit the compositional expression of fθ: repeated chain rule.
▶Use automated differentiation and zero-order methods.
▶Efficiency: proportional to the forward pass (computing fθ)
Stochastic Gradient Descent: just an intuition
▶Exploit that ˆ
L(θ) = PN
i=1 ℓ(fθ(yi), xi): replace the full sum with the one
on a randomly subsampled batch.
▶Combined with momentum, acceleration, and adaptive step sizes, it
provides efficient versions (Adam, AdaGrad, RMSProp)
Luca Ratti
PhD Summer School 2025 - Pre-course part II
14/17


---

## 第44页

How to train your network - part II
Other keywords involved in the training process of a network:
Luca Ratti
PhD Summer School 2025 - Pre-course part II
15/17


---

## 第45页

How to train your network - part II
Other keywords involved in the training process of a network:
Initialization:
[What?] Weights are initialized randomly (some methods: Xavier, He).
[Why?] Avoid symmetries, preserve variance across layers.
Luca Ratti
PhD Summer School 2025 - Pre-course part II
15/17


---

## 第46页

How to train your network - part II
Other keywords involved in the training process of a network:
Initialization:
[What?] Weights are initialized randomly (some methods: Xavier, He).
[Why?] Avoid symmetries, preserve variance across layers.
Scheduling:
[What?] Dynamically adjusts the learning rate during training.
[Why?] Crucial for stable convergence and escaping local minima.
Luca Ratti
PhD Summer School 2025 - Pre-course part II
15/17


---

## 第47页

How to train your network - part II
Other keywords involved in the training process of a network:
Initialization:
[What?] Weights are initialized randomly (some methods: Xavier, He).
[Why?] Avoid symmetries, preserve variance across layers.
Scheduling:
[What?] Dynamically adjusts the learning rate during training.
[Why?] Crucial for stable convergence and escaping local minima.
Early Stopping:
[What?] Interrupt training when the performance no longer improves.
[Why?] It prevents overfitting and reduces unnecessary computation.
Luca Ratti
PhD Summer School 2025 - Pre-course part II
15/17


---

## 第48页

How to train your network - part II
Other keywords involved in the training process of a network:
Initialization:
[What?] Weights are initialized randomly (some methods: Xavier, He).
[Why?] Avoid symmetries, preserve variance across layers.
Scheduling:
[What?] Dynamically adjusts the learning rate during training.
[Why?] Crucial for stable convergence and escaping local minima.
Early Stopping:
[What?] Interrupt training when the performance no longer improves.
[Why?] It prevents overfitting and reduces unnecessary computation.
Validation set:
[What?] A separate dataset used to evaluate generalization.
[Why?] Hyperparameter tuning (parameters of the network - number of
layers, channels - or of the optimizer - learning rate, batch size).
Luca Ratti
PhD Summer School 2025 - Pre-course part II
15/17


---

## 第49页

Learning, from the errors
Training outcome: eθ (⇝feθ)
obtained by minimizing b
L(θ) over Θ
through a numerical method
Luca Ratti
PhD Summer School 2025 - Pre-course part II
16/17


---

## 第50页

Learning, from the errors
Training outcome: eθ (⇝feθ)
obtained by minimizing b
L(θ) over Θ
through a numerical method
Empirical target: bθ (⇝fbθ)
obtained by minimizing b
L(θ) over Θ
Luca Ratti
PhD Summer School 2025 - Pre-course part II
16/17


---

## 第51页

Learning, from the errors
Training outcome: eθ (⇝feθ)
obtained by minimizing b
L(θ) over Θ
through a numerical method
Empirical target: bθ (⇝fbθ)
obtained by minimizing b
L(θ) over Θ
Optimal target: θ⋆(⇝fθ⋆)
obtained by minimizing L(θ) = L(fθ) over Θ
Luca Ratti
PhD Summer School 2025 - Pre-course part II
16/17


---

## 第52页

Learning, from the errors
Training outcome: eθ (⇝feθ)
obtained by minimizing b
L(θ) over Θ
through a numerical method
Empirical target: bθ (⇝fbθ)
obtained by minimizing b
L(θ) over Θ
Optimal target: θ⋆(⇝fθ⋆)
obtained by minimizing L(θ) = L(fθ) over Θ
Bayes estimator: f⋆= Eπ[x|y = ·]
obtained by minimizing L on all
measurable functions Y →X
Luca Ratti
PhD Summer School 2025 - Pre-course part II
16/17


---

## 第53页

Learning, from the errors
Training outcome: eθ (⇝feθ)
obtained by minimizing b
L(θ) over Θ
through a numerical method
Empirical target: bθ (⇝fbθ)
obtained by minimizing b
L(θ) over Θ
Optimal target: θ⋆(⇝fθ⋆)
obtained by minimizing L(θ) = L(fθ) over Θ
Bayes estimator: f⋆= Eπ[x|y = ·]
obtained by minimizing L on all
measurable functions Y →X
True solution: if there existed a way
to connect y ⇝x
Luca Ratti
PhD Summer School 2025 - Pre-course part II
16/17


---

## 第54页

Learning, from the errors
Training outcome: eθ (⇝feθ)
obtained by minimizing b
L(θ) over Θ
through a numerical method
Empirical target: bθ (⇝fbθ)
obtained by minimizing b
L(θ) over Θ
Optimal target: θ⋆(⇝fθ⋆)
obtained by minimizing L(θ) = L(fθ) over Θ
Bayes estimator: f⋆= Eπ[x|y = ·]
obtained by minimizing L on all
measurable functions Y →X
True solution: if there existed a way
to connect y ⇝x
Optimization error
(how good is my optimizer?)
Luca Ratti
PhD Summer School 2025 - Pre-course part II
16/17


---

## 第55页

Learning, from the errors
Training outcome: eθ (⇝feθ)
obtained by minimizing b
L(θ) over Θ
through a numerical method
Empirical target: bθ (⇝fbθ)
obtained by minimizing b
L(θ) over Θ
Optimal target: θ⋆(⇝fθ⋆)
obtained by minimizing L(θ) = L(fθ) over Θ
Bayes estimator: f⋆= Eπ[x|y = ·]
obtained by minimizing L on all
measurable functions Y →X
True solution: if there existed a way
to connect y ⇝x
Optimization error
(how good is my optimizer?)
Sample error
(how much does my result
depend on the training sample?)
Luca Ratti
PhD Summer School 2025 - Pre-course part II
16/17


---

## 第56页

Learning, from the errors
Training outcome: eθ (⇝feθ)
obtained by minimizing b
L(θ) over Θ
through a numerical method
Empirical target: bθ (⇝fbθ)
obtained by minimizing b
L(θ) over Θ
Optimal target: θ⋆(⇝fθ⋆)
obtained by minimizing L(θ) = L(fθ) over Θ
Bayes estimator: f⋆= Eπ[x|y = ·]
obtained by minimizing L on all
measurable functions Y →X
True solution: if there existed a way
to connect y ⇝x
Optimization error
(how good is my optimizer?)
Sample error
(how much does my result
depend on the training sample?)
Approximation error
(how expressive is my parametric
space of regularizers?)
Luca Ratti
PhD Summer School 2025 - Pre-course part II
16/17


---

## 第57页

Learning, from the errors
Training outcome: eθ (⇝feθ)
obtained by minimizing b
L(θ) over Θ
through a numerical method
Empirical target: bθ (⇝fbθ)
obtained by minimizing b
L(θ) over Θ
Optimal target: θ⋆(⇝fθ⋆)
obtained by minimizing L(θ) = L(fθ) over Θ
Bayes estimator: f⋆= Eπ[x|y = ·]
obtained by minimizing L on all
measurable functions Y →X
True solution: if there existed a way
to connect y ⇝x
Optimization error
(how good is my optimizer?)
Sample error
(how much does my result
depend on the training sample?)
Approximation error
(how expressive is my parametric
space of regularizers?)
Irreducible error
(if data are noisy,
I can’t reduce this!)
Luca Ratti
PhD Summer School 2025 - Pre-course part II
16/17


---

## 第58页

Implementation aspects


---

## 第59页

Image processing in Python
The most important libraries for image processing:
▶numpy: process tensors representing images - can handle arithmetic
operations, cropping...
▶skimage: basic image processing operations (thresholding, blurring,...)
and metrics (MSE, PSNR, SSIM,...)
▶matplotlib: image visualization, color adjustment,...
▶pytorch: neural network - definition, training, testing.
A small tutorial: tomorrow!
Luca Ratti
PhD Summer School 2025 - Pre-course part II
17/17


---

