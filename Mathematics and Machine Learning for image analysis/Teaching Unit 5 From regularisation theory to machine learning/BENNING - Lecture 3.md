# BENNING - Lecture 3.pdf

## 第1页

Faculty of Engineering, Department of Computer Science
Erasmus+ International PhD Summer School 2025 
Mathematics and Machine Learning for Image Analysis 
University of Bologna 
10 June 2025


---

## 第2页

Beyond backpropagation
• Lifted Bregman training of neural networks 
• Regularised inversion of neural networks 
• Inversion with theoretical guarantees? 
• Conclusions & outlook
Part I
Part II


---

## 第3页

Joint work with
Xiaoyu (Victor) Wang 
Heriot-Watt University
Acknowledgements:
Open access papers available
Alexandra Valavanis 
Queen Mary University of London
Audrey Repetti 
Heriot-Watt University
Front. Appl. Math. Stat. 9 2013
JMLR 24(232) 2023
(Inversion)
(Training)
Andreas Mang 
University of Houston
Azhir Mahmood 
University College London


---

## 第4页

Part I: Lifted Bregman training of 
neural networks


---

## 第5页

Training neural networks
An -layer (deep) neural network is a composition of activation functions  and affine-linear transformations 
applied to inputs , to produce outputs , i.e. 
L
σ
x
y
with parameters 
 
Θ = {(Wl, bl)}
L
l=1
y = 𝒩(x, Θ) = σ (WL (⋯σ (W1x + b1)⋯) + bL) ,


---

## 第6页

Training neural networks
An -layer (deep) neural network is a composition of activation functions  and affine-linear transformations 
applied to inputs , to produce outputs , i.e. 
L
σ
x
y
with parameters 
 
Θ = {(Wl, bl)}
L
l=1
Training usually boils down to the (approximate) minimisation of empirical risks of the form
E (Θ) =
s
∑
i=1
y = 𝒩(x, Θ) = σ (WL (⋯σ (W1x + b1)⋯) + bL) ,
1
s
yi 𝒩(xi , Θ)
ℓ( ,
)


---

## 第7页

Training neural networks
An -layer (deep) neural network is a composition of activation functions  and affine-linear transformations 
applied to inputs , to produce outputs , i.e. 
L
σ
x
y
with parameters 
 
Θ = {(Wl, bl)}
L
l=1
Training usually boils down to the (approximate) minimisation of empirical risks of the form
y = 𝒩(x, Θ) = σ (WL (⋯σ (W1x + b1)⋯) + bL) ,
1
2s
yi
𝒩(xi , Θ)
Example:
Mean-Squared Error (MSE)
E (Θ) =
s
∑
i=1
−
2


---

## 第8页

E (Θ) =
s
∑
i=1
Training neural networks
An -layer (deep) neural network is a composition of activation functions  and affine-linear transformations 
applied to inputs , to produce outputs , i.e. 
L
σ
x
y
with parameters 
 
Θ = {(Wl, bl)}
L
l=1
Training usually boils down to the (approximate) minimisation of empirical risks of the form
y = 𝒩(x, Θ) = σ (WL (⋯σ (W1x + b1)⋯) + bL) ,
1
2s
yi
𝒩(xi , Θ)
Example:
Mean-Squared Error (MSE)
−
2


---

## 第9页

Training neural networks
Training usually boils down to the (approximate) minimisation of empirical risks of the form
In the majority of cases, this approximate minimisation is carried out by a combination of
Gradient-based optimisation algorithm 
Gradient computation via backpropagation
E (Θ) =
s
∑
i=1
1
2s
yi
𝒩(xi , Θ)
−
2


---

## 第10页

Training neural networks
Training usually boils down to the (approximate) minimisation of empirical risks of the form
E (Θ) =
s
∑
i=1
1
2s
yi
𝒩(xi , Θ)
−
2
Θk+1 = Θk −τk
s
s
∑
i=1
(JΘ
𝒩(xi ,⋅) (Θk))
⊤
(𝒩(xi , Θk) −yi)
Example: gradient descent
Jacobian of  𝒩(xi,Θ) w.r.t  Θ


---

## 第11页

Training neural networks
Training usually boils down to the (approximate) minimisation of empirical risks of the form
E (Θ) =
s
∑
i=1
1
2s
yi
𝒩(xi , Θ)
−
2
Θk+1 = Θk −τk
s
s
∑
i=1
(JΘ
𝒩(xi ,⋅) (Θk))
⊤
(𝒩(xi , Θk) −yi)
Example: gradient descent
Forward pass
 Backward pass


---

## 第12页

Potential drawbacks of previous approach:
• Backpropagation algorithm is serial in nature 
• Differentiation of activation function  is required
σ
Training neural networks
Alternative:
•
Lifting of parameter space to aid distributed computation 
•
Using novel class of loss functions to avoid differentiation of σ


---

## 第13页

Lifted training of neural networks
E (Θ) =
s
∑
i=1
1
2s
yi
𝒩(xi , Θ)
−
2
Lifted training:
as
E(Θ
rewrite
subject to
xl
i
for all
l ∈{1,…, L}
and
x0
i = xi
2
yi
−
xL
i
=
) = 1
2s
s
∑
i=1
σ(
)
Wlxl−1
i
+ bl


---

## 第14页

Lifted training of neural networks
E (Θ) =
s
∑
i=1
1
2s
yi
𝒩(xi , Θ)
−
2
Lifted training:
with
where 
 and 
.
x0
i = xi
xL
i = yi
The notation  is short-hand for 
X
X = {xl
i}s,L−1
i,l=1
replace
2
•
Miguel Carreira-Perpinan and Weiran Wang. Distributed optimization of deeply nested systems. In Artificial Intelligence and Statistics, pages 
10–19, 2014.  
•
Askari, Armin, Geoffrey Negiar, Rajiv Sambharya, and Laurent El Ghaoui. "Lifted neural networks." arXiv preprint arXiv:1805.01532 (2018).
xl
i −σ(
)
E(Θ
) = 1
2s
s
∑
i=1
L
∑
l=1
, X
Wlxl−1
i
+ bl


---

## 第15页

BΨ (
,
)
Lifted Bregman training of neural networks
E (Θ) =
s
∑
i=1
1
2s
yi
𝒩(xi , Θ)
−
2
Lifted Bregman training:
with
with Bregman / Fenchel loss / penalty function
BΨ(y, z) = 1
2 ∥y∥2 + Ψ(y) + (
1
2∥⋅∥2 + Ψ)
*
(z) −⟨y, z⟩
Xiaoyu Wang, MB. Lifted Bregman Training of neural networks. JMLR 24(232):1—51, 
2023. 
xl
i Wlxl−1
i
+ bl
E(Θ
) = 1
2s
s
∑
i=1
L
∑
l=1
, X
replace


---

## 第16页

Lifted Bregman training of neural networks
Lifted Bregman training:
with Bregman / Fenchel loss / penalty function
BΨ(y, z) = 1
2 ∥y∥2 + Ψ(y) + (
1
2∥⋅∥2 + Ψ)
*
(z) −⟨y, z⟩
What is 
?  
And why would we replace the squared Euclidean 
norm with such a function?
Ψ
Xiaoyu Wang, MB. Lifted Bregman Training of neural networks. JMLR 24(232):1—51, 
2023. 
BΨ (
,
)
xl
i Wlxl−1
i
+ bl
E(Θ
) = 1
2s
s
∑
i=1
L
∑
l=1
, X


---

## 第17页

y = σ(Wx + b)
Lifted Bregman training of neural networks
Here  denotes the (usually nonlinear) activation function of the perceptron
σ
What activation functions do we allow?
Suppose we have samples 
 and want to find 
 such that
(x, y)
W, b


---

## 第18页

-3
-2
-1
0
1
2
3
-3
-2
-1
0
1
2
3
-4
-2
2
4
-1.0
-0.5
0.5
1.0
Suppose we choose common activation functions for our neural network, such as
What do all these functions have in common?
rectifier / ramp
soft-thresholding
hyperbolic tangent
Lifted Bregman training of neural networks


---

## 第19页

All previous activation functions are proximal maps: 
σ(z) = proxΨ(z) := arg min
u∈ℝn {
1
2∥u −z∥2 + Ψ(u)}
for some proper, convex and lower semi-continuous function Ψ : ℝn →ℝ∪{∞}
Example: Ψ(u) = {
0
u ≥0
∞u < 0
⟹
σ(z) = max(0, z)
Moreau, Jean Jacques. "Fonctions convexes duales et points proximaux dans un 
espace hilbertien." (1962).
Lifted Bregman training of neural networks


---

## 第20页

-3
-2
-1
0
1
2
3
x
-3
-2
-1
0
1
2
3
Example: Ψ(u) = α|u|
⟹
σ(z) =
z −α z > α
0
|z| ≤α
z + α z < −α
for some proper, convex and lower semi-continuous function Ψ : ℝn →ℝ∪{∞}
Moreau, Jean Jacques. "Fonctions convexes duales et points proximaux dans un 
espace hilbertien." (1962).
All previous activation functions are proximal maps: 
Lifted Bregman training of neural networks
σ(z) = proxΨ(z) := arg min
u∈ℝn {
1
2∥u −z∥2 + Ψ(u)}


---

## 第21页

-4
-2
2
4
-1.0
-0.5
0.5
1.0
Example: Ψ(u) = {
u tanh−1(u) + 1
2 (log(1 −u2) −u2)
|u| < 1
∞
otherwise
⟹
σ(z) = tanh(z)
for some proper, convex and lower semi-continuous function Ψ : ℝn →ℝ∪{∞}
Combettes, Patrick L., and Jean-Christophe Pesquet. "Deep neural network structures 
solving variational inequalities." Set-Valued and Variational Analysis (2020): 1-28.
All previous activation functions are proximal maps: 
Lifted Bregman training of neural networks
σ(z) = proxΨ(z) := arg min
u∈ℝn {
1
2∥u −z∥2 + Ψ(u)}


---

## 第22页

All previous activation functions are proximal maps: 
for some proper, convex and lower semi-continuous function Ψ : ℝn →ℝ∪{∞}
Hasannasab, M., Hertrich, J., Neumayer, S., Plonka, G., Setzer, S., & Steidl, G. (2020). Parseval proximal neural 
networks. Journal of Fourier Analysis and Applications, 26, 1-31. 
Combettes, Patrick L., and Jean-Christophe Pesquet. "Deep neural network structures solving variational 
inequalities." Set-Valued and Variational Analysis (2020): 1-28. 
Hertrich, J., Neumayer, S., & Steidl, G. (2021). Convolutional proximal neural networks and plug-and-play algorithms. 
Linear Algebra and its Applications, 631, 203-234. 
Le, H. T. V., Repetti, A., & Pustelnik, N. (2023). PNN: From proximal algorithms to robust unfolded image denoising 
networks and Plug-and-Play methods. arXiv preprint arXiv:2308.03139. 
and many many more…
Lifted Bregman training of neural networks
Lots of works focus on proximal maps as activation functions, e.g.
σ(z) = proxΨ(z) := arg min
u∈ℝn {
1
2∥u −z∥2 + Ψ(u)}


---

## 第23页

y = σ(Wx + b)
Wx + b
(y)
∈∂Ψ
−y
⟺
∂Ψ(y) = {p
Ψ(z) ≥Ψ(y) + ⟨p, z −y⟩, ∀z}
where
is the subdifferential of Ψ
⟺
y = arg min
z
{
1
2∥z −(Wx + b)∥2 + Ψ(z)}
Lifted Bregman training of neural networks
= proxΨ(Wx + b)
Suppose we have samples 
 and want to find 
 such that
(x, y)
W, b


---

## 第24页

1
2∥y∥2 + Ψ(y) + (
1
2∥⋅∥2 + Ψ)
*
(
(y)
∈∂(
1
2 ∥⋅∥2 +
)
Ψ
⟺
⟺
Wx + b
Legendre, Adrien-Marie. Mémoire sur l'intégration de quelques équations aux différences partielles. In 
Histoire de l'Académie royale des sciences, avec les mémoires de mathématique et de physique. Paris: 
Imprimerie royale. pp. 309–351, 1789 
Werner Fenchel, Convex cones, sets, and functions. Princeton University, 1953 
Theorem 23.5,  Ralph Tyrell Rockafellar, Convex analysis, Princeton university press, 1970
y = σ(Wx + b)
where
(
1
2 ∥⋅∥2 + Ψ)
*
(z) = sup
x {⟨x, z⟩−1
2∥x∥2 −Ψ(x)}
) = ⟨y,
⟩
Wx + b
Wx + b
Lifted Bregman training of neural networks
Suppose we have samples 
 and want to find 
 such that
(x, y)
W, b


---

## 第25页

with Bregman / Fenchel function
1
2 ∥y∥2 + Ψ(y) + (
1
2∥⋅∥2 + Ψ)
*
(
)
⟨y,
⟩
z
z
−
BΨ(y, z) =
Wang, X., & MB. A Lifted Bregman Formulation for the Inversion of Deep Neural 
Networks. Front. Appl. Math. Stat. 9, (2023).
Lifted Bregman training of neural networks
Lifted Bregman network
BΨ (
,
)
E(Θ, X) = 1
2s
s
∑
i=1
L
∑
l=1
xl
i Wl xl−1
i
+ bl


---

## 第26页

with Bregman / Fenchel function
What is great about this function?
1.
, for all 
 
2.
 
3.
 for 
BΨ(y, z) = 1
2∥y −proxΨ(z)∥2 + DproxΨ*(z)
Ψ
(y, proxΨ(z)) ≥1
2∥y −proxΨ(z)∥2 ≥0
y, z
∇2BΨ(y, z) = proxΨ(z) −y
BΨ(y, z) = Ez(y) −Ez(proxΨ(z)) = D0
Ez(y, proxΨ(z))
Ez(u) := 1
2∥u −z∥2 + Ψ(u)
1
2 ∥y∥2 + Ψ(y) + (
1
2∥⋅∥2 + Ψ)
*
(
)
⟨y,
⟩
z
z
−
BΨ(y, z) =
Xiaoyu Wang, MB. Lifted Bregman Training of neural networks. JMLR 24(232):1—51, 2023. 
Lifted Bregman training of neural networks
Lifted Bregman network
BΨ (
,
)
E(Θ, X) = 1
2s
s
∑
i=1
L
∑
l=1
xl
i Wl xl−1
i
+ bl


---

## 第27页

Optimality conditions for 
 and 
:
Wj
bj
0 = (σ (Wj xj−1 + bj) −xj) x⊤
j−1
0 = σ (Wj xj−1 + bj) −xj
Lifted Bregman network
with Bregman / Fenchel loss
Xiaoyu Wang, MB. Lifted Bregman Training of neural networks. JMLR 24(232):1—51, 2023. 
BΨ (
,
)
E(Θ, X) = 1
2s
s
∑
i=1
L
∑
l=1
xl
i
Lifted Bregman training of neural networks
Wl xl−1
i
+ bl
1
2 ∥y∥2 + Ψ(y) + (
1
2∥⋅∥2 + Ψ)
*
(
)
⟨y,
⟩
z
z
−
BΨ(y, z) =


---

## 第28页

Lifted Bregman training of neural networks
1
2 |σ(z) −1/2|2
BΨ(1/2,z)
vs
Illustration:
σ(z) = max(0,z)
σ(z) = tanh(z)
σ(z) = soft-thresholding(z,1/2)


---

## 第29页

© Wikimedia commons
Sparse (denoising) autoencoder toy example
Fashion MNIST 
image codec
Images are centred
Numerical results
©becominghuman.ai
Xiao, H., Rasul, K. and Vollgraf, R., 2017. Fashion-mnist: a novel image dataset for 
benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747.


---

## 第30页

Sparse (denoising) autoencoder toy example
Numerical results
© Wikimedia commons
min
s
∑
i=1
5
∑
j=1
BΨj (xi
j, Wjxi
j−1 + bj)
+ α
xi
3
1
Idea: make encoding sparse
Network architecture:
σj(z) =
max(0,z)
j ∈{1,2,4}
soft-thresholding(z, ρ)
j = 3
z
j = 5
Layer dimensions 784-784-784-784-784


---

## 第31页

Numerical results
Sparse 


---

## 第32页

Numerical results
Trained on 1000 images
Trained on 10000 images
LBN minimisation via implicit SGD and proximal gradient descent for subproblems*
*implementation details are on extra slide for the Q&A if anyone is interested


---

## 第33页

Numerical results
Example: Proximal Neural Networks (PNNs) for image denoising
uk+1 = proxτkg* (uk −τkL(L*uk −z))
̂x = arg min
x∈ℝn {
1
2∥x −z∥2 + g(Lx)}
A more traditional way to solve denoising problems is using proximal maps of the form
This problem can for instance be solved with the dual forward backward algorithm*, i.e.
̂x = z −lim
k→∞L*uk
for k = 0,1,…
*P. L. Combettes, Đ. Dũng, and B. C. Vũ, “Dualization of signal recovery problems,” Set-Valued Var. Anal., vol. 
18, no. 3, pp. 373–404, 2010.
Alternatively, one can unroll the algorithm for a fixed no. of iterations 
 and learn trainable parameters 
, i.e.
k*
Lk
uk+1 = proxτkg* (uk −τkLk (L*k uk −z))
xk* = z −L*k* uk*
for k = 0,1,…, k* −1


---

## 第34页

Numerical results
Example: Proximal Neural Networks (PNNs) for image denoising
Alternatively, one can unroll the algorithm for a fixed no. of iterations 
 and learn trainable parameters 
, i.e.
k*
Lk
uk+1 = proxτkg* (uk −τkLk (L*k uk −z))
xk* = z −L*k* uk*
Approach perfectly suits lifted Bregman approach, i.e.
min
s
∑
i=1 [ℓ(zi −L*k* ui
k*, xi) +
k*−1
∑
k=0
Bτkg* (ui
k+1, ui
k −τkLk (L*k ui
k −zi))]
X. Wang, MB, A. Repetti, A lifted Bregman strategy for training unfolded proximal neural 
network gaussian denoisers, in: 2024 IEEE 34th International Workshop on Machine 
Learning for Signal Processing (MLSP), IEEE, 2024, pp. 1–6. 
for k = 0,1,…, k* −1


---

## 第35页

Numerical results
Example: Proximal Neural Networks (PNNs) for image denoising
X. Wang, MB, A. Repetti, A lifted Bregman strategy for training unfolded proximal neural 
network gaussian denoisers, in: 2024 IEEE 34th International Workshop on Machine 
Learning for Signal Processing (MLSP), IEEE, 2024, pp. 1–6. 


---

## 第36页

Part II: Regularised inversion of 
neural networks 


---

## 第37页

N(
)
We consider the (deterministic) inverse 
problems of the form
where the goal is to recover      for given 
data
u†
f
u†
f
=
∈range(N)
u† = unknown solution
f
= measured data
 is a neural network
N
Engl, H. W., Hanke, M., & Neubauer, A. (1996). Regularization of inverse problems (Vol. 375). 
Springer Science & Business Media.
Benning, M., & Burger, M. (2018). Modern regularization methods for inverse problems. Acta 
Numerica, 27, 1-111.
Inverting neural networks


---

## 第38页

where the goal is to recover      for given 
data
f δ
f δ
f δ = measured data
Engl, H. W., Hanke, M., & Neubauer, A. (1996). Regularization of inverse problems (Vol. 375). 
Springer Science & Business Media.
Benning, M., & Burger, M. (2018). Modern regularization methods for inverse problems. Acta 
Numerica, 27, 1-111.
u† = unknown solution
N(
)
u† =
u†
 is a neural network
N
Inverting neural networks
We consider the (deterministic) inverse 
problems of the form


---

## 第39页

Example:
Inverting neural networks


---

## 第40页

Example:
Inverting neural networks


---

## 第41页

Example:
Inverting neural networks


---

## 第42页

Example:
Inverting neural networks


---

## 第43页

Example:
Why is this interesting?
Toy problem:
Train  such that 
A
A(u†
i ) ≈u†
i
Wang, X., & MB. A Lifted Bregman Formulation for the Inversion of Deep Neural 
Networks. Front. Appl. Math. Stat. 9, (2023).
Simple autoencoder A(u) = W2
+ b2
max (0,W1u + b1)
u†
i
A(u†
i )


---

## 第44页

Example:
Why is this interesting?
A(u) = W2
+ b2
Toy problem:
Train  such that 
A
A(u†
i ) ≈u†
i
max (0,W1u + b1)
u†
i
A(u†
i )
Wang, X., & MB. A Lifted Bregman Formulation for the Inversion of Deep Neural 
Networks. Front. Appl. Math. Stat. 9, (2023).
Simple autoencoder
Solve 
 for 
R(N(u†
i )) ≈u†
i
N(u) = max (0,W1u + b1)
R(N(u†
i ))
 does not require training and 
only depends on pre-trained 
R
N
We can improve pre-trained 
decoders by replacing them 
with reconstruction methods


---

## 第45页

Example:
Why is this interesting?
nonlinear inverse problems
From Alexander Denker, Johannes Hertrich, Zeljko Kereta, Silvia Cipiccia, Ecem Erin, and Simon Arridge. Plug-and-
play half-quadratic splitting for ptychography. arXiv preprint arXiv:2412.02548, 2024.
f = |ℱ(u†)|
Ptychography


---

## 第46页

Example:
Why is this interesting?
nonlinear inverse problems
From Alexander Denker, Johannes Hertrich, Zeljko Kereta, Silvia Cipiccia, Ecem Erin, and Simon Arridge. Plug-and-
play half-quadratic splitting for ptychography. arXiv preprint arXiv:2412.02548, 2024.
f = |ℱ(u†)|
Idea: replace non-linearity 
 with neural network approximation 
 and solve 
 instead
| ⋅|
N
f = N(ℱu†)
Ptychography


---

## 第47页

Inversion of neural networks
How can we invert neural networks?
We can design another neural network  to approximate the inverse of 
: 
R
N
R(N(u†)) ≈u†
Neural network (Encoder)
Regularisation (Decorder)


---

## 第48页

Inversion of neural networks
How can we invert neural networks?
We can design another neural network  to approximate the inverse of 
: 
R
N
R(N(u†)) ≈u†
Neural network (Encoder)
Regularisation (Decorder)
We choose neural networks such as
N(u) = proxΨ(Wu + b)
Perceptron
Proximal map


---

## 第49页

Inversion of neural networks
How can we invert neural networks?
We can design another neural network  to approximate the inverse of 
: 
R
N
R(N(u†)) ≈u†
Neural network (Encoder)
Regularisation (Decorder)
We choose neural networks such as
N(u) = proxΨ(Wu + b)
Perceptron


---

## 第50页

Inversion of neural networks
How can we invert neural networks?
We can design another neural network  to approximate the inverse of 
: 
R
N
R(N(u†)) ≈u†
Neural network (Encoder)
Regularisation (Decorder)
We choose neural networks such as
N(u) = Wl proxΨl−1(Wl−1⋯W2proxΨ1(W1u + b1) + b2)⋯bl−1) + bl
Feed-forward networks


---

## 第51页

Inversion of neural networks
How can we invert neural networks?
We can design another neural network  to approximate the inverse of 
: 
R
N
R(N(u†)) ≈u†
Neural network (Encoder)
Regularisation (Decorder)
We choose neural networks such as
N(u) = Wl ul−1 + bl
Residual neural networks
uj = uj−1 + Vj−1proxΨj−1 (Wj−1uj−1 + bj−1)


---

## 第52页

Inversion of neural networks
How can we invert neural networks?
We can design another neural network  to approximate the inverse of 
: 
R
N
R(N(u†)) ≈u†
Neural network (Encoder)
Regularisation (Decorder)
All previous networks can be written in compact form as
N(u) = Ku
Mu = VproxΨ (Wu + b)
tensor representation of 
all layers in one variable


---

## 第53页

Inversion of neural networks
How can we invert neural networks?
We can design another neural network  to approximate the inverse of 
: 
R
N
R(N(u†)) ≈u†
or more like
R( f δ) →u†
for
f δ →f = N(u†)
when
δ →0
Open questions:
• What architecture should we choose for  ? 
• Can we treat 
 as a black box or do we need to know its architecture and 
parameters when we construct  ? 
• Do we need to train , possibly from scratch? 
• Do we have any mathematical guarantees that  approximates the inverse of 
 ?
R
N
R
R
R
N


---

## 第54页

Inversion of neural networks
We can design another neural network  to approximate the inverse of 
: 
R
N
R(N(u†)) ≈u†
Open questions:
• What architecture should we choose for  ? 
• Can we treat 
 as a black box or do we need to know its architecture and 
parameters when we construct  ? 
• Do we need to train , possibly from scratch? 
• Do we have any mathematical guarantees that  approximates the inverse of 
 ?
R
N
R
R
R
N
Example:
Neural network with arbitrary architecture
No idea
Yes
Yes
No
R( f δ) = hl (hl−1 (⋯h1( f δ, p1), ⋯, pl−1), pl)
Activation functions h1, …, hl
Parameters p1, …, pl


---

## 第55页

J(u)}
Inversion of neural networks
We can design another neural network  to approximate the inverse of 
: 
R
N
R(N(u†)) ≈u†
Open questions:
• What architecture should we choose for  ? 
• Can we treat 
 as a black box or do we need to know its architecture and 
parameters when we construct  ? 
• Do we need to train , possibly from scratch? 
• Do we have any mathematical guarantees that  approximates the inverse of 
 ?
R
N
R
R
R
N
Example:
Variational regularisation with quadratic fidelity
Some ideas
No*
No
Possibly
R( f δ)
*Usually requires computation of backward-pass 
; and can be as challenging as if one were to use 
 directly
(∇N)*
K
∈arg min
u {
1
2 ∥N(u) −f δ∥2 +α


---

## 第56页

DJ(u, uk) }
∈arg min
u {
Inversion of neural networks
We can design another neural network  to approximate the inverse of 
: 
R
N
R(N(u†)) ≈u†
Open questions:
• What architecture should we choose for  ? 
• Can we treat 
 as a black box or do we need to know its architecture and 
parameters when we construct  ? 
• Do we need to train , possibly from scratch? 
• Do we have any mathematical guarantees that  approximates the inverse of 
 ?
R
N
R
R
R
N
Example:
Iterative regularisation with quadratic fidelity
Some ideas
No*
No
Possibly
*Usually requires computation of backward-pass 
; and can be as challenging as if one were to use 
 directly
(∇N)*
K
R( f δ) = uk*
for
+ stopping criterion
uk+1
1
2 ∥N(u) −f δ∥2 +α


---

## 第57页

}
∈arg min
u {
Inversion of neural networks
We can design another neural network  to approximate the inverse of 
: 
R
N
R(N(u†)) ≈u†
Open questions:
• What architecture should we choose for  ? 
• Can we treat 
 as a black box or do we need to know its architecture and 
parameters when we construct  ? 
• Do we need to train , possibly from scratch? 
• Do we have any mathematical guarantees that  approximates the inverse of 
 ?
R
N
R
R
R
N
Example:
Variational regularisation with bespoke fidelity
Several options
No
No
Yes
In the following, we will derive a suitable candidate for this bespoke data fidelity term
J(u)
R( f δ)
Bespoke (𝒩(u), f δ) +α


---

## 第58页

Inversion of neural networks
How can we invert neural networks?
We can design another neural network  to approximate the inverse of 
: 
R
N
R(N(u†)) ≈u†
Neural network (Encoder)
Regularisation (Decorder)
One possible choice for :
R
(uρ, zρ) ∈arg min
u,z {Eρ
Ψ(u, z) + J(u)}
(variational regularisation)


---

## 第59页

with regularisation function  and data fidelity 
 defined as
J
Eρ
Ψ
Eρ
Ψ(u, z) = λ
2 ∥Ku −f δ∥2 + BΨ(z, Wu + b) + χ=0(Mu −Vz) + ρ
2 ∥Mu −Vz∥2
Inversion of neural networks
One possible choice for :
R
(uρ, zρ) ∈arg min
u,z {Eρ
Ψ(u, z) + J(u)}
(variational regularisation)
with Fenchel / Bregman penalty function
BΨ(z, x) = (
1
2∥⋅∥2 + Ψ)(z) + (
1
2 ∥⋅∥2 + Ψ)
*
(x) −⟨z, x⟩


---

## 第60页

with regularisation function  and data fidelity 
 defined as
J
Eρ
Ψ
Eρ
Ψ(u, z) = λ
2 ∥Ku −f δ∥2 + BΨ(z, Wu + b) + χ=0(Mu −Vz) + ρ
2 ∥Mu −Vz∥2
Inversion of neural networks
One possible choice for :
R
(uρ, zρ) ∈arg min
u,z {Eρ
Ψ(u, z) + J(u)}
(variational regularisation)
with Fenchel / Bregman penalty function
BΨ(z, x) = (
1
2 ∥⋅∥2 + Ψ)(z) + (
1
2∥⋅∥2 + Ψ)
*
(x) −⟨z, x⟩
BΨ(z, Wx + b) = 0
⟺
z = proxΨ(Wx + b)
χ=0(Mu −Vz) = 0
⟺
Mu = Vz


---

## 第61页

Inversion of neural networks
Example:
Shallow two-layer neural networks (or linear combinations of 1d perceptrons)
N(u) =
m
∑
j=1
cj
u, wj, bj, cj ∈ℝ
proxΨj (
)
wj u + bj


---

## 第62页

Inversion of neural networks
Example:
Shallow two-layer neural networks (or linear combinations of 1d perceptrons)
u, wj, bj, cj ∈ℝ
uj =
∀j ∈{1,…, m}
N(u) =
m
∑
j=1
cj
proxΨj (
)
wj u + bj
uj


---

## 第63页

Inversion of neural networks
Example:
Shallow two-layer neural networks (or linear combinations of 1d perceptrons)
Corresponding variational regularisation framework:
uα ∈arg min
u∈ℝ1+m
1
2
f δ −
m
∑
j=1
cjuj
2
+
m
∑
j=1
BΨj(uj, wju0 + bj) + αJ(u0, u1, …, um)
Implicit/explicit coordinate descent implementation for choice J(u0, u1, …, um) = 1
2 |u0|2
u, wj, bj, cj ∈ℝ
uj =
∀j ∈{1,…, m}
N(u) =
m
∑
j=1
cj
proxΨj (
)
wj u + bj
uj


---

## 第64页

Inversion of neural networks
Implicit/explicit coordinate descent implementation for choice J(u0, u1, …, um) = 1
2 |u0|2
Corresponding variational regularisation framework:
uk+1
l
= prox(1+c2l )−1 Ψl
cl (f δ −∑l−1
j=1 cjuk+1
j
−∑m
j=l+1 cjuk
j ) + wluk
0 + bl
1 + c2l
uk+1
0
= (1 + α/∥w∥2)−1 uk
0 −∥w∥−2
m
∑
j=1
wj (proxΨj(wjuk
0 + bj) −uk+1
j
)
∀l ∈{1,…, m}
uα ∈arg min
u∈ℝ1+m
1
2
f δ −
m
∑
j=1
cjuj
2
+
m
∑
j=1
BΨj(uj, wju0 + bj) + αJ(u0, u1, …, um)


---

## 第65页

Inversion of neural networks
Decoder:
uk+1
l
= prox(1+c2l )−1 Ψl
cl (f δ −∑l−1
j=1 cjuk+1
j
−∑m
j=l+1 cjuk
j ) + wluk
0 + bl
1 + c2l
uk+1
0
= (1 + α/∥w∥2)−1 uk
0 −∥w∥−2
m
∑
j=1
wj (proxΨj(wjuk
0 + bj) −uk+1
j
)
∀l ∈{1,…, m}
Encoder:
R( f δ) = (u*0
u*1
⋯
u*m)
⊤
where 
 are solutions of the fixed-point iteration 
u*0 , u*1 , ⋯, u*m
N(u) =
m
∑
j=1
cj proxj (
)
wj u + bj


---

## 第66页

Inversion of neural networks
Example:
Ψj(v) = {
0
v ≥0
∞v < 0
⇒
proxΨj(z) = ReLU(z) = max(0,z)
N(u) =
m
∑
j=1
cj
wj = 1, ∀j ∈{1,…,25}
m = 25
proxj (
)
wj u + bj
α = 10−4
bj = −(j −1)h,
j ∈{1,…,25},
h = 3/50


---

## 第67页

Inversion of neural networks
Example:
N(u) =
m
∑
j=1
cj
wj = 1, ∀j ∈{1,…,25}
m = 25
max(0,
)
wj u + bj
Function K(u) = u(
2
3 u)
3u
α = 10−4
bj = −(j −1)h,
j ∈{1,…,25},
h = 3/50


---

## 第68页

Inversion of neural networks
Example:
N(u) =
m
∑
j=1
cj
wj = 1, ∀j ∈{1,…,25}
m = 25
max(0,
)
wj u + bj
Coefficients
α = 10−4
bj = −(j −1)h,
j ∈{1,…,25},
h = 3/50


---

## 第69页

Inversion of neural networks
Example:


---

## 第70页

Inversion of neural networks
Example:


---

## 第71页

Inversion of neural networks
Example:


---

## 第72页

Inversion of neural networks
Example:


---

## 第73页

Inversion of neural networks
Example:
Residual neural networks
N(u) = ul
uk = uk−1 + W⊤
k proxΨk(Wkuk−1 + bk)
∀k ∈{1,…, l}
Corresponding variational regularisation framework:
uα ∈arg min
u {
λ
2 ∥Ku −f δ∥2 + BΨ(z, Wu + b) + J(u)}
for
M =
−I
I
0
⋯
0
0
−I
I
0
⋮
⋱
⋱
⋮
⋮
−I
I
0
0
⋯
0
0
W =
W1
0
0
⋯
0
0
0
W2 0
⋯
0
0
⋮
⋱
⋮
⋮
0
0
0
⋯
Wl
0
0
0
0
⋯
0
0
b =
b1
b2
⋮
bl
0
subject to
l
∑
k=0
Ψk(zk)
Mu = W⊤z
Ψ(z0, …, zl) =


---

## 第74页

Inversion of neural networks
Example:
l = 20
Function K(u) = u3 + u


---

## 第75页

Inversion with theoretical 
guarantees? 


---

## 第76页

Inversion with theoretical guarantees?
Can we provide some theoretical properties for the objective function
or the regularisation operator?
R( f δ) ∈arg min
u {
λ
2 ∥Ku −f δ∥2 + BΨ(z, Wu + b) + χ=0(Mu −Vz) + ρ
2 ∥Mu −Vz∥2 + J(u)}
Eρ
Ψ(u, z) = λ
2 ∥Ku −f δ∥2 + BΨ(z, Wu + b) + χ=0(Mu −Vz) + ρ
2 ∥Mu −Vz∥2


---

## 第77页

Inversion with theoretical guarantees?
Can we provide some theoretical properties for the objective function
R( f δ) ∈arg min
u {
λ
2 ∥Ku −f δ∥2 + BΨ(z, Wu + b) + χ=0(Mu −Vz) + ρ
2 ∥Mu −Vz∥2 + J(u)}
Eρ
Ψ(u, z) = λ
2 ∥Ku −f δ∥2 + BΨ(z, Wu + b) + χ=0(Mu −Vz) + ρ
2 ∥Mu −Vz∥2


---

## 第78页

Inversion with theoretical guarantees?
Can we provide some theoretical properties for the objective function
A sufficient condition for convexity is 
.
⟨∂Eρ
Ψ(u1, z2) −∂Eρ
Ψ(u2, z2), (
u1
z1) −(
u2
z2)⟩≥0
It can be shown that for 
 a sufficient condition for achieving this inequality for all 
 is
V = W⊤
u1, u2, z1, z2
Q := λK⊤K −ρ−1I −M −M⊤⪰0
Eρ
Ψ(u, z) = λ
2 ∥Ku −f δ∥2 + BΨ(z, Wu + b) + χ=0(Mu −Vz) + ρ
2 ∥Mu −Vz∥2
uj = uj−1 + W⊤
j−1proxΨj−1 (Wj−1uj−1 + bj−1)
Example:
⟹
 is positive semi-definite
Q


---

## 第79页

Inversion with theoretical guarantees?
uj = uj−1 −W⊤
j−1proxΨj−1 (Wj−1uj−1 + bj−1)
Example:
f(x) = x + x3


---

## 第80页

Inversion with theoretical guarantees?
or the regularisation operator?
R( f δ) ∈arg min
u {
λ
2 ∥Ku −f δ∥2 + BΨ(z, Wu + b) + χ=0(Mu −Vz) + ρ
2 ∥Mu −Vz∥2 + J(u)}
Eρ
Ψ(u, z) = λ
2 ∥Ku −f δ∥2 + BΨ(z, Wu + b) + χ=0(Mu −Vz) + ρ
2 ∥Mu −Vz∥2


---

## 第81页

Inversion with theoretical guarantees?
or the regularisation operator?
R( f δ) ∈arg min
u {
λ
2 ∥Ku −f δ∥2 + BΨ(z, Wu + b) + χ=0(Mu −Vz) + ρ
2 ∥Mu −Vz∥2 + J(u)}
Eρ
Ψ(u, z) = λ
2 ∥Ku −f δ∥2 + BΨ(z, Wu + b) + χ=0(Mu −Vz) + ρ
2 ∥Mu −Vz∥2
No general results yet, but for
R( f δ) ∈arg min
u {BΨ( f δ, Wu + b) + αJ(u)}
we can show the following


---

## 第82页

Theorem:
DJ(u†, R( f δ)) ≤2δ2
α + α∥v†∥2 + 1
α (Ψ (f δ + αv†) + Ψ (f δ −αv†) −2Ψ( f δ))
Here 
 denotes the Bregman distance w.r.t. .
DJ
J
Burbea Rao divergence 
between 
 and 
f δ + αv†
f δ −αv†
suppose we have  = 
 and 
 and 
 satisfies the source 
condition 
. Then, a solution 
 satisfies
f
proxΨ(Wu† + b)
BΨ( f δ, Wu† + b) ≤δ2
u†
W⊤v† ∈∂J(u†)
uα ∈arg min
u {BΨ( f δ, Wu + b) + αJ(u)}
Wang, X., & MB. A Lifted Bregman Formulation for the Inversion of Deep Neural 
Networks. Front. Appl. Math. Stat. 9, (2023).
Inversion with theoretical guarantees?
Classical error estimate


---

## 第83页

Example
f = max(0,Wu† + b)
Ψ(z) = {
0
z ≥0
∞else
⇒
⇒
Ψ (f δ + αv†) + Ψ (f δ −αv†) −2Ψ( f δ) = 0
if v†
j ∈[−
f δ
j
α ,
f δ
j
α ]
If we choose 
, then we observe 
α(δ) = δ
2/∥v†∥
DJ (u†, uα(δ)) ≤2
2 ∥v†∥δ
= C
Inversion with theoretical guarantees?
Wang, X., & MB. A Lifted Bregman Formulation for the Inversion of Deep Neural 
Networks. Front. Appl. Math. Stat. 9, (2023).
⟶δ→0 0


---

## 第84页

Example:
ReLU Perceptron
N(u) = max(0,Wu† + b)
Ground truth u†
Data f
W : ℝ64×64 →ℝ512
b ∈ℝ512
Random entries 
with zero mean and std one 
Wang, X., & MB. A Lifted Bregman Formulation for the Inversion of Deep Neural 
Networks. Front. Appl. Math. Stat. 9, (2023).
Inversion with theoretical guarantees?


---

## 第85页

Data f δ
Random entries 
with zero mean and std one 
Wang, X., & MB. A Lifted Bregman Formulation for the Inversion of Deep Neural 
Networks. Front. Appl. Math. Stat. 9, (2023).
Example:
ReLU Perceptron
N(u) = max(0,Wu† + b)
Ground truth u†
Inversion with theoretical guarantees?
W : ℝ64×64 →ℝ512
b ∈ℝ512


---

## 第86页

Reconstruction R( f δ)
Data f δ
Random entries 
with zero mean and std one 
α = 0.015
J(u) = ∑
i=1 ∑
j=1
|∇u|2
i,j,1 + |∇u|2
i,j,2
Wang, X., & MB. A Lifted Bregman Formulation for the Inversion of Deep Neural 
Networks. Front. Appl. Math. Stat. 9, (2023).
Example:
ReLU Perceptron
N(u) = max(0,Wu† + b)
Inversion with theoretical guarantees?
W : ℝ64×64 →ℝ512
b ∈ℝ512


---

## 第87页

α = 9 × 10−3
Six-layer convolutional autoencoder. Invert code with TV-based variational regularisation
Dimension of code is 300
u†
A(u†)
R( f δ)
J(u) = ∑
i=1 ∑
j=1
|∇u|2
i,j,1 + |∇u|2
i,j,2
Wang, X., & MB. A Lifted Bregman Formulation for the Inversion of Deep Neural 
Networks. Front. Appl. Math. Stat. 9, (2023).
Inversion of neural networks
Example:


---

## 第88页

Conclusions & outlook


---

## 第89页

Conclusions & outlook
Conclusions:
Outlook:
we have
• Apply approach to real-world scenarios (blind deconvolution etc.) 
• Extend concepts to different architectures 
• Prove convergence results for architectures more complex than perceptrons 
• Explore parallel or distributed computing frameworks
• introduced a novel lifted training approach for feed-forward networks 
• shown that novel approach avoids differentiating activation functions 
• shown that approach can be used for inversion of neural networks (decoder without training!) 
• demonstrated that approach works empirically with numerical experiments 
• proven that for one layer we have a convergent regularisation method


---

## 第90页

Thank you for your attention!
Acknowledgements:
Relevant open access research papers (more to come)
Lifted Bregman Training
Lifted Bregman Inversion
Front. Appl. Math. Stat. 9 2013
JMLR 24(232) 2023


---

## 第91页

Implementation
We minimise 
BΨ (
,
E(Θ, X) = 1
2s
s
∑
i=1
L
∑
l=1
xl
i Wl xl−1
i
+bl)


---

## 第92页

BΨ (
,
)
Implementation
We minimise 
E(Θ, X) = 1
2s
s
∑
i=1
L
∑
l=1
xl
i Wl xl−1
i


---

## 第93页

We minimise 
via a combination of an implicit stochastic gradient method*
(Θk+1, Xk+1) = arg min
Θ,X
1
|Sp| ∑
i∈Sp [
L
∑
l=1
BΨ (xi
l, Wlxi
l−1) + 1
2τk ∥Wl −Wk
l ∥2
]
with random batch 
 and proximal gradient descent** for the inner problem:
Sp
BΨ (
,
)
E(Θ, X) = 1
2s
s
∑
i=1
L
∑
l=1
xl
i Wl xl−1
i
Wj+1
l
= Wj
l −
γj
l
|Sp|
∑
i∈Sp
[σ (Wj
l (xi
l−1)j) (xi
l−1)
⊤
j ] + 1
τk (Wj
l −Wk
l )
(xi
l)
j = proxμj
l(
1
2 ∥⋅∥2+ Ψ) ((xi
l)
j −μj
l ((Wj
l)
⊤
(σ (Wj
l (xi
l)
j
) −(xi
l+1)
j
) −Wj
l (xi
l−1)
j
))
Implementation


---

## 第94页

We minimise  via a combination of an implicit stochastic gradient method*
E
(Θk+1, Xk+1) = arg min
Θ,X
1
|Sp| ∑
i∈Sp [
L
∑
l=1
BΨ (xi
l, Wlxi
l−1) + 1
2τk ∥Wl −Wk
l ∥2
]
with random batch 
 and proximal gradient descent** for the inner problem:
Sp
Wj+1
l
= Wj
l −
γj
l
|Sp|
∑
i∈Sp
[σ (Wj
l (xi
l−1)j) (xi
l−1)
⊤
j ] + 1
τk (Wj
l −Wk
l )
(xi
l)
j = proxμj
l(
1
2 ∥⋅∥2+ Ψ) ((xi
l)
j −μj
l ((Wj
l)
⊤
(σ (Wj
l (xi
l)
j
) −(xi
l+1)
j
) −Wj
l (xi
l−1)
j
))
Implementation
*Toulis, P., & Airoldi, E. M. (2017). Asymptotic and finite-sample properties of estimators based on stochastic 
gradients. The Annals of Statistics, 45(4), 1694–1727. 
**Lions, P. L., & Mercier, B. (1979). Splitting algorithms for the sum of two nonlinear operators. SIAM Journal on 
Numerical Analysis, 16(6), 964-979.


---

## 第95页

Implementation
We solve 
xα ∈arg min
x
BΨ(yδ, Wx + b) + αR(x)


---

## 第96页

xα ∈arg min
x
BΨ(yδ, Wx + b) + α
Implementation
We solve 
∑
p=1 ∑
q=1
(∇x)p,q,1
2
+ (∇x)p,q,2
2
Here, 
 is a forward finite-difference discretisation of the gradient operator
∇x
xα ∈arg min
x
BΨ(yδ, Wx + b) + α sup
z
⟨∇x, z⟩−
∑
p=1 ∑
q=1
| ⋅p,q,1 |2 + | ⋅p,q,2 |2
⋆
(z)
We replace the regularisation function by its convex conjugate


---

## 第97页

Implementation
xα ∈arg min
x
BΨ(yδ, Wx + b) + α sup
z
⟨∇x, z⟩−
∑
p=1 ∑
q=1
| ⋅p,q,1 |2 + | ⋅p,q,2 |2
⋆
(z)
We replace the regularisation function by its convex conjugate
and solve this saddle-point problem with a generalised PDHG method* 
xk+1 = xk −τx (W⊤σ (Wxk + b) −yδ) −αdivzk)
˜zk = zk + τz (2α∇xk+1 −α∇xk)
zk+1
p,q,d = ˜zk
p,q,d/ max (1,
˜zk
p,q,1
2
+
˜zk
p,q,2
2
)
for d ∈{1,2}
*Chambolle, A., & Pock, T. (2016). An introduction to continuous optimization for 
imaging. Acta Numerica, 25, 161-319.


---

