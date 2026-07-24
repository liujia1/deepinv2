# ERASMUS+ INTERNATIONAL PHD SUMMER SCHOOL 2025 Mathematics and Machine Learning for image analysis Lecture 2 - Optimization vs. Sampling

Thomas Pock 

Institute of Visual Computing 

Graz University of Technolog 

University of Bologna, June 3-6 2025 

## Bayesian inverse problems

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2751fdc3-9bab-4b63-9ca2-e2281342eaa1/eeefb7fda8aae341e0aac5d552e6eece0817ebcacddbcbc5592cb9c86933bee3.jpg)


▶ Bayesian inference 

$$
\boxed {p _ {X \mid Y} (x \mid y) = \frac {p _ {Y \mid X} (y \mid x) p _ {X} (x)}{p _ {Y} (y)}}
$$

<sup>▶</sup> Can be seen as logic with uncertainty. 

<sup>▶</sup> The likelihood $p _ { Y \mid X }$ is often known due to the image formation process 

$$
p _ {Y \mid X} (y \mid x) \propto \exp \left(- \frac {\| \mathcal {A} (x) - y \| ^ {2}}{2 \sigma^ {2}}\right)
$$

▶ The prior $p _ { X }$ is usually unknown and should be learned from data. 

## Bayesian inverse problems

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2751fdc3-9bab-4b63-9ca2-e2281342eaa1/9ac56854d56e853f880b00a994b536433df00c12cd979cfd580b577fb0b81f0a.jpg)


<sup>▶</sup> Bayesian inference 

$$
\boxed {p _ {X \mid Y} (x \mid y) = \frac {p _ {Y \mid X} (y \mid x) p _ {X} (x)}{p _ {Y} (y)}}
$$

<sup>▶</sup> Can be seen as logic with uncertainty. 

<sup>▶</sup> The likelihood $p _ { Y \mid X }$ is often known due to the image formation process 

$$
p _ {Y \mid X} (y \mid x) \propto \exp \left(- \frac {\| \mathcal {A} (x) - y \| ^ {2}}{2 \sigma^ {2}}\right)
$$

▶ The prior $p _ { X }$ is usually unknown and should be learned from data. 

## Bayesian inverse problems

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2751fdc3-9bab-4b63-9ca2-e2281342eaa1/7498d2996c514079e3ae34fc295c000d1856f78dc6a55e068d2077e56b4e1838.jpg)


<sup>▶</sup> Bayesian inference 

$$
\boxed {p _ {X \mid Y} (x \mid y) = \frac {p _ {Y \mid X} (y \mid x) p _ {X} (x)}{p _ {Y} (y)}}
$$

<sup>▶</sup> Can be seen as logic with uncertainty. 

<sup>▶</sup> The likelihood $p _ { Y \mid X }$ is often known due to the image formation process 

$$
p _ {Y \mid X} (y \mid x) \propto \exp \left(- \frac {\| \mathcal {A} (x) - y \| ^ {2}}{2 \sigma^ {2}}\right)
$$

▶ The prior $p _ { X }$ is usually unknown and should be learned from data. 

## Bayesian inverse problems

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2751fdc3-9bab-4b63-9ca2-e2281342eaa1/6fdb4cd7187d512126454070b14c275416dbe3f195c5d609708512fadf8507c8.jpg)


<sup>▶</sup> Bayesian inference 

$$
\boxed {p _ {X \mid Y} (x \mid y) = \frac {p _ {Y \mid X} (y \mid x) p _ {X} (x)}{p _ {Y} (y)}}
$$

<sup>▶</sup> Can be seen as logic with uncertainty. 

<sup>▶</sup> The likelihood $p _ { Y \mid X }$ is often known due to the image formation process 

$$
p _ {Y \mid X} (y \mid x) \propto \exp \left(- \frac {\| \mathcal {A} (x) - y \| ^ {2}}{2 \sigma^ {2}}\right)
$$

▶ The prior $p _ { X }$ is usually unknown and should be learned from data. 

## Bayesian inverse problems

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2751fdc3-9bab-4b63-9ca2-e2281342eaa1/b06920cb9814695b9dcabe29aa4dc0b9660e075462964120908182d685384055.jpg)


▶ Bayesian inference 

$$
\boxed {p _ {X \mid Y} (x \mid y) = \frac {p _ {Y \mid X} (y \mid x) p _ {X} (x)}{p _ {Y} (y)}}
$$

<sup>▶</sup> Can be seen as logic with uncertainty. 

▶ The likelihood $p _ { Y \mid X }$ is often known due to the image formation process 

$$
p _ {Y \mid X} (y \mid x) \propto \exp \left(- \frac {\| \mathcal {A} (x) - y \| ^ {2}}{2 \sigma^ {2}}\right)
$$

▶ The prior $p _ { X }$ is usually unknown and should be learned from data. 

## Bayesian point estimators

<sup>▶</sup> There are a number of Bayesian estimators that are used to compute point estimates from the posterior 

$$
\hat {x} (y) \in \underset {z} {\operatorname{argmin}} \int_ {\mathcal {X}} \ell (x, z) p _ {X | Y = y} (x) \mathrm{d} x,
$$

depending of the choice of the loss function $\ell ( x , z )$ 

<sup>▶</sup> There are a number of Bayesian estimators that are used to compute point estimates from the posterior 

$$
\hat {x} (y) \in \underset {z} {\operatorname{argmin}} \int_ {\mathcal {X}} \ell (x, z) p _ {X | Y = y} (x) \mathrm{d} x,
$$

depending of the choice of the loss function $\ell ( x , z )$ 

<sup>▶</sup> The 0-1 loss leads to the maximum a-posteriori (MAP) estimate 

$$
\hat {x} (y) = \underset {x} {\operatorname{argmax}} p _ {X | Y = y} (x) = \underset {x} {\operatorname{argmin}} - \log p _ {X | Y = y} (x)
$$

<sup>▶</sup> There are a number of Bayesian estimators that are used to compute point estimates from the posterior 

$$
\hat {x} (y) \in \underset {z} {\operatorname{argmin}} \int_ {\mathcal {X}} \ell (x, z) p _ {X | Y = y} (x) \mathrm{d} x,
$$

depending of the choice of the loss function $\ell ( x , z )$ 

<sup>▶</sup> The 0-1 loss leads to the maximum a-posteriori (MAP) estimate 

$$
\hat {x} (y) = \underset {x} {\operatorname{argmax}} p _ {X | Y = y} (x) = \underset {x} {\operatorname{argmin}} - \log p _ {X | Y = y} (x)
$$

<sup>▶</sup> The squared loss leads to the posterior expectation, or minimum mean squared estimate (MMSE) 

$$
\bar {x} (y) = \int_ {\mathcal {X}} x p _ {X | Y = y} (x) d x = \mathbb {E} _ {X | Y = y} [ x ]
$$

## Example: Total variation regularized Gaussian image denoising

<sup>▶</sup> Let us assume we have a prior and likelihood that take the form of a Gibbs distribution [Boltzmann 1868], [Gibbs 1889] 

$$
p _ {X} (x) \propto \exp (- R (x)), \quad p _ {Y | X} (y | x) \propto \exp (- D (x, y)),
$$

with R the regularizer and D the data fidelity term, which are here given b 

$$
R (x) = \lambda \sum_ {i, j} | x _ {i + 1, j} - x _ {i, j} | + | x _ {i, j + 1} - x _ {i, j} |, \quad D (x, y) = \frac {\| x - y \| ^ {2}}{2 \sigma^ {2}}
$$

<sup>▶</sup> In summary, the negative log posterior is given by 

$$
- \log p _ {X | Y} (x | y) \propto E (x) := R (x) + D (x, y)
$$

The behavior of the total variation as a regularizer and a prior can be very diferent. 

▶ It is well-known that using R(x) as a regularization term (here in 1D) leads to piecewise constant signals [Chambolle, Lions, ’97]. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2751fdc3-9bab-4b63-9ca2-e2281342eaa1/6a872ec1efeb3970b34eccae21b6366ea09f3e198fa912e09e2ef0e0f26da623.jpg)


▶ This is in contrast to actual samples $x \sim p x ( x )$ which can be obtained for 1D TV in linear time via a Levy process [Bohra, et al., ’23] 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2751fdc3-9bab-4b63-9ca2-e2281342eaa1/a3b35d20254cd0e4a4f17b0de3752ba65021acf0b22a6d9523c85fe0b11f9294.jpg)


▶ ${ \mathsf { M A P } }$ estimation (minimization) 

$$
\hat {x} = \underset {x \in \mathcal {X}} {\operatorname{argmin}} R (x) + \frac {1}{2 \sigma^ {2}} \| x - y \| _ {2} ^ {2}.
$$

<sup>▶</sup> The optimal solution is given by the proximal map [Moreau ’62] 

$$
\hat {x} = \left(I + \sigma^ {2} \partial R\right) ^ {- 1} (y) := \operatorname{prox} _ {\sigma^ {2} R} (y),
$$

which can be written as one gradient step on the infimal-convolution of R with a quadratic function $\frac { \| \cdot \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } }$ (called Moreau envelope) 

$$
\hat {x} = y - \sigma^ {2} \nabla \hat {R} _ {\sigma^ {2}} (y), \quad \hat {R} _ {\sigma^ {2}} = (R \square \frac {\| \cdot \| _ {2} ^ {2}}{2 \sigma^ {2}}),
$$

also with a nice representation of its gradient 

$$
\nabla \hat {R} _ {\sigma^ {2}} (y) = \frac {y - \operatorname{prox} _ {\sigma^ {2} R} (y)}{\sigma^ {2}}.
$$

<sup>▶</sup> MMSE estimation (expectation) 

$$
\bar {x} = \frac {1}{Z} \int_ {\mathcal {X}} x \exp \left(- R (x) - \frac {1}{2 \sigma^ {2}} \| x - y \| ^ {2}\right) d x, \quad \bar {x} = \mathbb {E} [ x | y ],
$$

which can be written in the form of one gradient step on the soft-infimal-convolution of $\mathsf { e x p } ( - R )$ with a Gaussian exp $\begin{array} { r l r } {  { \big ( - \frac { \| { \cdot } \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \big ) } } \end{array}$ 

$$
\boxed {\bar {x} = y - \sigma^ {2} \nabla \bar {R} _ {\sigma^ {2}} (y), \quad \bar {R} _ {\sigma^ {2}} = - \log \left(\exp (- R) * \exp \left(- \frac {\| \cdot \| _ {2} ^ {2}}{2 \sigma^ {2}}\right)\right).}
$$

<sup>▶</sup> This formula is known as Tweedie’s formula [Robbins ’56], [Miyasawa ’60]. 

<sup>▶</sup> The negative gradients $- \nabla \bar { R } _ { \sigma ^ { 2 } }$ are also known as score functions in denoising difusion models at difusion time $t = \sigma ^ { 2 } / 2$ 

<sup>▶</sup> Let’s introduce a temperature parameter $T > 0$ 

$$
\bar {R} _ {\sigma^ {2}} ^ {T} (y) = - T \log \left(\int \exp \left(\frac {- R (x) - \frac {\| x - y \| _ {2} ^ {2}}{2 \sigma^ {2}}}{T}\right) d x\right)
$$

<sup>▶</sup> One can show that as $T \to 0 ^ { + }$ , the soft-infimal convolution $\bar { R } _ { \sigma ^ { 2 } } ^ { T } ( y )$ converges to the infima convolution $\hat { R } _ { \sigma ^ { 2 } } ( y )$ 

<sup>▶</sup> As a consequence, the “Tweedie-prox” becomes a “Moreau-prox”. 

▶ This gives rise to a “Difusion at absolute zero” [Habring, Falk, Zach, P. ’25], as an alternative to the standard denoising difusion models, where the score can be computed based on the proximal map. 

## Algorithms

<sup>▶</sup> In practice, we usually do not have direct access to the gradients of the (soft-) infimal-convolutions to compute the MAP or MMSE in one step. 

<sup>▶</sup> The previous formulas can only be applied to image denoising. 

<sup>▶</sup> One needs some more general (iterative algorithms) 

<sup>▶</sup> In practice, we usually do not have direct access to the gradients of the (soft-) infimal-convolutions to compute the ${ \mathsf { M A P } }$ or MMSE in one step. 

<sup>▶</sup> The previous formulas can only be applied to image denoising. 

▶ One needs some more general (iterative algorithms) 

<sup>▶</sup> The most basic optimization algorithm to compute the ${ \mathsf { M A P } }$ estimate is gradient descent (GD). 

$$
x ^ {k + 1} = x ^ {k} - \tau \nabla E (x ^ {k})
$$

where $\tau > 0$ is the step size. 

▶ In practice, we usually do not have direct access to the gradients of the (soft-) infimal-convolutions to compute the ${ \mathsf { M A P } }$ or MMSE in one step. 

<sup>▶</sup> The previous formulas can only be applied to image denoising. 

▶ One needs some more general (iterative algorithms) 

<sup>▶</sup> The most basic optimization algorithm to compute the ${ \mathsf { M A P } }$ estimate is gradient descent (GD). 

$$
x ^ {k + 1} = x ^ {k} - \tau \nabla E (x ^ {k})
$$

where $\tau > 0$ is the step size. 

<sup>▶</sup> The sampling analogue is the unadjusted Langevin algorithm (ULA) 

$$
\boxed {X ^ {k + 1} = X ^ {k} - \tau \nabla E (X ^ {k}) + \sqrt {2 \tau} N ^ {k}}
$$

where $\tau$ is the step size and $N ^ { k }$ is a vector of standard i.i.d Gaussian noise. 

<sup>▶</sup> Suficient decrease condition [Armijo ’66]. Set $\tau > 0$ such that 

$$
E (x ^ {k}) - E (x ^ {k + 1}) - \sigma \tau \left\| \nabla E (x ^ {k}) \right\| ^ {2} \geq 0,
$$

where $\sigma \in ( 0 , 1 )$ 

<sup>▶</sup> Suficient decrease condition [Armijo ’66]. Set $\tau > 0$ such that 

$$
E (x ^ {k}) - E (x ^ {k + 1}) - \sigma \tau \left\| \nabla E (x ^ {k}) \right\| ^ {2} \geq 0,
$$

where $\sigma \in ( 0 , 1 )$ 

<sup>▶</sup> The Metropolis-Hastings rule [Metropolis et al. ’53][Hastings $^ { \prime } { } ^ { 7 0 } ] \mathrm { : }$ : Accept $x ^ { k + 1 }$ if 

$$
\min \left\{0, E (x ^ {k}) - E (x ^ {k + 1}) - \tau \left\| \nabla E (x ^ {k + \frac {1}{2}}) \right\| ^ {2} - \frac {\sqrt {2 \tau}}{2} \left\langle \nabla E (x ^ {k + \frac {1}{2}}), N ^ {k} \right\rangle \right\} > \log u,
$$

where $u \sim \mathcal { U } [ 0 , 1 ]$ and $\nabla E ( x ^ { k + \frac { 1 } { 2 } } ) = ( \nabla E ( x ^ { k } ) + \nabla E ( x ^ { k + 1 } ) ) / 2$ is the average gradient. 

<sup>▶</sup> Application of GD and ULA to total variation regularized image denoising: 

$$
R (x) = \lambda \sum_ {i, j} | x _ {i + 1, j} - x _ {i, j} | _ {\varepsilon} + | x _ {i, j + 1} - x _ {i, j} | _ {\varepsilon}, D (x, y) = \frac {1}{2 \sigma^ {2}} \| x - y \| ^ {2},
$$

using $\varepsilon = 1 0 ^ { - 3 } , \lambda _ { M A P } = 1 0 , \lambda _ { M M S E } = 2 0 , \sigma = 0 . 1$ , step size $\tau = 2 / L$ 

<sup>▶</sup> Optimization: A lot of research has gone into optimization in order to develop faster algorithms: 

<sup>▶</sup> Optimization: A lot of research has gone into optimization in order to develop faster algorithms: 

Some milestones: Interior point methods, pre-conditioning, half-quadratic optimization, duality, accelerated gradient methods, block-coordinate descent, proximal methods, primal-dual methods, dynamic programming, etc. 

<sup>▶</sup> Optimization: A lot of research has gone into optimization in order to develop faster algorithms: 

▶ Some milestones: Interior point methods, pre-conditioning, half-quadratic optimization, duality, accelerated gradient methods, block-coordinate descent, proximal methods, primal-dual methods, dynamic programming, etc. 

<sup>▶</sup> Sampling: Despite clear structural and algorithmic similarity, the progress in developing faster sampling algorithms seems to be much slower. 

<sup>▶</sup> Optimization: A lot of research has gone into optimization in order to develop faster algorithms: 

▶ Some milestones: Interior point methods, pre-conditioning, half-quadratic optimization, duality, accelerated gradient methods, block-coordinate descent, proximal methods, primal-dual methods, dynamic programming, etc. 

<sup>▶</sup> Sampling: Despite clear structural and algorithmic similarity, the progress in developing faster sampling algorithms seems to be much slower. 

<sup>▶</sup> Question: Can we leverage ideas from optimization to develop faster sampling algorithms? 

<sup>▶</sup> A fruitful idea in optimization has always been lifting the objective function to a high-dimensiona space where the new representation is expected to ofer a better structure. 

<sup>▶</sup> Let us consider a popular technique called half-quadratic minimization [Geman, Reynolds, ’92], which expresses a function $f ( x )$ as 

$$
f (x) = \min _ {z} q (x, z)
$$

where q is quadratic in x. 

<sup>▶</sup> The lifting represents the function $f ( x )$ as the infimum over a family of quadratic functions. 

<sup>▶</sup> Minimizing $f ( x )$ is replaced by alternating minimization with respect to x and z. 

$$
\boxed {x ^ {k + 1} \in \underset {x} {\operatorname{argmin}} q (x, z ^ {k}), \quad z ^ {k + 1} = \underset {z} {\operatorname{argmin}} q (x ^ {k + 1}, z).}
$$

<sup>▶</sup> Many variants exist (multiplicative, additive) and there are also relations to the convex conjugate [Nikolova, $\mathsf { N g } ^ { \prime } 0 5 \rbrack$ 

## Minimum envelope

<sup>▶</sup> The total variation regularization term from the image denoising application can be written as a half-quadratic minimization problem [Chambolle, Lions $^ { , } 9 7 ]$ 

<sup>▶</sup> It is based on rewriting the absolute function as 

$$
R (x) = \lambda \sum_ {i, j} | x _ {i + 1, j} - x _ {i, j} | + | x _ {i, j + 1} - x _ {i, j} |, \quad | t | = \min _ {z > 0} \frac {| t | ^ {2}}{2 z} + \frac {z}{2},
$$

which is quadratic in x and simple in z with $z ^ { * } ( t ) = | t |$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2751fdc3-9bab-4b63-9ca2-e2281342eaa1/fc744f5bb62f0ffdefb4df430554947d4401e7e5b9b897e3e3124f03e87289bc.jpg)


## Example

<sup>▶</sup> For the TV denoising example the half-quadratic minimization algorithm converges within a few iterations. 

<sup>▶</sup> Can we develop a similar algorithm for sampling? 

<sup>▶</sup> Consider the following lifting of the prior $p _ { X }$ with latent variables z: 

$$
p _ {X} (x) = \int p _ {X, Z} (x, z) \mathrm{d} z,
$$

where $p _ { { X , Z } }$ is expected to have a better structure for sampling. 

<sup>▶</sup> The sampling analogue to alternating minimization is Gibbs sampling [Geman, Geman, ’84] 

$$
\boxed {x ^ {k + 1} \sim p _ {X | Z = z ^ {k}}, \quad z ^ {k + 1} \sim p _ {Z | X = x ^ {k + 1}},}
$$

which alternates sampling from the conditional distributions. 

<sup>▶</sup> Gibbs sampling can be shown to be a particular instance of the Metropolis-Hastings algorithm [Metropolis et al. ’53][Hastings ’70]. 

<sup>▶</sup> Assume the following Product of Experts (PoE) prior model [Hinton ’99] 

$$
p _ {X} (x) \propto \prod_ {j = 1} ^ {m} \phi_ {j} ((K x) _ {j}),
$$

where $K : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ is a linear operator and $\phi _ { j } : \mathbb { R } \to \mathbb { R } ^ { + }$ are 1D factors. 

<sup>▶</sup> In case K is a convolutional operator, the model is equivalent to the Fields of Experts (FoE) prior model [Roth, Black ’05]. 

<sup>▶</sup> In [Kuric, Zach, Habring, Unser, P. ’25] we show that a PoE prior admits the following lifted representation in the form of a Gaussian latent machine (GLM) 

$$
p _ {X, Z} (x, z) \propto \prod_ {j = 1} ^ {m} \mathcal {N} ((K x) _ {j} | \mu_ {j} (z _ {j}), \sigma_ {j} ^ {2} (z _ {j})) p _ {j} (z _ {j}) = \underbrace {\mathcal {N} (K x | \tilde {\mu} (z) , \tilde {\Sigma} (z))} _ {p _ {X | Z = z} (x)} \cdot \prod_ {j = 1} ^ {m} p _ {j} (z _ {j}),
$$

where $\tilde { \mu } ( z ) = ( \tilde { \mu } _ { 1 } ( z _ { 1 } ) , . . . , \tilde { \mu } _ { m } ( z _ { m } ) \mathrm { ~ a n d ~ } \tilde { \Sigma } ( z ) = \mathrm { d i a g } \left( \tilde { \sigma } _ { 1 } ^ { 2 } ( z _ { 1 } ) , . . . , \tilde { \sigma } _ { m } ^ { 2 } ( z _ { m } ) \right)$ 

## The conditional distributions

▶ For Gibbs sampling we need to have access to the conditional distributions. 

<sup>▶</sup> The conditional distribution $p _ { X \mid Z }$ is simply given by the multivariate Gaussian 

$$
p _ {X \mid Z} (x, z) = \mathcal {N} (K x | \tilde {\mu} (z), \tilde {\Sigma} (z)) = \mathcal {N} (x | \mu (z), \Sigma (z))
$$

where 

$$
\mu (z) = \left(K ^ {\top} \tilde {\Sigma} (z) ^ {- 1} K\right) ^ {- 1}) K ^ {\top} \tilde {\Sigma} (z) ^ {- 1} \tilde {\mu} (z), \quad \Sigma (z) = \left(K ^ {\top} \tilde {\Sigma} (z) ^ {- 1} K\right) ^ {- 1}.
$$

<sup>▶</sup> The conditional distribution $p _ { X \mid Z }$ decomposes into m independent univariate distributions 

$$
p _ {Z | X} = \prod_ {j = 1} ^ {m} p _ {Z _ {i} | X}, \quad p _ {Z _ {j} | X = x} (z _ {j}) \propto \mathcal {N} ((K x) _ {j} | \mu_ {j} (z _ {j}), \sigma_ {j} ^ {2} (z _ {j})) \cdot p _ {j} (z _ {j}),
$$

where $p _ { j }$ basically depends on the factors $\phi _ { j }$ 

<sup>▶</sup> In many situations of practical interest, e.g. 1D Gaussian mixture models, there are closed form solutions for the univariate distributions $p _ { Z _ { i } | X }$ 

<sup>▶</sup> Consider again the 2D total variation image prior 

$$
p _ {X} (x) \propto \prod_ {i, j} \exp (- \lambda | x _ {i + 1, j} - x _ {i, j} |) \exp (- \lambda | x _ {i, j + 1} - x _ {i, j} |).
$$

<sup>▶</sup> Next, we consider a lifted representation of 1D Laplacian factors $\phi ( t ) = \exp ( - \lambda | t | )$ ) as a Gaussian scale mixture 

$$
\frac {\lambda}{2} \exp (- \lambda | t |) = \int_ {\mathbb {R} ^ {+}} \frac {1}{\sqrt {2 \pi z}} \exp \left(- \frac {t ^ {2}}{2 z}\right) \frac {\lambda^ {2}}{2} \exp \left(- \lambda^ {2} \frac {z}{2}\right) d z
$$

<sup>▶</sup> From the Gaussian component, one directly sees that $\mu ( z ) = 0$ and $\sigma ^ { 2 } ( z ) = z$ 

<sup>▶</sup> The 1D conditional distribution $p _ { Z _ { i } | X = t }$ is given by a generalized inverse Gaussian 

$$
p _ {Z _ {i} | X = t} \propto z ^ {- \frac {1}{2}} \exp \left(- \left(\frac {t ^ {2}}{z} + \lambda^ {2} z\right) / 2\right),
$$

from which sampling is “relatively” easy. 

<sup>▶</sup> In the negative log-domain it turns out that we obtain a soft-minimum over quadratic functions 

$$
\lambda | t | = - \log \left(\int_ {\mathbb {R} ^ {+}} \frac {2}{\sqrt {2 \pi z}} \exp \left(- \underbrace {\left(\frac {t ^ {2}}{2 z} + \frac {\lambda^ {2} z}{2}\right)} _ {q (t, z)}\right) \mathrm{d} z\right),
$$

where $\boldsymbol { q } ( t , z )$ is indeed quadratic in t. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2751fdc3-9bab-4b63-9ca2-e2281342eaa1/18d3cb8dcc5eca2687b65aa9abaa0a2781fa3fc8ed2951b8bc43221c0b62bf42.jpg)


## Gibbs sampling on the Gaussian latent machine

<sup>▶</sup> The Gibbs sampler to sample from $p _ { X }$ alternates sampling from a multivariate Gaussian and m independent generalized inverse Gaussians 

$$
\left\{ \begin{array}{l} x ^ {k + 1} \sim \mathcal {N} (x | 0, (K ^ {\top} \mathrm{diag} \big (z _ {1} ^ {k},..., z _ {m} ^ {k} \big) ^ {- 1} K) ^ {- 1}), \\ z _ {j} ^ {k + 1} \sim G I G (\lambda^ {2}, (K x ^ {k + 1}) _ {j} ^ {2}, \frac {1}{2}),   j = 1... m \end{array} \right.
$$

<sup>▶</sup> Sampling from the prior $p _ { X } { \big ( } x { \big ) }$ can be easily extended to sampling from the posterior $p x _ { \mid } r = p _ { Y | X } \cdot p _ { X }$ by combining with a Gaussian likelihood term for denoising 

$$
p _ {Y | X = x} (y) = \mathcal {N} (y | x, \sigma^ {2} \cdot I)
$$

which only modifies the Gaussian in the first step of the Gibbs algorithm. 

$$
\left\{ \begin{array}{l} x ^ {k + 1} \sim \mathcal {N} (x | 0, (K ^ {\top} \mathrm{diag} \left(z _ {1} ^ {k},..., z _ {m} ^ {k}\right) ^ {- 1} K) ^ {- 1}) \cdot \mathcal {N} (y | x, \sigma^ {2} \cdot I), \\ z _ {j} ^ {k + 1} \sim G I G (\lambda^ {2}, (K x ^ {k + 1}) _ {j} ^ {2}, \frac {1}{2}), i = 1... m \end{array} \right.
$$

<sup>▶</sup> The MMSE computed from the samples of the GLM converges in a few iterations. 

<sup>▶</sup> Consider the following multivariate Gaussian distribution 

$$
\pi_ {X} (x) \propto \exp \left(- \frac {1}{2} (x - \mu) ^ {\top} Q (x - \mu)\right),
$$

<sup>▶</sup> Sampling: Generate i.i.d. samples from the distribution 

<sup>▶</sup> Optimization: Find the mode of the distribution that is minimizing the quadratic function 

$$
\min _ {x} \frac {1}{2} x ^ {\top} Q x - b ^ {\top} x.
$$

<sup>▶</sup> In optimization we assume that we only have access to $b = Q \mu$ 

<sup>▶</sup> Let us assume the following two-block structure 

$$
\mu = \left[ \begin{array}{c} \mu_ {1} \\ \mu_ {2} \end{array} \right], \quad Q = \left[ \begin{array}{c c} Q _ {1 1} & Q _ {1 2} \\ Q _ {2 1} & Q _ {2 2} \end{array} \right], \quad b = \left[ \begin{array}{c} b _ {1} \\ b _ {2} \end{array} \right] = \left[ \begin{array}{c} Q _ {1 1} \mu_ {1} + Q _ {1 2} \mu_ {2} \\ Q _ {2 1} \mu_ {1} + Q _ {2 2} \mu_ {2} \end{array} \right]
$$

<sup>▶</sup> A Gibbs sampling algorithm that takes advantage of the two-block structure is given by 

$$
\left\{ \begin{array}{l} X _ {1} ^ {k + 1} \sim \mathcal {N} (\mu_ {1} - Q _ {1 1} ^ {- 1} Q _ {1 2} (X _ {2} ^ {k} - \mu_ {2}), Q _ {1 1} ^ {- 1}), \\ X _ {2} ^ {k + 1} \sim \mathcal {N} (\mu_ {2} - Q _ {2 2} ^ {- 1} Q _ {2 1} (X _ {1} ^ {k + 1} - \mu_ {1}), Q _ {2 2} ^ {- 1}), \end{array} \right.
$$

<sup>▶</sup> A Gauss-Seidel (alternating minimization) algorithm takes the form 

$$
\left\{ \begin{array}{l} x _ {1} ^ {k + 1} = Q _ {1 1} ^ {- 1} (b _ {1} - Q _ {1 2} x _ {2} ^ {k}) = \mu_ {1} - Q _ {1 1} ^ {- 1} Q _ {1 2} (x _ {2} ^ {k} - \mu_ {2}), \\ x _ {2} ^ {k + 1} = Q _ {2 2} ^ {- 1} (b _ {2} - Q _ {2 1} x _ {1} ^ {k + 1}) = \mu_ {2} - Q _ {2 2} ^ {- 1} Q _ {2 1} (x _ {1} ^ {k} - \mu_ {1}), \end{array} \right.
$$

## <sup>▶</sup> Gibbs is just noisy Gauss-Seidel!

<sup>▶</sup> It is well known that the Gauss-Seidel algorithm can be accelerated using the method of successive overrelaxation (SOR) [Frankel ’50] 

$$
\left\{ \begin{array}{l} x _ {1} ^ {k + 1} = (1 - \omega) x _ {1} ^ {k} + \omega Q _ {1 1} ^ {- 1} (b _ {1} - Q _ {1 2} x _ {2} ^ {k}), \\ x _ {2} ^ {k + 1} = (1 - \omega) x _ {2} ^ {k} + \omega Q _ {2 2} ^ {- 1} (b _ {2} - Q _ {2 1} x _ {1} ^ {k + 1}), \end{array} \right.
$$

where $\omega \in ( 0 , 2 )$ is the relaxation parameter. 

<sup>▶</sup> It is well known that the Gauss-Seidel algorithm can be accelerated using the method of successive overrelaxation (SOR) [Frankel ’50] 

$$
\left\{ \begin{array}{l} x _ {1} ^ {k + 1} = (1 - \omega) x _ {1} ^ {k} + \omega Q _ {1 1} ^ {- 1} (b _ {1} - Q _ {1 2} x _ {2} ^ {k}), \\ x _ {2} ^ {k + 1} = (1 - \omega) x _ {2} ^ {k} + \omega Q _ {2 2} ^ {- 1} (b _ {2} - Q _ {2 1} x _ {1} ^ {k + 1}), \end{array} \right.
$$

where $\omega \in ( 0 , 2 )$ is the relaxation parameter. 

<sup>▶</sup> [Adler ’81] observed that the Gibbs sampling algorithm can also be overrelaxed 

$$
\left\{ \begin{array}{l} X _ {1} ^ {k + 1} \sim \mathcal {N} ((1 - \omega) X _ {1} ^ {k} + \omega (\mu_ {1} - Q _ {1 1} ^ {- 1} Q _ {1 2} (X _ {2} ^ {k} - \mu_ {2}))  , \omega (2 - \omega) Q _ {1 1} ^ {- 1}), \\ X _ {2} ^ {k + 1} \sim \mathcal {N} ((1 - \omega) X _ {2} ^ {k} + \omega (\mu_ {2} - Q _ {2 2} ^ {- 1} Q _ {2 1} (X _ {1} ^ {k + 1} - \mu_ {1}))  , \omega (2 - \omega) Q _ {2 2} ^ {- 1}), \end{array} \right.
$$

where it is crucial to rescale the covariance matrix by the factor $\omega ( 2 - \omega )$ in order to ensure the correct covariance of $X ^ { k }$ 

<sup>▶</sup> [Fox and Parker ’17] proved a convergence rate, matching that of SOR. 

<sup>▶</sup> Values of $\omega \to 2$ result in negative autocorrelation of successive samples. 

<sup>▶</sup> Let’s consider again the unadjusted Langevin algorithm (ULA) 

$$
X ^ {k + 1} = X ^ {k} - \tau \nabla E (X ^ {k}) + \sqrt {2 \tau} N ^ {k} \Longleftrightarrow X ^ {k + 1} \sim \mathcal {N} (X ^ {k} - \tau \nabla E (X ^ {k}), 2 \tau \cdot I)
$$

<sup>▶</sup> Using the following splitting trick [Falk, Habring, P. ’25] 

$$
X ^ {k + 1} = \underbrace {X ^ {k} - \tau \nabla E (X ^ {k}) + \sqrt {\tau_ {1}} N _ {1} ^ {k}} _ {\gamma^ {k + 1}} + \sqrt {\tau_ {2}} N _ {1} ^ {k}, \quad \tau_ {1} + \tau_ {2} = 2 \tau ,
$$

we can interprete ULA as a two-block Gibbs algorithm 

$$
\left\{ \begin{array}{l} Y ^ {k + 1} \sim \mathcal {N} (X ^ {k} - \tau \nabla E (X ^ {k}), \tau_ {1} \cdot I), \\ X ^ {k + 1} \sim \mathcal {N} (Y ^ {k + 1}, \tau_ {2} \cdot I), \end{array} \right.
$$

which however is not a true Gibbs algorithm because of a lack of a joint distribution in X and Y . 

<sup>▶</sup> Now let’s apply the overrelaxation to the obtained Gibbs sampling scheme 

$$
\left\{ \begin{array}{l} Y ^ {k + 1} \sim \mathcal {N} ((1 - \omega) Y ^ {k} + \omega \left(X ^ {k} - \tau \nabla E (X ^ {k})\right), \omega (2 - \omega) \tau_ {1} \cdot I), \\ X ^ {k + 1} \sim \mathcal {N} ((1 - \omega) X ^ {k} + \omega Y ^ {k + 1}, \omega (2 - \omega) \tau_ {2} \cdot I), \end{array} \right.
$$

<sup>▶</sup> The scheme can be reduced back to a single-variable scheme 

$$
X ^ {k + 1} = X ^ {k} - \gamma \nabla E (X ^ {k}) + \beta (X ^ {k} - X ^ {k - 1}) + \sqrt {2 \gamma (1 - \beta)} N ^ {k},
$$

where we have defined $\gamma = \omega ^ { 2 } \tau$ and $\beta = ( 1 - \omega ) ^ { 2 }$ 

▶ The obatined scheme, which we term Inertial Langevin Algorithm (ILA) is nothing than the sampling analogue of the heavy ball algorithm [Polyak ’64]. 

<sup>▶</sup> It is crucial to scale the noise by the factor $\sqrt { 2 \gamma ( 1 - \beta ) }$ 

## Comparison

<sup>▶</sup> The ground truth solution $x _ { M M S E } ^ { * }$ is computed using GLM. 

▶ Choosing an inertial parameter $\beta  1$ leads to a significant acceleration. 

<sup>▶</sup> A small bias remains but can be accounted for using a Metropolis-Hastings acceptance test. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2751fdc3-9bab-4b63-9ca2-e2281342eaa1/34bb2b1aa13f7e0bec70d92844c5359687bfdbee768baa80beb9dffbfb692284.jpg)


<sup>▶</sup> The discrete ILA scheme can be shown to be a discretization of the underdamped Lagevin SDE 

$$
\left\{ \begin{array}{l l} \mathrm{d} \bar {V} _ {t} & = \big (- \delta \bar {V} _ {t} - \nabla E (\bar {X} _ {t}) \big) \mathrm{d} t + \sqrt {2 \delta} \mathrm{d} W _ {t}. \\ \mathrm{d} \bar {X} _ {t} & = \bar {V} _ {t} \mathrm{d} t, \end{array} \right.
$$

with friction parameter $\delta > 0$ , which is known to have a stationary distribution 

$$
\pi (x, v) \propto \exp \left(- \left(E (x) + \frac {\| v \| ^ {2}}{2}\right)\right)
$$

<sup>▶</sup> For a proof of convergence, we consider a slightly diferent parametrization of the ILA scheme, which allows to control the discretization error with required order. 

▶ Theorem: Assume E is strongly convex with Lipschitz continuous gradient, then the sample distribution of the time-discrete ILA scheme conveges to the stationary distribution of the underdamped Langevin SDE in $\mathcal { W } _ { 2 }$ distance as $\Delta t  0$ 

<sup>▶</sup> There is an ongoing transition from pure optimization algorithms to more general sampling algorithms. 

<sup>▶</sup> Boils down to a diference between min and softmin, where the latter is more general. 

<sup>▶</sup> Many tricks that work well in optimization can be transformed to sampling. 

<sup>▶</sup> Half-quadratic minimization <sup>⇝</sup> Gaussian latent machine (GLM) 

<sup>▶</sup> Inertial/accelerated gradient descent <sup>⇝</sup> Inertial Langevin algorithm (ILA) 

<sup>▶</sup> Allows to perform Bayesian inference and uncertainty quantification for high-dimensional problems. 

<sup>▶</sup> For details and convergence rates see our recent preprints (will be online soon) 

<sup>▶</sup> There is an ongoing transition from pure optimization algorithms to more general sampling algorithms. 

<sup>▶</sup> Boils down to a diference between min and softmin, where the latter is more general. 

<sup>▶</sup> Many tricks that work well in optimization can be transformed to sampling. 

▶ Half-quadratic minimization <sup>⇝</sup> Gaussian latent machine (GLM) 

▶ Inertial/accelerated gradient descent <sup>⇝</sup> Inertial Langevin algorithm (ILA) 

<sup>▶</sup> Allows to perform Bayesian inference and uncertainty quantification for high-dimensional problems. 

<sup>▶</sup> For details and convergence rates see our recent preprints (will be online soon) 

Thanks for listening! 

## Appendix: Sampling from the multivariate Gaussian in the GLM

<sup>▶</sup> In the GLM, we need to sample from a multivariate Gaussian with negative log density (up to constants) 

$$
\frac {1}{2} x ^ {T} \left(K ^ {\top} \operatorname{diag} \left(z _ {1} ^ {k},..., z _ {m} ^ {k}\right) ^ {- 1} K\right) x + \frac {1}{2 \sigma^ {2}} \| x - y \| ^ {2},
$$

where z are the latent variables. 

<sup>▶</sup> This quadratic function be rewritten in the standard form (up to constants) 

$$
\frac {1}{2} (x - \mu) ^ {T} (A ^ {\top} A) (x - \mu)
$$

with 

$$
A = \left[ \begin{array}{c} \text {diag} \left(z _ {1} ^ {k},..., z _ {m} ^ {k}\right) ^ {- \frac {1}{2}} K \\ \frac {1}{\sigma} \cdot I \end{array} \right], \quad \mu = \left(A ^ {\top} A\right) ^ {- 1} y / \sigma^ {2}.
$$

<sup>▶</sup> The task is to sample from a Gaussian with negative log density 

$$
\frac {1}{2} (x - \mu) ^ {T} (A ^ {\top} A) (x - \mu)
$$

<sup>▶</sup> A sample x is computed from a standard Gaussian sample $\boldsymbol { z } \sim \mathcal { N } ( \boldsymbol { 0 } , \boldsymbol { I } )$ as 

$$
x = \mu + \left(A ^ {\top} A\right) ^ {- 1} A ^ {\top} z.
$$

<sup>▶</sup> A direct computation shows that 

$$
\operatorname{cov} [ x ] = \left(\left(A ^ {\top} A\right) ^ {- 1} A ^ {\top}\right) \left(\left(A ^ {\top} A\right) ^ {- 1} A ^ {\top}\right) ^ {\top} = \left(A ^ {\top} A\right) ^ {- 1}
$$

<sup>▶</sup> Consider the previous equation 

$$
x = \mu + \left(A ^ {\top} A\right) ^ {- 1} A ^ {\top} z
$$

which when inserting the formula for them mean $\mu$ gives 

$$
x = \left(A ^ {\top} A\right) ^ {- 1} y / \sigma^ {2} + \left(A ^ {\top} A\right) ^ {- 1} A ^ {\top} z.
$$

<sup>▶</sup> Multiplying from the left with $A ^ { \top } A$ we obtain the linear system of equations 

$$
\left(A ^ {\top} A\right) x = y / \sigma^ {2} + A ^ {\top} z,
$$

which can be solved using an iterative solver such as CG. 