## Luca Ratti

Department of Mathematics, Università degli studi di Bologna, 

luca.ratti5@unibo.it 

ERASMUS+ International PhD Summer School 2025 Mathematics and Machine Learning for image analysis 

University of Bologna 3<sup>rd</sup> June 2025 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/23e8ea0006023ea088c5062377e39a2326ab1cff31abc6ea15a5e77ad3746860.jpg)


## Statistical learning - a supervised regression problem

On a completely diferent note... 

[Cucker, Smale, On the mathematical foundations of learning, 2002] 

## A supervised regression problem

Given two random variables $_ { \mathcal { X } }$ on X and y on Y 

find a function f: $Y  X \thinspace \mathrm { S . t . } \ f ( y ) \approx x$ 

▶ (ideal setting) knowing the joint distribution $\pi _ { \boldsymbol { \mathcal { Y } } , \boldsymbol { x } } \sim \left( \boldsymbol { \mathcal { Y } } , \boldsymbol { \mathcal { x } } \right)$ 

▶ (real setting) knowing a sample $\{ ( \mathsf { y } _ { i } , \mathsf { x } _ { i } ) \} _ { i = 1 } ^ { N } \sim _ { i . i . d } \pi _ { \mathscr { u } , x } ;$ 

## Statistical learning - a supervised regression problem

On a completely diferent note... 

[Cucker, Smale, On the mathematical foundations of learning, 2002] 

## A supervised regression problem

Given two random variables x on X and y on Y find a function f: $Y  X \thinspace \mathrm { S . t . } \ f ( y ) \approx x$ 

▶ (ideal setting) knowing the joint distribution $\pi _ { \boldsymbol { \mathcal { Y } } , \boldsymbol { x } } \sim \left( \boldsymbol { \mathcal { Y } } , \boldsymbol { \mathcal { x } } \right)$ ; 

▶ (real setting) knowing a sample $\{ ( \mathsf { y } _ { i } , \mathsf { x } _ { i } ) \} _ { i = 1 } ^ { N } \sim _ { i . i . d } \pi _ { \mathscr { u } , x } ;$ 

1) Consider a loss function, e.g. $\ell ( \mathsf { x } ^ { \prime } , \mathsf { x } ) = \| \mathsf { x } ^ { \prime } - \mathsf { x } \| _ { x } ^ { 2 } .$ 

2) Introduce $\mathcal { F }$ : hypothesis space, a set of functions from Y to X. 

3.id) Minimize the Expected Loss: ⇝ $\operatorname* { m i n } _ { f \in { \mathcal { F } } } \mathbb { E } _ { ( x , y ) \sim \pi _ { x y } } \left[ \ell ( f ( y , x ) , x ) \right]$ 

3.re) Minimize the Empirical Risk: ⇝ $\operatorname* { m i n } _ { f \in \mathcal { F } } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell ( f ( \mathsf { y } _ { i } ) , \mathsf { x } _ { i } )$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/50fe2500c52f5865ec65a544216d129c86e120d5ff26fd85455ccc60ee29a0a4.jpg)


$$
X = \mathbb {R}, \quad Y = \mathbb {R},
$$

F: polynomials of degree 1 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/1dd9d6e25c571fd2315f131dcd1c12bee052b57e6a80855e563a1b2c12bf0716.jpg)


$$
X = \mathbb {R}, \quad Y = \mathbb {R},
$$


F: polynomials of degree 5


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/077c6bef703850271a8f2e638d83d199174c7e0c545f11066551b8559141b366.jpg)


$$
X = \mathbb {R}, \quad Y = \mathbb {R},
$$

F: polynomials of degree 10 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/dd514ac7d4a1c50756b7e856dc38e183c569f0a11b5b66bf01c43f23340e6875.jpg)


## ... but denoising can be seen as a regression task!

Learned reconstruction for inverse problems: 

suppose that $\boldsymbol { \mathcal { Y } } = \mathsf { A } \boldsymbol { \mathcal { x } } + \boldsymbol { \varepsilon }$ (forward model): $f ( \mathscr { y } ) \approx \mathscr { x } ~ $ find $f \approx \mathsf { A } ^ { - 1 }$ 

Denoising: let X = Y and A = Id, i.e.: $y = x + \varepsilon .$ 

Why not using simply $f = \mathsf { A } ^ { - 1 } = \mathsf { I d } ?$ 

## ... but denoising can be seen as a regression task!

Learned reconstruction for inverse problems: 

suppose that $\boldsymbol { \mathcal { Y } } = \mathsf { A } \boldsymbol { \mathcal { x } } + \boldsymbol { \varepsilon }$ (forward model): $f ( \mathscr { y } ) \approx \mathscr { x } ~ $ find $f \approx \mathsf { A } ^ { - 1 }$ 

Denoising: let $X = Y$ and $\mathsf { A } = \mathsf { I d } ,$ , i.e.: $y = x + \varepsilon .$ 

Why not using simply $f = \mathsf { A } ^ { - 1 } = \mathsf { I d } ?$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/16affb08f9b727cf2ca0ba649098b8594f785a64d319f9bd19917e69f30208ff.jpg)


The simplest example: 

$$
\blacktriangleright X = \mathbb {R} ^ {2};
$$

▶ ε ∼ N(0, 0.1<sup>2</sup> Id); 

▶ x : uniformly distributed on 

$$
\{0 \} \times [ - 1, 1 ] \cup [ - 1, 1 ] \times \{0 \};
$$

$y = x + \varepsilon .$ 

$f = | { \mathsf { d } }$ 

relative error: 20.70% 

## ... but denoising can be seen as a regression task!

Learned reconstruction for inverse problems: 

suppose that $\boldsymbol { \mathcal { Y } } = \mathsf { A } \boldsymbol { \mathcal { x } } + \boldsymbol { \varepsilon }$ (forward model): $f ( \mathscr { y } ) \approx \mathscr { x } ~ $ find $f \approx \mathsf { A } ^ { - 1 }$ 

Denoising: let $X = Y$ and $\mathsf { A } = \mathsf { I d } ,$ , i.e.: $y = x + \varepsilon .$ 

Why not using simply $f = \mathsf { A } ^ { - 1 } = \mathsf { I d } ?$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/288a9be893475c6f39f1e20f1188da9e198b3a8002b66d635aa3d86fcbd7301f.jpg)


The simplest example: 

$$
\blacktriangleright X = \mathbb {R} ^ {2};
$$

$\varepsilon \sim n ( 0 , 0 . 1 ^ { 2 } 1 { \mathsf { d } } ) ;$ 

▶ $_ { \mathcal { x } }$ : uniformly distributed on $\{ 0 \} \times [ - 1 , 1 ] \cup [ - 1 , 1 ] \times \{ 0 \} ;$ 

$y = x + \varepsilon .$ 

▶ Variational denoiser: Tikhonov 

$$
\begin{array}{l} f _ {T} (\mathbf {y}) = \arg \min _ {\mathbf {x}} \left\{\frac {1}{2} \| \mathbf {x} - \mathbf {y} \| _ {2} ^ {2} + \frac {\lambda}{2} \| \mathbf {x} \| _ {2} ^ {2} \right\} \\ = \frac {1}{1 + \lambda} \mathbf {y} \end{array}
$$

relative error: 19.21% 

## ... but denoising can be seen as a regression task!

Learned reconstruction for inverse problems: 

suppose that $\boldsymbol { \mathcal { Y } } = \mathsf { A } \boldsymbol { \mathcal { x } } + \boldsymbol { \varepsilon }$ (forward model): $f ( \mathscr { y } ) \approx \mathscr { x } ~ $ find $f \approx \mathsf { A } ^ { - 1 }$ 

Denoising: let $X = Y$ and $\mathsf { A } = \mathsf { I d } ,$ , i.e.: $y = x + \varepsilon .$ 

Why not using simply $f = \mathsf { A } ^ { - 1 } = \mathsf { I d } ?$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/20a899ec9d00cfce97ddad2d8d5fd5a18c01ed70a1ef7f8d8df628f392415e24.jpg)


The simplest example: 

$$
\blacktriangleright X = \mathbb {R} ^ {2};
$$

$\varepsilon \sim n ( 0 , 0 . 1 ^ { 2 } 1 { \mathsf { d } } ) ;$ 

▶ $_ { \mathcal { x } }$ : uniformly distributed on 

$$
\{0 \} \times [ - 1, 1 ] \cup [ - 1, 1 ] \times \{0 \};
$$

$y = x + \varepsilon .$ 

▶ Variational denoiser: Lasso 

$$
f _ {L} (y) = \operatorname{prox} _ {\lambda \| \cdot \| _ {1}} (y) = S T (y; \lambda)
$$

$$
[ f _ {L} (\mathbf {y}) ] _ {j} = \left\{ \begin{array}{c} \operatorname{sign} (y _ {j}) (| y _ {j} | - \lambda) \text {   if   } | y _ {j} | > \lambda \\ 0 \text {   if   } | y _ {j} | \leq \lambda \end{array} \right.
$$

relative error: 18.41% 

## ... but denoising can be seen as a regression task!

Learned reconstruction for inverse problems: 

suppose that $\boldsymbol { \mathcal { Y } } = \mathsf { A } \boldsymbol { \mathcal { x } } + \boldsymbol { \varepsilon }$ (forward model): $f ( \mathscr { y } ) \approx \mathscr { x } ~ $ find $f \approx \mathsf { A } ^ { - 1 }$ 

Denoising: let X = Y and A = Id, i.e.: $y = x + \varepsilon .$ 

Why not using simply $f = \mathsf { A } ^ { - 1 } = \mathsf { I d } ?$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/9f74a9fc42e44591cf5303353e1b47ec49606802fa28df08e7befc92c8bcd257.jpg)


The simplest example: 

$X = \mathbb { R } ^ { 2 } ;$ 

$\varepsilon \sim n ( 0 , 0 . 1 ^ { 2 } 1 { \mathsf { d } } ) ;$ 

▶ $_ { \mathcal { x } }$ : uniformly distributed on $\{ 0 \} \times [ - 1 , 1 ] \cup [ - 1 , 1 ] \times \{ 0 \} ;$ 

$y = x + \varepsilon .$ 

1 Learned denoiser (ideal case) f = . . . - Bayes denoiser relative error: 16.56% 

## Image Denoising as a regression problem

## Setup:

▶ Each image is a vector $\ b { \mathsf { x } } \in \mathbb { R } ^ { n }$ , where n = height × width × channels 

▶ Noisy image: $\boldsymbol { \mathsf { y } } = \boldsymbol { \mathsf { x } } + \boldsymbol { \varepsilon } ,$ , Clean image: x, Noise: ε 

▶ Goal: learn $f \colon  { \mathbb { R } } ^ { n } \to  { \mathbb { R } } ^ { n }$ such that $f ( \mathscr { y } ) \approx \mathscr { x }$ 

Loss Function: 

$$
\ell (f (\mathbf {y}), \mathbf {x}) = \operatorname{MSE} (f (\mathbf {y}), \mathbf {x}) = \frac {1}{n} \sum_ {j = 1} ^ {n} ([ f (\mathbf {y}) ] _ {j} - [ \mathbf {x} ] _ {j}) ^ {2} = \frac {1}{n} \| f (\mathbf {y}) - \mathbf {x} \| _ {2} ^ {2}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/4abd5989d7dac5504850b7814b4f6b5dc153239d5f6211efc1fa1c253624f2cb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/fca2c0c6334c5e2a33b328d39a98e72f3d31256a8ecda2fe7cad07ed9514113c.jpg)


In your experience, what are key ingredients of statistical learning? 

## In your experience, what are key ingredients of statistical learning?

## My personal list

A. a training dataset; 

B. a loss function; 

C. a (parametric) hypothesis class; 

D. an optimization algorithm. 

# A statistical learning perspective on image denoising

A. Training datasets 

## Supervised datasets and beyond

1) Supervised setting 

A dataset of paired noisy and clean images $\{ ( \mathsf { y } _ { i } , \mathsf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ is available. 

▶ empirical loss minimization. 

## Supervised datasets and beyond

## 1) Supervised setting

A dataset of paired noisy and clean images $\{ ( \mathsf { y } _ { i } , \mathsf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ is available. 

▶ empirical loss minimization. 

## 2) Self-supervised setting

A dataset of clean images $\{ ( \mathsf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ is available + the noise model is known (e.g., $\begin{array} { r } { y = x + \varepsilon , \varepsilon \sim n ( 0 , \sigma ^ { 2 } \vert ) , } \end{array}$ 

▶ sample $\{ \varepsilon _ { i } \} _ { i = 1 } ^ { N } \sim _ { i . i . d . } \pi _ { \varepsilon }$ , define $\mathsf { y } _ { i } = \mathsf { x } _ { i } + \varepsilon _ { i } ;$ 

▶ empirical loss minimization. 

## Supervised datasets and beyond

## 1) Supervised setting

A dataset of paired noisy and clean images $\{ ( \mathsf { y } _ { i } , \mathsf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ is available. 

▶ empirical loss minimization. 

## 2) Self-supervised setting

A dataset of clean images $\{ ( \mathsf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ is available + the noise model is known (e.g., $\begin{array} { r } { y = x + \varepsilon , \varepsilon \sim n ( 0 , \sigma ^ { 2 } \vert ) , } \end{array}$ 

▶ sample $\{ \varepsilon _ { i } \} _ { i = 1 } ^ { N } \sim _ { i . i . d . } \pi _ { \varepsilon }$ , define $\mathsf { y } _ { i } = \mathsf { x } _ { i } + \varepsilon _ { i } ;$ 

▶ empirical loss minimization. 

## 3) Unsupervised setting - case x

A dataset of clean images $\{ ( \mathsf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ is available. 

▶ learn the prior distribution $\pi _ { x }$ (e.g. via Tweedie’s formula); 

▶ use a model-adaptive algorithm to identify the noise model. 

## Supervised datasets and beyond

## 1) Supervised setting

A dataset of paired noisy and clean images $\{ ( \mathsf { y } _ { i } , \mathsf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ is available. 

▶ empirical loss minimization. 

## 2) Self-supervised setting

A dataset of clean images $\{ ( \mathsf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ is available + the noise model is known (e.g., $\begin{array} { r } { y = x + \varepsilon , \varepsilon \sim n ( 0 , \sigma ^ { 2 } \vert ) , } \end{array}$ 

▶ sample $\{ \varepsilon _ { i } \} _ { i = 1 } ^ { N } \sim _ { i . i . d . } \pi _ { \varepsilon }$ , define $\mathsf { y } _ { i } = \mathsf { x } _ { i } + \varepsilon _ { i } ;$ 

▶ empirical loss minimization. 

## 3) Unsupervised setting - case x

A dataset of clean images $\{ ( \mathsf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ is available. 

▶ learn the prior distribution $\pi _ { x }$ (e.g. via Tweedie’s formula); 

▶ use a model-adaptive algorithm to identify the noise model. 

## 4) Unsupervised setting - case y

A dataset of noisy images $\{ ( \mathsf { y } _ { i } ) \} _ { i = 1 } ^ { N }$ is available. 

▶ ad-hoc techniques (when available). 

# A statistical learning perspective on image denoising

B. Loss function 

## Loss minimization and regularization

▶ Loss function, ℓ: measures the quality of a single denoised image. Es.: $\ell ( { \mathsf X } ^ { \prime } , { \mathsf X } ) = \| { \mathsf X } ^ { \prime } - { \mathsf X } \| _ { 2 } ^ { 2 } , \quad \ell ( { \mathsf X } ^ { \prime } , { \mathsf X } ) = { \mathsf P } { \mathsf S } { \mathsf N } { \mathsf R } ( { \mathsf X } ^ { \prime } , { \mathsf X } ) .$ 

▶ Expected loss, L: measures the quality of a denoiser f, using the ful knowledge of $\pi _ { \boldsymbol { y } , \boldsymbol { x } } \ ( \mathrm { i d e a l } )$ Es. $L ( f ) = \mathbb { E } _ { ( \mathcal { y } , x ) \sim \pi _ { ( \mathcal { y } , x ) } } [ \ell ( f ( \mathcal { y } ) , x ) ]$ 

▶ Empirical risk, <sup>ˆ</sup>L: measures the quality of a denoiser f, using a supervised dataset. Es. $\begin{array} { r } { \hat { L } ( f ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { N } \ell ( f ( \mathsf { y } _ { i } ) , \mathsf { x } _ { i } ) } \end{array}$ 

## Loss minimization and regularization

▶ Loss function, ℓ: measures the quality of a single denoised image. Es.: $\ell ( { \mathsf X } ^ { \prime } , { \mathsf X } ) = \| { \mathsf X } ^ { \prime } - { \mathsf X } \| _ { 2 } ^ { 2 } , \quad \ell ( { \mathsf X } ^ { \prime } , { \mathsf X } ) = { \mathsf P } { \mathsf S } { \mathsf N } { \mathsf R } ( { \mathsf X } ^ { \prime } , { \mathsf X } ) .$ 

▶ Expected loss, L: measures the quality of a denoiser f, using the ful knowledge of $\pi _ { \boldsymbol { y } , \boldsymbol { x } } \ ( \mathrm { i d e a l } )$ Es. $L ( f ) = \mathbb { E } _ { ( \mathcal { y } , x ) \sim \pi _ { ( \mathcal { y } , x ) } } [ \ell ( f ( \mathcal { y } ) , x ) ]$ 

▶ Empirical risk, <sup>ˆ</sup>L: measures the quality of a denoiser f, using a supervised dataset. Es. $\begin{array} { r } { \hat { L } ( f ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { N } \ell ( f ( \mathsf { y } _ { i } ) , \mathsf { x } _ { i } ) } \end{array}$ 


The risk of overfitting


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/5d4af84765f813ea8226287ae9793e91e1843848869befd436959500cd57e552.jpg)


Minimizing <sup>ˆ</sup>L among all possible f: $Y  X$ is not a good idea: 

▶ the result f might work poorly on new data $\mathcal { Y } m + 1 , \mathcal { Y } m + 2 , \ldots .$ 

▶ the result f might be unstable $( \vert \vert f \vert \vert \gg 1 )$ 

## Loss minimization and regularization

▶ Loss function, ℓ: measures the quality of a single denoised image. Es.: $\ell ( { \mathsf X } ^ { \prime } , { \mathsf X } ) = \| { \mathsf X } ^ { \prime } - { \mathsf X } \| _ { 2 } ^ { 2 } , \quad \ell ( { \mathsf X } ^ { \prime } , { \mathsf X } ) = { \mathsf P } { \mathsf S } { \mathsf N } { \mathsf R } ( { \mathsf X } ^ { \prime } , { \mathsf X } ) .$ 

▶ Expected loss, L: measures the quality of a denoiser f, using the ful knowledge of $\pi _ { \boldsymbol { y } , \boldsymbol { x } } \ ( \mathrm { i d e a l } )$ Es. $L ( f ) = \mathbb { E } _ { ( \mathcal { y } , x ) \sim \pi _ { ( \mathcal { y } , x ) } } [ \ell ( f ( \mathcal { y } ) , x ) ]$ 

▶ Empirical risk, <sup>ˆ</sup>L: measures the quality of a denoiser f, using a supervised dataset. Es. $\begin{array} { r } { \hat { L } ( f ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { N } \ell ( f ( \mathsf { y } _ { i } ) , \mathsf { x } _ { i } ) } \end{array}$ 

## The risk of overfitting

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/96a97a22873762e9d408aa19ef8ef6d6bac21d23892d595ac08ceb1ee8469b97.jpg)


Minimizing <sup>ˆ</sup>L among all possible f: $Y  X$ is not a good idea: 

▶ the result f might work poorly on new data $\mathcal { Y } m + 1 , \mathcal { Y } m + 2 , \ldots .$ 

▶ the result f might be unstable $( \vert \vert f \vert \vert \gg 1 )$ 

## Explicit regularization

One possible solution: minimize $\hat { L } ( f ) + R ( f )$ 

Ex. $R ( f ) = \| f \| _ { \mathcal { H } } ,$ , H a suitable function space (e.g. a RKHS⇝kernel methods). 

C. Parametric hypothesis classes 

## Implicit regularization via parametric hypothesis spaces

Idea: restricting the space F in which optimizing <sup>ˆ</sup>L induces an implicit regularization. Ex: F = polynomials of degree 10, 5, 1. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/b6a7794142675ce17ad69de7ad287130a29cbac51332b60413d9209fe6c3c5a0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/34e185e2faab96a7d0a7e8133d226bd73c33fb582700a51f912d1d34ecb00111.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/52bf824f971a7f41144a11a01f9fc1de266543e2aaec0f2e3219f9eec709a306.jpg)


## Implicit regularization via parametric hypothesis spaces

Idea: restricting the space F in which optimizing <sup>ˆ</sup>L induces an implicit regularization. Ex: F = polynomials of degree 10, 5, 1. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/ba87f90a924a8a85d2832bcaf5164ac504eff9d47c5ccc1ddae16190c3db2b0c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/b76c6b563623068fc0d5bcb564bceb4cad915dee9ecad5cc07d477e58919c278.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/0f97c8e7a44b4b500bf66534ceb54a4db463a6ec19dccc44e643fb78546a1057.jpg)


## Parametric hypothesis spaces

Spaces of functions depending on $( p )$ parameters: 

$$
\mathcal {F} = \left\{f _ {\theta}: Y \rightarrow X, \quad \theta \in \Theta \cong \mathbb {R} ^ {p} \right\}.
$$

Examples $( X = Y = \mathbb { R } )$ : 

▶ polynomials: $\begin{array} { r } { f _ { \theta } ( y ) = \theta _ { 1 } + \theta _ { 2 } y + . . . + \theta _ { p } y ^ { p - 1 } ; } \end{array}$ ; 

▶ linear splines: $f _ { \theta } ( y ) = \theta _ { 1 } + \theta _ { 2 } \chi _ { [ \theta _ { 3 } , + \infty ) } ( y ) + \ldots \theta _ { p - 1 } \chi _ { [ \theta _ { p } , + \infty ) } ( y ) ;$ 

▶ Neural Networks, e.g. $f _ { \theta } ( y ) = \theta _ { 1 } \sigma ( \theta _ { 2 } y ) + . . . + \theta _ { p - 1 } \sigma ( \theta _ { p } y ) .$ 

## A bias-variance tradeof

A variation of a picture from [M. Belkina, D. Hsuc, S. Maa, S. Mandal, Reconciling modern machine-learning practice and the classical bias–variance trade-of, 2019] - part 1 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/312c2812a5702b775f0880cb8de48ceb55b6a9c34df9479eb8b9c51b94f679c0.jpg)


## A bias-variance tradeof

A variation of a picture from [M. Belkina, D. Hsuc, S. Maa, S. Mandal, Reconciling modern machine-learning practice and the classical bias–variance trade-of, 2019] - part 2 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/6681233991e83d8e90ac735abad55ab5f25ef3705d974bb61b1b268f40584522.jpg)


## Neural Networks: very expressive parametric functions from $\mathbb { I } ^ { \ast }$

## Neural Networks: a working expression

Given a nonlinear function $\sigma : \mathbb { R }  \mathbb { R }$ and its element-wise version s.t. $[ \sigma ( \mathsf { x } ) ] _ { j } = \sigma ( [ \mathsf { x } ] _ { j } )$ , given $W _ { l } \in \mathbb { R } ^ { n _ { l - 1 } \times n _ { l } }$ and $\mathsf { b } _ { l } \in \mathbb { R } ^ { n _ { l } } , l = 1 , \hdots , L _ { \hphantom { ( } }$ , let 

$$
f _ {\theta} (x) = W _ {L + 1} \sigma \left(W _ {L} \dots \sigma \left(W _ {1} x + b _ {1}\right) \dots + b _ {L}\right) + b _ {L + 1}.
$$

The learnable parameters are $\theta = \{ W _ { 1 } , \ldots , W _ { L } , \mathsf { b } _ { 1 } , \ldots , \mathsf { b } _ { L } \}$ . 

## Neural Networks: very expressive parametric functions from $\mathbb { I } _ { 1 } ^ { \perp } = \mathbb { I } _ { 2 } ^ { \perp }$

## Neural Networks: a working expression

Given a nonlinear function $\sigma : \mathbb { R }  \mathbb { R }$ and its element-wise version s.t. $[ \sigma ( \mathsf { x } ) ] _ { j } = \sigma ( [ \mathsf { x } ] _ { j } )$ , given $W _ { l } \in \mathbb { R } ^ { n _ { l - 1 } \times n _ { l } }$ and $\mathsf { b } _ { l } \in \mathbb { R } ^ { n _ { l } } , l = 1 , \hdots , L _ { \hphantom { ( } }$ , let 

$$
f _ {\theta} (x) = W _ {L + 1} \sigma \left(W _ {L} \dots \sigma \left(W _ {1} x + b _ {1}\right) \dots + b _ {L}\right) + b _ {L + 1}.
$$

The learnable parameters are $\theta = \{ W _ { 1 } , \ldots , W _ { L } , \mathsf { b } _ { 1 } , \ldots , \mathsf { b } _ { L } \}$ . 

## Examples:

▶ Multilayer Perceptron (MLP): fully connected layers (full matrices $W _ { l } )$ $\sigma = \mathsf { R e L U }$ , tanh - target: vector data. 

▶ Convolutional Neural Network (CNN): convolutional layers $( W _ { l } \mathbf { x } = \mathsf { K } _ { l } \ast \mathsf { X } )$ - target: image data. 

## Neural Networks: very expressive parametric functions from $\mathbb { I } _ { 1 } ^ { \perp } = \mathbb { I } _ { 2 } ^ { \perp }$

## Neural Networks: a working expression

Given a nonlinear function $\sigma : \mathbb { R }  \mathbb { R }$ and its element-wise version s.t. $[ \sigma ( \mathsf { x } ) ] _ { j } = \sigma ( [ \mathsf { x } ] _ { j } )$ , given $W _ { l } \in \mathbb { R } ^ { n _ { l - 1 } \times n _ { l } }$ and $\mathsf { b } _ { l } \in \mathbb { R } ^ { n _ { l } } , l = 1 , \hdots , L ,$ , let 

$$
f _ {\theta} (x) = W _ {L + 1} \sigma \left(W _ {L} \dots \sigma \left(W _ {1} x + b _ {1}\right) \dots + b _ {L}\right) + b _ {L + 1}.
$$

The learnable parameters are $\theta = \{ W _ { 1 } , \ldots , W _ { L } , \mathsf { b } _ { 1 } , \ldots , \mathsf { b } _ { L } \}$ 

## Examples:

▶ Multilayer Perceptron (MLP): fully connected layers (full matrices $W _ { l } ) _ { i }$ $\sigma = \mathsf { R e } \mathsf { L U } _ { \mathrm { i } }$ , tanh - target: vector data. 

▶ Convolutional Neural Network (CNN): convolutional layers $( W _ { l } \mathbf { x } = \mathsf { K } _ { l } \ast \mathsf { X } )$ - target: image data. 

## Theorem (Universal Approximation):

Under reasonable assumptions on $\sigma$ , any continuous function $f: [0,1]^n \to \mathbb{R}^n$ can be approximated arbitrarily well (in the $L^\infty$ norm) by a NN with $L = 1$ , provided that $W_1 \in \mathbb{R}^{n_1 \times n}$ and $W_2 \in \mathbb{R}^{n \times n_1}$ , and $n_1$ is sufficiently large. (Cybenko, 1989; Hornik, 1991) 

Variants: arbitrary width, higher regularity, avoid the curse of dimensionality 

## Network architectures for images: Convolutional Neural Networks (CNNs)

Key Idea: exploit spatial locality and translational invariance using local linear operations. 

▶ Layers apply convolutions: $\mathsf X \mapsto \sigma ( \mathsf K * \mathsf X + \mathsf b )$ 

▶ Common operations: ReLU, pooling, normalization; 

▶ Weight sharing between layers (parameter reduction) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/bd80a66dc6d55e13764e3dc4d1df918eb133ce46c0be9db175f2152e99c640fa.jpg)



Credits: Wikipedia, by Aphex34


## Network architectures for images: Convolutional Neural Networks (CNNs)

Key Idea: exploit spatial locality and translational invariance using local linear operations. 

▶ Layers apply convolutions: $\mathsf X \mapsto \sigma ( \mathsf K * \mathsf X + \mathsf b )$ 

▶ Common operations: ReLU, pooling, normalization; 

▶ Weight sharing between layers (parameter reduction) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/97beed78d0a635c3837dc11a20f3b98df09739756c4e618a2307409aab3aa0d4.jpg)



Credits: [Zhang et al., 2017]


## Image denoising example:

▶ DnCNN (Zhang et al., 2017): deep CNN trained to remove additive Gaussian noise. 

## Warning: input dimensions!

In these examples, the input of the network is not a vectorized image $\ b { \ b { \ b { x } } } \in \mathbb { R } ^ { n }$ but a tensor X of size: height × width × channels. 

## Network architectures for images: U-Nets

Key Idea: extract and process local features with symmetric skip connections. 

▶ An encoder branch extracts coarse features (convolution + downsampling) 

▶ A decoder branch reconstructs full-resolution output (upsampling) 

▶ Skip connections copy feature maps to enhance spatial details 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/e9d9ca4277a83879d38fb7015230b8294a5817ef409d083437aadc02734cae3d.jpg)



Credits: [Ronneberger, Fischer, Brox, 15]


## Network architectures for images: U-Nets

## Key Idea: extract and process local features with symmetric skip connections.

▶ An encoder branch extracts coarse features (convolution + downsampling) 

▶ A decoder branch reconstructs full-resolution output (upsampling) 

▶ Skip connections copy feature maps to enhance spatial details 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/b874069176835f528ce4c74a2686418f8ab0dc4b060a22c0a91d00f695fe8f8f.jpg)



Credits: [Ronneberger, Fischer, Brox, 15]


## Image denoising example:

▶ U-Net variants are widely used in medical image denoising, see e.g. Noise2Noise (Lehtinen et al., 2018), DRUNET (Devalla et al., 2018), ... 

## Network architectures for images: Vision Transformers

Key Idea: Process images as sequences of patches with self-attention, removing convolutional inductive bias. 

▶ Image is split into fixed-size patches (e.g., 16 × 16) 

▶ Each patch is embedded into a vector (via linear projection) 

▶ Transformer encoder applies global attention across all patches. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/490f5858-1ecd-4683-8742-a5d4709f58e8/36b5d5cd24298e2702aed0be49da4764913615933310c7494f40ffff1a01f4e4.jpg)


https://www.pinecone.io/learn/series/image-search/vision-transformers Image denoising example: 

▶ Restormer (Zamir et al., 2022): uses self-attention over multi-resolution image representations; can be applied to unsupervised settings. 

A statistical learning perspective on image denoising 

D. Optimization algorithms 

## How to train your network - part

Training a network: the process of finding θ that minimizes $\hat { \mathcal { L } } ( \theta ) = \hat { L } ( f _ { \theta } )$ 

## How to train your network - part I

Training a network: the process of finding $\theta$ that minimizes $\hat { \mathcal { L } } ( \theta ) = \hat { L } ( f _ { \theta } )$ An optimizer is a numerical algorithm to do so. Key features: 

1. first-order schemes: leverages derivatives of $\hat { \mathcal { L } }$ in $\theta ;$ 

2. stochastic optimization: it exploits random batches to reduce computations. 

## How to train your network - part I

Training a network: the process of finding $\theta$ that minimizes $\hat { \mathcal { L } } ( \theta ) = \hat { L } ( f _ { \theta } )$ An optimizer is a numerical algorithm to do so. Key features: 

1. first-order schemes: leverages derivatives of $\hat { \mathcal { L } }$ in $\theta ;$ 

2. stochastic optimization: it exploits random batches to reduce computations. 

## Backpropagation: just an intuition

▶ Exploit the compositional expression of $f _ { \theta } { \mathrm { : } }$ : repeated chain rule. 

▶ Use automated diferentiation and zero-order methods. 

▶ Eficiency: proportional to the forward pass (computing $f _ { \theta } )$ 

## How to train your network - part I

Training a network: the process of finding $\theta$ that minimizes $\hat { \mathcal { L } } ( \theta ) = \hat { L } ( f _ { \theta } )$ An optimizer is a numerical algorithm to do so. Key features: 

1. first-order schemes: leverages derivatives of $\hat { \mathcal { L } }$ in $\theta ;$ 

2. stochastic optimization: it exploits random batches to reduce computations. 

## Backpropagation: just an intuition

▶ Exploit the compositional expression of $f _ { \theta } { \mathrm { : } }$ : repeated chain rule. 

▶ Use automated diferentiation and zero-order methods. 

▶ Eficiency: proportional to the forward pass (computing $f _ { \theta } )$ 

## Stochastic Gradient Descent: just an intuition

Exploit that $\begin{array} { r } { \hat { \mathcal { L } } ( \boldsymbol { \theta } ) = \sum _ { i = 1 } ^ { N } \ell ( f _ { \boldsymbol { \theta } } ( \mathsf { y } _ { i } ) , \mathsf { x } _ { i } ) } \end{array}$ : replace the full sum with the one on a randomly subsampled batch. 

▶ Combined with momentum, acceleration, and adaptive step sizes, it provides eficient versions (Adam, AdaGrad, RMSProp) 

Other keywords involved in the training process of a network: 

## How to train your network - part II

Other keywords involved in the training process of a network: 

## Initialization:

[What?] Weights are initialized randomly (some methods: Xavier, He). 

[Why?] Avoid symmetries, preserve variance across layers. 

## How to train your network - part II

Other keywords involved in the training process of a network: 

## Initialization:

[What?] Weights are initialized randomly (some methods: Xavier, He). 

[Why?] Avoid symmetries, preserve variance across layers. 

## Scheduling:

[What?] Dynamically adjusts the learning rate during training 

[Why?] Crucial for stable convergence and escaping local minima. 

## How to train your network - part II

Other keywords involved in the training process of a network: 

## Initialization:

[What?] Weights are initialized randomly (some methods: Xavier, He). 

[Why?] Avoid symmetries, preserve variance across layers. 

## Scheduling:

[What?] Dynamically adjusts the learning rate during training 

[Why?] Crucial for stable convergence and escaping local minima. 

## Early Stopping:

[What?] Interrupt training when the performance no longer improves. 

[Why?] It prevents overfitting and reduces unnecessary computation. 

## How to train your network - part II

Other keywords involved in the training process of a network: 

## Initialization:

[What?] Weights are initialized randomly (some methods: Xavier, He). 

[Why?] Avoid symmetries, preserve variance across layers. 

## Scheduling:

[What?] Dynamically adjusts the learning rate during training 

[Why?] Crucial for stable convergence and escaping local minima. 

## Early Stopping:

[What?] Interrupt training when the performance no longer improves. 

[Why?] It prevents overfitting and reduces unnecessary computation. 

## Validation set:

[What?] A separate dataset used to evaluate generalization. 

[Why?] Hyperparameter tuning (parameters of the network - number of layers, channels - or of the optimizer - learning rate, batch size). 

Training outcome: $\widetilde { \theta } \left( \sim f _ { \widetilde { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ through a numerical method 

## Learning, from the errors

Training outcome: $\widetilde { \theta } \left( \sim f _ { \widetilde { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ through a numerical method 

Empirical target: $\widehat { \theta } \left( \sim f _ { \widehat { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ 

## Learning, from the errors

Training outcome: $\widetilde { \theta } \left( \sim f _ { \widetilde { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ through a numerical method 

Empirical target: $\widehat { \theta } \left( \sim f _ { \widehat { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ 

Optimal target: $\theta ^ { \star } ( \sim f _ { \theta ^ { \star } } )$ obtained by minimizing $\mathcal { L } ( \theta ) = L ( f _ { \theta } )$ over Θ 

## Learning, from the errors

Training outcome: $\widetilde { \theta } \left( \sim f _ { \widetilde { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ through a numerical method 

Empirical target: $\widehat { \theta } \left( \sim f _ { \widehat { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ 

Optimal target: $\theta ^ { \star } ( \sim f _ { \theta ^ { \star } } )$ obtained by minimizing $\mathcal { L } ( \theta ) = L ( f _ { \theta } )$ over Θ 

Bayes estimator: $f ^ { \star } = \mathbb { E } _ { \pi } [ x | y = \cdot ]$ obtained by minimizing L on all measurable functions $Y  X$ 

## Learning, from the errors

Training outcome: $\widetilde { \theta } \left( \sim f _ { \widetilde { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ through a numerical method 

Empirical target: $\widehat { \theta } \left( \sim f _ { \widehat { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ 

Optimal target: $\theta ^ { \star } ( \sim f _ { \theta ^ { \star } } )$ obtained by minimizing $\mathcal { L } ( \theta ) = L ( f _ { \theta } )$ over Θ 

Bayes estimator: $f ^ { \star } = \mathbb { E } _ { \pi } [ x | y = \cdot ]$ obtained by minimizing L on all measurable functions $Y  X$ 

True solution: if there existed a way to connect $\mathcal { Y } \sim \mathcal { x }$ 

Training outcome: $\widetilde { \theta } \left( \sim f _ { \widetilde { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ through a numerical method 

Empirical target: $\widehat { \theta } \left( \sim f _ { \widehat { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ 

Optimal target: $\theta ^ { \star } ( \sim f _ { \theta ^ { \star } } )$ obtained by minimizing $\mathcal { L } ( \theta ) = L ( f _ { \theta } )$ over Θ 

Bayes estimator: $f ^ { \star } = \mathbb { E } _ { \pi } [ x | y = \cdot ]$ obtained by minimizing L on all measurable functions $Y  X$ 

True solution: if there existed a way to connect $\mathcal { Y } \sim \mathcal { x }$ 

Optimization error (how good is my optimizer?) 

Training outcome: $\widetilde { \theta } \left( \sim f _ { \widetilde { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ through a numerical method 

Empirical target: $\widehat { \theta } \left( \sim f _ { \widehat { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ 

Optimal target: $\theta ^ { \star } ( \sim f _ { \theta ^ { \star } } )$ obtained by minimizing $\mathcal { L } ( \theta ) = L ( f _ { \theta } )$ over Θ 

Bayes estimator: $f ^ { \star } = \mathbb { E } _ { \pi } [ x | y = \cdot ]$ obtained by minimizing L on all measurable functions $Y  X$ 

True solution: if there existed a way to connect $\mathcal { Y } \sim \mathcal { x }$ 

Optimization error (how good is my optimizer?) 

Sample error (how much does my result depend on the training sample?) 

Training outcome: $\widetilde { \theta } \left( \sim f _ { \widetilde { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ through a numerical method 

Empirical target: $\widehat { \theta } \left( \sim f _ { \widehat { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ <sup>b</sup>over Θ 

Optimal target: $\theta ^ { \star } ( \sim f _ { \theta ^ { \star } } )$ obtained by minimizing $\mathcal { L } ( \theta ) = L ( f _ { \theta } )$ over Θ 

Bayes estimator: $f ^ { \star } = \mathbb { E } _ { \pi } [ x | y = \cdot ]$ obtained by minimizing L on all measurable functions $Y  X$ 

True solution: if there existed a way to connect $\mathcal { Y } \sim \mathcal { x }$ 

Optimization error (how good is my optimizer?) 

Sample error (how much does my result depend on the training sample?) 

Approximation error (how expressive is my parametric space of regularizers?) 

## Learning, from the errors

Training outcome: $\widetilde { \theta } \left( \sim f _ { \widetilde { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ through a numerical method 

Empirical target: $\widehat { \theta } \left( \sim f _ { \widehat { \theta } } \right)$ obtained by minimizing $\widehat { \mathcal { L } } ( \theta )$ over Θ 

Optimal target: $\theta ^ { \star } ( \sim f _ { \theta ^ { \star } } )$ obtained by minimizing $\mathcal { L } ( \theta ) = L ( f _ { \theta } )$ over Θ 

Bayes estimator: $f ^ { \star } = \mathbb { E } _ { \pi } [ x | y = \cdot ]$ obtained by minimizing L on all measurable functions $Y  X$ 

True solution: if there existed a way to connect $\mathcal { Y } \sim \mathcal { x }$ 

Optimization error (how good is my optimizer?) 

(how much does my result depend on the training sample?) 

Approximation error (how expressive is my parametric space of regularizers?) 

Irreducible error (if data are noisy, I can’t reduce this!) 

Implementation aspects 

## Image processing in Python

The most important libraries for image processing: 

▶ numpy: process tensors representing images - can handle arithmetic operations, cropping... 

▶ skimage: basic image processing operations (thresholding, blurring,...) and metrics (MSE, PSNR, SSIM,...) 

▶ matplotlib: image visualization, color adjustment,... 

▶ pytorch: neural network - definition, training, testing. 

A small tutorial: tomorrow! 