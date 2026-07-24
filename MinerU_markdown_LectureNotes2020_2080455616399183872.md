Inverse Problems 

Lecture notes, Michaelmas term 2020 University of Cambridge 

Yury Korolev and Jonas Latz 

May 18, 2021 

## Contents

1 Introduction to Inverse Problems 7
1.1 Well-posed and ill-posed problems 7
1.2 Examples of inverse problems 9
1.2.1 Signal deblurring 9
1.2.2 Heat equation 9
1.2.3 Differentiation 10
1.2.4 Matrix inversion 11
1.2.5 Tomography 12
1.2.6 Groundwater flow/hydraulic tomography 14

2 Generalised Solutions 15
2.1 Generalised Inverses 17
2.2 Compact Operators 21

3 Classical Regularisation Theory 27
3.1 What is Regularisation? 27
3.2 Parameter Choice Rules 29
3.2.1 A priori parameter choice rules 30
3.2.2 A posteriori parameter choice rules 31
3.2.3 Heuristic parameter choice rules 31
3.3 Spectral Regularisation 32
3.3.1 Truncated singular value decomposition 33
3.3.2 Tikhonov regularisation 34

4 Variational Regularisation 35
4.1 Background 35
4.1.1 Banach spaces and weak convergence 35
4.1.2 Convex analysis 38
4.1.3 Minimisers 43
4.1.4 Duality in convex optimisation 45
4.2 Well-posedness and Regularisation Properties 46
4.3 Total Variation Regularisation 52

5 Convex Duality 57
5.1 Dual Problem 58
5.2 Source Condition and Convergence Rates 60 

6 Bayesian probability and statistics 65  
6.1 From inverse problems to Bayesian inverse problems 65  
6.2 Reminder: measure, probability, and integration 66  
6.3 Conditional probability 71  
6.4 Bayesian statistics 75  
6.4.1 Statistical models 75  
6.4.2 Bayes' formula 76  
7 Bayesian inverse problems and well-posedness 79  
7.1 Bayesian inverse problems 79  
7.2 Metrics on spaces of probability measures 80  
7.3 Stability 81  
8 Function space priors and Monte Carlo 85  
8.1 Gaussian measures 85  
8.2 Monte Carlo techniques 88  
8.2.1 Standard Monte Carlo 88 

These lecture notes are based on the Inverse Problems course taught by Yury Korolev and Jonas Latz in Michaelmas term 2020 at the University of Cambridge.<sup>1</sup> Complementary material can be found in the following books, lecture notes and review papers: 

1. Heinz Werner Engl, Martin Hanke, and Andreas Neubauer. Regularization of Inverse Problems. Springer, 1996. 

2. Otmar Scherzer, Markus Grasmair, Harald Grossauer, Markus Haltmeier and Frank Lenzen. Variational Methods in Imaging. Springer, 2008. 

3. Kristian Bredies and Dirk Lorenz. Mathematical Image Processing. Springer, 2018 

4. Martin Benning and Martin Burger. Modern regularization methods for inverse problems. Acta Numerica, 27, 1-111 (2018) 

https://www.cambridge.org/core/journals/acta-numerica/article/ modern-regularization-methods-for-inverse-problems/ 1C84F0E91BF20EC36D8E846EF8CCB830 

5. K.Saxe. Beginning Functional Analysis. Springer, 2002 

6. Masoumeh Dashti and Andrew M. Stuart, The Bayesian approach to inverse problems, Handbook of Uncertainty Quantification, 2016. 

7. Jari Kaipio and Erkki Somersalo, Statistical and computational inverse problems, vol. 160 of Applied Mathematical Sciences, 2005. 

8. O. Kallenberg, Foundations of modern probability theory, Springer, 1997. 

9. Andrew M. Stuart, Inverse problems: a Bayesian perspective, Acta Numerica, 2010. 

These lecture notes are under constant redevelopment and might contain typos or errors. We very much appreciate if you report any mistakes found (to y.korolev@damtp.cam.ac.uk or jl2160@cam.ac.uk). Thanks! 

## Chapter 1

## Introduction to Inverse Problems

Inverse problems arise from the need to gain information about an unknown object of interest from given indirect measurements. Inverse problems have several applications varying from medical imaging and industrial process monitoring to ozone layer tomography and modelling of financial markets. The common feature for inverse problems is the need to understand indirect measurements and to overcome extreme sensitivity to noise and modelling inaccuracies. In this course we employ both deterministic and probabilistic approach to inverse problems to find stable and meaningful solutions that allow us quantify how inaccuracies in the data or model afect the obtained estimate. 

## 1.1 Well-posed and ill-posed problems

We start by considering the problem of finding $u \in \mathbb { R } ^ { d }$ that satisfies the equation 

$$
f = A u,\tag{1.1}
$$

where $f \in \mathbb { R } ^ { k }$ is given. We refer to f as observed data or measurement and u as an unknown. The physical phenomena that relates the unknown and the measurement is modelled by a matrix $A \in \mathbb { R } ^ { k \times d }$ . In real life the perfect data given in (1.1) is perturbed by noise and we observe measurements 

$$
f _ {n} = A u + n,\tag{1.2}
$$

where $\boldsymbol { n } \in \mathbb { R } ^ { k }$ represents the observational noise. 

We are interested in ill-posed inverse problems, where the inverse problem is more dificult to solve than the direct problem of finding $f _ { n }$ when u is given. To explain this we first need to introduce well-posedness as defined by Jacques Hadamard: 

Definition 1.1.1. A problem is called well-posed if 

1. There exists at least one solution. (Existence) 

2. There is at most one solution. (Uniqueness) 

3. The solution depends continuously on data. (Stability) 

The direct or forward problem is assumed to be well-posed. The inverse problems are ill-posed and break at least one of the above conditions. 

1. Assume that $d < k$ and $A : \mathbb { R } ^ { d }  \mathcal { R } ( A ) \subsetneq \mathbb { R } ^ { k }$ , where the range of A is a proper subset of $\mathbb { R } ^ { k }$ . Furthermore, we assume that A has a unique inverse $A ^ { - 1 } : { \mathcal { R } } ( A ) \to \mathbb { R } ^ { k }$ Because of the noise in the measurement $f _ { n } \not \in { \mathcal { R } } ( A )$ so that simply inverting A with the data given in (1.2) is not possible. Note that usually only the statistical properties of the noise n are known so we cannot just subtract it. 

2. Assume next that $d > k$ and $A : \mathbb { R } ^ { d }  \mathbb { R } ^ { k }$ , in which case the system is underdetermined. We then have more unknowns than equations which means that there are several possible solutions. 

3. Consider next case $d = k$ and there exist $A ^ { - 1 } : \mathbb { R } ^ { k }  \mathbb { R } ^ { k }$ but the condition number $\kappa = \lambda _ { 1 } / \lambda _ { k }$ , where $\lambda _ { 1 }$ and $\lambda _ { k }$ are the biggest and smallest eigenvalues of A, is very large. Such a matrix is said to be ill-conditioned and is almost singular. In this case the problem is sensitive even to smallest errors in the measurement. Hence the naive reconstruction $\widetilde { u } = A ^ { - 1 } f _ { n } = u + A ^ { - 1 } n$ does not produce a meaningful solution but will be dominated by $A ^ { - 1 } n$ . Note that $\| A ^ { - 1 } n \| _ { 2 } \approx \| n \| _ { 2 } / \lambda _ { k }$ can be arbitrarily large. 

The last part illustrates one of the key perspectives of inverse problem theory; How can we stabilise the reconstruction process while maintaining acceptable accuracy? 

A deterministic way of achieving a unique and stable solution for the problem (1.2) is to use regularisation theory. In the classical Tikhonov regularisation a solution is attained by solving 

$$
\min _ {u \in \mathbb {R} ^ {d}} \left(\| A u - f _ {n} \| ^ {2} + \alpha \| L u \| ^ {2}\right).\tag{1.3}
$$

Above α acts as a tuning parameter balancing the efect of the data fidelity term $\| A u - f _ { n } \| ^ { 2 }$ and the stabilising regularisation term $\| u \| ^ { 2 }$ . The first half of the course will concentrate on regularisation theory. 

Another way of tackling problems arising from ill-posedness is Bayesian inversion. The idea of statistical inversion methods is to rephrase the inverse problem as a question of statistical inference. We then consider problem 

$$
F = A U + N,\tag{1.4}
$$

where the measurement, unknown and noise are now modelled as random variables. This approach allows us to model the noise through its statistical properties. We can also encode our a priori knowledge of the unknown in form of a probability distribution that assigns higher probability to those values of u we expect to see. The solution to (1.4) is so-called posterior distribution, which is the conditional probability distribution of u given a measurement m. This distribution can then be used to obtain estimates that are most likely in some sense. We will return to the Bayesian approach to inverse problems in the second half of the course 

In this course we will concentrate on continuous inverse problems where in (1.1) and (1.2) $A : X  Y$ is a linear or non-linear forward operator acting between some spaces X and Y , typically Hilbert or Banach spaces, the measured data $f _ { n } \in Y$ is a function and $u \in X$ is the quantity we want to reconstruct from the data. Linear inverse problems include such important applications as computer tomography, magnetic resonance imaging and image deblurring in microscopy or astronomy. In other important applications, such as seismic imaging, the forward operator is non-linear (e.g., parameter identification problems for PDEs). Next we will take a look at some examples of linear and non-linear inverse problems to see what kind of challenges we face when trying to solve them. 

## 1.2 Examples of inverse problems

## 1.2.1 Signal deblurring

The deblurring (or deconvolution) problem of recovering an input signal u form an observed signal 

$$
f _ {n} (t) = \int_ {- \infty} ^ {\infty} a (t - s) u (s) d s + n (t)
$$

occurs in many imaging, and image- and signal processing applications. Here the function a is known as the blurring kernel. 

The noiseless data is given by $\begin{array} { r } { f ( t ) = \int _ { - \infty } ^ { \infty } a ( t - s ) u ( s ) d s } \end{array}$ and its Fourier transform is $\begin{array} { r } { \widehat { f } ( \xi ) = \int _ { - \infty } ^ { \infty } e ^ { - i \xi t } f ( t ) d t } \end{array}$ . The convolution theorem implies 

$$
\widehat {f} (\xi) = \widehat {a} (\xi) \widehat {u} (\xi),
$$

and hence by inverse Fourier transform 

$$
u (t) = \frac {1}{2 \pi} \int_ {- \infty} ^ {\infty} e ^ {i t \xi} \frac {\widehat {f} (\xi)}{\widehat {a} (\xi)} d \xi .
$$

However, we can only observe noisy measurements and hence we have on the frequency domain ${ \widehat { f _ { n } } } ( \xi ) = { \widehat { a } } ( \xi ) { \widehat { u } } ( \xi ) + { \widehat { n } } ( \xi )$ . The estimate $u _ { e s t }$ based on the convolution theorem is given by 

$$
u _ {e s t} (t) = u (t) + \frac {1}{2 \pi} \int_ {- \infty} ^ {\infty} e ^ {i t \xi} \frac {\widehat {n} (\xi)}{\widehat {a} (\xi)} d \xi ,
$$

which is often not even well defined, since usually the kernel a decreases exponentially (or has compact support), making the denominator small, whereas the Fourier transform of the noise will be non-zero. 

## 1.2.2 Heat equation

Next we study the problem of recovering the initial condition u of the heat equation from a noisy observation $f _ { n }$ of the solution at some time $T > 0$ . We consider the heat equation on a torus $\mathbb { T } ^ { d }$ , with Dirichlet boundary conditions 

$$
\left\{ \begin{array}{l l} \frac {d v}{d t} - \Delta v = 0 & \quad \text {on} \mathbb {T} ^ {d} \times \mathbb {R} _ {+} \\ v (x, t) = 0 & \quad \text {on} \partial \mathbb {T} ^ {d} \times \mathbb {R} _ {+} \\ v (x, T) = f (x) & \quad \text {on} \mathbb {T} ^ {d} \\ v (x, 0) = u (x) & \quad \text {on} \mathbb {T} ^ {d} \end{array} \right.
$$

where $\Delta$ denotes the Laplace operator and $\begin{array} { r } { \mathcal { D } ( \Delta ) = H _ { 0 } ^ { 1 } ( \mathbb { T } ^ { d } ) \cap H ^ { 2 } ( \mathbb { T } ^ { d } ) } \end{array}$ . Note that the operator $- \Delta$ is positive and self-adjoint on Hilbert space $\mathcal { H } = L ^ { 2 } (  { \mathbb { T } } ^ { d } )$ 

Given a function $u \in L ^ { 2 } (  { \mathbb { T } } ^ { d } )$ we can decompose it as a Fourier series 

$$
u (x) = \sum_ {n \in \mathbb {Z} ^ {d}} u _ {n} e ^ {2 \pi i \langle n, x \rangle},
$$

where $u _ { n } = \langle u , e ^ { 2 \pi i \langle n , x \rangle } \rangle$ are the Fourier coeficients, and the identity holds for almost every $\boldsymbol { x } \in \mathbb { T } ^ { d }$ . The $L ^ { 2 }$ norm of u is given by the Parseval’s identity $\| u \| _ { L ^ { 2 } } ^ { 2 } = \sum | u _ { n } | ^ { 2 }$ . Remember that the Sobolev space $H ^ { s } ( \mathbb { T } ^ { d } ) , \ s \in \mathbb { N }$ , consist of all $L ^ { 2 } (  { \mathbb { T } } ^ { d } )$ integrable functions whose $\alpha ^ { t h }$ order weak derivatives exist and are $L ^ { 2 } (  { \mathbb { T } } ^ { d } )$ integrable for all $| { \boldsymbol { \alpha } } | \leqslant s$ . The fractional Sobolev space $H ^ { s } ( \mathbb { T } ^ { d } )$ is given by the subspace of functions $u \in L ^ { 2 } (  { \mathbb { T } } ^ { d } )$ , such that 

$$
\| u \| _ {H ^ {s}} ^ {2} = \sum_ {n \in \mathbb {Z} ^ {d}} (1 + 4 \pi^ {2} | n | ^ {2}) ^ {s} | u _ {n} | ^ {2} <   \infty .\tag{1.5}
$$

Note that for a positive integer s, the above definition agrees with the definition $\mathrm { g i }$ ven using the weak derivatives. For $s < 0$ , we define $H ^ { s } ( \mathbb { T } ^ { d } )$ via duality or as the closure of $L ^ { 2 } (  { \mathbb { T } } ^ { d } )$ under the norm (1.5). The resulting spaces are separable for all $s \in \mathbb { R }$ 

The eigenvectors of $- \Delta$ in <sup>Td</sup> form the orthonormal basis of $L ^ { 2 } (  { \mathbb { T } } ^ { d } )$ and the eigenval ues are given by $4 \pi ^ { 2 } | n | ^ { 2 } , n \in \mathbb { Z } ^ { d }$ . We can also work on real-valued functions where the eigenfunctions $\{ \varphi _ { j } \} _ { j = 1 } ^ { \infty }$ comprise sine and cosine functions. The eigenvalues of $- \Delta$ , when ordered on a one-dimensional lattice, then satisfy $\lambda _ { j } \asymp j ^ { \frac { 2 } { d } }$ . The notation $\asymp$ means that there exist constants $C _ { 1 } , C _ { 2 } > 0$ , such that $C _ { 1 } j ^ { \frac { 2 } { d } } \leqslant \lambda _ { j } \leqslant C _ { 2 } j ^ { \frac { 2 } { d } }$ 

The solution to the forward heat equation can be written as 

$$
v (t) = \sum_ {j = 1} ^ {\infty} u _ {j} e ^ {- \lambda_ {j} t} \varphi_ {j}.
$$

We notice that 

$$
\| v (t) \| _ {H ^ {s}} ^ {2} \asymp \sum_ {j = 1} ^ {\infty} j ^ {\frac {2 s}{d}} e ^ {- 2 \lambda_ {j} t} | u _ {j} | ^ {2} = t ^ {- s} \sum_ {j = 1} ^ {\infty} (\lambda_ {j} t) ^ {s} e ^ {- 2 \lambda_ {j} t} | u _ {j} | ^ {2} \leqslant C t ^ {- s} \sum_ {j = 1} ^ {\infty} | u _ {j} | ^ {2} = C t ^ {- s} \| u \| _ {L ^ {2}}
$$

which implies that $v ( t ) \in H ^ { s } (  { \mathbb { T } } ^ { d } )$ for all $s > 0$ 

We now have observation model 

$$
f _ {n} = A u + n,
$$

where $A = e ^ { T \Delta }$ and n is the observational noise. The noise is not usually smooth (the often assumed white noise is not even an $L ^ { 2 }$ function) and hence measurement $f _ { n }$ is not in the image space $\mathcal { D } ( e ^ { T \Delta } ) \subset \cap _ { s > 0 } H ^ { s } ( \mathbb { T } ^ { d } )$ 

## 1.2.3 Diferentiation

Consider the problems of evaluation the derivative of a function $f \in L ^ { 2 } [ 0 , \pi / 2 ]$ . Let 

$$
D f = f ^ {\prime},
$$

where $D \colon L ^ { 2 } [ 0 , \pi / 2 ]  L ^ { 2 } [ 0 , \pi / 2 ]$ 

Proposition 1.2.1. The operator D is unbounded from $L ^ { 2 } [ 0 , \pi / 2 ]  L ^ { 2 } [ 0 , \pi / 2 ]$ 

Proof. Take a sequence $f _ { n } ( x ) = \sin ( n x ) , n = 1 , \ldots , \infty$ . Clearly, $f _ { n } \in L ^ { 2 } [ 0 , \pi / 2 ]$ for all n and $\| f _ { n } \| = { \sqrt { \frac { \pi } { 4 } } }$ . However, $D f _ { n } ( x ) = n \cos ( n x )$ and $\| D f _ { n } \| = n \to \infty$ as $n \to \infty$ . Therefore, D is unbounded. □ 

This shows that diferentiation is ill-posed from $L ^ { 2 }$ to $L ^ { 2 }$ . It does not mean that it can not be well-posed in other spaces. For instance, it is well-posed from $H ^ { 1 }$ (the Sobolev space of $L ^ { 2 }$ functions whose derivatives are also $L ^ { 2 } )$ to $L ^ { 2 }$ . Indeed, $\forall u \in H ^ { 1 }$ we get 

$$
\| D f \| _ {L ^ {2}} = \| f ^ {\prime} \| _ {L ^ {2}} \leqslant \| f \| _ {H ^ {1}} = \| f \| _ {L ^ {2}} + \| f ^ {\prime} \| _ {L ^ {2}}.
$$

However, since in practice we typically deal with functions corrupted by nonsmooth noise, the $L ^ { 2 }$ setting is practice-relevant, while the $H ^ { 1 }$ setting is not. 

Diferentiation can be written as an inverse problem for an integral equation. For instance, the derivative u of some function $f \in L ^ { 2 } [ 0 , 1 ]$ with $f ( 0 ) = 0$ satisfies 

$$
f (x) = \int_ {0} ^ {x} u (t) d t,
$$

which can be written as an operator equation $A u = f$ with $\textstyle ( A \cdot ) ( x ) : = \int _ { 0 } ^ { x } \cdot ( t ) d t$ 

## 1.2.4 Matrix inversion

In finite dimensions, the inverse problem (1.1) is a linear system. Linear systems are formally well-posed in the sense that the error in the solution is bounded by some constant times the error in the right-hand side, however, this constant depends on the condition number of the matrix A and can get arbitrary large for matrices with large condition numbers. In this case, we speak of ill-conditioned problems. 

Consider the problem (1.1) with $u \in \mathbb { R } ^ { n }$ and $f \in \mathbb { R } ^ { n }$ being n-dimensional vectors with real entries and $A \in \mathbb { R } ^ { n \times n }$ being a matrix with real entries. Assume further A to be symmetric and positive definite. 

We know from the spectral theory of symmetric matrices that there exist eigenvalues $\lambda _ { 1 } \ \geqslant \ \lambda _ { 2 } \ \geqslant \ . . . \geqslant \ \lambda _ { n } \ > \ 0$ and corresponding (orthonormal) eigenvectors $a _ { j } ~ \in ~ \mathbb { R } ^ { n }$ for $j \in \{ 1 , \ldots , n \}$ such that A can be written as 

$$
A = \sum_ {j = 1} ^ {n} \lambda_ {j} a _ {j} a _ {j} ^ {\top}.\tag{1.6}
$$

It is well known from numerical linear algebra that the condition number $\kappa = \lambda _ { 1 } / \lambda _ { n }$ is a measure of how stable (1.1) can be solved, which we will illustrate what follows. 

We assume that we measure $f _ { \delta }$ instead of $f ,$ with $\| f - f _ { \delta } \| _ { 2 } \leqslant \delta \| A \| = \delta \lambda _ { 1 }$ , where $\| \cdot \| _ { 2 }$ denotes the Euclidean norm of $\mathbb { R } ^ { n }$ and $\| A \|$ the operator norm of $A$ (which equals the largest eigenvalue of $A )$ . Then, if we further denote with $u _ { \delta }$ the solution of $A u _ { \delta } = f _ { \delta }$ , the diference between $u _ { \delta }$ and the solution u to (1.1) is 

$$
u - u _ {\delta} = \sum_ {j = 1} ^ {n} \lambda_ {j} ^ {- 1} a _ {j} a _ {j} ^ {\top} (f - f _ {\delta}).
$$

Therefore, we can estimate 

$$
\| u - u _ {\delta} \| _ {2} ^ {2} = \sum_ {j = 1} ^ {n} \lambda_ {j} ^ {- 2} \underbrace {\| a _ {j} \| _ {2} ^ {2}} _ {= 1} | a _ {j} ^ {\top} (f - f _ {\delta}) | ^ {2} \leqslant \lambda_ {n} ^ {- 2} \| f - f _ {\delta} \| _ {2} ^ {2},
$$

due to the orthonormality of eigenvectors, the Cauchy-Schwarz inequality, and $\lambda _ { n } \leqslant \lambda _ { j }$ Thus, taking square roots on both sides yields the estimate 

$$
\| u - u _ {\delta} \| _ {2} \leqslant \lambda_ {n} ^ {- 1} \| f - f _ {\delta} \| _ {2} \leqslant \kappa \delta .
$$

Hence, we observe that in the worst case an error δ in the data $y$ is amplified by the condition number κ of the matrix A. A matrix with large κ is therefore called ill-conditioned. We want to demonstrate the efect of this error amplification with a small example. 

Example 1.2.1. Let us consider the matrix 

$$
A = \left( \begin{array}{c c} 1 & 1 \\ 1 & \frac {1 0 0 1}{1 0 0 0} \end{array} \right),
$$

which has eigenvalues $\begin{array} { r } { \lambda _ { j } = 1 + \frac { 1 } { 2 0 0 0 } \pm \sqrt { 1 + \frac { 1 } { 2 0 0 0 ^ { 2 } } } } \end{array}$ , condition number $\kappa \approx 4 0 0 2 \gg 1$ , and operator norm $\| A \| \approx 2$ . For given data $f = ( 1 , 1 ) ^ { \top }$ the solution to $A u = f$ is $u = ( 1 , 0 ) ^ { \top }$ Now let us instead consider perturbed data $f _ { \delta } = ( 9 9 / 1 0 0 , 1 0 1 / 1 0 0 ) ^ { \top }$ . The solution u<sub>δ</sub> to $A u _ { \delta } = f _ { \delta }$ is then $u _ { \delta } = ( - 1 9 . 0 1 , 2 0 ) ^ { \top }$ 

Let us reflect on the amplification of the measurement error. By our initial assumption we find that $\delta = \| f - f _ { \delta } \| / \| A \| \approx \| ( 0 . 0 1 , - 0 . 0 1 ) ^ { \top } \| / 2 = \sqrt { 2 } / 2 0 0$ . Moreover, the norm of the error in the reconstruction is then $\lVert u - u _ { \delta } \rVert = \lVert ( 2 0 . 0 1 , 2 0 ) ^ { \top } \rVert \approx 2 0 \sqrt { 2 }$ . As a result, the amplification due to the perturbation is $\lVert u - u _ { \delta } \rVert / \delta \approx 4 0 0 0 \approx \kappa .$ 

## 1.2.5 Tomography

In almost any tomography application the underlying inverse problem is either the inversion of the Radon transform<sup>1</sup> or of the $\mathrm { X - r a y }$ transform. 

For $u \in C _ { 0 } ^ { \infty } ( \mathbb { R } ^ { n } ) , s \in \mathbb { R }$ , and $\theta \in S ^ { n - 1 }$ the Radon transform $R : C _ { 0 } ^ { \infty } ( \mathbb { R } ^ { n } ) \to C ^ { \infty } ( S ^ { n - 1 } \times$ <sup>R</sup>) can be defined as the integral operator 

$$
\begin{array}{c} f (\theta , s) = (\mathcal {R} u) (\theta , s) = \int_ {x \cdot \theta = s} u (x)   d x \\ = \int_ {\theta^ {\perp}} u (s \theta + y)   d y, \end{array}\tag{1.7}
$$

which, for $n = 2$ , coincides with the X-ray transform, 

$$
f (\theta , s) = (\mathcal {P} u) (\theta , s) = \int_ {\mathbb {R}} u (s \theta + t \theta^ {\perp}) d t,
$$

for $\theta \in S ^ { n - 1 }$ and $\theta ^ { \perp }$ being the vector orthogonal to θ. Hence, the X-ray transform (and therefore also the Radon transform in two dimensions) integrates the function u over lines in $\mathbb { R } ^ { n }$ , see Fig. 1.1<sup>2</sup>. 

Example 1.2.2. Let $n = 2$ . Then $S ^ { n - 1 }$ is simply the unit sphere $S ^ { 1 } = \{ \theta \in \mathbb { R } ^ { 2 } \ | \ \| \theta \| = 1 \}$ We can choose for instance $\theta \ : = \ : ( \cos ( \varphi ) , \sin ( \varphi ) ) ^ { \top }$ , for $\varphi \in [ 0 , 2 \pi )$ , and parametrise the Radon transform in terms of $\varphi$ and s, i.e. 

$$
f (\varphi , s) = (\mathcal {R} u) (\varphi , s) = \int_ {\mathbb {R}} u (s \cos (\varphi) - t \sin (\varphi), s \sin (\varphi) + t \cos (\varphi)) d t.\tag{1.8}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/7800197f25a457f7fc0e60902c7acd461f2273f6228c30a94379e69fdcdedb3a.jpg)



Figure 1.1: Visualization of the Radon transform in two dimensions (which coincides with the X-ray transform). The function u is integrated over the ray parametrized by θ and $s . { } ^ { 3 }$


Note that—with respect to the origin of the reference coordinate system—ϕ determines the angle of the line along one wants to integrate, while s is the ofset from that line from the centre of the coordinate system. 

It can be shown that the Radon transform is linear and continuous, i.e. $R \in \mathcal L ( L ^ { 2 } ( B ) , L ^ { 2 } ( Z ) )$ , and even compact. 

In X-ray Computed Tomography (CT), the unknown quantity u represents a spatially varying density that is exposed to X-radiation from diferent angles, and that absorbs the radiation according to its material or biological properties. 

The basic modelling assumption for the intensity decay of an X-ray beam is that within a small distance $\Delta t$ it is proportional to the intensity itself, the density, and the distance, i.e. 

$$
\frac {I (x + (t + \Delta t) \theta) - I (x + t \theta)}{\Delta t} = - I (x + t \theta) u (x + t \theta),
$$

for $x \in \theta ^ { \perp }$ . By taking the limit $\Delta t \to 0$ we end up with the ordinary diferential equation 

$$
\frac {d}{d t} I (x + t \theta) = - I (x + t \theta) u (x + t \theta),\tag{1.9}
$$

Let $R > 0$ be the radius of the domain of interest centred at the origin. Then, we integrate (1.9) from $t = - \sqrt { R ^ { 2 } - \| x \| _ { 2 } ^ { 2 } }$ , the position of the emitter, to $t = \sqrt { R ^ { 2 } - \| x \| _ { 2 } ^ { 2 } }$ , the position of the detector, and obtain 

$$
\int_ {- \sqrt {R ^ {2} - \| x \| _ {2} ^ {2}}} ^ {\sqrt {R ^ {2} - \| x \| _ {2} ^ {2}}} \frac {\frac {d}{d t} I (x + t \theta)}{I (x + t \theta)} d t = - \int_ {- \sqrt {R ^ {2} - \| x \| _ {2} ^ {2}}} ^ {\sqrt {R ^ {2} - \| x \| _ {2} ^ {2}}} u (x + t \theta) d t.
$$

Note that, due to d/dx $\log ( f ( x ) ) = f ^ { \prime } ( x ) / f ( x )$ , the left hand side in the above equation simplifies to 

$$
\int_ {- \sqrt {R ^ {2} - \| x \| _ {2} ^ {2}}} ^ {\sqrt {R ^ {2} - \| x \| _ {2} ^ {2}}} \frac {\frac {d}{d t} I (x + t \theta)}{I (x + t \theta)} d t = \log \left(I \left(x + \sqrt {R ^ {2} - \| x \| _ {2} ^ {2}} \theta\right)\right) - \log \left(I \left(x - \sqrt {R ^ {2} - \| x \| _ {2} ^ {2}} \theta\right)\right)
$$

As we know the radiation intensity at both the emitter and the detector, we therefore know $f ( x , \theta ) = \log ( I ( x - \theta { \sqrt { R ^ { 2 } - \| x \| _ { 2 } ^ { 2 } } } ) ) - \log ( I ( x + \theta { \sqrt { R ^ { 2 } - \| x \| _ { 2 } ^ { 2 } } } ) )$ and we can write the estimation of the unknown density u as the inverse problem of the X-ray transform (1.8) (if we further assume that u can be continuously extended to zero outside of the circle of radius R). 

## 1.2.6 Groundwater flow/hydraulic tomography

One goal in hydraulic tomography is to estimate the permeability of a groundwater reservoir. The permeability describes the conductivity of the groundwater reservoir and is, e.g., used to estimate the travel time of toxic or radioactive particles in the groundwater. 

To estimate the permeability, the water pressure in several position within the reservoir is measured. Pressure head and permeability are linked through Darcy’s law and the (assumed) incompressibility of water. 

Let $D \subseteq \mathbb { R } ^ { d } \ ( d = 1 , 2 , 3 )$ be an open, bounded, connected set with smooth boundary representing the groundwater reservoir. Let $a : { \overline { { D } } } \to ( 0 , \infty )$ be a continuously diferentiable function representing the permeability and let $s : \overline { { D } } \to \mathbb { R }$ be a continuous function representing the water sources in the reservoir. Furthermore, assume that the water pressure is 0 outside of D. 

Darcy’s law states that the pressure $p : D  \mathbb { R }$ , the flux $\vec { q } : D \to \mathbb { R } ^ { d }$ , and the permeability in the reservoir are related as follows: 

$$
\vec {q} (x) = - a (x) \nabla p (x) \qquad (x \in D).
$$

Incompressibility on the other hand requires that the divergence of the flux is fully controlled by in- and outflow given through the source term s: 

$$
\nabla \cdot \vec {q} (x) = s (x) \quad (x \in D).
$$

Finally, we can combine these assertions and obtain the elliptic partial diferential equation 

$$
\begin{array}{c} - \nabla \cdot a (x) \nabla p (x) = s (x) \\ p (x) = 0 \end{array}
$$

$$
\begin{array}{c} (x \in D) \\ (x \in \partial D). \end{array}
$$

In the described set-up, we now observe the pressure $p$ in several positions $x _ { 1 } , \dots , x _ { I } \in D$ $\mathrm { e . g . }$ , we observe $f _ { n } = ( p ( x _ { i } ) : i = 1 , . . . , I ) + n$ . We consider the inverse problem consisting in the estimation of the permeability a using the pressure measurements $f _ { n }$ . Indeed, using noisy point evaluations of the solution of the partial diferential equation, we try to estimate its difusion coeficient. Note that the map $a \mapsto ( p ( x _ { i } ) : i = 1 , . . . , I )$ is non-linear. Hence, this inverse problem is a non-linear inverse problem. 

## Chapter 2

## Generalised Solutions

Functional analysis is the basis of the theory that we will cover in this course. We cannot recall all basic concepts of functional analysis and instead refer to popular textbooks that deal with this subject, e.g., [12, 37, 33]. Nevertheless, we shall recall a few important definitions that will be used in this lecture. 

We will focus on inverse problems with bounded linear operators A, i.e. $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ with 

$$
\| A \| _ {\mathcal {L} (\mathcal {X}, \mathcal {Y})} := \sup _ {u \in \mathcal {X} \backslash \{0 \}} \frac {\| A u \| _ {\mathcal {Y}}}{\| u \| _ {\mathcal {X}}} = \sup _ {\| u \| _ {\mathcal {X}} \leqslant 1} \| A u \| _ {\mathcal {Y}} <   \infty .
$$

For $A \colon \mathcal { X } \to \mathcal { Y }$ we further want to denote by 

1. ${ \mathcal { D } } ( A ) : = { \mathcal { X } }$ the domain, 

2. ${ \mathcal { N } } ( A ) : = \{ u \in { \mathcal { X } } \mid A u = 0 \}$ the kernel, 

3. ${ \mathcal { R } } ( A ) : = \{ f \in { \mathcal { Y } } \mid f = A u , u \in { \mathcal { X } } \}$ the range 

of A. 

We say that A is continuous at $u \in \mathcal X$ if for all $\varepsilon > 0$ there exists $\delta > 0$ with 

$$
\| A u - A v \| _ {\mathcal {Y}} \leqslant \varepsilon \text {   for   all   } v \in \mathcal {X} \text {   with   } \| u - v \| _ {\mathcal {X}} \leqslant \delta .
$$

For linear K it can be shown that continuity is equivalent to boundedness, i.e. the existence of a constant $C > 0$ such that 

$$
\| A u \| _ {\mathcal {Y}} \leqslant C \| u \| _ {\mathcal {X}}
$$

for all $u \in \mathcal X$ . Note that this constant C actually equals the operator norm $\| A \| _ { \mathcal { L } ( \mathcal { X } , \mathcal { Y } ) }$ 

In this Chapter we only consider $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ with and being Hilbert spaces. From functional calculus we know that every Hilbert space is equipped with a scalar product, which we are going to denote by $\langle \cdot , \cdot \rangle _ { \mathscr { U } }$ (or simply $\langle \cdot , \cdot \rangle$ , whenever the space is clear from the context). In analogy to the transpose of a matrix, this scalar product structure together with the theorem of Fr´echet-Riesz [37, Section 2.10, Theorem 2.E] allows us to define the (unique) adjoint operator of $A ,$ denoted with $A ^ { * }$ , as follows: 

$$
\langle A u, v \rangle_ {\mathcal {Y}} = \langle u, A ^ {*} v \rangle_ {\mathcal {X}}, \text {   for   all   } u \in \mathcal {X}, v \in \mathcal {Y}.
$$

In addition to that, a scalar product can be used to define orthogonality. Two elements $u , v \in \mathcal { X }$ are said to be orthogonal if $\langle u , v \rangle = 0$ . For a subset $\mathcal { X } ^ { \prime } \subset \mathcal { X }$ the orthogonal complement of $\mathcal { X } ^ { \prime }$ in $\mathcal { X }$ is defined as 

$$
\mathcal {X} ^ {\prime \perp} := \left\{u \in \mathcal {X} \mid \langle u, v \rangle_ {\mathcal {X}} = 0 \text {   for   all   } v \in \mathcal {X} ^ {\prime} \right\}.
$$

One can show that $\mathcal { X } ^ { \prime \perp }$ is a closed subspace and that $\mathcal { X } ^ { \perp } = \{ 0 \}$ . Moreover, we have that $\mathcal { X } ^ { \prime } \subset ( \mathcal { X } ^ { \prime \bot } ) ^ { \bot }$ . If $\mathcal { X } ^ { \prime }$ is a closed subspace then we even have $\mathcal { X } ^ { \prime } = ( \mathcal { X } ^ { \prime \bot } ) ^ { \bot }$ . In this case there exists the orthogonal decomposition 

$$
\mathcal {X} = \mathcal {X} ^ {\prime} \oplus \mathcal {X} ^ {\prime \perp},
$$

which means that every element $u \in \mathcal X$ can uniquely be represented as 

$$
u = x + x ^ {\perp} \mathrm{with} x \in \mathcal {X} ^ {\prime} \mathrm{and} x ^ {\perp} \in \mathcal {X} ^ {\prime \perp},
$$

see for instance [37, Section 2.9, Corollary 1]. 

The mapping $u \mapsto x$ defines a linear operator $P _ { \mathcal { X } ^ { \prime } } \in \mathcal { L } ( \mathcal { X } , \mathcal { X } )$ that is called orthogonal projection on $\mathcal { X } ^ { \prime } .$ 

Lemma 2.0.1 (cf. [28, Section 5.16]). Let $\mathcal { X } ^ { \prime } \subset \mathcal { X }$ be a closed subspace. The orthogonal projection onto $\mathcal { X } ^ { \prime }$ satisfies the following conditions: 

1. $P _ { X ^ { \prime } }$ is self-adjoint, i.e. $P _ { \chi ^ { \prime } } ^ { * } = P _ { \chi ^ { \prime } }$ 2 

2. $\| P \chi ^ { \prime } \| _ { \mathcal { L } ( \mathcal { X } , \mathcal { X } ) } = 1 ~ ( i f ~ \mathcal { X } ^ { \prime } \neq \{ 0 \} ) ,$ 

3. $I - P _ { \mathcal { X } ^ { \prime } } = P _ { \mathcal { X } ^ { \prime } \bot }$ 

4. $\begin{array} { r } { \| u - P _ { \mathcal { X } ^ { \prime } } u \| _ { \mathcal { X } } \leqslant \| u - v \| _ { \mathcal { X } } \ f o r \ a l l \ v \in \mathcal { X } ^ { \prime } , } \end{array}$ 

5. $x = P _ { \mathcal { X } ^ { \prime } } u$ if and only $i f x \in \mathcal { X } ^ { \prime }$ and $u - x \in \mathcal { X } ^ { \prime \perp }$ 

Remark 2.0.2. Note that for a non-closed subspace $\mathcal { X } ^ { \prime }$ we only have $( \mathcal { X } ^ { \prime \bot } ) ^ { \bot } = \overline { { \mathcal { X } ^ { \prime } } }$ . For $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ we therefore have 

$\mathcal { R } ( A ) ^ { \perp } = \mathcal { N } ( A ^ { * } )$ and thus $\mathcal { N } ( A ^ { * } ) ^ { \perp } = \overline { { \mathcal { R } ( A ) } }$ 

$\mathcal { R } ( A ^ { * } ) ^ { \perp } = \mathcal { N } ( A )$ and thus ${ \mathcal { N } } ( A ) ^ { \perp } = { \overline { { { \mathcal { R } } ( A ^ { * } ) } } }$ 

Hence, we can deduce the following orthogonal decompositions 

$$
\mathcal {X} = \mathcal {N} (A) \oplus \overline {{\mathcal {R} (A ^ {*})}} \text {   and   } \mathcal {Y} = \mathcal {N} (A ^ {*}) \oplus \overline {{\mathcal {R} (A)}}.
$$

We will also need the follwoing relationship between the ranges of $A ^ { * }$ and $A ^ { * } A$ 

Lemma 2.0.3. Let $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ . Then ${ \overline { { \mathcal { R } ( A ^ { * } A ) } } } = { \overline { { \mathcal { R } ( A ^ { * } ) } } }$ 

Proof. It is clear that $\overline { { \mathcal { R } ( A ^ { * } A ) } } = \overline { { \mathcal { R } ( A ^ { * } | _ { \mathcal { R } ( A ) } ) } } \subseteq \overline { { \mathcal { R } ( A ^ { * } ) } }$ , so we are left to prove that ${ \overline { { \mathcal { R } ( A ^ { * } ) } } } \subseteq$ $\overline { { \mathcal { R } ( A ^ { * } A ) } }$ 

Let $u \in \mathcal { R } ( A ^ { * } )$ and let $\varepsilon > 0$ . Then, there exists $f \in \mathcal { N } ( A ^ { * } ) ^ { \perp } = \overline { { \mathcal { R } ( A ) } }$ with $\| A ^ { * } f - u \| _ { \mathcal { X } } <$ $\varepsilon / 2$ (recall the orthogonal decomposition in Remark 2.0.2). $\operatorname { A s } { \mathcal { N } } ( A ^ { * } ) ^ { \perp } = { \overline { { { \mathcal { R } } ( A ) } } }$ , there exists $x \in \mathcal { X }$ such that $\| A x - f \| _ { \mathcal { V } } < \varepsilon / ( 2 \| A \| _ { \mathcal { L } ( \mathcal { X } , \mathcal { V } ) } )$ . Putting these together we have 

$$
\begin{array}{c} \| A ^ {*} A x - u \| _ {\mathcal {X}} \leqslant \| A ^ {*} A x - A ^ {*} f \| _ {\mathcal {X}} + \| A ^ {*} f - u \| _ {\mathcal {X}} \\ \leqslant \underbrace {\| A ^ {*} \| _ {\mathcal {L} (\mathcal {Y} , \mathcal {X})} \| A x - f \| _ {\mathcal {Y}}} _ {<   \varepsilon / 2} + \underbrace {\| A ^ {*} f - u \| _ {\mathcal {X}}} _ {<   \varepsilon / 2} <   \varepsilon \end{array}
$$

which shows that $u \in { \overline { { \mathcal { R } ( A ^ { * } A ) } } }$ and thus also ${ \overline { { \mathcal { R } ( A ^ { * } ) } } } \subseteq { \overline { { \mathcal { R } ( A ^ { * } A ) } } }$ 

## 2.1 Generalised Inverses

Recall the inverse problem 

$$
A u = f,\tag{2.1}
$$

where $A \colon \mathcal { X } \to \mathcal { Y }$ is a linear bounded operator and  and  are Hilbert spaces. 

Definition 2.1.1 (Minimal-norm solutions). An element $u \in \mathcal X$ is called 

• a least-squares solution of (2.1) if 

$$
\| A u - f \| _ {\mathcal {Y}} = \inf \{\| A v - f \| _ {\mathcal {Y}}, \quad v \in \mathcal {X} \};
$$

• a minimal-norm solution of (2.1) (and is denoted by u<sup>†</sup>) if 

$\| u ^ { \dag } \| _ { \mathcal { X } } \leqslant \| v \| _ { \mathcal { X } }$ for all least squares solutions v. 

Remark 2.1.2. Since $\mathcal { R } ( A )$ is not closed in general (it is never closed for a compact operator, unless the range is finite-dimensional), a least-squares solution may not exist. If it exists, then the minimal-norm solution is unique (it is the orthogonal projection of the zero element onto an afine subspace defined by $\| A u - f \| _ { \mathcal { V } } = \operatorname* { m i n } \{ \| A v - f \| _ { \mathcal { V } } , \quad v \in \mathcal { X } \} )$ 

In numerical linear algebra it is a well known fact that the normal equations can be used to compute least-squares solutions. The same holds true in the infinite-dimensional case. 

Theorem 2.1.3. Let $f \in \mathcal { V }$ and $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ . Then, the following three assertions are equivalent. 

1. $u \in \mathcal X$ satisfies $A u = P _ { \overline { { \mathcal { R } ( A ) } } } f .$ 

2. u is a least squares solution of the inverse problem (2.1). 

3. u solves the normal equation 

$$
A ^ {*} A u = A ^ {*} f.\tag{2.2}
$$

Remark 2.1.4. The name normal equation is derived from the fact that for any solution u its residual $A u - f$ is orthogonal (normal) to $\mathcal { R } ( A )$ . This can be readily seen, as we have for any $v \in \mathcal X$ that 

$$
0 = \langle v, A ^ {*} (A u - f) \rangle_ {\mathcal {X}} = \langle A v, A u - f \rangle_ {\mathcal {Y}}
$$

which shows Au $f \in \mathcal { R } ( A ) ^ { \perp }$ 

Proof of Theorem 2.1.3. For $1 \Rightarrow 2 \colon$ Let $u \in \mathcal X$ such that $A u = P _ { \overline { { \mathcal { R } ( A ) } } } f$ and let $v \in \mathcal { X }$ be arbitrary. With the basic properties of the orthogonal projection, Lemma 2.0.1 4, we have 

$$
\| A u - f \| _ {\mathcal {Y}} = \| f f - P _ {\overline {{\mathcal {R} (A)}}} f \| _ {\mathcal {Y}} \leqslant \inf _ {g \in \overline {{\mathcal {R} (A)}}} \| g - f \| _ {\mathcal {Y}} \leqslant \inf _ {g \in \mathcal {R} (A)} \| g - f \| _ {\mathcal {Y}} = \inf _ {v \in \mathcal {X}} \| A v - f \| _ {\mathcal {Y}},
$$

which shows that u is a least squares solution. 

For $2 \Rightarrow 3 \colon$ : Let $u \in \mathcal X$ be a least squares solution and let $v \in \mathcal { X }$ an arbitrary element. We define the quadratic polynomial $F \colon  { \mathbb { R } } \to  { \mathbb { R } }$ , 

$$
F (\lambda) := \| A (u + \lambda v) - f \| _ {\mathcal {Y}} ^ {2} = \lambda^ {2} \| A v \| _ {\mathcal {Y}} ^ {2} - 2 \lambda \left\langle A v, f - A u \right\rangle_ {\mathcal {Y}} + \| f - A u \| _ {\mathcal {Y}} ^ {2}.
$$

A necessary condition for $u \in \mathcal X$ to be a least squares solution is $F ^ { \prime } ( 0 ) = 0$ , which leads to $\langle v , A ^ { * } ( f - A u ) \rangle _ { \mathcal { X } } = 0$ . As v was arbitrary, it follows that the normal equation (2.2) must hold. 

For $3 \Rightarrow 1 \colon$ From the normal equation it follows that $A ^ { * } ( f - A u ) = 0$ , which is equivalent to $f - A u \in \mathcal { R } ( A ) ^ { \perp }$ , see Remark 2.1.4. Since $\mathcal { R } ( A ) ^ { \perp } = \left( \overline { { \mathcal { R } ( A ) } } \right) ^ { \perp }$ and $A u \in \mathcal { R } ( A ) \subset \overline { { \mathcal { R } ( A ) } }$ 2 the assertion follows from Lemma 2.0.1 5: 

$$
A u = P _ {\overline {{\mathcal {R} (A)}}} f \Leftrightarrow A u \in \overline {{\mathcal {R} (A)}} \text {   and   } f - A u \in \left(\overline {{\mathcal {R} (A)}}\right) ^ {\perp}.
$$

Lemma 2.1.5. Let $f \in \mathcal { V }$ and let <sup>L</sup> be the set of least squares solutions to the inverse problem (2.1). Then, <sup>L</sup> is non-empty if and only if $f \in \mathcal { R } ( A ) \oplus \mathcal { R } ( A ) ^ { \perp }$ 

Proof. Let $u \in \mathbb { L }$ . It is easy to see that $f = A u + ( f - A u ) \in { \mathcal { R } } ( A ) \oplus { \mathcal { R } } ( A ) ^ { \perp }$ as the normal equations are equivalent to $f - A u \in \mathcal { R } ( A ) ^ { \perp }$ 

Consider now $f \in \mathcal { R } ( A ) \oplus \mathcal { R } ( A ) ^ { \perp }$ . Then there exists $u \in \mathcal X$ and $g \in \mathcal { R } ( A ) ^ { \perp } = \left( \overline { { \mathcal { R } ( A ) } } \right) ^ { \perp }$ such that $\textstyle f = A u + g$ and thus $P _ { \overline { { \mathcal { R } ( A ) } } } f = P _ { \overline { { \mathcal { R } ( A ) } } } A u + P _ { \overline { { \mathcal { R } ( A ) } } } g = A u$ u and the assertion follows from Theorem 2.1.3 1. □ 

Remark 2.1.6. If the dimensions of and $\mathcal { R } ( A )$ are finite, then $\mathcal { R } ( A )$ is closed, i.e. ${ \overline { { \mathcal { R } ( A ) } } } = { \mathcal { R } } ( A )$ . Thus, in a finite dimensional setting, there always exists a least squares solution. 

Theorem 2.1.7. Let $f \in \mathcal { R } ( A ) \oplus \mathcal { R } ( A ) ^ { \perp }$ . Then there exists a unique minimal norm solution $u ^ { \dagger }$ to the inverse problem (2.1) and all least squares solutions are given by $\{ u ^ { \dag } \} + \mathcal { N } ( A )$ 

Proof. From Lemma 2.1.5 we know that there exists a least squares solution. As noted in Remark 2.1.2, in this case the minimal-norm solution is unique. Let $\varphi$ be an arbitrary least-squares solution. Using Theorem 2.1.3 we get 

$$
A (\varphi - u ^ {\dagger}) = A \varphi - A u ^ {\dagger} = P _ {\overline {{\mathcal {R} (A)}}} f - P _ {\overline {{\mathcal {R} (A)}}} f = 0,\tag{2.3}
$$

which shows that $\varphi - u ^ { \dagger } \in { \mathcal { N } } ( A )$ , hence the assertion. 

If a least-squares solution exists for a given $f \in \mathcal { V }$ then the minimal-norm solution can be computed (at least in theory) using the Moore-Penrose generalised inverse. 

Definition 2.1.8. Let $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ and let 

$$
\widetilde {A} := A | _ {\mathcal {N} (A) ^ {\perp}}: \mathcal {N} (A) ^ {\perp} \to \mathcal {R} (A)
$$

denote the restriction of A to ${ \mathcal { N } } ( A ) ^ { \perp }$ . The Moore-Penrose inverse $A ^ { \dagger }$ is defined as the unique linear extension of $\widetilde { A } ^ { - 1 }$ to 

$$
\mathcal {D} (A ^ {\dagger}) = \mathcal {R} (A) \oplus \mathcal {R} (A) ^ {\perp}
$$

with 

$$
\mathcal {N} (A ^ {\dagger}) = \mathcal {R} (A) ^ {\perp}.
$$

Remark 2.1.9. Due to the restriction to ${ \mathcal { N } } ( A ) ^ { \perp }$ and $\mathcal { R } ( A )$ we have that $\widetilde { A }$ is injective and surjective. Hence, $\widetilde { A } ^ { - 1 }$ exists and is linear and – as a consequence $- \ A ^ { \dagger }$ is well-defined on $\mathcal { R } ( A )$ 

Moreover, due to the orthogonal decomposition ${ \mathcal { D } } ( A ^ { \dagger } ) = { \mathcal { R } } ( A ) \oplus { \mathcal { R } } ( A ) ^ { \perp }$ <sup>⊥</sup>, there exist for arbitrary $f \in { \mathcal { D } } ( A ^ { \dagger } )$ elements $f _ { 1 } \in \mathcal { R } ( A )$ and $f _ { 2 } \in { \mathcal { R } } ( A ) ^ { - }$ <sup>⊥</sup> with $f = f _ { 1 } + f _ { 2 }$ . Therefore, we have 

$$
A ^ {\dagger} f = A ^ {\dagger} f _ {1} + A ^ {\dagger} f _ {2} = A ^ {\dagger} f _ {1} = \widetilde {A} ^ {- 1} f _ {1} = \widetilde {A} ^ {- 1} P _ {\overline {{\mathcal {R} (A)}}} f,\tag{2.4}
$$

where we used that $f _ { 2 } \in \mathcal { R } ( A ) ^ { \perp } = \mathcal { N } ( A ^ { \dagger } )$ . Thus, $A ^ { \dagger }$ is well-defined on the entire domain $\mathcal { D } ( A ^ { \dagger } )$ 

Remark 2.1.10. As orthogonal complements are always closed we get that 

$$
\overline {{\mathcal {D} (A ^ {\dagger})}} = \overline {{\mathcal {R} (A)}} \oplus \mathcal {R} (A) ^ {\perp} = \mathcal {Y},
$$

and hence, $\mathcal { D } ( A ^ { \dagger } )$ is dense in $\mathcal { V } .$ Thus, if $\mathcal { R } ( A )$ is closed it follows that $\mathcal { D } ( A ^ { \dagger } ) = \mathcal { D }$ and on the other hand, $\mathcal { D } ( A ^ { \dagger } ) = \mathcal { D }$ implies $\mathcal { R } ( A )$ is closed. We note that for ill-posed problems $\mathcal { R } ( A )$ is usually not closed; for instance, if A is compact then $\mathcal { R } ( A )$ is closed if and only if it is finite-dimensional [1, Ex.1 Section 7.1]. 

If A is bijective we have that $A ^ { \dagger } = A ^ { - 1 }$ . We also highlight that the extension $A ^ { \dagger }$ is not necessarily continuous. 

Theorem 2.1.11 ([20, Prop. 2.4]). Let $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ . Then $A ^ { \dagger }$ is continuous, i.e. $A ^ { \dag } \in$ $\mathcal { L } ( \mathcal { D } ( A ^ { \dagger } ) , \mathcal { X } )$ , if and only if $\mathcal { R } ( A )$ is closed. 

Example 2.1.12. To illustrate the definition of the Moore-Penrose inverse we consider a simple example in finite dimensions. Let the linear operator $A : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 2 }$ be given by 

$$
A x = \left( \begin{array}{c c c} 2 & 0 & 0 \\ 0 & 0 & 0 \end{array} \right) \left( \begin{array}{c} x _ {1} \\ x _ {2} \\ x _ {3} \end{array} \right) = \binom{2 x _ {1}}{0}.
$$

It is easy to see that ${ \mathcal { R } } ( A ) = \{ f \in \mathbb { R } ^ { 2 } \ | \ f _ { 2 } = 0 \}$ and $\mathcal { N } ( A ) = \{ x \in \mathbb { R } ^ { 3 } \mid x _ { 1 } = 0 \}$ . Thus, $\mathcal { N } ( A ) ^ { \perp } = \{ x \in \mathbb { R } ^ { 3 } \mid x _ { 2 } , x _ { 3 } = 0 \}$ . Therefore, $\widetilde { A } \colon { \mathcal { N } } ( A ) ^ { \perp } \to { \mathcal { R } } ( A )$ , given by $x \mapsto ( 2 x _ { 1 } , 0 ) ^ { \top }$ , is bijective and its inverse $\widetilde A ^ { - 1 } \colon { \mathcal { R } } ( A ) \to { \mathcal { N } } ( A ) ^ { \perp }$ is given by $f \mapsto ( f _ { 1 } / 2 , 0 , 0 ) ^ { \top }$ 

To get the Moore-Penrose inverse $A ^ { \dagger }$ , we need to extend $\widetilde { A } ^ { - 1 }$ to $\mathcal { R } ( A ) \oplus \mathcal { R } ( A ) ^ { \perp }$ in such a way that $A ^ { \dagger } f = 0$ for all $f \in \mathcal { R } ( A ) ^ { \perp } = \{ f \in \mathbb { R } ^ { 2 } \mid f _ { 1 } = 0 \}$ . It is easy to see that the Moore-Penrose inverse $A ^ { \dagger } \colon  { \mathbb { R } } ^ { 2 } \to  { \mathbb { R } } ^ { 3 }$ is given by the following expression 

$$
A ^ {\dagger} f = \left( \begin{array}{c c} 1 / 2 & 0 \\ 0 & 0 \\ 0 & 0 \end{array} \right) \binom{f _ {1}}{f _ {2}} = \binom{f _ {1} / 2}{0}
$$

Let us consider data $\widetilde { f } = ( 8 , 1 ) ^ { \top } \notin \mathcal { R } ( A )$ . Then, $A ^ { \dagger } \widetilde { f } = A ^ { \dagger } ( 8 , 1 ) ^ { \top } = ( 4 , 0 , 0 ) ^ { \top }$ 

It can be shown that $A ^ { \dagger }$ can be characterised by the Moore-Penrose equations. 

Theorem 2.1.13 ([20, Prop. 2.3]). The Moore-Penrose inverse $A ^ { \dagger }$ satisfies $\mathcal { R } ( A ^ { \dagger } ) =$ ${ \mathcal { N } } ( A ) ^ { \perp }$ and the Moore-Penrose equations 

1. $A ^ { \dagger } A = P _ { \mathcal { N } ( A ) ^ { \bot } }$ 

2. $A A ^ { \dagger } = \left. P _ { { \overline { { \mathcal { R } ( A ) } } } } \right| _ { { \mathcal { D } } ( A ^ { \dagger } ) } ,$ 

3. $A A ^ { \dagger } A = A ,$ 

4. $A ^ { \dagger } A A ^ { \dagger } = A ^ { \dagger } ,$ 

where $P _ { \mathcal { N } ( A ) }$ and $P _ { \overline { { \mathcal { R } ( A ) } } }$ denote the orthogonal projections on ${ \mathcal { N } } ( A )$ and $\overline { { \mathcal { R } ( A ) } }$ , respectively. Proof. First, by the definition of the Moore-Penrose inverse we have for any $u \in \mathcal X$ 

$$
A ^ {\dagger} A u = A ^ {\dagger} A (P _ {\mathcal {N} (A)} u + P _ {\mathcal {N} (A) ^ {\perp}} u) = A ^ {\dagger} A P _ {\mathcal {N} (A) ^ {\perp}} u = \widetilde {A} ^ {- 1} A P _ {\mathcal {N} (A) ^ {\perp}} u = P _ {\mathcal {N} (A) ^ {\perp}} u,
$$

which proves 1. Now, for any $f \in { \mathcal { D } } ( A ^ { \dagger } )$ we have (see (2.4)) 

$$
A A ^ {\dagger} f = A \widetilde {A} ^ {- 1} P _ {\overline {{\mathcal {R} (A)}}} f = P _ {\overline {{\mathcal {R} (A)}}} f,
$$

which proves 2. Applying A to 1., we get 3., and applying $A ^ { \dagger }$ to 2., we get 4., which completes the proof. □ 

Corollary 2.1.14. The Moore-Penrose inverse is uniquely characterised by 1.–2., that is, if a linear operator $B \colon { \mathcal { R } } ( A ) \oplus { \mathcal { R } } ( A ) ^ { \perp } \to { \mathcal { N } } ( A )$ satisfies $B A = P _ { \mathcal { N } ( A ) } .$ ⊥ and $A B = P _ { \overline { { { \mathcal { R } } ( A ) } } }$ then $B = A ^ { \dagger }$ 

Proof. First we show that $B | _ { \mathcal { R } ( A ) } = \widetilde { A } ^ { - 1 }$ . Indeed, let $f = A u \in { \mathcal { R } } ( A )$ , where $u \in \mathcal { N } ( A ) ^ { \perp }$ Then 

$$
B f = B A u = P _ {\mathcal {N} (A) ^ {\perp}} u = u = \widetilde {A} ^ {- 1} f,
$$

where the last equality holds since $\widetilde { A }$ is bijective and hence uniquely invertible. 

Now we prove that $B | _ { \mathcal { R } ( A ) ^ { \perp } } = 0$ . Indeed, for any $f \in \mathcal { R } ( A ) ^ { \perp }$ we have 

$$
A B f = P _ {\overline {{\mathcal {R} (A)}}} f = 0.
$$

Therefore, B is an extension of $\widetilde { A } ^ { - 1 }$ to $\mathcal { R } ( A ) \oplus \mathcal { R } ( A ) ^ { \perp }$ with $\mathcal { N } ( B ) = \mathcal { R } ( A ) ^ { \perp }$ . Since such an extension is unique, $B = A ^ { \dagger }$ □ 

Remark 2.1.15. If an operator B satisfies only $A B A = A { \mathrm { ~ ( r e s p . ~ } } B A B = B )$ , it is called the inner inverse (resp. outer inverse) of A. 

The next theorem shows that minimal-norm solutions can indeed be computed using the Moore-Penrose generalised inverse. 

Theorem 2.1.16. For each $f \in { \mathcal { D } } ( A ^ { \dagger } )$ , the minimal norm solution $u ^ { \dagger }$ to the inverse problem (2.1) is given via 

$$
u ^ {\dagger} = A ^ {\dagger} f.
$$

Proof. As $f \in { \mathcal { D } } ( A ^ { \dagger } )$ , we know from Theorem 2.1.7 that the minimal norm solution $u ^ { \dagger }$ exists and is unique. With $u ^ { \dag } \in \mathcal { N } ( A ) ^ { \perp }$ , Lemma 2.1.13, and Theorem 2.1.3 we conclude that 

$$
u ^ {\dagger} = (I - P _ {\mathcal {N} (A)}) u ^ {\dagger} = A ^ {\dagger} A u ^ {\dagger} = A ^ {\dagger} P _ {\overline {{\mathcal {R} (A)}}} f = A ^ {\dagger} A A ^ {\dagger} f = A ^ {\dagger} f.
$$

As a consequence of Theorem 2.1.16 and Theorem 2.1.3, we find that the minimum norm solution $u ^ { \dagger }$ of $A u = f$ is a minimum norm solution of the normal equation (2.2), i.e. 

$$
u ^ {\dagger} = (A ^ {*} A) ^ {\dagger} A ^ {*} f.
$$

Thus, in order to compute $u ^ { \dagger }$ we can equivalently consider finding the minimum norm solution of the normal equation. 

## 2.2 Compact Operators

Definition 2.2.1. Let $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ . Then A is said to be compact if for any bounded set $B \subset { \mathcal { X } }$ the closure $o f$ its image $\overline { { A ( B ) } }$ is compact in . We denote the space $o f$ compact operators by $\kappa ( \boldsymbol { \mathcal { X } } , \boldsymbol { \mathcal { V } } )$ 

Remark 2.2.2. We can equivalently define an operator A to be compact if the image of a bounded sequence $\{ u _ { j } \} _ { j \in \mathbb { N } } \subset \mathcal { X }$ contains a convergent subsequence $\{ A u _ { j _ { k } } \} _ { k \in \mathbb { N } } \subset \mathcal { V }$ 

Compact operators are very common in inverse problems. In fact, almost all (linear) inverse problems involve the inversion of a compact operator. As the following result shows, compactness of the forward operator is a major source if ill-posedness. 

Theorem 2.2.3. Let $A \in \mathcal { K } ( \mathcal { X } , \mathcal { Y } )$ with an infinite dimensional range. Then, the Moore-Penrose inverse of A is discontinuous. 

Proof. As the range $\mathcal { R } ( A )$ is of infinite dimension, we can conclude that and ${ \mathcal { N } } ( A ) ^ { \perp }$ are also infinite dimensional. We can therefore find a sequence $\{ u _ { j } \} _ { j \in \mathbb { N } }$ with $u _ { j } \in \mathcal { N } ( A ) ^ { \perp }$ , $\| u _ { j } \| _ { \mathcal { X } } = 1$ and $\langle u _ { j } , u _ { k } \rangle _ { \mathcal { X } } = 0$ for $j \neq k$ . Since A is a compact operator the sequence $f _ { j } = A u _ { j }$ has a convergent subsequence, hence, for all $\delta > 0$ we can find $j , k$ such that $\| \ b { f } _ { j } - \ b { f } _ { k } \| _ { \mathcal { V } } < \delta$ . However, we also obtain 

$$
\begin{array}{r l}&{\| A ^ {\dagger} f _ {j} - A ^ {\dagger} f _ {k} \| _ {\mathcal {X}} ^ {2} = \| A ^ {\dagger} A u _ {j} - A ^ {\dagger} A u _ {k} \| _ {\mathcal {X}} ^ {2}}\\&{\qquad = \| u _ {j} - u _ {k} \| _ {\mathcal {X}} ^ {2} = \| u _ {j} \| _ {\mathcal {X}} ^ {2} - 2 \left<   u _ {j}, u _ {k} \right> _ {\mathcal {X}} + \| u _ {k} \| _ {\mathcal {X}} ^ {2} = 2,}\end{array}
$$

which shows that $A ^ { \dagger }$ is discontinuous. Here, the second identity follows from Lemma 2.1.13 1 and the fact that $u _ { j } , u _ { k } \in \mathcal { N } ( A ) ^ { \perp }$ □ 

To have a better understanding of when we have $f \in \overline { { \mathcal { R } ( A ) } } \backslash \mathcal { R } ( A )$ for compact operators A, we want to consider the singular value decomposition of compact operators. 

## Singular value decomposition of compact operators

Theorem 2.2.4 ([23, p. 225, Theorem 9.16]). Let be a Hilbert space and $A \in \mathcal { K } ( \mathcal { X } , \mathcal { X } )$ be self-adjoint. Then there exists an orthonormal basis $\{ x _ { j } \} _ { j \in \mathbb { N } } \subset \mathcal { X }$ of $\overline { { \mathcal { R } ( A ) } }$ and a sequence of eigenvalues $\{ \lambda _ { j } \} _ { j \in \mathbb { N } } \subset \mathbb { R }$ with $| \lambda _ { 1 } | \geqslant | \lambda _ { 2 } | \geqslant . . . > 0$ such that for all $u \in \mathcal X$ we have 

$$
A u = \sum_ {j = 1} ^ {\infty} \lambda_ {j} \left\langle u, x _ {j} \right\rangle_ {\mathcal {X}} x _ {j}.
$$

The sequence $\{ \lambda _ { j } \} _ { j \in \mathbb { N } }$ is either finite or we have $\lambda _ { j } \to 0$ 

Remark 2.2.5. The notation in the theorem above only makes sense if the sequence $\{ \lambda _ { j } \} _ { j \in \mathbb { N } }$ is infinite. For the case that there are only finitely many $\lambda _ { j }$ the sum has to be interpreted as a finite sum. 

Moreover, as the eigenvalues are sorted by absolute value $| \lambda _ { j } |$ , we have $\| A \| _ { \mathcal { L } ( \mathcal { X } , \mathcal { X } ) } = | \lambda _ { 1 } |$ 

If A is not self-adjoint, the decomposition in Theorem 2.2.4 does not hold any more. Instead, we can consider the so-called singular value decomposition of a compact linear operator. 

## Theorem 2.2.6. Let $A \in \mathcal { K } ( \mathcal { X } , \mathcal { Y } )$ . Then there exists

1. a not-necessarily infinite null sequence $\{ \sigma _ { \it j } \} _ { \it j \in \mathbb { N } }$ with σ<sub>1</sub> $\geqslant \sigma _ { 2 } \geqslant . . . > 0 .$ 

2. an orthonormal basis $\{ x _ { j } \} _ { j \in \mathbb { N } } \subset \mathcal { X }$ of ${ \mathcal { N } } ( A ) ^ { \perp }$ 

3. an orthonormal basis $\{ y _ { j } \} _ { j \in \mathbb { N } } \subset \mathcal { V } \ o f \overline { { \mathscr { R } ( A ) } }$ with 

$$
A x _ {j} = \sigma_ {j} y _ {j}, \quad A ^ {*} y _ {j} = \sigma_ {j} x _ {j}, \quad f o r a l l j \in \mathbb {N}.\tag{2.5}
$$

Moreover, for all $u \in \mathcal X$ we have the representation 

$$
A u = \sum_ {j = 1} ^ {\infty} \sigma_ {j} \left\langle u, x _ {j} \right\rangle y _ {j}.\tag{2.6}
$$

The sequence $\{ ( \sigma _ { j } , x _ { j } , y _ { j } ) \}$ is called singular system $o r$ singular value decomposition (SVD) of A. 

For the adjoint operator $A ^ { * }$ we have the representation 

$$
A ^ {*} f = \sum_ {j = 1} ^ {\infty} \sigma_ {j} \left\langle f, y _ {j} \right\rangle x _ {j} \quad \forall f \in \mathcal {Y}.\tag{2.7}
$$

Proof. Consider $B = A ^ { * } A$ and $C = A A ^ { * }$ . Both B and $C$ are compact, self-adjoint and positive semidefinite, so that by Theorem 2.2.4 both admit a spectral representation and, by positive semidefiniteness, their eigenvalues are positive. Therefore, we can write 

$$
C f = \sum_ {j = 1} ^ {\infty} \sigma_ {j} ^ {2} \left\langle f, y _ {j} \right\rangle y _ {j} \quad \forall f \in \mathcal {Y},
$$

where $\{ y _ { j } \}$ is an orthonormal basis of $\overline { { \mathcal { R } ( A A ^ { * } ) } } = \overline { { \mathcal { R } ( A ) } }$ (Lemma 2.0.3), $\sigma _ { j } > 0$ for all $j$ and $\sigma _ { j }  0$ as $j \to \infty$ 

Now consider the element $A ^ { * } y _ { j } \in { \mathcal { X } }$ . Since $\sigma _ { j } ^ { 2 }$ is an eigenvalue of C for the eigenvector $y _ { j }$ , we get that 

$$
\sigma_ {j} ^ {2} A ^ {*} y _ {j} = A ^ {*} (\sigma_ {j} ^ {2} y _ {j}) = A ^ {*} C y _ {j} = A ^ {*} A A ^ {*} y _ {j} = B A ^ {*} y _ {j}
$$

and therefore $\sigma _ { j } ^ { 2 }$ is also an eigenvalue of $B$ (for the eigenvector $A ^ { * } y _ { j } )$ . Now we will show that the system $\left\{ { \frac { A ^ { * } y _ { j } } { \sigma _ { j } } } \right\} _ { j \in \mathbb { N } }$ forms an orthonormal basis of $\overline { { \mathcal { R } ( A ^ { * } ) } } = \mathcal { N } ( A ) ^ { \perp }$ . Indeed, we have 

$$
\left\langle \frac {A ^ {*} y _ {j}}{\sigma_ {j}}, \frac {A ^ {*} y _ {k}}{\sigma_ {k}} \right\rangle = \frac {1}{\sigma_ {j} \sigma_ {k}} \left\langle y _ {j}, A A ^ {*} y _ {k} \right\rangle = \frac {1}{\sigma_ {j} \sigma_ {k}} \left\langle y _ {j}, \sigma_ {k} ^ {2} y _ {k} \right\rangle = \left\{ \begin{array}{l l} 1, & \text {if j = k ,} \\ 0, & \text {otherwise.} \end{array} \right.
$$

Hence, $\left\{ { \frac { A ^ { * } y _ { j } } { \sigma _ { j } } } \right\} _ { j \in \mathbb { N } }$ are orthonormal. It is also clear that they are dense in $\overline { { \mathcal { R } ( A ^ { * } ) } } = \mathcal { N } ( A ) ^ { \perp }$ hence they form a basis. Therefore, we can choose $\{ x _ { j } \} _ { j \in \mathbb { N } } = \left\{ \frac { A ^ { * } y _ { j } } { \sigma _ { j } } \right\} _ { j \in \mathbb { N } } ,$ i.e. 

$$
x _ {j} = \sigma_ {j} ^ {- 1} A ^ {*} y _ {j}
$$

and we get (by construction) that 

$$
A ^ {*} y _ {j} = \sigma_ {j} x _ {j}.
$$

We also observe that 

$$
A x _ {j} = \sigma_ {j} ^ {- 1} A A ^ {*} y _ {j} = \sigma_ {j} ^ {- 1} \sigma_ {j} ^ {2} y _ {j} = \sigma_ {j} y _ {j},
$$

which proves (2.5). 

Extending the basis $\{ x _ { j } \}$ of $\overline { { \mathcal { R } ( A ^ { * } ) } }$ to a basis $\{ \widetilde { x } _ { j } \}$ of , we expand an arbitrary $u \in \mathcal X$ as $\begin{array} { r } { u = \sum _ { j = 1 } ^ { \infty } \langle u , \widetilde { x } _ { j } \rangle \widetilde { x } _ { j } } \end{array}$ . Applying A and using the fact that $\mathcal { X } = \mathcal { N } ( A ) \oplus \overline { { \mathcal { R } ( A ^ { * } ) } }$ (Remark 2.0.2), we obtain the singular value decomposition (2.6) (and also (2.7) in a similar manner) 

$$
A u = \sum_ {j = 1} ^ {\infty} \sigma_ {j} \left\langle u, x _ {j} \right\rangle y _ {j} \quad \forall u \in \mathcal {X}, \quad A ^ {*} f = \sum_ {j = 1} ^ {\infty} \sigma_ {j} \left\langle f, y _ {j} \right\rangle x _ {j} \quad \forall f \in \mathcal {Y}.
$$

We can now derive a representation of the Moore-Penrose inverse in terms of the singular value decomposition. 

Theorem 2.2.7. Let $A \in \mathcal { K } ( \mathcal { X } , \mathcal { Y } )$ with singular system $\{ ( \sigma _ { j } , x _ { j } , y _ { j } ) \} _ { j \in \mathbb { N } }$ and $f \in { \mathcal { D } } ( A ^ { \dagger } )$ . Then the Moore-Penrose inverse of A can be written as 

$$
A ^ {\dagger} f = \sum_ {j = 1} ^ {\infty} \sigma_ {j} ^ {- 1} \left\langle f, y _ {j} \right\rangle x _ {j}.\tag{2.8}
$$

Proof. We know that, since $f \in \mathcal { D } ( A ^ { \dagger } ) , u ^ { \dagger } = A ^ { \dagger } f$ solves the normal equations 

$$
A ^ {*} A u ^ {\dagger} = A ^ {*} f.
$$

From Theorem 2.2.6 we know that 

$$
A ^ {*} A u ^ {\dagger} = \sum_ {j = 1} ^ {\infty} \sigma_ {j} ^ {2} \left\langle u ^ {\dagger}, x _ {j} \right\rangle x _ {j}, \quad A ^ {*} f = \sum_ {j = 1} ^ {\infty} \sigma_ {j} \left\langle f, y _ {j} \right\rangle x _ {j},\tag{2.9}
$$

which implies that 

$$
\left\langle u ^ {\dagger}, x _ {j} \right\rangle = \sigma_ {j} ^ {- 1} \left\langle f, y _ {j} \right\rangle
$$

Expanding $u ^ { \dag } \in \mathcal { N } ( A ) ^ { \perp }$ in the basis $\{ x _ { j } \}$ , we get 

$$
u ^ {\dagger} = \sum_ {j = 1} ^ {\infty} \left\langle u ^ {\dagger}, x _ {j} \right\rangle x _ {j} = \sum_ {j = 1} ^ {\infty} \sigma_ {j} ^ {- 1} \left\langle f, y _ {j} \right\rangle x _ {j} = A ^ {\dagger} f.
$$

The representation (2.8) makes it clear again that the Moore-Penrose inverse is unbounded if $\mathcal { R } ( A )$ is infinite dimensional. Indeed, taking the sequence $y _ { j }$ we note that $\| A ^ { \dagger } y _ { j } \| = \sigma _ { i } ^ { - 1 } \to \infty$ , although $\| y _ { j } \| = 1$ 

The unboundedness of the Moore-Penrose inverse is also reflected in the fact that the series in (2.8) may not converge for a given $f .$ . The convergence criterion for the series is called the Picard criterion. 

Definition 2.2.8. We say that the data f satisfy the Picard criterion, $i f$ 

$$
\| A ^ {\dagger} f \| ^ {2} = \sum_ {j = 1} ^ {\infty} \frac {| \langle f , y _ {j} \rangle | ^ {2}}{\sigma_ {j} ^ {2}} <   \infty .\tag{2.10}
$$

Remark 2.2.9. The Picard criterion is a condition on the decay of the coeficients $\langle f , y _ { j } \rangle$ As the singular values $\sigma _ { j }$ decay to zero as $j \to \infty$ , the Picard criterion is only met if the coeficients $\langle f , y _ { j } \rangle$ decay suficiently fast. 

In case the singular system is given by the Fourier basis, then the coeficients $\langle f , y _ { j } \rangle$ are just the Fourier coeficients of $f .$ . Therefore, the Picard criterion is a condition on the decay of the Fourier coeficients which is equivalent to the smoothness of $f .$ 

It turns our that the Picard criterion also can be used to characterise elements in the range of the forward operator. 

Theorem 2.2.10. Let $A \in \mathcal { K } ( \mathcal { X } , \mathcal { Y } )$ with singular system $\{ ( \sigma _ { j } , x _ { j } , y _ { j } ) \} _ { j \in \mathbb { N } }$ , and $f \in { \overline { { \mathcal { R } ( A ) } } }$ Then $f \in { \mathcal { R } } ( A )$ if and only if the Picard criterion 

$$
\sum_ {j = 1} ^ {\infty} \frac {\left| \langle f , y _ {j} \rangle_ {\mathcal {Y}} \right| ^ {2}}{\sigma_ {j} ^ {2}} <   \infty\tag{2.11}
$$

is met. 

Proof. Let $f \in { \mathcal { R } } ( A )$ , thus there is a $u \in \mathcal X$ such that $A u = f$ . It is easy to see that we have 

$$
\langle f, y _ {j} \rangle_ {\mathcal {Y}} = \langle A u, y _ {j} \rangle_ {\mathcal {Y}} = \langle u, A ^ {*} y _ {j} \rangle_ {\mathcal {X}} = \sigma_ {j} \left\langle u, x _ {j} \right\rangle_ {\mathcal {X}}
$$

and therefore 

$$
\sum_ {j = 1} ^ {\infty} \sigma_ {j} ^ {- 2} | \langle f, y _ {j} \rangle_ {\mathcal {Y}} | ^ {2} = \sum_ {j = 1} ^ {\infty} | \langle u, x _ {j} \rangle_ {\mathcal {X}} | ^ {2} \leqslant \| u \| _ {\mathcal {X}} ^ {2} <   \infty .
$$

Now let the Picard criterion (2.11) hold and define $\begin{array} { r } { u : = \sum _ { j = 1 } ^ { \infty } \sigma _ { j } ^ { - 1 } \left. f , y _ { j } \right. _ { \mathcal { Y } } x _ { j } \in \mathcal { X } } \end{array}$ . It is well-defined by the Picard criterion (2.11) and we conclude 

$$
A u = \sum_ {j = 1} ^ {\infty} \sigma_ {j} ^ {- 1} \left\langle f, y _ {j} \right\rangle_ {\mathcal {Y}} A x _ {j} = \sum_ {j = 1} ^ {\infty} \left\langle f, y _ {j} \right\rangle_ {\mathcal {Y}} y _ {j} = P _ {\overline {{\mathcal {R} (A)}}} f = f,
$$

which shows $f \in { \mathcal { R } } ( A )$ 

Although all ill-posed problems are not easy to solve, some are worse than others, depending on how fast the singular values decay to zero. 

Definition 2.2.11. We say that an ill-posed inverse problem (2.1) is mildly ill-posed if the singular values decay at most with polynomial speed, i.e. there exist $\gamma , C > 0$ such that $\sigma _ { j } \geqslant C j ^ { - \gamma }$ for all j. We call the ill-posed inverse problem severely ill-posed $i f$ its singular values decay faster than with polynomial speed, i.e. for all $\gamma , C > 0$ one has that $\sigma _ { j } \leqslant C j ^ { - \gamma }$ for j suficiently large. 

Example 2.2.12. Let us consider the example of diferentiation again, as introduced in Section 1.2.3. The forward operator A: $L ^ { 2 } ( [ 0 , 1 ] ) \to L ^ { 2 } ( [ 0 , 1 ] )$ in this problem is given by 

$$
(A u) (t) = \int_ {0} ^ {t} u (s) d s = \int_ {0} ^ {1} K (s, t) u (s) d s,
$$

with $K \colon [ 0 , 1 ] \times [ 0 , 1 ] \to \mathbb { R }$ defined as 

$$
K (s, t) := \left\{ \begin{array}{l l} 1 & s \leqslant t \\ 0 & \text { else } \end{array} \right..
$$

This is a special case of the integral operators as introduced in Section 1.2.1. Since the kernel K is square integrable, A is compact. 

The adjoint operator $A ^ { * }$ is given via 

$$
(A ^ {*} f) (s) = \int_ {0} ^ {1} K (t, s) f (t) d t = \int_ {s} ^ {1} v (t) d t.\tag{2.12}
$$

Now we want to compute the eigenvalues and eigenvectors of $A ^ { * } A$ , i.e. we look for $\sigma ^ { 2 }$ and $x \in L ^ { 2 } ( [ 0 , 1 ] )$ ) with 

$$
\sigma^ {2} x (s) = (A ^ {*} A x) (s) = \int_ {s} ^ {1} \int_ {0} ^ {t} x (r) d r d t.
$$

We immediately observe $x ( 1 ) = 0$ and further 

$$
\sigma^ {2} x ^ {\prime} (s) = \frac {d}{d s} \int_ {s} ^ {1} \int_ {0} ^ {t} x (r) d r d t = - \int_ {0} ^ {s} x (r) d r,
$$

from which we conclude $x ^ { \prime } ( 0 ) = 0$ . Taking the derivative another time thus yields the ordinary diferential equation 

$$
\sigma^ {2} x ^ {\prime \prime} (s) + x (s) = 0,
$$

for which solutions are of the form 

$$
x (s) = c _ {1} \sin (\sigma^ {- 1} s) + c _ {2} \cos (\sigma^ {- 1} s),
$$

with some constants $c _ { 1 } , c _ { 2 }$ . In order to satisfy the boundary conditions $x ( 1 ) = c _ { 1 } \sin ( \sigma ^ { - 1 } ) +$ c<sub>2</sub> cos $( \sigma ^ { - 1 } ) = 0$ and $x ^ { \prime } ( 0 ) = c _ { 1 } = 0$ , we chose $c _ { 1 } = 0$ and σ such that $\cos ( \sigma ^ { - 1 } ) = 0$ . Hence, we have 

$$
\sigma_ {j} = \frac {2}{(2 j - 1) \pi} \mathrm{for} j \in \mathbb {N},
$$

and by choosing $c _ { 2 } = \sqrt { 2 }$ we obtain the following normalised representation of $x _ { j }$ : 

$$
x _ {j} (s) = \sqrt {2} \cos \left(\left(j - \frac {1}{2}\right) \pi s\right).
$$

According to (2.5) we further obtain 

$$
y _ {j} (s) = \sigma_ {j} ^ {- 1} (A x _ {j}) (s) = \left(j - \frac {1}{2}\right) \pi \int_ {0} ^ {s} \sqrt {2} \cos \left(\left(j - \frac {1}{2}\right) \pi t\right) d t = \sqrt {2} \sin \left(\left(j - \frac {1}{2}\right) \pi s\right),
$$

and hence, for $f \in L ^ { 2 } ( [ 0 , 1 ] )$ the Picard criterion becomes 

$$
2 \sum_ {j = 1} ^ {\infty} \sigma_ {j} ^ {- 2} \left(\int_ {0} ^ {1} f (s) \sin \left(\sigma_ {j} ^ {- 1} s\right) d s\right) ^ {2} <   \infty .
$$

Expanding f in the basis $\{ y _ { j } \}$ 

$$
f (t) = 2 \sum_ {j = 1} ^ {\infty} \left(\int_ {0} ^ {1} f (s) \sin \left(\sigma_ {j} ^ {- 1} s\right) d s\right) \sin \left(\sigma_ {j} ^ {- 1} t\right)
$$

and formally diferentiating the series, we obtain 

$$
f ^ {\prime} (t) = 2 \sum_ {j = 1} ^ {\infty} \sigma_ {j} ^ {- 1} \left(\int_ {0} ^ {1} f (s) \sin \left(\sigma_ {j} ^ {- 1} s\right) d s\right) \cos \left(\sigma_ {j} ^ {- 1} t\right).
$$

Therefore, the Picard criterion is nothing but the condition for the legitimacy of such diferentiation, i.e. for the diferentiability of the Fourier series by diferentiating its components, and it holds if $f$ is diferentiable and $f ^ { \prime } \in L ^ { 2 } ( [ 0 , 1 ] )$ . 

From the decay of the singular values we see that this inverse problem is mildly ill-posed. 

Example 2.2.13 (Heat equation). Consider the problem of recovering the initial condition u of the heat equation from an observation f of the solution at some time $T > 0$ (see Section 1.2.2). We consider the heat equation on $( 0 , \pi ) \times \mathbb { R } _ { + }$ , with Dirichlet boundary conditions 

$$
\left\{ \begin{array}{l l} v _ {t} - v _ {x x} = 0 & \text { on } (0, \pi) \times \mathbb {R} _ {+}, \\ v (0, t) = v (\pi , t) = 0 & \text { on } \mathbb {R} _ {+}, \\ v (x, T) = f (x) & \text { on } (0, \pi), \\ v (x, 0) = u (x) & \text { on } (0, \pi). \end{array} \right.
$$

The solution to the forward problem (determine f given u) is given by 

$$
f = A u := \sum_ {j = 1} ^ {\infty} e ^ {- j ^ {2} T} \widehat {u} _ {j} \sin (j x),
$$

where $\widehat { u } _ { j } = \langle u , \sin ( j \cdot ) \rangle$ are Fourier coeficients of u. Hence, singular values of A are given by 

$$
\sigma_ {j} = e ^ {- j ^ {2} T}, \quad j \in \mathbb {N},
$$

and 

$$
\frac {1}{\sigma_ {j}} = e ^ {j ^ {2} T}.
$$

Singular values decay exponentially and the inverse problem is severely (exponentially) ill-posed. 

## Chapter 3

# Classical Regularisation Theory

## 3.1 What is Regularisation?

We have seen that the Moore-Penrose inverse $A ^ { \dagger }$ is unbounded if $\mathcal { R } ( A )$ is not closed. Therefore, given noisy data $f _ { \delta }$ such that $\| f _ { \delta } - f \| \leqslant \delta ,$ , we cannot expect convergence $A ^ { \dagger } f _ { \delta }  A ^ { \dagger } f$ as $\delta  0$ . To achieve convergence, we replace A<sup>†</sup> with a family of well-posed (bounded) operators $R _ { \alpha }$ with $\alpha = \alpha ( \delta , f _ { \delta } )$ and require that $R _ { \alpha ( \delta , f _ { \delta } ) } ( f _ { \delta } )  A ^ { \dagger } f$ for all $f \in { \mathcal { D } } ( A ^ { \dagger } )$ and all $f _ { \delta } \in \mathcal { V } \mathrm { ~ s . t . ~ } \| f - f _ { \delta } \| _ { \mathcal { V } } \leqslant \delta$ as $\delta \to 0$ 

We remind ourselves that $\mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ denotes the space of all bounded (equivalently, continuous) operators $\mathcal { X }  \mathcal { V }$ 

Definition 3.1.1. Let $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ be a bounded operator. A family $\{ R _ { \alpha } \} _ { \alpha > 0 }$ of continuous operators is called regularisation (or regularisation operator) of $A ^ { \dagger }$ if 

$$
R _ {\alpha} f \rightarrow A ^ {\dagger} f = u ^ {\dagger}
$$

for all $f \in { \mathcal { D } } ( A ^ { \dagger } )$ as $\alpha  0$ 

Definition 3.1.2. If the family $\{ R _ { \alpha } \} _ { \alpha > 0 }$ consists of linear operators, then one speaks of linear regularisation of $A ^ { \dagger }$ 

Hence, a regularisation is a pointwise approximation of the Moore–Penrose inverse with continuous operators. As in the interesting cases the Moore–Penrose inverse may not be continuous we cannot expect that the norm of $R _ { \alpha }$ stays bounded as $\alpha  0$ . This is confirmed by the following results (in the linear case). 

Theorem 3.1.3 (Banach–Steinhaus e.g. [12, p. 78], [38, p. 173]). Let $x , y$ be Hilbert spaces and $\{ A _ { j } \} _ { j \in \mathbb { N } } \subset \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ a family of point-wise bounded operators, i.e. for all $u \in \mathcal X$ there exists a constant $C ( u ) > 0 \ s . t .$ s $\begin{array} { r } { \operatorname { l p } _ { j \in \mathbb { N } } \| A _ { j } u \| _ { \mathcal { Y } } \leqslant C ( u ) } \end{array}$ . Then 

$$
\sup _ {j \in \mathbb {N}} \| A _ {j} \| _ {\mathcal {L} (\mathcal {X}, \mathcal {Y})} <   \infty .
$$

Corollary 3.1.4 $( [ 3 8 , \mathrm { p } . \ 1 7 4 ] )$ . Let $x , y$ be Hilbert spaces and $\{ A _ { j } \} _ { j \in \mathbb { N } } \subset \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ . Then the following two conditions are equivalent: 

1. There exists $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ such that 

$$
A u = \lim _ {j \to \infty} A _ {j} u \quad \text { for   all } u \in \mathcal {X}.
$$

2. There is a dense subset $\mathcal { X } ^ { \prime } \subset \mathcal { X }$ such that $\operatorname* { l i m } _ { j \to \infty } A _ { j } u$ exists for all $u \in \mathcal { X } ^ { \prime }$ and 

$$
\sup _ {j \in \mathbb {N}} \| A _ {j} \| _ {\mathcal {L} (\mathcal {X}, \mathcal {Y})} <   \infty .
$$

Theorem 3.1.5. Let , be Hilbert spaces, $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ and $\{ R _ { \alpha } \} _ { \alpha > 0 }$ a linear regularisation as defined in Definition 3.1.2. $I f A ^ { \dagger }$ is not continuous, $\{ R _ { \alpha } \} _ { \alpha > 0 }$ cannot be uniformly bounded. In particular, there exist $f \in \mathcal { V }$ and a sequence $\alpha _ { j }  0$ such that $\| R _ { \alpha _ { j } } f \|  \infty$ as $j \to \infty$ 

Proof. We prove the theorem by contradiction and assume that $\{ R _ { \alpha } \} _ { \alpha > 0 }$ is uniformly bounded. Hence, there exists a constant C with $\| R _ { \alpha } \| _ { \mathcal { L } ( \mathcal { V } , \mathcal { X } ) } \leqslant C$ for all $\alpha > 0$ . Due to Definition 3.1.1, we have $R _ { \alpha _ { j } }  A ^ { \dagger }$ on $\mathcal { D } ( A ^ { \dagger } )$ for any sequence $\alpha _ { j }  0$ . Since $\mathcal { D } ( A ^ { \dagger } )$ is dense in ${ \mathcal { V } } ,$ by Corollary 3.1.4 we get that $R _ { \alpha _ { i } }$ converges on $\overline { { { \mathcal { D } ( A ^ { \dag } ) } } } = \mathcal { D }$ and therefore $A ^ { \dagger }$ can be extended to a bounded operator on $\dot { \mathcal { L } } ( \mathcal { V } , \mathcal { X } )$ , which is a contradiction to the assumption that $A ^ { \dagger }$ is not continuous (on $\mathcal { D } ( A ^ { \dagger } ) )$ 

To prove the second statement, assume that for all $f \in \mathcal { V }$ and any sequence $\alpha _ { j }  0$ we have 

$$
\sup _ {j \in \mathbb {N}} \| R _ {\alpha_ {j}} f \| _ {\mathcal {Y}} \leqslant C (f) <   \infty .
$$

Then by Theorem 3.1.3 we have that 

$$
\sup _ {j \in \mathbb {N}} \| R _ {\alpha_ {j}} \| _ {\mathcal {L} (\mathcal {Y}, \mathcal {X})} \leqslant C <   \infty ,
$$

which contradicts the first part of the proof. 

With the additional assumption that $\| A R _ { \alpha } \| _ { \mathcal { L } ( \mathcal { X } , \mathcal { X } ) }$ is bounded, we can even show that $R _ { \alpha } f$ diverges for all $f \not \in { \mathcal { D } } ( A ^ { \dagger } )$ 

Theorem 3.1.6. Let $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ and $\{ R _ { \alpha } \} _ { \alpha > 0 }$ be a linear regularisation of $A ^ { \dagger }$ . If 

$$
\sup _ {\alpha > 0} \| A R _ {\alpha} \| _ {\mathcal {L} (\mathcal {X}, \mathcal {X})} <   \infty ,
$$

then $\| R _ { \alpha } f \| _ { \mathcal { X } } \to \infty$ for all $f \not \in { \mathcal { D } } ( A ^ { \dagger } )$ 

Proof. Define $u _ { \alpha } : = R _ { \alpha } f$ for $f \not \in { \mathcal { D } } ( A ^ { \dagger } )$ . Assume that there exists a sequence $\alpha _ { k }  0$ such that $\| u _ { \alpha _ { k } } \| _ { \mathcal { X } }$ is uniformly bounded. Since bounded sets in a Hilbert space are weakly pre-compact, there exists a weakly convergent subsequence $u _ { \alpha _ { k _ { l } } }$ with some limit $u \in \mathcal X$ , cf. [21, Section 2.2, Theorem 2.1]. As continuous linear operators are also weakly continuous, we further have $A u _ { \alpha _ { k _ { I } } }  A u$ 

On the other hand, for any $g \in { \mathcal { D } } ( A ^ { \dagger } )$ we have that $A R _ { \alpha _ { k _ { l } } g }  A A ^ { \dagger } g = P _ { \overline { { { \mathcal { R } ( A ) } } } } g$ as $l  \infty$ . By Corollary 3.1.4 we then conclude that this also holds for any $f \in \mathcal { V }$ , i.e. also for $f \not \in { \mathcal { D } } ( A ^ { \dagger } )$ ). Hence, we get that 

$$
A R _ {\alpha_ {k _ {l}}} f \to P _ {\overline {{\mathcal {R} (A)}}} f
$$

and (see first part of proof) 

$$
A R _ {\alpha_ {k _ {l}}} f = A u _ {\alpha_ {k _ {l}}} \rightharpoonup A u.
$$

Therefore, we get that $A u = P _ { \overline { { \mathcal { R } ( A ) } } } f$ . Since $\mathcal { V } = \overline { { \mathcal { R } ( A ) } } \oplus \mathcal { R } ( A ) ^ { \perp }$ , we get that $f \in \mathcal { R } ( A ) \oplus$ $\mathcal { R } ( A ) ^ { \perp } = \mathcal { D } ( A ^ { \dagger } )$ in contradiction to the assumption $f \not \in { \mathcal { D } } ( A ^ { \dagger } )$ □ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/02cdbfede7953ff9c34347999f5b2f1218beb609c05a3cdb12babdbda302280c.jpg)



Figure 3.1: The total error between a regularised solution and the minimal norm solution decomposes into the data error and the approximation error. These two errors have opposing trends: For a small regularisation parameter α the error in the data gets amplified through the ill-posedness of the problem and for large $\alpha$ the operator $R _ { \alpha }$ is a poor approximation of the Moore–Penrose inverse.


## 3.2 Parameter Choice Rules

We have stated in the beginning of this chapter that we would like to obtain a regularisation that would guarantee that $R _ { \alpha } ( f _ { \delta } )  A ^ { \dagger } f$ for all $f \in { \mathcal { D } } ( A ^ { \dagger } )$ and all $f _ { \delta } \in \mathcal { V } \mathrm { s . t . } \ \lVert f - f _ { \delta } \rVert _ { \mathcal { V } } \leqslant \delta$ as $\delta  0$ . This means that the parameter $\alpha ,$ , referred to as the regularisation parameter, needs to be chosen as a function of δ (and perhaps also $f _ { \delta } )$ so that $\alpha  0$ as $\delta \to 0$ (i.e. we need to regularise less as the data get more precise). 

This can be illustrated with the following observation. For linear regularisations we can split the total error between the regularised solution of the noisy problem $R _ { \alpha } f _ { \delta }$ and the minimal norm solution of the noise-free problem $u ^ { \dagger } = A ^ { \dagger } f$ as 

$$
\begin{array}{r l} & {\| R _ {\alpha} f _ {\delta} - u ^ {\dagger} \| _ {\mathcal {X}} \leqslant \| R _ {\alpha} f _ {\delta} - R _ {\alpha} f \| _ {\mathcal {X}} + \| R _ {\alpha} f - u ^ {\dagger} \| _ {\mathcal {X}}} \\ & {\quad \leqslant \underbrace {\delta \| R _ {\alpha} \| _ {\mathcal {L} (\mathcal {Y} , \mathcal {X})}} _ {\mathrm{dataerror}} + \underbrace {\| R _ {\alpha} f - A ^ {\dagger} f \| _ {\mathcal {X}}} _ {\mathrm{approximationerror}}.} \end{array}\tag{3.1}
$$

The first term of (3.1) is the data error; this term unfortunately does not stay bounded for $\alpha  0$ , which we can conclude from Theorem 3.1.5. The second term, known as the approximation error, however vanishes for $\alpha  0$ , due to the pointwise convergence of $R _ { \alpha }$ to $A ^ { \dagger }$ . Hence it becomes evident from (3.1) that a good choice of α depends on $\delta ,$ and needs to be chosen such that the approximation error becomes as small as possible, whilst the data error is being kept at bay. See Figure 3.1 for an illustration. 

Parameter choice rules are defined as follows. 

Definition 3.2.1. A function α: $\mathbb { R } _ { > 0 } \times \mathcal { Y }  \mathbb { R } _ { > 0 } , ( \delta , f _ { \delta } ) \mapsto \alpha ( \delta , f _ { \delta } )$ is called a parameter choice rule. We distinguish between 

1. a priori parameter choice rules, which depend on δ only; 

2. a posteriori parameter choice rules, which depend on both δ and $f _ { \delta }$ ; 

3. heuristic parameter choice rules, which depend on $f _ { \delta }$ only. 

Now we are ready to define a regularisation that ensures the convergence $R _ { \alpha ( \delta , f _ { \delta } ) } ( f _ { \delta } ) $ $A ^ { \dagger } f$ as $\delta \to 0$ 

Definition 3.2.2. Let $\{ R _ { \alpha } \} _ { \alpha > 0 }$ be a regularisation of $A ^ { \dagger }$ . If for all $f \in { \mathcal { D } } ( A ^ { \dagger } )$ there exists a parameter choice rule $\alpha : \mathbb { R } _ { > 0 } \times \mathcal { Y }  \mathbb { R } _ { > 0 }$ such that 

$$
\lim _ {\delta \to 0} \sup _ {f _ {\delta}: \| f - f _ {\delta} \| _ {\mathcal {Y}} \leqslant \delta} \| R _ {\alpha} f _ {\delta} - A ^ {\dagger} f \| _ {\mathcal {X}} = 0\tag{3.2}
$$

and 

$$
\lim _ {\delta \to 0} \sup _ {f _ {\delta}: \| f - f _ {\delta} \| _ {\mathcal {Y}} \leqslant \delta} \alpha (\delta , f _ {\delta}) = 0\tag{3.3}
$$

then the pair $( R _ { \alpha } , \alpha )$ is called a convergent regularisation. 

## 3.2.1 A priori parameter choice rules

First of all we want to discuss a priori parameter choice rules in more detail. Historically, they were the first to be studied. For every regularisation there exists an a priori parameter choice rule and thus a convergent regularisation. 

Theorem 3.2.3 ([20, Prop $3 . 4 ] )$ . Let $\{ R _ { \alpha } \} _ { \alpha > 0 }$ be a regularisation of $A ^ { \dagger } , f o r A \in { \mathcal { L } } ( { \mathcal { X } } , { \mathcal { Y } } )$ Then there exists an a priori parameter choice rule $\alpha = \alpha ( \delta )$ such that $( R _ { \alpha } , \alpha )$ is a convergent regularisation. 

For linear regularisations, an important characterisation of a priori parameter choice strategies that lead to convergent regularisation methods is as follows. 

Theorem 3.2.4. Let $\{ R _ { \alpha } \} _ { \alpha > 0 }$ be a linear regularisation, and $\alpha : \mathbb { R } _ { > 0 }  \mathbb { R } _ { > 0 }$ an a priori parameter choice rule. Then $( R _ { \alpha } , \alpha )$ is a convergent regularisation method if and only if 

a) lim $\iota _ { \delta \to 0 } \alpha ( \delta ) = 0$ 

b) lim $_ { \delta \to 0 } \delta \| R _ { \alpha ( \delta ) } \| _ { \mathcal { L } ( \mathcal { V } , \mathcal { X } ) } = 0$ 

Proof. : Let condition a) and b) be fulfilled. From (3.1) we then observe that for any $f \in { \mathcal { D } } ( A ^ { \dagger } )$ and $f _ { \delta } \in \mathcal { V }$ s.t. $\| f - f _ { \delta } \| _ { \mathcal { V } } \leqslant \delta$ 

$$
\left\| R _ {\alpha (\delta)} f _ {\delta} - A ^ {\dagger} f \right\| _ {\mathcal {X}} \to 0 \mathrm{for} \delta \to 0.
$$

Hence, $( R _ { \alpha } , \alpha )$ is a convergent regularisation method. 

$\Rightarrow :$ Now let $( R _ { \alpha } , \alpha )$ be a convergent regularisation method. We prove that conditions 1 and 2 have to follow from this by showing that violation of either one of them leads to a contradiction to $( R _ { \alpha } , \alpha )$ being a convergent regularisation method. If condition a) is violated, (3.3) is violated and hence, $( R _ { \alpha } , \alpha )$ is not a convergent regularisation method. If condition a) is fulfilled but condition b) is violated, there exists a null sequence $\{ \delta _ { k } \} _ { k \in \mathbb { N } }$ with $\delta _ { k } \| R _ { \alpha ( \delta _ { k } ) } \| _ { \mathcal { L } ( y , x ) } \geqslant C > 0$ , and hence, we can find a sequence $\{ g _ { k } \} _ { k \in \mathbb { N } } \subset \mathcal { V }$ with $\| g _ { k } \| _ { \mathcal { V } } = 1$ and $\delta _ { k } \| R _ { \alpha ( \delta _ { k } ) } g _ { k } \| _ { \mathcal { X } } \geqslant \widetilde { C }$ for some ${ \widetilde { C } } .$ . Let $f \in { \mathcal { D } } ( A ^ { \dagger } )$ be arbitrary and define $f _ { k } : = f + \delta _ { k } g _ { k }$ Then we have on the one hand $\| f - f _ { k } \| _ { \mathcal { V } } \leqslant \delta _ { k } .$ , but on the other hand the norm of 

$$
R _ {\alpha (\delta_ {k})} f _ {k} - A ^ {\dagger} f = R _ {\alpha (\delta_ {k})} f - A ^ {\dagger} f + \delta_ {k} R _ {\alpha (\delta_ {k})} g _ {k}
$$

cannot converge to zero, as the second term $\delta _ { k } R _ { \alpha ( \delta _ { k } ) } g _ { k }$ is bounded from below by a positive constant $C$ by construction. Hence, (3.2) is violated for $f _ { \delta } = f + \delta _ { k } g _ { k }$ and thus, $( R _ { \alpha } , \alpha )$ is not a convergent regularisation method. □ 

## 3.2.2 A posteriori parameter choice rules

It is easy to convince oneself that if an a priori parameter choice rule $\alpha = \alpha ( \delta )$ defines a convergence regularisation then $\widetilde { \alpha } = \alpha ( C \delta )$ with any $C > 0$ also defines a convergent regularisation (for linear regularisations, it is a trivial corollary of Theorem 3.2.4). Therefore, from the asymptotic point of view, all these regularisations are equivalent. For a fixed error level $\delta ,$ however, they can produce very diferent solutions. Since in practice we have to deal with a typically small, but fixed $\delta ,$ we would like to have a parameter choice rule that is sensitive to this value. To achieve this, we need to use more information than merely the error level $\delta$ to choose the parameter α and we will obtain this information from the approximate data $f _ { \delta }$ 

The basic idea is as follows. Let $f \in { \mathcal { D } } ( A ^ { \dagger } )$ and $f _ { \delta } \in \mathcal { V }$ such that $\| f - f _ { \delta } \| \leqslant \delta$ and consider the residual between $f _ { \delta }$ and $u _ { \alpha } : = R _ { \alpha } f _ { \delta }$ , i.e. 

$$
\left\| A u _ {\alpha} - f _ {\delta} \right\|.
$$

Let $u ^ { \dagger }$ be the minimal norm solution and define 

$$
\mu := \inf \{\| A u - f \|, u \in \mathcal {X} \} = \| A u ^ {\dagger} - f \|.
$$

We observe that $u ^ { \dagger }$ satisfies the following inequality 

$$
\| A u ^ {\dagger} - f _ {\delta} \| \leqslant \| A u ^ {\dagger} - f \| + \| f _ {\delta} - f \| \leqslant \mu + \delta
$$

and in some cases this estimate may be sharp. Hence, it appears not to be useful to choose $\alpha ( \delta , f _ { \delta } )$ with $\| A u _ { \alpha } - f _ { \delta } \| < \mu + \delta .$ . In general, it may be not straightforward to estimate $\mu ,$ but if $\mathcal { R } ( A )$ is dense in $\mathcal { V } ,$ , we get that $\mathcal { R } ( A ) ^ { \perp } = \{ 0 \}$ due to Remark 2.0.2 and $\mu = 0$ Therefore, we ideally ensure that $\mathcal { R } ( A )$ is dense. 

These observations motivate the Morozov’s discrepancy principle, which in the case $\mu = 0$ reads as follows. 

Definition 3.2.5 (Morozov’s discrepancy principle). Let $u _ { \alpha } = R _ { \alpha } f _ { \delta }$ with $\alpha ( \delta , f _ { \delta } )$ chosen as follows 

$$
\alpha (\delta , f _ {\delta}) = \sup \{\alpha > 0 | \| A u _ {\alpha} - f _ {\delta} \| \leqslant \eta \delta \}\tag{3.4}
$$

for given $\delta , \ f _ { \delta }$ and a fixed constant $\eta > 1$ . Then $u _ { \alpha ( \delta , f _ { \delta } ) } = R _ { \alpha ( \delta , f _ { \delta } ) } f _ { \delta }$ is said to satisfy Morozov’s discrepancy principle. 

It can be shown that the a-posteriori parameter choice rule (3.4) indeed yields a convergent regularization method [20, Chapter 4.3]. 

## 3.2.3 Heuristic parameter choice rules

As the measurement error $\delta$ is not always easy to obtain in practice, it is tempting to use a parameter choice rule that only depends on the measured data $f _ { \delta }$ and not on their error $\delta ,$ i.e. to use a heuristic parameter choice rule. Unfortunately, heuristic rules yield convergent regularisations only for well-posed problems, as the following result, known as the Bakushinskii veto [7], demonstrates. 

Theorem 3.2.6 ([20, Thm 3.3]). Let $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ and $\{ R _ { \alpha } \}$ be a regularization for $A ^ { \dagger }$ Let $\alpha = \alpha ( f _ { \delta } )$ be a parameter choice rule such that $( R _ { \alpha } , \alpha )$ is a convergent regularization. Then $A ^ { \dagger }$ is continuous from $\mathcal { V }$ to $\mathcal { X } .$ 

## 3.3 Spectral Regularisation

Recall the spectral representation (2.8) of the Moore-Penrose inverse $A ^ { \dagger }$ 

$$
A ^ {\dagger} f = \sum_ {j = 1} ^ {\infty} \frac {1}{\sigma_ {j}} \left\langle f, y _ {j} \right\rangle x _ {j},
$$

where $\{ ( \sigma _ { j } , x _ { j } , y _ { j } ) \}$ is the singular system of A. 

The source of ill-posedness of $A ^ { \dagger }$ are the eigenvalues $1 / \sigma _ { j }$ , which explode as $j  \infty$ since $\sigma _ { j }  0 \mathrm { ~ a s ~ } j  \infty$ . Let us construct a regularisation by modifying these eigenvalues as follows 

$$
R _ {\alpha} f := \sum_ {j = 1} ^ {\infty} g _ {\alpha} (\sigma_ {j}) \left\langle f, y _ {j} \right\rangle x _ {j}, f \in \mathcal {Y},\tag{3.5}
$$

with an appropriate function $g _ { \alpha } \colon \mathbb { R } _ { + } \to \mathbb { R } _ { + }$ such that $g _ { \alpha } ( \sigma )  { \frac { 1 } { \sigma } }$ as $\alpha  0$ for all $\sigma > 0$ and 

$$
g _ {\alpha} (\sigma) \leqslant C _ {\alpha} \text {   for   all   } \sigma \in \mathbb {R} _ {+}.\tag{3.6}
$$

Theorem 3.3.1. Let $g _ { \alpha } : \mathbb { R } _ { + } \to \mathbb { R } _ { + }$ be a piecewise continuous function satisfying (3.6), lim<sub>α</sub> $\begin{array} { r } { \gamma _ { 0 } g _ { \alpha } ( \sigma ) = \frac { 1 } { \sigma } } \end{array}$ and 

$$
\sup _ {\alpha , \sigma} \sigma g _ {\alpha} (\sigma) \leqslant \gamma\tag{3.7}
$$

for some constant $\gamma > 0$ . If $R _ { \alpha }$ is defined as in (3.5), we have 

$$
R _ {\alpha} f \rightarrow A ^ {\dagger} f a s \alpha \rightarrow 0
$$

for all $f \in { \mathcal { D } } ( A ^ { \dagger } )$ 

Proof. From the singular value decomposition of $A ^ { \dagger }$ and the definition of $R _ { \alpha }$ we obtain 

$$
R _ {\alpha} f - A ^ {\dagger} f = \sum_ {j = 1} ^ {\infty} \left(g _ {\alpha} (\sigma_ {j}) - \frac {1}{\sigma_ {j}}\right) \langle f, y _ {j} \rangle_ {\mathcal {Y}} x _ {j} = \sum_ {j = 1} ^ {\infty} \left(\sigma_ {j} g _ {\alpha} (\sigma_ {j}) - 1\right) \langle u ^ {\dagger}, x _ {j} \rangle_ {\mathcal {X}} x _ {j}.
$$

Consider 

$$
\| R _ {\alpha} f - A ^ {\dagger} f \| _ {\mathcal {X}} ^ {2} = \sum_ {j = 1} ^ {\infty} (\sigma_ {j} g _ {\alpha} (\sigma_ {j}) - 1) ^ {2} \left| \langle u ^ {\dagger}, x _ {j} \rangle_ {\mathcal {X}} \right| ^ {2}.
$$

From (3.7) we can conclude 

$$
(\sigma_ {j} g _ {\alpha} (\sigma_ {j}) - 1) ^ {2} \leqslant (1 + \gamma^ {2}),
$$

whilst 

$$
\sum_ {j = 1} ^ {\infty} (1 + \gamma^ {2}) \left| \langle u ^ {\dagger}, x _ {j} \rangle_ {\mathcal {X}} \right| ^ {2} = (1 + \gamma^ {2}) \| u ^ {\dagger} \| ^ {2} <   + \infty .
$$

Therefore, by the reverse Fatou lemma we get the following estimate 

$$
\begin{array}{l} \underset {\alpha \to 0} {\limsup} \left\| R _ {\alpha} f - A ^ {\dagger} f \right\| _ {\mathcal {X}} ^ {2} = \underset {\alpha \to 0} {\limsup} \sum_ {j = 1} ^ {\infty} (\sigma_ {j} g _ {\alpha} (\sigma_ {j}) - 1) ^ {2} \left(\langle u ^ {\dagger}, x _ {j} \rangle_ {\mathcal {X}}\right) ^ {2} \\ \leqslant \sum_ {j = 1} ^ {\infty} \left(\underset {\alpha \to 0} {\limsup} \sigma_ {j} g _ {\alpha} (\sigma_ {j}) - 1\right) ^ {2} \left| \langle u ^ {\dagger}, x _ {j} \rangle_ {\mathcal {X}} \right| ^ {2} = 0, \end{array}
$$

where the last equality is due to the pointwise convergence of $g _ { \alpha } ( \sigma _ { j } )$ to $1 / \sigma _ { j }$ . Hence, we have $\left\| R _ { \alpha } f - A ^ { \dagger } f \right\| _ { \mathcal { X } } \to 0$ for $\alpha  0$ for all $f \in { \mathcal { D } } ( A ^ { \dagger } )$ □ 

Theorem 3.3.2. Let the assumptions of Theorem 3.3.1 hold and let $\alpha = \alpha ( \delta )$ be an $a \mathrm { - }$ priori parameter choice rule. Then $( R _ { \alpha ( \delta ) } , \alpha ( \delta ) )$ with $R _ { \alpha }$ as defined in (3.5) is a convergent regularisation method if 

$$
\lim _ {\delta \to 0} \delta C _ {\alpha (\delta)} = 0.
$$

Proof. The result follows immediately from $\| R _ { \alpha ( \delta ) } \| _ { \mathcal { L } ( \mathcal { X } , \mathcal { V } ) } \leqslant C _ { \alpha ( \delta ) }$ and Theorem 3.2.4. 

## 3.3.1 Truncated singular value decomposition

As a first example for a spectral regularisation of the form (3.5) we want to consider the so-called truncated singular value decomposition. The idea is to discard all singular values below a certain threshold $\alpha ,$ which is achieved using the following function $g _ { \alpha }$ 

$$
g _ {\alpha} (\sigma) = \left\{ \begin{array}{l l} \frac {1}{\sigma} & \sigma \geqslant \alpha \\ 0 & \sigma <   \alpha \end{array} \right..\tag{3.8}
$$

Note that for all $\sigma > 0$ we naturally obtain lim $1 _ { \alpha \to 0 } g _ { \alpha } ( \sigma ) = 1 / \sigma$ . Condition (3.7) is obviously satisfied with $\gamma = 1$ and condition (3.6) with $\begin{array} { r } { C _ { \alpha } = \frac { 1 } { \alpha } } \end{array}$ . Therefore, truncated SVD is a convergent regularisation if 

$$
\lim _ {\delta \to 0} \frac {\delta}{\alpha} = 0.\tag{3.9}
$$

Equation (3.5) then reads as follows 

$$
R _ {\alpha} f = \sum_ {\sigma_ {j} \geqslant \alpha} \frac {1}{\sigma_ {j}} \langle f, y _ {j} \rangle_ {\mathcal {Y}} x _ {j},\tag{3.10}
$$

for all $f \in \mathcal { V }$ . Note that the sum in (3.10) is always well-defined (i.e. finite) for any $\alpha > 0$ as zero is the only accumulation point of singular vectors of compact operators. 

Let $A \in \mathcal { K } ( \mathcal { X } , \mathcal { Y } )$ with singular system $\{ ( \sigma _ { j } , x _ { j } , y _ { j } ) \} _ { j \in \mathbb { N } }$ , and choose for $\delta > 0$ an index function $j ^ { * } : \mathbb { R } _ { + } \to \mathbb { N }$ with $j ^ { * } ( \delta ) \to \infty$ for $\delta \  \ 0$ and lim<sub>δ</sub> $_ {  0 } \delta / \sigma _ { j ^ { * } ( \delta ) } = 0$ . We can then choose ${ \alpha ( \delta ) = \sigma _ { j ^ { * } ( \delta ) } }$ as an a-priori parameter choice rule to obtain a convergent regularisation. 

Note that in practice a larger δ implies that more and more singular values have to be cut of in order to guarantee a stable recovery that successfully suppresses the data error. 

A disadvantage of this approach is that it requires the knowledge of the singular vectors of A (only finitely many, but the number can still be large). 

## 3.3.2 Tikhonov regularisation

The main idea behind Tikhonov regularisation<sup>1</sup> is to consider the normal equations and shift the eigenvalues of $A ^ { * } A$ by a constant factor, which will be associated with the regularisation parameter α. This shift can be realised via the function 

$$
g _ {\alpha} (\sigma) = \frac {\sigma}{\sigma^ {2} + \alpha}\tag{3.11}
$$

and the corresponding Tikhonov regularisation (3.5) reads as follows 

$$
R _ {\alpha} f = \sum_ {j = 1} ^ {\infty} \frac {\sigma_ {j}}{\sigma_ {j} ^ {2} + \alpha} \langle f, y _ {j} \rangle y x _ {j}.\tag{3.12}
$$

Again, we immediately observe that for all $\sigma > 0$ we have lim $\alpha {  } 0 g _ { \alpha } ( \sigma ) = 1 / \sigma$ . Condition (3.7) is satisfied with $\gamma = 1$ . Since $0 \leqslant ( \sigma - { \sqrt { \alpha } } ) ^ { 2 } = \sigma ^ { 2 } - 2 \sigma { \sqrt { \alpha } } + \alpha$ , we get that $\sigma ^ { 2 } + \alpha \geqslant 2 \sigma \sqrt { \alpha }$ and 

$$
{\frac {\sigma}{\sigma^ {2} + \alpha}} \leqslant {\frac {1}{2 {\sqrt {\alpha}}}}.
$$

This estimate implies that (3.6) holds with $\begin{array} { r } { C _ { \alpha } = \frac { 1 } { 2 \sqrt { \alpha } } } \end{array}$ . Therefore, Tikhonov regularisation is a convergent regularisation if 

$$
\lim _ {\delta \to 0} \frac {\delta}{\sqrt {\alpha}} = 0.\tag{3.13}
$$

The formula (3.12) suggests that we need all singular vectors of A in order to compute the regularisation. However, we note that $\sigma _ { j } ^ { 2 }$ are the eigenvalues of A<sup>∗</sup>A and, hence, $\sigma _ { j } ^ { 2 } + \alpha$ are the eigenvectors of $A ^ { * } A +$ αI (where I is the identity operator). Applying this operator to the regularised solution $u _ { \alpha } = R _ { \alpha } f _ { \alpha }$ , we get 

$$
(A ^ {*} A + \alpha I) u _ {\alpha} = \sum_ {j = 1} ^ {\infty} (\sigma_ {j} ^ {2} + \alpha) \langle u _ {\alpha}, x _ {j} \rangle_ {\mathcal {X}} x _ {j} = \sum_ {j = 1} ^ {\infty} (\sigma_ {j} ^ {2} + \alpha) \frac {\sigma_ {j}}{\sigma_ {j} ^ {2} + \alpha} \langle f, y _ {j} \rangle_ {\mathcal {Y}} x _ {j} = A ^ {*} f.
$$

Therefore, the regularised solution $u _ { \alpha }$ can be computed without knowing the singular system of A by solving the following well-posed linear equation 

$$
(A ^ {*} A + \alpha I) u _ {\alpha} = A ^ {*} f.\tag{3.14}
$$

Remark 3.3.3. Rewriting equation (3.14) as 

$$
A ^ {*} (A u _ {\alpha} - f) + \alpha u _ {\alpha} = 0,
$$

we note that it looks like a condition for the minimum of some quadratic form. Indeed, it can be easily checked that (3.14) is the first order optimality condition for the following optimisation problem 

$$
\min _ {u \in \mathcal {X}} \frac {1}{2} \| A u - f \| ^ {2} + \alpha \| u \| ^ {2}.\tag{3.15}
$$

The condition (3.14) is necessary (and, by convexity, suficient) for the minimum of the functional in (3.15). Therefore, the regularised solution $u _ { \alpha }$ can also be computed by solving (numerically) the variational problem (3.15). This is the starting point for modern variational regularisation methods, which we will consider in the next chapter. 

## Chapter 4

# Variational Regularisation

Recall the variation formulation of Tikhonov regularisation for some data $f _ { \delta } \in \mathcal { V }$ 

$$
\min _ {u \in \mathcal {X}} \| A u - f _ {\delta} \| ^ {2} + \alpha \| u \| ^ {2}.
$$

The first term in this expression, $\| A u - f _ { \delta } \| ^ { 2 }$ , penalises the misfit between the predictions of the operator A and the measured data $f _ { \delta }$ and is called the fidelity function or fidelity term. The second term, $\| u \| ^ { 2 }$ penalises some unwanted features of the solution (in this case, a large norm) and is called the regularistaion term. The regularisation parameter α in this context balances the influence of these two terms on the functional to be minimised. 

More generally, using the notation $\mathcal { I } ( u )$ for the regulariser, we can formally write down the variational regularisation problem as follows 

$$
\min _ {u \in \mathcal {X}} \frac {1}{2} \| A u - f _ {\delta} \| ^ {2} + \alpha \mathcal {J} (u),\tag{4.1}
$$

(the $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ in front of the fidelity term is there to simplify notation later). The regularisation operator $R _ { \alpha }$ is defined as follows 

$$
R _ {\alpha} f _ {\delta} \in \underset {u \in \mathcal {X}} {\arg \min} \frac {1}{2} \| A u - f _ {\delta} \| ^ {2} + \alpha \mathcal {J} (u).
$$

In general, the minimiser doesn’t have to unique, hence the inclusion and not equality. Other fidelity terms (not just $\| A u - f _ { \delta } \| ^ { 2 } )$ are possible and useful in many situations. In this course, however, we will use the squared norm for the sake of simplicity. 

In this chapter, we will study the properties of (4.1) for diferent choices of $\mathcal { I }$ , but before that we will recall some necessary theoretical concepts. 

## 4.1 Background

## 4.1.1 Banach spaces and weak convergence

Banach spaces are complete, normed vector spaces (as Hilbert spaces) but they may not have an inner product. For every Banach space $\mathcal { X } ,$ , we can define the space of linear and continuous functionals which is called the dual space $\mathcal { X } ^ { \ast }$ of $\mathcal { X }$ , i.e. $\mathcal { X } ^ { \ast } : = \mathcal { L } ( \mathcal { X } , \mathbb { R } )$ . Let $u \in \mathcal X$ and $p \in \mathcal { X } ^ { \ast }$ , then we usually write the dual product $\langle p , u \rangle$ instead of $p ( u )$ . Moreover, for any $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ there exists a unique operator $A ^ { * } \colon \mathcal { V } ^ { * } \to \mathcal { X } ^ { * }$ , called the adjoint of A such that for all $u \in \mathcal X$ and $p \in \mathcal { V } ^ { * }$ we have 

$$
\langle A ^ {*} p, u \rangle = \langle p, A u \rangle .
$$

It is easy to see that either side of the equation are well-defined, e.g. $A ^ { \ast } p \in { \mathcal { X } } ^ { \ast }$ and $u \in \mathcal X$ 

The dual space of a Banach space  can be equipped with the following norm 

$$
\| p \| _ {\mathcal {X} ^ {*}} = \sup _ {u \in \mathcal {X}, \| u \| _ {\mathcal {X}} \leqslant 1} \langle p, u \rangle .
$$

With this norm the dual space is itself a Banach space. Therefore, it has a dual space as well which we will call the bi-dual space of $\mathcal { X }$ and denote it with $\mathcal { X } ^ { \ast \ast } : = ( \mathcal { X } ^ { \ast } ) ^ { \ast }$ . As every $u \in \mathcal X$ defines a continuous and linear mapping on the dual space $\mathcal { X } ^ { \ast }$ by 

$$
\langle E (u), p \rangle := \langle p, u \rangle ,
$$

the mapping $E \colon \mathcal X \to \mathcal X ^ { * * }$ is well-defined. It can be shown that E is a linear and continuous isometry (and thus injective). In the special case when $E$ is surjective, we call reflexive. Examples of reflexive Banach spaces include Hilbert spaces and $L ^ { q } , \ell ^ { q }$ spaces with $1 \ <$ $q < \infty$ . We call the space separable if there exists a set $\mathcal { X } ^ { \prime } \subset \mathcal { X }$ of at most countable cardinality such that ${ \overline { { { \mathcal { X } } ^ { \prime } } } } = { \mathcal { X } }$ 

A problem in infinite dimensional spaces is that bounded sequences may fail to have convergent subsequences. An example is for instance in $\ell ^ { 2 }$ the sequence $\{ u ^ { k } \} _ { k \in \mathbb { N } } \subset \ell ^ { 2 } , u _ { \ i } ^ { k } = 1$ if $k = j$ and 0 otherwise. It is easy to see that $\| u ^ { k } \| _ { \ell ^ { 2 } } = 1$ and that there is no $u \in \ell ^ { 2 }$ such that $u ^ { k } \to u$ . To circumvent this problem, we define a weaker topology on $\mathcal { X } .$ . We say that $\{ u ^ { k } \} _ { k \in \mathbb { N } } \subset \mathcal { X }$ converges weakly to $u \in \mathcal X$ if and only if for all $p \in \mathcal { X } ^ { \ast }$ the sequence of real numbers $\{ \langle p , u ^ { k } \rangle \} _ { k \in \mathbb { N } }$ converges and 

$$
\langle p, u _ {j} \rangle \rightarrow \langle p, u \rangle .
$$

We will denote weak convergence by $u ^ { k }  u . \ \mathrm { O n }$ a dual space $\mathcal { X } ^ { \ast }$ we could define another topology (in addition to the strong topology induced by the norm and the weak topology as the dual space is a Banach space as well). We say a sequence $\{ p ^ { k } \} _ { k \in \mathbb { N } } \subset \mathcal { X } ^ { * }$ converges weakly- to $p \in \mathcal { X } ^ { \ast }$ if and only if 

$$
\left\langle p ^ {k}, u \right\rangle\rightarrow \left\langle p, u \right\rangle \quad \text { for   all } u \in \mathcal {X}
$$

and we denote weak- convergence by $p ^ { k } \to ^ { * } p .$ . Similarly, for any topology $\tau$ on $\mathcal { X }$ we denote the convergence in that topology by $u ^ { k } \overset { \tau } { \to } u$ 

With these two new notions of convergence, we can solve the problem of bounded sequences: 

Theorem 4.1.1 (Banach-Alaoglu Theorem, e.g. [32, p. 70] or $[ 3 6 , \mathrm { p } . \ 1 4 1 ] )$ . Let $\mathcal { X } = ( \mathcal { X } ^ { \circ } ) ^ { * }$ be the dual of a Banach space $\mathcal { X } ^ { \diamond }$ . Then the unit ball $B _ { \mathcal { X } } = \{ u \in \mathcal { X } \colon \| x \| \leqslant 1 \}$ is compact in the weak- topology. $I f \mathcal { X } ^ { \diamond }$ is separable, then the weak- topology is metrisable on bounded sets and every bounded sequence $\{ u ^ { k } \} _ { k \in \mathbb { N } } \subset \mathcal X$ has a wea $k ^ { * }$ convergent subsequence. 

Theorem 4.1.2 $( [ 3 8 , \ \mathrm { p } . \ \ 6 4 ] )$ . Each bounded sequence $\{ u ^ { k } \} _ { k \in \mathbb { N } }$ in a reflexive separable Banach space has a weakly convergent subsequence. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/02daf066debb7bc0034f30ab1b9601138fd1b2beeb82f45058fe4808821f858d.jpg)



Figure 4.1: Visualisation of lower semi-continuity. The solid dot at a jump indicates the value that the function takes. The function on the left is continuous and thus lower semicontinuous. The functions in the middle and on the right are discontinuous. While the function in the middle is lower semi-continuous, the function on the right is not (due to the limit from the left at the discontinuity).


An important property of functionals, which we will need later, is sequential lower semicontinuity. Roughly speaking this means that the functional values for arguments near an argument u are either close to $E ( u )$ or greater than $E ( u )$ 

Definition 4.1.3. Let  be a Banach space with topology $\tau _ { \mathcal { X } }$ . The functional $E \colon \mathcal { X }  \bar { \mathbb { R } }$ is said to be sequentially lower semi-continuous with respect to $\tau _ { \mathcal { X } } ~ \left( \tau _ { \mathcal { X } }  – l . s . c . \right)$ at $u \in \mathcal { X } \mathrm { ~ } i f$ 

$$
E (u) \leqslant \operatorname * {l i m i n f} _ {j \to \infty} E (u _ {j})
$$

for all sequences $\{ u _ { j } \} _ { j \in \mathbb { N } } \subset \mathcal { X }$ with $u _ { j } \to u$ in the topology $\tau _ { \mathcal { X } }$ of $\mathcal { X }$ . 

Remark 4.1.4. For topologies that are not induced by a metric we have to difer between a topological property and its sequential version, e.g. continuous and sequentially continuous. If the topology is induced by a metric, then these two are the same. However, for instance the weak and weak- topology are generally not induced by a metric (but this is true on bounded sets). 

Example 4.1.5. The functional $\| \cdot \| _ { 1 } : \ell ^ { 2 } \to \bar { \mathbb { R } }$ with 

$$
\| u \| _ {1} = \left\{ \begin{array}{l l} \sum_ {j = 1} ^ {\infty} | u _ {j} | & \text { if } u \in \ell^ {1} \\ \infty & \text { else } \end{array} \right.
$$

is weakly (and, hence, strongly) lower semi-continuous in $\ell ^ { 2 }$ . 

Proof. Let $\{ u ^ { j } \} _ { j \in \mathbb { N } } \subset \ell ^ { 2 }$ be a weakly convergent sequence with $u ^ { j } \to u \in \ell ^ { 2 } .$ . We have with $\delta _ { k } : \ell ^ { 2 } \to \mathbb { R } , \langle \delta _ { k } , v \rangle = v _ { k }$ that for all $k \in \mathbb N$ 

$$
u _ {k} ^ {j} = \langle \delta_ {k}, u ^ {j} \rangle \rightarrow \langle \delta_ {k}, u \rangle = u _ {k}.
$$

The assertion follows then with Fatou’s lemma 

$$
\| u \| _ {1} = \sum_ {k = 1} ^ {\infty} | u _ {k} | = \sum_ {k = 1} ^ {\infty} \lim _ {j \to \infty} | u _ {k} ^ {j} | \leqslant \operatorname * {l i m i n f} _ {j \to \infty} \sum_ {k = 1} ^ {\infty} | u _ {k} ^ {j} | = \operatorname * {l i m i n f} _ {j \to \infty} \| u ^ {j} \| _ {1}.
$$

Note that it is not clear whether both the left and the right hand side are finite. 

## 4.1.2 Convex analysis

## Infinity calculus

We will look at functionals $E : \mathcal { X }  \bar { \mathbb { R } }$ whose range is modelled to be the extended real line $\bar { \mathbb { R } } : = \mathbb { R } \cup \{ - \infty , + \infty \}$ where the symbol + denotes an element that is not part of the real line that is by definition larger than any other element of the reals, i.e. 

$$
x <   + \infty
$$

for all $x \in \mathbb { R }$ (similarly, $x > - \infty$ for all $x \in \mathbb { R } )$ . This is useful to model constraints: for instance, if we were trying to minimise $E : [ - 1 , \infty ) \to \mathbb { R } , x \mapsto x ^ { 2 }$ we could remodel this minimisation problem by $\widetilde { E } : \mathbb { R } \to \bar { \mathbb { R } }$ 

$$
\widetilde {E} (x) = \left\{ \begin{array}{l l} x ^ {2} & \text { if } x \geqslant - 1 \\ \infty & \text { else } \end{array} \right..
$$

Obviously both functionals have the same minimiser but $\widetilde { E }$ is defined on a vector space and not only on a subset. This has two important consequences: on the on hand, it makes many theoretical arguments easier as we do not need to worry whether $E ( x + y )$ is defined or not. On the other hand, it makes practical implementations easier as we are dealing with unconstrained optimisation instead of constrained optimisation. This comes at a cost that some algorithms are not applicable any more, e.g. the function $\widetilde { E }$ is not diferentiable everywhere whereas E is (in the interior of its domain). 

It is useful to note that one can calculate on the extended real line $\bar { \mathbb R }$ as we are used to on the real line <sup>R</sup> but the operations with need yet to be defined. 

Definition 4.1.6. The extended real line is defined as $\bar { \mathbb { R } } : = \mathbb { R } \cup \{ - \infty , + \infty \}$ with the following rules that hold for any $x \in \mathbb { R }$ and $\lambda > 0$ : 

$$
\begin{array}{c} x \pm \infty := \pm \infty + x := \pm \infty \\ \lambda \cdot (\pm \infty) := \pm \infty \cdot \lambda := \pm \infty , - 1 \cdot (\pm \infty) := \mp \infty \\ x / (\pm \infty) := 0 \\ \infty + \infty := \infty , - \infty - \infty := - \infty . \end{array}
$$

Some calculations are not defined, e.g., 

$$
+ \infty - \infty \text {   and   } (\pm \infty) \cdot (\pm \infty).
$$

Using functions with values on the extended real line, one can easily describe sets $\mathcal { C } \subset \mathcal { X }$ 

Definition 4.1.7 (Characteristic function). Let $\mathcal { C } \subset \mathcal { X }$ be a set. The function $\chi c \colon \mathcal { X }  \bar { \mathbb { R } }$ 4 

$$
\chi_ {\mathcal {C}} (u) = \left\{ \begin{array}{l l} 0 & u \in \mathcal {C} \\ \infty & u \in \mathcal {X} \setminus \mathcal {C} \end{array} \right.
$$

is called the characteristic function of the set $\mathcal { C } .$ 

Using characteristic functions, one can easily write constrained optimisation problems as unconstrained ones: 

$$
\min _ {u \in \mathcal {C}} E (u) \quad \Leftrightarrow \quad \min _ {u \in \mathcal {X}} E (u) + \chi_ {\mathcal {C}} (u).
$$

## 4.1. BACKGROUND

Definition 4.1.8. Let be a vector space and E : <sup>R¯</sup> a functional. Then the efective domain of E is 

$$
\operatorname{dom} (E) := \left\{u \in \mathcal {X} \mid E (u) <   \infty \right\}.
$$

Definition 4.1.9. A functional E is called proper if the efective domain dom(E) is not empty. 

Convexity 

A property of fundamental importance of sets and functions is convexity. 

Definition 4.1.10. Let  be a vector space. A subset $\mathcal { C } \subset \mathcal { X }$ is called convex, $i f \lambda u + ( 1 -$ $\lambda ) v \in { \mathcal { C } }$ for all $\lambda \in ( 0 , 1 )$ and all $u , v \in { \mathcal { C } }$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/7b576edb83eb47ec961f15b98064e60452351ce944a18da50c3ba4c1547c456b.jpg)



Figure 4.2: Example of a convex set (left) and non-convex set (right).


Definition 4.1.11. A functional E : $\mathcal { X }  \bar { \mathbb { R } }$ is called convex, $i f$ 

$$
E (\lambda u + (1 - \lambda) v) \leqslant \lambda E (u) + (1 - \lambda) E (v)
$$

for all $\lambda \in \mathsf { \Gamma } ( 0 , 1 )$ and all $u , v \in \mathrm { d o m } ( E )$ with $u \ne v$ . It is called strictly convex if the inequality is strict. It is called strongly convex with constant θ if $E ( u ) - \theta \| u \| ^ { 2 }$ is convex. 

Obviously, strong convexity implies strict convexity and strict convexity implies convexity. 

Example 4.1.12. The absolute value function $\mathbb { R }  \mathbb { R } , x \mapsto | x |$ is convex but not strictly convex. The quadratic function $x \mapsto x ^ { 2 }$ is strongly (and hence strictly) convex. The function $x \mapsto x ^ { 4 }$ is strictly convex, but not strongly convex. For other examples, see Figure 4.3. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/7b6ae108fbcfeebaf631c5160defaa52d63bf0abbd8c67363f41ce5e172b454a.jpg)



Figure 4.3: Example of a convex function (left), a strictly convex function (middle) and a non-convex function (right).


Example 4.1.13. The characteristic function $\chi c ( u )$ is convex if and only if is a convex set. To see the convexity, let $u , v \in \mathrm { d o m } ( \chi c ) = \mathcal { C } .$ . Then by the convexity of the convex combination $\lambda u + ( 1 - \lambda ) v$ is as well in and both the left and the right hand side of the desired inequality are zero. 

Lemma 4.1.14. Let α <sup>></sup> 0 and $E , F \colon \mathcal { X } \  \ \bar { \mathbb { R } }$ be two convex functionals. Then $E +$ $\alpha F \colon \mathcal { X } \to \bar { \mathbb { R } }$ is convex. Furthermore, $i f \alpha > 0$ and F strictly convex, then $E + \alpha F$ is strictly convex. 

## Fenchel conjugate

In convex optimisation problems (i.e. those involving convex functions) the concept of Fenchel conjugates plays a very important role. 

Definition 4.1.15. Let $E \colon \mathcal { X }  \bar { \mathbb { R } }$ be a functional. The functional $E ^ { * } \colon { \mathcal { X } } ^ { * }  { \bar { \mathbb { R } } } _ { }$ 

$$
E ^ {*} (p) = \sup _ {u \in \mathcal {X}} [ \langle p, u \rangle - E (u) ],
$$

is called the Fenchel conjugate of E. 

Theorem 4.1.16 ([19, Prop. 4.1]). For any functional $E \colon \mathcal { X }  \bar { \mathbb { R } }$ the following inequality holds: 

$$
E ^ {* *} := (E ^ {*}) ^ {*} \leqslant E.
$$

If E is proper, lower-semicontinuous (see Def. 4.1.3) and convex, then 

$$
E ^ {* *} = E.
$$

## Subgradients

For convex functions one can generalise the concept of a derivative so that it would also make sense for non-diferentiable functions. 

Definition 4.1.17. A functional $E \colon \mathcal { X }  \bar { \mathbb { R } }$ is called subdiferentiable at $u \in \mathcal X$ , if there exists an element $p \in \mathcal { X } ^ { \ast }$ such that 

$$
E (v) \geqslant E (u) + \langle p, v - u \rangle
$$

holds, for all $v \in \mathcal { X }$ . Furthermore, we call p a subgradient at position u. The collection of all subgradients at position u, i.e. 

$$
\partial E (u) := \left\{p \in \mathcal {X} ^ {*} \mid E (v) \geqslant E (u) + \langle p, v - u \rangle , \forall v \in \mathcal {X} \right\},
$$

is called subdiferential of E at u. 

It is clear that if a convex functional $E \colon \mathcal { X }  \bar { \mathbb { R } }$ is proper, i.e. $\mathrm { d o m } ( E ) \neq \emptyset$ , then for all u dom(E) the subdiferential is empty. A suficient (but not necessary) condition for E to have a subgradient at $u \in \mathrm { d o m } ( E )$ is given by 

Proposition 4.1.18 ([19, Prop. 5.2]). Let $E \colon \mathcal { X }  \bar { \mathbb { R } }$ be a convex functional and $u \in$ dom(E) such that E is continuous at u. Then $\partial E ( u ) \neq \emptyset$ 

Theorem 4.1.19 ([4, Thm. 7.13]). Let $E \colon \mathcal { X }  \bar { \mathbb { R } }$ be a proper convex function and $u \in$ dom(E). Then $\partial E ( u )$ is a weak- compact convex subset of $\mathcal { X } ^ { \ast }$ 

## 4.1. BACKGROUND

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/a216b84fd918cd074594a0d49681fee9d25059f33cf78dae2667c6ed77cee444.jpg)



Figure 4.4: Visualisation of the subdiferential. Linear approximations of the functional have to lie completely underneath the function. For points where the function is not diferentiable there may be more than one such approximation.


For diferentiable functions the subdiferential consists of just one element – the derivative. For non-diferentiable functionals the subdiferential is multivalued; we want to con sider the subdiferential of the absolute value function as an illustrative example. 

Example 4.1.20. Let $E \colon \mathbb { R } $ <sup>R</sup> be the absolute value function $E ( u ) = | u |$ . Then, the subdiferential of E at u is given by 

$$
\partial E (u) = \left\{ \begin{array}{l l} \{1 \} & \text { for } u > 0 \\ [ - 1, 1 ] & \text { for } u = 0 \\ \{- 1 \} & \text { for } u <   0 \end{array} \right.,
$$

which you will prove as an exercise. A visual explanation is given in Figure 4.4. 

The subdiferential of a sum of two functions can be characterised as follows. 

Theorem 4.1.21 ([19, Prop. 5.6]). Let $E \colon \mathcal { X }  \bar { \mathbb { R } }$ and $F \colon \mathcal { X }  \bar { \mathbb { R } }$ be proper l.s.c. convex functions and suppose u $\in \mathrm { d o m } ( E )$ dom(F ) such that E is continuous at u. Then 

$$
\partial (E + F) = \partial E + \partial F.
$$

Using the subdiferential, one can characterise minimisers of convex functionals. 

Theorem 4.1.22. An element $u \in \mathcal X$ is a minimiser of the functional E : $\mathcal { X }  \bar { \mathbb { R } }$ if and only $i f \left( 0 \in \partial E ( u ) \right.$ 

Proof. By definition, $0 \in \partial E ( u )$ if and only if for all $v \in \mathcal { X }$ it holds 

$$
E (v) \geqslant E (u) + \langle 0, v - u \rangle = E (u),
$$

which is by definition the case if and only if u is a minimiser of E. 

The next result connects subgradients and convex conjugates 

Theorem 4.1.23 ([19, Prop. 5.1]). Let $E \colon { \mathcal { X } } $ <sup>R¯</sup> be a convex function and $E ^ { * } \colon { \mathcal { X } } ^ { * } \to$ R¯ its convex conjugate. Then $p \in \partial E ( u )$ if and only if 

$$
E (u) + E ^ {*} (p) = \langle p, u \rangle .
$$

Proof. Left as an exercise. 

## Bregman distances

Convex functions naturally define some distance measure that became known as the Bregman distance. 

Definition 4.1.24. Let $E \colon \mathcal { X }  \bar { \mathbb { R } }$ be a convex functional. Moreover, let $u , v \in \mathcal { X } , E ( v ) <$ and $q \in \partial E ( v )$ . Then the (generalised) Bregman distance of E between u and v is defined as 

$$
D _ {E} ^ {q} (u, v) := E (u) - E (v) - \langle q, u - v \rangle .\tag{4.2}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/dd69826dbc468e1f2f942c6fa5e7dabd64db9ba002a362203fdc1689b58d1cce.jpg)



Figure 4.5: Visualization of the Bregman distance.


Remark 4.1.25. It is easy to check that a Bregman distance somewhat resembles a metric as for all $u , v \in \mathcal { X } , q \in \partial E ( v )$ we have that $D _ { E } ^ { q } ( u , v ) \geqslant 0$ and $D _ { E } ^ { q } ( v , v ) = 0$ . There are functionals where the Bregman distance (up to a square root) is actually a metric; e.g. $\begin{array} { r } { E ( u ) : = \frac 1 2 \| u \| _ { \mathcal { X } } ^ { 2 } } \end{array}$ for Hilbert space $x ,$ , then $\begin{array} { r } { D _ { E } ^ { q } ( u , v ) = \frac 1 2 \| u - v \| _ { \mathcal { X } } ^ { 2 } } \end{array}$ . However, in general, Bregman distances are not symmetric and $D _ { E } ^ { q } ( u , v ) = 0$ does not imply $u = v ,$ , as you will see on the example sheets. 

To overcome the issue of non-symmetry, one can introduce the so-called symmetric Bregman distance. 

Definition 4.1.26. Let $E \colon \mathcal { X }  \bar { \mathbb { R } }$ be a convex functional. Moreover, let $u , v \in \mathcal { X } , E ( u ) <$ $\infty , E ( v ) < \infty , q \in \partial E ( v )$ and $p \in \partial E ( u )$ . Then the symmetric Bregman distance of E between u and v is defined as 

$$
D _ {E} ^ {s y m m} (u, v) := D _ {E} ^ {q} (u, v) + D _ {E} ^ {p} (v, u) = \langle p - q, u - v \rangle .\tag{4.3}
$$

Absolutely one-homogeneous functionals 

Definition 4.1.27. A functional $E \colon \mathcal { X }  \bar { \mathbb { R } }$ is called absolutely one-homogeneous if 

$$
E (\lambda u) = | \lambda | E (u) \quad \forall \lambda \in \mathbb {R}, \forall u \in \mathcal {X}.
$$

Absolutely one-homogeneous convex functionals have some useful properties, for example, it is obvious that $E ( 0 ) = 0$ . Some further properties are listed below. 

Proposition 4.1.28. Let $E ( \cdot )$ be a convex absolutely one-homogeneous functional and let $p \in \partial E ( u )$ . Then the following equality holds: 

$$
E (u) = \langle p, u \rangle .
$$

## 4.1. BACKGROUND

Proof. Left as exercise. 

Remark 4.1.29. The Bregman distance $D _ { E } ^ { p } ( v , u )$ in this case can be written as follows: 

$$
D _ {E} ^ {p} (v, u) = E (v) - \langle p, v \rangle .
$$

Proposition 4.1.30. Let $E ( \cdot )$ be a proper, convex, l.s.c. and absolutely one-homogeneous functional. Then the Fenchel conjugate $E ^ { * } ( \cdot )$ is the characteristic function of the convex set $\partial E ( 0 )$ 

Proof. Left as exercise. 

An obvious consequence of the above results is the following 

Proposition 4.1.31. For any u $\in { \mathcal { X } } , p \in \partial E ( u )$ if and only $i f p \in \partial E ( 0 )$ and $E ( u ) = ( p , u )$ 

## 4.1.3 Minimisers

Definition 4.1.32. Let $E \colon \mathcal { X }  \bar { \mathbb { R } }$ be a functional. We say that $u ^ { * } \in \mathcal { X }$ solves the minimisation problem 

$$
\min _ {u \in \mathcal {X}} E (u)
$$

if and only if $E ( u ^ { * } ) < \infty$ and $E ( u ^ { * } ) \leqslant E ( u )$ , for all $u \in \mathcal X$ . We call $u ^ { * }$ a minimiser of E. 

Definition 4.1.33. A functional $E \colon \mathcal { X }  \bar { \mathbb { R } }$ is called bounded from below if there exists a constant $C > - \infty$ such that for all $u \in \mathcal X$ we have $E ( u ) \geqslant C$ 

This condition is obviously necessary for the finiteness of the infimum inf $\overset { \cdot } { u } \in \mathcal { X }  \overset { \cdot } { E } ( u )$ 

## Existence

If all minimising sequences (that converge to the infimum assuming it exists) are unbounded, then there cannot exist a minimiser. A suficient condition to avoid such a scenario is coercivity. 

Definition 4.1.34. A functional $E \colon \mathcal { X } \  \ \bar { \mathbb { R } }$ is called coercive, if for all $\{ u _ { j } \} _ { j \in \mathbb { N } }$ with $\| u _ { j } \| _ { \mathcal { X } } \to \infty$ we have $E ( u _ { j } )  \infty$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/9119010e4bec705d9c9c511f2055e90702a48089c4c55c283cd0dbbcde13fdcc.jpg)



Figure 4.6: While the coercive function on the left has a minimiser, it is easy to see that the non-coercive function on the right does not have a minimiser.


Remark 4.1.35. Coercivity is equivalent to its negated statement which is “if the function values $\{ E ( u _ { j } ) \} _ { j \in \mathbb { N } } \subset \mathbb { R }$ are bounded, so is the sequence $\{ u _ { j } \} _ { j \in \mathbb { N } } \subset \mathcal { X } ^ { \prime \prime }$ 

Although coercivity is not strictly speaking necessary, it is suficient that all minimising sequences are bounded. 

Lemma 4.1.36. Let $E \colon \mathcal { X }  \bar { \mathbb { R } }$ be a proper, coercive functional and bounded from below. Then the infimum in $\mathrm { f } _ { u \in \mathcal { X } } E ( u )$ exists in <sup>R</sup>, there are minimising sequences, i.e. $\{ u _ { j } \} _ { j \in \mathbb { N } } \subset$ with $E ( u _ { j } ) \to \operatorname* { i n f } _ { { u } \in \mathcal { X } } E ( u )$ , and all minimising sequences are bounded. 

Proof. As E is proper and bounded from below, there exists a $C _ { 1 } > 0$ such that we have $- \infty < - C _ { 1 } < \operatorname* { i n f } _ { u } E ( u ) < \infty$ which also guarantees the existence of a minimising sequence. Let $\{ u _ { j } \} _ { j \in \mathbb { N } }$ be any minimising sequence, i.e. $E ( u _ { j } )  \mathrm { i n f } _ { u } E ( u )$ . Then there exists a $j _ { 0 } \in \mathbb { N }$ such that for all $j > j _ { 0 }$ we have 

$$
E (u _ {j}) \leqslant \underbrace {\inf _ {u} E (u) + 1} _ {=: C _ {2}} <   \infty .
$$

With $C : = \operatorname* { m a x } \{ C _ { 1 } , C _ { 2 } \}$ we have that $| E ( u _ { j } ) | < C$ for all $j > j _ { 0 }$ and thus from the coercivity it follows that $\{ u _ { j } \} _ { j > j _ { 0 } }$ is bounded, see Remark 4.1.35. Including a finite number of elements does not change its boundedness which proves the assertion. □ 

A positive answer about the existence of minimisers is given by the following Theorem known as the “direct method” or “fundamental theorem of optimisation”. 

Theorem 4.1.37 (“Direct method”, David Hilbert, around 1900). Let be a Banach space and τ a topology (not necessarily the one induced by the norm) on such that bounded sequences have τ -convergent subsequences. Let $E \colon \mathcal { X }  \bar { \mathbb { R } }$ be proper, bounded from below, coercive and $\tau _ { \mathcal { X } ^ { - } } l . s . c$ . Then E has a minimiser. 

Proof. From Lemma 4.1.36 we know that inf $\overset { \cdot } { u } \in \mathcal { X } ^ { \textit { E } ( u ) }$ is finite, minimising sequences exist and that they are bounded. Let $\{ u _ { j } \} _ { j \in \mathbb { N } } \in \mathcal { X }$ be a minimising sequence. Thus, from the assumption on the topology τ<sub>X</sub> there exists a subsequence $\{ u _ { j _ { k } } \} _ { k \in \mathbb { N } }$ and $u ^ { * } \in \mathcal { X }$ with $u _ { j _ { k } } \stackrel { \tau _ { \mathcal { X } } } {  } u ^ { * }$ for $k  \infty$ . From the sequential lower semi-continuity of E we obtain 

$$
E (u ^ {*}) \leqslant \operatorname * {l i m i n f} _ {k \to \infty} E (u _ {j _ {k}}) = \lim _ {j \to \infty} E (u _ {j}) = \inf _ {u \in \mathcal {X}} E (u) <   \infty ,
$$

which shows that $E ( u ^ { * } ) < \infty$ and $E ( u ^ { * } ) \leqslant E ( u )$ for all $u \in { \mathcal { X } } ;$ thus $u ^ { * }$ minimises $E . \quad \boxed { }$ 

The above theorem is very general but its conditions are hard to verify but the situation is a easier in reflexive Banach spaces (thus also in Hilbert spaces). 

Corollary 4.1.38. Let be a reflexive Banach space and $E \colon { \mathcal { X } } $ <sup>R¯</sup> be a functional which is proper, bounded from below, coercive and l.s.c. with respect to the weak topology. Then there exists a minimiser of E. 

Proof. The statement follows from the direct method, Theorem 4.1.37, as in reflexive Banach spaces bounded sequences have weakly convergent subsequences, see Theorem 4.1.2. □ 

Remark 4.1.39. For convex functionals, the situation is even easier. It can be shown that a convex function is l.s.c. with respect to the weak topology if and only if it is l.s.c. with respect to the strong topology (see e.g. [19, Corollary 2.2., p. 11] or [8, p. 149] for Hilbert spaces). 

## 4.1. BACKGROUND

Remark 4.1.40. It is easy to see that the key ingredient for the existence of minimisers is that bounded sequences have a convergent subsequence. In variational regularisation this is usually ensured by an appropriate choice of the regularisation functional. 

## Uniqueness

Theorem 4.1.41. Assume that the functional $E \colon \mathcal { X }  \bar { \mathbb { R } }$ has at least one minimiser and is strictly convex. Then the minimiser is unique. 

Proof. Let $u , v$ be two minimisers of E and assume that they are diferent, i.e. u $\neq v$ . Then it follows from the minimising properties of u and v as well as the strict convexity of E that 

$$
E (u) \leqslant E (\frac {1}{2} u + \frac {1}{2} v) <   \frac {1}{2} E (u) + \frac {1}{2} \underbrace {E (v)} _ {\leqslant E (u)} \leqslant E (u)
$$

which is a contradiction. Thus, $u = v$ and the assertion is proven. 

Example 4.1.42. Convex (but not strictly convex) functions may have have more than one minimiser, examples include constant and trapezoidal functions, see Figure 4.7. On the other hand, convex (and even non-convex) functions may have a unique minimiser, see Figure 4.7. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/c9645fdd2018c2e92662a6f6b3ee80265297b67f4b74318449c6966fdd22ece0.jpg)



Figure 4.7: a) Convex functions may not have a unique minimiser. b) Neither strict convexity nor convexity is necessary for the uniqueness of a minimiser.


## 4.1.4 Duality in convex optimisation

Consider the following optimisation problem 

$$
\inf _ {u \in \mathcal {X}} E (A u) + F (u),\tag{P}
$$

where $E \colon \mathcal { V }  \bar { \mathbb { R } }$ and $F \colon \mathcal { X }  \bar { \mathbb { R } }$ are proper, convex and lower semicontinuous functions and $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ is a linear bounded operator. 

Since E is convex and lower semicontinuous, it can be written as the convex conjugate of its conjugate $E ^ { * }$ 

$$
E (y) = \sup _ {\eta \in \mathcal {Y} ^ {*}} \langle \eta , y \rangle - E ^ {*} (\eta) \quad y \in \mathcal {Y}.
$$

Hence, we can rewrite ( ) as follows 

$$
\inf _ {u \in \mathcal {X}} \sup _ {\eta \in \mathcal {Y} ^ {*}} \langle \eta , A u \rangle - E ^ {*} (\eta) + F (u).\tag{S}
$$

This problem is referred to as the saddle point problem, whereas ( ) is referred to as the primal problem. Since inf sup $\geqslant$ sup inf always holds, we get that 

$$
\begin{array}{r c l} \inf _ {u \in \mathcal {X}} E (A u) + F (u) & \geqslant & \sup _ {\eta \in \mathcal {Y} ^ {*}} \inf _ {u \in \mathcal {X}} \langle \eta , A u \rangle - E ^ {*} (\eta) + F (u) \\ & = & \sup _ {\eta \in \mathcal {Y} ^ {*}} \inf _ {u \in \mathcal {X}} \langle A ^ {*} \eta , u \rangle - E ^ {*} (\eta) + F (u) \\ & = & \sup _ {\eta \in \mathcal {Y} ^ {*}} \left\{- E ^ {*} (\eta) - \sup _ {u \in \mathcal {X}} [ \langle - A ^ {*} \eta , u \rangle - F (u) ] \right\} \\ & = & \sup _ {\eta \in \mathcal {Y} ^ {*}} - E ^ {*} (\eta) - F ^ {*} (- A ^ {*} \eta). \end{array}
$$

The last problem 

$$
\sup _ {\eta \in \mathcal {Y} ^ {*}} - E ^ {*} (\eta) - F ^ {*} (- A ^ {*} \eta)\tag{D}
$$

is called the dual problem. The fact that the optimal value of the primal is always less or equal to the optimal value of the dual problem is referred to as weak duality and the diference between these two optimal values is referred to as the duality gap. Whenever the two optimal values are in fact equal, one speaks of strong duality. Suficient conditions for strong duality are given by 

Theorem 4.1.43 ([19, Ch.III Thm 4.1 and Rem. 4.2]). Suppose that 

(i) the function $E ( A u ) + F ( u ) \colon \mathcal { X } \to \bar { \mathbb { R } }$ is proper, convex, l.s.c. and coercive; 

(ii) <sub>∃</sub>u<sub>0 ∈</sub> <sub>X</sub> s.t. F (u<sub>0</sub>) < +<sub>∞</sub>, E(Au<sub>0</sub>) < +<sub>∞</sub> and $E ( y )$ is continuous at $y = A u _ { 0 }$ Then 

(i) The dual problem ( ) has at least one solution $\widehat { \eta } ;$ 

(ii) There is no duality gap between ( ) and ( ), i.e. strong duality holds; 

(iii) If ( ) has an optimal solution ${ \widehat { u } } ,$ then the following optimality conditions hold 

$$
- A ^ {*} \widehat {\eta} \in \partial F (\widehat {u}), \quad \widehat {\eta} \in \partial E (A \widehat {u}).
$$

Note that existence of a primal solution is not guaranteed by this theorem. 

## 4.2 Well-posedness and Regularisation Properties

Our goal is to study the properties of optimisation problem (4.1) as a convergent regularisation for the ill-posed problem 

$$
A u = f,\tag{4.4}
$$

where $A \colon \mathcal { X }  \mathcal { Y }$ is a linear bounded operator, is a Banach space and is the dual of a separable Banach space. In particular, we will ask questions of existence of minimisers (well-posedness of the regularised problem) and parameter choice rules that guarantee the convergence of the minimisers to an appropriate generalised solution of (4.4) for diferent choices of the regularisation functional. To this end, we need to extend the definition of a minimal-norm solution (Def. 2.1.1) to an arbitrary regularisation term. 

Definition 4.2.1 ( -minimising solutions). Let $u _ { \mathcal { I } } ^ { \dagger }$ be a least squares solution, i.e. 

$$
\| A u _ {\mathcal {J}} ^ {\dagger} - f \| y = \inf \{\| A v - f \| y, \quad v \in \mathcal {X} \}
$$

and 

$\mathcal { I } ( u _ { \mathcal { T } } ^ { \dagger } ) \leqslant \mathcal { I } ( \widetilde { u } )$ for all least squares solutions $\widetilde { u } .$ 

Then $u _ { \mathcal { I } } ^ { \dagger }$ is called a  -minimising solution of (4.4). 

We will assume that there exists a least-squares solution with a finite value of ${ \mathcal { I } } _ { : }$ , i.e. there exists at least one element u such that $\| A u - f \| _ { \mathcal { V } } = \operatorname* { i n f } \{ \| A v - f \| _ { \mathcal { V } } , v \in \mathcal { X } \}$ and $\mathcal { I } ( u ) < + \infty$ 

Remark 4.2.2. A <sub>J</sub> -minimising solution may not exist and if it does, it may be non-unique. We will later see conditions, under which a -minimising solution exists. Non-uniqueness, however, is common with popular choices of $\mathcal { I }$ . In this case we need to define a selection operator that will select a single element from all the -minimising solutions (see [9]). We will not explicitly mention this, stating all results for just a -minimising solution. 

We will need the following 

Lemma 4.2.3. Let $\begin{array} { r } { \mathcal { I } ( u ) = \sum _ { i = 1 } ^ { n } \mathcal { T } _ { i } ( u ) } \end{array}$ , where each $\mathcal { I } _ { i } ( u )$ is convex and $p _ { i } – h o m o g e n e o u s$ $( p _ { i } > 0 )$ , that is, 

$$
\mathcal {J} _ {i} (\lambda u) = | \lambda | ^ {p _ {i}} \mathcal {J} _ {i} (u) \quad \forall u \in \mathcal {X}, \lambda \in \mathbb {R}.
$$

The the set 

$$
\mathcal {N} (\mathcal {J}) := \{u \in \mathcal {X}: \mathcal {J} (u) = 0 \}
$$

is a linear subspace of  . 

Proof. First of all, we note that $\mathcal { I } _ { i } ( u ) \geq 0$ for all $u \in \mathcal X$ . Indeed, we have 

$$
0 = \mathcal {J} _ {i} (0) = \mathcal {J} _ {i} \left(\frac {1}{2} u - \frac {1}{2} u\right) \leqslant \frac {1}{2} \mathcal {J} _ {i} (u) + \frac {1}{2} \mathcal {J} _ {i} (- u) = \mathcal {J} _ {i} (u).
$$

Now let $u , v \in \mathcal { N } ( \mathcal { I } )$ be arbitrary. Then $\mathcal { I } _ { i } ( u ) = \mathcal { I } _ { i } ( v ) = 0$ for all $i = 1 , . . . , n$ , hence for any $\lambda \in \mathbb { R }$ 

$$
\begin{array}{r c l} 0 \leqslant \mathcal {J} _ {i} (\lambda u + v) & = & 2 ^ {p _ {i}} \mathcal {J} _ {i} \left(\frac {\lambda u}{2} + \frac {v}{2}\right) \leqslant 2 ^ {p _ {i}} \left(\frac {1}{2} \mathcal {J} _ {i} \left(\frac {\lambda u}{2}\right) + \frac {1}{2} \mathcal {J} _ {i} \left(\frac {v}{2}\right)\right) \\ & = & \frac {1}{2} \mathcal {J} _ {i} (\lambda u) + \frac {1}{2} \mathcal {J} _ {i} (v) = \frac {| \lambda | ^ {p _ {i}}}{2} \mathcal {J} _ {i} (u) + \frac {1}{2} \mathcal {J} _ {i} (v) = 0. \end{array}
$$

Therefore, $\mathcal { T } _ { i } ( \lambda u + v ) = 0$ for all i and hence $\mathcal { I } ( \lambda u + v ) = 0$ 

Lemma 4.2.4. Let assumptions of Lemma 4.2.3 be satisfied. Suppose that $u \in \mathcal X$ and $v \in \mathcal { N } ( \mathcal { I } )$ . Then $\mathcal { I } ( u + v ) = \mathcal { I } ( u )$ 

Proof. Left as exercise. 

If dim $\mathcal { N } ( \mathcal { I } ) < \infty$ , the subspace $\mathcal { N } ( \mathcal { I } )$ is complemented in [4, Thm. 5.89], i.e. there exists a closed subspace $\mathcal { X } _ { 0 } \subset \mathcal { X }$ such that $\mathcal { X } _ { 0 } \cap \mathcal { N } ( \mathcal { I } ) = \{ 0 \}$ and 

$$
\mathcal {X} = \mathcal {X} _ {0} \oplus \mathcal {N} (\mathcal {J}).\tag{4.5}
$$

We will use this to establish coercivity of the functional (4.1). 

Lemma 4.2.5. Suppose that the regularisation functional $\mathcal { I } \colon \mathcal { X } \to \bar { \mathbb { R } } _ { + }$ is proper, convex and satisfies conditions of Lemma (4.2.3) and let $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ be a bounded linear operator. Suppose also that 

(i) dim $\mathcal { N } ( \mathcal { I } ) < \infty$ and $\mathcal { I }$ is coercive on $\mathcal { X } _ { 0 }$ , where $\mathcal { X } _ { 0 }$ is such that $\mathcal { X } = \mathcal { X } _ { 0 } \oplus \mathcal { N } ( \mathcal { I } )$ ; 

(ii) the kernels of A and <sub>J</sub> have a trivial intersection, i.e. $\mathcal { N } ( A ) \cap \mathcal { N } ( \mathcal { I } ) = \{ 0 \}$ 

Then the function 

$$
\Phi_ {\alpha} (u) := \frac {1}{2} \| A u - f \| _ {\mathcal {Y}} ^ {2} + \alpha \mathcal {J} (u)
$$

is coercive on for any $\alpha > 0$ 

Proof. Let $\{ u _ { j } \} _ { j \in \mathbb { N } }$ be a sequence in <sub>X</sub> . Due to (4.5), there exists a unique decomposition 

$$
u _ {j} = u _ {j} ^ {0} + u _ {j} ^ {\mathcal {N}}, \quad u _ {j} ^ {0} \in \mathcal {X} _ {0}, u _ {j} ^ {\mathcal {N}} \in \mathcal {N} (\mathcal {J}).
$$

Let $\Phi _ { \alpha } ( u _ { j } ) \leqslant C$ for all $j \in \mathbb N .$ . Then $\mathcal { I } ( u _ { j } ) \leqslant C$ and 

$$
\mathcal {J} \left(u _ {j} ^ {0}\right) = \mathcal {J} \left(u _ {j} ^ {0} + u _ {j} ^ {\mathcal {N}}\right) = \mathcal {J} (u _ {j}) \leqslant C.
$$

Since $\mathcal { I }$ is coercive on $\mathcal { X } _ { 0 }$ , we get that $\| u _ { j } ^ { 0 } \| \leqslant C ^ { \prime }$ . Now, define 

$$
\widetilde {A} \colon \mathcal {N} (\mathcal {J}) \to A \mathcal {N} (\mathcal {J}), \quad \widetilde {A} = A | _ {\mathcal {N} (\mathcal {J})}.
$$

That is, $\widetilde { A }$ is the restriction of A to $\mathcal { N } ( \mathcal { I } )$ . Clearly, $\widetilde { A }$ is surjective and by assumption (ii) it is also injective. Since $\mathcal { N } ( \mathcal { I } )$ (and, subsequently, $A \mathcal { N } ( \mathcal { I } ) )$ is finite-dimensional, $\widetilde { A } ^ { - 1 }$ exists and is bounded. Denote $\| \widetilde { A } ^ { - 1 } \| = : \widetilde { C }$ . Then 

$$
\| u _ {j} ^ {\mathcal {N}} \| = \| \widetilde {A} ^ {- 1} (\widetilde {A} u _ {j} ^ {\mathcal {N}}) \| \leqslant \widetilde {C} \| A u _ {j} ^ {\mathcal {N}} \| = \widetilde {C} \| A u _ {j} ^ {\mathcal {N}} + A u _ {j} ^ {0} - f - (A u _ {j} ^ {0} - f) \|
$$

$$
\leqslant \widetilde {C} \left\| A u _ {j} - f \| + \| A u _ {j} ^ {0} - f \|\right) \leqslant \widetilde {C} (C + \| A \| \| u _ {j} ^ {0} \| + \| f \|) \leqslant C ^ {\prime \prime}.
$$

Therefore, 

$$
\| u _ {j} \| = \| u _ {j} ^ {0} + u _ {j} ^ {\mathcal {N}} \| \leqslant \| u _ {j} ^ {0} \| + \| u _ {j} ^ {\mathcal {N}} \| \leqslant C ^ {\prime \prime \prime},
$$

which means that $\Phi _ { \alpha }$ is coercive. 

Now we are ready to establish the existence of a -minimising solution and a regularised solution for any $\alpha > 0$ 

Theorem 4.2.6. Let and be a Banach spaces and $\tau _ { \mathcal { X } }$ and $\tau _ { \mathcal { V } }$ some topologies (not necessarily induced by the norm) in and $\mathcal { V } _ { i }$ , respectively. Assume that 

(i) bounded sequences in have τ<sub>X</sub> -convergent subsequences; 

(ii) $\mathcal { I } \colon \mathcal { X } \to \bar { \mathbb { R } } _ { + }$ is proper, convex $\tau _ { \mathcal { X } ^ { - } } l . s . c .$ . and satisfies assumptions of Lemma $4 . 2 . 5 ;$ 

(iii) $A \colon \mathcal { X } \to \mathcal { Y }$ is $\tau _ { \mathcal { X } } \to \tau _ { \mathcal { Y } }$ continuous; 

(iv) $\| \cdot \| _ { \mathcal { V } }$ is τ<sub>Y</sub>-lower semicontinuous; 

Then 

(i’) there exists a <sub>J</sub> -minimising solution $u _ { \mathcal { I } } ^ { \dagger }$ of (4.4); 

(ii’) for any fixed $\alpha > 0$ and $f \in \mathcal { V }$ there exists a minimiser 

$$
u ^ {\alpha} \in \underset {u \in \mathcal {X}} {\arg \min} \frac {1}{2} \| A u - f \| _ {\mathcal {Y}} ^ {2} + \alpha \mathcal {J} (u).\tag{4.6}
$$

Proof. (i) Let <sup>L</sup> be the set of least-squares solutions of (4.4). Then <sup>L</sup> can written as follows 

$$
\mathbb {L} = \{u \in \mathcal {X} \colon \| A u - f \| _ {\mathcal {Y}} \leqslant \mu \},
$$

where $\mu : = \operatorname* { i n f } \{ \| A v - f \| _ { \mathcal { V } } \colon v \in \mathcal { X } \}$ . Since A is $\tau _ { \mathcal { X } } \to \tau _ { \mathcal { Y } }$ continuous and $\| \cdot \| _ { \mathcal { V } }$ is τ<sub>Y</sub> -l.s.c., <sup>L</sup> is τ<sub>X</sub> -closed. 

Consider the following problem 

$$
\inf _ {u \in \mathbb {L}} \mathcal {J} (u) = \inf _ {u \in \mathcal {X}} \mathcal {J} (u) + \chi_ {\mathbb {L}} (u).\tag{4.7}
$$

By the assumption that we made in the beginning of this section, this problem is feasible, i.e. there exists $u \in \mathbb { L }$ with $\mathcal { I } ( u ) < \infty$ . The objective function in (4.7) is bounded from below. Using similar arguments as in Lemma 4.2.5, we conclude that it is also coercive. Since <sup>L</sup> is $\tau _ { \mathcal { X } }$ -closed, χ<sup>L</sup> is $\tau _ { \mathcal { X } ^ { - 1 . \mathrm { S . C . } } }$ . By assumption ii, is also τ<sub>X</sub> -l.s.c. So, (4.7) satisfies the assumptions of the direct method (Theorem 4.1.37) and hence a minimiser exists. 

(ii) From Lemma 4.2.5 we know that the objective function $\Phi _ { \alpha }$ in (4.6) is coercive. It is also bounded from below. Since $\mathcal { I }$ is τ<sub>X</sub> -l.s.c., A is $\tau _ { \mathcal { X } } \to \tau _ { \mathcal { Y } }$ continuous and $\| \cdot \| _ { \mathcal { V } }$ is τ -l.s.c., we get that $\Phi _ { \alpha }$ is τ -l.s.c. Using the direct method, we conclude that (4.6) has a minimiser. 

Now we study the behaviour of the minimiser of (4.6) with $f = f _ { \delta }$ (perturbed measurement) as $\delta  0$ when $\alpha = \alpha ( \delta )$ is chosen according to an appropriate a priori parameter choice rule. For simplicity, we will do this in the case when in $ { \mathbb { f } }  { \left\{ \| A v - f \| _ { \mathcal { V } } : v \in \mathcal { X } \right\} } = 0$ 2 i.e. least-squares solutions are actually solutions of (4.4). 

Theorem 4.2.7. Let the assumptions of Theorem 4.2.6 hold and suppose that inf $\{ \Vert { A v - }$ $f \| _ { \mathcal { V } } \colon v \in \mathcal { X } \} = 0$ . Let $\alpha = \alpha ( \delta )$ be such that 

$$
\lim _ {\delta \to 0} \alpha (\delta) = 0 \quad a n d \quad \operatorname * {l i m s u p} _ {\delta \to 0} \frac {\delta^ {2}}{\alpha (\delta)} = 0.
$$

Then $u _ { \delta } : = u _ { \delta } ^ { \alpha ( \delta ) } \stackrel { \tau _ { \mathcal { X } } } {  } u _ { \mathcal { I } } ^ { \dagger }$ as $\delta  0$ (possibly, along a subsequence) and $\mathcal { I } ( u _ { \delta } )  \mathcal { I } ( u _ { \mathcal { I } } ^ { \dagger } )$ ， where $u _ { \mathcal { I } } ^ { \dagger }$ is a -minimising solution. 

Proof. Let u be any -minimising solution (which exists by Theorem 4.2.6). Since $u _ { \delta }$ solves (4.6) with $\alpha = \alpha ( \delta )$ , we get that 

$$
\begin{array}{r c l} \frac {1}{2} \| A u _ {\delta} - f _ {\delta} \| _ {\mathcal {Y}} ^ {2} + \alpha (\delta) \mathcal {J} (u _ {\delta}) & \leqslant & \frac {1}{2} \| A u _ {0} - f _ {\delta} \| _ {\mathcal {Y}} ^ {2} + \alpha (\delta) \mathcal {J} (u _ {0}) \\ & \leqslant & \frac {\delta^ {2}}{2} + \alpha (\delta) \mathcal {J} (u _ {0}). \end{array}\tag{4.8}
$$

Therefore, we have the following two estimates 

$$
\mathcal {J} (u _ {\delta}) \leqslant \frac {\delta^ {2}}{2 \alpha (\delta)} + \mathcal {J} (u _ {0}) \leqslant C,\tag{4.9a}
$$

$$
\| A u _ {\delta} - f _ {\delta} \| y \leqslant \sqrt {\delta^ {2} + 2 \alpha (\delta) \mathcal {J} (u _ {0})} \leqslant C ^ {\prime},\tag{4.9b}
$$

The right-hand side in (4.9a) is bounded uniformly in $\delta ,$ because lim su $\mathrm { p } _ { \delta \to 0 } \delta ^ { 2 } / \alpha ( \delta ) = 0$ by assumption and $\mathcal { I } ( u _ { 0 } )$ is a constant independent of δ. The right-hand side in (4.9b) is bounded, because $\mathcal { I } ( u _ { 0 } )$ is a constant and $\delta , \alpha ( \delta )  0$ 

Therefore, both $\mathcal { I } ( u _ { \delta } )$ and $\| A u _ { \delta } - f _ { \delta } \| _ { \mathcal { V } }$ are uniformly bounded. Proceeding similarly to Lemma 4.2.5, we get that 

$$
\left\| u _ {\delta} \right\| \leqslant C
$$

for all $\delta .$ Now let $\delta _ { n } \downarrow 0$ be an arbitrary null sequence. Since $u _ { \delta _ { n } }$ is bounded, it contains a τ<sub>X</sub> -convergent subsequence (which we don’t relabel) 

$$
u _ {\delta_ {n}} \stackrel {\tau_ {\mathcal {X}}} {\rightarrow} u _ {\mathcal {J}} ^ {\dagger} \quad \mathrm{as} n \to \infty .
$$

We will show that $u _ { \mathcal { I } } ^ { \dagger }$ is a $\mathcal { T } \cdot$ -minimising solution. From (4.9b) we observe that 

$$
\operatorname * {l i m i n f} _ {n \to \infty} \| A u _ {\delta_ {n}} - f _ {\delta_ {n}} \| _ {\mathcal {Y}} \leqslant \operatorname * {l i m i n f} _ {n \to \infty} \sqrt {\delta_ {n} ^ {2} + 2 \alpha (\delta_ {n}) \mathcal {J} (u _ {0})} = 0.
$$

Since $A$ is $\tau _ { \mathcal { X } } \to \tau _ { \mathcal { Y } }$ continuous and $\| \cdot \| _ { \mathcal { V } }$ is $\tau _ { \mathcal { Y } ^ { - 1 . \mathrm { s . c . } } }$ , we get that 

$$
\| A u _ {\mathcal {J}} ^ {\dagger} - f \| y \leqslant \operatorname * {l i m i n f} _ {n \to \infty} \| A u _ {\delta_ {n}} - f \| y \leqslant \operatorname * {l i m i n f} _ {n \to \infty} (\| A u _ {\delta_ {n}} - f _ {\delta_ {n}} \| y + \| f - f _ {\delta_ {n}} \| y) = 0,
$$

which shows that $u _ { \mathcal { I } } ^ { \dagger }$ is a least-squares solution. Using the estimate (4.9a) and $\tau _ { \mathcal { X } }$ -lower semicontinuity of $\mathcal { I }$ , we obtain 

$$
\mathcal {J} (u _ {\mathcal {J}} ^ {\dagger}) \leqslant \operatorname * {l i m i n f} _ {n \to \infty} \mathcal {J} (u _ {\delta_ {n}}) \leqslant \operatorname * {l i m s u p} _ {n \to \infty} \mathcal {J} (u _ {\delta_ {n}}) \leqslant \operatorname * {l i m s u p} _ {n \to \infty} \frac {\delta^ {2}}{2 \alpha (\delta)} + \mathcal {J} (u _ {0}) = \mathcal {J} (u _ {0}).\tag{4.10}
$$

Since $u _ { 0 }$ was an arbitrary $\mathcal { I }$ -minimising solution and $\mathcal { I } ( u _ { \mathcal { I } } ^ { \dagger } ) \leqslant \mathcal { I } ( u _ { 0 } )$ , we conclude that $\mathcal { I } ( u _ { \mathcal { I } } ^ { \dagger } )$ is also a $\mathcal { I } .$ -minimising solution. Finally, since $\mathcal { I } ( u _ { \mathcal { T } } ^ { \dagger } ) ~ = ~ \mathcal { I } ( u _ { 0 } )$ , we conclude from (4.10) that 

$$
\liminf _ {n \to \infty} \mathcal {J} (u _ {\delta_ {n}}) = \limsup _ {n \to \infty} \mathcal {J} (u _ {\delta_ {n}}) = \lim _ {n \to \infty} \mathcal {J} (u _ {\delta_ {n}}) = \mathcal {J} (u _ {\mathcal {J}} ^ {\dagger}),
$$

which completes the proof. 

Remark 4.2.8. The theorem proves convergence of the regularised solutions in $\tau _ { \mathcal { X } }$ , which may difer from the strong topology. However, if $\mathcal { I }$ satisfies the Radon-Riesz property with respect to the topology $\tau _ { \mathcal { X } }$ , i.e. $u _ { j } \stackrel { \tau _ { \chi } } {  }$ u and $\mathcal { I } ( u _ { j } )  \mathcal { I } ( u )$ imply $\| u _ { j } - u \|  0$ , then we get convergence in the norm topology. An example of a functional satisfying the Radon-Riesz property is the norm in a Hilbert (or reflexive Banach) space with $\tau _ { \mathcal { X } }$ being the weak topology. 

## Examples of regularisers

Example 4.2.9. Let $\mathcal { X }$ be a Hilbert space and $\mathcal { I } ( u ) = \| u \| ^ { 2 }$ . The norm in a Hilbert space is weakly l.s.c. By Theorem 4.1.2 we know that (norm) bounded sequences have weakly convergent subsequences. Therefore, Assumption (ii) of Theorem 4.2.6 is satisfied with $\tau _ { \mathcal { X } }$ being the weak topology and we obtain weak convergence of the regularised solutions. However, since the norm in a Hilbert space has the Radon-Riesz property, we also get strong convergence. The same approach works in reflexive Banach spaces. 

A classical example is regularisation in Sobolev spaces such as the space $H ^ { 1 }$ of $L ^ { 2 }$ functions whose weak derivatives are also in $L ^ { 2 }$ . In the one-dimensional case, the space $H ^ { 1 }$ consists only of continuous functions (in higher dimensions it is true for Sobolev spaces with some other exponents), therefore, the regularised solutions will also be continuous. For this reason, the regulariser $\mathcal { I } ( u ) = \| u \| _ { H ^ { 1 } }$ is sometimes referred to as the smoothing functional. Whilst desirable in some applications, in imaging smooth reconstructions are usually not favourable, since images naturally contain edges and therefore are not continuous functions. To overcome this issue, other regularisers have been introduced that we will discuss later. 

Example 4.2.10 (`<sup>1</sup>-regularisation). Let $\mathcal { X } = \ell ^ { 2 }$ be space of all square summable sequences (i.e. such that $\begin{array} { r } { \| u \| _ { \ell ^ { 2 } } ^ { 2 } = \sum _ { i = 1 } ^ { \infty } u _ { i } ^ { 2 } < + \infty ) } \end{array}$ . For example, u can represent the coeficients of a function in a basis $( \mathrm { e . g . }$ , a Fourier basis or a wavelet basis). As a regularisation functional, let us use not the $\ell ^ { 2 } .$ -norm, but the $\ell ^ { 1 }$ -norm: 

$$
\mathcal {J} (u) = \| u \| _ {\ell^ {1}} = \sum_ {i = 1} ^ {\infty} | u _ {i} |.
$$

By Example 4.1.5 $\mathcal { I } ( \cdot )$ is weakly l.s.c. in $\ell ^ { 2 }$ . It is evident that $\ell ^ { q } \subset \ell ^ { p }$ and $\| \cdot \| _ { \ell ^ { p } } \leqslant \| \cdot \| _ { \ell ^ { q } }$ for $q \leqslant p$ . Therefore, $\mathcal { I } ( u ) \leqslant C$ implies that $\| \cdot \| _ { \ell ^ { 2 } } \leqslant C$ and, since $\ell ^ { 2 }$ is a Hilbert space and bounded sequences have weakly convergent subsequences, we conclude that the sublevel sets of $\mathcal { I } ( \cdot )$ are weakly sequentially compact in $\ell ^ { 2 }$ . Therefore, Assumption (ii) of Theorem 4.2.6 is satisfied with $\tau _ { \mathcal { X } }$ being the weak topology in $\ell ^ { 2 }$ . Hence, we get weak convergence of regularised solutions in $\ell ^ { 2 }$ 

The motivation for using the $\ell ^ { 1 } { \mathrm { - n o r m } }$ as the regulariser instead of the $\ell ^ { 2 } { \mathrm { - n o r m } }$ is as follows. If the forward operator is non-injective, the inverse problem has more than one solution and the solutions form an afine subspace. In the context of sequence spaces representing coeficients of the solution in a basis, it is sometimes beneficial to look for solutions that are sparse in the sense that they have finite support, i.e. $| \operatorname { s u p p } ( u ) | < \infty$ with $\mathrm { s u p p } ( u ) = \{ i \in \mathbb { N } | u _ { i } \neq 0 \}$ This allows explaining the signal with a finite (and often relatively small) number of basis functions and has widely ranging applications in, for instance, compressed sensing. A finite dimensional illustration of the sparsity of $\ell ^ { 1 } \cdot$ regularised solutions is given in Figure 4.8. The corresponding minimisation problem 

$$
\min _ {u \in \ell^ {2}} \left\{\frac {1}{2} \| A u - f \| _ {\ell^ {2}} ^ {2} + \alpha \| u \| _ {1} \right\}.\tag{4.11}
$$

is also called lasso in the statistical literature. 

Example 4.2.11 (Elastic net regularisation). The $\ell ^ { 1 }$ regulariser described in the previous example sometimes delivers undesirable results for problems where there are highly correlated features and we need to identify all relevant ones, e.g. microarray data analysis (analysis of genomic sequences), in that it tends to select only one feature out of the relevant group instead of all relevant features of the group, i.e. it fails to identify the group structure. Elastic net regularisation helps to overcome this issue. The elastic net regulariser $\mathcal { I } \colon \ell ^ { 2 } \to \bar { \mathbb { R } } _ { + }$ is defined as follows 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/f4f7fff2a31d6684057613935f66329e2be06666260a658a2bb24a38ee7dd091.jpg)



Figure 4.8: Non-injective operators have a non-trivial kernel such that the inverse problem has more than one solution and the solutions form an afine subspace visualised by the solid line. Diferent regularisation functionals favour diferent solutions. The circle and the diamond indicate all points with constant $\ell ^ { 2 } { \mathrm { - n o r m } }$ , respectively $\ell ^ { 1 } { \mathrm { - n o r m } }$ , and the minimal $\ell ^ { 2 } { \mathrm { - n o r m } }$ and $\ell ^ { 1 } .$ -norm solutions are the intersections of the line with the circle, respectively the diamond. As it can be seen, the minimal $\ell ^ { 2 } { \mathrm { - n o r m } }$ solution has two non-zero components while the minimal $\ell ^ { 1 } { \mathrm { - n o r m } }$ solution has only one non-zero component and thus is sparser.


$$
\mathcal {J} (u) := \alpha \| u \| _ {\ell^ {1}} + \beta \| u \| _ {\ell^ {2}} ^ {2},
$$

where $\alpha , \beta \ > \ 0$ are constants that balance the influence of the two terms. Since $\mathcal { I }$ is the sum of a 1-homogeneous term and a 2-homogeneous term, it satisfies assumptions of Lemma 4.2.3. 

## 4.3 Total Variation Regularisation

As pointed out in Example 4.2.9, in imaging we are interested in regularisers that allow for discontinuities while maintaining suficient regularity of the reconstructions. One popular choice is the so-called total variation regulariser [15]. 

Definition 4.3.1. Let $\Omega \subset \mathbb { R } ^ { n }$ be a bounded domain and $u \in L ^ { 1 } ( \Omega )$ . Let ${ \mathcal { D } } ( \Omega , \mathbb { R } ^ { n } )$ be the following set of vector-valued test functions (i.e. functions that map from Ω to <sup>Rn</sup>) 

$$
\mathcal {D} (\Omega , \mathbb {R} ^ {n}) := \Bigl \{\varphi \in C _ {0} ^ {\infty} (\Omega ; \mathbb {R} ^ {n})   \Big |   \sup _ {x \in \Omega} \| \varphi (x) \| _ {2} \leqslant 1 \Bigr \}.
$$

Total variation of $u \in L ^ { 1 } ( \Omega )$ is defined as follows 

$$
\operatorname{TV} (u) = \sup _ {\varphi \in \mathcal {D} (\Omega , \mathbb {R} ^ {n})} \int_ {\Omega} u (x) \operatorname{div} \varphi (x) d x.
$$

Remark 4.3.2. Definition 4.3.1 may seem a bit strange at the first glance, but we note that for a function $u \in L ^ { 1 } ( \Omega )$ whose weak derivative u exists and is also in $L ^ { 1 } ( \Omega , \mathbb { R } ^ { n } )$ (i.e. u belongs to the Sobolev space $W ^ { 1 , 1 } ( \Omega ) )$ we obtain, integrating by parts, that 

$$
\mathrm{TV} (u) = \sup _ {\varphi \in \mathcal {D} (\Omega , \mathbb {R} ^ {n})} \int_ {\Omega} - \left\langle \nabla u (x), \varphi (x) \right\rangle d x.
$$

By the Cauchy-Schwartz inequality we get that $| \left. \nabla u ( x ) , \varphi ( x ) \right. | \leqslant \| \nabla u ( x ) \| _ { 2 } \| \varphi ( x ) \| _ { 2 } ~ :$ 6 $\| \nabla u ( x ) \| _ { 2 }$ for a.e. $x \in \Omega$ . On the other hand, choosing $\varphi$ such that $\begin{array} { r } { \varphi ( x ) = - \frac { \nabla u ( x ) } { \| \nabla u ( x ) \| _ { 2 } } } \end{array}$ (technically, such $\varphi$ is not necessarily in ${ \mathcal { D } } ( \Omega , \mathbb { R } ^ { n } )$ , but we can approximate it with functions from ${ \mathcal { D } } ( \Omega , \mathbb { R } ^ { n } )$ , since any function in $W ^ { 1 , 1 } ( \Omega )$ can be approximated with smooth functions [2, Thm. 3.17]; we omit the technicalities here), we get that $- \left. \nabla u ( x ) , \varphi ( x ) \right. = \| \nabla u ( x ) \| _ { 2 }$ Therefore, the supremum over $\varphi \in { \mathcal { D } } ( \Omega , \mathbb { R } ^ { n } )$ is equal to 

$$
\operatorname{TV} (u) = \int_ {\Omega} \| \nabla u (x) \| _ {2} d x = \| \nabla u \| _ {L ^ {1}}.
$$

This shows that TV just penalises the the $L ^ { 1 }$ norm (of the pointwise 2-norm) of the gradient for any $u \in W ^ { 1 , 1 } ( \Omega )$ . However, we will see that the space of functions that have finite value of TV is larger than $W ^ { 1 , 1 } ( \Omega )$ and contains, for instance, discontinuous functions. 

Remark 4.3.3. It can be shown [13] that for any $u \in L ^ { 1 } ( \Omega )$ 

$$
\mathrm{TV} (u) = \| \nabla u \| _ {\mathfrak {M}},
$$

where $\nabla$ is the distributional gradient and $\| \cdot \| _ { \mathfrak { M } }$ is the Radon norm. That is, Total Variation extends the $L ^ { 1 }$ norm of the gradient for functions whose gradient is not a Lebesguemeasurable function. We will not use this interpretation of the Total Variation to simplify the presentation and refer the interested reader to [13] for details. 

Proposition 4.3.4. TV is a proper, convex and absolutely 1-homogeneous functional $L ^ { 1 } ( \Omega ) \to$ <sup>R¯</sup>. For any constant function c: $\mathbf { c } ( x ) \equiv c \in \mathbb { R }$ for all x and any $u \in L ^ { 1 } ( \Omega )$ 

$$
\operatorname{TV} (\mathbf {c}) = 0 \quad a n d \quad \operatorname{TV} (u + \mathbf {c}) = \operatorname{TV} (u).
$$

Proof. Left as exercise. 

Remark 4.3.5. It can be shown that the opposite implication holds, i.e. $\mathrm { T V } ( u ) = 0$ implies that u is constant. in other words, 

$$
\mathcal {N} (\mathrm{TV}) = \{u \in L ^ {1} (\Omega) \colon u = c o n s t \}.\tag{4.12}
$$

The easiest way to see this is using the Radon measure interpretation in Remark 4.3.3. Because time constraints, we will omit the proof. 

Example 4.3.6 (TV of an indicator function). Suppose $\mathcal { C } \subset \Omega \subset \mathbb { R } ^ { 2 }$ is a bounded domain with smooth boundary and $u ( \cdot ) = \mathbf { 1 } _ { \mathit { c } } ( \cdot )$ is its indicator function, i.e. 

$$
\mathbf {1} _ {\mathcal {C}} (x) = \left\{ \begin{array}{l l} 1 & x \in \mathcal {C} \\ 0 & x \in \mathcal {X} \setminus \mathcal {C} \end{array} \right..
$$

Then, using the divergence theorem, we $\mathrm { g e t }$ that for any test function $\varphi \in { \mathcal { D } } ( \Omega , \mathbb { R } ^ { n } )$ 

$$
\int_ {\Omega} u (x) \operatorname{div} \varphi (x) d x = \int_ {\mathcal {C}} \operatorname{div} \varphi (x) d x = \int_ {\partial \mathcal {C}} \left\langle \varphi (x), \mathbf {n} _ {\partial \mathcal {C}} (x) \right\rangle d l,
$$

where ∂ is the boundary of  and $\mathbf { n } _ { \partial \mathcal { C } } ( x )$ is the unit normal at x. Hence, 

$$
\begin{array}{r c l} \mathrm{TV} (u) & = & \sup _ {\varphi \in \mathcal {D} (\Omega , \mathbb {R} ^ {n})} \int_ {\Omega} u (x) \operatorname{div} \varphi (x) d x = \sup _ {\varphi \in \mathcal {D} (\Omega , \mathbb {R} ^ {n})} \int_ {\partial \mathcal {C}} \langle \varphi (x), \mathbf {n} _ {\partial \mathcal {C}} (x) \rangle d l \\ & \leqslant & \sup _ {\varphi \in \mathcal {D} (\Omega , \mathbb {R} ^ {n})} \int_ {\partial \mathcal {C}} \| \varphi (x) \| \| \mathbf {n} _ {\partial \mathcal {C}} (x) \| d l \leqslant \sup _ {\varphi \in \mathcal {D} (\Omega , \mathbb {R} ^ {n})} \int_ {\partial \mathcal {C}} d l = \mathrm{Per} _ {\mathcal {C}}, \end{array}
$$

where ${ \mathrm { P e r } } ( { \mathcal { C } } )$ is the perimeter of . On the other hand, since $\partial \mathcal { C }$ is smooth and $\| \mathbf { n } _ { \partial \mathcal { C } } ( x ) \| = 1$ for every x, n<sub>∂C</sub> can be extended to feasible vector field on Ω (i.e. one that is in $D ( \Omega , \mathbb { R } ^ { n } ) )$ ). Therefore, we get that 

$$
\mathrm{TV} (u) = \int_ {\partial \mathcal {C}} \langle \varphi (x), \mathbf {n} _ {\partial \mathcal {C}} (x) \rangle d l \geqslant \int_ {\partial \mathcal {C}} \| \mathbf {n} _ {\partial \mathcal {C}} (x) \| ^ {2} d l = \int_ {\partial \mathcal {C}} 1 \cdot d l = \mathrm{Per} (\mathcal {C}),
$$

Therefore, $\mathrm { T V } ( \mathbf { 1 } _ { \mathcal { C } } ) = \mathrm { P e r } _ { \mathcal { C } }$ for any domain with smooth boundary. This can be extended to domains with Lipschitz boundary by constructing a sequence of functions in $D ( \Omega , \mathbb { R } ^ { n } )$ that converge pointwise to $\mathbf { n } _ { \partial \mathcal { C } }$ 

We now study properties of functions that have a finite value of TV. 

Definition 4.3.7. The functions $u \in L ^ { 1 } ( \Omega )$ with a finite value of TV form a normed space called the space of functions of bounded variation (the BV-space) defined as follows 

$$
\operatorname{BV} (\Omega) := \left\{u \in L ^ {1} (\Omega) \Big | \| u \| _ {\operatorname{BV}} := \| u \| _ {L ^ {1}} + \operatorname{TV} (u) <   \infty \right\}.
$$

Remark 4.3.8. It can be shown that the space BV is the dual of a separable Banach space [13] and that $\mathrm { w e a k } \mathrm { - } ^ { \mathrm { * } }$ convergence $u _ { n } \to ^ { * }$ u in BV is equivalent to strong convergence $u _ { n } \to u$ in $L ^ { 1 }$ and convergence of the values $\mathrm { T V } ( u _ { n } )  \mathrm { T V } ( u )$ . The proof is outside the scope of these notes. 

We note that ${ \mathrm { B V } } ( \Omega )$ is compactly embedded in $L ^ { 1 } ( \Omega )$ . We start with the following classical result. 

Theorem 4.3.9 (Rellich-Kondrachov, [2, Thm. 6.3]). Let $\Omega \subset \mathbb { R } ^ { n }$ be a bounded Lipschitz domain (i.e. non-empty, open, connected and with Lipschitz boundary) and $p , m \in \mathbb { N }$ . Let 

$$
p ^ {*} := \left\{ \begin{array}{l l} \frac {n p}{n - m p} & \quad \text { if   } n > m p, \\ \infty & \quad \text { if   } n \leqslant m p. \end{array} \right.
$$

Then the embedding $W ^ { m , p } ( \Omega ) \to L ^ { q } ( \Omega )$ is continuous for all $1 \leqslant q \leqslant p ^ { * }$ and compact for all $1 \leqslant q < p ^ { * }$ 

Since functions from BV(Ω) can be approximated by functions in the Sobolev space $W ^ { 1 , 1 } ( \Omega ) \ [ 5 $ , Thm. 3.9], the Rellich-Kondrachov Theorem (with $p = 1 , m = 1 )$ gives us the following 

Corollary 4.3.10 ([5, Corrollary 3.49]). For any bounded Lipschitz domain $\Omega \subset \mathbb { R } ^ { n }$ , the embedding 

$$
\operatorname{BV} (\Omega) \subset \subset L ^ {1} (\Omega)
$$

is compact for any n $\geqslant 2$ and the embedding 

$$
\mathrm{BV} (\Omega) \hookrightarrow L ^ {2} (\Omega)
$$

is continuous for $n = 2$ 

Now we will show that TV is lower-semicontinuous in $L ^ { 1 }$ . 

Theorem 4.3.11. Let $\Omega \subset \mathbb { R } ^ { n }$ be open and bounded. Then the total variation is $l . s . c .$ . in $L ^ { 1 } ( \Omega )$ 

Proof. Let $\{ u _ { j } \} _ { j \in \mathbb { N } } \subset \mathrm { B V } ( \Omega )$ be a sequence converging in $L ^ { 1 } ( \Omega )$ with $u _ { j }  u \mathrm { ~ i n ~ } L ^ { 1 } ( \Omega )$ Then for any test function $\varphi \in { \mathcal { D } } ( \Omega , \mathbb { R } ^ { n } )$ we have that 

$$
\int_ {\Omega} u _ {j} (x) \operatorname{div} \varphi (x) d x \rightarrow \int_ {\Omega} u (x) \operatorname{div} \varphi (x) d x
$$

(strong convergence implies weak convergence) and therefore 

$$
\begin{array}{r c l} \mathrm{TV} (u) & = & \sup _ {\varphi \in \mathcal {D} (\Omega , \mathbb {R} ^ {n})} \int_ {\Omega} u (x)   \mathrm{div}   \varphi (x) d x \\ & = & \sup _ {\varphi \in \mathcal {D} (\Omega , \mathbb {R} ^ {n})} \lim _ {j \to \infty} \int_ {\Omega} u _ {j} (x)   \mathrm{div}   \varphi (x) d x \\ & \leqslant & \operatorname * {l i m i n f} _ {j \to \infty} \sup _ {\varphi \in \mathcal {D} (\Omega , \mathbb {R} ^ {n})} \int_ {\Omega} u _ {j} (x)   \mathrm{div}   \varphi (x) d x \\ & = & \operatorname * {l i m i n f} _ {j \to \infty} \mathrm{TV} (u _ {j}). \end{array}
$$

Here the lim inf appears when we swap the sup and the lim, because the limit of the suprema may not exist; however, the inequality holds for any subsequence and hence also for the lim inf. Note also that the left and right hand sides may not be finite. □ 

Since the null space of total variation (4.12) is nontrivial, TV cannot be coercive on $L ^ { 1 }$ However, the following result helps. 

Proposition 4.3.12 ([5, Remark 3.50]). Let $\Omega \subset \mathbb { R } ^ { n }$ be a bounded Lipschitz domain. Then there exists a constant $C > 0$ such that for all $u \in \mathrm { B V } ( \Omega )$ the Poincar´e inequality is satisfied 

$$
\left\| u - u _ {\Omega} \right\| _ {L ^ {1}} \leqslant C \operatorname{TV} (u),
$$

where $\begin{array} { r } { u _ { \Omega } : = \frac { 1 } { | \Omega | } \int _ { \Omega } u ( x ) d x } \end{array}$ is the mean-value of u over Ω. 

Corollary 4.3.13. It is often useful to consider a subspace $\mathrm { B V } _ { 0 } ( \Omega ) \subset \mathrm { B V } ( \Omega )$ of functions with zero mean, i.e. 

$$
\mathrm{BV} _ {0} (\Omega) := \{u \in \mathrm{BV} (\Omega): \int_ {\Omega} u (x) d x = 0 \}.\tag{4.13}
$$

Then for every function $u \in \mathrm { B V } _ { 0 } ( \Omega )$ we have that 

$$
\left\| u \right\| _ {L ^ {1}} \leqslant C \operatorname{TV} (u).
$$

Clearly, $\begin{array} { r } { \mathrm { B V } _ { 0 } \subset L _ { 0 } ^ { 1 } : = \{ u \in L ^ { 1 } \colon \int _ { \omega } u ( x ) d x = 0 \} } \end{array}$ in TV is coercive on this subspace. Since dim $( \mathcal { N } ( \mathrm { T V } ) ) = 1 < \infty$ , we have 

$$
L ^ {1} = L _ {0} ^ {1} \oplus \mathcal {N} (\mathrm{TV}).
$$

Combining all the above results we $\mathrm { g e t }$ 

Theorem 4.3.14. Let $\mathscr { X } = L ^ { 1 } ( \Omega )$ , where $\Omega \subset \mathbb { R } ^ { n }$ is bounded Lipschitz, and be a Banach space. Let $A \colon L ^ { 1 } \to \mathcal { Y }$ be a linear bounded operator such that $A { \bf 1 } \neq 0$ , where 1 is the constant-one function. Then minimisers of the following problem 

$$
\min _ {u \in L ^ {1} (\Omega)} \frac {1}{2} \| A u - f _ {\delta} \| _ {\mathcal {Y}} ^ {2} + \alpha (\delta) \operatorname{TV} (u)
$$

converge strongly in $L ^ { 1 }$ to a TV-minimising solution as $\delta  0 \ i f \alpha ( \delta )$ is chosen as required by Theorem 4.2.7. 

Proof. We have established all ingredients required for Theorem 4.2.7 to hold except that bounded sequences in $L ^ { 1 }$ may not have convergent subsequences $( L ^ { 1 }$ is not a dual space). However, the compact embedding from Corollary 4.3.10 guarantees that sequences with a bounded value of TV have subsequences that converge strongly in $L ^ { 1 }$ □ 

Remark 4.3.15. One can replace optimisation over $u \in L ^ { 1 }$ with optimisation over $u \in \mathrm { B V }$ which is the efective domain of the objective function. 

Total Variation is widely used in imaging applications [34]. The so-called Rudin–Osher– Fatemi (ROF) model for image denoising [31] consists in minimising the following functiona 

$$
\min _ {u \in \mathrm{BV} (\Omega)} \frac {1}{2} \| I u - f _ {\delta} \| _ {L ^ {2} (\Omega)} ^ {2} + \alpha \operatorname{TV} (u),\tag{4.14}
$$

where $\Omega \subset \mathbb { R } ^ { 2 }$ . In this case, the forward operator I is the embedding operator $\mathrm { B V } ( \Omega ) $ $L ^ { 2 } ( \Omega )$ , which is continuous for two-dimensional domains (see Corollary 4.3.10). Clearly, $A \mathbf { 1 } \neq 0$ is satisfied. More generally, one considers the following optimisation problem 

$$
\min _ {u \in \mathrm{BV} (\Omega)} \| A u - f _ {\delta} \| _ {2} ^ {2} + \alpha   \mathrm{TV} (u),\tag{4.15}
$$

where A : $\mathrm { B V } ( \Omega ) \to L ^ { 2 } ( \Omega )$ is such that $A \mathbf { 1 } \neq 0$ 

## Chapter 5

## Convex Duality

In Chapter 4 we have established convergence of a regularised solution $u _ { \delta }$ to a -minimising solution $u _ { \mathcal { I } } ^ { \dagger }$ as $\delta \to 0$ . However, we didn’t get any results on the speed of this convergence, which is referred to as the convergence rate. 

In modern regularisation methods, convergence rates are usually studied using Bregman distances associated with the (convex) regularisation functional $\mathcal { I }$ . Recall that for a convex functional $\mathcal { I } , u , v \in \mathcal { X }$ such that $\mathcal { I } ( v ) < \infty$ and $q \in \partial \mathcal { I } ( v )$ , the (generalised) Bregman distance is given by the following expression (cf. Def. 4.1.24) 

$$
D _ {\mathcal {J}} ^ {q} (u, v) = \mathcal {J} (u) - \mathcal {J} (v) - \left\langle q, u - v \right\rangle .
$$

Also widely used is the symmetric Bregman distance (cf. Def. 4.1.26) given by the following expression (here $p \in \partial \mathcal { I } ( u ) )$ 

$$
D _ {\mathcal {J}} ^ {s y m m} (u, v) = D _ {\mathcal {J}} ^ {q} (u, v) + D _ {\mathcal {J}} ^ {p} (v, u) = \left\langle p - q, u - v \right\rangle .
$$

Bregman distances appear to be a natural distance measure between a regularised solution $u _ { \delta }$ and a -minimising solution $u _ { \mathcal { I } } ^ { \dagger }$ . For instance, for classical Hilbert space regularisation with $\begin{array} { r } { \mathcal { I } ( u ) = \frac { 1 } { 2 } \| u \| _ { \mathcal { X } } ^ { 2 } } \end{array}$ , the subgradient at $u _ { \mathcal { I } } ^ { \dagger }$ is $p _ { u _ { \mathcal { I } } ^ { \dagger } } = u _ { \mathcal { I } } ^ { \dagger }$ (since $\mathcal { I }$ is diferentiable) and we get the following expression 

$$
\begin{array}{r l}&D _ {\mathcal {J}} ^ {u _ {\mathcal {J}} ^ {\dagger}} (u _ {\delta}, u _ {\mathcal {J}} ^ {\dagger}) = \frac {1}{2} \| u _ {\delta} \| _ {\mathcal {X}} ^ {2} - \frac {1}{2} \| u _ {\mathcal {J}} ^ {\dagger} \| _ {\mathcal {X}} ^ {2} - \left<   u _ {\mathcal {J}} ^ {\dagger}, u _ {\delta} - u _ {\mathcal {J}} ^ {\dagger} \right>\\&\qquad = \frac {1}{2} (\| u _ {\delta} \| _ {\mathcal {X}} ^ {2} - 2 \left<   u _ {\mathcal {J}} ^ {\dagger}, u _ {\delta} \right> + \| u _ {\mathcal {J}} ^ {\dagger} \| _ {\mathcal {X}} ^ {2}) = \frac {1}{2} \| u _ {\delta} - u _ {\mathcal {J}} ^ {\dagger} \| _ {\mathcal {X}} ^ {2},\end{array}
$$

which happens to coincide with the symmetric Bregman distance. Therefore, in the classical $L ^ { 2 } \mathrm { - c a s e }$ , the Bregman distance just measures the $L ^ { 2 } \mathrm { - d i s t a n c e }$ between a regularised solution and a $\mathcal { T } \mathrm { - m i n i m i s i n g }$ solution. As we have seen in an example sheet, subgradients of absolutely one-homogeneous functional carry structural information about the solution such as locations of non-zero components of a vector $u _ { \mathcal { T } } ^ { \dagger } \in \ell ^ { 1 }$ 

We are looking for a convergence rate of the following form 

$$
D _ {\mathcal {J}} ^ {s y m m} (u _ {\delta}, u _ {\mathcal {J}} ^ {\dagger}) \leqslant \psi (\delta),
$$

where $\psi : \mathbb { R } _ { + } \to \mathbb { R } _ { + }$ is a known function of δ such that $\psi ( \delta ) \to 0 \mathrm { ~ a s ~ } \delta \to 0$ 

## 5.1 Dual Problem

Recall that $u _ { \delta }$ solves the following problem 

$$
\min _ {u \in \mathcal {X}} \frac {1}{2} \| A u - f _ {\delta} \| _ {\mathcal {Y}} ^ {2} + \alpha \mathcal {J} (u).\tag{5.1}
$$

with an appropriately chosen $\alpha = \alpha ( \delta )$ , where and $\mathcal { V }$ are Banach spaces, $A \in \mathcal { L } ( \mathcal { X } , \mathcal { Y } )$ and $E \colon \mathcal { V }  \bar { \mathbb { R } }$ and $\mathcal { I } \colon \mathcal { X } $ <sup>R¯</sup> is proper, convex and l.s.c. and satisfies Assumptions of Theorem 4.2.6. For simplicity of presentation, we will also assume throughout this chapter that $\mathcal { I }$ is absolutely one-homogeneous and that inf $ { \left\{ \left\| A v - f \right\| : v \in \mathcal { X } \right\} } = 0$ , i.e. $A u _ { \mathcal { T } } ^ { \dagger } = f$ for any $\mathcal { T } \mathrm { \mathrm { \Omega } }$ -minimising solution. 

To apply the results of Section 4.1.4 to (5.1), we take (in the notation of Section 4.1.4) 

$$
E (y) := \frac {1}{2} \| y - f \| _ {\mathcal {Y}} ^ {2}, \quad F (u) := \alpha \mathcal {J} (u).
$$

Lemma 5.1.1. Let X be a Banach space with norm $\| \cdot \| _ { X }$ and let $\| \cdot \| _ { X }$ be the norm in the dual space of X. Let $\textstyle \varphi ( x ) : = { \frac { 1 } { 2 } } \| x \| _ { X } ^ { 2 }$ . Then the convex conjugate of $\varphi$ is 

$$
\varphi^ {*} (\xi) = \frac {1}{2} \| \xi \| _ {X ^ {*}} ^ {2}, \quad \xi \in X ^ {*}.
$$

Proof. First, we note that 

$$
\varphi^ {*} (\xi) = \sup _ {x \in X} \langle \xi , x \rangle - \frac {1}{2} \| x \| _ {X} ^ {2} \leqslant \sup _ {x \in X} \| x \| _ {X} \| \xi \| _ {X ^ {*}} - \frac {1}{2} \| x \| _ {X} ^ {2}.
$$

The function on the right-hand side is a parabola in the scalar variable $\| x \| _ { X }$ and its maximum is $\frac { 1 } { 2 } \| \xi \| _ { X ^ { * } } ^ { 2 }$ . Now, fix $\xi \in X ^ { * }$ . We have that 

$$
\| \xi \|_{X^{*}} = \sup_{\substack{x\in X\\ \| x\| = 1}}\langle \xi ,x\rangle = \sup_{\substack{x\in X\\ \| x\| = \| \xi \|}}\frac{\langle\xi,x\rangle}{\|\xi\|}.
$$

Let $x _ { n } ^ { \xi } \in X$ be a maximising sequence (that is, $\| x _ { n } ^ { \xi } \| = \| \xi \|$ and $\langle \xi , x _ { n } ^ { \xi } \rangle \to \| \xi \| ^ { 2 } )$ . Then 

$$
\varphi^ {*} (\xi) = \sup _ {x \in X} \langle \xi , x \rangle - \frac {1}{2} \| x \| _ {X} ^ {2} \geqslant \operatorname * {l i m s u p} _ {n \to \infty} \left(\langle \xi , x _ {n} ^ {\xi} \rangle - \frac {1}{2} \| x _ {n} ^ {\xi} \| _ {X} ^ {2}\right) = \| \xi \| ^ {2} - \frac {1}{2} \| \xi \| ^ {2} = \frac {1}{2} \| \xi \| ^ {2}.
$$

The inequality here is due to the fact that the lim sup is a supremum over a smaller set than the whole X. Hence, we have that $\begin{array} { r } { \frac 1 2 \| \xi \| ^ { 2 } \leqslant \varphi ^ { * } ( \xi ) \leqslant \frac { 1 } { 2 } \| \xi \| ^ { 2 } } \end{array}$ and the proof is complete. 

Corollary 5.1.2. Theorem 4.1.23 implies that for any $x \in X$ and any $\xi \in \partial \varphi ( x )$ it holds 

$$
\frac {1}{2} \| x \| _ {X} ^ {2} + \frac {1}{2} \| \xi \| _ {X ^ {*}} ^ {2} = \langle \xi , x \rangle .
$$

Using the Cauchy-Schwarz inequality on the right-hand side and rearranging terms, we get that $( \| x \| _ { X } - \| \xi \| _ { X ^ { * } } ) ^ { 2 } = 0$ and hence 

$$
\| \xi \| _ {X ^ {*}} = \| x \| _ {X}.
$$

Now, for E and F as defined above, we $\mathrm { g e t }$ 

$$
\begin{array}{r c l} E ^ {*} (\eta) & = & \sup _ {y \in \mathcal {Y}} \langle \eta , y \rangle - \frac {1}{2} \| y - f \| _ {\mathcal {Y}} ^ {2} = \langle \eta , f \rangle - \sup _ {z \in \mathcal {Y}} \left(\langle \eta , z \rangle - \frac {1}{2} \| z \| _ {\mathcal {Y}} ^ {2}\right) = \langle \eta , f \rangle + \frac {1}{2} \| \eta \| _ {\mathcal {Y} ^ {*}}, \\ F ^ {*} (p) & = & \chi_ {\partial \mathcal {J} (0)} \left(\frac {p}{\alpha}\right), \end{array}
$$

where the second equality holds since $F$ is absolutely one-homogeneous. Hence, the dual problem of (5.1) is given by 

$$
\sup _ {\eta \in \mathcal {Y} ^ {*}} - \langle \eta , f \rangle - \frac {1}{2} \| \eta \| _ {\mathcal {Y} ^ {*}} ^ {2} - \chi_ {\partial \mathcal {J} (0)} \left(\frac {- A ^ {*} \eta}{\alpha}\right).
$$

Denote $\textstyle \mu : = - { \frac { \eta } { \alpha } } \in \mathcal { V } ^ { * }$ . Sinc $\mathscr { z } - \chi _ { \partial \mathcal { I } ( 0 ) } = - \infty$ outside $\partial \mathcal { I } ( 0 )$ , we get the following equivalent problem 

$$
\sup_{\substack{\mu \in \mathcal{Y}^{*}\\ A^{*}\mu \in \partial \mathcal{J}(0)}}\alpha \left(\langle \mu ,f\rangle -\frac{\alpha}{2}\| \mu \|_{\mathcal{Y}}^{2}\right).\tag{5.2}
$$

Let us check if Assumptions of Theorem 4.1.43 are satisfied. Condition (i) (coercivity) is guaranteed by Lemma 4.2.5. Condition (ii) (continuity of E) is satisfied at $u _ { 0 } = 0$ Therefore, for any $\delta > 0$ there exists a solution $\mu _ { \delta }$ of the dual problem (5.2). 

Existence of a primal solution $u _ { \delta }$ is guaranteed by Theorem 4.2.6. Indeed, let us take $\tau _ { \mathcal { X } }$ to be the weak* topology in $\mathcal { X }$ and $\tau _ { \mathcal { V } }$ a topology in such that A is $\tau _ { \mathcal { X } ^ { - } } \tau _ { \mathcal { Y } }$ continuous and the norm in $\mathcal { V }$ is τ<sub>Y</sub>-l.s.c. (weak*, weak or strong topologies will work). For example, if $\mathcal { V }$ has a separable predual, we can take $\tau _ { \mathcal { V } }$ to be the weak* topology on . It can be easily verified that A is weak*-weak* continuous if it is the dual of another operator $A = B ^ { * }$ (where B acts from the predual of $\mathcal { V }$ into the predual of ). With these choices, the conditions of Theorem 4.2.6 are satisfied. 

Hence, by strong duality we have that 

$$
\frac {1}{2} \| A u _ {\delta} - f _ {\delta} \| _ {\mathcal {Y}} ^ {2} + \alpha \mathcal {J} (u _ {\delta}) = \alpha \left<   \mu_ {\delta}, f _ {\delta} \right> - \frac {\alpha^ {2}}{2} \| \mu_ {\delta} \| _ {\mathcal {Y}} ^ {2}.
$$

Optimality conditions (iii) from Theorem 4.1.43 take the following form 

$$
A ^ {*} \mu_ {\delta} \in \partial \mathcal {J} (u _ {\delta}), - \alpha \mu_ {\delta} \in \partial \left(\frac {1}{2} \| \cdot \| _ {\mathcal {Y}} ^ {2}\right) (A u _ {\delta} - f _ {\delta}).\tag{5.3}
$$

From Corollary 5.1.2 we conclude that 

$$
\| \alpha \mu_ {\delta} \| _ {\mathcal {Y} ^ {*}} = \| A u _ {\delta} - f _ {\delta} \| _ {\mathcal {Y}}.\tag{5.4}
$$

Also, comparing the values of ${ \frac { 1 } { 2 } } \| \cdot \| ^ { 2 }$ at 0 and at $A u _ { \delta } - f _ { \delta }$ and using the fact that $\mathbf { \nabla } \cdot \alpha \mu _ { \delta }$ is a subgradient, we get that 

$$
0 \geqslant \frac {1}{2} \| A u _ {\delta} - f _ {\delta} \| _ {\mathcal {Y}} ^ {2} + \langle - \alpha \mu_ {\delta}, 0 - (A u _ {\delta} - f _ {\delta}) \rangle
$$

and therefore 

$$
\langle \alpha \mu_ {\delta}, A u _ {\delta} - f _ {\delta} \rangle \leqslant - \frac {1}{2} \| A u _ {\delta} - f _ {\delta} \| _ {\mathcal {Y}} ^ {2}.\tag{5.5}
$$

We will use the estimates (5.4) and (5.5) later in Theorem 5.2.4. 

## 5.2 Source Condition and Convergence Rates

Formal limits of problems (5.1) and (5.2) at $\delta = 0$ are 

$$
\inf _ {u: A u = f} \mathcal {J} (u) = \inf _ {u \in \mathcal {X}} \chi_ {\{f \}} (A u) + \mathcal {J} (u)\tag{5.6}
$$

and 

$$
\begin{array}{r l} \sup _ {\mu \colon A ^ {*} \mu \in \partial \mathcal {J} (0)} \langle \mu , f \rangle = & \sup _ {\mu \colon A ^ {*} \mu \in \partial \mathcal {J} (0)} \left\langle \mu , A u _ {\mathcal {J}} ^ {\dagger} \right\rangle \\ = & \sup _ {\mu \colon A ^ {*} \mu \in \partial \mathcal {J} (0)} \left\langle A ^ {*} \mu , u _ {\mathcal {J}} ^ {\dagger} \right\rangle = \sup _ {v \in \mathcal {R} (A ^ {*}) \cap \partial \mathcal {J} (0)} \left\langle v, u _ {\mathcal {J}} ^ {\dagger} \right\rangle . \end{array}\tag{5.7}
$$

Since the characteristic function $\chi _ { \{ f \} } ( \cdot )$ is not continuous anywhere in its domain, Theorem 4.1.43 does not apply and we cannot guarantee that a solution of the dual limit problem (5.7) exists. Indeed, since $\mathcal { R } ( A ^ { * } )$ is not closed (strongly and hence weakly, since it is convex [18, Thm. V.3.13]), a solution may not exist. 

We shall see that existence is guaranteed by the following condition 

Definition 5.2.1 (Source condition [14]). We say that a -minimising solution $u _ { \mathcal { I } } ^ { \dagger }$ satisfies the source condition if 

$$
\exists \mu^ {\dagger} \in \mathcal {Y} ^ {*} \quad s u c h t h a t \quad A ^ {*} \mu^ {\dagger} \in \partial \mathcal {J} (u _ {\mathcal {J}} ^ {\dagger}),\tag{5.8}
$$

i.e. i $f \mathcal { R } ( A ^ { * } ) \cap \partial \mathcal { I } ( u _ { \mathcal { I } } ^ { \dagger } ) \neq \emptyset .$ 

First we will see that this condition is necessary for the dual solution $\mu _ { \delta }$ from (5.3) to stay bounded as $\delta  0$ 

Theorem 5.2.2 (Necessary conditions, [24]). Let and be Banach spaces and separable. Let conditions of Theorem $\it 4 . 2 . 6$ be satisfied and $\alpha = \alpha ( \delta )$ be chosen as required by Theorem 4.2.7. Suppose that the dual solution $\mu _ { \delta }$ is bounded uniformly in δ. Then there exists $\mu ^ { \dagger } \in \mathcal { V } ^ { * }$ such that $A ^ { * } \mu ^ { \dagger } \in \partial \mathcal { I } ( u _ { \mathcal { T } } ^ { \dagger } )$ 

Proof. Consider an arbitrary sequence $\delta _ { n } \downarrow 0$ . Since $\| \mu _ { \delta } \| _ { \mathcal { V } ^ { * } } \leqslant C$ for all $\delta ,$ by the Banach-Alaogly theorem we get that there exists a weakly-* convergent subsequence (that we do not relabel), i.e. 

$$
\mu_ {\delta_ {n}} \rightharpoonup^ {*} \mu^ {\dagger} \in \mathcal {Y} ^ {*}.
$$

Then we get that 

$$
A ^ {*} \mu_ {\delta_ {n}} \rightharpoonup^ {*} A ^ {*} \mu^ {\dagger}.
$$

Since $\partial \mathcal { I } ( 0 )$ is weakly-* closed (Theorem 4.1.19) and $A ^ { * } \mu _ { \delta _ { n } } \in \partial { \mathcal { T } } ( 0 )$ by (5.3), we get that 

$$
A ^ {*} \mu^ {\dagger} \in \partial \mathcal {J} (0).
$$

Since $\mathcal { I }$ is absolute one-homogeneous, we get by Proposition 4.1.28 that 

$$
\langle A ^ {*} \mu_ {\delta_ {n}}, u _ {\delta_ {n}} \rangle = \mathcal {J} (u _ {\delta_ {n}}) \to \mathcal {J} (u _ {\mathcal {J}} ^ {\dagger}),\tag{5.9}
$$

where convergence follows from Theorem 4.2.7. We also observe that 

$$
\begin{array}{r c l} | \langle A ^ {*} \mu_ {\delta}, u _ {\delta} \rangle - \langle A ^ {*} \mu^ {\dagger}, u _ {\mathcal {J}} ^ {\dagger} \rangle | & = & | \langle A ^ {*} \mu_ {\delta}, u _ {\delta} - u _ {\mathcal {J}} ^ {\dagger} \rangle - \langle A ^ {*} (\mu^ {\dagger} - \mu_ {\delta}), u _ {\mathcal {J}} ^ {\dagger} \rangle | \\ & \leqslant & | \langle \mu_ {\delta}, A u _ {\delta} - f \rangle | + | \langle \mu^ {\dagger} - \mu_ {\delta}, f \rangle | \\ & \leqslant & \| \mu_ {\delta} \| \| A u _ {\delta} - f \| + | \langle \mu^ {\dagger} - \mu_ {\delta}, f \rangle | \to 0, \end{array}
$$

since $\| \mu _ { \delta _ { n } } \| _ { \mathcal { V } ^ { * } }$ is bounded, $\| A u _ { \delta _ { n } } - f \| _ { \mathcal { V } } \to 0$ and $\mu _ { \delta _ { n } }  ^ { * } \mu ^ { \dagger }$ . Combining this with (5.9), we get that 

$$
\mathcal {J} (u _ {\mathcal {J}} ^ {\dagger}) = \left\langle A ^ {*} \mu^ {\dagger}, u _ {\mathcal {J}} ^ {\dagger} \right\rangle .
$$

Since $A ^ { \ast } \mu ^ { \dagger } \in \partial \mathcal { I } ( 0 )$ and $\mathcal { I } ( u _ { \mathcal { T } } ^ { \dagger } ) = \left. A ^ { * } \mu ^ { \dagger } , u _ { \mathcal { T } } ^ { \dagger } \right.$ , we conclude, using Proposition 4.1.31, that $A ^ { * } \mu ^ { \dagger } \in \partial \mathcal { I } ( u _ { \mathcal { T } } ^ { \dagger } )$ □ 

$\mathrm { S o } .$ , the source condition is necessary for the boundedness of the dual solutions $\mu _ { \delta }$ as $\delta  0$ . It turns out to be also suficient. 

Theorem 5.2.3 (Suficient conditions, [24]). Let and be Banach spaces and $\mathcal { V }$ separable. Let conditions of Theorem 4.2.6 be satisfied and $\alpha = \alpha ( \delta )$ be chosen as required by Theorem 4.2.7. Suppose that the source condition (5.8) is satisfied at a  -minimising solution $u _ { \mathcal { I } } ^ { \dagger }$ . Then $\mu _ { \delta }$ is bounded uniformly in $\delta .$ . Moreover, $\mu _ { \delta } \ \to ^ { * } \ \mu ^ { \dagger }$ in ${ \mathcal { V } } ^ { * }$ as $\delta  0$ (perhaps, up to a subsequence), where $\mu ^ { \dagger }$ is the solution of the dual limit problem (5.7) with minimal norm. 

Proof. We omit the proof for time reasons. It can be found in [24] (for Hilbert spaces). 

The next theorem shows that the source condition (5.8) implies a convergence rates in terms of the Bregman distance. 

Theorem 5.2.4. Let the source condition (5.8) be satisfied at a -minimising solution $u _ { \mathcal { I } } ^ { \dagger }$ and let $u _ { \delta }$ be a regularised solution solving (5.1). Then the following estimate holds 

$$
D _ {\mathcal {J}} ^ {p _ {\delta}, p ^ {\dagger}} (u _ {\delta}, u _ {\mathcal {J}} ^ {\dagger}) \leqslant \frac {1}{4 \alpha} \left(\delta + \alpha \| \mu^ {\dagger} \|\right) ^ {2} + \delta \| \mu^ {\dagger} \|.
$$

where $p _ { \delta } = A ^ { \ast } \mu _ { \delta } \in \partial \mathcal { T } ( u _ { \delta } )$ with $\mu _ { \delta }$ as defined in (5.3) and $p ^ { \dagger } = A ^ { * } \mu ^ { \dagger } \in \partial \mathcal { I } ( u _ { \mathcal { T } } ^ { \dagger } )$ is as defined in (5.8). $D _ { \mathcal { T } } ^ { p _ { \delta } , p ^ { \dagger } } ( u _ { \delta } , u _ { \mathcal { T } } ^ { \dagger } )$ denotes the symmetric Bregman distance between $u _ { \delta }$ and $u _ { \mathcal { I } } ^ { \dagger }$ . For the optimal choice $\begin{array} { r } { \alpha = \frac { \delta } { \Vert \mu ^ { \dagger } \Vert } } \end{array}$ we get that 

$$
D _ {\mathcal {J}} ^ {p _ {\delta}, p ^ {\dagger}} (u _ {\delta}, u _ {\mathcal {J}} ^ {\dagger}) \leqslant 3 \delta \| \mu^ {\dagger} \|.
$$

Proof. We start with the following estimate 

$$
\begin{array}{r c l} \alpha D _ {\mathcal {J}} ^ {p _ {\delta}, p ^ {\dagger}} (u _ {\delta}, u _ {\mathcal {J}} ^ {\dagger}) & = & \alpha \langle p _ {\delta} - p ^ {\dagger}, u _ {\delta} - u _ {\mathcal {J}} ^ {\dagger} \rangle \\ & = & \alpha \langle \mu_ {\delta} - \mu^ {\dagger}, A u _ {\delta} - f \rangle \\ & = & \alpha \langle \mu_ {\delta}, A u _ {\delta} - f _ {\delta} \rangle + \alpha \langle \mu_ {\delta}, f _ {\delta} - f \rangle - \alpha \langle \mu^ {\dagger}, A u _ {\delta} - f _ {\delta} \rangle - \alpha \langle \mu^ {\dagger}, f _ {\delta} - f \rangle . \end{array}
$$

From (5.5) we know that 

$$
\alpha \langle \mu_ {\delta}, A u _ {\delta} - f _ {\delta} \rangle \leqslant - \frac {1}{2} \| A u _ {\delta} - f _ {\delta} \| _ {\mathcal {Y}} ^ {2}.
$$

and from (5.4) that $\alpha \| \mu _ { \delta } \| = \| A u _ { \delta } - f _ { \delta } \|$ . Using these estimates, the Cauchy-Schwarz inequality and the estimate $\| f - f _ { \delta } \| \leqslant \delta$ , we get 

$$
\alpha D _ {\mathcal {J}} ^ {p _ {\delta}, p ^ {\dagger}} (u _ {\delta}, u _ {\mathcal {J}} ^ {\dagger}) \leqslant - \frac {1}{2} \| A u _ {\delta} - f _ {\delta} \| ^ {2} + (\delta + \alpha \| \mu^ {\dagger} \|) \| A u _ {\delta} - f _ {\delta} \| + \alpha \delta \| \mu^ {\dagger} \|.
$$

The right-hand side is the following quadratic function of the scalar variable $\lVert A u _ { \delta } - f _ { \delta } \rVert$ 

$$
\varphi (t) := - \frac {1}{2} t ^ {2} + (\delta + \alpha \| \mu^ {\dagger} \|) t + \alpha \delta \| \mu^ {\dagger} \|, \quad t \in \mathbb {R}.
$$

It achieves its maximum at $t _ { 0 } = ( \delta + \alpha \| \mu ^ { \dagger } \| )$ and this maximum value is equal to 

$$
\varphi (t _ {0}) = \frac {(\delta + \alpha \| \mu^ {\dagger} \|) ^ {2}}{2} + \alpha \delta \| \mu^ {\dagger} \|.
$$

Substituting this into the above estimate for the Bregman distance and dividing both sides by $\alpha .$ , we get the desired estimate 

$$
D _ {\mathcal {J}} ^ {p _ {\delta}, p ^ {\dagger}} (u _ {\delta}, u _ {\mathcal {J}} ^ {\dagger}) \leqslant \frac {(\delta + \alpha \| \mu^ {\dagger} \|) ^ {2}}{2 \alpha} + \delta \| \mu^ {\dagger} \|.
$$

Diferentiating the right-hand side w.r.t. α and setting the derivative to zero, we obtain the following optimality condition for α 

$$
0 = \frac {2 \alpha \| \mu^ {\dagger} \| (\delta + \alpha \| \mu^ {\dagger} \|) - (\delta + \alpha \| \mu^ {\dagger} \|) ^ {2}}{2 \alpha^ {2}} = \frac {\alpha^ {2} \| \mu^ {\dagger} \| ^ {2} - \delta^ {2}}{2 \alpha^ {2}}
$$

and 

$$
\alpha = \frac {\delta}{\| \mu^ {\dagger} \|}.
$$

With this optimal choice of α we get the following estimate 

$$
D _ {\mathcal {J}} ^ {p _ {\delta}, p ^ {\dagger}} (u _ {\delta}, u _ {\mathcal {J}} ^ {\dagger}) \leqslant 3 \delta \| \mu^ {\dagger} \|.
$$

Remark 5.2.5. Of course, we do not know $\mu ^ { \dagger }$ since we don’t know the $\mathcal { T } \cdot$ minimising solution $u _ { \mathcal { I } } ^ { \dagger }$ , but the theorem gives an optimal rate $\alpha \sim \delta$ for a priori parameter choice rules and a corresponding error estimate $D _ { \mathcal { T } } ^ { p _ { \delta } , p ^ { \dagger } } ( u _ { \delta } , u _ { \mathcal { T } } ^ { \dagger } ) = O ( \delta )$ 

Now we will look at two examples involving Total Variation to get a feeling for what the source condition ‘means’. 

Example 5.2.6 (Total Variation). Let $\Omega \subset \mathbb { R } ^ { 2 }$ be a bounded domain with a $C ^ { \infty }$ boundary. Let $\mathscr { X } = \mathrm { B V } ( \Omega )$ and $\mathcal { V } = L ^ { 2 } ( \Omega )$ and $\mathcal { I } ( \cdot ) = \mathrm { T V } ( \cdot )$ . Recall the ROF problem 

$$
\min _ {u \in \mathrm{BV}} \frac {1}{2} \| I u - f _ {\delta} \| _ {L ^ {2}} ^ {2} + \alpha \operatorname{TV} (u),
$$

where $I \colon \mathrm { B V } ( \Omega )  L ^ { 2 } ( \Omega )$ is the embedding operator, which is continuous since $\Omega \subset \mathbb { R } ^ { 2 }$ The adjoint $I ^ { * } \colon L ^ { 2 } ( \Omega ) \to \mathrm { B V } ^ { * } ( \Omega )$ continuously embeds $L ^ { 2 }$ into $\mathrm { B V } ^ { * }$ . Clearly, $I ^ { * }$ is not surjectuve and $\mathcal { R } ( I ^ { * } ) = L ^ { 2 } ( \Omega )$ 

From Example 4.3.6 we know that 

$$
\operatorname{TV} (\mathbf {1} _ {\mathcal {C}}) = \operatorname{Per} (\mathcal {C}),
$$

where $\mathbf { 1 } _ { \mathcal { C } }$ is the indicator function of the set . Denoting by $\mathbf { n } _ { \partial \mathcal { C } }$ the unit normal, we obtain 

$$
\mathrm{Per} (\mathcal {C}) = \int_ {\partial \mathcal {C}} 1 = \int_ {\partial \mathcal {C}} \left\langle \mathbf {n} _ {\partial \mathcal {C}}, \mathbf {n} _ {\partial \mathcal {C}} \right\rangle .
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/e6af376405a2a88026bef703737bfe9e04729b4b7e2291c302307e33e66e3a0e.jpg)



Figure 5.1: Example of a set whose indicator function does not satisfy the source condition.


Since $\mathbf { n } _ { \partial \mathcal { C } } \in C ^ { \infty } ( \partial \mathcal { C } , \mathbb { R } ^ { 2 } )$ and $\| \mathbf { n } _ { \partial \mathcal { C } } ( x ) \| _ { 2 } = 1$ for any $x ,$ we can extend $\mathbf { n } _ { \partial \mathcal { C } }$ to a $C _ { 0 } ^ { \infty } ( \Omega , \mathbb { R } ^ { 2 } )$ vector field $\psi$ with $\mathrm { s u p } _ { x \in \Omega } \| \psi ( x ) \| _ { 2 } \leqslant 1$ . Therefore, using the divergence theorem, we obtain that 

$$
\int_ {\partial \mathcal {C}} \left\langle \mathbf {n} _ {\partial \mathcal {C}}, \mathbf {n} _ {\partial \mathcal {C}} \right\rangle = \int_ {\partial \mathcal {C}} \left\langle \psi , \mathbf {n} _ {\partial \mathcal {C}} \right\rangle = \int_ {\mathcal {C}} \operatorname{div} \psi = \int_ {\Omega} \mathbf {1} _ {\mathcal {C}} \operatorname{div} \psi .
$$

Combining all these equalities, we $\mathrm { g e t }$ that 

$$
\mathrm{TV} (\mathbf {1} _ {\mathcal {C}}) = \int_ {\Omega} \mathbf {1} _ {\mathcal {C}} \operatorname{div} \psi = \langle \operatorname{div} \psi , \mathbf {1} _ {\mathcal {C}} \rangle .
$$

Taking an arbitrary $u \in \mathrm { B V } ( \Omega )$ , we note that 

$$
\begin{array}{c} \operatorname{TV} (u) - \langle \operatorname{div} \psi , u \rangle = \sup _ {\varphi \in C _ {0} ^ {\infty} (\Omega , \mathbb {R} ^ {2})} \langle \operatorname{div} \varphi , u \rangle - \langle \operatorname{div} \psi , u \rangle \geqslant 0, \\ \sup _ {x \in \Omega} \| \varphi (x) \| _ {2} \leqslant 1 \end{array}
$$

since $\varphi = \psi$ is feasible. Therefore, div $\psi \in \partial \mathrm { T V } ( 0 )$ and, since $\mathrm { T V } ( { \bf 1 } _ { \mathcal C } ) = \langle \mathrm { d i v } \psi , { \bf 1 } _ { \mathcal C } \rangle$ , we also get that 

$$
\operatorname{div} \psi \in \partial \operatorname{TV} (\mathbf {1} _ {\mathcal {C}}).
$$

Since $\psi \in C _ { 0 } ^ { \infty } ( \Omega , \mathbb { R } ^ { 2 } )$ , we have div $\psi \in C _ { 0 } ^ { \infty } ( \Omega ) \subset L ^ { 2 } ( \Omega ) = \mathcal { R } ( I ^ { * } )$ and the source condition is satisfied at $u = \mathbf { 1 } _ { \mathcal { C } }$ with $\mu ^ { \dagger } = \operatorname { d i v } \psi$ 

Example 5.2.7 (Total Variation). In the same setting as in Example 5.2.6, let be a domain with a nonsmooth boundary, e.g., a square $\mathcal { C } ~ = ~ [ 0 , 1 ] ^ { 2 }$ We will show in this example that in this case $\partial \mathrm { T V } ( \mathbf { 1 } _ { \mathcal { C } } ) \cap \mathcal { R } ( I ^ { * } ) = \emptyset$ , where $\mathcal { R } ( I ^ { \ast } ) = L ^ { 2 } ( \Omega )$ as before, i.e. the source condition fails. 

Assume that there exists $p _ { 0 } \in \partial \mathrm { T V } ( \mathbf { 1 } _ { \mathcal { C } } ) \cap L ^ { 2 } ( \Omega )$ . Then by the results of Example 4.3.6 we have that 

$$
\langle p _ {0}, \mathbf {1} _ {\mathcal {C}} \rangle = \operatorname{TV} (\mathbf {1} _ {\mathcal {C}}) = \operatorname{Per} (\mathcal {C}) = 4.
$$

Since $p _ { 0 }$ is a subgradient, we get that for any $u \in \mathrm { B V } ( \Omega )$ 

$$
\operatorname{TV} (u) - \langle p _ {0}, u \rangle \geqslant 0.
$$

Let us cut a triangle $\mathcal { C } _ { \varepsilon }$ of size $\varepsilon$ from a corner of $\mathcal { C }$ as shown in Figure 5.1. Then for $\boldsymbol { u } = \mathbf { 1 } _ { \mathcal { C } \backslash \mathcal { C } _ { \ell } }$ 6 we $\mathrm { g e t }$ 

$$
\mathrm{TV} (\mathbf {1} _ {\mathcal {C} \setminus \mathcal {C} _ {\varepsilon}}) \geqslant \left\langle p _ {0}, \mathbf {1} _ {\mathcal {C} \setminus \mathcal {C} _ {\varepsilon}} \right\rangle = \left\langle p _ {0}, \mathbf {1} _ {\mathcal {C}} \right\rangle - \left\langle p _ {0}, \mathbf {1} _ {\mathcal {C} _ {\varepsilon}} \right\rangle
$$

and therefore 

$$
\langle p _ {0}, \mathbf {1} _ {\mathcal {C} _ {\varepsilon}} \rangle \geqslant \mathrm{TV} (\mathbf {1} _ {\mathcal {C}}) - \mathrm{TV} (\mathbf {1} _ {\mathcal {C} \setminus \mathcal {C} _ {\varepsilon}}) = \mathrm{Per} (\mathcal {C}) - \mathrm{Per} (\mathcal {C} \setminus \mathcal {C} _ {\varepsilon}) = 4 - (4 - 2 \varepsilon + \sqrt {2} \varepsilon) = (2 - \sqrt {2}) \varepsilon > 0.
$$

$\mathrm { B y }$ H¨older’s inequality we get that 

$$
\left\langle p _ {0}, \mathbf {1} _ {\mathcal {C} _ {\varepsilon}} \right\rangle = \int_ {\mathcal {C} _ {\varepsilon}} p _ {0} \cdot \mathbf {1} \leqslant \left(\int_ {\mathcal {C} _ {\varepsilon}} | p _ {0} | ^ {2}\right) ^ {1 / 2} \left(\int_ {\mathcal {C} _ {\varepsilon}} 1\right) ^ {1 / 2} = \frac {1}{\sqrt {2}} \varepsilon \left(\int_ {\mathcal {C} _ {\varepsilon}} | p _ {0} | ^ {2}\right) ^ {1 / 2}.
$$

Combining the last two inequalities, we get 

$$
(2 - \sqrt {2}) \varepsilon \leqslant \langle p _ {0}, \mathbf {1} _ {\mathcal {C} _ {\varepsilon}} \rangle \leqslant \frac {1}{\sqrt {2}} \varepsilon \left(\int_ {\mathcal {C} _ {\varepsilon}} | p _ {0} | ^ {2}\right) ^ {1 / 2}
$$

and therefore 

$$
\int_ {\mathcal {C} _ {\varepsilon}} | p _ {0} | ^ {2} \geqslant 2 (2 - \sqrt {2}) ^ {2} > 0
$$

for all $\varepsilon > 0$ . However, since $p _ { 0 } \in L ^ { 2 } ( \Omega )$ by assumption, we must have 

$$
\int_ {\mathcal {C} _ {\varepsilon}} | p _ {0} | ^ {2} \to 0 \quad \mathrm{as} \varepsilon \to 0.
$$

This contradiction proves that such $p _ { 0 }$ does not exist and $\partial \mathrm { T V } ( \mathbf { 1 } _ { \mathcal { C } } ) \cap \mathcal { R } ( I ^ { * } ) = \varnothing$ 

## Chapter 6

# Bayesian probability and statistics

## 6.1 From inverse problems to Bayesian inverse problems

We consider an inverse problem of the form: 

$$
\text {   Find   } u \in \mathcal {X}: \mathcal {A} (u) + n = f _ {n},
$$

where $\mathcal { X }$ is a separable Banach space, $n \in \mathcal { V }$ is observational noise, is another separable Banach space, $f _ { n } \in \mathcal { N }$ is data, and $\mathcal { A } : \mathcal { X }  \mathcal { Y }$ is a measurable (possibly non-linear) operator. 

So far, we have studied techniques (pseudo-inverse, regularisation) to find estimates for the parameter u. In situation where the noise n is large or the data is non-informative, we should not only give an estimate for u, but also comment on the uncertainty left in the parameter. This is the problem we study in this part of the lecture. 

There are multiple ways to represent certainty, knowledge, risk, or uncertainty in a parameter, such as $u \in \mathcal X$ . Common models are Bayesian probability theory, fuzzy set theory, Dempster–Shafer theory, random set theory,... 

We follow Bayesian probability theory: model uncertain parameters as random variables. 

## Intuitions, concepts, questions, and answers:

1. Can we use randomness to model deterministic, uncertain objects? 

• Not with the usual “frequentist” interpretation of probability. Here, the probability of an event is the limit of the relative frequency of the occurrence of the event in infinitely repeated, independent experiments. If the object we study is deterministic, the frequentist approach will only give us probabilities in 0, 1 . 

• Indeed, with the “Bayesian” interpretation of probability. Here the probability of an event is the amount of money (in £) we would give in a game to win £1 if the event occurs. This ‘game’ does not require any inherent randomness. 

## 2. Can we represent the learning of information about a parameter?

• Learning that an event B occurred can be represented via conditional probability. Indeed, this learning process is given by the map $\mathbb { P } ( U \in \cdot ) \mapsto \mathbb { P } ( U \in \cdot | B )$ 

• In practice, we can often compute updates of this form through Bayes’ formula. 

3. Can we use Bayesian probability to argue about logical statements? 

• Cox’s Theorem [17]: Bayesian probability is a sensible extension of Aristotelian logic. 

4. Is Bayesian probability theory congruent with our everyday experience? 

• It probably is. See the example below. 

Example 6.1.1. ‘Tossing a coin’ can be modelled as a Bernoulli experiment 

$$
\mathbb {P} (\text { Coin   shows   Head }) = 0. 5 = \mathbb {P} (\text { Coin   shows   Tail }).
$$

Actually, this is a mechanical process that is completely deterministic. However, it is dificult to predict its outcome. The model is complicated and subject to many uncertain parameters: force, speed, gravity, air flow. . . Hence, it is easier to model the coin as a random variable. 

5. How do we employ these ideas in inverse problems? 

(a) We assume that noise n and parameter u are random variables N and $U .$ . The distributions of N and U describe our knowledge concerning noise and parameter before observing the data set. The distribution of U is called prior distribution $\mu _ { 0 } : = \mathbb { P } ( U \in \cdot )$ 

(b) We observe the data set $f _ { n }$ , indeed, we observe the occurrence of the event 

$$
\{f _ {n} = \mathcal {A} (U) + N \}.
$$

(c) We employ Bayes’ theorem to ‘update’ the prior by incorporating the observational data 

$$
\mu_ {0} = \mathbb {P} (U \in \cdot) \mapsto \mathbb {P} (U \in \cdot | f _ {n} = \mathcal {A} (U) + N) =: \mu_ {\text { post }}.
$$

As $\mu _ { \mathrm { p o s t } }$ now explains our knowledge after seeing the data, we call it posterior distribution. 

## 6.2 Reminder: measure, probability, and integration

During this course, we will make extensive use of measure-theoretic probability theory. Thus, we will briefly remind ourselves of some definitions, examples, and results from measure and probability theory that we will require throughout this lecture. In case the reader would like to get a more thorough reminder, we refer them to [6], [10], [25]. We commence with σ-algebras. 

Definition 6.2.1 (σ-algebra). Let Ω be a non-empty set, let $2 ^ { \Omega } : = \{ A : A \subseteq \Omega \}$ be the power set of Ω, and let ${ \mathcal { F } } \subseteq 2 ^ { \Omega }$ satisfy (i)-(iii): 

(i) $\Omega \in { \mathcal { F } } _ { \mathbf { \Delta } }$ 

(ii) for any $F \in { \mathcal { F } }$ , we have also $F ^ { c } : = \Omega \backslash F \in { \mathcal { F } }$ , and 

(iii) for any countable family $( F _ { n } : n \in \mathbb { N } ) \in \mathcal { F } ^ { \mathbb { N } }$ , we have also $\textstyle \bigcup _ { n \in \mathbb { N } } F _ { n } \in { \mathcal { F } }$ 

Then,  is called σ-algebra on Ω and $( \Omega , { \mathcal { F } } )$ is called measurable space. 

There are several ways to construct σ-algebras. They can for instance be induced by systems of sets or functions. 

Definition 6.2.2 (Induced σ-algebra). 1. Let Ω be non-empty and $\mathcal { E } \subseteq 2 ^ { \Omega }$ . We define the σ-algebra induced by on Ω by 

$$
\sigma_{\Omega}(\mathcal{E}):= \bigcap_{\substack{\mathcal{F}^{\prime}\supset \mathcal{E}\\ \mathcal{F}^{\prime} is \sigma -algebra on \Omega}}\mathcal{F}^{\prime}.
$$

2. Let Ω be non-empty, let $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ be a measurable space, and let $g : \Omega \to \Omega ^ { \prime }$ be a function. We define the σ-algebra induced by g on Ω by 

$$
\sigma_ {\Omega} (g) := \{\{g \in F ^ {\prime} \}: F ^ {\prime} \in \mathcal {F} ^ {\prime} \},
$$

where 

$$
\{g \in F ^ {\prime} \} := g ^ {- 1} (F ^ {\prime}) := \{\omega \in \Omega : g (\omega) \in F ^ {\prime} \}
$$

is the pre-image of $F ^ { \prime }$ under g. 

Example 6.2.1. Let Ω be a non-empty set. 

1. $2 ^ { \Omega }$ is the largest σ-algebra on Ω. , Ω is the smallest σ-algebra. 

2. Let Ω be a topological space with open sets $O \subseteq 2 ^ { \Omega }$ . The σ-algebra $\sigma _ { \Omega } ( O ) = : B \Omega$ is called Borel-σ-algebra on Ω. 

A σ-algebra is the natural space to define a (probability) measure on. 

Definition 6.2.3 (Measure and probability measure). Let $( \Omega , { \mathcal { F } } )$ be a measurable space and let $\mu : { \mathcal { F } }  [ 0 , \infty ]$ be a function, satisfying $( i ) , ( i i )$ 

(i) $\mu ( \varnothing ) = 0 ;$ 

(ii) for any countable family $( F _ { m } : m \in \mathbb { N } ) \in \mathcal { F } ^ { \mathbb { N } }$ of mutually disjoint sets, i.e. $F _ { n } \cap F _ { m } = \emptyset$ $( n \neq m )$ . Then, we have $\begin{array} { r } { \mu \left( \bigcup _ { m \in \mathbb { N } } F _ { m } \right) = \sum _ { m \in \mathbb { N } } \mu ( F _ { m } ) } \end{array}$ 

Then, µ is called measure on $( \Omega , { \mathcal { F } } )$ and $( \Omega , { \mathcal { F } } , \mu )$ is called measure space. If a measure µ additionally satisfies (iii): 

(iii) $\mu ( \Omega ) = 1$ 2 

the measure $\mu$ is called probability measure and $( \Omega , { \mathcal { F } } , \mu )$ is called probability space. Finally, a measure µ is called σ-finite, if 

(iv) there is a countable family $( F _ { m } : m \in \mathbb { N } ) \in \mathcal { F } ^ { \mathbb { N } }$ , with $\begin{array} { r } { \bigcup _ { m \in \mathbb { N } } F _ { m } = \Omega } \end{array}$ and $\mu ( F _ { m } ) < \infty$ $( m \in \mathbb { N } )$ 

Example 6.2.2. Let $( \Omega , { \mathcal { F } } )$ be some measurable space. 

$\# : \mathcal { F }  [ 0 , \infty ]$ defined by 

$$
\# (F) := \left\{ \begin{array}{l l} \infty , & \text { if   } F \text {   is   infinite } \\ | F |, & \text { otherwise. } \end{array} \right. \quad (F \in \mathcal {F})
$$

is a measure and called counting measure, 

• Let $\omega \in \Omega$ . Then, $\delta ( \cdot - \omega ) : { \mathcal { F } }  [ 0 , \infty ]$ defined by 

$$
\delta (F - \omega) := \left\{ \begin{array}{l l} 1, & \text { if } F \ni \omega \\ 0, & \text { otherwise } \end{array} \right. \qquad (F \in \mathcal {F})
$$

is called Dirac measure concentrated in ω. The Dirac measure is a probability measure. 

• Let $k \in \mathbb { N } , \Omega : = \mathbb { R } ^ { k }$ , and $\lambda _ { k } : B \mathbb { R } ^ { k }  [ 0 , \infty ]$ be the unique measure that satisfies 

$$
\lambda_ {k} \left(\prod_ {i = 1} ^ {k} [ a _ {i}, b _ {i})\right) = \prod_ {i = 1} ^ {k} (b _ {i} - a _ {i}),
$$

if $a _ { i } \leqslant b _ { i } \ ( i = 1 , . . . , k )$ . Then $\lambda _ { k }$ is called k-dimensional Lebesgue measure. 

Exercise 6.2.4. 1. Show that the Dirac and counting measure are measures. 

2. Show that Dirac and Lebesgue measure are σ-finite. 

3. When is the counting measure σ-finite? 

We already learned the concept of using a function to construct a σ-algebra. In the following, we would like to use functions to represent uncertainties (‘random variables’) and use measures to integrate functions. Here, we require the concept of ‘measurability’. 

Definition 6.2.5. Let $( \Omega , { \mathcal { F } } )$ and $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ be two measurable spaces and let $g : \Omega \to \Omega ^ { \prime }$ be a function. 

1. g is called measurable, i $f \left\{ g \in F ^ { \prime } \right\} \in \mathcal { F }$ , for any $F ^ { \prime } \in { \mathcal { F } } ^ { \prime }$ . In this case, we sometimes write $g : ( \Omega , \mathcal { F } )  ( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ 

2. Let g be measurable and µ be a measure on $( \Omega , { \mathcal { F } } )$ . Then, we define the push-forward measure $\mu ( g \in \cdot )$ . If in addition, µ is a probability measure, g is called random variable and $\mu ( g \in \cdot )$ is called (probability) distribution of g. 

This rather abstract definition of measurability does not appear to be very instructive in practice. A useful result is the following proposition 

Proposition 6.2.6. Let Ω be a topological space and $g : \Omega  \mathbb { R }$ be continuous, i.e. for any open $F ^ { \prime } \subseteq \mathbb { R }$ , the preimage $\{ g \in F ^ { \prime } \} \subseteq \Omega$ is open as well. Then, $g : ( \Omega , B \Omega )  ( \mathbb { R } , B \mathbb { R } )$ is measurable. 

Proof. Page 36 in [6]. 

Push-forward measures and probability distributions are well-defined measures and probability measures, respectively. 

Proposition 6.2.7. Let $( \Omega , { \mathcal { F } } , \mu )$ be a measure space, $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ be a measurable spaces, and let $g : ( \Omega , \mathcal { F } )  ( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ be a measurable function. Then, the pushforward measure $\mu ( g \in \cdot )$ is a measure on $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ . Moreover, if µ is a probability measure, then so is $\mu ( g \in \cdot )$ 

Proof. Exercise. 

Measurability is the basic concept needed to be able to integrate a function with respect to a measure. We start with simple functions. 

Definition 6.2.8. Let $( \Omega , { \mathcal { F } } , \mu )$ be a measure space. A function $g : \Omega  \mathbb { R }$ is called simple, $i f$ there exists an $m \in \mathbb { N }$ and $( F _ { i } : i = 1 , . . . , m ) \in \mathcal { F } ^ { m }$ , such that 

$$
g = \sum_ {i = 1} ^ {m} b _ {i} \mathbf {1} _ {F _ {i}},
$$

for some $b \in \mathbb { R } ^ { m }$ . Consider the following two assumptions: 

(i) $b \in [ 0 , \infty ) ^ { m }$ or $b \in ( - \infty , 0 ] ^ { m }$ 

(ii) for any $i \in \{ 1 , . . . , m \}$ , with $\mu ( F _ { i } ) = \infty$ , we have $b _ { i } = 0$ 

If either (i) or (ii) holds, we define the (Lebesgue) integral of g with respect to $\mu$ by 

$$
\int_ {\Omega} g \mathrm{d} \mu := \int_ {\Omega} g (\omega) \mathrm{d} \mu (\omega) := \int_ {\Omega} g (\omega) \mu (\mathrm{d} \omega) := \sum_ {i = 1; b _ {i} \neq 0} ^ {m} b _ {i} \mu (F _ {i}).
$$

If the expression on the right-hand side is finite, we call g (Lebesgue) integrable. 

Exercise 6.2.9. A simple function $g : \Omega \to { \mathbb { R } }$ is measurable from (Ω, ) to (<sup>R</sup>, <sup>R</sup>). 

To define the integral for more general functions g, we will approximate the function by simple functions. This gives us the following definition for the integral. 

Definition 6.2.10 (Lebesgue integral). Let $( \Omega , { \mathcal { F } } , \mu )$ be a measure space and let $g : ( \Omega , { \mathcal { F } } ) $ (<sup>R</sup>, <sup>R</sup>) be measurable and non-negative. Then, we define the (Lebesgue) integral of g by 

$$
\int_ {\Omega} g \mathrm{d} \mu := \sup \left\{\int_ {\Omega} h (\omega) \mathrm{d} \mu (\omega): 0 \leqslant h \leqslant g, h i s s i m p l e \right\}
$$

If the supremum is finite, we call g (Lebesgue) integrable. 

In the following proposition, we discuss the fundamental properties of the Lebesgue integral: linearity, monotonicity, and monotonic convergence. 

Proposition 6.2.11. Let $( \Omega , { \mathcal { F } } , \mu )$ be a measure space and let $g , h , g _ { 1 } , g _ { 2 } , \ldots : ( \Omega , \mathcal { F } ) $ (<sup>R</sup>, <sup>R</sup>) be measurable, non-negative functions. Then: 

1. If $g \leqslant h$ pointwise, then $\begin{array} { r } { \int _ { \Omega } g \mathrm { d } \mu \leqslant \int _ { \Omega } h \mathrm { d } \mu } \end{array}$ 

2. If $( g _ { m } : m \in \mathbb { N } )$ is pointwise increasing and lim $1 _ { m } \to \infty g _ { m } = g$ pointwise, then the sequence $\begin{array} { r } { \left( \int _ { \Omega } g _ { m } \mathrm { d } \mu : m \in \mathbb { N } \right) } \end{array}$ is increasing and lim $\begin{array} { r } { { \cal M } \to \infty \int _ { \Omega } g _ { m } \mathrm { d } \mu = \int _ { \Omega } g \mathrm { d } \mu } \end{array}$ 

3. For some $\alpha , \beta \in [ 0 , \infty ]$ , we have 

$$
\int_ {\Omega} \alpha g + \beta h \mathrm{d} \mu = \alpha \int_ {\Omega} g \mathrm{d} \mu + \beta \int_ {\Omega} h \mathrm{d} \mu .
$$

(We use the convention ${ } ^ { 6 4 } 0 \cdot \infty = 0 ^ { 3 3 } )$ 

Proof. Lemma 4.6 in [25]. 

Measurable functions $g$ taking values in <sup>R</sup> can be integrated by subtracting the integral of their negative part max $\left. \{ 0 , - g \right\}$ from the integral of their positive part max $\{ 0 , g \}$ , if one of them is integrable. 

Integrals of non-negative measurable functions give a natural way to define measures. 

Proposition and definition 6.2.12. Let $( \Omega , { \mathcal { F } } , \mu )$ be a measure space and let $g : ( \Omega , { \mathcal { F } } ) $ $( \mathbb { R } , B \mathbb { R } )$ be measurable and non-negative. Then, the map $\nu : \mathcal { F }  [ 0 , \infty ]$ , defined by 

$$
F \mapsto \int_ {\Omega} g \cdot \mathbf {1} _ {F} \mathrm{d} \mu =: \int_ {F} g \mathrm{d} \mu
$$

is a measure. ν is called measure with (µ-)density (function) $g .$ If ν is a probability measure, $g$ is called (µ-)probability density (function). 

Proof. Exercise. 

Definition 6.2.13. Let $( \Omega , { \mathcal { F } } , \mu ) : = ( \mathbb { R } , B \mathbb { R } , \lambda _ { 1 } )$ . Moreover, let $m \in \mathbb { R }$ and $\sigma > 0$ , and let $g : \Omega $ <sup>R</sup> be the measurable function 

$$
g (\omega) := \frac {1}{\sqrt {2 \pi} \sigma} \exp \left(- \frac {(\omega - m) ^ {2}}{2 \sigma^ {2}}\right).
$$

Then, the measure ν with $\lambda _ { 1 } - d e n s i t y \textit { g }$ is called Gaussian distribution on $\mathbb { \underline { { R } } }$ with mean m and variance $\sigma ^ { 2 }$ . We denote $\mathbf { n } ( \cdot ; m , \sigma ^ { 2 } ) : = g$ and $\mathrm { N } ( m , \sigma ^ { 2 } ) : = \nu$ . Moreover, we define the degenerate Gaussian distribution by $\mathrm { N } ( m , 0 ) : = \delta ( \cdot - m )$ 

A rather surprising result about measures and densities is the Radon–Nikodym Theorem. It is fundamental for the general definition of conditional expectations and also for the general form of $\mathrm { B a y e s } ^ { \mathrm { , } }$ theorem. Before stating the Radon–Nikodym Theorem, we define two more important notions regarding measures. 

Definition 6.2.14. Let $( \Omega , { \mathcal { F } } )$ be a measurable space and $\mu , \nu$ be two measure on that space. 

1. We define ν to be absolutely continuous with respect to $\mu ,$ if for all $F \in { \mathcal { F } }$ , with $\mu ( F ) = 0 .$ , we also have $\nu ( F ) = 0$ . In this case, we write $\nu \ll \mu .$ 

2. Let $A ( \omega )$ be a statement for all $\omega \in { \Omega }$ . We say that A holds µ-almost everywhere $( \mu - a . e . ) ,$ if there is a set $N \in { \mathcal { F } }$ such that $\mu ( N ) = 0$ and $A ( \omega )$ is true for $\varpi \in X \backslash N$ If µ is a probability measure, we sometimes say µ-almost surely $( \mu - a . s . )$ instead of µ-almost everywhere. 

Theorem 6.2.15 (Radon-Nikodym). Let $( \Omega , { \mathcal { F } } )$ be a measurable space and let µ, ν be $\sigma \mathrm { - }$ finite measures on $( \Omega , { \mathcal { F } } )$ . Then, the following two statements are equivalent: 

(i) $\nu \ll \mu$ 

(ii) There is a measurable function $g : ( \Omega , \mathcal { F } )  ( \mathbb { R } , B \mathbb { R } )$ , with 

$$
\nu (F) = \int_ {F} g \mathrm{d} \mu \quad (F \in \mathcal {F}).
$$

Moreover, the function g is $\mu { - } a { \cdot } e$ unique, called Radon–Nikodym derivative, and denoted $\begin{array} { r } { b y \ \frac { \mathrm { d } \nu } { \mathrm { d } \mu } : = g } \end{array}$ 

Proof. $( \mathrm { i i } ) \Rightarrow ( \mathrm { i } )$ : exercise. $( \mathrm { i } ) \Rightarrow ( \mathrm { i i } )$ : more complicated, see, $\mathrm { e . g . }$ , Corollary 7.34 in [25]. 

Exercise 6.2.16. Give an example for measures $\nu , \mu$ on $( \mathbb { R } , B \mathbb { R } )$ , with $\nu \ll \mu$ and $\mu$ not σ-finite, such that no Radon-Nikodym derivative exists. 

## 6.3 Conditional probability

For the remainder of the lecture, we always consider $( \Omega , \mathcal { F } , \mathbb { P } )$ as underlying probability space for any random variable. We typically omit its precise construction, but assume that Ω is a Polish space (separable and completely metrisable) and $\mathcal { F } : = \mathcal { B } \Omega$ . We denote integrals with respect to $\mathbb { P }$ sometimes by 

$$
\mathbb {E} [ \varphi ] := \int_ {\Omega} \varphi \mathrm{d} \mathbb {P},
$$

for some $\varphi : ( \Omega , { \mathcal { F } } )  ( \mathbb { R } , B \mathbb { R } )$ , for which this integral is well-defined. 

Example 6.3.1. Let $U : ( \Omega , { \mathcal { F } } ) \to ( \{ 1 , . . . , 6 \} , 2 ^ { \{ 1 , . . . , 6 \} } )$ be a random variable modelling the roll of a die, hence 

$$
\mathbb {P} (U = u) = \left\{ \begin{array}{l l} 1 / 6, & \text { if } u \in \{1,..., 6 \}, \\ 0, & \text { otherwise }. \end{array} \right.
$$

This probability measure models our knowledge concerning the outcome of the experiment. Now we consider an extended model. After the die is rolled and before its realisation is revealed, we are told whether the realisation is even or odd. Given this information, we can adjust our knowledge concerning the random variable $U { : }$ 

$$
\mathbb {P} (U = u | U \text {is even}) = \frac {\mathbb {P} (U = u \text {and} U \text {is even})}{\mathbb {P} (U \text {is even})},
$$

respectively 

$$
\mathbb {P} (U = u | U \text {   is   odd }) = \frac {\mathbb {P} (U = u \text {   and   } U \text {   is   odd })}{\mathbb {P} (U \text {   is   odd })}.
$$

In the example above, we used the elementary definition of conditional probabilities: 

$$
\mathbb {P} (F | F ^ {\prime}) = \frac {\mathbb {P} (F \cap F ^ {\prime})}{\mathbb {P} (F ^ {\prime})} \qquad (F, F ^ {\prime} \in \mathcal {F}, \mathbb {P} (F ^ {\prime}) > 0).
$$

This definition can only be used, if the event with respect to which the conditional probability is defined has a positive probability (here: U is even , U is odd ). 

This however is typically not the case in a Bayesian inverse problem since the probability measure of the noise is continuous. Hence, we need a more general definition of conditional probabilities. We start with conditional expectations. 

Theorem 6.3.2. Let $U : ( \Omega , { \mathcal { F } } ) \to ( \mathbb { R } , B \mathbb { R } )$ and $Y : ( \Omega , { \mathcal { F } } ) \to ( \mathscr { V } , B \mathscr { V } )$ be random variables and let $U$ be integrable. Then, there exists a measurable function $h : ( \mathcal { V } , B \mathcal { V } ) \to ( \mathbb { R } , B \mathbb { R } )$ , such that 

$$
\int_ {F} h (y) \mathbb {P} (Y \in \mathrm{d} y) = \int_ {\{Y \in F \}} U \mathrm{d} \mathbb {P} \quad (F \in \mathcal {B Y}).\tag{6.1}
$$

Moreover, h is $\mathbb { P } ( Y \in \cdot ) _ { - a . s }$ . unique. 

Proof. We assume without loss of generality that $U \ \geqslant \ 0 . \quad ( \mathrm { I f } \ U$ is real-valued, study max $\{ U , 0 \}$ and max $\{ - U , 0 \}$ separately.) Note that the map 

$$
F \mapsto \int_ {\{Y \in F \}} U \mathrm{d} \mathbb {P} =: \mu (F)
$$

defines a $\scriptstyle ( \sigma - )$ finite measure. We now show that $\mu \ll \mathbb { P } ( Y \in \cdot )$ : let $F _ { 0 } \in B \mathcal { V }$ be chosen such that $\mathbb { P } ( Y \in F _ { 0 } ) = 0$ . Then, 

$$
\int_ {\{Y \in F _ {0} \}} U \mathrm{d} \mathbb {P} = \int_ {\Omega} \mathbf {1} _ {\{Y \in F _ {0} \}} U \mathrm{d} \mathbb {P} = 0.
$$

By the Radon–Nikodym Theorem, there exists a $\mathbb { P } ( Y \in \cdot ) { \mathrm { - a . s } }$ . unique function $\begin{array} { r } { \boldsymbol { h } : = \frac { \mathrm { d } \boldsymbol { \mu } } { \mathrm { d } \mathbb { P } ( \boldsymbol { Y } \in \cdot ) } , } \end{array}$ 2 satisfying (6.1). □ 

Definition ${ \bf 6 . 3 . 3 . } \ h ( y )$ in Theorem 6.3.2 is called conditional expectation of U given $Y = y$ We write $h ( y ) = : \mathbb { E } [ U | Y = y ]$ , for $\mathbb { P } ( Y \in \cdot )$ -almost every $y \in \mathcal { V }$ 

Now we can define the conditional probability of some event $F$ by considering the indicator random variable $U = \mathbf { 1 } _ { F }$ . Since $x , y$ are Polish spaces, one can even find a $\mathbb { P } ( Y \in \cdot ) { \mathrm { - a . s } }$ . unique Markov kernel $( y , F ) \mapsto \mathbb { E } [ { \mathbf { 1 } } _ { F } | Y = y ]$ 

Definition 6.3.4. Let $( \Omega , { \mathcal { F } } ) , ( \Omega ^ { \prime } , { \mathcal { F } } ^ { \prime } )$ be measurable spaces. A map $M : \Omega \times \mathcal { F } ^ { \prime }  [ 0 , 1 ]$ is called Markov kernel from $\left( \Omega , \mathcal { F } \right) t o \ \left( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } \right)$ , if 

(i) $M ( \omega , \cdot )$ is a probability measure for all $\omega \in \Omega$ 2 

(ii) $M ( \cdot , F ^ { \prime } ) : ( \Omega , { \mathcal { F } } ) \to ( [ 0 , 1 ] , \mathcal { B } [ 0 , 1 ] )$ is measurable for all $F ^ { \prime } \in \mathcal { F } ^ { \prime }$ 

Theorem 6.3.5. Let $U : ( \Omega , { \mathcal { F } } ) \to ( { \mathcal { X } } , B { \mathcal { X } } )$ and $Y : ( \Omega , { \mathcal { F } } ) \to ( \mathscr { V } , B \mathscr { V } )$ be random variables. Then, there exist a Markov kernel M from $( \mathcal { V } , B \mathcal { V } )$ to ( , ), with 

$$
\int_ {F} M (y, F ^ {\prime}) \mathbb {P} (Y \in \mathrm{d} y) = \mathbb {P} (\{Y \in F \} \cap \{U \in F ^ {\prime} \}) \quad (F \in \mathcal {B Y}, F ^ {\prime} \in \mathcal {B X}).
$$

Moreover, M is $\mathbb { P } ( Y \in \cdot ) . s .$ unique. 

Proof. Non-trivial, but possible if Ω is Polish; see [26]. 

Definition 6.3.6. M in Theorem 6.3.5 is called (regular) conditional probability distribution of U given $Y = y$ . We write $M ( y , F ) : = \mathbb { P } ( U \in F | Y = y ) , f o r \ F \in \mathcal { B } \mathcal { X } , y \in \mathcal { Y }$ 

Example 6.3.7 (Example 6.3.1 rev.). In Example 6.3.1, we compute the conditional probability distribution of a die $U : ( \Omega , \mathcal { F } ) \to ( \{ 1 , . . . , 6 \} , 2 ^ { \{ 1 , . . . , 6 \} } )$ , given the information whether the outcome will be even or odd. Define a random variable $Y : ( \Omega , \mathcal { F } )  ( \{ 0 , 1 \} , 2 ^ { \{ 0 , 1 \} } )$ ) 

$$
\omega \mapsto \left\{ \begin{array}{l l} 0, & \text { if } U (\omega) \text { is   even } \\ 1, & \text { otherwise. } \end{array} \right.
$$

We can write 

$$
\mathbb {P} (U = u | U \text {   is   even }) =: \mathbb {P} (U = u | Y = 0), \quad \mathbb {P} (U = u | U \text {   is   odd }) =: \mathbb {P} (U = u | Y = 1).
$$

Indeed, one can show that these functions are conditional expectation/probability measures in the sense of definition 6.3.3. Let $F \in 2 ^ { \{ 0 , 1 \} }$ . We need to show that 

$$
\int_ {F} \mathbb {P} (U = u | Y = y) \mathbb {P} (Y \in \mathrm{d} y) = \mathbb {P} (\{U = u \} \cap \{Y \in F \}).
$$

Let $F : = \{ 0 \}$ . Then, we have 

$$
\begin{array}{l} \int_ {\{Y = 0 \}} \mathbf {1} _ {\{U = u \}} \mathrm{d} \mathbb {P} = \frac {1}{6} (\mathbf {1} _ {\{2 \}} (u) + \mathbf {1} _ {\{4 \}} (u) + \mathbf {1} _ {\{6 \}} (u)) \\ \qquad = \underbrace {\frac {1}{2}} _ {= \mathbb {P} (Y = 0)} \cdot \underbrace {\frac {1}{3} (\mathbf {1} _ {\{2 \}} (u) + \mathbf {1} _ {\{4 \}} (u) + \mathbf {1} _ {\{6 \}} (u))} _ {= \mathbb {P} (U = u | Y = 0)} \\ \qquad = \int_ {\{0 \}} \mathbb {P} (U = u | Y = y) \mathbb {P} (Y \in \mathrm{d} y) \end{array}
$$

Analogously, one can show condition (6.1) for $F = \varnothing , \{ 1 \} , \{ 0 , 1 \}$ 

In Theorem 6.3.5, we discuss that conditional probabilities are Markov kernels. Also the converse is true: given a Markov kernel, we can construct random variables such that the Markov kernel represents a conditional probability measure. 

Proposition 6.3.8. Let $M : \Omega ^ { \prime } \times \mathcal { F } ^ { \prime \prime }  [ 0 , 1 ]$ be a Markov kernel from $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ to $( \Omega ^ { \prime \prime } , \mathcal { F } ^ { \prime \prime } )$ Then, there is an underlying probability space $( \Omega , \mathcal { F } , \mathbb { P } )$ and random variables $X ^ { \prime } : \Omega  \Omega ^ { \prime }$ and $X ^ { \prime \prime } : \Omega  \Omega ^ { \prime \prime }$ such that: 

$$
M (\omega^ {\prime}, F ^ {\prime \prime}) = \mathbb {P} (X ^ {\prime \prime} \in F ^ {\prime \prime} | X ^ {\prime} = \omega^ {\prime}) \quad (F ^ {\prime \prime} \in \mathcal {F} ^ {\prime \prime} a n d \mathbb {P} (X ^ {\prime} \in \cdot) - a l m o s t a l l \omega^ {\prime} \in \Omega^ {\prime}).
$$

Proof. Define $( \Omega , \mathcal { F } ) : = ( \Omega ^ { \prime } \times \Omega ^ { \prime \prime } , \mathcal { F } ^ { \prime } \otimes \mathcal { F } ^ { \prime \prime } )$ . Let $\mu ^ { \prime }$ be some probability measure on $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ Moreover, let <sup>P</sup> be the measure satisfying 

$$
\mathbb {P} (F ^ {\prime} \times F ^ {\prime \prime}) = \int_ {F ^ {\prime}} M (\omega^ {\prime}, F ^ {\prime \prime}) \mathrm{d} \mu^ {\prime} (\omega^ {\prime}) (F ^ {\prime} \in \mathcal {F} ^ {\prime}, F ^ {\prime \prime} \in \mathcal {F} ^ {\prime \prime}).
$$

Let $X ^ { \prime } : \Omega  \Omega ^ { \prime } ~ ( \mathrm { r e s p . } ~ X ^ { \prime \prime } : \Omega  \Omega ^ { \prime \prime } )$ be the canonical projection on the first (resp. second) coordinate. Then $X ^ { \prime } \sim \mu ^ { \prime }$ and $X ^ { \prime \prime } \sim M ( X ^ { \prime } , \cdot )$ . Let $F ^ { \prime } \in \mathcal { F } ^ { \prime }$ and $F ^ { \prime \prime } \in \mathcal { F } ^ { \prime \prime }$ . Then it holds 

$$
\begin{array}{r l} & {\mathbb {P} (\{X ^ {\prime} \in F ^ {\prime} \} \cap \{X ^ {\prime \prime} \in F ^ {\prime \prime} \}) =  \int_ {\{X ^ {\prime} \in F ^ {\prime}, X ^ {\prime \prime} \in F ^ {\prime \prime} \}} \mathrm{d} \mathbb {P} =  \iint_ {\{X ^ {\prime} \in F ^ {\prime}, X ^ {\prime \prime} \in F ^ {\prime \prime} \}} M (\omega^ {\prime}, \mathrm{d} \omega^ {\prime \prime}) \mu^ {\prime} (\mathrm{d} \omega^ {\prime})} \\ & {\quad \overset {(*)} {=}  \int_ {F ^ {\prime}}  \int_ {F ^ {\prime \prime}} M (\omega^ {\prime}, \mathrm{d} \omega^ {\prime \prime}) \mu^ {\prime} (\mathrm{d} \omega^ {\prime}) =  \int_ {F ^ {\prime}} M (F ^ {\prime \prime}, \omega^ {\prime}) \mathbb {P} (X ^ {\prime} \in \mathrm{d} \omega^ {\prime}),} \end{array}
$$

where (*) is implied by Tonelli’s Theorem. Hence, $M ( F ^ { \prime \prime } , \omega ^ { \prime } ) = \mathbb { P } ( X ^ { \prime \prime } \in F ^ { \prime \prime } | X ^ { \prime } = \omega ^ { \prime } )$ is indeed a conditional probability distribution. □ 

As Markov kernels are consistent with conditional probabilities, we sometimes write $M ( \cdot | * ) : = M ( * , \cdot )$ 

Applying the concept of conditional expectations in general situations is not straightforward. However, probability measures are often given in terms of probability density functions. Given joint and marginal probability density functions, one can define the conditional probability in terms of a probability density function. 

Lemma 6.3.9. Let U, Y be random variables with joint probability distribution $\mathbb { P } ( ( U , Y ) \in \cdot )$ that is absolutely continuous with respect to a σ-finite measure ν on $( { \mathcal { X } } \times { \mathcal { Y } } , B { \mathcal { X } } \otimes B { \mathcal { Y } } )$ Assume that $\nu = \nu _ { U } \otimes \nu _ { Y }$ for σ-finite measure spaces $( \mathcal { X } , B \mathcal { X } , \nu _ { U } ) , ~ ( \mathcal { V } , B \mathcal { V } , \nu _ { Y } )$ . We write $g _ { U , Y } : = \frac { \mathrm { d } \mathbb { P } ( ( U , Y ) \in \cdot ) } { \mathrm { d } \nu }$ for the joint probability density function. Then, 

$$
\mathbb {P} (U \in \cdot) \ll \nu_ {U}, \mathbb {P} (Y \in \cdot) \ll \nu_ {Y},
$$

with probability density functions 

$$
g _ {U} := \int_ {\mathcal {Y}} g _ {U, Y} \mathrm{d} \nu_ {Y} = \frac {\mathrm{d} \mathbb {P} (U \in \cdot)}{\mathrm{d} \nu_ {U}} \quad (\nu_ {U} \text {-a.e.}),
$$

$$
g _ {Y} := \int_ {\mathcal {X}} g _ {U, Y} \mathrm{d} \nu_ {U} = \frac {\mathrm{d} \mathbb {P} (Y \in \cdot)}{\mathrm{d} \nu_ {Y}} \quad (\nu_ {Y} \text {-a.e.}).
$$

Proof. Let $A \in B { \mathcal { X } }$ . We have 

$$
\mathbb {P} (U \in A) = \mathbb {P} (U \in A, Y \in \mathcal {Y}) = \int_ {A \times \mathcal {Y}} g _ {U, Y} \mathrm{d} \nu = \int_ {A} \underbrace {\int_ {\mathcal {Y}} g _ {U , Y} \mathrm{d} \nu_ {Y}} _ {=: g _ {U}} \mathrm{d} \nu_ {U},
$$

by the Theorem of Tonelli. Hence, $\mathbb { P } ( U \in \cdot ) \ll \nu _ { U }$ . The statement about Y can be proven analoguously. □ 

Theorem 6.3.10. Under the assumptions of Lemma 6.3.9, we have $\mathbb { P } ( U \in \cdot | Y = y ) \ll \nu _ { U }$ with ν<sub>U</sub> -density: 

$$
g _ {U | Y = y} (u) := \left\{ \begin{array}{l l} \frac {g _ {U , Y} (u , y)}{g _ {Y} (y)}, & \text { if } g _ {Y} (y) > 0, \\ 0, & \text { otherwise } \end{array} \right. \quad (u \in \mathcal {X}, \nu_ {U} \text {-a.e.}; y \in \mathcal {Y}, \mathbb {P} (Y \in \cdot) \text {-a.e.}),
$$

and equivalently $\mathbb { P } ( Y \in \cdot | U = u ) \ll \nu _ { Y }$ with ν<sub>Y</sub> -density: 

$$
g _ {Y | U = u} (y) := \left\{ \begin{array}{l l} \frac {g _ {U , Y} (u , y)}{g _ {U} (u)}, & \text { if } g _ {U} (u) > 0, \\ 0, & \text { otherwise } \end{array} \right. \quad (y \in \mathcal {Y}, \nu_ {Y} \text {-a.e.}; u \in \mathcal {X}, \mathbb {P} (U \in \cdot) \text {-a.e.}).
$$

Proof. Let $A \in B \mathcal { X } , F \in B \mathcal { Y }$ . By Definition 6.3.6, $\mathbb { P } ( U \in A | Y = y )$ fulfills (6.1): 

$$
\begin{array}{r l} & {\mathbb {P} (U \in A, Y \in F) = \int_ {F} \mathbb {P} (U \in A | Y = y) \mathbb {P} (Y \in \mathrm{d} y)} \\ & {\qquad = \int_ {F} \mathbb {P} (U \in A | Y = y) g _ {Y} (y) \mathrm{d} \nu_ {Y} (y)} \\ & {\qquad = \int_ {F \cap \{g _ {Y} > 0 \}} \mathbb {P} (U \in A | Y = y) g _ {Y} (y) \mathrm{d} \nu_ {Y} (y),} \end{array}
$$

as $\begin{array} { r } { { \mathbb P } ( g _ { Y } ( Y ) = 0 ) = { \mathbb P } ( Y \in \{ g _ { Y } = 0 \} ) = \int _ { \mathcal V } \mathbf { 1 } _ { \{ g _ { Y } = 0 \} } { \mathbb P } ( Y \in \mathrm { d } y ) = \int _ { \{ g _ { Y } = 0 \} } g _ { Y } \mathrm { d } \nu _ { Y } = 0 } \end{array}$ . Note that we can write 

$$
\mathbb {P} (U \in A, Y \in F) = \int_ {F \cap \{g _ {Y} > 0 \}} \int_ {A} g _ {U, Y} (u, y) \mathrm{d} \nu_ {U} (u) \mathrm{d} \nu_ {Y} (y).
$$

This and the statement above imply 

$$
\mathbb {P} (U \in A | Y = y) g _ {Y} (y) = \int_ {A} g _ {U, Y} (u, y) \mathrm{d} \nu_ {U} (u) \qquad (\mathbb {P} (Y \in \cdot) \text {-a.s.}).
$$

Hence, we have 

$$
\mathbb {P} (U \in A | Y = y) = \int_ {A} \frac {g _ {U , Y} (u , y)}{g _ {Y} (y)} \mathrm{d} \nu_ {U} (u) \quad (\mathbb {P} (Y \in \cdot) \text {-a.s.}).
$$

This proves our statement about $\mathbb { P } ( U \in \cdot | Y = y )$ the reverse statement can be shown analoguously. □ 

Definition 6.3.11. Let $g _ { U } , g _ { Y } , g _ { U , Y } , g _ { U | Y = y } , g _ { Y | U = u }$ be the probability density functions in Theorem 6.3.10. We define 

$g _ { U } \ ( r e s p . \ g _ { Y } )$ to be the marginal probability density of $U ~ ( r e s p . ~ Y )$ 

$g _ { U , Y }$ to be the joint probability density of U and $Y ,$ 

$g _ { U } | Y { = } y$ to be the conditional density of U given $Y = y ,$ and 

$g _ { Y \mid U = u }$ to be the conditional density of Y given $U = u$ 

## 6.4 Bayesian statistics

We are now ready to, first, fit our inverse problem into a statistical framework and, second, determine the posterior measure 

## 6.4.1 Statistical models

Definition 6.4.1. Let $x , y$ be separable Banach spaces. We refer to $\mathcal { X }$ as parameter space and to as data space. Let $\mathcal { P } : = \{ M ( \cdot | u ) : u \in \mathcal { X } \}$ , where M is a Markov kernel from $( \mathcal { X } , B \mathcal { X } ) \ t o \ ( \overline { { \mathcal { V } , B \mathcal { V } ) } }$ . The tuple $( \mathcal { V } , \mathcal { P } )$ is called statistical model. The statistical model is called parametric, if is a subset of a Euclidean vector space, and non-parametric, otherwise. 

After defining statistical models, we should comment on their purpose. 

Remark 6.4.2. Let $u ^ { * } \in \mathcal { X }$ be some parameter, let $Y \sim M ( \cdot | u ^ { * } )$ , and let $y$ be a realisation of $Y$ . Statistical methods aim to find $u ^ { * } \in \mathcal { X }$ based on the realisation $y .$ The probability measure $M ( \cdot | u ^ { * } )$ is called data-generating distribution. 

Now, we give an example for a parametric statistical model. 

Example 6.4.3. We are given five independent realisations $y = ( 0 . 2 , - 0 . 3 2 , 0 . 8 , 1 . 2 , - 0 . 4 )$ of a one dimensional Gaussian random variable with variance $\sigma ^ { 2 } = 1$ . We do not know the mean of the random variable. Given $y ,$ we want to identify the mean. The statistical model associated with this task is given by: 

$$
(\mathcal {Y}, \mathcal {P}) := \left(\mathbb {R} ^ {5}, \left\{\mathrm{N} (u, 1) ^ {\otimes 5}: u \in \mathbb {R} \right\}\right).
$$

We can sometimes represent a statistical model in terms of a conditional density, the so-called likelihood. 

Definition 6.4.4. Let $( \mathcal { V } , \mathcal { P } )$ be a statistical model and let $L : ( \mathcal { X } \times \mathcal { Y } , \mathcal { B } \mathcal { X } \otimes \mathcal { B } \mathcal { Y } )  ( \mathbb { R } , \mathcal { B } \mathbb { R } )$ such that 

$$
\mathcal {P} := \left\{\mathcal {B Y} \ni F \mapsto \int_ {F} L (y | u) \mathrm{d} \mu (y): u \in \mathcal {X} \right\},
$$

for some measure $\mu$ on $( \mathcal { V } , B \mathcal { V } )$ . We refer to L as (data) likelihood. 

Note that the likelihood is a conditional density $\begin{array} { r } { L = g _ { Y | U = u } , } \end{array}$ for some random variable $U .$ . It informs us about the likelihood of observing a data set given that we assume it was sampled from $M ( \cdot | u )$ 

Example 6.4.5. Let $\mathcal { A } : ( \mathcal { X } , \mathcal { B } \mathcal { X } )  ( \mathcal { Y } , \mathcal { B } \mathcal { Y } )$ be a measurable operator. Moreover, let µ<sub>noise</sub> be a probability measure on (<sub>Y</sub>, <sub>BY</sub>). We consider the inverse problem of identifying $u \in X$ , where 

$$
\mathcal {A} (u) + N = f _ {n}
$$

with $N \sim \mu _ { \mathrm { n o i s e } }$ . We can now represent this inverse problem by a statistical model: 

$$
(\mathcal {Y}, \mathcal {P}) := (\mathcal {Y}, \{\mu_ {\text { noise }} (\cdot - \mathcal {A} (u)): u \in \mathcal {X} \}).
$$

The data set $f _ { n }$ is a realisation of the data-generating distribution $\mu _ { \mathrm { n o i s e } } ( \cdot - \mathcal { A } ( u ^ { * } ) )$ , where $u ^ { * }$ is the true parameter. 

Let $n \in \mathbb { N } , y : = \mathbb { R } ^ { n } , \Gamma \in \mathbb { R } ^ { n \times n }$ be positive definite, and $\mu _ { \mathrm { n o i s e } } : = \mathrm { N } ( 0 , \Gamma )$ . Then, we can represent the statistical model by a likelihood: 

$$
L (y | u) := (2 \pi) ^ {- k / 2} \mathrm{det} (\Gamma) ^ {- 1 / 2} \exp \left(- \frac {1}{2} \| \Gamma^ {- 1 / 2} (y - \mathcal {A} (u)) \| ^ {2}\right),
$$

where $u \in \mathcal X$ and $y \in \mathcal { V }$ 

## 6.4.2 Bayes’ formula

In Bayesian statistics, we model the unknown parameter u as a random variable $U \sim \mu _ { 0 }$ that is distributed according to a prior measure. $\mu _ { 0 }$ reflects our knowledge concerning the parameter u before seeing the data. Moreover, we are given a statistical model $( \mathcal { V } , \mathcal { P } )$ and the according Likelihood $L ,$ which is a conditional density $f _ { Y \mid U = u } .$ We aim to invert $\mathbb { P } ( Y \in \cdot | U = \cdot )$ to $\mathbb { P } ( U \in \cdot | Y = \cdot )$ . The conditional measure $\mathbb { P } ( U \in \cdot | Y = \cdot )$ is the updated prior $\mathbb { P } ( U \in \cdot ) : = \mu _ { 0 }$ . This updating/inversion process uses on Bayes’ formula. 

Theorem 6.4.6 (Bayes). Let $U , Y$ be random variables as in Theorem 6.3.10. Then, 

$$
g _ {U | Y = y} (u) = \frac {g _ {Y | U = u} (y) g _ {U} (u)}{g _ {Y} (y)},\tag{6.2}
$$

for $u \in { \mathcal { X } } , \nu _ { U } - a . e .$ . and $y \in { \mathcal { Y } } , { \mathbb { P } } ( Y \in \cdot ) { - a . e }$ . with $g _ { Y } ( y ) > 0$ 

Proof. We need to show that $g _ { Y | U = u } g _ { U } \ = \ g _ { U , Y } , \ \nu _ { U } \otimes \mathbb { P } ( Y \ \in \ \cdot ) \mathrm { - a . e . }$ .. Let $u \in \mathcal X$ with $g _ { U } ( u ) > 0$ . By definition, 

$$
g _ {Y \mid U = u} (y) g _ {U} (u) = \frac {g _ {U , Y} (u , y) g _ {U} (u)}{g _ {U} (u)} = g _ {U, Y} (u, y) \quad ((u, y) \in \{g _ {U} > 0 \} \times \mathcal {Y}, \nu_ {U} \otimes \mathbb {P} (Y \in \cdot) - \text {a.e.}).
$$

Conversely, let $u \in \mathcal X$ , with $g _ { U } ( u ) = 0$ . This implies that 

$$
0 = \int_ {\mathcal {Y}} g (u, y) \mathrm{d} \nu_ {Y} (y).
$$

Then, $g _ { U , Y } ( u , \cdot ) = 0 , \nu _ { Y ^ { - } \mathrm { a . e } }$ . and, thus, also $\mathbb { P } ( Y \in \cdot ) . . . { \mathrm { ~ } } \mathrm { ~ } \mathrm { ~ }$ .. Hence, $g _ { U , Y } = 0 = g _ { Y } | _ { U = u } g _ { U }$ . 

Definition 6.4.7. $Z ( y ) : = g _ { Y } ( y )$ is called (model) evidence or marginal $l i k e l i h o o d ^ { 1 }$ 

$L ( y | u ) : = g _ { Y | U = u } ( y )$ is called (data) likelihood, 

$\mu _ { 0 } : = \mathbb { P } ( U \in \cdot )$ is called prior (measure), 

$\mu _ { \mathrm { p o s t } } : = \mathbb { P } ( U \in \cdot | Y = y )$ is called posterior (measure), and 

In Theorem 6.4.6, we require that $\mu _ { 0 }$ has a probability density function $g _ { U }$ with respect to a measure $\nu _ { U }$ . In practice, $\nu _ { U }$ is often a Lebesgue measure or the counting measure. In some cases, neither of those two is well-defined or a sensible choice, e.g., when dim $\mathcal { X } = \infty$ However, we can always assume that $\nu _ { U } : = \mu _ { 0 }$ . In this case, we obtain the formulation of Stuart [35]: 

$$
\frac {\mathrm{d} \mu_ {\mathrm{post}}}{\mathrm{d} \mu_ {0}} (u) = \frac {L (y | u)}{Z (y)} \quad (u \in \mathcal {X}, \mu_ {0} \text {-a.s.}).
$$

Remark 6.4.8. When defining $\begin{array} { r } { Z ( y ) : = \int L ( y | u ) \mathrm { d } \mu _ { 0 } } \end{array}$ , it is not necessary for $L ( y | u )$ to be correctly normalised. Indeed, we can set $L ( y | u ) : = c \cdot g _ { Y | U = u } ( y )$ , for some constant $c > 0$ The factor c cancels with the same factor in $Z ( y )$ . However, then we have $Z ( y ) \neq f _ { Y } ( y )$ and call $Z ( y )$ normalising constant. 

# Bayesian inverse problems and well-posedness

In this chapter, we will define Bayesian inverse problems and study their well-posedness. Well-posedness requires existence and uniqueness of the posterior measure, as well as its stability with respect to marginal perturbations in the data. 

## 7.1 Bayesian inverse problems

A posterior measure is a conditional probability distribution and as such only for almost every data set uniquely defined. In the following, we will always pick one representing Markov kernel out of the set of kernels satisfying the equation in Theorem 6.3.5. We do so, by fixing the definition of the likelihood to a specific measurable function $\mathcal { X } \times \mathcal { Y }  \mathbb { R }$ and defining the posterior to satisfy Bayes’ formula with this likelihood. 

We first introduce some further notation. 

Definition 7.1.1. Let $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ be some measurable space. We define the space of probability measures on $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ by Prob $\left( \Omega ^ { \prime } , { \mathcal F } ^ { \prime } \right) : = \{ \mu : \mu$ is a probability measure on $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } ) \}$ Moreover, for some σ-finite measure ν on $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ , we define Prob $( \Omega ^ { \prime } , { \mathcal { F } } ^ { \prime } , \nu ) : = \{ \mu \in \mathfrak $ $\mathrm { P r o b } ( \Omega , \mathcal { F } ^ { \prime } ) : \mu \ll \nu \}$ 

Definition 7.1.2. Let $\mu _ { 0 } \in \mathrm { P r o b } ( { \mathcal { X } } , B { \mathcal { X } } )$ and $L : ( \mathcal { X } \times \mathcal { Y } , B \mathcal { X } \otimes B \mathcal { Y } ) \to ( \mathbb { R } , B \mathbb { R } )$ be a measureable function. We define the Bayesian inverse problem $( B I P )$ with prior $\mu _ { 0 }$ and likelihood L, to be the problem of finding $\mu _ { \mathrm { p o s t } } \in \mathrm { P r o b } ( \mathcal { X } , B \mathcal { X } )$ with 

$$
\frac {\mathrm{d} \mu_ {\mathrm{post}}}{\mathrm{d} \mu_ {0}} (u) = \frac {L (f _ {n} | u)}{\int_ {\mathcal {X}} L (f _ {n} | u) \mathrm{d} \mu_ {0} (u)} \qquad \qquad (u \in \mathcal {X}; \mu_ {0} \text {-a.s.})
$$

for any data set $f _ { n } \in \mathcal { V }$ 

We discussed previously how to construct a likelihood in the ‘classical’ inverse problem setting 

$$
\text { find } u \in \mathcal {X}: \mathcal {A} (u) + n = f _ {n}.
$$

We now allow for much more general likelihood functions; this includes non-additive noise, Poissonian models,... 

Definition 7.1.3. Consider a (BIP) with prior $\mu _ { 0 }$ and likelihood L. Let $P \subseteq \mathrm { P r o b } ( \mathcal { X } , B \mathcal { X } )$ be a space of probability measures and d $: P ^ { 2 }  [ 0 , \infty )$ be a metric on P . A Bayesian inverse problem is (P, d)-well-posed, if 

(i) for all $f _ { n } \in \mathcal { V }$ , the probability measure $\mu _ { \mathrm { p o s t } } \in P$ exists, (existence) 

(ii) for all $f _ { n } \in \mathcal { V }$ , the probability measure $\mu _ { \mathrm { p o s t } } \in P$ is unique, and (uniqueness) 

(iii) the map $\mathcal { V } \ni f _ { n } \mapsto \mu _ { \mathrm { p o s t } } \in P$ is continuous. (stability) 

Existence and uniqueness of the posterior in $P \in \{ \mathrm { P r o b } ( \mathcal { X } , B \mathcal { X } ) , \mathrm { P r o b } ( \mathcal { X } , B \mathcal { X } , \mu _ { 0 } ) \}$ is automatic, if $\textstyle \int _ { \mathcal { X } } L ( f _ { n } | u ) \mathrm { d } \mu _ { 0 } ( u ) \ \in \ ( 0 , \infty )$ . This is, for instance, the case in the following lemma. 

Lemma 7.1.4. Consider a (BIP) with prior µ and likelihood L. Let $L > 0 \ ( \mu _ { 0 } - a . s . )$ and $L ( f _ { n } | \cdot ) \in L ^ { 1 } ( \mathcal { X } , B \mathcal { X } , \mu _ { 0 } )$ for any $f _ { n } \in \mathcal { V }$ . Then, the posterior $\mu _ { \mathrm { p o s t } } \in \mathrm { P r o b } ( \mathcal { X } , B \mathcal { X } , \mu _ { 0 } )$ exists and is unique. 

Proof. We need to show that $\textstyle \int _ { \mathcal { X } } L ( f _ { n } | u ) \mathrm { d } \mu _ { 0 } ( u ) \ \in \ ( 0 , \infty )$ . Upper bound: trivial, since $L ( f _ { n } | \cdot ) \in L ^ { 1 } ( \mathcal { X } , B \mathcal { X } , \mu _ { 0 } )$ . Lower bound: exercise. □ 

Before we can actually speak about stability, we need to discuss metrics on spaces of probability measures. 

## 7.2 Metrics on spaces of probability measures

We consider metrics on subspaces of $\mathrm { P r o b } ( \mathcal { X } , B \mathcal { X } )$ to be able to show stability of the posterior measure with respect to perturbations in the data. We consider two diferent concept: total variation and weak convergence. 

Definition 7.2.1. (i) Let $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ be a measurable space. We define the total variation (TV) distance on $\mathrm { P r o b } ( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ by 

$$
d _ {\mathrm{TV}}: \operatorname{Prob} (\Omega^ {\prime}, \mathcal {F} ^ {\prime}) ^ {2} \to [ 0, \infty), (\mu , \nu) \mapsto \sup _ {F ^ {\prime} \in \mathcal {F} ^ {\prime}} | \mu (F ^ {\prime}) - \nu (F ^ {\prime}) |
$$

(ii) Let Ω<sup>0</sup> be a topological space and $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } ) : = ( \Omega ^ { \prime } , B \Omega ^ { \prime } )$ . Let $( \mu _ { n } ) _ { n \in \mathbb { N } } \in \mathrm { P r o b } ( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } ) ^ { \mathbb { N } }$ and $\mu \in \mathrm { P r o b } ( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ . We say $\mu _ { n }  \mu$ weakly, as $n  \infty , i f$ 

$$
\lim _ {n \rightarrow \infty} \int_ {\Omega^ {\prime}} g \mathrm{d} \mu_ {n} = \int_ {\Omega^ {\prime}} g \mathrm{d} \mu ,
$$

for any $g : ( \Omega ^ { \prime } , \mathcal { B } \Omega ^ { \prime } )  ( \mathbb { R } , \mathcal { B } \mathbb { R } )$ that is continuous and bounded. 

Remark 7.2.2. Weak convergence of measures on Prob( , ) can be represented by the (L´evy)-Prokhorov metric $d _ { \mathrm { L P } }$ . See [29] for details. Hence, when referring to the topology induced by weak convergence, we will usually speak about the metric space $( \mathrm { P r o b } ( \mathcal { X } , \mathcal { B X } ) , d _ { \mathrm { L P } } )$ , but not actually employ the (L´evy)-Prokhorov metric. 

We end this section with two more results about the total variation distance and weak convergence. First, we show that if a sequence of measures converges in the total variation distance, it converges weakly as well. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/7e88d0c423407ceac086d590188cb8fe527ecd23a30d3eeaf938f2ee3147bbf3.jpg)


Lemma 7.2.3. Let Ω<sup>0</sup> be a topological space and $( \Omega ^ { \prime } , { \mathcal F } ^ { \prime } ) : = ( \Omega ^ { \prime } , B \Omega ^ { \prime } )$ . Let $( \mu _ { n } ) _ { n \in \mathbb { N } } \in$ Prob(Ω<sup>0</sup>, <sup>0</sup>)<sup>N</sup> and $\mu \in \mathrm { P r o b } ( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ . Then 

$$
\lim _ {n \to \infty} d _ {\mathrm{TV}} (\mu_ {n}, \mu) = 0 \Longrightarrow \mu_ {n} \to \mu , w e a k l y a s n \to \infty .
$$

The converse statement $( \phantom { + } ^ { 6 6 } \Leftarrow 7 )$ is in general not true. 

Proof. Exercise. 

Second, we give a representation of the total variation distance of two measures having a density with respect to a third measure. 

Lemma 7.2.4. Let $\mu , \nu \in \mathrm { P r o b } ( \Omega , { \mathcal F } )$ and $\rho$ be a σ-finite measure with $\mu , \nu \ll \rho$ . Then, $\begin{array} { r } { d _ { \mathrm { T V } } ( \mu , \nu ) = \frac { 1 } { 2 } \int _ { \Omega } \left| \frac { \mathrm { d } \mu } { \mathrm { d } \rho } - \frac { \mathrm { d } \nu } { \mathrm { d } \rho } \right| \mathrm { d } \rho . } \end{array}$ 

Proof. Exercise. 

Note that this result is independent of the measure $\rho .$ . As a trivial dominating measure, one can always choose $\rho : = \mu + \nu$ 

## 7.3 Stability

We now give a set of assumptions under which we can prove $( P , d )$ -well-posedness, as defined in Definition 7.1.3, where $( P , d )$ refers to the space of probability measure on with $\mu _ { 0 ^ { - } }$ density and either total variation distance or weak convergence. 

Assumption 7.3.1. Given a (BIP) with prior $\mu _ { 0 }$ and likelihood L. Let the following assumptions hold for $u \in \mathcal { X } \ \mu _ { 0 } { \mathrm { - a . s } }$ . and $f _ { n } \in \mathcal { V }$ 

(A1) $L ( \cdot | u )$ is a strictly positive probability density function, 

(A2) $L ( f _ { n } | \cdot ) \in L ^ { 1 } ( \mathcal { X } , { B \mathcal { X } } , \mu _ { 0 } )$ 

(A3) some $h \in L ^ { 1 } ( \mathcal { X } , B \mathcal { X } , \mu _ { 0 } )$ exists, such that $L ( f _ { n } ^ { \prime } | \cdot ) \leqslant h$ for all $f _ { n } ^ { \prime } \in \mathcal { V }$ , and 

(A4) $L ( \cdot | u )$ is continuous. 

We now briefly comment on the assumptions. (A1) and (A2) were already required in Lemma 7.1.4. In (A3) we now not only ask for boundedness of the integral of the likelihood, but for its uniform boundedness by an integrable function g. This is for instance the case, if the likelihood is bounded by a constant (as $\mu _ { 0 }$ is a probability measure). In (A4) we ask for continuity in the data. (Continuity in the parameter is not required!) In inverse problems, this is true for a large number of noise distributions. 

Before proving well-posedness under Assumptions $( \mathrm { A } 1 ) – ( \mathrm { A } 4 )$ we cite a fundamental measure-theoretic result which is needed for the proof. 

Theorem 7.3.2 (Dominated Convergence Theorem (DCT; Lebesgue)). Let $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } , \mu ^ { \prime } )$ be a measure space. Let $g , ( g _ { m } ) _ { m \in \mathbb { N } } , h$ be measurable functions $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } ) \to ( \mathbb { R } , B \mathbb { R } )$ and $h \in$ $L ^ { 1 } ( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } , \mu ^ { \prime } )$ . Moreover, let $| g _ { m } | \leqslant h _ { \mathbf { \lambda } } ( \mu ^ { \prime } { - } a . e . )$ and $g _ { m }  g , \mu ^ { \prime } { - } a . e . a s m  \infty$ . Then, $g , g _ { m } \in L ^ { 1 } ( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } , \mu ^ { \prime } )$ and 

$$
\lim _ {m \to \infty} \int_ {\Omega^ {\prime}} g _ {m} \mathrm{d} \mu^ {\prime} = \int_ {\Omega^ {\prime}} g \mathrm{d} \mu^ {\prime}.
$$

Proof. Can be proved using monotonic convergence theorem (Proposition 6.2.11.2). See, $\mathrm { e . g . }$ , Theorem 1.6.9 [6] for a proof using Fatou’s Lemma. □ 

Remark 7.3.3. The DCT describes a case in which we are allowed to “exchange integral and limit”. The statement reads 

$$
\lim _ {m \to \infty} \int_ {\Omega^ {\prime}} g _ {m} \mathrm{d} \mu^ {\prime} = \int_ {\Omega^ {\prime}} \lim _ {n \to \infty} g _ {m} \mathrm{d} \mu^ {\prime}.
$$

Equivalently, we could say it describes cases in which the integral as a functional of the integrand is continuous. 

Theorem 7.3.4 (Well-posedness). Given a (BIP) with prior $\mu _ { 0 }$ and likelihood L that satisfies Assumptions $( A 1 ) { - } ( A 4 )$ . Moreover, let $P = \mathrm { P r o b } ( \mathcal { X } , B \mathcal { X } , \mu _ { 0 } )$ and $d \in \{ d _ { \mathrm { T V } } , d _ { \mathrm { L P } } \}$ Then, the (BIP) is (P, d)-well-posed. 

Proof. 1. Note that (A1), (A2) already imply existence and uniqueness by Lemma 7.1.4. In the remainder of the proof, we focus on showing continuity in the total variation distance. Continuity in weak convergence is then implied by Lemma 7.2.3. Indeed, we show that for all $f _ { n } \in \mathcal { V }$ and all $( f _ { n } ^ { ( m ) } ) _ { m \in \mathbb { N } } \in \mathcal { V } ^ { \mathbb { N } }$ , with $\begin{array} { r } { \operatorname* { l i m } _ { m  \infty } f _ { n } ^ { ( m ) } = f _ { n } } \end{array}$ , we have 

$$
\int_ {\mathcal {X}} \left| \frac {L (f _ {n} | u)}{Z (f _ {n})} - \frac {L (f _ {n} ^ {(m)} | u)}{Z (f _ {n} ^ {(m)})} \right| \mathrm{d} \mu_ {0} (u) \to 0 \qquad (m \to \infty).
$$

where $\begin{array} { r } { Z ( f _ { n } ) : = \int _ { \mathcal { X } } L ( f _ { n } | u ) \mathrm { d } \mu _ { 0 } ( u ) } \end{array}$ . By Lemma 7.2.4, this implies continuity of the posterior measure in the total variation distance. 

2. We first show that $\mathcal { V } \ni f _ { n } \mapsto L ( f _ { n } | \cdot ) \in L ^ { 1 } ( \mathcal { X } , B \mathcal { X } , \mu _ { 0 } )$ is continuous. Let $f _ { n } \in \mathcal { V }$ and $( f _ { n } ^ { ( m ) } ) _ { m \in \mathbb { N } } \in \mathcal { V } ^ { \mathbb { N } }$ , with lim $_ { 1 m  \infty } f _ { n } ^ { ( m ) } = f _ { n }$ . Note that 

$$
\lim _ {m \to \infty} \int_ {\mathcal {X}} \left| L (f _ {n} ^ {(m)} | u) - L (f _ {n} | u) \right| \mathrm{d} \mu_ {0} (u) = \int_ {\mathcal {X}} \lim _ {m \to \infty} \left| L (f _ {n} ^ {(m)} | u) - L (f _ {n} | u) \right| \mathrm{d} \mu_ {0} (u),
$$

due to the DCT since the integrand is bounded below by 0 and above by $2 h \in L ^ { 1 } ( \mathcal { X } , B \mathcal { X } , \mu _ { 0 } )$ Due to the continuity of $L ( \cdot | u )$ (required in (A4)), we have then 

$$
\lim _ {m \to \infty} \int_ {\mathcal {X}} \left| L (f _ {n} ^ {(m)} | u) - L (f _ {n} | u) \right| \mathrm{d} \mu_ {0} (u) = 0.
$$

With the same argument, we can show that $f _ { n } \mapsto Z _ { n } ( f _ { n } )$ is continuous. 

3. The rest of the proof is similar to showing continuity of the quotient of two continuous functions. Let $f _ { n } \in \mathcal { V }$ and $( f _ { n } ^ { ( m ) } ) _ { m \in \mathbb { N } } \in \mathcal { V } ^ { \mathbb { N } }$ , with $\begin{array} { r } { \operatorname* { l i m } _ { m  \infty } f _ { n } ^ { ( m ) } = f _ { n } } \end{array}$ . Then 

$$
\begin{array}{l}\int_ {\mathcal {X}} \left| \frac {L (f _ {n} | u)}{Z (f _ {n})} - \frac {L (f _ {n} ^ {(m)} | u)}{Z (f _ {n} ^ {(m)})} \right| \mathrm{d} \mu_ {0} (u)\\\leqslant Z (f _ {n}) ^ {- 1} \underbrace {\int_ {\mathcal {X}} \left| L (f _ {n} ^ {(m)} | u) - L (f _ {n} | u) \right| \mathrm{d} \mu_ {0} (u)} _ {\rightarrow 0 (m \rightarrow \infty)} + \int_ {\mathcal {X}} L (f _ {n} ^ {(m)} | u) \mathrm{d} \mu_ {0} (u) \underbrace {| Z (f _ {n}) ^ {- 1} - Z (f _ {n} ^ {(m)}) ^ {- 1} |} _ {\rightarrow 0 (m \rightarrow \infty)}\end{array}
$$

and the terms that do not converge to 0 are bounded. 

To illustrate the generality of this result, we now study again the inverse problem from Example 6.4.5. 

Corollary 7.3.5. Let $k \in \mathbb N$ and $( \mathcal { V } , B \mathcal { V } ) : = ( \mathbb { R } ^ { k } , B \mathbb { R } ^ { k } )$ and $\Gamma \in \mathbb { R } ^ { k \times k }$ be symmetric, positive definite. Moreover, let $\mathcal { A } : ( \mathcal { X } , B \mathcal { X } )  ( \mathcal { Y } , B \mathcal { Y } )$ be some function. Consider the (BIP) with some prior $\mu _ { 0 } \in \operatorname { P r o b } ( \mathcal { X } , B \mathcal { X } )$ and likelihood 

$$
L \left(f _ {n} | u\right) := (2 \pi) ^ {- k / 2} \det (\Gamma) ^ {- 1 / 2} \exp \left(- \frac {1}{2} \| \Gamma^ {- 1 / 2} \left(f _ {n} - \mathcal {A} (u)\right) \| ^ {2}\right) \quad (u \in \mathcal {X}, f _ {n} \in \mathcal {Y})
$$

Then, the (BIP) is $( P , d ) \mathrm { - w e l l - p o s e d } .$ , with $P = \mathrm { P r o b } ( \mathcal { X } , B \mathcal { X } , \mu _ { 0 } )$ and $d \in \{ d _ { \mathrm { T V } } , d _ { \mathrm { L P } } \}$ 

Proof. Follows trivially from Theorem 7.3.4. 

# Function space priors and Monte Carlo

In this last chapter, we would like to discuss two rather practical topics: 

• In inverse problems, we often consider infinite-dimensional parameter spaces. While we have discussed the well-posedness of Bayesian inverse problems in infinite dimensional setting, it is not clear yet how, e.g., a prior probability measure on such a space can be defined. We will discuss Gaussian prior measures on function spaces, so-called Gaussian random fields. For a more thorough introduction, we refer to the book by Bogachev [11]. 

• In practical situations, we need to approximate the posterior (or integrals with respect to it) numerically. We will discuss Monte Carlo techniques that are suitable for Bayesian inverse problems. Again, for a more thorough discussion of certain aspects, we refer to Agapiou et al. [3], Cotter et al. [16], and Robert and Casella [30]. 

## 8.1 Gaussian measures

We have defined Gaussian measures on (<sup>R</sup>, <sup>R</sup>) in Definition 6.2.13. We now extend this definition to measurable spaces like $( \mathcal { X } , \mathcal { B X } )$ , where is a separable Banach space. In this section, we assume that all Banach and Hilbert spaces are with respect to $\mathbb { R }$ 

Definition 8.1.1. Let $\mu$ be a probability measure on Prob $( \mathcal { X } , B \mathcal { X } )$ and let $U \sim \mu$ . We call $\mu$ Gaussian, if for all $\ell \in X ^ { * }$ , there exist $m \in \mathbb { R } , \sigma \geqslant 0$ , such that 

$$
\mathbb {P} (\langle \ell , U \rangle \in \cdot) = \mathrm{N} (m, \sigma^ {2}).
$$

Moreover, we define the mean of µ by $a _ { \mu } \in \mathcal X ^ { * * }$ , given by 

$$
a _ {\mu} (\ell) = \int_ {\mathcal {X}} \langle \ell , u \rangle \mathrm{d} \mu (u) \qquad (\ell \in \mathcal {X} ^ {*})
$$

and the covariance operator of µ by $R _ { \mu } : \mathcal { X } ^ { * } \to \mathcal { X } ^ { * * }$ , where 

$$
R _ {\mu} (\ell) (\ell^ {\prime}) = \int_ {\mathcal {X}} \left(\langle \ell , u \rangle - a _ {\mu} (\ell)\right) \left(\langle \ell^ {\prime}, u \rangle - a _ {\mu} (\ell^ {\prime})\right) \mathrm{d} \mu (u) \qquad (\ell , \ell^ {\prime} \in \mathcal {X} ^ {*}).
$$

If is a function space, we call U Gaussian random field. 

This definition does not immediately lead to a construction of a Gaussian measure on a general separable Banach space. There are two cases, in which we have techniques to construct a Gaussian measure on $x ;$ those are $\mathbb { R } ^ { k }$ and separable Hilbert spaces. In finite dimensions, one can define a Gaussian measure in terms of a probability density function with respect to the product of a Lebesgue measure and a Dirac measure. On a separable Hilbert space, we can construct a series expansion, the so-called Karhunen-Lo`eve expansion. 

Definition 8.1.2. Let be a separable Hilbert space and $C : \mathcal { X }  \mathcal { X }$ be a compact, self adjoint linear operator. Moreover, let $( \lambda _ { i } , \varphi _ { i } ) _ { i \in \mathbb { N } } \in ( \mathbb { R } \times \mathcal { X } ) ^ { \mathbb { N } }$ be the eigenpairs of C sorted decreasingly with respect to the absolute value of the eigenvalue and $( \varphi _ { i } ) _ { i \in \mathbb { N } }$ is orthonormal. Then, we can represent 

$$
C x = \sum_ {i = 1} ^ {\infty} \lambda_ {i} \langle x, \varphi_ {i} \rangle_ {\mathcal {X}} \varphi_ {i} \quad (x \in \mathcal {X}),
$$

see also Theorem 2.2.4. C is a trace class operator, $i f \left( \lambda _ { i } \right) _ { i \in \mathbb { N } } \in \ell ^ { 1 }$ 

Proposition 8.1.3. Let  be a separable Hilbert space and $C : \mathcal { X }  \mathcal { X }$ be a linear operator that is $s e l f \mathrm { - } a d j o i n t .$ , non-negative, and trace class. We denote the eigenpairs of C by $( \lambda _ { i } , \varphi _ { i } ) _ { i \in \mathbb { N } } \in ( \mathbb { R } \times \mathcal { X } ) ^ { \mathbb { N } } ,$ the eigenvalues are sorted decreasingly and $( \varphi _ { i } ) _ { i \in \mathbb { N } }$ is orthonormal. Finally, let $m \in { \mathcal { X } }$ and $\xi \sim \mathrm { N } ( 0 , 1 ^ { 2 } ) ^ { \otimes \mathbb { N } }$ . Then, 

$$
U := m + \sum_ {i = 1} ^ {\infty} \sqrt {\lambda_ {i}} \xi_ {i} \varphi_ {i}
$$

is distributed according to a Gaussian measure with mean m and covariance operator $C$ Proof. Let $k \in \mathbb N$ and $\begin{array} { r } { U _ { k } : = m + \sum _ { i = 1 } ^ { k } \sqrt { \lambda _ { i } } \xi _ { i } \varphi _ { i } } \end{array}$ . Moreover, let $x \in \mathcal { X }$ and $x _ { i } : = \langle { x , \varphi _ { i } } \rangle _ { \mathcal { X } }$ for $i \in \mathbb N$ . We first study the distribution of $\langle x , U \rangle _ { \mathcal { X } }$ 

$$
\begin{array}{l} \langle x, U _ {k} \rangle_ {\mathcal {X}} = \left\langle x, m + \sum_ {i = 1} ^ {k} \sqrt {\lambda_ {i}} \xi_ {i} \varphi_ {i} \right\rangle_ {\mathcal {X}} \\ \qquad = \langle x, m \rangle_ {\mathcal {X}} + \left\langle x, \sum_ {i = 1} ^ {k} \sqrt {\lambda_ {i}} \xi_ {i} \varphi_ {i} \right\rangle_ {\mathcal {X}} \\ \qquad = \langle x, m \rangle_ {\mathcal {X}} + \sum_ {i = 1} ^ {k} \sqrt {\lambda_ {i}} \langle x, \varphi_ {i} \rangle_ {\mathcal {X}} \xi_ {i} \\ \qquad = \langle x, m \rangle_ {\mathcal {X}} + \sum_ {i = 1} ^ {k} \underbrace {\sqrt {\lambda_ {i}} x _ {i} \xi_ {i}} _ {\sim N (0, \lambda_ {i} x _ {i} ^ {2})} \end{array}
$$

converges weakly to the Gaussian distribution $\begin{array} { r } { \mathrm { ~ N } (  x , m  _ { \mathcal { X } } , \sum _ { i = 1 } ^ { \infty } \lambda _ { i } x _ { i } ^ { 2 } ) ( k  \infty ) } \end{array}$ , if the sum $\textstyle \sum _ { i = 1 } ^ { \infty } \lambda _ { i } x _ { i } ^ { 2 }$ is finite. (This can be shown with the Fourier transform of Gaussian measures, as the $( \xi _ { i } ) _ { i \in \mathbb { N } }$ are mutually independent). By assumption, we have $( \lambda _ { i } ) _ { i \in \mathbb { N } } \in \ell ^ { 1 }$ and also $( x _ { i } ^ { 2 } ) _ { i \in \mathbb { N } } \in \ell ^ { 1 }$ , since $\textstyle \sum _ { i = 1 } ^ { \infty } x _ { i } ^ { 2 } = \| x \| _ { \mathcal { X } } ^ { 2 } < \infty$ . Hence, also $\textstyle \sum _ { i = 1 } ^ { \infty } \lambda _ { i } x _ { i } ^ { 2 } < \infty$ 

Next, we show that U takes values in with probability one, i.e. $\mathbb { P } ( \| U \| _ { \mathcal { X } } < \infty ) = 1$ By Parseval’s identity, we have 

$$
\| U \| _ {\mathcal {X}} = \sum_ {i = 1} ^ {\infty} | \langle U, \varphi_ {i} \rangle_ {\mathcal {X}} | ^ {2} = \sum_ {i = 1} ^ {\infty} \lambda_ {i} \xi_ {i} ^ {2}
$$

which is almost surely finite by Theorem 1.1.4 of [11], as $( \lambda _ { i } ) _ { i \in \mathbb { N } } \in \ell ^ { 1 }$ 

Now, we look at mean and covariance of U. We have 

$$
\begin{array}{l} a _ {\mu} (x) = \int_ {\mathcal {X}} \langle x, U \rangle \mathrm{d} \mathbb {P} = \langle x, m \rangle_ {\mathcal {X}} + \int_ {\mathbb {R} ^ {\mathbb {N}}} \sum_ {i = 1} ^ {\infty} \sqrt {\lambda_ {i}} x _ {i} \xi_ {i} \mathrm{dN} (0, 1) ^ {\otimes \mathbb {N}} (\xi) \\ \qquad = \langle x, m \rangle_ {\mathcal {X}} + \sum_ {i = 1} ^ {\infty} \sqrt {\lambda_ {i}} x _ {i} \underbrace {\int_ {\mathbb {R} ^ {\mathbb {N}}} \xi_ {i} \mathrm{dN} (0 , 1) ^ {\otimes \mathbb {N}} (\xi)} _ {= 0} \\ \qquad = \langle x, m \rangle_ {\mathcal {X}}, \end{array}
$$

where we used the Fubini-Tonelli theorem to switch infinite sum and integral: Note that 

$$
\sum_ {i = 1} ^ {\infty} \int_ {\mathbb {R} ^ {\mathbb {N}}} x | \sqrt {\lambda_ {i}} x _ {i} \xi_ {i} | \mathrm{dN} (0, 1) ^ {\otimes \mathbb {N}} (\xi) = \sum_ {i = 1} ^ {\infty} \sqrt {\frac {2}{\pi}} \leqslant \sqrt {\frac {2}{\pi}} \| \sqrt {\lambda_ {i}} \| _ {2} \| | x _ {i} | \| _ {2}
$$

by Cauchy-Schwarz. Moreover, the upper bound on the RHS is finite, since $( x _ { i } ) _ { i \in \mathbb { N } } , ( \lambda _ { i } ) _ { i \in \mathbb { N } } \in$ $\ell ^ { \bar { 2 } }$ . Hence, $a _ { \mu } = m$ . Furthermore, we have for $x ^ { \prime } \in \mathcal { X }$ : 

$$
\begin{array}{l} R _ {\mu} (x) (x ^ {\prime}) = \int_ {\mathcal {X}} \left(\langle u, x \rangle_ {\mathcal {X}} - a _ {\mu} (x)\right) \left(\langle u, x ^ {\prime} \rangle_ {\mathcal {X}} - a _ {\mu} (x ^ {\prime})\right) \mathrm{d} \mu (u) \\ = \int_ {\mathbb {R} ^ {\mathbb {N}}} \left\langle x, \sum_ {i = 1} ^ {\infty} \sqrt {\lambda_ {i}} \xi_ {i} \varphi_ {i} \right\rangle_ {\mathcal {X}} \left\langle \sum_ {j = 1} ^ {\infty} \sqrt {\lambda_ {j}} \xi_ {j} \varphi_ {j}, x ^ {\prime} \right\rangle_ {\mathcal {X}} \mathrm{dN} (0, 1) ^ {\otimes \mathbb {N}} (\xi) \\ = \int_ {\mathbb {R} ^ {\mathbb {N}}} \sum_ {i = 1} ^ {\infty} \sum_ {j = 1} ^ {\infty} \sqrt {\lambda_ {i}} \sqrt {\lambda_ {j}}   \langle x, \varphi_ {i} \rangle_ {\mathcal {X}}   \xi_ {i} \xi_ {j}   \big \langle \varphi_ {j}, x ^ {\prime} \big \rangle_ {\mathcal {X}}   \mathrm{dN} (0, 1) ^ {\otimes \mathbb {N}} (\xi) \\ = \sum_ {i = 1} ^ {\infty} \sum_ {j = 1} ^ {\infty} \sqrt {\lambda_ {i}} \sqrt {\lambda_ {j}}   \langle x, \varphi_ {i} \rangle_ {\mathcal {X}}   \underbrace {\int_ {\mathbb {R} ^ {\mathbb {N}}} \xi_ {i} \xi_ {j} \mathrm{dN} (0 , 1) ^ {\otimes \mathbb {N}} (\xi)} _ {= \mathbf {1} _ {\{j \}} (i)}   \big \langle \varphi_ {j}, x ^ {\prime} \big \rangle_ {\mathcal {X}} \\ = \sum_ {i = 1} ^ {\infty} \lambda_ {i}   \langle x, \varphi_ {i} \rangle_ {\mathcal {X}}   \big \langle \varphi_ {i}, x ^ {\prime} \big \rangle_ {\mathcal {X}} = \langle x, C x ^ {\prime} \rangle_ {\mathcal {X}}, \end{array}
$$

where we could remove the sum over $j$ above due to mutual independence of the $\xi _ { i } , \xi _ { j }$ with $i \neq j$ . We exchanged sums and integral again using the Fubini-Tonelli theorem: 

$$
\begin{array}{l} \sum_ {i = 1} ^ {\infty} \sum_ {j = 1} ^ {\infty} \int_ {\mathbb {R} ^ {\mathbb {N}}} | \sqrt {\lambda_ {i}} \sqrt {\lambda_ {j}} \left\langle x, \varphi_ {i} \right\rangle_ {\mathcal {X}} \xi_ {i} \xi_ {j} \left\langle \varphi_ {j}, x ^ {\prime} \right\rangle_ {\mathcal {X}} | \mathrm{dN} (0, 1) ^ {\otimes \mathbb {N}} (\xi) \\ = \sum_ {i = 1} ^ {\infty} \sum_ {j = 1} ^ {\infty} | \sqrt {\lambda_ {i}} \sqrt {\lambda_ {j}} \left\langle x, \varphi_ {i} \right\rangle_ {\mathcal {X}} \left\langle \varphi_ {j}, x ^ {\prime} \right\rangle_ {\mathcal {X}} | \cdot \frac {2}{\pi} \\ = \sum_ {i = 1} ^ {\infty} | \sqrt {\lambda_ {i}} \left\langle x, \varphi_ {i} \right\rangle_ {\mathcal {X}} | \sum_ {j = 1} ^ {\infty} | \sqrt {\lambda_ {j}} \left\langle \varphi_ {j}, x ^ {\prime} \right\rangle_ {\mathcal {X}} | \cdot \frac {2}{\pi}, \end{array}
$$

which is again finite, as $( x _ { i } ) _ { i \in \mathbb { N } } , ( x _ { i } ^ { \prime } ) _ { i \in \mathbb { N } } , ( \lambda _ { i } ) _ { i \in \mathbb { N } } \in \ell ^ { 2 }$ 

Definition 8.1.4. The expansion 

$$
m + \sum_ {i = 1} ^ {\infty} \sqrt {\lambda_ {i}} \xi_ {i} \varphi_ {i}
$$

in Proposition $8 . 1 . 3$ is called Karhunen–Lo`eve expansion $( K L E )$ . In the same proposition, we denote $\mu = : \mathrm { N } ( m , C )$ 

We can understand the KLE as the function space version of a principal component analysis. Indeed, random fields are often discretised by representing them as a KLE and truncating the expansion. We now study two examples of Gaussian random fields in $\mathcal { L } ^ { 2 }$ 

Example 8.1.5 (Gaussian random fields in 2 dimensions). Let $D = [ 0 , 1 ] ^ { 2 } , \chi : = \mathcal { L } ^ { 2 } ( D , \mathcal { B } D , \lambda _ { 2 } )$ ， $\ell > 0$ , and $\sigma ^ { 2 } \geqslant 0$ . We define the exponential covariance function 

$$
c _ {\exp} (x, y) := \sigma^ {2} \exp \left(- \frac {\| x - y \| _ {2}}{\ell}\right) \qquad (x, y \in D)
$$

and the Gaussian covariance function 

$$
c _ {\mathrm{N}} (x, y) := \sigma^ {2} \exp \left(- \frac {\| x - y \| _ {2} ^ {2}}{2 \ell^ {2}}\right) \qquad (x, y \in D).
$$

The parameter $\ell$ is called correlation length, $\sigma ^ { 2 }$ is called pointwise variance. We can now define the associated covariance operators for $c \in \{ c _ { \mathrm { e x p } } , c _ { \mathrm { N } } \}$ , by 

$$
C: \mathcal {X} \to \mathcal {X}, \varphi \mapsto \int_ {D} \varphi (x) c (x, \cdot) \mathrm{d} \lambda_ {2} (x).
$$

Well-definedness of these covariance operators can be shown with Mercer’s Theorem. In Figure 8.1, we show discretised samples of Gaussian random fields with both covariance functions and $\ell \in \{ 0 . 0 5 , 0 . 1 , 1 \}$ . The random fields have been discretised by a $1 0 0 ^ { 2 } .$ -dimensional piecewise-constant finite element approximation of the eigenpairs of the respective covariance operator. 

## 8.2 Monte Carlo techniques

## 8.2.1 Standard Monte Carlo

Monte Carlo techniques aim at approximating integrals of the form 

$$
\overline {{g}} := \int_ {\mathcal {X}} g \mathrm{d} \mu ,
$$

where $\mu$ is a probability distribution on ( , ) and $g : ( \mathcal { X } , \mathcal { B X } )  ( { \mathbb R } , \mathcal { B } { \mathbb R } )$ is an integrable function. Standard Monte Carlo approaches this problem by generating independent samples $U _ { 1 } , U _ { 2 } , \dots \sim \mu$ and computing the estimator 

$$
\widehat {g} _ {M} := \frac {1}{M} \sum_ {m = 1} ^ {M} g (U _ {m}),
$$

for some $M \in \mathbb { N }$ . Alternatively, we can understand Monte Carlo as a technique allowing us to approximate the probability measure $\mu$ by the probability measure 

$$
\widehat {\mu} _ {M} := \frac {1}{M} \sum_ {m = 1} ^ {M} \delta (\cdot - U _ {m}).
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/b626b668baea6e76b586c35dfefba6426c46d55124c235958e2d8ca4004adf6e.jpg)



Figure 8.1: Each row represents four samples from the Gaussian random field with mean $m = 0$ and the following covariance operators (from top to bottom): exponential with $\ell = 0 . 0 5$ , exponential with $\ell = 0 . 1$ , exponential with $\ell = 1$ , Gaussian with $\ell = 0 . 0 5$ , Gaussian with $\ell = 0 . 1$ , and Gaussian with ` = 1.


The Monte Carlo estimator can be analysed using the (strong) law of large numbers. We know that 

$$
\widehat {g} _ {M} \to \overline {{g}} \qquad (M \to \infty , \mathbb {P} \text {-a.s.}).
$$

If in addition $\begin{array} { r } { \mathrm { V a r } _ { \mu } ( g ) : = \int g ^ { 2 } \mathrm { d } \mu - \left( \int g \mathrm { d } \mu \right) ^ { 2 } < \infty } \end{array}$ , we obtain the following convergence rate 

$$
\sqrt {\mathbb {E} \left[ (\widehat {g} _ {M} - \overline {{g}}) ^ {2} \right]} = \frac {\sqrt {\mathrm{Var} _ {\mu} (g)}}{\sqrt {M}}.
$$

When thinking about standard algorithms for numerical quadrature (Gauss quadrature, Simpson’s rule,...) the rate $O ( M ^ { - 1 / 2 } )$ appears to be quite slow. A composite Simpson’s rule, e.g., for a very smooth function over $\mathcal { X } : = [ 0 , 1 ]$ has an absolute error of $O ( M ^ { - 4 } )$ Its advantage over classical methods is that the rate is independent of the smoothness of the function and the dimension of its domain. Hence, Monte Carlo methods are especially useful in problems that are non-smooth and/or high-dimensional. 

Unfortunately, standard Monte Carlo techniques are usually unsuitable for the approximation of posterior measures in Bayesian inverse problems: we are not able to sample independently from the posterior measure. Ideas: 

• sample dependently from $\mu _ { \mathrm { p o s t } } \ ( $ Markov chain Monte Carlo; this lecture) or 

• sample independently from a diferent measure and correct by choosing unequal weights 

$$
\widehat {g} _ {M} := \sum_ {m = 1} ^ {M} w _ {m} g (U _ {m}),
$$

with $w _ { m } \neq 1 / M , m = 1 , \dots , M$ ( Importance Sampling; exercise sheet 4) 

## Markov chain Monte Carlo

In Markov chain Monte Carlo (MCMC), we generate a Markov chain $( U _ { m } ) _ { m \in \mathbb { N } }$ that is stationary with respect to the posterior measure $\mu _ { \mathrm { p o s t } }$ and Harris recurrent. In this case, we also have a law of large numbers 

$$
\frac {1}{M} \sum_ {m = 1} ^ {M} g (U _ {m}) \rightarrow \int_ {\mathcal {X}} g \mathrm{d} \mu_ {\mathrm{post}} \qquad (M \rightarrow \infty , \mathbb {P} \mathrm{-a.s.}),
$$

for some integrable $g : ( \mathcal { X } , \mathcal { B X } )  ( { \mathbb R } , \mathcal { B } { \mathbb R } )$ ; see [30, Theorem 6.63]. We give a comparison of Monte Carlo and Markov chain Monte Carlo in Figure 8.2. 

In the figure, we see that sampling a Markov chain can be less eficient than independent sampling – making MCMC not appearing very natural just yet. However, it is often easier to generate such a Markov chain than to sample independently from the posterior. In the following, we will first recap some definitions concerning Markov chains. Then, we will introduce the Metropolis–Hastings algorithm and show that is is stationary with respect to our measure of interest; say the posterior measure. We will not discuss ergodicity/Harris recurrence in this short introduction, but refer to the work by Robert and Casella [30]. 

Definition 8.2.1. Let $( U _ { n } ) _ { n = 1 } ^ { \infty }$ be a sequence of -valued random variables - so-called states. $( U _ { n } ) _ { n = 1 } ^ { \infty }$ is called Markov chain, if for any $n \in \mathbb { N }$ 

$$
\mathbb {P} (U _ {n + 1} \in \cdot | U _ {1} = u _ {1}, U _ {2} = u _ {2},..., U _ {n - 1} = u _ {n - 1}, U _ {n} = u _ {n}) = \mathbb {P} (U _ {n + 1} \in \cdot | U _ {n} = u _ {n})\tag{8.1}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/6095286acffe795f968e4d84eb604c236e3b9f7ca7a7b10113f661b16d975eca.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/c8bc54b8-fb4a-4f28-a934-9cabfec81c73/d5eeb22e1251f52a9d1d22643951e309ffa696ae2a8659903265749f84b57666.jpg)



Figure 8.2: Comparison of Monte Carlo and Markov chain Monte Carlo samples. In the top row, we show 3000 independent samples of $\mathrm { { N } } ( 0 , 1 ^ { 2 } )$ and a kernel density estimate of these samples along with the true density. In the bottom row, we show 3000 samples generated with the Random Walk Metropolis algorithm targeting $\mathrm { { N } } ( 0 , 1 ^ { 2 } )$ The proposal kernel is $\mathrm { N } ( \cdot , 0 . 5 ^ { 2 } )$ . The samples in the bottom row are clearly dependent.


for any $u _ { 1 } , . . . , u _ { n - 1 } \in \mathcal { X }$ . A Markov chain is called time-homogeneous, if 

$$
\mathbb {P} (U _ {2} \in \cdot | U _ {1} = u) = \mathbb {P} (U _ {k + 2} \in \cdot | U _ {k + 1} = u) \quad (u \in \mathcal {X}, k \in \mathbb {N}).\tag{8.2}
$$

and otherwise time-inhomogeneous. A time-homogeneous Markov chain can be fully represented by a Markov kernel $\overline { { K : B \mathcal { X } } } \times \mathcal { X }  [ 0 , 1 ]$ 

$$
K (B | u) = \mathbb {P} (U _ {n + 1} \in B | U _ {n} = u) \quad (B \in \mathcal {B X}, u \in \mathcal {X}, n \in \mathbb {N}).
$$

Let $\mu \in \operatorname { P r o b } ( \mathcal { X } , B \mathcal { X } )$ be a probability measure. We denote the composition of µ and K by 

$$
\mu K (B) := \int_ {\mathcal {X}} K (B | u) \mathrm{d} \mu (u) \qquad (B \in \mathcal {B X}).
$$

The measure µ is stationary w.r.t. K, if $\mu K = \mu$ . Finally, we say, the Markov kernel K satisfies detailed balance w.r.t. $\mu ^ { \prime } \in \mathrm { P r o b } ( \mathcal { X } , B \mathcal { X } )$ , if 

$$
\int_ {B} K (A | u) \mathrm{d} \mu^ {\prime} (u) = \int_ {A} K (B | u) \mathrm{d} \mu^ {\prime} (u) \qquad (A, B \in \mathcal {B X}).
$$

The detailed balance condition implies that the measure with respect to which it was shown is the stationary measure: 

Lemma 8.2.2. Let $K : B \mathcal { X } \times \mathcal { X }  [ 0 , 1 ]$ be a Markov kernel that satisfies detailed balance with respect to $\mu \in \operatorname { P r o b } ( \mathcal { X } , B \mathcal { X } )$ . Then, K is stationary w.r.t. µ. 

Proof. Exercise. 

We now define the Metropolis–Hastings Markov Kernel, discuss it, and show that it is stationary with respect to the target measure. 

Definition 8.2.3 (Hastings 1970 [22]). Let $\mu \in \operatorname { P r o b } ( \mathcal { X } , B \mathcal { X } )$ and ν be a σ-finite measure with $\mu \ll \nu$ . Moreover let $g : ( \mathcal { X } , \mathcal { B X } )  ( { \mathbb R } , \mathcal { B } { \mathbb R } )$ be a positive function with 

$$
g = c \cdot \frac {\mathrm{d} \mu}{\mathrm{d} \nu},
$$

for some $c \in ( 0 , \infty )$ . Moreover, let $Q : \mathcal { X } \times B \mathcal { X }  [ 0 , 1 ]$ be a Markov kernel, given by a positive function $q : ( \mathcal { X } \times \mathcal { X } , B \mathcal { X } \otimes B \mathcal { X } )  ( \mathbb { R } , B \mathbb { R } )$ , with 

$$
Q (A | u) := \int_ {A} q (u ^ {\prime} | u) \mathrm{d} \nu (u ^ {\prime}) \qquad (A \in \mathcal {B X}, u \in \mathcal {X}).
$$

The Metropolis–Hastings Markov kernel is given by 

$$
K _ {\mathrm{MH}} (A | u) := \delta (A - u) \int_ {\mathcal {X}} (1 - \alpha (u, u ^ {\prime \prime})) Q (\mathrm{d} u ^ {\prime \prime} | u) + \int_ {A} \alpha (u, u ^ {\prime}) Q (\mathrm{d} u ^ {\prime} | u) \qquad (u \in \mathcal {X}, A \in \mathcal {B X}),
$$

where 

$$
\alpha (u, u ^ {\prime}) = \min \left\{1, \frac {g (u ^ {\prime}) q (u | u ^ {\prime})}{g (u) q (u ^ {\prime} | u)} \right\}.
$$

Interpreting this Markov kernel is rather dificult. Algorithmically, we can represent the Metropolis–Hastings MCMC method 

1. Start with some initial value $U _ { 1 } \in \mathcal { X }$ (say a.s. constant); set m $ 1$ ; 

2. Sample $U ^ { * } \sim Q ( \cdot | U _ { m } )$ ; (‘proposal step’) 

3. With probability $\alpha ( U _ { m } , U ^ { * } )$ set $U _ { m + 1 } \gets U ^ { * }$ otherwise $U _ { m + 1 } \gets U _ { m + 1 } ; \qquad ( \mathrm { ` a c c e p t a n c e ~ s t e p ' } )$ 

4. Increment $m \gets m + 1$ and go to 2. 

When looking at $K _ { \mathrm { M H } }$ , we see the proposal step in the Markov kernel Q and the acceptance step in the $( 1 - \alpha )$ and the α. 

Another remarkable observation is that we need to know the density g only up to a normalising constant. This is especially useful, when sampling from a posterior measure: we usually have only access to prior density and likelihood. Model evidence/normalising constant are not necessary. 

Proposition 8.2.4. K satisfies detailed balance w.r.t. $\mu$ 

Proof. Let $A , B \in B { \mathcal { X } }$ 

$$
\begin{array}{l} \int_ {B} K _ {\mathrm{MH}} (A | u) \mathrm{d} \mu (u) \\ = \int_ {B} \delta (A - u) \int_ {\mathcal {X}} (1 - \alpha (u, u ^ {\prime \prime})) Q (\mathrm{d} u ^ {\prime \prime} | u) + \int_ {A} \alpha (u, u ^ {\prime}) Q (\mathrm{d} u ^ {\prime} | u) \mathrm{d} \mu (u). \end{array}
$$

We discuss the two parts of this sum one after another. We first have 

$$
\begin{array}{r l} & {\int_ {B} \delta (A - u) \int_ {\mathcal {X}} (1 - \alpha (u, u ^ {\prime \prime})) Q (\mathrm{d} u ^ {\prime \prime} | u) \mathrm{d} \mu (u)} \\ & {\quad = \int_ {\mathcal {X}} \mathbf {1} _ {A \cap B} (u) \int_ {\mathcal {X}} (1 - \alpha (u, u ^ {\prime \prime})) Q (\mathrm{d} u ^ {\prime \prime} | u) g (u) \mathrm{d} \nu (u)} \\ & {\quad = \int_ {A} \delta (B - u) \int_ {\mathcal {X}} (1 - \alpha (u, u ^ {\prime \prime})) Q (\mathrm{d} u ^ {\prime \prime} | u) \mathrm{d} \mu (u).} \end{array}
$$

Secondly, 

$$
\begin{array}{l} \int_ {B} \int_ {A} \alpha (u, u ^ {\prime}) Q (\mathrm{d} u ^ {\prime} | u) \mathrm{d} \mu (u) \\ \qquad = \int_ {B} \int_ {A} \min \left\{1, \frac {g (u ^ {\prime}) q (u | u ^ {\prime})}{g (u) q (u ^ {\prime} | u)} \right\} q (u ^ {\prime} | u) \mathrm{d} \nu (u ^ {\prime}) \frac {g (u)}{c} \mathrm{d} \nu (u) \\ \qquad = \int_ {B} \int_ {A} \min \left\{g (u) q (u ^ {\prime} | u), g (u ^ {\prime}) q (u | u ^ {\prime}) \right\} \mathrm{d} \nu (u ^ {\prime}) \frac {1}{c} \mathrm{d} \nu (u) \\ \qquad = \int_ {A} \int_ {B} \min \left\{1, \frac {g (u) q (u ^ {\prime} | u)}{g (u ^ {\prime}) q (u | u ^ {\prime})} \right\} \frac {g (u ^ {\prime})}{c} q (u | u ^ {\prime}) \mathrm{d} \nu (u) \mathrm{d} \nu (u ^ {\prime}) \\ \qquad = \int_ {A} \int_ {B} \alpha (u ^ {\prime}, u) Q (\mathrm{d} u | u ^ {\prime}) \mathrm{d} \mu (u ^ {\prime}). \end{array}
$$

Combining these two results gives us detailed balance. 

We finish by giving typical examples for proposal kernels $Q$ used in Metropolis–Hastings MCMC. 

Example 8.2.5 (Independence Sampler). Let $\rho \in \operatorname { P r o b } ( \mathcal { X } , B \mathcal { X } )$ . The Metropolis-Hastings algorithm with proposal kernel 

$$
Q (\cdot | u) = \rho (u \in \mathcal {X})
$$

is called independence sampler. The acceptance probability is given by 

$$
\alpha (u, u ^ {\prime}) = \min \left\{1, \frac {g (u ^ {\prime}) q (u)}{g (u) q (u ^ {\prime})} \right\},
$$

where $q = \mathrm { d } \rho / \mathrm { d } \nu$ 

In a Bayesian inverse problem with prior $\mu _ { 0 } \in \operatorname { P r o b } ( \mathcal { X } , B \mathcal { X } )$ and likelihood $L ( f _ { n } | \cdot )$ , we can choose $\rho : = \nu : = \mu _ { 0 }$ . In this case, the acceptance probability simplifies to 

$$
\alpha (u, u ^ {\prime}) = \min \left\{1, \frac {L (f _ {n} | u ^ {\prime})}{L (f _ {n} | u)} \right\}.
$$

Please note that the independence sampler proposes moves independently of the current position. This does not imply that the generated samples are independent. The acceptance step couples the samples. 

Example 8.2.6 (Random Walk; Metropolis et al. 1953 [27]). Let $\rho \in \operatorname { P r o b } ( \mathcal { X } , B \mathcal { X } )$ have a symmetric density $q ^ { \prime } = \mathrm { d } \rho / \mathrm { d } \nu$ , i.e. $q ^ { \prime } = q ^ { \prime } ( - \cdot )$ . The Metropolis-Hastings algorithm with proposal kernel 

$$
Q (\cdot | u) = \rho (\cdot - u) \qquad (u \in \mathcal {X})
$$

is called Random Walk Metropolis sampler. The acceptance probability is given by 

$$
\alpha (u, u ^ {\prime}) = \min \left\{1, \frac {g (u ^ {\prime})}{g (u)} \right\}.
$$

Note that the acceptance probability is independent of the proposal distribution; indeed, it cancels: $q ( u | u ^ { \prime } ) = q ^ { \prime } ( u - u ^ { \prime } ) = q ^ { \prime } ( u ^ { \prime } - u ) = q ( u ^ { \prime } | u )$ 

Example 8.2.7 (Preconditioned Crank–Nicolson MCMC; Cotter et al. 2013 [16]). Let be a separable Hilbert space and let $\mu _ { 0 } = \mathrm { N } ( 0 , \mathcal { C } ) \in \mathrm { P r o b } ( \mathcal { X } , B \mathcal { X } )$ for some suitable operator ${ \mathcal { C } } : { \mathcal { X } }  { \mathcal { X } }$ . We consider the (BIP) with prior $\mu _ { 0 }$ and likelihood $L ( f _ { n } | \cdot )$ . Let $\beta \in ( 0 , 1 )$ The Metropolis-Hastings algorithm with proposal kernel 

$$
Q (\cdot | u) := \mathrm{N} (\sqrt {1 - \beta^ {2}} u, \beta^ {2} \mathcal {C})
$$

is called preconditioned Crank–Nicolson algorithm (pCN-MCMC). The acceptance probability is given by 

$$
\alpha (u, u ^ {\prime}) = \min \left\{1, \frac {L (f _ {n} | u ^ {\prime})}{L (f _ {n} | u)} \right\}.
$$

This method is particularly useful in high- and infinite dimension, where the random walk algorithm cannot be applied. Proving that α is the correct acceptance probability is rather simple in finite dimensions, not quite as easy in infinite dimensions. 

The method is referred to as pCN MCMC as the proposal can be derived as a Crank– Nicolson discretisation of some S(P)DE. 

## Bibliography



[1] <sup>Y.</sup> <sup>A.</sup> <sup>Abramovich</sup> <sup>and</sup> <sup>C.</sup> <sup>D.</sup> <sup>Aliprantis</sup>, An Invitation to Operator Theory, Graduate Studies in Mathematics, American Mathematical Society, 2002. 





[2] <sup>R.</sup> <sup>A.</sup> <sup>Adams</sup> <sup>and</sup> <sup>J.</sup> <sup>J.</sup> <sup>F.</sup> <sup>Fournier</sup>, Sobolev Spaces, Elsevier Science, Singapore, 2003. 





<sub>[3]</sub> S. Agapiou, O. Papaspiliopoulos, D. Sanz-Alonso, and A. M. Stuart<sub>, Importance</sub> sampling: intrinsic dimension and computational cost, Statist. Sci., 32 (2017), pp. 405–431. 





[4] <sup>C.</sup> <sup>D.</sup> <sup>Aliprantis</sup> <sup>and</sup> <sup>K.</sup> <sup>Border</sup>, Infinite Dimensional Analysis: A Hitchhiker’s Guide, Springer, 2006. 



[5] <sup>L.</sup> <sup>Ambrosio,</sup> <sup>N.</sup> <sup>Fusco,</sup> <sup>and</sup> <sup>D.</sup> <sup>Pallara</sup>, Functions of Bounded Variation and Free Discontinuity Problems, Clarendon Press, 2000. 



[6] <sup>R.</sup> <sup>B.</sup> <sup>Ash</sup> <sup>and</sup> <sup>C.</sup> <sup>A.</sup> <sup>Doleans-Dade</sup> <sup>´</sup> , Probability & Measure Theory, Harcourt Academic Press, 2000. 





[7] <sup>A.</sup> <sup>B.</sup> <sup>Bakushinskii</sup>, Remarks on the choice of regularization parameter from quasioptimality and relation tests, Zhurnal Vychislitel’no¨ı Matematiki i Matematichesko¨ı Fiziki, 24 (1984), pp. 1258–1259. 





[8] <sup>H.</sup> <sup>H.</sup> <sup>Bauschke</sup> <sup>and</sup> <sup>P.</sup> <sup>L.</sup> <sup>Combettes</sup>, Convex Analysis and Monotone Operator Theory in Hilbert Spaces, 2011. 





[9] <sup>M.</sup> <sup>Benning</sup> <sup>and</sup> <sup>M.</sup> <sup>Burger</sup>, Modern regularization methods for inverse problems, Acta Numerica, 27 (2018), pp. 1–111. 





[10] <sup>P.</sup> <sup>Billingsley</sup>, Probability and Measure, John Wiley and Sons, second ed., 1986. 





[11] <sup>V.</sup> <sup>I.</sup> <sup>Bogachev</sup>, Gaussian measures, vol. 62 of Mathematical Surveys and Monographs, American Mathematical Society, Providence, RI, 1998. 





[12] <sup>B.</sup> <sup>Bollobas´</sup> , Linear Analysis: An Introductory Course, Cambridge University Press, Cambridge, second ed., 1999. 





[13] <sup>K.</sup> <sup>Bredies</sup> <sup>and</sup> <sup>D.</sup> <sup>A.</sup> <sup>Lorenz</sup>, Mathematical Image Processing, Springer, 2018. 





[14] <sup>M.</sup> <sup>Burger</sup> <sup>and</sup> <sup>S.</sup> <sup>Osher</sup>, Convergence rates of convex variational regularization, Inverse Problems, 20 (2004), p. 1411. 





[15] , A guide to the tv zoo, in Level-Set and PDE-based Reconstruction Methods, M. Burger and S. Osher, eds., Springer, 2013. 





<sub>[16]</sub> S. L. Cotter, G. O. Roberts, A. M. Stuart, and D. White<sub>, MCMC</sub> <sub>Methods</sub> <sub>for</sub> Functions: Modifying Old Algorithms to Make Them Faster, Statist. Sci., 28 (2013), pp. 424– 446. 





[17] <sup>R.</sup> <sup>T.</sup> <sup>Cox</sup>, Probability, frequency and reasonable expectation, American Journal of Physics, 14 (1946), pp. 1–13. 





[18] <sup>N.</sup> <sup>Dunford</sup> <sup>and</sup> <sup>J.</sup> <sup>T.</sup> <sup>Schwartz</sup>, Linear Operators, Part 1: General Theory, Wiley Interscience Publishers, 1988. 





[19] <sup>I.</sup> <sup>Ekeland</sup> <sup>and</sup> <sup>R.</sup> <sup>Temam´</sup> , Convex Analysis and Variational Problems, 1976. 





[20] <sup>H.</sup> <sup>W.</sup> <sup>Engl,</sup> <sup>M.</sup> <sup>Hanke,</sup> <sup>and</sup> <sup>A.</sup> <sup>Neubauer</sup>, Regularization of inverse problems, vol. 375, Springer Science & Business Media, 1996. 





[21] <sup>C.</sup> <sup>W.</sup> <sup>Groetsch</sup>, Stable approximate evaluation of unbounded operators, Springer, 2006. 





[22] <sup>W.</sup> <sup>K.</sup> <sup>Hastings</sup>, Monte Carlo Sampling Methods Using Markov Chains and Their Applications, Biometrika, 57 (1970), pp. 97–109. 





[23] <sup>J.</sup> <sup>Hunter</sup> <sup>and</sup> <sup>B.</sup> <sup>Nachtergaele</sup>, Applied Analysis, World Scientific Publishing Company Incorporated, 2001. 





[24] <sup>J.</sup> <sup>A.</sup> <sup>Iglesias,</sup> <sup>G.</sup> <sup>Mercier,</sup> <sup>and</sup> <sup>O.</sup> <sup>Scherzer</sup>, A note on convergence of solutions of total variation regularized linear inverse problems, Inverse Problems, 34 (2018), p. 055011. 





[25] <sup>A.</sup> <sup>Klenke</sup>, Probability Theory: A comprehensive Course, Springer, 2014. 





[26] <sup>J.</sup> <sup>Leao,</sup> <sup>D,</sup> <sup>M.</sup> <sup>Fragoso,</sup> <sup>and</sup> <sup>P.</sup> <sup>Ruffino</sup>, Regular conditional probability, disintegration of probability and Radon spaces, Proyecciones, 23 (2004), pp. 15–29. 





<sub>[27]</sub> N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, and E. Teller<sub>,</sub> Equation of State Calculations by Fast Computing Machines, J. Chem. Phys., 21 (1953), pp. 1087–1092. 





[28] <sup>A.</sup> <sup>W.</sup> <sup>Naylor</sup> <sup>and</sup> <sup>G.</sup> <sup>R.</sup> <sup>Sell</sup>, Linear Operator Theory in Engineering and Science, Springer Science & Business Media, 2000. 





[29] <sup>Y.</sup> <sup>V.</sup> <sup>Prokhorov</sup>, Convergence of random processes and limit theorems in probability theory, Theory of Probability & Its Applications, 1 (1956), pp. 157–214. 





[30] <sup>C.</sup> <sup>P.</sup> <sup>Robert</sup> <sup>and</sup> <sup>G.</sup> <sup>Casella</sup>, Monte Carlo Statistical Methods, Springer, 2004. 





[31] <sup>L.</sup> <sup>I.</sup> <sup>Rudin,</sup> <sup>S.</sup> <sup>Osher,</sup> <sup>and</sup> <sup>E.</sup> <sup>Fatemi</sup>, Nonlinear total variation based noise removal algorithms, Physica D: Nonlinear Phenomena, 60 (1992), pp. 259–268. 





[32] <sup>W.</sup> <sup>Rudin</sup>, Functional Analysis, International series in pure and applied mathematics, McGraw-Hill, 1991. 





[33] <sup>K.</sup> <sup>Saxe</sup>, Beginning Functional Analysis, Springer, 2002. 





<sub>[34]</sub> O. Scherzer, M. Grasmair, H. Grossauer, M. Haltmeier, and F. Lenzen<sub>, Variational</sub> Methods in Imaging, Springer, 2009. 





[35] <sup>A.</sup> <sup>M.</sup> <sup>Stuart</sup>, Inverse problems: a Bayesian perspective, Acta Numerica, 19 (2010), pp. 451– 559. 





[36] <sup>T.</sup> <sup>Tao</sup>, Epsilon of Room, One, vol. 1, American Mathematical Soc., 2010. 





[37] <sup>E.</sup> <sup>Zeidler</sup>, Applied Functional Analysis: Applications to Mathematical Physics, vol. 108 of Applied Mathematical Sciences Series, Springer, 1995. 





[38] , Applied Functional Analysis: Main Principles and Their Applications, vol. 109 of Applied Mathematical Sciences Series, Springer, 1995. 

