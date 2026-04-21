## Page 1

# **Inverse Problems**

Lecture notes, Michaelmas term 2020 University of Cambridge

**Yury Korolev and Jonas Latz**

May 18, 2021

This work is licensed under a Creative Commons “Attribution-ShareAlike 3.0 Unported” license.


---

## Page 2

2


---

## Page 3

# **Contents**

**1 Introduction to Inverse Problems 7 **1.1 Well-posed and ill-posed problems . . . . . . . . . . . . . . . . . . . . . . . 7 1.2 Examples of inverse problems . . . . . . . . . . . . . . . . . . . . . . . . . . 9 1.2.1 Signal deblurring . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9 1.2.2 Heat equation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9 1.2.3 Diﬀerentiation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10 1.2.4 Matrix inversion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11 1.2.5 Tomography . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12 1.2.6 Groundwater ﬂow/hydraulic tomography . . . . . . . . . . . . . . . 14

**2 Generalised Solutions 15 **2.1 Generalised Inverses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17 2.2 Compact Operators . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

**3 Classical Regularisation Theory 27 **3.1 What is Regularisation? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27 3.2 Parameter Choice Rules . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29 3.2.1 A priori parameter choice rules . . . . . . . . . . . . . . . . . . . . . 30 3.2.2 A posteriori parameter choice rules . . . . . . . . . . . . . . . . . . . 31 3.2.3 Heuristic parameter choice rules . . . . . . . . . . . . . . . . . . . . 31 3.3 Spectral Regularisation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32 3.3.1 Truncated singular value decomposition . . . . . . . . . . . . . . . . 33 3.3.2 Tikhonov regularisation . . . . . . . . . . . . . . . . . . . . . . . . . 34

**4 Variational Regularisation 35 **4.1 Background . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35 4.1.1 Banach spaces and weak convergence . . . . . . . . . . . . . . . . . . 35 4.1.2 Convex analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38 4.1.3 Minimisers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43 4.1.4 Duality in convex optimisation . . . . . . . . . . . . . . . . . . . . . 45 4.2 Well-posedness and Regularisation Properties . . . . . . . . . . . . . . . . . 46 4.3 Total Variation Regularisation . . . . . . . . . . . . . . . . . . . . . . . . . 52

**5 Convex Duality 57 **5.1 Dual Problem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58 5.2 Source Condition and Convergence Rates . . . . . . . . . . . . . . . . . . . 60

3


---

## Page 4

4 *CONTENTS*

**6 Bayesian probability and statistics 65 **6.1 From inverse problems to Bayesian inverse problems . . . . . . . . . . . . . 65 6.2 Reminder: measure, probability, and integration . . . . . . . . . . . . . . . 66 6.3 Conditional probability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 71 6.4 Bayesian statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 75 6.4.1 Statistical models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 75 6.4.2 Bayes’ formula . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 76

**7 Bayesian inverse problems and well-posedness 79 **7.1 Bayesian inverse problems . . . . . . . . . . . . . . . . . . . . . . . . . . . . 79 7.2 Metrics on spaces of probability measures . . . . . . . . . . . . . . . . . . . 80 7.3 Stability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 81

**8 Function space priors and Monte Carlo 85 **8.1 Gaussian measures . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 85 8.2 Monte Carlo techniques . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 88 8.2.1 Standard Monte Carlo . . . . . . . . . . . . . . . . . . . . . . . . . . 88


---

## Page 5

*CONTENTS *5

These lecture notes are based on the Inverse Problems course taught by Yury Korolev and Jonas Latz in Michaelmas term 2020 at the University of Cambridge.^(1)^( )Complementary material can be found in the following books, lecture notes and review papers:

1. Heinz Werner Engl, Martin Hanke, and Andreas Neubauer.* Regularization of Inverse Problems.* Springer, 1996.

2. Otmar Scherzer, Markus Grasmair, Harald Grossauer, Markus Haltmeier and Frank Lenzen.* Variational Methods in Imaging*. Springer, 2008.

3. Kristian Bredies and Dirk Lorenz.* Mathematical Image Processing.* Springer, 2018

4. Martin Benning and Martin Burger.* Modern regularization methods for inverse prob- lems*. Acta Numerica, 27, 1-111 (2018)

https://www.cambridge.org/core/journals/acta-numerica/article/ modern-regularization-methods-for-inverse-problems/ 1C84F0E91BF20EC36D8E846EF8CCB830

5. K.Saxe.* Beginning Functional Analysis*. Springer, 2002

6. Masoumeh Dashti and Andrew M. Stuart,* The Bayesian approach to inverse problems*, Handbook of Uncertainty Quantiﬁcation, 2016.

7. Jari Kaipio and Erkki Somersalo,* Statistical and computational inverse problems*, vol. 160 of Applied Mathematical Sciences, 2005.

8. O. Kallenberg,* Foundations of modern probability theory*, Springer, 1997.

9. Andrew M. Stuart,* Inverse problems: a Bayesian perspective*, Acta Numerica, 2010.

These lecture notes are under constant redevelopment and might contain typos or errors. We very much appreciate if you report any mistakes found (to y.korolev@damtp.cam.ac.uk or jl2160@cam.ac.uk). Thanks!

1https://www.damtp.cam.ac.uk/research/cia/inverse-problems-michaelmas-2020


---

## Page 6

6 *CONTENTS*


---

## Page 7

# **Chapter 1**
# **Introduction to Inverse Problems**

Inverse problems arise from the need to gain information about an unknown object of inter- est from given indirect measurements. Inverse problems have several applications varying from medical imaging and industrial process monitoring to ozone layer tomography and modelling of ﬁnancial markets. The common feature for inverse problems is the need to understand indirect measurements and to overcome extreme sensitivity to noise and mod- elling inaccuracies. In this course we employ both deterministic and probabilistic approach to inverse problems to ﬁnd stable and meaningful solutions that allow us quantify how inaccuracies in the data or model aﬀect the obtained estimate.

### **1.1 Well-posed and ill-posed problems**

We start by considering the problem of ﬁnding* u** ∈*R^(*d*)^( )that satisﬁes the equation

*f* =* Au, *(1.1)

where* f** ∈*R^(*k*)^( )is given. We refer to* f* as observed data or measurement and* u* as an unknown. The physical phenomena that relates the unknown and the measurement is modelled by a matrix* A** ∈*R^(*k*)^(*×*)^(*d*). In real life the perfect data given in (1.1) is perturbed by noise and we observe measurements

*f**n* =* Au* +* n, *(1.2)

where* n** ∈*R^(*k*)^( )represents the observational noise. We are interested in ill-posed inverse problems, where the inverse problem is more diﬃcult to solve than the direct problem of ﬁnding* f**n* when* u* is given. To explain this we ﬁrst need to introduce well-posedness as deﬁned by Jacques Hadamard:

**Deﬁnition 1.1.1.*** A problem is called well-posed if*

*1. There exists at least one solution. (Existence)*

*2. There is at most one solution. (Uniqueness)*

*3. The solution depends continuously on data. (Stability)*

The direct or forward problem is assumed to be well-posed. The inverse problems are ill-posed and break at least one of the above conditions.

7


---

## Page 8

8 *CHAPTER 1. INTRODUCTION TO INVERSE PROBLEMS*

1. Assume that* d < k* and* A* : R^(*d*)^(* *)*→R*(*A*) ⊊R^(*k*), where the range of* A* is a proper subset of R^(*k*). Furthermore, we assume that* A* has a unique inverse* A*^(*−*)^(1)^( ):* R*(*A*)* →*R^(*k*). Because of the noise in the measurement* f**n** ̸∈R*(*A*) so that simply inverting* A* with the data given in (1.2) is not possible. Note that usually only the statistical properties of the noise* n* are known so we cannot just subtract it.

2. Assume next that* d > k* and* A* : R^(*d*)^(* *)*→*R^(*k*), in which case the system is underde- termined. We then have more unknowns than equations which means that there are several possible solutions.

3. Consider next case* d* =* k* and there exist* A*^(*−*)^(1)^( ): R^(*k*)^(* *)*→*R^(*k*)^( )but the condition number *κ* =* λ*1*/λ**k*, where* λ*1 and* λ**k* are the biggest and smallest eigenvalues of* A*, is very large. Such a matrix is said to be ill-conditioned and is almost singular. In this case the problem is sensitive even to smallest errors in the measurement. Hence the naive reconstruction e*u* =* A*^(*−*)^(1)*f**n* =* u* +* A*^(*−*)^(1)*n* does not produce a meaningful solution but will be dominated by* A*^(*−*)^(1)*n*. Note that* ∥**A*^(*−*)^(1)*n**∥*2* ≈∥**n**∥*2*/λ**k* can be arbitrarily large.

The last part illustrates one of the key perspectives of inverse problem theory; How can we stabilise the reconstruction process while maintaining acceptable accuracy? A deterministic way of achieving a unique and stable solution for the problem (1.2) is to use regularisation theory. In the classical Tikhonov regularisation a solution is attained by solving

min *u**∈*R^(*d*)

 *∥**Au** −**f**n**∥*^(2)^( )+* α**∥**Lu**∥*^(2)^( )*. *(1.3)

Above* α* acts as a tuning parameter balancing the eﬀect of the data ﬁdelity term* ∥**Au** −**f**n**∥*^(2)

and the stabilising regularisation term* ∥**u**∥*^(2). The ﬁrst half of the course will concentrate on regularisation theory. Another way of tackling problems arising from ill-posedness is Bayesian inversion. The idea of statistical inversion methods is to rephrase the inverse problem as a question of statistical inference. We then consider problem

*F* =* AU* +* N, *(1.4)

where the measurement, unknown and noise are now modelled as random variables. This approach allows us to model the noise through its statistical properties. We can also encode our* a priori* knowledge of the unknown in form of a probability distribution that assigns higher probability to those values of* u* we expect to see. The solution to (1.4) is so-called *posterior distribution*, which is the conditional probability distribution of* u* given a mea- surement* m*. This distribution can then be used to obtain estimates that are most likely in some sense. We will return to the Bayesian approach to inverse problems in the second half of the course In this course we will concentrate on continuous inverse problems where in (1.1) and (1.2)* A* :* X** →**Y* is a linear or non-linear forward operator acting between some spaces *X* and* Y* , typically Hilbert or Banach spaces, the measured data* f**n** ∈**Y* is a function and* u** ∈**X* is the quantity we want to reconstruct from the data. Linear inverse problems include such important applications as computer tomography, magnetic resonance imaging and image deblurring in microscopy or astronomy. In other important applications, such as seismic imaging, the forward operator is non-linear (e.g., parameter identiﬁcation problems for PDEs). Next we will take a look at some examples of linear and non-linear inverse problems to see what kind of challenges we face when trying to solve them.


---

## Page 9

*1.2. EXAMPLES OF INVERSE PROBLEMS *9

### **1.2 Examples of inverse problems**

**1.2.1 Signal deblurring**

The deblurring (or deconvolution) problem of recovering an input signal* u* form an observed signal

*f**n*(*t*) = Z* ∞*

*−∞ **a*(*t** −**s*)*u*(*s*)*ds* +* n*(*t*)

occurs in many imaging, and image- and signal processing applications. Here the function *a* is known as the blurring kernel. The noiseless data is given by* f*(*t*) = R* ∞ −∞*^(*a*)^(()^(*t*)^(* −*)^(*s*)^())^(*u*)^(()^(*s*)^())^(*ds*)^( and its Fourier transform is )b*f*(*ξ*) = R* ∞ −∞*^(*e*)^(*−*)^(*iξt*)^(*f*)^(()^(*t*)^())^(*dt*)^(. The convolution theorem implies)

b*f*(*ξ*) = b*a*(*ξ*)b*u*(*ξ*)*,*

and hence by inverse Fourier transform

*u*(*t*) = ^(1)

2*π*

Z* ∞*

*−∞ **e*^(*itξ*)^( b)^(*f*)^(()^(*ξ*)^())

b*a*(*ξ*)^(*dξ.*)

However, we can only observe noisy measurements and hence we have on the frequency domain ^(c )*f**n*(*ξ*) = b*a*(*ξ*)b*u*(*ξ*) + b*n*(*ξ*). The estimate* u**est* based on the convolution theorem is given by

*u**est*(*t*) =* u*(*t*) + ^(1)

2*π*

Z* ∞*

*−∞ **e*^(*itξ*)^( b)^(*n*)^(()^(*ξ*)^())

b*a*(*ξ*)^(*dξ,*)

which is often not even well deﬁned, since usually the kernel* a* decreases exponentially (or has compact support), making the denominator small, whereas the Fourier transform of the noise will be non-zero.

**1.2.2 Heat equation**

Next we study the problem of recovering the initial condition* u* of the heat equation from a noisy observation* f**n* of the solution at some time* T >* 0. We consider the heat equation on a torus T^(*d*), with Dirichlet boundary conditions      

    

*dv*

*dt** *^(*−*)^(∆)^(*v*)^( = 0 )on T^(*d*)^(* *)*×* R+ *v*(*x, t*) = 0 on* ∂*T^(*d*)^(* *)*×* R+ *v*(*x, T*) =* f*(*x*) on T^(*d*)

*v*(*x,* 0) =* u*(*x*) on T^(*d*)

where ∆denotes the Laplace operator and* D*(∆) =* H*^(1 )0^(()^(T)^(*d*)^())^(* ∩*)^(*H*)^(2)^(()^(T)^(*d*)^(). )Note that the operator* −*∆is positive and self-adjoint on Hilbert space* H* =* L*^(2)(T^(*d*)). Given a function* u** ∈**L*^(2)(T^(*d*)) we can decompose it as a Fourier series

*u*(*x*) = X

*n**∈*Z^(*d *)*u**n**e*^(2)^(*πi*)^(*⟨*)^(*n,x*)^(*⟩*)*,*


---

## Page 10

10 *CHAPTER 1. INTRODUCTION TO INVERSE PROBLEMS*

where* u**n* =* ⟨**u, e*^(2)^(*πi*)^(*⟨*)^(*n,x*)^(*⟩*)*⟩*are the Fourier coeﬃcients, and the identity holds for almost every *x** ∈*T^(*d*). The* L*^(2)^( )norm of* u* is given by the Parseval’s identity* ∥**u**∥*^(2 )*L*^(2)^( =)^( P)^(* |*)^(*u*)^(*n*)^(*|*)^(2)^(. Remember )that the Sobolev space* H*^(*s*)(T^(*d*)),* s** ∈*N, consist of all* L*^(2)(T^(*d*)) integrable functions whose *α*^(*th*)^( )order weak derivatives exist and are* L*^(2)(T^(*d*)) integrable for all* |**α**|* ⩽*s*. The fractional Sobolev space* H*^(*s*)(T^(*d*)) is given by the subspace of functions* u** ∈**L*^(2)(T^(*d*)), such that

*∥**u**∥*^(2 )*H*^(*s*)^( = )X

*n**∈*Z^(*d *)(1 + 4*π*^(2)*|**n**|*^(2))^(*s*)*|**u**n**|*^(2)^(* *)*<** ∞**. *(1.5)

Note that for a positive integer* s*, the above deﬁnition agrees with the deﬁnition given using the weak derivatives. For* s <* 0, we deﬁne* H*^(*s*)(T^(*d*)) via duality or as the closure of* L*^(2)(T^(*d*)) under the norm (1.5). The resulting spaces are separable for all* s** ∈*R. The eigenvectors of* −*∆in T^(*d*)^( )form the orthonormal basis of* L*^(2)(T^(*d*)) and the eigenval- ues are given by 4*π*^(2)*|**n**|*^(2),* n** ∈*Z^(*d*). We can also work on real-valued functions where the eigenfunctions* {**ϕ**j**}*^(*∞ *)*j*=1 ^(comprise sine and cosine functions. The eigenvalues of)^(* −*)^(∆, when)

ordered on a one-dimensional lattice, then satisfy* λ**j** ≍**j *2 *d* . The notation* ≍*means that there exist constants* C*1*, C*2* >* 0, such that* C*1*j *2 *d* ⩽*λ**j* ⩽*C*2*j *2 *d* . The solution to the forward heat equation can be written as

*v*(*t*) =

*∞ *X

*j*=1 *u**j**e*^(*−*)^(*λ*)^(*j*)^(*t*)*ϕ**j**.*

We notice that

*∥**v*(*t*)*∥*^(2 )*H*^(*s*)^(* ≍*)

*∞ *X

*j*=1 *j *2*s*

*d** e*^(*−*)^(2)^(*λ*)^(*j*)^(*t*)*|**u**j**|*^(2)^( )=* t*^(*−*)^(*s *)*∞ *X

*j*=1 (*λ**j**t*)^(*s*)*e*^(*−*)^(2)^(*λ*)^(*j*)^(*t*)*|**u**j**|*^(2)^( )⩽*Ct*^(*−*)^(*s *)*∞ *X

*j*=1 *|**u**j**|*^(2)^( )=* Ct*^(*−*)^(*s*)*∥**u**∥**L*2

which implies that* v*(*t*)* ∈**H*^(*s*)(T^(*d*)) for all* s >* 0. We now have observation model

*f**n* =* Au* +* n,*

where* A* =* e*^(*T*)^(∆)and* n* is the observational noise. The noise is not usually smooth (the often assumed white noise is not even an* L*^(2)^( )function) and hence measurement* f**n* is not in the image space* D*(*e*^(*T*)^(∆))* ⊂∩**s>*0*H*^(*s*)(T^(*d*)).

**1.2.3 Diﬀerentiation**

Consider the problems of evaluation the derivative of a function* f** ∈**L*^(2)[0*, π/*2]. Let

*Df* =* f*^(*′*)*,*

where* D*:* L*^(2)[0*, π/*2]* →**L*^(2)[0*, π/*2].

**Proposition 1.2.1.*** The operator** D** is unbounded from** L*^(2)[0*, π/*2]* →**L*^(2)[0*, π/*2]*.*

*Proof.* Take a sequence* f**n*(*x*) = sin(*nx*),* n* = 1*, . . . ,** ∞*. Clearly,* f**n** ∈**L*^(2)[0*, π/*2] for all* n* and *∥**f**n**∥*= ^(p)^(* π*)

4 ^(. However,)^(* Df*)^(*n*)^(()^(*x*)^() =)^(* n*)^( cos()^(*nx*)^() and)^(* ∥*)^(*Df*)^(*n*)^(*∥*)^(=)^(* n*)^(* →∞*)^(as)^(* n*)^(* →∞*)^(. Therefore,)^(* D *)is unbounded.


---

## Page 11

*1.2. EXAMPLES OF INVERSE PROBLEMS *11

This shows that diﬀerentiation is ill-posed from* L*^(2)^( )to* L*^(2). It does not mean that it can not be well-posed in other spaces. For instance, it is well-posed from* H*^(1)^( )(the Sobolev space of* L*^(2)^( )functions whose derivatives are also* L*^(2)) to* L*^(2). Indeed,* ∀**u** ∈**H*^(1)^( )we get

*∥**Df**∥**L*2 =* ∥**f*^(*′*)*∥**L*2 ⩽*∥**f**∥**H*1 =* ∥**f**∥**L*2 +* ∥**f*^(*′*)*∥**L*2*.*

However, since in practice we typically deal with functions corrupted by nonsmooth noise, the* L*^(2)^( )setting is practice-relevant, while the* H*^(1)^( )setting is not. Diﬀerentiation can be written as an inverse problem for an integral equation. For in- stance, the derivative* u* of some function* f** ∈**L*^(2)[0*,* 1] with* f*(0) = 0 satisﬁes

*f*(*x*) = Z* x*

0 *u*(*t*)* dt,*

which can be written as an operator equation* Au* =* f* with (*A**·*)(*x*) := R* x *0* *^(*·*)^(()^(*t*)^())^(* dt*)^(.)

**1.2.4 Matrix inversion**

In ﬁnite dimensions, the inverse problem (1.1) is a linear system. Linear systems are formally well-posed in the sense that the error in the solution is bounded by some constant times the error in the right-hand side, however, this constant depends on the condition number of the matrix* A* and can get arbitrary large for matrices with large condition numbers. In this case, we speak of* ill-conditioned* problems. Consider the problem (1.1) with* u** ∈*R^(*n*)^( )and* f** ∈*R^(*n*)^( )being* n*-dimensional vectors with real entries and* A** ∈*R^(*n*)^(*×*)^(*n*)^( )being a matrix with real entries. Assume further* A* to be symmetric and positive deﬁnite. We know from the spectral theory of symmetric matrices that there exist eigenvalues *λ*1 ⩾*λ*2 ⩾*. . .* ⩾*λ**n** >* 0 and corresponding (orthonormal) eigenvectors* a**j** ∈*R^(*n*)^( )for *j** ∈{*1*, . . . , n**}* such that* A* can be written as

*A* =

*n *X

*j*=1 *λ**j**a**j**a*^(*⊤ *)*j** *^(*. *)(1.6)

It is well known from numerical linear algebra that the condition number* κ* =* λ*1*/λ**n* is a measure of how stable (1.1) can be solved, which we will illustrate what follows. We assume that we measure* f**δ* instead of* f*, with* ∥**f** −**f**δ**∥*2 ⩽*δ**∥**A**∥*=* δλ*1, where* ∥· ∥*2 denotes the Euclidean norm of R^(*n*)^( )and* ∥**A**∥*the operator norm of* A* (which equals the largest eigenvalue of* A*). Then, if we further denote with* u**δ* the solution of* Au**δ* =* f**δ*, the diﬀerence between* u**δ* and the solution* u* to (1.1) is

*u** −**u**δ* =

*n *X

*j*=1 *λ*^(*−*)^(1 )*j** *^(*a*)^(*j*)^(*a*)^(*⊤ *)*j* ^(()^(*f*)^(* −*)^(*f*)^(*δ*)^())^(*.*)

Therefore, we can estimate

*∥**u** −**u**δ**∥*^(2 )2 ^(=)

*n *X

*j*=1 *λ*^(*−*)^(2 )*j **∥**a**j**∥*^(2 )2 | {z } =1

*|**a*^(*⊤ *)*j* ^(()^(*f*)^(* −*)^(*f*)^(*δ*)^())^(*|*)^(2)^( ⩽)^(*λ*)^(*−*)^(2 )*n** *^(*∥*)^(*f*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2 )2^(*,*)

due to the orthonormality of eigenvectors, the Cauchy-Schwarz inequality, and* λ**n* ⩽*λ**j*. Thus, taking square roots on both sides yields the estimate

*∥**u** −**u**δ**∥*2 ⩽*λ*^(*−*)^(1 )*n** *^(*∥*)^(*f*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2) ^(⩽)^(*κδ.*)


---

## Page 12

12 *CHAPTER 1. INTRODUCTION TO INVERSE PROBLEMS*

Hence, we observe that in the worst case an error* δ* in the data* y* is ampliﬁed by the condition number* κ* of the matrix* A*. A matrix with large* κ* is therefore called* ill-conditioned*. We want to demonstrate the eﬀect of this error ampliﬁcation with a small example.

**Example 1.2.1.** Let us consider the matrix

*A* = 1 1 1 1001 1000

 *,*

which has eigenvalues* λ**j* = 1 + 1 2000* *^(*± *)q

1 + 1 2000^(2)^( , condition number)^(* κ*)^(* ≈*)^(4002)^(* ≫*)^(1, and)

operator norm* ∥**A**∥≈*2. For given data* f* = (1*,* 1)^(*⊤*)the solution to* Au* =* f* is* u* = (1*,* 0)^(*⊤*). Now let us instead consider perturbed data* f**δ* = (99*/*100*,* 101*/*100)^(*⊤*). The solution* u**δ *to* Au**δ* =* f**δ* is then* u**δ* = (*−*19*.*01*,* 20)^(*⊤*). Let us reﬂect on the ampliﬁcation of the measurement error. By our initial assumption we ﬁnd that* δ* =* ∥**f** −**f**δ**∥**/**∥**A**∥≈∥*(0*.*01*,** −*0*.*01)^(*⊤*)*∥**/*2 = *√*

2*/*200. Moreover, the norm of the error in the reconstruction is then* ∥**u** −**u**δ**∥*=* ∥*(20*.*01*,* 20)^(*⊤*)*∥≈*20 *√*

2. As a result, the ampliﬁcation due to the perturbation is* ∥**u** −**u**δ**∥**/δ** ≈*4000* ≈**κ*.

**1.2.5 Tomography**

In almost any tomography application the underlying inverse problem is either the inversion of the Radon transform^(1)^( )or of the X-ray transform. For* u** ∈**C*^(*∞ *)0 ^(()^(R)^(*n*)^(),)^(* s*)^(* ∈*)^(R)^(, and)^(* θ*)^(* ∈*)^(*S*)^(*n*)^(*−*)^(1)^( the)^(* Radon transform*)^(* R*)^( :)^(* C*)^(*∞ *)0 ^(()^(R)^(*n*)^())^(* →*)^(*C*)^(*∞*)^(()^(*S*)^(*n*)^(*−*)^(1)^(* × *)R) can be deﬁned as the integral operator

*f*(*θ, s*) = (*R**u*)(*θ, s*) = Z

*x**·**θ*=*s **u*(*x*)* dx *(1.7)

= Z

*θ*^(*⊥*)^(*u*)^(()^(*sθ*)^( +)^(* y*)^())^(* dy,*)

which, for* n* = 2, coincides with the X-ray transform,

*f*(*θ, s*) = (*P**u*)(*θ, s*) = Z

R *u*(*sθ* +* tθ*^(*⊥*))* dt,*

for* θ** ∈**S*^(*n*)^(*−*)^(1)^( )and* θ*^(*⊥*)being the vector orthogonal to* θ*. Hence, the X-ray transform (and therefore also the Radon transform in two dimensions) integrates the function* u* over lines in R^(*n*), see Fig. 1.1^(2).

**Example 1.2.2.** Let* n* = 2. Then* S*^(*n*)^(*−*)^(1)^( )is simply the unit sphere* S*^(1)^( )=* {**θ** ∈*R^(2)^(* *)*| ∥**θ**∥*= 1*}*. We can choose for instance* θ* = (cos(*ϕ*)*,* sin(*ϕ*))^(*⊤*), for* ϕ** ∈*[0*,* 2*π*), and parametrise the Radon transform in terms of* ϕ* and* s*, i.e.

*f*(*ϕ, s*) = (*R**u*)(*ϕ, s*) = Z

R *u*(*s* cos(*ϕ*)* −**t* sin(*ϕ*)*, s* sin(*ϕ*) +* t* cos(*ϕ*))* dt. *(1.8)

1Named after the Austrian mathematician Johann Karl August Radon (16 December 1887 – 25 May 1956). 2Figure adapted from Wikipedia https://commons.wikimedia.org/w/index.php?curid=3001440, by Begemotv2718, CC BY-SA 3.0.


---

## Page 13

*1.2. EXAMPLES OF INVERSE PROBLEMS *13

*θ*

*s*

*u*(*x*)

*t*

*tθ*^(*⊥*)

Figure 1.1: Visualization of the Radon transform in two dimensions (which coincides with the X-ray transform). The function* u* is integrated over the ray parametrized by* θ* and* s*.^(3)

Note that—with respect to the origin of the reference coordinate system—*ϕ* determines the angle of the line along one wants to integrate, while* s* is the oﬀset from that line from the centre of the coordinate system. It can be shown that the Radon transform is linear and continuous, i.e.* R** ∈L*(*L*^(2)(*B*)*, L*^(2)(*Z*)), and even compact.

In** X-ray Computed Tomography (CT)**, the unknown quantity* u* represents a spa- tially varying density that is exposed to X-radiation from diﬀerent angles, and that absorbs the radiation according to its material or biological properties. The basic modelling assumption for the intensity decay of an X-ray beam is that within a small distance ∆*t* it is proportional to the intensity itself, the density, and the distance, i.e. *I*(*x* + (*t* + ∆*t*)*θ*)* −**I*(*x* +* tθ*)

∆*t *=* −**I*(*x* +* tθ*)*u*(*x* +* tθ*)*,*

for* x** ∈**θ*^(*⊥*). By taking the limit ∆*t** →*0 we end up with the ordinary diﬀerential equation

*d dt*^(*I*)^(()^(*x*)^( +)^(* tθ*)^() =)^(* −*)^(*I*)^(()^(*x*)^( +)^(* tθ*)^())^(*u*)^(()^(*x*)^( +)^(* tθ*)^())^(*, *)(1.9)

Let* R >* 0 be the radius of the domain of interest centred at the origin. Then, we integrate (1.9) from* t* =* − *p

*R*^(2)^(* *)*−∥**x**∥*^(2 )2^(, the position of the emitter, to)^(* t*)^( = )p

*R*^(2)^(* *)*−∥**x**∥*^(2 )2^(, the position )of the detector, and obtain

Z* *^(*√*)

*R*^(2)*−∥**x**∥*^(2 )2

*−*^(*√*)

*R*^(2)*−∥**x**∥*^(2 )2

*d dt*^(*I*)^(()^(*x*)^( +)^(* tθ*)^())

*I*(*x* +* tθ*)* *^(*dt*)^( =)^(* − *)Z* *^(*√*)

*R*^(2)*−∥**x**∥*^(2 )2

*−*^(*√*)

*R*^(2)*−∥**x**∥*^(2 )2 *u*(*x* +* tθ*)* dt* .

Note that, due to* d/dx* log(*f*(*x*)) =* f*^(*′*)(*x*)*/f*(*x*), the left hand side in the above equation simpliﬁes to

Z* *^(*√*)

*R*^(2)*−∥**x**∥*^(2 )2

*−*^(*√*)

*R*^(2)*−∥**x**∥*^(2 )2

*d dt*^(*I*)^(()^(*x*)^( +)^(* tθ*)^())

*I*(*x* +* tθ*)* *^(*dt*)^( = log ) *I * *x* + q

*R*^(2)^(* *)*−∥**x**∥*^(2 )2^(*θ *) *−*log  *I * *x** − *q

*R*^(2)^(* *)*−∥**x**∥*^(2 )2^(*θ *) .

As we know the radiation intensity at both the emitter and the detector, we therefore know* f*(*x, θ*) = log(*I*(*x** −**θ *p

*R*^(2)^(* *)*−∥**x**∥*^(2 )2^()))^(* −*)^(log()^(*I*)^(()^(*x*)^( +)^(* θ *)p

*R*^(2)^(* *)*−∥**x**∥*^(2 )2^()) and we can write the)


---

## Page 14

14 *CHAPTER 1. INTRODUCTION TO INVERSE PROBLEMS*

estimation of the unknown density* u* as the inverse problem of the X-ray transform (1.8) (if we further assume that* u* can be continuously extended to zero outside of the circle of radius* R*).

**1.2.6 Groundwater ﬂow/hydraulic tomography**

One goal in hydraulic tomography is to estimate the permeability of a groundwater reservoir. The permeability describes the conductivity of the groundwater reservoir and is, e.g., used to estimate the travel time of toxic or radioactive particles in the groundwater. To estimate the permeability, the water pressure in several position within the reser- voir is measured. Pressure head and permeability are linked through Darcy’s law and the (assumed) incompressibility of water. Let* D** ⊆*R^(*d*)^( )(*d* = 1*,* 2*,* 3) be an open, bounded, connected set with smooth boundary representing the groundwater reservoir. Let* a* :* D** →*(0*,** ∞*) be a continuously diﬀerentiable function representing the permeability and let* s* :* D** →*R be a continuous function repre- senting the water sources in the reservoir. Furthermore, assume that the water pressure is 0 outside of* D*. Darcy’s law states that the pressure* p* :* D** →*R, the ﬂux* ⃗q* :* D** →*R^(*d*), and the perme- ability in the reservoir are related as follows:

*⃗q*(*x*) =* −**a*(*x*)*∇**p*(*x*) (*x** ∈**D*)*.*

Incompressibility on the other hand requires that the divergence of the ﬂux is fully controlled by in- and outﬂow given through the source term* s*:

*∇·** ⃗q*(*x*) =* s*(*x*) (*x** ∈**D*)*.*

Finally, we can combine these assertions and obtain the elliptic partial diﬀerential equation

*−∇·** a*(*x*)*∇**p*(*x*) =* s*(*x*) (*x** ∈**D*)

*p*(*x*) = 0 (*x** ∈**∂D*)*.*

In the described set-up, we now observe the pressure* p* in several positions* x*1*, . . . , x**I** ∈**D*, e.g., we observe* f**n* = (*p*(*x**i*) :* i* = 1*, ..., I*) +* n*. We consider the inverse problem consisting in the estimation of the permeability* a* using the pressure measurements* f**n*. Indeed, using noisy point evaluations of the solution of the partial diﬀerential equation, we try to estimate its diﬀusion coeﬃcient. Note that the map* a** 7→*(*p*(*x**i*) :* i* = 1*, ..., I*) is non-linear. Hence, this inverse problem is a non-linear inverse problem.


---

## Page 15

# **Chapter 2**
# **Generalised Solutions**

Functional analysis is the basis of the theory that we will cover in this course. We cannot recall all basic concepts of functional analysis and instead refer to popular textbooks that deal with this subject, e.g., [12, 37, 33]. Nevertheless, we shall recall a few important deﬁnitions that will be used in this lecture. We will focus on inverse problems with* bounded linear operators** A*, i.e.* A** ∈L*(*X**,** Y*) with

*∥**A**∥**L*(*X**,**Y*) := sup *u**∈X\{*0*}*

*∥**Au**∥**Y*

*∥**u**∥**X *= sup *∥**u**∥**X* ⩽1 *∥**Au**∥**Y** <** ∞**.*

For* A*:* X →Y* we further want to denote by

1.* D*(*A*) :=* X* the domain,

2.* N*(*A*) :=* {**u** ∈X |** Au* = 0*}* the kernel,

3.* R*(*A*) :=* {**f** ∈Y |** f* =* Au, u** ∈X}* the range

of* A*. We say that* A* is continuous at* u** ∈X* if for all* ε >* 0 there exists* δ >* 0 with

*∥**Au** −**Av**∥**Y* ⩽*ε* for all* v** ∈X* with* ∥**u** −**v**∥**X* ⩽*δ.*

For linear* K* it can be shown that continuity is equivalent to boundedness, i.e. the existence of a constant* C >* 0 such that

*∥**Au**∥**Y* ⩽*C**∥**u**∥**X*

for all* u** ∈X*. Note that this constant* C* actually equals the operator norm* ∥**A**∥**L*(*X**,**Y*). In this Chapter we only consider* A** ∈L*(*X**,** Y*) with* X* and* Y* being Hilbert spaces. From functional calculus we know that every Hilbert space* U* is equipped with a* scalar product*, which we are going to denote by* ⟨·**,** ·⟩**U* (or simply* ⟨·**,** ·⟩*, whenever the space is clear from the context). In analogy to the transpose of a matrix, this scalar product structure together with the theorem of Fr´echet-Riesz [37, Section 2.10, Theorem 2.E] allows us to deﬁne the (unique)* adjoint operator* of* A*, denoted with* A*^(*∗*), as follows:

*⟨**Au, v**⟩**Y* =* ⟨**u, A*^(*∗*)*v**⟩**X** ,* for all* u** ∈X**, v** ∈Y**.*

15


---

## Page 16

16 *CHAPTER 2. GENERALISED SOLUTIONS*

In addition to that, a scalar product can be used to deﬁne orthogonality. Two elements *u, v** ∈X* are said to be* orthogonal* if* ⟨**u, v**⟩*= 0. For a subset* X** *^(*′*)^(* *)*⊂X* the* orthogonal complement* of* X** *^(*′*)^( )in* X* is deﬁned as

*X** *^(*′⊥*):=  *u** ∈X | ⟨**u, v**⟩**X* = 0 for all* v** ∈X** *^(*′*)^(	 )*.*

One can show that* X** *^(*′⊥*)is a closed subspace and that* X** *^(*⊥*)=* {*0*}*. Moreover, we have that *X** *^(*′*)^(* *)*⊂*(*X** *^(*′⊥*))^(*⊥*). If* X** *^(*′*)^( )is a closed subspace then we even have* X** *^(*′*)^( )= (*X** *^(*′⊥*))^(*⊥*). In this case there exists the* orthogonal decomposition*

*X* =* X** *^(*′*)^(* *)*⊕X** *^(*′⊥*)*,*

which means that every element* u** ∈X* can uniquely be represented as

*u* =* x* +* x*^(*⊥*)with* x** ∈X** *^(*′*)^( )and* x*^(*⊥*)*∈X** *^(*′⊥*)*,*

see for instance [37, Section 2.9, Corollary 1]. The mapping* u** 7→**x* deﬁnes a linear operator* P**X** ′** ∈L*(*X**,** X*) that is called* orthogonal projection* on* X** *^(*′*).

**Lemma 2.0.1** (cf. [28, Section 5.16])**.*** Let** X** *^(*′*)^(* *)*⊂X** be a closed subspace. The orthogonal projection onto** X** *^(*′*)^(* *)*satisﬁes the following conditions:*

*1.** P**X** ′** is self-adjoint, i.e.** P** *^(*∗ *)*X** *^(*′*)^( =)^(* P*)^(*X*)^(* ′*)^(*,*)

*2.** ∥**P**X** ′**∥**L*(*X**,**X*) = 1* (if** X** *^(*′*)^(* *)*̸*=* {*0*}**),*

*3.** I** −**P**X** ′* =* P**X** ′⊥**,*

*4.** ∥**u** −**P**X** ′**u**∥**X* ⩽*∥**u** −**v**∥**X** for all** v** ∈X** *^(*′*)*,*

*5.** x* =* P**X** ′**u** if and only if** x** ∈X** *^(*′*)^(* *)*and** u** −**x** ∈X** *^(*′⊥*)*.*

**Remark 2.0.2.** Note that for a non-closed subspace* X** *^(*′*)^( )we only have (*X** *^(*′⊥*))^(*⊥*)=* X** *^(*′*). For *A** ∈L*(*X**,** Y*) we therefore have

•* R*(*A*)^(*⊥*)=* N*(*A*^(*∗*)) and thus* N*(*A*^(*∗*))^(*⊥*)=* R*(*A*),

•* R*(*A*^(*∗*))^(*⊥*)=* N*(*A*) and thus* N*(*A*)^(*⊥*)=* R*(*A*^(*∗*)).

Hence, we can deduce the following orthogonal decompositions

*X* =* N*(*A*)* ⊕R*(*A*^(*∗*)) and* Y* =* N*(*A*^(*∗*))* ⊕R*(*A*)*.*

We will also need the follwoing relationship between the ranges of* A*^(*∗*)and* A*^(*∗*)*A*.

**Lemma 2.0.3.*** Let** A** ∈L*(*X**,** Y*)*. Then** R*(*A*^(*∗*)*A*) =* R*(*A*^(*∗*))*.*

*Proof.* It is clear that* R*(*A*^(*∗*)*A*) =* R*(*A*^(*∗*)*|**R*(*A*))* ⊆R*(*A*^(*∗*)), so we are left to prove that* R*(*A*^(*∗*))* ⊆ R*(*A*^(*∗*)*A*). Let* u** ∈R*(*A*^(*∗*)) and let* ε >* 0. Then, there exists* f** ∈N*(*A*^(*∗*))^(*⊥*)=* R*(*A*) with* ∥**A*^(*∗*)*f**−**u**∥**X** < ε/*2 (recall the orthogonal decomposition in Remark 2.0.2). As* N*(*A*^(*∗*))^(*⊥*)=* R*(*A*), there exists *x** ∈X* such that* ∥**Ax** −**f**∥**Y** < ε/*(2*∥**A**∥**L*(*X**,**Y*)). Putting these together we have

*∥**A*^(*∗*)*Ax** −**u**∥**X* ⩽*∥**A*^(*∗*)*Ax** −**A*^(*∗*)*f**∥**X* +* ∥**A*^(*∗*)*f** −**u**∥**X *⩽*∥**A*^(*∗*)*∥**L*(*Y**,**X*)*∥**Ax** −**f**∥**Y *| {z } *<ε/*2

+* ∥**A*^(*∗*)*f** −**u**∥**X *| {z } *<ε/*2

*< ε*

which shows that* u** ∈R*(*A*^(*∗*)*A*) and thus also* R*(*A*^(*∗*))* ⊆R*(*A*^(*∗*)*A*).


---

## Page 17

*2.1. GENERALISED INVERSES *17

### **2.1 Generalised Inverses**

Recall the inverse problem *Au* =* f, *(2.1)

where* A*:* X →Y* is a linear bounded operator and* X* and* Y* are Hilbert spaces.

**Deﬁnition 2.1.1** (Minimal-norm solutions)**.*** An element** u** ∈X** is called*

•* a least-squares solution of* (2.1)* if*

*∥**Au** −**f**∥**Y* = inf*{∥**Av** −**f**∥**Y**, v** ∈X}*;

•* a minimal-norm solution of* (2.1)* (and is denoted by** u*^(*†*)*) if*

*∥**u*^(*†*)*∥**X* ⩽*∥**v**∥**X **for all least squares solutions** v**.*

**Remark 2.1.2.** Since* R*(*A*) is not closed in general (it is never closed for a compact operator, unless the range is ﬁnite-dimensional), a least-squares solution may not exist. If it exists, then the minimal-norm solution is unique (it is the orthogonal projection of the zero element onto an aﬃne subspace deﬁned by* ∥**Au** −**f**∥**Y* = min*{∥**Av** −**f**∥**Y**, v** ∈X}*).

In numerical linear algebra it is a well known fact that the normal equations can be used to compute least-squares solutions. The same holds true in the inﬁnite-dimensional case.

**Theorem 2.1.3.*** Let** f** ∈Y** and** A** ∈L*(*X**,** Y*)*. Then, the following three assertions are equivalent.*

*1.** u** ∈X** satisﬁes** Au* =* P**R*(*A*)*f**.*

*2.** u** is a least squares solution of the inverse problem* (2.1)*.*

*3.** u** solves the* normal equation *A*^(*∗*)*Au* =* A*^(*∗*)*f. *(2.2)

**Remark 2.1.4.** The name normal equation is derived from the fact that for any solution *u* its residual* Au** −**f* is orthogonal (normal) to* R*(*A*). This can be readily seen, as we have for any* v** ∈X* that

0 =* ⟨**v, A*^(*∗*)(*Au** −**f*)*⟩**X* =* ⟨**Av, Au** −**f**⟩**Y*

which shows* Au** −**f** ∈R*(*A*)^(*⊥*).

*Proof of Theorem 2.1.3.* For 1* ⇒*2: Let* u** ∈X* such that* Au* =* P**R*(*A*)*f* and let* v** ∈X* be arbitrary. With the basic properties of the orthogonal projection, Lemma 2.0.1 4, we have

*∥**Au** −**f**∥**Y* =* ∥**ff** −**P**R*(*A*)*f**∥**Y* ⩽ inf *g**∈R*(*A*) *∥**g** −**f**∥**Y* ⩽ inf *g**∈R*(*A*)* *^(*∥*)^(*g*)^(* −*)^(*f*)^(*∥*)^(*Y*)^( = inf )*v**∈X** *^(*∥*)^(*Av*)^(* −*)^(*f*)^(*∥*)^(*Y*)^(*,*)

which shows that* u* is a least squares solution. For 2* ⇒*3: Let* u** ∈X* be a least squares solution and let* v** ∈X* an arbitrary element. We deﬁne the quadratic polynomial* F* : R* →*R,

*F*(*λ*) :=* ∥**A*(*u* +* λv*)* −**f**∥*^(2 )*Y* ^(=)^(* λ*)^(2)^(*∥*)^(*Av*)^(*∥*)^(2 )*Y** *^(*−*)^(2)^(*λ*)^(* ⟨*)^(*Av, f*)^(* −*)^(*Au*)^(*⟩*)*Y* ^(+)^(* ∥*)^(*f*)^(* −*)^(*Au*)^(*∥*)^(2 )*Y* ^(.)


---

## Page 18

18 *CHAPTER 2. GENERALISED SOLUTIONS*

A necessary condition for* u** ∈X* to be a least squares solution is* F** *^(*′*)(0) = 0, which leads to *⟨**v, A*^(*∗*)(*f** −**Au*)*⟩**X* = 0. As* v* was arbitrary, it follows that the normal equation (2.2) must hold. For 3* ⇒*1: From the normal equation it follows that* A*^(*∗*)(*f** −**Au*) = 0, which is equivalent

to* f** −**Au** ∈R*(*A*)^(*⊥*), see Remark 2.1.4. Since* R*(*A*)^(*⊥*)= 

*R*(*A*) *⊥ *and* Au** ∈R*(*A*)* ⊂R*(*A*), the assertion follows from Lemma 2.0.1 5:

*Au* =* P**R*(*A*)*f** ⇔**Au** ∈R*(*A*) and* f** −**Au** ∈ *

*R*(*A*) *⊥ **.*

**Lemma 2.1.5.*** Let** f** ∈Y** and let* L* be the set of least squares solutions to the inverse problem* (2.1)*. Then,* L* is non-empty if and only if** f** ∈R*(*A*)* ⊕R*(*A*)^(*⊥*)*.*

*Proof.* Let* u** ∈*L. It is easy to see that* f* =* Au* + (*f** −**Au*)* ∈R*(*A*)* ⊕R*(*A*)^(*⊥*)as the normal equations are equivalent to* f** −**Au** ∈R*(*A*)^(*⊥*).

Consider now* f** ∈R*(*A*)*⊕R*(*A*)^(*⊥*). Then there exists* u** ∈X* and* g** ∈R*(*A*)^(*⊥*)= 

*R*(*A*) *⊥*

such that* f* =* Au*+*g* and thus* P**R*(*A*)*f* =* P**R*(*A*)*Au*+*P**R*(*A*)*g* =* Au* and the assertion follows from Theorem 2.1.3 1.

**Remark 2.1.6.** If the dimensions of* X* and* R*(*A*) are ﬁnite, then* R*(*A*) is closed, i.e. *R*(*A*) =* R*(*A*). Thus, in a ﬁnite dimensional setting, there always exists a least squares solution.

**Theorem 2.1.7.*** Let** f** ∈R*(*A*)*⊕R*(*A*)^(*⊥*)*. Then there exists a unique minimal norm solution **u*^(*†*)^(* *)*to the inverse problem* (2.1)* and all least squares solutions are given by** {**u*^(*†*)*}* +* N*(*A*)*.*

*Proof.* From Lemma 2.1.5 we know that there exists a least squares solution. As noted in Remark 2.1.2, in this case the minimal-norm solution is unique. Let* ϕ* be an arbitrary least-squares solution. Using Theorem 2.1.3 we get

*A*(*ϕ** −**u*^(*†*)) =* Aϕ** −**Au*^(*†*)^( )=* P**R*(*A*)*f** −**P**R*(*A*)*f* = 0*, *(2.3)

which shows that* ϕ** −**u*^(*†*)^(* *)*∈N*(*A*), hence the assertion.

If a least-squares solution exists for a given* f** ∈Y* then the minimal-norm solution can be computed (at least in theory) using the Moore-Penrose generalised inverse.

**Deﬁnition 2.1.8.*** Let** A** ∈L*(*X**,** Y*)* and let*

e*A* :=* A**|**N*(*A*)*⊥*:* N*(*A*)^(*⊥*)*→R*(*A*)

*denote the restriction of** A** to** N*(*A*)^(*⊥*)*. The* Moore-Penrose inverse* A*^(*†*)^(* *)*is deﬁned as the unique linear extension of* ^(e)*A*^(*−*)^(1)^(* *)*to*

*D*(*A*^(*†*)) =* R*(*A*)* ⊕R*(*A*)^(*⊥*)

*with **N*(*A*^(*†*)) =* R*(*A*)^(*⊥*)*.*


---

## Page 19

*2.1. GENERALISED INVERSES *19

**Remark 2.1.9.** Due to the restriction to* N*(*A*)^(*⊥*)and* R*(*A*) we have that ^(e)*A* is injective and surjective. Hence, ^(e)*A*^(*−*)^(1)^( )exists and is linear and – as a consequence –* A*^(*†*)^( )is well-deﬁned on *R*(*A*). Moreover, due to the orthogonal decomposition* D*(*A*^(*†*)) =* R*(*A*)*⊕R*(*A*)^(*⊥*), there exist for arbitrary* f** ∈D*(*A*^(*†*)) elements* f*1* ∈R*(*A*) and* f*2* ∈R*(*A*)^(*⊥*)with* f* =* f*1 +* f*2. Therefore, we have

*A*^(*†*)*f* =* A*^(*†*)*f*1 +* A*^(*†*)*f*2 =* A*^(*†*)*f*1 = ^(e)*A*^(*−*)^(1)*f*1 = ^(e)*A*^(*−*)^(1)*P**R*(*A*)*f* , (2.4)

where we used that* f*2* ∈R*(*A*)^(*⊥*)=* N*(*A*^(*†*)). Thus,* A*^(*†*)^( )is well-deﬁned on the entire domain *D*(*A*^(*†*)).

**Remark 2.1.10.** As orthogonal complements are always closed we get that

*D*(*A*^(*†*)) =* R*(*A*)* ⊕R*(*A*)^(*⊥*)=* Y**,*

and hence,* D*(*A*^(*†*)) is dense in* Y*. Thus, if* R*(*A*) is closed it follows that* D*(*A*^(*†*)) =* Y* and on the other hand,* D*(*A*^(*†*)) =* Y* implies* R*(*A*) is closed. We note that for ill-posed problems *R*(*A*) is usually not closed; for instance, if* A* is compact then* R*(*A*) is closed if and only if it is ﬁnite-dimensional [1, Ex.1 Section 7.1].

If* A* is bijective we have that* A*^(*†*)^( )=* A*^(*−*)^(1). We also highlight that the extension* A*^(*†*)^( )is not necessarily continuous.

**Theorem 2.1.11** ([20, Prop. 2.4])**.*** Let** A** ∈L*(*X**,** Y*)*. Then** A*^(*†*)^(* *)*is continuous, i.e.** A*^(*†*)^(* *)*∈ L*(*D*(*A*^(*†*))*,** X*)*, if and only if** R*(*A*)* is closed.*

**Example 2.1.12.** To illustrate the deﬁnition of the Moore-Penrose inverse we consider a simple example in ﬁnite dimensions. Let the linear operator* A*: R^(3)^(* *)*→*R^(2)^( )be given by

*Ax* = 2 0 0 0 0 0

 ^()

 *x*1 *x*2 *x*3



= 2*x*1 0

 *.*

It is easy to see that* R*(*A*) =* {**f** ∈*R^(2)^(* *)*|** f*2 = 0*}* and* N*(*A*) =* {**x** ∈*R^(3)^(* *)*|** x*1 = 0*}*. Thus, *N*(*A*)^(*⊥*)=* {**x** ∈*R^(3)^(* *)*|** x*2*, x*3 = 0*}*. Therefore, ^(e)*A*:* N*(*A*)^(*⊥*)*→R*(*A*), given by* x** 7→*(2*x*1*,* 0)^(*⊤*), is bijective and its inverse ^(e)*A*^(*−*)^(1)^( ):* R*(*A*)* →N*(*A*)^(*⊥*)is given by* f** 7→*(*f*1*/*2*,* 0*,* 0)^(*⊤*). To get the Moore-Penrose inverse* A*^(*†*), we need to extend ^(e)*A*^(*−*)^(1)^( )to* R*(*A*)* ⊕R*(*A*)^(*⊥*)in such a way that* A*^(*†*)*f* = 0 for all* f** ∈R*(*A*)^(*⊥*)=* {**f** ∈*R^(2)^(* *)*|** f*1 = 0*}*. It is easy to see that the Moore-Penrose inverse* A*^(*†*)^( ): R^(2)^(* *)*→*R^(3)^( )is given by the following expression

*A*^(*†*)*f* =



 1*/*2 0 0 0 0 0



 *f*1 *f*2

 =



 *f*1*/*2 0 0



*.*

Let us consider data ^(e)*f* = (8*,* 1)^(*⊤*)*̸∈R*(*A*). Then,* A*^(*†*)^( e)*f* =* A*^(*†*)(8*,* 1)^(*⊤*)= (4*,* 0*,* 0)^(*⊤*).

It can be shown that* A*^(*†*)^( )can be characterised by the Moore-Penrose equations.

**Theorem 2.1.13** ([20, Prop. 2.3])**.*** The Moore-Penrose inverse** A*^(*†*)^(* *)*satisﬁes** R*(*A*^(*†*)) = *N*(*A*)^(*⊥*)*and the Moore-Penrose equations*


---

## Page 20

20 *CHAPTER 2. GENERALISED SOLUTIONS*

*1.** A*^(*†*)*A* =* P**N*(*A*)*⊥**,*

*2.** AA*^(*†*)^( )=* P**R*(*A*)

 *D*(*A*^(*†*))^(*,*)

*3.** AA*^(*†*)*A* =* A**,*

*4.** A*^(*†*)*AA*^(*†*)^( )=* A*^(*†*)*,*

*where** P**N*(*A*)* and** P**R*(*A*)* denote the orthogonal projections on** N*(*A*)* and** R*(*A*)*, respectively.*

*Proof.* First, by the deﬁnition of the Moore-Penrose inverse we have for any* u** ∈X*

*A*^(*†*)*Au* =* A*^(*†*)*A*(*P**N*(*A*)*u* +* P**N*(*A*)*⊥**u*) =* A*^(*†*)*AP**N*(*A*)*⊥**u* = ^(e)*A*^(*−*)^(1)*AP**N*(*A*)*⊥**u* =* P**N*(*A*)*⊥**u,*

which proves 1. Now, for any* f** ∈D*(*A*^(*†*)) we have (see (2.4))

*AA*^(*†*)*f* =* A* ^(e)*A*^(*−*)^(1)*P**R*(*A*)*f* =* P**R*(*A*)*f,*

which proves 2. Applying* A* to 1., we get 3., and applying* A*^(*†*)^( )to 2., we get 4., which completes the proof.

**Corollary 2.1.14.** The Moore-Penrose inverse is uniquely characterised by 1.–2., that is, if a linear operator* B* :* R*(*A*)* ⊕R*(*A*)^(*⊥*)*→N*(*A*) satisﬁes* BA* =* P**N*(*A*)*⊥*and* AB* =* P**R*(*A*) then* B* =* A*^(*†*).

*Proof.* First we show that* B**|**R*(*A*) = ^(e)*A*^(*−*)^(1). Indeed, let* f* =* Au** ∈R*(*A*), where* u** ∈N*(*A*)^(*⊥*). Then *Bf* =* BAu* =* P**N*(*A*)*⊥**u* =* u* = ^(e)*A*^(*−*)^(1)*f,*

where the last equality holds since ^(e)*A* is bijective and hence uniquely invertible. Now we prove that* B**|**R*(*A*)*⊥*= 0. Indeed, for any* f** ∈R*(*A*)^(*⊥*)we have

*ABf* =* P**R*(*A*)*f* = 0*.*

Therefore,* B* is an extension of ^(e)*A*^(*−*)^(1)^( )to* R*(*A*)* ⊕R*(*A*)^(*⊥*)with* N*(*B*) =* R*(*A*)^(*⊥*). Since such an extension is unique,* B* =* A*^(*†*).

**Remark 2.1.15.** If an operator* B* satisﬁes only* ABA* =* A* (resp.* BAB* =* B*), it is called the* inner inverse* (resp.* outer inverse*) of* A*.

The next theorem shows that minimal-norm solutions can indeed be computed using the Moore-Penrose generalised inverse.

**Theorem 2.1.16.*** For each** f** ∈D*(*A*^(*†*))*, the minimal norm solution** u*^(*†*)^(* *)*to the inverse problem* (2.1)* is given via **u*^(*†*)^( )=* A*^(*†*)*f.*

*Proof.* As* f** ∈D*(*A*^(*†*)), we know from Theorem 2.1.7 that the minimal norm solution* u*^(*†*)

exists and is unique. With* u*^(*†*)^(* *)*∈N*(*A*)^(*⊥*), Lemma 2.1.13, and Theorem 2.1.3 we conclude that *u*^(*†*)^( )= (*I** −**P**N*(*A*))*u*^(*†*)^( )=* A*^(*†*)*Au*^(*†*)^( )=* A*^(*†*)*P**R*(*A*)*f* =* A*^(*†*)*AA*^(*†*)*f* =* A*^(*†*)*f.*


---

## Page 21

*2.2. COMPACT OPERATORS *21

As a consequence of Theorem 2.1.16 and Theorem 2.1.3, we ﬁnd that the minimum norm solution* u*^(*†*)^( )of* Au* =* f* is a minimum norm solution of the normal equation (2.2), i.e.

*u*^(*†*)^( )= (*A*^(*∗*)*A*)^(*†*)*A*^(*∗*)*f.*

Thus, in order to compute* u*^(*†*)^( )we can equivalently consider ﬁnding the minimum norm solution of the normal equation.

### **2.2 Compact Operators**

**Deﬁnition 2.2.1.*** Let** A** ∈L*(*X**,** Y*)*. Then** A** is said to be* compact* if for any bounded set **B** ⊂X** the closure of its image** A*(*B*)* is compact in** Y**. We denote the space of compact operators by** K*(*X**,** Y*)*.*

**Remark 2.2.2.** We can equivalently deﬁne an operator* A* to be compact if the image of a bounded sequence* {**u**j**}**j**∈*N* ⊂X* contains a convergent subsequence* {**Au**j**k**}**k**∈*N* ⊂Y*.

Compact operators are very common in inverse problems. In fact, almost all (linear) inverse problems involve the inversion of a compact operator. As the following result shows, compactness of the forward operator is a major source if ill-posedness.

**Theorem 2.2.3.*** Let** A** ∈K*(*X**,** Y*)* with an inﬁnite dimensional range. Then, the Moore- Penrose inverse of** A** is discontinuous.*

*Proof.* As the range* R*(*A*) is of inﬁnite dimension, we can conclude that* X* and* N*(*A*)^(*⊥*)

are also inﬁnite dimensional. We can therefore ﬁnd a sequence* {**u**j**}**j**∈*N with* u**j** ∈N*(*A*)^(*⊥*), *∥**u**j**∥**X* = 1 and* ⟨**u**j**, u**k**⟩**X* = 0 for* j** ̸*=* k*. Since* A* is a compact operator the sequence *f**j* =* Au**j* has a convergent subsequence, hence, for all* δ >* 0 we can ﬁnd* j, k* such that *∥**f**j** −**f**k**∥**Y** < δ*. However, we also obtain

*∥**A*^(*†*)*f**j** −**A*^(*†*)*f**k**∥*^(2 )*X* ^(=)^(* ∥*)^(*A*)^(*†*)^(*Au*)^(*j*)* *^(*−*)^(*A*)^(*†*)^(*Au*)^(*k*)^(*∥*)^(2 )*X *=* ∥**u**j** −**u**k**∥*^(2 )*X* ^(=)^(* ∥*)^(*u*)^(*j*)^(*∥*)^(2 )*X** *^(*−*)^(2)^(* ⟨*)^(*u*)^(*j*)^(*, u*)^(*k*)^(*⟩*)*X* ^(+)^(* ∥*)^(*u*)^(*k*)^(*∥*)^(2 )*X* ^(= 2)^(*,*)

which shows that* A*^(*†*)^( )is discontinuous. Here, the second identity follows from Lemma 2.1.13 1 and the fact that* u**j**, u**k** ∈N*(*A*)^(*⊥*).

To have a better understanding of when we have* f** ∈R*(*A*)*\R*(*A*) for compact operators *A*, we want to consider the singular value decomposition of compact operators.

**Singular value decomposition of compact operators**

**Theorem 2.2.4** ([23, p. 225, Theorem 9.16])**.*** Let** X** be a Hilbert space and** A** ∈K*(*X**,** X*)* be self-adjoint. Then there exists an orthonormal basis** {**x**j**}**j**∈*N* ⊂X** of** R*(*A*)* and a sequence of eigenvalues** {**λ**j**}**j**∈*N* ⊂*R* with** |**λ*1*|* ⩾*|**λ*2*|* ⩾*. . . >* 0* such that for all** u** ∈X** we have*

*Au* =

*∞ *X

*j*=1 *λ**j** ⟨**u, x**j**⟩**X** x**j** .*

*The sequence** {**λ**j**}**j**∈*N* is either ﬁnite or we have** λ**j** →*0*.*


---

## Page 22

22 *CHAPTER 2. GENERALISED SOLUTIONS*

**Remark 2.2.5.** The notation in the theorem above only makes sense if the sequence *{**λ**j**}**j**∈*N is inﬁnite. For the case that there are only ﬁnitely many* λ**j* the sum has to be interpreted as a ﬁnite sum. Moreover, as the eigenvalues are sorted by absolute value* |**λ**j**|*, we have* ∥**A**∥**L*(*X**,**X*) =* |**λ*1*|*.

If* A* is not self-adjoint, the decomposition in Theorem 2.2.4 does not hold any more. Instead, we can consider the so-called* singular value decomposition* of a compact linear operator.

**Theorem 2.2.6.*** Let** A** ∈K*(*X**,** Y*)*. Then there exists*

*1. a not-necessarily inﬁnite null sequence** {**σ**j**}**j**∈*N* with** σ*1 ⩾*σ*2 ⩾*. . . >* 0*,*

*2. an orthonormal basis** {**x**j**}**j**∈*N* ⊂X** of** N*(*A*)^(*⊥*)*,*

*3. an orthonormal basis** {**y**j**}**j**∈*N* ⊂Y** of** R*(*A*)* with*

*Ax**j* =* σ**j**y**j**, A*^(*∗*)*y**j* =* σ**j**x**j**, **for all** j** ∈*N*. *(2.5)

*Moreover, for all** u** ∈X** we have the representation*

*Au* =

*∞ *X

*j*=1 *σ**j** ⟨**u, x**j**⟩**y**j**. *(2.6)

*The sequence** {*(*σ**j**, x**j**, y**j*)*}** is called* singular system* or* singular value decomposition (SVD)* of** A**. For the adjoint operator** A*^(*∗*)*we have the representation*

*A*^(*∗*)*f* =

*∞ *X

*j*=1 *σ**j** ⟨**f, y**j**⟩**x**j **∀**f** ∈Y**. *(2.7)

*Proof.* Consider* B* =* A*^(*∗*)*A* and* C* =* AA*^(*∗*). Both* B* and* C* are compact, self-adjoint and positive semideﬁnite, so that by Theorem 2.2.4 both admit a spectral representation and, by positive semideﬁniteness, their eigenvalues are positive. Therefore, we can write

*Cf* =

*∞ *X

*j*=1 *σ*^(2 )*j** *^(*⟨*)^(*f, y*)^(*j*)^(*⟩*)^(*y*)^(*j *)*∀**f** ∈Y**,*

where* {**y**j**}* is an orthonormal basis of* R*(*AA*^(*∗*)) =* R*(*A*) (Lemma 2.0.3),* σ**j** >* 0 for all* j* and *σ**j** →*0 as* j** →∞*. Now consider the element* A*^(*∗*)*y**j** ∈X*. Since* σ*^(2 )*j* ^(is an eigenvalue of)^(* C*)^( for the eigenvector )*y**j*, we get that *σ*^(2 )*j** *^(*A*)^(*∗*)^(*y*)^(*j*) ^(=)^(* A*)^(*∗*)^(()^(*σ*)^(2 )*j** *^(*y*)^(*j*)^() =)^(* A*)^(*∗*)^(*Cy*)^(*j*) ^(=)^(* A*)^(*∗*)^(*AA*)^(*∗*)^(*y*)^(*j*) ^(=)^(* BA*)^(*∗*)^(*y*)^(*j*)

and therefore* σ*^(2 )*j* ^(is also an eigenvalue of)^(* B*)^( (for the eigenvector)^(* A*)^(*∗*)^(*y*)^(*j*)^(). Now we will show)

that the system n*A**∗**y**j*

*σ**j*

o

*j**∈*N ^(forms an orthonormal basis of)^(* R*)^(()^(*A*)^(*∗*)^() =)^(* N*)^(()^(*A*)^())^(*⊥*)^(. Indeed, we)

have *A**∗**y**j*

*σ**j **, *^(*A*)^(*∗*)^(*y*)^(*k*)

*σ**k*

 = 1 *σ**j**σ**k **⟨**y**j**, AA*^(*∗*)*y**k**⟩*= 1 *σ**j**σ**k*

 *y**j**, σ*^(2 )*k*^(*y*)^(*k *) =

( 1*, *if* j* =* k, *0*, *otherwise*.*


---

## Page 23

*2.2. COMPACT OPERATORS *23

Hence, n*A**∗**y**j*

*σ**j*

o

*j**∈*N ^(are orthonormal. It is also clear that they are dense in)^(* R*)^(()^(*A*)^(*∗*)^() =)^(* N*)^(()^(*A*)^())^(*⊥*)^(,)

hence they form a basis. Therefore, we can choose* {**x**j**}**j**∈*N = n*A**∗**y**j*

*σ**j*

o

*j**∈*N^(, i.e.)

*x**j* =* σ*^(*−*)^(1 )*j** *^(*A*)^(*∗*)^(*y*)^(*j*)

and we get (by construction) that *A*^(*∗*)*y**j* =* σ**j**x**j**.*

We also observe that *Ax**j* =* σ*^(*−*)^(1 )*j** *^(*AA*)^(*∗*)^(*y*)^(*j*)^( =)^(* σ*)^(*−*)^(1 )*j** *^(*σ*)^(2 )*j** *^(*y*)^(*j*) ^(=)^(* σ*)^(*j*)^(*y*)^(*j*)^(*,*)

which proves (2.5). Extending the basis* {**x**j**}* of* R*(*A*^(*∗*)) to a basis* {*e*x**j**}* of* X*, we expand an arbitrary* u** ∈X* as *u* = ^(P)^(*∞ *)*j*=1* *^(*⟨*)^(*u,*)^( e)^(*x*)^(*j*)^(*⟩*)^(e)^(*x*)^(*j*)^(. Applying)^(* A*)^( and using the fact that)^(* X*)^( =)^(* N*)^(()^(*A*)^())^(*⊕R*)^(()^(*A*)^(*∗*)^() (Remark 2.0.2), )we obtain the singular value decomposition (2.6) (and also (2.7) in a similar manner)

*Au* =

*∞ *X

*j*=1 *σ**j** ⟨**u, x**j**⟩**y**j **∀**u** ∈X**, A*^(*∗*)*f* =

*∞ *X

*j*=1 *σ**j** ⟨**f, y**j**⟩**x**j **∀**f** ∈Y**.*

We can now derive a representation of the Moore-Penrose inverse in terms of the singular value decomposition.

**Theorem 2.2.7.*** Let** A** ∈K*(*X**,** Y*)* with singular system** {*(*σ**j**, x**j**, y**j*)*}**j**∈*N* and** f** ∈D*(*A*^(*†*))*. Then the Moore-Penrose inverse of** A** can be written as*

*A*^(*†*)*f* =

*∞ *X

*j*=1 *σ*^(*−*)^(1 )*j **⟨**f, y**j**⟩**x**j** . *(2.8)

*Proof.* We know that, since* f** ∈D*(*A*^(*†*)),* u*^(*†*)^( )=* A*^(*†*)*f* solves the normal equations

*A*^(*∗*)*Au*^(*†*)^( )=* A*^(*∗*)*f.*

From Theorem 2.2.6 we know that

*A*^(*∗*)*Au*^(*†*)^( )=

*∞ *X

*j*=1 *σ*^(2 )*j *D *u*^(*†*)*, x**j *E *x**j**, A*^(*∗*)*f* =

*∞ *X

*j*=1 *σ**j** ⟨**f, y**j**⟩**x**j**, *(2.9)

which implies that D *u*^(*†*)*, x**j *E =* σ*^(*−*)^(1 )*j **⟨**f, y**j**⟩*

Expanding* u*^(*†*)^(* *)*∈N*(*A*)^(*⊥*)in the basis* {**x**j**}*, we get

*u*^(*†*)^( )=

*∞ *X

*j*=1

D *u*^(*†*)*, x**j *E *x**j* =

*∞ *X

*j*=1 *σ*^(*−*)^(1 )*j **⟨**f, y**j**⟩**x**j* =* A*^(*†*)*f.*


---

## Page 24

24 *CHAPTER 2. GENERALISED SOLUTIONS*

The representation (2.8) makes it clear again that the Moore-Penrose inverse is un- bounded if* R*(*A*) is inﬁnite dimensional. Indeed, taking the sequence* y**j* we note that *∥**A*^(*†*)*y**j**∥*=* σ*^(*−*)^(1 )*j **→∞*, although* ∥**y**j**∥*= 1. The unboundedness of the Moore-Penrose inverse is also reﬂected in the fact that the series in (2.8) may not converge for a given* f*. The convergence criterion for the series is called the* Picard criterion*.

**Deﬁnition 2.2.8.*** We say that the data** f** satisfy the Picard criterion, if*

*∥**A*^(*†*)*f**∥*^(2)^( )=

*∞ *X

*j*=1

*|⟨**f, y**j**⟩|*^(2)

*σ*^(2 )*j **<** ∞**. *(2.10)

**Remark 2.2.9.** The Picard criterion is a condition on the decay of the coeﬃcients* ⟨**f, y**j**⟩*. As the singular values* σ**j* decay to zero as* j** →∞*, the Picard criterion is only met if the coeﬃcients* ⟨**f, y**j**⟩*decay suﬃciently fast. In case the singular system is given by the Fourier basis, then the coeﬃcients* ⟨**f, y**j**⟩*are just the Fourier coeﬃcients of* f*. Therefore, the Picard criterion is a condition on the decay of the Fourier coeﬃcients which is equivalent to the smoothness of* f*.

It turns our that the Picard criterion also can be used to characterise elements in the range of the forward operator.

**Theorem 2.2.10.*** Let** A** ∈K*(*X**,** Y*)* with singular system** {*(*σ**j**, x**j**, y**j*)*}**j**∈*N*, and** f** ∈R*(*A*)*. Then** f** ∈R*(*A*)* if and only if the* Picard criterion

*∞ *X

*j*=1

*⟨**f, y**j**⟩**Y *2

*σ*^(2 )*j **<** ∞ *(2.11)

*is met.*

*Proof.* Let* f** ∈R*(*A*), thus there is a* u** ∈X* such that* Au* =* f*. It is easy to see that we have

*⟨**f, y**j**⟩**Y* =* ⟨**Au, y**j**⟩**Y* =* ⟨**u, A*^(*∗*)*y**j**⟩**X* =* σ**j** ⟨**u, x**j**⟩**X*

and therefore

*∞ *X

*j*=1 *σ*^(*−*)^(2 )*j** *^(*| ⟨*)^(*f, y*)^(*j*)^(*⟩*)*Y** *^(*|*)^(2)^( =)

*∞ *X

*j*=1 *| ⟨**u, x**j**⟩**X** |*^(2)^( )⩽*∥**u**∥*^(2 )*X** *^(*<*)^(* ∞*)^(.)

Now let the Picard criterion (2.11) hold and deﬁne* u* := ^(P)^(*∞ *)*j*=1* *^(*σ*)^(*−*)^(1 )*j **⟨**f, y**j**⟩**Y** x**j** ∈X*. It is well-deﬁned by the Picard criterion (2.11) and we conclude

*Au* =

*∞ *X

*j*=1 *σ*^(*−*)^(1 )*j **⟨**f, y**j**⟩**Y** Ax**j* =

*∞ *X

*j*=1 *⟨**f, y**j**⟩**Y** y**j* =* P**R*(*A*)*f* =* f* ,

which shows* f** ∈R*(*A*).

Although all ill-posed problems are not easy to solve, some are worse than others, depending on how fast the singular values decay to zero.


---

## Page 25

*2.2. COMPACT OPERATORS *25

**Deﬁnition 2.2.11.*** We say that an ill-posed inverse problem* (2.1)* is* mildly ill-posed* if the singular values decay at most with polynomial speed, i.e. there exist** γ, C >* 0* such that **σ**j* ⩾*Cj*^(*−*)^(*γ*)^(* *)*for all** j**. We call the ill-posed inverse problem* severely ill-posed* if its singular values decay faster than with polynomial speed, i.e. for all** γ, C >* 0* one has that** σ**j* ⩽*Cj*^(*−*)^(*γ*)

*for** j** suﬃciently large.*

**Example 2.2.12.** Let us consider the example of diﬀerentiation again, as introduced in Section 1.2.3. The forward operator* A*:* L*^(2)([0*,* 1])* →**L*^(2)([0*,* 1]) in this problem is given by

(*Au*)(*t*) = Z* t*

0 *u*(*s*)* ds* = Z 1

0 *K*(*s, t*)*u*(*s*)* ds* ,

with* K* : [0*,* 1]* ×* [0*,* 1]* →*R deﬁned as

*K*(*s, t*) :=

( 1 *s* ⩽*t *0 else .

This is a special case of the integral operators as introduced in Section 1.2.1. Since the kernel* K* is square integrable,* A* is compact. The adjoint operator* A*^(*∗*)is given via

(*A*^(*∗*)*f*)(*s*) = Z 1

0 *K*(*t, s*)*f*(*t*)* dt* = Z 1

*s **v*(*t*)* dt* . (2.12)

Now we want to compute the eigenvalues and eigenvectors of* A*^(*∗*)*A*, i.e. we look for* σ*^(2)

and* x** ∈**L*^(2)([0*,* 1]) with

*σ*^(2)*x*(*s*) = (*A*^(*∗*)*Ax*)(*s*) = Z 1

*s*

Z* t*

0 *x*(*r*)* dr dt* .

We immediately observe* x*(1) = 0 and further

*σ*^(2)*x*^(*′*)(*s*) =* *^(*d*)

*ds*

Z 1

*s*

Z* t*

0 *x*(*r*)* dr dt* =* − *Z* s*

0 *x*(*r*)* dr* ,

from which we conclude* x*^(*′*)(0) = 0. Taking the derivative another time thus yields the ordinary diﬀerential equation

*σ*^(2)*x*^(*′′*)(*s*) +* x*(*s*) = 0 ,

for which solutions are of the form

*x*(*s*) =* c*1 sin(*σ*^(*−*)^(1)*s*) +* c*2 cos(*σ*^(*−*)^(1)*s*) ,

with some constants* c*1*, c*2. In order to satisfy the boundary conditions* x*(1) =* c*1 sin(*σ*^(*−*)^(1))+ *c*2 cos(*σ*^(*−*)^(1)) = 0 and* x*^(*′*)(0) =* c*1 = 0, we chose* c*1 = 0 and* σ* such that cos(*σ*^(*−*)^(1)) = 0. Hence, we have

*σ**j* = 2 (2*j** −*1)*π* ^(for)^(* j*)^(* ∈*)^(N)^( ,)


---

## Page 26

26 *CHAPTER 2. GENERALISED SOLUTIONS*

and by choosing* c*2 = *√*

2 we obtain the following normalised representation of* x**j*:

*x**j*(*s*) = *√*

2 cos  *j** −*^(1)

2

 *πs * .

According to (2.5) we further obtain

*y**j*(*s*) =* σ*^(*−*)^(1 )*j* ^(()^(*Ax*)^(*j*)^()()^(*s*)^() = ) *j** −*^(1)

2

 *π *Z* s*

0

*√*

2 cos  *j** −*^(1)

2

 *πt * *dt* = *√*

2 sin  *j** −*^(1)

2

 *πs * ,

and hence, for* f** ∈**L*^(2)([0*,* 1]) the Picard criterion becomes

2

*∞ *X

*j*=1 *σ*^(*−*)^(2 )*j*

Z 1

0 *f*(*s*) sin  *σ*^(*−*)^(1 )*j** *^(*s *) *ds *2 *<** ∞*.

Expanding* f* in the basis* {**y**j**}*

*f*(*t*) = 2

*∞ *X

*j*=1

Z 1

0 *f*(*s*) sin  *σ*^(*−*)^(1 )*j** *^(*s *) *ds * sin  *σ*^(*−*)^(1 )*j** *^(*t *)

and formally diﬀerentiating the series, we obtain

*f*^(*′*)(*t*) = 2

*∞ *X

*j*=1 *σ*^(*−*)^(1 )*j*

Z 1

0 *f*(*s*) sin  *σ*^(*−*)^(1 )*j** *^(*s *) *ds * cos  *σ*^(*−*)^(1 )*j** *^(*t *) *.*

Therefore, the Picard criterion is nothing but the condition for the legitimacy of such diﬀer- entiation, i.e. for the diﬀerentiability of the Fourier series by diﬀerentiating its components, and it holds if* f* is diﬀerentiable and* f*^(*′*)^(* *)*∈**L*^(2)([0*,* 1]). From the decay of the singular values we see that this inverse problem is mildly ill-posed.

**Example 2.2.13** (Heat equation)**.** Consider the problem of recovering the initial condition *u* of the heat equation from an observation* f* of the solution at some time* T >* 0 (see Section 1.2.2). We consider the heat equation on (0*, π*)* ×* R+, with Dirichlet boundary conditions      

    

*v**t** −**v**xx* = 0 on (0*, π*)* ×* R+*, v*(0*, t*) =* v*(*π, t*) = 0 on R+*, v*(*x, T*) =* f*(*x*) on (0*, π*)*, v*(*x,* 0) =* u*(*x*) on (0*, π*)*.*

The solution to the forward problem (determine* f* given* u*) is given by

*f* =* Au* :=

*∞ *X

*j*=1 *e*^(*−*)^(*j*)^(2)^(*T*)^( )b*u**j* sin(*jx*)*,*

where b*u**j* =* ⟨**u,* sin(*j**·*)*⟩*are Fourier coeﬃcients of* u*. Hence, singular values of* A* are given by *σ**j* =* e*^(*−*)^(*j*)^(2)^(*T*)^(* *)*, j** ∈*N*,*

and 1 *σ**j *=* e*^(*j*)^(2)^(*T*)^(* *)*.*

Singular values decay exponentially and the inverse problem is severely (exponentially) ill-posed.


---

## Page 27

# **Chapter 3**
# **Classical Regularisation Theory**
### **3.1 What is Regularisation?**

We have seen that the Moore-Penrose inverse* A*^(*†*)^( )is unbounded if* R*(*A*) is not closed. There- fore, given noisy data* f**δ* such that* ∥**f**δ** −**f**∥*⩽*δ*, we cannot expect convergence* A*^(*†*)*f**δ** →**A*^(*†*)*f *as* δ** →*0. To achieve convergence, we replace* A*^(*†*)^( )with a family of well-posed (bounded) operators* R**α* with* α* =* α*(*δ, f**δ*) and require that* R**α*(*δ,f**δ*)(*f**δ*)* →**A*^(*†*)*f* for all* f** ∈D*(*A*^(*†*)) and all* f**δ** ∈Y* s.t.* ∥**f** −**f**δ**∥**Y* ⩽*δ* as* δ** →*0. We remind ourselves that* L*(*X**,** Y*) denotes the space of all bounded (equivalently, con- tinuous) operators* X →Y*.

**Deﬁnition 3.1.1.*** Let** A** ∈L*(*X**,** Y*)* be a bounded operator. A family** {**R**α**}**α>*0* of continuous operators is called* regularisation* (or* regularisation operator*) of** A*^(*†*)^(* *)*if*

*R**α**f** →**A*^(*†*)*f* =* u*^(*†*)

*for all** f** ∈D*(*A*^(*†*))* as** α** →*0*.*

**Deﬁnition 3.1.2.*** If the family** {**R**α**}**α>*0* consists of linear operators, then one speaks of *linear regularisation* of** A*^(*†*)*.*

Hence, a regularisation is a pointwise approximation of the Moore–Penrose inverse with continuous operators. As in the interesting cases the Moore–Penrose inverse may not be continuous we cannot expect that the norm of* R**α* stays bounded as* α** →*0. This is conﬁrmed by the following results (in the linear case).

**Theorem 3.1.3** (Banach–Steinhaus e.g. [12, p. 78], [38, p. 173])**.*** Let** X**,** Y** be Hilbert spaces and** {**A**j**}**j**∈*N* ⊂L*(*X**,** Y*)* a family of point-wise bounded operators, i.e. for all** u** ∈X **there exists a constant** C*(*u*)* >* 0* s.t.* sup*j**∈*N* ∥**A**j**u**∥**Y* ⩽*C*(*u*)*. Then*

sup *j**∈*N *∥**A**j**∥**L*(*X**,**Y*)* <** ∞**.*

**Corollary 3.1.4** ([38, p. 174])**.** Let* X**,** Y* be Hilbert spaces and* {**A**j**}**j**∈*N* ⊂L*(*X**,** Y*). Then the following two conditions are equivalent:

1. There exists* A** ∈L*(*X**,** Y*) such that

*Au* = lim *j**→∞*^(*A*)^(*j*)^(*u *)for all* u** ∈X*.

27


---

## Page 28

28 *CHAPTER 3. CLASSICAL REGULARISATION THEORY*

2. There is a dense subset* X** *^(*′*)^(* *)*⊂X* such that lim*j**→∞**A**j**u* exists for all* u** ∈X** *^(*′*)^( )and

sup *j**∈*N *∥**A**j**∥**L*(*X**,**Y*)* <** ∞**.*

**Theorem 3.1.5.*** Let** X**,** Y** be Hilbert spaces,** A** ∈L*(*X**,** Y*)* and** {**R**α**}**α>*0* a linear regulari- sation as deﬁned in Deﬁnition 3.1.2. If** A*^(*†*)^(* *)*is not continuous,** {**R**α**}**α>*0* cannot be uniformly bounded. In particular, there exist** f** ∈Y** and a sequence** α**j** →*0* such that** ∥**R**α**j**f**∥→∞**as **j** →∞**.*

*Proof.* We prove the theorem by contradiction and assume that* {**R**α**}**α>*0 is uniformly bounded. Hence, there exists a constant* C* with* ∥**R**α**∥**L*(*Y**,**X*) ⩽*C* for all* α >* 0. Due to Deﬁnition 3.1.1, we have* R**α**j** →**A*^(*†*)^( )on* D*(*A*^(*†*)) for any sequence* α**j** →*0. Since* D*(*A*^(*†*)) is dense in* Y*, by Corollary 3.1.4 we get that* R**α**j* converges on* D*(*A*^(*†*)) =* Y* and therefore *A*^(*†*)^( )can be extended to a bounded operator on* L*(*Y**,** X*), which is a contradiction to the assumption that* A*^(*†*)^( )is not continuous (on* D*(*A*^(*†*))). To prove the second statement, assume that for all* f** ∈Y* and any sequence* α**j** →*0 we have sup *j**∈*N *∥**R**α**j**f**∥**Y* ⩽*C*(*f*)* <** ∞**.*

Then by Theorem 3.1.3 we have that

sup *j**∈*N *∥**R**α**j**∥**L*(*Y**,**X*) ⩽*C <** ∞**,*

which contradicts the ﬁrst part of the proof.

With the additional assumption that* ∥**AR**α**∥**L*(*X**,**X*) is bounded, we can even show that *R**α**f* diverges for all* f** ̸∈D*(*A*^(*†*)).

**Theorem 3.1.6.*** Let** A** ∈L*(*X**,** Y*)* and** {**R**α**}**α>*0* be a linear regularisation of** A*^(*†*)*. If*

sup *α>*0 *∥**AR**α**∥**L*(*X**,**X*)* <** ∞**,*

*then** ∥**R**α**f**∥**X** →∞**for all** f** ̸∈D*(*A*^(*†*))*.*

*Proof.* Deﬁne* u**α* :=* R**α**f* for* f** ̸∈D*(*A*^(*†*)). Assume that there exists a sequence* α**k** →*0 such that* ∥**u**α**k**∥**X* is uniformly bounded. Since bounded sets in a Hilbert space are weakly pre-compact, there exists a weakly convergent subsequence* u**α**kl* with some limit* u** ∈X*, cf. [21, Section 2.2, Theorem 2.1]. As continuous linear operators are also weakly continuous, we further have* Au**α**kl** ⇀Au*. On the other hand, for any* g** ∈D*(*A*^(*†*)) we have that* AR**α**kl**g** →**AA*^(*†*)*g* =* P**R*(*A*)*g* as *l** →∞*. By Corollary 3.1.4 we then conclude that this also holds for any* f** ∈Y*, i.e. also for* f** ̸∈D*(*A*^(*†*)). Hence, we get that

*AR**α**kl**f** →**P**R*(*A*)*f*

and (see ﬁrst part of proof) *AR**α**kl**f* =* Au**α**kl** ⇀Au.*

Therefore, we get that* Au* =* P**R*(*A*)*f*. Since* Y* =* R*(*A*)* ⊕R*(*A*)^(*⊥*), we get that* f** ∈R*(*A*)* ⊕*

*R*(*A*)^(*⊥*)=* D*(*A*^(*†*)) in contradiction to the assumption* f /**∈D*(*A*^(*†*)).


---

## Page 29

*3.2. PARAMETER CHOICE RULES *29

high low regularisation

high

low

error

data error approximation error total error

Figure 3.1: The* total error* between a regularised solution and the minimal norm solu- tion decomposes into the* data error* and the* approximation error*. These two errors have opposing trends: For a small regularisation parameter* α* the error in the data gets ampli- ﬁed through the ill-posedness of the problem and for large* α* the operator* R**α* is a poor approximation of the Moore–Penrose inverse.

### **3.2 Parameter Choice Rules**

We have stated in the beginning of this chapter that we would like to obtain a regularisation that would guarantee that* R**α*(*f**δ*)* →**A*^(*†*)*f* for all* f** ∈D*(*A*^(*†*)) and all* f**δ** ∈Y* s.t.* ∥**f** −**f**δ**∥**Y* ⩽*δ *as* δ** →*0. This means that the parameter* α*, referred to as the* regularisation parameter*, needs to be chosen as a function of* δ* (and perhaps also* f**δ*) so that* α** →*0 as* δ** →*0 (i.e. we need to regularise less as the data get more precise). This can be illustrated with the following observation. For linear regularisations we can split the* total error* between the regularised solution of the noisy problem* R**α**f**δ* and the minimal norm solution of the noise-free problem* u*^(*†*)^( )=* A*^(*†*)*f* as

*∥**R**α**f**δ** −**u*^(*†*)*∥**X* ⩽*∥**R**α**f**δ** −**R**α**f**∥**X* +* ∥**R**α**f** −**u*^(*†*)*∥**X*

⩽*δ**∥**R**α**∥**L*(*Y**,**X*) | {z } data error

+* ∥**R**α**f** −**A*^(*†*)*f**∥**X *| {z } approximation error

. (3.1)

The ﬁrst term of (3.1) is the* data error*; this term unfortunately does not stay bounded for* α** →*0, which we can conclude from Theorem 3.1.5. The second term, known as the *approximation error*, however vanishes for* α** →*0, due to the pointwise convergence of* R**α *to* A*^(*†*). Hence it becomes evident from (3.1) that a good choice of* α* depends on* δ*, and needs to be chosen such that the approximation error becomes as small as possible, whilst the data error is being kept at bay. See Figure 3.1 for an illustration. Parameter choice rules are deﬁned as follows.

**Deﬁnition 3.2.1.*** A function** α*: R*>*0* × Y →*R*>*0*,* (*δ, f**δ*)* 7→**α*(*δ, f**δ*)* is called a* parameter choice rule*. We distinguish between*

*1. a priori parameter choice rules, which depend on** δ** only;*

*2. a posteriori parameter choice rules, which depend on both** δ** and** f**δ**;*


---

## Page 30

30 *CHAPTER 3. CLASSICAL REGULARISATION THEORY*

*3. heuristic parameter choice rules, which depend on** f**δ** only.*

Now we are ready to deﬁne a regularisation that ensures the convergence* R**α*(*δ,f**δ*)(*f**δ*)* → **A*^(*†*)*f* as* δ** →*0.

**Deﬁnition 3.2.2.*** Let** {**R**α**}**α>*0* be a regularisation of** A*^(*†*)*. If for all** f** ∈D*(*A*^(*†*))* there exists a parameter choice rule** α* : R*>*0* × Y →*R*>*0* such that*

lim *δ**→*0 sup *f**δ* :* ∥**f**−**f**δ**∥**Y*⩽*δ **∥**R**α**f**δ** −**A*^(*†*)*f**∥**X* = 0 (3.2)

*and*

lim *δ**→*0 sup *f**δ* :* ∥**f**−**f**δ**∥**Y*⩽*δ **α*(*δ, f**δ*) = 0 (3.3)

*then the pair* (*R**α**, α*)* is called a* convergent regularisation*.*

**3.2.1 A priori parameter choice rules**

First of all we want to discuss a priori parameter choice rules in more detail. Historically, they were the ﬁrst to be studied. For every regularisation there exists an a priori parameter choice rule and thus a convergent regularisation.

**Theorem 3.2.3** ([20, Prop 3.4])**.*** Let** {**R**α**}**α>*0* be a regularisation of** A*^(*†*)*, for** A** ∈L*(*X**,** Y*)*. Then there exists an a priori parameter choice rule** α* =* α*(*δ*)* such that* (*R**α**, α*)* is a conver- gent regularisation.*

For linear regularisations, an important characterisation of a priori parameter choice strategies that lead to convergent regularisation methods is as follows.

**Theorem 3.2.4.*** Let** {**R**α**}**α>*0* be a linear regularisation, and** α* : R*>*0* →*R*>*0* an a priori parameter choice rule. Then* (*R**α**, α*)* is a convergent regularisation method if and only if*

*a)* lim*δ**→*0* α*(*δ*) = 0

*b)* lim*δ**→*0* δ**∥**R**α*(*δ*)*∥**L*(*Y**,**X*) = 0

*Proof.** ⇐*: Let condition a) and b) be fulﬁlled. From (3.1) we then observe that for any *f** ∈D*(*A*^(*†*)) and* f**δ** ∈Y* s.t.* ∥**f** −**f**δ**∥**Y* ⩽*δ **R**α*(*δ*)*f**δ** −**A**†**f * *X** *^(*→*)^(0 for)^(* δ*)^(* →*)^(0.)

Hence, (*R**α**, α*) is a convergent regularisation method. *⇒*: Now let (*R**α**, α*) be a convergent regularisation method. We prove that conditions 1 and 2 have to follow from this by showing that violation of either one of them leads to a contradiction to (*R**α**, α*) being a convergent regularisation method. If condition a) is violated, (3.3) is violated and hence, (*R**α**, α*) is not a convergent regularisation method. If condition a) is fulﬁlled but condition b) is violated, there exists a null sequence* {**δ**k**}**k**∈*N with *δ**k**∥**R**α*(*δ**k*)*∥**L*(*Y**,**X*) ⩾*C >* 0, and hence, we can ﬁnd a sequence* {**g**k**}**k**∈*N* ⊂Y* with* ∥**g**k**∥**Y* = 1 and* δ**k**∥**R**α*(*δ**k*)*g**k**∥**X* ⩾^(e)*C* for some ^(e)*C*. Let* f** ∈D*(*A*^(*†*)) be arbitrary and deﬁne* f**k* :=* f* +* δ**k**g**k*. Then we have on the one hand* ∥**f** −**f**k**∥**Y* ⩽*δ**k*, but on the other hand the norm of

*R**α*(*δ**k*)*f**k** −**A*^(*†*)*f* =* R**α*(*δ**k*)*f** −**A*^(*†*)*f* +* δ**k**R**α*(*δ**k*)*g**k*

cannot converge to zero, as the second term* δ**k**R**α*(*δ**k*)*g**k* is bounded from below by a positive constant* C* by construction. Hence, (3.2) is violated for* f**δ* =* f* +* δ**k**g**k* and thus, (*R**α**, α*) is not a convergent regularisation method.


---

## Page 31

*3.2. PARAMETER CHOICE RULES *31

**3.2.2 A posteriori parameter choice rules**

It is easy to convince oneself that if an a priori parameter choice rule* α* =* α*(*δ*) deﬁnes a convergence regularisation then e*α* =* α*(*Cδ*) with any* C >* 0 also deﬁnes a convergent regu- larisation (for linear regularisations, it is a trivial corollary of Theorem 3.2.4). Therefore, from the asymptotic point of view, all these regularisations are equivalent. For a ﬁxed error level* δ*, however, they can produce very diﬀerent solutions. Since in practice we have to deal with a typically small, but ﬁxed* δ*, we would like to have a parameter choice rule that is sensitive to this value. To achieve this, we need to use more information than merely the error level* δ* to choose the parameter* α* and we will obtain this information from the approximate data* f**δ*. The basic idea is as follows. Let* f** ∈D*(*A*^(*†*)) and* f**δ** ∈Y* such that* ∥**f** −**f**δ**∥*⩽*δ* and consider the* residual* between* f**δ* and* u**α* :=* R**α**f**δ*, i.e.

*∥**Au**α** −**f**δ**∥**.*

Let* u*^(*†*)^( )be the minimal norm solution and deﬁne

*µ* := inf*{∥**Au** −**f**∥**, u** ∈X}* =* ∥**Au*^(*†*)^(* *)*−**f**∥**.*

We observe that* u*^(*†*)^( )satisﬁes the following inequality

*∥**Au*^(*†*)^(* *)*−**f**δ**∥*⩽*∥**Au*^(*†*)^(* *)*−**f**∥*+* ∥**f**δ** −**f**∥*⩽*µ* +* δ*

and in some cases this estimate may be sharp. Hence, it appears not to be useful to choose *α*(*δ, f**δ*) with* ∥**Au**α** −**f**δ**∥**< µ* +* δ*. In general, it may be not straightforward to estimate *µ*, but if* R*(*A*) is dense in* Y*, we get that* R*(*A*)^(*⊥*)=* {*0*}* due to Remark 2.0.2 and* µ* = 0. Therefore, we ideally ensure that* R*(*A*) is dense. These observations motivate the Morozov’s discrepancy principle, which in the case *µ* = 0 reads as follows.

**Deﬁnition 3.2.5** (Morozov’s discrepancy principle)**.*** Let** u**α* =* R**α**f**δ** with** α*(*δ, f**δ*)* chosen as follows*

*α*(*δ, f**δ*) = sup*{**α >* 0* | ∥**Au**α** −**f**δ**∥*⩽*ηδ**} *(3.4)

*for given** δ**,** f**δ** and a ﬁxed constant** η >* 1*. Then** u**α*(*δ,f**δ*) =* R**α*(*δ,f**δ*)*f**δ** is said to satisfy Morozov’s discrepancy principle.*

It can be shown that the a-posteriori parameter choice rule (3.4) indeed yields a con- vergent regularization method [20, Chapter 4.3].

**3.2.3 Heuristic parameter choice rules**

As the measurement error* δ* is not always easy to obtain in practice, it is tempting to use a parameter choice rule that only depends on the measured data* f**δ* and not on their error* δ*, i.e. to use a heuristic parameter choice rule. Unfortunately, heuristic rules yield convergent regularisations only for well-posed problems, as the following result, known as the Bakushinskii veto [7], demonstrates.

**Theorem 3.2.6** ([20, Thm 3.3])**.*** Let** A** ∈L*(*X**,** Y*)* and** {**R**α**}** be a regularization for** A*^(*†*)*. Let** α* =* α*(*f**δ*)* be a parameter choice rule such that* (*R**α**, α*)* is a convergent regularization. Then** A*^(*†*)^(* *)*is continuous from** Y** to** X**.*


---

## Page 32

32 *CHAPTER 3. CLASSICAL REGULARISATION THEORY*

### **3.3 Spectral Regularisation**

Recall the spectral representation (2.8) of the Moore-Penrose inverse* A*^(*†*)

*A*^(*†*)*f* =

*∞ *X

*j*=1

1 *σ**j **⟨**f, y**j**⟩**x**j* ,

where* {*(*σ**j**, x**j**, y**j*)*}* is the singular system of* A*. The source of ill-posedness of* A*^(*†*)^( )are the eigenvalues 1*/σ**j*, which explode as* j** →∞*, since* σ**j** →*0 as* j** →∞*. Let us construct a regularisation by modifying these eigenvalues as follows

*R**α**f* :=

*∞ *X

*j*=1 *g**α*(*σ**j*)* ⟨**f, y**j**⟩**x**j** , f** ∈Y**, *(3.5)

with an appropriate function* g**α* : R+* →*R+ such that* g**α*(*σ*)* →*^(1)

*σ* ^(as)^(* α*)^(* →*)^(0 for all)^(* σ >*)^( 0 )and

*g**α*(*σ*) ⩽*C**α* for all* σ** ∈*R+. (3.6)

**Theorem 3.3.1.*** Let** g**α* : R+* →*R+* be a piecewise continuous function satisfying* (3.6)*, *lim*α**→*0* g**α*(*σ*) = ^(1)

*σ** *^(*and*)

sup *α,σ** *^(*σg*)^(*α*)^(()^(*σ*)^())^( ⩽)^(*γ *)(3.7)

*for some constant** γ >* 0*. If** R**α** is deﬁned as in* (3.5)*, we have*

*R**α**f** →**A*^(*†*)*f** as** α** →*0

*for all** f** ∈D*(*A*^(*†*))*.*

*Proof.* From the singular value decomposition of* A*^(*†*)^( )and the deﬁnition of* R**α* we obtain

*R**α**f** −**A*^(*†*)*f* =

*∞ *X

*j*=1

 *g**α*(*σ**j*)* −*^(1)

*σ**j*

 *⟨**f, y**j**⟩**Y** x**j* =

*∞ *X

*j*=1 (*σ**j**g**α*(*σ**j*)* −*1)* ⟨**u*^(*†*)*, x**j**⟩**X** x**j* .

Consider

*∥**R**α**f** −**A*^(*†*)*f**∥*^(2 )*X* ^(=)

*∞ *X

*j*=1 (*σ**j**g**α*(*σ**j*)* −*1)^(2)^( )*⟨**u*^(*†*)*, x**j**⟩**X * 2 *.*

From (3.7) we can conclude

(*σ**j**g**α*(*σ**j*)* −*1)^(2)^( )⩽(1 +* γ*^(2)) ,

whilst

*∞ *X

*j*=1 (1 +* γ*^(2)) *⟨**u**†**, x**j**⟩**X * 2 = (1 +* γ*^(2))*∥**u*^(*†*)*∥*^(2)^(* *)*<* +*∞**.*


---

## Page 33

*3.3. SPECTRAL REGULARISATION *33

Therefore, by the reverse Fatou lemma we get the following estimate

lim sup *α**→*0

*R**α**f** −**A**†**f * 2

*X* ^(= lim sup )*α**→*0

*∞ *X

*j*=1 (*σ**j**g**α*(*σ**j*)* −*1)^(2)^(  )*⟨**u*^(*†*)*, x**j**⟩**X *2

⩽

*∞ *X

*j*=1

 lim sup *α**→*0 *σ**j**g**α*(*σ**j*)* −*1 2 *⟨**u**†**, x**j**⟩**X * 2 = 0 ,

where the last equality is due to the pointwise convergence of* g**α*(*σ**j*) to 1*/σ**j*. Hence, we have *R**α**f** −**A**†**f * *X** *^(*→*)^(0 for)^(* α*)^(* →*)^(0 for all)^(* f*)^(* ∈D*)^(()^(*A*)^(*†*)^().)

**Theorem 3.3.2.*** Let the assumptions of Theorem 3.3.1 hold and let** α* =* α*(*δ*)* be an a- priori parameter choice rule. Then* (*R**α*(*δ*)*, α*(*δ*))* with** R**α** as deﬁned in* (3.5)* is a convergent regularisation method if*

lim *δ**→*0* *^(*δC*)^(*α*)^(()^(*δ*)^())^( = 0)^(*.*)

*Proof.* The result follows immediately from* ∥**R**α*(*δ*)*∥**L*(*X**,**Y*) ⩽*C**α*(*δ*) and Theorem 3.2.4.

**3.3.1 Truncated singular value decomposition**

As a ﬁrst example for a spectral regularisation of the form (3.5) we want to consider the so-called* truncated singular value decomposition*. The idea is to discard all singular values below a certain threshold* α*, which is achieved using the following function* g**α*

*g**α*(*σ*) =

( 1 *σ **σ* ⩾*α *0 *σ < α* ^(. )(3.8)

Note that for all* σ >* 0 we naturally obtain lim*α**→*0* g**α*(*σ*) = 1*/σ*. Condition (3.7) is obviously satisﬁed with* γ* = 1 and condition (3.6) with* C**α* = 1 *α*^(. Therefore, truncated SVD is a )convergent regularisation if

lim *δ**→*0 *δ α* ^(= 0)^(*. *)(3.9)

Equation (3.5) then reads as follows

*R**α**f* = X

*σ**j*⩾*α*

1 *σ**j **⟨**f, y**j**⟩**Y** x**j* , (3.10)

for all* f** ∈Y*. Note that the sum in (3.10) is always well-deﬁned (i.e. ﬁnite) for any* α >* 0 as zero is the only accumulation point of singular vectors of compact operators. Let* A** ∈K*(*X**,** Y*) with singular system* {*(*σ**j**, x**j**, y**j*)*}**j**∈*N, and choose for* δ >* 0 an index function* j*^(*∗*): R+* →*N with* j*^(*∗*)(*δ*)* →∞*for* δ** →*0 and lim*δ**→*0* δ/σ**j**∗*(*δ*) = 0. We can then choose* α*(*δ*) =* σ**j**∗*(*δ*) as an a-priori parameter choice rule to obtain a convergent regularisation. Note that in practice a larger* δ* implies that more and more singular values have to be cut oﬀin order to guarantee a stable recovery that successfully suppresses the data error. A disadvantage of this approach is that it requires the knowledge of the singular vectors of* A* (only ﬁnitely many, but the number can still be large).


---

## Page 34

34 *CHAPTER 3. CLASSICAL REGULARISATION THEORY*

**3.3.2 Tikhonov regularisation**

The main idea behind Tikhonov regularisation^(1)^( )is to consider the normal equations and shift the eigenvalues of* A*^(*∗*)*A* by a constant factor, which will be associated with the regularisation parameter* α*. This shift can be realised via the function

*g**α*(*σ*) = *σ σ*^(2)^( )+* α *(3.11)

and the corresponding Tikhonov regularisation (3.5) reads as follows

*R**α**f* =

*∞ *X

*j*=1

*σ**j **σ*^(2 )*j* ^(+)^(* α*)^(*⟨*)^(*f, y*)^(*j*)^(*⟩*)^(*Y*)^(* x*)^(*j*)^( . )(3.12)

Again, we immediately observe that for all* σ >* 0 we have lim*α**→*0* g**α*(*σ*) = 1*/σ*. Condi- tion (3.7) is satisﬁed with* γ* = 1. Since 0 ⩽(*σ** −*^(*√*)*α*)^(2)^( )=* σ*^(2)^(* *)*−*2*σ*^(*√*)*α* +* α*, we get that *σ*^(2)^( )+* α* ⩾2*σ*^(*√*)*α* and

*σ σ*^(2)^( )+* α* ^(⩽ )1 2^(*√*)*α*^(*.*)

This estimate implies that (3.6) holds with* C**α* = 1 2^(*√*)*α*^(. Therefore, Tikhonov regularisation )is a convergent regularisation if

lim *δ**→*0 *δ **√**α* = 0*. *(3.13)

The formula (3.12) suggests that we need all singular vectors of* A* in order to compute the regularisation. However, we note that* σ*^(2 )*j* ^(are the eigenvalues of)^(* A*)^(*∗*)^(*A*)^( and, hence,)^(* σ*)^(2 )*j* ^(+)^(* α *)are the eigenvectors of* A*^(*∗*)*A* +* αI* (where* I* is the identity operator). Applying this operator to the regularised solution* u**α* =* R**α**f*, we get

(*A*^(*∗*)*A* +* αI*)*u**α* =

*∞ *X

*j*=1 (*σ*^(2 )*j* ^(+)^(* α*)^())^(*⟨*)^(*u*)^(*α*)^(*, x*)^(*j*)^(*⟩*)^(*X*)* *^(*x*)^(*j*) ^(=)

*∞ *X

*j*=1 (*σ*^(2 )*j* ^(+)^(* α*)^() )*σ**j **σ*^(2 )*j* ^(+)^(* α*)^(*⟨*)^(*f, y*)^(*j*)^(*⟩*)^(*Y*)^(* x*)^(*j*)^( =)^(* A*)^(*∗*)^(*f.*)

Therefore, the regularised solution* u**α* can be computed without knowing the singular system of* A* by solving the following well-posed linear equation

(*A*^(*∗*)*A* +* αI*)*u**α* =* A*^(*∗*)*f. *(3.14)

**Remark 3.3.3.** Rewriting equation (3.14) as

*A*^(*∗*)(*Au**α** −**f*) +* αu**α* = 0*,*

we note that it looks like a condition for the minimum of some quadratic form. Indeed, it can be easily checked that (3.14) is the ﬁrst order optimality condition for the following optimisation problem

min *u**∈X *1 2^(*∥*)^(*Au*)^(* −*)^(*f*)^(*∥*)^(2)^( +)^(* α*)^(*∥*)^(*u*)^(*∥*)^(2)^(*. *)(3.15)

The condition (3.14) is necessary (and, by convexity, suﬃcient) for the minimum of the functional in (3.15). Therefore, the regularised solution* u**α* can also be computed by solv- ing (numerically) the variational problem (3.15). This is the starting point for modern variational regularisation methods, which we will consider in the next chapter.

1Named after the Russian mathematician Andrey Nikolayevich Tikhonov (30 October 1906 - 7 October 1993)


---

## Page 35

# **Chapter 4**
# **Variational Regularisation**

Recall the variation formulation of Tikhonov regularisation for some data* f**δ** ∈Y*

min *u**∈X** *^(*∥*)^(*Au*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2)^( +)^(* α*)^(*∥*)^(*u*)^(*∥*)^(2)^(*.*)

The ﬁrst term in this expression,* ∥**Au** −**f**δ**∥*^(2), penalises the misﬁt between the predictions of the operator* A* and the measured data* f**δ* and is called the* ﬁdelity function* or* ﬁdelity term*. The second term,* ∥**u**∥*^(2)^( )penalises some unwanted features of the solution (in this case, a large norm) and is called the* regularistaion term*. The regularisation parameter* α* in this context balances the inﬂuence of these two terms on the functional to be minimised. More generally, using the notation* J* (*u*) for the regulariser, we can formally write down the variational regularisation problem as follows

min *u**∈X *1 2^(*∥*)^(*Au*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2)^( +)^(* α*)^(*J*)^( ()^(*u*)^())^(*, *)(4.1)

(the ^(1)

2 ^(in front of the ﬁdelity term is there to simplify notation later). The regularisation )operator* R**α* is deﬁned as follows

*R**α**f**δ** ∈*arg min *u**∈X*

1 2^(*∥*)^(*Au*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2)^( +)^(* α*)^(*J*)^( ()^(*u*)^())^(*.*)

In general, the minimiser doesn’t have to unique, hence the inclusion and not equality. Other ﬁdelity terms (not just* ∥**Au** −**f**δ**∥*^(2)) are possible and useful in many situations. In this course, however, we will use the squared norm for the sake of simplicity. In this chapter, we will study the properties of (4.1) for diﬀerent choices of* J* , but before that we will recall some necessary theoretical concepts.

### **4.1 Background**

**4.1.1 Banach spaces and weak convergence**

Banach spaces are complete, normed vector spaces (as Hilbert spaces) but they may not have an inner product. For every Banach space* X*, we can deﬁne the space of linear and continuous functionals which is called the* dual space** X** *^(*∗*)of* X*, i.e.* X** *^(*∗*):=* L*(*X**,* R). Let *u** ∈X* and* p** ∈X** *^(*∗*), then we usually write the* dual product** ⟨**p, u**⟩*instead of* p*(*u*). Moreover,

35


---

## Page 36

36 *CHAPTER 4. VARIATIONAL REGULARISATION*

for any* A** ∈L*(*X**,** Y*) there exists a unique operator* A*^(*∗*):* Y*^(*∗*)*→X** *^(*∗*), called the* adjoint* of* A *such that for all* u** ∈X* and* p** ∈Y*^(*∗*)we have

*⟨**A*^(*∗*)*p, u**⟩*=* ⟨**p, Au**⟩**.*

It is easy to see that either side of the equation are well-deﬁned, e.g.* A*^(*∗*)*p** ∈X** *^(*∗*)and* u** ∈X*. The dual space of a Banach space* X* can be equipped with the following norm

*∥**p**∥**X** ∗*= sup *u**∈X**,**∥**u**∥**X* ⩽1 *⟨**p, u**⟩**.*

With this norm the dual space is itself a Banach space. Therefore, it has a dual space as well which we will call the bi-dual space of* X* and denote it with* X** *^(*∗∗*):= (*X** *^(*∗*))^(*∗*). As every *u** ∈X* deﬁnes a continuous and linear mapping on the dual space* X** *^(*∗*)by

*⟨**E*(*u*)*, p**⟩*:=* ⟨**p, u**⟩**,*

the mapping* E* :* X →X** *^(*∗∗*)is well-deﬁned. It can be shown that* E* is a linear and continuous isometry (and thus injective). In the special case when* E* is surjective, we call* X** reﬂexive*. Examples of reﬂexive Banach spaces include Hilbert spaces and* L*^(*q*)*, ℓ*^(*q*)^( )spaces with 1* < q <** ∞*. We call the space* X** separable* if there exists a set* X** *^(*′*)^(* *)*⊂X* of at most countable cardinality such that* X** *^(*′*)^( )=* X*. A problem in inﬁnite dimensional spaces is that bounded sequences may fail to have convergent subsequences. An example is for instance in* ℓ*^(2)^( )the sequence* {**u*^(*k*)*}**k**∈*N* ⊂**ℓ*^(2)*, u*^(*k *)*j* ^(= 1 )if* k* =* j* and 0 otherwise. It is easy to see that* ∥**u*^(*k*)*∥**ℓ*2 = 1 and that there is no* u** ∈**ℓ*^(2)^( )such that* u*^(*k*)^(* *)*→**u*. To circumvent this problem, we deﬁne a weaker topology on* X*. We say that *{**u*^(*k*)*}**k**∈*N* ⊂X** converges weakly* to* u** ∈X* if and only if for all* p** ∈X** *^(*∗*)the sequence of real numbers* { *
 *p, u*^(*k*)^( )*}**k**∈*N converges and

*⟨**p, u**j**⟩→⟨**p, u**⟩**.*

We will denote weak convergence by* u*^(*k*)^(* *)*⇀u*. On a dual space* X** *^(*∗*)we could deﬁne another topology (in addition to the strong topology induced by the norm and the weak topology as the dual space is a Banach space as well). We say a sequence* {**p*^(*k*)*}**k**∈*N* ⊂X** *^(*∗*)*converges weakly-**∗*to* p** ∈X** *^(*∗*)if and only if D *p*^(*k*)*, u *E *→⟨**p, u**⟩ *for all* u** ∈X*

and we denote weak-*∗*convergence by* p*^(*k*)^(* *)*⇀*^(*∗*)*p*. Similarly, for any topology* τ* on* X* we denote the convergence in that topology by* u*^(*k *)*τ**→**u*. With these two new notions of convergence, we can solve the problem of bounded se- quences:

**Theorem 4.1.1** (Banach-Alaoglu Theorem, e.g. [32, p. 70] or [36, p. 141])**.*** Let** X* = (*X** *^(*⋄*))^(*∗*)

*be the dual of a Banach space** X** *^(*⋄*)*. Then the unit ball** B**X* =* {**u** ∈X* :* ∥**x**∥*⩽1*}** is compact in the weak-**∗**topology. If** X** *^(*⋄*)*is separable, then the weak-**∗**topology is metrisable on bounded sets and every bounded sequence** {**u*^(*k*)*}**k**∈*N* ⊂X** has a weak-*^(*∗*)*convergent subsequence.*

**Theorem 4.1.2** ([38, p. 64])**.*** Each bounded sequence** {**u*^(*k*)*}**k**∈*N* in a reﬂexive separable Banach space** X** has a weakly convergent subsequence.*


---

## Page 37

*4.1. BACKGROUND *37

Figure 4.1: Visualisation of lower semi-continuity. The solid dot at a jump indicates the value that the function takes. The function on the left is continuous and thus lower semi- continuous. The functions in the middle and on the right are discontinuous. While the function in the middle is lower semi-continuous, the function on the right is not (due to the limit from the left at the discontinuity).

An important property of functionals, which we will need later, is sequential lower semicontinuity. Roughly speaking this means that the functional values for arguments near an argument* u* are either close to* E*(*u*) or greater than* E*(*u*).

**Deﬁnition 4.1.3.*** Let** X** be a Banach space with topology** τ**X** . The functional** E* :* X →*^(¯)R *is said to be* sequentially lower semi-continuous with respect to* τ**X** (**τ**X** -l.s.c.) at** u** ∈X** if*

*E*(*u*) ⩽lim inf *j**→∞*^(*E*)^(()^(*u*)^(*j*)^())

*for all sequences** {**u**j**}**j**∈*N* ⊂X** with** u**j** →**u** in the topology** τ**X** of** X**.*

**Remark 4.1.4.** For topologies that are not induced by a metric we have to diﬀer between a topological property and its sequential version, e.g. continuous and sequentially continuous. If the topology is induced by a metric, then these two are the same. However, for instance the weak and weak-*∗*topology are generally not induced by a metric (but this is true on bounded sets).

**Example 4.1.5.** The functional* ∥· ∥*1 :* ℓ*^(2)^(* *)*→*^(¯)R with

*∥**u**∥*1 =

(P*∞ **j*=1* *^(*|*)^(*u*)^(*j*)^(*| *)if* u** ∈**ℓ*^(1)

*∞ *else

is weakly (and, hence, strongly) lower semi-continuous in* ℓ*^(2).

*Proof.* Let* {**u*^(*j*)*}**j**∈*N* ⊂**ℓ*^(2)^( )be a weakly convergent sequence with* u*^(*j*)^(* *)*⇀u** ∈**ℓ*^(2). We have with *δ**k* :* ℓ*^(2)^(* *)*→*R*,** ⟨**δ**k**, v**⟩*=* v**k* that for all* k** ∈*N

*u*^(*j *)*k* ^(=)^(* ⟨*)^(*δ*)^(*k*)^(*, u*)^(*j*)^(*⟩→⟨*)^(*δ*)^(*k*)^(*, u*)^(*⟩*)^(=)^(* u*)^(*k*)^(* .*)

The assertion follows then with Fatou’s lemma

*∥**u**∥*1 =

*∞ *X

*k*=1 *|**u**k**|* =

*∞ *X

*k*=1 lim *j**→∞*^(*|*)^(*u*)^(*j *)*k*^(*|*)^( ⩽)^(lim inf )*j**→∞*

*∞ *X

*k*=1 *|**u*^(*j *)*k*^(*|*)^( = lim inf )*j**→∞*^(*∥*)^(*u*)^(*j*)^(*∥*)^(1)^(* .*)

Note that it is not clear whether both the left and the right hand side are ﬁnite.


---

## Page 38

38 *CHAPTER 4. VARIATIONAL REGULARISATION*

**4.1.2 Convex analysis**

**Inﬁnity calculus**

We will look at functionals* E* :* X →*^(¯)R whose range is modelled to be the* extended real line *¯R := R* ∪{−∞**,* +*∞}* where the symbol +*∞*denotes an element that is not part of the real line that is by deﬁnition larger than any other element of the reals, i.e.

*x <* +*∞*

for all* x** ∈*R (similarly,* x >** −∞*for all* x** ∈*R). This is useful to model constraints: for instance, if we were trying to minimise* E* : [*−*1*,** ∞*)* →*R*, x** 7→**x*^(2)^( )we could remodel this minimisation problem by ^(e)*E* : R* →*^(¯)R

e*E*(*x*) =

( *x*^(2 )if* x* ⩾*−*1 *∞ *else *.*

Obviously both functionals have the same minimiser but ^(e)*E* is deﬁned on a vector space and not only on a subset. This has two important consequences: on the on hand, it makes many theoretical arguments easier as we do not need to worry whether* E*(*x* +* y*) is deﬁned or not. On the other hand, it makes practical implementations easier as we are dealing with unconstrained optimisation instead of constrained optimisation. This comes at a cost that some algorithms are not applicable any more, e.g. the function ^(e)*E* is not diﬀerentiable everywhere whereas* E* is (in the interior of its domain). It is useful to note that one can calculate on the extended real line ^(¯)R as we are used to on the real line R but the operations with* ±∞*need yet to be deﬁned.

**Deﬁnition 4.1.6.*** The extended real line is deﬁned as* ^(¯)R := R* ∪{−∞**,* +*∞}** with the following rules that hold for any** x** ∈*R* and** λ >* 0*:*

*x** ± ∞*:=* ±∞*+* x* :=* ±∞*

*λ** ·* (*±∞*) :=* ±∞·** λ* :=* ±∞**, **−*1* ·* (*±∞*) :=* ∓∞*

*x/*(*±∞*) := 0

*∞*+* ∞*:=* ∞**, **−∞−∞*:=* −∞**.*

Some calculations are* not deﬁned*, e.g.,

+*∞−∞*and (*±∞*)* ·* (*±∞*)* .*

Using functions with values on the extended real line, one can easily describe sets* C ⊂X*.

**Deﬁnition 4.1.7** (Characteristic function)**.*** Let** C ⊂X** be a set. The function** χ**C* :* X →*^(¯)R*,*

*χ**C*(*u*) =

( 0 *u** ∈C ∞ **u** ∈X \ C*

*is called the characteristic function of the set** C**.*

Using characteristic functions, one can easily write constrained optimisation problems as unconstrained ones: min *u**∈C** *^(*E*)^(()^(*u*)^() )*⇔ *min *u**∈X** *^(*E*)^(()^(*u*)^() +)^(* χ*)^(*C*)^(()^(*u*)^())^(*.*)


---

## Page 39

*4.1. BACKGROUND *39

**Deﬁnition 4.1.8.*** Let** X** be a vector space and** E* :* X →*^(¯)R* a functional. Then the* eﬀective domain* of** E** is*

dom(*E*) :=* {**u** ∈X |** E*(*u*)* <** ∞}** .*

**Deﬁnition 4.1.9.*** A functional** E** is called* proper* if the eﬀective domain* dom(*E*)* is not empty.*

**Convexity**

A property of fundamental importance of sets and functions is convexity.

**Deﬁnition 4.1.10.*** Let** X** be a vector space. A subset** C ⊂X** is called* convex*, if** λu* + (1* − **λ*)*v** ∈C** for all** λ** ∈*(0*,* 1)* and all** u, v** ∈C**.*

Figure 4.2: Example of a convex set (left) and non-convex set (right).

**Deﬁnition 4.1.11.*** A functional** E* :* X →*^(¯)R* is called* convex*, if*

*E*(*λu* + (1* −**λ*)*v*) ⩽*λE*(*u*) + (1* −**λ*)*E*(*v*)

*for all** λ** ∈*(0*,* 1)* and all** u, v** ∈*dom(*E*)* with** u** ̸*=* v**. It is called* strictly convex* if the inequality is strict. It is called* strongly* convex with constant** θ** if** E*(*u*)* −**θ**∥**u**∥*^(2)^(* *)*is convex.*

Obviously, strong convexity implies strict convexity and strict convexity implies convex- ity.

**Example 4.1.12.** The absolute value function R* →*R*, x** 7→|**x**|* is convex but not strictly convex. The quadratic function* x** 7→**x*^(2)^( )is strongly (and hence strictly) convex. The function *x** 7→**x*^(4)^( )is strictly convex, but not strongly convex. For other examples, see Figure 4.3.

*∞*

*∅*

Figure 4.3: Example of a convex function (left), a strictly convex function (middle) and a non-convex function (right).


---

## Page 40

40 *CHAPTER 4. VARIATIONAL REGULARISATION*

**Example 4.1.13.** The characteristic function* χ**C*(*u*) is convex if and only if* C* is a convex set. To see the convexity, let* u, v** ∈*dom(*χ**C*) =* C*. Then by the convexity of* C* the convex combination* λu* + (1* −**λ*)*v* is as well in* C* and both the left and the right hand side of the desired inequality are zero.

**Lemma 4.1.14.*** Let** α* ⩾0* and** E, F* :* X →*^(¯)R* be two convex functionals. Then** E* + *αF* :* X →*^(¯)R* is convex. Furthermore, if** α >* 0* and** F** strictly convex, then** E* +*αF** is strictly convex.*

**Fenchel conjugate**

In convex optimisation problems (i.e. those involving convex functions) the concept of *Fenchel conjugates* plays a very important role.

**Deﬁnition 4.1.15.*** Let** E* :* X →*^(¯)R* be a functional. The functional** E*^(*∗*):* X** *^(*∗*)*→*^(¯)R*,*

*E*^(*∗*)(*p*) = sup *u**∈X *[*⟨**p, u**⟩−**E*(*u*)]*,*

*is called the Fenchel conjugate of** E**.*

**Theorem 4.1.16** ([19, Prop. 4.1])**.*** For any functional** E* :* X →*^(¯)R* the following inequality holds: **E*^(*∗∗*):= (*E*^(*∗*))^(*∗*)⩽*E.*

*If** E** is proper, lower-semicontinuous (see Def. 4.1.3) and convex, then*

*E*^(*∗∗*)=* E.*

**Subgradients**

For convex functions one can generalise the concept of a derivative so that it would also make sense for non-diﬀerentiable functions.

**Deﬁnition 4.1.17.*** A functional** E* :* X →*^(¯)R* is called* subdiﬀerentiable* at** u** ∈X**, if there exists an element** p** ∈X** *^(*∗*)*such that*

*E*(*v*) ⩾*E*(*u*) +* ⟨**p, v** −**u**⟩*

*holds, for all** v** ∈X**. Furthermore, we call** p** a* subgradient* at position** u**. The collection of all subgradients at position** u**, i.e.*

*∂E*(*u*) :=* {**p** ∈X** *^(*∗*)*|** E*(*v*) ⩾*E*(*u*) +* ⟨**p, v** −**u**⟩**,** ∀**v** ∈X}** ,*

*is called* subdiﬀerential* of** E** at** u**.*

It is clear that if a convex functional* E* :* X →*^(¯)R is proper, i.e. dom(*E*)* ̸*=* ∅*, then for all *u** ̸∈*dom(*E*) the subdiﬀerential is empty. A suﬃcient (but not necessary) condition for* E *to have a subgradient at* u** ∈*dom(*E*) is given by

**Proposition 4.1.18** ([19, Prop. 5.2])**.*** Let** E* :* X →*^(¯)R* be a convex functional and** u** ∈ *dom(*E*)* such that** E** is continuous at** u**. Then** ∂E*(*u*)* ̸*=* ∅**.*

**Theorem 4.1.19** ([4, Thm. 7.13])**.*** Let** E* :* X →*^(¯)R* be a proper convex function and** u** ∈ *dom(*E*)*. Then** ∂E*(*u*)* is a weak-**∗**compact convex subset of** X** *^(*∗*)*.*


---

## Page 41

*4.1. BACKGROUND *41

Figure 4.4: Visualisation of the subdiﬀerential. Linear approximations of the functional have to lie completely underneath the function. For points where the function is not diﬀerentiable there may be more than one such approximation.

For diﬀerentiable functions the subdiﬀerential consists of just one element – the deriva- tive. For non-diﬀerentiable functionals the subdiﬀerential is multivalued; we want to con- sider the subdiﬀerential of the absolute value function as an illustrative example.

**Example 4.1.20.** Let* E* : R* →*R be the absolute value function* E*(*u*) =* |**u**|*. Then, the subdiﬀerential of* E* at* u* is given by

*∂E*(*u*) =

  

 

*{*1*} *for* u >* 0 [*−*1*,* 1] for* u* = 0 *{−*1*} *for* u <* 0

*,*

which you will prove as an exercise. A visual explanation is given in Figure 4.4.

The subdiﬀerential of a sum of two functions can be characterised as follows.

**Theorem 4.1.21** ([19, Prop. 5.6])**.*** Let** E* :* X →*^(¯)R* and** F* :* X →*^(¯)R* be proper l.s.c. convex functions and suppose** ∃**u** ∈*dom(*E*)* ∩*dom(*F*)* such that** E** is continuous at** u**. Then*

*∂*(*E* +* F*) =* ∂E* +* ∂F.*

Using the subdiﬀerential, one can characterise minimisers of convex functionals.

**Theorem 4.1.22.*** An element** u** ∈X** is a minimiser of the functional** E* :* X →*^(¯)R* if and only if* 0* ∈**∂E*(*u*)*.*

*Proof.* By deﬁnition, 0* ∈**∂E*(*u*) if and only if for all* v** ∈X* it holds

*E*(*v*) ⩾*E*(*u*) +* ⟨*0*, v** −**u**⟩*=* E*(*u*)* ,*

which is by deﬁnition the case if and only if* u* is a minimiser of* E*.

The next result connects subgradients and convex conjugates

**Theorem 4.1.23** ([19, Prop. 5.1])**.*** Let** E* :* X →*^(¯)R* be a convex function and** E*^(*∗*):* X** *^(*∗*)*→*^(¯)R *its convex conjugate. Then** p** ∈**∂E*(*u*)* if and only if*

*E*(*u*) +* E*^(*∗*)(*p*) =* ⟨**p, u**⟩**.*

*Proof.* Left as an exercise.


---

## Page 42

42 *CHAPTER 4. VARIATIONAL REGULARISATION*

**Bregman distances**

Convex functions naturally deﬁne some distance measure that became known as the Breg- man distance.

**Deﬁnition 4.1.24.*** Let** E* :* X →*^(¯)R* be a convex functional. Moreover, let** u, v** ∈X**, E*(*v*)* < **∞**and** q** ∈**∂E*(*v*)*. Then the* (generalised) Bregman distance* of** E** between** u** and** v** is deﬁned as*

*D*^(*q *)*E*^(()^(*u, v*)^() :=)^(* E*)^(()^(*u*)^())^(* −*)^(*E*)^(()^(*v*)^())^(* −⟨*)^(*q, u*)^(* −*)^(*v*)^(*⟩*)^(*. *)(4.2)

*v u*

*D*^(*p *)*E*^(()^(*u, v*)^() )*E*(*u*)

*E*

*E*(*v*) +* ⟨**p, u** −**v**⟩*

Figure 4.5: Visualization of the Bregman distance.

**Remark 4.1.25.** It is easy to check that a Bregman distance somewhat resembles a metric as for all* u, v** ∈X**, q** ∈**∂E*(*v*) we have that* D*^(*q *)*E*^(()^(*u, v*)^())^( ⩾)^(0 and)^(* D*)^(*q *)*E*^(()^(*v, v*)^() = 0. There are )functionals where the Bregman distance (up to a square root) is actually a metric; e.g. *E*(*u*) := ^(1)

2^(*∥*)^(*u*)^(*∥*)^(2 )*X* ^(for Hilbert space)^(* X*)^(, then)^(* D*)^(*q *)*E*^(()^(*u, v*)^() =)^( 1)

2^(*∥*)^(*u*)^(* −*)^(*v*)^(*∥*)^(2 )*X* ^(. However, in general, )Bregman distances are not symmetric and* D*^(*q *)*E*^(()^(*u, v*)^() = 0 does not imply)^(* u*)^( =)^(* v*)^(, as you will )see on the example sheets.

To overcome the issue of non-symmetry, one can introduce the so-called* symmetric Bregman distance*.

**Deﬁnition 4.1.26.*** Let** E* :* X →*^(¯)R* be a convex functional. Moreover, let** u, v** ∈X**, E*(*u*)* < **∞**, E*(*v*)* <** ∞**,** q** ∈**∂E*(*v*)* and** p** ∈**∂E*(*u*)*. Then the* symmetric Bregman distance* of** E **between** u** and** v** is deﬁned as*

*D*^(*symm *)*E *(*u, v*) :=* D*^(*q *)*E*^(()^(*u, v*)^() +)^(* D*)^(*p *)*E*^(()^(*v, u*)^() =)^(* ⟨*)^(*p*)^(* −*)^(*q, u*)^(* −*)^(*v*)^(*⟩*)^(*. *)(4.3)

**Absolutely one-homogeneous functionals**

**Deﬁnition 4.1.27.*** A functional** E* :* X →*^(¯)R* is called absolutely one-homogeneous if*

*E*(*λu*) =* |**λ**|**E*(*u*) *∀**λ** ∈*R*,** ∀**u** ∈X**.*

Absolutely one-homogeneous convex functionals have some useful properties, for exam- ple, it is obvious that* E*(0) = 0. Some further properties are listed below.

**Proposition 4.1.28.*** Let** E*(*·*)* be a convex absolutely one-homogeneous functional and let **p** ∈**∂E*(*u*)*. Then the following equality holds:*

*E*(*u*) =* ⟨**p, u**⟩**.*


---

## Page 43

*4.1. BACKGROUND *43

*Proof.* Left as exercise.

**Remark 4.1.29.** The Bregman distance* D*^(*p *)*E*^(()^(*v, u*)^() in this case can be written as follows:)

*D*^(*p *)*E*^(()^(*v, u*)^() =)^(* E*)^(()^(*v*)^())^(* −⟨*)^(*p, v*)^(*⟩*)^(*.*)

**Proposition 4.1.30.*** Let** E*(*·*)* be a proper, convex, l.s.c. and absolutely one-homogeneous functional. Then the Fenchel conjugate** E*^(*∗*)(*·*)* is the characteristic function of the convex set** ∂E*(0)*.*

*Proof.* Left as exercise.

An obvious consequence of the above results is the following

**Proposition 4.1.31.*** For any** u** ∈X**,** p** ∈**∂E*(*u*)* if and only if** p** ∈**∂E*(0)* and** E*(*u*) = (*p, u*)*.*

**4.1.3 Minimisers**

**Deﬁnition 4.1.32.*** Let** E* :* X →*^(¯)R* be a functional. We say that** u*^(*∗*)*∈X** solves the min- imisation problem*

min *u**∈X** *^(*E*)^(()^(*u*)^())

*if and only if** E*(*u*^(*∗*))* <** ∞**and** E*(*u*^(*∗*)) ⩽*E*(*u*)*, for all** u** ∈X**. We call** u*^(*∗*)*a* minimiser* of** E**.*

**Deﬁnition 4.1.33.*** A functional** E* :* X →*^(¯)R* is called* bounded from below* if there exists a constant** C >** −∞**such that for all** u** ∈X** we have** E*(*u*) ⩾*C**.*

This condition is obviously necessary for the ﬁniteness of the inﬁmum inf*u**∈X** E*(*u*).

**Existence**

If all minimising sequences (that converge to the inﬁmum assuming it exists) are unbounded, then there cannot exist a minimiser. A suﬃcient condition to avoid such a scenario is *coercivity*.

**Deﬁnition 4.1.34.*** A functional** E* :* X →*^(¯)R* is called* coercive*, if for all** {**u**j**}**j**∈*N* with **∥**u**j**∥**X** →∞**we have** E*(*u**j*)* →∞**.*

*x*^(2)

*x*

exp(*x*)

*x*

Figure 4.6: While the coercive function on the left has a minimiser, it is easy to see that the non-coercive function on the right does not have a minimiser.

**Remark 4.1.35.** Coercivity is equivalent to its negated statement which is “if the function values* {**E*(*u**j*)*}**j**∈*N* ⊂*R are bounded, so is the sequence* {**u**j**}**j**∈*N* ⊂X*”.


---

## Page 44

44 *CHAPTER 4. VARIATIONAL REGULARISATION*

Although coercivity is not strictly speaking necessary, it is suﬃcient that all minimising sequences are bounded.

**Lemma 4.1.36.*** Let** E* :* X →*^(¯)R* be a proper, coercive functional and bounded from below. Then the inﬁmum* inf*u**∈X** E*(*u*)* exists in* R*, there are minimising sequences, i.e.** {**u**j**}**j**∈*N* ⊂ X** with** E*(*u**j*)* →*inf*u**∈X** E*(*u*)*, and all minimising sequences are bounded.*

*Proof.* As* E* is proper and bounded from below, there exists a* C*1* >* 0 such that we have *−∞**<** −**C*1* <* inf*u** E*(*u*)* <** ∞*which also guarantees the existence of a minimising sequence. Let* {**u**j**}**j**∈*N be any minimising sequence, i.e. *E*(*u**j*)* →*inf*u** E*(*u*). Then there exists a *j*0* ∈*N such that for all* j > j*0 we have

*E*(*u**j*) ⩽inf *u** *^(*E*)^(()^(*u*)^() + 1 )| {z } =:*C*2

*<** ∞**.*

With* C* := max*{**C*1*, C*2*}* we have that* |**E*(*u**j*)*|** < C* for all* j > j*0 and thus from the coercivity it follows that* {**u**j**}**j>j*0 is bounded, see Remark 4.1.35. Including a ﬁnite number of elements does not change its boundedness which proves the assertion.

A positive answer about the existence of minimisers is given by the following Theorem known as the “direct method” or “fundamental theorem of optimisation”.

**Theorem 4.1.37** (“Direct method”, David Hilbert, around 1900)**.*** Let** X** be a Banach space and** τ**X** a topology (not necessarily the one induced by the norm) on** X** such that bounded sequences have** τ**X** -convergent subsequences. Let** E* :* X →*^(¯)R* be proper, bounded from below, coercive and** τ**X** -l.s.c. Then** E** has a minimiser.*

*Proof.* From Lemma 4.1.36 we know that inf*u**∈X** E*(*u*) is ﬁnite, minimising sequences exist and that they are bounded. Let* {**u**j**}**j**∈*N* ∈X* be a minimising sequence. Thus, from the assumption on the topology* τ**X* there exists a subsequence* {**u**j**k**}**k**∈*N and* u*^(*∗*)*∈X* with *u**j**k **τ**X **→**u*^(*∗*)for* k** →∞*. From the sequential lower semi-continuity of* E* we obtain

*E*(*u*^(*∗*)) ⩽lim inf *k**→∞*^(*E*)^(()^(*u*)^(*j*)^(*k*)^() = lim )*j**→∞*^(*E*)^(()^(*u*)^(*j*)^() = inf )*u**∈X** *^(*E*)^(()^(*u*)^())^(* <*)^(* ∞*)^(*,*)

which shows that* E*(*u*^(*∗*))* <** ∞*and* E*(*u*^(*∗*)) ⩽*E*(*u*) for all* u** ∈X*; thus* u*^(*∗*)minimises* E*.

The above theorem is very general but its conditions are hard to verify but the situation is a easier in* reﬂexive* Banach spaces (thus also in Hilbert spaces).

**Corollary 4.1.38.** Let* X* be a reﬂexive Banach space and* E* :* X →*^(¯)R be a functional which is proper, bounded from below, coercive and l.s.c. with respect to the weak topology. Then there exists a minimiser of* E*.

*Proof.* The statement follows from the direct method, Theorem 4.1.37, as in reﬂexive Banach spaces bounded sequences have weakly convergent subsequences, see Theorem 4.1.2.

**Remark 4.1.39.** For convex functionals, the situation is even easier. It can be shown that a convex function is l.s.c. with respect to the weak topology if and only if it is l.s.c. with respect to the strong topology (see e.g. [19, Corollary 2.2., p. 11] or [8, p. 149] for Hilbert spaces).


---

## Page 45

*4.1. BACKGROUND *45

**Remark 4.1.40.** It is easy to see that the key ingredient for the existence of minimisers is that bounded sequences have a convergent subsequence. In variational regularisation this is usually ensured by an appropriate choice of the regularisation functional.

**Uniqueness**

**Theorem 4.1.41.*** Assume that the functional** E* :* X →*^(¯)R* has at least one minimiser and is strictly convex. Then the minimiser is unique.*

*Proof.* Let* u, v* be two minimisers of* E* and assume that they are diﬀerent, i.e.* u** ̸*=* v*. Then it follows from the minimising properties of* u* and* v* as well as the strict convexity of* E* that

*E*(*u*) ⩽*E*(^(1)

2^(*u*)^( +)^( 1)

2^(*v*)^())^(* <*)^( 1)

2^(*E*)^(()^(*u*)^() + 1)

2* *^(*E*)^(()^(*v*)^() )|{z} ⩽*E*(*u*)

⩽*E*(*u*)

which is a contradiction. Thus,* u* =* v* and the assertion is proven.

**Example 4.1.42.** Convex (but not strictly convex) functions may have have more than one minimiser, examples include constant and trapezoidal functions, see Figure 4.7. On the other hand, convex (and even non-convex) functions may have a unique minimiser, see Figure 4.7.

a) b)

Figure 4.7: a) Convex functions may not have a unique minimiser. b) Neither strict con- vexity nor convexity is necessary for the uniqueness of a minimiser.

**4.1.4 Duality in convex optimisation**

Consider the following optimisation problem

inf *u**∈X** *^(*E*)^(()^(*Au*)^() +)^(* F*)^(()^(*u*)^())^(*, *)(*P*)

where* E* :* Y →*^(¯)R and* F* :* X →*^(¯)R are proper, convex and lower semicontinuous functions and* A** ∈L*(*X**,** Y*) is a linear bounded operator. Since* E* is convex and lower semicontinuous, it can be written as the convex conjugate of its conjugate* E*^(*∗*)

*E*(*y*) = sup *η**∈Y*^(*∗*)^(*⟨*)^(*η, y*)^(*⟩−*)^(*E*)^(*∗*)^(()^(*η*)^() )*y** ∈Y**.*

Hence, we can rewrite (*P*) as follows

inf *u**∈X* ^(sup )*η**∈Y*^(*∗*)^(*⟨*)^(*η, Au*)^(*⟩−*)^(*E*)^(*∗*)^(()^(*η*)^() +)^(* F*)^(()^(*u*)^())^(*. *)(*S*)


---

## Page 46

46 *CHAPTER 4. VARIATIONAL REGULARISATION*

This problem is referred to as the* saddle point problem*, whereas (*P*) is referred to as the *primal problem*. Since inf sup ⩾sup inf always holds, we get that

inf *u**∈X** *^(*E*)^(()^(*Au*)^() +)^(* F*)^(()^(*u*)^() )⩾ sup *η**∈Y*^(*∗*)^(inf )*u**∈X*^(*⟨*)^(*η, Au*)^(*⟩−*)^(*E*)^(*∗*)^(()^(*η*)^() +)^(* F*)^(()^(*u*)^())

= sup *η**∈Y*^(*∗*)^(inf )*u**∈X*^(*⟨*)^(*A*)^(*∗*)^(*η, u*)^(*⟩−*)^(*E*)^(*∗*)^(()^(*η*)^() +)^(* F*)^(()^(*u*)^())

= sup *η**∈Y*^(*∗*)

 *−**E*^(*∗*)(*η*)* −*sup *u**∈X *[*⟨−**A*^(*∗*)*η, u**⟩−**F*(*u*)] 

= sup *η**∈Y*^(*∗*)^(*−*)^(*E*)^(*∗*)^(()^(*η*)^())^(* −*)^(*F*)^(* ∗*)^(()^(*−*)^(*A*)^(*∗*)^(*η*)^())^(*.*)

The last problem sup *η**∈Y*^(*∗*)^(*−*)^(*E*)^(*∗*)^(()^(*η*)^())^(* −*)^(*F*)^(* ∗*)^(()^(*−*)^(*A*)^(*∗*)^(*η*)^() )(*D*)

is called the* dual problem*. The fact that the optimal value of the primal is always less or equal to the optimal value of the dual problem is referred to as* weak duality* and the diﬀerence between these two optimal values is referred to as the* duality gap*. Whenever the two optimal values are in fact equal, one speaks of* strong duality*. Suﬃcient conditions for strong duality are given by

**Theorem 4.1.43** ([19, Ch.III Thm 4.1 and Rem. 4.2])**.*** Suppose that*

*(i) the function** E*(*Au*) +* F*(*u*):* X →*^(¯)R* is proper, convex, l.s.c. and coercive;*

*(ii)** ∃**u*0* ∈X** s.t.** F*(*u*0)* <* +*∞**,** E*(*Au*0)* <* +*∞**and** E*(*y*)* is continuous at** y* =* Au*0*.*

*Then*

*(i) The dual problem* (*D*)* has at least one solution* b*η**;*

*(ii) There is no duality gap between* (*P*)* and* (*D*)*, i.e. strong duality holds;*

*(iii) If* (*P*)* has an optimal solution* b*u**, then the following optimality conditions hold*

*−**A*^(*∗*)b*η** ∈**∂F*(b*u*)*, *b*η** ∈**∂E*(*A*b*u*)*.*

Note that existence of a primal solution is* not* guaranteed by this theorem.

### **4.2 Well-posedness and Regularisation Properties**

Our goal is to study the properties of optimisation problem (4.1) as a convergent regulari- sation for the ill-posed problem *Au* =* f, *(4.4)

where* A*:* X →Y* is a linear bounded operator,* Y* is a Banach space and* X* is the dual of a separable Banach space. In particular, we will ask questions of existence of minimisers (well-posedness of the regularised problem) and parameter choice rules that guarantee the convergence of the minimisers to an appropriate generalised solution of (4.4) for diﬀerent choices of the regularisation functional. To this end, we need to extend the deﬁnition of a minimal-norm solution (Def. 2.1.1) to an arbitrary regularisation term.


---

## Page 47

*4.2. WELL-POSEDNESS AND REGULARISATION PROPERTIES *47

**Deﬁnition 4.2.1** (*J* -minimising solutions)**.*** Let** u*^(*† *)*J** *^(*be a least squares solution, i.e.*)

*∥**Au*^(*† *)*J** *^(*−*)^(*f*)^(*∥*)^(*Y*)^( = inf)^(*{∥*)^(*Av*)^(* −*)^(*f*)^(*∥*)^(*Y*)^(*, *)*v** ∈X}*

*and **J* (*u*^(*† *)*J* ^())^( ⩽)^(*J*)^( ()^(e)^(*u*)^() )*for all least squares solutions* e*u.*

*Then** u*^(*† *)*J** *^(*is called a*)^(* J*)^(* -minimising solution of*)^( (4.4))^(*.*)

We will assume that there exists a least-squares solution with a ﬁnite value of* J* , i.e. there exists at least one element* u* such that* ∥**Au** −**f**∥**Y* = inf*{∥**Av** −**f**∥**Y**, v** ∈X}* and *J* (*u*)* <* +*∞*.

**Remark 4.2.2.** A* J* -minimising solution may not exist and if it does, it may be non-unique. We will later see conditions, under which a* J* -minimising solution exists. Non-uniqueness, however, is common with popular choices of* J* . In this case we need to deﬁne a* selection operator* that will select a single element from all the* J* -minimising solutions (see [9]). We will not explicitly mention this, stating all results for just* a** J* -minimising solution.

We will need the following

**Lemma 4.2.3.*** Let** J* (*u*) = ^(P)^(*n *)*i*=1* *^(*J*)^(*i*)^(()^(*u*)^())^(*, where each*)^(* J*)^(*i*)^(()^(*u*)^())^(* is convex and*)^(* p*)^(*i*)^(*-homogeneous *)*(**p**i** >* 0*), that is, **J**i*(*λu*) =* |**λ**|*^(*p*)^(*i*)*J**i*(*u*) *∀**u** ∈X**, λ** ∈*R*.*

*The the set **N*(*J* ) :=* {**u** ∈X* :* J* (*u*) = 0*}*

*is a linear subspace of** X**.*

*Proof.* First of all, we note that* J**i*(*u*) ⩾0 for all* u** ∈X*. Indeed, we have

0 =* J**i*(0) =* J**i*

1

2^(*u*)^(* −*)^(1)

2^(*u *) ⩽^(1)

2^(*J*)^(*i*)^(()^(*u*)^() + 1)

2^(*J*)^(*i*)^(()^(*−*)^(*u*)^() =)^(* J*)^(*i*)^(()^(*u*)^())^(*.*)

Now let* u, v** ∈N*(*J* ) be arbitrary. Then* J**i*(*u*) =* J**i*(*v*) = 0 for all* i* = 1*, ..., n*, hence for any *λ** ∈*R

0 ⩽*J**i*(*λu* +* v*) = 2^(*p*)^(*i*)*J**i*

*λu*

2 ^(+)^(* v*)

2

 ⩽2^(*p*)^(*i *)1

2^(*J*)^(*i*)

*λu*

2

 + ^(1)

2^(*J*)^(*i *)*v*

2

^()

= 1 2^(*J*)^(*i*)^(()^(*λu*)^() + 1)

2^(*J*)^(*i*)^(()^(*v*)^() =)^(* |*)^(*λ*)^(*|*)^(*p*)^(*i*)

2* *^(*J*)^(*i*)^(()^(*u*)^() + 1)

2^(*J*)^(*i*)^(()^(*v*)^() = 0)^(*.*)

Therefore,* J**i*(*λu* +* v*) = 0 for all* i* and hence* J* (*λu* +* v*) = 0.

**Lemma 4.2.4.*** Let assumptions of Lemma 4.2.3 be satisﬁed. Suppose that** u** ∈X** and **v** ∈N*(*J* )*. Then** J* (*u* +* v*) =* J* (*u*)*.*

*Proof.* Left as exercise.

If dim* N*(*J* )* <** ∞*, the subspace* N*(*J* ) is* complemented* in* X* [4, Thm. 5.89], i.e. there exists a closed subspace* X*0* ⊂X* such that* X*0* ∩N*(*J* ) =* {*0*}* and

*X* =* X*0* ⊕N*(*J* )*. *(4.5)

We will use this to establish coercivity of the functional (4.1).


---

## Page 48

48 *CHAPTER 4. VARIATIONAL REGULARISATION*

**Lemma 4.2.5.*** Suppose that the regularisation functional** J* :* X →*^(¯)R+* is proper, convex and satisﬁes conditions of Lemma* (4.2.3)* and let** A** ∈L*(*X**,** Y*)* be a bounded linear operator. Suppose also that*

*(i)* dim* N*(*J* )* <** ∞**and** J** is coercive on** X*0*, where** X*0* is such that** X* =* X*0* ⊕N*(*J* )*;*

*(ii) the kernels of** A** and** J** have a trivial intersection, i.e.** N*(*A*)* ∩N*(*J* ) =* {*0*}**.*

*Then the function*

Φ*α*(*u*) := ^(1)

2^(*∥*)^(*Au*)^(* −*)^(*f*)^(*∥*)^(2 )*Y* ^(+)^(* α*)^(*J*)^( ()^(*u*)^())

*is coercive on** X** for any** α >* 0*.*

*Proof.* Let* {**u**j**}**j**∈*N be a sequence in* X*. Due to (4.5), there exists a unique decomposition

*u**j* =* u*^(0 )*j* ^(+)^(* u*)^(*N *)*j** *^(*, *)*u*^(0 )*j** *^(*∈X*)^(0)^(*, u*)^(*N *)*j** *^(*∈N*)^(()^(*J*)^( ))^(*.*)

Let Φ*α*(*u**j*) ⩽*C* for all* j** ∈*N. Then* J* (*u**j*) ⩽*C* and

*J *� *u*^(0 )*j * =* J *� *u*^(0 )*j* ^(+)^(* u*)^(*N *)*j * =* J* (*u**j*) ⩽*C.*

Since* J* is coercive on* X*0, we get that* ∥**u*^(0 )*j*^(*∥*)^(⩽)^(*C*)^(*′*)^(. Now, deﬁne)

e*A*:* N*(*J* )* →**A**N*(*J* )*, *e*A* =* A**|**N*(*J* )* .*

That is, ^(e)*A* is the restriction of* A* to* N*(*J* ). Clearly, ^(e)*A* is surjective and by assumption (ii) it is also injective. Since* N*(*J* ) (and, subsequently,* A**N*(*J* )) is ﬁnite-dimensional, ^(e)*A*^(*−*)^(1)^( )exists and is bounded. Denote* ∥*^(e)*A*^(*−*)^(1)*∥*=: ^(e)*C*. Then

*∥**u*^(*N *)*j** *^(*∥ *)= *∥*^(e)*A*^(*−*)^(1)( ^(e)*Au*^(*N *)*j* ^())^(*∥*)^(⩽)^(e)^(*C*)^(*∥*)^(*Au*)^(*N *)*j** *^(*∥*)^(=)^( e)^(*C*)^(*∥*)^(*Au*)^(*N *)*j* ^(+)^(* Au*)^(0 )*j** *^(*−*)^(*f*)^(* −*)^(()^(*Au*)^(0 )*j** *^(*−*)^(*f*)^())^(*∥*)

⩽ e*C **Au**j** −**f**∥*+* ∥**Au*0 *j** *^(*−*)^(*f*)^(*∥ *) ⩽^(e)*C*(*C* +* ∥**A**∥∥**u*^(0 )*j*^(*∥*)^(+)^(* ∥*)^(*f*)^(*∥*)^())^( ⩽)^(*C*)^(*′′*)^(*.*)

Therefore, *∥**u**j**∥*=* ∥**u*^(0 )*j* ^(+)^(* u*)^(*N *)*j** *^(*∥*)^(⩽)^(*∥*)^(*u*)^(0 )*j*^(*∥*)^(+)^(* ∥*)^(*u*)^(*N *)*j** *^(*∥*)^(⩽)^(*C*)^(*′′′*)^(*,*)

which means that Φ*α* is coercive.

Now we are ready to establish the existence of a* J* -minimising solution and a regularised solution for any* α >* 0.

**Theorem 4.2.6.*** Let** X** and** Y** be a Banach spaces and** τ**X** and** τ**Y** some topologies (not necessarily induced by the norm) in** X** and** Y**, respectively. Assume that*

*(i) bounded sequences in** X** have** τ**X** -convergent subsequences;*

*(ii)** J* :* X →*^(¯)R+* is proper, convex** τ**X** -l.s.c. and satisﬁes assumptions of Lemma 4.2.5;*

*(iii)** A*:* X →Y** is** τ**X** →**τ**Y** continuous;*

*(iv)** ∥· ∥**Y** is** τ**Y**-lower semicontinuous;*

*Then*

*(i’) there exists a** J** -minimising solution** u*^(*† *)*J** *^(*of*)^( (4.4))^(*;*)


---

## Page 49

*4.2. WELL-POSEDNESS AND REGULARISATION PROPERTIES *49

*(ii’) for any ﬁxed** α >* 0* and** f** ∈Y** there exists a minimiser*

*u*^(*α*)^(* *)*∈*arg min *u**∈X*

1 2^(*∥*)^(*Au*)^(* −*)^(*f*)^(*∥*)^(2 )*Y* ^(+)^(* α*)^(*J*)^( ()^(*u*)^())^(*. *)(4.6)

*Proof. *(i) Let L be the set of least-squares solutions of (4.4). Then L can written as follows L =* {**u** ∈X* :* ∥**Au** −**f**∥**Y* ⩽*µ**}**,*

where* µ* := inf*{∥**Av** −**f**∥**Y* :* v** ∈X}*. Since* A* is* τ**X** →**τ**Y* continuous and* ∥· ∥**Y* is *τ**Y*-l.s.c., L is* τ**X* -closed.

Consider the following problem

inf *u**∈*L* *^(*J*)^( ()^(*u*)^() = inf )*u**∈X** *^(*J*)^( ()^(*u*)^() +)^(* χ*)^(L)^(()^(*u*)^())^(*. *)(4.7)

By the assumption that we made in the beginning of this section, this problem is feasible, i.e. there exists* u** ∈*L with* J* (*u*)* <** ∞*. The objective function in (4.7) is bounded from below. Using similar arguments as in Lemma 4.2.5, we conclude that it is also coercive. Since L is* τ**X* -closed,* χ*L is* τ**X* -l.s.c. By assumption ii,* J* is also *τ**X* -l.s.c. So, (4.7) satisﬁes the assumptions of the direct method (Theorem 4.1.37) and hence a minimiser exists.

(ii) From Lemma 4.2.5 we know that the objective function Φ*α* in (4.6) is coercive. It is also bounded from below. Since* J* is* τ**X* -l.s.c.,* A* is* τ**X** →**τ**Y* continuous and* ∥· ∥**Y* is *τ**Y*-l.s.c., we get that Φ*α* is* τ**X* -l.s.c. Using the direct method, we conclude that (4.6) has a minimiser.

Now we study the behaviour of the minimiser of (4.6) with* f* =* f**δ* (perturbed measure- ment) as* δ** →*0 when* α* =* α*(*δ*) is chosen according to an appropriate a priori parameter choice rule. For simplicity, we will do this in the case when inf*{∥**Av** −**f**∥**Y* :* v** ∈X}* = 0, i.e. least-squares solutions are actually solutions of (4.4).

**Theorem 4.2.7.*** Let the assumptions of Theorem 4.2.6 hold and suppose that* inf*{∥**Av** − **f**∥**Y* :* v** ∈X}* = 0*. Let** α* =* α*(*δ*)* be such that*

lim *δ**→*0* *^(*α*)^(()^(*δ*)^() = 0 )*and *lim sup *δ**→*0

*δ*^(2)

*α*(*δ*) ^(= 0)^(*.*)

*Then** u**δ* :=* u*^(*α*)^(()^(*δ*)^() )*δ τ**X **→**u*^(*† *)*J** *^(*as*)^(* δ*)^(* →*)^(0)^(* (possibly, along a subsequence) and*)^(* J*)^( ()^(*u*)^(*δ*)^())^(* →J*)^( ()^(*u*)^(*† *)*J* ^())^(*, *)*where** u*^(*† *)*J** *^(*is a*)^(* J*)^(* -minimising solution.*)

*Proof.* Let* u*0 be any* J* -minimising solution (which exists by Theorem 4.2.6). Since* u**δ *solves (4.6) with* α* =* α*(*δ*), we get that

1 2^(*∥*)^(*Au*)^(*δ*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2 )*Y* ^(+)^(* α*)^(()^(*δ*)^())^(*J*)^( ()^(*u*)^(*δ*)^() )⩽ 1 2^(*∥*)^(*Au*)^(0)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2 )*Y* ^(+)^(* α*)^(()^(*δ*)^())^(*J*)^( ()^(*u*)^(0)^())

⩽ *δ*^(2)

2 ^(+)^(* α*)^(()^(*δ*)^())^(*J*)^( ()^(*u*)^(0)^())^(*. *)(4.8)


---

## Page 50

50 *CHAPTER 4. VARIATIONAL REGULARISATION*

Therefore, we have the following two estimates

*J* (*u**δ*) ⩽ *δ*^(2)

2*α*(*δ*) ^(+)^(* J*)^( ()^(*u*)^(0)^())^( ⩽)^(*C, *)(4.9a)

*∥**Au**δ** −**f**δ**∥**Y *⩽ p

*δ*^(2)^( )+ 2*α*(*δ*)*J* (*u*0) ⩽*C*^(*′*)*, *(4.9b)

The right-hand side in (4.9a) is bounded uniformly in* δ*, because lim sup*δ**→*0* δ*^(2)*/α*(*δ*) = 0 by assumption and* J* (*u*0) is a constant independent of* δ*. The right-hand side in (4.9b) is bounded, because* J* (*u*0) is a constant and* δ, α*(*δ*)* →*0. Therefore, both* J* (*u**δ*) and* ∥**Au**δ** −**f**δ**∥**Y* are uniformly bounded. Proceeding similarly to Lemma 4.2.5, we get that *∥**u**δ**∥*⩽*C*

for all* δ*. Now let* δ**n** ↓*0 be an arbitrary null sequence. Since* u**δ**n* is bounded, it contains a *τ**X* -convergent subsequence (which we don’t relabel)

*u**δ**n **τ**X **→**u*^(*† *)*J *as* n** →∞*.

We will show that* u*^(*† *)*J* ^(is a)^(* J*)^( -minimising solution. From (4.9b) we observe that)

lim inf *n**→∞*^(*∥*)^(*Au*)^(*δ*)^(*n*)^(* −*)^(*f*)^(*δ*)^(*n*)^(*∥*)^(*Y*)^( ⩽)^(lim inf )*n**→∞*

p

*δ*^(2)*n* + 2*α*(*δ**n*)*J* (*u*0) = 0*.*

Since* A* is* τ**X** →**τ**Y* continuous and* ∥· ∥**Y* is* τ**Y*-l.s.c., we get that

*∥**Au*^(*† *)*J** *^(*−*)^(*f*)^(*∥*)^(*Y *)⩽ lim inf *n**→∞*^(*∥*)^(*Au*)^(*δ*)^(*n*)^(* −*)^(*f*)^(*∥*)^(*Y*)^( ⩽)^(lim inf )*n**→∞*^(()^(*∥*)^(*Au*)^(*δ*)^(*n*)^(* −*)^(*f*)^(*δ*)^(*n*)^(*∥*)^(*Y*)^( +)^(* ∥*)^(*f*)^(* −*)^(*f*)^(*δ*)^(*n*)^(*∥*)^(*Y*)^() = 0)^(*,*)

which shows that* u*^(*† *)*J* ^(is a least-squares solution. Using the estimate (4.9a) and)^(* τ*)^(*X*)^( -lower )semicontinuity of* J* , we obtain

*J* (*u*^(*† *)*J* ^())^( ⩽)^(lim inf )*n**→∞*^(*J*)^( ()^(*u*)^(*δ*)^(*n*)^())^( ⩽)^(lim sup )*n**→∞*^(*J*)^( ()^(*u*)^(*δ*)^(*n*)^())^( ⩽)^(lim sup )*n**→∞ **δ*^(2)

2*α*(*δ*) ^(+)^(* J*)^( ()^(*u*)^(0)^() =)^(* J*)^( ()^(*u*)^(0)^())^(*. *)(4.10)

Since* u*0 was an arbitrary* J* -minimising solution and* J* (*u*^(*† *)*J* ^())^( ⩽)^(*J*)^( ()^(*u*)^(0)^(), we conclude that )*J* (*u*^(*† *)*J* ^() is also a)^(* J*)^( -minimising solution. )Finally, since* J* (*u*^(*† *)*J* ^() =)^(* J*)^( ()^(*u*)^(0)^(), we conclude )from (4.10) that

lim inf *n**→∞*^(*J*)^( ()^(*u*)^(*δ*)^(*n*)^() = lim sup )*n**→∞*^(*J*)^( ()^(*u*)^(*δ*)^(*n*)^() = lim )*n**→∞*^(*J*)^( ()^(*u*)^(*δ*)^(*n*)^() =)^(* J*)^( ()^(*u*)^(*† *)*J* ^())^(*,*)

which completes the proof.

**Remark 4.2.8.** The theorem proves convergence of the regularised solutions in* τ**X* , which may diﬀer from the strong topology. However, if* J* satisﬁes the* Radon-Riesz property* with respect to the topology* τ**X* , i.e.* u**j τ**X **→**u* and* J* (*u**j*)* →J* (*u*) imply* ∥**u**j** −**u**∥→*0, then we get convergence in the norm topology. An example of a functional satisfying the Radon- Riesz property is the norm in a Hilbert (or reﬂexive Banach) space with* τ**X* being the weak topology.


---

## Page 51

*4.2. WELL-POSEDNESS AND REGULARISATION PROPERTIES *51

**Examples of regularisers**

**Example 4.2.9.** Let* X* be a Hilbert space and* J* (*u*) =* ∥**u**∥*^(2). The norm in a Hilbert space is weakly l.s.c. By Theorem 4.1.2 we know that (norm) bounded sequences have weakly convergent subsequences. Therefore, Assumption (ii) of Theorem 4.2.6 is satisﬁed with *τ**X* being the weak topology and we obtain weak convergence of the regularised solutions. However, since the norm in a Hilbert space has the Radon-Riesz property, we also get strong convergence. The same approach works in reﬂexive Banach spaces. A classical example is regularisation in Sobolev spaces such as the space* H*^(1)^( )of* L*^(2)

functions whose weak derivatives are also in* L*^(2). In the one-dimensional case, the space* H*^(1)

consists only of continuous functions (in higher dimensions it is true for Sobolev spaces with some other exponents), therefore, the regularised solutions will also be continuous. For this reason, the regulariser* J* (*u*) =* ∥**u**∥**H*1 is sometimes referred to as the* smoothing functional*. Whilst desirable in some applications, in imaging smooth reconstructions are usually not favourable, since images naturally contain edges and therefore are not continuous functions. To overcome this issue, other regularisers have been introduced that we will discuss later.

**Example 4.2.10** (*ℓ*^(1)-regularisation)**.** Let* X* =* ℓ*^(2)^( )be space of all square summable sequences (i.e. such that* ∥**u**∥*^(2 )*ℓ*^(2)^( =)^( P)^(*∞ *)*i*=1* *^(*u*)^(2 )*i** *^(*<*)^( +)^(*∞*)^(). For example,)^(* u*)^( can represent the coeﬃcients of a )function in a basis (e.g., a Fourier basis or a wavelet basis). As a regularisation functional, let us use not the* ℓ*^(2)-norm, but the* ℓ*^(1)-norm:

*J* (*u*) =* ∥**u**∥**ℓ*1 =

*∞ *X

*i*=1 *|**u**i**|**.*

By Example 4.1.5* J* (*·*) is weakly l.s.c. in* ℓ*^(2). It is evident that* ℓ*^(*q*)^(* *)*⊂**ℓ*^(*p*)^( )and* ∥· ∥**ℓ**p* ⩽*∥· ∥**ℓ**q *for* q* ⩽*p*. Therefore,* J* (*u*) ⩽*C* implies that* ∥· ∥**ℓ*2 ⩽*C* and, since* ℓ*^(2)^( )is a Hilbert space and bounded sequences have weakly convergent subsequences, we conclude that the sublevel sets of* J* (*·*) are weakly sequentially compact in* ℓ*^(2). Therefore, Assumption (ii) of Theorem 4.2.6 is satisﬁed with* τ**X* being the weak topology in* ℓ*^(2). Hence, we get weak convergence of regularised solutions in* ℓ*^(2). The motivation for using the* ℓ*^(1)-norm as the regulariser instead of the* ℓ*^(2)-norm is as follows. If the forward operator is non-injective, the inverse problem has more than one solution and the solutions form an aﬃne subspace. In the context of sequence spaces representing coeﬃcients of the solution in a basis, it is sometimes beneﬁcial to look for solutions that are* sparse* in the sense that they have ﬁnite support, i.e.* |* supp(*u*)*|** <** ∞ *with supp(*u*) =* {**i** ∈*N* |** u**i** ̸*= 0*}*. This allows explaining the signal with a ﬁnite (and often relatively small) number of basis functions and has widely ranging applications in, for instance, compressed sensing. A ﬁnite dimensional illustration of the sparsity of* ℓ*^(1)- regularised solutions is given in Figure 4.8. The corresponding minimisation problem

min *u**∈**ℓ*^(2)

1

2* *^(*∥*)^(*Au*)^(* −*)^(*f*)^(*∥*)^(2 )*ℓ*^(2)^( +)^(* α*)^(*∥*)^(*u*)^(*∥*)^(1)

 *. *(4.11)

is also called* lasso* in the statistical literature.

**Example 4.2.11** (Elastic net regularisation)**.** The* ℓ*^(1)^( )regulariser described in the previous example sometimes delivers undesirable results for problems where there are highly cor- related features and we need to identify all relevant ones, e.g. microarray data analysis


---

## Page 52

52 *CHAPTER 4. VARIATIONAL REGULARISATION*

minimal* ℓ*^(2)-norm minimal* ℓ*^(1)-norm

Figure 4.8: Non-injective operators have a non-trivial kernel such that the inverse problem has more than one solution and the solutions form an aﬃne subspace visualised by the solid line. Diﬀerent regularisation functionals favour diﬀerent solutions. The circle and the diamond indicate all points with constant* ℓ*^(2)-norm, respectively* ℓ*^(1)-norm, and the minimal *ℓ*^(2)-norm and* ℓ*^(1)-norm solutions are the intersections of the line with the circle, respectively the diamond. As it can be seen, the minimal* ℓ*^(2)-norm solution has two non-zero components while the minimal* ℓ*^(1)-norm solution has only one non-zero component and thus is* sparser*.

(analysis of genomic sequences), in that it tends to select only one feature out of the rel- evant group instead of all relevant features of the group, i.e. it fails to identify the group structure. Elastic net regularisation helps to overcome this issue. The elastic net regulariser *J* :* ℓ*^(2)^(* *)*→*^(¯)R+ is deﬁned as follows

*J* (*u*) :=* α**∥**u**∥**ℓ*1 +* β**∥**u**∥*^(2 )*ℓ*^(2)^(*,*)

where* α, β >* 0 are constants that balance the inﬂuence of the two terms. Since* J* is the sum of a 1-homogeneous term and a 2-homogeneous term, it satisﬁes assumptions of Lemma 4.2.3.

### **4.3 Total Variation Regularisation**

As pointed out in Example 4.2.9, in imaging we are interested in regularisers that allow for discontinuities while maintaining suﬃcient regularity of the reconstructions. One popular choice is the so-called* total variation* regulariser [15].

**Deﬁnition 4.3.1.*** Let* Ω*⊂*R^(*n*)^(* *)*be a bounded domain and** u** ∈**L*^(1)(Ω)*. Let** D*(Ω*,* R^(*n*))* be the following set of vector-valued* test functions* (i.e. functions that map from* Ω*to* R^(*n*)*)*

*D*(Ω*,* R^(*n*)) := n *ϕ** ∈**C*^(*∞ *)0 ^((Ω;)^( R)^(*n*)^() ) sup *x**∈*Ω *∥**ϕ*(*x*)*∥*2 ⩽1 o *.*

*Total variation of** u** ∈**L*^(1)(Ω)* is deﬁned as follows*

TV(*u*) = sup *ϕ**∈D*(Ω*,*R^(*n*))

Z

Ω *u*(*x*) div* ϕ*(*x*)* dx .*


---

## Page 53

*4.3. TOTAL VARIATION REGULARISATION *53

**Remark 4.3.2.** Deﬁnition 4.3.1 may seem a bit strange at the ﬁrst glance, but we note that for a function* u** ∈**L*^(1)(Ω) whose weak derivative* ∇**u* exists and is also in* L*^(1)(Ω*,* R^(*n*)) (i.e. *u* belongs to the Sobolev space* W* ^(1)^(*,*)^(1)(Ω)) we obtain, integrating by parts, that

TV(*u*) = sup *ϕ**∈D*(Ω*,*R^(*n*))

Z

Ω *−⟨∇**u*(*x*)*, ϕ*(*x*)*⟩**dx.*

By the Cauchy-Schwartz inequality we get that* | ⟨∇**u*(*x*)*, ϕ*(*x*)*⟩|* ⩽*∥∇**u*(*x*)*∥*2*∥**ϕ*(*x*)*∥*2 ⩽ *∥∇**u*(*x*)*∥*2 for a.e.* x** ∈*Ω. On the other hand, choosing* ϕ* such that* ϕ*(*x*) =* − **∇**u*(*x*) *∥∇**u*(*x*)*∥*2 ^((tech- )nically, such* ϕ* is not necessarily in* D*(Ω*,* R^(*n*)), but we can approximate it with functions from *D*(Ω*,* R^(*n*)), since any function in* W* ^(1)^(*,*)^(1)(Ω) can be approximated with smooth functions [2, Thm. 3.17]; we omit the technicalities here), we get that* −⟨∇**u*(*x*)*, ϕ*(*x*)*⟩*=* ∥∇**u*(*x*)*∥*2. Therefore, the supremum over* ϕ** ∈D*(Ω*,* R^(*n*)) is equal to

TV(*u*) = Z

Ω *∥∇**u*(*x*)*∥*2* dx* =* ∥∇**u**∥**L*1*.*

This shows that TV just penalises the the* L*^(1)^( )norm (of the pointwise 2-norm) of the gradient for any* u** ∈**W* ^(1)^(*,*)^(1)(Ω). However, we will see that the space of functions that have ﬁnite value of TV is larger than* W* ^(1)^(*,*)^(1)(Ω) and contains, for instance, discontinuous functions.

**Remark 4.3.3.** It can be shown [13] that for any* u** ∈**L*^(1)(Ω)

TV(*u*) =* ∥∇**u**∥*M*,*

where* ∇*is the distributional gradient and* ∥· ∥*M is the Radon norm. That is, Total Variation extends the* L*^(1)^( )norm of the gradient for functions whose gradient is not a Lebesgue- measurable function. We will not use this interpretation of the Total Variation to simplify the presentation and refer the interested reader to [13] for details.

**Proposition 4.3.4.** TV* is a proper, convex and absolutely* 1*-homogeneous functional** L*^(1)(Ω)* → *¯R*. For any constant function*** c**:** c**(*x*)* ≡**c** ∈*R* for all** x** and any** u** ∈**L*^(1)(Ω)

TV(**c**) = 0 *and *TV(*u* +** c**) = TV(*u*)*.*

*Proof.* Left as exercise.

**Remark 4.3.5.** It can be shown that the opposite implication holds, i.e. TV(*u*) = 0 implies that* u* is constant. in other words,

*N*(TV) =* {**u** ∈**L*^(1)(Ω):* u* =* const**}**. *(4.12)

The easiest way to see this is using the Radon measure interpretation in Remark 4.3.3. Because time constraints, we will omit the proof.

**Example 4.3.6** (TV of an indicator function)**.** Suppose* C ⊂*Ω*⊂*R^(2)^( )is a bounded domain with smooth boundary and* u*(*·*) =** 1***C*(*·*) is its indicator function, i.e.

**1***C*(*x*) =

( 1 *x** ∈C *0 *x** ∈X \ C** *^(*.*)


---

## Page 54

54 *CHAPTER 4. VARIATIONAL REGULARISATION*

Then, using the divergence theorem, we get that for any test function* ϕ** ∈D*(Ω*,* R^(*n*)) Z

Ω *u*(*x*) div* ϕ*(*x*)* dx* = Z

*C *div* ϕ*(*x*)* dx* = Z

*∂**C **⟨**ϕ*(*x*)*,*** n***∂**C*(*x*)*⟩**dl,*

where* ∂**C* is the boundary of* C* and** n***∂**C*(*x*) is the unit normal at* x*. Hence,

TV(*u*) = sup *ϕ**∈D*(Ω*,*R^(*n*))

Z

Ω *u*(*x*) div* ϕ*(*x*)* dx* = sup *ϕ**∈D*(Ω*,*R^(*n*))

Z

*∂**C **⟨**ϕ*(*x*)*,*** n***∂**C*(*x*)*⟩**dl*

⩽ sup *ϕ**∈D*(Ω*,*R^(*n*))

Z

*∂**C **∥**ϕ*(*x*)*∥∥***n***∂**C*(*x*)*∥**dl* ⩽ sup *ϕ**∈D*(Ω*,*R^(*n*))

Z

*∂**C **dl* = Per*C**,*

where Per(*C*) is the perimeter of* C*. On the other hand, since* ∂**C* is smooth and* ∥***n***∂**C*(*x*)*∥*= 1 for every* x*,** n***∂**C* can be extended to feasible vector ﬁeld on Ω(i.e. one that is in* D*(Ω*,* R^(*n*))). Therefore, we get that

TV(*u*) = Z

*∂**C **⟨**ϕ*(*x*)*,*** n***∂**C*(*x*)*⟩**dl* ⩾ Z

*∂**C **∥***n***∂**C*(*x*)*∥*^(2)^(* *)*dl* = Z

*∂**C *1* ·** dl* = Per(*C*)*,*

Therefore, TV(**1***C*) = Per*C* for any domain with smooth boundary. This can be extended to domains with Lipschitz boundary by constructing a sequence of functions in* D*(Ω*,* R^(*n*)) that converge pointwise to** n***∂**C*.

We now study properties of functions that have a ﬁnite value of TV.

**Deﬁnition 4.3.7.*** The functions** u** ∈**L*^(1)(Ω)* with a ﬁnite value of* TV* form a normed space called the space of functions of bounded variation (the* BV*-space) deﬁned as follows*

BV(Ω) := n *u** ∈**L*^(1)(Ω) * ∥**u**∥*BV :=* ∥**u**∥**L*1 + TV(*u*)* <** ∞ *o *.*

**Remark 4.3.8.** It can be shown that the space BV is the dual of a separable Banach space [13] and that weak-* convergence* u**n** ⇀*^(*∗*)*u* in BV is equivalent to strong convergence *u**n** →**u* in* L*^(1)^( )and convergence of the values TV(*u**n*)* →*TV(*u*). The proof is outside the scope of these notes.

We note that BV(Ω) is compactly embedded in* L*^(1)(Ω). We start with the following classical result.

**Theorem 4.3.9** (Rellich-Kondrachov, [2, Thm. 6.3])**.*** Let* Ω*⊂*R^(*n*)^(* *)*be a bounded Lipschitz domain (i.e. non-empty, open, connected and with Lipschitz boundary) and** p, m** ∈*N*. Let*

*p*^(*∗*):=

( *np n**−**mp **if** n > mp**,*

*∞ **if** n* ⩽*mp.*

*Then the embedding** W** *^(*m,p*)(Ω)* →**L*^(*q*)(Ω)* is continuous for all* 1 ⩽*q* ⩽*p*^(*∗*)*and compact for all* 1 ⩽*q < p*^(*∗*)*.*

Since functions from BV(Ω) can be approximated by functions in the Sobolev space *W* ^(1)^(*,*)^(1)(Ω) [5, Thm. 3.9], the Rellich-Kondrachov Theorem (with* p* = 1,* m* = 1) gives us the following


---

## Page 55

*4.3. TOTAL VARIATION REGULARISATION *55

**Corollary 4.3.10** ([5, Corrollary 3.49])**.** For any bounded Lipschitz domain Ω*⊂*R^(*n*), the embedding BV(Ω)* ⊂⊂**L*^(1)(Ω)

is compact for any* n* ⩾2 and the embedding

BV(Ω)* ,**→**L*^(2)(Ω)

is continuous for* n* = 2.

Now we will show that TV is lower-semicontinuous in* L*^(1).

**Theorem 4.3.11.*** Let* Ω*⊂*R^(*n*)^(* *)*be open and bounded. Then the total variation is l.s.c. in **L*^(1)(Ω)*.*

*Proof.* Let* {**u**j**}**j**∈*N* ⊂*BV(Ω) be a sequence converging in* L*^(1)(Ω) with* u**j** →**u* in* L*^(1)(Ω). Then for any test function* ϕ** ∈D*(Ω*,* R^(*n*)) we have that Z

Ω *u**j*(*x*) div* ϕ*(*x*)*dx** → *Z

Ω *u*(*x*) div* ϕ*(*x*)*dx*

(strong convergence implies weak convergence) and therefore

TV(*u*) = sup *ϕ**∈D*(Ω*,*R^(*n*))

Z

Ω *u*(*x*) div* ϕ*(*x*)*dx*

= sup *ϕ**∈D*(Ω*,*R^(*n*)) lim *j**→∞*

Z

Ω *u**j*(*x*) div* ϕ*(*x*)*dx*

⩽ lim inf *j**→∞ *sup *ϕ**∈D*(Ω*,*R^(*n*))

Z

Ω *u**j*(*x*) div* ϕ*(*x*)*dx*

= lim inf *j**→∞*^(TV()^(*u*)^(*j*)^())^(*.*)

Here the lim inf appears when we swap the sup and the lim, because the limit of the suprema may not exist; however, the inequality holds for any subsequence and hence also for the lim inf. Note also that the left and right hand sides may not be ﬁnite.

Since the null space of total variation (4.12) is nontrivial, TV cannot be coercive on* L*^(1). However, the following result helps.

**Proposition 4.3.12** ([5, Remark 3.50])**.*** Let* Ω*⊂*R^(*n*)^(* *)*be a bounded Lipschitz domain. Then there exists a constant** C >* 0* such that for all** u** ∈*BV(Ω)* the* Poincar´e* inequality is satisﬁed*

*∥**u** −**u*Ω*∥**L*1 ⩽*C* TV(*u*)*,*

*where** u*Ω:= 1 *|*Ω*| *R

Ω^(*u*)^(()^(*x*)^())^(*dx*)^(* is the mean-value of*)^(* u*)^(* over*)^( Ω)^(*.*)

**Corollary 4.3.13.** It is often useful to consider a subspace BV0(Ω)* ⊂*BV(Ω) of functions with zero mean, i.e.

BV0(Ω) :=* {**u** ∈*BV(Ω): Z

Ω *u*(*x*)*dx* = 0*}**. *(4.13)

Then for every function* u** ∈*BV0(Ω) we have that

*∥**u**∥**L*1 ⩽*C* TV(*u*)*.*

Clearly, BV0* ⊂**L*^(1 )0 ^(:=)^(* {*)^(*u*)^(* ∈*)^(*L*)^(1)^( : )R

*ω** *^(*u*)^(()^(*x*)^())^(* dx*)^( = 0)^(*}*)^( in TV is coercive on this subspace. Since )dim(*N*(TV)) = 1* <** ∞*, we have

*L*^(1)^( )=* L*^(1 )0* *^(*⊕N*)^((TV))^(*.*)


---

## Page 56

56 *CHAPTER 4. VARIATIONAL REGULARISATION*

Combining all the above results we get

**Theorem 4.3.14.*** Let** X* =* L*^(1)(Ω)*, where* Ω*⊂*R^(*n*)^(* *)*is bounded Lipschitz, and** Y** be a Banach space. Let** A*:* L*^(1)^(* *)*→Y** be a linear bounded operator such that** A***1*** ̸*= 0*, where*** 1*** is the constant-one function. Then minimisers of the following problem*

min *u**∈**L*^(1)(Ω) 1 2^(*∥*)^(*Au*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2 )*Y* ^(+)^(* α*)^(()^(*δ*)^() TV()^(*u*)^())

*converge strongly in** L*^(1)^(* *)*to a* TV*-minimising solution as** δ** →*0* if** α*(*δ*)* is chosen as required by Theorem 4.2.7.*

*Proof.* We have established all ingredients required for Theorem 4.2.7 to hold except that bounded sequences in* L*^(1)^( )may not have convergent subsequences (*L*^(1)^( )is not a dual space). However, the compact embedding from Corollary 4.3.10 guarantees that sequences with a bounded value of TV have subsequences that converge strongly in* L*^(1).

**Remark 4.3.15.** One can replace optimisation over* u** ∈**L*^(1)^( )with optimisation over* u** ∈*BV, which is the eﬀective domain of the objective function.

Total Variation is widely used in imaging applications [34]. The so-called Rudin–Osher– Fatemi (ROF) model for image denoising [31] consists in minimising the following functional

min *u**∈*BV(Ω) 1 2^(*∥*)^(*Iu*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2 )*L*^(2)(Ω) ^(+)^(* α*)^( TV()^(*u*)^())^(*, *)(4.14)

where Ω*⊂*R^(2). In this case, the forward operator* I* is the embedding operator BV(Ω)* → **L*^(2)(Ω), which is continuous for two-dimensional domains (see Corollary 4.3.10). Clearly, *A***1*** ̸*= 0 is satisﬁed. More generally, one considers the following optimisation problem

min *u**∈*BV(Ω)* *^(*∥*)^(*Au*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2 )2 ^(+)^(* α*)^( TV()^(*u*)^())^(*, *)(4.15)

where* A*: BV(Ω)* →**L*^(2)(Ω) is such that* A***1*** ̸*= 0.


---

## Page 57

# **Chapter 5**
# **Convex Duality**

In Chapter 4 we have established convergence of a regularised solution* u**δ* to a* J* -minimising solution* u*^(*† *)*J* ^(as)^(* δ*)^(* →*)^(0. However, we didn’t get any results on the)^(* speed*)^( of this convergence, )which is referred to as the* convergence rate*. In modern regularisation methods, convergence rates are usually studied using* Bregman distances* associated with the (convex) regularisation functional* J* . Recall that for a convex functional* J* ,* u, v** ∈X* such that* J* (*v*)* <** ∞*and* q** ∈**∂**J* (*v*), the (generalised) Bregman distance is given by the following expression (cf. Def. 4.1.24)

*D*^(*q *)*J* ^(()^(*u, v*)^() =)^(* J*)^( ()^(*u*)^())^(* −J*)^( ()^(*v*)^())^(* −⟨*)^(*q, u*)^(* −*)^(*v*)^(*⟩*)^(*.*)

Also widely used is the* symmetric* Bregman distance (cf. Def. 4.1.26) given by the following expression (here* p** ∈**∂**J* (*u*))

*D*^(*symm *)*J *(*u, v*) =* D*^(*q *)*J* ^(()^(*u, v*)^() +)^(* D*)^(*p *)*J* ^(()^(*v, u*)^() =)^(* ⟨*)^(*p*)^(* −*)^(*q, u*)^(* −*)^(*v*)^(*⟩*)^(*.*)

Bregman distances appear to be a natural distance measure between a regularised solu- tion* u**δ* and a* J* -minimising solution* u*^(*† *)*J* ^(. For instance, for classical Hilbert space regulari- )sation with* J* (*u*) = ^(1)

2^(*∥*)^(*u*)^(*∥*)^(2 )*X* ^(, the subgradient at)^(* u*)^(*† *)*J* ^(is)^(* p*)*u*^(*† *)*J* ^(=)^(* u*)^(*† *)*J* ^((since)^(* J*)^( is diﬀerentiable) )and we get the following expression

*D **u*^(*† *)*J **J* ^(()^(*u*)^(*δ*)^(*, u*)^(*† *)*J* ^() = 1)

2^(*∥*)^(*u*)^(*δ*)^(*∥*)^(2 )*X** *^(*−*)^(1)

2^(*∥*)^(*u*)^(*† *)*J** *^(*∥*)^(2 )*X** *^(*− *)D *u*^(*† *)*J** *^(*, u*)^(*δ*)^(* −*)^(*u*)^(*† *)*J *E

= ^(1)

2^(()^(*∥*)^(*u*)^(*δ*)^(*∥*)^(2 )*X** *^(*−*)^(2 )D *u*^(*† *)*J** *^(*, u*)^(*δ *)E +* ∥**u*^(*† *)*J** *^(*∥*)^(2 )*X* ^() = 1)

2^(*∥*)^(*u*)^(*δ*)^(* −*)^(*u*)^(*† *)*J** *^(*∥*)^(2 )*X** *^(*,*)

which happens to coincide with the symmetric Bregman distance. Therefore, in the clas- sical* L*^(2)-case, the Bregman distance just measures the* L*^(2)-distance between a regularised solution and a* J* -minimising solution. As we have seen in an example sheet, subgradients of absolutely one-homogeneous functional carry structural information about the solution such as locations of non-zero components of a vector* u*^(*† *)*J** *^(*∈*)^(*ℓ*)^(1)^(. )We are looking for a convergence rate of the following form

*D*^(*symm *)*J *(*u**δ**, u*^(*† *)*J* ^())^( ⩽)^(*ψ*)^(()^(*δ*)^())^(*,*)

where* ψ*: R+* →*R+ is a known function of* δ* such that* ψ*(*δ*)* →*0 as* δ** →*0.

57


---

## Page 58

58 *CHAPTER 5. CONVEX DUALITY*

### **5.1 Dual Problem**

Recall that* u**δ* solves the following problem

min *u**∈X *1 2^(*∥*)^(*Au*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2 )*Y* ^(+)^(* α*)^(*J*)^( ()^(*u*)^())^(*. *)(5.1)

with an appropriately chosen* α* =* α*(*δ*), where* X* and* Y* are Banach spaces,* A** ∈L*(*X**,** Y*) and* E* :* Y →*^(¯)R and* J* :* X →*^(¯)R is proper, convex and l.s.c. and satisﬁes Assumptions of Theorem 4.2.6. For simplicity of presentation, we will also assume throughout this chapter that* J* is absolutely one-homogeneous and that inf*{∥**Av** −**f**∥*:* v** ∈X}* = 0, i.e.* Au*^(*† *)*J* ^(=)^(* f *)for any* J* -minimising solution. To apply the results of Section 4.1.4 to (5.1), we take (in the notation of Section 4.1.4)

*E*(*y*) := ^(1)

2^(*∥*)^(*y*)^(* −*)^(*f*)^(*∥*)^(2 )*Y*^(*, *)*F*(*u*) :=* α**J* (*u*)*.*

**Lemma 5.1.1.*** Let** X** be a Banach space with norm** ∥· ∥**X** and let** ∥· ∥**X**∗**be the norm in the dual space of** X**. Let** ϕ*(*x*) := ^(1)

2^(*∥*)^(*x*)^(*∥*)^(2 )*X*^(*. Then the convex conjugate of*)^(* ϕ*)^(* is*)

*ϕ*^(*∗*)(*ξ*) = ^(1)

2^(*∥*)^(*ξ*)^(*∥*)^(2 )*X*^(*∗*)^(*, *)*ξ** ∈**X*^(*∗*)*.*

*Proof.* First, we note that

*ϕ*^(*∗*)(*ξ*) = sup *x**∈**X **⟨**ξ, x**⟩−*^(1)

2^(*∥*)^(*x*)^(*∥*)^(2 )*X* ^(⩽)^(sup )*x**∈**X **∥**x**∥**X**∥**ξ**∥**X**∗**−*^(1)

2^(*∥*)^(*x*)^(*∥*)^(2 )*X*^(*.*)

The function on the right-hand side is a parabola in the scalar variable* ∥**x**∥**X* and its maximum is ^(1)

2^(*∥*)^(*ξ*)^(*∥*)^(2 )*X*^(*∗*)^(. Now, ﬁx)^(* ξ*)^(* ∈*)^(*X*)^(*∗*)^(. We have that)

*∥**ξ**∥**X**∗*= sup *x**∈**X **∥**x**∥*=1

*⟨**ξ, x**⟩*= sup *x**∈**X **∥**x**∥*=*∥**ξ**∥*

*⟨**ξ, x**⟩*

*∥**ξ**∥*^(*.*)

Let* x*^(*ξ *)*n** *^(*∈*)^(*X*)^( be a maximising sequence (that is,)^(* ∥*)^(*x*)^(*ξ *)*n*^(*∥*)^(=)^(* ∥*)^(*ξ*)^(*∥*)^(and)^(* ⟨*)^(*ξ, x*)^(*ξ *)*n*^(*⟩→∥*)^(*ξ*)^(*∥*)^(2)^(). Then)

*ϕ*^(*∗*)(*ξ*) = sup *x**∈**X **⟨**ξ, x**⟩−*^(1)

2^(*∥*)^(*x*)^(*∥*)^(2 )*X* ^(⩾)^(lim sup )*n**→∞*

 *⟨**ξ, x*^(*ξ *)*n*^(*⟩−*)^(1)

2^(*∥*)^(*x*)^(*ξ *)*n*^(*∥*)^(2 )*X*

 =* ∥**ξ**∥*^(2)^(* *)*−*^(1)

2^(*∥*)^(*ξ*)^(*∥*)^(2)^( = 1)

2^(*∥*)^(*ξ*)^(*∥*)^(2)^(*.*)

The inequality here is due to the fact that the lim sup is a supremum over a smaller set than the whole* X*. Hence, we have that ^(1)

2^(*∥*)^(*ξ*)^(*∥*)^(2)^( ⩽)^(*ϕ*)^(*∗*)^(()^(*ξ*)^())^( ⩽)^(1)

2^(*∥*)^(*ξ*)^(*∥*)^(2)^( and the proof is complete.)

**Corollary 5.1.2.** Theorem 4.1.23 implies that for any* x** ∈**X* and any* ξ** ∈**∂ϕ*(*x*) it holds

1 2^(*∥*)^(*x*)^(*∥*)^(2 )*X* ^(+ 1)

2^(*∥*)^(*ξ*)^(*∥*)^(2 )*X*^(*∗*)^(=)^(* ⟨*)^(*ξ, x*)^(*⟩*)^(*.*)

Using the Cauchy-Schwarz inequality on the right-hand side and rearranging terms, we get that (*∥**x**∥**X** −∥**ξ**∥**X**∗*)^(2)^( )= 0 and hence

*∥**ξ**∥**X**∗*=* ∥**x**∥**X**.*


---

## Page 59

*5.1. DUAL PROBLEM *59

Now, for* E* and* F* as deﬁned above, we get

*E*^(*∗*)(*η*) = sup *y**∈Y **⟨**η, y**⟩−*^(1)

2^(*∥*)^(*y*)^(* −*)^(*f*)^(*∥*)^(2 )*Y* ^(=)^(* ⟨*)^(*η, f*)^(*⟩−*)^(sup )*z**∈Y*

 *⟨**η, z**⟩−*^(1)

2^(*∥*)^(*z*)^(*∥*)^(2 )*Y*

 =* ⟨**η, f**⟩*+ ^(1)

2^(*∥*)^(*η*)^(*∥*)^(*Y*)^(*∗*)^(*,*)

*F** *^(*∗*)(*p*) = *χ**∂**J* (0) * p*

*α*

 *,*

where the second equality holds since* F* is absolutely one-homogeneous. Hence, the dual problem of (5.1) is given by

sup *η**∈Y*^(*∗*)^(*−⟨*)^(*η, f*)^(*⟩−*)^(1)

2^(*∥*)^(*η*)^(*∥*)^(2 )*Y*^(*∗*)^(*−*)^(*χ*)*∂**J* (0)

*−**A**∗**η*

*α*

 *.*

Denote* µ* :=* −*^(*η*)

*α** *^(*∈Y*)^(*∗*)^(. Since)^(* −*)^(*χ*)^(*∂*)^(*J*)^( (0))^( =)^(* −∞*)^(outside)^(* ∂*)^(*J*)^( (0), we get the following equivalent )problem sup *µ**∈Y*^(*∗ *)*A*^(*∗*)*µ**∈**∂**J* (0)

*α * *⟨**µ, f**⟩−*^(*α*)

2* *^(*∥*)^(*µ*)^(*∥*)^(2 )*Y * *. *(5.2)

Let us check if Assumptions of Theorem 4.1.43 are satisﬁed. Condition (i) (coercivity) is guaranteed by Lemma 4.2.5. Condition (ii) (continuity of* E*) is satisﬁed at* u*0 = 0. Therefore, for any* δ >* 0 there exists a solution* µ**δ* of the dual problem (5.2). Existence of a primal solution* u**δ* is guaranteed by Theorem 4.2.6. Indeed, let us take *τ**X* to be the weak* topology in* X* and* τ**Y* a topology in* Y* such that* A* is* τ**X* -*τ**Y* continuous and the norm in* Y* is* τ**Y*-l.s.c. (weak*, weak or strong topologies will work). For example, if *Y* has a separable predual, we can take* τ**Y* to be the weak* topology on* Y*. It can be easily veriﬁed that* A* is weak*-weak* continuous if it is the dual of another operator* A* =* B*^(*∗*)

(where* B* acts from the predual of* Y* into the predual of* X*). With these choices, the conditions of Theorem 4.2.6 are satisﬁed. Hence, by strong duality we have that

1 2^(*∥*)^(*Au*)^(*δ*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2 )*Y* ^(+)^(* α*)^(*J*)^( ()^(*u*)^(*δ*)^() =)^(* α*)^(* ⟨*)^(*µ*)^(*δ*)^(*, f*)^(*δ*)^(*⟩−*)^(*α*)^(2)

2* *^(*∥*)^(*µ*)^(*δ*)^(*∥*)^(2 )*Y*^(*.*)

Optimality conditions (iii) from Theorem 4.1.43 take the following form

*A*^(*∗*)*µ**δ** ∈**∂**J* (*u**δ*)*, **−**αµ**δ** ∈**∂ *1

2^(*∥· ∥*)^(2 )*Y*

 (*Au**δ** −**f**δ*)*. *(5.3)

From Corollary 5.1.2 we conclude that

*∥**αµ**δ**∥**Y**∗*=* ∥**Au**δ** −**f**δ**∥**Y**. *(5.4)

Also, comparing the values of ^(1)

2^(*∥· ∥*)^(2)^( at 0 and at)^(* Au*)^(*δ*)^(* −*)^(*f*)^(*δ*)^( and using the fact that)^(* −*)^(*αµ*)^(*δ*)^( is )a subgradient, we get that

0 ⩾^(1)

2^(*∥*)^(*Au*)^(*δ*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2 )*Y* ^(+)^(* ⟨−*)^(*αµ*)^(*δ*)^(*,*)^( 0)^(* −*)^(()^(*Au*)^(*δ*)* *^(*−*)^(*f*)^(*δ*)^())^(*⟩*)

and therefore *⟨**αµ**δ**, Au**δ** −**f**δ**⟩*⩽*−*^(1)

2^(*∥*)^(*Au*)^(*δ*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2 )*Y*^(*. *)(5.5)

We will use the estimates (5.4) and (5.5) later in Theorem 5.2.4.


---

## Page 60

60 *CHAPTER 5. CONVEX DUALITY*

### **5.2 Source Condition and Convergence Rates**

Formal limits of problems (5.1) and (5.2) at* δ* = 0 are

inf *u*:* Au*=*f** *^(*J*)^( ()^(*u*)^() = inf )*u**∈X** *^(*χ*)^(*{*)^(*f*)^(*}*)^(()^(*Au*)^() +)^(* J*)^( ()^(*u*)^() )(5.6)

and

sup *µ*:* A*^(*∗*)*µ**∈**∂**J* (0) *⟨**µ, f**⟩*= sup *µ*:* A*^(*∗*)*µ**∈**∂**J* (0)

D *µ, Au*^(*† *)*J *E

= sup *µ*:* A*^(*∗*)*µ**∈**∂**J* (0)

D *A*^(*∗*)*µ, u*^(*† *)*J *E = sup *v**∈R*(*A*^(*∗*))*∩**∂**J* (0)

D *v, u*^(*† *)*J *E *. *(5.7)

Since the characteristic function* χ**{**f**}*(*·*) is not continuous anywhere in its domain, The- orem 4.1.43 does not apply and we cannot guarantee that a solution of the dual limit problem (5.7) exists. Indeed, since* R*(*A*^(*∗*)) is not closed (strongly and hence weakly, since it is convex [18, Thm. V.3.13]), a solution may not exist. We shall see that existence is guaranteed by the following condition

**Deﬁnition 5.2.1** (Source condition [14])**.*** We say that a** J** -minimising solution** u*^(*† *)*J** *^(*satisﬁes *)*the* source condition* if*

*∃**µ*^(*†*)^(* *)*∈Y*^(*∗ *)*such that **A*^(*∗*)*µ*^(*†*)^(* *)*∈**∂**J* (*u*^(*† *)*J* ^())^(*, *)(5.8)

*i.e. if** R*(*A*^(*∗*))* ∩**∂**J* (*u*^(*† *)*J* ^())^(* ̸*)^(=)^(* ∅*)^(*.*)

First we will see that this condition is necessary for the dual solution* µ**δ* from (5.3) to stay bounded as* δ** →*0.

**Theorem 5.2.2** (Necessary conditions, [24])**.*** Let** X** and** Y** be Banach spaces and** Y** sep- arable. Let conditions of Theorem 4.2.6 be satisﬁed and** α* =* α*(*δ*)* be chosen as required by Theorem 4.2.7. Suppose that the dual solution** µ**δ** is bounded uniformly in** δ**. Then there exists** µ*^(*†*)^(* *)*∈Y*^(*∗*)*such that** A*^(*∗*)*µ*^(*†*)^(* *)*∈**∂**J* (*u*^(*† *)*J* ^())^(*.*)

*Proof.* Consider an arbitrary sequence* δ**n** ↓*0. Since* ∥**µ**δ**∥**Y**∗*⩽*C* for all* δ*, by the Banach- Alaogly theorem we get that there exists a weakly-* convergent subsequence (that we do not relabel), i.e. *µ**δ**n** ⇀*^(*∗*)*µ*^(*†*)^(* *)*∈Y*^(*∗*)*.*

Then we get that *A*^(*∗*)*µ**δ**n** ⇀*^(*∗*)*A*^(*∗*)*µ*^(*†*)*.*

Since* ∂**J* (0) is weakly-* closed (Theorem 4.1.19) and* A*^(*∗*)*µ**δ**n** ∈**∂**J* (0) by (5.3), we get that

*A*^(*∗*)*µ*^(*†*)^(* *)*∈**∂**J* (0)*.*

Since* J* is absolute one-homogeneous, we get by Proposition 4.1.28 that

*⟨**A*^(*∗*)*µ**δ**n**, u**δ**n**⟩*=* J* (*u**δ**n*)* →J* (*u*^(*† *)*J* ^())^(*, *)(5.9)

where convergence follows from Theorem 4.2.7. We also observe that

*|⟨**A*^(*∗*)*µ**δ**, u**δ**⟩−⟨**A*^(*∗*)*µ*^(*†*)*, u*^(*† *)*J** *^(*⟩| *)= *|⟨**A*^(*∗*)*µ**δ**, u**δ** −**u*^(*† *)*J** *^(*⟩−⟨*)^(*A*)^(*∗*)^(()^(*µ*)^(*†*)^(* −*)^(*µ*)^(*δ*)^())^(*, u*)^(*† *)*J** *^(*⟩|*)

⩽ *|⟨**µ**δ**, Au**δ** −**f**⟩|* +* |⟨**µ*^(*†*)^(* *)*−**µ**δ**, f**⟩|*

⩽ *∥**µ**δ**∥∥**Au**δ** −**f**∥*+* |⟨**µ*^(*†*)^(* *)*−**µ**δ**, f**⟩| →*0*,*


---

## Page 61

*5.2. SOURCE CONDITION AND CONVERGENCE RATES *61

since* ∥**µ**δ**n**∥**Y**∗*is bounded,* ∥**Au**δ**n** −**f**∥**Y** →*0 and* µ**δ**n** ⇀*^(*∗*)*µ*^(*†*). Combining this with (5.9), we get that *J* (*u*^(*† *)*J* ^() = )D *A*^(*∗*)*µ*^(*†*)*, u*^(*† *)*J *E *.*

Since* A*^(*∗*)*µ*^(*†*)^(* *)*∈**∂**J* (0) and* J* (*u*^(*† *)*J* ^() = )D *A*^(*∗*)*µ*^(*†*)*, u*^(*† *)*J *E , we conclude, using Proposition 4.1.31, that

*A*^(*∗*)*µ*^(*†*)^(* *)*∈**∂**J* (*u*^(*† *)*J* ^().)

So, the source condition is necessary for the boundedness of the dual solutions* µ**δ* as *δ** →*0. It turns out to be also suﬃcient.

**Theorem 5.2.3** (Suﬃcient conditions, [24])**.*** Let** X** and** Y** be Banach spaces and** Y** sep- arable. Let conditions of Theorem 4.2.6 be satisﬁed and** α* =* α*(*δ*)* be chosen as required by Theorem 4.2.7. Suppose that the source condition* (5.8)* is satisﬁed at a** J** -minimising solution** u*^(*† *)*J** *^(*. Then*)^(* µ*)^(*δ*)^(* is bounded uniformly in*)^(* δ*)^(*. Moreover,*)^(* µ*)^(*δ*)^(* ⇀*)^(*∗*)^(*µ*)^(*†*)^(* in*)^(* Y*)^(*∗*)^(*as*)^(* δ*)^(* →*)^(0 )*(perhaps, up to a subsequence), where** µ*^(*†*)^(* *)*is the solution of the dual limit problem* (5.7)* with minimal norm.*

*Proof.* We omit the proof for time reasons. It can be found in [24] (for Hilbert spaces).

The next theorem shows that the source condition (5.8) implies a convergence rates in terms of the Bregman distance.

**Theorem 5.2.4.*** Let the source condition* (5.8)* be satisﬁed at a** J** -minimising solution** u*^(*† *)*J **and let** u**δ** be a regularised solution solving* (5.1)*. Then the following estimate holds*

*D*^(*p*)^(*δ*)^(*,p*)^(*† *)*J *(*u**δ**, u*^(*† *)*J* ^())^( ⩽)^(1)

4*α*

 *δ* +* α**∥**µ*^(*†*)*∥ *2 +* δ**∥**µ*^(*†*)*∥**.*

*where** p**δ* =* A*^(*∗*)*µ**δ** ∈**∂**J* (*u**δ*)* with** µ**δ** as deﬁned in* (5.3)* and** p*^(*†*)^( )=* A*^(*∗*)*µ*^(*†*)^(* *)*∈**∂**J* (*u*^(*† *)*J* ^())^(* is as*)

*deﬁned in* (5.8)*.** D*^(*p*)^(*δ*)^(*,p*)^(*† *)*J *(*u**δ**, u*^(*† *)*J* ^())^(* denotes the symmetric Bregman distance between*)^(* u*)^(*δ*)^(* and *)*u*^(*† *)*J** *^(*. For the optimal choice*)^(* α*)^( = )*δ **∥**µ*^(*†*)*∥*^(*we get that*)

*D*^(*p*)^(*δ*)^(*,p*)^(*† *)*J *(*u**δ**, u*^(*† *)*J* ^())^( ⩽)^(3)^(*δ*)^(*∥*)^(*µ*)^(*†*)^(*∥*)^(*.*)

*Proof.* We start with the following estimate

*αD*^(*p*)^(*δ*)^(*,p*)^(*† *)*J *(*u**δ**, u*^(*† *)*J* ^() )= *α**⟨**p**δ** −**p*^(*†*)*, u**δ** −**u*^(*† *)*J** *^(*⟩*)

= *α**⟨**µ**δ** −**µ*^(*†*)*, Au**δ** −**f**⟩*

= *α**⟨**µ**δ**, Au**δ** −**f**δ**⟩*+* α**⟨**µ**δ**, f**δ** −**f**⟩−**α**⟨**µ*^(*†*)*, Au**δ** −**f**δ**⟩−**α**⟨**µ*^(*†*)*, f**δ** −**f**⟩**.*

From (5.5) we know that

*α**⟨**µ**δ**, Au**δ** −**f**δ**⟩*⩽*−*^(1)

2^(*∥*)^(*Au*)^(*δ*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2 )*Y*^(*.*)

and from (5.4) that* α**∥**µ**δ**∥*=* ∥**Au**δ** −**f**δ**∥*. Using these estimates, the Cauchy-Schwarz inequality and the estimate* ∥**f** −**f**δ**∥*⩽*δ*, we get

*αD*^(*p*)^(*δ*)^(*,p*)^(*† *)*J *(*u**δ**, u*^(*† *)*J* ^() )⩽ *−*^(1)

2^(*∥*)^(*Au*)^(*δ*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2)^( + ) *δ* +* α**∥**µ*^(*†*)*∥ * *∥**Au**δ** −**f**δ**∥*+* αδ**∥**µ*^(*†*)*∥**.*


---

## Page 62

62 *CHAPTER 5. CONVEX DUALITY*

The right-hand side is the following quadratic function of the scalar variable* ∥**Au**δ** −**f**δ**∥*

*ϕ*(*t*) :=* −*^(1)

2^(*t*)^(2)^( + ()^(*δ*)^( +)^(* α*)^(*∥*)^(*µ*)^(*†*)^(*∥*)^())^(*t*)^( +)^(* αδ*)^(*∥*)^(*µ*)^(*†*)^(*∥*)^(*, *)*t** ∈*R*.*

It achieves its maximum at* t*0 = (*δ* +* α**∥**µ*^(*†*)*∥*) and this maximum value is equal to

*ϕ*(*t*0) = ^(()^(*δ*)^( +)^(* α*)^(*∥*)^(*µ*)^(*†*)^(*∥*)^())^(2)

2 +* αδ**∥**µ*^(*†*)*∥**.*

Substituting this into the above estimate for the Bregman distance and dividing both sides by* α*, we get the desired estimate

*D*^(*p*)^(*δ*)^(*,p*)^(*† *)*J *(*u**δ**, u*^(*† *)*J* ^())^( ⩽)^(()^(*δ*)^( +)^(* α*)^(*∥*)^(*µ*)^(*†*)^(*∥*)^())^(2)

2*α *+* δ**∥**µ*^(*†*)*∥**.*

Diﬀerentiating the right-hand side w.r.t.* α* and setting the derivative to zero, we obtain the following optimality condition for* α*

0 = ^(2)^(*α*)^(*∥*)^(*µ*)^(*†*)^(*∥*)^(()^(*δ*)^( +)^(* α*)^(*∥*)^(*µ*)^(*†*)^(*∥*)^())^(* −*)^(()^(*δ*)^( +)^(* α*)^(*∥*)^(*µ*)^(*†*)^(*∥*)^())^(2)

2*α*^(2 )=* *^(*α*)^(2)^(*∥*)^(*µ*)^(*†*)^(*∥*)^(2)^(* −*)^(*δ*)^(2)

2*α*^(2)

and *α* = *δ **∥**µ*^(*†*)*∥*^(*.*)

With this optimal choice of* α* we get the following estimate

*D*^(*p*)^(*δ*)^(*,p*)^(*† *)*J *(*u**δ**, u*^(*† *)*J* ^())^( ⩽)^(3)^(*δ*)^(*∥*)^(*µ*)^(*†*)^(*∥*)^(*.*)

**Remark 5.2.5.** Of course, we do not know* µ*^(*†*)^( )since we don’t know the* J* -minimising solution* u*^(*† *)*J* ^(, but the theorem gives an optimal)^(* rate*)^(* α*)^(* ∼*)^(*δ*)^( for a priori parameter choice rules)

and a corresponding error estimate* D*^(*p*)^(*δ*)^(*,p*)^(*† *)*J *(*u**δ**, u*^(*† *)*J* ^() =)^(* O*)^(()^(*δ*)^().)

Now we will look at two examples involving Total Variation to get a feeling for what the source condition ‘means’.

**Example 5.2.6** (Total Variation)**.** Let Ω*⊂*R^(2)^( )be a bounded domain with a* C*^(*∞*)boundary. Let* X* = BV(Ω) and* Y* =* L*^(2)(Ω) and* J* (*·*) = TV(*·*). Recall the ROF problem

min *u**∈*BV 1 2^(*∥*)^(*Iu*)^(* −*)^(*f*)^(*δ*)^(*∥*)^(2 )*L*^(2)^( +)^(* α*)^( TV()^(*u*)^())^(*,*)

where* I* : BV(Ω)* →**L*^(2)(Ω) is the embedding operator, which is continuous since Ω*⊂*R^(2). The adjoint* I*^(*∗*):* L*^(2)(Ω)* →*BV^(*∗*)(Ω) continuously embeds* L*^(2)^( )into BV^(*∗*). Clearly,* I*^(*∗*)is not surjectuve and* R*(*I*^(*∗*)) =* L*^(2)(Ω). From Example 4.3.6 we know that

TV(**1***C*) = Per(*C*)*,*

where** 1***C* is the indicator function of the set* C*. Denoting by** n***∂**C* the unit normal, we obtain

Per(*C*) = Z

*∂**C *1 = Z

*∂**C **⟨***n***∂**C**,*** n***∂**C**⟩**.*


---

## Page 63

*5.2. SOURCE CONDITION AND CONVERGENCE RATES *63

1

1

*ε*

*ε *0

*C*

*C**ε*

Figure 5.1: Example of a set whose indicator function does not satisfy the source condition.

Since** n***∂**C** ∈**C*^(*∞*)(*∂**C**,* R^(2)) and* ∥***n***∂**C*(*x*)*∥*2 = 1 for any* x*, we can extend** n***∂**C* to a* C*^(*∞ *)0 ^((Ω)^(*,*)^( R)^(2)^() )vector ﬁeld* ψ* with sup*x**∈*Ω*∥**ψ*(*x*)*∥*2 ⩽1. Therefore, using the divergence theorem, we obtain that Z

*∂**C **⟨***n***∂**C**,*** n***∂**C**⟩*= Z

*∂**C **⟨**ψ,*** n***∂**C**⟩*= Z

*C *div* ψ* = Z

Ω **1***C* div* ψ.*

Combining all these equalities, we get that

TV(**1***C*) = Z

Ω **1***C* div* ψ* =* ⟨*div* ψ,*** 1***C**⟩**.*

Taking an arbitrary* u** ∈*BV(Ω), we note that

TV(*u*)* −⟨*div* ψ, u**⟩*= sup *ϕ** ∈**C*^(*∞ *)0 ^((Ω)^(*,*)^( R)^(2)^() )sup*x**∈*Ω*∥**ϕ*(*x*)*∥*2 ⩽1

*⟨*div* ϕ, u**⟩−⟨*div* ψ, u**⟩*⩾0*,*

since* ϕ* =* ψ* is feasible. Therefore, div* ψ** ∈**∂*TV(0) and, since TV(**1***C*) =* ⟨*div* ψ,*** 1***C**⟩*, we also get that div* ψ** ∈**∂*TV(**1***C*)*.*

Since* ψ** ∈**C*^(*∞ *)0 ^((Ω)^(*,*)^( R)^(2)^(), we have div)^(* ψ*)^(* ∈*)^(*C*)^(*∞ *)0 ^((Ω))^(* ⊂*)^(*L*)^(2)^((Ω) =)^(* R*)^(()^(*I*)^(*∗*)^() and the source condition )is satisﬁed at* u* =** 1***C* with* µ*^(*†*)^( )= div* ψ*.

**Example 5.2.7** (Total Variation)**.** In the same setting as in Example 5.2.6, let* C* be a domain with a nonsmooth boundary, e.g., a square* C* = [0*,* 1]^(2). We will show in this example that in this case* ∂*TV(**1***C*)* ∩R*(*I*^(*∗*)) =* ∅*, where* R*(*I*^(*∗*)) =* L*^(2)(Ω) as before, i.e. the source condition fails. Assume that there exists* p*0* ∈**∂*TV(**1***C*)* ∩**L*^(2)(Ω). Then by the results of Example 4.3.6 we have that *⟨**p*0*,*** 1***C**⟩*= TV(**1***C*) = Per(*C*) = 4*.*

Since* p*0 is a subgradient, we get that for any* u** ∈*BV(Ω)

TV(*u*)* −⟨**p*0*, u**⟩*⩾0*.*

Let us cut a triangle* C**ε* of size* ε* from a corner of* C* as shown in Figure 5.1. Then for *u* =** 1***C\C**ε* we get TV(**1***C\C**ε*) ⩾ 
 *p*0*,*** 1***C\C**ε * =* ⟨**p*0*,*** 1***C**⟩−⟨**p*0*,*** 1***C**ε**⟩*

and therefore

*⟨**p*0*,*** 1***C**ε**⟩*⩾TV(**1***C*)*−*TV(**1***C\C**ε*) = Per(*C*)*−*Per(*C \C**ε*) = 4*−*(4*−*2*ε*+ *√*

2*ε*) = (2*− √*

2)*ε >* 0*.*


---

## Page 64

64 *CHAPTER 5. CONVEX DUALITY*

By H¨older’s inequality we get that

*⟨**p*0*,*** 1***C**ε**⟩*= Z

*C**ε **p*0* ·*** 1** ⩽ Z

*C**ε **|**p*0*|*^(2 )1*/*2 Z

*C**ε *1 1*/*2 = 1 *√*

2^(*ε *)Z

*C**ε **|**p*0*|*^(2 )1*/*2 *.*

Combining the last two inequalities, we get

(2* − √*

2)*ε* ⩽*⟨**p*0*,*** 1***C**ε**⟩*⩽ 1 *√*

2^(*ε *)Z

*C**ε **|**p*0*|*^(2 )1*/*2

and therefore Z

*C**ε **|**p*0*|*^(2)^( )⩾2(2* − √*

2)^(2)^(* *)*>* 0

for all* ε >* 0. However, since* p*0* ∈**L*^(2)(Ω) by assumption, we must have Z

*C**ε **|**p*0*|*^(2)^(* *)*→*0 as* ε** →*0*.*

This contradiction proves that such* p*0 does not exist and* ∂*TV(**1***C*)* ∩R*(*I*^(*∗*)) =* ∅*.


---

## Page 65

# **Chapter 6**
# **Bayesian probability and statistics**
### **6.1 From inverse problems to Bayesian inverse problems**

We consider an inverse problem of the form:

Find* u** ∈X* :* A*(*u*) +* n* =* f**n**,*

where* X* is a separable Banach space,* n** ∈Y* is observational noise,* Y* is another separable Banach space,* f**n** ∈Y* is data, and* A* :* X →Y* is a measurable (possibly non-linear) operator. So far, we have studied techniques (pseudo-inverse, regularisation) to ﬁnd estimates for the parameter* u*. In situation where the noise* n* is large or the data is non-informative, we should not only give an estimate for* u*, but also comment on the uncertainty left in the parameter. This is the problem we study in this part of the lecture. There are multiple ways to represent certainty, knowledge, risk, or uncertainty in a parameter, such as* u** ∈X*. Common models are Bayesian probability theory, fuzzy set theory, Dempster–Shafer theory, random set theory,... We follow Bayesian probability theory: model uncertain parameters as random variables.

**Intuitions, concepts, questions, and answers:**

1. Can we use randomness to model deterministic, uncertain objects?

• Not with the usual “frequentist” interpretation of probability. Here, the proba- bility of an event is the limit of the relative frequency of the occurrence of the event in inﬁnitely repeated, independent experiments. If the object we study is deterministic, the frequentist approach will only give us probabilities in* {*0*,* 1*}*.

• Indeed, with the “Bayesian” interpretation of probability. Here the probability of an event is the amount of money (in £) we would give in a game to win £1 if the event occurs. This ‘game’ does not require any inherent randomness.

2. Can we represent the learning of information about a parameter?

• Learning that an event* B* occurred can be represented via conditional probability. Indeed, this learning process is given by the map P(*U** ∈·*)* 7→*P(*U** ∈·|**B*).

• In practice, we can often compute updates of this form through Bayes’ formula.

65


---

## Page 66

66 *CHAPTER 6. BAYESIAN PROBABILITY AND STATISTICS*

3. Can we use Bayesian probability to argue about logical statements?

• Cox’s Theorem [17]: Bayesian probability is a sensible extension of Aristotelian logic.

4. Is Bayesian probability theory congruent with our everyday experience?

• It probably is. See the example below.

**Example 6.1.1.** ‘Tossing a coin’ can be modelled as a Bernoulli experiment

P(Coin shows Head) = 0*.*5 = P(Coin shows Tail)*.*

Actually, this is a mechanical process that is completely deterministic. However, it is diﬃcult to predict its outcome. The model is complicated and subject to many uncertain parameters: force, speed, gravity, air ﬂow. . . Hence, it is easier to model the coin as a random variable.

5. How do we employ these ideas in inverse problems?

(a) We assume that noise* n* and parameter* u* are random variables* N* and* U*. The distributions of* N* and* U* describe our knowledge concerning noise and parameter before observing the data set. The distribution of* U* is called prior distribution *µ*0 := P(*U** ∈·*).

(b) We observe the data set* f**n*, indeed, we observe the occurrence of the event

*{**f**n* =* A*(*U*) +* N**}**.*

(c) We employ Bayes’ theorem to ‘update’ the prior by incorporating the observa- tional data

*µ*0 = P(*U** ∈·*)* 7→*P(*U** ∈·|**f**n* =* A*(*U*) +* N*) =:* µ*post*.*

As* µ*post now explains our knowledge after seeing the data, we call it posterior distribution.

### **6.2 Reminder: measure, probability, and integration**

During this course, we will make extensive use of measure-theoretic probability theory. Thus, we will brieﬂy remind ourselves of some deﬁnitions, examples, and results from mea- sure and probability theory that we will require throughout this lecture. In case the reader would like to get a more thorough reminder, we refer them to [6], [10], [25]. We commence with* σ*-algebras.

**Deﬁnition 6.2.1** (*σ*-algebra)**.*** Let* Ω*be a non-empty set, let* 2^(Ω):=* {**A* :* A** ⊆*Ω*}** be the power set of* Ω*, and let** F ⊆*2^(Ω)*satisfy (i)-(iii):*

*(i)* Ω*∈F**,*

*(ii) for any** F** ∈F**, we have also** F** *^(*c*)^( ):= Ω*\**F** ∈F**, and*

*(iii) for any countable family* (*F**n* :* n** ∈*N)* ∈F*^(N)*, we have also* ^(S )*n**∈*N* *^(*F*)^(*n*)^(* ∈F*)^(*.*)


---

## Page 67

*6.2. REMINDER: MEASURE, PROBABILITY, AND INTEGRATION *67

*Then,** F** is called** σ**-algebra on* Ω*and* (Ω*,** F*)* is called measurable space.*

There are several ways to construct* σ*-algebras. They can for instance be induced by systems of sets or functions.

**Deﬁnition 6.2.2** (Induced* σ*-algebra)**. ***1. Let* Ω*be non-empty and** E ⊆*2^(Ω)*. We deﬁne the** σ**-algebra induced by** E** on* Ω*by*

*σ*Ω(*E*) := \

*F*^(*′*)*⊃E **F*^(*′*)^(* *)*is** σ**-algebra on* Ω

*F*^(*′*)*.*

*2. Let* Ω*be non-empty, let* (Ω^(*′*)*,** F*^(*′*))* be a measurable space, and let** g* : Ω*→*Ω^(*′*)^(* *)*be a function. We deﬁne the** σ**-algebra induced by** g** on* Ω*by*

*σ*Ω(*g*) :=* {{**g** ∈**F** *^(*′*)*}* :* F** *^(*′*)^(* *)*∈F*^(*′*)*}**,*

*where **{**g** ∈**F** *^(*′*)*}* :=* g*^(*−*)^(1)(*F** *^(*′*)) :=* {**ω** ∈*Ω:* g*(*ω*)* ∈**F** *^(*′*)*}*

*is the pre-image of** F** *^(*′*)^(* *)*under** g**.*

**Example 6.2.1.** Let Ωbe a non-empty set.

1. 2^(Ω)is the largest* σ*-algebra on Ω.* {∅**,* Ω*}* is the smallest* σ*-algebra.

2. Let Ωbe a topological space with open sets* O** ⊆*2^(Ω). The* σ*-algebra* σ*Ω(*O*) =:* B*Ωis called Borel-*σ*-algebra on Ω.

A* σ*-algebra is the natural space to deﬁne a (probability) measure on.

**Deﬁnition 6.2.3** (Measure and probability measure)**.*** Let* (Ω*,** F*)* be a measurable space and let** µ* :* F →*[0*,** ∞*]* be a function, satisfying (i),(ii):*

*(i)** µ*(*∅*) = 0*,*

*(ii) for any countable family* (*F**m* :* m** ∈*N)* ∈F*^(N)^(* *)*of mutually disjoint sets, i.e.** F**n**∩**F**m* =* ∅ *(*n** ̸*=* m*)*. Then, we have** µ *�S *m**∈*N* *^(*F*)^(*m *) = ^(P )*m**∈*N* *^(*µ*)^(()^(*F*)^(*m*)^())^(*.*)

*Then,** µ** is called measure on* (Ω*,** F*)* and* (Ω*,** F**, µ*)* is called measure space. If a measure** µ **additionally satisﬁes (iii):*

*(iii)** µ*(Ω) = 1*,*

*the measure** µ** is called probability measure and* (Ω*,** F**, µ*)* is called probability space. Finally, a measure** µ** is called** σ**-ﬁnite, if*

*(iv) there is a countable family* (*F**m* :* m** ∈*N)* ∈F*^(N)*, with* ^(S )*m**∈*N* *^(*F*)^(*m*)^( = Ω)^(*and*)^(* µ*)^(()^(*F*)^(*m*)^())^(* <*)^(* ∞ *)(*m** ∈*N)*.*

**Example 6.2.2.** Let (Ω*,** F*) be some measurable space.

• # :* F →*[0*,** ∞*] deﬁned by

#(*F*) :=

( *∞**, *if* F* is inﬁnite *|**F**|**, *otherwise. (*F** ∈F*)

is a measure and called counting measure,


---

## Page 68

68 *CHAPTER 6. BAYESIAN PROBABILITY AND STATISTICS*

• Let* ω** ∈*Ω. Then,* δ*(*· −**ω*) :* F →*[0*,** ∞*] deﬁned by

*δ*(*F** −**ω*) :=

( 1*, *if* F** ∋**ω *0*, *otherwise (*F** ∈F*)

is called Dirac measure concentrated in* ω*. The Dirac measure is a probability mea- sure.

• Let* k** ∈*N, Ω:= R^(*k*), and* λ**k* :* B*R^(*k*)^(* *)*→*[0*,** ∞*] be the unique measure that satisﬁes

*λ**k*

 *k *Y

*i*=1 [*a**i**, b**i*)

!

=

*k *Y

*i*=1 (*b**i** −**a**i*)*,*

if* a**i* ⩽*b**i* (*i* = 1*, ..., k*). Then* λ**k* is called* k*-dimensional Lebesgue measure.

**Exercise 6.2.4. **1. Show that the Dirac and counting measure are measures.

2. Show that Dirac and Lebesgue measure are* σ*-ﬁnite.

3. When is the counting measure* σ*-ﬁnite?

We already learned the concept of using a function to construct a* σ*-algebra. In the following, we would like to use functions to represent uncertainties (‘random variables’) and use measures to integrate functions. Here, we require the concept of ‘measurability’.

**Deﬁnition 6.2.5.*** Let* (Ω*,** F*)* and* (Ω^(*′*)*,** F*^(*′*))* be two measurable spaces and let** g* : Ω*→*Ω^(*′*)^(* *)*be a function.*

*1.** g** is called measurable, if** {**g** ∈**F** *^(*′*)*} ∈F**,** for any** F** *^(*′*)^(* *)*∈F*^(*′*)*. In this case, we sometimes write** g* : (Ω*,** F*)* →*(Ω^(*′*)*,** F*^(*′*))*.*

*2. Let** g** be measurable and** µ** be a measure on* (Ω*,** F*)*. Then, we deﬁne the push-forward*

*measure** µ*(*g** ∈·*)*. If in addition,** µ** is a probability measure,** g** is called random variable and** µ*(*g** ∈·*)* is called (probability) distribution of** g**.*

This rather abstract deﬁnition of measurability does not appear to be very instructive in practice. A useful result is the following proposition

**Proposition 6.2.6.*** Let* Ω*be a topological space and** g* : Ω*→*R* be continuous, i.e. for any open** F** *^(*′*)^(* *)*⊆*R*, the preimage** {**g** ∈**F** *^(*′*)*} ⊆*Ω*is open as well. Then,** g* : (Ω*,** B*Ω)* →*(R*,** B*R)* is measurable.*

*Proof.* Page 36 in [6].

Push-forward measures and probability distributions are well-deﬁned measures and probability measures, respectively.

**Proposition 6.2.7.*** Let* (Ω*,** F**, µ*)* be a measure space,* (Ω^(*′*)*,** F*^(*′*))* be a measurable spaces, and let** g* : (Ω*,** F*)* →*(Ω^(*′*)*,** F*^(*′*))* be a measurable function. Then, the pushforward measure** µ*(*g** ∈·*) *is a measure on* (Ω^(*′*)*,** F*^(*′*))*. Moreover, if** µ** is a probability measure, then so is** µ*(*g** ∈·*)*.*

*Proof.* Exercise.


---

## Page 69

*6.2. REMINDER: MEASURE, PROBABILITY, AND INTEGRATION *69

Measurability is the basic concept needed to be able to integrate a function with respect to a measure. We start with simple functions.

**Deﬁnition 6.2.8.*** Let* (Ω*,** F**, µ*)* be a measure space. A function** g* : Ω*→*R* is called simple, if there exists an** m** ∈*N* and* (*F**i* :* i* = 1*, ..., m*)* ∈F*^(*m*)*, such that*

*g* =

*m *X

*i*=1 *b**i***1***F**i**,*

*for some** b** ∈*R^(*m*)*. Consider the following two assumptions:*

*(i)** b** ∈*[0*,** ∞*)^(*m*)^(* *)*or** b** ∈*(*−∞**,* 0]^(*m*)*,*

*(ii) for any** i** ∈{*1*, ..., m**}**, with** µ*(*F**i*) =* ∞**, we have** b**i* = 0*.*

*If either (i) or (ii) holds, we deﬁne the (Lebesgue) integral of** g** with respect to** µ** by*

Z

Ω *g*d*µ* := Z

Ω *g*(*ω*)d*µ*(*ω*) := Z

Ω *g*(*ω*)*µ*(d*ω*) :=

*m *X

*i*=1;*b**i**̸*=0 *b**i**µ*(*F**i*)*.*

*If the expression on the right-hand side is ﬁnite, we call** g** (Lebesgue) integrable.*

**Exercise 6.2.9.** A simple function* g* : Ω*→*R is measurable from (Ω*,** F*) to (R*,** B*R).

To deﬁne the integral for more general functions* g*, we will approximate the function by simple functions. This gives us the following deﬁnition for the integral.

**Deﬁnition 6.2.10** (Lebesgue integral)**.*** Let* (Ω*,** F**, µ*)* be a measure space and let** g* : (Ω*,** F*)* → *(R*,** B*R)* be measurable and non-negative. Then, we deﬁne the (Lebesgue) integral of** g** by*

Z

Ω *g*d*µ* := sup Z

Ω *h*(*ω*)d*µ*(*ω*) : 0 ⩽*h* ⩽*g, h** is simple *

*If the supremum is ﬁnite, we call** g** (Lebesgue) integrable.*

In the following proposition, we discuss the fundamental properties of the Lebesgue integral: linearity, monotonicity, and monotonic convergence.

**Proposition 6.2.11.*** Let* (Ω*,** F**, µ*)* be a measure space and let** g, h, g*1*, g*2*, . . .* : (Ω*,** F*)* → *(R*,** B*R)* be measurable, non-negative functions. Then:*

*1. If** g* ⩽*h** pointwise, then *R

Ω^(*g*)^(d)^(*µ*)^( ⩽ )R

Ω^(*h*)^(d)^(*µ.*)

*2. If* (*g**m* :* m** ∈*N)* is pointwise increasing and* lim*m**→∞**g**m* =* g** pointwise, then the sequence *�R

Ω^(*g*)^(*m*)^(d)^(*µ*)^( :)^(* m*)^(* ∈*)^(N ) *is increasing and* lim*m**→∞ *R

Ω^(*g*)^(*m*)^(d)^(*µ*)^( = )R

Ω^(*g*)^(d)^(*µ*)^(*.*)

*3. For some** α, β** ∈*[0*,** ∞*]*, we have *Z

Ω *αg* +* βh*d*µ* =* α *Z

Ω *g*d*µ* +* β *Z

Ω *h*d*µ.*

*(We use the convention* “0* · ∞*= 0”*)*

*Proof.* Lemma 4.6 in [25].


---

## Page 70

70 *CHAPTER 6. BAYESIAN PROBABILITY AND STATISTICS*

Measurable functions* g* taking values in R can be integrated by subtracting the integral of their negative part max*{*0*,** −**g**}* from the integral of their positive part max*{*0*, g**}*, if one of them is integrable. Integrals of non-negative measurable functions give a natural way to deﬁne measures.

**Proposition and deﬁnition 6.2.12.*** Let* (Ω*,** F**, µ*)* be a measure space and let** g* : (Ω*,** F*)* → *(R*,** B*R)* be measurable and non-negative. Then, the map** ν* :* F →*[0*,** ∞*]*, deﬁned by*

*F** 7→ *Z

Ω *g** ·*** 1***F* d*µ* =: Z

*F **g*d*µ*

*is a measure.** ν** is called measure with (**µ**-)density (function)** g**. If** ν** is a probability measure, **g** is called (**µ**-)probability density (function).*

*Proof.* Exercise.

**Deﬁnition 6.2.13.*** Let* (Ω*,** F**, µ*) := (R*,** B*R*, λ*1)*. Moreover, let** m** ∈*R* and** σ >* 0*, and let **g* : Ω*→*R* be the measurable function*

*g*(*ω*) := 1 *√*

2*πσ* ^(exp ) *−*^(()^(*ω*)^(* −*)^(*m*)^())^(2)

2*σ*^(2)

 *.*

*Then, the measure** ν** with** λ*1*-density** g** is called Gaussian distribution on* R* with mean** m **and variance** σ*^(2)*. We denote* n(*·*;* m, σ*^(2)) :=* g** and* N(*m, σ*^(2)) :=* ν**. Moreover, we deﬁne the degenerate Gaussian distribution by* N(*m,* 0) :=* δ*(*· −**m*)*.*

A rather surprising result about measures and densities is the Radon–Nikodym Theorem. It is fundamental for the general deﬁnition of conditional expectations and also for the general form of Bayes’ theorem. Before stating the Radon–Nikodym Theorem, we deﬁne two more important notions regarding measures.

**Deﬁnition 6.2.14.*** Let* (Ω*,** F*)* be a measurable space and** µ, ν** be two measure on that space.*

*1. We deﬁne** ν** to be absolutely continuous with respect to** µ**, if for all** F** ∈F**, with **µ*(*F*) = 0*, we also have** ν*(*F*) = 0*. In this case, we write** ν** ≪**µ**.*

*2. Let** A*(*ω*)* be a statement for all** ω** ∈*Ω*. We say that** A** holds** µ**-almost everywhere*

*(**µ**-a.e.), if there is a set** N** ∈F** such that** µ*(*N*) = 0* and** A*(*ω*)* is true for** ω** ∈**X**\**N**. If** µ** is a probability measure, we sometimes say** µ**-almost surely (**µ**-a.s.) instead of **µ**-almost everywhere.*

**Theorem 6.2.15** (Radon-Nikodym)**.*** Let* (Ω*,** F*)* be a measurable space and let** µ, ν** be** σ**- ﬁnite measures on* (Ω*,** F*)*. Then, the following two statements are equivalent:*

*(i)** ν** ≪**µ*

*(ii) There is a measurable function** g* : (Ω*,** F*)* →*(R*,** B*R)*, with*

*ν*(*F*) = Z

*F **g*d*µ *(*F** ∈F*)*.*

*Moreover, the function** g** is** µ**-a.e. unique, called Radon–Nikodym derivative, and denoted by* ^(d)^(*ν*)

d*µ* ^(:=)^(* g*)^(*.*)

*Proof.* (ii)* ⇒*(i): exercise. (i)* ⇒*(ii): more complicated, see, e.g., Corollary 7.34 in [25].

**Exercise 6.2.16.** Give an example for measures* ν, µ* on (R*,** B*R), with* ν** ≪**µ* and* µ* not *σ*-ﬁnite, such that no Radon-Nikodym derivative exists.


---

## Page 71

*6.3. CONDITIONAL PROBABILITY *71

### **6.3 Conditional probability**

For the remainder of the lecture, we always consider (Ω*,** F**,* P) as underlying probability space for any random variable. We typically omit its precise construction, but assume that Ωis a Polish space (separable and completely metrisable) and* F* :=* B*Ω. We denote integrals with respect to P sometimes by

E[*ϕ*] := Z

Ω *ϕ*dP*,*

for some* ϕ* : (Ω*,** F*)* →*(R*,** B*R), for which this integral is well-deﬁned.

**Example 6.3.1.** Let* U* : (Ω*,** F*)* →*(*{*1*, . . . ,* 6*}**,* 2^(*{*)^(1)^(*,...,*)^(6)^(*}*)) be a random variable modelling the roll of a die, hence

P(*U* =* u*) =

( 1*/*6*, *if* u** ∈{*1*, ...,* 6*}**, *0*, *otherwise*.*

This probability measure models our knowledge concerning the outcome of the experiment. Now we consider an extended model. After the die is rolled and before its realisation is revealed, we are told whether the realisation is even or odd. Given this information, we can adjust our knowledge concerning the random variable* U*:

P(*U* =* u**|**U* is even) = ^(P)^(()^(*U*)^( =)^(* u*)^( and)^(* U*)^( is even))

P(*U* is even) *,*

respectively

P(*U* =* u**|**U* is odd) = ^(P)^(()^(*U*)^( =)^(* u*)^( and)^(* U*)^( is odd))

P(*U* is odd) *.*

In the example above, we used the elementary deﬁnition of conditional probabilities:

P(*F**|**F** *^(*′*)) = ^(P)^(()^(*F*)^(* ∩*)^(*F*)^(* ′*)^())

P(*F** *^(*′*)) (*F, F** *^(*′*)^(* *)*∈F**,* P(*F** *^(*′*))* >* 0)*.*

This deﬁnition can only be used, if the event with respect to which the conditional proba- bility is deﬁned has a positive probability (here:* {**U* is even*}**,** {**U* is odd*}*). This however is typically not the case in a Bayesian inverse problem since the probability measure of the noise is continuous. Hence, we need a more general deﬁnition of conditional probabilities. We start with conditional expectations.

**Theorem 6.3.2.*** Let** U* : (Ω*,** F*)* →*(R*,** B*R)* and** Y* : (Ω*,** F*)* →*(*Y**,** BY*)* be random variables and let** U** be integrable. Then, there exists a measurable function** h* : (*Y**,** BY*)* →*(R*,** B*R)*, such that *Z

*F **h*(*y*)P(*Y** ∈*d*y*) = Z

*{**Y** ∈**F**} **U*dP (*F** ∈BY*)*. *(6.1)

*Moreover,** h** is* P(*Y** ∈·*)*-a.s. unique.*

*Proof.* We assume without loss of generality that* U* ⩾0. (If* U* is real-valued, study max*{**U,* 0*}* and max*{−**U,* 0*}* separately.) Note that the map

*F** 7→ *Z

*{**Y** ∈**F**} **U*dP =:* µ*(*F*)


---

## Page 72

72 *CHAPTER 6. BAYESIAN PROBABILITY AND STATISTICS*

deﬁnes a (*σ*-)ﬁnite measure. We now show that* µ** ≪*P(*Y** ∈·*): let* F*0* ∈BY* be chosen such that P(*Y** ∈**F*0) = 0. Then, Z

*{**Y** ∈**F*0*} **U*dP = Z

Ω **1***{**Y** ∈**F*0*}**U*dP = 0*.*

By the Radon–Nikodym Theorem, there exists a P(*Y** ∈·*)-a.s. unique function* h* := d*µ *dP(*Y** ∈·*)^(, )satisfying (6.1).

**Deﬁnition 6.3.3.*** h*(*y*)* in Theorem 6.3.2 is called conditional expectation of** U** given** Y* =* y**. We write** h*(*y*) =: E[*U**|**Y* =* y*]*, for* P(*Y** ∈·*)*-almost every** y** ∈Y**.*

Now we can deﬁne the conditional probability of some event* F* by considering the indicator random variable* U* =** 1***F* . Since* X**,** Y* are Polish spaces, one can even ﬁnd a P(*Y** ∈·*)-a.s. unique Markov kernel (*y, F*)* 7→*E[**1***F** |**Y* =* y*].

**Deﬁnition 6.3.4.*** Let* (Ω*,** F*)*,* (Ω^(*′*)*,** F*^(*′*))* be measurable spaces. A map** M* : Ω*× F*^(*′*)^(* *)*→*[0*,* 1]* is called Markov kernel from* (Ω*,** F*)* to* (Ω^(*′*)*,** F*^(*′*))*, if*

*(i)** M*(*ω,** ·*)* is a probability measure for all** ω** ∈*Ω*,*

*(ii)** M*(*·**, F** *^(*′*)) : (Ω*,** F*)* →*([0*,* 1]*,** B*[0*,* 1])* is measurable for all** F** *^(*′*)^(* *)*∈F*^(*′*)*.*

**Theorem 6.3.5.*** Let** U* : (Ω*,** F*)* →*(*X**,** BX*)* and** Y* : (Ω*,** F*)* →*(*Y**,** BY*)* be random variables. Then, there exist a Markov kernel** M** from* (*Y**,** BY*)* to* (*X**,** BX*)*, with *Z

*F **M*(*y, F** *^(*′*))P(*Y** ∈*d*y*) = P(*{**Y** ∈**F**} ∩{**U** ∈**F** *^(*′*)*}*) (*F** ∈BY**, F** *^(*′*)^(* *)*∈BX*)*.*

*Moreover,** M** is* P(*Y** ∈·*)*-a.s. unique.*

*Proof.* Non-trivial, but possible if Ωis Polish; see [26].

**Deﬁnition 6.3.6.*** M** in Theorem 6.3.5 is called (regular) conditional probability distribu- tion of** U** given** Y* =* y**. We write** M*(*y, F*) := P(*U** ∈**F**|**Y* =* y*)*,** for** F** ∈BX**, y** ∈Y**.*

**Example 6.3.7** (Example 6.3.1 rev.)**.** In Example 6.3.1, we compute the conditional proba- bility distribution of a die* U* : (Ω*,** F*)* →*(*{*1*, . . . ,* 6*}**,* 2^(*{*)^(1)^(*,...,*)^(6)^(*}*)), given the information whether the outcome will be even or odd. Deﬁne a random variable* Y* : (Ω*,** F*)* →*(*{*0*,* 1*}**,* 2^(*{*)^(0)^(*,*)^(1)^(*}*))

*ω** 7→*

( 0*, *if* U*(*ω*) is even 1*, *otherwise.

We can write

P(*U* =* u**|**U* is even) =: P(*U* =* u**|**Y* = 0)*, *P(*U* =* u**|**U* is odd) =: P(*U* =* u**|**Y* = 1)*.*

Indeed, one can show that these functions are conditional expectation/probability measures in the sense of deﬁnition 6.3.3. Let* F** ∈*2^(*{*)^(0)^(*,*)^(1)^(*}*). We need to show that Z

*F *P(*U* =* u**|**Y* =* y*)P(*Y** ∈*d*y*) = P(*{**U* =* u**} ∩{**Y** ∈**F**}*)*.*


---

## Page 73

*6.3. CONDITIONAL PROBABILITY *73

Let* F* :=* {*0*}*. Then, we have Z

*{**Y* =0*} ***1***{**U*=*u**}*dP = ^(1)

6^(()^(**1**)^(*{*)^(2)^(*}*)^(()^(*u*)^() +)^(** 1**)^(*{*)^(4)^(*}*)^(()^(*u*)^() +)^(** 1**)^(*{*)^(6)^(*}*)^(()^(*u*)^()))

= 1 2 |{z} =P(*Y* =0)

*·* ^(1 )3^(()^(**1**)^(*{*)^(2)^(*}*)^(()^(*u*)^() +)^(** 1**)^(*{*)^(4)^(*}*)^(()^(*u*)^() +)^(** 1**)^(*{*)^(6)^(*}*)^(()^(*u*)^()) )| {z } =P(*U*=*u**|**Y* =0)

= Z

*{*0*} *P(*U* =* u**|**Y* =* y*)P(*Y** ∈*d*y*)

Analogously, one can show condition (6.1) for* F* =* ∅**,** {*1*}**,** {*0*,* 1*}*.

In Theorem 6.3.5, we discuss that conditional probabilities are Markov kernels. Also the converse is true: given a Markov kernel, we can construct random variables such that the Markov kernel represents a conditional probability measure.

**Proposition 6.3.8.*** Let** M* : Ω^(*′*)^(* *)*×F*^(*′′*)^(* *)*→*[0*,* 1]* be a Markov kernel from* (Ω^(*′*)*,** F*^(*′*))* to* (Ω^(*′′*)*,** F*^(*′′*))*. Then, there is an underlying probability space* (Ω*,** F**,* P)* and random variables** X*^(*′*)^( ): Ω*→*Ω^(*′*)

*and** X*^(*′′*)^( ): Ω*→*Ω^(*′′*)^(* *)*such that:*

*M*(*ω*^(*′*)*, F** *^(*′′*)) = P(*X*^(*′′*)^(* *)*∈**F** *^(*′′*)*|**X*^(*′*)^( )=* ω*^(*′*)) (*F** *^(*′′*)^(* *)*∈F*^(*′′*)^(* *)*and* P(*X*^(*′*)^(* *)*∈·*)*-almost all** ω*^(*′*)^(* *)*∈*Ω^(*′*))*.*

*Proof.* Deﬁne (Ω*,** F*) := (Ω^(*′*)^(* *)*×*Ω^(*′′*)*,** F*^(*′*)^(* *)*⊗F*^(*′′*)). Let* µ*^(*′*)^( )be some probability measure on (Ω^(*′*)*,** F*^(*′*)). Moreover, let P be the measure satisfying

P(*F** *^(*′*)^(* *)*×** F** *^(*′′*)) = Z

*F** *^(*′*)^(* M*)^(()^(*ω*)^(*′*)^(*, F*)^(* ′′*)^()d)^(*µ*)^(*′*)^(()^(*ω*)^(*′*)^() ()^(*F*)^(* ′*)^(* ∈F*)^(*′*)^(*, F*)^(* ′′*)^(* ∈F*)^(*′′*)^())^(*.*)

Let* X*^(*′*)^( ): Ω*→*Ω^(*′*)^( )(resp.* X*^(*′′*)^( ): Ω*→*Ω^(*′′*)) be the canonical projection on the ﬁrst (resp. second) coordinate. Then* X*^(*′*)^(* *)*∼**µ*^(*′*)^( )and* X*^(*′′*)^(* *)*∼**M*(*X*^(*′*)*,** ·*). Let* F** *^(*′*)^(* *)*∈F*^(*′*)^( )and* F** *^(*′′*)^(* *)*∈F*^(*′′*). Then it holds

P(*{**X*^(*′*)^(* *)*∈**F** *^(*′*)*} ∩{**X*^(*′′*)^(* *)*∈**F** *^(*′′*)*}*) = Z

*{**X*^(*′*)*∈**F** *^(*′*)*,X*^(*′′*)*∈**F** *^(*′′*)*} *dP = ZZ

*{**X*^(*′*)*∈**F** *^(*′*)*,X*^(*′′*)*∈**F** *^(*′′*)*} **M*(*ω*^(*′*)*,* d*ω*^(*′′*))*µ*^(*′*)(d*ω*^(*′*))

(*∗*) = Z

*F** *^(*′*)

Z

*F** *^(*′′*)^(* M*)^(()^(*ω*)^(*′*)^(*,*)^( d)^(*ω*)^(*′′*)^())^(*µ*)^(*′*)^((d)^(*ω*)^(*′*)^() = )Z

*F** *^(*′*)^(* M*)^(()^(*F*)^(* ′′*)^(*, ω*)^(*′*)^())^(P)^(()^(*X*)^(*′*)^(* ∈*)^(d)^(*ω*)^(*′*)^())^(*,*)

where (*) is implied by Tonelli’s Theorem. Hence,* M*(*F** *^(*′′*)*, ω*^(*′*)) = P(*X*^(*′′*)^(* *)*∈**F** *^(*′′*)*|**X*^(*′*)^( )=* ω*^(*′*)) is indeed a conditional probability distribution.

As Markov kernels are consistent with conditional probabilities, we sometimes write *M*(*·|∗*) :=* M*(*∗**,** ·*). Applying the concept of conditional expectations in general situations is not straight- forward. However, probability measures are often given in terms of probability density functions. Given joint and marginal probability density functions, one can deﬁne the con- ditional probability in terms of a probability density function.

**Lemma 6.3.9.*** Let** U, Y** be random variables with joint probability distribution* P((*U, Y* )* ∈·*) *that is absolutely continuous with respect to a** σ**-ﬁnite measure** ν** on* (*X × Y**,** BX ⊗BY*)*. Assume that** ν* =* ν**U** ⊗**ν**Y** for** σ**-ﬁnite measure spaces* (*X**,** BX**, ν**U*)*,* (*Y**,** BY**, ν**Y* )*. We write **g**U,Y* := ^(d)^(P)^((()^(*U,Y*)^( ))^(*∈·*)^())

d*ν **for the joint probability density function. Then,*

P(*U** ∈·*)* ≪**ν**U**, *P(*Y** ∈·*)* ≪**ν**Y** ,*


---

## Page 74

74 *CHAPTER 6. BAYESIAN PROBABILITY AND STATISTICS*

*with probability density functions*

*g**U* := Z

*Y **g**U,Y* d*ν**Y* = ^(d)^(P)^(()^(*U*)^(* ∈·*)^())

d*ν**U *(*ν**U**-a.e.*)*,*

*g**Y* := Z

*X **g**U,Y* d*ν**U* = ^(d)^(P)^(()^(*Y*)^(* ∈·*)^())

d*ν**Y *(*ν**Y** -a.e.*)*.*

*Proof.* Let* A** ∈BX*. We have

P(*U** ∈**A*) = P(*U** ∈**A, Y** ∈Y*) = Z

*A**×Y **g**U,Y* d*ν* = Z

*A*

Z

*Y **g**U,Y* d*ν**Y *| {z } =:*g**U*

d*ν**U**,*

by the Theorem of Tonelli. Hence, P(*U** ∈·*)* ≪**ν**U*. The statement about* Y* can be proven analoguously.

**Theorem 6.3.10.*** Under the assumptions of Lemma 6.3.9, we have* P(*U** ∈·|**Y* =* y*)* ≪**ν**U **with** ν**U**-density:*

*g**U**|**Y* =*y*(*u*) :=

(*g**U,Y* (*u,y*)

*g**Y* (*y*)* *^(*, *)*if** g**Y* (*y*)* >* 0*,*

0*, **otherwise *(*u** ∈X**, ν**U**-a.e.*;* y** ∈Y**,* P(*Y** ∈·*)*-a.e.*)*,*

*and equivalently* P(*Y** ∈·|**U* =* u*)* ≪**ν**Y** with** ν**Y** -density:*

*g**Y** |**U*=*u*(*y*) :=

(*g**U,Y* (*u,y*)

*g**U*(*u*)* *^(*, *)*if** g**U*(*u*)* >* 0*,*

0*, **otherwise *(*y** ∈Y**, ν**Y** -a.e.*;* u** ∈X**,* P(*U** ∈·*)*-a.e.*)*.*

*Proof.* Let* A** ∈BX**, F** ∈BY*. By Deﬁnition 6.3.6, P(*U** ∈**A**|**Y* =* y*) fulﬁlls (6.1):

P(*U** ∈**A, Y** ∈**F*) = Z

*F *P(*U** ∈**A**|**Y* =* y*)P(*Y** ∈*d*y*)

= Z

*F *P(*U** ∈**A**|**Y* =* y*)*g**Y* (*y*)d*ν**Y* (*y*)

= Z

*F**∩{**g**Y** >*0*} *P(*U** ∈**A**|**Y* =* y*)*g**Y* (*y*)d*ν**Y* (*y*)*,*

as P(*g**Y* (*Y* ) = 0) = P(*Y** ∈{**g**Y* = 0*}*) = R

*Y*** **^(**1**)^(*{*)^(*g*)*Y* ^(=0)^(*}*)^(P)^(()^(*Y*)^(* ∈*)^(d)^(*y*)^() = )R

*{**g**Y* =0*}** *^(*g*)^(*Y*)^( d)^(*ν*)^(*Y*)^( = 0)^(*.*)^( Note )that we can write

P(*U** ∈**A, Y** ∈**F*) = Z

*F**∩{**g**Y** >*0*}*

Z

*A **g**U,Y* (*u, y*)d*ν**U*(*u*)d*ν**Y* (*y*)*.*

This and the statement above imply

P(*U** ∈**A**|**Y* =* y*)*g**Y* (*y*) = Z

*A **g**U,Y* (*u, y*)d*ν**U*(*u*) (P(*Y** ∈·*)-a.s.)*.*

Hence, we have

P(*U** ∈**A**|**Y* =* y*) = Z

*A*

*g**U,Y* (*u, y*)

*g**Y* (*y*) d*ν**U*(*u*) (P(*Y** ∈·*)-a.s.)*.*

This proves our statement about P(*U** ∈·|**Y* =* y*) the reverse statement can be shown analoguously.


---

## Page 75

*6.4. BAYESIAN STATISTICS *75

**Deﬁnition 6.3.11.*** Let** g**U**, g**Y** , g**U,Y** , g**U**|**Y* =*y**, g**Y** |**U*=*u** be the probability density functions in Theorem 6.3.10. We deﬁne*

•* g**U** (resp.** g**Y** ) to be the marginal probability density of** U** (resp.** Y** ),*

•* g**U,Y** to be the joint probability density of** U** and** Y** ,*

•* g**U**|**Y* =*y** to be the conditional density of** U** given** Y* =* y**, and*

•* g**Y** |**U*=*u** to be the conditional density of** Y** given** U* =* u**.*

### **6.4 Bayesian statistics**

We are now ready to, ﬁrst, ﬁt our inverse problem into a statistical framework and, second, determine the posterior measure

**6.4.1 Statistical models**

**Deﬁnition 6.4.1.*** Let** X**,** Y** be separable Banach spaces. We refer to** X** as parameter space and to** Y** as data space. Let** P* :=* {**M*(*·|**u*) :* u** ∈X}**, where** M** is a Markov kernel from *(*X**,** BX*)* to* (*Y**,** BY*)*. The tuple* (*Y**,** P*)* is called statistical model. The statistical model is called parametric, if** X** is a subset of a Euclidean vector space, and non-parametric, otherwise.*

After deﬁning statistical models, we should comment on their purpose.

**Remark 6.4.2.** Let* u*^(*∗*)*∈X* be some parameter, let* Y** ∼**M*(*·|**u*^(*∗*)), and let* y* be a realisation of* Y* . Statistical methods aim to ﬁnd* u*^(*∗*)*∈X* based on the realisation* y*. The probability measure* M*(*·|**u*^(*∗*)) is called data-generating distribution.

Now, we give an example for a parametric statistical model.

**Example 6.4.3.** We are given ﬁve independent realisations* y* = (0*.*2*,** −*0*.*32*,* 0*.*8*,* 1*.*2*,** −*0*.*4), of a one dimensional Gaussian random variable with variance* σ*^(2)^( )= 1. We do not know the mean of the random variable. Given* y*, we want to identify the mean. The statistical model associated with this task is given by:

(*Y**,** P*) := � R^(5)*, * N(*u,* 1)^(*⊗*)^(5)^( ):* u** ∈*R 	 *.*

We can sometimes represent a statistical model in terms of a conditional density, the so-called likelihood.

**Deﬁnition 6.4.4.*** Let* (*Y**,** P*)* be a statistical model and let** L* : (*X ×Y**,** BX ⊗BY*)* →*(R*,** B*R) *such that*

*P* :=  *BY ∋**F** 7→ *Z

*F **L*(*y**|**u*)d*µ*(*y*) :* u** ∈X * *,*

*for some measure** µ** on* (*Y**,** BY*)*. We refer to** L** as (data) likelihood.*

Note that the likelihood is a conditional density* L* =* g**Y** |**U*=*u*, for some random variable *U*. It informs us about the likelihood of observing a data set given that we assume it was sampled from* M*(*·|**u*).


---

## Page 76

76 *CHAPTER 6. BAYESIAN PROBABILITY AND STATISTICS*

**Example 6.4.5.** Let* A* : (*X**,** BX*)* →*(*Y**,** BY*) be a measurable operator. Moreover, let *µ*noise be a probability measure on (*Y**,** BY*). We consider the inverse problem of identifying *u** ∈**X*, where *A*(*u*) +* N* =* f**n*

with* N** ∼**µ*noise. We can now represent this inverse problem by a statistical model:

(*Y**,** P*) := (*Y**,** {**µ*noise(*· −A*(*u*)) :* u** ∈X}*)*.*

The data set* f**n* is a realisation of the data-generating distribution* µ*noise(*· −A*(*u*^(*∗*))), where *u*^(*∗*)is the true parameter. Let* n** ∈*N,* Y* := R^(*n*), Γ* ∈*R^(*n*)^(*×*)^(*n*)^( )be positive deﬁnite, and* µ*noise := N(0*,* Γ). Then, we can represent the statistical model by a likelihood:

*L*(*y**|**u*) := (2*π*)^(*−*)^(*k/*)^(2)det(Γ)^(*−*)^(1)^(*/*)^(2)^( )exp  *−*^(1)

2^(*∥*)^(Γ)^(*−*)^(1)^(*/*)^(2)^(()^(*y*)^(* −A*)^(()^(*u*)^()))^(*∥*)^(2 ) *,*

where* u** ∈X* and* y** ∈Y*.

**6.4.2 Bayes’ formula**

In Bayesian statistics, we model the unknown parameter* u* as a random variable* U** ∼**µ*0 that is distributed according to a prior measure.* µ*0 reﬂects our knowledge concerning the parameter* u* before seeing the data. Moreover, we are given a statistical model (*Y**,** P*) and the according Likelihood* L*, which is a conditional density* f**Y** |**U*=*u*. We aim to* invert *P(*Y** ∈·|**U* =* ·*) to P(*U** ∈·|**Y* =* ·*). The conditional measure P(*U** ∈·|**Y* =* ·*) is the updated prior P(*U** ∈·*) :=* µ*0. This updating/inversion process uses on Bayes’ formula.

**Theorem 6.4.6** (Bayes)**.*** Let** U, Y** be random variables as in Theorem 6.3.10. Then,*

*g**U**|**Y* =*y*(*u*) =* *^(*g*)^(*Y*)^(* |*)^(*U*)^(=)^(*u*)^(()^(*y*)^())^(*g*)^(*U*)^(()^(*u*)^())

*g**Y* (*y*) *, *(6.2)

*for** u** ∈X**, ν**U**-a.e. and** y** ∈Y**,* P(*Y** ∈·*)*-a.e. with** g**Y* (*y*)* >* 0*.*

*Proof.* We need to show that* g**Y** |**U*=*u**g**U* =* g**U,Y* ,* ν**U** ⊗*P(*Y** ∈·*)-a.e.. Let* u** ∈X* with *g**U*(*u*)* >* 0. By deﬁnition,

*g**Y** |**U*=*u*(*y*)*g**U*(*u*) =* *^(*g*)^(*U,Y*)^( ()^(*u, y*)^())^(*g*)^(*U*)^(()^(*u*)^())

*g**U*(*u*) =* g**U,Y* (*u, y*) ((*u, y*)* ∈{**g**U** >* 0*} × Y**, ν**U** ⊗*P(*Y** ∈·*)-a.e.)*.*

Conversely, let* u** ∈X*, with* g**U*(*u*) = 0. This implies that

0 = Z

*Y **g*(*u, y*)d*ν**Y* (*y*)*.*

Then,* g**U,Y* (*u,** ·*) = 0,* ν**Y* -a.e. and, thus, also P(*Y** ∈·*)-a.s.. Hence,* g**U,Y* = 0 =* g**Y** |**U*=*u**g**U*.

**Deﬁnition 6.4.7. **•* Z*(*y*) :=* g**Y* (*y*)* is called (model) evidence or marginal likelihood*^(1)*,*

•* L*(*y**|**u*) :=* g**Y** |**U*=*u*(*y*)* is called (data) likelihood,*

1*Z*(*y*) is derived from German:* Zustandssumme (‘sum of states’)*


---

## Page 77

*6.4. BAYESIAN STATISTICS *77

•* µ*0 := P(*U** ∈·*)* is called prior (measure),*

•* µ*post := P(*U** ∈·|**Y* =* y*)* is called posterior (measure), and*

In Theorem 6.4.6, we require that* µ*0 has a probability density function* g**U* with respect to a measure* ν**U*. In practice,* ν**U* is often a Lebesgue measure or the counting measure. In some cases, neither of those two is well-deﬁned or a sensible choice, e.g., when dim* X* =* ∞*. However, we can always assume that* ν**U* :=* µ*0. In this case, we obtain the formulation of Stuart [35]: d*µ*post

d*µ*0 (*u*) =* *^(*L*)^(()^(*y*)^(*|*)^(*u*)^())

*Z*(*y*) (*u** ∈X**, µ*0-a.s.)*.*

**Remark 6.4.8.** When deﬁning* Z*(*y*) := R *L*(*y**|**u*)d*µ*0, it is not necessary for* L*(*y**|**u*) to be correctly normalised. Indeed, we can set* L*(*y**|**u*) :=* c** ·** g**Y** |**U*=*u*(*y*), for some constant* c >* 0. The factor* c* cancels with the same factor in* Z*(*y*). However, then we have* Z*(*y*)* ̸*=* f**Y* (*y*) and call* Z*(*y*) normalising constant.


---

## Page 78

78 *CHAPTER 6. BAYESIAN PROBABILITY AND STATISTICS*


---

## Page 79

# **Chapter 7**
# **Bayesian inverse problems and well-posedness**

In this chapter, we will deﬁne Bayesian inverse problems and study their well-posedness. Well-posedness requires existence and uniqueness of the posterior measure, as well as its stability with respect to marginal perturbations in the data.

### **7.1 Bayesian inverse problems**

A posterior measure is a conditional probability distribution and as such only for almost every data set uniquely deﬁned. In the following, we will always pick one representing Markov kernel out of the set of kernels satisfying the equation in Theorem 6.3.5. We do so, by ﬁxing the deﬁnition of the likelihood to a speciﬁc measurable function* X × Y →*R and deﬁning the posterior to satisfy Bayes’ formula with this likelihood. We ﬁrst introduce some further notation.

**Deﬁnition 7.1.1.*** Let* (Ω^(*′*)*,** F*^(*′*))* be some measurable space. We deﬁne the space of probabil- ity measures on* (Ω^(*′*)*,** F*^(*′*))* by* Prob(Ω^(*′*)*,** F*^(*′*)) :=* {**µ* :* µ** is a probability measure on* (Ω^(*′*)*,** F*^(*′*))*}**. Moreover, for some** σ**-ﬁnite measure** ν** on* (Ω^(*′*)*,** F*^(*′*))*, we deﬁne* Prob(Ω^(*′*)*,** F*^(*′*)*, ν*) :=* {**µ** ∈ *Prob(Ω*,** F*^(*′*)) :* µ** ≪**ν**}**.*

**Deﬁnition 7.1.2.*** Let** µ*0* ∈*Prob(*X**,** BX*)* and** L* : (*X × Y**,** BX ⊗BY*)* →*(R*,** B*R)* be a measureable function. We deﬁne the Bayesian inverse problem (BIP) with prior** µ*0* and likelihood** L**, to be the problem of ﬁnding** µ*post* ∈*Prob(*X**,** BX*)* with*

d*µ*post

d*µ*0 (*u*) = *L*(*f**n**|**u*) R

*X** *^(*L*)^(()^(*f*)^(*n*)^(*|*)^(*u*)^()d)^(*µ*)^(0)^(()^(*u*)^() )(*u** ∈X*;* µ*0*-a.s.*)

*for any data set** f**n** ∈Y**.*

We discussed previously how to construct a likelihood in the ‘classical’ inverse problem setting ﬁnd* u** ∈X* :* A*(*u*) +* n* =* f**n**.*

We now allow for much more general likelihood functions; this includes non-additive noise, Poissonian models,...

79


---

## Page 80

80 *CHAPTER 7. BAYESIAN INVERSE PROBLEMS AND WELL-POSEDNESS*

**Deﬁnition 7.1.3.*** Consider a (BIP) with prior** µ*0* and likelihood** L**. Let** P** ⊆*Prob(*X**,** BX*) *be a space of probability measures and** d* :* P* ^(2)^(* *)*→*[0*,** ∞*)* be a metric on** P**. A Bayesian inverse problem is* (*P, d*)*-well-posed, if*

*(i) for all** f**n** ∈Y**, the probability measure** µ*post* ∈**P** exists, *(existence)

*(ii) for all** f**n** ∈Y**, the probability measure** µ*post* ∈**P** is unique, and *(uniqueness)

*(iii) the map** Y ∋**f**n** 7→**µ*post* ∈**P** is continuous. *(stability)

Existence and uniqueness of the posterior in* P** ∈{*Prob(*X**,** BX*)*,* Prob(*X**,** BX**, µ*0)*}* is automatic, if R

*X** *^(*L*)^(()^(*f*)^(*n*)^(*|*)^(*u*)^()d)^(*µ*)^(0)^(()^(*u*)^())^(* ∈*)^((0)^(*,*)^(* ∞*)^())^(*.*)^( This is, for instance, the case in the following )lemma.

**Lemma 7.1.4.*** Consider a (BIP) with prior** µ*0* and likelihood** L**. Let** L >* 0* (**µ*0*-a.s.) and **L*(*f**n**|·*)* ∈**L*^(1)(*X**,** BX**, µ*0)* for any** f**n** ∈Y**. Then, the posterior** µ*post* ∈*Prob(*X**,** BX**, µ*0) *exists and is unique.*

*Proof.* We need to show that R

*X** *^(*L*)^(()^(*f*)^(*n*)^(*|*)^(*u*)^()d)^(*µ*)^(0)^(()^(*u*)^())^(* ∈*)^((0)^(*,*)^(* ∞*)^())^(*.*)^( Upper bound: )trivial, since *L*(*f**n**|·*)* ∈**L*^(1)(*X**,** BX**, µ*0). Lower bound: exercise.

Before we can actually speak about stability, we need to discuss metrics on spaces of probability measures.

### **7.2 Metrics on spaces of probability measures**

We consider metrics on subspaces of Prob(*X**,** BX*) to be able to show stability of the poste- rior measure with respect to perturbations in the data. We consider two diﬀerent concept: total variation and weak convergence.

**Deﬁnition 7.2.1. ***(i) Let* (Ω^(*′*)*,** F*^(*′*))* be a measurable space. We deﬁne the total variation (TV) distance on* Prob(Ω^(*′*)*,** F*^(*′*))* by*

*d*TV : Prob(Ω^(*′*)*,** F*^(*′*))^(2)^(* *)*→*[0*,** ∞*)*,* (*µ, ν*)* 7→*sup *F** *^(*′*)*∈F*^(*′*)^(* |*)^(*µ*)^(()^(*F*)^(* ′*)^())^(* −*)^(*ν*)^(()^(*F*)^(* ′*)^())^(*|*)

*(ii) Let* Ω^(*′*)^(* *)*be a topological space and* (Ω^(*′*)*,** F*^(*′*)) := (Ω^(*′*)*,** B*Ω^(*′*))*. Let* (*µ**n*)*n**∈*N* ∈*Prob(Ω^(*′*)*,** F*^(*′*))^(N)

*and** µ** ∈*Prob(Ω^(*′*)*,** F*^(*′*))*. We say** µ**n** →**µ** weakly, as** n** →∞**, if*

lim *n**→∞*

Z

Ω^(*′*)^(* g*)^(d)^(*µ*)^(*n*)^( = )Z

Ω^(*′*)^(* g*)^(d)^(*µ,*)

*for any** g* : (Ω^(*′*)*,** B*Ω^(*′*))* →*(R*,** B*R)* that is continuous and bounded.*

**Remark 7.2.2.** Weak convergence of measures on Prob(*X**,** BX*) can be represented by the (L´evy)-Prokhorov metric* d*LP. See [29] for details. Hence, when referring to the topology induced by weak convergence, we will usually speak about the metric space (Prob(*X**,** BX*)*, d*LP), but not actually employ the (L´evy)-Prokhorov metric.

We end this section with two more results about the total variation distance and weak convergence. First, we show that if a sequence of measures converges in the total variation distance, it converges weakly as well.


---

## Page 81

*7.3. STABILITY *81

**Lemma 7.2.3.*** Let* Ω^(*′*)^(* *)*be a topological space and* (Ω^(*′*)*,** F*^(*′*)) := (Ω^(*′*)*,** B*Ω^(*′*))*. Let* (*µ**n*)*n**∈*N* ∈ *Prob(Ω^(*′*)*,** F*^(*′*))^(N)^(* *)*and** µ** ∈*Prob(Ω^(*′*)*,** F*^(*′*))*. Then*

lim *n**→∞*^(*d*)^(TV)^(()^(*µ*)^(*n*)^(*, µ*)^() = 0 =)^(*⇒*)^(*µ*)^(*n*)^(* →*)^(*µ,*)^(* weakly as*)^(* n*)^(* →∞*)^(*.*)

*The converse statement (“**⇐**”) is in general not true.*

*Proof.* Exercise.

Second, we give a representation of the total variation distance of two measures having a density with respect to a third measure.

**Lemma 7.2.4.*** Let** µ, ν** ∈*Prob(Ω*,** F*)* and** ρ** be a** σ**-ﬁnite measure with** µ, ν** ≪**ρ**. Then, **d*TV(*µ, ν*) = ^(1)

2 R

Ω

 d*µ*

d*ρ** *^(*−*)^(d)^(*ν*)

d*ρ * d*ρ.*

*Proof.* Exercise.

Note that this result is independent of the measure* ρ*. As a trivial dominating measure, one can always choose* ρ* :=* µ* +* ν*.

### **7.3 Stability**

We now give a set of assumptions under which we can prove (*P, d*)-well-posedness, as deﬁned in Deﬁnition 7.1.3, where (*P, d*) refers to the space of probability measure on* X* with* µ*0- density and either total variation distance or weak convergence.

**Assumption 7.3.1.** Given a (BIP) with prior* µ*0 and likelihood* L*. Let the following assumptions hold for* u** ∈X** µ*0-a.s. and* f**n** ∈Y*.

(A1)* L*(*·|**u*) is a strictly positive probability density function,

(A2)* L*(*f**n**|·*)* ∈**L*^(1)(*X**,** BX**, µ*0),

(A3) some* h** ∈**L*^(1)(*X**,** BX**, µ*0) exists, such that* L*(*f*^(*′ *)*n*^(*|·*)^())^( ⩽)^(*h*)^( for all)^(* f*)^(*′ *)*n** *^(*∈Y*)^(, and)

(A4)* L*(*·|**u*) is continuous.

We now brieﬂy comment on the assumptions. (A1) and (A2) were already required in Lemma 7.1.4. In (A3) we now not only ask for boundedness of the integral of the likelihood, but for its uniform boundedness by an integrable function* g*. This is for instance the case, if the likelihood is bounded by a constant (as* µ*0 is a probability measure). In (A4) we ask for continuity in the data. (Continuity in the parameter is not required!) In inverse problems, this is true for a large number of noise distributions. Before proving well-posedness under Assumptions (A1)-(A4) we cite a fundamental measure-theoretic result which is needed for the proof.

**Theorem 7.3.2** (Dominated Convergence Theorem (DCT; Lebesgue))**.*** Let* (Ω^(*′*)*,** F*^(*′*)*, µ*^(*′*))* be a measure space. Let** g,* (*g**m*)*m**∈*N*, h** be measurable functions* (Ω^(*′*)*,** F*^(*′*))* →*(R*,** B*R)* and** h** ∈ **L*^(1)(Ω^(*′*)*,** F*^(*′*)*, µ*^(*′*))*. Moreover, let** |**g**m**|* ⩽*h** (**µ*^(*′*)*-a.e.) and** g**m** →**g**,** µ*^(*′*)*-a.e. as** m** →∞**. Then, **g, g**m** ∈**L*^(1)(Ω^(*′*)*,** F*^(*′*)*, µ*^(*′*))* and*

lim *m**→∞*

Z

Ω^(*′*)^(* g*)^(*m*)^(d)^(*µ*)^(*′*)^( = )Z

Ω^(*′*)^(* g*)^(d)^(*µ*)^(*′*)^(*.*)


---

## Page 82

82 *CHAPTER 7. BAYESIAN INVERSE PROBLEMS AND WELL-POSEDNESS*

*Proof.* Can be proved using monotonic convergence theorem (Proposition 6.2.11.2). See, e.g., Theorem 1.6.9 [6] for a proof using Fatou’s Lemma.

**Remark 7.3.3.** The DCT describes a case in which we are allowed to “exchange integral and limit”. The statement reads

lim *m**→∞*

Z

Ω^(*′*)^(* g*)^(*m*)^(d)^(*µ*)^(*′*)^( = )Z

Ω^(*′*)^( lim )*n**→∞*^(*g*)^(*m*)^(d)^(*µ*)^(*′*)^(*.*)

Equivalently, we could say it describes cases in which the integral as a functional of the integrand is continuous.

**Theorem 7.3.4** (Well-posedness)**.*** Given a (BIP) with prior** µ*0* and likelihood** L** that sat- isﬁes Assumptions (A1)–(A4). Moreover, let** P* = Prob(*X**,** BX**, µ*0)* and** d** ∈{**d*TV*, d*LP*}**. Then, the (BIP) is* (*P, d*)*-well-posed.*

*Proof.* 1. Note that (A1), (A2) already imply existence and uniqueness by Lemma 7.1.4. In the remainder of the proof, we focus on showing continuity in the total variation distance. Continuity in weak convergence is then implied by Lemma 7.2.3. Indeed, we show that for all* f**n** ∈Y* and all (*f*^(()^(*m*)^() )*n *)*m**∈*N* ∈Y*^(N), with lim*m**→∞**f*^(()^(*m*)^() )*n *=* f**n*, we have

Z

*X*

 *L*(*f**n**|**u*)

*Z*(*f**n*)* *^(*−*)^(*L*)^(()^(*f*)^(()^(*m*)^() )*n **|**u*)

*Z*(*f*^(()^(*m*)^() )*n *)

 d*µ*0(*u*)* →*0 (*m** →∞*)*.*

where* Z*(*f**n*) := R

*X** *^(*L*)^(()^(*f*)^(*n*)^(*|*)^(*u*)^()d)^(*µ*)^(0)^(()^(*u*)^(). By Lemma 7.2.4, this implies continuity of the posterior )measure in the total variation distance. 2. We ﬁrst show that* Y ∋**f**n** 7→**L*(*f**n**|·*)* ∈**L*^(1)(*X**,** BX**, µ*0) is continuous. Let* f**n** ∈Y* and (*f*^(()^(*m*)^() )*n *)*m**∈*N* ∈Y*^(N), with lim*m**→∞**f*^(()^(*m*)^() )*n *=* f**n*. Note that

lim *m**→∞*

Z

*X*

*L*(*f*(*m*) *n **|**u*)* −**L*(*f**n**|**u*)  d*µ*0(*u*) = Z

*X *lim *m**→∞*

*L*(*f*(*m*) *n **|**u*)* −**L*(*f**n**|**u*)  d*µ*0(*u*)*,*

due to the DCT since the integrand is bounded below by 0 and above by 2*h** ∈**L*^(1)(*X**,** BX**, µ*0). Due to the continuity of* L*(*·|**u*) (required in (A4)), we have then

lim *m**→∞*

Z

*X*

*L*(*f*(*m*) *n **|**u*)* −**L*(*f**n**|**u*)  d*µ*0(*u*) = 0*.*

With the same argument, we can show that* f**n** 7→**Z**n*(*f**n*) is continuous. 3. The rest of the proof is similar to showing continuity of the quotient of two continuous functions. Let* f**n** ∈Y* and (*f*^(()^(*m*)^() )*n *)*m**∈*N* ∈Y*^(N), with lim*m**→∞**f*^(()^(*m*)^() )*n *=* f**n*. Then

Z

*X*

 *L*(*f**n**|**u*)

*Z*(*f**n*)* *^(*−*)^(*L*)^(()^(*f*)^(()^(*m*)^() )*n **|**u*)

*Z*(*f*^(()^(*m*)^() )*n *)

 d*µ*0(*u*)

⩽*Z*(*f**n*)^(*−*)^(1 )Z

*X*

*L*(*f*(*m*) *n **|**u*)* −**L*(*f**n**|**u*)  d*µ*0(*u*) | {z } *→*0 (*m**→∞*)

+ Z

*X **L*(*f*^(()^(*m*)^() )*n **|**u*)d*µ*0(*u*)* |**Z*(*f**n*)^(*−*)^(1)^(* *)*−**Z*(*f*^(()^(*m*)^() )*n *)^(*−*)^(1)*| *| {z } *→*0 (*m**→∞*)

and the terms that do not converge to 0 are bounded.

To illustrate the generality of this result, we now study again the inverse problem from Example 6.4.5.


---

## Page 83

*7.3. STABILITY *83

**Corollary 7.3.5.** Let* k** ∈*N and (*Y**,** BY*) := (R^(*k*)*,** B*R^(*k*)) and Γ* ∈*R^(*k*)^(*×*)^(*k*)^( )be symmetric, positive deﬁnite. Moreover, let* A* : (*X**,** BX*)* →*(*Y**,** BY*) be some function. Consider the (BIP) with some prior* µ*0* ∈*Prob(*X**,** BX*) and likelihood

*L*(*f**n**|**u*) := (2*π*)^(*−*)^(*k/*)^(2)det(Γ)^(*−*)^(1)^(*/*)^(2)^( )exp  *−*^(1)

2^(*∥*)^(Γ)^(*−*)^(1)^(*/*)^(2)^(()^(*f*)^(*n*)^(* −A*)^(()^(*u*)^()))^(*∥*)^(2 ) (*u** ∈X**, f**n** ∈Y*)

Then, the (BIP) is (*P, d*)-well-posed, with* P* = Prob(*X**,** BX**, µ*0) and* d** ∈{**d*TV*, d*LP*}*.

*Proof.* Follows trivially from Theorem 7.3.4.


---

## Page 84

84 *CHAPTER 7. BAYESIAN INVERSE PROBLEMS AND WELL-POSEDNESS*


---

## Page 85

# **Chapter 8**
# **Function space priors and Monte Carlo**

In this last chapter, we would like to discuss two rather practical topics:

• In inverse problems, we often consider inﬁnite-dimensional parameter spaces. While we have discussed the well-posedness of Bayesian inverse problems in inﬁnite dimen- sional setting, it is not clear yet how, e.g., a prior probability measure on such a space can be deﬁned. We will discuss Gaussian prior measures on function spaces, so-called Gaussian random ﬁelds. For a more thorough introduction, we refer to the book by Bogachev [11].

• In practical situations, we need to approximate the posterior (or integrals with respect to it) numerically. We will discuss Monte Carlo techniques that are suitable for Bayesian inverse problems. Again, for a more thorough discussion of certain aspects, we refer to Agapiou et al. [3], Cotter et al. [16], and Robert and Casella [30].

### **8.1 Gaussian measures**

We have deﬁned Gaussian measures on (R*,** B*R) in Deﬁnition 6.2.13. We now extend this deﬁnition to measurable spaces like (*X**,** BX*)*,* where* X* is a separable Banach space. In this section, we assume that all Banach and Hilbert spaces are with respect to R.

**Deﬁnition 8.1.1.*** Let** µ** be a probability measure on* Prob(*X**,** BX*)* and let** U** ∼**µ**. We call **µ** Gaussian, if for all** ℓ**∈**X*^(*∗*)*, there exist** m** ∈*R*, σ* ⩾0*, such that*

P(*⟨**ℓ, U**⟩∈·*) = N(*m, σ*^(2))*.*

*Moreover, we deﬁne the mean of** µ** by** a**µ** ∈X** *^(*∗∗*)*, given by*

*a**µ*(*ℓ*) = Z

*X **⟨**ℓ, u**⟩*d*µ*(*u*) (*ℓ**∈X** *^(*∗*))

*and the covariance operator of** µ** by** R**µ* :* X** *^(*∗*)*→X** *^(*∗∗*)*, where*

*R**µ*(*ℓ*)(*ℓ*^(*′*)) = Z

*X *(*⟨**ℓ, u**⟩−**a**µ*(*ℓ*)) � *⟨**ℓ*^(*′*)*, u**⟩−**a**µ*(*ℓ*^(*′*))  d*µ*(*u*) (*ℓ, ℓ*^(*′*)^(* *)*∈X** *^(*∗*))*.*

*If** X** is a function space, we call** U** Gaussian random ﬁeld.*

85


---

## Page 86

86 *CHAPTER 8. FUNCTION SPACE PRIORS AND MONTE CARLO*

This deﬁnition does not immediately lead to a construction of a Gaussian measure on a general separable Banach space. There are two cases, in which we have techniques to construct a Gaussian measure on* X*; those are R^(*k*)^( )and separable Hilbert spaces. In ﬁnite dimensions, one can deﬁne a Gaussian measure in terms of a probability density function with respect to the product of a Lebesgue measure and a Dirac measure. On a separable Hilbert space, we can construct a series expansion, the so-called Karhunen-Lo`eve expansion.

**Deﬁnition 8.1.2.*** Let** X** be a separable Hilbert space and** C* :* X →X** be a compact, self adjoint linear operator. Moreover, let* (*λ**i**, ϕ**i*)*i**∈*N* ∈*(R* × X*)^(N)^(* *)*be the eigenpairs of** C** sorted decreasingly with respect to the absolute value of the eigenvalue and* (*ϕ**i*)*i**∈*N* is orthonormal. Then, we can represent*

*Cx* =

*∞ *X

*i*=1 *λ**i**⟨**x, ϕ**i**⟩**X** ϕ**i *(*x** ∈X*)*,*

*see also Theorem 2.2.4.** C** is a trace class operator, if* (*λ**i*)*i**∈*N* ∈**ℓ*^(1)*.*

**Proposition 8.1.3.*** Let** X** be a separable Hilbert space and** C* :* X →X** be a linear oper- ator that is self-adjoint, non-negative, and trace class. We denote the eigenpairs of** C** by *(*λ**i**, ϕ**i*)*i**∈*N* ∈*(R* × X*)^(N)*; the eigenvalues are sorted decreasingly and* (*ϕ**i*)*i**∈*N* is orthonormal. Finally, let** m** ∈X** and** ξ** ∼*N(0*,* 1^(2))^(*⊗*)^(N)*. Then,*

*U* :=* m* +

*∞ *X

*i*=1

p

*λ**i**ξ**i**ϕ**i*

*is distributed according to a Gaussian measure with mean** m** and covariance operator** C**.*

*Proof.* Let* k** ∈*N and* U**k* :=* m* + ^(P)^(*k *)*i*=1 *√**λ**i**ξ**i**ϕ**i*. Moreover, let* x** ∈X* and* x**i* :=* ⟨**x, ϕ**i**⟩**X* for *i** ∈*N. We ﬁrst study the distribution of* ⟨**x, U**⟩**X** .*

*⟨**x, U**k**⟩**X* =

*

*x, m* +

*k *X

*i*=1

p

*λ**i**ξ**i**ϕ**i*

+

*X*

=* ⟨**x, m**⟩**X* +

*

*x,*

*k *X

*i*=1

p

*λ**i**ξ**i**ϕ**i*

+

*X*

=* ⟨**x, m**⟩**X* +

*k *X

*i*=1

p

*λ**i** ⟨**x, ϕ**i**⟩**X** ξ**i*

=* ⟨**x, m**⟩**X* +

*k *X

*i*=1

p

*λ**i**x**i**ξ**i *| {z } *∼*N(0*,λ**i**x*^(2 )*i* ^())

converges weakly to the Gaussian distribution N � *⟨**x, m**⟩**X** ,* ^(P)^(*∞ *)*i*=1* *^(*λ*)^(*i*)^(*x*)^(2 )*i * (*k** →∞*), if the sum P*∞ **i*=1* *^(*λ*)^(*i*)^(*x*)^(2 )*i* ^(is ﬁnite. (This can be shown with the Fourier transform of Gaussian measures, )as the (*ξ**i*)*i**∈*N are mutually independent). By assumption, we have (*λ**i*)*i**∈*N* ∈**ℓ*^(1)^( )and also (*x*^(2 )*i* ^())^(*i*)^(*∈*)^(N)^(* ∈*)^(*ℓ*)^(1)^(, since)^( P)^(*∞ *)*i*=1* *^(*x*)^(2 )*i* ^(=)^(* ∥*)^(*x*)^(*∥*)^(2 )*X** *^(*<*)^(* ∞*)^(. Hence, also)^( P)^(*∞ *)*i*=1* *^(*λ*)^(*i*)^(*x*)^(2 )*i** *^(*<*)^(* ∞*)^(. )Next, we show that* U* takes values in* X* with probability one, i.e. P(*∥**U**∥**X** <** ∞*) = 1. By Parseval’s identity, we have

*∥**U**∥**X* =

*∞ *X

*i*=1 *|⟨**U, ϕ**i**⟩**X** |*^(2)^( )=

*∞ *X

*i*=1 *λ**i**ξ*^(2 )*i*


---

## Page 87

*8.1. GAUSSIAN MEASURES *87

which is almost surely ﬁnite by Theorem 1.1.4 of [11], as (*λ**i*)*i**∈*N* ∈**ℓ*^(1). Now, we look at mean and covariance of* U*. We have

*a**µ*(*x*) = Z

*X **⟨**x, U**⟩*dP =* ⟨**x, m**⟩**X* + Z

R^(N)

*∞ *X

*i*=1

p

*λ**i**x**i**ξ**i*dN(0*,* 1)^(*⊗*)^(N)(*ξ*)

=* ⟨**x, m**⟩**X* +

*∞ *X

*i*=1

p

*λ**i**x**i*

Z

R^(N)^(* ξ*)^(*i*)^(dN(0)^(*,*)^( 1))^(*⊗*)^(N)^(()^(*ξ*)^() )| {z } =0 =* ⟨**x, m**⟩**X** ,*

where we used the Fubini-Tonelli theorem to switch inﬁnite sum and integral: Note that

*∞ *X

*i*=1

Z

R^(N)^(* x*)^(*| *)p

*λ**i**x**i**ξ**i**|*dN(0*,* 1)^(*⊗*)^(N)(*ξ*) =

*∞ *X

*i*=1

r

2 *π* ^(⩽)

r

2 *π*^(*∥ *)p

*λ**i**∥*2*∥|**x**i**|∥*2

by Cauchy-Schwarz. Moreover, the upper bound on the RHS is ﬁnite, since (*x**i*)*i**∈*N*,* (*λ**i*)*i**∈*N* ∈ **ℓ*^(2). Hence,* a**µ* =* m*. Furthermore, we have for* x*^(*′*)^(* *)*∈X*:

*R**µ*(*x*)(*x*^(*′*)) = Z

*X *(*⟨**u, x**⟩**X** −**a**µ*(*x*)) � *⟨**u, x*^(*′*)*⟩**X** −**a**µ*(*x*^(*′*))  d*µ*(*u*)

= Z

R^(N)

*

*x,*

*∞ *X

*i*=1

p

*λ**i**ξ**i**ϕ**i*

+

*X*

** ∞ *X

*j*=1

p

*λ**j**ξ**j**ϕ**j**, x*^(*′ *)+

*X *dN(0*,* 1)^(*⊗*)^(N)(*ξ*)

= Z

R^(N)

*∞ *X

*i*=1

*∞ *X

*j*=1

p

*λ**i *p

*λ**j** ⟨**x, ϕ**i**⟩**X** ξ**i**ξ**j *
 *ϕ**j**, x*^(*′*)^()

*X* ^(dN(0)^(*,*)^( 1))^(*⊗*)^(N)^(()^(*ξ*)^())

=

*∞ *X

*i*=1

*∞ *X

*j*=1

p

*λ**i *p

*λ**j** ⟨**x, ϕ**i**⟩**X*

Z

R^(N)^(* ξ*)^(*i*)^(*ξ*)^(*j*)^(dN(0)^(*,*)^( 1))^(*⊗*)^(N)^(()^(*ξ*)^() )| {z } =**1***{**j**}*(*i*)

 *ϕ**j**, x*^(*′*)^()

*X*

=

*∞ *X

*i*=1 *λ**i** ⟨**x, ϕ**i**⟩**X *
 *ϕ**i**, x*^(*′*)^()

*X* ^(=)^(* ⟨*)^(*x, Cx*)^(*′*)^(*⟩*)^(*X*)^(* ,*)

where we could remove the sum over* j* above due to mutual independence of the* ξ**i**, ξ**j* with *i** ̸*=* j*. We exchanged sums and integral again using the Fubini-Tonelli theorem:

*∞ *X

*i*=1

*∞ *X

*j*=1

Z

R^(N)^(* | *)p

*λ**i *p

*λ**j** ⟨**x, ϕ**i**⟩**X** ξ**i**ξ**j *
 *ϕ**j**, x*^(*′*)^()

*X** *^(*|*)^(dN(0)^(*,*)^( 1))^(*⊗*)^(N)^(()^(*ξ*)^())

=

*∞ *X

*i*=1

*∞ *X

*j*=1 *| *p

*λ**i *p

*λ**j** ⟨**x, ϕ**i**⟩**X *
 *ϕ**j**, x*^(*′*)^()

*X** *^(*| ·*)^( 2)

*π*

=

*∞ *X

*i*=1 *| *p

*λ**i** ⟨**x, ϕ**i**⟩**X** |*

*∞ *X

*j*=1 *| *p

*λ**j *
 *ϕ**j**, x*^(*′*)^()

*X** *^(*| ·*)^( 2)

*π*^(*,*)

which is again ﬁnite, as (*x**i*)*i**∈*N*,* (*x*^(*′ *)*i*^())^(*i*)^(*∈*)^(N)^(*,*)^( ()^(*λ*)^(*i*)^())^(*i*)^(*∈*)^(N)^(* ∈*)^(*ℓ*)^(2)^(.)

**Deﬁnition 8.1.4.*** The expansion*

*m* +

*∞ *X

*i*=1

p

*λ**i**ξ**i**ϕ**i*


---

## Page 88

88 *CHAPTER 8. FUNCTION SPACE PRIORS AND MONTE CARLO*

*in Proposition 8.1.3 is called Karhunen–Lo`eve expansion (KLE). In the same proposition, we denote** µ* =: N(*m, C*)*.*

We can understand the KLE as the function space version of a principal component analysis. Indeed, random ﬁelds are often discretised by representing them as a KLE and truncating the expansion. We now study two examples of Gaussian random ﬁelds in* L*^(2).

**Example 8.1.5** (Gaussian random ﬁelds in 2 dimensions)**.** Let* D* = [0*,* 1]^(2),* X* :=* L*^(2)(*D,** B**D, λ*2), *ℓ>* 0, and* σ*^(2)^( )⩾0. We deﬁne the exponential covariance function

*c*exp(*x, y*) :=* σ*^(2)^( )exp  *−*^(*∥*)^(*x*)^(* −*)^(*y*)^(*∥*)^(2)

*ℓ*

 (*x, y** ∈**D*)

and the Gaussian covariance function

*c*N(*x, y*) :=* σ*^(2)^( )exp  *−*^(*∥*)^(*x*)^(* −*)^(*y*)^(*∥*)^(2 )2 2*ℓ*^(2)

 (*x, y** ∈**D*)*.*

The parameter* ℓ*is called correlation length,* σ*^(2)^( )is called pointwise variance. We can now deﬁne the associated covariance operators for* c** ∈{**c*exp*, c*N*}*, by

*C* :* X →X**, ϕ** 7→ *Z

*D **ϕ*(*x*)*c*(*x,** ·*)d*λ*2(*x*)*.*

Well-deﬁnedness of these covariance operators can be shown with Mercer’s Theorem. In Figure 8.1, we show discretised samples of Gaussian random ﬁelds with both covariance func- tions and* ℓ**∈{*0*.*05*,* 0*.*1*,* 1*}*. The random ﬁelds have been discretised by a 100^(2)-dimensional piecewise-constant ﬁnite element approximation of the eigenpairs of the respective covari- ance operator.

### **8.2 Monte Carlo techniques**

**8.2.1 Standard Monte Carlo**

Monte Carlo techniques aim at approximating integrals of the form

*g* := Z

*X **g*d*µ,*

where* µ* is a probability distribution on (*X**,** BX*) and* g* : (*X**,** BX*)* →*(R*,** B*R) is an inte- grable function. Standard Monte Carlo approaches this problem by generating independent samples* U*1*, U*2*, ...** ∼**µ* and computing the estimator

b*g**M* := ^(1)

*M*

*M *X

*m*=1 *g*(*U**m*)*,*

for some* M** ∈*N. Alternatively, we can understand Monte Carlo as a technique allowing us to approximate the probability measure* µ* by the probability measure

b*µ**M* := ^(1)

*M*

*M *X

*m*=1 *δ*(*· −**U**m*)*.*


---

## Page 89

*8.2. MONTE CARLO TECHNIQUES *89

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

0 0.2 0.4 0.6 0.8

0

0.2

0.4

0.6

0.8

Figure 8.1: Each row represents four samples from the Gaussian random ﬁeld with mean *m* = 0 and the following covariance operators (from top to bottom): exponential with *ℓ*= 0*.*05, exponential with* ℓ*= 0*.*1, exponential with* ℓ*= 1, Gaussian with* ℓ*= 0*.*05, Gaussian with* ℓ*= 0*.*1, and Gaussian with* ℓ*= 1*.*


---

## Page 90

90 *CHAPTER 8. FUNCTION SPACE PRIORS AND MONTE CARLO*

The Monte Carlo estimator can be analysed using the (strong) law of large numbers. We know that b*g**M** →**g *(*M** →∞**,* P-a.s.)*.*

If in addition Var*µ*(*g*) := R *g*^(2)d*µ** − *�R *g*d*µ *2* <** ∞*, we obtain the following convergence rate r

E h (b*g**M** −**g*)^(2)^(i )=

p

Var*µ*(*g*) *√*

*M .*

When thinking about standard algorithms for numerical quadrature (Gauss quadrature, Simpson’s rule,...) the rate* O*(*M*^(*−*)^(1)^(*/*)^(2)) appears to be quite slow. A composite Simpson’s rule, e.g., for a very smooth function over* X* := [0*,* 1] has an absolute error of* O*(*M*^(*−*)^(4)). Its advantage over classical methods is that the rate is independent of the smoothness of the function and the dimension of its domain. Hence, Monte Carlo methods are especially useful in problems that are non-smooth and/or high-dimensional. Unfortunately, standard Monte Carlo techniques are usually unsuitable for the approx- imation of posterior measures in Bayesian inverse problems: we are not able to sample independently from the posterior measure. Ideas:

• sample dependently from* µ*post (*→*Markov chain Monte Carlo; this lecture) or

• sample independently from a diﬀerent measure and correct by choosing unequal weights

b*g**M* :=

*M *X

*m*=1 *w**m**g*(*U**m*)*,*

with* w**m** ̸*= 1*/M*,* m* = 1*, . . . , M* (*→*Importance Sampling; exercise sheet 4)

**Markov chain Monte Carlo**

In Markov chain Monte Carlo (MCMC), we generate a Markov chain (*U**m*)*m**∈*N that is stationary with respect to the posterior measure* µ*post and Harris recurrent. In this case, we also have a law of large numbers

1 *M*

*M *X

*m*=1 *g*(*U**m*)* → *Z

*X **g*d*µ*post (*M** →∞**,* P-a.s.)*,*

for some integrable* g* : (*X**,** BX*)* →*(R*,** B*R); see [30, Theorem 6.63]. We give a comparison of Monte Carlo and Markov chain Monte Carlo in Figure 8.2. In the ﬁgure, we see that sampling a Markov chain can be less eﬃcient than independent sampling – making MCMC not appearing very natural just yet. However, it is often easier to generate such a Markov chain than to sample independently from the posterior. In the following, we will ﬁrst recap some deﬁnitions concerning Markov chains. Then, we will introduce the Metropolis–Hastings algorithm and show that is is stationary with respect to our measure of interest; say the posterior measure. We will not discuss ergodicity/Harris recurrence in this short introduction, but refer to the work by Robert and Casella [30].

**Deﬁnition 8.2.1.*** Let* (*U**n*)^(*∞ *)*n*=1* *^(*be a sequence of*)^(* X*)^(*-valued random variables - so-called *)*states.* (*U**n*)^(*∞ *)*n*=1* *^(*is called Markov chain, if for any*)^(* n*)^(* ∈*)^(N)^(*:*)

P(*U**n*+1* ∈·|**U*1 =* u*1*, U*2 =* u*2*, ..., U**n**−*1 =* u**n**−*1*, U**n* =* u**n*) = P(*U**n*+1* ∈·|**U**n* =* u**n*) (8.1)


---

## Page 91

*8.2. MONTE CARLO TECHNIQUES *91

0 0.2 0.4 -4

-2

0

2

4

0 500 1000 1500 2000 2500 3000 -4

-2

0

2

4

Space X

**Standard Monte Carlo**

0 0.2 0.4 -4

-2

0

2

4

Kernel density estimate of samples actual 1^(-density)

0 500 1000 1500 2000 2500 3000 Time/index of samples m

-4

-3

-2

-1

0

1

2

3

Space X

**Markov chain Monte Carlo**

Figure 8.2: Comparison of Monte Carlo and Markov chain Monte Carlo samples. In the top row, we show 3000 independent samples of N(0*,* 1^(2)) and a kernel density estimate of these samples along with the true density. In the bottom row, we show 3000 samples generated with the Random Walk Metropolis algorithm targeting N(0*,* 1^(2)). The proposal kernel is N(*·**,* 0*.*5^(2)). The samples in the bottom row are clearly dependent.

*for any** u*1*, ..., u**n**−*1* ∈X**. A Markov chain is called time-homogeneous, if*

P(*U*2* ∈·|**U*1 =* u*) = P(*U**k*+2* ∈·|**U**k*+1 =* u*) (*u** ∈X**, k** ∈*N)*. *(8.2)

*and otherwise time-inhomogeneous. A time-homogeneous Markov chain can be fully repre- sented by a Markov kernel** K* :* BX × X →*[0*,* 1]*:*

*K*(*B**|**u*) = P(*U**n*+1* ∈**B**|**U**n* =* u*) (*B** ∈BX**, u** ∈X**, n** ∈*N)*.*

*Let** µ** ∈*Prob(*X**,** BX*)* be a probability measure. We denote the composition of** µ** and** K** by*

*µK*(*B*) := Z

*X **K*(*B**|**u*)d*µ*(*u*) (*B** ∈BX*)*.*

*The measure** µ** is stationary w.r.t.** K**, if** µK* =* µ.** Finally, we say, the Markov kernel** K **satisﬁes detailed balance w.r.t.** µ*^(*′*)^(* *)*∈*Prob(*X**,** BX*)*, if *Z

*B **K*(*A**|**u*)d*µ*^(*′*)(*u*) = Z

*A **K*(*B**|**u*)d*µ*^(*′*)(*u*) (*A, B** ∈BX*)*.*

The detailed balance condition implies that the measure with respect to which it was shown is the stationary measure:

**Lemma 8.2.2.*** Let** K* :* BX × X →*[0*,* 1]* be a Markov kernel that satisﬁes detailed balance with respect to** µ** ∈*Prob(*X**,** BX*)*. Then,** K** is stationary w.r.t.** µ**.*


---

## Page 92

92 *CHAPTER 8. FUNCTION SPACE PRIORS AND MONTE CARLO*

*Proof.* Exercise.

We now deﬁne the Metropolis–Hastings Markov Kernel, discuss it, and show that it is stationary with respect to the target measure.

**Deﬁnition 8.2.3** (Hastings 1970 [22])**.*** Let** µ** ∈*Prob(*X**,** BX*)* and** ν** be a** σ**-ﬁnite measure with** µ** ≪**ν**. Moreover let** g* : (*X**,** BX*)* →*(R*,** B*R)* be a positive function with*

*g* =* c** ·* ^(d)^(*µ*)

d*ν *^(*,*)

*for some** c** ∈*(0*,** ∞*)*. Moreover, let** Q* :* X × BX →*[0*,* 1]* be a Markov kernel, given by a positive function** q* : (*X × X**,** BX ⊗BX*)* →*(R*,** B*R)*, with*

*Q*(*A**|**u*) := Z

*A **q*(*u*^(*′*)*|**u*)d*ν*(*u*^(*′*)) (*A** ∈BX**, u** ∈X*)*.*

*The Metropolis–Hastings Markov kernel is given by*

*K*MH(*A**|**u*) :=* δ*(*A**−**u*) Z

*X *(1*−**α*(*u, u*^(*′′*)))*Q*(d*u*^(*′′*)*|**u*)+ Z

*A **α*(*u, u*^(*′*))*Q*(d*u*^(*′*)*|**u*) (*u** ∈X**, A** ∈BX*)*,*

*where*

*α*(*u, u*^(*′*)) = min  1*, *^(*g*)^(()^(*u*)^(*′*)^())^(*q*)^(()^(*u*)^(*|*)^(*u*)^(*′*)^())

*g*(*u*)*q*(*u*^(*′*)*|**u*)

 *.*

Interpreting this Markov kernel is rather diﬃcult. Algorithmically, we can represent the Metropolis–Hastings MCMC method

1. Start with some initial value* U*1* ∈X* (say a.s. constant); set* m** ←*1;

2. Sample* U*^(*∗*)*∼**Q*(*·|**U**m*); (‘proposal step’)

3. With probability* α*(*U**m**, U*^(*∗*)) set* U**m*+1* ←**U*^(*∗*), otherwise* U**m*+1* ←**U**m*+1; (‘acceptance step’)

4. Increment* m** ←**m* + 1 and go to 2.

When looking at* K*MH, we see the proposal step in the Markov kernel* Q* and the acceptance step in the (1* −**α*) and the* α*. Another remarkable observation is that we need to know the density* g* only up to a normalising constant. This is especially useful, when sampling from a posterior measure: we usually have only access to prior density and likelihood. Model evidence/normalising constant are not necessary.

**Proposition 8.2.4.*** K*MH* satisﬁes detailed balance w.r.t.** µ**.*

*Proof.* Let* A, B** ∈BX*. Z

*B **K*MH(*A**|**u*)d*µ*(*u*)

= Z

*B **δ*(*A** −**u*) Z

*X *(1* −**α*(*u, u*^(*′′*)))*Q*(d*u*^(*′′*)*|**u*) + Z

*A **α*(*u, u*^(*′*))*Q*(d*u*^(*′*)*|**u*)d*µ*(*u*)*.*


---

## Page 93

*8.2. MONTE CARLO TECHNIQUES *93

We discuss the two parts of this sum one after another. We ﬁrst have Z

*B **δ*(*A** −**u*) Z

*X *(1* −**α*(*u, u*^(*′′*)))*Q*(d*u*^(*′′*)*|**u*)d*µ*(*u*)

= Z

*X ***1***A**∩**B*(*u*) Z

*X *(1* −**α*(*u, u*^(*′′*)))*Q*(d*u*^(*′′*)*|**u*)*g*(*u*)d*ν*(*u*)

= Z

*A **δ*(*B** −**u*) Z

*X *(1* −**α*(*u, u*^(*′′*)))*Q*(d*u*^(*′′*)*|**u*)d*µ*(*u*)*.*

Secondly, Z

*B*

Z

*A **α*(*u, u*^(*′*))*Q*(d*u*^(*′*)*|**u*)d*µ*(*u*)

= Z

*B*

Z

*A *min  1*, *^(*g*)^(()^(*u*)^(*′*)^())^(*q*)^(()^(*u*)^(*|*)^(*u*)^(*′*)^())

*g*(*u*)*q*(*u*^(*′*)*|**u*)

 *q*(*u*^(*′*)*|**u*)d*ν*(*u*^(*′*))^(*g*)^(()^(*u*)^())

*c *d*ν*(*u*)

= Z

*B*

Z

*A *min  *g*(*u*)*q*(*u*^(*′*)*|**u*)*, g*(*u*^(*′*))*q*(*u**|**u*^(*′*)) 	 d*ν*(*u*^(*′*))^(1)

*c*^(d)^(*ν*)^(()^(*u*)^())

= Z

*A*

Z

*B *min  1*, *^(*g*)^(()^(*u*)^())^(*q*)^(()^(*u*)^(*′*)^(*|*)^(*u*)^())

*g*(*u*^(*′*))*q*(*u**|**u*^(*′*))

* g*(*u**′*)

*c q*(*u**|**u*^(*′*))d*ν*(*u*)d*ν*(*u*^(*′*))

= Z

*A*

Z

*B **α*(*u*^(*′*)*, u*)*Q*(d*u**|**u*^(*′*))d*µ*(*u*^(*′*))*.*

Combining these two results gives us detailed balance.

We ﬁnish by giving typical examples for proposal kernels* Q* used in Metropolis–Hastings MCMC.

**Example 8.2.5** (Independence Sampler)**.** Let* ρ** ∈*Prob(*X**,** BX*). The Metropolis-Hastings algorithm with proposal kernel

*Q*(*·|**u*) =* ρ *(*u** ∈X*)

is called independence sampler. The acceptance probability is given by

*α*(*u, u*^(*′*)) = min  1*, *^(*g*)^(()^(*u*)^(*′*)^())^(*q*)^(()^(*u*)^())

*g*(*u*)*q*(*u*^(*′*))

 *,*

where* q* = d*ρ/*d*ν*. In a Bayesian inverse problem with prior* µ*0* ∈*Prob(*X**,** BX*) and likelihood* L*(*f**n**|·*), we can choose* ρ* :=* ν* :=* µ*0. In this case, the acceptance probability simpliﬁes to

*α*(*u, u*^(*′*)) = min  1*, *^(*L*)^(()^(*f*)^(*n*)^(*|*)^(*u*)^(*′*)^())

*L*(*f**n**|**u*)

 *.*

Please note that the independence sampler proposes moves independently of the current position. This does not imply that the generated samples are independent. The acceptance step couples the samples.

**Example 8.2.6** (Random Walk; Metropolis et al. 1953 [27])**.** Let* ρ** ∈*Prob(*X**,** BX*) have a symmetric density* q*^(*′*)^( )= d*ρ/*d*ν*, i.e.* q*^(*′*)^( )=* q*^(*′*)(*−·*). The Metropolis-Hastings algorithm with proposal kernel *Q*(*·|**u*) =* ρ*(*· −**u*) (*u** ∈X*)


---

## Page 94

94 *CHAPTER 8. FUNCTION SPACE PRIORS AND MONTE CARLO*

is called Random Walk Metropolis sampler. The acceptance probability is given by

*α*(*u, u*^(*′*)) = min  1*, *^(*g*)^(()^(*u*)^(*′*)^())

*g*(*u*)

 *.*

Note that the acceptance probability is independent of the proposal distribution; indeed, it cancels:* q*(*u**|**u*^(*′*)) =* q*^(*′*)(*u** −**u*^(*′*)) =* q*^(*′*)(*u*^(*′*)^(* *)*−**u*) =* q*(*u*^(*′*)*|**u*).

**Example 8.2.7** (Preconditioned Crank–Nicolson MCMC; Cotter et al. 2013 [16])**.** Let* X *be a separable Hilbert space and let* µ*0 = N(0*,** C*)* ∈*Prob(*X**,** BX*) for some suitable operator *C* :* X →X*. We consider the (BIP) with prior* µ*0 and likelihood* L*(*f**n**|·*). Let* β** ∈*(0*,* 1) The Metropolis-Hastings algorithm with proposal kernel

*Q*(*·|**u*) := N( p

1* −**β*^(2)*u, β*^(2)*C*)

is called preconditioned Crank–Nicolson algorithm (pCN-MCMC). The acceptance proba- bility is given by

*α*(*u, u*^(*′*)) = min  1*, *^(*L*)^(()^(*f*)^(*n*)^(*|*)^(*u*)^(*′*)^())

*L*(*f**n**|**u*)

 *.*

This method is particularly useful in high- and inﬁnite dimension, where the random walk algorithm cannot be applied. Proving that* α* is the correct acceptance probability is rather simple in ﬁnite dimensions, not quite as easy in inﬁnite dimensions. The method is referred to as pCN MCMC as the proposal can be derived as a Crank– Nicolson discretisation of some S(P)DE.


---

## Page 95

# **Bibliography**

[1] Y. A. Abramovich and C. D. Aliprantis,* An Invitation to Operator Theory*, Graduate Studies in Mathematics, American Mathematical Society, 2002.

[2] R. A. Adams and J. J. F. Fournier,* Sobolev Spaces*, Elsevier Science, Singapore, 2003.

[3] S. Agapiou, O. Papaspiliopoulos, D. Sanz-Alonso, and A. M. Stuart,* Importance sampling: intrinsic dimension and computational cost*, Statist. Sci., 32 (2017), pp. 405–431.

[4] C. D. Aliprantis and K. Border,* Inﬁnite Dimensional Analysis: A Hitchhiker’s Guide*, Springer, 2006.

[5] L. Ambrosio, N. Fusco, and D. Pallara,* Functions of Bounded Variation and Free Dis- continuity Problems*, Clarendon Press, 2000.

[6] R. B. Ash and C. A. Dol´eans-Dade,* Probability & Measure Theory*, Harcourt Academic Press, 2000.

[7] A. B. Bakushinskii,* Remarks on the choice of regularization parameter from quasioptimality and relation tests*, Zhurnal Vychislitel’no¨ı Matematiki i Matematichesko¨ı Fiziki, 24 (1984), pp. 1258–1259.

[8] H. H. Bauschke and P. L. Combettes,* Convex Analysis and Monotone Operator Theory in Hilbert Spaces*, 2011.

[9] M. Benning and M. Burger,* Modern regularization methods for inverse problems*, Acta Numerica, 27 (2018), pp. 1–111.

[10] P. Billingsley,* Probability and Measure*, John Wiley and Sons, second ed., 1986.

[11] V. I. Bogachev,* Gaussian measures*, vol. 62 of Mathematical Surveys and Monographs, Amer- ican Mathematical Society, Providence, RI, 1998.

[12] B. Bollob´as,* Linear Analysis: An Introductory Course*, Cambridge University Press, Cam- bridge, second ed., 1999.

[13] K. Bredies and D. A. Lorenz,* Mathematical Image Processing*, Springer, 2018.

[14] M. Burger and S. Osher,* Convergence rates of convex variational regularization*, Inverse Problems, 20 (2004), p. 1411.

[15] ,* A guide to the tv zoo*, in Level-Set and PDE-based Reconstruction Methods, M. Burger and S. Osher, eds., Springer, 2013.

[16] S. L. Cotter, G. O. Roberts, A. M. Stuart, and D. White,* MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster*, Statist. Sci., 28 (2013), pp. 424– 446.

95


---

## Page 96

96 *BIBLIOGRAPHY*

[17] R. T. Cox,* Probability, frequency and reasonable expectation*, American Journal of Physics, 14 (1946), pp. 1–13.

[18] N. Dunford and J. T. Schwartz,* Linear Operators, Part 1: General Theory*, Wiley Inter- science Publishers, 1988.

[19] I. Ekeland and R. T´emam,* Convex Analysis and Variational Problems*, 1976.

[20] H. W. Engl, M. Hanke, and A. Neubauer,* Regularization of inverse problems*, vol. 375, Springer Science & Business Media, 1996.

[21] C. W. Groetsch,* Stable approximate evaluation of unbounded operators*, Springer, 2006.

[22] W. K. Hastings,* Monte Carlo Sampling Methods Using Markov Chains and Their Applica- tions*, Biometrika, 57 (1970), pp. 97–109.

[23] J. Hunter and B. Nachtergaele,* Applied Analysis*, World Scientiﬁc Publishing Company Incorporated, 2001.

[24] J. A. Iglesias, G. Mercier, and O. Scherzer,* A note on convergence of solutions of total variation regularized linear inverse problems*, Inverse Problems, 34 (2018), p. 055011.

[25] A. Klenke,* Probability Theory: A comprehensive Course*, Springer, 2014.

[26] J. Leao, D, M. Fragoso, and P. Ruffino,* Regular conditional probability, disintegration of probability and Radon spaces*, Proyecciones, 23 (2004), pp. 15–29.

[27] N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, and E. Teller, *Equation of State Calculations by Fast Computing Machines*, J. Chem. Phys., 21 (1953), pp. 1087–1092.

[28] A. W. Naylor and G. R. Sell,* Linear Operator Theory in Engineering and Science*, Springer Science & Business Media, 2000.

[29] Y. V. Prokhorov,* Convergence of random processes and limit theorems in probability theory*, Theory of Probability & Its Applications, 1 (1956), pp. 157–214.

[30] C. P. Robert and G. Casella,* Monte Carlo Statistical Methods*, Springer, 2004.

[31] L. I. Rudin, S. Osher, and E. Fatemi,* Nonlinear total variation based noise removal algo- rithms*, Physica D: Nonlinear Phenomena, 60 (1992), pp. 259–268.

[32] W. Rudin,* Functional Analysis*, International series in pure and applied mathematics, McGraw- Hill, 1991.

[33] K. Saxe,* Beginning Functional Analysis*, Springer, 2002.

[34] O. Scherzer, M. Grasmair, H. Grossauer, M. Haltmeier, and F. Lenzen,* Variational Methods in Imaging*, Springer, 2009.

[35] A. M. Stuart,* Inverse problems: a Bayesian perspective*, Acta Numerica, 19 (2010), pp. 451– 559.

[36] T. Tao,* Epsilon of Room, One*, vol. 1, American Mathematical Soc., 2010.

[37] E. Zeidler,* Applied Functional Analysis: Applications to Mathematical Physics*, vol. 108 of Applied Mathematical Sciences Series, Springer, 1995.

[38] ,* Applied Functional Analysis: Main Principles and Their Applications*, vol. 109 of Applied Mathematical Sciences Series, Springer, 1995.


---
