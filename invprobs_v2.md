## Page 1

# **Inverse Problems**

Dirk Lorenz

Summer term 2022

### **Contents**

**Introductory remarks 2**

**1 Introduction and motivation 4**

**2 Examples and basic notions 10**

**3 Hilbert spaces 14**

**4 The singular value decomposition and the pseudo-inverse 18**

**5 Regularization 23**

**6 Tikhonov regularization 30**

**7 Spectral regularization 37**

**8 Parameter choice and error estimates 42**

**9 Convergence rates and smoothness spaces 47**

**10 Convergence rates for spectral regularization 52**

**11 Iterative regulariztion 58**

**12 A Bayesian perspective on regulariztion 65**

**13 Discretization by projection 70**

1


---

## Page 2

### **Introductory remarks**

These are the lecture notes for the lecture “Inverse Problems” I held in the summer term 2023 at TU Braunschweig. The lecture is aimed at students from mathematics, financial mathematics, com- putational science and engineering as well as data science. Solid knowledge in analysis and linear algebra is needed. Helpful would be a background in functional analysis, especially the notions of Hilbert space and linear operators, but we will also provide a little background on these topics in the course. Inverse problems are problems where one want to find some cause which can only measured indirectly, i.e. one can only observe the eﬀects, but not the cause itself. Hence, it is quite applied as a math topic, but still one can do serious mathematics and the charm of the field lies in the tight connection of theoretical results and real world applications. The books on inverse problems one can find also vary from quite theoretical to very applied. Here is a short commented list of books:

1. The books Engl u. a. [1996]; Rieder [2013] are on the theo- retical side of inverse problems. While they also present applications, they focus on the underlying theory. The latter one (Rieder [2013]) is in German and written in the style of a textbook. The more recent lecture notes Clason [2020] are also quite theoretical.

2. The older Groetsch [1993] is also written as a text book, fo- cuses on theory, but targets readers with less background in mathematics.

3. The book Moura Neto und da Silva Neto [2012] uses less mathematics and also targets students with less background in math. It also does not focus that much on theory.

Braunschweig, June 15, 2023 Dirk Lorenz d.lorenz@tu-braunschweig.de

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 2


---

## Page 3

[Clason 2020] Clason, Christian:* Regularization of inverse prob- lems*. arXiv preprint arXiv:2001.00617. 2020

[Engl u. a. 1996] Engl,Heinz W. ; Hanke,Martin ; Neubauer, Andreas:* Regularization of inverse problems*. Bd. 375. Springer Science & Business Media, 1996

[Groetsch 1993] Groetsch, Charles:* Inverse problems in the mathematical sciences*. Bd. 52. Springer, 1993

[Moura Neto und da Silva Neto 2012] Moura Neto, Fran- cisco D. ; Silva Neto, Antônio José da:* An introduction to inverse problems with applications*. Springer Science & Business Media, 2012

[Rieder 2013] Rieder, Andreas:* Keine Probleme mit inversen Prob- lemen: eine Einführung in ihre stabile Lösung*. Springer-Verlag, 2013

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 3


---

## Page 4

13.04.2023, VL 1-1

### **1 Introduction and motivation**

The central topic of this lecture are inverse and ill-posed problems. Both the terms “inverse” and “ill-posed” are not clearly defined up to now (and will be hard to pin down exactly). Instead of defining them right away, we start with a motivating example.

*Example* 1.1 (Diﬀerentiation)*.* The problem of finding the derivative *g*^(*′*)^( )of a given function* g* is quite straightforward as long as symbolic computations are considered. However, finding the slope of a function that is not given as analytic expression, but can only be evaluated through a black box, is more involved. We will show, that it is even inverse and ill-posed in some sense. Mathematically, we would like to invert the operator* A* which takes a function* f* (for simplicity defined on [0, 1]) to its integral, i.e.* A f* =* g* with

*A f* (*x*) :=* g*(*x*) := Z* x*

0* *^(*f*)^( ()^(*t*)^())^(d)^(*t*)^(.)

Hence, the task is: Given some* g*, find* f* such that* A f* =* g*. We would like to measure errors in the data* g* and also in the reconstruction* f* and hence, we introduce norms for these quantities. We use the following norms:

*∥**f** ∥**C* :=* ∥**f** ∥*∞:= max* {|** f* (*x*)*| |** x** ∈**I**} ∥**f** ∥**C*1 :=* ∥**f** ∥*∞+* ∥**f** *^(*′*)*∥*∞

*∥**f** ∥**Ck* := *k*
## * *∑
 *l*=0 *∥**f* ^(()^(*l*)^())*∥*∞.

In fact,these norms turn the appropriate vectorspaces into normed spaces: For an interval* I* let

*C*(*I*) :=* {** f* :* I** →***R*** |** f* continuous*}* ,

*C*^(1)(*I*) :=* {** f* :* I** →***R*** |** f* continuously diﬀerentiable*}* ,

*C*^(*k*)(*I*) :=* {** f* :* I** →***R*** |** f k*-times continuously diﬀerentiable*}* .

We may omit the argument* I* and just write* C* and* C*^(*k*)^( )if* I* is clear from the con- text or does not play a role. We also de- note* C*^(0)^( )=* C* which is consistent with the notation for* C*^(*k*). These norms in- duce a notion of convergence: We say that* fn** →**f* in* C*^(*k*)^( )if* ∥**fn** −**f** ∥**Ck** →*0. Convergence in* C* is exactly uniform convergence and convergence in* C*^(*k*)

means that the functions as well as their first* k* derivatives converge uniformly. We can model the operator* A* as a map between various spaces, e.g. we can write* A* :* C*([0, 1])* →**C*([0, 1]). Diﬀerentiation is a left inverse of* A*: We have that* DA f* (*x*) =* D*(^(R)^(* x *)0* *^(*f*)^( ()^(*t*)^())^(d)^(*t*)^() =)^(* f*)^( ()^(*x*)^())^(. In )It’s not a right inverse,since* AD f* (*x*) = R* x *0* *^(*f*)^(* ′*)^(()^(*x*)^())^(d)^(*x*)^( =)^(* f*)^( ()^(*x*)^())^(* −*)^(*f*)^( ()^(0)^())^(. )this sense, the problem of calculating the derivative is the inverse problem to calculating the integral. Now let us argue that the inverse problem of diﬀerentiation is ill-posed while the* direct* problem of integration is well posed: The map* A* :* C*([0, 1])* →**C*([0, 1]) is linear and bounded. Linearity follows from the known rules for integrals and boundedness is

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 4


---

## Page 5

13.04.2023, VL 1-2

seen as follows: For a function* f** ∈**C* we have

*∥**A f** ∥**C* = max  *| *Z* x*

0* *^(*f*)^( ()^(*t*)^())^(d)^(*t*)^(*| |*)^( 0)^(* ≤*)^(*x*)^(* ≤*)^(1 )

*≤*max Z* x*

0* *^(*|*)^(* f*)^( ()^(*t*)^())^(*|*)^(d)^(*t*)^(* |*)^( 0)^(* ≤*)^(*x*)^(* ≤*)^(1 ) *≤*max*|** f* (*x*)*|* =* ∥**f** ∥**C*.

This shows that* A* is bounded and even that the operator norm of *A* fulfills* ∥**A**∥≤*1. How about continuity of the inverse operation? Consider* g *with* g*(0) = 0 and* g*^(*′*)^( )=* f* (i.e,* A f* =* g*) and let us perturb* g* slightly to

*g*^(*δ*)(*x*) =* g*(*x*) +* δ* sin(*nx*)

for some* δ** >* 0 and* n** ∈***N**. Then we have

(*g*^(*δ*))^(*′*)(*x*) =* g*^(*′*)(*x*) +* δ**n* cos(*nx*) =:* f** *^(*δ*)(*x*).

Hence, we have* A f* =* g* and* A f** *^(*δ*)^( )=* g*^(*δ*). Moreover, we easily see that

*∥**g** −**g*^(*δ*)*∥**C* =* δ*

*∥**f** −**f** *^(*δ*)*∥**C* = max* {**δ**n* cos(*nx*)* |* 0* ≤**x** ≤*1*}* =* n**δ*.

If we couple* δ* = 1/^(*√*)*n* we get

*∥**g** −**g*^(*δ*)*∥*=* δ* = 1 *√**n n**→*∞ *−→*0, but *∥**f** −**f** *^(*δ*)*∥**C* =* n**δ* =* *^(*√*)

*n *^(*n*)^(*→*)^(∞ )*−→*∞.

This shows that small perturbations in* g* may lead to large pertur- bations in the derivative* f* =* g*^(*′*), and hence, taking the derivative is unstable, and thus ill-posed. At first, this may seem like a hopeless situation when is comes to numerical diﬀerentiation. We always have some round-oﬀerror, so does that mean that numerical diﬀerentiation can not work? Assume that* g* : [0, 1]* →***R** is a diﬀerentiable function, but we don’t have a formula for* g* but for every* x* we get the value* g*(*x*). How can we find* g*^(*′*)(*x*)? The simplest idea may be to use a central diﬀerence quotient

*g*(*x* +* h*)* −**g*(*x** −**h*)

2*h *=:* Dhg*(*x*).

This operator* Dh* is again a linear operator, and, in some sense, an approximation of the derivative* D*. Using Taylor expansion, we get for* g** ∈**C*^(2)^( )that

*g*(*x** ±** h*) =* g*(*x*)* ±** hg*^(*′*)(*x*) +* *^(*g*)^(*′′*)^(()^(*ξ*)^(*±*)^())

2 *h*^(2)

with some* ξ**±* between* x* and* x* +* h* and* x** −**h*, respectively. This gives us an estimate

*∥**g*^(*′ *)*−**Dhg**∥*∞= max n *|**g*^(*′*)(*x*)* − * *g*^(*′*)(*x*) +* *^(*h*)

2 *g*^(*′′*)(*ξ*+)*−**g*^(*′′*)(*ξ**i*)

2  *| *o *≤*^(*h*)

2^(*∥*)^(*g*)^(*′′∥*)^(∞)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 5


---

## Page 6

13.04.2023, VL 1-3

This is an estimate for the error we make when we use an ap- proximation of the derivative, and hence, we call this error the *approximation error*. Now we assume that we have an error in our available data, i.e. we do not get the exact values* g*(*x*), but slightly perturbed data

*g*^(*δ*)(*x*) =* g*(*x*) +* w*(*x*) with *∥**w**∥*∞*≤**δ*.

We are interested in the* total error*, i.e. for* f* =* g*^(*′*)^( )we would like to compute or estimate

*∥**f** −**Dhg*^(*δ*)*∥*∞,

which is the error between the unknown exact derivative* f* and the quantity* Dhg*^(*δ*)^( )we can actually compute. This error has a natural decomposition as

*∥**f** −**Dhg*^(*δ*)*∥*∞*≤∥**f** −**Dhg**∥*∞+* ∥**Dhg** −**Dhg*^(*δ*)*∥*∞

where we inserted the term* ±**Dhg* and used the triangle inequality. The first term on the right is the approximation error which we just estimated already. The second term is called* data error* and is also simple to estimate:

*∥**Dhg** −**Dhg*^(*δ*)*∥*∞=* ∥**Dhw**∥*∞= max n *|** *^(*w*)^(()^(*x*)^(+)^(*h*)^())^(*−*)^(*w*)^(()^(*x*)^(*−*)^(*h*)^())

2*h **| *o *≤*^(*δ*)

*h*^(.)

So we get for the total error

*∥**f** −**Dhg*^(*δ*)*∥*∞*≤**h** *^(*∥*)^(*g*)^(*′′∥*)^(∞)

2 +* *^(*δ*)

*h*^(.)

Overall, we see a very typical behavior:

*h*

*∥**f** −**Dhg*^(*δ*)*∥*

approximation error

data error

total error

For a fixed* noise level** δ*, there is a tradeoﬀbetween large and small parameters* h*: For small* h* the data error gets big, while for large* h *the approximation error gets big. Somewhere in the middle there is an optimum and a little calculus shows that the parameter* h *which minimizes our upper bound for the total error is

*h*^(*∗*)= q

2*δ **∥**g*^(*′′*)*∥*∞^(.)

With this value we get

*∥**f** −**Dh**∗**g*^(*δ*)*∥*∞*≤ *q

2*∥**g*^(*′′*)*∥*∞*δ*.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 6


---

## Page 7

13.04.2023, VL 1-4

We see that even when the operation of diﬀerentiation is ill-posed, we can still get a stable approximation of the derivative from noisy data. However, the error is not as small as one could have hoped: We obtain an error in the order of *√*

*δ* for noise of size* δ*. △ It is quite instructive, to see that higher smoothness can lead to a better total error: Assume that* g*^(*′′′*)^( )exists, derive a better estimate for the approximation error (by using more terms of the Taylor expansion, calculate the optimal* h* and deduce that the total error can be of order* δ*^(2/3)^( )in this case.

Here are a few important takeaways from the above example:

• The total error in the solution of an inverse problem decom- poses into an approximation error and a data error: Good approximation leads to a amplification of the error in the data, and keeping the data error small needs a large approx- imation error.

• For a fixed noise level* δ* there is a tradeoﬀbetween approxi- mation error and data error.

• A helpful estimate of the approximation errorneeds a smooth- ness assumption on the unknow data (in our case we needed that* g*^(*′′*)^( )=* f** *^(*′*)^( )exists and is bounded).

• The total error is, even in the best case, not of the order of the error in the data, but worse.

*Remark* 1.2*.* Our results is actually useful in practice: If you want to evaluate derivatives of functions numerically by a finite diﬀer- ence approximation, one usually uses* h* =* *^(*√*)eps where eps is the machine precision. For double precision numbers eps* ≈*10^(*−*)^(16), so* h* = 10^(*−*)^(8)^( )is recommended. Here is an example (written in Python):

# import libraries import numpy as np import matplotlib.pyplot as plt

# define functions def f(x): return np.sin(np.log(x)**4)**3

def fprime(x): return 3*np.sin(np.log(x)**4)**2*np.cos(np.log(x)**4)*4*np.log(x)**3/x

def Df(x,h): return (f(x+h)-f(x-h))/2/h

# some x values x = np.linspace(0.1,4,200,dtype=’float64’) # plot function plt.plot(x,f(x)) plt.show()

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 7


---

## Page 8

13.04.2023, VL 1-5

# Choose some stepsize h = 1e-5 # plot derivative and approximation by finite differences fig, axs = plt.subplots(3) axs[0].plot(x, fprime(x)) axs[0].set_title(’true derivative’) axs[1].plot(x, Df(x,h)) axs[1].set_title(’derivative by finite differences’) axs[2].plot(x, fprime(x)-Df(x,h)) axs[2].set_title(’difference of the two’) plt.show()

# compare derivative and approximation for different values of h h = np.logspace(-2,-16,15) e = np.zeros(15) for k in range(0,15):

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 8


---

## Page 9

13.04.2023, VL 1-6

e[k] = np.max(np.abs(fprime(x)-Df(x,h[k])))

plt.loglog(h,e) plt.show()

# square root of machine precision is close to optimal h: print(’sqrt(eps) = ’, np.sqrt(np.finfo(x.dtype).eps))

sqrt(eps) = 1.4901161193847656e-08

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 9


---

## Page 10

02.05.2022, VL 2-1

### **2 Examples and basic notions**

The notion of “inverse problems” is vague and hard to pin down. What is a “problem” anyway? Well, a problem always has “data” and “solution”, i.e. something that is given, and something that is wanted. Solving the problem means, taking the data, do some computations and arrive at a solution. For every problem, there is an “inverse problem”, namely: Having some solution, what is the corresponding data that gave rise to this solution?

Data Solution

solving problem

inverse problem

In a more physical context one could frame inverse problems as follows:

An inverse problems asks for some* cause* that is behind a given* observation*.

Cause Observation

physical process

inverse problem

*Example* 2.1 (Parameter identification in PDEs)*.* Assume that you can observe the heat distribution* u* of some matter in a domain Ω at a given time* t* =* T** >* 0. What was the heat distribution at time *t* = 0? This is an inverse problem. To formulate it mathematically, we use the heat equation. The distribution of heat follows the partial diﬀerential equation

*ut*(*t*,* x*) = ∆*u*(*t*,* x*) in [0,* T*]* ×* Ω

*u*(0,* x*) =* u*0(*x*) in Ω

*∂**nu*(*t*,* x*) = 0 in [0,* T*]* ×** ∂*Ω.

The forward problem would be: Given the initial data* u*0, calculate *u*(*T*,* x*). The corresponding inverse problem is: Given a measure- ment of* u*(*T*,* x*), find initial data* u*0 that explains the measurement. In the context of partial diﬀerential equations one can formu- late numerous inverse problems. Consider the following problem:

*ut** −**Lu* =* f *in [0,* T*]* ×* Ω

*u*(0,* x*) =* u*0(*x*) in Ω

*∂**nu* =* g *on [0,* T*]* ×** ∂*Ω

with some diﬀerential operator* L*, initial data* u*0, source term* f*, and boundary data* g*. The forward problem would be to compute

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 10


---

## Page 11

02.05.2022, VL 2-2

*u* from knowledge of* u*0,* f* and* g*, but there are various inverse problems and here are just two:

• Given* u*(*T*,* ·*),* f* and* g*, find* u*0.

• Given* u*, and* u*0, find* f* and* g*.

You can find more inverse problems easily. △

*Example* 2.2 (Computerized tomography)*.* The basic concept of CT is to measure the intensity of X-ray beams from a source with known intensity after passing through the body at a fixed plane. It is assumed that these beams travel on a straight line and their intensity is attenuated proportially to some material constant one is interested in reconstructing, which is usually the density. Denoting this material constant by* u* and* x* =* x*(*L*,* t*) the point in which the X-ray beam associated with the line* L* passes at time* t*, this can be modeled by the ordinary diﬀerential equation

*I*^(*′ *)*L*^(()^(*t*)^() =)^(* −*)^(*u*)^(()^(*x*)^(()^(*L*)^(,)^(* t*)^()))^(*IL*)^(()^(*t*)^())^(,)

so if* IL*(*T*) is measured for some time* T** >* 0 where the beam passed the object of interest,

*−*log * IL*(*T*)

*IL*(0)

 = Z* T*

0* *^(*u*)^(()^(*x*)^(()^(*L*)^(,)^(* t*)^()))^(d)^(*t*)^(.)

The left-hand side is known while the right-hand side constitutes the integral of* u* associated with the line* L* up to some factor. The principle of computed tomography is obtain these integrals by emitting and measuring X-ray beams along all possible lines. This is typically done by placing an X-ray point source on one side of the object, installing a detector array on the other and side rotating source and detector simultaneously, giving the intensities for all lines passing through a region of interest. Mathematically, the reconstruction problem in CT is now to compute* u* given all of its line integrals. Placing the origin in the center of the object of interest of radius* R** >* 0, a line* L* passing through the domain can uniquely be associated an angle* θ** ∈ *[*−**π*/2,* π*/2[ and oﬀset* s** ∈***R** at which the line crosses the axis spanned by* x*0(*θ*) = (cos(*θ*), sin(*θ*)).

*s*

*x*0 *θ **L*(*s*,* θ*)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 11


---

## Page 12

02.05.2022, VL 2-3

We denote the corresponding line by* L* =* L*(*s*,* θ*). Given* u*^(0 ): ]*−**R*,* R*[* ×* [*−**π*/2,* π*/2[* →***R**, the CT reconstruction problem is to find* u*^(† ):** R**^(2)^(* *)*→***R** such that

(*Ru*)(*s*,* θ*) := Z

*L*(*s*,*θ*)* *^(*u*)^(†d)^(*x*)^( =)^(* u*)^(0)^(()^(*s*)^(,)^(* θ*)^() )for all (*s*,* θ*)

where the mapping* R* is called the* Radon transform*. The inverse problem related to the Radon transform is the main problem in tomographic reconstruction. △

Here is the classical definition of “ill posedness” which is due to Hadamard and dates back to the 1920s.

**Definition 2.3** (Well- and ill-posed problems)**.** Let* X* and* Y* be topological spaces and* A* :* X** →**Y*. We say that the problem “solve *Ax* =* y*” is* well posed* if

(a) The equation* Ax* =* y* has a solution for every* y** ∈**Y*, and

(b) this solution is unique, and

(c) the solution depends continuously on the data. In short: the problem is well posed if the inverse* A*^(*−*)^(1 ):* Y** →**X* exists and is continuous. If one of these conditions is not fulfilled, we call the problem *ill-posed*.

In this lecture we will treat* linear* inverse problems. We will always assume* Hilbert spaces** X* and* Y* and a linear, bounded op- erator* A* :* X** →**Y*.The space* X* is the* solution space* and* Y* is the We will denote the set of linear and bounded operators from* X* to* Y* by *L*(*X*,* Y*). *data space*. The forward problem is “given* x*, evaluate* Ax*”. Since inverse problems always assume measurement data with error, the inverse problem is:

Given measured data* g*^(*δ*)^(* *)*∈**Y* which fulfills

*∥**A f* ^(†)^(* *)*−**g*^(*δ*)*∥≤**δ*

for some known* noise level** δ* and an* unknown true solu- tion** f* ^(†), find a good approximation to* f* ^(†).

For linear operators one can characterize the ill-posedness of the problem “Solve* Ax* =* y*” quite explicitly:

(a)* Ax* =* y* has a solution for every* y* exactly if* A* is surjective (also called onto), i.e. if rg(*A*) =* Y*.

(b) Solutions of* Ax* =* y* are unique exactly if* A* is injective (also called one-to-one), i.e. if ker(*A*) =* {*0*}*

(c) Solutions of* Ax* =* y* depend continuously on* y* exactly if* A*^(*−*)^(1)

is bounded.

A more quantitative way to describe ill-posedness of a problem is the notion of* condition* or conditioning of a problem. We say that a problem is

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 12


---

## Page 13

02.05.2022, VL 2-4

**well condtioned** if small changes in the data lead to small changes in the solution

**ill conditioned (or badly conditioned)** if small changes in the data lead to large errors in the solution.

*Example* 2.4 (Function evaluation)*.* Here the problem is simply “given* x* evaluate* y* =* f* (*x*). We consider a perturbation* x* + ∆*x* of* x*. The solution changes to* y* + ∆*y* =* f* (*x* + ∆*x*) and by linearization we get

*|*∆*y**| ≈|** f** *^(*′*)(*x*)*||*∆*x**|*.

So we say that the problem is ill-conditioned (with respect to absolute errors) if* |** f** *^(*′*)(*x*)*|* is large. If we consider relative errors, we get

*|*∆*y**|*

*|**y**| *^(*≈|*)^(* f*)^(* ′*)^(()^(*x*)^())^(*||*)^(∆)^(*x*)^(*|*)

*|** f* (*x*)*| *=* *^(*|*)^(* f*)^(* ′*)^(()^(*x*)^())^(*||*)^(*x*)^(*|*)

*|** f* (*x*)*| |*∆*x**|*

*|**x**|*

so we say that the problem is ill-conditioned with respect to relative error if* |** f** *^(*′*)(*x*)*||**x**|*/*|** f* (*x*)*|* is large. △

*Example* 2.5 (Solving linearequations)*.* Consideran invertible square matrix* A* and the problem: given* b*, find the solution of* Ax* =* b*. Changing the data to* b* + ∆*b*, the new solution fulfills

*A*(*x* + ∆*x*) =* b* + ∆*b*.

We see that the change in the solution is ∆*x* =* A*^(*−*)^(1)∆*b*. Using the operator norm we get that* ∥*∆*x**∥≤∥**A*^(*−*)^(1)*∥∥*∆*b**∥*and we see that the problem is ill-conditioned (with respect to absolute errors) if *∥**A*^(*−*)^(1)*∥*is large. If we consider relative errors, we get (using* ∥**b**∥*= *∥**Ax**∥≤∥**A**∥∥**x**∥*

*∥*∆*x**∥*

*∥**x**∥*^(=)^(* ∥*)^(*A*)^(*−*)^(1)^(∆)^(*b*)^(*∥*)

*∥**x**∥ ≤*^(*∥*)^(*A*)^(*−*)^(1)^(*∥∥*)^(∆)^(*b*)^(*∥*)

*∥**x**∥ ∥**b**∥ ∥**b**∥*^(*≤∥*)^(*A*)^(*−*)^(1)^(*∥∥*)^(*A*)^(*∥∥*)^(∆)^(*b*)^(*∥*)

*∥**b**∥*^(.)

△

**Definition 2.6.** The condition number of a square matrix* A* is

cond(*A*) = * ∥**A**∥∥**A**−*1*∥ *: if* A* is invertible ∞ : else.

Strictly speaking, the problem “Solve* Ax* =* b*” is only ill-posed if* A* is not invertible. Practically, a large condition number of* A *will still lead to a large increase of the error, so we consider the problem still ill conditioned or badly conditioned if the condition number is large. To understand all these things better, we introduce the notion of Hilbert spaces, linear bounded operators between these spaces and the singular value decomposition of these operators.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 13


---

## Page 14

27.04.2023, VL 3-1

### **3 Hilbert spaces**

Now we introduce abstract notions of vector spaces that we will use throughout the lecture. The main notions are* Hilbert spaces*. In a nutshell, they are spaces which behave the sae as the euclidean space** R**^(*n*)^( )when it comes to geometric and analytical structures like length, and orthogonality. First we define inner products:

**Definition 3.1.** A real* inner product space** X* is a real vector space that is equipped with an* inner product** ⟨·*,* ·⟩*:* X** ×** X** →***R** which fulfills

*⟨**x*,* y**⟩*=* ⟨**y*,* x**⟩ ⟨**α**x* +* y*,* z**⟩*=* α** ⟨**x*,* z**⟩*+* ⟨**y*,* z**⟩ ⟨**x*,* x**⟩**>* 0 if* x** ̸*= 0.

An inner product always induces a norm* ∥**x**∥*= p

*⟨**x*,* x**⟩ *(which can be shown using the Cauchy-Schwarz inequality* ⟨**x*,* y**⟩≤ ∥**x**∥∥**y**∥*).Since any norm induces a notion of convergence by saying On a Hilbert space* X* we will denote the norm by* ∥**x**∥**X*, but sometimes the subscript may be dropped, when the norm is clear from the context. that* xn n**→*∞ *−→**x* if* ∥**xn** −**x**∥→*0 we can also talk about* complete- ness* and this is the property that turns inner product spaces into Hilbert spaces.

**Definition 3.2.** A real* Hilbert space* is a complete real inner product space, i.e. a real inner product space with the property that every Cauchy sequence in the space converges, i.e. for every sequence* xn *in* X* which fulfills

*∀**ϵ** >* 0* ∃**N** ∀**m*,* n** ≥**N* :* ∥**xn** −**xm**∥≤**ϵ*

there exists some limit* x*^(*∗*), i.e.* xn n**→*∞ *−→**x*^(*∗*)(i.e. we have* ∥**xn** − **x*^(*∗*)*∥*^(*n*)^(*→*)^(∞ )*−→*0).

A little bit sloppy one could say that a Hilbert space is an inner product space that “you can’t leave with ‘convergent sequences’”

*Example* 3.3*. *1. The space** R**^(*n*)^( )with standard inner product

*⟨**x*,* y**⟩*:=* x** ·** y* :=* x*^(*T*)*y* = *n*
## * *∑
 *i*=1 *xiyi*

is a Hilbert space of dimension* n*. The corresponding norm

*∥**x**∥*= * n *∑ *i*=1 *x*^(2 )*i*

1/2 is the well known euclidean norm. The

standard inner product on** R**^(*d*)^( )is also called* dot product* and we will also use the notation* x** ·** y* for it. The euclidean norm of a vector* x* will also be denoten by* |**x**|*.

2. An example of an infinite dimensional Hilbert space is

*ℓ*^(2 ):=

(

(*xn*)*n*=1,2,...



∞
##  ∑
 *i*=1 *x*^(2 )*n** *^(*<*)^( ∞)

)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 14


---

## Page 15

27.04.2023, VL 3-2

of square summable sequences. When equipped with the inner product* ⟨**x*,* y**⟩*:= ∑^(∞ )*i*=1* *^(*xiyi*) ^(it is a Hilbert space. The )corresponding norm is denoted by

*∥**x**∥*2 :=  ∞
##  ∑
 *i*=1 *x*^(2 )*i *1/2 .

3. A diﬀerent example of an infinite dimensional Hilbert space is the* Lebesgue space** L*^(2)^( )of square integrable functions. For some domain Ω*⊂***R**^(*d*)^( )(i.e. a non-empty, connected and open subset of** R**^(*d*)) we can define

*L*^(2)(Ω) :=  *f* : Ω*→***R **

Z

Ω^(*f*)^( ()^(*x*)^())^(2d)^(*x*)^(* <*)^( ∞ )

and equipped with the inner product

*⟨**f*,* g**⟩**L*2 := Z

Ω^(*f*)^( ()^(*x*)^())^(*g*)^(()^(*x*)^())^(d)^(*x*)

this is a Hilbert space as well. The corresponding norm is denoted by

*∥**f** ∥**L*2 :=  ^(Z)

Ω^(*f*)^( ()^(*x*)^())^(2d)^(*x *)1/2 .

The are Sobolev spaces of higher order, i.e. for every* k** ∈***N **there is a Sobolev space* H*^(*k*)(Ω) which incorporates partial derivatives up to* k*-th order, but we will not define them here.

You may have noted that there is a prob- lem here: This is not a norm, since there are functions* f** ̸*= 0 with* ∥**f** ∥**L*2(Ω) = 0. This problem can be solved: These functions have the property of being ze- ros “almost everywhere” and one can “quotient out” all these functions from *L*^(2)(Ω). This means that one identifies two functions* f* and* g* if ^(R)^( )(* f** −**g*)^(2)^( )= 0. The full theory behind this can be found in books on real anaylsis or measure theory.

4. Slightly more complicated are the* Sobolev spaces* which gener- alize the Lebesgue space* L*^(2)^( )by also incorporating derivatives. The space* H*^(1)(Ω) is

*H*^(1)(Ω) :=

 

^(*f*)^( :)^( Ω)^(*→*)^(**R**)



Z

Ω *f* (*x*)^(2)d*x* + Z

Ω *|∇**f* (*x*)*|*^(2)d*x** <* ∞

 



and it is equipped with the inner product

*⟨**f*,* g**⟩**H*1 := Z

Ω *f* (*x*)*g*(*x*)d*x* + Z

Ω *∇**f* (*x*)* · ∇**g*(*x*)d*x*

where the second integral is over the dot product of the gradients. The corresponding* H*^(1)-norm is

*∥**f** ∥**H*1 :=



 Z

Ω^(*f*)^( ()^(*x*)^())^(2d)^(*x*)^( + )Z

Ω *|∇**f* (*x*)*|*^(2)d*x*





1/2

△

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 15


---

## Page 16

27.04.2023, VL 3-3

As important as the spaces are linear operators between these spaces. We will call a linear map* A* from one Hilbert space* X* to another* Y* an* operator* and say that an operator* A* is* bounded* if there exists a constant* C* such that* ∥**Ax**∥**Y** ≤**C**∥**x**∥**X* for all* x*.The Bounded operators are exactly the con- tinuous operators. infimum over all such constants is the* operator norm* of* A*, denoted by* ∥**A**∥*. Other ways to define the operator norm are

*∥**A**∥*= sup *∥**x**∥**X*=1 *∥**Ax**∥**Y* = sup *x**̸*=0

*∥**Ax**∥**Y*

*∥**x**∥**X *.

The set of all bounded linear operators from a Hilbert space* X *to another Hilbert space* Y* is denoted by* L*(*X*,* Y*). For every* A** ∈ **L*(*X*,* Y*) we have the so-called* adjoint* operator* A*^(*∗*)defined by

*∀**x** ∈**X*,* y** ∈**Y* :* ⟨**x*,* A*^(*∗*)*y**⟩*=* ⟨**Ax*,* y**⟩*.

Note that the adjoint maps* A* :* Y** →**X* and it is also bounded with *∥**A*^(*∗*)*∥*=* ∥**A**∥*. In** R**^(*n*), operators are just matrices and the adjoint with respect to the standard inner product is just the transpose of the matrix.

*Example* 3.4*. *1. Linear operators* A** ∈**L*(**R**^(*n*),** R**^(*m*)), i.e. from one euclidean space to another can be identified with matrices *A** ∈***R**^(*m*)^(*×*)^(*n*), i.e. the application of the operator* A* to a vec- tor* x* is given by matrix-vector multiplication* Ax*. In this case these is a simple expression for the operator norm of a matrix: It holds that

*∥**A**∥*= q

*λ*max(*A*^(*T*)*A*),

i.e. the operator norm is the square root of the largest eigen- value of the matrix* A*^(*T*)*A*. Note that* A*^(*T*)*A* is symmetric and positive definite and thus, only has real and eigenvalues greater or equal zero. The adjoint of matrix is given by the transposed matrix as it holds that

(*Ax*)* ·** y* = (*Ax*)^(*T*)^(* *)*·** y* =* x*^(*T*)*A*^(*T*)*y* =* x*^(*T*)(*A*^(*T*)*y*) =* x** ·* (*A*^(*T*)*y*).

2. A class of linearoperators between* L*^(2)^( )spaces* A** ∈**L*(*L*^(2)(Ω1),* L*^(2)(Ω2)) is given by so called* integral operators*. These are given by

*A f* (*y*) = Z

Ω1

*k*(*x*,* y*)* f* (*x*)d*x*

for some function* k* : Ω1* ×* Ω2* →***R**. For these operators to be well defined and bounded one needs that* k* is square integrable, i.e.* k** ∈**L*^(2)(Ω1* ×* Ω2). This can be seen by an ap- plication of the Cauchy-Schwarz inequality for the* L*^(2)^( )inner product as follows:

*∥**A f** ∥*^(2 )*L*^(2)^( = )Z

Ω2



 Z

Ω1

*k*(*x*,* y*)* f* (*x*)d*x*





2

d*y*.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 16


---

## Page 17

27.04.2023, VL 3-4

The inner integral is the* L*^(2)^( )inner product between* k*(*x*,* ·*) and* f* and hence, we have

*∥**A f** ∥*^(2 )*L*^(2)^(* ≤ *)Z

Ω2



 Z

Ω1

*k*(*x*,* y*)* f* (*x*)d*x*





2

d*y*

*≤ *Z

Ω2

Z

Ω1 *k*(*x*,* y*)^(2)d*x *Z

Ω1

*f* (*x*)^(2)d*x*d*y*

=* ∥**k**∥*^(2 )*L*^(2)(Ω1*×*Ω2)^(*∥*)^(*f*)^(* ∥*)^(2 )*L*^(2)(Ω1)^(.)

This also gives the upper bound* ∥**A**∥≤∥**k**∥**L*2(Ω1*×*Ω2) (which is usually not strict).

We compute the adjoint of an integral operator using Fubinis theorem to interchange the order of integrals

*⟨**A f*,* g**⟩**L*2(Ω2) = Z

Ω2

*A f* (*y*)*g*(*y*)d*y*

= Z

Ω2

Z

Ω1

*k*(*x*,* y*)* f* (*x*)d*xg*(*y*)d*y*

= Z

Ω1

*f* (*x*) Z

Ω2

*k*(*x*,* y*)*g*(*y*)d*y*

| {z } =*A*^(*∗*)*g*(*x*)

d*x* =* ⟨**f*,* A*^(*∗*)*g**⟩*.

△

There is another result that we will use frequently:

**Theorem 3.5** (Dominated convergence for series)**.*** Let** am*,*n** ∈***R ***with** am*,*n m**→*∞ *−→**a*^(*∗ *)*n*^(*. If there exists a sequence*)^(* bn*)* *^(*with*)^(* |*)^(*am*)^(,)^(*n*)^(*| ≤*)^(*bn*)* *^(*for all *)*m*,* n** and* ∑*n bn** <* ∞*, then it holds that*

## ∑
 *n am*,*n m**→*∞
###  *−→*∑
 *n a*^(*∗ *)*n*^(.)

Informally: If we have a sequence of series, where the sum- mands converge, we can pull the limit under the series if there is a dominating sequence which is summable.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 17


---

## Page 18

04.05.2023, VL 4-1

### **4 The singularvalue decomposition and the pseudo- inverse**

We will build the section on the spectral theorem for compact operators. Recall the notion of compact operator:

**Definition 4.1.** Some* A** ∈**L*(*X*,* Y*) is* compact*, if it holds that

(*xn*) bounded in* X *=*⇒*(*Axn*) has convergent subsequence in* Y*. Here are some equivalent descriptions of compact operators for those who know what weak convergence in Hilbert spaces mean:

1.* A *is compact if it maps bounded sets in* X* to precom- pact sets in* Y* (i.e. their closure is compact). (For bounded operators we only have that they map bounded sets to bounded sets.)

2.* A* is compact if it maps weakly convergent subsequences to strongly convergent ones, i.e. *Axn* is (strongly) convergent in *Y* whenever* xn* is weakly con- vergent in* X*. (For bounded op- erators we only have that they map strongly convergent se- quences to strongly convergent sequences and weakly conver- gent sequences to weakly con- vergent sequences.)

We will denote the set of compact operators from* X* to* Y *by* K*(*X*,* Y*). Directly from the definition we get that* K*(*X*,* Y*)* ⊂ **L*(*X*,* Y*).

*Example* 4.2*.* For* A** ∈**L*(*X*,* Y*) we can say:

1. If the range of* A* is finite dimensional and* A* is bounded,then *A* is compact, as bounded sequences in finite dimensional spaces always have convergent subsequences.

2. The identity id :* X** →**X* is always bounded, but only com- pact if* X* is finite dimensional.

3. If* A* is compact and* B* is bounded than* AB* is compact (if defined). Similarly,* BA* is compact (if defined).

4. Also the adjoint operator* A*^(*∗*)is compact if* A* is compact.

5. Finally, if* Kn* is a sequence of compact operators and we have that* ∥**Kn** −**K**∥→*0, then* K* is compact as well.

△

A class of non-trivial compact operators are* integral operators*:

*Example* 4.3*.* Let* X* =* L*^(2)(Ω1) and* Y* =* L*^(2)(Ω2) and* k** ∈**L*^(2)(Ω1* × *Ω2) and define the operator

*Kx*(*t*) = Z

Ω1

*k*(*s*,* t*)*x*(*s*)d*s*.

We have seen in Example 3.4 that* K* is bounded with operator norm *∥**K**∥≤∥**k**∥**L*2(Ω1*×*Ω2). However,* K* is also compact! To see this, note that we can ap- proximate the function* k* by “simple functions”, i.e. there is se- quence* kn* of functions of the form

*kn*(*s*,* t*) = ∑ *i*,*j **α**ij***1***Ei*(*s*)**1***Fj*(*t*)

for some disjoint sets* Ei** ⊂*Ω1 and* Fj** ⊂*Ω2 such that* kn* approx- Here we use** 1***E* for the so-called* char- acteristic function* of the set* E*, i.e. the function which is 1 on* E* and zero else- where.

imated* k*, more precisely* ∥**k** −**kn**∥**L*2(Ω1*×*Ω2)* →*0. The respective integral operators* Kn* all have finite dimensional range and hence, are compact. Moreover it holds that

*∥**K** −**Kn**∥≤∥**k** −**kn**∥**L*2(Ω1*×*Ω2)* →*0

and thus,* K* is compact as well. △

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 18


---

## Page 19

04.05.2023, VL 4-2

One central theorem for compact operators is the following:

**Theorem 4.4** (Spectral theorem for compact, selfadjoint operators)**. ***Let** X** be a Hilbert space and** K** ∈**K*(*X*,* X*)* be selfadjoint. Then there exists and orthonormal basis* (*un*)* of* cl rg(*K*)* and** λ**n** ∈***R*** \ {*0*}** such that*

### *Kx* = ∑
 *n **λ**n** ⟨**x*,* un**⟩**un*.

*If the dimension of* cl rg(*K*)* is infinite, we also have** λ**n** →*0*. *The proof can be found in H.W. Alt’s book “Linear functional analysis” where this theorem is Theorem 12.12. *Remark* 4.5*.* Plugging in* x* =* um*,we get that* Kum* = ∑ *n **λ**n** ⟨**um*,* un**⟩**un* =

*λ**mum* and we see that the* un* are actually eigenvectors of* K* foreigen- values* λ**n*. By convention, one sorts the eigenvalues by decreasing magnitude, i.e.* |**λ*1*| ≥|**λ*2*| ≥· · ·** >* 0.

From the spectral theorem we can deduce the existence of the singular value decomposition (SVD):

**Theorem 4.6** (Singularvalue decomposition)**.*** Forevery** K** ∈**K*(*X*,* Y*) *there exist*

*(i) an orthonormal basis* (*un*)* of* cl rg(*K*)* ⊂**Y**,*

*(ii) an orthonormal basis* (*vn*)* of* cl rg(*K*^(*∗*))* ⊂**X**,*

*(iii) numbers** σ*1* ≥**σ*2* ≥· · ·** >* 0

*such that for all** n*

*Kvn* =* σ**nun*, *and **K*^(*∗*)*un* =* σ**nvn*

*and for all** x** ∈**X*

### *Kx* = ∑
 *n **σ**n** ⟨**x*,* vn**⟩**un*.

*Proof.* Since* K*^(*∗*)*K* is selfadjoint and compact, we get from the spec- tral theorem the existence of* λ**n* and* vn* such that

*K*^(*∗*)*Kx* = ∑ *n **λ**n** ⟨**x*,* vn**⟩**vn*.

Since* λ**n**∥**vn**∥*^(2 )*X* ^(=)^(* ⟨*)^(*λ*)^(*nvn*)^(,)^(* vn*)^(*⟩*)^(=)^(* ⟨*)^(*K*)^(*∗*)^(*Kvn*)^(,)^(* vn*)^(*⟩*)^(=)^(* ⟨*)^(*Kvn*)^(,)^(* Kvn*)^(*⟩*)^(= )*∥**Kvn**∥*^(2 )*X** *^(*>*)^( 0)^( we get that)^(* λ*)^(*n*)^(* >*)^( 0)^(. Now we define)

*σ**n* = p

*λ**n*, and *un* = 1 *σ**n *^(*Kvn*)^(.)

Checking that the claimed equalities hold as well as checking orthonormality of the* un* is a routine calculation.

*Remark* 4.7*.* (a) We call (*σ**n*,* un*,* vn*) the* singular system* of* K*.

(b) We also get the singular value decomposition of* K*^(*∗*), namely

*K*^(*∗*)*y* = ∑ *n **σ**n** ⟨**y*,* un**⟩**vn*.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 19


---

## Page 20

04.05.2023, VL 4-3

(c) The* σ**n* are called* singular values*, the* un* are* left singular vectors *and the* vn* are* right singular vectors*.

(d) The singular vectors can be used to project onto the closures of the ranges of* K* and* K*^(*∗*), namely

*P*cl rg(*K*)*y* = ∑ *n **⟨**y*,* un**⟩**un*, *P*cl rg(*K**∗*)*x* = ∑ *n **⟨**x*,* un**⟩**un*.

(e) We have ∑*n**|⟨**x*,* vn**⟩|*^(2)^( )=* ∥**P*cl(rg(*K*))(*x*)*∥**X** ≤∥**x**∥**X*.

The singular value decomposition also allows to describe the boundary of the range:

**Theorem 4.8** (Picard condition)**.*** Let** K** ∈**K*(*X*,* Y*)* with singular system* (*σ**n*,* un*,* vn*)* and let** y** ∈*cl(rg(*K*))*. Then** y** ∈*rg(*K*)* exactly if*

## ∑
 *n*

*|⟨**y*,* un**⟩|*^(2)

*σ*^(2)*n **<* ∞. (P)

*Proof.* Let* y** ∈*rg(*K*), then there is* x* with* y* =* Kx* and we have

*⟨**y*,* un**⟩*=* ⟨**Kx*,* un**⟩*=* ⟨**x*,* K*^(*∗*)*un**⟩*=* σ**n** ⟨**x*,* vn**⟩*.

We get

## ∑
 *n*

*|⟨**y*,* un**⟩|*^(2)

*σ*^(2)*n*
### * *= ∑
 *n **|⟨**x*,* vn**⟩|*^(2)^(* *)*≤∥**x**∥*^(2 )*X** *^(*<*)^( ∞)^(.)

Conversely,the* y** ∈*cl(rg(*K*)) fulfill (P). We define* xN* = ∑^(*N *)*n*=1 1 *σ**n** *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*vn *)and from (P) it follows that* xN* is a Cauchy sequence, and thus,

*xN** →*∑ *n*

1 *σ**n** *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*vn*)^( =)^(:)^(* x*)^(.)

Finally, we get

*Kx* =* K*

 
## ∑
 *n*

1 *σ**n** *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*vn*)

!

### = ∑
 *n*

1 *σ**n** *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*Kvn*)^( =)^( )∑ *n **⟨**y*,* un**⟩**un*

=* P*cl(rg(*K*))*y* =* y*

which shows that* y** ∈*rg(*K*).

With the singular value decomposition, we can define the so-called* Moore-Penrose pseudo-inverse* (often just called pseudo- inverse).

**Definition 4.9** (Pseudo-inverse)**.** Let* K** ∈**K*(*X*,* Y*) with singular system (*σ**n*,* un*,* vn*). Then the* pseudo-inverse* of* K*, is* K*^(† ): rg(*K*)* ⊕ *rg(*K*)^(*⊥*)*→**X* defined by

*K*^(†)*y* = ∑ *n*

1 *σ**n** *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*vn*)^(.)

We denote by* D*(*K*^(†)) := rg(*K*)* ⊕*rg(*K*)^(*⊥*)*⊂**Y* the* domain* of the pseudo-inverse.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 20


---

## Page 21

04.05.2023, VL 4-4

*Remark* 4.10*. *1. Note that

*K*^(†)*Kx* = ∑ *n*

1 *σ**n** *^(*⟨*)^(*Kx*)^(,)^(* un*)^(*⟩*)^(*vn*)^( =)^( )∑ *n*

1 *σ**n** *^(*⟨*)^(*x*)^(,)^(* K*)^(*∗*)^(*un*)^(*⟩*)^(*vn*)

### = ∑
 *n **⟨**x*,* vn**⟩**vn* =* P*cl rg(*K**∗*)*x* =* P*ker(*K*)*⊥**x*,

i.e.* K*^(†)*K* =* P*ker(*K*)*⊥*.

2. Similarly, we have

*KK*^(†)*y* = ∑ *n **σ**n *D *K*^(†)*y*,* vn *E
###  *un* = ∑
 *n **σ**n*

*
##  ∑
 *m*

1 *σ**m** *^(*⟨*)^(*y*)^(,)^(* um*)^(*⟩*)^(*vm*)^(,)^(* vn*)

+

*un*

### = ∑
 *m*,*n **σ**n* ^(1)

*σ**m** *^(*⟨*)^(*y*)^(,)^(* um*)^(*⟩⟨*)^(*vm*)^(,)^(* vn*)^(*⟩*)^(*un*)^( =)^( )∑ *n **⟨**y*,* un**⟩**un* =* P*cl(rg(*K*))*y*

=* P*ker(*K**∗*)*⊥*,

i.e.* KK*^(†)^( )=* P*cl(rg(*K*)) =* P*ker(*K**∗*)*⊥*.

3. We have* K*^(†)*y* = 0 if* y** ∈*rg(*K*)^(*⊥*), i.e. ker(*K*^(†)) = rg(*K*)^(*⊥*).

4. Since (*vn*) is a basis of cl rg(*K*^(*∗*)) = ker(*K*)^(*⊥*), we have that rg(*K*^(†)) = cl rg(*K*^(*∗*)) = ker(*K*)^(*⊥*). Note that rg(*K*) is in gen- eral not closed (for compact operators it is only closed if it is finite dimensional), i.e. it is not a Hilbert space.

By the above remark, the pseudo-inverse is actually a kind of an inverse, namely of* K**|*ker(*K*)*⊥*: ker(*K*)^(*⊥*)*→*rg(*K*). There is a little more to say:

**Theorem 4.11.*** For every** y** ∈**D*(*K*^(†))* it holds that the equation** Kx* =* y **has a unique* minimum norm solution* which is** x*^(†)^( )=* K*^(†)*y**, i.e.** x*^(†)^(* *)*is a least squares solution of minimal norm, i.e. it holds that*

*∥**Kx*^(†)^(* *)*−**y**∥**Y* = min* {∥**Kx** −**y**∥**Y** |** x** ∈**X**} **and*

*∥**x*^(†)*∥**X* = min* {∥**z**∥**X** |** z** is a least squares solution of** Kx* =* y**}* .

*Moreover, the set of all least squares solutions is** x*^(†)^( )+ ker(*K*)*.*

The first equality defines “least squares solutions”.

*Proof.* That* x*^(†)^( )=* K*^(†)*y* is a least squares solution follows from Remark 4.10, 2.:

*∥**Kx*^(†)^(* *)*−**y**∥**Y* =* ∥**KK*^(†)*y** −**y**∥**Y* =* ∥**P*cl(rg(*K*))*y** −**y**∥**Y*.

Now recall that the orthogonal projection* P*cl(rg(*K*))*y* is the closest point to* y* within the closure of range of* K*. If* x*^(*′*)^( )is any least squares solution* x*^(*′*)^( )we can write* x*^(*′*)^( )=* x*^(†)^( )+ *v* with* v** ∈*ker(*K*), but since* x*^(†)^(* *)*∈*ker(*K*)^(*⊥*)we have (by the Pythagorean theorem)

*∥**x*^(*′*)*∥*^(2 )*X* ^(=)^(* ∥*)^(*x*)^(†)^(*∥*)^(2 )*X* ^(+)^(* ∥*)^(*v*)^(*∥*)^(2 )*X** *^(*≥∥*)^(*x*)^(†)^(*∥*)^(2 )*X*

which shows that* x*^(†)^( )has minimal norm among all least squares solutions.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 21


---

## Page 22

04.05.2023, VL 4-5

*Remark* 4.12*.* The pseudo inverse can also be defined for general bounded linear operators (not necessarily compact ones)* A** ∈ **L*(*X*,* Y*). There one defines the* A*^(†)*y* as the unique minimum norm least squares solution (and has to show that this is a meaningful definition). All properties of the pseudo inverse we have shown are still fulfilled in this case. We will use the pseudo-inverse also for merely bounded oper- ators in the following.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 22


---

## Page 23

16.05.2022, VL 5-1

### **5 Regularization**

We have seen in the previous section that the pseudo-inverse solves two of the problems with ill-posed linear problems: Existence and uniqueness. A little bit more explicit: The problem of existence is (roughly) solved by moving to least squares solutions (i.e. mini- mizing the residual* ∥**Kx** −**y**∥**Y* rather than solving* Kx* =* y*) and the problem of uniqueness is solved by considering minimum norm solutions, i.e. among all (least squares) solution we pick the one with minimal norm. What about the remaining problem of instability? Before we answer that, we note the following fact:

**Lemma 5.1.*** If** K** ∈**K*(*X*,* Y*)* has the singular system* (*σ**n*,* unvn*)*, then we have** ∥**K**∥*=* σ*1*.*

The proof is a good exercise Unfortunately, this shows that the pseudo-inverse is, in gen- eral, not bounded: If rg(*K*) is infinite dimensional, we have from Theorem 4.6 that* σ**n** →*0. But this implies

*∥**K*^(†)*un**∥*=* ∥*∑ *m*

1 *σ**m** *^(*⟨*)^(*un*)^(,)^(* um*)^(*⟩*)^(*vm*)^(*∥*)^(=)^(* ∥*)^(1)

*σ**n *^(*vn*)^(*∥*)^(= )1 *σ**n n**→*∞ *−→*∞

and thus,* K*^(†)^( )can not be bounded. The pseudo-inverse even helps to It’s worth to consider the finite dimen- sional case here: If* K* =* U*Σ*V*^(*T*)^( )is the singular value decomposition, then *∥**K**∥*=* σ*1 and* ∥**K*^(†)*∥*= 1/*σ**k* where* k *is the smallest singular value. The* con- dition number* of* K* is defined as* κ*(*K*) = *∥**K**∥∥**K*^(†)*∥*and hence, equal the ratio of the largest and smallest singular value of* K*. For inverse problems in infinite dimensions, the condition number can the infinite as there may be arbitrarily small singular values.

make the instability quite quantifiable: Consider the case that* y*^(†)^( )= *Kx*^(†)^( )for* x** ∈*ker(*K*)^(*⊥*). Then* x*^(†)^( )is the minimum norm least squares solution of* Kx* =* y*^(†). Let’s assume that we have measurement data *y*^(*δ*)^( )instead of* y*^(†)^( )and let us assume moreover, that we know that we have a small measurement error, i.e.* ∥**y*^(†)^(* *)*−**y*^(*δ*)*∥**Y** ≤**δ* for some known* δ** >* 0. Then the “noise” ns the data is

*η* =* y*^(*δ*)^(* *)*−**y*^(†)^(* *)*∈**Y*.

Let us blindly apply the pseudo inverse to* y*^(*δ*):

*K*^(†)*y*^(*δ*)^( )=* K*^(†)(*y*^(†)^( )+* η*) =* x*^(†)^( )+* K*^(†)*η* =* x*^(†)^( )+ ∑ *n*

1 *σ**n** *^(*⟨*)^(*η*)^(,)^(* un*)^(*⟩*)^(*vn*)^(.)

We see that the contribution of the noise is amplified unboundedly, i.e. the component* ⟨**η*,* un**⟩*of the noise in the* n*-th singular vector *un* is amplified by a factor of 1/*σ**n* and these factors grow beyond all bounds. Hence: If the noise contains contributions from singular vectors that correspond to small singular values, they get amplified a lot. Unfortunately, this is the standard situation: Singular vectors for small singular values tend to be oscillatory (i.e. be of high frequency) and hence, noise always tends to be amplified.

*Example* 5.2 (Discretized inverse problems)*.* One can check this observation numerically. After discretization, an inverse problem still reads as** Kx** =** y**^(*δ*)^( )with** x*** ∈***R**^(*n*),** y*** ∈***R**^(*m*)^( )and** K*** ∈***R**^(*m*)^(*×*)^(*n*). The singular value decomposition exists as well and if we write the

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 23


---

## Page 24

16.05.2022, VL 5-2

singular vectors** u***i* and** v***j* as colums in matrices** U** and** V** and the singular values* σ**i* on the diagonal of a matrix** Σ**, we get

### **Kx** = ∑
 *i **σ**i** ⟨***x**,** v***i**⟩***u***i* =** U****Σ****V**^(*T*)**x**.

The pseudo inverse is

**K**^(†)**y** = ∑ *i*

1 *σ**i** *^(*⟨*)^(**y**)^(,)^(** u**)^(*i*)^(*⟩*)^(**v**)^(*i*)^( =)^(** V**)^(**Σ**)^(†)^(**U**)^(*T*)^(**y**)^(. )(*)

where** Σ**^(†)^( )has the values 1/*σ**i* on the diagonal. Let us consider a (quite simple) discrete approximation of the inverse problem of diﬀerentiation, i.e. the inversion of* A* given by *A f* (*x*) = ^(R)^(* x *)0* *^(*f*)^( ()^(*t*)^())^(d)^(*t*)^(. This operator can be (roughly) discretized by )the matrix

**A** = ^(1)

*n*





1 0 *· · · *0 ... ... ... ... ... ... 0 1 *· · · · · · *1



 *∈***R**^(*n*)^(*×*)^(*n*).

Here is an example of the naive reconstruction (we can use a direct solve here, since the matrix is actually invertible (it is square and it smallest singular value is positive, but quite small). We could, in principle, also use pinv to calculate the pseudo-inverse or use the formula (*)).

import numpy as np import matplotlib.pyplot as plt

# problem size and matrix n = 100 A = np.tril(np.ones((n,n)))/n

# discretized interval t = np.linspace(0,1,n)

# true solution xdag = 1-t**2 # noise free data ydag = A@xdag

# noisy data eta = np.random.randn(n); eta /= np.sum(eta)

# noise level delta = 0.05 ydelta = ydag + delta*eta

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 24


---

## Page 25

16.05.2022, VL 5-3

# naive reconstruction x = np.linalg.solve(A,ydelta)

fig, axs = plt.subplots(2,2) axs[0,0].plot(t,xdag) axs[0, 0].set_title(’true solution’) axs[0,1].plot(t,ydag) axs[0, 0].set_title(’true data’) axs[1,0].plot(t,x) axs[0, 0].set_title(’naive reconstruction’) axs[1,1].plot(t,ydelta) axs[0, 0].set_title(’noisy data’) plt.show()

# compute svd U,S,VT = np.linalg.svd(A)

# show some singular vectors fig, axs = plt.subplots(5,2) axs[0,0].plot(t,U[:,0]) axs[0,0].set_title(’$u_1$’) axs[0,1].plot(t,VT[0,:]) axs[0,1].set_title(’$v_1$’) axs[1,0].plot(t,U[:,1]) axs[1,0].set_title(’$u_2$’) axs[1,1].plot(t,VT[1,:]) axs[1,1].set_title(’$v_2$’) axs[2,0].plot(t,U[:,2])

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 25


---

## Page 26

16.05.2022, VL 5-4

axs[2,0].set_title(’$u_3$’) axs[2,1].plot(t,VT[2,:]) axs[2,1].set_title(’$v_3$’) axs[3,0].plot(t,U[:,9]) axs[3,0].set_title(’$u_{10}$’) axs[3,1].plot(t,VT[9,:]) axs[3,1].set_title(’$v_{10}$’) axs[4,0].plot(t,U[:,-1]) axs[4,0].set_title(’$u_n$’) axs[4,1].plot(t,VT[-1,:]) axs[4,1].set_title(’$v_n$’) plt.show()

# plot singular vectors in semilog plot plt.semilogy(S) plt.show()

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 26


---

## Page 27

16.05.2022, VL 5-5

△

Now, let us fix our aim and let us define what a “regularization” shall be:

**Definition 5.3** (Regularization)**.** Let* A** ∈**L*(*X*,* Y*). A* regularization *of* A*^(†)^( )is a family of continuous maps* R**α* :* Y** →**X*,* α** >* 0 such that for all* y** ∈**D*(*A*^(†)) it holds that

*R**α**y** *^(*α*)^(*→*)^(0 )*−→**A*^(†)*y*.

If all* R**α* are linear, we speak of a* linear regularization*. The parameter *α* is called* regularization parameter*.

As a matter of fact, any linear regularization can not be uni- formly bounded (as they approximate an unbounded operator):

**Theorem 5.4.*** Let** A** ∈**L*(*X*,* Y*)* and** R**α** be a linear regularization of*

*A*^(†)*. If** A*^(†)^(* *)*is unbounded, then it holds that** ∥**R**α**∥**X **α**→*0 *−→*∞*. *This follows from the so-called uniform boundedness prinicple (also known as Banach-Steinhaus Theorem) and we do not discuss the proof here.

Now let us discuss the various error we defined in the example in Section 2: The* totel error** ∥**R**α**y*^(*δ*)^(* *)*−**x*^(†)*∥*(also called regularization error) can be decomposed (using* x*^(†)^( )=* A*^(†)*y*^(†)^( )and the triangle inequality) in the case of linear regularization into* data error* and *approximation error*

*∥**R**α**y*^(*δ*)^(* *)*−**x*^(†)*∥**X** ≤∥**R**α**y*^(*δ*)^(* *)*−**R**α**y*^(†)*∥**X* +* ∥**R**α**y*^(†)^(* *)*−**A*^(†)*y*^(†)*∥**X **≤∥**R**α**∥**δ* +* ∥**R**α**y*^(†)^(* *)*−**A*^(†)*y*^(†)*∥**X*. (*)

Again, we see that one needs to choose the regularization param- eter* α* carefully: At least we need to be able to balance the term *∥**R**α**∥**δ* as it blows up for small* α*. On the other hand, the term *∥**R**α**y*^(†)^(* *)*−**A*^(†)*y*^(†)*∥**X* tends to be small for small* α* and large for large *α* (exactly as we have seen in Section 2). We will often denote the regularized reconstruction by* x*^(*δ *)*α* ^(:)^(= )*R**α**y*^(*δ*). We will also use the notation* x**α* :=* R**α**y*^(†)^( )for the (in general

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 27


---

## Page 28

16.05.2022, VL 5-6

unknown) regularized reconstruction from idealized noiseless data. With this notation, our error decomposition is

*∥**x*^(*δ *)*α** *^(*−*)^(*x*)^(†)^(*∥*)^(*X *)| {z } total error

*≤∥**x*^(*δ *)*α** *^(*−*)^(*xa*)^(*∥*)^(*X *)| {z } data error

+ *∥**x**α** −**x*^(†)*∥**X *| {z } approximation error

.

**Definition 5.5** (Parameter choice)**.** A function

*α* : ]0, ∞[* ×** Y** →*]0, ∞[, (*δ*,* y*^(*δ*))* →**α*(*δ*,* y*^(*δ*))

is called a* parameter choice rule*. We distinguish further:* α* is

(i) an* a priori choice rule* if* α* does not depend on* y*^(*δ*),

(ii) an* a posteriori choice rule* if* α* depends on* δ* and* y*^(*δ*), and

(iii) a* heuristic rule* is* α* does not depend on* δ*. An* a priori* rule can be devised with- out having seen the actual data (it only needs knowledge of the noise level), hence one can, in principle, construct the operator* R**α*(*δ*) a priorily, before the data has arrived; hence, the name.

**Definition 5.6** (Convergent regularization)**.** If* R**α* is a regularization of* A*^(†)^( )and* α* is a parameter choice rule we say that (*R**α*,* α*) is a *convergent regularization method* if for all* y*^(†)^(* *)*∈**D*(*A*^(†)) it holds that

sup n *∥**R**α*(*δ*,*y**δ*)*y*^(*δ*)^(* *)*−**A*^(†)*y*^(†)*∥**X ** ∥**y**δ** −**y*†*∥**Y** ≤**δ *o* δ**→*0 *−→*0.

In other words: We want that the* worst case reconstruction error *goes to zero, i.e. even if the noisy data* y*^(*δ*)^( )is as bad as possible, given the noise level* δ*. Informally: We demand that in the regime of vanishing noise, we shall be able to approximate the true solution *x*^(†)^( )=* A*^(†)*y*^(†)^( )as good as possible. On the one hand, this sounds like a mean- ingless demand, since usually the noise level stays fixed. On the other hand, it sounds like something that should be the bare minimum: If we can not even guarantee this, what is the point of reg- ularization at all? Finally, it sounds quite ambitious, given that we already know that we try to approximate unbounded (i.e. discontinuous) operators with con- tinuous ones.

*Example* 5.7 (Truncated SVD)*.* Here is a simple idea for a regulariza- tion method for: For* K** ∈**K*(*X*,* Y*) with singular system (*σ**n*,* un*,* vn*) and* α** >* 0 we define

*R**α**y* = ∑ *σ**n**>**α*

1 *σ**n** *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*vn*)

i.e. we cut oﬀthe small singular values which lead to unbounded- ness of the pseudo-inverse. These* R**α* are indeed bounded opera- tors:

*∥**R**α**y**∥*^(2)
### ^( )*X* ^(=)^( )∑
 *σ**n**>**α*

1 *σ*^(2)*n** *^(*|⟨*)^(*y*)^(,)^(* un*)^(*⟩|*)^(2)^(* ≤*)^(sup )n 1 *σ*^(2)*n** *^(*|*)^(* σ*)^(*n*)^(* ≥*)^(*α *)o *∥**y**∥*^(2 )*Y** *^(*≤*)^(1)

*α*^(2)^(* ∥*)^(*y*)^(*∥*)^(2 )*Y*

i.e.* ∥**R**α**∥≤*^(1)

*α*^(. )Let us investigate if the truncated SVD is a convergent regular- ization method, i.e. if we can find a suitable parameter choice: To that end, we use our standard error decomposition (*)

*∥**R**α**y*^(*δ*)^(* *)*−**K*^(†)*y*^(†)*∥**X** ≤∥**R**α**∥**δ* +* ∥**R**α**y*^(†)^(* *)*−**A*^(†)*y*^(†)*∥**X **≤*^(*δ*)

*α* ^(+)^(* ∥*)^(*R*)^(*α*)^(*y*)^(†)^(* −*)^(*A*)^(†)^(*y*)^(†)^(*∥*)^(*X*)^(.)

We estimate the approximation error further

*R**α**y*^(†)^(* *)*−**A*^(†)*y*^(†)^( )= ∑ *σ**n**≥**α*

1 *σ**n*

D *y*^(†),* un *E *vn** −*∑ *n*

1 *σ**n*

D *y*^(†),* un *E *vn* =* −*∑ *σ**n**<**α*

1 *σ**n*

D *y*^(†),* un *E *vn*.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 28


---

## Page 29

16.05.2022, VL 5-7

This gives us

*∥**R**α**y*^(†)^(* *)*−**A*^(†)*y*^(†)*∥*^(2 )*X** *^(*≤*)∑ *σ**n**<**α*

1 *σ*^(2)*n** *^(*| *)D *y*^(†),* un *E *|*^(2).

Together we have

*∥**R**α**y*^(*δ*)^(* *)*−**K*^(†)*y*^(†)*∥**X** ≤∥**R**α**∥**δ* +* ∥**R**α**y*^(†)^(* *)*−**A*^(†)*y*^(†)*∥**X*

*≤*^(*δ*)

*α* ^(+ )r

## ∑
 *σ**n**<**α*

1 *σ*^(2)*n** *^(*|⟨*)^(*y*)^(†,)^(* un*)^(*⟩|*)^(2.)

Now we see: The second summand is the “rest of a convergent series” (recall the Picard condition, Theorem 4.8) and the smaller *α*, the later the rest of the series starts. Hence, we have r

## ∑
 *σ**n**<**α*

1 *σ*^(2)*n** *^(*|⟨*)^(*y*)^(†,)^(* un*)^(*⟩|*)^(2)^(* →*)^(0 )for *α** →*0.

For the first term we need that* α*(*δ*)* →*0 slower than* δ*. In conclu- sion: Any* α*(*δ*) with

*α*(*δ*)* *^(*δ*)^(*→*)^(0 )*−→*0, *δ α*(*δ*)

*δ**→*0 *−→*0

is a valid (a priori) parameter choice rule and we can claim that the truncated SVD together with this rule is a convergent regular- ization method. △ One could take, for example,* α*(*δ*) = *√*

*δ* (or =* δ*^(*κ*)^( )for 0* <** κ** <* 1, for that matter).

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 29


---

## Page 30

23.05.2022, VL 6-1

### **6 Tikhonov regularization**

The problem of instability of the solution of* Ax* =* y*^(*δ*)^( )comes from the small singular values which are the eigenvalues of the self- adjoint operator* A*^(*∗*)*A*. Another way to understand this, is via the normal equation: Some* x* is a minimizer of* ∥**Ax** −**y*^(*δ*)*∥*^(2 )*X* ^(exactly if )it solves the* normal equation*

*A*^(*∗*)*Ax* =* A*^(*∗*)*y*^(*δ*).

However, in general minimizers of* ∥**Ax** −**y*^(*δ*)*∥*^(2 )*X* ^(do not exist (i.e. )the normal equation does not have solutions) and this is (in the case of compact* A*) due to the eigenvalues of* A*^(*∗*)*A* converging to zero. To avoid this problem, we can simply shift them to be strictly positive: If* σ*^(2 )*i* ^(are the eigenvalues of)^(* A*)^(*∗*)^(*A*)^(, then the eigenvalues )of* A*^(*∗*)*A* +* α* id are* σ*^(2 )*n* ^(+)^(* α*)^(* ≥*)^(*α*)^(* >*)^( 0)^(. Hence, instead of the normal )equation, we consider for* α** >* 0 regularized normal equations

(*A*^(*∗*)*A* +* α* id)*x* =* A*^(*∗*)*y*^(*δ*).

Since the operator* A*^(*∗*)*A* +* α* id is always invertible, we can write this as

*x*^(*δ *)*α* ^(= ()^(*A*)^(*∗*)^(*A*)^( +)^(* α*)^( id)^())^(*−*)^(1)^(*A*)^(*∗*)^(*y*)^(*δ*)

and this method is called* Tikhonov regularization*. The shift of the singular values is one motivation for Tikhonov regularization. In fact, Tikhonov regularization also corresponds to a regularized least squares problem.

**Theorem 6.1.*** Let** A** ∈**L*(*X*,* Y*)*. The regularized normal equation *(*A*^(*∗*)*A* +* α* id)*x* =* A*^(*∗*)*y*^(*δ*)^(* *)*has a unique solution** x*^(*δ *)*α** *^(*which is exactly the *)*unique minimum of the Tikhonov functional*

*T**α*(*x*;* y*^(*δ*)) := ^(1)

2^(*∥*)^(*Ax*)^(* −*)^(*y*)^(*δ*)^(*∥*)^(2 )*Y* ^(+)^(* α*)

2^(*∥*)^(*x*)^(*∥*)^(2 )*X*^(.)

*Proof.* A minimizer* x* of the Tikhonov function is characterized by the condition that* T**α*(*x* +* th*;* y*^(*δ*))* ≥**T**α*(*x*;* y*^(*δ*)) for all* t** ∈***R** and *h** ∈**X*. Starting from the left hand side we get

*T**α*(*x* +* th*;* y*^(*δ*)) = ^(1)

2^(*∥*)^(*Ax*)^( +)^(* tAh*)^(* −*)^(*y*)^(*δ*)^(*∥*)^(2 )*Y* ^(+)^(* α*)

2^(*∥*)^(*x*)^( +)^(* th*)^(*∥*)^(2 )*X*

= ^(1)

2^(*∥*)^(*Ax*)^(* −*)^(*y*)^(*δ*)^(*∥*)^(2 )*Y* ^(+ )D *Ax** −**y*^(*δ*),* tAh *E + ^(1)

2^(*∥*)^(*tAh*)^(*∥*)^(2 )*Y*

+* *^(*α*)

2^(*∥*)^(*x*)^(*∥*)^(2 )*X* ^(+)^(* α*)^(* ⟨*)^(*x*)^(,)^(* th*)^(*⟩*)^(+)^(* α*)

2^(*∥*)^(*th*)^(*∥*)^(2 )*X*

=* T**α*(*x*;* y*^(*δ*)) +* t *D *A*^(*∗*)(*Ax** −**y*^(*δ*)) +* α**x*,* h *E +* t*^(2)( ^(1)

2^(*∥*)^(*Ah*)^(*∥*)^(2 )*Y* ^(+)^(* α*)

2^(*∥*)^(*h*)^(*∥*)^(2 )*X*^())^(.)

We see that* T**α*(*x* +* th*;* y*^(*δ*))* ≥**T**α*(*x*;* y*^(*δ*)) holds for all* t* and* h* exactly if D *A*^(*∗*)(*Ax** −**y*^(*δ*)) +* α**x*,* h *E = 0

for all* h** ∈**X* and this is exactly the case when* A*^(*∗*)(*Ax** −**y*^(*δ*)) +* α**x* = 0 which is just the regularized normal equation. Uniqueness of the minimizer follows since the Tikhonov functional is strictly convex.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 30


---

## Page 31

23.05.2022, VL 6-2

The description of Tikhonov regularization as a minimization framework allows for another interpretation: The regularization is a compromise of two things, namely finding a reconstruction *x*^(*δ *)*α* ^(that has a good)^(* data fit*)^(, i.e. it produces a small value for the )residual* ∥**Ax** −**y*^(*δ*)*∥**Y*, but, at the same time, also does not blow up, i.e. it has a small norm* ∥**x**∥**X*. These two demands are weighted by the regularization parameter* α*. Regularization methods that build upon the idea of minimizing a functional that balances the demands of data fit and “reasonable reconstruction” are also called “variational regularization methods” (as the theory that deals with minimization problems in infinite dimensional spaces is called “calculus of variations”). Aiming at a reconstruction with a bounded norm seems like a valid idea, but one may know a little more about the unknown solution. If we assume that we have a rough idea of the unknown* x*^(†), i.e. we know that* x*^(0)^( )is a good guess, we can of course minimize

*T**α*(*x*;* y*^(*δ*),* x*^(0)) := ^(1)

2^(*∥*)^(*Ax*)^(* −*)^(*y*)^(*δ*)^(*∥*)^(2 )*Y* ^(+)^(* α*)

2^(*∥*)^(*x*)^(* −*)^(*x*)^(0)^(*∥*)^(2 )*X*^(.)

Similar to the proof of Theorem 6.1 one shows that the unique minimizer here is given as a solution of

(*A*^(*∗*)*A* +* α* id)*x* =* A*^(*∗*)*y*^(*δ*)^( )+* α**x*^(0).

*Remark* 6.2 (Numerical realization of Tikhonov regularization)*. *Tikhonov regularization is popular, because its implementation is pretty straight forward. Let us consider the discrete case where **A*** ∈***K**^(*m*)^(*×*)^(*n*)^( )and** y**^(*δ*)^(* *)*∈***R**^(*m*). Then the regularized normal equation Both the overdetermined case* m** >** n *(where non-existence of solutions is a problem, due to measurement error) and the underdetermined case* m** <** n *(where non-uniqueness is a problem, due to not enough data) of can be con- sidered here.

(for** x**^(0)^( )= 0

(**A**^(*T*)**A** +* α**In*)**x** =** A**^(*T*)**y**^(*δ*)

is a square linear system in* n* dimensions and the matrix (**A**^(*T*)**A** + *α**In*) is symmetric positive definite. Hence, there are many meth- ods available to solve the problem numerically (one method is the method of conjugate gradients).

Is Tikhonov regularization indeed a convergence regulariza- tion method? To answer this question, we should find a parameter choice rule. We will analyze this question with the help of the singular value decomposition.

**Theorem 6.3.*** Let** K** ∈**K*(*X*,* Y*)* have the singular system* (*σ**n*,* un*,* vn*)*. Then solution** x*^(*δ *)*α** *^(*of*)^( ()^(*A*)^(*∗*)^(*A*)^( +)^(* α*)^( id)^())^(*x*)^( =)^(* A*)^(*∗*)^(*y*)^(*δ*)^(* is given by*)

*x*^(*δ*)
### ^(* *)*α* ^(=)^( )∑
 *n*

*σ**n **σ*^(2)*n*+*α*

D *y*^(*δ*),* un *E *vn*.

*Proof.* It holds that* x*^(*δ *)*α* ^(=)^(* P*)ker(*A*)^(*x*)^(*δ *)*α* ^(+)^(* P*)ker(*A*)^(*⊥*)^(*x*)^(*δ *)*α* ^(=)^(* P*)ker(*A*)^(*x*)^(*δ *)*α* ^(+)

∑*n *
 *x*^(*δ *)*α*^(,)^(* vn *) *vn*. Since (*σ*^(2 )*n*^(,)^(* vn*)^(,)^(* vn*)^())^( is the spectral decomposition of)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 31


---

## Page 32

23.05.2022, VL 6-3

*A*^(*∗*)*A* we get* A*^(*∗*)*Ax*^(*δ *)*α* ^(=)^( ∑)*n** *^(*σ*)^(2 )*n *
 *x*^(*δ *)*α*^(,)^(* vn *) *vn*. Also* A*^(*∗*)*y*^(*δ*)^( )= ∑*n** σ**n *
 *y*^(*δ*),* un * *vn*. Thus, the regularized normal equation is

## ∑
 *n **σ*^(2 )*n *D *x*^(*δ *)*α*^(,)^(* vn *)E *vn* +* α*

 

*P*ker(*A*)(*x*^(*δ*)
### ^(* *)*α*^() +)^( )∑
 *n*

D *x*^(*δ *)*α*^(,)^(* vn *)E *vn*

!

=* A*^(*∗*)*y*^(*δ*)

### = ∑
 *n **σ**n *D *y*^(*δ*),* un *E *vn*.

We see that necessarily* P*ker(*A*)(*x*^(*δ *)*α*^() =)^( 0)^( and that)

## ∑
 *n *(*σ*^(2 )*n* ^(+)^(* α*)^() )D *x*^(*δ *)*α*^(,)^(* vn *)E
###  *vn* = ∑
 *n **σ**n *D *y*^(*δ*),* un *E *vn*.

Comparing coeﬃcients shows that 
 *x*^(*δ *)*α*^(,)^(* vn *) = *σ**n **σ*^(2)*n*+*α *
 *y*^(*δ*),* un * which shows the claim.

The representation of* x*^(*δ *)*α* ^(from Theorem 6.3 via the singular )value decomposition is called* spectral representation*. We use it to prove the following result on regularization:

**Theorem 6.4** (Tikhonov with a-priori parameter choice)**.*** For an a-priori parameter choice** α*(*δ*)* that fulfills*

*α*(*δ*)* →*0 *δ*^(2)

*α*(*δ*)* *^(*→*)^(0 )*for **δ** →*0

*it holds that Tikhonov regularization is a convergent regularization method, i.e. it holds that** x*^(*δ *)*α* ^(:)^(= ()^(*A*)^(*∗*)^(*A*)^( +)^(* α*)^( id)^())^(*−*)^(1)^(*A*)^(*∗*)^(*y*)^(*δ*)^(* →*)^(*x*)^(† :)^(=)^(* A*)^(†)^(*y*)^(†)^(* whenever *)*∥**y*^(*δ*)^(* *)*−**y*^(†)*∥≤**δ** and** δ** →*0*.*

*Proof.* We set* x**α* = (*A*^(*∗*)*A* +* α* id)^(*−*)^(1)*A*^(*∗*)*y*^(†)^( )and decompose

*x*^(*δ *)*α** *^(*−*)^(*x*)^(†)^( =)^(* x*)^(*δ *)*α** *^(*−*)^(*x*)^(*α*) ^(+)^(* x*)^(*α*)* *^(*−*)^(*x*)^(†)

= (*A*^(*∗*)*A* +* α* id)^(*−*)^(1)*A*^(*∗*)(*y*^(*δ*)^(* *)*−**y*^(†)) | {z } data error

+ (*A*^(*∗*)*A* +* α* id)^(*−*)^(1)*A*^(*∗*)*y*^(†)^(* *)*−**A*^(†)*y*^(† )| {z } approx. error

.

The data error fulfills

(*A*^(*∗*)*A* +* α* id)^(*−*)^(1)*A*^(*∗*)(*y*^(*δ*)^(* *)*−**y*^(†)) = ∑ *n*

*σ**n **σ*^(2)*n*+*α*

D *y*^(*δ*)^(* *)*−**y*^(†),* un *E *vn*,

and hence, its norm is

*∥*(*A*^(*∗*)*A* +* α* id)^(*−*)^(1)*A*^(*∗*)(*y*^(*δ*)^(* *)*−**y*^(†))*∥*^(2)
### ^( )*Y* ^(=)^( )∑
 *n*

 *σ**n **σ*^(2)*n*+*α*

2 *| *D *y*^(*δ*)^(* *)*−**y*^(†),* un *E *|*^(2)

*≤*

 

sup 0*≤**σ**≤∥**A**∥*

*σ σ*^(2)^( )+* α*

!2 *∥**y*^(*δ*)^(* *)*−**y*^(†)*∥*^(2 )*Y*

For the approximation error and we use that* x*^(†)^( )=* A*^(†)*y*^(†)^( )implies 
 *x*^(†),* vn * = 1 *σ**n *
 *y*^(*δ*),* un * to get

(*A*^(*∗*)*A* +* α* id)^(*−*)^(1)*A*^(*∗*)*y*^(†)^(* *)*−**A*^(†)*y*^(†)^( )= ∑ *n*

*σ**n **σ*^(2)*n*+*α*

D *y*^(†),* un *E *vn** −*∑ *n*

1 *σ**n*

D *y*^(†),* un *E *vn*

### = ∑
 *n*

� *σ*^(2 )*n **σ*^(2)*n*+*α** *^(*−*)^(1 ) ^(D )*x*^(†),* vn *E *vn*.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 32


---

## Page 33

23.05.2022, VL 6-4

Together we arrive at the error estimate

*∥**x*^(*δ *)*α** *^(*−*)^(*x*)^(†)^(*∥*)^(*X*)* *^(*≤*)

 

sup 0*≤**σ**≤∥**A**∥*

*σ σ*^(2)+*α*

!

*δ* +

s

## ∑
 *n*

� *σ*^(2 )*n **σ*^(2)*n*+*α** *^(*−*)^(1 )2*|⟨**x*†,* vn**⟩|*2.

(*)

We estimate the supremum by

sup 0*≤**σ**≤∥**A**∥*

*σ σ*^(2)+*α** *^(*≤ *)1 2^(*√*)*α*^(.)

We maximize over all* σ** >* 0: The derivative of* σ*/(*σ*^(2)^( )+* α*) is ((*σ*^(2)^( )+* α*)* −*2*σ*^(2))/(*σ*^(2)^( )+* α*)^(2)^( )and hence, vanishes exactly at *σ* =* *^(*√*)*α*. Plugging this in gives the result.

By assumption* δ*/^(*√*)*α** →*0 for* δ** →*0, and thus, the first term on the right hand side of (*) goes to zero for* δ** →*0. Now we consider the square of second term in (*), which we write as ∑*n an*(*α*) with* an*(*α*) = � *σ*^(2 )*n **σ*^(2)*n*+*α** *^(*−*)^(1 )2*| *
 *x*^(†),* vn **|*2. It holds

that* an*(*α*)* →| *
 *x*^(†),* vn **|*2 for* α** →*0. We have the (very coarse)

estimate � *σ*^(2 )*n **σ*^(2)*n*+*α** *^(*−*)^(1 )2* ≤*4 and hence ∑*n an*(*α*)* ≤*4*∥**x**∥*2 *X*^(,and by )the dominated convergence theorem (Theorem 3.5), we get that the full sum ∑*n an*(*α*)* →*0 for* α** →*0. This proves the theorem.

The previous theorem shows that Tikhonov is indeed a conver- gent regularization method. However, we did not get an explicit er- ror estimate for the total error* ∥**x*^(*δ *)*α** *^(*−*)^(*x*)^(†)^(*∥*)^(*X*)^(. While we could bound )the data error by

*∥**x*^(*δ *)*α** *^(*−*)^(*x*)^(*α*)^(*∥*)^(*X*)* *^(*≤ *)*δ **√**α*,

we did not get an eﬀective bound on the approximation error *∥**x**α** −**x*^(†)*∥**X*. This is a general fact:

**Theorem 6.5** (No general worst case error bound for ill-posed problems)**.*** Let* (*R**α*,* α*(*δ*,* y*^(*δ*)))* be a convergent regularization method*

*for** A*^(†)*. If there exists a function** ψ* : ]0, ∞[* →*]0, ∞[* with** ψ*(*δ*)* *^(*δ*)^(*→*)^(0 )*−→*0 *such that for all** y*^(†)^(* *)*∈**D*(*A*^(†))

sup n *∥**R**α*(*δ*,*y**δ*)*y*^(*δ*)^(* *)*−**A*^(†)*y*^(†)*∥**X** |** y*^(†)^(* *)*∈**D*(*A*^(†)),* y*^(*δ*)^(* *)*∈**Y*,* with** ∥**y*^(†)^(* *)*−**y*^(*δ*)*∥**Y** ≤**δ *o *≤**ψ*(*δ*)

*then** A*^(†)^(* *)*is bounded. *The significance of this theorem is as follows: If* A*^(†)^( )is bounded, we can get a nice error bound* ∥**A*^(†)*y*^(*δ*)^(* *)*−**x*^(†)*∥**X** ≤ ∥**A*^(†)*∥**δ* and hence, the problem is not ill-posed.

*Proof.* Let* y*^(†),* yn** ∈**D*(*A*^(†)) with* ∥**y*^(†)^(* *)*−**yn**∥**Y* =* δ**n n**→*∞ *−→*0. Then we have

*∥**A*^(†)*yn** −**A*^(†)*y*^(†)*∥**Y** ≤∥**A*^(†)*yn** −**R**α*(*δ*,*yn*)*yn**∥*+* ∥**R**α*(*δ*,*yn*)*yn** −**A*^(†)*y**∥*.

By our assumption, we have that both terms on the right are bounded by* ψ*(*δ**n*), i.e.

*∥**A*^(†)*yn** −**A*^(†)*y*^(†)*∥**Y** ≤*2*ψ*(*δ**n*)* *^(*n*)^(*→*)^(∞ )*−→*0.

But this means that* A*^(†)^( )is continuous at* y*^(†)^( )and since* A*^(†)^( )is linear, this shown continuity everywhere.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 33


---

## Page 34

23.05.2022, VL 6-5

Here is an example of Tikhonov regularization in practice:

import numpy as np import matplotlib.pyplot as plt

# problem size and matrix n = 100 A = np.tril(np.ones((n,n)))/n

# discretized interval t = np.linspace(0,1,n)

# true solution xdag = np.maximum(1-2*t,0) # noise free data ydag = A@xdag

# noisy data eta = np.random.randn(n); eta /= np.linalg.norm(eta)

# noise level delta = 0.05 ydelta = ydag + delta*eta

# naive reconstruction x = np.linalg.solve(A,ydelta)

fig, axs = plt.subplots(2,2) axs[0,0].plot(t,xdag) axs[0,0].set_title(’true solution’) axs[0,1].plot(t,ydag) axs[0,1].set_title(’true data’) axs[1,0].plot(t,x) axs[1,0].set_title(’naive reconstruction’) axs[1,1].plot(t,ydelta) axs[1,1].set_title(’noisy data’) plt.show()

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 34


---

## Page 35

23.05.2022, VL 6-6

# reconstruct with Tikhonov # regularization parameter alpha = 0.01 # compute reconstruction xalphadelta = np.linalg.solve(A.T@A + alpha*np.identity(n),A.T@ydelta) plt.plot(t,xalphadelta ,label=’xalphadelta’) plt.plot(t,xdag,label=’xdagger’) plt.legend() plt.show()

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 35


---

## Page 36

23.05.2022, VL 6-7

# Plot for N different errors to illustrate error decomposition N = 30 alphas = np.logspace(0,-6,N) totalError = np.zeros(N) approximationError = np.zeros(N) dataError = np.zeros(N) # reconstruct and compute errors for k in range(N): alpha = alphas[k] xalphadelta = np.linalg.solve(A.T@A + alpha*np.identity(n),A.T@ydelta) xalpha = np.linalg.solve(A.T@A + alpha*np.identity(n),A.T@ydag) totalError[k] = np.linalg.norm(xalphadelta -xdag) approximationError[k] = np.linalg.norm(xalpha-xdag) dataError[k] = np.linalg.norm(xalphadelta -xalpha)

# Show errors is loglog -plot plt.loglog(alphas ,totalError ,label=’total error’) plt.loglog(alphas ,approximationError ,label=’approximation error’) plt.loglog(alphas ,dataError ,label=’data error’) plt.legend() plt.show()

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 36


---

## Page 37

30.05.2022, VL 7-1

### **7 Spectral regularization**

We have analyzed the regularization properties of the truncated singular value decomposition and Tikhonov regularization in Ex- ample 5.7 and Theorem 6.4. If you inspect the arguments, you’ll note that they are actually almost the same in both cases. In this section we will start to derive a general theory for linear regu- larization. This theory will contain the truncated SVD as well as Tikhonov regularization as special cases. Recall that the truncated SVD is

*R**α**y* = ∑ *σ**n**>**α*

1 *σ**n** *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*vn*)

while we could express Tikhonov regularization as

*R**α**y* = ∑ *n*

*σ**n **σ*^(2)*n*+*α** *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*vn*)^(.)

Both methods can be written in the following form:

*R**α**y* = ∑ *n **φα*(*σ*^(2 )*n*^())^(*σ*)^(*n*)* *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*vn *)(R)

with some function* φα*: For* φα*(*λ*) = 1 *λ*+*α* ^(we get)

*R**α**y* = ∑ *n*

1 *σ*^(2)*n*+*α*^(*σ*)^(*n*)^(* ⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*vn*)^(,)

i.e. exactly Tikhonov regularization. If we set

*φα*(*λ*) =  1 *λ *: *λ** ≥**α *0 : else,

we get

*R**α**y* = ∑ *σ*^(2)*n**≥**α*

1 *σ*^(2)*n** *^(*σ*)^(*n*)^(* ⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*vn*)^( =)^( ∑ )*σ**n**>*^(*√*)*α*

1 *σ**n** *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*vn*)^(,)

which is (up to a diﬀerent scaling of the regularization parameter) the truncated SVD from Example 5.7.

*Remark* 7.1*.* Note that we could have written* R**α**y* = ∑*n f**α*(*σ**n*)* ⟨**y*,* un**⟩**vn *with some function* f**α* as well. However,there is a reason why we did chose this slightly complicated form: Regularization methods approximate the minimum norm solution of the normal equation *K*^(*∗*)*Kx* =* K*^(*∗*)*y*, i.e.* x* = (*K*^(*∗*)*K*)^(†)*K*^(*∗*)*y*. If we express everything with the SVD of* K* we first get that (*K*^(*∗*)*K*)^(†)*z* = ∑*n** σ*^(*−*)^(2 )*n **⟨**z*,* vn**⟩**vn* and hence, for* z* =* K*^(*∗*)*y*

### *x* = ∑
 *n **σ*^(*−*)^(2 )*n **⟨**K*^(*∗*)*y*,* vn**⟩**vn* = ∑ *n **σ*^(*−*)^(2 )*n **⟨**y*,* Kvn**⟩**vn*

### = ∑
 *n **σ*^(*−*)^(2 )*n **⟨**y*,* σ**nun**⟩**vn* = ∑ *n **σ*^(*−*)^(2 )*n** *^(*σ*)^(*n*)* *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*vn*)^(.)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 37


---

## Page 38

30.05.2022, VL 7-2

To mimic this formula, we express regularization methods as

*R**α**y* = ∑ *n **φα*(*σ*^(2 )*n*^())^(*σ*)^(*n*)* *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*vn*)^(.)

and need that* φα*(*λ*)* ≈*1/*λ* for* R**α* being close to* A*^(†).

We will use the following* functional calculus* for compact op- erators: If* K* is compact with singular system (*σ**n*,* un*,* vn*) and* f* : [0,* ∥**K**∥*^(2)]* →***R** is piecewise continuous and bounded, we define another operator* f* (*K*^(*∗*)*K*) :* X** →**Y* by

*f* (*K*^(*∗*)*K*)*x* := ∑ *n f* (*σ*^(2 )*n*^())^(* ⟨*)^(*x*)^(,)^(* vn*)^(*⟩*)^(*vn*) ^(+)^(* f*)^( ()^(0)^())^(*P*)ker(*K*)^(*x*)^(.)

The additional term* P*ker(*K*)*x* takes into account that* f* (0)* ̸*= 0 and makes the identity id =* f* (*K*^(*∗*)*K*) for* f** ≡*1 cor- rect.

We observe that the series always converges (since* f* is only evaluated on the bounded interval [0,* ∥**K**∥*^(2)]) and we also get that

*∥**f* (*K*^(*∗*)*K*)*∥≤*sup *n **|** f* (*σ*^(2 )*n*^())^(*|*)^( +)^(* f*)^( ()^(0)^())^(* ≤*)^(2 )sup *λ**∈*[0,*∥**K**∥*^(2)] *|** f* (*λ*)*|** <* ∞

which shows that* f* (*K*^(*∗*)*K*)* ∈**L*(*X*,* X*). With functional calculus we can write

*R**α**y* = ∑ *n **φα*(*σ*^(2 )*n*^())^(*σ*)^(*n*)* *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*vn*) ^(=)^(* φα*)^(()^(*K*)^(*∗*)^(*K*)^())^(*K*)^(*∗*)^(*y*)^(.)

*Example* 7.2 (Absolute value of a compact operator)*.* For* f* (*t*) =* t *we get that* f* (*K*^(*∗*)*K*) =* K*^(*∗*)*K* and for* f* (*t*) = *√*

*t* we define* |**K**|* := *f* (*K*^(*∗*)*K*). It holds that

### *|**K**|**x* = ∑
 *n **σ**n** ⟨**x*,* vn**⟩**vn*.

△

We state some properties of the absolute value of an operator as we will use it later when we derive convergence rates in abstract smoothness spaces in Sections 9 and 10.

**Lemma 7.3** (Properties of functional calculus)**.*** Let** K** ∈**K*(*X*,* Y*)*.*

*(i) For** s*,* r** >* 0* it holds that** |**K**|*^(*r*)^(+)^(*s*)^( )=* |**K**|*^(*r*)*|**K**|*^(*s*)*.*

*(ii) For all** r** >* 0* the operator** |**K**|*^(*r*)^(* *)*is self-adjoint.*

*(iii) For all** x** ∈**X** it holds that** ∥|**K**|**x**∥**Y* =* ∥**Kx**∥**y**.*

*(iv) It holds that* rg(*|**K**|*) = rg(*K*^(*∗*))*.*

*Proof. *(i) This is a direct computation (using that* vn* is orthonor- mal)

*|**K**|*^(*r*)^(+)^(*s*)*x* = ∑ *n **σ*^(*r*)^(+)^(*s *)*n **⟨**x*,* vn**⟩**vn* = ∑ *n **σ*^(*r *)*n*

*
##  ∑
 *m **σ*^(*s *)*m** *^(*⟨*)^(*x*)^(,)^(* vm*)^(*⟩*)^(*vm*)^(,)^(* vn*)

+

*vn*

### = ∑
 *n **σ*^(*r *)*n** *^(*⟨|*)^(*K*)^(*|*)^(*sx*)^(,)^(* vn*)^(*⟩*)^(*vn*)^(.)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 38


---

## Page 39

30.05.2022, VL 7-3

(ii) Again a direct computation

*⟨|**K**|*^(*r*)*x*,* z**⟩*= ∑ *n **σ*^(*r *)*n** *^(*⟨*)^(*x*)^(,)^(* vn*)^(*⟩⟨*)^(*vn*)^(,)^(* z*)^(*⟩*)^(=)^(* ⟨*)^(*x*)^(,)^(* |*)^(*K*)^(*|*)^(*rz*)^(*⟩*)^(.)

(iii) Using the first two points we get

*∥|**K**|**x**∥*^(2 )*X* ^(=)^(* ⟨|*)^(*K*)^(*|*)^(*x*)^(,)^(* |*)^(*K*)^(*|*)^(*x*)^(*⟩*)^(= )
*|**K**|*2*x*,* x * =* ⟨**K**∗**Kx*,* x**⟩*=* ⟨**Kx*,* Kx**⟩*=* ∥**Kx**∥*2 *Y*^(.)

(iv) If (*σ**n*,* un*,* vn*) is the singular system of* K*, then* K*^(*∗*)has the singular system (*σ**n*,* vn*,* un*) and* |**K**|* has the singular system (*σ**n*,* vn*,* vn*). Now note that* x** ∈*rg(*K*^(*∗*)) exactly if* Kx** ∈*rg(*KK*^(*∗*)) and* x**⊥*ker(*K*). The Picard condition (Theorem 4.8) for* Kx** ∈ *rg(*KK*^(*∗*)) is

### ∞*>* ∑
 *n **σ*^(*−*)^(4 )*n** *^(*|⟨*)^(*Kx*)^(,)^(* un*)^(*⟩|*)^(2)^( =)^( )∑ *n **σ*^(*−*)^(4 )*n** *^(*|⟨*)^(*x*)^(,)^(* K*)^(*∗*)^(*un*)^(*⟩|*)^(2)^( =)^( )∑ *n **σ*^(*−*)^(2 )*n** *^(*|⟨*)^(*x*)^(,)^(* vn*)^(*⟩|*)^(2)

which is exactly the Picard condition for* x** ∈*rg(*|**K**|*) (for which* x**⊥*ker(*K*) is necessary anyway).

We will investigate regularization methods of the form (R). The following definition well be useful, as we will see:

**Definition 7.4** (Regularizing filter)**.** Let* K** ∈**K*(*X*,* Y*) with* κ* = *∥**K**∥*^(2)^( )and SVD (*σ**n*,* un*,* vn*). A family* φα* : [0,* κ*]* →***R** of piecewise continuous and bounded functions is called* regularizing filter* if it fulfills

(i) For all* λ** ∈*]0,* κ*] it holds that

*φα*(*λ*)* *^(*α*)^(*→*)^(0 )*−→*^(1)

*λ*^(.)

(ii) There exists* C**φ** >* 0 such that for all* λ** ∈*]0,* κ*] and* α** >* 0 it holds that

*λ**|**φα*(*λ*)*| ≤**C**φ*.

Now we aim to prove that regularizing filters give rise to con- vergent regularization methods. First we collect three useful facts in a lemma:

**Lemma 7.5** (Fundamental lemma of regularization theory)**.*** If** φα **is a regularizing filter and** R**α* =* φα*(*K*^(*∗*)*K*)*K*^(*∗*)*we have*

(1) *∥**KR**α**∥≤**C**φ*

(2) *∥**R**α**∥≤ *q

*C**φ *sup *λ**∈*]0,*∥**K**∥*^(2)]

q

*|**φα*(*λ*)*|*

(3) *K*^(†)*y** −**R**α**y* = ∑ *n *(1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(*n*)^())^(2)^() )D *x*^(†),* vn *E *vn **for** y** ∈**D*(*K*^(†))* and** x*^(†)^( )=* K*^(†)*y*.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 39


---

## Page 40

30.05.2022, VL 7-4

*Proof.* We first compute

*KR**α**y* =* K**φα*(*K*^(*∗*)*K*)*K*^(*∗*)*y* = ∑ *n **φα*(*σ*^(2 )*n*^())^(*σ*)^(*n*)* *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*Kvn*) ^(=)^( )∑ *n **φα*(*σ*^(2 )*n*^())^(*σ*)^(2 )*n** *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩*)^(*un*)

and then get that

*∥**KR**α**y**∥*^(2)
### ^( )*Y* ^(=)^( )∑
 *n **|**φα*(*σ*^(2 )*n*^())^(*σ*)^(2 )*n** *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩|*)^(2)

*≤*sup *n **|**φα*(*σ*^(2 )*n*^())^(*σ*)^(2 )*n*^(*|*)^(2)^(*∥*)^(*y*)^(*∥*)^(2 )*Y*

which, by definition of the constant* C**φ*, implies the claim (1). For the claim (2) compute

*∥**R**α**y**∥*^(2 )*X* ^(=)^(* ⟨*)^(*R*)^(*α*)^(*y*)^(,)^(* R*)^(*α*)^(*y*)^(*⟩*)^(=)^( )∑ *n **φα*(*σ*^(2 )*n*^())^(*σ*)^(*n*)* *^(*⟨*)^(*y*)^(,)^(* un*)^(*⟩⟨*)^(*R*)^(*α*)^(*y*)^(,)^(* vn*)^(*⟩*)

### = ∑
 *n **φα*(*σ*^(2 )*n*^())^(* ⟨*)^(*y*)^(,)^(* un*)^(*⟩⟨*)^(*R*)^(*α*)^(*y*)^(,)^(* K*)^(*∗*)^(*un*)^(*⟩*)

### = ∑
 *n **φα*(*σ*^(2 )*n*^())^(* ⟨*)^(*y*)^(,)^(* un*)^(*⟩⟨*)^(*KR*)^(*α*)^(*y*)^(,)^(* un*)^(*⟩*)

*≤*sup *n **|**φα*(*σ*^(2 )*n*^())^(*|*)∑ *n **⟨**y*,* un**⟩⟨**KR**α**y*,* un**⟩*

*≤*sup *n **|**φα*(*σ*^(2 )*n*^())^(*|*)

 
## ∑
 *n **⟨**y*,* un**⟩*^(2 )!1/2  
## ∑
 *n **⟨**KR**α**y*,* un**⟩*

!1/2

(by Cauchy-Schwarz)

*≤*sup *n **|**φα*(*σ*^(2 )*n*^())^(*|∥*)^(*y*)^(*∥∥*)^(*KR*)^(*α*)^(*y*)^(*∥*)

*≤*sup *n **|**φα*(*σ*^(2 )*n*^())^(*|*)^(*C*)^(*φ*)^(*∥*)^(*y*)^(*∥*)^(2 )*Y *(by claim (1))

which proves the claim. Finally, for claim (3) we note that if* x*^(†)^( )= *K*^(†)*y*, then* K*^(*∗*)*Kx*^(†)^( )=* K*^(*∗*)*y* and thus

*R**α**y* =* φα*(*K*^(*∗*)*K*)*K*^(*∗*)*y* =* φα*(*K*^(*∗*)*K*)*K*^(*∗*)*Kx*

and we get

*K*^(†)*y** −**R**α**y* = (id* −**φα*(*K*^(*∗*)*K*)*K*^(*∗*)*K*)*x*^(†)^( )= ∑ *n *(1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(2 )*n*^()) )D *x*^(†),* vn *E *vn*.

**Theorem 7.6** (Regularization with regularizing filters)**.*** Let** φα** be a regularizing filter and** R**α* =* φα*(*K*^(*∗*)*K*)*K*^(*∗*)*. Then it holds for all** y** ∈ **D*(*A*^(†))* that*

*R**α**y** *^(*α*)^(*→*)^(0 )*−→**K*^(†)*y*.

*Proof.* By Lemma 7.5 (3) we have

*∥**K*^(†)*y** −**R**α**y**∥*^(2)
### ^( )*X* ^(=)^( )∑
 *n *(1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(2 )*n*^()))^(2)^(*| *)D *x*^(†),* vn *E *|*^(2).

Since* φα*(*λ*)* →*1/*λ* for* α** →*0 we get (1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(2 )*n*^()))^(* →*)^(0)^( for)^(* α*)^(* → *)0. Moreover,* |*1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(2 )*n*^())^(*| ≤*)^(1)^( +)^(* C*)^(*φ*) ^(and hence, the convergence )*R**α**y** →**K*^(†)*y* follows from the dominated convergence theorem (Theorem 3.5).

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 40


---

## Page 41

30.05.2022, VL 7-5

Next we will show that there is a general strategy to construct a parameter choice rule that turns regularizing filters into conver- gent regularization methods.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 41


---

## Page 42

13.06.2022, VL 8-1

### **8 Parameter choice and error estimates**

We will now investigate the problem of a-priori parameter choice for general spectral regularization methods of the type* R**α* = *φα*(*K*^(*∗*)*K*)*K*^(*∗*)for a regularizing filter* φα*.

**Theorem 8.1.*** Let** K*^(†)^(* *)*be non-continuous and** R**α** be a regularization of** K*^(†)*. Then it holds: An a priori parameter choice** α*(*δ*)* fulfills that **∥**R**α*(*δ*)*y*^(*δ*)^(* *)*−**x*^(†)*∥**X** →*0* for** δ** →*0* exactly if*

*(i)** α*(*δ*)* →*0* for** δ** →*0*, and*

*(ii)** δ* sup0*<**λ**≤∥**K**∥*2 np

*|**φα*(*λ*)*| *o *→*0* for** δ** →*0*.*

*Proof.* We start with our standard error decomposition

*∥**R**α**y*^(*δ*)^(* *)*−**x*^(†)*∥**X** ≤∥**R**α**∥**δ* +* ∥**R**α**y*^(†)^(* *)*−**K*^(†)*y*^(†)*∥**X*.

By Theorem 7.6 and* α*(*δ*)* →*0 we get that the second term on right hand side goes to zero. By Lemma 7.5 (2) we have that* ∥**R**α**∥≤*

sup*λ**∈*]0,*∥**K**∥*2] np

*|**φα*(*λ*)*| *o and hence, the first term goes to zero as well. Conversely, assume that either (i) or (ii) does not hold. Let’s start with the case where (i) does not hold. Then* R**α*(*δ*) does not converge to* K*^(†)^( )pointwise. Hence, we can even set* y*^(*δ*)^( )=* y** ∈**D*(*K*^(†)), *x*^(†)^( )=* K*^(†)*y* and get that* ∥**R**α*(*δ*)*y*^(*δ*)^(* *)*−**x*^(†)*∥**X* =* ∥**R**α*(*δ*)*y** −**K*^(†)*y**∦→*0.

If (i) is fulfilled, but (ii) not, there exists* δ**n* with* δ**n n**→*∞ *−→*0 such that* δ**n**∥**R**α*(*δ**n*)*∥**>** ϵ* for some* ϵ*. Hence, there exists a sequence *zn** ∈**Y* with* ∥**zn**∥**Y* = 1 and* δ**n**∥**R**α*(*δ**n*)*zn**∥**X** >** ϵ*. Now let* y** ∈**D*(*K*^(†)) and set* yn* :=* y* +* δ**nzn*. Then* ∥**y** −**yn**∥*=* δ*, but

*R**α*(*δ**n*)*yn** −**K*^(†)*y* = (*R**α*(*δ**n*)*y** −**K*^(†)*y*) +* δ**nR**α*(*δ**n*)*zn** ̸→*0.

Now we aim for more sophisticated error estimates. What is needed, is a better estimate of the approximation error. As we have seen in Theorem 6.5, this is not possible without additional assumptions. By Lemma 7.5 (3) we have that=

*K*^(†)*y*^(†)^(* *)*−**R**α**y*^(†)^( )= ∑ *n *(1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(2 )*n*^()) )D *x*^(†),* vn *E *vn *(*)

if* x*^(†)^( )=* K*^(†)*y*^(†). However, bounding this error just in terms of* α* is not possible, since the decay of the terms 
 *x*^(†),* vn * is not known (the sequence has to be square summable, but that’s basically all we know). Here is a simple way to get a useful error bound:

We assume that our true solution* x*^(†)^( )is in rg(*K*^(*∗*)).

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 42


---

## Page 43

13.06.2022, VL 8-2

How does that help? Well, in this case we have some* w*^(†)^( )with *x*^(†)^( )=* K*^(*∗*)*w*^(†), we get from (*)

*∥**K*^(†)*y*^(†)^(* *)*−**R**α**y*^(†)*∥*^(2)
### ^( )*X* ^(=)^( )∑
 *n *(1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(2 )*n*^()))^(2)^(*| *)D *x*^(†),* vn *E *|*^(2)

### = ∑
 *n *(1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(2 )*n*^()))^(2)^(*| *)D *K*^(*∗*)*w*^(†),* vn *E *|*^(2)

### = ∑
 *n *(1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(2 )*n*^()))^(2)^(*| *)D *w*^(†),* Kvn *E *|*^(2)

### = ∑
 *n *(1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(2 )*n*^()))^(2)^(*σ*)^(2 )*n*^(*| *)D *w*^(†),* vn *E *|*^(2). (**)

Now we may get an error bound, if we can control the coeﬃcients (1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(2 )*n*^()))^(2)^(*σ*)^(2 )*n*^(. Expressed in the variable)^(* λ*)^( =)^(* σ*)^(2)^( this says )that we have to control the function* λ** 7→*(1* −**λφα*(*λ*)) *√*

*λ*. Let us investigate the situation for the truncated SVD and Tikhonov regularization:

*Example* 8.2*. *1. For the truncated SVD we have* φα*(*λ*) = 1/*λ *for* λ** ≥**α* and = 0 else. So we get Note the we change the threshold in comparison to Example 5.7: There we took 1/*σ**n* if* σ**n** <** α* and here we use 1/*σ**n* =* σ**n*/*σ*^(2)*n* if* σ*^(2)*n** >** α*, i.e. for* σ**n** > **√**α*. (1* −**λφα*(*λ*)) *√*

*λ* =  0 : *λ** ≥**α **√*

*λ *: *λ** <** α*

and thus, (1* −**λφα*(*λ*)) *√*

*λ** ≤*^(*√*)*α* or, equivalently

(1* −**σ*^(2)*φα*(*σ*^(2)))^(2)*σ*^(2)^(* *)*≤**α*.

Using this in (**), we obtain for the approximation error

*∥**K*^(†)*y*^(†)^(* *)*−**R**α**y*^(†)*∥*^(2)
### ^( )*X* ^(=)^( )∑
 *n *(1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(2 )*n*^()))^(2)^(*σ*)^(2 )*n*^(*| *)D *w*^(†),* vn *E *|*^(2)

### *≤**α*∑
 *n **| *D *w*^(†),* vn *E *|*^(2)^(* *)*≤**α**∥**w**∥*^(2 )*Y*

and thus

*∥**K*^(†)*y*^(†)^(* *)*−**R**α**y*^(†)*∥**X** ≤*^(*√*)

*α**∥**w*^(†)*∥**Y*.

2. For Tikhonov regulrization we have* φα*(*λ*) = 1/(*λ* +* α*) and we get

(1* −**λφα*(*λ*)) *√*

*λ* = (1* − **λ λ*+*α*^() )*√*

*λ* =* *^(*α *)*√*

*λ λ*+*α*^(.)

We want to maximize the right hand side over* λ** ≥*0. To this end we define* f* (*λ*) = *√*

*λ λ*+*α*^(, calculate)^(* f*)^(* ′*)^(()^(*λ*)^() =)^( 1)

2 *αλ*^(*−*)^(1/2)*−**λ*^(1/2)

(*λ*+*α*)^(2)

and see that* f** *^(*′*)(*λ*) = 0 for* λ* =* α*. Hence, we get that* *^(*α *)*√*

*λ λ*+*α** *^(*≤*)

*α*^(*√*)*α α*+*α* ^(=)^( 1)

2 *√**α*.

Again, using this in (**) gives* ∥**K*^(†)*y*^(†)^(* *)*−**R**α**y*^(†)*∥*^(2 )*X** *^(*≤*)^(*α*)

4^(*∥*)^(*w*)^(†)^(*∥*)^(2 )*Y *and thus

*∥**K*^(†)*y*^(†)^(* *)*−**R**α**y*^(†)*∥**X** ≤ √**α*

2* *^(*∥*)^(*w*)^(†)^(*∥*)^(*Y*)^(.)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 43


---

## Page 44

13.06.2022, VL 8-3

We conclude: If the unknown solution fulfills* x*^(†)^(* *)*∈*rg(*K*^(*∗*)), then the total error of both the truncated SVD and Tikhonov reg- ularization can be estimated by

*∥**x*^(*δ *)*α** *^(*−*)^(*x*)^(†)^(*∥*)^(*X*)* *^(*≤*)^(*δ*)^(*∥*)^(*R*)^(*α*)^(*∥*)^(+)^(* √*)

*α**C*

for some constant* C* (independent of* α* and* δ*). This our first quan- titative error bound, i.e. an upper bound for the total error that is explicit in* δ* and* α*. We also know that* ∥**R**α**∥≤*sup*λ *p

*|**φα*(*λ*)*| *and for both the truncated SVD and Tikhonov regularization we conclude by a simple calculation that* ∥**R**α**∥≤*1/^(*√*)*α*. This gives the even more explicit error estimate

*∥**x*^(*δ *)*α** *^(*−*)^(*x*)^(†)^(*∥*)^(*X*)* *^(*≤ *)*δ **√**α* +* *^(*√*)

*α**C*.

Now we can even choose the regularization parameter* α* in an optimal way: We can minimize the right hand side over* α* and see that the minimum is attained for* α*(*δ*) =* δ*/*C* and this gives the *error estimate*

*∥**x*^(*δ *)*α** *^(*−*)^(*x*)^(†)^(*∥*)^(*X*)* *^(*≤*)^(2 )*√*

*C **√*

*δ*.

Even if we do not know the constant* C*, we could still set* α* propor- tional to* δ*, i.e.* α*(*δ*) =* c**δ* for some constant* c*, and obtain

*∥**x*^(*δ *)*α** *^(*−*)^(*x*)^(†)^(*∥*)^(*X*)* *^(*≤O*)^(()^(*δ*)^(1/2)^())^(.)

Results of this form are called* convergence rates* of regularization methods. Assumption on the unknown solution* x*^(†)^( )such as* x*^(†)^(* *)*∈*rg(*K*^(*∗*)) are called* source conditions*. △

We will come back to error estimates and convergence rates later. Now we briefly discuss the most popular a posteriori parameter choice rule: Recall that our standing assumption is that the true data* y*^(†)^( )and our measured data* y*^(*δ*)^( )always fulfill* ∥**y*^(†)^(* *)*−**y*^(*δ*)*∥**Y** ≤**δ*. The main idea now is to look at the residuum for a reconstruction *R**α**y*^(*δ*), i.e. to consider

*∥**KR**α**y*^(*δ*)^(* *)*−**y*^(*δ*)*∥**Y*.

The residuum for the minimum norm solution* x*^(†)^( )fulfills* ∥**Kx*^(†)^(* *)*− **y*^(*δ*)*∥**Y* =* ∥**y*^(†)^(* *)*−**y*^(*δ*)*∥**Y** ≤**δ*, thus it seems reasonable to not aim for a smaller residuum for any other reconstruction. This is the idea of the following:

**Morozov’s discrepancy principle:** For some* δ** >* 0 and *y*^(*δ*)^( )with* ∥**y*^(†)^(* *)*−**y*^(*δ*)*∥**Y** ≤**δ* choose* α* =* α*(*δ*,* y*^(*δ*)) (as large as possible) such that

*∥**KR**α**y*^(*δ*)^(* *)*−**y*^(*δ*)*∥**Y** ≤**τδ*

for some* τ** >* 1. We want to choose* α* as large as possi- ble, to have the most stable reconstruc- tion. Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 44


---

## Page 45

13.06.2022, VL 8-4

*Remark* 8.3*.* This principle does not work without assumptions: For* y*^(†)^(* *)*∈*rg(*K*)^(*⊥*)*\ {*0*}* and exact data* y*^(*δ*)^( )=* y*^(†)^( )(i.e.* δ* = 0) even the minimum norm solution* x*^(†)^( )fulfills

*∥**Kx*^(†)^(* *)*−**y*^(*δ*)*∥**Y* =* ∥**KK*^(†)*y*^(†)^(* *)*−**y*^(†)*∥**Y* =* ∥**P*rg* Ky*^(†)^(* *)*−**y*^(†)*∥*=* ∥**y*^(†)*∥**Y** >* 0 =* τδ*.

Therefore one usually assumes that the range rg* K* is dense in* Y *(since then rg* K*^(*⊥*)=* {*0*}*).

For a practical realization of Morozov’s discrepancy principle one usually defines a decreasing sequence* α**n** →*0, computes *R**α**ny*^(*δ*)^( )for* n* = 1, 2, . . . and stops when* ∥**R**α**ny*^(*δ*)^(* *)*−**y*^(*δ*)*∥**Y** ≤**τδ* for the first time. This always works if the range of* K* is dense:

**Theorem 8.4.*** Let** R**α* =* φα*(*K*^(*∗*)*K*)*K*^(*∗*)*be a regularization of** K*^(†)*,* rg* K **dense in** Y**,** α**n** be a strictly decreasing null sequence and** τ** >* 1*. Then it holds: For all** y*^(†)^(* *)*∈**D*(*K*^(†))*, all** δ** >* 0* and** y*^(*δ*)^(* *)*with** ∥**y*^(†)^(* *)*−**y*^(*δ*)*∥**Y** ≤**δ **there exists an** n*^(*∗*)*such that for all** n** <** n*^(*∗*)

*∥**KR**α**n**∗**y*^(*δ*)^(* *)*−**y*^(*δ*)*∥**Y** ≤**τδ** <** ∥**KR**α**ny*^(*δ*)^(* *)*−**y*^(*δ*)*∥**Y*.

*Proof.* We study* ∥**KR**α**y*^(*δ*)^(* *)*−**y*^(*δ*)*∥*in dependence on* α*. Using the filter we get

*∥**KR**α**y*^(*δ*)^(* *)*−**y*^(*δ*)*∥*^(2)^( )= ∑ *n *(1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(2 )*n*^()))^(2)^(*| *)D *y*^(*δ*),* un *E *|*^(2)^( )+* ∥**P*rg(*K*)*⊥*(*y*^(*δ*))*∥*^(2)

### = ∑
 *n *(1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(2 )*n*^()))^(2)^(*| *)D *y*^(*δ*),* un *E *|*^(2)

since rg(*K*)^(*⊥*)=* {*0*}*. As we have seen in Theorem 7.6, the right hand side goes to zero which proves the claim.

If we do not assume that rg(*K*) is dense, we only get that* ∥**KR**α**y*^(*δ*)^(* *)*−**δ**∥ → ∥**P*rg(*K*)*⊥**y*^(*δ*)*∥≤∥**y*^(*δ*)*∥*. Hence, we need

that* ∥**y*^(*δ*)*∥≤**δ* for Morozov’s discrep- ancy principle to work. This seems rea- sonable: There should be “more signal than noise”.

We will show later that Morozov’s principle does indeed give a convergent regularization and now make a few remarks on* heuristic *rules: First and foremost, there a negative result, named* Bakushinkii veto*.

**Theorem 8.5** (Bakushinkii veto)**.*** Let** R**α** be a regularization for** K*^(†)*. If there exists a heuristic choice rule** α* =* α*(*y*^(*δ*))* such that* (*R**α*,* α*)* is a convergent regulariztion, then** K*^(†)^(* *)*is continuous.*

*Proof.* Let* α* :* Y** →*]0, ∞[ by such a parameter choice. Now let* y** ∈ **D*(*K*^(†)) and consider* yn** ∈**D*(*K*^(†)) with* yn** →**y*. But then (trivially) *∥**yn** −**yn**∥**Y** ≤**δ* for every* δ** >* 0 and by definition of convergent regularization we get* ∥**K*^(†)*yn** −**R**α*(*yn*)*yn**∥**Y* = 0, i.e.* R**α*(*yn*)*yn* = *K*^(†)*yn*. Moreover, we have for* δ**n* =* ∥**y** −**yn**∥**Y* (again by definition of convergent regulariztion) that

*K*^(†)*yn* =* R**α*(*yn*)*yn** →**K*^(†)*y*

which proves continuity of* K*^(†).

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 45


---

## Page 46

13.06.2022, VL 8-5

This theorem shows that heuristic rules only exist for inverse problems that aren’t ill-posed. Despite this negative result there are several heuristic rules that work remarkably well in practice. This phenomenon is still not fully understood. One explanation could be that one usually faces a perturbation* y*^(*δ*)^( )=* y*^(†)^( )+* η* by noise* η* in practice, but our theory uses general* η** ∈**Y*, i.e. also perturbations which do not look like noise at all are considered. Some rules that work well in practice are the quasi-optimality principle, the Hanke- Raus rule, the L-curve method, and generalized cross validation.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 46


---

## Page 47

20.06.2022, VL 9-1

### **9 Convergence rates and smoothness spaces**

In this section we will focus on the question on how to establish convergence rates, i.e. under what circumstances we can find a function* ψ* : ]0, ∞[* →*]0, ∞[ with* ψ*(*δ*)* →*0 for* δ** →*0 such that

*∥**R**α**y*^(*δ*)^(* *)*−**K*^(†)*y*^(†)*∥**X** ≤**ψ*(*δ*)

for a-priori or a-posteriori parameter choice rules. Recall that by Theorem 6.5 this can not hold without any further assumptions, i.e. it can’t be true if we consider* y*^(†)^( )arbitrary in the range of* K* (or, equivalently,* x*^(†)^( )arbitrary in* X*). However, we have seen in Example 8.2 that such a result can be achieved for the truncated SVD and Tikhonov regularization (for the a-priori choice rule* α*(*δ*) =* C **√*

*δ* and* ψ*(*δ*) =* C **√*

*δ*) if we assume* x*^(†)^(* *)*∈*rg(*K*^(*∗*)) ⊊*X*. This assumption is some kind of “abstract smoothness assumption”. The notion of smoothness that we will need will be formulated in terms of the operator* K* and may look confusing at first:

**Definition 9.1.** Let* X*,* Y* be Hilbert spaces and* K** ∈**K*(*X*,* Y*). For *ν** ≥*0 we define the subspaces* X*^(*ν*)^(* *)*⊂**X* as

*X*^(*ν*)^( ):= rg(*|**K**|*^(*ν*)) = n *|**K**|*^(*ν*)*z ** z** ∈*ker(*K*)*⊥*o .

Some first observations:

• For* ν* = 2*k* we have that

*X*^(2)^(*k*)^( )= rg(*|**K**|*^(2)^(*k*)) = rg( *√*

*K*^(*∗*)*K *2*k*) = rg((*K**∗**K*)*k*).

• If* ν** >** µ*, then* |**K**|*^(*ν*)^( )=* |**K**|*^(*µ*)*|**K**|*^(*ν*)^(*−*)^(*µ*), i.e. rg(*|**K**|*^(*ν*))* ⊂*rg(*|**K**|*^(*µ*)) and thus* X*^(*ν*)^(* *)*⊂**X*^(*µ*), i.e. the spaces get smaller, the larger the *ν*. The boundary case is* X*^(0)^( )= ker(*K*)^(*⊥*).

• The spaces* X*^(*ν*)^( )are characterized by summability assump- tions of the coeﬃcients in the singular basis: If there is *z* such that* x* =* |**K**|*^(*ν*)*z* we have by definition of* |**K**|*^(*ν*)^( )that *x* = ∑*n** σ*^(*ν *)*n** *^(*⟨*)^(*z*)^(,)^(* vn*)^(*⟩*)^(*un*) ^(and hence)

## ∑
 *n **σ*^(*−*)^(2)^(*ν *)*n **|⟨**x*,* vn**⟩|*^(2)^( )= ∑ *n **σ*^(*−*)^(2)^(*ν *)*n **σ*^(2)^(*ν *)*n** *^(*|⟨*)^(*z*)^(,)^(* vn*)^(*⟩|*)^(2)^( =)^(* ∥*)^(*z*)^(*∥*)^(2 )*X** *^(*<*)^( ∞)^(.)

In other words: For* x** ∈**X*^(*ν*)^( )we need that the sequence* ⟨**x*,* vn**⟩ *decays fast enough such that the decay of* |⟨**x*,* vn**⟩|*^(2)^( )compen- sates the growth of* σ*^(*−*)^(2)^(*ν *)*n *.

The last observation motivates the following definition:

**Definition 9.2.** On* X*^(*ν*)^( )we define the norm

*∥**x**∥*^(2 )*ν* ^(:)^(=)^(* ∥*)^(*z*)^(*∥*)^(2)
### ^( )*X* ^(=)^( )∑
 *n **σ*^(*−*)^(2)^(*ν *)*n **|⟨**x*,* vn**⟩|*^(2), *X*^(*ν*)^( )=* {**x** | ∥**x**∥**ν** <* ∞*}*

We call these norms* ν**-norms*. Note that the notation* X*^(*ν*)^( )and* ∥·∥**ν *do not include their dependency on* K*. Since these spaces are only considered for one* K* at a time, this usually does not lead to confusion. Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 47


---

## Page 48

20.06.2022, VL 9-2

In a certain sense, these norms measure smoothness, i.e. some *ν*-norm of* x* is finite, only if* x* is somehow smooth and the larger we can take* ν* while* ∥**x**∥**ν* stays finite, the smoother* x* is. A bit more precise: The* X*^(*ν*)^( )spaces demand a certain decay of the expansion coeﬃcients with respect to the singular basis; the faster the coef- ficients* ⟨**x*,* vn**⟩*decay, the “smoother” the* x* is. As we have already observed: The singular vectors* vn* with large* n* are usually highly oscillating and thus, they can not contribute much to a function that is in some* X*^(*ν*)^( )with large* ν*. In the following example we can make this a bit more precise:

*Example* 9.3*.* We consider the following integral operator:

*K f* (*t*) =

1 Z

0 *k*(*t*,* s*)* f* (*s*)d*s*, *k*(*t*,* s*) = * t*(1* −**s*) : *t** ≤**s s*(1* −**t*) : *s** ≤**t* ^(.)

This operator maps* L*^(2)([0, 1]) into itself and is compact (cf. Exam-

0

0.5

1

0 0.5

1 0

0.2

ple 4.3). One can show that

*K f* =* g** ⇐⇒ **−**g**′′* =* f*, and *g*(0) =* g*(1) = 0 .

Not a full proof but the first calculations: The equation* g* =* K f *is

*g*(*t*) =

1 Z

0 *k*(*s*,* t*)* f* (*s*)d*s*

=

*t *Z

0 *s*(1* −**t*)* f* (*s*)d*s* +

1 Z

*t t*(1* −**s*)* f* (*s*)d*s*

= (1* −**t*)

*t *Z

0 *s f* (*s*)d*s* +* t*

1 Z

*t *(1* −**s*)* f* (*s*)d*s*.

We directly see that* g*(0) =* g*(1) = 0 follows. We take the derivative on both sides (using the known rules) gives

*g*^(*′*)(*t*) =* −*

*t *Z

0 *s f* (*s*)d*s* + (1* −**t*)*t f* (*t*) +

1 Z

*t *(1* −**s*)* f* (*s*)d*s** −**t*(1* −**t*)* f* (*t*)

=* −*

*t *Z

0 *s f* (*s*)d*s* +

1 Z

*t *(1* −**s*)* f* (*s*)d*s*.

The second derivative is

*g*^(*′′*)(*t*) =* −**t f* (*t*)* −*(1* −**t*)* f* (*t*) =* −**f* (*t*).

In principle one can argue similarly in the opposite direction as well.

Moreover* K* is selfadjoint, since* k*(*s*,* t*) =* k*(*t*,* s*). One can also show that the SVD of* K* is given by

*σ**n* = 1 (*π**n*)^(2 , )*un*(*t*) =* vn*(*t*) = *√*

2 sin(*π**nt*).

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 48


---

## Page 49

20.06.2022, VL 9-3

This allows us to characterize the spaces* X*^(*ν*)^( )a quite explicitly: By Definition 9.2 we have

*f** ∈**X*^(*ν*)^(* *)*⇐⇒∥**f** ∥*^(2 )*ν* ^(=)^( 2)^(*π*)^(4)^(*ν*)^( )∑ *n n*^(4)^(*ν*)*|⟨**f*, sin(*π**n**·*)*⟩|*^(2)^(* *)*<* ∞.

Now one can show the following: For* f** ∈**C*^(2)^(*ν*)([0, 1]) with* f* ^(()^(2)^(*k*)^())(0) = *f* ^(()^(2)^(*k*)^())(1) = 0 for* k* = 0, . . . ,* ν** −*1 it holds that

*∥**f** ∥**ν* =* ∥**f* ^(()^(2)^(*ν*)^())*∥**L*2.

We start from the right and use that* vn*(*t*) = *√*

2 sin(*π**nt*) is an orthonormal basis of* L*^(2)([0, 1]) to get

*∥**f* ^(()^(2)^(*ν*)^())*∥*^(2 )*L*^(2)^( =)^( ∑ )*n **| *D *f* ^(()^(2)^(*ν*)^()),* vn *E *|*^(2).

For the inner products we use integration by parts two times, the boundary conditions of* f* and that the sine functions vanish at the boundary to get

D *f* ^(()^(2)^(*ν*)^()),* vn *E =

1 Z

0 *f* ^(()^(2)^(*ν*)^())(*t*) *√*

2 sin(*π**nt*)d*t*

=* f* ^(()^(2)^(*ν*)^(*−*)^(1)^())(*t*) *√*

2 sin(*π**nt*)  1

0 | {z } =0

*−*

1 Z

0 *f* ^(()^(2)^(*ν*)^(*−*)^(1)^())(*t*) *√*

2(*π**n*) cos(*π**nt*)d*t*

=* f* ^(()^(2)^(*ν*)^(*−*)^(2)^())(*t*) *√*

2(*π**n*) cos(*π**nt*)  1

0 | {z } =0

+

1 Z

0 *f* ^(()^(2)^(*ν*)^(*−*)^(2)^())^(*√*)

2(*π**n*)^(2 )sin(*π**nt*)d*t*

= (*π**n*)^(2)^( D )*f* ^(()^(2)^(*ν*)^(*−*)^(2)^()),* vn *E .

Recursively,this gives D *f* ^(()^(2)^(*ν*)^()),* vn *E = (*π**n*)^(2)^(*ν*)^(* *)*⟨**f*,* vn**⟩*and hence

*∥**f* ^(()^(2)^(*ν*)^())*∥*^(2 )*L*^(2)^( =)^( 2)^(*π*)^(4)^(*ν*)^( ∑ )*n n*^(4)^(*ν*)*|⟨**f*, sin(*π**n**·*)*⟩|*^(2)^( )=* ∥**f** ∥*^(2 )*ν*

as claimed.

This can be used to rigorously prove that it holds

*X*^(*ν*)^( )=

 *f** ∈**C*^(()^(2)^(*ν*)^())([0, 1]) * f* (2*k*)(0) =* f* (2*k*)(1) = 0 	*∥·∥**ν*,

i.e. the space* X*^(*ν*)^( )is the closure of the space of 2*ν*-times continuous diﬀerentiable functions with respective boundary conditions with respect to the* ν*-norm. △

Our aim is to construct methods (*R**α*,* α*) such that the total error* ∥**R**α**y*^(*δ*)^(* *)*−**x*^(†)*∥**X* is small. First we will establish a baseline and analyze the question of “how good can a reconstruction method *R* :* Y** →**X* be in general”. We introduce a little more notation:

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 49


---

## Page 50

20.06.2022, VL 9-4

**Definition 9.4** (Worst case error in* ν*-spaces)**.** Let* K** ∈**K*(*X*,* Y*) and *R* :* Y** →**X* continuous with* R*0 = 0. We define

*E**ν*(*δ*,* ρ*,* R*) = sup n *∥**Ry*^(*δ*)^(* *)*−**x*^(†)*∥**X ** x*†* ∈**X**ν*,* ∥**Kx*†* −**y**δ**∥**Y** ≤**δ*,* ∥**x*†*∥**ν** ≤**ρ *o .

This quantity is the* worst case total error of the method** R** over solutions in an** X*^(*ν*)*-ball of radius** ρ** and noise level** δ*. Furthermore we define

*E**ν*(*δ*,* ρ*) = inf* {**E**ν*(*δ*,* ρ*,* R*)* |** R* :* Y** →**X* continous with* R*0 = 0*}* .

This is the* best possible worst case error of any method over solutions in an** X*^(*ν*)*-ball of radius** ρ** and noise level** δ*.

This “best possible worst case error” may look a bit weird, but is actually not hard to quantify:

**Theorem 9.5.*** For** K** ∈**K*(*X*,* Y*)* it holds that*

*E**ν*(*δ*,* ρ*)* ≥*sup* {∥**x**∥**X** | ∥**Kx**∥**Y** ≤**δ*,* ∥**x**∥**ν** ≤**ρ**}* =:* e**ν*(*δ*,* ρ*).

The quantity on the right hand side has a simple explanation: It is a so-called *modulus of continuity* of* K*^(†)^( )restricted to some set defined by* δ*,* ν* and* ρ*. These three parameters have the following meaning: *δ*: Noise level *ν*: Degree of smoothness *ρ*: “Largeness” in the smoothness class. The noise level is often available (or can be estimated), the smoothness may be guessed, but the largeness in the smoothness class is basically never known.

*Proof.* Let* x** ∈**X*^(*ν*)^( )with* ∥**Kx**∥**Y** ≤**δ* and* ∥**x**∥**ν** ≤**ρ* and* R* :* Y** →**X *be arbitrary (given the conditions). We set* y* =* Kx* and* y*^(*δ*)^( )= 0. Then this* x* is in the feasible set for the supremum in the definition of* E**ν*(*δ*,* ρ*,* R*) and thus* E**ν*(*δ*,* ρ*,* R*)* ≥∥**x**∥**X*. Taking the supremum over all these* x* we get the inequality* e**ν*(*δ*,* ρ*)* ≤**E**ν*(*δ*,* ρ*,* R*). The claim follows by taking the infimum over all* R*.

The following theorem shows, how large the lower bound *e**ν*(*δ*,* ρ*) of the best worst case error can be:

**Theorem 9.6.*** Let** K** ∈**K*(*X*,* Y*)*. Then it holds for all** ν*,* ρ*,* δ** >* 0* that*

*e**ν*(*δ*,* ρ*)* ≤**δ ν ν*+1* ρ *1 *ν*+1 .

*Moreover, there exists a sequence** δ**n** with** δ**n n**→*∞ *−→*0* such that there is equality along that sequence.*

*Proof.* Let* x** ∈**X*^(*ν*)^( )with* ∥**x**∥**ν** ≤**ρ* and* ∥**Kx**∥**Y** ≤**δ*. Then there is *z** ∈**X* such that* x* =* |**K**|*^(*ν*)*z*. We would like to estimate* ∥**x**∥**X* = *∥|**K**|*^(*ν*)*z**∥**X* in terms of* ∥**z**∥**X* =* ∥**x**∥**ν* and* ∥**Kx**∥**Y*. To that end we use the following result (also known as* interpolation inequality*): For

*r** >** s** ≥*0 and all* x* it holds that* ∥|**K**|*^(*s*)*x**∥**X** ≤∥|**K**|*^(*r*)*x**∥ **s r X*^(*∥*)^(*x*)^(*∥ *)1*−*^(*s*)

*r X *.

The definition of* |**K**|*^(*s*)^( )gives

*∥|**K**|*^(*s*)*x**∥**X* = ∑ *n **σ*^(2)^(*s *)*n** *^(*|⟨*)^(*x*)^(,)^(* vn*)^(*⟩|*)^(2.)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 50


---

## Page 51

20.06.2022, VL 9-5

Now we define sequences* an* =* σ*^(2)^(*s *)*n** *^(*|⟨*)^(*x*)^(,)^(* vn*)^(*⟩|*)^(2)^(* s*)

*r* and* bn* =

*|⟨**x*,* vn**⟩|*^(2)^(*−*)^(2)^(* r*)

*s* and numbers* p* =* r*/*s* and* q* =* r*/(*r** −**s*). We use the Hölder inequality to get

*∥|**K**|*^(*s*)*x**∥*^(2 )*X* ^(=)^( )∑ *n **σ*^(2)^(*s *)*n** *^(*|⟨*)^(*x*)^(,)^(* vn*)^(*⟩|*)^(2)

### = ∑
 *n anbn** ≤*

 
### ∑
 *n a*^(*p *)*n*

!1/*p **·*

 
### ∑
 *n b*^(*q *)*n*

!1/*q*

=

 
### ∑
 *n **σ*^(2)^(*r *)*n** *^(*|⟨*)^(*x*)^(,)^(* vn*)^(*⟩|*)^(2 )!*s*/*r **·*

 
### ∑
 *n **|⟨**x*,* vn**⟩|*^(2 )!(*r**−**s*)/*r*

=* ∥|**K**|*^(*r*)*x**∥*^(2)^(*s*)^(/)^(*r *)*X **∥**x**∥*^(2)^(()^(*r*)^(*−*)^(*s*)^())^(/)^(*r *)*X*

which proves the claim.

We use this claim with* s* =* ν* and* r* =* ν* + 1 and

*∥**x**∥**X* =* ∥|**K**|*^(*ν*)*z**∥**X** ≤∥|**K**|*^(*ν*)^(+)^(1)*z**∥*

*ν ν*+1 *X **∥**z**∥*

1 *ν*+1 *X*

=* ∥**K** |**K**|*^(*ν*)*z *| {z } =*x **∥*

*ν ν*+1 *X **∥**z**∥*

1 *ν*+1 *X **≤**δ ν ν*+1* ρ *1 *ν*+1 .

For the equality we set* δ**n* =* ρσ*^(*ν*)^(+)^(1 )*n *and* xn* =* ρ**|**K**|*^(*ν*)*vn*. One can show that this gives indeed equality.

Note that by definition of* |**K**|*^(*ν*)^( )we have* x* =* ρσ*^(*ν *)*n*^(*vn*) ^(and by )definition of the* ν*-norm we have that* ∥**x**∥**ν* =* ∥**ρ**vn**∥**X* =* ρ *and* ∥**Kx**∥**Y* =* ∥**ρσ*^(*ν *)*n*^(*Kvn*)^(*∥*)*Y* ^(=)^(* ρσν *)*n*^(*∥*)^(*σ*)^(*nun*)^(*∥*)^(=)^(* ρσν*)^(+)^(1 )*n *=* δ**n* as needed. From* δ**n* =* ρσ*^(*ν*)^(+)^(1 )*n *we get that* σ**n* = (*δ**n*/*ρ*)^(1/)^(()^(*ν*)^(+)^(1)^())

and thus

*∥**x**∥**X* =* ρσ*^(*ν *)*n* ^(=)^(* ρ *) *δ**n*

*ρ * *ν ν*+1 =* δ*

*ν ν*+1 *n **ρ *1 *ν*+1

as desired.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 51


---

## Page 52

27.06.2022, VL 10-1

### **10 Convergence rates for spectral regularization**

The above Theorems 9.5 and 9.6 give a benchmark with which we can compare regularization methods: They can not do better than

the right hand side* δ ν ν*+1* ρ *1 *ν*+1 for data* x*^(†)^( )from* X*^(*ν*)^( )with* ∥**x*^(†)*∥**ν** ≤**ρ*. We fix this in the following definition:

**Definition 10.1.** A regularization method (*R**α*,* α*) is called* optimal *for parameters* ρ* and* ν*, if for all* x*^(†)^( )with* ∥**x*^(†)*∥**ν** ≤**ρ* and* y*^(*δ*)^( )with *∥**Kx*^(†)^(* *)*−**y*^(*δ*)*∥**Y** ≤**δ* it holds that

*∥**R**α**y*^(*δ*)^(* *)*−**x*^(†)*∥**X* =* δ ν ν*+1* ρ *1 *ν*+1 .

We call the method* order optimal* for parameters* ρ* and* ν* if there exists some* C* such that for all* x*^(†)^( )as above it holds that

*∥**R**α**y*^(*δ*)^(* *)*−**x*^(†)*∥**X* =* C**δ ν ν*+1* ρ *1 *ν*+1 .

Finally, we call a method* order optimal* for* ν*, if for all* x*^(†)^(* *)*∈**X*^(*ν*)^( )and *y*^(*δ*)^( )with* ∥**Kx*^(†)^(* *)*−**y*^(*δ*)*∥≤**δ* there exists* C* such that

*∥**R**α**y*^(*δ*)^(* *)*−**x*^(†)*∥≤**C**δ ν ν*+1 .

The assumptions* ∥**x*^(†)*∥**ν** ≤**ρ* or* x*^(†)^(* *)*∈**X*^(*ν*)^( )are called* source conditions *and the element* z* with* |**K**|*^(*ν*)*z* =* x*^(†)^( )is called* source element*. In the case of Example 9.3, the source condition* x*^(†)^(* *)*∈**X*^(*ν*)^( )means that* x*^(†)^( )is 2*ν*-times (weakly) diﬀerentiable (with additional boundary conditions) with 2*ν*-th weak derivative* z* which is an* L*^(2)- function.

Recall the Definition 7.4 of a regularizing filter: A family of functions* φα* (piecewise continuous and bounded) on the interval [0,* κ*] (*κ* =* ∥**K**∥*^(2)) is a regularizing filter, if for* λ** >* 0 it holds that

*φα*(*λ*)* *^(*α*)^(*→*)^(0 )*−→*^(1)

*λ*^(, )*λ**|**φα*(*λ*)*| ≤**C**φ*

for some* C**φ** >* 0. Theorem 7.6 showed that regularizing filters indeed lead to convergent regularizations. Now we want to answer the question when a regularizing filter is optimal or order optimal. The key to such results are estimates of the approximation error under the assumption that* x*^(†)^( )lies in a* ρ*-ball in* X*^(*ν*), i.e. under the assumption that* x*^(†)^( )is “*ν*-smooth” and “*ρ*-large”. We can express such estimates for a given filter* φα* with the functions

*ων*(*α*) := sup 0*<**λ**≤**κ λ*^(*ν*)^(/2)*|**r**α*(*λ*)*| *(recall* r**α*(*λ*) = 1* −**λφα*(*λ*))

= sup 0*<**λ**≤**κ λ*^(*ν*)^(/2)*|*1* −**λφα*(*λ*)*|*.

These functions can be used to get bounds on the approximation error, the depend on* α*.

**Lemma 10.2.*** Let** y*^(†)^(* *)*∈**D*(*K*^(†))* and** x*^(†)^(* *)*∈**X*^(*ν*)^(* *)*with** ∥**x*^(†)*∥**ν** ≤**ρ**. Further define** x**α* =* R**α**y*^(†)*. Then it holds for all** α** >* 0* that*

*∥**x**α** −**x*^(†)*∥**X** ≤**ων*(*α*)*ρ*,

*∥**Kx**α** −**Kx*^(†)*∥**Y** ≤**ων*+1(*α*)*ρ*.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 52


---

## Page 53

27.06.2022, VL 10-2

*Proof.* If* ∥**x*^(†)*∥**ν** ≤**ρ* we have that* x*^(†)^( )=* |**K**|*^(*ν*)*w* = (*K*^(*∗*)*K*)^(*ν*)^(/2)*w *with* ∥**w**∥**X** ≤**ρ*. Then from Lemma 7.5 (3) and using 
 *x*^(†),* vn * = 
(*K**∗**K*)*ν*/2*w*,* vn * =* σν **n** *^(*⟨*)^(*w*)^(,)^(* vn*)^(*⟩*)^(we get)

*x*^(†)^(* *)*−**x**α* = ∑ *n *(1* −**σ*^(2 )*n*^(*φα*)^(()^(*σ*)^(2 )*n*^()) )D *x*^(†),* vn *E *vn*

### = ∑
 *n r**α*(*σ*^(2 )*n*^())^(*σν *)*n** *^(*⟨*)^(*w*)^(,)^(* vn*)^(*⟩*)^(*vn*)^(.)

Taking the squared norm gives

*∥**x*^(†)^(* *)*−**x**α**∥*^(2)
### ^( )*X* ^(=)^( )∑
 *n *(*r**α*(*σ*^(2 )*n*^())^(*σν *)*n*^())^(2)^(*|⟨*)^(*w*)^(,)^(* vn*)^(*⟩|*)^(2)

*≤**ων*(*α*)^(2)^( )∑ *n **|⟨**w*,* vn**⟩|*^(2)^(* *)*≤**ων*(*α*)^(2)*∥**w**∥*^(2 )*X*

as claimed. For the second claim recall from Lemma 7.3 that *∥|**K**|**x**∥**X* =* ∥**Kx**∥**Y* and thus

*∥**Kx**α** −**Kx*^(†)*∥**Y* =* ∥|**K**|*(*x**α** −**x*^(†))*∥*.

Moreover,

*|**K**|*(*x**α** −**x*^(†)) = (*K*^(*∗*)*K*)^(1/2)*r**α*(*K*^(*∗*)*K*)(*K*^(*∗*)*K*)^(*ν*)^(/2)*w* = ∑ *n **σ**nr**α*(*σ*^(2 )*n*^())^(*σν *)*n** *^(*⟨*)^(*w*)^(,)^(* vn*)^(*⟩*)^(*vn*)^(.)

Since* |**σ**nr**α*(*σ*^(2 )*n*^())^(*σν *)*n*^(*|*)^(2)^( =)^(* |*)^(*r*)^(*α*)^(()^(*σ*)^(2 )*n*^())^(*σν*)^(+)^(1 )*n **|*^(2)^(* *)*≤**ων*+1(*α*)^(2)^( )the second claim follows similar to the first one.

Now we can show how to achieve order optimal regularization:

**Theorem 10.3** (Order optimal a-priori parameter choice)**.*** Let** K** ∈ **K*(*X*,* Y*)*. If** φα** is a regularizing filter for which fulfills*

sup 0*<**λ**≤∥**K**∥*^(2)^(*|*)^(*φα*)^(()^(*λ*)^())^(*| ≤*)^(*C*)^(*φα*)^(*−*)^(1)

*ων*(*α*)* ≤**C**να*^(*ν*)^(/2).

*If the a-priori choice** α** fulfills*

*c * *δ ρ * 2 *ν*+1* ≤**α*(*δ*)* ≤**C * *δ ρ * 2 *ν*+1

*for some* 0* <** c** <** C**, then* (*R**α*,* α*)* is an order optimal regularization method in the sense of Definition 10.1.*

*Proof.* As always we start with our error decomposition

*∥**x*^(*δ *)*α*(*δ*)* *^(*−*)^(*x*)^(†)^(*∥*)^(*X*)^(* ≤*)^(*δ*)^(*∥*)^(*R*)^(*α*)^(()^(*δ*)^())^(*∥*)^(+)^(* ∥*)^(*x*)^(*α*)^(()^(*δ*)^())* *^(*−*)^(*x*)^(†)^(*∥*)^(*X*)^(.)

From Lemma 7.5 (2) and our first assumption we know that

*∥**R**α*(*δ*)*∥≤ *q

*C**φ *r

sup 0*<**λ**≤∥**K**∥*^(2)^(*|*)^(*φα*)^(()^(*δ*)^()()^(*λ*)^())^(*| ≤*)^(*C*)^(*φα*)^(()^(*δ*)^())^(*−*)^(1/2.)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 53


---

## Page 54

27.06.2022, VL 10-3

From Lemma 10.2 and our second assumption we get

*∥**x**α*(*δ*)* −**x*^(†)*∥**X** ≤**ων*(*α*(*δ*))*ρ** ≤**C**να*(*δ*)^(*ν*)^(/2)*ρ*.

We use these estimates in the error decomposition and use the upper and lower bound of the parameter choice to get

*∥**x*^(*δ *)*α*(*δ*)* *^(*−*)^(*x*)^(†)^(*∥*)^(*X*)^(* ≤*)^(*C*)^(*φα*)^(()^(*δ*)^())^(*−*)^(1/2)^(*δ*)^( +)^(* C*)^(*να*)^(()^(*δ*)^())^(*ν*)^(/2)^(*ρ*)

*≤**C**φ**c*^(*−*)^(1/2)*δ*^(*− *)1 *ν*+1* ρ *1 *ν*+1* δ* +* C**ν**C*^(*ν*)^(/2)*δ ν ν*+1* ρ*^(*− *)*ν ν*+1* ρ*

= (*C**φ**c*^(*−*)^(1/2)^( )+* C**ν**C*^(*ν*)^(/2))*δ ν ν*+1* ρ *1 *ν*+1 .

Let us investigate the few filters we know if they can lead to order optimal methods To that end, let us collect the inequalities that we need:

*Example* 10.4 (Order optimality of the truncated SVD)*.* The filter is *φα*(*λ*) = 1/*λ* for* λ** ≥**α* and = 0 else. Thus* r**α*(*λ*) = 0 for* λ** ≥**α *and = 1 else. We get

sup *λ **|**φα*(*λ*)*|* = ^(1)

*α* ^((attained at)^(* λ*)^( =)^(* α*)^() )=*⇒**C**φ* = 1,

and

*ων*(*α*) = sup *λ λ*^(*ν*)^(/2)*|**r**α*(*λ*)*|* =* α*^(*ν*)^(/2)^( )(attained at* λ* =* α*) =*⇒**C**ν* = 1.

We see that the truncated SVD is indeed an order optimal regu- larization method (for any* ν** >* 0)! As such, the method can make use of any smoothness in the (unknown) solution. The smoother *x*^(†), the better the convergence rate will be since the exponent *ν*/(*ν* + 1) of* δ* in Theorem 10.3 will be larger for larger* ν*. △

*Example* 10.5 (Order optimality and saturation for Tikhonov regu- larization)*.* The filter is* φα*(*λ*) = (*λ* +* α*)^(*−*)^(1)^( )and thus

*r**α*(*λ*) = 1* −**λφα*(*λ*) = 1* − **λ λ*+*α* ^(= )*α λ*+*α*^(.)

We get

sup *λ **|**φα*(*λ*)*|* = ^(1)

*α* ^((attained at)^(* λ*)^( =)^( 0)^() )=*⇒**C**φ* = 1.

For the other condition we need to investigate the supremum of *λ*^(*ν*)^(/2)*|**r**α*(*λ*)*|* =* α *^(*λν*)^(/2)

*λ*+*α*^(. We define)^(* f*)^( ()^(*λ*)^() =)^(* λν*)^(/2)

*λ*+*α* ^(and note:)

•* ν** >* 2: The function* f* is unbounded on ]0, ∞[ (since* ν*/2* > *1) and hence, the supremum is infinite.

•* ν* = 2. Here* f* (*λ*)* <* 1 and even* f* (*λ*)* →*1 for* λ** →*∞and thus

sup *λ λ*^(*ν*)^(/2)*|**r**α*(*λ*)*|* =* α* =* α*^(*ν*)^(/2), =*⇒**C*1 = 1.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 54


---

## Page 55

27.06.2022, VL 10-4

• 0* <** ν** <* 2: Here we have* f* (0) = 0 and* f* (*λ*)* →*0 for* λ** →*0 and hence a finite maximum exists. The condition* f** *^(*′*)(*λ*) = 0

is 0 = (* *^(*ν*)

2^(*λ *)*ν *2* *^(*−*)^(1)(*λ* +* α*)* −**λ ν *2 )/(*λ* +* α*)^(2)^( )which holds exactly if

*λ ν *2* *^(*−*)^(1)(* *^(*ν*)

2^(()^(*λ*)^( +)^(* α*)^())^(* −*)^(*λ*)^() =)^( 0)^( Since)^( 0)^(* <*)^(* λ*)^(* <*)^( 2)^( the only solution )is* λ* = *ν *2*−**ν*^(*α*)^( and corresponds to a maximum. We plug this )is and get

sup *λ λ*^(*ν*)^(/2)*|**r**α*(*λ*)*|* =* α *� *ν *2*−**ν*^(*α *)*ν*/2 � *ν *2*−**ν*^(*α*)^( +)^(* α *)*−*1

=* α *� *ν *2*−**ν **ν*/2* α ν *2 ^(� )2 2*−**ν **−*1* α**−*1

= � *ν *2*−**ν ** *^(*ν*)

2 ^(2)^(*−*)^(*ν*)

2* *^(*αν*)^(/2)

=* *^(*νν*)^(/2)^(()^(2)^(*−*)^(*ν*)^()()^(2)^(*−*)^(*ν*)^())^(/2)

2 *α*^(*ν*)^(/2)^( )=*⇒**C**ν* = *√*

*ν*^(*ν*)(2*−**ν*)^(2)^(*−*)^(*ν*)

2 .

In conclusion: Tikhonov regularization is an order optimal

The constant* C**ν* is not as bad as it may look. Here is* C**ν* is dependence on* ν*:

0 0.5 1 1.5 2 0 0.2 0.4 0.6 0.8

1 regularization method for 0* ≤**ν** ≤*2 but not for* ν** >* 2. As such, it can take advantage of smoothness up to the space *X*^(2)^( )= rg* |**K**|*^(2)^( )= rg(*K*^(*∗*)*K*).

△

While the a-priori rule from Theorem 10.3 indeed leads to order optimal methods, one drawback is that they need knowledge about both* ν* and* ρ*. Without knowledge of* ρ* one could still choose *α*(*δ*) ∝*δ*^(2/)^(()^(*ν*)^(+)^(1)^())^( )and obtain on order optimal method (just go through the estimates in the proof and see that you get the right exponent for* δ* in the end). The symbol ∝indicates that the left hand side is proportional to the right hand side, i.e. that* α*(*δ*) =* C**δ*^(2/)^(()^(*ν*)^(+)^(1)^())

for some* C*. We could also consider *c**δ*^(2/)^(()^(*ν*)^(+)^(1)^())^(* *)*≤**α*(*δ*)* ≤**C**δ*^(2/)^(()^(*ν*)^(+)^(1)^())

Morozov’s discrepancy principle (the only a-posteriori method we know) does not need any knowledge about* ν* or* ρ*. Remarkably, this parameter choice also turns out to be order optimal (but in slightly less cases). Before we can formulate this, we make one more definition:

**Definition 10.6.** Let* φα* be a regularizing filterwithsup0*<**λ**≤∥**K**∥*2*|**φα*(*λ*)*| ≤ **C**φα*^(*−*)^(1). We say that this filter has* qualification** ν*0 if* ων*(*α*)* ≤**C**να*^(*ν*)^(/2)

is fulfilled for all 0* ≤**ν** ≤**ν*0. (It it holds for all* ν** ≥*0 we say that the qualification is* ν*0 = ∞).

**Theorem 10.7** (Optimality of Morozov’s discrepancy principal)**. ***The** φα** be a regularizing filter with qualification** ν*0* >* 0* and let*

*τ** > *sup *α**>*0, 0*<**λ**≤**κ **|**r**α*(*λ*)*|* =:* Cr*.

*Then the parameter choice defined by Morozov’s discrepancy principle with this** τ** (cf. Section 8) is an order optimal regularization method for *0* ≤**ν** ≤**ν*0* −*1*.*

Unfortunately, the proof does not fit into the lecture. It can be found in the lecture notes “Regularization of inverse problems” by Christian Clason, Theorem 5.11, available at https://arxiv. org/abs/2001.00617.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 55


---

## Page 56

27.06.2022, VL 10-5

Here is an example that shows that the results of Theorem 10.3 is indeed quite close to what one can observe in practice:

% Dimension of the problem n = 2000; % deriv2 from the regu-toolbox (from MATLAB’s file exchange) implements the % "inverse of the negative second derivative" from Example 8.3 [A,~,~] = deriv2(n);

% The constant 1 function should be in X^nu for nu<1/4. % We take the edge case anyway. x = ones(n,1); nu = 1/4;

% now do three choices for alpha of the form alpha = delta^s % one optimal , the other too large and small, resp. % According to Theorem 9.3 the optimal s is s=2/(nu+1) = 2/(5/4) = 1.6 % expected rate for alpha = delta^s is C*delta^r with r= min(1-s/2,s*nu/2) % See the proof of Theorem 9.3 s1 = 1.6; r1 = min(1-s1/2,s1*nu/2) s2 = 1; r2 = min(1-s2/2,s2*nu/2) s3 = 3; r3 = min(1-s3/2,s3*nu/2)

% Some noise levels going to zero deltas = logspace(0,-5,20); % Some noise levels going to zero % Precompute A’*A ATA = A’*A; y = A*x; for k = 1:length(deltas)

% for each delta construct data with that noise level delta = deltas(k); noise = randn(n,1);noise = noise/norm(noise); ydelta = y + delta*noise; C = 1; % C could be anything. It’s choice does not affect the rate,

% but it does affect the values or the errors. % these are our alphas alpha1 = C*delta^s1; alpha2 = C*delta^s2; alpha3 = C*delta^s3;

% Precompute A’*ydelta ATydelta = A’*ydelta; % reconstruct by Tikhonov

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 56


---

## Page 57

27.06.2022, VL 10-6

x1 =(ATA+alpha1*eye(n))\(ATydelta); x2 =(ATA+alpha2*eye(n))\(ATydelta); x3 =(ATA+alpha3*eye(n))\(ATydelta);

% measure the errors error1(k) = norm(x1 - x); error2(k) = norm(x2 - x); error3(k) = norm(x3 - x); end

offset = error1(1); %used to adjust the plots % plor total errors: loglog(deltas ,error1 ,deltas ,error2,deltas,error3) hold on % plot the expected rates loglog(deltas ,offset*deltas.^r1,... deltas ,offset*deltas.^r2,... deltas ,offset*deltas.^r3) legend(’s=1.6’, ’s=1’, ’s=3’,’rate for s=1.6’, ’rate for s=1’, ’rate for s=3’)

r1 = 0.2000 r2 = 0.1250 r3 = -0.5000

One should add that the plot changes quite a bit if we move to even smaller noise levels. There we will see that the plot for *s* = 3 (much too small regularization parameter) will go down again, contrary to what has been predicted by theory. This may due to eﬀects of the finite precision of floating point arithmetic.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 57


---

## Page 58

04.07.2022, VL 11-1

### **11 Iterative regulariztion**

The idea of iterative regularization is to use iterative methods that can either solve* Kx* =* y* or minimize* ∥**Kx** −**y**∥*^(2 )*Y* ^(exactly in the case )where* y* is in the range of* K* (or the domain of the pseudo inverse in the latter case), apply them to the case with some* y*^(*δ*)^( )instead of* y*, even though* y*^(*δ*)^( )is not in the range of* K*. We expect that the method will not converge in this case and hence, we stop them at some point. The simplest iterative regularization method is the *Landweber method* and we motivate the method two times:

*Example* 11.1 (Landweber as a fixed point iteration)*.* We start from the normal equations* K*^(*∗*)*Kx* =* K*^(*∗*)*y* and rewrite them as

*x* =* x** −**ω*(*K*^(*∗*)*Kx** −**K*^(*∗*)*y*) =* x** −**ω**K*^(*∗*)(*Kx** −**y*)

for some* ω** ∈***R** We turn this into a fixed point iteration

*xn*+1 =* xn** −**ω**K*^(*∗*)(*Kxn** −**y*).

For the sake of simplicity we start with* x*0 = 0. We analyze the convergence of the iteration by Banach’s fixed point theorem. The map under consideration is* x** 7→**x** −**ω**K*^(*∗*)(*Kx** −**y*), so we analyze

*∥**x** −**ω**K*^(*∗*)(*Kx** −**y*)* −*(*x*^(*′ *)*−**ω**K*^(*∗*)(*Kx*^(*′ *)*−**y*)*∥**X* =* ∥*(*I** −**ω**K*^(*∗*)*K*)(*x** −**x*^(*′*))*∥**X*.

We see that the iteration map is a contraction if* ∥*id* −**ω**K*^(*∗*)*K**∥**<* 1. If this holds, we can see inductively that from* x*0 = 0* ∈*rg(*K*^(*∗*)) it follows that* xn** ∈*rg(*K*^(*∗*)) as well, and hence we get convergence *xn** →**x*^(†)^( )=* K*^(†)*y*. For* y*^(*δ*)^( )/*∈*rg(*K*) we can’t expect convergence, and need to stop early. Here the stopping index* m* act as regularization parameter, but to be consistent with our convention “smaller regu- larization parameter is less regularization” it’s more like* α* = 1/*m *is the regularization parameter. △

**Lemma 11.2.*** If* 0* <** ω** <* 2/*∥**K**∥*^(2)*, then** ∥*id* −**ω**K*^(*∗*)*K**∥≤*1*.*

*Proof.* For the singular values* σ**j* of* K* we know that 0* <** σ**j** ≤ ∥**K**∥*. The operator id* −**ω**K*^(*∗*)*K* is self adjoint and has eigenvalues 1* −**ωσ*^(2 )*j** *^(*∈*)^(])^(1)^(* −*)^(*ω*)^(*∥*)^(*K*)^(*∥*)^(2, 1)^([)^( and since)^( 0)^(* <*)^(* ω*)^(* <*)^( 2/)^(*∥*)^(*K*)^(*∥*)^(2)^( we have )that the eigenvalues of id* −**ω**K*^(*∗*)*K* lie in ]*−*1, 1] as needed.

As a consequence, convergence of the Landweber method does not follow from Banach fixed point theorem (it does so, if* K* is injective and the smallest singular value exists and is positive, but then the problem is well posed). Here is another view on the Landweber method.

*Example* 11.3 (Landweber as gradient descent on the least squares functional)*.* We can also start with the least squares functional

*f* (*x*) = ^(1)

2^(*∥*)^(*Kx*)^(* −*)^(*y*)^(*∥*)^(2 )*Y*^(.)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 58


---

## Page 59

04.07.2022, VL 11-2

To calculate the derivative of* f* we do

*f* (*x* +* h*) = ^(1)

2^(*∥*)^(*K*)^(()^(*x*)^( +)^(* h*)^())^(* −*)^(*y*)^(*∥*)^(2 )*Y* ^(=)^( 1)

2^(*∥*)^(*Kx*)^(* −*)^(*y*)^( +)^(* Kh*)^(*∥*)^(2 )*Y *1 2^(*∥*)^(*Kx*)^(* −*)^(*y*)^(*∥*)^(2 )*y* ^(+)^(* ⟨*)^(*Kx*)^(* −*)^(*y*)^(,)^(* Kh*)^(*⟩*)^(+)^( 1)

2^(*∥*)^(*Kh*)^(*∥*)^(2 )*Y *=* f* (*x*) +* ⟨**K*^(*∗*)(*Kx** −**y*),* h**⟩*+* φ*(*h*)

with* φ*(*h*) = 1 2^(*∥*)^(*Kh*)^(*∥*)^(2 )*Y*^(. Since)^(* φ*)^(()^(*h*)^())^(/)^(*∥*)^(*h*)^(*∥*)^(*X*)^(* ≤∥*)^(*K*)^(*∥*)^(2)^(*∥*)^(*h*)^(*∥→*)^(0)^( for )*h** →*0 we get that the gradient of* f* is

*∇**f* (*x*) =* K*^(*∗*)(*Kx** −**y*).

Hence, gradient descent for* f* with constant stepsize* ω* is

*xn*+1 =* xn** −**ω**∇**f* (*xn*) =* xn** −**ω**K*^(*∗*)(*Kx** −**y*)

which is exactly the Landweber iteration we in the previous exam- ple. △ The next lemma shows how the* n*-th iterate can be written explicitly:

**Lemma 11.4.*** If** x*0 = 0*, then the** m**-th iterate of the Landweber method with stepsize** ω** is given by*

*xn* =* ω **m**−*1
##  ∑
 *n*=0 (id* −**ω**K*^(*∗*)*K*)^(*n*)*K*^(*∗*)*y*.

*Proof.* We prove this by induction: For* m* = 1 we have

*x*1 =* ω**K*^(*∗*)*y* =* ω*(id* −**K*^(*∗*)*K*)^(0)*K*^(*∗*)*y*.

For the induction step we start with

*xm*+1 =* xm** −**ω**K*^(*∗*)(*Kxm** −**y*) = (id* −**ω**K*^(*∗*)*K*)*xm* +* ω**K*^(*∗*)*y*

= (id* −**ω**K*^(*∗*)*K*)

 

*ω **m**−*1
##  ∑
 *n*=0 (id* −**ω**K*^(*∗*)*K*)^(*n*)*K*^(*∗*)*y*

!

+* ω**K*^(*∗*)*y*

=* ω **m**−*1
##  ∑
 *n*=0 (id* −**ω**K*^(*∗*)*K*)^(*n*)^(+)^(1)*K*^(*∗*)*y* +* ω*(id* −**ω**K*^(*∗*)*K*)^(0)*K*^(*∗*)*y*

=* ω **m*
## * *∑
 *m*=0 (id* −**ω**K*^(*∗*)*K*)^(*n*)*K*^(*∗*)*y*.

Hence,* m* steps of the Landweber method are the same as

*xm* =* φ**m*(*K*^(*∗*)*K*)*K*^(*∗*)*y*

with the filter function

*φ**m*(*λ*) =* ω **m**−*1
##  ∑
 *n*=0 (1* −**ωλ*)^(*n*).

Using the geometric sum *m *∑ *k*=0 *q*^(*k*)^( )= ^(1)^(*−*)^(*qm*)^(+)^(1)

1*−**q *this gives

*φ**m*(*λ*) =* ω **m**−*1
##  ∑
 *n*=0 (1* −**ωλ*)^(*n*)^( )=* ω* ^(1)^(*−*)^(()^(1)^(*−*)^(*ωλ*)^())^(*m*)

1*−*(1*−**ωλ*) ^(=)^( 1)^(*−*)^(()^(1)^(*−*)^(*ωλ*)^())^(*m*)

*λ *. (1)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 59


---

## Page 60

04.07.2022, VL 11-3

**Theorem 11.5.*** Let** φ**m** be defines by* (1)*. Then it holds that** Rm* = *φ**m*(*K*^(*∗*)*K*)*K*^(*∗*)*defined a regularization if* 0* <** ω** <* 2/*∥**K**∥*^(2)*.*

*Proof.* By Theorem 7.6 we only need to show that* φ**m* is a regular- izing filter, i.e. that* φ**m*(*λ*)* →*1/*λ* for* m** →*∞and 0* <** λ** <** ∥**K**∥*^(2)

(recall that* α* = 1/*m* act as regularization parameter) and that *λφ**m*(*λ*) is uniformly bounded for all* m*. Since we have 0* <** ω** <* 2/*∥**K**∥*^(2)^( )we have for all* λ* with 0* < **λ** ≤∥**K**∥*^(2)^( )that

*−*1* <* 1* −**ωλ** <* 1,

and hence (1* −**ωλ*)^(*m*)^(* *)*→*0 for* m** →*∞. Moreover we have for 0* ≤**λ** ≤∥**K**∥*^(2)

*λ**|**φ**m*(*λ*)*|* =* |*1* −*(1* −**ωλ*)^(*m*)*| ≤*2 =:* C**φ*.

We can even show that the Landweber method is an order optimal method. Recall from Section 10 that this holds for a-priori parameter choices of the form *α*(*δ*) ∝*δ*^(2/)^(()^(*ν*)^(+)^(1)^())^( )so here we should stop after about* m*^(*∗*)*≈**δ*^(*−*)^(2/)^(()^(*ν*)^(+)^(1)^())^( )itera- tions.

**Theorem 11.6** (Landweber is order optimal)**.*** Let* 0* <** ω** <* 2/*∥**K**∥*^(2)*. Then the Landweber method with a-priori rule** m*^(*∗*)=* m*(*δ*) ∝*δ*^(*−*)^(2/)^(()^(*ν*)^(+)^(1)^())

*is an order optimal regularization method for any** ν** >* 0*.*

*Proof.* We use Theorem 10.3 and need to show that

sup 0*<**λ**≤∥**K**∥*^(2)^(*|*)^(*φ*)^(*m*)^(()^(*λ*)^())^(*| ≤*)^(*C*)^(*φ*)^(*m*)

*ων*(*m*) =* C**ν**m*^(*−*)^(*ν*)^(/2)

(recall that* α* = 1/*m* and don’t confuse* ων* with the stepsize* ω*). For the first estimate consider recall that* −*1* <* 1* −**ωλ** <* 1 and hence, by Bernoulli’s inequality

*|**φ**m*(*λ*)*|* =* *^(*|*)^(1)^(*−*)^(()^(1)^(*−*)^(*ωλ*)^())^(*m*)^(*|*)

*λ *= ^(1)^(*−*)^(()^(1)^(*−*)^(*ωλ*)^())^(*m*)

*λ **≤*^(1)^(*−*)^(()^(1)^(*−*)^(*m*)^(*λω*)^())

*λ *=* ω**m*.

From above we already had* C**φ* = 2, so now we should set* C**φ* = max(2,* ω*). To estimate* ων*(*λ*) we use consider

*λ*^(*ν*)^(/2)(1* −**λφ**m*(*λ*)) =* λ*^(*ν*)^(/2)(1* −**ωλ*)^(*m*)

and substitute* t* =* m**λ*. Then this expression becomes

*h*(*t*) = �* t*

*m **ν*/2 (1* −**ω**t*

*m* ^())^(*m*)^(.)

We use the elementary inequality (1* −*^(*x*)

*m*^())^(*m*)^(* ≤*)^(*e*)^(*−*)^(*x*)^( and get)

*h*(*t*)*m*^(*−*)^(*ν*)^(/2)*t*^(*ν*)^(/2)*e*^(*−*)^(*ω*)^(*t*).

The derivative is

*h*^(*′*)(*λ*) =*m*^(*−*)^(*ν*)^(/2)^(  )*ν *2^(*t*)^(*ν*)^(/2)^(*−*)^(1)^(*e*)^(*−*)^(*ω*)^(*t*)^( +)^(* t*)^(*ν*)^(/2)^(()^(*−*)^(*ω*)^())^(*e*)^(*−*)^(*ω*)^(*t*)^()

=* m*^(*−*)^(*ν*)^(/2)*t*^(*ν*)^(/2)^(*−*)^(1)*e*^(*−*)^(*ω*)^(*t*)^( �)^(* ν*)

2* *^(*−*)^(*t*)^(*ω *)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 60


---

## Page 61

04.07.2022, VL 11-4

and thus,* h* has a global maximum at* t* =* ν*/(2*ω*). This shows that

*h*(*t*)* ≤**m*^(*−*)^(*ν*)^(/2)^( �)^(* ν*)

2*ω **ν*/2* e**−**ν*/2,

and thus

*ων*(*λ*)* ≤**C**ν**m*^(*−*)^(*ν*)^(/2),

i.e.* C**ν* = �* ν*

2*ω ** *^(*ν*)

2* e*^(*−*)^(*ν*)^(/2).

Note that iterative methods are well suited for Morozov’s dis- crepancy principle. One just monitors* ∥**Kxm** −**y*^(*δ*)*∥**Y* during the iteration and stops at the first* m*^(*∗*)such that* ∥**Kxm*^(*∗*)*−**y*^(*δ*)*∥**Y** ≤**τδ*. One can actually show a little bit more here: Let us denote by *x*^(*δ *)*m* ^(the)^(* m*)^(-th iterate of the Landweber method with)^(* y*)^(*δ*)^( instead of)^(* y*)^(.)

**Theorem 11.7.*** If** Kx*^(*δ *)*m** *^(*−*)^(*y*)^(*δ*)^(* ̸*)^(=)^( 0)^(*, then it holds for stepsizes*)^( 0)^(* <*)^(* ω*)^(* < *)2/*∥**K**∥*^(2)^(* *)*that*

*∥**Kx*^(*δ *)*m*+1* *^(*−*)^(*y*)^(*δ*)^(*∥*)^(*Y*)* *^(*≤∥*)^(*Kx*)^(*δ *)*m** *^(*−*)^(*y*)^(*δ*)^(*∥*)^(*Y*)^(.)

*Moreover, if** ∥**Kx*^(*δ *)*m** *^(*−*)^(*y*)^(*δ*)^(*∥*)^(*Y*)* *^(*>*)^( 2)^(*δ*)^(* and*)^( 0)^(* <*)^(* ω*)^(* <*)^( 1/)^(*∥*)^(*K*)^(*∥*)^(2)^(* we even have*)

*∥**x*^(*δ *)*m*+1* *^(*−*)^(*x*)^(†)^(*∥*)^(*X*)* *^(*<*)^(* ∥*)^(*x*)^(*δ *)*m** *^(*−*)^(*x*)^(†)^(*∥*)^(*X*)^(.)

*Proof.* We compute from the iteration

*Kx*^(*δ *)*m*+1* *^(*−*)^(*y*)^(*δ*)^( =)^(* K*)^((()^(id)^(* −*)^(*ω*)^(*K*)^(*∗*)^(*K*)^())^(*x*)^(*δ *)*m* ^(+)^(* ω*)^(*K*)^(*∗*)^(*y*)^(*δ*)^(* −*)^(*y*)^(*δ*)

= (id* −**ω**K*^(*∗*)*K*)(*Kx*^(*δ *)*m** *^(*−*)^(*y*)^(*δ*)^())^(.)

For stepsize* ω** ∈*]0, 2/*∥**K**∥*^(2)[ we get that* ∥*id* −**ω**K*^(*∗*)*K**∥≤*1 and thus shows the first claim. For the second claim we write* z*^(*δ *)*m* ^(:)^(=)^(* y*)^(*δ*)^(* −*)^(*Kx*)^(*δ *)*m* ^(and)^(* y*)^( =)^(* Kx*)^(†)

and get

*∥**x*^(*δ *)*m*+1* *^(*−*)^(*x*)^(†)^(*∥*)^(2 )*X* ^(=)^(* ∥*)^(*x*)^(*δ *)*m** *^(*−*)^(*x*)^(†)^(* −*)^(*ω*)^(*K*)^(*∗*)^(()^(*Kx*)^(*δ *)*m** *^(*−*)^(*y*)^(*δ*)^())^(*∥*)^(2 )*X*

=* ∥**x*^(*δ *)*m** *^(*−*)^(*x*)^(†)^(*∥*)^(2 )*X* ^(+)^( 2)^(*ω *)D *x*^(*δ *)*m** *^(*−*)^(*x*)^(†,)^(* K*)^(*∗*)^(*z*)^(*δ *)*m *E +* ω*^(2)*∥**K*^(*∗*)*z*^(*δ *)*m*^(*∥*)^(2 )*X*

=* ∥**x*^(*δ *)*m** *^(*−*)^(*x*)^(†)^(*∥*)^(2 )*X* ^(+)^( 2)^(*ω *)D *Kx*^(*δ *)*m** *^(*−*)^(*Kx*)^(†,)^(* z*)^(*δ *)*m *E +* ω*^(2)*∥**K*^(*∗*)*z*^(*δ *)*m*^(*∥*)^(2 )*X*

=* ∥**x*^(*δ *)*m** *^(*−*)^(*x*)^(†)^(*∥*)^(2 )*X* ^(+)^(* ω *)D *z*^(*δ *)*m* ^(+)^( 2)^(*Kx*)^(*δ *)*m** *^(*−*)^(2)^(*y*)^(,)^(* z*)^(*δ *)*m *E +* ω*(*ω**∥**K*^(*∗*)*z*^(*δ *)*m*^(*∥*)^(2 )*X** *^(*−∥*)^(*z*)^(*δ *)*m*^(*∥*)^(2 )*Y*^(.)

We aim to show that the last two terms are negative. For the first term we compute D *z*^(*δ *)*m* ^(+)^( 2)^(*Kx*)^(*δ *)*m** *^(*−*)^(2)^(*y*)^(,)^(* z*)^(*δ *)*m *E = D *y*^(*δ*)^(* *)*−**Kx*^(*δ *)*m* ^(+)^( 2)^(*Kx*)^(*δ *)*m** *^(*−*)^(2)^(*y*)^(,)^(* z*)^(*δ *)*m *E

= D *y*^(*δ*)^( )+* Kx*^(*δ *)*m** *^(*−*)^(2)^(*y*)^(,)^(* z*)^(*δ *)*m *E

= 2 D *y*^(*δ*)^(* *)*−**y*,* z*^(*δ *)*m *E *−∥**z*^(*δ *)*m*^(*∥*)^(2 )*Y*

*≤*2*δ**∥**z*^(*δ *)*m*^(*∥*)^(2 )*Y** *^(*−∥*)^(*z*)^(*δ *)*m*^(*∥*)^(2 )*Y *= (2*δ** −∥**Kx*^(*δ *)*m** *^(*−*)^(*y*)^(*δ*)^(*∥*)^(*Y*)^())^(*∥*)^(*z*)^(*δ *)*m*^(*∥*)^(*Y*)* *^(*<*)^( 0.)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 61


---

## Page 62

04.07.2022, VL 11-5

For the second term we use* ω** <* 1/*∥**K**∥*^(2)^( )to get

*ω**∥**K*^(*∗*)*z*^(*δ *)*m*^(*∥*)^(2 )*X** *^(*≤*)^(*ω*)^(*∥*)^(*K*)^(*∥*)^(2)^(*∥*)^(*z*)^(*δ *)*m*^(*∥*)^(2 )*Y** *^(*<*)^(* ∥*)^(*z*)^(*δ *)*m*^(*∥*)^(2 )*Y*

which shows that the last term is negative as well.

The theorem shows two important things: First, the Landwe- ber method always decreases the residual but, more importantly, even the distance to the* true* solution decreases if the residual is larger than 2*δ*, so it is always beneficial to use* τ** <* 2 in Morozov’s discrepancy principle. Note that the Landweber method can al- ways be applied when one is able to apply the operator* K* and its adjoint* K*^(*∗*). The is no need for singular value decomposition (as for the TSVD) and also we do not need to solve linear systems (like for Tikhonov regularization). One downside of the Landwe- ber method is that it usually needs a lot of iterations until a good reconstruction is achieved (or Morozov’s discrepancy principle kicks is). In practice one can use other iterative methods that solve the normal equations, e.g. the method of conjugate gradients (CG) which converges much faster. One can show (with very diﬀerent tool than here) that, combined with the discrepancy principle, is indeed a regularization method. Here is an example with the Landweber iteration:

% problem size and matrix n = 500; A = tril(ones(n))/n;

% discretized interval t = linspace(0,1,n)’;

% some true solutions. Uncomment the one you want to use %xdag = 1-t.^2; xdag = max(1-2*t,0); %xdag = (t<0.5); % true data ydag = A*xdag;

% noisy data eta = randn(n,1); eta = eta/norm(eta); % normalized noise delta = 0.05; % noise level ydelta = ydag + delta*eta;

% stepsize for the Landweber iteration normA = norm(A); omega = 1/normA; % constant for Morozov ’s discrepancy principle tau = 1.01;

% number of iterations

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 62


---

## Page 63

04.07.2022, VL 11-6

m = 3000; % initialization x = zeros(n,1);

residual = zeros(m,1); error = zeros(m,1); stopped = false; for k=1:m x = x - omega*A’*(A*x-ydelta); error(k) = norm(x-xdag); residual(k) = norm(A*x-ydelta); % if the residual falls below the noise level for the first time % record the index and the reconstruction at that time if stopped==false && residual(k)<tau*delta mstar = k; xrec = x; stopped = true; end end

semilogy(1:m,error ,1:m,residual ,1:m,tau*delta*ones(m,1)) title(’semi convergence of the Landweber method’) legend(’||x_m-x^+||’, ’||Kx_m-ydelta||’, ’\tau\delta’)

fprintf(’stopping index: m* = %d\n’,mstar)

subplot(1,2,1) plot(t,xrec,t,xdag) title(’Landweber stopped with Morozov’) legend(’xrec’,’xdag’) subplot(1,2,2)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 63


---

## Page 64

04.07.2022, VL 11-7

plot(t,A*xrec,t,ydag) title(’same on image side’) legend(’A*rec’,’ydag’)

stopping index: m* = 221

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 64


---

## Page 65

11.07.2022, VL 12-1

### **12 A Bayesian perspective on regulariztion**

In this section we would like to draw connections between regu- larization (especially Tikhonov regularization) and probabilistic approaches to inverse problems. This will be much simpler if we consider everything to be finite dimensional, i.e.* K** ∈***R**^(*m*)^(*×*)^(*n*)^( )is a matrix,* X* =** R**^(*n*)^( )and* Y* =** R**^(*m*). We start by modeling noise stochas- tically. Our data is* y*^(*δ*)^( )=* Kx*^(†)^( )+* η* where* η** ∈***R**^(*m*)^( )is a random vector, i.e. a realization of a random variable* H*. A vector valued random variable is a map* H* : Ω*→***R**^(*m*)^( )on some probability space Ω(which will actually not play a role). On Ωthere is probability measure* P *and the random variable* H* generates a probability distribution on** R**^(*m*)^( )by

*µ*(*B*) =* P*(*H*^(*−*)^(1)(*B*))

for all Borel sets* B** ⊂***R**^(*m*). However,* P* is never needed in practice and we only need to know the probability distribution* π*noise in **R**^(*m*)^( )since then

*µ*(*B*) =* P*(*H** ∈**B*) = Z

*B **π*noise(*η*)d*η*.

The mean value (or expectation) of* H* is

**E**(*H*) = Z

**R**^(*m *)*η* d*π*noise(*η*) = Z *ηπ*noise(*η*)d*η*

and the covariance is

cov(*H*) =** E**((*H** −***E**(*H*))(*H** −***E**(*H*))^(*T*)) = Z (*η** −**E*(*H*))(*η** −**E*(*H*))^(*T*)*π*noise(*η*)d*η*.

*Example* 12.1 (Gaussian noise)*.* The most simple (and also most widely used) example of additive noise is Gaussian noise. Usually we assume that the noise has zero mean and for simplicity we assume that the components of the random vectorare independent and identically distributed (i.i.d), each with variance* σ*^(2). Then the probability distribution of one entry of* H* is

1 *σ **√*

2*π*^(*e *)*−**x*^(2)

2*σ*^(2 ).

Hence, the full probability distribution of* H* is

*π*noise(*η*) = *m*
## * *∏
 *i*=1

1 *σ **√*

2*π*^(*e *)*−**η*^(2 )*i *2*σ*^(2)

= 1 *σ*^(*m*)^(*√*)

2*π **m e **−∥**η**∥*^(2)

2*σ*^(2 ).

The covariance of* H* is cov(*H*) =* σ*^(2)*Im* (where* Im* is the* m** ×** m *identity matrix). △

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 65


---

## Page 66

11.07.2022, VL 12-2

The goal of probabilistic approaches is to gain as much infor- mation as possible about the* posterior distribution** π*posterior(*x** |** y*), i.e. the distribution of the solution* x*, given that the data* y* has been observed. The posterior distribution is, by Bayes theorem,

*π*posterior(*x** |** y*) =* *^(*π*)^(()^(*y*)^(* |*)^(* x*)^())^(*π*)^(prior)^(()^(*x*)^())

*π*(*y*) .

where* π*prior(*x*) is the so-called* prior distribution*,* π*(*y** |** x*) is the probability of measuring the data* y* given that* x* is the solution and* π*(*y*) is the probability of the data* y*.

*Example* 12.2 (Maximum a-posteriori estimation)*.* One crucial in- formation that one can get from the posterior distribution is the mode of the distribution which is nothing else that the maximum *x*^(*∗*)*∈*argmax*x** π*posterior(*x** |** y*). This* x*^(*∗*)is the most likely* x* given the measured data* y* and the assumptions we made. This gives us a *point estimate* for our solution* x* and this one is called the* maximum a-posteriori estimator* or MAP estimator. The computation of the MAP estimator amounts the solution of a maximization problem in* n* dimensions. △

The MAP estimator may be the most likely* x*^(*∗*), but is it is not necessarily a typical one. One example of this phenomenon al- ready occurs for very simple distributions: A standard Gaussian distribution has its mode at zero while a sample that you draw has expected norm* *^(*√*)*n* in* n* dimensions. One can even show that it holds that the probability that the norm* ∥**x**∥*2 of a vector with independent standard Gaussian entries deviates from* *^(*√*)*n* is very small, more precisely

*P *�*|∥**x**∥*2* −√*

*n**| ≥**t ** ≤*2 exp(*−**ct*2)

for some* c*.

*Example* 12.3 (Conditional mean estimator)*.* Another point estimate of a distribution is its mean/expected value. Hence, we could also consider the so-called* conditional mean* of the posterior which is

*x*^(*∗*)=** E**(*x** |** y*) = Z

**R**^(*n *)*x** π*(*x** |** y*)d*x*

(provided that the integral exists). The computation of the con- ditional mean amount to the computation of* n* integrals (recall that* x** ∈***R**^(*n*)) over the full** R**^(*n*). Since* n* is the dimension of the solu- tion, this is by no means an easy task and standard approximation techniques for integrals (such as the trapezoidal rule) can not be applied. △

In the following we will only consider the MAP estimate fur- ther. For the computation of the MAP estimator we maximize *π*posterior(*x**|**y*) over* x*. Using Bayes theorem we note that we do not need to know anything about* π*(*y*), but only* π*(*y** |** x*) and* π*prior(*x*) are needed.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 66


---

## Page 67

11.07.2022, VL 12-3

*Example* 12.4 (Additive Gaussian noise again)*.* If we assume that our solution* X* and the noise* H* are independent, the probability density of* H* does not change, when we condition it on the realiza- tion* X* =* x*. Since* y* =* Kx* +* η* we also see that the* Y* conditioned on* X* =* x* is distributed like* H* but translated by* Kx*, i.e.

*π*(*y** |** x*) =* π*noise(*y** −**Kx*). (*)

In the case a Gaussian noise as above we get

*π*(*y** |** x*) = 1 *σ*^(*m*)^(*√*)

2*π **m e **−∥**y**−**Kx**∥*^(2)

2*σ*^(2 ).

△

Collection what we have so far we see that we still need to specify the prior distribution for* x*. This is up to us; we can design a prior distribution in any way we like. More precisely, we should design the prior such that it reflects all the prior information that we have about the solution. Once we have the prior, we can start thinking about how to compute the MAP estimator. If we assume a Gaussian prior, we actually end up with Tikhonov regularization:

**Theorem 12.5** (MAP for Gaussian prior and Gaussian noise gives Tikhonov regularization)**.*** Assume that** K** ∈***R**^(*m*)^(*×*)^(*n*)*,** y*^(*δ*)^(* *)*∈***R**^(*m*)^(* *)*be the data,** x*0* be an initial guess and** σ*,* τ** >* 0*. Further let the distribution of the noise and the prior be*

*π*noise(*η*) = 1 *σ*^(*m*)^(*√*)

2*π **m e **−∥**η**∥*^(2)

2*σ*^(2)

*π*prior(*x*) = 1 *τ*^(*n*)^(*√*)

2*π **n e **−∥**x**−**x*0*∥*^(2)

2*τ*^(2 ).

*Then the MAP estimator for** x** from** y*^(*δ*)^(* *)*is*

*x*^(*∗*)= (*K*^(*∗*)*K* +* *^(*σ*)^(2)

*τ*^(2 id)^())^(*−*)^(1)^(()^(*K*)^(*∗*)^(*y*)^(*δ*)^( +)^(* x*)^(0)^())

*Proof.* The posterior distribution is

*π*(*x** |** y*^(*δ*)) ∝*π*(*y*^(*δ*)^(* *)*|** x*)*π*prior(*x*)

Using (*) and the definition of* π*noise and* π*prior we get

*π*(*x** |** y*^(*δ*)) ∝*π*noise(*y*^(*δ*)^(* *)*−**Kx*)*π*prior(*x*)

= 1 *σ*^(*m*)^(*√*)

2*π **m e **−∥**y*^(*δ*)*−**Kx**∥*^(2)

2*σ*^(2 )1 *τ*^(*n*)^(*√*)

2*π **n e **−∥**x**−**x*0*∥*^(2)

2*τ*^(2 ).

To maximize this we equivalently maximize the logarithm of* π*(*x** | **y*^(*δ*)) (since everything is positive and the logarithm is monotone). This gives us the maximization problem.

argmax *x *log

 

1 *σ*^(*m*)^(*√*)

2*π **m e **−∥**y*^(*δ*)*−**Kx**∥*^(2)

2*σ*^(2 )1 *τ*^(*n*)^(*√*)

2*π **n e **−∥**x**−**x*0*∥*^(2)

2*τ*^(2 )!

= argmax *x*

h *−**m* log(*σ **√*

2*π*)* −*^(*∥*)^(*Kx*)^(*−*)^(*y*)^(*δ*)^(*∥*)^(2)

2*σ*^(2 )*−**n* log(*τ **√*

2*π*)* −*^(*∥*)^(*x*)^(*−*)^(*x*)^(0)^(*∥*)^(2)

2*τ*^(2 )i .

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 67


---

## Page 68

11.07.2022, VL 12-4

Since we only maximize with respect to* x* we can neglect the ad- ditive terms that do not depend on* x* and also scale by positive numbers o get

*x*^(*∗*)*∈*argmax *x **−*^(*∥*)^(*Kx*)^(*−*)^(*y*)^(*δ*)^(*∥*)^(2)

2*σ*^(2 )*−*^(*∥*)^(*x*)^(*−*)^(*x*)^(0)^(*∥*)^(2)

2*τ*^(2)

= argmin *x*

*∥**Kx**−**y*^(*δ*)*∥*^(2)

2*σ*^(2 )+* *^(*∥*)^(*x*)^(*−*)^(*x*)^(0)^(*∥*)^(2)

2*τ*^(2)

= argmin *x*

1 2^(*∥*)^(*Kx*)^(* −*)^(*y*)^(*δ*)^(*∥*)^(2)^( +)^(* σ*)^(2)

2*τ*^(2)^(* ∥*)^(*x*)^(* −*)^(*x*)^(0)^(*∥*)^(2.)

We recognize the Tikhonov functional with regularization param- eter* *^(*σ*)

*τ*^(. The minimizer)^(* x*)^(*∗*)^(is given by the solution of)

*K*^(*∗*)(*Kx*^(*∗*)*−**y*^(*δ*)) +* *^(*σ*)^(2)

*τ*^(2)^( ()^(*x*)^(*∗−*)^(*x*)^(0)^() =)^( 0)

which proves the claim.

The choice of the prior distribution is basically an art. Many suitable priors exists for various types of data. If we consider addi- tive Gaussian noise as above and a prior of the form* π*prior(*x*) = *e*^(*−*)^(*α*)^(Φ)^(()^(*x*)^())^( )one get, similarly to the above theorem, that the MAP estimate is gives as a solution of the minimization problem

min *x *1 2^(*∥*)^(*Kx*)^(* −*)^(*y*)^(*δ*)^(*∥*)^(2)^( +)^(* σ*)^(2)^(*α*)

2 ^(Φ)^(()^(*x*)^())^(.)

The noise distribution, however, is dictated by the noise model and if one does not have additive Gaussian noise, one can still formulate a regularization method:

*Example* 12.6 (Poisson noise)*.* The Poisson distribution models rare events. One example comes from photography-like applications with very low light as it occurs, for example, in electron microscopy. There each pixel collects incoming photos over a short time span. The number of incoming photons in some pixel* p*, when measured and averaged over a long time span, gives the true intensity* y*(*p*). Over a short time span, one collects a finite number of photons and the stochastic model for this number is that it is distributed according to the Poisson distribution with parameter* λ* =* y*(*p*) (the parameter* λ* is also the expected value of the distribution), this means that the probability to collect* y*^(*δ*)(*p*) =* k* photons in pixel* p* is

*P*(*y*^(*δ*)(*p*) =* k*) =* *^(*y*)^(()^(*p*)^())^(*ke*)^(*−*)^(*y*)^(()^(*p*)^())

*k*! .

Hence, the conditional probability that* y*^(*δ*)(*p*) is measured if* y*(*p*) is the true value is

*π*(*y*^(*δ*)(*p*)* |** y*(*p*)) =* *^(*y*)^(()^(*p*)^())^(*y*)^(*δ*)^(()^(*p*)^())^(*e*)^(*−*)^(*y*)^(()^(*p*)^())

(*y*^(*δ*)(*p*)! .

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 68


---

## Page 69

11.07.2022, VL 12-5

We still assume that the noise is the pixels is independent, i.e we have

*π*(*y*^(*δ*)^(* *)*|** y*) = ∏ *p*

*y*(*p*)^(*y*)^(*δ*)^(()^(*p*)^())*e*^(*−*)^(*y*)^(()^(*p*)^())

(*y*^(*δ*)(*p*)! .

If we assume that the image prior* π*prior is of the form* π*prior(*x*) ∝ *e*^(*−*)^(*α*)^(Φ)^(()^(*x*)^())^( )for some function Φ, the MAP estimate for* x* with* y* =* Kx *from* y*^(*δ*)^( )(where* y*^(*δ*)^( )is a version of* y* that is corrupted by Poisson noise) is

argmax *x **π*(*y*^(*δ*)^(* *)*|** Kx*)*π*prior(*x*) = argmax *x*
## * *∏
 *p*

(*Kx*(*p*))^(*y*)^(*δ*)^(()^(*p*)^())*e*^(*−*)^(*Kx*)^(()^(*p*)^())

(*y*^(*δ*)(*p*)! *·** e*^(*−*)^(*α*)^(Φ)^(()^(*x*)^()).

Equivalently, we minimize the negative logarithm of the objective which is

argmin *x*

h *−*log(*π*(*y*^(*δ*)^(* *)*|** Kx*))* −*log(*π*prior(*x*)) i

= argmin *x*
## * *∑
 *p*

h *−**y*^(*δ*)(*p*) log(*Kx*(*p*)) +* Kx*(*p*) i +* α*Φ(*x*).

△

The negative log of the noise prior may look strange at first. However, note that the function* fa*(*t*) =* −**a* log(*t*) +* t* has a unique minimum at* t* =* a* (here for *a* = 1.5).

0 1 2 3 1 2 3 4

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 69


---

## Page 70

18.07.2022, VL 13-1

### **13 Discretization by projection**

In our last section we finally treat some discretization methods. The backbone of discretization are projection operators. A bounded linear operator* P* :* X** →**X* on a normed space is a *projection onto a subspace** U* if* Px** ∈**U* for all* x** ∈**X* and* Px* =* x *for* x** ∈**U*. Moreover it holds that* P*^(2)^( )=* P* and* ∥**P**∥≥*1. If* X* is a Hilbert space and* P* is self-adjoint, then* P* is an orthonormal projection and it holds for all* u** ∈**U* that

*∥**Px** −**x**∥≤∥**u** −**x**∥*,

i.e.* Px* is the best approximation from* U* to* x*.

**Definition 13.1.** Let* X*,* Y* be Banach spaces,* K* :* X** →**Y* be bounded and linear and* Xn** ⊂**X*,* Ym** ⊂**Y* be* n*- and* m*-dimensional sub- spaces, respectively. Further, let* Qm* :* Y** →**Ym* be a projection onto* Ym*. The* projection method* for solving* Kx* =* y* is to solve the problem

*QmKxn* =* Qmy*, for* xn** ∈**Xn*.

If we choose bases* {* ˆ*x*1, . . . , ˆ*xn**}* and* {* ˆ*y*1, . . . , ˆ*ym**}* of* Xn* and* Ym*, respectively, we can write

*Qny* = *n*
## * *∑
 *i*=1 *β**i* ˆ*yi*, and *QnK* ˆ*xj* = *n*
## * *∑
 *i*=*i Aij* ˆ*yi*.

The solution* xn* can written as ∑^(*n *)*j*=1* *^(*α*)^(*j*) ^(ˆ)^(*xj*) ^(and thus, the coeﬃcients )can be determined by the linear system

*n*
## * *∑
 *j*=1 *Aij**α**j* =* β**i*.

Plugging the expansion for* xn* into *QmKxn* =* Qmy* gives

*n*
### * *∑
 *j*=1 *α**jQmK* ˆ*xj* = *m*
### * *∑
 *i*=1 *β**i* ˆ*yi*

and using* QnK* ˆ*xj* = *n *∑ *i*=*i Aij* ˆ*yi* gives the

result.

*Example* 13.2 (Galerkin method)*.* Let* X* and* Y* be Hilbert spaces, *Xn*,* Ym* as above and* Qm* an orthogonal projection onto* Ym*. The equation* QmKxn* =* Qmy* is then equivalently expressed as the so-called Galerkin equations

*⟨**Kxn*,* zm**⟩*=* ⟨**y*,* zm**⟩ *for all* zm** ∈**Ym*. (*)

Choosing bases as above gives us the system

*n*
## * *∑
 *j*=1 *α**j *
 *K* ˆ*xj*, ˆ*yi *

| {z } =:*Aij*

=* ⟨**y*, ˆ*yi**⟩ *| {z } =:*β**i*

, *i* = 1, . . . ,* m*. (**)

△

*Example* 13.3 (Collocation method)*.* Here we have any Banach space *X* but fix* Y* =* C*([*a*,* b*]). We choose so-called* collocation points **a* =* t*1* <** · · ·** <** tm* =* b* and consider the subspace* Ym* as the

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 70


---

## Page 71

18.07.2022, VL 13-2

space of functions that are continuous and piecewise linear on the intervals [*ti*,* ti*+1] (also known as the space of linear splines). As operator* Qm* we take the “linear interpolation operator”, i.e.

*Qmy* = *m*
## * *∑
 *i*=1 *y*(*ti*) ˆ*yi*

where the ˆ*yi* are the linear splines which are 1 at* ti* and zero at

ˆ*y*1 ˆ*y*2 ˆ*y*3 ˆ*y*4 ˆ*y*5 ˆ*y*6

*tj* with* j** ̸*=* i*. The projected equation* QmKxn* =* Qmy* is then equivalent to

(*Kxn*)(*ti*) =* y*(*ti*), *i* = 1, . . . ,* n*.

We choose an* n*-dimensional subspace* Xn* of* X* and a basis* {* ˆ*xj** | **j* = 1, . . . ,* n**}* of* Xn*. Then we can express* xn** ∈**Xn* by* xn* = ∑^(*n *)*j*=1* *^(*α*)^(*j*) ^(ˆ)^(*xj*)^(. The collocations equations for)^(* xn*)^(* ∈*)^(*Xn*)^( then become)

*K n*
## * *∑
 *j*=1 *α**j* ˆ*xj*(*ti*) =* y*(*ti*).

The left hand side is ∑^(*n *)*j*=1* *^(*K*)^( ˆ)^(*xj*)^(()^(*ti*)^())^(*α*)^(*j*) ^(and we see that the collocation )equations are equivalent to* A**α* =* β* with

*β**i* =* y*(*ti*), *Aij* =* K* ˆ*xj*(*ti*).

△

*Example* 13.4 (Galerkin and collocation for integral equations)*.* Now consider the special case* K* :* L*^(2)([*a*,* b*])* →**L*^(2)([*c*,* d*]),

*Kx*(*t*) =

*b *Z

*a k*(*t*,* s*)*x*(*s*)d*s* =* y*(*t*), *t** ∈*[*c*,* d*].

The Galerkin method uses the values (cf. (**))

*Aij* =

*d *Z

*c*

*b *Z

*a k*(*t*,* s*) ˆ*xj*(*s*) ˆ*yi*(*t*)d*s* d*t*, *β**i* =

*d *Z

*c y*(*t*) ˆ*yi*(*t*)d*t*

while the collocation method uses

*Aij* =

*b *Z

*a k*(*ti*,* s*) ˆ*xj*(*s*)d*s*, *β**i* =* y*(*ti*).

Note that the entries for the Galerkin method are harder to com- pute (more integrals…). △

In the following we will assume* m* =* n* and that the following conditions are fulfilled:* K* is injective, the union ^(S )*n *^(*Xn*) ^(is dense )in* X* and* QnK**|**Xn* :* Xn** →**Yn* is invertible. Then a solution of* QnKxn* =* Qny* exists and is given by

*xn* =* Rnyn*, with *Rn* := (*QnK**|**Xn*)^(*−*)^(1)*Qn* :* Y** →**Xn*.

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 71


---

## Page 72

18.07.2022, VL 13-3

We say that the projection method is* convergent* if for every* x** ∈**X *it holds that

*RnKx *^(*n*)^(*→*)^(∞ )*−→**x*.

Not every projection method converges, but there is a simple con-

Note that what we are doing here is to consider projection methods as regu- larizations! We can consider* α* = 1/*n *as regularization parameter. dition that ensures convergence:

**Theorem 13.5.*** Under our standing assumptions it holds that** xn* =* Rny **converges to** x** for every** y* =* Kx** exactly if there exists** c** >* 0* such that*

*∥**RnK**∥≤**c*. (+)

*Moreover, if this is fulfilled, then (with the same** c**)*

*∥**xn** −**x**∥**X** ≤*(1 +* c*) min *zn**∈**Xn*^(*∥*)^(*zn*)^(* −*)^(*x*)^(*∥*)^(*X*)^(.)

*Proof.* First assume that the method converges, i.e. that* RnKx *^(*n*)^(*→*)^(∞ )*−→ **x* for every* x*. Then the assertion follows from the uniform bound- edness principle. For us, the other direction is more interesting: Let* ∥**RnK**∥*be bounded. For* zn** ∈**Xn* we have that

*RnKzn* = (*QnK**|**Xn*)^(*−*)^(1)*QnKzn* = (*QnK**|**Xn*)^(*−*)^(1)*QnK**|**Xnzn* =* zn*

and thus,* RnK* is a projection. We conclude that

*xn** −**x* = (*RnK** −*id)*x* = (*RnK** −*id)(*x** −**zn*).

We obtain* ∥**xn** −**x**∥**X** ≤*(*c* + 1)*∥**x** −**zn**∥**X* and taking the mini- mum over all* zn* shows the inequality. The convergence* xn n**→*∞ *−→**x *follows since ^(S )*n *^(*Xn*) ^(is dense in)^(* X*)^(.)

Here is an errorestimate forthe Galerkin method. To express it, we define the* synthesis operator** S*^(*X *)*n* ^(:)^(** R**)^(*n*)^(* →*)^(*X*)^( in)^(* X*)^( by)^(* SX *)*n** *^(*α*)^( =)^( ∑)*j** *^(*α*)*j* ^(ˆ)^(*x*)*j *and similarly for* S*^(*Y *)*n*^(. We define the quantities )If we choose orthonormal bases, then we get* an* =* ∥**S*^(*X*)*n** ∥*= 1 and also* bn* = 1. *an* =* ∥**S*^(*X *)*n** *^(*∥*)^(=)^( max )n *∥**S*^(*X *)*n** *^(*α*)^(*∥*)^(*X *)* ∥**α**∥*2 = 1 o

*bn* = max n *∥**β**∥*2 * ∥**SY n** *^(*β*)^(*∥*)^(*Y*) ^(=)^( 1 )o

**Theorem 13.6.*** Assume that the Galerkin equations* (*)* from Exam- ple 13.2 are uniquely solvable.*

*(a) Let** y*^(*δ*)^(* *)*∈**Y** with** ∥**y** −**y*^(*δ*)*∥**Y** ≤**δ** and** x*^(*δ *)*n** *^(*be the solution of *)D *Kx*^(*δ *)*n*^(,)^(* zn *)E = D *y*^(*δ*),* zn *E *for all** zn** ∈**Yn*.

*Then it holds that*

*∥**x*^(*δ *)*n** *^(*−*)^(*x*)^(*∥*)^(*X*)* *^(*≤∥*)^(*Rn*)^(*∥*)^(*δ*)^( +)^(* ∥*)^(*RnKx*)^(* −*)^(*x*)^(*∥*)^(*X*)^(.)

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 72


---

## Page 73

18.07.2022, VL 13-4

*(b) Let** A** and** β** be given by* (**)* from Example 13.2 and let** ∥**β** −**β*^(*δ*)*∥≤**δ **hold and let** λ**n** be the smallest singular value of** A**. Let** α*^(*δ*)^(* *)*be the solution of** A**α*^(*δ*)^( )=* β*^(*δ*)^(* *)*and define** x*^(*δ *)*n* ^(=)^( ∑)^(*n *)*j*=1* *^(*α*)^(*j*) ^(ˆ)^(*xj*)^(*. Then it holds*)

*∥**x*^(*δ *)*n** *^(*−*)^(*x*)^(*∥*)^(*X*)* *^(*≤*)^(*an*)

*λ**n** *^(*δ*)^( +)^(* ∥*)^(*RnKx*)^(* −*)^(*x*)^(*∥*)^(*X*)

*Proof.* For part (a) we simply use the standard error decomposition

*∥**x*^(*δ *)*n** *^(*−*)^(*x*)^(*∥*)^(*X*)* *^(*≤∥*)^(*x*)^(*δ *)*n** *^(*−*)^(*Rny*)^(*∥*)^(+)^(* ∥*)^(*Rny*)^(* −*)^(*x*)^(*∥≤∥*)^(*Rn*)^(*∥∥*)^(*y*)^(*δ*)^(* −*)^(*y*)^(*∥*)^(*Y*) ^(+)^(* ∥*)^(*RnKx*)^(* −*)^(*x*)^(*∥*)^(*X*)

from which the estimate follows. For part (b) we just need to estimate the data error in the above decomposition. We write* Rnx* = ∑^(*n *)*j*=1* *^(*α*)^(*j*) ^(ˆ)^(*xj*)^(. Since)^(* x*)^(*δ *)*n** *^(*−*)^(*Rny*)^( =)

∑^(*n *)*j*=1^(()^(*αδ *)*j** *^(*−*)^(*α*)^(*j*)^())^( ˆ)^(*xj*)^( =)^(* SX *)*n* ^(()^(*αδ *)*n** *^(*−*)^(*α*)^())^( we get)

*∥**x*^(*δ *)*n** *^(*−*)^(*Rny*)^(*∥*)^(*X*)* *^(*≤*)^(*an*)^(*∥*)^(*αδ *)*n** *^(*−*)^(*α*)^(*∥*)^(2) ^(=)^(* an*)^(*∥*)^(*A*)^(*−*)^(1)^(()^(*βδ*)^(* −*)^(*β*)^())^(*∥*)^(2)* *^(*≤*)^(*an*)

*λ**n** *^(*δ*)

as desired.

*Example* 13.7 (Collocation method for the inverse integration prob- lem)*.* We consider the simple problem* Kx*(*t*) = ^(R)^(* t *)0* *^(*x*)^(()^(*s*)^())^(d)^(*s*)^( with )*K* :* C*([0, 1])* →**C*([0, 1]). We have to choose the collocation points *ti* and the basis ˆ*xj* of* Xn*. Once we have done this, the linear equa- tion* A**α* =* β* is given by* β**i* =* y*(*ti*) and

*Aij* =* K* ˆ*xj*(*ti*) =

*ti *Z

0 ˆ*xj*(*s*)d*s*.

First we choose the basis ˆ*xj*. Let us choose the characteristic func-

tions on the intervals* Ij* = [* *^(*j*)^(*−*)^(1)

*n* ^(,)^(* j*)

*n*^([)^(:)

ˆ*xj*(*t*) =* χ**Ij*(*t*).

,

1 *n*

*n**−*1

*n *1

ˆ*x*1 ˆ*xn*

The functions* K* ˆ*xj* are

*K* ˆ*xj*(*t*) =

  

 

0 : *t** ≤*^(*j*)^(*−*)^(1)

*n t** −*^(*j*)^(*−*)^(1)

*n *: *j**−*1

*n** *^(*≤*)^(*t*)^(* ≤*)^(*j*)

*n *1 *n *: *t** ≥*^(*j*)

*n*

. 1 *n*

1 *n*

*n**−*1

*n *1

*K* ˆ*x*1 *K* ˆ*xn*

Depending on the collocation points we get diﬀerent linear sys- tems:

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 73


---

## Page 74

18.07.2022, VL 13-5

1. We choose* ti* =* *^(*i*)^(*−*)^(1)

*n* ^(,)^(* i*)^( =)^( 1, . . . ,)^(* n*)^( +)^( 1)^(, i.e. we have)^(* m*)^( =)^(* n*)^( +)^( 1)^(. )This gives us

*Aij* =* K* ˆ*xj*(*xi*) =  0 : *i** ≤**j *1 *n *: else. , *A* = ^(1)

*n*



    

0

1 ... ... ... 0 1 *· · · *1



     *∈***R**^(*n*)^(+)^(1)^(*×*)^(*n*).

2. As a variant of the first choice we could only take the left ends, i.e.* ti* = *i**−*1

*n* ^(,)^(* i*)^( =)^( 1 . . . ,)^(* n*)^(, or the right ends)^(* ti*)^( = )*i n*^(, )*i* = 1, . . . ,* n* and get

*A* = ^(1)

*n*



    

0

1 ... ... ... ... 1 *· · · *1 0



     *∈***R**^(*n*)^(*×*)^(*n*), *A* = ^(1)

*n*



 

1 0 ... ... 1 *· · · *1



 *∈***R**^(*n*)^(*×*)^(*n*),

respectively.

3. We choose the middle points of the intervals* ti* = *i**−*^(1)

2 *n* ^(. This )way we get

*Aij* = ^(1)

*n*



    

1 2 1 ... ... ... ... 1 *· · · *1 1 2



     *∈***R**^(*n*)^(*×*)^(*n*)

△

Inverse Problems* |* Version of June 15, 2023* |* SoSe 2022 74


---
