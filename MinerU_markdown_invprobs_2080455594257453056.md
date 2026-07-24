## Inverse Problems

## Dirk Lorenz

## Summer term 2022

## Contents

Introductory remarks 2
1 Introduction and motivation 4
2 Examples and basic notions 10
3 Hilbert spaces 14
4 The singular value decomposition and the pseudo-inverse 18
5 Regularization 23
6 Tikhonov regularization 30
7 Spectral regularization 37
8 Parameter choice and error estimates 42
9 Convergence rates and smoothness spaces 47
10 Convergence rates for spectral regularization 52
11 Iterative regularization 58
12 A Bayesian perspective on regularization 65
13 Discretization by projection 70 

## Introductory remarks

These are the lecture notes for the lecture “Inverse Problems” I held in the summer term 2023 at TU Braunschweig. The lecture is aimed at students from mathematics, financial mathematics, computational science and engineering as well as data science. Solid knowledge in analysis and linear algebra is needed. Helpful would be a background in functional analysis, especially the notions of Hilbert space and linear operators, but we will also provide a little background on these topics in the course. 

Inverse problems are problems where one want to find some cause which can only measured indirectly, i.e. one can only observe the efects, but not the cause itself. Hence, it is quite applied as a math topic, but still one can do serious mathematics and the charm of the field lies in the tight connection of theoretical results and real world applications. 

The books on inverse problems one can find also vary from quite theoretical to very applied. Here is a short commented list of books: 

1. The books Engl u. a. [1996]; Rieder [2013] are on the theoretical side of inverse problems. While they also present applications, they focus on the underlying theory. The latter one (Rieder [2013]) is in German and written in the style of a textbook. The more recent lecture notes Clason [2020] are also quite theoretical. 

2. The older Groetsch [1993] is also written as a text book, focuses on theory, but targets readers with less background in mathematics. 

3. The book Moura Neto und da Silva Neto [2012] uses less mathematics and also targets students with less background in math. It also does not focus that much on theory. 

Braunschweig, June 15, 2023 

Dirk Lorenz 

d.lorenz@tu-braunschweig.de 



[Clason 2020] Clason, Christian: <sub>Regularization</sub> <sub>of</sub> <sub>inverse</sub> <sub>prob-</sub> <sub>lems</sub>. arXiv preprint arXiv:2001.00617. 2020 





[Engl u. a. 1996] Engl, Heinz W. ; Hanke, Martin ; Neubauer, <sup>Andreas:</sup> Regularization of inverse problems<sup>.</sup> <sup>Bd.</sup> <sup>375.</sup> <sup>Springer</sup> Science & Business Media, 1996 





[Groetsch 1993] Groetsch, Charles: <sub>Inverse</sub> <sub>problems</sub> <sub>in</sub> <sub>the</sub> mathematical sciences. Bd. 52. Springer, 1993 





[Moura Neto und da Silva Neto 2012] Moura Neto, Francisco D. ; S ilva Neto, Antônio José da: <sub>An</sub> <sub>introduction</sub> <sub>to</sub> inverse problems with applications<sup>.</sup> <sup>Springer</sup> <sup>Science</sup> <sup>&</sup> <sup>Business</sup> Media, 2012 





<sup>[Rieder</sup> <sup>2013] R</sup> <sup>ieder,</sup> <sup>Andreas:</sup> Keine Probleme mit inversen Problemen: eine Einführung in ihre stabile Lösung<sup>.</sup> <sup>Springer-Verlag,</sup> <sup>2013</sup> 



## 1 Introduction and motivation

The central topic of this lecture are inverse and ill-posed problems. Both the terms “inverse” and “ill-posed” are not clearly defined up to now (and will be hard to pin down exactly). Instead of defining them right away, we start with a motivating example. 

<sub>Example</sub> 1.1 (Diferentiation)<sub>.</sub> The problem of finding the derivative $g ^ { \prime }$ of a given function <sub>g</sub> is quite straightforward as long as symbolic computations are considered. However, finding the slope of a function that is not given as analytic expression, but can only be evaluated through a black box, is more involved. We will show, that it is even inverse and ill-posed in some sense. 

Mathematically, we would like to invert the operator $A$ which takes a function $f$ (for simplicity defined on $[ 0 , 1 ] )$ to its integral, i.e. $A f = g$ with 

$$
A f (x) := g (x) := \int_ {0} ^ {x} f (t) d t.
$$

Hence, the task is: Given some $g ,$ find $f$ such that $A f = g$ 

We would like to measure errors in the data $g$ and also in the reconstruction $f$ and hence, we introduce norms for these quantities. We use the following norms: 

$$
\begin{array}{l} \| f \| _ {C} := \| f \| _ {\infty} := \max \left\{\left| f (x) \right| \mid x \in I \right\} \\ \| f \| _ {C ^ {1}} := \| f \| _ {\infty} + \| f ^ {\prime} \| _ {\infty} \\ \| f \| _ {C ^ {k}} := \sum_ {l = 0} ^ {k} \| f ^ {(l)} \| _ {\infty}. \end{array}
$$

In fact, these norms turn the appropriate vector spaces into normed spaces: For an interval <sub>I</sub> let 

$$
\begin{array}{l} C (I) := \left\{f: I \to \mathbb {R} \mid f \text { continuous } \right\}, \\ C ^ {1} (I) := \left\{f: I \to \mathbb {R} \mid f \text { continuously   differentiable } \right\}, \\ C ^ {k} (I) := \left\{f: I \to \mathbb {R} \mid f \text { k - times   continuously   differentiable } \right\}. \end{array}
$$

We can model the operator <sub>A</sub> as a map between various spaces, e.g. we can write $A : C ( \hat { [ 0 , 1 ] } )  C ( [ 0 , 1 ] )$ . Diferentiation is a left inverse of <sub>A</sub>: We have that $\begin{array} { r } { D A f ( x ) = { \cal D } ( \int _ { 0 } ^ { x } f ( t ) \mathrm { d } t ) = f ( x ) } \end{array}$ . In this sense, the problem of calculating the derivative is the inverse problem to calculating the integral. 

Now let us argue that the inverse problem of diferentiation is ill-posed while the <sub>direct</sub> problem of integration is well posed: The map $A : C ( [ 0 , 1 ] ) \to C ( [ 0 , 1 ] )$ is linear and bounded. Linearity follows from the known rules for integrals and boundedness is 

We may omit the argument I and just write $C$ and $C ^ { k } \mathsf { i f } I$ is clear from the context or does not play a role. We also denote $C ^ { 0 } = C$ which is consistent with the notation for $C ^ { k }$ . These norms induce a notion of convergence: We say that $f _ { n } \to f \mathsf { i n } C ^ { k } \mathsf { i f } \parallel f _ { n } ^ { - } - f \parallel _ { C ^ { k } } \to \dot { 0 } .$ Convergence in C is exactly uniform convergence and convergence in $C ^ { k }$ means that the functions as well as their first k derivatives converge uniformly. It’s not a right inverse, since $A D f ( x ) =$ $\begin{array} { r } { \int _ { 0 } ^ { x } f ^ { \prime } ( x ) \overset { \vartriangle } { \mathrm { d } { x } } = f ( x ) - f ( 0 ) } \end{array}$ 

seen as follows: For a function $f \in C$ we have 

$$
\begin{array}{l} \| A f \| _ {C} = \max \left\{\left| \int_ {0} ^ {x} f (t) \mathrm{d} t \right| | 0 \leq x \leq 1 \right\} \\ \leq \max \left\{\int_ {0} ^ {x} | f (t) | \mathrm{d} t | 0 \leq x \leq 1 \right\} \leq \max | f (x) | = \| f \| _ {C}. \end{array}
$$

This shows that <sub>A</sub> is bounded and even that the operator norm of <sub>A</sub> fulfills $\| A \| \leq 1$ 

How about continuity of the inverse operation? Consider <sub>g</sub> with $g ( 0 ) = 0$ and $g ^ { \prime } = f ( \mathbf { i . e } , A f = g )$ and let us perturb <sub>g</sub> slightly to 

$$
g ^ {\delta} (x) = g (x) + \delta \sin (n x)
$$

for some $\delta > 0$ and $n \in \mathbb { N }$ . Then we have 

$$
(g ^ {\delta}) ^ {\prime} (x) = g ^ {\prime} (x) + \delta n \cos (n x) =: f ^ {\delta} (x).
$$

Hence, we have $A f = g$ and $A f ^ { \delta } = g ^ { \delta }$ . Moreover, we easily see that 

$$
\begin{array}{l} \| g - g ^ {\delta} \| _ {C} = \delta \\ \| f - f ^ {\delta} \| _ {C} = \max \left\{\delta n \cos (n x) \mid 0 \leq x \leq 1 \right\} = n \delta . \end{array}
$$

If we couple $\delta = 1 / { \sqrt { n } }$ we get 

$$
\| g - g ^ {\delta} \| = \delta = \frac {1}{\sqrt {n}} \stackrel {{n \rightarrow \infty}} {{\longrightarrow}} 0, \quad \text { but } \quad \| f - f ^ {\delta} \| _ {C} = n \delta = \sqrt {n} \stackrel {{n \rightarrow \infty}} {{\longrightarrow}} \infty .
$$

This shows that small perturbations in <sub>g</sub> may lead to large perturbations in the derivative $f = g ^ { \prime }$ , and hence, taking the derivative is unstable, and thus ill-posed. 

At first, this may seem like a hopeless situation when is comes to numerical diferentiation. We always have some round-of error, so does that mean that numerical diferentiation can not work? 

Assume that $g : [ 0 , 1 ] \to \mathbb { R }$ is a diferentiable function, but we don’t have a formula for <sub>g</sub> but for every <sub>x</sub> we get the value $g ( x )$ How can we find $g ^ { \prime } ( x ) \colon$ The simplest idea may be to use a central diference quotient 

$$
\frac {g (x + h) - g (x - h)}{2 h} =: D _ {h} g (x).
$$

This operator $D _ { h }$ is again a linear operator, and, in some sense, an approximation of the derivative <sub>D</sub>. Using Taylor expansion, we get for $g \in C ^ { 2 }$ that 

$$
g (x \pm h) = g (x) \pm h g ^ {\prime} (x) + \frac {g ^ {\prime \prime} (\xi_ {\pm})}{2} h ^ {2}
$$

with some $\xi _ { \pm }$ between <sub>x</sub> and $x + h$ and $x - h ,$ , respectively. This gives us an estimate 

$$
\left. \right.\left\| g ^ {\prime} - D _ {h} g \right\| _ {\infty} = \max \left\{\left| g ^ {\prime} (x) - \left(g ^ {\prime} (x) + \frac {h}{2} \frac {g ^ {\prime \prime} (\xi_ {+}) - g ^ {\prime \prime} (\xi_ {i})}{2}\right)\right|\right\} \leq \frac {h}{2} \| g ^ {\prime \prime} \| _ {\infty}
$$

This is an estimate for the error we make when we use an approximation of the derivative, and hence, we call this error the approximation error<sup>.</sup> 

Now we assume that we have an error in our available data, i.e. we do not get the exact values $g ( x )$ , but slightly perturbed data 

$$
g ^ {\delta} (x) = g (x) + w (x) \quad \text { with } \quad \| w \| _ {\infty} \leq \delta .
$$

We are interested in the <sub>total</sub> <sub>error</sub>, i.e. for $f = g ^ { \prime }$ we would like to compute or estimate 

$$
\| f - D _ {h} g ^ {\delta} \| _ {\infty},
$$

which is the error between the unknown exact derivative $f$ and the quantity $D _ { h } g ^ { \delta }$ we can actually compute. This error has a natural decomposition as 

$$
\| f - D _ {h} g ^ {\delta} \| _ {\infty} \leq \| f - D _ {h} g \| _ {\infty} + \| D _ {h} g - D _ {h} g ^ {\delta} \| _ {\infty}
$$

where we inserted the term $\pm D _ { h } g$ and used the triangle inequality. The first term on the right is the approximation error which we just estimated already. The second term is called <sub>data</sub> <sub>error</sub> and is also simple to estimate: 

$$
\| D _ {h} g - D _ {h} g ^ {\delta} \| _ {\infty} = \| D _ {h} w \| _ {\infty} = \max \left\{| \frac {w (x + h) - w (x - h)}{2 h} | \right\} \leq \frac {\delta}{h}.
$$

So we get for the total error 

$$
\| f - D _ {h} g ^ {\delta} \| _ {\infty} \leq h \frac {\| g ^ {\prime \prime} \| _ {\infty}}{2} + \frac {\delta}{h}.
$$

Overall, we see a very typical behavior: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/ebc1d5806dc9541a70caef194de070d34ee26a0c8daa953576d7ed1a9e897432.jpg)


<sup>For</sup> <sup>a</sup> <sup>fixed</sup> noise level $\delta ,$ there is a tradeof between large and small parameters <sub>h</sub>: For small <sub>h</sub> the data error gets big, while for large <sub>h</sub> the approximation error gets big. Somewhere in the middle there is an optimum and a little calculus shows that the parameter <sub>h</sub> which minimizes our upper bound for the total error is 

$$
h ^ {*} = \sqrt {\frac {2 \delta}{\| g ^ {\prime \prime} \| _ {\infty}}}.
$$

With this value we get 

$$
\| f - D _ {h ^ {*}} g ^ {\delta} \| _ {\infty} \leq \sqrt {2 \| g ^ {\prime \prime} \| _ {\infty} \delta}.
$$

We see that even when the operation of diferentiation is ill-posed, we can still get a stable approximation of the derivative from noisy data. However, the error is not as small as one could have hoped: We obtain an error in the order of $\sqrt { \delta }$ for noise of size $\delta . \quad \Delta$ 

Here are a few important takeaways from the above example: 

• The total error in the solution of an inverse problem decomposes into an approximation error and a data error: Good approximation leads to a amplification of the error in the data, and keeping the data error small needs a large approximation error. 

• For a fixed noise level <sub>δ</sub> there is a tradeof between approximation error and data error. 

• A helpful estimate of the approximation error needs a smoothness assumption on the unknow data (in our case we needed that $g ^ { \prime \prime } = \hat { f } ^ { \prime }$ exists and is bounded). 

• The total error is, even in the best case, not of the order of the error in the data, but worse. 

<sub>Remark</sub> 1.2<sub>.</sub> Our results is actually useful in practice: If you want to evaluate derivatives of functions numerically by a finite diference approximation, one usually uses $h = { \sqrt { \mathrm { e p s } } }$ where <sub>eps</sub> is the machine precision. For double precision numbers <sub>eps</sub> $\approx 1 0 ^ { - 1 6 }$ so $h = 1 0 ^ { - 8 }$ is recommended. 

Here is an example (written in Python): 

It is quite instructive, to see that higher smoothness can lead to a better total error: Assume that $g ^ { \prime \prime \prime }$ exists, derive a better estimate for the approximation error (by using more terms of the Taylor expansion, calculate the optimal h and deduce that the total error can be of order $\delta ^ { 2 / 3 }$ in this case. 

```python
import libraries
import numpy as np
import matplotlib.pyplot as plt

# define functions
def f(x):
    return np.sin(np.log(x)**4)**3

def fprime(x):
    return 3*np.sin(np.log(x)**4)**2*np.cos(np.log(x)**4)*4*np.log(x)**3/x

def Df(x,h):
    return (f(x+h)-f(x-h))/2/h

# some x values
x = np.linspace(0.1,4,200,dtype='float64')
# plot function
plt.plot(x,f(x))
plt.show() 
```

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/5ffd4739e015efb41c74591a7514a1abe9c290d979a9b288314ae62871755c31.jpg)


```txt
# Choose some stepsize
h = 1e-5
# plot derivative and approximation by finite differences
fig, axs = plt.subplots(3)
axs[0].plot(x, fprime(x))
axs[0].set_title('true derivative')
axs[1].plot(x, Df(x,h))
axs[1].set_title('derivative by finite differences')
axs[2].plot(x, fprime(x)-Df(x,h))
axs[2].set_title('difference of the two')
plt.show() 
```

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/36986943b9563d6eaca8c9ed1e7f1f4f8588df1ed5ca660752bf04cbeb49230a.jpg)


# square root of machine precision is close to optimal h: print(’sqrt(eps) = ’, np.sqrt(np.finfo(x.dtype ). eps )) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/91d6895d5026a67a009de33faa2af24eb8a5ca3a54d24126cd68774045168c96.jpg)


## 2 Examples and basic notions

The notion of “inverse problems” is vague and hard to pin down. What is a “problem” anyway? Well, a problem always has “data” and “solution”, i.e. something that is given, and something that is wanted. Solving the problem means, taking the data, do some computations and arrive at a solution. For every problem, there is an “inverse problem”, namely: Having some solution, what is the corresponding data that gave rise to this solution? 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/cebe4280a0ef211e701115c9ebda18e1ad17c5497589a76fab567d50d1b23695.jpg)


In a more physical context one could frame inverse problems as follows: 

An inverse problems asks for some <sub>cause</sub> that is behind <sup>a</sup> <sup>given</sup> observation<sup>.</sup> 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/129f380cbee84f3f89b6f24f194eb8054c9f3ec6b54c03282c18bc7c77076711.jpg)


<sub>Example</sub> 2.1 (Parameter identification in PDEs)<sub>.</sub> Assume that you can observe the heat distribution <sub>u</sub> of some matter in a domain <sub>Ω</sub> at a given time $t = T > 0$ . What was the heat distribution at time $t = 0 ?$ This is an inverse problem. To formulate it mathematically, we use the heat equation. The distribution of heat follows the partial diferential equation 

$$
\begin{array}{c} u _ {t} (t, x) = \Delta u (t, x) \quad \text { in } [ 0, T ] \times \Omega \\ u (0, x) = u _ {0} (x) \quad \text { in } \Omega \\ \partial_ {n} u (t, x) = 0 \quad \text { in } [ 0, T ] \times \partial \Omega . \end{array}
$$

The forward problem would be: Given the initial data $u _ { 0 } ,$ calculate $u ( T , x )$ . The corresponding inverse problem is: Given a measurement of <sub>u</sub> $( T , x )$ , find initial data $u _ { 0 }$ that explains the measurement. 

In the context of partial diferential equations one can formulate numerous inverse problems. Consider the following problem: 

$$
\begin{array}{c} u _ {t} - L u = f \quad \text {in} [ 0, T ] \times \Omega \\ u (0, x) = u _ {0} (x) \quad \text {in} \Omega \\ \partial_ {n} u = g \quad \text {on} [ 0, T ] \times \partial \Omega \end{array}
$$

with some diferential operator $L ,$ initial data $u _ { 0 } ,$ , source term $f ,$ and boundary data $g .$ . The forward problem would be to compute <sub>u</sub> from knowledge of $u _ { 0 } , f$ and $g ,$ , but there are various inverse problems and here are just two: 

• Given $u ( T , \cdot ) , f$ and $^ { g , }$ find $u _ { 0 }$ . 

• Given $u ,$ and $u _ { 0 } ,$ , find $f$ and $g .$ 

You can find more inverse problems easily. 

Example $^ { 2 . 2 }$ (Computerized tomography)<sub>.</sub> The basic concept of CT is to measure the intensity of X-ray beams from a source with known intensity after passing through the body at a fixed plane. It is assumed that these beams travel on a straight line and their intensity is attenuated proportially to some material constant one is interested in reconstructing, which is usually the density. Denoting this material constant by <sub>u</sub> and $x = x ( L , t )$ the point in which the $_ { \mathrm { X - r a y } }$ beam associated with the line <sub>L</sub> passes at time $t ,$ this can be modeled by the ordinary diferential equation 

$$
I _ {L} ^ {\prime} (t) = - u (x (L, t)) I _ {L} (t),
$$

so if $I _ { L } ( T )$ is measured for some time $T > 0$ where the beam passed the object of interest, 

$$
- \log \left(\frac {I _ {L} (T)}{I _ {L} (0)}\right) = \int_ {0} ^ {T} u (x (L, t)) d t.
$$

The left-hand side is known while the right-hand side constitutes the integral of <sub>u</sub> associated with the line <sub>L</sub> up to some factor. The principle of computed tomography is obtain these integrals by emitting and measuring X-ray beams along all possible lines. This is typically done by placing an X-ray point source on one side of the object, installing a detector array on the other and side rotating source and detector simultaneously, giving the intensities for all lines passing through a region of interest. 

Mathematically, the reconstruction problem in CT is now to compute <sub>u</sub> given all of its line integrals. Placing the origin in the center of the object of interest of radius $R > 0$ , a line <sub>L</sub> passing through the domain can uniquely be associated an angle $\theta \ \in$ $\left[ - \pi / 2 , \pi / 2 \right]$ and ofset $s \in \mathbb { R }$ at which the line crosses the axis spanned by $x _ { 0 } ( \theta ) = ( \cos ( \theta ) , \sin ( \theta ) )$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/f91ddbad8203f72c7eb1ec90365614724b2a3ae4f144a17abbe32aa892e7802e.jpg)


We denote the corresponding line by $L \ = \ L ( s , \theta )$ . Given $u ^ { 0 }$ : $] { - R , R [ \times [ - \pi / 2 , \pi / \overset { \textstyle } { 2 } ] \overset { } {  } \mathbb { R } }$ , the CT reconstruction problem is to find $u ^ { \dagger } : \mathbb { R } ^ { 2 } $ <sub>R</sub> such that 

$$
(R u) (s, \theta) := \int_ {L (s, \theta)} u ^ {\dagger} \mathrm{d} x = u ^ {0} (s, \theta) \quad \text {   for   all   } \quad (s, \theta)
$$

where the mapping <sub>R</sub> is called the <sub>Radon</sub> <sub>transform</sub>. The inverse problem related to the Radon transform is the main problem in tomographic reconstruction. $\bigtriangleup$ 

Here is the classical definition of “ill posedness” which is due to Hadamard and dates back to the 1920s. 

<sub>Definition</sub> <sub>2.3</sub> (Well- and ill-posed problems)<sub>.</sub> Let <sub>X</sub> and <sub>Y</sub> be topological spaces and $A : X \to Y$ . We say that the problem “solve $A x = y '$ <sup>is</sup> well posed <sup>if</sup> 

(a) The equation $A x = y$ has a solution for every $y \in Y ,$ , and 

(b) this solution is unique, and 

(c) the solution depends continuously on the data. 

If one of these conditions is not fulfilled, we call the problem ill-posed<sup>.</sup> 

In short: the problem is well posed if the inverse $A ^ { \dot { - } 1 } : Y \to X$ exists and is continuous. 

In this lecture we will treat <sub>linear</sub> inverse problems. We will always assume <sub>Hilbert</sub> <sub>spaces</sub> <sub>X</sub> and <sub>Y</sub> and a linear, bounded operator $A : X \to Y$ .The space <sub>X</sub> is the <sub>solution</sub> <sub>space</sub> and <sub>Y</sub> is the <sub>data</sub> <sub>space</sub>. The forward problem is “given <sub>x</sub>, evaluate $A x ^ { n }$ . Since inverse problems always assume measurement data with error, the inverse problem is: 

We will denote the set of linear and bounded operators from X to Y by $L ( X , Y )$ 

Given measured data $g ^ { \delta } \in Y$ which fulfills 

$$
\| A f ^ {\dagger} - g ^ {\delta} \| \leq \delta
$$

<sup>for</sup> <sup>some</sup> <sup>known</sup> noise level δ <sup>and</sup> <sup>an</sup> unknown true solution $f ^ { \dagger }$ , find a good approximation to $f ^ { \dagger }$ . 

For linear operators one can characterize the ill-posedness of the problem “Solve $A x = y '$ quite explicitly: 

(a) $A x = y$ has a solution for every <sub>y</sub> exactly if <sub>A</sub> is surjective (also called onto), i.e. $\operatorname { i f } \operatorname { r g } ( A ) = Y .$ 

(b) Solutions of $A x = y$ are unique exactly if <sub>A</sub> is injective (also called one-to-one), i.e. if <sub>ker(A)</sub> <sub>=</sub> <sub>{0}</sub> 

(c) Solutions of $\boldsymbol { \cdot } \boldsymbol { A } \boldsymbol { x } = \boldsymbol { y }$ depend continuously on <sub>y</sub> exactly if $A ^ { - 1 }$ is bounded. 

A more quantitative way to describe ill-posedness of a problem is the notion of <sub>condition</sub> or conditioning of a problem. We say that a problem is 

<sub>well</sub> <sub>condtioned</sub> if small changes in the data lead to small changes in the solution 

ill conditioned (or badly conditioned) <sup>if</sup> <sup>small</sup> <sup>changes</sup> <sup>in</sup> <sup>the</sup> <sup>data</sup> lead to large errors in the solution. 

<sub>Example</sub> 2.4 (Function evaluation)<sub>.</sub> Here the problem is simply “given <sub>x</sub> evaluate $y = f ( x )$ . We consider a perturbation $x + \Delta x \mathrm { o f } x$ The solution changes to $y + \Delta y = f ( x + \Delta x )$ and by linearization we get 

$$
| \Delta y | \approx | f ^ {\prime} (x) | | \Delta x |.
$$

So we say that the problem is ill-conditioned (with respect to absolute errors) $\mathrm { i f } | \hat { f ^ { \prime } } ( x ) |$ is large. 

If we consider relative errors, we get 

$$
\frac {| \Delta y |}{| y |} \approx \frac {| f ^ {\prime} (x) | | \Delta x |}{| f (x) |} = \frac {| f ^ {\prime} (x) | | x |}{| f (x) |} \frac {| \Delta x |}{| x |}
$$

so we say that the problem is ill-conditioned with respect to relative error $\mathrm { i f } \bar { | } f ^ { \prime } ( x ) | | x \bar { | } / | f ( x ) |$ is large. $\bigtriangleup$ 

<sub>Example</sub> 2.5 (Solving linear equations)<sub>.</sub> Consider an invertible square matrix <sub>A</sub> and the problem: given $b ,$ find the solution of $A x = b$ Changing the data to $b + \Delta b ,$ , the new solution fulfills 

$$
A (x + \Delta x) = b + \Delta b.
$$

We see that the change in the solution is $\Delta x = A ^ { - 1 } \Delta b$ . Using the operator norm we get that $\| \Delta x \| \leq \| A ^ { - 1 } \| \| \Delta b \|$ and we see that the problem is ill-conditioned (with respect to absolute errors) if $\| A ^ { \hat { - 1 } } \|$ is large. If we consider relative errors, we get $( { \mathrm { u s i n g ~ } } \| b \| =$ $\| A x \| \leq \| A \| \| x \|$ 

$$
\frac {\| \Delta x \|}{\| x \|} = \frac {\| A ^ {- 1} \Delta b \|}{\| x \|} \leq \frac {\| A ^ {- 1} \| \| \Delta b \|}{\| x \|} \frac {\| b \|}{\| b \|} \leq \| A ^ {- 1} \| \| A \| \frac {\| \Delta b \|}{\| b \|}.\tag{△}
$$

<sub>Definition</sub> <sub>2.6.</sub> The condition number of a square matrix <sub>A</sub> is 

$$
\operatorname{cond} (A) = \left\{ \begin{array}{l l} \| A \| \| A ^ {- 1} \| & : \text {   if   } A \text {   is   invertible   } \\ \infty & : \text {   else.   } \end{array} \right.
$$

Strictly speaking, the problem “Solve $A x = b ^ { n }$ is only ill-posed if <sub>A</sub> is not invertible. Practically, a large condition number of <sub>A</sub> will still lead to a large increase of the error, so we consider the problem still ill conditioned or badly conditioned if the condition number is large. 

To understand all these things better, we introduce the notion of Hilbert spaces, linear bounded operators between these spaces and the singular value decomposition of these operators. 

## 3 Hilbert spaces

Now we introduce abstract notions of vector spaces that we will use throughout the lecture. The main notions are <sub>Hilbert</sub> <sub>spaces</sub>. In a nutshell, they are spaces which behave the sae as the euclidean space $\mathbb { R } ^ { n }$ when it comes to geometric and analytical structures like length, and orthogonality. 

First we define inner products: 

Definition 3.1. <sup>A</sup> <sup>real</sup> inner product space X <sup>is</sup> <sup>a</sup> <sup>real</sup> <sup>vector</sup> <sup>space</sup> that is equipped with an <sub>inner</sub> <sub>product</sub> $\langle \cdot , \cdot \rangle : X \times X \to \mathbb { R }$ which fulfills 

$$
\begin{array}{c} \langle x, y \rangle = \langle y, x \rangle \\ \langle \alpha x + y, z \rangle = \alpha \langle x, z \rangle + \langle y, z \rangle \\ \langle x, x \rangle > 0 \quad \text { if } x \neq 0. \end{array}
$$

An inner product always induces a norm $\| x \| ~ = ~ { \sqrt { \langle x , x \rangle } }$ (which can be shown using the Cauchy-Schwarz inequality $\langle x , y \rangle \leq$ $| | x | | | | y | | )$ .Since any norm induces a notion of convergence by saying that $x _ { n } \ { \stackrel { n \to \infty } { \longrightarrow } } \ x \ { \mathrm { i f } } \| x _ { n } - x \| \ \to 0$ we can also talk about <sub>complete-</sub> <sub>ness</sub> and this is the property that turns inner product spaces into Hilbert spaces. 

On a Hilbert space X we will denote the norm by $\| { \dot { x } } \| _ { X } ,$ , but sometimes the subscript may be dropped, when the norm is clear from the context. 

<sub>Definition</sub> <sub>3.2.</sub> A real <sub>Hilbert</sub> <sub>space</sub> is a complete real inner product space, i.e. a real inner product space with the property that every Cauchy sequence in the space converges, i.e. for every sequence $x _ { n }$ in <sub>X</sub> which fulfills 

$$
\forall \epsilon > 0 \exists N \forall m, n \geq N: \| x _ {n} - x _ {m} \| \leq \epsilon
$$

there exists some limit $x ^ { * }$ , i.e. $x _ { n } \stackrel { n  \infty } { \longrightarrow } x ^ { \ast } ( \mathrm { i . e }$ . we have $\Vert x _ { n } -$ $x ^ { * } \parallel \xrightarrow [ ] { n  \infty } 0 )$ 

A little bit sloppy one could say that a Hilbert space is an inner product space that “you can’t leave with ‘convergent sequences’” <sub>Example</sub> 3.3<sub>.</sub> 1. The space $\mathbb { R } ^ { n }$ with standard inner product 

$$
\langle x, y \rangle := x \cdot y := x ^ {T} y = \sum_ {i = 1} ^ {n} x _ {i} y _ {i}
$$

is a Hilbert space of dimension <sub>n</sub>. The corresponding norm $\| x \| = \left( \sum _ { i = 1 } ^ { n } x _ { i } ^ { 2 } \right) ^ { 1 / 2 }$ is the well known euclidean norm. The standard inner product on $\mathbb { R } ^ { d }$ is also called <sub>dot</sub> <sub>product</sub> and we will also use the notation $x \cdot y$ for it. The euclidean norm of a vector <sub>x</sub> will also be denoten by $| x |$ . 

2. An example of an infinite dimensional Hilbert space is 

$$
\ell^ {2} := \left\{\left(x _ {n}\right) _ {n = 1, 2, \dots} \middle | \sum_ {i = 1} ^ {\infty} x _ {n} ^ {2} <   \infty \right\}
$$

of square summable sequences. When equipped with the inner product $\langle x , y \rangle : = \Sigma _ { i = 1 } ^ { \infty }$ <sub>x y</sub> it is a Hilbert space. The corresponding norm is denoted by 

$$
\| x \| _ {2} := \left(\sum_ {i = 1} ^ {\infty} x _ {i} ^ {2}\right) ^ {1 / 2}.
$$

3. A diferent example of an infinite dimensional Hilbert space <sup>is</sup> <sup>the</sup> Lebesgue space $L ^ { 2 }$ of square integrable functions. For some domain $\bar { \Omega } \subset \mathbb { R } ^ { d }$ (i.e. a non-empty, connected and open subset of $\mathbb { R } ^ { d } )$ we can define 

$$
L ^ {2} (\Omega) := \left\{f: \Omega \rightarrow \mathbb {R} \left| \int_ {\Omega} f (x) ^ {2} \mathrm{d} x <   \infty \right.\right\}
$$

and equipped with the inner product 

$$
\langle f, g \rangle_ {L ^ {2}} := \int_ {\Omega} f (x) g (x) d x
$$

this is a Hilbert space as well. The corresponding norm is denoted by 

$$
\left\| f \right\| _ {L ^ {2}} := \left(\int_ {\Omega} f (x) ^ {2} \mathrm{d} x\right) ^ {1 / 2}.
$$

The are Sobolev spaces of higher order, i.e. for every $k \in \mathbb { N }$ there is a Sobolev space $H ^ { k } ( { \bar { \Omega } } )$ which incorporates partial derivatives up to <sub>k</sub>-th order, but we will not define them here. 

4. Slightly more complicated are the <sub>Sobolev</sub> <sub>spaces</sub> which generalize the Lebesgue space $L ^ { 2 }$ by also incorporating derivatives. The space $H ^ { 1 } ( { \bar { \Omega } } )$ is 

$$
H ^ {1} (\Omega) := \left\{f: \Omega \rightarrow \mathbb {R} \left| \int_ {\Omega} f (x) ^ {2} \mathrm{d} x + \int_ {\Omega} | \nabla f (x) | ^ {2} \mathrm{d} x <   \infty \right.\right\}
$$

You may have noted that there is a problem here: This is not a norm, since there are functions $f \neq 0$ with $\| f \| _ { L ^ { 2 } ( \Omega ) } =$ 0. This problem can be solved: These functions have the property of being zeros “almost everywhere” and one can “quotient out” all these functions from $L ^ { 2 } ( \Omega )$ . This means that one identifies two functions f and g $\mathsf { i f } \int ( f - g ) ^ { 2 } = 0$ The full theory behind this can be found in books on real anaylsis or measure theory. 

and it is equipped with the inner product 

$$
\langle f, g \rangle_ {H ^ {1}} := \int_ {\Omega} f (x) g (x) \mathrm{d} x + \int_ {\Omega} \nabla f (x) \cdot \nabla g (x) \mathrm{d} x
$$

where the second integral is over the dot product of the gradients. The corresponding $H ^ { 1 }$ -norm is 

$$
\| f \| _ {H ^ {1}} := \left(\int_ {\Omega} f (x) ^ {2} \mathrm{d} x + \int_ {\Omega} | \nabla f (x) | ^ {2} \mathrm{d} x\right) ^ {1 / 2}
$$

△ 

As important as the spaces are linear operators between these spaces. We will call a linear map <sub>A</sub> from one Hilbert space <sub>X</sub> to another <sub>Y</sub> an <sub>operator</sub> and say that an operator <sub>A</sub> is <sub>bounded</sub> if there exists a constant <sub>C</sub> such that $\| A x \| _ { Y } ^ { - } \leq C \| x \| _ { X }$ for all <sub>x</sub>.The infimum over all such constants is the <sub>operator</sub> <sub>norm</sub> of $A _ { i }$ , denoted by $\| A \|$ . Other ways to define the operator norm are 

Bounded operators are exactly the continuous operators. 

$$
\| A \| = \sup _ {\| x \| _ {X} = 1} \| A x \| _ {Y} = \sup _ {x \neq 0} \frac {\| A x \| _ {Y}}{\| x \| _ {X}}.
$$

The set of all bounded linear operators from a Hilbert space <sub>X</sub> to another Hilbert space <sub>Y</sub> is denoted by $L ( X , Y )$ . For every $A \in$ $L ( X , Y )$ we have the so-called <sub>adjoint</sub> operator $A ^ { * }$ defined by 

$$
\forall x \in X, y \in Y: \langle x, A ^ {*} y \rangle = \langle A x, y \rangle .
$$

Note that the adjoint maps $A : Y  X$ and it is also bounded with $\| A ^ { * } \| = \| A \|$ . In $\mathbb { R } ^ { n }$ , operators are just matrices and the adjoint with respect to the standard inner product is just the transpose of the matrix. 

<sub>Example</sub> 3.4<sub>.</sub> 1. Linear operators $A \in L ( \mathbb { R } ^ { n } , \mathbb { R } ^ { m } )$ , i.e. from one euclidean space to another can be identified with matrices $A \in \mathbb { R } ^ { m \times n }$ , i.e. the application of the operator <sub>A</sub> to a vector <sub>x</sub> is given by matrix-vector multiplication <sub>Ax</sub>. In this case these is a simple expression for the operator norm of a matrix: It holds that 

$$
\left\| A \right\| = \sqrt {\lambda_ {\max} (A ^ {T} A)},
$$

i.e. the operator norm is the square root of the largest eigenvalue of the matrix $A ^ { T } A$ . Note that $A ^ { T } A$ is symmetric and positive definite and thus, only has real and eigenvalues greater or equal zero. The adjoint of matrix is given by the transposed matrix as it holds that 

$$
(A x) \cdot y = (A x) ^ {T} \cdot y = x ^ {T} A ^ {T} y = x ^ {T} (A ^ {T} y) = x \cdot (A ^ {T} y).
$$

2. A class of linear operators between $L ^ { 2 }$ spaces $A \in L \big ( L ^ { 2 } ( \Omega _ { 1 } ) , L ^ { 2 } ( \Omega _ { 2 } ) \big )$ is given by so called <sub>integral</sub> <sub>operators</sub>. These are given by 

$$
A f (y) = \int_ {\Omega_ {1}} k (x, y) f (x) \mathrm{d} x
$$

for some function $k : \Omega _ { 1 } \times \Omega _ { 2 } \to \mathbb { R }$ . For these operators to be well defined and bounded one needs that <sub>k</sub> is square integrable, i.e. $k \in L ^ { 2 } ( \Omega _ { 1 } \times \Omega _ { 2 } )$ . This can be seen by an application of the Cauchy-Schwarz inequality for the $\bar { L ^ { 2 } }$ inner product as follows: 

$$
\| A f \| _ {L ^ {2}} ^ {2} = \int_ {\Omega_ {2}} \left(\int_ {\Omega_ {1}} k (x, y) f (x) \mathrm{d} x\right) ^ {2} \mathrm{d} y.
$$

The inner integral is the $L ^ { 2 }$ inner product between $k ( x , \cdot )$ and <sub>f</sub> and hence, we have 

$$
\begin{array}{l} \| A f \| _ {L ^ {2}} ^ {2} \leq \int_ {\Omega_ {2}} \left(\int_ {\Omega_ {1}} k (x, y) f (x) \mathrm{d} x\right) ^ {2} \mathrm{d} y \\ \leq \int_ {\Omega_ {2}} \int_ {\Omega_ {1}} k (x, y) ^ {2} \mathrm{d} x \int_ {\Omega_ {1}} f (x) ^ {2} \mathrm{d} x \mathrm{d} y \\ = \| k \| _ {L ^ {2} (\Omega_ {1} \times \Omega_ {2})} ^ {2} \| f \| _ {L ^ {2} (\Omega_ {1})} ^ {2}. \end{array}
$$

This also gives the upper bound $\| A \| \leq \| k \| _ { L ^ { 2 } ( \Omega _ { 1 } \times \Omega _ { 2 } ) }$ (which is usually not strict). 

We compute the adjoint of an integral operator using Fubinis theorem to interchange the order of integrals 

$$
\begin{array}{l} \langle A f, g \rangle_ {L ^ {2} (\Omega_ {2})} = \int_ {\Omega_ {2}} A f (y) g (y) \mathrm{d} y \\ \qquad = \int_ {\Omega_ {2}} \int_ {\Omega_ {1}} k (x, y) f (x) \mathrm{d} x g (y) \mathrm{d} y \\ \qquad = \int_ {\Omega_ {1}} f (x) \underbrace {\int_ {\Omega_ {2}} k (x , y) g (y) \mathrm{d} y} _ {= A ^ {*} g (x)} \mathrm{d} x = \langle f, A ^ {*} g \rangle . \end{array}
$$

There is another result that we will use frequently: 

Theorem $^ { 3 \cdot 5 }$ (Dominated convergence for series)<sub>. Let</sub> $a _ { m , n } \in \mathbb { R }$ with $a _ { m , n } \stackrel { m  \infty } { \longrightarrow } a _ { n } ^ { * }$ . If there exists a sequence $b _ { n }$ with $| a _ { m , n } | \leq b _ { n }$ for all m, n and $\textstyle \sum _ { n } b _ { n } < \infty $ , then it holds that 

$$
\sum_ {n} a _ {m, n} \stackrel {m \to \infty} {\longrightarrow} \sum_ {n} a _ {n} ^ {*}.
$$

Informally: If we have a sequence of series, where the summands converge, we can pull the limit under the series if there is a dominating sequence which is summable. 

△ 

## 4 The singular value decomposition and the pseudoinverse

We will build the section on the spectral theorem for compact operators. Recall the notion of compact operator: 

Definition ${ \bf 4 . 1 }$ <sub>.</sub> Some $A \in L ( X , Y )$ is <sub>compact</sub>, if it holds that 

$\left( x _ { n } \right)$ bounded in $X \ \implies \ ( A x _ { n } )$ has convergent subsequence in $Y .$ 

We will denote the set of compact operators from <sub>X</sub> to <sub>Y</sub> by $K ( X , Y )$ . Directly from the definition we get that $K ( X , Y ) \subset$ $L ( X , Y )$ 

Example <sup>4.2</sup>. <sup>For</sup> $A \in L ( X , Y )$ we can say: 

1. If the range of <sub>A</sub> is finite dimensional and <sub>A</sub> is bounded, then <sub>A</sub> is compact, as bounded sequences in finite dimensional spaces always have convergent subsequences. 

2. The identity <sub>id</sub> $: X \to X$ is always bounded, but only compact if <sub>X</sub> is finite dimensional. 

3. If <sub>A</sub> is compact and <sub>B</sub> is bounded than <sub>AB</sub> is compact $( \mathrm { i f }$ defined). Similarly, <sub>BA</sub> is compact (if defined). 

4. Also the adjoint operator $A ^ { * }$ is compact if <sub>A</sub> is compact. 

5. Finally, if $K _ { n }$ is a sequence of compact operators and we have that $\left\| K _ { n } - K \right\| \to 0 ,$ then <sub>K</sub> is compact as well. 

A class of non-trivial compact operators are <sub>integral</sub> <sub>operators</sub>: Example $4 { \cdot } 3 \cdot$ <sub>.</sub> Let $X = L ^ { 2 } ( \Omega _ { 1 } )$ <sub>)</sub> and $Y = L ^ { 2 } ( \Omega _ { 2 } )$ and $k \in L ^ { 2 } ( \Omega _ { 1 } \times$ $\Omega _ { 2 } )$ and define the operator 

$$
K x (t) = \int_ {\Omega_ {1}} k (s, t) x (s) \mathrm{d} s.
$$

We have seen in Example $3 { \cdot } 4$ that <sub>K</sub> is bounded with operator norm $\lVert K \rVert \leq \lVert k \rVert _ { L ^ { 2 } ( \Omega _ { 1 } \times \Omega _ { 2 } ) }$ 

However, <sub>K</sub> is also compact! To see this, note that we can approximate the function <sub>k</sub> by “simple functions”, i.e. there is sequence $k _ { n }$ of functions of the form 

$$
k _ {n} (s, t) = \sum_ {i, j} \alpha_ {i j} \mathbb {1} _ {E _ {i}} (s) \mathbb {1} _ {F _ {j}} (t)
$$

for some disjoint sets $E _ { i } \subset \Omega _ { 1 }$ and $F _ { j } \subset \Omega _ { 2 }$ such that $k _ { n }$ approximated $k ,$ more precisely $\lVert k - k _ { n } \rVert _ { L ^ { 2 } ( \Omega _ { 1 } \times \Omega _ { 2 } ) } \to 0$ . The respective integral operators $K _ { n }$ all have finite dimensional range and hence, are compact. Moreover it holds that 

$$
\left\| K - K _ {n} \right\| \leq \left\| k - k _ {n} \right\| _ {L ^ {2} \left(\Omega_ {1} \times \Omega_ {2}\right)} \rightarrow 0
$$

and thus, <sub>K</sub> is compact as well. 

△ 

18 

Here are some equivalent descriptions of compact operators for those who know what weak convergence in Hilbert spaces mean: 

1. A is compact if it maps bounded sets in X to precompact sets in $Y ( \mathfrak { i . e }$ . their closure is compact). (For bounded operators we only have that they map bounded sets to bounded sets.) 

2. A is compact if it maps weakly convergent subsequences to strongly convergent ones, i.e. $A x _ { n }$ is (strongly) convergent in Y whenever $x _ { n }$ is weakly convergent in X. (For bounded operators we only have that they map strongly convergent sequences to strongly convergent sequences and weakly convergent sequences to weakly convergent sequences.) 

Here we use $\Im _ { E }$ for the so-called characteristic function of the set E, i.e. the function which is 1 on E and zero elsewhere. 

One central theorem for compact operators is the following: 

Theorem ${ 4 } { \cdot } { 4 }$ (Spectral theorem for compact, selfadjoint operators)<sub>.</sub> Let X be a Hilbert space and $K \in K ( X , { \bar { X } } )$ be selfadjoint. Then there exists and orthonormal basis $\left( u _ { n } \right)$ of cl rg(K) and $\lambda _ { n } \in \mathbb { R } \backslash \{ 0 \}$ such that 

$$
K x = \sum_ {n} \lambda_ {n} \left\langle x, u _ {n} \right\rangle u _ {n}.
$$

If the dimension of cl rg(K) is infinite, we also have $\lambda _ { n } \to 0$ 

The proof can be found in H.W. Alt’s book “Linear functional analysis” where this theorem is Theorem 12.12. 

Remark $4 { \cdot } 5 \cdot$ <sub>.</sub> Plugging in $x = u _ { m } ,$ , we get that $\begin{array} { r } { K u _ { m } = \underset { n } { \sum } \lambda _ { n } \left. u _ { m } , u _ { n } \right. u _ { n } = } \end{array}$ $\lambda _ { m } u _ { m }$ and we see that the $u _ { n }$ are actually eigenvectors of <sub>K</sub> for eigenvalues $\lambda _ { n } .$ . By convention, one sorts the eigenvalues by decreasing magnitude, $. . . | \lambda _ { 1 } | \geq | \lambda _ { 2 } | \geq \cdot \cdot \cdot > 0 ,$ 

From the spectral theorem we can deduce the existence of the singular value decomposition (SVD): 

Theorem $\mathbf { 4 . 6 }$ (Singular value decomposition)<sub>. For</sub> <sub>every</sub> $K \in K ( X , Y )$ there exist 

(i) an orthonormal basis $( u _ { n } ) \ { \mathfrak { o f c l } } \mathbf { r g } ( K ) \subset Y ,$ 

(ii) an orthonormal basis $( v _ { n } ) \ g f { \mathrm { c l } } \operatorname { r g } ( K ^ { * } ) \subset X _ { : }$ 

(iii) numbers $\sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdot \cdot \cdot > 0$ 

such that for all n 

$$
K v _ {n} = \sigma_ {n} u _ {n}, \quad a n d \quad K ^ {*} u _ {n} = \sigma_ {n} v _ {n}
$$

and for all $x \in X$ 

$$
K x = \sum_ {n} \sigma_ {n} \left\langle x, v _ {n} \right\rangle u _ {n}.
$$

Proof. <sup>Since</sup> $K ^ { * } K$ is selfadjoint and compact, we get from the spectral theorem the existence of $\lambda _ { n }$ and $v _ { n }$ such that 

$$
K ^ {*} K x = \sum_ {n} \lambda_ {n} \left\langle x, v _ {n} \right\rangle v _ {n}.
$$

Since $\lambda _ { n } \| v _ { n } \| _ { X } ^ { 2 } = \langle \lambda _ { n } v _ { n } , v _ { n } \rangle = \langle K ^ { * } K v _ { n } , v _ { n } \rangle = \langle K v _ { n } , K v _ { n } \rangle = $ $\| K v _ { n } \| _ { X } ^ { 2 } > 0$ we get that $\lambda _ { n } > 0$ . Now we define 

$$
\sigma_ {n} = \sqrt {\lambda_ {n}}, \quad \mathrm{and} \quad u _ {n} = \frac {1}{\sigma_ {n}} K v _ {n}.
$$

Checking that the claimed equalities hold as well as checking orthonormality of the $u _ { n }$ is a routine calculation. □ 

Remark $4 { \cdot } 7 \cdot$ (a) We call $\left( \sigma _ { n } , u _ { n } , v _ { n } \right)$ <sup>the</sup> singular system <sup>of</sup> $K .$ . 

(b) We also get the singular value decomposition of $K ^ { * }$ , namely 

$$
K ^ {*} y = \sum_ {n} \sigma_ {n} \left\langle y, u _ {n} \right\rangle v _ {n}.
$$

(c) The $\sigma _ { n }$ <sup>are</sup> <sup>called</sup> singular values<sup>,</sup> <sup>the</sup> $u _ { n }$ <sup>are</sup> left singular vectors and the $v _ { n }$ <sup>are</sup> right singular vectors<sup>.</sup> 

(d) The singular vectors can be used to project onto the closures of the ranges of <sub>K</sub> and $K ^ { * } { } _ { i }$ , namely 

$$
P _ {\mathrm{clrg} (K)} y = \sum_ {n} \left\langle y, u _ {n} \right\rangle u _ {n}, \quad P _ {\mathrm{clrg} (K ^ {*})} x = \sum_ {n} \left\langle x, u _ {n} \right\rangle u _ {n}.
$$

(e) We have $\begin{array} { r } { \sum _ { n } | \langle x , v _ { n } \rangle | ^ { 2 } = \| P _ { \mathrm { c l } ( \mathrm { r g } ( K ) ) } ( x ) \| _ { X } \leq \| x \| _ { X } . } \end{array}$ 

The singular value decomposition also allows to describe the boundary of the range: 

Theorem $_ { 4 } . 8$ (Picard condition)<sub>.</sub> <sub>Let</sub> $K \in K ( X , Y )$ with singular system $\left( \sigma _ { n } , u _ { n } , v _ { n } \right)$ and let $y \in \mathrm { c l } ( \mathbf { r g } ( K ) )$ ). Then $y \in \mathop { \bf r g } ( K )$ exactly $i f$ 

$$
\sum_ {n} \frac {| \langle y , u _ {n} \rangle | ^ {2}}{\sigma_ {n} ^ {2}} <   \infty .\tag{P}
$$

Proof. <sup>Let</sup> $y \in \mathop { \bf r g } ( K )$ , then there is <sub>x</sub> with $y = K x$ and we have 

$$
\left\langle y, u _ {n} \right\rangle = \left\langle K x, u _ {n} \right\rangle = \left\langle x, K ^ {*} u _ {n} \right\rangle = \sigma_ {n} \left\langle x, v _ {n} \right\rangle .
$$

We get 

$$
\sum_ {n} \frac {| \langle y , u _ {n} \rangle | ^ {2}}{\sigma_ {n} ^ {2}} = \sum_ {n} | \langle x, v _ {n} \rangle | ^ {2} \leq \| x \| _ {X} ^ {2} <   \infty .
$$

Conversely, the $y \in \mathrm { c l } ( \mathbf { r g } ( K ) )$ fulfill (P). We define $\begin{array} { r } { x _ { N } = \sum _ { n = 1 } ^ { N } \frac { 1 } { \sigma _ { n } } \left. y , u _ { n } \right. v _ { n } } \end{array}$ and from (P) it follows that $x _ { N }$ is a Cauchy sequence, and thus, 

$$
x _ {N} \rightarrow \sum_ {n} \frac {1}{\sigma_ {n}} \left<   y, u _ {n} \right> v _ {n} =: x.
$$

Finally, we get 

$$
\begin{array}{l} K x = K \left(\sum_ {n} \frac {1}{\sigma_ {n}} \langle y, u _ {n} \rangle v _ {n}\right) = \sum_ {n} \frac {1}{\sigma_ {n}} \langle y, u _ {n} \rangle K v _ {n} = \sum_ {n} \langle y, u _ {n} \rangle u _ {n} \\ = P _ {\mathrm{cl} (\mathrm{rg} (K))} y = y \end{array}
$$

which shows that $y \in \mathop { \bf r g } ( K )$ 

With the singular value decomposition, we can define the so-called <sub>Moore-Penrose</sub> <sub>pseudo-inverse</sub> (often just called pseudoinverse). 

Definition $\pmb { 4 } { \cdot } \pmb { 9 }$ (Pseudo-inverse)<sub>.</sub> Let $K \in K ( X , Y )$ with singular system $\left( \sigma _ { n } , u _ { n } , v _ { n } \right)$ <sup>.</sup> <sup>Then</sup> <sup>the</sup> pseudo-inverse <sup>of</sup> $K _ { r }$ , is $K ^ { \dagger } : \mathrm { r g } ( K )$ ⊕ $\mathrm { r g } ( K ) ^ { \perp } \to X$ defined by 

$$
K ^ {\dagger} y = \sum_ {n} \frac {1}{\sigma_ {n}} \left<   y, u _ {n} \right> v _ {n}.
$$

We denote by $D ( K ^ { \dagger } ) : = \mathbf { r g } ( K ) \oplus \mathbf { r g } ( K ) ^ { \perp } \subset Y$ the <sub>domain</sub> of the pseudo-inverse. 

Remark $4 { \cdot } 1 0 . \qquad 1$ . Note that 

$$
\begin{array}{c}K ^ {\dagger} K x = \sum_ {n} \frac {1}{\sigma_ {n}} \left<   K x, u _ {n} \right> v _ {n} = \sum_ {n} \frac {1}{\sigma_ {n}} \left<   x, K ^ {*} u _ {n} \right> v _ {n}\\= \sum_ {n} \left<   x, v _ {n} \right> v _ {n} = P _ {\mathrm{clrg} (K ^ {*})} x = P _ {\ker (K) ^ {\perp}} x,\end{array}
$$

i.e. $K ^ { \dagger } K = P _ { \mathrm { k e r } ( K ) ^ { \perp } } .$ 

2. Similarly, we have 

$$
\begin{array}{l} K K ^ {\dagger} y = \sum_ {n} \sigma_ {n} \left\langle K ^ {\dagger} y, v _ {n} \right\rangle u _ {n} = \sum_ {n} \sigma_ {n} \left\langle \sum_ {m} \frac {1}{\sigma_ {m}} \left\langle y, u _ {m} \right\rangle v _ {m}, v _ {n} \right\rangle u _ {n} \\ = \sum_ {m, n} \sigma_ {n} \frac {1}{\sigma_ {m}} \left\langle y, u _ {m} \right\rangle \left\langle v _ {m}, v _ {n} \right\rangle u _ {n} = \sum_ {n} \left\langle y, u _ {n} \right\rangle u _ {n} = P _ {\mathrm{cl} (\mathrm{rg} (K))} y \\ = P _ {\ker (K ^ {*}) ^ {\perp}}, \end{array}
$$

i.e. $K K ^ { \dagger } = P _ { \mathrm { c l ( r g ( } K \mathcal { ) } ) } = P _ { \mathrm { k e r ( } K ^ { * } ) ^ { \perp } } .$ 

3. We have $K ^ { \dagger } y = 0 \operatorname { i f } y \in \mathbf { r g } ( K ) ^ { \perp } , \operatorname { i . e . } \ker ( K ^ { \dagger } ) = \mathbf { r g } ( K ) ^ { \perp } .$ 

4. Since $\left( v _ { n } \right)$ is a basis of <sub>cl</sub> $\operatorname { r g } ( K ^ { * } ) = \ker ( K ) ^ { \perp }$ , we have that $\arg ( K ^ { \dagger } ) = \operatorname { c l } \operatorname { r g } ( K ^ { * } ) = \ker ( K ) ^ { \perp }$ . Note that $\mathrm { r g } ( K )$ is in general not closed (for compact operators it is only closed if it is finite dimensional), i.e. it is not a Hilbert space. 

By the above remark, the pseudo-inverse is actually a kind of an inverse, namely of $K | _ { \mathrm { k e r } ( K ) ^ { \perp } } : \mathrm { k e r } ( K ) ^ { \perp } \to \mathrm { r g } ( K )$ . There is a little more to say: 

Theorem 4.11. For every $y \in D ( K ^ { \dagger } )$ it holds that the equation $K x = y$ <sub>has</sub> <sub>a</sub> <sub>unique</sub> minimum norm solution <sub>which</sub> <sub>is</sub> $x ^ { \dagger } = K ^ { \dagger } y , i . e . x ^ { \dagger }$ is a least squares solution of minimal norm, i.e. it holds that 

∥Kx<sup>†</sup> − y∥<sub>Y</sub> = min {∥Kx − y∥<sub>Y</sub> | x ∈ X} and 

The first equality defines “least squares solutions”. 

$\| x ^ { \dag } \| _ { X } = \operatorname* { m i n } \left\{ \| z \| _ { X } \mid z \right.$ is a least squares solution of $K x = y \}$ 

Moreover, the set of all least squares solutions is $x ^ { \dagger } + \ker ( K )$ 

Proof. <sup>That</sup> $x ^ { \dagger } = K ^ { \dagger } y$ is a least squares solution follows from Remark 4.10, 2.: 

$$
\| K x ^ {\dagger} - y \| _ {Y} = \| K K ^ {\dagger} y - y \| _ {Y} = \| P _ {\operatorname{cl} (\operatorname{rg} (K))} y - y \| _ {Y}.
$$

Now recall that the orthogonal projection $P _ { \mathrm { c l ( r g ( } K ) ) } y$ is the closest point to <sub>y</sub> within the closure of range of <sub>K</sub>. 

If $\cdot _ { x ^ { \prime } }$ is any least squares solution $x ^ { \prime }$ we can write $x ^ { \prime } = x ^ { \dagger } +$ <sub>v</sub> with $v \in \ker ( K )$ , but since $x ^ { \dagger } \in \mathop { \mathrm { k e r } } ( K ) ^ { \perp }$ we have (by the Pythagorean theorem) 

$$
\| x ^ {\prime} \| _ {X} ^ {2} = \| x ^ {\dagger} \| _ {X} ^ {2} + \| v \| _ {X} ^ {2} \geq \| x ^ {\dagger} \| _ {X} ^ {2}
$$

which shows that $x ^ { \dagger }$ has minimal norm among all least squares solutions. □ 

Remark $4 { \cdot } 1 2 .$ <sub>.</sub> The pseudo inverse can also be defined for general bounded linear operators (not necessarily compact ones) $A \ \in$ $L ( X , Y )$ . There one defines the $A ^ { \dagger } y$ as the unique minimum norm least squares solution (and has to show that this is a meaningful definition). All properties of the pseudo inverse we have shown are still fulfilled in this case. 

We will use the pseudo-inverse also for merely bounded operators in the following. 

## 5 Regularization

We have seen in the previous section that the pseudo-inverse solves two of the problems with ill-posed linear problems: Existence and uniqueness. A little bit more explicit: The problem of existence is (roughly) solved by moving to least squares solutions $( { \mathrm { i . e . } }$ minimizing the residual $\| K x - { \bar { y } } \| _ { Y }$ rather than solving $K x = y )$ and the problem of uniqueness is solved by considering minimum norm solutions, i.e. among all (least squares) solution we pick the one with minimal norm. What about the remaining problem of instability? 

Before we answer that, we note the following fact: 

Lemma ${ \pmb 5 } { \cdot } { \bf 1 }$ . $H K \in K ( X , Y )$ has the singular system $\left( \sigma _ { n } , u _ { n } v _ { n } \right)$ , then we have $\| K \| = \sigma _ { 1 }$ 

The proof is a good exercise 

Unfortunately, this shows that the pseudo-inverse is, in general, not bounded: $\operatorname { I f } \operatorname { r g } ( K )$ is infinite dimensional, we have from Theorem $4 { \cdot } 6$ that $\sigma _ { n } \to 0$ . But this implies 

$$
\| K ^ {\dagger} u _ {n} \| = \| \sum_ {m} \frac {1}{\sigma_ {m}} \left\langle u _ {n}, u _ {m} \right\rangle v _ {m} \| = \| \frac {1}{\sigma_ {n}} v _ {n} \| = \frac {1}{\sigma_ {n}} \stackrel {n \to \infty} {\longrightarrow} \infty
$$

and thus, $K ^ { \dagger }$ can not be bounded. The pseudo-inverse even helps to make the instability quite quantifiable: Consider the case that $y ^ { \dagger } =$ $K x ^ { \dagger }$ for $x \in \ker ( K ) ^ { \perp }$ . Then $x ^ { \dagger }$ is the minimum norm least squares solution of $\boldsymbol { K } \boldsymbol { x } = \boldsymbol { y } ^ { \dagger }$ . Let’s assume that we have measurement data $y ^ { \delta }$ instead of $y ^ { \dagger }$ and let us assume moreover, that we know that we have a small measurement error, i.e. $\| y ^ { \dagger } - y ^ { \delta } \| _ { Y } \leq \delta$ for some known $\delta > 0$ . Then the “noise” ns the data is 

$$
\eta = y ^ {\delta} - y ^ {\dagger} \in Y.
$$

Let us blindly apply the pseudo inverse to $y ^ { \delta } \colon$ 

$$
K ^ {\dagger} y ^ {\delta} = K ^ {\dagger} (y ^ {\dagger} + \eta) = x ^ {\dagger} + K ^ {\dagger} \eta = x ^ {\dagger} + \sum_ {n} \frac {1}{\sigma_ {n}} \left\langle \eta , u _ {n} \right\rangle v _ {n}.
$$

We see that the contribution of the noise is amplified unboundedly, i.e. the component $\left. \eta , u _ { n } \right.$ of the noise in the <sub>n</sub>-th singular vector $u _ { n }$ is amplified by a factor of <sub>1</sub> $/ \sigma _ { n }$ and these factors grow beyond all bounds. Hence: If the noise contains contributions from singular vectors that correspond to small singular values, they get amplified a lot. Unfortunately, this is the standard situation: Singular vectors for small singular values tend to be oscillatory $( { \mathrm { i . e . } }$ . be of high frequency) and hence, noise always tends to be amplified. 

Example $5 { \cdot } 2$ (Discretized inverse problems)<sub>.</sub> One can check this observation numerically. After discretization, an inverse problem still reads as $\mathbf { K } \mathbf { x } = \mathbf { y } ^ { \delta }$ with $\mathbf { x } \in \mathbb { R } ^ { n } , \mathbf { y } \in \mathbb { R } ^ { m }$ and $\mathbf { K } \in \mathbb { R } ^ { m \times n }$ . The singular value decomposition exists as well and if we write the 

It’s worth to consider the finite dimensional case here: If $K \ = \ U \Sigma V ^ { T }$ is the singular value decomposition, then $\| K \| = \sigma _ { 1 }$ and $\lVert K ^ { \dagger } \rVert = \dot { 1 } / \sigma _ { k }$ where k is the smallest singular value. The condition number of K is defined as $\kappa ( K ) =$ $\| K \| \| K ^ { \dagger } \|$ and hence, equal the ratio of the largest and smallest singular value of $K .$ . For inverse problems in infinite dimensions, the condition number can the infinite as there may be arbitrarily small singular values. 

singular vectors $\mathbf { u } _ { i }$ and $\mathbf { v } _ { j }$ as colums in matrices <sub>U</sub> and <sub>V</sub> and the singular values $\sigma _ { i }$ on the diagonal of a matrix <sub>Σ</sub>, we get 

$$
\mathbf {K} \mathbf {x} = \sum_ {i} \sigma_ {i} \left\langle \mathbf {x}, \mathbf {v} _ {i} \right\rangle \mathbf {u} _ {i} = \mathbf {U} \boldsymbol {\Sigma} \mathbf {V} ^ {T} \mathbf {x}.
$$

The pseudo inverse is 

$$
\mathbf {K} ^ {\dagger} \mathbf {y} = \sum_ {i} \frac {1}{\sigma_ {i}} \left\langle \mathbf {y}, \mathbf {u} _ {i} \right\rangle \mathbf {v} _ {i} = \mathbf {V} \boldsymbol {\Sigma} ^ {\dagger} \mathbf {U} ^ {T} \mathbf {y}.\tag{*}
$$

where $\Sigma ^ { \dagger }$ has the values $1 / \sigma _ { i }$ on the diagonal. 

Let us consider a (quite simple) discrete approximation of the inverse problem of diferentiation, i.e. the inversion of <sub>A</sub> given by $\textstyle A f ( x ) { \overset { - } { = } } \int _ { 0 } ^ { x } f ( t ) \mathrm { d } t$ . This operator can be (roughly) discretized by the matrix 

$$
\mathbf {A} = \frac {1}{n} \left[ \begin{array}{c c c c} 1 & 0 & \dots & 0 \\ \vdots & \ddots & \ddots & \vdots \\ \vdots & & \ddots & 0 \\ 1 & \dots & \dots & 1 \end{array} \right] \in \mathbb {R} ^ {n \times n}.
$$

Here is an example of the naive reconstruction (we can use a direct solve here, since the matrix is actually invertible (it is square and it smallest singular value is positive, but quite small). We could, in principle, also use <sub>pinv</sub> to calculate the pseudo-inverse or use the formula (*)). 

```python
import numpy as np
import matplotlib.pyplot as plt 
```

```txt
# problem size and matrix
n = 100
A = np.tril(np.ones((n,n))) / n

# discretized interval
t = np.linspace(0,1,n)

# true solution
xdag = 1 - t**2
# noise free data
ydag = A@xdag

# noisy data
eta = np.random.randn(n);
eta /= np.sum(eta)

# noise level
delta = 0.05
ydelta = ydag + delta*eta 
```

```python
# naive reconstruction
x = np.linalg.solve(A, ydelta)

fig, axes = plt.subplots(2, 2)
axs[0, 0].plot(t, xdag)
axs[0, 0].set_title('true solution')
axs[0, 1].plot(t, ydag)
axs[0, 0].set_title('true data')
axs[1, 0].plot(t, x)
axs[0, 0].set_title('naive reconstruction')
axs[1, 1].plot(t, ydelta)
axs[0, 0].set_title('noisy data')
plt.show() 
```

```csv
noisy data
1.00
1.00
1.00
1.00
1.00
1.00
1.00
1.00
1.00
1.00
1.00
1.00
1.00
1.00
1.00
1.00
1.00
1.00
1.00
1.00
1.0 
```

```python
# compute svd
U,S,VT = np.linalg.svd(A)

# show some singular vectors
fig, axes = plt.subplots(5,2)
axs[0,0].plot(t,U[:,0])
axs[0,0].set_title("$u_1$')
axs[0,1].plot(t,VT[0,:])
axs[0,1].set_title("$v_1$')
axs[1,0].plot(t,U[:,1])
axs[1,0].set_title("$u_2$')
axs[1,1].plot(t,VT[1,:])
axs[1,1].set_title("$v_2$')
axs[2,0].plot(t,U[:,2]) 
```

# plot singular vectors in semilog plot plt.semilogy(S) plt.show () 

axs [2 ,0]. set_title (’$u_3$ ’) axs [2 ,1]. plot(t,VT [2 ,:]) axs [2 ,1]. set_title (’$v_3$ ’) axs [3 ,0]. plot(t,U[: ,9]) axs [3 ,0]. set_title (’$u_ {10}$’) axs [3 ,1]. plot(t,VT [9 ,:]) axs [3 ,1]. set_title (’$v_ {10}$’) axs [4 ,0]. plot(t,U[: , -1]) axs [4 ,0]. set_title (’$u_n$ ’) axs [4 ,1]. plot(t,VT[ -1 ,:]) axs [4 ,1]. set_title (’$v_n$ ’) plt.show () 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/fe404709f1d5ad401b64e700a55eab6260c15eddd84520babfdeedea16155697.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/a425fd44b02e4aea1ce2ec4b849cffcda78c8c489291574683739607767fe6f5.jpg)



Now, let us fix our aim and let us define what a “regularization” shall be:


<sub>Definition</sub> <sub>5.3</sub> (Regularization)<sub>.</sub> Let $A \in L ( X , Y )$ <sup>.</sup> <sup>A</sup> regularization of $A ^ { \dagger }$ is a family of continuous maps $R _ { \alpha } : Y \to X , \alpha > 0$ such that for all $y \in D ( \bar { A ^ { \dag } } )$ it holds that 

$$
R _ {\alpha} y \xrightarrow {\alpha \to 0} A ^ {\dagger} y.
$$

If all $R _ { \alpha }$ are linear, we speak of a <sub>linear</sub> <sub>regularization</sub>. The parameter α <sup>is</sup> <sup>called</sup> regularization parameter<sup>.</sup> 

As a matter of fact, any linear regularization can not be uniformly bounded (as they approximate an unbounded operator): 

Theorem 5.4. Let $A \in L ( X , Y )$ and $R _ { \alpha }$ be a linear regularization of $A ^ { \dagger } . H A ^ { \dagger }$ is unbounded, then it holds that $\| R _ { \alpha } \| _ { X } \xrightarrow { \alpha \to 0 } \infty$ 

Now let us discuss the various error we defined in the example in Section 2: The <sub>totel</sub> <sub>error</sub> $\| R _ { \alpha } y ^ { \delta } - x ^ { \dagger } \|$ (also called regularization error) can be decomposed (using $x ^ { \dagger } = A ^ { \dagger } y ^ { \dagger }$ and the triangle inequality) in the case of linear regularization into <sub>data</sub> <sub>error</sub> and approximation error 

This follows from the so-called uniform boundedness prinicple (also known as Banach-Steinhaus Theorem) and we do not discuss the proof here. 

$$
\begin{array}{c} \| R _ {\alpha} y ^ {\delta} - x ^ {\dagger} \| _ {X} \leq \| R _ {\alpha} y ^ {\delta} - R _ {\alpha} y ^ {\dagger} \| _ {X} + \| R _ {\alpha} y ^ {\dagger} - A ^ {\dagger} y ^ {\dagger} \| _ {X} \\ \leq \| R _ {\alpha} \| \delta + \| R _ {\alpha} y ^ {\dagger} - A ^ {\dagger} y ^ {\dagger} \| _ {X}. \end{array}\tag{*}
$$

Again, we see that one needs to choose the regularization parameter <sub>α</sub> carefully: At least we need to be able to balance the term $\| R _ { \alpha } \| \delta$ as it blows up for small <sub>α</sub>. On the other hand, the term $\| R _ { \alpha } \dot { y } ^ { \dagger } - A ^ { \dagger } y ^ { \dagger } \| _ { X }$ tends to be small for small <sub>α</sub> and large for large <sub>α</sub> (exactly as we have seen in Section 2). 

We will often denote the regularized reconstruction by $x _ { \alpha } ^ { \delta } : =$ $R _ { \alpha } y ^ { \delta }$ . We will also use the notation $x _ { \alpha } : = R _ { \alpha } y ^ { \dagger }$ for the (in general unknown) regularized reconstruction from idealized noiseless data. With this notation, our error decomposition is 

$$
\underbrace {\| x _ {\alpha} ^ {\delta} - x ^ {\dagger} \| _ {X}} _ {\text {total error}} \leq \underbrace {\| x _ {\alpha} ^ {\delta} - x _ {a} \| _ {X}} _ {\text {data error}} + \underbrace {\| x _ {\alpha} - x ^ {\dagger} \| _ {X}} _ {\text {approximation error}}.
$$

<sub>Definition</sub> <sub>5.5</sub> (Parameter choice)<sub>.</sub> A function 

$$
\alpha : ] 0, \infty [ \times Y \rightarrow ] 0, \infty [, (\delta , y ^ {\delta}) \rightarrow \alpha (\delta , y ^ {\delta})
$$

is called a <sub>parameter</sub> <sub>choice</sub> <sub>rule</sub>. We distinguish further: <sub>α</sub> is 

(i) an <sub>a</sub> <sub>priori</sub> <sub>choice</sub> <sub>rule</sub> if <sub>α</sub> does not depend on $y ^ { \delta } { } _ { \colon }$ , 

<sup>(ii)</sup> <sup>an</sup> a posteriori choice rule <sup>if</sup> α <sup>depends</sup> <sup>on</sup> δ <sup>and</sup> $y ^ { \delta }$ , and 

(iii) a <sub>heuristic</sub> <sub>rule</sub> is <sub>α</sub> does not depend on $\delta .$ 

Definition ${ \pmb 5 } { \pmb 6 }$ (Convergent regularization)<sub>.</sub> If $\mathrm { T } R _ { \alpha }$ is a regularization of $A ^ { \dagger }$ and <sub>α</sub> is a parameter choice rule we say that $\left( R _ { \alpha } , \alpha \right)$ is a convergent regularization method <sup>if</sup> <sup>for</sup> <sup>all</sup> $y ^ { \dag } \in \hat { \cal D ( A ^ { \dag } ) }$ it holds that 

An a priori rule can be devised without having seen the actual data (it only needs knowledge of the noise level), hence one can, in principle, construct the operator $R _ { \alpha ( \delta ) }$ a priorily, before the data has arrived; hence, the name. 

$$
\sup \left\{\| R _ {\alpha (\delta , y ^ {\delta})} y ^ {\delta} - A ^ {\dagger} y ^ {\dagger} \| _ {X} \mid \| y ^ {\delta} - y ^ {\dagger} \| _ {Y} \leq \delta \right\} \xrightarrow {\delta \to 0} 0.
$$

<sup>In</sup> <sup>other</sup> <sup>words:</sup> <sup>We</sup> <sup>want</sup> <sup>that</sup> <sup>the</sup> worst case reconstruction error goes to zero, i.e. even if the noisy data $y ^ { \delta }$ is as bad as possible, given the noise level . 

<sub>Example</sub> 5.7 (Truncated SVD)<sub>.</sub> Here is a simple idea for a regularization method for: For $K \in K ( X , Y )$ with singular system $\left( \sigma _ { n } , u _ { n } , v _ { n } \right)$ and $\alpha > 0$ we define 

$$
R _ {\alpha} y = \sum_ {\sigma_ {n} > \alpha} \frac {1}{\sigma_ {n}} \left<   y, u _ {n} \right> v _ {n}
$$

i.e. we cut of the small singular values which lead to unboundedness of the pseudo-inverse. These $R _ { \alpha }$ are indeed bounded operators: 

$$
\| R _ {\alpha} y \| _ {X} ^ {2} = \sum_ {\sigma_ {n} > \alpha} \frac {1}{\sigma_ {n} ^ {2}} | \langle y, u _ {n} \rangle | ^ {2} \leq \sup \left\{\frac {1}{\sigma_ {n} ^ {2}} \mid \sigma_ {n} \geq \alpha \right\} \| y \| _ {Y} ^ {2} \leq \frac {1}{\alpha^ {2}} \| y \| _ {Y} ^ {2}
$$

Informally: We demand that in the regime of vanishing noise, we shall be able to approximate the true solution $x ^ { \dagger } = A ^ { \dagger } y ^ { \dagger }$ as good as possible. On the one hand, this sounds like a meaningless demand, since usually the noise level stays fixed. On the other hand, it sounds like something that should be the bare minimum: If we can not even guarantee this, what is the point of regularization at all? Finally, it sounds quite ambitious, given that we already know that we try to approximate unbounded $( \mathsf { i . e . }$ discontinuous) operators with continuous ones. 

i.e. $\textstyle \| R _ { \alpha } \| \leq { \frac { 1 } { \alpha } }$ 

Let us investigate if the truncated SVD is a convergent regularization method, i.e. if we can find a suitable parameter choice: To that end, we use our standard error decomposition (*) 

$$
\begin{array}{c} \| R _ {\alpha} y ^ {\delta} - K ^ {\dagger} y ^ {\dagger} \| _ {X} \leq \| R _ {\alpha} \| \delta + \| R _ {\alpha} y ^ {\dagger} - A ^ {\dagger} y ^ {\dagger} \| _ {X} \\ \leq \frac {\delta}{\alpha} + \| R _ {\alpha} y ^ {\dagger} - A ^ {\dagger} y ^ {\dagger} \| _ {X}. \end{array}
$$

We estimate the approximation error further 

$$
R _ {\alpha} y ^ {\dagger} - A ^ {\dagger} y ^ {\dagger} = \sum_ {\sigma_ {n} \geq \alpha} \frac {1}{\sigma_ {n}} \left\langle y ^ {\dagger}, u _ {n} \right\rangle v _ {n} - \sum_ {n} \frac {1}{\sigma_ {n}} \left\langle y ^ {\dagger}, u _ {n} \right\rangle v _ {n} = - \sum_ {\sigma_ {n} <   \alpha} \frac {1}{\sigma_ {n}} \left\langle y ^ {\dagger}, u _ {n} \right\rangle v _ {n}.
$$

This gives us 

$$
\| R _ {\alpha} y ^ {\dagger} - A ^ {\dagger} y ^ {\dagger} \| _ {X} ^ {2} \leq \sum_ {\sigma_ {n} <   \alpha} \frac {1}{\sigma_ {n} ^ {2}} | \left<   y ^ {\dagger}, u _ {n} \right> | ^ {2}.
$$

Together we have 

$$
\begin{array}{c} \| R _ {\alpha} y ^ {\delta} - K ^ {\dagger} y ^ {\dagger} \| _ {X} \leq \| R _ {\alpha} \| \delta + \| R _ {\alpha} y ^ {\dagger} - A ^ {\dagger} y ^ {\dagger} \| _ {X} \\ \leq \frac {\delta}{\alpha} + \sqrt {\sum_ {\sigma_ {n} <   \alpha} \frac {1}{\sigma_ {n} ^ {2}} | \langle y ^ {\dagger} , u _ {n} \rangle | ^ {2}}. \end{array}
$$

Now we see: The second summand is the “rest of a convergent series” (recall the Picard condition, Theorem $_ { 4 \cdot 8 ) }$ and the smaller $\alpha ,$ the later the rest of the series starts. Hence, we have 

$$
\sqrt {\sum_ {\sigma_ {n} <   \alpha} \frac {1}{\sigma_ {n} ^ {2}} | \langle y ^ {\dagger} , u _ {n} \rangle | ^ {2}} \to 0 \quad \mathrm{for} \quad \alpha \to 0.
$$

For the first term we need that $\alpha ( \delta )  0$ slower than $\delta .$ In conclusion: Any $\alpha ( \delta )$ with 

$$
\alpha (\delta) \xrightarrow {\delta \to 0} 0, \quad \frac {\delta}{\alpha (\delta)} \xrightarrow {\delta \to 0} 0
$$

is a valid (a priori) parameter choice rule and we can claim that the truncated SVD together with this rule is a convergent regularization method. $\bigtriangleup$ 

One could take, for example, $\alpha ( \delta ) =$ $\sqrt { \delta } ( \mathsf { o r } = \delta ^ { \kappa }$ for $0 < \kappa < 1$ , for that matter). 

## 6 Tikhonov regularization

The problem of instability of the solution of $A x = y ^ { \delta }$ comes from the small singular values which are the eigenvalues of the selfadjoint operator $A ^ { * } A$ . Another way to understand this, is via the normal equation: Some <sub>x</sub> is a minimizer of $\| A x - y ^ { \delta } \| _ { X } ^ { 2 }$ exactly if <sup>it</sup> <sup>solves</sup> <sup>the</sup> normal equation 

$$
A ^ {*} A x = A ^ {*} y ^ {\delta}.
$$

However, in general minimizers of $\| A x - y ^ { \delta } \| _ { X } ^ { 2 }$ do not exist $( { \mathrm { i } } . { \mathsf { e } } .$ the normal equation does not have solutions) and this is (in the case of compact $A )$ due to the eigenvalues of $A ^ { * } A$ converging to zero. To avoid this problem, we can simply shift them to be strictly positive: $\operatorname { I f } \sigma _ { i } ^ { 2 }$ are the eigenvalues of $A ^ { * } A$ , then the eigenvalues of $A ^ { * } A + \alpha$ <sub>id</sub> are $\sigma _ { n } ^ { 2 } + \bar { \alpha _ { { \mathrm { ~ } } } } \geq \alpha > 0$ . Hence, instead of the normal equation, we consider for $\alpha > 0$ regularized normal equations 

$$
(A ^ {*} A + \alpha \mathrm{id}) x = A ^ {*} y ^ {\delta}.
$$

Since the operator $A ^ { * } A + \alpha$ <sub>id</sub> is always invertible, we can write this as 

$$
x _ {\alpha} ^ {\delta} = (A ^ {*} A + \alpha \mathrm{id}) ^ {- 1} A ^ {*} y ^ {\delta}
$$

and this method is called <sub>Tikhonov</sub> <sub>regularization</sub>. The shift of the singular values is one motivation for Tikhonov regularization. In fact, Tikhonov regularization also corresponds to a regularized least squares problem. 

Theorem ${ \bf 6 . 1 . }$ . Let $A \ \in \ L ( X , Y )$ . The regularized normal equation $( A ^ { * } A + \alpha \operatorname { i d } ) x = A ^ { * } y ^ { \delta }$ has a unique solution $x _ { \alpha } ^ { \delta }$ which is exactly the unique minimum of the Tikhonov functional 

$$
T _ {\alpha} (x; y ^ {\delta}) := \frac {1}{2} \| A x - y ^ {\delta} \| _ {Y} ^ {2} + \frac {\alpha}{2} \| x \| _ {X} ^ {2}.
$$

<sub>Proof.</sub> A minimizer <sub>x</sub> of the Tikhonov function is characterized by the condition that $T _ { \alpha } ( x + t h ; y ^ { \delta } ) \ge T _ { \alpha } ( x ; y ^ { \delta } )$ for all $t \in \mathbb { R }$ and $h \in X$ . Starting from the left hand side we get 

$$
\begin{array}{l} T _ {\alpha} (x + t h; y ^ {\delta}) = \frac {1}{2} \| A x + t A h - y ^ {\delta} \| _ {Y} ^ {2} + \frac {\alpha}{2} \| x + t h \| _ {X} ^ {2} \\ \qquad = \frac {1}{2} \| A x - y ^ {\delta} \| _ {Y} ^ {2} + \Big \langle A x - y ^ {\delta}, t A h \Big \rangle + \frac {1}{2} \| t A h \| _ {Y} ^ {2} \\ \qquad + \frac {\alpha}{2} \| x \| _ {X} ^ {2} + \alpha   \langle x, t h \rangle + \frac {\alpha}{2} \| t h \| _ {X} ^ {2} \\ \qquad = T _ {\alpha} (x; y ^ {\delta}) + t \left\langle A ^ {*} (A x - y ^ {\delta}) + \alpha x, h \right\rangle + t ^ {2} (\frac {1}{2} \| A h \| _ {Y} ^ {2} + \frac {\alpha}{2} \| h \| _ {X} ^ {2}). \end{array}
$$

We see that $T _ { \alpha } ( x + t h ; y ^ { \delta } ) \ge T _ { \alpha } ( x ; y ^ { \delta } )$ holds for all <sub>t</sub> and <sub>h</sub> exactly if 

$$
\left\langle A ^ {*} (A x - y ^ {\delta}) + \alpha x, h \right\rangle = 0
$$

for all $h \in X$ and this is exactly the case when $A ^ { * } ( A x - y ^ { \delta } ) + \alpha x =$ <sub>0</sub> which is just the regularized normal equation. Uniqueness of the minimizer follows since the Tikhonov functional is strictly convex. □ 

The description of Tikhonov regularization as a minimization framework allows for another interpretation: The regularization is a compromise of two things, namely finding a reconstruction $x _ { \alpha } ^ { \delta }$ that has a good <sub>data</sub> <sub>fit</sub>, i.e. it produces a small value for the residual $\| A x - y ^ { \delta } \| _ { Y }$ , but, at the same time, also does not blow up, i.e. it has a small norm $\| x \| _ { X }$ . These two demands are weighted by the regularization parameter <sub>α</sub>. Regularization methods that build upon the idea of minimizing a functional that balances the demands of data fit and “reasonable reconstruction” are also called “variational regularization methods” (as the theory that deals with minimization problems in infinite dimensional spaces is called “calculus of variations”). Aiming at a reconstruction with a bounded norm seems like a valid idea, but one may know a little more about the unknown solution. If we assume that we have a rough idea of the unknown $x ^ { \dagger }$ , i.e. we know that $x ^ { 0 }$ is a good guess, we can of course minimize 

$$
T _ {\alpha} (x; y ^ {\delta}, x ^ {0}) := \frac {1}{2} \| A x - y ^ {\delta} \| _ {Y} ^ {2} + \frac {\alpha}{2} \| x - x ^ {0} \| _ {X} ^ {2}.
$$

Similar to the proof of Theorem 6.1 one shows that the unique minimizer here is given as a solution of 

$$
(A ^ {*} A + \alpha \mathrm{id}) x = A ^ {*} y ^ {\delta} + \alpha x ^ {0}.
$$

Remark $6 . 2$ (Numerical realization of Tikhonov regularization)<sub>.</sub> Tikhonov regularization is popular, because its implementation is pretty straight forward. Let us consider the discrete case where $\mathbf { A } \in \mathbb { K } ^ { m \times n }$ and $\mathbf { y } ^ { \delta } \in \mathbb { R } ^ { m }$ . Then the regularized normal equation $( \mathrm { f o r } { \bf x } ^ { 0 } = 0 $ 

$$
(\mathbf {A} ^ {T} \mathbf {A} + \alpha I _ {n}) \mathbf {x} = \mathbf {A} ^ {T} \mathbf {y} ^ {\delta}
$$

Both the overdetermined case $m > n$ (where non-existence of solutions is a problem, due to measurement error) and the underdetermined case m < n (where non-uniqueness is a problem, due to not enough data) of can be considered here. 

is a square linear system in <sub>n</sub> dimensions and the matrix $( \mathbf { A } ^ { T } \mathbf { A } +$ $\alpha I _ { n } )$ is symmetric positive definite. Hence, there are many methods available to solve the problem numerically (one method is the method of conjugate gradients). 

Is Tikhonov regularization indeed a convergence regularization method? To answer this question, we should find a parameter choice rule. We will analyze this question with the help of the singular value decomposition. 

Theorem 6.3. Let $K \in K ( X , Y )$ have the singular system $\left( \sigma _ { n } , u _ { n } , v _ { n } \right)$ Then solution $x _ { \alpha } ^ { \delta } \circ f ( A ^ { * } A + \alpha \operatorname { i d } ) x = A ^ { * } y ^ { \delta }$ is given by 

$$
x _ {\alpha} ^ {\delta} = \sum_ {n} \frac {\sigma_ {n}}{\sigma_ {n} ^ {2} + \alpha} \left<   y ^ {\delta}, u _ {n} \right> v _ {n}.
$$

<sub>Proof.</sub> It holds that $x _ { \alpha } ^ { \delta } = P _ { \mathrm { k e r } ( A ) } x _ { \alpha } ^ { \delta } + P _ { \mathrm { k e r } ( A ) ^ { \perp } } x _ { \alpha } ^ { \delta } = P _ { \mathrm { k e r } ( A ) } x _ { \alpha } ^ { \delta } +$ $\textstyle \sum _ { n } \left. x _ { \alpha } ^ { \delta } , v _ { n } \right. v _ { n }$ . Since $\left( \sigma _ { n } ^ { 2 } , v _ { n } , v _ { n } \right)$ is the spectral decomposition of 

$A ^ { * } A$ we get $\begin{array} { r } { A ^ { * } A x _ { \alpha } ^ { \delta } = \sum _ { n } \sigma _ { n } ^ { 2 } \left. x _ { \alpha } ^ { \delta } , v _ { n } \right. v _ { n } } \end{array}$ . Also $\begin{array} { r } { A ^ { * } y ^ { \delta } = \sum _ { n } \sigma _ { n } \left. y ^ { \delta } , u _ { n } \right. v _ { n } } \end{array}$ Thus, the regularized normal equation is 

$$
\begin{array}{r l} \sum_ {n} \sigma_ {n} ^ {2} \left\langle x _ {\alpha} ^ {\delta}, v _ {n} \right\rangle v _ {n} + \alpha \left(P _ {\ker (A)} (x _ {\alpha} ^ {\delta}) + \sum_ {n} \left\langle x _ {\alpha} ^ {\delta}, v _ {n} \right\rangle v _ {n}\right) & = A ^ {*} y ^ {\delta} \\ & = \sum_ {n} \sigma_ {n} \left\langle y ^ {\delta}, u _ {n} \right\rangle v _ {n}. \end{array}
$$

We see that necessarily $P _ { \mathrm { k e r } ( A ) } \big ( x _ { \alpha } ^ { \delta } \big ) = 0$ and that 

$$
\sum_ {n} (\sigma_ {n} ^ {2} + \alpha) \left<   x _ {\alpha} ^ {\delta}, v _ {n} \right> v _ {n} = \sum_ {n} \sigma_ {n} \left<   y ^ {\delta}, u _ {n} \right> v _ {n}.
$$

Comparing coeficients shows that $\begin{array} { r } { \left. x _ { \alpha } ^ { \delta } , v _ { n } \right. = \frac { \sigma _ { n } } { \sigma _ { n } ^ { 2 } + \alpha } \left. y ^ { \delta } , u _ { n } \right. } \end{array}$ which shows the claim. □ 

The representation of $\mathbf { \dot { \psi } } _ { x _ { \alpha } } ^ { \delta }$ from Theorem $6 . 3$ via the singular value decomposition is called <sub>spectral</sub> <sub>representation</sub>. We use it to prove the following result on regularization: 

Theorem ${ \bf 6 . 4 }$ (Tikhonov with a-priori parameter choice)<sub>.</sub> <sub>For</sub> <sub>an</sub> a-priori parameter choice $\alpha ( \delta )$ that fulfills 

$$
\alpha (\delta) \rightarrow 0 \quad \frac {\delta^ {2}}{\alpha (\delta)} \rightarrow 0 \quad f o r \quad \delta \rightarrow 0
$$

it holds that Tikhonov regularization is a convergent regularization method, i.e. it holds that $x _ { \alpha } ^ { \delta } : = ( \overline { { { A } } } { } ^ { * } A + \alpha \operatorname { i d } ) ^ { - 1 } A ^ { * } y ^ { \delta } \overset { \smile } { \to } x ^ { \dagger } : = A ^ { \dagger } y ^ { \dagger }$ whenever $\lVert y ^ { \delta } - y ^ { \dagger } \rVert \leq \delta$ and $\delta  0 .$ 

Proof. <sup>We</sup> <sup>set</sup> $x _ { \alpha } = ( A ^ { * } A + \alpha \operatorname { i d } ) ^ { - 1 } A ^ { * } y ^ { \dagger }$ and decompose 

$$
\begin{array}{l} x _ {\alpha} ^ {\delta} - x ^ {\dagger} = x _ {\alpha} ^ {\delta} - x _ {\alpha} + x _ {\alpha} - x ^ {\dagger} \\ \qquad = \underbrace {(A ^ {*} A + \alpha \operatorname{id}) ^ {- 1} A ^ {*} (y ^ {\delta} - y ^ {\dagger})} _ {\text {data error}} + \underbrace {(A ^ {*} A + \alpha \operatorname{id}) ^ {- 1} A ^ {*} y ^ {\dagger} - A ^ {\dagger} y ^ {\dagger}} _ {\text {approx. error}}. \end{array}
$$

The data error fulfills 

$$
(A ^ {*} A + \alpha \mathrm{id}) ^ {- 1} A ^ {*} (y ^ {\delta} - y ^ {\dagger}) = \sum_ {n} \frac {\sigma_ {n}}{\sigma_ {n} ^ {2} + \alpha} \left<   y ^ {\delta} - y ^ {\dagger}, u _ {n} \right> v _ {n},
$$

and hence, its norm is 

$$
\begin{array}{c} \| (A ^ {*} A + \alpha   \mathrm{id}) ^ {- 1} A ^ {*} (y ^ {\delta} - y ^ {\dagger}) \| _ {Y} ^ {2} = \sum_ {n} \left(\frac {\sigma_ {n}}{\sigma_ {n} ^ {2} + \alpha}\right) ^ {2} | \big \langle y ^ {\delta} - y ^ {\dagger}, u _ {n} \big \rangle | ^ {2} \\ \leq \left(\sup _ {0 \leq \sigma \leq \| A \|} \frac {\sigma}{\sigma^ {2} + \alpha}\right) ^ {2} \| y ^ {\delta} - y ^ {\dagger} \| _ {Y} ^ {2} \end{array}
$$

For the approximation error and we use that $x ^ { \dagger } = A ^ { \dagger } y ^ { \dagger }$ implies 

$\begin{array} { r } { \left. { x ^ { \dagger } , v _ { n } } \right. \stackrel { \textstyle \sim } { = } \frac { 1 } { \sigma _ { n } } \left. { y ^ { \delta } , u _ { n } } \right. } \end{array}$ to get 

$$
\begin{array}{r}(A ^ {*} A + \alpha \mathrm{id}) ^ {- 1} A ^ {*} y ^ {\dagger} - A ^ {\dagger} y ^ {\dagger} = \sum_ {n} \frac {\sigma_ {n}}{\sigma_ {n} ^ {2} + \alpha} \left<   y ^ {\dagger}, u _ {n} \right> v _ {n} - \sum_ {n} \frac {1}{\sigma_ {n}} \left<   y ^ {\dagger}, u _ {n} \right> v _ {n}\\= \sum_ {n} \left(\frac {\sigma_ {n} ^ {2}}{\sigma_ {n} ^ {2} + \alpha} - 1\right) \left<   x ^ {\dagger}, v _ {n} \right> v _ {n}.\end{array}
$$

Together we arrive at the error estimate 

$$
\| x _ {\alpha} ^ {\delta} - x ^ {\dagger} \| _ {X} \leq \left(\sup _ {0 \leq \sigma \leq \| A \|} \frac {\sigma}{\sigma^ {2} + \alpha}\right) \delta + \sqrt {\sum_ {n} \left(\frac {\sigma_ {n} ^ {2}}{\sigma_ {n} ^ {2} + \alpha} - 1\right) ^ {2} | \langle x ^ {\dagger} , v _ {n} \rangle | ^ {2}}.\tag{*}
$$

We estimate the supremum by 

$$
\sup _ {0 \leq \sigma \leq \| A \|} \frac {\sigma}{\sigma^ {2} + \alpha} \leq \frac {1}{2 \sqrt {\alpha}}.
$$

We maximize over all $\sigma > 0 :$ The derivative of $\sigma / ( \sigma ^ { 2 } + \alpha )$ is $( ( \sigma ^ { 2 } + \alpha ) - 2 \sigma ^ { 2 } ) / ( \sigma ^ { 2 } + \alpha ) ^ { 2 }$ and hence, vanishes exactly at $\sigma = \sqrt { \alpha } . \mathsf { P l u g g i n g }$ this in gives the result. 

By assumption $\delta / \sqrt { \alpha } \to 0$ for $\delta  0$ , and thus, the first term on the right hand side of $( ^ { \star } )$ goes to zero for $\delta  0$ 

Now we consider the square of second term in $( ^ { \star } ) _ { i }$ , which we write as $\textstyle \sum _ { n } a _ { n } ( \alpha )$ with $\begin{array} { r } { a _ { n } \big ( \alpha \big ) = \big ( \frac { \sigma _ { n } ^ { 2 } } { \sigma _ { n } ^ { 2 } + \alpha } - 1 \big ) ^ { 2 } | \langle x ^ { \dagger } , v _ { n } \rangle | ^ { 2 } } \end{array}$ . It holds that $a _ { n } ( \alpha )  | \langle x ^ { \dagger } , v _ { n } \rangle | ^ { 2 }$ for $\alpha  0$ . We have the (very coarse) estimate $\left( \frac { \sigma _ { n } ^ { 2 } } { \sigma _ { n } ^ { 2 } + \alpha } - 1 \right) ^ { 2 } \leq 4$ and hence $\textstyle \sum _ { n } a _ { n } ( \alpha ) \leq 4 \| x \| _ { X ^ { ! } } ^ { 2 }$ ,and by the dominated convergence theorem (Theorem $3 { \cdot } 5 ) .$ , we get that the full sum $\textstyle \sum _ { n } a _ { n } ( \alpha ) \to 0$ for $\alpha  0 .$ . This proves the theorem. 

The previous theorem shows that Tikhonov is indeed a convergent regularization method. However, we did not get an explicit error estimate for the total error $\| x _ { \alpha } ^ { \delta } - x ^ { \dag } \| _ { X }$ . While we could bound the data error by 

$$
\left\| x _ {\alpha} ^ {\delta} - x _ {\alpha} \right\| _ {X} \leq \frac {\delta}{\sqrt {\alpha}},
$$

we did not get an efective bound on the approximation error $\| x _ { \alpha } - x ^ { \dagger } \| _ { X } .$ . This is a general fact: 

Theorem ${ \bf 6 . 5 }$ (No general worst case error bound for ill-posed problems)<sub>. Let</sub> $\left( R _ { \alpha } , \alpha ( \delta , y ^ { \delta } ) \right)$ be a convergent regularization method for $A ^ { \dagger }$ . If there exists a function $\psi : ] 0 , \infty [  ] 0 , \infty [$ [ with $\psi ( \delta ) \stackrel { \delta \to 0 } { \longrightarrow } ($ 0 such that for all $y ^ { \dag } \in \breve { D } ( A ^ { \dag } )$ 

$$
\sup \left\{\| R _ {\alpha (\delta , y ^ {\delta})} y ^ {\delta} - A ^ {\dagger} y ^ {\dagger} \| _ {X} \mid y ^ {\dagger} \in D (A ^ {\dagger}), y ^ {\delta} \in Y, w i t h \| y ^ {\dagger} - y ^ {\delta} \| _ {Y} \leq \delta \right\} \leq \psi (\delta)
$$

then $A ^ { \dagger }$ is bounded. 

Proof. <sup>Let</sup> $y ^ { \dagger } , y _ { n } \in D ( A ^ { \dagger } )$ with $\| y ^ { \dagger } - y _ { n } \| _ { Y } = \delta _ { n } \overset { n \to \infty } { \longrightarrow } 0$ . Then we have 

The significance of this theorem is as follows: If $A ^ { \dagger }$ is bounded, we can get a nice error bound $\| A ^ { \dagger } y ^ { \dot { \delta } } - x ^ { \dagger } \| _ { X } \leq$ $\| A ^ { \dag } \| \delta$ and hence, the problem is not ill-posed. 

$$
\| A ^ {\dagger} y _ {n} - A ^ {\dagger} y ^ {\dagger} \| _ {Y} \leq \| A ^ {\dagger} y _ {n} - R _ {\alpha (\delta , y _ {n})} y _ {n} \| + \| R _ {\alpha (\delta , y _ {n})} y _ {n} - A ^ {\dagger} y \|.
$$

By our assumption, we have that both terms on the right are bounded by $\psi ( \delta _ { n } )$ , i.e. 

$$
\| A ^ {\dagger} y _ {n} - A ^ {\dagger} y ^ {\dagger} \| _ {Y} \leq 2 \psi (\delta_ {n}) \stackrel {n \to \infty} {\longrightarrow} 0.
$$

But this means that $A ^ { \dagger }$ is continuous at $y ^ { \dagger }$ and since $A ^ { \dagger }$ is linear, this shown continuity everywhere. □ 

Here is an example of Tikhonov regularization in practice: 

```python
Here is an example of Tikhonov regularization in practice
import numpy as np
import matplotlib.pyplot as plt

# problem size and matrix
n = 100
A = np.tril(np.ones((n,n))) / n

# discretized interval
t = np.linspace(0,1,n)

# true solution
xdag = np.maximum(1-2*t,0)
# noise free data
ydag = A@xdag

# noisy data
eta = np.random.randn(n);
eta /= np.linalg.norm(eta)

# noise level
delta = 0.05
ydelta = ydag + delta*eta

# naive reconstruction
x = np.linalg.solve(A,ydelta)

fig, axes = plt.subplots(2,2)
axs[0,0].plot(t,xdag)
axs[0,0].set_title('true solution')
axs[0,1].plot(t,ydag)
axs[0,1].set_title('true data')
axs[1,0].plot(t,x)
axs[1,0].set_title('naive reconstruction')
axs[1,1].plot(t,ydelta)
axs[1,1].set_title('noisy data')
plt.show() 
```

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/8cd9d174662f7ad04589846f20f0978e7910faf3f9a1810dc4cf9f48c7cbfb33.jpg)


```python
# reconstruct with Tikhonov
# regularization parameter
alpha = 0.01
# compute reconstruction
xalphadelta = np.linalg.solve(A.T@A + alpha*np.identity(n), A.T@ydelta)
plt.plot(t, xalphadelta, label='xalphadelta')
plt.plot(t, xdag, label='xdagger')
plt.legend()
plt.show() 
```

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/a7e1fa7971c9b53752fa8848c5ec8e616be7979d536ee45af7b94ee65dd5d00a.jpg)


```python
# Plot for N different errors to illustrate error decomposition
N = 30
alphas = np.logspace(0, -6, N)
totalError = np.zeros(N)
approximationError = np.zeros(N)
dataError = np.zeros(N)
# reconstruct and compute errors
for k in range(N):
    alpha = alphas[k]
    xalphadelta = np.linalg.solve(A.T@A + alpha*np.identity(n), A.T@ydelta)
    xalpha = np.linalg.solve(A.T@A + alpha*np.identity(n), A.T@ydag)
    totalError[k] = np.linalg.norm(xalphadelta-xdag)
    approximationError[k] = np.linalg.norm(xalpha-xdag)
    dataError[k] = np.linalg.norm(xalphadelta-xalpha)

# Show errors is loglog-plot
plt.loglog(alphas, totalError, label='total error')
plt.loglog(alphas, approximationError, label='approximation error')
plt.loglog(alphas, dataError, label='data error')
plt.legend()
plt.show() 
```

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/6e7e908896186e3fd0d6988b777c3fb0139aa5ef9dd4e53df958c4078e4b9d10.jpg)


## 7 Spectral regularization

We have analyzed the regularization properties of the truncated singular value decomposition and Tikhonov regularization in Example $5 { \cdot } 7$ and Theorem $6 . 4 \cdot$ . If you inspect the arguments, you’ll note that they are actually almost the same in both cases. In this section we will start to derive a general theory for linear regularization. This theory will contain the truncated SVD as well as Tikhonov regularization as special cases. 

Recall that the truncated SVD is 

$$
R _ {\alpha} y = \sum_ {\sigma_ {n} > \alpha} \frac {1}{\sigma_ {n}} \left<   y, u _ {n} \right> v _ {n}
$$

while we could express Tikhonov regularization as 

$$
R _ {\alpha} y = \sum_ {n} \frac {\sigma_ {n}}{\sigma_ {n} ^ {2} + \alpha} \left\langle y, u _ {n} \right\rangle v _ {n}.
$$

Both methods can be written in the following form: 

$$
R _ {\alpha} y = \sum_ {n} \varphi_ {\alpha} (\sigma_ {n} ^ {2}) \sigma_ {n} \left<   y, u _ {n} \right> v _ {n}\tag{R}
$$

with some function $\varphi _ { \alpha } .$ : 

For $\begin{array} { r } { \varphi _ { \alpha } ( \lambda ) = \frac { 1 } { \lambda + \alpha } } \end{array}$ we get 

$$
R _ {\alpha} y = \sum_ {n} \frac {1}{\sigma_ {n} ^ {2} + \alpha} \sigma_ {n} \left<   y, u _ {n} \right> v _ {n},
$$

i.e. exactly Tikhonov regularization. If we set 

$$
\varphi_ {\alpha} (\lambda) = \left\{ \begin{array}{l l} \frac {1}{\lambda} & : \quad \lambda \geq \alpha \\ 0 & : \quad \text { else }, \end{array} \right.
$$

we get 

$$
R _ {\alpha} y = \sum_ {\sigma_ {n} ^ {2} \geq \alpha} \frac {1}{\sigma_ {n} ^ {2}} \sigma_ {n} \left\langle y, u _ {n} \right\rangle v _ {n} = \sum_ {\sigma_ {n} > \sqrt {\alpha}} \frac {1}{\sigma_ {n}} \left\langle y, u _ {n} \right\rangle v _ {n},
$$

which is (up to a diferent scaling of the regularization parameter) the truncated SVD from Example 5.7. 

Remark $7 . 1$ <sub>.</sub> Note that we could have written $\begin{array} { r } { R _ { \alpha } y = \sum _ { n } f _ { \alpha } ( \sigma _ { n } ) \langle y , u _ { n } \rangle v _ { n } } \end{array}$ with some function $f _ { \alpha }$ as well. However,there is a reason why we did chose this slightly complicated form: Regularization methods approximate the minimum norm solution of the normal equation $K ^ { * } K x = K ^ { * } y .$ , i.e. $x = ( K ^ { * } K ) ^ { \dagger } K ^ { * } y .$ . If we express everything with the SVD of <sub>K</sub> we first get that $\begin{array} { r } { ( \dot { K } ^ { * } K ) ^ { \dagger } z = \bar { \sum _ { n } } \sigma _ { n } ^ { - 2 } \left. \bar { z _ { \prime } } v _ { n } \right. \bar { v _ { n } } } \end{array}$ and hence, for $z = K ^ { * } y$ 

$$
\begin{array}{r}x = \sum_ {n} \sigma_ {n} ^ {- 2} \left<   K ^ {*} y, v _ {n} \right> v _ {n} = \sum_ {n} \sigma_ {n} ^ {- 2} \left<   y, K v _ {n} \right> v _ {n}\\= \sum_ {n} \sigma_ {n} ^ {- 2} \left<   y, \sigma_ {n} u _ {n} \right> v _ {n} = \sum_ {n} \sigma_ {n} ^ {- 2} \sigma_ {n} \left<   y, u _ {n} \right> v _ {n}.\end{array}
$$

To mimic this formula, we express regularization methods as 

$$
R _ {\alpha} y = \sum_ {n} \varphi_ {\alpha} (\sigma_ {n} ^ {2}) \sigma_ {n} \left<   y, u _ {n} \right> v _ {n}.
$$

and need that $\varphi _ { \alpha } ( \lambda ) \approx 1 / \lambda$ for $R _ { \alpha }$ being close to $A ^ { \dagger }$ 

We will use the following <sub>functional</sub> <sub>calculus</sub> for compact operators: If <sub>K</sub> is compact with singular system $\left( \sigma _ { n } , u _ { n } , v _ { n } \right)$ and $f :$ $[ 0 , \| K \| ^ { 2 } ] \to$ <sub>R</sub> is piecewise continuous and bounded, we define another operator $f ( K ^ { * } K ) : X \to Y$ by 

$$
f (K ^ {*} K) x := \sum_ {n} f (\sigma_ {n} ^ {2}) \left\langle x, v _ {n} \right\rangle v _ {n} + f (0) P _ {\ker (K)} x.
$$

We observe that the series always converges (since $f$ is only evaluated on the bounded interval $\lbrack \bar { 0 } , \| K \| ^ { 2 } ] )$ and we also get that 

The additional term $P _ { \mathrm { k e r } ( K ) } x$ takes into account that $f ( 0 ) \neq 0$ and makes the identity id $= f ( K ^ { * } K )$ for $f \equiv 1$ correct. 

$$
\| f (K ^ {*} K) \| \leq \sup _ {n} | f (\sigma_ {n} ^ {2}) | + f (0) \leq 2 \sup _ {\lambda \in [ 0, \| K \| ^ {2} ]} | f (\lambda) | <   \infty
$$

which shows that $f ( K ^ { * } K ) \in L ( X , X )$ 

With functional calculus we can write 

$$
R _ {\alpha} y = \sum_ {n} \varphi_ {\alpha} (\sigma_ {n} ^ {2}) \sigma_ {n} \left\langle y, u _ {n} \right\rangle v _ {n} = \varphi_ {\alpha} (K ^ {*} K) K ^ {*} y.
$$

Example $7 . 2$ (Absolute value of a compact operator)<sub>.</sub> For $f ( t ) = t$ we get that $f ( K ^ { * } K ) = K ^ { * } K$ and for ${ \bar { f } } ( t ) = { \sqrt { t } }$ we define $| K | : =$ $f ( K ^ { * } K )$ . It holds that 

$$
| K | x = \sum_ {n} \sigma_ {n} \left<   x, v _ {n} \right> v _ {n}.
$$

We state some properties of the absolute value of an operator as we will use it later when we derive convergence rates in abstract smoothness spaces in Sections 9 and 10. 

<sub>Lemma</sub> <sub>7.3</sub> (Properties of functional calculus)<sub>. Let</sub> $K \in K ( X , Y )$ 

(i) For $s , r > 0$ it holds that $| K | ^ { r + s } = | K | ^ { r } | K | ^ { s }$ 

(ii) For all $r > 0$ the operator $| K | ^ { r }$ is self-adjoint. 

(iii) For all $x \in X$ it holds that $\| | K | x \| _ { Y } = \| K x \| _ { y }$ 

(iv) It holds that $\mathbf { r g } ( | K | ) = \mathbf { r g } ( K ^ { * } )$ 

<sub>Proof.</sub> (i) This is a direct computation (using that $v _ { n }$ is orthonormal) 

$$
\begin{array}{c} | K | ^ {r + s} x = \sum_ {n} \sigma_ {n} ^ {r + s} \left\langle x, v _ {n} \right\rangle v _ {n} = \sum_ {n} \sigma_ {n} ^ {r} \left\langle \sum_ {m} \sigma_ {m} ^ {s} \left\langle x, v _ {m} \right\rangle v _ {m}, v _ {n} \right\rangle v _ {n} \\ = \sum_ {n} \sigma_ {n} ^ {r} \left\langle | K | ^ {s} x, v _ {n} \right\rangle v _ {n}. \end{array}
$$

□ 

(ii) Again a direct computation 

$$
\langle | K | ^ {r} x, z \rangle = \sum_ {n} \sigma_ {n} ^ {r} \left\langle x, v _ {n} \right\rangle \left\langle v _ {n}, z \right\rangle = \left\langle x, | K | ^ {r} z \right\rangle .
$$

(iii) Using the first two points we get 

$$
\| | K | x \| _ {X} ^ {2} = \langle | K | x, | K | x \rangle = \left\langle | K | ^ {2} x, x \right\rangle = \left\langle K ^ {*} K x, x \right\rangle = \left\langle K x, K x \right\rangle = \| K x \| _ {Y} ^ {2}.
$$

(iv) If $\left( \sigma _ { n } , u _ { n } , v _ { n } \right)$ is the singular system of ${ \mathrm { ~ \cal ~ K ~ } } ,$ then $K ^ { * }$ has the singular system $\left( \sigma _ { n } , v _ { n } , u _ { n } \right)$ and $| K |$ has the singular system $\left( \sigma _ { n } , v _ { n } , v _ { n } \right)$ . Now note that <sub>x</sub> $\in { \mathrm { r g } } ( K ^ { * } )$ exactly if <sub>Kx</sub> $\in { \bf r g } ( K K ^ { * } )$ and <sub>x⊥ ker(K)</sub>. The Picard condition (Theorem $_ { 4 \cdot 8 ) }$ for $K x \in$ $\mathrm { r g } ( K K ^ { * } )$ is 

$$
\infty > \sum_ {n} \sigma_ {n} ^ {- 4} | \langle K x, u _ {n} \rangle | ^ {2} = \sum_ {n} \sigma_ {n} ^ {- 4} | \langle x, K ^ {*} u _ {n} \rangle | ^ {2} = \sum_ {n} \sigma_ {n} ^ {- 2} | \langle x, v _ {n} \rangle | ^ {2}
$$

which is exactly the Picard condition for $x \in { \mathrm { r g } } ( | K | )$ (for which $x \bot \ker ( K )$ is necessary anyway). 

We will investigate regularization methods of the form (R). The following definition well be useful, as we will see: 

Definition $\mathbf { 7 . 4 }$ (Regularizing filter)<sub>.</sub> Let $K \in K ( X , Y )$ with $\kappa =$ $\| K \| ^ { 2 }$ and SVD $\left( \sigma _ { n } , u _ { n } , v _ { n } \right)$ . A family $\varphi _ { \alpha } : [ 0 , \kappa ] \to \mathbb { R }$ of piecewise continuous and bounded functions is called <sub>regularizing</sub> <sub>filter</sub> if it fulfills 

(i) For all $\lambda \in ] 0 , \kappa ]$ it holds that 

$$
\varphi_ {\alpha} (\lambda) \xrightarrow {\alpha \to 0} \frac {1}{\lambda}.
$$

(ii) There exists $C _ { \varphi } > 0$ such that for all $\lambda \in ] 0 , \kappa ]$ and $\alpha > 0$ it holds that 

$$
\lambda | \varphi_ {\alpha} (\lambda) | \leq C _ {\varphi}.
$$

Now we aim to prove that regularizing filters give rise to convergent regularization methods. First we collect three useful facts in a lemma: 

Lemma $^ { 7 . 5 }$ (Fundamental lemma of regularization theory)<sub>.</sub> ${ \cal I } f \varphi _ { \alpha }$ is a regularizing filter and $R _ { \alpha } = \varphi _ { \alpha } ( K ^ { * } K ) K ^ { * }$ we have 

(1) 

$$
\left\| K R _ {\alpha} \right\| \leq C _ {\varphi}\tag{2}
$$

$$
\| R _ {\alpha} \| \leq \sqrt {C _ {\varphi}} \sup _ {\lambda \in ] 0, \| K \| ^ {2} ]} \sqrt {| \varphi_ {\alpha} (\lambda) |}\tag{3}
$$

$$
K ^ {\dagger} y - R _ {\alpha} y = \sum_ {n} (1 - \sigma_ {n} ^ {2} \varphi_ {\alpha} (\sigma_ {n}) ^ {2}) \left\langle x ^ {\dagger}, v _ {n} \right\rangle v _ {n} \quad f o r y \in D (K ^ {\dagger}) a n d x ^ {\dagger} = K ^ {\dagger} y.
$$

<sub>Proof.</sub> We first compute 

$$
K R _ {\alpha} y = K \varphi_ {\alpha} (K ^ {*} K) K ^ {*} y = \sum_ {n} \varphi_ {\alpha} (\sigma_ {n} ^ {2}) \sigma_ {n} \left\langle y, u _ {n} \right\rangle K v _ {n} = \sum_ {n} \varphi_ {\alpha} (\sigma_ {n} ^ {2}) \sigma_ {n} ^ {2} \left\langle y, u _ {n} \right\rangle u _ {n}
$$

and then get that 

$$
\begin{array}{c} \| K R _ {\alpha} y \| _ {Y} ^ {2} = \sum_ {n} | \varphi_ {\alpha} (\sigma_ {n} ^ {2}) \sigma_ {n} ^ {2} \langle y, u _ {n} \rangle | ^ {2} \\ \leq \sup _ {n} | \varphi_ {\alpha} (\sigma_ {n} ^ {2}) \sigma_ {n} ^ {2} | ^ {2} \| y \| _ {Y} ^ {2} \end{array}
$$

which, by definition of the constant $C _ { \varphi }$ , implies the claim <sub>(1)</sub>. For the claim <sub>(2)</sub> compute 

$$
\begin{array}{l} \| R _ {\alpha} y \| _ {X} ^ {2} = \langle R _ {\alpha} y, R _ {\alpha} y \rangle = \sum_ {n} \varphi_ {\alpha} (\sigma_ {n} ^ {2}) \sigma_ {n} \left\langle y, u _ {n} \right\rangle \left\langle R _ {\alpha} y, v _ {n} \right\rangle \\ = \sum_ {n} \varphi_ {\alpha} (\sigma_ {n} ^ {2}) \left\langle y, u _ {n} \right\rangle \left\langle R _ {\alpha} y, K ^ {*} u _ {n} \right\rangle \\ = \sum_ {n} \varphi_ {\alpha} (\sigma_ {n} ^ {2}) \left\langle y, u _ {n} \right\rangle \left\langle K R _ {\alpha} y, u _ {n} \right\rangle \\ \leq \sup _ {n} | \varphi_ {\alpha} (\sigma_ {n} ^ {2}) | \sum_ {n} \left\langle y, u _ {n} \right\rangle \left\langle K R _ {\alpha} y, u _ {n} \right\rangle \\ \leq \sup _ {n} | \varphi_ {\alpha} (\sigma_ {n} ^ {2}) | \left(\sum_ {n} \left\langle y, u _ {n} \right\rangle^ {2}\right) ^ {1 / 2} \left(\sum_ {n} \left\langle K R _ {\alpha} y, u _ {n} \right\rangle\right) ^ {1 / 2} \\ (b y C a u c h y - S c h w a r z) \\ \leq \sup _ {n} | \varphi_ {\alpha} (\sigma_ {n} ^ {2}) | \| y \| \| K R _ {\alpha} y \| \\ \leq \sup _ {n} | \varphi_ {\alpha} (\sigma_ {n} ^ {2}) | C _ {\varphi} \| y \| _ {Y} ^ {2} \quad (b y c l a i m (1)) \end{array}
$$

which proves the claim. Finally, for claim <sub>(3)</sub> we note that if $\boldsymbol { x } ^ { \dagger } =$ $K ^ { \dagger } y ,$ , then $K ^ { * } K x ^ { \dagger } = K ^ { * } y$ and thus 

$$
R _ {\alpha} y = \varphi_ {\alpha} (K ^ {*} K) K ^ {*} y = \varphi_ {\alpha} (K ^ {*} K) K ^ {*} K x
$$

and we get 

$$
K ^ {\dagger} y - R _ {\alpha} y = (\mathrm{id} - \varphi_ {\alpha} (K ^ {*} K) K ^ {*} K) x ^ {\dagger} = \sum_ {n} (1 - \sigma_ {n} ^ {2} \varphi_ {\alpha} (\sigma_ {n} ^ {2})) \left\langle x ^ {\dagger}, v _ {n} \right\rangle v _ {n}.
$$

<sub>Theorem</sub> <sub>7.6</sub> (Regularization with regularizing filters)<sub>. Let</sub> $\varphi _ { \alpha }$ be a regularizing filter and $R _ { \alpha } = \varphi _ { \alpha } ( K ^ { * } K ) K ^ { * }$ . Then it holds for all $y \in$ $D ( \bar { A } ^ { \dag } )$ that 

$$
R _ {\alpha} y \xrightarrow {\alpha \to 0} K ^ {\dagger} y.
$$

<sub>Proof.</sub> By Lemma $\ 7 . 5 \left( 3 \right)$ we have 

$$
\| K ^ {\dagger} y - R _ {\alpha} y \| _ {X} ^ {2} = \sum_ {n} (1 - \sigma_ {n} ^ {2} \varphi_ {\alpha} (\sigma_ {n} ^ {2})) ^ {2} | \left<   x ^ {\dagger}, v _ {n} \right> | ^ {2}.
$$

Since $\varphi _ { \alpha } ( \lambda ) \to 1 / \lambda$ for $\alpha  0$ we get $( 1 - \sigma _ { n } ^ { 2 } \varphi _ { \alpha } ( \sigma _ { n } ^ { 2 } ) ) \to 0$ for $\alpha $ <sub>0</sub>. Moreover, $| 1 - \sigma _ { n } ^ { 2 } \varphi _ { \alpha } ( \sigma _ { n } ^ { 2 } ) | \leq 1 + C _ { \varphi }$ and hence, the convergence $R _ { \alpha } y  K ^ { \dagger } y$ follows from the dominated convergence theorem (Theorem 3.5). □ 

Next we will show that there is a general strategy to construct a parameter choice rule that turns regularizing filters into convergent regularization methods. 

## 8 Parameter choice and error estimates

We will now investigate the problem of a-priori parameter choice for general spectral regularization methods of the type $R _ { \alpha } \ =$ $\varphi _ { \alpha } ( K ^ { * } K ) K ^ { * }$ for a regularizing filter $\varphi _ { \alpha }$ 

Theorem 8.1. Let $K ^ { \dagger }$ be non-continuous and $R _ { \alpha }$ be a regularization of $K ^ { \dagger }$ . Then it holds: An a priori parameter choice $\alpha ( \delta ) \ f u l f i l s$ that $\| R _ { \alpha ( \delta ) } y ^ { \delta } - x ^ { \dagger } \| _ { X } \to 0 f o r \delta \to 0$ exactly if 

(i) $\alpha ( \delta )  0 f o r \delta  0 ,$ , and 

(ii) δ sup<sub>0< ≤∥K∥2</sub> <sup>np</sup>|φ<sub>α</sub>(λ)|<sup>o</sup> → 0 for δ → 0. 

<sub>Proof.</sub> We start with our standard error decomposition 

$$
\| R _ {\alpha} y ^ {\delta} - x ^ {\dagger} \| _ {X} \leq \| R _ {\alpha} \| \delta + \| R _ {\alpha} y ^ {\dagger} - K ^ {\dagger} y ^ {\dagger} \| _ {X}.
$$

By Theorem $7 . 6$ and $\alpha ( \delta )  0$ we get that the second term on right hand side goes to zero. By Lemma $7 . 5 \left( 2 \right)$ we have that $\| R _ { \alpha } \| \leq$ $\begin{array} { r } { \operatorname* { s u p } _ { \lambda \in ] 0 , \| K \| ^ { 2 } ] } \left\{ \sqrt { | \varphi _ { \alpha } ( \lambda ) | } \right\} } \end{array}$ and hence, the first term goes to zero as well. 

Conversely, assume that either (i) or (ii) does not hold. Let’s start with the case where (i) does not hold. Then $R _ { \alpha ( \delta ) }$ does not converge to $K ^ { \dagger }$ pointwise. Hence, we can even set $y ^ { \delta } = \overset { \cdot } { y } \in D ( K ^ { \dagger } )$ 1 $x ^ { \dagger } = \bar { K ^ { \dagger } y }$ and get that $\| R _ { \alpha ( \delta ) } y ^ { \delta } - x ^ { \dagger } \| _ { X } = \| R _ { \alpha ( \delta ) } \dot { y } - \dot { K ^ { \dagger } } y \| \neq 0 .$ 

If (i) is fulfilled, but (ii) not, there exists $\delta _ { n }$ with $\delta _ { n } \stackrel { n  \infty } { \longrightarrow } 0$ such that $\delta _ { n } \| R _ { \alpha ( \delta _ { n } ) } \| > \epsilon$ for some $\epsilon .$ . Hence, there exists a sequence $z _ { n } \in Y$ with $\| z _ { n } \| _ { Y } = 1$ and $\delta _ { n } \| R _ { \alpha ( \delta _ { n } ) } z _ { n } \| _ { X } > \epsilon$ . Now let $y \in D ( K ^ { \dagger } )$ and set $y _ { n } : = y + \delta _ { n } z _ { n }$ . Then $\| y - y _ { n } \| = \delta ,$ , but 

$$
R _ {\alpha (\delta_ {n})} y _ {n} - K ^ {\dagger} y = (R _ {\alpha (\delta_ {n})} y - K ^ {\dagger} y) + \delta_ {n} R _ {\alpha (\delta_ {n})} z _ {n} \not \to 0.
$$

Now we aim for more sophisticated error estimates. What is needed, is a better estimate of the approximation error. As we have seen in Theorem $6 . 5 \mathrm { { : } }$ , this is not possible without additional assumptions. 

By Lemma $\ 7 . 5 \left( 3 \right)$ we have that= 

$$
K ^ {\dagger} y ^ {\dagger} - R _ {\alpha} y ^ {\dagger} = \sum_ {n} (1 - \sigma_ {n} ^ {2} \varphi_ {\alpha} (\sigma_ {n} ^ {2})) \left<   x ^ {\dagger}, v _ {n} \right> v _ {n}\tag{*}
$$

if $\mathbf { \dot { \boldsymbol { x } } } ^ { \dagger } = K ^ { \dagger } \boldsymbol { y } ^ { \dagger }$ . However, bounding this error just in terms of <sub>α</sub> is not possible, since the decay of the terms $\left. \bar { x } ^ { \dagger } , v _ { n } \right.$ is not known (the sequence has to be square summable, but that’s basically all we know). Here is a simple way to get a useful error bound: 

We assume that our true solution $x ^ { \dagger }$ is in $\mathrm { r g } ( K ^ { * } )$ 

How does that help? Well, in this case we have some $w ^ { \dagger }$ with $x ^ { \dagger } = K ^ { \ast } w ^ { \dagger }$ , we get from (*) 

$$
\begin{array}{r l}&{\| K ^ {\dagger} y ^ {\dagger} - R _ {\alpha} y ^ {\dagger} \| _ {X} ^ {2} = \sum_ {n} (1 - \sigma_ {n} ^ {2} \varphi_ {\alpha} (\sigma_ {n} ^ {2})) ^ {2} | \left<   x ^ {\dagger}, v _ {n} \right> | ^ {2}}\\&{\qquad = \sum_ {n} (1 - \sigma_ {n} ^ {2} \varphi_ {\alpha} (\sigma_ {n} ^ {2})) ^ {2} | \left<   K ^ {*} w ^ {\dagger}, v _ {n} \right> | ^ {2}}\\&{\qquad = \sum_ {n} (1 - \sigma_ {n} ^ {2} \varphi_ {\alpha} (\sigma_ {n} ^ {2})) ^ {2} | \left<   w ^ {\dagger}, K v _ {n} \right> | ^ {2}}\\&{\qquad = \sum_ {n} (1 - \sigma_ {n} ^ {2} \varphi_ {\alpha} (\sigma_ {n} ^ {2})) ^ {2} \sigma_ {n} ^ {2} | \left<   w ^ {\dagger}, v _ {n} \right> | ^ {2}.}\end{array}\tag{**}
$$

Now we may get an error bound, if we can control the coeficients $( 1 - \sigma _ { n } ^ { 2 } \varphi _ { \alpha } ( { \bar { \sigma } } _ { n } ^ { 2 } ) ) ^ { 2 } \sigma _ { n } ^ { 2 }$ . Expressed in the variable $\lambda = \sigma ^ { 2 }$ this says that we have to control the function $\lambda \mapsto ( 1 - \lambda \varphi _ { \alpha } ( \lambda ) ) \sqrt { \lambda }$ . Let us investigate the situation for the truncated SVD and Tikhonov regularization: 

Example $8 . 2 . \quad \mathrm { ~ \bf ~ 1 ~ }$ . For the truncated SVD we have $\varphi _ { \alpha } ( \lambda ) = 1 / \lambda$ for $\lambda \geq \alpha$ and <sub>= 0</sub> else. So we get 

$$
(1 - \lambda \varphi_ {\alpha} (\lambda)) \sqrt {\lambda} = \left\{ \begin{array}{l l} 0 & : \quad \lambda \geq \alpha \\ \sqrt {\lambda} & : \quad \lambda <   \alpha \end{array} \right.
$$

Note the we change the threshold in comparison to Example $5 { . } 7 { : }$ : There we took $1 / \sigma _ { n }$ if $\sigma _ { n } <$ α and here we use $1 / \sigma _ { n } = \sigma _ { n } / \sigma _ { n } ^ { 2 } { \mathrm { ~ i f ~ } } \sigma _ { n } ^ { 2 } > \alpha , { \mathrm { i . e . } }$ . for $\sigma _ { n } >$ ${ \sqrt { \alpha } } .$ 

and thus, $( 1 - \lambda \varphi _ { \alpha } ( \lambda ) ) \sqrt { \lambda } \leq \sqrt { \alpha }$ or, equivalently 

$$
(1 - \sigma^ {2} \varphi_ {\alpha} (\sigma^ {2})) ^ {2} \sigma^ {2} \leq \alpha .
$$

Using this in (**), we obtain for the approximation error 

$$
\begin{array}{c}\| K ^ {\dagger} y ^ {\dagger} - R _ {\alpha} y ^ {\dagger} \| _ {X} ^ {2} = \sum_ {n} (1 - \sigma_ {n} ^ {2} \varphi_ {\alpha} (\sigma_ {n} ^ {2})) ^ {2} \sigma_ {n} ^ {2} | \left<   w ^ {\dagger}, v _ {n} \right> | ^ {2}\\\leq \alpha \sum_ {n} | \left<   w ^ {\dagger}, v _ {n} \right> | ^ {2} \leq \alpha \| w \| _ {Y} ^ {2}\end{array}
$$

and thus 

$$
\| K ^ {\dagger} y ^ {\dagger} - R _ {\alpha} y ^ {\dagger} \| _ {X} \leq \sqrt {\alpha} \| w ^ {\dagger} \| _ {Y}.
$$

2. For Tikhonov regulrization we have $\varphi _ { \alpha } ( \lambda ) = 1 / ( \lambda + \alpha )$ and we get 

$$
(1 - \lambda \varphi_ {\alpha} (\lambda)) \sqrt {\lambda} = (1 - \frac {\lambda}{\lambda + \alpha}) \sqrt {\lambda} = \frac {\alpha \sqrt {\lambda}}{\lambda + \alpha}.
$$

We want to maximize the right hand side over $\lambda \geq 0$ . To this end we define $\begin{array} { r } { f ( \lambda ) = \frac { \sqrt { \lambda } } { \lambda + \alpha } . } \end{array}$ , calculate $\begin{array} { r } { f ^ { \prime } ( \lambda ) = \frac { 1 } { 2 } \frac { \alpha \lambda ^ { - 1 / 2 } - \lambda ^ { 1 / 2 } } { ( \lambda + \alpha ) ^ { 2 } } } \end{array}$ and see that $f ^ { \prime } ( \lambda ) = 0$ for $\lambda = \alpha$ . Hence, we get that $\begin{array} { r } { \frac { \alpha \sqrt { \lambda } } { \lambda + \alpha } \leq } \end{array}$ $\begin{array} { r } { \frac { \alpha \sqrt { \alpha } } { \alpha + \alpha } = \frac { 1 } { 2 } \sqrt { \alpha } . } \end{array}$ 

Again, using this in (**) gives $\begin{array} { r } { \| K ^ { \dagger } y ^ { \dagger } - R _ { \alpha } y ^ { \dagger } \| _ { X } ^ { 2 } \leq \frac { \alpha } { 4 } \| w ^ { \dagger } \| _ { Y } ^ { 2 } } \end{array}$ and thus 

$$
\| K ^ {\dagger} y ^ {\dagger} - R _ {\alpha} y ^ {\dagger} \| _ {X} \leq \frac {\sqrt {\alpha}}{2} \| w ^ {\dagger} \| _ {Y}.
$$

We conclude: If the unknown solution fulfills $x ^ { \dagger } \in \mathrm { r g } ( K ^ { * } )$ then the total error of both the truncated SVD and Tikhonov regularization can be estimated by 

$$
\left\| x _ {\alpha} ^ {\delta} - x ^ {\dagger} \right\| _ {X} \leq \delta \left\| R _ {\alpha} \right\| + \sqrt {\alpha} C
$$

for some constant <sub>C</sub> (independent of <sub>α</sub> and <sub>δ</sub>). This our first quantitative error bound, i.e. an upper bound for the total error that is explicit in $\delta$ and <sub>α</sub>. We also know that $\| R _ { \alpha } \| \le \mathsf { s u p } _ { \lambda } \sqrt { | \varphi _ { \alpha } ( \lambda ) | }$ and for both the truncated SVD and Tikhonov regularization we conclude by a simple calculation that $\| R _ { \alpha } \| \leq 1 / \sqrt { \alpha }$ . This gives the even more explicit error estimate 

$$
\left\| x _ {\alpha} ^ {\delta} - x ^ {\dagger} \right\| _ {X} \leq \frac {\delta}{\sqrt {\alpha}} + \sqrt {\alpha} C.
$$

Now we can even choose the regularization parameter $\alpha$ in an optimal way: We can minimize the right hand side over <sub>α</sub> and see that the minimum is attained for $\alpha ( \delta ) = \delta / C$ and this gives the error estimate 

$$
\| x _ {\alpha} ^ {\delta} - x ^ {\dagger} \| _ {X} \leq 2 \sqrt {C} \sqrt {\delta}.
$$

Even if we do not know the constant $C ,$ we could still set <sub>α</sub> proportional to $\delta ,$ i.e. $\alpha ( \delta ) = c \delta$ for some constant $c ,$ and obtain 

$$
\| x _ {\alpha} ^ {\delta} - x ^ {\dagger} \| _ {X} \leq \mathcal {O} (\delta^ {1 / 2}).
$$

Results of this form are called <sub>convergence</sub> <sub>rates</sub> of regularization methods. 

Assumption on the unknown solution $x ^ { \dagger }$ such as $x ^ { \dagger } \in { \bf r g } ( K ^ { * } )$ <sup>are</sup> <sup>called</sup> source conditions<sup>.</sup> △ 

We will come back to error estimates and convergence rates later. 

Now we briefly discuss the most popular a posteriori parameter choice rule: Recall that our standing assumption is that the true data $y ^ { \dagger }$ and our measured data $y ^ { \delta }$ always fulfill $\| y ^ { \dagger } - y ^ { \delta } \| _ { Y } \leq \delta$ The main idea now is to look at the residuum for a reconstruction $R _ { \alpha } y ^ { \delta }$ , i.e. to consider 

$$
\| K R _ {\alpha} y ^ {\delta} - y ^ {\delta} \| _ {Y}.
$$

The residuum for the minimum norm solution $x ^ { \dagger }$ fulfills $\parallel K x ^ { \dagger } -$ $y ^ { \delta } \| _ { Y } = \| y ^ { \dagger } - y ^ { \delta } \| _ { Y } \leq \delta _ { \operatorname* { m a x } }$ , thus it seems reasonable to not aim for a smaller residuum for any other reconstruction. This is the idea of the following: 

Morozov’s discrepancy principle: <sup>For</sup> <sup>some</sup> $\delta > 0$ and $y ^ { \delta }$ with $\| y ^ { \dagger } - y ^ { \delta } \| _ { Y } \leq \delta$ choose $\alpha = \alpha ( \delta , y ^ { \delta } )$ (as large as possible) such that 

$$
\| K R _ {\alpha} y ^ {\delta} - y ^ {\delta} \| _ {Y} \leq \tau \delta
$$

for some $\tau > 1$ 

We want to choose α as large as possible, to have the most stable reconstruction. 

Remark $8 . 3$ <sub>.</sub> This principle does not work without assumptions: For $y ^ { \dagger } \in \mathbf { r g } ( K ) ^ { \bot } \bar  \langle 0 \}$ and exact data $y ^ { \delta } = y ^ { \dagger } \left( \mathrm { i . e . } \delta = 0 \right)$ even the minimum norm solution $x ^ { \dagger }$ fulfills 

$$
\| K x ^ {\dagger} - y ^ {\delta} \| _ {Y} = \| K K ^ {\dagger} y ^ {\dagger} - y ^ {\dagger} \| _ {Y} = \| P _ {\overline {{\operatorname{rg} K}}} y ^ {\dagger} - y ^ {\dagger} \| = \| y ^ {\dagger} \| _ {Y} > 0 = \tau \delta .
$$

Therefore one usually assumes that the range <sub>rg</sub> <sub>K</sub> is dense in <sub>Y</sub> (since then <sub>rg</sub> $K ^ { \perp } = \{ 0 \} )$ . 

For a practical realization of Morozov’s discrepancy principle one usually defines a decreasing sequence $\alpha _ { n } \  \ 0$ , computes $R _ { \alpha _ { n } } y ^ { \delta }$ for $n = 1 , 2 , \ldots$ <sub>.</sub> and stops when $\| R _ { \alpha _ { n } } y ^ { \delta } - y ^ { \delta } \| _ { Y } \leq \bar { \tau } \delta$ for the first time. This always works if the range of $\cdot _ { K }$ is dense: 

Theorem 8.4. Let $R _ { \alpha } = \varphi _ { \alpha } ( K ^ { * } K ) K ^ { * }$ be a regularization of $\cdot K ^ { \dagger }$ , rg K dense in $Y , \alpha _ { n }$ be a strictly decreasing null sequence and $\tau > 1$ . Then it holds: For all $y ^ { \dag } \in D ( \bar { K } ^ { \dag } )$ , all $\delta > 0$ and $y ^ { \delta }$ with $\| y ^ { \dagger } - y ^ { \delta } \| _ { Y } \leq \delta$ there exists an $n ^ { * }$ such that for all $n < n ^ { * }$ 

$$
\| K R _ {\alpha_ {n ^ {*}}} y ^ {\delta} - y ^ {\delta} \| _ {Y} \leq \tau \delta <   \| K R _ {\alpha_ {n}} y ^ {\delta} - y ^ {\delta} \| _ {Y}.
$$

<sub>Proof.</sub> We study $\| K R _ { \alpha } y ^ { \delta } - y ^ { \delta } \|$ in dependence on <sub>α</sub>. Using the filter we get 

$$
\begin{array}{c} \| K R _ {\alpha} y ^ {\delta} - y ^ {\delta} \| ^ {2} = \sum_ {n} (1 - \sigma_ {n} ^ {2} \varphi_ {\alpha} (\sigma_ {n} ^ {2})) ^ {2} | \left\langle y ^ {\delta}, u _ {n} \right\rangle | ^ {2} + \| P _ {\mathrm{rg} (K) ^ {\perp}} (y ^ {\delta}) \| ^ {2} \\ = \sum_ {n} (1 - \sigma_ {n} ^ {2} \varphi_ {\alpha} (\sigma_ {n} ^ {2})) ^ {2} | \left\langle y ^ {\delta}, u _ {n} \right\rangle | ^ {2} \end{array}
$$

since $\mathrm { r g } ( K ) ^ { \perp } = \{ 0 \}$ . As we have seen in Theorem $7 . 6 ,$ , the right hand side goes to zero which proves the claim. □ 

We will show later that Morozov’s principle does indeed give a convergent regularization and now make a few remarks on <sub>heuristic</sub> rules: First and foremost, there a negative result, named <sub>Bakushinkii</sub> veto<sup>.</sup> 

If we do not assume that $\mathrm { r g } ( K )$ is dense, we only get that $\| K R _ { \alpha } y ^ { \delta } - \delta \| $ $\| P _ { \mathbf { r g } ( K ) ^ { \perp } } \bar { y ^ { \delta } } \| \leq \| y ^ { \delta } \|$ . Hence, we need that $\| y ^ { \delta } \| \leq \delta$ for Morozov’s discrepancy principle to work. This seems reasonable: There should be “more signal than noise”. 

Theorem $\mathbf { 8 . 5 }$ (Bakushinkii veto)<sub>. Let</sub> $R _ { \alpha }$ be a regularization for $K ^ { \dagger }$ . If there exists a heuristic choice rule $\alpha = \alpha ( y ^ { \delta } )$ such that $\left( R _ { \alpha } , \alpha \right)$ is a convergent regulariztion, then $K ^ { \dagger }$ is continuous. 

Proof. <sup>Let</sup> $\alpha : Y  ] 0 ,$ <sub>, ∞[</sub> by such a parameter choice. Now let $y \in$ $D ( K ^ { \dagger } )$ and consider $y _ { n } \in \bar { D } ( K ^ { \dagger } )$ with $y _ { n }  y .$ . But then (trivially) $\| y _ { n } - y _ { n } \| _ { Y } \leq \delta$ for every $\delta > 0$ and by definition of convergent regularization we get $\| K ^ { \dagger } y _ { n } - R _ { \alpha ( y _ { n } ) } y _ { n } \| _ { Y } = 0 , { \mathrm { i . e . } } R _ { \alpha ( y _ { n } ) } y _ { n } =$ $K ^ { \dagger } y _ { n }$ . Moreover, we have for $\delta _ { n } = \| y - y _ { n } \| _ { Y }$ (again by definition of convergent regulariztion) that 

$$
K ^ {\dagger} y _ {n} = R _ {\alpha (y _ {n})} y _ {n} \rightarrow K ^ {\dagger} y
$$

which proves continuity of $\cdot K ^ { \dagger }$ . 

This theorem shows that heuristic rules only exist for inverse problems that aren’t ill-posed. Despite this negative result there are several heuristic rules that work remarkably well in practice. This phenomenon is still not fully understood. One explanation could be that one usually faces a perturbation $y ^ { \delta } = y ^ { \dagger } + \eta$ by noise <sub>η</sub> in practice, but our theory uses general $\eta \in Y ,$ i.e. also perturbations which do not look like noise at all are considered. Some rules that work well in practice are the quasi-optimality principle, the Hanke-Raus rule, the L-curve method, and generalized cross validation. 

## 9 Convergence rates and smoothness spaces

In this section we will focus on the question on how to establish convergence rates, i.e. under what circumstances we can find a function $\psi : ] 0 , \infty [  ] 0 , \infty [$ with $\psi ( \delta )  0$ for $\delta  0$ such that 

$$
\| R _ {\alpha} y ^ {\delta} - K ^ {\dagger} y ^ {\dagger} \| _ {X} \leq \psi (\delta)
$$

for a-priori or a-posteriori parameter choice rules. Recall that by Theorem $6 . 5$ this can not hold without any further assumptions, i.e. it can’t be true if we consider $y ^ { \dagger }$ arbitrary in the range of <sub>K</sub> (or, equivalently, $x ^ { \dagger }$ arbitrary in $X )$ 

However, we have seen in Example $8 . 2$ that such a result can be achieved for the truncated SVD and Tikhonov regularization (for the a-priori choice rule $\alpha ( \delta ) \ : = \ : C \sqrt { \delta }$ and $\psi ( \delta ) = C \sqrt { \delta } )$ if we assume $x ^ { \dagger } \in \mathrm { r g } ( K ^ { * } ) \subsetneq X$ . This assumption is some kind of “abstract smoothness assumption”. The notion of smoothness that we will need will be formulated in terms of the operator <sub>K</sub> and may look confusing at first: 

Definition 9.1. <sup>Let</sup> $X , Y$ be Hilbert spaces and $K \in K ( X , Y )$ . For $\nu \geq 0$ we define the subspaces $X ^ { \nu } \subset X$ as 

$$
X ^ {\nu} := \operatorname{rg} (| K | ^ {\nu}) = \left\{| K | ^ {\nu} z \mid z \in \ker (K) ^ {\perp} \right\}.
$$

Some first observations: 

• For $\nu = 2 k$ we have that 

$$
X ^ {2 k} = \mathrm{rg} (| K | ^ {2 k}) = \mathrm{rg} (\sqrt {K ^ {*} K} ^ {2 k}) = \mathrm{rg} ((K ^ {*} K) ^ {k}).
$$

• If $\dot { \nu } > \mu ,$ , then $| K | ^ { \nu } = | K | ^ { \mu } | K | ^ { \nu - \mu }$ , i.e. $\mathbf { r g } ( | K | ^ { \nu } ) \subset \mathbf { r g } ( | K | ^ { \mu } )$ and thus $X ^ { \nu } \subset X ^ { \mu }$ , i.e. the spaces get smaller, the larger the <sub>ν</sub>. The boundary case is $\bar { X ^ { 0 } } \bar { = } \ker ( K ) ^ { \perp }$ 

• The spaces $X ^ { \nu }$ are characterized by summability assumptions of the coeficients in the singular basis: If there is <sub>z</sub> such that $x = | K | ^ { \nu } z$ we have by definition of $| K | ^ { \nu }$ that $\begin{array} { r } { x = \sum _ { n } \sigma _ { n } ^ { \nu } \left. z , v _ { n } \right. u _ { n } } \end{array}$ and hence 

$$
\sum_ {n} \sigma_ {n} ^ {- 2 \nu} | \langle x, v _ {n} \rangle | ^ {2} = \sum_ {n} \sigma_ {n} ^ {- 2 \nu} \sigma_ {n} ^ {2 \nu} | \langle z, v _ {n} \rangle | ^ {2} = \| z \| _ {X} ^ {2} <   \infty .
$$

In other words: For $x \in X ^ { \nu }$ we need that the sequence $\left. x , v _ { n } \right.$ decays fast enough such that the decay of $\vert \langle x , \bar { v } _ { n } \rangle \vert ^ { 2 }$ compensates the growth of $\cdot \sigma _ { n } ^ { - 2 \nu }$ 

The last observation motivates the following definition: 

Definition 9.2. <sup>On</sup> $X ^ { \nu }$ we define the norm 

$$
\| x \| _ {\nu} ^ {2} := \| z \| _ {X} ^ {2} = \sum_ {n} \sigma_ {n} ^ {- 2 \nu} | \langle x, v _ {n} \rangle | ^ {2}, \quad X ^ {\nu} = \{x \mid \| x \| _ {\nu} <   \infty \}
$$

We call these norms <sub>-norms</sub>. 

Note that the notation $X ^ { \nu }$ and $\lVert \cdot \rVert _ { \nu }$ do not include their dependency on $K .$ Since these spaces are only considered for one K at a time, this usually does not lead to confusion. 

In a certain sense, these norms measure smoothness, i.e. some <sub>ν</sub>-norm of <sub>x</sub> is finite, only if <sub>x</sub> is somehow smooth and the larger we can take <sub>ν</sub> while $\Vert { x } \Vert _ { \nu }$ stays finite, the smoother <sub>x</sub> is. A bit more precise: The $X ^ { \nu }$ spaces demand a certain decay of the expansion coeficients with respect to the singular basis; the faster the coefficients $\left. x , v _ { n } \right.$ decay, the “smoother” the <sub>x</sub> is. As we have already observed: The singular vectors $v _ { n }$ with large <sub>n</sub> are usually highly oscillating and thus, they can not contribute much to a function that is in some $X ^ { \nu }$ with large <sub>ν</sub>. In the following example we can make this a bit more precise: 

<sub>Example</sub> 9.3<sub>.</sub> We consider the following integral operator: 

$$
K f (t) = \int_ {0} ^ {1} k (t, s) f (s) \mathrm{d} s, \quad k (t, s) = \left\{ \begin{array}{l l} t (1 - s) & : \quad t \leq s \\ s (1 - t) & : \quad s \leq t \end{array} \right..
$$

This operator maps $L ^ { 2 } ( [ 0 , 1 ] )$ into itself and is compact (cf. Example 4.3). One can show that 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/bf1403f9ade34bf469a61726054673a1b2a34c8d9edffd0a611fec57f73feece.jpg)


$$
K f = g \iff \left\{ \begin{array}{l} - g ^ {\prime \prime} = f, \text {   and   } \\ g (0) = g (1) = 0 \end{array} \right..
$$

Not a full proof but the first calculations: The equation $g = K f$ is 

$$
\begin{array}{l} g (t) = \int_ {0} ^ {1} k (s, t) f (s) \mathrm{d} s \\ = \int_ {0} ^ {t} s (1 - t) f (s) \mathrm{d} s + \int_ {t} ^ {1} t (1 - s) f (s) \mathrm{d} s \\ = (1 - t) \int_ {0} ^ {t} s f (s) \mathrm{d} s + t \int_ {t} ^ {1} (1 - s) f (s) \mathrm{d} s. \end{array}
$$

We directly see that $g ( 0 ) = g ( 1 ) = 0$ follows. We take the derivative on both sides (using the known rules) gives 

$$
\begin{array}{l} g ^ {\prime} (t) = - \int_ {0} ^ {t} s f (s) \mathrm{d} s + (1 - t) t f (t) + \int_ {t} ^ {1} (1 - s) f (s) \mathrm{d} s - t (1 - t) f (t) \\ = - \int_ {0} ^ {t} s f (s) \mathrm{d} s + \int_ {t} ^ {1} (1 - s) f (s) \mathrm{d} s. \end{array}
$$

The second derivative is 

$$
g ^ {\prime \prime} (t) = - t f (t) - (1 - t) f (t) = - f (t).
$$

In principle one can argue similarly in the opposite direction as well. 

Moreover <sub>K</sub> is selfadjoint, since $k ( s , t ) = k ( t , s )$ . One can also show that the SVD of <sub>K</sub> is given by 

$$
\sigma_ {n} = \frac {1}{(\pi n) ^ {2}}, \quad u _ {n} (t) = v _ {n} (t) = \sqrt {2} \sin (\pi n t).
$$

This allows us to characterize the spaces $X ^ { \nu }$ a quite explicitly: By Definition 9.2 we have 

$$
f \in X ^ {\nu} \iff \| f \| _ {\nu} ^ {2} = 2 \pi^ {4 \nu} \sum_ {n} n ^ {4 \nu} | \langle f, \sin (\pi n \cdot) \rangle | ^ {2} <   \infty .
$$

Now one can show the following: For $f \in C ^ { 2 \nu } ( [ 0 , 1 ] )$ with $f ^ { ( 2 k ) } ( 0 ) =$ $f ^ { ( 2 k ) } ( 1 ) = 0 \mathrm { f o r } k = 0 , \dots , \nu - 1$ it holds that 

$$
\| f \| _ {\nu} = \| f ^ {(2 \nu)} \| _ {L ^ {2}}.
$$

We start from the right and use that $v _ { n } ( t ) = { \sqrt { 2 } } \sin ( \pi n t )$ is an orthonormal basis of $L ^ { 2 } ( [ 0 , 1 ] )$ to get 

$$
\| f ^ {(2 \nu)} \| _ {L ^ {2}} ^ {2} = \sum_ {n} | \left\langle f ^ {(2 \nu)}, v _ {n} \right\rangle | ^ {2}.
$$

For the inner products we use integration by parts two times, the boundary conditions of $\dot { } f$ and that the sine functions vanish at the boundary to get 

$$
\begin{array}{l} \left\langle f ^ {(2 \nu)}, v _ {n} \right\rangle = \int_ {0} ^ {1} f ^ {(2 \nu)} (t) \sqrt {2} \sin (\pi n t) \mathrm{d} t \\ = \underbrace {f ^ {(2 \nu - 1)} (t) \sqrt {2} \sin (\pi n t) \Big | _ {0} ^ {1}} _ {= 0} - \int_ {0} ^ {1} f ^ {(2 \nu - 1)} (t) \sqrt {2} (\pi n) \cos (\pi n t) \mathrm{d} t \\ = \underbrace {f ^ {(2 \nu - 2)} (t) \sqrt {2} (\pi n) \cos (\pi n t) \Big | _ {0} ^ {1}} _ {= 0} + \int_ {0} ^ {1} f ^ {(2 \nu - 2)} \sqrt {2} (\pi n) ^ {2} \sin (\pi n t) \mathrm{d} t \\ = (\pi n) ^ {2} \left\langle f ^ {(2 \nu - 2)}, v _ {n} \right\rangle . \end{array}
$$

Recursively, this gives $\left. f ^ { ( 2 \nu ) } , v _ { n } \right. = ( \pi n ) ^ { 2 \nu } \left. f , v _ { n } \right.$ ⟩ and hence 

$$
\| f ^ {(2 \nu)} \| _ {L ^ {2}} ^ {2} = 2 \pi^ {4 \nu} \sum_ {n} n ^ {4 \nu} | \langle f, \sin (\pi n \cdot) \rangle | ^ {2} = \| f \| _ {\nu} ^ {2}
$$

as claimed. 

This can be used to rigorously prove that it holds 

$$
X ^ {\nu} = \overline {{\left\{f \in C ^ {(2 \nu)} ([ 0 , 1 ]) \mid f ^ {(2 k)} (0) = f ^ {(2 k)} (1) = 0 \right\} ^ {\| \cdot \| _ {\nu}}}},
$$

i.e. the space $X ^ { \nu }$ is the closure of the space of <sub>2ν</sub>-times continuous diferentiable functions with respective boundary conditions with respect to the <sub>ν</sub>-norm. $\bigtriangleup$ 

Our aim is to construct methods $\left( R _ { \alpha } , \alpha \right)$ such that the total error $\| R _ { \alpha } y ^ { \delta } - x ^ { \dagger } \| _ { X }$ is small. First we will establish a baseline and analyze the question of “how good can a reconstruction method $R : Y \to X$ be in general”. We introduce a little more notation: 

Definition ${ \pmb 9 } { \cdot } { \pmb 4 }$ (Worst case error in <sub>ν</sub>-spaces)<sub>.</sub> Let $K \in K ( X , Y )$ and $R : Y \to X$ continuous with $R 0 = 0$ . We define 

$$
E _ {\nu} (\delta , \rho , R) = \sup \left\{\| R y ^ {\delta} - x ^ {\dagger} \| _ {X} \Big | x ^ {\dagger} \in X ^ {\nu}, \| K x ^ {\dagger} - y ^ {\delta} \| _ {Y} \leq \delta , \| x ^ {\dagger} \| _ {\nu} \leq \rho \right\}.
$$

<sup>This</sup> <sup>quantity</sup> <sup>is</sup> <sup>the</sup> worst case total error of the method R over solutions in an $X ^ { \nu }$ -ball of radius $\rho$ <sub>and</sub> <sub>noise</sub> <sub>level δ</sub>. Furthermore we define 

$E _ { \nu } ( \delta , \rho ) = \operatorname* { i n f } \left\{ E _ { \nu } ( \delta , \rho , R ) \ | \ R : Y \to X \right.$ continous with $R 0 = 0 \}$ 

<sup>This</sup> <sup>is</sup> <sup>the</sup> best possible worst case error of any method over solutions in an $X ^ { \nu }$ -ball of radius $\rho$ and noise level $\delta .$ 

This “best possible worst case error” may look a bit weird, but is actually not hard to quantify: 

Theorem 9.5. For $K \in K ( X , Y )$ it holds that 

$$
E _ {\nu} (\delta , \rho) \geq \sup \left\{\| x \| _ {X} \mid \| K x \| _ {Y} \leq \delta , \| x \| _ {\nu} \leq \rho \right\} =: e _ {\nu} (\delta , \rho).
$$

Proof. <sup>Let</sup> $x \in X ^ { \nu }$ with $\| K x \| _ { Y } \leq \delta$ and $\| x \| _ { \nu } \leq \rho$ and $R : Y \to X$ be arbitrary (given the conditions). We set $y = K x$ and $y ^ { \delta } = 0$ Then this <sub>x</sub> is in the feasible set for the supremum in the definition of $E _ { \nu } ( \delta , \rho , R )$ and thus ${ \cal E } _ { \nu } ( \delta , \rho , R ) \geq \| x \| _ { X } ^ { - }$ . Taking the supremum over all these <sub>x</sub> we get the inequality $e _ { \nu } ( \delta , \rho ) \leq E _ { \nu } ( \delta , \rho , R )$ . The claim follows by taking the infimum over all <sub>R</sub>. □ 

δ: Noise level 

The quantity on the right hand side has a simple explanation: It is a so-called modulus of continuity of $K ^ { \dagger }$ restricted to some set defined by $\delta ,$ ν and $\rho .$ These three parameters have the following meaning: 

The following theorem shows, how large the lower bound $e _ { \nu } ( \delta , \rho )$ of the best worst case error can be: 

ν: Degree of smoothness 

Theorem 9.6. Let $K \in K ( X , Y )$ . Then it holds for all $\nu , \rho , \delta > 0$ that 

$\rho \colon ^ { \ast }$ “Largeness” in the smoothness class. The noise level is often available (or can be estimated), the smoothness may be guessed, but the largeness in the smoothness class is basically never known. 

$$
e _ {\nu} (\delta , \rho) \leq \delta^ {\frac {\nu}{\nu + 1}} \rho^ {\frac {1}{\nu + 1}}.
$$

Moreover, there exists a sequence $\delta _ { n }$ with $\delta _ { n } \ { \stackrel { n \to \infty } { \longrightarrow } } \ 0$ such that there is equality along that sequence. 

Proof. <sup>Let</sup> $x \in X ^ { \nu }$ with $\| x \| _ { \nu } \leq \rho$ and $\| K x \| _ { Y } \leq \delta .$ . Then there is $z \in X$ such that $x = | K | ^ { \nu } z . \mathrm { w e }$ would like to estimate $\| x \| _ { X } =$ $\| | K | ^ { \nu } z \| _ { X }$ in terms of $\| z \| _ { X } = \| x \| _ { \nu }$ and $\| K x \| _ { Y }$ . To that end we use the following result (also known as <sub>interpolation</sub> <sub>inequality</sub>): For $r > s \ge 0$ and all $x$ it holds that $\| | K | ^ { s } x \| _ { X } \leq \| | K | ^ { r } x \| _ { X } ^ { \frac { s } { r } } \| x \| _ { X } ^ { 1 - \frac { s } { r } }$ 

The definition of $| K | ^ { s }$ gives 

$$
\left| \left| | K | ^ {s} x \right| \right| _ {X} = \sum_ {n} \sigma_ {n} ^ {2 s} | \langle x, v _ {n} \rangle | ^ {2}.
$$

Now we define sequences $a _ { n } ~ = ~ \sigma _ { n } ^ { 2 s } | \langle x , v _ { n } \rangle | ^ { 2 \frac { s } { r } }$ and $b _ { n } \ =$ $\mid \langle x , v _ { n } \rangle \mid ^ { 2 - 2 { \frac { r } { s } } }$ and numbers $p = r / s$ and $q = r / ( r - s )$ . We use the Hölder inequality to get 

$$
\begin{array}{l} \vert \vert \vert K \vert^ {s} x \vert \vert_ {X} ^ {2} = \sum_ {n} \sigma_ {n} ^ {2 s} \vert \langle x, v _ {n} \rangle \vert^ {2} \\ = \sum_ {n} a _ {n} b _ {n} \leq \left(\sum_ {n} a _ {n} ^ {p}\right) ^ {1 / p} \cdot \left(\sum_ {n} b _ {n} ^ {q}\right) ^ {1 / q} \\ = \left(\sum_ {n} \sigma_ {n} ^ {2 r} \vert \langle x, v _ {n} \rangle \vert^ {2}\right) ^ {s / r} \cdot \left(\sum_ {n} \vert \langle x, v _ {n} \rangle \vert^ {2}\right) ^ {(r - s) / r} \\ = \vert \vert \vert K \vert^ {r} x \vert \vert_ {X} ^ {2 s / r} \vert \vert x \vert \vert_ {X} ^ {2 (r - s) / r} \end{array}
$$

which proves the claim. 

We use this claim with $s = \nu$ and $r = \nu + 1$ and 

$$
\begin{array}{r l} & {\| x \| _ {X} = | | | K | ^ {\nu} z \| _ {X} \leq | | | K | ^ {\nu + 1} z \| _ {X} ^ {\frac {\nu}{\nu + 1}} \| z \| _ {X} ^ {\frac {1}{\nu + 1}}} \\ & {\qquad = \| K \underbrace {| K | ^ {\nu} z} _ {= x} \| _ {X} ^ {\frac {\nu}{\nu + 1}} \| z \| _ {X} ^ {\frac {1}{\nu + 1}} \leq \delta^ {\frac {\nu}{\nu + 1}} \rho^ {\frac {1}{\nu + 1}}.} \end{array}
$$

For the equality we set $\delta _ { n } = \rho \sigma _ { n } ^ { \nu + 1 }$ and $x _ { n } = \rho | K | ^ { \nu } v _ { n }$ . One can show that this gives indeed equality. 

Note that by definition of $| K | ^ { \nu }$ we have $x = \rho \sigma _ { n } ^ { \nu } v _ { n }$ and by definition of the ν-norm we have that $\| x \| _ { \nu } = \| \rho v _ { n } \| _ { X } = \rho$ and $\| K x \| _ { Y } = \| \rho \sigma _ { n } ^ { \nu } K v _ { n } \| _ { Y } = \rho \sigma _ { n } ^ { \nu } \| \sigma _ { n } u _ { n } \| = \rho \sigma _ { n } ^ { \nu + 1 } = \delta _ { n }$ as needed. From $\delta _ { n } = \rho \sigma _ { n } ^ { \nu + 1 }$ we get that $\sigma _ { n } = ( \delta _ { n } / \rho ) ^ { 1 / ( \nu + 1 ) }$ and thus 

$$
\| x \| _ {X} = \rho \sigma_ {n} ^ {\nu} = \rho \left(\frac {\delta_ {n}}{\rho}\right) ^ {\frac {\nu}{\nu + 1}} = \delta_ {n} ^ {\frac {\nu}{\nu + 1}} \rho^ {\frac {1}{\nu + 1}}
$$

as desired. 

## 10 Convergence rates for spectral regularization

The above Theorems $9 { \cdot } 5$ and $9 . 6$ give a benchmark with which we can compare regularization methods: They can not do better than the right hand side $\delta ^ { \frac { \nu } { \nu + 1 } } \rho ^ { \frac { 1 } { \nu + 1 } }$ for data $x ^ { \dagger }$ from $X ^ { \nu }$ with $\| x ^ { \dagger } \| _ { \nu } \leq \rho$ 

We fix this in the following definition: 

<sub>Definition</sub> <sub>10.1.</sub> A regularization method $\left( R _ { \alpha } , \alpha \right)$ is called <sub>optimal</sub> for parameters $\rho$ and $\nu ,$ if for all $x ^ { \dagger }$ with $\Vert \boldsymbol { x } ^ { \dagger } \Vert _ { \nu } \leq \rho$ and $y ^ { \hat { \delta } }$ with $\| K \bar { x ^ { \dag } } - y ^ { \delta } \| _ { Y } \leq \dot { \delta }$ it holds that 

$$
\| R _ {\alpha} y ^ {\delta} - x ^ {\dagger} \| _ {X} = \delta^ {\frac {\nu}{\nu + 1}} \rho^ {\frac {1}{\nu + 1}}.
$$

We call the method <sub>order</sub> <sub>optimal</sub> for parameters $\rho$ and $\nu$ if there exists some <sub>C</sub> such that for all $x ^ { \dagger }$ as above it holds that 

$$
\| R _ {\alpha} y ^ {\delta} - x ^ {\dagger} \| _ {X} = C \delta^ {\frac {\nu}{\nu + 1}} \rho^ {\frac {1}{\nu + 1}}.
$$

Finally, we call a method <sub>order</sub> <sub>optimal</sub> for $\nu ,$ if for all $x ^ { \dag } \in X ^ { \nu }$ and $y ^ { \delta }$ with $\| K x ^ { \dagger } - y ^ { \delta } \| \leq \delta$ there exists <sub>C</sub> such that 

$$
\left\| R _ {\alpha} y ^ {\delta} - x ^ {\dagger} \right\| \leq C \delta^ {\frac {\nu}{\nu + 1}}.
$$

The assumptions $\| x ^ { \dagger } \| _ { \nu } \leq \rho$ or $x ^ { \dag } \in X ^ { \nu }$ <sup>are</sup> <sup>called</sup> source conditions and the element $z$ with $| K | ^ { \nu } z = x ^ { \dagger }$ <sup>is</sup> <sup>called</sup> source element<sup>.</sup> 

Recall the Definition $7 . 4$ of a regularizing filter: A family of functions $\varphi _ { \alpha }$ (piecewise continuous and bounded) on the interval $[ 0 , \kappa ] ( \kappa = \| K \| ^ { 2 } )$ is a regularizing filter, if for $\lambda > 0$ it holds that 

In the case of Example $9 . 3 ,$ the source condition $x ^ { \dag } \in X ^ { \nu }$ means that $x ^ { \dagger }$ is 2 -times (weakly) diferentiable (with additional boundary conditions) with 2 -th weak derivative z which is an $L ^ { 2 } .$ function. 

$$
\varphi_ {\alpha} (\lambda) \stackrel {{\alpha \rightarrow 0}} {{\longrightarrow}} \frac {1}{\lambda}, \quad \lambda | \varphi_ {\alpha} (\lambda) | \leq C _ {\varphi}
$$

for some $C _ { \varphi } > 0$ . Theorem $7 . 6$ showed that regularizing filters indeed lead to convergent regularizations. Now we want to answer the question when a regularizing filter is optimal or order optimal. 

The key to such results are estimates of the approximation error under the assumption that $x ^ { \dagger }$ lies in a <sub>ρ</sub>-ball in $X ^ { \nu }$ , i.e. under the assumption that $x ^ { \dagger }$ is “ -smooth” and $\dot { \mathfrak { \omega } } _ { \rho - \mathrm { l a r g e } ^ { \prime \prime } }$ 

We can express such estimates for a given filter $\varphi _ { \alpha }$ with the functions 

$$
\begin{array}{l} \omega_ {\nu} (\alpha) := \sup _ {0 <   \lambda \leq \kappa} \lambda^ {\nu / 2} | r _ {\alpha} (\lambda) | \quad (\text { recall }   r _ {\alpha} (\lambda) = 1 - \lambda \varphi_ {\alpha} (\lambda)) \\ = \sup _ {0 <   \lambda \leq \kappa} \lambda^ {\nu / 2} | 1 - \lambda \varphi_ {\alpha} (\lambda) |. \end{array}
$$

These functions can be used to get bounds on the approximation error, the depend on <sub>α</sub>. 

Lemma 10.2. Let $y ^ { \dag } \in D ( K ^ { \dag } )$ and $x ^ { \dag } \in X ^ { \nu }$ with $\| x ^ { \dagger } \| _ { \nu } \leq \rho .$ Further define $x _ { \alpha } = R _ { \alpha } y ^ { \dagger }$ . Then it holds for all $\alpha > 0$ that 

$$
\| x _ {\alpha} - x ^ {\dagger} \| _ {X} \leq \omega_ {\nu} (\alpha) \rho ,
$$

$$
\| K x _ {\alpha} - K x ^ {\dagger} \| _ {Y} \leq \omega_ {\nu + 1} (\alpha) \rho .
$$

Proof. $\operatorname { I f } \parallel x ^ { \dagger } \parallel _ { \nu } \leq \rho$ we have that $x ^ { \dagger } \ = \ | K | ^ { \nu } w \ = \ ( K ^ { * } K ) ^ { \nu / 2 } w$ with $\| w \| _ { X } \leq \rho .$ . Then from Lemma $7 . 5 \left( 3 \right)$ and using $\left. x ^ { \dagger } , \dot { v } _ { n } \right. =$ $\left. ( K ^ { * } K ) ^ { \nu / 2 } w , v _ { n } \right. = \sigma _ { n } ^ { \nu } \left. w , v _ { n } \right.$ we get 

$$
\begin{array}{r}x ^ {\dagger} - x _ {\alpha} = \sum_ {n} (1 - \sigma_ {n} ^ {2} \varphi_ {\alpha} (\sigma_ {n} ^ {2})) \left<   x ^ {\dagger}, v _ {n} \right> v _ {n}\\= \sum_ {n} r _ {\alpha} (\sigma_ {n} ^ {2}) \sigma_ {n} ^ {\nu} \left<   w, v _ {n} \right> v _ {n}.\end{array}
$$

Taking the squared norm gives 

$$
\begin{array}{r l} & {\| x ^ {\dagger} - x _ {\alpha} \| _ {X} ^ {2} = \sum_ {n} (r _ {\alpha} (\sigma_ {n} ^ {2}) \sigma_ {n} ^ {\nu}) ^ {2} | \langle w, v _ {n} \rangle | ^ {2}} \\ & {\qquad \leq \omega_ {\nu} (\alpha) ^ {2} \sum_ {n} | \langle w, v _ {n} \rangle | ^ {2} \leq \omega_ {\nu} (\alpha) ^ {2} \| w \| _ {X} ^ {2}} \end{array}
$$

as claimed. For the second claim recall from Lemma 7.3 that $\| | K | x \| _ { X } = \| K x \| _ { Y }$ and thus 

$$
\| K x _ {\alpha} - K x ^ {\dagger} \| _ {Y} = \| | K | (x _ {\alpha} - x ^ {\dagger}) \|.
$$

Moreover, 

$$
| K | (x _ {\alpha} - x ^ {\dagger}) = (K ^ {*} K) ^ {1 / 2} r _ {\alpha} (K ^ {*} K) (K ^ {*} K) ^ {\nu / 2} w = \sum_ {n} \sigma_ {n} r _ {\alpha} (\sigma_ {n} ^ {2}) \sigma_ {n} ^ {\nu} \left\langle w, v _ {n} \right\rangle v _ {n}.
$$

Since $| \sigma _ { n } r _ { \alpha } ( \sigma _ { n } ^ { 2 } ) \sigma _ { n } ^ { \nu } | ^ { 2 } = | r _ { \alpha } ( \sigma _ { n } ^ { 2 } ) \sigma _ { n } ^ { \nu + 1 } | ^ { 2 } \leq \omega _ { \nu + 1 } ( \alpha ) ^ { 2 }$ the second claim follows similar to the first one. □ 

Now we can show how to achieve order optimal regularization: 

<sub>Theorem</sub> <sub>10.3</sub> (Order optimal a-priori parameter choice)<sub>.</sub> <sub>Let</sub> $K \in$ $K ( X , Y ) . f \varphi _ { \alpha }$ is a regularizing filter for which fulfills 

$$
\begin{array}{c} \sup _ {0 <   \lambda \leq \| K \| ^ {2}} | \varphi_ {\alpha} (\lambda) | \leq C _ {\varphi} \alpha^ {- 1} \\ \omega_ {\nu} (\alpha) \leq C _ {\nu} \alpha^ {\nu / 2}. \end{array}
$$

If the a-priori choice α fulfills 

$$
c \left(\frac {\delta}{\rho}\right) ^ {\frac {2}{\nu + 1}} \leq \alpha (\delta) \leq C \left(\frac {\delta}{\rho}\right) ^ {\frac {2}{\nu + 1}}
$$

for some $0 < c < C$ , then $\left( R _ { \alpha } , \alpha \right)$ is an order optimal regularization method in the sense of Definition 10.1. 

<sub>Proof.</sub> As always we start with our error decomposition 

$$
\| x _ {\alpha (\delta)} ^ {\delta} - x ^ {\dagger} \| _ {X} \leq \delta \| R _ {\alpha (\delta)} \| + \| x _ {\alpha (\delta)} - x ^ {\dagger} \| _ {X}.
$$

From Lemma $7 . 5 \left( 2 \right)$ and our first assumption we know that 

$$
\| R _ {\alpha (\delta)} \| \leq \sqrt {C _ {\varphi}} \sqrt {\sup _ {0 <   \lambda \leq \| K \| ^ {2}} | \varphi_ {\alpha (\delta)} (\lambda) |} \leq C _ {\varphi} \alpha (\delta) ^ {- 1 / 2}.
$$

From Lemma 10.2 and our second assumption we get 

$$
\left\| x _ {\alpha (\delta)} - x ^ {\dagger} \right\| _ {X} \leq \omega_ {\nu} (\alpha (\delta)) \rho \leq C _ {\nu} \alpha (\delta) ^ {\nu / 2} \rho .
$$

We use these estimates in the error decomposition and use the upper and lower bound of the parameter choice to get 

$$
\begin{array}{r l} & {\| x _ {\alpha (\delta)} ^ {\delta} - x ^ {\dagger} \| _ {X} \leq C _ {\varphi} \alpha (\delta) ^ {- 1 / 2} \delta + C _ {\nu} \alpha (\delta) ^ {\nu / 2} \rho} \\ & {\quad \leq C _ {\varphi} c ^ {- 1 / 2} \delta^ {- \frac {1}{\nu + 1}} \rho^ {\frac {1}{\nu + 1}} \delta + C _ {\nu} C ^ {\nu / 2} \delta^ {\frac {\nu}{\nu + 1}} \rho^ {- \frac {\nu}{\nu + 1}} \rho} \\ & {\quad = (C _ {\varphi} c ^ {- 1 / 2} + C _ {\nu} C ^ {\nu / 2}) \delta^ {\frac {\nu}{\nu + 1}} \rho^ {\frac {1}{\nu + 1}}.} \end{array}
$$

Let us investigate the few filters we know if they can lead to order optimal methods To that end, let us collect the inequalities that we need: 

<sub>Example</sub> 10.4 (Order optimality of the truncated SVD)<sub>.</sub> The filter is $\varphi _ { \alpha } ( \lambda ) = 1 / \lambda$ for $\lambda \geq \alpha$ and <sub>= 0</sub> else. Thus $r _ { \alpha } ( \lambda ) = 0$ for $\lambda \geq \alpha$ and <sub>= 1</sub> else. We get 

$$
\sup _ {\lambda} | \varphi_ {\alpha} (\lambda) | = \frac {1}{\alpha} (\text { attained   at } \lambda = \alpha) \quad \Longrightarrow C _ {\varphi} = 1,
$$

and 

$$
\omega_ {\nu} (\alpha) = \sup _ {\lambda} \lambda^ {\nu / 2} | r _ {\alpha} (\lambda) | = \alpha^ {\nu / 2} (\text { attained   at } \lambda = \alpha) \quad \Longrightarrow C _ {\nu} = 1.
$$

We see that the truncated SVD is indeed an order optimal regularization method (for any $\nu > 0 ) !$ As such, the method can make use of any smoothness in the (unknown) solution. The smoother $x ^ { \dagger }$ , the better the convergence rate will be since the exponent $\nu / ( \nu + 1 )$ of <sub>δ</sub> in Theorem 10.3 will be larger for larger $\nu . \quad \Delta$ Example $_ { 1 0 . 5 }$ (Order optimality and saturation for Tikhonov regularization)<sub>.</sub> The filter is $\varphi _ { \alpha } ( \lambda ) = ( \lambda + \alpha ) ^ { - 1 }$ and thus 

$$
r _ {\alpha} (\lambda) = 1 - \lambda \varphi_ {\alpha} (\lambda) = 1 - \frac {\lambda}{\lambda + \alpha} = \frac {\alpha}{\lambda + \alpha}.
$$

We get 

$$
\sup _ {\lambda} | \varphi_ {\alpha} (\lambda) | = \frac {1}{\alpha} (\text { attained   at } \lambda = 0) \quad \Longrightarrow C _ {\varphi} = 1.
$$

For the other condition we need to investigate the supremum of $\begin{array} { r } { \lambda ^ { \nu / 2 } | r _ { \alpha } ( \lambda ) | = \alpha \frac { \lambda ^ { \nu / 2 } } { \lambda + \alpha } } \end{array}$ . We define $\begin{array} { r } { f ( \lambda ) = \frac { \lambda ^ { \nu / 2 } } { \lambda + \alpha } } \end{array}$ and note: 

$\nu > 2 :$ The function <sub>f</sub> is unbounded on $\mathrm { ] 0 , \infty [ }$ <sub>[</sub> (since $\nu / 2 >$ <sub>1</sub>) and hence, the supremum is infinite. 

$\nu = 2$ . Here $f ( \lambda ) < 1$ and even $f ( \lambda ) \to 1 \mathrm { f o r } \lambda \to$ <sub>∞</sub> and thus 

$$
\sup _ {\lambda} \lambda^ {\nu / 2} | r _ {\alpha} (\lambda) | = \alpha = \alpha^ {\nu / 2}, \implies C _ {1} = 1.
$$

$0 < \nu < 2 \colon$ Here we have $f ( 0 ) = 0$ and $f ( \lambda )  0$ for $\lambda  0$ and hence a finite maximum exists. The condition $f ^ { \prime } ( \lambda ) = 0$ i $: 0 = ( \frac { \nu } { 2 } \lambda ^ { \frac { \nu } { 2 } - 1 } ( \lambda + \alpha ) - \lambda ^ { \frac { \nu } { 2 } } ) / ( \lambda + \alpha ) ^ { 2 }$ which holds exactly if $\lambda ^ { \frac { \nu } { 2 } - 1 } ( \not { p } ( \lambda + \alpha ) - \lambda ) = 0$ Since $0 < \lambda < 2$ the only solution is $\begin{array} { r } { \lambda = \frac { \bar { \nu } } { 2 - \nu } \alpha } \end{array}$ and corresponds to a maximum. We plug this is and get 

$$
\begin{array}{l} \sup _ {\lambda} \lambda^ {\nu / 2} | r _ {\alpha} (\lambda) | = \alpha \left(\frac {\nu}{2 - \nu} \alpha\right) ^ {\nu / 2} \left(\frac {\nu}{2 - \nu} \alpha + \alpha\right) ^ {- 1} \\ \qquad = \alpha \left(\frac {\nu}{2 - \nu}\right) ^ {\nu / 2} \alpha^ {\frac {\nu}{2}} \left(\frac {2}{2 - \nu}\right) ^ {- 1} \alpha^ {- 1} \\ \qquad = \left(\frac {\nu}{2 - \nu}\right) ^ {\frac {\nu}{2}} \frac {2 - \nu}{2} \alpha^ {\nu / 2} \\ \qquad = \frac {\nu^ {\nu / 2} (2 - \nu) ^ {(2 - \nu) / 2}}{2} \alpha^ {\nu / 2} \implies C _ {\nu} = \frac {\sqrt {\nu^ {\nu} (2 - \nu) ^ {2 - \nu}}}{2}. \end{array}
$$

The constant $C _ { \nu }$ is not as bad as it may look. Here is $C _ { \nu }$ is dependence on ν: 

In conclusion: Tikhonov regularization is an order optimal regularization method for $0 \leq \nu \leq 2$ but not for $\nu > 2 .$ . As such, it can take advantage of smoothness up to the space $X ^ { 2 } = \mathrm { r g } | K | ^ { 2 } = \mathrm { r g } ( K ^ { * } K )$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/b26dab897c1235c7fae0e8f78b63faa0be796220ffde30fe677ce3b9e458f0ba.jpg)


△ 

While the a-priori rule from Theorem 10.3 indeed leads to order optimal methods, one drawback is that they need knowledge about both <sub>ν</sub> and $\rho .$ . Without knowledge of $\rho$ one could still choose $\alpha ( \delta ) \ \propto \ \delta ^ { 2 / ( \nu + 1 ) }$ and obtain on order optimal method ( just go through the estimates in the proof and see that you get the right exponent for $\delta$ in the end). 

Morozov’s discrepancy principle (the only a-posteriori method we know) does not need any knowledge about <sub>ν</sub> or $\rho .$ Remarkably, this parameter choice also turns out to be order optimal (but in slightly less cases). Before we can formulate this, we make one more definition: 

The symbol ∝ indicates that the left hand side is proportional to the right hand side, i.e. that $\alpha ( \delta ) = C \delta ^ { 2 / ( \nu \overline { { + } } 1 ) }$ for some C. We could also consider $c \delta ^ { 2 / ( \nu + 1 ) } \leq \alpha ( \delta ) \leq C \delta ^ { 2 / ( \nu + 1 ) }$ 

Definition 10.6. <sup>Let</sup> $\varphi _ { \alpha }$ be a regularizing filter with <sub>su</sub> $\begin{array} { r } { \mathsf { P } _ { 0 < \lambda \leq \| K \| ^ { 2 } } | \varphi _ { \alpha } ( \lambda ) | \leq } \end{array}$ $C _ { \varphi } \alpha ^ { - 1 }$ . We say that this filter has <sub>qualification</sub> $\nu _ { 0 } \mathrm { i f } \omega _ { \nu } ( \alpha ) \le C _ { \nu } \alpha ^ { \nu / 2 }$ is fulfilled for all $0 \leq \nu \leq \nu _ { 0 }$ . (It it holds for all $\nu \geq 0$ we say that the qualification is $\nu _ { 0 } = \infty )$ 

<sub>Theorem</sub> <sub>10.7</sub> (Optimality of Morozov’s discrepancy principal)<sub>.</sub> The $\varphi _ { \alpha }$ be a regularizing filter with qualification $\nu _ { 0 } > 0$ and let 

Then the parameter choice defined by Morozov’s discrepancy principle with this τ (cf. Section 8) is an order optimal regularization method for $0 \leq \nu \leq \nu _ { 0 } - 1$ 

$$
\tau > \sup _ {\alpha > 0, 0 <   \lambda \leq \kappa} | r _ {\alpha} (\lambda) | =: C _ {r}.
$$

Unfortunately, the proof does not fit into the lecture. It can be found in the lecture notes “Regularization of inverse problems” by Christian Clason, Theorem 5.11, available at <sub>https://arxiv.</sub> org/abs/2001.00617<sup>.</sup> 

Here is an example that shows that the results of Theorem 10.3 is indeed quite close to what one can observe in practice: 

```matlab
% Dimension of the problem
n = 2000;
% deriv2 from the regu-toolbox (from MATLAB's file exchange) implements the
% "inverse of the negative second derivative" from Example 8.3
[A,~,~,~] = deriv2(n);

% The constant 1 function should be in X^nu for nu<1/4.
% We take the edge case anyway.
x = ones(n,1);
nu = 1/4;

% now do three choices for alpha of the form alpha = delta^s
% one optimal, the other too large and small, resp.
% According to Theorem 9.3 the optimal s is s=2/(nu+1) = 2/(5/4) = 1.6
% expected rate for alpha = delta^s is C*delta^r with r= min(1-s/2,s*nu/2)
% See the proof of Theorem 9.3
s1 = 1.6;
r1 = min(1-s1/2,s1*nu/2)
s2 = 1;
r2 = min(1-s2/2,s2*nu/2)
s3 = 3;
r3 = min(1-s3/2,s3*nu/2)

% Some noise levels going to zero
deltas = logspace(0,-5,20);
% Some noise levels going to zero
% Precompute A'*A
ATA = A'*A;
y = A*x;
for k = 1:length(deltas)
    % for each delta construct data with that noise level
    delta = deltas(k);
    noise = randn(n,1);noise = noise/norm(noise);
    ydelta = y + delta*noise;
    C = 1; % C could be anything. It's choice does not affect the rate,
    % but it does affect the values or the errors.
    % these are our alphas
    alpha1 = C*delta^s1;
    alpha2 = C*delta^s2;
    alpha3 = C*delta^s3;

    % Precompute A'*ydelta
    ATydelta = A'*ydelta;
    % reconstruct by Tikhonov 
```

```matlab
x1 = (ATA+alpha1*eye(n))\ (ATydelta);
x2 = (ATA+alpha2*eye(n))\ (ATydelta);
x3 = (ATA+alpha3*eye(n))\ (ATydelta);

% measure the errors
error1(k) = norm(x1 - x);
error2(k) = norm(x2 - x);
error3(k) = norm(x3 - x);
end

offset = error1(1); %used to adjust the plots
% plor total errors:
loglog(deltas, error1, deltas, error2, deltas, error3)
hold on
% plot the expected rates
loglog(deltas, offset*deltas.^r1,...
    deltas, offset*deltas.^r2,...
    deltas, offset*deltas.^r3)
legend('s=1.6', 's=1', 's=3', 'rate for s=1.6', 'rate for s=1', 'rate for s=3')

r1 = 0.2000
r2 = 0.1250
r3 = -0.5000 
```

```txt
s=1.6, s=1, s=3, rate for s=1.6, rate for s=1, rate for s=3 
```

One should add that the plot changes quite a bit if we move to even smaller noise levels. There we will see that the plot for <sub>s = 3</sub> (much too small regularization parameter) will go down again, contrary to what has been predicted by theory. This may due to efects of the finite precision of floating point arithmetic. 

## 11 Iterative regulariztion

The idea of iterative regularization is to use iterative methods that can either solve $K x = y$ or minimize $\| K x - y \| _ { Y } ^ { 2 }$ exactly in the case where <sub>y</sub> is in the range of <sub>K</sub> (or the domain of the pseudo inverse in the latter case), apply them to the case with some $y ^ { \delta }$ instead o $\hat { \cdot } y ,$ even though $y ^ { \delta }$ is not in the range of <sub>K</sub>. We expect that the method will not converge in this case and hence, we stop them at some point. The simplest iterative regularization method is the <sub>Landweber</sub> <sub>method</sub> and we motivate the method two times: 

<sub>Example</sub> 11.1 (Landweber as a fixed point iteration)<sub>.</sub> We start from the normal equations $K ^ { * } K x = K ^ { * } y$ and rewrite them as 

$$
x = x - \omega (K ^ {*} K x - K ^ {*} y) = x - \omega K ^ {*} (K x - y)
$$

for some $\omega \in \mathbb { R }$ We turn this into a fixed point iteration 

$$
x _ {n + 1} = x _ {n} - \omega K ^ {*} (K x _ {n} - y).
$$

For the sake of simplicity we start with $x _ { 0 } = 0$ . We analyze the convergence of the iteration by Banach’s fixed point theorem. The map under consideration is $x \mapsto x - \omega K ^ { * } ( K x - y )$ , so we analyze 

$$
\| x - \omega K ^ {*} (K x - y) - (x ^ {\prime} - \omega K ^ {*} (K x ^ {\prime} - y) \| _ {X} = \| (I - \omega K ^ {*} K) (x - x ^ {\prime}) \| _ {X}.
$$

We see that the iteration map is a contraction if $\| \mathbf { i d } - \omega K ^ { * } K \| < 1$ If this holds, we can see inductively that from $x _ { 0 } = 0 \in { \mathrm { r g } } ( K ^ { * } )$ it follows that $x _ { n } \in { \mathrm { r g } } ( K ^ { * } )$ as well, and hence we get convergence $x _ { n }  x ^ { \dagger } = K ^ { \dagger } y .$ . For $y ^ { \delta } \notin \mathrm { r g } ( K )$ we can’t expect convergence, and need to stop early. Here the stopping index <sub>m</sub> act as regularization parameter, but to be consistent with our convention “smaller regularization parameter is less regularization” it’s more like $\alpha = 1 / m$ is the regularization parameter. △ 

Lemma 11.2. $f 0 < \omega < 2 / \| K \| ^ { 2 } , t h e n \| \mathrm { i d } - \omega K ^ { * } K \| \le 1$ 

<sub>Proof.</sub> For the singular values $\sigma _ { j }$ of $K$ we know that $0 ~ < ~ \sigma _ { j } ~ \le$ $\| K \|$ . The operator <sub>id</sub> $- \omega K ^ { * } K$ is self adjoint and has eigenvalues $\ddot { 1 } - \omega \sigma _ { i } ^ { 2 } \in \mathsf { \bar { \Gamma } } ] 1 - \omega \| K \| ^ { 2 } , 1 |$ <sub>[</sub> and since $0 ^ { - } < \omega < 2 / \| K \| ^ { 2 }$ we have that the eigenvalues of <sub>id</sub> $- \omega K ^ { * } K$ lie in <sub>]−1,</sub> <sub>1]</sub> as needed. □ 

As a consequence, convergence of the Landweber method does not follow from Banach fixed point theorem (it does so, if <sub>K</sub> is injective and the smallest singular value exists and is positive, but then the problem is well posed). 

Here is another view on the Landweber method. 

<sub>Example</sub> 11.3 (Landweber as gradient descent on the least squares functional)<sub>.</sub> We can also start with the least squares functional 

$$
f (x) = \frac {1}{2} \| K x - y \| _ {Y} ^ {2}.
$$

To calculate the derivative of <sub>f</sub> we do 

$$
\begin{array}{c} f (x + h) = \frac {1}{2} \| K (x + h) - y \| _ {Y} ^ {2} = \frac {1}{2} \| K x - y + K h \| _ {Y} ^ {2} \\ \frac {1}{2} \| K x - y \| _ {y} ^ {2} + \langle K x - y, K h \rangle + \frac {1}{2} \| K h \| _ {Y} ^ {2} \\ = f (x) + \langle K ^ {*} (K x - y), h \rangle + \varphi (h) \end{array}
$$

with $\varphi ( h ) \ = \ { \textstyle \frac { 1 } { 2 } } \| { \cal K } h \| _ { Y } ^ { 2 }$ . Since $\varphi ( h ) / \| h \| _ { X } \leq \| K \| ^ { 2 } \| h \| \to 0$ for $h  0$ we get that the gradient of <sub>f</sub> is 

$$
\nabla f (x) = K ^ {*} (K x - y).
$$

Hence, gradient descent for $f$ with constant stepsize <sub>ω</sub> is 

$$
x _ {n + 1} = x _ {n} - \omega \nabla f (x _ {n}) = x _ {n} - \omega K ^ {*} (K x - y)
$$

which is exactly the Landweber iteration we in the previous example. $\bigtriangleup$ 

The next lemma shows how the <sub>n</sub>-th iterate can be written explicitly: 

Lemma 11.4. $H x _ { 0 } = 0 ;$ , then the m-th iterate of the Landweber method with stepsize ω is given by 

$$
x _ {n} = \omega \sum_ {n = 0} ^ {m - 1} (\mathrm{id} - \omega K ^ {*} K) ^ {n} K ^ {*} y.
$$

<sub>Proof.</sub> We prove this by induction: For $m = 1$ we have 

$$
x _ {1} = \omega K ^ {*} y = \omega (\mathrm{id} - K ^ {*} K) ^ {0} K ^ {*} y.
$$

For the induction step we start with 

$$
\begin{array}{l} x _ {m + 1} = x _ {m} - \omega K ^ {*} (K x _ {m} - y) = (\mathrm{id} - \omega K ^ {*} K) x _ {m} + \omega K ^ {*} y \\ \qquad = (\mathrm{id} - \omega K ^ {*} K) \left(\omega \sum_ {n = 0} ^ {m - 1} (\mathrm{id} - \omega K ^ {*} K) ^ {n} K ^ {*} y\right) + \omega K ^ {*} y \\ \qquad = \omega \sum_ {n = 0} ^ {m - 1} (\mathrm{id} - \omega K ^ {*} K) ^ {n + 1} K ^ {*} y + \omega (\mathrm{id} - \omega K ^ {*} K) ^ {0} K ^ {*} y \\ \qquad = \omega \sum_ {m = 0} ^ {m} (\mathrm{id} - \omega K ^ {*} K) ^ {n} K ^ {*} y. \end{array}
$$

Hence, <sub>m</sub> steps of the Landweber method are the same as 

$$
x _ {m} = \varphi_ {m} (K ^ {*} K) K ^ {*} y
$$

with the filter function 

$$
\varphi_ {m} (\lambda) = \omega \sum_ {n = 0} ^ {m - 1} (1 - \omega \lambda) ^ {n}.
$$

Using the geometric sum $\textstyle \sum _ { k = 0 } ^ { m } q ^ { k } = { \frac { 1 - q ^ { m + 1 } } { 1 - q } }$ this gives 

$$
\varphi_ {m} (\lambda) = \omega \sum_ {n = 0} ^ {m - 1} (1 - \omega \lambda) ^ {n} = \omega \frac {1 - (1 - \omega \lambda) ^ {m}}{1 - (1 - \omega \lambda)} = \frac {1 - (1 - \omega \lambda) ^ {m}}{\lambda}.\tag{1}
$$

Inverse Problems <sub>|</sub> Version of June 15, 2023 <sub>|</sub> SoSe 2022 

Theorem 11.5. Let $\varphi _ { m }$ be defines by <sup>(1)</sup>. Then it holds that $R _ { m } =$ $\varphi _ { m } ( K ^ { * } K ) K ^ { * }$ defined a regularization $i f { \overset { . } { 0 } } < \omega < 2 / \| K \| ^ { 2 }$ 

<sub>Proof.</sub> By Theorem $7 . 6$ we only need to show that $\varphi _ { m }$ is a regularizing filter, i.e. that $\varphi _ { m } ( \lambda )  1 / \lambda$ for $m $ <sub>∞</sub> and $0 < \lambda < \| K \| ^ { 2 }$ (recall that $\alpha = 1 / m$ act as regularization parameter) and that $\lambda \varphi _ { m } ( \lambda )$ is uniformly bounded for all <sub>m</sub>. 

Since we have $0 < \omega < 2 / \| K \| ^ { 2 }$ we have for all <sub>λ</sub> with $0 ~ <$ $\lambda \leq \| K \| ^ { 2 }$ that 

$$
- 1 <   1 - \omega \lambda <   1,
$$

and hence $( 1 - \omega \lambda ) ^ { m } \to 0$ for $m  \infty$ . Moreover we have for $0 \leq \lambda \leq \| K \| ^ { 2 }$ 

$$
\lambda | \varphi_ {m} (\lambda) | = | 1 - (1 - \omega \lambda) ^ {m} | \leq 2 =: C _ {\varphi}.
$$

We can even show that the Landweber method is an order optimal method. 

<sub>Theorem</sub> <sub>11.6</sub> (Landweber is order optimal)<sub>. Let</sub> $0 < \omega < 2 / \| K \| ^ { 2 }$ Then the Landweber method with a-priori rule $\dot { m } ^ { * } = m ( \delta ) \propto \delta ^ { - 2 / ( \nu + 1 ) }$ is an order optimal regularization method for any $\nu > 0$ 

Recall from Section 10 that this holds for a-priori parameter choices of the form $\alpha ( \delta ) \ \propto \ \delta ^ { 2 / ( \nu + 1 ) }$ so here we should stop after about $m ^ { * }$ ≈ ${ \delta ^ { - 2 / \left( \nu + 1 \right) } }$ iterations. 

<sub>Proof.</sub> We use Theorem 10.3 and need to show that 

$$
\begin{array}{c} \sup _ {0 <   \lambda \leq \| K \| ^ {2}} | \varphi_ {m} (\lambda) | \leq C _ {\varphi} m \\ \omega_ {\nu} (m) = C _ {\nu} m ^ {- \nu / 2} \end{array}
$$

(recall that $\alpha = 1 / m$ and don’t confuse $\omega _ { \nu }$ with the stepsize $\omega ) .$ . 

For the first estimate consider recall that $- 1 < 1 - \omega \lambda < 1$ and hence, by Bernoulli’s inequality 

$$
| \varphi_ {m} (\lambda) | = \frac {| 1 - (1 - \omega \lambda) ^ {m} |}{\lambda} = \frac {1 - (1 - \omega \lambda) ^ {m}}{\lambda} \leq \frac {1 - (1 - m \lambda \omega)}{\lambda} = \omega m.
$$

From above we already had $C _ { \varphi } = 2 ,$ , so now we should set $C _ { \varphi } =$ max $( 2 , \omega )$ . To estimate $\omega _ { \nu } ( \lambda )$ we use consider 

$$
\lambda^ {\nu / 2} (1 - \lambda \varphi_ {m} (\lambda)) = \lambda^ {\nu / 2} (1 - \omega \lambda) ^ {m}
$$

and substitute $t = m \lambda$ . Then this expression becomes 

$$
h (t) = \left(\frac {t}{m}\right) ^ {\nu / 2} (1 - \frac {\omega t}{m}) ^ {m}.
$$

We use the elementary inequality $\begin{array} { r } { ( 1 - \frac { x } { m } ) ^ { m } \leq e ^ { - x } } \end{array}$ and get 

$$
h (t) m ^ {- \nu / 2} t ^ {\nu / 2} e ^ {- \omega t}.
$$

The derivative is 

$$
\begin{array}{c} h ^ {\prime} (\lambda) = m ^ {- \nu / 2} \left(\frac {\nu}{2} t ^ {\nu / 2 - 1} e ^ {- \omega t} + t ^ {\nu / 2} (- \omega) e ^ {- \omega t}\right) \\ = m ^ {- \nu / 2} t ^ {\nu / 2 - 1} e ^ {- \omega t} \left(\frac {\nu}{2} - t \omega\right) \end{array}
$$

and thus, <sub>h</sub> has a global maximum at $t = \nu / ( 2 \omega )$ . This shows that 

$$
h (t) \leq m ^ {- \nu / 2} \left(\frac {\nu}{2 \omega}\right) ^ {\nu / 2} e ^ {- \nu / 2},
$$

and thus 

$$
\omega_ {\nu} (\lambda) \leq C _ {\nu} m ^ {- \nu / 2},
$$

i.e. $\begin{array} { r } { C _ { \nu } = \left( \frac { \nu } { 2 \omega } \right) ^ { \frac { \nu } { 2 } } e ^ { - \nu / 2 } . } \end{array}$ 

Note that iterative methods are well suited for Morozov’s discrepancy principle. One just monitors $\| K x _ { m } - y ^ { \delta } \| _ { Y }$ during the iteration and stops at the first $m ^ { * }$ such that $\| K x _ { m ^ { * } } - y ^ { \delta } \| _ { Y } \leq \tau \delta .$ 

One can actually show a little bit more here: Let us denote by $x _ { m } ^ { \delta }$ the <sub>m</sub>-th iterate of the Landweber method with $y ^ { \delta }$ instead of <sub>y</sub>. 

Theorem 11.7. $\boldsymbol { { \it H K } } \boldsymbol { x } _ { m } ^ { \delta } - \boldsymbol { y } ^ { \delta } \neq 0$ , then it holds for stepsizes $0 < \omega <$ $2 / \| K \| ^ { 2 }$ that 

$$
\| K x _ {m + 1} ^ {\delta} - y ^ {\delta} \| _ {Y} \leq \| K x _ {m} ^ {\delta} - y ^ {\delta} \| _ {Y}.
$$

Moreover, $i f \| K x _ { m } ^ { \delta } - y ^ { \delta } \| _ { Y } > 2 \delta$ and $0 < \omega < 1 / \| K \| ^ { 2 }$ we even have 

$$
\| x _ {m + 1} ^ {\delta} - x ^ {\dagger} \| _ {X} <   \| x _ {m} ^ {\delta} - x ^ {\dagger} \| _ {X}.
$$

<sub>Proof.</sub> We compute from the iteration 

$$
\begin{array}{c} K x _ {m + 1} ^ {\delta} - y ^ {\delta} = K ((\mathrm{id} - \omega K ^ {*} K) x _ {m} ^ {\delta} + \omega K ^ {*} y ^ {\delta} - y ^ {\delta} \\ = (\mathrm{id} - \omega K ^ {*} K) (K x _ {m} ^ {\delta} - y ^ {\delta}). \end{array}
$$

For stepsize $\omega \in \left] 0 , 2 / \| K \| ^ { 2 } \right[$ <sub>[</sub> we get that $\| \mathbf { i d } - \omega K ^ { * } K \| \leq 1$ and thus shows the first claim. 

For the second claim we write $z _ { m } ^ { \delta } : = y ^ { \delta } - K x _ { m } ^ { \delta }$ and $y = K x ^ { \dagger }$ and get 

$$
\begin{array}{r l}&{\| x _ {m + 1} ^ {\delta} - x ^ {\dagger} \| _ {X} ^ {2} = \| x _ {m} ^ {\delta} - x ^ {\dagger} - \omega K ^ {*} (K x _ {m} ^ {\delta} - y ^ {\delta}) \| _ {X} ^ {2}}\\&{\qquad = \| x _ {m} ^ {\delta} - x ^ {\dagger} \| _ {X} ^ {2} + 2 \omega \left<   x _ {m} ^ {\delta} - x ^ {\dagger}, K ^ {*} z _ {m} ^ {\delta} \right> + \omega^ {2} \| K ^ {*} z _ {m} ^ {\delta} \| _ {X} ^ {2}}\\&{\qquad = \| x _ {m} ^ {\delta} - x ^ {\dagger} \| _ {X} ^ {2} + 2 \omega \left<   K x _ {m} ^ {\delta} - K x ^ {\dagger}, z _ {m} ^ {\delta} \right> + \omega^ {2} \| K ^ {*} z _ {m} ^ {\delta} \| _ {X} ^ {2}}\\&{\qquad = \| x _ {m} ^ {\delta} - x ^ {\dagger} \| _ {X} ^ {2} + \omega \left<   z _ {m} ^ {\delta} + 2 K x _ {m} ^ {\delta} - 2 y, z _ {m} ^ {\delta} \right> + \omega (\omega \| K ^ {*} z _ {m} ^ {\delta} \| _ {X} ^ {2} - \| z _ {m} ^ {\delta} \| _ {Y} ^ {2}.}\end{array}
$$

We aim to show that the last two terms are negative. For the first term we compute 

$$
\begin{array}{r l} & {\left\langle z _ {m} ^ {\delta} + 2 K x _ {m} ^ {\delta} - 2 y, z _ {m} ^ {\delta} \right\rangle = \left\langle y ^ {\delta} - K x _ {m} ^ {\delta} + 2 K x _ {m} ^ {\delta} - 2 y, z _ {m} ^ {\delta} \right\rangle} \\ & {\qquad = \left\langle y ^ {\delta} + K x _ {m} ^ {\delta} - 2 y, z _ {m} ^ {\delta} \right\rangle} \\ & {\qquad = 2 \left\langle y ^ {\delta} - y, z _ {m} ^ {\delta} \right\rangle - \| z _ {m} ^ {\delta} \| _ {Y} ^ {2}} \\ & {\qquad \leq 2 \delta \| z _ {m} ^ {\delta} \| _ {Y} ^ {2} - \| z _ {m} ^ {\delta} \| _ {Y} ^ {2}} \\ & {\qquad = (2 \delta - \| K x _ {m} ^ {\delta} - y ^ {\delta} \| _ {Y}) \| z _ {m} ^ {\delta} \| _ {Y} <   0.} \end{array}
$$

For the second term we use $\omega < 1 / \Vert K \Vert ^ { 2 }$ to get 

$$
\omega \| K ^ {*} z _ {m} ^ {\delta} \| _ {X} ^ {2} \leq \omega \| K \| ^ {2} \| z _ {m} ^ {\delta} \| _ {Y} ^ {2} <   \| z _ {m} ^ {\delta} \| _ {Y} ^ {2}
$$

which shows that the last term is negative as well. 

□ 

The theorem shows two important things: First, the Landweber method always decreases the residual but, more importantly, even the distance to the <sub>true</sub> solution decreases if the residual is larger than <sub>2δ</sub>, so it is always beneficial to use $\tau < 2$ in Morozov’s discrepancy principle. Note that the Landweber method can always be applied when one is able to apply the operator <sub>K</sub> and its adjoint <sub>K</sub>∗. The is no need for singular value decomposition (as for the TSVD) and also we do not need to solve linear systems (like for Tikhonov regularization). One downside of the Landweber method is that it usually needs a lot of iterations until a good reconstruction is achieved (or Morozov’s discrepancy principle kicks is). In practice one can use other iterative methods that solve the normal equations, e.g. the method of conjugate gradients (CG) which converges much faster. One can show (with very diferent tool than here) that, combined with the discrepancy principle, is indeed a regularization method. 

Here is an example with the Landweber iteration: 

```matlab
% problem size and matrix
n = 500;
A = tril(ones(n))/n;

% discretized interval
t = linspace(0,1,n)';

% some true solutions. Uncomment the one you want to use
%xdag = 1-t.^2;
xdag = max(1-2*t,0);
%xdag = (t<0.5);
% true data
ydag = A*xdag;

% noisy data
eta = randn(n,1); eta = eta/norm(eta); % normalized noise
delta = 0.05; % noise level
ydelta = ydag + delta*eta;

% stepsize for the Landweber iteration
normA = norm(A);
omega = 1/normA;
% constant for Morozov's discrepancy principle
tau = 1.01;

% number of iterations 
```

```matlab
m = 3000;
% initialization
x = zeros(n,1);

residual = zeros(m,1);
error = zeros(m,1);
stopped = false;
for k=1:m
    x = x - omega*A'*(A*x-ydelta);
    error(k) = norm(x-xdag);
    residual(k) = norm(A*x-ydelta);
    % if the residual falls below the noise level for the first time
    % record the index and the reconstruction at that time
    if stopped==false && residual(k)<tau*delta
    mstar = k;
    xrec = x;
    stopped = true;
    end
end

semilogy(1:m,error,1:m,residual,1:m,tau*delta*ones(m,1))
title('semi convergence of the Landweber method')
legend('||x_m-x^+||', '||Kx_m-ydelta||', '\tau\delta') 
```

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/73376f22226a1268231c8343931749936dc21f1a828fc2c51dda1079475f1fbe.jpg)


```txt
fprintf('stopping index: m* = %d\n',mstar) 
```

```csv
subplot(1,2,1)
plot(t,xrec,t,xdag)
title('Landweber stopped with Morozov')
legend('xrec','xdag')
subplot(1,2,2) 
```

plot(t,A*xrec ,t,ydag) title(’same on image side ’) legend(’A*rec ’,’ydag ’) 

stopping index: m* = 221 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/24e42a07bba8aae31ffa28d61caf245d57c4ab0d8e6328a73395975c5051531e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/deda53c24d92eca13553ef88298f03a253c4a9afc077d360f3cdc45b3cde0c6f.jpg)


## 12 A Bayesian perspective on regulariztion

In this section we would like to draw connections between regularization (especially Tikhonov regularization) and probabilistic approaches to inverse problems. This will be much simpler if we consider everything to be finite dimensional, i.e. $K \in \mathbb { R } ^ { m \times n }$ is a matrix, $X = \bar { \mathbb { R } ^ { n } }$ and $Y = \mathbb { R } ^ { m }$ . We start by modeling noise stochastically. Our data is $y ^ { \delta } = K x ^ { \dagger } + \eta$ where $\eta \in \mathbb { R } ^ { m }$ is a random vector, i.e. a realization of a random variable <sub>H</sub>. A vector valued random variable is a map $\boldsymbol { H } : \Omega \to \mathbb { R } ^ { m }$ on some probability space <sub>Ω</sub> (which will actually not play a role). On <sub>Ω</sub> there is probability measure <sub>P</sub> and the random variable <sub>H</sub> generates a probability distribution on $\mathbb { R } ^ { m }$ by 

$$
\mu (B) = P (H ^ {- 1} (B))
$$

for all Borel sets $B \subset \mathbb { R } ^ { m }$ . However, <sub>P</sub> is never needed in practice and we only need to know the probability distribution $\pi _ { \mathrm { n o i s e } }$ in $\mathbb { R } ^ { m }$ since then 

$$
\mu (B) = P (H \in B) = \int_ {B} \pi_ {\text { noise }} (\eta) \mathrm{d} \eta .
$$

The mean value (or expectation) of <sub>H</sub> is 

$$
\mathbb {E} (H) = \int_ {\mathbb {R} ^ {m}} \eta \mathrm{d} \pi_ {\text { noise }} (\eta) = \int \eta \pi_ {\text { noise }} (\eta) \mathrm{d} \eta
$$

and the covariance is 

$$
\operatorname{cov} (H) = \mathbb {E} ((H - \mathbb {E} (H)) (H - \mathbb {E} (H)) ^ {T}) = \int (\eta - E (H)) (\eta - E (H)) ^ {T} \pi_ {\mathrm{noise}} (\eta) \mathrm{d} \eta .
$$

<sub>Example</sub> 12.1 (Gaussian noise)<sub>.</sub> The most simple (and also most widely used) example of additive noise is Gaussian noise. Usually we assume that the noise has zero mean and for simplicity we assume that the components of the random vector are independent and identically distributed (i.i.d), each with variance $\sigma ^ { 2 }$ . Then the probability distribution of one entry of <sub>H</sub> is 

$$
\frac {1}{\sigma \sqrt {2 \pi}} e ^ {\frac {- x ^ {2}}{2 \sigma^ {2}}}.
$$

Hence, the full probability distribution of <sub>H</sub> is 

$$
\begin{array}{r} \pi_ {\mathrm{noise}} (\eta) = \prod_ {i = 1} ^ {m} \frac {1}{\sigma \sqrt {2 \pi}} e ^ {\frac {- \eta_ {i} ^ {2}}{2 \sigma^ {2}}} \\ = \frac {1}{\sigma^ {m} \sqrt {2 \pi} ^ {m}} e ^ {\frac {- \| \eta \| ^ {2}}{2 \sigma^ {2}}}. \end{array}
$$

The covariance of <sub>H</sub> is $\mathrm { c o v } ( H ) = \sigma ^ { 2 } I _ { m }$ (where $I _ { m }$ is the $m \times m$ identity matrix). △ 

$$
\text { Inverse   Problems } \mid \text { Version   of   June   15,2023 } \mid \text { SoSe   2022 }\tag{65}
$$

The goal of probabilistic approaches is to gain as much information as possible about the <sub>posterior</sub> <sub>distribution</sub> $\pi _ { \mathrm { p o s t e r i o r } } ( x \mid y )$ i.e. the distribution of the solution $x ,$ given that the data <sub>y</sub> has been observed. The posterior distribution is, by Bayes theorem, 

$$
\pi_ {\text { posterior }} (x \mid y) = \frac {\pi (y \mid x) \pi_ {\text { prior }} (x)}{\pi (y)}.
$$

where $\pi _ { \mathrm { p r i o r } } ( x )$ <sup>is</sup> <sup>the</sup> <sup>so-called</sup> prior distribution<sup>,</sup> $\pi ( y \mid x )$ is the probability of measuring the data <sub>y</sub> given that <sub>x</sub> is the solution and $\pi ( y )$ is the probability of the data <sub>y</sub>. 

Example $\mathbf { 1 2 . 2 }$ (Maximum a-posteriori estimation)<sub>.</sub> One crucial information that one can get from the posterior distribution is the mode of the distribution which is nothing else that the maximum $x ^ { * } \in \operatorname { a r g m a x } _ { x } \pi _ { \mathrm { p o s t e r i o r } } ( x \mid y )$ . This $x ^ { * }$ is the most likely <sub>x</sub> given the measured data <sub>y</sub> and the assumptions we made. This gives us a <sub>point</sub> <sub>estimate</sub> for our solution <sub>x</sub> and this one is called the <sub>maximum</sub> <sub>a-posteriori</sub> <sub>estimator</sub> or MAP estimator. The computation of the MAP estimator amounts the solution of a maximization problem in <sub>n</sub> dimensions. $\bigtriangleup$ 

The MAP estimator may be the most likely $x ^ { * }$ , but is it is not necessarily a typical one. One example of this phenomenon already occurs for very simple distributions: A standard Gaussian distribution has its mode at zero while a sample that you draw has expected norm $\sqrt { n }$ in <sub>n</sub> dimensions. One can even show that it holds that the probability that the norm $\| x \| _ { 2 }$ of a vector with independent standard Gaussian entries deviates from $\sqrt { n }$ is very small, more precisely 

$$
P \left(\left| \| x \| _ {2} - \sqrt {n} \right| \geq t\right) \leq 2 \exp (- c t ^ {2})
$$

for some <sub>c</sub>. 

<sub>Example</sub> 12.3 (Conditional mean estimator)<sub>.</sub> Another point estimate of a distribution is its mean/expected value. Hence, we could also consider the so-called <sub>conditional</sub> <sub>mean</sub> of the posterior which is 

$$
x ^ {*} = \mathbb {E} (x \mid y) = \int_ {\mathbb {R} ^ {n}} x \pi (x \mid y) \mathrm{d} x
$$

(provided that the integral exists). The computation of the conditional mean amount to the computation of <sub>n</sub> integrals (recall that $x \in \mathbb { R } ^ { n } )$ over the full $\mathbb { R } ^ { n }$ . Since <sub>n</sub> is the dimension of the solution, this is by no means an easy task and standard approximation techniques for integrals (such as the trapezoidal rule) can not be applied. $\bigtriangleup$ 

In the following we will only consider the MAP estimate further. For the computation of the MAP estimator we maximize $\pi _ { \mathrm { p o s t e r i o r } } ( x | y )$ over <sub>x</sub>. Using Bayes theorem we note that we do not need to know anything about $\pi ( y )$ , but only $\pi ( y \mid x )$ and $\pi _ { \mathrm { p r i o r } } ( x )$ are needed. 

△ 

Example $\mathbf { 1 2 . 4 }$ (Additive Gaussian noise again)<sub>.</sub> If we assume that our solution <sub>X</sub> and the noise <sub>H</sub> are independent, the probability density of <sub>H</sub> does not change, when we condition it on the realization $X = x$ . Since $y = K x + \eta$ we also see that the <sub>Y</sub> conditioned on $X = x$ is distributed like <sub>H</sub> but translated by <sub>Kx</sub>, i.e. 

$$
\pi (y \mid x) = \pi_ {\mathrm{noise}} (y - K x).\tag{*}
$$

In the case a Gaussian noise as above we get 

$$
\pi (y \mid x) = \frac {1}{\sigma^ {m} \sqrt {2 \pi^ {m}}} e ^ {\frac {- \| y - K x \| ^ {2}}{2 \sigma^ {2}}}.
$$

Collection what we have so far we see that we still need to specify the prior distribution for <sub>x</sub>. This is up to us; we can design a prior distribution in any way we like. More precisely, we should design the prior such that it reflects all the prior information that we have about the solution. Once we have the prior, we can start thinking about how to compute the MAP estimator. If we assume a Gaussian prior, we actually end up with Tikhonov regularization: 

<sub>Theorem</sub> <sub>12.5</sub> (MAP for Gaussian prior and Gaussian noise gives Tikhonov regularization)<sub>. Assume</sub> <sub>that</sub> $K \in \mathbb { R } ^ { m \times n } , y ^ { \delta } \in \mathbb { R } ^ { m }$ be the data, $x _ { 0 }$ be an initial guess and $\sigma , \tau > 0$ . Further let the distribution of the noise and the prior be 

$$
\begin{array}{r} \pi_ {\mathrm{noise}} (\eta) = \frac {1}{\sigma^ {m} \sqrt {2 \pi^ {m}}} e ^ {\frac {- \| \eta \| ^ {2}}{2 \sigma^ {2}}} \\ \pi_ {\mathrm{prior}} (x) = \frac {1}{\tau^ {n} \sqrt {2 \pi^ {n}}} e ^ {\frac {- \| x - x _ {0} \| ^ {2}}{2 \tau^ {2}}}. \end{array}
$$

Then the MAP estimator for x from $y ^ { \delta }$ is 

$$
x ^ {*} = (K ^ {*} K + \frac {\sigma^ {2}}{\tau^ {2}} \mathbf {i d}) ^ {- 1} (K ^ {*} y ^ {\delta} + x ^ {0})
$$

<sub>Proof.</sub> The posterior distribution is 

$$
\pi (x \mid y ^ {\delta}) \propto \pi (y ^ {\delta} \mid x) \pi_ {\mathrm{prior}} (x)
$$

Using (*) and the definition of $\pi _ { \mathrm { n o i s e } }$ and $\pi _ { \mathrm { p r i o r } }$ we get 

$$
\begin{array}{r l} & {\pi (x \mid y ^ {\delta}) \propto \pi_ {\mathrm{noise}} (y ^ {\delta} - K x) \pi_ {\mathrm{prior}} (x)} \\ & {\qquad = \frac {1}{\sigma^ {m} \sqrt {2 \pi^ {m}}} e ^ {\frac {- \| y ^ {\delta} - K x \| ^ {2}}{2 \sigma^ {2}}} \frac {1}{\tau^ {n} \sqrt {2 \pi^ {n}}} e ^ {\frac {- \| x - x _ {0} \| ^ {2}}{2 \tau^ {2}}}.} \end{array}
$$

To maximize this we equivalently maximize the logarithm of $\cdot _ { \pi ( x \mid }$ $y ^ { \delta } )$ (since everything is positive and the logarithm is monotone). This gives us the maximization problem. 

$$
\begin{array}{c} \underset {x} {\operatorname{argmax}} \log \left(\frac {1}{\sigma^ {m} \sqrt {2 \pi} ^ {m}} e ^ {\frac {- \| y ^ {\delta} - K x \| ^ {2}}{2 \sigma^ {2}}} \frac {1}{\tau^ {n} \sqrt {2 \pi} ^ {n}} e ^ {\frac {- \| x - x _ {0} \| ^ {2}}{2 \tau^ {2}}}\right) \\ = \underset {x} {\operatorname{argmax}} \left[ - m \log (\sigma \sqrt {2 \pi}) - \frac {\| K x - y ^ {\delta} \| ^ {2}}{2 \sigma^ {2}} - n \log (\tau \sqrt {2 \pi}) - \frac {\| x - x _ {0} \| ^ {2}}{2 \tau^ {2}} \right]. \end{array}
$$

Since we only maximize with respect to <sub>x</sub> we can neglect the additive terms that do not depend on <sub>x</sub> and also scale by positive numbers o get 

$$
\begin{array}{l} x ^ {*} \in \underset {x} {\operatorname{argmax}} - \frac {\| K x - y ^ {\delta} \| ^ {2}}{2 \sigma^ {2}} - \frac {\| x - x _ {0} \| ^ {2}}{2 \tau^ {2}} \\ = \underset {x} {\operatorname{argmin}} \frac {\| K x - y ^ {\delta} \| ^ {2}}{2 \sigma^ {2}} + \frac {\| x - x _ {0} \| ^ {2}}{2 \tau^ {2}} \\ = \underset {x} {\operatorname{argmin}} \frac {1}{2} \| K x - y ^ {\delta} \| ^ {2} + \frac {\sigma^ {2}}{2 \tau^ {2}} \| x - x _ {0} \| ^ {2}. \end{array}
$$

We recognize the Tikhonov functional with regularization parameter $\frac { \sigma } { \tau }$ . The minimizer $x ^ { * }$ is given by the solution of 

$$
K ^ {*} (K x ^ {*} - y ^ {\delta}) + \frac {\sigma^ {2}}{\tau^ {2}} (x ^ {*} - x ^ {0}) = 0
$$

which proves the claim. 

□ 

The choice of the prior distribution is basically an art. Many suitable priors exists for various types of data. If we consider additive Gaussian noise as above and a prior of the form $\pi _ { \mathrm { p r i o r } } ( x ) =$ $e ^ { - \alpha \Phi ( x ) }$ one get, similarly to the above theorem, that the MAP estimate is gives as a solution of the minimization problem 

$$
\min _ {x} \frac {1}{2} \| K x - y ^ {\delta} \| ^ {2} + \frac {\sigma^ {2} \alpha}{2} \Phi (x).
$$

The noise distribution, however, is dictated by the noise model and if one does not have additive Gaussian noise, one can still formulate a regularization method: 

<sub>Example</sub> 12.6 (Poisson noise)<sub>.</sub> The Poisson distribution models rare events. One example comes from photography-like applications with very low light as it occurs, for example, in electron microscopy. There each pixel collects incoming photos over a short time span. The number of incoming photons in some pixel $p ,$ when measured and averaged over a long time span, gives the true intensity $y ( p )$ Over a short time span, one collects a finite number of photons and the stochastic model for this number is that it is distributed according to the Poisson distribution with parameter $\lambda = y ( p )$ (the parameter <sub>λ</sub> is also the expected value of the distribution), this means that the probability to collect $\boldsymbol { y } ^ { \delta } ( \boldsymbol { p } ) = \boldsymbol { k }$ photons in pixel $p$ is 

$$
P (y ^ {\delta} (p) = k) = \frac {y (p) ^ {k} e ^ {- y (p)}}{k !}.
$$

Hence, the conditional probability that $y ^ { \delta } ( p )$ is measured if $y ( p )$ is the true value is 

$$
\pi (y ^ {\delta} (p) \mid y (p)) = \frac {y (p) ^ {y ^ {\delta} (p)} e ^ {- y (p)}}{(y ^ {\delta} (p) !}.
$$

We still assume that the noise is the pixels is independent, i.e we have 

$$
\pi (y ^ {\delta} \mid y) = \prod_ {p} \frac {y (p) ^ {y ^ {\delta} (p)} e ^ {- y (p)}}{(y ^ {\delta} (p) !}.
$$

If we assume that the image prior $\pi _ { \mathrm { p r i o r } }$ is of the form $\pi _ { \mathrm { p r i o r } } ( x ) \propto$ $e ^ { - \alpha \Phi ( x ) }$ for some function $\Phi ,$ the MAP estimate for <sub>x</sub> with $y = K x$ from $y ^ { \delta }$ (where $y ^ { \delta }$ is a version of $\dot { y }$ that is corrupted by Poisson noise) is 

$$
\underset {x} {\operatorname{argmax}} \pi (y ^ {\delta} \mid K x) \pi_ {\text {prior}} (x) = \underset {x} {\operatorname{argmax}} \prod_ {p} \frac {(K x (p)) ^ {y ^ {\delta} (p)} e ^ {- K x (p)}}{(y ^ {\delta} (p) !} \cdot e ^ {- \alpha \Phi (x)}.
$$

Equivalently, we minimize the negative logarithm of the objective which is 

$$
\begin{array}{c} \underset {x} {\operatorname{argmin}} \left[ - \log (\pi (y ^ {\delta} \mid K x)) - \log (\pi_ {\text { prior }} (x)) \right] \\ = \underset {x} {\operatorname{argmin}} \sum_ {p} \left[ - y ^ {\delta} (p) \log (K x (p)) + K x (p) \right] + \alpha \Phi (x). \end{array}
$$

The negative log of the noise prior may look strange at first. However, note that the function $f _ { a } ( t ) = - a \log ( t ) +$ t has a unique minimum at $t = a$ (here for $a = 1 . 5 )$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/26dfbc3d4272c1d6396fa68de22618c40a2497d6b4bb6dd05ff914e7a55c889e.jpg)


## 13 Discretization by projection

In our last section we finally treat some discretization methods. 

The backbone of discretization are projection operators. A bounded linear operator $P : X \to X$ on a normed space is a projection onto a subspace $U \operatorname { i f } P x \in U$ for all $x \in X$ and $P x = x$ $\mathrm { f o r } x \in U$ . Moreover it holds that $P ^ { 2 } = P$ and $\| \cal P \| \geq 1$ . If <sub>X</sub> is a Hilbert space and <sub>P</sub> is self-adjoint, then <sub>P</sub> is an orthonormal projection and it holds for all $u \in U$ that 

$$
\left\| P x - x \right\| \leq \left\| u - x \right\|,
$$

i.e. <sub>Px</sub> is the best approximation from <sub>U</sub> to <sub>x</sub>. 

Definition 13.1. Let X, Y be Banach spaces, $K : X  Y$ be bounded and linear and $X _ { n } \subset X , Y _ { m } \subset Y$ be <sub>n</sub>- and <sub>m</sub>-dimensional subspaces, respectively. Further, let $Q _ { m } : Y  Y _ { m }$ be a projection onto $Y _ { m }$ <sup>.</sup> <sup>The</sup> projection method <sup>for</sup> <sup>solving</sup> $K x = y$ is to solve the problem 

$$
Q _ {m} K x _ {n} = Q _ {m} y, \text {   for   } x _ {n} \in X _ {n}.
$$

If we choose bases $\big \{ \hat { x } _ { 1 } , \dots , \hat { x } _ { n } \big \}$ and $\big \{ \hat { y } _ { 1 } , \dots , \hat { y } _ { m } \big \}$ of $X _ { n }$ and $Y _ { m }$ , respectively, we can write 

$$
Q _ {n} y = \sum_ {i = 1} ^ {n} \beta_ {i} \hat {y} _ {i}, \quad \text { and } \quad Q _ {n} K \hat {x} _ {j} = \sum_ {i = i} ^ {n} A _ {i j} \hat {y} _ {i}.
$$

The solution $x _ { n }$ can written as $\scriptstyle \sum _ { j = 1 } ^ { n } \alpha _ { j } { \hat { x } } _ { j }$ and thus, the coeficients can be determined by the linear system 

$$
\sum_ {j = 1} ^ {n} A _ {i j} \alpha_ {j} = \beta_ {i}.
$$

Plugging the expansion for $x _ { n }$ into $Q _ { m } \bar { K } x _ { n } = Q _ { m } y$ gives 

Example $\mathbf { 1 } 3 . 2$ (Galerkin method)<sub>.</sub> Let <sub>X</sub> and <sub>Y</sub> be Hilbert spaces, $X _ { n } , Y _ { m }$ as above and $Q _ { m }$ an orthogonal projection onto $Y _ { m } .$ . The equation $Q _ { m } K x _ { n } = Q _ { m } y$ is then equivalently expressed as the so-called Galerkin equations 

$$
\sum_ {j = 1} ^ {n} \alpha_ {j} Q _ {m} K \hat {x} _ {j} = \sum_ {i = 1} ^ {m} \beta_ {i} \hat {y} _ {i}
$$

$$
\langle K x _ {n}, z _ {m} \rangle = \langle y, z _ {m} \rangle \quad \text { for   all } z _ {m} \in Y _ {m}.\tag{*}
$$

and using $Q _ { n } K \hat { x } _ { j } = \sum _ { i = i } ^ { n } A _ { i j } \hat { y } _ { i }$ gives the result. 

Choosing bases as above gives us the system 

$$
\sum_ {j = 1} ^ {n} \alpha_ {j} \underbrace {\left\langle K \hat {x} _ {j} , \hat {y} _ {i} \right\rangle} _ {=: A _ {i j}} = \underbrace {\left\langle y , \hat {y} _ {i} \right\rangle} _ {=: \beta_ {i}}, \quad i = 1, \ldots , m.\tag{**}
$$

△ 

<sub>Example</sub> 13.3 (Collocation method)<sub>.</sub> Here we have any Banach space <sub>X</sub> but fix $\begin{array} { r } { \dot { Y } = C ( [ a , b ] ) } \end{array}$ . We choose so-called <sub>collocation</sub> <sub>points</sub> $a = t _ { 1 } < \cdots < t _ { m } = b$ and consider the subspace $Y _ { m }$ as the space of functions that are continuous and piecewise linear on the intervals $[ t _ { i } , t _ { i + 1 } ]$ (also known as the space of linear splines). As operator $Q _ { m }$ we take the “linear interpolation operator”, i.e. 

$$
Q _ {m} y = \sum_ {i = 1} ^ {m} y (t _ {i}) \hat {y} _ {i}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/ffe26390f33c99e1487656c1a6b6160a88ef05227c2ac213a8c6d08114b8f993.jpg)


where the $\hat { y } _ { i }$ are the linear splines which are <sub>1</sub> at $t _ { i }$ and zero at $t _ { j }$ with $j \neq i .$ . The projected equation $Q _ { m } K x _ { n } = Q _ { m } y$ is then equivalent to 

$$
(K x _ {n}) (t _ {i}) = y (t _ {i}), \quad i = 1, \dots , n.
$$

We choose an <sub>n</sub>-dimensional subspace $X _ { n }$ of <sub>X</sub> and a basis $\{ \hat { x } _ { j } ~ \}$ $j = 1 , \dots , n \}$ of $X _ { n }$ . Then we can express $x _ { n } ~ \in ~ X _ { n }$ by $x _ { n } =$ $\scriptstyle \sum _ { j = 1 } ^ { n } \alpha _ { j } { \hat { x } } _ { j }$ . The collocations equations for $x _ { n } \in X _ { n }$ then become 

$$
K \sum_ {j = 1} ^ {n} \alpha_ {j} \hat {x} _ {j} (t _ {i}) = y (t _ {i}).
$$

The left hand side is $\textstyle \sum _ { j = 1 } ^ { n } K { \hat { x } } _ { j } ( t _ { i } ) \alpha _ { j }$ and we see that the collocation equations are equivalent to $A \alpha = \beta$ with 

$$
\beta_ {i} = y (t _ {i}), \quad A _ {i j} = K \hat {x} _ {j} (t _ {i}).
$$

Example $^ { 1 3 . 4 }$ (Galerkin and collocation for integral equations)<sub>.</sub> Now consider the special case $K : L ^ { 2 } ( [ a , b ] )  L ^ { 2 } ( [ c , d ] )$ 

$$
K x (t) = \int_ {a} ^ {b} k (t, s) x (s) \mathrm{d} s = y (t), \quad t \in [ c, d ].
$$

The Galerkin method uses the values (cf. (**)) 

$$
A _ {i j} = \int_ {c} ^ {d} \int_ {a} ^ {b} k (t, s) \hat {x} _ {j} (s) \hat {y} _ {i} (t) d s d t, \quad \beta_ {i} = \int_ {c} ^ {d} y (t) \hat {y} _ {i} (t) d t
$$

while the collocation method uses 

$$
A _ {i j} = \int_ {a} ^ {b} k (t _ {i}, s) \hat {x} _ {j} (s) d s, \quad \beta_ {i} = y (t _ {i}).
$$

Note that the entries for the Galerkin method are harder to compute (more integrals…). $\bigtriangleup$ 

In the following we will assume $m = n$ and that the following conditions are fulfilled: <sub>K</sub> is injective, the union $\textstyle \bigcup _ { n } X _ { n }$ is dense in <sub>X</sub> and $Q _ { n } K | _ { X _ { n } } : X _ { n } \to Y _ { n }$ is invertible. 

Then a solution of $Q _ { n } K x _ { n } = Q _ { n } y$ exists and is given by 

$$
x _ {n} = R _ {n} y _ {n}, \quad \text { with } \quad R _ {n} := (Q _ {n} K | _ {X _ {n}}) ^ {- 1} Q _ {n}: Y \to X _ {n}.
$$

We say that the projection method is <sub>convergent</sub> if for every $x \in X$ it holds that 

$$
R _ {n} K x \stackrel {n \to \infty} {\longrightarrow} x.
$$

Note that what we are doing here is to consider projection methods as regularizations! We can consider $\alpha = 1 / n$ as regularization parameter. 

Not every projection method converges, but there is a simple condition that ensures convergence: 

Theorem 13.5. Under our standing assumptions it holds that $x _ { n } = R _ { n } y$ converges to x for every $y = K x$ exactly if there exists $c > 0$ such that 

$$
\| R _ {n} K \| \leq c.\tag{+}
$$

Moreover, if this is fulfilled, then (with the same c) 

$$
\| x _ {n} - x \| _ {X} \leq (1 + c) \min _ {z _ {n} \in X _ {n}} \| z _ {n} - x \| _ {X}.
$$

<sub>Proof.</sub> First assume that the method converges, i.e. that $R _ { n } K x \stackrel { n \to \infty } { \longrightarrow }$ <sub>x</sub> for every <sub>x</sub>. Then the assertion follows from the uniform boundedness principle. 

For us, the other direction is more interesting: Let $\| R _ { n } K \|$ be bounded. For $z _ { n } \in X _ { n }$ we have that 

$$
R _ {n} K z _ {n} = (Q _ {n} K | _ {X _ {n}}) ^ {- 1} Q _ {n} K z _ {n} = (Q _ {n} K | _ {X _ {n}}) ^ {- 1} Q _ {n} K | _ {X _ {n}} z _ {n} = z _ {n}
$$

and thus, $R _ { n } K$ is a projection. We conclude that 

$$
x _ {n} - x = (R _ {n} K - \mathrm{id}) x = (R _ {n} K - \mathrm{id}) (x - z _ {n}).
$$

We obtain $\| x _ { n } - x \| _ { X } \leq ( c + 1 ) \| x - z _ { n } \| _ { X }$ and taking the minimum over all $z _ { n }$ shows the inequality. The convergence $x _ { n } \stackrel { n \to \infty } { \longrightarrow } x$ follows since $\textstyle \bigcup _ { n } X _ { n }$ is dense in <sub>X</sub>. □ 

Here is an error estimate for the Galerkin method. To express it, <sup>we</sup> <sup>define</sup> <sup>the</sup> synthesis operator $S _ { n } ^ { X } : \mathbb { R } ^ { n } \to X$ in <sub>X</sub> by $\begin{array} { r } { S _ { n } ^ { X } \alpha = \mathbf { \bar { Z } } _ { j } \alpha _ { j } \hat { x } _ { j } } \end{array}$ and similarly for $S _ { n } ^ { Y }$ . We define the quantities 

$$
a _ {n} = \left\| S _ {n} ^ {X} \right\| = \max \left\{\left\| S _ {n} ^ {X} \alpha \right\| _ {X} \mid \| \alpha \| _ {2} = 1 \right\}
$$

If we choose orthonormal bases, then we get $a _ { n } = \| S _ { n } ^ { X } \| = 1$ and also $b _ { n } =$ 1. 

$$
b _ {n} = \max \left\{\| \beta \| _ {2} \mid \| S _ {n} ^ {Y} \beta \| _ {Y} = 1 \right\}
$$

Theorem 13.6. Assume that the Galerkin equations $( ^ { \star } )$ from Example 13.2 are uniquely solvable. 

(a) Let $y ^ { \delta } \in Y$ with $\| y - y ^ { \delta } \| _ { Y } \leq \delta$ and $x _ { n } ^ { \delta }$ be the solution of 

$$
\left\langle K x _ {n} ^ {\delta}, z _ {n} \right\rangle = \left\langle y ^ {\delta}, z _ {n} \right\rangle \quad f o r a l l z _ {n} \in Y _ {n}.
$$

Then it holds that 

$$
\| x _ {n} ^ {\delta} - x \| _ {X} \leq \| R _ {n} \| \delta + \| R _ {n} K x - x \| _ {X}.
$$

(b) Let A and $\beta$ be given by $( ^ { \star \star } ) .$ from Example 13.2 and let $\| { \boldsymbol { \beta } } - { \boldsymbol { \beta } } ^ { \delta } \| \leq \delta$ hold and let $\lambda _ { n }$ be the smallest singular value of A. Let $\alpha ^ { \delta }$ be the solution of $A \alpha ^ { \delta } = \beta ^ { \delta }$ and define $\begin{array} { r } { x _ { n } ^ { \bar { \delta } } = \sum _ { j = 1 } ^ { n } \alpha _ { j } \hat { x _ { j } } } \end{array}$ . Then it holds 

$$
\| x _ {n} ^ {\delta} - x \| _ {X} \leq \frac {a _ {n}}{\lambda_ {n}} \delta + \| R _ {n} K x - x \| _ {X}
$$

<sub>Proof.</sub> For part (a) we simply use the standard error decomposition 

$$
\| x _ {n} ^ {\delta} - x \| _ {X} \leq \| x _ {n} ^ {\delta} - R _ {n} y \| + \| R _ {n} y - x \| \leq \| R _ {n} \| \| y ^ {\delta} - y \| _ {Y} + \| R _ {n} K x - x \| _ {X}
$$

from which the estimate follows. 

For part (b) we just need to estimate the data error in the above decomposition. We write $\begin{array} { r } { R _ { n } x = \sum _ { j = 1 } ^ { n } \alpha _ { j } \hat { x } _ { j } } \end{array}$ . Since $x _ { n } ^ { \delta } - R _ { n } y =$ $\Sigma _ { j = 1 } ^ { n } ( \alpha _ { j } ^ { \delta } - \alpha _ { j } ) \hat { x } _ { j } = S _ { n } ^ { X } ( \alpha _ { n } ^ { \delta } - \alpha )$ we get 

$$
\| x _ {n} ^ {\delta} - R _ {n} y \| _ {X} \leq a _ {n} \| \alpha_ {n} ^ {\delta} - \alpha \| _ {2} = a _ {n} \| A ^ {- 1} (\beta^ {\delta} - \beta) \| _ {2} \leq \frac {a _ {n}}{\lambda_ {n}} \delta
$$

as desired. 

<sub>Example</sub> 13.7 (Collocation method for the inverse integration problem)<sub>.</sub> We consider the simple problem $\begin{array} { r } { K x ( t ) = \int _ { 0 } ^ { t } x ( s ) } \end{array}$ <sub>ds</sub> with $K : \dot { C } ( [ 0 , 1 ] ) \to C ( [ 0 , 1 ] )$ . We have to choose the collocation points $t _ { i }$ and the basis $\hat { x } _ { j }$ of $X _ { n }$ . Once we have done this, the linear equation $A \alpha = \beta$ is given by $\beta _ { i } = y ( t _ { i } )$ and 

$$
A _ {i j} = K \hat {x} _ {j} (t _ {i}) = \int_ {0} ^ {t _ {i}} \hat {x} _ {j} (s) \mathrm{d} s.
$$

First we choose the basis $\hat { x } _ { j }$ . Let us choose the characteristic functions on the intervals $\begin{array} { r } { I _ { j } = [ \frac { j - 1 } { n } , \frac { j } { n } [ : } \end{array}$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-24/7ea2762e-1d09-4041-bcc2-84aace2f7119/2899eb474cb03f027843e1ec167114156a281681c7859cc0647f371766bd8686.jpg)


The functions $K \hat { x } _ { j }$ are 

$$
K \hat {x} _ {j} (t) = \left\{ \begin{array}{l l} 0 & : t \leq \frac {j - 1}{n} \\ t - \frac {j - 1}{n} & : \frac {j - 1}{n} \leq t \leq \frac {j}{n} \\ \frac {1}{n} & : t \geq \frac {j}{n} \end{array} \right. \cdot \begin{array}{c c} \uparrow & K \hat {x} _ {1} \\ \hline \frac {1}{n} & \frac {1}{n} - \frac {1}{n} - 1 \\ \hline 1 & 1 \end{array}
$$

Depending on the collocation points we get diferent linear systems: 

1. We choose $\begin{array} { r } { t _ { i } = \frac { i - 1 } { n } , i = 1 , \dots , n + 1 } \end{array}$ , i.e. we have $m = n + 1$ This gives us 

$$
A _ {i j} = K \hat {x} _ {j} (x _ {i}) = \left\{ \begin{array}{l l l} 0 & : & i \leq j \\ \frac {1}{n} & : & \text {else.} \end{array} \right., \quad A = \frac {1}{n} \left( \begin{array}{c c c} 0 & & \\ 1 & \ddots & \\ \vdots & \ddots & 0 \\ 1 & \dots & 1 \end{array} \right) \in \mathbb {R} ^ {n + 1 \times n}.
$$

2. As a variant of the first choice we could only take the left ends, i.e. $\begin{array} { r } { t _ { i } = \frac { i - 1 } { n } , i = 1 \ldots , n } \end{array}$ , or the right ends $\begin{array} { r } { t _ { i } = \frac { i } { n } , } \end{array}$ $i = 1 , \ldots , n$ and get 

$$
A = \frac {1}{n} \left( \begin{array}{c c c c} 0 & & & \\ 1 & \ddots & & \\ \vdots & \ddots & \ddots & \\ 1 & \dots & 1 & 0 \end{array} \right) \in \mathbb {R} ^ {n \times n}, \quad A = \frac {1}{n} \left( \begin{array}{c c c} 1 & & 0 \\ \vdots & \ddots & \\ 1 & \dots & 1 \end{array} \right) \in \mathbb {R} ^ {n \times n},
$$

respectively. 

3. We choose the middle points of the intervals $\begin{array} { r } { t _ { i } = \frac { i - \frac { 1 } { 2 } } { n } } \end{array}$ . This way we get 

$$
A _ {i j} = \frac {1}{n} \left( \begin{array}{c c c c} \frac {1}{2} & & & \\ 1 & \ddots & & \\ \vdots & \ddots & \ddots & \\ 1 & \dots & 1 & \frac {1}{2} \end{array} \right) \in \mathbb {R} ^ {n \times n}
$$