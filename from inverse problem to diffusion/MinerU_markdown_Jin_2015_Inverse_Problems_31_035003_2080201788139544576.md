PAPER • OPEN ACCESS 

# A tutorial on inverse problems for anomalous diffusion processes

To cite this article: Bangti Jin and William Rundell 2015 31 035003 

View the article online for updates and enhancements. 

You may also like 

An inverse random source problem for thetime fractional diffusion equation driven by a fractional Brownian motion Xiaoli Feng, Peijun Li and Xu Wang 

First passage time moments ofasymmetric Lévy flights Amin Padash, Aleksei V Chechkin, Bartomiej Dybiec et al. 

An undetermined coefficient problem for afractional diffusion equation Zhidong Zhang 

Inverse Problems 31 (2015) 035003 (40pp) 

# A tutorial on inverse problems for anomalous diffusion processes

Bangti Jin<sup>1</sup> and William Rundell<sup>2</sup> 

<sup>1</sup> Department of Computer Science, University College London, Gower Street, London WC1E 6BT, UK 

<sup>2</sup> Department of Mathematics, Texas A&M University, College Station, TX 77843- 3368, USA 

E-mail: bangti.jin@gmail.com and rundell@math.tamu.edu 

Received 26 September 2014, revised 11 November 2014 

Accepted for publication 25 November 2014 

Published 10 February 2015 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/59efc0105c453ddd8db14610209b974d7e3f2059d8f6395788cfc12dd7b3fe9e.jpg)


## Abstract

Over the last two decades, anomalous diffusion processes in which the mean squares variance grows slower or faster than that in a Gaussian process have found many applications. At a macroscopic level, these processes are adequately described by fractional differential equations, which involves fractional derivatives in time or/and space. The fractional derivatives describe either history mechanism or long range interactions of particle motions at a microscopic level. The new physics can change dramatically the behavior of the forward problems. For example, the solution operator of the time fractional diffusion diffusion equation has only limited smoothing property, whereas the solution for the space fractional diffusion equation may contain weak singularity. Naturally one expects that the new physics will impact related inverse problems in terms of uniqueness, stability, and degree of ill-posedness. The last aspect is especially important from a practical point of view, i.e., stably reconstructing the quantities of interest. In this paper, we employ a formal analytic and numerical way, especially the two-parameter Mittag-Lef<sup>fl</sup>er function and singular value decomposition, to examine the degree of illposedness of several ‘classical’ inverse problems for fractional differential equations involving a Djrbashian–Caputo fractional derivative in either time or space, which represent the fractional analogues of that for classical integral order differential equations. We discuss four inverse problems, i.e., backward fractional diffusion, sideways problem, inverse source problem and inverse potential problem for time fractional diffusion, and inverse Sturm–Liouville problem, Cauchy problem, backward fractional diffusion and sideways problem for space fractional diffusion. It is found that contrary to the wide belief, the in<sup>fl</sup>uence of anomalous diffusion on the degree of ill-posedness is not de<sup>fi</sup>nitive: it can either signi<sup>fi</sup>cantly improve or worsen the conditioning of related inverse problems, depending crucially on the speci<sup>fi</sup>c type of given data and quantity of interest. Further, the study exhibits distinct new features of ‘fractional’ inverse problems, and a partial list of surprising observations is given below. (a) Classical backward diffusion is exponentially ill-posed, whereas time fractional backward diffusion is only mildly ill-posed in the sense of norms on the domain and range spaces. However, this does not imply that the latter always allows a more effective reconstruction. (b) Theoretically, the time fractional sideways problem is severely ill-posed like its classical counterpart, but numerically can be nearly well-posed. (c) The classical Sturm–Liouville problem requires two pieces of spectral data to uniquely determine a general potential, but in the fractional case, one single Dirichlet spectrum may suf<sup>fi</sup>ce. (d) The space fractional sideways problem can be far more or far less ill-posed than the classical counterpart, depending on the location of the lateral Cauchy data. In many cases, the precise mechanism of these surprising observations is unclear, and awaits further analytical and numerical exploration, which requires new mathematical tools and ingenuities. Further, our <sup>fi</sup>ndings indicate fractional diffusion inverse problems also provide an excellent case study in the differences between theoretical ill-conditioning involving domain and range norms and the numerical analysis of a <sup>fi</sup>nite-dimensional reconstruction procedure. Throughout we will also describe known analytical and numerical results in the literature. 

Keywords: fractional inverse problem, fractional differential equation, anomalous diffusion, Djrbashian–Caputo fractional derivative, Mittag-Lef<sup>fl</sup>er function 

(Some <sup>fi</sup>gures may appear in colour only in the online journal) 

## 1. Introduction

Diffusion is one of the most prominent transport mechanisms found in nature. At a microscopic level, it is related to the random motion of individual particles, and the use of the Laplace operator and the <sup>fi</sup>rst-order derivative in the canonical diffusion model rests on a Gaussian process assumption on the particle motion, after Albert Einsteinʼs groundbreaking work [23]. Over the last two decades a large body of literature has shown that anomalous diffusion models in which the mean square variance grows faster (superdiffusion) or slower (subdiffusion) than that in a Gaussian process under certain circumstances can offer a superior <sup>fi</sup>t to experimental data (see the comprehensive reviews [5, 70, 72, 95] for physical background and practical applications). For example, anomalous diffusion is often observed in materials with memory, e.g., viscoelastic materials, and heterogeneous media, such as soil, heterogeneous aquifer, and underground <sup>fl</sup>uid <sup>fl</sup>ow. At a microscopic level, the subdiffusion process can be described by a continuous time random walk [75], where the waiting time of particle jumps follows some heavy tailed distribution, whereas the superdiffusion process can be described by Lévy <sup>fl</sup>ights or Lévy walk, where the length of particle jumps follows some heavy tailed distribution, re<sup>fl</sup>ecting the long-range interactions among particles. Following the aforementioned micro–macro correspondence, the macroscopic counterpart of a continuous time random walk is a differential equation with a fractional derivative in time, and that for a Lévy <sup>fl</sup>ight is a differential equation with a fractional derivative in space. We will refer to these two cases as time fractional diffusion and space fractional diffusion, respectively, and it is generically called a fractional derivative equation (FDE) below. In general the fractional derivative can appear in both time and space variables. 

Next we give the mathematical model in the simplest geometrical setting of one space dimension, taking the domain $\varOmega = ( 0 , 1 )$ ). Then a general, linear FDE is given by 

$$
\partial_ {t} ^ {\alpha} u - _ {0} ^ {C} D _ {x} ^ {\beta} u + q u = f (x, t) \in \Omega \times (0, T),\tag{1.1}
$$

where $T > 0$ is a <sup>fi</sup>xed time, and it is equipped with suitable boundary and initial conditions. The fractional orders $\alpha \in ( 0 , 1 )$ and $\beta \in ( 1 , 2 )$ are related to the parameters specifying the large-time behavior of the waiting-time distribution or long-range behavior of the particle jump distribution. For example, in hydrological studies, the parameter $\beta$ is used to characterize the heterogeneity of porous medium [17]. In theory, these parameters can be determined from the underlying stochastic model, but often in practice, they are determined from experimental data [34, 35, 61]. The notation $\partial _ { t } ^ { \alpha } = _ { 0 } ^ { C } \bar { D _ { t } ^ { \alpha } }$ is the Djrbashian–Caputo derivative operator of order $\alpha \in ( 0 ,$ , 1) in the time variable t, and ${ } _ { 0 } ^ { C } D _ { x } ^ { \beta }$ denotes the Djrbashian– Caputo derivative of order $\beta \in ( 1 , 2 )$ in the space variable x. For a real number $n - 1 < \gamma < n , \ n \in \mathbb { N } .$ , and $f \in H ^ { n } ( 0 ,$ 1), the left-sided Djrbashian–Caputo derivative ${ } _ { 0 } ^ { C } D _ { x } ^ { \gamma } f$ of order $\gamma$ is de<sup>fi</sup>ned by [53, p 91] 

$$
{ } _ { 0 } ^ { C } D _ { x } ^ { \gamma } f = \frac { 1 } { \varGamma ( n - \gamma ) } \int _ { 0 } ^ { x } ( x - s ) ^ { n - 1 - \gamma } f ^ { ( n ) } ( s ) \mathrm{d} s ,\tag{1.2}
$$

where $T ( z )$ denotes Eulerʼs Gamma function de<sup>fi</sup>ned by 

$$
\Gamma (z) = \int_ {0} ^ {\infty} s ^ {z - 1} \mathrm{e} ^ {- s} \mathrm{d} s, \quad \Re (z) > 0.
$$

The Djrbashian–Caputo derivative was <sup>fi</sup>rst introduced by Armenian mathematician Mkhitar M Djrbashian for studies on space of analytical functions and integral transforms in 1960s (see [19–21] for surveys on related works). Italian geophysicist Michele Caputo independently proposed the use of the derivative for modeling the dynamics of viscoelastic materials in 1967 [10]. We note that there are several alternative (and different) de<sup>fi</sup>nitions of fractional derivatives, notably the Riemann–Liouville fractional derivative, which formally is obtained from (1.2) by interchanging the order of integration and differentiation, i.e., the leftsided Riemann–Liouville fractional derivative ${ } _ { 0 } ^ { R } D _ { x } ^ { \gamma } { \bar { f } }$ of order $\gamma \in ( n - 1 , n ) , \ n \in \mathbb { N } .$ , is de<sup>fi</sup>ned by [53, p 70] 

$$
{ } _ { 0 } ^ { R } D _ { x } ^ { \gamma } f = \frac { \mathrm{d} ^ { n } } { \mathrm{d} x ^ { n } } \frac { 1 } { \varGamma ( n - \gamma ) } \int _ { 0 } ^ { x } ( x - s ) ^ { n - 1 - \gamma } f ( s ) \mathrm{d} s .
$$

In this work, we shall focus mostly on the Djrbashian–Caputo derivative since it allows a convenient treatment of the boundary and initial conditions. 

Under certain regularity assumption on the functions, with an integer order $\gamma ,$ the Djrbashian–Caputo and Riemann–Liouville derivatives both recover the usual integral order derivative (see for example $[ 7 9 , 9 \ 1 0 0 ]$ for the Djrbashian–Caputo case). For example, with $\alpha = 1$ and $\beta = 2$ , the Djrbashian–Caputo fractional derivatives $\partial _ { t } ^ { \alpha } u$ and ${ } _ { 0 } ^ { C } D _ { x } ^ { \beta }$ u coincide with the usual <sup>fi</sup>rst- and second-order derivatives $\frac { \partial u } { \partial t }$ and $\frac { \partial ^ { 2 } u } { \partial x ^ { 2 } }$ , respectively, for which the model (1.1) recovers the standard one-dimensional diffusion equation, and thus generally the model (1.1) is regarded as a fractional counterpart. The Djrbashian–Caputo derivative (and many others) is an integro-differential operator, and thus it is nonlocal in nature. As a consequence, many useful rules, e.g., product rule and integration by parts, from PDEs are either invalid or require signi<sup>fi</sup>cant modi<sup>fi</sup>cations. The nonlocality underlies most analytical and numerical challenges associated with the model (1.1). It signi<sup>fi</sup>cantly complicates the mathematical and numerical analysis of the model, including relevant inverse problems. 

In a fractional model, there are a number of parameters, e.g., fractional order(s), diffusion and potential coef<sup>fi</sup>cients (when using a second-order elliptic operator in space), initial condition, source term, boundary conditions and domain geometry, that cannot be measured/ speci<sup>fi</sup>ed directly, and have to be inferred indirectly from measured data. Typically, the data is the forward solution restricted to either the boundary or the interior of the physical domain. This gives rise to a large variety of inverse problems for FDEs, which have started to attract much attention in recent years, since the pioneering work [14]. An interesting question is how the nonlocal physics behind anomalous diffusion processes will in<sup>fl</sup>uence the behavior of related inverse problems, e.g., uniqueness, stability, and the degree of ill-posedness. The degree of ill-posedness is especially important for developing practical numerical reconstruction procedures. There is a now well known example of backward fractional diffusion, i.e., recovering the initial condition in a time fractional diffusion equation from the <sup>fi</sup>nal time data, which is only mildly ill-posed, instead of severely ill-posed for the classical backward diffusion problem. In some sense, this example has led to the belief that ‘fractionalizing inverse problems can always mitigate the degree of ill-posedness, and thus allows a better chance of an accurate numerical reconstruction. 

In this paper, we examine the degree of ill-posedness of ‘fractional’ inverse problems from a formal analytic and numerical point of view, and contrast their numerical stability properties with their classical, that is, the Gaussian diffusion counterparts, for which there are many deep analytical results [39, 40, 84]. Speci<sup>fi</sup>cally, we revisit a number of ‘classical’ inverse problems for the FDEs, e.g., the backward diffusion problem, sideways diffusion problem and inverse source problem, and numerically exhibit their degree of ill-posedness. These examples indicate that the answer to the aforementioned question is not de<sup>fi</sup>nitive: it depends crucially on the type (unknown and data) of the inverse problem we look at, and the nonlocality of the problem (fractional derivative) can either greatly improve or worsen the degree of ill-posedness. 

The mathematical theory of inverse problems for FDEs is still in its infancy, and thus in this work, we only discuss the topic formally to give a <sup>fl</sup>avor of inverse problems for FDEs— our goal is to give insight rather than to pursue an in-depth analysis. The technical developments that are available we leave to the references cited. In addition, known theoretical results and computational techniques in the literature will be brie<sup>fl</sup>y described, which however are not meant to be exhaustive. The rest of the paper is organized as follows. In section 2 we review two special functions, i.e., Mittag-Lef<sup>fl</sup>er function and Wright function, and their basic properties. The Mittag-Lef<sup>fl</sup>er function plays an extremely important role in understanding anomalous diffusion processes. We also recall the basic tool—singular value decomposition —for analyzing discrete inverse problems. Then in section 3 we study several inverse problems for FDEs with a time fractional derivative, including backward diffusion, inverse source problem, sideways problem and inverse potential problem. In section 4 we consider inverse problems for FDEs with a space fractional derivative, including the inverse Sturm– Liouville problem, Cauchy problem, backward diffusion and sideways problem. In the appendices, we give the implementation details of the computational methods for solving the time- and space fractional differential equations. These methods are employed throughout for computing the forward map (unknown-to-measurement map) so as to gain insight into related inverse problems. Throughout the notation $^ { c , }$ with or without a subscript, denote a generic constant, which may differ at different occurrences, but it is always independent of the unknown of interest. 

## 2. Preliminaries

We recall two important special functions, Mittag-Lef<sup>fl</sup>er function and Wright function, and one useful tool for analyzing discrete ill-posed problems, singular value decomposition. 

## 2.1. Mittag-Leffler function

We shall use extensively the two-parameter Mittag-Lef<sup>fl</sup>er function $E _ { \alpha , \beta } ( z )$ (with $\alpha > 0$ and $\beta \in \mathbb { R } )$ de<sup>fi</sup>ned by [53, equation (1.8.17), p 40] 

$$
E _ {\alpha , \beta} (z) = \sum_ {k = 0} ^ {\infty} \frac {z ^ {k}}{\Gamma (k \alpha + \beta)} z \in \mathbb {C}.\tag{2.1}
$$

This function with $\beta = 1$ was <sup>fi</sup>rst introduced by Gösta Mittag-Lef<sup>fl</sup>er in 1903 [74] and then generalized by others [1, 36]. It can be veri<sup>fi</sup>ed directly that 

$$
E _ {1, 1} (z) = \mathrm{e} ^ {z}, \quad E _ {2, 1} (z) = \cosh \sqrt {z}, \quad E _ {2, 2} (z) = \frac {\sinh \sqrt {z}}{\sqrt {z}}.
$$

Hence it represents a generalization of the exponential function in that $E _ { 1 , 1 } ( z ) = \mathbf { e } ^ { z }$ . The Mittag-Lef<sup>fl</sup>er function $E _ { \alpha , \beta } ( z )$ is an entire function of $z$ with order $\alpha ^ { - 1 }$ and type 1 [53, p 40]. Further, the function $E _ { \alpha , 1 } ( - t )$ is completely monotone on the positive real axis $\mathbb { R } ^ { + }$ [82], and thus it is positive on $\mathbb { R } ^ { + } ;$ ; see also [90] for extension to the two-parameter Mittag-Lef<sup>fl</sup>er function $E _ { \alpha , \beta } ( z )$ . It appears in the solution representation for FDEs: the functions $E _ { \alpha , 1 } ( - \lambda t ^ { \alpha } )$ and $t ^ { \alpha - 1 } E _ { \alpha , \alpha } ( - \lambda t ^ { \alpha } )$ appear in the kernel of the time fractional diffusion problem with initial data and the right-hand side, respectively, and $x E _ { \alpha , 2 } ( - \lambda x ^ { \alpha } )$ and $x ^ { \alpha - 1 } E _ { \alpha , \alpha } ( - \lambda x ^ { \alpha } )$ are eigenfunctions to the fractional Sturm–Liouville problem with a zero potential, cf section 4.1. 

In our discussions, the asymptotic behavior of the function $E _ { \alpha , \beta } ( z )$ will play a crucial role. It satis<sup>fi</sup>es the following exponential asymptotics [53, p 43, equations (2.8.17) and (2.8.18)], which was <sup>fi</sup>rst derived by Djrbashian [21], and re<sup>fi</sup>ned by many researchers [80, 105]. 

Lemma 2.1. Let $\alpha \in ( 0 , 2 )$ , $\beta \in \mathbb { R } ,$ , and μ ∈ ( 2, min ( , ))απ π απ . Then for $N \in \mathbb { N }$ 

$$
\begin{array}{l} E _ {\alpha , \beta} (z) = \frac {1}{\alpha} z ^ {(1 - \beta) / \alpha} \mathrm{e} ^ {z ^ {1 / \alpha}} - \sum_ {k = 1} ^ {N} \frac {1}{\Gamma (\beta - \alpha k)} \frac {1}{z ^ {k}} + O \left(\frac {1}{z ^ {N + 1}}\right) \text {with} | z | \to \infty , | \arg (z) | \leqslant \mu , \\ E _ {\alpha , \beta} (z) = - \sum_ {k = 1} ^ {N} \frac {1}{\Gamma (\beta - \alpha k)} \frac {1}{z ^ {k}} + O \left(\frac {1}{z ^ {N + 1}}\right) \text {with} | z | \to \infty , \mu \leqslant | \arg (z) | \leqslant \pi . \end{array}
$$

From these asymptotics, the Mittag-Lef<sup>fl</sup>er function $E _ { \alpha , \beta } ( z )$ decays only linearly on the negative real axis $\mathbb { R } ^ { - } .$ , which is much slower than the exponential decay for the exponential function $\mathbf { e } ^ { z } .$ . However, on the positive real axis $\mathbb { R } ^ { + }$ , it grows exponentially, and the growth rate increases with the fractional order $0 < \alpha < 2$ . To illustrate the distinct feature, we plot the functions $E _ { \alpha , 1 } ( - \pi ^ { 2 } t ^ { \alpha } )$ and $t ^ { \alpha - 1 } E _ { \alpha , \alpha } ( - \pi ^ { 2 } t ^ { \alpha } )$ in <sup>fi</sup>gure 1 for several different α values, where $\lambda = \pi ^ { 2 }$ is the <sup>fi</sup>rst Dirichlet eigenvalue of the negative Laplacian on the unit interva $\varOmega = ( 0 , 1 )$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/9d422042478dfcd25554f79b44ca20898936f4fd58df728d96df9cb077cb5ce4.jpg)



(a) $E _ { \alpha , 1 } ( - \pi ^ { 2 } t ^ { \alpha } )$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/6201c86245b341d37e4d748ac8b7714996912280b3e5556f33fd45a2babed97c.jpg)



(b) tα−1 Eα,α(−π2tα)



Figure 1. The pro<sup>fi</sup>les of Mittag-Lef<sup>fl</sup>er functions (a) $E _ { \alpha , 1 } ( - \pi ^ { 2 } t ^ { \alpha } )$ and (b) $t ^ { \alpha - 1 } E _ { \alpha , \alpha } ( - \pi ^ { 2 } t ^ { \alpha } )$ . The value $\pi ^ { 2 }$ is the <sup>fi</sup>rst Dirichlet eigenvalue of the negative Laplacian on the interval $( 0 , 1 )$ .


; see appendix A.1 for further details on the computation of the Mittag-Lef<sup>fl</sup>er function. Figure 1(a) can be viewed as the time evolution of $u ( 1 / 2 , t )$ , where $\partial _ { t } ^ { \alpha } u - u _ { x x } = 0$ with $u ( 0 , t ) = u ( 1 , t ) = 0$ , and initial data $u _ { 0 } ( x ) = \sin \pi x$ (the lowest Fourier eigenmode). The slow decay behavior at large time is clearly observed. In particular, at $t = 1$ , the function $E _ { \alpha , 1 } ( - \pi ^ { 2 } t )$ still takes values distinctly away from zero for any $0 < \alpha < 1$ , whereas the exponential function $\mathrm { e } ^ { - \pi ^ { 2 } t }$ almost vanishes identically. In contrast, for t close to zero, the picture is reversed: the Mittag-Lef<sup>fl</sup>er function $E _ { \alpha , 1 } ( - \pi ^ { 2 } t )$ decays much faster than the exponential function $\mathrm { e } ^ { - \pi ^ { 2 } t }$ . The drastically different behavior of the function $E _ { \alpha , 1 } ( - z )$ , in comparison with the exponential function $\mathrm { e } ^ { - z }$ , explains many unusual phenomena with inverse problems for FDEs to be described below. According to the exponential asymptotics, the function $E _ { \alpha , \alpha } ( z )$ decays faster on the negative real axis $\mathbb { R } ^ { - }$ , since $1 / T ( 0 ) = 0 , { \mathrm { i . e . } }$ , the <sup>fi</sup>rst term in the expansion vanishes. This is con<sup>fi</sup>rmed numerically in <sup>fi</sup>gure 1(b). Even though not shown, it is noted that the function $E _ { \alpha , \alpha } ( z )$ decays only quadratically on the negative real axis $\mathbb { R } ^ { - }$ for $\alpha \in ( 0 , 1 )$ or $\alpha \in ( 1 , 2 )$ , which is asymptotically much slower than the exponential decay. 

The distribution of zeros of the Mittag-Leffer function $E _ { \alpha , \beta } ( z )$ is of immense interest, especially in the related Sturm–Liouville problem; see section 4.1 below. The case of $\beta = 1$ was <sup>fi</sup>rst studied by Wiman [104]. It was revisited by Djrbashian [21], and many deep results were derived, especially for the case of $\alpha = 2$ . There are many further re<sup>fi</sup>nements [93]; see [83] for an updated account. 

## 2.2. Wright function

The Wright function $W _ { \rho , \mu } ( z )$ is de<sup>fi</sup>ned by [106, 107] 

$$
W _ {\rho , \mu} (z) = \sum_ {k = 0} ^ {\infty} \frac {z ^ {k}}{k ! \Gamma (\rho k + \mu)}, \quad \mu , \rho \in \mathbb {R}, \rho > - 1, \quad z \in \mathbb {C}.
$$

This is an entire function of order $1 / ( 1 + \rho )$ [30, theorem 2.4.1]. It was <sup>fi</sup>rst introduced in connection with a problem in number theory by Edward M Wright, and revived in recent years since it appears as the fundamental solution for FDEs [69]. The Wright function $W _ { \rho , \mu } ( z )$ has the following the asymptotic expansion in one sector containing the negative real axis <sup>−</sup> [67, theorem 3.2]. Like before, the exponential asymptotics can be used to deduce the distribution of its zeros [66]. 

Lemma 2.2. $L e t - 1 < \rho < 0 , ~ y = - z , ~ \mathrm { a r g } ( z ) \leqslant \pi , ~ - \pi < \arg ( y ) \leqslant \pi ,$ , and for all small $\epsilon > 0 , | \mathrm { a r g } ( y ) | \leqslant$ min (3π $( 1 + \rho ) / 2 , \pi ) - \epsilon .$ Then 

$$
W _ {\rho , \mu} (z) = Y ^ {1 / 2 - \mu} \mathrm{e} ^ {- Y} \left\{\sum_ {m = 0} ^ {M - 1} A _ {m} Y ^ {- m} + O (Y ^ {- M}) \right\}, \quad Y \rightarrow \infty ,
$$

where $Y = ( 1 + \rho ) ( ( - \rho ) ^ { - \rho } y ) ^ { 1 / ( 1 + \rho ) }$ and the coefficients $A _ { m } , m = 0 , 1 , \ldots$ are defined by the asymptotic expansion 

$$
\begin{array}{l} \frac {\Gamma (1 - \mu - \rho t)}{2 \pi (- \rho) ^ {- \rho t} (1 + \rho) ^ {(1 + \rho) (t + 1)} \Gamma (t + 1)} = \sum_ {m = 0} ^ {M - 1} \frac {(- 1) ^ {m} A _ {m}}{\Gamma \Big ((1 + \rho) t + \mu + \frac {1}{2} + m \Big)} \\ \qquad + O \left(\frac {1}{\Gamma \Big ((1 + \rho) t + \beta + \frac {1}{2} + M \Big)}\right), \end{array}
$$

valid for arg( ), arg( )t t−ρ , and arg $( 1 - \mu - \rho t )$ all lying between −π and π and t tending to infinity. 

The Wright function $W _ { \rho , \mu } ( z ) , - 1 < \rho < 0$ , decays exponentially on the negative real axis $\mathbb { R } ^ { - }$ , in a manner similar to the exponential function $\mathrm { e } ^ { z } .$ , but at a different decay rate. Its special role in fractional calculus is underscored by the fact that it forms the free-space fundamental solution $K _ { \alpha } ( x , t )$ to the one-dimensional time fractional diffusion equation [69] by 

$$
K _ {\alpha} (x, t) = \frac {1}{2 t ^ {\frac {\alpha}{2}}} W _ {- \frac {\alpha}{2}, 1 - \frac {\alpha}{2}} \Big (- | x | / t ^ {\alpha / 2} \Big).\tag{2.2}
$$

The multidimensional case is more complex and involves further special functions, in particular, the Fox H function [54, 91]. For $\alpha = 1$ , the formula (2.2) recovers the familiar freespace fundamental solution for the one-dimensional heat equation, i.e. 

$$
K (x, t) = \frac {1}{2 \sqrt {\pi t}} \mathrm{e} ^ {- \frac {x ^ {2}}{4 t}},
$$

which is a Gaussian distribution in x for any $t > 0$ . In the fractional case, the fundamental solution $K _ { \alpha } ( x , t )$ exhibits quite different behavior than the heat kernel. To see this, we show the pro<sup>fi</sup>le of $K _ { \alpha } ( x , t )$ in <sup>fi</sup>gure 2 for several α values; see appendix A.1 for a brief description of the computational details. For any $0 < \alpha < 1 , K _ { \alpha } ( x , t )$ decays slower at a polynomial rate as the argument |x $1 / t ^ { \alpha / 2 }$ tends to in<sup>fi</sup>nity, i.e., having a long tail, when compared with the Gaussian density. The long tail pro<sup>fi</sup>le is one of distinct features of slow diffusion [5]. Further, for any $\alpha < 1$ , the pro<sup>fi</sup>le is only continuous but not differentiable at $x = 0$ . The kink at the origin implies that the solution operator to time fractional diffusion may only have a limited smoothing property. 

## 2.3. Singular value decomposition

We shall follow the well-established practice in the inverse problem community, i.e., using the singular value decomposition, as the main tool for numerically analyzing the problem behavior [32]. Speci<sup>fi</sup>cally, we shall numerically compute the forward map $F ,$ and analyze its behavior to gain insights into the inverse problem. Given a matrix $\mathbf { A } \in \mathbb { R } ^ { n \times m }$ , its singular value decomposition is given by 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/bd07b30a5ba876c7ff8a0422a06714438b343a8e633f95850bebec69380ac86c.jpg)



(a) t = 0.1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/ebcf0ae333d0065b9ebfc744e069bc02048f31673b9bbbcc85a799132de2216b.jpg)



(b) t = 1



Figure 2. The pro<sup>fi</sup>le of the fundamental solution $K _ { \alpha } ( x , t )$ at (a) t = 0.1 and (b) $t = 1$


$$
\mathbf {A} = \mathbf {U} \boldsymbol {\Sigma} \mathbf {V} ^ {\mathrm{t}},
$$

where $\mathbf { U } = [ \mathbf { u } _ { 1 } \dots \mathbf { u } _ { n } ] \in \mathbb { R } ^ { n \times n }$ and ${ \bf V } = [ { \bf v } _ { 1 } \ldots { \bf v } _ { m } ] \in \mathbb { R } ^ { m \times m }$ are column orthonormal matrices, and $\varSigma \in \mathbb { R } ^ { n \times m } = \mathrm { d i a g } ( \sigma _ { 1 } , . . . , \sigma _ { r } , 0 , . . . , 0 )$ is a diagonal matrix, with the diagonal elements $\left\{ \sigma _ { i } \right\}$ being nonnegative and listed in a descending order $\sigma _ { 1 } > . . . > \sigma _ { r } > 0 .$ , and $r$ being the (numerical) rank of the matrix A. The diagonal element $\sigma _ { i }$ is known as the ith singular value, and the corresponding columns of U and V, i.e., $\mathbf { u } _ { i }$ and $\mathbf { v } _ { i } ,$ are called the left and right singular vectors, respectively. 

One simple measure of the conditioning of a linear inverse problem $\mathbf { A } \mathbf { x } = \mathbf { b }$ is the condition number cond( )A , which is de<sup>fi</sup>ned as the ratio of the largest to the smallest nonzero singular value, i.e. 

$$
\operatorname{cond} (\mathbf {A}) = \sigma_ {1} / \sigma_ {r}.
$$

In particular, if the condition number is small, then the data error will not be ampli<sup>fi</sup>ed much. In the case of a large condition number, the issue is more delicate: it may or may not amplify the data perturbation greatly. A more complete picture is provided by the singular value spectrum $( \sigma _ { 1 } , \sigma _ { 2 } , . . . , \sigma _ { r } )$ . Especially, a singular value spectrum gradually decaying to zero without a clear gap is characteristic of many discrete ill-posed problems, which is reminiscent of the spectral behavior of compact operators. We shall adopt these simple tools to analyze related inverse problems below. 

In addition, using singular value decomposition and regularization techniques, e.g. Tikhonov regularization or truncated singular value decomposition, one can conveniently obtain numerical reconstructions, even though this might not be the most ef<sup>fi</sup>cient way to do so. However, we shall not delve into the extremely important question of practical reconstructions, since it relies heavily on a priori knowledge on the sought for solution and the statistical nature (Gaussian, Poisson, Laplace $\cdots )$ of the contaminating noise in the data, which will depend very much on the speci<sup>fi</sup>c application. We refer interested readers to the monographs [26, 41, 92] and the survey [49] for updated accounts on regularization methods for constructing stable reconstructing procedures and ef<sup>fi</sup>cient computational techniques. We will also brie<sup>fl</sup>y mention below related works on the application of regularization techniques to inverse problems for FDEs. 

## 3. Inverse problems for time fractional diffusion

In this section, we consider several model inverse problems for the following one-dimensional time fractional diffusion equation on the unit interval $\varOmega = ( 0 , 1 )$ ): 

$$
\partial_ {t} ^ {\alpha} u - u _ {x x} + q u = f \quad \text { in } \Omega \times (0, T ],\tag{3.1}
$$

with the fractional order $\alpha \in ( 0 , 1 )$ , the initial condition $u ( 0 ) = \nu$ and suitable boundary conditions. Although we consider only the one-dimensional model, the analysis and computation in this part can be extended into the general multi-dimensional case, upon suitable modi<sup>fi</sup>cations. Recall that $\partial _ { t } ^ { \alpha } u$ denotes the Djrbashian–Caputo fractional derivative of order α with respect to time t. For $\alpha = 1$ , the fractional derivative $\partial _ { t } ^ { \alpha } u$ coincides with the usual <sup>fi</sup>rst-order derivative $u ^ { \prime } ,$ , and accordingly, the model (3.1) reduces to the classical diffusion equation. Hence it is natural to compare inverse problems for the model (3.1) with that for the standard diffusion equation. We shall discuss the following four inverse problems, i.e., the backward problem, sideways problem, inverse source problem and inverse potential problem. In the <sup>fi</sup>rst three cases, we shall assume a zero potential $q = 0$ . We will also discuss the solution of an inverse coef<sup>fi</sup>cient problem using fractional calculus. 

## 3.1. Backward fractional diffusion

First we consider the time fractional backward diffusion. By the linearity of the inverse problem, we may assume that equation (3.1) is prescribed with a homogeneous Dirichlet boundary condition, $\mathrm { i . e . , } u = 0 \mathrm { a t } x = 0 , 1$ , and the initial condition $u ( 0 ) = \nu .$ . Then the inverse problem reads: given the <sup>fi</sup>nal time data $g = u ( T )$ ), <sup>fi</sup>nd the initial condition v. It arises in, for example, the determination of a stationary contaminant source in underground <sup>fl</sup>uid <sup>fl</sup>ow. 

To gain insight, we apply the separation of variables. Let $\{ ( \lambda _ { j } , \phi _ { j } ) \}$ , with $\lambda _ { j } = ( j \pi ) ^ { 2 }$ and $\phi _ { j } = \sqrt { 2 }$ sin $j \pi x ,$ be the Dirichlet eigenpairs of the negative Laplacian on the interval Ω. The eigenfunctions $\{ \phi _ { j } \}$ form an orthonormal basis of the $L ^ { 2 } ( \varOmega )$ space. Then using the Mittag-Lef<sup>fl</sup>er function $E _ { \alpha , \beta } ( z )$ de<sup>fi</sup>ned in (2.1), the solution u to equation (3.1) can be expressed as 

$$
u (x, t) = \sum_ {j = 1} ^ {\infty} E _ {\alpha , 1} \left(- \lambda_ {j} t ^ {\alpha}\right) \left(v, \phi_ {j}\right) \phi_ {j} (x).
$$

Therefore, the <sup>fi</sup>nal time data $g = u ( T )$ is given by 

$$
g (x) = \sum_ {j = 1} ^ {\infty} E _ {\alpha , 1} \left(- \lambda_ {j} T ^ {\alpha}\right) \left(v, \phi_ {j}\right) \phi_ {j} (x).
$$

It follows directly that the initial data v is formally given by 

$$
v = \sum_ {j = 1} ^ {\infty} \frac {(g , \phi_ {j})}{E _ {\alpha , 1} (- \lambda_ {j} T ^ {\alpha})} \phi_ {j}.
$$

Since the function $E _ { \alpha , 1 } ( - t )$ is completely monotone on the positive real axis $\mathbb { R } ^ { + }$ [82] for any $\alpha \in ( 0$ , 1], the denominator in the representation does not vanish. In case of $\alpha = 1$ , the formula reduces to the familiar expression 

$$
v = \sum_ {j = 1} ^ {\infty} \mathrm{e} ^ {\lambda_ {j} T} \Big (g, \phi_ {j} \Big) \phi_ {j}.
$$

This formula shows clearly the well-known, severely ill-posed nature of the backward diffusion problem: the perturbation in the jth Fourier mode $( g , \phi _ { j } )$ of the (noisy) data $g$ is ampli<sup>fi</sup>ed by an exponentially growing factor $\mathbf { e } ^ { \lambda _ { j } T }$ , which can be astronomically large, even for a very small index $j ,$ if the terminal time $T$ is not very small. Hence it is always severely ill-conditioned and we must multiply the $j \mathrm { t h }$ Fourier mode of the data $g$ by a factor $\mathrm { e } ^ { \lambda _ { j } T }$ in order to recover the corresponding mode of the initial data v. 

In the fractional case, by lemma 2.1, the Mittag-Lef<sup>fl</sup>er function $E _ { \alpha , 1 } ( z )$ decays only linearly on the negative real axis $\mathbb { R } ^ { - }$ , and thus the multiplier $1 / E _ { \alpha , 1 } ( - \lambda _ { j } T ^ { \alpha } )$ grows only linearly in $\lambda _ { j } , \mathrm { i . e . , } 1 / E _ { \alpha , 1 } ( - \lambda _ { j } T ^ { \alpha } ) \sim \lambda _ { j }$ , which is very mild compared to the exponential growth $\mathbf { e } ^ { \lambda _ { j } T }$ for the case $\alpha = 1$ , and thus the fractional case is only mildly ill-posed. Roughly, the jth Fourier mode of the initial data v now equals the jth mode of the data $g$ multiplied by $\lambda _ { j }$ . More precisely, it amounts to the loss of two spatial derivatives [88, theorem 4.1] 

$$
\| v \| _ {L ^ {2} (\Omega)} \leqslant c \| u (T) \| _ {H ^ {2} (\Omega)}.\tag{3.2}
$$

Intuitively, the history mechanism of the anomalous diffusion process retains the complete dynamics of the physical process, including the initial data, and thus it is much easier to go backwards to the initial state v. This is in sharp contrast to classical diffusion, which has only a short memory and loses track of the preceding states quickly. This result has become quite well-known in the inverse problems community and has contributed to a belief that ‘inverse problems for FDEs are less ill-conditioned than their classical counterparts’—throughout this paper we will see that this conclusion as a general statement can be quite far from the truth. 

Does this mean that for all terminal time T the fractional case is always less ill-posed than the classical one? The answer is yes, in the sense of the norm on the data space in which the data $g$ lies. Does this mean that from a computational stability standpoint that one can always solve the backward fractional problem more effectively than for the classical case? The answer is no, and the difference can be substantial. To illustrate the point, let J be the highest frequency mode required of the initial data v and assume that we believe we are able to multiply the <sup>fi</sup>rst J modes $g _ { j } = ( g , \phi _ { j } ) , j = 1 , 2 , . . . , J $ , by a factor no larger than $M$ (which roughly assumes that the noise levels in both cases are comparable). By the monotonicity of the function $E _ { \alpha , 1 } ( - t )$ in t, it suf<sup>fi</sup>ces to examine the Jth mode. For the heat equation $\nu _ { J } : = ( \nu , \phi _ { J } ) = \mathrm { e } ^ { \lambda _ { J } T } g _ { J }$ and provided that $T = T _ { J } < \lambda _ { J } / \mathrm { l o g }$ M this is feasible. For a <sup>fi</sup>xed $J ,$ if $T _ { \alpha } ^ { \star }$ denotes the point where 

$$
\mathrm{e} ^ {- \lambda_ {J} T _ {\alpha} ^ {\star}} = E _ {\alpha , 1} \left(- \lambda_ {J} T _ {\alpha} ^ {\star}\right),
$$

then in the fractional case for $T < T _ { \alpha } ^ { \star }$ the growth factor on $g _ { J }$ will exceed M for any $T < T _ { \alpha } ^ { \star }$ In table 1, we present the critical value $T _ { \alpha } ^ { \star }$ for several values of the fractional order α and the maximum number of modes J. The numbers in the table are very telling. For example, for the case $J = 5 , \alpha = 1 / 4$ and $T = 0 . 0 2$ (which is approximately one half the value of $T _ { \alpha } ^ { \star } )$ , the growth factor is about 1.6 for the heat equation but about 113 for the fractional case. With $J = 1 0$ and $\alpha = 1 / 4$ and $T = T _ { \alpha } ^ { \star }$ the growth factor is around 336. If $T = T _ { \alpha } ^ { \star } / 1 0$ then it has again dropped to less than 2 for the heat equation but about 190 for the fractional case. Of course, for $T > T _ { \alpha } ^ { \star }$ the situation completely reverses. With $J = 1 0 , \alpha = 1 / 4$ and $T = 1 0 ~ T _ { \alpha } ^ { \star }$ the growth factor is a possibly workable value of around 600; while for the heat equation it is greater than $1 0 ^ { 2 5 }$ . We reiterate that the apparent contradiction between the theoretical illconditioning and numerical stability is due to the spectral cutoff present in any practical reconstruction procedure. 


Table 1. The critical values $T _ { \alpha } ^ { * }$ for fractional backward diffusion.


<table><tr><td>α\J</td><td>3</td><td>5</td><td>10</td></tr><tr><td>1/4</td><td>0.0442</td><td>0.0197</td><td>0.0059</td></tr><tr><td>1/2</td><td>0.0387</td><td>0.0163</td><td>0.0049</td></tr><tr><td>3/4</td><td>0.0351</td><td>0.0142</td><td>0.0040</td></tr></table>

Next we examine the in<sup>fl</sup>uence of the fractional order α on the inversion step more closely. To this end, we expand the initial condition v in the piecewise linear <sup>fi</sup>nite element basis functions de<sup>fi</sup>ned on a uniform partition of the domain $\varOmega = ( 0 , 1 )$ ) with 100 grid points. Then we compute the discrete forward map F from the initial condition to the <sup>fi</sup>nal time data $g = u ( T )$ , de<sup>fi</sup>ned on the same mesh. Numerically, this can be achieved by a fully discrete scheme based on the L1 approximation in time and the <sup>fi</sup>nite difference method in space; see appendix A.2 for a description of the numerical method. The ill-posed behavior of the discrete inverse problem is then analyzed using singular value decomposition. A similar experimental setup will be adopted for other examples below. 

The numerical results are shown in <sup>fi</sup>gure 3. The condition number of the (discrete) forward map F stays mostly around ${ \cal O } ( 1 0 ^ { 4 } )$ for a fairly broad range of α values, which holds for all three different terminal times T. This can be attributed to the fact that for any $\alpha \in ( 0 , 1 )$ , backward fractional diffusion amounts to a two spacial derivative loss, cf (3.2). Unsurprisingly, as the fractional order α approaches unity, the condition number eventually blows up, recovering the severely ill-posed nature of the classical backward diffusion problem, cf <sup>fi</sup>gure 3(a). Further, we observe that the smaller is the terminal time T, the quicker is the blowup. The precise mechanism for this observation remains unclear. Interestingly, the condition number is not monotone with respect to the fractional order $\alpha ,$ for a <sup>fi</sup>xed T. This might imply potential nonuniqueness in the simultaneous recovery of the fractional order α and the initial data v. The singular value spectra at $T = 0 . 0 1$ are shown in <sup>fi</sup>gure 3(b). Even though the condition numbers for $\alpha = 1 / 4$ and $\alpha = 1 / 2$ are quite close, their singular value spectra actually differ by a multiplicative constant, but their decay rates are almost identical, thereby showing comparable condition numbers. This shift in singular value spectra can be explained by the local decay behavior of the Mittag-Lef<sup>fl</sup>er function, cf <sup>fi</sup>gure 1(a): the smaller is the fractional order α, the faster is the decay around $t = 0$ 

Even though the condition number is very informative about the (discretized) problem, it does not provide a full picture, especially when the condition number is large. In this case the singular value spectrum can be far more revealing. The spectra for two different α values are given in <sup>fi</sup>gure 4. At $\alpha = 1 / 2$ , the singular values decay at almost the same algebraic rate, irrespective of the terminal time T. This is expected from the two-derivative loss for any $\alpha < 1$ . However, for $\alpha = 1$ , the singular values decay exponentially, and the decay rate increases dramatically with the increase of the terminal time T. For $T = 0 . 0 0 1$ , there are a handful of ‘signi<sup>fi</sup>cant’ singular values, say above $1 0 ^ { - 3 }$ , but when the time T increases to $T = 1$ , there is only one meaningful singular value remaining. The distribution of the singular values has important practical consequences. For a small time T, the <sup>fi</sup>rst few singular values for the classical diffusion case actually might be much larger than that for the fractional case, which indicates that the classical case is actually numerically much easier to recover in this regime, concurring with the observations drawn from table 1. For example, at $T = 0 . 0 0 1$ , the <sup>fi</sup>rst twenty singular values are larger than the fractional counterpart, cf <sup>fi</sup>gure 3(b), and hence, the <sup>fi</sup>rst twenty modes, i.e., left singular vectors, are more stable in the reconstruction procedure. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/9f057e17f0507191259db16559cc5e176523d28cdc9f0515ec6b074e8f4e1216.jpg)



(a) condition number


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/517c6667f78ef12f7046d12478a5d08a02fe27e1cbda57d0a8f6d44f34290913.jpg)



(b) singular value spectrum



Figure 3. (a) The condition number versus the fractional order α, and (b) the singular value spectrum at $T = 0 . 0 1$ for the backward fractional diffusion. We only display the <sup>fi</sup>rst 50 singular values.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/99299c720d7ab32dbe80829443b3a26d2a7ee6f1ca68f45ca32c04c3fc7bbc8c.jpg)



(a) $\alpha = 1 / 2$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/7e062fdea4feaaadd6fde4a08c3d86345faf7e31ffa010ca79c2f982d9ce8c54.jpg)



(b) $\alpha = 1$



Figure 4. The singular value spectrum of the forward map F from the initial data to the <sup>fi</sup>nal time data, for (a) $\alpha = 1 / 2$ and (b) $\alpha = 1$ , at four different times for the time fractional backward diffusion.


The mathematical model (3.1) is rescaled with a unit diffusion coef<sup>fi</sup>cient. In practice, there is always a diffusion coef<sup>fi</sup>cient σ in the elliptic operator, i.e. 

$$
\partial_ {t} ^ {\alpha} u - \nabla \cdot (\sigma \nabla u) + q u = f.
$$

For example, the thermal conductivity σ of the gun steel at moderate temperature is about $1 . 8 \times 1 0 ^ { - 5 } \mathrm { m } ^ { 2 } \mathrm { s } ^ { - 1 }$ [11] and the diffusion coef<sup>fi</sup>cient of the oxygen in water at $2 5 ~ ^ { \circ } \mathrm { C }$ is $2 . 1 0 \times 1 0 ^ { - 5 } \mathrm { c m ^ { 2 } s ^ { - 1 } }$ [18]. Mathematically, this does not change the ill-posed nature of the inverse problem. However, the presence of a diffusion coef<sup>fi</sup>cient $\sigma$ has important consequence: it enables the practical feasibility of the classical backward diffusion problem (and likely for many other inverse problems for the diffusion equation). Physically, a constant conductivity σ amounts to rescaling the <sup>fi</sup>nal time T by $T ^ { \prime } = T / \sigma$ . In the fractional case, a similar but nonlinear scaling law $T ^ { \prime } = T / \sigma ^ { 1 / \alpha }$ remains valid. 

Numerically, time fractional backward diffusion has been extensively studied. Liu and Yamamoto [65] proposed a numerical scheme for the one-dimensional fractional backward problem based on the quasi-reversibility method [57], and derived error estimates for the approximation, under a priori smoothness assumption on the initial condition. This represents one of the <sup>fi</sup>rst works on inverse problems in anomalous diffusion. Later, Wang and Liu [99] studied total variation regularization for two-dimensional fractional backward diffusion, and analyzed its well-posedness of the optimization problem and the convergence of an iterative scheme of Bregman type. Wei and Wang [102] developed a modi<sup>fi</sup>ed quasi-boundary value method for the problem in a general domain, and established error estimates for both a priori and a posteriori parameter choice rules. In view of better stability results in the fractional case, one naturally expects better error estimates than the classical diffusion equation, which is con<sup>fi</sup>rmed by these studies. 

## 3.2. Sideways fractional diffusion

Next we consider the sideways problem for time fractional diffusion. There are several possible formulations, e.g., the quarter plane and the <sup>fi</sup>nite space domain. The quarter plane sideways fractional diffusion problem is as follows. Let $\iota ( x , t )$ be de<sup>fi</sup>ned in $( 0 , \infty ) \times ( 0 , \infty )$ by 

$$
\partial_ {t} ^ {\alpha} u - u _ {x x} = 0, \qquad x > 0, t > 0,
$$

and the boundary and initial conditions 

$$
u (x, 0) = 0 \quad \text { and } \quad u (0, t) = f (t),
$$

where we assume |u $( x , t ) \vert \leqslant c _ { 1 } \mathbf { e } ^ { c _ { 2 } x ^ { 2 } }$ . We do not know the left boundary condition $f ,$ but are able to measure u at an intermediate point $x = L > 0 , h ( t ) = u ( L , t )$ . The inverse problem is: given the (noisy) data $h ,$ <sup>fi</sup>nd the boundary condition $f .$ The solution u of the forward problem is given by a convolution integral with the kernel being the spatial derivative $K _ { \alpha , x } ( x , s )$ of the fundamental solution $K _ { \alpha } ( x , s )$ , cf (2.2), by 

$$
u (x, t) = \int_ {0} ^ {t} K _ {\alpha , x} (x, t - s) f (s) \mathrm{d} s.
$$

This representation is well known for the case $\alpha = 1$ , and it was <sup>fi</sup>rst derived by Carasso [11]; see also [8] for related discussions. It leads to a convolution integral equation for the unknown $f$ in terms of the given data h 

$$
h (t) = \int_ {0} ^ {t} R _ {\alpha} (t - s) f (s) \mathrm{d} s,
$$

where the convolution kernel $R _ { \alpha } ( s )$ is given by a Wright function in the form 

$$
R _ {\alpha} (s) = \frac {1}{2 s ^ {\alpha}} W _ {- \frac {\alpha}{2}, 2 - \frac {\alpha}{2}} (- L s ^ {- \alpha / 2}) = \sum_ {k = 0} ^ {\infty} \frac {(- L) ^ {k}}{k ! \Gamma \left(- \frac {\alpha}{2} k + 2 - \frac {\alpha}{2}\right)} s ^ {- k \frac {\alpha}{2} - \alpha}.
$$

In case of $\alpha = 1$ , i.e., classical diffusion, the kernel $R ( s )$ is given explicitly by 

$$
R (s) = \frac {L}{2 \sqrt {\pi}} s ^ {- \frac {3}{2}} \mathrm{e} ^ {- \frac {L ^ {2}}{4 s}} \in C ^ {\infty} (0, \infty).
$$

Since all its derivatives vanish at $s = 0 ,$ the classical theory of Volterra integral equations of the <sup>fi</sup>rst kind [56] implies the extreme ill-conditioning of the problem. This is not surprising: we are, after all, mapping a function $f \in C ^ { 0 } ( 0 , \infty )$ to an element in $C ^ { \infty } ( 0 , \infty )$ . The conditioning of the time fractional sideways problem again depends on the convolution kernel $R _ { \alpha }$ and its derivatives at $s \ : = \ : 0$ and in this case is the value of the Wright function $W _ { - \alpha / 2 , 2 - \alpha / 2 } ( - z )$ and its derivatives as $z  \infty$ . These are again zero (in fact the Wright function $W _ { - \alpha / 2 , 2 - \alpha / 2 } \left( z \right)$ also decays exponentially to zero for large negative arguments, cf lemma $2 . 2 )$ , and thus the fractional sideways problem is also severely ill-posed. However, this analysis does not show their difference in the degree of ill-posedness: even though both are severely ill-posed, their practical computational behavior can still be quite different, as we shall see below. 

To see their difference in the degree of ill-posedness, we examine another variant of the sideways problem on a <sup>fi</sup>nite interval $\varOmega = ( 0 , 1 )$ , with Cauchy data prescribed on the axis $x = 0 .$ , i.e. given zero initial condition $u _ { 0 } = u ( x , 0 ) = 0$ , recovering $h = u ( 1 , t )$ from the lateral Cauchy data at $x = 0 \mathrm { : }$ 

$$
u (0, t) = f (t), \quad u _ {x} (0, t) = g (t), \quad t \geqslant 0.
$$

This problem is also known as the lateral Cauchy problem in the literature. In the case $\alpha = 1$ it is known that the inverse problem is severely ill-posed [8, 33]. To gain insight into the fractional case, we apply the Laplace transform. With  being the Laplace transform in time, and noting the Laplace transform of the Caputo derivative $\widehat { \partial _ { t } ^ { \alpha } u } = z ^ { \alpha } \widehat { u } \left( z \right) - z ^ { \alpha - 1 } u _ { 0 }$ [53, lemma 2.24], we deduce 

$$
z ^ {\alpha} \widehat {u} (x, z) - \widehat {u} _ {x x} (x, z) = 0, \widehat {u} (0) = \widehat {f}, \widehat {u} _ {x} (0) = \widehat {g}.
$$

The general solution u is given by ${ \widehat { u } } \left( x , z \right) = { \widehat { f } }$ cosh $z ^ { \alpha / 2 } x + { \widehat g } { \frac { \sinh z ^ { \alpha / 2 } x } { z ^ { \alpha / 2 } x } }$ and thus the solution $\widehat { h } \left( z \right) = \widehat { u } \left( 1 , z \right)$ at $x = 1$ is given by 

$$
\widehat {h} = \widehat {f} \cosh z ^ {\alpha / 2} + \widehat {g} \frac {\sinh z ^ {\alpha / 2}}{z ^ {\alpha / 2}}.
$$

The solution h(t) can then be recovered by an inverse Laplace transform 

$$
h (t) = \frac {1}{2 \pi \mathrm{i}} \int_ {\mathrm{Br}} \widehat {h} \mathrm{e} ^ {z t} \mathrm{d} z,\tag{3.3}
$$

where $\operatorname { B r } = \{ z \in \mathbb { C } \colon \Re z = \sigma , \sigma > 0 \}$ is the Bromwich path. Upon deforming the contour suitably, this formula will allow the development of an ef<sup>fi</sup>cient numerical scheme for the sideways problem via quadrature rules [103], provided that the lateral Cauchy data is available for all $t > 0$ . The expression (3.3) indicates that, in the fractional case, the sideways problem still suffers from severe ill-posedness in theory, since the high frequency modes of the data perturbation are ampli<sup>fi</sup>ed by an exponentially growing multiplier $\mathrm { e } ^ { z ^ { \alpha / 2 } }$ . However, numerically, the degree of ill-posedness decreases dramatically as the fractional order α decreases from unity to zero, since as $\alpha \to 0 ^ { + }$ , the multipliers are growing at a much slower rate, and thus we have a better chance of recovering many more modes of the boundary data. In other words, both the classical and fractional sideways problems are severely ill-posed in the sense of error estimates between the norms in the data and unknowns; but with a <sup>fi</sup>xed frequency range, the behavior of the time fractional sideways problem can be much less illposed. Hence, anomalous diffusion mechanism does help substantially since much more effective reconstructions are possible in the fractional case. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/ca311d8ee85894d2d7e5ebaa3dc590455cdd17c3245fbc2763ef165f61ecd167.jpg)



(a) condition number


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/0ce5c85c36a3adaa5f6b006619b8efcfa31cfbb962e1315f9f734efb0331952b.jpg)



(b) singular value spectrum



Figure 5. (a) The condition number and (b) singular value spectrum at $T = 1$ for the time fractional sideways problem.


Next we illustrate the point numerically. The numerical results for the sideways problem are given in <sup>fi</sup>gure 5. It is observed that the degree of ill-posedness of the <sup>fi</sup>nite-dimensional discretized version of the inverse problem indeed decreases dramatically with the decrease of the fractional order α, cf <sup>fi</sup>gure 5(a), which agrees well with the preceding discussions. Surprisingly, for $T = 1$ there is a sudden transition around $\alpha = 1 / 2$ , below which the sideways problem behaves as if nearly well-posed, but above which the conditioning deteriorates dramatically with the increase of the fractional order α and eventually it recovers the properties of the classical sideways problem. Similar transitions are observed for other terminal times. This might be related to the discrete setting, for which there is an inherent frequency cutoff. Further, as the fractional order α approaches zero, the problem reaches a quasi-steady state much quicker and thus the forward map F can have only fairly localized elements along the main diagonal. To give a more complete picture, we examine the singular value spectrum in <sup>fi</sup>gure 5(b). Unlike the backward diffusion problem discussed earlier, the singular values are actually decaying only algebraically, even for $\alpha = 1$ , and then there might be a few tiny singular values contributing to the large condition number. The larger is the fractional order α, the more tiny singular values are in the spectrum. Hence, in the discrete setting, even for $\alpha = 3 / 4$ , the problem is still nearly well-posed, despite the large apparent condition number, since a few tiny singular values with a distinct gap from the rest of the spectrum are harmless in most regularization techniques. 

Physically this can also be observed in <sup>fi</sup>gure 6, where the forward map $F$ is from the Dirichlet boundary condition $x = 1$ to the <sup>fl</sup>ux boundary condition at $x = 0$ , in a piecewise linear <sup>fi</sup>nite element basis. Pictorially, the forward map F is only located in the upper left corner and has a triangular structure, which re<sup>fl</sup>ects the casual or Volterra nature of the sideways problem for the fractional diffusion equation. We note that the causal structure should be utilized in developing reconstruction techniques, via, e.g., Lavrentiev regularization [56]. For small α values, $\mathrm { e . g . } , \alpha = 1 / 4$ , the <sup>fi</sup>nite element basis at the right end point $x = 1$ is almost instantly transported to the left end point $x = 0$ , whose magnitude is slightly decreased, but with little diffusive effect, resulting a diagonally dominant forward map. However, as the fractional order α increases towards unity, the diffusive effect eventually kicks in, and the information spreads over the whole interval. Further, for large α values, it takes much longer time to reach the other side and there is a lag of information arrival, which explains the presence of tiny singular values. The larger is the fractional order $\alpha ,$ the smaller is the magnitude, $\mathrm { i . e . , }$ the less is the amount of the information reached the other side. Hence, one feasible approach is to recover only the boundary condition over a smaller subinterval of the measurement time interval. This idea underlies one popular engineering approach, the sequential function speci<sup>fi</sup>cation method [4, 64]. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/6aea54c7b768a060a0e0759d4d804b82c4f1da8e44de4c6e9b3a2e35c7f9e525.jpg)



(a) $\alpha = 1 / 4$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/776e7bffded0fcbd848f46612e71e392f638d66332edebc5a6c67b449d790065.jpg)



(b) $\alpha = 1 / 2$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/411ff560c13bde7ebda5ea0e6498bc8db594c32423c148c7ad8e9f45b4d6cbd0.jpg)



(c) $\alpha = 3 / 4$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/be6617f1972ca8f995536311a98eadf59ae3d15179ff41ac6599883119d4de8d.jpg)



(d) $\alpha = 1$



Figure 6. The Jacobian map F for $\alpha = 1 / 4 , 1 / 2 , 3 / 4$ and 1, from the interval (0, 1) itself.


The sideways problem for the classical diffusion has been extensively studied, and many ef<sup>fi</sup>cient numerical methods have been developed and analyzed [8, 11, 24, 25]. In the fractional case, however, there are only a few works on numerical schemes, mostly for onedimensional problems, and there seems no theoretical study on stability etc. Murio [76, 77] developed several numerical schemes, e.g., based on space marching and <sup>fi</sup>nite difference method, for the sideways problem, but without any analysis. Qian [85] discussed about the illposedness of the quarter plane formulation of the sideways problem using the Fourier analysis, based on which a molli<sup>fi</sup>er method was proposed, with error estimates provided. In [87], the recovery of a nonlinear boundary condition from the lateral Cauchy data was studied using an integral equation approach, and a convergent <sup>fi</sup>xed point iteration method was suggested. The in<sup>fl</sup>uence of the imprecise speci<sup>fi</sup>cation of the fractional order α on the reconstruction was examined. Zheng and Wei [113] proposed a molli<sup>fi</sup>cation method for the quarter plane formulation of the sideways problem, by convoluting the fractional derivative with a smooth kernel, and derived error estimates for the approximation, under a prior bounds on the solution. The Cauchy problem of the time fractional diffusion has been numerically studied in [114]. In particular, with the separation of variables, a Volterra integral equation reformulation of the problem was derived, from which the ill-posedness of the Cauchy problem follows directly. All these works are concerned with the one-dimensional case, and the high dimensional case has not been studied. 

## 3.3. Inverse source problem

A third classical linear inverse problem for the diffusion equation is the inverse source problem, i.e., the recovery of the source term f from lateral boundary data or <sup>fi</sup>nal time data. Clearly, one piece of boundary data or <sup>fi</sup>nal time data alone is insuf<sup>fi</sup>cient to uniquely determine a general source term, due to dimensional disparity. To restore the possible uniqueness, as usual, we look for only a space- or time-dependent component of the source term f. With different combinations of the data and source term, we get several different (and not equivalent) formulations of the inverse source problems. Below we examine several of them brie<sup>fl</sup>y. By the linearity of the forward problem, we without loss of generality, assume a zero initial data $\nu = 0$ and a zero potential $q = 0$ throughout this part. 

First, suppose we can measure the solution u at the <sup>fi</sup>nal time $t = T ,$ and aim at recovering either a space dependent or time dependent component of the source term f. Like before, we resort to the separation of variables. For the case of a space dependent only source term f(x), the solution u to the forward problem is given by 

$$
\begin{array}{c} u (t) = \sum_ {j = 1} ^ {\infty} \int_ {0} ^ {t} (t - \tau) ^ {\alpha - 1} E _ {\alpha , \alpha} \Big (- \lambda_ {j} (t - \tau) ^ {\alpha} \Big) \Big (f, \phi_ {j} \Big) \phi_ {j} \mathrm{d} \tau \\ = \sum_ {j = 1} ^ {\infty} \frac {1}{\lambda_ {j}} \Big (1 - E _ {\alpha , 1} \Big (- \lambda_ {j} t ^ {\alpha} \Big) \Big) \Big (f, \phi_ {j} \Big) \phi_ {j}. \end{array}
$$

Hence the measured data $g = u ( T )$ is given by 

$$
g = \sum_ {j = 1} ^ {\infty} \frac {1}{\lambda_ {j}} \left(1 - E _ {\alpha , 1} \left(- \lambda_ {j} T ^ {\alpha}\right)\right) \left(f, \phi_ {j}\right) \phi_ {j}.
$$

By taking inner product with $\phi _ { j }$ on both sides, we arrive at the following representation of the source term f in terms of the measured data g 

$$
f = \sum_ {j = 1} ^ {\infty} \lambda_ {j} \frac {(g , \phi_ {j})}{1 - E _ {\alpha , 1} (- \lambda_ {j} T ^ {\alpha})} \phi_ {j}.\tag{3.4}
$$

By the complete monotonicity of the Mittag-Lef<sup>fl</sup>er function $E _ { \alpha , 1 } ( - t )$ on the positive real axis <sup>+</sup> [82], we deduce 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/0d366bb9abc2f785a276549139d78b0be4002302195311b2df88eac4edd65d42.jpg)



(a) condition number


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/50a880caaf8aa70c8d0c1e0d470da5fd6172b340efaf6be2be41da9c6be6dc1c.jpg)



(b) singular value spectrum



Figure 7. Numerical results for the inverse source problem with <sup>fi</sup>nal time data and a space dependent source term. (a) The condition number versus the fractional order α, and (b) singular value spectrum at $T = 1$


$$
1 > E _ {\alpha , 1} \left(- \lambda_ {1} T ^ {\alpha}\right) > E _ {\alpha , 1} \left(- \lambda_ {2} T ^ {\alpha}\right),
$$

and thus the formula (3.4) is well de<sup>fi</sup>ned for any $T > 0 $ , and gives the precise condition for the existence of a source term. Even with a modest value of the terminal time T, the factor $1 - E _ { \alpha , 1 } ( - \lambda _ { 1 } T ^ { \alpha } )$ is close to unity for all small α values, especially for those close to zero. Each frequency component $( f , \phi _ { j } )$ differs from $( g , \phi _ { j } )$ essentially by a factor $\lambda _ { j } ,$ , which amounts to two derivative loss in space. Actually one can show 

$$
\| f \| _ {L ^ {2} (\Omega)} \leqslant c \| g \| _ {H ^ {2} (\Omega)}.
$$

This behavior is identical with that for the backward fractional diffusion. The statement holds also for the inverse source problem for the classical diffusion case. This is not surprising, since with a space dependent source term $f ,$ the solution u to the forward problem can be split into the steady solution $u _ { s }$ and the decaying solution $u _ { d } , { \mathrm { i . e . , ~ } } u = u _ { s } + u _ { d } .$ , where $u _ { s }$ and $u _ { d }$ solve 

$$
- u _ {s} ^ {\prime \prime} = f, u _ {s} (0) = u _ {s} (1) = 0,
$$

and 

$$
\partial_ {t} ^ {\alpha} u _ {d} - u _ {d, x x} = 0, \quad u _ {d} (0, x) = f (x), \quad u _ {d} (0, t) = u _ {d} (1, t) = 0,
$$

respectively. By the decay behavior of the solution $u _ { d } ,$ the steady state component $u _ { s }$ is dominating, which amounts to a two spatial derivative loss. This is fully con<sup>fi</sup>rmed by the numerical experiments, cf <sup>fi</sup>gure 7. It is observed that the condition number is almost independent of the fractional order $\alpha ,$ and it is of order $O ( 1 0 ^ { 3 } )$ , re<sup>fl</sup>ecting the mildly ill-posed nature of the inverse problem. In particular, for large terminal time $T ,$ the singular value spectra are almost identical for all fractional orders, decaying to zero at an algebraic rate, cf <sup>fi</sup>gure 7(b). 

Next we turn to the time dependent case, i.e., seeking a source term $f$ of the form $f ( x , t ) = p ( t ) q ( x )$ , with a known spacial component $q ( x )$ , from the <sup>fi</sup>nal time data $g = u ( T )$ ). Mathematically, the inverse problem even for the classical diffusion equation has not been completely analyzed. The inclusion of a nontrivial term $q ( x )$ is important since without this there is nonuniqueness. To see this, we take u to satisfy $u _ { t } - u _ { x x } = f ( t )$ on $( 0 , 1 ) \times ( 0 , T )$ with initial data $u ( x , 0 ) = 1$ and a homogeneous Neumann boundary condition $- u _ { x } ( 0 , t ) = u _ { x } ( 1 , t ) = 0$ . Then one solution satisfying $u ( x , T ) = g ( x ) = 1$ is given by $u ( x , t ) = 1$ and $f \equiv 0$ , but another is $u ( x , t ) = \cos { ( 2 \pi t / T ) }$ and $f = ( - 2 \pi / T ) \sin { ( 2 \pi t / T ) }$ Likewise, in the fractional case, we can take $u = \cos { ( 2 \pi t / T ) }$ for the second solution and de<sup>fi</sup>ne f to be its αth order Djrbashian–Caputo fractional derivative in time. 

Like previously, the solution u to (3.1) is given by 

$$
u (t) = \sum_ {j = 1} ^ {\infty} \int_ {0} ^ {t} (t - \tau) ^ {\alpha - 1} E _ {\alpha , \alpha} \Bigl (- \lambda_ {j} (t - \tau) ^ {\alpha} \Bigr) p (\tau) \mathrm{d} \tau \Bigl (q, \phi_ {j} \Bigr) \phi_ {j}.
$$

Hence the measured data $g ( x ) = u ( x , T )$ is given by 

$$
g (x) = \sum_ {j = 1} ^ {\infty} \int_ {0} ^ {T} (T - \tau) ^ {\alpha - 1} E _ {\alpha , \alpha} \left(- \lambda_ {j} (T - \tau) ^ {\alpha}\right) p (\tau) \mathrm{d} \tau \left(q, \phi_ {j}\right) \phi_ {j} (x).
$$

By taking inner product with $\phi _ { j }$ on both sides, we deduce 

$$
\Big (g, \phi_ {j} \Big) = \Big (q, \phi_ {j} \Big) \int_ {0} ^ {T} (T - \tau) ^ {\alpha - 1} E _ {\alpha , \alpha} \Big (- \lambda_ {j} (T - \tau) ^ {\alpha} \Big) p (\tau) \mathrm{d} \tau .
$$

In the case of $\alpha = 1$ , the formula recovers the relation 

$$
\left(g, \phi_ {j}\right) = \left(q, \phi_ {j}\right) \int_ {0} ^ {T} \mathrm{e} ^ {- \lambda_ {j} (T - \tau)} p (\tau) \mathrm{d} \tau ,
$$

which resembles a <sup>fi</sup>nite-time Laplace transform or moment problem, and thus severely smoothing, which renders the inverse source problem severely ill-posed. Intuitively, the term $\mathrm { e } ^ { - \lambda _ { j } ( T - t ) }$ can only pick up the information for t close to the terminal time T, and for t away from T, the information is severely damped, especially for high frequency modes, which leads to the severely ill-posed nature of the inverse problem. In the fractional case, the forward map F from the unknown to the data is clearly compact, and thus the problem is still ill-posed. However, the kernel $t ^ { \alpha - 1 } E _ { \alpha , \alpha } ( - \lambda _ { j } t ^ { \alpha } )$ is less smooth and decays much slower, and one might expect that the problem is less ill-posed than the canonical diffusion counterpart. To examine the point, we present the numerical results for the inverse problem in <sup>fi</sup>gure 8. It is severely ill-posed irrespective of the fractional order α: the singular values decay exponentially to zero without a distinct gap in the spectrum. In particular, for the terminal time $T = 1$ , the spectrum is almost identical for all fractional orders α. For small T, the singular values still decay exponentially, but the rate is different: the smaller is the fractional order $\alpha ,$ the faster is the decay, cf <sup>fi</sup>gure 8(a). Consequently, a few more modes of the source term $p ( \tau )$ might be recovered. In other words, due to a slower local decay of the exponential function $\mathrm { e } ^ { - \lambda t }$ compared with the Mittag-Lef<sup>fl</sup>er function $t ^ { \alpha - 1 } E _ { \alpha , \alpha } ( - \lambda t ^ { \alpha } )$ , cf <sup>fi</sup>gure 1(a), actually more frequency modes can be picked up by normal diffusion than the fractional counterpart, cf <sup>fi</sup>gure 8(a). This indicates that with suf<sup>fi</sup>ciently accurate data, at a small time instance, the sideways problem for normal diffusion may allow recovering more modes, i.e., anomalous diffusion does not help solve the inverse problem. 

In practice, the accessible data can also be the <sup>fl</sup>ux data at the end point, $\mathrm { e . g . , } x = 0$ or $x = 1$ . We brie<sup>fl</sup>y discuss the case of recovering a time dependent component p(t) in the source term $f = q ( x ) p ( t )$ from the <sup>fl</sup>ux data at $x = 0$ . By repeating the preceding argument, the data $g : = - u _ { x } ( 0 , t )$ is related to the unknown p(t) by 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/0e558ee33a139678e808de44917bc5023a31a229e57e02880af242be1c022386.jpg)



(a) $T = 0 . 0 1$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/5c68c07c93c8d3b6b0253d4b4d098fced937effd77d83a31f273d2450d28a22d.jpg)



(b) $T = 1$



Figure 8. The singular value spectrum at two different terminal times for the inverse source problem with a <sup>fi</sup>nal time data at terminal time T, and $f ( x , t ) = x p ( t )$ , an unknown time dependent component $p ( t )$


$$
g (t) = - \sum_ {j = 1} ^ {\infty} \int_ {0} ^ {t} (t - \tau) ^ {\alpha - 1} E _ {\alpha , \alpha} \Bigl (- \lambda_ {j} (t - \tau) ^ {\alpha} \Bigr) p (\tau) \mathrm{d} \tau \Bigl (q (x), \phi_ {j} \Bigr) \phi_ {j} ^ {\prime} (0).
$$

In [88, theorem 4.4], a stability result was established for the recovery of the time dependent component $p ( t )$ . Along the same line of thought, under reasonable assumptions, one can deduce that 

$$
\| p \| _ {C [ 0, T ]} \leqslant c \| \partial_ {t} ^ {\alpha} g \| _ {C [ 0, T ]}.
$$

The inverse problem roughly amounts to taking the αth order Djrbashian–Caputo fractional derivative in time. Hence as the fractional order α decreases from unity to zero, it becomes less and less ill-posed. For $\alpha$ close to zero, it is nearly well-posed, at least numerically. In other words, anomalous diffusion can mitigate the degree of ill-posedness for the inverse problem. To illustrate the discussion, we present in <sup>fi</sup>gure 9 some numerical results, where the forward map $F$ is from the time dependent component $p ( t )$ to the <sup>fl</sup>ux data $g ( t )$ at $x = 0$ , both de<sup>fi</sup>ned over the interval [0, ]T , discretized using a continuous piecewise linear <sup>fi</sup>nite element basis. The condition number of the discrete forward map $F$ decreases monotonically as the fractional order α decreases from unity to zero, con<sup>fi</sup>rming the preceding discussions. Further, the terminal time $T$ does not affect the condition number to a large extent. 

It is widely accepted in inverse heat conduction that an inverse problem will be severely ill-posed when the data and unknown are not aligned in the same space/time direction, and only mildly ill-posed when they do align with each other. Our discussions with the inverse source problems indicate that the observation remains valid in the time fractional diffusion case. In particular, although not presented, we note that the inverse source problem of recovering a space dependent component from the lateral Cauchy data is severely ill-posed for both fractional and normal diffusion. In the simplest case of a space dependent-only source term, it is mathematically equivalent to unique continuation, a well known example of severely ill-posed inverse problems. 

The inverse source problems for the classical diffusion equation have been extensively studied; see e.g., [7, 9, 37]. Inverse source problems for FDEs have also been numerically studied. Zhang and Xu [111] established the unique recovery of a space dependent source term in (3.1) with pure Neumann boundary data and overspeci<sup>fi</sup>ed Dirichlet data at $x = 0$ . This is achieved by an eigenfunction expansion and Laplace transform, and the uniqueness follows from a unique continuation principle of analytic functions. Sakamoto and Yamamoto [89] discussed the inverse problem of determining a spatially varying function of the source term by <sup>fi</sup>nal overdetermined data in multi-dimensional space, and established its well-posedness in the Hadamard sense except for a discrete set of values of the diffusion constant, using an analytic Fredholm theory. Very recently, Luchko et al [68] showed the uniqueness of recovering a nonlinear source term from the boundary measurement, and developed a numerical scheme of <sup>fi</sup>xed point iteration type. Aleroev et al [2] showed the uniqueness of recovering a space dependent source term from integral type observational data. Recently, there are many numerical studies on this class of inverse problems. In [101], the numerical recovery of a spatially varying function of the source term from the <sup>fi</sup>nal time data in a general domain was studied using a quasi-boundary value problem method; see also [98, 112] related studies. Wang et al [100] proposed to determine the space-dependent source term from the <sup>fi</sup>nal time data in multi-dimension using a reproducing kernel Hilbert space method. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/33536d69699eb93cb7517b50921f0ccd4201767d586d3cc383e537fc86fb8bb8.jpg)



(a) condition number


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/21536908b21dc3a3e8b355ba0022bbb28fe45c019057b73a9b40939529a91abd.jpg)



(b) singular values



Figure 9. Numerical results for the inverse source problem with <sup>fl</sup>ux data at $x = 0$ and $f ( x , t ) = x p ( t )$ , an unknown time dependent component $p ( t ) .$ . (a) The condition number of the discrete forward map and (b) singular value spectrum at $T = 1$


## 3.4. Inverse potential problem

Now we consider a nonlinear inverse coef<sup>fi</sup>cient problem for the time fractional diffusion equation: given the <sup>fi</sup>nal time data $g = u ( T )$ ), <sup>fi</sup>nd the potential $q$ in the model 

$$
\partial_ {t} ^ {\alpha} u - u _ {x x} + q u = 0 \quad \mathrm{in} \Omega ,\tag{3.5}
$$

with a homogeneous Neumann boundary condition and initial data v. The parabolic counterpart has been extensively studied [15, 16, 38], where it was shown that the problem is nearly well-posed in the Hardamard sense in suitable Hölder space, under certain conditions, using the strong maximum principle. In [38], an elegant <sup>fi</sup>xed point method was developed, and the monotone convergence of the method was established. It can be adapted straightforwardly to the fractional case: given an initial guess $q ^ { 0 } ,$ compute the update $q ^ { k }$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/6c29d5cf307b0d0d2b75d9987682511c376b138cfebd77a3215da0cda6b47705.jpg)



(a) error at $T = 0 . 1$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/ca1c05dc2a8afb1b87f13996567b1e0cbfabdcfe836a1593b5c64ca07d58b6df.jpg)



(b) error at $\alpha = 1 / 2$



Figure 10. Numerical results, i.e., the relative $L ^ { 2 } ( \varOmega )$ error $e ,$ for the inverse potential problem from exact <sup>fi</sup>nal time data at (a) $T = 0 . 1$ and $( \mathbf { b } ) \alpha = 1 / 2$


recursively by 

$$
q ^ {k + 1} = \frac {g ^ {\prime \prime} - \partial_ {t} ^ {\alpha} u (x , T ; q ^ {k})}{g},
$$

where the notation $u ( x , T ; q ^ { k } )$ denote the solution to problem (3.5) with the potential $q ^ { k }$ at $t = T$ . Since the strong maximum principle is still valid for the time fractional diffusion equation [110], the scheme is monotonically convergent, under suitable conditions. 

As the terminal time $T \to \infty$ , the problem recovers a steady-state problem, and the scheme amounts to twice numerical differentiation in space and converges within one iteration, provided that the data $g$ is accurate enough. Hence, it is natural to expect that the convergence of the scheme will depends crucially on the time T: the larger is the time $T ,$ the closer is the solution $u$ to the steady state solution; and thus the faster is the convergence of the <sup>fi</sup>xed point scheme. By lemma 2.1, as the fractional order α approaches zero, the solution u decays much faster around $t = 0$ than the classical one, i.e., $\alpha = 1$ . In other words, the fractional diffusion problem can reach a ‘quasi-steady state’ much faster than the classical one, especially for α close to zero, and the scheme will then converge much faster. 

To illustrate the point, we present in <sup>fi</sup>gure 10 some numerical results of reconstructing a discontinuous potential $q = 1 + 2 x \chi _ { [ 0 , 0 . 5 ] } + 2 ( 1 - \chi _ { ( 0 . 5 , 1 ] } x )$ (with $\chi _ { S }$ being the characteristic function of the set S). In order to illustrate the convergence behavior of the <sup>fi</sup>xed point scheme we take exact data. In the <sup>fi</sup>gure, e denotes the relative $L ^ { 2 } ( \varOmega )$ error. The numerical results fully con<sup>fi</sup>rm the preceding discussions: at a <sup>fi</sup>xed time $T ,$ the smaller is the fractional order α, the faster is the convergence; and at <sup>fi</sup>xed $\alpha ,$ the larger is the time $T ,$ the faster is the convergence. Numerically, one also observes the monotone convergence of the scheme. 

Generally, the recovery of a coef<sup>fi</sup>cient in FDEs has not been extensively studied. Cheng et al [14] established the unique recovery of the fractional order $\alpha \in ( 0 , 1 )$ and the diffusion coef<sup>fi</sup>cient from the lateral boundary measurements. It represents one of the <sup>fi</sup>rst mathematical works on invere problems for FDEs, and has inspired many follow-up works. Yamamoto and Zhang [109] established conditional stability in determining a zeroth-order coef<sup>fi</sup>cient in a one-dimensional FDE with one half order Caputo derivative by a Carleman estimate. Carleman estimates for time fractional diffusion were discussed in [13, 62, 108]. In [73], the unique determination of the spatial coef<sup>fi</sup>cient and/or the fractional order from the data on a subdomain was shown for a positive initial condition. Wang and Wu [97] studied the simultaneous recovery of two time varying coef<sup>fi</sup>cients, i.e., a kernel function and a source function, from the additional integral observation in multi-dimension, using a <sup>fi</sup>xed point theorem. All these works are concerned with the theoretical analysis, ant there are even fewer works on the numerical analysis of related inverse problems. Li et al [58] suggested an optimal perturbation algorithm for the simultaneous numerical recovery of the diffusion coef<sup>fi</sup>cient and fractional order in a one-dimensional time fractional FDE. In [50], the authors considered the identi<sup>fi</sup>cation of a potential term from the lateral <sup>fl</sup>ux data at one <sup>fi</sup>xed time instance corresponding to a complete set of source terms, and established the unique determination for ‘small’ potentials. Further, a Newton type method was proposed in [50], and its convergence was shown. 

Even though our discussions have focused on time fractional diffusion, which involves one single fractional derivative in time, it is also possible to consider equations where the time derivative involves multiple factional orders, i.e., $\begin{array} { r } { \sum _ { k = 1 } ^ { m } c _ { k } \partial _ { t } ^ { \alpha _ { k } } } \end{array}$ for a sequence $\alpha _ { 1 } > \alpha _ { 2 } > \ldots > \alpha _ { m }$ [44, 60]; see [59] for some <sup>fi</sup>rst uniqueness results for inverse coef<sup>fi</sup>cient problems in the multi-dimensional case. Further extensions include the distributed-order, spatially and/or temporally variable-order and tempered fractional diffusion, to better capture certain physical processes, for which, however, related inverse problems have not been discussed at all. 

## 3.5. Fractional derivative as an inverse solution

One of the very <sup>fi</sup>rst undetermined coef<sup>fi</sup>cient problems for PDEs was discussed in the paper by Jones [52] (see also [8, chapter 13]). This is to determine the coef<sup>fi</sup>cient a(t) from 

$$
\begin{array}{c} {u _ {t} = a (t) u _ {x x}, \quad 0 <   x <   \infty , \quad t > 0} \\ {u (x, 0) = 0, \quad - a (t) u _ {x} (0, t) = g (t), \quad 0 <   t <   T} \end{array}
$$

under the over-posed condition of measuring the ‘temperature’ at $x = 0$ 

$$
u (0, t) = \psi (t)
$$

In [52], Jones provided a complete analysis of the problem, by giving necessary and suf<sup>fi</sup>cient conditions for a unique solution as well as determining the exact level of ill-conditioning. The key step in the analysis is a change of variables and conversion of the problem to an equivalent integral equation formulation. Perhaps surprisingly, this approach involves the use of a fractional derivative as we now show. 

The assumptions are that $g$ is continuous and positive and ψ is continuously differentiable with $\psi ( 0 ) = 0$ and $\psi ^ { \prime } > 0$ on $( 0 , T )$ . In addition, the function h(t) de<sup>fi</sup>ned by 

$$
h (t) = \frac {\sqrt {\pi} g (t)}{\int_ {0} ^ {t} (t - \tau) ^ {- 1 / 2} \psi^ {\prime} (\tau) \mathrm{d} \tau}
$$

satis<sup>fi</sup>es lim $_ { ! t  0 } h ( t ) = h _ { 0 } > 0$ . Note that h is the ratio of the two data functions; the <sup>fl</sup>ux $g$ and the Djrbashian–Caputo derivative of order 1 2 of ψ. If we de<sup>fi</sup>ne $h _ { i } = \operatorname { i n f } h$ and $h _ { s } = \operatorname { s u p }$ h on $[ 0 , T ]$ and look at the space $\mathcal { G } : = \{ a \in C [ 0 , T ) \colon h _ { i } ^ { 2 } \leqslant a ( t ) \leqslant h _ { s } ^ { 2 } \}$ }, then it was shown that any $a \in { \mathcal { G } }$ satis<sup>fi</sup>es the inverse problem must also solve the integral equation 

$$
a (t) = \frac {\sqrt {\pi} g (t)}{\int_ {0} ^ {t} \psi^ {\prime} (\tau) \left[ \int_ {\tau} ^ {t} a (s) \mathrm{d} s \right] ^ {- 1 / 2} \mathrm{d} \tau} =: \mathcal {T} a,
$$

and vice-versa. The main result in [52] is that the operator  has a unique <sup>fi</sup>xed point on $\mathcal { G }$ and indeed  is monotone in the sense of preserving the partial order on , i.e., if $a _ { 1 } \geqslant a _ { 2 }$ then $\tau _ { a _ { 1 } } \leqslant \tau _ { a _ { 2 } }$ 

Given these developments, it might seem that a parallel construction for the time fractional diffusion counterpart, $\partial _ { t } ^ { \alpha } u = a ( t ) u _ { x x }$ , would be relatively straightforward but this seems not to be the case. The basic steps for the parabolic version require items that just are not true in the fractional case, such as the product rule, and without these the above structure cannot be replicated or at least not without some further ingenuity. 

## 4. Inverse problems for space fractional diffusion

Now we turn to differential equations involving a fractional derivative in space. There are several possible choices of a fractional derivative in space, e.g., Djrbashian–Caputo fractional derivative, Riemann–Liouville fractional derivative, Riesz derivative, and fractional Laplacian [3], which all have received considerable attention. In recent years, the use of the fractional Laplacian is especially popular in high-dimensional spaces, and admits a welldeveloped analytical theory. We shall focus on the left-sided Djrbashian–Caputo fractional derivative ${ } _ { 0 } ^ { C } D _ { x } ^ { \beta ^ { \ast } } , \beta \in ( 1 , 2 )$ , and the one-dimensional case, and consider the following four inverse problems: inverse Sturm–Liouville problem, Cauchy problem for a fractional elliptic equation, backwards diffusion, and sideways problem. 

## 4.1. Inverse Sturm–Liouville problem

First we consider the following Sturm–Liouville problem on the unit interval $\varOmega = ( 0 , 1 )$ : <sup>fi</sup>nd $u \in H _ { 0 } ^ { 1 } ( \varOmega ) \cap H ^ { \beta } ( \varOmega )$ and $\lambda \in \mathbb { C }$ such that 

$$
- _ {0} ^ {C} D _ {x} ^ {\beta} u + q u = \lambda u \quad \mathrm{in} \Omega ,\tag{4.1}
$$

with a homogeneous Dirichlet boundary condition $u ( 0 ) = u ( 1 ) = 0$ . A Sturm–Liouville problem of this form was considered by Djrbashian [19, 22] in 1960s to construct certain biorthogonal basis for spaces of analytic functions; see also [78]. Like before, with $\beta = 2$ , it recovers the classical Sturm–Liouville problem. In the case of a general potential $q ,$ in the fractional case, little is known about the analytical properties of the eigenvalues and eigenfunctions. For the case of a zero potential $q = 0$ , there are countably many eigenvalues $\{ \lambda _ { j } \}$ to (4.1), which are zeros of the Mittag-Lef<sup>fl</sup>er function $E _ { \beta , 2 } ( - \lambda )$ . The corresponding eigenfunctions are given by $x E _ { \beta , 2 } ( - \lambda _ { j } x ^ { \beta } )$ . Using the exponential asymptotics on the Mittag-Lef<sup>fl</sup>er function in lemma 2.1, one [50, 93] can show that asymptotically, the eigenvalues $\lambda _ { j }$ are distributed as 

$$
\left| \lambda_ {j} \right| \sim (2 \pi j) ^ {\beta} \quad \mathrm{and} \quad \arg (\lambda_ {j}) \sim \frac {(2 - \beta) \pi}{2}.
$$

Hence, for any $\beta \in ( 1 , 2 )$ , there are only a <sup>fi</sup>nite number of real eigenvalues to (4.1), and the rest appears as complex conjugate pairs. 

It is well known that eigenvalues contain valuable information about the boundary value problem. For example it is known that the sequence of Dirichlet eigenvalues can uniquely determine a potential $q$ symmetric with respect to the point $x = 1 / 2$ , and together with additional spectral information, one can uniquely determine a general potential $q ;$ see [12, 86] for an overview of results on the classical inverse Sturm–Liouville problem. In the fractional case, the eigenvalues are generally genuinely complex, and a complex number may carry more information than a real one. Thus one naturally wonders whether these complex eigenvalues do contain more information about the potential. Numerically the answer is af<sup>fi</sup>rmative. To illustrate this, we show some numerical reconstructions in <sup>fi</sup>gure 11, obtained by using a frozen Newton method and representing the sought-for potential $q$ in Fourier series [50]. The Dirichlet eigenvalues can be computed ef<sup>fi</sup>ciently using a Galerkin <sup>fi</sup>nite element method [45]. One observes that one single Dirichlet spectrum can uniquely determine a general potential $q .$ Unsurprisingly, as the fractional order $\beta$ tends two, the reconstruction becomes less and less accurate, since in the limit $\beta = 2$ , the Dirichlet spectrum cannot uniquely determine a general potential $q .$ Theoretically, the surprising uniqueness in the fractional case remains to be established. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/8a089db5d660f89113dd434359a796db710f4c9aad476bfa4248533646f72c0b.jpg)



(a) β = 5/3


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/c5d566a3f4ff6485e95f72d8dea8e19e79e09c0a108c9067bb821088f559835f.jpg)



(b) $\beta = 7 / 4$



Figure 11. Numerical results for the inverse Sturm–Liouville problem with a Djrbashian–Caputo derivative for (a) $\beta = 5 / 3$ and (b) $\beta = 7 / 4$ . The reconstructions are computed from the <sup>fi</sup>rst eight eigenvalues (in absolute value) using a frozen Newton method [50].


Naturally, one can also consider the Riemann–Liouville case: 

$$
- _ {0} ^ {R} D _ {x} ^ {\beta} u + q u = \lambda u \quad \mathrm{in} \Omega ,\tag{4.2}
$$

with $u ( 0 ) = u ( 1 ) = 0$ . Like before, little is known about the analytical properties of the eigenvalues and eigenfunctions. In the case of a zero potential $q = 0 ;$ , there are countably many eigenvalues to (4.2), which are zeros of the Mittag-Lef<sup>fl</sup>er function $E _ { \beta , \beta } ( - \lambda )$ , and the corresponding eigenfunctions are given by $x ^ { \beta - 1 } E _ { \beta , \beta } ( - \lambda _ { j } x ^ { \beta } )$ . Further, the asymptotics of the eigenvalues are still valid. Hence, for any $\beta \in ( 1 , 2 )$ , there are only a <sup>fi</sup>nite number of real eigenvalues to (4.2), and the rest appears as complex conjugate pairs. 

The numerical results from the Dirichlet spectrum in the Riemann–Liouville case are shown in <sup>fi</sup>gure 12. For a general potential $q ,$ the reconstruction represents only the symmetric part, which is drastically different from the Djrbashian–Caputo case, but identical with that for the classical Sturm–Liouville problem. Further, if we assume that the potential $q$ is known on the left half interval, then the Dirichlet spectrum allows uniquely reconstructing the potential $q$ on the remaining half interval, cf <sup>fi</sup>gure 12(b). These results indicate that in the 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/0ddecce0da4e8b68797058c2418090207f074da352008a2cee934688d841282e.jpg)



(a) whole interval


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/5d878261bd1a026cf807ce0addb03b6c46978a832a47b696ca2e2363949a20d3.jpg)



(b) half interval



Figure 12. Numerical results for the inverse Sturm–Liouville problem with a Riemann– Liouville fractional derivative of order $\beta = 4 / 3$ . The reconstructions are computed from the <sup>fi</sup>rst eight eigenvalues (in absolute value) using a frozen Newton method [50].


Riemann–Liouville case the complex spectrum is not more informative than the classical Sturm–Liouville problem. The precise mechanism underlying the fundamental differences between the Djrbashian–Caputo and Riemann–Liouville cases awaits further study. However, as in the classical case $\beta = 2$ , one can show that the linearized derivative of the map $q \to u ( 1 ; \lambda , q )$ around $q = 0$ cannot span more than the subspace of even functions in $L ^ { 2 } ( \varOmega )$ 

In general, the Sturm–Liouville problem with a fractional derivative remains completely elusive, and numerical methods such as <sup>fi</sup>nite element method [46] provide a valuable (and often the only) tool for studying its analytical properties. For a variant of the fractional Sturm– Liouville problem, which contains a fractional derivative in the lower-order term, Malamud [71] established the existence of a similarity transformation, analogous to the well-known Gelʼfand–Levitan–Marchenko transformation, and also the unique recovery of the potential from multiple spectra. In the classical case, the Gelʼfand–Levitan–Marchenko transformation lends itself to a constructive algorithm [86]; however, it is unclear whether this is true in the fractional case. In [50], the authors proposed a Newton type method for reconstructing the potential, which numerically exhibits very good convergence behavior. However, a rigorous convergence analysis of the scheme is still missing. Further, the uniqueness and nonuniqueness issues of related inverse Sturm–Liouville problems are outstanding. Last, as noted above, there are other possible choices of the space fractional derivative, $\mathrm { e . g . }$ , fractional Laplacian and Riesz derivative. It is unknown whether the preceding observations are valid for these alternative derivatives. 

## 4.2. Cauchy problem for fractional elliptic equation

One classical elliptic inverse problem is the Cauchy problem for the Laplace equation, which plays a fundamental role in the study of many elliptic inverse problems [40]. A <sup>fi</sup>rst example was given by Jacques Hadamard [31] to illustrate the severe ill-posedness of the Cauchy problem, which motivated him to introduce the concept of well-posedness and ill-posedness for problems in mathematical physics. So a natural question is whether the Cauchy problem for the fractional elliptic equation is also as ill-posed? To illustrate this, we consider the following fractional elliptic problem on the rectangular domain $\varOmega = \{ ( x , y ) \in \mathbb { R } ^ { 2 }$ $0 < x < 1 , 0 < y < 1 \}$ 

$$
{ } _ { 0 } ^ { C } D _ { x } ^ { \beta } u + { } _ { 0 } ^ { C } D _ { y } ^ { \beta } u = 0 \text {   in   } \Omega ,\tag{4.3}
$$

with the fractional order $\beta \in ( 1 , 2 )$ and the Cauchy data 

$$
u (x, 0) = g (x) \quad \text { and } \quad \frac {\partial u}{\partial \nu} (x, 0) = h (x), \quad 0 <   x <   1,
$$

where $\nu$ is the unit outward normal direction. With $\beta = 2 ,$ , it recovers the Cauchy problem for the Laplace equation. By applying the separation of variables, we can assume that $u ( x , y ) = \phi ( x ) \psi ( y )$ , which directly gives for some scalar $\lambda \in \mathbb { C }$ that 

$$
\begin{array}{l} _ {0} ^ {C} D _ {x} ^ {\beta} \phi (x) = - \lambda \phi (x), \\ _ {0} ^ {C} D _ {y} ^ {\beta} \psi (y) = \lambda \psi (y), \end{array}
$$

Let $( \lambda _ { j } , \phi _ { j } )$ be a Dirichlet eigenpair of the Caputo derivative operator $- _ { 0 } ^ { C } D _ { x } ^ { \beta }$ on the unit interval $D = ( 0 , 1 )$ , i.e., $\phi _ { j } ( x ) = x E _ { \beta , 2 } ( - \lambda _ { j } x ^ { \beta } )$ , and $| \lambda _ { j } | \to$ ∞ as $j \to \infty [ 5 1 ] ;$ see section 4.1 for further details. With the choice $\phi = \phi _ { j }$ and the Cauchy data pair $( g , h _ { j } ) = ( 0 , - x E _ { \beta , 2 } ( \lambda _ { j } x ^ { \beta } ) / \lambda _ { j } )$ , the component $\psi _ { j }$ satis<sup>fi</sup>es 

$$
{ } _ { 0 } ^ { C } D _ { y } ^ { \beta } \psi _ { j } ( y ) = \lambda _ { j } \psi _ { j } ( y ) \text {~ in~ } y \in ( 0 , \infty ) ,
$$

with the initial condition $\psi _ { i } ( 0 ) = 0$ and $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } y } \psi _ { j } ( 0 ) = 1 / \lambda _ { j } } \end{array}$ . Using the relation $\begin{array} { r } { \frac { \mathrm { ~ d ~ ~ ~ } } { \mathrm { ~ d ~ } x } x ^ { \gamma - 1 } E _ { \beta , \gamma } ( \lambda x ^ { \beta } ) = \lambda x ^ { \gamma - 2 } E _ { \beta , \gamma - 1 } ( \lambda x ^ { \beta } ) } \end{array}$ [53, p 46], we deduce that the solution $\psi _ { j }$ to the fractional ordinary differential equation is given by 

$$
\psi_ {j} (y) = y E _ {\beta , 2} \Big (\lambda_ {j} y ^ {\beta} \Big) / \lambda_ {j}.
$$

Hence, $u _ { j } ( x , y ) = x E _ { \beta , 2 } ( - \lambda _ { j } x ^ { \beta } ) y E _ { \beta , 2 } ( \lambda _ { j } y ^ { \beta } ) / \lambda _ { j } ^ { 2 }$ is a solution to the Cauchy problem with $g = 0$ and $h _ { j } ( x ) = - { x } E _ { \beta , 2 } ( - \lambda _ { j } x ^ { \beta } ) / \lambda _ { j } .$ By the exponential asymptotics of the Mittag-Lef<sup>fl</sup>er function, cf lemma 2.1, we deduce that $h _ { j } ( x ) \to 0 { \mathrm { ~ a s ~ } } j \to \infty$ , whereas for any $y > 0$ , the solution $u _ { j } ( x , y ) $ ∞ as $j  \infty$ , in view of the exponential growth of the Mittag-Lef<sup>fl</sup>er function $E _ { \beta , 2 } ( z )$ , cf lemma 2.1. This indicates that the Cauchy problem for the fractional elliptic equation is also exponentially ill-posed. However, the interesting question of the degree of ill-posedness, in comparison withthe classical case, is unclear and certainly worthy of further study. Further, we note that the numerical solution of the fractional elliptic equation (4.3) is highly nontrivial, and there seems to be no ef<sup>fi</sup>cient yet rigorous solver in the literature and this seems to be due to a lack of theory about the solution to such problems. 

## 4.3. Backward problem

Now we return to the backward diffusion problem with fractional derivatives in the space variable(s). Let $\varOmega = ( 0 , 1 )$ be the unit interval. Then the one-dimensional space fractional diffusion equation is given by 

$$
u _ {t} - \mathbf {\Lambda} _ {0} ^ {C} D _ {x} ^ {\beta} u = 0, (x, t) \in \Omega \times (0, \infty),
$$

where the fractional order $\beta \in ( 1 , 2 )$ . The equation is equipped with the following initial condition $u ( x , 0 ) = \nu$ and zero boundary condition $u ( 0 , t ) = u ( 1 , t ) = 0$ . The backward problem is: given the <sup>fi</sup>nal time data $g ( x ) = u ( x , T )$ , <sup>fi</sup>nd the initial data v. Since the Djrbashian–Caputo derivative operator $^ C _ { 0 } \bar { D } _ { x } ^ { \beta }$ with the zero Dirichlet boundary is sectorial on suitable spaces [42], the existence of a solution u follows from the analytic semigroup theory [43, 81], and formally it can be represented by 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/230fb3e168c2b18c07134e25f9281480753f995cbda514754191a4137b02e41f.jpg)



(a) $T = 0 . 0 1$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/78b5756fdad515510dad2a85b71ec85c82c107112d3bc42a27031ec182472fa8.jpg)



(b) $T = 0 . 1$



Figure 13. Numerical results for the space fractional backward diffusion problem, the singular value spectrum at two different time instances, (a) $T = 0 . 0 1$ and (b) $T = 0 . 1$


$$
u (t) = \mathrm{e} ^ {- A t} v,
$$

where A is the representation of the Djrbashian–Caputo derivative operator $- _ { 0 } ^ { C } D _ { x } ^ { \beta }$ on its domain. Formally, the solution v to the space fractional backward problem is given by 

$$
v = \mathrm{e} ^ {A T} g.
$$

In case of $\beta = 2 ,$ , using the eigenpairs $\{ ( \lambda _ { j } , \phi _ { j } ) \}$ and the $L ^ { 2 } ( \varOmega )$ orthogonality of the eigenfunctions $\{ \phi _ { j } \}$ , it recovers the well known formula 

$$
v = \sum_ {j = 1} ^ {\infty} \mathrm{e} ^ {\lambda_ {j} T} \Big (g, \phi_ {j} \Big) \phi_ {j}.
$$

The growth factor $\mathrm { e } ^ { \lambda _ { j } T }$ explains the severely ill-posed nature of the inverse problem. In the fractional case, such an explicit representation is no longer available since the corresponding eigenfunctions $\{ \phi _ { j } \}$ are not orthogonal in $L ^ { 2 } ( \varOmega )$ (actually they can be almost linearly dependent), due to the non self adjoint nature of the Djrbashian–Caputo derivative operator $- _ { 0 } ^ { { \dot { C } } } D _ { x } ^ { \beta }$ . Nonetheless, according to the discussions in section 4.1, the eigenvalues $\{ \lambda _ { j } \}$ increase to in<sup>fi</sup>nity with the index $j ,$ and asymptotically lies on two rays. Hence, one naturally expects that the backward problem is also exponentially ill-posed. However, the magnitudes (and the real parts) of the eigenvalues grow at a rate slower than that of the standard Sturm–Liouville problem, and thus the space fractional backward problem is less ill-posed than the classical one. To illustrate the point, we present the numerical results in <sup>fi</sup>gure 13. For all fractional orders $\beta ,$ the singular values decay exponentially, but the decay rate increases dramatically with the increase of the fractional order $\beta$ and the terminal time T. Hence, anomalous superdiffusion does not change the exponentially ill-posed nature of the backward problem, but numerically it does enable recovering more Fourier modes of the initial data v. 

Last, we note that for other choices of the fractional derivative, e.g., the Riemann– Liouville fractional derivative and the fractional Laplacian [6, 55], the magnitude of eigenvalues of the operator also tends to in<sup>fi</sup>nity, and the growth rate increases with the fractional order $\beta .$ Therefore, the preceding observations on the space fractional backward problem are expected to be valid for these choices as well. 

## 4.4. Sideways problem

Last we return to the classical sideways diffusion problem but now with a fractional derivative in space rather than in time. Let $\varOmega = ( 0 , 1 )$ be the unit interval. Then the onedimensional space fractional diffusion equation is given by 

$$
u _ {t} - \mathbf {\Lambda} _ {0} ^ {C} D _ {x} ^ {\beta} u = 0, (x, t) \in \Omega \times (0, \infty),
$$

where the fractional order $\beta \in ( 1 , 2 )$ . The equation is equipped with an initial condition $u ( x , 0 ) = 0$ and the following lateral Cauchy boundary conditions 

$$
u (0, t) = f (t) \quad \text { and } \quad u _ {x} (0, t) = g (t), \quad t > 0.
$$

We wish to compute the solution at $x = 1$ , i.e., $h ( t ) : = u ( 1 , t )$ . In the case $\beta = 2$ , the model recovers the standard diffusion equation, and we have already discussed the severe illconditioning of the classical case. Due to the nonlocal nature of the fractional derivative, one might expect that in the space fractional case, the sideways problem is less ill-posed. To see this, we take Laplace transform in time to arrive at $( \mathrm { w i t h } \ ^ { \wedge }$ denoting the Laplace transform) 

$$
z \hat {u} (x, z) - _ {0} ^ {C} D _ {x} ^ {\beta} \hat {u} (x, z) = 0,
$$

with the initial conditions $( \mathrm { a t } x = 0 )$ 

$$
\widehat {u} (0, z) = \widehat {f} (z) \quad \text {and} \quad \widehat {u} _ {x} (0, z) = \widehat {g} (z).
$$

The solution $\widehat { u } \left( x , z \right)$ to the initial value problem is given by 

$$
\widehat {u} (x, z) = \widehat {f} (z) E _ {\beta , 1} \left(z x ^ {\beta}\right) + \widehat {g} (z) x E _ {\beta , 2} \left(z x ^ {\beta}\right)
$$

and thus 

$$
\widehat {h} (z) = \widehat {f} (z) E _ {\beta , 1} (z) + \widehat {g} (z) E _ {\beta , 2} (z).
$$

Like before, the boundary condition h(t) at $x = 1$ can be found by an inverse Laplace transform 

$$
h (t) = \int_ {\mathrm{Br}} \mathrm{e} ^ {z t} \widehat {h} (z) \mathrm{d} z.
$$

In case of $\beta = 2$ , this gives cosh $\sqrt { z }$ and sinh $\sqrt { z } / \sqrt { z }$ multipliers to the data $\widehat { f } \left( z \right)$ and $\widehat g \left( z \right)$ resulting in the exponential ill-conditioning of the sideways heat problem. In the case of a general $\beta \in ( 1 , 2 )$ , the exponential asymptotics in lemma 2.1 indicates that the problem still suffers from exponentially growing multipliers to the data, and thus the problem is still severely ill-conditioned. Simple computation shows that the multiplier is asymptotically larger for the fractional order $\beta$ closer to unity. In other words, anomalous diffusion in space does not mitigate the ill-conditioned nature of the sideways problem, but actually worsens the conditioning severely. 

To further illustrate the point, we compute the forward map F from the Dirichlet boundary condition at $x = 1$ to the <sup>fl</sup>ux at $x = 0$ numerically with a <sup>fi</sup>nite element in space and <sup>fi</sup>nite difference in time scheme, cf appendix ${ \mathrm { A } } . 3$ for the details. The numerical results are presented in <sup>fi</sup>gure 14. The singular value spectra clearly show the ill-posedness nature of the space fractional sideways problem: as the fractional order $\beta$ increases from one to two, the majority of the singular values move upward, the decay of the singular values slows down, and thus the sideways problem becomes less and less ill-posed (but still severely so). Further, there are more tiny singular values kicking in as the fractional order $\beta$ decreases to one, which indicates the inherent rank de<sup>fi</sup>ciency of the forward map F and might be relevant in the uniqueness of the inverse problem. This con<sup>fi</sup>rms the preceding analysis: the degree of illposedness worsens with the decrease of the fractional order $\beta ,$ and the fractional counterpart is more ill-posed than the classical one. In other words, anomalous diffusion actually severely worsens the conditioning of the already very ill-posed sideways problem. Further, the numerical results tend to indicate that the Djrbashian–Caputo derivative with an order $\beta \in ( 1 , 2 )$ acts as an interpolation between the diffusion and convection, which results in a history mechanism in space: when the history piece runs from the left to the right, it is unlikely to propagate the information in the reverse direction; and the closer is the fractional order β to unity, the stronger is the directional effect. The latter is not counterintuitive, since in the limit of $\beta = 1$ , the Djrbashian–Caputo fractional derivative ${ } _ { 0 } ^ { C } D _ { x } ^ { \beta }$ u recovers the <sup>fi</sup>rst order derivative ${ \frac { \partial u } { \partial x } } ,$ , and the problem is of convection type, and surely no information can be convected backwards! 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/717ba0a48a8f2fdc1ed9f100144830b5f408c52186e08c78ac3beb710157b76c.jpg)



(a) $T = 0 . 1$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/35d29d02c432da0b51daf7287e7e0c8b49a6fdda44ca612d3a6636f624415eb2.jpg)



(b) $T = 1$



Figure 14. Singular value spectrum of the forward map F at times $T = 0 . 1$ and $T = 1$ , for the sideways problem with Cauchy data at $x = 0$


In the case $\beta = 2$ , one may equally measure the lateral Cauchy data at $x = 1$ , and aims at recovering the Dirichlet boundary condition at $x = 0$ . Clearly, this does not change the nature of the inverse problem, and it is equally ill-posed. Due to the directional nature of the Djrbashian–Caputo derivative ${ } _ { 0 } ^ { C } D _ { x } ^ { \beta }$ , one naturally wonders whether this ‘directional’ feature does in<sup>fl</sup>uence the ill-posed nature of the sideways problem. To illustrate the point, we repeat the preceding arguments and deduce 

$$
z \hat {u} (x, z) - _ {0} ^ {C} D _ {x} ^ {\beta} \hat {u} (x, z) = 0,
$$

with the boundary conditions at $x = 1$ 

$$
\widehat {u} (1, z) = \widehat {f} (z) \quad \text { and } \quad \widehat {u} _ {x} (1, z) = \widehat {g} (z).
$$

To derive the solution, denote the initial conditions at $x \ = \ 0$ by $\tilde { f } \left( z \right) = \hat { u } \left( 0 , z \right)$ and $\tilde { g } ( z ) = \widehat { u } _ { x } ( 0 , z )$ Then the solution $\widehat { u } \left( x , z \right)$ to the initial value problem is given by 

$$
\hat {u} (x, z) = \tilde {f} (z) E _ {\beta , 1} \left(z x ^ {\beta}\right) + \tilde {g} x E _ {\beta , 2} \left(z x ^ {\beta}\right).
$$

Use the differentiation formula ${ \scriptstyle { \frac { \mathrm { d } } { \mathrm { d } x } } } x ^ { \gamma - 1 } E _ { \beta , \gamma } ( z x ^ { \beta } ) = z x ^ { \gamma - 2 } E _ { \beta , \gamma - 1 } ( z x ^ { \beta } )$ [53, p 46] we deduce that at $x = 1$ , there hold 

$$
\begin{array}{r l} & {\tilde {f} (z) E _ {\beta , 1} (z) + \tilde {g} E _ {\beta , 2} (z) = \widehat {f} (z),} \\ & {\tilde {f} (z) E _ {\beta , 0} (z) + \tilde {g} E _ {\beta , 1} (z) = z ^ {- 1} \widehat {g} (z).} \end{array}
$$

Solving the linear system yields the solution to the sideways problem 

$$
\tilde {f} (z) = \frac {E _ {\beta , 1} (z) \widehat {f} (z) - z ^ {- 1} E _ {\beta , 2} (z) \widehat {g} (z)}{E _ {\beta , 1} (z) ^ {2} - E _ {\beta , 0} (z) E _ {\beta , 2} (z)},
$$

and accordingly the solution $h ( t ) \equiv u ( 0 , t )$ is given by an inverse Laplace transform. The growth factors of the data $\widehat { f }$ and $\widehat g$ are $E _ { \beta , 1 } ( z ) / ( E _ { \beta , 1 } ( z ) ^ { 2 } - E _ { \beta , 0 } ( z ) E _ { \beta , 2 } ( z ) )$ and $z ^ { - 1 } E _ { \beta , 2 } ( z ) / ( E _ { \beta , 1 } ( z ) ^ { 2 } - E _ { \beta , 0 } ( z ) E _ { \beta , 2 } ( z ) )$ , respectively. The growth of these factors at large z argument determines the degree of ill-conditioning of the sideways problem. To this end, we appeal to the exponential asymptotic of the Mittag-Lef<sup>fl</sup>er function $E _ { \alpha , \beta } ( z )$ , cf lemma 2.1, and note that Bromwhich path lies in the sector |arg |z $\leqslant \pi / 2$ to deduce that for large |z |,there holds 

$$
\begin{array}{c} E _ {\beta , 1} (z) ^ {2} \sim \frac {1}{\beta^ {2}} \mathrm{e} ^ {2 z ^ {1 / \beta}} - \frac {2}{\beta \Gamma (1 - \beta) z} \mathrm{e} ^ {z ^ {1 / \beta}}, \\ E _ {\beta , 0} (z) E _ {\beta , 2} (z) \sim \frac {1}{\beta^ {2}} \mathrm{e} ^ {2 z ^ {1 / \beta}} - \frac {1}{\beta \Gamma (2 - \beta)} z ^ {1 / \beta - 1} \mathrm{e} ^ {z ^ {1 / \beta}}. \end{array}
$$

Hence, the numerator $E _ { \beta , 1 } ( z ) ^ { 2 } - E _ { \beta , 0 } ( z ) E _ { \beta , 2 } ( z )$ behaves like 

$$
E _ {\beta , 1} (z) ^ {2} - E _ {\beta , 0} (z) E _ {\beta , 2} (z) \sim \frac {1}{\beta \Gamma (2 - \beta)} z ^ {1 / \beta - 1} \mathrm{e} ^ {z ^ {1 / \beta}} \quad \text {as} | z | \to \infty .
$$

This together with the exponential asymptotic of $E _ { \beta , 1 } ( z )$ and $E _ { \beta , 2 } ( z )$ from lemma 2.1 indicates that the multipliers for $\widehat { f }$ and $\widehat g$ are growing at most at a very low-order polynomial rate, for large z. Hence, the high-frequency components of the data noise are not ampli<sup>fi</sup>ed much (at most polynomially instead of exponentially). The analysis indicates that the sideways problem with the lateral Cauchy data speci<sup>fi</sup>ed at the point $x = 1$ is nearly wellposed, as long as the fractional order $\beta$ is away from two, for which it recovers the classical ill-posed sideways problem for the heat equation. 

Next we illustrate the preceding discussions numerically. The behavior of the forward map $F$ from the Dirichlet boundary at $x = 0$ to the <sup>fl</sup>ux data at $x = 1$ is shown in <sup>fi</sup>gure 15. For a wide range of values of the fractional order $\beta ,$ the condition number of the forward map F is of order 100, which is fairly mild, in view of the size of the linear system, i.e., $1 0 0 \times 1 0 0$ When the fractional order $\beta$ increases towards two, the inverse problem recovers the classical sideways problem, and as expected, the condition number increases dramatically. However, the onset of the blowup depends on the terminal time $T \cdot$ the smaller is the time $T ,$ the smaller seems the onset value. The precise mechanism for this phenomenon is still unknown. For $\beta \leqslant 7 / 4$ , the singular value spectrum only spans a narrow interval, resulting in a very small condition number. Physically, like before, this can be explained as the ‘convective’ nature of the Djrbashian–Caputo fractional derivative: as the fractional order $\beta$ tends to unity, the information at $x = 0$ is transported to $x = 1$ , free from distortion, and thus the inverse problem is almost well-posed. In summary, depending on the location of the over-speci<sup>fi</sup>ed data, anomalous superdiffusion can either help or aggravate the conditioning of the sideways problem. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/22ee6c75f42142ee60d0a4c0a1ae71851a7b1ab4444c7c0dc79fbc48bca9579d.jpg)



(a) condition number


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/bad069c7-80e3-4e60-bc16-b058e192bb58/cdc49b9a049dfecd4744a2567eb4bab4fa5db6506e6101e841ae238ab94e6493.jpg)



(b) T = 1



Figure 15. Numerical results for the space fractional sideways problem, with the lateral Cauchy data at the point x = 1: (a) the condition number versus the fractional order β and (b) the singular value spectrum at T = 1.


Last, we would like to note that the study of space fractional inverse problems, either theoretical or numerical, is fairly scarce. This is partly attributed to the relatively poor understanding of forward problems for FDEs with a space fractional derivative: there are only a few mathematical studies on one-dimensional space fractional diffusion, and no mathematical study on multi-dimensional problems involving space fractional derivatives (of either Riemann–Liouville or Caputo type). Nonetheless, our preliminary numerical experiments show distinct new features for related inverse problems, which motivate their analytical studies. 

## 5. Concluding remarks

Anomalous diffusion processes arise in many disciplines, and the physics behind is very different from normal diffusion. The unusual physics greatly in<sup>fl</sup>uences the behavior of related forward problems. Further, it is well known that backward fractional diffusion is much less ill-posed than the classical backward diffusion, which has contributed to the belief that inverse problems for anomalous diffusion are always better behaved than that for the normal diffusion. In this work we have examined several exemplary inverse problems for anomalous diffusion processes in a numerical and semi-analytical manner. These include the sideways problem, backward problem, inverse source problem, inverse Sturm–Liouville problem and Cauchy problems. Our <sup>fi</sup>ndings indicate that anomalous diffusion can give rise to very unusual new features, but they only partially con<sup>fi</sup>rm the belief: depending on the data and unknown, it may in<sup>fl</sup>uence either positively or negatively the degree of ill-posedness of the inverse problem. 

The mathematical study of inverse problems in anomalous diffusion is still in its infancy. There are only a few rigorous theoretical results on the uniqueness, existence and stability, which mostly focus on the one-dimensional case, and there are many more open problems awaiting investigations. The development of stable and ef<sup>fi</sup>cient reconstruction procedures is an active ongoing research topic. However, due to the nonlocality of the forward model, the construction of ef<sup>fi</sup>cient schemes and their rigorous numerical analysis remain very challenging. This is especially true for space fractional FDEs, and there are almost no theoretical or rigorous numerical studies. 

## Acknowledgments

The authors are grateful for the anonymous referees for their constructive comments. The research of both authors is partly supported by NSF Grant DMS-1319052. 

## Appendix A. Numerical methods for special functions and FDEs

## A.1. Computation of the Mittag-Leffler and Wright functions

Like many special functions, the ef<sup>fi</sup>cient and accurate numerical computation of the Mittag-Lef<sup>fl</sup>er function $E _ { \alpha , \beta } ( z )$ is delicate [28, 29, 94]. An ef<sup>fi</sup>cient algorithm relies on partitioning the complex plane  into different regions, where different approximations, i.e., power series, integral representation and exponential asymptotic for small values of the argument, intermediate values and large values, respectively, are used for ef<sup>fi</sup>cient numerical computation; see [94] for the some partition and error estimates. The special case of the Mittag-Lef<sup>fl</sup>er function $E _ { \alpha , \beta } ( z )$ with a real argument $z \in \mathbb { R }$ , which plays a predominant role in time-fractional diffusion, can also be ef<sup>fi</sup>ciently computed with the Laplace transform and suitable quadrature rules [27]. 

The computation of the Wright function $W _ { \rho , \mu } ( z )$ is even more delicate. In theory, like before, it can be computed using power series for small values of the argument and a known asymptotic formula for large values, and for the intermediate case, values are obtained by using an integral representation [67]. The integral representation for the Wright function $W _ { \rho , \mu } ( z )$ for intermediate values in the case of interest for the fundamental solution in onedimension (where $\rho = - \alpha / 2 < 0 , 0 < \mu = 1 + \rho < 1$ and $z = - x , x > 0 )$ is given by 

$$
W _ {\rho , \mu} (- x) = \int_ {0} ^ {\infty} K (x, \rho , \mu , r) \mathrm{d} r
$$

where the kernel $K \left( x , \rho , \mu , r \right)$ is given by 

$$
K (x, \rho , \mu , r) = r ^ {- \mu} \mathrm{e} ^ {- r + x \cos (\pi \rho) r ^ {- \rho}} \sin (x \sin (\pi \rho) r ^ {- \rho} + \pi \mu).
$$

This is a singular kernel with a leading order $r ^ { - \mu } = r ^ { - 1 - \rho } ;$ , with successive singular kernels of the form $r ^ { - 1 - 2 \rho } , \ r ^ { - 1 - 3 \rho }$ etc, upon expanding the terms. Hence, a direct treatment via numerical quadrature is inef<sup>fi</sup>cient. A more ef<sup>fi</sup>cient approach is to use the change of variable $s = r ^ { - \rho } , \mathrm { i . e . , } r = s ^ { - 1 / \rho }$ , and the transformed kernel is 

$$
\widetilde {K} (x, \rho , \mu , s) = (- \rho) ^ {- 1} s ^ {(\neg \rho) ^ {- 1} (\neg \mu + 1) - 1} \mathrm{e} ^ {- s ^ {- \frac {1}{\rho}} + x \cos (\pi \rho) s} \sin (x \sin (\pi \rho) s + \pi \mu).
$$

The fundamental solution of the one-dimensional time-fractional diffusion equation is expressed in terms of a Wright function $W _ { \rho , \mu } ( - x )$ with the choice $\rho = - \alpha / 2$ and $\mu = 1 + \rho _ { ; }$ cf (2.2). In this case the resulting kernel K<sup>˜</sup> simpli<sup>fi</sup>es to 

$$
\widetilde {K} (x, \rho , 1 + \rho , s) = (- \rho) ^ {- 1} \mathrm{e} ^ {- s ^ {- \frac {1}{\rho}} + x \cos (\pi \rho) s} \sin (x \sin (\pi \rho) s + (1 + \rho) \pi).
$$

This kernel is free from the grave singularity, and thus the quadrature method is quite effective. In general, the integral can be computed ef<sup>fi</sup>ciently via the Gauss–Jacobi quadrature, with the weight function $s ^ { ( - \rho ) ^ { - 1 } ( - \mu + 1 ) - 1 }$ . We note that an algorithm for the Wright function $W _ { \rho , \mu } ( z )$ over the whole complex plane  with rigorous error analysis is still missing. The endeavor in this direction would almost certainly involve dividing the complex domain  into different regions, and using different approximations on each region separately. 

## A.2. Time fractional diffusion

We describe a <sup>fi</sup>nite difference method for the initial boundary value problem for the onedimensional time-fractional diffusion equation 

$$
\partial_ {t} ^ {\alpha} u - u _ {x x} + q u = f (x, t) \quad (x, t) \in \Omega \times (0, T),
$$

with the initial condition $u ( x , 0 ) = \nu$ and boundary condition 

$$
u (0, t) = g (t) \quad \mathrm{and} \quad u (1, t) = h (t), \quad t > 0.
$$

There are many ef<sup>fi</sup>cient numerical schemes for discretizing the problem. The discretization in space can be achieved by the standard central difference scheme, Galerkin <sup>fi</sup>nite element method [47] or spectral method, and the discretization in time can be achieved with the L1 approximation [63, 96] and convolution quadrature [48]. We shall adopt the L1 approximation in time and the central difference in space. Speci<sup>fi</sup>cally, we divide the interval [0, ]T into uniform subintervals, with nodes $t _ { k } = k \tau , k = 0 . . . , K ,$ and a time step size $\tau = T / K$ . Similarly, we partition the spatial domain Ω into uniform subintervals, with grid points $x _ { i } = \mathrm { i } h , \ i = \ 0 , . . . , N ,$ , and mesh size $h = 1 / N$ . Then the L1-approximation of the Djrbashian–Caputo fractional derivative $\partial _ { t } ^ { \alpha } u ( x , t _ { k } )$ developed in [63, 96] is given by: 

$$
\partial_ {t} ^ {\alpha} u (x, t _ {k}) \approx \tau^ {- \alpha} \Bigg [ b _ {0} u (x, t _ {k}) - b _ {k - 1} u (x, t _ {0}) + \sum_ {j = 1} ^ {k - 1} \Bigl (b _ {j} - b _ {j - 1} \Bigr) u \Bigl (x, t _ {k - j} \Bigr) \Bigg ],\tag{A.1}
$$

where the weights $b _ { j }$ are given by 

$$
b _ {j} = \Big ((j + 1) ^ {1 - \alpha} - j ^ {1 - \alpha} \Big) / \Gamma (2 - \alpha), j = 0, 1, \dots , K - 1.
$$

If the solution $u ( x , t )$ is $C ^ { 2 }$ continuous in time, the local truncation error of the L1 approximation is bounded by $c \tau ^ { 2 - \alpha }$ for some c depending only on u [63, equation (3.3)]. In general, one can show that the scheme is only <sup>fi</sup>rst-order accurate. Next with the central difference scheme in space and the notation $u _ { i } ^ { k } \approx u ( x _ { i } , t _ { k } )$ , we arrive at the following fully discrete scheme 

$$
\begin{array}{l} \left[ b _ {0} u _ {i} ^ {k} - b _ {k - 1} u _ {i} ^ {0} + \sum_ {j = 1} ^ {k - 1} \bigl (b _ {j} - b _ {j - 1} \bigr) u _ {i} ^ {k - j} \right] \\ \qquad + \frac {u _ {i - 1} ^ {k} - 2 u _ {i} ^ {k} + u _ {i + 1} ^ {k}}{h ^ {2}} + q _ {i} u _ {i} ^ {k} = f _ {i} ^ {k}, i = 1, \dots , N - 1, \end{array}
$$

with $q _ { i } = q ( x _ { i } )$ and $f _ { i } ^ { k } = f ( x _ { i } , t _ { k } )$ . We note that at each time step, one needs to solve a tridiagonal linear system. However, the right-hand side at the current step involves all previous steps, which can be quite expensive for a small step size, and this will most likely be required due to the <sup>fi</sup>rst order in time convergence. This history piece represents one of the main computational challenges for time fractional differential equations. There are high-order schemes, e.g., convolution quadrature generated by the second-order backward difference formula [48]. Further, we note that the <sup>fi</sup>nite difference scheme in space can be replaced with the Galerkin <sup>fi</sup>nite element method, which is especially suitable for high dimensional problems on a general domain and elliptic operator involving variable coef<sup>fi</sup>cients [47]. 

## A.3. Space fractional diffusion

Now we describe a fully discrete scheme based on the backward Euler method in time and a Galerkin <sup>fi</sup>nite element method in space for the space fractional diffusion problem on the unit interval $\varOmega = ( 0 , 1 )$ ) 

$$
u _ {t} - \mathbf {\Lambda} _ {0} ^ {C} D _ {x} ^ {\beta} u + q u = f \quad \mathrm{in} \Omega \times (0, T ],
$$

with the initial condition $u = \nu$ and the Dirichlet boundary condition 

$$
u (0, t) = g (t) \quad \text { and } \quad u (1, t) = h (t), \quad t > 0.
$$

The Galerkin <sup>fi</sup>nite element method relies on the variational formulation for the fractional elliptic problem 

$$
- _ {0} ^ {C} D _ {x} ^ {\beta} u + q u = f \quad \mathrm{in} \Omega ,
$$

with a homogeneous Dirichlet boundary condition $u ( 0 ) = u ( 1 ) = 0$ , recently developed in [45]. The variational formulation of the problem is given by: <sup>fi</sup>nd u $\in { \cal { U } } \equiv H _ { 0 } ^ { \beta / 2 } ( \varOmega )$ such that 

$$
- \Big (_ {0} ^ {R} D _ {x} ^ {\beta / 2} u,   _ {x} ^ {R} D _ {1} ^ {\beta / 2} v \Big) + (q u,   v) = (f,   v) \quad \forall v \in V,
$$

where ${ } _ { 0 } ^ { R } D _ { x } ^ { \gamma } \nu$ and ${ } _ { x } ^ { R } D _ { 1 } ^ { \gamma } \nu$ are the left-sided and right-sided Riemann–Liouville derivative of order $\gamma \in ( 0 , 1 )$ de<sup>fi</sup>ned by (with $c _ { \gamma } = 1 / { \varGamma ( 1 - \gamma ) } ,$ 0 

$$
{ } _ { 0 } ^ { R } D _ { x } ^ { \gamma } v ( x ) = c _ { \gamma } \frac { \mathrm{d} } { \mathrm{d} x } \int _ { 0 } ^ { x } ( x - s ) ^ { - \gamma } v ( s ) \mathrm{d} s \quad \text {and} \quad { } _ { x } ^ { R } D _ { 1 } ^ { \gamma } v ( x ) = - c _ { \gamma } \frac { \mathrm{d} } { \mathrm{d} x } \int _ { x } ^ { 1 } ( x - s ) ^ { - \gamma } v ( s ) \mathrm{d} s ,
$$

respectively, and the test space V is given by 

$$
V = \Big \{v \in H ^ {\beta / 2} (\Omega) \colon v (1) = 0, \left(x ^ {1 - \beta}, v\right) = 0 \Big \}.
$$

For the <sup>fi</sup>nite element discretization we <sup>fi</sup>rst divide the unit interval $\varOmega$ into a uniform mesh, with the grid points $x _ { i } = \mathrm { i } h , i = 0 , . . . , N$ and mesh size $h = 1 / N$ . Then for $U _ { h } \subset U$ we take the continuous piecewise linear <sup>fi</sup>nite element space, and for $V _ { h } \subset V$ we construct it from $U _ { h }$ Speci<sup>fi</sup>cally, with the <sup>fi</sup>nite element basis $\phi _ { i } ( x )$ $i = 1 , . . . , N - 1$ we take $\tilde { \phi } _ { i } ( x ) = \phi _ { i } ( x ) - \gamma _ { i } ( 1 - x ) \in V _ { h }$ , where the constant γ is determined by the integral condition $( x ^ { 1 - \beta } , \tilde { \phi } _ { i } ) = 0$ , i.e., $\gamma _ { i } = h ^ { 2 - \beta } ( ( \mathrm { i } - 1 ) ^ { 3 - \beta } + ( \mathrm { i } + 1 ) ^ { 3 - \beta } - 2 \mathrm { i } ^ { 3 - \beta } )$ . The computation of the leading term in the stiffness matrix and mass matrix can be carried out analytically, and the part involving the potential q can be computed ef<sup>fi</sup>ciently using quadrature rules; see [46] for details. 

Now for the time-dependent problem, like before, we divide the time interval $[ 0 , T ]$ into uniform subintervals, with $t _ { k } = k \tau , k = 0 , . . . , K .$ , and the time step size $\tau = T / K$ . Then with the backward Euler method in time, and the <sup>fi</sup>nite element method in space, the approximate solution $u _ { h } ^ { k }$ at time $t _ { k }$ can be split into $u _ { h } ^ { \ k } = \tilde { u } _ { h } ^ { k } + s ^ { k }$ with the particular solution $s ^ { k } = g ( t _ { k } ) + ( h ( t _ { k } ) - g ( t _ { k } ) ) x$ with the homogeneous solution $\tilde { u } _ { h } ^ { k } \in U _ { h }$ satisfying 

$$
\begin{array}{r l} & {\tau^ {- 1} \Big (\tilde {u} _ {h} ^ {k}, v _ {h} \Big) - \Big (_ {0} ^ {R} D _ {x} ^ {\beta / 2} \tilde {u} _ {h} ^ {k}, _ {x} ^ {R} D _ {1} ^ {\beta / 2} v _ {h} \Big) + \Big (q \tilde {u} _ {h} ^ {k}, v _ {h} \Big)} \\ & {\qquad = \Big (f ^ {k}, v _ {h} \Big) + \tau^ {- 1} \Big (u _ {h} ^ {k - 1} - s ^ {k}, v _ {h} \Big) - \Big (q s ^ {k}, v _ {h} \Big) \quad \forall v _ {h} \in V _ {h}.} \end{array}
$$

We note the resulting linear system is of lower Hessenberg form, due to the nonlocality of the fractional derivative operator. However, the coef<sup>fi</sup>cient matrix does not change during the time stepping procedure, and thus an LU factorization might be applied to speedup the computation. 

## References



[1] Agarwal R P 1953 A propos d’une note de M Pierre Humbert C. R. Acad. Sci., Paris 236 2031–2 





[2] Aleroev T S, Kirane M and Malik S A 2013 Determination of a source term for a time fractional diffusion equation with an integral type over-determining condition Electron. J. Differ. Equ. 16 270 





[3] Balakrishnan A V 1960 Fractional powers of closed operators and the semigroups generated by them Pac. J. Math. 10 419–37 





[4] Beck J V, Blackwell B and Claire C R St Jr 1985 Inverse Heat Conduction: Ill-Posed Problems (New York: Wiley) 





[5] Berkowitz B, Cortis A, Dentz M and Scher H 2006 Modeling non-Fickian transport in geological formations as a continuous time random walk Rev. Geophys. 44 49 





[6] Blumenthal R M and Getoor R K 1959 The asymptotic distribution of the eigenvalues for a class of Markov operators Pac. J. Math. 9 399–408 





[7] Cannon J R 1968 Determination of an unknown heat source from overspeci<sup>fi</sup>ed boundary data SIAM J. Numer. Anal. 5 275–86 





[8] Cannon J R 1984 The One-Dimensional Heat Equation (Reading, MA: Addison-Wesley) 





[9] Cannon J R and DuChateau P 1998 Structural identi<sup>fi</sup>cation of an unknown source term in a heat equation Inverse Problems 14 535–51 





[10] Caputo M 1967 Linear models of dissipation whose Q is almost frequency independent—II Geophys. J. Int. 13 529–39 





[11] Carasso A 1982 Determining surface temperatures from interior observations SIAM J. Appl. Math. 42 558–74 





[12] Chadan K, Colton D, Päivärinta L and Rundell W 1997 An Introduction to Inverse Scattering and Inverse Spectral Problems (Philadelphia: SIAM) 





[13] Cheng J, Lin C-L and Nakamura G 2013 Unique continuation property for the anomalous diffusion and its application J. Differ. Equ. 254 3715–28 





[14] Cheng J, Nakagawa J, Yamamoto M and Yamazaki T 2009 Uniqueness in an inverse problem for a one-dimensional fractional diffusion equation Inverse Problems 25 115002 





[15] Choulli M and Yamamoto M 1996 Generic well-posedness of an inverse parabolic problem—the Hölder-space approach Inverse Problems 12 195–205 





[16] Choulli M and Yamamoto M 1997 An inverse parabolic problem with non-zero initial condition Inverse Problems 13 19–27 





[17] Clarke D D, Meerschaert M M and Wheatcraft S W 2005 Fractal travel time estimates for dispersive contaminants Groundwater 3 401–7 





[18] Cussler E L 1997 Diffusion: Mass Transfer in Fluid Systems 2nd edn (New York: Cambridge University Press) 





[19] Djrbashian M 1993 Harmonic Analysis and Boundary Value Problems in the Complex Domain (Basel: Birkhäuser) 





[20] Djrbashian M M 1989 Differential operators of fractional order and boundary value problems in the complex domain The Gohberg Anniversary Collection, Operator Theory: Advances and Applications vol 41 (Berlin: Springer-Verlag) pp 153–72 





[21] Dzharbashyan M M 1966 Integral Transformations and Representation of Functions in a Complex Domain (Moscow: Nauka) (in Russian) 





[22] Džrbašjan M M 1970 A boundary value problem for a Sturm–Liouville type differential operator of fractional order Izv. Akad. Nauk Arm. SSR Ser. Mat. 5 71–96 





[23] Einstein A 1905 Über die von der molekularkinetischen Theorie der Wärme geforderte Bewegung von in ruhenden Flüssigkeiten suspendierten Teilchen. Ann. Phys. 322 549–60 





[24] Eldén L 1995 Numerical solution of the sideways heat equation by difference approximation in time Inverse Problems 11 913–23 





[25] Eldén L, Berntsson F and Regińska T 2000 Wavelet and fourier methods for solving the sideways heat equation SIAM J. Sci. Comput. 21 2187–205 





[26] Engl H W, Hanke M and Neubauer A 1996 Regularization of Inverse Problems (Dordrecht: Kluwer) 





[27] Garrappa R and Popolizio M 2013 Evaluation of generalized Mittag-Lef<sup>fl</sup>er functions on the real line Adv. Comput. Math. 39 205–25 





[28] Goren<sup>fl</sup>o R, Loutchko J and Luchko Y 2002 Computation of the Mittag-Lef<sup>fl</sup>er function $E _ { \alpha , \beta } ( z )$ and its derivative Fract. Calc. Appl. Anal. 5 491–518 





[29] Goren<sup>fl</sup>o R, Loutchko J and Luchko Y 2002 Correction: computation of the Mittag-Lef<sup>fl</sup>er function $E _ { \alpha , \beta } ( z )$ and its derivative Fract. Calc. Appl. Anal. 5 491–518 





Goren<sup>fl</sup>o R, Loutchko J and Luchko Y 2003 Fract. Calc. Appl. Anal. 6 111–2 





[30] Goren<sup>fl</sup>o R, Luchko Y and Mainardi F 1999 Analytical properties and applications of the Wright function Fract. Calc. Appl. Anal. 2 383–414 





[31] Hadamard J 1923 Lectures on Cauchyʼs Problem in Linear Partial Differential Equations (New Haven, CT: Yale University Press) 





[32] Hansen P C 1998 Rank-Deficient and Discrete Ill-posed Problems (Philadelphia: SIAM) 





[33] Hào D N and Reinhardt H-J 1997 On a sideways parabolic equation Inverse Problems 13 297–309 





[34] Hatano Y and Hatano N 1998 Dispersive transport of ions in column experiments: an explanation of long-tailed pro<sup>fi</sup>les Water Resour. Res. 34 1027–33 





[35] Hatano Y, Nakagawa J, Wang S and Yamamoto M 2013 Determination of order in fractional diffusion equation J. Math. Ind. A 5 51–57 





[36] Humbert P 1953 Quelques résultats relatifs à la fonction de Mittag-Lef<sup>fl</sup>er C. R. Acad. Sci., Paris 236 1467–8 





[37] Imanuvilov O Y and Yamamoto M 1998 Lipschitz stability in inverse parabolic problems by the Carleman estimate Inverse Problems 14 1229–45 





[38] Isakov V 1991 Inverse parabolic problems with the <sup>fi</sup>nal overdetermination Commun. Pure Appl. Math. 44 185–209 





[39] Isakov V 1999 Some inverse problems for the diffusion equation Inverse Problems 15 3–10 





[40] Isakov V 2006 Inverse Problems for Partial Differential Equations 2nd edn (New York: Springer) 





[41] Ito K and Jin B 2014 Inverse Problems: Tikhonov Theory and Algorithms (Singapore: World Scienti<sup>fi</sup>c) 





[42] Ito K, Jin B and Takeuchi T 2014 Legendre tau method for fractional elliptic problems with a Caputo derivative (unpublished) 





[43] Ito K and Kappel F 2002 Evolutions Equations and Approximations (Singapore: World Scienti<sup>fi</sup>c) 





[44] Jin B, Lazarov R, Liu Y and Zhou Z 2015 The Galerkin <sup>fi</sup>nite element method for a multi-term time-fractional diffusion equation J. Comput. Phys. 281 825–43 





[45] Jin B, Lazarov R, Pasciak J and Rundell W 2013 Variational formulation of problems involving fractional order differential operators Math. Comput. at press (arXiv:1307.4795) 





[46] Jin B, Lazarov R, Pasciak J and Rundell W 2014 A <sup>fi</sup>nite element method for the fractional Sturm–Liouville problem arXiv:1307.5114 





[47] Jin B, Lazarov R and Zhou Z 2013 Error estimates for a semidiscrete <sup>fi</sup>nite element method for fractional order parabolic equations SIAM J. Numer. Anal. 51 445–66 





[48] Jin B, Lazarov R and Zhou Z 2014 On two schemes for fractional diffusion and diffusion-wave equations arXiv:1404.3800 





[49] Jin B and Maass P 2012 Sparsity regularization for parameter identi<sup>fi</sup>cation problems Inverse Problems 28 





[50] Jin B and Rundell W 2012 An inverse problem for a one-dimensional time-fractional diffusion problem Inverse Problems 28 075010 





[51] Jin B and Rundell W 2012 An inverse Sturm–Liouville problem with a fractional derivative J. Comput. Phys. 231 4954–66 





[52] Jones B F Jr 1962 The determination of a coef<sup>fi</sup>cient in a parabolic differential equation: I. Existence and uniqueness J. Math. Mech. 11 907–18 





[53] Kilbas A, Srivastava H and Trujillo J 2006 Theory and Applications of Fractional Differential Equations (Amsterdam: Elsevier) 





[54] Kochubeı˘ A N 1990 Diffusion of fractional order Differentsial’ nye Uravneniya 26 660–70 733–734 





[55] Kwaśnicki M 2012 Eigenvalues of the fractional Laplace operator in the interval J. Funct. Anal. 262 2379–402 





[56] Lamm P K 2000 A survey of regularization methods for kirst-kind Volterra equations Surveys on Solution Methods for Inverse Problems ed D Colton, H W Engl, A K Louis, J R McLaughlin and W Rundell (Berlin: Springer) pp 53–82 





[57] Lattès R and Lions J-L 1969 The Method of Quasi-Reversibility. Applications to Partial Differential Equations (New York: Elsevier) 





[58] Li G, Zhang D, Jia X and Yamamoto M 2013 Simultaneous inversion for the space-dependent diffusion coef<sup>fi</sup>cient and the fractional order in the time-fractional diffusion equation Inverse Problems 29 065014 





[59] Li Z, Imanuvilov O Y and Yamamoto M 2014 Uniqueness in inverse boundary value problems for fractional diffusion equations arXiv:1404.7024 





[60] Li Z and Yamamoto M 2013 Initial-boundary value problems for linear diffusion equation with multiple time-fractional derivatives arXiv:1306.2778 





[61] Li Z and Yamamoto M 2014 Uniqueness for inverse problems of determining orders of multiterm time-fractional derivatives of diffusion equation Appl. Anal. at press doi:10.1080 00036811.2014.926335 





[62] Lin C-L and Nakamura G 2013 Carleman estimate and its application for anomalous slow diffusion equation arXiv:1312.7639 





[63] Lin Y and Xu C 2007 Finite difference/spectral approximations for the time-fractional diffusion equation J. Comput. Phys. 225 1533–52 





[64] Liu J 1996 A stability analysis on Beckʼs procedure for inverse heat conduction problems J. Comput. Phys. 123 65–73 





[65] Liu J J and Yamamoto M 2010 A backward problem for the time-fractional diffusion equation Appl. Anal. 89 1769–88 





[66] Luchko Y 2000 Asymptotics of zeros of the Wright function Z. Anal. Anwendungen 19 583–95 





[67] Luchko Y 2008 Algorithms for evaluation of the Wright function for the real arguments’ values Fract. Calc. Appl. Anal. 11 57–75 





[68] Luchko Y, Rundell W, Yamamoto M and Zuo L 2013 Uniqueness and reconstruction of an unknown semilinear term in a time-fractional reaction-diffusion equation Inverse Problems 29 065019 





[69] Mainardi F 1996 The fundamental solutions for the fractional diffusion-wave equation Appl. Math. Lett. 9 23–28 





[70] Mainardi F 2010 Fractional Calculus and Waves in Linear Viscoelasticity: An Introduction to Mathematical Models (Singapore: World Scienti<sup>fi</sup>c) 





[71] Malamud M M 1994 Similarity of Volterra operators and related problems in the theory of differential equations of fractional orders Tr. Mosk. Mat. Obshch. 55 365 





[72] Metzler R and Klafter J 2000 The random walkʼs guide to anomalous diffusion: a fractional dynamics approach Phys. Rep. 339 1–77 





[73] Miller L and Yamamoto M 2013 Coef<sup>fi</sup>cient inverse problem for a fractional diffusion equation Inverse Problems 29 075013 





[74] Mittag-Lef<sup>fl</sup>er G M 1903 Sur la nouvelle function E C. R. Acad. Sci., Paris 137 554–8 





[75] Montroll E W and Weiss G H 1965 Random walks on lattices: II. J. Math. Phys. 6 167–81 





[76] Murio D A 2007 Stable numerical solution of a fractional-diffusion inverse heat conduction problem Comput. Math. Appl. 53 1492–501 





[77] Murio D A 2008 Time fractional IHCP with Caputo fractional derivatives Comput. Math. Appl. 56 2371–81 





[78] Nahušev A M 1977 The Sturm–Liouville problem for a second order ordinary differential equation with fractional derivatives in the lower terms Dokl. Akad. Nauk SSSR 234 308–11 





[79] Nakagawa J, Sakamoto K and Yamamoto M 2010 Overview to mathematical analysis for fractional diffusion equations—new mathematical aspects motivated by industrial collaboration J. Math. Ind. 2A 99–108 





[80] Paris R B 2002 Exponential asymptotics of the Mittag-Lef<sup>fl</sup>er function Proc. R. Soc. A 458 3041–52 





[81] Pazy A 1992 Semigroups of Linear Operators and Applications to Partial Differential Equations (Berlin: Springer) 





[82] Pollard H 1948 The completely monotonic character of the Mittag-Lef<sup>fl</sup>er function $E _ { a } ( - x )$ Bull. Am. Math. Soc. 54 1115–6 





[83] Popov A Y and Sedletskiı˘ A M 2011 Distribution of roots of Mittag-Lef<sup>fl</sup>er functions Sovrem. Mat. Fundam. Napravl. 40 3–171 





[84] Prilepko A I, Orlovsky D G and Vasin I A 2000 Methods for Solving Inverse Problems in Mathematical Physics (New York: Dekker) 





[85] Qian Z 2010 Optimal modi<sup>fi</sup>ed method for a fractional-diffusion inverse heat conduction problem Inverse Probl. Sci. Eng. 18 521–33 





[86] Rundell W and Sacks P E 1992 Reconstruction techniques for classical inverse Sturm–Liouville problems Math. Comput. 58 161–83 





[87] Rundell W, Xu X and Zuo L 2013 The determination of an unknown boundary condition in a fractional diffusion equation Appl. Anal. 92 1511–26 





[88] Sakamoto K and Yamamoto M 2011 Initial value/boundary value problems for fractional diffusion-wave equations and applications to some inverse problems J. Math. Anal. Appl. 382 426–47 





[89] Sakamoto K and Yamamoto M 2011 Inverse source problem with a <sup>fi</sup>nal overdetermination for afractional diffusion equation Math. Control Relat. Fields 1 509–18 





[90] Schneider W R 1996 Completely monotone generalized Mittag-Lef<sup>fl</sup>er functions Exposition. Math. 14 3–16 





[91] Schneider W R and Wyss W 1989 Fractional diffusion and wave equations J. Math. Phys. 30 134–44 





[92] Schuster T, Kaltenbacher B, Hofmann B and Kazimierski K S 2012 Regularization Methods in Banach Spaces (Berlin: Walter de Gruyter) 





[93] Sedletskiı˘ A M 1994 Asymptotic formulas for zeros of functions of Mittag-Lef<sup>fl</sup>er type Anal. Math. 20 117–32 





[94] Seybold H and Hilfer R 2008 Numerical algorithm for calculating the generalized Mittag-Lef<sup>fl</sup>er function SIAM J. Numer. Anal. 47 69–88 





[95] Sokolov I M, Klafter J and Blumen A 2002 Fractional kinetics Phys. Today 55 48–54 





[96] Sun Z-Z and Wu X 2006 A fully discrete difference scheme for a diffusion-wave system Appl. Numer. Math. 56 193–209 





[97] Wang H and Wu B 2014 On the well-posedness of determination of two coef<sup>fi</sup>cients in a fractional integrodifferential equation Chin. Ann. Math. Ser. B 35 447–68 





[98] Wang J-G, Zhou Y-B and Wei T 2013 Two regularization methods to identify a space-dependen source for the time-fractional diffusion equation Appl. Numer. Math. 68 39–57 





[99] Wang L and Liu J 2013 Total variation regularization for a backward time-fractional diffusion problem Inverse Problems 29 115013 





[100] Wang W, Yamamoto M and Han B 2013 Numerical method in reproducing kernel space for an inverse source problem for the fractional diffusion equation Inverse Problems 29 095009 





[101] Wei T and Wang J 2014 A modi<sup>fi</sup>ed quasi-boundary value method for an inverse source problem of the time-fractional diffusion equation Appl. Numer. Math. 78 95–111 





[102] Wei T and Wang J-G 2014 A modi<sup>fi</sup>ed quasi-boundary value method for the backward timefractional diffusion problem ESAIM Math. Modelling Numer. Anal. 48 603–21 





[103] Weideman J A C and Trefethen L N 2007 Parabolic and hyperbolic contours for computing the Bromwich integral Math. Comput. 76 1341–56 





[104] Wiman A 1905 Über die Nullstellen der Funktionen E ( )x<sup>a</sup> Acta Math. 29 217–34 





[105] Wong R and Zhao Y-Q 2002 Exponential asymptotics of the Mittag-Lef<sup>fl</sup>er function Constr. Approx. 18 355–85 





[106] Wright E M 1933 On the coef<sup>fi</sup>cients of power series having exponential singularities J. London Math. Soc. 8 71–79 





[107] Wright E M 1940 The generalized Bessel function of order greater than one Q. J. Math. 11 36–48 





[108] Xu X, Cheng J and Yamamoto M 2011 Carleman estimate for a fractional diffusion equation with half order and application Appl. Anal. 90 1355–71 





[109] Yamamoto M and Zhang Y 2012 Conditional stability in determining a zeroth-order coef<sup>fi</sup>cient in a half-order fractional diffusion equation by a Carleman estimate Inverse Problems 28 105010 





[110] Zacher R 2013 A weak Harnack inequality for fractional evolution equations with discontinuous coef<sup>fi</sup>cients Ann. Sc. Norm. Super. Pisa Cl. Sci.(5) 12 903–40 





[111] Zhang Y and Xu X 2011 Inverse source problem for a fractional diffusion equation Inverse Problems 27 035010 





[112] Zhang Z Q and Wei T 2013 Identifying an unknown source in time-fractional diffusion equation by a truncation method Appl. Math. Comput. 219 5972–83 





[113] Zheng G H and Wei T 2011 A new regularization method for the time fractional inverse advection-dispersion problem SIAM J. Numer. Anal. 49 1972–90 





[114] Zheng G H and Wei T 2012 A new regularization method for a Cauchy problem of the time fractional diffusion equation Adv. Comput. Math. 36 377–98 

