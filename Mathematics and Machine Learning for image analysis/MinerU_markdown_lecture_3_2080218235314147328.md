# ERASMUS+ INTERNATIONAL PHD SUMMER SCHOOL 2025 Mathematics and Machine Learning for image analysis Lecture 3 - Learning Optimal Discretizations

Thomas Pock 

Institute of Visual Computing 

Graz University of Technology 

University of Bologna, June 3-6 2025 

<sup>▶</sup> Edges are among the most important features in images 

<sup>▶</sup> Image understanding relies on abstract discontinuity information 

▶ Most successful image descriptors are based on intensity gradients 

<sup>▶</sup> First layers in deep convolutional networks represent edge detectors 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/9d32f9a698114673ed74c8dec294c88d7e61490a3f2bae98c855107aad062472.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/7a11b37a05ef543190006ba8da9fac8eea75b049a862dfafbc55c8228910ede1.jpg)



(b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/4c1d5f3ae7722c10e095973c92e219e909d04308a4100bc9becdf06032d2e25a.jpg)



(c)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/58068814a1d1a4f0c6db82cfa863377c9dc6fdc51677567e0c87cb878e0e98d6.jpg)



(d)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/2358774743c01ae521d01943041193644aadc1a152dc129d3c5b4569fd4575f0.jpg)



(e)


## Edge statistics of natural images

▶ Randomly extracted 15M image patches of size $2 \times 2$ from a natural image data set. 

▶ Compute finite diferences in horizontal and vertical direction. 

<sup>▶</sup> Yields a heavy tailed distribution <sup>⇝</sup> most gradients are zero <sup>⇝</sup> sparse gradients. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/43eb59d0be61cd6b1cfbf849d427b448f7ea50898dac1007657d98a5ef6cb510.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/f0e905e36528e61e23056d703418f3bb24bd48634b4a70302cf9283cac0a9d74.jpg)


## The total variation

<sup>▶</sup> The log-statistics of $D u$ looks like an upside-down ice-cream cone. 

<sup>▶</sup> A simple fit to the negative log-statistics is given by the $\ell _ { 2 }$ norm, leading to the total variation: 

$$
\operatorname{TV} (u) = \| D u \| _ {2, 1} = \sum_ {i, j} \sqrt {\left(u _ {i + 1 , j} - u _ {i , j}\right) ^ {2} + \left(u _ {i , j + 1} - u _ {i , j}\right) ^ {2}}.
$$

<sup>▶</sup> Has been introduced in [Rudin, Osher, Fatemi ’92], [Chambolle, Lions $^ { , } 9 7 ]$ <sup>▶</sup> The discrete ROF model [Rudin, Osher, Fatemi ’92] is defined as the following minimization problem 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/ddcc30994172cf337edceb550f449fbb30aff85a18b4566f33ab8d27e0be0836.jpg)


$$
\min _ {u} \lambda \left\| D u \right\| _ {2, 1} + \frac {1}{2} \left\| u - g \right\| ^ {2}, \lambda > 0
$$

<sup>▶</sup> Defines ”the” prototypical variational model in mathematical image processing. 

<sup>▶</sup> Gives a good tradeof between simplicity of the model and denoising quality. 

▶ Allows for discontinuities (edges) in the image. 

<sup>▶</sup> It is a convex lower-semicontinuous function. 

<sup>▶</sup> It also has a nice geometric interpretation <sup>⇝</sup> minimal surfaces. 

## Advanced applications

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/1c19c5be3508db8c94f63840115624226885dcca8bfb561b0ad8ce77934275fa.jpg)



(a) Denoising


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/593c05765d719e31b88325e6c3bfcd7c0a87da3ed3aa37150ffcab5cc66a6f1c.jpg)



(b) Deblurring


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/3a4d10d209d77fe55e9d36865c9499c0d428acdffb0caf21bba7297d24adf558.jpg)



(c) MRI


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/ed837c3782dd89f8b91bb7274d025855df20786c8cc328d880f47504f0e33455.jpg)



(d) Motion


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/87e295985c4a92918e649960dc2f872ff95664c7b411c3521e73bc4d65a60690.jpg)



(e) Stereo


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/8304e754ef313ba0397f94e8128ecc297468e1e6d526ba3f0a41183e6a6ee1a8.jpg)



(f) Segmentation


<sup>▶</sup> For most practical problems, the standard discrete total variation gives suficiently good results. 

<sup>▶</sup> However, on free discontinuity problems such as image inpainting, the standard discretization yields strong artifacts. 

Inpainting of straight discontinuitie 

## Image inpainting

<sup>▶</sup> For most practical problems, the standard discrete total variation gives suficiently good results. 

<sup>▶</sup> However, on free discontinuity problems such as image inpainting, the standard discretization yields strong artifacts. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/bf4a2d8c42b7c78a4a179500ac7156837076435f7da7427a75acbc55ba6e0dbb.jpg)



Inpainting of straight discontinuities


Advanced free discontinuities problems Convexification of the Mumford-Shah functional [P., Cremers, Bischof, Chambolle ’09]: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/bfa5f4186a0c4636133d85cb38b1ee5226d29b3b7e44d6614e7fae33810fb104.jpg)



Convexification of Euler’s elastica [Chambolle, P. ’19]


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/9419c6a700790cade09aa64d24891b18fc9082114cc5b400189c11c677afc69e.jpg)



(data from J. Weickert)


Here, the discretization can make a diference between “working” and “not working”. 

Finding a good general discretization of the total variation is far from being trivial and hence many approaches have been proposed: 

<sup>▶</sup> Non-standard finite diferences for anisotropic difusion [Weickert, Welk, Wichert ’13] 

<sup>▶</sup> Graph-based / MRFs / crystalline energies [Boykov, Kolmogorov ’03], [Chambolle ’05] 

<sup>▶</sup> Upwind discretization [Chambolle, Levine, Lucier ’11] 

<sup>▶</sup> Shannon TV [Abergel, Moisan ’17] 

<sup>▶</sup> Conforming P1 finite elements [Bartels ’12] 

<sup>▶</sup> Non-conforming P1 (Crouzeix-Raviart) finite elements [Chambolle, P. 18] 

▶ Duality based discretization using H(div)-conforming Raviart-Thomas (RT0) vector fields [Herrmann, Herzog, Schmidt, Vidal, Wachsmuth ’18], [Caillaud, Chambolle ’20] 

<sup>▶</sup> Approximate Raviart-Thomas [Hinterm¨uller, Rautenberg, Hahn ’14], [Condat ’17] 

<sup>▶</sup> We introduce the finite diferences operator $D u = ( D ^ { 1 } u , D ^ { 2 } u )$ with 

$$
\left\{ \begin{array}{l l} (D ^ {1} u) _ {i + \frac {1}{2}, j} = u _ {i + 1, j} - u _ {i, j} & i = 1, \ldots , M - 1, j = 1, \ldots , N, \\ (D ^ {2} u) _ {i, j + \frac {1}{2}} = u _ {i, j + 1} - u _ {i, j} & i = 1, \ldots , M, j = 1, \ldots , N - 1. \end{array} \right.
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/659f8a7425dec1a9767618a64df78b0f45d5c6c4b9209d68e17efb5430689305.jpg)


<sup>▶</sup> The total variation is defined via duality as 

$$
T V (u) := \sup \left\{\langle \boldsymbol {p}, D u \rangle_ {Y}: \| \boldsymbol {F} \boldsymbol {p} \| _ {Z} ^ {*} \leq 1 \right\}
$$

where $\pmb { p } = ( p ^ { 1 } , p ^ { 2 } )$ are the dual variables and $\pmb { F } = ( \pmb { F } ^ { 1 } , . . . , \pmb { F } ^ { L } )$ are convolutional interpolation kernels defined as 

$$
(\boldsymbol {F} ^ {l} \boldsymbol {p}) _ {i, j} = \binom{(F ^ {l, 1} p ^ {1}) _ {i, j}}{(F ^ {l, 2} p ^ {2}) _ {i, j}} = \left( \begin{array}{c} \sum_ {m, n = - \nu} ^ {\nu} \xi_ {m, n} ^ {l} p _ {i + \frac {1}{2} - m, j - n} ^ {1} \\ \sum_ {m, n = - \nu} ^ {\nu} \eta_ {m, n} ^ {l} p _ {i - m, j + \frac {1}{2} - n} ^ {2} \end{array} \right)
$$

<sup>▶</sup> The primal form has the structure of a sparse coding problem 

$$
T V (u) = \min _ {\boldsymbol {q}: \boldsymbol {F} ^ {*} \boldsymbol {q} = D u} \| \boldsymbol {q} \| _ {Z},
$$

where ${ \pmb { F } } ^ { * }$ can be interpreted as a convolutional dictionary. 

<sup>▶</sup> Interpolation kernels (Nearest neighbor interpolation): 

$$
(F p) _ {i, j} = \binom{p _ {i + \frac {1}{2}, j} ^ {1}}{p _ {i, j + \frac {1}{2}} ^ {2}}.
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/6e8bcbaab8cb9714f7252139752c88b6a1ff6c89b66dadbfbd84c658b86709f6.jpg)



Interpolation kernels F


<sup>▶</sup> The Z-norm is given by 

$$
\| \boldsymbol {z} \| _ {Z} = \sum_ {i, j} \sqrt {(z _ {i + \frac {1}{2} , j} ^ {1}) ^ {2} + (z _ {i , j + \frac {1}{2}} ^ {2}) ^ {2}}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/e9c6d5f329cc51baf89a2e575a0ad3ab8a84b62786a48f91312353d1273e5cad.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/cdb237714512a3603bbcaafa41eeacf082bd437f35a7154399419feee56de14e.jpg)


<sup>▶</sup> Interpolation kernels (Nearest neighbor interpolation): 

$$
(F ^ {1} p) _ {i - \frac {1}{2}, j - \frac {1}{2}} = \binom{p _ {i - \frac {1}{2}, j} ^ {1}}{p _ {i, j - \frac {1}{2}} ^ {2}}, (F ^ {2} p) _ {i - \frac {1}{2}, j + \frac {1}{2}} = \binom{p _ {i - \frac {1}{2}, j} ^ {1}}{p _ {i, j + \frac {1}{2}} ^ {2}},
$$

$$
(F ^ {3} p) _ {i + \frac {1}{2}, j - \frac {1}{2}} = \binom{p _ {i + \frac {1}{2}, j} ^ {1}}{p _ {i, j - \frac {1}{2}} ^ {2}}, (F ^ {4} p) _ {i + \frac {1}{2}, j + \frac {1}{2}} = \binom{p _ {i + \frac {1}{2}, j} ^ {1}}{p _ {i, j + \frac {1}{2}} ^ {2}}.
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/af9b2529644b28a98ec3892a622fda3fa8a263baf192b18b474e0fcaa94f5822.jpg)



Interpolation kernels F


<sup>▶</sup> The Z-norm is given by 

$$
\left\| (\boldsymbol {z} ^ {1}, \boldsymbol {z} ^ {2}, \boldsymbol {z} ^ {3}, \boldsymbol {z} ^ {4}) \right\| _ {Z} := \sum_ {i, j} | \boldsymbol {z} _ {i - \frac {1}{2}, j - \frac {1}{2}} ^ {1} | _ {2} + | \boldsymbol {z} _ {i - \frac {1}{2}, j + \frac {1}{2}} ^ {2} | _ {2} + | \boldsymbol {z} _ {i + \frac {1}{2}, j - \frac {1}{2}} ^ {3} | _ {2} + | \boldsymbol {z} _ {i + \frac {1}{2}, j + \frac {1}{2}} ^ {4} | _ {2}
$$

<sup>▶</sup> Interpolation kernels (bilinear interpolation): 

$$
(F ^ {1} p) _ {i, j} = \binom{\frac {p _ {i - \frac {1}{2} , j} ^ {1} + p _ {i + \frac {1}{2} , j} ^ {1}}{2}}{\frac {p _ {i , j - \frac {1}{2}} ^ {2} + p _ {i , j + \frac {1}{2}} ^ {2}}{2}},
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/4b4aba5239aa981965d2c578fa10e01be70c4bbc9104073c44021048fe525158.jpg)


$$
(F ^ {2} p) _ {i + \frac {1}{2}, j} = \left( \begin{array}{c} p _ {i + \frac {1}{2}, j} ^ {1} \\ \frac {p _ {i , j - \frac {1}{2}} ^ {2} + p _ {i , j + \frac {1}{2}} ^ {2} + p _ {i + 1 , j - \frac {1}{2}} ^ {2} + p _ {i + 1 , j + \frac {1}{2}} ^ {2}}{4} \end{array} \right), (F ^ {3} p) _ {i, j + \frac {1}{2}} = \left( \begin{array}{c} p _ {i - \frac {1}{2}, j} ^ {1} + p _ {i + \frac {1}{2}, j} ^ {1} + p _ {i - \frac {1}{2}, j + 1} ^ {1} + p _ {i + \frac {1}{2}, j + 1} ^ {1} \\ \hline p _ {i, j + \frac {1}{2}} ^ {2} \end{array} \right)
$$

Interpolation kernels F 

<sup>▶</sup> The Z-norm is given by 

$$
\| (\boldsymbol {z} ^ {1}, \boldsymbol {z} ^ {2}, \boldsymbol {z} ^ {3}) \| _ {Z} := \sum_ {i, j} | \boldsymbol {z} _ {i, j} ^ {1} | _ {2} + | \boldsymbol {z} _ {i + \frac {1}{2}, j} ^ {2} | _ {2} + | \boldsymbol {z} _ {i, j + \frac {1}{2}} ^ {3} | _ {2}
$$

Input 


Forward diferences


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/ff96d5d911067010c619e34577ac0958d98b990715ac4476fa948f051a0b88f6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/580066a6265734eb75ba22ce729ac9e3d64b8c4b1e121dbe55d5f796294ba62a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/9fffa12580ea1b78272369a2c349ee61849e21b28a7c1dbfbeb0b69d78b264b0.jpg)


<sup>▶</sup> We define a family of discrete total variations for pixels of size $\varepsilon \times \varepsilon :$ 

$$
T V _ {\varepsilon} (u) = \min \left\{\varepsilon^ {2} \| \boldsymbol {q} \| _ {Z _ {\varepsilon}}: \boldsymbol {F} _ {\varepsilon} ^ {*} \boldsymbol {q} = D _ {\varepsilon} u \right\} = \sup \left\{\varepsilon^ {2} \langle \boldsymbol {p}, D _ {\varepsilon} u \rangle_ {Y _ {\varepsilon}}: \| \boldsymbol {F} _ {\varepsilon} \boldsymbol {p} \| _ {Z} ^ {*} \leq 1 \right\}
$$

## Theorem

Assume the supports and the weights of the convolutions defining $\pmb { F } _ { \varepsilon }$ are uniformly bounded that is 

$$
\sum_ {m, n} \xi_ {m, n} ^ {l} = \sum_ {m, n} \eta_ {m, n} ^ {l} = 1 \Longleftrightarrow F ^ {l, 1}, F ^ {l, 2} \in C _ {\Sigma = 1}.
$$

Then $T V _ { \varepsilon }$ Γ-converges to 

$$
T V (u) := \left\{ \begin{array}{l l} | D u | (\Omega) & \text { if } u \in B V (\Omega), \\ + \infty & \text { else }. \end{array} \right.
$$

As long as the filter coeficients sum up to one and are uniformly bounded, we are having a consistent discretization of the total variation <sup>⇝</sup> learning. 

<sup>▶</sup> We consider the following class of total variation minimization problems 

$$
\min _ {D u = \boldsymbol {F} ^ {*} \boldsymbol {q}} \lambda \| \boldsymbol {q} \| _ {Z} + G (u, g),
$$

with a saddle-point formulation 

$$
\min _ {u, \boldsymbol {q}} \max _ {\boldsymbol {p}} \left\langle D u - \boldsymbol {F} ^ {*} \boldsymbol {q}, \boldsymbol {p} \right\rangle + \lambda \| \boldsymbol {q} \| _ {Z} + G (u, g)
$$

<sup>▶</sup> Can be applied to a large class of inverse problems in imaging such as denoising, inpainting, segmentation, .... 

<sup>▶</sup> We need access to the proximal maps of $\lVert \cdot \rVert _ { Z }$ and $G ( \cdot , g )$ 

## Supervised learning

<sup>▶</sup> Assume we have given training data $( g _ { s } , t _ { s } ) , s = 1 , . . . , S$ 

<sup>▶</sup> We consider the following bilevel optimization problem: 

$$
\begin{array}{c} \min _ {\boldsymbol {F}} \mathcal {L} (\boldsymbol {F}) + \mathcal {R} (\boldsymbol {F}), \\ u _ {s} ^ {*} \in \arg \min _ {u, \boldsymbol {q}} \max _ {\boldsymbol {p}} \langle D u - \boldsymbol {F} ^ {*} \boldsymbol {q}, \boldsymbol {p} \rangle + \lambda \| \boldsymbol {q} \| _ {Z} + G (u, g _ {s}), \quad s = 1, \ldots , S \end{array}
$$

▶ $\mathcal { L } ( F )$ is a convex and diferentiable loss function 

$$
\mathcal {L} (\boldsymbol {F}) = \frac {1}{M N S} \sum_ {s = 1} ^ {S} \ell (u _ {s} ^ {*} (\boldsymbol {F}), t _ {s}),
$$

that measures the error between the targets $t _ { s }$ and the solutions $u _ { s } ^ { * }$ , here $\begin{array} { r } { \ell ( u , t ) = \frac 1 2 \left. u - t \right. _ { 2 } ^ { 2 } } \end{array}$ 

▶ $R ( F )$ can be used to impose the constraints on the filters F . 

$$
\mathcal {R} (\boldsymbol {F}) = \delta_ {(C _ {\Sigma = 1}) ^ {L, 2}} (\boldsymbol {F}) = \sum_ {l = 1} ^ {L} \delta_ {C _ {\Sigma = 1}} (F ^ {l, 1}) + \delta_ {C _ {\Sigma = 1}} (F ^ {l, 2})
$$

<sup>▶</sup> For gradient-based learning, we need to compute the derivatives of the loss function with respect to the linear operator F . 

## Interlude: Derivatives of saddle-points

<sup>▶</sup> We consider the following class of saddle-point problems 

$$
\min _ {x \in \mathcal {X}} \max _ {y \in \mathcal {Y}} \left\langle K x, y \right\rangle + g (x) - f ^ {*} (y),
$$

with corresponding primal and dual problems 

$$
\min _ {x \in \mathcal {X}} f (K x) + g (x) \Longleftrightarrow \max _ {y \in \mathcal {Y}} - f ^ {*} (y) - g ^ {*} (- K ^ {*} y)
$$

<sup>▶</sup> We assume that the problem has, for a given linear operator K, a unique saddle point $( \hat { x } , \hat { y } )$ characterized by 

$$
\left\{ \begin{array}{l} K \hat {x} - \partial f ^ {*} (\hat {y}) \ni 0 \\ K ^ {*} \hat {y} + \partial g (\hat {x}) \ni 0 \end{array} \right.
$$

<sup>▶</sup> Then we consider that we have given a convex loss function 

$$
\mathcal {L} (K) = \ell (\hat {x} (K), \hat {y} (K)).
$$

<sup>▶</sup> We are interested in the gradient of the loss with respect to the linear operator K. 

<sup>▶</sup> We will derive a formula based on a standard sensitivity analysis. 

<sup>▶</sup> Denote by $\hat { x } _ { s } = \hat { x } + s \xi _ { s }$ , and $\hat { y } _ { s } = \hat { y } + s \eta _ { s }$ the solution of the saddle-point problem perturbed by a small variation $K + s L , | s | \ll 1$ of the linear operator. 

▶ Substituting $( \hat { x } _ { s } , \hat { y } _ { s } )$ into the optimality system yields 

$$
\left\{ \begin{array}{l} K \hat {x} + s (K \xi_ {s} + L \hat {x} _ {s}) - [ \partial f ^ {*} (\hat {y}) + (\int_ {0} ^ {s} D ^ {2} f ^ {*} (\hat {y} + t \eta_ {s}) \mathrm{d} t) \eta_ {s} ] = 0, \\ K ^ {*} \hat {y} + s (K ^ {*} \eta_ {s} + L ^ {*} \hat {y} _ {s}) + [ \partial g (\hat {x}) + (\int_ {0} ^ {s} D ^ {2} g (\hat {x} + t \xi_ {s}) \mathrm{d} t) \xi_ {s} ] = 0. \end{array} \right.
$$

<sup>▶</sup> Again making use of the optimality condition and dividing by s gives 

$$
\left\{ \begin{array}{l} K \xi_ {s} + L \hat {x} _ {s} - \left(\frac {1}{s} \int_ {0} ^ {s} D ^ {2} f ^ {*} (\hat {y} + t \eta_ {s}) \mathrm{d} t\right) \eta_ {s} = 0, \\ K ^ {*} \eta_ {s} + L ^ {*} \hat {y} _ {s} + \left(\frac {1}{s} \int_ {0} ^ {s} D ^ {2} g (\hat {x} + t \xi_ {s}) \mathrm{d} t\right) \xi_ {s} = 0. \end{array} \right.
$$

<sup>▶</sup> Passing to the limit $s \to 0$ , one obtains the linear system in $( \xi , \eta )$ 

$$
\left\{ \begin{array}{l} K \xi + L \hat {x} - D ^ {2} f ^ {*} (\hat {y}) \eta = 0, \\ K ^ {*} \eta + L ^ {*} \hat {y} + D ^ {2} g (\hat {x}) \xi = 0. \end{array} \right. \iff \binom{\xi}{\eta} = \left( \begin{array}{c c} D ^ {2} g (\hat {x}) & K ^ {*} \\ - K & D ^ {2} f ^ {*} (\hat {y}) \end{array} \right) ^ {- 1} \binom{- L ^ {*} \hat {y}}{L \hat {x}}
$$

<sup>▶</sup> The directional derivative is then given by 

$$
\begin{array}{l} \mathcal {L} ^ {\prime} (K; L) = \Bigg \langle \nabla \ell (\hat {x}, \hat {y}),   \binom{\xi}{\eta} \Bigg \rangle = \Bigg \langle \nabla \ell (\hat {x}, \hat {y}),   \left( \begin{array}{c c} D ^ {2} g (\hat {x}) & K ^ {*} \\ - K & D ^ {2} f ^ {*} (\hat {y}) \end{array} \right) ^ {- 1} \binom{- L ^ {*} \hat {y}}{L \hat {x}} \Bigg \rangle \\ = \Bigg \langle \underbrace {\left( \begin{array}{c c} D ^ {2} g (\hat {x}) & K ^ {*} \\ - K & D ^ {2} f ^ {*} (\hat {y}) \end{array} \right) ^ {- 1} \nabla \ell (\hat {x} , \hat {y})} _ {= \binom{- \hat {X}}{\hat {Y}}} \Bigg . \left( \begin{array}{c} - L ^ {*} \hat {y} \\ L \hat {x} \end{array} \right) \Bigg \rangle = \Big \langle \hat {X},   L ^ {*} \hat {y} \Big \rangle + \Big \langle \hat {Y},   L \hat {x} \Big \rangle  , \end{array}
$$

where $\hat { X }$ and $\hat { Y }$ being the adjoint variables. 

<sup>▶</sup> Interestingly, the adjoint variables are themselves solutions of the quadratic saddle-point problem 

$$
\min _ {X} \max _ {Y} \left\langle K X,   Y \right\rangle + \frac {1}{2} \left\langle D ^ {2} g (\hat {x}) X,   X \right\rangle - \frac {1}{2} \left\langle D ^ {2} f ^ {*} (\hat {y}) Y,   Y \right\rangle + \left\langle \nabla \ell (\hat {x}, \hat {y}),   \binom{X}{Y} \right\rangle
$$

▶ This brings up the idea of running in parallel, a primary primal-dual algorithm solving the lower-level problem and a secondary primal-dual algorithm that solves the adjoint saddle-point problem. 

<sup>▶</sup> Such algorithmic scheme is denoted in the AD literature as “piggyback” algorithm [Griewank, Faure ’03]. 

Note that the secondary primal-dual algorithm depends on the solution of the primary primal-dua algorithm and hence must be analyzed as an algorithm with (summable) errors. 

<sup>▶</sup> The final gradient is then given by 

$$
\mathcal {L} ^ {\prime} (K; L) = \left\langle \hat {X}, L ^ {*} \hat {y} \right\rangle + \left\langle \hat {Y}, L \hat {x} \right\rangle \Longleftrightarrow \nabla \mathcal {L} (K) = \hat {X} \otimes \hat {y} + \hat {x} \otimes \hat {Y},
$$

which we usually compute via automatic diferentiation of the scalar products, in order to respect the structure and boundary conditions of the linear operator K (which can be complicated). 

## Piggyback primal-dual algorithm

<sup>▶</sup> The primary primal-dual algorithm is given by 

$$
\left\{ \begin{array}{l} x ^ {k + 1} = (I + \tau \nabla g)) ^ {- 1} (x ^ {k} - \tau K ^ {*} y ^ {k}) \\ \bar {x} ^ {k + 1} = x ^ {k + 1} + \theta (x ^ {k + 1} - x ^ {k}) \\ y ^ {k + 1} = (I + \sigma \nabla f ^ {*}) ^ {- 1} (y ^ {k} + \sigma K \bar {x} ^ {k + 1}). \end{array} \right.
$$

<sup>▶</sup> The secondary primal-dual algorithm is given by 

$$
\left\{ \begin{array}{l} X ^ {k + 1} = \nabla \operatorname{prox} _ {\tau_ {g}} (x ^ {k} - \tau K ^ {*} y ^ {k}) \cdot (X ^ {k} - \tau (K ^ {*} Y ^ {k} + \nabla_ {x} \ell (x ^ {k}, y ^ {k}))) \\ \bar {X} ^ {k + 1} = X ^ {k + 1} + \theta (X ^ {k + 1} - X ^ {k}) \\ Y ^ {k + 1} = \nabla \operatorname{prox} _ {\sigma f ^ {*}} (y ^ {k} + \sigma K \bar {x} ^ {k + 1}) \cdot (Y ^ {k} + \sigma (K \bar {X} ^ {k + 1} + \nabla_ {y} \ell (x ^ {k}, y ^ {k}))), \end{array} \right.
$$

where we have used the fact that ∇ prox $\mathbf { \chi } _ { \tau g } ( x ) = ( I + \tau D ^ { 2 } g ( \mathrm { p r o x } _ { \tau g } ( x ) ) ) ^ { - 1 }$ 

Theorem ([Bogensperger, Chambolle, P. ’22]) 

Assume that $f ^ { * } , g$ are strongly convex and $f , g ^ { * }$ are locally $C ^ { 2 , \alpha }$ then the piggyback primal-dual algorithm converges linearly. 

## Learning for inpainting

<sup>▶</sup> We train on 64 images of size 64 × 64 with directions uniformly sampled between [0, 2π] and we include random subpixel shifts. 

<sup>▶</sup> We train on a training set and evaluate on a test set. 

<sup>▶</sup> We experiment with diferent numbers of filters and diferent symmetry constraints for the filters. 

(a) Input images g<sub>s</sub> 

(b) Target images t 

$$
L = 2
$$

<table><tr><td>Data</td><td>FD</td><td>RT</td><td>CD</td><td><eq>L = 2</eq></td><td><eq>L = 2(s)</eq></td><td><eq>L = 3</eq></td><td><eq>L = 3(s)</eq></td><td><eq>L = 4(s)</eq></td><td><eq>L = 8(s)</eq></td></tr><tr><td>Train</td><td>135</td><td>195</td><td>6.69</td><td>1.26</td><td>1.22</td><td>1.19</td><td>1.27</td><td>0.85</td><td>0.77</td></tr><tr><td>Test</td><td>134</td><td>194</td><td>6.33</td><td>1.63</td><td>1.45</td><td>1.29</td><td>1.29</td><td>0.87</td><td>0.82</td></tr></table>


Table: $1 0 ^ { 5 } ~ \times$ the mean squared error (MSE) of handcrafted and learned filters evaluated on both the training and test data.


## Note that transpose symmetry is almost automatically learned!


Filter: L = 8 (s)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/138c83e4dfda905acf471b1148b9e12e23ca316af57b8f9740758a33ca4c4b33.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/e191b99b933672b718b6c6b6d7c073ac5592504afcc0781327c8cbbe071de927.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/eed7453b8acab2eaed30a23517db97c284f5018d0e59c8092bcf5b69da8c4982.jpg)


## Learning for disk regularization

<sup>▶</sup> We train on 64 images with binary disks of various radii and subpixel shifted centers. 

<sup>▶</sup> The ground truth solutions can be computed with an explicit formula. 

<sup>▶</sup> We train on a training set and evaluate on a test set. 

<sup>▶</sup> We experiment with diferent numbers of filters and diferent symmetry constraints for the filters. 

## aaaaaaaa aaaaaaaa

(a) Input images g<sub>s</sub> 

## a a a aa a aa

(b) Target images t<sub>s</sub> 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/db3dc8b4cc3c4d13200b88e2ad26865dbef48d5aa69eb3c0a781e98ca1013d4a.jpg)


<table><tr><td>Data</td><td>FD</td><td>RT</td><td>CD</td><td><eq>L = 2</eq></td><td><eq>L = 2(s)</eq></td><td><eq>L = 3</eq></td><td><eq>L = 3(s)</eq></td><td><eq>L = 4(s)</eq></td><td><eq>L = 8(s)</eq></td></tr><tr><td>Train</td><td>22.28</td><td>1.36</td><td>2.33</td><td>2.10</td><td>2.10</td><td>1.62</td><td>1.63</td><td>0.73</td><td>0.48</td></tr><tr><td>Test</td><td>22.36</td><td>1.32</td><td>2.30</td><td>2.10</td><td>2.10</td><td>1.60</td><td>1.60</td><td>0.72</td><td>0.47</td></tr></table>


Table: $1 0 ^ { 5 }$ times the mean squared error (MSE) of handcrafted and learned filters for the disk denoising problem.



Observe that $L = 4 \left( s \right)$ is very close to RT on cubic meshes!


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/b025561731f053459848c731ab0fa1230e51685eff1f4e4bfa96dc021fac14aa.jpg)



Target FD RT CD L = 2 (s) L = 3 (s) L = 4 (s) L = 8 (s)


## Natural image denoising

<sup>▶</sup> We extract 64 patches of size 64 × 64 from a natural image database. 

<sup>▶</sup> The input images $g _ { s }$ contain 5% Gaussian noise. 

<sup>▶</sup> We learn both the filter weights and the regularization parameter λ by projecting on the set of filters with sum equals λ > 0. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/227a860dbdb05db69c40c34fd071eb6672203a39dd3ce6b6c31f526e0e01c8ba.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/ade5508cc25fdeb063adca269899127636a52b5fb55b7e78779ad211ea12a580.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/f627c23fa91704d6653d11d0d362288cef0a3a2bfbaaaeee66efeb660beba229.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/c400e88d43983f3c7515111d2c3ba9e9a6005214917c1c9ce160a71bf3fd4c59.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/8735dbfd8b2fc45edf7f27b328b2b5288ed90fa36f2a55a43f6e5b95697687ea.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/0a6ba8ba1eb738d721701aa8112a87d05e35b5aa7d231839d39f859e4dfc6614.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/2d0bd2e586756d43e7095c3ead4b4fec6373f508ca32424ce370b26b36c5427f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/0c8a8e74359c3e1c632cef8e85ed516a5fec6691e99e29b59f718aa1b9b432b1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/8cdba4af96ecaea318a1be393e7d559d012015d2d203b2540dcc10d1b875cbdc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/4079f7d3546eee4e53fd6b30cbe7320f91e7c067f7313d64946e0631b9076cb2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/df89a61fc55459e046a840ee71d13b7fa19bf79d103978c3dd4bd5c33c59fe48.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/c5b40b3dd485134c0dd42f44ed5da20ff292e42ac153857411abedcd8d98fc36.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/ad52cc7ebf2a41c679a534730b05da6e64b1dcf5952be0e4c4cc1ead1559754a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/670c452bb817e8bfd1d0282a4607c248b4f6612f858251014b156491bb10fe7e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/b892165094340c3f238489dc3f943708fba35b001f1e5e73058161e9f99513a3.jpg)



(a) Input images g<sub>s</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/9898b84dec6573bacbab9fa5eb4514d8ca3d8b3a52673a8344d30473582f4589.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/4acb841a7ef294314a7196e89d46e58c7f80058e50e10ff11d9f57631a696251.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/4cb8287b6bf5adf0f3417024880d7a6ae963e5fcd991198787c2befc9ed0225e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/d13cda2833278099e306ba7c042ada10e8f634f858150ca26f43a29df16fcefa.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/c3ed26e0b60d4b3a1efa2a6e6b3ed7f406a8bd467d3fa30d9790cd5d1078b6f8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/28d46cd4375882c78a7f25367055bf15f7a359b6a5e477635f30e10bb0c211bc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/25c09245fb1080bf13cc051d013ad75992d5c62d30051fd963413a1272bc7330.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/f07d41cdb155108d72f7185b9beb0856bf9f7372b64a0fa88a216a22a3963f83.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/b792e53bf07d2f239bfdc088bbd2d01545a10abf91e5d3e75950efdc459d7ccd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/92df0cbe546bd74d9cea9df3a409653206ddc5b839d39781992d39d1660d7a1f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/bde6b922ec147fba76537747cf9411cd8446be07cc06e80d8f358979b8b4628b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/7740b978ec4a955052a7e658215c71dd0bcc7fb42252680ca27603a4e6c5f4f5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/3576ac67d9e0a231ea29f3eaf0ac41c7f0e2e5993e11079f5d91bb7017a06f22.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/ed008ace0119156031730b820c6c95080cf2fc8bd6b16d67509582a8d0b8a754.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/23ae8dff975d8953e903eea5238841167733b79d3f8837c98f478a863a589299.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/4450e474612f137360048d00a4152b39a287c42c96707e291d1b08e8d37b8daf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/449d8de6fc363b27dc3584ae93be4d5d5c503de1dc211fea8c3fd7a350dc88ba.jpg)



(b) Target images t


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/df704f4b031d76731922b324f3e4507fc15cb235caacb35e8416523a7d190d98.jpg)


<table><tr><td>Data</td><td>FD</td><td>RT</td><td>CD</td><td><eq>L = 8(s)</eq></td><td><eq>L = 40(s)</eq></td><td><eq>L = 40</eq></td></tr><tr><td>Train</td><td>5.05</td><td>5.33</td><td>4.87</td><td>4.58</td><td>4.31</td><td>4.22</td></tr><tr><td>Test</td><td>4.72</td><td>5.05</td><td>4.51</td><td>4.28</td><td>4.10</td><td>4.13</td></tr></table>


Table: $1 0 ^ { 4 } \times$ the mean squared error (MSE) of handcrafted and learned filters for natural image denoising.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/9fcacdfe57ab6dde4bec496dd09a0ab68fac56a07304f2c1cee3653a2b9df46d.jpg)



(a) Training set


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/03a29959f115db20dc7ec77ebf64a5a7f42a011a0108293426263d63bbf37cf9.jpg)



(b) Test set



Target t


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/72f04b96ee25bf59936f4d89f85da0766b6397d97bd138d57dea7c03c07d8618.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/7877ee7711cd915aec2951bc3c7529f3962f934b4cc1becdca981357505a8e7d.jpg)



(c) Example from the test set



CD, PSNR=28.98


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/0ff202d173d57dc4810bfcff33fd3bcb7b75396568d62c87bf3ebc7d503a2a07.jpg)



= 40 ( ), PSNR=29.77


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/3b3b8a828f5068b025e839d70bfc22936f19cb8f0fcf35c2129967a66c7f7c3e.jpg)


<sup>▶</sup> How well do the learned filters generalize to other tasks? 

<sup>▶</sup> We compare the filters $L = 8 \left( s \right)$ which gave good results on all tasks. 

<table><tr><td rowspan="2" colspan="2"></td><td colspan="3">Learning task</td><td>Handcrafted</td></tr><tr><td>Line</td><td>Disk</td><td>Natural</td><td>CD</td></tr><tr><td rowspan="3">Evaluation task</td><td>Line</td><td>0.82</td><td>243.55</td><td>50.71</td><td>6.33</td></tr><tr><td>Disk</td><td>1.88</td><td>0.47</td><td>4.08</td><td>2.30</td></tr><tr><td>Natural</td><td>48.68</td><td>49.65</td><td>42.80</td><td>45.10</td></tr></table>

<sup>▶</sup> The filters learned for inpainting generalize best, but there is no universal best discretization. 

## Extension to 3D

<sup>▶</sup> Very recently, we have extended the learning of the discrete total variation to 3D 

<sup>▶</sup> We learn 4 sets of 3D filters on 3D minimal surface problems for which closed form solutions are available 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/0b19e6c241d41f428b56b70c9c9e3ccd60b746eff81b9e74bbda655a89dea920.jpg)



Catenoid


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/5dffe790b6c3e68d523cf065fed2dff9deecd2a2dca8ae6a71ea69989d99fdcb.jpg)



3D filters


## Computing the Schwarz P surface

<sup>▶</sup> After learning, we can compute high-accuracy minimal surfaces for which no closed for solution is available. 

<sup>▶</sup> A well-known example is the Schwarz P surface 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/6e5f1e69208f42a8e95ee15e67e125a7f864e575e262a8985a97ed7168976330.jpg)


<sup>▶</sup> In recent work [Bogensperger, Chambolle, Efland, P. ’23] we have extended the framework to the secons order total generalized variation [Bredies, Kunsich, P. ’10] 

$$
\mathsf {T G V} _ {\alpha} ^ {2} (u) = \sup _ {p} \bigg \{\int_ {\Omega} u \mathrm{div} ^ {2} p \mathrm{d} x: p \in \mathcal {C} ^ {\infty} (\Omega , \mathsf {S y m} ^ {2 \times 2}), \| p \| _ {\infty} \leq \alpha_ {0}, \| \mathrm{div} p \| _ {\infty} \leq \alpha_ {1} \bigg \},
$$

<sup>▶</sup> In contrast to the total variation, the second order TGV can reconstruct piecewise afine images. 

<sup>▶</sup> In [Hosseini, Bredies ’22] a Condat-like discretization was proposed, which served as the starting point for our work. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/fa7072949133c007e9b5aa4a6be59f7e1052788eb60c6d4133c8eb1ce2393d85.jpg)



(a) Noisy


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/eaa366884114d5cefaf3ad9f4d7840bffcb5282c9bd991f3557a9e638e4b7a94.jpg)



(b) TV


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/d39368eb9affbfaefa5b95381c983382a5dc0f2e2e8170dc5102288f88f4367a.jpg)



(c) TGV2


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/a40b100a249d5a8c35b3d31a84f253f45288763bc2984171934b0e956772f6ba.jpg)



(d) Graph of TV


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/856bf013298422e819df38aa4cbc14b9bf950ec2268d2219ed2ccf62b07da4a2.jpg)



(e) Graph of TGV2


## Discrete model

<sup>▶</sup> The second-order TGV discretization in its primal form is given by 

$$
T G V (u) = \min _ {w} \alpha_ {1} \| D u - w \| + \alpha_ {0} \| E w \|,
$$

where $D$ is again the finite diferences operator $D u = ( ( D u ) ^ { 1 } , ( D u ) ^ { 2 } )$ , where 

$$
\begin{array}{l l} (D u) _ {i + \frac {1}{2}, j} ^ {1} = \frac {1}{h} (u _ {i + 1, j} - u _ {i, j}) & i \leq M - 1, j \leq N, \\ (D u) _ {i, j + \frac {1}{2}} ^ {2} = \frac {1}{h} (u _ {i, j + 1} - u _ {i, j}) & i \leq M, j \leq N - 1. \end{array}
$$

and $E$ is the symmetrized vectorial gradient operator given by $E w = { \binom { ( E w ) ^ { 1 } } { ( E w ) ^ { 2 } } } \quad ( E w ) ^ { 2 } \quad $ with 

$$
\begin{array}{l l} (E w) _ {i + 1, j} ^ {1} = \frac {1}{h} (w _ {i + \frac {3}{2}, j} ^ {1} - w _ {i + \frac {1}{2}, j} ^ {1}) & i \leq M - 1, j \leq N, \\ (E w) _ {i + \frac {1}{2}, j + \frac {1}{2}} ^ {2} = \frac {1}{2 h} (w _ {i + \frac {1}{2}, j + 1} ^ {1} - w _ {i + \frac {1}{2}, j} ^ {1} + w _ {i + 1, j + \frac {1}{2}} ^ {2} - w _ {i, j + \frac {1}{2}} ^ {2}) & i \leq M - 1, j \leq N - 1, \\ (E w) _ {i, j + 1} ^ {3} = \frac {1}{h} (w _ {i, j + \frac {3}{2}} ^ {2} - w _ {i, j + \frac {1}{2}} ^ {2}) & i \leq M, j \leq N - 1. \end{array}
$$

<sup>▶</sup> Now, introducing again interpolation operators L and K, applying some “convexity magic” the model becomes 

$$
T G V (u) = \sup _ {p} \langle u, \operatorname{div} ^ {2} p \rangle , \text {   s.t.   } \| L \operatorname{div} p \| _ {Z} ^ {*} \leq \alpha_ {1}, \| K p \| _ {Z} ^ {*} \leq \alpha_ {0}.
$$

<table><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td><eq>u_{i,j}</eq></td><td><eq>u_{i+1,j}</eq></td><td></td><td></td><td><eq>w_{i+1/2,j}^{1}</eq></td><td><eq>w_{i+3/2,j}^{1}</eq></td><td></td><td><eq>p_{i+1,j}^{1}</eq></td><td><eq>p_{i+2,j}^{1}</eq></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td><eq>w_{i,j+1/2}^{2}</eq></td><td><eq>w_{i+1,j+1/2}^{2}</eq></td><td></td><td><eq>p_{i+1/2,j+1/2}^{2}</eq></td><td><eq>p_{i+3/2,j+1/2}^{2}</eq></td><td></td></tr><tr><td></td><td><eq>u_{i,j+1}</eq></td><td><eq>u_{i+1,j+1}</eq></td><td></td><td></td><td><eq>w_{i+1/2,j+1}^{1}</eq></td><td><eq>w_{i+3/2,j+1}^{1}</eq></td><td></td><td><eq>p_{i,j+1}^{3}</eq></td><td><eq>p_{i+1,j+1}^{1}</eq></td><td><eq>p_{i+1,j+1}^{3}</eq></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td><eq>w_{i,j+3/2}^{2}</eq></td><td><eq>w_{i+1,j+3/2}^{2}</eq></td><td></td><td><eq>p_{i+1/2,j+3/2}^{2}</eq></td><td><eq>p_{i+3/2,j+3/2}^{2}</eq></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td><eq>p_{i,j+2}^{3}</eq></td><td><eq>p_{i+1,j+2}^{3}</eq></td><td></td></tr></table>

The discrete TGV model is given by 

$$
\begin{array}{l} \mathsf {T G V} _ {\alpha , h} ^ {2} (u ^ {h}) = \min _ {w ^ {h}, v _ {K} ^ {h}, v _ {L} ^ {h}} \bigg \{h ^ {2} \alpha_ {1} \| v _ {L} ^ {h} \| _ {Z} + h ^ {2} \alpha_ {0} \| v _ {K} ^ {h} \| _ {Z}: L _ {h} ^ {*} v _ {L} ^ {h} = D _ {h} u ^ {h} - w ^ {h}, K _ {h} ^ {*} v _ {K} ^ {h} = E _ {h} w ^ {h} \bigg \} \\ = \sup _ {p ^ {h}} \bigg \{h ^ {2} \langle \mathrm{div} _ {h} ^ {2} p ^ {h}, u ^ {h} \rangle : \| L _ {h}   \mathrm{div} _ {h}   p ^ {h} \| _ {Z} ^ {*} \leq \alpha_ {1}, \| K _ {h} p ^ {h} \| _ {Z} ^ {*} \leq \alpha_ {0} \bigg \}. \end{array}
$$

## Theorem

We consider the setting where u is afine plus periodic with period $1 \ i n \ \mathbb { R } ^ { 2 }$ , and w is 1-periodic. Then, for interpolation operators K and L that have local support and bounded filter coeficients, TG $\mathsf { V } _ { \alpha , h } ^ { 2 } ( u ^ { h } )$ Γ-converges to $\mathsf { T G V } _ { \alpha } ^ { 2 } ( u )$ 

## Quantitative results for image denoising

Table: Quantitative comparison of natural image denoising of the test set with 5% and 10% Gaussian noise for diferent handcrafted and learned discretizations. 

<table><tr><td rowspan="2"></td><td colspan="3">5% Gaussian noise</td><td colspan="3">10% Gaussian noise</td></tr><tr><td>PSNR</td><td>MSE·<eq>10^{-2}</eq></td><td>SSIM</td><td>PSNR</td><td>MSE·<eq>10^{-2}</eq></td><td>SSIM</td></tr><tr><td>Corrupted <eq>f</eq></td><td>26.04</td><td>0.2490</td><td>0.7885</td><td>20.02</td><td>0.9959</td><td>0.5382</td></tr><tr><td>TV</td><td>30.14</td><td>0.1049</td><td>0.9249</td><td>26.52</td><td>0.2445</td><td>0.8497</td></tr><tr><td>TGV</td><td>30.2</td><td>0.1043</td><td>0.9257</td><td>26.56</td><td>0.2431</td><td>0.8512</td></tr><tr><td>Handcrafted Disc. <eq>n_{K}=1, n_{L}=3</eq></td><td>30.24</td><td>0.1046</td><td>0.9267</td><td>26.69</td><td>0.2394</td><td>0.8553</td></tr><tr><td>Handcrafted Disc. <eq>n_{K}=4, n_{L}=4</eq></td><td>30.29</td><td>0.1030</td><td>0.9278</td><td>26.71</td><td>0.2370</td><td>0.8565</td></tr><tr><td>Learned Disc. <eq>n_{K}=1, n_{L}=3, 3 \times 3</eq></td><td>30.52</td><td>0.0935</td><td>0.9274</td><td>26.95</td><td>0.2172</td><td>0.8596</td></tr><tr><td>Learned Disc. <eq>n_{K}=4, n_{L}=4, 3 \times 3</eq></td><td>30.66</td><td>0.0906</td><td>0.9298</td><td>27.06</td><td>0.2123</td><td>0.8620</td></tr><tr><td>Learned Disc. <eq>n_{K}=8, n_{L}=8, 7 \times 7</eq></td><td>30.74</td><td>0.0896</td><td>0.9314</td><td>27.14</td><td>0.2090</td><td>0.8649</td></tr><tr><td>Learned Disc. <eq>n_{K}=8, n_{L}=8, 7 \times 7,</eq>sym.</td><td>30.72</td><td>0.0898</td><td>0.9311</td><td>27.15</td><td>0.2089</td><td>0.8649</td></tr><tr><td>Learned Disc. <eq>n_{K}=10, n_{L}=10, 7 \times 7</eq></td><td>30.73</td><td>0.0896</td><td>0.9313</td><td>27.17</td><td>0.2081</td><td>0.8657</td></tr><tr><td>Learned Disc. <eq>n_{K}=16, n_{L}=16, 7 \times 7</eq></td><td>30.77</td><td>0.0891</td><td>0.9319</td><td>27.16</td><td>0.2087</td><td>0.8654</td></tr><tr><td>Learned Disc. <eq>n_{K}=16, n_{L}=16, 7 \times 7,</eq>sym.</td><td>30.77</td><td>0.0890</td><td>0.9320</td><td>27.18</td><td>0.2074</td><td>0.8659</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/24c18d40c8fb4e0d23d7506d48a22bf2ac9d48cc71d0da91b6ab0b1574f07ba3.jpg)



learned filters K with $n _ { K } { = } 1 6$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/8cabf23bccc9fe515f7779b7beade47a1344abb384aca036b2a4e3b0329f1697.jpg)



learned filters L with $n _ { L } { = } 1 6$



Figure: Learned $7 \times 7$ filters using $n _ { L } = 1 6$ and $n _ { K } = 1 6$ for denoising (10% Gaussian noise). The row of a depicted filter denotes the component of the respective vector/tensor field that it acts upon, whereas the column refers to the specific filter r or l (with $r = 1 , \cdots , n _ { K }$ and $l = 1 , \cdots , n _ { L } . )$


learned 

standard TGV 

## Example results

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/11085a2bf74ae1730da27dd614f26ed095b52fb2522c72824e1168f52612de0a.jpg)



Figure: Sample reconstructions from natural test images (10% Gaussian noise) comparing the standard TGV, the handcrafted discretization scheme with $n _ { K } = 1 , n _ { L } = 3$ , and learned filters using $n _ { K } = 1 6 , n _ { L } = 1 6$ . For GV standard TGV corrupted han crafted standard TGV han crafted standard Than crlea<sub>completeness,</sub> <sub>the</sub> <sub>ground</sub> <sub>truth</sub> <sub>images</sub> <sub>are</sub> <sub>also</sub> <sub>shown.</sub>


## Beyond total variation regularization

<sup>▶</sup> In [Bogensperger, Chambolle, P. ’22], we applied the same learning framework to the shearlet transform [Kanghui, Kutyniok, Labate ’06]. 

<sup>▶</sup> A shearlet at scale j and shearing k is defined as. 

$$
\psi_ {j, k} ^ {d} = \left[ \left(S _ {k} \big ((p _ {j} * W _ {j}) _ {\uparrow 2 ^ {j / 2}} * _ {1} h _ {j / 2}\right)\right) * _ {1} \bar {h} _ {j / 2} \Big ] _ {\downarrow 2 ^ {j / 2}},
$$

which essentially is constructed from a 1D low-pass filter $h _ { 1 }$ and anisotropic 2D filter P . Additionally we also learn the importance weight $\lambda _ { j , k }$ of each shearlet. 

▶ Shearlets provide a multiscale framework similar to wavelets but better suited for encoding anisotropic features necessary for an eficient sparse representations of cartoon-like images. 

<sup>▶</sup> We tried to further optimize the shearlets using the piggy-back primal dual algorithm based on smooth-ℓ -regularized image denoising model 

$$
\min _ {u} \left\| K (\theta) u \right\| _ {1, \varepsilon} + \frac {1}{2} \left\| u - z \right\| _ {2} ^ {2},
$$

where $\theta$ is a placeholder for all learnable parameters and $\lVert \cdot \rVert _ { 1 , \iota }$ refers to a $C ^ { 2 , 1 }$ approximation of the $\ell _ { 1 }$ norm. 

<sup>▶</sup> Experiments are carried out on both natural images and synthetic piecewise afine images and we experimented with diferent settings of the smoothness parameter ε. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/7392902e5e5b3ed622a668457868df9585d0040e207ee4844a279aafbe42f223.jpg)


## Learned shearlets

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/014289368bc0ad9f867fdc6ab0bc26bf47488cb02b2953b11c86ed6368b9fb36.jpg)


## Image denoising

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/e20bb38cc90845b33783cc6c422f29fd0d3d89f994345b2aacf996335785fa2b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/a0ed0d921a572cdb7696452206956a9391bc76b81bb127446fabd754b0b8f1d8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/9d4912e8fc3f79ac5f4fb2b9ebf0646200ce1b2fff9aa28cfda9e578325ff134.jpg)



(a) ground truth sample test images t1.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/7d6b1ba2e4c948784f455e49837db24d038b5fb7fc7efd7cc4ddc541319103bd.jpg)



PSNR=29.04


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/73d3ba7c23b410e888e4090c164ce1035c12ba9826c83ec71f8a92be700f2075.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/0564e01d8ef90aca212fb8e1fee50043863afb7d43b2b7d73f2ca56de4d06054.jpg)



PSNR=28.71



PSNR=31.61



(c) sample denoised images ul with initial shearlet system parameters and global λ = 1.33


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/344c9dbcf8d734a0f51046c2877af941313a09e22c34e464051bc1f03b2ea339.jpg)



PSNR=26.01


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/ff5fa35698b4110b83c93217ee17ae07c9907eb0b98e99a8c833ddd437516223.jpg)



PSNR=26.02


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/dd1ed91bbe8c3df36ce59b92face5b0580bfe5c1a346717b3504d264cf12cbf4.jpg)



PSNR=26.02



(b) noisy sample images zl.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/e6eca64d44f110475ae528fa5b1ef9d75e43dbeec4d904552e34648eef02f92f.jpg)



PSNR=30.32


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/d4630b9b06f1d0f35830a19042ed3dae2d7c04905bae6b2eb50b74685d4d9c15.jpg)



PSNR=30.21


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/bca02e57863c643e5c7c0eb0e25330d730d36de6e73a25ec3cd2e451039b0cb9.jpg)



PSNR=32.61



(d) sample denoised images ul with learned shearlet system parameters shown in Figure 7.1.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/4a6b016b11e6c17e7bac94165e88c88ec174a3af62c1953ca78f9eafd8c2d0e6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/b3232cd133465dedb31970ae73ac45c94d7bf5b651d4cfabdd5068cd6aff0663.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/f94aa4c535c0e257310e74a4f77625d1838aaff05e7d4f87346ea58805045a68.jpg)



(a) ground truth sample test images tl.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/057eb418b1f74215840601facf6ed73d168cfa441a29cc7d9957afd684a8df85.jpg)



PSNR=39.04


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/dfc5b6d792b1be7b4ea4f8fde03ffa3e51e6e3905132ee36617a62e9498116b6.jpg)



PSNR=38.33


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/0b4154e7d4b978511fec3b11274425e9b6d15294f74cbcb39f7aefecde4e8366.jpg)



PSNR=39.68



(c) sample denoised images $u _ { l }$ with initial shearlet system parameters and global λ = 2.04.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/c88c246b7206b195fe7853467a566c0dc78053cb4f73f99444633ee2c2d2ba6d.jpg)



PSNR=26.01


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/61ff16405dd82692b682c53b3201d554524533f3c94a22a6238c7f9dd3f87782.jpg)



PSNR=26.03


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/1706c263c9cb592d5d7ea9bfabc6555bb6bf724bfe2414c6fcb43cddf7b6c74f.jpg)



PSNR=26.05



(b) noisy sample images $z _ { l }$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/0b1f3ff7da206cc42933cca9b09d530b11199246762259bf45c0ae881bbcaa9a.jpg)



PSNR=40.03


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/f3aa813d2ed9fdb5b1d9a2bb90c849d1fd9be113e3991a0735f4a6c7b1b7e84c.jpg)



PSNR=39.00


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/0ee0d908-a423-4bd4-8afb-124e18377c0e/f0b2b8e8e30c6b0560f282f540e08d4596d66af9cca13e89b1ab44066d77de56.jpg)



PSNR=40.92



(d) sample denoised images ul with learned shearlet system parameters shown in Figure 7.1



Comparison between the performance of the initial shearlet transform (hand tuned. $\lambda = 1 . 3 3$ and $\lambda = 2 . 0 4$ resp.) and the optimized shearlet transform for various setting of the smoothing parameter ε on the natural (N) and cartoon-like (C) dataset. Note that the learned transform clearly outperforms the initial shearlet transform and that smaller settings of ε lead to better results.


<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="2"><eq>\varepsilon = 10^{-1}</eq></td><td colspan="2"><eq>\varepsilon = 10^{-2}</eq></td><td colspan="2"><eq>\varepsilon = 10^{-3}</eq></td><td colspan="2"><eq>\varepsilon = 10^{-4}</eq></td><td colspan="2"><eq>\varepsilon = 0</eq></td></tr><tr><td>MSE</td><td>PSNR</td><td>MSE</td><td>PSNR</td><td>MSE</td><td>PSNR</td><td>MSE</td><td>PSNR</td><td>MSE</td><td>PSNR</td></tr><tr><td rowspan="2">N</td><td>Initial</td><td>0.001929</td><td>27.15</td><td>0.001187</td><td>29.38</td><td>0.001056</td><td>30.05</td><td>0.00105</td><td>30.1</td><td>0.00105</td><td>30.1</td></tr><tr><td>Optimized</td><td>0.001148</td><td>29.56</td><td>0.000931</td><td>30.53</td><td>0.000821</td><td>31.14</td><td>0.000813</td><td>31.2</td><td>0.000813</td><td>31.2</td></tr><tr><td rowspan="2">C</td><td>Initial</td><td>0.001392</td><td>28.57</td><td>0.000345</td><td>34.62</td><td>0.000155</td><td>38.14</td><td>0.000134</td><td>38.79</td><td>0.000132</td><td>38.84</td></tr><tr><td>Optimized</td><td>0.000293</td><td>35.44</td><td>0.000165</td><td>37.87</td><td>0.000115</td><td>39.45</td><td>0.000103</td><td>39.92</td><td>0.000101</td><td>40.02</td></tr></table>

<sup>▶</sup> We proposed learning optimized finite diferences discretizations of the total variation. 

▶ The learning is constraint to a class of consistent discretizations which Γ-converge to the continuous total variation. 

<sup>▶</sup> We proposed a piggy-back primal-dual algorithm for computing derivatives. 

▶ Symmetry constraints on the filters give better generalizations. 

<sup>▶</sup> The learned discretizations give significant improvements when optimized for certain applications but no best universal discretization could be learned. 

<sup>▶</sup> The learning framework has been extended to 3D TV and more complex regularization operators such as TGV and shearlets. 

<sup>▶</sup> We proposed learning optimized finite diferences discretizations of the total variation. 

▶ The learning is constraint to a class of consistent discretizations which Γ-converge to the continuous total variation. 

<sup>▶</sup> We proposed a piggy-back primal-dual algorithm for computing derivatives. 

▶ Symmetry constraints on the filters give better generalizations. 

<sup>▶</sup> The learned discretizations give significant improvements when optimized for certain applications but no best universal discretization could be learned. 

<sup>▶</sup> The learning framework has been extended to 3D TV and more complex regularization operators such as TGV and shearlets. 

Thank you for your attention! 