# Solving Inverse Problems via Diffusion Optimal Control

Henry Li <sup>∗</sup> Yale University henry.li@yale.edu 

Marcus Pereira Bosch Center for Artificial Intelligence marcus.pereira@us.bosch.com 

## Abstract

Existing approaches to diffusion-based inverse problem solvers frame the signal recovery task as a probabilistic sampling episode, where the solution is drawn from the desired posterior distribution. This framework suffers from several critical drawbacks, including the intractability of the conditional likelihood function, strict dependence on the score network approximation, and poor $\mathbf { x } _ { \mathrm { 0 } }$ prediction quality. We demonstrate that these limitations can be sidestepped by reframing the generative process as a discrete optimal control episode. We derive a diffusion-based optimal controller inspired by the iterative Linear Quadratic Regulator (iLQR) algorithm. This framework is fully general and able to handle any differentiable forward measurement operator, including super-resolution, inpainting, Gaussian deblurring, nonlinear deblurring, and even highly nonlinear neural classifiers. Furthermore, we show that the idealized posterior sampling equation can be recovered as a special case of our algorithm. We then evaluate our method against a selection of neural inverse problem solvers, and establish a new baseline in image reconstruction with inverse problems<sup>1</sup>. 

## 1 Introduction

Diffusion models Song and Ermon [2019], Ho et al. [2020] have been shown to be remarkably adept at conditional generation tasks Dhariwal and Nichol [2021], Ho and Salimans [2022], in part due to their iterative sampling algorithm, which allows the dynamics of an uncontrolled prior score function $\nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } )$ to be directed towards an arbitrary posterior distribution by introducing an additive guidance term u. When this guidance term is the conditional score $\nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { y } | \mathbf { x } )$ , the resulting sample is provably drawn from the desired conditional distribution $p ( \mathbf { x } | \mathbf { y } )$ Song et al. [2020]. 

A central obstacle to this framework is the general difficulty of obtaining the conditional score function $\nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { y } | \mathbf { x } _ { t } )$ due to its dependence on the noisy diffusion variate x<sub>t</sub> rather than just the final sample $\mathbf { x } _ { \mathrm { 0 } }$ Chung et al. [2023a]. In large-scale conditional generation tasks such as class- or text-conditional sampling the computational overhead of training a time-dependent conditional score function from scratch is deemed acceptable, and is indeed the approach taken by Rombach et al. [2022], Saharia et al. [2022], and many others. However, this solution is not acceptable in inverse problems where the goal is to design a generalized solver that will work in a zero-shot capacity for an arbitrary forward model. 

This bottleneck has spawned a flurry of recent research dedicated to approximating the conditional score $\nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { y } | \mathbf { x } _ { t } )$ as a simple function of the noiseless likelihood log p(y|x<sub>0</sub>) Choi et al. [2021], Chung et al. [2022], Rout et al. [2024], Chung et al. [2023a], Kawar et al. [2022], Chung et al. [2023b]. However, as we will demonstrate in this work, these approximations impose a significant cost to the performance of the resulting algorithm. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/699aafed-e030-4fc3-a2c9-b280b0c85bb7/4a646f6ca5166973ecf78c6da8dcc373855925bcabb33b8c35c76f98c9a498e1.jpg)



Figure 1: Conceptual illustration comparing a probabilistic posterior sampler to our proposed optimal control-based sampler. In a probabilistic sampler, the model relies on an approximation $\tilde { \mathbf { x } } _ { 0 } \approx \mathbf { x } _ { 0 }$ to guide each step (left). We are able to compute $\mathbf { x } _ { \mathrm { 0 } }$ exactly on each step, resulting in much higher quality gradients $\bar { \nabla } \log p ( \mathbf { y } | \tilde { \mathbf { x } } _ { 0 } )$ and an improved trajectory update (right).


To address these issues, we propose a novel framework built from optimal control theory where such approximations are no longer necessary. By framing the reverse diffusion process as an optimal control episode, we are able to detach the inverse problem solver from the strict requirements of the conditional sampling equation given by Song et al. [2020], while still leveraging the exceptionally powerful prior of the unconditional diffusion process. Moreover, we find that the desired score function directly arises as the Jacobian of the value function. 

We summarize our contributions as follows: 

• We present diffusion optimal control, a framework for solving inverse problems via the lens of optimal control theory, using pretrained unconditional off-the-shelf diffusion models. 

• We show that this perspective overcomes many core obstacles present in existing diffusionbased inverse problem solvers. In particular, the idealized posterior sampling score Song et al. [2021] — approximated by existing methods — can be recovered exactly as a specific case of our method. 

• We showcase the advantages of our model empirically with quantitative experiments and qualitative examples, and demonstrate state-of-the-art performance on the FFHQ $2 5 6 \times 2 5 6$ dataset. 

## 2 Background

Notation We use lowercase letters for denoting scalars $a \in \mathbb { R }$ , lowercase bold letters for vectors $\mathbf { a } \in \mathbb { R } ^ { n }$ and uppercase bold letters for matrices $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ . Subscripts indicate Jacobians and Hessians of scalar functions, e.g. $l _ { \mathbf { x } } \in \mathbb { R } ^ { n }$ and $l _ { \mathbf { x } \mathbf { x } } \in \mathbb { R } ^ { n \times n }$ for $l ( \mathbf { x } ) \bar { \mathbf { \Psi } } : \mathbb { R } ^ { n } \to \mathbb { R }$ , respectively. We overload notation for time-dependent variables, where subscripts imply dependence rather than derivatives w.r.t. time, $\mathbf { e . g . , x } _ { t } = \mathbf { x } ( t )$ . Furthermore, $V ( \mathbf { x } _ { t } )$ and $Q ( \mathbf { x } _ { t } , \mathbf { u } _ { t } )$ are scalar functions despite being uppercase, in line with existing optimal control literature Betts [1998]. 

## 2.1 Diffusion Models

The diffusion modeling literature uses the following reverse-time Itö SDE to generate samples Song et al. [2021], 

$$
\mathrm{d} \mathbf {x} _ {t} = \left[ \mathbf {f} (\mathbf {x} _ {t}) - g (t) ^ {2} \nabla_ {\mathbf {x} _ {t}} \log p _ {t} (\mathbf {x} _ {t}) \right] \mathrm{d} t + g (t) \mathrm{d} \mathbf {w} _ {t},\tag{1}
$$

where $\mathbf { x } _ { t } \in \mathbb { R } ^ { n }$ is the state vector, $\mathbf { f } : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ and $g : \mathbb { R }  \mathbb { I }$ <sup>R</sup> are drift and diffusion terms that can take different functional forms (e.g., Variance-Preserving SDEs (VPSDEs) and Variance-Exploding SDEs (VESDEs) in Song et al. [2021]), $\nabla _ { \mathbf { x } _ { t } }$ log $p _ { t } ( \mathbf { x } _ { t } )$ is the score-function and $\mathbf { w } _ { t } \in \mathbb { R } ^ { n }$ is a vector of mutually independent Brownian motions. The above SDE has an associated ODE called the 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/699aafed-e030-4fc3-a2c9-b280b0c85bb7/7cce5bf237e26c0ce65ee64e9b3d05258d6c1388157fb14bc7a87692b8707a60.jpg)



Figure 2: Predicted $\mathbf { x } _ { \mathrm { 0 } }$ used in a probabilistic framework (above) compared to ours (below) for a general diffusion trajectory. The full forward rollout in our proposed framework allows for the predicted $\mathbf { x } _ { \mathrm { 0 } }$ (and therefore $\nabla _ { \mathbf { x } _ { t } } \log p ( \mathbf { y } | \mathbf { x } _ { 0 } ) )$ to be efficiently computed for all $t = 0 , \ldots , T$


probability-flow (PF) ODE given by 

$$
\mathrm{d} \mathbf {x} _ {t} = \mathrm{d} \mathbf {x} _ {t} + \left[ \mathbf {f} (\mathbf {x} _ {t}) - \frac {1}{2} g (t) ^ {2} \nabla_ {\mathbf {x} _ {t}} \log p _ {t} (\mathbf {x} _ {t}) \right] \mathrm{d} t,\tag{2}
$$

with the same marginals $p _ { t } ( \mathbf { x } _ { t } )$ as the SDE, which allow for likelihood computation [Song et al., 2021, Li et al., 2024]. All practical implementations of diffusion samplers require a time-discretization of the PF-ODE. One such discretization is the well-known Euler-discretization which gives, 

$$
\mathbf {x} _ {t - 1} = \mathbf {x} _ {t} - [ \mathbf {f} (\mathbf {x} _ {t}) - \frac {1}{2} g (t) ^ {2} \nabla_ {\mathbf {x}} \log p _ {t} (\mathbf {x} _ {t}) ] \Delta t\tag{3}
$$

where, $\Delta t$ is the length of the discretization interval and we have reversed the time evolution by changing the sign of the drift. We are not restricted to only using the Euler-discretization and any high-order discretization techniques can also be employed. More concisely, we have, 

$$
\mathbf {x} _ {t - 1} = \mathbf {h} (\mathbf {x} _ {t}), \text { where } \mathbf {h}: \mathbb {R} ^ {n} \to \mathbb {R} ^ {n}\tag{4}
$$

which describes the general non-linear dynamics of the corresponding discrete-time diffusion sampler. 

## 2.2 Posterior Sampling for Inverse Problems

Inverse problems are a general class of problems where an unknown signal is reconstructed from observations obtained by a forward measurement process Ongie et al. [2020]. The forward process is usually lossy, resulting in an ill-posed signal recovery task where a unique solution does not exist. The forward model can generally be written as 

$$
y = \mathcal {A} (\mathbf {x} _ {0}) + \eta ,\tag{5}
$$

where $\mathcal { A } : \mathbb { R } ^ { n }  \mathbb { R } ^ { d }$ is the forward operator, $\boldsymbol { y } \in \mathbb { R } ^ { d }$ the measured signal, $\mathbf { x } _ { 0 } \in \mathbb { R } ^ { n }$ the unknown signal to be recovered, and $\boldsymbol { \eta } \sim \mathcal { N } ( \boldsymbol { 0 } , \bar { \sigma } \mathbf { I } _ { d } )$ the noise (with variance $\sigma ^ { 2 } )$ in the measurement process. 

Given the forward model Eq. (5) and a measurement y, sampling from the posterior distribution $p _ { \theta } ( \mathbf { x } | \mathbf { y } )$ can then be performed by solving the corresponding conditional Itö SDE 

$$
\mathrm{d} \mathbf {x} = [ \mathbf {f} (\mathbf {x}) - g (t) ^ {2} \nabla_ {\mathbf {x}} \log p _ {t} (\mathbf {x} | \mathbf {y}) ] \mathrm{d} t + g (t) \mathrm{d} \mathbf {w},\tag{6}
$$

where, invoking Bayes rule, 

$$
\nabla_ {\mathbf {x}} \log p _ {t} (\mathbf {x} | \mathbf {y}) = \nabla_ {\mathbf {x}} \log p _ {t} (\mathbf {x}) + \nabla_ {\mathbf {x}} \log p _ {t} (\mathbf {y} | \mathbf {x}).\tag{7}
$$

As with the unconditional dynamics, Eq. (6) has a corresponding ODE 

$$
\mathrm{d} \mathbf {x} = [ \mathbf {f} (\mathbf {x}) - \frac {1}{2} g (t) ^ {2} \nabla_ {\mathbf {x}} \log p _ {t} (\mathbf {x} | \mathbf {y}) ] \mathrm{d} t,\tag{8}
$$

which has an approximate solution obtained by the Euler discretization 

$$
\mathbf {x} _ {t - 1} = \mathbf {x} _ {t} + [ f (\mathbf {x} _ {t}) - \frac {1}{2} g (t) ^ {2} \nabla_ {\mathbf {x} _ {t}} \log p _ {t} (\mathbf {x} _ {t} | \mathbf {y}) ] \Delta t.\tag{9}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/699aafed-e030-4fc3-a2c9-b280b0c85bb7/4adab2dbbf2b2e8a5b80b61b54853b44b4e8361efcec6d6dbb8a875c4f976acf.jpg)



Figure 3: Inverse problem solution as a function of total diffusion timesteps T for the 4× super-resolution task. Compared to DPS (top row), our method (bottom row) produces solutions that are higher quality, in greater agreement with the inverse problem contraint $\mathcal { A } \mathbf { x } = \mathbf { y }$ , and more stable across T .


## 2.3 Optimal Control

Optimal control is the structured and principled approach to the guidance of dynamical systems over time. Many methods have been developed in the optimal control literature and are popularly referred to as trajectory optimization algorithms Betts [1998]. Perhaps the most well-known is the Iterative Linear Quadratic Regulator (iLQR) algorithm which uses a first-order approximation of the dynamics and second-order approximations of the value-function Li and Todorov [2004]. 

Formally, let us define an arbitrary user-defined global cost function 

$$
J _ {T} = \sum_ {t = T} ^ {1} \ell_ {t} (\mathbf {x} _ {t}, \mathbf {u} _ {t}) + \ell_ {0} (\mathbf {x} _ {0}),\tag{10}
$$

composed of a sum over scalar-valued running and terminal cost functions $\ell _ { t }$ and $\ell _ { 0 }$ . Optimal control theory dictates that the value function $\begin{array} { r } { V ( \mathbf { x } _ { t } , t ) : = \operatorname* { m i n } _ { \{ \mathbf { u } _ { n } \} _ { n = t } ^ { n = 1 } } J _ { t } } \end{array}$ satisfies the following recursive relation also known as Bellman’s Principle of Optimality 

$$
V (\mathbf {x} _ {t}, t) = \min _ {\mathbf {u} _ {t}} \Big [ \ell_ {t} (\mathbf {x} _ {t}, \mathbf {u} _ {t}) + V (\mathbf {x} _ {t - 1}, t - 1) \Big ].\tag{11}
$$

The iLQR algorithm centers around approximating the state-action value function, 

$$
Q (\mathbf {x} _ {t}, \mathbf {u} _ {t}) := \ell_ {t} (\mathbf {x} _ {t}, \mathbf {u} _ {t}) + V (\mathbf {x} _ {t - 1}, t - 1),\tag{12}
$$

from which the value function can be recovered as $V ( \mathbf { x } _ { t } , t ) = \operatorname* { m i n } _ { \mathbf { u } _ { t } } Q ( \mathbf { x } _ { t } , \mathbf { u } _ { t } )$ 

Then given a state transition function $\mathbf { x } _ { t } = \mathbf { h } ( \mathbf { x } _ { t + 1 } , \mathbf { u } _ { t + 1 } )$ where we crucially note that we have defined time to flow backwards from $t = T , \dots , 0$ , the iLQR algorithm has feedforward and feedback gains 

$$
\mathbf {k} = - Q _ {\mathbf {u u}} ^ {- 1} Q _ {\mathbf {u}} \quad \text {and} \quad \mathbf {K} = - Q _ {\mathbf {u u}} ^ {- 1} Q _ {\mathbf {u x}}\tag{13}
$$

The update equations can be written as 

$$
V _ {\mathbf {x}} = Q _ {\mathbf {x}} - \mathbf {K} ^ {T} Q _ {\mathbf {u u}} \mathbf {k} \qquad \mathrm{and} \qquad V _ {\mathbf {x x}} = Q _ {\mathbf {x x}} - \mathbf {K} ^ {T} Q _ {\mathbf {u u}} \mathbf {K}.\tag{14}
$$

Given the feedforward and feedback gains $\{ ( \mathbf { K } _ { t } , \mathbf { k } _ { t } ) \} _ { t = 0 } ^ { T }$ and $\bar { \mathbf { x } } _ { 0 } : = \mathbf { x } _ { 0 }$ , we can recursively obtain the locally optimal control at time t as a function of the present states $\mathbf { x } _ { t }$ and controls $\mathbf { u } _ { t }$ as 

$$
\bar {\mathbf {x}} _ {t} = \mathbf {h} (\bar {\mathbf {x}} _ {t + 1}, \mathbf {u} _ {t + 1} ^ {*}),\tag{15}
$$

$$
\mathbf {u} _ {t} ^ {*} = \mathbf {u} _ {t} + \lambda \mathbf {k} + \mathbf {K} (\bar {\mathbf {x}} _ {t} - \mathbf {x} _ {t}).\tag{16}
$$

For a more detailed treatment of iLQR as well as a derivation of the equations, please see Appendix B. 

## 3 Diffusion Optimal Control

We motivate our framework by observing that the reverse diffusion process Eq. (1) is an uncontrolled non-linear dynamical system that evolves from some initial state (at time $t = T )$ to some terminal state (at time $t = 0 )$ . By injecting control vectors $\mathbf { u } _ { t }$ into this system we can influence its behavior and hence its terminal state (i.e., the generated data) to sample from a desired $p ( \mathbf { x } | \mathbf { y } )$ . There are two obvious ways to inject control into this process: 

```matlab
Algorithm 1 Diffusion Optimal Control
Input: λ, T, y, xT
Initialize ut, kt, Kt as 0 for t = 1 ... T, {xt}T t=0 as uncontrolled dynamics
for iter = 1 to num_iters do
    Vx, Vxx ← ∇x0 log p(y|x0), ∇2x0 log p(y|x0) ▷ Initialize derivatives of V(xt, t)
    for t = 1 to T do
    Compute kt, Kt, Vx, Vxx ▷ See Eqs. (13), (14)
    end for
    for t = T to 1 do
    xt-1 ← h(xt, λkt + Kt(xt - xt'))
    xt' ← xt ▷ Update xt-1 with new ut
    end for
end for 
```

1. In input perturbation control, we apply the $\mathbf { u } _ { t }$ before the diffusion step: 

$$
\mathbf {x} _ {t - 1} = (\mathbf {x} _ {t} + \mathbf {u} _ {t}) - \left[ \mathbf {f} (\mathbf {x} _ {t} + \mathbf {u} _ {t}) - \frac {1}{2} g (t) ^ {2} \nabla_ {\mathbf {x}} \log p _ {t} (\mathbf {x} _ {t} + \mathbf {u} _ {t}) \right] \Delta t.\tag{17}
$$

2. In output perturbation control, $\mathbf { u } _ { t }$ is applied $a f t e r$ the diffusion step: 

$$
\mathbf {x} _ {t - 1} = \mathbf {x} _ {t} - \left[ \mathbf {f} (\mathbf {x} _ {t}) - \frac {1}{2} g (t) ^ {2} \nabla_ {\mathbf {x}} \log p _ {t} (\mathbf {x} _ {t}) \right] \Delta t + \mathbf {u} _ {t}.\tag{18}
$$

Observe that iLQR is formulated for general discrete-time dynamic processes. When applied specifically to the reverse diffusion dynamics of diffusion models, we are able to make several simplifications. First, we assume that we do not have access to any guidance except at time $t = 0 -$ $\mathrm { i . e . , } \ell _ { t } ( \mathbf x _ { t } , \mathbf u _ { t } )$ does not depend on $\mathbf { x } _ { t }$ 

In the case of input perturbation control, we observe from $\operatorname { E q }$ . (17) that $\mathbf { h } _ { \mathbf { x } } = \mathbf { h } _ { \mathbf { u } }$ , whereas output perturbation control implies that $\mathbf { h } _ { \mathbf { u } } = \mathbf { I } .$ , resulting in the left and right equations, respectively: 

$$
Q _ {\mathbf {x}} = \mathbf {h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x}} ^ {\prime}
$$

$$
Q _ {\mathbf {x}} = \mathbf {h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x}} ^ {\prime}\tag{19}
$$

$$
Q _ {\mathbf {u}} = \ell_ {\mathbf {u}} + \mathbf {h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x}} ^ {\prime}
$$

$$
Q _ {\mathbf {u}} = \ell_ {\mathbf {u}} + V _ {\mathbf {x}} ^ {\prime}\tag{20}
$$

$$
Q _ {\mathbf {x x}} = \mathbf {h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x x}} ^ {\prime} \mathbf {h} _ {\mathbf {x}}
$$

$$
Q _ {\mathbf {x x}} = \mathbf {h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x x}} ^ {\prime} \mathbf {h} _ {\mathbf {x}}\tag{21}
$$

$$
Q _ {\mathbf {u x}} = Q _ {\mathbf {x u}} = \mathbf {h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x x}} ^ {\prime} \mathbf {h} _ {\mathbf {x}}
$$

$$
Q _ {\mathbf {u x}} = Q _ {\mathbf {x u}} ^ {T} = V _ {\mathbf {x x}} ^ {\prime} \mathbf {h} _ {\mathbf {x}}\tag{22}
$$

$$
Q _ {\mathbf {u u}} = \ell_ {\mathbf {u u}} + \mathbf {h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x x}} ^ {\prime} \mathbf {h} _ {\mathbf {x}}
$$

$$
Q _ {\mathbf {u u}} = \ell_ {\mathbf {u u}} + V _ {\mathbf {x x}} ^ {\prime}.\tag{23}
$$

The derivatives of V can then be backpropagated using the following equations: 

$$
V _ {\mathbf {x}} = Q _ {\mathbf {x}} - \mathbf {K} ^ {T} Q _ {\mathbf {u u}} \mathbf {k} = Q _ {\mathbf {x x}} - \mathbf {K} ^ {T} Q _ {\mathbf {u u}} \mathbf {K}
$$

$$
= Q _ {\mathbf {x}} + Q _ {\mathbf {u x}} ^ {T} Q _ {\mathbf {u u}} ^ {- 1} Q _ {\mathbf {u}}\tag{24}
$$

$$
\begin{array}{r} V _ {\mathbf {x x}} = Q _ {\mathbf {x x}} - \mathbf {K} ^ {T} Q _ {\mathbf {u u}} \mathbf {K} \\ = Q _ {\mathbf {x x}} - Q _ {\mathbf {u x}} ^ {T} Q _ {\mathbf {u u}} ^ {- 1} Q _ {\mathbf {u x}}. \end{array}\tag{25}
$$

In high dimensional systems such as Eq. 3, matrices may be singular. Therefore, a Tikhonov regularized variant of iLQR is often employed, where matrix inverses are regularized by a diagonal matrix αI Tassa et al. [2014]. 

## 3.1 High Dimensional Contro

Compared to the dynamics in traditional application areas of optimal control, those we consider in Eqs. (17- 18) are much higher dimensional in the state x and control u variates. Therefore, iLQR faces several unique computational bottlenecks when applied to such control problems. 

In particular, the Jacobian matrices $\mathbf { h } _ { \mathbf { x } } , \mathbf { h } _ { \mathbf { u } }$ and the second-order derivative matrices $V _ { \mathbf { x } \mathbf { x } } , Q _ { \mathbf { x } \mathbf { x } } , Q _ { \mathbf { u } \mathbf { x } } , Q _ { \mathbf { x } \mathbf { u } }$ , and $Q _ { \mathbf { u u } }$ are particularly expensive to compute, store, and perform downstream operations against. For example, in a three-channel $2 5 6 \times 2 5 6$ image, these matrices naively contain $( 2 5 6 \times 2 5 6 \times 3 ) ^ { 2 } \approx 3 9 B$ parameters. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/699aafed-e030-4fc3-a2c9-b280b0c85bb7/04e75d318b9cb8b3691de8829f969b3565fbbde4999c0b67f101fa76e1896ad2.jpg)



Figure 4: Examples from inverse problem tasks on FFHQ $2 5 6 \times 2 5 6$ . From left to right each column contains ground truth, measurement, Diffusion Posterior Sampling (DPS), and ours.


In Appendix D.1 we propose and analyze three modifications to the standard iLQR algorithm: randomized low rank approximations, matrix-free evaluations, and action updates via an adaptive optimizer, that significantly reduce runtime and memory constraints while introducing minimal deterioration to performance on inverse problem solving tasks. 

## 4 Improved Posterior Sampling

We demonstrate that our optimal control-based sampler overcomes several practical obstacles that plague existing diffusion-based methods for inverse problem solvers. 

Brittleness to Discretization In a probabilistic framework, solutions to inverse problems incur a discretization error from the numerical solution of Eq. (8) that decays poorly with the total diffusion steps T of the diffusion process. While much research has been conducted on the acceleration of unconditional diffusion processes Song et al. [2020], Jolicoeur-Martineau et al. [2021], Karras et al. [2022], Meng et al. [2023], sample quality appears to decay much more aggressively in diffusion-based inverse problem solvers (Figure 3). 

We theorize that this is due to two reasons: 1) the posterior sampler Eq. (9) is only correct in the limit of infinitely small time steps, and 2) the quality of the approximated conditional score term $\nabla _ { \mathbf x } \log p ( \mathbf y | \mathbf x _ { t } )$ decays quickly with time (Figure 2), and so fewer timesteps lead to fewer chances at low t to correct errors made at high t. On the other hand, since optimal control directly casts the discretized process as an end-to-end control episode, it produces a feasible solution for any number of discretization steps T . 

Intractability of $\nabla _ { \mathbf { x } _ { t } } \log p ( \mathbf { y } | \mathbf { x } _ { t } )$ When the forward model A is known and η comes from a simple distribution, the conditional likelihood $p ( \mathbf { y } \vert \mathbf { x } _ { t } )$ can be derived in closed form for $t = 0$ . On the other hand, the dependence of y on x<sub>t</sub> for $t > 0$ is generally not known without explicitly computing x<sub>0</sub>, which requires sampling from the diffusion process. Ultimately, obtaining the conditional score term $\nabla _ { \mathbf { x } _ { t } } \log p ( \mathbf { y } | \mathbf { x } _ { t } )$ is a highly nontrivial task Song et al. [2021]. 

To sidestep this issue, many works Meng and Kabashima [2022], Song et al. [2022], Chung et al. [2023a] factorize this term as the integral 

$$
p (\mathbf {y} | \mathbf {x} _ {t}) = \int p (\mathbf {y} | \mathbf {x} _ {0}) p (\mathbf {x} _ {0} | \mathbf {x} _ {t}) d \mathbf {x} _ {0}\tag{26}
$$

and then apply a series of approximations to recover a computationally feasible estimate of the conditional score. First, the marginal $p ( \mathbf { x } _ { 0 } | \mathbf { x } _ { t } )$ is replaced by the marginal conditioned on $\mathbf { x } _ { 0 } .$ , i.e. 

$p ( \mathbf { x } _ { 0 } | \mathbf { x } _ { t } , \mathbf { x } _ { 0 } ) = \mathcal { N } ( \mathbf { x } _ { 0 } , \sigma ^ { 2 } \mathbf { I } )$ Kim and Ye [2021]. Next, the x -centered marginal is replaced by the posterior mean $\mathbb { E } [ { \bf x } _ { 0 } | { \bf x } _ { t } ]$ given by Tweedie’s formula Efron [2011]. Finally, the true score is replaced by the learned score network. 

While these approximations are necessary in a probabilistic framework, we show that they are not required in our method. Intuitively, this is because the linear quadratic regulator backpropagates the control cost log $p ( \mathbf { y } \vert \mathbf { x } )$ through a forward trajectory rollout, which naturally computes the true conditional score at each time t. Moreover, our model always estimates $\mathbf { x } _ { 0 } | \mathbf { x } _ { t }$ exactly (up to the discretization error induced by solving Eq. 3), rather than forming an approximation $\hat { \mathbf { x } } _ { 0 } \approx \mathbf { x } _ { 0 }$ (Figure 2). We formalize this observation with the following statement. 

Theorem 4.1. Let $E q .$ . 3 be the discretized sampling equation for the diffusion model with output perturbation mode control (Eq. 18). Moreover, let the terminal cost 

$$
\ell_ {0} (\mathbf {x} _ {0}) = - \log p (\mathbf {y} | \mathbf {x} _ {0})\tag{27}
$$

be twice-differentiable and the running costs 

$$
\ell_ {t} (\mathbf {x} _ {t}, \mathbf {u} _ {t}) = 0.\tag{28}
$$

Then the iterative linear quadratic regulator with Tikhonov regularizer α produces the control 

$$
\mathbf {u} _ {t} = \alpha \nabla_ {\mathbf {x} _ {t}} \log p (\mathbf {y} | \mathbf {x} _ {0}).\tag{29}
$$

In other words, by framing the inverse problem as an unconditional diffusion process with controls u , our proposed method produces controls that coincide precisely with the desired conditional scores $\nabla _ { \mathbf x _ { t } } \log p ( \mathbf y | \mathbf x _ { 0 } )$ . 

Let us further assume that log $p ( \mathbf { y } | \mathbf { x } _ { t } ) = \log p ( \mathbf { y } | \mathbf { x } _ { 0 } )$ ), i.e., $\mathbf { x } _ { t }$ contains no additional information about y than $\mathbf { x } _ { 0 } .$ . This assumption results in the posterior mean approximation in Chung et al. [2023a] under stochastic dynamics $( \mathrm { E q . ~ } 1 ) .$ , where we additionally obtain exact computation of $\mathbf { x } _ { 0 } ,$ rather than $\hat { \mathbf { x } } _ { 0 } \approx \mathbf { x } _ { 0 }$ via Tweedie’s formula Kim and Ye [2021]. Under the deterministic ODE dynamics (Eq. 2), we recover the true posterior sampler under appropriate choice of Tikhonov regularization constant α. 

Lemma 4.2. Under the deterministic sampler with output perturbation mode control, $\begin{array} { r } { \alpha = \frac { 1 } { g ( t ) ^ { 2 } \Delta t } } \end{array}$ recovers posterior sampling $( E q . ~ 9 )$ . 

We demonstrate a similar result with input mode perturbation. 

Theorem 4.3. Let $E q .$ 3 be the discretized sampling equation for the diffusion model with input perturbation mode control (Eq. 17). Moreover, let 

$$
\ell_ {0} (\mathbf {x} _ {0}) = \log p (\mathbf {y} | \mathbf {x} _ {0}),\tag{30}
$$

and the running costs 

$$
\ell_ {t} (\mathbf {x} _ {t}, \mathbf {u} _ {t}) = 0.\tag{31}
$$

Then the iterative linear quadratic regulator with Tikhonov regularizer $\begin{array} { r } { \alpha = \frac { 1 } { g ( t ) ^ { 2 } \Delta t } } \end{array}$ produces the dynamical sytem 

$$
\begin{array}{r} \widetilde {\mathbf {x}} _ {t} = \widetilde {\mathbf {x}} _ {t} + [ f (\widetilde {\mathbf {x}} _ {t}) - \frac {1}{2} g (t) ^ {2} (\nabla_ {\mathbf {x}} \log p _ {t} (\widetilde {\mathbf {x}} _ {t}) \\ + \nabla_ {\mathbf {x}} \log p _ {t} (\mathbf {y} | \mathbf {x} _ {t})) ] \Delta t, \end{array}\tag{32}
$$

where $\widetilde { \mathbf { x } } _ { t } : = \mathbf { x } _ { t } + \mathbf { u } _ { t }$ 

Observe that Eq. (32) can be understood as a predictor-corrector sampling method, where the predictor produces an unconditional reverse diffusion update and the corrector produces a conditional correction step on the intermediary variable ${ \bf x } _ { t } = \tilde { { \bf x } } _ { t } - { \bf u } _ { t }$ 

Ultimately, these results demonstrate that our proposed method is able to recover the idealized sampling procedure under mild assumptions on the diffusion optimal control algorithm. 

<table><tr><td rowspan="2"></td><td colspan="2">SR ×4</td><td colspan="2">Random Inpainting</td><td colspan="2">Box Inpainting</td><td colspan="2">Gaussian Deblurring</td><td colspan="2">Motion Deblurring</td></tr><tr><td>FID ↓</td><td>LPIPS ↓</td><td>FID ↓</td><td>LPIPS ↓</td><td>FID ↓</td><td>LPIPS ↓</td><td>FID ↓</td><td>LPIPS ↓</td><td>FID ↓</td><td>LPIPS ↓</td></tr><tr><td>Ours (NFE = 2500)</td><td>32.47</td><td>0.171</td><td>15.93</td><td>0.053</td><td>20.22</td><td>0.122</td><td>31.80</td><td>0.189</td><td>39.40</td><td>0.217</td></tr><tr><td>Ours (NFE = 1000)</td><td>37.53</td><td>0.189</td><td>20.75</td><td>0.108</td><td>23.88</td><td>0.164</td><td>35.24</td><td>0.191</td><td>45.99</td><td>0.233</td></tr><tr><td>PSLD (NFE = 1000)</td><td>34.28</td><td>0.201</td><td>21.34</td><td>0.096</td><td>43.11</td><td>0.167</td><td>41.53</td><td>0.221</td><td>-</td><td>-</td></tr><tr><td>Flash-Diffusion* (NFE = varies)</td><td>-</td><td>-</td><td>53.95</td><td>0.195</td><td>-</td><td>-</td><td>65.35</td><td>0.280</td><td>64.57</td><td>0.267</td></tr><tr><td>DDNM (NFE = 1000)</td><td>68.94</td><td>0.328</td><td>105.3</td><td>0.802</td><td>72.28</td><td>0.483</td><td>126.0</td><td>0.995</td><td>-</td><td>-</td></tr><tr><td>DPS (NFE = 1000)</td><td>39.35</td><td>0.214</td><td>33.12</td><td>0.168</td><td>21.19</td><td>0.212</td><td>44.05</td><td>0.257</td><td>39.92</td><td>0.242</td></tr><tr><td>DDRM (NFE = 1000)</td><td>62.15</td><td>0.294</td><td>42.93</td><td>0.204</td><td>69.71</td><td>0.587</td><td>74.92</td><td>0.332</td><td>-</td><td>-</td></tr><tr><td>MCG (NFE = 1000)</td><td>87.64</td><td>0.520</td><td>40.11</td><td>0.309</td><td>29.26</td><td>0.286</td><td>101.2</td><td>0.340</td><td>310.5</td><td>0.702</td></tr><tr><td>PNP-ADMM</td><td>66.52</td><td>0.353</td><td>151.9</td><td>0.406</td><td>123.6</td><td>0.692</td><td>90.42</td><td>0.441</td><td>89.08</td><td>0.405</td></tr><tr><td>Score-SDE (NFE = 1000)</td><td>96.72</td><td>0.563</td><td>60.06</td><td>0.331</td><td>76.54</td><td>0.612</td><td>109.0</td><td>0.403</td><td>292.2</td><td>0.657</td></tr><tr><td>ADMM-TV</td><td>110.6</td><td>0.428</td><td>68.94</td><td>0.322</td><td>181.5</td><td>0.463</td><td>186.7</td><td>0.507</td><td>152.3</td><td>0.508</td></tr></table>


Table 1: Quantitative evaluation (FID, LPIPS) of model performance on inverse problems on the FFHQ 256x256-1K dataset.


Dependence on the Approximate Score While our theoretical results require that the learned score function $s _ { \theta } ( \mathbf { x } _ { t } , t )$ approximates the true data score log $p _ { t } ( \mathbf { x } _ { t } , t )$ , we emphasize that the performance of our method does not necessitate this condition. In fact, we find that reconstruction performance is theoretically and empirically robust to the accuracy of the approximated prior score $s _ { \theta } ( \mathbf { x } _ { t } , t )$ ≈ $\nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } )$ or conditional score $\nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { y } | \mathbf { x } _ { 0 } ) \approx \nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { y } | \mathbf { x } _ { t } )$ terms. This is because the optimal control-based solution is formulated for the optimization of generalized dynamical systems, and thus agnostic to the diffusion sampling process. 

Certainly, improved approximation of the score terms result in a better-informed prior and usually higher sample quality. However, we demonstrate that our sampler produces remarkably reasonable solutions even in the case of randomly initialized diffusion models. Conversely, probabilistic posterior samplers can only sample from $p ( \mathbf { y } \vert \mathbf { x } _ { 0 } )$ when the terms composing the posterior sampling equation (Eq. (8)) are well approximated (Figure 6). Modeling errors can occur even in foundation models. For example, this scenario may arise in models trained on regions where there are underrepresented examples in the data. When these arise from existing social or ethical biases, they can further perpetuate or amplify biases to the resulting model if left unaddressedBolukbasi et al. [2016], Birhane et al. [2021], Srivastava et al. [2022]. 

There exist several methods that seek to alleviate the errors incurred by Tweedie’s formula (being a mean approximation of the diffusion process), including Song et al. [2024] which imposes a hard data consistency optimization loop at various points in the diffusion process, and Rout et al. [2023] which includes a stochastic averaging loop in each step of the diffusion process. However, these methods still rely on Tweedie’s formula for the error reduction scheme, which assumes access to a ground truth score function. Ultimately, the aforementioned problems in the present section are exacerbated in existing samplers, and relatively less consequential in our solver. 

## 5 Related Work

The recent success of diffusion models in image generation Song and Ermon [2019], Ho et al. [2020], Song et al. [2021], Rombach et al. [2022] has spawned a surge of research in deep learning-based solvers to inverse problems. Song et al. [2021] demonstrated a strategy for provably sampling from the solution set $p ( \mathbf { x } | \mathbf { y } )$ of a general inverse problem $\mathbf { y } = { \mathcal { A } } ( \mathbf { x } )$ using only an unconditional prior score model $\nabla _ { \mathbf { x } } \log p _ { t } ( \mathbf { x } )$ and a forward probabilistic model log $p ( \mathbf { y } \vert \mathbf { x } _ { t } )$ . However, a crucial problem arises in the intractability of forward probabilistic model, which depends on the noisy $\mathbf { x } _ { t }$ rather than the final $\mathbf { x } _ { \mathrm { 0 } }$ . This has resulted in a series of approximation algorithms Choi et al. [2021], Kawar et al. [2022], Chung et al. [2022, 2023a,b], Kawar et al. [2023] for the true conditional diffusion dynamics. 

Topics in control theory have been applied to deep learning Liu et al. [2020], Pereira et al. [2020] as well as diffusion modeling Berner et al. [2022]. Optimal control can also be connected to diffusion processes via forward-backward SDEs Chen et al. [2021]. However, these ideas have not been applied to guided conditional diffusion processes solely at inference time, nor for guided conditional sampling. Our proposed optimal control-based algorithm is, to our knowledge, the first such framework for deep inverse problem solvers. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/699aafed-e030-4fc3-a2c9-b280b0c85bb7/21515e8f421c719ba175e2ad36ee84db8b9a638a490e013049388f8b63b3c200.jpg)



Figure 5: Examples from the classconditional inverse problem. DPS (left) is compared against ours (right). Each row is a different target MNIST class.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/699aafed-e030-4fc3-a2c9-b280b0c85bb7/242bc861320b4b1b0a77ad2a1cb2f8a1ab89f95544791a6dac1f50b5ac4fc682.jpg)



Figure 6: Robustness to approximation quality of the score function. We consider the 4× super-resolution task with a randomly initialized diffusion model. Since the reverse diffusion process is no longer well approximated, DPS cannot produce a feasible solution, while our method still can.


## 6 Experiments

Following previous work Chung et al. [2023a], Meng and Kabashima [2022], Kawar et al. [2022], we consider five inverse problems. 1) In 4× image super-resolution, we use the bicubic downsampling operator. 2) In randomized inpainting, we uniformly omit 92% of all pixels (across all channels). 3) In box inpainting, we mask out a 128 × 128 block uniformly sampled from a 16 pixel margin from each side of the image, as in Chung et al. [2022]. 4) In Gaussian deblurring, we use a kernel of size 61 × 61 and standard deviation 3.0. In motion deblurring, we generate images according to a library<sup>2</sup> of point spread functions with kernel size 61 × 61 and intensity 0.5. Following the experimental design in Chung et al. [2023a], we apply Gaussian noise with standard deviation 0.05 to all measurements of the forward model. 

We compare against a generalized diffusion inverse sampler (Score-SDE) proposed in Song et al. [2021], Diffusion Posterior Sampling (DPS) Chung et al. [2023a], Denoising Diffusion Restoration Models Kawar et al. [2022], Manifold Constrained Gradients (MCG) Chung et al. [2022], as well as two recent latent diffusion-based methods Fabian et al. [2023] (Flash-Diffusion<sup>3</sup>) and Rout et al. [2024] (PSLD). For non-diffusion baselines, we compare against Plug-and-Play Alternating Direction Method of Multipliers (PnP-ADMM) with neural proximal maps Chan et al. [2016], Zhang et al. [2017], and a total-variation based alternating direction method of multipliers (TV-ADMM) baseline proposed in Chung et al. [2023a]. 

We validate our results on the high resolution human face dataset FFHQ 256 × 256 Karras et al. [2019]. Several methods are model agnostic (DPS, DDRM, MCG, and thus evaluated with the same pre-trained diffusion models. To fairly compare between all models, all methods use the model weights from Chung et al. [2023a], which are trained on 49K FFHQ images, with 1K images left as a held-out set for evaluation. We compare our algorithm against competing frameworks on these last 1K images. We report our results on FFHQ 256 × 256 in Table 1, and demonstrate improvements on all tasks against previous methods. Finally, we demonstrate the performance of our algorithm on the nonlinear inverse problem of class-conditional generation. Namely, let A(x) = classifier(x) and p(y|x) be its associated probability. We compare our method to DPS on the inverse task of generating an MNIST digit given a label y. Compared to images generated by DPS, images from our method exhibit more pronounced class alignment and higher overall sample quality (Figure 5). 

## 7 Conclusion

In this paper we presented a novel perspective on tackling inverse problems with diffusion models – framing the discretized reverse diffusion process as a discrete time optimal control episode. We demonstrate that this framework alleviates several core problems in probabilistic solvers: its dependence on the approximation quality of the underlying terms in the diffusion process, its sensitivity to the temporal discretization scheme, its inherent inaccuracy due to the intractability of the conditional score function. We also show that the diffusion posterior sampler can be seen as a specific case of our optimal control-based sampler. Finally, leveraging the improvements granted by our solver, we validate the performance of our algorithm on several inverse problem tasks across several datasets, and demonstrate highly competitive results. 

## References



Julius Berner, Lorenz Richter, and Karen Ullrich. An optimal control perspective on diffusion-based generative modeling. arXiv preprint arXiv:2211.01364, 2022. 





John T Betts. Survey of numerical methods for trajectory optimization. Journal of guidance, control, and dynamics, 21(2):193–207, 1998. 





Abeba Birhane, Vinay Uday Prabhu, and Emmanuel Kahembwe. Multimodal datasets: misogyny, pornography, and malignant stereotypes. arXiv preprint arXiv:2110.01963, 2021. 





Tolga Bolukbasi, Kai-Wei Chang, James Y Zou, Venkatesh Saligrama, and Adam T Kalai. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. Advances in neural information processing systems, 29, 2016. 





Stanley H Chan, Xiran Wang, and Omar A Elgendy. Plug-and-play admm for image restoration: Fixed-point convergence and applications. IEEE Transactions on Computational Imaging, 3(1): 84–98, 2016. 





Tianrong Chen, Guan-Horng Liu, and Evangelos A Theodorou. Likelihood training of schr\" odinger bridge using forward-backward sdes theory. arXiv preprint arXiv:2110.11291, 2021. 





Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: Condi tioning method for denoising diffusion probabilistic models. arXiv preprint arXiv:2108.02938, 2021. 





Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. Advances in Neural Information Processing Systems, 35:25683–25696, 2022. 





Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. International Conference on Learning Representations, 2023a. 





Hyungjin Chung, Jeongsol Kim, and Jong Chul Ye. Direct diffusion bridge using data consistency for inverse problems. arXiv preprint arXiv:2305.19809, 2023b. 





Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 





Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 106(496):1602–1614, 2011. 





Zalan Fabian, Berk Tinaz, and Mahdi Soltanolkotabi. Adapt and diffuse: Sample-adaptive reconstruction via latent diffusion models. arXiv preprint arXiv:2309.06642, 2023. 





Nathan Halko, Per-Gunnar Martinsson, and Joel A Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review, 53(2): 217–288, 2011. 





Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 





Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 





Matthew D Houghton, Alexander B Oshin, Michael J Acheson, Evangelos A Theodorou, and Irene M Gregory. Path planning: Differential dynamic programming and model predictive path integral control on vtol aircraft. In AIAA SCITECH 2022 Forum, page 0624, 2022. 





David H Jacobson. New second-order and first-order algorithms for determining optimal control: A differential dynamic programming approach. Journal of Optimization Theory and Applications, 2: 411–440, 1968. 





Alexia Jolicoeur-Martineau, Ke Li, Rémi Piché-Taillefer, Tal Kachman, and Ioannis Mitliagkas. Gotta go fast when generating data with score-based models. arXiv preprint arXiv:2105.14080, 2021. 





Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 





Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022. 





Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. Advances in Neural Information Processing Systems, 35:23593–23606, 2022. 





Bahjat Kawar, Noam Elata, Tomer Michaeli, and Michael Elad. Gsure-based diffusion model training with corrupted data. arXiv preprint arXiv:2305.13128, 2023. 





Kwanyoung Kim and Jong Chul Ye. Noise2score: tweedie’s approach to self-supervised image denoising without clean images. Advances in Neural Information Processing Systems, 34:864–874, 2021. 





Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 





Dana A Knoll and David E Keyes. Jacobian-free newton–krylov methods: a survey of approaches and applications. Journal of Computational Physics, 193(2):357–397, 2004. 





Henry Li, Ronen Basri, and Yuval Kluger. Likelihood training of cascaded diffusion models via hierarchical volume-preserving maps. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=sojpn00o8z. 





Weiwei Li and Emanuel Todorov. Iterative linear quadratic regulator design for nonlinear biological movement systems. In First International Conference on Informatics in Control, Automation and Robotics, volume 2, pages 222–229. SciTePress, 2004. 





Guan-Horng Liu, Tianrong Chen, and Evangelos A Theodorou. Ddpnopt: Differential dynamic programming neural optimizer. arXiv preprint arXiv:2002.08809, 2020. 





Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14297–14306, 2023. 





Xiangming Meng and Yoshiyuki Kabashima. Diffusion model based posterior sampling for noisy linear inverse problems. arXiv preprint arXiv:2211.12343, 2022. 





Gregory Ongie, Ajil Jalal, Christopher A Metzler, Richard G Baraniuk, Alexandros G Dimakis, and Rebecca Willett. Deep learning techniques for inverse problems in imaging. IEEE Journal on Selected Areas in Information Theory, 1(1):39–56, 2020. 





Samet Oymak, Zalan Fabian, Mingchen Li, and Mahdi Soltanolkotabi. Generalization guarantees for neural networks via harnessing the low-rank structure of the jacobian. arXiv preprint arXiv:1906.05392, 2019. 





Marcus Pereira, Ziyi Wang, Tianrong Chen, Emily Reed, and Evangelos Theodorou. Feynman-kac neural network architectures for stochastic control using second-order fbsde theory. In Learning for Dynamics and Control, pages 728–738. PMLR, 2020. 





Kaare Brandt Petersen, Michael Syskind Pedersen, et al. The matrix cookbook. Technical University of Denmark, 7(15):510, 2008. 





Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF confer ence on computer vision and pattern recognition, pages 10684–10695, 2022. 





Litu Rout, Yujia Chen, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Beyond first-order tweedie: Solving inverse problems using latent diffusion. arXiv preprint arXiv:2312.00852, 2023. 





Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alex Dimakis, and Sanjay Shakkottai. Solving linear inverse problems provably via posterior sampling with latent diffusion models. Advances in Neural Information Processing Systems, 36, 2024. 





Levent Sagun, Utku Evci, V Ugur Guney, Yann Dauphin, and Leon Bottou. Empirical analysis of the hessian of over-parametrized neural networks. arXiv preprint arXiv:1706.04454, 2017. 





Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 





Tomohiro Sasaki, Koki Ho, and E Glenn Lightsey. Nonlinear spacecraft formation flying using constrained differential dynamic programming. In Proceedings of AAS/AIAA Astrodynamics Specialist Conference, 2022. 





Bowen Song, Soo Min Kwon, Zecheng Zhang, Xinyu Hu, Qing Qu, and Liyue Shen. Solving inverse problems with latent diffusion models via hard data consistency. arXiv preprint arXiv:2307.08123, 2024. 





Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 





Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations, 2022. 





Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019. 





Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id= PxTIG12RRHS. 





Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022. 





Yuval Tassa, Tom Erez, and William Smart. Receding horizon differential dynamic programming. Advances in neural information processing systems, 20, 2007. 





Yuval Tassa, Nicolas Mansard, and Emo Todorov. Control-limited differential dynamic programming. In 2014 IEEE International Conference on Robotics and Automation (ICRA), pages 1168–1175. IEEE, 2014. 





Emanuel Todorov and Weiwei Li. A generalized iterative lqg method for locally-optimal feedback control of constrained nonlinear stochastic systems. In Proceedings of the 2005, American Control Conference, 2005., pages 300–306. IEEE, 2005. 





Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE transactions on image processing, 26(7): 3142–3155, 2017. 



## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? 

Answer: [Yes] 

Justification: We demonstrate our results through rigorous analysis of our algorithm and extensive experiments on multiple inverse problem settings over several datasets. 

Guidelines: 

• The answer NA means that the abstract and introduction do not include the claims made in the paper. 

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. 

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. 

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper. 

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? 

Answer: [Yes] 

Justification: Yes, the paper discusses the runtime cost of the work, and provides an equivalent budget analysis, where it still demonstrates competitive performance on each benchmark. 

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? 

Answer: [Yes] 

Justification: The paper provides full proofs for all theory in the appendix. 

## 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? 

Answer: [Yes] 

Justification: The paper discloses all hyperparameters and implementation details in the appendix. 

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? 

Answer: [Yes] 

Justification: The paper provides open access to the data, which is publicly available. The authors will release code upon acceptance. 

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? 

Answer: [Yes] 

Justification: The paper provides all details in the appendix. 

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? 

Answer: [NA] 

Justification: Experiments for other works do not provide error bars, therefore error bars would not benefit the analysis in this paper. 

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: Experiments can be run on any GPU A4000 or later. 

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? 

Answer: [Yes] 

Justification: We confirm to the NeurIPS Code of Ethics in every respect. 

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? 

Answer: [Yes] 

Justification: The paper discusses this in the appendix. 

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? 

Answer: [NA] 

Justification: The results in this paper paper do not have high risk for misuse. 

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? 

Answer: [Yes] 

Justification: We credit all creators and original owners of assets. 

## 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? 

Answer: [NA] 

Justification: No new assets are introduced. 

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? 

Answer: [NA] 

Justification: No research is performed with human subjects. 

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? 

Answer: [NA] 

Justification: No research is performed with human subjects. 

## A Impact Statement

This paper builds on a large body of existing work and presents an improved technique for solving generic nonlinear inverse problems, which can be seen as a generalization of guided diffusion modeling. Controlling the diffusion process in a generative model has many societal applications, and thus a broad range of downstream impacts. We believe that understanding the capabilities and limitations of such models in a public forum and open community is essential for practical and responsible integration of these technologies with society. However, the ideas presented in this work, as well as any other work in this field, must be deployed with caution to the inherent dangers of these technologies. 

## B Deriving the Iterative Linear Quadratic Regulator (iLQR)

Differential Dynamic Programming (DDP) is a very popular trajectory optimization algorithm that has a rich history of theoretical results Jacobson [1968] as well as successful practical applications in robotics Tassa et al. [2007, 2014], aerospace Houghton et al. [2022], Sasaki et al. [2022] and biomechanics Todorov and Li [2005]. It falls under the class of indirect methods for trajectory optimization, wherein Bellman’s principle of optimality defines the so-called optimal value-function which in turn can be used to determine the optimal control. This is in contrast to so-called direct methods which cast the problem at hand into a nonlinear constrained optimization problem. 

To formulate an optimal control algorithm we first define the state transition function of a dynamical system as 

$$
\mathbf {x} _ {t - 1} = \mathbf {h} (\mathbf {x} _ {t}, \mathbf {u} _ {t}).\tag{33}
$$

The next ingredient that we need for our optimal control approach is a cost function $J ( \mathbf { x } _ { t } , \mathbf { u } _ { t } ) \in \mathbb { R }$ This is used to define a performance criterion that iLQR can optimize with respect to the set of controls $\{ \mathbf { u } _ { t } \} _ { t = T } ^ { t = 1 } \ ( \mathrm { i . e }$ , the control trajectory going backwards from time $t = T \tan t = 1 )$ . The cost-function is defined as follows: 

$$
J _ {T} = \sum_ {t = T} ^ {1} \ell_ {t} (\mathbf {x} _ {t}, \mathbf {u} _ {t}) + \ell_ {0} (\mathbf {x} _ {0}),\tag{34}
$$

where, $\ell _ { t }$ and $\ell _ { 0 }$ are scalar-valued functions which are commonly referred to as the running costfunction and the terminal cost-function respectively. 

To obtain the sequence of optimal controls, we employ the dynamic programming principle. To do so, we first introduce the notion of the Value-function defined as follows: 

$$
V (\mathbf {x} _ {t}, t) = \min _ {\{\mathbf {u} _ {n} \} _ {n = t} ^ {n = 1}} J _ {t} = \min _ {\{\mathbf {u} _ {n} \} _ {n = t} ^ {n = 1}} \big [ \sum_ {n = t} ^ {1} \ell_ {n} (\mathbf {x} _ {n}, \mathbf {u} _ {n}) + \ell_ {0} (\mathbf {x} _ {0}) \big ]\tag{35}
$$

Intuitively, the Value-function resembles the optimal cost-to-go starting from time step t and state $\mathbf { x } _ { t }$ until the end of the time horizon $( \mathrm { i } . \mathsf { e } . , t = 0 )$ . Using this definition, one can easily derive the following recursive relation also known as Bellman’s Principle of Optimality: 

$$
V (\mathbf {x} _ {t}, t) = \min _ {\mathbf {u} _ {t}} \Big [ \ell_ {t} (\mathbf {x} _ {t}, \mathbf {u} _ {t}) + V (\mathbf {x} _ {t - 1}, t - 1) \Big ].\tag{36}
$$

A often useful defintion used in the derivation of the iLQR Riccati equations is that of the State-Action Value-Function $Q ( \mathbf { x } _ { t } , \mathbf { u } _ { t } )$ given by, 

$$
Q (\mathbf {x} _ {t}, \mathbf {u} _ {t}) = \ell_ {t} (\mathbf {x} _ {t}, \mathbf {u} _ {t}) + V (\mathbf {x} _ {t - 1}, t - 1)\tag{37}
$$

$$
\text { Therefore, } V (\mathbf {x} _ {t}, t) = \min _ {\mathbf {u} _ {t}} Q (\mathbf {x} _ {t}, \mathbf {u} _ {t})\tag{38}
$$

A sketch of the derivation of the Riccati equations is as follows: we take second-order Taylor expansions of both $Q ( \mathbf { x } _ { t } , \mathbf { u } _ { t } )$ and $V ( \mathbf { x } _ { t } , t )$ around nominal state and action trajectories of $\{ \bar { \mathbf { x } } _ { t } \} _ { t = T } ^ { t = 0 }$ and $\{ \bar { \mathbf { u } } _ { t } \} _ { t = T } ^ { t = 1 }$ respectively. Next, we substitute these into Eq.(37) and equate the first- and secondorder terms to yield the following relations between the derivatives of $Q .$ ℓ and $V { : }$ 

$$
Q _ {\mathbf {x}} = \ell_ {\mathbf {x}} + \mathbf {h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x}} ^ {\prime}\tag{39}
$$

$$
Q _ {\mathbf {u}} = \ell_ {\mathbf {u}} + \mathbf {h} _ {\mathbf {u}} ^ {T} V _ {\mathbf {x}} ^ {\prime}\tag{40}
$$

$$
Q _ {\mathbf {x x}} = \ell_ {\mathbf {x x}} + \mathbf {h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x x}} ^ {\prime} \mathbf {h} _ {\mathbf {x}}\tag{41}
$$

$$
Q _ {\mathbf {x u}} = \ell_ {\mathbf {x u}} + \mathbf {h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x x}} ^ {\prime} \mathbf {h} _ {\mathbf {u}}\tag{42}
$$

$$
Q _ {\mathbf {u x}} = \ell_ {\mathbf {u x}} + \mathbf {h} _ {\mathbf {u}} ^ {T} V _ {\mathbf {x x}} ^ {\prime} \mathbf {h} _ {\mathbf {x}}\tag{43}
$$

$$
Q _ {\mathbf {u u}} = \ell_ {\mathbf {u u}} + \mathbf {h} _ {\mathbf {u}} ^ {T} V _ {\mathbf {x x}} ^ {\prime} \mathbf {h} _ {\mathbf {u}},\tag{44}
$$

where $\mathbf { h } _ { \mathbf { x } _ { t } }$ and $\mathbf { h } _ { \mathbf { u } _ { t } }$ are the Jacobians of the dynamics function $\mathbf { h } ( \mathbf { x } _ { t } , \mathbf { u } _ { t } )$ , evaluated at time step $t ,$ w.r.t the state and the control vectors respectively. For ease of notation, we have dropped the subscript t and therefore all derivatives above should be considered to be evaluated at time step t, while we use $V _ { \mathbf { x } } ^ { \prime }$ and $V _ { \mathbf { x } \mathbf { x } } ^ { \prime }$ above to indicate the gradient and hessian of the Value-function evaluated at the next time step (i.e., at time step $t - 1 \AA ,$ ). 

Next, we substitute for the second-order approximation of $Q ( \mathbf { x } _ { t } , \mathbf { u } _ { t } )$ into $\operatorname { E q } .$ . (38) and note that $\mathbf { u } _ { t }$ can be written in terms of the nominal control as follows: 

$$
\mathbf {u} _ {t} = \bar {\mathbf {u}} _ {t} + \delta \mathbf {u} _ {t}.
$$

This results in a quadratic objective w.r.t $\delta { \mathbf { u } } _ { t }$ and the minimization in Eq. (38) can be performed exactly resulting in the following optimal perturbation from the nominal control trajectory: 

$$
\delta \mathbf {u} _ {t} ^ {*} = \mathbf {k} _ {t} + \mathbf {K} _ {t} \delta \mathbf {x} _ {t}\tag{45}
$$

where, the feedforward and feedback gains are given by the following expressions: 

$$
\mathbf {k} = - Q _ {\mathbf {u u}} ^ {- 1} Q _ {\mathbf {u}}\tag{46}
$$

$$
\mathbf {K} = - Q _ {\mathbf {u u}} ^ {- 1} Q _ {\mathbf {u x}}\tag{47}
$$

Finally, by substituting for the optimal $\delta \mathbf { u } _ { t } ^ { * }$ back into Eq.(38), we can drop the min operator and equate the first- and second-order terms on both sides. This results the following Riccati equations: 

$$
V _ {\mathbf {x}} = Q _ {\mathbf {x}} - \mathbf {K} ^ {T} Q _ {\mathbf {u u}} \mathbf {k}\tag{48}
$$

$$
V _ {\mathbf {x x}} = Q _ {\mathbf {x x}} - \mathbf {K} ^ {T} Q _ {\mathbf {u u}} \mathbf {K}.\tag{49}
$$

This concludes the sketch derivation of the Riccati equations. The algorithm roughly proceeds as follows: 

1. We start with an initial guess of the the nominal control trajectory $\{ \bar { \mathbf { u } } _ { t } \} _ { t = T } ^ { 1 }$ and generate the corresponding nominal state trajectory $\{ \bar { \mathbf { x } } _ { t } \} _ { t = T } ^ { 0 }$ using $\mathbf x _ { t } = \mathbf h ( \mathbf x _ { t + 1 } , \mathbf u _ { t + 1 } )$ . 

2. By noticing from Eq. (35) that $V ( \mathbf { x } _ { 0 } , 0 ) = \ell ( \mathbf { x } _ { 0 } )$ we can obtain expressions for $V _ { \mathbf { x } }$ and $V _ { \mathbf { x } \mathbf { x } }$ evaluated at $\bar { \bf x } _ { 0 }$ 

3. Next, we compute the derivatives of $Q$ given by equations. (39)-(44) using $\{ \bar { \mathbf { u } } _ { t } \} _ { t = T } ^ { 1 }$ and $\{ \bar { \mathbf { x } } _ { t } \} _ { t = T } ^ { 1 }$ 

4. Using the derivatives of $Q .$ , we can compute the feedforward and feedback gains using equations (46)-(47). 

5. Finally, using the Riccati equations (48)-(49), we can propagate both $V _ { \mathbf { x } }$ and $V _ { \mathbf { x } \mathbf { x } }$ one step backwards in time. 

6. We then repeat the steps 3, 4 and 5 until we backpropagate the derivatives of $V$ to time step $t = T$ 

7. This completes one iteration of iLQR. At the end of each iteration the gains are used to produce the updated nominal control trajectory as follows: 

$$
\bar {\mathbf {u}} _ {t} ^ {*} = \bar {\mathbf {u}} _ {t} + \alpha \mathbf {k} + \mathbf {K} (\bar {\mathbf {x}} _ {t} - \mathbf {x} _ {t})\tag{50}
$$

where, $\mathbf { x } _ { t }$ is the state obtained by unrolling the dynamics subject to the updated controls: 

$$
\mathbf {x} _ {t} = \mathbf {h} (\mathbf {x} _ {t + 1}, \bar {\mathbf {u}} _ {t + 1} ^ {*}).
$$

8. The new nominal control trajectory $\bar { \mathbf { u } } _ { t } ^ { * }$ is used to produce a new nominal state trajectory $\bar { \mathbf { x } } _ { t } ^ { * }$ and the algorithm is repeated from step 2 onwards until convergence or a fixed number of iterations. 

## C Proofs

Theorem 4.1. Let Eq. 3 be the discretized sampling equation for the diffusion model with output perturbation mode control (Eq. 18). Moreover, let the terminal cost 

$$
\ell_ {0} (\mathbf {x} _ {0}) = - \log p (\mathbf {y} | \mathbf {x} _ {0})\tag{27}
$$

be twice-differentiable and the running costs 

$$
\ell_ {t} (\mathbf {x} _ {t}, \mathbf {u} _ {t}) = 0.\tag{28}
$$

Then the iterative linear quadratic regulator with Tikhonov regularizer α produces the control 

$$
\mathbf {u} _ {t} = \alpha \nabla_ {\mathbf {x} _ {t}} \log p (\mathbf {y} | \mathbf {x} _ {0}).\tag{29}
$$

Proof. We demonstrate the result via induction for $t = 1 , \dots , T$ 

Since we assume that $\ell _ { \mathbf { u u } } = \mathbf { 0 } , V _ { \mathbf { x } \mathbf { x } }$ vanishes: 

$$
V _ {\mathbf {x x}} = Q _ {\mathbf {x x}} - Q _ {\mathbf {u x}} ^ {T} Q _ {\mathbf {u u}} ^ {- 1} Q _ {\mathbf {u x}}\tag{51}
$$

$$
= h _ {\mathbf {x}} ^ {T} V _ {\mathbf {x x}} ^ {\prime} h _ {\mathbf {x}} - h _ {\mathbf {x}} ^ {T} V _ {\mathbf {x x}} ^ {\prime} (V _ {\mathbf {x x}} ^ {\prime}) ^ {- 1} V _ {\mathbf {x x}} ^ {\prime} h _ {\mathbf {x}}\tag{52}
$$

(53) 

Similarly, $V _ { \mathbf { x } }$ also greatly simplifies as 

$$
V _ {\mathbf {x}} = Q _ {\mathbf {x}} + Q _ {\mathbf {u x}} ^ {T} Q _ {\mathbf {u u}} ^ {- 1} Q _ {\mathbf {u}}\tag{54}
$$

$$
= h _ {\mathbf {x}} ^ {T} V _ {\mathbf {x}} ^ {\prime} + h _ {\mathbf {x}} ^ {T} V _ {\mathbf {x x}} ^ {\prime} (V _ {\mathbf {x x}} ^ {\prime}) ^ {- 1} V _ {\mathbf {x}} ^ {\prime}\tag{55}
$$

$$
= h _ {\mathbf {x}} ^ {T} V _ {\mathbf {x}} ^ {\prime}.\tag{56}
$$

Turning to the Tikhonov regularized feedforward term, 

$$
\begin{array}{r l} & {\mathbf {k} = - Q _ {\mathbf {u u}} ^ {- 1} Q _ {\mathbf {u}}} \\ & {\quad = - (h _ {\mathbf {x}} ^ {T} \underbrace {V _ {\mathbf {x x}}} _ {\mathbf {0}} h _ {\mathbf {x}} + \alpha \mathbf {I}) ^ {- 1} Q _ {\mathbf {u}}} \end{array}\tag{57}
$$

(58) 

$$
= - (\mathbf {0} + \alpha \mathbf {I}) ^ {- 1} Q _ {\mathbf {u}}\tag{59}
$$

$$
= - \frac {1}{\alpha} V _ {\mathbf {x}} ^ {\prime}.\tag{60}
$$

Finally, the feedback term disappears due to the vanishing $V _ { \mathbf { x } \mathbf { x } }$ 

$$
\begin{array}{r l} & {\mathbf {K} = - Q _ {\mathbf {u u}} ^ {- 1} Q _ {\mathbf {u x}}} \\ & {\quad = \mathbf {0}.} \end{array}\tag{61}
$$

(62) 

Explicitly denoting the dependence of $V _ { \mathbf { x } }$ and $V _ { \mathbf { x } } ^ { \prime }$ on t, we can rewrite Eq. 56 as 

$$
\begin{array}{r} V _ {\mathbf {x}} ^ {(t)} = h _ {\mathbf {x}} ^ {T} V _ {\mathbf {x}} ^ {(t - 1)} \\ = \frac {\partial \mathbf {x} _ {t - 1}}{\partial \mathbf {x} _ {t}} \frac {\partial}{\partial \mathbf {x} _ {t - 1}} V. \end{array}
$$

Combining this observation with the fact that $\ell _ { 0 } = - \log p ( \mathbf { y } | \mathbf { x } _ { 0 } )$ , we can conclude that 

$$
V _ {\mathbf {x}} ^ {(t)} = - \nabla_ {\mathbf {x} _ {t}} \log p (\mathbf {y} | \mathbf {x} _ {0}),\tag{63}
$$

where $\mathbf { x } _ { \mathrm { 0 } }$ depends on $\mathbf { x } _ { t }$ via the state transition function h (Eq. 18). Therefore, we have that 

$$
\begin{array}{r l} & {\mathbf {k} = - \frac {1}{\alpha} V _ {\mathbf {x}} ^ {\prime}} \\ & {\quad = \frac {1}{\alpha} \nabla_ {\mathbf {x} _ {t}} \log p (\mathbf {y} | \mathbf {x} _ {0})} \\ & {\mathbf {K} = 0.} \end{array}
$$

Finally, given our action update (Eq. 15), we can conclude our desired result 

$$
\mathbf {u} _ {t} = \frac {1}{\alpha} \nabla_ {\mathbf {x} _ {t}} \log p (\mathbf {y} | \mathbf {x} _ {0}).\tag{64}
$$

Lemma C.1. Under the deterministic sampler with output perturbation mode control, $\begin{array} { r } { \alpha = \frac { 1 } { g ( t ) ^ { 2 } \Delta t } } \end{array}$ recovers posterior sampling $( E q . ~ 9 )$ . 

Proof. Substituting in $\begin{array} { r } { \alpha = \frac { 1 } { g ( t ) ^ { 2 } \Delta t } } \end{array}$ to Eq. 29, we observe that Eq. 18 can now be written as 

$$
\mathbf {x} _ {t - 1} = [ f (\mathbf {x} _ {t}) - \frac {1}{2} g (t) ^ {2} (\nabla_ {\mathbf {x} _ {t}} \log p _ {t} (\mathbf {x} _ {t}) + \nabla_ {\mathbf {x} _ {t}} \log p _ {t} (\mathbf {y} | \mathbf {x} _ {0})) ] \Delta t.\tag{65}
$$

Under the determinstic sampler, we can conclude that log ${ \bf \nabla } ; p _ { t } ( { \bf y } | { \bf x } _ { 0 } ) = \log p _ { t } ( { \bf y } | { \bf x } _ { t } )$ , since each $\mathbf { x } _ { t }$ has a unique path through the sample space. Therefore, we conclude that Eq. 65 resembles the ideal posterior sampler equation 9. We conclude our proof. □ 

Theorem 4.3. Let Eq. 3 be the discretized sampling equation for the diffusion model with input perturbation mode control (Eq. 17). Moreover, let 

$$
\ell_ {0} (\mathbf {x} _ {0}) = \log p (\mathbf {y} | \mathbf {x} _ {0}),\tag{30}
$$

and the running costs 

$$
\ell_ {t} (\mathbf {x} _ {t}, \mathbf {u} _ {t}) = 0.\tag{31}
$$

Then the iterative linear quadratic regulator with Tikhonov regularizer $\begin{array} { r } { \alpha = \frac { 1 } { g ( t ) ^ { 2 } \Delta t } } \end{array}$ produces the dynamical sytem 

$$
\begin{array}{r} \widetilde {\mathbf {x}} _ {t} = \widetilde {\mathbf {x}} _ {t} + [ f (\widetilde {\mathbf {x}} _ {t}) - \frac {1}{2} g (t) ^ {2} (\nabla_ {\mathbf {x}} \log p _ {t} (\widetilde {\mathbf {x}} _ {t}) \\ + \nabla_ {\mathbf {x}} \log p _ {t} (\mathbf {y} | \mathbf {x} _ {t})) ] \Delta t, \end{array}\tag{32}
$$

where $\widetilde { \mathbf { x } } _ { t } : = \mathbf { x } _ { t } + \mathbf { u } _ { t }$ 

Proof. We similarly demonstrate the result via induction for $t = 1 , \dots , T$ 

Again, assuming that $\ell _ { \mathbf { u u } } = 0 , V _ { \mathbf { x } \mathbf { x } }$ vanishes: 

$$
V _ {\mathbf {x x}} = Q _ {\mathbf {x x}} - Q _ {\mathbf {u x}} ^ {T} Q _ {\mathbf {u u}} ^ {- 1} Q _ {\mathbf {u x}}\tag{66}
$$

$$
= Q _ {\mathbf {x x}} - Q _ {\mathbf {x x}} (\underbrace {\ell_ {\mathbf {u u}}} _ {= 0} + Q _ {\mathbf {x x}}) ^ {- 1} Q _ {\mathbf {x x}}\tag{67}
$$

$$
= \mathbf {0},\tag{68}
$$

whereas $V _ { \mathbf { x } }$ greatly simplifies as 

$$
V _ {\mathbf {x}} = Q _ {\mathbf {x}} + Q _ {\mathbf {u x}} ^ {T} Q _ {\mathbf {u u}} ^ {- 1} Q _ {\mathbf {u}}
$$

$$
= h _ {\mathbf {x}} ^ {T} V _ {\mathbf {x}} ^ {\prime}.\tag{69}
$$

(70) 

Turning to the feedforward and feedback terms, we have 

$$
\mathbf {k} = - Q _ {\mathbf {u u}} ^ {- 1} Q _ {\mathbf {u}}\tag{71}
$$

$$
= - (h _ {\mathbf {x}} ^ {T} \underbrace {V _ {\mathbf {x x}}} _ {\mathbf {0}} h _ {\mathbf {x}} + \alpha \mathbf {I}) ^ {- 1} Q _ {\mathbf {u}}\tag{72}
$$

$$
= - (\mathbf {0} + \alpha \mathbf {I}) ^ {- 1} Q _ {\mathbf {u}}\tag{73}
$$

$$
= - \frac {1}{\alpha} h _ {\mathbf {x}} ^ {T} V _ {\mathbf {x}} ^ {\prime},\tag{74}
$$

and 

$$
\begin{array}{c} \mathbf {K} = - Q _ {\mathbf {u u}} ^ {- 1} Q _ {\mathbf {u x}} \\ = \mathbf {0}. \end{array}\tag{75}
$$

We observe that 

$$
V _ {\mathbf {x}} ^ {(t)} = - \frac {1}{\alpha} h _ {\mathbf {x}} ^ {T} V _ {\mathbf {x}} ^ {(t - 1)}.
$$

<table><tr><td></td><td>SR ×4</td><td>Random Inpainting</td><td>Box Inpainting</td><td>Gaussian Deblurring</td><td>Motion Deblurring</td></tr><tr><td>T</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td></tr><tr><td>num_iters</td><td>50</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>step_size</td><td>1e-3</td><td>1e-3</td><td>1e-3</td><td>1e-3</td><td>1e-3</td></tr><tr><td><eq>\ell_0(\mathbf{x}_0)</eq></td><td><eq>||\mathcal{A}(\mathbf{x}_0)-\mathbf{y}||</eq></td><td><eq>||\mathcal{A}(\mathbf{x}_0)-\mathbf{y}||</eq></td><td><eq>||\mathcal{A}(\mathbf{x}_0)-\mathbf{y}||</eq></td><td><eq>||\mathcal{A}(\mathbf{x}_0)-\mathbf{y}||</eq></td><td><eq>||\mathcal{A}(\mathbf{x}_0)-\mathbf{y} ||</eq></td></tr><tr><td><eq>\alpha</eq></td><td>1e-4</td><td>1e-4</td><td>1e-4</td><td>1e-4</td><td>1e-4</td></tr><tr><td><eq>\ell_t(\mathbf{x}_t,\mathbf{u}_t)</eq></td><td><eq>\alpha ||\mathbf{u}_t||</eq></td><td><eq>\alpha ||\mathbf{u}_t||</eq></td><td><eq>\alpha ||\mathbf{u}_t||</eq></td><td><eq>\alpha ||\mathbf{u}_t||</eq></td><td><eq>\alpha ||\mathbf{u}_t||</eq></td></tr><tr><td>k</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>control_mode</td><td>input mode</td><td>input mode</td><td>input mode</td><td>input mode</td><td>input mode</td></tr></table>


Table 2: Hyperparameters for FFHQ experiments.


Therefore, noting that $V _ { \mathbf { x } } ^ { ( 0 ) } = \log p ( \mathbf { y } | \mathbf { x } _ { 0 } )$ , we have 

$$
\begin{array}{r l} & {\mathbf {k} = - V _ {\mathbf {x}} ^ {(t)}} \\ & {\quad = - \frac {1}{\alpha} (h _ {\mathbf {x}} ^ {(t)}) ^ {T} V _ {\mathbf {x}} ^ {(t - 1)}} \\ & {\quad = - \frac {1}{\alpha} \nabla_ {\mathbf {x} _ {t}} \log p (\mathbf {y} | \mathbf {x} _ {0})} \\ & {\quad = - \frac {1}{\alpha} \nabla_ {\mathbf {x} _ {t}} \log p (\mathbf {y} | \mathbf {x} _ {0} (\mathbf {x} _ {t})).} \end{array}
$$

Applying the feedforward terms to the diffusion sampling process, we have 

$$
\begin{array}{r} \mathbf {x} _ {t - 1} = (\mathbf {x} _ {t} + \mathbf {u} _ {t}) + [ f (\mathbf {x} _ {t} + \mathbf {u} _ {t}) \\ - \frac {1}{2} g (t) ^ {2} \nabla_ {\mathbf {x}} \log p _ {t} (\mathbf {x} _ {t} + \mathbf {u} _ {t}) ] \Delta t. \end{array}
$$

We define the intermediary variable 

$$
\widetilde {\mathbf {x}} _ {t} = \mathbf {x} _ {t} + \mathbf {u} _ {t},\tag{77}
$$

which has dynamics 

$$
\widetilde {\mathbf {x}} _ {t} = \widetilde {\mathbf {x}} _ {t} + [ f (\widetilde {\mathbf {x}} _ {t}) - \frac {1}{2} g (t) ^ {2} \nabla_ {\mathbf {x}} \log p _ {t} (\widetilde {\mathbf {x}} _ {t}) ] \Delta t + \mathbf {u} _ {t}.\tag{78}
$$

We now can see that, letting $\alpha = \Delta t g ( t ) ^ { 2 }$ , we obtain 

$$
\widetilde {\mathbf {x}} _ {t} = \widetilde {\mathbf {x}} _ {t} + [ f (\widetilde {\mathbf {x}} _ {t}) - \frac {1}{2} g (t) ^ {2} (\nabla_ {\mathbf {x}} \log p _ {t} (\widetilde {\mathbf {x}} _ {t}) + \nabla_ {\mathbf {x}} \log p _ {t} (\mathbf {y} | \mathbf {x} _ {0})) ] \Delta t.
$$

## D Implementation

For all experiments, we use publicly available datasets and pre-trained model weights. For the FFHQ 256 × 256 experiments, we use the last 1K images of the dataset for evaluation. For MNIST, we do not use images directly in the inverse classification task. The images were only used for training the pretrained diffusion model. 

For models, we used the pretrained weights from Chung et al. [2023a] for FFHQ 256 × 256 tasks, and the Hugging Face 1aurent/mnist-28 diffusion model for MNIST experiments. No further training is performed on any models. Further hyperparameters can be found in Table 2. For the classifier $p ( \mathbf { y } \vert \mathbf { x } )$ in MNIST class-guided classification, we use a simple convolutional neural network with two convolutional layers and two MLP layers, trained on the entire MNIST dataset. 

## D.1 High Dimensional Control

To speed up our proposed method, we leverage the following three modifications to the standard iLQR algorithm. 

<table><tr><td rowspan="2"></td><td colspan="3">SR ×4</td><td colspan="3">Random Inpainting</td><td colspan="3">Box Inpainting</td><td colspan="3">Gaussian Deblurring</td><td colspan="3">Motion Deblurring</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>MSE ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>MSE ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>MSE ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>MSE ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>MSE ↓</td></tr><tr><td>Ours (T = 50)</td><td>27.45</td><td>0.792</td><td>117.0</td><td>31.84</td><td>0.882</td><td>42.57</td><td>25.33</td><td>0.804</td><td>190.6</td><td>24.99</td><td>0.694</td><td>206.1</td><td>25.08</td><td>0.721</td><td>201.9</td></tr><tr><td>DPS (T = 1000)</td><td>25.67</td><td>0.852</td><td>176.2</td><td>22.47</td><td>0.873</td><td>368.2</td><td>25.23</td><td>0.851</td><td>195.0</td><td>24.25</td><td>0.811</td><td>244.4</td><td>24.92</td><td>0.859</td><td>209.4</td></tr><tr><td>DDRM (T = 1000)</td><td>25.36</td><td>0.835</td><td>189.3</td><td>22.24</td><td>0.869</td><td>388.2</td><td>9.19</td><td>0.319</td><td>7835</td><td>23.36</td><td>0.767</td><td>300.0</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MCG (T = 1000)</td><td>20.05</td><td>0.559</td><td>642.8</td><td>19.97</td><td>0.703</td><td>654.8</td><td>21.57</td><td>0.751</td><td>453.0</td><td>6.72</td><td>0.051</td><td>13838</td><td>6.72</td><td>0.055</td><td>13838</td></tr><tr><td>PNP-ADMM</td><td>26.55</td><td>0.865</td><td>143.9</td><td>11.65</td><td>0.642</td><td>4447</td><td>8.41</td><td>0.325</td><td>9377</td><td>24.93</td><td>0.812</td><td>208.9</td><td>24.65</td><td>0.825</td><td>222.9</td></tr><tr><td>Score-SDE (T = 1000)</td><td>17.62</td><td>0.617</td><td>1124</td><td>18.51</td><td>0.678</td><td>916.4</td><td>13.52</td><td>0.437</td><td>2891</td><td>7.12</td><td>0.109</td><td>12620</td><td>6.58</td><td>0.102</td><td>14291</td></tr><tr><td>ADMM-TV</td><td>23.86</td><td>0.803</td><td>267.4</td><td>17.81</td><td>0.814</td><td>1076</td><td>22.03</td><td>0.784</td><td>407.5</td><td>22.37</td><td>0.801</td><td>376.8</td><td>21.36</td><td>0.758</td><td>475.4</td></tr></table>


Table 3: Quantitative evaluation (PSNR, SSIM, MSE) of performance on inverse problems on the FFHQ 256x256-1K dataset.


<table><tr><td rowspan="2"></td><td colspan="4">SR ×4</td><td colspan="4">Random Inpainting</td></tr><tr><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>MSE ↓</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>MSE ↓</td></tr><tr><td>k=0</td><td>0.254</td><td>24.00</td><td>0.691</td><td>141.2</td><td>0.121</td><td>28.33</td><td>0.755</td><td>56.74</td></tr><tr><td>k=1</td><td>0.171</td><td>27.45</td><td>0.792</td><td>117.0</td><td>0.053</td><td>31.84</td><td>0.882</td><td>42.57</td></tr><tr><td>k=4</td><td>0.171</td><td>27.47</td><td>0.794</td><td>116.4</td><td>0.052</td><td>31.99</td><td>0.883</td><td>41.12</td></tr><tr><td>k=16</td><td>0.170</td><td>27.43</td><td>0.799</td><td>117.5</td><td>0.050</td><td>32.12</td><td>0.891</td><td>39.90</td></tr></table>


Table 4: Ablative study on the effect of rank in the low rank and matrix-free approximations on performance (LPIPS, PSNR, SSIM, NMSE) of our proposed model on the FFHQ 256x256-1K dataset dataset.


Randomized Low-Rank Approximation The first and second order terms in Eqs. (19-25) are corresponding Taylor expansions of deep neural functions. Even with the use of automatic differentiation libraries, the formation of these matrices is incredibly expensive, requiring at least dim(x) backpropagation passes (where dim(x) ≈ 39B in some experiments). To reduce the cost of computing these matrices, we utilize their known low rank structure Sagun et al. [2017], Oymak et al. [2019]. 

Leveraging advanced techniques in randomized numerical linear algebra, we estimate Eqs. (19-25) using randomized SVD Halko et al. [2011]. For any matrix $\mathbf { A } \in \breve { \mathbb { R } ^ { m \times n } }$ this is a four step process. 1) We sample a random matrix $\boldsymbol \Omega \sim \mathcal { N } ( \mathbf 0 , \mathbf I _ { n \times k } ) . \dot { 2 } )$ We obtain $\mathbf { A } \Omega = \mathbf { Y } \in \mathbb { R } ^ { m \times k } . 3 )$ We form a basis over the columns of Y, e.g. by taking the Q matrix in a QR factorization QR = Y. 4) We approximate $\mathbf { A } \approx \mathbf { Q } ^ { T } \mathbf { Q } \mathbf { A }$ 

Notably, we observe that when A is a Jacobian (or Hessian) matrix, it can be approximated purely through Jacobian-vector and vector-Jacobian (Hessian-vector and vector-Hessian, resp.) products — without ever materializing A itself. Moreover, a key result in randomized linear algebra is that this algorithm can approximate A up to accuracy $\mathcal { O } ( m n k \sigma _ { k + 1 } )$ (Theorem 1.1 in Halko et al. [2011]). Notably, if A has low rank structure where ∃k such that the k + 1th singular value $\sigma _ { k + 1 } = 0$ , then the approximation is exact. 


Matrix-Free Evaluation Inspired by matrix-free techniques in numerical optimization Knoll and Keyes [2004], we demonstrate a strategy for forming the action update (15) without materializing the costly dim $( \mathbf { x } ) \times \mathrm { d i m } ( \mathbf { x } )$ matrices in the iLQR algorithm (19-25), which we shall denote as an


<table><tr><td rowspan="2"></td><td colspan="4">SR ×4</td><td colspan="4">Random Inpainting</td></tr><tr><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>MSE ↓</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>MSE ↓</td></tr><tr><td>α = 0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>α = 1e-7</td><td>0.173</td><td>27.49</td><td>0.794</td><td>115.9</td><td>0.050</td><td>31.80</td><td>0.879</td><td>42.96</td></tr><tr><td>α = 1e-4</td><td>0.171</td><td>27.45</td><td>0.792</td><td>117.0</td><td>0.053</td><td>31.84</td><td>0.882</td><td>42.57</td></tr><tr><td>α = 1</td><td>0.172</td><td>27.43</td><td>0.799</td><td>117.5</td><td>0.050</td><td>31.85</td><td>0.891</td><td>42.47</td></tr><tr><td>α from Lemma 4.2</td><td>0.170</td><td>27.44</td><td>0.788</td><td>117.3</td><td>0.051</td><td>31.86</td><td>0.880</td><td>42.44</td></tr><tr><td>T = 10</td><td>0.198</td><td>27.48</td><td>0.783</td><td>125.6</td><td>0.168</td><td>27.46</td><td>0.771</td><td>123.7</td></tr><tr><td>T = 20</td><td>0.1923</td><td>31.79</td><td>0.859</td><td>117.0</td><td>0.108</td><td>34.41</td><td>0.910</td><td>42.57</td></tr><tr><td>T = 50</td><td>0.171</td><td>27.45</td><td>0.792</td><td>90.79</td><td>0.053</td><td>31.84</td><td>0.882</td><td>40.56</td></tr><tr><td>T = 200</td><td>0.155</td><td>28.55</td><td>0.811</td><td>43.05</td><td>0.048</td><td>32.05</td><td>0.899</td><td>23.17</td></tr></table>


Table 5: Ablative study on the effect of the Tikhonov regularization coefficient α on performance (LPIPS, PSNR, SSIM, NMSE) of our proposed model on the FFHQ 256x256-1K dataset dataset. No results are reported for $\alpha = 0 .$ , as the algorithm encountered numerical precision errors during matrix inversion.


Table 6: Ablative study on the effect of $T$ on performance (LPIPS, PSNR, SSIM, NMSE) of our proposed model on the FFHQ 256x256-1K dataset dataset. 

indexed set of matrices $\left\{ \mathbf { A } _ { i } \right\}$ . We do this by forming projections of each A<sub>i</sub> against a corresponding set of dim $( \mathbf { x } ) \times { \boldsymbol { \ell } }$ column-orthogonal matrices $\{ \bar { \bf Q } _ { i } \}$ , which we denote as $\mathbf { \bar { B } } _ { i } : = \mathbf { Q } _ { i } ^ { T } \mathbf { A } _ { i }$ . These matrices can then be stored at reduced cost as $( \mathbf { Q } _ { i } , \mathbf { B } _ { i } )$ pairs. 

Matrix multiplications between any $\mathbf { A } _ { i } \mathbf { A } _ { j }$ can then be approximated up to rank ℓ with respect to the projected matrix, $\mathbf { Q } _ { i } \mathbf { A } _ { i , \mathbf { Q } _ { i } } , \mathrm { i . e }$ 

$$
\mathbf {A} _ {i} \mathbf {A} _ {j} \approx \mathbf {Q} _ {i} \mathbf {B} _ {i} \mathbf {Q} _ {j} ^ {T} \mathbf {B} _ {j}.\tag{79}
$$

However, to prevent materialization of the full size of any matrices, we drop the leading $\mathbf { Q } _ { i }$ , obtaining a new projected-matrix pair $\mathbf { ( Q _ { k } , B _ { k } ) }$ , where $\mathbf { Q _ { k } } = \mathbf { Q _ { i } }$ <sub>i</sub>. 

Adam Optimizer Finally, we precondition gradients via the Adam optimizer Kingma and Ba [2014] before applying the feedback gains, rather than applying a backtracking line search Tassa et al. [2014], resulting in the action update 

$$
\mathbf {u} _ {t} = \mathbf {P k} _ {t} + \mathbf {K} _ {t} (\mathbf {x} _ {t} - \mathbf {x} _ {t} ^ {\prime}),\tag{80}
$$

where P is the preconditioning matrix produced by the Adam optimizer. This reduces the overall runtime of the algorithm while still accounting for second-order information that respects the nonlinearity of the optimization landscape. 

## D.2 Computational Complexity Analysis

Incorporating all three modifications, we can provide a realistic runtime and space complexity analysis of our presented algorithm with respect to the rank k, the data dimension d, diffusion steps m, and number of iLQR iterations n. 

Combining both the low rank and matrix-free approximations, we obtain the updated equations for input mode perturbation (where projection matrices are written as P to avoid overloading the Q function notation): 

$$
Q _ {\mathbf {x}} = \mathbf {h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x}} ^ {\prime}\tag{81}
$$

$$
Q _ {\mathbf {u}} = \ell_ {\mathbf {u}} + \mathbf {h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x}} ^ {\prime}\tag{82}
$$

$$
\mathbf {P} Q _ {\mathbf {x x}} \mathbf {P} ^ {T} = \mathbf {P} Q _ {\mathbf {u x}} \mathbf {P} ^ {T} = \mathbf {P} Q _ {\mathbf {x u}} \mathbf {P} ^ {T} = \mathbf {P h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x x}} ^ {\prime} \mathbf {h} _ {\mathbf {x}} \mathbf {P} ^ {T}\tag{83}
$$

$$
\mathbf {P} Q _ {\mathbf {u u}} \mathbf {P} ^ {T} = \mathbf {P} \ell_ {\mathbf {u u}} \mathbf {P} ^ {T} + \mathbf {P h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x x}} ^ {\prime} \mathbf {h} _ {\mathbf {x}} \mathbf {P} ^ {T}.\tag{84}
$$

To simplify notation, each projection matrix P is the same — in reality, this need not be the case. Note that $\mathbf { Q } _ { x }$ and $\mathbf { Q } _ { u }$ are simply of size d and therefore image-sized. For all our datasets, these each take 0.2 MB to store and are therefore negligible, and we do not project these variables. When $\ell _ { { \bf u u } }$ is diagonal (as it is in our case), we can obtain the projected inverse for $Q _ { \mathbf { u u } }$ as 

$$
\mathbf {P} Q _ {\mathbf {u u}} ^ {- 1} \mathbf {P} ^ {T} = \mathbf {P} \ell_ {\mathbf {u u}} ^ {- 1} \mathbf {P} ^ {T} + \mathbf {P} \ell_ {\mathbf {u u}} ^ {- 1} \mathbf {P} ^ {T} (\mathbf {C} ^ {- 1} + \mathbf {P} ^ {T} \ell_ {\mathbf {u u}} ^ {- 1} \mathbf {P}) ^ {- 1} \mathbf {P} \ell_ {\mathbf {u u}} ^ {- 1} \mathbf {P} ^ {T} \quad \text { where } \mathbf {C} = \mathbf {P h} _ {\mathbf {x}} ^ {T} V _ {\mathbf {x x}} ^ {\prime} \mathbf {h} _ {\mathbf {x}} \mathbf {P}\tag{85}
$$

via a direct application of the Woodbury matrix inversion formula Petersen et al. [2008], which has cost $\mathcal { O } ( k ^ { 3 } + \dot { k } d ^ { 2 } )$ . Finally, we compute the projected updates $V _ { \mathbf { x } \mathbf { x } } , \mathbf { K }$ as well as the full-precision 

$V _ { \mathbf { x } }$ , k terms via 

$$
\mathbf {k} = - \mathbf {P} ^ {T} \mathbf {P} Q _ {\mathbf {u u}} ^ {- 1} \mathbf {P} ^ {T} \mathbf {P} Q _ {\mathbf {u}}\tag{86}
$$

$$
V _ {\mathbf {x}} = Q _ {\mathbf {x}} - \mathbf {P} ^ {T} \mathbf {P} \mathbf {K} ^ {T} \mathbf {P} ^ {T} \mathbf {P} Q _ {\mathbf {u u}} \mathbf {P} ^ {T} \mathbf {P} \mathbf {k}\tag{87}
$$

$$
\mathbf {P K P} ^ {T} = - \mathbf {P} Q _ {\mathbf {u u}} ^ {- 1} \mathbf {P} ^ {T} \mathbf {P} Q _ {\mathbf {u x}} \mathbf {P} ^ {T}\tag{88}
$$

$$
\mathbf {P} V _ {\mathbf {x x}} \mathbf {P} ^ {T} = \mathbf {P} Q _ {\mathbf {x x}} \mathbf {P} ^ {T} - \mathbf {P} \mathbf {K} ^ {T} \mathbf {P} ^ {T} \mathbf {P} Q _ {\mathbf {u u}} \mathbf {P} ^ {T} \mathbf {P} \mathbf {K} \mathbf {P} ^ {T}.\tag{89}
$$

Where applicable, we leverage vector-Jacobian products from standard automatic differentiation libraries (e.g. torch.func.vjp) which have runtime complexity O(1). Computing the $V _ { \mathbf { x } } , V _ { \mathbf { x } \mathbf { x } } , \mathbf { k }$ , K terms in Eqs. (46)-(49) costs $\mathbf { \hat { O } } ( k ^ { 3 } + k d ^ { 2 } )$ FLOPs in terms of matrix multiplications (dominated by the matrix inverse of $k \times k$ matrix $\mathbf { q } ^ { T } Q _ { \mathbf { u u } } \mathbf { q } )$ . Crucially, it incurs ${ \mathcal { O } } ( k )$ neural function evaluations (NFEs), which dominates the runtime of the algorithm. Since this computation is performed for each diffusion step and iLQR iteration, the total runtime complexity of our algorithm is $O ( n m ( k ^ { 3 } + k d ^ { 2 } ) )$ matrix multiplication FLOPs and $\mathcal { O } ( n m k )$ NFEs, with $O ( \dot { m \boldsymbol { k ^ { 2 } } } + d )$ space complexity. In terms of time complexity, the NFEs are the dominating cost, accounting for 97% of computation time. 

## D.3 Sensitivity to Hyperparameters

In Tables 4, 5, 6, we investigate the effect of the rank of the low rank approximation and matrix-free projections, the Tikhonov regularization coefficient α, and the diffusion time T on the performance of our method on the FFHQ 256x256 dataset. We evaluate performance on the super-resolution and random inpainting tasks, with the same setup as in Section 6. 

Low-Rank and Matrix-Free Rank From Table 4, it is clear that there is a significant performance gain from even a rank one approximation of the first- and second-order matrices. The gains from subsequent increases in the rank approximation diminish quickly. This is because increasing the rank of the approximation only improves the approximation of the second-order terms. The first order $V _ { \mathbf { x } } , Q _ { \mathbf { x } } , Q _ { \mathbf { u } }$ terms are always modeled exactly in $\mathcal { O } ( 1 )$ time per iteration due to their amenability to vector-Jacobian products. From Theorems 4.1-4.3 we see that even when the second order terms are zero (i.e., the result of assumption $\ell _ { t } = 0 )$ , we exactly recover the true posterior sampler. Therefore, the second-order terms are less important, though still useful for imposing a quadratic trust-region regularization to the algorithm. Therefore, we ultimately choose $k = 1$ for three reasons: 

1. the rank only affects the quadratic approximation of the iLQR algorithm (and does not affect our theoretical results in Theorems 4.1-4.3) 

2. $k = 1$ already allows second-order propagation of the quadratic trust-region regularization, and 

3. subsequent increases in k have a minimal effect on the performance of the algorithm. 

Tikhonov Regularizer Table 5 demonstrates that our algorithm is relatively robust to the Tikhonov regularization parameter, except when $\alpha = 0$ . Under this condition, any ill-conditioning of $Q _ { \mathbf { u u } }$ results in division by zero errors, resulting in the failure of the algorithm. Therefore, we simply choose to let $\alpha = 1 e - 4$ , since the effect of Tikhonov regularizer is minimal. 

Diffusion Steps Finally, we observe in Table 6 that increasing the diffusion time results in higher quality samples — though at the cost of increased computation time. Therefore, choice of T requires balancing computational cost and sample quality, and is ultimately highly user-dependent. When the computational and latency budget is relatively high, large T can be used to improve sample quality. Conversely, when this budget is low, we find that even $\check { T } = 2 0$ provides reasonable samples. 