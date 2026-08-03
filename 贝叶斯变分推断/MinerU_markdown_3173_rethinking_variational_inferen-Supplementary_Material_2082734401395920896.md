# Appendix for Rethinking Variational Inference for Probabilistic Programs with Stochastic Support

Tim Reichelt<sup>1</sup> Luke Ong<sup>1,2</sup> Tom Rainforth<sup>1</sup> <sup>1</sup> University of Oxford <sup>2</sup> Nanyang Technological University, Singapore {tim.reichelt,lo}@cs.ox.ac.uk rainforth@stats.ox.ac.uk 

## A KL Divergence Derivation

## A.1 Breaking Down the Global ELBO

The global ELBO is given by 

$$
\mathcal {L} (\phi , \lambda) = \mathbb {E} _ {q (x; \phi , \lambda)} \left[ \log \frac {\gamma (x)}{q (x ; \phi , \lambda)} \right],\tag{16}
$$

$$
= \int_ {\mathcal {X}} q (x; \phi , \lambda) \log \frac {\gamma (x)}{q (x ; \phi , \lambda)} d x,\tag{17}
$$

using the fact that the subsets $\mathcal { X } _ { k }$ provide a partition of we can write the integral as 

$$
= \sum_ {k = 1} ^ {K} \int_ {\mathcal {X} _ {k}} q (x; \phi , \lambda) \log \frac {\gamma (x)}{q (x ; \phi , \lambda)} d x,\tag{18}
$$

using the factorization of $q ( x ; \phi , \lambda )$ and the fact that for $x \in \mathcal { X } _ { k }$ the program density satisfies $\gamma ( \boldsymbol { x } ) = \gamma _ { \boldsymbol { k } } ( \boldsymbol { x } )$ we get 

$$
= \sum_ {k = 1} ^ {K} \int_ {\mathcal {X} _ {k}} q _ {k} (x; \phi_ {k}) q (k; \lambda) \log \frac {\gamma_ {k} (x)}{q _ {k} (x ; \phi_ {k}) q (k ; \lambda)} d x\tag{19}
$$

then using the fact that $q ( k ; \lambda )$ does not depend on x we have 

$$
= \sum_ {k = 1} ^ {K} q (k; \lambda) \int_ {\mathcal {X} _ {k}} q _ {k} (x; \phi_ {k}) \log \frac {\gamma_ {k} (x)}{q _ {k} (x ; \phi_ {k})} d x - \log q (k; \lambda),\tag{20}
$$

which we can write concisely as 

$$
= \mathbb {E} _ {q (k; \lambda)} \left[ \mathcal {L} _ {k} (\phi_ {k}) - \log q (k; \lambda) \right],\tag{21}
$$

where 

$$
\mathcal {L} _ {k} (\phi_ {k}) := \mathbb {E} _ {q _ {k} (x; \phi_ {k})} \left[ \log \frac {\gamma_ {k} (x)}{q _ {k} (x ; \phi_ {k})} \right].
$$

## A.2 Optimal Setting of $q ( k ; \lambda )$

Proposition 1. Let $L = \{ \mathcal { L } _ { 1 } , . . . , \mathcal { L } _ { K } \}$ be the set of local ELBOs, defined as per (7), where $L$ is countable but potentially not finite. $\begin{array} { r } { I f 0 < \sum _ { k = 1 } ^ { K } \exp ( \mathcal { L } _ { k } ) < \infty , } \end{array}$ , then the optimal corresponding $q ( k ; \lambda )$ in terms ofthe global ELBO (6) is given by 

$$
q (k; \lambda) = \exp (\mathcal {L} _ {k}) / \sum_ {\ell = 1} ^ {K} \exp (\mathcal {L} _ {\ell}).\tag{8}
$$

Proof. By the assumption that $\begin{array} { r } { 0 < \sum _ { k = 1 } ^ { K } \exp ( \mathcal { L } _ { k } ) < \infty } \end{array}$ , we have that $\exp ( \mathcal { L } _ { k } ) \big / { \sum _ { k = 1 } ^ { K } \exp ( \mathcal { L } _ { k } ) }$ forms a valid probability mass function over $k \tilde { \in \{ 1 , \ldots , K \} }$ . We can therefore rewrite (7) as 

$$
\mathcal {L} (\phi , \lambda) = \mathbb {E} _ {q (k; \lambda)} \left[ \log \frac {\exp (\mathcal {L} _ {k})}{\sum_ {k = 1} ^ {K} \exp (\mathcal {L} _ {k})} - \log q (k; \lambda) \right] + \log \sum_ {k = 1} ^ {K} \exp (\mathcal {L} _ {k})\tag{22}
$$

$$
= - \mathrm{KL} \left(q (k; \lambda) \| \frac {\exp (\mathcal {L} _ {k})}{\sum_ {k = 1} ^ {K} \exp (\mathcal {L} _ {k})}\right) + \log \sum_ {k = 1} ^ {K} \exp (\mathcal {L} _ {k})\tag{23}
$$

Now as second term in the above is constant in $q ( k ; \lambda )$ and a KL divergence is minimized when the two distributions are the same, we can immediately conclude the desired result that the optimal $q ( k ; \lambda )$ is 

$$
q (k; \lambda) = \frac {\exp (\mathcal {L} _ {k})}{\sum_ {\ell = 1} ^ {K} \exp (\mathcal {L} _ {\ell})}.\tag{24}
$$

Additionally, from (23) it follows that for the optimal setting of the mixture distribution $q ( k ; \lambda )$ the global ELBO is given by 

$$
\mathcal {L} (\phi , \lambda^ {*}) = \log \sum_ {k = 1} ^ {K} \exp (\mathcal {L} _ {k}).
$$

## B Details on Resource Allocation

## B.1 Background on Successive Halving

Successive Halving (SH) divides a total budget of $T$ iterations into $L = \lceil \log _ { 2 } ( K ) \rceil + 1$ phases and starts by optimizing each of K candidates, in our case the SLPs, for $\dot { \lfloor T / ( K L ) \rfloor }$ iterations. It then ranks each of the candidates in terms of their performance, in our case the values of $\exp ( \mathcal { L } _ { k } )$ , before eliminating the bottom half. This process then repeats, with each of the remaining candidates run for $2 ^ { \ell - 1 } T \big / ( K L )$ iterations at the ℓ-th phase. This results in an exponential distribution of resources allocated to the different candidates, with more resources allocated to those that are more promising after intermediate evaluation. 

Adapting it to our setting of treating the problem as a top-m identification is done by simply using $L = \mathsf { \bar { \rho } } [ \log _ { 2 } ( K ) - \log _ { 2 } ( \bar { m _ { \alpha } } ) ] + 1$ phases instead of $L = \lceil \bar { \log _ { 2 } ( K ) } \rceil + 1$ 

## B.2 Online Resource Allocation

Here, we present an online version of Algo. 1, where the term ‘online’ refers to the fact that the algorithm considers more and more SLPs as the computational budget increases. The online variant of the algorithm is useful if a user is unsure about the total iteration budget that they want to spend on the input program. This user might want to run SDVI with an initial iteration budget $T _ { 1 }$ and after having observed the results, they might decide that they want to keep further optimizing the guide parameters. We therefore need to adapt Algo. 1 so that it can be ‘restarted’ after it has terminated. A naive approach to this would be to simply run Algo. 1 again but re-use the $q _ { k } \mathrm { ^ { * } s }$ for the SLPs that have already been discovered and only initialize the $q _ { k }$ from scratch for SLPs which have not been seen before. However, this scheme is limited as it disproportionately favours SLPs which were discovered in the previous run. This is because for those SLPs the local ELBOs will already be relatively large compared to the newly added SLPs. As a consequence, SH will not assign significant computational budget to the SLPs that were added after the algorithm was restarted. 

To safeguard against this behaviour we instead propose an online version of SDVI in Algo. 2 which is using a modified ‘reward’ for SH. Instead of ranking the different SLPs according to $\mathcal { L } _ { k } ( \phi _ { k } ( t _ { k } ) )$ q we instead propose the objective $\exp ( \alpha \mathcal { L } _ { k } ( \phi _ { k } ( t _ { k } ) ) ) / i \bar { \phi _ { k } }$ where $0 < \alpha \leqslant 1$ . The reward is scaled by the reciprocal of $t _ { k }$ because we are no longer aiming to select the SLPs with the highest $\mathcal { L } _ { k } ( \phi _ { k } ( t _ { k } ) )$ but instead aim to choose the SLPs which have been ‘underselected’ compared to other $\mathrm { S L P s } .$ , assuming we should have selected them in proportion to $\exp ( \alpha \mathcal { L } _ { k } ( \phi _ { k } ( t _ { k } ) ) )$ . The scaling by the scalar α is a further mechanism to encourage more exploration, with setting $\alpha = 0$ equivalent to uniform sampling in the limit of repeated SH runs. Since this adapted objective takes into account the computational budget that was spent on each SLP, it is a more suitable objective when running SH repeatedly. 

```julia
Algorithm 2 Online SDVI
Require: Target program γ, iteration budget per SH run T, minimum no. of SH candidates m, parameter controlling α > 0 exploration
1: Extract SLPs {γk}K k=1 from γ and set C = {1, . . ., K}
2: Formulate guide qk for each SLP and initialize parameters φk
3: tk = 0 for all k ∈ C
4: while Stopping criteria not satisfied do
5:    C' ← C
6:    Phases in successive halving L = [log₂(|C|) - log₂(m)] + 1
7:    for l = 1, . . ., L do
8:    Number of iterations nl = [T/L|C']
9:    for k ∈ C' do
10:    Perform nl optimization iterations of φk targeting L_surr,k(φk)
11:    Estimate L_surr,k(φk) using Monte Carlo estimate of Eq. (11)
12:    tk = tk + nl
13:    end for
14:    Remove min([|C'/2], |C'/-m) SLPs from C' with the lowest exp(α L_surr,k(φk))/tk
15:    end for
16:    Extract new SLPs from γ and add them to C, set tk' = 0 for each new SLP with index k'
17: end while
18: Truncate qk outside of SLP support, Xk, using Eq. (13)
19: Estimate each Lk(φk) using Monte Carlo estimate of Eq. (7)
20: Calculate q(k; λ) according to Eq. (8) and return q(x; φ, λ) as per Eq. (4) 
```

## C Details for Training Local Guides

## C.1 Density Estimation of the Prior

Before we can define the KL divergence we first have to carefully define global and local prior distributions We first define what we informally call the global ‘prior’ distribution of the program as the product of all the terms added to the program density by the sample statements 

$$
\pi_ {\mathrm{prior}} (x _ {1: n _ {x}}) := \prod_ {i = 1} ^ {n _ {x}} f _ {a _ {i}} (x _ {i} | \eta_ {i}).\tag{25}
$$

However, here we are using the term prior only informally, since (25) is not a prior in the conventional Bayesian sense since the $\eta _ { i }$ can be functions of the observed data $y .$ . Note that here $n _ { x }$ in (25) is again a random variable since the raw random draws $x _ { 1 : n _ { i } }$ of the program do not necessarily have fixed length. Then similarly we define local ‘prior’ distributions 

$$
\pi_ {\mathrm{prior}, k} (x _ {1: n _ {k}}) := \frac {\mathbb {I} [ x _ {1 : n _ {k}} \in \mathcal {X} _ {k} ] \prod_ {i = 1} ^ {n _ {k}} f _ {A _ {k} [ i ]} (x _ {i} | \eta_ {i})}{Z _ {\mathrm{prior} , k}} = \frac {\mathbb {I} [ x _ {1 : n _ {k}} \in \mathcal {X} _ {k} ] \pi_ {\mathrm{prior}} (x _ {1 : n _ {k}})}{Z _ {\mathrm{prior} , k}},,\tag{26}
$$

where 

$$
Z _ {\text { prior }, k} := \int_ {\mathcal {X}} \mathbb {I} \big [ x \in \mathcal {X} _ {k} \big ] \pi_ {\text { prior }} (x) d x.\tag{27}
$$

Note that for our purposes we will never actually have to estimate $Z _ { p r i o r , k }$ , we only defined it to ensure that $\pi _ { p r i o r , k }$ is a normalized density. This allows us to define the forward KL divergence which we would like to optimize with respect to ϕ 

$$
\mathrm{KL} (\pi_ {\text {prior}, k} (x) \parallel \tilde {q} _ {k} (x; \phi_ {k})) = \mathbb {E} _ {\pi_ {\text {prior}, k} (x)} \left[ \log \frac {\pi_ {\text {prior} , k} (x)}{\tilde {q} _ {k} (x ; \phi_ {k})} \right]\tag{28}
$$

which we can rewrite as 

$$
= \mathbb {E} _ {\pi_ {\text { prior }, k} (x)} \left[ \log \pi_ {\text { prior }, k} (x) \right] - \mathbb {E} _ {\pi_ {\text { prior }, k} (x)} \left[ \log \tilde {q} _ {k} (x; \phi_ {k}) \right].\tag{29}
$$

The first term is a constant with respect to $\phi _ { k }$ and therefore does not affect the optimization 

$$
\propto \mathbb {E} _ {\pi_ {\mathrm{prior}, k} (x)} \left[ - \log \tilde {q} _ {k} (x; \phi_ {k}) \right],\tag{30}
$$

then by the definition of $\pi _ { \mathrm { p r i o r } , k } ( x )$ in Eq. (26) this is equivalent to 

$$
= - \frac {1}{Z _ {\mathrm{prior} , k}} \mathbb {E} _ {\pi_ {\mathrm{prior}} (x)} \left[ \mathbb {I} [ x \in \mathcal {X} _ {k} ] \log \tilde {q} _ {k} (x; \phi_ {k}) \right].\tag{31}
$$

Finally, $Z _ { p r i o r , k }$ is a constant with respect to $\phi _ { k }$ and can be dropped 

$$
\propto \mathbb {E} _ {\pi_ {\mathrm{prior}} (x)} \left[ - \mathbb {I} \big [ x \in \mathcal {X} _ {k} \big ] \log \tilde {q} _ {k} \big (x; \phi_ {k} \big) \right].\tag{32}
$$

We can estimate the gradients of the objective in $\operatorname { E q . }$ . (32) using a Monte Carlo estimator 

$$
\nabla_ {\phi_ {k}} \mathbb {E} _ {\pi_ {\mathrm{prior}} (x)} \left[ - \mathbb {I} [ x \in \mathcal {X} _ {k} ] \log \tilde {q} _ {k} (x; k, \phi_ {k}) \right] \approx \frac {1}{N} \sum_ {j = 1} ^ {N} \mathbb {I} [ x ^ {(j)} \in \mathcal {X} _ {k} ] \nabla_ {\phi_ {k}} \log \tilde {q} _ {k} (x ^ {(j)}; k, \phi_ {k})\tag{33}
$$

where $x ^ { ( j ) }$ are raw random draws generated by executing the input program forward. These gradient estimates can then be used in a stochastic gradient descent optimization procedure. In our experiments, we generate a fixed set of N samples and re-use the same set of samples for the entire optimization process. Other approaches are also possible such as periodically collecting a new set of samples and using local MCMC moves to collect samples instead of repeatedly sampling from the prior. 

## C.2 Exploiting Program Structure: Discrete Branching Optimization

In practice, many user-defined programs have structural properties which can be exploited to construct a valid local guide directly and deterministically (without resorting to the stochastic mechanism described in Sec. 4.5). Specifically, consider the class of programs whose program paths are determined by variables sampled from discrete distributions. For these programs, we can assume that for each SLP (kth, say) there is an (ordered) set of indices $I _ { \mathrm { b r a n c h } } \subset \{ 1 , \cdot \cdot . . , n _ { k } \} = I$ and a set of constants $r _ { k , 1 } , \ldots , r _ { k , | I _ { \mathrm { b r a n c h } } | } \in \mathbb { Z }$ such that the local unnormalized densities are expressible as 

$$
\gamma_ {k} (x _ {1: n _ {k}}) = \gamma (x _ {1: n _ {k}}) \prod_ {l = 1} ^ {| I _ {\mathrm{branch}} |} \mathbb {I} \left[ x _ {I _ {\mathrm{branch}} [ l ]} = r _ {k, l} \right]
$$

where $I _ { \mathrm { b r a n c h } } [ j ]$ means the jth element in $I _ { \mathrm { b r a n c h } }$ . It follows that we can construct densities for the kth SLP on a subset of variables in $x _ { 1 : n _ { k } }$ by eliminating all the variables given by indices $I _ { \mathrm { b r a n c h } }$ (by instantiating them to constants). This is effectively equivalent to replacing the sample statements corresponding to the variables which influence the control flow with observe statements which induces a new program density that has the form 

$$
\tilde {\gamma} _ {k} (x _ {1: n _ {k} ^ {\prime}}) = \prod_ {i = 1} ^ {n _ {k} ^ {\prime}} f _ {A _ {k} [ I ^ {\prime} [ i ] ]} (x _ {i} | \eta_ {i}) \prod_ {l = 1} ^ {| I _ {\mathrm{branch}} |} f _ {A _ {k} [ I _ {\mathrm{branch}} [ l ] ]} (r _ {k, l} \mid \eta_ {l}) \prod_ {j = 1} ^ {n _ {y}} g _ {b _ {j}} (y _ {j} \mid \phi_ {j})\tag{34}
$$

where $I ^ { \prime } : = [ 1 , \ldots , n _ { k } ] \backslash I _ { \mathrm { b r a n c h } }$ , and $n _ { k } ^ { \prime } : = | I ^ { \prime } |$ . Furthermore, if all the remaining r.v. are continuous distributions with support in R (i.e. supp $\mathbf { \Psi } ( f _ { A _ { k } [ i ] } ) = \mathbb { R }$ for $i \in I ^ { \prime } )$ then $\tilde { \gamma } _ { k } \big ( \boldsymbol { x } _ { 1 : n _ { k } ^ { \prime } } \big )$ itself has support in $\mathbb { R } ^ { n _ { k } ^ { \prime } }$ . It is then straightforward to construct a guide $q _ { k }$ with support in $\mathbb { R } ^ { n _ { k } ^ { \prime } }$ using existing methods, and we can get gradient estimates using the reparameterization gradient estimator (assuming there are no more discontinuities in $\tilde { \gamma } _ { k } )$ 

To realize the discrete branching optimization in our Pyro implementation we allow users to annotate the sample statements which influence the branching. While it is in principle possible to automatically identify programs with discrete branching using program analysis, formalizing and implementing such a program analysis tool to work with arbitrary Pyro program would be a significant contribution in itself which is out of scope for this paper as we are focused on the statistical evaluation of SDVI. Specifically, the relevant sample statements within a Pyro program can be annotated as follows: pyro.sample $( " \mathbf { z } "$ , dist.Poisson(7), infer={"branching": True}). Our implementation of SDVI is then able to use these annotations to create the density $\tilde { \gamma } _ { k }$ in (34). 

## D Additional Details for Experiments

For all experiments that rely on optimization we use the Adam optimizer [70]. The experiments were executed on an internal cluster which uses a range of different computer architectures. 

## D.1 Model From Figure 1

Listing 1: Pyro Code for Figure 1. 

```python
import pyro
import pyro.distributions as dist

def model():
    x = pyro.sample("x", dist.Normal(0, 1))
    if x < 0:
    z1 = pyro.sample("z1", dist.Normal(-3, 1))
    else:
    z1 = pyro.sample("z2", dist.Normal(3, 1))

    x = pyro.sample("x", dist.Normal(z1, 2), obs=torch.tensor(2.0))

guide = pyro.infer.autoguide.AutoNormalMessenger(model)
optim = pyro.optim.Adam({"lr": 0.01})
svi = pyro.infer.SVI(
    model, guide, optim, loss=pyro.infer.Trace_ELBO()
)

for j in range(2000):
    svi.step() 
```

The full Pyro code for the model in Fig. 1, including automatically generating and training the guide is given in Listing 1. The code for BBVI and SDVI is provided in the code supplementary. For Pyro’s AutoGuide and BBVI we run the optimization for 2000 iterations with a learning rate of 0.01. Similarly, for SDVI we have a total iteration budget of $T = 2 0 0 0$ and use a learning rate of 0.01; we set the minimum number of SH candidates to $m = 2$ 

## D.2 Program with Normal Distributions

For SDVI, we use $1 0 ^ { 3 }$ samples from the prior to discover SLPs. To train the local guides to place support within the SLP boundaries we collect $1 0 ^ { 2 }$ samples per SLP and optimize the objective in Equation (12) for $1 0 ^ { 3 }$ iterations. We run Algorithm 1 with a total budget of $\dot { T } = 1 0 ^ { 5 }$ with 5 particles for the ELBO and to estimate the final SLP weights we use $1 0 ^ { 3 }$ samples per SLP. We use a learning rate of 0.01. 

For Pyro AutoGuide, we run the optimization for $1 0 ^ { 5 }$ steps with 1 ELBO particle. For BBVI, we run the optimization for $1 0 ^ { 4 }$ steps with 10 ELBO particles. For both we use a learning rate of 0.01. 

## D.3 Infinite Gaussian Mixture Model

For SDVI, we use $1 0 ^ { 3 }$ samples from the prior to discover SLPs, run Algorithm 1 with a total budget of $T = 2 * 1 0 ^ { 4 }$ with 10 particles for the ELBO and to estimate the final SLP weights we use $\mathrm { i 0 ^ { 2 } }$ samples per SLP. We use a learning rate of 0.1. 

For BBVI, we run for $2 * 1 0 ^ { 4 }$ iterations using 10 particles for the ELBO and a learning rate of 0.1. In the guide, we use a categorical distribution for number of components K over the range $K \in [ 1 , 2 5 ]$ We ran initial experiments with instead using a Poisson distribution paramterized by the rate but we found this leads to an explosion in the number of components in the guide which resulted in the program running out of memory. For each $\mu _ { k }$ the variational approximation is a diagonal Normal distribution parameterized by the mean and the diagonal entries in the covariance matrix. 

For DCC, we run for 200 iterations, at each iteration we run 10 independent RMH chains generating 10 samples and to get a marginal likelihood estimate we use PI-MAIS [66] which places a proposal distribution (in our case a Gaussian) on the outputs of the RMH chains and samples from this proposal M times; we set $M = 1 0$ 

## D.4 Inferring Gaussian Process Kernels

## D.4.1 Model Details

Our probabilistic context-free grammar for the kernel structure has the production rules 

$$
\mathcal {K} \rightarrow \mathrm{SE} | \mathrm{RQ} | \mathrm{PER} | \mathrm{LIN} | \mathcal {K} \times \mathcal {K} | \mathcal {K} + \mathcal {K}.\tag{35}
$$

with the production probabilities $[ 0 . 2 , 0 . 2 , 0 . 2 , 0 . 2 , 0 . 1 , 0 . 1 ]$ . On each base kernel hyperparameter we place an InverseGamma $( \alpha = \bar { 2 } , \beta = 1 )$ prior. For each base kernel the specific hyperparameters we wish to do inference over are:<sup>2</sup> 

• Squared Exponential (SE): Lengthscale 

• Rational Quadratic (RQ): Lengthscale, Scale Mixture 

• Periodic (PER): Lengthscale, Period 

• Linear (LIN): Bias 

Assuming we have N observations with inputs $\mathbf { x } \in \mathbb { R } ^ { N }$ and outputs $\mathbf { y } \in \mathbb { R } ^ { N }$ our model can then be written as 

$$
\mathcal {K} \sim \operatorname{PCFG} (), \quad \sigma \sim \operatorname{HalfNormal} (0, 1), \quad \mathbf {y} \sim \mathcal {N} (0, \mathcal {K} (\mathbf {x}) + \sigma^ {2} \mathrm{I})\tag{36}
$$

where PCFGpq samples a kernel (and its hyperparameters) from the probabilistic context-free grammar and $\kappa ( \mathbf { x } )$ is the $N \times N$ covariance matrix computed from kernel . 

## D.4.2 Algorithm Configurations

For SDVI, we use $1 0 ^ { 3 }$ samples from the prior to discover SLPs, run Algorithm 1 with a total budget of $T = 1 0 ^ { 6 }$ with 1 particles for the ELBO and to estimate the final SLP weights we use $1 0 ^ { 2 }$ samples per SLP. We use a learning rate of 0.005. 

For BBVI, we run for $1 0 ^ { 5 }$ iterations using 10 particles for the ELBO and a learning rate of 0.005. The guide uses a log-normal distribution for the kernel hyperparameters and the observation noise, and for the discrete variables which influence the kernel structure we use categorical distributions. For DCC, we run for $1 0 ^ { 3 }$ iterations and otherwise use the exact same hyperparameters as in the Gaussian Mixture Model experiment. 

## E Additional Experimental Results

## E.1 Program with Normal Distributions

For completeness we include here the results for DCC on the model from Sec. 6.1. DCC does not have the same fundamental limitations as the BBVI baselines therefore is competitive with SDVI and provides a similar squared error for the SLP weights. In fact, it is quite impressive that SDVI is able to match the performance of DCC because DCC leverages marginal likelihood estimators which asymptotically converge to the true marginal likelihood whereas SDVI calculates the weights based on the ELBO. This 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/1484dee9-8c11-4455-858f-5f99548dc420/ed10f179c828b85deb4e6055cbe568d7c2a53d946fc19202600be1d57506e8cd.jpg)



Figure 4: Squared error for the model in § 6.1 with DCC baseline. Conventions as in Fig. 2a.


is therefore a further indicator that for this model SDVI is able to provide good posterior approximations for each SLP. 

## F Difficulties of Parameter Learning for Models with Stochastic Support

In static support settings, one often uses variational bounds not only as a mechanism for inference, but also for training model parameters themselves [24, 41]. Using our notation from Sec. 2.2, this setting corresponds to having model parameters, $\theta ,$ that we wish to optimize alongside the variational parameters, $\phi ,$ such that the unnormalized density can be written as $\gamma ( \boldsymbol { x } ; \boldsymbol { \theta } )$ , with corresponding normalization constant $Z ( \theta )$ . The ELBO then depends on both the variational and model parameters $\mathcal { L } ( \phi , \theta ) : = \mathbb { E } _ { q ( x ; \phi ) } \left[ \log \gamma ( x ; \theta ) / q ( x ; \phi ) \right]$ s. Provided $Z ( \theta )$ is differentiable with respect to θ, both ϕ and θ can then, at least in principle, be simultaneously optimized using stochastic gradient ascent. 

However, similar as to the case of pure inference, naively extending this scheme to models with stochastic support is non-trivial and quickly runs into both conceptual and practical problems. 

Parameters, $\theta _ { k }$ , that are inherently local to only a single SLP can be dealt with straightforwardly: as $\nabla _ { \theta _ { k } } \mathcal { L } _ { \ell } = 0 \forall \ell \neq k$ for such parameters, we can simply ignore parameters not associated with the SLP we are updating, that is we only take a gradient step for $\left\{ \phi _ { k } , \theta _ { k } \right\}$ on Line 5 of Algo. 1. 

Problems start to occur, though, in the more common scenario where parameters are shared between SLPs, in the sense that they influence more than one $\gamma _ { k }$ . Consider, for example, the GP model from Sec. 6.3 and assume that instead of doing inference over the observation noise, σ, we instead wish to treat this as a learnable parameter instead. Here σ could be seen as a ‘global’ model parameter as it appears in every SLP, so could be viewed as shared between them. 

This now creates an issue in ‘balancing’ updates from different SLPs; the need to learn a shared θ breaks the separability between inference problems for individual SLPs. Consequently, we can no longer directly treat how often we update each SLPs as just a resource allocation problem: making more updates on a given SLP now increases the influence that SLP has on the θ which are learned. This problem is unlikely to be insurmountable—one could maintain a running estimate of $q ( k ; \lambda )$ during training and then use this to either directly control the resource allocation or scale the updates of θ depending on how often the corresponding SLP has been used—but it does represent a notable complication that would require its own careful consideration. 

Beyond this specific practical challenge, there is also a more fundamental and general issue for parameter learning under stochastic support: should shared parameters be treated globally when we are learning them? Going back to the example of the observation noise, $\sigma ,$ in our GP example, it will actually be quite inappropriate here to learn a single global value for $\sigma ,$ , as the optimal observation noise will be different depending on the kernel structure. Thus, though the variable is shared between SLPs in the program itself, it would be advantageous to learn separate values for it for each SLP, regardless of the inference approach we take. 

The natural solution to this issue would be to perform parameter learning separately for each SLP, e.g. learning a separate $\sigma _ { k }$ for each SLP in the GP example above. However, this raises a variety of issues in its own, not least the fact that the inference algorithm will now start to influence the model itself: SDVI and BBVI will learn fundamentally different models. There may also be settings where it is important for a parameter to be truly global and thus shared across the SLPs, e.g. because such sharing is an explicit prior assumption we wish to make. 

Further problems occur when we consider that it is also feasible for learnable parameters to influence the control flow of the program, or even the set of possible SLPs. For example, a learnable parameter could impact the maximum possible recursion depth of a recursive program. This will create challenging interactions between SLPs: updates of one will influence the desirable behavior for the variational approximation of another. In turn, this can substantially complicate the resource allocation process and even the SLP discovery process itself. 

Together, these aforementioned issues demonstrate that parameter learning for models with stochastic support is a complex issue, requiring specialist consideration beyond the scope of the current paper. 

## G Issues with Directly Training $q _ { k }$

A natural question one might ask with the SDVI method is why do we not directly train $q _ { k }$ to (7) by treating it as an implicit variational approximation defined by $\tilde { q } _ { k } { : }$ Namely, we can express (7) in 

terms of $\tilde { q } _ { k }$ as follows 

$$
\mathcal {L} _ {k} (\phi_ {k}) = \log \tilde {Z} _ {k} (\phi_ {k}) + \frac {1}{\tilde {Z} _ {k} (\phi_ {k})} \mathbb {E} _ {\tilde {q} _ {k} (x; \phi_ {k})} \left[ \mathbb {I} [ x \in \mathcal {X} _ {k} ] \log \frac {\gamma_ {k} (x)}{\tilde {q} _ {k} (x ; \phi_ {k})} \right],\tag{37}
$$

which, in principle, could be directly optimized with respect to $\phi _ { k }$ 

There are unfortunately two reasons that make this impractical. Firstly, though $\tilde { Z } _ { k } ( \phi _ { k } )$ can easily be estimated using Monte Carlo, we actually cannot generate conventional unbiased estimates of log $\tilde { Z } _ { k } ( \phi _ { k } )$ and $1 / \tilde { Z } _ { k } ( \phi _ { k } )$ (or their gradients) because mapping the Monte Carlo estimator induces a bias. Second, this objective applies no pressure to learn a $\tilde { q } _ { k }$ with a high acceptance rate, i.e. which actually concentrates on $\mathrm { S L P } k ,$ , such that it can easily learn a variational approximation that is very difficult to draw truncated samples from at test time. 

By contrast, using our surrogate objective in (11) allows us to produce unbiased gradient estimates. Because of the mode seeking behaviour of variational inference, it also naturally forces us to learn a variational approximation with a high acceptance rate, provided we use a suitably low value of c. If desired, one can even take $c \to 0$ during training to learn an approximation which only produces samples from the target SLP without requiring any rejection. Figure 5 shows that empirically we learn a $\tilde { q } _ { k }$ with a very high acceptance rates for the problem in Section 6.1. 

Note that the surrogate and true ELBOs are exactly equal for any variational approximation that is confined to the $\mathrm { S L P }$ (as these have $\tilde { Z } _ { k } ( \phi _ { k } ) = 1 )$ . This does not always necessarily mean that they have the same optima in $\phi _ { k }$ for restricted variational families, even in the limit $c \to 0$ . However, such differences originate from the fact that the trunctation can itself actually generalize the variational family (e.g. if $\tilde { q } _ { k }$ is Gaussian, then $q _ { k }$ will be a truncated Gaussians). As such, any hypothetical gains from targeting (7) directly will always be offset against drops in the acceptance rate of the rejection sampler. 

## References



[70] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 





[66] L. Martino, V. Elvira, D. Luengo, and J. Corander. Layered adaptive importance sampling. Statistics and Computing, 27(3), May 2017. 



![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/1484dee9-8c11-4455-858f-5f99548dc420/0944e39e8da21584e508dee3aa24ff97764b7ba01797e9cbfc8059bc8cfded45.jpg)



Figure 5: Acceptance rates for evaluating the local ELBOs in each SLP for the model from Sec. 6.1. Each plot represents a separate SLP; the plot with $\mathrm { ^ { * } S L P i ^ { , } }$ corresponds to the SLP with $z = \mathrm { i }$ in Eq. (15). We can see that for all SLPs the acceptance rate approaches 1 with more iterations, confirming the mode seeking behaviour that arises when maximizing the surrogate ELBO in Eq. (11).
