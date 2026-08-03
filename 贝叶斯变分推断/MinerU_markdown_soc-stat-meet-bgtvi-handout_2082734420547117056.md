# A Beginner’s Guide to Variational Inference

Haziq Jamil 

Social Statistics 

London School of Economics and Political Science 

1 February 2018 

Social Statistics Meeting 

http://socialstats.haziqj.ml 

Outline
① Introduction
    Idea
    Comparison to EM
    Mean-field distributions
    Coordinate ascent algorithm
② Examples
    Univariate Gaussian
    Gaussian mixtures
③ Discussion
    Exponential families
    Zero-forcing vs Zero-avoiding
    Quality of approximation
    Advanced topics 

## Introduction

• Consider a statistical model where we have observations $\mathbf { y } = ( y _ { 1 } , \dots , y _ { n } )$ and also some latent variables $\pmb { z } = ( z _ { 1 } , \dots , z _ { m } )$ 

• Want to evaluate the intractable integral 

$$
\mathcal {I} := \int p (\mathbf {y} | \mathbf {z}) p (\mathbf {z}) d \mathbf {z}
$$

I Bayesian posterior analysis 

I Random efects models 

I Mixture models 

• Variational inference approximates the “posterior” $p ( \boldsymbol { z } | \mathbf { y } )$ by a tractably close distribution in the Kullback-Leibler sense. 

• Advantages: 

I Computationally fast 

I Convergence easily assessed 

I Works well in practice 

## In the literature


Google Scholar results for 'variational inference


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/211e74b86ccd7a4c16ef9113a570d6faf109c22f7a58037116fcbb0b032e8429.jpg)


• Well known in the machine learning community. 

• In social statistics: 

<sup>I</sup> E. A. Erosheva et al. (2007). “Describing disability through individual-leve mixture models for multivariate binary data”. Ann. Appl. Stat, 1.2, p. 346 

<sup>I</sup> J. Grimmer (2010). “An introduction to Bayesian inference via variational approximations”. Political Analysis 19.1, pp. 32–47 

Y. S. Wang et al. (2017). “A variational EM method for mixed membership models with multivariate rank data: An analysis of public policy preferences”. arXiv: 1512.08731 

## Recommended texts



• M. J. Beal and Z. Ghahramani (2003). “The variational Bayesian EM algorithm for incomplete data: With application to scoring graphical model structures”. In: Bayesian Statistics 7. Proceedings of the Seventh Valencia International Meeting. Ed. by J. M. Bernardo et al. Oxford: Oxford University Press, pp. 453–464 





• C. M. Bishop (2006). Pattern Recognition and Machine Learning. Springer 





• K. P. Murphy (2012). Machine Learning: A Probabilistic Perspective. The MIT Press 



• D. M. Blei et al. (2017). “Variational inference: A review for statisticians”. J. Am. Stat. Assoc, to appear 

Idea 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/07b77c2196f9f7153ac35ab40451dd044ebb1ea8d7c2a6f21ec32692a7849415.jpg)


• Minimise Kullback-Leibler divergence (using calculus of variations) 

$$
\operatorname{KL} (q \| p) = - \int \log \frac {p (z | y)}{q (z)} q (z) d z.
$$

• ISSUE: $\mathsf { K L } ( q \| p )$ is intractable. 

D. M. Blei (2017). “Variational Inference: Foundations and Innovations”. URL: 

## The Evidence Lower Bound (ELBO)

• Let $q ( \boldsymbol { z } )$ be some density function to approximate $p ( \boldsymbol { z } | \mathbf { y } )$ . Then the log-marginal density can be decomposed as follows: 

$$
\begin{array}{l} \log p (\mathbf {y}) = \log p (\mathbf {y}, \mathbf {z}) - \log p (\mathbf {z} | \mathbf {y}) \\ \qquad = \int \left\{\log \frac {p (\mathbf {y} , \mathbf {z})}{q (\mathbf {z})} - \log \frac {p (\mathbf {z} | \mathbf {y})}{q (\mathbf {z})} \right\} q (\mathbf {z}) d \mathbf {z} \\ \qquad = \mathcal {L} (q) + K L (q \| p) \\ \qquad \geq \mathcal {L} (q) \end{array}
$$

• $\mathcal { L }$ is referred to as the “lower-bound”, and it serves as a surrogate function to the marginal. 

• Maximising $\mathcal { L } ( q )$ is equivalent to minimising $\mathsf { K L } ( q \| p )$ 

• ISSUE: $\mathcal { L } ( q )$ is (generally) not convex. 

## Comparison to the EM algorithm

• Suppose for this part, the marginal density $p ( \mathsf { y } | \boldsymbol { \theta } )$ depends on parameters $\theta$ 

• In the EM algorithm, the true posterior density is used, i.e. $q ( \mathbf { \boldsymbol { z } } ) \equiv p ( \mathbf { \boldsymbol { z } } | \mathbf { \boldsymbol { y } } , \theta )$ 

• Thus, 

$$
\begin{array}{l} \log p (\mathbf {y} | \theta) = \int \left\{\log \frac {p (\mathbf {y} , \mathbf {z} | \theta)}{p (\mathbf {z} | \mathbf {y} , \theta)} - \log \frac {p (\mathbf {z} | \mathbf {y} , \theta)}{p (\mathbf {z} | \mathbf {y} , \theta)} \right\} p (\mathbf {z} | \mathbf {y}, \theta^ {(t)})   \mathrm{d} \mathbf {z} \\ = E _ {\theta^ {(t)}} [ \log p (\mathbf {y}, \mathbf {z} | \theta) ] - E _ {\theta^ {(t)}} [ \log p (\mathbf {z} | \mathbf {y}, \theta) ] \\ = Q (\theta | \theta^ {(t)}) + \text {entropy.} \end{array}
$$

• Minimising the KL divergence corresponds to the E-step. 

• For any θ, $\theta _ { i }$ 

$$
\begin{array}{c} \log p (\mathbf {y} | \theta) - \log p (\mathbf {y} | \theta^ {(t)}) = Q (\theta | \theta^ {(t)}) - Q (\theta^ {(t)} | \theta^ {(t)}) + \Delta \text { entropy } \\ \geq Q (\theta | \theta^ {(t)}) - Q (\theta^ {(t)} | \theta^ {(t)}). \end{array}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/aadbf29133648da8b0c9e9ed8923bf53eea72f49cdeed03fbfd19ebfe53005ac.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/fcc544f3f413c1674a6d00c761b241d52fece82716c9ea093771bed4adfbc03f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/3b7970a9331ccc720484ed306268f773352c861079baa8f5714c8f66d37be93c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/0b729f79445e43566a81bb16537fe2ffb1e3612472d7e7de02989c7a4ad28981.jpg)


## Factorised distributions (Mean-field theory)

• Maximising $\mathcal { L }$ over all possible q not feasible. Need some restrictions, but only to achieve tractability. 

• Suppose we partition elements of z into M disjoint groups $\mathbf { z } = ( \mathbf { z } ^ { ( 1 ) } , \dots , \mathbf { z } ^ { ( M ) } )$ , and assume 

$$
q (\mathbf {z}) = \prod_ {j = 1} ^ {M} q _ {j} (\mathbf {z} ^ {(j)}).
$$

• Under this restriction, the solution to arg max $_ q \mathcal { L } ( q )$ is 

$$
\tilde {q} _ {j} (\mathbf {z} ^ {(j)}) \propto \exp \left(\mathrm{E} _ {- j} [ \log p (\mathbf {y}, \mathbf {z}) ]\right)\tag{1}
$$

for $j \in \{ 1 , \dots , m \}$ 

• In practice, these unnormalised densities are of recognisable form (especially if conjugacy is considered). 

## Coordinate ascent mean-field variational inference (CAVI)

• The optimal distributions are coupled with another, i.e. each $\tilde { q } _ { j } ( \boldsymbol { z } ^ { ( j ) } )$ depends on the optimal moments of $\mathbf { z } ^ { ( k ) } , k \in \{ 1 , \dots , M : k \neq j \}$ 

• One way around this to employ an iterative procedure. 

• Assess convergence by monitoring the lower bound 

$$
\mathcal {L} (q) = \mathrm{E} _ {q} [ \log p (\mathbf {y}, \mathbf {z}) ] - \mathrm{E} _ {q} [ \log q (\mathbf {z}) ].
$$

Algorithm 1 CAVI
1: initialise Variational factors $q_{j}(\mathbf{z}^{(j)})$ 2: while $\mathcal{L}(q)$ not converged do
3:    for $j=1,\ldots,M$ do
4: $\log q_{j}(\mathbf{z}^{(j)})\leftarrow E_{-j}[\log p(\mathbf{y},\mathbf{z})]+\text{const.}$ ▷ from (1)
5:    end for
6: $\mathcal{L}(q)\leftarrow E_{q}[\log p(\mathbf{y},\mathbf{z})]-E_{q}[\log q(\mathbf{z})]$ 7: end while
8: return $\tilde{q}(\mathbf{z})=\prod_{j=1}^{M}\tilde{q}_{j}(\mathbf{z}^{(j)})$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/be8a6c633c80d492e2f285501447c39618a56578f5807f4121e21f9378c37617.jpg)


## 1 Introduction

## 2 Examples

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/e2e8d16af84dadea4a316df519ae62a77c71d87b7172bdf725a924dd13d84679.jpg)


## 3 Discussion

Estimation of a 1-dim Gaussian mean and variance 

• GOAL: Bayesian inference of mean $\mu$ and variance $\psi ^ { - 1 }$ 

$$
y _ {i} \stackrel {{\mathrm{iid}}} {{\sim}} \mathsf {N} (\mu , \psi^ {- 1})
$$

Data 

$$
\begin{array}{c} \mu | \psi \sim \mathsf {N} \left(\mu_ {0}, (\kappa_ {0} \psi) ^ {- 1}\right) \\ \psi \sim \Gamma (a _ {0}, b _ {0}) \end{array}
$$

Priors 

$$
i = 1, \dots , n
$$

• Substitute $p ( { \boldsymbol { \mu } } , \psi | \mathbf { y } )$ with the mean-field approximation 

$$
q (\mu , \psi) = q _ {\mu} (\mu) q _ {\psi} (\psi).
$$

• From (1), we can work out the solutions 

$$
\tilde {q} _ {\mu} (\mu) \equiv \mathsf {N} \left(\frac {\kappa_ {0} \mu_ {0} + n \bar {y}}{\kappa_ {0} + n}, \frac {1}{(\kappa_ {0} + n) \mathsf {E} _ {q} [ \psi ]}\right) \quad \text {and} \quad \tilde {q} _ {\psi} (\psi) \equiv \Gamma (\tilde {a}, \tilde {b})
$$

$$
\tilde {a} = a _ {0} + \frac {n}{2} \qquad \tilde {b} = b _ {0} + \frac {1}{2} \mathsf {E} _ {q} \left[ \sum_ {i = 1} ^ {n} (y _ {i} - \mu) ^ {2} + \kappa_ {0} (\mu - \mu_ {0}) ^ {2} \right]
$$

## Estimation of a 1-dim Gaussian mean and variance (cont.)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/ae1aade1aee8608b97ed99fec516e5f99d8abfdfdcbbc0a5d68b312d5164e138.jpg)


## Estimation of a 1-dim Gaussian mean and variance (cont.)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/b5a16fe411e5c2c442cb01ecd3abad9da212b401e6c7320f4c4ebb2673a31a46.jpg)


## Estimation of a 1-dim Gaussian mean and variance (cont.)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/598250042eb1cfdc95ead22d82921aaa68321ddb6308c658746f0de4507d9a9f.jpg)


## Estimation of a 1-dim Gaussian mean and variance (cont.)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/f540824de3d2c26c3f6ccdf1bcac898b0085a2a431330b5b3f8771f819407952.jpg)


## Estimation of a 1-dim Gaussian mean and variance (cont.)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/e52d07d906cfd85c56bde4004e35ab975f0bc53e37d71ff4857015806be0f78f.jpg)


## Comparison of solutions

## Variational posterior

$$
\mu \sim \mathsf {N} \left(\frac {\kappa_ {0} \mu_ {0} + n \bar {y}}{\kappa_ {0} + n}, \frac {1}{(\kappa_ {0} + n) \mathsf {E} [ \psi ]}\right)
$$

$$
\psi \sim \Gamma \left(a _ {0} + \frac {n}{2}, b _ {0} + \frac {1}{2} c\right)
$$

True posterior 

$$
c = \mathsf {E} \left[ \sum_ {i = 1} ^ {n} (y _ {i} - \mu) ^ {2} + \kappa_ {0} (\mu - \mu_ {0}) ^ {2} \right]
$$

$$
\mu | \psi \sim \mathsf {N} \left(\frac {\kappa_ {0} \mu_ {0} + n \bar {y}}{\kappa_ {0} + n}, \frac {1}{(\kappa_ {0} + n) \psi}\right)
$$

$$
\psi \sim \Gamma \left(a _ {0} + \frac {n}{2}, b _ {0} + \frac {1}{2} c ^ {\prime}\right)
$$

$$
c ^ {\prime} = \sum_ {i = 1} ^ {n} (y _ {i} - \bar {y}) ^ {2} + \frac {\kappa_ {0}}{\kappa_ {0} + n} (\bar {y} - \mu_ {0}) ^ {2}
$$

$\mathsf { C o v } ( \mu , \psi ) = 0$ by design in VI solutions. 

• For this simple example, it is possible to decouple and solve explicitly 

• VI solutions leads to unbiased MLE if $\kappa _ { 0 } = \mu _ { 0 } = a _ { 0 } = b _ { 0 } = 0 .$ 


Gaussian mixture model (Old Faithful data set)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/27c88f78af162cbc10451f7a56b3e3fdfe3496dbc591081191973f30c1e4233a.jpg)


• Let $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ and assume x <sup>iid</sup>∼ $\textstyle \sum _ { k = 1 } ^ { K } \pi _ { k } \mathbb { N } _ { d } ( \mu _ { k } , \Psi _ { k } ^ { - 1 } )$ for $i = 1 , \ldots , n .$ 

Gaussian mixture model 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/293c62231dcaa8bd0d8e89d0ecdb4ab708994dd128e2adb0eb633b03c0c34a55.jpg)


$$
\begin{array}{l} p (\mathbf {x}, \mathbf {z}, \boldsymbol {\pi}, \boldsymbol {\mu}, \boldsymbol {\Psi}) \\ = p (\mathbf {x} | \mathbf {z}, \boldsymbol {\mu}, \boldsymbol {\Psi}) p (\mathbf {z} | \boldsymbol {\pi}) \\ \quad \times p (\boldsymbol {\pi}) p (\boldsymbol {\mu} | \boldsymbol {\Psi}) p (\boldsymbol {\Psi}) \\ = p (\mathbf {x} | \mathbf {z}, \boldsymbol {\mu}, \boldsymbol {\Psi}) p (\mathbf {z} | \boldsymbol {\pi}) \\ \quad \times \operatorname{Dir} _ {K} (\boldsymbol {\pi} | \alpha_ {0 1}, \ldots , \alpha_ {0 K}) \\ \quad \times \prod_ {k = 1} ^ {K} N _ {d} (\boldsymbol {\mu} _ {k} | \mathsf {m} _ {0}, (\kappa_ {0} \boldsymbol {\Psi} _ {k}) ^ {- 1}) \\ \quad \times \prod_ {k = 1} ^ {K} W i s _ {d} (\boldsymbol {\Psi} _ {k} | \mathsf {W} _ {0}, \nu_ {0}) \end{array}
$$

• Introduce $\pmb { z } _ { i } = ( z _ { i 1 } , \dots , z _ { i K } )$ , a $1 { - } \mathsf { o f } { - } K$ binary vector, where each $z _ { i k } \sim \mathsf { B e r n } ( \pi _ { k } )$ 

• Assuming $\mathbf { z } = \{ \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { n } \}$ are observed along with $\mathbf { x } = \{ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n } \}$ 

$$
p (\mathbf {x} | \mathbf {z}, \boldsymbol {\mu}, \boldsymbol {\Psi}) = \prod_ {i = 1} ^ {n} \prod_ {k = 1} ^ {K} \mathrm{N} _ {d} (\mathbf {x} _ {i} | \boldsymbol {\mu} _ {k}, \boldsymbol {\Psi} _ {k} ^ {- 1}) ^ {z _ {i k}}.
$$

## Variational inference for GMM

• Assume the mean-field posterior density 

$$
\begin{array}{c} q (z, \pi , \mu , \Psi) = q (z) q (\pi , \mu , \Psi) \\ = q (z) q (\pi) q (\mu | \Psi) q (\Psi). \end{array}
$$

Algorithm 2 CAVI for GMM details
1: initialise Variational factors $q(z)$ , $q(\pi)$ and $q(\mu, \Psi)$ 2: while $\mathcal{L}(q)$ not converged do
3: $q(z_{ik}) \leftarrow \text{Bern}(\cdot)$ 4: $q(\pi) \leftarrow \text{Dir}_K(\cdot)$ 5: $q(\mu|\Psi) \leftarrow N_d(\cdot, \cdot)$ 6: $q(\Psi) \leftarrow \text{Wis}_d(\cdot, \cdot)$ 7: $\mathcal{L}(q) \leftarrow E_q[\log p(x,z,\pi,\mu,\Psi)] - E_q[\log q(z,\pi,\mu,\Psi)]$ 8: end while
9: return $\tilde{q}(z,\pi,\mu,\Psi) = \tilde{q}(z)\tilde{q}(\pi)\tilde{q}(\mu|\Psi)\tilde{q}(\Psi)$ 

## Variational inference for GMM (cont.)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/188d430db64fea58c6295094843ba2dfe821f1253e7df8ab970c7281dafe57ee.jpg)


## Variational inference for GMM (cont.)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/70df12c965dd90eeab89ee35871544ff1d2f8730d576a9e60d80420cde85d87f.jpg)


## Variational inference for GMM (cont.)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/ed7308c5362f7016229f05eff479e85a782ca08f7023d6e7833c38c4576494d9.jpg)


## Final thoughts on variational GMM

• Similar algorithm to the EM, and therefore similar computational time. 

• Can extend to mixture of bernoullis a.k.a. latent class analysis. 

• PROS: 

I Automatic selection of number of mixture components. 

I Less pathological special cases compared to EM solutions because regularised by prior information. 

I Less sensitive to number of parameters/components. 

## • CONS:

I Hyperparameter tuning. 

## 1 Introduction

## 2 Examples

## 3 Discussion

## Exponential families

• For the mean-field variational method, suppose that each complete conditional is in the exponential family: 

$$
p (\mathbf {z} ^ {(j)} | \mathbf {z} _ {- j}, \mathbf {y}) = h (\mathbf {z} ^ {(j)}) \exp \left(\eta_ {j} (\mathbf {z} _ {- j}, \mathbf {y}) \cdot \mathbf {z} ^ {(j)} - A (\eta_ {j})\right).
$$

• Then, from (1), 

$$
\begin{array}{l} \tilde {q} _ {j} (\mathbf {z} ^ {(j)}) \propto \exp \left(\operatorname{E} _ {- j} [ \log p (\mathbf {z} ^ {(j)} | \mathbf {z} _ {- j}, \mathbf {y}) ]\right) \\ = \exp \left(\log h (\mathbf {z} ^ {(j)}) + \operatorname{E} [ \eta_ {j} (\mathbf {z} _ {- j}, \mathbf {y}) ] \cdot \mathbf {z} ^ {(j)} - \operatorname{E} [ A (\eta_ {j}) ]\right) \\ \propto h (\mathbf {z} ^ {(j)}) \exp \left(\operatorname{E} [ \eta_ {j} (\mathbf {z} _ {- j}, \mathbf {y}) ] \cdot \mathbf {z} ^ {(j)}\right) \end{array}
$$

is also in the same exponential family. 

• C.f. Gibbs conditional densities. 

• ISSUE: What if not in exponential family? Importance sampling or Metropolis sampling. 

## Non-convexity of ELBO

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/741bdc3489bb9acfa5a1180a5579124001d77dc778433a7da9bb03a2ea80c016.jpg)



• CAVI only guarantees converges to a local optimum.


• Multiple local optima may exist. 

## Zero-forcing vs Zero-avoiding

• Back to the KL divergence: 

$$
\operatorname{KL} (q \| p) = \int \log \frac {q (z)}{p (z | y)} q (z) d z
$$

$\mathsf { K L } ( q \| p )$ is large when $p ( \boldsymbol { z } | \mathbf { y } )$ is close to zero, unless $q ( \boldsymbol { z } )$ is also close to zero (zero-forcing). 

• What about other measures of closeness? For instance, 

$$
\operatorname{KL} (p \| q) = \int \log \frac {p (z | y)}{q (z | y)} p (z | y) d z.
$$

• This gives the Expectation Propagation (EP) algorithm. 

• It is zero-avoiding, because $\mathsf { K L } ( p \| q )$ is small when both $p ( \boldsymbol { z } | \mathbf { y } )$ and $q ( \boldsymbol { z } )$ are non-zero. 


Zero-forcing vs Zero-avoiding (cont.)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/a9cb860f04bdae70e6ebc44a3721b91214a3443405feb9d6abd7150ae836fb18.jpg)



Zero-forcing vs Zero-avoiding (cont.)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/a36f03517ed522a6f13c4a37868d77e1cfe199ef35065e962c8bb04312a43622.jpg)



Zero-forcing vs Zero-avoiding (cont.)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/9f3e0ac89dd23be206ba63c8e4bbf240a3772aade5890bbd47fe50883a10850f.jpg)


## Distortion of higher order moments

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/8ba06f7b7d8cca31c6b1e70887c6eabeab26bac4f983846a3c37d8d12b26b3fc.jpg)


• Consider $\mathbf { z } = ( z _ { 1 } , z _ { 2 } ) ^ { \top } \sim \mathsf { N } _ { 2 } ( \pmb { \mu } , \pmb { \Psi } ^ { - 1 } ) , \mathsf { C o v } ( z _ { 1 } , z _ { 2 } ) \neq 0$ 

• Approximating $p ( z )$ by $q ( \boldsymbol { z } ) = q ( z _ { 1 } ) q ( z _ { 2 } )$ yields 

$$
\tilde {q} (z _ {1}) = \mathsf {N} (z _ {1} | \mu_ {1}, \psi_ {1 1} ^ {- 1}) \text {and} \tilde {q} (z _ {2}) = \mathsf {N} (z _ {2} | \mu_ {2}, \psi_ {2 2} ^ {- 1})
$$

and by definition, $\mathsf { C o v } ( z _ { 1 } , z _ { 2 } ) = 0$ under $\tilde { q } .$ . 

• This leads to underestimation of variances (widely reported in the literature—Zhao and Marriott 2013). 

## Quality of approximation

• Variational inference converges to a diferent optimum than ML, except for certain models (Gunawardana and Byrne 2005). 

• But not much can be said about the quality of approximation. 

• Statistical properties not well understood—what is its statistical profile relative to the exact posterior? 

• Speed trumps accuracy? 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/c8a4394995520fe10e20caac23a5c1fd37ff67e18d3a12858db73da0c901dfea.jpg)


## Advanced topics

## • Local variational bounds

I Not using the mean-field assumption. 

I Instead, find a bound for the marginalising integral I. 

I Used for Bayesian logistic regression as follows: 

$$
\mathcal {I} = \int \operatorname{expit} (x ^ {\top} \beta) p (\beta) d \beta \geq \int f (x ^ {\top} \beta , \xi) p (\beta) d \beta .
$$

## • Stochastic variational inference

I Use ideas from stochastic optimisation—gradient based improvement of ELBO from subsamples of the data. 

I Scales to massive data. 

## • Black box variational inference

I Beyond exponential families and model-specific derivations. 

## Thank you!



Beal, M. J. and Z. Ghahramani (2003). “The variational Bayesian EM algorithm for incomplete data: With application to scoring graphical model structures”. In: Bayesian Statistics 7. Proceedings of the Seventh Valencia International Meeting. Ed. by J. M. Bernardo, A. P. Dawid, J. O. Berger, M. West, D. Heckerman, M. Bayarri, and A. F. Smith. Oxford: Oxford University Press, pp. 453–464. 





Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer. 





Blei, D. M. (2017). “Variational Inference: Foundations and Innovations”. URL: https://simons.berkeley.edu/talks/david-blei-2017-5-1. 





Blei, D. M., A. Kucukelbir, and J. D. McAulife (2017). “Variationa inference: A review for statisticians”. Journal of the American Statistical Association, to appear. 



## References II



Erosheva, E. A., S. E. Fienberg, and C. Joutard (2007). “Describing disability through individual-level mixture models for multivariate binary data”. Annals of Applied Statistics, 1.2, p. 346. 





Grimmer, J. (2010). “An introduction to Bayesian inference via variational approximations”. Political Analysis 19.1, pp. 32–47. 





Gunawardana, A. and W. Byrne (2005). “Convergence theorems for generalized alternating minimization procedures”. Journal of Machine Learning Research 6, pp. 2049–2073. 





Kass, R. and A. Raftery (1995). “Bayes Factors”. Journal of the American Statistical Association 90.430, pp. 773–795. 





Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. The MIT Press. 



## References III



Wang, Y. S., R. Matsueda, and E. A. Erosheva (2017). “A variational EM method for mixed membership models with multivariate rank data: An analysis of public policy preferences”. arXiv: 1512.08731. 





Zhao, H. and P. Marriott (2013). “Diagnostics for variational Bayes approximations”. arXiv: 1309.5117. 



## Additional material

The variational principle 

Laplace’s method 

## The variational principle

• Name derived from calculus of variations which deals with maximising or minimising functionals. 

Functions 

$$
p: \theta \mapsto \mathbb {R}
$$

(standard calculus) 

Functionals 

$$
\mathcal {H}: p \mapsto \mathbb {R}
$$

(variational calculus) 

• Using standard calculus, we can solve 

$$
\arg \max _ {\theta} p (\theta) =: \hat {\theta}
$$

e.g. $p$ is a likelihood function, and $\hat { \theta }$ is the ML estimate. 

• Using variational calculus, we can solve 

$$
\arg \max _ {p} \mathcal {H} (p) =: \tilde {p}
$$

e.g. H is the entropy $\begin{array} { r } { \mathcal { H } = - \int p ( \boldsymbol { x } ) \log p ( \boldsymbol { x } ) \mathrm { d } \boldsymbol { x } } \end{array}$ , and $\tilde { p }$ is the entropy maximising distribution. 

## Laplace’s method

• Interested in $p ( \mathbf { f } | \mathbf { y } ) \propto p ( \mathbf { y } | \mathbf { f } ) p ( \mathbf { f } ) = : e ^ { Q ( \mathbf { f } ) }$ , with normalising constant $\begin{array} { r } { p ( \mathbf { y } ) = \int e ^ { Q ( \mathbf { f } ) } } \end{array}$ df. The Taylor expansion of $Q$ about its mode $\tilde { \pmb f }$ 

$$
Q (\mathbf {f}) \approx Q (\tilde {\mathbf {f}}) - \frac {1}{2} (\mathbf {f} - \tilde {\mathbf {f}}) ^ {\top} \mathbf {A} (\mathbf {f} - \tilde {\mathbf {f}})
$$

is recognised as the logarithm of an unnormalised Gaussian density, with $\mathsf { A } = - \mathsf { D } ^ { 2 } Q ( \mathsf { f } )$ being the negative Hessian of Q evaluated at $\tilde { \mathbf { f } } .$ 

• The posterior $p ( \mathbf { f } | \mathbf { y } )$ is approximated by $\mathsf { N } ( \tilde { \mathsf { f } } , \mathsf { A } ^ { - 1 } )$ , and the margina by 

$$
p (\mathbf {y}) \approx (2 \pi) ^ {n / 2} | \mathbf {A} | ^ {- 1 / 2} p (\mathbf {y} | \tilde {\mathbf {f}}) p (\tilde {\mathbf {f}})
$$

• Won’t scale with large n; dificult to find modes in high dimensions. 

## Comparison of approximations (density)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/8d19989aeeec11f1db728a13d60eb41e817b3f1795a7946f315eed260bb429d5.jpg)


Deviance (−2 x Log−density) 

## Comparison of approximations (deviance)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-30/15a75679-78e5-4f07-b4de-a7cbe8894aa8/937013dc0fa9b04d423bc7ed2256ed48f2431aee22dee19da6512396763e9ace.jpg)


Variational solutions to Gaussian mixture model 

## Variational M-step

$$
\tilde {q} (\mathbf {z}) = \prod_ {i = 1} ^ {n} \prod_ {k = 1} ^ {K} r _ {i k} ^ {z _ {i k}}, \quad r _ {i k} = \rho_ {i k} / \sum_ {k = 1} ^ {K} \rho_ {i k}
$$

$$
\begin{array}{c} \log \rho_ {i k} = \mathsf {E} [ \log \pi_ {k} ] + \frac {1}{2} \mathsf {E} \left[ \log | \boldsymbol {\Psi} _ {k} | \right] - \frac {d}{2} \log 2 \pi \\ - \frac {1}{2} \mathsf {E} \left[ (\mathbf {x} _ {i} - \boldsymbol {\mu} _ {k}) ^ {\top} \boldsymbol {\Psi} _ {k} (\mathbf {x} _ {i} - \boldsymbol {\mu} _ {k}) \right] \end{array}
$$

## Variational E-step

$$
\tilde {q} (\pi_ {1}, \dots , \pi_ {K}) = \operatorname{Dir} _ {K} (\boldsymbol {\pi} | \tilde {\boldsymbol {\alpha}}), \quad \tilde {\alpha} _ {k} = \alpha_ {0 k} + \sum_ {i = 1} ^ {n} r _ {i k}
$$

$$
\tilde {q} (\boldsymbol {\mu}, \boldsymbol {\Psi}) = \prod_ {k = 1} ^ {K} \mathsf {N} _ {d} \left(\boldsymbol {\mu} _ {k} | \tilde {\mathbf {m}} _ {k}, (\tilde {\kappa} _ {k} \boldsymbol {\Psi} _ {k}) ^ {- 1}\right) \operatorname{Wis} _ {d} \left(\boldsymbol {\Psi} _ {k} | \tilde {\mathbf {W}} _ {k}, \tilde {\nu} _ {k}\right)
$$

Variational solutions to Gaussian mixture model (cont.) 

$$
\tilde {\kappa} _ {k} = \kappa_ {0} + \sum_ {i = 1} ^ {n} r _ {i k}
$$

$$
\tilde {\mathbf {m}} _ {k} = \left(\kappa_ {0} \mathbf {m} _ {0} + \sum_ {i = 1} ^ {n} r _ {i k} \mathbf {x} _ {i}\right) / \tilde {\kappa} _ {k}
$$

$$
\mathsf {W} _ {k} ^ {- 1} = \mathsf {W} _ {0} ^ {- 1} + \sum_ {i = 1} ^ {n} r _ {i k} (\mathbf {x} _ {i} - \bar {\mathbf {x}} _ {k}) (\mathbf {x} _ {i} - \bar {\mathbf {x}} _ {k}) ^ {\top}
$$

$$
\bar {\mathbf {x}} _ {k} = \sum_ {i = 1} ^ {n} r _ {i k} \mathbf {x} _ {i} \bigg / \sum_ {i = 1} ^ {n} r _ {i k}
$$

$$
\nu_ {k} = \nu_ {0} + \sum_ {i = 1} ^ {n} r _ {i k}
$$

Also useful 

$$
\mathsf {E} \left[ \left(\mathbf {x} _ {i} - \boldsymbol {\mu} _ {k}\right) ^ {\top} \boldsymbol {\Psi} _ {k} \left(\mathbf {x} _ {i} - \boldsymbol {\mu} _ {k}\right) \right] = d / \tilde {\kappa} _ {k} + \nu_ {k} \left(\mathbf {x} _ {i} - \tilde {\mathbf {m}} _ {k}\right) ^ {\top} \tilde {\mathbf {W}} _ {k} \left(\mathbf {x} _ {i} - \tilde {\mathbf {m}} _ {k}\right)
$$

$$
\mathsf {E} [ \log \pi_ {k} ] = \sum_ {i = 1} ^ {d} \psi \left(\frac {\nu_ {k} + 1 - i}{2}\right) + d \log 2 + \log | \tilde {\mathbf {W}} _ {k} |
$$

$\begin{array} { r } { \mathsf { E } \left[ \log | \Psi _ { k } | \right] = \psi ( \tilde { \alpha } _ { k } ) - \psi \big ( \sum _ { k = 1 } ^ { K } \tilde { \alpha } _ { k } \big ) } \end{array}$ $\psi ( \cdot )$ is the digamma function 