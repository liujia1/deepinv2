# Elucidating the Design Space of Diffusion-Based Generative Models

Tero Karras NVIDIA 

Miika Aittala NVIDIA 

Timo Aila NVIDIA 

Samuli Laine NVIDIA 

## Abstract

We argue that the theory and practice of diffusion-based generative models are currently unnecessarily convoluted and seek to remedy the situation by presenting a design space that clearly separates the concrete design choices. This lets us identify several changes to both the sampling and training processes, as well as preconditioning of the score networks. Together, our improvements yield new state-of-the-art FID of 1.79 for CIFAR-10 in a class-conditional setting and 1.97 in an unconditional setting, with much faster sampling (35 network evaluations per image) than prior designs. To further demonstrate their modular nature, we show that our design changes dramatically improve both the efficiency and quality obtainable with pre-trained score networks from previous work, including improving the FID of a previously trained ImageNet-64 model from 2.07 to near-SOTA 1.55, and after re-training with our proposed improvements to a new SOTA of 1.36. 

## 1 Introduction

Diffusion-based generative models [46] have emerged as a powerful new framework for neural image synthesis, in both unconditional [16, 37, 49] and conditional [17, 36, 37, 39, 40, 42, 43, 49] settings, even surpassing the quality of GANs [13] in certain situations [9]. They are also rapidly finding use in other domains such as audio [28, 38] and video [19] generation, image segmentation [4, 57] and language translation [35]. As such, there is great interest in applying these models and improving them further in terms of image/distribution quality, training cost, and generation speed. 

The literature on these models is dense on theory, and derivations of sampling schedule, training dynamics, noise level parameterization, etc., tend to be based as directly as possible on theoretical frameworks, which ensures that the models are on a solid theoretical footing. However, this approach has a danger of obscuring the available design space — a proposed model may appear as a tightly coupled package where no individual component can be modified without breaking the entire system. 

As our first contribution, we take a look at the theory behind these models from a practical standpoint, focusing more on the “tangible” objects and algorithms that appear in the training and sampling phases, and less on the statistical processes from which they might be derived. The goal is to obtain better insights into how these components are linked together and what degrees of freedom are available in the design of the overall system. We focus on the broad class of models where a neural network is used to model the score [22] of a noise level dependent marginal distribution of the training data corrupted by Gaussian noise. Thus, our work is in the context of denoising score matching [54]. 

Our second set of contributions concerns the sampling processes used to synthesize images using diffusion models. We identify the best-performing time discretization for sampling, apply a higherorder Runge–Kutta method for the sampling process, evaluate different sampler schedules, and analyze the usefulness of stochasticity in the sampling process. The result of these improvements is a significant drop in the number of sampling steps required during synthesis, and the improved sampler can be used as a drop-in replacement with several widely used diffusions models [37, 49]. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/eda3f1dd826645182b07d72c26b0c0485ff690de6f2c1a7567ea519951750b75.jpg)



Figure 1: Denoising score matching on CIFAR-10. (a) Images from the training set corrupted with varying levels of additive Gaussian noise. High levels of noise lead to oversaturated colors; we normalize the images for cleaner visualization. (b) Optimal denoising result from minimizing Eq. 2 analytically (see Appendix B.3). With increasing noise level, the result approaches dataset mean.


The third set of contributions focuses on the training of the score-modeling neural network. While we continue to rely on the commonly used network architectures (DDPM [16], NCSN [48]), we provide the first principled analysis of the preconditioning of the networks’ inputs, outputs, and loss functions in a diffusion model setting and derive best practices for improving the training dynamics. We also suggest an improved distribution of noise levels during training, and note that non-leaking augmentation [25] — typically used with GANs — is beneficial for diffusion models as well. 

Taken together, our contributions enable significant improvements in result quality, e.g., leading to record FIDs of 1.79 for CIFAR-10 [29] and 1.36 for ImageNet [8] in 64×64 resolution. With all key ingredients of the design space explicitly tabulated, we believe that our approach will allow easier innovation on the individual components, and thus enable more extensive and targeted exploration of the design space of diffusion models. Our implementation and pre-trained models are available at https://github.com/NVlabs/edm 

## 2 Expressing diffusion models in a common framework

Let us denote the data distribution by $p _ { \mathrm { d a t a } } ( \pmb { x } )$ , with standard deviation $\sigma _ { \mathrm { d a t a } }$ , and consider the family of mollified distributions $p ( { \pmb x } ; { \boldsymbol \sigma } )$ obtained by adding i.i.d. Gaussian noise of standard deviation σ to the data. For $\sigma _ { \mathrm { m a x } } \gg \sigma _ { \mathrm { d a t a } } , p ( { \pmb x } ; \sigma _ { \mathrm { m a x } } )$ is practically indistinguishable from pure Gaussian noise. The idea of diffusion models is to randomly sample a noise image $\mathbf { \boldsymbol { x } } _ { 0 } \sim \mathcal { N } ( \mathbf { \boldsymbol { 0 } } , \mathbf { \dot { \sigma } } _ { \operatorname* { m a x } } ^ { 2 } \mathbf { I } )$ , and sequentially denoise it into images $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ with noise levels $\sigma _ { 0 } = \sigma _ { \mathrm { m a x } } > \sigma _ { 1 } > \cdot \cdot \cdot > \sigma _ { N } = 0$ so that at each noise level $\pmb { x } _ { i } \sim p ( \pmb { x } _ { i } ; \sigma _ { i } )$ . The endpoint $\mathbf { \mathcal { x } } _ { N }$ of this process is thus distributed according to the data. 

Song et al. [49] present a stochastic differential equation (SDE) that maintains the desired distribution p as sample x evolves over time. This allows the above process to be implemented using a stochastic solver that both removes and adds noise at each iteration. They also give a corresponding “probability flow” ordinary differential equation (ODE) where the only source of randomness is the initial noise image $\scriptstyle { \pmb x } _ { 0 }$ . Contrary to the usual order of treatment, we begin by examining the ODE, as it offers a fruitful setting for analyzing sampling trajectories and their discretizations. The insights carry over to stochastic sampling, which we reintroduce as a generalization in Section 4. 

ODE formulation. A probability flow ODE [49] continuously increases or reduces noise level of the image when moving forward or backward in time, respectively. To specify the ODE, we must first choose a schedule $\sigma ( t )$ that defines the desired noise level at time t. For example, setting $\sigma ( t ) \propto \sqrt { t }$ is mathematically natural, as it corresponds to constant-speed heat diffusion [12]. However, we will show in Section 3 that the choice of schedule has major practical implications and should not be made on the basis of theoretical convenience. 

The defining characteristic of the probability flow ODE is that evolving a sample $\pmb { x } _ { a } \sim p \big ( \pmb { x } _ { a } ; \sigma ( t _ { a } ) \big )$ from time $t _ { a }$ to $t _ { b }$ (either forward or backward in time) yields a sample $\pmb { x } _ { b } \sim p \big ( \pmb { x } _ { b } ; \sigma ( t _ { b } ) \big )$ . Following previous work [49], this requirement is satisfied (see Appendix B.1 and B.2) by 

$$
\mathrm{d} \boldsymbol {x} = - \dot {\sigma} (t) \sigma (t) \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma (t)) \mathrm{d} t,\tag{1}
$$

where the dot denotes a time derivative. $\nabla _ { \pmb { x } } \log p ( \pmb { x } ; \sigma )$ is the scorefunction [22], a vector field that points towards higher density of data at a given noise level. Intuitively, an infinitesimal forward step of this ODE nudges the sample away from the data, at a rate that depends on the change in noise level. Equivalently, a backward step nudges the sample towards the data distribution. 

Denoising score matching. The score function has the remarkable property that it does not depend on the generally intractable normalization constant of the underlying density function $p ( \pmb { x } ; \sigma ) \ [ 2 2 ]$ and thus can be much easier to evaluate. Specifically, if $D ( \pmb { x } ; \sigma )$ is a denoiser function that minimizes the expected $L _ { 2 }$ denoising error for samples drawn from $p _ { \mathrm { d a t a } }$ separately for every σ, i.e., 


Table 1: Specific design choices employed by different model families. N is the number of ODE solver iterations that we wish to execute during sampling. The corresponding sequence of time steps is $\{ t _ { 0 } , t _ { 1 } , \ldots , t _ { N } \}$ , where $t _ { N } = 0$ . If the model was originally trained for specific choices of N and $\{ t _ { i } \}$ , the originals are denoted by M and $\{ u _ { j } \}$ , respectively. The denoiser is defined as $D _ { \theta } ( { \pmb x } ; \sigma ) = c _ { \mathrm { s k i p } } ( \sigma ) { \pmb x } + c _ { \mathrm { o u t } } ( \sigma ) F _ { \theta } \left( c _ { \mathrm { i n } } ( \sigma ) { \pmb x } ; c _ { \mathrm { n o i s e } } ( \sigma ) \right)$ ; F represents the raw neural network layers.


<table><tr><td colspan="2"></td><td>VP [49]</td><td>VE [49]</td><td>iDDPM [37] + DDIM [47]</td><td>Ours (“EDM”)</td></tr><tr><td colspan="6">Sampling (Section 3)</td></tr><tr><td>ODE solver</td><td></td><td>Euler</td><td>Euler</td><td>Euler</td><td><eq>2^{nd}</eq> order Heun</td></tr><tr><td>Time steps</td><td><eq>t_{i} &lt; N</eq></td><td><eq>1 + \frac{i}{N-1}(\epsilon_s - 1)</eq></td><td><eq>\sigma_{\max}^2 (\sigma_{\min}^2/\sigma_{\max}^2)^{\frac{i}{N-1}}</eq></td><td><eq>u_{\lfloor j_0 + \frac{M-1-j_0}{N-1}i+\frac{1}{2}\rfloor}</eq>, where <eq>u_M = 0</eq> <eq>u_{j-1}=\sqrt{\frac{u_j^2+1}{\max(\bar{\alpha}_{j-1}/\bar{\alpha}_j,C_1)}-1}</eq></td><td><eq>(\sigma_{\max}^{\frac{1}{\rho}} + \frac{i}{N-1}(\sigma_{\min}^{\frac{1}{\rho}}-\sigma_{\max}^{\frac{1}{\rho}}))^{\rho}</eq></td></tr><tr><td>Schedule</td><td><eq>\sigma(t)</eq></td><td><eq>\sqrt{e^{\frac{1}{2}\beta_d t^2+\beta_{\min}t}-1}</eq></td><td><eq>\sqrt{t}</eq></td><td><eq>t</eq></td><td><eq>t</eq></td></tr><tr><td>Scaling</td><td><eq>s(t)</eq></td><td><eq>1/\sqrt{e^{\frac{1}{2}\beta_d t^2+\beta_{\min}t}}</eq></td><td>1</td><td>1</td><td>1</td></tr><tr><td colspan="6">Network and preconditioning (Section 5)</td></tr><tr><td colspan="2">Architecture of <eq>F_\theta</eq></td><td>DDPM++</td><td>NCSN++</td><td>DDPM</td><td>(any)</td></tr><tr><td>Skip scaling</td><td><eq>c_{\text{skip}}(\sigma)</eq></td><td>1</td><td>1</td><td>1</td><td><eq>\sigma_{{data}}^2/(\sigma^2+\sigma_{{data}}^2)</eq></td></tr><tr><td>Output scaling</td><td><eq>c_{\text{out}}(\sigma)</eq></td><td><eq>-\sigma</eq></td><td><eq>\sigma</eq></td><td><eq>-\sigma</eq></td><td><eq>\sigma \cdot \sigma_{\text{data}}/\sqrt{\sigma_{{data}}^2+\sigma^2}</eq></td></tr><tr><td>Input scaling</td><td><eq>c_{\text{in}}(\sigma)</eq></td><td><eq>1/\sqrt{\sigma^2+1}</eq></td><td>1</td><td><eq>1/\sqrt{\sigma^2+1}</eq></td><td><eq>1/\sqrt{\sigma^2+\sigma_{{data}}^2}</eq></td></tr><tr><td>Noise cond.</td><td><eq>c_{\text{noise}}(\sigma)</eq></td><td><eq>(M-1)\sigma^{-1}(\sigma)</eq></td><td><eq>\ln(\frac{1}{2}\sigma)</eq></td><td><eq>M-1-\arg\min_j|u_j-\sigma|</eq></td><td><eq>\frac{1}{4}\ln(\sigma)</eq></td></tr><tr><td colspan="6">Training (Section 5)</td></tr><tr><td colspan="2">Noise distribution</td><td><eq>\sigma^{-1}(\sigma) \sim \mathcal{U}(\epsilon_t,1)</eq></td><td><eq>\ln(\sigma) \sim \mathcal{U}(\ln(\sigma_{\min}), \ln(\sigma_{\max}))</eq></td><td><eq>\sigma = u_j</eq>, <eq>j \sim \mathcal{U}\{0,M-1\}</eq></td><td><eq>\ln(\sigma) \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)</eq></td></tr><tr><td>Loss weighting</td><td><eq>\lambda(\sigma)</eq></td><td><eq>1/\sigma^2</eq></td><td><eq>1/\sigma^2</eq></td><td><eq>1/\sigma^2</eq> (note: *)</td><td><eq>(\sigma^2+\sigma_{{data}}^2)/(\sigma \cdot \sigma_{\text{data}})^2</eq></td></tr><tr><td rowspan="3" colspan="2">Parameters</td><td><eq>\beta_d = 19.9, \beta_{\min} = 0.1</eq></td><td><eq>\sigma_{\min} = 0.02</eq></td><td><eq>\bar{\alpha}_j = \sin^2(\frac{\pi}{2}\frac{j}{M(C_2+1)})</eq></td><td><eq>\sigma_{\min} = 0.002, \sigma_{\max} = 80</eq></td></tr><tr><td><eq>\epsilon_s = 10^{-3}, \epsilon_t = 10^{-5}</eq></td><td><eq>\sigma_{\max} = 100</eq></td><td><eq>C_1 = 0.001, C_2 = 0.008</eq></td><td><eq>\sigma_{\text{data}} = 0.5, \rho = 7</eq></td></tr><tr><td><eq>M = 1000</eq></td><td></td><td><eq>M = 1000, j_0 = 8^\dagger</eq></td><td><eq>P_{\text{mean}} = -1.2, P_{\text{std}} = 1.2</eq></td></tr></table>


<sup>∗</sup> iDDPM also employs a second loss term $\overline { { L _ { \mathrm { v l b } } \ } } ^ { \dagger }$ In our tests, $j _ { 0 } = 8$ yielded better FID than $j _ { 0 } = 0$ used by iDDPM 


$$
\mathbb {E} _ {\boldsymbol {y} \sim p _ {\mathrm{data}}} \mathbb {E} _ {\boldsymbol {n} \sim \mathcal {N} (\boldsymbol {0}, \sigma^ {2} \mathbf {I})} \| D (\boldsymbol {y} + \boldsymbol {n}; \sigma) - \boldsymbol {y} \| _ {2} ^ {2}, \text {then} \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma) = (D (\boldsymbol {x}; \sigma) - \boldsymbol {x}) / \sigma^ {2},\tag{2,3}
$$

where y is a training image and n is noise. In this light, the score function isolates the noise component from the signal in x, and Eq. 1 amplifies (or diminishes) it over time. Figure 1 illustrates the behavior of ideal D in practice. The key observation in diffusion models is that $D ( \pmb { x } ; \sigma )$ can be implemented as a neural network $D _ { \theta } ( { \pmb x } ; { \boldsymbol \sigma } )$ trained according to Eq. 2. Note that $D _ { \theta }$ may include additional pre- and post-processing steps, such as scaling x to an appropriate dynamic range; we will return to such preconditioning in Section 5. 

Time-dependent signal scaling. Some methods (see Appendix C.1) introduce an additional scale schedule s(t) and consider ${ \pmb x } = s ( t ) \hat { \pmb x }$ to be a scaled version of the original, non-scaled variable xˆ. This changes the time-dependent probability density, and consequently also the ODE solution trajectories. The resulting ODE is a generalization of Eq. 1: 

$$
\mathrm{d} \boldsymbol {x} = \left[ \frac {\dot {s} (t)}{s (t)} \boldsymbol {x} - s (t) ^ {2} \dot {\sigma} (t) \sigma (t) \nabla_ {\boldsymbol {x}} \log p \left(\frac {\boldsymbol {x}}{s (t)}; \sigma (t)\right) \right] \mathrm{d} t.\tag{4}
$$

Note that we explicitly undo the scaling of x when evaluating the score function to keep the definition of $p ( { \pmb x } ; { \boldsymbol \sigma } )$ independent of s(t). 

Solution by discretization. The ODE to be solved is obtained by substituting Eq. 3 into Eq. 4 to define the point-wise gradient, and the solution can be found by numerical integration, i.e., taking finite steps over discrete time intervals. This requires choosing both the integration scheme (e.g., Euler or a variant of Runge–Kutta), as well as the discrete sampling times $\{ t _ { 0 } , t _ { 1 } , \ldots , t _ { N } \}$ . Many prior works rely on Euler’s method, but we show in Section 3 that a ${ \bf \bar { \rho } } _ { 2 ^ { \mathrm { n d } } }$ order solver offers a better computational tradeoff. For brevity, we do not provide a separate pseudocode for Euler’s method applied to our ODE here, but it can be extracted from Algorithm 1 by omitting lines 6–8. 

Putting it together. Table 1 presents formulas for reproducing deterministic variants of three earlier methods in our framework. These methods were chosen because they are widely used and achieve state-of-the-art performance, but also because they were derived from different theoretical foundations. Some of our formulas appear quite different from the original papers as indirection and recursion have been removed; see Appendix C for details. The main purpose of this reframing is to bring into light all the independent components that often appear tangled together in previous work. In our framework, there are no implicit dependencies between the components — any choices (within reason) for the individual formulas will, in principle, lead to a functioning model. In other words, changing one component does not necessitate changes elsewhere in order to, e.g., maintain the property that the model converges to the data in the limit. In practice, some choices and combinations will of course work better than others. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/879257fa7f42650b1ebc03321ef9644941a85eb786b7d84fefb401a4bca13dbe.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/f5914c4d587001c548d76feb4b0a92ce9966a58e18616acd54fd889de2e18343.jpg)



(a) Uncond. CIFAR-10, VP ODE (b) Uncond. CIFAR-10, VE ODE (c) Class-cond. ImageNet-64, DDIM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/a811c248735e77f28fe2aeea07837b058a06b6c2e21f0ab9e149fcce2e303188.jpg)



Figure 2: Comparison of deterministic sampling methods using three pre-trained models. For each curve, the dot indicates the lowest NFE whose FID is within 3% of the lowest observed FID.


## 3 Improvements to deterministic sampling

Improving the output quality and/or decreasing the computational cost of sampling are common topics in diffusion model research (e.g., [10, 24, 31, 32, 33, 37, 44, 53, 55, 56, 59]). Our hypothesis is that the choices related to the sampling process are largely independent of the other components, such as network architecture and training details. In other words, the training procedure of $D _ { \theta }$ should not dictate σ(t), s(t), and $\{ t _ { i } \}$ , nor vice versa; from the viewpoint of the sampler, $D _ { \theta }$ is simply a black box [55, 56]. We test this by evaluating different samplers on three pre-trained models, each representing a different theoretical framework and model family. We first measure baseline results for these models using their original sampler implementations, and then bring these samplers into our unified framework using the formulas in Table 1, followed by our improvements. This allows us to evaluate different practical choices and propose general improvements to the sampling process that are applicable to all models. 

We evaluate the “DDPM++ cont. $\mathbf { ( } \mathbf { V } \mathbf { P } \mathbf { ) } ^ { \mathbf { , } \mathbf { \curlyeq } }$ and “NCSN++ cont. $\mathrm { ( V E ) } ^ { \mathsf { , , } \mathsf { , } }$ models by Song et al. [49] trained on unconditional CIFAR-10 [29] at 32×32, corresponding to the variance preserving (VP) and variance exploding (VE) formulations [49], originally inspired by DDPM [16] and SMLD [48]. We also evaluate the “ADM (dropout)” model by Dhariwal and Nichol [9] trained on class-conditional Im ageNet [8] at 64×64, corresponding to the improved DDPM (iDDPM) formulation [37]. This model was trained using a discrete set of M = 1000 noise levels. Further details are given in Appendix C. 

We evaluate the result quality in terms of Fréchet inception distance (FID) [15] computed between 50,000 generated images and all available real images. Figure 2 shows FID as a function of neural function evaluations (NFE), i.e., how many times $D _ { \theta }$ is evaluated to produce a single image. Given that the sampling process is dominated entirely by the cost of $D _ { \theta }$ , improvements in NFE translate directly to sampling speed. The original deterministic samplers are shown in blue, and the reimplementations of these methods in our unified framework (orange) yield similar but consistently better results. The differences are explained by certain oversights in the original implementations as well as our more careful treatment of discrete noise levels in the case of DDIM; see Appendix C. Note that our reimplementations are fully specified by Algorithm 1 and Table 1, even though the original codebases are structured very differently from each other. 

Discretization and higher-order integrators. Solving an ODE numerically is necessarily an approximation of following the true solution trajectory. At each step, the solver introduces truncation error that accumulates over the course of N steps. The local error generally scales superlinearly with respect to step size, and thus increasing N improves the accuracy of the solution. 

The commonly used Euler’s method is a first order ODE solver with $\mathcal { O } ( h ^ { 2 } )$ local error with respect to step size h. Higher-order Runge–Kutta methods [50] scale more favorably but require multiple evaluations of $D _ { \theta }$ per step. Linear multistep methods have also been recently proposed for sampling diffusion models [31, 59]. Through extensive tests, we have found Heun’s $2 ^ { \mathrm { n d } }$ order method [2] (a.k.a. improved Euler, trapezoidal rule) — previously explored in the context of diffusion models by Jolicoeur-Martineau et al. [24] — to provide an excellent tradeoff between truncation error and NFE. As illustrated in Algorithm 1, it introduces an additional correction step for $\pmb { x } _ { i + 1 }$ to account for change in dx/dt between $t _ { i }$ and $t _ { i + 1 }$ . This correction leads to $\mathcal { O } ( h ^ { 3 } )$ local error at the cost of one additional evaluation of $D _ { \theta }$ per step. Note that stepping to $\sigma = 0$ would result in a division by zero, so we revert to Euler’s method in this case. We discuss the general family of $2 ^ { \mathrm { n d } }$ order solvers in Appendix D.2. 

Algorithm 1 Deterministic sampling using Heun's $2^{\text{nd}}$ order method with arbitrary $\sigma(t)$ and $s(t)$ .

1: procedure HEUN SAMPLER( $D_{\theta}(\boldsymbol{x};\sigma)$ , $\sigma(t)$ , $s(t)$ , $t_{i\in\{0,\ldots,N\}}$ )
2: sample $\boldsymbol{x}_{0} \sim \mathcal{N}\big(\boldsymbol{0}$ , $\sigma^{2}(t_{0}) s^{2}(t_{0}) \mathbf{I}\big)$ ▷ Generate initial sample at $t_{0}$ 3: for $i \in \{0,\ldots,N-1\}$ do ▷ Solve Eq. 4 over N time steps
4: $\boldsymbol{d}_{i} \leftarrow \left(\frac{\dot{\sigma}(t_{i})}{\sigma(t_{i})} + \frac{\dot{s}(t_{i})}{s(t_{i})}\right)\boldsymbol{x}_{i} - \frac{\dot{\sigma}(t_{i})s(t_{i})}{\sigma(t_{i})} D_{\theta}\left(\frac{\boldsymbol{x}_{i}}{s(t_{i})};\sigma(t_{i})\right)$ ▷ Evaluate d $\boldsymbol{x}$ /dt at $t_{i}$ 5: $\boldsymbol{x}_{i+1} \leftarrow \boldsymbol{x}_{i} + (t_{i+1} - t_{i})\boldsymbol{d}_{i}$ ▷ Take Euler step from $t_{i}$ to $t_{i+1}$ 6: if $\sigma(t_{i+1}) \neq 0$ then ▷ Apply $2^{\text{nd}}$ order correction unless $\sigma$ goes to zero
7: $\boldsymbol{d}_{i}' \leftarrow \left(\frac{\dot{\sigma}(t_{i+1})}{\sigma(t_{i+1})} + \frac{\dot{s}(t_{i+1})}{s(t_{i+1})}\right)\boldsymbol{x}_{i+1} - \frac{\dot{\sigma}(t_{i+1})s(t_{i+1})}{\sigma(t_{i+1})} D_{\theta}\left(\frac{\boldsymbol{x}_{i+1}}{s(t_{i+1})};\sigma(t_{i+1})\right)$ ▷ Eval. d $\boldsymbol{x}$ /dt at $t_{i+1}$ 8: $\boldsymbol{x}_{i+1} \leftarrow \boldsymbol{x}_{i} + (t_{i+1} - t_{i})\left(\frac{1}{2}\boldsymbol{d}_{i} + \frac{1}{2}\boldsymbol{d}_{i}'\right)$ ▷ Explicit trapezoidal rule at $t_{i+1}$ 9: return $\boldsymbol{x}_{N}$ ▷ Return noise-free sample at $t_{N}$ 

The time steps $\{ t _ { i } \}$ determine how the step sizes and thus truncation errors are distributed between different noise levels. We provide a detailed analysis in Appendix D.1, concluding that the step size should decrease monotonically with decreasing σ and it does not need to vary on a per-sample basis. We adopt a parameterized scheme where the time steps are defined according to a sequence of noise levels $\{ \sigma _ { i } \} , \mathrm { i . e . , } t _ { i } = \sigma ^ { - 1 } ( \sigma _ { i } )$ . We set $\sigma _ { i < N } = ( A i + B ) ^ { \rho }$ and select the constants A and B so that $\sigma _ { 0 } = \sigma _ { \operatorname* { m a x } }$ and $\sigma _ { N - 1 } = \sigma _ { \mathrm { m i n } }$ , which gives 

$$
\sigma_ {i <   N} = \big (\sigma_ {\max} ^ {\frac {1}{\rho}} + \frac {i}{N - 1} \big (\sigma_ {\min} ^ {\frac {1}{\rho}} - \sigma_ {\max} ^ {\frac {1}{\rho}} \big) \big) ^ {\rho} \quad \mathrm{and} \quad \sigma_ {N} = 0.\tag{5}
$$

Here $\rho$ controls how much the steps near $\sigma _ { \mathrm { m i n } }$ are shortened at the expense of longer steps near $\sigma _ { \mathrm { m a x } }$ Our analysis in Appendix D.1 shows that setting $\rho = 3$ nearly equalizes the truncation error at each step, but that $\rho$ in range of 5 to 10 performs much better for sampling images. This suggests that errors near $\sigma _ { \mathrm { m i n } }$ have a large impact. We set $\rho = 7$ for the remainder of this paper. 

Results for Heun’s method and Eq. 5 are shown as the green curves in Figure 2. We observe consistent improvement in all cases: Heun’s method reaches the same FID as Euler’s method with considerably lower NFE. 

Trajectory curvature and noise schedule. The shape of the ODE solution trajectories is defined by functions $\sigma ( t )$ and $s ( t )$ . The choice of these functions offers a way to reduce the truncation errors discussed above, as their magnitude can be expected to scale proportional to the curvature of dx/dt. We argue that the best choice for these functions is $\sigma ( t ) = t$ and $s ( t ) = 1$ , which is also the choice made in DDIM [47]. With this choice, the ODE of Eq. 4 simplifies to dx $/ \mathrm { d } t = \big ( { \pmb x } - D ( { \pmb x } ; t ) \big ) / t$ and σ and t become interchangeable. 

An immediate consequence is that at any x and t, a single Euler step to $t = 0$ yields the denoised image $D _ { \theta } ( \pmb { x } ; t )$ . The tangent of the solution trajectory therefore always points towards the denoiser output. This can be expected to change only slowly with the noise level, which corresponds to largely linear solution trajectories. The 1D ODE sketch of Figure 3c supports this intuition; the solution trajectories approach linear at both large and small noise levels, and have substantial curvature in only a small region in between. The same effect can be seen with real data in Figure 1b, where the change between different denoiser targets occurs in a relatively narrow σ range. With the advocated schedule, this corresponds to high ODE curvature being limited to this same range. 

The effect of setting $\sigma ( t ) = t$ and $s ( t ) = 1$ is shown as the red curves in Figure 2. As DDIM already employs these same choices, the red curve is identical to the green one for ImageNet-64. However, VP and VE benefit considerably from switching away from their original schedules. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/b039f417d9b06db9a043ec89bf4bd965659a4e8b741643fe42c77101fb44c8d5.jpg)



(a) Variance preserving ODE [49]


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/345d9604afad298f5c680327509e5f9b55c95c3d8c1717613d96eeb4576df9e9.jpg)



(b) Variance exploding ODE [49]


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/92408d7b6965ede22a4799d3889381494ad64652724afb62ca116bae40796de8.jpg)



(c) DDIM [47] / Our ODE



Figure 3: A sketch of ODE curvature in 1D where $p _ { \mathrm { d a t a } }$ is two Dirac peaks at ${ \pmb x } = \pm 1$ . Horizontal t axis is chosen to show $\sigma \in$ [0, 25] in each plot, with insets showing $\sigma \in [ 0 , 1 ]$ near the data. Example local gradients are shown with black arrows. (a) Variance preserving ODE of Song et al. [49] has solution trajectories that flatten out to horizontal lines at large σ. Local gradients start pointing towards data only at small σ. (b) Variance exploding variant has extreme curvature near data and the solution trajectories are curved everywhere. (c) With the schedule used by DDIM [47] and us, as σ increases the solution trajectories approach straight lines that point towards the mean of data. As $\sigma \to 0$ , the trajectories become linear and point towards the data manifold.


Discussion. The choices that we made in this section to improve deterministic sampling are summarized in the Sampling part of Table 1. Together, they reduce the NFE needed to reach highquality results by a large factor: 7.3× for VP, 300× for VE, and 3.2× for DDIM, corresponding to the highlighted NFE values in Figure 2. In practice, we can generate 26.3 high-quality CIFAR-10 images per second on a single NVIDIA V100. The consistency of improvements corroborates our hypothesis that the sampling process is orthogonal to how each model was originally trained. As further validation, we show results for the adaptive RK45 method [11] using our schedule as the dashed black curves in Figure 2; the cost of this sophisticated ODE solver outweighs its benefits. 

## 4 Stochastic sampling

Deterministic sampling offers many benefits, e.g., the ability to turn real images into their corresponding latent representations by inverting the ODE. However, it tends to lead to worse output quality [47, 49] than stochastic sampling that injects fresh noise into the image in each step. Given that ODEs and SDEs recover the same distributions in theory, what exactly is the role of stochasticity? 

Background. The SDEs of Song et al. [49] can be generalized [20, 58] as a sum of the probability flow ODE of Eq. 1 and a time-varying Langevin diffusion SDE [14] (see Appendix B.5): 

$$
\mathrm{d} \boldsymbol {x} _ {\pm} = \underbrace {- \dot {\sigma} (t) \sigma (t) \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x} ; \sigma (t))   \mathrm{d} t} _ {\text { probability flow ODE (Eq. 1)}} \pm \underbrace {\beta (t) \sigma (t) ^ {2} \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x} ; \sigma (t))   \mathrm{d} t} _ {\text { deterministic noise decay }} + \underbrace {\sqrt {2 \beta (t)} \sigma (t)   \mathrm{d} \omega_ {t}} _ {\text { noise injection}},\tag{6}
$$

where $\omega _ { t }$ is the standard Wiener process. $\mathrm { d } \pmb { x } _ { + }$ and dx are now separate SDEs for moving forward and backward in time, related by the time reversal formula of Anderson [1]. The Langevin term can further be seen as a combination of a deterministic score-based denoising term and a stochastic noise injection term, whose net noise level contributions cancel out. As such, $\bar { \beta } ( t )$ effectively expresses the relative rate at which existing noise is replaced with new noise. The SDEs of Song et al. [49] are recovered with the choice $\beta ( \bar { t } ) = \dot { \sigma } ( t ) / \sigma \bar { ( } t )$ , whereby the score vanishes from the forward SDE. 

This perspective reveals why stochasticity is helpful in practice: The implicit Langevin diffusion drives the sample towards the desired marginal distribution at a given time, actively correcting for any errors made in earlier sampling steps. On the other hand, approximating the Langevin term with discrete SDE solver steps introduces error in itself. Previous results [3, 24, 47, 49] suggest that non-zero $\beta ( t )$ is helpful, but as far as we can tell, the implicit choice for β(t) in Song et al. [49] enjoys no special properties. Hence, the optimal amount of stochasticity should be determined empirically. 

Our stochastic sampler. We propose a stochastic sampler that combines our $2 ^ { \mathrm { n d } }$ order deterministic ODE integrator with explicit Langevin-like “churn” of adding and removing noise. A pseudocode is given in Algorithm 2. At each step i, given the sample $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ at noise level $t _ { i } ( = \sigma ( t _ { i } ) )$ , we perform two sub-steps. First, we add noise to the sample according to a factor $\gamma _ { i } \geq 0$ to reach a higher noise level $\hat { t } _ { i } = t _ { i } + \gamma _ { i } t _ { i }$ . Second, from the increased-noise sample $\hat { \mathbf { x } } _ { i }$ , we solve the ODE backward from $\hat { t } _ { i }$ to $t _ { i + 1 }$ with a single step. This yields a sample ${ \pmb x } _ { i + 1 }$ with noise level $t _ { i + 1 }$ , and the iteration continues. We stress that this is not a general-purpose SDE solver, but a sampling procedure tailored for the specific problem. Its correctness stems from the alternation of two sub-steps that each maintain the correct distribution (up to truncation error in the ODE step). The predictor-corrector sampler of Song et al. [49] has a conceptually similar structure to ours. 

Algorithm 2 Our stochastic sampler with $\sigma(t) = t$ and $s(t) = 1$ .

1: procedure STOCHASTICSAMPLER( $D_{\theta}(\boldsymbol{x};\sigma)$ , $t_{i\in\{0,\ldots,N\}}$ , $\gamma_{i\in\{0,\ldots,N-1\}}$ , $S_{noise}$ )
2: sample $\boldsymbol{x}_0 \sim \mathcal{N}(\boldsymbol{0}, t_0^2 \boldsymbol{I})$ 3: for $i \in \{0,\ldots,N-1\}$ do
4: sample $\epsilon_i \sim \mathcal{N}(\boldsymbol{0}, S_{noise}^2 \boldsymbol{I})$ 5: $\hat{t}_i \leftarrow t_i + \gamma_i t_i$ 6: $\hat{\boldsymbol{x}}_i \leftarrow \boldsymbol{x}_i + \sqrt{\hat{t}_i^2 - t_i^2} \epsilon_i$ 7: $\boldsymbol{d}_i \leftarrow (\hat{\boldsymbol{x}}_i - D_\theta(\hat{\boldsymbol{x}}_i; \hat{t}_i)) / \hat{t}_i$ 8: $\boldsymbol{x}_{i+1} \leftarrow \hat{\boldsymbol{x}}_i + (t_{i+1} - \hat{t}_i) \boldsymbol{d}_i$ 9: if $t_{i+1} \neq 0$ then
10: $\boldsymbol{d}_i' \leftarrow (\boldsymbol{x}_{i+1} - D_\theta(\boldsymbol{x}_{i+1}; t_{i+1})) / t_{i+1}$ 11: $\boldsymbol{x}_{i+1} \leftarrow \hat{\boldsymbol{x}}_i + (t_{i+1} - \hat{t}_i) (\frac{1}{2} \boldsymbol{d}_i + \frac{1}{2} \boldsymbol{d}_i')$ 12: return $\boldsymbol{x}_N$ 

To analyze the main difference between our method and Euler–Maruyama, we first note a subtle discrepancy in the latter when discretizing Eq. 6. One can interpret Euler–Maruyama as first adding noise and then performing an ODE step, not from the intermediate state after noise injection, but assuming that x and $\sigma$ remained at the initial state at the beginning of the iteration step. In our method, the parameters used to evaluate $D _ { \theta }$ on line 7 of Algorithm $\bar { 2 }$ correspond to the state after noise injection, whereas an Euler–Maruyama -like method would use ${ \pmb x } _ { i } ; t _ { i }$ instead of $\hat { \pmb { x } } _ { i } ; \hat { t } _ { i }$ . In the limit of $\Delta _ { t }$ approaching zero there may be no difference between these choices, but the distinction appears to become significant when pursuing low NFE with large steps. 

Practical considerations. Increasing the amount of stochasticity is effective in correcting errors made by earlier sampling steps, but it has its own drawbacks. We have observed (see Appendix E.1) that excessive Langevin-like addition and removal of noise results in gradual loss of detail in the generated images with all datasets and denoiser networks. There is also a drift toward oversaturated colors at very low and high noise levels. We suspect that practical denoisers induce a slightly nonconservative vector field in Eq. 3, violating the premises of Langevin diffusion and causing these detrimental effects. Notably, our experiments with analytical denoisers (such as the one in Figure 1b) have not shown such degradation. 

If the degradation is caused by flaws in $D _ { \theta } ( \pmb { x } ; \sigma )$ , they can only be remedied using heuristic means during sampling. We address the drift toward oversaturated colors by only enabling stochasticity within a specific range of noise levels $t _ { i } \in [ S _ { \operatorname { t m i n } } , S _ { \operatorname { t m a x } } ]$ . For these noise levels, we define $\gamma _ { i } =$ $S _ { \mathrm { c h u r n } } / N$ , where $S _ { \mathrm { c h u r n } }$ controls the overall amount of stochasticity. We further clamp $\gamma _ { i }$ to never introduce more new noise than what is already present in the image. Finally, we have found that the loss of detail can be partially counteracted by setting $S _ { \mathrm { n o i s e } }$ slightly above 1 to inflate the standard deviation for the newly added noise. This suggests that a major component of the hypothesized non-conservativity of $\dot { D } _ { \boldsymbol { \theta } } ( \mathbf { \boldsymbol { x } } ; \boldsymbol { \sigma } )$ is a tendency to remove slightly too much noise — most likely due to regression toward the mean that can be expected to happen with any $L _ { 2 }$ -trained denoiser [30]. 

Evaluation. Figure 4 shows that our stochastic sampler outperforms previous samplers [24, 37, 49] by a significant margin, especially at low step counts. Jolicoeur-Martineau et al. [24] use a standard higher-order adaptive SDE solver [41] and its performance is a good baseline for such solvers in general. Our sampler has been tailored to the use case by, e.g., performing noise injection and ODE step sequentially, and it is not adaptive. It is an open question if adaptive solvers can be a net win over a well-tuned fixed schedule in sampling diffusion models. 

Through sampler improvements alone, we are able to bring the ImageNet-64 model that originally achieved FID 2.07 [9] to 1.55 that is very close to the state-of-the-art; previously, FID 1.48 has been reported for cascaded diffusion [17], 1.55 for classifier-free guidance [18], and 1.52 for StyleGAN-XL [45]. While our results showcase the potential gains achievable through sampler improvements, they also highlight the main shortcoming of stochasticity: For best results, one must make several heuristic choices — either implicit or explicit — that depend on the specific model. Indeed, we had to find the optimal values of $\mathbf { \dot { \{ } }  S _ { \mathrm { c h u r n } } , S _ { \mathrm { t m i n } } , S _ { \mathrm { t m a x } } , S _ { \mathrm { n o i s e } } \mathbf { \dot { \{ } } $ on a case-by-case basis using grid search (Appendix E.2). This raises a general concern that using stochastic sampling as the primary means of evaluating model improvements may inadvertently end up influencing the design choices related to model architecture and training. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/ed29c000f5c1dfa3fd5fc7e0ae1ca627a096ad4ec584c0cf2a7f66e6a986b95b.jpg)



(a) Uncond. CIFAR-10, VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/fcb7fa5d645be16555925dac908fa02134e5ebe5f49639045d7d3fafb794fc7a.jpg)



(b) Uncond. CIFAR-10, VE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/3bc298a381fde614928dba67cefc41e3c3c4ffd763f11cdaafad03530105cc84.jpg)



(c) Class-cond. ImageNet-64



Figure 4: Evaluation of our stochastic sampler (Algorithm 2). The purple curve corresponds to optimal choices for $\{ S _ { \mathrm { c h u r n } } , S _ { \mathrm { t m i n } } , S _ { \mathrm { t m a x } } , S _ { \mathrm { n o i s e } } \}$ ; orange, blue, and green correspond to disabling the effects of $S _ { \mathrm { t m i n , t m a x } }$ and/or $S _ { \mathrm { n o i s e } } .$ The red curves show reference results for our deterministic sampler (Algorithm 1), equivalent to setting $S _ { \mathrm { c h u r n } } = 0 .$ . The dashed black curves correspond to the original stochastic samplers from previous work: Euler–Maruyama [49] for VP, predictor-corrector [49] for ${ \mathrm { V E } } ,$ and iDDPM [37] for ImageNet-64. The dots indicate lowest observed FID.


## 5 Preconditioning and training

There are various known good practices for training neural networks in a supervised fashion. For example, it is advisable to keep input and output signal magnitudes fixed $\mathrm { t o } , \mathrm { e . g . }$ , unit variance, and to avoid large variation in gradient magnitudes on a per-sample basis [5, 21]. Training a neural network to model D directly would be far from ideal — for example, as the input ${ \pmb x } = { \pmb y } + { \pmb n }$ is a combination of clean signal y and noise $\pmb { n } \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } )$ , its magnitude varies immensely depending on noise level $\sigma .$ For this reason, the common practice is to not represent $D _ { \theta }$ as a neural network directly, but instead train a different network $F _ { \theta }$ from which $D _ { \theta }$ is derived. 

Previous methods [37, 47, 49] address the input scaling via a σ-dependent normalization factor and attempt to precondition the output by training $F _ { \theta }$ to predict n scaled to unit variance, from which the signal is then reconstructed via $D _ { \theta } ( { \pmb x } ; \sigma ) = { \pmb x } - \sigma F _ { \theta } ( \cdot )$ . This has the drawback that at large σ, the network needs to fine-tune its output carefully to cancel out the existing noise n exactly and give the output at the correct scale; note that any errors made by the network are amplified by a factor of σ. In this situation, it would seem much easier to predict the expected output $D ( \pmb { x } ; \sigma )$ directly. In the same spirit as previous parameterizations that adaptively mix signal and noise $( \mathrm { e . g . , } [ 1 0 , 4 4 , 5 3 ] )$ we propose to precondition the neural network with a σ-dependent skip connection that allows it to estimate either y or n, or something in between. We thus write $D _ { \theta }$ in the following form: 

$$
D _ {\theta} (\boldsymbol {x}; \sigma) = c _ {\text { skip }} (\sigma) \boldsymbol {x} + c _ {\text { out }} (\sigma) F _ {\theta} \left(c _ {\text { in }} (\sigma) \boldsymbol {x}; c _ {\text { noise }} (\sigma)\right),\tag{7}
$$

where $F _ { \theta }$ is the neural network to be trained, $c _ { \mathrm { s k i p } } ( \sigma )$ modulates the skip connection, $c _ { \mathrm { i n } } ( \sigma )$ and $c _ { \mathrm { o u t } } ( \sigma )$ scale the input and output magnitudes, and $c _ { \mathrm { n o i s e } } ( \sigma )$ maps noise level σ into a conditioning in put for $F _ { \theta }$ . Taking a weighted expectation of Eq. 2 over the noise levels gives the overall training loss $\mathbf { \check { \mathbb { E } } } _ { \sigma , \pmb { y } , \pmb { n } } \left[ \lambda ( \sigma ) \mathbf { \epsilon } \| D ( \pmb { y } + \pmb { \bar { n } } ; \sigma ) - \pmb { \bar { y } } \| _ { 2 } ^ { 2 } \right]$ , where $\sigma \sim p _ { \mathrm { t r a i n } } , { \pmb y } \sim p _ { \mathrm { d a t a } } .$ and $\bar { \pmb { n } } \sim \mathcal { N } ( \pmb { 0 } , \sigma ^ { 2 } \mathbf { I } )$ . The probabil ity of sampling a given noise level σ is given by $p _ { \mathrm { t r a i n } } ( \sigma )$ and the corresponding weight is given by $\lambda ( \sigma )$ . We can equivalently express this loss with respect to the raw network output $F _ { \theta }$ in Eq. 7: 

$$
\mathbb {E} _ {\sigma , \boldsymbol {y}, \boldsymbol {n}} \Big [ \underbrace {\lambda (\sigma) c _ {\text {out}} (\sigma) ^ {2}} _ {\text {effective weight}} \Big \| \underbrace {F _ {\theta} \big (c _ {\text {in}} (\sigma) \cdot (\boldsymbol {y} + \boldsymbol {n}) ; c _ {\text {noise}} (\sigma) \big)} _ {\text {network output}} - \underbrace {\frac {1}{c _ {\text {out}} (\sigma)} \big (\boldsymbol {y} - c _ {\text {skip}} (\sigma) \cdot (\boldsymbol {y} + \boldsymbol {n}) \big)} _ {\text {effective training target}} \Big \| _ {2} ^ {2} \Big ].\tag{8}
$$

This form reveals the effective training target of $F _ { \theta } ,$ allowing us to determine suitable choices for the preconditioning functions from first principles. As detailed in Appendix B.6, we derive our choices shown in Table 1 by requiring network inputs and training targets to have unit variance $( \boldsymbol { c } _ { \mathrm { i n } } , \boldsymbol { c } _ { \mathrm { o u t } } )$ , and amplifying errors in $F _ { \theta }$ as little as possible $( c _ { \mathrm { s k i p } } )$ . The formula for $\mathrm { \mathcal { C } _ { n o i s e } }$ is chosen empirically. 


Table 2: Evaluation of our training improvements. The starting point (config A) is VP & VE using our deterministic sampler. At the end (configs E,F), VP & VE only differ in the architecture of $F _ { \theta } .$


<table><tr><td rowspan="2"></td><td colspan="4">CIFAR-10 [29] at 32×32</td><td colspan="2">FFHQ [27] 64×64</td><td colspan="2">AFHQv2 [7] 64×64</td></tr><tr><td colspan="2">Conditional</td><td colspan="2">Unconditional</td><td colspan="2">Unconditional</td><td colspan="2">Unconditional</td></tr><tr><td>Training configuration</td><td>VP</td><td>VE</td><td>VP</td><td>VE</td><td>VP</td><td>VE</td><td>VP</td><td>VE</td></tr><tr><td>A Baseline [49] (*pre-trained)</td><td>2.48</td><td>3.11</td><td>3.01*</td><td>3.77*</td><td>3.39</td><td>25.95</td><td>2.58</td><td>18.52</td></tr><tr><td>B + Adjust hyperparameters</td><td>2.18</td><td>2.48</td><td>2.51</td><td>2.94</td><td>3.13</td><td>22.53</td><td>2.43</td><td>23.12</td></tr><tr><td>C + Redistribute capacity</td><td>2.08</td><td>2.52</td><td>2.31</td><td>2.83</td><td>2.78</td><td>41.62</td><td>2.54</td><td>15.04</td></tr><tr><td>D + Our preconditioning</td><td>2.09</td><td>2.64</td><td>2.29</td><td>3.10</td><td>2.94</td><td>3.39</td><td>2.79</td><td>3.81</td></tr><tr><td>E + Our loss function</td><td>1.88</td><td>1.86</td><td>2.05</td><td>1.99</td><td>2.60</td><td>2.81</td><td>2.29</td><td>2.28</td></tr><tr><td>F + Non-leaky augmentation</td><td>1.79</td><td>1.79</td><td>1.97</td><td>1.98</td><td>2.39</td><td>2.53</td><td>1.96</td><td>2.16</td></tr><tr><td>NFE</td><td>35</td><td>35</td><td>35</td><td>35</td><td>79</td><td>79</td><td>79</td><td>79</td></tr></table>

Table 2 shows FID for a series of training setups, evaluated using our deterministic sampler from Section 3. We start with the baseline training setup of Song et al. [49], which differs considerably between the VP and VE cases; we provide separate results for each (config A). To obtain a more meaningful point of comparison, we re-adjust the basic hyperparameters (config B) and improve the expressive power of the model (config C) by removing the lowest-resolution layers and doubling the capacity of the highest-resolution layers instead; see Appendix F.3 for further details. We then replace the original choices of $\{ c _ { \mathrm { i n } } , c _ { \mathrm { o u t } } , c _ { \mathrm { n o i s e } } , c _ { \mathrm { s k i p } } \}$ with our preconditioning (config D), which keeps the results largely unchanged — except for VE that improves considerably at 64×64 resolution. Instead of improving FID per se, the main benefit of our preconditioning is that it makes the training more robust, enabling us to turn our focus on redesigning the loss function without adverse effects. 

Loss weighting and sampling. Eq. 8 shows that training $F _ { \theta }$ as preconditioned in Eq. 7 incurs an effective per-sample loss weight of $\lambda ( \sigma ) c _ { \mathrm { o u t } } ( \sigma ) ^ { 2 }$ . To balance the effective loss weights, we set $\lambda ( \sigma ) = 1 / c _ { \mathrm { o u t } } ( \sigma ) ^ { 2 }$ , which also equalizes the initial training loss over the entire σ range as shown in Figure 5a (green curve). Finally, we need to select $p _ { \mathrm { t r a i n } } ( \sigma )$ , i.e., how to choose noise levels during training. Inspecting the per-σ loss after training (blue and orange curves) reveals that a significant reduction is possible only at intermediate noise levels; at very low levels, it is both difficult and irrelevant to discern the vanishingly small noise component, whereas at high levels the training targets are always dissimilar from the correct answer that approaches dataset average. Therefore, we target the training efforts to the relevant range using a simple log-normal distribution for $p _ { \mathrm { t r a i n } } ( \sigma )$ as detailed in Table 1 and illustrated in Figure 5a (red curve). 

Table 2 shows that our proposed $p _ { \mathrm { t r a i n } }$ and λ (config E) lead to a dramatic improvement in FID in all cases when used in conjunction with our preconditioning (config D). In concurrent work, Choi et al. [6] propose a similar scheme to prioritize noise levels that are most relevant w.r.t. forming the perceptually recognizable content of the image. However, they only consider the choice of λ in isolation, which results in a smaller overall improvement. 

Augmentation regularization. To prevent potential overfitting that often plagues diffusion models with smaller datasets, we borrow an augmentation pipeline from the GAN literature [25]. The pipeline consists of various geometric transformations (see Appendix F.2) that we apply to a training image prior to adding noise. To prevent the augmentations from leaking to the generated images, we provide the augmentation parameters as a conditioning input to $F _ { \theta } ;$ during inference we set the them to zero to guarantee that only non-augmented images are generated. Table 2 shows that data augmentation provides a consistent improvement (config F) that yields new state-of-the-art FIDs of 1.79 and 1.97 for conditional and unconditional CIFAR-10, beating the previous records of 1.85 [45] and 2.10 [53]. 

Stochastic sampling revisited. Interestingly, the relevance of stochastic sampling appears to diminish as the model itself improves, as shown in Figure 5b,c. When using our training setup in CIFAR-10 (Figure 5b), the best results were obtained with deterministic sampling, and any amount of stochastic sampling was detrimental. 

ImageNet-64. As a final experiment, we trained a class-conditional ImageNet-64 model from scratch using our proposed training improvements. This model achieved a new state-of-the-art FID of 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/762558d08bef15f14a81304c0c7a3c6220c6e240d1a86a149e7898bfe74f8545.jpg)



(a) Loss & noise distribution


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/9bf026bbee9554d4abbdb4aaea9a5d3cec13f6a42e9959bcaa83fcf9ad556f56.jpg)



(b) Stochasticity on CIFAR-10


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/88835001b9580118fec959643384e502ba7a090576fe0b684e23c9f9caa334db.jpg)



(c) Stochasticity on ImageNet-64



Figure 5: (a) Observed initial (green) and final loss per noise level, representative of the the 32×32 (blue) and 64×64 (orange) models considered in this paper. The shaded regions represent the standard deviation over 10k random samples. Our proposed training sample density is shown by the dashed red curve. (b) Effect of $S _ { \mathrm { c h u r n } }$ on unconditional CIFAR-10 with 256 steps $( \mathrm { N F E } = 5 1 1 )$ ). For the original training setup of Song et al. [49], stochastic sampling is highly beneficial (blue, green), while deterministic sampling $( S _ { \mathrm { c h u r n } } = 0 )$ leads to relatively poor FID. For our training setup, the situation is reversed (orange, red); stochastic sampling is not only unnecessary but harmful. (c) Effect of $S _ { \mathrm { c h u r n } }$ on class-conditional ImageNet-64 with 256 steps $( \mathrm { N F E } = 5 1 1 )$ . In this more challenging scenario, stochastic sampling turns out to be useful again. Our training setup improves the results for both deterministic and stochastic sampling.


1.36 compared to the previous record of 1.48 [17]. We used the ADM architecture [9] with no changes, and trained it using our config E with minimal tuning; see Appendix F.3 for details. We did not find overfitting to be a concern, and thus chose to not employ augmentation regularization. As shown in Figure 5c, the optimal amount of stochastic sampling was much lower than with the pre-trained model, but unlike with CIFAR-10, stochastic sampling was clearly better than deterministic sampling. This suggests that more diverse datasets continue to benefit from stochastic sampling. 

## 6 Conclusions

Our approach of putting diffusion models to a common framework exposes a modular design. This allows a targeted investigation of individual components, potentially helping to better cover the viable design space. In our tests this let us simply replace the samplers in various earlier models, drastically improving the results. For example, in ImageNet-64 our sampler turned an average model (FID 2.07) to a challenger (1.55) for the previous SOTA model (1.48) [17], and with training improvements achieved SOTA FID of 1.36. We also obtained new state-of-the-art results on CIFAR-10 while using only 35 model evaluations, deterministic sampling, and a small network. The current high-resolution diffusion models rely either on separate super-resolution steps [17, 36, 40], subspace projection [23], very large networks [9, 49], or hybrid approaches [39, 42, 53] — we believe that our contributions are orthogonal to these extensions. That said, many of our parameter values may need to be re-adjusted for higher resolution datasets. Furthermore, we feel that the precise interaction between stochastic sampling and the training objective remains an interesting question for future work. 

Societal impact. Our advances in sample quality can potentially amplify negative societal effects when used in a large-scale system like DALL·E 2, including types of disinformation or emphasizing sterotypes and harmful biases [34]. The training and sampling of diffusion models needs a lot of electricity; our project consumed ∼250MWh on an in-house cluster of NVIDIA V100s. 

Acknowledgments. We thank Jaakko Lehtinen, Ming-Yu Liu, Tuomas Kynkäänniemi, Axel Sauer, Arash Vahdat, and Janne Hellsten for discussions and comments, and Tero Kuosmanen, Samuel Klenberg, and Janne Hellsten for maintaining our compute infrastructure. 

## References



[1] B. D. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313–326, 1982. 





[2] U. M. Ascher and L. R. Petzold. Computer Methods for Ordinary Differential Equations and Differential Algebraic Equations. Society for Industrial and Applied Mathematics, 1998. 





[3] F. Bao, C. Li, J. Zhu, and B. Zhang. Analytic-DPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In Proc. ICLR, 2022. 





[4] D. Baranchuk, A. Voynov, I. Rubachev, V. Khrulkov, and A. Babenko. Label-efficient semantic segmentation with diffusion models. In Proc. ICLR, 2022. 





[5] C. M. Bishop. Neural networksfor pattern recognition. Oxford University Press, USA, 1995. 





[6] J. Choi, J. Lee, C. Shin, S. Kim, H. Kim, and S. Yoon. Perception prioritized training of diffusion models. In Proc. CVPR, 2022. 





[7] Y. Choi, Y. Uh, J. Yoo, and J.-W. Ha. StarGAN v2: Diverse image synthesis for multiple domains. In Proc. CVPR, 2020. 





[8] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A large-scale hierarchical image database. In Proc. CVPR, 2009. 





[9] P. Dhariwal and A. Q. Nichol. Diffusion models beat GANs on image synthesis. In Proc. NeurIPS, 2021. 





[10] T. Dockhorn, A. Vahdat, and K. Kreis. Score-based generative modeling with critically-damped Langevin diffusion. In Proc. ICLR, 2022. 





[11] J. R. Dormand and P. J. Prince. A family of embedded Runge-Kutta formulae. Journal ofcomputational and applied mathematics, 6(1):19–26, 1980. 





[12] J. B. J. Fourier, G. Darboux, et al. Théorie analytique de la chaleur, volume 504. Didot Paris, 1822. 





[13] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial networks. In Proc. NIPS, 2014. 





[14] U. Grenander and M. I. Miller. Representations of knowledge in complex systems. Journal ofthe Royal Statistical Society: Series B (Methodological), 56(4):549–581, 1994. 





[15] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In Proc. NIPS, 2017. 





[16] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In Proc. NeurIPS, 2020. 





[17] J. Ho, C. Saharia, W. Chan, D. J. Fleet, M. Norouzi, and T. Salimans. Cascaded diffusion models for high fidelity image generation. Journal ofMachine Learning Research, 23, 2022. 





[18] J. Ho and T. Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 





[19] J. Ho, T. Salimans, A. A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet. Video diffusion models. In Proc. ICLR Workshop on Deep Generative Modelsfor Highly Structured Data, 2022. 





[20] C.-W. Huang, J. H. Lim, and A. C. Courville. A variational perspective on diffusion-based generative models and score matching. In Proc. NeurIPS, 2021. 





[21] L. Huang, J. Qin, Y. Zhou, F. Zhu, L. Liu, and L. Shao. Normalization techniques in training DNNs: Methodology, analysis and application. CoRR, abs/2009.12836, 2020. 





[22] A. Hyvärinen. Estimation of non-normalized statistical models by score matching. Journal ofMachine Learning Research, 6(24):695–709, 2005. 





[23] B. Jing, G. Corso, R. Berlinghieri, and T. Jaakkola. Subspace diffusion generative models. In Proc. ECCV, 2022. 





[24] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas. Gotta go fast when generating data with score-based models. CoRR, abs/2105.14080, 2021. 





[25] T. Karras, M. Aittala, J. Hellsten, S. Laine, J. Lehtinen, and T. Aila. Training generative adversarial networks with limited data. In Proc. NeurIPS, 2020. 





[26] T. Karras, M. Aittala, S. Laine, E. Härkönen, J. Hellsten, J. Lehtinen, and T. Aila. Alias-free generative adversarial networks. In Proc. NeurIPS, 2021. 





[27] T. Karras, S. Laine, and T. Aila. A style-based generator architecture for generative adversarial networks. In Proc. CVPR, 2018. 





[28] Z. Kong, W. Ping, J. Huang, K. Zhao, and B. Catanzaro. DiffWave: A versatile diffusion model for audio synthesis. In Proc. ICLR, 2021. 





[29] A. Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009. 





[30] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, and T. Aila. Noise2Noise: Learning image restoration without clean data. In Proc. ICML, 2018. 





[31] L. Liu, Y. Ren, Z. Lin, and Z. Zhao. Pseudo numerical methods for diffusion models on manifolds. In Proc. ICLR, 2022. 





[32] C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu. DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In Proc. NeurIPS, 2022. 





[33] E. Luhman and T. Luhman. Knowledge distillation in iterative generative models for improved sampling speed. CoRR, abs/2101.02388, 2021. 





[34] P. Mishkin, L. Ahmad, M. Brundage, G. Krueger, and G. Sastry. DALL·E 2 preview – risks and limitations. OpenAI, 2022. 





[35] E. Nachmani and S. Dovrat. Zero-shot translation using diffusion models. CoRR, abs/2111.01471, 2021. 





[36] A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen. GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models. In Proc. ICML, 2022. 





[37] A. Q. Nichol and P. Dhariwal. Improved denoising diffusion probabilistic models. In Proc. ICML, volume 139, pages 8162–8171, 2021. 





[38] V. Popov, I. Vovk, V. Gogoryan, T. Sadekova, and M. Kudinov. Grad-TTS: A diffusion probabilistic model for text-to-speech. In Proc. ICML, volume 139, pages 8599–8608, 2021. 





[39] K. Preechakul, N. Chatthee, S. Wizadwongsa, and S. Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. In Proc. CVPR, 2022. 





[40] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen. Hierarchical text-conditional image generation with CLIP latents. Technical report, OpenAI, 2022. 





[41] A. J. Roberts. Modify the improved Euler scheme to integrate stochastic differential equations. CoRR, abs/1210.0933, 2012. 





[42] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proc. CVPR, 2022. 





[43] C. Saharia, W. Chan, H. Chang, C. A. Lee, J. Ho, T. Salimans, D. J. Fleet, and M. Norouzi. Palette: Image-to-image diffusion models. In Proc. SIGGRAPH, 2022. 





[44] T. Salimans and J. Ho. Progressive distillation for fast sampling of diffusion models. In Proc. ICLR, 2022. 





[45] A. Sauer, K. Schwarz, and A. Geiger. StyleGAN-XL: Scaling StyleGAN to large diverse datasets. In Proc. SIGGRAPH, 2022. 





[46] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proc. ICML, pages 2256–2265, 2015. 





[47] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. In Proc. ICLR, 2021. 





[48] Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution. In Proc. NeurIPS, 2019. 





[49] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. In Proc. ICLR, 2021. 





[50] E. Süli and D. F. Mayers. An Introduction to Numerical Analysis. Cambridge University Press, 2003. 





[51] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the Inception architecture for computer vision. In Proc. CVPR, 2016. 





[52] M. Tancik, P. P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. T. Barron, and R. Ng. Fourier features let networks learn high frequency functions in low dimensional domains. In Proc. NeurIPS, 2020. 





[53] A. Vahdat, K. Kreis, and J. Kautz. Score-based generative modeling in latent space. In Proc. NeurIPS, 2021. 





[54] P. Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7):1661–1674, 2011. 





[55] D. Watson, W. Chan, J. Ho, and M. Norouzi. Learning fast samplers for diffusion models by differentiating through sample quality. In Proc. ICLR, 2022. 





[56] D. Watson, J. Ho, M. Norouzi, and W. Chan. Learning to efficiently sample from diffusion probabilistic models. CoRR, abs/2106.03802, 2021. 





[57] J. Wolleb, R. Sandkühler, F. Bieder, P. Valmaggia, and P. C. Cattin. Diffusion models for implicit image segmentation ensembles. In Medical Imaging with Deep Learning, 2022. 





[58] Q. Zhang and Y. Chen. Diffusion normalizing flow. In Proc. NeurIPS, 2021. 





[59] Q. Zhang and Y. Chen. Fast sampling of diffusion models with exponential integrator. CoRR, abs/2204.13902, 2022. 



## Appendices

## A Additional results

Figure 6 presents generated images for class-conditional ImageNet-64 [8] using the pre-trained ADM model by Dhariwal and Nichol [9]. The original DDIM [47] and iDDPM [37] samplers are compared to ours in both deterministic and stochastic settings (Sections 3 and 4). Figure 7 shows the corresponding results that we obtain by training the model from scratch using our improved training configuration (Section 5). 

The original samplers and training configurations by Song et al. [49] are compared to ours in Figures 8 and 9 (unconditional CIFAR-10 [29]), Figure 10 (class-conditional CIFAR-10), and Figure 11 (FFHQ [27] and AFHQv2 [7]). For ease of comparison, the same latent codes $\scriptstyle { \pmb x } _ { 0 }$ are used for each dataset/scenario across different training configurations and ODE choices. Figure 12 shows generated image quality with various NFE when using deterministic sampling. 

Tables 3 and 4 summarize the numerical results on deterministic and stochastic sampling methods in various datasets, previously shown as functions of NFE in Figures 2 and 4. 

## B Derivation of formulas

## B.1 Original ODE / SDE formulation from previous work

Song et al. [49] define their forward SDE (Eq. 5 in [49]) as 

$$
\mathrm{d} \boldsymbol {x} = \boldsymbol {f} (\boldsymbol {x}, t) \mathrm{d} t + g (t) \mathrm{d} \omega_ {t},\tag{9}
$$

where $\omega _ { t }$ is the standard Wiener process and $\pmb { f } ( \cdot , t ) : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ and $g ( \cdot ) : \mathbb { R } \to$ R are the drift and diffusion coefficients, respectively, where d is the dimensionality of the dataset. These coefficients are selected differently for the variance preserving (VP) and variance exploding (VE) formulations, and $f ( \cdot )$ is always of the form $\mathbf { \boldsymbol { f } } ( \mathbf { \boldsymbol { x } } , t ) \mathbf { \bar { \theta } } = \mathbf { \boldsymbol { f } } ( t )$ x, where $f ( \cdot ) : \mathbb { R } \to \mathbf { \bar { \mathbb { R } } }$ . Thus, the SDE can be equivalently written as 

$$
\mathrm{d} \boldsymbol {x} = f (t) \boldsymbol {x} \mathrm{d} t + g (t) \mathrm{d} \omega_ {t}.\tag{10}
$$

The perturbation kernels of this SDE (Eq. 29 in [49]) have the general form 

$$
p _ {0 t} \big (\boldsymbol {x} (t) \mid \boldsymbol {x} (0) \big) = \mathcal {N} \big (\boldsymbol {x} (t); s (t) \boldsymbol {x} (0), s (t) ^ {2} \sigma (t) ^ {2} \mathbf {I} \big),\tag{11}
$$

where $\mathcal { N } ( \boldsymbol { x } ; \boldsymbol { \mu } , \boldsymbol { \Sigma } )$ denotes the probability density function of $\scriptstyle { \mathcal { N } } ( \mu , \Sigma )$ evaluated at $\mathbf { \delta } _ { \mathbf { x } , \mathbf { \delta } }$ 

$$
s (t) = \exp \left(\int_ {0} ^ {t} f (\xi) \mathrm{d} \xi\right), \quad \text { and } \quad \sigma (t) = \sqrt {\int_ {0} ^ {t} \frac {g (\xi) ^ {2}}{s (\xi) ^ {2}} \mathrm{d} \xi}.\tag{12}
$$

The marginal distribution $p _ { t } ( { \pmb x } )$ is obtained by integrating the perturbation kernels over x(0): 

$$
p _ {t} (\boldsymbol {x}) = \int_ {\mathbb {R} ^ {d}} p _ {0 t} (\boldsymbol {x} \mid \boldsymbol {x} _ {0}) p _ {\mathrm{data}} (\boldsymbol {x} _ {0}) \mathrm{d} \boldsymbol {x} _ {0}.\tag{13}
$$

Song et al. [49] define the probability flow ODE (Eq. 13 in [49]) so that it obeys this same $p _ { t } ( \pmb { x } )$ 

$$
\mathrm{d} \boldsymbol {x} = \left[ f (t) \boldsymbol {x} - \frac {1}{2} g (t) ^ {2} \nabla_ {\boldsymbol {x}} \log p _ {t} (\boldsymbol {x}) \right] \mathrm{d} t.\tag{14}
$$

## B.2 Our ODE formulation (Eq. 1 and Eq. 4)

The original ODE formulation (Eq. 14) is built around the functions f and $g$ that correspond directly to specific terms that appear in the formula; the properties of the marginal distribution (Eq. 12) can only be derived indirectly based on these functions. However, f and g are of little practical interest in themselves, whereas the marginal distributions are of utmost importance in terms of training the model in the first place, bootstrapping the sampling process, and understanding how the ODE behaves in practice. Given that the idea of the probability flow ODE is to match a particular set of marginal 


Deterministic, Original sampler (DDIM)



Deterministic, Our sampler (Alg. 1)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/1047fe25ff19f7eaf1525648bd84ddec5d5a9ece262ac10d278c75e4e2a84ef6.jpg)



Figure 6: Results for different samplers on class-conditional ImageNet [8] at 64×64 resolution, using the pre-trained model by Dhariwal and Nichol [9]. The cases correspond to dots in Figures 2c and 4c.



FID 1.36 NFE 511



Deterministic, Our sampler & training configuration


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/7772f7686eb79157ffd231abec71aa191af95b25664d39d46ac3125c3e5e9fe2.jpg)



Stochastic, Our sampler & training configuration


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/188b34dd0d35535b1d4374ef55235e8c165dd74fe01254e57a2221a377d07f95.jpg)



Figure 7: Results for our training configuration on class-conditional ImageNet [8] at 64×64 resolution, using our deterministic and stochastic samplers.



Deterministic, Original sampler (p.flow), VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/172933649ec05139fae71a0d56af1839b247eea00fd9e60ab4a6e1573d6d10ae.jpg)



FID 2.94 NFE 256



Deterministic, Original sampler (p.flow), VE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/ca1a7a854f41f4674f1a4a06e291fd53f753ce665c36da34f4b25d3faac9b3d6.jpg)



FID 5.45 NFE 8192



Deterministic, Our sampler (Alg. 1), VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/6534a6ad3bfd7bc25e4972e086df060cd17543d44d891c464191ae95b2ba4040.jpg)



FID 3.01 NFE 35



Deterministic, Our sampler (Alg. 1), VE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/76319a31d12847f9b9be718c4caeb7d1340a74a73f2f30e182faa24f8aa43c9c.jpg)



FID 3.82 NFE 27



Stochastic, Original sampler (E–M), VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/65a68a80d0ab8e3051807e6b65d292bb52efe3aa11cff81071e09afdaa90d2aa.jpg)



FID 2.55 NFE 1024



Stochastic, Original sampler (P–C), VE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/e834b3e003d8ad518df19fd048afdbacc7de76562e058cdcdff476ac71885d4f.jpg)



FID 2.46 NFE 2048



Stochastic, Our sampler (Alg. 2), VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/9431a8e7dcfbf753fb49d5064eea0f181ad2717ed706e2b2cd122b4dbdaa8b27.jpg)



FID 2.27 NFE 511



Stochastic, Our sampler (Alg. 2), VE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/5fea90ed184c6569f40dee639e51a63b68a6a806589874cab6b0dc9b71fd78a5.jpg)



FID 2.23 NFE 2047



Figure 8: Results for different samplers on unconditional CIFAR-10 [29] at 32×32 resolution, using the pre-trained models by Song et al. [49]. The cases correspond to dots in Figures 2a,b and 4a,b.



Original training (config A), VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/1b471f0cdb5d26e1e15b6e98628ac2b3c903ddbab3aad3024fed60e9459dcd2c.jpg)



FID 3.01 NFE 35



Original training (config A), VE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/098ebdfd7106662fb8a5b5a84716b9614c0ab04d71302fad419a192d3cb7b3f7.jpg)



FID 3.77 NFE 35



Our training (config F), VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/50c4f5dcacd321d8b86827931408e13d1ece39f0c3c6bfce8f880b999a7b5e42.jpg)



FID 1.97 NFE 35



Our training (config F), VE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/af8f355c53267e1ac3a5b5a16a019805601c3f81fd7cac398f071c4a50d854db.jpg)



FID 1.98 NFE 35



Figure 9: Results for different training configurations on unconditional CIFAR-10 [29] at 32×32 resolution, using our deterministic sampler with the same set of latent codes (x<sub>0</sub>) in each case.



Original training (config A), VP



Original training (config A), VE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/9c8a311b626a06eee25e102deb6645f307e7029e576f8340d3c36e40a84ad4a0.jpg)



Figure 10: Results for different training configurations on class-conditional CIFAR-10 [29] at 32×32 resolution, using our deterministic sampler with the same set of latent codes (x<sub>0</sub>) in each case.



FFHQ, Original training (config A), VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/9b36f13a08e541c5b80fb24f543ba1246a4113ac871d9234b1beb1eba135209d.jpg)



FID 3.39 NFE 79



FFHQ, Original training (config A), VE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/4f13457e3b675c1f6454e7b205b441e955fe04bb15202c5ae164a2acee335030.jpg)



FID 25.95 NFE 79



FFHQ, Our training (config F), VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/40513f098851ae739cb6bab99762ed32de6ede285e7772b6bb4156675f26980f.jpg)



FID 2.39 NFE 79



FFHQ, Our training (config F), VE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/530bff5fabe48fff5167b000781aee07cb7067bbc9dec92f8168ddc2ef68f32b.jpg)



FID 2.53 NFE 79



AFHQv2, Original training (config A), VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/d3c9205dcf7653460db50f9a577e7c190e0b1e198c62ea2ccbbcc746090b6784.jpg)



FID 2.58 NFE 79



AFHQv2, Original training (config A), VE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/aa421540e1c40d25c11eb61ce8bba910d085c138a96237bb107db20065f8610b.jpg)



FID 18.52 NFE 79



AFHQv2, Our training (config F), VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/9cbccee5c229134ad45eac68a7ff54d1acb121894d9cd69a6b00396b2c66ee6d.jpg)



FID 1.96 NFE 79



AFHQv2, Our training (config F), VE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/8babfea88d7b658451e4966cc6bbc209d0c50b327abab2a96e8e436c89917d81.jpg)



FID 2.16 NFE 79



Figure 11: Results for different training configurations on FFHQ [27] and AFHQv2 [7] at 64×64 resolution, using our deterministic sampler with the same set of latent codes (x<sub>0</sub>) in each case.


5.13 19 

FID 142.34 NFE 7 


Class-conditional ImageNet-64, Pre-trained


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/3b24b6da1d8ae106a7295985c3712c923361e24f51d373a5c0c2b453e16de7a5.jpg)



Class-conditional CIFAR-10, Our training, VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/42d57e3170b285a873c637fe7b5bb4792e20be2ed4f8ba8c81cb547d1d4bb397.jpg)



Unconditional FFHQ, Our training, VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/7781d83573442326828f02257a7976dc2ab385785b5f55dc484b38e8dc6ede5f.jpg)



Unconditional AFHQv2, Our Training, VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/de9caef56282036881a4a3b09be071886a10bb2b89a25e1a810a4c24c197ba75.jpg)



Figure 12: Image quality and FID as a function of NFE using our deterministic sampler. At 32×32 resolution, reasonable image quality is reached around NFE = 13, but FID keeps improving until NFE = 35. At 64×64 resolution, reasonable image quality is reached around NFE = 19, but FID keeps improving until NFE = 79.



Table 3: Evaluation of our improvements to deterministic sampling. The values correspond to the curves shown in Figure 2. We summarize each curve with two key values: the lowest observed FID for any NFE $( ^ { 6 4 } \mathrm { F I D } ^ { 3 } )$ , and the lowest NFE whose FID is within 3% of the lowest FID (“NFE”). The values marked with “–” are identical to the ones above them, because our sampler uses the same $\sigma ( t )$ and $s ( t )$ as DDIM.


<table><tr><td rowspan="2"></td><td colspan="4">Unconditional CIFAR-10 at 32×32</td><td rowspan="2" colspan="2">Class-conditional ImageNet-64</td></tr><tr><td colspan="2">VP</td><td colspan="2">VE</td></tr><tr><td>Sampling method</td><td>FID ↓</td><td>NFE ↓</td><td>FID ↓</td><td>NFE ↓</td><td>FID ↓</td><td>NFE ↓</td></tr><tr><td>Original sampler [49, 9]</td><td>2.85</td><td>256</td><td>5.45</td><td>8192</td><td>2.85</td><td>250</td></tr><tr><td>Our Algorithm 1</td><td>2.79</td><td>512</td><td>4.78</td><td>8192</td><td>2.73</td><td>384</td></tr><tr><td>+ Heun &amp; our <eq>t_i</eq></td><td>2.88</td><td>255</td><td>4.23</td><td>191</td><td>2.64</td><td>79</td></tr><tr><td>+ Our <eq>\sigma(t)</eq> &amp; <eq>s(t)</eq></td><td>2.93</td><td>35</td><td>3.73</td><td>27</td><td>-</td><td>-</td></tr><tr><td>Black-box RK45</td><td>2.94</td><td>115</td><td>3.69</td><td>93</td><td>2.66</td><td>131</td></tr></table>


Table 4: Evaluation and ablations of our improvements to stochastic sampling. The values correspond to the curves shown in Figure 4.


<table><tr><td rowspan="2"></td><td colspan="4">Unconditional CIFAR-10 at 32×32</td><td rowspan="2" colspan="2">Class-conditional ImageNet-64</td></tr><tr><td colspan="2">VP</td><td colspan="2">VE</td></tr><tr><td>Sampling method</td><td>FID ↓</td><td>NFE ↓</td><td>FID ↓</td><td>NFE ↓</td><td>FID ↓</td><td>NFE ↓</td></tr><tr><td>Deterministic baseline (Alg. 1)</td><td>2.93</td><td>35</td><td>3.73</td><td>27</td><td>2.64</td><td>79</td></tr><tr><td>Alg. 2, <eq>S_{\text{tmin,tmax}} = [0, \infty]</eq>, <eq>S_{\text{noise}} = 1</eq></td><td>2.69</td><td>95</td><td>2.97</td><td>383</td><td>1.86</td><td>383</td></tr><tr><td>Alg. 2, <eq>S_{\text{tmin,tmax}} = [0, \infty]</eq></td><td>2.54</td><td>127</td><td>2.51</td><td>511</td><td>1.63</td><td>767</td></tr><tr><td>Alg. 2, <eq>S_{\text{noise}} = 1</eq></td><td>2.52</td><td>95</td><td>2.84</td><td>191</td><td>1.84</td><td>255</td></tr><tr><td>Alg. 2, Optimal settings</td><td>2.27</td><td>383</td><td>2.23</td><td>767</td><td>1.55</td><td>511</td></tr><tr><td>Previous work [49, 9]</td><td>2.55</td><td>768</td><td>2.46</td><td>1024</td><td>2.01</td><td>384</td></tr></table>

distributions, it makes sense to treat the marginal distributions as first-class citizens and define the ODE directly based on $\sigma ( t )$ and $s ( t )$ , eliminating the need for $f ( t )$ and $g ( t )$ . 

Let us start by expressing the marginal distribution of Eq. 13 in closed form: 

$$
{p _ {t} (\pmb {x})} = {\int_ {\mathbb {R} ^ {d}} p _ {0 t} (\pmb {x} \mid \pmb {x} _ {0}) p _ {\mathrm{data}} (\pmb {x} _ {0}) \mathrm{d} \pmb {x} _ {0}}\tag{15}
$$

$$
{ = } { \int _ { \mathbb { R } ^ { d } } p _ { \mathrm{data} } ( \pmb { x } _ { 0 } ) \left[ \mathcal { N } \big ( \pmb { x } ; s ( t ) \pmb { x } _ { 0 } , s ( t ) ^ { 2 } \sigma ( t ) ^ { 2 } \mathbf { I } \big ) \right] \mathrm{d} \pmb { x } _ { 0 } }\tag{16}
$$

$$
{ = } { \int _ { \mathbb { R } ^ { d } } p _ { \mathrm{data} } ( \boldsymbol { x } _ { 0 } ) \left[ s ( t ) ^ { - d } \mathcal { N } \big ( \boldsymbol { x } / s ( t ) ; \boldsymbol { x } _ { 0 } , \sigma ( t ) ^ { 2 } \mathbf { I } \big ) \right] \mathrm{d} \boldsymbol { x } _ { 0 } }\tag{17}
$$

$$
{ = } { s ( t ) ^ { - d } \int _ { \mathbb { R } ^ { d } } p _ { \mathrm{data} } ( \pmb { x } _ { 0 } ) \mathcal { N } \big ( \pmb { x } / s ( t ) ; \pmb { x } _ { 0 } , \sigma ( t ) ^ { 2 } \mathbf { I } \big ) \mathrm{d} \pmb { x } _ { 0 } }\tag{18}
$$

$$
= s (t) ^ {- d} \left[ p _ {\text { data }} * \mathcal {N} \big (\mathbf {0}, \sigma (t) ^ {2} \mathbf {I} \big) \right] \big (\boldsymbol {x} / s (t) \big),\tag{19}
$$

where $p _ { a } * p _ { b }$ denotes the convolution of probability density functions $p _ { a }$ and $p _ { b }$ . The expression inside the brackets corresponds to a mollified version of $p _ { \mathrm { d a t a } }$ obtained by adding i.i.d. Gaussian noise to the samples. Let us denote this distribution by $p ( { \pmb x } ; { \boldsymbol \sigma } )$ 

$$
p (\boldsymbol {x}; \sigma) = p _ {\text { data }} * \mathcal {N} \bigl (\mathbf {0}, \sigma (t) ^ {2} \mathbf {I} \bigr) \quad \text { and } \quad p _ {t} (\boldsymbol {x}) = s (t) ^ {- d}   p \bigl (\boldsymbol {x} / s (t); \sigma (t) \bigr).\tag{20}
$$

We can now express the probability flow ODE (Eq. 14) using $p ( { \pmb x } ; { \boldsymbol \sigma } )$ instead of $p _ { t } ( { \pmb x } )$ : 

$$
{\mathrm{d} \pmb {x}} = {\left[ f (t) \pmb {x} - \frac {1}{2} g (t) ^ {2} \nabla_ {\pmb {x}} \log \left[ p _ {t} (\pmb {x}) \right] \right] \mathrm{d} t}\tag{21}
$$

$$
{ = } { \left[ f ( t ) \pmb { x } - \frac { 1 } { 2 } g ( t ) ^ { 2 } \nabla _ { \pmb { x } } \log \left[ s ( t ) ^ { - d } p \big ( \pmb { x } / s ( t ) ; \sigma ( t ) \big ) \right] \right] \mathrm{d} t }\tag{22}
$$

$$
= \left[ f (t) \boldsymbol {x} - \frac {1}{2} g (t) ^ {2} \left[ \nabla_ {\boldsymbol {x}} \log s (t) ^ {- d} + \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x} / s (t); \sigma (t)) \right] \right] \mathrm{d} t\tag{23}
$$

$$
{ = } { \left[ f ( t ) \pmb { x } - \frac { 1 } { 2 } g ( t ) ^ { 2 } \nabla _ { \pmb { x } } \log p \big ( \pmb { x } / s ( t ) ; \sigma ( t ) \big ) \right] \mathrm{d} t . }\tag{24}
$$

Next, let us rewrite $f ( t )$ in terms of $s ( t )$ based on Eq. 12: 

$$
\exp \left(\int_ {0} ^ {t} f (\xi) \mathrm{d} \xi\right) = s (t)\tag{25}
$$

$$
\int_ {0} ^ {t} f (\xi) \mathrm{d} \xi = \log s (t)\tag{26}
$$

$$
\mathrm{d} \left[ \int_ {0} ^ {t} f (\xi) \mathrm{d} \xi \right] / \mathrm{d} t = \mathrm{d} [ \log s (t) ] / \mathrm{d} t\tag{27}
$$

$$
f (t) = \dot {s} (t) / s (t).\tag{28}
$$

Similarly, we can also rewrite $g ( t )$ in terms of $\sigma ( t )$ : 

$$
\sqrt {\int_ {0} ^ {t} \frac {g (\xi) ^ {2}}{s (\xi) ^ {2}} \mathrm{d} \xi} = \sigma (t)\tag{29}
$$

$$
\int_ {0} ^ {t} \frac {g (\xi) ^ {2}}{s (\xi) ^ {2}} \mathrm{d} \xi = \sigma (t) ^ {2}\tag{30}
$$

$$
\mathrm{d} \left[ \int_ {0} ^ {t} \frac {g (\xi) ^ {2}}{s (\xi) ^ {2}} \mathrm{d} \xi \right] / \mathrm{d} t = \mathrm{d} [ \sigma (t) ^ {2} ] / \mathrm{d} t\tag{31}
$$

$$
g (t) ^ {2} / s (t) ^ {2} = 2 \dot {\sigma} (t) \sigma (t)\tag{32}
$$

$$
{g (t) / s (t)} = {\sqrt {2 \dot {\sigma} (t) \sigma (t)}}\tag{33}
$$

$$
{ g ( t ) } { = } { s ( t ) \sqrt { 2 \dot { \sigma } ( t ) \sigma ( t ) } . }\tag{34}
$$

Finally, substitute f (Eq. 28) and g (Eq. 34) into the ODE of Eq. 24: 

$$
\mathrm{d} \boldsymbol {x} = \left[ [ f (t) ] \boldsymbol {x} - \frac {1}{2} [ g (t) ] ^ {2} \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x} / s (t); \sigma (t)) \right] \mathrm{d} t\tag{35}
$$

$$
= \left[ \left[ \dot {s} (t) / s (t) \right] \boldsymbol {x} - \frac {1}{2} \left[ s (t) \sqrt {2 \dot {\sigma} (t) \sigma (t)} \right] ^ {2} \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x} / s (t); \sigma (t)) \right] d t\tag{36}
$$

$$
= \left[ \left[ \dot {s} (t) / s (t) \right] \boldsymbol {x} - \frac {1}{2} \left[ 2 s (t) ^ {2} \dot {\sigma} (t) \sigma (t) \right] \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x} / s (t); \sigma (t)) \right] d t\tag{37}
$$

$$
{ = } { \left[ \frac { \dot { s } ( t ) } { s ( t ) } \pmb { x } - s ( t ) ^ { 2 } \dot { \sigma } ( t ) \sigma ( t ) \nabla _ { \pmb { x } } \log p \left( \frac { \pmb { x } } { s ( t ) } ; \sigma ( t ) \right) \right] \mathrm{d} t . }\tag{38}
$$

Thus we have obtained Eq. 4 in the main paper, and Eq. 1 is recovered by setting $s ( t ) = 1$ 

$$
\mathrm{d} \boldsymbol {x} = - \dot {\sigma} (t) \sigma (t) \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma (t)) \mathrm{d} t.\tag{39}
$$

Our formulation (Eq. 4) highlights the fact that every realization of the probability flow ODE is simply a reparameterization of the same canonical ODE; changing $\sigma ( t )$ corresponds to reparameterizing t, whereas changing $s ( t )$ corresponds to reparameterizing x. 

## B.3 Denoising score matching (Eq. 2 and Eq. 3)

For the sake of completeness, we derive the connection between score matching and denoising for a finite dataset. For a more general treatment and further background on the topic, see Hyvärinen [22] and Vincent [54]. 

Let us assume that our training set consists of a finite number of samples $\{ \pmb { y } _ { 1 } , \dotsc , \pmb { y } _ { Y } \}$ . This implies $p _ { \mathrm { d a t a } } ( \pmb { x } )$ is represented by a mixture of Dirac delta distributions: 

$$
p _ {\text { data }} (\boldsymbol {x}) = \frac {1}{Y} \sum_ {i = 1} ^ {Y} \delta \big (\boldsymbol {x} - \boldsymbol {y} _ {i} \big),\tag{40}
$$

which allows us to also express $p ( \pmb { x } ; \sigma )$ in closed form based on Eq. 20: 

$$
p (\boldsymbol {x}; \sigma) = p _ {\text { data }} * \mathcal {N} \bigl (\mathbf {0}, \sigma (t) ^ {2} \mathbf {I} \bigr)\tag{41}
$$

$$
{ = } { \int _ { \mathbb { R } ^ { d } } p _ { \mathrm{data} } ( \boldsymbol { x } _ { 0 } ) \mathcal { N } \big ( \boldsymbol { x } ; \boldsymbol { x } _ { 0 } , \sigma ^ { 2 } \mathbf { I } \big ) \mathrm{d} \boldsymbol { x } _ { 0 } }\tag{42}
$$

$$
= \int_ {\mathbb {R} ^ {d}} \left[ \frac {1}{Y} \sum_ {i = 1} ^ {Y} \delta (\boldsymbol {x} _ {0} - \boldsymbol {y} _ {i}) \right] \mathcal {N} (\boldsymbol {x}; \boldsymbol {x} _ {0}, \sigma^ {2} \mathbf {I}) d \boldsymbol {x} _ {0}\tag{43}
$$

$$
= \frac {1}{Y} \sum_ {i = 1} ^ {Y} \int_ {\mathbb {R} ^ {d}} \mathcal {N} (\boldsymbol {x}; \boldsymbol {x} _ {0}, \sigma^ {2} \mathbf {I}) \delta (\boldsymbol {x} _ {0} - \boldsymbol {y} _ {i}) d \boldsymbol {x} _ {0}\tag{44}
$$

$$
= \frac {1}{Y} \sum_ {i = 1} ^ {Y} \mathcal {N} (\boldsymbol {x}; \boldsymbol {y} _ {i}, \sigma^ {2} \mathbf {I}).\tag{45}
$$

Let us now consider the denoising score matching loss of Eq. 2. By expanding the expectations, we can rewrite the formula as an integral over the noisy samples x: 

$$
\mathcal {L} (D; \sigma) = \mathbb {E} _ {\boldsymbol {y} \sim p _ {\mathrm{data}}} \mathbb {E} _ {\boldsymbol {n} \sim \mathcal {N} (\boldsymbol {0}, \sigma^ {2} \mathbf {I})} \left\| D (\boldsymbol {y} + \boldsymbol {n}; \sigma) - \boldsymbol {y} \right\| _ {2} ^ {2}\tag{46}
$$

$$
{ = } { \mathbb { E } _ { \pmb { y } \sim p _ { \mathrm{data} } } \mathbb { E } _ { \pmb { x } \sim \mathcal { N } ( \pmb { y } , \sigma ^ { 2 } \mathbf { I } ) } \left\| D ( \pmb { x } ; \sigma ) - \pmb { y } \right\| _ { 2 } ^ { 2 } }\tag{47}
$$

$$
= \mathbb {E} _ {\boldsymbol {y} \sim p _ {\mathrm{data}}} \int_ {\mathbb {R} ^ {d}} \mathcal {N} (\boldsymbol {x}; \boldsymbol {y}, \sigma^ {2} \mathbf {I}) \| D (\boldsymbol {x}; \sigma) - \boldsymbol {y} \| _ {2} ^ {2} \mathrm{d} \boldsymbol {x}\tag{48}
$$

$$
= \frac {1}{Y} \sum_ {i = 1} ^ {Y} \int_ {\mathbb {R} ^ {d}} \mathcal {N} (\boldsymbol {x}; \boldsymbol {y} _ {i}, \sigma^ {2} \mathbf {I}) \| D (\boldsymbol {x}; \sigma) - \boldsymbol {y} _ {i} \| _ {2} ^ {2} d \boldsymbol {x}\tag{49}
$$

$$
= \int_ {\mathbb {R} ^ {d}} \underbrace {\frac {1}{Y} \sum_ {i = 1} ^ {Y} \mathcal {N} (\boldsymbol {x} ; \boldsymbol {y} _ {i} , \sigma^ {2} \mathbf {I}) \left\| D (\boldsymbol {x} ; \sigma) - \boldsymbol {y} _ {i} \right\| _ {2} ^ {2}} _ {=: \mathcal {L} (D; \boldsymbol {x}, \sigma)}   \mathrm{d} \boldsymbol {x}.\tag{50}
$$

Eq. 50 means that we can minimize $\mathcal { L } ( D ; \sigma )$ by minimizing $\mathcal { L } ( D ; \pmb { x } , \sigma )$ independently for each x: 

$$
D (\boldsymbol {x}; \sigma) = \arg \min _ {D (\boldsymbol {x}; \sigma)} \mathcal {L} (D; \boldsymbol {x}, \sigma).\tag{51}
$$

This is a convex optimization problem; its solution is uniquely identified by setting the gradient w.r.t. $D ( \pmb { x } ; \sigma )$ to zero: 

$$
\mathbf {0} = \nabla_ {D (\boldsymbol {x}; \sigma)} \left[ \mathcal {L} (D; \boldsymbol {x}, \sigma) \right]\tag{52}
$$

$$
\mathbf {0} = \nabla_ {D (\boldsymbol {x}; \sigma)} \left[ \frac {1}{Y} \sum_ {i = 1} ^ {Y} \mathcal {N} (\boldsymbol {x}; \boldsymbol {y} _ {i}, \sigma^ {2} \mathbf {I}) \| D (\boldsymbol {x}; \sigma) - \boldsymbol {y} _ {i} \| _ {2} ^ {2} \right]\tag{53}
$$

$$
\mathbf {0} = \sum_ {i = 1} ^ {Y} \mathcal {N} (\boldsymbol {x}; \boldsymbol {y} _ {i}, \sigma^ {2} \mathbf {I}) \nabla_ {D (\boldsymbol {x}; \sigma)} \left[ \| D (\boldsymbol {x}; \sigma) - \boldsymbol {y} _ {i} \| _ {2} ^ {2} \right]\tag{54}
$$

$$
\mathbf {0} = \sum_ {i = 1} ^ {Y} \mathcal {N} (\boldsymbol {x}; \boldsymbol {y} _ {i}, \sigma^ {2} \mathbf {I}) \left[ 2 D (\boldsymbol {x}; \sigma) - 2 \boldsymbol {y} _ {i} \right]\tag{55}
$$

$$
\mathbf {0} = \left[ \sum_ {i = 1} ^ {Y} \mathcal {N} (\boldsymbol {x}; \boldsymbol {y} _ {i}, \sigma^ {2} \mathbf {I}) \right] D (\boldsymbol {x}; \sigma) - \sum_ {i = 1} ^ {Y} \mathcal {N} (\boldsymbol {x}; \boldsymbol {y} _ {i}, \sigma^ {2} \mathbf {I}) \boldsymbol {y} _ {i}\tag{56}
$$

$$
D (\boldsymbol {x}; \sigma) = \frac {\sum_ {i} \mathcal {N} (\boldsymbol {x} ; \boldsymbol {y} _ {i} , \sigma^ {2} \mathbf {I}) \boldsymbol {y} _ {i}}{\sum_ {i} \mathcal {N} (\boldsymbol {x} ; \boldsymbol {y} _ {i} , \sigma^ {2} \mathbf {I})},\tag{57}
$$

which gives a closed-form solution for the ideal denoiser $D ( { \boldsymbol x } ; { \boldsymbol \sigma } )$ . Note that Eq. 57 is feasible to compute in practice for small datasets — we show the results for CIFAR-10 in Figure 1b. 

Next, let us consider the score of the distribution $p ( { \pmb x } ; { \boldsymbol \sigma } )$ defined in Eq. 45: 

$$
\nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma) = \frac {\nabla_ {\boldsymbol {x}} p (\boldsymbol {x} ; \sigma)}{p (\boldsymbol {x} ; \sigma)}\tag{58}
$$

$$
= \frac {\nabla_ {\boldsymbol {x}} \left[ \frac {1}{Y} \sum_ {i} \mathcal {N} (\boldsymbol {x} ; \boldsymbol {y} _ {i} , \sigma^ {2} \mathbf {I}) \right]}{\left[ \frac {1}{Y} \sum_ {i} \mathcal {N} (\boldsymbol {x} ; \boldsymbol {y} _ {i} , \sigma^ {2} \mathbf {I}) \right]}\tag{59}
$$

$$
= \frac {\sum_ {i} \nabla_ {\boldsymbol {x}} \mathcal {N} (\boldsymbol {x} ; \boldsymbol {y} _ {i} , \sigma^ {2} \mathbf {I})}{\sum_ {i} \mathcal {N} (\boldsymbol {x} ; \boldsymbol {y} _ {i} , \sigma^ {2} \mathbf {I})}.\tag{60}
$$

We can simplify the numerator of $\mathrm { E q }$ . 60 further: 

$$
\nabla_ {\boldsymbol {x}} \mathcal {N} (\boldsymbol {x}; \boldsymbol {y} _ {i}, \sigma^ {2} \mathbf {I}) = \nabla_ {\boldsymbol {x}} \left[ (2 \pi \sigma^ {2}) ^ {- \frac {d}{2}} \exp \frac {\| \boldsymbol {x} - \boldsymbol {y} _ {i} \| _ {2} ^ {2}}{- 2 \sigma^ {2}} \right]\tag{61}
$$

$$
= \left(2 \pi \sigma^ {2}\right) ^ {- \frac {d}{2}} \nabla_ {\boldsymbol {x}} \left[ \exp \frac {\| \boldsymbol {x} - \boldsymbol {y} _ {i} \| _ {2} ^ {2}}{- 2 \sigma^ {2}} \right]\tag{62}
$$

$$
{ = } { \left[ ( 2 \pi \sigma ^ { 2 } ) ^ { - \frac { d } { 2 } } \exp \frac { \| \bm { x } - \bm { y } _ { i } \| _ { 2 } ^ { 2 } } { - 2 \sigma ^ { 2 } } \right] \nabla _ { \bm { x } } \left[ \frac { \| \bm { x } - \bm { y } _ { i } \| _ { 2 } ^ { 2 } } { - 2 \sigma ^ { 2 } } \right] }\tag{63}
$$

$$
= \mathcal {N} (\boldsymbol {x}; \boldsymbol {y} _ {i}, \sigma^ {2} \mathbf {I}) \nabla_ {\boldsymbol {x}} \left[ \frac {\| \boldsymbol {x} - \boldsymbol {y} _ {i} \| _ {2} ^ {2}}{- 2 \sigma^ {2}} \right]\tag{64}
$$

$$
= \mathcal {N} (\boldsymbol {x}; \boldsymbol {y} _ {i}, \sigma^ {2} \mathbf {I}) \left[ \frac {\boldsymbol {y} _ {i} - \boldsymbol {x}}{\sigma^ {2}} \right].\tag{65}
$$

Let us substitute the result back to Eq. 60: 

$$
\nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma) = \frac {\sum_ {i} \nabla_ {\boldsymbol {x}} \mathcal {N} (\boldsymbol {x} ; \boldsymbol {y} _ {i} , \sigma^ {2} \mathbf {I})}{\sum_ {i} \mathcal {N} (\boldsymbol {x} ; \boldsymbol {y} _ {i} , \sigma^ {2} \mathbf {I})}\tag{66}
$$

$$
\sum_ {i} \mathcal {N} (\boldsymbol {x}; \boldsymbol {y} _ {i}, \sigma^ {2} \mathbf {I}) \left[ \frac {\boldsymbol {y} _ {i} - \boldsymbol {x}}{\sigma^ {2}} \right]
$$

$$
\sum_ {i} \mathcal {N} (\boldsymbol {x}; \boldsymbol {y} _ {i}, \sigma^ {2} \mathbf {I})\tag{67}
$$

$$
= \left(\frac {\sum_ {i} \mathcal {N} (\boldsymbol {x} ; \boldsymbol {y} _ {i} , \sigma^ {2} \mathbf {I}) \boldsymbol {y} _ {i}}{\sum_ {i} \mathcal {N} (\boldsymbol {x} ; \boldsymbol {y} _ {i} , \sigma^ {2} \mathbf {I})} - \boldsymbol {x}\right) / \sigma^ {2}.\tag{68}
$$

Notice that the fraction in Eq. 68 is identical to Eq. 57. We can thus equivalently write Eq. 68 as 

$$
\nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma) = \left(D (\boldsymbol {x}; \sigma) - \boldsymbol {x}\right) / \sigma^ {2},\tag{69}
$$

which matches Eq. 3 in the main paper. 

## B.4 Evaluating our ODE in practice (Algorithm 1)

Let us consider x to be a scaled version of an original, non-scaled variable xˆ and substitute $\mathbf { \boldsymbol { x } } = s ( t )$ xˆ into the score term that appears in our scaled ODE (Eq. 4): 

$$
\nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x} / s (t); \sigma (t))\tag{70}
$$

$$
= \nabla_ {[ s (t) \hat {\boldsymbol {x}} ]} \log p ([ s (t) \hat {\boldsymbol {x}} ] / s (t); \sigma (t))\tag{71}
$$

$$
{ = } { \nabla _ { s ( t ) \hat { \pmb x } } \log p \big ( \hat { \pmb x } ; \sigma ( t ) \big ) }\tag{72}
$$

$$
= \frac {1}{s (t)} \nabla_ {\hat {\boldsymbol {x}}} \log p (\hat {\boldsymbol {x}}; \sigma (t)).\tag{73}
$$

We can further rewrite this with respect to $D ( \cdot )$ using Eq. 3: 

$$
\nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x} / s (t); \sigma (t)) = \frac {1}{s (t) \sigma (t) ^ {2}} \left(D (\hat {\boldsymbol {x}}; \sigma (t)) - \hat {\boldsymbol {x}}\right).\tag{74}
$$

Let us now substitute Eq. 74 into Eq. 4, approximating the ideal denoiser $D ( \cdot )$ with our trained model $D _ { \theta } ( \cdot )$ 

$$
\mathrm{d} \pmb {x} = \left[ \dot {s} (t) \pmb {x} / s (t) - s (t) ^ {2} \dot {\sigma} (t) \sigma (t) \left[ \frac {1}{s (t) \sigma (t) ^ {2}} \Bigl (D _ {\theta} (\hat {\pmb {x}}; \sigma (t)) - \hat {\pmb {x}} \Bigr) \right] \right] \mathrm{d} t\tag{75}
$$

$$
{ = } { \left[ \frac { \dot { s } ( t ) } { s ( t ) } \pmb { x } - \frac { \dot { \sigma } ( t ) s ( t ) } { \sigma ( t ) } \Bigl ( D _ { \theta } \bigl ( \hat { \pmb { x } } ; \sigma ( t ) \bigr ) - \hat { \pmb { x } } \Bigr ) \right] \mathrm{d} t . }\tag{76}
$$

Finally, backsubstitute $\hat { \mathbf { x } } = \mathbf { x } / s ( t )$ : 

$$
\mathrm{d} \boldsymbol {x} = \left[ \frac {\dot {s} (t)}{s (t)} \boldsymbol {x} - \frac {\dot {\sigma} (t) s (t)}{\sigma (t)} \left(D _ {\theta} ([ \hat {\boldsymbol {x}} ]; \sigma (t)) - [ \hat {\boldsymbol {x}} ]\right) \right] \mathrm{d} t\tag{77}
$$

$$
= \left[ \frac {\dot {s} (t)}{s (t)} \boldsymbol {x} - \frac {\dot {\sigma} (t) s (t)}{\sigma (t)} \left(D _ {\theta} ([ \boldsymbol {x} / s (t) ]; \sigma (t)) - [ \boldsymbol {x} / s (t) ]\right) \right] d t\tag{78}
$$

$$
{ = } { \left[ \frac { \dot { s } ( t ) } { s ( t ) } \pmb { x } - \frac { \dot { \sigma } ( t ) s ( t ) } { \sigma ( t ) } D _ { \theta } \big ( \pmb { x } / s ( t ) ; \sigma ( t ) \big ) + \frac { \dot { \sigma } ( t ) } { \sigma ( t ) } \pmb { x } \right] \mathrm{d} t }\tag{79}
$$

$$
= \left[ \left(\frac {\dot {\sigma} (t)}{\sigma (t)} + \frac {\dot {s} (t)}{s (t)}\right) \boldsymbol {x} - \frac {\dot {\sigma} (t) s (t)}{\sigma (t)} D _ {\theta} \big (\boldsymbol {x} / s (t); \sigma (t) \big) \right] d t.\tag{80}
$$

We can equivalenty write Eq. 80 as 

$$
\mathrm{d} \boldsymbol {x} / \mathrm{d} t = \left(\frac {\dot {\sigma} (t)}{\sigma (t)} + \frac {\dot {s} (t)}{s (t)}\right) \boldsymbol {x} - \frac {\dot {\sigma} (t) s (t)}{\sigma (t)} D _ {\theta} \left(\frac {\boldsymbol {x}}{s (t)}; \sigma (t)\right),\tag{81}
$$

matching lines 4 and 7 of Algorithm 1. 

## B.5 Our SDE formulation (Eq. 6)

We derive the SDE of Eq. 6 by the following strategy: 

• The desired marginal densities $p ( \pmb { x } ; \sigma ( t ) )$ are convolutions of the data density $p _ { \mathrm { d a t a } }$ and an isotropic Gaussian density with standard deviation $\sigma ( t )$ (see Eq. 20). Hence, considered as a function of the time t, the density evolves according to a heat diffusion PDE with time-varying diffusivity. As a first step, we find this PDE. 

• We then use the Fokker–Planck equation to recover a family of SDEs for which the density evolves according to this PDE. Eq. 6 is obtained from a suitable parametrization of this family. 

## B.5.1 Generating the marginals by heat diffusion

We consider the time evolution of a probability density $\boldsymbol { q } ( \boldsymbol { \boldsymbol { x } } , t )$ . Our goal is to find a PDE whose solution with the initial value $q ( \pmb { x } , 0 ) : = p _ { \mathrm { d a t a } } ( \pmb { x } )$ is $q ( { \pmb x } , t ) = p \big ( { \pmb x } , \pmb \sigma ( t ) \big )$ . That is, the PDE should reproduce the marginals we postulate in Eq. 20. 

The desired marginals are convolutions of $p _ { \mathrm { d a t a } }$ with isotropic normal distributions of time-varying standard deviation $\sigma ( t )$ , and as such, can be generated by the heat equation with time-varying diffusivity $\kappa ( t )$ The situation is most conveniently analyzed in the Fourier domain, where the marginal densities are simply pointwise products of a Gaussian function and the transformed data density. To find the diffusivity that induces the correct standard deviations, we first write down the heat equation PDE: 

$$
\frac {\partial q (\pmb {x} , t)}{\partial t} = \kappa (t) \Delta_ {\pmb {x}} q (\pmb {x}, t).\tag{82}
$$

The Fourier transformed counterpart of Eq. 82, where the transform is taken along the x-dimension, is given by 

$$
\frac {\partial \hat {q} (\pmb {\nu} , t)}{\partial t} = - \kappa (t) | \pmb {\nu} | ^ {2} \hat {q} (\pmb {\nu}, t).\tag{83}
$$

The target solution $\boldsymbol { q } ( \boldsymbol { x } , t )$ and its Fourier transform $\hat { q } ( \nu , t )$ are given by Eq. 20: 

$$
q (\boldsymbol {x}, t) = p \bigl (\boldsymbol {x}; \sigma (t) \bigr) = p _ {\text { data }} (\boldsymbol {x}) * \mathcal {N} \bigl (\boldsymbol {0}, \sigma (t) ^ {2} \mathbf {I} \bigr)\tag{84}
$$

$$
\hat {q} (\boldsymbol {\nu}, t) = \hat {p} _ {\text { data }} (\boldsymbol {\nu}) \exp \left(- \frac {1}{2} | \boldsymbol {\nu} | ^ {2} \sigma (t) ^ {2}\right).\tag{85}
$$

Differentiating the target solution along the time axis, we have 

$$
\frac {\partial \hat {q} (\pmb {\nu} , t)}{\partial t} = - \dot {\sigma} (t) \sigma (t) | \pmb {\nu} | ^ {2} \hat {p} _ {\mathrm{data}} (\pmb {\nu}) \exp \left(- \frac {1}{2} | \pmb {\nu} | ^ {2} \sigma (t) ^ {2}\right)\tag{86}
$$

$$
{ = } { - \dot { \sigma } ( t ) \sigma ( t ) | \pmb { \nu } | ^ { 2 } \hat { q } ( \pmb { \nu } , t ) . }\tag{87}
$$

Eqs. 83 and 87 share the same left hand side. Equating them allows us to solve for $\kappa ( t )$ that generates the desired evolution: 

$$
- \kappa (t) | \pmb {\nu} | ^ {2} \hat {q} (\pmb {\nu}, t) = - \dot {\sigma} (t) \sigma (t) | \pmb {\nu} | ^ {2} \hat {q} (\pmb {\nu}, t)\tag{88}
$$

$$
\kappa (t) = \dot {\sigma} (t) \sigma (t).\tag{89}
$$

To summarize, the desired marginal densities corresponding to noise levels $\sigma ( t )$ are generated by the PDE 

$$
\frac {\partial q (\pmb {x} , t)}{\partial t} = \dot {\sigma} (t) \sigma (t) \Delta_ {\pmb {x}} q (\pmb {x}, t)\tag{90}
$$

from the initial density $q ( \pmb { x } , 0 ) = p _ { \mathrm { d a t a } } ( \pmb { x } )$ 

## B.5.2 Derivation of our SDE

Given an SDE 

$$
\mathrm{d} \boldsymbol {x} = \boldsymbol {f} (\boldsymbol {x}, t) \mathrm{d} t + \boldsymbol {g} (\boldsymbol {x}, t) \mathrm{d} \omega_ {t},\tag{91}
$$

the Fokker–Planck PDE describes the time evolution of its solution probability density $r ( { \pmb x } , t )$ as 

$$
\frac {\partial r (\boldsymbol {x} , t)}{\partial t} = - \nabla_ {\boldsymbol {x}} \cdot \left(\boldsymbol {f} (\boldsymbol {x}, t) r (\boldsymbol {x}, t)\right) + \frac {1}{2} \nabla_ {\boldsymbol {x}} \nabla_ {\boldsymbol {x}}: \left(\mathbf {D} (\boldsymbol {x}, t) r (\boldsymbol {x}, t)\right),\tag{92}
$$

where $\begin{array} { r } { \mathbf { D } _ { i j } = \sum _ { k } \pmb { g } _ { i k } \pmb { g } _ { j k } } \end{array}$ is the diffusion tensor. We consider the special case $\mathbf { \boldsymbol { g } } ( \mathbf { \boldsymbol { x } } , t ) = \mathbf { \boldsymbol { g } } ( t )$ I of x-independent white noise addition, whereby the equation simplifies to 

$$
\frac {\partial r (\boldsymbol {x} , t)}{\partial t} = - \nabla_ {\boldsymbol {x}} \cdot \left(\boldsymbol {f} (\boldsymbol {x}, t) r (\boldsymbol {x}, t)\right) + \frac {1}{2} g (t) ^ {2} \Delta_ {\boldsymbol {x}} r (\boldsymbol {x}, t).\tag{93}
$$

We are seeking an SDE whose solution density is described by the PDE in Eq. 90. Setting $r ( { \pmb x } , t ) =$ $\boldsymbol { q } ( \boldsymbol { x } , t )$ and equating Eqs. 93 and 90, we find the sufficient condition that the SDE must satisfy 

$$
- \nabla_ {\boldsymbol {x}} \cdot (\boldsymbol {f} (\boldsymbol {x}, t) q (\boldsymbol {x}, t)) + \frac {1}{2} g (t) ^ {2} \Delta_ {\boldsymbol {x}} q (\boldsymbol {x}, t) = \dot {\sigma} (t) \sigma (t) \Delta_ {\boldsymbol {x}} q (\boldsymbol {x}, t)\tag{94}
$$

$$
\nabla_ {\boldsymbol {x}} \cdot (\boldsymbol {f} (\boldsymbol {x}, t) q (\boldsymbol {x}, t)) = \left(\frac {1}{2} g (t) ^ {2} - \dot {\sigma} (t) \sigma (t)\right) \Delta_ {\boldsymbol {x}} q (\boldsymbol {x}, t).\tag{95}
$$

Any choice of functions $\scriptstyle f ( { \boldsymbol { \boldsymbol { x } } } , t )$ and $g ( t )$ satisfying this equation constitute a sought after SDE. Let us now find a specific family of such solutions. The key idea is given by the identity $\nabla _ { x } \cdot \nabla _ { x } = \Delta _ { x }$ Indeed, if we set $\begin{array} { r } { \pmb { f } ( \pmb { x } , t ) q ( \pmb { x } , t ) = \pmb { \upsilon } ( t ) \nabla _ { \pmb { x } } q ( \pmb { x } , t ) } \end{array}$ for any choice of $\boldsymbol { v } ( t )$ , the term $\Delta _ { x } q ( { \pmb x } , t )$ appears on both sides and cancels out: 

$$
\nabla_ {\boldsymbol {x}} \cdot (\upsilon (t) \nabla_ {\boldsymbol {x}} q (\boldsymbol {x}, t)) = \left(\frac {1}{2} g (t) ^ {2} - \dot {\sigma} (t) \sigma (t)\right) \Delta_ {\boldsymbol {x}} q (\boldsymbol {x}, t)\tag{96}
$$

$$
\upsilon (t) \Delta_ {\boldsymbol {x}} q (\boldsymbol {x}, t) = \left(\frac {1}{2} g (t) ^ {2} - \dot {\sigma} (t) \sigma (t)\right) \Delta_ {\boldsymbol {x}} q (\boldsymbol {x}, t)\tag{97}
$$

$$
{ v ( t ) } { = } { \frac { 1 } { 2 } g ( t ) ^ { 2 } - \dot { \sigma } ( t ) \sigma ( t ) . }\tag{98}
$$

The stated $\pmb { f } ( \pmb { x } , t )$ is in fact proportional to the score function, as the formula matches the gradient of the logarithm of the density: 

$$
\boldsymbol {f} (\boldsymbol {x}, t) = v (t) \frac {\nabla_ {\boldsymbol {x}} q (\boldsymbol {x} , t)}{q (\boldsymbol {x} , t)}\tag{99}
$$

$$
{ = } { v ( t ) \nabla _ { \pmb { x } } \log q ( \pmb { x } , t ) }\tag{100}
$$

$$
= \left(\frac {1}{2} g (t) ^ {2} - \dot {\sigma} (t) \sigma (t)\right) \nabla_ {\boldsymbol {x}} \log q (\boldsymbol {x}, t).\tag{101}
$$

Substituting this back into Eq. 91 and writing $p ( \pmb { x } ; \sigma ( t ) )$ ) in place of $\mathbf { \nabla } q ( \pmb { x } , t )$ , we recover a family of SDEs whose solution densities have the desired marginals with noise levels $\sigma ( t )$ for any choice of g(t): 

$$
\mathrm{d} \boldsymbol {x} = \left(\frac {1}{2} g (t) ^ {2} - \dot {\sigma} (t) \sigma (t)\right) \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma (t)) \mathrm{d} t + g (t) \mathrm{d} \omega_ {t}.\tag{102}
$$

The free parameter $g ( t )$ effectively specifies the rate of noise replacement at any given time instance. The special case choice of $g ( t ) = 0$ corresponds to the probability flow ODE. The parametrization by $g ( \bar { t } )$ is not particularly intuitive, however. To obtain a more interpretable parametrization, we set $g ( t ) = \sqrt { 2 \beta ( t ) } \sigma ( t )$ , which yields the (forward) SDE of Eq. 6 in the main paper: 

$$
\mathrm{d} \boldsymbol {x} _ {+} = - \dot {\sigma} (t) \sigma (t) \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma (t)) \mathrm{d} t + \beta (t) \sigma (t) ^ {2} \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma (t)) \mathrm{d} t + \sqrt {2 \beta (t)} \sigma (t) \mathrm{d} \omega_ {t}.\tag{103}
$$

The noise replacement is now proportional to the standard deviation $\sigma ( t )$ of the noise, with the proportionality factor $\beta ( t )$ . Indeed, expanding the score function in the middle term according to Eq. 3 yields $\beta ( t ) \left[ D \big ( \pmb { x } ; \pmb { \sigma } ( t ) \big ) - \pmb { x } \right]$ dt, which changes x proportionally to the negative noise component; the stochastic term injects new noise at the same rate. Intuitively, scaling the magnitude of Langevin exploration according to the current noise standard deviation is a reasonable baseline, as the data manifold is effectively “spread out” by this amount due to the blurring of the density. 

The reverse SDE used in denoising diffusion is simply obtained by applying the time reversal formula of Anderson [1] (as stated in Eq. 6 of Song et al. [49]) on Eq. 103; the entire effect of the reversal is a change of sign in the middle term. 

The scaled generalization of the SDE can be derived using a similar approach as with the ODE previously. As such, the derivation is omitted here. 

## B.6 Our preconditioning and training (Eq. 8)

Following Eq. 2, the denoising score matching loss for a given denoiser $D _ { \theta }$ on a given noise level σ is given by 

$$
\mathcal {L} (D _ {\theta}; \sigma) = \mathbb {E} _ {\boldsymbol {y} \sim p _ {\text { data}}} \mathbb {E} _ {\boldsymbol {n} \sim \mathcal {N} (\boldsymbol {0}, \sigma^ {2} \mathbf {I})} \left\| D _ {\theta} (\boldsymbol {y} + \boldsymbol {n}; \sigma) - \boldsymbol {y} \right\| _ {2} ^ {2}.\tag{104}
$$

We obtain overall training loss by taking a weighted expectation of $\mathcal { L } ( D _ { \theta } ; \sigma )$ over the noise levels: 

$$
\mathcal {L} (D _ {\theta}) = \mathbb {E} _ {\sigma \sim p _ {\mathrm{train}}} [ \lambda (\sigma) \mathcal {L} (D _ {\theta}; \sigma) ]\tag{105}
$$

$$
= \mathbb {E} _ {\sigma \sim p _ {\text { train }}} \left[ \lambda (\sigma) \mathbb {E} _ {\boldsymbol {y} \sim p _ {\text { data }}} \mathbb {E} _ {\boldsymbol {n} \sim \mathcal {N} (\boldsymbol {0}, \sigma^ {2} \mathbf {I})} \| D _ {\theta} (\boldsymbol {y} + \boldsymbol {n}; \sigma) - \boldsymbol {y} \| _ {2} ^ {2} \right]\tag{106}
$$

$$
= \mathbb {E} _ {\sigma \sim p _ {\text { train}}} \mathbb {E} _ {\boldsymbol {y} \sim p _ {\text { data }}} \mathbb {E} _ {\boldsymbol {n} \sim \mathcal {N} (\boldsymbol {0}, \sigma^ {2} \mathbf {I})} \left[ \lambda (\sigma) \| D _ {\theta} (\boldsymbol {y} + \boldsymbol {n}; \sigma) - \boldsymbol {y} \| _ {2} ^ {2} \right]\tag{107}
$$

$$
= \mathbb {E} _ {\sigma , \boldsymbol {y}, \boldsymbol {n}} \left[ \lambda (\sigma) \left\| D _ {\theta} (\boldsymbol {y} + \boldsymbol {n}; \sigma) - \boldsymbol {y} \right\| _ {2} ^ {2} \right],\tag{108}
$$

where the noise levels are distributed according to $\sigma \sim p _ { \mathrm { t r a i n } }$ and weighted by $\lambda ( \sigma )$ 

Using our definition of $D _ { \theta } ( \cdot )$ from Eq. 7, we can further rewrite $\mathcal { L } ( D _ { \theta } )$ as 

$$
\mathbb {E} _ {\sigma , \boldsymbol {y}, \boldsymbol {n}} \left[ \lambda (\sigma) \left\| c _ {\text {skip}} (\sigma) (\boldsymbol {y} + \boldsymbol {n}) + c _ {\text {out}} (\sigma) F _ {\theta} \left(c _ {\text {in}} (\sigma) (\boldsymbol {y} + \boldsymbol {n}); c _ {\text {noise}} (\sigma)\right) - \boldsymbol {y} \right\| _ {2} ^ {2} \right]\tag{109}
$$

$$
= \mathbb {E} _ {\sigma , \boldsymbol {y}, \boldsymbol {n}} \left[ \lambda (\sigma) \left\| c _ {\text { out }} (\sigma) F _ {\theta} \left(c _ {\text { in }} (\sigma) (\boldsymbol {y} + \boldsymbol {n}); c _ {\text { noise }} (\sigma)\right) - \left(\boldsymbol {y} - c _ {\text { skip }} (\sigma) (\boldsymbol {y} + \boldsymbol {n})\right) \right\| _ {2} ^ {2} \right]\tag{110}
$$

$$
= \mathbb {E} _ {\sigma , \boldsymbol {y}, \boldsymbol {n}} \left[ \lambda (\sigma) c _ {\text {out}} (\sigma) ^ {2} \left\| F _ {\theta} \left(c _ {\text {in}} (\sigma) (\boldsymbol {y} + \boldsymbol {n}); c _ {\text {noise}} (\sigma)\right) - \frac {1}{c _ {\text {out}} (\sigma)} \left(\boldsymbol {y} - c _ {\text {skip}} (\sigma) (\boldsymbol {y} + \boldsymbol {n})\right) \right\| _ {2} ^ {2} \right] \tag {111}
$$

$$
= \mathbb {E} _ {\sigma , \boldsymbol {y}, \boldsymbol {n}} \left[ w (\sigma) \left\| F _ {\theta} \left(c _ {\text {in}} (\sigma) (\boldsymbol {y} + \boldsymbol {n}); c _ {\text {noise}} (\sigma)\right) - F _ {\text {target}} (\boldsymbol {y}, \boldsymbol {n}; \sigma) \right\| _ {2} ^ {2} \right],\tag{112}
$$

which matches Eq. 8 and corresponds to traditional supervised training of $F _ { \theta }$ using standard $L _ { 2 }$ loss with effective weight w(·) and target $F _ { \mathrm { t a r g e t } } ( \cdot )$ given by 

$$
w (\sigma) = \lambda (\sigma) c _ {\text { out }} (\sigma) ^ {2} \quad \text { and } \quad F _ {\text { target }} (\boldsymbol {y}, \boldsymbol {n}; \sigma) = \frac {1}{c _ {\text { out }} (\sigma)} \big (\boldsymbol {y} - c _ {\text { skip }} (\sigma) (\boldsymbol {y} + \boldsymbol {n}) \big),\tag{113}
$$

We can now derive formulas for $c _ { \mathrm { i n } } ( \sigma ) , c _ { \mathrm { o u t } } ( \sigma ) , c _ { \mathrm { s k i p } } ( \sigma )$ , and $\lambda ( \sigma )$ from first principles, shown in the “Ours” column of Table 1. 

First, we require the training inputs of $F _ { \theta } ( \cdot )$ to have unit variance: 

$$
\operatorname{Var} _ {\boldsymbol {y}, \boldsymbol {n}} \left[ c _ {\text { in }} (\sigma) (\boldsymbol {y} + \boldsymbol {n}) \right] = 1\tag{114}
$$

$$
c _ {\mathrm{in}} (\sigma) ^ {2} \operatorname{Var} _ {\boldsymbol {y}, \boldsymbol {n}} [ \boldsymbol {y} + \boldsymbol {n} ] = 1\tag{115}
$$

$$
c _ {\mathrm{in}} (\sigma) ^ {2} \bigl (\sigma_ {\mathrm{data}} ^ {2} + \sigma^ {2} \bigr) = 1\tag{116}
$$

$$
{c _ {\mathrm{in}} (\sigma)} = {1 / \sqrt {\sigma^ {2} + \sigma_ {\mathrm{data}} ^ {2}}.}\tag{117}
$$

Second, we require the effective training target $F _ { \mathrm { t a r g e t } }$ to have unit variance: 

$$
\operatorname{Var} _ {\boldsymbol {y}, \boldsymbol {n}} \left[ F _ {\mathrm{target}} (\boldsymbol {y}, \boldsymbol {n}; \sigma) \right] = 1\tag{118}
$$

$$
\operatorname{Var} _ {\boldsymbol {y}, \boldsymbol {n}} \left[ \frac {1}{c _ {\text { out }} (\sigma)} \big (\boldsymbol {y} - c _ {\text { skip }} (\sigma) (\boldsymbol {y} + \boldsymbol {n}) \big) \right] = 1\tag{119}
$$

$$
\frac {1}{c _ {\mathrm{out}} (\sigma) ^ {2}} \operatorname{Var} _ {\boldsymbol {y}, \boldsymbol {n}} \left[ \boldsymbol {y} - c _ {\mathrm{skip}} (\sigma) (\boldsymbol {y} + \boldsymbol {n}) \right] = 1\tag{120}
$$

$$
{c _ {\mathrm{out}} (\sigma) ^ {2}} = {\operatorname{Var} _ {\boldsymbol {y}, \boldsymbol {n}} \left[ \boldsymbol {y} - c _ {\mathrm{skip}} (\sigma) (\boldsymbol {y} + \boldsymbol {n}) \right]}\tag{121}
$$

$$
c _ {\mathrm{out}} (\sigma) ^ {2} = \operatorname{Var} _ {\boldsymbol {y}, \boldsymbol {n}} \left[ \left(1 - c _ {\mathrm{skip}} (\sigma)\right) \boldsymbol {y} + c _ {\mathrm{skip}} (\sigma) \boldsymbol {n} \right]\tag{122}
$$

$$
c _ {\text { out }} (\sigma) ^ {2} = \left(1 - c _ {\text { skip }} (\sigma)\right) ^ {2} \sigma_ {\text { data }} ^ {2} + c _ {\text { skip }} (\sigma) ^ {2} \sigma^ {2}.\tag{123}
$$

Third, we select $c _ { \mathrm { s k i p } } ( \sigma )$ to minimize $c _ { \mathrm { o u t } } ( \sigma )$ , so that the errors of $F _ { \theta }$ are amplified as little as possible: 

$$
c _ {\text { skip }} (\sigma) = \arg \min _ {c _ {\text { skip }} (\sigma)} c _ {\text { out }} (\sigma).\tag{124}
$$

Since $c _ { \mathrm { o u t } } ( \sigma ) \geq 0$ , we can equivalently write 

$$
c _ {\text { skip }} (\sigma) = \arg \min _ {c _ {\text { skip }} (\sigma)} c _ {\text { out }} (\sigma) ^ {2}.\tag{125}
$$

This is a convex optimization problem; its solution is uniquely identified by setting the derivative w.r.t. $c _ { \mathrm { s k i p } } ( \sigma )$ to zero: 

$$
{ 0 } { = } { \mathrm{d} \left[ c _ { \mathrm{out} } ( \sigma ) ^ { 2 } \right] / \mathrm{d} c _ { \mathrm{skip} } ( \sigma ) }\tag{126}
$$

$$
{ 0 } { = } { \mathrm{d} \Big [ \big ( 1 - c _ { \mathrm{skip} } ( \sigma ) \big ) ^ { 2 } \sigma _ { \mathrm{data} } ^ { 2 } + c _ { \mathrm{skip} } ( \sigma ) ^ { 2 } \sigma ^ { 2 } \Big ] / \mathrm{d} c _ { \mathrm{skip} } ( \sigma ) }\tag{127}
$$

$$
{ 0 } { = } { \sigma _ { \mathrm{data} } ^ { 2 } \mathrm{d} \left[ \left( 1 - c _ { \mathrm{skip} } ( \sigma ) \right) ^ { 2 } \right] / \mathrm{d} c _ { \mathrm{skip} } ( \sigma ) + \sigma ^ { 2 } \mathrm{d} \left[ c _ { \mathrm{skip} } ( \sigma ) ^ { 2 } \right] / \mathrm{d} c _ { \mathrm{skip} } ( \sigma ) }\tag{128}
$$

$$
{ 0 } { = } { \sigma _ { \mathrm{data} } ^ { 2 } \left[ 2 c _ { \mathrm{skip} } ( \sigma ) - 2 \right] + \sigma ^ { 2 } \left[ 2 c _ { \mathrm{skip} } ( \sigma ) \right] }\tag{129}
$$

$$
{ 0 } { = } { \left( \sigma ^ { 2 } + \sigma _ { \mathrm{data} } ^ { 2 } \right) c _ { \mathrm{skip} } ( \sigma ) - \sigma _ { \mathrm{data} } ^ { 2 } }\tag{130}
$$

$$
c _ {\mathrm{skip}} (\sigma) = \sigma_ {\mathrm{data}} ^ {2} / \left(\sigma^ {2} + \sigma_ {\mathrm{data}} ^ {2}\right).\tag{131}
$$

We can now substitute Eq. 131 into Eq. 123 to complete the formula for $c _ { \mathrm { o u t } } ( \sigma )$ 

$$
c _ {\mathrm{out}} (\sigma) ^ {2} = \left(1 - \left[ c _ {\mathrm{skip}} (\sigma) \right]\right) ^ {2} \sigma_ {\mathrm{data}} ^ {2} + \left[ c _ {\mathrm{skip}} (\sigma) \right] ^ {2} \sigma^ {2}\tag{132}
$$

$$
{c _ {\mathrm{out}} (\sigma) ^ {2}} = {\left(1 - \left[ \frac {\sigma_ {\mathrm{data}} ^ {2}}{\sigma^ {2} + \sigma_ {\mathrm{data}} ^ {2}} \right]\right) ^ {2} \sigma_ {\mathrm{data}} ^ {2} + \left[ \frac {\sigma_ {\mathrm{data}} ^ {2}}{\sigma^ {2} + \sigma_ {\mathrm{data}} ^ {2}} \right] ^ {2} \sigma^ {2}}\tag{133}
$$

$$
{c _ {\mathrm{out}} (\sigma) ^ {2}} = {\left[ \frac {\sigma^ {2} \sigma_ {\mathrm{data}}}{\sigma^ {2} + \sigma_ {\mathrm{data}} ^ {2}} \right] ^ {2} + \left[ \frac {\sigma_ {\mathrm{data}} ^ {2} \sigma}{\sigma^ {2} + \sigma_ {\mathrm{data}} ^ {2}} \right] ^ {2}}\tag{134}
$$

$$
c _ {\text {out}} (\sigma) ^ {2} = \frac {\left(\sigma^ {2} \sigma_ {\text {data}}\right) ^ {2} + \left(\sigma_ {\text {data}} ^ {2} \sigma\right) ^ {2}}{\left(\sigma^ {2} + \sigma_ {\text {data}} ^ {2}\right) ^ {2}}\tag{135}
$$

$$
c _ {\mathrm{out}} (\sigma) ^ {2} = \frac {(\sigma \cdot \sigma_ {\mathrm{data}}) ^ {2} (\sigma^ {2} + \sigma_ {\mathrm{data}} ^ {2})}{(\sigma^ {2} + \sigma_ {\mathrm{data}} ^ {2}) ^ {2}}\tag{136}
$$

$$
{c _ {\mathrm{out}} (\sigma) ^ {2}} = {\frac {(\sigma \cdot \sigma_ {\mathrm{data}}) ^ {2}}{\sigma^ {2} + \sigma_ {\mathrm{data}} ^ {2}}}\tag{137}
$$

$$
{c _ {\mathrm{out}} (\sigma)} = {\sigma \cdot \sigma_ {\mathrm{data}} / \sqrt {\sigma^ {2} + \sigma_ {\mathrm{data}} ^ {2}}.}\tag{138}
$$

Fourth, we require the effective weight $w ( \sigma )$ to be uniform across noise levels: 

$$
w (\sigma) = 1\tag{139}
$$

$$
\lambda (\sigma) c _ {\mathrm{out}} (\sigma) ^ {2} = 1\tag{140}
$$

$$
\lambda (\sigma) = 1 / c _ {\mathrm{out}} (\sigma) ^ {2}\tag{141}
$$

$$
\lambda (\sigma) = 1 / \left[ \frac {\sigma \cdot \sigma_ {\mathrm{data}}}{\sqrt {\sigma^ {2} + \sigma_ {\mathrm{data}} ^ {2}}} \right] ^ {2}\tag{142}
$$

$$
\lambda (\sigma) = 1 / \left[ \frac {(\sigma \cdot \sigma_ {\mathrm{data}}) ^ {2}}{\sigma^ {2} + \sigma_ {\mathrm{data}} ^ {2}} \right]\tag{143}
$$

$$
\lambda (\sigma) = \left(\sigma^ {2} + \sigma_ {\text { data }} ^ {2}\right) / (\sigma \cdot \sigma_ {\text { data }}) ^ {2}.\tag{144}
$$

We follow previous work and initialize the output layer weights to zero. Consequently, upon initialization $F _ { \theta } ( \cdot ) = 0$ and the expected value of the loss at each noise level is 1. This can be seen by substituting the choices of $\lambda ( \sigma )$ and $c _ { \mathrm { s k i p } } ( \sigma )$ into Eq. 109, considered at a fixed σ: 

$$
\mathbb {E} _ {\boldsymbol {y}, \boldsymbol {n}} \left[ \lambda (\sigma) \left\| c _ {\text {skip}} (\sigma) (\boldsymbol {y} + \boldsymbol {n}) + c _ {\text {out}} (\sigma) F _ {\theta} \left(c _ {\text {in}} (\sigma) (\boldsymbol {y} + \boldsymbol {n}); c _ {\text {noise}} (\sigma)\right) - \boldsymbol {y} \right\| _ {2} ^ {2} \right]\tag{145}
$$

$$
{ = } { \mathbb { E } _ { \boldsymbol { y } , \boldsymbol { n } } \bigg [ \frac { \sigma ^ { 2 } + \sigma _ { \mathrm{data} } ^ { 2 } } { ( \sigma \cdot \sigma _ { \mathrm{data} } ) ^ { 2 } } \bigg \| \frac { \sigma _ { \mathrm{data} } ^ { 2 } } { \sigma ^ { 2 } + \sigma _ { \mathrm{data} } ^ { 2 } } ( \boldsymbol { y } + \boldsymbol { n } ) - \boldsymbol { y } \bigg \| _ { 2 } ^ { 2 } \bigg ] }\tag{146}
$$

$$
{ = } { \mathbb { E } _ { \pmb { y } , \pmb { n } } \bigg [ \frac { \sigma ^ { 2 } + \sigma _ { \mathrm{data} } ^ { 2 } } { ( \sigma \cdot \sigma _ { \mathrm{data} } ) ^ { 2 } } \bigg \| \frac { \sigma _ { \mathrm{data} } ^ { 2 } \pmb { n } - \sigma ^ { 2 } \pmb { y } } { \sigma ^ { 2 } + \sigma _ { \mathrm{data} } ^ { 2 } } \bigg \| _ { 2 } ^ { 2 } \bigg ] }\tag{147}
$$

$$
= \mathbb {E} _ {\boldsymbol {y}, \boldsymbol {n}} \left[ \frac {1}{\sigma^ {2} + \sigma_ {\mathrm{data}} ^ {2}} \left\| \frac {\sigma_ {\mathrm{data}}}{\sigma} \boldsymbol {n} - \frac {\sigma}{\sigma_ {\mathrm{data}}} \boldsymbol {y} \right\| _ {2} ^ {2} \right]\tag{148}
$$

$$
{ = } { \frac { 1 } { \sigma ^ { 2 } + \sigma _ { \mathrm{data} } ^ { 2 } } \mathbb { E } _ { \boldsymbol { y } , \boldsymbol { n } } \bigg [ \frac { \sigma _ { \mathrm{data} } ^ { 2 } } { \sigma ^ { 2 } } \langle \boldsymbol { n } , \boldsymbol { n } \rangle + \frac { \sigma ^ { 2 } } { \sigma _ { \mathrm{data} } ^ { 2 } } \langle \boldsymbol { y } , \boldsymbol { y } \rangle - 2 \langle \boldsymbol { y } , \boldsymbol { n } \rangle \bigg ] }\tag{149}
$$

$$
= \frac {1}{\sigma^ {2} + \sigma_ {\text {data}} ^ {2}} \left[ \frac {\sigma_ {\text {data}} ^ {2}}{\sigma^ {2}} \underbrace {\operatorname{Var} (\boldsymbol {n})} _ {= \sigma^ {2}} + \frac {\sigma^ {2}}{\sigma_ {\text {data}} ^ {2}} \underbrace {\operatorname{Var} (\boldsymbol {y})} _ {= \sigma_ {\text {data}} ^ {2}} - 2 \underbrace {\operatorname{Cov} (\boldsymbol {y} , \boldsymbol {n})} _ {= 0} \right]\tag{150}
$$

(151) 

## C Reframing previous methods in our framework

In this section, we derive the formulas shown in Table 1 for previous methods, discuss the corresponding original samplers and pre-trained models, and detail the practical considerations associated with using them in our framework. 

In practice, the original implementations of these methods differ considerably in terms of the definitions of model inputs and outputs, dynamic range of image data, scaling of x, and interpretation of σ. We eliminate this variation by standardizing on a unified setup where the model always matches our definition of $F _ { \theta }$ , image data is always represented in the continuous range $[ - 1 , 1 ]$ , and the details of x and σ are always in agreement with Eq. 4. 

We minimize the accumulation of floating point round-off errors by always executing Algorithms 1 and 2 at double precision (float64). However, we still execute the network $F _ { \theta } ( \cdot )$ at single precision (float32) to minimize runtime and remain faithful to previous work in terms of network architecture. 

## C.1 Variance preserving formulation

## C.1.1 VP sampling

Song et al. [49] define the VP SDE (Eq. 32 in [49]) as 

$$
\mathrm{d} \boldsymbol {x} = - \frac {1}{2} \left(\beta_ {\min} + t \left(\beta_ {\max} - \beta_ {\min}\right)\right) \boldsymbol {x} \mathrm{d} t + \sqrt {\beta_ {\min} + t \left(\beta_ {\max} - \beta_ {\min}\right)} \mathrm{d} \omega_ {t},\tag{152}
$$

which matches Eq. 10 with the following choices for f and $g \colon$ 

$$
f (t) = - \frac {1}{2} \beta (t), \quad g (t) = \sqrt {\beta (t)}, \quad \text { and } \quad \beta (t) = \left(\beta_ {\max} - \beta_ {\min}\right) t + \beta_ {\min}.\tag{153}
$$

Let α(t) denote the integral of $\beta ( t )$ 

$$
\alpha (t) = \int_ {0} ^ {t} \beta (\xi) \mathrm{d} \xi\tag{154}
$$

$$
= \int_ {0} ^ {t} \left[ \left(\beta_ {\max} - \beta_ {\min}\right) \xi + \beta_ {\min} \right] d \xi\tag{155}
$$

$$
= \frac {1}{2} \left(\beta_ {\max} - \beta_ {\min}\right) t ^ {2} + \beta_ {\min} t\tag{156}
$$

$$
{ = } { \frac { 1 } { 2 }   \beta _ { \mathrm{d} }   t ^ { 2 } + \beta _ { \operatorname* { m i n } }   t , }\tag{157}
$$

where $\beta _ { \mathrm { d } } = \beta _ { \mathrm { m a x } } - \beta _ { \mathrm { m i n } }$ . We can now obtain the formula for σ(t) by substituting Eq. 153 into Eq. 12: 

$$
\sigma (t) = \sqrt {\int_ {0} ^ {t} \frac {[ g (\xi) ] ^ {2}}{[ s (\xi) ] ^ {2}} \mathrm{d} \xi}\tag{158}
$$

$$
= \sqrt {\int_ {0} ^ {t} \frac {\left[ \sqrt {\beta (\xi)} \right] ^ {2}}{\left[ 1 / \sqrt {e ^ {\alpha (\xi)}} \right] ^ {2}} d \xi}\tag{159}
$$

$$
= \sqrt {\int_ {0} ^ {t} \frac {\beta (\xi)}{1 / e ^ {\alpha (\xi)}} \mathrm{d} \xi}\tag{160}
$$

$$
= \sqrt {\int_ {0} ^ {t} \dot {\alpha} (\xi) e ^ {\alpha (\xi)} \mathrm{d} \xi}\tag{161}
$$

$$
{ = } { \sqrt { e ^ { \alpha ( t ) } - e ^ { \alpha ( 0 ) } } }\tag{162}
$$

$$
{ = } { \sqrt { e ^ { \frac { 1 } { 2 } \beta _ { \mathrm{d} } t ^ { 2 } + \beta _ { \operatorname* { m i n } } t } - 1 } , }\tag{163}
$$

which matches the “Schedule” row of Table 1. Similarly for s(t): 

$$
{ s ( t ) } { = } { \exp \left( \int _ { 0 } ^ { t } [ f ( \xi ) ] \mathrm{d} \xi \right) }\tag{164}
$$

$$
= \exp \left(\int_ {0} ^ {t} \left[ - \frac {1}{2} \beta (\xi) \right] \mathrm{d} \xi\right)\tag{165}
$$

$$
= \exp \left(- \frac {1}{2} \left[ \int_ {0} ^ {t} \beta (\xi) \mathrm{d} \xi \right]\right)\tag{166}
$$

$$
{ = } { \exp \left( - \frac { 1 } { 2 } \alpha ( t ) \right) }\tag{167}
$$

$$
{ = } { 1 / \sqrt { e ^ { \alpha ( t ) } } }\tag{168}
$$

$$
{ = } { 1 / \sqrt { e ^ { \frac { 1 } { 2 } \beta _ { \mathrm{d} } t ^ { 2 } + \beta _ { \operatorname* { m i n } } t } } , }\tag{169}
$$

which matches the “Scaling” row of Table 1. We can equivalently write Eq. 169 in a slightly simpler form by utilizing Eq. 163: 

$$
s (t) = 1 / \sqrt {\sigma (t) ^ {2} + 1}.\tag{170}
$$

Song et al. [49] choose to distribute the sampling time steps $\left\{ t _ { 0 } , \dots , t _ { N - 1 } \right\}$ at uniform intervals within $[ \epsilon _ { \mathrm { s } } , 1 ]$ . This corresponds to setting 

$$
t _ {i <   N} = 1 + \frac {i}{N - 1} (\epsilon_ {\mathrm{s}} - 1),\tag{171}
$$

which matches the “Time steps” row of Table 1. 

Finally, Song et al. [49] set $\beta _ { \mathrm { m i n } } = 0 . 1 , \beta _ { \mathrm { m a x } } = 2 0$ , and $\epsilon _ { \mathrm { s } } = 1 0 ^ { - 3 }$ (Appendix C in [49]), and choose to represent images in the range [−1, 1]. These choices are readily compatible with our formulation and are reflected by the “Parameters” section of Table 1. 

## C.1.2 VP preconditioning

In the VP case, Song et al. [49] approximate the score of $p _ { t } ( { \pmb x } )$ of Eq. 13 as<sup>1</sup> 

$$
\nabla_ {\boldsymbol {x}} \log p _ {t} (\boldsymbol {x})   \approx   \underbrace {- \frac {1}{\bar {\sigma} (t)}   F _ {\theta} \big (\boldsymbol {x} ; (M - 1) t \big)} _ {\text { score } (\boldsymbol {x}; F _ {\theta}, t)},\tag{172}
$$

where $M = 1 0 0 0 , F _ { \theta }$ denotes the network, and $\bar { \sigma } ( t )$ corresponds to the standard deviation of the perturbation kernel of Eq. 11. 

Let us expand the definitions of $p _ { t } ( \pmb { x } )$ and $\bar { \sigma } ( t )$ from Eqs. 20 and 11, respectively, and substitute $\mathbf { \boldsymbol { x } } = s ( t ) ;$ xˆ to obtain the corresponding formula with respect to the non-scaled variable xˆ: 

$$
\nabla_ {\boldsymbol {x}} \log \left[ p (\boldsymbol {x} / s (t); \sigma (t)) \right] \approx - \frac {1}{[ s (t) \sigma (t) ]} F _ {\theta} (\boldsymbol {x}; (M - 1) t)\tag{173}
$$

$$
\nabla_ {[ s (t) \hat {\boldsymbol {x}} ]} \log p ([ s (t) \hat {\boldsymbol {x}} ] / s (t); \sigma (t)) \approx - \frac {1}{s (t) \sigma (t)} F _ {\theta} ([ s (t) \hat {\boldsymbol {x}} ]; (M - 1) t)\tag{174}
$$

$$
\frac {1}{s (t)} \nabla_ {\hat {\pmb x}} \log p \big (\hat {\pmb x}; \sigma (t) \big) \approx - \frac {1}{s (t) \sigma (t)} F _ {\theta} \big (s (t) \hat {\pmb x}; (M - 1) t \big)\tag{175}
$$

$$
\nabla_ {\hat {\boldsymbol {x}}} \log p (\hat {\boldsymbol {x}}; \sigma (t)) \approx - \frac {1}{\sigma (t)} F _ {\theta} (s (t) \hat {\boldsymbol {x}}; (M - 1) t).\tag{176}
$$

We can now replace the left-hand side with Eq. 3 and expand the definition of s(t) from Eq. 170: 

$$
\left[ \left(D (\hat {\boldsymbol {x}}; \sigma (t)) - \hat {\boldsymbol {x}}\right) / \sigma (t) ^ {2} \right] \approx - \frac {1}{\sigma (t)} F _ {\theta} (s (t) \hat {\boldsymbol {x}}; (M - 1) t)\tag{177}
$$

$$
{D \big (\hat {\boldsymbol {x}}; \sigma (t) \big)} \approx {\hat {\boldsymbol {x}} - \sigma (t) F _ {\theta} \big (s (t) \hat {\boldsymbol {x}}; (M - 1) t \big)}\tag{178}
$$

$$
D \big (\hat {\boldsymbol {x}}; \sigma (t) \big) \approx \hat {\boldsymbol {x}} - \sigma (t) F _ {\theta} \left(\left[ \frac {1}{\sqrt {\sigma (t) ^ {2} + 1}} \right] \hat {\boldsymbol {x}}; (M - 1) t\right),\tag{179}
$$

which can be further expressed in terms of σ by replacing $\sigma ( t )  \sigma$ and $t \to \sigma ^ { - 1 } ( \sigma )$ 

$$
D (\hat {\boldsymbol {x}}; \sigma) \approx \hat {\boldsymbol {x}} - \sigma F _ {\theta} \left(\frac {1}{\sqrt {\sigma^ {2} + 1}} \hat {\boldsymbol {x}}; (M - 1) \sigma^ {- 1} (\sigma)\right).\tag{180}
$$

We adopt the right-hand side of Eq. 180 as the definition of $D _ { \theta } .$ , obtaining 

$$
D _ {\theta} (\hat {\boldsymbol {x}}; \sigma) = \underbrace {1 \cdot} _ {c _ {\mathrm{skip}}} \hat {\boldsymbol {x}} \underbrace {- \sigma} _ {c _ {\mathrm{out}}} \cdot F _ {\theta} \Big (\underbrace {\frac {1}{\sqrt {\sigma^ {2} + 1}}} _ {c _ {\mathrm{in}}} \cdot \hat {\boldsymbol {x}}; \underbrace {(M - 1)   \sigma^ {- 1} (\sigma)} _ {c _ {\mathrm{noise}}} \Big),\tag{181}
$$

where $c _ { \mathrm { s k i p } } , c _ { \mathrm { o u t } } , c _ { \mathrm { i n } }$ , and $\mathrm { \vec { { C } } _ { n o i s e } }$ match the “Network and preconditioning” section of Table 1. 

## C.1.3 VP training

Song et al. [49] define their training loss $\mathrm { a s } ^ { 2 }$ 

$$
\mathbb {E} _ {t \sim \mathcal {U} (\epsilon_ {\mathrm{t}}, 1), \boldsymbol {y} \sim p _ {\text { data }}, \bar {\boldsymbol {n}} \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} \left[ \left\| \bar {\sigma} (t) \text {   score   } (s (t) \boldsymbol {y} + \bar {\sigma} (t) \bar {\boldsymbol {n}}; F _ {\theta}, t) + \bar {\boldsymbol {n}} \right\| _ {2} ^ {2} \right],\tag{182}
$$

where the definition of score(·) is the same as in Eq. 172. Let us simplify the formula by substituting $\bar { \sigma } ( t ) = s ( t ) \sigma ( t )$ and $\bar { \pmb { n } } = \pmb { n } / \sigma ( t )$ , where $\pmb { n } \sim \mathcal { N } ( \bar { \bf 0 } , \sigma ( t ) ^ { 2 } \mathbf { I } )$ 

$$
\mathbb {E} _ {t, \boldsymbol {y}, \bar {\boldsymbol {n}}} \Big [ \big \| s (t) \sigma (t) \mathrm{score} \left(s (t) \boldsymbol {y} + [ s (t) \sigma (t) ] \bar {\boldsymbol {n}}; F _ {\theta}, t\right) + \bar {\boldsymbol {n}} \big \| _ {2} ^ {2} \Big ]\tag{183}
$$

$$
= \mathbb {E} _ {t, \boldsymbol {y}, \boldsymbol {n}} \left[ \left\| s (t) \sigma (t) \text {score} (s (t) \boldsymbol {y} + s (t) \sigma (t) [ \boldsymbol {n} / \sigma (t) ]; F _ {\theta}, t) + [ \boldsymbol {n} / \sigma (t) ] \right\| _ {2} ^ {2} \right]\tag{184}
$$

$$
{ = } { \mathbb { E } _ { t , \boldsymbol { y } , \boldsymbol { n } } \Big [ \big \| s ( t ) \sigma ( t ) \text {~ score~ } \big ( s ( t ) ( \boldsymbol { y } + \boldsymbol { n } ) ; F _ { \theta } , t \big ) + \boldsymbol { n } / \sigma ( t ) \big \| _ { 2 } ^ { 2 } \Big ] . }\tag{185}
$$

We can express score(·) in terms of $D _ { \theta } ( \cdot )$ by combining Eqs. 172, 170, and 74: 

$$
\text { score } (s (t) \boldsymbol {x}; F _ {\theta}, t) = \frac {1}{s (t) \sigma (t) ^ {2}} \left(D _ {\theta} (\boldsymbol {x}; \sigma (t)) - \boldsymbol {x}\right).\tag{186}
$$

Substituting this back into Eq. 185 gives 

$$
\mathbb {E} _ {t, \boldsymbol {y}, \boldsymbol {n}} \Big [ \big \| s (t) \sigma (t) \left[ \frac {1}{s (t) \sigma (t) ^ {2}} \Big (D _ {\theta} \big (\boldsymbol {y} + \boldsymbol {n}; \sigma (t) \big) - (\boldsymbol {y} + \boldsymbol {n}) \Big) \right] + \frac {1}{\sigma (t)} \left. \boldsymbol {n} \right\| _ {2} ^ {2} \Big ]\tag{187}
$$

$$
{ = } { \mathbb { E } _ { t , \boldsymbol { y } , \boldsymbol { n } } \Big [ \big \| \frac { 1 } { \sigma ( t ) } \Big ( D _ { \theta } \big ( \boldsymbol { y } + \boldsymbol { n } ; \sigma ( t ) \big ) - ( \boldsymbol { y } + \boldsymbol { n } ) \Big ) + \frac { 1 } { \sigma ( t ) } \left. \boldsymbol { n } \right\| _ { 2 } ^ { 2 } \Big ] }\tag{188}
$$

$$
{ = } { \mathbb { E } _ { t , \boldsymbol { y } , \boldsymbol { n } } \Big [ \frac { 1 } { \sigma ( t ) ^ { 2 } } \left\| D _ { \theta } \big ( \boldsymbol { y } + \boldsymbol { n } ; \sigma ( t ) \big ) - \boldsymbol { y } \right\| _ { 2 } ^ { 2 } \Big ] . }\tag{189}
$$

We can further express this in terms of σ by replacing $\sigma ( t )  \sigma$ and $t \to \sigma ^ { - 1 } ( \sigma )$ : 

$$
\underbrace {\mathbb {E} _ {\sigma^ {- 1} (\sigma) \sim \mathcal {U} (\epsilon_ {\mathrm{t}} , 1)}} _ {p _ {\text {train}}} \mathbb {E} _ {\boldsymbol {y}, \boldsymbol {n}} \Big [ \underbrace {\frac {1}{\sigma^ {2}}} _ {\lambda} \left\| D _ {\theta} \big (\boldsymbol {y} + \boldsymbol {n}; \sigma \big) - \boldsymbol {y} \right\| _ {2} ^ {2} \Big ],\tag{190}
$$

which matches Eq. 108 with the choices for $p _ { \mathrm { t r a i n } }$ and λ shown in the “Training” section of Table 1. 

## C.1.4 VP practical considerations

The pre-trained VP model that we use on CIFAR-10 corresponds to the “DDPM++ cont. $\mathbf { ( V P ) } ^ {  , \ast }$ checkpoint<sup>3</sup> provided by Song et al. [49]. It contains a total of 62 million trainable parameters and supports a continuous range of noise levels $\sigma \in \bigl [ \sigma ( \epsilon _ { \mathrm { t } } ) , \sigma ( 1 ) \bigr ] \approx [ 0 . 0 0 1 , 1 5 2 ]$ , i.e., wider than our preferred sampling range [0.002, 80]. We import the model directly as $F _ { \theta } ( \cdot )$ and run Algorithms 1 and 2 using the definitions in Table 1. 

In Figure 2a, the differences between the original sampler (blue) and our reimplementation (orange) are explained by oversights in the implementation of Song et al. [49], also noted by Jolicoeur-Martineau et al. [24] (Appendix D in [24]). First, the original sampler employs an incorrect multiplier<sup>4</sup> in the Euler step: it multiplies dx/dt by $- 1 / N$ instead of $( \dot { \epsilon _ { \mathrm { s } } } - 1 ) \bar { / } ( \dot { N } - 1 )$ Second, it either overshoots or undershoots on the last step by going from $t _ { N - 1 } = \epsilon _ { \mathrm { s } } \tan t _ { N } = \epsilon _ { \mathrm { s } } - 1 / N$ , where $t _ { N } < 0$ when $N < 1 0 0 0$ . In practice, this means that the generated images contain noticeable noise that becomes quite severe with, $\mathrm { e } . \mathrm { g } . , N = 1 2 8$ . Our formulation avoids these issues, because the step sizes in Algorithm 1 are computed consistently from $\{ t _ { i } \}$ and $t _ { N } = 0$ 

## C.2 Variance exploding formulation

## C.2.1 VE sampling in theory

Song et al. [49] define the VE SDE (Eq. 30 in [49]) as 

$$
\mathrm{d} \boldsymbol {x} = \sigma_ {\min} \left(\frac {\sigma_ {\max}}{\sigma_ {\min}}\right) ^ {t} \sqrt {2 \log \frac {\sigma_ {\max}}{\sigma_ {\min}}} \mathrm{d} \omega_ {t},\tag{191}
$$

which matches Eq. 10 with 

$$
f (t) = 0, \quad g (t) = \sigma_ {\min} \sqrt {2 \log \sigma_ {\mathrm{d}}} \sigma_ {\mathrm{d}} ^ {t}, \quad \text { and } \quad \sigma_ {\mathrm{d}} = \sigma_ {\max} / \sigma_ {\min}.\tag{192}
$$

The VE formulation does not employ scaling, which can be easily seen from Eq. 12: 

$$
s (t) = \exp \left(\int_ {0} ^ {t} [ f (\xi) ] \mathrm{d} \xi\right) = \exp \left(\int_ {0} ^ {t} [ 0 ] \mathrm{d} \xi\right) = \exp (0) = 1.\tag{193}
$$

Substituting Eq. 192 into Eq. 12 suggests the following form for $\sigma ( t )$ : 

$$
\sigma (t) = \sqrt {\int_ {0} ^ {t} \frac {[ g (\xi) ] ^ {2}}{[ s (\xi) ] ^ {2}} \mathrm{d} \xi}\tag{194}
$$

$$
= \sqrt {\int_ {0} ^ {t} \frac {\left[ \sigma_ {\mathrm{min}} \sqrt {2 \log \sigma_ {\mathrm{d}}} \sigma_ {\mathrm{d}} ^ {\xi} \right] ^ {2}}{\left[ 1 \right] ^ {2}} \mathrm{d} \xi}\tag{195}
$$

$$
{ = }  \sqrt { \int _ { 0 } ^ { t } \sigma _ { \operatorname* { m i n } } ^ { 2 } \left[ 2 \log \sigma _ { \mathrm{d} } \right] \left[ \sigma _ { \mathrm{d} } ^ { 2 \xi } \right] \mathrm{d} \xi }\tag{196}
$$

$$
= \sigma_ {\min} \sqrt {\int_ {0} ^ {t} \left[ \log \left(\sigma_ {\mathrm{d}} ^ {2}\right) \right] \left[ \left(\sigma_ {\mathrm{d}} ^ {2}\right) ^ {\xi} \right] \mathrm{d} \xi}\tag{197}
$$

$$
{ = } { \sigma _ { \operatorname * { m i n } } \sqrt { \left( \sigma _ { \mathrm{d} } ^ { 2 } \right) ^ { t } - \left( \sigma _ { \mathrm{d} } ^ { 2 } \right) ^ { 0 } } }\tag{198}
$$

$$
{ = } { \sigma _ { \operatorname * { m i n } } \sqrt { \sigma _ { \mathrm{d} } ^ { 2 t } - 1 } . }\tag{199}
$$

Eq. 199 is consistent with the perturbation kernel reported by Song et al. (Eq. 29 in [49]). However, we note that this does not fulfill their intended definition of $\begin{array} { r } { \sigma ( t ) = \sigma _ { \mathrm { m i n } } \left( \frac { \sigma _ { \mathrm { m a x } } } { \sigma _ { \mathrm { m i n } } } \right) ^ { t } } \end{array}$ (Appendix C in [49]). 

## C.2.2 VE sampling in practice

The original implementation<sup>5</sup> of Song et al. [49] uses reverse diffusion predictor<sup>6</sup> to integrate discretized reverse probability flow<sup>7</sup> of discretized VE SDE<sup>8</sup>. Put together, these yield the following update rule for ${ \pmb x } _ { i + 1 }$ 

$$
\pmb {x} _ {i + 1} = \pmb {x} _ {i} + \frac {1}{2} \left(\bar {\sigma} _ {i} ^ {2} - \bar {\sigma} _ {i + 1} ^ {2}\right) \nabla_ {\pmb {x}} \log \bar {p} _ {i} (\pmb {x}),\tag{200}
$$

where 

$$
\bar {\sigma} _ {i <   N} = \sigma_ {\min} \left(\frac {\sigma_ {\max}}{\sigma_ {\min}}\right) ^ {1 - i / (N - 1)} \quad \text { and } \quad \bar {\sigma} _ {N} = 0.\tag{201}
$$

Interestingly, Eq. 200 is identical to the Euler iteration of our ODE with the following choices: 

$$
s (t) = 1, \quad \sigma (t) = \sqrt {t}, \quad \text { and } \quad t _ {i} = \bar {\sigma} _ {i} ^ {2}.\tag{202}
$$

These formulas match the “Sampling” section of Table 1, and their correctness can be verified by substituting them into line 5 of Algorithm 1: 

$$
\boldsymbol {x} _ {i + 1} = \boldsymbol {x} _ {i} + (t _ {i + 1} - t _ {i}) \boldsymbol {d} _ {i}\tag{203}
$$

$$
= \boldsymbol {x} _ {i} + \left(t _ {i + 1} - t _ {i}\right) \left[ \left(\frac {\dot {\sigma} (t)}{\sigma (t)} + \frac {\dot {s} (t)}{s (t)}\right) \boldsymbol {x} - \frac {\dot {\sigma} (t) s (t)}{\sigma (t)} D \left(\frac {\boldsymbol {x}}{s (t)}; \sigma (t)\right) \right]\tag{204}
$$

$$
{ = } { \pmb { x } _ { i } + ( t _ { i + 1 } - t _ { i } ) \bigg [ \frac { \dot { \sigma } ( t ) } { \sigma ( t ) } \pmb { x } - \frac { \dot { \sigma } ( t ) } { \sigma ( t ) } D \big ( \pmb { x } ; \sigma ( t ) \big ) \bigg ] }\tag{205}
$$

$$
= \boldsymbol {x} _ {i} - \left(t _ {i + 1} - t _ {i}\right) \dot {\sigma} (t) \sigma (t) \left[ \left(D (\boldsymbol {x}; \sigma (t)) - \boldsymbol {x}\right) / \sigma (t) ^ {2} \right]\tag{206}
$$

$$
= \boldsymbol {x} _ {i} - (t _ {i + 1} - t _ {i}) \dot {\sigma} (t) \sigma (t) \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma (t))\tag{207}
$$

$$
{ = } { \pmb { x } _ { i } - ( t _ { i + 1 } - t _ { i } ) \Big [ \frac { 1 } { 2 \sqrt { t } } \Big ] \Big [ \sqrt { t } \Big ] \nabla _ { \pmb { x } } \log p \big ( \pmb { x } ; \sigma ( t ) \big ) }\tag{208}
$$

$$
= \boldsymbol {x} _ {i} + \frac {1}{2} (t _ {i} - t _ {i + 1}) \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma (t))\tag{209}
$$

$$
= \boldsymbol {x} _ {i} + \frac {1}{2} \left(\bar {\sigma} _ {i} ^ {2} - \bar {\sigma} _ {i + 1} ^ {2}\right) \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma (t)),\tag{210}
$$

which is made identical to Eq. 200 by the choice $\bar { p } _ { i } ( { \pmb x } ) = p \big ( { \pmb x } ; { \pmb \sigma } ( t _ { i } ) \big )$ 

Finally, Song et al. [49] set $\sigma _ { \mathrm { m i n } } = 0 . 0 1$ and $\sigma _ { \mathrm { m a x } } = 5 0$ for CIFAR-10 (Appendix C in [49]), and choose to represent their images in the range [0, 1] to match previous SMLD models. Since our standardized range [−1, 1] is twice as large, we must multiply $\sigma _ { \mathrm { m i n } }$ and $\sigma _ { \mathrm { m a x } }$ by 2× to compensate. The “Parameters” section of Table 1 reflects these adjusted values. 

## C.2.3 VE preconditioning

In the VE case, Song et al. [49] approximate the score of $p _ { t } ( { \pmb x } )$ of Eq. 13 directly $\mathrm { { a s } ^ { 9 } }$ 

$$
\nabla_ {\pmb {x}} \log p _ {t} (\pmb {x}) \approx \bar {F} _ {\theta} \big (\pmb {x}; \sigma (t) \big),\tag{211}
$$

where the network $\bar { F } _ { \theta }$ is designed to include additional pre-<sup>10</sup> and<sup>11</sup> postprocessing<sup>12</sup> steps: 

$$
\bar {F} _ {\theta} (\boldsymbol {x}; \sigma) = \frac {1}{\sigma} F _ {\theta} (2 \boldsymbol {x} - 1; \log (\sigma)).\tag{212}
$$

For consistency, we handle the pre- and postprocessing using $\{ c _ { \mathrm { s k i p } } , c _ { \mathrm { o u t } } , c _ { \mathrm { i n } } , c _ { \mathrm { n o i s e } } \}$ as opposed to baking them into the network itself. 

We cannot use Eqs. 211 and 212 directly in our framework, however, because they assume that the images are represented in range [0, 1]. In order to use $[ - 1 , 1 ]$ instead, we replace $p _ { t } ( { \pmb x } )  p _ { t } ( 2 { \pmb x } - 1 )$ ${ \pmb x }  \textstyle { \frac { 1 } { 2 } } { \pmb x } + \textstyle { \frac { 1 } { 2 } }$ and $\begin{array} { r } { \sigma  \frac { 1 } { 2 } \sigma ; } \end{array}$ 

$$
\nabla_ {[ \frac {1}{2} \pmb {x} + \frac {1}{2} ]} \log p _ {t} \big (2 [ \frac {1}{2} \pmb {x} + \frac {1}{2} ] - 1 \big) \approx \frac {1}{[ \frac {1}{2} \sigma ]} F _ {\theta} \big (2 [ \frac {1}{2} \pmb {x} + \frac {1}{2} ] - 1; \log [ \frac {1}{2} \sigma ] \big)\tag{213}
$$

$$
2 \nabla_ {\boldsymbol {x}} \log p _ {t} (\boldsymbol {x}) \approx \frac {2}{\sigma} F _ {\theta} \left(\boldsymbol {x}; \log \left(\frac {1}{2} \sigma\right)\right)\tag{214}
$$

$$
\nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma) \approx \frac {1}{\sigma} F _ {\theta} \left(\boldsymbol {x}; \log \left(\frac {1}{2} \sigma\right)\right).\tag{215}
$$

We can now express the model in terms of $D _ { \theta } ( \cdot )$ by replacing the left-hand side of Eq. 215 with Eq. 3: 

$$
\left(D _ {\theta} (\boldsymbol {x}; \sigma) - \boldsymbol {x}\right) / \sigma^ {2} = \frac {1}{\sigma} F _ {\theta} \left(\boldsymbol {x}; \log \left(\frac {1}{2} \sigma\right)\right)\tag{216}
$$

$$
D _ {\theta} (\boldsymbol {x}; \sigma) = \underbrace {1 \cdot} _ {c _ {\text {skip}}} \boldsymbol {x} + \underbrace {\sigma \cdot} _ {c _ {\text {out}}} F _ {\theta} \Big (\underbrace {1 \cdot} _ {c _ {\text {in}}} \boldsymbol {x}; \underbrace {\log \left(\frac {1}{2} \sigma\right)} _ {c _ {\text {noise}}} \Big),\tag{217}
$$

where $c _ { \mathrm { s k i p } } , c _ { \mathrm { o u t } } , c _ { \mathrm { i n } }$ , and $\mathrm { \vec { { C } } _ { n o i s e } }$ match the “Network and preconditioning” section of Table 1. 

## C.2.4 VE training

Song et al. [49] define their training loss similarly for VP and VE, so we can reuse Eq. 185 by borrowing the definition of score(·) from Eq. 216: 

$$
\mathbb {E} _ {t, \pmb {y}, \pmb {n}} \Big [ \big | \big | s (t) \sigma (t) \mathrm{score} \big (s (t) (\pmb {y} + \pmb {n}); F _ {\theta}, t \big) + \pmb {n} / \sigma (t) \big | \big | _ {2} ^ {2} \Big ]\tag{218}
$$

$$
{ = } { \mathbb { E } _ { t , \boldsymbol { y } , \boldsymbol { n } } \Big [ \big \| \sigma ( t ) \text {~ s c o r e~ } ( \boldsymbol { y } + \boldsymbol { n } ; F _ { \theta } , t ) + \boldsymbol { n } / \sigma ( t ) \big \| _ { 2 } ^ { 2 } \Big ] }\tag{219}
$$

$$
= \mathbb {E} _ {t, \boldsymbol {y}, \boldsymbol {n}} \left[ \left\| \sigma (t) \left[ \left(D _ {\theta} (\boldsymbol {y} + \boldsymbol {n}; \sigma (t)) - (\boldsymbol {y} + \boldsymbol {n})\right) / \sigma (t) ^ {2} \right] + \boldsymbol {n} / \sigma (t) \right\| _ {2} ^ {2} \right]\tag{220}
$$

$$
{ = } { \mathbb { E } _ { t , \boldsymbol { y } , \boldsymbol { n } } \Big [ \frac { 1 } { \sigma ( t ) ^ { 2 } } \left\| D _ { \theta } \big ( \boldsymbol { y } + \boldsymbol { n } ; \sigma ( t ) \big ) - \boldsymbol { y } \right\| _ { 2 } ^ { 2 } \Big ] . }\tag{221}
$$

For VE training, the original implementation<sup>13</sup> defines $\begin{array} { r } { \sigma ( t ) = \sigma _ { \mathrm { m i n } } \left( \frac { \sigma _ { \mathrm { m a x } } } { \sigma _ { \mathrm { m i n } } } \right) ^ { t } } \end{array}$ . We can thus rewrite Eq. 221 as 

$$
\underbrace {\mathbb {E} _ {\ln (\sigma) \sim \mathcal {U} (\ln (\sigma_ {\min}) , \ln (\sigma_ {\max}))}} _ {p _ {\mathrm{train}}} \mathbb {E} _ {\boldsymbol {y}, \boldsymbol {n}} \Big [ \underbrace {\frac {1}{\sigma^ {2}}} _ {\lambda} \left\| D _ {\theta} \big (\boldsymbol {y} + \boldsymbol {n}; \sigma \big) - \boldsymbol {y} \right\| _ {2} ^ {2} \Big ],\tag{222}
$$

which matches Eq. 108 with the choices for $p _ { \mathrm { t r a i n } }$ and $\lambda$ shown in the “Training” section of Table 1. 

## C.2.5 VE practical considerations

The pre-trained VE model that we use on CIFAR-10 corresponds to the “NCSN++ cont. $\mathrm { ( V E ) } ^ { \dag }$ checkpoint<sup>14</sup> provided by Song et al. [49]. It contains a total of 63 million trainable parameters and supports a continuous range of noise levels $\sigma \in \big [ \sigma ( \epsilon _ { \mathrm { t } } ) , \sigma ( 1 ) \big ] \approx [ 0 . 0 2 , 1 0 0 ]$ . This is narrower than our preferred sampling range [0.002, 80], so we set $\sigma _ { \mathrm { m i n } } = 0 . 0 2$ in all related experiments. Note that this limitation is lifted by our training improvements in config E, so we revert back to using $\sigma _ { \mathrm { m i n } } = 0 . 0 0 2$ with configs E and F in Table 2. When importing the model, we remove the pre- and postprocessing steps shown in Eq. 212 to stay consistent with the definition of $F _ { \theta } ( \cdot )$ in Eq. 217. With these changes, we can run Algorithms 1 and 2 using the definitions in Table 1. 

In Figure 2b, the differences between the original sampler (blue) and our reimplementation (orange) are explained by floating point round-off errors that the original implementation suffers from at high step counts. Our results are more accurate in these cases because we represent $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ at double precision in Algorithm 1. 

## C.3 Improved DDPM and DDIM

## C.3.1 DDIM ODE formulation

Song et al. [47] make the observation that their deterministic DDIM sampler can be expressed as Euler integration of the following ODE (Eq. 14 in [47]): 

$$
\mathrm{d} \boldsymbol {x} (t) = \epsilon_ {\theta} ^ {(t)} \left(\frac {\boldsymbol {x} (t)}{\sqrt {\sigma (t) ^ {2} + 1}}\right) \mathrm{d} \sigma (t),\tag{223}
$$

where ${ \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf } { \mathbf { } } { \mathbf { } } { \mathbf } { } \mathbf { }  { \mathbf { } \mathbf { } } { \mathbf { } \mathbf { } } { \mathbf { } \mathbf { } } { \mathbf } { \mathbf { } } { \mathbf } { \mathbf } { } \mathbf { } \mathbf { } \mathbf { } \mathbf { }  { \mathbf } { \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } } { \mathbf } { \mathbf } { \mathbf } { \mathbf } { \mathbf } { \mathbf } { \mathbf } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf $ is a scaled version of the iterate that appears in their discrete update formula (Eq. 10 in [47]) and $\epsilon _ { \theta }$ is a model trained to predict the normalized noise vector, i.e., $\epsilon _ { \theta } ^ { ( t ) } \big ( { \pmb x } ( t ) / \sqrt { \sigma ( t ) ^ { 2 } + 1 } \big )$ ≈ ${ \mathbf { } } n ( t ) / \sigma ( t )$ for ${ \pmb x } ( t ) = { \pmb y } ( t ) + { \pmb n } ( t )$ . In our formulation, $D _ { \theta }$ is trained to approximate the clean signal, i.e., $D _ { \theta } \big ( \mathbf { \boldsymbol { x } } ( t ) ; \boldsymbol { \sigma } ( t ) \big ) \approx \mathbf { \boldsymbol { y } } ,$ , so we can reinterpret $\epsilon _ { \theta }$ in terms of $D _ { \theta }$ as follows: 

$$
\boldsymbol {n} (t) = \boldsymbol {x} (t) - \boldsymbol {y} (t)\tag{224}
$$

$$
\left[ \boldsymbol {n} (t) / \sigma (t) \right] = \left(\boldsymbol {x} (t) - \left[ \boldsymbol {y} (t) \right]\right) / \sigma (t)\tag{225}
$$

$$
\epsilon_ {\theta} ^ {(t)} \big (\boldsymbol {x} (t) / \sqrt {\sigma (t) ^ {2} + 1} \big) = \big (\boldsymbol {x} (t) - D _ {\theta} \big (\boldsymbol {x} (t); \sigma (t) \big) \big) / \sigma (t).\tag{226}
$$

Assuming ideal $\epsilon ( \cdot )$ and $D ( \cdot )$ in $L _ { 2 }$ sense, we can further simplify the above formula using Eq. 3: 

$$
\epsilon^ {(t)} \big (\boldsymbol {x} (t) / \sqrt {\sigma (t) ^ {2} + 1} \big) = \big (\boldsymbol {x} (t) - D \big (\boldsymbol {x} (t); \sigma (t) \big) \big) / \sigma (t)\tag{227}
$$

$$
= - \sigma (t) \left[ \left(D (\boldsymbol {x} (t); \sigma (t)) - \boldsymbol {x} (t)\right) / \sigma (t) ^ {2} \right]\tag{228}
$$

$$
= - \sigma (t) \nabla_ {\boldsymbol {x} (t)} \log p (\boldsymbol {x} (t); \sigma (t)).\tag{229}
$$

Substituting Eq. 229 back into Eq. 223 gives 

$$
\mathrm{d} \boldsymbol {x} (t) = - \sigma (t) \nabla_ {\boldsymbol {x} (t)} \log p (\boldsymbol {x} (t); \sigma (t)) \mathrm{d} \sigma (t),\tag{230}
$$

which we can further simplify by setting $\sigma ( t ) = t \cdot$ 

$$
\mathrm{d} \boldsymbol {x} = - t \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma (t)) \mathrm{d} t.\tag{231}
$$

This matches our Eq. 4 with $s ( t ) = 1$ and $\sigma ( t ) = t .$ , reflected by the “Sampling” section of Table 1. 

## C.3.2 iDDPM time step discretization

The original DDPM formulation of Ho et al. [16] defines the forward process (Eq. 2 in [16]) as a Markov chain that gradually adds Gaussian noise to $\bar { \mathbf { x } } _ { 0 } \sim p _ { \mathrm { d a t a } }$ according to a discrete variance schedule $\{ \beta _ { 1 } , . . . , \bar { \beta _ { T } } \}$ 

$$
q (\bar {\boldsymbol {x}} _ {t} \mid \bar {\boldsymbol {x}} _ {t - 1}) = \mathcal {N} \bigl (\bar {\boldsymbol {x}} _ {t}; \sqrt {1 - \beta_ {t}}   \bar {\boldsymbol {x}} _ {t - 1},   \beta_ {t}   \mathbf {I} \bigr).\tag{232}
$$

The corresponding transition probability from $\bar { \mathbf { x } } _ { 0 }$ to $\bar { \mathbf { x } } _ { t }$ (Eq. 4 in [16]) is given by 

$$
q \left(\bar {\boldsymbol {x}} _ {t} \mid \bar {\boldsymbol {x}} _ {0}\right) = \mathcal {N} \left(\bar {\boldsymbol {x}} _ {t}; \sqrt {\bar {\alpha} _ {t}} \bar {\boldsymbol {x}} _ {0}, (1 - \bar {\alpha} _ {t}) \mathbf {I}\right), \quad \text { where } \quad \bar {\alpha} _ {t} = \prod_ {s = 1} ^ {t} (1 - \beta_ {s}).\tag{233}
$$

Ho et al. [16] define $\{ \beta _ { t } \}$ based on a linear schedule and then calculate the corresponding $\left\{ \bar { \alpha } _ { t } \right\}$ from Eq. 233. Alternatively, one can also define $\left\{ \bar { \alpha } _ { t } \right\}$ first and then solve for $\{ \beta _ { t } \}$ : 

$$
\bar {\alpha} _ {t} = \prod_ {s = 1} ^ {t} (1 - \beta_ {s})\tag{234}
$$

$$
{\bar {\alpha} _ {t}} = {\bar {\alpha} _ {t - 1} (1 - \beta_ {t})}\tag{235}
$$

$$
\beta_ {t} = 1 - \frac {\bar {\alpha} _ {t}}{\bar {\alpha} _ {t - 1}}.\tag{236}
$$

The improved DDPM formulation of Nichol and Dhariwal [37] employs a cosine schedule for $\bar { \alpha } _ { t }$ (Eq. 17 in [37]), defined as 

$$
\bar {\alpha} _ {t} = \frac {f (t)}{f (0)}, \quad \text { where } \quad f (t) = \cos^ {2} \left(\frac {t / T + s}{1 + s} \cdot \frac {\pi}{2}\right),\tag{237}
$$

where $s = 0 . 0 0 8$ . In their implementation $^ { 1 5 }$ , however, Nichol et al. leave out the division by f(0) and simply define<sup>16</sup> 

$$
\bar {\alpha} _ {t} = \cos^ {2} \left(\frac {t / T + s}{1 + s} \cdot \frac {\pi}{2}\right).\tag{238}
$$

To prevent singularities near $t = T$ , they also clamp $\beta _ { t }$ to 0.999. We can express the clamping in terms of $\bar { \alpha } _ { t }$ by utilizing Eq. 233 and Eq. 234: 

$$
{\bar {\alpha} _ {t} ^ {\prime}} = {\prod_ {s = 1} ^ {t} \left(1 - [ \beta_ {s} ^ {\prime} ]\right)}\tag{239}
$$

$$
= \prod_ {s = 1} ^ {t} \left(1 - \min \left([ \beta_ {s} ], 0. 9 9 9\right)\right)\tag{240}
$$

$$
= \prod_ {s = 1} ^ {t} \left(1 - \min \left(1 - \frac {\bar {\alpha} _ {s}}{\bar {\alpha} _ {s - 1}}, 0. 9 9 9\right)\right)\tag{241}
$$

$$
= \prod_ {s = 1} ^ {t} \max \left(\frac {\bar {\alpha} _ {s}}{\bar {\alpha} _ {s - 1}}, 0. 0 0 1\right).\tag{242}
$$

Let us now reinterpret the above formulas in our unified framework. Recall from Table 1 that we denote the original iDDPM sampling steps by $\{ u _ { j } \}$ in the order of descending noise level $\sigma ( u _ { j } )$ where $j \in \{ 0 , \dot { \ldots } , M \}$ . To harmonize the notation of Eq. 233, Eq. 238, and Eq. 239, we thus have to replace $T \longrightarrow M$ and $t \longrightarrow M - j \colon$ 

$$
q (\bar {\boldsymbol {x}} _ {j} \mid \bar {\boldsymbol {x}} _ {M}) = \mathcal {N} \bigl (\bar {\boldsymbol {x}} _ {j}; \sqrt {\bar {\alpha} _ {j} ^ {\prime}}   \bar {\boldsymbol {x}} _ {M}, (1 - \bar {\alpha} _ {j} ^ {\prime})   \mathbf {I} \bigr),\tag{243}
$$

$$
\bar {\alpha} _ {j} = \cos^ {2} \left(\frac {(M - j) / M + C _ {2}}{1 + C _ {2}} \cdot \frac {\pi}{2}\right), \quad \text { and }\tag{244}
$$

$$
\bar {\alpha} _ {j} ^ {\prime} = \prod_ {s = M - 1} ^ {j} \max \left(\frac {\bar {\alpha} _ {j}}{\bar {\alpha} _ {j + 1}}, C _ {1}\right) = \bar {\alpha} _ {j + 1} ^ {\prime} \max \left(\frac {\bar {\alpha} _ {j}}{\bar {\alpha} _ {j + 1}}, C _ {1}\right),\tag{245}
$$

where the constants are $C _ { 1 } = 0 . 0 0 1$ and $C _ { 2 } = 0 . 0 0 8 .$ 

We can further simplify Eq. 244: 

$$
\bar {\alpha} _ {j} = \cos^ {2} \left(\frac {(M - j) / M + C _ {2}}{1 + C _ {2}} \cdot \frac {\pi}{2}\right)\tag{246}
$$

$$
= \cos^ {2} \left(\frac {\pi}{2} \frac {(1 + C _ {2}) - j / M}{1 + C _ {2}}\right)\tag{247}
$$

$$
{ = } { \cos ^ { 2 } \left( \frac { \pi } { 2 } - \frac { \pi } { 2 } \frac { j } { M ( 1 + C _ { 2 } ) } \right) }\tag{248}
$$

$$
= \sin^ {2} \left(\frac {\pi}{2} \frac {j}{M (1 + C _ {2})}\right),\tag{249}
$$

giving the formula shown in the “Parameters” section of Table 1. 

To harmonize the definitions of x and x¯, we must match the perturbation kernel of Eq. 11 with the transition probability of Eq. 243 for each time step $t = u _ { j }$ 

$$
{p _ {0 t} \big (\pmb {x} (u _ {j}) \mid \pmb {x} (0) \big)} = {q (\bar {\pmb {x}} _ {j} \mid \bar {\pmb {x}} _ {M})}\tag{250}
$$

$$
\mathcal {N} \left(\boldsymbol {x} (u _ {j}); s (t) \boldsymbol {x} (0), s (u _ {j}) ^ {2} \sigma (u _ {j}) ^ {2} \mathbf {I}\right) = \mathcal {N} \left(\bar {\boldsymbol {x}} _ {j}; \sqrt {\bar {\alpha} _ {j} ^ {\prime}} \bar {\boldsymbol {x}} _ {M}, \left(1 - \bar {\alpha} _ {j} ^ {\prime}\right) \mathbf {I}\right).\tag{251}
$$

Substituting $s ( t ) = 1$ and $\sigma ( t ) = t$ from Appendix C.3.1, as well as $\bar { \pmb x } _ { M } = { \pmb x } ( 0 )$ 

$$
\mathcal {N} \big (\boldsymbol {x} (u _ {j}); \boldsymbol {x} (0), u _ {j} ^ {2} \mathbf {I} \big) = \mathcal {N} \left(\bar {\boldsymbol {x}} _ {j}; \sqrt {\bar {\alpha} _ {j} ^ {\prime}} \boldsymbol {x} (0), \left(1 - \bar {\alpha} _ {j} ^ {\prime}\right) \mathbf {I}\right).\tag{252}
$$

We can match the means of these two distributions by defining $\bar { \pmb { x } } _ { j } = \sqrt { \bar { \alpha } _ { j } ^ { \prime } } \pmb { x } ( u _ { j } ) \colon$ 

$$
\mathcal {N} \left(\boldsymbol {x} \left(u _ {j}\right); \boldsymbol {x} (0), u _ {j} ^ {2} \mathbf {I}\right) = \mathcal {N} \left(\sqrt {\bar {\alpha} _ {j} ^ {\prime}} \boldsymbol {x} \left(u _ {j}\right); \sqrt {\bar {\alpha} _ {j} ^ {\prime}} \boldsymbol {x} (0), \left(1 - \bar {\alpha} _ {j} ^ {\prime}\right) \mathbf {I}\right)\tag{253}
$$

$$
= \mathcal {N} \left(\boldsymbol {x} (u _ {j}); \boldsymbol {x} (0), \frac {1 - \bar {\alpha} _ {j} ^ {\prime}}{\bar {\alpha} _ {j} ^ {\prime}} \mathbf {I}\right).\tag{254}
$$

Matching the variances and solving for $\bar { \alpha } _ { j } ^ { \prime }$ gives 

$$
u _ {j} ^ {2} = (1 - \bar {\alpha} _ {j} ^ {\prime}) / \bar {\alpha} _ {j} ^ {\prime}\tag{255}
$$

$$
u _ {j} ^ {2} \bar {\alpha} _ {j} ^ {\prime} = 1 - \bar {\alpha} _ {j} ^ {\prime}\tag{256}
$$

$$
u _ {j} ^ {2} \bar {\alpha} _ {j} ^ {\prime} + \bar {\alpha} _ {j} ^ {\prime} = 1\tag{257}
$$

$$
\left(u _ {j} ^ {2} + 1\right) \bar {\alpha} _ {j} ^ {\prime} = 1\tag{258}
$$

$$
\bar {\alpha} _ {j} ^ {\prime} = 1 / (u _ {j} ^ {2} + 1).\tag{259}
$$

Finally, we can expand the left-hand side using Eq. 245 and solve for $u _ { j - 1 } \colon$ 

$$
\bar {\alpha} _ {j + 1} ^ {\prime} \max (\bar {\alpha} _ {j} / \bar {\alpha} _ {j + 1}, C _ {1}) = 1 / (u _ {j} ^ {2} + 1)\tag{260}
$$

$$
\bar {\alpha} _ {j} ^ {\prime} \max (\bar {\alpha} _ {j - 1} / \bar {\alpha} _ {j}, C _ {1}) = 1 / (u _ {j - 1} ^ {2} + 1)\tag{261}
$$

$$
\left[ 1 / (u _ {j} ^ {2} + 1) \right] \max (\bar {\alpha} _ {j - 1} / \bar {\alpha} _ {j}, C _ {1}) = 1 / (u _ {j - 1} ^ {2} + 1)\tag{262}
$$

$$
\max (\bar {\alpha} _ {j - 1} / \bar {\alpha} _ {j}, C _ {1}) (u _ {j - 1} ^ {2} + 1) = u _ {j} ^ {2} + 1\tag{263}
$$

$$
u _ {j - 1} ^ {2} + 1 = (u _ {j} ^ {2} + 1) / \max (\bar {\alpha} _ {j - 1} / \bar {\alpha} _ {j}, C _ {1})\tag{264}
$$

$$
u _ {j - 1} = \sqrt {\frac {u _ {j} ^ {2} + 1}{\max (\bar {\alpha} _ {j - 1} / \bar {\alpha} _ {j} , C _ {1})} - 1},\tag{265}
$$

giving a recurrence formula for $\{ u _ { j } \}$ , bootstrapped by $u _ { M } = 0$ , that matches the “Time steps” row of Table 1. 

## C.3.3 iDDPM preconditioning and training

We can solve $D _ { \theta } ( \cdot )$ from Eq. 227 by substituting $\sigma ( t ) = t$ from Appendix C.3.1: 

$$
\epsilon_ {\theta} ^ {(j)} \left(\boldsymbol {x} / \sqrt {\sigma^ {2} + 1}\right) = \left(\boldsymbol {x} - D _ {\theta} (\boldsymbol {x}; \sigma)\right) / \sigma\tag{266}
$$

$$
D _ {\theta} (\boldsymbol {x}; \sigma) = \boldsymbol {x} - \sigma \epsilon_ {\theta} ^ {(j)} \left(\boldsymbol {x} / \sqrt {\sigma^ {2} + 1}\right).\tag{267}
$$

We choose to define $F _ { \theta } ( \cdot ; j ) = \epsilon _ { \theta } ^ { ( j ) } ( \cdot )$ and solve j from σ by finding the nearest $u _ { j } \colon$ 

$$
D _ {\theta} (\boldsymbol {x}; \sigma) = \underbrace {1 \cdot} _ {c _ {\text {skip}}} \boldsymbol {x} \underbrace {- \sigma} _ {c _ {\text {out}}} \cdot F _ {\theta} \Big (\underbrace {\frac {1}{\sqrt {\sigma^ {2} + 1}}} _ {c _ {\text {in}}} \cdot \boldsymbol {x}; \underbrace {\arg \min _ {j} | u _ {j} - \sigma |} _ {c _ {\text {noise}}} \Big),\tag{268}
$$

where $c _ { \mathrm { s k i p } } , c _ { \mathrm { o u t } } , c _ { \mathrm { i n } }$ , and $\mathrm { \vec { { C } } _ { n o i s e } }$ match the “Network and preconditioning” section of Table 1. 

Note that Eq. 268 is identical to the VP preconditioning formula in Eq. 181. Furthermore, Nichol and Dhariwal [37] define their main training loss $L _ { \mathrm { s i m p l e } }$ (Eq. 14 in [37]) the same way as Song et al. [49], with σ drawn uniformly from $\{ u _ { j } \}$ . Thus, we can reuse Eq. 190 with $\sigma = u _ { j }$ $j \sim \mathcal { U } ( 0 , M - 1 )$ , and $\lambda ( \sigma ) = 1 / \sigma ^ { 2 }$ , matching the “Training” section of Table 1. In addition to $L _ { \mathrm { s i m p l e } }$ , Nichol and Dhariwal [37] also employ a secondary loss term $L _ { \mathrm { v l b } } ;$ we refer the reader to Section 3.1 in [37] for details. 

## C.3.4 iDDPM practical considerations

The pre-trained iDDPM model that we use on ImageNet-64 corresponds to the “ADM (dropout)” checkpoint<sup>17</sup> provided by Dhariwal and Nichol [9]. It contains 296 million trainable parameters and supports a discrete set of $M = 1 0 0 0$ noise levels $\bar { \sigma } \in \{ u _ { j } \} \approx \{ 2 0 2 9 1 , 6 4 2 , 3 2 1 , 2 1 4 ,$ , 160, 128, 106, $9 2 , 8 0 , 7 1 , \dots , 0 . 0 0 6 4 \}$ . The fact that we can only evaluate $F _ { \theta }$ these specific choices of σ presents three practical challenges: 

1. In the context of DDIM, we must choose how to resample $\{ u _ { j } \}$ to yield $\{ t _ { i } \}$ for $N \neq M$ Song et al. [47] employ a simple resampling scheme where $t _ { i } = u _ { k } .$ <sub>i</sub> for resampling factor $k \in \mathbb { Z } ^ { + }$ . This scheme, however, requires that $1 0 0 0 \equiv 0$ (mod $N )$ , which limits the possible choices for N considerably. Nichol and Dhariwal [37], on the other hand, employ a more flexible scheme where $t _ { i } = u _ { j }$ with $j = \lfloor ( M - 1 ) / ( N - 1 ) \cdot i \rfloor$ . We note, however, that in practice the values of $u _ { j < 8 }$ are considerably larger than our preferred $\sigma _ { \mathrm { m a x } } = 8 0$ . We choose to skip these values by defining $j = \bar { \lfloor \ j _ { 0 } + ( M - 1 - j _ { 0 } ) \bar { / } ( N - 1 ) \cdot i \rfloor }$ with $j _ { 0 } = 8$ matching the “Time steps” row in Table 1. In Figure 2c, the differences between the original sampler (blue) and our reimplementation (orange) are explained by this choice. 

2. In the context of our time step discretization (Eq. 5), we must ensure that $\sigma _ { i } ~ \in ~ \{ u _ { j } \}$ We accomplish this by rounding each $\sigma _ { i }$ to its nearest supported counterpart, $\mathrm { i } . \mathrm { e } . , \sigma _ { i } \gets$ $\begin{array} { r } { u _ { \mathrm { a r g m i n } _ { j } } \left| u _ { j } - \sigma _ { i } \right| } \end{array}$ , and setting $\sigma _ { \mathrm { m i n } } = 0 . 0 0 6 4 ~ \approx ~ u _ { N - 1 }$ . This is sufficient, because Algorithm 1 only evaluates $D _ { \theta } ( \cdot ; \sigma )$ with $\sigma \in \{ \sigma _ { i < N } \}$ 

3. In the context of our stochastic sampler, we must ensure that $\hat { t } _ { i } \in \{ u _ { j } \}$ . We accomplish this by replacing line 5 of Algorithm 2 with $\hat { t } _ { i } \gets u _ { \mathrm { a r g m i n } _ { j } } \ : | u _ { j } - ( t _ { i } + \gamma _ { i } t _ { i } ) |$ 

With these changes, we are able to import the pre-trained model directly as $F _ { \theta } ( \cdot )$ and run Algorithms 1 and 2 using the definitions in Table 1. Note that the model outputs both $\epsilon _ { \theta } ( \cdot )$ and $\Sigma _ { \theta } ( \cdot )$ , as described in Section 3.1 of [37]; we use only the former and ignore the latter. 

## D Further analysis of deterministic sampling

## D.1 Truncation error analysis and choice of discretization parameters

As discussed in Section 3, the fundamental reason why diffusion models tend to require a large number of sampling steps is that any numerical ODE solver is necessarily an approximation; the larger the steps, the farther away we drift from the true solution at each step. Specifically, given the value of $\mathbf { \delta } _ { \mathbf { \mathcal { X } } i - 1 }$ at time step $i - 1$ , the solver approximates the true $\boldsymbol { \mathscr { x } } _ { i } ^ { * }$ as $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ , resulting in local truncation error $\pmb { \tau } _ { i } = \pmb { x } _ { i } ^ { * } - \pmb { x } _ { i }$ . The local errors get accumulated over the $\dot { N }$ steps, ultimately leading to global truncation error $e _ { N }$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/8de753858d339e227f8ec7b268264418fe51441278721d56b28dcd61eeeb5395.jpg)



(a) Truncation error, VE + Euler


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/9fa2e32c0ac2a14589e15fe0e78b6806c516d29d2f4449724597eaaeff3aed63.jpg)



(b) Truncation error, VE + Heun


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/9252988f6f67f26d9575efae54ecfe92eaeaf47fad268894751bfc4d0f29edbf.jpg)



(c) FID as a function of ρ



Figure 13: (a) Local truncation error (y-axis) at different noise levels (x-axis) using Euler’s method with the VE-based CIFAR-10 model. Each curve corresponds to a different time step discretization, defined for $N = 6 4$ and a specific choice for the polynomial exponent $\rho .$ The values represent the root mean square error (RMSE) between one Euler iteration and a sequence of multiple smaller Euler iterations, representing the ground truth. The shaded regions, barely visible at low $\sigma ,$ , represent standard deviation over different latents $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ . (b) Corresponding error curves for Heun’s $2 ^ { \mathrm { n d } }$ order method (Algorithm 1). (c) FID (y-axis) as a function of the polynomial exponent (x-axis) for different models, measured using Heun $\mathbf { \bar { s } } \ 2 ^ { \mathrm { n d } }$ order method. The shaded regions indicate the range of variation between the lowest and highest observed FID, and the dots indicate the value of $\rho$ that we use in all other experiments.


Euler’s method is a first order ODE solver, meaning that $\tau _ { i } = \mathcal { O } \left( h _ { i } ^ { 2 } \right)$ for any sufficiently smooth ${ \pmb x } ( t )$ , where $h _ { i } = | t _ { i } - t _ { i - 1 } |$ is the local step size [50]. In other words, there exist some $C$ and H such that $| | \pmb { \tau } _ { i } | | < C h _ { i } ^ { 2 }$ for every $h _ { i } < H , \mathrm { i . e . }$ , halving $h _ { i }$ reduces $\tau _ { i }$ by $4 \times$ . Furthermore, if we assume that $D _ { \theta }$ is Lipschitz continuous — which is true for all network architectures considered in this paper — the global truncation error is bounded by $\left\| e _ { N } \right\| \le E \operatorname* { m a x } _ { i } \left\| \tau _ { i } \right\|$ , where the value of E depends on $N , t _ { 0 } , t _ { N }$ , and the Lipschitz constant [50]. Thus, reducing the global error for given $N$ , which in turn enables reducing N itself, boils down to choosing the solver and $\{ t _ { i } \}$ so that max<sub>i</sub> $| | \tau _ { i } | |$ is minimized. 

To gain insight on how the local truncation error behaves in practice, we measure the values of $\tau _ { i }$ over different noise levels using the VE-based CIFAR-10 model. For a given noise level, we set $t _ { i } = \sigma ^ { - 1 } ( \sigma _ { i } )$ and choose some $t _ { i - 1 } ~ > ~ t _ { i }$ depending on the case. We then sample $\mathbf { \delta } _ { \mathbf { \mathcal { X } } i - 1 }$ from $p ( \pmb { x } ; \sigma _ { i - 1 } )$ and estimate the true x<sup>∗</sup> by performing 200 Euler steps over uniformly selected subintervals between $t _ { i - 1 }$ and t. Finally, we plot the mean and standard deviation of the root mean square error $( \mathrm { R M S E } ) , \mathrm { i . e . , } | | \pmb { \tau } _ { i } | | / \sqrt { \mathrm { d i m } \tau } .$ , as a function of $\sigma _ { i } ,$ , averaged over 200 random samples of $\mathbf { \delta } _ { \mathbf { \mathcal { X } } i - 1 }$ . Results for Euler’s method are shown in Figure 13a, where the blue curve corresponds to uniform step size $h _ { \sigma } = 1 . 2 5$ with respect to σ, i.e., $\sigma _ { i - 1 } = \sigma _ { i } + h _ { \sigma }$ and $t _ { i - 1 } = \sigma ^ { - 1 } ( \sigma _ { i - 1 } )$ . We see that the error is very large $( \mathrm { R M S E } \approx 0 . 5 6 )$ for low noise levels $( \sigma _ { i } \le 0 . 5 )$ and considerably smaller for high noise levels. This is in line with the common intuition that, in order to reduce $e _ { N }$ , the step size should be decreased monotonically with decreasing $\sigma .$ Each curve is surrounded by a shaded region that indicates standard deviation, barely visible at low values of $\sigma .$ . This indicates that $\tau _ { i }$ is nearly constant with respect to $\mathbf { \delta } _ { \mathbf { x } _ { i - 1 } }$ , and thus there would be no benefit in varying $\{ t _ { i } \}$ schedule on a per-sample basis. 

A convenient way to vary the local step size depending on the noise level is to define $\{ \sigma _ { i } \}$ as a linear resampling of some monotonically increasing, unbounded warp function $w ( z )$ . In other words, $\sigma _ { i < N } = w ( A i + B )$ and $\sigma _ { N } = 0$ , where constants A and $B$ are selected so that $\sigma _ { 0 } = \sigma _ { \operatorname* { m a x } }$ and $\sigma _ { N - 1 } = \sigma _ { \operatorname* { m i n } }$ . In practice, we set $\sigma _ { \mathrm { m i n } } = \operatorname* { m a x } ( \sigma _ { \mathrm { l o } } , 0 . 0 0 2 )$ and $\sigma _ { \mathrm { m a x } } = \mathrm { m i n } ( \sigma _ { \mathrm { h i } } , 8 0 )$ , where $\sigma _ { \mathrm { l o } }$ and $\sigma _ { \mathrm { h i } }$ are the lowest and highest noise levels supported by a given model, respectively; we have found these choices to perform reasonably well in practice. Now, to balance $\tau _ { i }$ between low and high noise levels, we can, for example, use a polynomial warp function $w ( z ) = z ^ { \rho }$ parameterized by the exponent $\rho .$ This choice leads to the following formula for $\{ \sigma _ { i } \}$ : 

$$
\sigma_ {i <   N} = \left(\sigma_ {\max} ^ {\frac {1}{\rho}} + \frac {i}{N - 1} \left(\sigma_ {\min} ^ {\frac {1}{\rho}} - \sigma_ {\max} ^ {\frac {1}{\rho}}\right)\right) ^ {\rho}, \sigma_ {N} = 0,\tag{269}
$$

which reduces to uniform discretization when $\rho = 1$ and gives more and more emphasis to low noise levels as $\rho$ increases. $^ { 1 8 }$ 

Based on the value of $\sigma _ { i } .$ , we can now compute $\sigma _ { i - 1 } = ( \sigma _ { i } ^ { 1 / \rho } - A ) ^ { \rho }$ , which enables us to visualize $\tau _ { i }$ for different choices of $\rho$ in Figure 13a. We see that increasing ρ reduces the error for low noise levels $( \sigma < 1 0 )$ while increasing it for high noise levels $( \sigma > 1 0 )$ . Approximate balance is achieved at $\rho = 2$ , but RMSE remains relatively high $( \sim 0 . 0 3 )$ , meaning that Euler’s method drifts away from the correct result by several ULPs at each step. While the error could be reduced by increasing $N$ we would ideally like the RMSE to be well below 0.01 even with low step counts. 

Heun’s method introduces an additional correction step for $\pmb { x } _ { i + 1 }$ to account for the fact that $\mathrm { d } { \boldsymbol { x } } / \mathrm { d } t$ may change between $t _ { i }$ and $t _ { i + 1 } ;$ Euler’s method assumes it to be constant. The correction leads to cubic convergence of the local truncation error, i.e., $\tau _ { i } = \mathcal { O } \left( h _ { i } ^ { 3 } \right)$ , at the cost of one additional evaluation of $D _ { \theta }$ per step. We discuss the general family of Heun-like schemes later in Appendix D.2. Figure 13b shows local truncation error for Heun’s method using the same setup as Figure 13a. We see that the differences in $| | \tau _ { i } | |$ are generally more pronounced, which is to be expected given the quadratic vs. cubic convergence of the two methods. Cases where Euler’s method has low RMSE tend to have even lower RMSE with Heun’s method, and vice versa for cases with high RMSE. Most remarkably, the red curve shows almost constant RMSE $\in [ 0 . 0 0 3 0 , 0 . 0 0 4 5 ]$ . This means that the combination of Eq. 269 and Heun’s method is, in fact, very close to optimal with $\rho = 3$ 

Thus far, we have only considered the raw numerical error, i.e., component-wise deviation from the true result in RGB space. The raw numerical error is relevant for certain use cases, e.g., image manipulation where the ODE is first evaluated in the direction of increasing t and then back to $t = 0$ again — in this case, $\left| \left| e _ { N } \right| \right|$ directly tells us how much the original image degrades in the process and we can use $\rho = 3$ to minimize it. Considering the generation of novel images from scratch, however, it is reasonable to expect different noise levels to introduce different kinds of errors that may not necessarily be on equal footing considering their perceptual importance. We investigate this in Figure 13c, where we plot FID as a function of $\rho$ for different models and different choices of $N$ Note that the ImageNet-64 model was only trained for a discrete set of noise levels; in order to use it with Eq. 269, we round each $t _ { i }$ to its nearest supported counterpart, i.e., $\begin{array} { r } { t _ { i } ^ { \prime } = u _ { \mathrm { a r g m i n } _ { j } } \left| u _ { j } - t _ { i } \right| . } \end{array}$ 

From the plot, we can see that even though $\rho = 3$ leads to relatively good FID, it can be reduced further by choosing $\rho > 3$ . This corresponds to intentionally introducing error at high noise levels to reduce it at low noise levels, which makes intuitive sense because the value of $\sigma _ { \mathrm { m a x } }$ is somewhat arbitrary to begin with — increasing $\sigma _ { \mathrm { m a x } }$ can have a large impact on $\lvert \lvert e _ { N } \rvert \rvert$ , but it does not affect the resulting image distribution nearly as much. In general, we have found $\rho = 7$ to perform reasonably well in all cases, and use this value in all other experiments. 

## D.2 General family of $2 ^ { \mathbf { n d } }$ order Runge–Kutta variants

Heun’s method illustrated in Algorithm 1 belongs to a family of explicit two-stage $2 ^ { \mathrm { n d } }$ order Runge– Kutta methods, each having the same computational cost. A common parameterization [50] of this family is, 

$$
\boldsymbol {d} _ {i} = f (\boldsymbol {x} _ {i}; t _ {i}) \quad ; \quad \boldsymbol {x} _ {i + 1} = \boldsymbol {x} _ {i} + h \left[ \left(1 - \frac {1}{2 \alpha}\right) \boldsymbol {d} _ {i} + \frac {1}{2 \alpha} f (\boldsymbol {x} _ {i} + \alpha h \boldsymbol {d} _ {i}; t _ {i} + \alpha h) \right],\tag{270}
$$

where $h = t _ { i + 1 } - t _ { i }$ and α is a parameter that controls where the additional gradient is evaluated and how much it influences the step taken. Setting $\alpha = 1$ corresponds to Heun’s method, and $\begin{array} { r } { \alpha = \frac { 1 } { 2 } } \end{array}$ and $\textstyle \alpha = { \frac { 2 } { 3 } }$ yield so-called midpoint and Ralston methods, respectively. All these variants differ in the kind of approximation error they incur due to the geometry of the underlying function $f .$ 

To establish the optimal α in our use case, we ran a separate series of experiments. According to the results, it appears that $\alpha = 1$ is very close to being optimal. Nonetheless, the experimentally best choice was $\alpha = 1 . 1$ that performed slightly better, even though values greater than one are theoretically hard to justify as they overshoot the target $t _ { i + 1 }$ . As we have no good explanation for this observation and cannot tell if it holds in general, we chose not to make α a new hyperparameter and instead fixed it to 1, corresponding exactly to Heun’s method. Further analysis is left as future work, including the possibility of having α vary during sampling. 

Algorithm 3 Deterministic sampling using general $2^{\text{nd}}$ order Runge-Kutta, $\sigma(t) = t$ and $s(t) = 1$ .

1: procedure ALPHASAMPLER( $D_{\theta}(\boldsymbol{x}; \sigma)$ , $t_{i \in \{0, \dots, N\}}$ , $\alpha$ )
2: sample $\boldsymbol{x}_0 \sim \mathcal{N}(\boldsymbol{0}, t_0^2 \mathbf{I})$ 3: for $i \in \{0, \dots, N-1\}$ do
4: $h_i \leftarrow t_{i+1} - t_i$ ▷ Step length
5: $\boldsymbol{d}_i \leftarrow (\boldsymbol{x}_i - D_{\theta}(\boldsymbol{x}_i; t_i)) / t_i$ ▷ Evaluate $\mathrm{d}\boldsymbol{x}/\mathrm{d}t$ at $(\boldsymbol{x}, t_i)$ 6: $(\boldsymbol{x}_i', t_i') \leftarrow (\boldsymbol{x}_i + \alpha h \boldsymbol{d}_i, t_i + \alpha h)$ ▷ Additional evaluation point
7: if $t_i' \neq 0$ then
8: $\boldsymbol{d}_i' \leftarrow (\boldsymbol{x}_i' - D_{\theta}(\boldsymbol{x}_i'; t_i')) / t_i'$ ▷ Evaluate $\mathrm{d}\boldsymbol{x}/\mathrm{d}t$ at $(\boldsymbol{x}_i', t_i')$ 9: $\boldsymbol{x}_{i+1} \leftarrow \boldsymbol{x}_i + h\left[\left(1 - \frac{1}{2\alpha}\right) \boldsymbol{d}_i + \frac{1}{2\alpha} \boldsymbol{d}_i'\right]$ ▷ Second order step from $t_i$ to $t_{i+1}$ 10: else
11: $\boldsymbol{x}_{i+1} \leftarrow \boldsymbol{x}_i + h \boldsymbol{d}_i$ ▷ Euler step from $t_i$ to $t_{i+1}$ 12: return $\boldsymbol{x}_N$ 

An additional benefit of setting $\alpha = 1$ is that it makes it possible to use pre-trained neural networks $D _ { \theta } ( \pmb { x } ; \sigma )$ that have been trained only for specific values of $\sigma$ . This is because a Heun step evaluates the additional gradient at exactly $t _ { i + 1 }$ unlike the other $2 ^ { \mathrm { n d } }$ order variants. Hence it is sufficient to ensure that each $t _ { i }$ corresponds to a value of σ that the network was trained for. 

Algorithm 3 shows the pseudocode for a general $2 ^ { \mathrm { n d } }$ order solver parameterized by α. For clarity, the pseudocode assumes the specific choices of $\sigma ( t ) = t$ and $s ( t ) = 1$ that we advocate in Section 3. Note that the fallback to Euler step (line 11) can occur only when $\alpha \geq 1$ 

## E Further results with stochastic sampling

## E.1 Image degradation due to excessive stochastic iteration

Figure 14 illustrates the image degradation caused by excessive Langevin iteration (Section 4, “Practical considerations”). These images are generated by doing a specified number of iterations at a fixed noise level σ so that at each iteration an equal amount of noise is added and removed. In theory, Langevin dynamics should bring the distribution towards the ideal distribution $p ( { \pmb x } ; { \boldsymbol \sigma } )$ but as noted in Section 4, this holds only if the denoiser $D _ { \theta } ( { \pmb x } ; { \boldsymbol \sigma } )$ induces a conservative vector field in Eq. 3. 

As seen in the figure, it is clear that the image distribution suffers from repeated iteration in all cases, although the exact failure mode depends on dataset and noise level. For low noise levels (below 0.2 or so), the images tend to oversaturate starting at 2k iterations and become fully corrupted after that. Our heuristic of setting $S _ { \mathrm { t m i n } } > 0$ is designed to prevent stochastic sampling altogether at very low noise levels to avoid this effect. 

For high noise levels, we can see that iterating without the standard deviation correction, i.e., when $S _ { \mathrm { n o i s e } } = 1 . 0 0 0$ , the images tend to become more abstract and devoid of color at high iteration counts; this is especially visible in the 10k column of CIFAR-10 where the images become mostly black and white with no discernible backgrounds. Our heuristic inflation of standard deviation by setting $S _ { \mathrm { n o i s e } } > 1$ counteracts this tendency efficiently, as seen in the corresponding images on the right hand side of the figure. Notably, this still does not fix the oversaturation and corruption at low noise levels, suggesting multiple sources for the detrimental effects of excessive iteration. Further research will be required to better understand the root causes of these observed effects. 

Figure 15 presents the output quality of our stochastic sampler in terms of FID as a function of $S _ { \mathrm { c h u r n } }$ at fixed NFE, using pre-trained networks of Song et al. [49] and Dhariwal and Nichol [9]. Generally, for each case and combination of our heuristic corrections, there is an optimal amount of stochasticity after which the results start to degrade. It can also be seen that regardless of the value of $S _ { \mathrm { c h u r n } } .$ the best results are obtained by enabling all corrections, although whether $S _ { \mathrm { n o i s e } } \ \mathrm { o r } \ S _ { \mathrm { t m i n , t m a x } }$ is more important depends on the case. 


Uncond. CIFAR-10, Pre-trained, VP, S = 1.000



Uncond. CIFAR-10, Pre-trained, VP, S = 1.007


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/2163afd3d7373f5ad1e921923bba76b0f8e871869cf4decae41eaf7c14f582ec.jpg)



Figure 14: Gradual image degradation with repeated addition and removal of noise. We start with a random image drawn from $p ( { \pmb x } ; { \boldsymbol \sigma } )$ (first column) and run Algorithm 2 for a certain number of steps (remaining columns) with fixed $\gamma _ { i } = \sqrt { 2 } - 1$ . Each row corresponds to a specific choice of σ (indicated in the middle) that we keep fixed throughout the entire process. We visualize the results after running them through the denoiser, i.e., $D _ { \theta } ( \pmb { x } _ { i } ; \sigma )$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/124946e2b93bb7b3da867cc58cc901722192fa4d503fe8a84bfa94223f9b7456.jpg)



(a) Uncond. CIFAR-10, VP


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/24d395665d9f9066e3c027c42dfbf109c084d3579463a9c7e60e720b5be6e305.jpg)



(b) Uncond. CIFAR-10, VE


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-16/dcdec356-a541-4beb-9d26-268461f841cb/30da1187c7fb5e6f015e5c845366041fda002ee1718fd93864204bec5a7dde80.jpg)



(c) Class-cond. ImageNet-64



Figure 15: Ablations of our stochastic sampler (Algorithm 2) parameters using pre-trained networks of Song et al. [49] and Dhariwal and Nichol [9]. Each curve shows FID (y-axis) as a function of $S _ { \mathrm { c h u r n } } \left( x { \cdot } \mathrm { a x i s } \right)$ for $N = 2 5 6$ steps $( \mathrm { N F E } = 5 1 1 )$ . The dashed red lines correspond to our deterministic sampler (Algorithm 1), equivalent to setting $S _ { \mathrm { c h u r n } } = 0$ . The purple curves correspond to optimal choices for $\bar { \{ S _ { \mathrm { t m i n } } , S _ { \mathrm { t m a x } } , S _ { \mathrm { n o i s e } } \} }$ , found separately for each case using grid search. Orange, blue, and green correspond to disabling the effects of $S _ { \mathrm { t m i n , t m a x } }$ and/or $S _ { \mathrm { n o i s e } } .$ . The shaded regions indicate the range of variation between the lowest and highest observed FID.



Table 5: Parameters used for the stochastic sampling experiments in Section 4.


<table><tr><td rowspan="2">Parameter</td><td colspan="2">CIFAR-10</td><td colspan="2">ImageNet</td><td rowspan="2">Grid search</td></tr><tr><td>VP</td><td>VE</td><td>Pre-trained</td><td>Our model</td></tr><tr><td><eq>S_{churn}</eq></td><td>30</td><td>80</td><td>80</td><td>40</td><td>0, 10, 20, 30, ..., 70, 80, 90, 100</td></tr><tr><td><eq>S_{tmin}</eq></td><td>0.01</td><td>0.05</td><td>0.05</td><td>0.05</td><td>0, 0.005, 0.01, 0.02, ..., 1, 2, 5, 10</td></tr><tr><td><eq>S_{tmax}</eq></td><td>1</td><td>1</td><td>50</td><td>50</td><td>0.2, 0.5, 1, 2, ..., 10, 20, 50, 80</td></tr><tr><td><eq>S_{noise}</eq></td><td>1.007</td><td>1.007</td><td>1.003</td><td>1.003</td><td>1.000, 1.001, ..., 1.009, 1.010</td></tr></table>

## E.2 Stochastic sampling parameters

Table 5 lists the values for $S _ { \mathrm { c h u r n } } , S _ { \mathrm { t m i n } } , S _ { \mathrm { t m a x } } ,$ and $S _ { \mathrm { n o i s e } }$ that we used in our stochastic sampling experiments. These were determined with a grid search over the combinations listed in the rightmost column. It can be seen that the optimal parameters depend on the case; better understanding of the degradation phenomena will hopefully give rise to more direct ways of handling the problem in the future. 

## F Implementation details

We implemented our techniques in a newly written codebase, based loosely on the original implementations by Song et al.<sup>19</sup> [49], Dhariwal and Nichol<sup>20</sup> [9], and Karras et al.<sup>21</sup> [26]. We performed extensive testing to verify that our implementation produced exactly the same results as previous work, including samplers, pre-trained models, network architectures, training configurations, and evaluation. We ran all experiments using PyTorch 1.10.0, CUDA 11.4, and CuDNN 8.2.0 on NVIDIA DGX-1’s with 8 Tesla V100 GPUs each. 

Our implementation and pre-trained models are available at https://github.com/NVlabs/edm 


Table 6: Our augmentation pipeline. Each training image undergoes a combined geometric transformation based on 8 random parameters that receive non-zero values with a certain probability. The model is conditioned with an additional 9-dimensional input vector derived from these parameters.


<table><tr><td>Augmentation</td><td>Transformation</td><td>Parameters</td><td>Prob.</td><td>Conditioning</td><td>Constants</td></tr><tr><td>x-flip</td><td><eq>\text{SCALE2D}(1 - 2a_0, 1)</eq></td><td><eq>a_0 \sim \mathcal{U}\{0,1\}</eq></td><td>100%</td><td><eq>a_0</eq></td><td><eq>A_{\text{prob}} = 12\%</eq></td></tr><tr><td>y-flip</td><td><eq>\text{SCALE2D}(1, 1 - 2a_1)</eq></td><td><eq>a_1 \sim \mathcal{U}\{0,1\}</eq></td><td><eq>A_{\text{prob}}</eq></td><td><eq>a_1</eq></td><td>or 15%</td></tr><tr><td>Scaling</td><td><eq>\text{SCALE2D}((A_{\text{scale}})^{a_2}, (A_{\text{scale}})^{a_2})</eq></td><td><eq>a_2 \sim \mathcal{N}(0,1)</eq></td><td><eq>A_{\text{prob}}</eq></td><td><eq>a_2</eq></td><td><eq>A_{\text{scale}} = 2^{0.2}</eq></td></tr><tr><td>Rotation</td><td><eq>\text{ROTATE2D}(-a_3)</eq></td><td><eq>a_3 \sim \mathcal{U}(-\pi,\pi)</eq></td><td><eq>A_{\text{prob}}</eq></td><td><eq>\cos a_3 - 1 \sin a_3</eq></td><td></td></tr><tr><td>Anisotropy</td><td><eq>\text{ROTATE2D}(a_4)</eq><eq>\text{SCALE2D}((A_{\text{aniso}})^{a_5}, 1/(A_{\text{aniso}})^{a_5})</eq><eq>\text{ROTATE2D}(-a_4)</eq></td><td><eq>a_4 \sim \mathcal{U}(-\pi,\pi)</eq><eq>a_5 \sim \mathcal{N}(0,1)</eq></td><td><eq>A_{\text{prob}}</eq></td><td><eq>a_5 \cos a_4</eq><eq>a_5 \sin a_4</eq></td><td><eq>A_{\text{aniso}} = 2^{0.2}</eq></td></tr><tr><td>Translation</td><td><eq>\text{TRANSLATE2D}((A_{\text{trans}}) a_6, (A_{\text{trans}}) a_7)</eq></td><td><eq>a_6 \sim \mathcal{N}(0,1)</eq><eq>a_7 \sim \mathcal{N}(0,1)</eq></td><td><eq>A_{\text{prob}}</eq></td><td><eq>a_6</eq><eq>a_7</eq></td><td><eq>A_{\text{trans}} = 1/8</eq></td></tr></table>

## F.1 FID calculation

We calculate FID [15] between 50,000 generated images and all available real images, without any augmentation such as x-flips. We use the pre-trained Inception-v3 model provided with StyleGAN3<sup>22</sup> [26] that is, in turn, a direct PyTorch translation of the original TensorFlow-based model<sup>23</sup>. We have verified that our FID implementation produces identical results compared to Dhariwal and Nichol [9] and Karras et al. [26]. To reduce the impact of random variation, typically in the order of ±2%, we compute FID three times in each experiment and report the minimum. We also highlight the difference between the highest and lowest achieved FID in Figures 4, 5b, 13c, and 15. 

## F.2 Augmentation regularization

In Section 5, we propose to combat overfitting of $D _ { \theta }$ using conditional augmentation. We build our augmentation pipeline around the same concepts that were originally proposed by Karras et al. [25] in the context of GANs. In practice, we employ a set of 6 geometric transformations; we have found other types of augmentations, such as color corruption and image-space filtering, to be consistently harmful for diffusion-based models. 

The details of our augmentation pipeline are shown in Table 6. We apply the augmentations independently to each training image $y \sim p _ { \mathrm { d a t a } }$ prior to adding the noise $\pmb { n } \sim \bar { \mathcal { N } } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } )$ . First, we determine whether to enable or disable each augmentation based on a weighted coin toss. The probability of enabling a given augmentation (“Prob.” column) is fixed to 12% for CIFAR-10 and 15% for FFHQ and AFHQv2, except for x-flips that are always enabled. We then draw 8 random parameters from their corresponding distributions (“Parameters” column); if a given augmentation is disabled, we override the associated parameters with zero. Based on these, we construct a homogeneous 2D transformation matrix based on the parameters (“Transformation” column). This transformation is applied to the image using the implementation of [25] that employs 2× supersampled high-quality Wavelet filters. Finally, we construct a 9-dimensional conditioning input vector (“Conditioning” column) and feed it to the denoiser network, in addition to the image and noise level inputs. 

The role of the conditioning input is to present the network with a set of auxiliary tasks; in addition to the main task of modeling p(x; σ), we effectively ask the network to also model an infinite set of distributions $p ( { \pmb x } ; \sigma , { \pmb a } )$ for each possible choice of the augmentation parameters a. These auxiliary tasks provide the network with a large variety of unique training samples, preventing it from overfitting to any individual sample. Still, the auxiliary tasks appear to be beneficial for the main task; we speculate that this is because the denoising operation itself is similar for every choice of a. 


Table 7: Hyperparameters used for the training runs in Section 5.


<table><tr><td rowspan="2">Hyperparameter</td><td colspan="2">CIFAR-10</td><td colspan="2">FFHQ &amp; AFHQv2</td><td>ImagetNet</td></tr><tr><td>Baseline</td><td>Ours</td><td>Baseline</td><td>Ours</td><td>Ours</td></tr><tr><td>Number of GPUs</td><td>4</td><td>8</td><td>4</td><td>8</td><td>32</td></tr><tr><td>Duration (Mimg)</td><td>200</td><td>200</td><td>200</td><td>200</td><td>2500</td></tr><tr><td>Minibatch size</td><td>128</td><td>512</td><td>128</td><td>256</td><td>4096</td></tr><tr><td>Gradient clipping</td><td>√</td><td>-</td><td>√</td><td>-</td><td>-</td></tr><tr><td>Mixed-precision (FP16)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>√</td></tr><tr><td>Learning rate ×<eq>10^4</eq></td><td>2</td><td>10</td><td>2</td><td>2</td><td>1</td></tr><tr><td>LR ramp-up (Mimg)</td><td>0.64</td><td>10</td><td>0.64</td><td>10</td><td>10</td></tr><tr><td>EMA half-life (Mimg)</td><td>0.89 / 0.9(VP / VE)</td><td>0.5</td><td>0.89 / 0.9(VP / VE)</td><td>0.5</td><td>50</td></tr><tr><td>Dropout probability</td><td>10%</td><td>13%</td><td>10%</td><td>5% / 25%(FFHQ / AFHQ)</td><td>10%</td></tr><tr><td>Channel multiplier</td><td>128</td><td>128</td><td>128</td><td>128</td><td>192</td></tr><tr><td>Channels per resolution</td><td>1-2-2-2</td><td>2-2-2</td><td>1-1-2-2-2</td><td>1-2-2-2</td><td>1-2-3-4</td></tr><tr><td>Dataset x-flips</td><td>√</td><td>-</td><td>√</td><td>-</td><td>-</td></tr><tr><td>Augment probability</td><td>-</td><td>12%</td><td>-</td><td>15%</td><td>-</td></tr></table>

We have designed the conditioning input so that zero corresponds to the case where no augmentations were applied. During sampling, we simply set ${ \pmb a } = { \bf 0 }$ to obtain results consistent with the main task. We have not observed any leakage between the auxiliary tasks and the main task; the generated images exhibit no traces of out-of-domain geometric transformations even with $A _ { \mathrm { p r o b } } = 1 0 0 \%$ . In practice, this means that we are free to choose the constants $\{ A _ { \mathrm { p r o b } } , A _ { \mathrm { s c a l e } } , A _ { \mathrm { a n i s o } } , A _ { \mathrm { t r a n s } } \}$ any way we like as long as the results improve. Horizontal flips serve as an interesting example. Most of the prior work augments the training set with random x-flips, which is beneficial for most datasets but has the downside that any text or logos may appear mirrored in the generated images. With our non-leaky augmentations, we get the same benefits without the downsides by executing the x-flip augmentation with 100% probability. Thus, we rely exclusively on our augmentation scheme and disable dataset x-flips to ensure that the generated images stay true to the original distribution. 

## F.3 Training configurations

Table 7 shows the exact set of hyperparameters that we used in our training experiments reported in Section 5. We will first detail the configurations used with CIFAR-10, FFHQ, and AFHQv2, and then discuss the training of our improved ImageNet model. 

Config A of Table 2 (“Baseline”) corresponds to the original setup of Song et al. [49] for the two cases (VP and VE), and config F (“Ours”) corresponds to our improved setup. We trained each model until a total of 200 million images had been drawn from the training set, abbreviated as “200 Mimg” in Table 7; this corresponds to a total of ∼400,000 training iterations using a batch size of 512. We saved a snapshot of the model every 2.5 million images and reported results for the snapshot that achieved the lowest FID according to our deterministic sampler with NFE = 35 or NFE = 79, depending on the resolution. 

In config B, we re-adjust the basic hyperparameters to enable faster training and obtain a more meaningful point of comparison. Specifically, we increase the parallelism from 4 to 8 GPUs and batch size from 128 to 512 or 256, depending on the resolution. We also disable gradient clipping, i.e., forcing $\| \mathrm { d } \mathcal { L } ( D _ { \theta } ) / \mathrm { d } \theta \| _ { 2 } \leq 1$ , that we found to provide no benefit in practice. Furthermore, we increase the learning rate from 0.0002 to 0.001 for CIFAR-10, ramping it up during the first 10 million images, and standardize the half-life of the exponential moving average of θ to 0.5 million images. Finally, we adjust the dropout probability for each dataset as shown in Table 7 via a full grid search at 1% increments. Our total training time is approximately 2 days for CIFAR-10 at 32×32 resolution and 4 days for FFHQ and AFHQv2 at 64×64 resolution. 


Table 8: Details of the network architectures used in this paper.


<table><tr><td>Parameter</td><td>DDPM++ (VP)</td><td>NCSN++ (VE)</td><td>ADM (ImageNet)</td></tr><tr><td>Resampling filter</td><td>Box</td><td>Bilinear</td><td>Box</td></tr><tr><td>Noise embedding</td><td>Positional</td><td>Fourier</td><td>Positional</td></tr><tr><td>Skip connections in encoder</td><td>-</td><td>Residual</td><td>-</td></tr><tr><td>Skip connections in decoder</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Residual blocks per resolution</td><td>4</td><td>4</td><td>3</td></tr><tr><td>Attention resolutions</td><td>{16}</td><td>{16}</td><td>{32, 16, 8}</td></tr><tr><td>Attention heads</td><td>1</td><td>1</td><td>6-9-12</td></tr><tr><td>Attention blocks in encoder</td><td>4</td><td>4</td><td>9</td></tr><tr><td>Attention blocks in decoder</td><td>2</td><td>2</td><td>13</td></tr></table>

In config C, we improve the expressive power of the model by removing the 4×4 layers and doubling the capacity of the 16×16 layers instead; we found the former to mainly contribute to overfitting, whereas the latter were critical for obtaining high-quality results. The original models of Song et al. [49] employ 128 channels at 64×64 (where applicable) and $3 2 \times 3 2$ , and 256 channels at 16×16, 8×8, and 4×4. We change these numbers to 128 channels at 64×64 (where applicable), and 256 channels at 32×32, 16×16, and $8 \times 8 .$ . We abbreviate these counts in Table 7 as multiples of 128, listed from the highest resolution to the lowest. In practice, this rebalancing reduces the total number of trainable parameters slightly, resulting in ∼56 million parameters for each model at $3 2 \times 3 2$ resolution and ${ \sim } 6 2$ million parameters at 64×64 resolution. 

In config D, we replace the original preconditioning with our improved formulas (“Network and preconditioning” section in Table 1). In config $\mathrm { E , }$ we do the same for the noise distribution and loss weighting (“Training” section in Table 1). Finally, in config F, we enable augmentation regularization as discussed in Appendix F.2. The other hyperparameters remain the same as in config C. 

With ImageNet-64, it is necessary to train considerably longer compared to the other datasets in order to reach state-of-the-art results. To reduce the training time, we employed 32 NVIDIA Ampere GPUs (4 nodes) with a batch size of 4096 (128 per GPU) and utilized the high-performance Tensor Cores via mixed-precision FP16/FP32 training. In practice, we store the trainable parameters as FP32 but cast them to FP16 when evaluating $F _ { \theta }$ , except for the embedding and self-attention layers, where we found the limited exponent range of FP16 to occasionally lead to stability issues. We trained the model for two weeks, corresponding to ∼2500 million images drawn from the training set and ∼600,000 training iterations, using learning rate 0.0001, exponential moving average of 50 million images, and the same model architecture and dropout probability as Dhariwal and Nichol [9]. We did not find overfitting to be a concern, and thus chose to not employ augmentation regularization. 

## F.4 Network architectures

As a result of our training improvements, the VP and VE cases become otherwise identical in config F except for the network architecture; VP employs the DDPM++ architecture while VE employs NCSN++, both of which were originally proposed by Song et al. [49]. These architectures correspond to relatively straightforward variations of the same U-net backbone with three differences, as illustrated in Table 8. First, DDPM++ employs box filter [1, 1] for the upsampling and downsampling layers whereas NCSN++ employs bilinear filter [1, 3, 3, 1]. Second, DDPM++ inherits its positional encoding scheme for the noise level directly from DDPM [16] whereas NCSN++ replaces it with random Fourier features [52]. Third, NCSN++ incorporates additional residual skip connections from the input image to each block in the encoder, as explained in Appendix H of [49] (“progressive growing architectures”). 

For class conditioning and augmentation regularization, we extend the original DDPM++ and NCSN++ arhictectures by introducing two optional conditioning inputs alongside the noise level input. We represent class labels as one-hot encoded vectors that we first scale by $\sqrt { C }$ , where $C$ is the total number of classes, and then feed through a fully-connected layer. For the augmentation parameters, we feed the conditioning inputs of Appendix F.2 through a fully-connected layer as-is. 

We then combine the resulting feature vectors with the original noise level conditioning vector through elementwise addition. 

For class-conditional ImageNet-64, we use the ADM architecture of Dhariwal and Nichol [9] with no changes. The model has a total of ∼296 million trainable parameters. As detailed in Tables 7 and 8, the most notable differences to DDPM++ include the use of a slightly shallower model (3 residual blocks per resolution instead of 4) with considerably more channels (e.g., 768 in the lowest resolution instead of 256), more self-attention layers interspersed throughout the network (22 instead of 6), and the use of multi-head attention (e.g., 12 heads in the lowest resolution). We feel that the precise impact of architectural choices remains an interesting question for future work. 

## F.5 Licenses

Datasets: 

• CIFAR-10 [29]: MIT license 

• FFHQ [27]: Creative Commons BY-NC-SA 4.0 license 

• AFHQv2 [7]: Creative Commons BY-NC 4.0 license 

• ImageNet [8]: The license status is unclear 

Pre-trained models: 

• CIFAR-10 models by Song et al. [49]: Apache V2.0 license 

• ImageNet-64 model by Dhariwal and Nichol [9]: MIT license 

• Inception-v3 model by Szegedy et al. [51]: Apache V2.0 license 