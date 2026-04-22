## Page 1

## **Tutorial on Diffusion Models for Imaging and Vision**

Stanley Chan^(1)

January 9, 2025

**Abstract**. The astonishing growth of generative tools in recent years has empowered many exciting applications in text-to-image generation and text-to-video generation. The underlying principle behind these generative tools is the concept of* diffusion*, a particular sampling mechanism that has overcome some longstanding shortcomings in previous approaches. The goal of this tutorial is to discuss the essential ideas underlying these diffusion models. The target audience of this tutorial includes undergraduate and graduate students who are interested in doing research on diffusion models or applying these tools to solve other problems.

### **Contents**

**1 Variational Auto-Encoder (VAE) 2 **1.1 Building Blocks of VAE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2 1.2 Evidence Lower Bound . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5 1.3 Optimization in VAE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10 1.4 Concluding Remark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

**2 Denoising Diffusion Probabilistic Model (DDPM) 18 **2.1 Building Blocks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19 2.2 Evidence Lower Bound . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25 2.3 Distribution of the Reverse Process . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30 2.4 Training and Inference . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34 2.5 Predicting Noise . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37 2.6 Denoising Diffusion Implicit Model (DDIM) . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39 2.7 Concluding Remark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43

**3 Score-Matching Langevin Dynamics (SMLD) 44 **3.1 Sampling from a Distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44 3.2 (Stein’s) Score Function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48 3.3 Score Matching Techniques . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49 3.4 Concluding Remark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53

**4 Stochastic Differential Equation (SDE) 54 **4.1 From Iterative Algorithms to Ordinary Differential Equations . . . . . . . . . . . . . . . . . . 54 4.2 What is an SDE? . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56 4.3 Stochastic Differential Equation for DDPM and SMLD . . . . . . . . . . . . . . . . . . . . . . 58 4.4 Numerical Solvers for ODE and SDE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63 4.5 Concluding Remark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64

**5 Langevin and Fokker-Planck Equations 66 **5.1 Brownian Motion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 66 5.2 Masters Equation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 73 5.3 Kramers-Moyal Expansion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 77 5.4 Fokker-Planck Equation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 81 5.5 Concluding Remark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 86

**6 Conclusion 87**

1School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN 47907. Email: stanchan@purdue.edu.

© 2024 Stanley Chan. All Rights Reserved. 1

# arXiv:2403.18103v3  [cs.LG]  8 Jan 2025


---

## Page 2

### **1 Variational Auto-Encoder (VAE)**

A long time ago, in a galaxy far far away, we wanted to build a generator — a generator that generates texts, speeches, or images from some inputs with which we give to the computer. While this may sound magical at first, the problem has actually been studied for a long time. To kick off the discussion of this tutorial, we shall first consider the** variational autoencoder** (VAE). VAE was proposed by Kingma and Welling in 2014 [23]. According to their 2019 tutorial [24], the VAE was inspired by the Helmholtz Machine [10] as the marriage of graphical models and deep learning. In what follows, we will discuss VAE’s problem setting, its building blocks, and the optimization tools associated with the training.

**1.1 Building Blocks of VAE**

We start by discussing the schematic diagram of a VAE. As shown in the figure below, the VAE consists of a pair of models (often realized by deep neural networks). The one located near the input is called an** encoder **whereas the one located near the output is called a** decoder**. We denote the input (typically an image) as a vector** x**, and the output (typically another image) as a vector b**x**. The vector located in the middle between the encoder and the decoder is called a** latent variable**, denoted as** z**. The job of the encoder is to extract a meaningful representation for** x**, whereas the job of the decoder is to generate a new image from the latent variable** z**.

Figure 1.1: A variational autoencoder consists of an encoder that converts an input** x** to a latent variable** z**, and a decoder that synthesizes an output b**x** from the latent variable.

The latent variable** z** has two special roles in this setup. With respect to the input, the latent variable encapsulates the information that can be used to describe** x**. The encoding procedure could be a lossy process, but our goal is to preserve the important content of** x** as much as we can. With respect to the output, the latent variable serves as the “seed” from which an image b**x** can be generated. Two different** z**’s should in theory give us two different generated images. A slightly more formal definition of a latent variable is given below.

**Definition 1.1. Latent Variables[24]**. In a probabilistic model, latent variables** z** are variables that we do not observe and hence are not part of the training dataset, although they are part of the model.

**Example 1.1.** Getting a latent representation of an image is not an alien thing. Back in the time of JPEG compression (which is arguably a dinosaur), we used discrete cosine transform (DCT) basis func- tions*** φ****n* to encode the underlying image/patches of an image. The coefficient vector** z** = [*z*1*, . . . , z**N*]^(*T*)

is obtained by projecting the image** x** onto the space spanned by the basis, via* z**n* =* ⟨****φ****n**,*** x***⟩*. So, given an image** x**, we can produce a coefficient vector** z**. From** z**, we can use the inverse transform to recover (i.e. decode) the image.

Figure 1.2: In discrete cosine transform (DCT), we can think of the encoder as taking an image** x** and generating a latent variable** z** by projecting** x** onto the basis functions.

© 2024 Stanley Chan. All Rights Reserved. 2


---

## Page 3

In this example, the coefficient vector** z** is the latent variable. The encoder is the DCT transform, and the decoder is the inverse DCT transform.

The term “variational” in VAE is related to the subject of calculus of variations which studies opti- mization over functions. In VAE, we are interested in searching for the optimal probability distributions to describe** x** and** z**. In light of this, we need to consider a few distributions:

•* p*(**x**): The true distribution of** x**. It is never known. The whole universe of diffusion models is to find ways to draw samples from* p*(**x**). If we knew* p*(**x**) (say, we have a formula that describes* p*(**x**)), we can just draw a sample** x** that maximizes log* p*(**x**). •* p*(**z**): The distribution of the latent variable. Typically, we make it a zero-mean unit-variance Gaussian *N*(0*,*** I**). One reason is that linear transformation of a Gaussian remains a Gaussian, and so this makes the data processing easier. Doersch [12] also has an excellent explanation. It was mentioned that any distribution can be generated by mapping a Gaussian through a sufficiently complicated function. For example, in a one-variable setting, the inverse cumulative distribution function (CDF) technique [7, Chapter 4] can be used for any continuous distribution with an invertible CDF. In general, as long as we have a sufficiently powerful function (e.g., a neural network), we can learn it and map the i.i.d. Gaussian to whatever latent variable needed for our problem. •* p*(**z***|***x**): The conditional distribution associated with the** encoder**, which tells us the likelihood of** z **when given** x**. We have no access to it.* p*(**z***|***x**) itself is* not* the encoder, but the encoder has to do something so that it will behave consistently with* p*(**z***|***x**). •* p*(**x***|***z**): The conditional distribution associated with the** decoder**, which tells us the posterior proba- bility of getting** x** given** z**. Again, we have no access to it.

When we switch from the classical parameteric models to deep neural networks, the notion of latent variables is changed to* deep* latent variables. Kingma and Welling [24] gave a good definition below.

**Definition 1.2. Deep Latent Variables**[24]. Deep Latent Variables are latent variables whose distributions* p*(**z**),* p*(**x***|***z**), or* p*(**z***|***x**) are parameterized by a neural network.

The advantage of deep latent variables is that they can model very complex data distributions* p*(**x**) even though the structures of the prior distributions and the conditional distributions are relatively simple (e.g. Gaussian). One way to think about this is that the neural networks can be used to estimate the mean of a Gaussian. Although the Gaussian itself is simple, the mean is a function of the input data, which passes through a neural network to generate a data-dependent mean. So the expressiveness of the Gaussian is significantly improved. Let’s go back to the four distributions above. Here is a somewhat trivial but educational example that can illustrate the idea:

**Example 1.2.** Consider a random variable** X** distributed according to a Gaussian mixture model with a latent variable* z** ∈{*1*, . . . , K**}* denoting the cluster identity such that* p**Z*(*k*) = P[*Z* =* k*] =* π**k* for *k* = 1*, . . . , K*. We assume ^(P)^(*K *)*k*=1* *^(*π*)^(*k*)^( = 1. Then, if we are told that we need to look at the)^(* k*)^(-th cluster )only, the conditional distribution of** X** given* Z* is

*p***X***|**Z*(**x***|**k*) =* N*(**x*** |**** µ****k**, σ*^(2 )*k*^(**I**)^())^(*.*)

The marginal distribution of** x** can be found using the law of total probability, giving us

*p***X**(**x**) =

*K *X

*k*=1 *p***X***|**Z*(**x***|**k*)*p**Z*(*k*) =

*K *X

*k*=1 *π**k**N*(**x*** |**** µ****k**, σ*^(2 )*k*^(**I**)^())^(*. *)(1.1)

Therefore, if we start with* p***X**(**x**), the design question for the encoder is to build a magical encoder such that for every sample** x*** ∼**p***X**(**x**), the latent code will be* z** ∈{*1*, . . . , K**}* with a distribution* z** ∼**p**Z*(*k*). To illustrate how the encoder and decoder work, let’s assume that the mean and variance are known and are fixed. Otherwise we will need to estimate the mean and variance through an expectation-

© 2024 Stanley Chan. All Rights Reserved. 3


---

## Page 4

maximization (EM) algorithm. It is doable, but the tedious equations will defeat the educational purpose of this illustration. **Encoder**: How do we obtain* z* from** x**? This is easy because at the encoder, we know* p***X**(**x**) and *p**Z*(*k*). Imagine that you only have two classes* z** ∈{*1*,* 2*}*. Effectively you are just making a binary decision of where the sample** x** should belong to. There are many ways you can do the binary decision. If you like the maximum-a-posteriori decision rule, you can check

*p**Z**|***X**(1*|***x**) ≷^(class 1 )class 2* *^(*p*)*Z**|***X**^((2)^(*|*)^(**x**)^())^(*,*)

and this will return you a simple decision: You give us** x**, we tell you* z** ∈{*1*,* 2*}*. **Decoder**: On the decoder side, if we are given a latent code* z** ∈{*1*, . . . , K**}*, the magical decoder just needs to return us a sample** x** which is drawn from* p***X***|**Z*(**x***|**k*) =* N*(**x*** |**** µ****k**, σ*^(2 )*k*^(**I**)^(). A different)^(* z*)^( will )give us one of the* K* mixture components. If we have enough samples, the overall distribution will follow the Gaussian mixture.

This example is certainly oversimplified because real-world problems can be much harder than a Gaussian mixture model with known means and known variances. But one thing we realize is that if we want to find the magical encoder and decoder, we must have a way to find the two conditional distributions* p*(**z***|***x**) and *p*(**x***|***z**). However, they are both high-dimensional. In order for us to say something more meaningful, we need to impose additional structures so that we can generalize the concept to harder problems. To this end, we consider the following two proxy distributions:

•* q****ϕ***(**z***|***x**): The proxy for* p*(**z***|***x**), which is also the distribution associated with the* encoder*.* q****ϕ***(**z***|***x**) can be any directed graphical model and it can be parameterized using deep neural networks [24, Section 2.1]. For example, we can define

(***µ****,**** σ***^(2)) = EncoderNetwork***ϕ***(**x**)*,*

*q****ϕ***(**z***|***x**) =* N*(**z*** |**** µ****,* diag(***σ***^(2)))*. *(1.2)

This model is a widely used because of its tractability and computational efficiency. •* p****θ***(**x***|***z**): The proxy for* p*(**x***|***z**), which is also the distribution associated with the* decoder*. Like the encoder, the decoder can be parameterized by a deep neural network. For example, we can define

*f****θ***(**z**) = DecoderNetwork***θ***(**z**)*,*

*p****θ***(**x***|***z**) =* N*(**x*** |** f****θ***(**z**)*, σ*^(2 )dec^(**I**)^())^(*, *)(1.3)

where* σ*dec is a hyperparameter that can be pre-determined or it can be learned.

The relationship between the input** x** and the latent** z**, as well as the conditional distributions, are summarized in Figure 1.3. There are two nodes** x** and** z**. The “forward” relationship is specified by* p*(**z***|***x**) (and approximated by* q****ϕ***(**z***|***x**)), whereas the “reverse” relationship is specified by* p*(**x***|***z**) (and approximated by* p****θ***(**x***|***z**)).

Figure 1.3: In a variational autoencoder, the variables** x** and** z** are connected by the conditional distributions* p*(**x***|***z**) and* p*(**z***|***x**). To make things work, we introduce proxy distributions* p****θ***(**x***|***z**) and* q****ϕ***(**z***|***x**).

**Example 1.3.** Suppose that we have a random variable** x*** ∈*R^(*d*)^( )and a latent variable** z*** ∈*R^(*d*)^( )such that

**x*** ∼**p*(**x**) =* N*(**x*** |**** µ****, σ*^(2)**I**)*,*

**z*** ∼**p*(**z**) =* N*(**z*** |* 0*,*** I**)*.*

© 2024 Stanley Chan. All Rights Reserved. 4


---

## Page 5

We want to construct a VAE. By this, we mean that we want to build two mappings Encoder(*·*) and Decoder(*·*). The encoder will take a sample** x** and map it to the latent variable** z**, whereas the decoder will take the latent variable** z** and map it to the generated variable b**x**. If we* knew* what* p*(**x**) is, then there is a trivial solution where** z** = (**x*** −****µ***)*/σ* and b**x** =*** µ*** +* σ***z**. In this case, the true distributions can be determined and they can be expressed in terms of delta functions:

*p*(**x***|***z**) =* δ* (**x*** −*(*σ***z** +*** µ***))* ,*

*p*(**z***|***x**) =* δ* (**z*** −*(**x*** −****µ***)*/σ*)* .*

Suppose now that we do not know* p*(**x**) so we need to build an encoder and a decoder to estimate **z** and b**x**. Let’s first define the encoder. Our encoder in this example takes the input** x** and generates a pair of parameters b***µ***(**x**) and b*σ*(**x**)^(2), denoting the parameters of a Gaussian. Then, we define* q****ϕ***(**z***|***x**) as a Gaussian:

(b***µ***(**x**)*,* b*σ*(**x**)^(2)) = Encoder***ϕ***(**x**)*,*

*q****ϕ***(**z***|***x**) =* N*(**z*** |* b***µ***(**x**)*,* b*σ*(**x**)^(2)**I**)*.*

For the purpose of discussion, we assume that b***µ*** is an affine function of** x** such that b***µ***(**x**) =* a***x** +** b** for some parameters* a* and** b**. Similarly, we assume that b*σ*(**x**)^(2)^( )=* t*^(2)^( )for some scalar* t*. This will give us

*q****ϕ***(**z***|***x**) =* N*(**z*** |** a***x** +** b***, t*^(2)**I**)*.*

For the decoder, we deploy a similar structure by considering

(e***µ***(**z**)*,* e*σ*(**z**)^(2)) = Decoder***θ***(**z**)*,*

*p****θ***(**x***|***z**) =* N*(**x*** |* e***µ***(**z**)*,* e*σ*(**z**)^(2)**I**)*.*

Again, for the purpose of discussion, we assume that e***µ*** is affine so that e***µ***(**z**) =* c***z** +** v** for some parameters* c* and** v** and e*σ*(**z**)^(2)^( )=* s*^(2)^( )for some scalar* s*. Therefore,* p****θ***(**x***|***z**) takes the form of:

*p****θ***(**x***|***z**) =* N*(**z*** |** c***x** +** v***, s*^(2)**I**)*.*

We will discuss how to determine the parameters later.

**1.2 Evidence Lower Bound**

How do we use these two proxy distributions to achieve our goal of determining the encoder and the decoder? If we treat*** ϕ*** and*** θ*** as optimization variables, then we need an objective function (or the loss function) so that we can optimize*** ϕ*** and*** θ*** through training samples. The loss function we use here is called the Evidence Lower BOund (ELBO) [24]:

**Definition 1.3.** (**Evidence Lower Bound**) The Evidence Lower Bound is defined as

ELBO(**x**) def = E*q**ϕ*(**z***|***x**)

 log* *^(*p*)^(()^(**x**)^(*,*)^(** z**)^())

*q****ϕ***(**z***|***x**)

 *. *(1.4)

You are certainly puzzled how on the Earth people can come up with this loss function!? Let’s see what ELBO means and how it is derived.

© 2024 Stanley Chan. All Rights Reserved. 5


---

## Page 6

In a nutshell, ELBO is a** lower bound** for the prior distribution log* p*(**x**) because we can show that

log* p*(**x**) = some magical steps to be derived

= E*q**ϕ*(**z***|***x**)

 log* *^(*p*)^(()^(**x**)^(*,*)^(** z**)^())

*q****ϕ***(**z***|***x**)

 + DKL(*q****ϕ***(**z***|***x**)*∥**p*(**z***|***x**)) (1.5)

*≥*E*q**ϕ*(**z***|***x**)

 log* *^(*p*)^(()^(**x**)^(*,*)^(** z**)^())

*q****ϕ***(**z***|***x**)

def = ELBO(**x**)*,*

where the inequality follows from the fact that the KL divergence is always non-negative. Therefore, ELBO is a valid lower bound for log* p*(**x**). Since we never have access to log* p*(**x**), if we somehow have access to ELBO and if ELBO is a good lower bound, then we can effectively maximize ELBO to achieve the goal of maximizing log* p*(**x**) which is the gold standard. Now, the question is how good the lower bound is. As you can see from the equation and also Figure 1.4, the inequality will become an equality when our proxy *q****ϕ***(**z***|***x**) can match the true distribution* p*(**z***|***x**) exactly. So, part of the game is to ensure* q****ϕ***(**z***|***x**) is close to *p*(**z***|***x**).

Figure 1.4: Visualization of log* p*(**x**) and ELBO. The gap between the two is determined by the KL divergence DKL(*q****ϕ***(**z***|***x**)*∥**p*(**z***|***x**)).

The derivation of Eqn (1.5) is as follows.

**Theorem 1.1. Decomposition of Log-Likelihood**. The log likelihood log* p*(**x**) can be decomposed as

log* p*(**x**) = E*q**ϕ*(**z***|***x**)

 log* *^(*p*)^(()^(**x**)^(*,*)^(** z**)^())

*q****ϕ***(**z***|***x**)

| {z }

def = ELBO(**x**)

+ DKL(*q****ϕ***(**z***|***x**)*∥**p*(**z***|***x**))*. *(1.6)

**Proof**. The trick is to use our magical proxy* q****ϕ***(**z***|***x**) to poke around* p*(**x**) and derive the bound.

log* p*(**x**) = log* p*(**x**)* × *Z *q****ϕ***(**z***|***x**)*d***z **| {z } =1

(multiply 1)

= Z log* p*(**x**) | {z } some constant wrt** z**

*× **q****ϕ***(**z***|***x**) | {z } distribution in** z**

*d***z **(move log* p*(**x**) into integral)

= E*q****ϕ***(**z***|***x**)[log* p*(**x**)]*, *(1.7)

where the last equality is the fact that R *a**×**p**Z*(*z*)*dz* = E[*a*] =* a* for any random variable* Z* and a scalar *a*. See, we have already got E*q****ϕ***(**z***|***x**)[*·*]. Just a few more steps. Let’s use Bayes theorem which states

© 2024 Stanley Chan. All Rights Reserved. 6


---

## Page 7

that* p*(**x***,*** z**) =* p*(**z***|***x**)*p*(**x**):

E*q****ϕ***(**z***|***x**)[log* p*(**x**)] = E*q****ϕ***(**z***|***x**)

 log* *^(*p*)^(()^(**x**)^(*,*)^(** z**)^())

*p*(**z***|***x**)

(Bayes Theorem)

= E*q****ϕ***(**z***|***x**)

 log* *^(*p*)^(()^(**x**)^(*,*)^(** z**)^())

*p*(**z***|***x**)* *^(*×*)^(* q*)^(***ϕ***)^(()^(**z**)^(*|*)^(**x**)^())

*q****ϕ***(**z***|***x**)

(Multiply* *^(*q*)^(***ϕ***)^(()^(**z**)^(*|*)^(**x**)^())

*q****ϕ***(**z***|***x**)^())

= E*q****ϕ***(**z***|***x**)

 log* *^(*p*)^(()^(**x**)^(*,*)^(** z**)^())

*q****ϕ***(**z***|***x**)

| {z } ELBO

+ E*q****ϕ***(**z***|***x**)

 log* *^(*q*)^(***ϕ***)^(()^(**z**)^(*|*)^(**x**)^())

*p*(**z***|***x**)

| {z } DKL(*q****ϕ***(**z***|***x**)*∥**p*(**z***|***x**))

*, *(1.8)

where we recognize that the first term is exactly ELBO, whereas the second term is exactly the KL divergence. Comparing Eqn (1.8) with Eqn (1.5), we complete the proof.

**Example 1.4.** Using the previous example, we can minimize the gap between log* p*(**x**) and ELBO(**x**) if we* knew** p*(**z***|***x**). To see that, we note that log* p*(**x**) is

log* p*(**x**) = ELBO(**x**) + DKL(*q****ϕ***(**z***|***x**)*∥**p*(**z***|***x**))* ≥*ELBO(**x**)*.*

The equality holds if and only if the KL-divergence term is zero. For the KL divergence to be zero, it is necessary that* q****ϕ***(**z***|***x**) =* p*(**z***|***x**). However, since* p*(**z***|***x**) is a delta function, the only possibility is to have

*q****ϕ***(**z***|***x**) =* N*(**z*** |*** **^(**x**)^(*−*)^(***µ***)

*σ** *^(*,*)^( 0))

=* δ*(**z*** −*^(**x**)^(*−*)^(***µ***)

*σ* ^())^(*, *)(1.9)

i.e., we set the standard deviation to be* t* = 0. To determine* p****θ***(**x***|***z**), we need some additional steps to simplify ELBO.

We now have ELBO. But this ELBO is still not too useful because it involves* p*(**x***,*** z**), something we have no access to. So, we need to do a little more work.

**Theorem 1.2. Interpretation of ELBO**. ELBO can be decomposed as

ELBO(**x**) = E*q**ϕ*(**z***|***x**)[log

a Gaussian z }| { *p****θ***(**x***|***z**) ] | {z } how good your decoder is

*− *DKL ^(a Gaussian )z }| { *q**ϕ*(**z***|***x**)* ∥*

a Gaussian z}|{ *p*(**z**) 

| {z } how good your encoder is

*. *(1.10)

**Proof**. Let’s take a closer look at ELBO

ELBO(**x**) def = E*q****ϕ***(**z***|***x**)

 log* *^(*p*)^(()^(**x**)^(*,*)^(** z**)^())

*q****ϕ***(**z***|***x**)

(definition)

= E*q****ϕ***(**z***|***x**)

 log* *^(*p*)^(()^(**x**)^(*|*)^(**z**)^())^(*p*)^(()^(**z**)^())

*q****ϕ***(**z***|***x**)

(*p*(**x***,*** z**) =* p*(**x***|***z**)*p*(**z**))

= E*q****ϕ***(**z***|***x**) [log* p*(**x***|***z**)] + E*q****ϕ***(**z***|***x**)

 log *p*(**z**) *q****ϕ***(**z***|***x**)

(split expectation)

= E*q****ϕ***(**z***|***x**) [log* p****θ***(**x***|***z**)]* −*DKL(*q****ϕ***(**z***|***x**)*∥**p*(**z**))*, *(definition of KL)

where we replaced the inaccessible* p*(**x***|***z**) by its proxy* p****θ***(**x***|***z**).

This is a* beautiful* result. We just showed something very easy to understand. Let’s look at the two terms in Eqn (1.10):

© 2024 Stanley Chan. All Rights Reserved. 7


---

## Page 8

•** Reconstruction**. The first term is about the* decoder*. We want the decoder to produce a good image **x** if we feed a latent** z** into the decoder (of course!!). So, we want to* maximize* log* p****θ***(**x***|***z**). It is similar to maximum likelihood where we want to find the model parameter to maximize the likelihood of observing the image. The expectation here is taken with respect to the samples** z** (conditioned on **x**). This shouldn’t be a surprise because the samples** z** are used to assess the quality of the decoder. It cannot be an arbitrary noise vector but a meaningful latent vector. So,** z** needs to be sampled from *q**ϕ*(**z***|***x**). •** Prior Matching**. The second term is the KL divergence for the* encoder*. We want the encoder to turn** x** into a latent vector** z** such that the latent vector will follow our choice of distribution, e.g., **z*** ∼N*(0*,*** I**). To be slightly more general, we write* p*(**z**) as the target distribution. Because the KL divergence is a distance (which increases when the two distributions become more dissimilar), we need to put a negative sign in front so that it increases when the two distributions become more similar.

**Example 1.5.** Following up on the previous example, we continue to assume that we* knew** p*(**z***|***x**). Then the reconstruction term in ELBO will give us

E*q****ϕ***(**z***|***x**)[log* p****θ***(**x***|***z**)] = E*q****ϕ***(**z***|***x**)[log* N*(**x*** |** c***z** +** v***, s*^(2)**I**)]

= E*q****ϕ***(**z***|***x**)

 *−*^(1)

2 ^(log 2)^(*π*)^(* −*)^(log)^(* s*)^(* −∥*)^(**x**)^(* −*)^(()^(*c*)^(**z**)^( +)^(** v**)^())^(*∥*)^(2)

2*s*^(2)

=* −*^(1)

2 ^(log 2)^(*π*)^(* −*)^(log)^(* s*)^(* −*)^(*c*)^(2)

2*s*^(2)^( E)^(*q*)^(***ϕ***)^(()^(**z**)^(*|*)^(**x**)^() ) *∥***z*** −*^(**x**)^(*−*)^(**v**)

*c** *^(*∥*)^(2)^()

=* −*^(1)

2 ^(log 2)^(*π*)^(* −*)^(log)^(* s*)^(* −*)^(*c*)^(2)

2*s*^(2)^( E )*δ * **z***−*^(**x**)^(*−*)^(***µ***)

*σ*

 ^( )*∥***z*** −*^(**x**)^(*−*)^(**v**)

*c** *^(*∥*)^(2)^()

=* −*^(1)

2 ^(log 2)^(*π*)^(* −*)^(log)^(* s*)^(* −*)^(*c*)^(2)

2*s*^(2 ) *∥*^(**x**)^(*−*)^(***µ***)

*σ **−*^(**x**)^(*−*)^(**v**)

*c** *^(*∥*)^(2)^()

*≤−*^(1)

2 ^(log 2)^(*π*)^(* −*)^(log)^(* s,*)

where the upper bound is tight if and only if the norm-square term is zero, which holds when** v** =*** µ ***and* c* =* σ*. For the remaining terms, it is clear that* −*log* s* is a monotonically decreasing function in* s *with* −*log* s** →∞*as* s** →*0. Therefore, when** v** =*** µ*** and* c* =* σ*, it follows that E*q****ϕ***(**z***|***x**)[log* p****θ***(**x***|***z**)] is maximized when* s* = 0. This implies that

*p****θ***(**x***|***z**) =* N*(**x*** |** σ***z** +*** µ****,* 0)

=* δ*(**x*** −*(*σ***z** +*** µ***))*. *(1.11)

**Limitation of ELBO**. ELBO is practically useful, but it is* not* the same as the true likelihood log* p*(**x**). As we mentioned, ELBO is exactly equal to log* p*(**x**) if and only if DKL(*q****ϕ***(**z***|***x**)*∥**p*(**z***|***x**)) = 0 which happens when* q****ϕ***(**z***|***x**) =* p*(**z***|***x**). In the following example, we will show a case where the* q****ϕ***(**z***|***x**) obtained from maximizing ELBO is not the same as* p*(**z***|***x**).

**Example 1.6.** (**Limitation of ELBO**). In the previous example, if we have no idea about* p*(**z***|***x**), we need to train the VAE by maximizing ELBO. However, since ELBO is only a lower bound of the true distribution log* p*(**x**), maximizing ELBO will not return us the delta functions as we hope. Instead, we will obtain something that is quite meaningful but not exactly the delta functions. For simplicity, let’s consider the distributions that will return us unbiased estimates of the mean but with unknown variances:

*q****ϕ***(**z***|***x**) =* N*(**z*** |*** **^(**x**)^(*−*)^(***µ***)

*σ** *^(*, t*)^(2)^(**I**)^())^(*,*)

*p****θ***(**x***|***z**) =* N*(**x*** |** σ***z** +*** µ****, s*^(2)**I**)*.*

This is partially “cheating” because in theory we should not assume anything about the estimates of

© 2024 Stanley Chan. All Rights Reserved. 8


---

## Page 9

the means. But from an intuitive angle, since* q****ϕ***(**z***|***x**) and* p****θ***(**x***|***z**) are proxies to* p*(**z***|***x**) and* p*(**x***|***z**), they must resemble some properties of the delta functions. The closest choice is to define* q****ϕ***(**z***|***x**) and *p****θ***(**x***|***z**) as Gaussians with means consistent with those of the two delta functions. The variances are unknown, and they are the subject of interest in this example. Our focus here is to maximize ELBO which consists of the prior matching term and the reconstruc- tion term. For the prior matching error, we want to minimize the KL-divergence:

DKL(*q****ϕ***(**z***|***x**)*∥**p*(**z**)) = DKL � *N*(**z*** |*** **^(**x**)^(*−*)^(***µ***)

*σ** *^(*, t*)^(2)^(**I**)^())^(* ∥N*)^(()^(**z**)^(* |*)^( 0)^(*,*)^(** I**)^() ) *.*

The KL-divergence of two multivariate Gaussians* N*(**z***|****µ***0*,*** Σ**0) and* N*(**z***|****µ***1*,*** Σ**1) has a closed form expression which can be found in Wikipedia:

DKL(*N*(***µ***0*,*** Σ**0)*∥N*(***µ***1*,*** Σ**1))

= ^(1)

2

 Tr(**Σ**^(*−*)^(1 )1** **^(**Σ**)^(0)^())^(* −*)^(*d*)^( + ()^(***µ***)1* *^(*−*)^(***µ***)0^())^(*T*)^(** Σ**)^(*−*)^(1 )1 ^(()^(***µ***)1* *^(*−*)^(***µ***)0^() + log det)^(**Σ**)^(1)

det**Σ**0

 *.*

Using this result (and with some algebra), we can show that

DKL � *N*(**z*** |*** **^(**x**)^(*−*)^(***µ***)

*σ** *^(*, t*)^(2)^(**I**)^())^(* ∥N*)^(()^(**z**)^(* |*)^( 0)^(*,*)^(** I**)^() ) = ^(1)

2  *t*^(2)*d** −**d* +* ∥*^(**x**)^(*−*)^(***µ***)

*σ** *^(*∥*)^(2)^(* −*)^(2)^(*d*)^( log)^(* t *) *,*

where* d* is the dimension of** x** and** z**. To minimize the KL-divergence, we take derivative with respect to* t* and show that

*∂ ∂t*

1

2  *t*^(2)*d** −**d* +* ∥*^(**x**)^(*−*)^(***µ***)

*σ** *^(*∥*)^(2)^(* −*)^(2)^(*d*)^( log)^(* t *)^( )=* t** ·** d** −*^(*d*)

*t *^(*.*)

Setting this to zero will give us* t* = 1. Therefore, we can show that

*q****ϕ***(**z***|***x**) =* N*(**z*** |*** **^(**x**)^(*−*)^(***µ***)

*σ** *^(*,*)^(** I**)^())^(*.*)

For the reconstruction term, we can show that

E*q****ϕ***(**z***|***x**)[log* p****θ***(**x***|***z**)] = E*q****ϕ***(**z***|***x**)

"

log 1

( *√*

2*πs*^(2))^(*d*)^( exp ) *−*^(*∥*)^(**x**)^(* −*)^(()^(*σ*)^(**z**)^( +)^(*** µ***)^())^(*∥*)^(2)

2*s*^(2)

^(#)

= E*q****ϕ***(**z***|***x**)

 *−*^(*d*)

2 ^(log 2)^(*π*)^(* −*)^(*d*)^( log)^(* s*)^(* −∥*)^(**x**)^(* −*)^(()^(*σ*)^(**z**)^( +)^(*** µ***)^())^(*∥*)^(2)

2*s*^(2)

=* −*^(*d*)

2 ^(log 2)^(*π*)^(* −*)^(*d*)^( log)^(* s*)^(* −*)^(*σ*)^(2)

2*s*^(2)^( E)^(*q*)^(***ϕ***)^(()^(**z**)^(*|*)^(**x**)^() )h**z*** −***x***−****µ***

*σ *2^(i)

=* −*^(*d*)

2 ^(log 2)^(*π*)^(* −*)^(*d*)^( log)^(* s*)^(* −*)^(*σ*)^(2)

2*s*^(2)^( Trace )n E*q****ϕ***(**z***|***x**) h� **z*** −*^(**x**)^(*−*)^(***µ***)

*σ * � **z*** −*^(**x**)^(*−*)^(***µ***)

*σ **T* ^(io)

=* −*^(*d*)

2 ^(log 2)^(*π*)^(* −*)^(*d*)^( log)^(* s*)^(* −*)^(*σ*)^(2)

2*s*^(2)^(* ·*)^(* d,*)

because the covariance of** z*** ∼**q****ϕ***(**z***|***x**) is** I** and so the trace will give us* d*. Taking derivatives with respect to* s* will give us

*d ds*

 *−*^(*d*)

2 ^(log 2)^(*π*)^(* −*)^(*d*)^( log)^(* s*)^(* −*)^(*dσ*)^(2)

2*s*^(2)

 =* −*^(*d*)

*s* ^(+)^(* dσ*)^(2)

*s*^(3)^( = 0)^(*.*)

Equating this to zero will give us* s* =* σ*. Therefore,

*p****θ***(**x***|***z**) =* N*(**x*** |** σ***z** +*** µ****, σ*^(2)**I**)*.*

© 2024 Stanley Chan. All Rights Reserved. 9


---

## Page 10

As we can see in this example and the previous example, while the ideal distributions are delta functions, the proxy distributions we obtain have a finite variance. This finite variance adds additional randomness to the samples generated by the VAE. There is nothing wrong with this VAE — we do it correctly by maximizing ELBO. It is just that maximizing the ELBO is not the same as maximizing log* p*(**x**).

**1.3 Optimization in VAE**

In the previous two subsections we introduced the building blocks of VAE and ELBO. The goal of this subsection is to discuss how to train a VAE and how to do inference. VAE is a model that aims to approximate the true distribution* p*(**x**) so that we can draw samples. A VAE is parameterized by (***ϕ****,**** θ***). Therefore, training a VAE is equivalent to solving an optimization problem that encapsulates the essence of* p*(**x**) while being tractable. However, since* p*(**x**) is not accessible, the natural alternative is to optimize the ELBO which is the lower bound of log* p*(**x**). That means, the learning goal of VAE is to solve the following problem.

**Definition 1.4.** The optimization objective of VAE is to maximize the ELBO:

(***ϕ****,**** θ***) = argmax ***ϕ****,****θ***

X

**x***∈X *ELBO(**x**)*, *(1.12)

where* X* =* {***x**^(()^(*ℓ*)^())^(* *)*|** ℓ*= 1*, . . . , L**}* is the training dataset.

**Intractability of ELBO’s Gradient**. The challenge associated with the above optimization is that the gradient of ELBO with respect to (***ϕ****,**** θ***) is intractable. Since the majority of today’s neural network opti- mizers use first-order methods and backpropagate the gradient to update the network weights, an intractable gradient will pose difficulties in training the VAE. Let’s elaborate more about the intractability of the gradient. We first substitute Definition 1.3 into the above objective function. The gradient of ELBO is: ^(2)

*∇****θ****,****ϕ*** ELBO(**x**) =* ∇****θ****,****ϕ***

 E*q**ϕ*(**z***|***x**)

 log* *^(*p*)^(***θ***)^(()^(**x**)^(*,*)^(** z**)^())

*q****ϕ***(**z***|***x**)

=* ∇****θ****,****ϕ ***n E*q**ϕ*(**z***|***x**) h log* p****θ***(**x***,*** z**)* −*log* q****ϕ***(**z***|***x**) io *. *(1.13)

The gradient contains two parameters. Let’s first look at*** θ***. We can show that

*∇****θ*** ELBO(**x**) =* ∇****θ ***n E*q**ϕ*(**z***|***x**) h log* p****θ***(**x***,*** z**)* −*log* q****ϕ***(**z***|***x**) io

=* ∇****θ***

Z h log* p****θ***(**x***,*** z**)* −*log* q****ϕ***(**z***|***x**) i *·** q**ϕ*(**z***|***x**)*d***z **

= Z *∇****θ ***n log* p****θ***(**x***,*** z**)* −*log* q****ϕ***(**z***|***x**) o *·** q**ϕ*(**z***|***x**)* d***z**

= E*q**ϕ*(**z***|***x**) h *∇****θ ***n log* p****θ***(**x***,*** z**)* −*log* q****ϕ***(**z***|***x**) oi

= E*q**ϕ*(**z***|***x**) h *∇****θ ***n log* p****θ***(**x***,*** z**) oi

*≈*^(1)

*L*

*L *X

*ℓ*=1 *∇****θ ***n log* p****θ***(**x***,*** z**^(()^(*ℓ*)^())) o *, *(where **z**^(()^(*ℓ*)^())^(* *)*∼**q****ϕ***(**z***|***x**)) (1.14)

where the last equality is the Monte Carlo approximation of the expectation. In the above equation, if* p****θ***(**x***,*** z**) is realized by a computable model such as a neural network, then its gradient* ∇****θ****{*log* p****θ***(**x***,*** z**)*}* can be computed via automatic differentiation. Thus, the maximization can be achieved by backpropagating the gradient.

2The original definition of ELBO uses the true joint distribution* p*(**x***,*** z**). In practice, since* p*(**x***,*** z**) is not accessible, we replace it by its proxy* p****θ***(**x***,*** z**) which is a computable distribution.

© 2024 Stanley Chan. All Rights Reserved. 10


---

## Page 11

The gradient with respect to*** ϕ*** is more difficult. We can show that

*∇****ϕ*** ELBO(**x**) =* ∇****ϕ ***n E*q**ϕ*(**z***|***x**) h log* p****θ***(**x***,*** z**)* −*log* q****ϕ***(**z***|***x**) io

=* ∇****ϕ***

Z h log* p****θ***(**x***,*** z**)* −*log* q****ϕ***(**z***|***x**) i *·** q**ϕ*(**z***|***x**)*d***z **

= Z *∇****ϕ ***n [log* p****θ***(**x***,*** z**)* −*log* q****ϕ***(**z***|***x**)]* ·** q**ϕ*(**z***|***x**) o *d***z**

*̸*= Z *∇****ϕ ***n log* p****θ***(**x***,*** z**)* −*log* q****ϕ***(**z***|***x**) o *·** q**ϕ*(**z***|***x**)* d***z**

= E*q**ϕ*(**z***|***x**) h *∇****ϕ ***n log* p****θ***(**x***,*** z**)* −*log* q****ϕ***(**z***|***x**) oi

= E*q**ϕ*(**z***|***x**) h *∇****ϕ ***n *−*log* q****ϕ***(**z***|***x**) oi

*≈*^(1)

*L*

*L *X

*ℓ*=1 *∇****ϕ ***n *−*log* q****ϕ***(**z**^(()^(*ℓ*)^())*|***x**) o *, *(where **z**^(()^(*ℓ*)^())^(* *)*∼**q****ϕ***(**z***|***x**))*. *(1.15)

As we can see, even though we* wish* to maintain a similar structure as we did for*** θ***, the expectation and the gradient operators in the above derivations cannot be switched. This forbids us from doing any backpropagation of the gradient to maximize ELBO.

**Reparameterization Trick**. The intractability of ELBO’s gradient is inherited from the fact that we need to draw samples** z** from a distribution* q****ϕ***(**z***|***x**) which itself is a function of*** ϕ***. As noted by Kingma and Welling [23], for continuous latent variables, it is possible to compute an unbiased estimate of* ∇****θ****,****ϕ*** ELBO(**x**) so that we can approximately calculate the gradient and hence maximize ELBO. The idea is to employ an technique known as the* reparameterization trick* [23]. Recall that the latent variable** z** is a sample drawn from the distribution* q****ϕ***(**z***|***x**). The idea of repa- rameterization trick is to express** z** as some differentiable and invertible transformation of another random variable*** ϵ*** whose distribution is independent of** x** and*** ϕ***. That is, we define a differentiable and invertible function** g** such that **z** =** g**(***ϵ****,**** ϕ****,*** x**)*, *(1.16)

for some random variable*** ϵ**** ∼**p*(***ϵ***). To make our discussions easier, we pose an additional requirement that

*q****ϕ***(**z***|***x**)* · *det *∂***z**

*∂****ϵ***

 =* p*(***ϵ***)*, *(1.17)

where* *^(*∂*)^(**z**)

*∂****ϵ*** ^(is the Jacobian, and det()^(*·*)^() is the matrix determinant. This requirement is related to change of )variables in multivariate calculus. The following example will make it clear.

**Example 1.7.** Suppose** z*** ∼**q****ϕ***(**z***|***x**) def =* N*(**z*** |**** µ****,* diag(***σ***^(2))). We can define

**z** =** g**(***ϵ****,**** ϕ****,*** x**) def =*** ϵ**** ⊙****σ*** +*** µ****, *(1.18)

where*** ϵ**** ∼N*(0*,*** I**) and “*⊙*” means elementwise multiplication. The parameter*** ϕ*** is*** ϕ*** = (***µ****,**** σ***^(2)). For this choice of the distribution, we can show that by letting*** ϵ*** =** **^(**z**)^(*−*)^(***µ***)

***σ*** ^(:)

*q****ϕ***(**z***|***x**)* · *det *∂***z**

*∂****ϵ***

 =

*d *Y

*i*=1

1 p

2*πσ*^(2 )*i *exp  *−*^(()^(*z*)^(*i*)^(* −*)^(*µ*)^(*i*)^())^(2)

2*σ*^(2 )*i*

 *·*

*d *Y

*i*=1 *σ**i*

= 1 ( *√*

2*π*)^(*d*)^( exp ) *−*^(*∥*)^(***ϵ***)^(*∥*)^(2)

2

 =* N*(0*,*** I**) =* p*(***ϵ***)*.*

With this re-parameterization of** z** by expressing it in terms of*** ϵ***, we can look at* ∇****ϕ***E*q****ϕ***(**z***|***x**)[*f*(**z**)] for some general function* f*(**z**). (Later we will consider* f*(**z**) =* −*log* q****ϕ***(**z***|***x**).) For notational simplicity, we

© 2024 Stanley Chan. All Rights Reserved. 11


---

## Page 12

write** g**(***ϵ***) instead of** g**(***ϵ****,**** ϕ****,*** x**) although we understand that** g** has three inputs. By change of variables, we can show that

E*q****ϕ***(**z***|***x**)[*f*(**z**)] = Z *f*(**z**)* ·** q****ϕ***(**z***|***x**)* d***z**

= Z *f*(**g**(***ϵ***))* ·** q****ϕ***(**g**(***ϵ***)*|***x**)* d***g**(***ϵ***)*, *(**z** =** g**(***ϵ***))

= Z *f * **g**(***ϵ***)  *·** q****ϕ***(**g**(***ϵ***)*|***x**)* · *det *∂***g**(***ϵ***)

*∂****ϵ***

* d****ϵ ***(Jacobian due to change of variable)

= Z *f*(**z**)* ·** p*(***ϵ***)* d****ϵ ***(use Eqn (1.17))

= E*p*(***ϵ***) [*f*(**z**)]* . *(1.19)

So, if we want to take the gradient with respect to*** ϕ***, we can show that

*∇****ϕ***E*q****ϕ***(**z***|***x**)[*f*(**z**)] =* ∇****ϕ***E*p*(***ϵ***) [*f*(**z**)] =* ∇****ϕ***

Z *f*(**z**)* ·** p*(***ϵ***)* d****ϵ ***

= Z *∇****ϕ**** {**f*(**z**)* ·** p*(***ϵ***)*}** d****ϵ***

= Z *{∇****ϕ****f*(**z**)*} ·** p*(***ϵ***)* d****ϵ***

= E*p*(***ϵ***) [*∇****ϕ****f*(**z**)]* , *(1.20)

which can be approximated by Monte Carlo. Substituting* f*(**z**) =* −*log* q****ϕ***(**z***|***x**), we can show that

*∇****ϕ***E*q****ϕ***(**z***|***x**)[*−*log* q****ϕ***(**z***|***x**)] = E*p*(***ϵ***) [*−∇****ϕ*** log* q****ϕ***(**z***|***x**)]

*≈−*^(1)

*L*

*L *X

*ℓ*=1 *∇****ϕ*** log* q****ϕ***(**z**^(()^(*ℓ*)^())*|***x**)*, *(where **z**^(()^(*ℓ*)^())^( )=** g**(***ϵ***^(()^(*ℓ*)^())*,**** ϕ****,*** x**))

=* −*^(1)

*L*

*L *X

*ℓ*=1 *∇****ϕ***

 log* p*(***ϵ***^(()^(*ℓ*)^()))* −*log det* ∂***z**(*ℓ*)

*∂****ϵ***^(()^(*ℓ*)^())



= ^(1)

*L*

*L *X

*ℓ*=1 *∇****ϕ***

 log det* ∂***z**(*ℓ*)

*∂****ϵ***^(()^(*ℓ*)^())



 *.*

So, as long as the determinant is differentiable with respect to*** ϕ***, the Monte Carlo approximation can be numerically computed.

**Example 1.8.** Suppose that the parameters and the distribution* q****ϕ*** are defined as follows:

(***µ****,**** σ***^(2)) = EncoderNetwork***ϕ***(**x**)

*q****ϕ***(**z***|***x**) =* N*(**z*** |**** µ****,* diag(***σ***^(2)))*.*

We can define** z** =*** µ*** +*** σ**** ⊙****ϵ***, with*** ϵ**** ∼N*(0*,*** I**). Then, we can show that

log det* ∂***z**

*∂****ϵ***

 = log det *∂*(***µ*** +*** σ**** ⊙****ϵ***)

*∂****ϵ***



= log det  diag* {****σ****} *

= log

*d *Y

*i*=1 *σ**i* =

*d *X

*i*=1 log* σ**i**.*

© 2024 Stanley Chan. All Rights Reserved. 12


---

## Page 13

Therefore, we can show that

*∇****ϕ***E*q****ϕ***(**z***|***x**)[*−*log* q****ϕ***(**z***|***x**)]* ≈*^(1)

*L*

*L *X

*ℓ*=1 *∇****ϕ***

 log det* ∂***z**(*ℓ*)

*∂****ϵ***^(()^(*ℓ*)^())



= ^(1)

*L*

*L *X

*ℓ*=1 *∇****ϕ***

"* d *X

*i*=1 log* σ**i*

#

=* ∇****ϕ***

"* d *X

*i*=1 log* σ**i*

#

= ^(1)

***σ**** *^(*⊙∇*)^(***ϕ ***)n ***σ******ϕ***(**x**) o *,*

where we emphasize that*** σ******ϕ***(**x**) is the output of the encoder which is a neural network.

As we can see in the above example, for some specific choices of the distributions (e.g., Gaussian), the gradient of ELBO can be significantly easier to derive.

**VAE Encoder**. After discussing the reparameterizing trick, we can now discuss the specific structure of the encoder in VAE. To make our discussions focused, we assume a relatively common choice of the encoder:

(***µ****, σ*^(2)) = EncoderNetwork***ϕ***(**x**)

*q****ϕ***(**z***|***x**) =* N*(**z*** |**** µ****, σ*^(2)**I**)*.*

The parameters*** µ*** and* σ* are technically* neural networks* because they are the outputs of EncoderNetwork***ϕ***(*·*). Therefore, it will be helpful if we denote them as

***µ*** = ***µ******ϕ ***|{z} neural network

(**x**)*,*

*σ*^(2)^( )= *σ*^(2 )***ϕ ***|{z} neural network

(**x**)*,*

Our notation is slightly more complicated because we want to emphasize that*** µ*** is a function of** x**; You give us an image** x**, our job is to return you the parameters of the Gaussian (i.e., mean and variance). If you give us a different** x**, then the parameters of the Gaussian should also be different. The parameter*** ϕ*** specifies that*** µ*** is controlled (or parameterized) by*** ϕ***. Suppose that we are given the* ℓ*-th training sample** x**^(()^(*ℓ*)^()). From this** x**^(()^(*ℓ*)^())^( )we want to generate a latent variable** z**^(()^(*ℓ*)^())^( )which is a sample from* q****ϕ***(**z***|***x**). Because of the Gaussian structure, it is equivalent to say that

**z**^(()^(*ℓ*)^())^(* *)*∼N * **z ***** µ******ϕ***(**x**(*ℓ*))*, σ*^(2 )***ϕ***^(()^(**x**)^(()^(*ℓ*)^())^())^(**I **) *. *(1.21)

The interesting thing about this equation is that we use a neural network EncoderNetwork***ϕ***(*·*) to estimate the mean and variance of the Gaussian. Then, from this Gaussian we draw a sample** z**^(()^(*ℓ*)^()), as illustrated in Figure 1.5.

Figure 1.5: Implementation of a VAE encoder. We use a neural network to take the image** x **and estimate the mean*** µ******ϕ*** and variance* σ*^(2 )***ϕ*** ^(of the Gaussian distribution.)

A more convenient way of expressing Eqn (1.21) is to realize that the sampling operation** z*** ∼N*(***µ****, σ*^(2)**I**) can be done using the reparameterization trick.

© 2024 Stanley Chan. All Rights Reserved. 13


---

## Page 14

**Reparameterization Trick for High-dimensional Gaussian**:

**z*** ∼N*(***µ****, σ*^(2)**I**) *⇐⇒ ***z** =*** µ*** +* σ****ϵ****, ****ϵ**** ∼N*(0*,*** I**)*. *(1.22)

Using the reparameterization trick, Eqn (1.21) can be written as

**z**^(()^(*ℓ*)^())^( )=*** µ******ϕ***(**x**^(()^(*ℓ*)^())) +* σ****ϕ***(**x**^(()^(*ℓ*)^()))***ϵ****, ****ϵ**** ∼N*(0*,*** I**)*.*

**Proof**. We will prove a general case for an arbitrary covariance matrix** Σ** instead of a diagonal matrix *σ*^(2)**I**. For any high-dimensional Gaussian** z*** ∼N*(**z***|****µ****,*** Σ**), the sampling process can be done via the transformation of white noise **z** =*** µ*** +** Σ **1 2*** ϵ****, *(1.23)

where*** ϵ**** ∼N*(0*,*** I**). The half matrix** Σ **1 2 can be obtained through eigen-decomposition or Cholesky factorization. If** Σ** has an eigen-decomposition** Σ** =** USU**^(*T*)^( ), then** Σ **1 2 =** US **1 2** U**^(*T*)^( ). The square root of the eigenvalue matrix** S** is well-defined because** Σ** is a positive semi-definite matrix. We can calculate the expectation and covariance of** x**:

E[**z**] = E[***µ*** +** Σ **1 2*** ϵ***] =*** µ*** +** Σ **1 2 E[***ϵ***] |{z} =0

=*** µ****,*

Cov(**z**) = E[(**z*** −****µ***)(**z*** −****µ***)^(*T*)^( )] = E h **Σ **1 2*** ϵϵ***^(*T*)^( )(**Σ **1 2 )^(*T*)^( i )=** Σ **1 2 E[***ϵϵ***^(*T*)^( )] | {z } =**I**

(**Σ **1 2 )^(*T*)^( )=** Σ***.*

Therefore, for diagonal matrices** Σ** =* σ*^(2)**I**, the above is reduced to

**z** =*** µ*** +* σ****ϵ****, *where*** ϵ**** ∼N*(0*,*** I**)*. *(1.24)

Given the VAE encoder structure and* q****ϕ***(**z***|***x**), we can go back to ELBO. Recall that ELBO consists of the prior matching term and the reconstruction term. The prior matching term is measured in terms of the KL divergence DKL (*q****ϕ***(**z***|***x**)*∥**p*(**z**)). Let’s evaluate this KL divergence. To evaluate the KL divergence, we (re)use a result which we summarize below:

**Theorem 1.3. KL-Divergence of Two Gaussian**. The KL divergence for two* d*-dimensional Gaussian distributions* N*(***µ***0*,*** Σ**0) and* N*(***µ***1*,*** Σ**1) is

DKL  *N*(***µ***0*,*** Σ**0)* ∥N*(***µ***1*,*** Σ**1) 

= ^(1)

2

 Tr(**Σ**^(*−*)^(1 )1** **^(**Σ**)^(0)^())^(* −*)^(*d*)^( + ()^(***µ***)1* *^(*−*)^(***µ***)0^())^(*T*)^(** Σ**)^(*−*)^(1 )1 ^(()^(***µ***)1* *^(*−*)^(***µ***)0^() + log det)^(**Σ**)^(1)

det**Σ**0

 *. *(1.25)

Substituting our distributions by considering

***µ***0 =*** µ******ϕ***(**x**)*, ***Σ**0 =* σ*^(2 )***ϕ***^(()^(**x**)^())^(**I**)

***µ***1 = 0*, ***Σ**1 =** I***,*

we can show that the KL divergence has an analytic expression

DKL  *q****ϕ***(**z***|***x**)* ∥**p*(**z**)  = ^(1)

2

 *σ*^(2 )***ϕ***^(()^(**x**)^())^(*d*)^(* −*)^(*d*)^( +)^(* ∥*)^(***µ***)***ϕ***^(()^(**x**)^())^(*∥*)^(2)^(* −*)^(2)^(*d*)^( log)^(* σ*)^(***ϕ***)^(()^(**x**)^() ) *, *(1.26)

where* d* is the dimension of the vector** z**. The gradient of the KL-divergence with respect to*** ϕ*** does not have a closed form but they can be calculated numerically:

*∇****ϕ***DKL  *q****ϕ***(**z***|***x**)* ∥**p*(**z**)  = ^(1)

2^(*∇*)^(***ϕ ***) *σ*^(2 )***ϕ***^(()^(**x**)^())^(*d*)^(* −*)^(*d*)^( +)^(* ∥*)^(***µ***)***ϕ***^(()^(**x**)^())^(*∥*)^(2)^(* −*)^(2)^(*d*)^( log)^(* σ*)^(***ϕ***)^(()^(**x**)^() ) *. *(1.27)

© 2024 Stanley Chan. All Rights Reserved. 14


---

## Page 15

The gradient with respect to*** θ*** is zero because there is nothing dependent on*** θ***.

**VAE Decoder**. The decoder is implemented through a neural network. For notation simplicity, let’s define it as DecoderNetwork***θ***(*·*) where*** θ*** denotes the network parameters. The job of the decoder network is to take a latent variable** z** and generate an image* f****θ***(**z**):

*f****θ***(**z**) = DecoderNetwork***θ***(**z**)*. *(1.28)

The distribution* p****θ***(**x***|***z**) can be defined as

*p****θ***(**x***|***z**) =* N*(**x*** |** f****θ***(**z**)*, σ*^(2 )dec^(**I**)^())^(*, *)for some hyperparameter* σ*dec. (1.29)

The interpretation of* p****θ***(**x***|***z**) is that we estimate* f****θ***(**z**) through a network and put it as the mean of the Gaussian. If we draw a sample** x** from* p****θ***(**x***|***z**), then by the reparameterization trick we can write the generated image b**x** as b**x** =* f****θ***(**z**) +* σ*dec***ϵ****, ****ϵ**** ∼N*(0*,*** I**)*.*

Moreover, if we take the log of the likelihood, we can show that

log* p****θ***(**x***|***z**) = log* N*(**x*** |** f****θ***(**z**)*, σ*^(2 )dec^(**I**)^())

= log 1 p

(2*πσ*^(2 )dec^())^(*d*)^( exp ) *−*^(*∥*)^(**x**)^(* −*)^(*f*)^(***θ***)^(()^(**z**)^())^(*∥*)^(2)

2*σ*^(2 )dec

=* −*^(*∥*)^(**x**)^(* −*)^(*f*)^(***θ***)^(()^(**z**)^())^(*∥*)^(2)

2*σ*^(2 )dec *− *log q

(2*πσ*^(2 )dec^())^(*d *)| {z } independent of*** θ*** so we can drop it

*. *(1.30)

Going back to ELBO, we want to compute E*q****ϕ***(**z***|***x**)[log* p****θ***(**x***|***z**)]. If we straightly calculate the expectation, we will need to compute an integration

E*q****ϕ***(**z***|***x**)[log* p****θ***(**x***|***z**)] = Z log  *N*(**x*** |** f****θ***(**z**)*, σ*^(2 )dec^(**I**)^() ) *· N*(**z*** |**** µ******ϕ***(**x**)*, σ*^(2 )***ϕ***^(()^(**x**)^()))^(*d*)^(**z**)

=* − *Z* ∥***x*** −**f****θ***(**z**)*∥*2

2*σ*^(2 )dec *· N * **z ***** µ******ϕ***(**x**)*, σ*2 ***ϕ***^(()^(**x**)^() ) *d***z** +* C,*

where the constant* C* coming out of the log of the Gaussian can be dropped. By using the reparameterization trick, we write** z** =*** µ******ϕ***(**x**) +* σ****ϕ***(**x**)***ϵ*** and substitute it into the above equation. This will give us^(3)

E*q****ϕ***(**z***|***x**)[log* p****θ***(**x***|***z**)] =* − *Z* ∥***x*** −**f****θ***(**z**)*∥*2

2*σ*^(2 )dec *· N * **z ***** µ******ϕ***(**x**)*, σ*2 ***ϕ***^(()^(**x**)^() ) *d***z**

*≈−*^(1)

*M*

*M *X

*m*=1

*∥***x*** −**f****θ***(**z**^(()^(*m*)^()))*∥*^(2)

2*σ*^(2 )dec (1.31)

=* −*^(1)

*M*

*M *X

*m*=1

*∥***x*** −**f****θ *** ***µ******ϕ***(**x**) +* σ****ϕ***(**x**)***ϵ***^(()^(*m*)^())^( )*∥*^(2)

2*σ*^(2 )dec *.*

The approximation above is due to Monte Carlo where the randomness is based on the sampling of the ***ϵ**** ∼N*(***ϵ**** |* 0*,*** I**). The index* M* specifies the number of Monte Carlo samples we want to use to approximate the expectation. Note that the input image** x** is fixed because E*q****ϕ***(**z***|***x**)[log* p****θ***(**x***|***z**)] is a function of** x**. The gradient of E*q****ϕ***(**z***|***x**)[log* p****θ***(**x***|***z**)] with respect to*** θ*** is relatively easy to compute. Since only* f****θ ***depends on*** θ***, we can do automatic differentiation. The gradient with respect to*** ϕ*** is slightly harder, but it is still computable because we use chain rule and go into*** µ******ϕ***(**x**) and*** ϕ******ϕ***(**x**). Inspecting Eqn (1.31), we notice one interesting thing that the loss function is simply the* ℓ*2 norm between the reconstructed image* f****θ***(**z**) and the ground truth image** x**. This means that if we have the generated image* f****θ***(**z**), we can do a direct comparison with the ground truth** x** via the usual* ℓ*2 loss as illustrated in Figure 1.6.

3The negative sign here is not a mistake. We want to* maximize* E*q****ϕ***(**z***|***x**)[log* p****θ***(**x***|***z**)], which is equivalent to* minimize* the negative of the* ℓ*2 norm.

© 2024 Stanley Chan. All Rights Reserved. 15


---

## Page 16

Figure 1.6: Implementation of a VAE decoder. We use a neural network to take the latent vector** z** and generate an image* f****θ***(**z**). The log likelihood will give us a quadratic equation if we assume a Gaussian distribution.

**Training the VAE**. Given a training dataset* X* =* {*(**x**^(()^(*ℓ*)^()))*}*^(*L *)*ℓ*=1 ^(of clean images, the training objective )of VAE is to maximize the ELBO

argmax ***θ****,****ϕ***

X

**x***∈X *ELBO***ϕ****,****θ***(**x**)*,*

where the summation is taken with respect to the entire training dataset. The individual ELBO is based on the sum of the terms we derived above

ELBO***ϕ****,****θ***(**x**) = E*q****ϕ***(**z***|***x**)[log* p****θ***(**x***|***z**)]* −*DKL  *q****ϕ***(**z***|***x**)* ∥**p*(**z**)  *. *(1.32)

Here, the reconstruction term is:

E*q****ϕ***(**z***|***x**)[log* p****θ***(**x***|***z**)]* ≈−*^(1)

*M*

*M *X

*m*=1

*∥***x*** −**f****θ *** ***µ******ϕ***(**x**) +* σ****ϕ***(**x**)***ϵ***^(()^(*m*)^())^( )*∥*^(2)

2*σ*^(2 )dec *, *(1.33)

whereas the prior matching term is

DKL  *q****ϕ***(**z***|***x**)* ∥**p*(**z**)  = ^(1)

2

 *σ*^(2 )***ϕ***^(()^(**x**)^())^(*d*)^(* −*)^(*d*)^( +)^(* ∥*)^(***µ***)***ϕ***^(()^(**x**)^())^(*∥*)^(2)^(* −*)^(2 log)^(* σ*)^(***ϕ***)^(()^(**x**)^() ) *. *(1.34)

To optimize for*** θ*** and*** ϕ***, we can run stochastic gradient descent. The gradients can be taken based on the tensor graphs of the neural networks. On computers, this is done automatically by the automatic differentiation. Let’s summarize these.

**Theorem 1.4.** (**VAE Training**). To train a VAE, we need to solve the optimization problem

argmax ***θ****,****ϕ***

X

**x***∈X *ELBO***ϕ****,****θ***(**x**)*,*

where

ELBO***ϕ****,****θ***(**x**) =* −*^(1)

*M*

*M *X

*m*=1

*∥***x*** −**f****θ *** ***µ******ϕ***(**x**) +* σ****ϕ***(**x**)***ϵ***^(()^(*m*)^())^( )*∥*^(2)

2*σ*^(2 )dec

+ ^(1)

2

 *σ*^(2 )***ϕ***^(()^(**x**)^())^(*d*)^(* −*)^(*d*)^( +)^(* ∥*)^(***µ***)***ϕ***^(()^(**x**)^())^(*∥*)^(2)^(* −*)^(2)^(*d*)^( log)^(* σ*)^(***ϕ***)^(()^(**x**)^() ) *. *(1.35)

**VAE Inference**. The inference of an VAE is relatively simple. Once the VAE is trained, we can drop the encoder and only keep the decoder, as shown in Figure 1.7. To generate a new image from the model, we pick a random latent vector** z*** ∈*R^(*d*). By sending this** z** through the decoder* f****θ***, we will be able to generate a new image b**x** =* f****θ***(**z**).

© 2024 Stanley Chan. All Rights Reserved. 16


---

## Page 17

Figure 1.7: Using VAE to generate image is as simple as sending a latent noise code** z** through the decoder.

**1.4 Concluding Remark**

For readers who are looking for additional references, we highly recommend the tutorial by Kingma and Welling [24] which is based on their original VAE paper [23]. A shorter tutorial by Doersch et al [12] can also be helpful. [24] includes a long list of good papers including a paper by Rezende and Mohamed [32] on normalizing flow which was published around the same time as Kingma and Welling’s VAE paper. VAE has many linkages to the classical variational inference and graphical models [45]. VAE is also relevant to the generative adversarial networks (GAN) by Goodfellow et al. [15]. Kingma and Welling com- mented in [24] that VAE and GAN have complementary properties; while GAN produces better perceptual quality images, there is a weaker linkage with the data likelihood. VAE can meet the data likelihood criterion better but the samples are at times not perceptually as good.

© 2024 Stanley Chan. All Rights Reserved. 17


---

## Page 18

### **2 Denoising Diffusion Probabilistic Model (DDPM)**

In this section, we discuss the diffusion models. There are many different perspectives on how the diffusion models can be derived, e.g., score matching, differential equation, etc. We will follow the approach outlined by the original paper on denoising diffusion probability model by Ho et al. [16]. Before we discuss the mathematical details, let’s summarize DDPM from the perspective of VAE’s extension:

Diffusion models are* incremental* updates where the assembly of the whole gives us the encoder-decoder structure.

Why increment? It’s like turning the direction of a giant ship. You need to turn the ship slowly towards your desired direction or otherwise you will lose control. The same principle applies to your company HR and your university administration.

Bend one inch at a time.

Okay. Enough philosophy. Let’s get back to our business.

DDPM has a lot of linkage to a piece of earlier work by Sohl-Dickstein et al in 2015 [38]. Sohl-Dickstein et al asked the question of how to convert from one distribution to another distribution. VAE provides one approach: Referring to the previous section, we can think of the source distribution being the latent variable** z*** ∼**p*(**z**) and the target distribution being the input variable** x*** ∼**p*(**x**). Then by setting up the proxy distributions* p****θ***(**x***|***z**) and* q****ϕ***(**z***|***x**), we can train the encoder and decoder so that the decoder will serve the goal of generating images. But VAE is largely a* one-step* generation — if you give us a latent code** z**, we ask the neural network* f****θ***(*·*) to immediately return us the generated signal** x*** ∼N*(**x*** |** f****θ***(**z**)*, σ*^(2 )dec^(**I**)^(). In )some sense, this is asking a lot from the neural network. We are asking it to use a few layers of neurons to immediately convert from one distribution* p*(**z**) to another distribution* p*(**x**). This is too much. The idea Sohl-Dickstein et al proposed was to construct a chain of conversions instead of a one-step process. To this end they defined two processes analogous to the encoder and decoder in a VAE. They call the encoder as the forward process, and the decoder as the reverse process. In both processes, they consider a sequence of variables** x**0*, . . . ,*** x***T* whose joint distribution is denoted as* q****ϕ***(**x**0:*T* ) and* p****θ***(**x**0:*T* ) respectively for the forward and reverse processes. To make both processes tractable (and also flexible), they impose a Markov chain structure (i.e., memoryless) where

forward from** x**0 to** x***T* : *q****ϕ***(**x**0:*T* ) =* q*(**x**0)

*T *Y

*t*=1 *q****ϕ***(**x***t** |*** x***t**−*1)*,*

reverse from** x***T* to** x**0 : *p****θ***(**x**0:*T* ) =* p*(**x***T* )

*T *Y

*t*=1 *p****θ***(**x***t**−*1* |*** x***t*)*.*

In both equations, the transition distributions are only dependent on its immediate previous stage. Therefore, if each transition is realized through some form of neural networks, the overall generation process is broken down into many smaller tasks. It does not mean that we will need* T* times more neural networks. We are just re-using one network for* T* times. Breaking the overall process into smaller steps allows us to use simple distributions at each step. As will be discussed in the following subsections, we can use Gaussian distributions for the transitions. Thanks to the properties of a Gaussian, the posterior will remain a Gaussian if the likelihood and the prior are both Gaussians. Therefore, if each transitional distribution above is a Gaussian, the joint distribution is also a Gaussian. Since a Gaussian is fully characterized by the first two moments (mean and variance), the computation is highly tractable. In the original paper of Sohl-Dickstein et al, there is also a case study of binomial diffusion processes.

After providing a high-level overview of the concepts, let’s talk about some details. The starting point of the diffusion model is to consider the VAE structure and make it a chain of incremental updates as shown in

© 2024 Stanley Chan. All Rights Reserved. 18


---

## Page 19

Figure 2.1. This particular structure is called the** variational diffusion model**, a name given by Kingma et al in 2021 [22]. The variational diffusion model has a sequence of states** x**0*,*** x**1*, . . . ,*** x***T* with the following interpretations:

•** x**0: It is the original image, which is the same as** x** in VAE. •** x***T* : It is the latent variable, which is the same as** z** in VAE. As explained above, we choose** x***T** ∼N*(0*,*** I**) for simplicity, tractability, and computational efficiency. •** x**1*, . . . ,*** x***T** −*1: They are the intermediate states. They are also the latent variables, but they are not white Gaussian.

The structure of the variational diffusion model consists of two paths. The forward and the reverse paths are analogous to the paths of a single-step variational autoencoder. The difference is that the encoders and decoders have identical input-output dimensions. The assembly of all the forward building blocks will give us the encoder, and the assembly of all the reverse building blocks will give us the decoder.

Figure 2.1: Variational diffusion model by Kingma et al [22]. In this model, the input image is** x**0 and the white noise is** x***T* . The intermediate variables (or states)** x**1*, . . . ,*** x***T** −*1 are latent variables. The transition from** x***t**−*1 to** x***t* is analogous to the forward step (encoder) in VAE, whereas the transition from** x***t* to** x***t**−*1 is analogous to the reverse step (decoder) in VAE. In variational diffusion models, the input dimension and the output dimension of the encoders/decoders are identical.

**2.1 Building Blocks**

Let’s talk about the building blocks of the variational diffusion model. There are three classes of building blocks: the transition block, the initial block, and the final block. **Transition Block** The* t*-th transition block consists of three states** x***t**−*1,** x***t*, and** x***t*+1. There are two possible paths to get to state** x***t*, as illustrated in Figure 2.2.

Figure 2.2: The transition block of a variational diffusion model consists of three nodes. The transition distributions* p*(**x***t**|***x***t*+1) and* p*(**x***t**|***x***t**−*1) are not accessible, but we can approximate them by Gaussians.

© 2024 Stanley Chan. All Rights Reserved. 19


---

## Page 20

• The first path is the forward transition going from** x***t**−*1 to** x***t*. The associated transition distribution is* p*(**x***t**|***x***t**−*1). In plain words, if you tell us** x***t**−*1, we can draw a sample** x***t* according to* p*(**x***t**|***x***t**−*1). However, just like a VAE, the transition distribution* p*(**x***t**|***x***t**−*1) is not accessible. We can approximate it by some simple distributions such as a Gaussian. The approximated distribution is denoted as *q****ϕ***(**x***t**|***x***t**−*1). We will discuss the exact form of* q****ϕ*** later. • The second path is the reverse transition going from** x***t*+1 to** x***t*. Again, we do not know* p*(**x***t**|***x***t*+1) and so we have another proxy distribution, e.g. a Gaussian, to approximate the true distribution. This proxy distribution is denoted as* p****θ***(**x***t**|***x***t*+1).

**Initial Block** The initial block of the variational diffusion model focuses on the state** x**0. Since we start at** x**0, we only need the reverse transition from** x**1 to** x**0. The forward transition from** x***−*1 to** x**0 can be dropped. Therefore, we only need to consider* p*(**x**0*|***x**1). But since* p*(**x**0*|***x**1) is never accessible, we approximate it by a Gaussian* p****θ***(**x**0*|***x**1) where the mean is computed through a neural network. See Figure 2.3 for illustration.

Figure 2.3: The initial block of a variational diffusion model focuses on the node** x**0. Since there is no state before time* t* = 0, we only have a reverse transition from** x**1 to** x**0.

**Final Block**. The final block focuses on the state** x***T* . Remember that** x***T* is supposed to be our final latent variable which is a white Gaussian noise vector. Because it is the final block, we only need a forward transition from** x***T** −*1 to** x***T* , and nothing such as** x***T* +1 to** x***T* . The forward transition is approximated by *q****ϕ***(**x***T** |***x***T** −*1) which is a Gaussian. See Figure 2.4 for illustration.

Figure 2.4: The final block of a variational diffusion model focuses on the node** x***T* . Since there is no state after time* t* =* T*, we only have a forward transition from** x***T** −*1 to** x***T* .

**Understanding the Transition Distribution**. Before we proceed further, we need to explain the transition distribution* q****ϕ***(**x***t**|***x***t**−*1). We know that it is a Gaussian. But what is the mean and variance of this Gaussian?

**Definition 2.1. Transition Distribution*** q****ϕ***(**x***t**|***x***t**−*1). In a variational diffusion model (and also DDPM which we will discuss later), the transition distribution* q****ϕ***(**x***t**|***x***t**−*1) is defined as

*q****ϕ***(**x***t**|***x***t**−*1) def =* N*(**x***t** | *^(*√*)*α**t***x***t**−*1*,* (1* −**α**t*)**I**)*. *(2.1)

In other words,* q****ϕ***(**x***t**|***x***t**−*1) is a Gaussian. The mean is* *^(*√*)*α**t***x***t**−*1 and the variance is 1* −**α**t*. The choice of the scaling factor* *^(*√*)*α**t* is to make sure that the variance magnitude is preserved so that it will not explode and vanish after many iterations.

© 2024 Stanley Chan. All Rights Reserved. 20


---

## Page 21

**Example 2.1.** Let’s consider a Gaussian mixture model

**x**0* ∼**p*0(**x**) =* π*1*N*(**x***|**µ*1*, σ*^(2 )1^() +)^(* π*)^(2)^(*N*)^(()^(**x**)^(*|*)^(*µ*)^(2)^(*, σ*)^(2 )2^())^(*.*)

Given the transition probability, we know that if** x***t** ∼**q****ϕ***(**x***t**|***x***t**−*1) then

**x***t* =* *^(*√*)*α**t***x***t**−*1 + p

(1* −**α**t*)***ϵ****, *where*** ϵ**** ∼N*(0*,*** I**)*.*

Our goal is to see whether this iterative procedure (using the above transition probability) will give us a white Gaussian in the equilibrium state (i.e., when* t** →∞*). For a mixture model, it is not difficult to show that the probability distribution of** x***t* can be calculated recursively via the algorithm for* t* = 1*,* 2*, . . . , T*: (the proof will be shown later)

**x***t** ∼**p**t*(**x**) =*π*1*N*(**x***|*^(*√*)*α**t**µ*1*,t**−*1*, α**t**σ*^(2 )1*,t**−*1 ^(+ (1)^(* −*)^(*α*)^(*t*)^()))

+* π*2*N*(**x***|*^(*√*)*α**t**µ*2*,t**−*1*, α**t**σ*^(2 )2*,t**−*1 ^(+ (1)^(* −*)^(*α*)^(*t*)^()))^(*, *)(2.2)

where* µ*1*,t**−*1 is the mean for class 1 at time* t** −*1, with* µ*1*,*0 =* µ*1 being the initial mean. Similarly, *σ*^(2 )1*,t**−*1 ^(is the variance for class 1 at time)^(* t*)^(* −*)^(1, with)^(* σ*)^(2 )1*,*0 ^(=)^(* σ*)^(2 )1 ^(being the initial variance. )In the figure below, we show a numerical example where* π*1 = 0*.*3,* π*2 = 0*.*7,* µ*1 =* −*2,* µ*2 = 2, *σ*1 = 0*.*2, and* σ*2 = 1. The rate is defined as* α**t* = 0*.*97 for all* t*. We plot the probability distribution function for different* t*.

Figure 2.5: Evolution of the distribution* p**t*(**x**). As time* t* progresses, the bimodal distribution gradually becomes a Gaussian.

**Proof of Eqn** (2.2). For those who would like to understand how we derive the probability density of a mixture model in Eqn (2.2), we can show a simple derivation. Consider a mixture model

*p*(**x**) =

*K *X

*k*=1 *π**k**N*(**x***|**µ**k**, σ*^(2 )*k*^(**I**)^() )| {z } *p*(**x***|**k*)

*.*

If we consider a new variable** y** =* *^(*√*)*α***x** +* *^(*√*)1* −**α****ϵ*** where*** ϵ**** ∼N*(0*,*** I**), then the distribution of** y** can be derived by using the law of total probability:

*p*(**y**) =

*K *X

*k*=1 *p*(**y***|**k*)*p*(*k*) =

*K *X

*k*=1 *π**k**p*(**y***|**k*)*.*

Since** y***|**k* =* *^(*√*)*α***x***|**k* +* *^(*√*)1* −**α****ϵ*** is a linear combination of a (conditioned) Gaussian random variable **x***|**k* and another Gaussian random variable*** ϵ***, the sum** y***|**k* will remain as a Gaussian. The mean and

© 2024 Stanley Chan. All Rights Reserved. 21


---

## Page 22

variance are

E[**y***|**k*] =* *^(*√*)*α*E[**x***|**k*] + *√*

1* −**α*E[***ϵ***] =* *^(*√*)*αµ**k**,*

Var[**y***|**k*] =* α*Var[**x***|**k*] + (1* −**α*)Var[***ϵ***] =* ασ*^(2 )*k* ^(+ (1)^(* −*)^(*α*)^())^(*,*)

where we used the fact that E[***ϵ***] = 0, and Var[***ϵ***] = 1. Since we just argued that** y***|**k* is a Gaussian, the distribution of** y***|**k* is completely specified once we know the mean and variance. Substituting the above derived results, we know that* p*(**y***|**k*) =* N*(**y***|*^(*√*)*αµ**k**, ασ*^(2 )*k* ^(+ (1)^(* −*)^(*α*)^()). This completes the derivation.)

**The magical scalars*** *^(*√*)*α**t*** and** 1* −**α**t*. You may wonder how the genius people (the authors of the denoising diffusion papers) come up with the magical scalars* *^(*√*)*α**t* and (1* −**α**t*) for the above transition probability. To demystify this, let’s consider two unrelated scalars* a** ∈*R and* b** ∈*R, and define the transition distribution as *q****ϕ***(**x***t**|***x***t**−*1) =* N*(**x***t** |** a***x***t**−*1*, b*^(2)**I**)*. *(2.3)

Here is the finding:

**Theorem 2.1. (Why*** *^(*√*)*α*** and** 1* −**α***?) **Suppose that* q****ϕ***(**x***t**|***x***t**−*1) =* N*(**x***t** |** a***x***t**−*1*, b*^(2)**I**) for some constants* a* and* b*. If we want to choose* a* and* b* such that the distribution of** x***t* will become* N*(0*,*** I**), then it is necessary that *a* =* *^(*√*)*α *and *b* = *√*

1* −**α.*

Therefore, the transition distribution is

*q****ϕ***(**x***t**|***x***t**−*1) def =* N*(**x***t** | *^(*√*)*α***x***t**−*1*,* (1* −**α*)**I**)*. *(2.4)

Remark: You can replace* α* by* α**t*, if you prefer a noise schedule.

**Proof**. We want to show that* a* =* *^(*√*)*α* and* b* =* *^(*√*)1* −**α*. For the distribution shown in Eqn (2.3), the equivalent sampling step is:

**x***t* =* a***x***t**−*1 +* b****ϵ****t**−*1*, *where ***ϵ****t**−*1* ∼N*(0*,*** I**)*. *(2.5)

We can carry on the recursion to show that

**x***t* =* a***x***t**−*1 +* b****ϵ****t**−*1 =* a*(*a***x***t**−*2 +* b****ϵ****t**−*2) +* b****ϵ****t**−*1 (substitute** x***t**−*1 =* a***x***t**−*2 +* b****ϵ****t**−*2)

=* a*^(2)**x***t**−*2 +* ab****ϵ****t**−*2 +* b****ϵ****t**−*1 (regroup terms )

= ^(..).

=* a*^(*t*)**x**0 +* b * ***ϵ****t**−*1 +* a****ϵ****t**−*2 +* a*^(2)***ϵ****t**−*3 +* . . .* +* a*^(*t*)^(*−*)^(1)***ϵ***0 

| {z }

def =** w***t*

*. *(2.6)

The finite sum above is a sum of independent Gaussian random variables. The mean vector E[**w***t*] remains zero because everyone has a zero mean. The covariance matrix (for a zero-mean vector) is

Cov[**w***t*] def = E[**w***t***w**^(*T *)*t* ^(])

=* b*^(2)(Cov(***ϵ****t**−*1) +* a*^(2)Cov(***ϵ****t**−*2) +* . . .* + (*a*^(*t*)^(*−*)^(1))^(2)Cov(***ϵ***0))

=* b*^(2)(1 +* a*^(2)^( )+* a*^(4)^( )+* . . .* +* a*^(2()^(*t*)^(*−*)^(1)))**I**

=* b*^(2)^(* *)*·* ^(1)^(* −*)^(*a*)^(2)^(*t*)

1* −**a*^(2)^(** I**)^(*.*)

© 2024 Stanley Chan. All Rights Reserved. 22


---

## Page 23

As* t** →∞*,* a*^(*t*)^(* *)*→*0 for any 0* < a <* 1. Therefore, at the limit when* t* =* ∞*,

lim *t**→∞*^(Cov[)^(**w**)^(*t*)^(] = )*b*^(2)

1* −**a*^(2)^(** I**)^(*.*)

So, if we want lim*t**→∞*Cov[**w***t*] =** I** (so that the distribution of** x***t* will approach* N*(0*,*** I**)), then we need

1 = *b*^(2)

1* −**a*^(2)^(* ,*)

or equivalently* b* = *√*

1* −**a*^(2). Now, if we let* a* =* *^(*√*)*α*, then* b* =* *^(*√*)1* −**α*. This will give us

**x***t* =* *^(*√*)*α***x***t**−*1 + *√*

1* −**α****ϵ****t**−*1*. *(2.7)

**Distribution*** q****ϕ***(**x***t**|***x**0). With the understanding of the magical scalars, we can talk about the distri- bution* q****ϕ***(**x***t**|***x**0). That is, we want to know how** x***t* will be distributed if we are given** x**0.

**Theorem 2.2. (Conditional Distribution*** q****ϕ***(**x***t**|***x**0)**)**. The conditional distribution* q****ϕ***(**x***t**|***x**0) is given by *q****ϕ***(**x***t**|***x**0) =* N*(**x***t** | √*

*α**t***x**0*,* (1* −**α**t*)**I**)*, *(2.8)

where* α**t* = ^(Q)^(*t *)*i*=1* *^(*α*)^(*i*)^(.)

**Proof**. To see how Eqn (2.8) is derived, we can re-do the recursion but this time we use* *^(*√*)*α**t***x***t**−*1 and (1* −**α**t*)**I** as the mean and covariance, respectively. This will give us

**x***t* =* *^(*√*)*α**t***x***t**−*1 + *√*

1* −**α**t****ϵ****t**−*1

=* *^(*√*)*α**t*(^(*√*)*α**t**−*1**x***t**−*2 + p

1* −**α**t**−*1***ϵ****t**−*2) + *√*

1* −**α**t****ϵ****t**−*1

=* *^(*√*)*α**t**α**t**−*1**x***t**−*2 +* *^(*√*)*α**t *p

1* −**α**t**−*1***ϵ****t**−*2 + *√*

1* −**α**t****ϵ****t**−*1 | {z } **w**1

*. *(2.9)

Therefore, we have a sum of two Gaussians. But since sum of two Gaussians remains a Gaussian, we can just calculate its new covariance (because the mean remains zero). The new covariance is

E[**w**1**w**^(*T *)1 ^(] = [()^(*√*)^(*α*)^(*t *)p

1* −**α**t**−*1)^(2)^( )+ ( *√*

1* −**α**t*)^(2)]**I**

= [*α**t*(1* −**α**t**−*1) + 1* −**α**t*]**I** = [1* −**α**t**α**t**−*1]**I***.*

Returning to Eqn (2.9), we can show that the recursion becomes a linear combination of** x***t**−*2 and a noise vector*** ϵ****t**−*2:

**x***t* =* *^(*√*)*α**t**α**t**−*1**x***t**−*2 + p

1* −**α**t**α**t**−*1***ϵ****t**−*2

=* *^(*√*)*α**t**α**t**−*1*α**t**−*2**x***t**−*3 + p

1* −**α**t**α**t**−*1*α**t**−*2***ϵ****t**−*3

= ^(..).

=





v u u t

*t*Y

*i*=1 *α**i*



**x**0 +





v u u t1* −*

*t*Y

*i*=1 *α**i*



***ϵ***0*. *(2.10)

So, if we define* α**t* = ^(Q)^(*t *)*i*=1* *^(*α*)^(*i*)^(, we can show that)

**x***t* =* *^(*√*)*α**t***x**0 + *√*

1* −**α**t****ϵ***0*. *(2.11)

© 2024 Stanley Chan. All Rights Reserved. 23


---

## Page 24

In other words, the distribution* q****ϕ***(**x***t**|***x**0) is

**x***t** ∼**q****ϕ***(**x***t**|***x**0) =* N*(**x***t** | *^(*√*)*α**t***x**0*,* (1* −**α**t*)**I**)*. *(2.12)

The utility of the new distribution* q****ϕ***(**x***t**|***x**0) is its one-shot forward diffusion step compared to the chain **x**0* →***x**1* →**. . .** →***x***T** −*1* →***x***T* . In every step of the forward diffusion model, since we already know** x**0 and we assume that all subsequence transitions are Gaussian, we will know** x***t* for any* t*. The situation can be understood from Figure 2.6.

Figure 2.6: The difference between* q****ϕ***(**x***t**|***x***t**−*1) and* q****ϕ***(**x***t**|***x**0).

**Example 2.2.** For a Gaussian mixture model such that** x**0* ∼**p*0(**x**) = ^(P)^(*K *)*k*=1* *^(*π*)^(*k*)^(*N*)^(()^(**x**)^(*|*)^(***µ***)*k*^(*, σ*)^(2 )*k*^(**I**)^(), we can )show that the distribution at time* t* is

**x***t** ∼**p**t*(**x**) =

*K *X

*k*=1 *π**k**N*(**x*** | *^(*√*)*α**t****µ****k**,* (1* −**α**t*)**I** +* α**t**σ*^(2 )*k*^(**I**)^() )(2.13)

=

*K *X

*k*=1 *π**k**N*(**x*** | √*

*α*^(*t*)***µ****k**,* (1* −**α*^(*t*))**I** +* α*^(*t*)*σ*^(2 )*k*^(**I**)^())^(*, *)if* α**t* =* α* so that* α**t* =

*t*Y

*i*=1 *α* =* α*^(*t*)*.*

If you are curious about how the probability distribution* p**t* evolves over time* t*, we can visualize the trajectory of a Gaussian mixture distribution we discussed in Example 2.1. We use Eqn (2.13) to plot the heatmap. You can see that when* t* = 0, the initial distribution is a mixture of two Gaussians. As we progress by following the transition defined in Eqn (2.13), we can see that the distribution gradually becomes the single Gaussian* N*(0*,*** I**).

Figure 2.7: Realizations of random trajectories made by** x***t*. The color map in the background indicates the probability distribution* p**t*(**x**).

In the same plot, we overlay and show a few instantaneous trajectories of the random samples** x***t *as a function of time* t*. The equation we used to generate the samples is

**x***t* =* *^(*√*)*α**t***x***t**−*1 + *√*

1* −**α**t****ϵ****, ****ϵ**** ∼N*(0*,*** I**)*.*

As you can see, the trajectories of** x***t* more or less follow the distribution* p**t*(**x**).

© 2024 Stanley Chan. All Rights Reserved. 24


---

## Page 25

A confusing point to many readers is that if the goal is to convert from an image* p*(**x**0) to white noise *p*(**x***T* ), what is the point of deriving* q****ϕ***(**x***t**|***x***t**−*1) and* q****ϕ***(**x***t** |*** x**0)? The answer is that so far we have been just talking about the* forward* process. In a diffusion model, the forward process is chosen such that they can be expressed in a closed form. The more interesting part is the* reverse* process. As will be discussed, the reverse process is realized through a chain of denoising operations. Each denoising step should be coupled with the corresponding step in the forward process.* q****ϕ***(**x***t** |*** x**0) just provides us a slightly more convenient way to implement the forward process.

**2.2 Evidence Lower Bound**

Now that we understand the structure of the variational diffusion model, we can write down the ELBO and hence train the model.

**Theorem 2.3. (ELBO for Variational Diffusion Model)**. The ELBO for the variational diffusion model is

ELBO***ϕ****,****θ***(**x**) = E*q****ϕ***(**x**1*|***x**0) h log *p****θ***(**x**0*|***x**1) | {z } how good the initial block is

i

*−*E*q****ϕ***(**x***T** −*1*|***x**0) h DKL  *q****ϕ***(**x***T** |***x***T** −*1)*∥**p*(**x***T* ) 

| {z } how good the final block is

i

*−*

*T** −*1 X

*t*=1 E*q****ϕ***(**x***t**−*1*,***x***t*+1*|***x**0) h DKL  *q****ϕ***(**x***t**|***x***t**−*1)*∥**p****θ***(**x***t**|***x***t*+1) 

| {z } how good the transition blocks are

i *, *(2.14)

where** x**0 =** x**, and** x***T** ∼N*(0*,*** I**).

If you are a casual reader of this tutorial, we hope that this equation does not throw you off. While it appears a monster, it does have structures. We just need to be patient when we try to understand it. **Reconstruction** (Initial Block). Let’s first look at the term

E*q****ϕ***(**x**1*|***x**0) h log* p****θ***(**x**0*|***x**1) i *.*

This term is based on the initial block and it is analogous to Eqn (1.10). The subject inside the expectation is the log-likelihood log* p****θ***(**x**0*|***x**1). This log-likelihood measures how good the neural network (associated with* p****θ***) can recover** x**0 from the latent variable** x**1. The expectation is taken with respect to the samples drawn from* q****ϕ***(**x**1*|***x**0). Recall that* q****ϕ***(**x**1*|***x**0) is the distribution that generates** x**1. We require** x**1 to be drawn from this distribution because** x**1 do not come from the sky but* created* by the forward transition* q****ϕ***(**x**1*|***x**0). The conditioning on** x**0 is needed here because we need to know what the original image is. The reason why expectation is used here is that* p****θ***(**x**0*|***x**1) is a function of** x**1 (and so if** x**1 is random then *p****θ***(**x**0*|***x**1) is random too). For a different intermediate state** x**1, the probability* p****θ***(**x**0*|***x**1) will be different. The expectation eliminates the dependency on** x**1.

**Prior Matching** (Final Block). The prior matching term is

*−*E*q****ϕ***(**x***T** −*1*|***x**0) h DKL  *q****ϕ***(**x***T** |***x***T** −*1)*∥**p*(**x***T* ) i *, *(2.15)

and it is based on the final block. We use the KL divergence to measure the difference between* q****ϕ***(**x***T** |***x***T** −*1) and* p*(**x***T* ). The distribution* q****ϕ***(**x***T** |***x***T** −*1) is the forward transition from** x***T** −*1 to** x***T* . This describes how **x***T* is generated. The second distribution is* p*(**x***T* ). Because of our laziness, we assume that* p*(**x***T* ) =* N*(0*,*** I**). We want* q****ϕ***(**x***T** |***x***T** −*1) to be as close to* N*(0*,*** I**) as possible. When computing the KL-divergence, the variable** x***T* is a dummy variable. However, since* q****ϕ*** is condi- tioned on** x***T** −*1, the KL-divergence calculated here is a function of the conditioned variable** x***T** −*1. Where

© 2024 Stanley Chan. All Rights Reserved. 25


---

## Page 26

does** x***T** −*1 come from? It is generated by* q****ϕ***(**x***T** −*1*|***x**0). We use a conditional distribution* q****ϕ***(**x***T** −*1*|***x**0) because** x***T** −*1 depends on what** x**0 we use in the first place. The expectation over* q****ϕ***(**x***T** −*1*|***x**0) says that for each of the** x***T** −*1 generated by* q****ϕ***(**x***T** −*1*|***x**0), we will have a value of the KL divergence. We take the expectation over all the possible** x***T** −*1 generated to eliminate the dependency.

**Consistency**. (Transition Blocks) The consistency term is

*−*

*T** −*1 X

*t*=1 E*q****ϕ***(**x***t**−*1*,***x***t*+1*|***x**0) h DKL  *q****ϕ***(**x***t**|***x***t**−*1)*∥**p****θ***(**x***t**|***x***t*+1) i *, *(2.16)

and it is based on the transition blocks. There are two directions if you recall Figure 2.2. The forward transition is determined by the distribution* q****ϕ***(**x***t**|***x***t**−*1) whereas the reverse transition is determined by another distribution* p****θ***(**x***t**|***x***t*+1). The consistency term uses the KL divergence to measure the deviation. The expectation is taken with respect to the pair of samples (**x***t**−*1*,*** x***t*+1), drawn from* q****ϕ***(**x***t**−*1*,*** x***t*+1*|***x**0). The reason is that the KL divergence above is a function of** x***t**−*1 and** x***t*+1. (You can ignore** x***t* because it is a dummy variable that will be eliminated during the integration process when we calculate the expectation.) Because of the dependencies on** x***t**−*1 and** x***t*+1, we need to take the expectation.

**Proof of Theorem 2.3**. Let’s define the following notation:** x**0:*T* =* {***x**0*, . . . ,*** x***T** }* means the collection of all state variables from* t* = 0 to* t* =* T*. We also recall that the prior distribution* p*(**x**) is the distribution for the image** x**0. So it is equivalent to* p*(**x**0). With these in mind, we can show that

log* p*(**x**) = log* p*(**x**0)

= log Z *p*(**x**0:*T* )*d***x**1:*T *(Marginalize by integrating over** x**1:*T* )

= log Z *p*(**x**0:*T* )^(*q*)^(***ϕ***)^(()^(**x**)^(1:)^(*T*)^(* |*)^(**x**)^(0)^())

*q****ϕ***(**x**1:*T** |***x**0)^(*d*)^(**x**)^(1:)^(*T *)(Multiply and divide* q****ϕ***(**x**1:*T** |***x**0))

= log Z

*q****ϕ***(**x**1:*T** |***x**0)

 *p*(**x**0:*T* ) *q****ϕ***(**x**1:*T** |***x**0)

 *d***x**1:*T *(Rearrange terms)

= log E*q****ϕ***(**x**1:*T** |***x**0)

 *p*(**x**0:*T* ) *q****ϕ***(**x**1:*T** |***x**0)

(Definition of expectation)*.*

Now, we need to use Jensen’s inequality, which states that for any random variable* X* and any concave function* f*, it holds that* f*(E[*X*])* ≥*E[*f*(*X*)]. By recognizing that* f*(*·*) = log(*·*), we can show that

log* p*(**x**) = log E*q****ϕ***(**x**1:*T** |***x**0)

 *p*(**x**0:*T* ) *q****ϕ***(**x**1:*T** |***x**0)

 *≥*E*q****ϕ***(**x**1:*T** |***x**0)

 log *p*(**x**0:*T* ) *q****ϕ***(**x**1:*T** |***x**0)

 (2.17)

Let’s take a closer look at* p*(**x**0:*T* ). Inspecting Figure 2.2, we notice that if we want to decouple* p*(**x**0:*T* ), we should do conditioning for** x***t**−*1*|***x***t*. This leads to:

*p*(**x**0:*T* ) =* p*(**x***T* )

*T *Y

*t*=1 *p*(**x***t**−*1*|***x***t*) =* p*(**x***T* )*p*(**x**0*|***x**1)

*T *Y

*t*=2 *p*(**x***t**−*1*|***x***t*)*. *(2.18)

As for* q****ϕ***(**x**1:*T** |***x**0), Figure 2.2 suggests that we need to do the conditioning for** x***t**|***x***t**−*1. However, because of the sequential relationship, we can write

*q****ϕ***(**x**1:*T** |***x**0) =

*T *Y

*t*=1 *q****ϕ***(**x***t**|***x***t**−*1) =* q****ϕ***(**x***T** |***x***T** −*1)

*T** −*1 Y

*t*=1 *q****ϕ***(**x***t**|***x***t**−*1)*. *(2.19)

© 2024 Stanley Chan. All Rights Reserved. 26


---

## Page 27

Substituting Eqn (2.18) and Eqn (2.19) back to Eqn (2.17), we can show that

log* p*(**x**)* ≥*E*q****ϕ***(**x**1:*T** |***x**0)

 log *p*(**x**0:*T* ) *q****ϕ***(**x**1:*T** |***x**0)

= E*q****ϕ***(**x**1:*T** |***x**0)

"

log* *^(*p*)^(()^(**x**)^(*T*)^( ))^(*p*)^(()^(**x**)^(0)^(*|*)^(**x**)^(1)^())^( Q)^(*T *)*t*=2* *^(*p*)^(()^(**x**)^(*t*)^(*−*)^(1)^(*|*)^(**x**)^(*t*)^())

*q****ϕ***(**x***T** |***x***T** −*1) ^(Q)^(*T*)^(* −*)^(1 )*t*=1* *^(*q*)^(***ϕ***)^(()^(**x**)^(*t*)^(*|*)^(**x**)^(*t*)^(*−*)^(1)^())

#

= E*q****ϕ***(**x**1:*T** |***x**0)

"

log* *^(*p*)^(()^(**x**)^(*T*)^( ))^(*p*)^(()^(**x**)^(0)^(*|*)^(**x**)^(1)^())^(Q)^(*T*)^(* −*)^(1 )*t*=1* *^(*p*)^(()^(**x**)^(*t*)^(*|*)^(**x**)^(*t*)^(+1)^())

*q****ϕ***(**x***T** |***x***T** −*1) ^(Q)^(*T*)^(* −*)^(1 )*t*=1* *^(*q*)^(***ϕ***)^(()^(**x**)^(*t*)^(*|*)^(**x**)^(*t*)^(*−*)^(1)^())

#

(shift* t* to* t* + 1)

= E*q****ϕ***(**x**1:*T** |***x**0)

 log* *^(*p*)^(()^(**x**)^(*T*)^( ))^(*p*)^(()^(**x**)^(0)^(*|*)^(**x**)^(1)^())

*q****ϕ***(**x***T** |***x***T** −*1)

 + E*q****ϕ***(**x**1:*T** |***x**0)

"

log

*T** −*1 Y

*t*=1

*p*(**x***t**|***x***t*+1) *q****ϕ***(**x***t**|***x***t**−*1)

#

(split expectation)

The first term above can be further decomposed into two expectations

E*q****ϕ***(**x**1:*T** |***x**0)

 log* *^(*p*)^(()^(**x**)^(*T*)^( ))^(*p*)^(()^(**x**)^(0)^(*|*)^(**x**)^(1)^())

*q****ϕ***(**x***T** |***x***T** −*1)

 = E*q****ϕ***(**x**1:*T** |***x**0)

 log* p*(**x**0*|***x**1) 

| {z } Reconstruction

+ E*q****ϕ***(**x**1:*T** |***x**0)

 log *p*(**x***T* ) *q****ϕ***(**x***T** |***x***T** −*1)

| {z } Prior Matching

*.*

The Reconstruction term can be simplified as

E*q****ϕ***(**x**1:*T** |***x**0)

 log* p*(**x**0*|***x**1)  = E*q****ϕ***(**x**1*|***x**0)

 log* p*(**x**0*|***x**1)  *,*

where we used the fact that the conditioning** x**1:*T** |***x**0 is equivalent to** x**1*|***x**0 when the subject of interest (i.e., log* p*(**x**0*|***x**1)) only involves** x**0 and** x**1. The Prior Matching term is

E*q****ϕ***(**x**1:*T** |***x**0)

 log *p*(**x***T* ) *q****ϕ***(**x***T** |***x***T** −*1)

 = E*q****ϕ***(**x***T** ,***x***T** −*1*|***x**0)

 log *p*(**x***T* ) *q****ϕ***(**x***T** |***x***T** −*1)

 *,*

where we note that the conditional expectation can be simplified to samples** x***T* and** x***T** −*1 only, because log *p*(**x***T* ) *q****ϕ***(**x***T** |***x***T** −*1) ^(only depends on)^(** x**)^(*T*)^( and)^(** x**)^(*T*)^(* −*)^(1)^(. For the expectation term, chain rule of probability tells )us that* q****ϕ***(**x***T** ,*** x***T** −*1*|***x**0) =* q****ϕ***(**x***T** |***x***T** −*1*,*** x**0)*q****ϕ***(**x***T** −*1*|***x**0). Since* q****ϕ*** is Markovian, we can further write *q****ϕ***(**x***T** |***x***T** −*1*,*** x**0) =* q****ϕ***(**x***T** |***x***T** −*1). Therefore, the joint expectation E*q****ϕ***(**x***T** ,***x***T** −*1*|***x**0) can be written as a product of two expectations E*q****ϕ***(**x***T** −*1*|***x**0)E*q****ϕ***(**x***T** |***x***T** −*1). This will give us

E*q****ϕ***(**x***T** ,***x***T** −*1*|***x**0)

 log *p*(**x***T* ) *q****ϕ***(**x***T** |***x***T** −*1)

 = E*q****ϕ***(**x***T** −*1*|***x**0)

 E*q****ϕ***(**x***T** |***x***T** −*1)

 log *p*(**x***T* ) *q****ϕ***(**x***T** |***x***T** −*1)

=* −*E*q****ϕ***(**x***T** −*1*|***x**0)

"

DKL (*q****ϕ***(**x***T** |***x***T** −*1)*∥**p*(**x***T* ))

#

*.*

Finally, we look at the product term. We can show that

E*q****ϕ***(**x**1:*T** |***x**0)

"

log

*T** −*1 Y

*t*=1

*p*(**x***t**|***x***t*+1) *q****ϕ***(**x***t**|***x***t**−*1)

#

=

*T** −*1 X

*t*=1 E*q****ϕ***(**x**1:*T** |***x**0)

 log* *^(*p*)^(()^(**x**)^(*t*)^(*|*)^(**x**)^(*t*)^(+1)^())

*q****ϕ***(**x***t**|***x***t**−*1)

=

*T** −*1 X

*t*=1 E*q****ϕ***(**x***t**−*1*,***x***t**,***x***t*+1*|***x**0)

 log* *^(*p*)^(()^(**x**)^(*t*)^(*|*)^(**x**)^(*t*)^(+1)^())

*q****ϕ***(**x***t**|***x***t**−*1)

 *,*

where again we use the fact the expectation only needs** x***t**−*1,** x***t*, and** x***t*+1. Then, by using the same

© 2024 Stanley Chan. All Rights Reserved. 27


---

## Page 28

conditional independence argument, we can show that

*T** −*1 X

*t*=1 E*q****ϕ***(**x***t**−*1*,***x***t**,***x***t*+1*|***x**0)

 log* *^(*p*)^(()^(**x**)^(*t*)^(*|*)^(**x**)^(*t*)^(+1)^())

*q****ϕ***(**x***t**|***x***t**−*1)

 =

*T** −*1 X

*t*=1 E*q****ϕ***(**x***t**−*1*,***x***t*+1*|***x**0)

 E*q****ϕ***(**x***t**|***x**0)

 log* *^(*p*)^(()^(**x**)^(*t*)^(*|*)^(**x**)^(*t*)^(+1)^())

*q****ϕ***(**x***t**|***x***t**−*1)

=* −*

*T** −*1 X

*t*=1 E*q****ϕ***(**x***t**−*1*,***x***t*+1*|***x**0)

"

DKL (*q****ϕ***(**x***t**|***x***t**−*1)*∥**p*(**x***t**|***x***t*+1))

#

*.*

By replacing* p*(**x**0*|***x**1) with* p****θ***(**x**0*|***x**1) and* p*(**x***t**|***x***t*+1) with* p****θ***(**x***t**|***x***t*+1), we are done.

**Rewrite the Consistency Term**. The nightmare of the above variational diffusion model is that we need to draw samples (**x***t**−*1*,*** x***t*+1) from a joint distribution* q****ϕ***(**x***t**−*1*,*** x***t*+1*|***x**0). We don’t know what *q****ϕ***(**x***t**−*1*,*** x***t*+1*|***x**0) is! It is a Gaussian by our choice, but we still need to use future samples** x***t*+1 to draw the current sample** x***t*. This is odd. Inspecting the consistency term, we notice that* q****ϕ***(**x***t**|***x***t**−*1) and* p****θ***(**x***t**|***x***t*+1) are moving along two opposite directions. Thus, it is unavoidable that we need to use** x***t**−*1 and** x***t*+1. The question we need to ask is: Can we come up with something so that we do not need to handle two opposite directions while we are able to check consistency? So, here is the simple trick called Bayes theorem which will give us

*q*(**x***t**|***x***t**−*1) =* *^(*q*)^(()^(**x**)^(*t*)^(*−*)^(1)^(*|*)^(**x**)^(*t*)^())^(*q*)^(()^(**x**)^(*t*)^())

*q*(**x***t**−*1)

condition on** x**0 =*⇒ **q*(**x***t**|***x***t**−*1*,*** x**0) =* *^(*q*)^(()^(**x**)^(*t*)^(*−*)^(1)^(*|*)^(**x**)^(*t*)^(*,*)^(** x**)^(0)^())^(*q*)^(()^(**x**)^(*t*)^(*|*)^(**x**)^(0)^())

*q*(**x***t**−*1*|***x**0) *. *(2.20)

With this change of the conditioning order, we can switch* q*(**x***t**|***x***t**−*1*,*** x**0) to* q*(**x***t**−*1*|***x***t**,*** x**0) by adding one more condition variable** x**0. (If you do not condition on** x**0, there is no way that we can draw samples from *q*(**x***t**−*1), for example, because the specific state of** x***t**−*1 depends on the initial image** x**0.) The direction *q*(**x***t**−*1*|***x***t**,*** x**0) is now parallel to* p****θ***(**x***t**−*1*|***x***t*) as shown in Figure 2.8. So, if we want to rewrite the consistency term, a natural option is to calculate the KL divergence between* q****ϕ***(**x***t**−*1*|***x***t**,*** x**0) and* p****θ***(**x***t**−*1*|***x***t*).

Figure 2.8: If we consider the Bayes theorem in Eqn (2.20), we can define a distribution *q****ϕ***(**x***t**−*1*|***x***t**,*** x**0) that has a direction parallel to* p****θ***(**x***t**−*1*|***x***t*).

If we manage to go through a few (boring) algebraic derivations, we can show that the ELBO is now:

**Theorem 2.4. (ELBO for Variational Diffusion Model)**. Let** x** =** x**0, and** x***T** ∼N*(0*,*** I**). The ELBO for a variational diffusion model in Theorem 2.3 can be equivalently written as

ELBO***ϕ****,****θ***(**x**) = E*q****ϕ***(**x**1*|***x**0)[log* p****θ***(**x**0*|***x**1) | {z } same as before

]* −*DKL  *q****ϕ***(**x***T** |***x**0)*∥**p*(**x***T* ) 

| {z } new prior matching

*−*

*T *X

*t*=2 E*q****ϕ***(**x***t**|***x**0) h DKL  *q****ϕ***(**x***t**−*1*|***x***t**,*** x**0)*∥**p****θ***(**x***t**−*1*|***x***t*) 

| {z } new consistency

i *. *(2.21)

Let’s quickly make three interpretations:

© 2024 Stanley Chan. All Rights Reserved. 28


---

## Page 29

•** Reconstruction**. The new reconstruction term is the same as before. We are still maximizing the log-likelihood. •** Prior Matching**. The new prior matching is simplified to the KL divergence between* q****ϕ***(**x***T** |***x**0) and *p*(**x***T* ). The change is due to the fact that we now condition upon** x**0. Thus, there is no need to draw samples from* q****ϕ***(**x***T** −*1*|***x**0) and take expectation. •** Consistency**. The new consistency term is different from the previous one in two ways. Firstly, the running index* t* starts at* t* = 2 and ends at* t* =* T*. Previously it was from* t* = 1 to* t* = *T** −*1. Accompanied by this is the distribution matching, which is now between* q****ϕ***(**x***t**−*1*|***x***t**,*** x**0) and *p****θ***(**x***t**−*1*|***x***t*). So, instead of asking a forward transition to match with a reverse transition, we use* q****ϕ ***to construct a reverse transition and use it to match with* p****θ***.

**Proof of Theorem 2.4**. We begin with Eqn (2.17) by showing that

log* p*(**x**)* ≥*E*q****ϕ***(**x**1:*T** |***x**0)

 log *p*(**x**0:*T* ) *q****ϕ***(**x**1:*T** |***x**0)

(By Eqn (2.17))

= E*q****ϕ***(**x**1:*T** |***x**0)

"

log* *^(*p*)^(()^(**x**)^(*T*)^( ))^(*p*)^(()^(**x**)^(0)^(*|*)^(**x**)^(1)^())^( Q)^(*T *)*t*=2* *^(*p*)^(()^(**x**)^(*t*)^(*−*)^(1)^(*|*)^(**x**)^(*t*)^())

*q****ϕ***(**x**1*|***x**0) ^(Q)^(*T *)*t*=2* *^(*q*)^(***ϕ***)^(()^(**x**)^(*t*)^(*|*)^(**x**)^(*t*)^(*−*)^(1)^(*,*)^(** x**)^(0)^())

#

(split the chain)

= E*q****ϕ***(**x**1:*T** |***x**0)

 log* *^(*p*)^(()^(**x**)^(*T*)^( ))^(*p*)^(()^(**x**)^(0)^(*|*)^(**x**)^(1)^())

*q****ϕ***(**x**1*|***x**0)

 + E*q****ϕ***(**x**1:*T** |***x**0)

"

log

*T *Y

*t*=2

*p*(**x***t**−*1*|***x***t*) *q****ϕ***(**x***t**|***x***t**−*1*,*** x**0)

#

(2.22)

Let’s consider the second term:

*T *Y

*t*=2

*p*(**x***t**−*1*|***x***t*) *q****ϕ***(**x***t**|***x***t**−*1*,*** x**0) ^(=)

*T *Y

*t*=2

*p*(**x***t**−*1*|***x***t*)

*q****ϕ***(**x***t**−*1*|***x***t**,***x**0)*q****ϕ***(**x***t**|***x**0)

*q****ϕ***(**x***t**−*1*|***x**0)

(Bayes rule, Eqn (2.20))

=

*T *Y

*t*=2

*p*(**x***t**−*1*|***x***t*) *q****ϕ***(**x***t**−*1*|***x***t**,*** x**0)* *^(*×*)

*T *Y

*t*=2

*q****ϕ***(**x***t**−*1*|***x**0)

*q****ϕ***(**x***t**|***x**0) ( Rearrange denominator)

=

*T *Y

*t*=2

*p*(**x***t**−*1*|***x***t*) *q****ϕ***(**x***t**−*1*|***x***t**,*** x**0)* *^(*×*)^(* q*)^(***ϕ***)^(()^(**x**)^(1)^(*|*)^(**x**)^(0)^())

*q****ϕ***(**x***T** |***x**0)^(*, *)( Recursion cancels terms)

where the last equation uses the fact that for any sequence* a*1*, . . . , a**T* , we have ^(Q)^(*T *)*t*=2 *a**t**−*1

*a**t *=* *^(*a*)^(1)

*a*2* *^(*×*)^(* a*)^(2)

*a*3* *^(*× *)*. . .** ×** *^(*a*)^(*T*)^(* −*)^(1)

*a**T *=* *^(*a*)^(1)

*a**T* ^(. Going back to the Eqn (2.22), we can see that)

E*q****ϕ***(**x**1:*T** |***x**0)

 log* *^(*p*)^(()^(**x**)^(*T*)^( ))^(*p*)^(()^(**x**)^(0)^(*|*)^(**x**)^(1)^())

*q****ϕ***(**x**1*|***x**0)

 + E*q****ϕ***(**x**1:*T** |***x**0)

"

log

*T *Y

*t*=2

*p*(**x***t**−*1*|***x***t*) *q****ϕ***(**x***t**|***x***t**−*1*,*** x**0)

#

= E*q****ϕ***(**x**1:*T** |***x**0)

 log* *^(*p*)^(()^(**x**)^(*T*)^( ))^(*p*)^(()^(**x**)^(0)^(*|*)^(**x**)^(1)^())

*q****ϕ***(**x**1*|***x**0) + log* *^(*q*)^(***ϕ***)^(()^(**x**)^(1)^(*|*)^(**x**)^(0)^())

*q****ϕ***(**x***T** |***x**0)

 + E*q****ϕ***(**x**1:*T** |***x**0)

"

log

*T *Y

*t*=2

*p*(**x***t**−*1*|***x***t*) *q****ϕ***(**x***t**−*1*|***x***t**,*** x**0)

#

= E*q****ϕ***(**x**1:*T** |***x**0)

 log* *^(*p*)^(()^(**x**)^(*T*)^( ))^(*p*)^(()^(**x**)^(0)^(*|*)^(**x**)^(1)^())

*q****ϕ***(**x***T** |***x**0)

 + E*q****ϕ***(**x**1:*T** |***x**0)

"

log

*T *Y

*t*=2

*p*(**x***t**−*1*|***x***t*) *q****ϕ***(**x***t**−*1*|***x***t**,*** x**0)

#

*,*

where we canceled* q****ϕ***(**x**1*|***x**0) in the numerator and denominator since log* *^(*a*)

*b* ^(+ log)^(* b*)

*c* ^(= log)^(* a*)

*c* ^(for any )positive constants* a*,* b*, and* c*. This will give us

E*q****ϕ***(**x**1:*T** |***x**0)

 log* *^(*p*)^(()^(**x**)^(*T*)^( ))^(*p*)^(()^(**x**)^(0)^(*|*)^(**x**)^(1)^())

*q****ϕ***(**x***T** |***x**0)

 = E*q****ϕ***(**x**1:*T** |***x**0) [log* p*(**x**0*|***x**1)] + E*q****ϕ***(**x**1:*T** |***x**0)

 log *p*(**x***T* ) *q****ϕ***(**x***T** |***x**0)

= E*q****ϕ***(**x**1*|***x**0) [log* p*(**x**0*|***x**1)] | {z } reconstruction

*−*DKL(*q****ϕ***(**x***T** |***x**0)*∥**p*(**x***T* )) | {z } prior matching

*.*

© 2024 Stanley Chan. All Rights Reserved. 29


---

## Page 30

The last term is

E*q****ϕ***(**x**1:*T** |***x**0)

"

log

*T *Y

*t*=2

*p*(**x***t**−*1*|***x***t*) *q****ϕ***(**x***t**−*1*|***x***t**,*** x**0)

#

= X

*t*=2 E*q****ϕ***(**x***t**,***x***t**−*1*|***x**0) log *p*(**x***t**−*1*|***x***t*) *q****ϕ***(**x***t**−*1*|***x***t**,*** x**0)

= X

*t*=2

ZZ log *p*(**x***t**−*1*|***x***t*) *q****ϕ***(**x***t**−*1*|***x***t**,*** x**0)* *^(*·*)^(* q*)^(***ϕ***)^(()^(**x**)^(*t*)^(*,*)^(** x**)^(*t*)^(*−*)^(1)^(*|*)^(**x**)^(0)^())^(*d*)^(**x**)^(*t*)^(*−*)^(1)^(*d*)^(**x**)^(*t*)

= X

*t*=2

ZZ log *p*(**x***t**−*1*|***x***t*) *q****ϕ***(**x***t**−*1*|***x***t**,*** x**0)* *^(*·*)^(* q*)^(***ϕ***)^(()^(**x**)^(*t*)^(*−*)^(1)^(*|*)^(**x**)^(*t*)^(*,*)^(** x**)^(0)^())^(*q*)^(***ϕ***)^(()^(**x**)^(*t*)^(*|*)^(**x**)^(0)^())^(*d*)^(**x**)^(*t*)^(*−*)^(1)^(*d*)^(**x**)^(*t*)

= X

*t*=2

Z Z log *p*(**x***t**−*1*|***x***t*) *q****ϕ***(**x***t**−*1*|***x***t**,*** x**0)* *^(*·*)^(* q*)^(***ϕ***)^(()^(**x**)^(*t*)^(*−*)^(1)^(*,*)^(** x**)^(*t*)^(*|*)^(**x**)^(0)^())^(*d*)^(**x**)^(*t*)^(*−*)^(1)

 *q****ϕ***(**x***t**|***x**0)*d***x***t*

=* − *X

*t*=2 E*q****ϕ***(**x***t**|***x**0)DKL(*q****ϕ***(**x***t**−*1*|***x***t**,*** x**0)*∥**p*(**x***t**−*1*|***x***t*))

| {z } consistency

*.*

Finally, replace* p*(**x***t**−*1*|***x***t*) by* p****θ***(**x***t**−*1*|***x***t*), and* p*(**x**0*|***x**1) by* p****θ***(**x**0*|***x**1). Done!

**2.3 Distribution of the Reverse Process**

Now that we know the new ELBO for the variational diffusion model, we should spend some time discussing its core component which is* q****ϕ***(**x***t**−*1*|***x***t**,*** x**0). In a nutshell, what we want to show is that

•* q****ϕ***(**x***t**−*1*|***x***t**,*** x**0) is still a Gaussian. • Since it is a Gaussian, it is fully characterized by the mean and covariance. It turns out that

*q****ϕ***(**x***t**−*1*|***x***t**,*** x**0) =* N*(**x***t**−*1* | ♡***x***t* +* ♠***x**0*,** ♣***I**)*, *(2.23)

for some magical scalars* ♡*,* ♠*and* ♣*defined below.

**Theorem 2.5.** The distribution* q****ϕ***(**x***t**−*1*|***x***t**,*** x**0) takes the form of

*q****ϕ***(**x***t**−*1*|***x***t**,*** x**0) =* N*(**x***t**−*1* |**** µ****q*(**x***t**,*** x**0)*,*** Σ***q*(*t*))*, *(2.24)

where

***µ****q*(**x***t**,*** x**0) = ^((1)^(* −*)^(*α*)^(*t*)^(*−*)^(1)^())^(*√*)^(*α*)^(*t*)

1* −**α**t ***x***t* + ^((1)^(* −*)^(*α*)^(*t*)^())^(*√*)^(*α*)^(*t*)^(*−*)^(1)

1* −**α**t ***x**0 (2.25)

**Σ***q*(*t*) = ^((1)^(* −*)^(*α*)^(*t*)^()(1)^(* −√*)^(*α*)^(*t*)^(*−*)^(1)^())

1* −**α**t ***I **def =* σ*^(2 )*q*^(()^(*t*)^())^(**I**)^(*, *)(2.26)

where* α**t* = ^(Q)^(*t *)*i*=1* *^(*α*)^(*i*)^(.)

Eqn (2.25) reveals an interesting fact that the mean*** µ****q*(**x***t**,*** x**0) is a linear combination of** x***t* and** x**0. Geometrically,*** µ****q*(**x***t**,*** x**0) lives on the straight line connecting** x***t* and** x**0, as illustrated in Figure 2.9.

Figure 2.9: According to Eqn (2.25), the mean*** µ****q*(**x***t**,*** x**0) is a linear combination of** x***t* and **x**0.

© 2024 Stanley Chan. All Rights Reserved. 30


---

## Page 31

**Proof of Theorem 2.5**. Using the Bayes theorem stated in Eqn (2.20),* q*(**x***t**−*1*|***x***t**,*** x**0) can be deter- mined if we evaluate the following product of Gaussians

*q*(**x***t**−*1*|***x***t**,*** x**0) =* *^(*N*)^(()^(**x**)^(*t*)^(*|√*)^(*α*)^(*t*)^(**x**)^(*t*)^(*−*)^(1)^(*,*)^( (1)^(* −*)^(*α*)^(*t*)^())^(**I**)^())^(*N*)^(()^(**x**)^(*t*)^(*−*)^(1)^(*|√*)^(*α*)^(*t*)^(*−*)^(1)^(**x**)^(0)^(*,*)^( (1)^(* −*)^(*α*)^(*t*)^(*−*)^(1)^(**I**)^()))

*N*(**x***t**|*^(*√*)*α**t***x**0*,* (1* −**α**t*)**I**) *. *(2.27)

For simplicity we will treat the vectors are scalars. Then the above product of Gaussians will become

*q*(**x***t**−*1*|***x***t**,*** x**0)* ∝*exp (**x***t** −√**α**t***x***t**−*1)2

2(1* −**α**t*) + ^(()^(**x**)^(*t*)^(*−*)^(1)^(* −√*)^(*α*)^(*t*)^(*−*)^(1)^(**x**)^(0)^())^(2)

2(1* −**α**t**−*1) *−*^(()^(**x**)^(*t*)^(* −√*)^(*α*)^(*t*)^(**x**)^(0)^())^(2)

2(1* −**α**t*)

 *. *(2.28)

We consider the following mapping:

*x* =** x***t**, a* =* α**t **y* =** x***t**−*1*, b* =* α**t**−*1 *z* =** x**0*, c* =* α**t**.*

Consider a quadratic function

*f*(*y*) = ^(()^(*x*)^(* −√*)^(*ay*)^())^(2)

2(1* −**a*) + ^(()^(*y*)^(* − *)*√*

*bz*)^(2)

2(1* −**b*) *−*^(()^(*x*)^(* −√*)^(*cz*)^())^(2)

2(1* −**c*) *. *(2.29)

We know that no matter how we rearrange the terms, the resulting function remains a quadratic equation. The minimizer of* f*(*y*) is the mean of the resulting Gaussian. So, we can calculate the derivative of* f* and show that

*f** *^(*′*)(*y*) = 1* −**ab *(1* −**a*)(1* −**b*)^(*y*)^(* −*)

 *√**a*

1* −**a*^(*x*)^( +)

*√*

*b *1* −**b*^(*z*)

!

*.*

Setting* f** *^(*′*)(*y*) = 0 yields

*y* = ^((1)^(* −*)^(*b*)^())^(*√*)^(*a*)

1* −**ab x* + ^((1)^(* −*)^(*a*)^() )*√*

*b *1* −**ab z. *(2.30)

We note that* ab* =* α**t**α**t**−*1 =* α**t*. So,

***µ****q*(**x***t**,*** x**0) = ^((1)^(* −*)^(*α*)^(*t*)^(*−*)^(1)^())^(*√*)^(*α*)^(*t*)

1* −**α**t ***x***t* + ^((1)^(* −*)^(*α*)^(*t*)^())^(*√*)^(*α*)^(*t*)^(*−*)^(1)

1* −**α**t ***x**0*. *(2.31)

Similarly, for the variance, we can check the curvature* f** *^(*′′*)(*y*). We can easily show that

*f** *^(*′′*)(*y*) = 1* −**ab *(1* −**a*)(1* −**b*) ^(= )1* −**α**t *(1* −**α**t*)(1* −**α**t**−*1)^(*.*)

Taking the reciprocal will give us

**Σ***q*(*t*) = ^((1)^(* −*)^(*α*)^(*t*)^()(1)^(* −*)^(*α*)^(*t*)^(*−*)^(1)^())

1* −**α**t ***I***. *(2.32)

In combination weight in the above theorem deserves some study. Recall that

***µ****q*(**x***t**,*** x**0) = ^((1)^(* −*)^(*α*)^(*t*)^(*−*)^(1)^())^(*√*)^(*α*)^(*t*)

1* −**α**t ***x***t* + ^((1)^(* −*)^(*α*)^(*t*)^())^(*√*)^(*α*)^(*t*)^(*−*)^(1)

1* −**α**t ***x**0*.*

One question we can ask is how does the two coefficients behave as* t* goes from* T* to 1? We show an example in Figure 2.10. For this particular example, we use* α**t* = 0*.*9 for all* t*. We plot the coefficients as a function of* t*. Figure 2.10 suggests that the coefficient for** x***t* shrinks as* t* decreases from* t* =* T* to* t* = 1, whereas the coefficient for** x**0 grows as* t* decreases.

© 2024 Stanley Chan. All Rights Reserved. 31


---

## Page 32

Figure 2.10: The trajectory of the coefficients for** x***t* and for** x**0, as* t* grows.

As* t* goes from* T* to 1, the variance* σ*^(2 )*q*^(()^(*t*)^() will also change. Figure 2.11 shows the trajectory of)^(** x**)^(*t*) ^(as )a function of* t* by sampling** x***t* according to* q****ϕ***(**x***t**−*1*|***x***t**,*** x**0). On the same plot, we show the radius of the Gaussian, defined by* σ*^(2 )*q*^(()^(*t*)^(). Our plot indicates that when)^(* t*)^( =)^(* T*)^(, the variance)^(* σ*)^(2 )*q*^(()^(*t*)^() is fairly large so that )**x***t* is closer to white Gaussian noise. As* t* drops to* t* = 1, the variance* σ*^(2 )*q*^(()^(*t*)^() also drops to zero. This makes )sense because eventually we want** x**0 to be the clean image which is noise free.

Figure 2.11: The trajectory of** x***t* and the associated radius of the Gaussian* σ*^(2 )*q*^(()^(*t*)^().)

**Constructing*** p****θ***(**x***t**−*1*|***x***t*). The interesting part of Eqn (2.24) is that* q****ϕ***(**x***t**−*1*|***x***t**,*** x**0) is* completely characterized* by** x***t* and** x**0. There is no neural network required to estimate the mean and variance! (You can compare this with VAE where a network is needed.) Since a network is not needed, there is really nothing to “learn”. The distribution* q****ϕ***(**x***t**−*1*|***x***t**,*** x**0) is automatically determined if we know** x***t* and** x**0. The realization here is important. Let’s look at the consistency term in Eqn (2.21):

ELBO***ϕ****,****θ***(**x**) = E*q****ϕ***(**x**1*|***x**0)[log* p****θ***(**x**0*|***x**1) | {z } same as before

]* −*DKL  *q****ϕ***(**x***T** |***x**0)*∥**p*(**x***T* ) 

| {z } new prior matching

*−*

*T *X

*t*=2 E*q****ϕ***(**x***t**|***x**0) h DKL  *q****ϕ***(**x***t**−*1*|***x***t**,*** x**0)*∥**p****θ***(**x***t**−*1*|***x***t*) 

| {z } new consistency

i *, *(from Eqn (2.21))

where we just showed that

*q****ϕ***(**x***t**−*1*|***x***t**,*** x**0) =* N*(**x***t**−*1* |**** µ****q*(**x***t**,*** x**0)*,*** Σ***q*(*t*))*.*

There is no “learning” for* q****ϕ***(**x***t**−*1*|***x***t**,*** x**0) because it is defined once the hyperparameter* α**t* are defined. Therefore, the consistency term is a summation of many KL divergence terms where the* t*-th term is

DKL(* q****ϕ***(**x***t**−*1*|***x***t**,*** x**0) | {z } nothing to learn

*∥ **p****θ***(**x***t**−*1*|***x***t*) | {z } need to do something

)*. *(2.33)

© 2024 Stanley Chan. All Rights Reserved. 32


---

## Page 33

So, to compute the KL divergence, we need to do something about* p****θ***(**x***t**−*1*|***x***t*). The big idea here is that* q****ϕ***(**x***t**−*1*|***x***t**,*** x**0) is Gaussian. If we want to quickly calculate the KL divergence, then it would be good if* p****θ***(**x***t**−*1*|***x***t*) is also a Gaussian. So, we* choose** p****θ***(**x***t**−*1*|***x***t*) to be a Gaussian. Moreover, we should match the form of the mean and variance! Therefore, we define

*p****θ***(**x***t**−*1*|***x***t*) =* N * **x***t**−*1*| ****µ******θ***(**x***t*) | {z } neural network

*, σ*^(2 )*q*^(()^(*t*)^())^(**I **) *, *(2.34)

where we assume that the mean vector can be determined using a neural network. As for the variance, we *choose* the variance to be* σ*^(2 )*q*^(()^(*t*)^(). This is)^(* identical*)^( to Eqn (2.26)! Thus, if we put Eqn (2.24) side by side with )*p****θ***(**x***t**−*1*|***x***t*), we notice a parallel relation between the two:

*q****ϕ***(**x***t**−*1*|***x***t**,*** x**0) =* N * **x***t**−*1* |**** µ****q*(**x***t**,*** x**0) | {z } known

*, σ*^(2 )*q*^(()^(*t*)^())^(**I **)| {z } known

 *, *(2.35)

*p****θ***(**x***t**−*1*|***x***t*) =* N * **x***t**−*1*| ****µ******θ***(**x***t*) | {z } neural network

*, σ*^(2 )*q*^(()^(*t*)^())^(**I **)| {z } known

 *. *(2.36)

Therefore, the KL divergence is simplified to

DKL  *q****ϕ***(**x***t**−*1*|***x***t**,*** x**0)* ∥**p****θ***(**x***t**−*1*|***x***t*) 

= DKL  *N*(**x***t**−*1* |**** µ****q*(**x***t**,*** x**0)*, σ*^(2 )*q*^(()^(*t*)^())^(**I**)^())^(* ∥N*)^(()^(**x**)^(*t*)^(*−*)^(1)* *^(*|*)^(*** µ***)***θ***^(()^(**x**)^(*t*)^())^(*, σ*)^(2 )*q*^(()^(*t*)^())^(**I**)^() )

= 1 2*σ*^(2)*q*(*t*)^(*∥*)^(***µ***)^(*q*)^(()^(**x**)^(*t*)^(*,*)^(** x**)^(0)^())^(* −*)^(***µ***)^(***θ***)^(()^(**x**)^(*t*)^())^(*∥*)^(2)^(*, *)(2.37)

where we used the fact that the KL divergence between two identical-variance Gaussians is just the Euclidean distance square between the two mean vectors. Substituting Eqn (2.37) to the definition of ELBO in Eqn (2.21), we can rewrite ELBO as follows.

**Theorem 2.6.** The** ELBO for a variational diffusion model** in Eqn (2.21) can be simplified to

ELBO***θ***(**x**) = E*q*(**x**1*|***x**0)[log* p****θ***(**x**0*|***x**1)]* −*DKL  *q*(**x***T** |***x**0)*∥**p*(**x***T* ) 

| {z } nothing to train

*−*

*T *X

*t*=2 E*q*(**x***t**|***x**0) h 1 2*σ*^(2)*q*(*t*)^(*∥*)^(***µ***)^(*q*)^(()^(**x**)^(*t*)^(*,*)^(** x**)^(0)^())^(* −*)^(***µ***)^(***θ***)^(()^(**x**)^(*t*)^())^(*∥*)^(2)^(i )*, *(2.38)

where** x** =** x**0, and** x***T** ∼N*(0*,*** I**).

One remark for Theorem 2.6 is that the subscript*** ϕ*** is dropped because the distribution* q****ϕ*** defined in Theorem 2.5 is fully characterized by** x***t* and** x**0. There is nothing to learn, and so the optimization does not need to include*** ϕ***. Because of this, we can drop the KL-divergence term in Eqn (2.38). This leaves us the

reconstruction term E*q*(**x**1*|***x**0)[log* p****θ***(**x**0*|***x**1)] and transition term ^(P)^(*T *)*t*=2 ^(E)^(*q*)^(()^(**x**)*t*^(*|*)^(**x**)0^() )h 1 2*σ*^(2)*q*(*t*)^(*∥*)^(***µ***)^(*q*)^(()^(**x**)^(*t*)^(*,*)^(** x**)^(0)^())^(*−*)^(***µ***)^(***θ***)^(()^(**x**)^(*t*)^())^(*∥*)^(2)^(i ). The reconstruction term can be simplified, since

log* p****θ***(**x**0*|***x**1) = log* N*(**x**0*|****µ******θ***(**x**1)*, σ*^(2 )*q*^((1))^(**I**)^())

= log 1

( q

2*πσ*^(2)*q*(1))^(*d*)^( exp ) *−*^(*∥*)^(**x**)^(0)^(* −*)^(***µ***)^(***θ***)^(()^(**x**)^(1)^())^(*∥*)^(2)

2*σ*^(2)*q*(1)

=* −*^(*∥*)^(**x**)^(0)^(* −*)^(***µ***)^(***θ***)^(()^(**x**)^(1)^())^(*∥*)^(2)

2*σ*^(2)*q*(1) *−*^(*d*)

2 ^(log )� 2*πσ*^(2 )*q*^((1) ) *.*

So, as soon as we know** x**1, we can send it to a network*** µ******θ***(**x**1) to return us a mean estimate. The mean estimate will then be used to compute the likelihood.

© 2024 Stanley Chan. All Rights Reserved. 33


---

## Page 34

**2.4 Training and Inference**

In this section we discuss how to train a variational diffusion model, and turn into a* denoising* diffusion probabilistic model. We start by looking at the ELBO defined in Theorem 2.6. Eqn (2.38) suggests that we need to find a network*** µ******θ*** that can somehow minimize the loss:

1 2*σ*^(2)*q*(*t*)^(*∥*)^(***µ***)^(*q*)^(()^(**x**)^(*t*)^(*,*)^(** x**)^(0)^() )| {z } known

*−****µ******θ***(**x***t*) | {z } network

*∥*^(2)*. *(2.39)

Recall from Eqn (2.25) that

***µ****q*(**x***t**,*** x**0) = ^((1)^(* −*)^(*α*)^(*t*)^(*−*)^(1)^())^(*√*)^(*α*)^(*t*)

1* −**α**t ***x***t* + ^((1)^(* −*)^(*α*)^(*t*)^())^(*√*)^(*α*)^(*t*)^(*−*)^(1)

1* −**α**t ***x**0*. *(2.40)

We see that it is a function of** x***t* and** x**0. Therefore,*** µ****q*(**x***t**,*** x**0) is known and determined once we know** x***t *and** x**0. The subject of interest is*** µ******θ***. Since*** µ******θ*** is our* design*, there is no reason why we cannot define it as something more convenient. So here is an option. We define

***µ******θ***(**x***t*) | {z } a network

def = ^((1)^(* −*)^(*α*)^(*t*)^(*−*)^(1)^())^(*√*)^(*α*)^(*t*)

1* −**α**t ***x***t* + ^((1)^(* −*)^(*α*)^(*t*)^())^(*√*)^(*α*)^(*t*)^(*−*)^(1)

1* −**α**t *b**x*****θ***(**x***t*) | {z } another network

*. *(2.41)

Substituting Eqn (2.40) and Eqn (2.41) into Eqn (2.39) will give us

1 2*σ*^(2)*q*(*t*)^(*∥*)^(***µ***)^(*q*)^(()^(**x**)^(*t*)^(*,*)^(** x**)^(0)^())^(* −*)^(***µ***)^(***θ***)^(()^(**x**)^(*t*)^())^(*∥*)^(2)^( = )1 2*σ*^(2)*q*(*t*)

 (1* −**α**t*)^(*√*)*α**t**−*1

1* −**α**t *(b**x*****θ***(**x***t*)* −***x**0) 

2

= 1 2*σ*^(2)*q*(*t*) (1* −**α**t*)^(2)*α**t**−*1

(1* −**α**t*)^(2 )*∥*b**x*****θ***(**x***t*)* −***x**0*∥*^(2)^(* *)*. *(2.42)

Therefore, ELBO can be written as

ELBO***θ***(**x**) = E*q*(**x**1*|***x**0)[log* p****θ***(**x**0*|***x**1)]* −*

*T *X

*t*=2 E*q*(**x***t**|***x**0) h 1 2*σ*^(2)*q*(*t*) (1* −**α**t*)^(2)*α**t**−*1

(1* −**α**t*)^(2 )*∥*b**x*****θ***(**x***t*)* −***x**0*∥*^(2)^( i )*, *(2.43)

where we dropped the term DKL  *q*(**x***T** |***x**0)*∥**p*(**x***T* )  .

Next, we want to simplify ELBO so that we can absorb E*q*(**x**1*|***x**0)[log* p****θ***(**x**0*|***x**1)] into the summation. The following is the result.

**Theorem 2.7.** The ELBO for** denoising diffusion probabilistic model** is

ELBO***θ***(**x**) =* −*

*T *X

*t*=1

1 2*σ*^(2)*q*(*t*) (1* −**α**t*)^(2)*α**t**−*1

(1* −**α**t*)^(2 )E*q*(**x***t**|***x**0) h *∥*b**x*****θ***(**x***t*)* −***x**0*∥*^(2)^( i )*. *(2.44)

**Proof**. Substituting Eqn (2.42) into Eqn (2.38), we can see that

ELBO***θ***(**x**) = E*q*(**x**1*|***x**0)[log* p****θ***(**x**0*|***x**1)]* −*

*T *X

*t*=2 E*q*(**x***t**|***x**0) h 1 2*σ*^(2)*q*(*t*)^(*∥*)^(***µ***)^(*q*)^(()^(**x**)^(*t*)^(*,*)^(** x**)^(0)^())^(* −*)^(***µ***)^(***θ***)^(()^(**x**)^(*t*)^())^(*∥*)^(2)^(i)

= E*q*(**x**1*|***x**0)[log* p****θ***(**x**0*|***x**1)]* −*

*T *X

*t*=2 E*q*(**x***t**|***x**0) h 1 2*σ*^(2)*q*(*t*) (1* −**α**t*)^(2)*α**t**−*1

(1* −**α**t*)^(2 )*∥*b**x*****θ***(**x***t*)* −***x**0*∥*^(2)^( i )*. *(2.45)

© 2024 Stanley Chan. All Rights Reserved. 34


---

## Page 35

The first term is

log* p****θ***(**x**0*|***x**1) = log* N*(**x**0*|****µ******θ***(**x**1)*, σ*^(2 )*q*^((1))^(**I**)^())^(* ∝− *)1 2*σ*^(2)*q*(1)^(*∥*)^(***µ***)^(***θ***)^(()^(**x**)^(1)^())^(* −*)^(**x**)^(0)^(*∥*)^(2 )(definition)

=* − *1 2*σ*^(2)*q*(1)

 (1* −**α*0)^(*√*)*α*1

1* −**α*1 **x**1 + ^((1)^(* −*)^(*α*)^(1)^())^(*√*)^(*α*)^(0)

1* −**α*1 b**x*****θ***(**x**1)* −***x**0



2

(recall* α*0 = 1)

=* − *1 2*σ*^(2)*q*(1)

 (1* −**α*1)

1* −**α*1 b**x*****θ***(**x**1)* −***x**0



2

=* − *1 2*σ*^(2)*q*(1)* *^(*∥*)^(b)^(**x**)^(***θ***)^(()^(**x**)^(1)^())^(* −*)^(**x**)^(0)^(*∥*)^(2)^(* . *)(recall* α*1 =* α*1) (2.46)

Substituting Eqn (2.46) into Eqn (2.45) will simplify ELBO as

ELBO***θ***(**x**) =* −*

*T *X

*t*=1 E*q*(**x***t**|***x**0) h 1 2*σ*^(2)*q*(*t*) (1* −**α**t*)^(2)*α**t**−*1

(1* −**α**t*)^(2 )*∥*b**x*****θ***(**x***t*)* −***x**0*∥*^(2)^( i )*.*

The loss function defined in Eqn (2.44) is very intuitive. Ignoring the constants and expectations, the main subject of interest, for a particular** x***t*, is

argmin ***θ ****∥*b**x*****θ***(**x***t*)* −***x**0*∥*^(2)^(* *)*.*

This is nothing but a denoising problem because we need to find a network b**x*****θ*** such that the denoised image b**x*****θ***(**x***t*) will be close to the ground truth** x**0. What makes it not a typical denoiser is the following reasons:

• E*q*(**x***t**|***x**0): We are not trying to denoise any random noisy image. Instead, we are carefully choosing the noisy image to be

**x***t** ∼**q*(**x***t**|***x**0) =* N*(**x***t** | *^(*√*)*α**t***x**0*,* (1* −**α**t*)**I**)

*⇔ ***x***t* =* *^(*√*)*α**t***x**0 + p

(1* −**α**t*)***ϵ****t**, *where ***ϵ****t** ∼N*(0*,*** I**)*.*

• 1 2*σ*^(2)*q*(*t*) (1*−**α**t*)^(2)*α**t**−*1

(1*−**α**t*)^(2 ): We do not weight the denoising loss equally for all steps. Instead, there is a scheduler to control the relative emphasis on each denoising loss. Considering this, and using Monte Carlo to approximate the expectation, we can write the optimization problem as

argmax ***θ***

X

**x**0*∈X *ELBO(**x**0)

= argmin ***θ***

X

**x**0*∈X*

*T *X

*t*=1 E*q*(**x***t**|***x**0)

 1 2*σ*^(2)*q*(*t*) (1* −**α**t*)^(2)*α**t**−*1

(1* −**α**t*)^(2)

b**x*****θ ****√**α**t***x**0 + p

(1* −**α**t*)***ϵ****t * *−***x**0  2^()

= argmin ***θ***

X

**x**0*∈X*

*T *X

*t*=1

1 *M*

*M *X

*m*=1

1 2*σ*^(2)*q*(*t*) (1* −**α**t*)^(2)*α**t**−*1

(1* −**α**t*)^(2)

b**x*****θ ****√**α**t***x**0 + p

(1* −**α**t*)***ϵ***^(()^(*m*)^() )*t * *−***x**0  2 *, *(2.47)

where*** ϵ***^(()^(*m*)^() )*t **∼N*(0*,*** I**), and the summation over** x**0* ∈X* means we consider all samples in the training set* X*. As you can see, the training of this model involves training a* denoiser* b**x*****θ***(*·*). For this reason, the resulting model is known as the* denoising* diffusion probabilistic model (DDPM).

**Forward Diffusion in DDPM**. The training of a DDPM involves two parallel branches. The first branch is the forward diffusion. The goal of forward diffusion is to generate the intermediate variables **x**1*, . . . ,*** x***T** −*1 by using

**x***t** ∼**q*(**x***t**|***x**0) =* N*(**x***t** | *^(*√*)*α**t***x**0*,* (1* −**α**t*)**I**)*, t* = 1*, . . . , T** −*1*.*

© 2024 Stanley Chan. All Rights Reserved. 35


---

## Page 36

Figure 2.12: Forward diffusion process.

The forward diffusion does not require any training. If you give us the clean image** x**0, we can run the forward diffusion and prepare the images** x**1*, . . . ,*** x***T* . A pictorial illustration is shown in Figure 2.12.

**Training DDPM**. Once the training samples** x**0*, . . . ,*** x***T* are prepared, we can train the DDPM. The training of DDPM is summarized by the optimization in Eqn (2.47) and Figure 2.13. The goal is to train* one *denoiser for* all* noise levels. It is a one denoiser for all noise levels because each** x***t* has a different variance 1* −**α**t*. We are not interested in training many denoisers because it is computationally not feasible.

Figure 2.13: Training of a denoising diffusion probabilistic model. For the same neural network b**x*****θ***, we send noisy inputs** x***t* to the network. The gradient of the loss is back-propagated to update the network. Note that the noisy images are not arbitrary. They are generated according to the forward sampling process.

The training of a denoiser is no different than any conventional supervised learning. Given a pair of clean and noisy image, which in our case is** x**0 and** x***t*, we train the denoiser b**x*****θ***(*·*). The training loss in Eqn (2.47) has three summations. If we run stochastic gradient descent, we can simplify the above optimization into the following procedure.

**Training Algorithm for DDPM**. For every image** x**0 in your training dataset:

• Repeat the following steps until convergence. • Pick a random time stamp* t** ∼*Uniform[1*, T*]. • Draw a sample** x**^(()^(*m*)^() )*t **∼N*(**x***t** | *^(*√*)*α**t***x**0*,* (1* −**α**t*)**I**), i.e.,

**x**^(()^(*m*)^() )*t *=* α**t***x**0 + p

(1* −**α**t*)***ϵ***^(()^(*m*)^() )*t **, ****ϵ***^(()^(*m*)^() )*t **∼N*(0*,*** I**)*.*

© 2024 Stanley Chan. All Rights Reserved. 36


---

## Page 37

• Take gradient descent step on

*∇****θ***

( 1 *M*

*M *X

*m*=1

b**x*****θ***(**x**(*m*) *t *)* −***x**0  2 )

*.*

You can do this in batches, just like how you train any other neural networks. Note that, here, you are training** one** denoising network b**x*****θ*** for** all** noisy conditions.

**Inference of DDPM – the Reverse Diffusion**. Once the denoiser b**x*****θ*** is trained, we can apply it to do the inference. The inference is about sampling images from the distributions* p****θ***(**x***t**−*1*|***x***t*) over the sequence of states** x***T** ,*** x***T** −*1*, . . . ,*** x**1. Since it is the reverse diffusion process, we need to do it recursively via:

**x***t**−*1* ∼**p****θ***(**x***t**−*1* |*** x***t*) =* N*(**x***t**−*1* |**** µ******θ***(**x***t*)*, σ*^(2 )*q*^(()^(*t*)^())^(**I**)^())^(*.*)

By reparameterization, we have

**x***t**−*1 =*** µ******θ***(**x***t*) +* σ**q*(*t*)***ϵ****, *where ***ϵ**** ∼N*(0*,*** I**)

= ^((1)^(* −*)^(*α*)^(*t*)^(*−*)^(1)^())^(*√*)^(*α*)^(*t*)

1* −**α**t ***x***t* + ^((1)^(* −*)^(*α*)^(*t*)^())^(*√*)^(*α*)^(*t*)^(*−*)^(1)

1* −**α**t *b**x*****θ***(**x***t*) +* σ**q*(*t*)***ϵ****.*

This leads to the following inferencing algorithm. In plain words, we sequentially run the denoiser* T* times from the white noise vector** x***T* back to the generated image b**x**0. A pictorial illustration is shown in Figure 2.14.

**Inference of DDPM**.

• You give us a white noise vector** x***T** ∼N*(0*,*** I**). • Repeat the following for* t* =* T, T** −*1*, . . . ,* 1. • We calculate b**x*****θ***(**x***t*) using our trained denoiser. • Update according to

**x***t**−*1 = ^((1)^(* −*)^(*α*)^(*t*)^(*−*)^(1)^())^(*√*)^(*α*)^(*t*)

1* −**α**t ***x***t* + ^((1)^(* −*)^(*α*)^(*t*)^())^(*√*)^(*α*)^(*t*)^(*−*)^(1)

1* −**α**t *b**x*****θ***(**x***t*) +* σ**q*(*t*)***ϵ****, ****ϵ**** ∼N*(0*,*** I**)*. *(2.48)

Figure 2.14: Inference of a denoising diffusion probabilistic model.

**2.5 Predicting Noise**

The final step of our derivation is to connect our results back to the original paper of Ho et al [16] so that the notations will be more consistent with the literature. **Training**. If you are familiar with the denoising literature, you probably know the residue-type of algorithm that predicts the noise instead of the signal. The same spirit applies denoising diffusion, where

© 2024 Stanley Chan. All Rights Reserved. 37


---

## Page 38

we can learn to predict the noise. To see why this is the case, we consider Eqn (2.11). If we re-arrange the terms we will obtain

**x***t* =* *^(*√*)*α**t***x**0 + *√*

1* −**α**t****ϵ***0

*⇒ ***x**0 =** **^(**x**)^(*t*)^(* −√*)^(1)^(* −*)^(*α*)^(*t*)^(***ϵ***)^(0 )*√**α**t **.*

Substituting this into*** µ****q*(**x***t**,*** x**0), we can show that

***µ****q*(**x***t**,*** x**0) = *√**α**t*(1* −**α**t**−*1)**x***t* +* √**α**t**−*1(1* −**α**t*)**x**0

1* −**α**t*

=

*√**α**t*(1* −**α**t**−*1)**x***t* +* √**α**t**−*1(1* −**α**t*)* ·*** x***t**−*^(*√*)1*−**α**t****ϵ***0 *√**α**t *1* −**α**t *= a few more algebraic steps which we shall skip

= 1 *√**α**t ***x***t** − *1* −**α**t **√*1* −**α**t **√**α**t ****ϵ***0*. *(2.49)

In words, we have converted*** µ****q*(**x***t**,*** x**0) from a function of** x**0 to a function of*** ϵ***0. Since we do this modification, naturally we should modify the mean estimator*** µ******θ***. In order to match the form of*** µ******θ*** with that of*** µ****q*(**x***t**,*** x**0), we choose

***µ******θ***(**x***t*) = 1 *√**α**t ***x***t** − *1* −**α**t **√*1* −**α**t **√**α**t *b***ϵ******θ***(**x***t*)*. *(2.50)

Substituting Eqn (2.49) and Eqn (2.50) into Eqn (2.39) will give us a new ELBO

ELBO***θ***(**x**0*,**** ϵ***0) =* −*

*T *X

*t*=1 E*q*(**x***t**|***x**0) h 1 2*σ*^(2)*q*(*t*) (1* −**α**t*)^(2)

(1* −**α**t*)*α**t **∥*b***ϵ******θ***(**x***t*)* −****ϵ***0*∥*^(2)^( i)

=* −*

*T *X

*t*=1 E*q*(**x***t**|***x**0) h 1 2*σ*^(2)*q*(*t*) (1* −**α**t*)^(2)

(1* −**α**t*)*α**t*

b***ϵ******θ ****√**α**t***x**0 + *√*

1* −**α**t****ϵ***0  *−****ϵ***0  2 ^(i)

We remark that this ELBO is a function of** x**0 and*** ϵ***0. Therefore, to train the denoiser, we need to solve the optimization

argmin ***θ ***E**x**0*,****ϵ***0ELBO***θ***(**x**0*,**** ϵ***0)

*≈*argmin ***θ***

X

**x**0*∼X*

1 *M*

*M *X

*m*=1 ELBO***θ***(**x**0*,**** ϵ***^(()^(*m*)^() )0 )*,*

where the superscript denotes the* m*-th initial noise term. If we run stochastic gradient descent, we can simplify the above description to the following procedure.

**Training DDPM using** b***ϵ******θ***(**x***t*). For every image** x**0 in your training dataset:

• Repeat the following steps until convergence. • Pick a random time stamp* t** ∼*Uniform[1*, T*]. • Draw a sample*** ϵ***0* ∼N*(0*,*** I**). • Draw a sample** x***t** ∼N*(**x***t** | *^(*√*)*α**t***x**0*,* (1* −**α**t*)**I**), i.e.,

**x***t* =* *^(*√*)*α**t***x**0 + p

(1* −**α**t*)***ϵ***0*.*

• Take gradient descent step on *∇****θ**** ∥*b***ϵ******θ***(**x***t*)* −****ϵ***0*∥*^(2)^(* *)*.*

© 2024 Stanley Chan. All Rights Reserved. 38


---

## Page 39

**Inference**. For the inference, we know that

**x***t**−*1* ∼**p****θ***(**x***t**−*1* |*** x***t*) =* N*(**x***t**−*1* |**** µ******θ***(**x***t*)*, σ**q*(*t*)^(2)**I**)*.*

Using reparameterization, we can show that

**x***t**−*1 =*** µ******θ***(**x***t*) +* σ**q*(*t*)**z***, *where **z*** ∼N*(0*,*** I**)

=  1 *√**α**t ***x***t** − *1* −**α**t **√*1* −**α**t **√**α**t *b***ϵ******θ***(**x***t*)  +* σ**q*(*t*)**z**

= 1 *√**α**t*

 **x***t** −*^(1)^(* −*)^(*α*)^(*t *)*√*1* −**α**t *b***ϵ******θ***(**x***t*)  +* σ**q*(*t*)**z***. *(2.51)

Summarizing it here, we have

**Inference of DDPM using** b***ϵ******θ***(**x***t*).

• You give us a white noise vector** x***T** ∼N*(0*,*** I**). • Repeat the following for* t* =* T, T** −*1*, . . . ,* 1. • We calculate b**x*****θ***(**x***t*) using our trained denoiser. • Update according to

**x***t**−*1 = 1 *√**α**t*

 **x***t** −*^(1)^(* −*)^(*α*)^(*t *)*√*1* −**α**t *b***ϵ******θ***(**x***t*)  +* σ**q*(*t*)**z***, ***z*** ∼N*(0*,*** I**)*. *(2.52)

**2.6 Denoising Diffusion Implicit Model (DDIM)**

**From DDPM to DDIM**. One of the most prevalent drawbacks of DDPM is that they need a large number of iterations to generate a reasonably good looking image. As mentioned by Song et al in [39], a DDPM would take more than 1000 hours to generate 50k images (of size 256*×*256) on a standard GPU. The reason is that when running the reverse diffusion steps, we need to perform denoising. If the reverse diffusion process intrinsically requires many steps to converge, then it will take us many denoising steps. Therefore, to speed up the computing, it is necessary to reduce the number of iterations. DDIM was an invention to overcome this difficulty. Recall from Eqn (2.1) that the original DDPM transition probability takes the form

*q*(**x***t**|***x***t**−*1) def =* N * **x***t** | *^(*√*)*α**t***x***t**−*1*,* (1* −**α**t*)**I ** *.*

In addition, Eqn (2.8) shows that the probability of** x***t* given** x**0 is

*q*(**x***t**|***x**0) =* N * **x***t** | *^(*√*)*α**t***x**0*,* (1* −**α**t*)**I ** *,*

where* α**t* = ^(Q)^(*t *)*i*=0* *^(*α*)^(*i*)^( for any decreasing sequence 0)^(* < α*)^(*t*)^(* ≤*)^(1. )One important observation here is that the transition probability* q*(**x***t**|***x***t**−*1) follows a* Markov chain*, meaning that the probability of** x***t* is purely dependent on** x***t**−*1 but not the previous states** x***t**−*2 and so on. The advantage of a Markovian structure is that the system is memoryless. Once we know** x***t**−*1, we will know** x***t*. But the downside is that a Markov chain can take many steps to converge. DDIM overcomes this issue by departing from the Markovian structure to non-Markovian. **Probability Distributions in DDIM**. To start our discussion, let’s follow [39] by picking a special choice of parameters where we replace* α**t* by a ratio* α**t**/α**t**−*1. This means

*q*(**x***t**|***x***t**−*1) def =* N * **x***t * r* α**t*

*α**t**−*1 **x***t**−*1*,* (1* − **α**t **α**t**−*1 )**I ** *.*

© 2024 Stanley Chan. All Rights Reserved. 39


---

## Page 40

There is no particularly strong physical meaning for this choice, except that it makes the notation simpler. With this choice, the product term is simplified to* α**t* = ^(Q)^(*t *)*i*=1 *α**i **α**i**−*1 ^(=)^(* α*)^(*t*)^( assuming that)^(* α*)^(0)^( = 1. Therefore,)

*q*(**x***t**|***x**0) =* N * **x***t** | *^(*√*)*α**t***x**0*,* (1* −**α**t*)**I ** *. *(2.53)

Expressing Eqn (2.53) by means of reparametrization, we note that** x***t* in Eqn (2.53) can be represented in terms of** x**0 as follows

**x***t* =* *^(*√*)*α**t***x**0 + *√*

1* −**α**t****ϵ****, *where ***ϵ**** ∼N*(0*,*** I**)*.*

By the same argument, we can write

**x***t**−*1 =* *^(*√*)*α**t**−*1**x**0 + p

1* −**α**t**−*1***ϵ****, *where ***ϵ**** ∼N*(0*,*** I**)*. *(2.54)

So here comes an interesting trick. Let’s replace*** ϵ*** by something so that** x***t**−*1 is no longer** x**0 perturbed by white noise. Perhaps we can consider the following derivation

**x***t* =* *^(*√*)*α**t***x**0 + *√*

1* −**α**t****ϵ***

=*⇒ √*

1* −**α**t****ϵ*** =** x***t** −*^(*√*)*α**t***x**0

=*⇒ ****ϵ*** =** **^(**x**)^(*t*)^(* −√*)^(*α*)^(*t*)^(**x**)^(0 )*√*1* −**α**t **.*

So, substituting*** ϵ*** into Eqn (2.54), we obtain

**x***t**−*1 =* *^(*√*)*α**t**−*1**x**0 + p

1* −**α**t**−*1***ϵ***

=* *^(*√*)*α**t**−*1**x**0 + p

1* −**α**t**−*1

**x***t** −√**α**t***x**0 *√*1* −**α**t*

*. *(2.55)

The difference between Eqn (2.55) and Eqn (2.54) is that in Eqn (2.54), the noise term is*** ϵ*** which is* N*(0*,*** I**). It is this Gaussian that makes the derivations of DDPM easy, but it is also this Gaussian that makes the reverse diffusion slow. In contrast, Eqn (2.55) replaces the Gaussian by an estimate. This estimate uses the previous signal** x***t* combined with the initial signal** x**0. Of course, one can argue that in DDPM (e.g., Eqn (2.24)) also uses a combination of** x***t* and** x**0. The difference is that the combination in Eqn (2.55) allows us to do something that Eqn (2.24) does not, which is the derivation of the marginal distribution* q*(**x***t**−*1*|***x**0) and make it to a desired form. Let’s elaborate more on the marginal distribution. Referring to Eqn (2.55), we notice that we can* choose*

*q*(**x***t**−*1*|***x***t**,*** x**0) =* N **√**α**t**−*1**x**0 + p

1* −**α**t**−*1

**x***t** −√**α**t***x**0 *√*1* −**α**t*

 *, *something  *.*

where “something” stands for the variance of the Gaussian which can be made as* σ*^(2 )*t*** **^(**I**)^( for some hyperpa- )rameter* σ**t*. One important (likely the most important) argument in DDIM is that we want the marginal distribution* q*(**x***t**−*1*|***x**0) to have the same form as* q*(**x***t**|***x**0):

*q*(**x***t**−*1*|***x**0) =* N*(^(*√*)*α**t**−*1**x**0*,* (1* −**α**t**−*1)**I**)*.*

The reason of aiming for this distribution is that ultimately we care about the marginal distribution* q*(**x***t**|***x**0) which we want it to become pure white noise when* t* =* T* and it is the original image when* t* = 0. Therefore, while we can have millions of different choice of the transitional distribution* q*(**x***t**−*1*|***x***t**,*** x**0), only some very specialized transition probabilities can ensure that* q*(**x***t**−*1*|***x**0) takes a form we like. **Derivation of the Transition Distribution**. With this goal in mind, we now state our mathematical problem. Suppose that

*q*(**x***t**|***x**0) =* N **√**α**t***x**0*,* (1* −**α**t*)**I ** *,*

*q*(**x***t**−*1*|***x***t**,*** x**0) =* N **√**α**t**−*1**x**0 + p

1* −**α**t**−*1

**x***t** −√**α**t***x**0 *√*1* −**α**t*

 *, σ*^(2 )*t*** **^(**I **) *, *(2.56)

can we ensure that* q*(**x***t**−*1*|***x**0) =* N*(^(*√*)*α**t**−*1**x**0*,* (1* −**α**t**−*1)**I**)? If not, what additional changes do we need? The answer to this mathematical question requires some tools from textbooks. We recall the following result from Bishop’s textbook [4].

© 2024 Stanley Chan. All Rights Reserved. 40


---

## Page 41

**Theorem 2.8. Bishop [4, Eqn 2.115]** Suppose that we have two random variables** x** and** y** following the distributions

*p*(**x**) =* N*(***µ****,*** Λ**^(*−*)^(1))*,*

*p*(**y***|***x**) =* N*(**Ax** +** b***,*** L**^(*−*)^(1))*.*

Then we can show that the marginal distribution is

*p*(**y**) = Z *p*(**y***|***x**)*p*(**x**)*d***x** =* N * **A*****µ*** +** b***, ***L**^(*−*)^(1)^( )+** AΛ**^(*−*)^(1)**A**^(*−*)^(1)^( )*.*

Let’s see how we can apply this result to our problem. Looking at Eqn (2.56), we can identify the following qualities:

**A** = r

1* −**α**t**−*1

1* −**α**t **, ****µ*** =* *^(*√*)*α**t***x**0*, ***b** =* *^(*√*)*α**t**−*1**x**0* − *r

1* −**α**t**−*1

1* −**α**t*

*√**α**t***x**0*.*

Suppose that* q*(**x***t**−*1*|***x**0) =* N*(***µ****t**−*1*, σ*^(2 )*t**−*1^(**I**)^() for some unknown choices of mean)^(*** µ***)*t**−*1 ^(and variance)^(* σ*)^(2 )*t**−*1^(. )If we can show that*** µ****t**−*1 =* *^(*√*)*α**t**−*1**x**0 and* σ*^(2 )*t**−*1 ^(= (1)^(* −*)^(*α*)^(*t*)^(*−*)^(1)^(), then we are done. To this end, we show that)

***µ****t**−*1 =** A*****µ*** +** b**

= r

1* −**α**t**−*1

1* −**α**t **· *^(*√*)*α**t***x**0 +* *^(*√*)*α**t**−*1**x**0* − *r

1* −**α**t**−*1

1* −**α**t*

*√**α**t***x**0

=* *^(*√*)*α**t**−*1**x**0*.*

Oh, great news! We have shown that*** µ****t**−*1 =* *^(*√*)*α**t**−*1**x**0. That means for the transitional distribution *q*(**x***t**−*1*|***x***t**,*** x**0) we have chosen, the marginal distribution has the desired mean. So it remains to check the variance. We show that

*σ*^(2 )*t**−*1 ^(=)^(** L**)^(*−*)^(1)^( +)^(** AΛ**)^(*−*)^(1)^(**A**)^(*T*)

=* σ*^(2 )*t* ^(+ )r

1* −**α**t**−*1

1* −**α**t **·* (1* −**α**t*)* · *r

1* −**α**t**−*1

1* −**α**t *=* σ*^(2 )*t* ^(+ (1)^(* −*)^(*α*)^(*t*)^(*−*)^(1)^())^(*.*)

Oh, no! We cannot show that* σ*^(2 )*t**−*1 ^(= 1)^(* −*)^(*α*)^(*t*)^(*−*)^(1)^(. There is an additional term)^(* σ*)^(2 )*t* ^(here. But this is not such a )big deal. How about we do a quick fix by* adding** σ*^(2 )*t* ^(into)^(** A**)^(:)

*σ*^(2 )*t**−*1 ^(=)^(* σ*)^(2 )*t* ^(+)

s

1* −**α**t**−*1* −**σ*^(2 )*t *1* −**α**t **·* (1* −**α**t*)* ·*

s

1* −**α**t**−*1* −**σ*^(2 )*t *1* −**α**t*

=* σ*^(2 )*t* ^(+ 1)^(* −*)^(*α*)^(*t*)^(*−*)^(1)* *^(*−*)^(*σ*)^(2 )*t *= 1* −**α**t**−*1*.*

Aha!* σ*^(2 )*t**−*1 ^(now takes the desired form such that)^(* σ*)^(2 )*t**−*1 ^(= 1)^(* −*)^(*α*)^(*t*)^(*−*)^(1)^(. Let’s do a quick check to make sure this )additional* σ*^(2 )*t* ^(does not affect the mean:)

***µ****t**−*1 =** A*****µ*** +** b**

=

s

1* −**α**t**−*1* −**σ*^(2 )*t *1* −**α**t **· *^(*√*)*α**t***x**0 +* *^(*√*)*α**t**−*1**x**0* −*

s

1* −**α**t**−*1* −**σ*^(2 )*t *1* −**α**t*

*√**α**t***x**0

=* *^(*√*)*α**t**−*1**x**0*.*

So, the mean remains the desired form despite we changed the variance term. To summarize, we can* choose** q*(**x***t**−*1*|***x***t**,*** x**0) to be the following.

© 2024 Stanley Chan. All Rights Reserved. 41


---

## Page 42

**Theorem 2.9. DDIM Transition Distribution**. In DDIM, the transition distribution is defined as

*q*(**x***t**−*1*|***x***t**,*** x**0) =* N **√**α**t**−*1**x**0 + q

1* −**α**t**−*1* −**σ*^(2 )*t*

**x***t** −√**α**t***x**0 *√*1* −**α**t*

 *, σ*^(2 )*t*** **^(**I **) *. *(2.57)

If* q*(**x***t**|***x**0) =* N **√**α**t***x**0*,* (1*−**α**t*)**I ** , then it follows from our derivation that* q*(**x***t**−*1*|***x**0) =* N*(^(*√*)*α**t**−*1**x**0*,* (1*−*

*α**t**−*1)**I**). **Inference for DDIM**. The inference for DDIM is derived based on the transition distribution. Starting with the forward process, if we want to perform the reverse, we will need to find out** x**0 from the following equation:

**x***t *|{z} given

= *√**α**t***x**0 | {z } want to find

+ *√*

1* −**α**t****ϵ ***| {z } estimated by network

*.*

By rearranging the terms, we see that

**x**0 = 1 *√**α**t*

� **x***t** − √*

1* −**α**t****ϵ ***

=*⇒ **f* ^(()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^())

def = 1 *√**α**t*

 **x***t** − √*

1* −**α**t****ϵ***^(()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^())

 *.*

There are two new terms in this equation. The first one is*** ϵ***^(()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^() which replaces)^(*** ϵ***)^(. It is the estimate )of the noise based on the current input** x***t*. The second term is* f* ^(()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^() which is defined as the equation)

1 *√**α**t*

 **x***t** −*^(*√*)1* −**α**t****ϵ***^(()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^() ) . We can think of it as the estimate of the true signal** x**0.

Going back to the transition distribution, we recall that it is denoted by* q*(**x***t**−*1*|***x***t**,*** x**0). If we do not have access to** x**0, we can replace it* f* ^(()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^(). This means that)

*p****θ***(**x***t**−*1*|***x***t*) =* q*(**x***t**−*1*|***x***t**,*** x**0)

=*⇒ **p****θ***(**x***t**−*1*|***x***t*) def =* q*(**x***t**−*1*|***x***t**, f* ^(()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^()))

=* N*

 *√**α**t**−*1*f* (*t*) ***θ*** ^(()^(**x**)^(*t*)^() + )q

1* −**α**t**−*1* −**σ*^(2 )*t** *^(*·*)^(** x**)^(*t*)^(* −√*)^(*α*)^(*t*)^(*f*)^( ()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^() )*√*1* −**α**t **, σ*^(2 )*t*** **^(**I**)

!

=* N*

 *√**α**t**−*1* · *1 *√**α**t*

 **x***t** − √*

1* −**α**t****ϵ***^(()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^() )

+

+ q

1* −**α**t**−*1* −**σ*^(2 )*t** *^(*· *)**x***t** −*^(*√*)*α**t*

 1 *√**α**t*

 **x***t** −*^(*√*)1* −**α**t****ϵ***^(()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^() )

*√*1* −**α**t **, σ*^(2 )*t*** **^(**I**)

!

=* N*

 *√**α**t**−*1

 **x***t** −*^(*√*)1* −**α**t****ϵ***^(()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^() )*√**α**t*

!

+ q

1* −**α**t**−*1* −**σ*^(2 )*t** *^(*·*)^(*** ϵ***)^(()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^())^(*, *)*σ*^(2 )*t*** **^(**I**)

!

*. *(2.58)

For the special case where* t* = 1, we define* p****θ***(**x***t**−*1*|***x***t*) =* N*(*f* ^((1) )***θ*** ^(()^(**x**)^(1)^())^(*, σ*)^(2 )1^(**I**)^() so that the reverse process is )supported everywhere. Looking at Eqn (2.58), we use reparametrization to write it as follows and interpret the equation according to [39].

(DDIM) **x***t**−*1 =* *^(*√*)*α**t**−*1

 **x***t** −*^(*√*)1* −**α**t****ϵ***^(()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^() )*√**α**t*

!

| {z } predicted** x**0

+ q

1* −**α**t**−*1* −**σ*^(2 )*t** *^(*·*)^(*** ϵ***)^(()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^() )| {z } direction pointing to** x***t*

+* σ**t ****ϵ****t *|{z} *∼N*(0*,***I**)

*. *(2.59)

It would be helpful to compare this equation with the DDPM equation in Eqn (2.52):

(DDPM) **x***t**−*1 = 1 *√**α**t*

 **x***t** −*^(1)^(* −*)^(*α*)^(*t *)*√*1* −**α**t ****ϵ***^(()^(*t*)^() )***θ*** ^(()^(**x**)^(*t*)^() ) +* σ**t****ϵ****t**, ****ϵ****t** ∼N*(0*,*** I**)*. *(2.60)

© 2024 Stanley Chan. All Rights Reserved. 42


---

## Page 43

The main difference between DDPM and DDIM is subtle. While they both use** x***t* and*** ϵ***^(()^(*t*)^())(**x***t*) in their updates, the specific update formula makes a different convergence speed. In fact, later in the differential equation literature where people connect DDIM and DDPM with stochastic differential equations, it was observed that DDIM employed some special accelerated first-order numerical schemes when solving the differential equation.

**2.7 Concluding Remark**

The literature of DDPM is quickly exploding. The original paper by Sohl-Dickstein et al. [38] and Ho et al. [16] are the must-reads to understand the topic. For a more “user-friendly” version, we found that the tutorial by Luo very useful [27]. Some follow up works are highly cited, including the denoising diffusion implicit models by Song et al. [39]. In terms of application, people have been using DDPM for various image synthesis applications, e.g., [34, 35].

© 2024 Stanley Chan. All Rights Reserved. 43


---

## Page 44

### **3 Score-Matching Langevin Dynamics (SMLD)**

Score-based generative models [42] are alternative approaches to generate data from a desired distribution. There are several core ingredients: the Langevin equation, the (Stein) score function, and the score-matching loss. The focus of this section is more on the latter two. We will make a few hand-waving arguments about the Langevin equation and explain how to make it computationally feasible for our generative task. More in-depth discussions about the Langevin equation will be postponed to the next section.

**3.1 Sampling from a Distribution**

Imagine that we are given a distribution* p*(**x**) and we want to draw samples from* p*(**x**). If** x** is a one- dimensional variable, we can achieve the goal easily by inverting the cumulative distribution function (CDF) [7, Chapter 4] and sending a uniform[0*,* 1] random variable through this inverted CDF. For high dimensional distributions, the same CDF technique does not apply. However, the intuition of giving a higher weight to samples with a higher probability remains valid. For example, if we want to draw a sample from the orange dataset, it is almost certain that we want a spherical orange than a cubical orange.

Figure 3.1: If our goal is to generate a fruit image, then we should expect the underlying distribution* p*(**x**) will have a higher value for those “normal-looking” fruit images than those “weird-looking” images. Therefore, to sample from a distribution, it is more natural to pick a sample from a higher position of the distribution. The fruit image is taken from https: //cognitiveseo.com/blog/4224/unnatural-links-definition-examples/

Therefore, in a non-rigorous way, we can argue that if we are given* p*(**x**), we should aim to draw samples from a location where* p*(**x**) has a high value. This idea of searching for a higher probability can be translated into an optimization **x**^(*∗*)= argmax **x **log* p*(**x**)*,*

where the goal is to maximize the log-likelihood of the distribution* p*(**x**). Certainly, this maximization does not provide any clue on how to draw a low probability sample which we will explain through the lens of Langevin equation. For now, we want to make a remark about the difference between the maximization here and maximum likelihood estimation. In maximum likelihood, the data point** x** is fixed but the model parameters are changing. Here, the model parameters are fixed but the data point is changing. We are given a* fixed* model. Our goal is to draw the most likely sample from* this* model. The table below shows the difference between our sampling problem and maximum likelihood estimation.

Problem Sampling Maximum Likelihood

Optimization target A sample** x **Model parameter*** θ***

Formulation **x**^(*∗*)= argmax **x **log* p*(**x**;*** θ***) ***θ***^(*∗*)= argmax ***θ ***log* p*(**x**;*** θ***)

Let’s continue our argument about the maximization. If* p*(**x**) is a simple parametric model, the max- imization will have analytic solutions. However, in general, optimizations in a high-dimensional space are ill-posed with many local minima. Therefore, there is no single algorithm that is globally converging in all situations. A reasonable trade-off between computational complexity, memory requirement, difficulty of

© 2024 Stanley Chan. All Rights Reserved. 44


---

## Page 45

implementation, and solution quality is the gradient descent algorithm which is a first-order method. For our optimization where the objective function is log* p*(**x**), the gradient descent algorithm defines a sequence of iterates via the update rule

**x***t*+1 =** x***t* +* τ**∇***x** log* p*(**x***t*)*,*

where* ∇***x** log* p*(**x***t*) denotes the gradient of log* p*(**x**) evaluated at** x***t*, and* τ* is the step size. Here we use “+” instead of the typical “*−*” because we are solving a maximization problem. If you agree with the above approach, we can now provide an information introduction about the Langevin equation. Without worrying too much about its roots in physics, we can treat Langevin equation as an iterative procedure that allows us to draw samples.

**Definition 3.1.** The (discrete-time)** Langevin equation** for sampling from a known distribution* p*(**x**) is an iterative procedure for* t* = 1*, . . . , T*:

**x***t*+1 =** x***t* +* τ**∇***x** log* p*(**x***t*) + *√*

2*τ***z***, ***z*** ∼N*(0*,*** I**)*, *(3.1)

where* τ* is the step size which users can control, and** x**0 is white noise.

**Example 3.1.** Consider a Gaussian distribution* p*(*x*) =* N*(*x** |** µ, σ*^(2)), we can show that the Langevin equation is

*x**t*+1 =* x**t* +* τ** · ∇**x* log  1 *√*

2*πσ*^(2)^(* e*)^(*−*)^(()^(*xt*)^(*−*)^(*µ*)^()2)

2*σ*^(2 ) + *√*

2*τz*

=* x**t** −**τ** ·** *^(*x*)^(*t*)^(* −*)^(*µ*)

*σ*^(2 )+ *√*

2*τz, z** ∼N*(0*,* 1)*,*

where the initial state can be set as* x*0* ∼N*(0*,* 1).

If we ignore the noise term *√*

2*τ***z**, the Langevin equation in Eqn (3.1) is exactly** gradient descent**, but for a particular function — the log likelihood of the random variable** x**. Therefore, while gradient descent is a generic first-order optimization algorithm for any objective function, the Langevin equation focuses on the distribution if we use it in the context of generative models. The gradient descent algorithm plus a small noise perturbation gives us a simple summary.

In generative models,

Langevin equation =** gradient descent + noise**

applied to the log-likelihood function.

But why do we want gradient descent + noise instead of gradient descent? One interpretation is that we are not interested in solving the optimization problem. Instead, we are more interested in* sampling* from a distribution. By introducing the random noise to the gradient descent step, we randomly pick a sample that is following the objective function’s trajectory while not staying at where it is. If we are closer to the peak, we will move left and right slightly. If we are far from the peak, the gradient direction will pull us towards the peak. If the curvature around the peak is sharp, we will concentrate most of the steady state points** x***T *there. If the curvature around the peak is flat, we will spread around. Therefore, by repeatedly initializing the gradient descent (plus noise) algorithm at a uniformly distributed location, we will eventually collect samples that will follow the distribution we designate. A slightly more formal way to justify the Langevin equation is the Fokker-Planck equation. The Fokker- Planck equation is a fundamental result of stochastic processes. For any Markovian processes (e.g., Wiener process and Brownian motion), the dynamics of the solution** x***t* is described by a stochastic differential equa- tion (i.e., Langevin equation). However, since** x***t* is a random variable at any time* t*, there is an underlying probability distribution* p*(**x***, t*) associated with each** x***t*. Fokker-Planck equation provides a mathematical statement about the distribution. The distribution must satisfy a partial differential equation. Roughly speaking, in the context of our problem, the Fokker-Planck equation can be described below.

© 2024 Stanley Chan. All Rights Reserved. 45


---

## Page 46

**Theorem 3.1.** In the context of our problem, the solution** x***t* of the Langevin equation will have a probability distribution* p*(**x***, t*) at time* t* satisfying the** Fokker-Planck Equation**

*∂**t**p*(**x***, t*) =* −**∂***x **n [*∂***x**(log* p*(**x**))]*p*(**x***, t*) o +* ∂*^(2 )**x**^(*p*)^(()^(**x**)^(*, t*)^())^(*, *)(3.2)

where log* p*(**x**) is the log-likelihood of the ground truth distribution.

Deriving the Fokker-Planck equation will take a tremendous amount of effort. However, if we are given a candidate solution, verifying whether it satisfies the Fokker-Planck equation is not hard.

**Verification of Theorem**. Suppose that we have run Langevin equation for long enough that we have reached a converging solution** x***t* as* t** →∞*. We argue that this limiting distribution is* p*(**x**). Indeed, we can show that

*∂***x **n log* p*(**x**) o =* *^(*∂*)^(**x**)^(*p*)^(()^(**x**)^())

*p*(**x**)* *^(*.*)

Then, substituting* p*(**x***, t*) by* p*(**x**) when* t** →∞*, we can show that

*−**∂***x **n [*∂***x**(log* p*(**x**))]*p*(**x**) o +* ∂*^(2 )**x**^(*p*)^(()^(**x**)^() =)^(* ∂*)^(**x **)n [*−**∂***x**(log* p*(**x**))]*p*(**x**) +* ∂***x***p*(**x**) o

=* ∂***x **n *−*^(*∂*)^(**x**)^(*p*)^(()^(**x**)^())

*p*(**x**)* *^(*p*)^(()^(**x**)^() +)^(* ∂*)^(**x**)^(*p*)^(()^(**x**)^() )o

=* ∂***x **n *−**∂***x***p*(**x**) +* ∂***x***p*(**x**) o = 0*.*

On the other hand, when* t** →∞*, it holds that* ∂**t**p*(**x**) = 0. Therefore, the Fokker-Planck equation is verified.

**Example 3.2.** Consider a Gaussian mixture* p*(*x*) =* π*1*N*(*x** |** µ*1*, σ*^(2 )1^()+)^(*π*)^(2)^(*N*)^(()^(*x*)^(* |*)^(* µ*)^(2)^(*, σ*)^(2 )2^(). We can calculate )the gradient* ∇**x* log* p*(*x*) analytically or numerically. For demonstration, we choose* π*1 = 0*.*6.* µ*1 = 2, *σ*1 = 0*.*5,* π*2 = 0*.*4,* µ*2 =* −*2,* σ*2 = 0*.*2. We initialize* x*0 = 0. We choose* τ* = 0*.*05. We run the above gradient descent iteration for* T* = 500 times, and we plot the trajectory of the values* p*(*x**t*) for *t* = 1*, . . . , T*. As we can see in the figure below, the sequence* {**x*1*, x*2*, . . . , x**T** }* simply follows the shape of the Gaussian and climb to one of the peaks. What is more interesting is when we add the noise term. Instead of landing at the peak, the sequence* x**t* moves around the peak and finishes somewhere near the peak. (Remark: To terminate the algorithm, we can gradually make* τ* smaller or we can early stop.)

**x***t*+1 =** x***t* +* τ**∇***x** log* p*(**x***t*) **x***t*+1 =** x***t* +* τ**∇***x** log* p*(**x***t*) + *√*

2*τ***z**

Figure 3.2: Deterministic algorithm aiming to pick a sample that maximizes the likelihood, versus a stochastic algorithm which adds noise at every iteration.

Figure 3.3 shows an interesting description of the sample trajectory. Starting with an arbitrary location, the data point** x***t* will do a random walk according to the Langevin dynamics equation. The direction of the

© 2024 Stanley Chan. All Rights Reserved. 46


---

## Page 47

random walk is not completely arbitrary. There is a certain amount of pre-defined drift while at every step there is some level of randomness. The drift is determined by* ∇***x** log* p*(**x**) whereas the randomness comes from** z**.

Figure 3.3: Trajectory of sample evolutions using the Langevin dynamics. We colored the two modes of the Gaussian mixture in different colors for better visualization. The setting here is identical to the example above, except that the step size is* τ* = 0*.*001.

**Example 3.3.** Following the previous example we again consider a Gaussian mixture

*p*(*x*) =* π*1*N*(*x** |** µ*1*, σ*^(2 )1^() +)^(* π*)^(2)^(*N*)^(()^(*x*)^(* |*)^(* µ*)^(2)^(*, σ*)^(2 )2^())^(*.*)

We choose* π*1 = 0*.*6.* µ*1 = 2,* σ*1 = 0*.*5,* π*2 = 0*.*4,* µ*2 =* −*2,* σ*2 = 0*.*2. Suppose we initialize* M* = 10000 uniformly distributed samples* x*0* ∼*Uniform[*−*3*,* 3]. We run Langevin updates for* t* = 100 steps. The histograms of generated samples are shown in the figures below.

Figure 3.4: Samples generated by Langevin dynamics. Initially the samples are uniformly dis- tributed. As time progresses, the distribution of the samples become the desired distribution.

**Remark 1: Stochastic Gradient Langevin Dynamics**. The dynamical behavior of** x***t* governed by the Langevin equation is often known as the Langevin dynamics. Langevin dynamics uses gradient descent plus noise. This is not the same as stochastic gradient descent (SGD). SGD uses minibatches to approximate the full gradient. There is no noise. The randomness in SGD comes from the minibatch which are uniformly sampled from the training dataset. SGD can be paired with Langevin dynamics to make stochastic gradient Langevin dynamics as outlined in [46] which provided a clear comparison in the context of classical maximum-a-posteriori (MAP) estimation.

Stochast Gradient Descent ∆***θ****t* =* *^(*ϵ*)^(*t*)

2

 

*∇****θ*** log* p*(***θ****t*) +* *^(*N*)

*n*

*n *X

*i*=1 *∇*log* p*(**y***t**i**|****θ****t*)

!

Langevin Dynamics ∆***θ****t* =* *^(*ϵ*)

2

 

*∇****θ*** log* p*(***θ****t*) +

*N *X

*i*=1 *∇*log* p*(**y***i**|****θ****t*)

!

+*** η****t*

SG Langevin Dynamics ∆***θ****t* =* *^(*ϵ*)^(*t*)

2

 

*∇****θ*** log* p*(***θ****t*) +* *^(*N*)

*n*

*n *X

*i*=1 *∇*log* p*(**y***t**i**|****θ****t*)

!

+*** η****t**,*

© 2024 Stanley Chan. All Rights Reserved. 47


---

## Page 48

where* ϵ**t* is a sequence of step sizes satisfying the properties that ^(P)^(*∞ *)*t*=1* *^(*ϵ*)^(*t*)^( =)^(* ∞*)^(and)^( P)^(*∞ *)*t*=1* *^(*ϵ*)^(2 )*t** *^(*<*)^(* ∞*)^(, and )***η****t** ∼N*(0*,*** I**) is white noise. The set* {***y**1*, . . . ,*** y***N**}* denotes the training set. The constants* N* and* n *denote the number of training samples in the full training set and in the minibatch, respectively.

**3.2 (Stein’s) Score Function**

Putting aside the Langevin dynamics and the underlying Fokker-Planck equation, the key subject of the iterative procedure is the gradient of the log-likelihood function* ∇***x** log* p*(**x**). It has a formal name known as the** Stein’s score function**, denoted by

**s*****θ***(**x**) def =* ∇***x** log* p****θ***(**x**)*. *(3.3)

We should be careful not to confuse Stein’s score function with the** ordinary score function** which is defined as **s****x**(***θ***) def =* ∇****θ*** log* p****θ***(**x**)*. *(3.4)

The ordinary score function is the gradient (with respect to*** θ***) of the log-likelihood. In contrast, Stein’s score function is the gradient with respect to the data point** x**. Maximum likelihood estimation uses the ordinary score function, whereas Langevin dynamics uses Stein’s score function. However, since most people in the diffusion literature calls Stein’s score function as the score function, we follow this culture.

**Example 3.4.** If* p*(*x*) is a Gaussian with* p*(*x*) = 1 *√*

2*πσ*^(2)^(* e*)^(*−*)^(()^(*x*)^(*−*)^(*µ*)^()2)

2*σ*^(2)^( ), then

*s*(*x*) =* ∇**x* log* p*(*x*) =* −*^(()^(*x*)^(* −*)^(*µ*)^())

*σ*^(2 )*.*

**Example 3.5.** If* p*(*x*) is a Gaussian mixture with* p*(*x*) = ^(P)^(*N *)*i*=1* *^(*π*)^(*i *)1 *√*

2*πσ*^(2 )*i** *^(*e *)*−*^(()^(*x*)^(*−*)^(*µi*)^()2)

2*σ*^(2 )*i *, then

*s*(*x*) =* ∇**x* log* p*(*x*) =* −*

P*N j*=1* *^(*π*)^(*j *)1 *√*

2*πσ*^(2 )*j** *^(*e *)*− *(*x**−**µj* )^(2)

2*σ*^(2 )*j *(*x**−**µ**j*)

*σ*^(2 )*j *P*N i*=1* *^(*π*)^(*i *)1 *√*

2*πσ*^(2 )*i** *^(*e *)*−*^(()^(*x*)^(*−*)^(*µi*)^()2)

2*σ*^(2 )*i **.*

The probability density function and the corresponding score function of the above two examples are shown in Figure 3.5.

(a)* N*(1*,* 1) (b) 0*.*6*N*(2*,* 0*.*5^(2)) + 0*.*4*N*(*−*2*,* 0*.*2^(2))

Figure 3.5: Examples of score functions

**Geometric Interpretations of the Score Function**. The way to understand the score function is to remember that it is the gradient with respect to the data** x**. For any high-dimensional distribution* p*(**x**), the gradient will give us a vector field. There are a few useful interpretations of the score functions:

© 2024 Stanley Chan. All Rights Reserved. 48


---

## Page 49

• The magnitude of the vectors are the strongest at places where the change of log* p*(**x**) is the biggest. Therefore, in regions where log* p*(**x**) is close to the peak will be mostly very weak gradient.

• The vector field indicates* how* a data point should travel in the contour map. In Figure 3.6 we show the contour map of a Gaussian mixture (with two Gaussians). We draw arrows to indicate the vector field. Now if we consider a data point living the space, the Langevin dynamics equation will basically move the data points along the direction pointed by the vector field towards the basin.

• In physics, the score function is equivalent to the “*drift*”. This name suggests how the diffusion particles should flow to the lowest energy state.

(a) vector field of* ∇***x** log* p*(**x**) (b)** x***t* trajectory

Figure 3.6: The contour map of the score function, and the corresponding trajectory of two samples.

**3.3 Score Matching Techniques**

The most difficult question in Langevin dynamics is how to obtain* ∇***x** log* p*(**x**) because we have no access to *p*(**x**). In this section, we briefly discuss a few known techniques. **Explicit Score-Matching** [43]. Suppose that we are given a dataset* X* =* {***x**^((1))*, . . . ,*** x**^(()^(*M*)^())*}*. The solution people came up with is to consider the classical kernel density estimation by defining a distribution

*q**h*(**x**) = ^(1)

*M*

*M *X

*m*=1

1 *h*^(*K *)**x*** −***x**(*m*)

*h*

 *, *(3.5)

where* h* is just some hyperparameter for the kernel function* K*(*·*), and** x**^(()^(*m*)^())^( )is the* m*-th sample in the training set. Figure 3.7 illustrates the idea of kernel density estimation. In the cartoon figure shown on the left, we show multiple kernels* K*(*·*) centered at different data points** x**^(()^(*m*)^()). The sum of all these individual kernels gives us the overall kernel density estimate* q*(**x**). On the right hand side we show a real histogram and the corresponding kernel density estimate. We remark that* q*(**x**) is at best an approximation to the true data distribution* p*(**x**) which is never known. Since* q*(**x**) is an approximation to* p*(**x**) which is never accessible, we can learn** s*****θ***(**x**) based on* q*(**x**). This leads to the following definition of a loss function which can be used to train a network.

**Theorem 3.2.** The** explicit score matching** loss is

*J*ESM(***θ***) def = ^(1)

2^(E)^(*p*)^(()^(**x**)^())^(*∥*)^(**s**)^(***θ***)^(()^(**x**)^())^(* −∇*)^(**x**)^( log)^(* p*)^(()^(**x**)^())^(*∥*)^(2)

*≈*^(1)

2^(E)^(*q*)^(*h*)^(()^(**x**)^())^(*∥*)^(**s**)^(***θ***)^(()^(**x**)^())^(* −∇*)^(**x**)^( log)^(* q*)^(*h*)^(()^(**x**)^())^(*∥*)^(2)^(*. *)(3.6)

© 2024 Stanley Chan. All Rights Reserved. 49


---

## Page 50

Figure 3.7: Illustration of kernel density estimation.

By substituting the kernel density estimation, we can show that the loss is

*J*ESM(***θ***) = E*q**h*(**x**)*∥***s*****θ***(**x**)* −∇***x** log* q**h*(**x**)*∥*^(2)

= Z *∥***s*****θ***(**x**)* −∇***x** log* q**h*(**x**)*∥*^(2)*q**h*(**x**)*d***x**

*≈*^(1)

*M*

*M *X

*m*=1

Z *∥***s*****θ***(**x**)* −∇***x** log* q**h*(**x**)*∥*^(2)^( 1)

*h*^(*K *)**x*** −***x**(*m*)

*h*

 *d***x***. *(3.7)

So, we have derived a loss function that can be used to train the network. Once we train the network** s*****θ***, we can replace it in the Langevin dynamics equation to obtain the recursion:

**x***t*+1 =** x***t* +* τ***s*****θ***(**x***t*) + *√*

2*τ***z***. *(3.8)

The issue of explicit score matching is that the kernel density estimation is a fairly poor non-parameter estimation of the true distribution. Especially when we have a limited number of samples and the samples live in a high dimensional space, the kernel density estimation performance can be poor.

**Implicit Score Matching** [18]. In implicit score matching, the explicit score matching loss is replaced by an implicit one.

*J*ISM(***θ***) def = E*p*(**x**)

 Tr(*∇***x****s*****θ***(**x**)) + ^(1)

2^(*∥*)^(**s**)^(***θ***)^(()^(**x**)^())^(*∥*)^(2 ) *, *(3.9)

where* ∇***x****s*****θ***(**x**) denotes the Jacobian of** s*****θ***(**x**). The implicit score matching loss can be approximated by Monte Carlo

*J*ISM(***θ***)* ≈*^(1)

*M*

*M *X

*m*=1

X

*i*

 *∂**i***s*****θ***(**x**^(()^(*m*)^())) + ^(1)

2^(*|*)^([)^(**s**)^(***θ***)^(()^(**x**)^(()^(*m*)^())^()])^(*i*)^(*|*)^(2 ) *,*

where* ∂**i***s*****θ***(**x**^(()^(*m*)^())) = *∂ ∂x**i* ^([)^(**s**)^(***θ***)^(()^(**x**)^()])^(*i*)^( = )*∂*^(2)

*∂x*^(2 )*i* ^(log)^(* p*)^(()^(**x**)^(). If the model for the score function is realized by a deep )neural network, the trace operator can be difficult to compute, hence making the implicit score matching not scalable [40].

**Denoising Score Matching**. Given the potential drawbacks of explicit and implicit score matching, we now introduce a more popular score matching known as the denoising score matching (DSM) by Vincent [43]. In DSM, the loss function is defined as follows.

*J*DSM(***θ***) def = E*q*(**x***,***x***′*)

1

2* *^(*∥*)^(**s**)^(***θ***)^(()^(**x**)^())^(* −∇*)^(**x**)^( log)^(* q*)^(()^(**x**)^(*|*)^(**x**)^(*′*)^())^(*∥*)^(2 ) (3.10)

The key difference here is that we replace the distribution* q*(**x**) by a conditional distribution* q*(**x***|***x**^(*′*)). The former requires an approximation, e.g., via kernel density estimation, whereas the latter does not.

© 2024 Stanley Chan. All Rights Reserved. 50


---

## Page 51

In the special case where* q*(**x***|***x**^(*′*)) =* N*(**x*** |*** x**^(*′*)*, σ*^(2)), we can let** x** =** x**^(*′*)^( )+* σ***z**. This will give us

*∇***x** log* q*(**x***|***x**^(*′*)) =* ∇***x** log 1

( *√*

2*πσ*^(2))^(*d*)^( exp ) *−*^(*∥*)^(**x**)^(* −*)^(**x**)^(*′*)^(*∥*)^(2)

2*σ*^(2)

=* ∇***x**

 *−*^(*∥*)^(**x**)^(* −*)^(**x**)^(*′*)^(*∥*)^(2)

2*σ*^(2 )*−*log( *√*

2*πσ*^(2))^(*d *)

=* −*^(**x**)^(* −*)^(**x**)^(*′*)

*σ*^(2 )=* −*^(**z**)

*σ *^(*.*)

As a result, the loss function of the denoising score matching becomes

*J*DSM(***θ***) def = E*q*(**x***,***x***′*)

1

2* *^(*∥*)^(**s**)^(***θ***)^(()^(**x**)^())^(* −∇*)^(**x**)^( log)^(* q*)^(()^(**x**)^(*|*)^(**x**)^(*′*)^())^(*∥*)^(2 )

= E*q*(**x***′*)

1

2

**s*****θ***(**x***′* +* σ***z**) +** z**

*σ*

 2^( )*.*

If we replace the dummy variable** x**^(*′*)^( )by** x**, and we note that sampling from* q*(**x**) can be replaced by sampling from* p*(**x**) when we are given a training dataset, we can conclude the following.

**Theorem 3.3.** The** Denoising Score Matching** has a loss function defined as

*J*DSM(***θ***) = E*p*(**x**)

1

2

**s*****θ***(**x** +* σ***z**) +** z**

*σ*

 2^( )(3.11)

The beauty about Eqn (3.11) is that it is highly interpretable. The quantity** x** +* σ***z** is effectively adding noise* σ***z** to a clean image** x**. The score function** s*****θ*** is supposed to take this noisy image and predict the noise **z ***σ*^(. Predicting noise is equivalent to denoising, because any denoised image plus the predicted noise will give )us the noisy observation. Therefore, Eqn (3.11) is a* denoising* step. The following theorem, proven by Vincent [43], establishes the equivalence between DSM and ESM. It is this equivalence that allows us to use DSM to estimate the score function.

**Theorem 3.4.** [Vincent [43]] For up to a constant* C* which is independent of the variable*** θ***, it holds that *J*DSM(***θ***) =* J*ESM(***θ***) +* C. *(3.12)

**Proof of Theorem 3.4** The proof here is based on [43]. We start with the explicit score matching loss function, which is given by

*J*ESM(***θ***) = E*q*(**x**)

1

2* *^(*∥*)^(**s**)^(***θ***)^(()^(**x**)^())^(* −∇*)^(**x**)^( log)^(* q*)^(()^(**x**)^())^(*∥*)^(2 )

= E*q*(**x**) h1

2* *^(*∥*)^(**s**)^(***θ***)^(()^(**x**)^())^(*∥*)^(2)^(* −*)^(**s**)^(***θ***)^(()^(**x**)^())^(*T*)^(* ∇*)^(**x**)^( log)^(* q*)^(()^(**x**)^() + 1)

2* *^(*∥∇*)^(**x**)^( log)^(* q*)^(()^(**x**)^())^(*∥*)^(2)

| {z }

def =* C*1*,*independent of*** θ***

i *.*

Let’s zoom into the second term. We can show that

E*q*(**x**)  **s*****θ***(**x**)^(*T*)^(* *)*∇***x** log* q*(**x**)  = Z � **s*****θ***(**x**)^(*T*)^(* *)*∇***x** log* q*(**x**)  *q*(**x**)*d***x***, *(expectation)

= Z  **s*****θ***(**x**)^(*T*)^(* ∇*)^(**x**)^(*q*)^(()^(**x**)^())

^( )*q*(**x**)

^( )*q*(**x**)*d***x***, *(gradient)

= Z **s*****θ***(**x**)^(*T*)^(* *)*∇***x***q*(**x**)*d***x***.*

© 2024 Stanley Chan. All Rights Reserved. 51


---

## Page 52

Next, we consider conditioning by recalling* q*(**x**) = R *q*(**x**^(*′*))*q*(**x***|***x**^(*′*))*d***x**^(*′*). This will give us Z **s*****θ***(**x**)^(*T*)^(* *)*∇***x***q*(**x**)*d***x** = Z **s*****θ***(**x**)^(*T*)^(* *)*∇***x**

Z *q*(**x**^(*′*))*q*(**x***|***x**^(*′*))*d***x**^(*′ *)

| {z } =*q*(**x**)

*d***x **(conditional)

= Z **s*****θ***(**x**)^(*T *)Z *q*(**x**^(*′*))*∇***x***q*(**x***|***x**^(*′*))*d***x**^(*′ *) *d***x **(move gradient)

= Z **s*****θ***(**x**)^(*T *)Z *q*(**x**^(*′*))*∇***x***q*(**x***|***x**^(*′*))* ×** *^(*q*)^(()^(**x**)^(*|*)^(**x**)^(*′*)^())

*q*(**x***|***x**^(*′*))^(*d*)^(**x**)^(*′ *) *d***x **(multiple and divide)

= Z **s*****θ***(**x**)^(*T *)Z *q*(**x**^(*′*)) *∇***x***q*(**x***|***x***′*)

*q*(**x***|***x**^(*′*))

| {z } =*∇***x** log* q*(**x***|***x**^(*′*))

*q*(**x***|***x**^(*′*))*d***x**^(*′*)*d***x **(rearrange terms)

= Z **s*****θ***(**x**)^(*T *)Z *q*(**x**^(*′*))  *∇***x** log* q*(**x***|***x**^(*′*))  *q*(**x***|***x**^(*′*))*d***x**^(*′ *) *d***x**

= Z Z *q*(**x***|***x**^(*′*))*q*(**x**^(*′*)) | {z } =*q*(**x***,***x**^(*′*))

 **s*****θ***(**x**)^(*T*)^(* *)*∇***x** log* q*(**x***|***x**^(*′*))  *d***x**^(*′*)*d***x **(move integration)

= E*q*(**x***,***x***′*)  **s*****θ***(**x**)^(*T*)^(* *)*∇***x** log* q*(**x***|***x**^(*′*))  *.*

So, if we substitute this result back to the definition of ESM, we can show that

*J*ESM(***θ***) = E*q*(**x**) h1

2* *^(*∥*)^(**s**)^(***θ***)^(()^(**x**)^())^(*∥*)^(2)^( i )*−*E*q*(**x***,***x***′*)  **s*****θ***(**x**)^(*T*)^(* *)*∇***x** log* q*(**x***|***x**^(*′*))  +* C*1*.*

Comparing this with the definition of DSM, we can observe that

*J*DSM(***θ***) def = E*q*(**x***,***x***′*)

1

2* *^(*∥*)^(**s**)^(***θ***)^(()^(**x**)^())^(* −∇*)^(**x**)^(*q*)^(()^(**x**)^(*|*)^(**x**)^(*′*)^())^(*∥*)^(2 )

= E*q*(**x***,***x***′*) h1

2* *^(*∥*)^(**s**)^(***θ***)^(()^(**x**)^())^(*∥*)^(2)^(* −*)^(**s**)^(***θ***)^(()^(**x**)^())^(*T*)^(* ∇*)^(**x**)^( log)^(* q*)^(()^(**x**)^(*|*)^(**x**)^(*′*)^() + 1)

2* *^(*∥∇*)^(**x**)^( log)^(* q*)^(()^(**x**)^(*|*)^(**x**)^(*′*)^())^(*∥*)^(2)

| {z }

def =* C*2*,*independent of*** θ***

i

= E*q*(**x**) h1

2* *^(*∥*)^(**s**)^(***θ***)^(()^(**x**)^())^(*∥*)^(2)^( i )*−*E*q*(**x***,***x***′*)  **s*****θ***(**x**)^(*T*)^(* *)*∇***x** log* q*(**x***|***x**^(*′*))  +* C*2*.*

Therefore, we conclude that *J*DSM(***θ***) =* J*ESM(***θ***)* −**C*1 +* C*2*.*

The** training** procedure in a score matching model is typically done by minimizing the denoising score matching loss function. If we are given a training dataset* {***x**^(()^(*m*)^())*}*^(*M *)*m*=1^(, the optimization goal is to)

***θ***^(*∗*)= argmin ***θ ***E*p*(**x**)

1

2

**s*****θ***(**x** +* σ***z**) +** z**

*σ*

 2^()

*≈*argmin ***θ***

1 *M*

*M *X

*m*=1

1 2

**s*****θ *** **x**^(()^(*m*)^())^( )+* σ***z**^(()^(*m*)^())^( )+** **^(**z**)^(()^(*m*)^())

*σ*



2 *, *where **z**^(()^(*m*)^())^(* *)*∼N*(0*,*** I**)*.*

Figure 3.8 illustrates the training procedure of the score function** s*****θ***(**x**). The above training procedure assumes a fixed noise level* σ*. Generalizing it to multiple noise levels is not difficult. The noise conditioned score network (NCSN) by Song and Ermon [40] argued that one can instead optimize the following loss

*J*NCSN(*θ*) = ^(1)

*L*

*L *X

*i*=1 *λ*(*σ**i*)*ℓ*(***θ***;* σ**i*)*, *(3.13)

© 2024 Stanley Chan. All Rights Reserved. 52


---

## Page 53

Figure 3.8: Training of** s*****θ*** for denoising score matching. The network** s*****θ*** is trained to estimate the noise.

where the individual loss function is defined according to the noise levels* σ*1*, . . . , σ**L*:

*ℓ*(***θ***;* σ*) = E*p*(**x**)

1

2

**s*****θ***(**x** +* σ***z**) +** z**

*σ*

 2^( )*.*

The coefficient function* λ*(*σ**i*) is often chosen as* λ*(*σ*) =* σ*^(2)^( )based on empirical findings [40]. The noise level sequence often satisfies* *^(*σ*)^(1)

*σ*2 ^(=)^(* . . .*)^( =)^(* σ*)^(*L*)^(*−*)^(1)

*σ**L **>* 1. For** inference**, we assume that we have already trained the score estimator** s*****θ***. To generate an image, we use the Langevin equation to iteratively draw samples by denoising the image. In case of NCSN, the corresponding Langevin equation can be implemented via an annealed importance sampling:

**x***t*+1 =** x***t* +* *^(*α*)^(*i*)

2** **^(**s**)^(***θ***)^(()^(**x**)^(*t*)^(*, σ*)^(*i*)^() +)^(* √*)^(*α*)^(*i*)^(**z**)^(*t*)^(*, *)**z***t** ∼N*(0*,*** I**)*,*

where* α**i* =* σ*^(2 )*i** *^(*/σ*)^(2 )*L* ^(is the step size and)^(** s**)^(***θ***)^(()^(**x**)^(*t*)^(*, σ*)^(*i*)^() denotes the score matching function for noise level)^(* σ*)^(*i*)^(. )The iteration over* t* is repeated sequentially for each* σ**i* from* i* = 1 to* L*. For additional details of the implementation, we refer readers to Algorithm 1 of the original paper by Song and Ermon [40].

**3.4 Concluding Remark**

Additional readings about score-matching should start with Vincent’s technical report [43]. A very popular paper in the recent literature is Song and Ermon [40], their follow up work [41], and [42]. In their papers, they brought up an important discussion that when training a score function, we need a noise schedule so that the score function is trained better. Score matching has a wide range of applicability beyond generating images. They can be used in solving many important image restoration problems such as deblurring, denoising, super-resolution, etc. Kadkhodaie and Simoncellil [19] is among the earlier papers to explicitly employ the score function as part of the image reconstruction process. A similar concept is presented by Kawar et al. [21], where they sample from a posterior distribution which contains the forward image formation model and the prior. For problems with a better structured forward model, it has been shown that employing the ideas of proximal maps would improve the performance. This line of work can be built upon the operator splitting strategy such as the plug-and-play ADMM [8] by extending it to Plug-and-Play diffusion model, e.g., Zhu et al. [**?**] or the generative plug and play model by Bouman and Buzzard [5]. Recently, it was also observed that one can directly perform the regression to mean when we do not have access to the degradation process [11]. Various applications papers [9, 17, 37].

© 2024 Stanley Chan. All Rights Reserved. 53


---

## Page 54

### **4 Stochastic Differential Equation (SDE)**

In the previous two sections we studied the diffusion models via the DDPM and the SMLD perspectives. In this section, we will study diffusion models through the lens of differential equation. As we mentioned during our discussions about the Langevin equation, the steady state solution of the Langevin equation is a random variable whose probability distribution satisfies the Fokker-Planck equation. This unusual linkage between the iterative algorithm and a differential equation makes the topic remarkably interesting. The purpose of this section is to provide the basic principles of stochastic differential equations and how they are used to understand diffusion models.

**4.1 From Iterative Algorithms to Ordinary Differential Equations**

The first and the foremost question, at least to readers with little background in differential equations, is why an iterative algorithm can be related to a differential equation. Let’s understand this through two examples.

**Example 4.1. Simple First-Order ODE**. Imagine that we are given a discrete-time algorithm with the iterations defined by the recursion:

**x***i* =  1* −*^(*β*)^(∆)^(*t*)

2

 **x***i**−*1*, *for* i* = 1*,* 2*, . . . , N, *(4.1)

for some hyperparameter* β* and a step-size parameter ∆*t*. We can turn this iterative scheme into a continuous-time differential equation. Suppose that there is a continuous time function** x**(*t*). We define a discretization scheme by letting **x***i* =** x**(* *^(*i*)

*N* ^() for)^(* i*)^( = 1)^(*, . . . , N*)^(, and ∆)^(*t*)^( = )1 *N* ^(, and)^(* t*)^(* ∈{*)^(0)^(*,*)^( 1)

*N** *^(*, . . . ,*)^(* N*)^(*−*)^(1)

*N** *^(*}*)^(. Then the above recursion can be )written as

**x**(*t* + ∆*t*) =  1* −*^(*β*)^(∆)^(*t*)

2

 **x**(*t*)*.*

Rearranging the terms will give us

**x**(*t* + ∆*t*)* −***x**(*t*)

∆*t *=* −*^(*β*)

2** **^(**x**)^(()^(*t*)^())^(*,*)

where at the limit when ∆*t** →*0, we can write the discrete equation as an ordinary differential equation (ODE) *d***x**(*t*)

*dt *=* −*^(*β*)

2** **^(**x**)^(()^(*t*)^())^(*. *)(4.2)

Not only that, we can solve for an analytic solution for the ODE where the solution is given by

**x**(*t*) =* e*^(*−*)^(*β*)

2* *^(*t*)*. *(4.3)

Verification of the solution can be done by substituting Eqn (4.3) into Eqn (4.2).

Figure 4.1: Analytic solution and the estimates produced by the numerical scheme.

© 2024 Stanley Chan. All Rights Reserved. 54


---

## Page 55

The power of the ODE is that it offers us an* analytic* solution. Instead of resorting to the iterative scheme (which will take hundreds to thousands of iterations), the analytic solution tells us exactly the behavior of the solution at* any* time* t*. To illustrate this fact, we show in the figure above the trajectory of the solution** x**1*,*** x**2*, . . . ,*** x***i**, . . . ,*** x***N* defined by the algorithm. Here, we choose ∆*t* = 0*.*1. In the same plot, we directly plot the continuous-time solution** x**(*t*) = exp*{−**βt/*2*}* for arbitrary* t*. As shown, the analytic solution is exactly the same as the trajectory predicted by the iterative scheme.

What we observe in this motivating example are two interesting facts:

• The discrete-time iterative scheme can be written as a continuous-time ODE. In fact, many discrete- time algorithms have their associating ODEs.

• For simple ODEs, we can write down the analytic solution in closed form. More complicated ODE would be hard to write an analytic solution. But we can still use ODE tools to analyze the behavior of the solution. We can also derive the limiting solution* t** →*0.

**Example 4.2. Gradient Descent**. Recall that a gradient descent algorithm for a (well-behaved) convex function* f* is the following recursion. For* i* = 1*,* 2*, . . . , N*, do

**x***i* =** x***i**−*1* −**β**i**−*1*∇**f*(**x***i**−*1)*, *(4.4)

for step-size parameter* β**i*. Using the same discretization as we did in the previous example, we can show that (by letting* β**i**−*1 =* β*(*t*)∆*t*):

**x***i* =** x***i**−*1* −**β**i**−*1*∇**f*(**x***i**−*1) =*⇒ ***x**(*t* + ∆*t*) =** x**(*t*)* −**β*(*t*)∆*t**∇**f*(**x**(*t*))

=*⇒ ***x**(*t* + ∆*t*)* −***x**(*t*)

∆*t *=* −**β*(*t*)*∇**f*(**x**(*t*))

=*⇒ **d***x**(*t*)

*dt *=* −**β*(*t*)*∇**f*(**x**(*t*))*. *(4.5)

The ordinary differential equation shown on the right has a solution trajectory** x**(*t*). This** x**(*t*) is known as the* gradient flow* of the function* f*. For simplicity, we can make* β*(*t*) =* β* for all* t*. Then there are two simple facts about this ODE. First, we can show that

*d dt*^(*f*)^(()^(**x**)^(()^(*t*)^()) =)^(* ∇*)^(*f*)^(()^(**x**)^(()^(*t*)^()))^(*T*)^(* d*)^(**x**)^(()^(*t*)^())

*dt *(chain rule)

=* ∇**f*(**x**(*t*))^(*T*)^( )[*−**β**∇**f*(**x**(*t*))] (Eqn (4.5))

=* −**β**∇**f*(**x**(*t*))^(*T*)^(* *)*∇**f*(**x**(*t*))

=* −**β**∥∇**f*(**x**(*t*))*∥*^(2)^(* *)*≤*0 (norm-squares)*.*

Therefore, as we move from** x***i**−*1 to** x***i*, the objective value* f*(**x**(*t*)) has to go down. This is consistent with our expectation because a gradient descent algorithm should bring the cost down as the iteration goes on. Second, at the limit when* t** →∞*, we know that* *^(*d*)^(**x**)^(()^(*t*)^())

*dt **→*0. Hence,* *^(*d*)^(**x**)^(()^(*t*)^())

*dt *=* −**β**∇**f*(**x**(*t*)) will imply that *∇**f*(**x**(*t*))* →*0*, *as* t** →∞**. *(4.6)

Therefore, the solution trajectory** x**(*t*) will approach the minimizer for the function* f*.

**Forward and Backward Updates**.

Let’s use the gradient descent example to illustrate one more aspect of the ODE. Going back to Eqn (4.4), we recognize that the recursion can be written equivalently as (assuming* β*(*t*) =* β*):

**x***i** −***x***i**−*1 | {z } ∆**x**

=* −**β**i**−*1 |{z} *β*∆*t*

*∇**f*(**x***i**−*1) *⇒ **d***x** =* −**β**∇**f*(**x**)*dt, *(4.7)

© 2024 Stanley Chan. All Rights Reserved. 55


---

## Page 56

where the continuous equation holds when we set ∆*t** →*0 and ∆**x*** →*0. The interesting point about this equality is that it gives us a summary of the update ∆**x** by writing it in terms of* dt*. It says that if we move the along the time axis by* dt*, then the solution** x** will be updated by* d***x**. Eqn (4.7) defines the relationship between* changes*. If we consider a sequence of iterates* i* = 1*,* 2*, . . . , N*, and if we are told that the progression of the iterates follows Eqn (4.7), then we can write

(forward) **x***i* =** x***i**−*1 + ∆**x***i**−*1* ≈***x***i**−*1 +* d***x**

=** x***i**−*1* −∇**f*(**x***i**−*1)*βdt*

*≈***x***i**−*1* −**β**i**−*1*∇**f*(**x***i**−*1)*.*

We call this as the** forward** equation because we update** x** by** x** + ∆**x** assuming that* t** ←**t* + ∆*t*. Now, consider a sequence of iterates* i* =* N, N** −*1*, . . . ,* 2*,* 1. If we are told that the progression of the iterates follows Eqn (4.7), then the time-reversal iterates will be

(reverse) **x***i**−*1 =** x***i** −*∆**x***i** ≈***x***i** −**d***x**

=** x***i* +* β**∇**f*(**x***i*)*dt*

*≈***x***i* +* β**i**∇**f*(**x***i*)*.*

Note the change in sign when reversing the progression direction. We call this the** reverse** equation.

**4.2 What is an SDE?**

Stochastic differential equation (SDE) can be regarded as an extension to the ordinary differential equation (ODE). In this subsection, we briefly introduce the notations and explain their meanings.

**ODE**. We consider a first-order ODE of the following form:

*d***x**(*t*)

*dt *=** f**(*t,*** x**)*,*

where** f**(*t,*** x**) is some function of* t* and** x**. Assuming that the initial condition is** x**(0) =** x**0, the solution takes the form of

**x**(*t*) =** x**0 + Z* t*

0 **f**(*s,*** x**(*s*))*ds.*

The integral form of the solution often comes with a short-hand notation known as the** differential form**, which can be written as *d***x** =** f**(*t,*** x**)*dt. *(4.8)

**SDE**. In an SDE, in addition to a deterministic function** f**(*t,*** x**), we consider a stochastic perturbation. For example, the stochastic perturbation can take the following form:

*d***x**(*t*)

*dt *=** f**(*t,*** x**) +** g**(*t,*** x**)***ξ***(*t*)*, *where ***ξ***(*t*)* ∼N*(0*,*** I**)*,*

where*** ξ***(*t*) is a noise function, e.g., white noise. We can define* d***w** =*** ξ***(*t*)*dt*, where* d***w** is often known as the differential form of the Brownian motion. Then, the differential form of this SDE can be written as

*d***x** =** f**(*t,*** x**)*dt* +** g**(*t,*** x**)*d***w***. *(4.9)

Because*** ξ***(*t*) is random, the solution to this differential equation is also random. To be explicit about the randomness of the solution, we should interpret the differential form via the integral equation

**x**(*t, ω*) =** x**0 + Z* t*

0 **f**(*s,*** x**(*s, ω*))*ds* + Z* t*

0 **g**(*s,*** x**(*s, ω*))*d***w**(*s, ω*)*,*

where* ω* denotes the index of the state of** x**. Therefore, as we pick a particular state of the random process **w**(*s, ω*), we solve a differential equation corresponding to this particular* ω*.

© 2024 Stanley Chan. All Rights Reserved. 56


---

## Page 57

**Example 4.3.** Consider the stochastic differential equation

*d***x** =* ad***w***,*

for some constants* a*. Based on our discussions above, the solution trajectory will take the form

**x**(*t*) =** x**0 + Z* t*

0 *ad***w**(*s*) =** x**0 +* a *Z* t*

0 ***ξ***(*s*)*ds,*

where the last equality uses the fact that* d***w** =*** ξ***(*t*)*dt*. We can visualize the solution trajectory by numerically implementing* d***x** =* ad***w** via the discrete-time iteration:

**x***i** −***x***i**−*1 =* a*(**w***i** −***w***i**−*1) | {z }

def =** z***i**−*1*∼N*(0*,***I**)

*⇒ ***x***i* =** x***i**−*1 +* a***z***i**−*1*.*

For example, if* a* = 0*.*05, one possible trajectory of** x**(*t*) will behave as shown below. The initial point **x**0 = 0 is marked as in red to indicate that the process is moving forward in time.

Figure 4.2: Trajectory of the process* d***x** =* a d***w**.

**Example 4.4.** Consider the stochastic differential equation

*d***x** =* −*^(*α*)

2** **^(**x**)^(*dt*)^( +)^(* βd*)^(**w**)^(*.*)

The solution to this problem is given by [Risken Eqn 3.7]

**x**(*t*) =** x**0*e*^(*−*)^(*α*)

2* *^(*t*)^( )+* β *Z* t*

0 *e*^(*−*)^(*α*)

2 ^(()^(*t*)^(*−*)^(*s*)^())***ξ***(*s*)*ds.*

To visualize the trajectory, we consider* α* = 1 and* β* = 0*.*1. The discretization will give us

**x***i** −***x***i**−*1 =* −*^(*α*)

2** **^(**x**)^(*i*)^(*−*)^(1)^( +)^(* β*)^(()^(**w**)^(*i*)^(* −*)^(**w**)^(*i*)^(*−*)^(1)^() )*⇒ ***x***i* =  1* −*^(*α*)

2

 **x***i**−*1 +* β***z***i**−*1*.*

Figure 4.3: Trajectory of the process* d***x** =* −*^(*α*)

2** **^(**x**)^(*dt*)^( +)^(* βd*)^(**w**)^(.)

© 2024 Stanley Chan. All Rights Reserved. 57


---

## Page 58

**4.3 Stochastic Differential Equation for DDPM and SMLD**

Without drilling into the physics of SDE (which we will do later), we want to connect SDE with the denoising diffusion probabilistic model (DDPM) and the score matching Langevin dynamics (SMLD) we studied in the previous sections.

**Forward and Reverse Diffusion**. Diffusion models involve a pair of equations: the forward diffusion process and the reverse diffusion process. Expressed in terms of derivatives, a forward diffusion process can be written as *d***x**(*t*)

*dt *=** f**(**x***, t*) +* g*(*t*)***ξ***(*t*)*, *where ***ξ***(*t*)* ∼N*(0*,*** I**)*.*

We emphasize that this is a particular diffusion process based on Brownian motion. The random perturbation ***ξ*** is assumed to be a random process with*** ξ***(*t*) being an i.i.d. Gaussian random variable at all* t*. Thus, the autocorrelation function is a delta function E[***ξ***(*t*)***ξ***(*t*^(*′*))] =* δ*(*t** −**t*^(*′*)). For this diffusion process, we can write it in terms of the differential:

**Definition 4.1. Forward Diffusion**.

*d***x** =** f**(**x***, t*) | {z } drift

*dt* + *g*(*t*) |{z} diffusion

*d***w***. *(4.10)

Here, the differential is* d***w** =*** ξ***(*t*)*dt*. This suggests that we can view*** ξ***(*t*) as some rate of change (over* t*) from which the integration of*** ξ***(*t*)*dt* will give us* d***w**. The two terms** f**(**x***, t*) and* g*(*t*) carry physical meanings. The drift coefficient is a vector-valued function **f**(**x***, t*) defining how molecules in a closed system would move in the absence of random effects. For the gradient descent algorithm, the drift is defined by the negative gradient of the objective function. That is, we want the solution trajectory to follow the gradient of the objective. The diffusion coefficient* g*(*t*) is a scalar function describing how the molecules would randomly walk from one position to another. The function* g*(*t*) determines how strong the random movement is. The reverse direction of the diffusion equation is to move backward in time. The reverse-time SDE, according to Anderson [2], is given as follows.

**Definition 4.2. Reverse Diffusion**.

*d***x** = [**f**(**x***, t*) | {z } drift

*−**g*(*t*)^(2)*∇***x** log* p**t*(**x**) | {z } score function

]* dt *+ *g*(*t*)*d***w **| {z } reverse-time diffusion

*, *(4.11)

where* p**t*(**x**) is the probability distribution of** x** at time* t*, and** w** is the Wiener process when time flows backward.

Let’s briefly talk about the reverse-time diffusion. The reverse-time diffusion is nothing but a random process that proceeds in the reverse time order. So while the forward diffusion defines** z***i* =** w***i*+1* −***w***i*, the reverse diffusion defines** z***i* =** w***i**−*1* −***w***i*. The following is an example.

**Example 4.5.** Consider the reverse diffusion equation

*d***x** =* ad***w***. *(4.12)

We can write the discrete-time recursion as follows. For* i* =* N, N** −*1*, . . . ,* 1, do

**x***i**−*1 =** x***i* +* a*(**w***i**−*1* −***w***i*) | {z } =**z***i*

=** x***i* +* a***z***i**, ***z***i** ∼N*(0*,*** I**)*.*

In the figure below we show the trajectory of this reverse-time process. Note that the initial point marked in red is at** x***N*. The process is tracked backward to** x**0.

© 2024 Stanley Chan. All Rights Reserved. 58


---

## Page 59

Figure 4.4: Trajectory of the reverse diffusion* d***x** =* ad***w**.

**Stochastic Differential Equation for DDPM**. In order to draw the connection between DDPM and SDE, we consider the discrete-time DDPM forward iteration. For* i* = 1*,* 2*, . . . , N*:

**x***i* = p

1* −**β**i***x***i**−*1 + p

*β**i***z***i**−*1*, ***z***i**−*1* ∼N*(0*,*** I**)*. *(4.13)

We can show that this equation can be derived from the forward SDE equation below.

**Theorem 4.1.** The forward sampling equation of** DDPM** can be written as an SDE via

*d***x** =* −*^(*β*)^(()^(*t*)^())

2 **x **| {z } =**f**(**x***,t*)

*dt* + p

*β*(*t*) | {z } =*g*(*t*)

*d***w***. *(4.14)

**Proof**. We define a step size ∆*t* = 1 *N* ^(, and consider an auxiliary noise level)^(* {*)^(*β*)^(*i*)^(*}*)^(*N *)*i*=1 ^(where)^(* β*)^(*i*)^( =)^(* β*)^(*i*)

*N* ^(. )Then *β**i* =* β *�* i*

*N *

| {z }

*β**i*

*·* ^(1 )*N* ^(=)^(* β*)^(()^(*t*)^( + ∆)^(*t*)^()∆)^(*t,*)

where we assume that in the* N** →∞*,* β**i** →**β*(*t*) which is a continuous time function for 0* ≤**t** ≤*1. Similarly, we define

**x***i* =** x **�* i*

*N * =** x**(*t* + ∆*t*)*, ***z***i* =** z **�* i*

*N * =** z**(*t* + ∆*t*)*.*

Hence, we have

**x***i* = p

1* −**β**i***x***i**−*1 + p

*β**i***z***i**−*1

*⇒ ***x***i* = q

1* −*^(*β*)^(*i*)

*N*** **^(**x**)^(*i*)^(*−*)^(1)^( + )q

*β**i **N*** **^(**z**)^(*i*)^(*−*)^(1)

*⇒ ***x**(*t* + ∆*t*) = p

1* −**β*(*t* + ∆*t*)* ·* ∆*t*** x**(*t*) + p

*β*(*t* + ∆*t*)* ·* ∆*t*** z**(*t*)

*⇒ ***x**(*t* + ∆*t*)* ≈ * 1* −*^(1)

2^(*β*)^(()^(*t*)^( + ∆)^(*t*)^())^(* ·*)^( ∆)^(*t *) **x**(*t*) + p

*β*(*t* + ∆*t*)* ·* ∆*t*** z**(*t*)

*⇒ ***x**(*t* + ∆*t*)* ≈***x**(*t*)* −*^(1)

2^(*β*)^(()^(*t*)^()∆)^(*t*)^(** x**)^(()^(*t*)^() + )p

*β*(*t*)* ·* ∆*t*** z**(*t*)*.*

Thus, as ∆*t** →*0, we have

*d***x** =* −*^(1)

2^(*β*)^(()^(*t*)^())^(**x**)^(*dt*)^( + )p

*β*(*t*)* d***w***. *(4.15)

Therefore, we showed that the DDPM forward update iteration can be equivalently written as an SDE.

Being able to write the DDPM forward update iteration as an SDE means that the DDPM estimates can be determined by solving the SDE. In other words, for an appropriately defined SDE solver, we can

© 2024 Stanley Chan. All Rights Reserved. 59


---

## Page 60

throw the SDE into the solver. The solution returned by an appropriately chosen solver will be the DDPM estimate. Of course, we are not required to use the SDE solver because the DDPM iteration itself is solving the SDE. It may not be the best SDE solver because the DDPM iteration is only a first order method. Nevertheless, if we are not interested in using the SDE solver, we can still use the DDPM iteration to obtain a solution. Here is an example.

**Example 4.6.** Consider the DDPM forward equation with* β**i* = 0*.*05 for all* i* = 0*, . . . , N** −*1. We initialize the sample** x**0 by drawing it from a Gaussian mixture such that

**x**0* ∼*

*K *X

*k*=1 *π**k**N*(**x**0*|****µ****k**, σ*^(2 )*k*^(**I**)^())^(*,*)

where* π*1 =* π*2 = 0*.*5,* σ*1 =* σ*2 = 1,*** µ***1 = 3 and*** µ***2 =* −*3. Then, using the equation

**x***i* = p

1* −**β**i***x***i**−*1 + p

*β**i***z***i**−*1*, ***z***i**−*1* ∼N*(0*,*** I**)*,*

we can plot the trajectory and the distribution as follows.

Figure 4.5: Realizations of the trajectories of** x***t*, starting with a Gaussian mixture and ending with a single Gaussian.

The reverse diffusion equation follows from Eqn (4.11) by substituting the appropriate quantities: **f**(**x***, t*) =* −*^(*β*)^(()^(*t*)^())

2 and* g*(*t*) = p

*β*(*t*). This will give us

*d***x** = [**f**(**x***, t*)* −**g*(*t*)^(2)*∇***x** log* p**t*(**x**)]*dt* +* g*(*t*)*d***w**

=  *−*^(*β*)^(()^(*t*)^())

2 **x*** −**β*(*t*)*∇***x** log* p**t*(**x**)  *dt* + p

*β*(*t*)*d***w***,*

which will give us the following equation:

**Theorem 4.2.** The reverse sampling equation of** DDPM** can be written as an SDE via

*d***x** =* −**β*(*t*) h**x**

2 ^(+)^(* ∇*)^(**x**)^( log)^(* p*)^(*t*)^(()^(**x**)^() )i *dt* + p

*β*(*t*)*d***w***. *(4.16)

**Proof**. The iterative update scheme can be written by considering* d***x** =** x**(*t*)* −***x**(*t** −*∆*t*), and* d***w** = **w**(*t** −*∆*t*)* −***w**(*t*) =* −***z**(*t*). Then, letting* dt* = ∆*t*, we can show that

**x**(*t*)* −***x**(*t** −*∆*t*) =* −**β*(*t*)∆*t ***x**(*t*)

2 +* ∇***x** log* p**t*(**x**(*t*))  *− *p

*β*(*t*)∆*t***z**(*t*)

*⇒ ***x**(*t** −*∆*t*) =** x**(*t*) +* β*(*t*)∆*t ***x**(*t*)

2 +* ∇***x** log* p**t*(**x**(*t*))  + p

*β*(*t*)∆*t***z**(*t*)*.*

© 2024 Stanley Chan. All Rights Reserved. 60


---

## Page 61

By grouping the terms, and assuming that* β*(*t*)∆*t** ≪*1, we recognize that

**x**(*t** −*∆*t*) =** x**(*t*)  1 +* *^(*β*)^(()^(*t*)^()∆)^(*t*)

2

 +* β*(*t*)∆*t**∇***x** log* p**t*(**x**(*t*)) + p

*β*(*t*)∆*t***z**(*t*)

*≈***x**(*t*)  1 +* *^(*β*)^(()^(*t*)^()∆)^(*t*)

2

 +* β*(*t*)∆*t**∇***x** log* p**t*(**x**(*t*)) + ^(()^(*β*)^(()^(*t*)^()∆)^(*t*)^())^(2)

2 *∇***x** log* p**t*(**x**(*t*)) + p

*β*(*t*)∆*t***z**(*t*)

=  1 +* *^(*β*)^(()^(*t*)^()∆)^(*t*)

2

  **x**(*t*) +* β*(*t*)∆*t**∇***x** log* p**t*(**x**(*t*))  + p

*β*(*t*)∆*t***z**(*t*)*.*

Then, following the discretization scheme by letting* t** ∈{*0*, . . . ,** *^(*N*)^(*−*)^(1)

*N** *^(*}*)^(, ∆)^(*t*)^( = 1)^(*/N*)^(,)^(** x**)^(()^(*t*)^(* −*)^(∆)^(*t*)^() =)^(** x**)^(*i*)^(*−*)^(1)^(, )**x**(*t*) =** x***i*, and* β*(*t*)∆*t* =* β**i*, we can show that

**x***i**−*1 = (1 +* *^(*β*)^(*i*)

2 ^() )h **x***i* +* *^(*β*)^(*i*)

2* *^(*∇*)^(**x**)^( log)^(* p*)^(*i*)^(()^(**x**)^(*i*)^() )i + p

*β**i***z***i*

*≈ *1 *√*1*−**β**i*

h **x***i* +* *^(*β*)^(*i*)

2* *^(*∇*)^(**x**)^( log)^(* p*)^(*i*)^(()^(**x**)^(*i*)^() )i + p

*β**i***z***i**, *(4.17)

where* p**i*(**x**) is the probability density function of** x** at time* i*. For practical implementation, we can replace* ∇***x** log* p**i*(**x***i*) by the estimated score function** s*****θ***(**x***i*).

So, we have recovered the DDPM iteration that is consistent with the one defined by Song and Ermon in [42]. This is an interesting result, because it allows us to connect DDPM’s iteration using the score function. Song and Ermon [42] called the SDE an** variance preserving** (VP) SDE.

**Example 4.7.** Following from the previous example, we perform the reverse diffusion equation using

**x***i**−*1 = 1 *√*1*−**β**i*

h **x***i* +* *^(*β*)^(*i*)

2* *^(*∇*)^(**x**)^( log)^(* p*)^(*i*)^(()^(**x**)^(*i*)^() )i + p

*β**i***z***i**,*

where** z***i** ∼N*(0*,*** I**). The trajectory of the iterates is shown below.

Figure 4.6: Realizations of the trajectories of** x***t*, starting with a single Gaussian and ending with a Gaussian mixture.

**Stochastic Differential Equation for SMLD**. The score-matching Langevin Dynamics model can also be described by an SDE. To start with, we notice that in the SMLD setting, there isn’t really a “forward diffusion step”. However, we can roughly argue that if we divide the noise scale in the SMLD training into *N* levels, then the recursion should follow a Markov chain

**x***i* =** x***i**−*1 + q

*σ*^(2 )*i** *^(*−*)^(*σ*)^(2 )*i**−*1^(**z**)^(*i*)^(*−*)^(1)^(*, *)*i* = 1*,* 2*, . . . , N. *(4.18)

This is not too hard to see. If we assume that the variance of** x***i**−*1 is* σ*^(2 )*i**−*1^(, then we can show that)

Var[**x***i*] = Var[**x***i**−*1] + (*σ*^(2 )*i** *^(*−*)^(*σ*)^(2 )*i**−*1^())

=* σ*^(2 )*i**−*1 ^(+ ()^(*σ*)^(2 )*i** *^(*−*)^(*σ*)^(2 )*i**−*1^() =)^(* σ*)^(2 )*i** *^(*.*)

© 2024 Stanley Chan. All Rights Reserved. 61


---

## Page 62

Therefore, given a sequence of noise levels, Eqn (4.18) will indeed generate estimates** x***i* such that the noise statistics will satisfy the desired property. If we agree Eqn (4.18), it is easy to derive the SDE associated with Eqn (4.18). We summarize our results as follows.

**Theorem 4.3.** The forward sampling equation of** SMLD** can be written as an SDE via

*d***x** =

r

*d*[*σ*(*t*)^(2)]

*dt d***w***. *(4.19)

**Proof**. Assuming that in the limit* {**σ**i**}*^(*N *)*i*=1 ^(becomes the continuous time)^(* σ*)^(()^(*t*)^() for 0)^(* ≤*)^(*t*)^(* ≤*)^(1, and)^(* {*)^(**x**)^(*i*)^(*}*)^(*N *)*i*=1 becomes** x**(*t*) where** x***i* =** x**(* *^(*i*)

*N* ^() if we let)^(* t*)^(* ∈{*)^(0)^(*,*)^( 1)

*N** *^(*, . . . ,*)^(* N*)^(*−*)^(1)

*N** *^(*}*)^(. Then we have)

**x**(*t* + ∆*t*) =** x**(*t*) + p

*σ*(*t* + ∆*t*)^(2)^(* *)*−**σ*(*t*)^(2)**z**(*t*)

*≈***x**(*t*) +

r

*d*[*σ*(*t*)^(2)]

*dt *∆*t*** z**(*t*)*.*

At the limit when ∆*t** →*0, the equation converges to

*d***x** =

r

*d*[*σ*(*t*)^(2)]

*dt d***w***.*

Mapping this to Eqn (4.11), we derive the reverse-time diffusion equation.

**Theorem 4.4.** The reverse sampling equation of** SMLD** can be written as an SDE via

*d***x** =* − **d*[*σ*(*t*)2]

*dt **∇***x** log* p**t*(**x**)  *dt* +

r

*d*[*σ*(*t*)^(2)]

*dt d***w***. *(4.20)

**Proof**. We recognize that

**f**(**x***, t*) = 0*, *and *g*(*t*) =

r

*d*[*σ*(*t*)^(2)]

*dt .*

As a result, if we write the reverse equation Eqn (4.11), we should have

*d***x** = [**f**(**x***, t*)* −**g*(*t*)^(2)*∇***x** log* p**t*(**x**)]* dt* +* g*(*t*)*d***w**

=* − **d*[*σ*(*t*)2]

*dt **∇***x** log* p**t*(**x**(*t*))  *dt* +

r

*d*[*σ*(*t*)^(2)]

*dt d***w***.*

For the discrete-time iterations, we first define* α*(*t*) =* *^(*d*)^([)^(*σ*)^(()^(*t*)^())^(2)^(])

*dt *. Then, using the same set of discretization setups as the DDPM case, we can show that

**x**(*t* + ∆*t*)* −***x**(*t*) =* − * *α*(*t*)*∇***x** log* p**t*(**x**)  ∆*t** − *p

*α*(*t*)∆*t*** z**(*t*)

*⇒ ***x**(*t*) =** x**(*t* + ∆*t*) +* α*(*t*)∆*t**∇***x** log* p**t*(**x**) + p

*α*(*t*)∆*t*** z**(*t*)

*⇒ ***x***i**−*1 =** x***i* +* α**i**∇***x** log* p**i*(**x***i*) +* *^(*√*)*α**i*** z***i *(4.21)

*⇒ ***x***i**−*1 =** x***i* + (*σ*^(2 )*i** *^(*−*)^(*σ*)^(2 )*i**−*1^())^(*∇*)^(**x**) ^(log)^(* p*)^(*i*)^(()^(**x**)^(*i*)^() + )q

(*σ*^(2 )*i** *^(*−*)^(*σ*)^(2 )*i**−*1^())^(** z**)^(*i*)^(*,*)

which is identical to the SMLD reverse update equation. Song and Ermon [42] called the SDE an** variance exploding** (VE) SDE.

© 2024 Stanley Chan. All Rights Reserved. 62


---

## Page 63

**Equivalence between VP and VE’s Inference**. As we have just seen, DDPM and SMLD corre- spond to two variants of the stochastic differential equation: the variance exploding (VE) and the variance preserving (VP). An observation made by Kawar et al. [21] was that the* inference* process determined by VP and VE are equivalent. Therefore, if we want to use a diffusion model as part of another task such as image restoration, it does not matter if we use VE or VP. Note, however, that the specific choice of hyperparameters will still affect the* training* and hence the model itself.

**4.4 Numerical Solvers for ODE and SDE**

SDE and ODE are not easy to solve analytically. In many situations, these differential equations are solved numerically. In this subsection, we briefly highlight the overall idea of these methods. To make our discussion slightly easier, we shall focus on a simple ODE:

*d***x**(*t*)

*dt *=** f**(*t,*** x**(*t*))*. *(4.22)

Geometrically, this ODE means that the* derivative* of** x**(*t*) equals to the function** f**(*t,*** x**) evaluated at** x**(*t*). For this to happen, we need the slope of** x**(*t*) to match with the functional value. In the 1D case, the above geometric interpretation can be translated to the following equation:

*x*(*t*)* −**x*(*t*0)

*t** −**t*0 =* f*(*t*0*, x*0)*,*

where* t*0 is a time fairly close to* t*. By rearranging the terms, we see that this equation is equivalent to

*x*(*t*) =* x*(*t*0) +* f*(*t*0*, x*0)(*t** −**t*0)*.*

So we have recovered something very similar to gradient descent. This is the Euler method.

**Euler Method**. The Euler method is a first-order numerical method for solving the ODE. Given *dx*(*t*)

*dt *=* f*(*t, x*), and* x*(*t*0) =* x*0, Euler method solves the problem via an iterative scheme for* i* = 0*,* 1*, . . . , N**−*1 such that

*x**i*+1 =* x**i* +* α** ·** f*(*t**i**, x**i*)*, i* = 0*,* 1*, . . . , N** −*1*,*

where* α* is the step size. Let’s consider a simple example.

**Example 4.8.** [3, Example 2.2] Consider the following ODE

*dx*(*t*)

*dt *=* *^(*x*)^(()^(*t*)^() +)^(* t*)^(2)^(* −*)^(2)

*t* + 1 *.*

If we apply the Euler method with a step size* α*, then the iteration will take the form

*x**i*+1 =* x**i* +* α** ·** f*(*t**i**, x**i*) =* x**i* +* α** ·* ^(()^(*x*)^(*i*)^( +)^(* t*)^(2 )*i** *^(*−*)^(2) )*t**i* + 1 *.*

**Runge-Kutta (RK) Method**. Another popularly used ODE solver is the Runge-Kutta (RK) method. The classical RK-4 algorithm solves the ODE via the iteration

*x**i*+1 =* x**i* +* *^(*α*)

6* *^(*· *) *k*1 + 2*k*2 + 2*k*3 +* k*4  *, i* = 1*,* 2*, . . . , N,*

where the quantities* k*1,* k*2,* k*3 and* k*4 are defined as

*k*1 =* f*(*x**i**, t**i*)*,*

*k*2 =* f *� *t**i* +* *^(*α*)

2* *^(*, x*)^(*i*)^( +)^(* α*)^(* k*)^(1)

2  *,*

*k*3 =* f *� *t**i* +* *^(*α*)

2* *^(*, x*)^(*i*)^( +)^(* α*)^(* k*)^(2)

2  *,*

*k*4 =* f* (*t**i* +* α, x**i* +* αk*3)* .*

© 2024 Stanley Chan. All Rights Reserved. 63


---

## Page 64

For more details, you can consult numerical methods textbooks such as [3].

**Predictor-Corrector Algorithm** [42]. Since different numerical solvers have different behavior in terms of the error of approximation, throwing the ODE (or SDE) into an off-the-shelf numerical solver will result in various degrees of error [20]. However, if we are specifically trying to solve the reverse diffusion equation, it is possible to use techniques other than numerical ODE/SDE solvers to make the appropriate corrections, as illustrated in Figure 4.7.

Figure 4.7: Prediction and correction algorithm.

Let’s use DDPM as an example. In DDPM, the reverse diffusion equation is given by

**x***i**−*1 = 1 *√*1*−**β**i*

h **x***i* +* *^(*β*)^(*i*)

2* *^(*∇*)^(**x**)^( log)^(* p*)^(*i*)^(()^(**x**)^(*i*)^() )i + p

*β**i***z***i**.*

We can consider it as an Euler method for the reverse diffusion. However, if we have already trained the score function** s*****θ***(**x***i**, i*), we can run the score-matching equation, i.e.,

**x***i**−*1 =** x***i* +* ϵ**i***s*****θ***(**x***i**, i*) + *√*

2*ϵ**i***z***i**,*

for* M* times to make the correction. The algorithm below summarizes the idea. (Note that we have replaced the score function by the estimate.)

**Algorithm 1** Prediction Correction Algorithm for DDPM. [42]

**x***N* =* N*(0*,*** I**). **for*** i* =* N** −*1*, . . . ,* 0** do**

(Prediction) **x***i**−*1 = 1 *√*1*−**β**i*

h **x***i* +* *^(*β*)^(*i*)

2** **^(**s**)^(***θ***)^(()^(**x**)^(*i*)^(*, i*)^() )i + p

*β**i***z***i**. *(4.23)

**for*** m* = 1*, . . . , M*** do**

(Correction) **x***i**−*1 =** x***i* +* ϵ**i***s*****θ***(**x***i**, i*) + *√*

2*ϵ**i***z***i**, *(4.24)

**end for end for**

For the SMLD algorithm, the two equations are:

**x***i**−*1 =** x***i* + (*σ*^(2 )*i** *^(*−*)^(*σ*)^(2 )*i**−*1^())^(**s**)^(***θ***)^(()^(**x**)^(*i*)^(*, σ*)^(*i*)^() + )q

*σ*^(2 )*i** *^(*−*)^(*σ*)^(2 )*i**−*1^(**z **)Prediction*,*

**x***i**−*1 =** x***i* +* ϵ**i**∇***x****s*****θ***(**x***i**, σ**i*) +* *^(*√*)*ϵ**i*** z **Correction*.*

We can pair them up as in the case of DDPM’s prediction-correction algorithm by repeating the correction iteration a few times.

**4.5 Concluding Remark**

The connection between the iterative algorithms with stochastic differential equations is not only a discovery that has its own scientific value, the finding has practical utilities as commented by some papers. By unifying multiple diffusion models to the same SDE framework, one can compare the algorithms. In some cases, one can improve the numerical scheme by borrowing ideas from the SDE literature as well as the probabilistic sampling literature. For example, the predictor-corrector scheme in [42] was a hybrid SDE solver coupled

© 2024 Stanley Chan. All Rights Reserved. 64


---

## Page 65

with a Markov Chain Monte Carlo. Mapping the diffusion iterations to SDE, according to some papers such as [1], offers more design flexibility. Outside the context diffusion algorithms, in general stochastic gradient descent algorithms have corresponding SDE such as the Fokker-Planck equations. People have demonstrated how to theoretically analyze the limiting distribution of the estimates, in exact closed-form. This alleviates the difficulty of analyzing the random algorithm by means of analyzing a well-defined limiting distribution. There is a growing interest in developing accelerated SDE solvers. Lu et al. [26] showed that the typical SDE we encounter in diffusion models can be decomposed into a semi-linear term and a nonlinear term. Since the semi-linear term has a closed-form solution, one just needs to use a specialized numerical solver to handle the nonlinear term. It was further showed that DDIM can be considered as an accelerated SDE solver. Other approaches to accelerate the SDE solver is by means of knowledge distillation [36, 28].

© 2024 Stanley Chan. All Rights Reserved. 65


---

## Page 66

### **5 Langevin and Fokker-Planck Equations**

Many recent papers on diffusion models are focusing on either improving the image generation quality (by generalizing it to more semantically meaningful images) or applying it to problems in new domains (e.g. medical data). To help beginner readers understand better the foundations of these applications, we want to go backward in time by discussing the* physics* of the stochastic differential equations associated with the diffusion models. By studying the fundamental principles of these SDEs, we want to gain a deeper understanding of where these equations are coming from. More specifically, we are interested in studying two major sets of equations related to diffusion: the Langevin equation and the Fokker-Planck equation. There are several goals of this section. First, we want to discuss the origin of the Langevin equation, and explain why is it related to diffusion models. We want to explain the general properties of a Markovian process, and its associated differential equations. Finally, we want to show how the Fokker-Planck equation is derived, and discuss why it plays an important role in diffusion models.

**5.1 Brownian Motion**

**Historical Perspective**. In 1827, botanist Robert Brown observed a phenomenon that small pollen grains have irregular motion when placed in water [6]. The motion of the pollen grains is later named the* Brownian motion*. The explanation of the Brownian motion was made by Albert Einstein in 1905 [14] and independently by Marian Smoluchowski [44] around the same time. Einstein’s main argument was that the motion of the pollen grains is caused by the impact of the water molecules. However, since there are trillions of molecules in the system and we never know the initial state of individual molecules, it is nearly impossible to use classical analysis to study the microscope states. Einstein showed that the mean squared displacement can be related to a diffusion coefficient, and so he introduced a probabilistic approach by considering the statistical behavior of the molecules^(4). Einstein’s theoretical prediction was then empirically confirmed by Jean-Baptiste Perrin who later won the Nobel Prize for Physics in 1926. A few years later since Einstien’s paper, in 1908, French physicist Paul Langevin constructed a random Markovian force to describe the collision and interactions of the particles. This then led to the development of a partial differential equation by Dutch physicist Adriaan Fokker in 1914 and German physicist Max Planck in 1917, which is now known as the Fokker- Planck equation. The Kramers-Moyal expansion was introduced by Hans Kramers in 1940 and Jos´e Enrique Moyal in 1949, which showed a Taylor expansion technique to describe the time evolution of a probability distribution.

**Derivation of Brownian Motion**. So, what is Brownian motion and how is it related to diffusion models? Assume that there is a particle suspended in fluid. Stoke’s law states that the friction applied to the particle is given by *F*(*t*) =* −**αv*(*t*)*, *(5.1)

where* F* is the friction,* v* is the velocity, and* α* = 6*πµR*. Here,* R* is the radius of the particle, and* µ* is the viscosity of the fluid. By Newton’s second law, we further know that* F*(*t*) =* m*˙*v*(*t*), where* m* is the mass of the particle. Equating the two equations ( *F*(*t*) =* −**αv*(*t*) *F*(*t*) =* m** *^(*dv*)^(()^(*t*)^())

*dt** *^(*,*)

we will obtain the following differential equation

*m*^(*dv*)^(()^(*t*)^())

*dt *+* αv*(*t*) = 0*.*

By defining* γ *def = *α m*^(, we can simplify the above equation as)

*dv*(*t*)

*dt *+* γv*(*t*) = 0*. *(5.2)

4A remark here is that Andrey Kolmogorov’s probability book wasn’t published until 1933 [25]. So back in 1905, the notion of probability theory was in its infancy.

© 2024 Stanley Chan. All Rights Reserved. 66


---

## Page 67

The above deterministic equation is accurate when the particle is significantly more massive than the fluid molecules. The reason is that, by conservation of momentum, the massive particle will gain little velocity when it collides with the fluid molecule. However, since pollen grains are so light, the bombardment of water molecules will cause them to accelerate in a small but non-negligible way. This will create a random fluctuation. The total force of the water molecule acting on the particle is then modified to

*F*(*t*) =* −**αv*(*t*) +* F**f*(*t*)*,*

where* F**f*(*t*) is a stochastic term. This will then give us a modified differential equation

*m*^(*dv*)^(()^(*t*)^())

*dt *=* −**αv*(*t*) +* F**f*(*t*)*.*

By defining Γ(*t*) =* F**f*(*t*)*/m*, we arrive at a new differential equation

*dv*(*t*)

*dt *+* γv*(*t*) = Γ(*t*)*, *(5.3)

which can be written in terms of a short-hand notation

˙*v* +* γv* = Γ(*t*)*. *(5.4)

In Eqn (5.4), the random process Γ(*t*) represents a stochastic force known as the** Langevin force**. It satisfies two properties:

(i) E[Γ(*t*)] = 0 for all* t* so that its mean function is a constant zero;

(ii) E[Γ(*t*)Γ(*t*^(*′*))] =* qδ*(*t** −**t*^(*′*)) for all* t* and* t*^(*′*), i.e., the autocorrelation function is a delta function with an amplitude* q*.

**Remark**. Properties (i) and (ii) are special cases of a wide sense stationary process. A wide sense stationary process is a random process that has a constant mean function (the constant is not necessarily

zero), and that the autocorrelation function* R*(*t, t*^(*′*)) def = E[Γ(*t*)Γ(*t*^(*′*))] is a function of the difference *t** −**t*^(*′*). This function is not necessarily a delta function. For example* R*(*t, t*^(*′*)) =* e*^(*−|*)^(*t*)^(*−*)^(*t*)^(*′*)^(*|*)^( )can be a valid autocorrelation function of a wide sense stationary process. A random process satisfying properties (i) and (ii) are sometime called a delta-correlated process in the statistical mechanics literature. There are different ways to construct a delta-correlated process. For example, we can assume Γ(*t*)* ∼N*(0*,* 1) for every* t*, or any other independent and identically distributions defined in the same way. Gaussian distributions are more often used because many physical phenomena can be described by a Gaussian, e.g., thermal noise. A Gaussian random process satisfying properties (i) and (ii) is called a Gaussian white noise. For any wide sense stationary process,** Wiener-Khinchin Theorem** says that the* power spectral density* can be defined through the Fourier transform of the autocorrelation function. More specifically, if* R*(*τ*) = E[Γ(*t* +* τ*)Γ(*t*)] is the autocorrelation function (we can write* R*(*t, t*^(*′*)) as* R*(*τ*) if Γ(*t*) is a wide sense stationary process), Wiener-Khinchin Theorem states that the power spectral density is

*S*(*ω*) = Z* ∞*

*−∞ **R*(*τ*)*e*^(*−*)^(*jωτ*)*dτ. *(5.5)

So if* R*(*τ*) is a delta function,* S*(*ω*) will have a constant value for all* ω*.

**Remark**. The name “**Gaussian white noise**” comes from the fact that the power spectral density *S*(*ω*) is uniform for every frequency* ω* (so it contains all the colors in the visible spectrum). A white noise is defined as Γ(*t*)* ∼N*(0*, σ*^(2)) for all* t*. It is easy to show that such a Γ(*t*) would satisfy the above two criteria. Firstly, E[Γ(*t*)] is E[Γ(*t*)] = 0 by construction (since Γ(*t*)* ∼N*(0*, σ*^(2))). Secondly, if Γ(*t*)* ∼N*(0*, σ*^(2)), it is necessary that* R*(*τ*) = E[Γ(*t* +* τ*)Γ(*t*)] is a delta function. Wiener-Khinchin Theorem then states that the power spectral density is flat because it is the Fourier transform of a delta function. The

© 2024 Stanley Chan. All Rights Reserved. 67


---

## Page 68

figures below shows the random realizations of a white noise, and their autocorrelation functions.

**0 500 1000 1500 2000 -0.2**

**0**

**0.2**

**0.4**

**0.6**

**0.8**

**1**

**1.2**

**correlation of sample 1 correlation of sample 2 auto-correlation function**

**0 200 400 600 800 1000 -4**

**-3**

**-2**

**-1**

**0**

**1**

**2**

**3**

**4**

(a)* R*(*τ*) (b) Γ(*t*)

Figure 5.1: (a) Autocorrelation function* R*(*τ*) of a white Gaussian noise. (b) The random realization of the random process Γ(*t*).

**From Physics to Generative AI**. Because of the randomness exhibited in Γ(*t*), the differential equa- tion given by Eqn (5.4) is a stochastic differential equation (SDE). The solution to this SDE is therefore a random process where the value* v*(*t*) is a random variable at any time* t*. Brownian motion refers to the trajectory of this random process* v*(*t*) as a function of time. The resulting SDE in Eqn (5.4) is a special case of the* Langevin equation*. We call it a linear Langevin equation with a* δ*-correlated Langevin force:

**Definition 5.1.** A** linear Langevin equation** with a* δ*-correlated Langevin force is a stochastic differential equation of the form ˙*ξ* +* γξ* = Γ(*t*)*, *(5.6)

where Γ(*t*) is a random process satisfying two properties that (i) E[Γ(*t*)] = 0 for all* t* and (ii) E[Γ(*t*)Γ(*t*^(*′*))] = *qδ*(*t** −**t*^(*′*)) for all* t* and* t*^(*′*).

At this point we can connect the Langevin equation in Eqn (5.6) with a diffusion model, e.g., DDPM.

**Example 5.1. Forward DDPM**. Recall that a DDPM forward diffusion equation is given by

*d***x** =* −*^(*β*)^(()^(*t*)^())

2 |{z} =*f*(*t*)

**x*** dt* + p

*β*(*t*) | {z } =*g*(*t*)

*d***w***.*

Expressing it in the Langevin equation form, we can write it as

˙***ξ***(*t*) +* f*(*t*)***ξ***(*t*) =* g*(*t*)Γ(*t*)*, *where Γ(*t*)* ∼N*(0*,*** I**)*.*

**Example 5.2. Reverse DDPM**. The reverse DDPM diffusion is given by Eqn (4.16)

*d***x** =* −**β*(*t*) h**x**

2 ^(+)^(* ∇*)^(**x**)^( log)^(* p*)^(*t*)^(()^(**x**)^() )i

| {z } =**f**(***ξ****,t*)

*dt* + p

*β*(*t*) | {z } =*g*(*t*)

*d***w***.*

© 2024 Stanley Chan. All Rights Reserved. 68


---

## Page 69

Expressing it in the Langevin equation form, we can write it as

˙***ξ***(*t*) =** f**(***ξ****, t*) +* g*(*t*)Γ(*t*)*, *where Γ(*t*)* ∼N*(0*,*** I**)*.*

We can continue these examples for other diffusion models such as SMLD. We leave these as exercises for the readers. Our bottom message is that the diffusion equations we see in the previous chapters can all be formulated through the Langevin equation. Therefore, if we want to know the probability distributions of what these diffusion equations produce, we should look for tools in the literature of Langevin equations. **Solution to (Linear) Langevin Equation**. The linear Langevin equation presented in Eqn (5.6) is a simple one. It is possible to analytically derive the solution* ξ*(*t*) at any time* t*. We start by considering the simpler problem where Γ(*t*) = 0. In this case, the differential equation is

˙*ξ*(*t*) +* γξ*(*t*) = 0*,*

and it is called a first-order homogeneous differential equation. The solution of this differential equation is as follows.

**Theorem 5.1.** Consider the following differential equation

˙*ξ*(*t*) +* γξ*(*t*) = 0*,*

with an initial condition* ξ*(0) =* ξ*0. The solution is given by

*ξ*(*t*) =* ξ*0*e*^(*−*)^(*γt*)*. *(5.7)

**Proof**. By rearranging the terms, we can show that

˙*ξ*(*t*) *ξ*(*t*) ^(=)^(* −*)^(*γ,*)

where we assume that* ξ*(*t*)* ̸*= 0 for all* t* so that we can take 1*/ξ*(*t*). Integrating both sides will give us

Z* t*

0

˙*ξ*(*t*^(*′*)) *ξ*(*t*^(*′*))* *^(*dt*)^(*′*)^( =)^(* − *)Z* t*

0 *γ dt*^(*′*)*.*

The left hand side of the equation will give us log* ξ*(*t*)* −*log* ξ*(0) where as the right hand side will give us* −**γt*. Equating them will give us

log* ξ*(*t*)* −*log* ξ*(0) =* −**γt *=*⇒ **ξ*(*t*) =* ξ*0*e*^(*−*)^(*γt*)*.*

Now let’s consider the case where Γ(*t*) is present. The differential equation becomes

˙*ξ* +* γξ* = Γ(*t*)*,*

which is called a first-order non-homogeneous differential equation. To solve this differential equation, we employ a technique known as the* variation of parameter* or* variation of constant* [29, Theorem 1.2.3]. The idea can be summarized in two steps. We know from our previous derivation that the solution to a homogeneous equation is* ξ*(*t*) =* ξ*0*e*^(*−*)^(*γt*). So let’s make an educated guess about the solution of the non- homogenous case that the solution takes the form of* s*(*t*) =* A*(*t*)*e*^(*−*)^(*γt*)^( )for some* A*(*t*). For notation simplicity we define* h*(*t*) =* e*^(*−*)^(*γt*). If* s*(*t*) is indeed the solution to the differential equation, then we can evaluate ˙*s*(*t*) +* γs*(*t*) = Γ(*t*). The left hand side of this equation is

˙*s*(*t*) +* γs*(*t*) = [*A*(*t*)*h*(*t*)]^(*′*)^( )+* γ*[*A*(*t*)*h*(*t*)]

=* A*^(*′*)(*t*)*h*(*t*) +* A*(*t*)*h*^(*′*)(*t*) +* γA*(*t*)*h*(*t*)

=* A*(*t*)[*h*^(*′*)(*t*) +* γh*(*t*)] +* A*^(*′*)(*t*)*h*(*t*)

=* A*^(*′*)(*t*)*h*(*t*)*,*

© 2024 Stanley Chan. All Rights Reserved. 69


---

## Page 70

where the last equality follows from the fact that* h*(*t*) =* e*^(*−*)^(*γt*)^( )is a solution to the homogeneous equation, hence* h*^(*′*)(*t*) +* γh*(*t*) = 0. Therefore, for ˙*s*(*t*) +* γs*(*t*) = Γ(*t*), it is necessary for* A*^(*′*)(*t*)*h*(*t*) = Γ(*t*) by finding an appropriate* A*^(*′*)(*t*). But this is not difficult. The equation* A*^(*′*)(*t*)*h*(*t*) = Γ(*t*) can be written as

*A*^(*′*)(*t*)*e*^(*−*)^(*γt*)^( )= Γ(*t*) *⇒ **A*^(*′*)(*t*) =* e*^(*γt*)Γ(*t*)*.*

Integrating both side will give us

*A*(*t*) = Z* t*

0 *e*^(*γt*)^(*′*)Γ(*t*^(*′*))* dt*^(*′*)*,*

and since* s*(*t*) =* A*(*t*)*e*^(*−*)^(*γt*), we can show that

*s*(*t*) =* e*^(*−*)^(*γt *)Z* t*

0 *e*^(*γt*)^(*′*)Γ(*t*^(*′*))* dt*^(*′*)^( )= Z* t*

0 *e*^(*−*)^(*γ*)^(()^(*t*)^(*−*)^(*t*)^(*′*)^())Γ(*t*^(*′*))* dt*^(*′*)*.*

Therefore, the complete solution (which is the sum of the homogeneous part and the non-homogeneous part) is

*ξ*(*t*) =* ξ*0*e*^(*−*)^(*γt*)^( )+ Z* t*

0 *e*^(*−*)^(*γ*)^(()^(*t*)^(*−*)^(*t*)^(*′*)^())Γ(*t*^(*′*))* dt*^(*′*)*.*

We summarize the result as follows.

**Theorem 5.2.** Consider the following differential equation

˙*ξ*(*t*) +* γξ*(*t*) = Γ(*t*)*,*

with an initial condition* ξ*(0) =* ξ*0. The solution is given by

*ξ*(*t*) =* ξ*0*e*^(*−*)^(*γt*)^( )+ Z* t*

0 *e*^(*−*)^(*γ*)^(()^(*t*)^(*−*)^(*t*)^(*′*)^())Γ(*t*^(*′*))* dt*^(*′*)*. *(5.8)

**Distribution at Equilibrium**. The previous result shows that the solution* ξ*(*t*) is a function of a random process Γ(*t*). Since we do not know the particular realization of Γ(*t*) every time we run the (Brownian motion) experiment, it is often more useful to characterize* ξ*(*t*) by looking at the probability distribution of *ξ*(*t*). In what follows, we follow Risken [33] to analyze the probability distribution at the equilibrium where *t** →∞*and* ξ*(*t*)* →**x*.

**Theorem 5.3.** Consider the Langevin equation in Definition 5.1 where

˙*ξ* +* γξ* = Γ(*t*)*, *(5.9)

and Γ(*t*) is a white Gaussian noise so that it satisfies the properties aforementioned. Let* ξ*(*t*) =* x* be the solution at equilibrium to this SDE, and let* p*(*x*) be the probability distribution of* x*. It holds that

*p*(*x*) = 1 *√*

2*πσ*^(2)^(* e*)^(*−*)^(*x*)^(2)

2*σ*^(2)^(* *)*, *(5.10)

where* σ* = q

*q *2*γ* ^(. In other words, the solution)^(* ξ*)^(()^(*t*)^() =)^(* x*)^( at equilibrium is a zero-mean Gaussian random)

variable.

**Proof**. Let* ξ*0 =* ξ*(0) be the initial condition. Then, the solution of the SDE takes the form

*ξ*(*t*) =* ξ*0*e*^(*−*)^(*γt*)^( )+ Z* t*

0 *e*^(*−*)^(*γ*)^(()^(*t*)^(*−*)^(*t*)^(*′*)^())Γ(*t*^(*′*))*dt*^(*′*)*. *(5.11)

At equilibrium when* t** →∞*, we can drop* ξ*0*e*^(*−*)^(*γt*). Moreover, by letting* τ* =* t** −**t*^(*′*), we can write the

© 2024 Stanley Chan. All Rights Reserved. 70


---

## Page 71

solution as

*ξ*(*t*) = Z* ∞*

0 *e*^(*−*)^(*γ*)^(()^(*t*)^(*−*)^(*t*)^(*′*)^())Γ(*t*^(*′*))*dt*^(*′*)^( )= Z* ∞*

0 *e*^(*−*)^(*γτ*)Γ(*t** −**τ*)*dτ.*

The probability density function* p*(*ξ*) can be determined by taking the inverse Fourier transform of the characteristic function. Recall that the characteristic function of a random variable* v*(*t*) is

*C*(*u*) = E[exp*{**iu** ·** ξ*(*t*)*}*] = 1 +

*∞ *X

*n*=1

(*iu*)^(*n*)

*n*! E[*ξ*(*t*)^(*n*)]*.*

So, to find* C*(*u*) we need to determine the moments E[*ξ*(*t*)^(*n*)]. Using a result in Risken (Chapter 3 Eqns 3.26 and 3.27), we can show that

E[*ξ*(*t*)^(2)^(*n*)^(+1)] = 0

E[*ξ*(*t*)^(2)^(*n*)] = ^((2)^(*n*)^()!)

2^(*n*)*n*!

Z* ∞*

0

Z* ∞*

0 *e*^(*−*)^(*γ*)^(()^(*τ*)^(1)^(+)^(*τ*)^(2)^())*qδ*(*τ*1* −**τ*2)*dτ*1*dτ*2

*n *(5.12)

= ^((2)^(*n*)^()!)

2^(*n*)*n*!

 *q *Z* ∞*

0 *e*^(*−*)^(2)^(*γτ*)^(2)*dτ*2

*n *= ^((2)^(*n*)^()!)

2^(*n*)*n*!

* q*

2*γ*

*n **. *(5.13)

Substituting this result into the characteristic function will give us

*C*(*u*) = 1 + 0 + ^(2!)

2

* q*

2*γ*

 + 0 + ^((2)^(* ·*)^( 2)!)

2^(2)2!

* q*

2*γ*

2 +* . . .*

=

*∞ *X

*n*=0

(*iu*)^(2)^(*n*)E[*ξ*(*t*)^(2)^(*n*)]

(2*n*)!

=

*∞ *X

*n*=0

(*iu*)^(2)^(*n*)

(2*n*)!* *^(*·*)^( (2)^(*n*)^()!)

2^(*n*)*n*!

* q*

2*γ*

*n*

=

*∞ *X

*n*=0

1 *n*!

 *−*^(*u*)^(2)^(*q*)

4*γ*

*n *=* e*^(*−*)^(*u*)^(2)^(*q*)

4*γ** . *(5.14)

Recognizing that this is the characteristic function of a Gaussian, we can use inverse Fourier transform to retrieve the probability density function

*p*(*x*) = r* γ*

*πq *^(*e*)^(*−*)^(*γx*)^(2)

*q** .*

**Example 5.3. (Forward DDPM Distribution at Equilibrium)** Let’s do a sanity check by applying our result to the forward DDPM equation, and see what probability distribution will we obtain at the equilibrium state. For simplicity let’s assume a constant learning rate for the DDPM equation

*d***x** =* −*^(*β*)

2** **^(**x**)^(* dt*)^( + )p

*βd***w***.*

The associate Langevin equation is

˙*ξ*(*t*) +* *^(*β*)

2* *^(*ξ*)^(()^(*t*)^() = )p

*β*Γ(*t*)*.*

© 2024 Stanley Chan. All Rights Reserved. 71


---

## Page 72

As* t** →∞*, our theorem above suggests that

*p*(*x*) = r* γ*

*πq *^(*e*)^(*−*)^(*γx*)^(2)

*q *= 1 *√*

2*π *^(*e*)^(*−*)^(*x*)^(2)

2* ,*

where we substituted* γ* =* β/*2, and* q* =* *^(*√*)*β *2 =* β*. Therefore, the probability distribution of* ξ*(*t*) when *t** →∞*is* N*(0*,* 1). This is consistent with what we expect.

**Wiener Process**. In the special case where* γ* = 0, the linear Langevin equation is simplified to

˙*ξ* = Γ(*t*)*.*

By letting* γ* = 0 in Eqn (5.11), we can show that

*ξ*(*t*) =* ξ*0 + Z* t*

0 Γ(*t*^(*′*))*dt*^(*′*)*,*

which is also known as the Wiener process. The probability distribution of the solution of the Wiener process can be derived as follows.

**Theorem 5.4. Wiener Process**. Consider the Wiener process

˙*ξ* = Γ(*t*)*, *(5.15)

where Γ(*t*) is the Gaussian white noise with E[Γ(*t*)] = 0 and E[Γ(*t*)Γ(*t*^(*′*))] =* qδ*(*t** −**t*^(*′*)). The probability distribution* p*(*x, t*) of the solution* ξ*(*t*) where* ξ*(*t*) =* x* is

*p*(*x, t*) = 1 *√*2*πqte*^(*−*)^(()^(*x*)^(*−*)^(*ξ*)^(0)2)

2*qt **. *(5.16)

**Proof**. The main difference between this result and Theorem 5.3 is that here we are interested in the distribution at any time* t*. To do so, we notice that* ξ*(*t*) =* ξ*0 + R* t *0 ^(Γ()^(*t*)^(*′*)^())^(*dt*)^(*′*)^(. So, to eliminate the non-zero )mean, we can consider* ξ*(*t*)* −**ξ*0 instead. Substituting this into Eqn (5.13), we can show that

E[(*ξ*(*t*)* −**ξ*0)^(2)^(*n*)^(+1)] = 0

E[(*ξ*(*t*)* −**ξ*0)^(2)^(*n*)] = ^((2)^(*n*)^()!)

2^(*n*)*n*!

 *q *Z* t*

0 *e*^(0)*dτ*2

*n *= ^((2)^(*n*)^()!)

2^(*n*)*n*! ^(()^(*qt*)^())^(*n*)^(*.*)

This implies that the characteristic function of* ξ*(*t*)* −**ξ*0 is

*C*(*u*) =

*∞ *X

*n*=0

1 *n*!

 *−*^(*u*)^(2)^(*qt*)

2

*n *=* e*^(*−*)^(*u*)^(2)^(*qt*)

2* . *(5.17)

Taking the inverse Fourier transform will give us the probability distribution for* ξ*(*t*):

*p*(*x, t*) = 1 *√*2*πqte*^(*−*)^(()^(*x*)^(*−*)^(*ξ*)^(0)2)

2*qt **. *(5.18)

To gain insights about this equation, let’s assume* ξ*0 = 0 and* q* = 2*k* for some constant* k*. This will give us *p*(*x, t*) = 1 *√*

4*πkt e*^(*−*)^(*x*)^(2)

4*kt** .*

An interesting observation of this result, which can be found in many thermal dynamics textbooks, is that the probability distribution* p*(*x, t*) derived above is in fact the solution of the** heat equation**:

*∂ ∂t*^(*p*)^(()^(*x, t*)^() =)^(* k ∂*)^(2)

*∂x*^(2)^(* p*)^(()^(*x, t*)^())^(*, *)(5.19)

© 2024 Stanley Chan. All Rights Reserved. 72


---

## Page 73

assuming that the initial condition is* p*(*x,* 0) =* δ*(*x*). To see this, we just need to substitute the probability distribution into the heat equation. We can then see that

*∂ ∂t*^(*p*)^(()^(*x, t*)^() = 1)

2*t** *^(*· *)* x*2

2*kt** *^(*−*)^(1 ) *· *1 *√*

4*πkt e*^(*−*)^(*x*)^(2)

4*kt*

*∂*^(2)

*∂x*^(2)^(* p*)^(()^(*x, t*)^() = )1 2*kt** *^(*· *)* x*2

2*kt** *^(*−*)^(1 ) *· *1 *√*

4*πkt e*^(*−*)^(*x*)^(2)

4*kt** .*

The solution to the heat equation behaves like a Gaussian starting at the origin and expanding outward as time increases. The significance of this result is that while it is relatively difficult to know the exact trajectory of the random process* ξ*(*t*) defined by the Langevin equation, the heat equation provides a full picture of the probability distribution. Figure 5.2 shows the random realization of a Wiener process* ξ*(*t*), and its corresponding probability distribution* p*(*x, t*).

(a) Realizations of* ξ*(*t*) (b)* p*(*x, t*)

Figure 5.2: Realization of a Wiener process. (a) The random process follows the stochastic differential equation. We show a few realizations of the random process. (b) The underlying probability distribution* p*(*x, t*). As* t* increases, the variance of the Gaussian also increases.

For more complicated Langevin equations (involving nonlinear terms), it seems natural to expect a similar partial equation characterizing the probability distribution. More specifically, it seems reasonable to expect that on one side of the equation, we will have* ∂/∂t*. And on the other side of the equation, we will have* ∂*^(2)*/∂x*^(2). As we will show later, the Fokker-Planck equation will have a form similar to this. Indeed, one can derive the heat equation from the Fokker-Planck equation.

**Remark**: For a system of homogeneous differential equations of the form

˙*ξ**i* +

*N *X

*j*=1 *γ**ij**ξ**j* = Γ*i*(*t*)*, i* = 1*, . . . , N,*

with E[Γ*i*(*t*)] = 0 and E[Γ*i*(*t*)Γ*j*(*t*)] =* q**ij**δ*(*t** −**t*^(*′*)), and* q**ij* =* q**ji*, the corresponding random process is known as the** Ornstein-Uhlenbeck** process.

**5.2 Masters Equation**

Thus far we have been studying the linear Langevin equation ^(˙)*ξ*(*t*) +* γξ*(*t*) = Γ(*t*). This equation allows us to handle most of the* forward* diffusions where the goal is to add noise to the sample (i.e., convert the input distribution to a Gaussian.) For* reverse* diffusions such as the reverse DDPM equation and the reverse SMLD equation, we would need something more general. The equation we consider now is the nonlinear Langevin equation, expressed as follows.

© 2024 Stanley Chan. All Rights Reserved. 73


---

## Page 74

**Definition 5.2.** A** Nonlinear Langevin Equation** takes the form of

˙*ξ* =* h*(*ξ, t*) +* g*(*ξ, t*)Γ(*t*)*, *(5.20)

where* h*(*ξ, t*) and* g*(*ξ, t*) are functions denoting the drift and diffusion, respectively. Like before, we assume that Γ(*t*) is a Gaussian white noise so that E[Γ(*t*)] = 0 for all* t*, and E[Γ(*t*)Γ(*t*^(*′*))] = 2*δ*(*t** −**t*^(*′*)).

Readers can refer to Example 5.2 to see how the reverse DDPM would fit this equation. The difficulty of analyzing the nonlinear Langevin equation is that there is no simple closed-form solution. Therefore, we need to develop some mathematical tools to help us understand the nonlinear Langevin equation.

**Markov Property**. Let’s first define a Markov process. Suppose that* ξ*(*t*) has a value* x**n* =* ξ*(*t**n*) at time* t**n*, and let* t*1* ≤**t*2* . . .** ≤**t**n*. We will use the notation* p*(*x**n**, t**n*) to describe the probability density of having* ξ*(*t**n*) =* x**n*. We also introduce the following short-hand notation

**x***n* = [*x**n**, . . . , x*1]*, *and **t***n* = [*t**n**, . . . , t*1]*.*

Therefore,* p*(**x***n**,*** t***n*) =* p*(*x**n**, t**n**, . . . , x*1*, t*1) is the joint distribution of (*ξ*(*t**n*)*, . . . , ξ*(*x*1)). Let’s define a Markov process. We say that a random process* ξ*(*t*) is Markovian if the following memo- ryless condition is met.

**Definition 5.3.** A random process* ξ*(*t*) is said to be a** Markov process** if

*p*(*x**n**, t**n** |*** x***n**−*1*,*** t***n**−*1) =* p*(*x**n**, t**n** |** x**n**−*1*, t**n**−*1)*. *(5.21)

That is, the probability of getting state* x**n* at* t**n* given* all* the previous states is the same as if we are only conditioning on the immediate previous state* x**n**−*1 at* t**n**−*1.

The random process* ξ*(*t*) satisfying the nonlinear Langevin equation defined in Definition 5.2 is Markov, as long as Γ(*t*) is* δ*-correlated. That means, the conditional probability at* t**n* only depends on that value at* t**n**−*1. The reason was summarized by Risken [33]: (i) A first-order differential equation is uniquely determined by its initial value; (ii) A* δ*-correlated Langevin force Γ(*t*) at a former time* t < t**n**−*1 cannot change the conditional probability at a later time* t > t**n**−*1. Risken further elaborates that the Markovian property is destroyed if Γ(*t*) is no longer* δ*-correlated. For example, if Γ(*t*) is such that E[Γ(*t*)Γ(*t*^(*′*))] = *q *2*γ** *^(*e*)^(*−*)^(*γ*)^(*|*)^(*t*)^(*−*)^(*t*)^(*′*)^(*|*)^(, then)

the process described by ^(˙)*ξ*(*t*) =* h*(*ξ*) + Γ(*t*) will be non-Markovian. From now on, we will focus only on the Markov processes.

**Chapman-Kolmogorov Equation**. Consider a Markov process* ξ*(*t*). We can derive a useful result known as the Chapman-Kolmogorov equation. The Chapman-Kolmogorov equation states that the joint distribution at* t*3 and* t*1 can be found by integrating the conditional probabilities of* t*3 given* t*2 and then* t*2 given* t*1. The two key arguments here are the Bayes Theorem plus the definition of marginalization, and the memoryless property of a Markov process.

**Theorem 5.5. Chapman-Kolmogorov Equation**. Let* ξ*(*t*) be a Markov process, and let* x**n* =* ξ*(*t**n*) be the state of* ξ*(*t*) at time* t**n*. Then

*p*(*x*3*, t*3* |** x*1*, t*1) = Z *p*(*x*3*, t*3*|**x*2*, t*2)*p*(*x*2*, t*2*|**x*1*, t*1)*dx*2*, *(5.22)

assuming* t*1* ≤**t*2* ≤**t*3.

© 2024 Stanley Chan. All Rights Reserved. 74


---

## Page 75

**Proof**. For notational simplicity, let’s denote** x***n* =* {**x**n**, . . . , x*1*}* and** t***n* =* {**t**n**, . . . , t*1*}*. If* ξ*(*t*) is Markov, then by the memoryless property of Markov, we have that

*p*(**x***n**,*** t***n*) =* p*(*x**n**, t**n**|***x***n**−*1*,*** t***n**−*1)*p*(**x***n**−*1*,*** t***n**−*1)

=* p*(*x**n**, t**n**|**x**n**−*1*, t**n**−*1)*p*(**x***n**−*1*,*** t***n**−*1)*.*

Repeating the above argument will give us

*p*(**x***n**,*** t***n*) =* p*(*x**n**, t**n**|**x**n**−*1*, t**n**−*1)* · · ·** p*(*x*2*, t*2*|**x*1*, t*1)* ·** p*(*x*1*, t*1)*.*

Consequently, by marginalizing* p*(**x**3*,*** t**3) over* x*2, we can show that

*p*(*x*3*, t*3*, x*1*, t*1) = Z *p*(**x**3*,*** t**3)*dx*2

= Z *p*(*x*3*, t*2*|**x*2*, t*2)*p*(*x*2*, t*2*|**x*1*, t*1)*p*(*x*1*, t*1)*dx*2*.*

By writing* p*(*x*3*, t*3*, x*1*, t*1) =* p*(*x*3*, t*3* |** x*1*, t*1)*p*(*x*1*, t*1), we can show that

*p*(*x*3*, t*3* |** x*1*, t*1) = Z *p*(*x*3*, t*2*|**x*2*, t*2)*p*(*x*2*, t*2*|**x*1*, t*1)*dx*2*. *(5.23)

This completes the proof.

As a corollary of the Chapman-Kolmogorov equation, we can let* x* =* x*3 be the current state,* x*0 =* x*1 be the initial state, and* x*^(*′*)^( )=* x*2 be the intermediate state. Then the equation can be written as

*p*(*x, t** |** x*0*, t*0) = Z *p*(*x, t**|**x*^(*′*)*, t*^(*′*))*p*(*x*^(*′*)*, t*^(*′*)*|**x*0*, t*0)*dx*^(*′*)*, *(5.24)

If we further drop the conditioning on* x*0, we can write

*p*(*x, t*) = Z *p*(*x, t**|**x*^(*′*)*, t*^(*′*))*p*(*x*^(*′*)*, t*^(*′*))*dx*^(*′*)*.*

**Masters Equation**. Based on the Chapman-Kolmogorov equation, we can derive a fundamental equa- tion for Markov processes. This equation is called the Masters Equation.

**Theorem 5.6.** Let* ξ*(*t*) be a Markov process. The** Masters Equation** states that

*∂ ∂t*^(*p*)^(()^(*x, t*)^() = )Z h *W*(*x**|**x*^(*′*))*p*(*x*^(*′*)*, t*)* −**W*(*x*^(*′*)*|**x*)*p*(*x, t*) i *dx*^(*′*)*, *(5.25)

where* W*(*x**|**x*^(*′*)) is the probability density function per unit time.

**Proof**. Recall the Chapman-Kolmogorov Equation

*p*(*x*3*, t*3* |** x*1*, t*1) = Z *p*(*x*3*, t*3*|**x*2*, t*2)*p*(*x*2*, t*2*|**x*1*, t*1)*dx*2*. *(5.26)

We consider the following mapping:

(*x*1*, t*1)* −→*(*x*0*, t*0)

(*x*2*, t*2)* −→*(*x, t*)

(*x*3*, t*3)* −→*(*x, t* + ∆*t*)

© 2024 Stanley Chan. All Rights Reserved. 75


---

## Page 76

Then, the Chapman-Kolmogorov Equation can be written as

*p*(*x, t* + ∆*t** |** x*0*, t*0) = Z *p*(*x, t* + ∆*t** |** x*^(*′*)*, t*)*p*(*x*^(*′*)*, t** |** x*0*, t*0)* dx*^(*′*)*. *(5.27)

Since our goal is to obtain the partial derivative in time, we consider the time derivative of *p*(*x, t** |** x*0*, t*0):

*∂ ∂t*^(*p*)^(()^(*x, t*)^(* |*)^(* x*)^(0)^(*, t*)^(0)^() = lim )∆*t**→*0 *p*(*x, t* + ∆*t** |** x*0*, t*0)* −**p*(*x, t** |** x*0*, t*0)

∆*t*

= lim ∆*t**→*0

R *p*(*x, t* + ∆*t** |** x*^(*′*)*, t*)*p*(*x*^(*′*)*, t** |** x*0*, t*0)*dx*^(*′*)^(* *)*−**p*(*x, t** |** x*0*, t*0)

∆*t .*

We note that on the right-hand side of the equation above, there is an integration. If we switch the variables* x* and* x*^(*′*), we can use the following observation: Z *p*(*x*^(*′*)*, t* + ∆*t** |** x, t*)*dx*^(*′*)^( )= 1*.*

We can insert it into the above equation and obtain

lim ∆*t**→*0 1 ∆*t*

h ^(Z )*p*(*x, t* + ∆*t** |** x*^(*′*)*, t*)*p*(*x*^(*′*)*, t** |** x*0*, t*0)*dx*^(*′*)^(* *)*− *Z *p*(*x*^(*′*)*, t* + ∆*t** |** x, t*)*p*(*x, t** |** x*0*, t*0)*dx*^(*′*)^(i)

Next, we can move the limits into the integration. Let’s define

*W*(*x, t** |** x*^(*′*)*, t*) = lim ∆*t**→*0 1 ∆*t*^(*p*)^(()^(*x, t*)^( + ∆)^(*t*)^(* |*)^(* x*)^(*′*)^(*, t*)^())

*W*(*x*^(*′*)*, t** |** x, t*) = lim ∆*t**→*0 1 ∆*t*^(*p*)^(()^(*x*)^(*′*)^(*, t*)^( + ∆)^(*t*)^(* |*)^(* x, t*)^())

So, we have

*∂ ∂t*^(*p*)^(()^(*x, t*)^(* |*)^(* x*)^(0)^(*, t*)^(0)^() = )Z h *W*(*x, t** |** x*^(*′*)*, t*)*p*(*x*^(*′*)*, t** |** x*0*, t*0)* −**W*(*x*^(*′*)*, t** |** x, t*)*p*(*x, t** |** x*0*, t*0) i *dx*^(*′ *)(5.28)

If we fix (*x*0*, t*0), then we can drop the conditioning. This will give us

*∂ ∂t*^(*p*)^(()^(*x, t*)^() = )Z h *W*(*x, t** |** x*^(*′*)*, t*)*p*(*x*^(*′*)*, t*)* −**W*(*x*^(*′*)*, t** |** x, t*)*p*(*x, t*) i *dx*^(*′*)*. *(5.29)

In the derivation above, the terms* W*(*x, t** |** x*^(*′*)*, t*) and* W*(*x*^(*′*)*, t** |** x, t*) are known as the** transition rates**. They are the transition probability* per unit time*, with a unit [time^(*−*)^(1)]. Thus, if we integrate them with respect to time, we will obtain Z *W*(*x, t** |** x*^(*′*)*, t*)*dt* =* p*(*x, t** |** x*^(*′*)*, t*) Z *W*(*x*^(*′*)*, t** |** x, t*)*dt* =* p*(*x*^(*′*)*, t** |** x, t*)*.*

One way to visualize the Masters Equation is to consider R *W*(*x, t** |** x*^(*′*)*, t*)*p*(*x*^(*′*)*, t*)*dx*^(*′*)^( )as the* in-flow* and R *W*(*x*^(*′*)*, t** |** x, t*)*p*(*x, t*)*dx*^(*′*)^( )as the* out-flow* of the transition probability from state* x*^(*′*)^( )to* x* (and from* x* to* x*^(*′*)). So if we view the probability as the density of particles in a room, then the Masters Equation says that the rate of the change of the density is the difference between the in-flow and the out-flow of the particles:

*∂ ∂t*^(*p*)^(()^(*x, t*)^() )| {z } rate of change

= Z h *W*(*x, t** |** x*^(*′*)*, t*)*p*(*x*^(*′*)*, t*) i *dx*^(*′*)

| {z } in-flow of probability

*− *Z h *W*(*x*^(*′*)*, t** |** x, t*)*p*(*x, t*) i *dx*^(*′*)

| {z } out-flow of probability

*. *(5.30)

© 2024 Stanley Chan. All Rights Reserved. 76


---

## Page 77

Masters Equation is used widely in chemistry, biology, and many disciplines. The notion of in-flow and out-flow of participles is particularly useful to study the dynamics of a system. Another important aspect of the Masters Equation is that it relates time* ∂t* with the state* dx*^(*′*). This will become prevalent in the Fokker-Planck equation. One thing we can complain about the above proof is that although it is rigorous, it lacks the physics intuition during the proof. In what follows, we present an alternative proof which is more intuitive but less rigorous. The proof is based on a Lecture Note of Luca Donati [13].

**Intuitive Proof**. Consider a particle that can take only two states either 1 or 2. The probabilities of landing on a particular state are* p*(*x*1*, t*) and* p*(*x*2*, t*), such that* p*(*x*1*, t*) +* p*(*x*2*, t*) = 1 for any* t*. Now consider a small interval (*t, t* +* dt*). In this interval, the particle can either stay at its current state or it can jump to the other state. This means that at the end of* t* +* dt*, we can write

*p*(*x*1*, t* +* dt*) =* p*(*x*1*, t*)P[stay in* x*1] +* p*(*x*2*, t*)P[move from* x*2 to* x*1]

=* p*(*x*1*, t*)  1* −*P[move from* x*1 to* x*2]  +* p*(*x*2*, t*)P[move from* x*2 to* x*1]

Let define the rate* W*(*x*2*|**x*1) and* W*(*x*1*|**x*2) such that

P[move from* x*1 to* x*2] =* W*(*x*2*|**x*1)*dt*

P[move from* x*2 to* x*1] =* W*(*x*1*|**x*2)*dt.*

Notice here we have implicitly assumed that the transition distribution is Markov so that the current state only depends on its previous state and not the entire history. Then the above equation can be written as

*p*(*x*1*, t* +* dt*) =* p*(*x*1*, t*)(1* −**W*(*x*2*|**x*1)*dt*) +* p*(*x*2*, t*)*W*(*x*1*|**x*2)*dt* +* O*(*dt*^(2))*.*

The high-order term is there to account for multiple jumps during (*t, t* +* dt*), e.g., jump from* x*1 to* x*2 and then from* x*2 to* x*1 within the interval. However, this term will vanish if* dt** →*0. By rearranging the terms, we can write

*dp*(*x*1*, t*)

*dt *=* −**W*(*x*2*|**x*1)*p*(*x*1*, t*) +* W*(*x*1*|**x*2)*p*(*x*2*, t*)*.*

We can generalize this result to multiple states to and from* x*1. For example,

*dp*(*x*1*, t*)

*dt *= X

*j**̸*=1 [*−**W*(*x**j**|**x*1)*p*(*x*1*, t*) +* W*(*x*1*|**x**j*)*p*(*x**j**, t*)]* .*

To make it even more general, we can consider a continuum of* x**j*. By rearranging the terms, we will obtain

*dp*(*x, t*)

*dt *= Z h *W*(*x**|**x*^(*′*))*p*(*x*^(*′*)*, t*)* −**W*(*x*^(*′*)*|**x*)*p*(*x, t*) i *dx*^(*′*)*,*

which is the Masters equation.

**5.3 Kramers-Moyal Expansion**

With the Masters Equation developed, we can now tackle the nonlinear Langevin Equation. Recall that the nonlinear Langevin Equation does not have a closed form solution, and hence we cannot write down the probability distribution of the solution analytically. Masters Equation allows us to write down the* conditions *for the probability distribution through a partial differential equation known as the Fokker-Planck Equation. The derivation of the Fokker-Planck Equation requires a mathematical result known as the Kramers-Moyal Expansion.

© 2024 Stanley Chan. All Rights Reserved. 77


---

## Page 78

**Theorem 5.7.** Let* ξ*(*t*) be a Markov process and let* p*(*x, t*) be the probability distribution of* ξ*(*t*) taking a value* x* at time* t*. The** Kramers-Moyal Expansion** states that

*∂ ∂t*^(*p*)^(()^(*x, t*)^() =)

*∞ *X

*m*=1

 *−*^(*∂*)^(*m*)

*∂x*^(*m*)^(* D*)^(()^(*m*)^())^(()^(*x, t*)^())^(*p*)^(()^(*x, t*)^() ) *,*

where the Kramers-Moyal expansion coefficients are defined as

*D*^(()^(*m*)^())(*x, t*) = ^(1)

*m*! ^(lim )∆*t**→*0

 1

∆*t*^(E)^( [()^(*ξ*)^(()^(*t*)^( + ∆)^(*t*)^())^(* −*)^(*x*)^())^(*m*)^(] ) *ξ*(*t*)=*x*

 *.*

**Proof**. Let’s start with the Masters Equation. The Masters Equation states that

*∂ ∂t*^(*p*)^(()^(*x, t*)^(* |*)^(* x*)^(0)^(*, t*)^(0)^() = lim )∆*t**→*0 1 ∆*t*

" Z *p*(*x, t* + ∆*t** |** x*^(*′*)*, t*)*p*(*x*^(*′*)*, t** |** x*0*, t*0)*dx*^(*′*)

*− *Z *p*(*x*^(*′*)*, t* + ∆*t** |** x, t*)*p*(*x, t** |** x*0*, t*0)*dx*^(*′ *)#

*.*

We can inject a test function* φ*(*x*) such that

*∂ ∂t*

Z *φ*(*x*)*p*(*x, t** |** x*0*, t*0)*dx* = lim ∆*t**→*0 1 ∆*t*

(Z *φ*(*x*)

Z *p*(*x, t* + ∆*t** |** x*^(*′*)*, t*)*p*(*x*^(*′*)*, t** |** x*0*, t*0)*dx*^(*′*)*dx*

*−*

Z *φ*(*x*)

Z *p*(*x*^(*′*)*, t* + ∆*t** |** x, t*)*p*(*x, t** |** x*0*, t*0)*dx*^(*′*)*dx*

)

*.*

Taylor expansion of the test function will give us an infinite series

*φ*(*x*) =* φ*(*x*^(*′*)) +

*∞ *X

*m*=1

(*x** −**x*^(*′*))^(*m*)

*m*! *∂*^(*m*)

*∂x*^(*m*)^(* φ*)^(()^(*x*)^() ) *x*=*x*^(*′*)^(*.*)

Substituting the expansion in the Masters Equation will give us

*∂ ∂t*

Z *φ*(*x*)*p*(*x, t** |** x*0*, t*0)*dx*

= lim ∆*t**→*0 1 ∆*t*

" ZZ

*φ*(*x*^(*′*))*p*(*x, t* + ∆*t** |** x*^(*′*)*, t*)*p*(*x*^(*′*)*, t** |** x*0*, t*0)*dx*^(*′*)*dx*

+ ZZ *∞ *X

*m*=1

(*x** −**x*^(*′*))^(*m*)

*m*! *∂*^(*m*)

*∂x*^(*m*)^(* φ*)^(()^(*x*)^() ) *x*=*x*^(*′*)^(*p*)^(()^(*x, t*)^( + ∆)^(*t*)^(* |*)^(* x*)^(*′*)^(*, t*)^())^(*p*)^(()^(*x*)^(*′*)^(*, t*)^(* |*)^(* x*)^(0)^(*, t*)^(0)^())^(*dx*)^(*′*)^(*dx*)

*− *ZZ *φ*(*x*)*p*(*x*^(*′*)*, t* + ∆*t** |** x, t*)*p*(*x, t** |** x*0*, t*0)*dx*^(*′*)*dx*

#

(5.31)

We notice that the last double integral in the equation above has dummy variables* x*^(*′*)^( )and* x*. We can switch the dummy variables, and write ZZ *φ*(*x*)*p*(*x*^(*′*)*, t* + ∆*t** |** x, t*)*p*(*x, t** |** x*0*, t*0)*dx*^(*′*)*dx* = ZZ *φ*(*x*^(*′*))*p*(*x, t* + ∆*t** |** x*^(*′*)*, t*)*p*(*x*^(*′*)*, t** |** x*0*, t*0)*dx*^(*′*)*dx*

© 2024 Stanley Chan. All Rights Reserved. 78


---

## Page 79

Then, the first and the third double integrals in Eqn (5.31) can be canceled. This will leave us

*∂ ∂t*

Z *φ*(*x*)*p*(*x, t** |** x*0*, t*0)*dx*

= lim ∆*t**→*0 1 ∆*t*

ZZ *∞ *X

*m*=1

(*x** −**x*^(*′*))^(*m*)

*m*! *∂*^(*m*)

*∂x*^(*m*)^(* φ*)^(()^(*x*)^() ) *x*=*x*^(*′*)^(*p*)^(()^(*x, t*)^( + ∆)^(*t*)^(* |*)^(* x*)^(*′*)^(*, t*)^())^(*p*)^(()^(*x*)^(*′*)^(*, t*)^(* |*)^(* x*)^(0)^(*, t*)^(0)^())^(*dx*)^(*′*)^(*dx *)(5.32)

Now let’s define *D*^(()^(*m*)^())(*x*^(*′*)*, t*) = ^(1)

*m*! ^(lim )∆*t**→*0 1 ∆*t*

Z (*x** −**x*^(*′*))^(*m*)*p*(*x, t* + ∆*t** |** x*^(*′*)*, t*)* dx.*

Then, the Masters Equation becomes

*∂ ∂t*

Z *φ*(*x*)*p*(*x, t** |** x*0*, t*0)*dx*

= Z *∞ *X

*m*=1 *D*^(()^(*m*)^())(*x*^(*′*)*, t*)* *^(*∂*)^(*m*)

*∂x*^(*m*)^(* φ*)^(()^(*x*)^() ) *x*=*x*^(*′*)^(*p*)^(()^(*x*)^(*′*)^(*, t*)^(* |*)^(* x*)^(0)^(*, t*)^(0)^())^(*dx*)^(*′ *)Substitute* D*^(()^(*m*)^())(*x*^(*′*)*, t*)

=

*∞ *X

*m*=1

Z *D*^(()^(*m*)^())(*x*^(*′*)*, t*)*p*(*x*^(*′*)*, t** |** x*0*, t*0)* *^(*∂*)^(*m*)

*∂x*^(*m*)^(* φ*)^(()^(*x*)^() ) *x*=*x*^(*′*)^(*dx*)^(*′ *)Switch summation and integration

=

*∞ *X

*m*=1 (*−*1)^(*m *)Z *φ*(*x*^(*′*))* *^(*∂*)^(*m*)

*∂x*^(*m*)

h *D*^(()^(*m*)^())(*x, t*)*p*(*x, t, x*0*, t*0) i *dx*^(*′ *)Generalized integration by part*,*

where the last step is known as the generalized integration by part which states that for any continuously differentiable functions* f* and* g*, Z *g** ·** *^(*∂*)^(*m*)^(*f*)

*∂x*^(*m*)^(* dx*)^( = ()^(*−*)^(1))^(*m *)Z *f *^(*∂*)^(*m*)^(*g*)

*∂x*^(*m*)^(* dx.*)

Combining all these, and recognizing that the above result holds for any arbitrary* φ*(*x*), it follows that *∂ ∂t*^(*p*)^(()^(*x, t*)^(* |*)^(* x*)^(0)^(*, t*)^(0)^() =)

*∞ *X

*m*=1 (*−*1)^(*m*)^(* ∂*)^(*m*)

*∂x*^(*m*)

h *D*^(()^(*m*)^())(*x, t*)*p*(*x, t, x*0*, t*0) i *. *(5.33)

If we further drop the conditioning on (*x*0*, t*0), we will obtain

*∂ ∂t*^(*p*)^(()^(*x, t*)^() =)

*∞ *X

*m*=1

1 *m*!^(()^(*−*)^(1))^(*m*)^(* ∂*)^(*m*)

*∂x*^(*m*)

h *D*^(()^(*m*)^())(*x, t*)*p*(*x, t*) i (5.34)

which completes the proof.

Kramers-Moyal expansion expresses the time-derivative* ∂t* of the probability distribution of any Markov process (including the solution of the nonlinear Langevin Equation) through the spatial-derivative* ∂x*. How- ever, the expansion has infinitely many terms. An important question now is whether we allowed to truncate any of these terms. If so, how many terms can be truncated? Pawula Theorem provides an answer to this question: [30]

**Theorem 5.8. Pawula Theorem**. The Kramers-Moyal expansion may stop at one of the following three cases:

•* m* = 1: The resulting differential equation is known as the Liouville Equation which is a deter- ministic process.

© 2024 Stanley Chan. All Rights Reserved. 79


---

## Page 80

•* m* = 2: The resulting differential equation is known as the Fokker-Planck Equation.

•* m* =* ∞*, i.e., the expansion cannot be truncated.

**Proof**. Recall that the Kramer’s-Moyal coefficients are defined as

*D*^(()^(*m*)^())(*x, t*) = ^(1)

*m*! ^(lim )∆*t**→*0

 1

∆*t*^(E)^( [()^(*ξ*)^(()^(*t*)^( + ∆)^(*t*)^())^(* −*)^(*x*)^())^(*m*)^(] ) *ξ*(*t*)=*x*

 *.*

Denote* x*^(*′*)^( )=* ξ*(*t* + ∆*t*), the subject of interest here is the* m*-th moment E [(*x*^(*′*)^(* *)*−**x*)^(*m*)]. We apply Cauchy-Schwarz inequality which states that for any functions* f* and* g*, and random variables* X* and* Y* , it follows that

E[*f*(*X*)*g*(*Y* )]^(2)^(* *)*≤*E[*f*(*X*)^(2)]E[*g*(*Y* )^(2)]*.*

We consider two possibilities:

• Suppose that* m** ≥*3 and* m* is odd, then

E[(*x*^(*′*)^(* *)*−**x*)^(*m*)]^(2)^( )= E h (*x*^(*′*)^(* *)*−**x*) *m**−*1

2 (*x*^(*′*)^(* *)*−**x*) *m*+1

2 i2 *≤*E  (*x*^(*′*)^(* *)*−**x*)^(*m*)^(*−*)^(1)^( )E  (*x*^(*′*)^(* *)*−**x*)^(*m*)^(+1)^( )*.*

• Suppose that* m** ≥*4 and* m* = is even, then

E[(*x*^(*′*)^(* *)*−**x*)^(*m*)]^(2)^( )= E h (*x*^(*′*)^(* *)*−**x*) *m**−*2

2 (*x*^(*′*)^(* *)*−**x*) *m*+2

2 i2 *≤*E  (*x*^(*′*)^(* *)*−**x*)^(*m*)^(*−*)^(2)^( )E  (*x*^(*′*)^(* *)*−**x*)^(*m*)^(+2)^( )*.*

Note that we cannot apply the above arguments for* m* = 0*,* 1*,* 2 because they will give trivial equalities. From these two relationship, and suppose that we denote* D**m* =* D*^(()^(*m*)^())(*x, t*), then the above two cases can be written as

*D*^(2 )*m** *^(*≤*)^(*D*)^(*m*)^(*−*)^(1)^(*D*)^(*m*)^(+1)^(*, *)*m* odd and* m** ≥*3*,*

*D*^(2 )*m** *^(*≤*)^(*D*)^(*m*)^(*−*)^(2)^(*D*)^(*m*)^(+2)^(*, *)*m* even and* m** ≥*4*.*

Our goal now is to show that this recurring relationship will give us* D**m* = 0 for any* m** ≥*3. Suppose first that* D*4 = 0. Then* D*6* ≤**D*4*D*8 implies that* D*6 = 0. But if* D*6 = 0 then* D*8* ≤**D*6*D*10 implies that* D*8 = 0. Repeating the process will give us* D**m* = 0 for* m* = 4*,* 6*,* 8*,* 10*, . . .*. Similarly, suppose that* D*6 = 0. Then* D*4* ≤**D*2*D*6 implies that* D*4 = 0. But if* D*4 = 0, we can go back to the first case and show that* D**m* = 0 for* m* = 4*,* 6*,* 8*,* 10*, . . .*. In general, all even* m** ≥*4 should be zero if any one of these even* m** ≥*4 is zero. For the odd* m*’s: If* D*4 = 0, then* D*3* ≤**D*2*D*4 implies that* D*3 = 0. Similarly, if* D*6 = 0, we will have* D*5 = 0. So if* D*4 =* D*6 =* D*8 =* . . .* = 0, then* D*3 =* D*5 =* D*7 =* . . .* = 0. Therefore, if* D**m* = 0 for any even* m* such that* m** ≥*4, then* D**m* = 0 all integers* m** ≥*3. The above analysis suggests that if Kramers-Moyal expansion is truncated up to* m* = 3 so that *D*3* ̸*= 0 and* D*4 =* D*5 =* . . .* = 0, then* D*4 = 0 will force* D*3 = 0. So we will have* D**m* = 0 for all *m** ≥*3. Similarly, if the Kramers-Moyal expansion is truncated up to* m* = 4 so that* D*4* ̸*= 0 and *D*5 =* D*6 =* . . .* = 0, then* D*6 = 0 will force* D*4 = 0. So we will have* D**m* = 0 for all* m** ≥*3 again. By repeating the above argument for other* m** ≥*3, we see that it is impossible to have Kramers- Moyal expansion be truncated for any* m** ≥*3. In other words, we can either truncate the expansion for *m* = 1,* m* = 2, or we never truncate it.

Pawula Theorem does not say that the Fokker-Planck Equation (truncating Kramers-Moyal Expansion up to* m* = 2) is a good approximation to the underlying Masters Equation. It only says that we can either exactly approximate the Masters Equation using* m* = 1 or* m* = 2, or we cannot approximate at all.

© 2024 Stanley Chan. All Rights Reserved. 80


---

## Page 81

**5.4 Fokker-Planck Equation**

We can now discuss the Fokker-Planck Equation. The Fokker-Planck Equation is the truncation of the Kramers-Moyal’s Expansion using* m* = 2.

**Definition 5.4.** The** Fokker-Planck Equation** is obtained by truncating the Kramers-Moyal expan- sion to* m* = 2. That is, for any Markov process* ξ*(*t*), the probability distribution* p*(*x, t*) of* ξ*(*t*) =* x* at time* t* will satisfy the following partial differential equation:

*∂ ∂t*^(*p*)^(()^(*x, t*)^() =)^(* −*)^(*∂*)

*∂x*^(*D*)^((1))^(()^(*x, t*)^())^(*p*)^(()^(*x, t*)^() +)^(* ∂*)^(2)

*∂x*^(2)^(* D*)^((2))^(()^(*x, t*)^())^(*p*)^(()^(*x, t*)^())^(*. *)(5.35)

Fokker-Planck Equation is a general result for* any* Markov random processes because it is a consequence of the Chapman-Kolmogorov Equation and the Masters Equation. Processes we study in this tutorial, e.g., Langevin Equation, are special cases of this big family of random processes. Therefore, if we have a Langevin Equation, it is necessary that the solution* ξ*(*t*) will have a probability distribution satisfying the Fokker-Planck Equation.

**Nonlinear Langevin Equation**. If we focus on the nonlinear Langevin equation

˙*ξ* =* h*(*ξ, t*) +* g*(*ξ, t*)Γ(*t*)*,*

we can evaluate the Kramers-Moyal coefficients* D*^(()^(*m*)^())(*x, t*)*p*(*x, t*). The following theorem summarizes the coefficients. We remark that during the proof of this theorem, it will become clear why* D*^(()^(*m*)^())(*x, t*)*p*(*x, t*) is only limited to* m* = 1 and* m* = 2.

**Theorem 5.9. Fokker-Planck for nonlinear Langevin Equation**. Consider the nonlinear Langevin equation ˙*ξ* =* h*(*ξ, t*) +* g*(*ξ, t*)Γ(*t*)*,*

for functions* h*(*ξ, t*) and* g*(*ξ, t*). The Fokker-Planck Equation for this nonlinear Langevin equation will have Kramers-Moyal coefficients:

(Drift) *D*^((1))(*x, t*) =* h*(*x, t*) +* g*^(*′*)(*x, t*)*g*(*x, t*) (5.36)

(Diffusion) *D*^((2))(*x, t*) =* g*^(2)(*x, t*)*. *(5.37)

**Proof**. Recall the definition of the Kramers-Moyal coefficient:

*D*^(()^(*m*)^())(*x, t*) = ^(1)

*m*! ^(lim )*τ**→*0 E [(*ξ*(*t* +* τ*)* −**x*)^(*m*)]

*τ*

 *ξ*(*t*)=*x*^(*.*)

The hard part is how to evaluate the moments E [(*ξ*(*t* +* τ*)* −**x*)^(*m*)]. We start by looking at the nonlinear Langevin equation:

˙*ξ* =* h*(*ξ, t*) +* g*(*ξ, t*)Γ(*t*)*.*

Expressing* ξ*(*t* +* τ*)* −**ξ*(*t*) = R* t*+*τ t *˙*ξ*(*t*^(*′*))*dt*^(*′*) for a small* τ* and let* ξ*(*t*) =* x*, we can write

*ξ*(*t* +* τ*)* −**x* = Z* t*+*τ*

*t*

h *h*(*ξ*(*t*^(*′*))*, t*^(*′*)) +* g*(*ξ*(*t*^(*′*))*, t*^(*′*))Γ(*t*^(*′*)) i *dt*^(*′*)*.*

We assume that* h* and* g* can be expanded as

*h*(*ξ*(*t*^(*′*))*, t*^(*′*)) =* h*(*x, t*^(*′*)) +* h*^(*′*)(*x, t*^(*′*))(*ξ*(*t*^(*′*))* −**x*) +* . . .*

*g*(*ξ*(*t*^(*′*))*, t*^(*′*)) =* g*(*x, t*^(*′*)) +* g*^(*′*)(*x, t*^(*′*))(*ξ*(*t*^(*′*))* −**x*) +* . . . .*

© 2024 Stanley Chan. All Rights Reserved. 81


---

## Page 82

This will give us

*ξ*(*t* +* τ*)* −**x* = Z* t*+*τ*

*t*

h *h*(*x, t*^(*′*)) +* h*^(*′*)(*x, t*^(*′*))(*ξ*(*t*^(*′*))* −**x*) +* . . . *i *dt*^(*′*)

+ Z* t*+*τ*

*t*

h *g*(*x, t*^(*′*)) +* g*^(*′*)(*x, t*^(*′*))(*ξ*(*t*^(*′*))* −**x*) +* . . . *i Γ(*t*^(*′*))*d*^(*′*)*t*

= Z* t*+*τ*

*t **h*(*x, t*^(*′*))*dt*^(*′*)^( )+ Z* t*+*τ*

*t **h*^(*′*)(*x, t*^(*′*))(*ξ*(*t*^(*′*))* −**x*)*dt*^(*′*)^( )+* . . .*

+ Z* t*+*τ*

*t **g*(*x, t*^(*′*))Γ(*t*^(*′*))*dt*^(*′*)^( )+ Z* t*+*τ*

*t **g*^(*′*)(*x, t*^(*′*))(*ξ*(*t*^(*′*))* −**x*)Γ(*t*^(*′*))*dt*^(*′*)^( )+* . . . .*

Now, we iterate the above equation and replace* ξ*(*t*^(*′*))* −**x* by the integrations. This will give us

*ξ*(*t* +* τ*)* −**x* = Z* t*+*τ*

*t **h*(*x, t*^(*′*))*dt*^(*′*)^( )+ Z* t*+*τ*

*t **h*^(*′*)(*x, t*^(*′*))

"Z* t**′*

*t **h*(*x, t*^(*′′*))*dt*^(*′′ *)#

*dt*^(*′*)+

+ Z* t*+*τ*

*t **h*^(*′*)(*x, t*^(*′*))

"Z* t**′*

*t **g*(*x, t*^(*′′*))Γ(*t*^(*′′*))*dt*^(*′′ *)#

*dt*^(*′*)^( )+* . . .*

+ Z* t*+*τ*

*t **g*(*x, t*^(*′*))Γ(*t*^(*′*))*dt*^(*′*)^( )+ Z* t*+*τ*

*t **g*^(*′*)(*x, t*^(*′*))

"Z* t**′*

*t **h*(*x, t*^(*′′*))*dt*^(*′′ *)#

Γ(*t*^(*′*))*dt*^(*′*)

+ Z* t*+*τ*

*t **g*^(*′*)(*x, t*^(*′*))

"Z* t**′*

*t **g*(*x, t*^(*′′*))Γ(*t*^(*′′*))*dt*^(*′′ *)#

Γ(*t*^(*′*))*dt*^(*′*)^( )+* . . . *(5.38)

where we only write the terms involving* h*,* g* and Γ. Terms involving* ξ*(*t*^(*′′*))* −**x* are not dropped. Take expectation, and noticing that E[Γ(*t*)] = 0, we can show that only the first two terms and the last term in Eqn (5.38) will survive. Thus, we have

E[*ξ*(*t* +* τ*)* −**x*] = Z* t*+*τ*

*t **h*(*x, t*^(*′*))*dt*^(*′*)^( )+ Z* t*+*τ*

*t*

Z* t*^(*′*)

*t **h*^(*′*)(*x, t*^(*′*))*h*(*x, t*^(*′′*))*dt*^(*′′*)*dt*^(*′*)^( )+* . . .*

+ Z* t*+*τ*

*t **g*^(*′*)(*x, t*^(*′*)) Z* t*^(*′*)

*t **g*(*x, t*^(*′′*))2*δ*(*t*^(*′′*)^(* *)*−**t*^(*′*))*dt*^(*′′*)

| {z } =*g*(*x,t*^(*′*))

*dt*^(*′*)^( )+* . . . . *(5.39)

where we follow Risken’s definition that R* t*^(*′*)

*t* ^(2)^(*δ*)^(()^(*t*)^(*′′*)^(* −*)^(*t*)^(*′*)^())^(*dt*)^(*′′*)^( = 1 [33]. As)^(* τ*)^(* →*)^(0, it follows that the first )and third terms of Eqn (5.39) are

lim *τ**→*0

Z* t*+*τ*

*t **h*(*x, t*^(*′*))*dt*^(*′*)^( )=* h*(*x, t*)*,*

lim *τ**→*0

Z* t*+*τ*

*t **g*^(*′*)(*x, t*^(*′*))*g*(*x, t*^(*′*))*dt*^(*′*)^( )=* g*^(*′*)(*x, t*)*g*(*x, t*)*.*

For the second term in Eqn (5.39), we can show that

lim *τ**→*0

Z* t*+*τ*

*t*

Z* t*^(*′*)

*t **h*^(*′*)(*x, t*^(*′*))*h*(*x, t*^(*′′*))*dt*^(*′′*)*dt*^(*′*)^( )= lim *τ**→*0

Z* t*+*τ*

*t **h*^(*′*)(*x, t*^(*′*))(*H*(*x, t*^(*′*))* −**H*(*x, t*))*dt*^(*′*)

=* h*^(*′*)(*x, t*)(*H*(*x, t*)* −**H*(*x, t*)) = 0*,*

© 2024 Stanley Chan. All Rights Reserved. 82


---

## Page 83

where we assume* h*(*x, t*) is integrable over* t* and so we can define* H*(*x, t*) = R* t *0* *^(*h*)^(()^(*x, t*)^(*′*)^())^(*dt*)^(*′*)^(. Therefore, we )arrive at

*D*^((1))(*x, t*) =* h*(*x, t*) +* g*^(*′*)(*x, t*)*g*(*x, t*)*.*

The derivation of* D*^((2))(*x, t*) follows essentially the same set of arguments. The key to note here is that when we take the square in E[(*ξ*(*t* +* τ*)* −**x*)^(2)], the integrals in Eqn (5.39) will give us contributions proportional to* τ* ^(2). When* τ** →*0, all these terms will vanish because there is only one 1*/τ* in the definition of* D*^((2))(*x, t*). As a result, the only term that can survive is

*D*^((2))(*x, t*) = ^(1)

2 ^(lim )*τ**→*0 1 *τ*

Z* t*+*τ*

*t*

Z* t*+*τ*

*t **g*(*x, t*^(*′*))*g*(*x, t*^(*′′*))2*δ*(*t*^(*′*)^(* *)*−**t*^(*′′*))*dt*^(*′*)*dt*^(*′′*)

=* g*^(2)(*x, t*)*,*

which completes the proof.

**Example 5.4.** Consider the following Langevin Equation

˙*ξ* =* A*(*ξ*)*ξ* +* σ*Γ(*t*)*.*

Then, the probability distribution* p*(*x, t*) for the solution* ξ*(*t*) will satisfy the following Fokker-Planck equation

*∂ ∂t*^(*p*)^(()^(*x, t*)^() =)^(* −*)^(*∂*)

*∂x*

h *h*(*x, t*) +* g*^(*′*)(*x, t*)*g*(*x, t*) i *p*(*x, t*) +* *^(*∂*)^(2)

*∂x*^(2)

h *g*^(2)(*x, t*)*p*(*x, t*) i

=* −*^(*∂*)

*∂x* ^([()^(*A*)^(()^(*x*)^() + 0)^(* ·*)^(* σ*)^())^(* p*)^(()^(*x, t*)^()] +)^(* ∂*)^(2)

*∂x*^(2)

h *σ*^(2)*p*(*x, t*) i

=* −*^(*∂*)

*∂x* ^([)^(*A*)^(()^(*x*)^())^(*p*)^(()^(*x, t*)^()] +)^(* σ*)^(2)^(* ∂*)^(2)

*∂x*^(2)^( [)^(*p*)^(()^(*x, t*)^()])^(*.*)

**Example 5.5.** For the special case where* A*(*x*) = 0, the Langevin equation is simplified to a Wiener process: ˙*ξ* =* σ*Γ(*t*)*.*

The corresponding Fokker-Planck equation is

*∂ ∂t*^(*p*)^(()^(*x, t*)^() =)^(* σ*)^(2)^(* ∂*)^(2)

*∂x*^(2)^(* p*)^(()^(*x, t*)^())^(*.*)

This equation is known as the** heat equation** or the** diffusion equation**. If the initial condition is that* p*(*x,* 0) =* δ*(*x*), the solution is (See derivation below.)

*p*(*x, t*) = 1 *√*

4*πσ*^(2)*t e*^(*− *)*x*^(2)

4*σ*^(2)*t** .*

**Solution to Heat Equation**. The heat equation can be solved using Fourier transforms. For nota- tional simplicity we denote* u**t* =* *^(*∂u*)

*∂t* ^(and)^(* u*)^(*xx*)^( =)^(* ∂*)^(2)^(*u*)

*∂x*^(2)^( . Consider a generic heat equation:)

*u**t*(*x, t*) =* ku**xx*(*x, t*)*, x** ∈*R*, t >* 0*,*

with initial condition* u*(*x,* 0) =* ϕ*(*x*). We can take Fourier transform (to map between* x** ↔**ω*) on both

© 2024 Stanley Chan. All Rights Reserved. 83


---

## Page 84

sides by defining

b*u**t*(*ω, t*) =* F *n *u**t*(*x, t*) o = Z* ∞*

*−∞ **u**t*(*x, t*)*e*^(*jωx*)*dx*

b*u**xx*(*ω, t*) =* F *n *u**xx*(*x, t*) o = Z* ∞*

*−∞ **u**xx*(*x, t*)*e*^(*jωx*)*dx.*

This will give us

b*u**t*(*ω, t*) =* k*b*u**xx*(*ω, t*)*.*

Using the differentiation property of Fourier transform, we can write the right hand side of the equation as

*k*b*u**xx*(*ω, t*) =* k*(*jω*)^(2)b*u*(*ω, t*) =* −**kω*^(2)b*u*(*ω, t*)*,*

which will give us an ordinary differential equation

b*u**t*(*ω, t*) =* −**kω*^(2)b*u*(*ω, t*)*. *(5.40)

The solution to this differential equation (in* t*) is given by

b*u*(*ω, t*) = ^(b)*ϕ*(*ω*)*e*^(*−*)^(*kω*)^(2)^(*t*)*. *(5.41)

(Remark: Eqn (5.40) is just a simple differential equation* f** *^(*′*)(*t*) =* af*(*t*) whose solution can be found by integration.) Therefore, if we take the inverse Fourier transform on b*u*(*ω, t*)(with respect to* ω** ↔**x*), we will have

*u*(*x, t*) =* F*^(*−*)^(1)*{*b*u*(*ω, t*)*}* =* F*^(*−*)^(1)^( n )b*ϕ*(*ω*)*e*^(*−*)^(*kω*)^(2)^(*t*)^(o)

which is the inverse Fourier transform on the product of ^(b)*ϕ*(*ω*) and ^(b)*f*(*ω*) def =* e*^(*−*)^(*kω*)^(2)^(*t*). Since multiplication in the Fourier domain is convolution in the spatial domain, it follows that* u*(*x, t*) is the convolution of *ϕ*(*x*) and* f*(*x*) =* F*^(*−*)^(1)(*e*^(*−*)^(*kω*)^(2)^(*t*)). But* F*^(*−*)^(1)(*e*^(*−*)^(*kω*)^(2)^(*t*)) = 1 *√*

2*kt*^(*e*)^(*−*)^(*x*)^(2)^(*/*)^((4)^(*kt*)^())^(. Therefore, we can show that the )solution is

*u*(*x, t*) = 1 *√*

2*π*

Z* ∞*

*−∞ **ϕ*(*x** −**x*^(*′*))*f*(*x*^(*′*))*dx*^(*′*)

= 1 *√*

2*π*

Z* ∞*

*−∞ **ϕ*(*x** −**x*^(*′*)) 1 *√*

2*kt e*^(*−*)^(()^(*x*)^(*′*)^())^(2)^(*/*)^((4)^(*kt*)^())*dx*^(*′*)*.*

So, if* ϕ*(*x*) =* δ*(*x*) so that* ϕ*(*x** −**x*^(*′*)) =* δ*(*x** −**x*^(*′*)), it follows that

*u*(*x, t*) = 1 *√*

4*kπt e*^(*−*)^(*x*)^(2)

4*kt** . *(5.42)

**Probability Current**. The Fokker-Planck Equation has some interesting physics interpretations. Recall that the Fokker-Planck Equation is

*∂ ∂t*^(*p*)^(()^(*x, t*)^() =)^(* −*)^(*∂*)

*∂x*^(*D*)^((1))^(()^(*x, t*)^())^(*p*)^(()^(*x, t*)^() +)^(* ∂*)^(2)

*∂x*^(2)^(* D*)^((2))^(()^(*x, t*)^())^(*p*)^(()^(*x, t*)^())^(*. *)(5.43)

Let’s define a quantity

*S*(*x, t*) =  *D*^((1))(*x, t*)* −*^(*∂*)

*∂x*^(*D*)^((2))^(()^(*x, t*)^() ) *p*(*x, t*)*. *(5.44)

© 2024 Stanley Chan. All Rights Reserved. 84


---

## Page 85

Then, the Fokker-Planck Equation can be written as

*∂p*

*∂t* ^(+)^(* ∂S*)

*∂x* ^(= 0)^(*. *)(5.45)

One way to interpret Eqn (5.45) is that it is the** probability current**.

**Intuitive Derivation of Eqn** (5.45). Conversation of energy tells us that if some increases/decreases in a spatial region (e.g., particles or charges), the change in the amount should be equal to the change in its surface. So, if* p*(*x, t*) represents some sort of density, then* p*(*x, t*)*dx* will be the amount of particles sitting between (*x, x* +* dx*) at time* t*. Here,* S*(*x, t*) can be viewed as the current of the particle flowing per unit time across* x*. For a time interval (*t, t* +* dt*) and a spatial interval (*x, x* +* dx*), the change in amount of particles is

number of particles increased/decreased = [*p*(*x, t* +* dt*)* −**p*(*x, t*)]*dx.*

Because of the conversation of energy, for this change to happen, there must be some flow of the current. The current over the interval is

number of particles flowing in/out = [*S*(*x, t*)* −**S*(*x* +* dx, t*)]*dt.*

By equating the two, we will obtain

[*p*(*x, t* +* dt*)* −**p*(*x, t*)]*dx* = [*S*(*x, t*)* −**S*(*x* +* dx, t*)]*dt.*

Taylor approximation to the first order will give us* p*(*x, t*+*dt*)*−**p*(*x, t*) =* *^(*∂p*)

*∂t* ^(and)^(* S*)^(()^(*x, t*)^())^(*−*)^(*S*)^(()^(*x*)^(+)^(*dx, t*)^() = )*∂S ∂x* ^(. This will then give us )*∂p*

*∂t* ^(+)^(* ∂S*)

*∂x* ^(= 0)^(*. *)(5.46)

Therefore, the Fokker-Planck Equation can be regarded as one form of conversation of energy where the change in* p* (over time) should be equal to change in* S* (over space). **Equilibrium Solution**. At equilibrium, the current vanishes and so we have* S* = 0. Consequently, we can show the following.

**Theorem 5.10.** At equilibrium, since the probability current vanishes, the probability distribution *p*(*x*) satisfies

*D*^((1))(*x, t*)*p*(*x*) =* *^(*∂*)

*∂x*

h *D*^((2))(*x, t*)*p*(*x*) i *.*

**Example 5.6.** For example, if* D*^((1))^( )=* −**γx* and* D*^((2))^( )=* *^(*γkT*)

*m* ^(, then)^(* S*)^( = 0 will give us ) *−**γx** −*^(*γkT*)

*m ∂ ∂x*

 *p*(*x*) = 0*.*

Then* −**γxp*(*x*) =* *^(*γkT*)

*m ∂ ∂x*^(*p*)^(()^(*x*)^(). Solving this differential equation will give us)

*p*(*x*) = r *m *2*πkT *^(*e*)^(*−*)^(*mx*)^(2)

2*kT** .*

**Example 5.7. Connection with SMLD** Let’s try to map our results with Eqn (3.1) defined in

© 2024 Stanley Chan. All Rights Reserved. 85


---

## Page 86

Definition 3.1. We shall consider the 1D case. Consider the following Langevin equation

*∂x*

*∂t *|{z} ˙*ξ*

=* τ *^(*∂*)

*∂x* ^(log)^(* p*)^(()^(*x*)^() )| {z } *h*(*ξ,t*)

+ *σ *|{z} *g*(*ξ,t*) Γ(*t*)*.*

To avoid notational confusions, we let* W*(*x, t*) be the probability distribution of the solution* x*(*t*) for this Langevin equation. The Kramers-Moyal coefficients for this Langevin Equation are

*D*^((1))(*x, t*) =* h*(*x, t*) +* g*^(*′*)(*x, t*)*g*(*x, t*) =* τ *^(*∂*)

*∂x* ^(log)^(* p*)^(()^(*x*)^() )def =* A*(*x*)

*D*^((2))(*x, t*) =* g*(*x, t*)^(2)^( )=* σ*^(2)*.*

So, the corresponding Fokker-Planck Equation is

*∂ ∂t*^(*W*)^(()^(*x, t*)^() =)^(* −*)^(*∂*)

*∂x*

h *D*^((1))(*x, t*)*W*(*x, t*) i +* *^(*∂*)^(2)

*∂x*^(2)

h *D*^((2))(*x, t*)*W*(*x, t*) i

=* −*^(*∂*)

*∂x* ^([)^(*A*)^(()^(*x*)^())^(*W*)^(()^(*x, t*)^()] +)^(* σ*)^(2)^(* ∂*)^(2)

*∂x*^(2)^(* W*)^(()^(*x, t*)^())^(*.*)

At equilibrium when* t** →∞*, the probability distribution* W*(*x, t*) can be written as* W*(*x*). Since the probability current vanishes, it follows that

*A*(*x*)*W*(*x*) =* σ*^(2)^(* ∂*)

*∂x*^(*W*)^(()^(*x*)^())^(*.*)

Recall that* A*(*x*) =* τ** *^(*∂*)

*∂x* ^(log)^(* p*)^(()^(*x*)^(), it follows that)

*τ *^(*∂*)

*∂x* ^(log)^(* p*)^(()^(*x*)^() =)^(* σ*)^(2)^(* ∂*)

*∂x*^(*W*)^(()^(*x*)^())^(*.*)

Since we have the freedom to choose* σ*, we will just make it* σ* =* *^(*√*)*τ*. Then the above equation is simplified to *∂ ∂x* ^(log)^(* p*)^(()^(*x*)^() = )*∂ ∂x*^(*W*)^(()^(*x*)^(). Integrating both sides with respect to)^(* x*)^( will give us)

log* p*(*x*) =* W*(*x*) +* C, *(5.47)

for some constant* C*. Let* U*(*x*) =* e*^(*W*)^( ()^(*x*)^())^( )be a probability distribution so that* W*(*x*) = log* U*(*x*) is the log-likelihood, Eqn (5.47) will give us* p*(*x*) =* U*(*x*)*e*^(*C*). Since* p*(*x*) and* U*(*x*) are probability distributions, we must have R *p*(*x*)*dx* = 1 and R *U*(*x*)*dx* = 1. Thus, we can show that* C* = 0. Therefore, we conclude that if we run the Langevin equation until convergence, the probability distribution* W*(*x*) of the solution is exactly the ground truth distribution* p*(*x*). Moreover, the noise level* σ* and the step size* τ* is related by* σ* =* *^(*√*)*τ*.

**5.5 Concluding Remark**

In this section we discussed the physics behind Brownian motion, Langevin equation, and Fokker-Planck equation, and demonstrated several classical theorems in the statistical physics literature. Many of our results are general, as they can be applied to any Markov random process. Going beyond the Markov processes can be done by limiting the spatial/temporal correlation to a small interval. There are a plethora of references on this topic, many of which originated from physics. Risken’s textbook [33] is a classic reference on this subject which contains essentially all the ingredients. For readers looking for something slightly more general, Reichl’s statistical physics book [31] would serve the purpose.

© 2024 Stanley Chan. All Rights Reserved. 86


---

## Page 87

### **6 Conclusion**

This tutorial covers a few basic concepts underpinning the diffusion-based generative models in the recent literature. We find it particularly important to go deeper into these fundamental principles rather than staying at the surface of Python programming. As we write this tutorial, a few lessons we learned are worth sharing. The development of DDPM is in many senses an extension of the VAE, both in terms of the structure, the usage of the evidence lower bound, and the re-parameterization trick. While DDPM is still one of the state-of-the-art methods, it would be more ideal if future models can avoid any iteration. Some recent papers are beginning to explore the feasibility of using knowledge distillation to reduce the number of iterations. Some are investigating acceleration methods by borrowing ideas from the differential equation literature. For readers who are interested in imaging, the score matching Langevin dynamics would likely continue to play a fundamental role in solving inverse problems. The training of the score-matching function is nearly identical to training an image denoiser, and the application of a score-matching step is identical to a denoising step. Therefore, as soon as we know how to split the inverse problem into the forward model and prior distribution, we will be able to leverage score matching to perform the posterior sampling. Recent approaches have also begun to explore different forms of Bayesian diffusion models to connect the physical model and the data-driven model. The SDE and Fokker-Planck equations offer great theoretical intuitions as to why the SMLD equation was derived in a certain way. Historically speaking, the development of SMLD does not appear to start with the SDE but the realization today helps us understand the behavior of the solution statistics. Looking into the future, the biggest challenges of diffusion models are the consistency with the physical world, let alone their high computational complexity. Class-specific learning will continue to be influential. E.g., one can train a customized diffusion model using the gallery images on a cell phone. Temporal consis- tency would demand for larger models with more memory to store the frames. New architectures are needed to leverage the increasing number of temporal inputs to the model. Another open question is how to bring language and semantics into image generation. Should images continue to be represented as an array of pixels (or pixels of features), or are there new ways to describe the scene using a few words without losing information? Finally, information forensics is going to be the biggest challenge for the next decades until we can develop effective countermeasures (or policies).

**Acknowledgement**

This work is supported, in part, by the National Science Foundation under the awards 2030570, 2134209, and 2133032, as well as by SRC JUMP 2.0 Center, and research awards from Samsung Research America. Since this draft was posted on the internet in March 2024, we received numerous constructive feedback from readers from all over the world. Thank you all for your input. Thanks also to many of the graduate students at Purdue who shared good thoughts about the content of the tutorial. We want to give a special thanks to William Chi-Kin Yau who worked tirelessly with us on the section about Langevin and Fokker-Planck equations.

© 2024 Stanley Chan. All Rights Reserved. 87


---

## Page 88

### **References**

[1] Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. https://arxiv.org/abs/2303.08797.

[2] Brian Anderson. Reverse-time diffusion equation models.* Stochastic Process. Appl.*, 12(3):313–326, May 1982. https://www.sciencedirect.com/science/article/pii/0304414982900515.

[3] Kendall Atkinson, Weimin Han, and David Stewart.* Numerical solution of ordinary differential equa- tions*. Wiley, 2009. https://homepage.math.uiowa.edu/~atkinson/papers/NAODE_Book.pdf.

[4] Christopher Bishop.* Pattern Recognition and Machine Learning*. Springer, 2006.

[5] Charles A. Bouman and Gregery T. Buzzard. Generative plug and play: Posterior sampling for inverse problems. In* 2023 59th Annual Allerton Conference on Communication, Control, and Computing (Allerton)*, pages 1–7, 2023. https://arxiv.org/abs/2306.07233.

[6] Robert Brown. A brief account of microscopical observations on the particles contained in the pollen of plants and the general existence of active molecules in organic and inorganic bodies.* Edinburgh New Philosophical Journal*, pages 358–371, 1828.

[7] Stanley H. Chan.* Introduction to Probability for Data Science*. Michigan Publishing, 2021. https: //probability4datascience.com/.

[8] Stanley H. Chan, Xiran Wang, and Omar Elgendy. Plug-and-Play ADMM for image restoration: Fixed point convergence and applications.* IEEE Trans. Computational Imaging*, 3(5):84–98, Mar 2017. https: //arxiv.org/abs/1605.01710.

[9] Hyungjin Chung and Jong Chul Ye. Score-based diffusion models for accelerated mri.* Medical Image Analysis*, 80:102479, 2022. https://arxiv.org/abs/2110.05243.

[10] Peter Dayan, Geoffrey E. Hinton, Radford M. Neal, and Richard S. Zemel. The Helmholtz machine. *Neural Computation*, 7(5):889–904, 1995.

[11] Mauricio Delbracio and Peyman Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration.* Transactions on Machine Learning Research (TMLR)*, 2023. https: //openreview.net/forum?id=VmyFF5lL3F.

[12] Carl Doersch. Tutorial on variational autoencoders, 2016. https://arxiv.org/abs/1606.05908.

[13] Luca Donati. From Chapman-Kolmogorov equation to Master equation and Fokker-Planck equation. https://www.zib.de/userpage/donati/stochastics2023/03/lecture_notes/L03_dCKeq.pdf.

[14] Albert Einstein. On the movement of small particles suspended in stationary liquids required by the molecular-kinetic theory of heat.* Annalen der Physik*, pages 549–560, 1905.

[15] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In* Advances in Neural Information Processing Systems (NeurIPS)*, volume 27, 2014. https://arxiv.org/abs/1406.2661.

[16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In* Advances in Neural Information Processing Systems (NeurIPS)*, 2020. https://arxiv.org/abs/2006.11239.

[17] Jason Hu, Bowen Song, Xiaojian Xu, Liyue Shen, and Jeffrey A. Fessler. Learning image priors through patch-based diffusion models for solving inverse problems, 2024. https://arxiv.org/abs/2406.02462.

[18] Aapo Hyv¨arinen. Estimation of non-normalized statistical models by score matching. *Journal of Machine Learning Research (JMLR)*, 6(24):695–709, 2005. https://jmlr.org/papers/volume6/ hyvarinen05a/hyvarinen05a.pdf.

© 2024 Stanley Chan. All Rights Reserved. 88


---

## Page 89

[19] Zahra Kadkhodaie and Eero P. Simoncelli. Solving linear inverse problems using the prior implicit in a denoiser, 2020. https://arxiv.org/abs/2007.13640.

[20] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion- based generative models. In* Advances in Neural Information Processing Systems (NeurIPS)*, 2022. https://arxiv.org/abs/2206.00364.

[21] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. In* Advances in Neural Information Processing Systems (NeurIPS)*, 2022. https://arxiv.org/abs/ 2201.11793.

[22] Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2021. https://arxiv.org/abs/2107. 00630.

[23] Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. In* International Conference on Learning Representations (ICLR)*, 2014. https://openreview.net/forum?id=33X9fd2-9FyZd.

[24] Diederik P. Kingma and Max Welling. An introduction to variational autoencoders.* Foundations and Trends in Machine Learning*, 12(4):307–392, 2019. https://arxiv.org/abs/1906.02691.

[25] Andrey Kolmogorov. *Foundations of the Theory of Probability*. Dover, 2018. The orig- inal version was published in 1933 in German. https://dn790007.ca.archive.org/0/items/ foundationsofthe00kolm/foundationsofthe00kolm.pdf.

[26] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In* Advances in Neural Information Processing Systems (NeurIPS)*, 2022. https://arxiv.org/abs/2206.00927.

[27] Calvin Luo. Understanding diffusion models: A unified perspective, 2022. https://arxiv.org/abs/ 2208.11970.

[28] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In* IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 14297–14306, 2023. https://arxiv.org/abs/2210.03142.

[29] Gabriel Nagy. MTH 235 differential equations, 2024. https://users.math.msu.edu/users/gnagy/ teaching/ade.pdf.

[30] R. Pawula. Generalizations and extensions of the Fokker-Planck-Kolmogorov equations.* IEEE Trans- actions on Information Theory*, 13(1):33–41, 1967.

[31] L. E. Reichl.* A Modern Course in Statistical Physics*. John Wiley and Sons, Inc, 2 edition, 1998.

[32] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In* Proceedings of International Conference on Machine Learning (ICML)*, pages 1530–1538, 2015. https://arxiv.org/ abs/1505.05770.

[33] Hannes Risken. *The Fokker-Planck Equations: Methods of solutions and applications*. Springer, 2 edition, 1989.

[34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High- resolution image synthesis with latent diffusion models. In* IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 10684–10695, 2022. https://arxiv.org/abs/2112.10752.

[35] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language un- derstanding. In* Advances in Neural Information Processing Systems (NeurIPS)*, volume 35, pages 36479–36494, 2022. https://arxiv.org/abs/2205.11487.

© 2024 Stanley Chan. All Rights Reserved. 89


---

## Page 90

[36] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In *International Conference on Learning Representations (ICLR)*, 2022. https://arxiv.org/abs/2202. 00512.

[37] Yash Sanghvi, Yiheng Chi, and Stanley H. Chan. Kernel diffusion: An alternate approach to blind deconvolution. In* European Conference on Computer Vision (ECCV)*, 2024. https://arxiv.org/abs/ 2312.02319.

[38] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In* Proceedings of International Conference on Machine Learning (ICML)*, volume 37, pages 2256–2265, 2015. https://arxiv.org/abs/1503.03585.

[39] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In* Interna- tional Conference on Learning Representations (ICLR)*, 2023. https://openreview.net/forum?id= St1giarCHLP.

[40] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In* Advances in Neural Information Processing Systems (NeurIPS)*, 2019. https://arxiv.org/abs/ 1907.05600.

[41] Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2020. https://arxiv.org/abs/2006. 09011.

[42] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In* Interna- tional Conference on Learning Representations (ICLR)*, 2021. https://openreview.net/forum?id= PxTIG12RRHS.

[43] Pascal Vincent. A connection between score matching and denoising autoencoders.* Neural Compu- tation*, 23(7):1661–1674, 2011. https://www.iro.umontreal.ca/~vincentp/Publications/smdae_ techreport.pdf.

[44] M. von Smoluchowski. Zur kinetischen theorie der brownschen molekularbewegung und der suspensio- nen.* Annalen der Physik*, pages 756–780, 1906.

[45] Martin J. Wainwright and Michael I. Jordan. Graphical models, exponential families, and variational inference.* Foundations and Trends in Machine Learning*, 1(1-2):1–305, 2008. https://cba.mit.edu/ events/03.11.ASE/docs/Wainwright.1.pdf.

[46] Max Welling and Yee Whye Teh. Bayesian learning via stochastic gradient Langevin dynamics. In *Proceedings of International Conference on Machine Learning (ICML)*, pages 681–686, 6 2011. https: //www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf.

© 2024 Stanley Chan. All Rights Reserved. 90


---
