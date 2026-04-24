Diffusion Model for Graph Inverse Problems: Towards
Effective Source Localization on Complex Networks

Xin Yan1,∗Hui Fang2,∗ Qiang He1,†
1College of Medicine and Biological Information Engineering, Northeastern University, China
2RIIS & SIME, Shanghai University of Finance and Economics, China
yanx_in@outlook.com,fang.hui@mail.shufe.edu.cn,heqiangcai@gmail.com

Abstract

Information diffusion problems, such as the spread of epidemics or rumors, are
widespread in society. The inverse problems of graph diffusion, which involve
locating the sources and identifying the paths of diffusion based on currently
observed diffusion graphs, are crucial to controlling the spread of information.
The problem of localizing the source of diffusion is highly ill-posed, presenting
a major obstacle in accurately assessing the uncertainty involved. Besides, while
comprehending how information diffuses through a graph is crucial, there is a
scarcity of research on reconstructing the paths of information propagation. To
tackle these challenges, we propose a probabilistic model called DDMSL (Discrete
Diffusion Model for Source Localization). Our approach is based on the natural
diffusion process of information propagation over complex networks, which can
be formulated using a message-passing function. First, we model the forward
diffusion of information using Markov chains. Then, we design a reversible
residual network to construct a denoising-diffusion model in discrete space for both
source localization and reconstruction of information diffusion paths. We provide
rigorous theoretical guarantees for DDMSL and demonstrate its effectiveness
through extensive experiments on five real-world datasets.

1

Introduction

Information diffusion is a pervasive phenomenon in our daily lives. Data scientists have conducted ex-
tensive research on information diffusion along the direction of entropy increase, such as maximizing
the influence of nodes in social networks [15, 3], and developing control policies to curb the scale of
epidemics in disease transmission networks [53]. However, merely comprehending the mechanism of
forward diffusion is insufficient. When destructive information spreads across a network, it can cause
huge damage on the entire system. For example, the rapid spread of rumors in social networks can
cause harm to society [54], and the spread of computer viruses on the Internet can paralyze a system
consisting of millions of users [1]. Additionally, pandemics such as SARS and COVID-19 in human
interaction networks pose serious challenges to human health and social functioning [31]. Therefore,
accurately identifying the sources of transmission and cutting off their possible transmission paths in
time is crucial. This can help limit the spread of negative information and maintain the stability of
the network.

Common information dissemination models like SIR (Susceptible-Infected-Recovered) and SI
(Susceptible-Infected) [19] are subject to uncertainties during the process of information diffu-
sion. As illustrated in Figure 1, the diffusion graph of information should follow a non-explicit
distribution, implying that a single diffusion source may correspond to multiple different diffusion

∗Equal contribution. †Corresponding author.

37th Conference on Neural Information Processing Systems (NeurIPS 2023).

graphs, and a diffusion graph may have multiple different diffusion sources. Therefore, the inverse
problem of information diffusion is underdetermined.

Figure 1: The SIR diffusion process of information over complex networks. Distinct sources may
produce identical infection graphs (xT ), and initiating forward diffusion from xT may lead to
different diffusion outcomes. The distribution of sources and forward diffusion can be represented by
probability models P −1 and P , respectively.

Previous source localization research has primarily relied on manually formulating rules to filter
source nodes, such as using Jordan centrality [24, 26] to locate multiple source nodes in SI model, or
unbiased intermediary centrality to identify source nodes in SIR model [49]. Besides, IVGD [43]
exploited fixed point theorem for source localization of IC model [18]. However, these algorithms
impose relatively strict requirements on information diffusion patterns, which could hardly
be satisfied in real-world applications. To address this issue, some algorithms [38, 36] have been
proposed using graph neural networks (GNN) [35] to learn source nodes under different diffusion
models. Unfortunately, most deep learning-based algorithms ignore the underdeterminacy of infor-
mation propagation and attempt to establish a direct mapping between observed data and source
nodes. As a result, inference results are often deterministic and fail to quantify the uncertainty
of source localization. In addition to source localization, it is equally important to recover the
possible propagation paths of information. For example, when an infectious disease breaks out,
it is necessary to promptly trace the virus transmission trajectory and corresponding close contacts.
However, in reality, this can only be achieved by analyzing the genetic evolution tree of the virus
strain [40], which is both time-consuming and labor-intensive. In summary, there is a current
dearth of methods that can simultaneously locate information sources and restore information
propagation paths.

Inspired by diffusion phenomena, the diffusion denoising model [16] has garnered significant achieve-
ments in the realm of image generation. It effectively restores the original image within a noise
distribution through gradual refinement. This process bears resemblance to the challenges encountered
in information reconstruction and diffusion, demanding our attention. Throughout the propagation
process, the source information continually undergoes blurring, presenting an exceedingly arduous
task of restoring the original source node. In this scenario, employing a diffusion model for the
reverse recovery of the entire information dissemination process proves highly appropriate. However,
using diffusion models to solve inverse problems in graphs presents several challenges: (1) Infor-
mation diffusion occurs on non-Euclidean space graphs, making learning discrete non-Euclidean
data difficult; (2) Diffusion over graphs is governed by information propagation rules that operate
in non-Euclidean spaces, making it challenging to establish both the forward process and reverse
inference process of information diffusion within the network; and (3) Most existing algorithms only
work for specific propagation patterns, suffering from the problem of model generalization.
To address the aforementioned challenges, we propose a new framework, DDMSL2(Discrete Diffusion
Model for Source Localization). DDMSL excels at source localization and reconstructing the evolu-
tion of diffusion, showing promising results across different propagation patterns. Our contributions
can be summarized as follows:

2The code for DDMSL is available at https://github.com/Ashenone2/DDMSL.

2

P-1……ABCDEFGHIJKABCDEFGHIJKABCDEFGHIJKABCDEFGHIJKABCDEFGHIJKABCDEFGHIJKABCDEFGHIJKABCDEFGHIJKABCDEFGHIJK)|(21xxp)|(21xxp)|(21xxp)|(21xxp)|(21xxp)|(32xxp)|(32xxp)|(1TTxxp−)|(1TTxxp−ABCDEFGHIJKABCDEFGHIJKt=1t=2t=TPABCDEFGHIJKABCDEFGHIJKABCDEFGHIJKt=T+1SusceptiblesInfectivesRecoveredSusceptiblesInfectivesRecovered1Q2QTQ1+TQ1Q2QTQ1+TQ• We model the information diffusion process using Markov chains and demonstrate that at each
moment, the state transition matrix converges, enabling us to reconstruct the information diffusion
paths from any time point.

• We design a residual network block based on graph convolutional networks to approximate the

information diffusion process and provide a theoretical proof for the model’s reversibility.

• We propose an end-to-end framework being capable of both reconstructing the evolution of
information diffusion and source localization. To the best of our knowledge, this is the first study
to employ denoising diffusion models for solving graph inverse problems. Extensive experiments
on five real-world datasets demonstrate its effectiveness.

2 Related work

Diffusion models for inverse problems. The diffusion denoising model [16, 29] is currently one
of the best available deep generative models, which has been extensively applied in image-related
inverse problems [33, 23, 34]. Previous studies have extended DP3Ms [2] to discrete spaces and
used this work as a foundation for developing multiple generative diffusion models for graph data
[14, 42]. Additionally, Stevens et al. [39] and Chung et al. [7] demonstrated that the diffusion model
can solve nonlinear inverse problems. These works collectively demonstrate the effectiveness of
diffusion models across different domains in addressing inverse problems.

Localization of information sources on the graph. To facilitate the localization of infection sources,
various algorithms have been proposed [25, 52, 44, 9]. Early algorithms focused on screening source
nodes by employing feature engineering. For instance, Luo et al. [25] identified the diffusion source
by the centrality of the diffusion subgraphs, while Prakash et al. [30] and Zhu et al. [52] respectively
developed NETSLEUTH and OJC algorithms based on the minimum description length of nodes and
the shortest path. Wang et al. [44] designed a series of label propagation algorithms based on the
aggregation process of information diffusion, on which Dong et al. [9] further improved using GCN
[20]. Recently, the proposed reversible perception algorithm IVGD based on graph diffusion [43]
has been applied to the IC model. However, these algorithms generally have strict limitations on
diffusion patterns or network structures. DDMIX [8] was the first algorithm to use a generative
model for reconstructing the dissemination paths of information. It leverages a VAE (variational
autoencoder) to learn the state of nodes across all time steps; however, as the number of time steps
increases, the solution space of the dissemination path becomes significantly more complex, leading
to low accuracy in source localization. Ling et al. later introduced SLVAE [22], which is also based
on VAE and can efficiently localize the source node but cannot directly reconstruct the propagation
paths. Current generative model-based approaches for source localization cannot simultaneously
handle the problems of source localization and reconstructing diffusion paths.

3 Methodology

0 , xs2

3.1 Problem definition
We consider an undirected graph G = (V, E) where V = (cid:8)x1, x2, . . . , xN (cid:9) is the node set and E
is the edge set. Let S = {xs1
0 } denote the set of initial diffusion source nodes at t = 0.
The source nodes undergo diffusion on the graph G for T time steps, governed by the diffusion pattern
g(·), and the set of node states at time step t is denoted by Xt = (cid:8)x1
t ∈ {0, 1}M
(e.g., if g(·) is the SIR model, then M ∈ {S, I, R}, where S, I, R denote susceptible, infected and
recovered state, respectively). If the node i ∈ S, then xi
0 = 0. We define the
research problem as finding the intermediate state X = {X0, X1, . . . , XT−1} that maximizes the
likelihood function X∗ = argmaxP {XT|X,G}

0 = 1, otherwise xi

, given the observed XT.

0 , . . . xsm

(cid:9) with xi

t , . . . , xN
t

X

3.2

Information diffusion in discrete spaces

Susceptible-Infected-Recovered (SIR) and Susceptible-Infected (SI) models [19] are commonly used
to model diffusion phenomena in nature, such as the spread of epidemics [51] and rumors [50]. In this
paper, we will demonstrate our approach using the SIR model. The SIR model categorizes node states
into susceptible (S), infected (I), and recovered (R) states. The S state represents individuals who are

3

(1)

(2)

(3)

susceptible to infection, while the I state represents those who have been infected, and the R state
indicates recovery from the infected state. The transition between these three states is irreversible.
We assume that all nodes on the graph are homogeneous and follow the same transition process for
the different states of the SIR model3:
(cid:40)

(cid:16)

P (cid:0)xi
P (cid:0)xi

t+1 = I | xi
t+1 = R | xi

t = S(cid:1) = 1 − (cid:81)
t = I(cid:1) = γi
R(t)

j

1 − βj

(cid:17)
I (t)AijIj(t)

And, the SIR diffusion process can be represented using a state transfer matrix:



1 − βi
0
0
t denotes the state transfer matrix of node i at moment t, and (cid:2)Qi

βi
I (t)
1 − γi
R(t) γi
0

0
R(t)
1

I (t)

t =

Qi







where Qi
of transferring from state u to state v4. βi
S(t), P i
at moment t, respectively. Let [P i
I (t) and γi
states S, I, and R at time t, then βi
I (t + 1) − P i
P i
P i
R(t + 1) − P i
P i
P i




γi
R(t) =

βi
I (t) =

I (t)

(cid:3)
uv denotes the probability
R(t) denote the infection rate and recovery rate
R(t)] to be the probabilities of node i being in three

I (t) and γi
I (t), P i

t

R(t) can be calculated by the following equation:
I (t)(1 − γi
S(t)
R(t)

S(t + 1)
P i
S(t)

R(t))

= 1 −

P i

Theorem 3.1 Given graph G and the infected seed set S, it can be determined that the state transfer
matrix Qi

t for the SIR diffusion process on G converges at all times.

Sketch of Proof. Given G and S , the initial state of information diffusion can be determined. Based
on the SIR propagation rule, an iterative equation can be constructed for the state distribution of the
nodes at any given time. Finally, Qi

t can be calculated according to Equations 2 and 3.

Referring to Equations 2 and 3, we can readily demonstrate the convergence of the state transition
matrix Qi
t at every moment, rendering the discrete diffusion model a practicable solution [2]. The
proof is shown in Theorem3.1.

3.3 Discrete diffusion model for source localization

3.3.1 Forward process

We represent the state of node i by an one-hot vector xi
at time t is written as:

t ∈ R1×M , and the state distribution of node i

q (cid:0)xi

(cid:1) = xi

t | xi

txi
t
Information diffusion is a Markov process that allows for inference of the node state at any given
moment based on the initial state:
t
(cid:89)

t; p = xi

t−1Qi
t

t−1Qi

(cid:88)

t−1

(4)

T

q (cid:0)xi

k | xi

k−1

(cid:1) = xi

¯Qi

txi
t

0

∼ Cat (cid:0)xi

t; p = xi
0

¯Qi
t

(cid:1)

q (cid:0)xi

t | xi
0

(cid:1) =

(5)

T

∼ Cat (cid:0)xi

(cid:1)

xi

1:t−1

k=1

3.3.2 Reverse process

Reconstructing the information diffusion process requires backward inference of the forward process,
which can be achieved through Bayesian formulation5:

q (cid:0)xi

t−1 | xi

t, xi
0

(cid:1) ∼ Cat


xi

t−1; p =

(cid:16)

xi

tQi
t

T (cid:17)

⊙ (cid:0)xi

0

¯Qi

t−1

xi
0

¯Qi

txi
t

T



(cid:1)



(6)

3A denotes the graph’s adjacency matrix, and Ij(t) represents whether neighbor j is in an infected state at t.
4In the SIR model, u, v ∈ {1, 2, 3}, representing the three states: S, I and R, respectively.
5Please refer to Appendix B for the formula derivation of forward and backward processes.

4

(cid:1) becomes in-
In the absence of knowledge about xi
tractable to compute directly. As a result, we must approximate this distribution using other methods.
Continuing with Bayes’ theorem, we can express the posterior distribution as follows:
(cid:80)

0, the posterior distribution q (cid:0)xi

t−1 | xi

t, xi
0

(cid:1)

q (cid:0)xi

t−1 | xi

t, xi
0

(cid:1) = q (cid:0)xi

t−1 | xi
t

(cid:1) =

xi
0

q (cid:0)xi

t−1, xi
(cid:1)

q (cid:0)xi

t

t, xi
0

= E

q(xi

(cid:0)xi

t−1 | xi
t

(cid:1)

t−1 | xi

t)q (cid:0)xi
0|xi
(cid:1) and estimate q (cid:0)xi

t, xi
0

(7)

0 | xi
t

(cid:1).

We utilize a neural network model (nnθ) to learn about pθ
The detailed design of nnθ will be elaborated later in Section 3.4.
t, xi
0

t−1 | xi
t

t−1 | xi
t

(cid:1) ≈ pθ

(cid:1) = E

q (cid:0)xi

(cid:0)xi

(cid:1)

0|xi

0∼pθ(xi
xi
q (cid:0)xi
t | xi

t−1

t)q (cid:0)xi
(cid:1) (cid:104)(cid:80)
j q

t−1 | xi
(cid:16)

=

xi
q (cid:0)xi

t−1 | xi
0
(cid:1)
t | xi
0

(j)(cid:17)

(cid:16)

xi
0

pθ

(j)

| xi
t

(cid:17)(cid:105)

(8)

0 and then use this prediction to learn the distribution of q (cid:0)xi

(cid:1). This
Similar to [2], we predict xi
approach offers several benefits. First, it leverages prior knowledge by exploiting the fact that at t = 0
there are only two states: susceptible and infected. This makes nnθ easier to train. Second, locating
xi
0 is a key target of our task. Finally, predicting xi
0 can help constrain errors in the information
reconstruction process (Refer to Appendix C).

0 | xi
t

vb =E
Li

0)
q(xi




DKL
(cid:124)

(cid:2)q (cid:0)xi

T | xi
0
(cid:123)(cid:122)
LT

(cid:1) ∥p (cid:0)xi

T

(cid:1)(cid:3)

+

(cid:125)

T
(cid:88)

E

t=2

(cid:124)

q(xi

t|xi

0)[DKL

(cid:2)q (cid:0)xi

t, xi
0

t−1 | xi
(cid:123)(cid:122)
Lt−1

(cid:1) ∥pθ

(cid:0)xi

t−1 | xi
t

(cid:1)(cid:3)]
(cid:125)

− E
(cid:124)

q(xi

0)
1|xi

(cid:0)xi

0 | xi
1

(cid:2)log pθ
(cid:123)(cid:122)
L0



(cid:1)(cid:3)




(cid:125)

(9)

Equation 9 presents the variational lower bound loss function of the denoising diffusion model [16],
where DKL represents relative entropy. Furthermore, as highlighted in our earlier discussion, it is
crucial for us to supervise xi
0 directly. As a result, we arrive at the simplified variational lower bound
loss (Lsimple), which is denoted as:

Lsimple =

(cid:16)

(cid:80)N

i=1

1
N

vb + E
Li

0)
q(xi

E

q(xi

0)
t|xi

(cid:2)− log (cid:101)pθ

(cid:0)xi

0 | xi
t

(cid:1)(cid:3)(cid:17)

(10)

3.3.3 Supervision of information propagation rules

Our reconstructed information diffusion must adhere to the propagation rules of model g(·). During
early training, the predictions of nnθ for xi
t−1 may not be accurate enough, which could lead to
violations of the propagation rules. For example, using the SIR model, if the state of node i at time t
is classified as I, then the state of i at time t − 1 can only be either S or I. However, the prediction
(cid:1).
result of nnθ may erroneously output state R, resulting in the failure to calculate q (cid:0)xi
To prevent violations of the propagation rules, we can take two measures: (1) In cases where the
inference result of nnθ violates the propagation rule, q(xi
0) should be set to [1, 0, 0]. and
(2) To supervise the training of nnθ, the propagation rule loss function Lconstrain = Lconstrain1 +
Lconstrain2 is introduced. The detailed derivations of (1) and (2) are shown in Appendix C.

t−1 | xi

t−1|xi

t, xi
0

t, xi

(cid:40) Lconstrain1 = Relu (Xt−1 − (A + I)X0)
(cid:17)(cid:13)
2
(cid:13)
(cid:13)
2

t−1 − X(i)

Lconstrain2 =

(cid:13)
(cid:13)
(cid:13)max

0, X(j)

t−1

(cid:16)

, ∀X(i)

0 ⊇ X(j)

0

(11)

In order to evaluate Equation 11, we must first sample Xt−1 from the discrete probability distribution
computed in Equation 8. To ensure gradient preservation during the sampling process, we employ
the Gumbel-Sof tmax technique [17]: Xt−1 ∼ Gumbel-Sof tmax(q(xi
0)). Equation 11
demonstrates that each node’s contribution to the information diffusion process is non-negative. In
summary, the constrained loss function is Lconstrain= = Lconstrain1 + Lconstrain2, and the loss
function of DDMSL is:

t−1|xi

t, xi

Loss = Lsimple + Lconstrain

(12)

5

3.4 Design of the inference model

In this section, we discuss the
design of the nnθ in detail. Let
(cid:9) and YT =
X0 = (cid:8)x1
0, x2
0 . . . xN
0
(cid:8)x1
(cid:9). The diffusion
T . . . , xN
T , x2
T
of information on complex net-
works can be modeled using the
following processes [46, 43]:

X0

fw−→ ξ F−→ YT

(13)

Figure 2: Multi-layer reversible residual network.

Within this framework, fw and F
serve as the feature vector construc-
tion and feature propagation func-
tions, respectively. The relationship between YT and X0 be expressed as:YT = P(X0) =
F(fw(X0)). Our research objective focuses on the inverse problem of the SIR or SI diffusion
model, which entails a large number of stochastic processes. Thus, when F = RD, solving P−1 can
be extremely challenging. In the following subsections, we will explore how to construct suitable
functions fw and F to facilitate the calculation of P−1.

3.4.1 Relationship between GCN and SIR diffusion models

According to [37], diffusion models such as SIR are analogous to message passing neural networks
(MPNNs [11]), where each node’s state is only updated based on the states of its neighboring nodes.
However, the operational architecture must be designed to enable each node to aggregate both its
own features and those of its neighbors, and update its state via nonlinear activation as shown in
Equation 26. Therefore, we propose using MPNNs to remodel the SIR diffusion process. Denoting
Ω(t) ≡ P (xi
P i
t = Ω) as the probability that node i is in state Ω ∈ {S, I, R} at time t in the SIR
diffusion process. It can be observed that this process is structurally equivalent to:






mi
P i
h(P i

Ω(t + 1) = (cid:80)
Ω(t + 1) = Ut

j∈N (i) Mt
(cid:0)h(P i
= σ (cid:0)WΩP i

Ω(t))

Ω(t)), mi
Ω(t) + bΩ

h(P i

Ω(t)), h(P j
Ω(t + 1)(cid:1)
(cid:1)

(cid:16)

Ω(t)), eij

(cid:17)

(14)

where Mt is the aggregate function and Ut represents the node status update function. σ(·) is a
nonlinear activation function. eij represents the edge between nodes i and j. Additionally, since
h(P i
Ω(t)) already contains nonlinear transformations, more complex forms of transformation are not
necessary and Ut can be defined as: Ut(a, b) = a + b. To sum up, Equation 14 can be simplified as:

P i
Ω(t + 1) = h(P i

Ω(t)) + σ







h(P j

Ω(t))



(cid:88)

j∈N (i)

(15)

We can achieve this using a residual block composed of graph convolutional networks, which allows
us to easily fit the SIR model:

i) with U ∈ RC×M


h(0)
i,t = SN(Uxt

g(h) = σg(D−1/2AD−1/2 · h · w + b)

i,t = h(l)
h(l+1)

i,t + σ

h(l)
i,t

BN

SN

(cid:16)

(cid:16)

(cid:16)

(cid:16)

g

(cid:17)(cid:17)(cid:17)(cid:17)

(16)

Among them, U is a linear transformation, h(l)
i,t denotes the representation of the l-th layer of the
network, D is the degree matrix, SN(·) and BN(·) represent spectral normalization and batch
normalization, respectively. σ(·) and σg(·) are nonlinear activation functions6. In fact, fw and F7 in
Equation 13 are the SN (U(·)) and residual blocks in Equation 16.

6We set σ(·) and σg(·) to M ish and LeakyReLu respectively.
7F = ξ + fθ(ξ), where fθ(ξ) = σ (SN (BN (ξ))).

6

x1x2…SII…IRIII…RR……YTX0Dense…Residual Block × nGCNBNssGCNBNsGCNBNssResidual BlockGCNBNsResidual BlockDPDPFCSigmoid………SxNSIIISxNSIII…Residual BlockF3.4.2 Reversibility of residual blocks

We now discuss the invertibility of Equation 13 as reconstructed from Equation 16. Equation 13 can
be written as:

(cid:26) SN (U(X0)) = fw(X0) = ξ;

ξ + σ (SN (BN (g(ξ)))) = ξ + fθ(ξ) = F(ξ) = YT.

(17)

Lemma 3.1 Denoting the Lipschitz constants of fw and fθ to be Lw and Lθ, then Lw < 1 and
Lθ < 1.

Sketch of Proof. Since fw and BN (g(ξ)) are spectrally normalized, their Lipschitz coefficients are
less than 1 [28]. Additionally, since the σ(·) is set to M ish(·), it can be determined that the Lipschitz
constant of their composite function (i.e., fθ) is less than 1.
Theorem 3.2 If Lw < 1 and Lθ < 1, then YT = F(fw(X0)) is reversible.

Sketch of Proof. YT = F(fw(X0)) can be written as

(cid:26) X0 = ξ + X0 − fw(X0)

ξ = YT − fθ(ξ)

. Since

fw < 1 and fθ < 1, constructing the iteration according to the Banach fixed point theorem [4],
YT = F(fw(X0)) is invertible.

In practical applications, we typically use multiple residual blocks to form a residual network. This
approach is employed to improve the receptive field of the GCN and alleviate the problem of node
smoothing caused by multi-layer graph convolution. The multi-layer reversible residual network that
we designed is depicted in Figure 2, where DP represents dropout. If the dropout rate is r and the
i-th layer residual block is Fi, then we have Fi(ξ) = DP(ξ + fw(ξ)).
Theorem 3.3 If dropout-rate = 0.5, Lw < 1, and Lθ < 1, the residual network YT =
(F1◦F2◦ . . . ◦Fn)(fw(X0)) is reversible.

Sketch of Proof. Denoting this multilayer residual network as F , the upper bound of LF is the
product of the Lipschitz constants of each function [13]. Additionally, with a dropout rate of r, the
Lipschitz constants of the functions will be limited to (1 − r) times their original values. Using these
two conclusions, we can calculate the upper bound of the Lipschitz constant of a multilayer reversible
network with LF ≤ 1 when r = 0.5.

4 Experiments

4.1 Datasets and evaluation metrics

Datasets. The diffusion of information occurs in a broad range of network types, and DDMSL was
evaluated on five distinct realistic datasets: Karate [48], Jazz [12], Cora-ML [27], Power Grid
[45], and PGP [5]. The detailed parameters of these networks are provided in the Appendix F.2.
Following previous research [43, 9, 22], we conducted SIR and SI diffusion simulations on each
dataset by randomly selecting 10% of the nodes as source nodes at the initial moment, and stopping
the simulation when approximately 50% of the nodes were infected8 (For PGP dataset, simulation
stopped at 30% infection rate). We randomly divided each generated dataset into a training set, a
validation set and a test set in the ratio of 8 : 1 : 1.

Evaluation Metrics. Our objective consists of two components: source localization and information
diffusion paths recovery. The source localization task is a binary classification task, and thus evaluated
using four metrics: Precision (PR), which denotes the proportion of nodes predicted as sources that
are true sources; Recall (RE), which represents the proportion of actual source nodes that are correctly
predicted; F1 score, the harmonic mean of PR and RE; and ROC-AUC (AUC), quantifying the model’s
ability to classify accurately. To evaluate the performance of the recovering information diffusion
(cid:13)
2
paths, we adopted the Mean Squared Error (MSE) error in [8]: M SE = 1
(cid:13)
,
(cid:13)
N T
where ˆXt is the ground truth. This metric is solely computed for infected nodes, with nodes in the
recovered and susceptible states being marked as 0.

(cid:13)
(cid:13)Xt − ˆXt
(cid:13)

(cid:80)T −1
t=0

8While some previous studies [9, 44] commenced source localization when the infection rate reached 30%,

we set it to 50% to consider more complex scenarios.

7

4.2 Baseline algorithms and experimental settings

Six algorithms for source location were selected to compare with DDMSL: DDMIX [8] employs a
Variational Autoencoder (VAE) to reconstruct the outbreak propagation process; LPSI [44] aggregates
information to learn source detection through label propagation; GCNSI [9] uses graph convolu-
tional network to learn manually formulated node features, enabling the discrimination of multiple
source nodes; OJC [52] locates the source node by minimising the eccentricity of the infection;
NETSLEUTH [30] filters multi-source nodes based on minimum description distance, but works
solely on the SI diffusion model; and SLVAE [22] deploys generative models to learn the distribution
of source nodes. Appendix F.1 provides the experimental settings and tuned hyperparameters for
these algorithms, while implementation details on DDMSL are shown in Appendix E.

4.3 Experimental results

while comparing DDMSL with other algorithms on different diffusion models, we have focused on the
widely used SIR and SI models, which are effective in capturing the dynamics of various information
diffusion processes found in nature. It is noteworthy that DDMSL’s flexibility and robustness enable
it to handle real-world scenarios where the diffusion process may undergo significant variations by
simply designing the corresponding state transfer matrices9.

Table 1: Performance comparison (mean of five rounds) of source localization under the SIR model.
The best results are highlighted in bold, while the underlined indicates the best of the five baselines.
Statistical significance from paired t-test depicts: ∗∗∗, ∗∗, ∗ for p-value < 0.01, 0.05, 0.1 respectively.

Karate

Jazz

Cora Ml

Power Grid

PGP

PR

RE

F1 AUC PR RE

F1 AUC PR RE

Methods
F1 AUC
DDMSL 0.708 0.736 0.722 0.853 0.817 0.881 0.848 0.930 0.894 0.867 0.880 0.928 0.833 0.879 0.855 0.930 0.856 0.903 0.879 0.943
0.275 0.410 0.329 0.671 0.301 0.363 0.330 0.641 0.247 0.273 0.260 0.591 0.165 0.182 0.173 0.540 0.554 0.543 0.549 0.748
GCNSI
0.211 0.393 0.274 0.646 0.400 0.098 0.158 0.543 0.246 0.026 0.048 0.509 0.193 0.012 0.022 0.503 0.518 0.437 0.474 0.696
LPSI
0.552 0.400 0.464 0.696 0.750 0.576 0.651 0.778 0.815 0.721 0.765 0.852 0.908 0.719 0.803 0.856 0.817 0.721 0.766 0.851
SLVAE
0.178 0.265 0.213 0.594 0.147 0.161 0.154 0.535 0.114 0.114 0.114 0.508 0.109 0.109 0.109 0.505 0.128 0.128 0.128 0.516
OJC
DDMIX 0.289 0.234 0.258 0.308 0.215 0.197 0.205 0.238 0.162 0.273 0.204 0.250 0.333 0.253 0.287 0.346 0.172 0.194 0.182 0.213
Improve. 28.3% 84.3% 55.7% 22.5% 8.9% 52.9% 30.1% 19.6% 9.7% 20.2% 15.0% 8.9% -8.3% 22.2% 6.6% 8.7% 4.8% 25.3% 14.8% 10.8%

F1 AUC PR RE

F1 AUC PR RE

Significance ∗∗∗

∗∗∗

∗∗∗

∗∗∗

∗∗

∗∗∗

∗∗

∗

∗∗∗

∗

∗∗

∗

∗∗∗

∗∗∗

∗∗∗

∗∗∗

∗∗∗

∗∗∗

∗∗∗

∗∗∗

Table 2: Performance comparison (mean of five rounds) of source localization under the SI model.
The best results are highlighted in bold, while the underlined indicates the best of the six baselines.
Statistical significance from paired t-test depicts: ∗∗∗, ∗∗, ∗ for p-value < 0.01, 0.05, 0.1 respectively.

Karate

Jazz

Cora Ml

Power Grid

PGP

PR

RE

RE

F1 AUC PR

F1 AUC PR RE

F1 AUC
Methods
DDMSL 0.706 0.980 0.798 0.972 0.782 0.853 0.813 0.914 0.790 0.908 0.845 0.941 0.763 0.966 0.852 0.966 0.754 0.887 0.815 0.928
0.357 0.456 0.401 0.687 0.366 0.426 0.394 0.676 0.321 0.354 0.337 0.636 0.335 0.325 0.330 0.639 0.487 0.370 0.421 0.665
GCNSI
0.339 0.414 0.351 0.681 0.474 0.097 0.156 0.544 0.494 0.207 0.291 0.592 0.343 0.277 0.306 0.609 0.453 0.284 0.349 0.623
LPSI
0.591 0.477 0.503 0.733 0.888 0.579 0.691 0.785 0.841 0.728 0.780 0.856 0.815 0.780 0.797 0.880 0.844 0.633 0.723 0.810
SLVAE
0.267 0.396 0.318 0.663 0.120 0.127 0.123 0.517 0.125 0.125 0.125 0.514 0.178 0.178 0.178 0.544 0.118 0.118 0.118 0.510
OJC
DDMIX 0.253 0.377 0.303 0.654 0.244 0.133 0.172 0.212 0.220 0.222 0.221 0.247 0.345 0.235 0.280 0.340 0.189 0.186 0.187 0.207
NetSleuth 0.239 0.339 0.279 0.634 0.216 0.252 0.233 0.580 0.217 0.229 0.223 0.569 0.206 0.216 0.211 0.562 0.200 0.210 0.205 0.558
Improve. 19.4% 105.5% 58.5% 32.5% -11.9% 47.3% 17.6% 16.3% -6.0% 24.8% 8.3% 9.9% -6.3% 23.8% 7.0% 9.8% -10.6% 40.2% 12.7% 14.5%

F1 AUC PR RE

F1 AUC PR

RE

Significance ∗∗∗

∗∗∗

∗∗∗

∗∗∗

∗∗∗

∗∗∗

∗∗∗

∗∗∗

∗∗∗

∗∗∗

∗∗∗

∗∗∗

∗

∗

∗∗

∗∗

∗∗∗

∗∗∗

∗∗∗

∗∗∗

Source Location. Table 1 shows the performance of DDMSL in SIR diffusion mode, where it
outperforms all baseline algorithms regarding almost all metrics. Notably, DDMSL leads the best
of the baseline algorithms by approximately 25% and 14% on F1 score and AUC, respectively.
Meanwhile, Table 2 shows similar results in the SI diffusion mode. While SLVAE has a higher
precision than DDMSL, DDMSL surpasses SLVAE by approximately 21% and 17% in terms of F1
score and AUC, indicating that DDMSL is more accurate in identifying source nodes. Moreover, the
performance of DDMSL varies across datasets and is mainly influenced by the network’s typologies
(Refer to Table 6) and the scale of information diffusion. The performance of DDMSL in solving
the inverse problem generally has a negative correlation with the average clustering coefficient and
density of the network. The larger these two parameters are, the more complex the scenario of
information diffusion is, and the accuracy of DDMSL in detecting source points decreases, which
aligns with our intuition.

9Please refer to Appendix E for measuring the state transition matrices for different propagation models.

8

(a) MSE reconstruction error in SIR mode.

(b) MSE reconstruction error in SI mode.

Figure 3: Comparisons of DDMSL and DDMIX on reconstructed information diffusion processes.

Reconstructing information diffusion paths. Both DDMSL and DDMIX can reconstruct the
evolution of information diffusion, but DDMIX can only recover the states of susceptible nodes
(S) and infected nodes (I), whereas DDMSL can reconstruct nodes of all states. To facilitate fair
comparison, we calculate MSE loss for node reconstruction in states S and I, as shown in Figure 3.
DDMSL consistently outperforms DDMIX on all datasets, especially in SI diffusion mode where
DDMSL’s average reconstruction loss is 69% lower than DDMIX’s. Unlike DDMIX, which recovers
the entire graph state, DDMSL allows for fine-grained node-level reconstruction.

Additional experiments. DDMSL represents a source detection algorithm relying on diffusion
models, necessitating an assessment of its aptitude for generalization. Concurrently, we assessed its
computational complexity while also incorporating experiments utilizing two real diffusion datasets.
We have documented the results of these experiments in Appendix G.

Visualization. To further showcase the efficacy of DDMSL in solving the inverse problem of
information diffusion on complex networks, we designed a visualization that demonstrates source
localization and the propagation evolution process. Due to space constraints, we have presented the
visualization in Appendix H.

4.4 Ablation study

Table 3: The results of the ablation study.

Karate

Jazz

Cora Ml

PGP

PR RE

F1 AUC PR RE

Methods
F1 AUC
DDMSL 0.708 0.736 0.722 0.853 0.817 0.881 0.848 0.930 0.894 0.867 0.880 0.928 0.833 0.879 0.855 0.930 0.856 0.903 0.879 0.943
DDMSL(a) 0.264 0.910 0.379 0.827 0.254 0.683 0.367 0.741 0.277 0.626 0.384 0.722 0.459 0.605 0.522 0.706 0.366 0.857 0.513 0.846
DDMSL(b) 0.655 0.850 0.718 0.860 0.648 0.822 0.723 0.889 0.834 0.856 0.845 0.915 0.818 0.875 0.845 0.919 0.811 0.888 0.848 0.922
DDMSL(c) 0.339 0.185 0.236 0.592 0.931 0.111 0.1943 0.556 0.817 0.043 0.08 0.521 0.998 0.2756 0.432 0.638 0.997 0.656 0.792 0.823

F1 AUC PR RE

F1 AUC PR RE

F1 AUC PR

Power Grid
RE

To evaluate the effectiveness of each component,
we conducted ablation experiments on DDMSL
in SIR diffusion mode. One essential compo-
nent of DDMSL is the reversible residual net-
work. We removed this component and replaced
each reversible residual block with a GCN net-
work, denoted as DDMSL(a). Additionally,
DDMSL is also supervised by Lconstarin on
the propagation rules. The model without the
propagation rule supervision module is denoted
as DDMSL(b). Finally, to assess the impact
of GCN (Graph Convolutional Network) in re-
versible residual blocks, DDMSL(c) represents
the model after removing the GCN modules
from residual blocks. We ran the three vari-
ants on five datasets and compared the results,

Figure 4: MSE error in ablations.

9

0.0570.1160.1170.1230.1020.1110.1750.1370.1510.253KarateJazzCora mlPower GridPGP0.000.050.100.150.200.250.300.35MSE DDMIX DDMSL0.0740.1160.1150.1160.1120.0980.1220.1730.2010.299KarateJazzCora mlPower GridPGP0.000.050.100.150.200.250.300.35MSE DDMIX DDMSLKarateJazzCora MlPower GridPGP(cid:2)(cid:1)(cid:2)(cid:2)(cid:2)(cid:1)(cid:2)(cid:6)(cid:2)(cid:1)(cid:3)(cid:2)(cid:2)(cid:1)(cid:3)(cid:6)(cid:2)(cid:1)(cid:4)(cid:2)(cid:2)(cid:1)(cid:4)(cid:6)(cid:2)(cid:1)(cid:5)(cid:2)(cid:2)(cid:1)(cid:5)(cid:6)MSE DDMSL DDMSL(a) DDMSL(b) DDMSL(c)which are shown in Table 3 and Figure 4. As can be seen in table 3, we have these observations: (1)
removing the residual module (DDMSL(a)) leads to a significant degradation in model performance,
with the F1 score and AUC decreasing by 49% and 17%, respectively. Although the GCN network is
also a form of MPNNs, we noticed that the output of the inference network composed of GCNs were
very similar, indicating node feature oversmoothing, and further highlighting the effectiveness of the
reversible residual network module; (2) Lconstraint is also effective, which contributes to 5% and
2% performance improvement in terms of F1 score and AUC, respectively. Similar results can be
obtained in Figure 4 regarding the task of reconstructing the information diffusion processes; and
(3) upon the drop of the GCNs from the reversible residual network (DDMSL(c)), a noteworthy
deterioration model performance was observed, manifesting as a substantial decrease in F1 scores
and AUC by 59% and 32% respectively. This compellingly signifies the indispensable role of GCN
in acquiring the diffusion mode of the SIR model through the process of learning. Similar results can
be obtained in Figure 4 regarding the task of reconstructing the information diffusion processes.

5 Conclusions

In this paper, we introduced a reversible residual network block based on the relationship between
diffusion phenomena and message passing neural networks, while ensuring the reversibility of the
network by limiting its Lipshitz coefficient. Using this, we constructed a discrete denoising diffusion
model (DDMSL) which can locate the source of graph diffusion and restore the diffusion paths.
Extensive experiments on five real datasets have demonstrated the effectiveness of DDMSL and its
constituent modules. Our work offers insights into how to calculate the distribution of solutions to
graph diffusion inverse problems based on the information propagation laws on complex networks.

Solving the inverse problem of graph diffusion plays a crucial role in many social operations, including
controlling the spread of infectious diseases, rumors, and computer viruses. It provides valuable
insights on enhancing source detection performance and fills the gap in methods for recovering
diffusion evolution. However, our work has some limitations. For instance, DDMSL requires the
prior knowledge about propagation models, such as infection rate and recovery rate. Although we can
infer these parameters from existing observation data [32, 21], it limits the application of DDMSL in
situations with insufficient observations of propagation conditions. Our future research will focus on
reducing the dependence of DDMSL on prior conditions.

6 Acknowledgement

This work was supported in part by the National Key R&D Program of China under Grant No.
2021YFC3300302, the National Natural Science Foundation of China (Grant No. 62202089,
72192832 and U22A2004), the General project of Liaoning Provincial Department of Education
(Grant No. LJKZ0005), the Shanghai Rising-Star Program (Grant No. 23QA1403100), the Natural
Science Foundation of Shanghai (Grant No. 21ZR1421900), and the Fundamental Research Funds
for the Central Universities (Grant No. N2319004).

10

References

[1] Olalekan Adeyinka. Internet attack methods and internet security technology. In 2008 Second

Asia International Conference on Modelling & Simulation (AMS), pages 77–82, 2008.

[2] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg.
Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information
Processing Systems, 34:17981–17993, 2021.

[3] Suman Banerjee, Mamata Jenamani, and Dilip Kumar Pratihar. A survey on influence maxi-
mization in a social network. Knowledge and Information Systems, 62:3417–3455, 2020.

[4] J. Behrmann, W. Grathwohl, Rtq Chen, D. Duvenaud, and J. H. Jacobsen. Invertible residual

networks. In International Conference on Machine Learning, 2019.

[5] Marián Boguná, Romualdo Pastor-Satorras, Albert Díaz-Guilera, and Alex Arenas. Models of
social networks based on social distance attachment. Physical review E, 70(5):056122, 2004.

[6] Zongmai Cao, Kai Han, and Jianfu Zhu. Information diffusion prediction via dynamic graph neu-
ral networks. In 2021 IEEE 24th international conference on computer supported cooperative
work in design (CSCWD), pages 1099–1104. IEEE, 2021.

[7] Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffu-
sion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687,
2022.

[8] Gojko ˇCutura, Boning Li, Ananthram Swami, and Santiago Segarra. Deep demixing: Recon-
structing the evolution of epidemics using graph neural networks. In 2021 29th European Signal
Processing Conference (EUSIPCO), pages 2204–2208. IEEE, 2021.

[9] Ming Dong, Bolong Zheng, Nguyen Quoc Viet Hung, Han Su, and Guohui Li. Multiple
rumor source detection with graph convolutional networks. In Proceedings of the 28th ACM
international conference on information and knowledge management, pages 569–578, 2019.

[10] Fei Gao, Jiang Zhang, and Yan Zhang. Neural enhanced dynamic message passing.

In
International Conference on Artificial Intelligence and Statistics, pages 10471–10482. PMLR,
2022.

[11] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural
message passing for quantum chemistry. In International conference on machine learning,
pages 1263–1272. PMLR, 2017.

[12] Pablo M Gleiser and Leon Danon. Community structure in jazz. Advances in complex systems,

6(04):565–573, 2003.

[13] H. Gouk, E. Frank, B. Pfahringer, and M. J. Cree. Regularisation of neural networks by

enforcing lipschitz continuity. Machine Learning, 110(2):393–416, 2021.

[14] Kilian Konstantin Haefeli, Karolis Martinkus, Nathanaël Perraudin, and Roger Wattenhofer.
Diffusion models for graphs benefit from discrete state spaces. arXiv preprint arXiv:2210.01549,
2022.

[15] Qiang He, Hui Fang, Jie Zhang, and Xingwei Wang. Dynamic opinion maximization in social
networks. IEEE Transactions on Knowledge and Data Engineering, 35(1):350–361, 2021.

[16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances

in Neural Information Processing Systems, 33:6840–6851, 2020.

[17] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax.

arXiv preprint arXiv:1611.01144, 2016.

[18] David Kempe, Jon Kleinberg, and Éva Tardos. Maximizing the spread of influence through
a social network. In Proceedings of the ninth ACM SIGKDD international conference on
Knowledge discovery and data mining, pages 137–146, 2003.

11

[19] William Ogilvy Kermack and Anderson G McKendrick. A contribution to the mathematical
theory of epidemics. Proceedings of the royal society of london. Series A, Containing papers of
a mathematical and physical character, 115(772):700–721, 1927.

[20] Thomas Kipf and Max Welling. Semi-supervised classification with graph convolutional

networks. ArXiv, abs/1609.02907, 2017.

[21] Daniel B Larremore, Bailey K Fosdick, Kate M Bubar, Sam Zhang, and Yonatan H Grad.
Estimating sars-cov-2 seroprevalence and epidemiological parameters with uncertainty from
serological surveys. eLife Sciences, 10, 2021.

[22] Chen Ling, Junji Jiang, Junxiang Wang, and Zhao Liang. Source localization of graph diffusion
via variational autoencoders for graph inverse problems. In Proceedings of the 28th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1010–1020, 2022.

[23] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc
Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11461–11471,
2022.

[24] Wuqiong Luo and Wee Peng Tay. Estimating infection sources in a network with incomplete
observations. In 2013 IEEE Global Conference on Signal and Information Processing, pages
301–304. IEEE, 2013.

[25] Wuqiong Luo and Wee Peng Tay. Estimating infection sources in a network with incomplete
observations. In 2013 IEEE Global Conference on Signal and Information Processing, pages
301–304. IEEE, 2013.

[26] Wuqiong Luo, Wee Peng Tay, and Mei Leng. How to identify an infection source with limited
observations. IEEE Journal of Selected Topics in Signal Processing, 8(4):586–597, 2014.

[27] Andrew Kachites McCallum, Kamal Nigam, Jason Rennie, and Kristie Seymore. Automating
the construction of internet portals with machine learning. Information Retrieval, 3:127–163,
2000.

[28] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization

for generative adversarial networks. arXiv preprint arXiv:1802.05957, 2018.

[29] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic
models. In International Conference on Machine Learning, pages 8162–8171. PMLR, 2021.

[30] B Aditya Prakash, Jilles Vreeken, and Christos Faloutsos. Spotting culprits in epidemics: How
many and which ones? In 2012 IEEE 12th International Conference on Data Mining, pages
11–20. IEEE, 2012.

[31] Maciel M Queiroz, Dmitry Ivanov, Alexandre Dolgui, and Samuel Fosso Wamba. Impacts of
epidemic outbreaks on supply chains: mapping a research agenda amid the covid-19 pandemic
through a structured literature review. Annals of operations research, pages 1–38, 2020.

[32] Jonathan M. Read, Jessica R. E. Bridgen, Derek A. T. Cummings, Antonia Ho, and Chris P.
Jewell. Novel coronavirus 2019-ncov (covid-19): early estimation of epidemiological parameters
and epidemic size estimates. Philosophical Transactions of the Royal Society B: Biological
Sciences, 376(1829):20200265–, 2021.

[33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-
resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[34] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad
Norouzi. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis
and Machine Intelligence, 2022.

[35] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini.

The graph neural network model. IEEE transactions on neural networks, 20(1):61–80, 2008.

12

[36] Hao Sha, Mohammad Al Hasan, and George Mohler. Source detection on networks using
spatial temporal graph convolutional networks. In 2021 IEEE 8th International Conference on
Data Science and Advanced Analytics (DSAA), pages 1–11. IEEE, 2021.

[37] Chintan Shah, Nima Dehmamy, Nicola Perra, Matteo Chinazzi, Albert-László Barabási, Alessan-
dro Vespignani, and Rose Yu. Finding patient zero: Learning contagion source with graph
neural networks. CoRR, abs/2006.11913, 2020.

[38] Xincheng Shu, Bin Yu, Zhongyuan Ruan, Qingpeng Zhang, and Qi Xuan. Information source
In Graph Data Mining, pages 1–27.

estimation with multi-channel graph neural network.
Springer, 2021.

[39] Tristan SW Stevens, Jean-Luc Robert, Faik C Yu, Jun Seob Shin, and Ruud JG van Sloun.
Removing structured noise with diffusion models. arXiv preprint arXiv:2302.05290, 2023.

[40] Xiaolu Tang, Changcheng Wu, Xiang Li, Yuhe Song, Xinmin Yao, Xinkai Wu, Yuange Duan,
Hong Zhang, Yirong Wang, Zhaohui Qian, et al. On the origin and continuing evolution of
sars-cov-2. National science review, 7(6):1012–1023, 2020.

[41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information
processing systems, 30, 2017.

[42] Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pas-
cal Frossard. Digress: Discrete denoising diffusion for graph generation. arXiv preprint
arXiv:2209.14734, 2022.

[43] Junxiang Wang, Junji Jiang, and Liang Zhao. An invertible graph diffusion neural network for
source localization. In Proceedings of the ACM Web Conference 2022, pages 1058–1069, 2022.

[44] Zheng Wang, Chaokun Wang, Jisheng Pei, and Xiaojun Ye. Multiple source detection without
knowing the underlying propagation model. In Proceedings of the Thirty-First AAAI Conference
on Artificial Intelligence, AAAI’17, page 217–223. AAAI Press, 2017.

[45] Duncan J Watts and Steven H Strogatz. Collective dynamics of ‘small-world’networks. nature,

393(6684):440–442, 1998.

[46] Wenwen Xia, Yuchen Li, Jun Wu, and Shenghong Li. Deepis: Susceptibility estimation on
social networks. In Proceedings of the 14th ACM International Conference on Web Search and
Data Mining, pages 761–769, 2021.

[47] KiJung Yoon, Renjie Liao, Yuwen Xiong, Lisa Zhang, Ethan Fetaya, Raquel Urtasun, Richard
Zemel, and Xaq Pitkow. Inference in probabilistic graphical models by graph neural networks.
In 2019 53rd Asilomar Conference on Signals, Systems, and Computers, pages 868–875. IEEE,
2019.

[48] Wayne W Zachary. An information flow model for conflict and fission in small groups. Journal

of anthropological research, 33(4):452–473, 1977.

[49] Wenyu Zang, Peng Zhang, Chuan Zhou, and Li Guo. Locating multiple sources in social
networks under the sir model: A divide-and-conquer approach. Journal of Computational
Science, 10:278–287, 2015.

[50] Laijun Zhao, Hongxin Cui, Xiaoyan Qiu, Xiaoli Wang, and Jiajia Wang. Sir rumor spreading
model in the new media age. Physica A: Statistical Mechanics and its Applications, 392(4):995–
1003, 2013.

[51] Linhua Zhou and Meng Fan. Dynamics of an sir epidemic model with limited medical resources

revisited. Nonlinear Analysis: Real World Applications, 13(1):312–324, 2012.

[52] Kai Zhu, Zhen Chen, and Lei Ying. Catch’em all: Locating multiple diffusion sources in
networks with partial observations. In Proceedings of the Thirty-First AAAI Conference on
Artificial Intelligence, AAAI’17, page 1676–1682. AAAI Press, 2017.

13

[53] Linhe Zhu, Gui Guan, and Yimin Li. Nonlinear dynamical analysis and control strategies
of a network-based sis epidemic model with time delay. Applied Mathematical Modelling,
70:512–531, 2019.

[54] Arkaitz Zubiaga, Ahmet Aker, Kalina Bontcheva, Maria Liakata, and Rob Procter. Detection and
resolution of rumours in social media: A survey. ACM Computing Surveys (CSUR), 51(2):1–36,
2018.

14

A Broader impacts

Overall, our work offers valuable insights into how to limit the spread of malicious information. For
example, by tracing the spread of infectious diseases, we can identify potential contacts of infected
individuals, effectively containing the outbreak. However, it is important to consider the potential
negative implications of this approach for society, such as compromising the privacy of individuals
living with infectious diseases like HIV. Addressing these concerns should be important.

B Derivation of forward and backward processes

Derivation of forward processes. Extending Equation 4 by means of the Markov property:

q (cid:0)xi

t | xi
0

(cid:1) =

=

=

(cid:88)

t
(cid:89)

xi

1:t−1

k=1

(cid:88)

t
(cid:89)

q (cid:0)xi

k | xi

k−1

(cid:1)

k−1Qi
xi

kxi
k

T

k=1

T

0Qi
xi

1xi
1

xi
1:t−1
(cid:88)

xi

1:t−1

· · · xi

k−1Qi

kxi
k

T

· · · xi

t−1Qi

txi
t

T

(18)







= xi

0Qi
1

(cid:88)



T

xi
1

xi
1

 · · ·



(cid:88)

T

xi

t−1

xi

t−1


 Qi

txi
t

T

xi
1

xi

t−1

= xi

= xi
0

0Qi
¯Qi

2 · · · IQi
1IQi
T
∼ Cat (cid:0)xi
txi
t

kI · · · IQi

txi
t
¯Qi
t; p = xi
t
0

(cid:1)

T

Where I is the identity matrix.

Derivation of backward processes. The detailed derivation of Equation 6 is as follows:

t−1 | xi
0

(cid:1)

q (cid:0)xi

t−1

(cid:1) q (cid:0)xi
t−1, xi
0
(cid:1)
q (cid:0)xi
t | xi
0
(cid:1) q (cid:0)xi
t−1 | xi
0
(cid:1)
t | xi
0
¯Qi
t−1xi
· xi
0
T
¯Qi
txi
t
T (cid:17)

(cid:16)

T

txi
t
xi
0
T xi

·

t−1

t−1

(cid:1)

T

q (cid:0)xi

t−1 | xi

t, xi
0

(cid:1) =

=

=

=

=

q (cid:0)xi

t | xi

q (cid:0)xi

t | xi

xi

t−1Qi

xi

tQi
t

(cid:16)

(cid:16)

T (cid:17)

xi

tQi
t

(cid:1) (cid:16)

xi

t−1

T (cid:17)

T

xi
0
⊙ (cid:0)xi

¯Qi
txi
t
¯Qi

t−1

0
¯Qi

xi
0

T

txi
t
(cid:16)
xi

tQi
t

T (cid:17)

⊙ (cid:0)xi

0

¯Qi

t−1

xi
0

¯Qi

txi
t

T



(cid:1)




xi

t−1; p =

∼ Cat

15

xi
0

¯Qi

t−1xi

t−1

T (cid:17)

(19)

From the Bayesian formula, it follows that:

q (cid:0)xi

t−1 | xi

t, xi
0

(cid:1) = q (cid:0)xi

t−1 | xi
t

(cid:1) =

=

=

(cid:1)

t, xi
0

(cid:80)

xi
0

(cid:80)

xi
0

q (cid:0)xi

t−1, xi
(cid:1)

q (cid:0)xi

t
t−1 | xi

q (cid:0)xi

(cid:1) q (cid:0)xi
(cid:1)

t, xi
0
q (cid:0)xi

t

0 | xi
t

(cid:1) q (cid:0)xi

t

(cid:1)

(20)

(cid:88)

q (cid:0)xi

t−1 | xi

t, xi
0

(cid:1) q (cid:0)xi

0 | xi
t

(cid:1)

xi
0
= E

t)q (cid:0)xi

t−1 | xi

t, xi
0

(cid:1)

q(xi

0|xi

Fitting this distribution using a neural network:
(cid:0)xi

(cid:1) = E

q (cid:0)xi

(cid:1) ≈ pθ

t−1 | xi
t

t−1 | xi
t

0∼pθ(xi
xi

t)q (cid:0)xi
0|xi
q (cid:0)xi
t | xi

q (cid:0)xi

t | xi

= E

pϑ(xi

t)
0|xi

= E

p0(xi

t)
0|xi

(cid:80)
j

(cid:104)
q (cid:0)xi

t | xi

t−1

q (cid:0)xi

t | xi

t−1

(cid:1) (cid:104)(cid:80)

=

=

t−1 | xi
0

(cid:1)

(cid:1)

t−1

t−1 | xi
t, xi
0
(cid:1) q (cid:0)xi
t−1, xi
0
(cid:1)
q (cid:0)xi
t | xi
0
(cid:1) q (cid:0)xi
t−1 | xi
0
(cid:1)
t | xi
0
t−1 | xi
xi
0
q (cid:0)xi
(cid:1)
t | xi
0
(cid:16)
t−1 | xi
0
(cid:1)
t | xi
0

xi
q (cid:0)xi

q (cid:0)xi
(cid:16)
(cid:1) q

(j)(cid:17)

(j)(cid:17)

j q

(cid:1)

pθ

pθ

(21)

(cid:16)

xi
0

(j)

| xi
t

(cid:17)(cid:105)

(cid:16)

xi
0

(j)

| xi
t

(cid:17)(cid:105)

C Propagation rule constraint of information diffusion reconstruction

(cid:16)

xi

T (cid:17)

tQi
t

and (cid:0)xi

To investigate the circumstances under which the propagation rules may be violated, let’s revisit
Equation 6. Note that the denominator serves as the normalization term, while the numerator is
(cid:1) - that are crucial in preserving the propagation
¯Qi
composed of two key terms -
rule.
The three rows of Qi
t
states: S, I, and R. Similarly, the three rows of ¯Qi
is in one of three states: S, I, and R. Let
then qa

t−1, xi
t represent the distribution of q(xi
and

t is in one of three
0) when xi
0
t respectively,

correspond to the distribution of q(xi

be denoted by qa

0) when xi

t and qb

t−1|xi

(cid:104) ¯Qi

(cid:104) ¯Qi

t|xi

t−1

12

13

(cid:105)

(cid:105)

T

0

t

t

1, qb
1 = γi
1 = 0.

(cid:81)t
k=1(1 − βi
0
0



¯Qi

t =

k) (cid:81)t−1

k=1(1 − βi
(cid:81)t

t−1(1 − γi
t + qa
k)βi
t)
k=1(1 − γi
k)
0

(cid:80)t

j=1

t−1(γi
qa
(cid:81)t−j

t) + qb
t−1
k)γi
k=1(1 − γi
1

k+1



 (22)

(cid:16)

T (cid:17)

t−1|xi

t and xi

0 are in different states, the results of q(xi

When xi
shown in Table 4. The table reveals that q(xi
violated. Since we are predicting xi
(Infected), with xi
to [1, 0, 0] to resolve the issue. Furthermore, if xi
the propagation rules. However, as shown in the 9th row of Table 4, the probability of xi
is much smaller than that of xi
t−1 = I and xi
failure. Hence, there is no need for any specific handling of this scenario.

(cid:1) are
tQi
t
0) = [0, 0, 0] when the propagation rule is
0: S (Susceptible) and I
t−1|xi
0) can be set
t−1 = S would violate
t−1 = S
t−1 = R, and such situations will not lead to training

0 = R (Recovered) being excluded. In such instances, q(xi
0 = S, then xi

t, xi
0, there are only two possible states for xi

t = R and xi

⊙ (cid:0)xi

t−1|xi

0) =

t, xi

t, xi

¯Qi

xi

t−1

0

To further minimize propagation rule violations during the training process, we incorporate supervi-
sion of the propagation rule. Specifically, when using this supervision function, nodes that have a
state of R are set to I to enforce the propagation rule.

Lconstrain1 = Relu (Xt−1 − (A + I)X0)

(23)

16

Table 4: The distribution of the unnormalized q(xi

t−1|xi

t, xi

0) in different cases.

t xi
xi
0

S (cid:81)t
I
R
S
I
R
S

S

I

R

k=1(1 − βi

k=1(1 − βi
k)

S
k) (cid:81)t−1
0
0
(cid:81)t−1

qa
t

qb
t

k=1(1 − βi
k)

0
0
(cid:81)t−1

k=1(1 − βi
k)

I

R

0

0

q(xi

t−1|xi
t, xi
0)
I
0
0
0

k=1(1 − γi

(cid:81)t
k=1(1 − γi

(cid:81)t

t−1

k)qa
k) (cid:81)t−1
k=1(1 − γi
k)
0
(cid:81)t−j
k=1(1 − γi
k=1(1 − γi
k)γi
k=1(1 − γi
k)
0

k)γi

k+1

k+1

j=1

(cid:81)t−j
· (cid:81)t−1

qa
t−1

(cid:80)t

(cid:80)t

j=1

R
0
0
0
0
0
0
qb
t−1

(cid:80)t

j=1

(cid:81)t−j

k=1(1 − γi
1

k)γi

k+1

where (A + I)X0 represents the total number of infected nodes in a given node’s first-order neigh-
borhood. Equation 23 penalizes X0 if the node is infected and there are no other infected nodes
within its first-order neighborhood. To maintain the stability of inferred X0 generating Xt−1, we
apply the same constraint to the process. Specifically, we utilize the monotonicity regularization of
information diffusion from [22]. If the source set X(i)
0 , then the generated Xt−1
resulting from their diffusion needs to satisfy the following equation.

0 is a superset of X(j)

Lconstrain2 =

(cid:16)

(cid:13)
(cid:13)
(cid:13)max

0, X(j)

t−1 − X(i)

t−1

(cid:17)(cid:13)
2
(cid:13)
(cid:13)

D Proofs of lemmas and theorems

D.1 The proof of theorem 3.1

, ∀X(i)

0 ⊇ X(j)

0

(24)

Proof Set the initial infection seed set as: S = {xs1
infection status distribution of node i is:

0 , xs2

0 , . . . xsm

0 }. At the initial moment, the






(0) = 0, P i=Sm
(0) = 1, P i̸=Sm

I

(0) = 1
(0) = 0

I

P i=Sm
S
P i̸=Sm
S
P i
R(0) = 0

At time t, the infection status distribution of node i is:






I (t) = P i
P i
P i
S(t) = P i
R(t) = P i
P i

I (t − 1)(1 − γi
S(t − 1)
I (t − 1)γi

R(t)

(cid:104)(cid:81)

R(t − 1)) + P i
I (t))AijP j

j(1 − βj

1 − (cid:81)
S(t − 1)
(cid:105)
I (t − 1)

(cid:104)

j(1 − βj

I (t))AijP j

I (t − 1)

(25)

(cid:105)

(26)

where A is the adjacency matrix and j is the neighbor of node i. When dealing with a static graph
G, A is fixed, allowing for determination of the state distribution at any time based on the initial
node state. Specifically, if both graph G and the seed node set S are known, it becomes possible to
calculate the state distribution of each node at any given time using Equation 26. Utilizing Equations
2 and 3, Qi

t can also be determined.

D.2 Proof of lemma 3.1

Proof The graph convolution layer with batch normalization BN (g(ξ)) can be abbreviated as
GConv. In our approach, we apply spectral normalization to both the linear transformation U
and convolutional layers GConv. As a result, the weight parameters of both the networks fw and
GConv possess 1-Lipschitz continuity after spectral normalization, as described in [28].

17

Let GConv : Rn → Rm, x1, x2 ∈ Rn. Note that the nonlinear activation function σ(·) is set to
M ish(·) , which obviously possesses 1-Lipschitz continuity.

∥fθ(x1) − fθ(x2)∥p = ∥σ(GConv(x1)) − σ(GConv(x2))∥p
≤ ∥GConv(x1) − GConv(x2)∥p
< ∥x1 − x2∥p

(27)

where ∥·∥p represents the p-norm (p = 2 or p = ∞). Therefore, the Lipschitz constant of fθ is less
than 1.

D.3 Proof of theorem 3.2

Proof To prove that YT = F(fw(X0)) is reversible, it is necessary to ensure that F and fw are
reversible [43].

(cid:26) fw(X0) = X0 + fw(X0) − X0 = ξ

ξ + fθ(ξ) = YT

⇔

(cid:26) X0 = ξ + X0 − fw(X0)

ξ = YT − fθ(ξ)

We construct the following iterative formula:

(cid:26) Xk+1

0 = ξ + Xk

0), X0
ξk+1 = YT − fθ(ξk), ξ0 = YT

0 − fw(Xk

0 = ξ

⇒

(cid:26) limk→∞ Xk

0 = X0

limk→∞ ξk = ξ

(28)

(29)

0 − fw(Xk

By Lemma 2, we can ensure that Lw < 1 and Lθ < 1. Moreover, the Lipschitz constant of
(Xk
0) is 1 − Lw, which is less than 1. Thus, when the number of iterations k is sufficiently
large, it follows that the transformation YT = F(fw(X0)) is reversible according to the Banach
fixed point theorem [4].

D.4 Proof of theorem 3.3

Proof Specifically, Theorem 3.2 proves that (F1◦F2◦ . . . ◦Fn)(ξ) is invertible when n = 1. There-
fore, we are currently examining whether (F1◦F2◦ . . . ◦Fn)(ξ) retains its reversibility for n > 1. To
make the notation simpler, we denote (F1◦F2◦ . . . ◦Fi)(ξ) as ˆFi.

ˆFn = (F1◦F2◦ . . . ◦Fn)(ξ) = DP ◦ · · · ◦ DP
(cid:125)

(cid:124)

(cid:123)(cid:122)
n

(cid:34)

ξ + fw(ξ) +

(cid:35)

(cid:17)
(cid:16) ˆFi(ξ)

fw

n−1
(cid:88)

i=1

(30)

The application of the dropout function DP will limit the Lipschitz constant of any function f to
(cid:1) ≤
1 − r times its original value: LDP(f ) = (1 − r)Lf . Additionally, we have L ˆFi
(1 − r)i(1 + Lfw )i [13]. Therefore, the Lipschitz constant of ˆFn is expressed as:

≤ (cid:81)i

(cid:0)LFj

j=1

(cid:34)

L ˆFn

≤ (1 − r)n

1 + Lfw +

n−1
(cid:88)

i=1

Lfw · L ˆFi

(cid:34)

≤ (1 − r)n

1 + Lfw

n−1
(cid:88)

i=0

[(1 − r)(1 + Lfw )]i

= (1 − r)n

(cid:20)
1 + Lfw ·

1 − [(1 − r)(1 + Lfw )]n
1 − (1 − r)(1 + Lfw )

(cid:35)

(cid:35)

(cid:21)

(31)

Note that when Lfw < 1 and n > 1, we only need to set r = 1/2 to ensure that L ˆFn
guaranteeing that ˆFn is reversible.

< 1, hereby

E DDMSL implementation details

The DDMSL approach has been previously explained, and we will now provide further details on the
implementation of DDMSL. The linear transform U refers to a fully connected layer, and in Figure 2,
the dense layer is composed of two fully connected layers that undergo spectral normalization. The

18

final output of the nnθ is represented by an N × 1 matrix, indicating the probability that each node is
in an infected state at t = 0. We designate nodes with an infection probability higher than a certain
threshold as infected nodes.
(cid:9),
Given a complete information diffusion instance X = {X0, . . . , XT} where Xt = (cid:8)x1
we sample t ∈ {1, . . . , T } to be included in the neural network nnθ based on the probability
distribution p(t) ∼ t
t=1 t , and use the sine-cosine position encoding [41] to embed t. The training
(cid:80)T
and inference processes are shown in Algorithm 1 and Algorithm 2, respectively. Additionally, the
variable T remains consistent with the information diffusion step size.

t , . . . , xN
t

Algorithm 1: Training
Input: X0, G, Xt, threshold α

1 repeat
2 Qt ← Equation 2 and Equation 3
3 q(xt−1|xt) ← Calculate Equation 6 using Xt and Qt
4 t ∼ P({1, . . . , T }), P (t) = t
(cid:80)T
5 Xt−1

0 = nnθ(G, Xt, t)

t=1 t

// Using data from t, nnθ infers the source node Xt−1
used to reconstruct the diffusion graph at t − 1.

0 , which is then

[Xt−1
[Xt−1
0

6 Xt−1
0 > α] = 1
0
7 Xt−1
! = 1] = 0
0
′
t ← Equation 2 and Equation 3
8 Q
t is generated by Xt−1
0 .

// Q

′

9 Pθ(xt−1|xt) ← Calculate Equation 8 using Xt, Xt−1
10 Xt−1 ← Gumbel − Sof tmax(Pθ(xt−1|xt))
11 Take gradient descent step on:
12 ∇θ(Lsimple + Lconstrain)

0

and Q

′
t

Algorithm 2: Infering
Input: Xt, G, Empty set X, threshold α
Output: X

3

4

[Xt−1
[Xt−1
0

1 for t starts from T to 1 do
Xt−1
0 = nnθ(G, Xt, t)
2
Xt−1
0 > α] = 1
0
Xt−1
! = 1] = 0
0
′
t ← Equation 2 and Equation 3
Q
Pθ(xt−1|xt) ← Calculate Equation 8 using Xt, Xt−1
Xt−1 ← Gumbel − Sof tmax(Pθ(xt−1|xt))
X[t] ← Xt−1

0

6

5

7

8
9 end

′
and Q
t

S(t), P i

t , . . . , QN
t

I (t), and P i

(cid:9), where Qi

In Algorithm 1, we have Qt = (cid:8)Q1
t can be calculated using Equations 2 and
3. Additionally, the P i
R(t) in Equation 3 can be computed using various methods.
Monte Carlo simulations provide the most accurate results, but require at least 105 simulations to be
sufficiently precise, leading to a high time complexity. An alternative approach is to utilize a neural
network model [10, 47] to learn (cid:2)P i
R(t)(cid:3), which significantly reduces the training time
I (t), P i
of the model. When applied to the SI model, DDMSL utilizes the state transfer matrix by:
(cid:20)1 − βi
0

(cid:21)
I (t) βi
I (t)
1

S(t), P i

t =

(32)

Qi

0

where Xt−1
represents the predicted source node using Xt, while Q
t is generated using the same
method as above. Ultimately, Pθ(xt−1|xt) is calculated using Xt−1
, Xt, and Xt−1
. We obtain
0
Xt−1 by sampling from Pθ(xt−1|xt), and label nodes as 0(S), 1(I), or 2(R) based on their state.
Algorithm 2 proceeds in a similar manner to the process described above.

0

′

19

F Additional algorithms and dataset parameters

F.1 Hyperparameters of the algorithms

The hyperparameters for each algorithm have been set according to the values shown in Table 5.
Any parameter that is not stated as default is common to both SI and SIR models. In the updated
version of SLVAE, the original three-layer MLP network encoder was replaced with a three-layer GCN
network, resulting in improved performance. For hyperparameters and implementation details of other
algorithms, please refer to the corresponding original papers. DDMSL and deep learning comparison

Table 5: Hyperparameter settings of different algorithms.

Algorithms Hyper-parameter

power grid
Initial learning rate 2 × 10−3 2 × 10−3 2 × 10−3 2 × 10−3

cora_ml

karate

jazz

PGP
3 × 10−3

Search space
[2 × 10−3,4 × 10−3,5 × 10−3]

Description

DDMSL

SLVAE

DDMIX

Learning Rate
Decline Interval

n

Dropout rate
α in SIR model
α in SI model

Epoch

α
Learning rate
GCN-based
encoder parameters
MLP-based
decoder parameters
Epoch
α
Learning rate
Epoch

[1200,1500] [200,1000]

[500, 1200] [500, 1200] [200,500,800,1200]

Determined by the LOSS curves
of the training and validation sets.

6

0.5
0.4
0.4

2000

0.55

6

0.5
0.6
0.45

1600

0.5

6

0.5
0.4
0.4

1600

0.55

6

0.5
0.4
0.4

1600

0.55

2 × 10−3 2 × 10−3 2 × 10−3 2 × 10−3

8

0.5
0.45
0.45

1600

0.45
3 × 10−3

[min = 3,max = 9,step = 1]

Determined by Theorem \3.3
[min=0.3,max=0.7,step=0.05]
[min=0.3,max=0.7,step=0.05]
Determined by the LOSS curves
of the training and validation sets.
[min=0.3,max=0.7,step=0.05]
[2 × 10−3,4 × 10−3,5 × 10−3]

[64,128,256] [64,128,256] [64,128,256] [64,128,256]

[64,128,256]

[64,128,256],[128,256,512]

[256,128,1] [256,128,1] [256,128,1] [256,128,1]

[256,128,1]

[256,128,1],[512,256,1]

200
0.5

200
0.5

200
0.5

200
0.5

2 × 10−3 2 × 10−3 2 × 10−3 2 × 10−3

100

100

100

100

200
0.5
2 × 10−3
100

\
[min = 3,max = 9,step = 1]
[2 × 10−3,4 × 10−3,5 × 10−3]
\

Learning rate
decreases by 0.97
times the set epoach.
Number of
residual blocks

Threshold

Threshold

The hidden dim
of the encoder
The hidden dim
of the decoder

Threshold

algorithms are both running on Windows 10 systems and trained using a 4090 graphics card. The code
for DDMSL is already open source, please refer to: https://github.com/Ashenone2/DDMSL.

F.2 Additional details of the datasets

The description of the data sets used for the experiments and their statistics are shown as below:

• Karate [48]: It includes a network of interrelationships between 34 members of the Karate
club, comprising 34 nodes and 78 edges. The Karate dataset is a real dataset, widely
employed in complex network community discovery research.

• Jazz [12]: The Jazz dataset is a network dataset that captures the collaborative relationships
between jazz musicians. It comprises 198 nodes and 2,742 edges, and has been extensively
used in research on complex network community discovery, node importance metrics, and
other related studies.

• Cora-ML [27]: Cora-ML is a citation network dataset containing papers from the field
of machine learning. Nodes represent papers and edges represent citation relationships
between papers.

• Power Grid [45]: The Power Grid dataset is a network dataset describing the topology of

the Northeastern US power grid, containing 4,941 nodes and 6,594 edges.

• PGP [5]: It is a User network of the Pretty-Good-Privacy algorithm for secure information

exchange, consisting of 10,680 nodes and 24,316 edges.

Table 6: Statistics of the five datasets.

#Nodes #Edges #Avg(degree) #Average clustering coefficient #Density #Diameter

Datasets
Karate
Jazz
Cora_ml

34
198
2,810
Power Grid 4,941

78
2,742
7,981
6,594
10,680 24,316

PGP

2.29
27.70
5.68
1.33
4.55

0.57
0.62
0.28
0.08
0.27

0.14
0.14
0.002
0.005
0.0004

5
6
17
46
24

20

G Additional experiments

Experiments on Real Diffusion Datasets. In order to gauge the efficacy of DDMSL on real-
world propagation datasets, we opted for the Twitter [6] and Douban [6] datasets, encompassing
12,627 nodes with 309,631 edges, and 23,123 nodes with 348,280 edges, respectively. The detailed
performance metrics can be found in Table 7.

Table 7: Additional experiments on real diffusion datasets.

Twitter

PR
Methods
DDMSL 0.445
0.310
SLVAE

RE
0.286
0.317

F1
0.313
0.253

AUC
0.625
0.578

PR
0.484
0.412

Douban

RE
0.324
0.140

F1
0.381
0.209

AUC
0.622
0.547

Generalization Performance. We conducted extensive tests on datasets of varying scales to assess
the generalization performance of both DDMSL and SLVAE algorithms. The comparative results are
presented in Table 8, revealing that DDMSL demonstrates commendable generalization performance
across a majority of scenarios.

Table 8: Additional generalization experiments: Test results on different network topologies after
one training on a real network, where the original performance represents the test performance of
DDMSL on real networks.

Training data
Network

Cora Ml

Power Grid

PGP

Twitter

Douban

Original
performance

Small World

ER

BA Tree

BA Dense

PR RE

F1 AUC PR RE

F1 AUC PR RE

F1 AUC PR RE

F1 AUC
DDMSL 0.790 0.908 0.845 0.941 0.763 0.966 0.852 0.966 0.754 0.887 0.815 0.928 0.445 0.286 0.313 0.625 0.484 0.324 0.381 0.622
SLVAE 0.721 0.765 0.852 0.908 0.908 0.719 0.803 0.856 0.817 0.721 0.766 0.851 0.310 0.317 0.253 0.578 0.412 0.140 0.209 0.547
DDMSL 0.732 0.826 0.776 0.896 0.812 0.499 0.62 0.744 0.987 0.684 0.808 0.841 0.375 0.299 0.290 0.618 0.409 0.295 0.301 0.624
SLVAE 0.824 0.576 0.678 0.781 0.626 0.335 0.436 0.656 0.982 0.539 0.696 0.769 0.308 0.241 0.227 0.572 0.375 0.124 0.186 0.544
DDMSL 0.722 0.584 0.645 0.779 0.349 0.687 0.463 0.773 0.997 0.614 0.632 0.731 0.439 0.289 0.309 0.624 0.320 0.367 0.307 0.614
SLVAE 0.894 0.586 0.708 0.789 0.747 0.409 0.528 0.697 0.956 0.555 0.703 0.776 0.310 0.311 0.285 0.569 0.368 0.118 0.179 0.537
DDMSL 0.482 0.832 0.609 0.866 0.872 0.774 0.82 0.881 0.947 0.672 0.786 0.834 0.327 0.377 0.323 0.612 0.451 0.289 0.314 0.623
SLVAE 0.961 0.577 0.721 0.787 0.939 0.399 0.560 0.698 0.993 0.548 0.708 0.774 0.252 0.285 0.193 0.517 0.312 0.126 0.180 0.525
DDMSL 0.749 0.683 0.715 0.829 0.654 0.354 0.459 0.667 0.972 0.598 0.741 0.798 0.369 0.305 0.288 0.622 0.423 0.289 0.304 0.623
SLVAE 0.662 0.611 0.635 0.788 0.731 0.441 0.550 0.712 0.997 0.446 0.617 0.723 0.286 0.405 0.315 0.567 0.374 0.132 0.195 0.553

F1 AUC PR RE

Time Complexity. Lastly, we conducted a comparative evaluation of the time complexity between
DDMSL and baseline algorithms across diverse datasets, revealing the outcomes illustrated in Table 9.
Owing to the gradual inference of diffusion state for each time step, the time complexity of DDMSL
tends to be substantial. However, optimization through parallel computing can effectively mitigate
this disparity.

Table 9: Additional Time complexity experiment.

Test time
DDMSL
SLVAE
DDMIX
GCNSI
LPSI
OJC
NetSleuth

Cora-Ml
15.84s
4.77s
9.34s
1.4s
2m14s
6m11s
2m40s

Training Time Cora-Ml
10m36s
18s
16m5s
3m53

DDMSL
SLVAE
DDMIX
GCNSI

Power-Grid
22.19s
7.28s
15.7s
8.14s
1m51s
50m17s
4m39s
Power-Grid
16m57s
39s
24m17s
20m6s

PGP
22.72s
11.29s
23.26s
15.41s
21m37s
2h41m52s
21m24s
PGP
32m03s
1m10s
37m53s
34m26s

21

H Visualization

H.1 Visualization of reconstructing diffusion paths

To conserve space, we displayed the actual node states and corresponding prediction results every
20% of the time. Figures 5 to 9 showcase the results, where the blue nodes denote susceptible ones,
the red nodes denote infected nodes, and the green nodes denote recovered nodes. The findings
indicate that DDMSL accurately restored the node states at different times. In contrast, DDMIX
could only restore infected nodes, revealing that DDMSL far surpasses DDMIX in its expression
capability.

Figure 5: DDMSL reconstructs SIR diffusion on Karate.

H.2 Visual comparison of source localization

Due to spatial limitations, we only presented the source localization results of DDMSL and benchmark
algorithms on the Karate and Jazz datasets. The results are depicted in Figures 10 and 11. The
baseline algorithm’s performance is unsatisfactory, as evidenced by significant positioning errors in
the source nodes. On the contrary, DDMSL outperforms other benchmark algorithms in accurately
identifying source nodes. Moreover, even when misidentifying source nodes, DDMSL locates them
near the actual nodes.

22

 W   7 W    7   W    7   W    7   * U R X Q G  7 U X W K W    3 U H G L F W L R QFigure 6: DDMSL reconstructs SIR diffusion on Jazz.

Figure 7: DDMSL reconstructs SIR diffusion on Coral ml.

23

 W   7 W    7   W    7   W    7   * U R X Q G  7 U X W K W    3 U H G L F W L R Q W   7 W    7   W    7   W    7   * U R X Q G  7 U X W K W    3 U H G L F W L R QFigure 8: DDMSL reconstructs SIR diffusion on Power grid.

Figure 9: DDMSL reconstructs SIR diffusion on PGP.

24

 W   7 W    7   W    7   W    7   * U R X Q G  7 U X W K W    3 U H G L F W L R Q W   7 W    7   W    7   W    7   * U R X Q G  7 U X W K W    3 U H G L F W L R QFigure 10: Visualization comparisons of source localization on Karate.

Figure 11: Visualization comparisons of source localization on Jazz.

25

  D   ' ' 0 6 /  E   6 / 9 $ (  F   ' ' P L [  G   / 3 6 ,  H   * & 1 6 ,  I   2 - &  J   * U R X Q G  7 U X W K  D   ' ' 0 6 /  E   6 / 9 $ (  F   ' ' P L [  G   / 3 6 ,  H   * & 1 6 ,  I   2 - &  J   * U R X Q G  7 U X W K