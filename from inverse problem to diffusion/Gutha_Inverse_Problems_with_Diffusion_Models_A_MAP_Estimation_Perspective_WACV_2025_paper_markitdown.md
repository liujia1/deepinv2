ThisWACVpaperistheOpenAccessversion,providedbytheComputerVisionFoundation.
Exceptforthiswatermark,itisidenticaltotheacceptedversion;
thefinalpublishedversionoftheproceedingsisavailableonIEEEXplore.
Inverse Problems with Diffusion Models: A MAP Estimation Perspective
SaiBharathChandraGutha1,RicardoVinuesa2,HosseinAzizpour1
1RPL,KTHRoyalInstituteofTechnology,Sweden
2FLOW,KTHRoyalInstituteofTechnology,Sweden
|     |     |     | sbcgutha@kth.se, |     |     | rvinuesa@mech.kth.se, |     | azizpour@kth.se |     |     |     |     |     |     |
| --- | --- | --- | ---------------- | --- | --- | --------------------- | --- | --------------- | --- | --- | --- | --- | --- | --- |
Abstract and the task is to infer the original data x given the obser-
|     |     |     |     |     |     |     | vation y. | The | function | A : | Rn → | Rm is | known | as the |
| --- | --- | --- | --- | --- | --- | --- | --------- | --- | -------- | --- | ---- | ----- | ----- | ------ |
Inverseproblemshavemanyapplicationsinscienceand forwardoperator,andtypicallyn ≫ m,indicatingthatthe
engineering. InComputervision,severalimagerestoration observation y ∈ Rm corresponds to a severely degraded
| tasks such | as inpainting, |     | deblurring, |     | and super-resolution |     |     |     |     |     |     |     |     |     |
| ---------- | -------------- | --- | ----------- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
signal,fromwhichoneneedstorecovertheoriginalsignal
can be formally modeled as inverse problems. Recently, x ∈ Rn,whichmakesthetaskhighlychallenging. Forlin-
methodshavebeendevelopedforsolvinginverseproblems ear inverse problems, A denotes a linear mapping and can
∈Rm×n.
that only leverage a pre-trained unconditional diffusion besubstitutedwithamatrixH
| model and | do not | require | additional | task-specific |     | training. |     |     |     |     |     |     |     |     |
| --------- | ------ | ------- | ---------- | ------------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
Insuchmethods,however,theinherentintractabilityofde- y =A(x)+η (1)
terminingtheconditionalscorefunctionduringthereverse
|                                      |      |                  |     |                       |                 |               | Several              | conventional |                | approaches    |     | for solving |       | inverse  |
| ------------------------------------ | ---- | ---------------- | --- | --------------------- | --------------- | ------------- | -------------------- | ------------ | -------------- | ------------- | --- | ----------- | ----- | -------- |
| diffusionprocessposesarealchallenge, |      |                  |     |                       | leavingthemeth- |               |                      |              |                |               |     |             |       |          |
|                                      |      |                  |     |                       |                 |               | problems             | exist        | [1].           | These include |     | approaches  |       | based on |
| ods to settle                        | with | an approximation |     | instead,              |                 | which affects |                      |              |                |               |     |             |       |          |
|                                      |      |                  |     |                       |                 |               | functional-analytic, |              | probabilistic, |               |     | data-driven |       | methods, |
| theirperformanceinpractice.          |      |                  |     | Here,weproposeaMAPes- |                 |               |                      |              |                |               |     |             |       |          |
|                                      |      |                  |     |                       |                 |               | and more.            | Recently,    |                | Deep Learning |     | (DL)        | based | methods  |
timationframeworktomodelthereverseconditionalgener-
|     |     |     |     |     |     |     | have been | applied | to  | solve | inverse | problems |     | and have |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------- | --- | ----- | ------- | -------- | --- | -------- |
ationprocessofacontinuoustimediffusionmodelasanop-
|     |     |     |     |     |     |     | shown great | success. |     | In a Bayesian |     | framework, |     | solving |
| --- | --- | --- | --- | --- | --- | --- | ----------- | -------- | --- | ------------- | --- | ---------- | --- | ------- |
timizationprocessoftheunderlyingMAPobjective,whose
aninverseproblemnaturallycorrespondstoestimatingthe
| gradient           | term is    | tractable.   | In            | theory,  | the proposed | frame-       |                  |          |                                     |           |             |     |                 |         |
| ------------------ | ---------- | ------------ | ------------- | -------- | ------------ | ------------ | ---------------- | -------- | ----------------------------------- | --------- | ----------- | --- | --------------- | ------- |
|                    |            |              |               |          |              |              | posteriorP(x|y). |          | TypicalDL-basedapproachesforsolving |           |             |     |                 |         |
| work can           | be applied | to           | solve general |          | inverse      | problems us- |                  |          |                                     |           |             |     |                 |         |
|                    |            |              |               |          |              |              | inverse          | problems | fall                                | into two  | categories. |     | 1.              | Methods |
| ing gradient-based |            | optimization |               | methods. | However,     | given        |                  |          |                                     |           |             |     |                 |         |
|                    |            |              |               |          |              |              | that directly    | learn    | the                                 | posterior | P(x|y)      |     | via conditional |         |
thehighlynon-convexnatureofthelossobjective,findinga
|     |     |     |     |     |     |     | generative | models | [14,19], |     | and 2. | Methods | that | learn |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ------ | -------- | --- | ------ | ------- | ---- | ----- |
perfectgradient-basedoptimizationalgorithmcanbequite
|     |     |     |     |     |     |     | P(x) via | an unconditional |     | generative |     | model | and | use it |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---------------- | --- | ---------- | --- | ----- | --- | ------ |
challenging,nevertheless,ourframeworkoffersseveralpo-
|                  |             |             |              |          |            |           | to infer  | P(x|y)   | [5,12,21,29]. |                 | Methods   |                 | of the   | former   |
| ---------------- | ----------- | ----------- | ------------ | -------- | ---------- | --------- | --------- | -------- | ------------- | --------------- | --------- | --------------- | -------- | -------- |
| tential research |             | directions. | We           | use our  | proposed   | formula-  |           |          |               |                 |           |                 |          |          |
|                  |             |             |              |          |            |           | category  | require  | task-specific |                 | training, | i.e.            | training | with     |
| tion to develop  |             | empirically | effective    |          | algorithms | for image |           |          |               |                 |           |                 |          |          |
|                  |             |             |              |          |            |           | a dataset | of pairs | (x,y),        | where           |           | the degradation |          | y is     |
| restoration.     | We          | validate    | our proposed |          | algorithms | with ex-  |           |          |               |                 |           |                 |          |          |
|                  |             |             |              |          |            |           | computed  | using    | x and         | a task-specific |           | forward         |          | operator |
| tensive          | experiments | over        | multiple     | datasets | across     | several   |           |          |               |                 |           |                 |          |          |
restorationtasks. A. This limits the out-of-the-box applicability of the
|     |     |     |     |     |     |     | model to | a different |         | task (different |            | forward  | operator). |          |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | ------- | --------------- | ---------- | -------- | ---------- | -------- |
|     |     |     |     |     |     |     | On the   | contrary,   | methods | of              | the latter | category |            | train an |
1.Introduction unconditional generative model to learn P(x), and this
|     |     |     |     |     |     |     | training | is task-independent |     | since | it  | only | needs | a dataset |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------------------- | --- | ----- | --- | ---- | ----- | --------- |
x.
Inverseproblemsareubiquitousinscienceandengineer- of original data samples These methods then use the
ingwithawiderangeofdownstreamapplications [2,28]. trained model for P(x) and since P(y|x) is tractable (i.e.
InComputervision,severalimagerestorationtaskssuchas fromEq.(1),P(y|x)=N(A(x),σ2I)),utilizingtheBayes
y
inpainting, deblurring, super-resolution, and more, can be rule,theyinfertheposteriorP(x|y)∝P(y|x)P(x).
| formallymodeledasinverseproblems. |     |     |     |     | Inaninverseprob- |     |     |     |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
lem, characterized by Eq. (1), y ∈ Rm is a (potentially Several choices for Deep Generative Models (DGMs)
Rn,
noisy) observation of the original data x ∈ and η is exist, each with its advantages and disadvantages. There
a random variable denoting i.i.d. noise, typically assumed have been approaches using Generative Adversarial Net-
N(0,σ2I),
to be Gaussian with a known variance i.e η ∼ works (GAN) [9] and Normalizing Flow (NF) [17] based
y
4153

DGMs for solving inverse problems, with more recent dw¯denotesthestandardWienerprocesswhentflowsback-
methods focusing on Diffusion models [10,20,26], owing wards from T to 0 with dt denoting an infinitesimal neg-
totheirstate-of-the-artperformanceinseveralvision-based ative time step. The term ∇ x logP t (x) is called the score
generativetasks. Thisworkfocusesonmethodsthatusea functionofthemarginaldistributionP (x). Ifweknowthis
t
pre-trainedunconditionaldiffusionmodelasthepriorP(x) score function for each marginal distribution i.e. for all t,
and infer the posterior P(x|y) for solving inverse prob- then one could solve the reverse-SDE in Eq. (3) and gen-
lems.Sec.2providessomebackgroundondiffusionmodels erate samples from the data distribution. In general, the
and related works that use unconditional diffusion models score function is not analytically tractable and is hard to
forsolvinginverseproblemsandtheirinherentlimitations. estimate, however, one could train a time-indexed neural
Later in Sec. 3, we propose our Maximum A Posteriori networkmodeltolearnthescorefunctionviascorematch-
(MAP)estimationframeworkforcontinuous-timediffusion ing techniques [25,27]. The trained score model S θ (x,t)
models and discuss the practical implementation. Specifi- can be substituted in place of ∇ logP (x) in Eq. (3),
|     |     |     |     |     |     |     |     |     |     |     | x   | t   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
cally, we propose a novel MAP formulation that employs and the reverse-SDE can be solved using traditional SDE
| a reparameterization |     | based | on consistency |     | models | [23] to | solvers[26]. |     |     |     |     |     |     |     |
| -------------------- | --- | ----- | -------------- | --- | ------ | ------- | ------------ | --- | --- | --- | --- | --- | --- | --- |
modelthereverseconditionaldiffusionprocessasMAPop-
1
timization. In Sec. 4, we use our proposed framework to dx=[f(x,t)− g(t)2∇ logP (x)]dt (4)
|     |     |     |     |     |     |     |     |     |     | 2   | x   | t   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
developempiricallyeffectivealgorithmsforimagerestora-
| tion,andin | Sec.5,wevalidatethealgorithmswithexten- |     |     |     |     |     |         |            |     |         |           |     |     |        |
| ---------- | --------------------------------------- | --- | --- | --- | --- | --- | ------- | ---------- | --- | ------- | --------- | --- | --- | ------ |
|            |                                         |     |     |     |     |     | For the | stochastic |     | process | described | by  | the | SDE in |
siveexperimentsondeblurring,super-resolution,andimage Eq. (2), there exists a corresponding Ordinary Differential
| inpainting. | In Sec.6,wepresentabriefdiscussionbefore |     |     |     |     |     |          |       |       |        |                 |     |             |     |
| ----------- | ---------------------------------------- | --- | --- | --- | --- | --- | -------- | ----- | ----- | ------ | --------------- | --- | ----------- | --- |
|             |                                          |     |     |     |     |     | Equation | (ODE) | shown | in Eq. | (4), describing |     | a determin- |     |
concludingourfindingsin Sec.7. istic process whose trajectories share the same marginal
{P(x(t))}T
|     |     |     |     |     |     |     | probability | densities |     |     | as  | those | simulated | by  |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --------- | --- | --- | --- | ----- | --------- | --- |
t=0
2.Background the SDE. This is called the Probability Flow ODE (PF
|                |     |      |                |     |            |     | ODE) in                                         | the literature |     | [26]. | So equivalently, |     | one | could |
| -------------- | --- | ---- | -------------- | --- | ---------- | --- | ----------------------------------------------- | -------------- | --- | ----- | ---------------- | --- | --- | ----- |
| GivenadatasetD |     | = {x | }N ,whereeachx |     | isani.i.d. |     |                                                 |                |     |       |                  |     |     |       |
|                |     |      | i i=1          |     | i          |     | alsouseODEsolverstosolveEq.(4)inreversetimefrom |                |     |       |                  |     |     |       |
sampledrawnfromanunknowndatadistributionP data (x), t=T until0togeneratesamplesfromthedatadistribution.
| aDGMlearnstoapproximateP |     |     | fromthesamplesinD. |     |     |     |     |     |     |     |     |     |     |     |
| ------------------------ | --- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
data
|     |     |     |     |     |     |     | Hereon, | we  | assume | the default | choice | for | drift | and dif- |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | ------ | ----------- | ------ | --- | ----- | -------- |
(cid:113)
| 2.1.DiffusionModels |     |     |     |     |     |     |                     |     |     |        |         |      |     | dσ2(t), |
| ------------------- | --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | ------ | ------- | ---- | --- | ------- |
|                     |     |     |     |     |     |     | fusion coefficients |     | as  | f(x,t) | = 0 and | g(t) | =   |         |
dt
Diffusionmodels [10,20,26]arearecentfamilyofgen- where σ(t) is a monotonically increasing noise schedule
erative models. The methodology involves simulating a from t = 0 to T, with σ(T) being very high. This choice
| stochasticprocess{x(t)}T |     |     | describedbyaStochasticDif- |     |     |     |          |           |     |          |      |              |     |        |
| ------------------------ | --- | --- | -------------------------- | --- | --- | --- | -------- | --------- | --- | -------- | ---- | ------------ | --- | ------ |
|                          |     |     | t=0                        |     |     |     | of f and | g results | in  | a closed | form | perturbation |     | kernel |
ferentialEquation(SDE)suchasEq.(2), wheret ∈ [0,T] P(x(t)|x(0)) = N(x(0),σ2(t)−σ2(0)) and P(x(T)) ≈
is a continuous time variable, x(0) ∼ P = P is the N(0,σ2(T)). Therecanbemultipledesignchoicesforthe
0 data
datadistributionforwhichwehaveadatasetDofsamples,
|     |     |     |     |     |     |     | noise schedule |     | σ(t), | resulting | in various | formulations |     | of  |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ----- | --------- | ---------- | ------------ | --- | --- |
x(T) ∼ P isatractablepriordistribution. Thefunctions diffusionmodels [11].
T
| f(·,t):Rn | →Rnandg(·):R→Rarecalledthedriftand |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --------- | ---------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
2.2.SolvingInverseproblemswithDiffusionmodels
| diffusion                        | coefficients | of x(t) | respectively, |                 | and dw denotes |     |                                                      |     |         |            |     |         |         |     |
| -------------------------------- | ------------ | ------- | ------------- | --------------- | -------------- | --- | ---------------------------------------------------- | --- | ------- | ---------- | --- | ------- | ------- | --- |
| thestandardWienerprocess.        |              |         | Typicallyf    | andg            | arechosen      |     |                                                      |     |         |            |     |         |         |     |
|                                  |              |         |               |                 |                |     | As described                                         |     | in Sec. | 1, solving | an  | inverse | problem | en- |
| inawaythatyieldsatractablepriorP |              |         |               | whichcontainsno |                |     |                                                      |     |         |            |     |         |         |     |
|                                  |              |         |               | T               |                |     | tailsestimationof(orsamplingfrom)theposteriorP(x|y), |     |         |            |     |         |         |     |
| informationaboutP                |              | (i.e.   | P ).          |                 |                |     |                                                      |     |         |            |     |         |         |     |
|                                  |              | 0       | data          |                 |                |     | whereyisthenoisydegradationoftheoriginaldatasample   |     |         |            |     |         |         |     |
x.Inthecontextofsolvinginverseproblemsusingdiffusion
dx=f(x,t)dt+g(t)dw (2) models, sampling from the posterior P(x(0)|y) involves
conditioningthereversediffusionprocessonywhichtrans-
Eq.(2)alsodescribesthe”forwardprocess”,inwhich,start-
latestosolvingthemodifiedreverse-SDEinEq.(5).
| ing from | an initially | clean | data sample, | a small | amount | of  |     |     |     |     |     |     |     |     |
| -------- | ------------ | ----- | ------------ | ------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
noise is progressively added at each step until it turns into dx=[f(x,t)−g(t)2∇
|         |        |              |              |     |         |       |     |     |     | x logP | t (x|y)]dt+g(t)dw¯ |     |     | (5) |
| ------- | ------ | ------------ | ------------ | --- | ------- | ----- | --- | --- | --- | ------ | ------------------ | --- | --- | --- |
| a noisy | sample | of the prior | distribution | P   | T . The | back- |     |     |     |        |                    |     |     |     |
ward/reverse process which transforms a noisy sample of Similartomethodsofthefirstcategory(Sec.1),whichdi-
P intoacleansampleofthedatadistributionisdescribed rectlylearntheposteriorP(x|y)aspartoftheirtraining,it
T
bythecorrespondingreverse-SDEinEq.(3) ispossibletotrainaconditionaldiffusionmodelthatlearns
|     |     |     |     |     |     |     | the conditional |     | score | function | directly. | More | specifically, |     |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ----- | -------- | --------- | ---- | ------------- | --- |
dx=[f(x,t)−g(t)2∇
x logP t (x)]dt+g(t)dw¯ (3) onecouldlearnS θ (x,y,t)usingconditionalscorematching
4154

objectivesinplaceoftheusualunconditionalscorefunction instead. From a theoretical perspective, it is unclear how
S (x,t)inSec.2.1. Inthiswork, wefocusonmethodsof this reparameterization should help. Also, both works ig-
θ
thesecondcategory, whichonlyleverageanunconditional nore the prior term while focusing only on the data term.
diffusionmodelforP(x), toinferP(x|y). Wemodifythe If the data distribution has full support, i.e. P(x ) ̸= 0
0
notations to denote x(t) with x , σ(t) with σ , and P (x) ∀x ∈Rn,ignoringthepriorterm,especiallyinthecaseof
|     |     |     |     | t   |     | t   | t   | 0   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
withP(x )forthesakeofconvenienceandtobeconsistent anoisymeasurement,istheoreticallyunjustified. Ourpro-
t
withpreviousworks[5,21]. posedmethodissimilartoZSIRandDMPlugfromaprac-
|     |        |       |        |     |     |          |       | tical perspective, |     | however, | our MAP formulation |     | is based |
| --- | ------ | ----- | ------ | --- | --- | -------- | ----- | ------------------ | --- | -------- | ------------------- | --- | -------- |
| ∇   | logP(x | |y)=∇ | logP(x | )+∇ |     | logP(y|x | ) (6) |                    |     |          |                     |     |          |
xt t xt t xt t on sound theoretical motivation that justifies the reparam-
(cid:82) eterization based on PF ODE, providing new insights into
|     | P(y|x | )=  | P(y|x | )P(x | |x  | )dx | (7) |     |     |     |     |     |     |
| --- | ----- | --- | ----- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |       | t   | x0    | 0    | 0   | t 0 |     |     |     |     |     |     |     |
modelingtheconditionalgenerationprocessasMAPopti-
| Solving | Eq. | (5) involves | estimating |     | the | conditional | score |     |     |     |     |     |     |
| ------- | --- | ------------ | ---------- | --- | --- | ----------- | ----- | --- | --- | --- | --- | --- | --- |
mization.
| function  | ∇     | logP(x | |y).     | The pre-trained |           | unconditional |             |                  |     |     |     |     |     |
| --------- | ----- | ------ | -------- | --------------- | --------- | ------------- | ----------- | ---------------- | --- | --- | --- | --- | --- |
|           | xt    |        | t        |                 |           |               |             |                  |     |     |     |     |     |
| diffusion | model | can    | be used  | to              | estimate  | ∇             | logP(x ),   |                  |     |     |     |     |     |
|           |       |        |          |                 |           | xt            | t           | 3.OurMethodology |     |     |     |     |     |
| however,  | the   | term ∇ | logP(y|x |                 | ) becomes |               | intractable |                  |     |     |     |     |     |
|           |       |        | xt       |                 | t         |               |             |                  |     |     |     |     |     |
(ref Eqs. (6) and (7)). At its core, the intractability of 3.1.Background: ConsistencyModels
| ∇ logP(y|x |     | ) arises | from | the fact | that | P(x | |x ) is in- |     |     |     |     |     |     |
| ---------- | --- | -------- | ---- | -------- | ---- | --- | ----------- | --- | --- | --- | --- | --- | --- |
| x t        |     | t        |      |          |      | 0   | t           |     |     |     |     |     |     |
tra c table [21], and hence, the conditional score is hard to Consider the PF ODE described in Eq. (4). The solu-
tiontrajectoriesofthisODEaresmooth,andmapthesam-
estimatewhileonlyleveragingtheunconditionalscore.
|     |     |     |     |     |     |     |     | plesonthedatamanifoldtopurenoise. |     |     |     | In [23], | aconsis- |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------- | --- | --- | --- | -------- | -------- |
2.3.Relatedworks
tencymodelisdefinedasthefunctionthatmapsanypoint
onthePFODEtrajectorytoitscorrespondingorigin(initial
| PGDM | [21] | approximates |     | P(x | |x ) | with a | Gaussian |     |     |     |     |     |     |
| ---- | ---- | ------------ | --- | --- | ---- | ------ | -------- | --- | --- | --- | --- | --- | --- |
|      |      |              |     |     | 0 t  |        |          |     |     |     |     |     |     |
xˆ r 2, point on the data manifold). There exist efficient method-
| distribution |     | having | mean | t and | variance |     | t where |     |     |     |     |     |     |
| ------------ | --- | ------ | ---- | ----- | -------- | --- | ------- | --- | --- | --- | --- | --- | --- |
E(x ologies [22] to train these consistency models in practice.
| xˆ = | |x  | ) = x | +σ2∇ | logP(x | )(usingTweedie’s |     |     |     |     |     |     |     |     |
| ---- | --- | ----- | ---- | ------ | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| t    | 0   | t t   | t    | xt     | t                |     |     |     |     |     |     |     |     |
formula). The standard deviation r is a hyperparameter, Pleasereferto [23]foradetaileddescription.
t
| chosenproportionallytoσ |     |     | t .DPS[5]approximatesP(y|x |     |     |     | t ) |     |     |     |     |     |     |
| ----------------------- | --- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
3.2.ProposedMAPestimationframework
| with the | point       | estimate | P(y|x   | =   | xˆ) and | has      | an almost |     |     |     |     |     |     |
| -------- | ----------- | -------- | ------- | --- | ------- | -------- | --------- | --- | --- | --- | --- | --- | --- |
|          |             |          |         | 0   | t       |          |           |     |     |     |     |     |     |
| similar  | formulation |          | as PGDM | up  | to a    | constant | factor,   |     |     |     |     |     |     |
though the motivation is slightly different. Boys et al. [3] x∗ =argmax logP(x |y) (8)
|                                                |     |     |     |     |     |     |      |     |     | 0   | 0   |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- |
| alsouseGaussianapproximationbutfurtherreplacer |     |     |     |     |     |     | with |     |     |     | x0  |     |     |
t
|     |     |     |     |     | 2∂  | x ˆt |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
the covariance matrix Cov[x 0 |x t ] = σ . Computing Eq.(8)referstotheusualMAPformulationforsolvingan
|     |     |     |     |     | t   | ∂ x t |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
this matrix is expensive in practice, so they resort to diag- inverseproblem. Findinganoptimalx∗ usinggradientas-
0
| onal | and row-sum | approximations |     |     | of the | matrix | instead. |     |     |     |     |     |     |
| ---- | ----------- | -------------- | --- | --- | ------ | ------ | -------- | --- | --- | --- | --- | --- | --- |
centinvolvestheupdatestepinEq.(9),withkdenotingthe
Peng et al. [16] proposes to find an optimal covariance kthiterateandλdenotingthestepsize.
matrixusinglearnedcovariancesfromthediffusionmodel.
xk+1 =xk+λ∗∇
Allthesemethods,however,stillassumesimplifiedapprox- logP(x |y) (9)
|          |     |        |          |        |       |             |     |     | 0   | 0   | x0  | 0   |     |
| -------- | --- | ------ | -------- | ------ | ----- | ----------- | --- | --- | --- | --- | --- | --- | --- |
| imations | for | P(x |x | ), which | limits | their | performance | in  |     |     |     |     |     |     |
|          |     | 0 t    |          |        |       |             |     |     |     |     |     |     |     |
practice, given the complicated and multimodal nature of The update step requires computing the gradient term
the true data distribution. Other works [6,7,12,30] try to ∇ logP(x |y)=∇ logP(y|x )+∇ logP(x ).The
|     |     |     |     |     |     |     |     | x0  | 0   | x0  | 0   | x0  | 0   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
circumventthistermbyprojectingtheintermediatex onto former term is tractable since P(y|x ) is Gaussian, and
|     |     |     |     |     |     |     | t   |     |     |     | 0   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
themeasurementsubspaceusingheuristicapproximations. the latter is the score function evaluated at x 0 and can
|     |     |     |     |     |     |     |     | be replaced | with | S (x ,0). | In practice, | S (x | ,0) is only |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ---- | --------- | ------------ | ---- | ----------- |
|     |     |     |     |     |     |     |     |             |      | θ 0       |              | θ    | 0           |
DiffPIR [33] poses the problem as MAP optimization accurate when x lies closer to the data manifold and is
0
with data and prior terms and utilizes the HQS [8] algo- typically inaccurate for x 0 in low-likelihood regions out-
rithm to solve a relaxed problem where the data and the side the data manifold [24]. This makes the score esti-
x
prior terms can be optimized alternatively in a decoupled mate inaccurate in the beginning (when 0 is initialized
manner. Specifically, in each reverse diffusion step, this randomly) and during the gradient ascent updates, since
amountstosolvingarelaxedMAPobjective. DDS[6]uses the intermediate xk are not constrained to lie on the data
0
asimilarframeworkbutemployssubspaceprojectionmeth- manifold. This issue can be avoided when inverse prob-
odstosolvetheintermediateMAPobjectivesateachdiffu- lemsaretypicallysolvedthroughthereversediffusionpro-
sionstep.ZSIR[4]andDMPlug[29]alsoposetheproblem cess (Sec. 2.2) which drifts the noisy sample towards the
as MAP optimization, where instead of optimizing for the data manifold using S (x ,t) while simultaneously ensur-
|     |     |     |     |     |     |     |     |     |     | θ   | t   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
originaldatax directly,theyreparameterizex viatheini- ingmeasurementconsistency. Buttherethechallengeisto
|     |     | 0   |     |     |     | 0   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tial diffusion noise x T and solve for the optimal noise x T estimate∇ xt logP(y|x t ),whichisagainintractable.
4155

Here, we present our proposed MAP formulation. Let Algorithm1:MAP-GA(MAP-Gradient-Ascent)
| z ∼   | P(x    | ) = N(0,σ2I), |                | denote | a purely | noisy  | sample,    |         |        |      |        |           |          |     |     |
| ----- | ------ | ------------- | -------------- | ------ | -------- | ------ | ---------- | ------- | ------ | ---- | ------ | --------- | -------- | --- | --- |
|       | T      |               | T              |        |          |        |            | input   | :τ =(τ | ,..τ | ,τ ),f | ,S ,y,num | iter,λ,σ |     |     |
|       |        |               |                |        |          |        |            |         |        | n 1  | 0 θ    | θ         |          |     |     |
| and M | denote | the           | data Manifold. |        | The      | PF ODE | trajectory | ∼N(0,σ2 |        | I)   |        |           |          |     |     |
z
| maps  | z to | a sample        | x ∈ | M,     | given | by x    | = f (z,T),    |                      |     | τn  |     |     |     |     |     |
| ----- | ---- | --------------- | --- | ------ | ----- | ------- | ------------- | -------------------- | --- | --- | --- | --- | --- | --- | --- |
|       |      |                 | 0   |        |       | 0       | θ             | foriin(n,n−1,..,1)do |     |     |     |     |     |     |     |
| where | f is | the consistency |     | model. | It    | is also | evident that, |                      |     |     |     |     |     |     |     |
|       | θ    |                 |     |        |       |         |               | t=τ                  |     |     |     |     |     |     |     |
i
∀x ∈ M,∃z ∼ P(x )suchthatx = f (z,T). Hence, forj in(1,2,..,num iter)do
| 0                                        |     |             | T      |        | 0      | θ             |        |     |          |       |          |            |     |     |     |
| ---------------------------------------- | --- | ----------- | ------ | ------ | ------ | ------------- | ------ | --- | -------- | ----- | -------- | ---------- | --- | --- | --- |
| the usual                                | MAP | formulation |        | in Eq. | (8)    | is equivalent | to the |     |          |       |          |            |     |     |     |
|                                          |     |             |        |        |        |               |        |     | z =z+λ∗∇ |       | z logP(f | θ (z,t)|y) |     |     |     |
| proposedMAPformulationinEqs.(10)and(11). |     |             |        |        |        |               |        | end |          |       |          |            |     |     |     |
|                                          |     |             |        |        |        |               |        | xˆ  | =f       | (z,t) |          |            |     |     |     |
|                                          | z∗  |             |        |        |        |               |        |     | 0 θ      |       |          |            |     |     |     |
|                                          |     | =argmax     | logP(x |        | 0 =f θ | (z,T)|y)      | (10)   |     |          | 2     | I)       |            |     |     |     |
|                                          |     |             | z      |        |        |               |        | z   | =N(xˆ    | 0 ,σ  |          |            |     |     |     |
τ i−1
end
|     |     |     | x∗ =f | (z∗,T) |     |     | (11) |     |     |     |     |     |     |     |     |
| --- | --- | --- | ----- | ------ | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     | 0     | θ      |     |     |      |     |     |     |     |     |     |     |     |
output:z
Withourproposedformulation,weupdatezwithgradient-
| ascent | steps | as in | Eq. (12) | for | finding | z∗. | The up- |     |     |     |     |     |     |     |     |
| ------ | ----- | ----- | -------- | --- | ------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
date step now requires computing the gradient term trajectorytothecorrespondingx insteadofx .
|          |     |           |       |     |     |              |      |     |     |     |     | ϵ   |     | 0   |     |
| -------- | --- | --------- | ----- | --- | --- | ------------ | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
| ∇ logP(f |     | (z,T)|y), | which | can | be  | reformulated | as a |     |     |     |     |     |     |     |     |
| z        |     | θ         |       |     |     |              |      |     |     |     |     |     |     |     |     |
vector-Jacobian product (vjp) as shown in Eq. (13). The z∗ =argmax logP(x =f (z,T)|y) (14)
|     |     |     |     |     |     |     |     |     |     |     |     | ϵ   | θ   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
z
| vectorinthisvjpisthegradientterm∇ |     |     |     |     | x0  | logP(x | 0 |y)eval- |     |     |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | --- | --- | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
uatedatx = f (z,T)whichliesonthedatamanifold(by x∗ ≈x∗ =f (z∗,T) (15)
|     | 0   | θ   |     |     |     |     |     |     |     | 0   | ϵ   | θ   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
definitionofconsistencymodel)andcanbeaccuratelyeval-
|     |     |     |     |     |     |     |     | Here, we | describe | in detail | the | computation |     | of the | gradi- |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------- | --------- | --- | ----------- | --- | ------ | ------ |
uated,unlikethepreviouscase.
|     |     |     |     |     |     |     |     | ent term | ∇ logP(f | (z,t)|y) | in  | practice. | From | Eqs. | (16) |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------- | -------- | --- | --------- | ---- | ---- | ---- |
z θ
zk+1 =zk+λ∗∇ logP(f (z,T)|y) (12) and(17),thisrequirestheestimationofthegradientoflog-
|     |     |     |     | z   | θ   |          |     |                 |     |          |                            |     |     |     |     |
| --- | --- | --- | --- | --- | --- | -------- | --- | --------------- | --- | -------- | -------------------------- | --- | --- | --- | --- |
|     |     |     |     |     |     |          |     | likelihoodi.e.∇ |     | logP(y|x | )andthegradientoflog-prior |     |     |     |     |
|     |     |     |     |     |     | (cid:12) |     |                 |     | x ϵ      | ϵ                          |     |     |     |     |
(cid:16) ∂fθ (z ,T) (cid:17)⊺ (cid:12) (13) i.e. ∇ logP(x ). Theterm sP(y|x )andP(x )arealso
| ∇zlogP(fθ(z,T)|y)= |            |     |              | ∇x0      | logP(x0|y) | (cid:12)   |               | xϵ                                                |     | ϵ   |           | ϵ   |          | ϵ   |     |
| ------------------ | ---------- | --- | ------------ | -------- | ---------- | ---------- | ------------- | ------------------------------------------------- | --- | --- | --------- | --- | -------- | --- | --- |
|                    |            |     | ∂ z          |          |            | (cid:12)   |               |                                                   |     |     |           |     |          |     |     |
|                    |            |     |              |          |            | x0=fθ(z,T) |               | referredtoasthelikelihoodandthepriorrespectively. |     |     |           |     |          |     |     |
| Here               | we provide |     | a high-level | overview |            | of a       | practical im- |                                                   |     |     |           |     |          |     |     |
|                    |            |     |              |          |            |            |               |                                                   |     |     | (cid:17)⊺ |     | (cid:12) |     |     |
plementationusingourMAPformulation. Inpractice,even (cid:16) ∂fθ ( z,t) (cid:12) (16)
|     |     |     |     |     |     |     |     | ∇ z logP(f | θ (z,t)|y)= |     | ∇   | xϵ logP(x | ϵ |y) (cid:12) |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ----------- | --- | --- | --------- | -------------- | --- | --- |
|     |     |     |     |     |     |     |     |            |             |     | ∂ z |           | (cid:12)       |     |     |
a consistency model f θ can benefit from multi-step sam- xϵ=fθ(z,t)
pling [23]. Thereforeweproposeamulti-stepgradientas- (cid:12) (cid:26) (cid:27)(cid:12)
|     |     |     |     |     |     |     |     | logP(xϵ|y)(cid:12) |     |     |     |     |     | (cid:12) | (17) |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | --- | --- | --- | --- | -------- | ---- |
cent scheme called MAP-Gradient-Ascent (MAP-GA) as ∇xϵ (cid:12) = ∇xϵ logP(y|xϵ)+∇xϵ logP(xϵ) (cid:12)
|                        |     |     |     |           |                    |     |     |     | (cid:12) xϵ=fθ(z,t) |     |     |     |     | (cid:12) xϵ=fθ(z,t) |     |
| ---------------------- | --- | --- | --- | --------- | ------------------ | --- | --- | --- | ------------------- | --- | --- | --- | --- | ------------------- | --- |
| describedinAlgorithm1. |     |     |     | Notethatτ | referstoatime-step |     |     |     |                     |     |     |     |     |                     |     |
schedule with T = τ > τ > .. > τ > τ = 0 Computingthegradientoflog-likelihood
|     |     |     | n   | n−1 |     | 1   | 0   |         |         |     |          |     |        |     |         |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | ------- | --- | -------- | --- | ------ | --- | ------- |
|     |     |     |     |     |     |     |     | P(y|x ) | = N(A(x | ),σ | 2I), and | P(x | |x ) = | N(x | ,σ 2I), |
andσreferstothemonotonicallyincreasingnoiseschedule 0 0 y ϵ 0 0 ϵ
for t ∈ [0,T], with σ = 0, and σ = ∞ (high value in given by Eq. (1) and the diffusion perturbation ker-
|     |     |     | 0   |     | T   |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
practice), y is the measurement, f and S are the consis- nel respectively. Since σ ≈ 0, P(x ) ≈ P(x )
|     |     |     |     |     | θ   | θ   |     |     |     |     | ϵ   |     | 0   |     | ϵ   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tency model and the score model respectively. num iter and P(x 0 |x ϵ ) ≈ P(x ϵ |x 0 ). We can approximate
denotes the number of gradient ascent iterations per time P(x |x ) = N(x ,σ 2I), andforalinearforwardoperator
|       |     |         |              |     |       |           |          | 0 ϵ    |     | ϵ ϵ     |     |            |       |     |      |
| ----- | --- | ------- | ------------ | --- | ----- | --------- | -------- | ------ | --- | ------- | --- | ---------- | ----- | --- | ---- |
|       | λ   |         |              |     |       |           |          | i.e. A | = H | ∈ Rm×n, | we  | can derive | using | Eq. | (7), |
| step, | and | denotes | the learning |     | rate. | Algorithm | 1 can be |        |     |         |     |            |       |     |      |
|       |     |         |              |     |       |           |          |        |     |         | 2I  | 2HH⊺).     |       |     |      |
appliedtosolveanyinverseproblemeffectively,giventhat P(y|x ϵ ) = N(Hx ϵ ,σ + σ For non-linear
|     |     |     |     |     |     |     |     |     |     |     | y   | ϵ   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
we know the optimal hyper-parameters, such as the time- A, similar approximations can be made by linearizing it
aroundx
stepscheduleτ,thelearningrateλ,learningrateschedule, ϵ . Giventhetractableformofthelikelihoodterm
num steps,etc. However,findingthoseinpracticecanbe above,thegradientofthelog-likelihoodisapparent.
quitechallenging.
Computingthegradientoflog-prior
3.3.PracticalImplementation
|                                                  |     |     |     |     |     |     |     | The gradient | of        | the log  | prior     | i.e. ∇ | logP(x | )       | is es- |
| ------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | ------------ | --------- | -------- | --------- | ------ | ------ | ------- | ------ |
|                                                  |     |     |     |     |     |     |     |              |           |          |           |        | xϵ     | ϵ       |        |
|                                                  |     |     |     |     |     |     |     | sentially    | the score | function | evaluated |        | at x   | . Given | a      |
| Inpractice,toavoidnumericalissuesandtoensurethat |     |     |     |     |     |     |     |              |           |          |           |        |        | ϵ       |        |
the gradient term ∇ logP(x ) exists, instead of x∗ , we score function S (x,t), learned by the unconditional
|            |     |          | x0  | 0      |        |                | 0   |           |        | θ        |     |       |         |          |     |
| ---------- | --- | -------- | --- | ------ | ------ | -------------- | --- | --------- | ------ | -------- | --- | ----- | ------- | -------- | --- |
| solveforx∗ |     |          |     |        |        |                |     | diffusion | model, | ∇ logP(x |     | ) = S | (x ,ϵ). | Learning |     |
|            |     | = argmax | xϵ  | logP(x | ϵ |y), | forasmallϵsuch |     |           |        | xϵ       | ϵ   |       | θ ϵ     |          |     |
ϵ
that σ ≈ 0. We solve the MAP formulation in Eqs. (14) the score function is equivalent to learning a denoiser,
ϵ
and (15). Since, P(x |x ) = N(x ,σ2I), for very small and vice-versa [11]. Hereon, we denote the uncondi-
|        |      |                   | ϵ 0 |         | 0   | ϵ   |              |                  |     |       |             |     |          |     |        |
| ------ | ---- | ----------------- | --- | ------- | --- | --- | ------------ | ---------------- | --- | ----- | ----------- | --- | -------- | --- | ------ |
|        |      |                   |     |         |     |     |              | tional diffusion |     | model | as learning | the | denoiser | D   | (x,t), |
| values | of σ | , the distinction |     | between | x   | and | x remain in- |                  |     |       |             |     |          | θ   |        |
|        |      | ϵ                 |     |         |     | 0   | ϵ            |                  |     |       |             |     |          |     |        |
significantforallpracticalpurposes. Theconsistencymod- from which the score function can be computed as
|        |      |          |         |        |     |        |           | logP(xt)= | Dθ(x | t, t)−xt. |     |     |     |     |     |
| ------ | ---- | -------- | ------- | ------ | --- | ------ | --------- | --------- | ---- | --------- | --- | --- | --- | --- | --- |
| els in | [23] | are also | learned | to map | the | points | on PF ODE | ∇xt       |      |           |     |     |     |     |     |
σ t 2
4156

4.ImagerestorationwithMAP-GA In an empirical setting, [11] also observes trajectories be-
|             |         |           |                  |     |          |      |                |          | comemorelinearwhenσ(t)    |                    |              | → 0. Whilemoreanalysison |                  |               |
| ----------- | ------- | --------- | ---------------- | --- | -------- | ---- | -------------- | -------- | ------------------------- | ------------------ | ------------ | ------------------------ | ---------------- | ------------- |
|             | Several | image     | restoration      |     | tasks    | such | as inpainting, |          |                           |                    |              |                          |                  |               |
|             |         |           |                  |     |          |      |                |          | this is                   | still due,         | it motivates | us to look               | at the           | denoiser as a |
| deblurring, |         | and       | super-resolution |     | can      | be   | modeled        | as       |                           |                    |              |                          |                  |               |
|             |         |           |                  |     |          |      |                |          | proxy                     | of the consistency |              | model, which             | gradually        | becomes       |
| linear      | inverse | problems. |                  | For | example, | in   | image          | inpaint- |                           |                    |              |                          |                  |               |
|             |         |           |                  |     |          |      |                |          | moreandmoreaccurateasσ(t) |                    |              | → 0.                     | Hence,wealsocon- |               |
ing[15,31],givenamaskedimage(andthecorresponding
|        |        |     |          |     |            |               |     |     | sider | settings that | replace | the consistency |     | model with the |
| ------ | ------ | --- | -------- | --- | ---------- | ------------- | --- | --- | ----- | ------------- | ------- | --------------- | --- | -------------- |
| binary | mask), |     | the goal | is  | to recover | (reconstruct) |     | the |       |               |         |                 |     |                |
denoiserinouralgorithm.
Rn
| missing | pixels   |        | of the | masked | image. | Let x    | ∈     | denote |     |     |     |     |     |     |
| ------- | -------- | ------ | ------ | ------ | ------ | -------- | ----- | ------ | --- | --- | --- | --- | --- | --- |
| the     | original | image, | y      | ∈ Rm   | denote | a masked | image | with   |     |     |     |     |     |     |
Algorithm2:MAP-GAforImagerestoration
| only | visible | pixels | (m  | ≤ n) | and H | ∈ Rm×n | denotes | a   |       |                 |     |          |      |       |
| ---- | ------- | ------ | --- | ---- | ----- | ------ | ------- | --- | ----- | --------------- | --- | -------- | ---- | ----- |
|      |         |        |     |      |       |        |         |     | input | : timeschedule: |     | τ =[τ ,τ | ,..τ | ,τ ], |
corresponding linear forward operator for a given mask. n n−1 1 0
The inpainting problem is characterized by y = Hx+η, noiseschedule: σ(.),denoiser: D ,
θ
|       | η   | ∼ N(0,σ | 2I). |      |          |             |     |         |     | consistencymodel: |     | C , |     |     |
| ----- | --- | ------- | ---- | ---- | -------- | ----------- | --- | ------- | --- | ----------------- | --- | --- | --- | --- |
| where |     |         | y    | Note | that for | inpainting, |     | a given |     |                   |     | θ   |     |     |
mask defines H with a defined structure, where the rows measurement: y,learningrate: λ,
of H are one-hot and are orthogonal i.e. HH⊺ = I . numgradientascentiter: num iter,
m×m
|       |       |             |     |          |      |     |            |     |     | boolean: | use | prior(defaultTrue), |     |     |
| ----- | ----- | ----------- | --- | -------- | ---- | --- | ---------- | --- | --- | -------- | --- | ------------------- | --- | --- |
| Other | image | restoration |     | problems | such | as  | deblurring | and |     |          |     |                     |     |     |
super-resolution can be modeled accordingly with their forwardoperatormatrix: H,
| correspondingforwardoperators. |     |     |     |     |     |     |     |     |     | measurementnoise: |     | σ   |     |     |
| ------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | --- | --- | --- |
y
|     |     |     |     |     |     |     |     |     |     |        | τ =ϵ,τ | =T), |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | ------ | ---- | --- | --- |
|     |     |     |     |     |     |     |     |     |     | (Note: | 0      | n    |     |     |
I)
|     | We expand |     | Algorithm | 1   | for image | restoration |     | and | z ∼N(0,σ | 2   |     |     |     |     |
| --- | --------- | --- | --------- | --- | --------- | ----------- | --- | --- | -------- | --- | --- | --- | --- | --- |
τ n
|         |     |         |          |         |     |           |     |     | foriin(n,n−1, |     |     | ..1)do |     |     |
| ------- | --- | ------- | -------- | ------- | --- | --------- | --- | --- | ------------- | --- | --- | ------ | --- | --- |
| include |     | all the | specific | details | in  | Algorithm | 2.  | The |               |     |     |        |     |     |
t=τ
| core | term | in the | algorithm |     | that needs | to be | evaluated | is  |     | i   |     |     |     |     |
| ---- | ---- | ------ | --------- | --- | ---------- | ----- | --------- | --- | --- | --- | --- | --- | --- | --- |
∇ logP(f (z,t)|y), which involves the estimation of forj in(1,2, ..num iter)do
|     | z   | θ   |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
xˆ =C (z,t)
| gradients |     | of the | log-likelihood |     | and | the log-prior |     | terms |     | ϵ   | θ   |     |     |     |
| --------- | --- | ------ | -------------- | --- | --- | ------------- | --- | ----- | --- | --- | --- | --- | --- | --- |
σ 2
( E q s . (1 6 ) a nd ( 1 7) ) . In th e a l g or i th m , w e m a k e a c h o ic e H⊺( y I+HH ⊺ )−1(y−Hxˆϵ)
|     |     |     |     |     |     |     |     |     |     | grad |            | = σ ϵ 2 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ---------- | ------- | --- | --- |
|     |     |     |     |     |     |     |     |     |     |      | likelihood |         | σ 2 |     |
( in d i ca te d b y th e u s e p rio r k e y w o r d )o f re ta in i n g or d r o p - ϵ
ifuse priorthen
| pingthegradientofthelog-prior. |     |     |     |     | Droppingthepriorterm |     |     |     |     |     |     |                 |     |     |
| ------------------------------ | --- | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --------------- | --- | --- |
|                                |     |     |     |     |                      |     |     |     |     |     |     | Dθ(xˆ ϵ, ϵ)−xˆϵ |     |     |
implies a choice of uniform prior, and the algorithm now grad =
|     |     |     |     |     |     |     |     |     |     |     | prior | σ 2 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- |
ϵ
| optimizes |     | for the   | maximum |             | likelihood | estimate | instead     | of  |     | end  |     |     |     |     |
| --------- | --- | --------- | ------- | ----------- | ---------- | -------- | ----------- | --- | --- | ---- | --- | --- | --- | --- |
| the       | MAP | estimate, | by      | considering | any        | sample   | (consistent |     |     | else |     |     |     |     |
|           |     |           |         |             |            |          |             |     |     | grad |     | =0  |     |     |
with our measurement) on the data manifold to be equally prior
| good. | This    | is also | the | setting | considered | in  | the concurrent |     |     | end   |           |            |       |       |
| ----- | ------- | ------- | --- | ------- | ---------- | --- | -------------- | --- | --- | ----- | --------- | ---------- | ----- | ----- |
| works | [4,29]. |         |     |         |            |     |                |     |     | grad  |           | =grad      | +grad |       |
|       |         |         |     |         |            |     |                |     |     |       | posterior | likelihood |       | prior |
|       |         |         |     |         |            |     |                |     |     |       | (cid:16)  | (cid:17)⊺  |       |       |
|       |         |         |     |         |            |     |                |     |     | grad= | ∂Cθ(z,t)  | grad       |       |       |
posterior
|     | Note | that the | algorithm | makes | use | of both | the | consis- |     |     |     | ∂z  |     |     |
| --- | ---- | -------- | --------- | ----- | --- | ------- | --- | ------- | --- | --- | --- | --- | --- | --- |
z =z+λ∗grad
| tencymodel(C |     |     | )andthedenoiser(D |     |     |                    |     |     |     |     |     |     |     |     |
| ------------ | --- | --- | ----------------- | --- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
|              |     |     | θ                 |     |     | θ ), whichmakesour |     |     |     |     |     |     |     |     |
end
| method                                          | more | demanding |     | compared |     | to other | methods | that |     |             |        |        |     |     |
| ----------------------------------------------- | ---- | --------- | --- | -------- | --- | -------- | ------- | ---- | --- | ----------- | ------ | ------ | --- | --- |
|                                                 |      |           |     |          |     |          |         |      |     | xˆ =C (z,t) |        |        |     |     |
| onlyusethedenoiser.Wemakeanargumentasfollows.We |      |           |     |          |     |          |         |      |     | ϵ θ         |        |        |     |     |
|                                                 |      |           |     |          |     |          |         |      |     | z =N(xˆ     | ,σ2    | −σ2 I) |     |     |
|                                                 |      |           |     |          |     |          |         |      |     |             | ϵ τi−1 | τ0     |     |     |
usethepre-traineddenoiserandtheconsistencymodelfrom
end
| [11] | and | [23] | respectively, | with | noise | schedule | σ(t) | = t |     |     |     |     |     |     |
| ---- | --- | ---- | ------------- | ---- | ----- | -------- | ---- | --- | --- | --- | --- | --- | --- | --- |
output:z
| andt | ∈ [ϵ,T]. |     | ConsiderthecorrespondingPFODE(from |     |     |     |     |     |     |     |     |     |     |     |
| ---- | -------- | --- | ---------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Eq.(4))fortheabovesetting,asfollows.
|     |     |     |     |     |     |     |     |     | When | using | Algorithm | 2 for noisy | image | restoration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ----- | --------- | ----------- | ----- | ----------- |
D θ (x,t)−x
dx=− dt (σ y >0)inpractice,weobservethatitrequirescarefultun-
t
ingofthelearningrateandotherhyperparameters.Toavoid
ThisPFODEdeterminesthetrajectoryandsolvingthetra- suchsensitivehyperparameters,wepresentAlgorithm3for
jectory origin x involves solving the ODE above. Note noisy image restoration, based on our empirical observa-
0
that this x is what the consistency model C is trained to tions from Tab. 3. The motivation for Algorithm 3, is to
|     |     | 0   |     |     |     |     | θ   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
predict. As a rough approximation, solving the ODE with find an approximate solution using MAP-GA at diffusion
abackwardEulerdiscretizationstepfromtto0,whiches- timet=τ whereσ =σ ,anduseitasaninitializationfor
|     |     |     |     |     |     |     |     |     |     |     | τ   | y   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
sentially assumes the trajectory curves are linear, i.e. the PGDM[21]forσ <σ .Specifically,weuseMAP-GAun-
|     |     |     |     |     |     |     |     |     |     |     | t   | y   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Jacobian dx isconstantfortheinterval[0,t],andthisgives tilσ =σ ,tofindx andlaterusethisasaninitialization
|     |     | dt  |     |     |     |     |     |     | t   | y   | σy  |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
C (x ,t) = x ≈ D (x ,t), with the approximation get- toPGDMforσ <σ . Wedonotusethepriortermforthe
| θ   | t   | 0   |     | θ t |     |     |     |     |     | t   | y   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tingmoreaccurateasthetrajectorycurvegetsmorelinear. MAP-GApartinAlgorithm3,aswefinditmoreeffective.
4157

WeevaluatetheperformanceofMAP-GA(Algorithm2)
Algorithm3:MAP-GA-PGDMImagerestoration onImageNet[18]1Kvalidationsetwith64×64resolution
input : σ(.),denoiser: D , (in Tab. 1) and on 100 random images of LSUNCat [32]
|     | noiseschedule: |     |     |     | θ   |     |     |     |     |     |     |
| --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
consistencymodel: C , with 256 × 256 resolution (in Tab. 2). We use the pre-
θ
measurement: y,learningrate: λ, traineddenoisersandtheconsistencymodelsfrom [11,23]
numgradientascentiter: num iter, with their default settings. In the experiments, the setting
forwardoperatormatrix: H, MAP-GA denotes the default Algorithm 2, MAP-GA(NP)
measurementnoise: σ denotes MAP-GA with no prior, MAP-GA(D) denotes
y
=[τmap,τmap,..τmap,τmap],
τmap−ga MAP-GA with denoiser replacing the consistency model,
|     |     |     | n   | n−1 | 1 0 |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
τpgdm =[τpgdm,τpgdm,..τpgdm,τpgdm], and MAP-GA(D,NP) denotes MAP-GA with no prior
|     |     |     | m   | m−1 | 1 0 |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(Note: σ =σ andτ map =T) and the denoiser replacing the consistency model. We
|     |        | τmap |     | y n      |      |         |              |      |          |            |      |
| --- | ------ | ---- | --- | -------- | ---- | ------- | ------------ | ---- | -------- | ---------- | ---- |
|     |        | 0    |     |          |      | compare | against DDRM | [12] | and PGDM | [21] (both | only |
|     | (Note: | σ    | =σ  | andτpgdm | =ϵ), |         |              |      |          |            |      |
τm pgdm y 0 use the denoiser) and against the zero-shot image editing
=======MAP-GA=======
|     |     | I)  |     |     |     | (CT-ZSIE)algorithmproposedin |     |     | [23]whichonlyusethe |     |     |
| --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | ------------------- | --- | --- |
z ∼N(0,σ2
τn map consistency model. The results from Tabs. 1 and 2 show
| foriin(n,n−1, |              |       | ..1)do |            |               |                                              |                |              |         |                      |                |
| ------------- | ------------ | ----- | ------ | ---------- | ------------- | -------------------------------------------- | -------------- | ------------ | ------- | -------------------- | -------------- |
|               | t=τmap       |       |        |            |               | MAP-GA                                       | and variants   | outperform   |         | DDRM, PGDM,          | and            |
|               | i            |       |        |            |               | CT-ZSIEwithasignificantmarginonseveraltasks. |                |              |         |                      |                |
|               | forj in(1,2, |       | ..num  | iter)do    |               |                                              |                |              |         |                      |                |
|               | xˆ =C        | (z,t) |        |            |               |                                              |                |              |         |                      |                |
|               | ϵ            | θ     |        |            |               |                                              |                |              |         |                      |                |
|               |              |       |        |            |               | M A P-                                       | G A u s es g r | ad i e nt as | c e n t | ( fi rs t- ord e r g | r a d i e n t- |
|               |              |       | H⊺(    | σ y 2 I+HH | ⊺ )−1(y−Hxˆϵ) |                                              |                |              |         |                      |                |
2
grad = σ ϵ bas ed m e th od) t o op t im i z e the u n d e rl y in g M A P o b j e c t iv e ,
|     |      | likelihood |       |            | σ 2 |                                                    |     |     |     |     |     |
| --- | ---- | ---------- | ----- | ---------- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- |
|     |      |            |       |            | ϵ   | whichistypicallyhighlynon-convexandisnotguaranteed |     |     |     |     |     |
|     | grad | posterior  | =grad | likelihood |     |                                                    |     |     |     |     |     |
(cid:16) (cid:17)⊺ tofindtheglobaloptima. However,itcouldconvergewith
∂Cθ(z,t)
|     | grad= |     |     | grad |     |     |     |     |     |     |     |
| --- | ----- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
∂z posterior aninitializationclosertotheglobaloptima. Tocorroborate
z =z+λ∗grad this,wedesignthefollowingtoyexperiment. Weconsider
end
|     |       |       |     |      |     | the noiseless | inpainting | task                           | from earlier, | but, in | Algo- |
| --- | ----- | ----- | --- | ---- | --- | ------------- | ---------- | ------------------------------ | ------------- | ------- | ----- |
|     | xˆ =C | (z,t) |     |      |     |               |            |                                |               |         |       |
|     | ϵ θ   |       |     |      |     | rithm2,wesetτ | = τˆ       | ≪ T,insteadofthedefaultsetting |               |         |       |
|     |       | 2     |     | 2 I) |     |               | n          |                                |               |         |       |
z =N(xˆ ϵ ,σ −σ τ =T andwea lsoinitializez ∼N(x ,σ 2 I),wherex is
|     |     | τ m | a p | τ map |     | n                                                 |     |     |     | 0 τ ˆ | 0   |
| --- | --- | --- | --- | ----- | --- | ------------------------------------------------- | --- | --- | --- | ----- | --- |
|     |     | i − | 1   | 0     |     |                                                   |     |     |     |       |     |
| end |     |     |     |       |     | thecorrespondinggroundtruthimageforthemeasurement |     |     |     |       |     |
========PGDM======== y. FromTab.3,weobservesignificantimprovementsinthe
x pg d m =z performance. (Weuseτˆ=0.5,notethatσ 0 .5 =0.5,aswe
τ m
fo r i i n(m,m−1, ..1)do usethenoisescheduleσ = t). Thisshow s thatMAP-GA
t
|     |     |     |     |     |     | is only limited | by the | choice | of optimizer | and reinforces |     |
| --- | --- | --- | --- | --- | --- | --------------- | ------ | ------ | ------------ | -------------- | --- |
t=τpgdm
i
xˆ =D (z,t) the need for better optimization algorithms. While it
|     | t     | θ        |      |          |                |                   |                 |            |            |                  |             |
| --- | ----- | -------- | ---- | -------- | -------------- | ----------------- | --------------- | ---------- | ---------- | ---------------- | ----------- |
|     |       |          |      |          |                | i s i m p o r tan | t to c o n s id | e r b e tt | e r d e si | g n c h o ic e s | (f o r th e |
|     | µ =xˆ | + σ 2 ∗∇ | logP | (y | x ) | * p gdmupdate* |                   |                 |            |            |                  |             |
t t t x t t s ch e d u le s a nd hy p e r p ar a m e te r s ), a d a p tiv e -g r a d ien t- b as ed
|     | x        | = N (µ | ,σ 2     | − σ 2 | I ) |     |     |     |     |     |     |
| --- | -------- | ------ | -------- | ----- | --- | --- | --- | --- | --- | --- | --- |
|     | τ p g dm |        | t p g dm | pgdm  |     |     |     |     |     |     |     |
i − 1 τ i − 1 τ 0 optimizers (such as momentum, Adam), or higher-order
| end |     |     |     |     |     | methods, | we leave this | for future | work | as it requires | a   |
| --- | --- | --- | --- | --- | --- | -------- | ------------- | ---------- | ---- | -------------- | --- |
output:x
|     | τpgdm |     |     |     |     | thoroughanalysis. |     |     |     |     |     |
| --- | ----- | --- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- |
0
|     |     |     |     |     |     | In Tab.         | 4, we report | the | results    | on noisy inpainting |     |
| --- | --- | --- | --- | --- | --- | --------------- | ------------ | --- | ---------- | ------------------- | --- |
|     |     |     |     |     |     | using Algorithm | 3 on         | the | ImageNet64 | 1K validation       |     |
5.Experiments
|     |     |     |     |     |     | set. In Algorithm | 3, and | all its | variants, | we fix m | = 20 |
| --- | --- | --- | --- | --- | --- | ----------------- | ------ | ------- | --------- | -------- | ---- |
We consider the tasks of image inpainting, deblurring, time steps for PGDM. Even with high levels of mea-
4×
and super-resolution. Inspired by [13], for image surement noise, MAP-GA-PGDM shows promising
inpainting, we evaluate the performance across six dif- improvements over PGDM. In all our experiments, for
ferent mask settings (Fig. 1) denoting varying levels of MAP-GA and variants, we fix a budget of 1000 steps and
degradation. Themasksettingsbox50andbox25indicatea run ablations for (num steps,num iter) from the set
squarecropatthecenteroftheimage,withthecropwidth {(20,50),(50,20),(100,10),(200,5),(250,4),(500,2),(1000,1)}.
equalto50%and25%oftheimagewidthrespectively. In For DDRM, PGDM, and CT-ZSIE, we run ablations for
half,wemaskouttherighthalfoftheimage,expandisthe num steps from the set {20,50,100,200,250,500,1000}.
complementofbox25. sr2xdenotesa2×super-resolution Thelearningrateinallourexperimentswassettoσ2+σ2.
|     |     |     |     |     |     |     |     |     |     |     | y ϵ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
mask,andaltlinesmasksoutalternaterowsofpixels.
4158

|     |     |              |        | box50        | half        |              | expand | box25        | sr2x         | altlines    | deblur       | supres4x           |     |
| --- | --- | ------------ | ------ | ------------ | ----------- | ------------ | ------ | ------------ | ------------ | ----------- | ------------ | ------------------ | --- |
|     |     |              | Method | FID↓ LPIPS↓  | FID↓ LPIPS↓ | FID↓         | LPIPS↓ | FID↓ LPIPS↓  | FID↓ LPIPS↓  | FID↓ LPIPS↓ | FID↓ LPIPS↓  | FID↓ LPIPS↓        |     |
|     |     | MAP-GA(D,NP) |        | 32.312 0.096 | 37.986      | 0.157 93.068 | 0.419  | 11.019 0.019 | 30.072 0.034 | 18.282      | 0.015 19.704 | 0.007 66.968 0.148 |     |
|     |     | MAP-GA(D)    |        | 35.761 0.100 | 43.315      | 0.164 106.32 | 0.430  | 11.169 0.019 | 33.188 0.036 | 19.103      | 0.016 19.779 | 0.008 78.791 0.179 |     |
|     |     | MAP-GA(NP)   |        | 34.944 0.113 | 39.243      | 0.162 69.004 | 0.388  | 12.818 0.027 | 30.303 0.035 | 20.321      | 0.018 22.090 | 0.008 46.349 0.112 |     |
|     |     | MAP-GA       |        | 36.733 0.113 | 41.151      | 0.164 87.952 | 0.400  | 12.836 0.027 | 33.752 0.036 | 20.895      | 0.018 21.314 | 0.010 59.624 0.120 |     |
|     |     | PGDM[21]     |        | 49.370 0.151 | 54.261      | 0.245 127.95 | 0.479  | 14.255 0.021 | 38.433 0.046 | 20.446      | 0.019 19.857 | 0.007 89.614 0.238 |     |
|     |     | DDRM[12]     |        | 51.477 0.165 | 56.643      | 0.264 136.06 | 0.492  | 15.000 0.023 | 35.033 0.041 | 19.331      | 0.017 23.195 | 0.009 78.712 0.235 |     |
|     |     | CT-ZSIE[23]  |        | 38.017 0.129 | 44.152      | 0.191 70.634 | 0.424  | 13.040 0.025 | 42.500 0.060 | 26.737      | 0.029 29.223 | 0.013 56.698 0.116 |     |
Table1.NoiselessimagerestorationonImageNet641KvalidationsetusingMAP-GAandvariants.ThesettingMAP-GAdenotesthedefault
Algorithm2,MAP-GA(NP)denotesMAP-GAwithnoprior,MAP-GA(D)denotesMAP-GAwithdenoiserreplacingtheconsistencymodel,
andMAP-GA(D,NP)denotesMAP-GAwithnopriorandthedenoiserreplacingtheconsistencymodel.
|     |     |              |        | box50        | half        |              | expand | box25        | sr2x         | altlines    | deblur       | supres4x           |     |
| --- | --- | ------------ | ------ | ------------ | ----------- | ------------ | ------ | ------------ | ------------ | ----------- | ------------ | ------------------ | --- |
|     |     |              | Method | FID↓ LPIPS↓  | FID↓ LPIPS↓ | FID↓         | LPIPS↓ | FID↓ LPIPS↓  | FID↓ LPIPS↓  | FID↓ LPIPS↓ | FID↓ LPIPS↓  | FID↓ LPIPS↓        |     |
|     |     | MAP-GA(D,NP) |        | 70.880 0.128 | 68.022      | 0.269 174.96 | 0.695  | 18.221 0.028 | 22.819 0.059 | 13.632      | 0.036 81.461 | 0.214 59.895 0.205 |     |
|     |     | MAP-GA(D)    |        | 74.995 0.137 | 75.748      | 0.280 197.91 | 0.696  | 16.795 0.025 | 24.420 0.063 | 13.669      | 0.033 86.585 | 0.220 63.308 0.214 |     |
|     |     | MAP-GA(NP)   |        | 98.749 0.165 | 86.975      | 0.297 155.55 | 0.645  | 30.788 0.046 | 24.434 0.054 | 16.126      | 0.034 80.337 | 0.211 49.622 0.146 |     |
|     |     | MAP-GA       |        | 108.87 0.171 | 85.936      | 0.291 175.27 | 0.652  | 34.711 0.053 | 25.666 0.056 | 15.139      | 0.036 84.971 | 0.214 68.156 0.169 |     |
|     |     | PGDM[21]     |        | 120.82 0.194 | 94.920      | 0.360 227.97 | 0.765  | 28.357 0.041 | 27.927 0.070 | 14.211      | 0.037 94.629 | 0.227 77.533 0.248 |     |
|     |     | DDRM[12]     |        | 131.86 0.198 | 101.86      | 0.379 224.93 | 0.778  | 29.571 0.041 | 23.686 0.065 | 12.040      | 0.026 105.28 | 0.266 75.923 0.251 |     |
|     |     | CT-ZSIE[23]  |        | 118.27 0.209 | 121.11      | 0.375 200.03 | 0.704  | 34.246 0.046 | 47.353 0.145 | 24.663      | 0.070 97.343 | 0.264 50.608 0.157 |     |
Table2. Noiselessimagerestorationon100LSUNCat256imagesusingMAP-GAandvariants. ThesettingMAP-GAdenotesthedefault
Algorithm2,MAP-GA(NP)denotesMAP-GAwithnoprior,MAP-GA(D)denotesMAP-GAwithdenoiserreplacingtheconsistencymodel,
andMAP-GA(D,NP)denotesMAP-GAwithnopriorandthedenoiserreplacingtheconsistencymodel.
Figure1.Lefttoright:originalimage,andmasksettings:box50,half,expand,box25,sr2x,altlines
6.Discussion
|     |     |     |     |     |     |     |     |     | oretical | motivation, | and | connects the PF ODE, | the consis- |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | --- | -------------------- | ----------- |
tencymodelwiththeMAPoptimizationforsolvinginverse
6.1.Runtimecomparison problems. UnlikeZSIRandDMPlug,weconsidertheprior
termandshowthatthegradientofthelog-prioristractable,
| Tab.     | 5 compares | the   | wall-clock | time | of      | MAP-GA | and   |     |                                               |     |     |     |        |
| -------- | ---------- | ----- | ---------- | ---- | ------- | ------ | ----- | --- | --------------------------------------------- | --- | --- | --- | ------ |
|          |            |       |            |      |         |        |       |     | makingthegradientofthelog-posteriortractable. |     |     |     | Weshow |
| variants | against    | DDRM, | PGDM,      | and  | CT-ZSIE | for    | image |     |                                               |     |     |     |        |
inpainting on ImageNet64. To ensure a fair comparison, that MAP-GA is only limited by the optimizer’s choice in
|         |           |      |          |        |         |         |     |     | practice. | ZSIRandDMPlugreplacethePFODEtrajectory |     |     |     |
| ------- | --------- | ---- | -------- | ------ | ------- | ------- | --- | --- | --------- | -------------------------------------- | --- | --- | --- |
| we keep | the batch | size | fixed at | 50 for | all the | methods | and |     |           |                                        |     |     |     |
originwithamulti-stepdenoiserapproximationandrequire
| comparetheirruntimeperiteration. |     |     |     | ForDDRM,CT-ZSIE, |     |     |     |     |     |     |     |     |     |
| -------------------------------- | --- | --- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
backpropagationthroughthechainofcascadedfunctionsto
| and PGDM, | it is   | the total | runtime | divided | by    | num        | steps |     |                  |     |             |                        |     |
| --------- | ------- | --------- | ------- | ------- | ----- | ---------- | ----- | --- | ---------------- | --- | ----------- | ---------------------- | --- |
|           |         |           |         |         |       |            |       |     | optimizetheloss. |     | Incontrast, | MAP-GAvariantsrequirea |     |
| i.e. the  | reverse | diffusion | time    | steps,  | while | for MAP-GA |       |     |                  |     |             |                        |     |
and variants, it is the effective runtime per num steps singlevector-Jacobianproductperiteration.
| per num  | iter (i.e. | it     | is the runtime |              | when | num steps | =     |     |              |     |     |     |     |
| -------- | ---------- | ------ | -------------- | ------------ | ---- | --------- | ----- | --- | ------------ | --- | --- | --- | --- |
| num iter | = 1).      | MAP-GA |                | and variants | are  | 1.5×      | to 2× |     | 7.Conclusion |     |     |     |     |
slowerperiterationthanPGDMand3×to4×slowerthan
Inthispaper,weproposedanovelMAPformulationfor
DDRMandCT-ZSIE.
|     |     |     |     |     |     |     |     |     | solving          | inverse | problems                             | using pre-trained | unconditional |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | ------- | ------------------------------------ | ----------------- | ------------- |
|     |     |     |     |     |     |     |     |     | diffusionmodels. |         | Notethatconditionalgenerationisacore |                   |               |
6.2.Concurrentworks
|     |     |     |     |     |     |     |     |     | requirement | in  | solving inverse | problems. | We connect the |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --------------- | --------- | -------------- |
The algorithms presented in this paper are similar to ProbabilityFlowODEandtheconsistencymodelwiththe
ZSIR [4], and DMPlug [29] from a practical perspective. optimizationprocessfortheMAPobjectiveintasksthatin-
However, ourMAP formulationis novel, hasa strong the- volve conditional generation. We showed that the gradi-
4159

|     |     |              |     | box50  | half         |              | expand | box25        |              | sr2x         | altlines |     |
| --- | --- | ------------ | --- | ------ | ------------ | ------------ | ------ | ------------ | ------------ | ------------ | -------- | --- |
|     |     | Method       |     | FID↓   | LPIPS↓ FID↓  | LPIPS↓ FID↓  | LPIPS↓ | FID↓ LPIPS↓  | FID↓         | LPIPS↓ FID↓  | LPIPS↓   |     |
|     |     | MAP-GA(D,NP) |     | 26.661 | 0.043 30.621 | 0.064 73.510 | 0.155  | 9.165 0.010  | 29.878       | 0.027 16.971 | 0.012    |     |
|     |     | MAP-GA(D)    |     | 34.689 | 0.052 41.098 | 0.083 90.768 | 0.175  | 10.891 0.011 | 34.334       | 0.031 18.718 | 0.013    |     |
|     |     | MAP-GA(NP)   |     | 25.205 | 0.043 27.208 | 0.047 42.671 | 0.075  | 10.936 0.017 | 27.496       | 0.023 18.732 | 0.013    |     |
|     |     | MAP-GA       |     | 29.508 | 0.045 33.653 | 0.050 73.318 | 0.106  | 11.257 0.016 | 34.364       | 0.026 20.041 | 0.013    |     |
|     |     | PGDM[21]     |     | 31.952 | 0.056 36.639 | 0.082 71.352 | 0.138  | 10.886       | 0.009 33.199 | 0.041 19.551 | 0.018    |     |
Table3. NoiselessinpaintingonImageNet641Kvalidationset. Usingthegroundtruthimage(x )forthemeasurementy,wecreatea
0
sampleatt=0.5via(x =x +0.5∗η,where,η∼N(0,I))andinitializeAlgorithm2withz=x ,andτ =0.5
|     |     | 0.5            | 0   |             |              |             |              |              |              |             | 0.5 n        |     |
| --- | --- | -------------- | --- | ----------- | ------------ | ----------- | ------------ | ------------ | ------------ | ----------- | ------------ | --- |
|     |     |                |     |             | box50        | half        | expand       | box25        |              | sr2x        | altlines     |     |
|     |     | Method         |     | σy          | FID↓ LPIPS↓  | FID↓ LPIPS↓ | FID↓ LPIPS↓  | FID↓         | LPIPS↓       | FID↓ LPIPS↓ | FID↓ LPIPS↓  |     |
|     |     | MAP-GA-PGDM(D) |     | 0.05 56.173 | 0.125 62.026 | 0.193       | 108.16 0.438 | 38.725       | 0.039        | 57.41 0.080 | 46.046 0.046 |     |
|     |     | MAP-GA-PGDM    |     | 0.05 58.283 | 0.147 61.588 | 0.184       | 91.508       | 0.406 44.667 | 0.085 57.308 | 0.073       | 45.265 0.042 |     |
|     |     | PGDM[21]       |     | 0.05 77.824 | 0.175 80.248 | 0.257       | 135.99 0.495 | 53.136       | 0.049 86.289 | 0.126       | 66.203 0.066 |     |
|     |     | MAP-GA-PGDM(D) |     | 0.1 72.543  | 0.166 79.535 | 0.230       | 114.51 0.464 | 57.556       | 0.076 76.733 | 0.145       | 65.154 0.096 |     |
|     |     | MAP-GA-PGDM    |     | 0.1 74.130  | 0.191 78.322 | 0.225       | 103.85       | 0.440 65.720 | 0.161 76.248 | 0.134       | 63.134 0.089 |     |
|     |     | PGDM[21]       |     | 0.1 96.485  | 0.216 99.170 | 0.286       | 145.40 0.519 | 78.620       | 0.100 109.47 | 0.231       | 90.925 0.138 |     |
Table4. NoisyinpaintingonImageNet641Kvalidationset. σ denotesthemeasurementnoise. ThesettingMAP-GA-PGDMdenotesthe
y
defaultAlgorithm3,MAP-GA-PGDM(D)denoteMAP-GA-PGDMwithdenoiserreplacingtheconsistencymodel.
|     | Method       |     | Runtimeperiteration |       |     |     |     |     |     |     |     |     |
| --- | ------------ | --- | ------------------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | DDRM[12]     |     |                     | 150ms |     |     |     |     |     |     |     |     |
|     | CT-ZSIE[23]  |     |                     | 150ms |     |     |     |     |     |     |     |     |
|     | PGDM[21]     |     |                     | 304ms |     |     |     |     |     |     |     |     |
|     | MAP-GA(D,NP) |     |                     | 456ms |     |     |     |     |     |     |     |     |
|     | MAP-GA(D)    |     |                     | 602ms |     |     |     |     |     |     |     |     |
|     | MAP-GA(NP)   |     |                     | 455ms |     |     |     |     |     |     |     |     |
|     | MAP-GA       |     |                     | 603ms |     |     |     |     |     |     |     |     |
Table5.RuntimecomparisononNVIDIAA10040GBGPU
|     |     |     |     |     |     |     | Figure3.Noisyinpaintingtask(σ |     |     |     | =0.1).Lefttoright:original, |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------------- | --- | --- | --- | --------------------------- | --- |
y
maskedimage,restoredimagesusingMAP-GA-PGDM,PGDM.
|     |     |     |     |     |     |     | work | in practice, |     | we proposed | an algorithm | with a multi- |
| --- | --- | --- | --- | --- | --- | --- | ---- | ------------ | --- | ----------- | ------------ | ------------- |
stepgradientascentstrategyforMAPoptimization.Weval-
idatedouralgorithmswithextensiveexperimentsonimage
deblurring,super-resolution,andinpainting.
Acknowledgements
|     |     |     |     |     |     |     |     | This work | was | funded by | the Marie Skłodowska-Curie |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | --------- | -------------------------- | --- |
ActionsprojectMODELAIR,throughgrantagreementno.
|                                   |     |     |     |              |                |     | 101072559. |     | Thecomputationsandthedatahandlingwere |          |                 |          |
| --------------------------------- | --- | --- | --- | ------------ | -------------- | --- | ---------- | --- | ------------------------------------- | -------- | --------------- | -------- |
| Figure2. Noiselessinpaintingtask. |     |     |     | Lefttoright: | originalimage, |     |            |     |                                       |          |                 |          |
|                                   |     |     |     |              |                |     | enabled    | by  | resources                             | provided | by the National | Academic |
maskedimage,restoredimagesusingMAP-GA,PGDM. InfrastructureforSupercomputinginSweden(NAISS),par-
|     |     |     |     |     |     |     | tially            | funded | by  | the Swedish | Research               | Council through |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | ------ | --- | ----------- | ---------------------- | --------------- |
|     |     |     |     |     |     |     | grantagreementno. |        |     | 2022-06725. | BharaththanksSebastian |                 |
ent of the MAP objective is tractable, allowing the use of GerardandHengFangfortheirfeedbackonimprovingthe
| gradient-based | optimization |     | methods. |     | To use our | frame- | paperpresentation. |     |     |     |     |     |
| -------------- | ------------ | --- | -------- | --- | ---------- | ------ | ------------------ | --- | --- | --- | --- | --- |
4160

References [15] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor
|     |     |     |     |     |     |     | Darrell, | and Alexei | A Efros. |     | Context encoders: |     | Feature |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | -------- | --- | ----------------- | --- | ------- |
O¨ktem,
[1] Simon Arridge, Peter Maass, Ozan and Carola- learning by inpainting. In Proceedings of the IEEE con-
| Bibiane       | Scho¨nlieb.                 | Solving | inverse | problems | using | data- |                 |          |        |     |                      |     |       |
| ------------- | --------------------------- | ------- | ------- | -------- | ----- | ----- | --------------- | -------- | ------ | --- | -------------------- | --- | ----- |
|               |                             |         |         |          |       |       | ference on      | computer | vision | and | pattern recognition, |     | pages |
| drivenmodels. | ActaNumerica,28:1–174,2019. |         |         |          | 1     |       |                 |          |        |     |                      |     |       |
|               |                             |         |         |          |       |       | 2536–2544,2016. |          | 5      |     |                      |     |       |
[2] Ashish Bora, Ajil Jalal, Eric Price, and Alexandros G Di- [16] Xinyu Peng, Ziyang Zheng, Wenrui Dai, Nuoqian Xiao,
makis. Compressedsensingusinggenerativemodels. InIn- ChenglinLi,JunniZou,andHongkaiXiong.Improvingdif-
ternationalconferenceonmachinelearning,pages537–546.
fusionmodelsforinverseproblemsusingoptimalposterior
| PMLR,2017. | 1   |     |     |     |     |     |             |                                           |     |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- | --- | ----------- | ----------------------------------------- | --- | --- | --- | --- | --- |
|            |     |     |     |     |     |     | covariance. | InForty-firstInternationalConferenceonMa- |     |     |     |     |     |
[3] Benjamin Boys, Mark Girolami, Jakiw Pidstrigach, Sebas- chineLearning,2024. 3
tian Reich, Alan Mosca, and O Deniz Akyildiz. Tweedie [17] DaniloRezendeandShakirMohamed.Variationalinference
moment projected diffusions for inverse problems. arXiv InInternationalconferenceonma-
withnormalizingflows.
| preprintarXiv:2310.06721,2023. |     |     |     | 3   |     |     |                                         |     |     |     |     |     |     |
| ------------------------------ | --- | --- | --- | --- | --- | --- | --------------------------------------- | --- | --- | --- | --- | --- | --- |
|                                |     |     |     |     |     |     | chinelearning,pages1530–1538.PMLR,2015. |     |     |     |     | 1   |     |
[4] HamadiChihaoui,AbdelhakLemkhenter,andPaoloFavaro. [18] OlgaRussakovsky,JiaDeng,HaoSu,JonathanKrause,San-
Zero-shotimagerestorationviadiffusioninversion,2024. 3, jeevSatheesh,SeanMa,ZhihengHuang,AndrejKarpathy,
5,7
|     |     |     |     |     |     |     | Aditya Khosla, |     | Michael | Bernstein, | et al. | Imagenet | large |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ------- | ---------- | ------ | -------- | ----- |
[5] Hyungjin Chung, Jeongsol Kim, Michael Thompson Mc- scalevisualrecognitionchallenge. Internationaljournalof
cann,MarcLouisKlasky,andJongChulYe. Diffusionpos- computervision,115:211–252,2015. 6
teriorsamplingforgeneralnoisyinverseproblems. InInter- [19] ChitwanSaharia,WilliamChan,HuiwenChang,ChrisLee,
| nationalConferenceonLearningRepresentations,2023. |     |     |     |     |     | 1,  |          |         |           |       |        |              |     |
| ------------------------------------------------- | --- | --- | --- | --- | --- | --- | -------- | ------- | --------- | ----- | ------ | ------------ | --- |
|                                                   |     |     |     |     |     |     | Jonathan | Ho, Tim | Salimans, | David | Fleet, | and Mohammad |     |
3
|     |     |     |     |     |     |     | Norouzi. | Palette: | Image-to-image |     | diffusion | models. | In  |
| --- | --- | --- | --- | --- | --- | --- | -------- | -------- | -------------- | --- | --------- | ------- | --- |
[6] HyungjinChung,SuhyeonLee,andJongChulYe. Decom- ACMSIGGRAPH2022conferenceproceedings,pages1–10,
| poseddiffusionsamplerforacceleratinglarge-scaleinverse |     |     |     |     |     |     | 2022. 1 |     |     |     |     |     |     |
| ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- |
problems. arXivpreprintarXiv:2303.05754,2023. 3 [20] JaschaSohl-Dickstein, EricWeiss, NiruMaheswaranathan,
[7] Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and and Surya Ganguli. Deep unsupervised learning using
JongChulYe. Improvingdiffusionmodelsforinverseprob- nonequilibrium thermodynamics. In International confer-
lemsusingmanifoldconstraints. AdvancesinNeuralInfor- enceonmachinelearning,pages2256–2265.PMLR,2015.
| mationProcessingSystems,35:25683–25696,2022. |     |     |     |     |     | 3   | 2   |     |     |     |     |     |     |
| -------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
[8] DonaldGemanandChengdaYang. Nonlinearimagerecov- [21] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan
erywithhalf-quadraticregularization. IEEEtransactionson Kautz. Pseudoinverse-guided diffusion models for inverse
image processing : a publication of the IEEE Signal Pro- problems. InInternationalConferenceonLearningRepre-
cessingSociety,47:932–46,1995. 3 sentations,2023. 1,3,5,6,7,8
[9] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing [22] Yang Song and Prafulla Dhariwal. Improved tech-
Xu,DavidWarde-Farley,SherjilOzair,AaronCourville,and niques for training consistency models. arXiv preprint
Yoshua Bengio. Generative adversarial nets. Advances in arXiv:2310.14189,2023. 3
neuralinformationprocessingsystems,27,2014. 1 [23] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya
[10] JonathanHo,AjayJain,andPieterAbbeel. Denoisingdif- Sutskever. Consistency models. arXiv preprint
fusionprobabilisticmodels. Advancesinneuralinformation arXiv:2303.01469,2023. 2,3,4,5,6,7,8
processingsystems,33:6840–6851,2020. 2 [24] YangSongandStefanoErmon.Generativemodelingbyesti-
matinggradientsofthedatadistribution.Advancesinneural
| [11] Tero Karras, | Miika | Aittala, | Timo | Aila, and | Samuli | Laine. |     |     |     |     |     |     |     |
| ----------------- | ----- | -------- | ---- | --------- | ------ | ------ | --- | --- | --- | --- | --- | --- | --- |
Elucidating the design space of diffusion-based generative informationprocessingsystems,32,2019. 3
models.Advancesinneuralinformationprocessingsystems, [25] Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon.
35:26565–26577,2022. 2,4,5,6 Slicedscorematching: Ascalableapproachtodensityand
|                    |         |     |               |        |     |         | score estimation. |     | In Uncertainty |     | in Artificial | Intelligence, |     |
| ------------------ | ------- | --- | ------------- | ------ | --- | ------- | ----------------- | --- | -------------- | --- | ------------- | ------------- | --- |
| [12] Bahjat Kawar, | Michael |     | Elad, Stefano | Ermon, | and | Jiaming |                   |     |                |     |               |               |     |
Song. Denoising diffusion restoration models. Advances pages574–584.PMLR,2020. 2
inNeuralInformationProcessingSystems,35:23593–23606, [26] YangSong,JaschaSohl-Dickstein,DiederikPKingma,Ab-
2022. 1,3,6,7,8 hishekKumar,StefanoErmon,andBenPoole. Score-based
|     |     |     |     |     |     |     | generative | modeling | through | stochastic | differential |     | equa- |
| --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | ------- | ---------- | ------------ | --- | ----- |
[13] AndreasLugmayr,MartinDanelljan,AndresRomero,Fisher
Yu,RaduTimofte,andLucVanGool. Repaint: Inpainting tions. arXivpreprintarXiv:2011.13456,2020. 2
usingdenoisingdiffusionprobabilisticmodels. InProceed- [27] PascalVincent. Aconnectionbetweenscorematchingand
ings of the IEEE/CVF conference on computer vision and denoising autoencoders. Neural computation, 23(7):1661–
|                                           |     |     |     |     |     |     | 1674,2011. | 2   |     |     |     |     |     |
| ----------------------------------------- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- |
| patternrecognition,pages11461–11471,2022. |     |     |     |     | 6   |     |            |     |     |     |     |     |     |
[14] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav [28] RicardoVinuesaandStevenLBrunton. Enhancingcompu-
Shyam,PamelaMishkin,BobMcGrew,IlyaSutskever,and tationalfluiddynamicswithmachinelearning. NatureCom-
MarkChen. Glide:Towardsphotorealisticimagegeneration putationalScience,2(6):358–366,2022. 1
andeditingwithtext-guideddiffusionmodels.arXivpreprint [29] HengkangWang,XuZhang,TaihuiLi,YuxiangWan,Tian-
arXiv:2112.10741,2021. 1 congChen,andJuSun.Dmplug:Aplug-inmethodforsolv-
4161

| inginverseproblemswithdiffusionmodels. |             |         |             | arXivpreprint |     |
| -------------------------------------- | ----------- | ------- | ----------- | ------------- | --- |
| arXiv:2405.16749,2024.                 |             | 1,3,5,7 |             |               |     |
| [30] Yinhuai                           | Wang, Jiwen | Yu, and | Jian Zhang. | Zero-shot     | im- |
agerestorationusingdenoisingdiffusionnull-spacemodel.
| arXivpreprintarXiv:2212.00490,2022. |     |     | 3   |     |     |
| ----------------------------------- | --- | --- | --- | --- | --- |
[31] RaymondAYeh,ChenChen,TeckYianLim,AlexanderG
| Schwing, | Mark Hasegawa-Johnson, |      | and Minh        | N       | Do. Se- |
| -------- | ---------------------- | ---- | --------------- | ------- | ------- |
| mantic   | image inpainting       | with | deep generative | models. | In      |
ProceedingsoftheIEEEconferenceoncomputervisionand
| patternrecognition,pages5485–5493,2017. |     |     |     | 5   |     |
| --------------------------------------- | --- | --- | --- | --- | --- |
[32] FisherYu,YindaZhang,ShuranSong,AriSeff,andJianx-
| iong                                         | Xiao. Lsun: | Construction | of a large-scale |     | image  |
| -------------------------------------------- | ----------- | ------------ | ---------------- | --- | ------ |
| datasetusingdeeplearningwithhumansintheloop. |             |              |                  |     | ArXiv, |
| abs/1506.03365,2015.                         |             | 6            |                  |     |        |
[33] YuanzhiZhu,KaiZhang,JingyunLiang,JiezhangCao,Bi-
| hanWen,RaduTimofte,andLucVanGool.             |     |     |     | Denoisingdif- |        |
| --------------------------------------------- | --- | --- | --- | ------------- | ------ |
| fusionmodelsforplug-and-playimagerestoration. |     |     |     |               | InPro- |
ceedingsoftheIEEE/CVFConferenceonComputerVision
| andPatternRecognition,pages1219–1229,2023. |     |     |     | 3   |     |
| ------------------------------------------ | --- | --- | --- | --- | --- |
4162