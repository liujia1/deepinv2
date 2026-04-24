1
Solving Inverse Problems in Imaging with
Diffusion Models
Matthew M. Sato, satomm@stanford.edu
Abstract—Diffusionmodelsareapromisingmethodforsolvinginverseproblemsinimaging.Givenanoisymeasurement,diffusion
modelscanincrementallydenoisetheimagebyremovingsomeofthenoiseateachstepoftheprocess.Inthisproject,thenotation
anddiffusionmodelprocessesusedbyvariousapproachesareunified.Then,avarietyofdiffusionmodelapproachesareappliedto
theinverseproblem,withvaryinglevelsofconditioningonthenoisymeasurement.Theseapproachesaredemonstratedonadiffusion
modeltrainedonhumanfaces.Theabilitytogenerateanimageofahumanfacewithoutconditioningonthenoisymeasurementisfirst
shownasabaseline.Then,SDEdit,amethodwhichusesaheuristictoguidetheimagetowardstheoriginalnoisyimage,isexamined.
Last,threemethods(ILVR,ScoreALD,andDPS)whichexplicitlyestimatethegradientoftheloglikelihoodareinvestigatedand
compared.Ultimately,thisprojectshowsthatdiffusionmodelsareaneffectivemethodforsolvinginverseproblemsinimaging.
IndexTerms—ComputationalImaging,InverseProblems,DiffusionModels
✦
1 INTRODUCTION
NOISE in images is a longstanding problem, with im- models are explained and notation used by different meth-
perfections or limitations of cameras adding noise to ods are unified. The forward noising process and uncondi-
images. For example, a moving camera or moving object tional image generation is shown as a baseline. A heuristic
maycreatemotionblur,animproperlyfocusedcameramay approach (SDEedit) which guides the image generation
createablurryimage,oranobstructionmayblockapartof towards the original image without explicit conditioning is
an image. Solving the inverse problem, or reconstructing a investigated. Last, three methods (ILVR, ScoreALD, DPS)
denoisedimagefromthenoisyimage,isadifficultproblem that solve the inverse problem by conditioning on a noisy
because there are an infinite number of possible solutions capturedimagearecompared.Theremainderofthispaper
basedonthecapturedimage. includes Related Work in Section 2, Methods in Section 3,
Althoughmanyapproacheshavebeenproposedtosolve Evaluation Metrics in Section 4, Results in Section 5, and a
inverseproblemsinimaging,therecentadventofdiffusion DiscussionandConclusioninSection6.
models show significant promise. Diffusion models are a
generativemachinelearningmethodandlearntoiteratively
2 RELATED WORK
denoiseanimage,transformingnoisetoaclearerrepresen-
tation of an image at each step of the process. To train a Theinverseproblemforimagingisnotnew,withmanyap-
diffusion model, noise is gradually added to images and a proachesproposedovertheyears.Popularmethodsinclude
deepneuralnetworklearnstoreversethisprocess.Diffusion optimization-basedapproaches,suchastheHalfQuadratic
models are most popular for their applications in image Splitting(HQS)methodandtheAlternatingDirectionMeth-
andvideogeneration[1],wherediffusionmodelstransform ods of Multipliers (ADMM) [6]. More recently, with the
noise into a realistic image. As investigated in this project, explosion of computing power and data, neural networks
diffusionmodelscanbeextendedtosolveinverseproblems trained under supervised learning have been proposed for
byconditioningtheprocessonanoisymeasuredimage. solvingtheinverseproblem[7].However,theHQS/ADMM
Exactly conditioning the denoising process on the mea- approaches canbe slowto converge(if at all)and thedeep
suredimageisintractable.However,bothheuristicsandap- learning approach performs poorly for out-of-distribution
proximationshavebeenproposedfortheconditioningstep. samples.
This project compares different heuristics/approximations Generative adversarial networks (GANs) are a related
and evaluates their effectiveness for solving inverse prob- method that are also generative. GANs simultaneously
lems, specifically the inpainting and deconvolution prob- trainsageneratortogenerateanimageandadiscriminator
lems. Since diffusion models are relatively new for solv- to detect generated images. GANs have been proposed for
ing inverse problems, this project does not compare the solving inverse problems for imaging [8]. However, GANs
diffusion models to existing approaches, but rather com- arenotoriouslyunstableanddifficulttotrain,limitingtheir
pares different diffusion model appraoches. In particular, practicaluse.
theSDEdit[2],ILVR[3],ScoreALD[4],andDPS[5]methods Morerecently,diffusionmodelshavebeenproposedfor
arecompared. image generation. The first description of using diffusion
Inthisproject,unconditionedandconditioneddiffusion models for images was by [9], where the basic framework
fordiffusionmodelswasproposed.Besidestheapproaches
for inverse problems examined in this project, a non-
• ThisisthefinalprojectfortheWinter2025iterationofEE367atStanford
exhaustive list of other techniques include Score-SDE [10],

2
ΠGDM[11],BlindDPS[12],andMomentMatching[13].All Algorithm1ReverseDiffusion
ofthesemethodsconditionthediffusionmodelbyexplicitly 1: x T ∼N(0,I)
makingsomeapproximationtomatchthemeasurement. 2: fort=T to1do
3: z ∼N(0,I)ift>1,elsez =0
3 METHODS 4: xˆ 0 = √1 α¯√t (x t +(1−α¯ t )s θ √ (x t ,t))
In this section, the basic formulation of diffusion models 5 6 : : x x ′ t t − − 1 1 = =x′ t α − t 1 1 (1 − + − α¯ α¯ t ζ t t − g 1 ( ) x x t t , + y) α¯t− 1− 1( α¯ 1 t −αt)xˆ 0 +σz
for solving inverse problems is presented. Since notation 7: endfor
canvarywidely,theapproachesareunifiedwithacommon 8: return x 0
notation. Then, the methods examined in this project for
solvinginverseproblemsaredescribed.
Although the two formulations are equivalent, the denois-
3.1 DiffusionModels ing step using (4) and (5) is used in the remainder of this
project.
The diffusion model can be separated into a forward noise
Thereversediffusionprocessdescribedhasbeenformu-
process and a reverse denoising process. In the forward
lated using the score function, s θ (x t ,t); however, a noise-
noiseprocess,noiseisgraduallyaddedtoanimageusinga
forwardnoisemodel.Inthiscase,x 0istheoriginalunnoisy p
ca
r
s
e
e
d
,
ic
a
ti
s
o
in
n
g
n
le
et
r
w
ev
o
e
r
r
k
s
,
e
ϵ
d
ϕ (
e
x
n
t
o
,
i
t
s
)
in
,c
g
a
s
n
te
b
p
e
i
l
s
earnedinstead.Inthis
image and x t is the image after t steps of added noise.
This project uses the variance-preserving (VP) formulation 1 (cid:18) 1−α (cid:19)
ofdiffusionmodels.IntheVPformulationprovidedby[9], x t−1 = √ α x t − √ 1−α¯ t ϵ θ (x t ,t) , (7)
t t
theforwardnoisemodelisdescribedby
where the equivalence of the formulation in (6) and (7)
(cid:112) (cid:112)
x t = 1−β t x t−1 + β t z t−1 , (1) is derived in the Appendix. Since these approaches are
equivalent, the score function formulation is used for the
where β t is the noise schedule and z t−1 ∼ N(0,I). This
remainderofthisproject.
forwardnoisemodeliscomputationallyefficientforanytif
rewrittenintoanequivalentformulationdependingonlyon
theoriginalimageandasinglenoiseterm: 3.2 ConditionedDiffusionModels
√ √
x t = α¯ t x 0 + 1−α¯ t z, (2) Thus far, the unconditional denoising process has been de-
scribed.However,tosolveaninverseproblem,thediffusion
where α t = 1−β t, α¯ t =
(cid:81)t
i=1 α i, and z ∼ N(0,I). The process should be conditioned on the original noisy mea-
derivationforthisformulationisprovidedintheAppendix. surement,y.Theimageformationmodelisy = A(x)+n,
Inthereversedenoisingprocess,adiffusionmodelitera- where A(·) is the noisy measurement operator and n is
tivelyreversesthenoisingprocessof(2).Tweedie’sformula zeromeanGaussiannoise.AlthoughA(·)canbebothnon-
providestheestimateforthecompletelydenoisedimageat
linearandlinear,onlylinearoperatorsareconsideredinthis
timet:
project since only some of the methods are applicable for
1 non-linearoperators.
xˆ 0 = √ α¯ (x t +(1−α¯ t )∇ xt logp t (x t )) (3) To condition the diffusion process on the measurement,
t
The learnable function of a diffusion model is the score
∇
xt
logp
t
(x
t
)in(3)isreplacedwith∇
xt
logp
t
(x
t
|y).Using
Baye’srule,
function,s
θ
(x
t
,t),whichistrainedtomatch∇
xt
logp
t
(x
t
).
Details on training the score function are beyond the scope ∇ xt logp t (x t |y)=∇ xt logp t (x t )+∇ xt logp t (y|x t ) (8)
ofthisproject.Usingthisscorefunction,theapproximation (cid:124) (cid:123)(cid:122) (cid:125)
forthedenoisedimageis
sθ(xt,t)
xˆ 0 = √
1
α¯ (x t +(1−α¯ t )s θ (x t ,t)) (4)
T
tr
h
a
e
cta
g
b
r
l
a
e
d
,
i
r
e
e
n
q
t
u
o
ir
f
in
th
g
e
an
log
ap
l
p
ik
ro
el
x
i
i
h
m
oo
at
d
io
,∇
n.
xt
T
l
h
o
e
gp
a
t
p
(
p
y
r
|
o
x
a
t
c
)
h
,
e
i
s
s
e
in
x-
-
t
plored in this project use various approximations for
andtheincrementaldenoisingstepcanbewrittenas ∇ xt logp t (y|x t ). To create a common notation for the dif-
√ √ ferentapproaches,let
α (1−α¯ ) α¯ (1−α )
x t−1 = t 1−α¯ t t−1 x t + t− 1 1 −α¯ t t xˆ 0 +σz (5) ζ t g(x t ,y)≈∇ xt logp t (y|x t ), (9)
wheretheGaussiannoisezaddsrobustnessandguarantees
where g(x t ,y) is the approximation for the gradient of the
unique denoising steps when the algorithm is run repeat-
loglikelihoodandζ tisanapproximationdependentscaling
edly. An alternative but equivalent denoising step can be
term. Then, the denoising process described in (4) and (5)
derived by substituting xˆ 0 from (4) into (5) (derived in the can be updated to include the measurement conditioning.
Appendix), such that a single step of the reverse diffusion
The complete denoising process is summarized in Algo-
processcanbewritten:
rithm 1 and the remaining subsections details the different
1 heuristics or approximations used by different approaches
x t−1 = √ α t (x t +(1−α t )s θ (x t ,t))+σz. (6) forg(x t ,y)andζ t.

3
TABLE1 wheretheestimatecanbecomputedwithbackpropagation.
Theposteriorapproximationandscaleforthevariousmethods. The approximation in (12) is only correct for t = 0, and
theauthorsproposeanannealingterm,γ t,thatreducesthe
Method g(xt,y) ζt scaleζ
t
forlargetwhentheapproximationispoor.Theset
SDEdit 0 0
ILVR ϕN(y t−1 )−ϕN(x′ t−1 ) 1 of annealing terms {γ t } is a hyperparameter that must be
ScoreALD −∇xt ∥y−A(xt)∥2
2 σ2+
1
γt 2
chosenforeachproblem.
DPS −∇xt ∥y−A(xˆ0)∥2
2 ∥y−A
ζ
(xˆ0)∥2
3.3.4 DPS
TheDPSmethod[5]estimatesthegradientoftheloglikeli-
3.3 ConditionedDiffusionModelMethods
hoodtermsimilarlytoScoreALD.However,insteadofusing
In this subsection, the different approaches for solving the x t,DPSusestheestimateofthedenoisedimaged,xˆ 0:
conditioned diffusion problem for inverse problems are
described. The choices for g(x t ,y) and ζ t are summarized ζ g(x ,y)≈∇ logp (y|x )≈−ζ ∇ ∥y−A(xˆ )∥2,
inTable1forallthemethodsdescribedinthissubsection.
t t xt t t t xt 0 2
(13)
where xˆ 0 is computed as in line 4 of Algorithm 1 and
3.3.1 SDEdit
the gradient is computed with backpropagation. The ζ t
The first method is SDEdit [2], which does not attempt to term is a hyperparameter, and the authors suggest using
a
to
pp
gu
ro
id
xi
e
m
th
a
e
te
re
∇
v
x
er
t
s
l
e
og
di
p
f
t
f
(
u
y
s
|
i
x
on
t )
p
,
r
b
o
u
c
t
es
i
s
ns
to
te
w
a
a
d
rd
u
s
se
th
s
e
a
m
h
e
e
a
u
s
r
u
is
re
ti
d
c
t
ζ
h
t
o
=
rsq
∥
u
y
a
−
n
A ζ
t
(
i
xˆ
fy
0)
a
∥
n
, w
up
it
p
h
er
ζ
b
∈
ou
[
n
0
d
.1 e
,
r
1
r
.
o
0
r
].
o
F
n
u
t
r
h
th
e
e
a
r
p
m
p
o
r
r
o
e
x
,
im
th
a
e
ti
a
o
u
n
-
,
image. SDEdit starts at an intermediate denoising step and althoughthedetailsarebeyondthescopeofthisproject.
instead of initializing x T with random noise, SDEdit com-
binesthenoisewiththemeasuredimagebyusing(2).Thus,
line1inAlgorithm1isreplacedwith
√ √ 4 EVALUATION AND COMPARISON OF THE METH-
x T = α¯ T y+ 1−α¯ T z, (10) ODS
where z ∼ N(0,I). The remaining steps remain the same,
To evaluate the different solution methods for solving in-
with g(x t ,y) ≡ 0 and ζ t = 0 ∀ t. The starting step T verseproblemswithdiffusionmodels,twometricsareused:
is a hyperparameter which should be tuned. A larger T
peak signal-to-noise-ratio (PSNR) and Learned Perceptual
resultsinmoreinitialnoiseleadingtolargerrealism,while
Image Patch Similarity (LPIPS) distance. PSNR is a com-
asmallerT haslessnoiseandismorefaithfultotheoriginal
monly used metric to compare different image signals and
measurement.
isdefinedas
3.3.2 ILVR
(cid:18)MAX2(cid:19)
PSNR=10log , (14)
ILVR [3] is the first method investigated in this project that 10 MSE
explicitly conditions the denoising on the measurement.
ILVR uses a low-pass filtering operation, ϕ N (·), which is whereMAX isthemaximumpossiblevalueofapixeland
MSE isthemeansquarederrorbetweentheimageandthe
a sequence of downsampling and upsampling by a factor
of N. The goal of ILVR is to match the downsampled ground truth. Although PSNR is a commonly used metric
and can evaluate the similarity between two images, the
versionofthegeneratedimagetothedownsampledversion
of the measured image: ϕ N (x 0 ) = ϕ N (y). This matching metric is criticized for not representing visual quality well,
sinceareconstructionwithalargePSNRmaynotlookgood
is enforced at each step of the reverse diffusion process:
ϕ N (x t )=ϕ N (y t ),wherey t usestheforwardnoiseprocess toahumanobserver.
LPIPS, however, is a better metric for analyzing the
describedin(2).Thus,theapproximationbecomes
visual quality. LPIPS compares the distance between acti-
g(x t ,y)=ϕ N (y t−1 )−ϕ N (x′ t−1 ), (11) vation layers from a neural network for the reconstructed
image and the ground truth [14]. The key idea for LPIPS is
with ζ t = 1. This process can be viewed as removing
that the activation layers of a neural network aligns more
thelow-frequencycomponentofthecurrentapproximation
closely to the features that the human eye observes. Thus,
andaddingthelow-frequencycomponentoftheassociated
a lower LPIPS value indicates a better reconstruction of a
measurement.Thiscorrectiontermisonlyappliedfort>b,
noisyimage.
wherebisthestoppingtimeforthecorrection.Thestopping
Two different forward noise models are considered for
time, b, and the downsampling rate, N, are both hyperpa-
evaluatingthemethods: inpaintinganddeconvolution.For
rametersofthismethod.
inpainting,a50×50pixelboxismaskedoutoftheoriginal
image. In deconvolution, a Gaussian blur kernel of size
3.3.3 ScoreALD
61 and standard deviation 3 is applied to the image. The
In ScoreALD [4], the gradient of the log likelihood is esti-
PSNRandLPIPSvaluesarecomputedforeachmethodand
matedbyusingthecurrentx t:
the reconstructed images are qualitatively compared. The
1 different methods are all evaluated on the same image for
ζ g(x ,y)≈∇ logp (y|x )≈− ∇ ∥y−A(x )∥2
t t xt t t σ2+γ2 xt t 2 consistency. The ground truth and noisy measurements of
t
(12) theimagetobetestedisshowninFig.1.

4
Fig.1.Thegroundtruth,inpainting,anddenconvolutionimages.
5 RESULTS
| In this section, |     | the proposed |     | methods | are | evaluated. | First, |     |     |     |     |     |     |     |     |
| ---------------- | --- | ------------ | --- | ------- | --- | ---------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
unconditionalgenerationwithadiffusionmodelisdemon-
stratedalongwithestimatesofadenoisedimage.Then,the
| results from | the | proposed | methods, |     | SDEdit, | ILVR, | Score- |     |     |     |     |     |     |     |     |
| ------------ | --- | -------- | -------- | --- | ------- | ----- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
ALD,andDPS,areshownandcompared.
5.1 DiffusionModel
| A pretrained | diffusion |         | model | from                | [5] is | used. The | dif-   |                                                        |     |     |     |     |     |     |        |
| ------------ | --------- | ------- | ----- | ------------------- | ------ | --------- | ------ | ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | ------ |
|              |           |         |       |                     |        |           |        | Fig.2.ImagescreatedthroughunconditionaldenoisingusingT |     |     |     |     |     |     | =1000. |
| fusion model | is        | trained | on    | the Flickr-Faces-HQ |        |           | (FFHQ) |                                                        |     |     |     |     |     |     |        |
dataset[15],adatasetwithawidevariationofhumanfaces.
TABLE2
| This diffusion | model | is  | trained | to learn | the | score function, |     |     |     |     |     |     |     |     |     |
| -------------- | ----- | --- | ------- | -------- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Evaluationmetricsfortheestimateddenoisedimageatvariousnoising
s θ (x t ,t) = ∇ xt logp t (x t ), rather than the noise function. steps.
Implementationofthediffusionmodelandinverseproblem
| methodsareimplementedwithPyTorch. |               |     |         |         |     |                 |     |     |     | Human |        | RedPanda |        |     |     |
| --------------------------------- | ------------- | --- | ------- | ------- | --- | --------------- | --- | --- | --- | ----- | ------ | -------- | ------ | --- | --- |
|                                   |               |     |         |         |     |                 |     |     | t   | PSNR  | LPIPS  | PSNR     | LPIPS  |     |     |
| First,                            | the diffusion |     | model’s | ability | to  | unconditionally |     |     |     |       |        |          |        |     |     |
|                                   |               |     |         |         |     |                 |     |     | 30  | 37.4  | 0.0360 | 34.4     | 0.0473 |     |     |
generateanimagefromnoiseisdemonstrated.Specifically,
|     |     |     |     |     |         |     |     |     | 100 | 32.6 | 0.0886 | 29.2 | 0.219 |     |     |
| --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | ---- | ------ | ---- | ----- | --- | --- |
|     |     |     |     |     | g(x ,y) | = 0 |     |     |     |      |        |      |       |     |     |
no posterior sampling is used and t in line 6 300 27.2 0.204 25.1 0.574
|              |     |          |      |        |     |            |     |     | 500 | 23.5 | 0.327 | 21.4 | 0.603 |     |     |
| ------------ | --- | -------- | ---- | ------ | --- | ---------- | --- | --- | --- | ---- | ----- | ---- | ----- | --- | --- |
| of Algorithm | 1.  | Starting | from | x 1000 | ∼   | N(0,I) and | un- |     |     |      |       |      |       |     |     |
conditionallydenoisingfor1000steps,Fig.2showsseveral
| examples      | of denoised     | images.  |        | As seen   | in    | Fig. 2, each | time   |                                    |         |              |         |        |                |        |           |
| ------------- | --------------- | -------- | ------ | --------- | ----- | ------------ | ------ | ---------------------------------- | ------- | ------------ | ------- | ------ | -------------- | ------ | --------- |
|               |                 |          |        |           |       |              |        | 5.2.1                              | SDEdit  |              |         |        |                |        |           |
| the algorithm | is              | executed | a      | different | image | is generated |        |                                    |         |              |         |        |                |        |           |
|               |                 |          |        |           |       |              |        | SDEditisimplementedwithstarttimesT |         |              |         |        | ={250,500,750} |        |           |
| due to the    | random          | starting | noise. | Fig.      | 2     | demonstrates | the    |                                    |         |              |         |        |                |        |           |
|               |                 |          |        |           |       |              |        | to demonstrate                     |         | the tradeoff | between |        | realism        | and    | faithful- |
| importance    | of conditioning |          | the    | denoising |       | process      | on the |                                    |         |              |         |        |                |        |           |
|               |                 |          |        |           |       |              |        | ness. The                          | results | shown        | in Fig. | 4 show | that           | with a | smaller   |
measurementtosolvetheinverseprobleminsteadofgener-
|     |     |     |     |     |     |     |     | starting | time, the | denoised | image | is  | more | faithful | to the |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | --------- | -------- | ----- | --- | ---- | -------- | ------ |
atingarandomimage.
|     |     |     |     |     |     |     |     | originalmeasurementbutlessreal.ForT |     |     |     |     | =250,thisresults |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --- | ---------------- | --- | --- |
Next,theestimateofthedenoisedimageatagiventime
inapoordenoisingwiththedeconvolutionresultstillshow-
stepisinvestigated.First,noiseisaddedtotheimageattime
|     |     |     |     |     |     |     |     | ing blur | and the | inpainting | result | still | showing | a   | blocked |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ------- | ---------- | ------ | ----- | ------- | --- | ------- |
steptusing(2).Then,theestimateforthedenoisedimageis
|          |       |         |         |     |               |     |       | out region. | As the | starting | time | becomes | larger, | the | image |
| -------- | ----- | ------- | ------- | --- | ------------- | --- | ----- | ----------- | ------ | -------- | ---- | ------- | ------- | --- | ----- |
| computed | using | (4). An | example | of  | this estimate | is  | shown |             |        |          |      |         |         |     |       |
becomemorerealbutlessfaithfultothemeasurement.The
| for a human   | face      | and a       | red panda |            | face in  | Fig. 3, with    | the |                              |                                      |          |                   |                        |       |          |       |
| ------------- | --------- | ----------- | --------- | ---------- | -------- | --------------- | --- | ---------------------------- | ------------------------------------ | -------- | ----------------- | ---------------------- | ----- | -------- | ----- |
|               |           |             |           |            |          |                 |     | imagesforT                   | =750showthemosthumanlookingfaces,but |          |                   |                        |       |          |       |
| resulting     | PSNR      | and LPIPS   | shown     |            | in Table | 2. Predictably, |     |                              |                                      |          |                   |                        |       |          |       |
|               |           |             |           |            |          |                 |     | do not                       | resemble                             | the girl | from              | the measurement.       |       | Choosing |       |
| the PSNR      | and LPIPS | show        | better    | results    |          | when predicting |     |                              |                                      |          |                   |                        |       |          |       |
|               |           |             |           |            |          |                 |     | anintermediatestartingtime,T |                                      |          |                   | =500,resultsinacompro- |       |          |       |
|               | t,        |             |           |            |          |                 |     | t                            |                                      |          |                   |                        |       |          |       |
| for a smaller |           | since there | is        | less noise | in       | the image.      | As  |                              |                                      |          |                   |                        |       |          |       |
|               |           |             |           |            |          |                 |     | mise between                 | realism                              |          | and faithfulness. |                        | These | images   | don’t |
becomeslarger,thepredicteddenoisedimageisfartherfrom
|     |     |     |     |     |     |     |     | show any | residual | deconvolution |     | or  | inpainting, | but | only |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------- | ------------- | --- | --- | ----------- | --- | ---- |
thegroundtruth.Asexpected,theresultsfortheredpanda
somewhatresembletheoriginalgirl.
areworsethanforthehumanface,whichcanbeattributed
|     |     |     |     |     |     |     |     | The | quantitative | results | summarized |     | in  | Table | 3 reflect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ------- | ---------- | --- | --- | ----- | --------- |
tothediffusionmodelbeingtrainedonhumanfacesrather
|     |     |     |     |     |     |     |     | the tradeoff | between |     | realism and | faithfulness. |     | The | PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ------- | --- | ----------- | ------------- | --- | --- | ---- |
thananimalfaces.
andLPIPSshowthebestvaluesforasmallT,showingthat
|     |     |     |     |     |     |     |     | these are    | the most | faithful | to the          | ground | truth.  | What     | these |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | -------- | -------- | --------------- | ------ | ------- | -------- | ----- |
|     |     |     |     |     |     |     |     | quantitative | measures |          | fail to capture |        | are the | realness | that  |
5.2 ResultsofInvestigatedMethods
|     |     |     |     |     |     |     |     | arelackingintheresultsforT |     |     | =250. |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | ----- | --- | --- | --- | --- |
ThissubsectionshowstheresultsfromusingSDEdit,ILVR,
|     |     |     |     |     |     |     |     | 5.2.2 | ILVR |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- | ---- | --- | --- | --- | --- | --- | --- |
ScoreALD,andDPSontheinverseproblemshowninFig.1.
Thequantitativeresults(PSNRandLPIPS)aresummarized ILVR is implemented for downsampling/upsampling rates
foreachmethodinTable3. of N = {4,8,16}. The denoising process starts from T =

5
Fig.3.Estimatesofthedenoisedimagesatvariouslevelsofnoise.
TABLE3
AsummaryofthePSNRandLPIPSforalltheinvestigatedmethods.
| SDEdit |       | ILVR         |            | ScoreALD   | DPS    |       |
| ------ | ----- | ------------ | ---------- | ---------- | ------ | ----- |
| T PSNR | LPIPS | N PSNR LPIPS | AnnealSch. | PSNR LPIPS | ζ PSNR | LPIPS |
250 23.5 0.139 4 20.8 0.197 [10,15] 24.3 0.110 0.1 29.7 0.073
Inpainting 500 20.4 0.186 8 20.2 0.189 [17,22] 26.3 0.079 0.3 34.6 0.028
| 750 14.6 | 0.410 | 16 19.6 0.240 |     |     | 1 36.3 | 0.010 |
| -------- | ----- | ------------- | --- | --- | ------ | ----- |
250 23.8 0.183 4 23.3 0.180 [10,15] 23.8 0.138 0.1 25.1 0.091
Deconvolution 500 20.2 0.233 8 23.3 0.144 [15,20] 21.7 0.158 0.3 27.0 0.078
| 750 14.2 | 0.400 | 16 20.7 0.192 |     |     | 1 28.3 | 0.054 |
| -------- | ----- | ------------- | --- | --- | ------ | ----- |
Fig.5.TheresultsfromILVR.TheILVRadjustmentwasstoppedatt=
300fordeconvolutionandt=500forinpainting.
Fig.4.TheresultsfromSDEditusingdifferentstartingtimes.
1000, and the conditioning is applied for t ∈ [300,1000] are summarized in Table 3. Like SDEdit, faithfulness de-
for deconvolution and t ∈ [500,1000] for inpainting. The creasesasN increases.Alargerdownsamplingfactorleads
results are shown in Fig. 5 and the quantitative results to a correction that is less like the original measurement.

6
Fig.6.TheresultsfromScoreALDusingdifferentannealingschedules,
γt.
| However, | with | too small | of a | downsampling |     | (N = | 4), the |     |     |     |     |     |     |     |
| -------- | ---- | --------- | ---- | ------------ | --- | ---- | ------- | --- | --- | --- | --- | --- | --- | --- |
noiseoftheoriginalmeasurementisnotfilteredenoughand
| artifacts | of noise      | remain   | in the  | denoised | image:        | blurriness |          |     |     |     |     |     |     |     |
| --------- | ------------- | -------- | ------- | -------- | ------------- | ---------- | -------- | --- | --- | --- | --- | --- | --- | --- |
| for the   | deconvolution |          | problem | and      | an inpainting |            | artifact |     |     |     |     |     |     |     |
| for the   | inpainting    | problem. | The     | PSNR     | and           | LPIPS      | values   |     |     |     |     |     |     |     |
Fig.7.TheresultsfromDPSusingdifferentscalevalues,ζ.
| show that | the      | intermediate |              | N =      | 8 downsampling |      | rate  |     |     |     |     |     |     |     |
| --------- | -------- | ------------ | ------------ | -------- | -------------- | ---- | ----- | --- | --- | --- | --- | --- | --- | --- |
| images    | have the | best         | quantitative | results, | where          | this | level |     |     |     |     |     |     |     |
ofdownsamplingguidesthedenoisingprocesstowardsthe agoodreconstruction.Theapproximationofg(x ,y)clearly
t
measurementwhileremovingthedeconvolutionorinpaint- does a good job in guiding the diffusion process towards
ing. ILVR results in PSNR and LPIPS that are comparable thegroundtruth,diminishingthedependenceonchoosing
| withSDEditanddonotshowsignificantimprovement. |     |     |     |     |     |     |     | agoodζ. |     |     |     |     |     |     |
| --------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- |
5.2.3 ScoreALD
|     |     |     |     |     |     |     |     | 6 DISCUSSION |     | AND | CONCLUSION |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | --- | ---------- | --- | --- | --- |
ScoreALDisimplementedandtestedfordifferentannealing
schedules with T = 1000. The final denoised result are In this project, several methods are shown for solving in-
| highly sensitive |     | to the | annealing | schedule |     | and this | hyper- |                |     |        |           |                   |     |      |
| ---------------- | --- | ------ | --------- | -------- | --- | -------- | ------ | -------------- | --- | ------ | --------- | ----------------- | --- | ---- |
|                  |     |        |           |          |     |          |        | verse problems |     | with a | diffusion | model conditioned |     | on a |
parameter must be tuned to yield acceptable results. The noisy measurement image. The methods are evaluated on
annealing schedule is a linear schedule γ t ∈ [a,b], where a noisy human face. First, the SDEdit and ILVR methods
| γ = b | and γ | = a. | The results | for | deconvolution |     | and |                                                   |     |     |     |     |     |     |
| ----- | ----- | ---- | ----------- | --- | ------------- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| T     |       | 0    |             |     |               |     |     | areshown.SDEditdoesnotapproximatethegradientofthe |     |     |     |     |     |     |
inpaintingundervariousannealingschedulesareshownin log likelihood, and suffers from a tradeoff between realism
Fig. 6 and the quantitative results are in Table 3. ScoreALD andfaithfulness.ILVRattemptstomatchthelow-frequency
results in images that most closely resemble the ground components of the reconstructed image and the measured
| truth of | all methods | thus | far. | The PSNR | and | LPIPS | show |        |            |         |         |       |        |           |
| -------- | ----------- | ---- | ---- | -------- | --- | ----- | ---- | ------ | ---------- | ------- | ------- | ----- | ------ | --------- |
|          |             |      |      |          |     |       |      | image. | ILVR shows | varying | results | based | on the | choice of |
significant improvements compared to SDEdit and ILVR. downsamplingrate.
However, the importance of choosing a proper annealing DPS produces the best results on the test image. The
schedulemustbeemphasized,sinceapoorannealingsched-
|     |     |     |     |     |     |     |     | DPS approximation |     | for | the gradient | of the | log likelihood |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | --- | ------------ | ------ | -------------- | --- |
ulemayresultinpoorresultssuchastheinpaintingsolution has proven error bounds, which may contribute to its su-
for γ ∈ [10,15]. Because the approximation error of the perior performance. DPS is also notable due to its low
t
| gradient                                | of log | likelihood | may | be large, | a large | enough     | γ   |                                                  |                   |     |         | ζ.  |        |      |
| --------------------------------------- | ------ | ---------- | --- | --------- | ------- | ---------- | --- | ------------------------------------------------ | ----------------- | --- | ------- | --- | ------ | ---- |
|                                         |        |            |     |           |         |            |     | t sensitivity                                    | to hyperparameter |     | choice, | The | second | best |
| isrequiredtoreducethecontributionsofg(x |        |            |     |           |         | ,y)whenthe |     |                                                  |                   |     |         |     |        |      |
|                                         |        |            |     |           |         | t          |     | methodisScoreALD,withanapproximationsimilartoDPS |                   |     |         |     |        |      |
approximationerrorislarge. that is less accurate. ScoreALD requires a good choice of
annealingscheduletoproduceconsistentresults.WhileDPS
5.2.4 DPS
|     |     |     |     |     |     |     |     | and ScoreALD | outperform |     | the other | methods, | these | two |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | ---------- | --- | --------- | -------- | ----- | --- |
DPSisimplementedandevaluatedusingT =1000anddif- methodsarealsomorecomputationallyexpensive.DPSand
ferentscales,ζ.AsshowninFig.7,theresultingimagesare ScoreALDrequirebackpropagationforcomputingg(x ,y),
t
veryclosetotheoriginalgroundtruth.Additionally,thede- whichresultsinaslowerdenoisingprocess.Note,however,
noisedimagesarenotverysensitivetothehyperparameter, that the original ScoreALD paper [4] shows a closed form
ζ.ThequantitativeresultsinTable3supportthequalitative solutiong(x ,y)≈AH(y−Ax )thatmaybefaster.
|     |     |     |     |     |     |     |     |     | t   |     | t   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
results, with the PSNR and LPIPS outperforming all other Ultimately, this project has shown various approaches
methods no matter the choice of ζ. The choice of scale, ζ, to solving the inverse problem for imaging using diffu-
doesaffecttheresults,butevenapoorchoiceofζ produces sion models. As shown, generative diffusion models can

7
reconstruct high quality images from a noisy ground truth, whereα t = 1−β t andz i ∼ N(0,I). Thisformulation can
providinganewmethodforsolvinginverseproblems. be rewritten as a conditional distribution that depends on
onlyt=0:
REFERENCES
√ (cid:112)
[1] H.Cao,C.Tan,Z.Gao,Y.Xu,G.Chen,P.-A.Heng,andS.Z.Li, x = α x + β z
t t t−1 t t−1
“ASurveyonGenerativeDiffusionModels,”IEEETransactionson √ √ (cid:112) (cid:112)
KnowledgeandDataEngineering,vol.36,no.7,pp.2814–2830,Jul. = α t ( α t−1 x t−2 + β t−1 z t−2 )+ β t z t−1
2024. √ (cid:112) (cid:112)
= α α x + α β z + β z
[2] C.Meng,Y.He,Y.Song,J.Song,J.Wu,J.-Y.Zhu,andS.Ermon, t t−1 t−2 t t−1 t−2 t t−1
“SDEdit: Guided Image Synthesis and Editing with Stochastic .
.
DifferentialEquations,”inInternationalConferenceonLearningRep- .
resentations,2022. (cid:118)
[3] J.Choi,S.Kim,Y.Jeong,Y.Gwon,andS.Yoon,“ILVR:Condition- √ (cid:112) (cid:112) (cid:117) (cid:117)(cid:89) t
ingMethodforDenoisingDiffusionProbabilisticModels,”in2021 = α¯ t x 0 + β t z t−1 + α t β t−1 z t−2 +···+(cid:116) α i β 1 z 0
IEEE/CVFInternationalConferenceonComputerVision(ICCV),2021,
i=2
pp.14347–14356. (cid:124) (cid:123)(cid:122) (cid:125)
[4] A.Jalal,M.Arvinte,G.Daras,E.Price,A.G.Dimakis,andJ.Tamir, (⋆)
“RobustCompressedSensingMRIwithDeepGenerativePriors,”
inAdvancesinNeuralInformationProcessingSystems,M.Ranzato,
A.Beygelzimer,Y.Dauphin,P.S.Liang,andJ.W.Vaughan,Eds.,
vol.34,2021,pp.14938–14954. where α¯ t =
(cid:81)t
i=1 α i. Recall, z i ∼ N(0,I), and using the
[5] H.Chung,J.Kim,M.T.Mccann,M.L.Klasky,andJ.C.Ye,“Diffu- propertiesofnormaldistributionsresultsin
sionPosteriorSamplingforGeneralNoisyInverseProblems,”in
InternationalConferenceonLearningRepresentations,2023.
[6] S.Boyd,N.Parikh,E.Chu,B.Peleato,andJ.Eckstein,“Distributed
Optimization and Statistical Learning via the Alternating Direc- t
(cid:89)
tion Method of Multipliers,” Foundations and Trends in Machine (⋆)∼N(0,(β +α β +α α β +···+ α β )I)
t t t−1 t t−1 t−2 i 1
Learning,vol.3,no.1,pp.1–122,2011.
i=2
[7] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, “Beyond a (cid:124) (cid:123)(cid:122) (cid:125)
Gaussian Denoiser: Residual Learning of Deep CNN for Image (⋄)
Denoising,” IEEE Transactions on Image Processing, vol. 26, no. 7,
pp.3142–3155,Jul.2017.
[8] V. Shah and C. Hegde, “Solving Linear Inverse Problems Using
Gan Priors: An Algorithm with Provable Guarantees,” in 2018 Simplifyingthecovarianceterm:
IEEE International Conference on Acoustics, Speech and Signal Pro-
cessing(ICASSP),Apr.2018,pp.4609–4613,iSSN:2379-190X.
[9] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic
t
models,”inProceedingsofthe34thInternationalConferenceonNeural (cid:89)
(⋄)=β +α β +α α β +···+ α β
InformationProcessingSystems,ser.NIPS’20,Dec.2020,pp.6840– t t t−1 t t−1 t−2 i 1
6851. i=2
[10] Y.Song,J.Sohl-Dickstein,D.P.Kingma,A.Kumar,S.Ermon,and t
(cid:89)
B. Poole, “Score-Based Generative Modeling through Stochastic =1−α +α β +α α β +···+ α β
t t t−1 t t−1 t−2 i 1
DifferentialEquations,”inInternationalConferenceonLearningRep-
resentations,2021. i=2
[11] J. Song, A. Vahdat, M. Mardani, and J. Kautz, “Pseudoinverse- (cid:32) t (cid:89) −1 (cid:33)
Guided Diffusion Models for Inverse Problems,” in International =1−α t 1−β t−1 −α t−1 β t−2 −···− α i β 1
ConferenceonLearningRepresentations,2023. i=2
[12] H.Chung,J.Kim,S.Kim,andJ.C.Ye,“ParallelDiffusionModels (cid:32) t−2 (cid:33)
of Operator and Image for Blind Inverse Problems,” in 2023 (cid:89)
=1−α α 1−β −···− α β
IEEE/CVF Conference on Computer Vision and Pattern Recognition t t−1 t−2 i 1
(CVPR),2023,pp.6059–6069. i=2
[13] F.Rozet,G.Andry,F.Lanusse,andG.Louppe,“LearningDiffu- .
.
sionPriorsfromObservationsbyExpectationMaximization,”in .
TheThirty-eighthAnnualConferenceonNeuralInformationProcessing =1−α α ···α (1−β )
Systems,2024. t t−1 2 1
[14] R.Zhang,P.Isola,A.A.Efros,E.Shechtman,andO.Wang,“The =1−α¯
t
UnreasonableEffectivenessofDeepFeaturesasaPerceptualMet-
ric,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern
Recognition,Jun.2018,pp.586–595,iSSN:2575-7075. √
[15] T.Karras,S.Laine,andT.Aila,“AStyle-BasedGeneratorArchi- Thus, (⋆) ∼ N(0,(1 − α¯ t )I) = 1−α¯ t N(0,I) and the
tecture for Generative Adversarial Networks,” IEEE Transactions forwarddiffusionstepcanbewrittenas
on Pattern Analysis and Machine Intelligence, vol. 43, no. 12, pp.
4217–4228,Dec.2021.
√ √
x = α¯ x + 1−α¯ z,
t t 0 t
APPENDIX
In this appendix, several derivations relevant for imple-
mentingthediffusionmodelareshown. wherez ∼N(0,I).
First, the equivalency of (1) and (2) is demonstrated. ■
Recall, the forward noise model of a variance-preserving
Next, the equivalency of the reverse diffusion process
diffusionmodelis: (cid:112)
(cid:112) (cid:112) √ (cid:112) o (cid:112) f (4)/(5) and (6) is sh √ own. First, note that α¯ t−1 /α¯ t =
x t = 1−β t x t−1 + β t z t−1 = α t x t−1 + β t z t−1 , α¯ t−1 /(α t α¯ t−1 ) = 1/ α t. Substituting xˆ 0 from (4) into

8
theexpressionforx in(5): Last, the equivalency of the reverse diffusion process
t−1
|     | √       |     |     |     |     |     | using | the score | function |     | and noise-prediction |       | network | is  |
| --- | ------- | --- | --- | --- | --- | --- | ----- | --------- | -------- | --- | -------------------- | ----- | ------- | --- |
|     | α (1−α¯ | )   |     |     |     |     |       |           |          |     |                      |       |         |     |
|     | t       | t−1 |     |     |     |     |       |           |          |     |                      | ϵ ( x | , t )   |     |
x t−1 = x t s h o w n . N o tic e t h at if s ϕ ( x t ,t ) = − √ θ t is d er i v e d , t h e
|     | 1−α¯ |     |     |     |     |     |     |     |     |     |     | 1   | − α¯ t |     |
| --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- |
√t e q u iv a le nc y o f ( 6) an d ( 7 ) i s a ls o d er i v e d . To d o t h i s , st a r t
|     |     | α¯   | (1−α | )   |     |     |     |     |     |     |     |     |     |     |
| --- | --- | ---- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | √t−1 |      | t   |     |     |     |     |     |     |     |     |     |     |
+ (x t +(1−α¯ t )s θ (x t ,t)) fromtheforwarddiffusionprocessof(2):
|     |           | α¯ (1−α¯  |       | )           |      |            |                                 |     |     | √         | √      |         |       |     |
| --- | --------- | --------- | ----- | ----------- | ---- | ---------- | ------------------------------- | --- | --- | --------- | ------ | ------- | ----- | --- |
|     | √         | t         | √t    |             |      |            |                                 |     |     |           |        |         |       |     |
|     | (cid:18)  |           |       |             |      | )(cid:19)  |                                 |     | x = | α¯ x      | + 1−α¯ | ϵ (x    | ,t)   |     |
|     | α t (1−α¯ | t−1       | )     | α¯ t−1 (1−α | t    |            |                                 |     | t   | t 0       |        | t ϕ     | t     |     |
| =   |           |           | +     | √           |      | x          |                                 |     |     |           |        |         |       |     |
|     | 1−α¯      |           |       | α¯ (1−α¯    | )    | t          |                                 |     |     |           |        |         |       |     |
|     |           | t         |       | √t          | t    |            | SubstitutingTweedie’sformula,   |     |     |           |        |         |       |     |
|     |           |           |       | α¯          | (1−α | )          |                                 |     |     |           |        |         |       |     |
|     |           |           | +     | t−√1        |      | t s (x ,t) |                                 |     | 1   |           |        |         |       |     |
|     |           |           |       |             |      | θ t        |                                 | x   | = √ | (x +(1−α¯ |        | )∇ logp | (x )) |     |
|     |           |           |       |             | α¯ t |            |                                 | 0   |     | t         |        | t xt    | t t   |     |
|     |           |           |       |             |      |            |                                 |     | α¯  | t         |        |         |       |     |
|     | 1         | (cid:18)√ |       |             | (1−α | )(cid:19)  |                                 |     |     |           |        |         |       |     |
| =   |           | α         | (1−α¯ | /α )+       |      | t x        |                                 |     | 1   |           |        |         |       |     |
|     |           | t         |       | t t         | √    | t          |                                 |     | = √ | (x +(1−α¯ |        | )s (x   | ,t)), |     |
|     | (1−α¯     | )         |       |             | α    |            |                                 |     |     | t         |        | t θ t   |       |     |
|     | t         |           |       |             |      | t          |                                 |     | α¯  | t         |        |         |       |     |
|     |           |           |       |             | (1−α | t )        |                                 |     |     |           |        |         |       |     |
|     |           |           |       |             | + √  | s (x ,t)   | intotheforwarddiffusionprocess: |     |     |           |        |         |       |     |
|     |           |           |       |             |      | α θ t      |                                 |     |     |           |        |         |       |     |
t
|     |       |         |     |       |      |            |       | √            | 1          |        |          |          | √      |        |
| --- | ----- | ------- | --- | ----- | ---- | ---------- | ----- | ------------ | ---------- | ------ | -------- | -------- | ------ | ------ |
|     | 1     | √       | √   |       | √    | √          | x     | = α¯         | √ (x       | +(1−α¯ | )s       | (x ,t)+  | 1−α¯ ϵ | (x ,t) |
| =   |       | ( α −α¯ | /   | α +1/ | α    | − α )x     | t     | t            |            | t      | t        | θ t      | t      | ϕ t    |
|     | (1−α¯ | ) t     | t   | t     | t    | t t        |       |              | α¯ t       |        |          |          |        |        |
|     | t     |         |     |       |      |            |       |              |            | √      |          |          |        |        |
|     |       |         |     |       | (1 − | α )        | ⇒ (1  | − α¯ t ) s θ | (x t ,t )= | 1 −    | α¯ t ϵ ϕ | (x t ,t) |        |        |
|     |       |         |     |       | + √  | t s (x ,t) |       |              |            |        |          |          |        |        |
|     |       |         |     |       |      | θ t        |       |              | 1          |        |          |          |        |        |
|     |       |         |     |       |      | α          | ⇒ s ( | x , t ) =    | √          | ϵ (    | x ,t )   |          |        |        |
|     |       |         |     |       |      | t          | θ     | t            |            | ϕ      | t        |          |        |        |
|     | 1     |         |     |       |      |            |       |              | 1−α¯       |        |          |          |        |        |
t
| =   | √ (x | t +(1−α | t )s θ | (x t ,t)), |     |     |                                      |     |     |     |     |     |     |     |
| --- | ---- | ------- | ------ | ---------- | --- | --- | ------------------------------------ | --- | --- | --- | --- | --- | --- | --- |
|     | α    |         |        |            |     |     | whichshowstheformulationsarethesame. |     |     |     |     |     |     |     |
t
■
whichistheformulationof(6).
■