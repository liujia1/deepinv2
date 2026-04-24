Inverse Problems
PAPER • OPEN ACCESS You may also like
A tutorial on inverse problems for anomalous -An inverse random source problem for the
time fractional diffusion equation driven by
a fractional Brownian motion
diffusion processes
Xiaoli Feng, Peijun Li and Xu Wang
-First passage time moments of
asymmetric Lévy flights
To cite this article: Bangti Jin and William Rundell 2015 Inverse Problems 31 035003 Amin Padash, Aleksei V Chechkin,
Bartomiej Dybiec et al.
-An undetermined coefficient problem for a
fractional diffusion equation
Zhidong Zhang
View the article online for updates and enhancements.
This content was downloaded from IP address 154.3.32.197 on 23/04/2026 at 07:28

InverseProblems
InverseProblems31(2015)035003(40pp) doi:10.1088/0266-5611/31/3/035003
A tutorial on inverse problems for
anomalous diffusion processes
Bangti Jin1 and William Rundell2
1DepartmentofComputerScience,UniversityCollegeLondon,GowerStreet,London
WC1E6BT, UK
2Department of Mathematics, Texas A&MUniversity, CollegeStation, TX 77843-
3368, USA
E-mail: bangti.jin@gmail.com andrundell@math.tamu.edu
Received 26September 2014,revised 11November 2014
Accepted forpublication 25November 2014
Published 10February 2015
Abstract
Over the last two decades, anomalous diffusion processes in which the mean
squares variance grows slower or faster than that in a Gaussian process have
found many applications. At a macroscopic level, these processes are ade-
quately described by fractional differential equations, which involves frac-
tional derivatives in time or/and space. The fractional derivatives describe
either history mechanism or long range interactions of particle motions at a
microscopic level. The new physics can change dramatically the behavior of
theforwardproblems.Forexample,thesolutionoperatorofthetimefractional
diffusiondiffusionequationhasonlylimitedsmoothingproperty,whereasthe
solution for the space fractional diffusion equation may contain weak singu-
larity. Naturally one expects that the new physics will impact related inverse
problems in terms of uniqueness, stability, and degree of ill-posedness. The
last aspect is especially important from a practical point of view, i.e., stably
reconstructing the quantities of interest. In this paper, we employ a formal
analytic and numerical way, especially the two-parameter Mittag-Leffler
function and singular value decomposition, to examine the degree of ill-
posedness of several ‘classical’ inverse problems for fractional differential
equationsinvolvingaDjrbashian–Caputofractionalderivativeineithertimeor
space, which represent the fractional analogues of that for classical integral
order differential equations. We discuss four inverse problems, i.e., backward
fractional diffusion, sideways problem, inverse source problem and inverse
potential problem for time fractional diffusion, and inverse Sturm–Liouville
ContentfromthisworkmaybeusedunderthetermsoftheCreativeCommons
Attribution3.0licence.Anyfurtherdistributionofthisworkmustmaintainattributionto
theauthor(s)andthetitleofthework,journalcitationandDOI.
0266-5611/15/035003+40$33.00 ©2015IOPPublishingLtd PrintedintheUK 1

InverseProblems31(2015)035003 BJinandWRundell
problem, Cauchy problem, backward fractional diffusion and sideways pro-
blemforspacefractionaldiffusion.Itisfoundthatcontrarytothewidebelief,
the influence of anomalous diffusion on the degree of ill-posedness is not
definitive: it can either significantly improve or worsen the conditioning of
related inverse problems, depending crucially on the specific type of given
data and quantity of interest. Further, the study exhibits distinct new features
of‘fractional’inverseproblems,andapartiallistofsurprisingobservationsis
given below. (a) Classical backward diffusion is exponentially ill-posed,
whereas time fractional backward diffusion is only mildly ill-posed in the
senseofnormsonthedomainandrangespaces.However,thisdoesnotimply
thatthelatteralwaysallowsamoreeffectivereconstruction.(b)Theoretically,
the time fractional sideways problem is severely ill-posed like its classical
counterpart, but numerically can be nearly well-posed. (c) The classical
Sturm–Liouville problem requires two pieces of spectral data to uniquely
determine a general potential, but in the fractional case, one single Dirichlet
spectrum may suffice. (d) The space fractional sideways problem can be far
more or far less ill-posed than the classical counterpart, depending on the
location of the lateral Cauchy data. In many cases, the precise mechanism of
these surprising observations is unclear, and awaits further analytical and
numericalexploration,whichrequiresnewmathematicaltoolsandingenuities.
Further, our findings indicate fractional diffusion inverse problems also pro-
vide an excellent case study in the differences between theoretical ill-con-
ditioning involving domain and range norms and the numerical analysis of a
finite-dimensionalreconstructionprocedure.Throughoutwewillalsodescribe
known analytical and numerical results in the literature.
Keywords: fractional inverse problem, fractional differential equation,
anomalous diffusion, Djrbashian–Caputo fractional derivative, Mittag-Leffler
function
(Some figures may appear in colour only in the online journal)
1. Introduction
Diffusion is one of the most prominent transport mechanisms found in nature. At a micro-
scopic level, it is related to the random motion of individual particles, and the use of the
Laplace operator and the first-order derivative in the canonical diffusion model rests on a
Gaussian process assumption on the particle motion, after Albert Einsteinʼs groundbreaking
work [23]. Over the last two decades a large body of literature has shown that anomalous
diffusion models in which the mean square variance grows faster (superdiffusion) or slower
(subdiffusion)thanthatinaGaussianprocessundercertaincircumstancescanofferasuperior
fit to experimental data (see the comprehensive reviews [5, 70, 72, 95] for physical back-
ground and practical applications). For example, anomalous diffusion is often observed in
materials with memory, e.g., viscoelastic materials, and heterogeneous media, such as soil,
heterogeneous aquifer, and underground fluid flow. At a microscopic level, the subdiffusion
process can be described by a continuous time random walk [75], where the waiting time of
particlejumpsfollowssomeheavytaileddistribution,whereasthesuperdiffusionprocesscan
be described byLévy flightsor Lévy walk, where the lengthof particle jumpsfollows some
2

| InverseProblems31(2015)035003 |     |     |     |     |     |     |     |     | BJinandWRundell |     |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- |
heavytaileddistribution,reflectingthelong-rangeinteractionsamongparticles.Followingthe
micro–macro
aforementioned correspondence, the macroscopic counterpart of a continuous
timerandomwalkisadifferentialequationwithafractionalderivativeintime,andthatfora
Lévy flight is a differential equation with a fractional derivative in space. We will refer to
thesetwocasesastimefractionaldiffusionandspacefractionaldiffusion,respectively,andit
is generically called a fractional derivative equation (FDE) below. In general the fractional
| derivative | can | appear in | both | time and | space variables. |     |     |     |     |     |
| ---------- | --- | --------- | ---- | -------- | ---------------- | --- | --- | --- | --- | --- |
Next we give the mathematical model in the simplest geometrical setting of one space
dimension, taking the domain Ω = (0, 1). Then a general, linear FDE is given by
|     |     | ∂αu −CDβu | +   | qu = f | (x, t) ∈ | Ω × (0, | T), |     |     | (1.1) |
| --- | --- | --------- | --- | ------ | -------- | ------- | --- | --- | --- | ----- |
|     |     | t 0       | x   |        |          |         |     |     |     |       |
isafixedtime,anditisequippedwithsuitableboundaryandinitialconditions.
| whereT | > 0 |     |     |     |     |     |     |     |     |     |
| ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
The fractional orders α ∈ (0, 1) and β ∈ (1, 2) are related to the parameters specifying the
large-time behavior of the waiting-time distribution or long-range behavior of the particle
β
jump distribution. For example, in hydrological studies, the parameter is used to
characterize the heterogeneity of porous medium [17]. In theory, these parameters can be
determined from the underlying stochastic model, but often in practice, they are determined
|                   |     |      |      |          |              |     | =CDα  | Djrbashian–Caputo |     |     |
| ----------------- | --- | ---- | ---- | -------- | ------------ | --- | ----- | ----------------- | --- | --- |
| from experimental |     | data | [34, | 35, 61]. | The notation | ∂α  |       | is the            |     |     |
|                   |     |      |      |          |              |     | t 0 t |                   |     |     |
derivativeoperatoroforderα 1)inthetimevariablet,andCDβdenotestheDjrbashian–
|     |     |     |     | ∈ (0, |     |     |     | 0 x |     |     |
| --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
Caputo derivative of order β ∈ (1, 2) in the space variable x. For a real number
n − 1 < γ < n, n ∈ , and f ∈ Hn(0, 1), the left-sided Djrbashian–Caputo derivative
| CDγ  |       | γ defined |     |        |     |     |     |     |     |     |
| ---- | ----- | --------- | --- | ------ | --- | --- | --- | --- | --- | --- |
| f of | order | is        | by  | [53, p | 91] |     |     |     |     |     |
0 x
|     |     |         | 1   |     | x                      |     |     |     |     |       |
| --- | --- | ------- | --- | --- | ---------------------- | --- | --- | --- | --- | ----- |
|     |     | CDγ f = |     | ∫   | (x − s)n−1−γf(n)(s)ds, |     |     |     |     | (1.2) |
0 x
|            |         |         | Γ(n − | γ) 0 |                  |     |     |     |     |     |
| ---------- | ------- | ------- | ----- | ---- | ---------------- | --- | --- | --- | --- | --- |
| where Γ(z) | denotes | Eulerʼs | Gamma |      | function defined | by  |     |     |     |     |
∞
|     |     | Γ(z) = ∫ | sz−1e−sds, |     | R(z) | > 0. |     |     |     |     |
| --- | --- | -------- | ---------- | --- | ---- | ---- | --- | --- | --- | --- |
0
| Djrbashian–Caputo |     |     |     |     | first |     |     |     |     |     |
| ----------------- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- |
The derivative was introduced by Armenian mathematician Mkhitar
M Djrbashian for studies on space of analytical functions and integral transforms in 1960s
[19–21]
(see for surveys on related works). Italian geophysicist Michele Caputo
independently proposed the use of the derivative for modeling the dynamics of viscoelastic
materialsin1967[10].Wenotethatthereareseveralalternative(anddifferent)definitionsof
fractionalderivatives,notablytheRiemann–Liouvillefractionalderivative,whichformallyis
obtainedfrom(1.2)byinterchangingtheorderofintegrationanddifferentiation,i.e.,theleft-
| Riemann–Liouville |     |     |     |     |     | RDγ |     |     |     |     |
| ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
sided fractional derivative f of order γ ∈ (n − 1, n), n ∈ , is
|         |         |       |         |      |      | 0 x     |         |     |     |     |
| ------- | ------- | ----- | ------- | ---- | ---- | ------- | ------- | --- | --- | --- |
| defined | by [53, | p 70] |         |      |      |         |         |     |     |     |
|         |         |       | dn      | 1    | x    |         |         |     |     |     |
|         |         | RDγ   |         |      | ∫    | s)n−1−γ |         |     |     |     |
|         |         | f =   |         |      | (x − |         | f(s)ds. |     |     |     |
|         |         | 0 x   | dxn Γ(n | − γ) |      |         |         |     |     |     |
0
Djrbashian–Caputo
In this work, we shall focus mostly on the derivative since it allows a
| convenient | treatment | of  | the boundary |     | and initial | conditions. |     |     |     |     |
| ---------- | --------- | --- | ------------ | --- | ----------- | ----------- | --- | --- | --- | --- |
Under certain regularity assumption on the functions, with an integer order γ, the
| Djrbashian–Caputo |     |     | Riemann–Liouville |     |             |      |         |           |          |       |
| ----------------- | --- | --- | ----------------- | --- | ----------- | ---- | ------- | --------- | -------- | ----- |
|                   |     | and |                   |     | derivatives | both | recover | the usual | integral | order |
derivative (see for example [79, p 100] for the Djrbashian–Caputo case). For example, with
|          |     | Djrbashian–Caputo |     |     |            | derivatives∂αu |     | andCDβu |            |      |
| -------- | --- | ----------------- | --- | --- | ---------- | -------------- | --- | ------- | ---------- | ---- |
| α = 1and | β = | 2, the            |     |     | fractional |                |     | t 0     | x coincide | with |
the usual first- and second-order derivatives ∂u and ∂2u, respectively, for which the model
|     |     |     |     |     | ∂t  | ∂x2 |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(1.1)recoversthestandardone-dimensionaldiffusionequation,andthusgenerallythemodel
3

InverseProblems31(2015)035003 BJinandWRundell
(1.1) is regarded as a fractional counterpart. The Djrbashian–Caputo derivative (and many
others)isanintegro-differentialoperator,andthusitisnonlocalinnature.Asaconsequence,
manyusefulrules,e.g.,productruleandintegrationbyparts,fromPDEsareeitherinvalidor
require significant modifications. The nonlocality underlies most analytical and numerical
challengesassociatedwiththemodel(1.1).Itsignificantlycomplicatesthemathematicaland
numerical analysis of the model, including relevant inverse problems.
Inafractionalmodel,thereareanumberofparameters,e.g.,fractionalorder(s),diffusion
and potential coefficients (when using a second-order elliptic operator in space), initial
condition,sourceterm,boundaryconditionsanddomaingeometry,thatcannotbemeasured/
specifieddirectly,andhavetobeinferredindirectlyfrommeasureddata.Typically,thedatais
the forward solution restricted to either the boundary or the interior of the physical domain.
This gives rise to a large variety ofinverse problems for FDEs, which have started to attract
muchattentioninrecentyears,sincethepioneeringwork[14].Aninterestingquestionishow
the nonlocal physics behind anomalous diffusion processes will influence the behavior of
related inverse problems, e.g., uniqueness, stability, and the degree of ill-posedness. The
degree of ill-posedness is especially important for developing practical numerical recon-
struction procedures. There is a now well known example of backward fractional diffusion,
i.e.,recoveringtheinitialconditioninatimefractionaldiffusionequationfromthefinaltime
data, which is only mildly ill-posed, instead of severely ill-posed for the classical backward
diffusion problem. In some sense, this example has led to the belief that ‘fractionalizing’
inverse problems can always mitigate the degree of ill-posedness, and thus allows a better
chance of an accurate numerical reconstruction.
In this paper, we examine the degree of ill-posedness of ‘fractional’ inverse problems
from a formal analytic and numerical point of view, and contrast their numerical stability
propertieswiththeirclassical,thatis,theGaussiandiffusioncounterparts,forwhichthereare
many deep analytical results [39, 40, 84]. Specifically, we revisit a number of ‘classical’
inverse problems for the FDEs, e.g., the backward diffusion problem, sideways diffusion
problem and inverse source problem, and numerically exhibit their degree of ill-posedness.
These examples indicate that the answer to the aforementioned question is not definitive: it
dependscruciallyonthetype(unknownanddata)oftheinverseproblemwelookat,andthe
nonlocality of the problem (fractional derivative) can either greatly improve or worsen the
degree of ill-posedness.
ThemathematicaltheoryofinverseproblemsforFDEsisstillinitsinfancy,andthusin
thiswork,weonlydiscussthetopicformallytogiveaflavorofinverseproblemsforFDEs—
our goal is to give insight rather than to pursue an in-depth analysis. The technical devel-
opments that are available we leave to the references cited. In addition, known theoretical
resultsandcomputationaltechniquesintheliteraturewillbebrieflydescribed,whichhowever
arenotmeanttobeexhaustive.Therestofthepaperisorganizedasfollows.Insection2we
reviewtwospecialfunctions,i.e.,Mittag-LefflerfunctionandWrightfunction,andtheirbasic
properties. The Mittag-Leffler function plays an extremely important role in understanding
anomalous diffusion processes. We also recall the basic tool—singular value decomposition
—for analyzing discrete inverse problems. Then in section 3 we study several inverse pro-
blems for FDEs with a time fractional derivative, including backward diffusion, inverse
source problem, sideways problem and inverse potential problem. In section 4 we consider
inverse problems for FDEs with a space fractional derivative, including the inverse Sturm–
Liouville problem, Cauchy problem, backward diffusion and sideways problem. In the
appendices,wegivetheimplementationdetailsofthecomputationalmethodsforsolvingthe
time-andspacefractionaldifferentialequations.Thesemethodsareemployedthroughoutfor
computingtheforwardmap(unknown-to-measurementmap)soastogaininsightintorelated
4

| InverseProblems31(2015)035003 |     |     |     |     |     |     |     | BJinandWRundell |     |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- |
inverse problems. Throughout the notation c, with or without a subscript, denote a generic
constant, which may differ at different occurrences, but it is always independent of the
| unknown | of interest. |     |     |     |     |     |     |     |     |
| ------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
2. Preliminaries
Mittag-Leffler
We recall two important special functions, function and Wright function, and
one useful tool for analyzing discrete ill-posed problems, singular value decomposition.
| 2.1. Mittag-Leffler | function |     |     |     |                |     |     |            |     |
| ------------------- | -------- | --- | --- | --- | -------------- | --- | --- | ---------- | --- |
|                     |          |     |     |     | Mittag-Leffler |     | E   | (z) (withα |     |
We shall use extensively the two-parameter function α,β > 0 and
defined
| β ∈ ) | by  | [53, equation | (1.8.17), | p   | 40] |     |     |     |     |
| ------ | --- | ------------- | --------- | --- | --- | --- | --- | --- | --- |
∞ zk
∑
|     | E (z) | =   |        | z   | ∈ . |     |     |     | (2.1) |
| --- | ----- | --- | ------ | --- | ---- | --- | --- | --- | ----- |
|     | α,β   |     | Γ(kα + | β)  |      |     |     |     |       |
k=0
Thisfunction with β = 1wasfirst introduced by Gösta Mittag-Leffler in1903[74]and then
| generalized | by others | [1, 36]. | It can | be verified | directly | that |      |     |     |
| ----------- | --------- | -------- | ------ | ----------- | -------- | ---- | ---- | --- | --- |
|             |           |          |        |             |          |      | sinh | z   |     |
ez,
|     | E (z) | =   | E (z) | = cosh | z,  | E (z) = |     | .   |     |
| --- | ----- | --- | ----- | ------ | --- | ------- | --- | --- | --- |
|     | 1,1   |     | 2,1   |        |     | 2,2     | z   |     |     |
Hence it represents a generalization of the exponential function in that E (z) = ez. The
1,1
Mittag-LefflerfunctionE (z)isanentirefunctionofzwithorderα−1andtype1[53,p40].
α,β
Further, the function E (−t) is completely monotone on the positive real axis+ [82], and
α,1
thus it is positive on +; see also [90] for extension to the two-parameter Mittag-Leffler
functionE (z).ItappearsinthesolutionrepresentationforFDEs:thefunctionsE (−λtα)
|          | α,β    |     |     |     |     |     |     |     | α,1 |
| -------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
| andtα−1E | (−λtα) |     |     |     |     |     |     |     |     |
α,α appear in the kernel of the time fractional diffusion problem with initial
data and the right-hand side, respectively, and xE (−λxα) and xα−1E (−λxα) are
|     |     |     |     |     |     | α,2 |     | α,α |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
eigenfunctionstothefractionalSturm–Liouvilleproblemwithazeropotential,cfsection4.1.
Inourdiscussions,theasymptoticbehaviorofthefunctionE (z)willplayacrucialrole.
α,β
It satisfies the following exponential asymptotics [53, p 43, equations (2.8.17) and (2.8.18)],
|     | first |     |     |     | refined |     |     |     |     |
| --- | ----- | --- | --- | --- | ------- | --- | --- | --- | --- |
which was derived by Djrbashian [21], and by many researchers [80, 105].
Lemma 2.1. Let α ∈ (0, 2), β ∈ , and μ ∈ (απ 2, min(π, απ)). Then for N ∈ 
|        |               |     | N     |       | ⎛    | ⎞      |     |           |      |
| ------ | ------------- | --- | ----- | ----- | ---- | ------ | --- | --------- | ---- |
|        | 1 z(1−β)αez1α |     | 1     | 1     |      | 1      |     |           |      |
| E (z)= |               | −   | ∑     |       | + O⎜ | ⎟ with | z → | ∞, arg(z) | ⩽ μ, |
| α,β    | α             |     | Γ(β − | αk)zk | ⎝    | zN+1⎠  |     |           |      |
k=1
|        | N   | 1       | 1 ⎛     | 1 ⎞ |        |          |        |      |     |
| ------ | --- | ------- | ------- | --- | ------ | -------- | ------ | ---- | --- |
| (z)=−∑ |     |         | O⎜      | ⎟   |        |          |        |      |     |
| E α,β  |     |         | +       |     | with z | → ∞, μ ⩽ | arg(z) | ⩽ π. |     |
|        | Γ(β | − αk)zk | ⎝ zN+1⎠ |     |        |          |        |      |     |
k=1
Mittag-Leffler
From these asymptotics, the function E α,β (z) decays only linearly on the
negative real axis −, which is much slower than the exponential decay for the exponential
functionez.However,onthepositiverealaxis+,itgrowsexponentially,andthegrowthrate
increases with the fractional order 0 < α < 2. To illustrate the distinct feature, we plot the
functions E (−π2tα) and tα−1E (−π2tα) in figure 1 for several different α values, where
|     | α,1 |     | α,α |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
λ = π2isthefirstDirichleteigenvalueofthenegativeLaplacianontheunitintervalΩ = (0, 1)
5

| InverseProblems31(2015)035003 |        |          |             |                |                 |            | BJinandWRundell |
| ----------------------------- | ------ | -------- | ----------- | -------------- | --------------- | ---------- | --------------- |
|                               |        | The      | profiles of | Mittag-Leffler | functions       | (a) E      | (−π2tα) and (b) |
|                               | Figure | 1.       |             |                |                 | α,1        |                 |
|                               | tα−1E  | (−π2tα). | The value   | π2 is the      | first Dirichlet | eigenvalue | of the negative |
α,α
|     | Laplacian | onthe interval(0,1). |     |     |     |     |     |
| --- | --------- | -------------------- | --- | --- | --- | --- | --- |
;see appendix A.1 for further details on the computation of the Mittag-Leffler function.
Figure 1(a) can be viewed as the time evolution of u(1 2, t), where ∂αu − u = 0 with
|     |     |     |     |     |     | t   | xx  |
| --- | --- | --- | --- | --- | --- | --- | --- |
u(0, t) = u(1, t) = 0, and initial data u (x) = sinπx (the lowest Fourier eigenmode). The
0
slow decay behavior at large time is clearly observed. In particular, at t = 1, the function
(−π2t)
E α,1 still takes values distinctly away from zero for any 0 < α < 1, whereas the
exponentialfunctione−π2t
almostvanishesidentically.Incontrast,fortclosetozero,thepicture
is reversed: the Mittag-Leffler function E (−π2t) decays much faster than the exponential
α,1
functione−π2t. The drastically different behavior ofthe function E (−z), in comparisonwith
α,1
the exponential function e−z, explains many unusual phenomena with inverse problems for
FDEs to be described below. According to the exponential asymptotics, the function E (z)
α,α
decaysfasteronthenegativerealaxis−,since1 Γ(0) = 0,i.e.,thefirsttermintheexpansion
vanishes.Thisisconfirmednumericallyinfigure1(b).Eventhoughnotshown,itisnotedthat
the function E (z) decays only quadratically on the negative real axis − for α ∈ (0, 1) or
α,α
| α ∈ (1, 2), | which is | asymptotically | much | slower than | the exponential | decay. |     |
| ----------- | -------- | -------------- | ---- | ----------- | --------------- | ------ | --- |
The distribution of zeros of the Mittag-Leffer function E (z) is of immense interest,
α,β
especially in the related Sturm–Liouville problem; see section 4.1 below. The case of β = 1
wasfirststudiedbyWiman[104].ItwasrevisitedbyDjrbashian[21],andmanydeepresults
were derived, especially for the case ofα = 2. There are many further refinements [93]; see
| [83] for an | updated account. |        |            |            |     |     |     |
| ----------- | ---------------- | ------ | ---------- | ---------- | --- | --- | --- |
| 2.2. Wright | function         |        |            |            |     |     |     |
| The Wright  | functionW        | (z) is | defined by | [106, 107] |     |     |     |
ρ,μ
∞
zk
|     | W (z) | = ∑ |     | , μ, | ρ ∈ , ρ | > −1, z ∈ | .  |
| --- | ----- | --- | --- | ---- | -------- | --------- | --- |
ρ,μ
|     |     | k!Γ(ρk | + μ) |     |     |     |     |
| --- | --- | ------ | ---- | --- | --- | --- | --- |
k=0
first
This is an entire function of order1 (1 + ρ) [30, theorem 2.4.1]. It was introduced in
connection with a problem in number theory by Edward M Wright, and revived in recent
6

| InverseProblems31(2015)035003 |     |     |     |     |     |     |     | BJinandWRundell |     |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- |
yearssinceitappearsasthefundamentalsolutionforFDEs[69].TheWrightfunctionW (z)
ρ,μ
hasthefollowingtheasymptoticexpansioninonesectorcontainingthenegativerealaxis−
[67, theorem 3.2]. Like before, the exponential asymptotics can be used to deduce the
| distribution | of its zeros | [66]. |     |     |     |     |     |     |     |
| ------------ | ------------ | ----- | --- | --- | --- | --- | --- | --- | --- |
Lemma 2.2. Let −1 < ρ < 0, y = −z, arg(z) ⩽ π, −π < arg(y) ⩽ π, and for all small
| ϵ > 0,|arg(y)| | ⩽ min(3π(1 | +   | ρ) 2, π) | − ϵ. | Then |     |     |     |     |
| -------------- | ---------- | --- | -------- | ---- | ---- | --- | --- | --- | --- |
|                |            |     | ⎧        |      |      |     | ⎫   |     |     |
M−1
|     |       | Y12−μe−Y⎪⎨ | ∑A  |     |     | ( )⎪⎬, |     |      |     |
| --- | ----- | ---------- | --- | --- | --- | ------ | --- | ---- | --- |
|     | W (z) | =          |     | Y−m | + O | Y−M    | Y   | → ∞, |     |
|     | ρ,μ   |            | ⎪⎩  | m   |     |        | ⎪⎭  |      |     |
m=0
whereY = (1 + ρ)((−ρ)−ρy)1(1+ρ) and the coefficients A , m = 0, 1,… are defined by the
m
| asymptotic | expansion   |     |                 |     |      |      |          |               |     |
| ---------- | ----------- | --- | --------------- | --- | ---- | ---- | -------- | ------------- | --- |
|            |             |     |                 |     |      | M −1 | (−1)mA   |               |     |
|            |             | Γ(1 | − μ − ρt)       |     |      |      |          | m             |     |
|            |             |     |                 |     | =    | ∑    |          |               |     |
|            | 2π(−ρ)−ρt(1 |     | ρ)(1+ρ)(t+1)Γ(t |     |      | (    |          | 1             | )   |
|            |             | +   |                 |     | + 1) | Γ    | (1 + ρ)t | + μ + +       | m   |
|            |             |     |                 |     |      | m=0  |          | 2             |     |
|            |             |     |                 |     |      | ⎛    |          |               | ⎞   |
|            |             |     |                 |     |      | ⎜    |          | 1             | ⎟   |
|            |             |     |                 |     |      | +O   |          |               | ⎟,  |
|            |             |     |                 |     |      | ⎜    | (        |               | )   |
|            |             |     |                 |     |      | Γ    | (1 +     | ρ)t + β + 1 + | M   |
|            |             |     |                 |     |      | ⎝    |          |               | ⎠   |
2
π
valid forarg(t), arg(−ρt), andarg(1 − μ − ρt) all lyingbetween−π and andttending to
infinity.
TheWrightfunctionW (z),−1 < ρ < 0,decaysexponentiallyonthenegativerealaxis
ρ,μ
−,inamannersimilartotheexponentialfunctionez,butatadifferentdecayrate.Itsspecial
role infractional calculus isunderscored bythefactthat it forms thefree-space fundamental
solution K (x, t) to the one-dimensional time fractional diffusion equation [69] by
α
1
|     |         |      |           | (   | tα2 ) |     |     |     |       |
| --- | ------- | ---- | --------- | --- | ----- | --- | --- | --- | ----- |
|     | K α (x, | t) = | W −α ,1−α | −x  | .     |     |     |     | (2.2) |
|     |         | 2tα  | 2 2       |     |       |     |     |     |       |
2
The multidimensional case is more complex and involves further special functions, in
particular,theFoxHfunction[54,91].Forα = 1,theformula(2.2)recoversthefamiliarfree-
| space fundamental | solution | for | the one-dimensional |     | heat | equation, | i.e. |     |     |
| ----------------- | -------- | --- | ------------------- | --- | ---- | --------- | ---- | --- | --- |
1 2
|     | K(x, | t) = | e−x t, |     |     |     |     |     |     |
| --- | ---- | ---- | ------ | --- | --- | --- | --- | --- | --- |
4
2 πt
which is a Gaussian distribution in x for any t > 0. In the fractional case, the fundamental
solutionK (x, t)exhibitsquitedifferent behaviorthantheheatkernel.Toseethis,weshow
α
theprofileofK t)infigure2forseveralαvalues;seeappendixA.1forabriefdescription
α (x,
ofthecomputationaldetails.Forany0 < α < 1,K (x, t)decayssloweratapolynomialrate
α
as the argument |x| tα2 tends to infinity, i.e., having a long tail, when compared with the
Gaussiandensity.Thelongtailprofileisoneofdistinctfeaturesofslowdiffusion[5].Further,
for anyα < 1, the profile is only continuous but not differentiable at x = 0. The kink at the
origin implies that the solution operator to time fractional diffusion may only have a limited
| smoothing     | property.          |     |     |     |     |     |     |     |     |
| ------------- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.3. Singular | valuedecomposition |     |     |     |     |     |     |     |     |
We shall follow the well-established practice in the inverse problem community, i.e., using
the singular value decomposition, as the main tool for numerically analyzing the problem
7

InverseProblems31(2015)035003 BJinandWRundell
Figure2.TheprofileofthefundamentalsolutionK
α
(x,t) at(a)t=0.1and(b)t=1.
behavior[32].Specifically,weshallnumericallycomputetheforwardmapF,andanalyzeits
behavior to gain insights into the inverse problem. Given a matrix A ∈ n×m, its singular
value decomposition is given by
A = UΣVt,
where U = [u … u ] ∈ n×n and V = [v … v ] ∈ m×m are column orthonormal
1 n 1 m
matrices, and Σ ∈ n×m = diag(σ,…, σ, 0,…, 0) is a diagonal matrix, with the diagonal
1 r
elements{σ} being nonnegative and listed in a descending order σ > … > σ > 0, and r
i 1 r
being the (numerical) rank of the matrix A. The diagonal element σ is known as the ith
i
singular value, and the corresponding columns ofU andV, i.e.,u andv, are called the left
i i
and right singular vectors, respectively.
One simple measure of the conditioning of a linear inverse problem Ax = b is the
conditionnumbercond(A),whichisdefinedastheratioofthelargesttothesmallestnonzero
singular value, i.e.
cond(A) = σ σ.
1 r
Inparticular,iftheconditionnumberissmall,thenthedataerrorwillnotbeamplifiedmuch.
Inthecaseofalargeconditionnumber,theissueismoredelicate:itmayormaynotamplify
the data perturbation greatly. A more complete picture is provided by the singular value
spectrum (σ, σ ,…, σ ). Especially, a singular value spectrum gradually decaying to zero
1 2 r
withoutacleargapischaracteristicofmanydiscreteill-posedproblems,whichisreminiscent
of the spectral behavior of compact operators. We shall adopt these simple tools to analyze
related inverse problems below.
In addition, using singular value decomposition and regularization techniques, e.g.
Tikhonov regularization or truncated singular value decomposition, one can conveniently
obtainnumericalreconstructions,eventhoughthismightnotbethemostefficientwaytodo
so. However, we shall not delve into the extremely important question of practical recon-
structions, since it relies heavily on a priori knowledge on the sought for solution and the
8

| InverseProblems31(2015)035003 |     |     |     |     |     |     | BJinandWRundell |     |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --------------- | --- |
statistical nature (Gaussian, Poisson, Laplace …) of the contaminating noise in the data,
specific
which will depend very much on the application. We refer interested readers to the
monographs[26,41,92]andthesurvey[49]forupdatedaccountsonregularizationmethods
forconstructing stablereconstructing proceduresandefficientcomputational techniques.We
brieflymentionbelowrelatedworksontheapplicationofregularization
| willalso   |          |           |                 |     |           |     | techniques |     |
| ---------- | -------- | --------- | --------------- | --- | --------- | --- | ---------- | --- |
| to inverse | problems | for FDEs. |                 |     |           |     |            |     |
| 3. Inverse | problems | for       | time fractional |     | diffusion |     |            |     |
In this section, we consider several model inverse problems for the following one-dimen-
| sional time | fractional | diffusion | equation | on  | the unit interval | Ω = (0, 1): |     |     |
| ----------- | ---------- | --------- | -------- | --- | ----------------- | ----------- | --- | --- |
∂αu
|     |     | − u xx | + qu = f | inΩ × | (0, T], |     |     | (3.1) |
| --- | --- | ------ | -------- | ----- | ------- | --- | --- | ----- |
t
with the fractional order α ∈ (0, 1), the initial condition u(0) = v and suitable boundary
conditions. Although we consider only the one-dimensional model, the analysis and
computation in this part can be extended into the general multi-dimensional case, upon
suitablemodifications.Recallthat∂αudenotestheDjrbashian–Caputofractionalderivativeof
t
orderαwithrespecttotimet.Forα = 1,thefractionalderivative∂αucoincideswiththeusual
t
first-order
derivative u′, and accordingly, the model (3.1) reduces to the classical diffusion
equation.Henceitisnaturaltocompareinverseproblemsforthemodel(3.1)withthatforthe
standard diffusion equation. We shall discuss the following four inverse problems, i.e., the
backwardproblem,sidewaysproblem,inversesourceproblemandinversepotentialproblem.
In the first three cases, we shall assume a zero potential q = 0. We will also discuss the
coefficient
| solution of   | an inverse |           | problem | using | fractional | calculus. |     |     |
| ------------- | ---------- | --------- | ------- | ----- | ---------- | --------- | --- | --- |
| 3.1. Backward | fractional | diffusion |         |       |            |           |     |     |
First we consider the time fractional backward diffusion. By the linearity of the inverse
problem, we may assume that equation (3.1) is prescribed with a homogeneous Dirichlet
boundarycondition,i.e.,u=0atx=0,1,andtheinitialconditionu(0) = v.Thentheinverse
problemreads:giventhefinaltimedatag u(T),findtheinitialconditionv.Itarisesin,for
=
example, the determination of a stationary contaminant source in underground fluid flow.
Togaininsight,weapplytheseparationofvariables.Let{(λ , ϕ )},withλ = (jπ)2 and
|     |     |     |     |     |     | j j | j   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
ϕ = 2 sinjπx,betheDirichleteigenpairsofthenegativeLaplacianontheintervalΩ.The
j
eigenfunctions{ϕ }form an orthonormal basis of the L2(Ω) space. Then using the Mittag-
j
| Lefflerfunction |     | definedin(2.1),thesolution |     |     |             |            |           |     |
| --------------- | --- | -------------------------- | --- | --- | ----------- | ---------- | --------- | --- |
|                 | E   | (z)                        |     |     | utoequation | (3.1)canbe | expressed | as  |
α,β
|     |      |       | ∞    | tα)( | )        |     |     |     |
| --- | ---- | ----- | ---- | ---- | -------- | --- | --- | --- |
|     | u(x, | t) ∑E | ( −λ | v,   | ϕ ϕ (x). |     |     |     |
|     |      | =     | α,1  | j    |          |     |     |     |
j j
j=1
| Therefore, | the final | time | data g = u(T) | is given | by     |     |     |     |
| ---------- | --------- | ---- | ------------- | -------- | ------ | --- | --- | --- |
|            |           | ∞    | Tα)(          |          | )      |     |     |     |
|            |           | ∑E   | (             |          |        |     |     |     |
|            | g(x)      | =    | α,1 −λ j      | v, ϕ     | ϕ (x). |     |     |     |
j j
j=1
9

InverseProblems31(2015)035003 BJinandWRundell
| It follows | directly that the | initial data | v is formally | given by |     |
| ---------- | ----------------- | ------------ | ------------- | -------- | --- |
( )
|     | ∞   | g, ϕ |     |     |     |
| --- | --- | ---- | --- | --- | --- |
j
|     | v = ∑ | ϕ   | .   |     |     |
| --- | ----- | --- | --- | --- | --- |
( Tα) j
|     | E       | −λ  |     |     |     |
| --- | ------- | --- | --- | --- | --- |
|     | j=1 α,1 | j   |     |     |     |
SincethefunctionE (−t)iscompletelymonotoneonthepositiverealaxis+ [82]forany
α,1
α ∈ (0, 1], the denominator in the representation does not vanish. In case of α = 1, the
| formula reduces | to the familiar | expression |     |     |     |
| --------------- | --------------- | ---------- | --- | --- | --- |
|                 | ∞ (             | )          |     |     |     |
∑eλjT
|     | v = | g, ϕ ϕ | .   |     |     |
| --- | --- | ------ | --- | --- | --- |
j j
j=1
This formula shows clearly the well-known, severely ill-posed nature of the backward
diffusion problem: the perturbation in the jth Fourier mode (g, ϕ ) of the (noisy) data g is
j
| amplified |     |     | eλjT, |     |     |
| --------- | --- | --- | ----- | --- | --- |
by an exponentially growing factor which can be astronomically large, even
for a verysmall index j, if the terminal time T isnot very small. Hence it isalways severely
ill-conditioned and we must multiply the jth Fourier mode of the data g by a factor eλjT in
| order to recover | the corresponding | mode | of the | initial data v. |     |
| ---------------- | ----------------- | ---- | ------ | --------------- | --- |
In the fractional case, by lemma 2.1, the Mittag-Leffler function E (z) decays only
α,1
linearly on the negative real axis −, and thus the multiplier E (−λ Tα) grows only
1 α,1 j
linearly in λ , i.e., 1 E (−λ Tα) ∼ λ , which is very mild compared to the exponential
|     | j α,1 | j   | j   |     |     |
| --- | ----- | --- | --- | --- | --- |
growtheλjT forthecaseα = 1,andthusthefractionalcaseisonlymildlyill-posed.Roughly,
thejthFouriermodeoftheinitialdatavnowequalsthejthmodeofthedatagmultipliedby
λ j . More precisely, it amounts to the loss of two spatial derivatives [88, theorem 4.1]
|     | ∥v∥ ⩽ c∥u(T)∥ |       | .   |     | (3.2) |
| --- | ------------- | ----- | --- | --- | ----- |
|     | L2(Ω)         | H2(Ω) |     |     |       |
Intuitively, the history mechanism of the anomalous diffusion process retains the complete
dynamics of the physical process, including the initial data, and thus it is much easier to go
backwardstotheinitialstatev.Thisisinsharpcontrasttoclassicaldiffusion,whichhasonly
ashortmemoryandlosestrackofthepreceding statesquickly.Thisresulthasbecomequite
well-known in the inverse problems community and has contributed to a belief that ‘inverse
problemsforFDEsarelessill-conditionedthantheirclassicalcounterparts’—throughoutthis
paper we will see that this conclusion as a general statement can be quite far from the truth.
DoesthismeanthatforallterminaltimeTthefractionalcaseisalwayslessill-posedthan
theclassicalone?Theanswerisyes,inthesenseofthenormonthedataspaceinwhichthe
dataglies.Doesthismeanthatfromacomputationalstabilitystandpointthatonecanalways
solve the backward fractional problem more effectively than for the classical case? The
answerisno,andthedifferencecanbesubstantial.Toillustratethepoint,letJbethehighest
frequency mode required of the initial data v and assume that we believe we are able to
first
multiply the J modes g = (g, ϕ ), j = 1, 2,…, J, by a factor no larger than M (which
j j
roughly assumes that thenoise levels inboth cases are comparable). By the monotonicity of
the function E (−t) in t, it suffices to examine the Jth mode. For the heat equation
α,1
v :=(v, ϕ ) = eλJTg and provided thatT = T < λ logM this is feasible. For a fixed J, if
| J   | J J |     | J   | J   |     |
| --- | --- | --- | --- | --- | --- |
T⋆
| denotes | the point where |     |     |     |     |
| ------- | --------------- | --- | --- | --- | --- |
α
|     | e−λJT ⋆   | ( ⋆)   |     |     |     |
| --- | --------- | ------ | --- | --- | --- |
|     | α = E α,1 | −λ J T | ,   |     |     |
α
|     |     | T⋆  |     |     | T⋆. |
| --- | --- | --- | --- | --- | --- |
theninthefractionalcaseforT < α thegrowthfactorong J willexceedMforanyT < α
Intable1,wepresentthecriticalvalueT⋆ forseveralvaluesofthefractionalorderαandthe
α
10

InverseProblems31(2015)035003 BJinandWRundell
Table1.Thecritical valuesT* for fractional backward diffusion.
α
α⧹J 3 5 10
1/4 0.0442 0.0197 0.0059
1/2 0.0387 0.0163 0.0049
3/4 0.0351 0.0142 0.0040
maximumnumberofmodesJ.Thenumbersinthetableareverytelling.Forexample,forthe
case J = 5, α = 1 4 and T = 0.02 (which is approximately one half the value of T⋆), the
α
growth factor is about 1.6 for the heat equation but about 113 for the fractional case. With
J = 10 andα = 1 4 andT = T⋆ the growth factor is around 336. IfT = T⋆ 10 then it has
α α
again dropped to less than 2 for the heat equation but about 190 for the fractional case. Of
course, forT > T⋆ the situation completely reverses. With J = 10, α = 1 4 andT = 10T⋆
α α
thegrowthfactorisapossiblyworkablevalueofaround600;whilefortheheatequationitis
greater than 1025. We reiterate that the apparent contradiction between the theoretical ill-
conditioning and numerical stability is due to the spectral cutoff present in any practical
reconstruction procedure.
Next we examine the influence of the fractional order α on the inversion step more
closely. To this end, we expand the initial condition v in the piecewise linear finite element
basisfunctionsdefinedonauniformpartitionofthedomainΩ = (0, 1)with100gridpoints.
ThenwecomputethediscreteforwardmapFfromtheinitialconditiontothefinaltimedata
g = u(T), defined on the same mesh. Numerically, this can be achieved by a fully discrete
schemebasedontheL1approximationintimeandthefinitedifferencemethodinspace;see
appendixA.2foradescriptionofthenumericalmethod.Theill-posedbehaviorofthediscrete
inverseproblemisthenanalyzedusingsingularvaluedecomposition.Asimilarexperimental
setup will be adopted for other examples below.
The numerical results are shown in figure 3. The condition number of the (discrete)
forwardmapFstaysmostlyaroundO(104)forafairlybroadrangeofαvalues,whichholds
for all three different terminal times T. This can be attributed to the fact that for any
α ∈ (0, 1), backward fractional diffusion amounts to a two spacial derivative loss, cf (3.2).
Unsurprisingly, as the fractional order α approaches unity, the condition number eventually
blows up, recovering the severely ill-posed nature of the classical backward diffusion pro-
blem,cffigure3(a).Further,weobservethatthesmalleristheterminaltimeT,thequickeris
the blowup. The precise mechanism for this observation remains unclear. Interestingly, the
condition number is not monotone with respect to the fractional order α, for a fixed T. This
might imply potential nonuniqueness in the simultaneous recovery of the fractional order α
and the initial data v. The singular value spectra at T = 0.01 are shown in figure 3(b). Even
though the condition numbers for α = 1 4 and α = 1 2 are quite close, their singular value
spectra actually differ by a multiplicative constant, buttheir decayrates arealmost identical,
thereby showing comparable condition numbers. This shift in singular value spectra can be
explained by the local decay behavior of the Mittag-Leffler function, cf figure 1(a): the
smaller is the fractional order α, the faster is the decay around t = 0.
Eventhoughtheconditionnumberisveryinformativeaboutthe(discretized)problem,it
doesnotprovideafullpicture,especiallywhentheconditionnumberislarge.Inthiscasethe
singularvaluespectrumcanbefarmorerevealing.Thespectrafortwodifferentαvaluesare
given in figure 4. At α = 1 2, the singular values decay at almost the same algebraic rate,
irrespective of the terminal time T. This is expected from the two-derivative loss for any
α < 1. However, for α = 1, the singular values decay exponentially, and the decay rate
11

InverseProblems31(2015)035003 BJinandWRundell
Figure3.(a)Theconditionnumberversusthefractionalorderα,and(b)thesingular
valuespectrumatT=0.01forthebackwardfractionaldiffusion.Weonlydisplaythe
first50singular values.
Figure4.ThesingularvaluespectrumoftheforwardmapFfromtheinitialdatatothe
final time data, for (a) α=12 and (b) α=1, at four different times for the time
fractional backward diffusion.
increases dramatically with the increase of the terminal time T. For T = 0.001, there are a
handful of ‘significant’ singular values, say above 10 −3, but when the time T increases to
T=1,thereisonlyonemeaningfulsingularvalueremaining.Thedistributionofthesingular
valueshasimportantpracticalconsequences.ForasmalltimeT,thefirstfewsingularvalues
fortheclassicaldiffusioncaseactuallymightbemuchlargerthanthatforthefractionalcase,
which indicates that the classical case is actually numerically much easier to recover in this
regime,concurringwiththeobservationsdrawnfromtable1.Forexample,atT=0.001,the
firsttwentysingularvaluesarelargerthanthefractionalcounterpart,cffigure3(b),andhence,
12

InverseProblems31(2015)035003 BJinandWRundell
the first twenty modes, i.e., left singular vectors, are more stable in the reconstruction
procedure.
The mathematical model (3.1) is rescaled with a unit diffusion coefficient. In practice,
| there is | always a | diffusion coefficient | σ in the elliptic | operator, i.e. |
| -------- | -------- | --------------------- | ----------------- | -------------- |
|          | ∂αu      | −· (σu)             | + qu = f.         |                |
t
σ
For example, the thermal conductivity of the gun steel at moderate temperature is about
1.8 × 10−5m2s−1 [11] and the diffusion coefficient of the oxygen in water at 25°C is
2.10 × 10−5cm2s−1 [18]. Mathematically, this does not change the ill-posed nature of the
coefficient σ
inverse problem. However, the presence of a diffusion has important
consequence: it enables the practical feasibility of the classical backward diffusion problem
(andlikelyformanyotherinverseproblemsforthediffusionequation).Physically,aconstant
conductivity σ amounts to rescaling the final time T by T′ = T σ. In the fractional case, a
| similar but | nonlinear | scaling lawT′ | = T σ1α remains | valid. |
| ----------- | --------- | ------------- | --------------- | ------ |
Numerically, time fractional backward diffusion has been extensively studied. Liu and
Yamamoto [65] proposed a numerical scheme for the one-dimensional fractional backward
problem based on the quasi-reversibility method [57], and derived error estimates for the
approximation,underapriorismoothnessassumptionontheinitialcondition.Thisrepresents
oneofthefirstworksoninverseproblemsinanomalousdiffusion.Later,WangandLiu[99]
studied total variation regularization for two-dimensional fractional backward diffusion, and
analyzed its well-posedness of the optimization problem and the convergence of an iterative
scheme of Bregman type. Wei and Wang [102] developed a modified quasi-boundary value
methodfortheprobleminageneraldomain,andestablishederrorestimatesforbothapriori
and a posteriori parameter choice rules. In view of better stability results in the fractional
case, one naturallyexpects better error estimates than theclassical diffusion equation, which
| is confirmed            | by these | studies.  |     |     |
| ----------------------- | -------- | --------- | --- | --- |
| 3.2. Sidewaysfractional |          | diffusion |     |     |
Next we consider the sideways problem for time fractional diffusion. There are several
possible formulations, e.g., the quarter plane and the finite space domain. The quarter plane
sidewaysfractionaldiffusionproblemisasfollows.Letu(x, t)bedefinedin(0, ∞) × (0, ∞)
by
|         | ∂αu      | − u = 0,               | x > 0, t > 0,   |     |
| ------- | -------- | ---------------------- | --------------- | --- |
|         | t        | xx                     |                 |     |
| and the | boundary | and initial conditions |                 |     |
|         | u(x,     | 0) = 0 and             | u(0, t) = f(t), |     |
ec2x2.
where we assume|u(x, t)| ⩽ c We do not know the left boundary condition f, but are
1
abletomeasureuatanintermediatepointx = L > 0,h(t) = u(L, t).Theinverseproblemis:
giventhe(noisy)datah,findtheboundaryconditionf.Thesolutionuoftheforwardproblem
isgivenbyaconvolutionintegralwiththekernelbeingthespatialderivativeK (x, s)ofthe
α,x
| fundamental | solution | K (x, s), cf | (2.2), by |     |
| ----------- | -------- | ------------ | --------- | --- |
α
t
∫
|     | u(x, | t) = K | (x, t − s)f(s)ds. |     |
| --- | ---- | ------ | ----------------- | --- |
α,x
0
Thisrepresentationiswellknownforthecaseα = 1,anditwasfirstderivedbyCarasso[11];
seealso[8]forrelateddiscussions.Itleadstoaconvolutionintegralequationfortheunknown
| f in terms | of the | given data h |     |     |
| ---------- | ------ | ------------ | --- | --- |
13

| InverseProblems31(2015)035003 |     |     |     |     |     |     |     |     | BJinandWRundell |     |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- |
t
∫
|     | h(t) | =   | R α (t | − s)f(s)ds, |     |     |     |     |     |     |
| --- | ---- | --- | ------ | ----------- | --- | --- | --- | --- | --- | --- |
0
| where the | convolution | kernel | R   | (s) is | given by | a Wright | function | in the | form |     |
| --------- | ----------- | ------ | --- | ------ | -------- | -------- | -------- | ------ | ---- | --- |
α
|            |        |                 |            |        |     | ∞           |          | )k         |          |     |
| ---------- | ------ | --------------- | ---------- | ------ | --- | ----------- | -------- | ---------- | -------- | --- |
|            |        |                 | 1          | (      | )   |             | ( −      | L          |          |     |
|            | R      | (s)=            | W          | −Ls−α2 |     | = ∑         |          |            | s−kα −α. |     |
|            | α      |                 | sα −α      | ,2−α   |     |             | ( α      |            | α) 2     |     |
|            |        | 2               | 2          | 2      |     |             | k!Γ −    | k + 2 −    |          |     |
|            |        |                 |            |        |     | k=0         | 2        |            | 2        |     |
| In case of | α = 1, | i.e., classical | diffusion, |        | the | kernel R(s) | is given | explicitly | by       |     |
L
|     | R(s) | =   | s−3 2e−L | 2 ∈ | C∞(0, | ∞). |     |     |     |     |
| --- | ---- | --- | -------- | --- | ----- | --- | --- | --- | --- | --- |
4 s
2 π
Since all its derivatives vanish at s = 0, the classical theory of Volterra integral equations of
thefirstkind[56]implies theextremeill-conditioningoftheproblem.Thisisnotsurprising:
we are, after all, mapping a function f ∈ C0(0, ∞) to an element in C∞(0, ∞). The
conditioningofthetimefractionalsidewaysproblemagaindependsontheconvolutionkernel
R α and its derivatives at s = 0 and in this case is the value of the Wright function
W (−z) and its derivatives as z → ∞. These are again zero (in fact the Wright
−α2,2−α2
function W (z) also decays exponentially to zero for large negative arguments, cf
−α2,2−α2
lemma2.2),andthusthefractionalsidewaysproblemisalsoseverelyill-posed.However,this
analysis does not show their difference in the degree of ill-posedness: even though both are
severely ill-posed, their practical computational behavior can still be quite different, as we
shall see below.
Toseetheirdifferenceinthedegreeofill-posedness, weexamineanothervariantofthe
finite
sideways problem on a interval Ω = (0, 1), with Cauchy data prescribed on the axis
x = 0, i.e. given zero initial condition u = u(x, 0) = 0, recovering h = u(1, t) from the
0
| lateral Cauchy | data | at x | = 0:  |          |         |     |     |     |     |     |
| -------------- | ---- | ---- | ----- | -------- | ------- | --- | --- | --- | --- | --- |
|                | u(0, | t) = | f(t), | u (0, t) | = g(t), | t ⩾ | 0.  |     |     |     |
x
ThisproblemisalsoknownasthelateralCauchyproblemintheliterature.Inthecaseα = 1,
it is known that the inverse problem is severely ill-posed [8, 33]. To gain insight into the
fractionalcase,weapplytheLaplacetransform.With beingtheLaplacetransformintime,
|     |     |     |     |     |     |     |     | αu | zαu(z) zα−1u |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- |
and noting the Laplace transform of the Caputo derivative ∂ t = − 0 [53,
| lemma 2.24], | we      | deduce |          |      |       |     |         |     |     |     |
| ------------ | ------- | ------ | -------- | ---- | ----- | --- | ------- | --- | --- | --- |
|              | zαu(x, |        | u       |      | u(0) | =f | u      | g. |     |     |
|              |         | z)     | − xx (x, | z) = | 0,    |     | , x (0) | =   |     |     |
The general solutionu is given byu(x, z) =f coshzα2x + gsinhzα2x and thus the solution
zα2x
h(z)
| =   | u(1, z) at | x = 1 | is given | by  |     |     |     |     |     |     |
| --- | ----------- | ----- | -------- | --- | --- | --- | --- | --- | --- | --- |
sinhzα2
|     | h =f | coshzα2 |     | + g | .   |     |     |     |     |     |
| --- | ------ | ------- | --- | ---- | --- | --- | --- | --- | --- | --- |
zα2
| The solution | h(t) | can then | be recovered |     | by an | inverse | Laplace | transform |     |     |
| ------------ | ---- | -------- | ------------ | --- | ----- | ------- | ------- | --------- | --- | --- |
1
|     | h(t) | =   | ∫ heztdz, |     |     |     |     |     |     | (3.3) |
| --- | ---- | --- | ---------- | --- | --- | --- | --- | --- | --- | ----- |
|     |      | 2πi | Br         |     |     |     |     |     |     |       |
where Br = {z ∈ : Rz = σ, σ > 0} is the Bromwich path. Upon deforming the contour
efficient
suitably, this formula will allow the development of an numerical scheme for the
sideways problem via quadrature rules [103], provided that the lateral Cauchy data is
availableforallt > 0.Theexpression(3.3)indicatesthat,inthefractionalcase,thesideways
problem still suffers from severe ill-posedness in theory, since the high frequency modes of
14

InverseProblems31(2015)035003 BJinandWRundell
Figure5.(a) The condition number and (b) singular value spectrum at T = 1 for the
time fractional sideways problem.
the data perturbation are amplified by an exponentially growing multiplier ezα2. However,
numerically, the degree of ill-posedness decreases dramatically as the fractional order α
decreases from unity to zero, since as α → 0+, the multipliers are growing at a much slower
rate,andthuswehaveabetterchanceofrecoveringmanymoremodesoftheboundarydata.
In other words, both the classical and fractional sideways problems are severely ill-posed in
the sense of error estimates between the norms in the data and unknowns; but with a fixed
frequency range, the behavior of the time fractional sideways problem can be much less ill-
posed. Hence, anomalous diffusion mechanism does help substantially since much more
effective reconstructions are possible in the fractional case.
Nextweillustratethepointnumerically.Thenumericalresultsforthesidewaysproblem
are given in figure 5. It is observed that the degree of ill-posedness of the finite-dimensional
discretizedversionoftheinverseproblemindeeddecreasesdramaticallywiththedecreaseof
the fractional order α, cf figure 5(a), which agrees well with the preceding discussions.
Surprisingly,forT=1thereisasuddentransitionaroundα = 1 2,belowwhichthesideways
problem behaves as if nearly well-posed, but above which the conditioning deteriorates
dramatically with the increase of the fractional order α and eventually it recovers the prop-
erties of the classical sideways problem. Similar transitions are observed for other terminal
times. This might be related to the discrete setting, for which there is an inherent frequency
cutoff.Further,asthefractionalorderαapproacheszero,theproblemreachesaquasi-steady
statemuchquickerandthustheforwardmapFcanhaveonlyfairlylocalizedelementsalong
themaindiagonal.Togiveamorecompletepicture,weexaminethesingularvaluespectrum
in figure 5(b). Unlike the backward diffusion problem discussed earlier, the singular values
are actually decaying only algebraically, even forα = 1, and then there might be a few tiny
singularvaluescontributingtothelargeconditionnumber.Thelargeristhefractionalorderα,
the more tiny singular values are in the spectrum. Hence, in the discrete setting, even for
α = 3 4,theproblem isstillnearlywell-posed, despite thelargeapparent condition number,
sinceafewtinysingularvalueswithadistinctgapfromtherestofthespectrumareharmless
in most regularization techniques.
Physically this can also be observed in figure 6, where the forward map F is from the
Dirichlet boundary condition x = 1 to the flux boundary condition at x = 0, in a piecewise
15

InverseProblems31(2015)035003 BJinandWRundell
Figure6.TheJacobianmapFforα=14,12,34and1,fromtheinterval(0,1)itself.
linear finite element basis. Pictorially, the forward map F is only located in the upper left
corner and has a triangular structure, which reflects the casual or Volterra nature of the
sideways problem for the fractional diffusion equation. We note that the causal structure
shouldbeutilizedindevelopingreconstructiontechniques,via,e.g.,Lavrentievregularization
[56].Forsmallαvalues,e.g.,α = 1 4,thefiniteelementbasisattherightendpointx=1is
almostinstantlytransportedtotheleftendpointx=0,whosemagnitudeisslightlydecreased,
butwithlittlediffusiveeffect,resultingadiagonallydominantforwardmap.However,asthe
fractional order α increases towards unity, the diffusive effect eventually kicks in, and the
informationspreadsoverthewholeinterval.Further,forlargeαvalues,ittakesmuchlonger
time to reach the other side and there is a lag of information arrival, which explains the
presence of tiny singular values. The larger is the fractional order α, the smaller is the
magnitude, i.e., the less is the amount of the information reached the other side. Hence, one
feasibleapproachistorecoveronlytheboundaryconditionoverasmaller subintervalofthe
measurement time interval. This idea underlies one popular engineering approach, the
sequential function specification method [4, 64].
Thesidewaysproblemfortheclassicaldiffusionhasbeenextensivelystudied,andmany
efficient numerical methods have been developed and analyzed [8, 11, 24, 25]. In the frac-
tional case, however, there are only a few works on numerical schemes, mostly for one-
dimensional problems, and there seems no theoretical study on stability etc. Murio [76, 77]
developed several numerical schemes, e.g., based on space marching and finite difference
method,forthesidewaysproblem,butwithoutanyanalysis.Qian[85]discussedabouttheill-
16

| InverseProblems31(2015)035003 |     |     |     |     |     |     | BJinandWRundell |     |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --------------- | --- |
posedness of the quarter plane formulation of the sideways problem using the Fourier ana-
lysis,basedonwhichamollifiermethodwasproposed,witherrorestimatesprovided.In[87],
the recovery of a nonlinear boundary condition from the lateral Cauchy data was studied
using an integral equation approach, and a convergent fixed point iteration method was
|     |     | influence |     |     | specification |     |     | α   |
| --- | --- | --------- | --- | --- | ------------- | --- | --- | --- |
suggested. The of the imprecise of the fractional order on the
reconstruction was examined. Zheng and Wei [113] proposed a mollification method for the
quarter plane formulation of the sideways problem, by convoluting the fractional derivative
withasmoothkernel,andderivederrorestimatesfortheapproximation,underapriorbounds
on the solution. The Cauchy problem of the time fractional diffusion has been numerically
studied in [114]. In particular, with the separation of variables, a Volterra integral equation
reformulation of the problem was derived, from which the ill-posedness of the Cauchy
problem follows directly. All these works are concerned with the one-dimensional case, and
| the high     | dimensional |         | case has | not been | studied. |     |     |     |
| ------------ | ----------- | ------- | -------- | -------- | -------- | --- | --- | --- |
| 3.3. Inverse | source      | problem |          |          |          |     |     |     |
A third classical linear inverse problem for the diffusion equation is the inverse source
finaltime
problem, i.e., therecovery of thesource term f from lateral boundary data or data.
Clearly, one piece of boundary data or final time data alone is insufficient to uniquely
determine a general source term, due to dimensional disparity. To restore the possible
uniqueness, as usual, we look for only a space- or time-dependent component of the source
termf.Withdifferentcombinationsofthedataandsourceterm,wegetseveraldifferent(and
not equivalent) formulations of the inverse source problems. Below we examine several of
thembriefly.Bythelinearityoftheforwardproblem,wewithoutlossofgenerality,assumea
| zero initial | data | v = 0 | and a zero | potential | q = 0 | throughout | this part. |     |
| ------------ | ---- | ----- | ---------- | --------- | ----- | ---------- | ---------- | --- |
First,supposewecanmeasurethesolutionuatthefinaltimet=T,andaimatrecovering
either a space dependentortime dependentcomponent ofthesource termf.Likebefore, we
resorttotheseparationofvariables.Forthecaseofaspacedependentonlysourcetermf(x),
| the solution | u to | the | forward problem |     | is given by |     |     |     |
| ------------ | ---- | --- | --------------- | --- | ----------- | --- | --- | --- |
∞
|     |     |         | t   |          | (     | τ)α)( | )         |     |
| --- | --- | ------- | --- | -------- | ----- | ----- | --------- | --- |
|     |     | u(t)=∑∫ | (t  | − τ)α−1E | −λ    | (t −  | f, ϕ ϕ dτ |     |
|     |     |         |     |          | α,α j |       | j j       |     |
0
j=1
∞
|     |     | =∑  | 1 ( |     | ( tα))( |      | )   |     |
| --- | --- | --- | --- | --- | ------- | ---- | --- | --- |
|     |     |     | 1   | − E | −λ      | f, ϕ | ϕ . |     |
|     |     |     | λ   | α,1 | j       | j    | j   |     |
j
j=1
| Hence the | measured |     | data g = | u(T) is | given by |     |     |     |
| --------- | -------- | --- | -------- | ------- | -------- | --- | --- | --- |
∞
|     |     |       | 1 ( | (    | Tα))( | )   |     |     |
| --- | --- | ----- | --- | ---- | ----- | --- | --- | --- |
|     |     | g = ∑ | 1 − | E −λ | f,    | ϕ ϕ | .   |     |
|     |     |       |     | α,1  | j     | j   | j   |     |
λ j
j=1
Bytakinginnerproductwithϕ onbothsides,wearriveatthefollowingrepresentationofthe
j
| source term | f in | terms | of the measured |      | data g |     |     |     |
| ----------- | ---- | ----- | --------------- | ---- | ------ | --- | --- | --- |
|             |      |       |                 | ( )  |        |     |     |     |
|             |      | ∞     |                 | g, ϕ |        |     |     |     |
j
|     |     | f = ∑λ |       |     | ϕ .   |     |     | (3.4) |
| --- | --- | ------ | ----- | --- | ----- | --- | --- | ----- |
|     |     |        | j     | (   | Tα) j |     |     |       |
|     |     |        | 1 − E | −λ  |       |     |     |       |
|     |     | j=1    |       | α,1 | j     |     |     |       |
BythecompletemonotonicityoftheMittag-LefflerfunctionE (−t)onthepositiverealaxis
α,1
| + [82], | we deduce |     |     |     |     |     |     |     |
| -------- | --------- | --- | --- | --- | --- | --- | --- | --- |
17

| InverseProblems31(2015)035003 |     |     |     |     |     |     |     | BJinandWRundell |     |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- |
final
|     | Figure7.Numerical |     | results | for the | inverse source | problem | with | time data | and a |
| --- | ----------------- | --- | ------- | ------- | -------------- | ------- | ---- | --------- | ----- |
α,
|     | spacedependentsourceterm. |          |               | (a)Theconditionnumber |       |     | versusthe | fractionalorder |     |
| --- | ------------------------- | -------- | ------------- | --------------------- | ----- | --- | --------- | --------------- | --- |
|     | and(b)                    | singular | valuespectrum | at                    | T= 1. |     |           |                 |     |
|     | 1 > E                     | (−λ Tα)  | > E           | (−λ Tα),              |       |     |           |                 |     |
|     |                           | α,1 1    |               | α,1 2                 |       |     |           |                 |     |
and thus the formula (3.4) is well defined for anyT > 0, and gives the precise condition for
the existence of a source term. Even with a modest value of the terminal time T, the factor
α
1 − E (−λ Tα) is close to unity for all small values, especially for those close to zero.
| α,1 | 1   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Each frequency component (f, ϕ ) differs from (g, ϕ ) essentially by a factor λ , which
|            |                |        | j         |          | j       |      |     | j   |     |
| ---------- | -------------- | ------ | --------- | -------- | ------- | ---- | --- | --- | --- |
| amounts to | two derivative | loss   | in space. | Actually | one can | show |     |     |     |
|            | ∥f∥            | ⩽ c∥g∥ |           | .        |         |      |     |     |     |
|            | L2(Ω)          |        | H2(Ω)     |          |         |      |     |     |     |
Thisbehaviorisidenticalwiththatforthebackwardfractionaldiffusion.Thestatementholds
also for the inverse source problem for the classical diffusion case. This is not surprising,
sincewithaspacedependentsourcetermf,thesolutionutotheforwardproblemcanbesplit
into the steady solution u and the decaying solution u , i.e., u = u s + u d , where u and u
|     |     | s   |     |     | d   |     |     | s   | d   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
solve
−u″=
|     | s   | f, u s (0) | = u | s (1) = 0, |     |     |     |     |     |
| --- | --- | ---------- | --- | ---------- | --- | --- | --- | --- | --- |
and
|     | ∂αu − | u =  | 0, u | (0, x) = | f(x), u | (0, t) = | u (1, t) | = 0, |     |
| --- | ----- | ---- | ---- | -------- | ------- | -------- | -------- | ---- | --- |
|     | t d   | d,xx |      | d        | d       |          | d        |      |     |
respectively. By the decay behavior of the solution u , the steady state component u is
|     |     |     |     |     | d   |     |     |     | s   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
confirmed
dominating, which amounts to a two spatial derivative loss. This is fully by the
numerical experiments, cf figure 7. It is observed that the condition number is almost
independentofthefractionalorderα,anditisoforderO(103),reflectingthemildlyill-posed
nature of the inverse problem. In particular, for large terminal time T, the singular value
spectra are almost identical for all fractional orders, decaying to zero at an algebraic rate, cf
figure
7(b).
Next we turn to the time dependent case, i.e., seeking a source term f of the form
p(t)q(x),withaknownspacialcomponentq(x),fromthefinaltimedatag
| f(x, t) = |     |     |     |     |     |     |     | =   | u(T). |
| --------- | --- | --- | --- | --- | --- | --- | --- | --- | ----- |
Mathematically, the inverse problem even for the classical diffusion equation has not been
18

| InverseProblems31(2015)035003 |     |     |     |     |     |     |     |     | BJinandWRundell |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --------------- |
completely analyzed. The inclusion of a nontrivial term q(x) is important since without this
there is nonuniqueness. To see this, we take u to satisfy u t − u xx = f(t) on(0, 1) × (0, T)
with initial data u(x, 0) = 1 and a homogeneous Neumann boundary condition
−u (0, t) = u (1, t) = 0. Then one solution satisfying u(x, T) = g(x) = 1 is given by
|     | x   | x   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
u(x, t) = 1 and f ≡ 0, but another is u(x, t) = cos(2πt T) and f = (−2π T)sin(2πt T).
Likewise, in the fractional case, we can take u = cos(2πt T) for the second solution and
| define |      |             | αth   | Djrbashian–Caputo |            |            |            |     |       |
| ------ | ---- | ----------- | ----- | ----------------- | ---------- | ---------- | ---------- | --- | ----- |
|        | f to | be its      | order |                   |            | fractional | derivative | in  | time. |
|        | Like | previously, | the   | solution          | u to (3.1) | is given   | by         |     |       |
∞
|     |     |      | ∑∫  | t   |          | (     | τ)α)     | (   | )     |
| --- | --- | ---- | --- | --- | -------- | ----- | -------- | --- | ----- |
|     |     | u(t) | =   | (t  | − τ)α−1E | −λ (t | − p(τ)dτ | q,  | ϕ ϕ . |
|     |     |      |     |     |          | α,α j |          |     | j j   |
0
j=1
| Hence | the    | measured | data         | g(x) = | u(x, T) | is given by      |      |        |             |
| ----- | ------ | -------- | ------------ | ------ | ------- | ---------------- | ---- | ------ | ----------- |
|       |        |          | ∞            | T      |         |                  |      |        |             |
|       |        |          | ∑∫           |        | τ)α−1E  | (                | τ)α) | (      | )           |
|       |        | g(x)     | =            | (T     | −       | α,α −λ j (T      | −    | p(τ)dτ | q, ϕ ϕ (x). |
|       |        |          |              |        |         |                  |      |        | j j         |
|       |        |          | j=1          | 0      |         |                  |      |        |             |
| By    | taking | inner    | product with | ϕ      | on both | sides, we deduce |      |        |             |
j
|     |     | (   | )   | (    | )∫ T |          |         |        |         |
| --- | --- | --- | --- | ---- | ---- | -------- | ------- | ------ | ------- |
|     |     | g,  | ϕ = | q, ϕ | (T   | − τ)α−1E | ( −λ (T | − τ)α) | p(τ)dτ. |
|     |     |     | j   |      | j    | α,α      | j       |        |         |
0
| In  | the case | of α | = 1, the | formula | recovers | the relation     |     |     |     |
| --- | -------- | ---- | -------- | ------- | -------- | ---------------- | --- | --- | --- |
|     |          | (    | )        | (       | )∫ T     |                  |     |     |     |
|     |          | g,   | ϕ =      | q, ϕ    |          | e−λj(T−τ)p(τ)dτ, |     |     |     |
|     |          |      | j        |         | j        |                  |     |     |     |
0
which resembles a finite-time Laplace transform or moment problem, and thus severely
smoothing,whichrenderstheinversesourceproblemseverelyill-posed.Intuitively,theterm
e−λj(T−t)
can only pick up the information for t close to the terminal time T, and for t away
fromT,theinformationisseverelydamped,especiallyforhighfrequencymodes,whichleads
totheseverelyill-posednatureoftheinverseproblem.Inthefractionalcase,theforwardmap
F from the unknown to the data is clearly compact, and thus the problem is still ill-posed.
However,thekerneltα−1E (−λ tα)islesssmoothanddecaysmuchslower,andonemight
|     |     |     |     | α,α | j   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
expectthattheproblemislessill-posedthanthecanonicaldiffusioncounterpart.Toexamine
the point, we present the numerical results for the inverse problem in figure 8. It is severely
ill-posedirrespectiveofthefractionalorderα:thesingularvaluesdecayexponentiallytozero
withoutadistinctgapinthespectrum.Inparticular,fortheterminaltimeT=1,thespectrum
is almost identical for all fractional orders α. For small T, the singular values still decay
α,
exponentially, but the rate is different: the smaller is the fractional order the faster is the
decay, cf figure 8(a). Consequently, a few more modes of the source term p(τ) might be
recovered. In other words, due to a slower local decay of the exponential function e−λt,
|     |     |     | Mittag-Leffler |     |     |     |     | figure |     |
| --- | --- | --- | -------------- | --- | --- | --- | --- | ------ | --- |
compared with the function tα−1E (−λtα), cf 1(a), actually more
α,α
frequency modes can be picked up by normal diffusion than the fractional counterpart, cf
| figure |     |     |     |     | sufficiently |     |     |     |     |
| ------ | --- | --- | --- | --- | ------------ | --- | --- | --- | --- |
8(a). This indicates that with accurate data, at a small time instance, the
sideways problem for normal diffusion may allow recovering more modes, i.e., anomalous
| diffusion |     | does not | help solve | the | inverse | problem. |     |     |     |
| --------- | --- | -------- | ---------- | --- | ------- | -------- | --- | --- | --- |
flux
In practice, the accessible data can also be the data at the end point, e.g., x = 0 or
x=1.Webrieflydiscussthecaseofrecoveringatimedependentcomponentp(t)inthesource
q(x)p(t)fromthefluxdataatx=0.Byrepeatingtheprecedingargument,thedata
term f =
| g:= | −u (0, | t) is | related | to the unknown |     | p(t) by |     |     |     |
| --- | ------ | ----- | ------- | -------------- | --- | ------- | --- | --- | --- |
x
19

| InverseProblems31(2015)035003 |             |         |                         |                |              |                        | BJinandWRundell   |         |
| ----------------------------- | ----------- | ------- | ----------------------- | -------------- | ------------ | ---------------------- | ----------------- | ------- |
|                               | Figure8.The |         | singular                | value spectrum | at           | two different terminal | times for the     | inverse |
|                               | source      | problem | with                    | a final        | time data at | terminal time T,       | and f(x,t)=xp(t), | an      |
|                               | unknown     |         | time dependentcomponent |                | p(t).        |                        |                   |         |
∞ t
|     |      | −∑∫ |     | τ)α−1E | (        | τ)α) (      | )              |     |
| --- | ---- | --- | --- | ------ | -------- | ----------- | -------------- | --- |
|     | g(t) | =   |     | (t −   | α,α −λ j | (t − p(τ)dτ | q(x), ϕ ϕ′(0). |     |
j j
j=1 0
In [88, theorem 4.4], a stability result was established for the recovery of the time
dependent component p(t). Along the same line of thought, under reasonable assumptions,
| one can | deduce that |     |     |     |     |     |     |     |
| ------- | ----------- | --- | --- | --- | --- | --- | --- | --- |
c∥∂αg∥
|     | ∥p∥ |        | ⩽   | .        |     |     |     |     |
| --- | --- | ------ | --- | -------- | --- | --- | --- | --- |
|     |     | C[0,T] |     | t C[0,T] |     |     |     |     |
The inverse problem roughly amounts to taking the αth order Djrbashian–Caputo fractional
α
derivative in time. Hence as the fractional order decreases from unity to zero, it becomes
less and less ill-posed. For α close to zero, it is nearly well-posed, at least numerically. In
other words, anomalous diffusion can mitigate the degree of ill-posedness for the inverse
problem.Toillustratethediscussion,wepresentinfigure9somenumericalresults,wherethe
forwardmapFisfromthetimedependentcomponentp(t)tothefluxdatag(t)atx=0,both
definedovertheinterval[0, T],discretizedusingacontinuouspiecewiselinearfiniteelement
basis. The condition number of the discrete forward map F decreases monotonically as the
fractionalorderαdecreasesfromunitytozero,confirmingtheprecedingdiscussions.Further,
| the terminal | time | T does | not affect | the condition | number | to a large extent. |     |     |
| ------------ | ---- | ------ | ---------- | ------------- | ------ | ------------------ | --- | --- |
Itiswidelyacceptedininverseheatconductionthataninverseproblemwillbeseverely
ill-posed when the data and unknown are not aligned in the same space/time direction, and
only mildly ill-posed when they do align with each other. Our discussions with the inverse
source problems indicate that the observation remains valid in the time fractional diffusion
case. In particular, although not presented, we note that the inverse source problem of
recovering a space dependent component from the lateral Cauchy data is severely ill-posed
for both fractional and normal diffusion. In the simplest case of a space dependent-only
sourceterm,itismathematicallyequivalenttouniquecontinuation,awellknownexampleof
| severely | ill-posed | inverse | problems. |     |     |     |     |     |
| -------- | --------- | ------- | --------- | --- | --- | --- | --- | --- |
The inverse source problems for the classical diffusion equation have been extensively
studied; see e.g., [7, 9, 37]. Inverse source problems for FDEs have also been numerically
20

| InverseProblems31(2015)035003 |     |     |     |     | BJinandWRundell |     |
| ----------------------------- | --- | --- | --- | --- | --------------- | --- |
Figure9.Numericalresultsfortheinversesourceproblemwithfluxdataatx=0and
|     | f(x,t)=xp(t), | an unknown             | time dependent | component p(t).        | (a) The | condition |
| --- | ------------- | ---------------------- | -------------- | ---------------------- | ------- | --------- |
|     | number        | ofthe discrete forward | map and(b)     | singular valuespectrum | at T=   | 1.        |
studied. Zhang and Xu [111] established the unique recovery of a space dependent source
termin(3.1)withpureNeumannboundarydataandoverspecifiedDirichletdataatx=0.This
isachievedbyaneigenfunctionexpansionandLaplacetransform,andtheuniquenessfollows
from a unique continuation principle of analytic functions. Sakamoto and Yamamoto [89]
discussed theinverse problem ofdetermining a spatially varying function ofthesource term
final
by overdetermined data in multi-dimensional space, and established its well-posedness
in the Hadamard sense except for a discrete set of values of the diffusion constant, using an
analytic Fredholm theory. Very recently, Luchko et al [68] showed the uniqueness of reco-
vering a nonlinear source term from the boundary measurement, and developed a numerical
schemeoffixedpointiterationtype.Aleroevetal[2]showedtheuniquenessofrecoveringa
spacedependentsourcetermfromintegraltypeobservationaldata.Recently,therearemany
numerical studies on this class of inverse problems. In [101], the numerical recovery of a
spatiallyvaryingfunctionofthesourcetermfromthefinaltimedatainageneraldomainwas
studied using a quasi-boundary value problem method; see also [98, 112] related studies.
Wangetal[100]proposedtodeterminethespace-dependentsourcetermfromthefinaltime
| data in multi-dimension |           | using a reproducing | kernel Hilbert | space method. |     |     |
| ----------------------- | --------- | ------------------- | -------------- | ------------- | --- | --- |
| 3.4. Inverse            | potential | problem             |                |               |     |     |
Now we consider a nonlinear inverse coefficient problem for the time fractional diffusion
|           |           | final         | find      |                    |       |       |
| --------- | --------- | ------------- | --------- | ------------------ | ----- | ----- |
| equation: | given the | time data g = | u(T), the | potential q in the | model |       |
|           | ∂αu       | u qu inΩ,     |           |                    |       |       |
|           | t −       | xx + = 0      |           |                    |       | (3.5) |
with a homogeneous Neumann boundary condition and initial data v. The parabolic
counterparthasbeenextensivelystudied[15,16,38],whereitwasshownthattheproblemis
nearlywell-posedintheHardamardsenseinsuitableHölderspace,undercertainconditions,
using the strong maximum principle. In [38], an elegant fixed point method was developed,
and the monotone convergence of the method was established. It can be adapted
straightforwardly to the fractional case: given an initial guess q0, compute the update qk
21

| InverseProblems31(2015)035003 |     |     |     |     | BJinandWRundell |
| ----------------------------- | --- | --- | --- | --- | --------------- |
L2(Ω)
|     | Figure10.Numerical | results, i.e., | the relative | error e, for | the inverse potential |
| --- | ------------------ | -------------- | ------------ | ------------ | --------------------- |
exactfinal
|             | problem from | timedata       | at (a) T= 0.1and(b)α=12. |     |     |
| ----------- | ------------ | -------------- | ------------------------ | --- | --- |
| recursively | by           |                |                          |     |     |
|             |              | ( )            |                          |     |     |
|             | g″           | − ∂αu x, T; qk |                          |     |     |
t
|     | qk+1= | ,   |     |     |     |
| --- | ----- | --- | --- | --- | --- |
g
where the notation u(x, T; qk) denote the solution to problem (3.5) with the potential qk at
t = T. Since the strong maximum principle is still valid for the time fractional diffusion
equation [110], the scheme is monotonically convergent, under suitable conditions.
As the terminal time T → ∞, the problem recovers a steady-state problem, and the
scheme amounts to twice numerical differentiation in space and converges within one
iteration, provided that the data g is accurate enough. Hence, it is natural to expect that the
convergence oftheschemewilldependscrucially onthetime T:thelarger isthetime T,the
closer is the solution u to the steady state solution; and thus the faster is the convergence of
thefixedpointscheme.Bylemma2.1,asthefractionalorderαapproacheszero,thesolution
| u   |     | t   |     | α   |     |
| --- | --- | --- | --- | --- | --- |
decays much faster around = 0 than the classical one, i.e., = 1. In other words, the
‘quasi-steady state’
fractional diffusion problem can reach a much faster than the classical
one, especially for α close to zero, and the scheme will then converge much faster.
Toillustratethepoint,wepresentinfigure10somenumericalresultsofreconstructinga
discontinuous potentialq = 1 + 2xχ + 2(1 − χ x) (with χ being thecharacteristic
|     |     | [0,0.5] | (0.5,1] | S   |     |
| --- | --- | ------- | ------- | --- | --- |
functionofthesetS).Inordertoillustratetheconvergencebehaviorofthefixedpointscheme
figure,
we take exact data. In the e denotes the relative L2(Ω) error. The numerical results
fullyconfirmtheprecedingdiscussions:atafixedtimeT,thesmalleristhefractionalorderα,
|     |     | fixed | α,  |     |     |
| --- | --- | ----- | --- | --- | --- |
the faster is the convergence; and at the larger is the time T, the faster is the
convergence. Numerically, one also observes the monotone convergence of the scheme.
Generally,therecoveryofacoefficientinFDEshasnotbeenextensivelystudied.Cheng
etal[14]establishedtheuniquerecoveryofthefractionalorderα ∈ (0, 1)andthediffusion
coefficientfromthelateralboundarymeasurements.Itrepresentsoneofthefirstmathematical
worksoninvereproblemsforFDEs,andhasinspiredmanyfollow-upworks.Yamamotoand
Zhang [109] established conditional stability in determining a zeroth-order coefficient in a
22

InverseProblems31(2015)035003 BJinandWRundell
one-dimensional FDE with one half order Caputo derivative by a Carleman estimate. Car-
leman estimates for time fractional diffusion were discussed in [13, 62, 108]. In [73], the
unique determination of the spatial coefficient and/or the fractional order from the data on a
subdomain was shown for a positive initial condition. Wang and Wu [97] studied the
simultaneous recovery of two time varying coefficients, i.e., a kernel function and a source
function, from the additional integral observation in multi-dimension, using a fixed point
theorem.Alltheseworksareconcernedwiththetheoreticalanalysis,antthereareevenfewer
works on the numerical analysis of related inverse problems. Li et al [58] suggested an
optimal perturbation algorithm for the simultaneous numerical recovery of the diffusion
coefficientandfractionalorderinaone-dimensionaltimefractionalFDE.In[50],theauthors
considered the identification of a potential term from the lateral flux data at one fixed time
instance corresponding to a complete set of source terms, and established the unique deter-
minationfor‘small’potentials.Further,aNewtontypemethodwasproposedin[50],andits
convergence was shown.
Even though our discussions have focused on time fractional diffusion, which involves
onesinglefractionalderivativeintime,itisalsopossibletoconsiderequationswherethetime
derivative involves multiple factional orders, i.e., ∑m c ∂αk for a sequence
k=1 k t
α > α > … > α [44,60];see[59]forsomefirstuniquenessresultsforinversecoefficient
1 2 m
problems in the multi-dimensional case. Further extensions include the distributed-order,
spatiallyand/ortemporallyvariable-orderandtemperedfractionaldiffusion,tobettercapture
certain physical processes, for which, however, related inverse problems have not been
discussed at all.
3.5. Fractional derivativeas aninversesolution
OneoftheveryfirstundeterminedcoefficientproblemsforPDEswasdiscussedinthepaper
by Jones [52] (see also [8, chapter 13]). This is to determine the coefficient a(t) from
u =a(t)u , 0 < x < ∞, t > 0
t xx
u(x, 0)=0, −a(t)u (0, t) = g(t), 0 < t < T
x
under the over-posed condition of measuring the ‘temperature’ at x = 0
u(0, t) = ψ(t)
In[52],Jonesprovidedacompleteanalysisoftheproblem,bygivingnecessaryandsufficient
conditionsforauniquesolutionaswellasdeterminingtheexactlevelofill-conditioning.The
key step in the analysis is a change of variables and conversion of the problem to an
equivalentintegralequationformulation.Perhapssurprisingly,thisapproachinvolvestheuse
of a fractional derivative as we now show.
Theassumptionsarethatgiscontinuousandpositiveandψiscontinuouslydifferentiable
with ψ(0) = 0 and ψ′ > 0 on(0, T). In addition, the function h(t) defined by
πg(t)
h(t) =
∫ t (t − τ)−12ψ′(τ)dτ
0
satisfieslim h(t) = h > 0.Notethathistheratioofthetwodatafunctions;thefluxgand
t→0 0
theDjrbashian–Caputoderivativeoforder1 2ofψ.Ifwedefineh = infhandh = suph on
i s
[0, T] and look at the space  :={a ∈ C[0, T): h2 ⩽ a(t) ⩽ h2}, then it was shown that
i s
any a ∈  satisfies the inverse problem must also solve the integral equation
23

| InverseProblems31(2015)035003 |     |     |     |     |     |     | BJinandWRundell |     |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --------------- | --- |
πg(t)
=:a,
a(t) =
|     |     | t        | ⎡   | t       | ⎤−12 |     |     |     |
| --- | --- | -------- | --- | ------- | ---- | --- | --- | --- |
|     |     | ∫ ψ′(τ)⎣ | ∫   | a(s)ds⎦ | dτ   |     |     |     |
|     |     | 0        |     | τ       |      |     |     |     |
and vice-versa. The main result in [52] is that the operator has a unique fixed point on
|     |    |     |     |     |     |     | ,  |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
and indeed is monotone in the sense of preserving the partial order on i.e., if a ⩾ a
1 2
| thena | ⩽ a . |     |     |     |     |     |     |     |
| ------ | ------ | --- | --- | --- | --- | --- | --- | --- |
| 1      | 2      |     |     |     |     |     |     |     |
Given these developments, it might seem that a parallel construction for the time frac-
tional diffusion counterpart, ∂αu = a(t)u , would be relatively straightforward but this
|     |     |     | t   | xx  |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
seems nottobe thecase. The basic steps for theparabolic version require items that just are
nottrueinthefractionalcase,suchastheproductrule,andwithoutthesetheabovestructure
| cannot be  | replicated | or at | least not | without    | some further | ingenuity. |     |     |
| ---------- | ---------- | ----- | --------- | ---------- | ------------ | ---------- | --- | --- |
| 4. Inverse | problems   | for   | space     | fractional | diffusion    |            |     |     |
Now we turn to differential equations involving a fractional derivative in space. There are
severalpossiblechoicesofafractionalderivativeinspace,e.g.,Djrbashian–Caputofractional
Riemann–Liouville
derivative, fractional derivative, Riesz derivative, and fractional Lapla-
cian [3], which all have received considerable attention. In recent years, the use of the
fractional Laplacian is especially popular in high-dimensional spaces, and admits a well-
Djrbashian–Caputo
developed analytical theory. We shall focus on the left-sided fractional
derivative CDβ, β ∈ (1, 2), and the one-dimensional case, and consider the following four
0 x
Sturm–Liouville
inverse problems: inverse problem, Cauchy problem for a fractional elliptic
| equation, | backwards | diffusion, | and | sideways | problem. |     |     |     |
| --------- | --------- | ---------- | --- | -------- | -------- | --- | --- | --- |
Sturm–Liouville
| 4.1. Inverse |     |     | problem |     |     |     |     |     |
| ------------ | --- | --- | ------- | --- | --- | --- | --- | --- |
FirstweconsiderthefollowingSturm–LiouvilleproblemontheunitintervalΩ = (0, 1):find
| u ∈ H1(Ω)∩Hβ(Ω) |     | and λ | ∈  such | that |     |     |     |     |
| --------------- | --- | ----- | -------- | ---- | --- | --- | --- | --- |
0
−CDβu
|     |     | + qu | = λu | inΩ, |     |     |     | (4.1) |
| --- | --- | ---- | ---- | ---- | --- | --- | --- | ----- |
0 x
with a homogeneous Dirichlet boundary condition u(0) = u(1) = 0. A Sturm–Liouville
problem of this form was considered by Djrbashian [19, 22] in 1960s to construct certain
biorthogonal basis for spacesofanalytic functions; seealso [78].Likebefore, with β = 2, it
recovers the classical Sturm–Liouville problem. In the case of a general potential q, in the
fractional case, little is known about the analytical properties of the eigenvalues and
eigenfunctions. For the case of a zero potential q = 0, there are countably many eigenvalues
| {λ  |     |     |     | Mittag-Leffler |     | E (−λ). |     |     |
| --- | --- | --- | --- | -------------- | --- | ------- | --- | --- |
j } to (4.1), which are zeros of the function β,2 The corresponding
eigenfunctionsaregivenby xE (−λ xβ).UsingtheexponentialasymptoticsontheMittag-
β,2 j
Lefflerfunction
|     |     | inlemma2.1,one[50,93]canshowthatasymptotically, |     |     |     |     | theeigenvalues | λ   |
| --- | --- | ----------------------------------------------- | --- | --- | --- | --- | -------------- | --- |
j
| are distributed | as  |     |     |     |     |     |     |     |
| --------------- | --- | --- | --- | --- | --- | --- | --- | --- |
(2 − β)π
|     |     | (2πj)β |     | (     | )   |     |     |     |
| --- | --- | ------ | --- | ----- | --- | --- | --- | --- |
|     | λ   | j ∼    | and | arg λ | j ∼ | .   |     |     |
2
Hence,forany β ∈ (1, 2),thereareonlyafinitenumberofrealeigenvaluesto(4.1),andthe
| rest appears | as complex | conjugate |     | pairs. |     |     |     |     |
| ------------ | ---------- | --------- | --- | ------ | --- | --- | --- | --- |
Itiswellknownthateigenvaluescontainvaluableinformationabouttheboundaryvalue
problem. For example it is known that the sequence of Dirichlet eigenvalues can uniquely
determine a potential q symmetric with respect to the point x = 1 2, and together with
additionalspectralinformation,onecanuniquelydetermineageneralpotentialq;see[12,86]
24

InverseProblems31(2015)035003 BJinandWRundell
Figure 11. Numerical results for the inverse Sturm–Liouville problem with a
Djrbashian–Caputo derivative for (a) β=53 and (b) β=74. The reconstructions
arecomputedfromthefirsteighteigenvalues(inabsolutevalue)usingafrozenNewton
method [50].
foranoverviewofresultsontheclassicalinverseSturm–Liouvilleproblem.Inthefractional
case, the eigenvalues are generally genuinely complex, and a complex number may carry
more information than a real one. Thus one naturally wonders whether these complex
eigenvalues do contain more information about the potential. Numerically the answer is
affirmative.Toillustratethis,weshowsomenumericalreconstructionsinfigure11,obtained
byusingafrozenNewtonmethodandrepresentingthesought-forpotentialqinFourierseries
[50]. The Dirichlet eigenvalues can be computed efficiently using a Galerkin finite element
method [45]. One observes that one single Dirichlet spectrum can uniquely determine a
general potential q. Unsurprisingly, as the fractional order β tends two, the reconstruction
becomes less and less accurate, since in the limit β = 2, the Dirichlet spectrum cannot
uniquely determine a general potential q. Theoretically, the surprising uniqueness in the
fractional case remains to be established.
Naturally, one can also consider the Riemann–Liouville case:
−RDβu + qu = λu inΩ, (4.2)
0 x
with u(0) = u(1) = 0. Like before, little is known about the analytical properties of the
eigenvaluesandeigenfunctions.Inthecaseofazeropotentialq=0,therearecountablymany
eigenvalues to (4.2), which are zeros of the Mittag-Leffler function E (−λ), and the
β,β
corresponding eigenfunctions are given by xβ−1E (−λ xβ). Further, the asymptotics of the
β,β j
eigenvalues are still valid. Hence, for any β ∈ (1, 2), there are only a finite number of real
eigenvalues to (4.2), and the rest appears as complex conjugate pairs.
The numerical results from the Dirichlet spectrum in the Riemann–Liouville case are
showninfigure12.Forageneralpotentialq,thereconstructionrepresentsonlythesymmetric
part, which is drastically different from the Djrbashian–Caputo case, but identical with that
fortheclassicalSturm–Liouvilleproblem.Further,ifweassumethatthepotentialqisknown
on the left half interval, then the Dirichlet spectrum allows uniquely reconstructing the
potential q on the remaining half interval, cf figure 12(b). These results indicate that in the
25

InverseProblems31(2015)035003 BJinandWRundell
Figure12.NumericalresultsfortheinverseSturm–LiouvilleproblemwithaRiemann–
Liouvillefractionalderivativeoforderβ=43.Thereconstructionsarecomputedfrom
the firsteight eigenvalues (in absolute value) using afrozenNewtonmethod [50].
Riemann–Liouville case the complex spectrum is not more informative than the classical
Sturm–Liouville problem. The precise mechanism underlying the fundamental differences
betweentheDjrbashian–CaputoandRiemann–Liouvillecasesawaitsfurtherstudy.However,
as in the classical case β = 2, one can show that the linearized derivative of the map
q → u(1; λ, q)aroundq=0cannotspanmorethanthesubspaceofevenfunctionsinL2(Ω).
Ingeneral,theSturm–Liouvilleproblemwithafractionalderivativeremainscompletely
elusive, and numerical methods such as finite element method [46] provide a valuable (and
oftentheonly)toolforstudyingitsanalyticalproperties.ForavariantofthefractionalSturm–
Liouville problem, which contains a fractional derivative in the lower-order term, Malamud
[71] established the existence of a similarity transformation, analogous to the well-known
Gelʼfand–Levitan–Marchenko transformation, and also the unique recovery of the potential
frommultiplespectra.Intheclassicalcase,theGelʼfand–Levitan–Marchenkotransformation
lends itselftoa constructive algorithm [86]; however, itisunclear whether thisistrue inthe
fractional case. In [50], the authors proposed a Newton type method for reconstructing the
potential, which numerically exhibits very good convergence behavior. However, a rigorous
convergence analysis of the scheme is still missing. Further, the uniqueness and non-
uniquenessissuesofrelatedinverseSturm–Liouvilleproblemsareoutstanding.Last,asnoted
above, there are other possible choices of the space fractional derivative, e.g., fractional
Laplacian and Riesz derivative. It is unknown whether the preceding observations are valid
for these alternative derivatives.
4.2. Cauchy problemforfractional elliptic equation
OneclassicalellipticinverseproblemistheCauchyproblemfortheLaplaceequation,which
playsafundamentalroleinthestudyofmanyellipticinverseproblems[40].Afirstexample
was given by Jacques Hadamard [31] to illustrate the severe ill-posedness of the Cauchy
problem, which motivated him to introduce the concept of well-posedness and ill-posedness
for problems in mathematical physics. So a natural question is whether the Cauchy problem
for the fractional elliptic equation is also as ill-posed? To illustrate this, we consider the
26

| InverseProblems31(2015)035003 |     |     |     |     |     |     |     | BJinandWRundell |     |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- |
following fractional elliptic problem on the rectangular domain Ω = {(x, y) ∈ 2:
| 0 <  | x < 1,         | 0 < y | < 1}  |         |        |            |      |     |       |
| ---- | -------------- | ----- | ----- | ------- | ------ | ---------- | ---- | --- | ----- |
|      |                | CDβu  | +CDβu | =       | 0inΩ,  |            |      |     | (4.3) |
|      |                | 0     | x     | 0 y     |        |            |      |     |       |
| with | the fractional |       | order | β ∈ (1, | 2) and | the Cauchy | data |     |       |
∂u
|     |     | u(x, | 0)  | = g(x) | and | (x, 0) | = h(x), 0 | < x < 1, |     |
| --- | --- | ---- | --- | ------ | --- | ------ | --------- | -------- | --- |
∂ν
whereνistheunitoutwardnormaldirection.Withβ = 2,itrecoverstheCauchyproblemfor
the Laplace equation. By applying the separation of variables, we can assume that
| u(x, | y) = | ϕ(x)ψ(y), | which | directly | gives | for some | scalar | λ ∈  that |     |
| ---- | ---- | --------- | ----- | -------- | ----- | -------- | ------ | ---------- | --- |
CDβϕ(x)=−λϕ(x),
|     |     | 0   | x   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
CDβψ(y)=λψ(y),
|     |     | 0   | y   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
−CDβ
Let (λ , ϕ ) be a Dirichlet eigenpair of the Caputo derivative operator on the unit
|     | j   | j   |     |     |     |     |     | 0   | x   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
intervalD = (0, 1),i.e.,ϕ (x) = xE (−λ xβ),and|λ | → ∞as j → ∞[51];seesection4.1
|     |     |     |     | j   | β,2 | j   | j   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
for further details. With the choice ϕ = ϕ and the Cauchy data pair
j
|     |         |          |        | xβ)      |               |     | satisfies |     |     |
| --- | ------- | -------- | ------ | -------- | ------------- | --- | --------- | --- | --- |
| (g, | h j ) = | (0, − xE | β,2 (λ | j λ j ), | the component |     | ψ         |     |     |
j
|     |     | CDβψ | (y) | = λ ψ | (y) iny | ∈ (0, ∞), |     |     |     |
| --- | --- | ---- | --- | ----- | ------- | --------- | --- | --- | --- |
|     |     | 0    | y j | j     | j       |           |     |     |     |
with the initial condition ψ (0) = 0 and dψ (0) = 1λ . Using the relation
|     |     |     |     |     | j   |     | j   | j   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
dy
| dxγ−1E     |     | (λxβ)    | λxγ−2E       | (λxβ)    |      |          |           |                   |          |
| ---------- | --- | -------- | ------------ | -------- | ---- | -------- | --------- | ----------------- | -------- |
|            | β,γ | =        |              | β,γ−1    | [53, | p 46],   | we deduce | that the solution | ψ to the |
| dx         |     |          |              |          |      |          |           |                   | j        |
| fractional |     | ordinary | differential | equation |      | is given | by        |                   |          |
|            |     |          |              | (        | )    |          |           |                   |          |
|            |     | ψ        | (y) =        | yE λ     | yβ λ | .        |           |                   |          |
|            |     |          | j            | β,2      | j j  |          |           |                   |          |
Hence, u (x, y) = xE (−λ xβ)yE (λ yβ) λ2 is a solution to the Cauchy problem with
|     | j   |     | β,2 | j   | β,2 | j j |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
g = 0 and h (x) = −xE (−λ xβ) λ . By the exponential asymptotics of the Mittag-Leffler
|     |     | j   |     | β,2 j | j   |     |     |     |     |
| --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- |
function, cf lemma 2.1, we deduce that h (x) → 0 as j → ∞, whereas for any y > 0, the
j
Mittag-Leffler
| solution | u   | (x, y) | → ∞ | as j → | ∞, in view | of the | exponential | growth of the |     |
| -------- | --- | ------ | --- | ------ | ---------- | ------ | ----------- | ------------- | --- |
j
function E (z), cf lemma 2.1. This indicates that the Cauchy problem for the fractional
β,2
elliptic equation is also exponentially ill-posed. However, the interesting question of the
degreeofill-posedness,incomparisonwiththeclassicalcase,isunclearandcertainlyworthy
of further study. Further, we note that the numerical solution of the fractional elliptic
equation(4.3)ishighlynontrivial,andthereseemstobenoefficientyetrigoroussolverinthe
literature and this seems to be due to a lack of theory about the solution to such problems.
| 4.3. | Backward | problem |     |     |     |     |     |     |     |
| ---- | -------- | ------- | --- | --- | --- | --- | --- | --- | --- |
Now we return to the backward diffusion problem with fractional derivatives in the space
variable(s). Let Ω = (0, 1) be the unit interval. Then the one-dimensional space fractional
| diffusion |     | equation | is given | by   |        |       |         |     |     |
| --------- | --- | -------- | -------- | ---- | ------ | ----- | ------- | --- | --- |
|           |     | u        | −CDβu    | = 0, | (x, t) | ∈ Ω × | (0, ∞), |     |     |
|           |     | t        | 0        | x    |        |       |         |     |     |
where the fractional order β ∈ (1, 2). The equation is equipped with the following initial
condition u(x, 0) = v and zero boundary condition u(0, t) = u(1, t) = 0. The backward
problem is: given the final time data g(x) u(x, T), find the initial data v. Since the
=
Djrbashian–Caputo derivative operatorCDβ with the zero Dirichlet boundary is sectorial on
0 x
27

InverseProblems31(2015)035003 BJinandWRundell
Figure13.Numericalresultsforthespacefractionalbackwarddiffusionproblem,the
singular valuespectrum at twodifferent timeinstances, (a) T=0.01 and(b) T= 0.1.
suitablespaces[42],theexistenceofasolutionufollowsfromtheanalyticsemigrouptheory
[43, 81], and formally it can be represented by
u(t) = e−Atv,
where A is the representation of the Djrbashian–Caputo derivative operator −CDβon its
0 x
domain. Formally, the solution v to the space fractional backward problem is given by
v = eATg.
In case of β = 2, using the eigenpairs {(λ , ϕ )} and the L2(Ω) orthogonality of the
j j
eigenfunctions{ϕ }, it recovers the well known formula
j
∞
( )
v = ∑eλjT g, ϕ ϕ .
j j
j=1
The growth factor eλjT explains the severely ill-posed nature of the inverse problem. In the
fractionalcase,suchanexplicitrepresentation isnolongeravailablesincethecorresponding
eigenfunctions {ϕ } are not orthogonal in L2(Ω) (actually they can be almost linearly
j
dependent), due to the non self adjoint nature of the Djrbashian–Caputo derivative operator
−CDβ.Nonetheless,accordingtothediscussionsinsection4.1,theeigenvalues{λ }increase
0 x j
toinfinitywiththeindexj,andasymptoticallyliesontworays.Hence,onenaturallyexpects
thatthebackwardproblemisalsoexponentiallyill-posed.However,themagnitudes(andthe
realparts)oftheeigenvaluesgrowatarateslowerthanthatofthestandardSturm–Liouville
problem, and thus the space fractional backward problem is less ill-posed than the classical
one. To illustrate the point, we present the numerical results in figure 13. For all fractional
orders β, the singular values decay exponentially, but the decay rate increases dramatically
with the increase of the fractional order β and the terminal time T. Hence, anomalous
superdiffusion does not change the exponentially ill-posed nature of the backward problem,
but numerically it does enable recovering more Fourier modes of the initial data v.
Last, we note that for other choices of the fractional derivative, e.g., the Riemann–
Liouville fractional derivative and the fractional Laplacian [6, 55], the magnitude of
28

InverseProblems31(2015)035003 BJinandWRundell
eigenvalues of the operator also tends to infinity, and the growth rate increases with the
β.
fractional order Therefore, the preceding observations on the space fractional backward
| problem | are expected | to  | be valid | for | these choices | as well. |
| ------- | ------------ | --- | -------- | --- | ------------- | -------- |
4.4. Sidewaysproblem
Last we return to the classical sideways diffusion problem but now with a fractional deri-
vative in space rather than in time. Let Ω = (0, 1) be the unit interval. Then the one-
| dimensional | space   | fractional | diffusion |        | equation is | given by |
| ----------- | ------- | ---------- | --------- | ------ | ----------- | -------- |
|             | u −CDβu |            | = 0,      | (x, t) | ∈ Ω × (0,   | ∞),      |
|             | t       | 0 x        |           |        |             |          |
where the fractional order β ∈ (1, 2). The equation is equipped with an initial condition
| u(x, 0) | = 0 and the | following |      | lateral | Cauchy boundary | conditions   |
| ------- | ----------- | --------- | ---- | ------- | --------------- | ------------ |
|         | u(0,        | t) =      | f(t) | and     | u (0, t) =      | g(t), t > 0. |
x
Wewishtocomputethesolutionatx= 1,i.e.,h(t):= u(1, t).Inthecase β = 2,themodel
recovers the standard diffusion equation, and we have already discussed the severe ill-
conditioningoftheclassicalcase.Duetothenonlocalnatureofthefractionalderivative,one
might expect that in the space fractional case, the sideways problem is less ill-posed. To see
this, we take Laplace transform in time to arrive at (with ^ denoting the Laplace transform)
−CDβu(x,
|     | zu(x, | z)  |     | z)  | = 0, |     |
| --- | ------ | --- | --- | --- | ---- | --- |
0 x
| with the | initial conditions |        | (at x | = 0) |             |        |
| -------- | ------------------ | ------ | ----- | ---- | ----------- | ------ |
|          | u(0,              | z) =f | (z)   | and  | u (0, z) = | g(z). |
x
| The solution | u(x, | z) to the | initial | value | problem   | is given by |
| ------------ | ----- | --------- | ------- | ----- | --------- | ----------- |
|              |       | =f       |         | ( zxβ | )         | ( zxβ )     |
|              | u(x, | z)        | (z)E    | β,1   | + g(z)xE | β,2         |
and thus
h(z) =f
|     |     |     | (z)E β,1 | (z) + | g(z)E β,2 (z). |     |
| --- | --- | --- | -------- | ----- | --------------- | --- |
Like before, the boundary condition h(t) at x = 1 can be found by an inverse Laplace
transform
|     | h(t) | = ∫ | ezth(z)dz. |     |     |     |
| --- | ---- | --- | ----------- | --- | --- | --- |
Br
In case of β = 2, this gives cosh z andsinh z z multipliers to the data f (z) and g(z)
resulting in the exponential ill-conditioning of the sideways heat problem. In the case of a
general β ∈ (1, 2), theexponential asymptotics inlemma 2.1 indicates that the problem still
suffers from exponentially growing multipliers to the data, and thus the problem is still
severely ill-conditioned. Simple computation shows that the multiplier is asymptotically
largerforthefractional orderβclosertounity.Inotherwords,anomalousdiffusioninspace
doesnotmitigatetheill-conditionednatureofthesidewaysproblem,butactuallyworsensthe
| conditioning | severely. |     |     |     |     |     |
| ------------ | --------- | --- | --- | --- | --- | --- |
To further illustrate the point, we compute the forward map F from the Dirichlet
boundaryconditionatx=1tothefluxatx=0numericallywithafiniteelementinspaceand
finite difference in time scheme, cf appendix A.3 for the details. The numerical results are
presentedinfigure14.Thesingularvaluespectraclearlyshowtheill-posednessnatureofthe
space fractional sideways problem: as the fractional order β increases from one to two, the
29

| InverseProblems31(2015)035003 |     |     |     | BJinandWRundell |
| ----------------------------- | --- | --- | --- | --------------- |
Figure14.SingularvaluespectrumoftheforwardmapFattimesT=0.1andT=1,for
|     | the sideways | problem with | Cauchy data at x= 0. |     |
| --- | ------------ | ------------ | -------------------- | --- |
majority of the singular values move upward, the decay of the singular values slows down,
andthusthesidewaysproblembecomeslessandlessill-posed(butstillseverelyso).Further,
therearemoretinysingularvalueskickinginasthefractionalorderβdecreasestoone,which
indicates the inherent rank deficiency of the forward map F and might be relevant in the
uniqueness of the inverse problem. This confirms the preceding analysis: the degree of ill-
posednessworsenswiththedecreaseofthefractionalorderβ,andthefractionalcounterpartis
more ill-posed than the classical one. In other words, anomalous diffusion actually severely
worsens the conditioning of the already very ill-posed sideways problem. Further, the
Djrbashian–Caputo
numerical results tend to indicate that the derivative with an order
β ∈ (1, 2) acts as an interpolation between the diffusion and convection, which results in a
history mechanism in space: when the history piece runs from the left to the right, it is
unlikelytopropagate theinformationinthereverse direction;andthecloseristhefractional
orderβtounity,thestrongeristhedirectionaleffect.Thelatterisnotcounterintuitive,sincein
|            | 1,theDjrbashian–CaputofractionalderivativeCDβ |     |     | recoversthefirstorder |
| ---------- | --------------------------------------------- | --- | --- | --------------------- |
| thelimitof | β =                                           |     |     | u                     |
0 x
derivative ∂u, and the problem is of convection type, and surely no information can be
∂x
| convected | backwards! |     |     |     |
| --------- | ---------- | --- | --- | --- |
Inthecase β = 2,onemayequallymeasurethelateralCauchydataatx=1,andaimsat
recoveringtheDirichletboundaryconditionatx=0.Clearly,thisdoesnotchangethenature
of the inverse problem, and it is equally ill-posed. Due to the directional nature of the
Djrbashian–Caputo derivative CDβ, one naturally wonders whether this ‘directional’ feature
0 x
doesinfluencetheill-posednatureofthesidewaysproblem.Toillustratethepoint,werepeat
| the preceding | arguments | and deduce |     |     |
| ------------- | --------- | ---------- | --- | --- |
−CDβu(x,
|     | zu(x, z) | z)  | = 0, |     |
| --- | --------- | --- | ---- | --- |
0 x
| with the | boundary conditions | at x = | 1   |     |
| -------- | ------------------- | ------ | --- | --- |
=f
|     | u(1, z) | (z) and | u (1, z) = g(z). |     |
| --- | -------- | ------- | ------------------ | --- |
x
f˜(z)
To derive the solution, denote the initial conditions at x = 0 by = u(0, z) and
g˜(z) = u (0, z). Then the solution u(x, z) to the initial value problem is given by
x
30

| InverseProblems31(2015)035003 |       |             |     |     |        |       |     | BJinandWRundell |
| ----------------------------- | ----- | ----------- | --- | --- | ------ | ----- | --- | --------------- |
|                               |       |             |     | ( ) |        | ( )   |     |                 |
|                               | u(x, | z) = f˜(z)E |     | zxβ | + g˜xE | zxβ . |     |                 |
|                               |       |             | β,1 |     |        | β,2   |     |                 |
dxγ−1E
Usethedifferentiationformula (zxβ) = zxγ−2E (zxβ)[53,p46]wededucethat
|                 |        |         | dx     | β,γ           |         | β,γ−1        |         |     |
| --------------- | ------ | ------- | ------ | ------------- | ------- | ------------ | ------- | --- |
| at x = 1, there | hold   |         |        |               |         |              |         |     |
|                 | f˜(z)E | (z)     | + g˜E  | (z)=          | f (z), |              |         |     |
|                 |        | β,1     |        | β,2           |         |              |         |     |
|                 | f˜(z)E |         |        | (z)=z−1g(z). |         |              |         |     |
|                 |        | β,0 (z) | + g˜E  | β,1           |         |              |         |     |
| Solving the     | linear | system  | yields | the solution  | to      | the sideways | problem |     |
(z)f
|     |       | E     | (z)    | − z−1E | (z)g(z) |     |     |     |
| --- | ----- | ----- | ------ | ------ | -------- | --- | --- | --- |
|     | f˜(z) | = β,1 |        |        | β,2      | ,   |     |     |
|     |       |       | E (z)2 | − E    | (z)E     | (z) |     |     |
|     |       |       | β,1    | β,0    | β,2      |     |     |     |
and accordingly the solution h(t) ≡ u(0, t) is given by an inverse Laplace transform. The
growth factors of the data f and g are E (z)(E (z)2 − E (z)E (z)) and
|     |     |     |     |     |     | β,1 | β,1 | β,0 β,2 |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- |
z−1E (z)(E (z)2 − E (z)E (z)), respectively. The growth of these factors at large z
| β,2 | β,1 | β,0 | β,2 |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
argumentdeterminesthedegreeofill-conditioningofthesidewaysproblem.Tothisend,we
appealtotheexponentialasymptoticoftheMittag-LefflerfunctionE
(z),cflemma2.1,and
α,β
note that Bromwhich path lies in the sector |argz| ⩽ π 2 to deduce that for large |z|,there
holds
|     |     |     |       | 1     |     | 2     |     |     |
| --- | --- | --- | ----- | ----- | --- | ----- | --- | --- |
|     |     | E   | (z)2∼ | e2z1β | −   | ez1β, |     |     |
β,1
|            |           |         | β2     |        | βΓ(1 | − β)z      |      |     |
| ---------- | --------- | ------- | ------ | ------ | ---- | ---------- | ---- | --- |
|            |           |         |        | 1      |      | 1          |      |     |
|            |           |         |        | e2z1β  |      | z1β−1ez1β. |      |     |
|            | E         | (z)E    | (z)∼   |        | −    |            |      |     |
|            |           | β,0 β,2 | β2     |        | βΓ(2 | − β)       |      |     |
| Hence, the | numerator | E       | (z)2 − | E (z)E | (z)  | behaves    | like |     |
|            |           | β,1     |        | β,0    | β,2  |            |      |     |
1
|     | E   | (z)2 − | E (z)E | (z) | ∼    | z1β−1ez1β |     | as z → ∞. |
| --- | --- | ------ | ------ | --- | ---- | --------- | --- | --------- |
|     | β,1 |        | β,0    | β,2 |      |           |     |           |
|     |     |        |        |     | βΓ(2 | − β)      |     |           |
This together with the exponential asymptotic of E (z) and E (z) from lemma 2.1
|     |     |     |     |     |     | β,1 |     | β,2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
f
indicatesthatthemultipliersfor andg aregrowingatmostataverylow-orderpolynomial
rate, for large z. Hence, the high-frequency components of the data noise are not amplified
much (at most polynomially instead of exponentially). The analysis indicates that the
sideways problem with the lateral Cauchy data specified at the point x = 1 is nearly well-
β
posed, as long as the fractional order is away from two, for which it recovers the classical
| ill-posed sideways |     | problem | for the | heat | equation. |     |     |     |
| ------------------ | --- | ------- | ------- | ---- | --------- | --- | --- | --- |
Next we illustrate the preceding discussions numerically. The behavior of the forward
mapFfromtheDirichletboundaryatx=0tothefluxdataatx=1isshowninfigure15.For
awiderangeofvaluesofthefractionalorderβ,theconditionnumberoftheforwardmapFis
of order 100, which is fairly mild, in view of the size of the linear system, i.e., 100 × 100.
Whenthefractionalorderβincreasestowardstwo,theinverseproblemrecoverstheclassical
sideways problem, and as expected, the condition number increases dramatically. However,
theonsetoftheblowupdependsontheterminaltimeT:thesmalleristhetimeT,thesmaller
seems the onset value. The precise mechanism for this phenomenon is still unknown. For
β ⩽ 7 4, the singular value spectrum only spans a narrow interval, resulting in a very small
conditionnumber.Physically,likebefore,thiscanbeexplainedasthe‘convective’natureof
the Djrbashian–Caputo fractional derivative: as the fractional order β tends to unity, the
informationatx=0istransportedtox=1,freefromdistortion,andthustheinverseproblem
is almost well-posed. In summary, depending on the location of the over-specified data,
31

InverseProblems31(2015)035003 BJinandWRundell
Figure15.Numericalresultsforthespacefractionalsidewaysproblem,withthelateral
Cauchydataatthepointx=1:(a)theconditionnumberversusthefractionalorderβ
and(b) the singular valuespectrum at T= 1.
anomalous superdiffusion can either help or aggravate the conditioning of the sideways
problem.
Last, we would like to note that the study of space fractional inverse problems, either
theoretical or numerical, is fairly scarce. This is partly attributed to the relatively poor
understandingofforwardproblemsforFDEswithaspacefractionalderivative:thereareonly
a few mathematical studies on one-dimensional space fractional diffusion, and no mathe-
maticalstudyonmulti-dimensionalproblemsinvolvingspacefractionalderivatives(ofeither
Riemann–Liouville or Caputo type). Nonetheless, our preliminary numerical experiments
show distinct new features for related inverse problems, which motivate their analytical
studies.
5. Concluding remarks
Anomalous diffusion processes arise in many disciplines, and the physics behind is very
different from normal diffusion. The unusual physics greatly influences the behavior of
relatedforwardproblems.Further,itiswellknownthatbackwardfractionaldiffusionismuch
less ill-posed than the classical backward diffusion, which has contributed to the belief that
inverseproblemsforanomalousdiffusionarealwaysbetterbehavedthanthatforthenormal
diffusion.Inthisworkwehaveexaminedseveralexemplaryinverseproblemsforanomalous
diffusion processes in a numerical and semi-analytical manner. These include the sideways
problem, backward problem, inverse source problem, inverse Sturm–Liouville problem and
Cauchy problems. Our findings indicate that anomalous diffusion can give rise to very
unusual new features, but they only partially confirm the belief: depending on the data and
unknown, it may influence either positively or negatively the degree of ill-posedness of the
inverse problem.
Themathematicalstudyofinverseproblemsinanomalousdiffusionisstillinitsinfancy.
There are only a few rigorous theoretical results on the uniqueness, existence and stability,
which mostly focus on the one-dimensional case, and there are many more open problems
32

| InverseProblems31(2015)035003 |     |     |     |     |     |     | BJinandWRundell |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --------------- |
awaitinginvestigations.Thedevelopmentofstableandefficientreconstructionproceduresis
an active ongoing research topic. However, due to thenonlocality of the forward model, the
construction of efficient schemes and their rigorous numerical analysis remain very chal-
lenging.ThisisespeciallytrueforspacefractionalFDEs,andtherearealmostnotheoretical
| or rigorous | numerical studies. |     |     |     |     |     |     |
| ----------- | ------------------ | --- | --- | --- | --- | --- | --- |
Acknowledgments
The authors are grateful for the anonymous referees for their constructive comments. The
| research of      | both authors         | is partly | supported           | by NSF  | Grant     | DMS-1319052. |     |
| ---------------- | -------------------- | --------- | ------------------- | ------- | --------- | ------------ | --- |
| Appendix         | A. Numerical         | methods   | for                 | special | functions | and FDEs     |     |
| A.1. Computation | of theMittag-Leffler |           | and Wrightfunctions |         |           |              |     |
Likemanyspecialfunctions,theefficientandaccuratenumericalcomputationoftheMittag-
Leffler function E (z) is delicate [28, 29, 94]. An efficient algorithm relies on partitioning
α,β
thecomplexplane intodifferentregions,wheredifferentapproximations,i.e.,powerseries,
integral representation and exponential asymptotic for small values of the argument, inter-
efficient
mediate values and large values, respectively, are used for numerical computation;
see [94] for the some partition and error estimates. The special case of the Mittag-Leffler
function E (z) with a real argument z ∈ , which plays a predominant role in time-frac-
α,β
efficiently
tional diffusion, can also be computed with the Laplace transform and suitable
| quadrature | rules [27]. |     |     |     |     |     |     |
| ---------- | ----------- | --- | --- | --- | --- | --- | --- |
The computation of the Wright function W ρ,μ (z) is even more delicate. In theory, like
before,itcanbecomputedusingpowerseriesforsmallvaluesoftheargumentandaknown
asymptotic formula for large values, and for the intermediate case, values are obtained by
using an integral representation [67]. The integral representation for the Wright function
W (z) for intermediate values in the case of interest for the fundamental solution in one-
ρ,μ
dimension (where ρ = −α 2 < 0, 0 < μ = 1 + ρ < 1and z = −x, x > 0) is given by
∫ ∞
|     | W (−x) | = K(x, | ρ, μ, | r)dr |     |     |     |
| --- | ------ | ------ | ----- | ---- | --- | --- | --- |
ρ,μ
0
| where the | kernel K(x, | ρ, μ, r) is                     | given by |     |            |        |     |
| --------- | ----------- | ------------------------------- | -------- | --- | ---------- | ------ | --- |
|           | K(x, ρ,     | μ, r) = r−μe−r+xcos(πρ)r−ρsin(x |          |     | sin(πρ)r−ρ | + πμ). |     |
Thisisasingularkernelwithaleadingorderr−μ= r−1−ρ,withsuccessivesingularkernelsof
the form r−1−2ρ, r−1−3ρ etc, upon expanding the terms. Hence, a direct treatment via
numericalquadratureisinefficient.Amoreefficientapproachistousethechangeofvariable
| s = r−ρ, i.e.,r | = s−1ρ, | and the transformed        |     | kernel | is               |          |        |
| --------------- | ------- | -------------------------- | --- | ------ | ---------------- | -------- | ------ |
|                 | K͠      | (−ρ)−1s(−ρ)−1(−μ+1)−1e−s−1 |     |        | ρ+xcos(πρ)ssin(x |          |        |
|                 | (x, ρ,  | μ, s) =                    |     |        |                  | sin(πρ)s | + πμ). |
The fundamental solution of the one-dimensional time-fractional diffusion equation is
expressedintermsofaWright functionW (−x) withthechoice ρ = −α 2 and μ = 1 + ρ,
ρ,μ
| cf (2.2). In | this case the | resulting | kernel K˜   | simplifies       | to  |            |            |
| ------------ | ------------- | --------- | ----------- | ---------------- | --- | ---------- | ---------- |
|              | K͠            |           | (−ρ)−1e−s−ρ | 1+xcos(πρ)ssin(x |     |            |            |
|              | (x, ρ,        | 1 + ρ, s) | =           |                  |     | sin(πρ)s + | (1 + ρ)π). |
33

InverseProblems31(2015)035003 BJinandWRundell
This kernel is free from the grave singularity, and thus the quadrature method is quite
effective. In general, the integral can be computed efficiently via the Gauss–Jacobi
quadrature,withtheweightfunctions(−ρ)−1(−μ+1)−1.WenotethatanalgorithmfortheWright
functionW (z)overthewholecomplexplane withrigorouserroranalysisisstillmissing.
ρ,μ
Theendeavorinthisdirectionwouldalmostcertainlyinvolvedividingthecomplexdomain
into different regions, and using different approximations on each region separately.
A.2. Timefractional diffusion
We describe a finite difference method for the initial boundary value problem for the one-
dimensional time-fractional diffusion equation
∂αu − u + qu = f(x, t) (x, t) ∈ Ω × (0, T),
t xx
with the initial condition u(x, 0) = v and boundary conditions
u(0, t) = g(t) and u(1, t) = h(t), t > 0.
Therearemanyefficientnumericalschemesfordiscretizingtheproblem.Thediscretizationin
space can be achieved by the standard central difference scheme, Galerkin finite element
method [47] or spectral method, and the discretization in time can be achieved with the L1
approximation [63, 96] and convolution quadrature [48]. We shall adopt the L1
approximationintimeandthecentraldifferenceinspace.Specifically,wedividetheinterval
[0, T] into uniform subintervals, with nodes t = kτ, k = 0,…, K, and a time step size
k
τ = T K. Similarly, we partition the spatial domain Ω into uniform subintervals, with grid
points x = ih, i = 0,…,N, and mesh size h = 1 N. Then the L1-approximation of the
i
Djrbashian–Caputo fractional derivative ∂αu(x, t ) developed in [63, 96] is given by:
t k
⎡ ⎤
k−1
∂αu(x, t ) ≈ τ−α ⎢ b u(x, t ) − b u(x, t ) + ∑( b − b ) u ( x, t )⎥ , (A.1)
t k ⎢ 0 k k−1 0 j j−1 k−j ⎥
⎣ ⎦
j=1
where the weights b are given by
j
( )
b = (j + 1)1−α − j1−α Γ(2 − α), j = 0, 1,…, K − 1.
j
If the solution u(x, t) is C2 continuous in time, the local truncation error of the L1
approximation is bounded bycτ2−α for some c depending only on u [63, equation (3.3)]. In
general, one can show that the scheme is only first-order accurate. Next with the central
difference scheme in space and the notation uk ≈ u(x, t ), we arrive at the following fully
i i k
discrete scheme
⎡ ⎤
k−1
⎢ b uk − b u0 + ∑( b − b ) uk−j⎥
⎢ 0 i k−1 i j j−1 i ⎥
⎣ ⎦
j=1
uk − 2uk + uk
+ i−1 i i+1 + quk = fk, i = 1,…, N − 1,
h2 i i i
with q = q(x ) and fk = f(x, t ). We note that at each time step, one needs to solve a
i i i i k
tridiagonal linear system. However, the right-hand side at the current step involves all
previoussteps,whichcanbequiteexpensiveforasmallstepsize,andthiswillmostlikelybe
required due to the first order in time convergence. This history piece represents one of the
maincomputationalchallengesfortimefractionaldifferentialequations.Therearehigh-order
schemes, e.g., convolution quadrature generated by the second-order backward difference
34

InverseProblems31(2015)035003 BJinandWRundell
formula[48].Further,wenotethatthefinitedifferenceschemeinspacecanbereplacedwith
the Galerkin finite element method, which is especially suitable for high dimensional
problems on a general domain and elliptic operator involving variable coefficients [47].
A.3. Spacefractional diffusion
NowwedescribeafullydiscreteschemebasedonthebackwardEulermethodintimeanda
Galerkinfiniteelementmethodinspaceforthespacefractionaldiffusionproblemontheunit
interval Ω = (0, 1)
u −CDβ u + qu = f inΩ × (0, T],
t 0 x
with the initial condition u = v and the Dirichlet boundary condition
u(0, t) = g(t) and u(1, t) = h(t), t > 0.
The Galerkin finite element method relies on the variational formulation for the fractional
elliptic problem
−CDβu + qu = f inΩ,
0 x
with a homogeneous Dirichlet boundary condition u(0) = u(1) = 0, recently developed in
[45].Thevariationalformulationoftheproblemisgivenby:findu ∈ U ≡ Hβ2(Ω)suchthat
0
− (RDβ 2u, RDβ 2v ) + (qu, v) = (f, v) ∀v ∈ V,
0 x x 1
whereRDγvandRDγvaretheleft-sidedandright-sidedRiemann–Liouvillederivativeoforder
0 x x 1
γ ∈ (0, 1) defined by (with c = 1 Γ(1 − γ))
γ
d x d 1
RDγv(x) = c ∫ (x − s)−γv(s)ds and RDγv(x) = −c ∫ (x − s)−γv(s)ds,
0 x γ dx 0 x 1 γ dx x
respectively, and the test space V is given by
{ ( ) }
V = v ∈ Hβ 2(Ω): v(1) = 0, x1−β, v = 0 .
For the finite element discretization we first divide the unit interval Ω into a uniform mesh,
withthegridpointsx = ih,i=0,…,Nandmeshsizeh = 1 N.ThenforU ⊂ U wetakethe
i h
continuous piecewise linear finite element space, and for V ⊂ V we construct it from U .
h h
Specifically, with the finite element basis ϕ(x), i = 1,…, N − 1, we take
i
ϕ˜ (x) = ϕ(x) − γ(1 − x) ∈ V,wheretheconstantγ isdeterminedbytheintegralcondition
i i i h i
(x1−β, ϕ˜) = 0, i.e., γ = h2−β((i − 1)3−β + (i + 1)3−β − 2i3−β). The computation of the
i i
leading term in the stiffness matrix and mass matrix can be carried out analytically, and the
partinvolvingthepotentialqcanbecomputedefficientlyusingquadraturerules;see[46]for
details.
Nowforthetime-dependentproblem,likebefore,wedividethetimeinterval[0, T]into
uniformsubintervals,witht = kτ,k=0,…,K,andthetimestepsizeτ = T K.Thenwiththe
k
backward Euler method in time, and the finite element method in space, the approximate
solution uk at time t can be split into uk = u˜k + sk with the particular solution
h k h h
sk = g(t ) + (h(t ) − g(t ))x with the homogeneous solutionu˜k ∈ U satisfying
k k k h h
τ−1 ( u˜k, v ) − (RDβ 2u˜k, RDβ 2v ) + ( qu˜k, v )
h h 0 x h x 1 h h h
= ( fk, v ) + τ−1 ( uk−1− sk, v ) − ( qsk, v ) ∀v ∈ V.
h h h h h h
35

InverseProblems31(2015)035003 BJinandWRundell
WenotetheresultinglinearsystemisoflowerHessenbergform,duetothenonlocalityofthe
coefficient
fractional derivative operator. However, the matrix does not change during the
time stepping procedure, and thus an LU factorization might be applied to speedup the
computation.
References
[1] AgarwalRP1953Aproposd’unenotedeMPierreHumbertC.R.Acad.Sci.,Paris2362031–2
[2] AleroevTS,KiraneMandMalikSA2013Determinationofasourcetermforatimefractional
diffusion equation with an integral type over-determining condition Electron. J. Differ. Equ.
16270
[3] BalakrishnanAV1960Fractionalpowersofclosedoperatorsandthesemigroupsgeneratedby
| them Pac.J.Math. | 10419–37 |     |     |
| ---------------- | -------- | --- | --- |
[4] BeckJV,BlackwellBandClaireCRStJr1985InverseHeatConduction:Ill-PosedProblems
| (NewYork: | Wiley) |     |     |
| --------- | ------ | --- | --- |
[5] BerkowitzB,CortisA,DentzMandScherH2006Modelingnon-Fickiantransportingeological
| formations | as acontinuous timerandom | walk Rev.Geophys.4449 |     |
| ---------- | ------------------------- | --------------------- | --- |
[6] BlumenthalRMandGetoorRK1959Theasymptoticdistributionoftheeigenvaluesforaclass
| of Markovoperators | Pac.J. Math.9 | 399–408 |     |
| ------------------ | ------------- | ------- | --- |
[7] Cannon J R 1968 Determination of an unknown heat source from overspecified boundary data
| SIAM J.Numer. | Anal.5 275–86 |     |     |
| ------------- | ------------- | --- | --- |
[8] Cannon J R1984TheOne-Dimensional Heat Equation(Reading, MA:Addison-Wesley)
[9] CannonJRandDuChateauP1998Structuralidentificationofanunknownsourceterminaheat
| equation | InverseProblems 14535–51 |     |     |
| -------- | ------------------------ | --- | --- |
[10] Caputo M 1967 Linear models of dissipation whose Q is almost frequency independent—II
Geophys. J.Int.13529–39
[11] Carasso A 1982 Determining surface temperatures from interior observations SIAM J. Appl.
Math.42558–74
[12] ChadanK,ColtonD,PäivärintaLandRundellW1997AnIntroductiontoInverseScatteringand
| InverseSpectral | Problems (Philadelphia: | SIAM) |     |
| --------------- | ----------------------- | ----- | --- |
[13] Cheng J, Lin C-L and Nakamura G 2013 Unique continuation property for the anomalous
| diffusion | anditsapplication J. Differ.Equ.2543715–28 |     |     |
| --------- | ------------------------------------------ | --- | --- |
[14] ChengJ,NakagawaJ,YamamotoMandYamazakiT2009Uniquenessinaninverseproblemfor
| aone-dimensional | fractional diffusion | equation InverseProblems | 25115002 |
| ---------------- | -------------------- | ------------------------ | -------- |
ChoulliMandYamamotoM1996Genericwell-posednessofaninverseparabolicproblem—the
[15]
12195–205
| Hölder-space | approach InverseProblems |     |     |
| ------------ | ------------------------ | --- | --- |
[16] ChoulliMandYamamotoM1997Aninverseparabolicproblemwithnon-zeroinitialcondition
1319–27
InverseProblems
[17] Clarke D D, Meerschaert M M and Wheatcraft S W 2005 Fractal travel time estimates for
3 401–7
| dispersive | contaminants Groundwater |     |     |
| ---------- | ------------------------ | --- | --- |
[18] Cussler E L 1997 Diffusion: Mass Transfer in Fluid Systems 2nd edn (New York: Cambridge
| University | Press) |     |     |
| ---------- | ------ | --- | --- |
[19] DjrbashianM1993HarmonicAnalysisandBoundaryValueProblemsintheComplexDomain
(Basel: Birkhäuser)
[20] DjrbashianMM1989Differentialoperatorsoffractionalorderandboundaryvalueproblemsin
the complex domain The Gohberg Anniversary Collection, Operator Theory: Advances and
pp153–72
| Applications | vol 41(Berlin: Springer-Verlag) |     |     |
| ------------ | ------------------------------- | --- | --- |
[21] Dzharbashyan M M 1966 Integral Transformations and Representation of Functions in a
| Complex | Domain (Moscow: Nauka) | (in Russian) |     |
| ------- | ---------------------- | ------------ | --- |
DžrbašjanMM1970AboundaryvalueproblemforaSturm–Liouvilletypedifferentialoperator
[22]
Mat.5 71–96
| of fractional | orderIzv. Akad.NaukArm. | SSRSer. |     |
| ------------- | ----------------------- | ------- | --- |
[23] Einstein A 1905 Über die von der molekularkinetischen Theorie der Wärme geforderte
Ann.Phys.322549–60
| Bewegung | vonin ruhenden Flüssigkeiten | suspendierten | Teilchen. |
| -------- | ---------------------------- | ------------- | --------- |
[24] EldénL1995Numericalsolutionofthesidewaysheatequationbydifferenceapproximationin
11913–23
| timeInverse | Problems |     |     |
| ----------- | -------- | --- | --- |
36

InverseProblems31(2015)035003 BJinandWRundell
[25] EldénL,BerntssonFandRegińskaT2000Waveletandfouriermethodsforsolvingthesideways
heat equation SIAM J.Sci.Comput. 212187–205
[26] Engl H W, Hanke M and Neubauer A 1996 Regularization of Inverse Problems (Dordrecht:
Kluwer)
[27] GarrappaRandPopolizioM2013EvaluationofgeneralizedMittag-Lefflerfunctionsonthereal
line Adv.Comput.Math. 39205–25
[28] GorenfloR,LoutchkoJandLuchkoY2002ComputationoftheMittag-LefflerfunctionE (z)
α,β
anditsderivative Fract.Calc. Appl. Anal. 5 491–518
[29] Gorenflo R, Loutchko J and Luchko Y 2002 Correction: computation of the Mittag-Leffler
function E (z) anditsderivative Fract. Calc. Appl.Anal. 5 491–518
α,β
Gorenflo R,Loutchko J andLuchkoY 2003Fract. Calc.Appl. Anal. 6 111–2
[30] GorenfloR,LuchkoYandMainardiF1999AnalyticalpropertiesandapplicationsoftheWright
function Fract. Calc. Appl.Anal. 2 383–414
[31] HadamardJ1923LecturesonCauchyʼsProbleminLinearPartialDifferentialEquations(New
Haven,CT: Yale University Press)
[32] Hansen PC1998Rank-Deficient andDiscrete Ill-posed Problems (Philadelphia: SIAM)
[33] Hào D N and Reinhardt H-J 1997 On a sideways parabolic equation Inverse Problems 13
297–309
[34] HatanoYandHatanoN1998Dispersivetransportofionsincolumnexperiments:anexplanation
of long-tailed profiles Water Resour.Res. 341027–33
[35] Hatano Y, Nakagawa J, Wang S and Yamamoto M 2013 Determination of order in fractional
diffusion equation J.Math. Ind.A 551–57
[36] HumbertP1953QuelquesrésultatsrelatifsàlafonctiondeMittag-LefflerC.R.Acad.Sci.,Paris
2361467–8
[37] ImanuvilovOYandYamamotoM1998Lipschitzstabilityininverseparabolicproblemsbythe
Carleman estimateInverse Problems 141229–45
[38] IsakovV1991InverseparabolicproblemswiththefinaloverdeterminationCommun.PureAppl.
Math.44185–209
[39] IsakovV 1999Someinverse problems for the diffusion equation InverseProblems 153–10
[40] Isakov V 2006 Inverse Problems for Partial Differential Equations 2nd edn (New York:
Springer)
[41] Ito K and Jin B 2014 Inverse Problems: Tikhonov Theory and Algorithms (Singapore: World
Scientific)
[42] Ito K, Jin B and Takeuchi T 2014 Legendre tau method for fractional elliptic problems with a
Caputoderivative (unpublished)
[43] Ito K and Kappel F 2002 Evolutions Equations and Approximations (Singapore: World
Scientific)
[44] JinB,LazarovR,LiuYandZhouZ2015TheGalerkinfiniteelementmethodforamulti-term
time-fractional diffusion equation J.Comput. Phys.281825–43
[45] JinB,LazarovR,PasciakJandRundellW2013Variationalformulationofproblemsinvolving
fractional orderdifferential operators Math.Comput.at press (arXiv:1307.4795)
[46] Jin B, Lazarov R, Pasciak J and Rundell W 2014 A finite element method for the fractional
Sturm–Liouville problem arXiv:1307.5114
[47] JinB,LazarovRandZhouZ2013Errorestimatesforasemidiscretefiniteelementmethodfor
fractional orderparabolic equations SIAMJ. Numer.Anal. 51445–66
[48] JinB,LazarovRandZhouZ2014Ontwoschemesforfractionaldiffusionanddiffusion-wave
equations arXiv:1404.3800
[49] Jin B and Maass P 2012 Sparsity regularization for parameter identification problems Inverse
Problems 28123001
[50] Jin BandRundellW2012Aninverse problemfor aone-dimensional time-fractional diffusion
problem InverseProblems 28075010
[51] Jin B and Rundell W 2012 An inverse Sturm–Liouville problem with a fractional derivative
J.Comput. Phys.2314954–66
[52] Jones B F Jr 1962 The determination of a coefficient in a parabolic differential equation: I.
Existence anduniqueness J.Math. Mech. 11907–18
[53] Kilbas A, Srivastava H and Trujillo J 2006 Theory and Applications of Fractional Differential
Equations(Amsterdam: Elsevier)
37

InverseProblems31(2015)035003 BJinandWRundell
[54] Kochubeı˘ A N 1990 Diffusion of fractional order Differentsial’ nye Uravneniya 26 660–70
733–734
[55] KwaśnickiM2012EigenvaluesofthefractionalLaplaceoperatorintheintervalJ.Funct.Anal.
2622379–402
[56] LammPK2000Asurveyofregularizationmethodsforkirst-kindVolterraequationsSurveyson
Solution Methods for Inverse Problems ed D Colton, H W Engl, A K Louis,
| J RMcLaughlinand | WRundell(Berlin: | Springer)pp53–82 |     |
| ---------------- | ---------------- | ---------------- | --- |
[57] Lattès R and Lions J-L 1969 The Method of Quasi-Reversibility. Applications to Partial
| Differential | Equations(New | York:Elsevier) |     |
| ------------ | ------------- | -------------- | --- |
[58] Li G,Zhang D,Jia X andYamamoto M 2013 Simultaneous inversion for the space-dependent
diffusion coefficient and the fractional order in the time-fractional diffusion equation Inverse
Problems 29065014
[59] LiZ,ImanuvilovOYandYamamotoM2014Uniquenessininverseboundaryvalueproblems
| for fractional | diffusion equations | arXiv:1404.7024 |     |
| -------------- | ------------------- | --------------- | --- |
[60] LiZandYamamotoM2013Initial-boundaryvalueproblemsforlineardiffusionequationwith
| multiple | time-fractional derivatives | arXiv:1306.2778 |     |
| -------- | --------------------------- | --------------- | --- |
[61] Li Z and Yamamoto M 2014 Uniqueness for inverse problems of determining orders of multi-
term time-fractional derivatives of diffusion equation Appl. Anal. at press doi:10.1080/
00036811.2014.926335
[62] Lin C-L and Nakamura G 2013 Carleman estimate and its application for anomalous slow
| diffusion | equation arXiv:1312.7639 |     |     |
| --------- | ------------------------ | --- | --- |
[63] LinYandXuC2007Finitedifference/spectralapproximationsforthetime-fractionaldiffusion
| equation | J. Comput.Phys.2251533–52 |     |     |
| -------- | ------------------------- | --- | --- |
[64] Liu J 1996 A stability analysis on Beckʼs procedure for inverse heat conduction problems
| J.Comput. | Phys.12365–73 |     |     |
| --------- | ------------- | --- | --- |
[65] Liu J J andYamamoto M 2010A backward problem for the time-fractional diffusion equation
| Appl. Anal. | 891769–88 |     |     |
| ----------- | --------- | --- | --- |
[66] LuchkoY2000AsymptoticsofzerosoftheWrightfunctionZ.Anal.Anwendungen19583–95
[67] LuchkoY2008AlgorithmsforevaluationoftheWrightfunctionfortherealarguments’values
| Fract. Calc. | Appl.Anal. 1157–75 |     |     |
| ------------ | ------------------ | --- | --- |
[68] Luchko Y, Rundell W, Yamamoto M and Zuo L 2013 Uniqueness and reconstruction of an
unknownsemilinearterminatime-fractionalreaction-diffusionequationInverseProblems29
065019
[69] Mainardi F 1996 The fundamental solutions for the fractional diffusion-wave equation Appl.
| Math.Lett. | 9 23–28 |     |     |
| ---------- | ------- | --- | --- |
[70] Mainardi F 2010 Fractional Calculus and Waves in Linear Viscoelasticity: An Introduction to
| Mathematical | Models (Singapore:World | Scientific) |     |
| ------------ | ----------------------- | ----------- | --- |
[71] Malamud M M 1994 Similarity of Volterra operators and related problems in the theory of
| differential | equations of fractional | orders Tr. Mosk.Mat. | Obshch.55365 |
| ------------ | ----------------------- | -------------------- | ------------ |
[72] Metzler R and Klafter J 2000 The random walkʼs guide to anomalous diffusion: a fractional
| dynamics | approach Phys.Rep.3391–77 |     |     |
| -------- | ------------------------- | --- | --- |
[73] MillerLandYamamotoM2013Coefficientinverseproblemforafractionaldiffusionequation
| InverseProblems | 29075013 |     |     |
| --------------- | -------- | --- | --- |
[74] Mittag-Leffler G M1903Sur lanouvelle function E C.R. Acad.Sci., Paris 137554–8
a
[75] Montroll EWandWeiss GH 1965Randomwalks onlattices: II.J. Math.Phys.6 167–81
[76] Murio D A 2007 Stable numerical solution of a fractional-diffusion inverse heat conduction
| problem | Comput.Math. Appl. | 531492–501 |     |
| ------- | ------------------ | ---------- | --- |
[77] MurioDA2008TimefractionalIHCPwithCaputofractionalderivativesComput.Math.Appl.
562371–81
[78] Nahušev A M 1977 The Sturm–Liouville problem for a second order ordinary differential
equation with fractional derivatives in the lowerterms Dokl. Akad.Nauk SSSR234308–11
[79] Nakagawa J, Sakamoto K and Yamamoto M 2010 Overview to mathematical analysis for
fractionaldiffusionequations—newmathematicalaspectsmotivatedbyindustrialcollaboration
| J.Math. | Ind.2A 99–108 |     |     |
| ------- | ------------- | --- | --- |
[80] Paris R B 2002 Exponential asymptotics of the Mittag-Leffler function Proc. R. Soc. A 458
3041–52
[81] PazyA1992SemigroupsofLinearOperatorsandApplicationstoPartialDifferentialEquations
(Berlin: Springer)
38

| InverseProblems31(2015)035003 |     |     |     |     |     |     |     |     | BJinandWRundell |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --------------- |
[82] PollardH1948ThecompletelymonotoniccharacteroftheMittag-LefflerfunctionE (−x)Bull.
a
|     | Am.Math. | Soc.541115–6 |     |     |     |     |     |     |     |
| --- | -------- | ------------ | --- | --- | --- | --- | --- | --- | --- |
[83] PopovAYandSedletskiı˘AM2011DistributionofrootsofMittag-LefflerfunctionsSovrem.
|     | Mat.Fundam. | Napravl. |     | 403–171 |     |     |     |     |     |
| --- | ----------- | -------- | --- | ------- | --- | --- | --- | --- | --- |
[84] Prilepko A I, Orlovsky D G and Vasin I A 2000 Methods for Solving Inverse Problems in
|     | Mathematical | Physics(NewYork: |     |     | Dekker) |     |     |     |     |
| --- | ------------ | ---------------- | --- | --- | ------- | --- | --- | --- | --- |
[85] QianZ2010Optimalmodifiedmethodforafractional-diffusioninverseheatconductionproblem
|     | InverseProbl. | Sci.Eng.18521–33 |     |     |     |     |     |     |     |
| --- | ------------- | ---------------- | --- | --- | --- | --- | --- | --- | --- |
[86] RundellWandSacksPE1992ReconstructiontechniquesforclassicalinverseSturm–Liouville
|     | problems | Math.Comput. |     | 58161–83 |     |     |     |     |     |
| --- | -------- | ------------ | --- | -------- | --- | --- | --- | --- | --- |
[87] Rundell W, Xu X and Zuo L 2013 The determination of an unknown boundary condition in a
|     | fractional | diffusion | equation | Appl. Anal. | 921511–26 |     |     |     |     |
| --- | ---------- | --------- | -------- | ----------- | --------- | --- | --- | --- | --- |
[88] Sakamoto K and Yamamoto M 2011 Initial value/boundary value problems for fractional
diffusion-waveequationsandapplicationstosomeinverseproblemsJ.Math.Anal.Appl.382
426–47
[89] SakamotoKandYamamotoM2011Inversesourceproblemwithafinaloverdeterminationfor
|     | afractional | diffusion | equation | Math. | Control | Relat.Fields | 1 509–18 |     |     |
| --- | ----------- | --------- | -------- | ----- | ------- | ------------ | -------- | --- | --- |
[90] Schneider W R 1996 Completely monotone generalized Mittag-Leffler functions Exposition.
Math.143–16
[91] Schneider W R and Wyss W 1989 Fractional diffusion and wave equations J. Math. Phys. 30
134–44
[92] SchusterT,KaltenbacherB,HofmannBandKazimierskiKS2012RegularizationMethodsin
|     | Banach Spaces(Berlin: |     | Walter | deGruyter) |     |     |     |     |     |
| --- | --------------------- | --- | ------ | ---------- | --- | --- | --- | --- | --- |
[93] Sedletskiı˘ A M 1994 Asymptotic formulas for zeros of functions of Mittag-Leffler type Anal.
Math.20117–32
[94] SeyboldHandHilferR2008NumericalalgorithmforcalculatingthegeneralizedMittag-Leffler
|     | function | SIAM J.Numer. |     | Anal.4769–88 |     |     |     |     |     |
| --- | -------- | ------------- | --- | ------------ | --- | --- | --- | --- | --- |
[95] Sokolov IM, Klafter J andBlumenA 2002Fractional kinetics Phys.Today5548–54
[96] SunZ-ZandWuX2006Afullydiscretedifferenceschemeforadiffusion-wavesystemAppl.
|     | Numer. Math.56193–209 |     |     |     |     |     |     |     |     |
| --- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
[97] Wang H and Wu B 2014 On the well-posedness of determination of two coefficients in a
|     | fractional | integrodifferential |     | equation | Chin. | Ann.Math.Ser. | B35447–68 |     |     |
| --- | ---------- | ------------------- | --- | -------- | ----- | ------------- | --------- | --- | --- |
[98] WangJ-G,ZhouY-BandWeiT2013Tworegularizationmethodstoidentifyaspace-dependent
|     | sourcefor | the time-fractional |     | diffusion | equation | Appl. | Numer.Math. | 6839–57 |     |
| --- | --------- | ------------------- | --- | --------- | -------- | ----- | ----------- | ------- | --- |
[99] Wang L and Liu J 2013 Total variation regularization for a backward time-fractional diffusion
|     | problem | InverseProblems |     | 29115013 |     |     |     |     |     |
| --- | ------- | --------------- | --- | -------- | --- | --- | --- | --- | --- |
[100] WangW,YamamotoMandHanB2013Numericalmethodinreproducingkernelspaceforan
|     | inverse sourceproblem |     | for | the fractional | diffusion | equation | InverseProblems |     | 29095009 |
| --- | --------------------- | --- | --- | -------------- | --------- | -------- | --------------- | --- | -------- |
[101] WeiTandWangJ2014Amodifiedquasi-boundaryvaluemethodforaninversesourceproblem
|     | of the time-fractional |     | diffusionequation |     | Appl.Numer. |     | Math.7895–111 |     |     |
| --- | ---------------------- | --- | ----------------- | --- | ----------- | --- | ------------- | --- | --- |
[102] Wei T and Wang J-G 2014 A modified quasi-boundary value method for the backward time-
|     | fractional | diffusion | problem | ESAIM | Math.Modelling |     | Numer. Anal. | 48603–21 |     |
| --- | ---------- | --------- | ------- | ----- | -------------- | --- | ------------ | -------- | --- |
[103] WeidemanJACandTrefethenLN2007Parabolicandhyperboliccontoursforcomputingthe
|       | Bromwich | integral    | Math.Comput. | 761341–56 |               |       |                   |     |     |
| ----- | -------- | ----------- | ------------ | --------- | ------------- | ----- | ----------------- | --- | --- |
| [104] | Wiman A  | 1905Überdie | Nullstellen  |           | derFunktionen | Ea(x) | ActaMath.29217–34 |     |     |
[105] Wong R and Zhao Y-Q 2002 Exponential asymptotics of the Mittag-Leffler function Constr.
Approx.18355–85
[106] WrightEM1933OnthecoefficientsofpowerserieshavingexponentialsingularitiesJ.London
|     | Math.Soc. | 8 71–79 |     |     |     |     |     |     |     |
| --- | --------- | ------- | --- | --- | --- | --- | --- | --- | --- |
[107] WrightEM1940ThegeneralizedBesselfunctionofordergreaterthanoneQ.J.Math.1136–48
[108] Xu X, Cheng J and Yamamoto M 2011 Carleman estimate for a fractional diffusion equation
|     | with half | orderandapplication |     | Appl. | Anal. | 901355–71 |     |     |     |
| --- | --------- | ------------------- | --- | ----- | ----- | --------- | --- | --- | --- |
[109] YamamotoMandZhangY2012Conditionalstabilityindeterminingazeroth-ordercoefficient
in a half-order fractional diffusion equation by a Carleman estimate Inverse Problems 28
105010
[110] ZacherR2013AweakHarnackinequalityforfractionalevolutionequationswithdiscontinuous
|     | coefficients | Ann.Sc.Norm. |     | Super.PisaCl. |     | Sci.(5)12903–40 |     |     |     |
| --- | ------------ | ------------ | --- | ------------- | --- | --------------- | --- | --- | --- |
39

InverseProblems31(2015)035003 BJinandWRundell
[111] Zhang Y and Xu X 2011 Inverse source problem for a fractional diffusion equation Inverse
Problems 27035010
[112] ZhangZQandWeiT2013Identifyinganunknownsourceintime-fractionaldiffusionequation
| byatruncation | method Appl.Math. | Comput.2195972–83 |     |
| ------------- | ----------------- | ----------------- | --- |
[113] Zheng G H and Wei T 2011 A new regularization method for the time fractional inverse
| advection-dispersion | problemSIAM | J.Numer. Anal. | 491972–90 |
| -------------------- | ----------- | -------------- | --------- |
[114] Zheng G H and Wei T 2012 A new regularization method for a Cauchy problem of the time
| fractional diffusion | equation Adv.Comput. | Math.36377–98 |     |
| -------------------- | -------------------- | ------------- | --- |
40