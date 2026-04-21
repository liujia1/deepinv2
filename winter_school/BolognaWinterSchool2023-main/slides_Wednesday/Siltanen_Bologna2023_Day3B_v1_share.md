# Siltanen_Bologna2023_Day3B_v1_share.pdf

## 第1页

Emission tomography
Samuli Siltanen
PhD Winter School 2023
Advanced methods for mathematical image analysis
Bologna, Italy
January 25, 2023
Instagram:
@samuntiede
@monday_spider
YouTube:
@professor_sam
@Samuntiedekanava


---

## 第2页

Outline
Medical positron emission tomography (PET)
Nuclear power 101
Passive Gamma Emission Tomography (PGET)


---

## 第3页

Fluorodeoxyglucose molecule
Fluorine-18 will decay (β+) into stable oxygen-18 with a half-life of
about two hours.


---

## 第4页

Recall the structure of (positive and negative)
radioactive β-decay
In β−decay, a neutron in an
atomic nucleus transforms into
a triplet of a proton, an elec-
tron and a neutrino:
n →



p+
e−
νe
After the event, the nucleus
has one more proton than be-
fore and so becomes another
chemical element.
It moves
one position up in the periodic
table of elements.
In
β+
decay,
a
proton
in
an atomic nucleus transforms
into a triplet of a neutron, a
positron (=anti-electron) and
a antineutrino:
p+ →



n
e+
νe
After the event, the nucleus
has one less proton than before
and so it moves one position
down in the periodic table of
elements.


---

## 第7页

PET images show where nutrients are absorbed in
the tissue, for example in the brain
Early Alzheimer’s
Late Alzheimer’s


---

## 第8页

Outline
Medical positron emission tomography (PET)
Nuclear power 101
Passive Gamma Emission Tomography (PGET)


---

## 第24页

Wait for it...


---

## 第26页

A nuclear power plant is a giant hot water kettle


---

## 第27页

Fuel is placed inside reactor core


---

## 第28页

Energy comes from splitting uranium-235 nuclei
Krypton-92 will decay (β−)
into rubidium-92 with a half-
life of 1.84 seconds.
Rubidium-92 will decay (β−)
into strontium-92 with half-life
of 4.5 seconds, which in turn
becomes yttrium 92 with half-
life of about 3 hours.
The next β−decay has half-life
of three and a half hours, pro-
ducing stable zirkonium-92.


---

## 第29页

Energy comes from splitting uranium-235 nuclei
Barium-141 will decay (β−)
into
lanthanum-141
with
a
half-life of 18 minutes.
Lanthanum-141
will
de-
cay
(β−)
into
cerium-141
with
half-life
of
about
4
hours.
Cerium-141 becomes
promethium-141 via β−decay
with a half-life of a month.
Promethium-141
under-
goes
β+
decay
into
stable
neudymium-142 with a half-
life of 21 minutes.


---

## 第30页

Carbon emissions of energy production (median)
Method
CO2-gram/kWh
Coal
820
Gas
490
Biomass
230
Solar
41
Geothermal
38
Hydropower
24
Nuclear
12
Wind
11
Source: IPCC; see page 7 in the document
https://www.ipcc.ch/site/assets/uploads/2018/02/ipcc_wg3_ar5_annex-iii.pdf


---

## 第31页

Outline
Medical positron emission tomography (PET)
Nuclear power 101
Passive Gamma Emission Tomography (PGET)


---

## 第32页

Let’s start the PGET story with a video


---

## 第33页

A nuclear fuel assembly consists of rods ﬁlled with
pellets containing uranium-235 and uranium-238


---

## 第34页

Measurement with the PGET device at a nuclear
power plant


---

## 第35页

PGET measurement device
▶Passive Gamma Emission
Tomography
▶Similar idea as in medical SPECT
▶PGET strength: ability to image
activity of single fuel pins
▶IAEA started development in the
80’s, approved for inspections in
2017
▶Only one device exists at the
moment, two more are being built
https://ideas.unite.un.org/iaea-
tomography/Page/Home
.


---

## 第36页

Measurement geometry of the PGET device


---

## 第37页

Data is collected by rotating the system
around the fuel assembly


---

## 第38页

Data is collected by rotating the system
around the fuel assembly


---

## 第39页

Data is collected by rotating the system
around the fuel assembly


---

## 第40页

PGET sinograms


---

## 第41页

Forward model
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
10
9
8
7
6


---

## 第42页

Forward model
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
10
9
8
7
6


---

## 第43页

Reconstruction as a minimization problem
Reconstruction images are obtained by solving
min(λ,µ)
(
∥F (λ, µ) −m∥2
2 +
X
i
αiPi (λ, µ)
)
▶λ is the emission image, µ is the attenuation image.
▶Data ﬁt term ∥F (λ, µ) −m∥2
2 measures how well the forward
projection F (λ, µ) matches the measurement m.
▶The regularization terms P
i αiPi (λ, µ) incorporate prior
knowledge into the reconstruction process.


---

## 第44页

How We Won Silver in IAEA PGET Challenge


---

## 第45页

These are the mock-up fuel assemblies


---

## 第46页

The cobolt “fuel rods” were activated in a reactor


---

## 第47页

Reconstruction by Filtered Back-Projection


---

## 第48页

Classiﬁcation by Filtered Back-Projection
Ground truth
Classiﬁcation


---

## 第49页

Reconstruction as a minimization problem 1/2
Reconstruction images are obtained by solving
min(λ,µ)
(
∥F (λ, µ) −m∥2
2 +
X
i
αiPi (λ, µ)
)
▶λ is the emission image, µ is the attenuation image.
▶Data ﬁt term ∥F (λ, µ) −m∥2
2 measures how well the forward
projection F (λ, µ) matches the measurement m.
▶The regularization terms P
i αiPi (λ, µ) incorporate prior
knowledge into the reconstruction process.


---

## 第50页

Reconstruction as a minimization problem 2/2
We write the Tikhonov regularization task as nonlinear least
squares problem in stacked form
arg min
λ,µ

F(λ, µ) −m
√αλMλλ
√αµMµµ

2
2
,
where the matrices Mλ and Mµ incorporate the a priori information
into the reconstruction process. We solve the problem using the
Levenberg-Marquardt method.


---

## 第51页

One piece of a priori information we put into the
reconstruction is the physicality of materials
Need to set bounds for the emission and attenuation values in the
minimization problem to produce reasonable images.
▶Excludes the possibility of a
material with high emission
but low attenuation value.
▶Some way of estimating
these bounds before the
minimization is needed.
Emission λ (a.u.)
Attenuation µ (mm-1)


---

## 第52页

Reconstruction by geometry-aware prior


---

## 第53页

Classiﬁcation by geometry-aware prior
Ground truth
Classiﬁcation
[Backholm, Bubba, Bélanger-Champagne, Helin, Dendooven & S 2020]


---

## 第54页

Many fuel rods are currently imaged in Finnish
nuclear power plants. Here VVER-440 assembly
Emission
Attenuation
Classiﬁcation
Calculations and images:
DI Riina Virta


---

## 第55页

Thank you for your attention!
←−Slime mold called Lycogala conicum


---

## 第56页

Thank you for your attention!


---

