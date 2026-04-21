# Siltanen_Bologna2023_Day3A_v2_share.pdf

## 第1页

Limited-angle tomography
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
Where do we encounter limited-angle tomography?
Industrial case study: low-dose 3D dental X-ray imaging
Matrix-based limited angle tomography and SVD
Ill-posedness of limited-angle Radon transform
Recent progress in limited-angle tomography
Learning the unknown WF set with shearlets
Learning the unknown WF set with complex wavelets
Estimating the unknown WF set with computational topology
Helsinki tomography challenge 2022


---

## 第3页

In 3D mammography, the imaging geometry
restricts the angular range of the data


---

## 第4页

Paralle slices in the reconstruction can be
improved with Bayesian inversion
Tomosynthesis
MAP estimate with Besov prior,
p = 1.5 = q and s = 0.5
[Rantala et al. (2006), US patent 7215730]


---

## 第5页

This part is a joint work with
Alexander Meaney, University of Helsinki, Finland
Esa Niemi, Eniram Ltd., Finland
Aaro Salosensaari, University of Helsinki, Finland
Industrial partners:
Kemppi Ltd. (welding tool manufacturer)
Ajat Ltd. (X-ray detector manufacturer)


---

## 第6页

Two steel pipes partly welded together


---

## 第12页

This is the limited-angle measurement geometry
for a narrow CaTd direct conversion detector


---

## 第13页

Traditional reconstruction by tomosynthesis
Simulated phantom:
Tomosynthesis:


---

## 第14页

TVR-DART with domain restriction
Simulated phantom:
TVR-DART:


---

## 第15页

Reconstructions from measured data
Tomosynthesis
TVR-DART
[Niemi, Salosensaari, Meaney & S, submitted manuscript]


---

## 第16页

Tomography appears in adaptive optics
▶Modern telescope
imaging suﬀers from
turbulence in the
atmosphere
⇒blurring of images
▶Adaptive optics corrects
the perturbed incoming
light in real-time
▶Major challenge in
wide-ﬁeld AO:
atmospheric tomography
European Extremely Large Telescope (2024)
Helin, Kindermann, Lehtonen & Ramlau 2018
Yudytskiy, Helin & Ramlau 2014


---

## 第17页

Photograph of planet Neptune with and without
adaptive optics (image: ESO/P. Weilbacher)
No adaptive optics
Adaptive optics


---

## 第18页

Outline
Where do we encounter limited-angle tomography?
Industrial case study: low-dose 3D dental X-ray imaging
Matrix-based limited angle tomography and SVD
Ill-posedness of limited-angle Radon transform
Recent progress in limited-angle tomography
Learning the unknown WF set with shearlets
Learning the unknown WF set with complex wavelets
Estimating the unknown WF set with computational topology
Helsinki tomography challenge 2022


---

## 第19页

Application: dental implant planning, where a
missing tooth is replaced with an implant


---

## 第20页

This is the classical imaging procedure
of the panoramic X-ray device


---

## 第21页

The resulting image shows a sharp layer
positioned inside the dental arc


---

## 第22页

Nowadays, a digital panoramic imaging device is
standard equipment at dental clinics
A panoramic dental image oﬀers a
general overview showing all teeth
and other structures simultaneously.
Panoramic images are not suitable
for dental implant planning because
of unavoidable geometric distortion.
•
X-ray source
Narrow detector


---

## 第23页

We reprogram the panoramic X-ray device so that
it collects projection data by scanning


---

## 第24页

We reprogram the panoramic X-ray device so that
it collects projection data by scanning
Number of projection images: 11
Angle of view: 40 degrees
Image size: 1000×1000 pixels
The
unknown
vector
f
has
7 000 000 elements.


---

## 第25页

Standard Cone Beam CT reconstruction delivers
100 times more radiation than VT imaging
Cone Beam CT
VT imaging
Kolehmainen, Vanne, S, Järvenpää, Kaipio,
Lassas & Kalke 2006
Kolehmainen, Lassas & S 2008
Cederlund, Kalke & Welander 2009
Hyvönen, Kalke, Lassas, Setälä & S 2010
U.S. patent 7269241, thousands of VT units in use


---

## 第26页

The VT device was developed in 2001–2012 by
Nuutti Hyvönen
Seppo Järvenpää
Jari Kaipio
Martti Kalke
Petri Koistinen
Ville Kolehmainen
Matti Lassas
Jan Moberg
Kati Niinimäki
Juha Pirttilä
Maaria Rantala
Eero Saksman
Henri Setälä
Erkki Somersalo
Antti Vanne
Simopekka Vänskä
Richard L. Webber


---

## 第27页

Outline
Where do we encounter limited-angle tomography?
Industrial case study: low-dose 3D dental X-ray imaging
Matrix-based limited angle tomography and SVD
Ill-posedness of limited-angle Radon transform
Recent progress in limited-angle tomography
Learning the unknown WF set with shearlets
Learning the unknown WF set with complex wavelets
Estimating the unknown WF set with computational topology
Helsinki tomography challenge 2022


---

## 第28页

Discretize the unknown by dividing it into pixels
Target (unknown)
32×32 pixel grid


---

## 第29页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第30页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第31页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第32页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第33页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第34页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第35页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第36页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第37页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第38页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第39页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第40页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第41页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第42页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第43页

Construction of limited-angle sinogram
0◦
90◦
180◦


---

## 第44页

SVD reveals the ill-posedness of the limited-angle
problem, see Davison 1983 and Louis 1986
1
200
400
600 735
10-15
10-10
10-5
100
Singular values of A
(diagonal of D)
735×1024 system matrix A,
only nonzero elements shown


---

## 第45页

Filtered Back-Projection (FBP) reconstruction
from limited-angle data
Original phantom sampled at
32×32 resolution
Filtered back-projection


---

## 第46页

Non-negative Tikhonov regularization
arg min
f ∈Rn
+

∥Af −m∥2
2 + α∥f ∥2
2
	
Original phantom sampled at
32×32 resolution
Tikhonov regularized reconstruction


---

## 第47页

Non-negative limited-angle TV regularization
arg min
f ∈Rn
+

∥Af −m∥2
2 + α∥∇f ∥1
	
Original phantom sampled at
32×32 resolution
TV reconstruction


---

## 第48页

Outline
Where do we encounter limited-angle tomography?
Industrial case study: low-dose 3D dental X-ray imaging
Matrix-based limited angle tomography and SVD
Ill-posedness of limited-angle Radon transform
Recent progress in limited-angle tomography
Learning the unknown WF set with shearlets
Learning the unknown WF set with complex wavelets
Estimating the unknown WF set with computational topology
Helsinki tomography challenge 2022


---

## 第50页

Deﬁnition of the Radon transform
Let f (x) = f (x1, x2) be the X-ray attenuation coeﬃcient. The
classical model for tomographic data is the Radon transform
Rf (θ, s) =
Z
x·⃗θ=s
f (x)dx =
Z
τ∈R
f (s⃗θ+τ⃗θ⊥)dτ,
⃗θ ∈S1, s ∈R,
where S1 is the unit circle, ⃗θ⊥is a unit vector perpendicular to the
unit vector ⃗θ = (cos θ, sin θ), and x · ⃗θ denotes vector inner product.
Note that f is deﬁned on R2 and Rf is deﬁned on S1×R1.


---

## 第51页

The Fourier slice theorem
Let f : R2 →R be smooth and compactly supported. Denote
Rθf (s) := Rf (θ, s) for ⃗θ ∈S1 and s ∈R. Then
d
Rθf (ξ) = bf (ξ⃗θ).
Proof. The change of coordinates x = s⃗θ + τ⃗θ⊥gives s = ⃗θ · x
and dx = dτ ds. Calculate
d
Rθf (ξ)
=
Z ∞
−∞
e−iξsRθf (s) ds
=
Z ∞
−∞
e−iξs
Z ∞
−∞
f (s⃗θ + τ⃗θ⊥)dτ ds
=
Z
R2 e−iξ⃗θ·xf (x)dx
=
bf (ξθ).
□


---

## 第52页

Practically Dubious Theorem:
Unique determination from limited-angle data
Let f : R2 →R be smooth and compactly supported. Let ε > 0.
Then f is uniquely determined from the limited-angle sinogram
Rf (θ, s)
with −ε < θ < ε and s ∈R.
Proof. Let f and g be compactly supported smooth functions
deﬁned on R2 satisfying Rf (θ, s) = Rg(θ, s) for −ε < θ < ε.
By the Fourier slice theorem we know that
bf (ξ⃗θ) ≡bg(ξ⃗θ)
in the open set Cε := {(ξ, θ) ∈R2 | ξ > 0, −ε < θ < ε}. The
Fourier transform of a compactly supported smooth function is
analytic. So bf = bg on the open set Cε, and due to analyticity
bf = bg on the whole frequency domain R2. Therefore f ≡g. □


---

## 第53页

Limited angle measurement information looks like
a bowtie in the frequency domain
Range of measured angles is (−ε, ε)
Frequency domain
ε
ε


---

## 第54页

What do we know about the singular values of the
Radon transform?
Roughly speaking,
▶the full-angle Radon transform allows a singular system where
the singular values decay as dn ∼1/n when n →∞;
▶the singular values of limited-angle Radon transform decay
exponentially even if the interval of missing angles is just
(−ε, ε) with any ε > 0.
The details are available in Natterer 1986, Sections IV.3 and VI.2.
For more information, see Davison 1983 and Louis 1986.


---

## 第55页

Limited data gives only part of the wavefront set
Stable part of wavefront set
Unstable part of wavefront set
See [Greenleaf & Uhlmann 1989], [Quinto 1993], and [Frikel & Quinto 2013]


---

## 第56页

Filtered Back-Projection (FBP) reconstruction
from limited-angle data
Stable part of wavefront set
Filtered back-projection


---

## 第57页

Non-negative Tikhonov regularization
arg min
f ∈Rn
+

∥Af −m∥2
2 + α∥f ∥2
2
	
Stable part of wavefront set
Tikhonov regularized reconstruction


---

## 第58页

Constrained total variation (TV) regularization
arg min
f ∈Rn
+

∥Af −m∥2
2 + α∥∇f ∥1
	
Stable part of wavefront set
TV regularized reconstruction


---

## 第59页

Interesting papers that generalize the theme of
detecting singularities
Borg, Frikel, Jørgensen and Quinto:
Analyzing Reconstruction Artifacts from Arbitrary Incomplete X-ray
CT Data.
Link.
Borg, Frikel, Jørgensen and Sporring:
Reduction of variable-truncation artifacts from beam occlusion
during in situ x-ray tomography. Link.


---

## 第60页

Outline
Where do we encounter limited-angle tomography?
Industrial case study: low-dose 3D dental X-ray imaging
Matrix-based limited angle tomography and SVD
Ill-posedness of limited-angle Radon transform
Recent progress in limited-angle tomography
Learning the unknown WF set with shearlets
Learning the unknown WF set with complex wavelets
Estimating the unknown WF set with computational topology
Helsinki tomography challenge 2022


---

## 第61页

Outline
Where do we encounter limited-angle tomography?
Industrial case study: low-dose 3D dental X-ray imaging
Matrix-based limited angle tomography and SVD
Ill-posedness of limited-angle Radon transform
Recent progress in limited-angle tomography
Learning the unknown WF set with shearlets
Learning the unknown WF set with complex wavelets
Estimating the unknown WF set with computational topology
Helsinki tomography challenge 2022


---

## 第62页

Filtered back-projection fails to recover
the invisible parts of boundaries
Ground truth
Filtered back-projection


---

## 第63页

When we learn the invisible parts of boundaries
we can recover them
Ground truth
Invisible parts learned
[Bubba, Kutyniok, Lassas, Maerz, Samek, Siltanen and Srinivasan,
Inverse Problems 2019]


---

## 第64页

Outline
Where do we encounter limited-angle tomography?
Industrial case study: low-dose 3D dental X-ray imaging
Matrix-based limited angle tomography and SVD
Ill-posedness of limited-angle Radon transform
Recent progress in limited-angle tomography
Learning the unknown WF set with shearlets
Learning the unknown WF set with complex wavelets
Estimating the unknown WF set with computational topology
Helsinki tomography challenge 2022


---

## 第65页

Process pipeline invented by Siiri Rautio


---

## 第66页

Machine learning grasps “candy-wrap” geometry
of complex dual-tree wavelet coeﬃcients
[Rautio, Murthy, Bubba, Lassas and S, submitted]


---

## 第67页

Outline
Where do we encounter limited-angle tomography?
Industrial case study: low-dose 3D dental X-ray imaging
Matrix-based limited angle tomography and SVD
Ill-posedness of limited-angle Radon transform
Recent progress in limited-angle tomography
Learning the unknown WF set with shearlets
Learning the unknown WF set with complex wavelets
Estimating the unknown WF set with computational topology
Helsinki tomography challenge 2022


---

## 第68页

Computational homology identiﬁes missing edges
Unpublished work by Elli Karvonen


---

## 第69页

Outline
Where do we encounter limited-angle tomography?
Industrial case study: low-dose 3D dental X-ray imaging
Matrix-based limited angle tomography and SVD
Ill-posedness of limited-angle Radon transform
Recent progress in limited-angle tomography
Learning the unknown WF set with shearlets
Learning the unknown WF set with complex wavelets
Estimating the unknown WF set with computational topology
Helsinki tomography challenge 2022


---

## 第70页

Check it out: https://www.ﬁps.ﬁ/HTC2022.php
Challenge production team: Alexander Meaney, Fernando Moura,
Siiri Rautio, Salla Latva-Äijö, Tommi Heikkilä and S.


---

## 第71页

Limited angle tomography is diﬃcult
Full angle FBP
Limited angle FBP
Limited angle TV
Original
Segmented
See [Greenleaf & Uhlmann 1989], [Quinto 1993], and [Frikel & Quinto 2013]


---

## 第72页

We made several plastic phantoms with
diﬀerently shaped holes in them
Shown are segmented FBP reconstructions from full-angle data


---

## 第73页

We measured full-angle X-ray data of the discs
using our lab in Helsinki
Shown are segmented FBP reconstructions from full-angle data


---

## 第74页

The diﬃculty of reconstruction was raised
by limiting the angle of view step by step
Shown are segmented FBP reconstructions from limited-angle data.
Angular ranges were 90◦→80◦→. . . →30◦


---

## 第76页

Helsinki Tomography Challenge 2022:
phantom diﬃculty groups
Group
Angular range
Angle increment
Number of projections
1
90◦
0.5
181
2
80◦
0.5
161
3
70◦
0.5
141
4
60◦
0.5
121
5
50◦
0.5
101
6
40◦
0.5
81
7
30◦
0.5
61


---

## 第77页

The Helsinki Tomography Challenge 2022
9 particapting teams from 7 countries:
• Austria
• Brazil
• China
• Denmark
• Germany
• India
• Singapore
Altogether 22 diﬀerent algorithms were submitted.


---

## 第78页

The Helsinki Tomography Challenge 2022
The top ranking teams are:
1. Technical University Dortmund, Department of Computer
Science & Heinrich Heine University Düsseldorf, Department
of Computer Science - Germany
2. University of Bremen, Center for Industrial Mathematics
(ZeTeM) - Germany
3. Technical University of Denmark, Department of Applied
Mathematics and Computer Science - Denmark
The full results are viewable at
https://ﬁps.ﬁ/HTC2022_results.pdf.
All the data and participating algorithms are openly available.


---

## 第79页

Scoring the quality of reconstructions
Reconstructed binary image is Ir, ground truth binary image is It.
The score of the reconstruction is given by the Matthews
correlation coeﬃcient (MCC). Deﬁne
TP =
X
i,j
(It ∩Ir)ij,
FP =
X
i,j
(¯It ∩Ir)ij,
FN =
X
i,j
(It ∩¯Ir)ij,
TN =
X
i,j
(¯It ∩¯Ir)ij.
Calculate MCC score S ∈[−1, 1] as
S =
TP × TN −FP × FN
p
(TP + FP)(TP + FN)(TN + FP)(TN + FN)
.
A score of +1 (best) represents a perfect reconstruction, 0 no
better than random reconstruction, and −1 (worst) indicates total
disagreement between reconstruction and ground truth.


---

## 第80页

The scores of the ﬁve best teams in HTC2022


---

## 第81页

Thank you for your attention!
←−Slime mold called Lycogala conicum


---

## 第82页

Thank you for your attention!


---

