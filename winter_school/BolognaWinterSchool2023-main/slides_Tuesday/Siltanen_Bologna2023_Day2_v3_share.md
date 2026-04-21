# Siltanen_Bologna2023_Day2_v3_share.pdf

## 第1页

Tomographic models based on
pixels and matrices
Samuli Siltanen
PhD Winter School 2023
Advanced methods for mathematical image analysis
Bologna, Italy
January 24, 2023
Instagram:
@samuntiede
@monday_spider
YouTube:
@professor_sam
@Samuntiedekanava


---

## 第2页

Wilhelm Conrad Röntgen invented X-rays and was
awarded the ﬁrst Nobel Prize in Physics in 1901


---

## 第3页

But even before Röntgen, Nikola Tesla
had observed X-rays in his own way


---

## 第4页

Outline
Why pixel-based tomographic modelling?
Restricted time →sparse tomography
Restricted radiation dose →sparse tomography
Restricted money →sparse tomography
The Beer-Lambert Law
Pixel-based measurement model
Matrix model for sparse tomography
Transpose of A: backprojection
Ill-posedness of sparse tomography
Total variation regularization
Regularization
Tikhonov regularization
Total variation regularization
Frame-sparsity methods


---

## 第5页

We collected X-ray projection data of a walnut
from 1200 directions
Data collection:
thanks to Keijo
Hämäläinen and Aki Kallonen, Uni-
versity of Helsinki.
The data is openly available at
http://ﬁps.ﬁ/dataset.php, thanks to
Esa Niemi and Antti Kujanpää


---

## 第6页

Reconstructions of a 2D slice through the walnut
using ﬁltered back-projection (FBP)
FBP with comprehensive data
(1200 projections)
FBP with sparse data
(20 projections)


---

## 第7页

Sparse-data reconstruction of the walnut using
non-negative total variation regularization
Filtered back-projection
Constrained TV regularization
arg min
f ∈Rn
+

∥Af −m∥2
2 + α∥∇f ∥1
	


---

## 第8页

Haar wavelet sparsity reconstruction


---

## 第9页

Total Variation (TV) reconstruction


---

## 第10页

Daubechies 2 wavelet sparsity reconstruction


---

## 第11页

Total Generalized Variation (TGV) reconstruction


---

## 第12页

Shearlet sparsity reconstruction


---

## 第14页

Outline
Why pixel-based tomographic modelling?
Restricted time →sparse tomography
Restricted radiation dose →sparse tomography
Restricted money →sparse tomography
The Beer-Lambert Law
Pixel-based measurement model
Matrix model for sparse tomography
Transpose of A: backprojection
Ill-posedness of sparse tomography
Total variation regularization
Regularization
Tikhonov regularization
Total variation regularization
Frame-sparsity methods


---

## 第15页

Image by Bruce Blaus, CC BY-SA 4.0
https://commons.wikimedia.org/w/index.php?curid=44968165


---

## 第16页

We consider small specimens of human bone
imaged using microtomography
Slice of 3D reconstruction by FDK
based on 596 angles
Three-dimensional structure


---

## 第17页

We pick out a smaller region of interest
for osteoarthritis analysis
Slice of 3D reconstruction by FDK
based on 596 angles
Slice of 3D region of interest
after binary thresholding


---

## 第18页

We use two numerical quality measures applied
to segmented three-dimensional bone structure
Trabecular thickness
Trabecular separation
[Bouxsein, Boyd, Christiansen, Guldberg, Jepsen, & Müller 2010]


---

## 第19页

The goal is to reduce measurement time
by recording fewer radiographs
3D FDK reconstruction
based on 40 angles
3D shearlet-sparsity reconstruction
based on 40 angles


---

## 第20页

Thickness
Separation
Thickness
Separation
0.34
0.71
0.37
0.35
Projections: 300
Projections: 300
[Purisha et al. 2019]
Bone quality parameters from ground truth


---

## 第21页

Thickness
Separation
0.34
0.71
0.37
0.35
Projections: 300
Projections: 50
Projections: 30
Projections: 300
Projections: 50
Projections: 30
[Purisha et al. 2019]
Results from FDK reconstructions


---

## 第22页

Thickness
Separation
0.34
0.71
0.37
0.35
Projections: 300
Projections: 50
Projections: 30
Projections: 300
Projections: 50
Projections: 30
[Purisha et al. 2019]
Results from 3D shearlet-sparsity reconstructions


---

## 第23页

The osteoarthritis project was a joint work with
Tatiana Bubba, University of Helsinki, Finland
Sakari Karhula, Oulu University Hospital, Finland
Juuso Ketola, Oulu University Hospital, Finland
Maximilian März, TU Berlin
Miika T. Nieminen, University of Oulu, Finland
Zenith Purisha, University of Helsinki, Finland
Juho Rimpeläinen, University of Helsinki, Finland
Simo Saarakkala, Oulu University Hospital, Finland


---

## 第24页

Outline
Why pixel-based tomographic modelling?
Restricted time →sparse tomography
Restricted radiation dose →sparse tomography
Restricted money →sparse tomography
The Beer-Lambert Law
Pixel-based measurement model
Matrix model for sparse tomography
Transpose of A: backprojection
Ill-posedness of sparse tomography
Total variation regularization
Regularization
Tikhonov regularization
Total variation regularization
Frame-sparsity methods


---

## 第25页

Climate change is predicted using climate models
Source: Wikipedia


---

## 第26页

Climeta models have a lot of details, and plant
metabolism is crucial to model accurately
Le Treut, Somerville, Cubasch, Ding, Mauritzen, Mokssit, Peterson & Prather 2007


---

## 第27页

Tomography study jointly with physicists,
biologists, radiochemists and climate scientists


---

## 第28页

Time-dependent sparse tomography reveals
the movement of iodine in the phloem
Bubba, Heikkilä, Help, Huotari, Salmon & S, in press
0 minutes
166 minutes
235 minutes


---

## 第29页

Outline
Why pixel-based tomographic modelling?
Restricted time →sparse tomography
Restricted radiation dose →sparse tomography
Restricted money →sparse tomography
The Beer-Lambert Law
Pixel-based measurement model
Matrix model for sparse tomography
Transpose of A: backprojection
Ill-posedness of sparse tomography
Total variation regularization
Regularization
Tikhonov regularization
Total variation regularization
Frame-sparsity methods


---

## 第30页

The next example of sparse-data tomography
is about sawing tree trunks into planks
Pictures: Stora Enso


---

## 第31页

The sawmill industry wants to cut tree trunks so
that the amount of material produced is maximal
Decoder
Decoder
Decoder
Log measurements
X-ray
Laser scan
Tomography/
log model
Sawing
model 
{α1, p1, t1}
Board 
quality/value
prediction
Sawing
optimization
Board
quality 
computation
{α, p, t}
^ ^ ^
RGB images of boards
sawing
Prior
...
...
...
Veriﬁcation
angle
pattern
trimming
Sawing parameters
Virtual boards
Optimal
sawing parameters
Data acquisition
Log reconstruction
and defect detection
Virtual sawing
Sawing optimization
Knot 
segmentation
Picture courtesy of Sebastian Springer and Andreas Hauptmann


---

## 第32页

Finnish spin-oﬀcompany Finnos designs X-ray
imaging devices for sawing
Picture: Finnos


---

## 第33页

Each angle needs X-ray source →sparse data
Reference
360 angles
Kalman ﬁlter
3 angles
Kalman ﬁlter
9 angles
Springer, Glielmo, Senchukova, Kauppi, Suuronen,
Roininen, Haario and Hauptmann (arXiv:2206.09595v1)


---

## 第34页

Laser tomography enables monitoring gases
of industry and agriculture (Mirico)


---

## 第35页

Laser tomography enables monitoring gases
of industry and agriculture (Mirico)


---

## 第36页

Outline
Why pixel-based tomographic modelling?
Restricted time →sparse tomography
Restricted radiation dose →sparse tomography
Restricted money →sparse tomography
The Beer-Lambert Law
Pixel-based measurement model
Matrix model for sparse tomography
Transpose of A: backprojection
Ill-posedness of sparse tomography
Total variation regularization
Regularization
Tikhonov regularization
Total variation regularization
Frame-sparsity methods


---

## 第37页

X-ray intensity attenuates inside matter,
here shown with a homogeneous block


---

## 第38页

Formula for X-ray attenuation
along a line inside homogeneous matter
An X-ray with intensity I0 enters a homogeneous physical body.
I0
I1
•
-
|
{z
}
s
The intensity I1 of the X-ray when it exits the material is
I1 = I0e−µs,
where s is the length of the path of the X-ray inside the body and
µ > 0 is X-ray attenuation coeﬃcient.


---

## 第39页

X-ray intensity attenuates inside matter,
here shown with two homogeneous blocks


---

## 第40页

A digital X-ray detector counts how many
photons arrive at each pixel
X-ray source
1000
photon
count
1000
•
-
Detector


---

## 第41页

Adding material between the source and detector
reveals the exponential X-ray attenuation law
1000
1000
1000
photon
count
1000
500
250
•
-
•
-
•
-


---

## 第42页

Physical material kind of acts as “sunglasses”
for X-rays
1000
1000
1000
photon
count
1000
500
250
•
-
•
-
•
-


---

## 第43页

We take logarithm of the photon counts to
compensate for the exponential attenuation law
log
6.9
6.2
5.5
1000
1000
1000
photon
count
1000
500
250
•
-
•
-
•
-


---

## 第44页

Final calibration step is to subtract the logarithms
from the empty space value (here 6.9)
log
6.9
6.2
5.5
1000
1000
1000
photon
count
1000
500
250
•
-
•
-
•
-
line
integral
0.0
0.7
1.4


---

## 第45页

Formula for X-ray attenuation along a line:
Beer-Lambert law
Let f : [a, b] →R be a nonnegative function modelling X-ray
attenuation along a line inside a physical body.
Beer-Lambert law connects the initial and ﬁnal intensities:
I1 = I0e−
R b
a f (x)dx.
We can also write it in the form
−log(I1/I0) =
Z b
a
f (x)dx,
where I0 is known from calibration and I1 from measurement.


---

## 第46页

Attenuation process is complicated inside a head
because there are diﬀerent tissues


---

## 第47页

After calibration we are observing how much
attenuating matter the X-ray encounters in total


---

## 第48页

Outline
Why pixel-based tomographic modelling?
Restricted time →sparse tomography
Restricted radiation dose →sparse tomography
Restricted money →sparse tomography
The Beer-Lambert Law
Pixel-based measurement model
Matrix model for sparse tomography
Transpose of A: backprojection
Ill-posedness of sparse tomography
Total variation regularization
Regularization
Tikhonov regularization
Total variation regularization
Frame-sparsity methods


---

## 第49页

Outline
Why pixel-based tomographic modelling?
Restricted time →sparse tomography
Restricted radiation dose →sparse tomography
Restricted money →sparse tomography
The Beer-Lambert Law
Pixel-based measurement model
Matrix model for sparse tomography
Transpose of A: backprojection
Ill-posedness of sparse tomography
Total variation regularization
Regularization
Tikhonov regularization
Total variation regularization
Frame-sparsity methods


---

## 第50页

Our test example is a cartoon bunny image.
We simulate projection data along 11 directions
1024×1024 pixels


---

## 第51页

Angle 1


---

## 第52页

Angle 2


---

## 第53页

Angle 3


---

## 第54页

Angle 4


---

## 第55页

Angle 5


---

## 第56页

Angle 6


---

## 第57页

Angle 7


---

## 第58页

Angle 8


---

## 第59页

Angle 9


---

## 第60页

Angle 10


---

## 第61页

Angle 11


---

## 第62页

Discretize the unknown by dividing it into pixels;
this is necessary for ﬁnite computation
Target (unknown)
12×12 pixel grid


---

## 第63页

x1
x13 x25 x37 x49 x61 x73 x85 x97 x109 x121 x133
x2
x14 x26 x38 x50 x62 x74 x86 x98 x110 x122 x134
x3
x15 x27 x39 x51 x63 x75 x87 x99 x111 x123 x135
x4
x16 x28 x40 x52 x64 x76 x88 x100 x112 x124 x136
x5
x17 x29 x41 x53 x65 x77 x89 x101 x113 x125 x137
x6
x18 x30 x42 x54 x66 x78 x90 x102 x114 x126 x138
x7
x19 x31 x43 x55 x67 x79 x91 x103 x115 x127 x139
x8
x20 x32 x44 x56 x68 x80 x92 x104 x116 x128 x140
x9
x21 x33 x45 x57 x69 x81 x93 x105 x117 x129 x141
x10 x22 x34 x46 x58 x70 x82 x94 x106 x118 x130 x142
x11 x23 x35 x47 x59 x71 x83 x95 x107 x119 x131 x143
x12 x24 x36 x48 x60 x72 x84 x96 x108 x120 x132 x144


---

## 第64页

Then we aim for best possible reconstruction at
the resolution provided by the grid
Target (unknown)
12×12 pixel grid
Desired reconstruction
(downsampled target)


---

## 第65页

We need to model the ﬁnite detector
Our simulated detector has 21 pixels. This number comes from the
structure of the Matlab routine radon.m applied to a phantom of
size 12×12.


---

## 第66页

Angle 1


---

## 第67页

Angle 2


---

## 第68页

Angle 3


---

## 第69页

Angle 4


---

## 第70页

Angle 5


---

## 第71页

Angle 6


---

## 第72页

Angle 7


---

## 第73页

Angle 8


---

## 第74页

Angle 9


---

## 第75页

Angle 10


---

## 第76页

Angle 11


---

## 第77页

Let us build a discrete model for the 11-angle
measurement shown above
We have 11 directions and 21 X-rays for each direction.
Every X-ray corresponds to one row in the matrix.
Therefore, our system matrix A has 11 · 21 = 231 rows.
Our unknown image has 12×12 = 144 pixels, and consequently
the matrix A has 144 columns.
Denote x ∈R144 and m ∈R231. The matrix model is
Ax = m.
Next we see how to construct the matrix A.


---

## 第78页

Measurement direction 1


---

## 第79页

1
2
3
4
5
6
7
8
9
Paths of X-rays numbered from 1 to 9


---

## 第80页



0
0
0
0
0
0
0
· · ·
0
0
0
0
0
0
0
· · ·
0
0
0
0
0
0
0
· · ·
0
0
0
0
0
0
0
· · ·
0
0
0
0
0
0
0
· · ·
1
0
0
0
0
0
0
· · ·
0
1
0
0
0
0
0
· · ·
0
0
1
0
0
0
0
· · ·
0
0
0
1
0
0
0
· · ·
0
0
0
0
1
0
0
· · ·
...
...
...
...
...
...
...
...


Ray 1
Ray 2
Ray 3
Ray 4
Ray 5
Ray 6
Ray 7
Ray 8
Ray 9
Ray 10
x1
x2
x3
x4
x5
x6
x7
Ten ﬁrst rows of the tomographic matrix, if we
used geometric lengths of the rays as entries


---

## 第81页

A =


0
0
0
0
0
0
0
· · ·
0
0
0
0
0
0
0
· · ·
0
0
0
0
0
0
0
· · ·
0
0
0
0
0
0
0
· · ·
0.125
0
0
0
0
0
0
· · ·
0.750
0.125
0
0
0
0
0
· · ·
0.125
0.750
0.125
0
0
0
0
· · ·
0
0.125
0.750
0.125
0
0
0
· · ·
0
0
0.125
0.750
0.125
0
0
· · ·
0
0
0
0.125
0.750
0.125
0
· · ·
...
...
...
...
...
...
...
...


Ray 1
Ray 2
Ray 3
Ray 4
Ray 5
Ray 6
Ray 7
Ray 8
Ray 9
Ray 10
x1
x2
x3
x4
x5
x6
x7
Matlab’s radon.m routine produces the
tomographic matrix using the pencil-beam model


---

## 第82页



· · ·
0
0
0
0
0
0
· · ·
· · ·
0
0
0
0
0
0
· · ·
· · ·
0
0
0
0
0
0
· · ·
· · ·
0
0
0
0
0
0
· · ·
· · ·
0.125
0
0
0
0
0
· · ·
· · ·
0.750
0.125
0
0
0
0
· · ·
· · ·
0.125
0.750
0.125
0
0
0
· · ·
· · ·
0
0.125
0.750
0.125
0
0
· · ·
· · ·
0
0
0.125
0.750
0.125
0
· · ·
· · ·
0
0
0
0.125
0.750
0.125
· · ·
...
...
...
...
...
...
...


Ray 1
Ray 2
Ray 3
Ray 4
Ray 5
Ray 6
Ray 7
Ray 8
Ray 9
Ray 10
x13
x14
x15
x16
x17
x18
Here we see some columns starting from 13


---

## 第83页

Nonzero entries of matrix A corresponding to the
ﬁrst measurement direction


---

## 第84页

Measurement direction 2


---

## 第85页

22
23
24
25
26
27
28
29
Paths of X-rays used in the second angle


---

## 第86页

A =


...
...
...
...
...
...
...
0
0
0
0
0
0
0
· · ·
0
0
0
0
0
0
0
· · ·
0.064
0
0
0
0
0
0
· · ·
0.782
0.139
0
0
0
0
0
· · ·
0.154
0.790
0.257
0
0
0
0
· · ·
0
0.071
0.714
0.386
0
0
0
· · ·
0
0
0.030
0.614
0.544
0.012
0
· · ·
0
0
0
0
0.456
0.679
0.052
· · ·
0
0
0
0
0
0.309
0.758
· · ·
0
0
0
0
0
0
0.190
· · ·
...
...
...
...
...
...
...
...


Ray 22
Ray 23
Ray 24
Ray 25
Ray 26
Ray 27
Ray 28
Ray 29
Ray 30
Ray 31
x1
x2
x3
x4
x5
x6
x7
Matlab’s radon.m routine produces the
tomographic matrix using the pencil-beam model


---

## 第87页

A =


...
...
...
...
...
...
...
· · ·
0
0
0
0
0
0
· · ·
· · ·
0
0
0
0
0
0
· · ·
· · ·
0
0
0
0
0
0
· · ·
· · ·
0.369
0
0
0
0
0
· · ·
· · ·
0.631
0.528
0.008
0
0
0
· · ·
· · ·
0
0.472
0.670
0.048
0
0
· · ·
· · ·
0
0
0.322
0.750
0.107
0
· · ·
· · ·
0
0
0
0.203
0.790
0.208
· · ·
· · ·
0
0
0
0
0.103
0.746
· · ·
· · ·
0
0
0
0
0
0.046
· · ·
...
...
...
...
...
...
...


Ray 22
Ray 23
Ray 24
Ray 25
Ray 26
Ray 27
Ray 28
Ray 29
Ray 30
Ray 31
x13
x14
x15
x16
x17
x18
Matlab’s radon.m routine produces the
tomographic matrix using the pencil-beam model


---

## 第88页

Nonzero entries of matrix A corresponding to the
ﬁrst two measurement directions


---

## 第89页

Nonzero entries of matrix A corresponding to
measurement directions 1–3


---

## 第90页

Nonzero entries of matrix A corresponding to
measurement directions 1–4


---

## 第91页

Nonzero entries of matrix A corresponding to
measurement directions 1–5


---

## 第92页

Nonzero entries of matrix A corresponding to
measurement directions 1–6


---

## 第93页

Nonzero entries of matrix A corresponding to
measurement directions 1–7


---

## 第94页

Nonzero entries of matrix A corresponding to
measurement directions 1–8


---

## 第95页

Nonzero entries of matrix A corresponding to
measurement directions 1–9


---

## 第96页

Nonzero entries of matrix A corresponding to
measurement directions 1–10


---

## 第97页

Nonzero entries of matrix A corresponding to
measurement directions 1–11


---

## 第98页

Outline
Why pixel-based tomographic modelling?
Restricted time →sparse tomography
Restricted radiation dose →sparse tomography
Restricted money →sparse tomography
The Beer-Lambert Law
Pixel-based measurement model
Matrix model for sparse tomography
Transpose of A: backprojection
Ill-posedness of sparse tomography
Total variation regularization
Regularization
Tikhonov regularization
Total variation regularization
Frame-sparsity methods


---

## 第99页

Let’s apply the transpose matrix AT to a sinogram
having just one nonzero entry
Sinogram in image form:
g ∈R21×11
Back-projection image:
ATg ∈R12×12
AT
−→
Note: you see the path of the X-ray indicated by the sinogram’s white pixel.


---

## 第100页

Let’s apply the transpose matrix AT to a sinogram
having just one nonzero entry
Sinogram in image form:
g ∈R21×11
Back-projection image:
ATg ∈R12×12
AT
−→
Note: you see the path of the X-ray indicated by the sinogram’s white pixel.


---

## 第101页

Consider a target with just one white pixel.
The sinogram is like a sine curve!
Target image:
f ∈R12×12
Sinogram in image form:
Af ∈R21×11
A
−→


---

## 第102页

Let’s apply the transpose matrix AT to a sinogram
arising from a target with just one white pixel
Sinogram in image form:
g ∈R21×11
Back-projection image:
ATg ∈R12×12
AT
−→


---

## 第103页

Outline
Why pixel-based tomographic modelling?
Restricted time →sparse tomography
Restricted radiation dose →sparse tomography
Restricted money →sparse tomography
The Beer-Lambert Law
Pixel-based measurement model
Matrix model for sparse tomography
Transpose of A: backprojection
Ill-posedness of sparse tomography
Total variation regularization
Regularization
Tikhonov regularization
Total variation regularization
Frame-sparsity methods


---

## 第104页

Naive reconstruction using the minimum norm
solution from the normal equation (ATA)f † = ATm
Ground truth: 12×12 resolution,
values between 0 and 1
Reconstruction: minimum pixel value
−294, maximum pixel value 380


---

## 第105页

Naive reconstruction using the minimum norm
solution with non-negativity constraint
Ground truth: 12×12 resolution,
values between 0 and 1
Reconstruction: minimum pixel value
0, maximum pixel value 2.3


---

## 第106页

Sinogram


---

## 第107页

Sinogram


---

## 第108页

Sinogram


---

## 第109页

Sinogram


---

## 第110页

Sinogram


---

## 第111页

Sinogram


---

## 第112页

Sinogram


---

## 第113页

Sinogram


---

## 第114页

Sinogram


---

## 第115页

Sinogram


---

## 第116页

Sinogram


---

## 第117页

Ill-posedness: ﬁrst example of an almost-ghost
RRSE=134%
RRSE=0.1%
@
 @
 A
A


---

## 第118页

Ill-posedness: ﬁrst example of an almost-ghost
RRSE=134%
RRSE=0.1%
@
 @
 A
A


---

## 第119页

Ill-posedness: second example of an almost-ghost
RRSE=134%
RRSE=0.1%
@
 @
 A
A


---

## 第120页

Almost-ghost A with non-negativity constraint
RRSE=58%, SSIM=0.74
RRSE=12%
@
 @
 A
A


---

## 第121页

Singular Value Decomposition for k×n matrix A:
A = UDV T with UUT=I =UTU and VV T=I =V TV
A = UDV T = U


d1
0
· · ·
0
· · ·
0
0
d2
...
...
...
dr
0
...
...
...
0
· · ·
· · ·
0


V T
The singular values dj satisfy d1 ≥d2 ≥· · · ≥dr > 0
and dr+1 = dr+2 = · · · = dmin{k,n} = 0. Note that r = rank(A).
If n = k and all singular values are positive, then A is invertible.
However, the condition number cond(A) := d1/dr may be large.
In that case A−1 is a numerically unstable matrix.


---

## 第122页

Singular value decomposition is A = UDV T,
where UTU = I = UUT and V TV = I = V V T
A =
U
D
V T
231×231
231×144
144×144


---

## 第123页

Singular values of A
Nonzero elements of A
In ill-posed problems, singular values decrease
gradually from large to extremely small


---

## 第124页

Outline
Why pixel-based tomographic modelling?
Restricted time →sparse tomography
Restricted radiation dose →sparse tomography
Restricted money →sparse tomography
The Beer-Lambert Law
Pixel-based measurement model
Matrix model for sparse tomography
Transpose of A: backprojection
Ill-posedness of sparse tomography
Total variation regularization
Regularization
Tikhonov regularization
Total variation regularization
Frame-sparsity methods


---

## 第125页

Total variation (TV) regularization is a technique
for preserving edges in the reconstruction
We consider calculating the minimizer of the TV functional
∥Ax −m∥2
2 + α {∥LHx∥1 + ∥LVx∥1}
=
∥Ax −m∥2
2 + α
n X
j
X
i

|xi(j+1) −xij| + |x(i+1)j −xij|
o
where LH and LV are horizontal and vertical ﬁrst-order diﬀerence
matrices. [Rudin, Osher and Fatemi 1992]


---

## 第126页

Non-negative Total Variation (TV) regularization
with too small parameter α = 0.0001
Ground truth: 12×12 resolution,
values between 0 and 1
TV regularized reconstruction
RRSE=66% ,SSIM=0.40


---

## 第127页

Non-negative Total Variation (TV) regularization
with pretty good parameter α = 0.3
Ground truth: 12×12 resolution,
values between 0 and 1
TV regularized reconstruction
RRSE=32% ,SSIM=0.87


---

## 第128页

Non-negative Total Variation (TV) regularization
with too large parameter α = 4
Ground truth: 12×12 resolution,
values between 0 and 1
TV regularized reconstruction
RRSE=45% ,SSIM=0.70


---

## 第129页

Remember the nonnegative almost-ghost A
RRSE=58%, SSIM=0.74
RRSE=12%
@
 @
 A
A


---

## 第130页

TV reconstruction with α = 0.2, ghost A
RRSE=18%
SSIM=0.97
SSIM=0.85
RRSE=12%
 @
 @
TV
TV


---

## 第131页

Remember the nonnegative almost-ghost B
RRSE=58%, SSIM=0.74
RRSE=12%
@
 @
 A
A


---

## 第132页

TV reconstruction with α = 0.2, ghost B
RRSE=19%
SSIM=0.97
SSIM=0.91
RRSE=12%
 @
 @
TV
TV


---

## 第134页

What can we expect to see from sparse data?
[Cormack 1963], [Smith, Solmon & Wagner 1977, Theorem 4.2]


---

## 第135页

Measuring the closeness of two images:
root relative squared error (RRSE)
We need a number describing the similarity of images A and B.
For example, we might want to compare a reconstruction to the
ground truth, or quantify measurement noise amplitude as the
diﬀerence between an ideal sinogram and a noisy sinogram.
The most classical method is RRSE, deﬁned by
RRSE(A, B) =
qP
i
P
j(Aij −Bij)2
qP
i
P
j A2
ij
,
where i is row index and j is column index.
Note that RRSE(A, A) = 0 means perfect ﬁt, and that there is no
upper bound for RRSE(A, B).


---

## 第136页

Measuring the closeness of two images:
structural similarity index (SSIM)
RRSE has the downside of ignoring image structure: scrambling
pixels makes no diﬀerence in RRSE. In imaging applications we need
methods that better match the human perception of image quality.
One such option is SSIM, deﬁned by
SSIM(A, B) =
(2µAµB + c1)(2σAB + c2)
(µ2
Aµ2
B + c1)(σ2
A + σ2
B + c2),
where µA, µB are averages, σA, σB variances, and σAB the
covariance of A and B. For constants c1, c2 and other further
details see [Wang, Bovik, Sheikh and Simoncelli 2004].
Note that SSIM(A, A) = 1 means perfect ﬁt, and that always
−1 ≤SSIM(A, B) ≤1.


---

## 第137页

Outline
Why pixel-based tomographic modelling?
Restricted time →sparse tomography
Restricted radiation dose →sparse tomography
Restricted money →sparse tomography
The Beer-Lambert Law
Pixel-based measurement model
Matrix model for sparse tomography
Transpose of A: backprojection
Ill-posedness of sparse tomography
Total variation regularization
Regularization
Tikhonov regularization
Total variation regularization
Frame-sparsity methods


---

## 第138页

We saw before that for ill-posed inverse problems
such as tomography, naive inversion fails
We need reconstruction methods that are robust against noise and
modelling errors.
There are several methodologies for that, including
▶variational regularization,
▶Bayesian inversion,
▶machine learning.
In these slides we will take a look at the ﬁrst.


---

## 第139页

Inverse problem of X-ray tomography: given
noisy sinogram, ﬁnd a stable approximation to f
Model space X = R32×32
Data space Y = R32×49
D(A)
A(D(A))
f
=
Af =
m
A


---

## 第140页

Robust solution of ill-posed inverse problems
requires regularization
Model space X = R32×32
Data space Y = R39×49
D(A)
A(D(A))
f
Af
m
A
δ
Γα
Γα(m)
We need to deﬁne a family of continuous functions Γα : Y →X so that
the reconstruction error ∥Γα(δ)(m) −x∥X vanishes asymptotically at the
zero-noise level δ →0.


---

## 第141页

You can ﬁnd all the slides and codes of this
course in GitHub
https://github.com/ssiltane/BolognaWinterSchool2023


---

## 第142页

Outline
Why pixel-based tomographic modelling?
Restricted time →sparse tomography
Restricted radiation dose →sparse tomography
Restricted money →sparse tomography
The Beer-Lambert Law
Pixel-based measurement model
Matrix model for sparse tomography
Transpose of A: backprojection
Ill-posedness of sparse tomography
Total variation regularization
Regularization
Tikhonov regularization
Total variation regularization
Frame-sparsity methods


---

## 第143页

Tikhonov regularization is the classical method for
noise-robust tomographic reconstruction
Write a penalty functional
Φ(f ) = ∥Af −m∥2
2 + α∥f ∥2
2,
where 0 < α < ∞is a regularization parameter. Deﬁne Γα(m) by
Φ(Γα(m)) = min
f ∈X{Φ(f )}.
We denote
Γα(m) = arg min
f ∈X
{∥Af −m∥2
2 + α∥f ∥2
2}.


---

## 第144页

Tikhonov regularization can be expressed as
ﬁltering the singular values of the matrix A
Γα(m) = V


d1
d2
1 + α
0
· · ·
0
0
...
...
...
...
0
0
· · ·
0
dmin{k,n}
d2
min{k,n} + α


UTm
In large-scale computations it is better to use the formula
Γα(m) = (ATA + αI)−1ATm
and an iterative solver such as the conjugate gradient method.


---

## 第145页

Implementation of the matrix A and its transpose
AT (back-projection) are crucial things
In my small-scale two-dimensional Matlab examples I just use a
brute-force trick for constructing A by calling the radon.m routine
of Matlab repeatedly.
In practice it is important not to construct A or the back-projection
operator AT at all as matrices. Rather, one can use GPU-powered
algorithms for calculating the maps f 7→Af and g 7→ATg for
vectors f and g given by the solution method at each iteration.
For eﬃcient matrix-free implementations, check out
https://tomopedia.github.io/


---

## 第146页

Outline
Why pixel-based tomographic modelling?
Restricted time →sparse tomography
Restricted radiation dose →sparse tomography
Restricted money →sparse tomography
The Beer-Lambert Law
Pixel-based measurement model
Matrix model for sparse tomography
Transpose of A: backprojection
Ill-posedness of sparse tomography
Total variation regularization
Regularization
Tikhonov regularization
Total variation regularization
Frame-sparsity methods


---

## 第147页

Recall the Lp norms for Rn
Let f ∈Rn. The Lp norms for 1 ≤p < ∞are deﬁned by
∥f ∥p =
 n
X
j=1
|fj|p1/p.
In particular we use the following two cases:
∥f ∥2
2 =
n
X
j=1
|fj|2,
∥f ∥1 =
n
X
j=1
|fj|.


---

## 第148页

Total variation (TV) regularization is a technique
for preserving edges in the reconstruction
We consider calculating the minimizer of the TV functional
∥Af −m∥2
2 + α {∥LHf ∥1 + ∥LVf ∥1}
=
∥Af −m∥2
2 + α
n X
j
X
i

|fi(j+1) −fij| + |f(i+1)j −fij|
o
where LH and LV are horizontal and vertical ﬁrst-order diﬀerence
matrices. [Rudin, Osher and Fatemi 1992]


---

## 第149页

Computational resources for total variation
You can ﬁnd Matlab code for the above calculations at the
following links. Many thanks to Professor Kristian Bredies for
sharing his primal-dual codes!
https://blog.ﬁps.ﬁ/tomography/x-ray/
total-variation-regularization-for-x-ray-tomography/
Another computational method for the same problem is here:
https://blog.ﬁps.ﬁ/tomography/x-ray/
total-variation-regularization-for-x-ray-tomography-experimental-data/
The 2×2 pixel example is here:
https://github.com/ssiltane/SiltanenSparseTomography2x2
Also, check out Hendrik Dirks’ repository FlexBox:
https://github.com/HendrikMuenster/ﬂexBox


---

## 第150页

TV tomography: arg min
f ∈Rn
{∥Af −m∥2
2 + α∥∇f ∥1}
1992 Rudin, Osher & Fatemi: denoise images by taking A = I
1998 Delaney & Bresler
2001 Persson, Bone & Elmqvist
2003 Kolehmainen, S, Järvenpää, Kaipio, Koistinen, Lassas, Pirttilä
& Somersalo (ﬁrst TV work with measured X-ray data)
2006 Kolehmainen, Vanne, S, Järvenpää, Kaipio, Lassas & Kalke
2006 Sidky, Kao & Pan
2008 Liao & Sapiro
2008 Sidky & Pan
2008 Herman & Davidi
2009 Tang, Nett & Chen
2009 Duan, Zhang, Xing, Chen & Cheng
2010 Bian, Han, Sidky, Cao, Lu, Zhou & Pan
2011 Jensen, Jørgensen, Hansen & Jensen
2011 Tian, Jia, Yuan, Pan & Jiang
2012–present: hundreds of articles indicated by Google Scholar


---

## 第151页

There are many computational approaches for
computing the minimum
Primal-dual algorithms Bredies, Chambolle, Chan, Chen, Esser,
Golub, Mulet, Nesterov, Zhang
Thresholding Candès, Chambolle, Chaux, Combettes, Daubechies,
Defrise, DeMol, Donoho, Pesquet, Starck, Teschke, Vese, Wajs
Bregman iteration Cai, Burger, Darbon, Dong, Goldfarb, Mao, Osher,
Shen, Xu, Yin, Zhang
Splitting approaches Chan, Esser, Fornasier, Goldstein, Langer,
Osher, Schönlieb, Setzer, Wajs
Nonlocal TV Bertozzi, Bresson, Burger, Chan, Lou, Osher, Zhang
There is also the simple quadratic programming trick we used for
the 2×2 example before. That only works for relatively coarse
pixelizations.


---

## 第152页

Quadratic programming (QP) for TV
regularization
The minimizer of the functional
arg min
f ∈Rn
+

∥Af −m∥2
2 + α∥LHf ∥1 + α∥LVf ∥1
	
can be transformed into the standard form
arg min
z∈R5n
1
2zTQz + cTz

,
z ≥0,
Ez = b,
where Q is symmetric and E implements equality constraints.
Large-scale primal-dual interior point QP method was developed in
Kolehmainen, Lassas, Niinimäki & S (2012) and
Hämäläinen, Kallonen, Kolehmainen, Lassas, Niinimäki & S (2013).


---

## 第153页

Reduction to arg min
z∈R5n
1
2zTQz + cTz
	
Denote horizontal and vertical diﬀerences by
LHf = u+
H −u−
H
and
LVf = u+
V −u−
V ,
where u±
H , u±
V ≥0. TV minimization is now
arg min
f ∈Rn
+
n
f TATAf −2f TATm + α1T(u+
H + u−
H + u+
V + u−
V )
o
,
where 1 ∈Rn is vector of all ones. Further, we denote
z =


f
u+
H
u−
H
u+
V
u−
V


,
Q =


1
σ2 ATA
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0


,
c =


−2ATm
α1
α1
α1
α1


.


---

## 第154页

Non-negative TV regularization
arg min
f ∈Rn
+

∥Af −m∥2
2 + α∥∇f ∥1
	
Original phantom sampled at
32×32 resolution
TV regularized reconstruction
Relative square norm error 7%


---

## 第155页

Let’s consider a square phantom
@
 A
f ∈R32×32
Af ∈R49×39


---

## 第156页

Naive reconstruction using the Moore-Penrose
pseudoinverse; data has 0.1% relative noise
Original phantom, values between
zero (black) and one (white)
Naive reconstruction with minimum
−14.9 and maximum 18.5


---

## 第157页

Standard Tikhonov regularization
arg min
f ∈Rn

∥Af −m∥2
2 + α∥f ∥2
2
	
Original phantom
Reconstruction
Relative square norm error 35%


---

## 第158页

Constrained Tikhonov regularization
arg min
f ∈Rn
+

∥Af −m∥2
2 + α∥f ∥2
2
	
Original phantom
Reconstruction
Relative square norm error 13%


---

## 第159页

Constrained total variation (TV) regularization
arg min
f ∈Rn
+

∥Af −m∥2
2 + α {∥LHf ∥1 + ∥LVf ∥1}
	
Original phantom
TV regularized reconstruction
Relative square norm error 3%


---

## 第160页

In variational regularization, the penalty term
expresses a priori knowledge about the unknown
Standard Tikhonov regularization:
arg min
f ∈Rn

∥Af −m∥2
2 + α∥f ∥2
2
	
Non-negativity constrained Tikhonov regularization:
arg min
f ∈Rn
+

∥Af −m∥2
2 + α∥f ∥2
2
	
Non-negativity constrained Total Variation (TV) regularization:
arg min
f ∈Rn
+

∥Af −m∥2
2 + α∥∇f ∥1
	


---

## 第161页

Outline
Why pixel-based tomographic modelling?
Restricted time →sparse tomography
Restricted radiation dose →sparse tomography
Restricted money →sparse tomography
The Beer-Lambert Law
Pixel-based measurement model
Matrix model for sparse tomography
Transpose of A: backprojection
Ill-posedness of sparse tomography
Total variation regularization
Regularization
Tikhonov regularization
Total variation regularization
Frame-sparsity methods


---

## 第162页

Daubechies, Defrise and de Mol introduced
a revolutionary inversion method in 2004
Consider the sparsity-promoting variational regularization
arg min
f ∈Rn

∥Af −m∥2
2 + µ∥Wf ∥1
	
,
where W is an orthonormal wavelet transform. The minimizer can
be computed using the iteration
fj+1 = W −1SµW

fj + AT(m −Afj)

,
where the soft-thresholding operation
Sµ(x) =



x + µ
2
if x ≤−µ
2,
0
if |x| < µ
2,
x −µ
2
if x ≥µ
2,
is applied to each wavelet coeﬃcient separately.


---

## 第163页

Illustration of the Haar wavelet transform


---

## 第164页

How to choose the thresholding parameter µ?
Here it is too small.


---

## 第165页

How to choose the thresholding parameter µ?
Here it is too large.


---

## 第166页

Automatic parameter choice using
controlled wavelet-domain sparsity (CWDS)
Assume given the a priori sparsity level 0 ≤Cpr ≤1.
Denote by Cj the sparsity of the jth iterate fj ∈Rn:
Cj = (number of nonzero elements in Wfj ∈Rn)/n.
The CWDS iteration is based on proportional-integral-derivative
(PID) controllers:
µ(i+1) = µ(i) + β(C(i) −Cpr).
[Purisha, Rimpeläinen, Bubba & S 2018]


---

## 第167页

CWDS choice of the thresholding parameter µ


---

## 第168页

CWDS choice of the thresholding parameter µ


---

## 第169页

We modify the method so that non-negativity
constraint has rigorous mathematical foundation
The minimizer
argmin
f ∈Rn
+
(
1
2∥Af −m∥2
2 + µ ∥Wf ∥1
)
can be computed using this iteration:
y(i+1) = PC

f (i) −τ∇g(f (i)) −λW Tv(i)
v(i+1) =

I −Sµ

Wy(i+1) + v(i)
f (i+1) = PC

f (i) −τ∇g(f (i)) −λW Tv(i+1)
where τ > 0, λ > 0 and g(f ) = 1
2∥Af −m∥2
2. Here PC denotes
projection to the non-negative “quadrant.”
[Loris & Verhoeven 2011], [Chen, Huang & Zhang 2016]


---

## 第170页

Sparse-data reconstruction of the walnut using
Haar wavelet sparsity
Filtered back-projection
Constrained Besov regularization
arg min
f ∈Rn
+
n
∥Af −m∥2
2 + α∥f ∥B1
11
o


---

## 第171页

Computational resources for frame sparsity
You can ﬁnd Matlab code for the above calculations at
https://blog.ﬁps.ﬁ/tomography/x-ray/
automatic-regularization-parameter-selection-controlled-
wavelet-domain-sparsity/


---

## 第172页

A note on Besov function spaces
We are using the norm ∥Wf ∥1 as a regularizer. Here Wf is the set
of wavelet coeﬃcientd of f , organized as a vector. What kind of
norm is that?
There is a theory of function spaces Bs
pq, named after Oleg Besov.
They consist of functions having a speciﬁc order s of weak
diﬀerentiability and certain integrability properties (p, q). In
particular, these two norms are equivalent:
∥Wf ∥1 ∼= ∥f ∥B1
11.
This is nice since B1
11 is concerned with L1 norms of ﬁrst derivatives,
a bit like TV, but more manageable as a space than TV or BV.
For information about these functions, check out the book Function
Spaces and Wavelets on Domains (EMS 2008) by Hans Triebel.


---

## 第173页

X-ray tomographic datasets
Finnish Inverse Problems Society: https://www.ﬁps.ﬁ/dataset.php
Tomobank: https://tomobank.readthedocs.io/en/latest/


---

