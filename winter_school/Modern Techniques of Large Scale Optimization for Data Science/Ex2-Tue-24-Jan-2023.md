# Ex2-Tue-24-Jan-2023.pdf

## 第1页

Jacek Gondzio, Margherita Porcelli, Filippo Zanetti
1
Advanced Methods for Mathematical Image Analysis Winter School
Modern Techniques of Large Scale Optimization
for Data Science
Exercise sheet 2
Problem 1: An interior point method for X-ray tomography
In this exercise, you will see an IPM in action on an X-ray tomography reconstruction problem. In
particular, the problem consists in understanding the distribution of two materials inside a 2-dimensional
object. The problem takes the following form
ˆg =
 ˆg(1)
ˆg(2)

= argming(j)≥0

∥m −Ag∥2
2 + αR(g) + βS(g)
	
,
where ˆg(j) contains the distribution of material j, m is the measurement vector, R(g) = ∥g∥2 is a standard
Tikhonov regularizer, with parameter α, while S(g) = 2 (ˆg(1))T ˆg(2) is a regularizer that promotes the
separation between the two materials, with parameter β. A is the following operator
A :=

c11A
c12A
c21A
c22A

,
where c11, c12, c21, c22 are the attenuation coeﬃcients of the two materials at two diﬀerent X-ray energies
and A is a matrix that depends on the geometry of the tomographic imaging process.
This problem can be rewritten as a standard quadratic program
argming(j)≥0 −mT Ag + 1
2gT Qg,
where
Q := Q1 + Q2,
Q1 := AT A =

(c2
11 + c2
21)AT A
(c11c12 + c21c22)AT A
(c11c12 + c21c22)AT A
(c2
12 + c2
22)AT A

,
Q2 :=
αI
βI
βI
αI

.
An IPM applied to this problem ﬁnds the Newton direction (∆g, ∆s) in the following way
(Q + G−1S)∆g = r1 + G−1r2,
∆s = G−1(r2 −S∆g),
(1)
where G = diag(g), S = diag(s) and r1 and r2 are vectors computed by the IPM algorithm.
Download the ﬁles for this session and open EXERCISE XRAY.m. This script sets up the problem and
calls an IPM to solve it (ipm xray.m). You are given the functions apply Operator A.m, that applies
A to a vector (i.e. performs the matrix vector product with A), apply Operator At.m, that applies AT ,
apply Q2.m, that applies Q2, and apply prec.m, that applies a preconditioner.
For the IPM to work, you need to complete the following tasks:
Task 1: Implement the function apply NE.m, that applies the normal equations matrix to a
vector, i.e. that performs the operation (Q + G−1S)x.
Task 2:
Inside ipm xray.m, call the PCG to solve the normal equations system (1); use
apply NE.m from the previous task and the provided preconditioner, set a tolerance of 10−6
and 500 as maximum number of iterations.
Now the script EXAMPLE XRAY.m should run correctly. At the beginning of the script, you can choose the
size of the problem; try to run it with N = 8, N = 16 or N = 32. You can also try to run the PCG with
or without preconditioner and observe the behaviour of the number of PCG iterations. Try to change the
value of the parameter β; what eﬀect does it have on the number of PCG iterations and on the quality
of the ﬁnal images?
You can ﬁnd more details about this problem and the speciﬁc interior point method in the paper:
J. Gondzio, M. Lassas, S. Latva-Aijo, S. Siltanen and F. Zanetti,
Material-separating regularizer for
multi-energy X-ray tomography. Inverse Problems 38, 2, 2022, 025013.
Extra Task (if you have time): Have a look at apply prec.m; can you understand how the
preconditioner works?


---

## 第2页

Jacek Gondzio, Margherita Porcelli, Filippo Zanetti
2
Problem 2: An interior point method to recover a partially known image
In this exercise, you will see an IPM in action on the problem of computing a low-rank approximation of an
image that is only partially known because some pixels are missing. We analyze the case when the missing
pixels are randomly distributed, see Figure 1 (downloaded from http://www.imageprocessingplace.com)
(a) True image
(b) 50% of missing pixels
Figure 1: The Lake 512 × 512 original grayscale image and the inpainted version (50% of missing pixels).
We formulate it as a matrix completion problem that takes the following form
min
rank( ¯X)
s.t.
¯XΩ= BΩ,
(2)
where Ωis the set of locations corresponding to the observed pixels of original image B and the equality
is meant element-wise, that is ¯Xs,t = Bs,t, for all (s, t) ∈Ω. Let m be the cardinality of Ωand r be the
rank of B. We expect this rank to be small. Deﬁne
X =
W1
¯X
¯XT
W2

,
where ¯X ∈Rn1×n2 is the matrix to be recovered and W1, W2 are symmetric matrices. Observe that the
minimization of trace(X) = trace(W1) + trace(W2) naturally promotes that ¯X has small rank.
Then problem (2) can be formulated as a standard semideﬁnite programming (SDP) problem
min
1
2I • X
s.t.

0
Θst
ΘT
st
0

• X = B(s,t),
(s, t) ∈Ω
X ⪰0,
(3)
where for each (s, t) ∈Ωthe matrix Θst is deﬁned element-wise as (Θst)kl =
 1/2
if (k, l) = (s, t)
0
otherwise,
.
We observe that the recovered image ¯X corresponds to the (1,2)-block of the primal variable X, the
symmetric matrix C in the objective of a standard SDP is a scaled identity matrix, the vector b ∈Rm is
deﬁned by the known pixels of the image B and, for i = 1, . . . , m, each constraint matrix Ai, corresponds
to the known elements of the image B stored in bi.
Since we are interested in ﬁnding a low-rank solution X of (3), a specialised relaxed version of the IPM
will be used: the Relaxed Interior Point Algorithm for Low Rank SDP (IPLR) is an IPM that promotes
the computation of low-rank solutions of SDPs.


---

## 第3页

Jacek Gondzio, Margherita Porcelli, Filippo Zanetti
3
Download the ﬁles for this session and open and read the documentation of IPLR BB.m.
This is an
implementation of IPLR for general standard SDPs where a Barzilai-Borwein gradient method is used
for computing the Newton steps. The function header is
function [X, y, S] = IPLR BB(C,A,b,r,S,y)
where C, A, b are the SDP data, r is the sought rank and (S, y) is a dual feasible starting pair. The
output is a primal-dual solution where X has rank r.
In order to recover the inpainted image in Figure 1, you need to complete the following tasks:
Task 1: Set up the SDP problem
Write the script mc image.m which:
• reads the image ’lake.bmp’ and shows it in Matlab
B = imread(’lake.bmp’);
figure(1)
imshow(B);
• converts the image into a double variable and assigns the ﬁrst slice to variable B true
(also recovers the image dimensions);
B = double(B); B true = B(:,:,1); [n1,n2] = size(B true);
• produces an image with p% of known pixels by generating m random entries and scales it by its
norm:
p = 0.8; m = round(p*(n1*n2));
Omega = randsample(n1*n2,m);
Implement the routine
function [A,b,C] = getInput(B true, Omega)
to deﬁne the matrices A, C and the vector b of the SDP formulation (3) and call it in the script.
To produce the (normalized) vector b you need to:
• Deﬁne b = B true(Omega);
• Set nb = norm(b); b = b/nb
Moreover, you can use the supplied function A = generate constraints(Omega,n1,n2) to generate the
matrix A of the constraints in (3). Such function is provided in the routine getInput.m
Task 2: Recover the image using IPLR
Inside mc image.m, deﬁne a dual feasible starting point for S and y, set a value of the rank r, say r = 80,
and call IPLR BB.m.
Task 3: Analyze the results
Extract the recovered image B rec from the output of IPLR BB.m, rescale it back by nB and show it. Also,
display the error in Frobenius norm wrt the true image both considering all the pixels (B true) and only
those in Ω(B part), and ﬁnally display the value of the signal-to-noise ratio:
PSNR = 10 ∗log10(n1 ∗n2 ∗2552/(norm(B true −B rec)2)).
Try diﬀerent values for p (percentage of known pixels) and for the rank r. What eﬀect does it have on
the number of IPM iterations and on the quality of the ﬁnal images?
You can ﬁnd more details about this problem and the speciﬁc interior point method in the paper:
S. Bellavia, J. Gondzio, and M. Porcelli, A relaxed interior point method for low-rank semideﬁnite pro-
gramming problems with applications to matrix completion, Journal of Scientiﬁc Computing, 89 (2021),
pp. 1-36.
Extra Task (if you have time): compute the best approximation of rank r of the original
image B true using the SVD; can you understand the pros and cons of using this technique
compared to the SDP approach?


---

