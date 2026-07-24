![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/8f24ca6b6b271a935241430f044d94ffe5a342eb4eb86edbc331bebc6a19c576.jpg)


S<sub>c</sub>h<sub>oo</sub>l <sub>o</sub>f M<sub>a</sub>th<sub>ema</sub>ti<sub>cs</sub> 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/5625d3657d2773e32a6fb3904700f6f57afb8d12fdbd83602562914a34765412.jpg)


# A<sub>pp</sub>li<sub>ca</sub>ti<sub>ons o</sub>f IPM<sub>s :</sub> F<sub>rom</sub> S<sub>parse</sub> A<sub>pprox</sub>i<sub>ma</sub>ti<sub>ons</sub> t<sub>o</sub> Di<sub>scre</sub>t<sub>e</sub> O<sub>p</sub>ti<sub>ma</sub>l T<sub>ranspor</sub>t

J<sub>ace</sub>k G<sub>on</sub>d<sub>z</sub>i<sub>o</sub> E<sub>ma</sub>il <sub>:</sub> J <sub>.</sub> G<sub>on</sub>d<sub>z</sub> i <sub>o</sub>@<sub>e</sub>d <sub>. ac . u</sub>k URL : htt<sub>p</sub> : / /www <sub>.</sub> maths <sub>.</sub> ed <sub>.</sub> ac <sub>.</sub> uk/ <sup>~</sup> <sub>g</sub>ondz i o 

## O<sub>u</sub>tli<sub>ne</sub>

• M<sub>o</sub>ti<sub>va</sub>ti<sub>on : spars</sub>it<sub>y a</sub> d<sub>es</sub>i<sub>re</sub>d f<sub>ea</sub>t<sub>ure</sub> −→ for example <sub>,</sub> ℓ <sub>1</sub> -regularized least squares ( LAS S O ) 

• 1 <sub>s</sub>t<sub>-or</sub>d<sub>er vs</sub> 2<sub>n</sub>d<sub>-or</sub>d<sub>er me</sub>th<sub>o</sub>d<sub>s</sub> 

• I<sub>nexac</sub>t N<sub>ew</sub>t<sub>on me</sub>th<sub>o</sub>d 

– H<sub>ow muc</sub>h <sub>o</sub>f H<sub>ess</sub>i<sub>an</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on</sub> i<sub>s nee</sub>d<sub>e</sub>d? 

– It<sub>era</sub>ti<sub>ve me</sub>th<sub>o</sub>d<sub>s w</sub>ith <sub>su</sub>it<sub>a</sub>bl<sub>e precon</sub>dit i<sub>oners</sub> 

−→ Newton Conj ugte Gradients 

−→ ( Inexact ) Interior Point Methods 

• A<sub>pp</sub>li<sub>ca</sub>ti<sub>ons</sub> 

• C<sub>onc</sub>l<sub>us</sub>i<sub>ons</sub> 

## S<sub>parse</sub> A<sub>pprox</sub>i<sub>ma</sub>ti<sub>ons</sub>

• St <sub>a</sub>t i<sub>s</sub>t i<sub>cs :</sub> E<sub>s</sub>t i<sub>ma</sub>t<sub>e x</sub> f<sub>rom o</sub>b<sub>serva</sub>t i<sub>ons</sub> 

• M <sub>ac</sub>hi<sub>ne</sub> L<sub>earn</sub>i<sub>ng :</sub> Cl<sub>ass</sub>ifi<sub>ca</sub>ti<sub>ons</sub> SVM<sub>s e</sub>t<sub>c</sub> 

• I<sub>nverse</sub> P<sub>ro</sub>bl<sub>ems</sub> 

• Wavelet-based signal/image reconstruction & restoration 

• Compressed Sensing ( Signal Processing) 

S<sub>uc</sub>h <sub>pro</sub>bl<sub>ems</sub> l<sub>ea</sub>d t<sub>o some</sub> d<sub>ens e, o</sub>ft<sub>en s</sub>t<sub>ruc</sub>t<sub>ure</sub>d<sub>, poss</sub>ibl<sub>y very</sub> large optimization instances (LP QP or N LP) : 

$$
\min _ {x} f (x) + \tau_ {1} \| x \| _ {1} + \tau_ {2} \| L x \| _ {1}
$$

$$
\mathrm{s.t.} A x = b.
$$

C<sub>u</sub>tti<sub>ng-e</sub>d<sub>ge op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> t<sub>ec</sub>h<sub>n</sub>i<sub>ques are nee</sub>d<sub>e</sub>d ! 

Pl<sub>e</sub>th<sub>ora o</sub>f hi<sub>g</sub>hl<sub>y spec</sub>i<sub>a</sub>li<sub>se</sub>d 1 <sub>s</sub>t<sub>-or</sub>d<sub>er me</sub>t h<sub>o</sub>d<sub>s ex</sub>i<sub>s</sub>t <sub>.</sub> W<sub>or</sub>k <sub>o</sub>f Y<sub>u .</sub> N<sub>es</sub>t<sub>erov</sub> S <sub>.</sub> W<sub>r</sub>i<sub>g</sub>ht <sub>an</sub>d <sub>an army o</sub>f f<sub>o</sub>ll<sub>owers .</sub> 

## 1 <sub>s</sub>t<sub>-or</sub>d<sub>er me</sub>th<sub>o</sub>d<sub>s vs</sub> 2<sub>n</sub>d<sub>-or</sub>d<sub>er me</sub>th<sub>o</sub>d<sub>s</sub>

Th<sub>e</sub> 2<sub>n</sub>d<sub>-or</sub>d<sub>er me</sub>th<sub>o</sub>d<sub>s are some</sub>ti<sub>mes cr</sub>iti<sub>c</sub>i<sub>se</sub>d <sub>as unsu</sub>it<sub>a</sub>bl<sub>e :</sub> <sup>“</sup>computing/using the 2nd-order information is too expensive<sup>”</sup> <sub>.</sub> 

A<sub>n un</sub>f<sub>oun</sub>d<sub>e</sub>d <sub>cr</sub>iti<sub>c</sub>i<sub>sm</sub> b<sub>ase</sub>d <sub>on an un</sub>f<sub>a</sub>i<sub>r compar</sub>i<sub>son:</sub> <sub>spec</sub>i<sub>a</sub>li<sub>s e</sub>d 1 <sub>s</sub>t<sub>-or</sub>d<sub>er me</sub>th<sub>o</sub>d<sub>s compare</sub>d <sub>w</sub>ith gen eral (of-the-shelf) 2nd-order methods <sub>.</sub> 

Th<sub>e</sub> 1 <sub>s</sub>t<sub>-or</sub>d<sub>er me</sub>th<sub>o</sub>d<sub>s</sub> h<sub>ave c</sub>l<sub>ear</sub> d<sub>raw</sub>b<sub>ac</sub>k<sub>s :</sub> 

• th<sub>ey s</sub>t<sub>rugg</sub>l<sub>e w</sub>ith <sub>accuracy an</sub>d 

• th<sub>ey wor</sub>k <sub>on</sub>l<sub>y</sub> f<sub>or</sub> t<sub>r</sub>i<sub>v</sub>i<sub>a</sub>l <sub>we</sub>ll <sub>con</sub>diti<sub>one</sub>d <sub>pro</sub>bl<sub>ems .</sub> 

## Th<sub>e spec</sub>i<sub>a</sub>li<sub>se</sub>d 2<sub>n</sub>d<sub>-or</sub>d<sub>er me</sub>th<sub>o</sub>d<sub>s</sub>

<sub>overcome</sub> th<sub>ese</sub> d<sub>raw</sub>b<sub>ac</sub>k<sub>s an</sub>d <sub>are very compe</sub>titi<sub>ve .</sub> 

Thi<sub>s</sub> t<sub>a</sub>lk <sub>w</sub>ill d<sub>emons</sub>t<sub>ra</sub>t<sub>e w</sub>h<sub>y.</sub> 

## ℓ<sub>1- regu</sub>l<sub>ar</sub>i<sub>za</sub>t i<sub>on</sub>

$$
\min _ {x} f (x) + \tau \| x \| _ {1}.
$$

Thi<sub>n</sub>k <sub>o</sub>f LASSO <sub>:</sub> 

$$
\min _ {x} \| A x - b \| _ {2} ^ {2} + \tau \| x \| _ {1}.
$$

## U<sub>ncons</sub>t<sub>ra</sub>i<sub>ne</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on ⇒ easy</sub> S<sub>er</sub>i<sub>ous</sub> I<sub>ssue : non</sub>dif<sub>eren</sub>ti<sub>a</sub>bilit<sub>y o</sub>f $\left. . \right. _ { 1 }$

T<sub>wo p oss</sub>ibl<sub>e</sub> t<sub>r</sub>i<sub>c</sub>k<sub>s :</sub> 

• S<sub>p</sub>litti<sub>ng</sub> $x = u - v$ <sub>w</sub>ith $u , v \geq 0$ 

• S<sub>moo</sub>thi<sub>ng w</sub>ith <sub>pseu</sub>d<sub>o-</sub>H<sub>u</sub>b<sub>er approx</sub>i<sub>ma</sub>ti<sub>on</sub> 

re<sub>p</sub><sup>l</sup>aces $\lVert x \rVert _ { 1 }$ <sub>w</sub>ith $\begin{array} { r } { \psi _ { \mu } ( x ) = \sum _ { i = 1 } ^ { n } ( \sqrt { \mu ^ { 2 } + x _ { i } ^ { 2 } } - \mu ) } \end{array}$ 

## H<sub>u</sub>b<sub>er:</sub>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/280b931fdc556d0cb2e1ea062122c7f9b88595489f662f9d471e982d030f4826.jpg)



B<sub>o</sub>l<sub>ogna</sub> J<sub>anuary</sub> 2023


## C<sub>on</sub>ti<sub>nua</sub>ti<sub>on</sub>

E<sub>m</sub>b<sub>e</sub>d i<sub>nexac</sub>t N<sub>ew</sub>t<sub>on</sub> M<sub>e</sub>th<sub>o</sub>d i<sub>n</sub>t<sub>o a</sub> h<sub>omo</sub>t<sub>opy approac</sub>h <sub>:</sub> 

• I<sub>nequa</sub>liti<sub>es</sub> $u \geq 0 , v \geq 0 \quad \longrightarrow$ <sub>use</sub> IP M re<sub>p</sub><sup>l</sup>ace $z \geq 0$ <sub>w</sub>ith $- \mu l o g z$ <sub>an</sub>d d<sub>r</sub>i<sub>ve</sub> $\mu$ to zero <sub>.</sub> 

• P<sub>seu</sub>d<sub>o-</sub>H<sub>u</sub>b<sub>er regress</sub>i<sub>on</sub> −→ <sub>use con</sub>t i<sub>nua</sub>t i<sub>on</sub> re<sub>p</sub><sup>l</sup>ace $| x _ { i } |$ <sub>w</sub>ith $\mu ( \sqrt { 1 + \frac { x _ { i } ^ { 2 } } { \mu ^ { 2 } } } - 1 )$ <sub>an</sub>d d<sub>r</sub>i<sub>ve</sub> $\mu$ to zero <sub>.</sub> 

## Quest ions :

$\mathrm { H o w ? }$ 

• Th<sub>eory</sub>? 

• P<sub>rac</sub>ti<sub>ce</sub>? 

## H<sub>ow:</sub> U<sub>se approx</sub>i<sub>ma</sub>t<sub>e</sub> H<sub>ess</sub>i<sub>an</sub>

Use 2nd-order information ( Newton direction) <sub>.</sub> 

B<sub>u</sub>t d<sub>o no</sub>t <sub>was</sub>t<sub>e</sub> ti<sub>me on compu</sub>ti<sub>ng exac</sub>t di<sub>rec</sub>ti<sub>on .</sub> 

## U<sub>se</sub> I<sub>nexac</sub>t N<sub>ew</sub>t<sub>on</sub> M<sub>e</sub>th<sub>o</sub>d

D<sub>em</sub>b<sub>o</sub> Ei<sub>sens</sub>t<sub>a</sub>t <sub>an</sub>d St<sub>e</sub>ih<sub>aug</sub> I<sub>nexac</sub>t N<sub>ew</sub>t<sub>on</sub> M<sub>e</sub>th<sub>o</sub>d<sub>s</sub> SIA M J<sub>.</sub> on Numerical A nalysis 1 9 ( 1 982) 400–408 <sub>.</sub> 

B<sub>e</sub>ll<sub>av</sub>i<sub>a</sub> I<sub>nexac</sub>t I<sub>n</sub>t<sub>er</sub>i<sub>or</sub> P<sub>o</sub>i<sub>n</sub>t M<sub>e</sub>th<sub>o</sub>d Journal of Optimization Theory and Appls 96 ( 1 998) 1 09–1 2 1 <sub>.</sub> 

## I<sub>nexac</sub>t N<sub>ew</sub>t<sub>on</sub> M<sub>e</sub>th<sub>o</sub>d

R<sub>ep</sub>l<sub>ace an exac</sub>t N<sub>ew</sub>t<sub>on</sub> di<sub>rec</sub>ti<sub>on</sub> 

$$
\nabla^ {2} f (x) \Delta x = - \nabla f (x)
$$

<sub>w</sub>ith <sub>an</sub> i<sub>nexac</sub>t <sub>one :</sub> 

$$
\nabla^ {2} f (x) \Delta x = - \nabla f (x) + \pmb {r},
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e error r</sub> i<sub>s sma</sub>ll <sub>:</sub> $\| r \| \leq \eta \| \nabla f ( x ) \| , \eta \in ( 0 , 1 )$ 

U<sub>se</sub> it<sub>era</sub>ti<sub>ve me</sub>th<sub>o</sub>d<sub>s o</sub>f li<sub>near a</sub>l<sub>ge</sub>b<sub>ra:</sub> 

• C<sub>on</sub>ti<sub>nua</sub>ti<sub>on</sub> → N<sub>ew</sub>t<sub>on</sub> CG 

• I P M<sub>s</sub> → I<sub>nexac</sub>t I P M → It<sub>era</sub>ti<sub>ve sc</sub>h<sub>emes</sub> f<sub>or</sub> KKT <sub>sys</sub>t<sub>ems</sub> 

IMP <sub>s :</sub> Th<sub>eorem :</sub> S<sub>uppose</sub> th<sub>e</sub> f<sub>eas</sub>ibl<sub>e</sub> IPM f<sub>or Q</sub>P i<sub>s use</sub>d <sub>.</sub> If th<sub>e me</sub>th<sub>o</sub>d <sub>opera</sub>t<sub>es</sub> i<sub>n</sub> th<sub>e sma</sub>ll <sub>ne</sub>i<sub>g</sub>hb<sub>our</sub>h<sub>oo</sub>d 

$$
\mathcal {N} _ {2} (\theta) := \{(x, y, s) \in \mathcal {F} ^ {0}: \| X S e - \mu e \| _ {2} \leq \theta \mu \}
$$

<sub>an</sub>d <sub>uses</sub> th<sub>e</sub> i<sub>n exac</sub>t N<sub>ew</sub>t<sub>on</sub> di<sub>rec</sub>ti<sub>on w</sub>ith $\eta ~ = ~ 0 . 3$ th<sub>en</sub> it conver<sub>g</sub>es in at most 

$$
K = \mathcal {O} (\sqrt {n} \ln (1 / \epsilon)) \quad \mathrm{iterations}.
$$

If th<sub>e me</sub>th<sub>o</sub>d <sub>opera</sub>t<sub>es</sub> i<sub>n</sub> th<sub>e symme</sub>t<sub>r</sub>i<sub>c ne</sub>i<sub>g</sub>hb<sub>our</sub>h<sub>oo</sub>d 

$$
\mathcal {N} _ {S} (\gamma) := \{(x, y, s) \in \mathcal {F} ^ {0}: \gamma \mu \leq x _ {i} s _ {i} \leq (1 / \gamma) \mu \}
$$

<sub>an</sub>d <sub>uses</sub> th<sub>e</sub> i<sub>n exac</sub>t N<sub>ew</sub>t<sub>on</sub> di<sub>rec</sub>ti<sub>on w</sub>ith $\eta ~ = ~ 0 . 0 5$ <sub>,</sub> t h<sub>en</sub> it conver<sub>g</sub>es in at most 

$$
K = \mathcal {O} (\pmb {n} \ln (1 / \epsilon)) \quad \mathrm{iterations}.
$$

## C<sub>on</sub>ti<sub>nua</sub>ti<sub>on:</sub> C<sub>ompresse</sub>d S<sub>ens</sub>i<sub>ng</sub> C<sub>ase</sub>

R<sub>ep</sub>l<sub>ace</sub> 

<sub>w</sub>ith 

$$
\begin{array}{r l} \underset {x} {\min} f (x) = \tau \| W ^ {T} x \| _ {1} + \frac {1}{2} \| A x - b \| _ {2} ^ {2}, & \longrightarrow \pmb {x} _ {\pmb {\tau}} \\ \underset {x} {\min} f _ {\mu} (x) = \tau \psi_ {\mu} (W ^ {T} x) + \frac {1}{2} \| A x - b \| _ {2} ^ {2}, & \longrightarrow \pmb {x} _ {\pmb {\tau}, \pmb {\mu}} \end{array}
$$

Solve approximately a family of problems for a (short) decreasing se<sub>q</sub>uence o<sup>f</sup> $\mu \mathrm { { s } . }$ $\mu _ { 0 } > \mu _ { 1 } > \mu _ { 2 } \cdots$ 

## Theorem (Brief description)

Th<sub>ere ex</sub>i<sub>s</sub>t<sub>s a</sub> $\tilde { \mu }$ <sub>suc</sub>h th<sub>a</sub>t $\forall \mu \le \tilde { \mu }$ th<sub>e</sub> dif<sub>erence o</sub>f th<sub>e</sub> t<sub>wo so</sub>l<sub>u</sub>ti<sub>ons</sub> <sub>sa</sub>t i<sub>s</sub>fi<sub>es</sub> k x<sub>τ µ</sub> − x<sub>τ</sub> k <sub>2</sub> = O (µ <sup>1 /2</sup> ) ∀ τ<sub>,</sub> µ <sub>.</sub> 

## Primal-Dual Newton Conjugate Gradient Method:



F<sub>oun</sub>t<sub>ou</sub>l<sub>a</sub>ki<sub>s an</sub>d G<sub>on</sub>d<sub>z</sub>i<sub>o</sub> A S<sub>econ</sub>d-<sub>or</sub>d<sub>er</sub> M<sub>e</sub>th<sub>o</sub>d f<sub>or</sub> St<sub>rong</sub>l<sub>y</sub> C<sub>onvex</sub> ℓ<sub>1</sub>-<sub>regu</sub>l<sub>ar</sub>i<sub>za</sub>ti<sub>on</sub> P<sub>ro</sub>bl<sub>ems</sub> Mathematical Programming 1 56 ( 20 1 6) 1 89–2 1 9 <sub>.</sub> 





Dassios <sub>,</sub> Fountoulakis and Gondzio A Preconditioner for a Primal-Dual Newton Conj ugate Gradient Method for Compressed Sensing Problems SIA M J on Scientific Computing 37 ( 20 1 5 ) A2 783–A28 1 2 



## E<sub>xamp</sub>l<sub>es</sub>

## E<sub>xamp</sub>l<sub>es o</sub>f ℓ<sub>1-regu</sub>l<sub>ar</sub>i<sub>za</sub>ti<sub>on</sub>

• C<sub>ompresse</sub>d S<sub>ens</sub>i<sub>ng</sub> <sub>w</sub>ith K <sub>.</sub> F<sub>oun</sub>t<sub>ou</sub>l<sub>a</sub>ki<sub>s an</sub>d P<sub>.</sub> Zhl<sub>o</sub>bi<sub>c</sub>h 

$$
\min _ {x} \tau \| x \| _ {1} + \frac {1}{2} \| A x - b \| _ {2} ^ {2}, \quad A \in \mathcal {R} ^ {m \times n}
$$

• Compressed Sensing (Coherent and Redundant Dict <sub>.</sub> ) <sub>w</sub>ith I<sub>.</sub> D<sub>ass</sub>i<sub>os an</sub>d K <sub>.</sub> F<sub>oun</sub>t<sub>ou</sub>l<sub>a</sub>ki<sub>s</sub> 

$$
\min _ {x} \tau \| W ^ {*} x \| _ {1} + \frac {1}{2} \| A x - b \| _ {2} ^ {2}, \quad W \in \mathcal {C} ^ {n \times l}, A \in \mathcal {R} ^ {m \times n}
$$

think of Total Variation 

• Big Data optimization (Machine Learning) LASSO <sub>w</sub>ith K <sub>.</sub> F<sub>oun</sub>t<sub>ou</sub>l<sub>a</sub>ki<sub>s</sub> 

## E<sub>xamp</sub>l<sub>e</sub> 1 <sub>:</sub> C<sub>ompresse</sub>d S<sub>ens</sub>i<sub>ng</sub> <sub>w</sub>ith K <sub>.</sub> F<sub>oun</sub>t<sub>ou</sub>l<sub>a</sub>ki<sub>s an</sub>d P<sub>.</sub> Zhl<sub>o</sub>bi<sub>c</sub>h

L<sub>arge</sub> d<sub>ense qua</sub>d<sub>ra</sub>ti<sub>c op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em :</sub> 

$$
\min _ {x} \tau \| x \| _ {1} + \frac {1}{2} \| A x - b \| _ {2} ^ {2},
$$

<sub>w</sub>h<sub>ere</sub> $A \in \mathcal { R } ^ { m \times n }$ i<sub>s a very spec</sub>i<sub>a</sub>l <sub>ma</sub>t<sub>r</sub>i<sub>x.</sub> 

F<sub>oun</sub>t<sub>ou</sub>l<sub>a</sub>ki<sub>s</sub> G<sub>on</sub>d<sub>z</sub>i<sub>o</sub> Zhl<sub>o</sub>bi<sub>c</sub>h M<sub>a</sub>t<sub>r</sub>i<sub>x</sub>-f<sub>ree</sub> IP M f<sub>or</sub> C<sub>ompresse</sub>d S<sub>ens</sub>i<sub>ng</sub> P<sub>ro</sub>bl<sub>ems</sub> Mathematical Programming Computation 6 ( 20 1 4) pp <sub>.</sub> 1–3 1 <sub>.</sub> 

D<sub>ass</sub>i<sub>os</sub> F<sub>oun</sub>t<sub>ou</sub>l<sub>a</sub>ki<sub>s</sub> G<sub>on</sub>d<sub>z</sub>i<sub>o</sub> A Preconditioner for a Primal-Dual Newton Conj ugate Gradient Method for Compressed Sensing Problems SIA M J on Scientific Computing 37 ( 20 1 5 ) A2 783–A28 1 2 <sub>.</sub> 

Software available at htt<sub>p</sub> : / /www <sub>.</sub> maths <sub>.</sub> ed <sub>.</sub> ac <sub>.</sub> uk/ERGO / 

## Restricted Isometry Property (RIP )

• rows of A are orthogonal to each other (A is built of a subset <sub>o</sub>f <sub>rows o</sub>f <sub>an o</sub>th<sub>onorma</sub>l <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $U \in \mathcal { R } ^ { n \times n } )$ 

$$
A A ^ {T} = I _ {m}.
$$

• <sub>sma</sub>ll <sub>su</sub>b<sub>se</sub>t<sub>s o</sub>f <sub>co</sub> l<sub>umns o</sub>f A <sub>are near</sub>l<sub>y-or</sub>th<sub>ogona</sub>l t<sub>o eac</sub>h other : Restricted Is ometry Property (RIP) 

$$
\| \bar {A} ^ {T} \bar {A} - \frac {m}{n} I _ {k} \| \leq \delta_ {k} \in (0, 1).
$$

C<sub>an</sub>d <sup>`</sup><sub>es</sub> R<sub>om</sub>b<sub>erg an</sub>d T<sub>ao</sub> St<sub>a</sub>bl<sub>e</sub> Si<sub>gna</sub>l R<sub>ecovery</sub> f<sub>rom</sub> I<sub>ncomp</sub>l<sub>e</sub>t<sub>e an</sub>d I<sub>naccura</sub>t<sub>e</sub> M<sub>easuremen</sub>t<sub>s</sub> Comm on Pure and Applied Mathematics 59 (2006) 1 207- 1 233 <sub>.</sub> 

## Restricted Isometr<sub>y</sub> Pro<sub>p</sub>ert<sub>y</sub>

M <sub>a</sub>t<sub>r</sub>i<sub>x</sub> $\bar { A } \in \mathcal { R } ^ { m \times k } \left( k \ll n \right)$ i<sub>s</sub> b<sub>u</sub>ilt <sub>o</sub>f <sub>a su</sub>b<sub>se</sub>t <sub>o</sub>f <sub>co</sub>l<sub>umns</sub> <sub>o</sub>f $A \in \mathcal { R } ^ { m \times n }$ 

$$
A = \boxed { \begin{array}{c c c c c c c c} \hline & & & & & & \\ \hline & & & & & & \\ \hline & & & & & & \\ \hline \end{array} } \quad \longrightarrow \quad \bar {A} =
$$

$$
\bar {A} ^ {T} \bar {A} = \boxed {\quad} \quad \boxed {\quad} = \boxed {\quad} \approx \frac {m}{n} I _ {k}.
$$

Thi<sub>s y</sub>i<sub>e</sub>ld<sub>s a very we</sub>ll <sub>con</sub>diti<sub>one</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em .</sub> 

## P<sub>ro</sub>bl<sub>em</sub> R<sub>e</sub>f<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub>

$$
\min _ {x} \tau \| x \| _ {1} + \frac {1}{2} \| A x - b \| _ {2} ^ {2}
$$

R<sub>ep</sub>l<sub>ace</sub> $x = x ^ { + } - x ^ { - }$ t<sub>o</sub> b<sub>e a</sub>bl<sub>e</sub> t<sub>o use</sub> $| x | = x ^ { + } + x ^ { - }$ U<sub>se</sub> $| x _ { i } | = z _ { i } + z _ { i + n }$ to re<sub>p</sub><sup>l</sup>ace $\lVert x \rVert _ { 1 }$ <sub>w</sub>ith $\| x \| _ { 1 } = 1 _ { 2 n } ^ { T } z$ ( Increases problem dimension from n to $2 n .$ ) 

$$
\min _ {z \geq 0} c ^ {T} z + \frac {1}{2} z ^ {T} Q z,
$$

<sub>w</sub>h<sub>ere</sub> 

$$
Q = \left[ \begin{array}{c} A ^ {T} \\ - A ^ {T} \end{array} \right] [ A - A ] = \left[ \begin{array}{c c} A ^ {T} A & - A ^ {T} A \\ - A ^ {T} A & A ^ {T} A \end{array} \right] \in \mathcal {R} ^ {2 n \times 2 n}
$$

## P<sub>recon</sub>diti<sub>oner</sub>

A<sub>pprox</sub>i<sub>ma</sub>t<sub>e</sub> 

$$
\mathcal {M} = \left[ \begin{array}{c c} A ^ {T} A & - A ^ {T} A \\ - A ^ {T} A & A ^ {T} A \end{array} \right] + \left[ \begin{array}{c c} \Theta_ {1} ^ {- 1} & \\ & \Theta_ {2} ^ {- 1} \end{array} \right]
$$

<sub>w</sub>ith 

$$
\mathcal {P} = \frac {m}{n} \left[ \begin{array}{c c} I _ {n} & - I _ {n} \\ - I _ {n} & I _ {n} \end{array} \right] + \left[ \begin{array}{c c} \Theta_ {1} ^ {- 1} & \\ & \Theta_ {2} ^ {- 1} \end{array} \right].
$$

We expect ( op timal partition) : 

• $k$ <sub>en</sub>t <sub>r</sub>i<sub>es o</sub>f $\Theta ^ { - 1 } \to 0 , \quad k \ll 2 n$ 

$2 n - k$ <sub>en</sub>t <sub>r</sub>i<sub>es o</sub>f $\Theta ^ { - 1 } \to \infty$ 

## S<sub>pec</sub>t<sub>ra</sub>l P<sub>roper</sub>ti<sub>es o</sub>f $\mathcal { P } ^ { - 1 } \mathcal { M }$

## Th<sub>eorem</sub>

• E<sub>xac</sub>tl<sub>y n e</sub>i<sub>genva</sub>l<sub>ues o</sub>f ${ \mathcal { P } } ^ { - 1 } { \mathcal { M } }$ <sub>are</sub> 1 <sub>.</sub> 

• Th<sub>e rema</sub>i<sub>n</sub>i<sub>ng n e</sub>i<sub>genva</sub>l<sub>ues sa</sub>ti<sub>s</sub>f<sub>y</sub> 

$$
| \lambda (\mathcal {P} ^ {- 1} \mathcal {M}) - 1 | \leq \delta_ {k} + \frac {n}{m \delta_ {k} L},
$$

<sub>w</sub>h<sub>ere</sub> $\delta _ { k }$ i<sub>s</sub> th<sub>e</sub> RI P<sub>-cons</sub>t<sub>an</sub>t <sub>an</sub>d 

L i<sub>s a</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>o</sub>f <sup>“</sup>l<sub>arge</sub><sup>”</sup> $( \Theta _ { 1 } + \Theta _ { 2 } ) ^ { - 1 }$ 

F<sub>oun</sub>t<sub>ou</sub>l<sub>a</sub>ki<sub>s</sub> G<sub>on</sub>d<sub>z</sub>i<sub>o</sub> Zhl<sub>o</sub>bi<sub>c</sub>h 

M<sub>a</sub>t<sub>r</sub>i<sub>x</sub>-f<sub>ree</sub> IP M f<sub>or</sub> C<sub>ompresse</sub>d S<sub>ens</sub>i<sub>ng</sub> P<sub>ro</sub>bl<sub>ems</sub> 

Mathematical Programming Computation 6 ( 20 1 4) pp <sub>.</sub> 1–3 1 <sub>.</sub> 

## P<sub>recon</sub>diti<sub>on</sub>i<sub>ng</sub>


Matrix<sub>−</sub>vector <sub>p</sub>rod ucts <sub>p</sub>er CG/PCG cal l


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/512c915bf035363717b2b34c9d4a7e58c025623068c112cd8ff27ae690136af9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/e75645c6798f71f25461d21c55996d06e4fb3b8642e4f995292ddfea201ef2f9.jpg)


−→ <sub>goo</sub>d <sub>c</sub>l<sub>us</sub>t<sub>er</sub>i<sub>ng o</sub>f <sub>e</sub>i<sub>genva</sub>l<sub>ues</sub> 

<sub>m</sub>f - I PM <sub>compares</sub> f<sub>avoura</sub>bl<sub>y w</sub>ith N<sub>e s</sub>tA <sub>on easy pro</sub>b<sub>s</sub> (Ne stA : Becker Bobin and Cand <sup>´</sup>es) <sub>.</sub> 

## E<sub>xamp</sub>l<sub>e</sub> 2 <sub>:</sub> Si<sub>mp</sub>l<sub>e</sub> t<sub>es</sub>t f<sub>or</sub> $\ell _ { 1 }$ <sub>-regu</sub>l<sub>ar</sub>i<sub>za</sub>t i<sub>on</sub>

$$
\min _ {x} \tau \| x \| _ {1} + \| A x - b \| _ {2} ^ {2}
$$

S<sub>pec</sub>i<sub>a</sub>l <sub>ma</sub>t<sub>r</sub>i<sub>x g</sub>i<sub>ven</sub> i<sub>n</sub> SVD f<sub>orm</sub> $A = U \Sigma V ^ { T }$ <sub>w</sub>h<sub>ere</sub> U <sub>an</sub>d V <sub>are</sub> <sub>pro</sub> d<sub>uc</sub>t<sub>s o</sub>f Gi<sub>vens ro</sub>t <sub>a</sub>ti<sub>ons .</sub> Th<sub>e user con</sub>t<sub>ro</sub>l<sub>s :</sub> 

• th<sub>e con</sub>diti<sub>on num</sub>b<sub>er</sub> $\kappa ( A )$ 

• th<sub>e spars</sub>it<sub>y o</sub>f <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $A .$ 

M <sub>a</sub>tl<sub>a</sub>b <sub>genera</sub>t<sub>or :</sub> htt<sub>p</sub> s : / /www <sub>.</sub> maths <sub>.</sub> ed <sub>.</sub> ac <sub>.</sub> uk/ERGO /t r i l l i on/ 

## Excessive Computational Tests (4 mths of CPU)

• FISTA ( Fast Iterative Shrinkage-Thresholding Algorithm) 

• PC D M ( Parallel Coordinate Descent Method) 

• P S Sgb ( Proj ected Scaled Subgradient Gafni- Bertsekas) 

• pdNCG (primal-dual Newton Conj ugate Gradient) 

Th<sub>e</sub> 1 <sub>s</sub>t <sub>or</sub>d<sub>er me</sub>th<sub>o</sub>d<sub>s :</sub> 

• <sub>wor</sub>k <sub>we</sub>ll if th<sub>e con</sub>diti<sub>on num</sub>b<sub>er</sub> $\kappa ( A ) \leq 1 0 ^ { 2 }$ 

• stru<sub>gg</sub><sup>l</sup>e w<sup>h</sup>en $\kappa ( A ) \geq 1 0 ^ { 3 }$ 

• <sub>s</sub>t<sub>a</sub>ll <sub>w</sub>h<sub>en</sub> $\kappa ( A ) \geq 1 0 ^ { 4 }$ 

The 2nd order method (pdNCG <sub>,</sub> diagonal preconditioner) : 

• <sub>wor</sub>k<sub>s we</sub>ll if th<sub>e con</sub>diti<sub>on num</sub>b<sub>er</sub> $\kappa ( A ) \leq 1 0 ^ { 6 }$ 

## L<sub>e</sub>t <sub>us go</sub> bi<sub>g: a</sub> t<sub>r</sub>illi<sub>on</sub> $( 2 ^ { 4 0 } \approx 1 0 ^ { 1 2 } )$ <sub>var</sub>i<sub>a</sub>bl<sub>es</sub>

<table><tr><td>n (billions)</td><td>Processors</td><td>Memory (TB)</td><td>time (s)</td></tr><tr><td>1</td><td>64</td><td>0.192</td><td>1923</td></tr><tr><td>4</td><td>256</td><td>0.768</td><td>1968</td></tr><tr><td>16</td><td>1024</td><td>3.072</td><td>1986</td></tr><tr><td>64</td><td>4096</td><td>12.288</td><td>1970</td></tr><tr><td>256</td><td>16384</td><td>49.152</td><td>1990</td></tr><tr><td>1,024</td><td>65536</td><td>196.608</td><td>2006</td></tr></table>

ARCHER (ranked 25 on t op500 <sub>.</sub> c om 1 1 March 20 1 5) Linpack Performance ( Rmax) 1 <sub>,</sub> 642 <sub>.</sub> 54 TFlop/s Theoretical Peak ( Rpeak) 2 <sub>,</sub> 550 <sub>.</sub> 53 TFlop/s 

F<sub>oun</sub>t<sub>ou</sub>l<sub>a</sub>ki<sub>s an</sub>d G<sub>on</sub>d<sub>z</sub>i<sub>o</sub> 

P<sub>er</sub>f<sub>ormance o</sub>f Fi<sub>rs</sub>t- <sub>an</sub>d S<sub>econ</sub>d- O<sub>r</sub>d<sub>er</sub> M<sub>e</sub>th<sub>o</sub> d<sub>s</sub> f<sub>or</sub> ℓ<sub>1</sub> -<sub>regu</sub>l<sub>ar</sub>i<sub>ze</sub>d L<sub>eas</sub>t S<sub>quares</sub> P<sub>ro</sub>bl<sub>ems ,</sub> Computational Optimization and Applications 65 ( 20 1 6 ) 605–635 <sub>.</sub> 

## M<sub>ore</sub> E<sub>xamp</sub>l<sub>es o</sub>f S<sub>parse</sub> A<sub>pprox</sub>i<sub>ma</sub>ti<sub>ons</sub>

• S<sub>parse</sub> A<sub>pprox</sub>i<sub>ma</sub>ti<sub>ons w</sub>ith IP M<sub>s</sub> → ℓ <sub>1-regu</sub>l<sub>ar</sub>i<sub>ze</sub>d <sub>pro</sub>bl<sub>ems</sub> <sub>wor</sub>k <sub>w</sub>ith V<sub>.</sub> D<sub>e</sub> Si<sub>mone ,</sub> D <sub>.</sub> di S<sub>era</sub>fi<sub>no ,</sub> S <sub>.</sub> P<sub>oug</sub>k<sub>a</sub>ki<sub>o</sub>ti<sub>s</sub> M<sub>.</sub> Vi<sub>o</sub>l<sub>a</sub> 

• Di<sub>scre</sub>t<sub>e</sub> O<sub>p</sub>ti<sub>ma</sub>l T<sub>ranspor</sub>t <sub>w</sub>ith I P M<sub>s</sub> → l<sub>arge</sub> b<sub>u</sub>t hi<sub>g</sub>hl<sub>y s</sub>t<sub>ruc</sub>t<sub>ure</sub>d <sub>wor</sub>k <sub>w</sub>ith F Z<sub>ane</sub>tti 

## M<sub>ore</sub> S<sub>parse</sub> A<sub>pprox</sub>i<sub>ma</sub>ti<sub>ons:</sub> U<sub>se</sub> IPM<sub>s</sub>

P<sub>ro</sub>bl<sub>ems o</sub>f th<sub>e</sub> f<sub>orm</sub> 

$$
\begin{array}{r l} & {\min f (x) + \tau_ {1} \| x \| _ {1} + \tau_ {2} \| L x \| _ {1}} \\ & {\mathrm{s.t.} A x = b.} \end{array}
$$

• S<sub>parse por</sub>tf<sub>o</sub>li<sub>o se</sub>l<sub>ec</sub>ti<sub>on</sub> <sub>compar</sub>i<sub>son w</sub>ith S<sub>p</sub>lit B<sub>regman me</sub>th<sub>o</sub>d 

Cl<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on mo</sub>d<sub>e</sub>l<sub>s</sub> f<sub>or</sub> f<sub>unc</sub>t <sup>’</sup> l M <sub>agne</sub>ti<sub>c</sub> R<sub>esonance</sub> I<sub>mag</sub>i<sub>ng</sub> <sub>compar</sub>i<sub>son w</sub>ith FISTA <sub>an</sub>d ADMM 

• TV<sub>-</sub>b<sub>ase</sub>d P<sub>o</sub>i<sub>sson</sub> I<sub>mage</sub> R<sub>es</sub>t<sub>ora</sub>ti<sub>on</sub> <sub>compar</sub>i<sub>son w</sub>ith PDAL 

• Li<sub>near</sub> Cl<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on v</sub>i<sub>a</sub> R<sub>egu</sub>l<sub>ar</sub>i<sub>ze</sub>d L<sub>og</sub>i<sub>s</sub>ti<sub>c</sub> R<sub>egress</sub>i<sub>on</sub> <sub>compar</sub>i<sub>son w</sub>ith <sub>new</sub>GLMNET <sub>an</sub>d ADMM 

D<sub>e</sub> Si<sub>mone</sub> di S<sub>era</sub>fi<sub>no</sub> G<sub>on</sub>d<sub>z</sub>i<sub>o</sub> P<sub>oug</sub>k<sub>a</sub>ki<sub>o</sub>ti<sub>s</sub> Vi<sub>o</sub>l<sub>a</sub> 

S<sub>parse</sub> A<sub>pprox</sub>i<sub>ma</sub>ti<sub>ons w</sub>ith I<sub>n</sub>t<sub>er</sub>i<sub>or</sub> P<sub>o</sub>i<sub>n</sub>t M<sub>e</sub>th<sub>o</sub>d<sub>s</sub> 

SIA M Review 64 ( 202 2 ) pp <sub>.</sub> 954–988 <sub>.</sub> ht tp s : / / arx iv <sub>.</sub> org/ ab s / 2 1 0 2 <sub>.</sub> 1 3608 

## E<sub>xamp</sub>l<sub>e</sub> 3 <sub>:</sub> Bi<sub>nary</sub> Cl<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on o</sub>f fMRI D<sub>a</sub>t<sub>a</sub>

$$
\min _ {w} \frac {1}{2 s} \left\| D w - \hat {y} \right\| ^ {2} + \tau_ {1} \left\| w \right\| _ {1} + \tau_ {2} \left\| L w \right\| _ {1}
$$

<sub>w</sub>h<sub>ere :</sub> $\tau _ { 1 } , \tau _ { 2 } > 0 , \quad \| L w \| _ { 1 }$ i<sub>s a</sub> di<sub>scre</sub>t<sub>e an</sub>i<sub>so</sub>t<sub>rop</sub>i<sub>c</sub> TV <sub>o</sub>f <sub>w</sub> 

<sub>an</sub>d $L = [ L _ { x } ^ { T } L _ { y } ^ { T } L _ { z } ^ { T } ] ^ { T } \in \mathcal { R } ^ { l \times q }$ <sub>are</sub> th<sub>e</sub> fi<sub>rs</sub>t<sub>-or</sub>d<sub>er</sub> f<sub>orwar</sub>d fi<sub>n</sub>it<sub>e</sub> dif<sub>erences</sub> i<sub>n x , y , z .</sub> 

## Cl<sub>ass</sub>ifi<sub>ca</sub>ti<sub>on mo</sub>d<sub>e</sub>l<sub>s</sub> f<sub>or</sub> fMRI

Comparison of IPM <sub>,</sub> FISTA and ADM M (opt tol $1 0 ^ { - 5 } )$ <sub>.</sub> <sup>W</sup>e re<sub>p</sub> ort : 

• classification accuracy (AC C ) <sub>,</sub> 

• corrected pairwis e overlap (CO RR OVR) ; <sub>measures</sub> th<sub>e</sub> <sup>“</sup><sub>s</sub>t<sub>a</sub>bilit<sub>y</sub><sup>”</sup> <sub>o</sub>f <sub>eac</sub>h <sub>voxe</sub>l <sub>se</sub>l<sub>ec</sub>ti<sub>on</sub> 

• s o luti on density ( D EN ) <sub>.</sub> 

<table><tr><td>Algorithm</td><td><eq>\tau_1 = \tau_2</eq></td><td>ACC</td><td>CORR OVR</td><td>DEN</td></tr><tr><td rowspan="3">IP-PMM</td><td><eq>10^{-2}</eq></td><td>86.16 ± 7.11</td><td>43.47 ± 9.09</td><td>20.56 ± 6.63</td></tr><tr><td><eq>5 \cdot 10^{-2}</eq></td><td>84.90 ± 4.80</td><td>62.70 ± 10.39</td><td>3.77 ± 0.84</td></tr><tr><td><eq>10^{-1}</eq></td><td>82.29 ± 6.22</td><td>82.60 ± 9.24</td><td>2.49 ± 0.34</td></tr><tr><td rowspan="3">FISTA</td><td><eq>10^{-2}</eq></td><td>86.90 ± 5.01</td><td>5.43 ± 0.43</td><td>88.97 ± 0.71</td></tr><tr><td><eq>5 \cdot 10^{-2}</eq></td><td>84.15 ± 5.92</td><td>65.50 ± 2.68</td><td>19.36 ± 0.86</td></tr><tr><td><eq>10^{-1}</eq></td><td>81.62 ± 7.58</td><td>80.44 ± 5.72</td><td>5.14 ± 0.44</td></tr><tr><td rowspan="3">ADMM</td><td><eq>10^{-2}</eq></td><td>86.46 ± 6.91</td><td>0.03 ± 0.01</td><td>98.70 ± 0.03</td></tr><tr><td><eq>5 \cdot 10^{-2}</eq></td><td>85.57 ± 5.37</td><td>0.15 ± 0.04</td><td>97.97 ± 0.05</td></tr><tr><td><eq>10^{-1}</eq></td><td>82.07 ± 6.51</td><td>0.26 ± 0.13</td><td>97.50 ± 0.19</td></tr></table>

W<sub>e wan</sub>t<sub>:</sub> ACC <sub>an</sub>d CORR OVR <sub>c</sub>l<sub>ose</sub> t<sub>o</sub> 1 00 <sub>an</sub>d <sub>sma</sub>ll DEN <sub>.</sub> 

## Classification models for fMRI (cont <sup>’</sup>d)

P<sub>er</sub>f<sub>ormance compar</sub>i<sub>son</sub> i<sub>n</sub> t<sub>erms o</sub>f <sub>e</sub>l<sub>apse</sub>d ti<sub>me :</sub> 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/f2e24ff2357b5ef163ec7d65115f2158b32368df046aff01643a4ca4612408cc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/5890ac4e58c59ef289da766fe2d1cabed26ea82e14ecb5d20fe9f6a77d683193.jpg)



E<sub>vo</sub>l<sub>u</sub>ti<sub>on o</sub>f ACC DEN <sub>an</sub>d CORR OVR <sub>w</sub>ith ti<sub>me;</sub> IP-PM M ( left ) and FISTA ( right ) <sub>.</sub>



W<sub>e repor</sub>t <sub>average measures w</sub>ith 95% <sub>con</sub>fid<sub>ence</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>s .</sub>


## O<sub>p</sub>ti<sub>ma</sub>l T<sub>ranspor</sub>t

Si<sub>gn</sub>ifi<sub>can</sub>t <sub>researc</sub>h i<sub>n</sub>t<sub>eres</sub>t <sub>:</sub> Gaspard Monge ( 1 78 1 ) Leonid Kantorovich ( 1 942) Nobel Prize in 1 975 Alessio Figalli (2008) Fields Medal in 2 0 1 8 

G<sub>oo</sub>d <sub>rea</sub>di<sub>ng:</sub> 

F <sub>.</sub> S<sub>an</sub>t<sub>am</sub>b<sub>rog</sub>i<sub>o</sub> O<sub>p</sub>ti<sub>ma</sub>l T<sub>ranspor</sub>t f<sub>or</sub> A<sub>pp</sub>li<sub>e</sub>d M<sub>a</sub>th<sub>ema</sub>ti<sub>c</sub>i<sub>ans</sub> Bi<sub>r</sub>kh<sub>auser</sub> B<sub>ase</sub>l 20 1 6 <sub>.</sub> 

G <sub>.</sub> P<sub>eyr</sub> <sup>´</sup><sub>e an</sub>d M<sub>.</sub> C<sub>u</sub>t<sub>ur</sub>i C<sub>ompu</sub>t<sub>a</sub>ti<sub>ona</sub>l O<sub>p</sub>ti<sub>ma</sub>l T<sub>ranspor</sub>t <sub>:</sub> With A<sub>pp</sub>li<sub>ca</sub>ti<sub>ons</sub> t<sub>o</sub> D<sub>a</sub>t<sub>a</sub> Science <sub>.</sub> Foundations and Trends in Machine Learning 1 1 (20 1 9) N<sub>o</sub> 5<sub>-</sub>6 <sub>, pp .</sub> 355–607<sub>.</sub> 

## E<sub>xamp</sub>l<sub>e</sub> 4 <sub>:</sub> Di<sub>scre</sub>t<sub>e</sub> O<sub>p</sub>ti<sub>ma</sub>l T<sub>ranspor</sub>t

K<sub>an</sub>t<sub>orov</sub>i<sub>c</sub>h f<sub>ormu</sub>l<sub>a</sub>ti<sub>on o</sub>f th<sub>e</sub> di<sub>scre</sub>t<sub>e</sub> O<sub>p</sub>ti<sub>ma</sub>l T<sub>ranspor</sub>t <sub>pro</sub>bl<sub>em :</sub> <sub>g</sub>iven a startin<sub>g</sub> vector a $\in \mathcal { R } _ { + } ^ { m }$ <sub>an</sub>d <sub>a</sub> fi<sub>na</sub>l <sub>vec</sub>t<sub>or</sub> $\mathbf { b } \in \mathcal { R } _ { + } ^ { n }$ , <sub>suc</sub>h th<sub>a</sub>t $\sum { \bf a } _ { j } = \sum { \bf b } _ { j }$ fi<sub>n</sub>d <sub>a coup</sub>li<sub>ng ma</sub>t<sub>r</sub>i<sub>x</sub> P i<sub>ns</sub>id<sub>e</sub> th<sub>e se</sub>t 

$$
U (\mathbf {a}, \mathbf {b}) = \left\{\mathcal {P} \in \mathcal {R} _ {+} ^ {m \times n}, \mathcal {P} \mathbf {e} _ {n} = \mathbf {a}, \mathcal {P} ^ {T} \mathbf {e} _ {m} = \mathbf {b} \right\}
$$

th<sub>a</sub>t i<sub>s op</sub>ti<sub>ma</sub>l <sub>w</sub>ith <sub>respec</sub>t t<sub>o a cer</sub>t<sub>a</sub>i<sub>n cos</sub>t <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $\mathcal { C } \in \mathcal { R } _ { + } ^ { m \times n }$ i <sub>. e .</sub> fi<sub>n</sub>d th<sub>e so</sub>l<sub>u</sub>ti<sub>on o</sub>f th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em</sub> 

$$
\min _ {\mathcal {P} \in U (\mathbf {a}, \mathbf {b})} \sum_ {i, j} \mathcal {C} _ {i j} \mathcal {P} _ {i j}.
$$

M<sub>ove</sub> th<sub>e mass</sub> i<sub>n</sub> th<sub>e con</sub>fi<sub>gura</sub>ti<sub>on a</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e con</sub>fi<sub>gura</sub>ti<sub>on</sub> b <sub>.</sub> 

## D iscrete Optimal Transport (cont <sup>’</sup>d)

W<sub>e can rewr</sub>it<sub>e</sub> th<sub>e op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em as a s</sub>t<sub>an</sub>d<sub>ar</sub>d LP <sub>:</sub> 

$$
\begin{array}{r l} \underset {\mathbf {p} \in \mathcal {R} ^ {m n}} {\min} & \mathbf {c} ^ {T} \mathbf {p} \\ \mathrm{s.t.} & \left[ \begin{array}{l} \mathbf {e} _ {n} ^ {T} \otimes I _ {m} \\ I _ {n} \otimes \mathbf {e} _ {m} ^ {T} \end{array} \right] \mathbf {p} = \left[ \begin{array}{l} \mathbf {a} \\ \mathbf {b} \end{array} \right] = \mathbf {f}, \\ & \mathbf {p} \geq 0, \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $\bigotimes$ d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> K<sub>ronec</sub>k<sub>er pro</sub>d<sub>uc</sub>t $\mathbf { c } \in \mathcal { R } ^ { m n }$ <sub>an</sub>d $\mathbf { p } \in \mathcal { R } ^ { m n }$ <sub>are</sub> th<sub>e vec</sub>t<sub>or</sub>i<sub>ze</sub>d <sub>vers</sub>i<sub>ons o</sub>f $\mathcal { C }$ <sub>an</sub>d $\mathcal { P } _ { i }$ res<sub>p</sub>ect<sup>i</sup>ve<sup>l</sup><sub>y</sub> $\mathbf { c } = \mathrm { v e c } ( \mathcal { C } )$ <sub>an</sub>d $\mathbf { p } = \mathrm { v e c } ( \mathcal { P } )$ 

LP <sub>w</sub>ith $m + n$ <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s a</sub>nd $m \times n$ <sub>var</sub>i <sub>a</sub> b l<sub>es.</sub> 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/ffa0796449e44d73ba597dd27b6f0a46d6b2981de38353d70d643afd253055ff.jpg)


## S<sub>ma</sub>ll OT E<sub>xamp</sub>l<sub>e</sub>

M<sub>ove</sub> th<sub>e mass</sub> i<sub>n</sub> th<sub>e re</sub>d <sub>con</sub>fi<sub>gura</sub>ti<sub>on</sub> i<sub>n</sub>t<sub>o</sub> th<sub>e</sub> bl<sub>ue con</sub>fi<sub>gura</sub>ti<sub>on .</sub> Ri<sub>g</sub>ht fi<sub>gure :</sub> th<sub>e correspon</sub>di<sub>ng</sub> bi<sub>par</sub>tit<sub>e grap</sub>h <sub>.</sub> → S<sub>parse so</sub>l<sub>u</sub>ti<sub>on</sub> ! 

## IPM S<sub>pec</sub>i<sub>a</sub>li<sub>ze</sub>d f<sub>or</sub> Di<sub>scre</sub>t<sub>e</sub> OT P<sub>ro</sub>bl<sub>ems</sub>

• I<sub>gnore</sub> <sup>“</sup>l<sub>ong</sub><sup>”</sup> <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> A −<sup>→</sup> use co<sup>l</sup>u<sup>mn</sup>-ge<sup>n</sup>e<sup>r</sup>a<sup>ti</sup>o<sup>n</sup>-<sup>t</sup>ype app<sup>r</sup>oac<sup>h</sup> 

• W<sub>or</sub>k <sub>w</sub>ith <sub>expec</sub>t<sub>e</sub>d <sup>“</sup><sub>sparse</sub><sup>”</sup> <sub>so</sub>l<sub>u</sub>ti<sub>on se</sub>t −→ d<sub>o no</sub>t <sub>up</sub>d<sub>a</sub>t<sub>e a</sub>ll <sub>var</sub>i<sub>a</sub>bl<sub>es x</sub> 

• U<sub>se s</sub>i<sub>mp</sub>l<sub>ex-</sub>t<sub>ype pr</sub>i<sub>c</sub>i<sub>ng mec</sub>h<sub>an</sub>i<sub>sm</sub> −→ <sub>up</sub>d<sub>a</sub>t<sub>e</sub> d<sub>ua</sub>l <sub>s</sub>l<sub>ac</sub>k<sub>s on</sub>l<sub>y</sub> f<sub>or a su</sub>b<sub>se</sub>t <sub>o</sub>f <sub>var</sub>i<sub>a</sub>bl<sub>es x</sub> 

• Si<sub>mp</sub>lif<sub>y norma</sub>l <sub>equa</sub>ti<sub>ons</sub> −→ re<sub>p</sub><sup>l</sup>ace $\textstyle \sum _ { j = 1 } ^ { N } \theta _ { j } A _ { j } A _ { j } ^ { T }$ <sub>w</sub>ith $\textstyle \sum _ { j \in { \cal S } } \theta _ { j } A _ { j } A _ { j } ^ { T }$ <sub>w</sub>h<sub>ere</sub> S i<sub>s a</sub> lik<sub>e</sub>l<sub>y</sub> <sup>“</sup><sub>sparse</sub><sup>”</sup> <sub>so</sub>l<sub>u</sub>t i<sub>on se</sub>t 

• P<sub>recon</sub>diti<sub>on</sub> Ch<sub>o</sub>l<sub>es</sub>k<sub>y ma</sub>t<sub>r</sub>i<sub>x o</sub>f th<sub>e norma</sub>l <sub>equa</sub>ti<sub>ons</sub> −→ k<sub>eep</sub> it <sub>sparse a</sub>t <sub>a</sub>ll ti<sub>mes</sub> 

## T<sub>es</sub>t <sub>examp</sub>l<sub>es</sub> f<sub>rom</sub> D OT<sub>mar</sub>k <sub>co</sub>ll<sub>ec</sub>ti<sub>on</sub>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/49ab739eac2e0e02186970235ce2c9b96d7e454e952fcadc44124a15606e1665.jpg)



C l<sub>ass</sub> 1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/7f9ea05b1cb2c6e4f1900295b62e2f9a2ad4d1ff8d730aae02ad66fa6cb7d28d.jpg)



Cl<sub>ass</sub> 2


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/88c93c064f28aa979065a5a4a7f2074568b86e15bf68c47d18ecf4ce20c22210.jpg)



Cl<sub>ass</sub> 3


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/0fd34682565c072786057e9d17e85d16f07cc7c6b063f48271b6151ee9686776.jpg)



Cl<sub>ass</sub> 4


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/cd8ed07d290e45b25bb7d5b6a699dcef863f35e47f9078e7bebab0700e5849da.jpg)



Cl<sub>ass</sub> 5


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/bb706a86f9b4fb78bd6fd945050d3bf7ffc1193e26317b1461bd038cdecec7ca.jpg)



Cl<sub>ass</sub> 6


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/443e318285b60b7e126bba4dbda3fbbe05cd65f78ecaaa5c13806f77def2a1be.jpg)



Cl<sub>ass</sub> 7


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/b25d12e92a509d714d8f8c41190a5e28bc6b3f9c636d6e4e4fed571bb6f534c9.jpg)



Cl<sub>ass</sub> 8


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/f87ed7dec2472d881b1da828d581084c2ab10f51959025f315a9e9c7d5c4525e.jpg)



Cl<sub>ass</sub> 9


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/d372ab606170a7b80ed125e7d816e4b26e40a7bf5b225de689818aa357e5f61d.jpg)



C l<sub>ass</sub> 1 0


F<sub>or</sub> th<sub>e reso</sub>l<sub>u</sub>ti<sub>on r</sub> th<sub>e</sub> LP h<sub>as</sub> $2 r ^ { 2 }$ <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s a</sub>nd $r ^ { 4 }$ <sub>var</sub>i <sub>a</sub> b l<sub>es.</sub> 

F<sub>or r</sub> = 32 <sub>:</sub> 2 048 <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s an</sub>d 1 <sub>m</sub>illi<sub>on var</sub>i<sub>a</sub>bl<sub>es ;</sub> 

F<sub>or r</sub> = 64 <sub>:</sub> 8 1 92 <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s an</sub>d 1 6 <sub>.</sub> 8 <sub>m</sub>illi<sub>on var</sub>i<sub>a</sub>bl<sub>es ;</sub> 

F<sub>or r</sub> = 1 28 <sub>:</sub> 32 768 <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s an</sub>d 268 <sub>.</sub> 4 <sub>m</sub>illi<sub>on var</sub>i<sub>a</sub>bl<sub>es ;</sub> 

F<sub>or</sub> $r = 2 5 6 \colon$ 1 3 1 072 <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s an</sub>d 4 <sub>.</sub> 295 billi<sub>on var</sub>i<sub>a</sub>bl<sub>es .</sub> 

## D iscrete Optimal Transport (cont <sup>’</sup>d)

## D O T<sub>mar</sub>k t<sub>es</sub>t <sub>co</sub>ll<sub>ec</sub>ti<sub>on :</sub>

S<sub>c</sub>h<sub>r</sub>i<sub>e</sub>b<sub>er</sub> S<sub>c</sub>h<sub>u</sub>h<sub>mac</sub>h<sub>er an</sub>d G<sub>o</sub>tt<sub>sc</sub>hli<sub>c</sub>h 

D OTmark - A Benchmark for Discrete Optimal Transport <sub>,</sub> IEEE A ccess<sub>,</sub> 5 ( 20 1 7) <sub>,</sub> pp <sub>.</sub> 2 7 1–282 <sub>.</sub> 

## S<sub>o</sub>ft<sub>wares compare</sub>d <sub>:</sub>

• C <sub>u</sub>t <sub>ur</sub>i Si<sub>n</sub>kh<sub>orn</sub> di<sub>s</sub>t<sub>ances :</sub> Li<sub>g</sub>ht<sub>spee</sub>d <sub>compu</sub>t<sub>a</sub>ti<sub>on o</sub>f <sub>op</sub>ti<sub>ma</sub>l t<sub>ranspor</sub>t <sub>,</sub> Proc <sub>.</sub> NIPS<sub>,</sub> ( 20 1 3 ) <sub>,</sub> pp <sub>.</sub> 2 292–2300 <sub>.</sub> 

• G<sub>o</sub>tt<sub>sc</sub>hli<sub>c</sub>h <sub>an</sub>d S<sub>c</sub>h<sub>u</sub>h<sub>mac</sub>h<sub>er</sub> Th<sub>e</sub> Sh<sub>or</sub>tli<sub>s</sub>t M<sub>e</sub>th<sub>o</sub>d f<sub>or</sub> F<sub>as</sub>t C<sub>ompu</sub>t<sub>a</sub>ti<sub>on o</sub>f th<sub>e</sub> E<sub>ar</sub>th M<sub>over</sub> <sup>’</sup> <sub>s</sub> Di<sub>s</sub>t<sub>ance an</sub>d Fi<sub>n</sub>di<sub>ng</sub> O<sub>p</sub>ti<sub>ma</sub>l S<sub>o</sub>l<sub>u</sub>ti<sub>ons</sub> t<sub>o</sub> T<sub>ranspor</sub>t<sub>a</sub>ti<sub>on</sub> P<sub>ro</sub>bl<sub>ems</sub> PLoS ONE 9 ( 20 1 4) p <sub>.</sub> e 1 1 02 1 4 <sub>.</sub> 

• M<sub>er</sub>i<sub>go</sub>t A M<sub>u</sub>lti<sub>sca</sub>l<sub>e</sub> A<sub>pproac</sub>h t<sub>o</sub> O<sub>p</sub>ti<sub>ma</sub>l T<sub>ranspor</sub>t Computer Graphics Forum<sub>,</sub> 30 ( 20 1 1 ) <sub>,</sub> pp <sub>.</sub> 1 583–1 592 <sub>.</sub> 

• N<sub>e</sub>t<sub>wor</sub>k Si<sub>mp</sub>l<sub>ex</sub> M<sub>e</sub>th<sub>o</sub>d IBM ILOG CPLEX<sub>.</sub> ht t<sub>p</sub> s : / /www <sub>.</sub> ibm <sub>.</sub> c om/ anal<sub>y</sub>t i c s / c<sub>p</sub>l ex- o<sub>p</sub>t imi z er <sub>.</sub> 

• K<sub>ovacs</sub> Minimum-cost flow algorithms : An experimental evaluation OMS 30 ( 1 ) : 94–1 2 7 <sub>.</sub> ht t<sub>p</sub> s : / / l emon <sub>.</sub> c s <sub>.</sub> e lt e <sub>.</sub> hu/t rac / l emon <sub>.</sub> 

## C<sub>ompar</sub>i<sub>son:</sub> S<sub>parse</sub>IPM <sub>vs</sub> C<sub>p</sub>l<sub>ex</sub> N<sub>e</sub>t<sub>wor</sub>k

<table><tr><td></td><td colspan="4">Res = 32 × 32</td><td colspan="4">Res = 64 × 64</td></tr><tr><td>Class</td><td>Iter</td><td>IPM t</td><td>Cplex t</td><td>RWE</td><td>Iter</td><td>IPM t</td><td>Cplex t</td><td>RWE</td></tr><tr><td>1</td><td>11.4</td><td>0.35</td><td>0.62</td><td>1.2e-07</td><td>14.4</td><td>2.18</td><td>20.92</td><td>5.5e-08</td></tr><tr><td>2</td><td>11.7</td><td>0.39</td><td>0.60</td><td>1.4e-07</td><td>18.1</td><td>3.46</td><td>20.64</td><td>4.5e-08</td></tr><tr><td>3</td><td>15.9</td><td>0.59</td><td>0.61</td><td>2.4e-08</td><td>26.8</td><td>6.02</td><td>20.83</td><td>2.1e-08</td></tr><tr><td>4</td><td>20.3</td><td>0.85</td><td>0.57</td><td>2.0e-08</td><td>38.4</td><td>9.69</td><td>20.69</td><td>2.1e-08</td></tr><tr><td>5</td><td>25.6</td><td>1.16</td><td>0.61</td><td>1.4e-08</td><td>40.8</td><td>10.78</td><td>21.84</td><td>1.6e-08</td></tr><tr><td>6</td><td>18.8</td><td>0.72</td><td>0.64</td><td>3.3e-08</td><td>36.2</td><td>9.04</td><td>23.25</td><td>1.3e-08</td></tr><tr><td>7</td><td>30.8</td><td>1.47</td><td>0.57</td><td>3.8e-08</td><td>72.2</td><td>39.11</td><td>21.80</td><td>2.3e-08</td></tr><tr><td>8</td><td>17.4</td><td>0.65</td><td>0.58</td><td>3.8e-08</td><td>52.5</td><td>21.69</td><td>18.55</td><td>8.7e-08</td></tr><tr><td>9</td><td>14.9</td><td>0.52</td><td>0.60</td><td>2.8e-08</td><td>25.0</td><td>5.24</td><td>21.27</td><td>1.4e-08</td></tr><tr><td>10</td><td>22.4</td><td>0.92</td><td>0.62</td><td>2.0e-08</td><td>40.8</td><td>10.48</td><td>18.33</td><td>2.1e-08</td></tr></table>

## CPU time of SparseIPM ( 1 − norm 1 28 pixels)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/883ca850dcb26fd9646b00014b0b34d77a020a8af283abbf50c1a31f75f47e0f.jpg)


$$
\begin{array}{l} {m = 2 r ^ {2}} \\ {n = r ^ {4}} \end{array}
$$


B<sub>o</sub>l<sub>ogna</sub> J<sub>anuary</sub> 2023


## C<sub>ompar</sub>i<sub>son:</sub> S<sub>ca</sub>l<sub>a</sub>bilit<sub>y o</sub>f th<sub>ree so</sub>l<sub>vers</sub>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/848773617ac3eeeb733ad87909bbce6c6ecba015bfebab668ec97235656e46a4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/e3a15224e9cedef2556e285fc8bc49188903740e43bbc3c4022be6e45f26e009.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/2298f94c-8558-4f49-af41-3cc8b05608f4/edc71d87f90b3bea7754bc579080388181355fb3af2fa4d81ae0b337877cc09e.jpg)



S <sub>parse</sub>IP M f<sub>or</sub> Di<sub>scre</sub>t<sub>e</sub> OT



Cplex (Simplex Method for Network Problems)


LEM O N (Specialized Network Algorithm) 

# O<sub>verarc</sub>hi<sub>ng</sub> F<sub>ea</sub>t<sub>ure o</sub>f IPM<sub>s</sub>

They poss ess an un equal led ability to identify th<sub>e</sub> <sup>“</sup><sub>essen</sub>t i<sub>a</sub>l <sub>su</sub>b<sub>space</sub><sup>”</sup> i<sub>n w</sub>hi<sub>c</sub>h th <sub>e op</sub> ti<sub>ma</sub>l <sub>s o</sub> l<sub>u</sub>ti<sub>on</sub> i<sub>s</sub> hidd<sub>en .</sub> 

## C<sub>onc</sub>l<sub>us</sub>i<sub>ons</sub>

2<sub>n</sub>d<sub>-or</sub>d<sub>er me</sub>th<sub>o</sub>d<sub>s</sub> f<sub>or op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on :</sub> 

• <sub>emp</sub>l<sub>oy</sub> i<sub>nexac</sub>t N<sub>ew</sub>t<sub>on me</sub>th<sub>o</sub>d 

• <sub>re</sub>l<sub>y on precon</sub>dit i<sub>oners</sub> 

• enj oy matrix- free implementation 

T<sub>r</sub>i<sub>c</sub>k <sub>:</sub> 

• fi<sub>n</sub>d th<sub>e</sub> <sup>“</sup><sub>essen</sub>ti<sub>a</sub>l <sub>su</sub>b<sub>space</sub><sup>”</sup> <sub>an</sub>d 

• <sub>exp</sub>l<sub>o</sub>it it t<sub>o s</sub>i<sub>mp</sub>lif<sub>y</sub> th<sub>e</sub> li<sub>near a</sub>l<sub>ge</sub>b<sub>ra</sub> 

– <sub>wor</sub>k<sub>s</sub> i<sub>n</sub> IP M<sub>s</sub> f<sub>or</sub> LP 

– <sub>wor</sub>k<sub>s</sub> i<sub>n</sub> N<sub>ew</sub>t<sub>on</sub> CG f<sub>or</sub> $\ell _ { 1 } { \mathrm { - r e g u l a r i z a t i o n } }$ 

Si<sub>mp</sub>l<sub>e re</sub>li<sub>a</sub>bl<sub>e</sub> t<sub>es</sub>t <sub>examp</sub>l<sub>e</sub> f<sub>or</sub> $\ell _ { 1 } { \mathrm { - r e g u l a r i z a t i o n } } \colon$ 

htt<sub>p</sub> : / /www <sub>.</sub> maths <sub>.</sub> ed <sub>.</sub> ac <sub>.</sub> uk/ERGO /t r i l l i on/ 