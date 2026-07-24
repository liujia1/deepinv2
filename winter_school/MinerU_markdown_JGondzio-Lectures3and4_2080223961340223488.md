# IPM<sub>s</sub> f<sub>or</sub> C<sub>onvex</sub> O<sub>p</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on:</sub> QP NLP SOCP and SDP

J<sub>ace</sub>k G<sub>on</sub>d<sub>z</sub>i<sub>o</sub> E<sub>ma</sub>il <sub>:</sub> J <sub>.</sub> G<sub>on</sub>d<sub>z</sub> i <sub>o</sub>@<sub>e</sub>d <sub>. ac . u</sub>k URL : htt<sub>p</sub> : / /www <sub>.</sub> maths <sub>.</sub> ed <sub>.</sub> ac <sub>.</sub> uk/ <sup>~</sup> <sub>g</sub>ondz i o 

## O<sub>u</sub>tli<sub>ne</sub>

## • IPM for Quadratic and Nonlinear Programming

– <sub>qua</sub>d<sub>ra</sub>ti<sub>c</sub> f<sub>orms</sub> NLP <sub>no</sub>t<sub>a</sub>ti<sub>on</sub> 

– d<sub>ua</sub>lit<sub>y,</sub> L<sub>agrang</sub>i<sub>an ,</sub> fi<sub>rs</sub>t <sub>or</sub>d<sub>er op</sub>ti<sub>ma</sub>lit<sub>y con</sub>diti<sub>ons</sub> 

– <sub>pr</sub>i<sub>ma</sub>l<sub>-</sub>d<sub>ua</sub>l f<sub>ramewor</sub>k 

• S<sub>e</sub>lf<sub>-concor</sub>d<sub>an</sub>t b<sub>arr</sub>i<sub>er</sub> 

## • S<sub>econ</sub>d<sub>-</sub>O<sub>r</sub>d<sub>er</sub> C<sub>one</sub> P<sub>rogramm</sub>i<sub>ng</sub>

– exam<sub>p</sub><sup>l</sup>e cones 

– <sub>examp</sub>l<sub>e</sub> SOCP <sub>pro</sub>bl<sub>ems</sub> 

– l<sub>ogar</sub>ith<sub>m</sub>i<sub>c</sub> b<sub>arr</sub>i<sub>er</sub> f<sub>unc</sub>ti<sub>on</sub> 

## • S<sub>em</sub>id<sub>e</sub>fi<sub>n</sub>it<sub>e</sub> P<sub>rogramm</sub>i<sub>ng</sub>

– background (linear matrix inequalities) 

– <sub>examp</sub>l<sub>e</sub> SDP <sub>pro</sub>bl<sub>ems</sub> 

– l<sub>ogar</sub>ith<sub>m</sub>i<sub>c</sub> b<sub>arr</sub>i<sub>er</sub> f<sub>unc</sub>ti<sub>on</sub> 

## • Fi<sub>na</sub>l C<sub>ommen</sub>t<sub>s</sub>

IPM for Convex QP 

## Convex Quadratic Programs

D<sub>e</sub>f<sub>.</sub> A <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $Q \in \mathcal { R } ^ { n \times n }$ i<sub>s p os</sub>iti<sub>ve sem</sub>id<sub>e</sub>fi<sub>n</sub>it<sub>e</sub> if $x ^ { T } Q x \ge 0$ f<sub>or</sub> <sup>an</sup>y $x \neq 0$ W<sub>e wr</sub>it<sub>e</sub> $Q \succeq 0$ 

Th<sub>e qua</sub>d<sub>ra</sub>ti<sub>c</sub> f<sub>unc</sub>ti<sub>on</sub> 

$$
f (x) = x ^ {T} Q x
$$

i<sub>s convex</sub> if <sub>an</sub>d <sub>on</sub>l<sub>y</sub> if th<sub>e ma</sub>t<sub>r</sub>i<sub>x</sub> $Q$ i<sub>s pos</sub>iti<sub>ve</sub> d<sub>e</sub>fi<sub>n</sub>it<sub>e .</sub> I<sub>n suc</sub>h <sub>case</sub> th<sub>e qua</sub>d<sub>ra</sub>ti<sub>c programm</sub>i<sub>ng pro</sub>bl<sub>em</sub> 

$$
\begin{array}{r l} \min & c ^ {T} x + \frac {1}{2} x ^ {T} Q x \\ \mathrm{s.t.} & A x = b, \\ & x \geq 0, \end{array}
$$

i<sub>s we</sub>ll d<sub>e</sub>fi<sub>ne</sub>d <sub>.</sub> 

If there exists a feasi b le solut ion to it th<sub>en</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s an op</sub> ti<sub>ma</sub>l <sub>so</sub>l<sub>u</sub>ti<sub>on .</sub> 

## QP with IPMs

A<sub>pp</sub>l<sub>y</sub> th<sub>e usua</sub>l <sub>proce</sub>d<sub>ure :</sub> 

• <sub>rep</sub>l<sub>ace</sub> i<sub>nequa</sub>liti<sub>es w</sub>ith l<sub>og</sub> b<sub>arr</sub>i<sub>ers ;</sub> 

• f<sub>orm</sub> th<sub>e</sub> L<sub>agrang</sub>i<sub>an ;</sub> 

• <sub>wr</sub>it<sub>e</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>or</sub>d<sub>er op</sub>ti<sub>ma</sub>lit<sub>y con</sub>diti<sub>ons ;</sub> 

• <sub>app</sub>l<sub>y</sub> N<sub>ew</sub>t<sub>on me</sub>th<sub>o</sub>d t<sub>o</sub> th<sub>em .</sub> 

Replace the primal QP 

$$
\min c ^ {T} x + \frac {1}{2} x ^ {T} Q x
$$

$$
\mathrm{s.t.} \qquad A x = b,
$$

$$
x \geq 0,
$$

with the primal barrier QP 

$$
\min c ^ {T} x + \frac {1}{2} x ^ {T} Q x - \sum_ {j = 1} ^ {n} \ln x _ {j}
$$

$$
\mathrm{s.t.} A x = b.
$$

## Fi<sub>rs</sub>t O<sub>r</sub>d<sub>er</sub> O<sub>p</sub>t i<sub>ma</sub>lit<sub>y</sub> C<sub>on</sub>dit i<sub>ons</sub>

C<sub>ons</sub>id<sub>er</sub> th<sub>e pr</sub>i<sub>ma</sub>l b<sub>arr</sub>i<sub>er qua</sub>d<sub>ra</sub>ti<sub>c program</sub> 

$$
\begin{array}{r l r} & {\min} & {c ^ {T} x + \frac {1}{2} x ^ {T} Q x - \mu \sum_ {j = 1} ^ {n} \ln x _ {j}} \\ & {\mathrm{s.t.}} & {A x = b,} \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $\mu \geq 0$ i<sub>s a</sub> b<sub>arr</sub>i<sub>er parame</sub>t<sub>er .</sub> 

W<sub>r</sub>it<sub>e ou</sub>t th<sub>e</sub> L<sub>agrang</sub>i<sub>an</sub> 

$$
L (x, y, \mu) = c ^ {T} x + \frac {1}{2} x ^ {T} Q x - y ^ {T} (A x - b) - \mu \sum_ {j = 1} ^ {n} \ln x _ {j},
$$

## First Order Opt imality Condit ions (cont <sup>’</sup> d)

Th<sub>e con</sub>diti<sub>ons</sub> f<sub>or a s</sub>t<sub>a</sub>ti<sub>onary po</sub>i<sub>n</sub>t <sub>o</sub>f th<sub>e</sub> L<sub>agrang</sub>i<sub>an :</sub> 

$$
L (x, y, \mu) = c ^ {T} x + \frac {1}{2} x ^ {T} Q x - y ^ {T} (A x - b) - \mu \sum_ {j = 1} ^ {n} \ln x _ {j},
$$

are 

$$
\begin{array}{r l r} \nabla_ {x} L (x, y, \mu) = c - A ^ {T} y - \mu X ^ {- 1} e + Q x = 0 \\ \nabla_ {y} L (x, y, \mu) = & A x - b = 0, \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $X ^ { - 1 } = d i a g \{ x _ { 1 } ^ { - 1 } , x _ { 2 } ^ { - 1 } , \cdots , x _ { n } ^ { - 1 } \} .$ 

L<sub>e</sub>t <sub>us</sub> d<sub>eno</sub>t<sub>e</sub> 

$$
s = \mu X ^ {- 1} e, \quad \mathrm{i.e.} \quad X S e = \mu e.
$$

Th<sub>e</sub> Fi<sub>rs</sub>t O <sub>r</sub>d<sub>er</sub> O <sub>p</sub>t i<sub>ma</sub>lit<sub>y</sub> C<sub>on</sub>dit i<sub>ons are :</sub> 

$$
\begin{array}{r l} {A x} & {= b,} \\ {A ^ {T} y + s - Q x} & {= c,} \\ {X S e} & {= \mu e.} \end{array}
$$

## A<sub>pp</sub>l<sub>y</sub> N<sub>ew</sub>t<sub>on</sub> M<sub>e</sub>th<sub>o</sub>d t<sub>o</sub> th<sub>e</sub> FOC

Th<sub>e</sub> fi<sub>rs</sub>t <sub>or</sub>d<sub>er op</sub>ti<sub>ma</sub>lit<sub>y con</sub>diti<sub>ons</sub> f<sub>or</sub> th<sub>e</sub> b<sub>arr</sub>i<sub>er pro</sub>bl<sub>em</sub> f<sub>orm a</sub> l<sub>arge sys</sub>t<sub>em o</sub>f <sub>non</sub>li<sub>near equa</sub>ti<sub>ons</sub> 

$$
F (x, y, s) = 0,
$$

<sub>w</sub>h<sub>ere</sub> $F : \mathcal { R } ^ { 2 n + m } \mapsto \mathcal { R } ^ { 2 n + m }$ i<sub>s an app</sub>li<sub>ca</sub>ti<sub>on</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows :</sub> 

$$
F (x, y, s) = \left[ \begin{array}{c c} A x & - b \\ A ^ {T} y + s & - Q x - c \\ X S e & - \mu e \end{array} \right].
$$

A<sub>c</sub>t<sub>ua</sub>ll<sub>y</sub> th<sub>e</sub> fi<sub>rs</sub>t t<sub>wo</sub> t<sub>erms o</sub>f it <sub>are</sub> li<sub>n ear; on</sub>l<sub>y</sub> th<sub>e</sub> l<sub>as</sub>t <sub>one</sub> <sub>correspon</sub>di<sub>ng</sub> t<sub>o</sub> th<sub>e comp</sub>l<sub>emen</sub>t<sub>ar</sub>it<sub>y con</sub>diti<sub>on</sub> i<sub>s non</sub>li<sub>near.</sub> N<sub>o</sub>t<sub>e</sub> th<sub>a</sub>t 

$$
\nabla F (x, y, s) = \left[ \begin{array}{c c c} A & 0 & 0 \\ - Q & A ^ {T} & I \\ S & 0 & X \end{array} \right].
$$

## Newton Method for the FOC (cont <sup>’</sup>d)

Th<sub>us</sub> f<sub>or a g</sub>i<sub>ven po</sub>i<sub>n</sub>t $( x , y , s )$ <sub>we</sub> fi<sub>n</sub>d th<sub>e</sub> N<sub>ew</sub>t<sub>on</sub> di<sub>rec</sub>ti<sub>on</sub> $( \Delta x , \Delta y , \Delta s )$ b<sub>y so</sub>l<sub>v</sub>i<sub>ng</sub> th<sub>e sys</sub>t<sub>em o</sub>f li<sub>near equa</sub>ti<sub>ons :</sub> 

$$
\left[ \begin{array}{c c c} A & 0 & 0 \\ - Q & A ^ {T} & I \\ S & 0 & X \end{array} \right] \cdot \left[ \begin{array}{l} \Delta x \\ \Delta y \\ \Delta s \end{array} \right] = \left[ \begin{array}{l} b - A x \\ c - A ^ {T} y - s + Q x \\ \mu e - X S e \end{array} \right].
$$

Interior-Point QP Algorithm

Initialize $k = 0, \quad (x^{0}, y^{0}, s^{0}) \in \mathcal{F}^{0}, \quad \mu_{0} = \frac{1}{n} \cdot (x^{0})^{T} s^{0}, \quad \alpha_{0} = 0.9995$ Repeat until optimality $k = k + 1$ $\mu_{k} = \sigma \mu_{k-1}$ , where $\sigma \in (0,1)$ $\Delta = \text{Newton direction towards } \mu\text{-center}$ Ratio test: $\alpha_{P} := \max \left\{ \alpha > 0 : x + \alpha \Delta x \geq 0 \right\}$ , $\alpha_{D} := \max \left\{ \alpha > 0 : s + \alpha \Delta s \geq 0 \right\}$ .

Make step: $x^{k+1} = x^{k} + \alpha_{0} \alpha_{P} \Delta x$ , $y^{k+1} = y^{k} + \alpha_{0} \alpha_{D} \Delta y$ , $s^{k+1} = s^{k} + \alpha_{0} \alpha_{D} \Delta s$ . 

## From LP to QP

QP problem 

$$
\begin{array}{r l} \min & c ^ {T} x + \frac {1}{2} x ^ {T} Q x \\ \mathrm{s.t.} & A x = b, \\ & x \geq 0. \end{array}
$$

First order condit ions (for barrier problem) 

$$
\begin{array}{r l} {A x} & {= b,} \\ {A ^ {T} y + s - Q x} & {= c,} \\ {X S e} & {= \mu e.} \end{array}
$$

IPM<sub>s</sub> f<sub>or</sub> C<sub>onvex</sub> NLP 

## C<sub>onvex</sub> N<sub>on</sub>li<sub>near</sub> O<sub>p</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub>

C<sub>ons</sub>id<sub>er</sub> th<sub>e non</sub>li<sub>near op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em</sub> 

<sub>m</sub>i<sub>n</sub> $f ( x )$ s <sub>.</sub> t <sub>.</sub> $g ( x ) \leq 0 ,$ 

<sub>w</sub>h<sub>ere</sub> $x \in \mathcal { R } ^ { n }$ <sub>an</sub>d $f : \mathcal { R } ^ { n } \mapsto \mathcal { R }$ <sub>an</sub>d $g : \mathcal { R } ^ { n } \mapsto \mathcal { R } ^ { m }$ are convex t<sub>w</sub>i<sub>ce</sub> dif<sub>eren</sub>ti<sub>a</sub>bl<sub>e .</sub> 

```txt
Assumptions:
f and g are convex
⇒ If there exists a local minimum then it is a global one.
f and g are twice differentiable
⇒ We can use the second order Taylor approximations. 
```

Some additional (technical) conditions ⇒ W<sub>e nee</sub>d th<sub>em</sub> t<sub>o prove</sub> th<sub>a</sub>t th<sub>e po</sub>i<sub>n</sub>t <sub>w</sub>hi<sub>c</sub>h <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e</sub> fi<sub>rs</sub>t <sub>or</sub>d<sub>er op</sub>ti<sub>ma</sub>lit<sub>y con</sub>diti<sub>ons</sub> i<sub>s</sub> th<sub>e op</sub>ti<sub>mum .</sub> W<sub>e won</sub> <sup>’</sup>t <sub>us e</sub> th <sub>em</sub> i<sub>n</sub> thi<sub>s cours e .</sub> 

## N<sub>on</sub>li<sub>near</sub> O<sub>p</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on w</sub>ith IPM<sub>s</sub>

## Nonlinear Optimization via QPs:

Sequential Quadratic Programming (SQP) <sub>.</sub> 

R<sub>epea</sub>t <sub>un</sub>til <sub>op</sub>ti<sub>ma</sub>lit<sub>y :</sub> 

• approximate NLP (locally) with a QP ; 

• solve (approximately) the Q P<sub>.</sub> 

## N<sub>on</sub>li<sub>near</sub> O<sub>p</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on w</sub>ith IPM<sub>s :</sub>

works similarl<sub>y</sub> to S Q P scheme <sub>.</sub> 

However the (local) QP approximations are not solved to optimalit<sub>y.</sub> I<sub>ns</sub>t<sub>ea</sub>d <sub>, on</sub>l<sub>y one s</sub>t<sub>ep</sub> i<sub>n</sub> th<sub>e</sub> N<sub>ew</sub>t<sub>on</sub> di<sub>rec</sub>ti<sub>on correspon</sub>di<sub>ng</sub> t<sub>o</sub> a given QP approximation is made and the new QP approximation i<sub>s compu</sub>t<sub>e</sub>d <sub>.</sub> 

## NLP N<sub>o</sub>t<sub>a</sub>ti<sub>on</sub>

C<sub>ons</sub>id<sub>er</sub> th<sub>e non</sub>li<sub>near op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em</sub> 

$$
\min f (x) \quad \mathrm{s.t.} \quad g (x) \leq 0,
$$

<sub>w</sub>h<sub>ere</sub> $x \in \mathcal { R } ^ { n }$ <sub>an</sub>d $f : \mathcal { R } ^ { n } \mapsto \mathcal { R }$ <sub>an</sub>d $g : \mathcal { R } ^ { n } \mapsto \mathcal { R } ^ { m }$ are convex t<sub>w</sub>i<sub>ce</sub> dif<sub>eren</sub>ti<sub>a</sub>bl<sub>e .</sub> 

Th<sub>e vec</sub>t<sub>or-va</sub>l<sub>ue</sub>d f<sub>unc</sub>ti<sub>on</sub> $g : \mathcal { R } ^ { n } \mapsto \mathcal { R } ^ { m }$ h<sub>as a</sub> d<sub>er</sub>i<sub>va</sub>ti<sub>ve</sub> 

$$
A (x) = \nabla g (x) = \left[ \frac {\partial g _ {i}}{\partial x _ {j}} \right] _ {i = 1.. m, j = 1.. n} \in \mathcal {R} ^ {m \times n}
$$

<sub>w</sub>hi<sub>c</sub>h i<sub>s ca</sub>ll<sub>e</sub>d th<sub>e</sub> J<sub>aco</sub>bi<sub>an o</sub>f $g .$ 

## NLP Notation (cont <sup>’</sup>d)

Th<sub>e</sub> L<sub>agrang</sub>i<sub>an assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>w</sub>ith th<sub>e</sub> NLP i<sub>s :</sub> 

$$
\mathcal {L} (x, y) = f (x) + y ^ {T} g (x),
$$

<sub>w</sub>h<sub>ere</sub> $y \in \mathcal { R } ^ { m } , y \geq 0$ are Lagrange multipliers (dual variables) <sub>.</sub> 

Th<sub>e</sub> fi<sub>rs</sub>t d<sub>er</sub>i<sub>va</sub>ti<sub>ves o</sub>f th<sub>e</sub> L<sub>agrang</sub>i<sub>an :</sub> 

$$
\begin{array}{r l} & {\nabla_ {x} \mathcal {L} (x, y) = \nabla f (x) + \nabla g (x) ^ {T} y} \\ & {\nabla_ {y} \mathcal {L} (x, y) = g (x).} \end{array}
$$

Th<sub>e</sub> H<sub>ess</sub>i<sub>an o</sub>f th<sub>e</sub> L<sub>agrang</sub>i<sub>an ,</sub> $Q ( x , y ) \in \mathcal { R } ^ { n \times n }$ 

$$
Q (x, y) = \nabla_ {x x} ^ {2} \mathcal {L} (x, y) = \nabla^ {2} f (x) + \sum_ {i = 1} ^ {m} y _ {i} \nabla^ {2} g _ {i} (x).
$$

## C<sub>onvex</sub>it<sub>y</sub> i<sub>n</sub> NLP

L<sub>emma</sub> 2 <sub>:</sub> If $f : \mathcal { R } ^ { n } \mapsto \mathcal { R }$ <sub>an</sub>d $g : \mathcal { R } ^ { n } \mapsto \mathcal { R } ^ { m }$ <sub>are convex</sub> t<sub>w</sub>i<sub>ce</sub> dif<sub>eren</sub>ti<sub>a</sub>bl<sub>e</sub> th<sub>en</sub> th<sub>e</sub> H<sub>ess</sub>i<sub>an o</sub>f th<sub>e</sub> L<sub>agrang</sub>i<sub>an</sub> 

$$
Q (x, y) = \nabla^ {2} f (x) + \sum_ {i = 1} ^ {m} y _ {i} \nabla^ {2} g _ {i} (x)
$$

i<sub>s pos</sub>iti<sub>ve sem</sub>id<sub>e</sub>fi<sub>n</sub>it<sub>e</sub> f<sub>or any x an</sub>d <sub>any</sub> $y \geq 0$ <sub>.</sub> If f is st rict l<sub>y</sub> <sub>convex</sub> th<sub>en</sub> $Q ( x , y )$ i<sub>s pos</sub>iti<sub>ve</sub> d<sub>e</sub>fi<sub>n</sub>it<sub>e</sub> f<sub>or any x an</sub>d <sub>any</sub> $y \geq 0$ 

Proof: The convexit<sub>y</sub> of f implies that $\nabla ^ { 2 } f ( x )$ i<sub>s p os</sub>it i<sub>ve sem</sub>id<sub>e</sub>f<sub>-</sub> i<sub>n</sub>it<sub>e</sub> f<sub>or any</sub> $x .$ <sub>.</sub> Si<sub>m</sub>il<sub>ar</sub>l<sub>y</sub> th<sub>e convex</sub>it<sub>y o</sub>f <sub>g</sub> i<sub>mp</sub>li<sub>es</sub> th<sub>a</sub>t f<sub>or a</sub>ll $i = 1 , 2 , . . . , m , \nabla ^ { 2 } g _ { i } ( x )$ i<sub>s pos</sub>iti<sub>ve sem</sub>id<sub>e</sub>fi<sub>n</sub>it<sub>e</sub> f<sub>or any</sub> $x .$ Si<sub>nce</sub> $y _ { i } \geq 0$ f<sub>or a</sub>ll $i = 1 , 2 , . . . , m$ <sub>an</sub>d $Q ( x , y )$ i<sub>s</sub> th<sub>e sum o</sub>f <sub>p os</sub>iti<sub>ve</sub> <sub>sem</sub>id<sub>e</sub>fi<sub>n</sub>it<sub>e ma</sub>t<sub>r</sub>i<sub>ces we</sub> h<sub>ave</sub> th<sub>a</sub>t $Q ( x , y )$ i<sub>s pos</sub>iti<sub>ve sem</sub>id<sub>e</sub>fi<sub>n</sub>it<sub>e .</sub> 

If f is strictl<sub>y</sub> convex then $\nabla ^ { 2 } f ( x )$ i<sub>s pos</sub>iti<sub>ve</sub> d<sub>e</sub>fi<sub>n</sub>it<sub>e an</sub>d <sub>so</sub> i<sub>s</sub> $Q ( x , y )$ 

## IPM f<sub>or</sub> NLP

Add <sub>s</sub>l<sub>ac</sub>k <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> t<sub>o non</sub>li<sub>near</sub> i<sub>nequa</sub>liti<sub>es :</sub> 

$$
\begin{array}{r l} \min & f (x) \\ \mathrm{s.t.} & g (x) + z = 0 \\ & z \geq 0, \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $z \in \mathcal { R } ^ { m }$ <sub>.</sub> R<sub>ep</sub>l<sub>ace</sub> i<sub>nequa</sub>lit<sub>y</sub> $z \geq 0$ <sub>w</sub>ith th<sub>e</sub> l<sub>ogar</sub>ith<sub>m</sub>i<sub>c</sub> b <sub>arr</sub>i<sub>er :</sub> 

$$
\begin{array}{r l} \min & f (x) - \mu \sum_ {i = 1} ^ {m} \ln z _ {i} \\ \mathrm{s.t.} & g (x) + z = 0. \end{array}
$$

W<sub>r</sub>it<sub>e ou</sub>t th<sub>e</sub> L<sub>agrang</sub>i<sub>an</sub> 

$$
L (x, y, z, \mu) = f (x) + y ^ {T} (g (x) + z) - \mu \sum_ {i = 1} ^ {m} \ln z _ {i},
$$

## IPM f<sub>or</sub> NLP

F<sub>or</sub> th<sub>e</sub> L<sub>agrang</sub>i<sub>an</sub> 

$$
L (x, y, z, \mu) = f (x) + y ^ {T} (g (x) + z) - \mu \sum_ {i = 1} ^ {m} \ln z _ {i},
$$

<sub>wr</sub>it<sub>e</sub> th<sub>e con</sub>diti<sub>ons</sub> f<sub>or a s</sub>t<sub>a</sub>ti<sub>onary po</sub>i<sub>n</sub>t 

$$
\nabla_ {x} L (x, y, z, \mu) = \nabla f (x) + \nabla g (x) ^ {T} y = 0
$$

$$
\nabla_ {y} L (x, y, z, \mu) = g (x) + z = 0
$$

$$
\nabla_ {z} L (x, y, z, \mu) = y - \mu Z ^ {- 1} e = 0,
$$

<sub>w</sub>h<sub>ere</sub> $Z ^ { - 1 } = d i a g \{ z _ { 1 } ^ { - 1 } , z _ { 2 } ^ { - 1 } , \cdots , z _ { m } ^ { - 1 } \}$ 

Th<sub>e</sub> Fi<sub>rs</sub>t O <sub>r</sub>d<sub>er</sub> O <sub>p</sub>t i<sub>ma</sub>lit<sub>y</sub> C<sub>on</sub>dit i<sub>ons are :</sub> 

$$
\begin{array}{r} \nabla f (x) + \nabla g (x) ^ {T} y = 0, \\ g (x) + z = 0, \\ Y Z e = \mu e. \end{array}
$$

## N<sub>ew</sub>t<sub>on</sub> M<sub>e</sub>th<sub>o</sub>d f<sub>or</sub> th<sub>e</sub> FOC

Th<sub>e</sub> fi<sub>rs</sub>t <sub>or</sub>d<sub>er op</sub>ti<sub>ma</sub>lit<sub>y con</sub>diti<sub>ons</sub> f<sub>or</sub> th<sub>e</sub> b<sub>arr</sub>i<sub>er pro</sub>bl<sub>em</sub> f<sub>orm a</sub> l<sub>arge sys</sub>t<sub>em o</sub>f <sub>non</sub>li<sub>near equa</sub>ti<sub>ons</sub> 

$$
F (x, y, z) = 0,
$$

<sub>w</sub>h<sub>ere</sub> $F : \mathcal { R } ^ { n + 2 m } \mapsto \mathcal { R } ^ { n + 2 m }$ i<sub>s an app</sub>li<sub>ca</sub>ti<sub>on</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows :</sub> 

$$
F (x, y, z) = \left[ \begin{array}{c} \nabla f (x) + \nabla g (x) ^ {T} y \\ g (x) + z \\ Y Z e - \mu e \end{array} \right].
$$

N<sub>o</sub>t<sub>e</sub> th<sub>a</sub>t <sub>a</sub>ll th<sub>ree</sub> t<sub>erms o</sub>f it <sub>are non</sub>li<sub>n ear.</sub> 

( In LP and Q P the first two terms were lin ear<sub>.</sub> ) 

## N<sub>ew</sub>t<sub>on</sub> M<sub>e</sub>th<sub>o</sub>d f<sub>or</sub> th<sub>e</sub> FOC

Ob<sub>serve</sub> th<sub>a</sub>t 

$$
\nabla F (x, y, z) = \left[ \begin{array}{c c c} Q (x, y) & A (x) ^ {T} & 0 \\ A (x) & 0 & I \\ 0 & Z & Y \end{array} \right],
$$

<sub>w</sub>h<sub>ere</sub> $A ( x )$ i<sub>s</sub> th<sub>e</sub> J<sub>aco</sub>bi<sub>an o</sub>f $g$ <sub>an</sub>d $Q ( x , y )$ i<sub>s</sub> th<sub>e</sub> H<sub>ess</sub>i<sub>an o</sub>f $\mathcal { L }$ 

Th<sub>ey are</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>as</sub> f<sub>o</sub>ll<sub>ows :</sub> 

$$
\begin{array}{r} A (x) = \nabla g (x) \in \mathcal {R} ^ {m \times n} \\ Q (x, y) = \nabla^ {2} f (x) + \sum_ {i = 1} ^ {m} y _ {i} \nabla^ {2} g _ {i} (x) \in \mathcal {R} ^ {n \times n} \end{array}
$$

## Newton Method (cont <sup>’</sup>d)

F<sub>o</sub>r <sub>a g</sub>i<sub>ve</sub>n <sub>po</sub>int $( x , y , z )$ <sub>we</sub> fi<sub>n</sub>d th<sub>e</sub> N<sub>ew</sub>t<sub>on</sub> di<sub>rec</sub>ti<sub>on</sub> $( \Delta x , \Delta y , \Delta z )$ b<sub>y so</sub>l<sub>v</sub>i<sub>ng</sub> th<sub>e sys</sub>t<sub>em o</sub>f li<sub>near equa</sub>ti<sub>ons :</sub> 

$$
\left[ \begin{array}{c c c} Q (x, y) & A (x) ^ {T} & 0 \\ A (x) & 0 & I \\ 0 & Z & Y \end{array} \right] \left[ \begin{array}{c} \Delta x \\ \Delta y \\ \Delta z \end{array} \right] = \left[ \begin{array}{c} - \nabla f (x) - A (x) ^ {T} y \\ - g (x) - z \\ \mu e - Y Z e \end{array} \right].
$$

U<sub>s</sub>i<sub>ng</sub> th<sub>e</sub> thi<sub>r</sub>d <sub>equa</sub>ti<sub>on we e</sub>li<sub>m</sub>i<sub>na</sub>t<sub>e</sub> 

$$
\Delta z = \mu Y ^ {- 1} e - Z e - Z Y ^ {- 1} \Delta y,
$$

f<sub>rom</sub> th<sub>e secon</sub>d <sub>equa</sub>ti<sub>on an</sub>d <sub>ge</sub>t 

$$
\left[ \begin{array}{c c} Q (x, y) & A (x) ^ {T} \\ A (x) & - Z Y ^ {- 1} \end{array} \right] \left[ \begin{array}{c} \Delta x \\ \Delta y \end{array} \right] = \left[ \begin{array}{c} - \nabla f (x) - A (x) ^ {T} y \\ - g (x) - \mu Y ^ {- 1} e \end{array} \right].
$$

Interior-Point NLP Algorithm

Initialize $k = 0$ $(x^0, y^0, z^0)$ such that $y^0 > 0$ and $z^0 > 0$ , $\mu_0 = \frac{1}{m} \cdot (y^0)^T z^0$ Repeat until optimality $k = k + 1$ $\mu_k = \sigma \mu_{k-1}$ , where $\sigma \in (0, 1)$ Compute $A(x)$ and $Q(x, y)$ $\Delta = \text{Newton direction towards } \mu\text{-center}$ Ratio test: $\alpha_1 := \max \{ \alpha > 0 : y + \alpha \Delta y \geq 0 \}$ , $\alpha_2 := \max \{ \alpha > 0 : z + \alpha \Delta z \geq 0 \}$ .

Choose the step: (use trust region or line search) $\alpha \leq \min \{ \alpha_1, \alpha_2 \}$ Make step: $x^{k+1} = x^k + \alpha \Delta x$ , $y^{k+1} = y^k + \alpha \Delta y$ , $z^{k+1} = z^k + \alpha \Delta z$ . 

## From QP to NLP

Newton direction for Q P 

$$
{\left[ \begin{array}{l l l} - Q & A ^ {T} & I \\ A & 0 & 0 \\ S & 0 & X \end{array} \right]} {\left[ \begin{array}{l} \Delta x \\ \Delta y \\ \Delta s \end{array} \right]} = {\left[ \begin{array}{l} \xi_ {d} \\ \xi_ {p} \\ \xi_ {\mu} \end{array} \right]}.
$$

A<sub>ugmen</sub>t<sub>e</sub>d <sub>sys</sub>t<sub>em</sub> f<sub>or</sub> $\mathrm { Q P }$ 

$$
\left[ \begin{array}{c c} - Q - S X ^ {- 1} & A ^ {T} \\ A & 0 \end{array} \right] \left[ \begin{array}{c} \Delta x \\ \Delta y \end{array} \right] = \left[ \begin{array}{c} \xi_ {d} - X ^ {- 1} \xi_ {\mu} \\ \xi_ {p} \end{array} \right].
$$

## From QP to NLP

N<sub>ew</sub>t<sub>on</sub> di<sub>rec</sub>ti<sub>on</sub> f<sub>or</sub> NLP 

$$
\left[ \begin{array}{c c c} Q (x, y) & A (x) ^ {T} & 0 \\ A (x) & 0 & I \\ 0 & Z & Y \end{array} \right] \left[ \begin{array}{c} \Delta x \\ \Delta y \\ \Delta z \end{array} \right] = \left[ \begin{array}{c} - \nabla f (x) - A (x) ^ {T} y \\ - g (x) - z \\ \mu e - Y Z e \end{array} \right].
$$

A<sub>ugmen</sub>t<sub>e</sub>d <sub>sys</sub>t<sub>em</sub> f<sub>or</sub> NLP 

$$
\left[ \begin{array}{c c} Q (x, y) & A (x) ^ {T} \\ A (x) & - Z Y ^ {- 1} \end{array} \right] \left[ \begin{array}{c} \Delta x \\ \Delta y \end{array} \right] = \left[ \begin{array}{c} - \nabla f (x) - A (x) ^ {T} y \\ - g (x) - \mu Y ^ {- 1} e \end{array} \right].
$$

## C<sub>onc</sub>l<sub>us</sub>i<sub>on:</sub>

N LP is a natural extension of QP<sub>.</sub> 

N<sub>ew</sub>t<sub>on</sub> M<sub>e</sub>th<sub>o</sub>d <sub>an</sub>d S<sub>e</sub>lf<sub>-concor</sub>d<sub>an</sub>t B<sub>arr</sub>i<sub>ers</sub> 

## A<sub>no</sub>th<sub>er</sub> Vi<sub>ew o</sub>f N<sub>ew</sub>t<sub>on</sub> M <sub>.</sub> f<sub>or</sub> O<sub>p</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub>

## N<sub>ew</sub>t<sub>on</sub> M<sub>e</sub>th<sub>o</sub>d f<sub>or</sub> O<sub>p</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub>

L<sub>e</sub>t $f : \mathcal { R } ^ { n } \mapsto \mathcal { R }$ b<sub>e a</sub> t<sub>w</sub>i<sub>ce con</sub>ti<sub>nuous</sub>l<sub>y</sub> dif<sub>eren</sub>ti<sub>a</sub>bl<sub>e</sub> f<sub>unc</sub>ti<sub>on</sub> S<sub>uppose we</sub> b<sub>u</sub>ild <sub>a qua</sub>d<sub>ra</sub>ti<sub>c mo</sub>d<sub>e</sub>l $\tilde { f }$ <sub>o</sub>f $f$ <sub>aroun</sub>d <sub>a g</sub>i<sub>ven po</sub>i<sub>n</sub>t $x ^ { k } , \operatorname { i . e . }$ <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> $\Delta \boldsymbol { x } = \boldsymbol { x } - \boldsymbol { x } ^ { k }$ <sub>an</sub>d <sub>wr</sub>it<sub>e :</sub> 

$$
\tilde {f} (x) = f (x ^ {k}) + \nabla f (x ^ {k}) ^ {T} \Delta x + \frac {1}{2} \Delta x ^ {T} \nabla^ {2} f (x ^ {k}) \Delta x
$$

N<sub>ow we op</sub>ti<sub>m</sub>i<sub>ze</sub> th<sub>e mo</sub>d<sub>e</sub>l $\tilde { f }$ i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f <sub>op</sub>t i<sub>m</sub>i<sub>z</sub>i<sub>ng</sub> $f$ A minimum (or more generally a stationary point) of the quadratic <sub>mo</sub>d<sub>e</sub>l <sub>sa</sub>ti<sub>s</sub>fi<sub>es :</sub> 

$$
\nabla \tilde {f} (x) = \nabla f (x ^ {k}) + \nabla^ {2} f (x ^ {k}) \Delta x = 0,
$$

i <sub>. e .</sub> 

$$
\Delta x = x - x ^ {k} = - (\nabla^ {2} f (x ^ {k})) ^ {- 1} \nabla f (x ^ {k}),
$$

<sub>w</sub>hi<sub>c</sub>h <sub>re</sub>d<sub>uces</sub> t<sub>o</sub> th<sub>e usua</sub>l <sub>equa</sub>ti<sub>on :</sub> 

$$
x ^ {k + 1} = x ^ {k} - (\nabla^ {2} f (x ^ {k})) ^ {- 1} \nabla f (x ^ {k}).
$$

## S<sub>e</sub>lf<sub>-concor</sub>d<sub>an</sub>t F<sub>unc</sub>ti<sub>ons</sub>

Th<sub>ere</sub> i<sub>s a n</sub>i<sub>ce proper</sub>t<sub>y o</sub>f th<sub>e</sub> f<sub>unc</sub>ti<sub>on</sub> th<sub>a</sub>t i<sub>s respons</sub>ibl<sub>e</sub> f<sub>or a</sub> <sub>goo</sub>d b<sub>e</sub>h<sub>av</sub>i<sub>our o</sub>f th<sub>e</sub> N<sub>ew</sub>t<sub>on me</sub>th<sub>o</sub>d <sub>.</sub> 

D<sub>e</sub>f L<sub>e</sub>t $C \in \mathcal { R } ^ { n }$ <sup>be an o</sup>p<sup>en nonem</sup>p<sup>t</sup>y <sup>con</sup>v<sup>e</sup>x <sup>set</sup> . 

L<sub>e</sub>t $f : C \mapsto \mathcal { R }$ b<sub>e a</sub> th<sub>ree</sub> ti<sub>mes con</sub>ti<sub>nuous</sub>l<sub>y</sub> dif<sub>eren</sub>ti<sub>a</sub>bl<sub>e convex</sub> f<sub>unc</sub>t i<sub>on .</sub> 

A function f is called self- concordant if there exists a constant $p > 0$ <sub>suc</sub>h th<sub>a</sub>t 

$$
| \nabla^ {3} f (x) [ h, h, h ] | \leq 2 p ^ {- 1 / 2} (\nabla^ {2} f (x) [ h, h ]) ^ {3 / 2},
$$

$$
\forall x \in C, \forall h: x + h \in C.
$$

(We then say that f is p-self-concordant ) <sub>.</sub> 

N<sub>o</sub>t<sub>e</sub> th<sub>a</sub>t <sub>a se</sub>lf<sub>-concor</sub>d<sub>an</sub>t f<sub>unc</sub>ti<sub>on</sub> i<sub>s a</sub>l<sub>ways we</sub>ll <sub>approx</sub>i<sub>ma</sub>t<sub>e</sub>d b<sub>y</sub> th<sub>e qua</sub>d<sub>ra</sub>ti<sub>c mo</sub>d<sub>e</sub>l b<sub>ecause</sub> th<sub>e error o</sub>f <sub>suc</sub>h <sub>an approx</sub>i<sub>ma</sub>ti<sub>on can</sub> be bounded by the 3/2 power of $\nabla ^ { 2 } f ( x ) [ h , h ]$ 

## S<sub>e</sub>lf<sub>-concor</sub>d<sub>an</sub>t B<sub>arr</sub>i<sub>ers</sub>

## L<sub>emma</sub>

Th<sub>e</sub> b<sub>arr</sub>i<sub>er</sub> f<sub>unc</sub>ti<sub>on</sub> − l<sub>og x</sub> i<sub>s se</sub>lf<sub>-concor</sub>d<sub>an</sub>t <sub>on</sub> $\mathcal { R } _ { + }$ 

P<sub>roo</sub>f C<sub>ons</sub>id<sub>er</sub> $f ( x ) = - \log x$ 

C<sub>ompu</sub>t<sub>e</sub> 

$f ^ { ' } ( x ) = - x ^ { - 1 } , f ^ { \prime \prime } ( x ) = x ^ { - 2 } \mathrm { ~ a n d ~ } f ^ { \prime \prime \prime } ( x ) = - 2 x ^ { - 3 }$ <sub>an</sub>d <sub>c</sub>h<sub>ec</sub>k th<sub>a</sub>t th<sub>e se</sub>lf<sub>-concor</sub>d<sub>ance con</sub>diti<sub>on</sub> i<sub>s sa</sub>ti<sub>s</sub>fi<sub>e</sub>d f<sub>or</sub> $p = 1$ 

## L<sub>emma</sub>

Th<sub>e</sub> b<sub>arr</sub>i<sub>er</sub> f<sub>unc</sub>ti<sub>on</sub> $1 / x ^ { \alpha }$ <sub>w</sub>ith $\alpha \in ( 0 , \infty )$ i<sub>s no</sub>t <sub>se</sub>lf<sub>-concor</sub>d<sub>an</sub>t on $\mathcal { R } _ { + }$ 

## L<sub>emma</sub>

Th<sub>e</sub> b<sub>arr</sub>i<sub>er</sub> f<sub>unc</sub>ti<sub>on</sub> $e ^ { 1 / x }$ i<sub>s no</sub>t <sub>se</sub>lf<sub>-concor</sub>d<sub>an</sub>t <sub>on</sub> $\mathcal { R } _ { + }$ 

## U<sub>se se</sub>lf<sub>-concor</sub>d<sub>an</sub>t b<sub>arr</sub>i<sub>ers</sub> i<sub>n op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub>

# Second- Order Cone Programming (S OCP )

## C<sub>ones:</sub> B<sub>ac</sub>k<sub>groun</sub>d

D <sub>e</sub>f<sub>.</sub> A <sub>se</sub>t $K \in \mathcal { R } ^ { n }$ i<sub>s ca</sub>ll<sub>e</sub>d <sub>a cone</sub> if f<sub>or any</sub> $x \in K$ <sub>an</sub>d f<sub>or any</sub> $\lambda \geq 0 , \lambda x \in K$ 

C<sub>onvex</sub> C<sub>one:</sub> 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/9ecf958e-cbed-49b1-9728-6c74857ccd0b/486962707c6b7a74c611c84b9ba5e3236356d340b60df164bd5cebb4934c618d.jpg)


E<sub>xamp</sub>l<sub>e:</sub> 

$$
K = \{x \in \mathcal {R} ^ {n}: x _ {1} ^ {2} \geq \sum_ {j = 2} ^ {n} x _ {j} ^ {2}, x _ {1} \geq 0 \}.
$$

E<sub>xamp</sub>l<sub>e:</sub> Th<sub>ree</sub> C<sub>ones</sub> 

$R _ { + }$ 

$$
R _ {+} = \{x \in \mathcal {R}: x \geq 0 \}.
$$

Quadratic Cone: 

$$
K _ {q} = \{x \in \mathcal {R} ^ {n}: x _ {1} ^ {2} \geq \sum_ {j = 2} ^ {n} x _ {j} ^ {2}, x _ {1} \geq 0 \}.
$$

Rotated Quadratic Cone: 

$$
K _ {r} = \{x \in \mathcal {R} ^ {n}: 2 x _ {1} x _ {2} \geq \sum_ {j = 3} ^ {n} x _ {j} ^ {2}, x _ {1}, x _ {2} \geq 0 \}.
$$

## M<sub>a</sub>t<sub>r</sub>i<sub>x</sub> R<sub>epresen</sub>t<sub>a</sub>ti<sub>on o</sub>f C<sub>ones</sub>

E<sub>ac</sub>h <sub>o</sub>f th<sub>e</sub> th<sub>ree mos</sub>t <sub>common cones</sub> h<sub>as a ma</sub>t<sub>r</sub>i<sub>x represen</sub>t<sub>a</sub>ti<sub>on</sub> using orthogonal matrices T and/or Q <sub>.</sub> 

(Orthogonal matrix: $Q ^ { T } Q = I )$ 

Quadratic Cone $K _ { q }$ <sub>.</sub> D<sub>e</sub>fi<sub>ne</sub> 

$$
Q = \left[ \begin{array}{l l l l l} 1 & & & & \\ & - 1 & & & \\ & & - 1 & & \\ & & & \ddots & \\ & & & & - 1 \end{array} \right]
$$

<sub>an</sub>d <sub>wr</sub>it<sub>e :</sub> 

$$
K _ {q} = \{x \in \mathcal {R} ^ {n}: x ^ {T} Q x \geq 0, x _ {1} \geq 0 \}.
$$

E<sub>xamp</sub>l<sub>e:</sub> $x _ { 1 } ^ { 2 } \geq x _ { 2 } ^ { 2 } + x _ { 3 } ^ { 2 } + \cdot \cdot \cdot + x _ { n } ^ { 2 } .$ 

## Matrix Representation of Cones (cont <sup>’</sup>d)

Rotated Quadratic Cone $K _ { r }$ <sub>.</sub> D<sub>e</sub>fi<sub>ne</sub> 

$$
Q = \left[ \begin{array}{l l l l l} 0 & 1 & & & \\ 1 & 0 & & & \\ & & - 1 & & \\ & & & \ddots & \\ & & & & - 1 \end{array} \right]
$$

<sub>an</sub>d <sub>wr</sub>it<sub>e :</sub> 

$$
K _ {r} = \{x \in \mathcal {R} ^ {n}: x ^ {T} Q x \geq 0, x _ {1}, x _ {2} \geq 0 \}.
$$

E<sub>xamp</sub>l<sub>e:</sub> 

$$
2 x _ {1} x _ {2} \geq x _ {3} ^ {2} + x _ {4} ^ {2} + \dots + x _ {n} ^ {2}.
$$

## Matrix Representation of Cones (cont <sup>’</sup>d)

C<sub>ons</sub>id<sub>er a</sub> li<sub>near</sub> t<sub>rans</sub>f<sub>orma</sub>ti<sub>on</sub> $T : \mathcal { R } ^ { 2 } \mapsto \mathcal { R } ^ { 2 }$ 

$$
T _ {2} = \left[ \begin{array}{c c} \frac {1}{\sqrt {2}} & \frac {1}{\sqrt {2}} \\ \frac {1}{\sqrt {2}} & - \frac {1}{\sqrt {2}} \end{array} \right].
$$

It <sub>correspon</sub>d<sub>s</sub> t<sub>o a ro</sub>t<sub>a</sub>ti<sub>on</sub> b<sub>y</sub> $\pi / 4$ <sub>.</sub> I<sub>n</sub>d<sub>ee</sub>d <sub>wr</sub>it<sub>e :</sub> 

$$
{\left[ \begin{array}{l} z \\ y \end{array} \right]} = T _ {2} {\left[ \begin{array}{l} u \\ v \end{array} \right]}
$$

t h<sub>a</sub>t i<sub>s</sub> 

$$
z = \frac {u + v}{\sqrt {2}}, y = \frac {u - v}{\sqrt {2}}
$$

to <sub>g</sub>et 

$$
2 y z = u ^ {2} - v ^ {2}.
$$

## Matrix Representation of Cones (cont <sup>’</sup>d)

N<sub>ow</sub> d<sub>e</sub>fi<sub>ne</sub> 

$$
T = \left[ \begin{array}{l l l l} \frac {1}{\sqrt {2}} & \frac {1}{\sqrt {2}} & & \\ \frac {1}{\sqrt {2}} & - \frac {1}{\sqrt {2}} & & \\ & & 1 & \\ & & & \ddots \\ & & & 1 \end{array} \right]
$$

<sub>an</sub>d <sub>o</sub>b<sub>serve</sub> th<sub>a</sub>t th<sub>e ro</sub>t<sub>a</sub>t<sub>e</sub>d <sub>qua</sub>d<sub>ra</sub>ti<sub>c cone sa</sub>ti<sub>s</sub>fi<sub>es</sub> 

$$
T x \in K _ {r} \quad \mathrm{iff} \quad x \in K _ {q}.
$$

## E<sub>xamp</sub>l<sub>e:</sub> C<sub>on</sub>i<sub>c cons</sub>t<sub>ra</sub>i<sub>n</sub>t

C<sub>ons</sub>id<sub>er a cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>:</sub> 

$$
\frac {1}{2} \| x \| ^ {2} + a ^ {T} x \leq b.
$$

Ob<sub>serve</sub> th<sub>a</sub>t $g ( x ) = { \textstyle { \frac { 1 } { 2 } } } x ^ { T } x + a ^ { T } x - b$ i<sub>s convex</sub> h<sub>ence</sub> th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t d<sub>e</sub>fi<sub>nes a convex se</sub>t <sub>.</sub> 

Th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>may</sub> b<sub>e re</sub>f<sub>ormu</sub>l<sub>a</sub>t<sub>e</sub>d <sub>as an</sub> i<sub>n</sub>t<sub>ersec</sub>ti<sub>on o</sub>f <sub>an a</sub>fi<sub>ne</sub> (linear) constraint and a quadratic one : 

$$
\begin{array}{r l} & a ^ {T} x + z = b \\ & y = 1 \\ & \| x \| ^ {2} \leq 2 y z, y, z \geq 0. \end{array}
$$

## Example : Conic constraint (cont <sup>’</sup>d)

N<sub>ow su</sub>b<sub>s</sub>tit<sub>u</sub>t<sub>e :</sub> 

$$
z = \frac {u + v}{\sqrt {2}}, y = \frac {u - v}{\sqrt {2}}
$$

to <sub>g</sub>et 

$$
a ^ {T} x + \frac {u + v}{\sqrt {2}} = b
$$

$$
u - v = \sqrt {2}
$$

$$
\| x \| ^ {2} + v ^ {2} \leq u ^ {2}.
$$

## D<sub>ua</sub>l C<sub>one</sub>

L<sub>e</sub>t $K \in \mathcal { R } ^ { n }$ b<sub>e a cone .</sub> 

D <sub>e</sub>f<sub>.</sub> Th<sub>e se</sub>t <sub>:</sub> 

$$
K _ {*} := \{s \in \mathcal {R} ^ {n}: s ^ {T} x \geq 0, \forall x \in K \}
$$

i<sub>s ca</sub>ll<sub>e</sub>d th<sub>e</sub> d<sub>ua</sub>l <sub>cone .</sub> 

D <sub>e</sub>f<sub>.</sub> Th<sub>e se</sub>t <sub>:</sub> 

$$
K _ {P} := \{s \in \mathcal {R} ^ {n}: s ^ {T} x \leq 0, \forall x \in K \}
$$

is called the p olar cone ( Fig below) <sub>.</sub> 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/9ecf958e-cbed-49b1-9728-6c74857ccd0b/c27756d0b50bb047081ff16e784afcc06c8af58087d973436792da6ed6695dbd.jpg)


## C<sub>on</sub>i<sub>c</sub> O<sub>p</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub>

C<sub>ons</sub>id<sub>er an op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em :</sub> 

$$
\begin{array}{r l} \min & c ^ {T} x \\ \mathrm{s.t.} & A x = b, \\ & x \in K, \end{array}
$$

<sub>w</sub>h<sub>ere</sub> K i<sub>s a convex c</sub>l<sub>ose</sub>d <sub>cone .</sub> 

W<sub>e assume</sub> th<sub>a</sub>t 

$$
K = K ^ {1} \times K ^ {2} \times \dots \times K ^ {k},
$$

th<sub>a</sub>t i<sub>s cone</sub> K i<sub>s a pro</sub>d<sub>uc</sub>t <sub>o</sub>f <sub>severa</sub>l i<sub>n</sub>di<sub>v</sub>id<sub>ua</sub>l <sub>cones eac</sub>h <sub>o</sub>f <sub>w</sub>hi<sub>c</sub>h i<sub>s one o</sub>f th<sub>e</sub> th<sub>ree cones</sub> d<sub>e</sub>fi<sub>ne</sub>d <sub>ear</sub>li<sub>er .</sub> 

## P<sub>r</sub>i<sub>ma</sub>l <sub>an</sub>d D<sub>ua</sub>l SOCP<sub>s</sub>

C<sub>ons</sub>id<sub>er a pr</sub>i<sub>ma</sub>l SOCP 

$$
\begin{array}{r l} \min & c ^ {T} x \\ \mathrm{s.t.} & A x = b, \\ & x \in K, \end{array}
$$

<sub>w</sub>h<sub>ere</sub> K i<sub>s a convex c</sub>l<sub>ose</sub>d <sub>cone .</sub> 

Th<sub>e assoc</sub>i<sub>a</sub>t<sub>e</sub>d d<sub>ua</sub>l SOCP 

$$
\begin{array}{r l r} \max & b ^ {T} y \\ \mathrm{s.t.} & A ^ {T} y + s = c, \\ & s \in K _ {*}. \end{array}
$$

## W<sub>ea</sub>k D<sub>ua</sub>lit<sub>y:</sub>

I f $( x , y , s )$ i<sub>s a pr</sub>i<sub>ma</sub>l<sub>-</sub>d<sub>ua</sub>l f<sub>eas</sub>ibl<sub>e so</sub>l<sub>u</sub>ti<sub>on</sub> th<sub>en</sub> 

$$
c ^ {T} x - b ^ {T} y = x ^ {T} s \geq 0.
$$

## IPM f<sub>or</sub> C<sub>on</sub>i<sub>c</sub> O<sub>p</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub>

C<sub>on</sub>i<sub>c</sub> O<sub>p</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>ems can</sub> b<sub>e so</sub>l<sub>ve</sub>d i<sub>n po</sub>l<sub>ynom</sub>i<sub>a</sub>l ti<sub>me w</sub>ith I P M <sub>s .</sub> 

C<sub>ons</sub>id<sub>er a qua</sub>d<sub>ra</sub>ti<sub>c cone</sub> 

$$
K _ {q} = \{(x, t): x \in \mathcal {R} ^ {n - 1}, t \in \mathcal {R}, t ^ {2} \geq \| x \| ^ {2}, t \geq 0 \},
$$

and define the (convex) logarit hmic barrier funct ion for this cone $f : \mathcal { R } ^ { n } \mapsto \mathcal { R }$ 

$$
f (x, t) = \left\{ \begin{array}{l l} - \ln (t ^ {2} - | | x | | ^ {2}) & \text {if} | | x | | <   t \\ + \infty & \text {otherwise.} \end{array} \right.
$$

## Th<sub>eorem:</sub>

$f ( x , t )$ i<sub>s a se</sub>lf<sub>-concor</sub>d<sub>an</sub>t b<sub>arr</sub>i<sub>er on</sub> $K _ { q }$ 

E<sub>xerc</sub>i<sub>se :</sub> P<sub>rove</sub> it i<sub>n case</sub> $n = 2$ 

# Semidefinite Programming (SDP)

## SDP <sub>:</sub> B<sub>ac</sub>k<sub>groun</sub>d

D<sub>e</sub>f<sub>.</sub> A <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $H \in \mathcal { R } ^ { n \times n }$ i<sub>s p os</sub>iti<sub>ve sem</sub>id<sub>e</sub>fi<sub>n</sub>it<sub>e</sub> if $x ^ { T } H x \geq 0$ <sup>f</sup>or an<sub>y</sub> $x \neq 0$ W<sub>e wr</sub>it<sub>e</sub> $H \succeq 0$ 

D<sub>e</sub>f<sub>.</sub> A <sub>ma</sub>t<sub>r</sub>i<sub>x</sub> $H \in \mathcal { R } ^ { n \times n }$ i<sub>s p os</sub>it i<sub>ve</sub> d<sub>e</sub>fi<sub>n</sub>it<sub>e</sub> if $x ^ { T } H x > 0$ <sup>f</sup>or an<sub>y</sub> $x \neq 0$ <sub>.</sub> W<sub>e wr</sub>it<sub>e</sub> $H \succ 0$ 

W<sub>e</sub> d<sub>eno</sub>t<sub>e w</sub>ith $S \mathcal { R } ^ { n \times n } \left( S \mathcal { R } _ { + } ^ { n \times n } \right)$ th<sub>e se</sub>t <sub>o</sub>f <sub>symme</sub>t<sub>r</sub>i<sub>c an</sub>d <sub>sym-</sub> <sub>me</sub>t<sub>r</sub>i<sub>c pos</sub>iti<sub>ve sem</sub>id<sub>e</sub>fi<sub>n</sub>it<sub>e ma</sub>t<sub>r</sub>i<sub>ces .</sub> 

L<sub>e</sub>t U $V \in S \mathcal { R } ^ { n \times n }$ <sub>.</sub> W<sub>e</sub> d<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> i<sub>nner pro</sub>d<sub>uc</sub>t b<sub>e</sub>t<sub>ween</sub> U <sub>an</sub>d V <sub>as</sub> $U \bullet V = t r a c e ( U ^ { T } V )$ <sub>w</sub>h<sub>ere</sub> t<sub>race</sub> $\textstyle ( H ) = \sum _ { i = 1 } ^ { n } h _ { i i }$ 

Th<sub>e assoc</sub>i<sub>a</sub>t<sub>e</sub>d <sub>norm</sub> i<sub>s</sub> th<sub>e</sub> F<sub>ro</sub>b<sub>en</sub>i<sub>us norm</sub> written $\| U \| _ { F } = ( U \bullet U ) ^ { 1 / 2 } \left( \mathrm { o r ~ j u s t } \ \| U \| \right)$ 

## Li<sub>near</sub> M<sub>a</sub>t<sub>r</sub>i<sub>x</sub> I<sub>nequa</sub>liti<sub>es</sub>

Def<sub>.</sub> Li<sub>near</sub> M<sub>a</sub>t<sub>r</sub>i<sub>x</sub> I<sub>nequa</sub>liti<sub>es</sub> L<sub>e</sub>t $U , V \in { \mathcal { S R } } ^ { n \times n }$ 

W<sub>e wr</sub>it<sub>e</sub> $U \succeq V$ if $U - V \succeq 0$ 

W<sub>e wr</sub>it<sub>e</sub> $U \succ V$ if $U - V \succ 0 .$ 

W<sub>e wr</sub>it<sub>e</sub> $U \preceq V$ if $U - V \preceq 0 .$ 

W<sub>e wr</sub>it<sub>e</sub> $U \prec V$ if $U - V \prec 0$ 

## P<sub>roper</sub>t i<sub>es</sub>

1 <sub>.</sub> I f $P \in \mathcal { R } ^ { m \times n }$ <sub>an</sub>d $Q \in \mathcal { R } ^ { n \times m }$ th<sub>en</sub> t<sub>race</sub> $( P Q ) = t r a c e ( Q P )$ 

2 <sub>.</sub> I f $U , V \in { \mathcal { S R } } ^ { n \times n }$ <sub>, an</sub>d $Q \in \mathcal { R } ^ { n \times n }$ is ort hogonal ( i <sub>.</sub> e <sub>.</sub> $Q ^ { T } Q = I )$ th<sub>en</sub> $U \bullet V = ( Q ^ { T } U Q ) \bullet ( Q ^ { T } V Q )$ M<sub>ore genera</sub>ll<sub>y</sub> if P i<sub>s nons</sub>i<sub>ngu</sub>l<sub>ar</sub> th<sub>en</sub> $U \bullet V = ( P U P ^ { T } ) \bullet ( P ^ { - T } V { \breve { P } } ^ { - 1 } )$ 

3 <sub>.</sub> E<sub>very</sub> $U \in { \mathcal { S } } { \mathcal { R } } ^ { n \times n }$ <sub>can</sub> b<sub>e wr</sub>itt<sub>en as</sub> $U = Q \Lambda Q ^ { T } ,$ where Q is <sub>or</sub>th<sub>ogona</sub>l <sub>an</sub>d Λ i<sub>s</sub> di<sub>agona</sub>l <sub>.</sub> Th<sub>en</sub> $U Q = Q \Lambda$ In other words the columns of Q are the eigenvectors and the diag-<sub>ona</sub>l <sub>en</sub>t<sub>r</sub>i<sub>es o</sub>f Λ th<sub>e correspon</sub>di<sub>ng e</sub>i<sub>genva</sub>l<sub>ues o</sub>f U <sub>.</sub> 

4 <sub>.</sub> I f $U \in { \mathcal { S } } { \mathcal { R } } ^ { n \times n }$ <sub>an</sub>d $U = Q \Lambda Q ^ { T }$ <sub>,</sub> th<sub>en</sub> trace $\begin{array} { r } { \left( U \right) = t r a c e ( \Lambda ) = \sum _ { i } \dot { \lambda _ { i } } } \end{array}$ 

## Propert ies (cont <sup>’</sup> d)

5. For $U \in S R^{n \times n}$ , the following are equivalent:
(i) $U \succeq 0 (U \succ 0)$ (ii) $x^{T}Ux \geq 0, \forall x \in R^{n} (x^{T}Ux > 0, \forall 0 \neq x \in R^{n})$ .
(iii) If $U = Q\Lambda Q^{T}$ , then $\Lambda \succeq 0 (\Lambda \succ 0)$ .
(iv) $U = P^{T}P$ for some matrix $P(U = P^{T}P$ for some square nonsingular matrix P).
6. Every $U \in S R^{n \times n}$ has a square root $U^{1/2} \in S R^{n \times n}$ .
Proof: From Property 5 (ii) we get $U = Q\Lambda Q^{T}$ .
Take $U^{1/2} = Q\Lambda^{1/2}Q^{T}$ , where $\Lambda^{1/2}$ is the diagonal matrix whose diagonal contains the (nonnegative) square roots of the eigenvalues of U, and verify that $U^{1/2}U^{1/2} = U$ . 

Propert ies (cont <sup>’</sup> d) 

7<sub>.</sub> S<sub>uppose</sub> 

$$
U = \left[ \begin{array}{l l} A & B ^ {T} \\ B & C \end{array} \right],
$$

<sub>w</sub>h<sub>ere</sub> A <sub>an</sub>d C <sub>are symme</sub>t<sub>r</sub>i<sub>c an</sub>d $A \succ 0$ 

Th<sub>en</sub> $U \succeq 0 \left( U \succ 0 \right) \quad \mathrm { ~ i f ~ } \quad C - B A ^ { - 1 } B ^ { T } \succeq 0 \left( \succ 0 \right) .$ 

Th<sub>e ma</sub>t<sub>r</sub>i<sub>x</sub> $\dot { C } - B A ^ { - 1 } B ^ { T }$ i<sub>s ca</sub>ll<sub>e</sub>d th<sub>e</sub> S<sub>c</sub>h<sub>ur comp</sub> l<sub>em en</sub>t <sub>o</sub>f A i<sub>n</sub> U <sub>.</sub> 

P <sub>ro o</sub>f<sub>:</sub> f<sub>o</sub>ll<sub>ows eas</sub>il<sub>y</sub> f<sub>rom</sub> th<sub>e</sub> f<sub>ac</sub>t<sub>or</sub>i<sub>za</sub>ti<sub>on :</sub> 

$$
\left[ \begin{array}{c c} A & B ^ {T} \\ B & C \end{array} \right] = \left[ \begin{array}{c c} I & 0 \\ B A ^ {- 1} & I \end{array} \right] \left[ \begin{array}{c c} A & 0 \\ 0 & C - B A ^ {- 1} B ^ {T} \end{array} \right] \left[ \begin{array}{c c} I & A ^ {- 1} B ^ {T} \\ 0 & I \end{array} \right].
$$

8 <sub>.</sub> I f $U \in { \mathcal { S } } { \mathcal { R } } ^ { n \times n }$ <sub>an</sub>d $x \in \mathcal { R } ^ { n }$ <sub>,</sub> th<sub>en</sub> $x ^ { T } U x = U \bullet x x ^ { T }$ 

## P<sub>r</sub>i<sub>ma</sub>l<sub>-</sub>D<sub>ua</sub>l P<sub>a</sub>i<sub>r o</sub>f SDP<sub>s</sub>

<sub>m</sub>i<sub>n</sub> $C \bullet X$ 

s <sub>.</sub> t <sub>.</sub> $A _ { i } \bullet X = b _ { i } , i = 1 . . m$ 

$$
b ^ {T} y
$$

s <sub>.</sub> t <sub>.</sub> $\textstyle \sum _ { i = 1 } ^ { m } y _ { i } A _ { i } + S = C ,$ 

$$
X \succeq 0;
$$

$$
S \succeq 0,
$$

<sub>w</sub>h<sub>ere</sub> $A _ { i } \in \mathcal { S R } ^ { n \times n } , \ b \in \mathcal { R } ^ { m } , \ C \in \mathcal { S R } ^ { n \times n }$ a<sup>r</sup>e g<sup>i</sup>ve<sup>n</sup> ; <sub>an</sub>d $\ b X , \ b S \in { S \mathcal R ^ { n \times n } } , \ \ b y \in \mathcal R ^ { m }$ <sub>are</sub> th<sub>e var</sub>i<sub>a</sub>bl<sub>es .</sub> 

Si<sub>mp</sub>lifi<sub>e</sub>d <sub>no</sub>t<sub>a</sub>ti<sub>on :</sub> 

<table><tr><td>Primal</td><td>Dual</td></tr><tr><td>min C • X</td><td>max b^T y</td></tr><tr><td>s.t. AX = b,</td><td>s.t. A*y + S = C,</td></tr><tr><td>X ≥ 0;</td><td>S ≥ 0.</td></tr></table>

## Th<sub>eorem:</sub> W<sub>ea</sub>k D<sub>ua</sub>lit<sub>y</sub> i<sub>n</sub> SDP

If X i<sub>s</sub> f<sub>eas</sub>ibl<sub>e</sub> i<sub>n</sub> th<sub>e pr</sub>i<sub>ma</sub>l <sub>an</sub>d $( y , S )$ i<sub>n</sub> th<sub>e</sub> d<sub>ua</sub>l th<sub>en</sub> 

$$
C \bullet X - b _ {m} ^ {T} y = X \bullet S \geq 0.
$$

P<sub>roo</sub>f<sub>:</sub> 

$$
\begin{array}{r l} & C \bullet X - b ^ {T} y = (\sum_ {i = 1} ^ {m} y _ {i} A _ {i} + S) \bullet X - b ^ {T} y \\ & \qquad = \sum_ {i = 1} ^ {m} (A _ {i} \bullet X) y _ {i} + S \bullet X - b ^ {T} y \\ & \qquad = S \bullet X = X \bullet S. \end{array}
$$

F<sub>ur</sub>th<sub>er , s</sub>i<sub>nce</sub> $X$ i<sub>s pos</sub>iti<sub>ve sem</sub>id<sub>e</sub>fi<sub>n</sub>it<sub>e ,</sub> it h<sub>as a square roo</sub>t $X ^ { 1 / 2 }$ ( Property 6) <sub>,</sub> and so 

$$
X \bullet S = t r a c e (X S) = t r a c e (X ^ {1 / 2} X ^ {1 / 2} S) = t r a c e (X ^ {1 / 2} S X ^ {1 / 2}) \geq 0.
$$

W<sub>e use</sub> P<sub>roper</sub>t<sub>y</sub> 1 <sub>an</sub>d th<sub>e</sub> f<sub>ac</sub>t th<sub>a</sub>t $S$ <sub>an</sub>d $X ^ { 1 / 2 }$ <sub>are pos</sub>iti<sub>ve</sub> <sub>sem</sub>id<sub>e</sub>fi<sub>n</sub>it<sub>e</sub> h<sub>ence</sub> $X ^ { 1 / 2 } S X ^ { 1 / 2 }$ i<sub>s pos</sub>iti<sub>ve sem</sub>id<sub>e</sub>fi<sub>n</sub>it<sub>e an</sub>d it<sub>s</sub> t<sub>race</sub> i<sub>s</sub> n<sub>o</sub>nn<sub>ega</sub>ti<sub>ve .</sub> 

SDP E<sub>xamp</sub>l<sub>e</sub> 1 <sub>:</sub> Mi<sub>n</sub>i<sub>m</sub>i<sub>ze</sub> th<sub>e</sub> M<sub>ax.</sub> Ei<sub>genva</sub>l<sub>ue</sub> W<sub>e w</sub>i<sub>s</sub>h t<sub>o c</sub>h<sub>oose</sub> $\boldsymbol { x } \in \mathcal { R } ^ { k }$ t<sub>o m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>ze</sub> th<sub>e max</sub>i<sub>mum e</sub>i<sub>genva</sub>l<sub>ue o</sub>f $A ( x ) = A _ { 0 } + x _ { 1 } A _ { 1 } + . . . + x _ { k } A _ { k }$ <sub>w</sub>h<sub>ere</sub> $A _ { i } \in \mathcal { R } ^ { n \times n }$ <sub>an</sub>d ${ \ddot { A } } _ { i } = A _ { i } ^ { T }$ Ob<sub>serve</sub> th<sub>a</sub>t 

$$
\lambda_ {m a x} (A (x)) \leq t
$$

if <sub>an</sub>d <sub>on</sub>l<sub>y</sub> if 

$$
\lambda_ {m a x} (A (x) - t I) \leq 0 \quad \Longleftrightarrow \quad \lambda_ {m i n} (t I - A (x)) \geq 0.
$$

Thi<sub>s</sub> h<sub>o</sub>ld<sub>s</sub> if 

$$
t I - A (x) \succeq 0.
$$

S<sub>o we ge</sub>t th<sub>e</sub> S DP i<sub>n</sub> th<sub>e</sub> d<sub>ua</sub>l f<sub>orm :</sub> 

$$
\max - t
$$

$$
\mathrm{s.t.} t I - A (x) \succeq 0,
$$

<sub>w</sub>h<sub>ere</sub> th<sub>e var</sub>i<sub>a</sub>bl<sub>e</sub> i<sub>s</sub> $y : = ( t , x )$ 

## L<sub>ogar</sub>ith<sub>m</sub>i<sub>c</sub> B<sub>arr</sub>i<sub>er</sub> F<sub>unc</sub>ti<sub>on</sub>

D<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> l<sub>ogar</sub>it h<sub>m</sub>i<sub>c</sub> b<sub>arr</sub>i<sub>er</sub> f<sub>unc</sub>t i<sub>on</sub> f<sub>or</sub> th<sub>e cone</sub> $S R _ { + } ^ { n \times n }$ <sub>o</sub>f <sub>pos</sub>iti<sub>ve</sub> d<sub>e</sub>fi<sub>n</sub>it<sub>e ma</sub>t<sub>r</sub>i<sub>ces .</sub> 

$$
f: \mathcal {S R} _ {+} ^ {n \times n} \mapsto \mathcal {R}
$$

$$
f (X) = \left\{ \begin{array}{l l} - \ln \det X & \text {if} X \succ 0 \\ + \infty & \text {otherwise.} \end{array} \right.
$$

L<sub>e</sub>t <sub>us eva</sub>l<sub>ua</sub>t<sub>e</sub> it<sub>s</sub> d<sub>er</sub>i<sub>va</sub>ti<sub>ves .</sub> 

L<sub>e</sub>t $\textstyle X \succ 0 , H \in S { \mathcal { R } } ^ { n \times n }$ <sub>.</sub> Th<sub>en</sub> 

$$
\begin{array}{r l} {f (X + \alpha H) = - \ln \det [ X (I + \alpha X ^ {- 1} H) ]} \\ {=} & {- \ln \det X - \ln (1 + \alpha t r a c e (X ^ {- 1} H) + \mathcal {O} (\alpha^ {2}))} \\ {=} & {f (X) - \alpha X ^ {- 1} \bullet H + \mathcal {O} (\alpha^ {2}),} \end{array}
$$

<sub>so</sub> th<sub>a</sub>t $f ^ { \prime } ( X ) = - X ^ { - 1 }$ <sub>an</sub>d $D f ( X ) [ H ] = - X ^ { - 1 } \bullet H$ 

## Logarithmic Barrier Function (cont <sup>’</sup>d)

S i<sub>m</sub>il<sub>ar</sub>l<sub>y</sub> 

$$
\begin{array}{r l} f ^ {\prime} (X + \alpha H) & = - [ X (I + \alpha X ^ {- 1} H) ] ^ {- 1} \\ & = - [ I - \alpha X ^ {- 1} H + \mathcal {O} (\alpha^ {2}) ] X ^ {- 1} \\ & = f ^ {\prime} (X) + \alpha X ^ {- 1} H X ^ {- 1} + \mathcal {O} (\alpha^ {2}), \end{array}
$$

<sub>so</sub> th<sub>a</sub>t $f ^ { \prime \prime } ( X ) [ H ] = X ^ { - 1 } H X ^ { - 1 }$ 

<sub>an</sub>d $D ^ { 2 } f ( X ) [ H , G ] = X ^ { - 1 } H X ^ { - 1 } \bullet G .$ 

$$
\begin{array}{l} \text {Finally,} \\ f ^ {\prime \prime \prime} (X) [ H, G ] = - X ^ {- 1} H X ^ {- 1} G X ^ {- 1} - X ^ {- 1} G X ^ {- 1} H X ^ {- 1}. \end{array}
$$

## Logarithmic Barrier Function (cont <sup>’</sup>d)

Th<sub>eorem:</sub> $f ( X ) = -$ l<sub>n</sub> d<sub>e</sub>t X i<sub>s a convex</sub> b<sub>arr</sub>i<sub>er</sub> f<sub>or</sub> $S R _ { + } ^ { n \times n }$ 

P<sub>roo</sub>f<sub>:</sub> D<sub>e</sub>fi<sub>ne</sub> $\phi ( \alpha ) = f ( X + \alpha H )$ <sub>.</sub> W<sub>e</sub> k<sub>now</sub> th<sub>a</sub>t $f$ i<sub>s convex</sub> if <sup>f</sup>or ever<sub>y</sub> $X \in S \mathcal { R } _ { + } ^ { n \times n }$ an<sup>d</sup> ever<sub>y</sub> $H \in S \mathcal { R } ^ { n \times n } , \phi ( \alpha )$ i<sub>s convex</sub> i<sub>n</sub> α <sub>.</sub> 

C<sub>ons</sub>id<sub>er a se</sub>t <sub>o</sub>f <sub>α suc</sub>h th<sub>a</sub>t $X + \alpha H \vdash 0$ <sub>.</sub> O <sub>n</sub> t hi<sub>s se</sub>t 

$$
\phi^ {\prime \prime} (\alpha) = D ^ {2} f (\bar {X}) [ H, H ] = \bar {X} ^ {- 1} H \bar {X} ^ {- 1} \bullet H,
$$

<sub>w</sub>h<sub>ere</sub> $\bar { X } = X + \alpha H$ 

Si<sub>nce</sub> $\bar { X } \succ 0$ <sub>, so</sub> i<sub>s</sub> $V = \bar { X } ^ { - 1 / 2 }$ ( Property 6) <sub>,</sub> and 

$$
\begin{array}{r l} & {\phi^ {\prime \prime} (\alpha) = V ^ {2} H V ^ {2} \bullet H = t r a c e (V ^ {2} H V ^ {2} H)} \\ & {\quad = t r a c e ((V H V) (V H V)) = \| V H V \| _ {F} ^ {2} \geq 0.} \end{array}
$$

S<sub>o</sub> $\phi$ i<sub>s convex.</sub> 

Wh<sub>en</sub> $X ~ \succ ~ 0$ <sub>approac</sub>h<sub>es a s</sub>i<sub>ngu</sub>l<sub>ar ma</sub>t<sub>r</sub>i<sub>x</sub> it<sub>s</sub> d<sub>e</sub>t<sub>erm</sub>i<sub>nan</sub>t <sub>ap-</sub> <sub>proac</sub>h<sub>es zero an</sub>d $f ( X )  \infty$ 

## S<sub>o</sub>l<sub>v</sub>i<sub>ng</sub> SDP<sub>s w</sub>ith IPM<sub>s</sub>

R<sub>ep</sub>l<sub>ace</sub> th<sub>e pr</sub>i<sub>ma</sub>l SDP 

$$
\begin{array}{r l} \min & C \bullet X \\ \mathrm{s.t.} & \mathcal {A} X = b, \\ & X \succeq 0, \end{array}
$$

<sub>w</sub>ith th<sub>e pr</sub>i<sub>ma</sub>l b<sub>arr</sub>i<sub>er</sub> SDP 

$$
\begin{array}{r l} & {\min C \bullet X + \mu f (X)} \\ & {\mathrm{s.t.} \mathcal {A} X = b,} \end{array}
$$

(with a barrier parameter $\mu \geq 0 )$ 

F<sub>ormu</sub>l<sub>a</sub>t<sub>e</sub> th<sub>e</sub> L<sub>agrang</sub>i<sub>an</sub> 

$$
L (X, y, S) = C \bullet X + \mu f (X) - y ^ {T} (\mathcal {A} X - b),
$$

<sub>w</sub>ith $y \in \mathcal { R } ^ { m }$ and write the first order conditions ( FO C ) for a <sub>s</sub>t<sub>a</sub>ti<sub>onary po</sub>i<sub>n</sub>t <sub>o</sub>f L <sub>:</sub> 

$$
C + \mu f ^ {\prime} (X) - \mathcal {A} ^ {*} y = 0.
$$

## Solving SDPs with IPMs (cont <sup>’</sup>d)

U<sub>se</sub> $f ( X ) = - \ln \operatorname* { d e t } ( X )$ <sub>an</sub>d $f ^ { \prime } ( X ) = - X ^ { - 1 }$ Th<sub>ere</sub>f<sub>ore</sub> th<sub>e</sub> FO C b<sub>ecome :</sub> 

$$
C - \mu X ^ {- 1} - \mathcal {A} ^ {*} y = 0.
$$

D<sub>eno</sub>t<sub>e</sub> $S = \mu X ^ { - 1 } , \mathrm { i . e . , } X S = \mu I$ 

F<sub>or a pos</sub>iti<sub>ve</sub> d<sub>e</sub>fi<sub>n</sub>it<sub>e ma</sub>t<sub>r</sub>i<sub>x</sub> X it<sub>s</sub> i<sub>nverse</sub> i<sub>s a</sub>l<sub>so pos</sub>iti<sub>ve</sub> d<sub>e</sub>fi<sub>n</sub>it<sub>e .</sub> 

Th<sub>e</sub> FOC <sub>now</sub> b<sub>ecome:</sub> 

$$
\mathcal {A} X = b,
$$

$$
\mathcal {A} ^ {*} y + S = C,
$$

$$
X S = \mu I,
$$

<sub>w</sub>ith $X \succ 0$ <sub>an</sub>d $S \succ 0$ 

Th<sub>en app</sub>l<sub>y</sub> N<sub>ew</sub>t<sub>on me</sub>th<sub>o</sub>d t<sub>o</sub> th<sub>e</sub> FOC <sub>.</sub> 

## Th<sub>e</sub> R<sub>an</sub>k Mi<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> P<sub>ro</sub>bl<sub>em</sub>

$$
\begin{array}{l} \min \operatorname{rank} (X) \\ \text {s.t.} \quad \mathcal {A} (X) = b \end{array}
$$

$X \in \mathcal { R } ^ { n \times n }$ i<sub>s</sub> th<sub>e un</sub>k<sub>nown an</sub>d th<sub>e</sub> li<sub>near map</sub> $\mathcal { A } : \mathcal { R } ^ { n \times n }  \mathcal { R } ^ { m }$ <sub>an</sub>d th<sub>e vec</sub>t<sub>or</sub> $b \in \mathcal { R } ^ { m }$ are <sub>g</sub><sup>i</sup>ven <sub>.</sub> 

• NP<sub>-</sub>h<sub>ar</sub>d <sub>pro</sub>bl<sub>em</sub> 

• A<sub>pp</sub>li<sub>ca</sub>ti<sub>ons : ma</sub>t<sub>r</sub>i<sub>x comp</sub>l<sub>e</sub>ti<sub>on</sub> (Netflix problem triangulation from incomplete data) <sub>nonnega</sub>ti<sub>ve</sub> f<sub>ac</sub>t<sub>or</sub>i<sub>za</sub>ti<sub>on con</sub>t<sub>ro</sub>l <sub>an</sub>d <sub>sys</sub>t<sub>em</sub> th<sub>eory</sub> <sup>i</sup>ma<sub>g</sub>e com<sub>p</sub>ress<sup>i</sup>on <sub>.</sub> 

## A R<sub>an</sub>k Mi<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> H<sub>eur</sub>i<sub>s</sub>ti<sub>c</sub>

min rank(X) 

$$
\| X \| _ {*}
$$

h<sub>eur</sub>i<sub>s</sub>t i<sub>c</sub> 

s <sub>.</sub> t <sub>.</sub> ${ \ddot { A } } ( { \ddot { X } } ) = b$ 

<sub>w</sub>h<sub>ere</sub> $\Vert \cdot \Vert _ { * }$ denotes the nuclear norm (the sum of singular values) <sub>.</sub> 

• C<sub>onvex op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em</sub> 

• S<sub>pec</sub>i<sub>a</sub>l <sub>case :</sub> 

i f $X { = } d i a g ( x )$ th<sub>e pro</sub>bl<sub>em re</sub>d<sub>uces</sub> t<sub>o</sub> $\ell _ { 1 } { - } n o r m$ <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on:</sub> 

$$
\begin{array}{l l} \min & \operatorname{card} (x) \\ \text {s.t.} & A x = b \end{array} \quad \underbrace {\Rightarrow} _ {\text {heuristic}} \quad \begin{array}{l l} \min & \| x \| _ {1} \\ \text {s.t.} & A x = b \end{array}
$$

## SDP f<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub>

## Primal-dual convex formulation (heuristic)

<sub>m</sub>i<sub>n</sub> $\| X \| _ { * }$ 

s <sub>.</sub> t <sub>.</sub> ${ \mathcal { A } } ( X ) = b$ 

max $b ^ { T } y$ 

s <sub>.</sub> t <sub>.</sub> $\left| | \mathcal { A } ^ { * } ( y ) \right| | \leq 1$ 

## P<sub>r</sub>i<sub>ma</sub>l<sub>-</sub>d<sub>ua</sub>l SDP f<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub>

<sub>m</sub>i<sub>n</sub> $\begin{array} { r } { \frac { 1 } { 2 } ( T r ( W _ { 1 } ) + T r ( W _ { 2 } ) ) } \end{array}$ max $b ^ { T } y$ 

s <sub>.</sub> t <sub>.</sub> ${ \left[ \begin{array} { l l } { W _ { 1 } } & { X } \\ { X ^ { T } } & { W _ { 2 } } \end{array} \right] } \succeq 0$ 

s <sub>.</sub> t <sub>.</sub> $\left[ \begin{array} { c c c } { { I _ { m } } } & { { \mathcal { A } ^ { * } ( y ) } } \\ { { \mathcal { A } ^ { * } ( y ) ^ { T } } } & { { I _ { n } } } \end{array} \right] \succeq 0$ 

<sub>w</sub>h<sub>ere</sub> $y \in \mathcal { R } ^ { m } , W _ { 1 } , W _ { 2 } \in \mathcal { R } ^ { n \times n }$ 

$\mathcal { A } ^ { * } : \mathcal { R } ^ { m }  \mathcal { R } ^ { n \times n }$ is the adj oint of $A$ 

$\Vert \cdot \Vert$ denotes the operator norm (the maximum singular value) <sub>.</sub> 

## Netflix Problem (Matrix Completion) R<sub>ecommen</sub>d<sub>er sys</sub>t<sub>ems</sub>

amazon.com 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/9ecf958e-cbed-49b1-9728-6c74857ccd0b/5c00477606533847497a89ea69386a6fb1b9ba1fed54e4ffaa249cf487437387.jpg)


NETFLIX 

The Netflix Prize ( $ 1 M ) : In 2006 Netflix held the first Netflix P<sub>r</sub>i<sub>ze compe</sub>titi<sub>on</sub> t<sub>o</sub> fi<sub>n</sub>d <sub>a</sub> b<sub>e</sub>tt<sub>er program</sub> t<sub>o pre</sub>di<sub>c</sub>t <sub>user pre</sub>f<sub>er-</sub> <sub>ences an</sub>d b<sub>ea</sub>t it<sub>s ex</sub>i<sub>s</sub>ti<sub>ng</sub> N<sub>e</sub>tfli<sub>x mov</sub>i<sub>e recommen</sub>d<sub>a</sub>ti<sub>on sys</sub>t<sub>em</sub> b<sub>y a</sub>t l<sub>eas</sub>t 1 0% <sub>.</sub> 

• Gi<sub>ven</sub> 1 00 <sub>m</sub>illi<sub>on ra</sub>ti<sub>ngs on a sca</sub>l<sub>e o</sub>f 1 t<sub>o</sub> 5 <sub>pre</sub>di<sub>c</sub>t 3 <sub>m</sub>illi<sub>on ra</sub>ti<sub>ngs</sub> t<sub>o</sub> hi<sub>g</sub>h<sub>es</sub>t <sub>accuracy</sub> 

• 1 7770 t<sub>o</sub>t<sub>a</sub>l <sub>mov</sub>i<sub>es x</sub> 480 1 89 t<sub>o</sub>t<sub>a</sub>l <sub>users</sub> ⇒ <sub>over</sub> 8 billi<sub>on</sub> t<sub>o</sub>t<sub>a</sub>l <sub>ra</sub>ti<sub>ngs</sub> 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/9ecf958e-cbed-49b1-9728-6c74857ccd0b/848f2e999c4ae9f10c96bdca68609359d6ee936c6c20bd066b67ed771209b9ed.jpg)


$B _ { i j }$ k<sub>nown</sub> f<sub>or</sub> bl<sub>ac</sub>k <sub>ce</sub>ll<sub>s un</sub>k<sub>nown</sub> f<sub>or w</sub>hit<sub>e</sub> R<sub>ow</sub> i<sub>n</sub>d<sub>ex: mov</sub>i<sub>e</sub> C<sub>o</sub>l<sub>umn</sub> i<sub>n</sub>d<sub>ex: us er</sub> Fi<sub>n</sub>d l<sub>ow-ran</sub>k W <sub>suc</sub>h th<sub>a</sub>t W ≈ B 

## Th<sub>e</sub> M<sub>a</sub>t<sub>r</sub>i<sub>x</sub> C<sub>omp</sub>l<sub>e</sub>ti<sub>on</sub> P<sub>ro</sub>bl<sub>em</sub>

A <sub>sma</sub>ll <sub>num</sub>b<sub>er o</sub>f <sub>en</sub>t<sub>r</sub>i<sub>es o</sub>f <sub>a ma</sub>t<sub>r</sub>i<sub>x</sub> $\boldsymbol { B } \in \mathcal { R } ^ { \hat { m } \times \hat { n } }$ i<sub>s</sub> k<sub>nown :</sub> <sub>a</sub>ll <sub>en</sub>t <sub>r</sub>i<sub>es</sub> $B _ { i , j }$ <sub>w</sub>ith $( i , j ) \in \Omega$ <sub>w</sub>h<sub>ere</sub> $| \Omega | = m \ll { \hat { m } } { \hat { n } }$ Fi<sub>n</sub>d <sub>an approx</sub>i<sub>ma</sub>ti<sub>on</sub> $W \in \mathcal { R } ^ { \hat { m } \times \hat { n } }$ <sub>o</sub>f B <sub>suc</sub>h th<sub>a</sub>t <sub>:</sub> 

• W h<sub>as sma</sub>ll <sub>ran</sub>k <sub>an</sub>d 

• W <sub>an</sub>d B <sub>agree on</sub> $\Omega .$ 

## M<sub>a</sub>t<sub>r</sub>i<sub>x</sub> C<sub>omp</sub>l<sub>e</sub>ti<sub>on</sub> P<sub>ro</sub>bl<sub>em</sub>

min rank(W) 

$$
\mathrm{s.t.} W _ {i j} = B _ {i j}, \forall (i, j) \in \Omega .
$$

## SDP R<sub>e</sub>l<sub>axa</sub>ti<sub>on o</sub>f M<sub>a</sub>t<sub>r</sub>i<sub>x</sub> C<sub>omp</sub>l<sub>e</sub>ti<sub>on</sub> SDP R<sub>e</sub>l<sub>axa</sub>ti<sub>on</sub>

<sub>m</sub>i<sub>n</sub> ${ \frac { 1 } { 2 } } ( T r ( W _ { 1 } ) + T r ( W _ { 2 } ) )  C \bullet X$ 

$$
\left[ \begin{array}{c c} W _ {1} & W \\ W ^ {T} & W _ {2} \end{array} \right] \succeq 0 \qquad \leftrightarrow X \succeq 0\tag{s.t.}
$$

$$
\bar {W} _ {i j} = B _ {i j} ^ {-} (i, j) \in \Omega \leftrightarrow A _ {l} \bullet X = b _ {l}
$$

$W \in \mathcal { R } ^ { \hat { m } \times \hat { n } } , W _ { 1 } \in \mathcal { R } ^ { \hat { m } \times \hat { m } } , W _ { 2 } \in \mathcal { R } ^ { \hat { n } \times \hat { n } }$ <sub>un</sub>k<sub>nowns</sub> $B _ { i j } , ( i , j ) \in \Omega$ <sub>g</sub><sup>i</sup>ven 

$\bullet \ C = I _ { n } , X = \left[ W _ { 1 } \ W \right] \in \mathcal { R } ^ { n \times n } .$ <sub>w</sub>ith $n = ( \hat { m } + \hat { n } )$ 

$A _ { l } = \textstyle { \frac { 1 } { 2 } } \left[ { \underset { ( \Theta ^ { i j } ) ^ { T } } { 0 } } \Theta ^ { i j } \right] , l = 1 , \dots , m$ <sub>w</sub>h<sub>ere</sub> f<sub>or eac</sub>h $( i , j ) \in \Omega$ 

$\Theta ^ { i j } \in \mathcal { R } ^ { \mathbf { \bar { \omega } } \times \hat { n } } \colon ~ ( \Theta ^ { i j } ) _ { s t } = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ } ( s , t ) = ( i , j ) } \\ { 0 } & { \mathrm { e l s e } } \end{array} \right. ( A _ { l }$ of rank 2) <sub>.</sub> 

## L<sub>ogar</sub>ith<sub>m</sub>i<sub>c</sub> B<sub>arr</sub>i<sub>er</sub> F<sub>unc</sub>ti<sub>on</sub>

f<sub>or</sub> th<sub>e cone</sub> $S R _ { + } ^ { n \times n }$ <sub>o</sub>f <sub>pos</sub>iti<sub>ve</sub> d<sub>e</sub>fi<sub>n</sub>it<sub>e ma</sub>t<sub>r</sub>i<sub>ces</sub> $f : S \mathcal { R } _ { + } ^ { n \times n } \mapsto \mathcal { R }$ 

$$
f (X) = \left\{ \begin{array}{l l} - \ln \det X & \text {if} X \succ 0 \\ + \infty & \text {otherwise.} \end{array} \right.
$$

LP <sub>:</sub> R<sub>ep</sub>l<sub>ace</sub> $x \geq 0$ <sub>w</sub>ith $- \mu \sum _ { j = 1 } ^ { n } \ln x _ { j }$ 

SDP <sub>:</sub> R<sub>ep</sub>l<sub>ace</sub> $X \succeq 0$ <sub>w</sub>ith $- \mu \textstyle \sum _ { j = 1 } ^ { n } \ln \lambda _ { j } = - \mu \ln ( \prod _ { j = 1 } ^ { n } \lambda _ { j } )$ 

N<sub>es</sub>t<sub>erov an</sub>d N<sub>em</sub>i<sub>rovs</sub>kii I<sub>n</sub>t<sub>er</sub>i<sub>or</sub> P<sub>o</sub>i<sub>n</sub>t P<sub>o</sub>l<sub>ynom</sub>i<sub>a</sub>l A l<sub>gor</sub>ith<sub>ms</sub> i<sub>n</sub> C<sub>onvex</sub> P<sub>rogramm</sub>i<sub>ng:</sub> Th<sub>eory an</sub>d A<sub>pp</sub>li<sub>ca</sub>ti<sub>ons</sub> SIAM Phil<sub>a</sub>d<sub>e</sub>l<sub>p</sub>hi<sub>a</sub> 1 994 <sub>.</sub> 

L<sub>emma</sub> Th<sub>e</sub> b<sub>arr</sub>i<sub>er</sub> f<sub>unc</sub>ti<sub>on</sub> $f ( X )$ i<sub>s se</sub>lf<sub>-concor</sub>d<sub>an</sub>t <sub>on</sub> $S R _ { + } ^ { n \times n }$ 

## I<sub>n</sub>t<sub>er</sub>i<sub>or</sub> P<sub>o</sub>i<sub>n</sub>t M<sub>e</sub>th<sub>o</sub>d<sub>s:</sub>

• Logarithmic barrier functions for LP QP SOCP and SDP S<sub>e</sub>lf<sub>-concor</sub>d<sub>an</sub>t b<sub>arr</sub>i<sub>ers</sub> → polynomial complexity (predictable behaviour) 

• U<sub>n</sub>ifi<sub>e</sub>d <sub>v</sub>i<sub>ew o</sub>f <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> → from LP via QP to NLP SOCP SDP 

• Efi<sub>c</sub>i<sub>ency</sub> 

– <sub>goo</sub>d f<sub>or</sub> SOCP 

– <sub>pro</sub>bl<sub>ema</sub>ti<sub>c</sub> f<sub>or</sub> S D P b<sub>ecause so</sub>l<sub>v</sub>i<sub>ng</sub> th<sub>e pro</sub>bl<sub>em o</sub>f <sub>s</sub>i<sub>ze</sub> <sub>n</sub> i<sub>nvo</sub>l<sub>ves</sub> li<sub>near a</sub>l<sub>ge</sub>b<sub>ra opera</sub>ti<sub>ons</sub> i<sub>n</sub> di<sub>mens</sub>i<sub>on</sub> $n ^ { 2 }$ → <sub>an</sub>d thi<sub>s requ</sub>i<sub>res</sub> $n ^ { 6 }$ fl<sub>ops</sub> ! 

## U<sub>se</sub> IPM<sub>s</sub> i<sub>n your researc</sub>h !