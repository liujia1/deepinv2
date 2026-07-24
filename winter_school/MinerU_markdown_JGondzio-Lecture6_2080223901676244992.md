![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/71628f03-ee95-4eab-ae41-e4053879471b/279651799be62b55c6fdbe7271e01c4b4b2a8a3931ff6cab3c0394398b299b26.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/71628f03-ee95-4eab-ae41-e4053879471b/97e54668437cd9698c6ea427cba22fdc95a22ab85cc68d6b2725f7045511ba8a.jpg)


# Alt<sub>erna</sub>ti<sub>ng</sub> Di<sub>rec</sub>ti<sub>on</sub> Method of Multipliers (ADMM)

J<sub>ace</sub>k G<sub>on</sub>d<sub>z</sub>i<sub>o</sub> E<sub>ma</sub>il <sub>:</sub> J <sub>.</sub> G<sub>on</sub>d<sub>z</sub> i <sub>o</sub>@<sub>e</sub>d <sub>. ac . u</sub>k URL : htt<sub>p</sub> : / /www <sub>.</sub> maths <sub>.</sub> ed <sub>.</sub> ac <sub>.</sub> uk/ <sup>~</sup> <sub>g</sub>ondz i o 

## Alt<sub>erna</sub>ti<sub>ng</sub> Di<sub>rec</sub>ti<sub>on</sub> M<sub>e</sub>th<sub>o</sub>d <sub>o</sub>f M<sub>u</sub>lti<sub>p</sub>li<sub>ers</sub> (ADMM)

• E<sub>xp</sub>l<sub>o</sub>it<sub>s</sub> D<sub>ua</sub>lit<sub>y</sub> 

• H <sub>as</sub> i<sub>nexpens</sub>i<sub>ve</sub> it<sub>era</sub>ti<sub>ons</sub> 

• S<sub>u</sub>it<sub>a</sub>bl<sub>e</sub> f<sub>or pro</sub>bl<sub>ems w</sub>ith l<sub>oose</sub>l<sub>y coup</sub>l<sub>e</sub>d <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> 

• N<sub>umerous app</sub>li<sub>ca</sub>ti<sub>ons :</sub> 

– machine learning/statistics (large data sets) <sub>,</sub> 

– <sup>i</sup>mage process<sup>i</sup>ng <sub>,</sub> 

– d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on .</sub> 

## D<sub>ua</sub>l D<sub>ecompos</sub>iti<sub>on</sub>

C<sub>ons</sub>id<sub>er equa</sub>lit<sub>y cons</sub>t<sub>ra</sub>i<sub>ne</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>em</sub> 

$$
\begin{array}{c c} \min & f (x) \\ \mathrm{s.t.} & A x = b \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $x \in \mathcal { R } ^ { n } , f : \mathcal { R } ^ { n } { \mapsto } \mathcal { R } , A { \in } \mathcal { R } ^ { m \times n } , b { \in } \mathcal { R } ^ { m }$ <sub>.</sub> U<sub>sua</sub>ll<sub>y</sub> $m \leq n$ In this lecture f is convex<sub>.</sub> ( In general it does not have to be <sub>.</sub> ) W<sub>e assoc</sub>i<sub>a</sub>t<sub>e</sub> L<sub>agrange mu</sub>lti<sub>p</sub>li<sub>ers</sub> $y \in \mathcal { R } ^ { m }$ <sub>w</sub>ith <sub>equa</sub>lit<sub>y cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s</sub> $A x = b$ <sub>an</sub>d <sub>wr</sub>it<sub>e</sub> th<sub>e</sub> L<sub>agrang</sub>i<sub>an:</sub> 

$$
L (x, y) = f (x) + y ^ {T} (A x - b),
$$

d<sub>ua</sub>l f<sub>unc</sub>ti<sub>on :</sub> 

$$
L _ {D} (y) = \inf _ {x} L (x, y)
$$

<sub>an</sub>d d<sub>ua</sub>l <sub>pro</sub>bl<sub>em :</sub> 

$$
\max _ {y} L _ {D} (y).
$$

H<sub>av</sub>i<sub>ng</sub> f<sub>oun</sub>d th<sub>e so</sub>l<sub>u</sub>ti<sub>on o</sub>f th<sub>e</sub> d<sub>ua</sub>l <sub>a</sub>t $\hat { y }$ we recover 

$$
\hat {x} = \operatorname{argmin} _ {x} L (x, \hat {y}).
$$

## D<sub>ua</sub>l A<sub>scen</sub>t M<sub>e</sub>th<sub>o</sub>d

Apply a (simple) gradient method for the dual problem i <sub>.</sub> e <sub>.</sub> make <sub>s</sub>t<sub>eps</sub> i<sub>n</sub> di<sub>rec</sub>ti<sub>on o</sub>f $\nabla L _ { D } ( y )$ 

$$
y ^ {k + 1} = y ^ {k} + \alpha^ {k} \nabla L _ {D} (y ^ {k}).
$$

Ob<sub>serve</sub> th<sub>a</sub>t 

$$
\nabla L _ {D} (y ^ {k}) = A \tilde {x} - b,
$$

<sub>w</sub>h<sub>ere</sub> $\tilde { x } = \mathrm { a r g m i n } _ { x } L ( x , y ^ { k } )$ 

D<sub>ua</sub>l A<sub>scen</sub>t M<sub>e</sub>th<sub>o</sub>d<sub>:</sub> 

<sub>repea</sub>t <sub>un</sub>til <sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> i<sub>s reac</sub>h<sub>e</sub>d <sub>:</sub> 

$$
x ^ {k + 1} = \operatorname{argmin} _ {x} L (x, y ^ {k})
$$

$$
y ^ {k + 1} = y ^ {k} + \alpha^ {k} (A x ^ {k + 1} - b)
$$

Th<sub>eory:</sub> 

St<sub>rong assump</sub>ti<sub>ons are requ</sub>i<sub>re</sub>d f<sub>or suc</sub>h <sub>a s</sub>i<sub>mp</sub>l<sub>e me</sub>th<sub>o</sub>d t<sub>o wor</sub>k<sub>.</sub> 

Dual Decomposition and Separable Objective Suppose the obj ective function is separable : 

$$
f (x) = f _ {1} (x _ {1}) + f _ {2} (x _ {2}) + \dots + f _ {p} (x _ {p}), \quad x = (x _ {1}, x _ {2}, \ldots , x _ {p}).
$$

R<sub>ewr</sub>it<sub>e</sub> th<sub>e pro</sub>bl<sub>em</sub> i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> f<sub>orm :</sub> 

$$
\begin{array}{r l} & {\min f _ {1} (x _ {1}) + f _ {2} (x _ {2}) + \dots + f _ {p} (x _ {p})} \\ & {\mathrm{s.t.} A _ {1} x _ {1} + A _ {2} x _ {2} + \dots + A _ {p} x _ {p} = b} \end{array}
$$

<sub>an</sub>d <sub>o</sub>b<sub>serve</sub> th<sub>a</sub>t th<sub>e</sub> L<sub>agrang</sub>i<sub>an</sub> i<sub>s separa</sub>bl<sub>e</sub> i<sub>n</sub> $x \mathrm { : }$ 

$$
L (x, y) = L _ {1} (x _ {1}, y) + L _ {2} (x _ {2}, y) + \dots + L _ {p} (x _ {p}, y) - y ^ {T} b,
$$

<sub>w</sub>h<sub>ere</sub> $L _ { i } ( x _ { i } , y ) = f _ { i } ( x _ { i } ) + y ^ { T } A _ { i } x _ { i } , \ i = 1 , 2 , \ldots , p .$ 

H<sub>ence</sub> th<sub>e m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> i<sub>n x may</sub> b<sub>e sp</sub>lit i<sub>n</sub>t<sub>o</sub> $p$ se<sub>p</sub>arate t as<sup>k</sup>s : 

$$
x _ {i} ^ {k + 1} = \operatorname{argmin} _ {x _ {i}} L _ {i} (x _ {i}, y ^ {k}), i = 1, 2, \ldots , p
$$

<sub>w</sub>hi<sub>c</sub>h d<sub>o no</sub>t d<sub>epen</sub>d <sub>on eac</sub>h <sub>o</sub>th<sub>er an</sub>d <sub>may</sub> b<sub>e execu</sub>t<sub>e</sub>d i<sub>n para</sub>ll<sub>e</sub>l<sub>.</sub> 

## D<sub>ua</sub>l D<sub>ecompos</sub>iti<sub>on</sub> i<sub>n</sub> S<sub>epara</sub>bl<sub>e</sub> C<sub>ase</sub>

Dual Ascent Method (Separable Case) : 

<sub>repea</sub>t <sub>un</sub>til <sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> i<sub>s reac</sub>h<sub>e</sub>d <sub>:</sub> 

$$
x _ {i} ^ {k + 1} = \operatorname{argmin} _ {x _ {i}} L _ {i} (x _ {i}, y ^ {k}), \quad i = 1, 2, \ldots , p,
$$

$$
y ^ {k + 1} = y ^ {k} + \alpha^ {k} (\sum_ {i = 1} ^ {p} A _ {i} x _ {i} ^ {k + 1} - b)
$$

<sup>‘</sup> D<sub>ecompos</sub>iti<sub>on</sub> <sup>’</sup> b<sub>ecause we</sub> d<sub>ecompose a</sub> l<sub>arge pro</sub>bl<sub>em</sub> i<sub>n</sub>t<sub>o p</sub>i<sub>eces .</sub> T<sub>wo w</sub>id<sub>e</sub>l<sub>y use</sub>d d<sub>ecompos</sub>iti<sub>on sc</sub>h<sub>emes re</sub>l<sub>y on suc</sub>h <sub>a</sub> f<sub>ramewor</sub>k<sub>:</sub> • Dantzig-Wolfe Decomposition ( 1 960) 

• Benders Decomposition ( 1 96 1 ) 

−→ <sub>essen</sub>ti<sub>a</sub>l t<sub>oo</sub>l<sub>s</sub> f<sub>or so</sub>l<sub>v</sub>i<sub>ng com</sub>bi<sub>na</sub>t<sub>or</sub>i<sub>a</sub>l <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>ems .</sub> Th<sub>ey</sub> h<sub>ave a wea</sub>k<sub>ness</sub> th<sub>oug</sub>h <sub>:</sub> th<sub>ey may</sub> b<sub>e s</sub>l<sub>ow.</sub> 

## M<sub>e</sub>th<sub>o</sub>d <sub>o</sub>f M<sub>u</sub>lti<sub>p</sub>l<sub>ers</sub>

A<sub>n</sub> id<sub>en</sub>tifi<sub>a</sub>bl<sub>e wea</sub>k<sub>ness o</sub>f D<sub>ua</sub>l D<sub>ecompos</sub>iti<sub>on</sub> i<sub>s</sub> th<sub>e</sub> difi<sub>cu</sub>lt<sub>y</sub> t<sub>o</sub> <sub>sa</sub>t i<sub>s</sub>f<sub>y</sub> t h<sub>e cons</sub>t <sub>ra</sub>i<sub>n</sub>t $A x = b$ <sub>.</sub> Thi<sub>s may</sub> b<sub>e a</sub>dd<sub>resse</sub>d b<sub>y g</sub>i<sub>v</sub>i<sub>ng</sub> thi<sub>s</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>a more prom</sub>i<sub>nen</sub>t <sub>ro</sub>l<sub>e an</sub>d <sub>a</sub>ddi<sub>ng</sub> t<sub>o</sub> th<sub>e</sub> L<sub>agrang</sub>i<sub>an</sub> th<sub>e</sub> <sub>qua</sub>d<sub>ra</sub>ti<sub>c pena</sub>lt<sub>y o</sub>f th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t <sub>v</sub>i<sub>o</sub>l<sub>a</sub>ti<sub>on .</sub> 

D<sub>e</sub>fi<sub>ne</sub> th<sub>e</sub> A<sub>ugmen</sub>t<sub>e</sub>d L<sub>agrang</sub>i<sub>an:</sub> 

$$
L _ {\rho} (x, y) = f (x) + y ^ {T} (A x - b) + \frac {\rho}{2} \| (A x - b) \| ^ {2},
$$

<sub>w</sub>h<sub>ere</sub> $\rho$ i<sub>s a we</sub>i<sub>g</sub>ht <sub>o</sub>f th<sub>e pena</sub>lt<sub>y.</sub> 

This gives the Met ho d of Mult ipliers ( Hestenes 1 969 Powell 1 969 ) : 

<sub>repea</sub>t <sub>un</sub>til <sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> i<sub>s reac</sub>h<sub>e</sub>d <sub>:</sub> 

$$
x ^ {k + 1} = \operatorname{argmin} _ {x} L _ {\rho} (x, y ^ {k})
$$

$$
y ^ {k + 1} = y ^ {k} + \rho (A x ^ {k + 1} - b)
$$

Th<sub>e me</sub>th<sub>o</sub>d i<sub>s s</sub>i<sub>m</sub>il<sub>ar</sub> t<sub>o</sub> th<sub>e</sub> d<sub>ua</sub>l <sub>ascen</sub>t <sub>.</sub> 

It <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub> $L _ { \rho } ( x , y ^ { k } )$ i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f $L ( x , y ^ { k } )$ 

<sub>an</sub>d <sub>uses a</sub> fi<sub>xe</sub>d d<sub>ua</sub>l <sub>up</sub>d<sub>a</sub>t<sub>e s</sub>t<sub>eps</sub>i<sub>ze</sub> $\rho$ i<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f $\alpha .$ . 

If the obj ective function in the optimization problem 

min f (x ) s <sub>.</sub> t <sub>.</sub> Ax = b 

i<sub>s</sub> dif<sub>eren</sub>ti<sub>a</sub>bl<sub>e</sub> th<sub>en</sub> th<sub>e op</sub>ti<sub>ma</sub>lit<sub>y con</sub>diti<sub>ons are :</sub> 

$$
\begin{array}{r l} \nabla f (\hat {x}) + A ^ {T} \hat {y} = 0 & d u a l f e a s i b i l i t y \\ A \hat {x} = b & p r i m a l f e a s i b i l i t y \end{array}
$$

Ob<sub>serve</sub> th<sub>a</sub>t <sub>s</sub>i<sub>nce</sub> $x ^ { k + 1 }$ <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub> $L _ { \rho } ( x , y ^ { k } )$ th<sub>e</sub> d<sub>ua</sub>l <sub>up</sub>d<sub>a</sub>t<sub>e</sub> 

$$
y ^ {k + 1} = y ^ {k} + \rho (A x ^ {k + 1} - b)
$$

<sub>ensures</sub> th<sub>a</sub>t $( x ^ { k + 1 } , y ^ { k + 1 } )$ is dual feasi b le<sub>.</sub> Indeed : 

$$
\begin{array}{r l} 0 = \nabla_ {x} L _ {\rho} (x ^ {k + 1}, y ^ {k}) & = \nabla_ {x} f (x ^ {k + 1}) + A ^ {T} (y ^ {k} + \rho (A x ^ {k + 1} - b)) \\ & = \nabla_ {x} f (x ^ {k + 1}) + A ^ {T} y ^ {k + 1}. \end{array}
$$

However the primal feasi bility is attained onl<sub>y</sub> in the limit 

$$
A x ^ {k + 1} - b \rightarrow 0.
$$

## M<sub>e</sub>th<sub>o</sub>d <sub>o</sub>f M<sub>u</sub>lti<sub>p</sub>l<sub>ers vs</sub> D<sub>ua</sub>l D<sub>ecompos</sub>iti<sub>on</sub> M<sub>e</sub>th<sub>o</sub>d <sub>o</sub>f M<sub>u</sub>lti<sub>p</sub>l<sub>ers :</sub>

<sub>converges un</sub>d<sub>er more re</sub>l<sub>axe</sub>d <sub>assump</sub>ti<sub>ons</sub> ( f can be nondiferentiable) 

• d<sub>ea</sub>l<sub>s</sub> b <sub>e</sub>tt<sub>er w</sub>it h t h<sub>e pr</sub>i<sub>ma</sub>l f<sub>eas</sub>ibilit<sub>y</sub> $A x - b$ (presence of $\| A x - b \| ^ { 2 }$ in the Augmented Lagrangian helps) b<sub>u</sub>t 

• th<sub>e qua</sub>d<sub>ra</sub>ti<sub>c pena</sub>lt<sub>y</sub> $\| A x - b \| ^ { 2 }$ i<sub>n</sub> $L _ { \rho }$ d<sub>es</sub>t<sub>roys separa</sub>bilit<sub>y</sub> → <sub>canno</sub>t b<sub>e use</sub>d i<sub>n</sub> d<sub>ecompos</sub>iti<sub>on.</sub> 

## Alt<sub>erna</sub>ti<sub>ng</sub> Di<sub>rec</sub>ti<sub>on</sub> M<sub>e</sub>th<sub>o</sub>d <sub>o</sub>f M<sub>u</sub>lti<sub>p</sub>li<sub>ers</sub> (ADMM)

ADM M <sub>o</sub>f<sub>ers a comprom</sub>i<sub>se :</sub> 

• enj oys some of the benefits of the m ethod of multip li ers 

• i<sub>s we</sub>ll<sub>-su</sub>it<sub>e</sub>d t<sub>o</sub> d<sub>ecompos</sub>iti<sub>on</sub> 

Gabay and Mercier ( 1 976) Glowinski and M arrocco ( 1 975) <sub>.</sub> 

## Alt<sub>erna</sub>ti<sub>ng</sub> Di<sub>rec</sub>ti<sub>on</sub> M<sub>e</sub>th<sub>o</sub>d <sub>o</sub>f M<sub>u</sub>lti<sub>p</sub>li<sub>ers</sub> (ADMM)

C<sub>ons</sub>id<sub>er a pro</sub>bl<sub>em</sub> i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> f<sub>orm :</sub> 

$$
\begin{array}{r l} & {\min f _ {1} (x _ {1}) + f _ {2} (x _ {2})} \\ & {\mathrm{s.t.} A _ {1} x _ {1} + A _ {2} x _ {2} = b,} \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $f _ { 1 } : \mathcal { R } ^ { n _ { 1 } } \mapsto \mathcal { R }$ <sub>an</sub>d $f _ { 2 } : \mathcal { R } ^ { n _ { 2 } } \mapsto \mathcal { R }$ are convex functions (do not have to be diferentiable) $A _ { i } \in \mathcal { R } ^ { m \times n _ { i } } , ~ i = 1 , 2 , ~ b \in \mathcal { R } ^ { m }$ Observe that the obj ective is s epara b le but the constraint links $x _ { 1 }$ <sub>an</sub>d $x _ { 2 }$ 

W<sub>r</sub>it<sub>e</sub> d<sub>own</sub> th<sub>e assoc</sub>i<sub>a</sub>t<sub>e</sub>d A<sub>ugmen</sub>t<sub>e</sub>d L<sub>agrang</sub>i<sub>an:</sub> 

$$
\begin{array}{r} L _ {\rho} (x _ {1}, x _ {2}, y) = f _ {1} (x _ {1}) + f _ {2} (x _ {2}) + y ^ {T} (A _ {1} x _ {1} + A _ {2} x _ {2} - b) \\ + \frac {\rho}{2} \| (A _ {1} x _ {1} + A _ {2} x _ {2} - b) \| ^ {2}. \end{array}\tag{1}
$$

## Alt<sub>erna</sub>ti<sub>ng</sub> Di<sub>rec</sub>ti<sub>on</sub> M<sub>e</sub>th<sub>o</sub>d <sub>o</sub>f M<sub>u</sub>lti<sub>p</sub>li<sub>ers</sub> (ADMM)

<sub>repea</sub>t <sub>un</sub>til <sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> i<sub>s reac</sub>h<sub>e</sub>d <sub>:</sub> 

$$
\begin{array}{r l} & x _ {1} ^ {k + 1} = \operatorname * {a r g m i n} _ {x _ {1}} L _ {\rho} (x _ {1}, x _ {2} ^ {k}, y ^ {k}) \\ & x _ {2} ^ {k + 1} = \operatorname * {a r g m i n} _ {x _ {2}} L _ {\rho} (x _ {1} ^ {k + 1}, x _ {2}, y ^ {k}) \\ & y ^ {k + 1} = y ^ {k} + \rho (A _ {1} x _ {1} ^ {k + 1} + A _ {2} x _ {2} ^ {k + 1} - b) \end{array}
$$

<sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>ze</sub> i<sub>n</sub> $x _ { 1 }$ 

<sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>ze</sub> i<sub>n x2</sub> 

<sub>up</sub> d<sub>a</sub>t<sub>e mu</sub>lti<sub>p</sub>li<sub>ers y</sub> 

Ob<sub>serve</sub> th<sub>a</sub>t th<sub>e op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> i<sub>n</sub> $x _ { 1 }$ <sub>uses</sub> th<sub>e</sub> <sup>“</sup><sub>o</sub>ld<sup>”</sup> $x _ { 2 } ^ { k }$ b<sub>u</sub>t th<sub>e op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> i<sub>n</sub> $x _ { 2 }$ uses already the <sup>“</sup>new<sup>”</sup> (up dated) $x _ { 1 } ^ { k + 1 }$ 

C<sub>onvergence o</sub>f t h<sub>e</sub> AD MM If th<sub>e</sub> f<sub>unc</sub>ti<sub>ons</sub> $f _ { \mathrm { 1 } }$ <sub>an</sub>d $f _ { 2 }$ in the obj ective are diferentiable then the optimality conditions 

are : 

$$
\begin{array}{r l} \nabla f _ {1} (\hat {x} _ {1}) + A _ {1} ^ {T} \hat {y} = 0 & \text {1st dual feasibility} \\ \nabla f _ {2} (\hat {x} _ {2}) + A _ {2} ^ {T} \hat {y} = 0 & \text {2nd dual feasibility} \\ A _ {1} \hat {x} _ {1} + A _ {2} \hat {x} _ {2} = b & \text {primal feasibility} \end{array}
$$

N<sub>o</sub>t<sub>e</sub> th<sub>a</sub>t <sub>s</sub>i<sub>nce</sub> $x _ { 2 } ^ { k + 1 }$ <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>zes</sub> $L _ { \rho } ( x _ { 1 } ^ { k + 1 } , x _ { 2 } , y ^ { k } )$ <sub>,</sub> th<sub>e</sub> d<sub>ua</sub>l <sub>up</sub>d<sub>a</sub>t<sub>e</sub> 

$$
y ^ {k + 1} = y ^ {k} + \rho (A x ^ {k + 1} - b)
$$

<sub>g</sub>uarantees t<sup>h</sup>at $( x _ { 1 } ^ { k + 1 } , x _ { 2 } ^ { k + 1 } , y ^ { k + 1 } )$ <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> th<sub>e</sub> 2<sub>n</sub>d d<sub>ua</sub>l f<sub>eas</sub>ibilit<sub>y</sub> <sub>cons</sub>t<sub>ra</sub>i<sub>n</sub>t I<sub>n</sub>d<sub>ee</sub>d <sub>:</sub> 

$$
\begin{array}{r l} & 0 = \nabla_ {x _ {2}} f _ {2} (x _ {2} ^ {k + 1}) + A _ {2} ^ {T} y ^ {k} + \rho A _ {2} ^ {T} (A _ {1} x _ {1} ^ {k + 1} + A _ {2} x _ {2} ^ {k + 1} - b) \\ & \quad = \nabla_ {x _ {2}} f _ {2} (x _ {2} ^ {k + 1}) + A _ {2} ^ {T} (y ^ {k} + \rho (A x ^ {k + 1} - b)) \\ & \quad = \nabla_ {x _ {2}} f _ {2} (x _ {2} ^ {k + 1}) + A _ {2} ^ {T} y ^ {k + 1}. \end{array}
$$

However the 1 st dual feasi bility and primal feasi bility are attained only in the limit (at convergence) <sub>.</sub> 

## C<sub>onvergence o</sub>f th<sub>e</sub> ADMM

C<sub>ons</sub>id<sub>er a pro</sub>bl<sub>em</sub> i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> f<sub>orm :</sub> 

$$
\min F (x _ {1}, x _ {2}) = f _ {1} (x _ {1}) + f _ {2} (x _ {2})
$$

$$
\mathrm{s.t.} A _ {1} x _ {1} + A _ {2} x _ {2} = b,
$$

<sub>w</sub>h<sub>ere</sub> $f _ { 1 } : \mathcal { R } ^ { n _ { 1 } } \mapsto \mathcal { R }$ <sub>, an</sub>d $f _ { 2 } : \mathcal { R } ^ { n _ { 2 } } \mapsto \mathcal { R }$ are convex functions (do not have to be diferentiable) $A _ { i } \in \mathcal { R } ^ { m \times n _ { i } } , ~ i = 1 , 2 , ~ b \in \mathcal { R } ^ { m }$ 

## C<sub>onvergence o</sub>f th<sub>e</sub> ADMM

Th<sub>eorem.</sub> S<sub>uppose</sub> $f _ { 1 }$ <sub>an</sub>d $f _ { 2 }$ <sub>are c</sub>l<sub>ose</sub>d <sub>convex</sub> f<sub>unc</sub>ti<sub>ons an</sub>d $\gamma$ i<sub>s any cons</sub>t<sub>an</sub>t <sub>w</sub>hi<sub>c</sub>h <sub>sa</sub>ti<sub>s</sub>fi<sub>es</sub> $\bar { \gamma } > 2 \| \hat { y } \| _ { 2 }$ <sub>.</sub> Th<sub>en</sub> 

$$
F (x _ {1} ^ {t}, x _ {2} ^ {t}) - F (\hat {x} _ {1}, \hat {x} _ {2}) \leq \frac {\| x _ {2} ^ {0} - \hat {x} _ {2} \| _ {\rho A _ {2} ^ {T} A _ {2}} ^ {2} + (\gamma + \| y ^ {0} \| _ {2}) ^ {2} / \rho}{2 (t + 1)}
$$

$$
\| A _ {1} x _ {1} ^ {t} + A _ {2} x _ {2} ^ {t} - b \| _ {2} \leq \frac {\| x _ {2} ^ {0} - \hat {x} _ {2} \| _ {\rho A _ {2} ^ {T} A _ {2}} ^ {2} + (\gamma + \| y ^ {0} \| _ {2}) ^ {2} / \rho}{\gamma (t + 1)},
$$

<sub>w</sub>h<sub>ere</sub> $\begin{array} { r } { x _ { 1 } ^ { t } : = \frac { 1 } { t + 1 } \sum _ { k = 1 } ^ { t + 1 } x _ { 1 } ^ { k } , x _ { 2 } ^ { t } : = \frac { 1 } { t + 1 } \sum _ { k = 1 } ^ { t + 1 } x _ { 2 } ^ { k } , } \end{array}$ <sub>an</sub>d f<sub>or</sub> $C \succeq 0$ <sub>we</sub> d<sub>e</sub>fi<sub>ne</sub> $\| u \| _ { C } ^ { 2 } : = u ^ { T } C u$ 

Theoretical convergence of ADMM is slow: $\mathcal{O}(1 / t)$ convergence rate and $\mathcal{O}(1 / \epsilon)$ iteration complexity. (To compare: IPMs enjoy $\mathcal{O}(\log (1 / \epsilon))$ iteration complexity.) 

## ADMM <sub>:</sub> F<sub>rom</sub> 2 bl<sub>oc</sub>k<sub>s</sub> t<sub>o</sub> $p$ bl<sub>oc</sub>k<sub>s</sub>

C<sub>ons</sub>id<sub>er a pro</sub>bl<sub>em</sub> i<sub>n</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng</sub> f<sub>orm :</sub> 

$$
\begin{array}{r l} \min & \sum_ {i = 1} ^ {p} f _ {i} (x _ {i}) \\ \mathrm{s.t.} & \sum_ {i = 1} ^ {p} A _ {i} x _ {i} = b, \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $f _ { i } : \mathcal { R } ^ { n _ { i } } \mapsto \mathcal { R } , i = 1 , 2 , \ldots , p$ are convex functions (do not have to be diferentiable) $A _ { i } \in \mathcal { R } ^ { m \times n _ { i } } , \ i = 1 , 2 , \dots , p , \ b \in \mathcal { R } ^ { m }$ Observe that the obj ective is s epara b le but the constraint links all th<sub>e var</sub>i<sub>a</sub>bl<sub>es</sub> $x _ { i }$ 

W<sub>r</sub>it<sub>e</sub> d<sub>own</sub> th<sub>e assoc</sub>i<sub>a</sub>t<sub>e</sub>d A<sub>ugmen</sub>t<sub>e</sub>d L<sub>agrang</sub>i<sub>an:</sub> 

$$
\begin{array}{r} L _ {\rho} (x _ {1}, x _ {2}, \ldots , x _ {p}, y) = \sum_ {i = 1} ^ {p} f _ {i} (x _ {i}) + y ^ {T} (\sum_ {i = 1} ^ {p} A _ {i} x _ {i} - b) \\ + \frac {\rho}{2} | | (\sum_ {i = 1} ^ {p} A _ {i} x _ {i} - b) | | ^ {2}. \end{array}
$$

## ADMM <sub>:</sub> F<sub>rom</sub> 2 bl<sub>oc</sub>k<sub>s</sub> t<sub>o</sub> $p$ bl<sub>oc</sub>k<sub>s</sub>

M<sub>u</sub>lti<sub>p</sub>l<sub>e</sub> bl<sub>oc</sub>k <sub>vers</sub>i<sub>on o</sub>f AD MM <sub>:</sub> <sub>repea</sub>t <sub>un</sub>til <sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> i<sub>s reac</sub>h<sub>e</sub>d <sub>:</sub> 

$$
\begin{array}{l l} x _ {1} ^ {k + 1} = \operatorname * {a r g m i n} _ {x _ {1}} L _ {\rho} (x _ {1}, x _ {2} ^ {k}, \ldots , x _ {p} ^ {k}, y ^ {k}) & \mathrm{minimizein} x _ {1} \\ x _ {2} ^ {k + 1} = \operatorname * {a r g m i n} _ {x _ {2}} L _ {\rho} (x _ {1} ^ {k + 1}, x _ {2}, \ldots , x _ {p} ^ {k}, y ^ {k}) & \mathrm{minimizein} x _ {2} \\ \vdots & \vdots \\ x _ {p} ^ {k + 1} = \operatorname * {a r g m i n} _ {x _ {p}} L _ {\rho} (x _ {1} ^ {k + 1}, x _ {2} ^ {k + 1}, \ldots , x _ {p}, y ^ {k}) & \mathrm{minimizein} x _ {p} \\ y ^ {k + 1} = y ^ {k} + \rho (\sum_ {i = 1} ^ {p} A _ {i} x _ {i} ^ {k + 1} - b) & \mathrm{update} y \end{array}
$$

## C<sub>ommen</sub>t<sub>s on</sub> C<sub>onvergence</sub>

While (under suitable assumptions) the 2-block AD M M is proved t<sub>o converge</sub> th<sub>e p-</sub>bl<sub>oc</sub>k <sub>vers</sub>i<sub>on</sub> d<sub>oes no</sub>t h<sub>ave</sub> t<sub>o converge see :</sub> C <sub>.</sub> Ch<sub>en</sub> B <sub>.</sub> H<sub>e</sub> Y<sub>.</sub> Y<sub>e</sub> X <sub>.</sub> Y<sub>uan</sub> <sup>“</sup>Th<sub>e</sub> di<sub>rec</sub>t <sub>ex</sub>t<sub>ens</sub>i<sub>on o</sub>f AD M M f<sub>or mu</sub>lti<sub>-</sub>bl<sub>oc</sub>k <sub>convex m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on pro</sub>bl<sub>ems</sub> i<sub>s no</sub>t <sub>neces-</sub> sarily convergent<sup>”</sup> <sub>,</sub> M athematical Prog A 1 55 (20 1 6) pp <sub>.</sub> 57–79 <sub>.</sub> Example with nul l obj ective : 

$$
\begin{array}{r l r} \min & 0 \\ \mathrm{s.t.} & A _ {1} x _ {1} + A _ {2} x _ {2} + A _ {3} x _ {3} = 0, \end{array}
$$

<sub>w</sub>h<sub>ere</sub> 

$$
A _ {1} = \left[ \begin{array}{l} 1 \\ 1 \\ 1 \end{array} \right], A _ {2} = \left[ \begin{array}{l} 1 \\ 1 \\ 2 \end{array} \right], A _ {3} = \left[ \begin{array}{l} 1 \\ 2 \\ 2 \end{array} \right].
$$

Ob<sub>serve</sub> th<sub>a</sub>t $A = [ A _ { 1 } , A _ { 2 } , A _ { 3 } ]$ i<sub>s nons</sub>i<sub>ngu</sub>l<sub>ar .</sub> Si<sub>nce</sub> th<sub>e r</sub>i<sub>g</sub>ht<sub>-</sub>h<sub>an</sub>d<sub>-</sub> <sub>s</sub>id<sub>e</sub> $b = 0$ th<sub>e</sub> f<sub>eas</sub>ibl<sub>e se</sub>t <sub>con</sub>t<sub>a</sub>i<sub>ns a s</sub>i<sub>ng</sub>l<sub>e e</sub>l<sub>emen</sub>t ${ \hat { x } } _ { 1 } = { \hat { x } } _ { 2 } = { \hat { x } } _ { 3 } =$ 0 <sub>.</sub> Since the obj ective is null the optimal Lagrange multiplier $\hat { y } = 0$ Th<sub>e</sub> 3<sub>-</sub>bl<sub>oc</sub>k AD M M i<sub>s</sub> di<sub>vergen</sub>t f<sub>or</sub> thi<sub>s pro</sub>bl<sub>em .</sub> 

## A<sub>pp</sub>li<sub>ca</sub>ti<sub>ons</sub>

AD M M i<sub>s par</sub>ti<sub>cu</sub>l<sub>ar</sub>l<sub>y a</sub>tt<sub>rac</sub>ti<sub>ve w</sub>h<sub>en</sub> th<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za-</sub> t i<sub>ons</sub> i<sub>n</sub> $x _ { i }$ <sub>are s</sub>i<sub>gn</sub>ifi<sub>can</sub>tl<sub>y eas</sub>i<sub>er</sub> th<sub>an</sub> th<sub>e m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on o</sub>f th<sub>e ag-</sub> gregate obj ective $\textstyle \sum _ { i = 1 } ^ { p } f _ { i } ( x _ { i } )$ 

S<sub>ome</sub>ti<sub>mes a non-separa</sub>bl<sub>e pro</sub>bl<sub>em</sub> i<sub>s conver</sub>t<sub>e</sub>d t<sub>o a separa</sub>bl<sub>e one</sub> j ust to be able to apply AD M M because of its attractive features <sub>,</sub> <sub>name</sub>l<sub>y,</sub> it<sub>s a</sub>bilit<sub>y</sub> t<sub>o ma</sub>k<sub>e</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>ons</sub> i<sub>n</sub> $x _ { i }$ E<sub>xamp</sub>l<sub>e :</sub> C<sub>ons</sub>id<sub>er a non-separa</sub>bl<sub>e pro</sub>bl<sub>em</sub> 

$$
\begin{array}{r l} \min & f _ {1} (x) + f _ {2} (x) \\ \mathrm{s.t.} & A x = b, \end{array}
$$

i<sub>n w</sub>hi<sub>c</sub>h b<sub>o</sub>th f<sub>unc</sub>ti<sub>ons</sub> $f _ { 1 }$ <sub>an</sub>d $f _ { 2 }$ d<sub>epen</sub>d <sub>on</sub> th<sub>e same var</sub>i<sub>a</sub>bl<sub>e x .</sub> W<sub>e crea</sub>t<sub>e a copy o</sub>f <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> $x$ <sub>an</sub>d <sub>rewr</sub>it<sub>e</sub> th<sub>e a</sub>b<sub>ove pro</sub>bl<sub>em as :</sub> 

$$
\min f _ {1} (x) + f _ {2} (z)
$$

$$
\mathrm{s.t.} \qquad A x = b
$$

$$
x - z = 0
$$

i<sub>n a</sub> f<sub>orm su</sub>it<sub>a</sub>bl<sub>e</sub> f<sub>or</sub> ADM M 

E<sub>xamp</sub>l<sub>e :</sub> ADMM f<sub>or</sub> ℓ<sub>1-regu</sub>l<sub>ar</sub>i<sub>ze</sub>d L<sub>eas</sub>t S<sub>quares</sub> R<sub>eca</sub>ll ℓ <sub>1-regu</sub>l<sub>ar</sub>i<sub>ze</sub>d l<sub>eas</sub>t <sub>squares</sub> 

$$
\min \tau \| x \| _ {1} + \frac {1}{2} \| A x - b \| _ {2} ^ {2},
$$

<sub>w</sub>h<sub>ere</sub> $A \in \mathcal { R } ^ { m \times n } , b \in \mathcal { R } ^ { m }$ <sub>.</sub> U<sub>sua</sub>ll<sub>y</sub> $m \geq n$ (and often $m \gg n )$ Thi<sub>s pro</sub>bl<sub>em may</sub> b<sub>e cas</sub>t i<sub>n a</sub> f<sub>orm su</sub>it<sub>a</sub>bl<sub>e</sub> f<sub>or</sub> AD M M <sub>:</sub> 

$$
\begin{array}{r l} & {\min \tau \| z \| _ {1} + \frac {1}{2} \| A x - b \| _ {2} ^ {2}} \\ & {\mathrm{s.t.} x - z = 0.} \end{array}
$$

W<sub>r</sub>it<sub>e</sub> d<sub>own</sub> th<sub>e assoc</sub>i<sub>a</sub>t<sub>e</sub>d A<sub>ugmen</sub>t<sub>e</sub>d L<sub>agrang</sub>i<sub>an:</sub> 

$$
L _ {\rho} (x, z, y) = \tau \| z \| _ {1} + \frac {1}{2} \| A x - b \| _ {2} ^ {2} + y ^ {T} (x - z) + \frac {\rho}{2} \| x - z \| _ {2} ^ {2}.
$$

E<sub>xamp</sub>l<sub>e :</sub> ADMM f<sub>or</sub> ℓ<sub>1-regu</sub>l<sub>ar</sub>i<sub>ze</sub>d L<sub>eas</sub>t S<sub>quares</sub> With <sub>suc</sub>h A<sub>ugmen</sub>t<sub>e</sub>d L<sub>agrang</sub>i<sub>an :</sub> 

$$
L _ {\rho} (x, z, y) = \tau \| z \| _ {1} + \frac {1}{2} \| A x - b \| _ {2} ^ {2} + y ^ {T} (x - z) + \frac {\rho}{2} \| x - z \| _ {2} ^ {2}.
$$

M i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> i<sub>n x exp</sub>l<sub>o</sub>it<sub>s</sub> th<sub>e</sub> dif<sub>eren</sub>ti<sub>a</sub>bilit<sub>y o</sub>f $L _ { \rho }$ i<sub>n</sub> $x \mathrm { : }$ 

$$
\nabla_ {x} L _ {\rho} (x, z, y) = A ^ {T} (A x - b) + \rho (x - z) + y = 0,
$$

<sub>w</sub>hi<sub>c</sub>h <sub>g</sub>i<sub>ves</sub> 

$$
x = (A ^ {T} A + \rho I) ^ {- 1} (A ^ {T} b + \rho z - y).
$$

Mi<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> i<sub>n</sub> $z$ re<sub>q</sub>u<sup>i</sup>res : 

$$
\min _ {z} \left(\tau \| z \| _ {1} + \frac {\rho}{2} \| z - x - y / \rho \| _ {2} ^ {2}\right),
$$

<sub>an</sub>d i<sub>s per</sub>f<sub>ec</sub>tl<sub>y separa</sub>bl<sub>e</sub> i<sub>n</sub>t<sub>o n</sub> i<sub>n</sub>d<sub>epen</sub>d<sub>en</sub>t <sub>coor</sub>di<sub>na</sub>t<sub>es :</sub> 

$$
\min _ {z _ {i}} \left(\tau | z _ {i} | + \frac {\rho}{2} (z _ {i} - x _ {i} - y _ {i} / \rho) ^ {2}\right), \quad i = 1, 2, \ldots n.
$$

## S<sub>o</sub>ft Th<sub>res</sub>h<sub>o</sub>ldi<sub>ng</sub>

I<sub>n</sub> $\ell _ { 1 }$ -regularized least squares (and in many other applications) th<sub>ere</sub> i<sub>s a nee</sub>d t<sub>o per</sub>f<sub>orm a one-</sub>di<sub>mens</sub>i<sub>ona</sub>l <sub>up</sub> d<sub>a</sub>t<sub>e o</sub>f $z _ { i }$ : 

$$
z _ {i} ^ {+} := \operatorname{argmin} _ {z _ {i}} \left(\tau | z _ {i} | + \frac {\rho}{2} (z _ {i} - u) ^ {2}\right),
$$

Alth<sub>oug</sub>h th<sub>e</sub> fi<sub>rs</sub>t t<sub>erm</sub> i<sub>s no</sub>t dif<sub>eren</sub>ti<sub>a</sub>bl<sub>e</sub> b<sub>ecause</sub> it i<sub>nvo</sub>l<sub>ves</sub> th<sub>e</sub> <sub>a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e va</sub>l<sub>ue we can eas</sub>il<sub>y compu</sub>t<sub>e a c</sub>l<sub>ose</sub>d<sub>-</sub>f<sub>orm so</sub>l<sub>u</sub>ti<sub>on :</sub> 

$$
z _ {i} ^ {+} := S _ {\tau / \rho} (u),
$$

where the s oft thresholding operator S is defined as : 

$$
S _ {\tau / \rho} (u) = \left\{ \begin{array}{l l} u - \tau / \rho & \mathrm{if} \quad u > \tau / \rho \\ 0 & \mathrm{if} \quad | u | \leq \tau / \rho \\ u + \tau / \rho & \mathrm{if} \quad u <   - \tau / \rho , \end{array} \right.
$$

<sub>or equ</sub>i<sub>va</sub>l<sub>en</sub>tl<sub>y :</sub> 

$$
S _ {\tau / \rho} (u) = (u - \tau / \rho) _ {+} - (- u - \tau / \rho) _ {+},
$$

<sub>w</sub>h<sub>ere</sub> $v _ { + } : = m a x ( v , 0 )$ 

E<sub>xamp</sub>l<sub>e :</sub> ADMM f<sub>or</sub> ℓ<sub>1-regu</sub>l<sub>ar</sub>i<sub>ze</sub>d L<sub>eas</sub>t S<sub>quares</sub> $\ell _ { 1 }$ <sub>-regu</sub>l<sub>ar</sub>i<sub>ze</sub>d l<sub>eas</sub>t <sub>squares pro</sub>bl<sub>em cas</sub>t i<sub>n a</sub> f<sub>orm su</sub>it<sub>a</sub>bl<sub>e</sub> f<sub>or</sub> ADMM<sub>:</sub> 1 2 

$$
\begin{array}{r l} & {\min \tau \| z \| _ {1} + \frac {1}{2} \| A x - b \| _ {2} ^ {2}} \\ & {\mathrm{s.t.} x - z = 0.} \end{array}
$$

<sub>w</sub>h<sub>ere</sub> $A \in \mathcal { R } ^ { m \times n } , b \in \mathcal { R } ^ { m }$ <sub>.</sub> U<sub>sua</sub>ll<sub>y</sub> $m \geq n$ (and often $m \gg n )$ ADMM<sub>:</sub> 

<sub>repea</sub>t <sub>un</sub>til <sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> i<sub>s reac</sub>h<sub>e</sub>d <sub>:</sub> 

$$
x ^ {k + 1} = (A ^ {T} A + \rho I) ^ {- 1} (A ^ {T} b + \rho z ^ {k} - y ^ {k})
$$

$$
z ^ {k + 1} = S _ {\tau / \rho} (x ^ {k + 1} + y ^ {k} / \rho)
$$

$$
y ^ {k + 1} = y ^ {k} + \rho (x ^ {k + 1} - z ^ {k + 1}),
$$

<sub>w</sub>h<sub>ere</sub> $S _ { \tau / \rho } ( . )$ is the s oft thresho lding operator : 

$$
S _ {\tau / \rho} (u) = (u - \tau / \rho) _ {+} - (- u - \tau / \rho) _ {+}.
$$

Th<sub>e op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> i<sub>n</sub> $z$ is split component-wise and enj oys a trivial <sub>so</sub>l<sub>u</sub>t i<sub>on .</sub> 

E<sub>xamp</sub>l<sub>e:</sub> ADMM f<sub>or</sub> C<sub>onsensus op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> C<sub>ons</sub>id<sub>er a non-separa</sub>bl<sub>e pro</sub>bl<sub>em</sub> 

$$
\min \sum_ {i = 1} ^ {p} f _ {i} (x)
$$

i<sub>n w</sub>hi<sub>c</sub>h <sub>a</sub>ll f<sub>unc</sub>ti<sub>ons</sub> $f _ { i } , i = 1 , 2 , \ldots p$ d<sub>epen</sub>d <sub>on</sub> th<sub>e same var</sub>i<sub>a</sub>bl<sub>e</sub> $x .$ I<sub>n mac</sub>hi<sub>ne</sub> l<sub>earn</sub>i<sub>ng</sub> $f _ { i }$ <sub>m</sub>i<sub>g</sub>ht b<sub>e</sub> th<sub>e</sub> l<sub>oss</sub> f<sub>unc</sub>ti<sub>on</sub> f<sub>or</sub> th<sub>e</sub> i<sub>-</sub>th bl<sub>oc</sub>k <sub>o</sub>f t<sub>ra</sub>i<sub>n</sub>i<sub>ng</sub> d<sub>a</sub>t<sub>a.</sub> 

W<sub>e crea</sub>t<sub>e p cop</sub>i<sub>es o</sub>f <sub>var</sub>i<sub>a</sub>bl<sub>e</sub> $x ,$ <sub>ca</sub>ll th<sub>em</sub> $x _ { i } ,$ <sub>a</sub>dd <sub>new cons</sub>t<sub>ra</sub>i<sub>n</sub>t<sub>s</sub> $x _ { i } = z , \forall i$ <sub>an</sub>d th<sub>en rewr</sub>it<sub>e</sub> th<sub>e a</sub>b<sub>ove pro</sub>bl<sub>em as :</sub> 

$$
\begin{array}{r l} & {\min \sum_ {i = 1} ^ {p} f _ {i} (x _ {i})} \\ & {\mathrm{s.t.} x _ {i} - z = 0, i = 1, 2, \ldots , p} \end{array}
$$

i<sub>n a</sub> f<sub>orm su</sub>it<sub>a</sub>bl<sub>e</sub> f<sub>or</sub> AD M M <sub>.</sub> I<sub>n</sub> thi<sub>s pro</sub>bl<sub>em :</sub> $x _ { i }$ <sub>are</sub> th<sub>e</sub> l<sub>oca</sub>l <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> $z$ i<sub>s a g</sub>l<sub>o</sub> b<sub>a</sub>l <sub>var</sub>i<sub>a</sub>bl<sub>e an</sub>d th<sub>e cons</sub>t<sub>ra</sub>i<sub>n</sub>t $x _ { i } - z = 0$ forces all (independent ) sub-problems to agree on a <sub>common va</sub>l<sub>ue</sub> $z ,$ i <sub>. e .</sub> t<sub>o reac</sub>h <sub>a consensus .</sub> 

## E<sub>xamp</sub>l<sub>e:</sub> ADMM f<sub>or</sub> C<sub>onsensus op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> $\mathrm { ( c o n t ^ { \circ } d ) }$

W<sub>r</sub>it<sub>e</sub> d<sub>own</sub> th<sub>e assoc</sub>i<sub>a</sub>t<sub>e</sub>d A<sub>ugmen</sub>t<sub>e</sub>d L<sub>agrang</sub>i<sub>an:</sub> 

$$
L _ {\rho} (x _ {1}, x _ {2}, \dots , x _ {p}, z, y _ {1}, y _ {2}, \dots , y _ {p}) = \sum_ {i = 1} ^ {p} \Bigl (f _ {i} (x _ {i}) + y _ {i} ^ {T} (x _ {i} - z) + \frac {\rho}{2} \| x _ {i} - z \| ^ {2} \Bigr).
$$

## ADMM<sub>:</sub>

<sub>repea</sub>t <sub>un</sub>til <sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> i<sub>s reac</sub>h<sub>e</sub>d <sub>:</sub> 

$$
\begin{array}{r l} & x _ {i} ^ {k + 1} = \operatorname * {a r g m i n} _ {x _ {i}} \left(f _ {i} (x _ {i}) + (y _ {i} ^ {k}) ^ {T} (x _ {i} - z ^ {k}) + \frac {\rho}{2} \| x _ {i} - z ^ {k} \| ^ {2}\right), i = 1.. p \\ & z ^ {k + 1} = \frac {1}{p} \sum_ {i = 1} ^ {p} (x _ {i} ^ {k + 1} + y _ {i} ^ {k} / \rho) \\ & y _ {i} ^ {k + 1} = y _ {i} ^ {k} + \rho (x _ {i} ^ {k + 1} - z ^ {k + 1}), i = 1.. p \end{array}
$$

Ob<sub>serve</sub> th<sub>a</sub>t <sub>averag</sub>i<sub>ng</sub> i<sub>s per</sub>f<sub>orme</sub>d i<sub>n</sub> th<sub>e up</sub>d<sub>a</sub>t<sub>e o</sub>f <sub>z .</sub> 

## Example: ADMM for QP

C<sub>ons</sub>id<sub>er a convex qua</sub>d<sub>ra</sub>ti<sub>c programm</sub>i<sub>ng pro</sub>bl<sub>em</sub> 

$$
\begin{array}{r l} & {\min \frac {1}{2} x ^ {T} H x + c ^ {T} x} \\ & {\mathrm{s.t.} A x = b,} \end{array}
$$

$$
x = \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] \in \mathcal {R} ^ {n}, x _ {i} \in \mathcal {R} ^ {n _ {i}}, n _ {1} + n _ {2} = n, H = \left[ \begin{array}{c c} H _ {1 1} & H _ {2 1} ^ {T} \\ H _ {2 1} & H _ {2 2} \end{array} \right] \in \mathcal {R} ^ {n \times n}
$$

i<sub>s a symme</sub>t<sub>r</sub>i<sub>c pos .</sub> d<sub>e</sub>fi<sub>n</sub>it<sub>e ma</sub>t<sub>r</sub>i<sub>x</sub> $A = [ A _ { 1 } , A _ { 2 } ] \in \mathcal { R } ^ { m \times n } , \ b \in \mathcal { R } ^ { m }$ W<sub>r</sub>it<sub>e</sub> d<sub>own</sub> th<sub>e assoc</sub>i<sub>a</sub>t<sub>e</sub>d A<sub>ugmen</sub>t<sub>e</sub>d L<sub>agrang</sub>i<sub>an:</sub> 

$$
\begin{array}{r l} & L _ {\rho} (x _ {1}, x _ {2}, y) = \frac {1}{2} x ^ {T} H x + c ^ {T} x + y ^ {T} (A x - b) + \frac {\rho}{2} \| (A x - b) \| ^ {2} \\ & \qquad = \frac {1}{2} \left[ x _ {1} ^ {T}, x _ {2} ^ {T} \right] \left[ \begin{array}{l l} H _ {1 1} + \rho A _ {1} ^ {T} A _ {1} & H _ {2 1} ^ {T} + \rho A _ {1} ^ {T} A _ {2} \\ H _ {2 1} + \rho A _ {2} ^ {T} A _ {1} & H _ {2 2} + \rho A _ {2} ^ {T} A _ {2} \end{array} \right] \left[ \begin{array}{l} x _ {1} \\ x _ {2} \end{array} \right] + \\ & \qquad + (c _ {1} + A _ {1} ^ {T} y + \rho A _ {1} ^ {T} b) ^ {T} x _ {1} + (c _ {2} + A _ {2} ^ {T} y + \rho A _ {2} ^ {T} b) ^ {T} x _ {2} + \\ & \qquad + \frac {\rho}{2} b ^ {T} b - b ^ {T} y. \end{array}
$$

## Example: ADMM for QP (continued)

R<sub>eca</sub>ll th<sub>e genera</sub>l AD MM <sub>:</sub> 

<sub>repea</sub>t <sub>un</sub>til <sub>op</sub>ti<sub>ma</sub>lit<sub>y</sub> i<sub>s reac</sub>h<sub>e</sub>d <sub>:</sub> 

$$
x _ {1} ^ {k + 1} = \operatorname{argmin} _ {x _ {1}} L _ {\rho} (x _ {1}, x _ {2} ^ {k}, y ^ {k})
$$

<sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>ze</sub> i<sub>n</sub> $x _ { 1 }$ 

$$
x _ {2} ^ {k + 1} = \operatorname{argmin} _ {x _ {2}} L _ {\rho} (x _ {1} ^ {k + 1}, x _ {2}, y ^ {k})
$$

<sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>ze</sub> i<sub>n</sub> $x _ { 2 }$ 

$$
y ^ {k + 1} = y ^ {k} + \rho (A _ {1} x _ {1} ^ {k + 1} + A _ {2} x _ {2} ^ {k + 1} - b)
$$

<sub>up</sub> d<sub>a</sub>t<sub>e mu</sub>lti<sub>p</sub>li<sub>ers y</sub> 

For convex QP the first two tasks have closed form solutions 

$$
\nabla_ {x _ {1}} L _ {\rho} = (H _ {1 1} + \rho A _ {1} ^ {T} A _ {1}) x _ {1} + (H _ {2 1} ^ {T} + \rho A _ {1} ^ {T} A _ {2}) x _ {2} + (c _ {1} + A _ {1} ^ {T} y + \rho A _ {1} ^ {T} b) = 0
$$

$$
\nabla_ {x _ {2}} L _ {\rho} = (H _ {2 1} + \rho A _ {2} ^ {T} A _ {1}) x _ {1} + (H _ {2 2} + \rho A _ {2} ^ {T} A _ {2}) x _ {2} + (c _ {2} + A _ {2} ^ {T} y + \rho A _ {2} ^ {T} b) = 0
$$

hence AD MM for Q P repeats the following steps : 

$$
x _ {1} ^ {k + 1} = - (H _ {1 1} + \rho A _ {1} ^ {T} A _ {1}) ^ {- 1} \Big ((H _ {2 1} ^ {T} + \rho A _ {1} ^ {T} A _ {2}) x _ {2} ^ {k} + (c _ {1} + A _ {1} ^ {T} y + \rho A _ {1} ^ {T} b) \Big)
$$

$$
x _ {2} ^ {k + 1} = - (H _ {2 2} + \rho A _ {2} ^ {T} A _ {2}) ^ {- 1} \Big ((H _ {2 1} + \rho A _ {2} ^ {T} A _ {1}) x _ {1} ^ {k + 1} + (c _ {2} + A _ {2} ^ {T} y + \rho A _ {2} ^ {T} b) \Big)
$$

$$
y ^ {k + 1} = y ^ {k} + \rho (A _ {1} x _ {1} ^ {k + 1} + A _ {2} x _ {2} ^ {k + 1} - b)
$$

## R<sub>e</sub>l<sub>a</sub>ti<sub>on</sub> b<sub>e</sub>t<sub>ween</sub> ADMM <sub>an</sub>d G<sub>auss-</sub>S<sub>e</sub>id<sub>e</sub>l

C<sub>ons</sub>id<sub>er a</sub> l<sub>arge sys</sub>t<sub>em o</sub>f li<sub>near equa</sub>ti<sub>ons</sub> $Q x = r$ <sub>w</sub>hi<sub>c</sub>h i<sub>nvo</sub>l<sub>ves</sub> <sub>a pos</sub>iti<sub>ve</sub> d<sub>e</sub>fi<sub>n</sub>it<sub>e ma</sub>t<sub>r</sub>i<sub>x</sub> $Q$ th<sub>a</sub>t i<sub>s</sub> d<sub>ecompose</sub>d i<sub>n</sub>t<sub>o</sub> $p \times p$ bl<sub>oc</sub>k<sub>s :</sub> 

$$
\left[ \begin{array}{c c c c} Q _ {1 1} & Q _ {1 2} & \dots & Q _ {1 p} \\ Q _ {2 1} & Q _ {2 2} & \dots & Q _ {2 p} \\ \vdots & \vdots & \ddots & \vdots \\ Q _ {p 1} & Q _ {p 2} & \dots & Q _ {p p} \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \\ \vdots \\ x _ {p} \end{array} \right] = \left[ \begin{array}{c} r _ {1} \\ r _ {2} \\ \vdots \\ r _ {p} \end{array} \right]
$$

(blocks may have diferent sizes) <sub>.</sub> 

G <sub>auss-</sub> S <sub>e</sub>id<sub>e</sub>l M<sub>e</sub>t h<sub>o</sub> d <sub>repea</sub>t<sub>s</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng s</sub>t<sub>eps :</sub> 

$$
x _ {1} ^ {k + 1} = Q _ {1 1} ^ {- 1} (r _ {1} - Q _ {1 2} x _ {2} ^ {k} - \ldots - Q _ {1 p} x _ {p} ^ {k})
$$

$$
x _ {2} ^ {k + 1} = Q _ {2 2} ^ {- 1} (r _ {2} - Q _ {2 1} x _ {1} ^ {k + 1} - Q _ {2 3} x _ {3} ^ {k} - \ldots - Q _ {2 p} x _ {p} ^ {k})
$$

$$
x _ {p} ^ {k + 1} = Q _ {p p} ^ {- 1} (r _ {p} - Q _ {p 1} x _ {1} ^ {k + 1} - Q _ {p 2} x _ {2} ^ {k + 1} - \ldots - Q _ {p, p - 1} x _ {p - 1} ^ {k + 1}).
$$

S <sub>.</sub> Ci<sub>po</sub>ll<sub>a,</sub> J <sub>.</sub> G<sub>on</sub>d<sub>z</sub>i<sub>o ,</sub> ADMM <sub>an</sub>d i<sub>nexac</sub>t ALM <sub>:</sub> th<sub>e</sub> $\mathrm { Q P }$ case <sub>.</sub> 

htt<sub>p</sub> s : / /www <sub>.</sub> maths <sub>.</sub> ed <sub>.</sub> ac <sub>.</sub> uk/ <sup>~</sup> <sub>g</sub>ondz i o /re<sub>p</sub>ort s /ADMMandI ALM <sub>.</sub> html 

## (Block) Gauss- Seidel Method

C<sub>ons</sub>id<sub>er</sub> th<sub>e</sub> f<sub>o</sub>ll<sub>ow</sub>i<sub>ng sp</sub>litt i<sub>ng o</sub>f th<sub>e ma</sub>t<sub>r</sub>i<sub>x</sub> 

$$
\left[ \begin{array}{c c c c} Q _ {1 1} & Q _ {1 2} & \dots & Q _ {1 p} \\ Q _ {2 1} & Q _ {2 2} & \dots & Q _ {2 p} \\ \vdots & \vdots & \ddots & \vdots \\ Q _ {p 1} & Q _ {p 2} & \dots & Q _ {p p} \end{array} \right] = \underbrace {\left[ \begin{array}{c c c c} Q _ {1 1} & 0 & \cdots & 0 \\ Q _ {2 1} & Q _ {2 2} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ Q _ {p 1} & Q _ {p 2} & \cdots & Q _ {p p} \end{array} \right]} _ {L} + \underbrace {\left[ \begin{array}{c c c c} 0 & Q _ {1 2} & \cdots & Q _ {1 p} \\ 0 & 0 & \cdots & Q _ {2 p} \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 0 \end{array} \right]} _ {U},
$$

<sub>an</sub>d <sub>rearrange</sub> th<sub>e equa</sub>ti<sub>on</sub> 

$$
Q x = (L + U) x = r \quad \Leftrightarrow \quad L x = r - U x \quad \Leftrightarrow \quad x = L ^ {- 1} (r - U x).
$$

Gauss- Seidel Method is a fixed point iteration: 

$$
x ^ {k + 1} = L ^ {- 1} (r - U x ^ {k}).
$$

G<sub>auss-</sub> S<sub>e</sub>id<sub>e</sub>l it<sub>era</sub>ti<sub>on overwr</sub>it<sub>es</sub> th<sub>e approx</sub>i<sub>ma</sub>t<sub>e so</sub>l<sub>u</sub>ti<sub>on w</sub>ith th<sub>e</sub> <sub>new va</sub>l<sub>ue as soon as</sub> it i<sub>s compu</sub>t<sub>e</sub>d <sub>:</sub> 

$$
x _ {i} ^ {k + 1} = Q _ {i i} ^ {- 1} (r _ {i} - \sum_ {j <   i} Q _ {i j} x _ {j} ^ {k + 1} - \sum_ {j > i} Q _ {i j} x _ {j} ^ {k}),
$$

AD MM for Q P acts as a Gauss- Seidel iterat ion <sub>.</sub> 

## Fi<sub>na</sub>l R<sub>emar</sub>k<sub>s</sub>

## Alternating Direction Method of Multipliers (ADMM)

• i<sub>s su</sub>it<sub>a</sub>bl<sub>e</sub> f<sub>or pro</sub>bl<sub>ems w</sub>ith l<sub>oose</sub>l<sub>y coup</sub>l<sub>e</sub>d <sub>var</sub>i<sub>a</sub>bl<sub>es</sub> 

• h<sub>as</sub> i<sub>nexpens</sub>i<sub>ve</sub> it<sub>era</sub>ti<sub>ons</sub> 

h<sub>ence</sub> i<sub>s a</sub>tt<sub>rac</sub>ti<sub>ve</sub> f<sub>or very</sub> l<sub>arge sca</sub>l<sub>e op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> 

• <sub>may</sub> b<sub>e s</sub>l<sub>ow</sub> b<sub>u</sub>t i<sub>s o</sub>ft<sub>en su</sub>fi<sub>c</sub>i<sub>en</sub>tl<sub>y</sub> f<sub>as</sub>t <sub>w</sub>h<sub>en appropr</sub>i<sub>a</sub>t<sub>e</sub>l<sub>y</sub> t<sub>une</sub>d 

• h<sub>as numerous app</sub>li<sub>ca</sub>ti<sub>ons</sub> d<sub>ue</sub> t<sub>o</sub> it<sub>s</sub> <sup>‘</sup> d<sub>ecoup</sub>li<sub>ng</sub> <sup>’</sup> <sub>a</sub>bilit<sub>y :</sub> 

– machine learning/statistics (large data sets) 

– <sup>i</sup>mage process<sup>i</sup>ng <sub>,</sub> 

– d<sub>ecen</sub>t<sub>ra</sub>li<sub>ze</sub>d <sub>op</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on .</sub> 

# M<sub>o</sub>d<sub>ern</sub> T<sub>ec</sub>h<sub>n</sub>i<sub>ques</sub> <sub>o</sub>f L<sub>arge</sub> S<sub>ca</sub>l<sub>e</sub> O<sub>p</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> f<sub>or</sub> D<sub>a</sub>t<sub>a</sub> S<sub>c</sub>i<sub>ence</sub>

Th<sub>an</sub>k <sub>you</sub> f<sub>or your a</sub>tt<sub>en</sub>ti<sub>on</sub>! 