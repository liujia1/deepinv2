# X-ray tomography minicourse: Monday exercises

Samuli Siltanen 

PhD Winter School 

Advanced methods for mathematical image analysis 

Bologna, January 23, 2023 

Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise

Tomography with 1×2 pixels: ill-posedness
Exercises, collected 

## Outline

$$
\begin{array}{c c} \text {   X - ray   source   } & 8 (= 2 + 6) \\ \hline \boxed {2} & 6 \\ \hline 2 & 7 \end{array}
$$

$$
\begin{array}{c c} \framebox {2} & 6 \\ \framebox {2} & 7 \end{array} \quad \begin{array}{c c} 8 (= 2 + 6) \\ 9 (= 2 + 7) \end{array}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/59250e902dfc9eb6f2e96672ef1edf70602ef36a699051fca712ebd6f2740f8c.jpg)



4 13


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/a732d4f5991afb89159fbc6ccb1a33f5825c204aee4436aee7acc1038735bf64.jpg)



4 13


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/17fe7a9010d46c3cfaf97a4062eda285864eb7e43bd18cc96094feed4426786f.jpg)



4 13


8 9 

$$
\begin{array}{c} 8 \\ 9 \end{array}
$$

8 9 

$$
\begin{array}{c} 8 \\ 9 \end{array}
$$

8 9 

8 9 

8 9 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/74a36d0439af2a2432ce4454f127b8a7ca206f9da95f88db47f424bdfe27f0c5.jpg)


$$
\begin{array}{c c} \framebox {2} & 6 \\ \framebox {2} & 7 \\ \framebox {4} & 1 3 \end{array} ^ {8} + \begin{array}{c c} \framebox {2} & - 2 \\ \framebox {- 2} & 2 \\ \framebox {0} & 0 \end{array} ^ {0}
$$

$$
= \begin{array}{c c} \hline 4 & 4 \\ \hline 0 & 9 \\ \hline 4 & 1 3 \end{array} ^ {8}
$$

## Outline

Tomography with 2×2 pixels: non-uniqueness Matrix model for the measurement First 2×2 exercise Total variation regularization Second 2×2 exercise Numerical implementation Third 2×2 exercise 

Tomography with 1×2 pixels: ill-posedness 

Exercises, collected 

Each data point gives rise to one row in the measurement matrix 

$$
\boxed {\begin{array}{c c}&\\X _ {1}&X _ {3}\\\hline X _ {2}&X _ {4}\end{array}} \rightarrow 8 \quad \left[\begin{array}{l l l l}1&0&1&0\\&&&\\&&&\end{array}\right]\left[\begin{array}{l}x _ {1}\\x _ {2}\\x _ {3}\\x _ {4}\end{array}\right] = \left[\begin{array}{l}8\\\end{array}\right]
$$

Each data point gives rise to one row in the measurement matrix 

$$
\begin{array}{c c} \boxed { \begin{array}{c c} & \\ x _ {1} & x _ {3} \end{array} } & 8 \\ \boxed { \begin{array}{c c} & \\ x _ {2} & x _ {4} \end{array} } & 9 \end{array} \quad \left[ \begin{array}{c c c c} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \\ x _ {3} \\ x _ {4} \end{array} \right] = \left[ \begin{array}{c} 8 \\ 9 \end{array} \right]
$$

Each data point gives rise to one row in the measurement matrix 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/25aee0bcc156ad2b4e7ea999943ca9ac164bd3749cdfe2d065a57830575730c4.jpg)


$$
{\left[ \begin{array}{l l l l} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \\ \end{array} \right]} {\left[ \begin{array}{l} x _ {1} \\ x _ {2} \\ x _ {3} \\ x _ {4} \end{array} \right]} = {\left[ \begin{array}{l} 8 \\ 9 \\ 4 \\ \end{array} \right]}
$$

Each data point gives rise to one row in the measurement matrix 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/8a783d3ee15c354a92875ccbcb193c877f46fb8d2032f7b3d02569a0e82a19c0.jpg)


$$
{\left[ \begin{array}{l l l l} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{array} \right]} {\left[ \begin{array}{l} x _ {1} \\ x _ {2} \\ x _ {3} \\ x _ {4} \end{array} \right]} = {\left[ \begin{array}{l} 8 \\ 9 \\ 4 \\ 1 3 \end{array} \right]}
$$

$$
A x = m
$$

## Outline

Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise

Tomography with 1×2 pixels: ill-posedness
Exercises, collected 

First $_ { 2 \times 2 }$ exercise 

Determine the kernel of the measurement matrix 

$$
A = \left[ \begin{array}{c c c c} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{array} \right].
$$

How is the kernel related to “ghosts”, or objects that are nontrivial but give zero measurement? 

Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise

Tomography with 1×2 pixels: ill-posedness
Exercises, collected 

## Outline

## Let’s study the two penalties used in regularization. We focus on three examples

Original patient 

$$
\begin{array}{c c} \hline 2 & 6 \\ \hline 2 & 7 \\ \hline \end{array}
$$

Flat candidate 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/2145b44d58a6c7d21522ca70a1a62cd73ea2158f10b579eb720dff2d139f3db8.jpg)



Wrong data, good “tissue type”



Spooky candidate


$$
\begin{array}{c c} \hline 4 & 4 \\ \hline 0 & 9 \\ \hline \end{array}
$$


Correct data, bad “tissue type”


Calculate data penalty for the original phantom 

$$
\begin{array}{c c} \hline & \\ 2 & 6 \\ \hline 2 & 7 \\ \hline \end{array} \quad (8 - 8) ^ {2}
$$

Calculate data penalty for the original phantom 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/1be3101cc81512597e3a7552b351036007956cf55f7c347d62dcf28466bc1ac3.jpg)


Data penalty: $( 8 - 8 ) ^ { 2 } + ( 9 - 9 ) ^ { 2 }$ 

Calculate data penalty for the original phantom 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/5f6c0db354c4613d82c597697353973b8d59ad243cef851a27db2e7d600bbfc5.jpg)


$$
(4 - 4) ^ {2} (1 3 - 1 3) ^ {2}
$$

Data penalty: $( 8 - 8 ) ^ { 2 } + ( 9 - 9 ) ^ { 2 } + ( 4 - 4 ) ^ { 2 } + ( 1 3 - 1 3 ) ^ { 2 } = 0 .$ 

# Calculate prior penalty for the original phantom

$$
\begin{array}{c c} \hline 2 & 6 \\ \hline 2 & 7 \\ \hline \end{array}
$$

Prior penalty: |2 − 6| 

Calculate prior penalty for the original phantom 

$$
\begin{array}{c c} \hline 2 & 6 \\ \hline 2 & 7 \\ \hline \end{array}
$$

Prior penalty: $\left| 2 - 6 \right| + \left| 2 - 7 \right|$ 

Calculate prior penalty for the original phantom 

<table><tr><td>2</td><td>6</td></tr><tr><td>2</td><td>7</td></tr></table>

Prior penalty: $\left| 2 - 6 \right| + \left| 2 - 7 \right| + \left| 2 - 2 \right|$ 

Calculate prior penalty for the original phantom 

<table><tr><td>2</td><td>6</td></tr><tr><td>2</td><td>7</td></tr></table>

Prior penalty: $| 2 - 6 | + | 2 - 7 | + | 2 - 2 | + | 6 - 7 | = 4 + 5 + 0 + 1 = 1 0 .$ 

Total penalty is the sum of data&prior penalties 

$$
\begin{array}{c c} \hline 2 & 6 \\ \hline 2 & 7 \\ \hline \end{array}
$$

data penalty 0 + prior penalty 10 = total penalty e10 

Data penalty for flat candidate 

$$
\begin{array}{c c} \hline 3 & 3 \\ \hline 3 & 3 \\ \hline (6 - 4) ^ {2} & (6 - 1 3) ^ {2} \end{array} \begin{array}{c} (6 - 8) ^ {2} \\ (6 - 9) ^ {2} \end{array}
$$

Data penalty: $2 ^ { 2 } + 3 ^ { 2 } + 2 ^ { 2 } + 7 ^ { 2 } = 4 + 9 + 4 + 4 9 = 6 6 .$ 

Prior penalty for flat candidate 

$$
\begin{array}{c c} \hline 3 & 3 \\ \hline 3 & 3 \\ \hline \end{array}
$$

Prior penalty: $| 3 - 3 | + | 3 - 3 | + | 3 - 3 | + | 3 - 3 | = 0 .$ 

## Total penalty for flat candidate

<table><tr><td>3</td><td>3</td></tr><tr><td>3</td><td>3</td></tr></table>

data penalty 66 + prior penalty 0 = total penalty e66 

Data penalty for spooky candidate 

$$
\begin{array}{c c c} \hline (8 - 8) ^ {2} & 4 & 4 \\ \hline (9 - 9) ^ {2} & 0 & 9 \\ \hline & (4 - 4) ^ {2} & (1 3 - 1 3) ^ {2} \\ \hline \end{array}
$$

Data penalty: $( 8 - 8 ) ^ { 2 } + ( 9 - 9 ) ^ { 2 } + ( 4 - 4 ) ^ { 2 } + ( 1 3 - 1 3 ) ^ { 2 } = 0 .$ 

Prior penalty for spooky candidate 

<table><tr><td>4</td><td>4</td></tr><tr><td>0</td><td>9</td></tr></table>

Prior penalty: $| 4 - 4 | + | 0 - 9 | + | 4 - 0 | + | 4 - 9 | = 0 + 9 + 4 + 5 = 1 8 .$ 

## Comparison of the three candidates

Original patient 

$$
\begin{array}{c c} \hline 2 & 6 \\ \hline 2 & 7 \\ \hline \end{array}
$$

Flat candidate 

$$
\begin{array}{c c} \hline 3 & 3 \\ \hline 3 & 3 \\ \hline \end{array}
$$

$$
\begin{array}{c c} \hline 4 & 4 \\ \hline 0 & 9 \\ \hline \end{array}
$$

Spooky candidate 

In practice we do not have three candidates. We need a general reconstruction algorithm 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/6603091caf54ff1b4b192cecf0d6762177a8ea5d17986efe8747331a8eadc999.jpg)


Find numbers $x _ { 1 } \geq 0 , \ x _ { 2 } \geq 0 , \ x _ { 3 } \geq 0$ and $x _ { 4 } ~ \geq ~ 0$ such that the sum of these two penalties is as small as possible: 

Data penalt $\begin{array} { c } { { / : \ ( x _ { 1 } + x _ { 3 } - 8 ) ^ { 2 } + ( x _ { 2 } + x _ { 4 } - 9 ) ^ { 2 } } } \\ { { + ( x _ { 1 } + x _ { 2 } - 4 ) ^ { 2 } + ( x _ { 3 } + x _ { 4 } - 1 3 ) ^ { 2 } } } \end{array}$ 

Prior penalty $\begin{array} { c } { { \mathrm { : ~ } \left| x _ { 1 } - x _ { 3 } \right| + \left| x _ { 2 } - x _ { 4 } \right| } } \\ { { \mathrm { + ~ } \left| x _ { 1 } - x _ { 2 } \right| + \left| x _ { 3 } - x _ { 4 } \right| } } \end{array}$ 

This method is called (anisotropic) total variation regularization. 

The minimizer of the TV penalty functional has two “internal organs”, as does the original 

Original patient 

TV minimizer 

$$
\begin{array}{c c} \hline 2 & 6 \\ \hline 2 & 7 \\ \hline \end{array}
$$

<table><tr><td></td><td>$ 6\frac{1}{4} $</td><td>$ 6\frac{1}{4} $</td></tr><tr><td>$ 2\frac{1}{4} $</td><td>$ 2\frac{1}{4} $</td><td>$ 2\frac{1}{4} $</td></tr></table>

data penalty 0 + prior penalty 10 = total penalty e10 

data penalty 1 + prior penalty 8 = total penalty e9 

Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise

Tomography with 1×2 pixels: ill-posedness
Exercises, collected 

## Outline

## Second $_ { 2 \times 2 }$ exercise, slide ${ \bf 1 } / 2$

Assume that we know three pixel values and look for the fourth one, called x. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/f6b5040c936375e3f17641698af71eaa667722cdc6a6ed6d3786013c177fde7a.jpg)


## Second $_ { 2 \times 2 }$ exercise, slide $2 / 2$

Take $\alpha = 1$ . Write down the total variation penalty functional in the form 

$$
\widetilde {x} = \underset {x \in \mathbb {R}} {\arg \min} \{f (x) \}.
$$

<sup>I</sup> Give the formula for f . 

<sup>I</sup> Plot $f ( x )$ 

<sup>I</sup> At what points does f fail to be diferentiable? 

7 Find the minimizing argument $\widetilde { x } \in \mathbb { R }$ approximately. You can either use brute-force forking or apply an optimization method. 

Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise

Tomography with 1×2 pixels: ill-posedness
Exercises, collected 

## Outline

Recall the matrix measurement model 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/c79dee2801360731248f0667a77259f160b3d576184b8c7a8de111be579a7ad3.jpg)


$$
{\left[ \begin{array}{l l l l} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{array} \right]} {\left[ \begin{array}{l} x _ {1} \\ x _ {2} \\ x _ {3} \\ x _ {4} \end{array} \right]} = {\left[ \begin{array}{l} 8 \\ 9 \\ 4 \\ 1 3 \end{array} \right]}
$$

$$
A x = m
$$

We can now formulate (anisotropic) total variation regularization mathematically 

$$
x _ {\mathrm{TV}} = \underset {x \in \mathbb {R} ^ {4}} {\arg \min} \left\{\| A x - m \| _ {2} ^ {2} + \| L _ {H} x \| _ {1} + \| L _ {V} x \| _ {1} \right\}
$$

Writing the prior penalty in matrix form: construction of the horizontal diference matrix $L _ { H }$ 

$$
\begin{array}{c c} \hline \mathbf {x} _ {1} & \mathbf {x} _ {3} \\ \hline \mathbf {x} _ {2} & \mathbf {x} _ {4} \\ \hline \end{array} \qquad \left[ \begin{array}{r r r r} 1 & 0 & - 1 & 0 \\ & & & \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \\ x _ {3} \\ x _ {4} \end{array} \right] = \left[ \begin{array}{c} x _ {1} - x _ {3} \\ \end{array} \right]
$$

Writing the prior penalty in matrix form: construction of the horizontal diference matrix $L _ { H }$ 

$$
\begin{array}{c c} \hline x _ {1} & x _ {3} \\ \hline x _ {2} & x _ {4} \\ \hline \end{array}
$$

$$
\underbrace {\left[ \begin{array}{l l l l} 1 & 0 & - 1 & 0 \\ 0 & 1 & 0 & - 1 \end{array} \right]} _ {L _ {H}} \left[ \begin{array}{l} x _ {1} \\ x _ {2} \\ x _ {3} \\ x _ {4} \end{array} \right] = \left[ \begin{array}{l} x _ {1} - x _ {3} \\ x _ {2} - x _ {4} \end{array} \right]
$$

Writing the prior penalty in matrix form: construction of the vertical diference matrix $L _ { V }$ 

$$
\begin{array}{c c} \hline \mathbf {x} _ {1} & \mathbf {x} _ {3} \\ \hline \mathbf {x} _ {2} & \mathbf {x} _ {4} \\ \hline \end{array} \qquad \left[ \begin{array}{r r r r} 1 & - 1 & 0 & 0 \\ & & & \end{array} \right] \left[ \begin{array}{c} x _ {1} \\ x _ {2} \\ x _ {3} \\ x _ {4} \end{array} \right] = \left[ \begin{array}{c} x _ {1} - x _ {2} \end{array} \right]
$$

Writing the prior penalty in matrix form: construction of the vertical diference matrix $L _ { V }$ 

<table><tr><td>$ x_{1} $</td><td>$ x_{3} $</td></tr><tr><td>$ x_{2} $</td><td>$ x_{4} $</td></tr></table>

$$
\underbrace {\left[ \begin{array}{c c c c} 1 & - 1 & 0 & 0 \\ 0 & 0 & 1 & - 1 \end{array} \right]} _ {L _ {V}} \left[ \begin{array}{l} x _ {1} \\ x _ {2} \\ x _ {3} \\ x _ {4} \end{array} \right] = \left[ \begin{array}{l} x _ {1} - x _ {2} \\ x _ {3} - x _ {4} \end{array} \right]
$$

## Matrix formulation of the anisotropic total variation prior penalty

Our minimization problem: 

$$
x _ {\mathrm{TV}} = \underset {x \in \mathbb {R} ^ {4}} {\arg \min} \left\{\| A x - m \| _ {2} ^ {2} + \| L _ {H} x \| _ {1} + \| L _ {V} x \| _ {1} \right\}
$$

Recall that for a vector $\boldsymbol { y } \in \mathbb { R } ^ { n }$ we have 

$$
\left\| y \right\| _ {1} = \left| y _ {1} \right| + \left| y _ {2} \right| + \dots + \left| y _ {n} \right|.
$$

Therefore, the prior penalty can be written as (why? check!) 

$$
\begin{array}{r c l} \| L _ {H} x \| _ {1} + \| L _ {V} x \| _ {1} & = & | x _ {1} - x _ {3} | + | x _ {2} - x _ {4} | \\ & & + | x _ {1} - x _ {2} | + | x _ {3} - x _ {4} |. \end{array}
$$

## Reformulation as a quadratic problem

We want to minimize the non-quadratic functional 

$$
\left\| A x - m \right\| _ {2} ^ {2} + \left\| L _ {H} x \right\| _ {1} + \left\| L _ {V} x \right\| _ {1}
$$

over non-negative image vectors $x \in \mathbb { R } ^ { 4 }$ . This task can be converted into minimizing the quadratic functiona 

$$
\frac {1}{2} z ^ {T} Q z + c ^ {T} z
$$

over non-negative $z \in \mathbb { R } ^ { 1 2 }$ with equality constraints $E z = b$ 

## Rewriting the TV regularization using the trick of non-negative vectors

Write the horizontal and vertical diferences in the form 

$$
L _ {\mathrm{H}} x = u _ {\mathrm{H}} ^ {+} - u _ {\mathrm{H}} ^ {-} \quad \text {and} \quad L _ {\mathrm{V}} x = u _ {\mathrm{V}} ^ {+} - u _ {\mathrm{V}} ^ {-},
$$

using non-negative vectors $u _ { \mathsf { H } } ^ { \pm } , u _ { \mathsf { V } } ^ { \pm } \in \mathbb { R } ^ { 2 }$ 

Then TV regularization is equivalent to minimizing 

$$
x ^ {T} A ^ {T} A x - 2 x ^ {T} A ^ {T} m + \left[ \begin{array}{c} 1 \\ 1 \end{array} \right] ^ {T} (u _ {\mathrm{H}} ^ {+} + u _ {\mathrm{H}} ^ {-} + u _ {\mathrm{V}} ^ {+} + u _ {\mathrm{V}} ^ {-}),
$$

over non-negative vectors $x \in \mathbb { R } ^ { 4 }$ (why? check!). 

Reduction of TV regularization to the quadratic problem arg min $\begin{array} { r } { \left\{ \frac { 1 } { 2 } z ^ { T } Q z + c ^ { T } z \right\} } \end{array}$ with $E z = b$ $z { \in } \mathbb { R } _ { + } ^ { 1 2 }$ 

So we aim to minimize ${ \textstyle { \frac { 1 } { 2 } } } z ^ { T } Q z + c ^ { T } z$ with 

$$
z = \left[ \begin{array}{c} x \\ u _ {\mathrm{H}} ^ {+} \\ u _ {\mathrm{H}} ^ {-} \\ u _ {\mathrm{V}} ^ {+} \\ u _ {\mathrm{V}} ^ {-} \end{array} \right] \in \mathbb {R} _ {+} ^ {1 2},
$$

$$
Q = \left[ \begin{array}{c c c c} 2 A ^ {T} A & 0 & \ldots & 0 \\ 0 & 0 & \ldots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & \ldots & & 0 \end{array} \right], \qquad c = \left[ \begin{array}{c} - 2 A ^ {T} m \\ 1 \\ \vdots \\ 1 \end{array} \right].
$$

## Explicit form of the equality constraint, slide ${ \bf 1 } / 2$

The equality constraint $E z = b$ is needed for enforcing the identities $L _ { \mathsf { H } } x - u _ { \mathsf { H } } ^ { + } + u _ { \mathsf { H } } ^ { - } = 0$ and $L _ { \mathsf { v } } x - u _ { \mathsf { v } } ^ { + } + u _ { \mathsf { v } } ^ { - } = 0$ 

Since 

$$
z = \left[ \begin{array}{c} x \\ u _ {\mathrm{H}} ^ {+} \\ u _ {\mathrm{H}} ^ {-} \\ u _ {\mathrm{V}} ^ {+} \\ u _ {\mathrm{V}} ^ {-} \end{array} \right] \in \mathbb {R} ^ {1 2},
$$

we have 

$$
E z = \left[ \begin{array}{c c c c c} L _ {H} & - I & I & 0 & 0 \\ L _ {V} & 0 & 0 & - I & I \end{array} \right] z = 0.
$$

Explicit form of the equality constraint, slide $2 / 2$ 

Finally we get 

$$
E = \left[ \begin{array}{c c c c c c c c c c c c} 1 & 0 & - 1 & 0 & - 1 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & - 1 & 0 & - 1 & 0 & 1 & 0 & 0 & 0 \\ \hline 1 & - 1 & 0 & 0 & 0 & 0 & 0 & 0 & - 1 & 0 & 1 & 0 \\ 0 & 0 & 1 & - 1 & 0 & 0 & 0 & 0 & 0 & - 1 & 0 & 1 \end{array} \right]
$$

and 

$$
b = \left[ \begin{array}{c} 0 \\ 0 \\ 0 \\ 0 \end{array} \right].
$$

```matlab
% Record the size of the unknown. Here M*M=n, % since the unknown is a MxM pixel image.
n = 4; M = 2; 
```

## Implementation in Matlab: preliminaries

```matlab
% Construct the 2x2 pixel image target as a vertical vector target = [2;2;6;7]; 
```

```matlab
% Construct the measurement matrix
A = [1 0 1 0; 0 1 0 1; 1 1 0 0; 0 0 1 1]; 
```

```matlab
% Compute an ideal X-ray measurement
m = A*target; 
```

## Regularization parameter

We are actually considering the TV regularization problem in a restricted form. In general it is advisable to solve 

$$
x _ {\text {TV} (\alpha)} = \underset {x \in \mathbb {R} _ {+} ^ {2 \times 2}} {\arg \min} \{\| A x - m \| _ {2} ^ {2} + \alpha \| L _ {H} x \| _ {1} + \alpha \| L _ {V} x \| _ {1} \},
$$

where $\alpha > 0$ is a regularization parameter. However, for now we keep α = 1 and write the following in Matlab: 

```matlab
% Regularization parameter
alpha = 1; 
```

```txt
% Construct prior matrices
LH = [1 0 -1 0; 0 1 0 -1];
LV = [1 -1 0 0; 0 0 1 -1]; 
```

```matlab
% Construct the quadratic optimization problem matrix
Q = zeros(n+4*M*(M-1));
Q(1:n,1:n) = 2*A.*A; 
```

```matlab
% Construct the vector h of the linear term
c = alpha*ones(n+4*M*(M-1),1);
c(1:n) = -2*(A.)*m(:); 
```

```matlab
% Construct input arguments for quadprog.m
Z = zeros(M*(M-1));
Aeq = [[LH, -eye(M*(M-1)), eye(M*(M-1)), Z, Z]; ...
    [LV, Z, Z, -eye(M*(M-1)), eye(M*(M-1))]];
beq = zeros(2*M*(M-1), 1);
lb = zeros(n+4*M*(M-1), 1);
ub = Inf(5*n, 1);
AA = -eye(n+4*M*(M-1));
AA(1:n, 1:n) = zeros(n, n);
iniguess = zeros(n+4*M*(M-1), 1);
b = [repmat(10, n, 1); zeros(4*M*(M-1), 1)];
QPopt = optimset('quadprog');
QPopt = optimset(QPopt, 'Algorithm', ...
    'interior-point-convex', 'Display', 'iter'); 
```

```matlab
% Compute reconstruction using quadprog
z = quadprog(Q, c, AA, b, Aeq, beq, lb, ub, iniguess, QPopt);
% Pick out the reconstructed image
recn = z(1:n);
% Show the reconstruction in image format
reshape(recn, M, M) 
```

>> reshape(recn,M,M)) 

Total variation regularization 

<table><tr><td></td><td>$ 6\frac{1}{4} $</td><td>$ 6\frac{1}{4} $</td></tr><tr><td>$ 2\frac{1}{4} $</td><td>$ 2\frac{1}{4} $</td><td>$ 2\frac{1}{4} $</td></tr></table>

data penalty 1 + prior penalty 8 = total penalty e9 

## Outline

Tomography with 2×2 pixels: non-uniqueness Matrix model for the measurement First 2×2 exercise Total variation regularization Second 2×2 exercise Numerical implementation Third 2×2 exercise 

Tomography with 1×2 pixels: ill-posedness 

Exercises, collected 

## Third $_ { 2 \times 2 }$ exercise

Run the computation of the previous slide using the Matlab routine tomo2x2_TV_comp_quadprog.m in the Git repository https://github.com/ssiltane/BolognaWinterSchool2023 Note that you need the Optimization Toolbox. 

Then repeat the computation with several values of regularization parameter $\alpha > 0$ . What choice of $\alpha > 0$ gives the smallest diference (measured in standard Euclidean norm of $\mathbb { R } ^ { 4 } )$ between the true target and the regularized solution? Give the optimal α with the accuracy of two correct digits after the decimal point. 

## Outline

Tomography with 2×2 pixels: non-uniqueness Matrix model for the measurement First 2×2 exercise Total variation regularization Second 2×2 exercise Numerical implementation Third 2×2 exercise 

Tomography with 1×2 pixels: ill-posedness 

Exercises, collected 

The first X-ray in our measurement travels horizontally 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/eb6e36b4632d119eb90f9af63a466aca4d3f3f011eca63caad003fc5faa065a3.jpg)


Second X-ray in the measurement has slope $1 / 2$ Note the geometric parameter $0 < h < 1 / 2$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/401dad9febb4f72f87f5f777d35b41631fbe0539149ccf09dd2eab45292e5726.jpg)


## First $\mathbf { 1 } \times 2$ exercise: construct measurement matrix

Assuming that the side length of pixel is one, write down the $2 \times 2$ matrix $A _ { h }$ modelling the measurement. (Some of the matrix elements may depend on $h > 0 . )$ 

<sup>I</sup> Show that $A _ { h }$ is invertible for any $0 < h < 1 / 2$ 

<sup>I</sup> What happens to $\mathsf { d e t } ( A _ { h } )$ when $h \to 0 ? \ \mathsf { W h y ? }$ 

I We assume everywhere else that $0 < h < 1 / 2$ . However, in this problem we step outside that assumption a bit. Is $A _ { h }$ invertible when $1 / 2 \le h < 1 ?$ How about the case $h \geq 1 ?$ 

## Naive inversion

The measurement model is 

$$
A _ {h} \left[ \begin{array}{c} x _ {1} \\ x _ {2} \end{array} \right] = \left[ \begin{array}{c} m _ {1} \\ m _ {2} \end{array} \right],
$$

or $A _ { h } x = m$ in short. Now assume that we have noisy data 

$$
\widetilde {m} = A _ {h} x + \varepsilon .
$$

Here $\varepsilon \in \mathbb { R } ^ { 2 }$ is a random noise vector. If $A _ { h }$ is invertible, we can attempt naive inversion $A _ { h } ^ { - 1 } \widetilde { m }$ . In the next exercise you will analyse this idea. 

## Second 1×2 exercise: ill-posedness of naive inversion

Naive reconstruction is an approximation of the unknown x, as we can see by this calculation: 

$$
A _ {h} ^ {- 1} \widetilde {m} = A _ {h} ^ {- 1} (A _ {h} x + \varepsilon) = x + A _ {h} ^ {- 1} \varepsilon .
$$

So we can bound the error by 

$$
\| A _ {h} ^ {- 1} \varepsilon \| _ {\mathbb {R} ^ {2}} \leq \| A _ {h} ^ {- 1} \| _ {\mathbb {R} ^ {2} \to \mathbb {R} ^ {2}} \| \varepsilon \| _ {\mathbb {R} ^ {2}},
$$

where $\| A _ { h } ^ { - 1 } \| _ { \mathbb { R } ^ { 2 } \to \mathbb { R } ^ { 2 } }$ is the operator norm of $A _ { h }$ . 

Compute the eigenvalues $\lambda _ { 1 } ( h ) > 0$ and $\lambda _ { 2 } ( h ) > 0$ of the matrix $A _ { h } ^ { T } A _ { h }$ numerically for a sequence of h values approaching zero. The numbers $s _ { j } ( h ) = \sqrt { \lambda _ { j } ( h ) }$ are called singular values of $A _ { h }$ . We order them so that $s _ { 1 } \geq s _ { 2 }$ 

<sup>I</sup> Now $\| A _ { h } ^ { - 1 } \| _ { \mathbb { R } ^ { 2 } \to \mathbb { R } ^ { 2 } } = 1 / s _ { 2 } ( h )$ . What happens to $\| A _ { h } ^ { - 1 } \| _ { \mathbb { R } ^ { 2 } \to \mathbb { R } ^ { 2 } }$ when $h  0 ?$ What does that mean for the error bound? 

Tomography with 2×2 pixels: non-uniqueness
Matrix model for the measurement
First 2×2 exercise
Total variation regularization
Second 2×2 exercise
Numerical implementation
Third 2×2 exercise

Tomography with 1×2 pixels: ill-posedness
Exercises, collected 

## Outline

First $_ { 2 \times 2 }$ exercise 

Determine the kernel of the measurement matrix 

$$
A = \left[ \begin{array}{c c c c} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{array} \right].
$$

How is the kernel related to “ghosts”, or objects that are nontrivial but give zero measurement? 

## Second $_ { 2 \times 2 }$ exercise, slide ${ \bf 1 } / 2$

Assume that we know three pixel values and look for the fourth one, called x. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/46fc1370-6c1e-4c72-97af-d3b866bc7af7/1ccd1d6883e247ceadfee6c77d053941834a5630c8d5001c40289f257aeeb9e5.jpg)


## Second $_ { 2 \times 2 }$ exercise, slide $2 / 2$

Take $\alpha = 1$ . Write down the total variation penalty functional in the form 

$$
\widetilde {x} = \underset {x \in \mathbb {R}} {\arg \min} \{f (x) \}.
$$

<sup>I</sup> Give the formula for f . 

<sup>I</sup> Plot $f ( x )$ 

<sup>I</sup> At what points does f fail to be diferentiable? 

7 Find the minimizing argument $\widetilde { x } \in \mathbb { R }$ approximately. You can either use brute-force forking or apply an optimization method. 

## Third $_ { 2 \times 2 }$ exercise

Run the computation of the previous slide using the Matlab routine tomo2x2_TV_comp_quadprog.m in the Git repository https://github.com/ssiltane/BolognaWinterSchool2023 Note that you need the Optimization Toolbox. 

Then repeat the computation with several values of regularization parameter $\alpha > 0$ . What choice of $\alpha > 0$ gives the smallest diference (measured in standard Euclidean norm of $\mathbb { R } ^ { 4 } )$ between the true target and the regularized solution? Give the optimal α with the accuracy of two correct digits after the decimal point. 

## First $\mathbf { 1 } \times 2$ exercise: construct measurement matrix

Assuming that the side length of pixel is one, write down the $2 \times 2$ matrix $A _ { h }$ modelling the measurement. (Some of the matrix elements may depend on $h > 0 . )$ 

<sup>I</sup> Show that $A _ { h }$ is invertible for any $0 < h < 1 / 2$ 

<sup>I</sup> What happens to $\mathsf { d e t } ( A _ { h } )$ when $h \to 0 ? \ \mathsf { W h y ? }$ 

I We assume everywhere else that $0 < h < 1 / 2$ . However, in this problem we step outside that assumption a bit. Is $A _ { h }$ invertible when $1 / 2 \le h < 1 ?$ How about the case $h \geq 1 ?$ 

## Second 1×2 exercise: ill-posedness of naive inversion

Naive reconstruction is an approximation of the unknown x, as we can see by this calculation: 

$$
A _ {h} ^ {- 1} \widetilde {m} = A _ {h} ^ {- 1} (A _ {h} x + \varepsilon) = x + A _ {h} ^ {- 1} \varepsilon .
$$

So we can bound the error by 

$$
\| A _ {h} ^ {- 1} \varepsilon \| _ {\mathbb {R} ^ {2}} \leq \| A _ {h} ^ {- 1} \| _ {\mathbb {R} ^ {2} \to \mathbb {R} ^ {2}} \| \varepsilon \| _ {\mathbb {R} ^ {2}},
$$

where $\| A _ { h } ^ { - 1 } \| _ { \mathbb { R } ^ { 2 } \to \mathbb { R } ^ { 2 } }$ is the operator norm of $A _ { h }$ . 

Compute the eigenvalues $\lambda _ { 1 } ( h ) > 0$ and $\lambda _ { 2 } ( h ) > 0$ of the matrix $A _ { h } ^ { T } A _ { h }$ numerically for a sequence of h values approaching zero. The numbers $s _ { j } ( h ) = \sqrt { \lambda _ { j } ( h ) }$ are called singular values of $A _ { h }$ . We order them so that $s _ { 1 } \geq s _ { 2 }$ 

<sup>I</sup> Now $\| A _ { h } ^ { - 1 } \| _ { \mathbb { R } ^ { 2 } \to \mathbb { R } ^ { 2 } } = 1 / s _ { 2 } ( h )$ . What happens to $\| A _ { h } ^ { - 1 } \| _ { \mathbb { R } ^ { 2 } \to \mathbb { R } ^ { 2 } }$ when $h  0 ?$ What does that mean for the error bound? 

## SCIENCE

THETRUE FANTASY 