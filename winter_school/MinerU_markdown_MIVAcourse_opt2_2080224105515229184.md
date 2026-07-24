## Lecture 2: Convex non-smooth optimisation

Luca Calatroni CR CNRS, Laboratoire I3S CNRS, UCA, Inria SAM, France 

MIVA ERASMUS BIP PhD winter school Advanced methods for mathematical image analysis University of Bologna, IT January 18-20 2022 

## Table of contents

1. Non-smooth optimisation Subgradients The proximal operator Projected gradient descent 

2. The proximal gradient algorithm Convergence properties 

3. Acceleration strategies FISTA Strongly convex FISTA 

4. Extensions Inexact algorithms Backtracking strategies for FISTA 

5. Non-convex algorithms 

In many applications the function $\boldsymbol { g }$ in 

$$
\boxed {\min _ {x \in \mathbb {R} ^ {n}} \left\{F (x) := f (x) + g (x) \right\},}
$$

is diferent from 0. Typically, $\boldsymbol { g }$ is convex, but non diferentiable so its gradient (and henceforth the one of $F )$ cannot be defined in a standard way. 

Note: take implicit gradient-descent for suitable $\tau > 0$ 

$$
x _ {k + 1} = x _ {k} - \tau \nabla f (x _ {k + 1}) \quad \Leftrightarrow \quad \nabla f (x _ {k + 1}) + \frac {x _ {k + 1} - x _ {k}}{\tau} = 0,
$$

So if $x _ { k + 1 }$ exists, it is a critical point of the function: 

$$
x \mapsto f (x) + \frac {\| x _ {k} - x \| ^ {2}}{2 \tau}
$$

If $f \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ (not necessarily smooth!), $x _ { k + 1 }$ is indeed the unique critical point of this function. . . 

non-smoothness encoded via “implicit” updates? 

Non-smooth optimisation 

# Non-smooth optimisation

Subgradients 

## A preliminary observation

One can show that if $f : \mathbb { R } ^ { n } \to { \overline { { \mathbb { R } } } }$ is diferentiable: 

$$
f \text {   is   convex   } \quad \Leftrightarrow \quad (\forall x, y \in \mathbb {R} ^ {n}) \quad f (y) \geq \underbrace {f (x) + \nabla f (x) ^ {T} (y - x)} _ {=: \phi (y; x)}
$$

• the function $\phi ( \cdot ; x )$ is an afine lower bound/estimator of $f ( \cdot )$ 

• the tangent to $f$ at any $x \in \mathsf { d o m } ( f )$ is below f at all points. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/806b24738afab6d10cee13e29b77a7a8267edef11c717b118d47a96c56b697cf.jpg)


## A preliminary observation

One can show that if $f : \mathbb { R } ^ { n } \to { \overline { { \mathbb { R } } } }$ is diferentiable: 

$$
f \text {   is   convex   } \quad \Leftrightarrow \quad (\forall x, y \in \mathbb {R} ^ {n}) \quad f (y) \geq \underbrace {f (x) + \nabla f (x) ^ {T} (y - x)} _ {=: \phi (y; x)}
$$

• the function $\phi ( \cdot ; x )$ is an afine lower bound/estimator of $f ( \cdot )$ 

• the tangent to $f$ at any $x \in \mathsf { d o m } ( f )$ is below $f$ at all points. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/7e54302abb077cf6c39cd5bbcc4e4eef5c933481309feb7aebe1a3c3baa7d604.jpg)


Recall: If $f$ is $\mu -$ strongly convex, then, analogously, $f$ has a quadratic lower bound 

$$
f (y) \geq f (x) + \langle \nabla f (x), y - x \rangle + \frac {\mu}{2} \| x - y \| ^ {2}, \quad \forall x, y \in \mathbb {R} ^ {n}.
$$

## Subgradients and subdiferential

Definition (Subgradients and subdifferential) Let $g \in \mathcal{P}$ be convex. Then, a vector $p \in \mathbb{R}^n$ is a subgradient of $g$ at point $x \in \operatorname{dom}(g)$ iff: $g(y) \geq g(x) + p^T(y - x), \quad \forall y \in \mathbb{R}^n$ If $x \notin \operatorname{dom}(g)$ , we set $\partial g(x) = \emptyset$ . The set of all subgradients at a point $x \in \mathbb{R}^n$ is called the subdifferential of $g$ in $x$ , and it is the denoted by: $\partial g(x) = \{p \in \mathbb{R}^n : p \text{ is a subgradient of } g \text{ at point } x\}$ 

## Interpretation:

$p \in \partial g ( x )$ if and only if $\phi ( y ; x ) = g ( x ) + p ^ { T } ( y - x )$ is a lower afine bound for $g$ 

$\partial g ( x )$ collects all the slopes of the tangent lines through x. 

## Remarks

In general, $\partial g ( \cdot ) : \mathbb { R } ^ { n } \to 2 ^ { \mathbb { R } ^ { n } }$ is not a singleton 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/9792f2e419b1299b9d26e5d22e60d8c852cef234959909c3030c968d20d3ef52.jpg)


Multiple subgradients at a non-diferentiable point $x _ { 0 }$ 

Example: $g : \mathbb { R }  \overline { { \mathbb { R } } } , g ( x ) = | x |$ 

$$
\partial g (x) = \left\{ \begin{array}{l l} \{1 \} & \text { if } \quad x > 0 \\ \{- 1 \} & \text { if } \quad x <   0 \\ [ - 1, 1 ] & \text { if } \quad x = 0. \end{array} \right.
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/5b25bdd48d683ca03a87d7213070785668a5899255eae25a6c7fab8c0f91709c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/9ce3b7ce18212d56b148f310fe2a06ca5592776bc112bc512d7968752a67a9ed.jpg)


## Remarks

In general, $\partial g ( \cdot ) : \mathbb { R } ^ { n } \to 2 ^ { \mathbb { R } ^ { n } }$ is not a singleton 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/ac96df427edd84713e4d5f8d90dc90c34b3dfa2451f7a06e548f5a399a281628.jpg)


Multiple subgradients at a non-diferentiable point $x _ { 0 }$ 

Example: $g : \mathbb { R }  \overline { { \mathbb { R } } } , g ( x ) = | x |$ 

$$
\partial g (x) = \left\{ \begin{array}{l l} \{1 \} & \text { if } \quad x > 0 \\ \{- 1 \} & \text { if } \quad x <   0 \\ [ - 1, 1 ] & \text { if } \quad x = 0. \end{array} \right.
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/81d827c85d6af738fbda6deda84df882c83f3a79a71b529f8491981e9be0bc81.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/1d11f6786ae2d96b22e6111a4d25c2c43e1189960759e282e6ffe02d15238af1.jpg)


Proposition (subdiferential at diferentiable points) 

If g is convex and diferentiable in $x \in \mathsf { d o m } ( g )$ , then: 

$$
\partial g (x) = \{\nabla g (x) \}.
$$

## Subdiferential of norm

Compute $\partial \| x \|$ for all $x \in \mathbb { R } ^ { n }$ 

$g ( x ) = \| x \|$ is diferentiable for all $x \neq 0$ . There, $\textstyle \partial \| x \| = { \frac { x } { \| x \| } }$ 

• The point of interest (non-diferentiability) is 0 

In $x = 0$ subgradients $p \in \mathbb { R } ^ { n }$ verify: 

$$
\| y \| \geq 0 + p ^ {T} (y - 0) = p ^ {T} y \quad \forall y \in \mathbb {R} ^ {n}
$$

Take the maximum on both sides for all $y : \| y \| \leq 1$ , you get: 

$$
1 = \max _ {y: \| y \| \leq 1} \| y \| \geq \max _ {y: \| y \| \leq 1} p ^ {T} y = \| p \|
$$

Contrarily, $\mathsf { i f } \parallel p \parallel \leq 1$ , then by Cauchy-Schwarz inequality there holds: 

$$
p ^ {T} y \leq \| p \| \| y \| \leq \| y \|
$$

Hence, we proved $p \in \partial \Vert 0 \Vert$ if and only $\mathsf { i f } \parallel p \parallel \leq 1$ . Hence 

$$
\partial \| 0 \| = \{p \in \mathbb {R} ^ {n}: \| p \| \leq 1 \} = B _ {1} (0) \quad \Rightarrow \quad \partial \| x \| = \left\{ \begin{array}{l l} \frac {x}{\| x \|} & \quad x \neq 0 \\ B _ {1} (0) & \quad x = 0. \end{array} \right.
$$

## Calculus rules: separable functions

Often, the n-dimensional function you deal with, can be nicely expressed as the sum of 1D components. For instance, think of: 

• norms $\begin{array} { r } { \| { \boldsymbol { x } } \| _ { p } ^ { p } , p \geq 1 ; \ \| { \boldsymbol { x } } \| _ { p } ^ { p } = \sum _ { i = 1 } ^ { n } | x _ { i } | ^ { p } . } \end{array}$ . . . 

• sum of norms, e.g. $\begin{array} { r } { g ( x ) = \| x \| _ { 1 } + \frac { \lambda } { 2 } \| x \| _ { 2 } ^ { 2 } = \sum _ { i = 1 } ^ { n } \left( | x _ { i } | + \lambda | x _ { i } | ^ { 2 } \right) } \end{array}$ 

• . . . 

## Calculus rules: separable functions

Often, the n-dimensional function you deal with, can be nicely expressed as the sum of 1D components. For instance, think of: 

• norms $\begin{array} { r } { \| { \boldsymbol { x } } \| _ { p } ^ { p } , p \geq 1 ; \ \| { \boldsymbol { x } } \| _ { p } ^ { p } = \sum _ { i = 1 } ^ { n } | x _ { i } | ^ { p } . } \end{array}$ 

• sum of norms, e.g. $\begin{array} { r } { g ( x ) = \| x \| _ { 1 } + \frac { \lambda } { 2 } \| x \| _ { 2 } ^ { 2 } = \sum _ { i = 1 } ^ { n } \left( | x _ { i } | + \lambda | x _ { i } | ^ { 2 } \right) } \end{array}$ 

• . . . 

## Definition (separable function)

Let $g \in { \mathcal { P } }$ be convex. We say that $g$ is separable if there exist proper, univariate convex functions $\begin{array} { r } { g _ { i } : \mathbb { R }  \overline { { \mathbb { R } } } } \end{array}$ such that 

$$
g (x) = \sum_ {i = 1} ^ {n} g _ {i} (x _ {i}), \quad \forall x \in \mathbb {R} ^ {n}.
$$

## Proposition (subdiferential of separable functions)

Let $g \in { \mathcal { P } }$ be convex and separable. Then, for all $x \in \mathsf { d o m } ( g )$ : 

$$
\partial g (x) = (\partial g _ {i} (x _ {i})) _ {i = 1} ^ {n} = (\partial g _ {1} (x _ {1})) \times \ldots \times (\partial g _ {n} (x _ {n})).
$$

## Calculus rules: sum and multiplication by scalar

## Proposition (Moreau-Rockafellar)

Let $g , g _ { 2 } : \mathbb { R } ^ { n }  \overline { { \mathbb { R } } }$ be two proper convex functions. Then: 

$$
\partial g _ {1} (x) + \partial g _ {2} (x) \subset \partial \left(g _ {1} (\cdot) + g _ {2} (\cdot)\right) (x).
$$

Moreover, if $\mathfrak { i n t } ( d o m ( g _ { 1 } ) ) \cap \mathfrak { i n t } ( d o m ( g _ { 2 } ) ) \neq \emptyset$ , then for all $x \in \mathbb { R } ^ { n }$ : 

$$
\partial g _ {1} (x) + \partial g _ {2} (x) = \partial \left(g _ {1} (\cdot) + g _ {2} (\cdot)\right) (x).
$$

For $\lambda \in \mathbb { R } _ { + + }$ , there holds: 

$$
\partial (\lambda f) (x) = \lambda \partial f (x), \quad \forall x \in \mathbb {R} ^ {n}.
$$

Example: $\partial ( g _ { 1 } ( \cdot ) + g _ { 2 } ( \cdot ) ) ( x )$ may difer indeed from $\partial g _ { 1 } ( x ) + \partial g _ { 2 } ( x ) !$ In $\mathbb { R }$ take: 

$$
g _ {1} (x) := \left\{ \begin{array}{l l} 0 & \text {if x\leq 0} \\ + \infty & \text {if x > 0.} \end{array} \right. \quad g _ {2} (x) := \left\{ \begin{array}{l l} + \infty & \text {if x <   0} \\ - \sqrt {x} & \text {if x\geq 0.} \end{array} \right.
$$

We have: 

$$
\partial g _ {1} (x) = \left\{ \begin{array}{l l} 0 & \text {   if   } x <   0 \\ [ 0, + \infty) & \text {   if   } x = 0 \\ \emptyset & \text {   if   } x > 0 \end{array} \right. \quad \partial g _ {2} (x) = \left\{ \begin{array}{l l} \emptyset & \text {   if   } x \leq 0 \\ - \frac {1}{2 \sqrt {x}} & \text {   if   } x > 0. \end{array} \right.
$$

Hence, $\partial g _ { 1 } ( x ) + \partial g _ { 2 } ( x ) = \varnothing$ for $\mathsf { a l l } \boldsymbol { x } \in \mathbb { R }$ . However, $g _ { 1 } ( x ) + g _ { 2 } ( x ) = \iota _ { 0 } ( x )$ and $\partial \iota _ { 0 } ( 0 ) = \mathbb { R }$ 

## Proposition

Let $f \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ be diferentiable at $x \in \mathbb { R } ^ { n }$ and let $g \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ , then: 

$$
\partial (f + g) (x) = \{\nabla f (x) \} + \partial g (x).
$$

## Proposition

Let $\boldsymbol { L } \in \mathbb { R } ^ { N \times n }$ and $ { \boldsymbol { g } } : \mathbb { R } ^ { N } \to \bar { \mathbb { R } }$ a proper convex function. Then: 

$$
(\forall x \in \mathbb {R} ^ {n}) L ^ {T} \partial g (L x) \subset \partial (g \circ L) (x).
$$

Moreover, if $\mathfrak { i n t } ( d o m ( g ) \cap R ( L ) \neq \emptyset$ , then: 

$$
(\forall x \in \mathbb {R} ^ {n}) L ^ {T} \partial g (L x) = \partial (g \circ L) (x).
$$

## Optimality conditions

Analogous to Fermat’s rule in non-smooth case. 

Theorem (optimality conditions in non-smooth, convex case) Let $g \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ . Then: 

$$
x ^ {*} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} g (x) \qquad \Longleftrightarrow \qquad 0 \in \partial g (x ^ {*}).
$$

Interpretation: 

• If the vector $0 \in \mathbb { R } ^ { n }$ belongs to $\partial g ( x ^ { * } ) \ ( \ ^ { * } { \mathsf { f } } { \mathsf { l a t \ p l o t " } } )$ , then $x ^ { * }$ is a minimiser. 

• If $g$ is diferentiable, the result reads $0 = \nabla g ( x ^ { * } )$ (Fermat’s rule). 

## Stationary points

If f , $g \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ and f is smooth 

$$
\underset {x \in \mathbb {R} ^ {n}} {\arg \min} \left\{F (x) := f (x) + g (x) \right\}
$$

$$
x ^ {*} \in \underset {x \in \mathbb {R} ^ {n}} {\arg \min} F (x) \Leftrightarrow 0 \in \partial F (x ^ {*}) = \underbrace {\partial f (x ^ {*})} _ {f \text {   is   smooth }} + \partial g (x ^ {*}) = \{\nabla f (x ^ {*}) \} + \partial g (x ^ {*})
$$

## Definition (stationary point)

A point $x ^ { \ast } \in \mathbb { R } ^ { n }$ verifying: 

$$
0 \in \{\nabla f (x ^ {*}) \} + \partial g (x ^ {*}) \Leftrightarrow - \nabla f (x ^ {*}) \in \partial g (x ^ {*})
$$

is said to be a stationary point of the composite functional $F : = f + g$ 

The proximal operator 

## The proximal operator: definition

Crucial tool for the development of non-smooth optimisation algorithms. Relations with activation functions in the context of deep networks (Combettes, Pesquet, ’20) 

## Definition

Let $g \in { \mathcal { P } }$ . Then, the proximal operator of $\boldsymbol { g }$ with parameter $\gamma > 0$ is defined as the multi-valued map p $\mathsf { r o x } _ { \gamma g } : \mathbb { R } ^ { n } \to 2 ^ { \mathbb { R } ^ { n } }$ defined for all $x \in \mathbb { R } ^ { n }$ 

$$
\operatorname{prox} _ {\gamma g} (x) := \underset {y \in \mathbb {R} ^ {n}} {\arg \min} \underbrace {g (y) + \frac {1}{2 \gamma} \| y - x \| ^ {2}} _ {=: h (y; x)}
$$

With no further conditions on ${ \boldsymbol { g } } , \mathsf { p r o x } _ { \gamma { \boldsymbol { g } } } ( { \boldsymbol { x } } )$ is a multivalued set and there may exist ${ \hat { x } } \in \mathbb { R } ^ { n }$ s.t. $\mathsf { p r o x } _ { \gamma g } ( \hat { x } ) = \varnothing$ 

## The proximal operator: definition

Crucial tool for the development of non-smooth optimisation algorithms. Relations with activation functions in the context of deep networks (Combettes, Pesquet, ’20) 

## Definition

Let $g \in { \mathcal { P } }$ Then, the proximal operator of $\boldsymbol { g }$ with parameter $\gamma > 0$ is defined as the multi-valued map pro $\mathbf { \boldsymbol { x } } _ { \gamma g } : \mathbb { R } ^ { n }  2 ^ { \mathbb { R } ^ { n } }$ defined for all $x \in \mathbb { R } ^ { n }$ : 

$$
\operatorname{prox} _ {\gamma g} (x) := \underset {y \in \mathbb {R} ^ {n}} {\arg \min} \underbrace {g (y) + \frac {1}{2 \gamma} \| y - x \| ^ {2}} _ {=: h (y; x)}
$$

With no further conditions on ${ \boldsymbol { g } } , \mathsf { p r o x } _ { \gamma { \boldsymbol { g } } } ( { \boldsymbol { x } } )$ is a multivalued set and there may exist ${ \hat { x } } \in \mathbb { R } ^ { n }$ s.t. $\mathsf { p r o x } _ { \gamma g } ( \hat { x } ) = \varnothing$ 

## Proposition (uniqueness of the proximal point)

If $g \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ , then $\mathsf { p r o x } _ { \gamma g } ( x )$ exists and it is unique for all $x \in \mathbb { R } ^ { n }$ 

“Proof”: For all $x \in \mathbb { R } ^ { n }$ , the function $h ( \cdot ; x )$ is $\scriptstyle { \frac { 1 } { \gamma } } - s \mathrm { t r o n g l y }$ (hence strictly) convex, hence it admits a unique minimiser. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/568f7a50f31c0e25bc16bebb4e91728bbb43d3fe8000f587be5ae6207d5d13df.jpg)


Thin black lines: level lines of $g .$ . Thick black lines: boundary of domain. Blue points: evaluation points are moved to the red points in the minimisation with an amount depending on $\gamma .$ . Note: points are moved to the minimum of the function. 

For $\gamma > 0$ and $x \in \mathbb { R } ^ { n }$ , let $z : = \mathsf { p r o x } _ { \gamma g } ( x )$ . We have: 

$$
\begin{array}{r c l} z := \operatorname{prox} _ {\gamma g} (x) & \Leftrightarrow & z = \underset {y \in \mathbb {R} ^ {n}} {\arg \min} g (y) + \frac {1}{2 \gamma} \| y - x \| ^ {2} \\ (\text { optimality}) & \Leftrightarrow & 0 \in \partial g (z) + \frac {1}{\gamma} (z - x) \\ (\text { rearranging }) & \Leftrightarrow & x \in z + \gamma \partial g (z) \\ (\text { using   operators }) & \Leftrightarrow & x \in (I d + \gamma \partial g) (z) \\ (\text { uniqueness }) & \Leftrightarrow & z = (I d + \gamma \partial g) ^ {- 1} (x) \end{array}
$$

For $\gamma > 0$ and $x \in \mathbb { R } ^ { n }$ , let $z : = \mathsf { p r o x } _ { \gamma g } ( x )$ . We have: 

$$
\begin{array}{r c l} z := \operatorname{prox} _ {\gamma g} (x) & \Leftrightarrow & z = \underset {y \in \mathbb {R} ^ {n}} {\arg \min} g (y) + \frac {1}{2 \gamma} \| y - x \| ^ {2} \\ (\text { optimality }) & \Leftrightarrow & 0 \in \partial g (z) + \frac {1}{\gamma} (z - x) \\ (\text { rearranging }) & \Leftrightarrow & x \in z + \gamma \partial g (z) \\ (\text { using   operators }) & \Leftrightarrow & x \in (I d + \gamma \partial g) (z) \\ (\text { uniqueness }) & \Leftrightarrow & z = (I d + \gamma \partial g) ^ {- 1} (x) \end{array}
$$

For those of you who are familiar with convex analysis. . . 

## Remark<sup>1</sup>

$z = \mathsf { p r o x } _ { \gamma g } ( x )$ is given by the resolvent of the maximal monotone operator $\gamma \partial g$ evaluated at x. 

## Firm non-expansiveness of the proximal operator

## Proposition (firm non-expansiveness)

Let $g \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ . Then: 

$$
\left(\forall x \in \mathbb {R} ^ {n}\right) \| \operatorname{prox} _ {g} (x) - \operatorname{prox} _ {g} (y) \| ^ {2} \leq \langle x - y, \operatorname{prox} _ {g} (x) - \operatorname{prox} _ {g} (y) \rangle
$$

Proof: There holds: 

$$
x - \operatorname{prox} _ {g} (x) \in \partial f (\operatorname{prox} _ {g} (x)), \quad y - \operatorname{prox} _ {g} (y) \in \partial f (\operatorname{prox} _ {g} (y)).
$$

By definition of subdiferential: 

$$
f (\operatorname{prox} _ {g} (y)) \geq f (\operatorname{prox} _ {g} (x)) + \langle x - \operatorname{prox} _ {g} (x), \operatorname{prox} _ {g} (y) - \operatorname{prox} _ {g} (x) \rangle ,
$$

and similarly inverting x and y. Summing: 

$$
\begin{array}{l} \underline {{f (\operatorname{prox} _ {g} (y)) + f (\operatorname{prox} _ {g} (x))}} \\ \geq \underline {{f (\operatorname{prox} _ {g} (y)) + f (\operatorname{prox} _ {g} (x)) + \langle y - f (\operatorname{prox} _ {g} (y)) - x + f (\operatorname{prox} _ {g} (x)) , f (\operatorname{prox} _ {g} (x)) - f (\operatorname{prox} _ {g} (y)) \rangle .}} \end{array}
$$

This implies non-expansiveness since: 

$$
\| \operatorname{prox} _ {g} (x) - \operatorname{prox} _ {g} (y) \| ^ {\frac {1}{2}} \leq \langle x - y, \operatorname{prox} _ {g} (x) - \operatorname{prox} _ {g} (y) \rangle \leq \| x - y \| \| \operatorname{prox} _ {g} (x) - \operatorname{prox} _ {g} (y) \|
$$

## Computation of proximal operators: indicator function

Example: Let $C \subset \mathbb { R } ^ { n }$ be a closed and convex set. Recall indicator function of C as: 

$$
\iota_ {C} (x) := \left\{ \begin{array}{l l} 0 & \quad \text { if } x \in C \\ + \infty & \quad \text { if } x \notin C \end{array} \right.
$$

The function $\iota _ { C } ( x )$ is proper, convex and l.s.c. 

## Computation of proximal operators: indicator function

Example: Let $C \subset \mathbb { R } ^ { n }$ be a closed and convex set. Recall indicator function of C as: 

$$
\iota_ {C} (x) := \left\{ \begin{array}{l l} 0 & \quad \text { if } x \in C \\ + \infty & \quad \text { if } x \notin C \end{array} \right.
$$

The function $\iota _ { C } ( x )$ is proper, convex and l.s.c. 

$$
\operatorname{prox} _ {\gamma \iota_ {C}} (x) = \underset {y \in \mathbb {R} ^ {n}} {\arg \min} \iota_ {C} (y) + \frac {1}{2 \gamma} \| y - x \| ^ {2} = \underset {y \in C} {\arg \min} \frac {1}{2 \gamma} \| y - x \| ^ {2} = P _ {C} (x),
$$

i.e. the projection of x onto C (the closest point $y \in C \ { \mathrm { t o } } \ x )$ 

The notion of prox for functions g more general than $\iota _ { C }$ is the reason why the prox operator is often referred to as generalised projection. 

## Computation of proximal operators: $1 5 - 1 = 1 5$ norm

Example: Let $g ( x ) = | x |$ and $\gamma > 0$ : 

$$
w = \operatorname{prox} _ {\gamma g} (x) = \underset {y \in \mathbb {R}} {\arg \min} | y | + \frac {1}{2 \gamma} (y - x) ^ {2}
$$

By optimality: 

$$
\gamma p + w - x = 0, \quad p \in \partial | w | \quad \Leftrightarrow \quad w = x - \gamma p, \quad p \in \partial | w |
$$

Recalling the expression of $\partial | \cdot |$ , one finds the definition of the soft-thresholding function 

$$
w = \operatorname{prox} _ {\gamma g} (x) = \left\{ \begin{array}{l l} x - \gamma & \text { if } \quad x > \gamma \\ x + \gamma & \text { if } \quad x <   - \gamma \\ 0 & \text { if } \quad - \gamma \leq x \leq \gamma \end{array} \right. = \mathcal {T} _ {\gamma} (x) := \operatorname{sign} (x) \max \{| x | - \gamma , 0 \}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/7a44c505e88e457d879794cc209a9628b8490a66da75762cfd11340ffb9a5361.jpg)


## A non-convex example: the $1 5 0$ pseudo-norm

Example: Take 

$$
g (x) = \lambda | x | _ {0} := \left\{ \begin{array}{l l} \lambda & \quad \text { if } x \neq 0 \\ 0 & \quad \text { if } x = 0 \end{array} \right.
$$

We want to compute: 

$$
\operatorname{prox} _ {\lambda | \cdot | _ {0}} (z) = \underset {y \in \mathbb {R}} {\arg \min} h (y) := \frac {1}{2 \lambda} (y - z) ^ {2} + | y | _ {0}
$$

• if $y = 0$ , then $\begin{array} { r } { h ( 0 ) = \frac { 1 } { 2 \lambda } z ^ { 2 } } \end{array}$ 

• if $y \neq 0$ , then the minimum is reached at $y ^ { * } = z ,$ , and $h ( y ^ { * } ) = 1$ 

By comparison we get: 

$$
h (0) = \frac {1}{2 \lambda} z ^ {2} \leq h (y ^ {*}) = 1 \Leftrightarrow z ^ {2} \leq 2 \lambda \Leftrightarrow - \sqrt {2 \lambda} <   z <   \sqrt {2 \lambda}
$$

Therefore: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/2055531547f2582d4ea9dac50e2476983209bc8024e653a59b0df4b809c22d34.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/eba751c345184cf6c497778c5b941f6846b0fc9020ae9ab951234b334882f440.jpg)


$$
\mathcal {H} _ {\sqrt {2 \lambda}} (z) := \operatorname{prox} _ {\lambda | \cdot | _ {0}} (z) = \left\{ \begin{array}{l l} 0 & \text {if} | z | <   \sqrt {2 \lambda} \\ z & \text {if} | z | > \sqrt {2 \lambda} \\ \{0, z \} & \text {if} | z | = \sqrt {2 \lambda} \end{array} \right.
$$

Soft VS. hard thresholding. 

## Computation of proximal points: properties

## Proposition (proximal operator of separable functions)

Let $g \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ be separable, i.e. $\begin{array} { r } { g ( x ) = \sum _ { i = 1 } ^ { n } g _ { i } ( x _ { i } ) } \end{array}$ for functions $ { \boldsymbol { g } } _ { i } \in  { \Gamma _ { 0 } } (  { \mathbb { R } } )$ Then for $\gamma > 0$ 

$$
\operatorname{prox} _ {\gamma g} (x) = \left(\operatorname{prox} _ {\gamma g _ {1}} (x _ {1}), \dots , \operatorname{prox} _ {\gamma g _ {n}} (x _ {n})\right),
$$

$g ( x ) = \lambda \| x \| _ { 1 }$ , then prox $\lambda \| \cdot \| _ { 1 } ( x ) = ( { \mathcal T } _ { \lambda } ( x _ { i } ) ) _ { i = 1 } ^ { n } = { \mathcal T } _ { \lambda } ( x )$ 

$g ( x ) = \lambda \| x \| _ { 0 }$ , then: 

$$
\operatorname{prox} _ {\lambda \| \cdot \| _ {0}} = \mathcal {H} _ {\sqrt {2 \lambda}} (x _ {1}) \times \dots \times \mathcal {H} _ {\sqrt {2 \lambda}} (x _ {n}).
$$

## Proposition (proximal operators of rescaled and perturbed functions)

Let $g \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ and $\lambda \neq 0$ . Define $h _ { 1 } ( x ) : = \lambda g ( x / \lambda )$ . Then, for $\gamma \in \mathbb { R } _ { + + }$ : 

$$
\operatorname{prox} _ {\gamma h _ {1}} (x) = \lambda \operatorname{prox} _ {\frac {\gamma}{\lambda} g} (x / \lambda).
$$

Let $\begin{array} { r } { h _ { 2 } ( x ) : = \alpha g ( x ) + \frac { \beta } { 2 } \| x \| ^ { 2 } } \end{array}$ , for $\alpha , \beta \in \mathbb { R } _ { + + }$ . Then, for $\gamma \in \mathbb { R } _ { + + }$ 

$$
\operatorname{prox} _ {\gamma h _ {2}} (x) = \operatorname{prox} _ {\frac {\alpha \gamma}{1 + \beta \gamma} g} \left(\frac {x}{1 + \beta \gamma}\right).
$$

Let $h _ { 3 } ( x ) : = g ( W x )$ where $W \in \mathbb { R } ^ { m \times n }$ is orthogonal, $W ^ { T } W = I d$ . Then, for $\gamma \in \mathbb { R } _ { + + }$ 

$$
\operatorname{prox} _ {\gamma h _ {3}} (x) = W ^ {T} \operatorname{prox} _ {\gamma g} (W x).
$$

## Computation of proximal points in general cases

## Important remark

Having formulas for closed-form expressions of proximal points is very handy. Otherwise, a minimisation problem needs to be solved! 

However, general regularisers do not have this property! 

For more examples of easily-proximable function, see, e.g.: 

Beck, First-order methods in optimization 2006 (Chapter 6): many examples of proximal operators 

• Parikh, Boyd, Proximal algorithms, 2013 

• http://proximity-operator.net/index.html 

In the lab class, we will make use of easily proximable (aka simple) functions. For non-proximable functions (e.g. TV) alternative strategies/algorithms should be found: 

• Fenchel duality 

• Smoothing 

• Other algorithms (e.g., ADMM: Alessandro Lanza’s computational imaging lab) 

Projected gradient descent 

## Towards forward-bacwkard splitting: projected gradient descent

$$
f \in \Gamma_ {0} (\mathbb {R} ^ {n})
$$

$$
C \in \mathbb {R} ^ {n}
$$

$$
\underset {x \in C} {\arg \min} f (x) = \underset {x \in \mathbb {R} ^ {n}} {\arg \min} f (x) + \iota_ {C} (x)
$$

Algorithm: Projected Gradient Descent (PGD) algorithm

Input: $\tau\in(0,\frac{1}{L}],x^{0}\in\mathbb{R}^{n}$ .

for $k\geq0$ do $x_{k+\frac{1}{2}}=x_{k}-\tau\nabla f(x_{k})$ $x_{k+1}=P_{C}(x_{k+\frac{1}{2}})=\underset{y\in C}{\arg\min}\frac{1}{2}\|y-x_{k+\frac{1}{2}}\|^{2}$ $=\underset{y\in\mathbb{R}^{n}}{\arg\min}\iota_{C}(y)+\frac{1}{2}\|y-x_{k+\frac{1}{2}}\|^{2}=\operatorname{prox}_{\iota_{C}}(x_{k+\frac{1}{2}})$ end for 

• First: gradient step, next projection step 

- Starting point for generalisation to more general convex, non-differentiable functions $g \ldots$ 

## Towards forward-backward splitting: explicit/implict GD

Let $f , g \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ and let f be smooth. Want to solve: 

$$
\underset {x \in \mathbb {R} ^ {n}} {\arg \min} f (x) + g (x)
$$

Consider for $x _ { 0 } \in \mathbb { R } ^ { n }$ , suitable $\tau > 0$ and $k \geq 0$ , the following iterative scheme: 

$$
\begin{array}{r c l} x _ {k + 1} \in x _ {k} - \tau \nabla f (x _ {k}) - \tau \partial g (x _ {k + 1}) & \Leftrightarrow & (I d + \tau \partial g (\cdot)) (x _ {k + 1}) \in x _ {k} - \tau \nabla f (x _ {k}) \\ x _ {k + 1} \in (I d + \tau \partial g (\cdot)) ^ {- 1} (x _ {k} - \tau \nabla f (x _ {k})) & \Leftrightarrow & x _ {k + 1} = \text {prox} _ {\tau g} (x _ {k} - \tau \nabla f (x _ {k})) \end{array}
$$

• Explicit GD on the smooth part f 

• Implicit GD on the non-smooth part g 

# The proximal gradient algorithm

## Framework: recap

$$
\underset {x \in \mathbb {R} ^ {n}} {\arg \min} \left\{F (x) := f (x) + g (x) \right\},
$$

$f \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ is diferentiable with L-Lipschitz continuous gradient 

$$
\exists L > 0, \quad (\forall x, y \in \mathbb {R} ^ {n}) \quad \| \nabla f (x) - \nabla f (y) \| \leq L \| x - y \|
$$

$g \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ is typically non-smooth but (assume) easily-proximable! 

Examples: $\begin{array} { r } { g ( x ) = \iota _ { C } ( x ) , g ( x ) = \| x \| _ { 1 } , g ( x ) = \| x \| _ { 1 } + \iota _ { \geq 0 } ( x ) , g ( x ) = \| x \| _ { 1 } + \frac { \lambda } { 2 } \| x \| _ { 2 } ^ { 2 } , } \end{array}$ $g ( x ) = \| W x \| _ { 1 }$ with W orthogonal. . . 

Algorithm: Forward-backward splitting (FB/FBS) algorithm $^{2}$ Input: $x_{0} \in R^{n}, \tau \in (0, \frac{1}{L}]$ .

for $k \geq 0$ do $x_{k+1} = \text{prox}_{\tau g}(x_{k} - \tau \nabla f(x_{k}))$ end for 

• Step-size τ: still depending on the inverse of L, as for GD. If L is unknown/dificult to compute, backtracking strategies can be used, $\tau = \tau _ { k }$ with suitable update rules. 

• If g is easily proximable: no inner minimisation. Otherwise: need to solve a nested minimisation problem up to some accuracy (inexact algorithms). 

• Computational cost/complexity: evaluation of ∇f may be costly (matrix/vector products), number of iterations before convergence depends on τ. 

* Too small τ: unnecessary too many iterations 

* Too big τ: risk of moving to a point z for which $F ( z ) > F ( x _ { k } )$ 

• If $g \equiv 0$ : smooth-optimisation problem. FBS reduces to GD. 

• If $g ( x ) = \iota _ { C } ( x )$ for closed and convex $C \to \mathsf { P G D }$ 

• If $g ( x ) = \lambda \| W x \| _ { 1 }$ for $\lambda > 0$ and orthogonal $W \in \mathbb { R } ^ { N \times n }$ (Wavelet basis. . . ) 

$$
\min _ {x \in \mathbb {R} ^ {n}} f (x) + \lambda \| W x \| _ {1},
$$

then the algorithm takes the structure of the Iterative Soft-Thresholding Algorithm (ISTA) 

## Iterative Soft Thresholding Algorithm (ISTA)<sup>3</sup>

The FB iteration takes the form: 

$$
x _ {k + 1} = W ^ {T} \mathcal {T} _ {\tau \lambda} (W x _ {k} - \tau W \nabla f (x _ {k})),
$$

where $\tau _ { \tau \lambda } ( \cdot )$ is the soft-thresholding operator: 

$$
\mathcal {T} _ {\tau \lambda} (z) = (\mathcal {T} _ {\tau \lambda} (z _ {j})) _ {j = 1, \dots , n} = \Big (\left[ | z _ {j} | - \lambda \tau \right] _ {+} \operatorname{sign} (z _ {j}) \Big) _ {j = 1, \dots , n}
$$

The proximal gradient algorithm 

Convergence properties 

## Theorem (convergence of $\mathsf { F B } ) ^ { 4 }$

Let $\left( \boldsymbol { x } _ { k } \right) _ { k }$ the sequence of iterates generated by FB. Then, if $\tau \in ( 0 , 1 / L ]$ , there holds: 

$$
F (x _ {k}) - F (x ^ {*}) \leq \frac {\| x ^ {0} - x ^ {*} \| ^ {2}}{2 \tau k}.
$$

If, additionally, $f$ or $\boldsymbol { g }$ are strongly convex with parameters $\mu _ { f } , \mu _ { g } > 0$ with $\mu : = \mu _ { f } + \mu _ { g }$ , then: 

$$
F (x _ {k}) - F (x ^ {*}) + \frac {1 + \tau \mu_ {g}}{2 \tau} \| x _ {k} - x ^ {*} \| ^ {2} \leq \omega^ {k} \frac {(1 + \tau \mu_ {g}) \| x ^ {0} - x ^ {*} \| ^ {2}}{2 \tau},
$$

with $\begin{array} { r } { \omega = \frac { 1 - \tau \mu _ { f } } { 1 + \tau \mu _ { g } } < 1 . } \end{array}$ . 

Same $O ( 1 / k ) / O ( \omega ^ { k } )$ rates as for GD! Alternative way of seeing this: for $\epsilon > 0$ , the iterates to get an -solution, i.e. $x _ { k }$ s.t.: 

$$
F (x _ {k}) - F (x ^ {*}) \leq \epsilon
$$

is $k \geq \lceil C / \epsilon \rceil$ and $k \geq \lceil C \log ( 1 / \epsilon ) \rceil$ 

## Towards the proof: a generalised descent lemma

For all $k \geq$ and $\tau \in ( 0 , 1 / L ]$ let: 

$$
x _ {k + 1} = T _ {\tau} (x _ {k}) := \operatorname{prox} _ {\tau g} (x _ {k} - \tau \nabla f (x _ {k}))
$$

## Generalised descent lemma

Let $\mu : = \mu _ { f } + \mu _ { g } \geq 0$ . Then, for all $x \in \mathbb { R } ^ { n }$ , there holds: 

$$
F (x _ {k + 1}) + (1 + \tau \mu_ {g}) \frac {\| x - x _ {k + 1} \| ^ {2}}{2 \tau} \leq F (x) + (1 - \tau \mu_ {f}) \frac {\| x - x _ {k} \| ^ {2}}{2 \tau}
$$

Proof: By definition $x _ { k + 1 }$ solves: 

$$
x _ {k + 1} = \underset {x} {\arg \min} g (x) + f (x _ {k}) + \langle \nabla f (x _ {k}), x - x _ {k} \rangle + \frac {\| x - x _ {k} \| ^ {2}}{2 \tau}
$$

By strong convexity there holds: 

$$
\overbrace {f (x) + g (x)} ^ {F (x)} + (1 - \tau \mu_ {f}) \frac {\| x - x _ {k} \| ^ {2}}{2 \tau} \overset {\text {s.c. of} f} {\geq} f (x _ {k}) + \langle \nabla f (x _ {k}), x - x _ {k} \rangle + \frac {\| x - x _ {k} \| ^ {2}}{2 \tau} + g (x)
$$

minimality and $\mu _ { g } + { \frac { 1 } { \tau } } { \mathsf { s . c } }$ 

$$
\widehat {\geq} \quad f (x _ {k}) + g (x _ {k + 1}) + \langle \nabla f (x _ {k}), x _ {k + 1} - x _ {k} \rangle + \frac {\| x _ {k + 1} - x _ {k} \| ^ {2}}{2 \tau} + (1 + \tau \mu_ {g}) \frac {\| x - x _ {k + 1} \| ^ {2}}{2 \tau}
$$

≥ . . . 

Since f is L-Lipschitz there holds: $\begin{array} { r } { \begin{array} { r } { \boldsymbol { f } ( \boldsymbol { x } _ { k } ) + \langle \nabla f ( \boldsymbol { x } _ { k } ) , \boldsymbol { x } _ { k + 1 } - \boldsymbol { x } _ { k } \rangle \geq f ( \boldsymbol { x } _ { k + 1 } ) - \frac { L } { 2 } \| \boldsymbol { x } _ { k + 1 } - \boldsymbol { x } _ { k } \| ^ { 2 } } \end{array} } \end{array}$ , hence: 

$$
\ldots \geq F (x _ {k + 1}) + (1 + \tau \mu_ {g}) \frac {\| x - x _ {k + 1} \| ^ {2}}{2 \tau} + \underbrace {\left(\frac {1}{2 \tau} - \frac {L}{2}\right)} _ {\geq 0} \| x _ {k + 1} - x _ {k} \| ^ {2}.
$$

Proof: Apply the generalised descent lemma for $x = x _ { k }$ , get: 

$$
F (x _ {k + 1}) \leq F (x _ {k + 1}) + (1 + \tau \mu_ {g}) \frac {\| x _ {k} - x _ {k + 1} \| ^ {2}}{2 \tau} \leq F (x _ {k}),
$$

so $F$ is decreasing. Define $\begin{array} { r } { \omega : = \frac { 1 - \tau \mu _ { f } } { 1 + \tau \mu _ { g } } \leq 1 } \end{array}$ , apply again the generalised descent lemma, which for $k = 0 , \ldots , K - 1$ can be multiplied by $\omega ^ { - k - 1 }$ and summed: 

$$
\sum_ {k = 1} ^ {K} \omega^ {- K} \left(F (x _ {k}) - F (x)\right) + \sum_ {k = 1} ^ {K} \omega^ {- k} \frac {1 + \tau \mu_ {g}}{2 \tau} \| x - x _ {k} \| ^ {2} \leq \sum_ {k = 0} ^ {K - 1} \omega^ {- k - 1} \frac {1 - \tau \mu_ {f}}{2 \tau} \| x - x _ {k} \| ^ {2}.
$$

After cancellations, and using that $F ( x _ { k } ) \geq F ( x _ { K } )$ , for all $k = 0 , \ldots , K$ , we get: 

$$
\omega^ {- K} \left(\sum_ {k = 0} ^ {K - 1} \omega^ {k}\right) (F (x _ {K}) - F (x)) + \omega^ {- K} \frac {1 + \tau \mu_ {g}}{2 \tau} \| x - x _ {K} \| ^ {2} \leq \frac {1 + \tau \mu_ {g}}{2 \tau} \| x - x _ {0} \| ^ {2}.
$$

$\mu = 0 , \omega = 1$ : we deduce the result observing tha $\begin{array} { r } { \sum _ { k = 0 } ^ { K - 1 } \omega ^ { k } = \sum _ { k = 0 } ^ { K - 1 } 1 = K } \end{array}$ 

$\mu > 0 , \omega < 1$ : we deduce the linear rate by multiplying by $\omega ^ { K }$ and observing that $\begin{array} { r } { \sum _ { k = 0 } ^ { K - 1 } \omega ^ { k } = \frac { 1 - \omega ^ { K } } { 1 - \omega } \geq 1 } \end{array}$ 

## Analysis of the forward-backward algorithm: convergence of the sequence

We focus on the simple convex case (i.e. $\mu = 0 )$ . For $\mu > 0$ this holds a fortiori. 

## Proposition (Fej´er monotonicity)

Let $\left( \boldsymbol { x } _ { k } \right)$ be the seguence generated by the FB algorithm with a constant stepsize $\tau \in ( 0 , 1 / L ]$ . Then, for any $x ^ { \ast } \in \mathsf { a r g }$ min F , there holds: 

$$
\| x _ {k + 1} - x ^ {*} \| \leq \| x _ {k} - x ^ {*} \|.
$$

## Lemma (convergence under Fej´er monotonicity)

Let $( x _ { k } ) \subset \mathbb { R } ^ { n }$ be a sequence and let: $D : = \{ \tilde { x } : \tilde { x }$ is a limiting pont of $\left( x _ { k } \right) \}$ } . Let S s.t. $D \subseteq S . \ \mathsf { l f } \ \left( x _ { k } \right)$ is Fej´er monotone for all elements $x ^ { * } \in S$ , then it converges to a point in $D .$ 

## Analysis of the forward-backward algorithm: convergence of the sequence

We focus on the simple convex case $( \mathfrak { i . e . } ~ \mu = 0 )$ . For $\mu > 0$ this holds a fortiori. 

## Proposition (Fej´er monotonicity)

Let $\left( \boldsymbol { x } _ { k } \right)$ be the seguence generated by the FB algorithm with a constant stepsize $\tau \in ( 0 , 1 / L ]$ . Then, for any $x ^ { \ast } \in \mathsf { a r g }$ min F , there holds: 

$$
\| x _ {k + 1} - x ^ {*} \| \leq \| x _ {k} - x ^ {*} \|.
$$

## Lemma (convergence under Fej´er monotonicity)

Let $( x _ { k } ) \subset \mathbb { R } ^ { n }$ be a sequence and let: $D : = \{ \tilde { x } : \tilde { x }$ is a limiting pont of $\left( x _ { k } \right) \}$ } . Let S s.t. $D \subseteq S . \ \mathsf { l f } \ \left( x _ { k } \right)$ is Fej´er monotone for all elements $x ^ { * } \in S$ , then it converges to a point in D. 

## Theorem (convergence of the iterates of FB)

Let $\left( \boldsymbol { x } _ { k } \right)$ be the sequence generated by the FB algorithm with a constant step-size $\tau \in ( 0 , 1 / L ]$ . Then, $x _ { k } \to x ^ { * }$ , where $x ^ { * } \in \mathsf { a r g } \mathsf { m i n } F$ 

Proof: Let ˜x be a limit point of $\left( x _ { k } \right)$ . Then, there exists a subsequence $( x _ { k _ { j } } )$ ) such that $x _ { k _ { j } } \to \tilde { x } .$ . Then, since 

$$
F (x _ {k _ {j}}) - F (x ^ {*}) \rightarrow 0, \quad \text { for } j \rightarrow + \infty .
$$

and F is I.s.c.. we deduce 

$$
F (\tilde {x}) \leq \liminf _ {j \to + \infty} F (x _ {k _ {j}}) = F (x ^ {*}).
$$

By minimality, $\tilde { x } \in \mathsf { a r g }$ min F. By now defining $S : = a r g m i n ~ F$ and applying the Lemma the thesis follows since all limiting points are elements of S. 

Acceleration strategies 

Acceleration strategies 

FISTA 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/eb1002255e41b15f79709d01d7afcabe787da52d365e7f5c60b462cd39a8a11d.jpg)


## Accelerated proximal gradient algorithm

Idea: add inertia to “shift” the sequence of iterates. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/2eb34cf31b82fd3f8439c08b75f24d4d67464d525c7f4447dbf62b8668b8a22f.jpg)


Algorithm: Fast Iterative Soft-Thresholding Algorithm $( \mathsf { F l S T A } ) ^ { 5 }$ 

Input: $x_0 = y_0 \in \mathbb{R}^n$ , $\tau \in (0, \frac{1}{L}]$ , $t_0 = 1$ .  
for $k \geq 0$ do $x_{k+1} = \text{prox}_{\tau g}(y_k - \tau \nabla f(y_k))$ $t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}$ $y_{k+1} = x_{k+1} + \frac{t_k - 1}{t_{k+1}}(x_{k+1} - x_k)$ end for 

## Properties of the parameter sequence

## Proposition

Let $\left\{ t _ { k } \right\}$ be the sequence defined by $t _ { 0 } = 1$ and $\begin{array} { r } { t _ { k + 1 } = \frac { 1 + \sqrt { 1 + 4 t _ { k } ^ { 2 } } } { 2 } } \end{array}$ for $k \geq 0$ Then: 

$$
t _ {k} \geq \frac {k + 2}{3} \quad \forall k \geq 0.
$$

Proof: By induction. For $k = 0 ;$ :, obviously there holds: $\begin{array} { r } { t _ { 0 } = 1 \geq \frac { 0 + 2 } { 2 } = 1 } \end{array}$ . Suppose the claim holds for some $k > 0$ . Using the recursion: 

$$
t _ {k + 1} = \frac {1 + \sqrt {1 + 4 t _ {k} ^ {2}}}{2} \geq \frac {1 + \sqrt {1 + (k + 2) ^ {2}}}{2} \geq \frac {1 + \sqrt {(k + 2) ^ {2}}}{2} = \frac {k + 3}{2}.
$$

Alternative choices: The sequence $\left\{ t _ { k } \right\}$ can alternatively be chosen so as to satisfy the following two properties holding for all $k \geq 0$ 

$$
t _ {k} \geq \frac {k + 2}{2}
$$

$$
\bullet \quad t _ {k + 1} ^ {2} - t _ {k + 1} \leq t _ {k} ^ {2}.
$$

For instance, the choice $\begin{array} { r } { t _ { k } = \frac { k + 2 } { 2 } } \end{array}$ satisfies both properties (Chambolle, Dossal, ’15). 

## Theorem (Accelerated convergence of FISTA)

Let $\left( \boldsymbol { x } _ { k } \right)$ the sequence of iterates generated by FISTA with $\tau \in ( 0 , 1 / L ]$ Then, for any $x ^ { \ast } \in$ arg min $F _ { \ast }$ , there holds: 

$$
F (x _ {k}) - F (x ^ {*}) \leq \frac {2 \| x _ {0} - x ^ {*} \| ^ {2}}{\tau (k + 1) ^ {2}}
$$

Proof: you will see this in the exercise class tomorrow with $\tau = 1 / L$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/f5c77c4da010fef0f877a61af51998ea1ead21901c57a1423687dc4a132b75f7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/59eebd2f936c1e15a0c4fa554fdd85d064e4205e0a4aab9ec738366766624727.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/3a8f1be4069e9776edc486b1bd554caab67a2cd217cb2a4a8046da5ac8d249c3.jpg)


Accuracy viewpoint: w.r.t. to the vanilla FB algorithm, an -accurate solution, i.e.: 

$$
F (x _ {k}) - F (x ^ {*}) \leq \epsilon
$$

is obtained for $k \geq \lceil C / \sqrt { \epsilon } - 1 \rceil$ 

Acceleration strategies 

Strongly convex FISTA 

Assume now that f is strongly convex with $\mu _ { f } > 0$ . Consider the algorithm: 

Assume now that $F$ is strongly convex with $\mu_f > 0$ . Consider the algorithm.

Algorithm: Strongly convex FISTA - V-FISTA $^6$ Input: $x_0 = y_0 \in \mathbb{R}^n$ , $\tau = \frac{1}{L}$ , and $\kappa := \frac{L}{\mu_f}$ .

for $k \geq 0$ do $x_{k+1} = \text{prox}_{\frac{1}{L} g}(y_k - \frac{1}{L} \nabla f(y_k))$ $y_{k+1} = x_{k+1} + \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)(x_{k+1} - x_k)$ end for 

Note: constant inertial parameter defined in terms of $\kappa \geq 1$ 

. . . Both L and $\mu _ { f }$ are required (dificult to estimate in practice)! 

## Convergence rates for strongly convex FISTA

## Theorem (convergence of strongly convex FISTA<sup>7</sup>)

Let $\left( \boldsymbol { x } _ { k } \right)$ be the sequence of iterates generated by the strongly convex variant of the FISTA algorithm. Then, there holds: 

$$
F (x _ {k}) - F (x ^ {*}) \leq \left(1 - \frac {1}{\sqrt {\kappa}}\right) ^ {k} \left(F (x _ {0}) - F (x ^ {*}) + \frac {\mu_ {f}}{2} \| x _ {0} - x ^ {*} \| ^ {2}\right),
$$

Proof: you will see this in the exercise classes. 



• In Chambolle, Pock, ’16, Calatroni, Chambolle, ’19, Rebegoldi, Calatroni ’22: strongly convex variant of FISTA allowing strong convexity both in f and in g (better in g !) 



• In Aujol, Dossal, Labarriere, Rondebierre, ’21: FISTA algorithm under PL condition for f with an automatic estimate of the strong convexity parameter µ<sub>f</sub> 

## The FISTA club

• Convergence of iterates: OK for FB (based on monotonicity arguments), proved for FISTA in Chambolle, Dossal, ’15; 

• Monotone variants: MFISTA (Beck, Teboulle, ’09) 

• Non-Euclidean, inexact variants:, Schmidt, Roux and Bach, ’11, Villa, Salzo, Baldassarre, Verri, ’13, Bonettini, Rebegoldi, Ruggiero, ’19 

• Strongly convex, inexact and scaled: SAGE-FISTA (Rebegoldi, Calatroni, ’22) 

• Adaptive backtracking for estimating τ ‘on-the-fly’: Scheinberg, Goldfarb, Bai, ’14, Calatroni, Chambolle, ’19, Florea, Vorobyov, ’20 

• Restarting schemes: heuristic (O’Donoghue, Cand`es, ’15), rigorous (Alamo et al., ’19, Aujol, Dossal, Labarriere, Rondepierre et al., ’21) 

• ODE interpretation: interpretation as discretised dynamical systems (with diferent inertial/friction/damping terms) Su, Boyd, Cand`es, ’14, lot of works by Attouch, Cabot, Chbani, Peypouquet 

• Learned versions: LISTA (Gregor, Le Cunn, 2010) 

• Faster-FISTA, Adaptive FISTA. . . 

We discussed the use of proximal-based algorithms for convex structured (smooth+non-smooth) optimisation problems in the form: 

$$
\underset {x} {\arg \min} f (x) + g (x)
$$

• We revised basic tools of convex analysis for generalising derivatives to non-smooth functions 

• We defined, characterised and looked at some fundamental properties of the proximal operator 

• We defined the forward-backward (aka proximal gradient method) generalising the GD algorithm to the structured case and show a general convergence result for strongly convex functions 

• We discussed acceleration strategies `a la Nesterov: FISTA and its strongly covex variants 

Inexact algorithms 

$$
p = \operatorname{prox} _ {g} (a) \Leftrightarrow p = \operatorname{argmin} _ {x} \left\{\phi (x) := g (x) + \frac {1}{2} \| x - a \| ^ {2} \right\} \Leftrightarrow p - a \in \partial g (p)
$$

$$
p = \operatorname{prox} _ {g} (a) \Leftrightarrow p = \operatorname{argmin} _ {x} \left\{\phi (x) := g (x) + \frac {1}{2} \| x - a \| ^ {2} \right\} \Leftrightarrow p - a \in \partial g (p)
$$

There are various ways to relax this to incorporate errors<sup>8</sup> 

- Type 1 errors : $\hat { p } \approx _ { 1 } ^ { \varepsilon } p$ if 

$$
\hat {p} \in \varepsilon - \operatorname{argmin} _ {x} \phi (x) := \left\{x ^ {\prime} \in \mathbb {R} ^ {n}: \phi (x ^ {\prime}) \leq \inf \phi (x) + \varepsilon \right\}
$$

$$
p = \operatorname{prox} _ {g} (a) \Leftrightarrow p = \operatorname{argmin} _ {x} \left\{\phi (x) := g (x) + \frac {1}{2} \| x - a \| ^ {2} \right\} \Leftrightarrow p - a \in \partial g (p)
$$

There are various ways to relax this to incorporate errors<sup>8</sup> 

- Type 1 errors: $\hat { p } \approx _ { 1 } ^ { \varepsilon } p$ if 

$$
\hat {p} \in \varepsilon - \operatorname{argmin} _ {x} \phi (x) := \left\{x ^ {\prime} \in \mathbb {R} ^ {n}: \phi (x ^ {\prime}) \leq \inf \phi (x) + \varepsilon \right\}
$$

- Type 2 errors: $\hat { p } \approx _ { 2 } ^ { \varepsilon } p$ if 

$$
\hat {p} - a \in \partial_ {\varepsilon^ {2}} g (\hat {p}) = \left\{u \in \mathbb {R} ^ {n}: g (x ^ {\prime}) \geq g (\hat {p}) + u ^ {T} (x ^ {\prime} - \hat {p}) - \varepsilon^ {2} \forall x ^ {\prime} \right\}
$$

$$
p = \operatorname{prox} _ {g} (a) \Leftrightarrow p = \operatorname{argmin} _ {x} \left\{\phi (x) := g (x) + \frac {1}{2} \| x - a \| ^ {2} \right\} \Leftrightarrow p - a \in \partial g (p)
$$

There are various ways to relax this to incorporate errors<sup>8</sup> 

- Type 1 errors: $\hat { p } \approx _ { 1 } ^ { \varepsilon } p$ if 

$$
\hat {p} \in \varepsilon - \operatorname{argmin} _ {x} \phi (x) := \left\{x ^ {\prime} \in \mathbb {R} ^ {n}: \phi (x ^ {\prime}) \leq \inf \phi (x) + \varepsilon \right\}
$$

- Type 2 errors : $\hat { p } \approx _ { 2 } ^ { \varepsilon } p$ if 

$$
\hat {p} - a \in \partial_ {\varepsilon^ {2}} g (\hat {p}) = \left\{u \in \mathbb {R} ^ {n}: g (x ^ {\prime}) \geq g (\hat {p}) + u ^ {T} (x ^ {\prime} - \hat {p}) - \varepsilon^ {2} \forall x ^ {\prime} \right\}
$$

- Type 3 errors : $\begin{array} { r } { \hat { p } \approx _ { 3 } ^ { \varepsilon } p \mathrm { ~ i f ~ } \hat { p } = \mathsf { p r o x } _ { g } ( a + e ) , \| e \| \le \varepsilon . } \end{array}$ 

$$
p = \operatorname{prox} _ {g} (a) \Leftrightarrow p = \operatorname{argmin} _ {x} \left\{\phi (x) := g (x) + \frac {1}{2} \| x - a \| ^ {2} \right\} \Leftrightarrow p - a \in \partial g (p)
$$

There are various ways to relax this to incorporate errors<sup>8</sup> 

- Type 1 errors: $\hat { p } \approx _ { 1 } ^ { \varepsilon } p$ if 

$$
\hat {p} \in \varepsilon - \operatorname{argmin} _ {x} \phi (x) := \left\{x ^ {\prime} \in \mathbb {R} ^ {n}: \phi (x ^ {\prime}) \leq \inf \phi (x) + \varepsilon \right\}
$$

- Type 2 errors : $\hat { p } \approx _ { 2 } ^ { \varepsilon } p$ if 

$$
\hat {p} - a \in \partial_ {\varepsilon^ {2}} g (\hat {p}) = \left\{u \in \mathbb {R} ^ {n}: g (x ^ {\prime}) \geq g (\hat {p}) + u ^ {T} (x ^ {\prime} - \hat {p}) - \varepsilon^ {2} \forall x ^ {\prime} \right\}
$$

- Type 3 errors: $\begin{array} { r } { \hat { p } \approx _ { 3 } ^ { \varepsilon } p \mathrm { ~ i f ~ } \hat { p } = \mathsf { p r o x } _ { g } ( a + e ) , \| e \| \le \varepsilon . } \end{array}$ 

## Theorem (convergence of inexact FISTA)

For $\tau \leq 1 / L , \mathsf { i f } \varepsilon _ { k } = O ( 1 / k ^ { q } )$ with $q > 3 / 2$ , then the sequence $\left( x _ { k } \right)$ of the accelerated inexac FB algorithm satisfies: 

$$
F (x _ {k}) - F (x ^ {*}) = O \left(\frac {1}{k ^ {2}}\right)
$$

## Extensions

Backtracking strategies for FISTA 

## FISTA with monotone backtracking<sup>9</sup>

For f convex and diferentiable, define the Bregman “distance”” 

$$
D _ {f} (x, y) := f (x) - f (y) - \langle \nabla f (y), x - y \rangle \geq 0, \quad \forall x, y \in \mathbb {R} ^ {n}
$$

Popular for mirror descent algorithms and regularisation of inverse problems (Burger, ’16). 

Algorithm: FISTA with non-decreasing backtracking

Input: $x_{0} = y_{0} \in R^{n}, \tau_{0} > 0, t_{0} = 1, \rho \in (0,1)$ .
for $k \geq 0$ do
    for $i = 0, 1, \ldots$ repeat $\tau_{k+1} = \rho^{j} \tau_{k}$ $x_{k+1} = \text{prox}_{\tau_{k+1} g}(y_{k} - \tau_{k+1} \nabla f(y_{k}))$ $t_{k+1} = \frac{1 + \sqrt{1 + 4t_{k}^{2}}}{2}$ $y_{k+1} = x_{k+1} + \frac{t_{k} - 1}{t_{k+1}}(x_{k+1} - x_{k})$ until $D_{f}(x^{k+1}, y^{k+1}) \leq \|x^{k+1} - y^{k+1}\|^{2}/2\tau_{k+1}$ end for 

## Convergence guarantee for FISTA with non-adaptive backtracking

## Theorem (FISTA with non-adaptive backtracking)

Let $\left( x _ { k } \right)$ the sequence of iterates generated by FISTA with non-adaptive backtracking. Then, for any $x ^ { \ast } \in$ arg min $F _ { \ast }$ , there holds: 

$$
F (x _ {k}) - F (x ^ {*}) \leq \frac {2 \| x _ {0} - x ^ {*} \| ^ {2}}{\tau \rho (k + 1) ^ {2}}
$$

• Basically the same rate as before, just depending on $\rho \in ( 0 , 1 )$ 

• Idea: start in an optimistic way $\tau _ { 0 } \gg 1$ . If at any step $k \geq 1$ the step-size is too big, it will be decreased up to guarantee decay 

## Non-monotone FISTA backtracking

Algorithm: FISTA with adaptive backtracking

Input: $x_{0} = y_{0} \in R^{n}$ , $\tau_{0} > 0$ , $t_{0} = 1$ , $\rho \in (0,1)$ , $\delta \in (0,1)$ .

for $k \geq 0$ do $\tau_{k+1}^{0} = \frac{\tau_{k}}{\delta};$ (*)

for $i = 0, 1, \ldots$ repeat $\tau_{k+1} = \rho^{i} \tau_{k+1}^{0}$ $x_{k+1} = \text{prox}_{\tau_{k+1} g}(y_{k} - \tau_{k+1} \nabla f(y_{k}))$ $t_{k+1} = \frac{1 + \sqrt{1 + 4t_{k}^{2}}}{2}$ $y_{k+1} = x_{k+1} + \frac{t_{k} - 1}{t_{k+1}}(x_{k+1} - x_{k})$ until $D_{f}(x^{k+1}, y^{k+1}) \leq \|x^{k+1} - y^{k+1}\|^{2}/2\tau_{k+1}$ .

end for 

• Only diference: tentative step where you try to increase the previous step-size. 

• Practically, you may even add a max number of backtracking iterations $i _ { \mathrm { m a x } } \approx 1 0$ 

## Convergence guarantee for FISTA with adaptive backtracking)

## Theorem (FISTA with adaptive backtracking<sup>10</sup>)

Let $\left( \boldsymbol { x } _ { k } \right)$ the sequence of iterates generated by FISTA with non-adaptive backtracking. Then, for any $x ^ { \ast } \in$ arg min $F$ , there holds: 

$$
F (x _ {k}) - F (x ^ {*}) \leq \frac {2 \bar {L} _ {k}}{k ^ {2}} \| x ^ {0} - x ^ {*} \| ^ {2} \leq \frac {2 L}{\rho k ^ {2}} \| x ^ {0} - x ^ {*} \| ^ {2}
$$

where $\begin{array} { r } { \sqrt { \bar { L } _ { k } } : = \frac { 1 } { \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \frac { 1 } { \sqrt { L _ { i } } } } , L _ { i } : = 1 / \tau _ { i } . } \end{array}$ 

From standard harmonic/arithmetic mean inequalities: 

$$
\sqrt {\bar {L} _ {k}} \leq \frac {1}{k} \sum_ {i = 1} ^ {k} \sqrt {L _ {i}} \leq \sqrt {\frac {1}{k} \sum_ {i = 1} ^ {k} L _ {i}} \leq \sqrt {\frac {L}{\rho}}
$$

• “Local” estimates: you don’t need the dependence on $L _ { f }$ in final rates (which is in principle unknown), you have acceleration depending on harmonic mean 

• Extensions in Rebegoldi, Calatroni’ 22 to inexact proximal algorithms, with scaling. 

• For step-size selection strategies in non-convex problems see Ochs, Chen, Brox, Pock, ’14 

## Backtracking performance

In Calatroni, Chambolle, ’19 we considered a variation for strongly convex functions. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/3542d91d73ab488b604dde168bd299e2a7565eb156e99a85ca34da284f196c90.jpg)


## Backtracking performance

In Calatroni, Chambolle, ’19 we considered a variation for strongly convex functions. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/b22d6202-d025-4fb0-9d62-82e5bc6eba5a/140dfe1631c8d5829f5f37a3b1f8687538cccc160f18da363281f0f849234626.jpg)


Non-convex algorithms 

Let $f$ be $\textsf { a } C ^ { 2 }$ , L-smooth function which is coercive and bounded from below. Using Taylor expansion with integral form of remainder we have that: 

$$
\begin{array}{l} f (x _ {k + 1}) = f (x _ {k} - \tau \nabla f (x _ {k})) \\ \qquad = f (x _ {k}) - \tau \langle \nabla f (x _ {k}), \nabla f (x _ {k}) \rangle + \int_ {0} ^ {\tau} (\tau - t) \langle \nabla^ {2} f (x _ {k} - t \nabla f (x _ {k})) \nabla f (x _ {k}), \nabla f (x _ {k}) \rangle d t \\ \qquad \leq f (x _ {k}) - \tau \left(1 - \frac {\tau L}{2}\right) \| \nabla f (x _ {k}) \| ^ {2} \end{array}
$$

as long as $\nabla ^ { 2 } f \preceq L { \sf I d }$ . Hence, if $\tau < 2 / L$ , the GD algorithm is decreasing and we can deduce that subsequences of $\left( \boldsymbol { x } _ { k } \right)$ converge to some critical point. 

## A glimpse on the use of proximal gradient methods for non-convex problems

## Theorem (Convergence of FB for non-convex f )

Let f be proper and L-smooth and $g \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ . Let argmin $F \neq \emptyset$ . Let $\left( \boldsymbol { x } _ { k } \right)$ be the sequence generated by the FB algorithm with a constant stepsize $\bar { L } \in \left( \frac { L } { 2 } , + \infty \right)$ Then: 

the sequence $\left( F ( x _ { k } ) \right)$ is non-increasing and $F ( x _ { k + 1 } ) < F ( x _ { k } )$ if and only if $x _ { k }$ is not a stationary point; 

• The (generalised) gradient mapping $G _ { L } : \mathsf { i n t } ( d o m ( f ) ) \to \mathbb { R } ^ { n }$ defined by: 

$$
G _ {\bar {L}} (x) := \bar {L} \left(x - \operatorname{prox} _ {\frac {1}{\bar {L}} g} \left(x - \frac {1}{\bar {L}} \nabla f (x)\right)\right)
$$

is such that $G _ { \bar { L } } ( x _ { k } ) \to 0$ as $k \to + \infty$ 

• All limiting points of $\left( \boldsymbol { x } _ { k } \right)$ are stationary points for the functional F . 

• Earlier works by Fukushima, Mine, ’81, Chouzenoux, Pesquet, Repetti, ’14, Bredies, Lorenz, Reiterer, ’15, Nesterov, ’13. 

• For results on accelerated algorithms see, e.g., Ochs, Chen, Brox, Pock, ’14 

General convergence theory under the (non-restrictive) Kurdyka- Lojasiewicz property (Bolte, Daniilidis, Lewis, ’06, Attouch, Bolte, Svaiter, ’13, Attouch, Bolte, Redont, Subeyran, ’14) 

## Questions?

calatroni@i3s.unice.fr 