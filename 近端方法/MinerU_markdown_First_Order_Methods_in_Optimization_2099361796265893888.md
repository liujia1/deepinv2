<table><tr><td>F(X)</td><td>prox<eq>_{F}</eq>(X)</td><td>Reference</td></tr><tr><td>α||X||<eq>_{F}^{2}</eq></td><td><eq>\frac{1}{1+2\alpha}</eq>X</td><td>Section 6.2.3</td></tr><tr><td>α||X||<eq>_{F}</eq></td><td><eq>\left(1-\frac{\alpha}{\max\{||\mathbf{X}\|_{F},\alpha\}}\right)</eq>X</td><td>Example 6.19</td></tr><tr><td>α||X||<eq>_{S_{1}}</eq></td><td>Udg(<eq>\mathcal{T}_{\alpha}(\boldsymbol{\sigma}(\mathbf{X}))</eq>)V<eq>^{T}</eq></td><td>Example 6.8</td></tr><tr><td>α||X||<eq>_{S_{\infty}}</eq></td><td><eq>\mathbf{X}-\alpha\text{Udg}(P_{B_{\|\cdot\|_{1}}}[\mathbf{0},1](\boldsymbol{\sigma}(\mathbf{X})/\alpha))\mathbf{V}^{T}</eq></td><td>Example 6.48</td></tr><tr><td>α||X||<eq>_{&lt;k&gt;}</eq></td><td><eq>\mathbf{X}-\alpha\text{Udg}(P_{C}(\boldsymbol{\sigma}(\mathbf{X})/\alpha))\mathbf{V}^{T},</eq><eq>C=B_{\|\cdot\|_{1}}[\mathbf{0},k]\cap B_{\|\cdot\|_{\infty}}[\mathbf{0},1]</eq></td><td>Example 6.51</td></tr></table>

Note that $\| \mathbf { X } \| _ { S _ { \infty } }$ can be written as either $\sigma _ { 1 } ( { \mathbf { X } } )$ or $\| \mathbf { X } \| _ { 2 , 2 }$ 

A set $T \subseteq \mathbb { R } ^ { m \times n }$ is called a symmetric spectral set in $\mathbb { R } ^ { m \times n }$ if the indicator function $\delta _ { T }$ is a symmetric spectral function over $\mathbb { R } ^ { m \times n }$ , meaning that it has the form $\delta _ { T } = \delta _ { C } \circ \sigma$ , where $\delta _ { C }$ is an absolutely permutation symmetric function. The set $C \subseteq \mathbb { R } ^ { m \times n }$ is the associated set. Since $\mathrm { p r o x } _ { \delta _ { T } } = P _ { T }$ and $\mathrm { p r o x } _ { \delta _ { C } } = P _ { C }$ , it follows by the spectral prox formula that if C is nonempty closed and convex, then 

$$
P _ {T} (\mathbf {X}) = \mathbf {U} \mathrm{dg} (P _ {C} (\boldsymbol {\sigma} (\mathbf {X}))) \mathbf {V} ^ {T}, \mathbf {X} \in \mathbb {R} ^ {m \times n},\tag{7.11}
$$

where $\mathbf { U } \in \mathbb { O } ^ { m } , \mathbf { V } \in \mathbb { O } ^ { n }$ are assumed to satisfy ${ \bf X } = { \bf U } \mathrm { d } { \bf g } ( { \pmb \sigma } ( { \bf X } ) ) { \bf V } ^ { T }$ 


Example 7.31. Using formula (7.11), we present in the following table expressions for the orthogonal projection onto several symmetric spectral sets in $\mathbb { R } ^ { m \times n }$ . The table also includes references to the corresponding results on the orthogonal projection onto the associated subset of $\mathbb { R } ^ { r }$ . The matrices $\mathbf { U } \in \mathbb { O } ^ { m } , \mathbf { V } \in \mathbb { O } ^ { n }$ are assumed to satisfy ${ \bf X } = { \bf U } \mathrm { d } { \bf g } ( { \pmb \sigma } ( { \bf X } ) ) { \bf V } ^ { T }$


<table><tr><td>set (T)</td><td><eq>P_{T}(\mathbf{X})</eq></td><td>Assumptions</td><td>Reference</td></tr><tr><td><eq>B_{\parallel \cdot \|S_{\infty}}[\mathbf{0},\alpha]</eq></td><td><eq>\mathbf{U}\mathrm{dg}(\mathbf{v})\mathbf{V}^{T}, v_{i} = \min\{\sigma_{i}(\mathbf{X}),\alpha\}</eq></td><td><eq>\alpha &gt; 0</eq></td><td>Lemma 6.26</td></tr><tr><td><eq>B_{\parallel \cdot \|F}[\mathbf{0},r]</eq></td><td><eq>\frac{r}{\max\{\|\mathbf{X}\|_{F},r\}}\mathbf{X}</eq></td><td><eq>r &gt; 0</eq></td><td>Lemma 6.26</td></tr><tr><td><eq>B_{\parallel \cdot \|S_1}[\mathbf{0},\alpha]</eq></td><td><eq>\left\{ \begin{array}{ll}\mathbf{X}, &amp; \| \mathbf{X}\|_{S_1} \leq \alpha, \\ \mathbf{U}\mathrm{dg}(\mathcal{T}_{\beta^*}(\sigma(\mathbf{X})))\mathbf{V}^{T}, &amp; \| \mathbf{X}\|_{S_1} &gt; \alpha, \\ \| \mathcal{T}_{\beta^*}(\boldsymbol{\sigma}(\mathbf{X}))\|_{1} = \alpha, \beta^* &gt; 0 \end{array} \right.</eq></td><td><eq>\alpha &gt; 0</eq></td><td>Example 6.33</td></tr></table>

Chapter 8 

# Primal and Dual Projected Subgradient Methods

Underlying Space: In this chapter E is a Euclidean space, meaning a finite dimensional space endowed with an inner product $\langle \cdot , \cdot \rangle$ and the Euclidean norm $\| \cdot \| = \sqrt { \langle \cdot , \cdot \rangle }$ 

## 8.1 From Gradient Descent to Subgradient Descent

## 8.1.1 Descent Directions?

Consider the unconstrained problem 

$$
\text {(P)} \quad \min \{f (\mathbf {x}): \mathbf {x} \in \mathbb {E} \}.
$$

If f is diferentiable over E, then a well-known method for solving problem (P) is the gradient method, also known as steepest descent, which takes the form 

$$
\mathbf {x} ^ {k + 1} = \mathbf {x} ^ {k} - t _ {k} \nabla f (\mathbf {x} ^ {k}),\tag{8.1}
$$

where $t _ { k }$ is an appropriately chosen stepsize. A key property of the direction of the negative of the gradient is that it is a descent direction, a notion that is now recalled. 

Definition 8.1 (descent direction). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be an extended realvalued function, and let $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ . A vector $\mathbf { 0 } \neq \mathbf { d } \in \mathbb { E }$ is called a descent direction of f at x if the directional derivative $f ^ { \prime } ( \mathbf { x } ; \mathbf { d } )$ exists and is negative. 

An important property of descent directions, which can be directly deduced from their definition, is that taking small enough steps along these directions leads to a decrease in function value. 

Lemma 8.2 (descent property of descent directions [10, Lemma 4.2]). Let $f : \mathbb { E }  ( - \infty , \infty ]$ be an extended real-valued function. $L e t \mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ , and assume that $\mathbf { 0 } \neq \mathbf { d } \in \mathbb { E }$ is a descent direction of f at x. Then there exists $\varepsilon > 0$ such that $\mathbf { x } + t \mathbf { d } \in \operatorname { d o m } ( f )$ and 

$$
f (\mathbf {x} + t \mathbf {d}) <   f (\mathbf {x})
$$

for any $t \in ( 0 , \varepsilon ]$ 

Coming back to the gradient method, we note that the directional derivative of f at $\mathbf { x } ^ { k }$ in the direction of $- \nabla f ( \mathbf { x } ^ { k } )$ is negative as long as $\nabla f ( \mathbf { x } ^ { k } ) \neq \mathbf { 0 }$ : 

$$
f ^ {\prime} (\mathbf {x} ^ {k}; - \nabla f (\mathbf {x} ^ {k})) = \langle \nabla f (\mathbf {x} ^ {k}), - \nabla f (\mathbf {x} ^ {k}) \rangle = - \| \nabla f (\mathbf {x} ^ {k}) \| ^ {2} <   0,\tag{8.2}
$$

where Theorem 3.29 was used in the first equality. We have thus shown that $- \nabla f ( \mathbf { x } ^ { k } )$ is a descent direction of $f$ at $\mathbf { x } ^ { k }$ , which by Lemma 8.2 implies that there exists $\varepsilon > 0$ such that $f ( { \mathbf { x } } ^ { k } - t \nabla f ( { \mathbf { x } } ^ { k } ) ) < f ( { \mathbf { x } } ^ { k } )$ for any $t \in ( 0 , \varepsilon ]$ . In particular, this means that $t _ { k }$ can always be chosen in a way that guarantees a decrease in the function value from one iteration to the next. For example, one choice of stepsize that guarantees descent is the exact line search strategy in which $t _ { k }$ is chosen as 

$$
t _ {k} \in \operatorname{argmin} _ {t \geq 0} f (\mathbf {x} ^ {k} - t \nabla f (\mathbf {x} ^ {k})).
$$

If f is not diferentiable, then scheme (8.1) is not well defined. Under our convexity assumption, a natural generalization to the nonsmooth case will consist in replacing the gradient by a subgradient (assuming that it exists): 

$$
\mathbf {x} ^ {k + 1} = \mathbf {x} ^ {k} - t _ {k} \mathbf {g} ^ {k}, \quad \mathbf {g} ^ {k} \in \partial f (\mathbf {x} ^ {k}),\tag{8.3}
$$

where we assume that the choice of the subgradient from $\partial f ( \mathbf { x } ^ { k } )$ is arbitrary. The scheme (8.3) is called the subgradient method. One substantial diference between the gradient and subgradient methods is that the direction of minus the subgradient is not necessarily a descent direction. This means that $t _ { k }$ cannot be chosen in a way that will guarantee a descent property in function values of the scheme (8.3). 

Example 8.3 (non-descent subgradient direction).<sup>40</sup> Consider the function $f : \mathbb { R } \times \mathbb { R } \to \mathbb { R }$ given by $f ( x _ { 1 } , x _ { 2 } ) = | x _ { 1 } | + 2 | x _ { 2 } |$ . Then 

$$
\partial f (1, 0) = \{(1, x): | x | \leq 2 \}.
$$

In particular, $( 1 , 2 ) \in \partial f ( 1 , 0 )$ . However, the direction $- ( 1 , 2 )$ is not a descent direction. To show this, note that for any $t > 0$ 2 

$$
g (t) \equiv f ((1, 0) - t (1, 2)) = f (1 - t, - 2 t) = | 1 - t | + 4 t = \left\{ \begin{array}{l l} 1 + 3 t, & t \in (0, 1 ], \\ 5 t - 1, & t \geq 1. \end{array} \right.\tag{8.4}
$$

In particular, 

$$
f ^ {\prime} ((1, 0); - (1, 2)) = g _ {+} ^ {\prime} (0) = 3 > 0,
$$

showing that $- ( 1 , 2 )$ is not a descent direction. It is also interesting to note that by (8.4), it holds that 

$$
f ((1, 0) - t (1, 2)) \geq 1 = f (1, 0) \mathrm{forany} t > 0,
$$

which actually shows that there is no point in the ray $\{ ( 1 , 0 ) - t ( 1 , 2 ) : t > 0 \}$ with a smaller function value than (1, 0). 

## 8.1.2 Wolfe’s Example

To better understand the efect of nonsmoothness, we recall a famous example of Wolfe. The example deals with the gradient method employed on a nonsmooth convex function with stepsizes chosen by exact line search. The function is differentiable at all the iterate vectors generated by the method, which in particular means that all the directions picked by the method are descent directions, and the sequence of function values strictly decreases. However, although it seems that the nonsmoothness is “bypassed,” this is hardly the case. The sequence generated by the method converges to a nonoptimal point. 

Let $\gamma > 1$ , and consider the function $f : \mathbb { R } \times \mathbb { R } \to \mathbb { R }$ given by 

$$
f (x _ {1}, x _ {2}) = \left\{ \begin{array}{l l} \sqrt {x _ {1} ^ {2} + \gamma x _ {2} ^ {2}}, & | x _ {2} | \leq x _ {1}, \\ \frac {x _ {1} + \gamma | x _ {2} |}{\sqrt {1 + \gamma}} & \text { else }. \end{array} \right.\tag{8.5}
$$

We begin by showing in Lemma 8.5 below that the function $f$ is closed and convex and describe its subdiferential set at any point in $\mathbb { R } \times \mathbb { R }$ . For that, we will prove that $f$ is actually a support function of a closed and convex set.<sup>41</sup> The proof of Lemma 8.5 uses the following simple technical lemma, whose trivial proof is omitted. 

Lemma 8.4. Consider the problem 

$$
\text {(P)} \quad \max \{g (\mathbf {y}): f _ {1} (\mathbf {y}) \leq 0, f _ {2} (\mathbf {y}) \leq 0 \},
$$

where $g : \mathbb { E }  \mathbb { R }$ is concave and $f _ { 1 } , f _ { 2 } : \mathbb { E }  \mathbb { R }$ are convex. Assume that the problem max $\{ g ( \mathbf { y } ) : f _ { 1 } ( \mathbf { y } ) \leq 0 \}$ has a unique solution $\tilde { \mathbf { y } }$ . Let $Y ^ { * }$ be the optimal set of problem (P). Then exactly one of the following two options holds: 

(i) $f _ { 2 } ( \tilde { \mathbf { y } } ) \leq 0 ,$ , and in this case $Y ^ { * } = \{ \tilde { \mathbf { y } } \}$ 

(ii) $f _ { 2 } ( \tilde { \mathbf { y } } ) > 0$ , and in this case $Y ^ { * } = \operatorname { a r g m a x } \{ g ( \mathbf { y } ) : f _ { 1 } ( \mathbf { y } ) \leq 0 , f _ { 2 } ( \mathbf { y } ) = 0 \}$ 

Lemma 8.5. Let f be given by (8.5). Then 

(a) $f = \sigma _ { C }$ , where 

$$
C = \left\{(y _ {1}, y _ {2}) \in \mathbb {R} \times \mathbb {R}: y _ {1} ^ {2} + \frac {y _ {2} ^ {2}}{\gamma} \leq 1, y _ {1} \geq \frac {1}{\sqrt {1 + \gamma}} \right\};
$$

(b) f is closed and convex; 

$$
\partial f (x _ {1}, x _ {2}) = \left\{ \begin{array}{l l} C, & x _ {1} = x _ {2} = 0, \\ \frac {(x _ {1} , \gamma x _ {2})}{\sqrt {x _ {1} ^ {2} + \gamma x _ {2} ^ {2}}}, & | x _ {2} | \leq x _ {1}, x _ {1} \neq 0, \\ \left(\frac {1}{\sqrt {1 + \gamma}}, \frac {\gamma \mathrm{sgn} (x _ {2})}{\sqrt {1 + \gamma}}\right), & | x _ {2} | > x _ {1}, x _ {2} \neq 0, \\ \left\{\frac {1}{\sqrt {\gamma + 1}} \right\} \times \left[ - \frac {\gamma}{\sqrt {1 + \gamma}}, \frac {\gamma}{\sqrt {1 + \gamma}} \right], & x _ {2} = 0, x _ {1} <   0. \end{array} \right.\tag{c}
$$

Proof. By the definition of support functions, 

$$
\sigma_ {C} \left(x _ {1}, x _ {2}\right) = \max _ {y _ {1}, y _ {2}} \left\{x _ {1} y _ {1} + x _ {2} y _ {2}: y _ {1} ^ {2} + \frac {y _ {2} ^ {2}}{\gamma} \leq 1, y _ {1} \geq \frac {1}{\sqrt {1 + \gamma}} \right\}.\tag{8.6}
$$

Note that if $( x _ { 1 } , x _ { 2 } ) = ( 0 , 0 )$ , then $\sigma _ { C } ( x _ { 1 } , x _ { 2 } ) = 0$ and 

$$
\operatorname{argmax} _ {y _ {1}, y _ {2}} \left\{x _ {1} y _ {1} + x _ {2} y _ {2}: y _ {1} ^ {2} + \frac {y _ {2} ^ {2}}{\gamma} \leq 1, y _ {1} \geq \frac {1}{\sqrt {1 + \gamma}} \right\} = C.
$$

Assume that $( x _ { 1 } , x _ { 2 } ) \neq ( 0 , 0 )$ . Denoting $g ( y _ { 1 } , y _ { 2 } ) = x _ { 1 } y _ { 1 } + x _ { 2 } y _ { 2 } , \ f _ { 1 } ( y _ { 1 } , y _ { 2 } ) =$ $\begin{array} { r } { y _ { 1 } ^ { 2 } + \frac { y _ { 2 } ^ { 2 } } { \gamma } - 1 } \end{array}$ and $\begin{array} { r } { f _ { 2 } ( y _ { 1 } , y _ { 2 } ) = - y _ { 1 } + \frac { 1 } { \sqrt { 1 + \gamma } } } \end{array}$ , problem (8.6) becomes 

$$
\max _ {y _ {1}, y _ {2}} \{g (y _ {1}, y _ {2}): f _ {1} (y _ {1}, y _ {2}) \leq 0, f _ {2} (y _ {1}, y _ {2}) \leq 0 \}.
$$

The assumptions made in Lemma 8.4 are all met: g is concave, $f _ { 1 } , f _ { 2 }$ are convex, and the optimal solution of 

$$
\max _ {y _ {1}, y _ {2}} \{g (y _ {1}, y _ {2}): f _ {1} (y _ {1}, y _ {2}) \leq 0 \}
$$

is unique and equal to $\begin{array} { r } { ( \tilde { y } _ { 1 } , \tilde { y } _ { 2 } ) = \frac { ( x _ { 1 } , \gamma x _ { 2 } ) } { \sqrt { x _ { 1 } ^ { 2 } + \gamma x _ { 2 } ^ { 2 } } } } \end{array}$ . Thus, by Lemma 8.4, there are two options: 

Case I: $f _ { 2 } ( \tilde { y } _ { 1 } , \tilde { y } _ { 2 } ) \leq 0$ , meaning that $\begin{array} { r } { \frac { x _ { 1 } } { \sqrt { x _ { 1 } ^ { 2 } + \gamma x _ { 2 } ^ { 2 } } } \geq \frac { 1 } { \sqrt { 1 + \gamma } } } \end{array}$ . It can be easily seen that the last inequality is equivalent to the condition $| x _ { 2 } | \le x _ { 1 }$ . Under this condition, by Lemma 8.4, $\begin{array} { r } { ( \tilde { y } _ { 1 } , \tilde { y } _ { 2 } ) = \frac { ( x _ { 1 } , \gamma x _ { 2 } ) } { \sqrt { x _ { 1 } ^ { 2 } + \gamma x _ { 2 } ^ { 2 } } } } \end{array}$ is the unique optimal solution of problem (8.6) with a corresponding function value of $\sigma _ { C } ( x _ { 1 } , x _ { 2 } ) = \sqrt { x _ { 1 } ^ { 2 } + \gamma x _ { 2 } ^ { 2 } } .$ 

Case II: $f _ { 2 } ( \tilde { y } _ { 1 } , \tilde { y } _ { 2 } ) > 0$ , which is the same as $x _ { 1 } < | x _ { 2 } |$ . In this case, by Lemma 8.4, all the optimal solutions of problem (8.6) satisfy $\begin{array} { r } { \dot { y } _ { 1 } = \frac { 1 } { \sqrt { 1 + \gamma } } } \end{array}$ , and the problem thus amounts to 

$$
\max _ {y _ {2}} \left\{\frac {1}{\sqrt {1 + \gamma}} x _ {1} + x _ {2} y _ {2}: y _ {2} ^ {2} \leq \frac {\gamma^ {2}}{1 + \gamma} \right\}.
$$

The set of maximizers of the above problem is either $\textstyle \left\{ { \frac { \gamma \mathrm { s g n } ( x _ { 2 } ) } { \sqrt { 1 + \gamma } } } \right\}$ if $x _ { 2 } ~ \neq ~ 0$ or $\textstyle \left[ - { \frac { \gamma } { \sqrt { 1 + \gamma } } } , { \frac { \gamma } { \sqrt { 1 + \gamma } } } \right] { \mathrm { ~ i f ~ } } x _ { 2 } = 0$ . In both options, $\begin{array} { r } { \sigma _ { C } ( x _ { 1 } , x _ { 2 } ) = \frac { x _ { 1 } + \gamma | x _ { 2 } | } { \sqrt { 1 + \gamma } } } \end{array}$ 

To summarize, we have shown that 

$$
\sigma_ {C} (x _ {1}, x _ {2}) = \left\{ \begin{array}{l l} \sqrt {x _ {1} ^ {2} + \gamma x _ {2} ^ {2}}, & | x _ {2} | \leq x _ {1}, \\ \frac {x _ {1} + \gamma | x _ {2} |}{\sqrt {1 + \gamma}} & \text { else }, \end{array} \right.
$$

establishing part (a), meaning that $f = \sigma _ { C }$ . Therefore, $f ,$ as a support function, is a closed and convex function, and we have thus established part (b) as well. To prove part (c), note that we also showed that 

$$
\begin{array}{l} \operatorname{argmax} _ {y _ {1}, y _ {2}} \{x _ {1} y _ {1} + x _ {2} y _ {2}: (y _ {1}, y _ {2}) \in C \} \\ = \left\{ \begin{array}{l l} C, & x _ {1} = x _ {2} = 0, \\ \frac {(x _ {1} , \gamma x _ {2})}{\sqrt {x _ {1} ^ {2} + \gamma x _ {2} ^ {2}}}, & | x _ {2} | \leq x _ {1}, x _ {1} \neq 0, \\ \left(\frac {1}{\sqrt {1 + \gamma}}, \frac {\gamma \mathrm{sgn} (x _ {2})}{\sqrt {1 + \gamma}}\right), & | x _ {2} | > x _ {1}, x _ {2} \neq 0, \\ \left\{\frac {1}{\sqrt {\gamma + 1}} \right\} \times \left[ - \frac {\gamma}{\sqrt {1 + \gamma}}, \frac {\gamma}{\sqrt {1 + \gamma}} \right], & x _ {2} = 0, x _ {1} <   0. \end{array} \right. \end{array}
$$

Combining this with the conjugate subgradient theorem (Corollary 4.21), as well as Example 4.9 and the closedness and convexity of C, implies 

$$
\begin{array}{l} \partial f (x _ {1}, x _ {2}) = \partial \sigma_ {C} (x _ {1}, x _ {2}) \\ \qquad = \operatorname{argmax} _ {y _ {1}, y _ {2}} \{x _ {1} y _ {1} + x _ {2} y _ {2} - \sigma_ {C} ^ {*} (y _ {1}, y _ {2}) \} \\ \qquad = \operatorname{argmax} _ {y _ {1}, y _ {2}} \{x _ {1} y _ {1} + x _ {2} y _ {2} - \delta_ {C} (y _ {1}, y _ {2}) \} \\ \qquad = \operatorname{argmax} _ {y _ {1}, y _ {2}} \{x _ {1} y _ {1} + x _ {2} y _ {2}: (y _ {1}, y _ {2}) \in C \} \\ = \left\{ \begin{array}{l l} C, & x _ {1} = x _ {2} = 0, \\ \frac {(x _ {1} , \gamma x _ {2})}{\sqrt {x _ {1} ^ {2} + \gamma x _ {2} ^ {2}}}, & | x _ {2} | \leq x _ {1}, x _ {1} \neq 0, \\ \left(\frac {1}{\sqrt {1 + \gamma}}, \frac {\gamma \mathrm{sgn} (x _ {2})}{\sqrt {1 + \gamma}}\right), & | x _ {2} | > x _ {1}, x _ {2} \neq 0, \\ \left\{\frac {1}{\sqrt {\gamma + 1}} \right\} \times \left[ - \frac {\gamma}{\sqrt {1 + \gamma}}, \frac {\gamma}{\sqrt {1 + \gamma}} \right], & x _ {2} = 0, x _ {1} <   0. \end{array} \right. \end{array}
$$

Note that a direct result of part (c) of Lemma 8.5 and Theorem 3.33 is that f is not diferentiable only at the nonpositive part of the $x _ { 1 }$ axis. 

In the next result we will show that the gradient method with exact line search employed on f with a certain initialization converges to the nonoptimal point (0, 0) even though all the points generated by the gradient method are points in which f is diferentiable. 

Lemma 8.6. Let $\{ ( x _ { 1 } ^ { ( k ) } , x _ { 2 } ^ { ( k ) } ) \} _ { k \geq 0 }$ be the sequence generated by the gradient method with exact line search employed on f with initial point $( x _ { 1 } ^ { 0 } , x _ { 2 } ^ { 0 } ) = ( \gamma , 1 )$ , where $\gamma > 1$ Then for any $k \geq 0$ 

(a) f is diferentiable at $( x _ { 1 } ^ { ( k ) } , x _ { 2 } ^ { ( k ) } )$ ; 

(b) $| x _ { 2 } ^ { ( k ) } | \leq x _ { 1 } ^ { ( k ) }$ and $x _ { 1 } ^ { ( k ) } \neq 0$ ; 

$$
\text {(c)} \left(x _ {1} ^ {(k)}, x _ {2} ^ {(k)}\right) = \left(\gamma \left(\frac {\gamma - 1}{\gamma + 1}\right) ^ {k}, \left(- \frac {\gamma - 1}{\gamma + 1}\right) ^ {k}\right).
$$

Proof. We only need to show part (c) since part (b) follows directly from the expression of $( x _ { 1 } ^ { ( k ) } , x _ { 2 } ^ { ( k ) } )$ ) given in (c), and part (a) is then a consequence of Lemma 8.5(c). 

We will prove part (c) by induction. The claim is obviously correct for $k = 0$ by the choice of initial point. Assume that the claim is correct for k, that is, 

$$
(x _ {1} ^ {(k)}, x _ {2} ^ {(k)}) = \left(\gamma \left(\frac {\gamma - 1}{\gamma + 1}\right) ^ {k}, \left(- \frac {\gamma - 1}{\gamma + 1}\right) ^ {k}\right).
$$

We will prove that it is correct for $k + 1$ , meaning that 

$$
(x _ {1} ^ {(k + 1)}, x _ {2} ^ {(k + 1)}) = (\beta_ {k}, \gamma_ {k}),\tag{8.7}
$$

where 

$$
\beta_ {k} = \gamma \left(\frac {\gamma - 1}{\gamma + 1}\right) ^ {k + 1}, \gamma_ {k} = \left(- \frac {\gamma - 1}{\gamma + 1}\right) ^ {k + 1}.
$$

Since $| x _ { 2 } ^ { ( k ) } | \leq x _ { 1 } ^ { ( k ) }$ and $x _ { 1 } ^ { ( k ) } \neq 0$ , we have $f ( x _ { 1 } ^ { ( k ) } , x _ { 2 } ^ { ( k ) } ) = \sqrt { ( x _ { 1 } ^ { ( k ) } ) ^ { 2 } + \gamma ( x _ { 2 } ^ { ( k ) } ) ^ { 2 } }$ , and by Lemma $8 . 5 ( \mathrm { c } ) , f$ is diferentiable at $( x _ { 1 } ^ { ( k ) } , x _ { 2 } ^ { ( k ) } )$ with 

$$
\nabla f (x _ {1} ^ {(k)}, x _ {2} ^ {(k)}) = \frac {1}{\sqrt {(x _ {1} ^ {(k)}) ^ {2} + \gamma (x _ {2} ^ {(k)}) ^ {2}}} (x _ {1} ^ {(k)}, \gamma x _ {2} ^ {(k)}).
$$

What is important in the above formula is that $\nabla f ( x _ { 1 } ^ { ( k ) } , x _ { 2 } ^ { ( k ) } )$ can be written in the form 

$$
\nabla f (x _ {1} ^ {(k)}, x _ {2} ^ {(k)}) = \alpha_ {k} (x _ {1} ^ {(k)}, \gamma x _ {2} ^ {(k)})\tag{8.8}
$$

for some positive constant $\alpha _ { k }$ . To show the validity of (8.7), we will define $g ( t ) \equiv$ $f ( ( x _ { 1 } ^ { ( k ) } , x _ { 2 } ^ { ( k ) } ) - t ( x _ { 1 } ^ { ( k ) } , \gamma x _ { 2 } ^ { ( k ) } ) )$ and prove the following two statements: 

$$
\mathrm{(A)} (\beta_ {k}, \gamma_ {k}) = (x _ {1} ^ {(k)}, x _ {2} ^ {(k)}) - \frac {2}{\gamma + 1} (x _ {1} ^ {(k)}, \gamma x _ {2} ^ {(k)}).
$$

$$
\text {(B)} g ^ {\prime} \left(\frac {2}{\gamma + 1}\right) = 0.
$$

(A) and (B) are enough to show (8.7) since g is strictly convex. The proof of $\mathrm { ( A ) }$ follows by the computations below: 

$$
x _ {1} ^ {(k)} - \frac {2}{\gamma + 1} x _ {1} ^ {(k)} = \frac {\gamma - 1}{\gamma + 1} x _ {1} ^ {(k)} = \frac {\gamma - 1}{\gamma + 1} \gamma \left(\frac {\gamma - 1}{\gamma + 1}\right) ^ {k} = \gamma \left(\frac {\gamma - 1}{\gamma + 1}\right) ^ {k + 1} = \beta_ {k},
$$

$$
x _ {2} ^ {(k)} - \frac {2 \gamma}{\gamma + 1} x _ {2} ^ {(k)} = \frac {- \gamma + 1}{\gamma + 1} x _ {2} ^ {(k)} = \frac {- \gamma + 1}{\gamma + 1} \left(- \frac {\gamma - 1}{\gamma + 1}\right) ^ {k} = \left(- \frac {\gamma - 1}{\gamma + 1}\right) ^ {k + 1} = \gamma_ {k}.
$$

To prove (B), note that 

$$
\begin{array}{l} g (t) = f \left((x _ {1} ^ {(k)}, x _ {2} ^ {(k)}) - t (x _ {1} ^ {(k)}, \gamma x _ {2} ^ {(k)})\right) = f ((1 - t) x _ {1} ^ {(k)}, (1 - \gamma t) x _ {2} ^ {(k)}) \\ = \sqrt {(1 - t) ^ {2} (x _ {1} ^ {(k)}) ^ {2} + \gamma (1 - \gamma t) ^ {2} (x _ {2} ^ {(k)}) ^ {2}}. \end{array}
$$

Therefore, 

$$
g ^ {\prime} (t) = \frac {(t - 1) (x _ {1} ^ {(k)}) ^ {2} + \gamma^ {2} (\gamma t - 1) (x _ {2} ^ {(k)}) ^ {2}}{\sqrt {(1 - t) ^ {2} (x _ {1} ^ {(k)}) ^ {2} + \gamma (1 - \gamma t) ^ {2} (x _ {2} ^ {(k)}) ^ {2}}}.\tag{8.9}
$$

To prove that $\begin{array} { r } { g ^ { \prime } \big ( \frac { 2 } { \gamma + 1 } \big ) = 0 } \end{array}$ , it is enough to show that the nominator in the last expression is equal to zero at $\textstyle t = { \frac { 2 } { \gamma + 1 } }$ . Indeed, 

$$
\begin{array}{l} \left(\frac {2}{\gamma + 1} - 1\right) (x _ {1} ^ {(k)}) ^ {2} + \gamma^ {2} \left(\gamma \cdot \frac {2}{\gamma + 1} - 1\right) (x _ {2} ^ {(k)}) ^ {2} \\ = \left(- \frac {\gamma - 1}{\gamma + 1}\right) \gamma^ {2} \left(\frac {\gamma - 1}{\gamma + 1}\right) ^ {2 k} + \gamma^ {2} \left(\frac {\gamma - 1}{\gamma + 1}\right) \left(- \frac {\gamma - 1}{\gamma + 1}\right) ^ {2 k} \\ = 0. \quad \square \end{array}
$$

Obviously, by Lemma 8.6, the sequence generated by the gradient method with exact line search and initial point $( \gamma , 1 )$ converges to (0, 0), which is not a minimizer of f since f is not bounded below (take $x _ { 2 } = 0$ and $x _ { 1 } \to - \infty )$ . Actually, $( - 1 , 0 )$ is a descent direction of f at (0, 0). The contour lines of the function along with the iterates of the gradient method are described in Figure 8.1. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/4c4d6e0d43c773e5f046318576237d6fe7ef2773085b5445d92092bea92e24c9.jpg)



Figure 8.1. Contour lines of Wolfe’s function with $\begin{array} { r } { \gamma = \frac { 1 6 } { 9 } } \end{array}$ along with the iterates of the gradient method with exact line search.


## 8.2 The Projected Subgradient Method

The main model that will be discussed in this section is 

$$
\min \{f (\mathbf {x}): \mathbf {x} \in C \},\tag{8.10}
$$

where the following assumption will be made throughout this section. 

## Assumption 8.7.

(A) $f : \mathbb { E } \to ( - \infty , \infty ]$ is proper closed and convex. 

(B) $C \subseteq \mathbb { E }$ is nonempty closed and convex. 

(C) $C \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ 

(D) The optimal set of (8.10) is nonempty and denoted by $X ^ { * }$ . The optimal value of the problem is denoted by $f _ { \mathrm { o p t } }$ 

Remark 8.8 (subdiferentiability of f and closedness of $X ^ { * } )$ . Since f is convex and $C \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ , it follows by Theorem 3.14 that $f$ is subdiferentiable over C. Also, since f is closed, 

$$
X ^ {*} = C \cap \operatorname{Lev} (f, f _ {\mathrm{opt}})
$$

is closed. This means in particular that for any ${ \textbf { x } } \notin X ^ { * }$ the distance $d _ { X ^ { * } } ( \mathbf { x } )$ is positive. 

From now on, we will use the following notation: $f ^ { \prime } ( \mathbf { x } )$ will denote a certain subgradient of f at x, meaning a member in $\partial f ( \mathbf { x } )$ . Thus, $f ^ { \prime }$ is actually a function from $C$ to $\mathbb { E } ^ { * }$ . The rule for choosing $f ^ { \prime } ( \mathbf { x } )$ out of the members of $\partial f ( \mathbf { x } )$ can be arbitrary but has to be deterministic, meaning that if $f ^ { \prime } ( \mathbf { x } )$ is evaluated twice, the results have to be the same. 

Equipped with the observations of the previous section, we can speculate that a method which utilizes subgradients rather than gradients will not necessarily be a descent method and will not have to be based on a line search procedure for choosing its stepsizes. We will see that this is indeed the case for the projected subgradient method. 

## 8.2.1 The Method

Each iteration of the projected subgradient method consists of a step taken toward the negative of the chosen subgradient followed by an orthogonal projection onto the underlying set C. 

Projected Subgradient Method
Initialization: pick $x^{0} \in C$ arbitrarily.
General step: for any $k = 0, 1, 2, \ldots$ execute the following steps:
(a) pick a stepsize $t_{k} > 0$ and a subgradient $f'(\mathbf{x}^{k}) \in \partial f(\mathbf{x}^{k})$ ;
(b) set $\mathbf{x}^{k+1} = P_{C}(\mathbf{x}^{k} - t_{k}f'(\mathbf{x}^{k}))$ . 

The sequence generated by the projected subgradient method is $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 } .$ while the sequence of function values generated by the method is $\{ f ( \mathbf { x } ^ { k } ) \} _ { k \geq 0 }$ . As was already discussed, the sequence of function values is not necessarily monotone, and we will be also interested in the sequence of best achieved function values, which is defined by 

$$
f _ {\mathrm{best}} ^ {k} \equiv \min _ {n = 0, 1, \dots , k} f (\mathbf {x} ^ {n}).\tag{8.11}
$$

Obviously, the sequence $\{ f _ { \mathrm { b e s t } } ^ { k } \} _ { k \geq 0 }$ is nonincreasing. 

Remark 8.9 (stopping criterion for the projected subgradient method). In actual implementations of the projected subgradient method, a stopping criterion has to be incorporated, but as a rule, we will not deal in this book with stopping criteria but rather concentrate on issues of convergence. 

Remark 8.10 (zero subgradients). In the unlikely case where $f ^ { \prime } ( \mathbf { x } ^ { k } ) = \mathbf { 0 }$ for some k, then by Fermat’s optimality condition (Theorem 3.63), $\mathbf { x } ^ { k }$ is a minimizer of f over E, and since $\mathbf { x } ^ { k } \in C$ , it is also a minimizer of f over C. In this situation, the method is “stuck” at the optimal solution $\mathbf { x } ^ { k }$ from iteration k onward, meaning that $\mathbf { x } ^ { n } = \mathbf { x } ^ { k }$ for all $n \geq k$ 

The analysis of the projected subgradient method relies on the following simple technical lemma. 

Lemma 8.11 (fundamental inequality for projected subgradient). Suppose that Assumption 8.7 holds. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the projected subgradient method. Then for any $\mathbf { x } ^ { * } \in X ^ { * }$ and $k \geq 0$ , 

$$
\| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2} \leq \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {k} (f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}) + t _ {k} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {k}) \| ^ {2}.\tag{8.12}
$$

Proof. 

$$
\begin{array}{r l} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2} = & \| P _ {C} (\mathbf {x} ^ {k} - t _ {k} f ^ {\prime} (\mathbf {x} ^ {k})) - P _ {C} (\mathbf {x} ^ {*}) \| ^ {2} \\ \overset {(*)} {\leq} & \| \mathbf {x} ^ {k} - t _ {k} f ^ {\prime} (\mathbf {x} ^ {k}) - \mathbf {x} ^ {*} \| ^ {2} \\ = & \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {k} \langle f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \rangle + t _ {k} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {k}) \| ^ {2} \\ \overset {(* *)} {\leq} & \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {k} (f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}) + t _ {k} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {k}) \| ^ {2}, \end{array}
$$

where the inequality (∗) is due to the nonexpansiveness of the orthogonal projection operator (Theorem 6.42), and (∗∗) follows by the subgradient inequality. 

## 8.2.2 Convergence under Polyak’s Stepsize Rule

We will require an assumption in addition to Assumption 8.7 in order to prove convergence of the sequence of function values generated by the projected subgradient method. 

Assumption 8.12. There exists a constant $L _ { f } > 0$ for which $\| \mathbf { g } \| \leq L _ { f }$ for all $\mathbf { g } \in \partial f ( \mathbf { x } ) , \mathbf { x } \in C$ 

Since $C \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ (Assumption $8 . 7 ( \mathrm { C } ) )$ , it follows by Theorem 3.61 that Assumption 8.12 implies that f is Lipschitz continuous over $C$ with constant $L _ { f } \mathbf { : }$ : 

$$
| f (\mathbf {x}) - f (\mathbf {y}) | \leq L _ {f} \| \mathbf {x} - \mathbf {y} \| \text {   for   all   } \mathbf {x}, \mathbf {y} \in C.
$$

In addition, since (again) $C \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ , it follows by Theorem 3.16 that $\mathrm { A s } -$ sumption 8.12 holds if C is assumed to be compact. 

One natural way to choose the stepsize $t _ { k }$ is by taking it as the minimizer of the right-hand side of (8.12) over $t _ { k } \geq 0$ 

$$
t _ {k} = \frac {f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}}{\| f ^ {\prime} (\mathbf {x} ^ {k}) \| ^ {2}}.
$$

When $f ^ { \prime } ( \mathbf { x } ^ { k } ) = \mathbf { 0 }$ , the above formula is not defined, and by Remark 8.10, $\mathbf { x } ^ { k }$ is an optimal solution of (8.10). We will artificially define $t _ { k } = 1$ (any other positive number could also have been chosen). The complete formula is therefore 

$$
t _ {k} = \left\{ \begin{array}{l l} \frac {f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}}{\| f ^ {\prime} (\mathbf {x} ^ {k}) \| ^ {2}}, & f ^ {\prime} (\mathbf {x} ^ {k}) \neq \mathbf {0}, \\ 1, & f ^ {\prime} (\mathbf {x} ^ {k}) = \mathbf {0}. \end{array} \right.\tag{8.13}
$$

We will refer to this stepsize rule as Polyak’s stepsize $r u l e . ^ { 4 2 }$ 

The main convergence result of the projected subgradient method with Polyak’s stepsize rule is given in the next theorem. 

Theorem 8.13 (convergence of projected subgradient with Polyak’s stepsize). Suppose that Assumptions 8.7 and 8.12 hold. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the projected subgradient method with Polyak’s stepsize rule (8.13). Then 

$$
\left(\mathrm{a}\right) \left\| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \right\| ^ {2} \leq \left\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \right\| ^ {2} f o r a n y k \geq 0 a n d \mathbf {x} ^ {*} \in X ^ {*};
$$

(b) $f ( { \bf x } ^ { k } ) \to f _ { \mathrm { o p t } } \ a s \ k \to \infty ,$ 

(c) $\begin{array} { r } { f _ { \mathrm { b e s t } } ^ { k } - f _ { \mathrm { o p t } } \leq \frac { L _ { f } d _ { X ^ { * } } ( \mathbf { x } ^ { 0 } ) } { \sqrt { k + 1 } } } \end{array}$ for any $k \geq 0$ 

Proof. Let n be a nonnegative integer and $\mathbf { x } ^ { * } \in X ^ { * }$ . By Lemma 8.11, 

$$
\left\| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \right\| ^ {2} \leq \left\| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \right\| ^ {2} - 2 t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\text { o   p   t }}) + t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| ^ {2}.\tag{8.14}
$$

If $f ^ { \prime } ( \mathbf { x } ^ { n } ) \neq \mathbf { 0 }$ , then by substituting $\begin{array} { r } { t _ { n } = \frac { f ( \mathbf { x } ^ { n } ) - f _ { \mathrm { o p t } } } { \Vert f ^ { \prime } ( \mathbf { x } ^ { n } ) \Vert ^ { 2 } } } \end{array}$ into (8.14), it follows that 

$$
\| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} \leq \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - \frac {(f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) ^ {2}}{\| f ^ {\prime} (\mathbf {x} ^ {n}) \| ^ {2}}.
$$

Using the bound $\| f ^ { \prime } ( \mathbf { x } ^ { n } ) \| \leq L _ { f }$ , we thus obtain 

$$
\left\| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \right\| ^ {2} \leq \left\| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \right\| ^ {2} - \frac {\left(f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}\right) ^ {2}}{L _ {f} ^ {2}}.\tag{8.15}
$$

Inequality (8.15) also holds when $f ^ { \prime } ( \mathbf { x } ^ { n } ) = \mathbf { 0 }$ , since in this case $f ( \mathbf { x } ^ { n } ) = f _ { \mathrm { o p t } }$ and $\mathbf { x } ^ { n + \bar { 1 } } = \bar { \mathbf { x } } ^ { n }$ . A direct result of (8.15) is that 

$$
\| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} \leq \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2},
$$

and part (a) is thus proved (by plugging $n = k )$ . Summing inequality (8.15) over $n = 0 , 1 , \ldots , k$ , we obtain that 

$$
\frac {1}{L _ {f} ^ {2}} \sum_ {n = 0} ^ {k} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) ^ {2} \leq \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} - \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2},
$$

and thus 

$$
\sum_ {n = 0} ^ {k} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) ^ {2} \leq L _ {f} ^ {2} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}.
$$

Since the above inequality holds for any $\mathbf { x } ^ { * } \in X ^ { * }$ , it follows that 

$$
\sum_ {n = 0} ^ {k} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) ^ {2} \leq L _ {f} ^ {2} d _ {X ^ {*}} ^ {2} (\mathbf {x} ^ {0}),\tag{8.16}
$$

which in particular implies that $f ( \mathbf { x } ^ { n } ) { - } f _ { \mathrm { o p t } } \to 0$ as $n \to \infty$ , and the validity of (b) is established. To prove part (c), note that since $f ( \mathbf { x } ^ { n } ) \geq f _ { \mathrm { b e s t } } ^ { k }$ for any $n = 0 , 1 , \ldots , k$ 2 it follows that 

$$
\sum_ {n = 0} ^ {k} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) ^ {2} \geq (k + 1) (f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}}) ^ {2},
$$

which, combined with (8.16), yields 

$$
(k + 1) (f _ {\text { best }} ^ {k} - f _ {\text { opt }}) ^ {2} \leq L _ {f} ^ {2} d _ {X ^ {*}} ^ {2} (\mathbf {x} ^ {0}),
$$

and hence 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {L _ {f} d _ {X ^ {*}} (\mathbf {x} ^ {0})}{\sqrt {k + 1}}. \quad \square
$$

Remark 8.14. Note that in the convergence result of Theorem 8.13 we can replace the constant $L _ { f }$ with $\operatorname* { m a x } _ { n = 0 , 1 , \ldots , k } \| f ^ { \prime } ( \mathbf { x } ^ { n } ) \|$ 

The property of the sequence generated by the projected subgradient method described in part (a) of Theorem 8.13 is known as Fej´er monotonicity. 

Definition 8.15 (Fej´er monotonicity). A sequence $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 } \subseteq \mathbb { E }$ is called Fej´er monotone w.r.t. a set $S \subseteq \mathbb { E }$ if 

$$
\left\| \mathbf {x} ^ {k + 1} - \mathbf {y} \right\| \leq \left\| \mathbf {x} ^ {k} - \mathbf {y} \right\| f o r a l l k \geq 0 a n d \mathbf {y} \in S.
$$

Since Fej´er monotonicity w.r.t. a set $S$ implies that for all $k \geq 0$ and any $\mathbf { y } \in S , \| \mathbf { x } ^ { k } - \mathbf { y } \| \leq \| \mathbf { x } ^ { 0 } - \mathbf { y } \|$ , it follows that Fej´er monotone sequences are always bounded. We will now prove that sequences which are Fej´er monotone w.r.t. sets containing their limit points are convergent. 

Theorem 8.16 (convergence under Fej´er monotonicity). Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 } \subseteq \mathbb { E }$ be a sequence, and let S be a set satisfying $D \subseteq S$ , where D is the set comprising all the limit points of $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ . If $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ is Fej´er monotone w.r.t. S, then it converges to a point in D. 

Proof. Since $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ is Fej´er monotone, it is also bounded and hence has limit points. Let x˜ be a limit point of the sequence $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ , meaning that there exists a subsequence $\{ \mathbf { x } ^ { k _ { j } } \} _ { j \geq 0 }$ such that $\mathbf { x } ^ { k _ { j } }  \tilde { \mathbf { x } }$ . Since $\tilde { \mathbf { x } } \in D \subseteq S$ , it follows by the Fej´er monotonicity w.r.t. S that for any $k \geq 0$ , 

$$
\left\| \mathbf {x} ^ {k + 1} - \tilde {\mathbf {x}} \right\| \leq \left\| \mathbf {x} ^ {k} - \tilde {\mathbf {x}} \right\|.
$$

Thus, $\{ \| \mathbf { x } ^ { k } - \tilde { \mathbf { x } } \| \} _ { k \geq 0 }$ is a nonincreasing sequence which is bounded below (by zero) and hence convergent. Since $\| \mathbf { x } ^ { k _ { j } } - \tilde { \mathbf { x } } \|  0$ as $j \to \infty$ , it follows that the whole sequence $\{ \| \mathbf { x } ^ { k } - \tilde { \mathbf { x } } \| \} _ { k \geq 0 }$ converges to zero, and consequently $\mathbf { x } ^ { k } \to \tilde { \mathbf { x } }$ as $k \to \infty$ □ 

Equipped with the last theorem, we can now prove convergence of the sequence generated by the projected subgradient method with Polyak’s stepsize rule. 

Theorem 8.17 (convergence of the sequence generated by projected subgradient with Polyak’s stepsize rule). Suppose that Assumptions 8.7 and 8.12 hold. Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the projected subgradient method with Polyak’s stepsize rule (8.13). Then $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ converges to a point in $X ^ { * }$ 

Proof. By Theorem $8 . 1 3 ( \mathrm { a } )$ , the sequence is Fej´er monotone w.r.t. $X ^ { * }$ . Therefore, by Theorem 8.16, to show convergence to a point in $X ^ { * }$ , it is enough to show that any limit point of the sequence is necessarily in $X ^ { * }$ (that is, an optimal solution of the problem). Let then $\tilde { \mathbf { x } }$ be a limit point of the sequence. Then there exists a subsequence $\{ \mathbf { x } ^ { k _ { j } } \} _ { j \geq 0 }$ converging to $\tilde { \mathbf { x } } .$ . By the closedness of $C , { \tilde { \mathbf { x } } } \in C$ . By Theorem 8.13(b), 

$$
f (\mathbf {x} ^ {k _ {j}}) \to f _ {\mathrm{opt}} \mathrm{as} j \to \infty .\tag{8.17}
$$

Since $\tilde { \mathbf { x } } \in C \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ , it follows by Theorem 2.21 that f is continuous at $\tilde { \mathbf { x } } .$ which, combined with (8.17), implies that $f ( \tilde { \mathbf { x } } ) = f _ { \mathrm { o p t } }$ , meaning that $\tilde { \mathbf { x } } \in X ^ { * }$ □ 

Part (c) of Theorem 8.13 provides an upper bound on the rate of convergence in which the sequence $\{ f _ { \mathrm { b e s t } } ^ { k } \} _ { k \geq 0 }$ converges to $f _ { \mathrm { o p t } }$ . Specifically, the result shows that the distance of $f _ { \mathrm { b e s t } } ^ { k }$ to $f _ { \mathrm { o p t } }$ is bounded above by a constant factor of $\scriptstyle { \frac { 1 } { \sqrt { k + 1 } } }$ with k being the iteration index. We will sometimes refer to it as an $^ { 6  } O ( 1 / { \sqrt { k } } )$ rate of convergence result” with a slight abuse of the “big $\mathrm { O ^ { \mathfrak { s } } }$ notation (which actually refers to asymptotic results). We can also write the rate of convergence result as a complexity result. For that, we first introduce the concept of an ε-optimal solution. A vector $\mathbf { x } \in C$ is called an ε-optimal solution of problem (8.10) if $f ( \mathbf { x } ) - f _ { \mathrm { o p t } } \leq \varepsilon .$ In complexity analysis, the following question is asked: how many iterations are required to obtain an ε-optimal solution $\mathscr { l }$ That is, how many iterations are required to obtain the condition 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \varepsilon ?\tag{8.18}
$$

Using Theorem $8 . 1 3 ( \mathrm { c ) }$ , it follows that a suficient condition for (8.18) to hold is the following inequality: C 

$$
\frac {L _ {f} d _ {X ^ {*}} (\mathbf {x} ^ {0})}{\sqrt {k + 1}} \leq \varepsilon ,\tag{8.19}
$$

which is the same as 

$$
k \geq \frac {L _ {f} ^ {2} d _ {X *} ^ {2} (\mathbf {x} ^ {0})}{\varepsilon^ {2}} - 1.
$$

Therefore, an order of $\textstyle { \frac { 1 } { \varepsilon ^ { 2 } } }$ iterations is required to obtain an ε-optimal solution. We summarize the discussion in the following theorem. 

Theorem 8.18 (complexity of projected subgradient with Polyak’s stepsize). Suppose that Assumptions 8.7 and 8.12 hold. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the projected subgradient method with Polyak’s stepsize rule (8.13). Then for any nonnegative integer k satisfying 

$$
k \geq \frac {L _ {f} ^ {2} d _ {X ^ {*}} ^ {2} (\mathbf {x} ^ {0})}{\varepsilon^ {2}} - 1,
$$

it holds that 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \varepsilon .
$$

Example 8.19. Consider the problem 

$$
\min _ {x _ {1}, x _ {2}} \left\{f (x _ {1}, x _ {2}) = | x _ {1} + 2 x _ {2} | + | 3 x _ {1} + 4 x _ {2} | \right\}.
$$

Since in this chapter the underlying spaces are Euclidean, it follows that the underlying space in this example is ${ \dot { \mathbb { R } } } ^ { 2 }$ endowed with the dot product and the $l _ { 2 } { \mathrm { - n o r m } } .$ The optimal solution of the problem is $( x _ { 1 } , x _ { 2 } ) = ( 0 , 0 )$ , and the optimal value is $f _ { \mathrm { o p t } } = 0$ . Clearly, both Assumptions 8.7 and 8.12 hold. Since $f ( \mathbf { x } ) = \| \mathbf { A x } \| _ { 1 }$ , where $\mathbf { A } = { \bigl ( } { \begin{array} { l l } { 1 } & { 2 } \\ { 3 } & { 4 } \end{array} } { \bigr ) }$ , it follows that for any $\mathbf { x } \in \mathbb { R } ^ { 2 }$ , 

$$
\partial f (\mathbf {x}) = \mathbf {A} ^ {T} \partial h (\mathbf {A x}),
$$

where $h ( \mathbf { x } ) = \| \mathbf { x } \| _ { 1 }$ . By Example 3.41, for any $\mathbf { w } \in \mathbb { R } ^ { 2 }$ 

$$
\partial h (\mathbf {w}) = \left\{\mathbf {z} \in \mathbb {R} ^ {2}: z _ {i} = \mathrm{sgn} (w _ {i}), i \in I _ {\neq} (\mathbf {w}), | z _ {j} | \leq 1, j \in I _ {0} (\mathbf {w}) \right\},
$$

where 

$$
I _ {0} (\mathbf {w}) = \{i: w _ {i} = 0 \}, I _ {\neq} (\mathbf {w}) = \{i: w _ {i} \neq 0 \}.
$$

Hence, if $\pmb { \eta } \in \partial h ( \mathbf { A x } )$ , then $\pmb { \eta } \in [ - 1 , 1 ] \times [ - 1 , 1 ]$ , and, in particular, $\| \pmb { \eta } \| _ { 2 } \le \sqrt { 2 }$ Therefore, since any $\mathbf { g } \in \partial f ( \mathbf { x } )$ can be written as $\mathbf { g } = \mathbf { A } ^ { T } \boldsymbol { \eta }$ for some $\pmb { \eta } \in \partial h ( \mathbf { A x } )$ ， we have 

$$
\| \mathbf {g} \| _ {2} = \| \mathbf {A} ^ {T} \boldsymbol {\eta} \| _ {2} \leq \| \mathbf {A} ^ {T} \| _ {2, 2} \| \boldsymbol {\eta} \| _ {2} \leq \| \mathbf {A} ^ {T} \| _ {2, 2} \cdot \sqrt {2} = 7. 7 2 8 7.
$$

We can thus choose $L _ { f } = 7 . 7 2 8 7$ 

The subgradient method update step takes the form 

$$
\binom{x _ {1} ^ {k + 1}}{x _ {2} ^ {k + 1}} = \binom{x _ {1} ^ {k}}{x _ {2} ^ {k}} - \frac {| x _ {1} ^ {k} + 2 x _ {2} ^ {k} | + | 3 x _ {1} ^ {k} + 4 x _ {2} ^ {k} |}{\| \mathbf {v} (x _ {1} ^ {k} , x _ {2} ^ {k}) \| _ {2} ^ {2}} \mathbf {v} (x _ {1} ^ {k}, x _ {2} ^ {k}),
$$

where we choose 

$$
\mathbf {v} (x _ {1}, x _ {2}) = \binom{\operatorname{sgn} (x _ {1} + 2 x _ {2}) + 3 \operatorname{sgn} (3 x _ {1} + 4 x _ {2})}{2 \operatorname{sgn} (x _ {1} + 2 x _ {2}) + 4 \operatorname{sgn} (3 x _ {1} + 4 x _ {2})} \in \partial f (x _ {1}, x _ {2}).
$$

Note that in the terminology of this book $\operatorname { s g n } ( 0 ) = 1$ (see Section 1.7.2), which dictates the choice of the subgradient among the vectors in the subdiferential set in cases where $f$ is not diferentiable at the given point. We can immediately see that there are actually only four possible choices of directions $\mathbf { v } ( x _ { 1 } , x _ { 2 } )$ depending on the two possible values of sgn $( x _ { 1 } + 2 x _ { 2 } )$ and the two possible choices of $\mathrm { s g n } ( 3 x _ { 1 } + 4 x _ { 2 } )$ . The four possible directions are 

$$
\mathbf {u} _ {1} = \binom{- 4}{- 6}, \quad \mathbf {u} _ {2} = \binom{2}{2}, \quad \mathbf {u} _ {3} = \binom{- 2}{- 2}, \quad \mathbf {u} _ {4} = \binom{4}{6}.
$$

By Remark 8.14, the constant $L _ { f }$ can be chosen as m $\operatorname { a x } _ { i } \{ \| \mathbf { u } _ { i } \| _ { 2 } \} = 7 . 2 1 1 1$ , which is a slightly better bound than 7.7287. The first 100 iterations of the method with a starting point $( 1 , 2 ) ^ { T }$ are described in Figure 8.2. Note that the sequence of function values is indeed not monotone (although convergence to $f _ { \mathrm { o p t } }$ is quite apparent) and that actually only two directions are being used by the method: $( - 2 , - 2 ) ^ { T }$ 2 $( 4 , 6 ) ^ { T }$ 

## 8.2.3 The Convex Feasibility Problem

Let $S _ { 1 } , S _ { 2 } , \ldots , S _ { m } \subseteq \mathbb { E }$ be closed and convex sets. Assume that 

$$
S \equiv \bigcap_ {i = 1} ^ {m} S _ {i} \neq \emptyset .\tag{8.20}
$$

The convex feasibility problem is the problem of finding a point x in the intersection $\cap _ { i = 1 } ^ { m } S _ { i }$ . We can formulate the problem as the following minimization problem: 

$$
\min _ {\mathbf {x}} \left\{f (\mathbf {x}) \equiv \max _ {i = 1, 2, \dots , m} d _ {S _ {i}} (\mathbf {x}) \right\}.\tag{8.21}
$$

Since we assume that the intersection is nonempty, we have that $f _ { \mathrm { o p t } } = 0$ and that the optimal set is S. Another property of $f$ is that it is Lipschitz continuous with constant 1. 

Lemma 8.20. Let $S _ { 1 } , S _ { 2 } , \ldots , S _ { m }$ be nonempty closed and convex sets. Then the function $f$ given in (8.21) is Lipschitz continuous with constant 1. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/de5a8fdcf6772ca7dbe6b69b28bfa04dd98259ab066b03b0644285d6f56d3dba.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/b5c9ecd2d569decf266ba219186dfad00c5a80c8e93044894ecdabe02cd5dc2e.jpg)


Figure 8.2. First 100 iterations of the subgradient method applied to the function $f ( x _ { 1 } , x _ { 2 } ) = | x _ { 1 } + 2 x _ { 2 } | + | 3 x _ { 1 } + 4 x _ { 2 } |$ with Polyak’s stepsize rule and starting point $( 1 , 2 ) ^ { T }$ . The $l e f t$ image describes the function values at each iteration, and the right image shows the contour lines along with the iterations. 

Proof. Let $i \in \{ 1 , 2 , \dots , m \}$ , and let $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ . Then 

$$
\begin{array}{l l} d _ {S _ {i}} (\mathbf {x}) = \| \mathbf {x} - P _ {S _ {i}} (\mathbf {x}) \| \\ \leq \| \mathbf {x} - P _ {S _ {i}} (\mathbf {y}) \| & \left[ \| \mathbf {x} - P _ {S _ {i}} (\mathbf {x}) \| = \operatorname * {a r g m i n} _ {\mathbf {v} \in S _ {i}} \| \mathbf {x} - \mathbf {v} \| \right] \\ \leq \| \mathbf {x} - \mathbf {y} \| + \| \mathbf {y} - P _ {S _ {i}} (\mathbf {y}) \| & \text {[triangle inequality]} \\ = \| \mathbf {x} - \mathbf {y} \| + d _ {S _ {i}} (\mathbf {y}). & \left[ d _ {S _ {i}} (\mathbf {y}) = \| \mathbf {y} - P _ {S _ {i}} (\mathbf {y}) \| \right] \end{array}
$$

Thus, 

$$
d _ {S _ {i}} (\mathbf {x}) - d _ {S _ {i}} (\mathbf {y}) \leq \| \mathbf {x} - \mathbf {y} \|.\tag{8.22}
$$

Replacing the roles of x and y, we obtain that 

$$
d _ {S _ {i}} (\mathbf {y}) - d _ {S _ {i}} (\mathbf {x}) \leq \| \mathbf {x} - \mathbf {y} \|,
$$

which, combined with (8.22), yields the inequality 

$$
| d _ {S _ {i}} (\mathbf {x}) - d _ {S _ {i}} (\mathbf {y}) | \leq \| \mathbf {x} - \mathbf {y} \|.\tag{8.23}
$$

Finally, for any $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ 8 

$$
| f (\mathbf {x}) - f (\mathbf {y}) | = \left| \max _ {i = 1, 2, \dots , m} d _ {S _ {i}} (\mathbf {x}) - \max _ {i = 1, 2, \dots , m} d _ {S _ {i}} (\mathbf {y}) \right| = | | \left\| \mathbf {v} _ {\mathbf {x}} \right\| _ {\infty} - \| \mathbf {v} _ {\mathbf {y}} \| _ {\infty} |,\tag{8.24}
$$

Greedy Projection Algorithm
Input: m nonempty closed and convex sets $S_{1}, S_{2}, \ldots, S_{m}$ .
Initialization: pick $x^{0} \in E$ .
General step: for any $k = 0, 1, 2, \ldots$ , execute the step $\mathbf{x}^{k+1} = P_{S_{i_{k}}}(\mathbf{x}^{k})$ ,
where $i_{k} \in \arg\max_{i=1,2,\ldots,m} d_{S_{i}}(\mathbf{x}^{k})$ . 

where $\mathbf { v _ { x } } = ( d _ { S _ { i } } ( \mathbf { x } ) ) _ { i = 1 } ^ { m } \in \mathbb { R } ^ { m }$ and $\mathbf { v } _ { \mathbf { y } } = ( d _ { S _ { i } } ( \mathbf { y } ) ) _ { i = 1 } ^ { m } \in \mathbb { R } ^ { m }$ . Using the triangle inequality for norms, we can continue (8.24) and obtain 

$$
\begin{array}{r c l} | f (\mathbf {x}) - f (\mathbf {y}) | & \leq & \| \left| \mathbf {v} _ {\mathbf {x}} \right| \| _ {\infty} - \| \left| \mathbf {v} _ {\mathbf {y}} \right| \| _ {\infty} | \\ & \leq & \| \mathbf {v} _ {\mathbf {x}} - \mathbf {v} _ {\mathbf {y}} \| _ {\infty} \\ & = & \max _ {i = 1, 2, \ldots , m} | d _ {S _ {i}} (\mathbf {x}) - d _ {S _ {i}} (\mathbf {y}) | \\ & \overset {(8. 2 3)} {\leq} & \| \mathbf {x} - \mathbf {y} \|. \quad \square \end{array}
$$

Let us write explicitly the projected subgradient method with Polyak’s stepsize rule as applied to problem (8.21). The method starts with an arbitrary $\mathbf { x } ^ { 0 } \in \mathbb { E }$ . If the kth iteration satisfies $\mathbf { x } ^ { k } \in S$ , then we can pick $f ^ { \prime } ( \mathbf { x } ^ { k } ) = \mathbf { 0 }$ and hence $\mathbf { x } ^ { k + 1 } = \mathbf { x } ^ { k }$ Otherwise, we take a step toward minus of the subgradient with Polyak’s stepsize. By Theorem 3.50, to compute a subgradient of the objective function at the kth iterate, we can use the following procedure: 

(i) compute $i _ { k } \in \mathrm { a r g m a x } _ { i = 1 , 2 , \dots , m } d _ { S _ { i } } ( \mathbf { x } ^ { k } )$ 

(ii) take any $\mathbf { g } ^ { k } \in \partial d _ { S _ { i _ { k } } } ( \mathbf { x } ^ { k } )$ 

By Example 3.49, we can (and actually must) choose the subgradient in $\partial d _ { S _ { i _ { k } } } ( \mathbf { x } ^ { k } )$ as $\begin{array} { r } { \mathbf { g } ^ { k } = \frac { \mathbf { x } ^ { k } - P _ { S _ { i } } ( \mathbf { x } ^ { k } ) } { d _ { S _ { i } } ( \mathbf { x } ^ { k } ) } } \end{array}$ , and in this case the update step becomes 

$$
\begin{array}{r l} & {\mathbf {x} ^ {k + 1} = \mathbf {x} ^ {k} - \frac {d _ {S _ {i _ {k}}} (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}}{\| \mathbf {g} ^ {k} \| ^ {2}} \cdot \frac {\mathbf {x} ^ {k} - P _ {S _ {i _ {k}}} (\mathbf {x} ^ {k})}{d _ {S _ {i _ {k}}} (\mathbf {x} ^ {k})}} \\ & {\qquad = \mathbf {x} ^ {k} - d _ {S _ {i _ {k}}} (\mathbf {x} ^ {k}) \frac {\mathbf {x} ^ {k} - P _ {S _ {i _ {k}}} (\mathbf {x} ^ {k})}{d _ {S _ {i _ {k}}} (\mathbf {x} ^ {k})}} \\ & {\qquad = P _ {S _ {i _ {k}}} (\mathbf {x} ^ {k}),} \end{array}
$$

where we used in the above the facts that $f _ { \mathrm { o p t } } = 0$ and $\| \mathbf { g } ^ { k } \| = 1$ . What we actually obtained is the greedy projection algorithm, which at each iteration projects the current iterate $\mathbf { x } ^ { k }$ onto the farthest set among $S _ { 1 } , S _ { 2 } , \ldots , S _ { m }$ . The algorithm is summarized below. 

We can invoke Theorems 8.13 and 8.17 to obtain the following convergence result of the algorithm. 

Theorem 8.21 (convergence of the greedy projection algorithm). Let $S _ { 1 } , S _ { 2 } , \ldots , S _ { m } \subseteq \mathbb { E }$ be closed and convex sets such that $\begin{array} { r } { S \equiv \bigcap _ { i = 1 } ^ { m } S _ { i } \neq \emptyset } \end{array}$ . Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the greedy projection algorithm. 

(a) For any $k \geq 0$ 

$$
\min _ {n = 0, 1, 2, \dots , k} \left\{\max _ {i = 1, 2, \dots , m} d (\mathbf {x} ^ {n}, S _ {i}) \right\} \leq \frac {d _ {S} (\mathbf {x} ^ {0})}{\sqrt {k + 1}}.\tag{8.25}
$$

(b) There exists ${ \mathbf { x } } ^ { * } \in S$ such that $\mathbf { x } ^ { k } \to \mathbf { x } ^ { * }$ as $k \to \infty$ 

Proof. To prove part (a), define $f ( \mathbf { x } ) \equiv \mathrm { m a x } _ { i = 1 , 2 , \ldots , m } d ( \mathbf { x } , S _ { i } )$ and $C = \mathbb { E }$ . Then the optimal set of the problem 

$$
\min \{f (\mathbf {x}): \mathbf {x} \in C \}
$$

is $X ^ { * } = S$ . Assumption 8.7 is satisfied since f is proper closed and convex and $C = \mathbb { E }$ is obviously nonempty closed and convex and contained in int $( \operatorname { d o m } ( f ) ) = \mathbb { E }$ The optimal set $X ^ { * } = S$ is nonempty by the assumption in the premise of the theorem. Assumption 8.12 is satisfied with $L _ { f } = 1$ by Lemma 8.20 and Theorem 3.61. Therefore, all the assumptions of Theorem 8.13 are satisfied, and hence, since the greedy projection algorithm is the same as the projected subgradient method with Polyak’s stepsize rule, the result (8.25) holds, as it is exactly part (c) of Theorem 8.13. Part (b) follows by invoking Theorem 8.17. 

When $m = 2 .$ , the algorithm amounts to the alternating projection method, which is described below. 

Alternating Projection Method
Input: two nonempty closed and convex sets $S_{1}, S_{2}$ .
Initialization: pick $x^{0} \in S_{2}$ arbitrarily.
General step: for any $k = 0, 1, 2, \ldots$ , execute the following step: $\mathbf{x}^{k+1} = P_{S_{2}}(P_{S_{1}}(\mathbf{x}^{k}))$ . 

If $S _ { 1 } \cap S _ { 2 } \neq \emptyset$ , by Theorem 8.21, the sequence generated by the alternating projection method converges to a point in $S _ { 1 } \cap S _ { 2 }$ 

Corollary 8.22 (convergence of alternating projection). Let $S _ { 1 } , S _ { 2 }$ be closed and convex sets such that $S \equiv S _ { 1 } \cap S _ { 2 } \neq \emptyset$ . Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the alternating projection method with initial point $\bar { \mathbf { x } } ^ { 0 } \in S _ { 2 }$ . Then 

(a) for any $k \geq 0$ 

$$
\min _ {n = 0, 1, 2, \dots , k} d (\mathbf {x} ^ {n}, S _ {1}) \leq \frac {d _ {S} (\mathbf {x} ^ {0})}{\sqrt {k + 1}};
$$

(b) there exists ${ \mathbf { x } } ^ { * } \in S$ such that $\mathbf { x } ^ { k } \to \mathbf { x } ^ { * }$ as $k \to \infty$ 

Example 8.23 (solution of linear feasibility problems). Consider the following system of linear equalities and inequalities: 

$$
\mathbf {A} \mathbf {x} = \mathbf {b}, \mathbf {x} \geq \mathbf {0},\tag{8.26}
$$

where $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ has full row rank and b $\in \mathbb { R } ^ { m }$ . The system (8.26) is one of the standard forms of feasible sets of linear programming problems. One way to solve the problem of finding a solution to (8.26) is by employing the alternating projection method. Define 

$$
S _ {1} = \{\mathbf {x} \in \mathbb {R} ^ {n}: \mathbf {A x} = \mathbf {b} \}, \quad S _ {2} = \mathbb {R} _ {+} ^ {n}.
$$

The projections on $S _ { 1 }$ and $S _ { 2 }$ have analytic expressions (see Lemma 6.26): 

$$
P _ {S _ {1}} (\mathbf {x}) = \mathbf {x} - \mathbf {A} ^ {T} (\mathbf {A A} ^ {T}) ^ {- 1} (\mathbf {A x} - \mathbf {b}), P _ {S _ {2}} (\mathbf {x}) = [ \mathbf {x} ] _ {+}.
$$

The alternating projection method for finding a solution to (8.26) takes the following form: 

$$
\begin{array}{l} \text { Algorithm   1 } \\ \bullet \text { Initialization:   pick   } \mathbf {x} ^ {0} \in \mathbb {R} _ {+} ^ {n}. \\ \bullet \text { General   step   } (k \geq 0) \colon \mathbf {x} ^ {k + 1} = \left[ \mathbf {x} ^ {k} - \mathbf {A} ^ {T} (\mathbf {A A} ^ {T}) ^ {- 1} (\mathbf {A x} ^ {k} - \mathbf {b}) \right] _ {+}. \end{array}
$$

The general step of the above scheme involves the computation of the expression $( \mathbf { A } \mathbf { A } ^ { \bar { T } } ) ^ { - 1 } ( \mathbf { A } \mathbf { x } ^ { k } - \mathbf { b } )$ , which requires the computation of the matrix $\mathbf { A A } ^ { T }$ , as well as the solution of the linear system $( \mathbf { A } \mathbf { A } ^ { T } ) \hat { \mathbf { z } } = \mathbf { A } \mathbf { x } ^ { k } - \mathbf { b }$ . In cases when these computations are too demanding $( \mathrm { e . g . }$ , when the dimension is large), we can employ a diferent projection algorithm that avoids the necessity of solving a linear system. Specifically, denoting the ith row of A by $\mathbf { a } _ { i } ^ { T }$ and defining 

$$
T _ {i} = \{\mathbf {x} \in \mathbb {R} ^ {n}: \mathbf {a} _ {i} ^ {T} \mathbf {x} = b _ {i} \}, i = 1, 2, \ldots , m, T _ {m + 1} = \mathbb {R} _ {+} ^ {n},
$$

we obtain that finding a solution to (8.26) is the same as finding a point in the intersection $\cap _ { i = 1 } ^ { m + 1 } T _ { i }$ . Note that (see Lemma 6.26) 

$$
P _ {T _ {i}} (\mathbf {x}) = \mathbf {x} - \frac {\mathbf {a} _ {i} ^ {T} \mathbf {x} - b _ {i}}{\| \mathbf {a} _ {i} \| _ {2} ^ {2}} \mathbf {a} _ {i}, i = 1, 2, \ldots , m.
$$

Hence, 

$$
d _ {T _ {i}} (\mathbf {x}) = \| \mathbf {x} - P _ {T _ {i}} (\mathbf {x}) \| = \frac {| \mathbf {a} _ {i} ^ {T} \mathbf {x} - b _ {i} |}{\| \mathbf {a} _ {i} \| _ {2}}.
$$

We can now invoke the greedy projection method that has the following form: 

Algorithm 2
- Initialization: pick $\mathbf{x}^0\in \mathbb{E}$ .
- General step ( $k = 0,1,\ldots$ ):
    - compute $i_k\in \mathrm{argmax}_{i = 1,2,\dots,m}\frac{|\mathbf{a}_i^T\mathbf{x}^k - b_i|}{\|\mathbf{a}_i\|_2}$ .
    - if $\frac{|\mathbf{a}_{i_k}^T\mathbf{x}^k - b_{i_k}|}{\|\mathbf{a}_{i_k}\|_2} >\|\mathbf{x}^k -[\mathbf{x}^k ]_+\| _2$ , then $\mathbf{x}^{k + 1} = \mathbf{x}^k -\frac{\mathbf{a}_{i_k}^T\mathbf{x}^k - b_{i_k}}{\|\mathbf{a}_{i_k}\|_2^2}\mathbf{a}_{i_k}$ .
    else, $\mathbf{x}^{k + 1} = [\mathbf{x}^k ]_+$ . 

Algorithm 2 is simpler than Algorithm 1 in the sense that it requires much less operations per iteration. However, simplicity has its cost. Consider, for example, the instance 

$$
\mathbf {A} = \left( \begin{array}{c c c c} 0 & 6 & - 7 & 1 \\ - 1 & 2 & 1 0 & - 1 \end{array} \right), \quad \mathbf {b} = \binom{0}{1 0}.
$$

Figure 8.3 shows the constraint violation of the two sequences generated by the two algorithms initialized with the zeros vector in the first 20 iterations. Obviously, in this case, Algorithm 1 (alternating projection) reached substantially better accuracies than Algorithm 2 (greedy projection). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/b438e8a0abf92a91b813823f1b2d788917376a91aa1ec38dfad585a65d4b061c.jpg)



Figure 8.3. Constraints violation of alternating and greedy projection methods. Here $\begin{array} { r } { f ( { \bf x } ) { \bf \phi } = { \bf \phi } \operatorname* { m a x } \left\{ \frac { | { \bf a } _ { 1 } ^ { T } { \bf x } - b _ { 1 } | } { | | { \bf a } _ { 1 } | | _ { 2 } } , \frac { | { \bf a } _ { 2 } ^ { T } { \bf x } - b _ { 2 } | } { | | { \bf a } _ { 2 } | | _ { 2 } } , | | { \bf x } - [ { \bf x } ] _ { + } | | _ { 2 } \right\} } \end{array}$ , where ${ \bf a } _ { 1 } ^ { T } { \bf \Psi } = { \bf \Psi }$ $( 0 , 6 , - 7 , 1 ) , \mathbf { a } _ { 2 } ^ { T } = ( - 1 , 2 , 1 0 , - 1 )$ , and ${ \bf b } = ( 0 , 1 0 ) ^ { T }$


## 8.2.4 Projected Subgradient with Dynamic Stepsizes

Polyak’s stepsize is optimal in the sense that it minimizes the upper bound given in the fundamental inequality (8.12). However, a major disadvantage of this rule is that usually the optimal value $f _ { \mathrm { o p t } }$ is unknown, and in these (frequent) cases, the stepsize is incomputable. In this section we will show how to find computable stepsize rules that still maintain the $O ( 1 / \sqrt { k } )$ rate of convergence result of the projected subgradient method. Theorem 8.25 below describes a simple condition on the stepsizes under which convergence of $f _ { \mathrm { b e s t } } ^ { k }$ to $f _ { \mathrm { o p t } }$ is guaranteed. The result uses the following technical lemma. 

Lemma 8.24. Suppose that Assumption 8.7 holds. Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the projected subgradient method with positive stepsizes $\{ t _ { k } \} _ { k \geq 0 }$ . Then for any $\mathbf { x } ^ { * } \in X ^ { * }$ and nonnegative integer k, 

$$
\sum_ {n = 0} ^ {k} t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) \leq \frac {1}{2} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} + \frac {1}{2} \sum_ {n = 0} ^ {k} t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| ^ {2}.\tag{8.27}
$$

Proof. By Lemma 8.11, for any $n \geq 0$ and $\mathbf { x } ^ { * } \in X ^ { * }$ 2 

$$
\frac {1}{2} \| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} \leq \frac {1}{2} \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) + \frac {t _ {n} ^ {2}}{2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| ^ {2}.
$$

Summing the above inequality over $n = 0 , 1 , \ldots , k$ and arranging terms yields the following inequality: 

$$
\sum_ {n = 0} ^ {k} t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) \leq \frac {1}{2} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} - \frac {1}{2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2} + \sum_ {n = 0} ^ {k} \frac {t _ {n} ^ {2}}{2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| ^ {2}
$$

$$
\leq \frac {1}{2} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} + \frac {1}{2} \sum_ {n = 0} ^ {k} t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| ^ {2}. \quad \square
$$

Theorem 8.25 (stepsize conditions warranting convergence of projected subgradient). Suppose that Assumptions 8.7 and 8.12 hold. Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the projected subgradient method with positive stepsizes $\{ t _ { k } \} _ { k \geq 0 }$ If 

$$
\frac {\sum_ {n = 0} ^ {k} t _ {n} ^ {2}}{\sum_ {n = 0} ^ {k} t _ {n}} \rightarrow 0 a s k \rightarrow \infty ,\tag{8.28}
$$

then 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \rightarrow 0 a s k \rightarrow \infty ,\tag{8.29}
$$

where $\{ f _ { \mathrm { b e s t } } ^ { k } \} _ { k \geq 0 }$ is the sequence of best achieved values defined in (8.11). 

Proof. Let $L _ { f }$ be a constant for which $\| \mathbf { g } \| \leq L _ { f }$ for any $\mathbf { g } \in \partial f ( \mathbf { x } ) , \mathbf { x } \in C$ whose existence is warranted by Assumption 8.12. Employing Lemma 8.24 and using the inequalities $\| f ^ { \prime } ( \mathbf { x } ^ { n } ) \| \leq L _ { f }$ and $f ( \mathbf { x } ^ { n } ) \geq f _ { \mathrm { b e s t } } ^ { k }$ for $n \leq k$ , we obtain 

$$
\left(\sum_ {n = 0} ^ {k} t _ {n}\right) \left(f _ {\text {best}} ^ {k} - f _ {\text {opt}}\right) \leq \frac {1}{2} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} + \frac {L _ {f} ^ {2}}{2} \sum_ {n = 0} ^ {k} t _ {n} ^ {2}.
$$

Therefore, 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {1}{2} \frac {\| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}}{\sum_ {n = 0} ^ {k} t _ {n}} + \frac {L _ {f} ^ {2}}{2} \frac {\sum_ {n = 0} ^ {k} t _ {n} ^ {2}}{\sum_ {n = 0} ^ {k} t _ {n}}.
$$

The result (8.29) now follows by (8.28), and the fact that (8.28) implies the limit $\textstyle \sum _ { n = 0 } ^ { k } t _ { n } \to \infty { \mathrm { ~ a s ~ } } k \to \infty$ □ 

By Theorem 8.25, we can pick, for example, the stepsizes as $\begin{array} { r } { t _ { k } = \frac { 1 } { \sqrt { k + 1 } } } \end{array}$ , and convergence of function values to $f _ { \mathrm { o p t } }$ will be guaranteed since $\scriptstyle \sum _ { n = 0 } ^ { k } { \frac { 1 } { \sqrt { n + 1 } } }$ is of the order of $\sqrt { k }$ and $\scriptstyle \sum _ { n = 0 } ^ { k } { \frac { 1 } { n + 1 } }$ is of the order of $\log ( k )$ . We will analyze the conver-gence rate of the projected subgradient method when the stepsizes are chosen as $\begin{array} { r } { t _ { k } = \frac { 1 } { \| f ^ { \prime } ( \mathbf { x } ^ { k } ) \| \sqrt { k + 1 } } } \end{array}$ in Theorem 8.28 below. Note that in addition to proving the limit $f _ { \mathrm { b e s t } } ^ { k }  f _ { \mathrm { o p t } }$ , we will further show that the function values of a certain sequence of averages also converges to the optimal value. Such a result is called an ergodic convergence result. 

To prove the result, we will be need to upper and lower bound sums of sequences of real numbers. For that, we will use the following technical lemma from calculus. 

Lemma 8.26. Let $f : [ a - 1 , b + 1 ] \to \mathbb { R }$ be a continuous nonincreasing function over $[ a - 1 , b + 1 ]$ , where a and b are integer numbers satisfying $a \leq b$ . Then 

$$
\int_ {a} ^ {b + 1} f (t) d t \leq f (a) + f (a + 1) + \dots + f (b) \leq \int_ {a - 1} ^ {b} f (t) d t.
$$

Using Lemma 8.26, we can prove the following lemma that will be useful in proving Theorem 8.28, as well as additional results in what follows. 

## Lemma 8.27. Let $D \in \mathbb { R }$ . Then

(a) for any $k \geq 1$ 

$$
\frac {D + \sum_ {n = 0} ^ {k} \frac {1}{n + 1}}{\sum_ {n = 0} ^ {k} \frac {1}{\sqrt {n + 1}}} \leq \frac {D + 1 + \log (k + 1)}{\sqrt {k + 1}};\tag{8.30}
$$

(b) for any $k \geq 2$ 

$$
\frac {D + \sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{n + 1}}{\sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{\sqrt {n + 1}}} \leq \frac {4 (D + \log (3))}{\sqrt {k + 2}}.\tag{8.31}
$$

Proof. (a) Using Lemma 8.26, we obtain the following inequalities: 

$$
\sum_ {n = 0} ^ {k} \frac {1}{n + 1} = 1 + \sum_ {n = 1} ^ {k} \frac {1}{n + 1} \leq 1 + \int_ {0} ^ {k} \frac {1}{x + 1} d x = 1 + \log (k + 1),\tag{8.32}
$$

$$
\sum_ {n = 0} ^ {k} \frac {1}{\sqrt {n + 1}} \geq \int_ {0} ^ {k + 1} \frac {1}{\sqrt {x + 1}} d x = 2 \sqrt {k + 2} - 2 \geq \sqrt {k + 1},\tag{8.33}
$$

where the last inequality holds for all $k \geq 1$ . The result (8.30) now follows immediately from (8.32) and (8.33). 

(b) Using Lemma 8.26, we obtain the following inequalities for any $k \geq 2 \colon$ 

$$
\begin{array}{l} \sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{n + 1} \leq \int_ {\lceil k / 2 \rceil - 1} ^ {k} \frac {d t}{t + 1} = \log (k + 1) - \log (\lceil k / 2 \rceil) \\ \qquad = \log \left(\frac {k + 1}{\lceil 0 . 5 k \rceil}\right) \leq \log \left(\frac {k + 1}{0 . 5 k}\right) = \log \left(2 + \frac {2}{k}\right) \\ \qquad \leq \log (3) \end{array}\tag{8.34}
$$

and 

$$
\begin{array}{l} \sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{\sqrt {n + 1}} \geq \int_ {\lceil k / 2 \rceil} ^ {k + 1} \frac {d t}{\sqrt {t + 1}} = 2 \sqrt {k + 2} - 2 \sqrt {\lceil k / 2 \rceil + 1} \\ \qquad \qquad \qquad \qquad \qquad \geq 2 \sqrt {k + 2} - 2 \sqrt {k / 2 + 2} = \frac {4 (k + 2) - 4 (0 . 5 k + 2)}{2 \sqrt {k + 2} + 2 \sqrt {0 . 5 k + 2}} \\ \qquad \qquad \qquad \qquad = \frac {k}{\sqrt {k + 2} + \sqrt {0 . 5 k + 2}} \geq \frac {k}{2 \sqrt {k + 2}} \\ \qquad \qquad \qquad \qquad \geq \frac {1}{4} \sqrt {k + 2}, \end{array}\tag{8.35}
$$

where the last inequality holds since $k \geq 2$ . The result (8.31) now follows by combining (8.34) and (8.35). 

We are now ready to prove the convergence result. 

Theorem 8.28 $( O ( \log ( k ) / \sqrt { k } )$ rate of convergence of projected subgradient). Suppose that Assumptions 8.7 and 8.12 hold. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the projected subgradient method with stepsizes $\begin{array} { r } { t _ { k } = \frac { 1 } { \| f ^ { \prime } ( \mathbf { x } ^ { k } ) \| \sqrt { k + 1 } } } \end{array}$ if $f ^ { \prime } ( \mathbf { x } ^ { k } ) \neq 0$ and $\begin{array} { r } { t _ { k } = \frac { 1 } { L _ { f } } } \end{array}$ otherwise. Then 

(a) for any $k \geq 1$ 

$$
f _ {\text { best }} ^ {k} - f _ {\text { opt }} \leq \frac {L _ {f}}{2} \frac {\| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} + 1 + \log (k + 1)}{\sqrt {k + 1}},
$$

where $\{ f _ { \mathrm { b e s t } } ^ { k } \} _ { k \geq 0 }$ is the sequence of best achieved values defined in (8.11); 

(b) for any $k \geq 1$ 

$$
f (\mathbf {x} ^ {(k)}) - f _ {\mathrm{opt}} \leq \frac {L _ {f}}{2} \frac {\| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} + 1 + \log (k + 1)}{\sqrt {k + 1}},
$$

where 

$$
\mathbf {x} ^ {(k)} = \frac {1}{\sum_ {n = 0} ^ {k} t _ {n}} \sum_ {n = 0} ^ {k} t _ {n} \mathbf {x} ^ {n}.
$$

Proof. Using (8.27) along with the inequality $f ( \mathbf { x } ^ { n } ) \geq f _ { \mathrm { b e s t } } ^ { k }$ for any $n = 0 , 1 , 2 , \ldots , k .$ 2 we obtain 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {1}{2} \frac {\| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} + \sum_ {n = 0} ^ {k} t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| ^ {2}}{\sum_ {n = 0} ^ {k} t _ {n}}.\tag{8.36}
$$

Alternatively, by Jensen’s inequality 

$$
f (\mathbf {x} ^ {(k)}) \leq \frac {1}{\sum_ {n = 0} ^ {k} t _ {n}} \sum_ {n = 0} ^ {k} t _ {n} f (\mathbf {x} ^ {n}),
$$

which, along with (8.27), yields 

$$
f (\mathbf {x} ^ {(k)}) - f _ {\mathrm{opt}} \leq \frac {1}{2} \frac {\| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} + \sum_ {n = 0} ^ {k} t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| ^ {2}}{\sum_ {n = 0} ^ {k} t _ {n}}.\tag{8.37}
$$

Therefore, combining (8.36) and (8.37), we have 

$$
\max \{f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}}, f (\mathbf {x} ^ {(k)}) - f _ {\mathrm{opt}} \} \leq \frac {1}{2} \frac {\| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} + \sum_ {n = 0} ^ {k} t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| ^ {2}}{\sum_ {n = 0} ^ {k} t _ {n}}.
$$

By the definition of $\begin{array} { r } { t _ { n } , t _ { n } ^ { 2 } \| f ^ { \prime } ( \mathbf { x } ^ { n } ) \| ^ { 2 } \leq \frac { 1 } { n + 1 } } \end{array}$ (satisfied as equality when $f ^ { \prime } ( \mathbf { x } ^ { n } ) \neq \mathbf { 0 }$ and as a strict inequality when $f ^ { \prime } ( \mathbf { x } ^ { n } ) \stackrel { \cdots } { = } \bar { \mathbf { 0 } } )$ ; in addition, since $\| f ^ { \prime } ( \mathbf { x } ^ { n } ) \| \leq L _ { f }$ , we have $\begin{array} { r } { t _ { n } \geq \frac { 1 } { L _ { f } \sqrt { n + 1 } } } \end{array}$ . Therefore, 

$$
\max \{f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}}, f (\mathbf {x} ^ {(k)}) - f _ {\mathrm{opt}} \} \leq \frac {L _ {f}}{2} \frac {\| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} + \sum_ {n = 0} ^ {k} \frac {1}{n + 1}}{\sum_ {n = 0} ^ {k} \frac {1}{\sqrt {n + 1}}}.\tag{8.38}
$$

Invoking Lemma 8.27(a) with $D = \| \mathbf { x } ^ { 0 } - \mathbf { x } ^ { * } \| ^ { 2 }$ implies the inequality 

$$
\max \left\{f _ {\text { best }} ^ {k} - f _ {\text { opt }}, f (\mathbf {x} ^ {(k)}) - f _ {\text { opt }} \right\} \leq \frac {L _ {f}}{2} \frac {\| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} + 1 + \log (k + 1)}{\sqrt {k + 1}},
$$

which is equivalent to the validity of the two claims (a) and (b). 

Remark 8.29. The sequence of averages $\mathbf { x } ^ { ( k ) }$ as defined in Theorem 8.28 can be computed in an adaptive way by noting that the following simple recursion relation holds: 

$$
\mathbf {x} ^ {(k + 1)} = \frac {T _ {k}}{T _ {k + 1}} \mathbf {x} ^ {(k)} + \frac {t _ {k + 1}}{T _ {k + 1}} \mathbf {x} ^ {k + 1},
$$

where $\begin{array} { r } { T _ { k } \equiv \sum _ { n = 0 } ^ { k } t _ { n } } \end{array}$ can be computed by the obvious recursion relation $T _ { k + 1 } =$ $T _ { k } + t _ { k + 1 }$ 

The $O ( \log ( k ) / \sqrt { k } )$ rate of convergence proven in Theorem 8.28 is worse than the $O ( 1 / \sqrt { k } )$ rate established in Theorem 8.13 for the version of the projected subgradient method with Polyak’s stepsize. It is possible to prove an $O ( 1 / \sqrt { k } )$ rate of convergence if we assume in addition that the feasible set C is compact. Note that by Theorem 3.16, the compactness of C implies the validity of Assumption 8.12, but we will nonetheless explicitly state it in the following result. 

Theorem 8.30 $( O ( 1 / { \sqrt { k } } )$ rate of convergence of projected subgradient). Suppose that Assumptions 8.7 and 8.12 hold and assume that C is compact. Let Θ be an upper bound on the half-squared diameter of C: 

$$
\Theta \geq \max _ {\mathbf {x}, \mathbf {y} \in C} \frac {1}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.
$$

Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the projected subgradient method with stepsizes chosen as either 

$$
t _ {k} = \frac {\sqrt {2 \Theta}}{L _ {f} \sqrt {k + 1}}\tag{8.39}
$$

or 

$$
t _ {k} = \left\{ \begin{array}{l l} \frac {\sqrt {2 \Theta}}{\| f ^ {\prime} (\mathbf {x} ^ {k}) \| \sqrt {k + 1}}, & f ^ {\prime} (\mathbf {x} ^ {k}) \neq \mathbf {0}, \\ \frac {\sqrt {2 \Theta}}{L _ {f} \sqrt {k + 1}}, & f ^ {\prime} (\mathbf {x} ^ {k}) = \mathbf {0}. \end{array} \right.\tag{8.40}
$$

Then for all $k \geq 2$ 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {\delta L _ {f} \sqrt {2 \Theta}}{\sqrt {k + 2}},
$$

where $\delta = 2 ( 1 + \log ( 3 ) )$ and $f _ { \mathrm { b e s t } } ^ { k }$ is the sequence of best achieved values defined in (8.11). 

Proof. By Lemma 8.11, for any $n \geq 0$ 

$$
\frac {1}{2} \| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} \leq \frac {1}{2} \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) + \frac {t _ {n} ^ {2}}{2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| ^ {2}.
$$

Summing the above inequality over $n = \lceil k / 2 \rceil , \lceil k / 2 \rceil + 1 , . . . , k$ , we obtain 

$$
\sum_ {n = \lceil k / 2 \rceil} ^ {k} t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) \leq \frac {1}{2} \| \mathbf {x} ^ {\lceil k / 2 \rceil} - \mathbf {x} ^ {*} \| ^ {2} - \frac {1}{2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2} + \sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {t _ {n} ^ {2}}{2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| ^ {2}
$$

$$
\leq \Theta + \sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {t _ {n} ^ {2}}{2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| ^ {2}
$$

$$
\leq \Theta + \Theta \sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{n + 1},\tag{8.41}
$$

where the last inequality is due to the fact that in either of the definitions of the stepsizes (8.39), (8.40), $\begin{array} { r } { \dot { t } _ { n } ^ { 2 } \| f ^ { \prime } ( \mathbf { x } ^ { n } ) \| ^ { 2 } \leq \frac { 2 \Theta } { n + 1 } } \end{array}$ 

Since $\begin{array} { r } { t _ { n } \ge \frac { \sqrt { 2 \Theta } } { L _ { f } \sqrt { n + 1 } } } \end{array}$ and $f ( \mathbf { x } ^ { n } ) \geq f _ { \mathrm { b e s t } } ^ { k }$ for all $n \leq k$ , it follows that 

$$
\sum_ {n = \lceil k / 2 \rceil} ^ {k} t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) \geq \left(\sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {\sqrt {2 \Theta}}{L _ {f} \sqrt {n + 1}}\right) (f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}}).\tag{8.42}
$$

Therefore, combining (8.41) and (8.42) yields 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {L _ {f} \sqrt {\Theta}}{\sqrt {2}} \frac {1 + \sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{n + 1}}{\sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{\sqrt {n + 1}}},\tag{8.43}
$$

which, combined with Lemma 8.27(b), yields the desired result. 

## 8.2.5 The Strongly Convex Case<sup>43</sup>

We will now show that if f is in addition strongly convex, then the $O ( 1 / \sqrt { k } )$ rate of convergence result can be improved to a rate of $O ( 1 / k )$ . The stepsizes used in order to achieve this improved rate diminish at an order of $1 / k .$ . We will also use the growth property of strongly convex functions described in Theorem 5.25(b) in order to show a result on the rate of convergence of the sequence $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ to an optimal solution. 

Theorem 8.31 $( O ( 1 / k )$ rate of convergence of projected subgradient for strongly convex functions). Suppose that Assumptions 8.7 and 8.12 hold. Assume in addition that f is σ-strongly convex for some $\sigma > 0$ , and let $\mathbf { x } ^ { * }$ be its unique minimizer. Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the projected subgradient method with stepsize $\begin{array} { r } { t _ { k } = \frac { 2 } { \sigma ( k + 1 ) } } \end{array}$ 

(a) Let $\{ f _ { \mathrm { b e s t } } ^ { k } \} _ { k \geq 0 }$ be the sequence of best achieved values defined in (8.11). Then for any $k \geq 0$ 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {2 L _ {f} ^ {2}}{\sigma (k + 1)}.\tag{8.44}
$$

In addition, 

$$
\| \mathbf {x} ^ {i _ {k}} - \mathbf {x} ^ {*} \| \leq \frac {2 L _ {f}}{\sigma \sqrt {k + 1}},\tag{8.45}
$$

where $i _ { k } \in \mathrm { a r g m i n } _ { i = 0 , 1 , \dots , k } f ( \mathbf { x } ^ { i } )$ 

(b) Define the sequence of averages: 

$$
\mathbf {x} ^ {(k)} = \sum_ {n = 0} ^ {k} \alpha_ {n} ^ {k} \mathbf {x} ^ {n},
$$

where $\begin{array} { r } { \alpha _ { n } ^ { k } \equiv \frac { 2 n } { k ( k + 1 ) } } \end{array}$ . Then for all $k \geq 0$ 

$$
f (\mathbf {x} ^ {(k)}) - f _ {\mathrm{opt}} \leq \frac {2 L _ {f} ^ {2}}{\sigma (k + 1)}.\tag{8.46}
$$

In addition, 

$$
\| \mathbf {x} ^ {(k)} - \mathbf {x} ^ {*} \| \leq \frac {2 L _ {f}}{\sigma \sqrt {k + 1}}.\tag{8.47}
$$

Proof. (a) Repeating the arguments in the proof of Lemma 8.11, we can write for any $n \geq 0$ 

$$
\begin{array}{r l} & {\| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} = \| P _ {C} (\mathbf {x} ^ {n} - t _ {n} f ^ {\prime} (\mathbf {x} ^ {n})) - P _ {C} (\mathbf {x} ^ {*}) \| ^ {2}} \\ & {\qquad \leq \| \mathbf {x} ^ {n} - t _ {n} f ^ {\prime} (\mathbf {x} ^ {n}) - \mathbf {x} ^ {*} \| ^ {2}} \\ & {\qquad = \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {n} \langle f ^ {\prime} (\mathbf {x} ^ {n}), \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \rangle + t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| ^ {2}.} \end{array}\tag{8.48}
$$

Since $f$ is σ-strongly convex, it follows by Theorem 5.24 that 

$$
f (\mathbf {x} ^ {*}) \geq f (\mathbf {x} ^ {n}) + \langle f ^ {\prime} (\mathbf {x} ^ {n}), \mathbf {x} ^ {*} - \mathbf {x} ^ {n} \rangle + \frac {\sigma}{2} \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2}.
$$

That is, 

$$
\langle f ^ {\prime} (\mathbf {x} ^ {n}), \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \rangle \geq f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}} + \frac {\sigma}{2} \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2}.
$$

Plugging the above into (8.48), we obtain that 

$$
\| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} \leq (1 - \sigma t _ {n}) \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) + t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| ^ {2}.
$$

Rearranging terms, dividing by $2 t _ { n }$ , and using the bound $\| f ^ { \prime } ( \mathbf { x } ^ { n } ) \| \leq L _ { f }$ leads to the following inequality: 

$$
f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}} \leq \frac {1}{2} (t _ {n} ^ {- 1} - \sigma) \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - \frac {1}{2} t _ {n} ^ {- 1} \| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} + \frac {t _ {n}}{2} L _ {f} ^ {2}.
$$

Plugging $\begin{array} { r } { t _ { n } = \frac { 2 } { \sigma ( n + 1 ) } } \end{array}$ into the latter inequality, we obtain 

$$
f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}} \leq \frac {\sigma (n - 1)}{4} \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - \frac {\sigma (n + 1)}{4} \| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} + \frac {1}{\sigma (n + 1)} L _ {f} ^ {2}.
$$

Multiplying the above by n yields the following inequality: 

$$
n (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) \leq \frac {\sigma n (n - 1)}{4} \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - \frac {\sigma (n + 1) n}{4} \| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} + \frac {n}{\sigma (n + 1)} L _ {f} ^ {2}.
$$

Summing over $n = 0 , 1 , \ldots , k$ , we conclude that 

$$
\sum_ {n = 0} ^ {k} n (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) \leq 0 - \frac {\sigma}{4} k (k + 1) \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2} + \frac {L _ {f} ^ {2}}{\sigma} \sum_ {n = 0} ^ {k} \frac {n}{n + 1} \leq \frac {L _ {f} ^ {2} k}{\sigma}.\tag{8.49}
$$

Therefore, using the inequality $f ( \mathbf { x } ^ { n } ) \geq f _ { \mathrm { b e s t } } ^ { k }$ for all $n = 0 , 1 , \ldots , k$ , it follows that 

$$
\left(\sum_ {n = 0} ^ {k} n\right) \left(f _ {\text { best }} ^ {k} - f _ {\text { opt }}\right) \leq \frac {L _ {f} ^ {2} k}{\sigma},
$$

which by the known identity $\textstyle \sum _ { n = 0 } ^ { k } n = { \frac { k ( k + 1 ) } { 2 } }$ shows that 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {2 L _ {f} ^ {2}}{\sigma (k + 1)},\tag{8.50}
$$

meaning that (8.44) holds. To prove (8.45), note that $f _ { \mathrm { b e s t } } ^ { k } = f ( \mathbf { x } ^ { i _ { k } } )$ , and hence by Theorem 5.25(b) employed on the σ-strongly convex function $f + \delta _ { C }$ and (8.50), 

$$
\frac {\sigma}{2} \| \mathbf {x} ^ {i _ {k}} - \mathbf {x} ^ {*} \| ^ {2} \leq f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {2 L _ {f} ^ {2}}{\sigma (k + 1)},
$$

which is the same as 

$$
\left\| \mathbf {x} ^ {i _ {k}} - \mathbf {x} ^ {*} \right\| \leq \frac {2 L _ {f}}{\sigma \sqrt {k + 1}}.
$$

(b) To establish the ergodic convergence, we begin by dividing (8.49) by $\textstyle { \frac { k ( k + 1 ) } { 2 } }$ to obtain 

$$
\sum_ {n = 0} ^ {k} \alpha_ {n} ^ {k} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) \leq \frac {2 L _ {f} ^ {2}}{\sigma (k + 1)}.
$$

By Jensen’s inequality (utilizing the fact that $( \alpha _ { n } ^ { k } ) _ { n = 0 } ^ { k } \in \Delta _ { k + 1 } )$ 2 

$$
f (\mathbf {x} ^ {(k)}) - f _ {\mathrm{opt}} = f \left(\sum_ {n = 0} ^ {k} \alpha_ {n} ^ {k} \mathbf {x} ^ {n}\right) - f _ {\mathrm{opt}} \leq \sum_ {n = 0} ^ {k} \alpha_ {n} ^ {k} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) \leq \frac {2 L _ {f} ^ {2}}{\sigma (k + 1)},
$$

meaning that (8.46) holds. The result (8.47) now follows by the same arguments used to prove (8.45) in part (a). 

Remark 8.32. The sequence of averages $\mathbf { x } ^ { ( k ) }$ as defined in Theorem 8.31 can be computed in an adaptive way by noting that the following simple recursion relation holds: 

$$
\mathbf {x} ^ {(k + 1)} = \frac {k}{k + 2} \mathbf {x} ^ {(k)} + \frac {2}{k + 2} \mathbf {x} ^ {k + 1}.
$$

The $O ( 1 / k )$ rate of convergence of the sequence of function values naturally leads to the observation that to obtain an ε-optimal solution, an order of $1 / \varepsilon$ iterations is required. The proof is trivial and follows the argument of the proof of Theorem 8.18. 

Theorem 8.33 (complexity of projected subgradient for strongly convex functions). Under the setting and assumptions of Theorem 8.31, for any nonnegative integer k satisfying 

$$
k \geq \frac {2 L _ {f} ^ {2}}{\sigma \varepsilon} - 1,
$$

it holds that 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \varepsilon
$$

and 

$$
f (\mathbf {x} ^ {(k)}) - f _ {\mathrm{opt}} \leq \varepsilon .
$$

## 8.3 The Stochastic Projected Subgradient Method

## 8.3.1 Setting and Method

In this section we still study the model (8.10) under Assumption 8.7. The main diference will be that at each iteration we do not necessarily utilize a subgradient at the current iterate $\mathbf { x } ^ { k }$ as the update direction vector, but rather a random estimator $\mathbf { g } ^ { k }$ of a subgradient of $f$ at $\mathbf { x } ^ { k }$ (a precise characterization will be given in Assumption 8.34). The method is therefore given as follows. 

The Stochastic Projected Subgradient Method
Initialization: pick $x^{0} \in C$ arbitrarily.
General step: for any $k = 0, 1, 2, \ldots$ execute the following steps:
(A) pick a stepsize $t_{k} > 0$ and a random vector $g^{k} \in E$ ;
(B) set $x^{k+1} = P_{C}(x^{k} - t_{k}g^{k})$ . 

Obviously, since the vectors $\mathbf { g } ^ { k }$ are random vectors, so are the iterate vectors $\mathbf { x } ^ { k }$ . The exact assumptions on the random vectors $\mathbf { g } ^ { k }$ are given below. 

## Assumption 8.34.

(A) (unbiasedness) For any $k \geq 0 , \mathsf { E } ( \mathbf { g } ^ { k } | \mathbf { x } ^ { k } ) \in \partial f ( \mathbf { x } ^ { k } )$ 

(B) (boundedness) There exists a constant $\tilde { L } _ { f } > 0$ such that for any $k \geq 0$ $\begin{array} { r } { \dot { \mathsf { E } } ( \| \mathbf { g } ^ { k } \| ^ { 2 } | \mathbf { x } ^ { k } ) \leq \dot { \tilde { L } } _ { f } ^ { 2 } } \end{array}$ 

Part $\mathrm { ( A ) }$ of the assumption says that $\mathbf { g } ^ { k }$ is an unbiased estimator of a subgradient at $\mathbf { x } ^ { k }$ . This assumption can also be written as 

$$
f (\mathbf {z}) \geq f \left(\mathbf {x} ^ {k}\right) + \langle \mathrm{E} \left(\mathbf {g} ^ {k} \mid \mathbf {x} ^ {k}\right), \mathbf {z} - \mathbf {x} ^ {k} \rangle \text {   for   all   } \mathbf {z} \in \operatorname{dom} (f).
$$

The constant ${ \tilde { L } } _ { f }$ from part (B) of Assumption 8.34 is not necessarily a Lipschitz constant of f as in the deterministic case. 

## 8.3.2 Analysis

The analysis of the stochastic projected subgradient is almost identical to the analysis of the deterministic method. We gather the main results in the following theorem. 

Theorem 8.35 (convergence of stochastic projected gradient). Suppose that Assumptions 8.7 and 8.34 hold. Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the stochastic projected subgradient method with positive stepsizes $\{ t _ { k } \} _ { k \geq 0 }$ , and let $\{ f _ { \mathrm { b e s t } } ^ { k } \} _ { k \geq 0 }$ be the sequence of best achieved values defined in (8.11). 

(a) $\begin{array} { r } { I f { \frac { \sum _ { n = 0 } ^ { k } t _ { n } ^ { 2 } } { \sum _ { n = 0 } ^ { k } t _ { n } } } \to 0 } \end{array}$ as k → ∞, then $\mathsf { E } ( f _ { \mathrm { b e s t } } ^ { k } ) \to f _ { \mathrm { o p t } }$ as $k \to \infty$ 

(b) Assume that C is compact. Let ${ \tilde { L } } _ { f }$ be the positive constant defined in $A s _ { - }$ sumption 8.34, and let $\Theta$ be an upper bound on the half-squared diameter of C: 

$$
\Theta \geq \max _ {\mathbf {x}, \mathbf {y} \in C} \frac {1}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.\tag{8.51}
$$

If $\begin{array} { r } { t _ { k } = \frac { \sqrt { 2 \Theta } } { \tilde { L } _ { f } \sqrt { k + 1 } } } \end{array}$ , then for all $k \geq 2$ 

$$
\mathsf {E} (f _ {\mathrm{best}} ^ {k}) - f _ {\mathrm{opt}} \leq \frac {\delta \tilde {L} _ {f} \sqrt {2 \Theta}}{\sqrt {k + 2}},
$$

where $\delta = 2 ( 1 + \log ( 3 ) )$ . 

Proof. We have for any $n \geq 0$ 

$$
\begin{array}{r c l} \mathsf {E} \left(\| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} | \mathbf {x} ^ {n}\right) & = & \mathsf {E} \left(\| P _ {C} (\mathbf {x} ^ {n} - t _ {n} \mathbf {g} ^ {n}) - P _ {C} (\mathbf {x} ^ {*}) \| ^ {2} | \mathbf {x} ^ {n}\right) \\ & \overset {(*)} {\leq} & \mathsf {E} \left(\| \mathbf {x} ^ {n} - t _ {n} \mathbf {g} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} | \mathbf {x} ^ {n}\right) \\ & = & \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {n} \mathsf {E} \left(\langle \mathbf {g} ^ {n}, \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \rangle | \mathbf {x} ^ {n}\right) + t _ {n} ^ {2} \mathsf {E} \left(\| \mathbf {g} ^ {n} \| ^ {2} | \mathbf {x} ^ {n}\right) \\ & = & \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {n} \langle \mathsf {E} (\mathbf {g} ^ {n} | \mathbf {x} ^ {n}), \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \rangle + t _ {n} ^ {2} \mathsf {E} \left(\| \mathbf {g} ^ {n} \| ^ {2} | \mathbf {x} ^ {n}\right) \\ & \overset {(* *)} {\leq} & \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {n} \langle \mathsf {E} (\mathbf {g} ^ {n} | \mathbf {x} ^ {n}), \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \rangle + t _ {n} ^ {2} \tilde {L} _ {f} ^ {2} \\ & \overset {(* * *)} {\leq} & \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) + t _ {n} ^ {2} \tilde {L} _ {f} ^ {2}, \end{array}
$$

where $( * )$ follows by the nonexpansiveness property of the orthogonal projection operator (Theorem 6.42), and (∗∗) and (∗∗∗) follow by Assumption 8.34. 

Taking expectation $\mathrm { w . r . t . ~ } \mathbf { x } ^ { n }$ , we obtain 

$$
\mathsf {E} \left(\| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2}\right) \leq \mathsf {E} \left(\| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2}\right) - 2 t _ {n} (\mathsf {E} (f (\mathbf {x} ^ {n})) - f _ {\mathrm{opt}}) + t _ {n} ^ {2} \tilde {L} _ {f} ^ {2}.
$$

Summing over $n = m , m + 1 , \ldots , k$ (where m is an integer satisfying $m \leq k )$ 

$$
\mathsf {E} \left(\| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2}\right) \leq \mathsf {E} \left(\| \mathbf {x} ^ {m} - \mathbf {x} ^ {*} \| ^ {2}\right) - 2 \sum_ {n = m} ^ {k} t _ {n} (\mathsf {E} (f (\mathbf {x} ^ {n})) - f _ {\mathrm{opt}}) + \tilde {L} _ {f} ^ {2} \sum_ {n = m} ^ {k} t _ {n} ^ {2}.
$$

Therefore, 

$$
\sum_ {n = m} ^ {k} t _ {n} (\mathsf {E} (f (\mathbf {x} ^ {n})) - f _ {\mathrm{opt}}) \leq \frac {1}{2} \left[ \mathsf {E} \left(\| \mathbf {x} ^ {m} - \mathbf {x} ^ {*} \| ^ {2}\right) + \tilde {L} _ {f} ^ {2} \sum_ {n = m} ^ {k} t _ {n} ^ {2} \right],
$$

which implies 

$$
\left(\sum_ {n = m} ^ {k} t _ {n}\right) \left(\min _ {n = m, m + 1, \dots , k} \mathsf {E} (f (\mathbf {x} ^ {n})) - f _ {\text {opt}}\right) \leq \frac {1}{2} \left[ \mathsf {E} \left(\| \mathbf {x} ^ {m} - \mathbf {x} ^ {*} \| ^ {2}\right) + \tilde {L} _ {f} ^ {2} \sum_ {n = m} ^ {k} t _ {n} ^ {2} \right].
$$

Using the inequality<sup>44</sup> 

$$
\mathsf {E} (f _ {\text {best}} ^ {k}) \leq \mathsf {E} \left(\min _ {n = m, m + 1, \dots , k} f (\mathbf {x} ^ {n})\right) \leq \min _ {n = m, m + 1, \dots , k} \mathsf {E} (f (\mathbf {x} ^ {n})),
$$

we can conclude that 

$$
\mathsf {E} (f _ {\mathrm{best}} ^ {k}) - f _ {\mathrm{opt}} \leq \frac {\mathsf {E} (\| \mathbf {x} ^ {m} - \mathbf {x} ^ {*} \| ^ {2}) + \tilde {L} _ {f} ^ {2} \sum_ {n = m} ^ {k} t _ {n} ^ {2}}{2 \sum_ {n = m} ^ {k} t _ {n}}.\tag{8.52}
$$

Plugging $m = 0$ in (8.52), we obtain 

$$
\mathsf {E} (f _ {\mathrm{best}} ^ {k}) - f _ {\mathrm{opt}} \leq \frac {\| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} + \tilde {L} _ {f} ^ {2} \sum_ {n = 0} ^ {k} t _ {n} ^ {2}}{2 \sum_ {n = 0} ^ {k} t _ {n}}.
$$

Therefore, if $\scriptstyle { \frac { \sum _ { n = 0 } ^ { k } t _ { n } ^ { 2 } } { \sum _ { n = 0 } ^ { k } t _ { n } } } \to 0$ , then $\mathsf { E } ( f _ { \mathrm { b e s t } } ^ { k } ) \to f _ { \mathrm { o p t } }$ as $k  \infty$ , proving claim (a). To -<sub>show</sub> <sub>the</sub> <sub>validity</sub> <sub>of</sub> <sub>claim</sub> <sub>(b),</sub> <sub>use</sub> <sub>(8.52)</sub> <sub>with</sub> $m = \lceil k / 2 \rceil$ and the bound (8.51) to obtain ī 2 

$$
\mathsf {E} (f _ {\mathrm{best}} ^ {k}) - f _ {\mathrm{opt}} \leq \frac {\Theta + \frac {\tilde {L} _ {f} ^ {2}}{2} \sum_ {n = \lceil k / 2 \rceil} ^ {k} t _ {n} ^ {2}}{\sum_ {n = \lceil k / 2 \rceil} ^ {k} t _ {n}}.
$$

Taking $\begin{array} { r } { t _ { n } = \frac { \sqrt { 2 \Theta } } { \tilde { L } _ { f } \sqrt { n + 1 } } } \end{array}$ , we get 

$$
\mathsf {E} (f _ {\mathrm{best}} ^ {k}) - f _ {\mathrm{opt}} \leq \frac {\tilde {L} _ {f} \sqrt {2 \Theta}}{2} \frac {1 + \sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{n + 1}}{\sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{\sqrt {n + 1}}},
$$

which, combined with Lemma 8.27(b), yields the desired result. 

Example 8.36 (minimization of sum of convex functions). Consider the optimization model 

$$
\mathrm{(P)} \quad \min \left\{f (\mathbf {x}) \equiv \sum_ {i = 1} ^ {m} f _ {i} (\mathbf {x}): \mathbf {x} \in C \right\},
$$

where $f _ { 1 } , f _ { 2 } , \ldots , f _ { m } : \mathbb { E } \to ( - \infty , \infty ]$ are proper closed and convex functions. Suppose that Assumption 8.7 holds and that C is compact, which in particular implies the validity of Assumption 8.12 with some constant $L _ { f }$ . By Theorem 3.61 $L _ { f }$ is a Lipschitz constant of f over C. Let Θ be some upper bound on the half-squared diameter of C: 

$$
\frac {1}{2} \max _ {\mathbf {x}, \mathbf {y} \in C} \| \mathbf {x} - \mathbf {y} \| ^ {2} \leq \Theta .
$$

In addition, we will assume that for any $i = 1 , 2 , \dots , m$ , there exists a constant $\boldsymbol { L } _ { f _ { i } }$ for which 

$$
\| \mathbf {g} \| \leq L _ {f _ {i}} \text {   for   all   } \mathbf {g} \in \partial f _ {i} (\mathbf {x}), \mathbf {x} \in C.
$$

By Theorem 3.61, $\boldsymbol { L } _ { f _ { i } }$ is a Lipschitz constant of $f _ { i }$ over C. We can consider two options for solving the main problem (P). The first is to employ the projected subgradient method (we assume that $f ^ { \prime } ( \mathbf { x } ^ { k } ) \neq \mathbf { 0 } )$ 

## Algorithm 1

• Initialization: pick $\mathbf { x } ^ { 0 } \in C .$ 

• General step $\mathbf { \left( \right)} k \geq \mathbf { 0 } $ : choose $f _ { i } ^ { \prime } ( \mathbf { x } ^ { k } ) \in \partial f _ { i } ( \mathbf { x } ^ { k } ) , i = 1 , 2 , \ldots , m$ , and compute 

$$
\mathbf {x} ^ {k + 1} = P _ {C} \left(\mathbf {x} ^ {k} - \frac {\sqrt {2 \Theta}}{\| \sum_ {i = 1} ^ {m} f _ {i} ^ {\prime} (\mathbf {x} ^ {k}) \| \sqrt {k + 1}} \left(\sum_ {i = 1} ^ {m} f _ {i} ^ {\prime} (\mathbf {x} ^ {k})\right)\right)
$$

By Theorem 8.30, the following eficiency estimate holds for any $k \geq 2 \colon$ 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {\delta L _ {f} \sqrt {2 \Theta}}{\sqrt {k + 2}},\tag{8.53}
$$

Algorithm 2
- Initialization: pick $\mathbf{x}^0 \in C$ .
- General step ( $k \geq 0$ ):
    - pick $i_k \in \{1, 2, \ldots, m\}$ randomly via a uniform distribution and $f_{i_k}'(\mathbf{x}^k) \in \partial f_{i_k}(\mathbf{x}^k)$ ;
- compute $\mathbf{x}^{k+1} = P_C\left(\mathbf{x}^k - \frac{\sqrt{2\Theta}m}{\tilde{L}_f\sqrt{k+1}} f_{i_k}'(\mathbf{x}^k)\right)$ ,
    where $\tilde{L}_f = \sqrt{m}\sqrt{\sum_{i=1}^m L_{f_i}^2}$ . 

where $\delta = 2 ( 1 \mathrm { { + } l o g ( 3 ) } )$ . A direct consequence is that in order to obtain an ε-optimal solution, 

$$
N _ {1} = \max \left\{\frac {2 \delta^ {2} L _ {f} ^ {2} \Theta}{\varepsilon^ {2}} - 2, 2 \right\}
$$

iterations are suficient. Since the computation of the subgradient of $f$ at $\mathbf { x } ^ { k }$ by the formula $\scriptstyle \sum _ { i = 1 } ^ { m } f _ { i } ^ { \prime } ( \mathbf { x } ^ { k } )$ might be too expensive in cases where m is large, we can alternatively employ the stochastic projected subgradient method where at iteration $k ,$ we define the unbiased estimate of $f ^ { \prime } ( \mathbf { x } ^ { k } )$ as 

$$
\mathbf {g} ^ {k} = m f _ {i _ {k}} ^ {\prime} (\mathbf {x} ^ {k}),
$$

where $i _ { k }$ is randomly picked from $\{ 1 , 2 , \ldots , m \}$ via a uniform distribution. Obviously, 

$$
\mathsf {E} (\mathbf {g} ^ {k} | \mathbf {x} ^ {k}) = \sum_ {i = 1} ^ {m} \frac {1}{m} m f _ {i} ^ {\prime} (\mathbf {x} ^ {k}) = \sum_ {i = 1} ^ {m} f _ {i} ^ {\prime} (\mathbf {x} ^ {k}) \in \partial f (\mathbf {x} ^ {k}),
$$

where the inclusion in $\partial f ( \mathbf { x } ^ { k } )$ follows by the sum rule of subdiferential calculus (Corollary 3.38). Also, 

$$
\mathsf {E} (\| \mathbf {g} ^ {k} \| ^ {2} | \mathbf {x} ^ {k}) = \frac {1}{m} \sum_ {i = 1} ^ {m} m ^ {2} \| f _ {i} ^ {\prime} (\mathbf {x} ^ {k}) \| ^ {2} \leq m \sum_ {i = 1} ^ {m} L _ {f _ {i}} ^ {2} \equiv \tilde {L} _ {f} ^ {2}.
$$

The stochastic projected subgradient method employed on problem (P) therefore takes the following form: 

Invoking Theorem 8.35, we obtain that 

$$
\mathsf {E} (f _ {\mathrm{best}} ^ {k}) - f _ {\mathrm{opt}} \leq \frac {\delta \sqrt {m} \sqrt {\sum_ {i = 1} ^ {m} L _ {f _ {i}} ^ {2}} \sqrt {2 \Theta}}{\sqrt {k + 2}}.\tag{8.54}
$$

In particular, 

$$
N _ {2} = \max \left\{\frac {2 \delta^ {2} m \Theta \sum_ {i = 1} ^ {m} L _ {f _ {i}} ^ {2}}{\varepsilon^ {2}} - 2, 2 \right\}
$$

iterations are suficient in order to ensure that an ε-optimal solution in expectation is reached. The natural question that arises is, is it possible to compare between the two algorithms? The answer is actually not clear. We can compare the two quantities $N _ { 2 }$ and $N _ { 1 }$ , but there are two major flaws in such a comparison. First, in a sense this is like comparing apples and oranges since $N _ { 1 }$ considers a sequence of function values, while $N _ { 2 }$ refers to a sequence of expected function values. In addition, recall that $N _ { 2 }$ and $N _ { 1 }$ only provide upper bounds on the amount of iterations required to obtain an ε-optimal solution (deterministically or in expectation). Comparison of upper bounds might be influenced dramatically by the tightness of the upper bounds. Disregarding these drawbacks, estimating the ratio between $N _ { 2 }$ and $N _ { 1 }$ while neglecting the constant terms, which do not depend on ε, we get 

$$
\frac {N _ {2}}{N _ {1}} \approx \frac {\frac {2 \delta^ {2} m \Theta \sum_ {i = 1} ^ {m} L _ {f _ {i}} ^ {2}}{\varepsilon^ {2}}}{\frac {2 \delta^ {2} L _ {f} ^ {2} \Theta}{\varepsilon^ {2}}} = \frac {m \sum_ {i = 1} ^ {m} L _ {f _ {i}} ^ {2}}{L _ {f} ^ {2}} \equiv \beta .
$$

The value of $\beta$ obviously depends on the specific problem at hand. Let us, for example, consider the instance in which $f _ { i } ( \mathbf { x } ) = \vert \mathbf { a } _ { i } ^ { T } \mathbf { x } + b _ { i } \vert , i = 1 , 2 , \ldots , m$ , where $\mathbf { a } _ { i } \in \mathbb { R } ^ { n } , b _ { i } \in \mathbb { R }$ , and $C = B _ { \parallel \cdot \parallel _ { 2 } } [ \mathbf { 0 } , 1 ]$ . In this case, 

$$
f (\mathbf {x}) = \left\| \mathbf {A} \mathbf {x} + \mathbf {b} \right\| _ {1},
$$

where A is the $m \times n$ matrix whose rows are $\mathbf { a } _ { i } ^ { T }$ and $\mathbf { b } = ( b _ { i } ) _ { i = 1 } ^ { m }$ . Since 

$$
\partial f _ {i} (\mathbf {x}) = \left\{ \begin{array}{l l} \mathbf {a} _ {i}, & \mathbf {a} _ {i} ^ {T} \mathbf {x} + b _ {i} > 0, \\ - \mathbf {a} _ {i}, & \mathbf {a} _ {i} ^ {T} \mathbf {x} + b _ {i} <   0, \\ \{\xi \mathbf {a} _ {i}: \xi \in [ - 1, 1 ] \}, & \mathbf {a} _ {i} ^ {T} \mathbf {x} + b _ {i} = 0, \end{array} \right.
$$

it follows that we can choose $L _ { f _ { i } } = \| \mathbf { a } _ { i } \| _ { 2 }$ . To estimate $L _ { f }$ , note that by Example 3.44, any $\mathbf { g } \in \partial f ( \mathbf { x } )$ has the form $\mathbf { g } = \mathbf { A } ^ { T } \boldsymbol { \eta }$ for some $\eta \in [ - 1 , 1 ] ^ { m }$ , which in particular implies that $\| \pmb { \eta } \| _ { 2 } \le \sqrt { m }$ . Thus, 

$$
\| \mathbf {g} \| _ {2} = \| \mathbf {A} ^ {T} \boldsymbol {\eta} \| _ {2} \leq \| \mathbf {A} ^ {T} \| _ {2, 2} \| \boldsymbol {\eta} \| _ {2} \leq \sqrt {m} \| \mathbf {A} ^ {T} \| _ {2, 2},
$$

where $\| \cdot \| _ { 2 , 2 }$ is the spectral norm. We can therefore choose $L _ { f } = { \sqrt { m } } \| \mathbf { A } ^ { T } \| _ { 2 , 2 }$ Thus, 

$$
\beta = \frac {m \sum_ {i = 1} ^ {m} \| \mathbf {a} _ {i} \| _ {2} ^ {2}}{m \| \mathbf {A} ^ {T} \| _ {2 , 2} ^ {2}} = \frac {\| \mathbf {A} ^ {T} \| _ {F} ^ {2}}{\| \mathbf {A} ^ {T} \| _ {2 , 2} ^ {2}} = \frac {\sum_ {i = 1} ^ {n} \lambda_ {i} (\mathbf {A A} ^ {T})}{\max _ {i = 1 , 2 , \ldots , n} \lambda_ {i} (\mathbf {A A} ^ {T})},
$$

where $\lambda _ { 1 } ( \mathbf { A A } ^ { T } ) \geq \lambda _ { 2 } ( \mathbf { A A } ^ { T } ) \geq \cdots \geq \lambda _ { n } ( \mathbf { A A } ^ { T } )$ are the eigenvalues of $\mathbf { A A } ^ { T }$ ordered nonincreasingly. Using the fact that for any nonnegative numbers $\alpha _ { 1 } , \alpha _ { 2 } , \ldots , \alpha _ { m }$ the inequalities 

$$
\max _ {i = 1, 2, \ldots , m} \alpha_ {i} \leq \sum_ {i = 1} ^ {m} \alpha_ {i} \leq m \max _ {i = 1, 2, \ldots , m} \alpha_ {i}
$$

hold, we obtain that $1 \le \beta \le m$ . The extreme case $\beta = m$ is actually quite logical in the sense that the number of subgradient computations per iteration in Algorithm 1 is m times larger than what is required in Algorithm 2, and it is thus not surprising that the amount of iterations of Algorithm 2 might be m times larger than what is required by Algorithm 1 to obtain the same level of accuracy. What is much less intuitive is the case when $\beta$ is close 1. In these instances, the two algorithms require (modulo the faults of this comparison) the same order of iterations to obtain the same order of accuracy. For example, when A “close” to be of rank one, then $\beta$ will be close to 1. In these cases, the two algorithms should perform similarly, although Algorithm 2 is much less computationally demanding. We can explain this result by the fact that in this instance the vectors ${ \bf a } _ { i }$ are “almost” proportional to each other, and thus all the subgradient directions $f _ { i } ^ { \prime } ( \mathbf { x } ^ { k } )$ are similar. 

## 8.3.3 Stochastic Projected Subgradient—The Strongly Convex Case

The analysis of the stochastic projected subgradient method is almost identical to the one presented for the deterministic case in Theorem 8.31, but for the sake of completeness we present the result and its complete proof. 

Theorem 8.37 (convergence of stochastic projected subgradient for strongly convex functions). Suppose that Assumptions 8.7 and 8.34 hold. Let ${ \tilde { L } } _ { f }$ be the positive constant defined in Assumption 8.34. Assume in addition that $f$ is σ-strongly convex for some $\sigma > 0$ . Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the stochastic projected subgradient method with stepsizes $\begin{array} { r } { t _ { k } = \frac { 2 } { \sigma ( k + 1 ) } } \end{array}$ 

(a) Let $\{ f _ { \mathrm { b e s t } } ^ { k } \} _ { k \geq 0 }$ be the sequence of best achieved values defined in (8.11). Then for any $k \geq 0$ 2 

$$
\mathsf {E} (f _ {\mathrm{best}} ^ {k}) - f _ {\mathrm{opt}} \leq \frac {2 \tilde {L} _ {f} ^ {2}}{\sigma (k + 1)}.
$$

(b) Define the sequence of averages 

$$
\mathbf {x} ^ {(k)} = \sum_ {n = 0} ^ {k} \alpha_ {n} ^ {k} \mathbf {x} ^ {n},
$$

where $\begin{array} { r } { \alpha _ { n } ^ { k } \equiv \frac { 2 n } { k ( k + 1 ) } } \end{array}$ . Then 

$$
\mathsf {E} (f (\mathbf {x} ^ {(k)})) - f _ {\mathrm{opt}} \leq \frac {2 \tilde {L} _ {f} ^ {2}}{\sigma (k + 1)}.
$$

Proof. (a) For any $\mathbf { x } ^ { * } \in X ^ { * }$ and $n \geq 0$ 

$$
\begin{array}{r l} & {\mathsf {E} \left(\| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} | \mathbf {x} ^ {n}\right) = \mathsf {E} \left(\| P _ {C} (\mathbf {x} ^ {n} - t _ {n} \mathbf {g} ^ {n}) - P _ {C} (\mathbf {x} ^ {*}) \| ^ {2} | \mathbf {x} ^ {n}\right)} \\ & {\qquad \leq \mathsf {E} \left(\| \mathbf {x} ^ {n} - t _ {n} \mathbf {g} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} | \mathbf {x} ^ {n}\right)} \\ & {\qquad = \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {n} \langle \mathsf {E} (\mathbf {g} ^ {n} | \mathbf {x} ^ {n}), \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \rangle} \\ & {\qquad + t _ {n} ^ {2} \mathsf {E} (\| \mathbf {g} ^ {n} \| ^ {2} | \mathbf {x} ^ {n}).} \end{array}\tag{8.55}
$$

Since $f$ is σ-strongly convex and $\mathsf { E } ( \mathbf { g } ^ { n } | \mathbf { x } ^ { n } ) \in \partial f ( \mathbf { x } ^ { n } )$ , it follows by Theorem 5.24(ii) that σ 

$$
f (\mathbf {x} ^ {*}) \geq f (\mathbf {x} ^ {n}) + \langle E (\mathbf {g} ^ {n} | \mathbf {x} ^ {n}), \mathbf {x} ^ {*} - \mathbf {x} ^ {n} \rangle + \frac {\sigma}{2} \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2}.
$$

That is, 

$$
\langle \mathsf {E} (\mathbf {g} ^ {n} | \mathbf {x} ^ {n}), \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \rangle \geq f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}} + \frac {\sigma}{2} \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2}.
$$

Plugging the above into (8.55), we obtain that 

$$
\mathsf {E} \left(\| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} | \mathbf {x} ^ {n}\right) \leq (1 - \sigma t _ {n}) \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) + t _ {n} ^ {2} \mathsf {E} (\| \mathbf {g} ^ {n} \| ^ {2} | \mathbf {x} ^ {n}).
$$

Rearranging terms, dividing by $2 t _ { n }$ , and using the bound $\mathsf { E } ( \| \mathbf { g } ^ { n } \| ^ { 2 } | \mathbf { x } ^ { n } ) \leq \tilde { L } _ { f } ^ { 2 }$ leads to the following inequality: 

$$
f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}} \leq \frac {1}{2} (t _ {n} ^ {- 1} - \sigma) \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - \frac {1}{2} t _ {n} ^ {- 1} \mathsf {E} (\| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} | \mathbf {x} ^ {n}) + \frac {t _ {n}}{2} \tilde {L} _ {f} ^ {2}.
$$

Plugging $\begin{array} { r } { t _ { n } = \frac { 2 } { \sigma ( n + 1 ) } } \end{array}$ into the last inequality, we obtain 

$$
f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}} \leq \frac {\sigma (n - 1)}{4} \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - \frac {\sigma (n + 1)}{4} \mathsf {E} (\| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} | \mathbf {x} ^ {n}) + \frac {1}{\sigma (n + 1)} \tilde {L} _ {f} ^ {2}.
$$

Multiplying the above by n and taking expectation w.r.t. $\mathbf { x } ^ { n }$ yields the following inequality: 

$$
\begin{array}{l} n (\mathsf {E} (f (\mathbf {x} ^ {n})) - f _ {\mathrm{opt}}) \leq \frac {\sigma n (n - 1)}{4} \mathsf {E} (\| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2}) - \frac {\sigma (n + 1) n}{4} \mathsf {E} (\| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2}) \\ \qquad + \frac {n}{\sigma (n + 1)} \tilde {L} _ {f} ^ {2}. \end{array}
$$

Summing over $n = 0 , 1 , \ldots , k$ 

$$
\sum_ {n = 0} ^ {k} n (\mathsf {E} (f (\mathbf {x} ^ {n})) - f _ {\mathrm{opt}}) \leq 0 - \frac {\sigma}{4} k (k + 1) \mathsf {E} (\| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2}) + \frac {\tilde {L} _ {f} ^ {2}}{\sigma} \sum_ {n = 0} ^ {k} \frac {n}{n + 1} \leq \frac {\tilde {L} _ {f} ^ {2} k}{\sigma}.\tag{8.56}
$$

Therefore, using the inequality $\mathsf E ( f ( \mathbf x ^ { n } ) ) \ge \mathsf E ( f _ { \mathrm { b e s t } } ^ { k } )$ for all $n = 0 , 1 , \ldots , k .$ , it follows that 

$$
\left(\sum_ {n = 0} ^ {k} n\right) \left(\mathsf {E} (f _ {\text { best }} ^ {k}) - f _ {\text { opt }}\right) \leq \frac {\tilde {L} _ {f} ^ {2} k}{\sigma},
$$

which, by the identity $\textstyle \sum _ { n = 0 } ^ { k } n = { \frac { k ( k + 1 ) } { 2 } }$ , implies that 

$$
\mathsf {E} (f _ {\mathrm{best}} ^ {k}) - f _ {\mathrm{opt}} \leq \frac {2 \tilde {L} _ {f} ^ {2}}{\sigma (k + 1)}.
$$

(b) Divide (8.56) by $\textstyle { \frac { k ( k + 1 ) } { 2 } }$ to obtain 

$$
\sum_ {n = 0} ^ {k} \alpha_ {n} ^ {k} (\mathsf {E} (f (\mathbf {x} ^ {n})) - f _ {\mathrm{opt}}) \leq \frac {2 L _ {f} ^ {2}}{\sigma (k + 1)}.
$$

By Jensen’s inequality (utilizing the fact that $( \alpha _ { n } ^ { k } ) _ { n = 0 } ^ { k } \in \Delta _ { k + 1 } )$ , we finally obtain 

$$
\begin{array}{l} \mathsf {E} (f (\mathbf {x} ^ {(k)})) - f _ {\text {opt}} = \mathsf {E} \left(f \left(\sum_ {n = 0} ^ {k} \alpha_ {n} ^ {k} \mathbf {x} ^ {n}\right)\right) - f _ {\text {opt}} \leq \sum_ {n = 0} ^ {k} \alpha_ {n} ^ {k} (\mathsf {E} (f (\mathbf {x} ^ {n})) - f _ {\text {opt}}) \\ \leq \frac {2 \tilde {L} _ {f} ^ {2}}{\sigma (k + 1)}. \quad \square \end{array}
$$

## 8.4 The Incremental Projected Subgradient Method

Consider the main model (8.10), where f has the form $\begin{array} { r } { f ( \mathbf { x } ) = \sum _ { i = 1 } ^ { m } f _ { i } ( \mathbf { x } ) } \end{array}$ . That is, we consider the problem 

$$
\min \left\{f (\mathbf {x}) = \sum_ {i = 1} ^ {m} f _ {i} (\mathbf {x}): \mathbf {x} \in C \right\}.\tag{8.57}
$$

In addition to Assumption 8.7, we make the following assumption. 

Assumption 8.38. 

(a) $f _ { i }$ is proper closed and convex for any $i = 1 , 2 , \dots , m$ 

(b) There exists $L > 0$ for which $\left\| \mathbf { g } \right\| \leq L$ for any $\mathbf { g } \in \partial f _ { i } ( \mathbf { x } ) , i = 1 , 2 , \ldots , m$ $\mathbf { x } \in C$ 

In Example 8.36 the same model was also considered, and a projected subgradient method that takes a step toward a direction of the form $- f _ { i _ { k } } ^ { \prime } ( \mathbf { x } ^ { k } )$ was analyzed. The index $i _ { k }$ was chosen in Example 8.36 randomly by a uniform distribution over the indices $\{ 1 , 2 , \ldots , m \}$ , and the natural question that arises is whether we can obtain similar convergence results when $i _ { k }$ is chosen in a deterministic manner. We will consider the variant in which the indices are chosen in a deterministic cyclic order. The resulting method is called the incremental projected subgradient method. We will show that although the analysis is much more involved, it is still possible to obtain similar rates of convergence (albeit with worse constants). 

An iteration of the incremental projected subgradient method is divided into subiterations. Let $\mathbf { x } ^ { k }$ be the kth iterate vector. Then we define $\mathbf { x } ^ { k , 0 } = \mathbf { x } ^ { k }$ and produce m subiterations $\mathbf { x } ^ { k , 1 } , \mathbf { x } ^ { k , 2 } , \ldots , \mathbf { x } ^ { k , m }$ by the rule that $\mathbf { x } ^ { k , i + 1 } = P _ { C } ( \mathbf { x } ^ { k , i } - t _ { k } \mathbf { g } ^ { \bar { k } , i } )$ where $\mathbf { g } ^ { k , i } \in \partial f _ { i + 1 } ( \mathbf { x } ^ { k , i } )$ and $t _ { k } > 0$ is a positive stepsize. Finally, the next iterate is defined by $\mathbf { x } ^ { k + 1 } = \mathbf { x } ^ { k , m }$ 

The Incremental Projected Subgradient Method
Initialization: pick $x^{0} \in C$ arbitrarily.
General step: for any $k = 0, 1, 2, \ldots$ execute the following steps:
(a) set $x^{k,0} = x^{k}$ and pick a stepsize $t_{k} > 0$ ;
(b) for any $i = 0, 1, \ldots, m - 1$ compute $\mathbf{x}^{k,i+1} = P_{C}(\mathbf{x}^{k,i} - t_{k}\mathbf{g}^{k,i})$ ,
where $g^{k,i} \in \partial f_{i+1}(x^{k,i})$ ;
(c) set $x^{k+1} = x^{k,m}$ . 

The fundamental inequality from which convergence results can be deduced is proven in the following lemma. The result is similar to the result in Lemma 8.11, but the proof is considerably more complicated. 

Lemma 8.39 (fundamental inequality for the incremental projected subgradient method).<sup>45</sup> Suppose that Assumptions 8.7 and 8.38 hold, and let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the incremental projected subgradient method with positive stepsizes $\{ t _ { k } \} _ { k \geq 0 }$ . Then for any $k \geq 0$ 

$$
\| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2} \leq \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {k} (f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}) + t _ {k} ^ {2} m ^ {2} L ^ {2}.\tag{8.58}
$$

Proof. For any $\mathbf { x } ^ { * } \in X ^ { * } , k \geq 0$ and $i \in \{ 0 , 1 , \ldots , m - 1 \}$ , 

$$
\begin{array}{r c l} \| \mathbf {x} ^ {k, i + 1} - \mathbf {x} ^ {*} \| ^ {2} & = & \| P _ {C} (\mathbf {x} ^ {k, i} - t _ {k} \mathbf {g} ^ {k, i}) - \mathbf {x} ^ {*} \| ^ {2} \\ & = & \| P _ {C} (\mathbf {x} ^ {k, i} - t _ {k} \mathbf {g} ^ {k, i}) - P _ {C} (\mathbf {x} ^ {*}) \| ^ {2} \\ & \overset {(*)} {\leq} & \| \mathbf {x} ^ {k, i} - t _ {k} \mathbf {g} ^ {k, i} - \mathbf {x} ^ {*} \| ^ {2} \\ & \overset {(* *)} {\leq} & \| \mathbf {x} ^ {k, i} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {k} \langle \mathbf {g} ^ {k, i}, \mathbf {x} ^ {k, i} - \mathbf {x} ^ {*} \rangle + t _ {k} ^ {2} L ^ {2} \\ & \overset {(* * *)} {\leq} & \| \mathbf {x} ^ {k, i} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {k} (f _ {i + 1} (\mathbf {x} ^ {k, i}) - f _ {i + 1} (\mathbf {x} ^ {*})) + t _ {k} ^ {2} L ^ {2}, \end{array}
$$

where (∗) follows by the nonexpansivity property of the orthogonal projection operator (Theorem 6.42(b)), (∗∗) by Assumption $8 . 3 8 ( \mathrm { b } )$ , and (∗∗∗) by the subgradient inequality. Summing the inequality over $i = 0 , 1 , \ldots , m - 1$ and using the identities $\mathbf { x } ^ { k , 0 ^ { \star } } = \mathbf { x } ^ { \tilde { k } } , \mathbf { x } ^ { k , m } = \mathbf { x } ^ { \Breve { k } + 1 }$ , we obtain that for any $\mathbf { x } ^ { * } \in X ^ { * }$ 

$$
\begin{array}{l} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2} \leq \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {k} \sum_ {i = 0} ^ {m - 1} \left(f _ {i + 1} (\mathbf {x} ^ {k, i}) - f _ {i + 1} (\mathbf {x} ^ {*})\right) + t _ {k} ^ {2} m L ^ {2} \\ = \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {k} \left(f (\mathbf {x} ^ {k}) - f _ {\text {opt}} + \sum_ {i = 0} ^ {m - 1} (f _ {i + 1} (\mathbf {x} ^ {k, i}) - f _ {i + 1} (\mathbf {x} ^ {k}))\right) + t _ {k} ^ {2} m L ^ {2} \end{array}
$$

$$
\leq \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {k} (f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}) + 2 t _ {k} \sum_ {i = 0} ^ {m - 1} L \| \mathbf {x} ^ {k, i} - \mathbf {x} ^ {k} \| + t _ {k} ^ {2} m L ^ {2},\tag{8.59}
$$

where in the last inequality we used the fact that by Assumptions 8.7 and 8.38, $C \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) ) \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f _ { i + 1 } ) )$ and $\| \mathbf { g } \| \leq L$ for all $\mathbf { g } \in \partial f _ { i + 1 } ( \mathbf { x } ) , \mathbf { x } \in C$ , and thus, by Theorem $3 . 6 1 , f _ { i + 1 }$ is Lipschitz with constant $L$ over $C .$ 

Now, using the nonexpansivity of the orthogonal projection operator, 

$$
\left\| \mathbf {x} ^ {k, 1} - \mathbf {x} ^ {k} \right\| = \left\| P _ {C} \left(\mathbf {x} ^ {k, 0} - t _ {k} \mathbf {g} ^ {k, 0}\right) - P _ {C} \left(\mathbf {x} ^ {k}\right) \right\| \leq t _ {k} \| \mathbf {g} ^ {k, 0} \| \leq t _ {k} L.
$$

Similarly, 

$$
\| \mathbf {x} ^ {k, 2} - \mathbf {x} ^ {k} \| = \| P _ {C} (\mathbf {x} ^ {k, 1} - t _ {k} \mathbf {g} ^ {k, 1}) - P _ {C} (\mathbf {x} ^ {k}) \| \leq \| \mathbf {x} ^ {k, 1} - \mathbf {x} ^ {k} \| + t _ {k} \| \mathbf {g} ^ {k, 1} \| \leq 2 t _ {k} L.
$$

In general, for any $i = 0 , 1 , 2 , \ldots , m - 1$ ， 

$$
\| \mathbf {x} ^ {k, i} - \mathbf {x} ^ {k} \| \leq t _ {k} i L,
$$

and we can thus continue (8.59) and deduce that 

$$
\begin{array}{r l} & {\| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2} \leq \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {k} (f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}) + 2 t _ {k} ^ {2} \sum_ {i = 0} ^ {m - 1} i L ^ {2} + t _ {k} ^ {2} m L ^ {2}} \\ & {\qquad = \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {k} (f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}) + t _ {k} ^ {2} m ^ {2} L ^ {2}. \quad \square} \end{array}
$$

From this point, equipped with Lemma 8.39, we can use the same techniques used in the proofs of Theorems 8.25 and 8.30, for example, and establish the following result, whose proof is detailed here for the sake of completeness. 

Theorem 8.40 (convergence of incremental projected subgradient). Suppose that Assumptions 8.7 and 8.38 hold. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the incremental stochastic projected subgradient method with positive stepsizes $\{ t _ { k } \} _ { k \geq 0 }$ , and let $\{ f _ { \mathrm { b e s t } } ^ { k } \} _ { k \geq 0 }$ be the sequence of best achieved values defined in (8.11). 

(a) $\begin{array} { r } { I f { \frac { \sum _ { n = 0 } ^ { k } t _ { n } ^ { 2 } } { \sum _ { n = 0 } ^ { k } t _ { n } } } \to 0 } \end{array}$ as $k \to \infty$ , then $f _ { \mathrm { b e s t } } ^ { k }  f _ { \mathrm { o p t } }$ as $k \to \infty$ 

(b) Assume that C is compact. Let Θ be an upper bound on the half-squared diameter of C: 

$$
\Theta \geq \max _ {\mathbf {x}, \mathbf {y} \in C} \frac {1}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.
$$

$\begin{array} { r } { I f t _ { k } = \frac { \sqrt { \Theta } } { L m \sqrt { k + 1 } } } \end{array}$ , then for all $k \geq 2$ 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {\delta m L \sqrt {\Theta}}{\sqrt {k + 2}},
$$

where $\delta = 2 ( 2 + \log ( 3 ) )$ . 

Proof. By Lemma 8.39, for any $n \geq 0$ 

$$
\| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {*} \| ^ {2} \leq \| \mathbf {x} ^ {n} - \mathbf {x} ^ {*} \| ^ {2} - 2 t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) + L ^ {2} m ^ {2} t _ {n} ^ {2}.\tag{8.60}
$$

Summing (8.60) over $n = p , p + 1 , \dotsc , k$ , we obtain 

$$
\| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2} \leq \| \mathbf {x} ^ {p} - \mathbf {x} ^ {*} \| ^ {2} - 2 \sum_ {n = p} ^ {k} t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) + L ^ {2} m ^ {2} \sum_ {n = p} ^ {k} t _ {n} ^ {2}.
$$

Therefore, 

$$
2 \sum_ {n = p} ^ {k} t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) \leq \| \mathbf {x} ^ {p} - \mathbf {x} ^ {*} \| ^ {2} + L ^ {2} m ^ {2} \sum_ {n = p} ^ {k} t _ {n} ^ {2},
$$

and hence 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {\| \mathbf {x} ^ {p} - \mathbf {x} ^ {*} \| ^ {2} + L ^ {2} m ^ {2} \sum_ {n = p} ^ {k} t _ {n} ^ {2}}{2 \sum_ {n = p} ^ {k} t _ {n}}.\tag{8.61}
$$

Plugging $p = 0$ into (8.61), we obtain 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {\| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} + L ^ {2} m ^ {2} \sum_ {n = 0} ^ {k} t _ {n} ^ {2}}{2 \sum_ {n = 0} ^ {k} t _ {n}}.
$$

Therefore, if $\frac { \sum _ { n = 0 } ^ { k } t _ { n } ^ { 2 } } { \sum _ { n = 0 } ^ { k } t _ { n } }  0$ as $k  \infty$ , then $f _ { \mathrm { b e s t } } ^ { k }  f _ { \mathrm { o p t } }$ as $k  \infty$ , proving claim -<sub>(a).</sub> <sub>To</sub> <sub>show</sub> <sub>the</sub> <sub>validity</sub> <sub>of</sub> <sub>claim</sub> <sub>(b),</sub> <sub>use</sub> <sub>(8.61)</sub> <sub>with</sub> $p = \lceil k / 2 \rceil$ to obtain 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {2 \Theta + L ^ {2} m ^ {2} \sum_ {n = \lceil k / 2 \rceil} ^ {k} t _ {n} ^ {2}}{2 \sum_ {n = \lceil k / 2 \rceil} ^ {k} t _ {n}}.
$$

Take $\begin{array} { r } { t _ { n } = \frac { \sqrt { \Theta } } { L m \sqrt { n + 1 } } } \end{array}$ . Then we get 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {L m \sqrt {\Theta}}{2} \frac {2 + \sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{n + 1}}{\sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{\sqrt {n + 1}}},
$$

which, combined with Lemma 8.27(b) (with $D = 2 )$ , yields the desired result. 

## 8.5 The Dual Projected Subgradient Method

## 8.5.1 The Dual Problem

Consider the problem 

$$
\begin{array}{r l} {f _ {\mathrm{opt}} =} & {\min f (\mathbf {x})} \\ & {\mathrm{s.t.} \mathbf {g} (\mathbf {x}) \leq \mathbf {0},} \\ & {\mathbf {x} \in X,} \end{array}\tag{8.62}
$$

where the following assumptions are made. 

Assumption 8.41. 

(A) $X \subseteq \mathbb { E }$ is convex. 

(B) $f : \mathbb { E } \to \mathbb { R }$ is convex. 

(C) $\mathbf { g } ( \cdot ) = ( g _ { 1 } ( \cdot ) , g _ { 2 } ( \cdot ) , \ldots , g _ { m } ( \cdot ) ) ^ { T }$ , where $g _ { 1 } , g _ { 2 } , \ldots , g _ { m } : \mathbb { E } \to \mathbb { R }$ are convex. 

(D) The problem has a finite optimal value denoted by $f _ { \mathrm { o p t } }$ , and the optimal set, denoted by $X ^ { \ast }$ , is nonempty. 

(E) There exists $\bar { \bf x } \in X$ for which ${ \bf g } ( \bar { \bf x } ) < { \bf 0 }$ 

(F) For any $\lambda \in \mathbb { R } _ { + } ^ { m }$ , the problem mi $\mathfrak { l } _ { \mathbf { x } \in X } \{ f ( \mathbf { x } ) + \lambda ^ { T } \mathbf { g } ( \mathbf { x } ) \}$ has an optimal solution. 

The Lagrangian dual objective function of problem (8.62) is given by 

$$
q (\pmb {\lambda}) = \min _ {\mathbf {x} \in X} \left\{L (\mathbf {x}; \pmb {\lambda}) \equiv f (\mathbf {x}) + \pmb {\lambda} ^ {T} \mathbf {g} (\mathbf {x}) \right\}.\tag{8.63}
$$

By Assumption 8.41(F), the minimization problem in (8.63) possesses a solution, and thus, in particular, $q ( \lambda )$ is finite for any $\lambda \in \mathbb { R } _ { + } ^ { m }$ . Recall that q is concave over $\mathbb { R } _ { + } ^ { m }$ (as a minimum of afine and, in particular, concave functions), and hence the dual problem, which is given by 

$$
q _ {\mathrm{opt}} = \max \{q (\pmb {\lambda}): \pmb {\lambda} \in \mathbb {R} _ {+} ^ {m} \},\tag{8.64}
$$

is a convex problem, as it consists of maximizing a concave function over a convex set. We note that the dual problem is defined in the space $\mathbb { R } ^ { m }$ , which we assume in this context to be endowed with the dot product and the $l _ { 2 } \mathrm { - n o r m }$ 

By Theorem A.1 and Assumption 8.41, it follows that strong duality holds for the primal-dual pair of problems (8.62) and (8.64), namely, 

$$
f _ {\mathrm{opt}} = q _ {\mathrm{opt}}
$$

and the optimal solution of the dual problem is attained. We will denote the optimal set of the dual problem as $\Lambda ^ { * }$ 

An interesting property of the dual problem under the Slater-type assumption (part (E) of Assumption 8.41) is that its superlevel sets are bounded. 

Theorem 8.42 (boundedness of superlevel sets of the dual objective function).<sup>46</sup> Suppose that Assumption 8.41 holds. $L e t \bar { \bf x } \in X$ be a point satisfying ${ \bf g } ( \bar { \bf x } ) <$ 0 whose existence is warranted by Assumption 8.41(E). Let $\mu \in \mathbb { R }$ . Then for any $\lambda \in S _ { \mu } \equiv \{ \lambda \in \mathbb { R } _ { + } ^ { m } : q ( \lambda ) \geq \mu \}$ 2 

$$
\| \pmb {\lambda} \| _ {2} \leq \frac {f (\bar {\mathbf {x}}) - \mu}{\min _ {j = 1 , 2 , \ldots , m} \{- g _ {j} (\bar {\mathbf {x}}) \}}.
$$

Proof. Since $\lambda \in S _ { \mu }$ , we have 

$$
\mu \leq q (\boldsymbol {\lambda}) \leq f (\bar {\mathbf {x}}) + \boldsymbol {\lambda} ^ {T} \mathbf {g} (\bar {\mathbf {x}}) = f (\bar {\mathbf {x}}) + \sum_ {j = 1} ^ {m} \lambda_ {j} g _ {j} (\bar {\mathbf {x}}).
$$

Therefore, 

$$
- \sum_ {j = 1} ^ {m} \lambda_ {j} g _ {j} (\bar {\mathbf {x}}) \leq f (\bar {\mathbf {x}}) - \mu ,
$$

which, by the facts that $\lambda _ { j } \geq 0$ and $g _ { j } ( \bar { \bf x } ) < 0$ for all $j ,$ implies that 

$$
\sum_ {j = 1} ^ {m} \lambda_ {j} \leq \frac {f (\bar {\mathbf {x}}) - \mu}{\min _ {j = 1 , 2 , \ldots , m} \{- g _ {j} (\bar {\mathbf {x}}) \}}.
$$

Finally, since $\lambda \geq 0$ , we have that $\begin{array} { r } { \| \pmb { \lambda } \| _ { 2 } \le \sum _ { j = 1 } ^ { m } \lambda _ { j } } \end{array}$ , and the desired result is established. 

Taking $\mu = f _ { \mathrm { o p t } } = q _ { \mathrm { o p t } }$ , we have $S _ { \mu } = \Lambda ^ { * }$ , and Theorem 8.42 amounts to the following corollary describing a bound on the dual optimal set. 

Corollary 8.43 (boundedness of the optimal dual set). Suppose that Assumption 8.41 holds, and let $\Lambda ^ { * }$ be the optimal set of the dual problem (8.64). Let $\bar { \bf x } \in X$ be a point satisfying ${ \bf g } ( \bar { \bf x } ) < 0$ whose existence is warranted by Assumption 8.41(E). Then for any $\lambda \in \Lambda ^ { * }$ 

$$
\| \pmb {\lambda} \| _ {2} \leq \frac {f (\bar {\mathbf {x}}) - f _ {\mathrm{opt}}}{\min _ {j = 1 , 2 , \ldots , m} \{- g _ {j} (\bar {\mathbf {x}}) \}}.
$$

## 8.5.2 The Dual Projected Subgradient Method

We begin by recalling how to compute a subgradient of minus of the dual objective function. By Example 3.7, if for a given $\lambda \in \mathbb { R } _ { + } ^ { m }$ the minimum of the problem defining $q ( \lambda )$ is attained at $\mathbf { x } _ { \lambda } \in X$ , meaning if $q ( \pmb { \lambda } ) = f ( \mathbf { x } _ { \pmb { \lambda } } ) + \pmb { \lambda } ^ { T } \mathbf { g } ( \mathbf { x } _ { \pmb { \lambda } } )$ , then $- \mathbf { g } ( \mathbf { x } _ { \lambda } ) \in \partial ( - q ) ( \lambda )$ 

Using the above expression for the subgradient of $- q .$ , we can define the projected subgradient method employed on the dual problem. 

The Dual Projected Subgradient Method
Initialization: pick $\lambda^{0} \in R_{+}^{m}$ arbitrarily.
General step: for any $k = 0, 1, 2, \ldots$ execute the following steps:
(a) pick a positive number $\gamma_{k}$ ;
(b) compute $x^{k} \in \arg\min_{x \in X} \left\{ f(x) + (\lambda^{k})^{T} g(x) \right\}$ ;
(c) if $g(x^{k}) = 0$ , then terminate with an output $x^{k}$ ; otherwise, $\lambda^{k+1} = \left[\lambda^{k} + \gamma_{k} \frac{g(x^{k})}{\|g(x^{k})\|_{2}}\right]_{+}$ . 

The stepsize $\frac { \gamma _ { k } } { \| \mathbf { g } ( \mathbf { x } ^ { k } ) \| _ { 2 } }$ is similar in form to the normalized stepsizes considered in Section 8.2.4. The fact that the condition $\mathbf { g } ( \mathbf { x } ^ { k } ) = \mathbf { 0 }$ guarantees that $\mathbf { x } ^ { k }$ is an optimal solution of problem (8.62) is established in the following lemma. 

Lemma 8.44. Suppose that Assumption 8.41 holds. Let $\bar { \lambda } \in \mathbb R _ { + } ^ { m }$ , and let $\bar { \bf x } \in X$ be such that 

$$
\bar {\mathbf {x}} \in \operatorname{argmin} _ {\mathbf {x} \in X} \left\{f (\mathbf {x}) + \bar {\boldsymbol {\lambda}} ^ {T} \mathbf {g} (\mathbf {x}) \right\}\tag{8.65}
$$

and ${ \bf g } ( \bar { \bf x } ) = { \bf 0 }$ . Then x¯ is an optimal solution of problem (8.62). 

Proof. Let $\mathbf { x }$ be a feasible point of problem (8.62), meaning that $\textbf { x } \in \ X$ and $\mathbf { g } ( \mathbf { x } ) \leq \mathbf { 0 }$ . Then 

$$
\begin{array}{r l} f (\mathbf {x}) \geq f (\mathbf {x}) + \bar {\boldsymbol {\lambda}} ^ {T} \mathbf {g} (\mathbf {x}) & [ \mathbf {g} (\mathbf {x}) \leq \mathbf {0}, \bar {\boldsymbol {\lambda}} \geq \mathbf {0} ] \\ \geq f (\bar {\mathbf {x}}) + \bar {\boldsymbol {\lambda}} ^ {T} \mathbf {g} (\bar {\mathbf {x}}) & [ (8. 6 5) ] \\ = f (\bar {\mathbf {x}}), & [ g (\bar {\mathbf {x}}) = \mathbf {0} ] \end{array}
$$

establishing the optimality of x¯. 

## 8.5.3 Convergence Analysis

Proving convergence of the dual objective function sequence $\{ q ( \lambda ^ { k } ) \} _ { k \geq 0 }$ under various choices of the parameters $\{ \gamma _ { k } \} _ { k \ge 0 }$ is an easy task since such results were already proven in the previous sections. The more interesting question is whether we can prove convergence in some sense of a primal sequence. The answer is yes, but perhaps quite surprisingly the sequence $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ is not the “correct” primal sequence. We will consider the following two possible definitions of the primal sequence that involve averaging of the sequence $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ 

• Full averaging sequence. In this option, we perform averaging of the entire history of iterates: 

$$
\mathbf {x} ^ {(k)} = \sum_ {n = 0} ^ {k} \mu_ {n} ^ {k} \mathbf {x} ^ {n}\tag{8.66}
$$

with $\mu _ { n } ^ { k }$ defined by 

$$
\mu_ {n} ^ {k} = \frac {\gamma_ {n} / \| \mathbf {g} (\mathbf {x} ^ {n}) \| _ {2}}{\sum_ {j = 0} ^ {k} \frac {\gamma_ {j}}{\| \mathbf {g} (\mathbf {x} ^ {j}) \| _ {2}}}, \quad n = 0, 1, \ldots , k.\tag{8.67}
$$

• Partial averaging sequence. Here, at iteration $k ,$ we only perform averaging of iterations $\lceil k / 2 \rceil , \lceil k / 2 \rceil + 1 , . . . , k \colon$ 

$$
\mathbf {x} ^ {\langle k \rangle} = \sum_ {n = \lceil k / 2 \rceil} ^ {k} \eta_ {n} ^ {k} \mathbf {x} ^ {n}\tag{8.68}
$$

with $\eta _ { n } ^ { k }$ defined by 

$$
\eta_ {n} ^ {k} = \frac {\gamma_ {n} / \| \mathbf {g} (\mathbf {x} ^ {n}) \| _ {2}}{\sum_ {j = \lceil k / 2 \rceil} ^ {k} \frac {\gamma_ {j}}{\| \mathbf {g} (\mathbf {x} ^ {j}) \| _ {2}}}, \quad n = \lceil k / 2 \rceil , \ldots , k.\tag{8.69}
$$

Our underlying assumption will be that the method did not terminate, meaning that $\mathbf { g } ( \mathbf { x } ^ { k } ) \neq \mathbf { 0 }$ for any k. 

Lemma 8.45. Suppose that Assumption 8.41 holds, and assume further that there exists $L > 0$ such that $\| \mathbf { g } ( \mathbf { x } ) \| _ { 2 } \leq L$ for any $\mathbf { x } \in X$ . Let $\rho > 0$ be some positive number, and let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ and $\{ \lambda ^ { k } \} _ { k \ge 0 }$ be the sequences generated by the dual projected subgradient method. Then for any $k \geq 2$ 

$$
f (\mathbf {x} ^ {(k)}) - f _ {\mathrm{opt}} + \rho \| [ \mathbf {g} (\mathbf {x} ^ {(k)}) ] _ {+} \| _ {2} \leq \frac {L}{2} \frac {(\| \boldsymbol {\lambda} ^ {0} \| _ {2} + \rho) ^ {2} + \sum_ {n = 0} ^ {k} \gamma_ {n} ^ {2}}{\sum_ {n = 0} ^ {k} \gamma_ {n}}\tag{8.70}
$$

and 

$$
f (\mathbf {x} ^ {\langle k \rangle}) - f _ {\mathrm{opt}} + \rho \| [ \mathbf {g} (\mathbf {x} ^ {\langle k \rangle}) ] _ {+} \| _ {2} \leq \frac {L}{2} \frac {(\| \boldsymbol {\lambda} ^ {\lceil k / 2 \rceil} \| _ {2} + \rho) ^ {2} + \sum_ {n = \lceil k / 2 \rceil} ^ {k} \gamma_ {n} ^ {2}}{\sum_ {n = \lceil k / 2 \rceil} ^ {k} \gamma_ {n}},\tag{8.71}
$$

where $\mathbf { x } ^ { ( k ) }$ and $\mathbf { x } ^ { \langle k \rangle }$ are given in (8.66) and (8.68), respectively. 

Proof. Let $\bar { \lambda } \in \mathbb { R } _ { + } ^ { m }$ . Then for every $n \geq 0$ 

$$
\begin{array}{r l} & {\| \pmb {\lambda} ^ {n + 1} - \bar {\pmb {\lambda}} \| _ {2} ^ {2} = \left\| \left[ \pmb {\lambda} ^ {n} + \gamma_ {n} \frac {\mathbf {g} (\mathbf {x} ^ {n})}{\| \mathbf {g} (\mathbf {x} ^ {n}) \| _ {2}} \right] _ {+} - [ \bar {\pmb {\lambda}} ] _ {+} \right\| _ {2} ^ {2}} \\ & {\qquad \leq \left\| \pmb {\lambda} ^ {n} + \gamma_ {n} \frac {\mathbf {g} (\mathbf {x} ^ {n})}{\| \mathbf {g} (\mathbf {x} ^ {n}) \| _ {2}} - \bar {\pmb {\lambda}} \right\| _ {2} ^ {2}} \\ & {\qquad = \| \pmb {\lambda} ^ {n} - \bar {\pmb {\lambda}} \| _ {2} ^ {2} + \gamma_ {n} ^ {2} + \frac {2 \gamma_ {n}}{\| \mathbf {g} (\mathbf {x} ^ {n}) \| _ {2}} \mathbf {g} (\mathbf {x} ^ {n}) ^ {T} (\pmb {\lambda} ^ {n} - \bar {\pmb {\lambda}}),} \end{array}
$$

where the inequality follows by the nonexpansivity of the orthogonal projection operator (Theorem $6 . 4 2 ( \mathrm { b } ) )$ . Let $p \in \{ 0 , 1 , 2 , \ldots , k \}$ . Summing the above inequality for $n = p , p + 1 , \dotsc , k$ , we obtain that 

$$
\| \pmb {\lambda} ^ {k + 1} - \bar {\pmb {\lambda}} \| _ {2} ^ {2} \leq \| \pmb {\lambda} ^ {p} - \bar {\pmb {\lambda}} \| _ {2} ^ {2} + \sum_ {n = p} ^ {k} \gamma_ {n} ^ {2} + 2 \sum_ {n = p} ^ {k} \frac {\gamma_ {n}}{\| \mathbf {g} (\mathbf {x} ^ {n}) \| _ {2}} \mathbf {g} (\mathbf {x} ^ {n}) ^ {T} (\pmb {\lambda} ^ {n} - \bar {\pmb {\lambda}}).
$$

Therefore, 

$$
2 \sum_ {n = p} ^ {k} \frac {\gamma_ {n}}{\| \mathbf {g} (\mathbf {x} ^ {n}) \| _ {2}} \mathbf {g} (\mathbf {x} ^ {n}) ^ {T} (\bar {\boldsymbol {\lambda}} - \boldsymbol {\lambda} ^ {n}) \leq \| \boldsymbol {\lambda} ^ {p} - \bar {\boldsymbol {\lambda}} \| _ {2} ^ {2} + \sum_ {n = p} ^ {k} \gamma_ {n} ^ {2}.\tag{8.72}
$$

To facilitate the proof of the lemma, we will define for any $p \in \{ 0 , 1 , \ldots , k \}$ 

$$
\mathbf {x} ^ {k, p} \equiv \sum_ {n = p} ^ {k} \alpha_ {n} ^ {k, p} \mathbf {x} ^ {n},\tag{8.73}
$$

where 

$$
\alpha_ {n} ^ {k, p} = \frac {\frac {\gamma_ {n}}{\| \mathbf {g} (\mathbf {x} ^ {n}) \| _ {2}}}{\sum_ {j = p} ^ {k} \frac {\gamma_ {j}}{\| \mathbf {g} (\mathbf {x} ^ {j}) \| _ {2}}}.
$$

In particular, the sequences $\{ \mathbf { x } ^ { k , 0 } \} _ { k \geq 0 } , \{ \mathbf { x } ^ { k , \lceil k / 2 \rceil } \} _ { k \geq 0 }$ are the same as the sequences $\{ \mathbf { x } ^ { ( k ) } \} _ { k \geq 0 }$ and $\{ { \bf x } ^ { \langle k \rangle } \} _ { k \ge 0 }$ , respectively. Using the above definition of $\alpha _ { n } ^ { k , p }$ and the fact that $\| \mathbf { g } ( \mathbf { x } ^ { n } ) \| _ { 2 } \leq L$ , we conclude that (8.72) implies the following inequality: 

$$
\sum_ {n = p} ^ {k} \alpha_ {n} ^ {k, p} \mathbf {g} (\mathbf {x} ^ {n}) ^ {T} (\bar {\boldsymbol {\lambda}} - \boldsymbol {\lambda} ^ {n}) \leq \frac {L}{2} \frac {\| \boldsymbol {\lambda} ^ {p} - \bar {\boldsymbol {\lambda}} \| _ {2} ^ {2} + \sum_ {n = p} ^ {k} \gamma_ {n} ^ {2}}{\sum_ {n = p} ^ {k} \gamma_ {n}}.\tag{8.74}
$$

By the definition of $\mathbf { x } ^ { n }$ , we have for any $\mathbf { x } ^ { * } \in X ^ { * }$ 2 

$$
\begin{array}{r l} f (\mathbf {x} ^ {*}) \geq f (\mathbf {x} ^ {*}) + (\boldsymbol {\lambda} ^ {n}) ^ {T} \mathbf {g} (\mathbf {x} ^ {*}) & [ \boldsymbol {\lambda} ^ {n} \geq \mathbf {0}, \mathbf {g} (\mathbf {x} ^ {*}) \leq \mathbf {0} ] \\ \geq f (\mathbf {x} ^ {n}) + (\boldsymbol {\lambda} ^ {n}) ^ {T} \mathbf {g} (\mathbf {x} ^ {n}). & [ \mathbf {x} ^ {n} \in \operatorname * {a r g m i n} _ {\mathbf {x} \in X} \{f (\mathbf {x}) + (\boldsymbol {\lambda} ^ {n}) ^ {T} \mathbf {g} (\mathbf {x}) \} ] \end{array}
$$

Thus, 

$$
- (\boldsymbol {\lambda} ^ {n}) ^ {T} \mathbf {g} (\mathbf {x} ^ {n}) \geq f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}},
$$

and hence 

$$
\begin{array}{l} \sum_ {n = p} ^ {k} \alpha_ {n} ^ {k, p} \mathbf {g} (\mathbf {x} ^ {n}) ^ {T} (\bar {\boldsymbol {\lambda}} - \boldsymbol {\lambda} ^ {n}) \geq \sum_ {n = p} ^ {k} \alpha_ {n} ^ {k, p} \mathbf {g} (\mathbf {x} ^ {n}) ^ {T} \bar {\boldsymbol {\lambda}} + \sum_ {n = p} ^ {k} \alpha_ {n} ^ {k, p} f (\mathbf {x} ^ {n}) - \sum_ {n = p} ^ {k} \alpha_ {n} ^ {k, p} f _ {\text { opt }} \\ \geq \bar {\boldsymbol {\lambda}} ^ {T} \mathbf {g} (\mathbf {x} ^ {k, p}) + f (\mathbf {x} ^ {k, p}) - f _ {\text { opt }}, \end{array} \tag {8}\tag{8.75}
$$

where the last inequality follows by Jensen’s inequality (recalling that $f$ and the components of g are convex) and the definition (8.73) of $\mathbf { x } ^ { k , p }$ . Combining (8.74) and (8.75), while using the obvious inequality $\| \lambda ^ { p } - \bar { \lambda } \| _ { 2 } \leq \| \lambda ^ { p } \| _ { 2 } + \| \bar { \lambda } \| _ { 2 }$ , we obtain 

$$
f (\mathbf {x} ^ {k, p}) - f _ {\mathrm{opt}} + \bar {\boldsymbol {\lambda}} ^ {T} \mathbf {g} (\mathbf {x} ^ {k, p}) \leq \frac {L}{2} \frac {(\| \boldsymbol {\lambda} ^ {p} \| _ {2} + \| \bar {\boldsymbol {\lambda}} \| _ {2}) ^ {2} + \sum_ {n = p} ^ {k} \gamma_ {n} ^ {2}}{\sum_ {n = p} ^ {k} \gamma_ {n}}.\tag{8.76}
$$

Plugging 

$$
\bar {\boldsymbol {\lambda}} = \left\{ \begin{array}{l l} \rho \frac {[ \mathbf {g} (\mathbf {x} ^ {k , p}) ] _ {+}}{\| [ \mathbf {g} (\mathbf {x} ^ {k , p}) ] _ {+} \| _ {2}}, & [ \mathbf {g} (\mathbf {x} ^ {k, p}) ] _ {+} \neq \mathbf {0}, \\ \mathbf {0}, & [ \mathbf {g} (\mathbf {x} ^ {k, p}) ] _ {+} = \mathbf {0} \end{array} \right.
$$

into (8.76), we obtain the inequality 

$$
f (\mathbf {x} ^ {k, p}) - f _ {\mathrm{opt}} + \rho \| [ \mathbf {g} (\mathbf {x} ^ {k, p}) ] _ {+} \| _ {2} \leq \frac {L}{2} \frac {(\| \pmb {\lambda} ^ {p} \| _ {2} + \rho) ^ {2} + \sum_ {n = p} ^ {k} \gamma_ {n} ^ {2}}{\sum_ {n = p} ^ {k} \gamma_ {n}}.\tag{8.77}
$$

Substituting $p = 0$ and $p = \lceil k / 2 \rceil$ in (8.77) yields the inequalities (8.70) and (8.71), respectively. 

## Analysis of the Full Averaging Scheme

We begin by developing a convergence rate related to the sequence $\{ \mathbf { x } ^ { ( k ) } \} _ { k \geq 0 }$ given by (8.66). Similarly to the analysis for the primal projected subgradient, choosing $\begin{array} { r } { \gamma _ { k } ~ = ~ \frac { 1 } { \sqrt { k + 1 } } } \end{array}$ will imply that the right-hand side of (8.70) will converge to zero. In principle, the fact that the left-hand side of (8.70) converges to zero does not necessarily imply that both the expression for the distance to optimality in function values $f ( \mathbf { \dot { x } } ^ { ( k ) } ) - f _ { \mathrm { o p t } }$ and the expression for the constraints violation $\lVert [ \mathbf { g } ( \mathbf { x } ^ { ( k ) } ) ] ] _ { + } \rVert _ { 2 }$ converge to zero. However, using Theorem 3.60, we can show the convergence of these terms as long as $\rho$ is chosen appropriately. 

Theorem 8.46 $( O ( \log ( k ) / \sqrt { k } )$ rate of convergence of the full averaging sequence). Suppose that Assumption 8.41 holds, and assume further that there exists $L > 0$ for which $\| \mathbf { g } ( \mathbf { x } ) \| _ { 2 } \leq L$ for any $\mathbf { x } \in X$ . Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ , and let $\{ \lambda ^ { k } \} _ { k \geq 0 }$ be the sequences generated by the dual projected subgradient method with $\textstyle \gamma _ { k } = { \frac { 1 } { \sqrt { k + 1 } } }$ Then for any $k \geq 1$ 

$$
f (\mathbf {x} ^ {(k)}) - f _ {\mathrm{opt}} \leq \frac {L}{2} \frac {(\| \boldsymbol {\lambda} ^ {0} \| _ {2} + 2 \alpha) ^ {2} + 1 + \log (k + 1)}{\sqrt {k + 1}},\tag{8.78}
$$

$$
\| [ \mathbf {g} (\mathbf {x} ^ {(k)}) ] _ {+} \| _ {2} \leq \frac {L}{2 \alpha} \frac {\left(\| \boldsymbol {\lambda} ^ {0} \| _ {2} + 2 \alpha\right) ^ {2} + 1 + \log (k + 1)}{\sqrt {k + 1}},\tag{8.79}
$$

where $\{ \mathbf { x } ^ { ( k ) } \} _ { k \geq 0 }$ is given in (8.66) and 

$$
\alpha = \frac {f (\bar {\mathbf {x}}) - f _ {\mathrm{opt}}}{\min _ {j = 1 , 2 , \ldots , m} \{- g _ {j} (\bar {\mathbf {x}}) \}},
$$

with x¯ being a Slater point whose existence is guaranteed by Assumption 8.41(E). 

Proof. Employing Lemma 8.45 with $\rho = 2 \alpha$ , and substituting $\textstyle \gamma _ { n } = { \frac { 1 } { \sqrt { n + 1 } } }$ , we have 

$$
f (\mathbf {x} ^ {(k)}) - f _ {\mathrm{opt}} + 2 \alpha \| [ \mathbf {g} (\mathbf {x} ^ {(k)}) ] _ {+} \| _ {2} \leq \frac {L}{2} \frac {(\| \pmb {\lambda} ^ {0} \| _ {2} + 2 \alpha) ^ {2} + \sum_ {n = 0} ^ {k} \frac {1}{n + 1}}{\sum_ {n = 0} ^ {k} \frac {1}{\sqrt {n + 1}}}.\tag{8.80}
$$

Using Lemma 8.27(a), we have 

$$
\frac {(\| \boldsymbol {\lambda} ^ {0} \| _ {2} + 2 \alpha) ^ {2} + \sum_ {n = 0} ^ {k} \frac {1}{n + 1}}{\sum_ {n = 0} ^ {k} \frac {1}{\sqrt {n + 1}}} \leq \frac {(\| \boldsymbol {\lambda} ^ {0} \| _ {2} + 2 \alpha) ^ {2} + 1 + \log (k + 1)}{\sqrt {k + 1}},
$$

which, combined with (8.80), yields the inequality 

$$
f (\mathbf {x} ^ {(k)}) - f _ {\text { opt }} + 2 \alpha \| [ \mathbf {g} (\mathbf {x} ^ {(k)}) ] _ {+} \| _ {2} \leq \frac {L}{2} \frac {(\| \boldsymbol {\lambda} ^ {0} \| _ {2} + 2 \alpha) ^ {2} + 1 + \log (k + 1)}{\sqrt {k + 1}}.\tag{8.81}
$$

Since by Corollary 8.43 2α is an upper bound on twice the $l _ { 2 } { \mathrm { - n o r m } }$ of any dual optimal solution, it follows by Theorem 3.60 that the inequality (8.81) implies the two inequalities (8.78) and (8.79). 

## Analysis of the Partial Averaging Scheme

We will now show an $O ( 1 / \sqrt { k } )$ rate of convergence in terms of function values as well as constraint violation of the partial averaging sequence given in (8.68). The proof is similar to the proof of Theorem 8.46 and utilizes inequality (8.71) but in addition utilizes the boundedness of the sequence of dual variables—a fact established in the next lemma. 

Lemma $\mathbf { 8 . 4 7 . ^ { 4 7 } }$ Suppose that Assumption 8.41 holds and assume further that there exists $L > 0$ for which $\| \mathbf { g } ( \mathbf { x } ) \| _ { 2 } \leq L$ for any $\mathbf { x } \in X$ . Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ and $\{ \lambda ^ { k } \} _ { k \geq 0 }$ be the sequences generated by the dual projected subgradient method with positive stepsizes $\gamma _ { k }$ satisfying $\gamma _ { k } \le \gamma _ { 0 }$ for all $k \geq 0$ . Then 

$$
\| \boldsymbol {\lambda} ^ {k} \| _ {2} \leq M,\tag{8.82}
$$

where<sup>48</sup> 

$$
M = \left\{\| \pmb {\lambda} ^ {0} \| _ {2} + 2 \alpha , \frac {f (\bar {\mathbf {x}}) - q _ {\mathrm{opt}}}{\beta} + \frac {\gamma_ {0} L}{2 \beta} + 2 \alpha + \gamma_ {0} \right\},\tag{8.83}
$$

with 

$$
\alpha = \frac {f (\bar {\mathbf {x}}) - f _ {\mathrm{opt}}}{\min _ {j = 1 , 2 , \ldots , m} \{- g _ {j} (\bar {\mathbf {x}}) \}}, \qquad \beta = \min _ {j = 1, 2, \ldots , m} \{- g _ {j} (\bar {\mathbf {x}}) \},
$$

where x¯ is a Slater point of problem (8.62) whose existence is guaranteed by $A s _ { - }$ sumption 8.41(E). 

Proof. Let $\lambda ^ { * }$ be an optimal solution of the dual problem (8.64). We begin by showing by induction on k that for any $k \geq 0$ 

$$
\| \boldsymbol {\lambda} ^ {k} - \boldsymbol {\lambda} ^ {*} \| _ {2} \leq \max \left\{\| \boldsymbol {\lambda} ^ {0} - \boldsymbol {\lambda} ^ {*} \| _ {2}, \frac {f (\bar {\mathbf {x}}) - q _ {\mathrm{opt}}}{\beta} + \frac {\gamma_ {0} L}{2 \beta} + \| \boldsymbol {\lambda} ^ {*} \| _ {2} + \gamma_ {0} \right\}.\tag{8.84}
$$

The inequality holds trivially for $k = 0$ . Assume that it holds for $k ,$ and we will show that it holds for $k + 1$ . We will consider two cases. 

Case I. Assume that $\begin{array} { r } { q ( \boldsymbol { \lambda } ^ { k } ) \ge q _ { \mathrm { o p t } } - \frac { \gamma _ { k } L } { 2 } } \end{array}$ . Then, by Theorem 8.42, 

$$
\| \boldsymbol {\lambda} ^ {k} \| _ {2} \leq \frac {f (\bar {\mathbf {x}}) - q _ {\mathrm{opt}} + \frac {\gamma_ {k} L}{2}}{\beta},
$$

where $\begin{array} { r } { \beta = \operatorname* { m i n } _ { i = 1 , 2 , . . . , m } \{ - g _ { i } ( \bar { \bf x } ) \} } \end{array}$ . Therefore, 

$$
\begin{array}{l} \| \boldsymbol {\lambda} ^ {k + 1} - \boldsymbol {\lambda} ^ {*} \| _ {2} \leq \left\| \boldsymbol {\lambda} ^ {k} + \frac {\gamma_ {k}}{\| \mathbf {g} (\mathbf {x} ^ {k}) \| _ {2}} \mathbf {g} (\mathbf {x} ^ {k}) - \boldsymbol {\lambda} ^ {*} \right\| _ {2} \\ \quad \leq \| \boldsymbol {\lambda} ^ {k} \| _ {2} + \| \boldsymbol {\lambda} ^ {*} \| _ {2} + \gamma_ {k} \\ \quad \leq \frac {f (\bar {\mathbf {x}}) - q _ {\mathrm{opt}}}{\beta} + \frac {\gamma_ {k} L}{2 \beta} + \| \boldsymbol {\lambda} ^ {*} \| _ {2} + \gamma_ {k} \\ \quad \leq \frac {f (\bar {\mathbf {x}}) - q _ {\mathrm{opt}}}{\beta} + \frac {\gamma_ {0} L}{2 \beta} + \| \boldsymbol {\lambda} ^ {*} \| _ {2} + \gamma_ {0}. \end{array}
$$

Case II. Now assume that $\begin{array} { r } { q ( \lambda ^ { k } ) < q _ { \mathrm { o p t } } - \frac { \gamma _ { k } L } { 2 } } \end{array}$ . In this case we can write 

$$
\begin{array}{l} \| \boldsymbol {\lambda} ^ {k + 1} - \boldsymbol {\lambda} ^ {*} \| _ {2} ^ {2} = \left\| \left[ \boldsymbol {\lambda} ^ {k} + \frac {\gamma_ {k}}{\| \mathbf {g} (\mathbf {x} ^ {k}) \| _ {2}} \mathbf {g} (\mathbf {x} ^ {k}) \right] _ {+} - \boldsymbol {\lambda} ^ {*} \right\| _ {2} ^ {2} \\ \leq \left\| \boldsymbol {\lambda} ^ {k} - \boldsymbol {\lambda} ^ {*} + \frac {\gamma_ {k}}{\| \mathbf {g} (\mathbf {x} ^ {k}) \| _ {2}} \mathbf {g} (\mathbf {x} ^ {k}) \right\| _ {2} ^ {2} \\ = \| \boldsymbol {\lambda} ^ {k} - \boldsymbol {\lambda} ^ {*} \| _ {2} ^ {2} + 2 \frac {\gamma_ {k}}{\| \mathbf {g} (\mathbf {x} ^ {k}) \| _ {2}} (\boldsymbol {\lambda} ^ {k} - \boldsymbol {\lambda} ^ {*}) ^ {T} \mathbf {g} (\mathbf {x} ^ {k}) + \gamma_ {k} ^ {2}. \end{array}\tag{8.85}
$$

Since $- \mathbf { g } ( \mathbf { x } ^ { k } ) \in \partial ( - q ) ( \lambda ^ { k } )$ (Example 3.7), it follows by the subgradient inequality that 

$$
- q _ {\mathrm{opt}} \geq - q (\pmb {\lambda} ^ {k}) - \mathbf {g} (\mathbf {x} ^ {k}) ^ {T} (\pmb {\lambda} ^ {*} - \pmb {\lambda} ^ {k}).
$$

Therefore, continuing (8.85), 

$$
\begin{array}{r l} & {\| \pmb {\lambda} ^ {k + 1} - \pmb {\lambda} ^ {*} \| ^ {2} \leq \| \pmb {\lambda} ^ {k} - \pmb {\lambda} ^ {*} \| _ {2} ^ {2} + 2 \frac {\gamma_ {k}}{\| \mathbf {g} (\mathbf {x} ^ {k}) \| _ {2}} (q (\pmb {\lambda} ^ {k}) - q _ {\mathrm{opt}}) + \gamma_ {k} ^ {2}} \\ & {\qquad \leq \| \pmb {\lambda} ^ {k} - \pmb {\lambda} ^ {*} \| _ {2} ^ {2} + 2 \frac {\gamma_ {k}}{L} (q (\pmb {\lambda} ^ {k}) - q _ {\mathrm{opt}}) + \gamma_ {k} ^ {2}} \\ & {\qquad = \| \pmb {\lambda} ^ {k} - \pmb {\lambda} ^ {*} \| _ {2} ^ {2} + 2 \frac {\gamma_ {k}}{L} \left(q (\pmb {\lambda} ^ {k}) - q _ {\mathrm{opt}} + \frac {\gamma_ {k} L}{2}\right)} \\ & {\qquad <   \| \pmb {\lambda} ^ {k} - \pmb {\lambda} ^ {*} \| _ {2} ^ {2},} \end{array}
$$

where in the last inequality we used our assumption that $\begin{array} { r } { q ( \Delta ^ { k } ) < q _ { \mathrm { o p t } } - \frac { \gamma _ { k } L } { 2 } } \end{array}$ . We can now use the induction hypothesis and conclude that 

$$
\| \boldsymbol {\lambda} ^ {k + 1} - \boldsymbol {\lambda} ^ {*} \| _ {2} \leq \max \left\{\| \boldsymbol {\lambda} ^ {0} - \boldsymbol {\lambda} ^ {*} \| _ {2}, \frac {f (\bar {\mathbf {x}}) - q _ {\mathrm{opt}}}{\beta} + \frac {\gamma_ {0} L}{2 \beta} + \| \boldsymbol {\lambda} ^ {*} \| _ {2} + \gamma_ {0} \right\}.
$$

We have thus established the validity of (8.84) for all $k \geq 0$ . The result (8.82) now follows by recalling that by Corollary 8.43, $\| \boldsymbol { \lambda } ^ { * } \| _ { 2 } \leq \alpha$ , and hence 

$$
\begin{array}{l} \| \boldsymbol {\lambda} ^ {k} \| _ {2} \leq \| \boldsymbol {\lambda} ^ {k} - \boldsymbol {\lambda} ^ {*} \| _ {2} + \| \boldsymbol {\lambda} ^ {*} \| _ {2} \\ \quad \leq \max \left\{\| \boldsymbol {\lambda} ^ {0} - \boldsymbol {\lambda} ^ {*} \| _ {2}, \frac {f (\bar {\mathbf {x}}) - q _ {\mathrm{opt}}}{\beta} + \frac {\gamma_ {0} L}{2 \beta} + \| \boldsymbol {\lambda} ^ {*} \| _ {2} + \gamma_ {0} \right\} + \| \boldsymbol {\lambda} ^ {*} \| _ {2} \\ \quad \leq \max \left\{\| \boldsymbol {\lambda} ^ {0} \| _ {2} + 2 \alpha , \frac {f (\bar {\mathbf {x}}) - q _ {\mathrm{opt}}}{\beta} + \frac {\gamma_ {0} L}{2 \beta} + 2 \alpha + \gamma_ {0} \right\}. \quad \square \end{array}
$$

Equipped with the upper bound on the sequence of dual variables, we can prove, using a similar argument to the one used in the proof of Theorem 8.46, an $\bar { O } ( 1 / \sqrt { k } )$ rate of convergence related to the partial averaging sequence generated by the dual projected subgradient method. 

Theorem 8.48 $( O ( 1 / { \sqrt { k } } )$ rate of convergence of the partial averaging sequence). Suppose that Assumption 8.41 holds, and assume further that there exists $L > 0$ for which $\| \mathbf { g } ( \mathbf { x } ) \| _ { 2 } \leq L$ for any $\mathbf { x } \in X$ . Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ , and let $\{ \lambda ^ { k } \} _ { k \geq 0 }$ be the sequences generated by the dual projected subgradient method with $\textstyle \gamma _ { k } = { \frac { 1 } { \sqrt { k + 1 } } }$ Then for any $k \geq 2$ 

$$
f (\mathbf {x} ^ {\langle k \rangle}) - f _ {\mathrm{opt}} \leq \frac {2 L ((M + 2 \alpha) ^ {2} + \log (3))}{\sqrt {k + 2}},\tag{8.86}
$$

$$
\| [ \mathbf {g} (\mathbf {x} ^ {\langle k \rangle}) ] _ {+} \| _ {2} \leq \frac {2 L ((M + 2 \alpha) ^ {2} + \log (3))}{\alpha \sqrt {k + 2}},\tag{8.87}
$$

where $\{ \mathbf { x } ^ { ( k ) } \} _ { k \geq 0 }$ is given in (8.68), M in (8.83), and 

$$
\alpha = \frac {f (\bar {\mathbf {x}}) - f _ {\mathrm{opt}}}{\min _ {j = 1 , 2 , \dots , m} \{- g _ {j} (\bar {\mathbf {x}}) \}}
$$

with x¯ being a Slater point whose existence is guaranteed by Assumption 8.41(E). 

Proof. Employing Lemma 8.45 with $\rho = 2 \alpha$ , and substituting $\textstyle \gamma _ { n } = { \frac { 1 } { \sqrt { n + 1 } } }$ , we have 

$$
\begin{array}{r l} f (\mathbf {x} ^ {\langle k \rangle}) - f _ {\mathrm{opt}} + 2 \alpha \| [ \mathbf {g} (\mathbf {x} ^ {\langle k \rangle}) ] _ {+} \| _ {2} & \leq \frac {L}{2} \frac {(\| \boldsymbol {\lambda} ^ {\lceil k / 2 \rceil} \| _ {2} + 2 \alpha) ^ {2} + \sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{n + 1}}{\sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{\sqrt {n + 1}}} \\ & \leq \frac {L}{2} \frac {(M + 2 \alpha) ^ {2} + \sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{n + 1}}{\sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{\sqrt {n + 1}}}, \end{array}\tag{8.88}
$$

where in the last inequality we used the bound on the dual iterates given in Lemma 8.47. Now, using Lemma 8.27(b), we have 

$$
\frac {(M + 2 \alpha) ^ {2} + \sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{n + 1}}{\sum_ {n = \lceil k / 2 \rceil} ^ {k} \frac {1}{\sqrt {n + 1}}} \leq \frac {4 ((M + 2 \alpha) ^ {2} + \log (3))}{\sqrt {k + 2}},
$$

which, combined with (8.88), yields the inequality 

$$
f (\mathbf {x} ^ {\langle k \rangle}) - f _ {\mathrm{opt}} + 2 \alpha \| [ \mathbf {g} (\mathbf {x} ^ {\langle k \rangle}) ] _ {+} \| _ {2} \leq \frac {2 L ((M + 2 \alpha) ^ {2} + \log (3))}{\sqrt {k + 2}}.\tag{8.89}
$$

Since, by Corollary 8.43, 2α is an upper bound on twice the $l _ { 2 } { \mathrm { - n o r m } }$ of any dual optimal solution, it follows by Theorem 3.60 that the inequality (8.89) implies the two inequalities (8.86) and (8.87). 

To derive a complexity result for the dual projected subgradient method, we should first note that the primal sequence is not feasible, as it does not necessarily satisfy the inequality constraints $\mathbf { g } ( \mathbf { x } ) \leq \mathbf { 0 }$ . Therefore, there is no point in asking how many iterations are required to obtain an ε-optimal solution. Instead, we will consider the related concept of an ε-optimal and feasible solution. A vector $\mathbf { x } \in X$ is called an ε-optimal and feasible solution of problem (8.62) if $f ( \mathbf { x } ) - f _ { \mathrm { o p t } } \leq \varepsilon$ and $\| [ \mathbf { g ( x ) } ] _ { + } \| _ { 2 } \leq \varepsilon$ . Theorem 8.48 immediately implies a complexity result stating that an order of $\textstyle { \frac { 1 } { \varepsilon ^ { 2 } } }$ iterations are required to obtain an ε-optimal and feasible solution. 

Corollary 8.49 $( O ( 1 / \varepsilon ^ { 2 } )$ complexity result for the dual projected subgradient method). Under the setting of Theorem 8.48, $i f k \geq 2$ satisfies 

$$
k \geq \frac {4 L ^ {2} ((M + 2 \alpha) ^ {2} + \log (3)) ^ {2}}{\min \{\alpha^ {2} , 1 \} \varepsilon^ {2}} - 2,
$$

then 

$$
\begin{array}{r l} & f (\mathbf {x} ^ {\langle k \rangle}) - f _ {\mathrm{opt}} \leq \varepsilon , \\ & \| [ \mathbf {g} (\mathbf {x} ^ {\langle k \rangle}) ] _ {+} \| _ {2} \leq \varepsilon . \end{array}
$$

Example 8.50 (linear programming example). Consider the linear programming problem 

$$
\begin{array}{l l} \min & \mathbf {c} ^ {T} \mathbf {x} \\ \text {s.t.} & \mathbf {A x} \leq \mathbf {b}, \\ & \mathbf {x} \in \Delta_ {n}, \end{array}\tag{LP}
$$

where $\mathbf { c } \in \mathbb { R } ^ { n } , \mathbf { A } \in \mathbb { R } ^ { m \times n }$ , and b $\in \mathbb { R } ^ { m }$ . We will consider the dual projected subgradient method when the underlying set X is $\Delta _ { n }$ and $\mathbf { g } ( \mathbf { x } ) \equiv \mathbf { A } \mathbf { x } - \mathbf { b }$ . The vector $\mathbf { x } ^ { k }$ is calculated by the update rule $\begin{array} { r } { \mathbf { x } ^ { k } \in \mathop { \mathrm { a r g m i n } } _ { \mathbf { x } \in \Delta _ { n } } ( \mathbf { \dot { c } } + \mathbf { A } ^ { T } \pmb { \lambda } ^ { k } ) ^ { T } \mathbf { x } } \end{array}$ . It is easy to see that an optimal solution of this subproblem is given by e , where i is an index for which $( \mathbf { c } + \bar { \mathbf { A } } ^ { T } \lambda ^ { k } )$ is minimal. Therefore, the algorithm (with $\begin{array} { r } { \gamma _ { k } = \frac { 1 } { \sqrt { k + 1 } } ) } \end{array}$ takes the following form: 

Dual Projected Subgradient for solving (LP)  
- Initialization: pick $\lambda^0\in \mathbb{R}_+^m$ - General step ( $k\geq 0$ ): $i_k\in \mathrm{argmin}_{i = 1,2,\dots,n}v_i;\mathbf{v} = \mathbf{c} + \mathbf{A}^T\pmb{\lambda}^k,$ $\mathbf{x}^k = \mathbf{e}_{i_k},$ $\pmb{\lambda}^{k + 1} = \left[\pmb{\lambda}^k +\frac{1}{\sqrt{k + 1}}\frac{\mathbf{A}\mathbf{x}^k - \mathbf{b}}{\|\mathbf{A}\mathbf{x}^k - \mathbf{b}\|_2}\right]_+$ . 

Note that we make the implicit assumption that $\mathbf { A } \mathbf { x } ^ { k } \neq \mathbf { b }$ . The above description of the dual projected subgradient method illustrates the fact that the sequence $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ is not the “correct” primal sequence. Indeed, in this case, the vectors $\mathbf { x } ^ { k }$ are always unit vectors, and there is no particular reason why the solution of (LP) should be attained at a unit vector. As a specific example, consider the problem 

$$
\begin{array}{r l} \min & x _ {1} + 3 x _ {2} + 2 x _ {3} \\ \text {s.t.} & 3 x _ {1} + 2 x _ {2} - x _ {3} \leq 1, \\ & - 2 x _ {3} \leq 2, \\ & x _ {1} + x _ {2} + x _ {3} = 1, \\ & x _ {1}, x _ {2}, x _ {3} \geq 0, \end{array}\tag{8.90}
$$

which fits problem (LP) with 

$$
\mathbf {A} = \left( \begin{array}{c c c} 3 & 2 & - 1 \\ 0 & 0 & - 2 \end{array} \right), \quad \mathbf {b} = \binom{1}{2}, \quad \mathbf {c} = \left( \begin{array}{c} 1 \\ 3 \\ 2 \end{array} \right).
$$

The optimal solution of problem (8.90) is $\textstyle { \left( { \frac { 1 } { 2 } } , 0 , { \frac { 1 } { 2 } } \right) }$ . We employed the dual projected subgradient method as described above with $\mathbf { \nabla } \lambda ^ { 0 } = \mathbf { 0 }$ and compared the behavior of the full and partial averaging schemes during the first 100 iterations. The results are described in Figure 8.4. Obviously, the partial averaging scheme exhibits superior behavior compared to the full averaging scheme. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/8629717540f5ace0269d67cb34e01e282f576387ad69133b919627219245ae22.jpg)



Figure 8.4. First 100 iterations of the dual projected subgradient method employed on problem (8.90). The y-axis describes (in log scale) the quantities max $\{ f ( \mathbf { x } ^ { ( k ) } ) - f _ { \mathrm { o p t } } , \| [ \mathbf { A x } ^ { ( k ) } - \mathbf { b } ] _ { + } \| _ { 2 } \}$ and max $\{ f ( \mathbf { x } ^ { \langle k \rangle } ) - f _ { \mathrm { o p t } } , \| [ \mathbf { A } \mathbf { x } ^ { \langle k \rangle } - \mathbf { b } ] _ { + } \| _ { 2 } \}$


## 8.5.4 Example—Network Utility Maximization

Consider a network that consists of a set $\boldsymbol { S } = \{ 1 , 2 , \ldots , S \}$ of sources and a set $\mathcal { L } = \{ 1 , 2 , \dots , L \}$ of links, where a link has a capacity $c _ { \ell }$ . For each source $s \in { \mathcal { S } }$ we denote by ${ \mathcal { L } } ( s ) \subseteq { \mathcal { L } }$ the set of all links used by source s. Similarly, for a given link $\ell \in { \mathcal { L } }$ , the set $S ( \ell ) \subseteq S$ comprises all sources that use link . In particular, for a pair $\ell \in \mathcal L$ and $s \in S$ , the relation $s \in S ( \ell )$ holds if and only if $\ell \in \mathcal { L } ( s )$ . Each source $s \in S$ is associated with a concave utility function $u _ { s } : \mathbb { R }  \mathbb { R }$ , meaning that if source s sends data at a rate $x _ { s }$ , it gains a utility $u _ { s } ( x _ { s } )$ . We also assume that the rate of source s is constrained to be in the interval $I _ { s } = [ 0 , M _ { s } ]$ , where $M _ { s } \in \mathbb { R } _ { + + }$ The goal of the network utility maximization problem (abbreviated NUM) is to allocate the source rates as the optimal solution of the following convex problem: 

$$
\begin{array}{l l} \max & \sum_ {s \in \mathcal {S}} u _ {s} (x _ {s}) \\ \text {s.t.} & \sum_ {s \in \mathcal {S} (\ell)} x _ {s} \leq c _ {\ell}, \quad \ell \in \mathcal {L}, \\ & x _ {s} \in I _ {s}, \quad s \in \mathcal {S}. \end{array}\tag{8.91}
$$

Problem (8.91) in its minimization form is a convex problem and fits the main model (8.62) with 

$$
\begin{array}{c} \mathbf {g} (\mathbf {x}) = \left(\sum_ {s \in \mathcal {S} (\ell)} x _ {s} - c _ {\ell}\right) _ {\ell = 1, 2, \ldots , L}, \\ X = I _ {1} \times I _ {2} \times \dots \times I _ {S}, \\ f (\mathbf {x}) = - \sum_ {s = 1} ^ {S} u _ {s} (x _ {s}). \end{array}
$$

At iteration $k ,$ the vector $\mathbf { x } ^ { k }$ is picked as an optimal solution of the problem $\begin{array} { r } { \operatorname* { m i n } _ { \mathbf { x } \in X } \{ f ( \mathbf { x } ) + ( \lambda ^ { k } ) ^ { T } \mathbf { g } ( \mathbf { x } ) \} } \end{array}$ , meaning 

$$
\begin{array}{l} \mathbf {x} ^ {k} \in \operatorname * {a r g m i n} _ {\mathbf {x} \in X} \left\{f (\mathbf {x}) + (\boldsymbol {\lambda} ^ {k}) ^ {T} \mathbf {g} (\mathbf {x}) \right\} \\ = \operatorname * {a r g m i n} _ {\mathbf {x} \in X} \left\{- \sum_ {s = 1} ^ {S} u _ {s} (x _ {s}) + \sum_ {\ell = 1} ^ {L} \lambda_ {\ell} ^ {k} \left[ \sum_ {s \in \mathcal {S} (\ell)} x _ {s} - c _ {\ell} \right] \right\} \\ = \operatorname * {a r g m i n} _ {\mathbf {x} \in X} \left\{- \sum_ {s = 1} ^ {S} u _ {s} (x _ {s}) + \sum_ {\ell = 1} ^ {L} \sum_ {s \in \mathcal {S} (\ell)} \lambda_ {\ell} ^ {k} x _ {s} \right\} \\ = \operatorname * {a r g m i n} _ {\mathbf {x} \in X} \left\{- \sum_ {s = 1} ^ {S} u _ {s} (x _ {s}) + \sum_ {s = 1} ^ {S} \left[ \sum_ {\ell \in \mathcal {L} (s)} \lambda_ {\ell} ^ {k} \right] x _ {s} \right\}. \end{array}
$$

The above minimization problem is separable w.r.t. the decision variables $x _ { 1 } , x _ { 2 } , \ldots ,$ $x _ { S }$ . Therefore, the sth element of $\mathbf { x } ^ { k }$ can be chosen via the update rule (returning to the max form), 

$$
x _ {s} ^ {k} \in \operatorname{argmax} _ {x _ {s} \in I _ {s}} \left\{u _ {s} (x _ {s}) - \left[ \sum_ {\ell \in \mathcal {L} (s)} \lambda_ {\ell} ^ {k} \right] x _ {s} \right\}.
$$

The dual projected subgradient method employed on problem (8.91) with stepsizes $\alpha _ { k }$ and initialization $\boldsymbol { \lambda } ^ { 0 } = \mathbf { 0 }$ therefore takes the form below. Note that we do not consider here a normalized stepsize (actually, in many practical scenarios, a constant stepsize is used). 

Dual Projected Subgradient Method for Solving the NUM Problem (8.91) 

Initialization: define $\lambda _ { \ell } ^ { 0 } = 0$ for all $\ell \in { \mathcal { L } }$ 

(A) Source-rate update: 

$$
x _ {s} ^ {k} = \operatorname{argmax} _ {x _ {s} \in I _ {s}} \left\{u _ {s} (x _ {s}) - \left[ \sum_ {\ell \in \mathcal {L} (s)} \lambda_ {\ell} ^ {k} \right] x _ {s} \right\},
$$

$$
s \in \mathcal {S}.\tag{8.92}
$$

(B) Link-price update: 

$$
\lambda_ {\ell} ^ {k + 1} = \left[ \lambda_ {\ell} ^ {k} + \alpha_ {k} \left(\sum_ {s \in \mathcal {S} (\ell)} x _ {s} ^ {k} - c _ {\ell}\right) \right] _ {+}, \qquad \ell \in \mathcal {L}.
$$

The multipliers $\lambda _ { \ell } ^ { k }$ can actually be seen as prices that are associated with the links. The algorithm above can be implemented in a distributed manner in the following sense: 

(a) Each source s needs to solve the optimization problem (8.92) involving only its own utility function $u _ { s }$ and the multipliers (i.e., prices) associated with the links that it uses, meaning $\lambda _ { \ell } ^ { k } , \ell \in \mathcal L ( s )$ 

(b) The price (i.e., multiplier) at each link  is updated according to the rates of the sources that use the link , meaning $x _ { s } , s \in S ( \ell )$ 

Therefore, the algorithm only requires local communication between sources and links and can be implemented in a decentralized manner by letting both the sources and the links cooperatively seek an optimal solution of the problem by following the source-rate/price-link update scheme described above. This is one example of a distributed optimization method. 

# Chapter 9 Mirror Descent

This chapter is devoted to the study of the mirror descent method and some of its variations. The method is essentially a generalization of the projected subgradient method to the non-Euclidean setting. Therefore, naturally, we will not assume in the chapter that the underlying space is Euclidean. 

## 9.1 From Projected Subgradient to Mirror Descent

Consider the optimization problem 

$$
\text {(P)} \quad \min \{f (\mathbf {x}): \mathbf {x} \in C \},\tag{9.1}
$$

where we assume the following.<sup>49</sup> 

Assumption 9.1. 

(A) $f : \mathbb { E } \to ( - \infty , \infty ]$ is proper closed and convex. 

(B) $C \subseteq \mathbb { E }$ is nonempty closed and convex. 

(C) $C \subseteq$ int(dom(f)). 

(D) The optimal set of (P) is nonempty and denoted by $X ^ { * }$ . The optimal value of the problem is denoted by $f _ { \mathrm { o p t } }$ 

The projected subgradient method for solving problem (P) was studied in Chapter 8. One of the basic assumptions made in Chapter 8, which was used throughout the analysis, is that the underlying space is Euclidean, meaning that $\| \cdot \| = \sqrt { \langle \cdot , \cdot \rangle }$ . Recall that the general update step of the projected subgradient method has the form 

$$
\mathbf {x} ^ {k + 1} = P _ {C} (\mathbf {x} ^ {k} - t _ {k} f ^ {\prime} (\mathbf {x} ^ {k})), \quad f ^ {\prime} (\mathbf {x} ^ {k}) \in \partial f (\mathbf {x} ^ {k}),\tag{9.2}
$$

for an appropriately chosen stepsize $t _ { k }$ . When the space is non-Euclidean, there is actually a “philosophical” problem with the update rule (9.2)—the vectors $\mathbf { x } ^ { k }$ and $f ^ { \prime } ( \mathbf { x } ^ { k } )$ are in diferent spaces; one is in E, while the other in $\mathbb { E } ^ { * }$ . This issue is of course not really problematic since we can use our convention that the vectors in E and $\mathbb { E } ^ { * }$ are the same, and the only diference is in the norm associated with each of the spaces. Nonetheless, this issue is one of the motivations for seeking generalizations of the projected subgradient method better suited to the non-Euclidean setting. 

To understand the role of the Euclidean norm in the definition of the projected subgradient method, we will consider the following reformulation of the update step (9.2): 

$$
\mathbf {x} ^ {k + 1} = \operatorname * {a r g m i n} _ {\mathbf {x} \in C} \left\{f (\mathbf {x} ^ {k}) + \langle f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} - \mathbf {x} ^ {k} \rangle + \frac {1}{2 t _ {k}} \| \mathbf {x} - \mathbf {x} ^ {k} \| ^ {2} \right\},\tag{9.3}
$$

which actually shows that $\mathbf { x } ^ { k + 1 }$ is constructed by minimizing a linearization of the objective function plus a quadratic proximity term. The equivalence between the two forms (9.2) and (9.3) in the Euclidean case is evident by the following identity: 

$$
f (\mathbf {x} ^ {k}) + \langle f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} - \mathbf {x} ^ {k} \rangle + \frac {1}{2 t _ {k}} \| \mathbf {x} - \mathbf {x} ^ {k} \| ^ {2} = \frac {1}{2 t _ {k}} \left\| \mathbf {x} - \left[ \mathbf {x} ^ {k} - t _ {k} f ^ {\prime} (\mathbf {x} ^ {k}) \right] \right\| ^ {2} + D,
$$

where D is a constant $( { \mathrm { i . e . } }$ , does not depend on $\mathbf { x } )$ . 

Coming back to the non-Euclidean case, the idea will be to replace the Euclidean “distance” function $\begin{array} { r } { { \frac { 1 } { 2 } } \| \mathbf x - \mathbf y \| ^ { 2 } } \end{array}$ in (9.3) by a diferent distance, which is not based on the Euclidean norm. The non-Euclidean distances that we will use are Bregman distances. 

Definition 9.2 (Bregman distance). Let $\omega : \mathbb { E }  ( - \infty , \infty ]$ be a proper closed and convex function that is diferentiable over dom $( \partial \omega )$ . The Bregman distance associated with $\omega$ is the function $B _ { \omega } : \mathrm { d o m } ( \omega ) \times \mathrm { d o m } ( \partial \omega )  \mathbb { R }$ given by 

$$
B _ {\omega} (\mathbf {x}, \mathbf {y}) = \omega (\mathbf {x}) - \omega (\mathbf {y}) - \langle \nabla \omega (\mathbf {y}), \mathbf {x} - \mathbf {y} \rangle .
$$

The assumptions on ω (given a set $C )$ are gathered in the following. 

Assumption 9.3 (properties of ω). 

• ω is proper closed and convex. 

• ω is diferentiable over dom $( \partial \omega )$ . 

$C \subseteq \operatorname { d o m } ( \omega )$ 

$\omega + \delta _ { C }$ is σ-strongly convex $( \sigma > 0 )$ 

A Bregman distance is actually not necessarily a distance. It is nonnegative and equal to zero if and only if its two arguments coincide, but other than that, in general it is not symmetric and does not satisfy the triangle inequality. The properties of Bregman distances that do hold are summarized in the following lemma. 

(b) Let $\mathbf{x} \in C$ and $\mathbf{y} \in C \cap \mathrm{dom}(\partial \omega)$ . Then
- $B_{\omega}(\mathbf{x}, \mathbf{y}) \geq 0$ ;
- $B_{\omega}(\mathbf{x}, \mathbf{y}) = 0$ if and only if $\mathbf{x} = \mathbf{y}$ . 

Lemma 9.4 (basic properties of Bregman distances). Suppose that $C \subseteq \mathbb { E }$ is nonempty closed and convex and that ω satisfies the properties in Assumption 9.3. Let $B _ { \omega }$ be the Bregman distance associated with $\omega$ . Then 

(a) $\begin{array} { r } { B _ { \omega } ( \mathbf { x } , \mathbf { y } ) \geq \frac { \sigma } { 2 } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } ~ f o r ~ a l l ~ \mathbf { x } \in C , \mathbf { y } \in C \cap \operatorname { d o m } ( \partial \omega ) . } \end{array}$ 

Proof. Part (a) follows by the first-order characterization of strongly convex functions described in Theorem 5.24(ii). Part (b) is a direct consequence of part (a). 

Assume that $\mathbf { x } ^ { k } \in C \cap$ dom $( \partial \omega )$ . Replacing the term $\scriptstyle { \frac { 1 } { 2 } } \| \mathbf { x } - \mathbf { x } ^ { k } \| ^ { 2 }$ in formula (9.3) by a Bregman distance $B _ { \omega } ( \mathbf { x } , \mathbf { x } ^ { k } )$ leads to the following update step: 

$$
\mathbf {x} ^ {k + 1} = \operatorname * {a r g m i n} _ {\mathbf {x} \in C} \left\{f (\mathbf {x} ^ {k}) + \langle f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} - \mathbf {x} ^ {k} \rangle + \frac {1}{t _ {k}} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {k}) \right\}.\tag{9.4}
$$

Omitting constant terms, (9.4) becomes 

$$
\mathbf {x} ^ {k + 1} = \operatorname{argmin} _ {\mathbf {x} \in C} \left\{\langle f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} \rangle + \frac {1}{t _ {k}} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {k}) \right\}.\tag{9.5}
$$

Further simplification of the update formula can be achieved by noting the following simple identity: 

$$
\begin{array}{l} \langle f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} \rangle + \frac {1}{t _ {k}} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {k}) \\ = \frac {1}{t _ {k}} \left[ \langle t _ {k} f ^ {\prime} (\mathbf {x} ^ {k}) - \nabla \omega (\mathbf {x} ^ {k}), \mathbf {x} \rangle + \omega (\mathbf {x}) \right] \underbrace {- \frac {1}{t _ {k}} \omega (\mathbf {x} ^ {k}) + \frac {1}{t _ {k}} \langle \nabla \omega (\mathbf {x} ^ {k}) , \mathbf {x} ^ {k} \rangle} _ {\text {constant}}. \end{array}
$$

Therefore, the update formula in its most simplified form reads as 

$$
\mathbf {x} ^ {k + 1} = \operatorname * {a r g m i n} _ {\mathbf {x} \in C} \left\{\langle t _ {k} f ^ {\prime} (\mathbf {x} ^ {k}) - \nabla \omega (\mathbf {x} ^ {k}), \mathbf {x} \rangle + \omega (\mathbf {x}) \right\}.
$$

We are now ready to define the mirror descent method. 

The Mirror Descent Method
Initialization: pick $x^{0} \in C \cap \text{dom}(\partial\omega)$ .
General step: for any $k = 0, 1, 2, \ldots$ execute the following steps:
(a) pick a stepsize $t_{k} > 0$ and a subgradient $f'(\mathbf{x}^{k}) \in \partial f(\mathbf{x}^{k})$ ;
(b) set $\mathbf{x}^{k+1} = \arg\min_{\mathbf{x} \in C} \left\{ \langle t_{k} f'(\mathbf{x}^{k}) - \nabla \omega(\mathbf{x}^{k}), \mathbf{x} \rangle + \omega(\mathbf{x}) \right\}$ . 

(9.6) 

Remark 9.5. Although (9.6) is the most simplified form of the update step of the mirror descent method, the formula (9.5), which can also be written as 

$$
\mathbf {x} ^ {k + 1} = \operatorname * {a r g m i n} _ {\mathbf {x} \in C} \left\{\langle t _ {k} f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} \rangle + B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {k}) \right\},\tag{9.7}
$$

will also prove itself to be useful. 

Remark 9.6. Defining $\tilde { \omega } = \omega + \delta _ { C }$ , we can write the step (9.6) as 

$$
\mathbf {x} ^ {k + 1} = \operatorname * {a r g m i n} _ {\mathbf {x} \in \mathbb {E}} \left\{\langle t _ {k} f ^ {\prime} (\mathbf {x} ^ {k}) - \nabla \omega (\mathbf {x} ^ {k}), \mathbf {x} \rangle + \tilde {\omega} (\mathbf {x}) \right\}.\tag{9.8}
$$

Since $\nabla \omega ( \mathbf { x } ^ { k } ) \in \partial \tilde { \omega } ( \mathbf { x } ^ { k } )$ , we can write it as $\tilde { \omega } ^ { \prime } ( \mathbf { x } ^ { k } )$ , so (9.8) becomes 

$$
\mathbf {x} ^ {k + 1} = \operatorname * {a r g m i n} _ {\mathbf {x} \in \mathbb {E}} \left\{\langle t _ {k} f ^ {\prime} (\mathbf {x} ^ {k}) - \tilde {\omega} ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} \rangle + \tilde {\omega} (\mathbf {x}) \right\}.\tag{9.9}
$$

Finally, by the conjugate correspondence theorem (Theorem 5.26), whose assumptions hold (properness, closedness, and strong convexity $o f \tilde { \omega } ) , \tilde { \omega } ^ { * }$ is diferentiable, which, combined with the conjugate subgradient theorem (Corollary 4.21), yields that (9.9) is equivalent to the following known formula for the mirror descent method: 

$$
\mathbf {x} ^ {k + 1} = \nabla \tilde {\omega} ^ {*} (\tilde {\omega} ^ {\prime} (\mathbf {x} ^ {k}) - t _ {k} f ^ {\prime} (\mathbf {x} ^ {k})).
$$

The basic step of the mirror descent method (9.6) is of the form 

$$
\min _ {\mathbf {x} \in C} \left\{\langle \mathbf {a}, \mathbf {x} \rangle + \omega (\mathbf {x}) \right\}\tag{9.10}
$$

for some $\mathbf { a _ { \lambda } } \in \mathbb { E } ^ { \ast }$ . To show that the method is well defined, Theorem 9.8 below establishes the fact that the minimum of problem (9.10) is uniquely attained at a point in $C \cap \mathrm { d o m } ( \partial \omega )$ . The reason why it is important to show that the minimizer is in $\mathrm { d o m } ( \partial \omega )$ is that the method requires computing the gradient of ω at the new iterate vector (recall that ω is assumed to be diferentiable over dom(∂ω)). We will prove a more general lemma that will also be useful in other contexts. 

Lemma 9.7. Assume the following: 

$\omega : \mathbb { E }  ( - \infty , \infty ]$ is a proper closed and convex function diferentiable over dom(∂ω). 

$\psi : \mathbb { E }  ( - \infty , \infty ]$ is a proper closed and convex function satisfying dom $( \psi ) \subseteq$ dom(ω). 

$\omega + \delta _ { \mathrm { d o m } ( \psi ) }$ is σ-strongly convex $( \sigma > 0 )$ 

Then the minimizer of the problem 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \left\{\psi (\mathbf {x}) + \omega (\mathbf {x}) \right\}\tag{9.11}
$$

is uniquely attained at a point in dom(ψ) ∩ dom(∂ω). 

Proof. Problem (9.11) is the same as 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \varphi (\mathbf {x}),\tag{9.12}
$$

where $\varphi = \psi + \omega$ . The function $\varphi$ is closed since both ψ and $\omega$ are closed; it is proper by the fact that dom $( \varphi ) = \mathrm { d o m } ( \psi ) \neq \emptyset$ . Since $\omega + \delta _ { \mathrm { d o m } ( \psi ) }$ is σ-strongly convex and $\psi$ is convex, their sum $\psi + \omega + \delta _ { \mathrm { d o m } ( \psi ) } = \psi + \omega = \varphi$ is σ-strongly convex. To conclude, $\varphi$ is proper closed and σ-strongly convex, and hence, by Theorem $5 . 2 5 ( \mathrm { a } )$ , problem (9.12) has a unique minimizer $\mathbf { x } ^ { * }$ in dom $( \varphi ) = \operatorname { d o m } ( \psi )$ . To show that $\mathbf { x } ^ { * } \in \mathrm { d o m } ( \partial \omega )$ , note that by Fermat’s optimality condition (Theorem 3.63), $\mathbf { 0 } \in \partial \varphi ( \mathbf { x } ^ { * } )$ , and in particular $\partial \varphi ( \mathbf { x } ^ { * } ) \neq \emptyset$ . Therefore, since by the sum rule of subdiferential calculus (Theorem 3.40), $\partial \varphi ( \mathbf { x } ^ { * } ) = \partial \psi ( \mathbf { x } ^ { * } ) + \partial \omega ( \mathbf { x } ^ { * } )$ , it follows in particular that $\partial \omega ( \mathbf { x } ^ { * } ) \neq \emptyset$ , meaning that $\mathbf { x } ^ { * } \in$ dom(∂ω). 

The fact that the mirror descent method is well defined can now be easily deduced. 

Theorem 9.8 (mirror descent is well defined). Suppose that Assumptions 9.1 and 9.3 hold. Let $\mathbf { a } \in \mathbb { E } ^ { * }$ . Then the problem 

$$
\min _ {\mathbf {x} \in C} \{\langle \mathbf {a}, \mathbf {x} \rangle + \omega (\mathbf {x}) \}
$$

has a unique minimizer in $C \cap \mathrm { d o m } ( \partial \omega )$ 

Proof. The proof follows by invoking Lemma 9.7 with $\psi ( \mathbf { x } ) \equiv \langle \mathbf { a } , \mathbf { x } \rangle + \delta _ { C } ( \mathbf { x } )$ □ 

Two very common choices of strongly convex functions are described below. 

Example 9.9 (squared Euclidean norm). Suppose that Assumption 9.1 holds and that E is Euclidean, meaning that its norm satisfies $\| \cdot \| = \sqrt { \langle \cdot , \cdot \rangle }$ . Define 

$$
\omega (\mathbf {x}) = \frac {1}{2} \| \mathbf {x} \| ^ {2}.
$$

Then $\omega$ obviously satisfies the properties listed in Assumption 9.3—it is proper closed and 1-strongly convex. Since $\nabla \omega ( \mathbf { x } ) = \mathbf { x }$ , then the general update step of the mirror descent method reads as 

$$
\mathbf {x} ^ {k + 1} = \operatorname * {a r g m i n} _ {\mathbf {x} \in C} \left\{\langle t _ {k} f ^ {\prime} (\mathbf {x} ^ {k}) - \mathbf {x} ^ {k}, \mathbf {x} \rangle + \frac {1}{2} \| \mathbf {x} \| ^ {2} \right\},
$$

which is the same as the projected subgradient update step: $\mathbf { x } ^ { k + 1 } = P _ { C } ( \mathbf { x } ^ { k } -$ $t _ { k } f ^ { \prime } ( \mathbf { x } ^ { k } ) )$ . This is of course not a surprise since the method was constructed as a generalization of the projected subgradient method. 

Example 9.10 (negative entropy over the unit simplex). Suppose that Assumption 9.1 holds with $\mathbb { E } = \mathbb { R } ^ { n }$ endowed with the $l _ { 1 } { \mathrm { - n o r m } }$ and $C = \Delta _ { n }$ . We will take $\omega$ to be the negative entropy over the nonnegative orthant: 

$$
\omega (\mathbf {x}) = \left\{ \begin{array}{l l} \sum_ {i = 1} ^ {n} x _ {i} \log x _ {i}, & \mathbf {x} \in \mathbb {R} _ {+} ^ {n}, \\ \infty & \text { else }. \end{array} \right.
$$

As usual, we use the convention that 0 log $0 = 0$ . By Example 5.27, $\omega + \delta _ { \Delta . }$ is 1-strongly convex w.r.t. the $l _ { \mathrm { 1 - n o r m } }$ . In this case, 

$$
\mathrm{dom} (\partial \omega) = \mathbb {R} _ {+ +} ^ {n},
$$

on which $\omega$ is indeed diferentiable. Thus, all the properties of Assumption 9.3 hold. The associated Bregman distance is given for any $\mathbf { x } \in \Delta _ { n }$ and $\mathbf { y } \in \Delta _ { n } ^ { + } \equiv$ $\{ \mathbf { x } \in \mathbb { R } _ { + + } ^ { n } : \mathbf { e } ^ { T } \mathbf { x } = 1 \}$ by 

$$
\begin{array}{l} B _ {\omega} (\mathbf {x}, \mathbf {y}) = \sum_ {i = 1} ^ {n} x _ {i} \log x _ {i} - \sum_ {i = 1} ^ {n} y _ {i} \log y _ {i} - \sum_ {i = 1} ^ {n} (\log (y _ {i}) + 1) (x _ {i} - y _ {i}) \\ \qquad = \sum_ {i = 1} ^ {n} x _ {i} \log (x _ {i} / y _ {i}) + \sum_ {i = 1} ^ {n} y _ {i} - \sum_ {i = 1} ^ {n} x _ {i} \\ \qquad = \sum_ {i = 1} ^ {n} x _ {i} \log (x _ {i} / y _ {i}), \end{array}\tag{9.13}
$$

which is the so-called Kullback-Leibler divergence distance measure. The general update step of the mirror descent method has the form $( f _ { i } ^ { \prime } ( \mathbf { x } ^ { k } )$ is the ith component of $f ^ { \prime } ( \mathbf { x } ^ { k } ) )$ , 

$$
\mathbf {x} ^ {k + 1} = \operatorname * {a r g m i n} _ {\mathbf {x} \in \Delta_ {n}} \left\{\sum_ {i = 1} ^ {n} (t _ {k} f _ {i} ^ {\prime} (\mathbf {x} ^ {k}) - 1 - \log (x _ {i} ^ {k})) x _ {i} + \sum_ {i = 1} ^ {n} x _ {i} \log x _ {i} \right\}.\tag{9.14}
$$

By Example 3.71, the optimal solution of problem (9.14) is 

$$
x _ {i} ^ {k + 1} = \frac {e ^ {\log (x _ {i} ^ {k}) + 1 - t _ {k} f _ {i} ^ {\prime} (\mathbf {x} ^ {k})}}{\sum_ {j = 1} ^ {n} e ^ {\log (x _ {j} ^ {k}) + 1 - t _ {k} f _ {j} ^ {\prime} (\mathbf {x} ^ {k})}}, i = 1, 2, \ldots , n,
$$

which can be simplified into the following: 

$$
x _ {i} ^ {k + 1} = \frac {x _ {i} ^ {k} e ^ {- t _ {k} f _ {i} ^ {\prime} (\mathbf {x} ^ {k})}}{\sum_ {j = 1} ^ {n} x _ {j} ^ {k} e ^ {- t _ {k} f _ {j} ^ {\prime} (\mathbf {x} ^ {k})}}, \quad i = 1, 2, \ldots , n. \qquad \blacksquare
$$

The natural question that arises is how to choose the stepsizes. The convergence analysis that will be developed in the next section will reveal some possible answers to this question. 

## 9.2 Convergence Analysis

## 9.2.1 The Toolbox

The following identity, also known as the three-points lemma, is essential in the analysis of the mirror descent lemma. 

Lemma 9.11 (three-points lemma).<sup>50</sup> Suppose that $\omega : \mathbb { E }  ( - \infty , \infty ]$ is proper closed and convex. Suppose in addition that ω is diferentiable over dom(∂ω). Assume that $\mathbf { a } , \mathbf { b } \in \mathrm { d o m } ( \partial \omega )$ and $\mathbf { c } \in \mathrm { d o m } ( \omega )$ . Then the following equality holds: 

$$
\langle \nabla \omega (\mathbf {b}) - \nabla \omega (\mathbf {a}), \mathbf {c} - \mathbf {a} \rangle = B _ {\omega} (\mathbf {c}, \mathbf {a}) + B _ {\omega} (\mathbf {a}, \mathbf {b}) - B _ {\omega} (\mathbf {c}, \mathbf {b}).
$$

Proof. By definition of $B _ { \omega }$ , 

$$
B _ {\omega} (\mathbf {c}, \mathbf {a}) = \omega (\mathbf {c}) - \omega (\mathbf {a}) - \langle \nabla \omega (\mathbf {a}), \mathbf {c} - \mathbf {a} \rangle ,
$$

$$
B _ {\omega} (\mathbf {a}, \mathbf {b}) = \omega (\mathbf {a}) - \omega (\mathbf {b}) - \langle \nabla \omega (\mathbf {b}), \mathbf {a} - \mathbf {b} \rangle ,
$$

$$
B _ {\omega} (\mathbf {c}, \mathbf {b}) = \omega (\mathbf {c}) - \omega (\mathbf {b}) - \langle \nabla \omega (\mathbf {b}), \mathbf {c} - \mathbf {b} \rangle .
$$

Hence, 

$$
\begin{array}{c} B _ {\omega} (\mathbf {c}, \mathbf {a}) + B _ {\omega} (\mathbf {a}, \mathbf {b}) - B _ {\omega} (\mathbf {c}, \mathbf {b}) = - \langle \nabla \omega (\mathbf {a}), \mathbf {c} - \mathbf {a} \rangle - \langle \nabla \omega (\mathbf {b}), \mathbf {a} - \mathbf {b} \rangle + \langle \nabla \omega (\mathbf {b}), \mathbf {c} - \mathbf {b} \rangle \\ = \langle \nabla \omega (\mathbf {b}) - \nabla \omega (\mathbf {a}), \mathbf {c} - \mathbf {a} \rangle . \quad \square \end{array}
$$

Another key lemma is an extension of the second prox theorem (Theorem 6.39) to the case of non-Euclidean distances. 

Theorem 9.12 (non-Euclidean second prox theorem). Let 

$\omega : \mathbb { E }  ( - \infty , \infty ]$ be a proper closed and convex function diferentiable over $\mathrm { d o m } ( \partial \omega )$ 

$\psi : \mathbb { E }  ( - \infty , \infty ]$ be a proper closed and convex function satisfying dom $( \psi ) \subseteq$ dom(ω); 

$\omega + \delta _ { \mathrm { d o m } ( \psi ) }$ be σ-strongly convex $( \sigma > 0 )$ 

Assume that b $\in \mathrm { d o m } ( \partial \omega )$ , and let a be defined by 

$$
\mathbf {a} = \operatorname{argmin} _ {\mathbf {x} \in \mathbb {E}} \left\{\psi (\mathbf {x}) + B _ {\omega} (\mathbf {x}, \mathbf {b}) \right\}.\tag{9.15}
$$

Then $\mathbf { a } \in \mathrm { d o m } ( \partial \omega )$ and for all $\mathbf { u } \in \mathrm { d o m } ( \psi )$ 

$$
\langle \nabla \omega (\mathbf {b}) - \nabla \omega (\mathbf {a}), \mathbf {u} - \mathbf {a} \rangle \leq \psi (\mathbf {u}) - \psi (\mathbf {a}).\tag{9.16}
$$

Proof. Using the definition of $B _ { \omega }$ , (9.15) can be rewritten as 

$$
\mathbf {a} = \operatorname{argmin} _ {\mathbf {x} \in \mathbb {E}} \left\{\psi (\mathbf {x}) - \langle \nabla \omega (\mathbf {b}), \mathbf {x} \rangle + \omega (\mathbf {x}) \right\}.\tag{9.17}
$$

The fact that $\mathbf { a } \in \mathrm { d o m } ( \partial \omega )$ follows by invoking Lemma 9.7 with $\psi ( \mathbf { x } ) - \langle \nabla \omega ( \mathbf { b } ) , \mathbf { x } \rangle$ taking the role of $\psi ( \mathbf { x } )$ Using Fermat’s optimality condition (Theorem 3.63), it follows by (9.17) that there exists $\psi ^ { \prime } ( \mathbf { a } ) \in \partial \psi ( \mathbf { a } )$ for which 

$$
\psi^ {\prime} (\mathbf {a}) + \nabla \omega (\mathbf {a}) - \nabla \omega (\mathbf {b}) = \mathbf {0}.
$$

Hence, by the subgradient inequality, for any $\mathbf { u } \in \mathrm { d o m } ( \psi )$ 2 

$$
\langle \nabla \omega (\mathbf {b}) - \nabla \omega (\mathbf {a}), \mathbf {u} - \mathbf {a} \rangle = \langle \psi^ {\prime} (\mathbf {a}), \mathbf {u} - \mathbf {a} \rangle \leq \psi (\mathbf {u}) - \psi (\mathbf {a}),
$$

proving the desired result. 

Using the non-Euclidean second prox theorem and the three-points lemma, we can now establish a fundamental inequality satisfied by the sequence generated by the mirror descent method. The inequality can be seen as a generalization of Lemma 8.11. 

Lemma 9.13 (fundamental inequality for mirror descent). Suppose that Assumptions 9.1 and 9.3 hold. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the mirror descent method with positive stepsizes $\{ t _ { k } \} _ { k \geq 0 }$ . Then for any $\mathbf { x } ^ { * } \in X ^ { * }$ and $k \geq 0$ 

$$
t _ {k} (f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}) \leq B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {k}) - B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {k + 1}) + \frac {t _ {k} ^ {2}}{2 \sigma} \| f ^ {\prime} (\mathbf {x} ^ {k}) \| _ {*} ^ {2}.
$$

Proof. By the update formula (9.7) for $\mathbf { x } ^ { k + 1 }$ and the non-Euclidean second prox theorem (Theorem 9.12) invoked with b ${ \mathbf { \mu } } = { \mathbf { x } } ^ { k }$ and $\psi ( { \bf x } ) \equiv t _ { k } \langle f ^ { \prime } ( { \bf x } ^ { k } ) , { \bf x } \rangle + \delta _ { C } ( { \bf x } )$ (and hence $\mathbf { a } = \mathbf { x } ^ { k + 1 } )$ , we have for any $\mathbf { u } \in C$ 

$$
\langle \nabla \omega (\mathbf {x} ^ {k}) - \nabla \omega (\mathbf {x} ^ {k + 1}), \mathbf {u} - \mathbf {x} ^ {k + 1} \rangle \leq t _ {k} \langle f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {u} - \mathbf {x} ^ {k + 1} \rangle .\tag{9.18}
$$

By the three-points lemma (with $\mathbf { a } = \mathbf { x } ^ { k + 1 } , \mathbf { b } = \mathbf { x } ^ { k }$ , and $\mathbf { c } = \mathbf { u } )$ )， 

$$
\langle \nabla \omega (\mathbf {x} ^ {k}) - \nabla \omega (\mathbf {x} ^ {k + 1}), \mathbf {u} - \mathbf {x} ^ {k + 1} \rangle = B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {k + 1}) + B _ {\omega} (\mathbf {x} ^ {k + 1}, \mathbf {x} ^ {k}) - B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {k}),
$$

which, combined with (9.18), gives 

$$
B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {k + 1}) + B _ {\omega} (\mathbf {x} ^ {k + 1}, \mathbf {x} ^ {k}) - B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {k}) \leq t _ {k} \langle f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {u} - \mathbf {x} ^ {k + 1} \rangle .
$$

Therefore, 

$$
\begin{array}{r l} & t _ {k} \langle f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} ^ {k} - \mathbf {u} \rangle \\ & \leq B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {k}) - B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {k + 1}) - B _ {\omega} (\mathbf {x} ^ {k + 1}, \mathbf {x} ^ {k}) + t _ {k} \langle f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} ^ {k} - \mathbf {x} ^ {k + 1} \rangle \\ & \overset {(*)} {\leq} B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {k}) - B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {k + 1}) - \frac {\sigma}{2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \| ^ {2} + t _ {k} \langle f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} ^ {k} - \mathbf {x} ^ {k + 1} \rangle \\ & = B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {k}) - B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {k + 1}) - \frac {\sigma}{2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \| ^ {2} + \left\langle \frac {t _ {k}}{\sqrt {\sigma}} f ^ {\prime} (\mathbf {x} ^ {k}), \sqrt {\sigma} (\mathbf {x} ^ {k} - \mathbf {x} ^ {k + 1}) \right\rangle \\ & \overset {(* *)} {\leq} B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {k}) - B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {k + 1}) - \frac {\sigma}{2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \| ^ {2} + \frac {t _ {k} ^ {2}}{2 \sigma} \| f ^ {\prime} (\mathbf {x} ^ {k}) \| _ {*} ^ {2} + \frac {\sigma}{2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \| ^ {2} \\ & = B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {k}) - B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {k + 1}) + \frac {t _ {k} ^ {2}}{2 \sigma} \| f ^ {\prime} (\mathbf {x} ^ {k}) \| _ {*} ^ {2}, \end{array}
$$

where the inequality (∗) follows by Lemma $9 . 4 ( \mathrm { a } )$ and (∗∗) by Fenchel’s inequality (Theorem 4.6) employed on the function $\scriptstyle { \frac { 1 } { 2 } } \left\| \mathbf { x } \right\| ^ { 2 }$ (whose conjugate is $\frac { 1 } { 2 } \lVert \mathbf { y } \rVert _ { * } ^ { 2 }$ —see Section 4.4.15). Plugging in $\mathbf { u } = \mathbf { x } ^ { * }$ and using the subgradient inequality, we obtain 

$$
t _ {k} (f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}) \leq B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {k}) - B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {k + 1}) + \frac {t _ {k} ^ {2}}{2 \sigma} \| f ^ {\prime} (\mathbf {x} ^ {k}) \| _ {*} ^ {2}.
$$

Under a boundedness assumption on $B _ { \omega } ( \mathbf { x } , \mathbf { x } ^ { 0 } )$ over $C _ { i }$ , we can deduce a useful bound on the sequence of best achieved function values defined by 

$$
f _ {\mathrm{best}} ^ {k} \equiv \min _ {n = 0, 1, \dots , k} f (\mathbf {x} ^ {n}).\tag{9.19}
$$

Lemma 9.14. Suppose that Assumptions 9.1 and 9.3 hold and that $\| f ^ { \prime } ( \mathbf { x } ) \| _ { * } \leq L _ { f }$ for $a l l \textbf { x } \in \boldsymbol { C }$ , where $L _ { f } > 0$ . Suppose that $B _ { \omega } ( \mathbf { x } , \mathbf { x } ^ { 0 } )$ is bounded over $C _ { i }$ and let $\Theta ( \mathbf { x } ^ { 0 } )$ satisfy 

$$
\Theta (\mathbf {x} ^ {0}) \geq \max _ {\mathbf {x} \in C} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {0}).
$$

Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the mirror descent method with positive stepsizes $\{ t _ { k } \} _ { k \geq 0 }$ . Then for any $N \geq 0$ , 

$$
f _ {\mathrm{best}} ^ {N} - f _ {\mathrm{opt}} \leq \frac {\Theta (\mathbf {x} ^ {0}) + \frac {L _ {f} ^ {2}}{2 \sigma} \sum_ {k = 0} ^ {N} t _ {k} ^ {2}}{\sum_ {k = 0} ^ {N} t _ {k}},\tag{9.20}
$$

where $f _ { \mathrm { b e s t } } ^ { N }$ is defined in (9.19). 

Proof. Let $\mathbf { x } ^ { * } \in X ^ { * }$ . By Lemma 9.13 it follows that for any $k \geq 0$ 

$$
t _ {k} (f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}) \leq B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {k}) - B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {k + 1}) + \frac {t _ {k} ^ {2}}{2 \sigma} \| f ^ {\prime} (\mathbf {x} ^ {k}) \| _ {*} ^ {2}.\tag{9.21}
$$

Summing (9.21) over $k = 0 , 1 , \ldots , N$ , we obtain 

$$
\begin{array}{c} \sum_ {k = 0} ^ {N} t _ {k} (f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}) \leq B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {0}) - B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {N + 1}) + \sum_ {k = 0} ^ {N} \frac {t _ {k} ^ {2}}{2 \sigma} \| f ^ {\prime} (\mathbf {x} ^ {k}) \| _ {*} ^ {2} \\ = \Theta (\mathbf {x} ^ {0}) + \frac {L _ {f} ^ {2}}{2 \sigma} \sum_ {k = 0} ^ {N} t _ {k} ^ {2}, \end{array}
$$

which, combined with the inequality $\begin{array} { r } { ( \sum _ { k = 0 } ^ { n } t _ { k } ) ( f _ { \mathrm { b e s t } } ^ { N } - f _ { \mathrm { o p t } } ) \le \sum _ { k = 0 } ^ { N } t _ { k } ( f ( \mathbf { x } ^ { k } ) - } \end{array}$ $f _ { \mathrm { o p t } } )$ , yields the result (9.20). 

## 9.2.2 Fixed Number of Iterations

Let us begin by fixing the number of iterations N and deduce what the “optimal” stepsizes are in the sense that they bring the right-hand side of (9.20) to a minimum. For that, we will prove the following technical lemma. 

Lemma 9.15. The optimal solution of the problem 

$$
\min _ {t _ {1}, \ldots , t _ {m} > 0} \frac {\alpha + \beta \sum_ {k = 1} ^ {m} t _ {k} ^ {2}}{\sum_ {k = 1} ^ {m} t _ {k}},\tag{9.22}
$$

where $\alpha , \beta > 0$ , is given by $\begin{array} { r } { t _ { k } = \sqrt { \frac { \alpha } { \beta m } } , k = 1 , 2 , \ldots , m } \end{array}$ . The optimal value is $2 { \sqrt { \frac { \alpha \beta } { m } } }$ 

Proof. Denote the objective function of (9.22) by 

$$
\phi (\mathbf {t}) \equiv \frac {\alpha + \beta \sum_ {k = 1} ^ {m} t _ {k} ^ {2}}{\sum_ {k = 1} ^ {m} t _ {k}}.
$$

Note that $\phi$ is a permutation symmetric function, meaning that $\phi ( \mathbf { t } ) = \phi ( \mathbf { P t } )$ for any permutation matrix $\mathbf { P } \in \Lambda _ { m }$ . A consequence of this observation is that if problem (9.22) has an optimal solution, then it necessarily has an optimal solution in which all the variables are the same. To show this, take an arbitrary optimal solution $\mathbf { t } ^ { * }$ and a permutation matrix $\mathbf { P } \in \Lambda _ { m }$ . Since $\phi ( \mathbf { P t ^ { * } } ) = \phi ( \mathbf { t ^ { * } } )$ , it follows that $\mathbf { P t ^ { * } }$ is also an optimal solution of (9.22). Therefore, since $\phi$ is convex over the positive orthant,<sup>51</sup> it follows that 

$$
\frac {1}{m !} \sum_ {\mathbf {P} \in \Lambda_ {m}} \mathbf {P t} ^ {*} = \frac {1}{m} \left( \begin{array}{c} \mathbf {e} ^ {T} \mathbf {t} \\ \vdots \\ \mathbf {e} ^ {T} \mathbf {t} \end{array} \right)
$$

is also an optimal solution, showing that there always exists an optimal solution with equal components. Problem (9.22) therefore reduces to (after substituting $t _ { 1 } = t _ { 2 } = \cdot \cdot \cdot = t _ { m } = t )$ 

$$
\min _ {t > 0} \frac {\alpha + \beta m t ^ {2}}{m t},
$$

whose optimal solution is $\begin{array} { r } { t = \sqrt { \frac { \alpha } { \beta m } } } \end{array}$ , and thus an optimal solution of problem (9.22) is given by $\begin{array} { r } { t _ { k } = \sqrt { \frac { \alpha } { \beta m } } , k = 1 , 2 , . . . , m } \end{array}$ . Substituting this value into $\phi .$ we obtain that the optimal value is $2 { \sqrt { \frac { \alpha \beta } { m } } }$ 

Using Lemma 9.15 with $\begin{array} { r } { \alpha = \Theta ( \mathbf { x } ^ { 0 } ) , \beta = \frac { L _ { f } ^ { 2 } } { 2 \sigma } } \end{array}$ and $m = N + 1$ , we conclude that the minimum of the right-hand side of (9.20) is attained at $\begin{array} { r } { t _ { k } = \frac { \sqrt { 2 \Theta ( \mathbf { x } ^ { 0 } ) \sigma } } { L _ { f } \sqrt { N + 1 } } } \end{array}$ . The $O ( 1 / \sqrt { N } )$ rate of convergence follows immediately. 

Theorem 9.16 $( O ( 1 / { \sqrt { N } } )$ rate of convergence of mirror descent with fixed amount of iterations). Suppose that Assumptions 9.1 and 9.3 hold and that $\| f ^ { \prime } ( \mathbf { x } ) \| _ { * } \le L _ { f }$ for all $\textbf { x } \in \mathbf { \Sigma } C$ for some $L _ { f } ~ > ~ 0$ . Assume that $B _ { \omega } ( \mathbf { x } , \mathbf { x } ^ { 0 } )$ is bounded over $C ,$ , and let $\Theta ( \mathbf { x } ^ { 0 } )$ satisfy 

$$
\Theta (\mathbf {x} ^ {0}) \geq \max _ {\mathbf {x} \in C} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {0}).
$$

Let N be a positive integer, and let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the mirror descent method with 

$$
t _ {k} = \frac {\sqrt {2 \Theta (\mathbf {x} ^ {0}) \sigma}}{L _ {f} \sqrt {N + 1}}, k = 0, 1, \ldots , N.\tag{9.23}
$$

Then 

$$
f _ {\mathrm{best}} ^ {N} - f _ {\mathrm{opt}} \leq \frac {\sqrt {2 \Theta (\mathbf {x} ^ {0})} L _ {f}}{\sqrt {\sigma} \sqrt {N + 1}},
$$

where $f _ { \mathrm { b e s t } } ^ { N }$ is defined in (9.19). 

Proof. By Lemma 9.14, 

$$
f _ {\mathrm{best}} ^ {N} - f _ {\mathrm{opt}} \leq \frac {\Theta (\mathbf {x} ^ {0}) + \frac {L _ {f} ^ {2}}{2 \sigma} \sum_ {k = 0} ^ {N} t _ {k} ^ {2}}{\sum_ {k = 0} ^ {N} t _ {k}}.
$$

Plugging the expression (9.23) for the stepsizes into the above inequality, the result follows. 

Example 9.17 (optimization over the unit simplex). Consider the problem 

$$
\min \{f (\mathbf {x}): \mathbf {x} \in \Delta_ {n} \},
$$

where $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ is proper closed convex and satisfies $\Delta _ { n } \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ Consider two possible algorithms. 

• Euclidean setting. We assume that the underlying norm on $\mathbb { R } ^ { n }$ is the $l _ { 2 ^ { - } }$ norm and $\begin{array} { r } { \omega ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { x } \| _ { 2 } ^ { 2 } } \end{array}$ , which is 1-strongly convex w.r.t. the $l _ { 2 } { \mathrm { - n o r m } }$ . In this case, the mirror descent algorithm is the same as the projected subgradient method: 

$$
\mathbf {x} ^ {k + 1} = P _ {\Delta_ {n}} (\mathbf {x} ^ {k} - t _ {k} f ^ {\prime} (\mathbf {x} ^ {k})).
$$

We will assume that the method starts with the vector $\textstyle \mathbf { x } ^ { 0 } = { \frac { 1 } { n } } \mathbf { e }$ . For this choice, 

$$
\max _ {\mathbf {x} \in \Delta_ {n}} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {0}) = \max _ {\mathbf {x} \in \Delta_ {n}} \frac {1}{2} \left\| \mathbf {x} - \frac {1}{n} \mathbf {e} \right\| _ {2} ^ {2} = \frac {1}{2} \left(1 - \frac {1}{n}\right),
$$

and we will take $\Theta ( \mathbf { x } ^ { 0 } ) = 1$ . By Theorem 9.16, we have that given a positive integer N, by appropriately choosing the stepsizes, we obtain that 

$$
f _ {\mathrm{best}} ^ {N} - f _ {\mathrm{opt}} \leq \underbrace {\frac {\sqrt {2} L _ {f , 2}}{\sqrt {N + 1}}} _ {C _ {\mathrm{e}} ^ {f}},\tag{9.24}
$$

where $L _ { f , 2 } = \operatorname* { m a x } _ { \mathbf { x } \in \Delta _ { n } } \| f ^ { \prime } ( \mathbf { x } ) \| _ { 2 }$ 

• Non-Euclidean setting. Here we assume that the underlying norm on $\mathbb { R } ^ { n }$ is the $l _ { 1 } { \mathrm { - n o r m } }$ and that the convex function ω is chosen as the negative entropy function 

$$
\omega (\mathbf {x}) = \left\{ \begin{array}{l l} \sum_ {i = 1} ^ {n} x _ {i} \log (x _ {i}), & \mathbf {x} \in \mathbb {R} _ {+ +} ^ {n}, \\ \infty & \text { else }. \end{array} \right.\tag{9.25}
$$

By Example 5.27, $\omega + \delta _ { \Delta _ { r } }$ is 1-strongly convex w.r.t. the $l _ { 1 } { \mathrm { - n o r m } }$ . By Example 9.10, the mirror descent method takes the form 

$$
x _ {i} ^ {k + 1} = \frac {x _ {i} ^ {k} e ^ {- t _ {k} f _ {i} ^ {\prime} (\mathbf {x} ^ {k})}}{\sum_ {j = 1} ^ {n} x _ {j} ^ {k} e ^ {- t _ {k} f _ {j} ^ {\prime} (\mathbf {x} ^ {k})}}, i = 1, 2, \ldots , n.
$$

As in the Euclidean setting, we will also initialize the method with $\textstyle \mathbf { x } ^ { 0 } = { \frac { 1 } { n } } \mathbf { e }$ For this choice, using the fact that the Bregman distance coincides with the Kullback–Leibler divergence (see (9.13)), we obtain 

$$
\begin{array}{c} \max _ {\mathbf {x} \in \Delta_ {n}} B _ {\omega} \left(\mathbf {x}, \frac {1}{n} \mathbf {e}\right) = \max _ {\mathbf {x} \in \Delta_ {n}} \sum_ {i = 1} ^ {n} x _ {i} \log (n x _ {i}) = \log (n) + \max _ {\mathbf {x} \in \Delta_ {n}} \sum_ {i = 1} ^ {n} x _ {i} \log x _ {i} \\ = \log (n). \end{array}
$$

We will thus take $\Theta ( { \mathbf { x } } ^ { 0 } ) = \log ( n )$ . By Theorem 9.16, we have that given a positive integer N, by appropriately choosing the stepsizes, we obtain that 

$$
f _ {\mathrm{best}} ^ {N} - f _ {\mathrm{opt}} \leq \underbrace {\frac {\sqrt {2 \log (n)} L _ {f , \infty}}{\sqrt {N + 1}}} _ {C _ {\mathrm{ne}} ^ {f}},\tag{9.26}
$$

where $\begin{array} { r } { L _ { f , \infty } = \operatorname* { m a x } _ { \mathbf { x } \in \Delta _ { n } } \| f ^ { \prime } ( \mathbf { x } ) \| _ { \infty } , } \end{array}$ 

The ratio of the two upper bounds in (9.24) and (9.26) is given by 

$$
\rho^ {f} = \frac {C _ {\mathrm{ne}} ^ {f}}{C _ {\mathrm{e}} ^ {f}} = \sqrt {\log (n)} \frac {L _ {f , \infty}}{L _ {f , 2}}.
$$

Whether or not $\rho ^ { f }$ is greater than 1 (superiority of the Euclidean setting) or smaller than 1 (superiority of the non-Euclidean setting) depends on the properties of the function $f .$ In any case, since $\| \mathbf { y } \| _ { \infty } \leq \| \mathbf { y } \| _ { 2 } \leq { \sqrt { n } } \| \mathbf { y } \| _ { \infty }$ for all $\mathbf { y } \in \mathbb { R } ^ { n }$ , it follows that 

$$
\frac {1}{\sqrt {n}} \leq \frac {L _ {f , \infty}}{L _ {f , 2}} \leq 1,
$$

and hence that 

$$
\frac {\sqrt {\log (n)}}{\sqrt {n}} \leq \rho^ {f} \leq \sqrt {\log (n)}.
$$

Therefore, the ratio between the eficiency estimates ranges between $\frac { \sqrt { \log ( n ) } } { \sqrt { n } }$ (superiority of the non-Euclidean setting) and $\sqrt { \log ( n ) }$ (slight superiority of the Euclidean setting). 

## 9.2.3 Dynamic Stepsize Rule

The constant stepsize rule is relatively easy to analyze but has the disadvantage of requiring the a priori knowledge of the total number of iterations employed by the method. In practical situations, the number of iterations is not fixed a priori, and a stopping criteria diferent than merely fixing the total number of iterations is usually imposed. This is why dynamic (namely, nonconstant) stepsize rules are important. Similarly to the analysis in Chapter 8 for the projected subgradient method, it is possible to use the fundamental inequality for the mirror descent method (Lemma 9.13) to establish convergence results under dynamic stepsize rules. 

Theorem 9.18 (convergence of mirror descent with dynamic stepsizes). Suppose that Assumptions 9.1 and 9.3 hold and that $\| f ^ { \prime } ( \mathbf { x } ) \| _ { * } \leq L _ { f }$ for any $\mathbf { x } \in C$ for some $L _ { f } ~ > ~ 0$ . Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the mirror descent method with positive stepsizes $\{ t _ { k } \} _ { k \geq 0 }$ , and let $\{ f _ { \mathrm { b e s t } } ^ { k } \} _ { k \geq 0 }$ be the sequence of best achieved values defined in (9.19). 

(a) $\begin{array} { r } { I f { \frac { \sum _ { n = 0 } ^ { k } t _ { n } ^ { 2 } } { \sum _ { n = 0 } ^ { k } t _ { n } } } \to 0 } \end{array}$ as $k \to \infty$ , then $f _ { \mathrm { b e s t } } ^ { k }  f _ { \mathrm { o p t } }$ as $k \to \infty$ 

(b) $I f t _ { k }$ is chosen as either (predefined diminishing stepsize) 

$$
t _ {k} = \frac {\sqrt {2 \sigma}}{L _ {f} \sqrt {k + 1}}
$$

or (adaptive stepsize) 

$$
t _ {k} = \left\{ \begin{array}{l l} \frac {\sqrt {2 \sigma}}{\| f ^ {\prime} (\mathbf {x} ^ {k}) \| _ {*} \sqrt {k + 1}}, & f ^ {\prime} (\mathbf {x} ^ {k}) \neq \mathbf {0}, \\ \frac {\sqrt {2 \sigma}}{L _ {f} \sqrt {k + 1}}, & f ^ {\prime} (\mathbf {x} ^ {k}) = \mathbf {0}, \end{array} \right.
$$

then for all $k \geq 1$ 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {L _ {f}}{\sqrt {2 \sigma}} \frac {B _ {\omega} (\mathbf {x} ^ {*} , \mathbf {x} ^ {0}) + 1 + \log (k + 1)}{\sqrt {k + 1}}.
$$

Proof. By the fundamental inequality for mirror descent (Lemma 9.13), we have, for all $n \geq 0$ , 

$$
t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) \leq B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {n}) - B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {n + 1}) + \frac {t _ {n} ^ {2}}{2 \sigma} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| _ {*} ^ {2}.
$$

Summing the above inequality over $n = 0 , 1 , \ldots , k$ gives 

$$
\sum_ {n = 0} ^ {k} t _ {n} (f (\mathbf {x} ^ {n}) - f _ {\mathrm{opt}}) \leq B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {0}) - B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {k + 1}) + \frac {1}{2 \sigma} \sum_ {n = 0} ^ {k} t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| _ {*} ^ {2}.
$$

Using the inequalities $B _ { \omega } ( \mathbf { x } ^ { * } , \mathbf { x } ^ { k + 1 } ) \ge 0$ and $f ( \mathbf { x } ^ { n } ) \geq f _ { \mathrm { b e s t } } ^ { k } ( n \leq k )$ , we obtain 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {B _ {\omega} (\mathbf {x} ^ {*} , \mathbf {x} ^ {0}) + \frac {1}{2 \sigma} \sum_ {n = 0} ^ {k} t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| _ {*} ^ {2}}{\sum_ {n = 0} ^ {k} t _ {n}}.\tag{9.27}
$$

Since $\| f ^ { \prime } ( \mathbf { x } ^ { n } ) \| _ { * } \leq L _ { f }$ , we can deduce that 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {B _ {\omega} (\mathbf {x} ^ {*} , \mathbf {x} ^ {0}) + \frac {L _ {f} ^ {2}}{2 \sigma} \sum_ {n = 0} ^ {k} t _ {n} ^ {2}}{\sum_ {n = 0} ^ {k} t _ {n}}.
$$

Therefore, if $\frac { \sum _ { n = 0 } ^ { k } t _ { n } ^ { 2 } } { \sum _ { n = 0 } ^ { k } t _ { n } }  0 .$ , then $f _ { \mathrm { b e s t } } ^ { k }  f _ { \mathrm { o p t } }$ as $k \to \infty$ , proving claim (a). 

-<sub>To show the validity of claim (b), note that for both stepsize rules we have</sub> $\begin{array} { r } { t _ { n } ^ { 2 } \| f ^ { \prime } ( \mathbf { x } ^ { n } ) \| _ { * } ^ { 2 } \leq \frac { 2 \sigma } { n + 1 } } \end{array}$ and $\begin{array} { r } { t _ { n } \ge \frac { \sqrt { 2 \sigma } } { L _ { f } \sqrt { n + 1 } } } \end{array}$ . Hence, by (9.27), 

$$
f _ {\mathrm{best}} ^ {k} - f _ {\mathrm{opt}} \leq \frac {L _ {f}}{\sqrt {2 \sigma}} \frac {B _ {\omega} (\mathbf {x} ^ {*} , \mathbf {x} ^ {0}) + \sum_ {n = 0} ^ {k} \frac {1}{n + 1}}{\sum_ {n = 0} ^ {k} \frac {1}{\sqrt {n + 1}}},
$$

which, combined with Lemma 8.27(a), yields the desired result. 

Example 9.19 (mirror descent vs. projected subgradient—numerical example). Consider the problem 

$$
\min \left\{\| \mathbf {A x} - \mathbf {b} \| _ {1}: \mathbf {x} \in \Delta_ {n} \right\},\tag{9.28}
$$

where $\mathbf { A } \in \mathbb { R } ^ { n \times n }$ and b $\in \mathbb { R } ^ { n }$ . Following Example 9.17, we consider two methods. The first is the projected subgradient method where $\mathbb { R } ^ { n }$ is assumed to be endowed with the Euclidean $l _ { 2 } { \mathrm { - n o r m } }$ . The update formula is given by 

$$
\mathbf {x} ^ {k + 1} = P _ {\Delta_ {n}} (\mathbf {x} ^ {k} - t _ {k} f ^ {\prime} (\mathbf {x} ^ {k})),
$$

with $f ^ { \prime } ( \mathbf { x } ^ { k } )$ taken as $\mathbf { A } ^ { T } \mathrm { s g n } ( \mathbf { A x } ^ { k } - \mathbf { b } )$ and the stepsize $t _ { k }$ chosen by the adaptive stepsize rule (in practice, $f ^ { \prime } ( \mathbf { x } ^ { k } )$ is never the zeros vector): 

$$
t _ {k} = \frac {\sqrt {2}}{\| f ^ {\prime} (\mathbf {x} ^ {k}) \| _ {2} \sqrt {k + 1}}.
$$

The second method is mirror descent in which the underlying norm on $\mathbb { R } ^ { n }$ is the $l _ { 1 } { \mathrm { - n o r m } }$ and $\omega$ is chosen to be the negative entropy function given in (9.25). In this case, the method has the form (see Example 9.17) 

$$
x _ {i} ^ {k + 1} = \frac {x _ {i} ^ {k} e ^ {- t _ {k} f _ {i} ^ {\prime} (\mathbf {x} ^ {k})}}{\sum_ {j = 1} ^ {n} x _ {j} ^ {k} e ^ {- t _ {k} f _ {j} ^ {\prime} (\mathbf {x} ^ {k})}}, i = 1, 2, \ldots , n,
$$

where here we take 

$$
t _ {k} = \frac {\sqrt {2}}{\| f ^ {\prime} (\mathbf {x} ^ {k}) \| _ {\infty} \sqrt {k + 1}}.
$$

Note that the strong convexity parameter is $\sigma = 1$ in both settings. We created an instance of problem (9.28) with $n = 1 0 0$ by generating the components of A and b independently via a standard normal distribution. The values of $f ( \mathbf { x } ^ { k } ) -$ $f _ { \mathrm { o p t } }$ and $f _ { \mathrm { b e s t } } ^ { k } - f _ { \mathrm { o p t } }$ for both methods are described in Figure 9.1. Evidently, the non-Euclidean method, referred to as md, is superior to the Euclidean projected subgradient method (ps). 

## 9.3 Mirror Descent for the Composite Model<sup>52</sup>

In this section we will consider a more general model than model (9.1), which was discussed in Sections 9.1 and 9.2. Consider the problem 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \{F (\mathbf {x}) \equiv f (\mathbf {x}) + g (\mathbf {x}) \},\tag{9.29}
$$

where the following set of assumptions is made on $f$ and $g$ 

Assumption 9.20 (properties of f and $^ { g ) }$ . 

(A) $f , g : \mathbb { E }  ( - \infty , \infty ]$ are proper closed and convex. 

(B) dom $( g ) \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/a19abc1f43e29fb0cbb9db6617c3ba4c795333f75cfda3a713c31b5cd24116d3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/167bcf20f47ef3222840e8bf61349a5e0228efc4c67f500197da74a3b68763dd.jpg)


Figure 9.1. The values $f ( \mathbf { x } ^ { k } ) - f _ { \mathrm { o p t } }$ and $f _ { \mathrm { b e s t } } ^ { k } \mathrm { ~ - ~ } f _ { \mathrm { o p t } }$ generated by the mirror descent and projected subgradient methods. 

(C) $\| f ^ { \prime } ( \mathbf { x } ) \| _ { * } \leq L _ { f }$ for any $\mathbf { x } \in \mathrm { d o m } ( g ) \ ( L _ { f } > 0 ) .$ 53 

(D) The optimal set of (9.29) is nonempty and denoted by $X ^ { * }$ . The optimal value of the problem is denoted by $F _ { \mathrm { o p t } }$ 

We will also assume, as usual, that we have at our disposal a convex function ω that satisfies the following properties, which are a slight adjustment of the properties in Assumption 9.3. 

Assumption 9.21 (properties of $\omega )$ . 

• $\omega$ is proper closed and convex. 

• $\omega$ is diferentiable over dom $( \partial \omega )$ . 

• dom $( g ) \subseteq \mathrm { d o m } ( \omega )$ 

$\omega + \delta _ { \mathrm { d o m } ( g ) }$ is σ-strongly convex $( \sigma > 0 )$ 

We can obviously ignore the composite structure of problem (9.29) and just try to employ the mirror descent method on the function $F = f + g$ with dom(g) taking the role of C: 

$$
\mathbf {x} ^ {k + 1} = \operatorname * {a r g m i n} _ {\mathbf {x} \in C} \left\{\langle f ^ {\prime} (\mathbf {x} ^ {k}) + g ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} \rangle + \frac {1}{t _ {k}} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {k}) \right\}.\tag{9.30}
$$

However, employing the above scheme might be problematic. First, we did not assume that $C = \operatorname { d o m } ( g )$ is closed, and thus the argmin in (9.30) might be empty. Second, even if the update step is well defined, we did not assume that $g$ is Lipschitz over $C$ like we did on $f$ in Assumption 9.20(C); this is a key element in the convergence analysis of the mirror descent method. Final ${ \mathrm { l y } } ,$ even if $g$ is Lipschitz over $C ,$ it might be that the Lipschitz constant of the sum function $F = f + g$ is much larger than the Lipschitz constant of $f ,$ , and our objective will be to define a method whose eficiency estimate will depend only on the Lipschitz constant of $f$ over dom $( g )$ 

Instead of linearizing both $f$ and $^ { g , }$ as is done in (9.30), we will linearize $f$ and keep $g$ as it is. This leads to the following scheme: 

$$
\mathbf {x} ^ {k + 1} = \operatorname{argmin} _ {\mathbf {x}} \left\{\langle f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} \rangle + g (\mathbf {x}) + \frac {1}{t _ {k}} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {k}) \right\},\tag{9.31}
$$

which can also be written as 

$$
\mathbf {x} ^ {k + 1} = \operatorname{argmin} _ {\mathbf {x}} \left\{\langle t _ {k} f ^ {\prime} (\mathbf {x} ^ {k}) - \nabla \omega (\mathbf {x} ^ {k}), \mathbf {x} \rangle + t _ {k} g (\mathbf {x}) + \omega (\mathbf {x}) \right\}.
$$

The algorithm that performs the above update step will be called the mirror-C method. 

The Mirror-C Method
Initialization: pick $\mathbf{x}^{0}\in\mathrm{dom}(g)\cap\mathrm{dom}(\partial\omega)$ .
General step: for any $k=0,1,2,\ldots$ execute the following steps:
(a) pick a stepsize $t_{k}>0$ and a subgradient $f'(\mathbf{x}^{k})\in\partial f(\mathbf{x}^{k})$ ;
(b) set $\mathbf{x}^{k+1}=\operatorname{argmin}_{\mathbf{x}}\left\{\langle t_{k}f^{\prime}(\mathbf{x}^{k})-\nabla\omega(\mathbf{x}^{k}),\mathbf{x}\rangle+t_{k}g(\mathbf{x})+\omega(\mathbf{x})\right\}$ 

(9.32) 

Remark 9.22. The update formula (9.32) can also be rewritten as 

$$
\mathbf {x} ^ {k + 1} = \operatorname{argmin} _ {\mathbf {x}} \left\{\langle t _ {k} f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} \rangle + t _ {k} g (\mathbf {x}) + B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {k}) \right\}.\tag{9.33}
$$

Remark 9.23 (Euclidean setting—proximal subgradient method). When the underlying space $\mathbb { E }$ is Euclidean and $\begin{array} { r } { \omega ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { x } \| ^ { 2 } } \end{array}$ , then the update formula (9.33) reduces to 

$$
\mathbf {x} ^ {k + 1} = \operatorname * {a r g m i n} _ {\mathbf {x}} \left\{\langle t _ {k} f ^ {\prime} (\mathbf {x} ^ {k}), \mathbf {x} \rangle + t _ {k} g (\mathbf {x}) + \frac {1}{2} \| \mathbf {x} - \mathbf {x} ^ {k} \| ^ {2} \right\},
$$

which, after some rearrangement of terms and removal of constant terms, takes the form 

$$
\mathbf {x} ^ {k + 1} = \operatorname{argmin} _ {\mathbf {x}} \left\{t _ {k} g (\mathbf {x}) + \frac {1}{2} \left\| \mathbf {x} - [ \mathbf {x} ^ {k} - t _ {k} f ^ {\prime} (\mathbf {x} ^ {k}) ] \right\| ^ {2} \right\}.
$$

By the definition of the prox operator (see Chapter 6), the last equation can be rewritten as 

$$
\mathbf {x} ^ {k + 1} = \mathrm{prox} _ {t _ {k} g} (\mathbf {x} ^ {k} - t _ {k} f ^ {\prime} (\mathbf {x} ^ {k})).
$$

Thus, at each iteration the method takes a step toward minus of the subgradient followed by a prox step. The resulting method is called the proximal subgradient method. The method will be discussed extensively in Chapter 10 in the case where $f$ possesses some diferentiability properties. 

Of course, the mirror-C method coincides with the mirror descent method when taking $g = \delta _ { C }$ with C being a nonempty closed and convex set. We begin by showing that the mirror-C method is well defined, meaning that the minimum in (9.32) is uniquely attained at dom(g) ∩ dom(∂ω). 

Theorem 9.24 (mirror-C is well defined). Suppose that Assumptions 9.20 and 9.21 hold. Let $\mathbf { a } \in \mathbb { E } ^ { * }$ . Then the problem 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \{\langle \mathbf {a}, \mathbf {x} \rangle + g (\mathbf {x}) + \omega (\mathbf {x}) \}
$$

has a unique minimizer in dom(g) ∩ dom(∂ω). 

Proof. The proof follows by invoking Lemma 9.7 with $\psi ( \mathbf { x } ) \equiv \langle \mathbf { a } , \mathbf { x } \rangle + g ( \mathbf { x } )$ 0 

The analysis of the mirror-C method is based on arguments similar to those used in Section 9.2 to analyze the mirror descent method. We begin by proving a technical lemma establishing an inequality similar to the one derived in Lemma 9.14. Note that in addition to our basic assumptions, we assume that $g$ is a nonnegative function and that the stepsizes are nonincreasing. 

Lemma 9.25. Suppose that Assumptions 9.20 and 9.21 hold and that g is a nonnegative function. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the mirror-C method with positive nonincreasing stepsizes $\{ t _ { k } \} _ { k \geq 0 }$ . Then for any $\mathbf { x } ^ { * } \in X ^ { * }$ and $k \geq 0$ 

$$
\min _ {n = 0, 1, \ldots , k} F (\mathbf {x} ^ {n}) - F _ {\mathrm{opt}} \leq \frac {t _ {0} g (\mathbf {x} ^ {0}) + B _ {\omega} (\mathbf {x} ^ {*} , \mathbf {x} ^ {0}) + \frac {1}{2 \sigma} \sum_ {n = 0} ^ {k} t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| _ {*} ^ {2}}{\sum_ {n = 0} ^ {k} t _ {n}}.\tag{9.34}
$$

Proof. By the update formula (9.33) and the non-Euclidean second prox theorem (Theorem 9.12) invoked with b ${ \bf \Pi } = { \bf x } ^ { n } , { \bf a } = { \bf x } ^ { n + 1 }$ , and $\psi ( \mathbf { x } ) \equiv t _ { n } \langle f ^ { \prime } ( \mathbf { x } ^ { n } ) , \mathbf { x } \rangle + t _ { n } g ( \mathbf { x } )$ ， we have 

$$
\langle \nabla \omega (\mathbf {x} ^ {n}) - \nabla \omega (\mathbf {x} ^ {n + 1}), \mathbf {u} - \mathbf {x} ^ {n + 1} \rangle \leq t _ {n} \langle f ^ {\prime} (\mathbf {x} ^ {n}), \mathbf {u} - \mathbf {x} ^ {n + 1} \rangle + t _ {n} g (\mathbf {u}) - t _ {n} g (\mathbf {x} ^ {n + 1}).\tag{9.35}
$$

Invoking the three-points lemma (Lemma 9.11) with $\mathbf { a } = \mathbf { x } ^ { n + 1 } , \mathbf { b } = \mathbf { x } ^ { n }$ , and $\mathbf c = \mathbf u$ yields 

$$
\langle \nabla \omega (\mathbf {x} ^ {n}) - \nabla \omega (\mathbf {x} ^ {n + 1}), \mathbf {u} - \mathbf {x} ^ {n + 1} \rangle = B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {n + 1}) + B _ {\omega} (\mathbf {x} ^ {n + 1}, \mathbf {x} ^ {n}) - B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {n}),
$$

which, combined with (9.35), gives 

$$
B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {n + 1}) + B _ {\omega} (\mathbf {x} ^ {n + 1}, \mathbf {x} ^ {n}) - B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {n}) \leq t _ {n} \langle f ^ {\prime} (\mathbf {x} ^ {n}), \mathbf {u} - \mathbf {x} ^ {n + 1} \rangle + t _ {n} g (\mathbf {u}) - t _ {n} g (\mathbf {x} ^ {n + 1}).
$$

Therefore, 

$$
\begin{array}{l} t _ {n} \langle f ^ {\prime} (\mathbf {x} ^ {n}), \mathbf {x} ^ {n} - \mathbf {u} \rangle + t _ {n} g (\mathbf {x} ^ {n + 1}) - t _ {n} g (\mathbf {u}) \\ \leq B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {n}) - B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {n + 1}) - B _ {\omega} (\mathbf {x} ^ {n + 1}, \mathbf {x} ^ {n}) + t _ {n} \langle f ^ {\prime} (\mathbf {x} ^ {n}), \mathbf {x} ^ {n} - \mathbf {x} ^ {n + 1} \rangle \\ \leq B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {n}) - B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {n + 1}) - \frac {\sigma}{2} \| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {n} \| ^ {2} + t _ {n} \langle f ^ {\prime} (\mathbf {x} ^ {n}), \mathbf {x} ^ {n} - \mathbf {x} ^ {n + 1} \rangle \\ = B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {n}) - B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {n + 1}) - \frac {\sigma}{2} \| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {n} \| ^ {2} + \left\langle \frac {t _ {n}}{\sqrt {\sigma}} f ^ {\prime} (\mathbf {x} ^ {n}), \sqrt {\sigma} (\mathbf {x} ^ {n} - \mathbf {x} ^ {n + 1}) \right\rangle \\ \leq B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {n}) - B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {n + 1}) - \frac {\sigma}{2} \| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {n} \| ^ {2} + \frac {t _ {n} ^ {2}}{2 \sigma} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| _ {*} ^ {2} + \frac {\sigma}{2} \| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {n} \| ^ {2} \\ = B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {n}) - B _ {\omega} (\mathbf {u}, \mathbf {x} ^ {n + 1}) + \frac {t _ {n} ^ {2}}{2 \sigma} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| _ {*} ^ {2}. \end{array}
$$

Plugging in $\mathbf { u } = \mathbf { x } ^ { * }$ and using the subgradient inequality, we obtain 

$$
t _ {n} \left[ f (\mathbf {x} ^ {n}) + g (\mathbf {x} ^ {n + 1}) - F _ {\mathrm{opt}} \right] \leq B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {n}) - B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {n + 1}) + \frac {t _ {n} ^ {2}}{2 \sigma} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| _ {*} ^ {2}.
$$

Summing the above over $n = 0 , 1 , \ldots , k$ 

$$
\sum_ {n = 0} ^ {k} t _ {n} \left[ f (\mathbf {x} ^ {n}) + g (\mathbf {x} ^ {n + 1}) - F _ {\mathrm{opt}} \right] \leq B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {0}) - B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {k + 1}) + \frac {1}{2 \sigma} \sum_ {n = 0} ^ {k} t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| _ {*} ^ {2}.
$$

Adding the term $t _ { 0 } g ( \mathbf { x } ^ { 0 } ) - t _ { k } g ( \mathbf { x } ^ { k + 1 } )$ to both sides and using the nonnegativity of the Bregman distance, we get 

$$
t _ {0} (F (\mathbf {x} ^ {0}) - F _ {\mathrm{opt}}) + \sum_ {n = 1} ^ {k} [ t _ {n} f (\mathbf {x} ^ {n}) + t _ {n - 1} g (\mathbf {x} ^ {n}) - t _ {n} F _ {\mathrm{opt}} ]
$$

$$
\leq t _ {0} g (\mathbf {x} ^ {0}) - t _ {k} g (\mathbf {x} ^ {k + 1}) + B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {0}) + \frac {1}{2 \sigma} \sum_ {n = 0} ^ {k} t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| _ {*} ^ {2}.
$$

Using the fact that $t _ { n } \leq t _ { n - 1 }$ and the nonnegativity of $g ( \mathbf { x } ^ { k + 1 } )$ , we conclude that 

$$
\sum_ {n = 0} ^ {k} t _ {n} \left[ F (\mathbf {x} ^ {n}) - F _ {\mathrm{opt}} \right] \leq t _ {0} g (\mathbf {x} ^ {0}) + B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {0}) + \frac {1}{2 \sigma} \sum_ {n = 0} ^ {k} t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| _ {*} ^ {2},
$$

which, combined with the fact that 

$$
\left(\sum_ {n = 0} ^ {k} t _ {n}\right) \left(\min _ {n = 0, 1, \dots , k} F (\mathbf {x} ^ {n}) - F _ {\mathrm{opt}}\right) \leq \sum_ {n = 0} ^ {k} t _ {n} \left[ F (\mathbf {x} ^ {n}) - F _ {\mathrm{opt}} \right],
$$

implies the inequality (9.34). 

Using Lemma 9.25, it is now easy to derive a convergence result under the assumption that the number of iterations is fixed. 

Theorem 9.26 $( O ( 1 / { \sqrt { N } } )$ rate of convergence of mirror-C with fixed amount of iterations). Suppose that Assumptions 9.20 and 9.21 hold and that $g$ is nonnegative. Assume that $B _ { \omega } ( \mathbf { x } , \mathbf { x } ^ { 0 } )$ is bounded above over dom(g), and let $\Theta ( \mathbf { x } ^ { 0 } )$ satisfy 

$$
\Theta (\mathbf {x} ^ {0}) \geq \max _ {\mathbf {x} \in \operatorname{dom} (g)} B (\mathbf {x}, \mathbf {x} ^ {0}).
$$

Suppose that $g ( \mathbf { x } ^ { 0 } ) = 0$ . Let N be a positive integer, and let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the mirror-C method with constant stepsize 

$$
t _ {k} = \frac {\sqrt {2 \Theta (\mathbf {x} ^ {0}) \sigma}}{L _ {f} \sqrt {N}}.\tag{9.36}
$$

Then 

$$
\min _ {n = 0, 1, \dots , N - 1} F (\mathbf {x} ^ {n}) - F _ {\text {opt}} \leq \frac {\sqrt {2 \Theta (\mathbf {x} ^ {0})} L _ {f}}{\sqrt {\sigma} \sqrt {N}}.
$$

Proof. By Lemma 9.25, using the fact that $g ( { \bf x } ^ { 0 } ) ~ = ~ 0$ and the inequalities $\| f ^ { \prime } ( \mathbf { x } ^ { n } ) \| _ { * } \leq L _ { f }$ and $B _ { \omega } ( \mathbf { x } ^ { * } , \mathbf { x } ^ { 0 } ) \leq \Theta ( \mathbf { x } ^ { 0 } )$ , we have 

$$
\min _ {n = 0, 1, \dots , N - 1} F (\mathbf {x} ^ {n}) - F _ {\mathrm{opt}} \leq \frac {\Theta (\mathbf {x} ^ {0}) + \frac {L _ {f} ^ {2}}{2 \sigma} \sum_ {n = 0} ^ {N - 1} t _ {n} ^ {2}}{\sum_ {n = 0} ^ {N - 1} t _ {n}}.
$$

Plugging the expression (9.36) for the stepsizes into the above inequality, the result follows. 

We can also establish a rate of convergence of the mirror-C method with a dynamic stepsize rule. 

Theorem 9.27 (O(log $k / { \sqrt { k } } )$ rate of convergence of mirror-C with dynamic stepsizes). Suppose that Assumptions 9.20 and 9.21 hold and that $g$ is nonnegative. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the mirror-C method with stepsizes $\{ t _ { k } \} _ { k \geq 0 }$ chosen as 

$$
t _ {k} = \frac {\sqrt {2 \sigma}}{L _ {f} \sqrt {k + 1}}.
$$

Then for all $k \geq 1$ 

$$
\min _ {n = 0, 1, \ldots , k} F (\mathbf {x} ^ {n}) - F _ {\mathrm{opt}} \leq \frac {L _ {f}}{\sqrt {2 \sigma}} \frac {B _ {\omega} (\mathbf {x} ^ {*} , \mathbf {x} ^ {0}) + \frac {\sqrt {2 \sigma}}{L _ {f}} g (\mathbf {x} ^ {0}) + 1 + \log (k + 1)}{\sqrt {k + 1}}.\tag{9.37}
$$

Proof. By Lemma 9.25, taking into account the fact that $\begin{array} { r } { t _ { 0 } = \frac { \sqrt { 2 \sigma } } { L _ { f } } } \end{array}$ 2 

$$
\min _ {n = 0, 1, \ldots , k} F (\mathbf {x} ^ {n}) - F _ {\mathrm{opt}} \leq \frac {B _ {\omega} (\mathbf {x} ^ {*} , \mathbf {x} ^ {0}) + \frac {\sqrt {2 \sigma}}{L _ {f}} g (\mathbf {x} ^ {0}) + \frac {1}{2 \sigma} \sum_ {n = 0} ^ {k} t _ {n} ^ {2} \| f ^ {\prime} (\mathbf {x} ^ {n}) \| _ {*} ^ {2}}{\sum_ {n = 0} ^ {k} t _ {n}},\tag{9.38}
$$

which, along with the relations $\begin{array} { r } { t _ { n } ^ { 2 } \| f ^ { \prime } ( \mathbf { x } ^ { n } ) \| _ { * } ^ { 2 } \leq \frac { 2 \sigma } { n + 1 } } \end{array}$ and $\begin{array} { r } { t _ { n } = \frac { \sqrt { 2 \sigma } } { L _ { f } \sqrt { n + 1 } } } \end{array}$ , yields the inequality 

$$
\min _ {n = 0, 1, \ldots , k} F (\mathbf {x} ^ {n}) - f _ {\text {opt}} \leq \frac {L _ {f}}{\sqrt {2 \sigma}} \frac {B _ {\omega} (\mathbf {x} ^ {*} , \mathbf {x} ^ {0}) + \frac {\sqrt {2 \sigma}}{L _ {f}} g (\mathbf {x} ^ {0}) + \sum_ {n = 0} ^ {k} \frac {1}{n + 1}}{\sum_ {n = 0} ^ {k} \frac {1}{\sqrt {n + 1}}}.
$$

The result (9.37) now follows by invoking Lemma 8.27(a). 

Example 9.28. Suppose that the underlying space is $\mathbb { R } ^ { n }$ endowed with the Euclidean $l _ { 2 } \mathrm { - n o r m }$ . Let $f : \mathbb { R } ^ { n } $ R be a convex function, which is Lipschitz over $\mathbb { R } ^ { n }$ 2 implying that there exists $L _ { f } > 0$ for which $\| f ^ { \prime } ( \mathbf { x } ) \| _ { 2 } \leq L _ { f }$ for all $\mathbf { x } \in \mathbb { R } ^ { n }$ . Now consider the problem 

$$
\min _ {\mathbf {x} \in \mathbb {R} _ {+ +} ^ {n}} \left\{F (\mathbf {x}) \equiv f (\mathbf {x}) + \sum_ {i = 1} ^ {n} \frac {1}{x _ {i}} \right\}
$$

with $\omega$ chosen as $\begin{array} { r } { \omega ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { x } \| _ { 2 } ^ { 2 } } \end{array}$ . In this case, the mirror descent and mirror-C methods coincide with the projected subgradient and proximal subgradient methods, respectively. It is not possible to employ the projected subgradient method on the problem—it is not even clear what is the feasible set $C .$ . If we take it as the open set $\mathbb { R } _ { + + } ^ { n }$ , then projections onto $C$ will in general not be in $C .$ . In any case, since $F$ is obviously not Lipschitz, no convergence is guaranteed. On the other hand, employing the proximal subgradient method is definitely possible by taking $\begin{array} { r } { g ( \mathbf { x } ) \equiv \sum _ { i = 1 } ^ { n } \frac { 1 } { x _ { i } } + \delta _ { \mathbb { R } _ { + + } ^ { n } } } \end{array}$ . Both Assumptions 9.20 and 9.21 hold for $f , g$ and $\begin{array} { r } { \omega ( \mathbf { x } ) = \frac { 1 } { 2 } \Vert \mathbf { x } \Vert ^ { 2 } } \end{array}$ , and in addition $g$ is nonnegative. The resulting method is 

$$
\mathbf {x} ^ {k + 1} = \mathrm{prox} _ {t _ {k} g} \left(\mathbf {x} ^ {k} - t _ {k} f ^ {\prime} (\mathbf {x} ^ {k})\right).
$$

The computation of $\mathrm { p r o x } _ { t _ { k } g }$ amounts to solving n cubic scalar equations. 

Example 9.29 (projected subgradient vs. proximal subgradient). Suppose that the underlying space is $\mathbb { R } ^ { n }$ endowed with the Euclidean $l _ { 2 } { \mathrm { - n o r m } }$ and consider the problem 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {n}} \left\{F (\mathbf {x}) \equiv \| \mathbf {A x} - \mathbf {b} \| _ {1} + \lambda \| \mathbf {x} \| _ {1} \right\},\tag{9.39}
$$

where $\mathbf { A } \in \mathbb { R } ^ { m \times n } , \mathbf { b } \in \mathbb { R } ^ { m }$ , and $\lambda > 0$ . We will consider two possible methods to solve the problem: 

• projected subgradient employed on problem (9.39), where here $C = \mathbb { R } ^ { n }$ The method takes the form (when making the choice of the subgradient of $\| \mathbf { y } \| _ { 1 }$ as $\operatorname { s g n } ( \mathbf { y } ) )$ 

$$
\mathbf {x} ^ {k + 1} = \mathbf {x} ^ {k} - t _ {k} (\mathbf {A} ^ {T} \operatorname{sgn} (\mathbf {A x} ^ {k} - \mathbf {b}) + \lambda \operatorname{sgn} (\mathbf {x})).
$$

The stepsize is chosen according to Theorem 8.28 as $\begin{array} { r } { t _ { k } = \frac { 1 } { \| F ^ { \prime } ( \mathbf { x } ^ { k } ) \| _ { 2 } \sqrt { k + 1 } } } \end{array}$ 

• proximal subgradient, where we take $f ( \mathbf { x } ) = \| \mathbf { A x } - \mathbf { b } \| _ { 1 }$ and $g ( \mathbf { x } ) = \lambda \| \mathbf { x } \| _ { 1 }$ 2 so that $F = f + g$ . The method then takes the form 

$$
\mathbf {x} ^ {k + 1} = \operatorname{prox} _ {s _ {k} g} (\mathbf {x} ^ {k} - s _ {k} \mathbf {A} ^ {T} \operatorname{sgn} (\mathbf {A x} ^ {k} - \mathbf {b})).
$$

Since $g ( \mathbf { x } ) = \lambda \| \mathbf { x } \| _ { 1 }$ , it follows that prox $\zeta _ { s _ { k } g }$ is a soft thresholding operator. Specifically, by Example 6.8, $\mathrm { p r o x } _ { s _ { k } g } = \mathcal { T } _ { \lambda s _ { k } }$ , and hence the general update rule becomes 

$$
\mathbf {x} ^ {k + 1} = \mathcal {T} _ {\lambda s _ {k}} (\mathbf {x} ^ {k} - s _ {k} \mathbf {A} ^ {T} \mathrm{sgn} (\mathbf {A x} ^ {k} - \mathbf {b})).
$$

The stepsize is chosen as $\begin{array} { r } { s _ { k } = \frac { 1 } { \| f ^ { \prime } ( \mathbf { x } ^ { k } ) \| _ { 2 } \sqrt { k + 1 } } } \end{array}$ 

A priori it seems that the proximal subgradient method should have an advantage over the projected subgradient method since the eficiency estimate bound of the proximal subgradient method depends on $L _ { f } ,$ while the corresponding constant for the projected subgradient method depends on the larger constant $L _ { F }$ . This observation is also quite apparent in practice. We created an instance of problem (9.39) with $m = 1 0 , n = 1 5$ by generating the components of A and b independently via a standard normal distribution. The values of $F ( \mathbf { x } ^ { k } ) { - } F _ { \mathrm { o p t } }$ for both methods are described in Figure 9.2. Evidently, in this case, the proximal subgradient method is better by orders of magnitude than the projected subgradient method. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/0cd919df2af8c213dab52baf49958cab7611902b811bd8d1bf414ff39e5e41a1.jpg)



Figure 9.2. First 1000 iterations of the projected and proximal subgradient methods employed on problem (9.39). The y-axis describes (in log scale) the quantity $F ( \mathbf { x } ^ { k } ) - F _ { \mathrm { o p t } }$


# Chapter 10 The Proximal Gradient Method

Underlying Space: In this chapter, with the exception of Section 10.9, E is a Euclidean space, meaning a finite dimensional space endowed with an inner product $\langle \cdot , \cdot \rangle$ and the Euclidean norm $\| \cdot \| = \sqrt { \langle \cdot , \cdot \rangle }$ 

## 10.1 The Composite Model

In this chapter we will be mostly concerned with the composite model 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \{F (\mathbf {x}) \equiv f (\mathbf {x}) + g (\mathbf {x}) \},\tag{10.1}
$$

where we assume the following. 

Assumption 10.1. 

(A) $g : \mathbb { E }  ( - \infty , \infty ]$ is proper closed and convex. 

(B) $f : \mathbb { E } \to ( - \infty , \infty ]$ is proper and closed, dom $( f )$ is convex, dom $( g ) \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ and $f$ is $L _ { f } { - } s m o o t h$ over int(dom(f)). 

(C) The optimal set of problem (10.1) is nonempty and denoted by $X ^ { * }$ . The optimal value of the problem is denoted by $F _ { \mathrm { o p t } }$ 

Three special cases of the general model (10.1) are gathered in the following example. 

Example 10.2. 

• Smooth unconstrained minimization. If $g \equiv 0$ and $\operatorname { d o m } ( f ) = \mathbb { E }$ , then (10.1) reduces to the unconstrained smooth minimization problem 

$$
\min _ {\mathbf {x} \in \mathbb {E}} f (\mathbf {x}),
$$

where $f : \mathbb { E } \to \mathbb { R }$ is an $L _ { f }$ -smooth function. 

• Convex constrained smooth minimization. If $g \ = \ \delta _ { C }$ , where C is a nonempty closed and convex set, then (10.1) amounts to the problem of minimizing a diferentiable function over a nonempty closed and convex set: 

$$
\min _ {\mathbf {x} \in C} f (\mathbf {x}),
$$

where here $f$ is $L _ { f } { \mathrm { - s m o o t h } }$ over int $\left( \operatorname { d o m } ( f ) \right)$ and $C \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ 

• l<sub>1</sub>-regularized minimization. Taking $g ( \mathbf { x } ) = \lambda \| \mathbf { x } \| .$ <sub>1</sub> for some $\lambda > 0$ , (10.1) amounts to the l<sub>1</sub>-regularized problem 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \{f (\mathbf {x}) + \lambda \| \mathbf {x} \| _ {1} \}
$$

with $f$ being an $L _ { f }$ -smooth function over the entire space E. 

## 10.2 The Proximal Gradient Method

To understand the idea behind the method for solving (10.1) we are about to study, we begin by revisiting the projected gradient method for solving (10.1) in the case where $g = \delta _ { C }$ with $C$ being a nonempty closed and convex set. In this case, the problem takes the form 

$$
\min \{f (\mathbf {x}): \mathbf {x} \in C \}.\tag{10.2}
$$

The general update step of the projected gradient method for solving (10.2) takes the form 

$$
\mathbf {x} ^ {k + 1} = P _ {C} (\mathbf {x} ^ {k} - t _ {k} \nabla f (\mathbf {x} ^ {k})),
$$

where $t _ { k }$ is the stepsize at iteration k. It is easy to verify that the update step can be also written as (see also Section 9.1 for a similar discussion on the projected subgradient method) 

$$
\mathbf {x} ^ {k + 1} = \operatorname * {a r g m i n} _ {\mathbf {x} \in C} \left\{f (\mathbf {x} ^ {k}) + \langle \nabla f (\mathbf {x} ^ {k}), \mathbf {x} - \mathbf {x} ^ {k} \rangle + \frac {1}{2 t _ {k}} \| \mathbf {x} - \mathbf {x} ^ {k} \| ^ {2} \right\}.
$$

That is, the next iterate is the minimizer over C of the sum of the linearization of the smooth part around the current iterate plus a quadratic prox term. 

Back to the more general model (10.1), it is natural to generalize the above idea and to define the next iterate as the minimizer of the sum of the linearization of $f$ around $\mathbf { x } ^ { k }$ , the nonsmooth function $^ { g , }$ and a quadratic prox term: 

$$
\mathbf {x} ^ {k + 1} = \operatorname * {a r g m i n} _ {\mathbf {x} \in \mathbb {E}} \left\{f (\mathbf {x} ^ {k}) + \langle \nabla f (\mathbf {x} ^ {k}), \mathbf {x} - \mathbf {x} ^ {k} \rangle + g (\mathbf {x}) + \frac {1}{2 t _ {k}} \| \mathbf {x} - \mathbf {x} ^ {k} \| ^ {2} \right\}.\tag{10.3}
$$

After some simple algebraic manipulation and cancellation of constant terms, we obtain that (10.3) can be rewritten as 

$$
\mathbf {x} ^ {k + 1} = \operatorname{argmin} _ {\mathbf {x} \in \mathbb {E}} \left\{t _ {k} g (\mathbf {x}) + \frac {1}{2} \left\| \mathbf {x} - \left(\mathbf {x} ^ {k} - t _ {k} \nabla f (\mathbf {x} ^ {k})\right) \right\| ^ {2} \right\},
$$

which by the definition of the proximal operator is the same as 

$$
\mathbf {x} ^ {k + 1} = \mathrm{prox} _ {t _ {k} g} (\mathbf {x} ^ {k} - t _ {k} \nabla f (\mathbf {x} ^ {k})).
$$

The above method is called the proximal gradient method, as it consists of a gradient step followed by a proximal mapping. From now on, we will take the stepsizes as $\begin{array} { r } { t _ { k } = \frac { 1 } { L _ { k } } } \end{array}$ , leading to the following description of the method. 

The Proximal Gradient Method
Initialization: pick $\mathbf{x}^{0} \in \operatorname{int}(\operatorname{dom}(f))$ .
General step: for any $k = 0, 1, 2, \ldots$ execute the following steps:
(a) pick $L_{k} > 0$ ;
(b) set $\mathbf{x}^{k+1} = \operatorname{prox}_{\frac{1}{L_{k}} g} \left( \mathbf{x}^{k} - \frac{1}{L_{k}} \nabla f(\mathbf{x}^{k}) \right)$ . 

The general update step of the proximal gradient method can be compactly written as 

$$
\mathbf {x} ^ {k + 1} = T _ {L _ {k}} ^ {f, g} (\mathbf {x} ^ {k}),
$$

where $T _ { L } ^ { f , g }$ : int $( \operatorname { d o m } ( f ) ) \to \mathbb { E } \ ( L > 0 )$ is the so-called prox-grad operator defined by 

$$
T _ {L} ^ {f, g} (\mathbf {x}) \equiv \operatorname{prox} _ {\frac {1}{L} g} \left(\mathbf {x} - \frac {1}{L} \nabla f (\mathbf {x})\right).
$$

When the identities of f and g are clear from the context, we will often omit the superscripts $f , g$ and write $T _ { L } ( \cdot )$ instead of $T _ { L } ^ { f , g } ( \cdot )$ 

Later on, we will consider two stepsize strategies, constant and backtracking, where the meaning of “backtracking” slightly changes under the diferent settings that will be considered, and hence several backtracking procedures will be defined. 


Example 10.3. The table below presents the explicit update step of the proximal gradient method when applied to the three particular models discussed in Example $\mathrm { \overline { { 1 0 . 2 . } } ^ { 5 4 } }$ The exact assumptions on the models are described in Example 10.2.


<table><tr><td>Model</td><td>Update step</td><td>Name of method</td></tr><tr><td><eq>\min_{\mathbf{x} \in \mathbb{E}} f(\mathbf{x})</eq></td><td><eq>\mathbf{x}^{k+1} = \mathbf{x}^k - t_k \nabla f(\mathbf{x}^k)</eq></td><td>gradient</td></tr><tr><td><eq>\min_{\mathbf{x} \in C} f(\mathbf{x})</eq></td><td><eq>\mathbf{x}^{k+1} = P_C(\mathbf{x}^k - t_k \nabla f(\mathbf{x}^k))</eq></td><td>projected gradient</td></tr><tr><td><eq>\min_{\mathbf{x} \in \mathbb{E}} \{f(\mathbf{x}) + \lambda \| \mathbf{x}\|_1 \}</eq></td><td><eq>\mathbf{x}^{k+1} = \mathcal{T}_{\lambda t_k}(\mathbf{x}^k - t_k \nabla f(\mathbf{x}^k))</eq></td><td>ISTA</td></tr></table>

The third method is known as the iterative shrinkage-thresholding algorithm (ISTA) in the literature, since at each iteration a soft-thresholding operation (also known as “shrinkage”) is performed. 

## 10.3 Analysis of the Proximal Gradient Method— The Nonconvex Case 55

## 10.3.1 Suficient Decrease

To establish the convergence of the proximal gradient method, we will prove a suficient decrease lemma for composite functions. 

Lemma 10.4 (suficient decrease lemma). Suppose that f and g satisfy properties $\mathrm { ( A ) }$ and (B) of Assumption 10.1. Let $F = f + g$ and $T _ { L } \equiv T _ { L } ^ { f , g }$ . Then for any ${ \textbf { x } } \in$ int(dom(f)) and $\begin{array} { r } { L \in \left( \frac { L _ { f } } { 2 } , \infty \right) } \end{array}$ the following inequality holds: 

$$
F (\mathbf {x}) - F (T _ {L} (\mathbf {x})) \geq \frac {L - \frac {L _ {f}}{2}}{L ^ {2}} \left\| G _ {L} ^ {f, g} (\mathbf {x}) \right\| ^ {2},\tag{10.4}
$$

where $G _ { L } ^ { f , g }$ : int $( \operatorname { d o m } ( f ) ) \to { \mathbb { E } }$ is the operator defined by $G _ { L } ^ { f , g } ( { \bf x } ) = { L } ( { \bf x } - { T } _ { L } ( { \bf x } ) )$ for all $\mathbf { x \in }$ int $\left( \operatorname { d o m } ( f ) \right)$ 

Proof. For the sake of simplicity, we use the shorthand notation $\mathbf { x } ^ { + } = T _ { L } ( \mathbf { x } )$ . By the descent lemma (Lemma 5.7), we have that 

$$
f (\mathbf {x} ^ {+}) \leq f (\mathbf {x}) + \left\langle \nabla f (\mathbf {x}), \mathbf {x} ^ {+} - \mathbf {x} \right\rangle + \frac {L _ {f}}{2} \| \mathbf {x} - \mathbf {x} ^ {+} \| ^ {2}.\tag{10.5}
$$

By the second prox theorem (Theorem 6.39), since $\begin{array} { r } { \mathbf { x } ^ { + } = \mathrm { p r o x } _ { \frac { 1 } { r } g } \left( \mathbf { x } - \frac { 1 } { L } \nabla f ( \mathbf { x } ) \right) } \end{array}$ , we have 

$$
\left\langle \mathbf {x} - \frac {1}{L} \nabla f (\mathbf {x}) - \mathbf {x} ^ {+}, \mathbf {x} - \mathbf {x} ^ {+} \right\rangle \leq \frac {1}{L} g (\mathbf {x}) - \frac {1}{L} g (\mathbf {x} ^ {+}),
$$

from which it follows that 

$$
\left\langle \nabla f (\mathbf {x}), \mathbf {x} ^ {+} - \mathbf {x} \right\rangle \leq - L \left\| \mathbf {x} ^ {+} - \mathbf {x} \right\| ^ {2} + g (\mathbf {x}) - g (\mathbf {x} ^ {+}),
$$

which, combined with (10.5), yields 

$$
f (\mathbf {x} ^ {+}) + g (\mathbf {x} ^ {+}) \leq f (\mathbf {x}) + g (\mathbf {x}) + \left(- L + \frac {L _ {f}}{2}\right) \left\| \mathbf {x} ^ {+} - \mathbf {x} \right\| ^ {2}.
$$

Hence, taking into account the definitions of $\mathbf { x } ^ { + } , G _ { L } ^ { f , g } ( \mathbf { x } )$ and the identities $F ( \mathbf { x } ) =$ $f ( \mathbf { x } ) + g ( \mathbf { x } ) , F ( \mathbf { x } ^ { + } ) = f ( \mathbf { x } ^ { + } ) + g ( \mathbf { x } ^ { + } )$ , the desired result follows. 

## 10.3.2 The Gradient Mapping

The operator $G _ { L } ^ { f , g }$ that appears in the right-hand side of (10.4) is an important mapping that can be seen as a generalization of the notion of the gradient. 

Definition 10.5 (gradient mapping). Suppose that f and g satisfy properties (A) and (B) of Assumption 10.1. Then the gradient mapping is the operator 

$G _ { L } ^ { f , g }$ : int $( \operatorname { d o m } ( f ) ) \to \mathbb { E }$ defined by 

$$
G _ {L} ^ {f, g} (\mathbf {x}) \equiv L \left(\mathbf {x} - T _ {L} ^ {f, g} (\mathbf {x})\right)
$$

for any ${ \textbf { x } } \in$ int $\left( \operatorname { d o m } ( f ) \right)$ 

When the identities of $f$ and g will be clear from the context, we will use the notation $G _ { L }$ instead of $G _ { L } ^ { f , g }$ . With the terminology of the gradient mapping, the update step of the proximal gradient method can be rewritten as 

$$
\mathbf {x} ^ {k + 1} = \mathbf {x} ^ {k} - \frac {1}{L _ {k}} G _ {L _ {k}} (\mathbf {x} ^ {k}).
$$

In the special case where $L = L _ { f }$ , the suficient decrease inequality (10.4) takes a simpler form. 

Corollary 10.6. Under the setting of Lemma 10.4, the following inequality holds for any $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ : 

$$
F (\mathbf {x}) - F (T _ {L _ {f}} (\mathbf {x})) \geq \frac {1}{2 L _ {f}} \left\| G _ {L _ {f}} (\mathbf {x}) \right\| ^ {2}.
$$

The next result shows that the gradient mapping is a generalization of the “usual” gradient operator $\mathbf { x } \mapsto \nabla f ( \mathbf { x } )$ in the sense that they coincide when $g \equiv 0$ and that, for a general $^ { g , }$ the points in which the gradient mapping vanishes are the stationary points of the problem of minimizing $f + g$ Recall (see Definition 3.73) that a point $\mathbf { x } ^ { * } \in \mathrm { d o m } ( g )$ is a stationary point of problem (10.1) if and only if $- \nabla f ( \mathbf { x } ^ { * } ) \in \partial g ( \mathbf { x } ^ { * } )$ and that this condition is a necessary optimality condition for local optimal points (see Theorem 3.72). 

Theorem 10.7. Let f and g satisfy properties $\mathrm { ( A ) }$ and (B) of Assumption 10.1 and let $L > 0$ . Then 

(a) $G _ { L } ^ { f , g _ { 0 } } ( { \bf x } ) = \nabla f ( { \bf x } )$ for any $\mathbf { x \in }$ int(dom(f)), where $g _ { 0 } ( \mathbf { x } ) \equiv 0 ;$ 

(b) for $\mathbf { x } ^ { * } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ , it holds that $G _ { L } ^ { f , g } ( { \bf x } ^ { * } ) = { \bf 0 }$ if and only if $\mathbf { x } ^ { * }$ is a stationary point of problem (10.1). 

Proof. (a) Since prox ${ \bf \nabla } _ { \frac { 1 } { L } g _ { 0 } } ( \mathbf { y } ) = \mathbf { y }$ for all $\mathbf { y } \in \mathbb { E }$ , it follows that 

$$
\begin{array}{c} G _ {L} ^ {f, g _ {0}} (\mathbf {x}) = L (\mathbf {x} - T _ {L} ^ {f, g _ {0}} (\mathbf {x})) = L \left(\mathbf {x} - \operatorname{prox} _ {\frac {1}{L} g _ {0}} \left(\mathbf {x} - \frac {1}{L} \nabla f (\mathbf {x})\right)\right) \\ = L \left(\mathbf {x} - \left(\mathbf {x} - \frac {1}{L} \nabla f (\mathbf {x})\right)\right) = \nabla f (\mathbf {x}). \end{array}
$$

(b) $G _ { L } ^ { f , g } ( \mathbf { x } ^ { * } ) = \mathbf { 0 }$ if and only if $\begin{array} { r } { \mathbf { x } ^ { * } = \mathrm { p r o x } _ { \frac { 1 } { r } g } \left( \mathbf { x } ^ { * } - \frac { 1 } { L } \nabla f ( \mathbf { x } ^ { * } ) \right) } \end{array}$ . By the second prox theorem (Theorem 6.39), the latter relation holds if and only if 

$$
\mathbf {x} ^ {*} - \frac {1}{L} \nabla f (\mathbf {x} ^ {*}) - \mathbf {x} ^ {*} \in \frac {1}{L} \partial g (\mathbf {x} ^ {*}),
$$

that is, if and only if 

$$
- \nabla f (\mathbf {x} ^ {*}) \in \partial g (\mathbf {x} ^ {*}),
$$

which is exactly the condition for stationarity. 

If in addition f is convex, then stationarity is a necessary and suficient optimality condition (Theorem 3.72(b)), which leads to the following corollary. 

Corollary 10.8 (necessary and suficient optimality condition under convexity). Let f and g satisfy properties (A) and (B) of Assumption 10.1, and let $L > 0$ . Suppose that in addition f is convex. Then for $\mathbf { x } ^ { * } \in \mathrm { d o m } ( g ) , G _ { L } ^ { f , g } ( \mathbf { x } ^ { * } ) = \mathbf { 0 }$ $i f$ and only if $\mathbf { x } ^ { * }$ is an optimal solution of problem (10.1). 

We can think of the quantity $\| G _ { L } ( \mathbf { x } ) \|$ as an “optimality measure” in the sense that it is always nonnegative, and equal to zero if and only if x is a stationary point. The next result establishes important monotonicity properties of $\| G _ { L } ( \mathbf { x } ) \|$ w.r.t. the parameter L. 

Theorem 10.9 (monotonicity of the gradient mapping). Suppose that f and g satisfy properties (A) and (B) of Assumption 10.1 and let $G _ { L } \equiv G _ { L } ^ { f , g }$ . Suppose that $L _ { 1 } \ge L _ { 2 } > 0$ . Then 

$$
\| G _ {L _ {1}} (\mathbf {x}) \| \geq \| G _ {L _ {2}} (\mathbf {x}) \|\tag{10.6}
$$

and 

$$
\frac {\| G _ {L _ {1}} (\mathbf {x}) \|}{L _ {1}} \leq \frac {\| G _ {L _ {2}} (\mathbf {x}) \|}{L _ {2}}\tag{10.7}
$$

for any $\mathbf { x \in }$ int(dom(f)). 

Proof. Recall that by the second prox theorem (Theorem 6.39), for any $\mathbf { v } , \mathbf { w } \in \mathbb { E }$ and $L > 0$ , the following inequality holds: 

$$
\langle \mathbf {v} - \mathrm{prox} _ {\frac {1}{L} g} (\mathbf {v}), \mathrm{prox} _ {\frac {1}{L} g} (\mathbf {v}) - \mathbf {w} \rangle \geq \frac {1}{L} g \left(\mathrm{prox} _ {\frac {1}{L} g} (\mathbf {v})\right) - \frac {1}{L} g (\mathbf {w}).
$$

Plugging $\begin{array} { r } { L = L _ { 1 } , \mathbf v = \mathbf x - \frac { 1 } { L _ { 1 } } \nabla f ( \mathbf x ) } \end{array}$ , and $\begin{array} { r } { \mathbf { w } = \mathrm { p r o x } _ { \frac { 1 } { L _ { 2 } } g } \big ( \mathbf { x } - \frac { 1 } { L _ { 2 } } \nabla f ( \mathbf { x } ) \big ) = T _ { L _ { 2 } } ( \mathbf { x } ) } \end{array}$ into the last inequality, it follows that 

$$
\left\langle \mathbf {x} - \frac {1}{L _ {1}} \nabla f (\mathbf {x}) - T _ {L _ {1}} (\mathbf {x}), T _ {L _ {1}} (\mathbf {x}) - T _ {L _ {2}} (\mathbf {x}) \right\rangle \geq \frac {1}{L _ {1}} g (T _ {L _ {1}} (\mathbf {x})) - \frac {1}{L _ {1}} g (T _ {L _ {2}} (\mathbf {x}))
$$

or 

$$
\left\langle \frac {1}{L _ {1}} G _ {L _ {1}} (\mathbf {x}) - \frac {1}{L _ {1}} \nabla f (\mathbf {x}), \frac {1}{L _ {2}} G _ {L _ {2}} (\mathbf {x}) - \frac {1}{L _ {1}} G _ {L _ {1}} (\mathbf {x}) \right\rangle \geq \frac {1}{L _ {1}} g (T _ {L _ {1}} (\mathbf {x})) - \frac {1}{L _ {1}} g (T _ {L _ {2}} (\mathbf {x})).
$$

Exchanging the roles of $L _ { 1 }$ and $L _ { 2 }$ yields the following inequality: 

$$
\left\langle \frac {1}{L _ {2}} G _ {L _ {2}} (\mathbf {x}) - \frac {1}{L _ {2}} \nabla f (\mathbf {x}), \frac {1}{L _ {1}} G _ {L _ {1}} (\mathbf {x}) - \frac {1}{L _ {2}} G _ {L _ {2}} (\mathbf {x}) \right\rangle \geq \frac {1}{L _ {2}} g (T _ {L _ {2}} (\mathbf {x})) - \frac {1}{L _ {2}} g (T _ {L _ {1}} (\mathbf {x})).
$$

Multiplying the first inequality by $L _ { 1 }$ and the second by $L _ { 2 }$ and adding them, we obtain 

$$
\left\langle G _ {L _ {1}} (\mathbf {x}) - G _ {L _ {2}} (\mathbf {x}), \frac {1}{L _ {2}} G _ {L _ {2}} (\mathbf {x}) - \frac {1}{L _ {1}} G _ {L _ {1}} (\mathbf {x}) \right\rangle \geq 0,
$$

which after some expansion of terms can be seen to be the same as 

$$
\frac {1}{L _ {1}} \| G _ {L _ {1}} (\mathbf {x}) \| ^ {2} + \frac {1}{L _ {2}} \| G _ {L _ {2}} (\mathbf {x}) \| ^ {2} \leq \left(\frac {1}{L _ {1}} + \frac {1}{L _ {2}}\right) \langle G _ {L _ {1}} (\mathbf {x}), G _ {L _ {2}} (\mathbf {x}) \rangle .
$$

Using the Cauchy–Schwarz inequality, we obtain that 

$$
\frac {1}{L _ {1}} \| G _ {L _ {1}} (\mathbf {x}) \| ^ {2} + \frac {1}{L _ {2}} \| G _ {L _ {2}} (\mathbf {x}) \| ^ {2} \leq \left(\frac {1}{L _ {1}} + \frac {1}{L _ {2}}\right) \| G _ {L _ {1}} (\mathbf {x}) \| \cdot \| G _ {L _ {2}} (\mathbf {x}) \|.\tag{10.8}
$$

Note that if $G _ { L _ { 2 } } ( \mathbf { x } ) = \mathbf { 0 }$ , then by the last inequality, $G _ { L _ { 1 } } ( \mathbf { x } ) = \mathbf { 0 }$ , implying that in this case the inequalities (10.6) and (10.7) hold trivially. Assume then that $G _ { L _ { 2 } } ( \mathbf { x } ) \neq \mathbf { 0 }$ and define $\begin{array} { r } { t = \frac { | | G _ { L _ { 1 } } ^ { * } ( \mathbf { x } ) | | } { | | G _ { L _ { 2 } } ( \mathbf { x } ) | | } } \end{array}$ . Then, by (10.8), 

$$
\frac {1}{L _ {1}} t ^ {2} - \left(\frac {1}{L _ {1}} + \frac {1}{L _ {2}}\right) t + \frac {1}{L _ {2}} \leq 0.
$$

Since the roots of the quadratic function on the left-hand side of the above inequality are $\begin{array} { r } { t = 1 , \frac { L _ { 1 } } { L _ { 2 } } } \end{array}$ , we obtain that 

$$
1 \leq t \leq \frac {L _ {1}}{L _ {2}},
$$

showing that 

$$
\left\| G _ {L _ {2}} (\mathbf {x}) \right\| \leq \left\| G _ {L _ {1}} (\mathbf {x}) \right\| \leq \frac {L _ {1}}{L _ {2}} \left\| G _ {L _ {2}} (\mathbf {x}) \right\|. \quad \square
$$

A straightforward result of the nonexpansivity of the prox operator and the L<sub>f</sub>-smoothness of $f$ over int(dom(f)) is that $G _ { L } ( \cdot )$ is Lipschitz continuous with constant $2 L + L _ { f }$ . Indeed, for any $\mathbf { x } , \mathbf { y } \in$ int(dom(f)), 

$$
\begin{array}{l} \| G _ {L} (\mathbf {x}) - G _ {L} (\mathbf {y}) \| = L \left\| \mathbf {x} - \operatorname{prox} _ {\frac {1}{L} g} \left(\mathbf {x} - \frac {1}{L} \nabla f (\mathbf {x})\right) - \mathbf {y} + \operatorname{prox} _ {\frac {1}{L} g} \left(\mathbf {y} - \frac {1}{L} \nabla f (\mathbf {y})\right) \right\| \\ \leq L \| \mathbf {x} - \mathbf {y} \| + L \left\| \operatorname{prox} _ {\frac {1}{L} g} \left(\mathbf {x} - \frac {1}{L} \nabla f (\mathbf {x})\right) - \operatorname{prox} _ {\frac {1}{L} g} \left(\mathbf {y} - \frac {1}{L} \nabla f (\mathbf {y})\right) \right\| \\ \leq L \| \mathbf {x} - \mathbf {y} \| + L \left\| \left(\mathbf {x} - \frac {1}{L} \nabla f (\mathbf {x})\right) - \left(\mathbf {y} - \frac {1}{L} \nabla f (\mathbf {y})\right) \right\| \\ \leq 2 L \| \mathbf {x} - \mathbf {y} \| + \| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| \\ \leq (2 L + L _ {f}) \| \mathbf {x} - \mathbf {y} \|. \end{array}
$$

In particular, for $L = L _ { f }$ , we obtain the inequality 

$$
\left\| G _ {L _ {f}} (\mathbf {x}) - G _ {L _ {f}} (\mathbf {y}) \right\| \leq 3 L _ {f} \| \mathbf {x} - \mathbf {y} \|.
$$

The above discussion is summarized in the following lemma. 

Lemma 10.10 (Lipschitz continuity of the gradient mapping). Let f and g satisfy properties (A) and (B) of Assumption 10.1. Let $G _ { L } = G _ { L } ^ { f , g }$ . Then 

(a) $\| G _ { L } ( \mathbf { x } ) - G _ { L } ( \mathbf { y } ) \| \leq ( 2 L + L _ { f } ) \| \mathbf { x } - \mathbf { y } \|$ for any $\mathbf { x } , \mathbf { y } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ ; 

(b) $\| G _ { L _ { f } } ( \mathbf { x } ) - G _ { L _ { f } } ( \mathbf { y } ) \| \leq 3 L _ { f } \| \mathbf { x } - \mathbf { y } \| \ f o r \ a n y \ \mathbf { x } , \mathbf { y } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ 

Lemma 10.11 below shows that when $f$ is assumed to be convex and $L _ { f ^ { - } }$ smooth over the entire space, then the operator $\frac { 3 } { 4 L _ { f } } G _ { L _ { f } }$ is firmly nonexpansive. A direct consequence is that $G _ { L _ { f } }$ is Lipschitz continuous with constant $\textstyle { \frac { 4 L _ { f } } { 3 } }$ 

Lemma 10.11 (firm nonexpansivity of $\frac { 3 } { 4 L _ { f } } G _ { L _ { f } } )$ . Let f be a convex and $L _ { f ^ { - } }$ smooth function $( L _ { f } > 0 )$ , and let $g : \mathbb { E }  ( - \dot { \infty } , \infty ]$ be a proper closed and convex function. Then 

(a) the gradient mapping $G _ { L _ { f } } \equiv G _ { L _ { f } } ^ { f , g }$ satisfies the relation 

$$
\left\langle G _ {L _ {f}} (\mathbf {x}) - G _ {L _ {f}} (\mathbf {y}), \mathbf {x} - \mathbf {y} \right\rangle \geq \frac {3}{4 L _ {f}} \left\| G _ {L _ {f}} (\mathbf {x}) - G _ {L _ {f}} (\mathbf {y}) \right\| ^ {2}\tag{10.9}
$$

for any x, $\mathbf { y } \in \mathbb { E } ;$ 

$$
\text {(b)} \| G _ {L _ {f}} (\mathbf {x}) - G _ {L _ {f}} (\mathbf {y}) \| \leq \frac {4 L _ {f}}{3} \| \mathbf {x} - \mathbf {y} \| \text {   for   any   } \mathbf {x}, \mathbf {y} \in \mathbb {E}.
$$

Proof. Part (b) is a direct consequence of $\mathrm { ( a ) }$ and the Cauchy–Schwarz inequality. We will therefore prove $\mathrm { ( a ) }$ . To simplify the presentation, we will use the notation $L = L _ { f }$ . By the firm nonexpansivity of the prox operator (Theorem $6 . 4 2 ( \mathrm { a } ) )$ , it follows that for any $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ 

$$
\left\langle T _ {L} (\mathbf {x}) - T _ {L} (\mathbf {y}), \left(\mathbf {x} - \frac {1}{L} \nabla f (\mathbf {x})\right) - \left(\mathbf {y} - \frac {1}{L} \nabla f (\mathbf {y})\right) \right\rangle \geq \| T _ {L} (\mathbf {x}) - T _ {L} (\mathbf {y}) \| ^ {2},
$$

where $T _ { L } \equiv T _ { L } ^ { f , g }$ is the prox-grad mapping. Since $\begin{array} { r } { T _ { L } = \mathcal { I } - \frac { 1 } { L } G _ { L } } \end{array}$ , we obtain that 

$$
\left\langle \left(\mathbf {x} - \frac {1}{L} G _ {L} (\mathbf {x})\right) - \left(\mathbf {y} - \frac {1}{L} G _ {L} (\mathbf {y})\right), \left(\mathbf {x} - \frac {1}{L} \nabla f (\mathbf {x})\right) - \left(\mathbf {y} - \frac {1}{L} \nabla f (\mathbf {y})\right) \right\rangle
$$

$$
\geq \left\| \left(\mathbf {x} - \frac {1}{L} G _ {L} (\mathbf {x})\right) - \left(\mathbf {y} - \frac {1}{L} G _ {L} (\mathbf {y})\right) \right\| ^ {2},
$$

which is the same as 

$$
\left\langle \left(\mathbf {x} - \frac {1}{L} G _ {L} (\mathbf {x})\right) - \left(\mathbf {y} - \frac {1}{L} G _ {L} (\mathbf {y})\right), (G _ {L} (\mathbf {x}) - \nabla f (\mathbf {x})) - (G _ {L} (\mathbf {y}) - \nabla f (\mathbf {y})) \right\rangle \geq 0.
$$

Therefore, 

$$
\begin{array}{c} \langle G _ {L} (\mathbf {x}) - G _ {L} (\mathbf {y}), \mathbf {x} - \mathbf {y} \rangle \geq \frac {1}{L} \left\| G _ {L} (\mathbf {x}) - G _ {L} (\mathbf {y}) \right\| ^ {2} + \langle \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}), \mathbf {x} - \mathbf {y} \rangle \\ - \frac {1}{L} \left\langle G _ {L} (\mathbf {x}) - G _ {L} (\mathbf {y}), \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \right\rangle . \end{array}
$$

Since $f$ is L-smooth, it follows from Theorem 5.8 (equivalence between (i) and (iv)) that 

$$
\langle \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}), \mathbf {x} - \mathbf {y} \rangle \geq \frac {1}{L} \| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| ^ {2}.
$$

Consequently, 

$$
\begin{array}{r l} & L \left\langle G _ {L} (\mathbf {x}) - G _ {L} (\mathbf {y}), \mathbf {x} - \mathbf {y} \right\rangle \geq \| G _ {L} (\mathbf {x}) - G _ {L} (\mathbf {y}) \| ^ {2} + \| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| ^ {2} \\ & \qquad - \left\langle G _ {L} (\mathbf {x}) - G _ {L} (\mathbf {y}), \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \right\rangle . \end{array}
$$

From the Cauchy–Schwarz inequality we get 

$$
\begin{array}{r l} L \left\langle G _ {L} (\mathbf {x}) - G _ {L} (\mathbf {y}), \mathbf {x} - \mathbf {y} \right\rangle & \geq \| G _ {L} (\mathbf {x}) - G _ {L} (\mathbf {y}) \| ^ {2} + \| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| ^ {2} \\ & \quad - \| G _ {L} (\mathbf {x}) - G _ {L} (\mathbf {y}) \| \cdot \| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \|. \end{array}\tag{10.10}
$$

By denoting $\alpha = \left\| G _ { L } \left( \mathbf { x } \right) - G _ { L } \left( \mathbf { y } \right) \right\|$ and $\beta = \| \nabla f \left( \mathbf { x } \right) - \nabla f \left( \mathbf { y } \right) \|$ , the right-hand side of (10.10) reads as $\alpha ^ { 2 } + \beta ^ { 2 } - \alpha \beta$ and satisfies 

$$
\alpha^ {2} + \beta^ {2} - \alpha \beta = \frac {3}{4} \alpha^ {2} + \left(\frac {\alpha}{2} - \beta\right) ^ {2} \geq \frac {3}{4} \alpha^ {2},
$$

which, combined with (10.10), yields the inequality 

$$
L \left\langle G _ {L} (\mathbf {x}) - G _ {L} (\mathbf {y}), \mathbf {x} - \mathbf {y} \right\rangle \geq \frac {3}{4} \left\| G _ {L} (\mathbf {x}) - G _ {L} (\mathbf {y}) \right\| ^ {2}.
$$

Thus, (10.9) holds. 

The next result shows a diferent kind of a monotonicity property of the gradient mapping norm under the setting of Lemma 10.11—the norm of the gradient mapping does not increase if a prox-grad step is employed on its argument. 

Lemma 10.12 (monotonicity of the norm of the gradient mapping w.r.t. the prox-grad operator).<sup>56</sup> Let f be a convex and L -smooth function $\left( L _ { f } > 0 \right)$ and let $g : \mathbb { E }  ( - \infty , \infty ]$ be a proper closed and convex function. Then for any $\mathbf { x } \in \mathbb { E }$ 2 

$$
\left\| G _ {L _ {f}} \left(T _ {L _ {f}} (\mathbf {x})\right) \right\| \leq \left\| G _ {L _ {f}} (\mathbf {x}) \right\|,
$$

where $G _ { L _ { f } } \equiv G _ { L _ { f } } ^ { f , g }$ and $T _ { L _ { f } } \equiv T _ { L _ { f } } ^ { f , g }$ 

Proof. Let $\mathbf { x } \in \mathbb { E }$ . We will use the shorthand notation $\mathbf { x } ^ { + } = T _ { L _ { f } } ( \mathbf { x } )$ . By Theorem 5.8 (equivalence between (i) and (iv)), it follows that 

$$
\| \nabla f (\mathbf {x} ^ {+}) - \nabla f (\mathbf {x}) \| ^ {2} \leq L _ {f} \langle \nabla f (\mathbf {x} ^ {+}) - \nabla f (\mathbf {x}), \mathbf {x} ^ {+} - \mathbf {x} \rangle .\tag{10.11}
$$

Denoting $\mathbf { a } = \nabla f ( \mathbf { x } ^ { + } ) - \nabla f ( \mathbf { x } )$ and ${ \bf b } = { \bf x } ^ { + } - { \bf x }$ , inequality (10.11) can be rewritten as $\| \mathbf { a } \| ^ { 2 } \leq L _ { f } \langle \mathbf { a } , \mathbf { b } \rangle$ , which is the same as 

$$
\left\| \mathbf {a} - \frac {L _ {f}}{2} \mathbf {b} \right\| ^ {2} \leq \frac {L _ {f} ^ {2}}{4} \| \mathbf {b} \| ^ {2}
$$

and as 

$$
\left\| \frac {1}{L _ {f}} \mathbf {a} - \frac {1}{2} \mathbf {b} \right\| \leq \frac {1}{2} \| \mathbf {b} \|.
$$

Using the triangle inequality, 

$$
\left\| \frac {1}{L _ {f}} \mathbf {a} - \mathbf {b} \right\| \leq \left\| \frac {1}{L _ {f}} \mathbf {a} - \mathbf {b} + \frac {1}{2} \mathbf {b} \right\| + \frac {1}{2} \| \mathbf {b} \| \leq \| \mathbf {b} \|.
$$

Plugging the expressions for a and b into the above inequality, we obtain that 

$$
\left\| \mathbf {x} - \frac {1}{L _ {f}} \nabla f (\mathbf {x}) - \mathbf {x} ^ {+} + \frac {1}{L _ {f}} \nabla f (\mathbf {x} ^ {+}) \right\| \leq \| \mathbf {x} ^ {+} - \mathbf {x} \|.
$$

Combining the above inequality with the nonexpansivity of the prox operator (Theorem 6.42(b)), we finally obtain 

$$
\begin{array}{r l} & {\| G _ {L _ {f}} (T _ {L _ {f}} (\mathbf {x})) \| = \| G _ {L _ {f}} (\mathbf {x} ^ {+}) \| = L _ {f} \| \mathbf {x} ^ {+} - T _ {L _ {f}} (\mathbf {x} ^ {+}) \| = L _ {f} \| T _ {L _ {f}} (\mathbf {x}) - T _ {L _ {f}} (\mathbf {x} ^ {+}) \|} \\ & {\qquad = L _ {f} \left\| \mathrm{prox} _ {\frac {1}{L _ {f}} g} \left(\mathbf {x} - \frac {1}{L _ {f}} \nabla f (\mathbf {x})\right) - \mathrm{prox} _ {\frac {1}{L _ {f}} g} \left(\mathbf {x} ^ {+} - \frac {1}{L _ {f}} \nabla f (\mathbf {x} ^ {+})\right) \right\|} \\ & {\qquad \leq L _ {f} \left\| \mathbf {x} - \frac {1}{L _ {f}} \nabla f (\mathbf {x}) - \mathbf {x} ^ {+} + \frac {1}{L _ {f}} \nabla f (\mathbf {x} ^ {+}) \right\|} \\ & {\qquad \leq L _ {f} \| \mathbf {x} ^ {+} - \mathbf {x} \| = L _ {f} \| T _ {L _ {f}} (\mathbf {x}) - \mathbf {x} \| = \| G _ {L _ {f}} (\mathbf {x}) \|,} \end{array}
$$

which is the desired result. 

## 10.3.3 Convergence of the Proximal Gradient Method— The Nonconvex Case

We will now analyze the convergence of the proximal gradient method under the validity of Assumption 10.1. Note that we do not assume at this stage that $f$ is convex. The two stepsize strategies that will be considered are constant and backtracking. 

• Constant. $\begin{array} { r } { L _ { k } = \bar { L } \in \left( \frac { L _ { f } } { 2 } , \infty \right) } \end{array}$ for all k. 

• Backtracking procedure B1. The procedure requires three parameters $( s , \gamma , \eta )$ , where $s > 0 , \gamma \in ( 0 , 1 )$ , and $\eta > 1$ . The choice of $L _ { k }$ is done as follows. First, $L _ { k }$ is set to be equal to the initial guess s. Then, while 

$$
F (\mathbf {x} ^ {k}) - F (T _ {L _ {k}} (\mathbf {x} ^ {k})) <   \frac {\gamma}{L _ {k}} \| G _ {L _ {k}} (\mathbf {x} ^ {k}) \| ^ {2},
$$

we set $L _ { k } : = \eta L _ { k }$ . In other words, $L _ { k }$ is chosen as $L _ { k } = s \eta ^ { i _ { k } }$ , where $i _ { k }$ is the smallest nonnegative integer for which the condition 

$$
F (\mathbf {x} ^ {k}) - F (T _ {s \eta^ {i _ {k}}} (\mathbf {x} ^ {k})) \geq \frac {\gamma}{s \eta^ {i _ {k}}} \| G _ {s \eta^ {i _ {k}}} (\mathbf {x} ^ {k}) \| ^ {2}
$$

is satisfied. 

Remark 10.13. Note that the backtracking procedure is finite under Assumption 10.1. Indeed, plugging $\mathbf { x } = \mathbf { x } ^ { k }$ into (10.4), we obtain 

$$
F (\mathbf {x} ^ {k}) - F (T _ {L} (\mathbf {x} ^ {k})) \geq \frac {L - \frac {L _ {f}}{2}}{L ^ {2}} \left\| G _ {L} (\mathbf {x} ^ {k}) \right\| ^ {2}.\tag{10.12}
$$

If $\begin{array} { r } { L \geq \frac { L _ { f } } { 2 ( 1 - \gamma ) } } \end{array}$ , then $\begin{array} { r } { \frac { L - \frac { L _ { f } } { 2 } } { L } \geq \gamma } \end{array}$ , and hence, by (10.12), the inequality 

$$
F (\mathbf {x} ^ {k}) - F (T _ {L} (\mathbf {x} ^ {k})) \geq \frac {\gamma}{L} \| G _ {L} (\mathbf {x} ^ {k}) \| ^ {2}
$$

holds, implying that the backtracking procedure must end when $\begin{array} { r } { L _ { k } \geq \frac { L _ { f } } { 2 ( 1 - \gamma ) } } \end{array}$ 

We can also compute an upper bound on $L _ { k } .$ either $L _ { k }$ is equal to s, or the backtracking procedure is invoked, meaning that $\begin{array} { r } { \frac { L _ { k } } { \eta } } \end{array}$ did not satisfy the backtracking condition, which by the above discussion implies that $\begin{array} { r } { \frac { L _ { k } } { \eta } < \frac { L _ { f } } { 2 ( 1 - \gamma ) } } \end{array}$ , so that $L _ { k } ~ <$ $\frac { \eta L _ { f } } { 2 ( 1 - \gamma ) }$ . To summarize, in the backtracking procedure B1, the parameter $L _ { k }$ satisfies 

$$
L _ {k} \leq \max \left\{s, \frac {\eta L _ {f}}{2 (1 - \gamma)} \right\}.\tag{10.13}
$$

The convergence of the proximal gradient method in the nonconvex case is heavily based on the suficient decrease lemma (Lemma 10.4). We begin with the following lemma showing that consecutive function values of the sequence generated by the proximal gradient method decrease by at least a constant times the squared norm of the gradient mapping. 

Lemma 10.14 (suficient decrease of the proximal gradient method). $S u p \ –$ pose that Assumption 10.1 holds. Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the proximal gradient method for solving problem (10.1) with either a constant stepsize defined by $\begin{array} { r } { L _ { k } = \bar { L } \in \left( \frac { L _ { f } } { 2 } , \infty \right) } \end{array}$ or with a stepsize chosen by the backtracking procedure B1 with parameters $( s , \gamma , \eta )$ , where $s > 0 , \gamma \in ( 0 , 1 ) , \eta > 1$ . Then for any $k \geq 0$ 

$$
F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \geq M \| G _ {d} (\mathbf {x} ^ {k}) \| ^ {2},\tag{10.14}
$$

where 

$$
M = \left\{ \begin{array}{l l} \frac {\bar {L} - \frac {L _ {f}}{2}}{\big (\bar {L} \big) ^ {2}}, & \text {constant stepsize}, \\ \frac {\gamma}{\max \Big \{s , \frac {\eta L _ {f}}{2 (1 - \gamma)} \Big \}}, & \text {backtracking}, \end{array} \right.\tag{10.15}
$$

and 

$$
d = \left\{ \begin{array}{l l} \bar {L}, & \text { constant   stepsize }, \\ s, & \text { backtracking }. \end{array} \right.\tag{10.16}
$$

Proof. The result for the constant stepsize setting follows by plugging $L = \bar { L }$ and $\mathbf { x } = \mathbf { x } ^ { k }$ into (10.4). As for the case where the backtracking procedure is used, by its definition we have 

$$
F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \geq \frac {\gamma}{L _ {k}} \| G _ {L _ {k}} (\mathbf {x} ^ {k}) \| ^ {2} \geq \frac {\gamma}{\max \left\{s , \frac {\eta L _ {f}}{2 (1 - \gamma)} \right\}} \| G _ {L _ {k}} (\mathbf {x} ^ {k}) \| ^ {2},
$$

where the last inequality follows from the upper bound on $L _ { k }$ given in (10.13). The result for the case where the backtracking procedure is invoked now follows by the monotonicity property of the gradient mapping (Theorem 10.9) along with the bound $L _ { k } \ge s$ , which imply the inequality $\| G _ { L _ { k } } ( \mathbf { x } ^ { \bar { k } } ) \| \geq \| G _ { s } ( \mathbf { x } ^ { k } ) \|$ 0 

We are now ready to prove the convergence of the norm of the gradient mapping to zero and that limit points of the sequence generated by the method are stationary points of problem (10.1). 

Theorem 10.15 (convergence of the proximal gradient method—nonconvex case). Suppose that Assumption 10.1 holds and let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the proximal gradient method for solving problem (10.1) either with a constant stepsize defined by $\begin{array} { r } { L _ { k } = \bar { L } \in \left( \frac { L _ { f } } { 2 } , \infty \right) } \end{array}$ or with a stepsize chosen by the backtracking procedure B1 with parameters $( s , \gamma , \eta )$ , where $s > 0 , \gamma \in ( 0 , 1 )$ , and $\eta > 1$ . Then 

(a) the sequence $\{ F ( \mathbf { x } ^ { k } ) \} _ { k \geq 0 }$ is nonincreasing. In addition, $F ( \mathbf { x } ^ { k + 1 } ) < F ( \mathbf { x } ^ { k } )$ if and only $i f \mathbf { x } ^ { k }$ is not a stationary point of (10.1); 

(b) $G _ { d } ( \mathbf { x } ^ { k } )  \mathbf { 0 }$ as $k \to \infty$ , where d is given in (10.16); 

(c) 

$$
\min _ {n = 0, 1, \dots , k} \| G _ {d} (\mathbf {x} ^ {n}) \| \leq \frac {\sqrt {F (\mathbf {x} ^ {0}) - F _ {\mathrm{opt}}}}{\sqrt {M (k + 1)}},\tag{10.17}
$$

where M is given in (10.15); 

(d) all limit points of the sequence $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ are stationary points of problem (10.1). 

Proof. (a) By Lemma 10.14 we have that 

$$
F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \geq M \| G _ {d} (\mathbf {x} ^ {k}) \| ^ {2},\tag{10.18}
$$

from which it readily follows that $F ( \mathbf { x } ^ { k } ) \geq F ( \mathbf { x } ^ { k + 1 } )$ . If $\mathbf { x } ^ { k }$ is not a stationary point of problem (10.1), then $G _ { d } ( \mathbf { x } ^ { k } ) \neq \mathbf { 0 }$ , and hence, by (10.18), $F ( \mathbf { x } ^ { k } ) > F ( \mathbf { x } ^ { k + 1 } )$ . If $\mathbf { x } ^ { k }$ is a stationary point of problem (10.1), then $G _ { L _ { k } } ( \mathbf { x } ^ { k } ) = \mathbf { 0 }$ , from which it follows that $\begin{array} { r } { { \bf x } ^ { k + 1 } = { \bf x } ^ { k } - \frac { 1 } { I _ { s } } G _ { L _ { k } } ( \bar { \bf x } ^ { k } ) = { \bf x } ^ { k } } \end{array}$ , and consequently $\dot { F } ( \mathbf { x } ^ { k } ) = F ( \mathbf { x } ^ { k + 1 } )$ 

(b) Since the sequence $\{ F ( \mathbf { x } ^ { k } ) \} _ { k \geq 0 }$ is nonincreasing and bounded below, it converges. Thus, in particular, $F ( \mathbf { x } ^ { k } ) - F ( \mathbf { x } ^ { k + 1 } )  0$ as $k  \infty$ , which, combined with (10.18), implies that $\| G _ { d } ( \mathbf { x } ^ { k } ) \|  0$ as $k \to \infty$ 

(c) Summing the inequality 

$$
F (\mathbf {x} ^ {n}) - F (\mathbf {x} ^ {n + 1}) \geq M \| G _ {d} (\mathbf {x} ^ {n}) \| ^ {2}
$$

over $n = 0 , 1 , \ldots , k .$ , we obtain 

$$
F (\mathbf {x} ^ {0}) - F (\mathbf {x} ^ {k + 1}) \geq M \sum_ {n = 0} ^ {k} \| G _ {d} (\mathbf {x} ^ {n}) \| ^ {2} \geq M (k + 1) \min _ {n = 0, 1, \dots , k} \| G _ {d} (\mathbf {x} ^ {n}) \| ^ {2}.
$$

Using the fact that $F ( \mathbf { x } ^ { k + 1 } ) \geq F _ { \mathrm { o p t } }$ , the inequality (10.17) follows. 

(d) Let x¯ be a limit point of $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ . Then there exists a subsequence $\{ { \bf x } ^ { k _ { j } } \} _ { j \ge 0 }$ converging to x¯. For any $j \geq 0$ 9 

$$
\| G _ {d} (\bar {\mathbf {x}}) \| \leq \| G _ {d} (\mathbf {x} ^ {k _ {j}}) - G _ {d} (\bar {\mathbf {x}}) \| + \| G _ {d} (\mathbf {x} ^ {k _ {j}}) \| \leq (2 d + L _ {f}) \| \mathbf {x} ^ {k _ {j}} - \bar {\mathbf {x}} \| + \| G _ {d} (\mathbf {x} ^ {k _ {j}}) \|,\tag{10.19}
$$

where Lemma 10.10(a) was used in the second inequality. Since the right-hand side of (10.19) goes to 0 as $j \to \infty$ , it follows that $G _ { d } ( \bar { \bf x } ) = { \bf 0 }$ , which by Theorem 10.7(b) implies that x¯ is a stationary point of problem (10.1). 

## 10.4 Analysis of the Proximal Gradient Method— The Convex Case

## 10.4.1 The Fundamental Prox-Grad Inequality

The analysis of the proximal gradient method in the case where f is convex is based on the following key inequality (which actually does not assume that f is convex). 

Theorem 10.16 (fundamental prox-grad inequality). Suppose that f and g satisfy properties (A) and (B) of Assumption 10.1. For any $\mathbf { x } \in \mathbb { E } , \mathbf { y } \in \mathrm { i n t } ( \mathrm { d o m } ( f ) )$ and $L > 0$ satisfying 

$$
f (T _ {L} (\mathbf {y})) \leq f (\mathbf {y}) + \langle \nabla f (\mathbf {y}), T _ {L} (\mathbf {y}) - \mathbf {y} \rangle + \frac {L}{2} \| T _ {L} (\mathbf {y}) - \mathbf {y} \| ^ {2},\tag{10.20}
$$

it holds that 

$$
F (\mathbf {x}) - F (T _ {L} (\mathbf {y})) \geq \frac {L}{2} \| \mathbf {x} - T _ {L} (\mathbf {y}) \| ^ {2} - \frac {L}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2} + \ell_ {f} (\mathbf {x}, \mathbf {y}),\tag{10.21}
$$

where 

$$
\ell_ {f} (\mathbf {x}, \mathbf {y}) = f (\mathbf {x}) - f (\mathbf {y}) - \langle \nabla f (\mathbf {y}), \mathbf {x} - \mathbf {y} \rangle .
$$

Proof. Consider the function 

$$
\varphi (\mathbf {u}) = f (\mathbf {y}) + \langle \nabla f (\mathbf {y}), \mathbf {u} - \mathbf {y} \rangle + g (\mathbf {u}) + \frac {L}{2} \| \mathbf {u} - \mathbf {y} \| ^ {2}.
$$

Since $\varphi$ is an L-strongly convex function and $T _ { L } ( \mathbf { y } ) = \mathrm { a r g m i n } _ { \mathbf { u } \in \mathbb { E } } \varphi ( \mathbf { u } )$ , it follows by Theorem 5.25(b) that 

$$
\varphi (\mathbf {x}) - \varphi (T _ {L} (\mathbf {y})) \geq \frac {L}{2} \| \mathbf {x} - T _ {L} (\mathbf {y}) \| ^ {2}.\tag{10.22}
$$

Note that by (10.20), 

$$
\begin{array}{l} \varphi (T _ {L} (\mathbf {y})) = f (\mathbf {y}) + \langle \nabla f (\mathbf {y}), T _ {L} (\mathbf {y}) - \mathbf {y} \rangle + \frac {L}{2} \| T _ {L} (\mathbf {y}) - \mathbf {y} \| ^ {2} + g (T _ {L} (\mathbf {y})) \\ \qquad \qquad \qquad \geq f (T _ {L} (\mathbf {y})) + g (T _ {L} (\mathbf {y})) = F (T _ {L} (\mathbf {y})), \end{array}
$$

and thus (10.22) implies that for any $\mathbf { x } \in \mathbb { E }$ 2 

$$
\varphi (\mathbf {x}) - F (T _ {L} (\mathbf {y})) \geq \frac {L}{2} \| \mathbf {x} - T _ {L} (\mathbf {y}) \| ^ {2}.
$$

Plugging the expression for $\varphi ( \mathbf { x } )$ into the above inequality, we obtain 

$$
f (\mathbf {y}) + \langle \nabla f (\mathbf {y}), \mathbf {x} - \mathbf {y} \rangle + g (\mathbf {x}) + \frac {L}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2} - F (T _ {L} (\mathbf {y})) \geq \frac {L}{2} \| \mathbf {x} - T _ {L} (\mathbf {y}) \| ^ {2},
$$

which is the same as the desired result: 

$$
\begin{array}{r l} & F (\mathbf {x}) - F (T _ {L} (\mathbf {y})) \geq \frac {L}{2} \| \mathbf {x} - T _ {L} (\mathbf {y}) \| ^ {2} - \frac {L}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2} \\ & \qquad + f (\mathbf {x}) - f (\mathbf {y}) - \langle \nabla f (\mathbf {y}), \mathbf {x} - \mathbf {y} \rangle . \end{array}
$$

Remark 10.17. Obviously, by the descent lemma, (10.20) is satisfied for $L = L _ { f }$ 2 and hence, for any $\mathbf { x } \in \mathbb { E }$ and $\mathbf { y } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ , the inequality 

$$
F (\mathbf {x}) - F (T _ {L _ {f}} (\mathbf {y})) \geq \frac {L _ {f}}{2} \| \mathbf {x} - T _ {L _ {f}} (\mathbf {y}) \| ^ {2} - \frac {L _ {f}}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2} + \ell_ {f} (\mathbf {x}, \mathbf {y})
$$

holds. 

A direct consequence of Theorem 10.16 is another version of the suficient decrease lemma (Lemma 10.4). This is accomplished by substituting $\mathbf { y } = \mathbf { x }$ in the fundamental prox-grad inequality. 

Corollary 10.18 (suficient decrease lemma—second version). Suppose that f and g satisfy properties (A) and (B) of Assumption 10.1. For any $\mathbf { x \in }$ int(dom(f)) for which 

$$
f (T _ {L} (\mathbf {x})) \leq f (\mathbf {x}) + \langle \nabla f (\mathbf {x}), T _ {L} (\mathbf {x}) - \mathbf {x} \rangle + \frac {L}{2} \| T _ {L} (\mathbf {x}) - \mathbf {x} \| ^ {2},
$$

it holds that 

$$
F (\mathbf {x}) - F (T _ {L} (\mathbf {x})) \geq \frac {1}{2 L} \| G _ {L} (\mathbf {x}) \| ^ {2}.
$$

## 10.4.2 Stepsize Strategies in the Convex Case

When f is also convex, we will consider, as in the nonconvex case, both constant and backtracking stepsize strategies. The backtracking procedure, which we will refer to as “backtracking procedure B2,” will be slightly diferent than the one considered in the nonconvex case, and it will aim to find a constant $L _ { k }$ satisfying 

$$
f (\mathbf {x} ^ {k + 1}) \leq f (\mathbf {x} ^ {k}) + \langle \nabla f (\mathbf {x} ^ {k}), \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \rangle + \frac {L _ {k}}{2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \| ^ {2}.\tag{10.23}
$$

In the special case where $g \equiv 0$ , the proximal gradient method reduces to the gradient method $\begin{array} { r } { \mathbf { x } ^ { k + 1 } = \mathbf { x } ^ { k } - \frac { 1 } { L _ { k } } \nabla f ( \mathbf { x } ^ { k } ) } \end{array}$ , and condition (10.23) reduces to 

$$
f (\mathbf {x} ^ {k}) - f (\mathbf {x} ^ {k + 1}) \geq \frac {1}{2 L _ {k}} \| \nabla f (\mathbf {x} ^ {k}) \| ^ {2},
$$

which is similar to the suficient decrease condition described in Lemma 10.4, and this is why condition (10.23) can also be viewed as a “suficient decrease condition.” 

• Constant. $L _ { k } = L _ { f }$ for all k. 

• Backtracking procedure B2. The procedure requires two parameters $( s , \eta )$ , where $s > 0$ and $\eta > 1$ . Define $L _ { - 1 } = s$ . At iteration k $( k \geq 0 )$ the choice of $L _ { k }$ is done as follows. First, $L _ { k }$ is set to be equal to $L _ { k - 1 }$ Then, while 

$$
f (T _ {L _ {k}} (\mathbf {x} ^ {k})) > f (\mathbf {x} ^ {k}) + \langle \nabla f (\mathbf {x} ^ {k}), T _ {L _ {k}} (\mathbf {x} ^ {k}) - \mathbf {x} ^ {k} \rangle + \frac {L _ {k}}{2} \| T _ {L _ {k}} (\mathbf {x} ^ {k}) - \mathbf {x} ^ {k} \| ^ {2},
$$

we set $L _ { k } : = \eta L _ { k }$ . In other words, $L _ { k }$ is chosen as $L _ { k } = L _ { k - 1 } \eta ^ { i _ { k } }$ , where $i _ { k }$ is the smallest nonnegative integer for which the condition 

$$
\begin{array}{c} f (T _ {L _ {k - 1} \eta^ {i _ {k}}} (\mathbf {x} ^ {k})) \leq f (\mathbf {x} ^ {k}) + \langle \nabla f (\mathbf {x} ^ {k}), T _ {L _ {k - 1} \eta^ {i _ {k}}} (\mathbf {x} ^ {k}) - \mathbf {x} ^ {k} \rangle + \\ \frac {L _ {k}}{2} \| T _ {L _ {k - 1} \eta^ {i _ {k}}} (\mathbf {x} ^ {k}) - \mathbf {x} ^ {k} \| ^ {2} \end{array}
$$

is satisfied. 

Remark 10.19 (upper and lower bounds on $L _ { k } )$ . Under Assumption 10.1 and by the descent lemma (Lemma 5.7), it follows that both stepsize rules ensure that the suficient decrease condition (10.23) is satisfied at each iteration. In addition, the constants $L _ { k }$ that the backtracking procedure B2 produces satisfy the following bounds for all $k \geq 0$ 

$$
s \leq L _ {k} \leq \max \{\eta L _ {f}, s \}.\tag{10.24}
$$

The inequality $s \leq L _ { k }$ is obvious. To understand the inequality $L _ { k } \le \operatorname* { m a x } \{ \eta L _ { f } , s \}$ note that there are two options. Either $L _ { k } = s \ o r \ L _ { k } > s$ , and in the latter case there exists an index $0 \leq k ^ { \prime } \leq k$ for which the inequality (10.23) is not satisfied with $k = k ^ { \prime }$ and $\frac { L _ { k } } { \eta }$ replacing $L _ { k }$ . By the descent lemma, this implies in particular that $\begin{array} { r } { \frac { L _ { k } } { \eta } < L _ { f } } \end{array}$ , and we have thus shown that $L _ { k } \le \operatorname* { m a x } \{ \eta L _ { f } , s \}$ . We also note that the bounds on $L _ { k }$ can be rewritten as 

$$
\beta L _ {f} \leq L _ {k} \leq \alpha L _ {f},
$$

where 

$$
\alpha = \left\{ \begin{array}{l l} 1, & \text {constant}, \\ \max \left\{\eta , \frac {s}{L _ {f}} \right\}, & \text {backtracking}, \end{array} \right. \beta = \left\{ \begin{array}{l l} 1, & \text {constant}, \\ \frac {s}{L _ {f}}, & \text {backtracking}. \end{array} \right.\tag{10.25}
$$

Remark 10.20 (monotonicity of the proximal gradient method). Since condition (10.23) holds for both stepsize rules, for any $k \geq 0$ , we can invoke the fundamental prox-grad inequality (10.21) with $\mathbf { y } = \mathbf { x } = \mathbf { x } ^ { k } , L = L _ { k }$ and obtain the inequality 

$$
F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \geq \frac {L _ {k}}{2} \| \mathbf {x} ^ {k} - \mathbf {x} ^ {k + 1} \| ^ {2},
$$

which in particular implies that $F ( \mathbf { x } ^ { k } ) \geq F ( \mathbf { x } ^ { k + 1 } )$ , meaning that the method produces a nonincreasing sequence of function values. 

## 10.4.3 Convergence Analysis in the Convex Case

We will assume in addition to Assumption 10.1 that f is convex. We begin by establishing an $O ( 1 / k )$ rate of convergence of the generated sequence of function values to the optimal value. Such rate of convergence is called a sublinear rate. This is of course an improvement over the $O ( 1 / { \sqrt { k } } )$ rate that was established for the projected subgradient and mirror descent methods. It is also not particularly surprising that an improved rate of convergence can be established since additional properties are assumed on the objective function. 

Theorem 10.21 $( O ( 1 / k )$ rate of convergence of proximal gradient). Suppose that Assumption 10.1 holds and that in addition f is convex. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the proximal gradient method for solving problem (10.1) with either a constant stepsize rule in which $L _ { k } \equiv L _ { f }$ for all $k \geq 0$ or the backtracking procedure B2. Then for any $\mathbf { x } ^ { * } \in X ^ { * }$ and $k \geq 0$ 

$$
F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} \leq \frac {\alpha L _ {f} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}}{2 k},\tag{10.26}
$$

where $\alpha = 1$ in the constant stepsize setting and $\begin{array} { r } { \alpha = \operatorname* { m a x } \left\{ \eta , \frac { s } { L _ { f } } \right\} } \end{array}$ if the backtracking rule is employed. 

Proof. For any $n \geq 0$ , substituting $L = L _ { n } , \mathbf { x } = \mathbf { x } ^ { * }$ , and $\mathbf { y } = \mathbf { x } ^ { n }$ in the fundamental prox-grad inequality (10.21) and taking into account the fact that in both stepsize rules condition (10.20) is satisfied, we obtain 

$$
\begin{array}{r l} & {\frac {2}{L _ {n}} (F (\mathbf {x} ^ {*}) - F (\mathbf {x} ^ {n + 1})) \geq \| \mathbf {x} ^ {*} - \mathbf {x} ^ {n + 1} \| ^ {2} - \| \mathbf {x} ^ {*} - \mathbf {x} ^ {n} \| ^ {2} + \frac {2}{L _ {n}} \ell_ {f} (\mathbf {x} ^ {*}, \mathbf {x} ^ {n})} \\ & {\qquad \geq \| \mathbf {x} ^ {*} - \mathbf {x} ^ {n + 1} \| ^ {2} - \| \mathbf {x} ^ {*} - \mathbf {x} ^ {n} \| ^ {2},} \end{array}
$$

where the convexity of $f$ was used in the last inequality. Summing the above inequality over $n = 0 , 1 , \ldots , k - 1$ and using the bound $L _ { n } \leq \alpha L _ { f }$ for all $n \geq 0$ (see Remark 10.19), we obtain 

$$
\frac {2}{\alpha L _ {f}} \sum_ {n = 0} ^ {k - 1} (F (\mathbf {x} ^ {*}) - F (\mathbf {x} ^ {n + 1})) \geq \| \mathbf {x} ^ {*} - \mathbf {x} ^ {k} \| ^ {2} - \| \mathbf {x} ^ {*} - \mathbf {x} ^ {0} \| ^ {2}.
$$

Thus, 

$$
\sum_ {n = 0} ^ {k - 1} (F (\mathbf {x} ^ {n + 1}) - F _ {\mathrm{opt}}) \leq \frac {\alpha L _ {f}}{2} \| \mathbf {x} ^ {*} - \mathbf {x} ^ {0} \| ^ {2} - \frac {\alpha L _ {f}}{2} \| \mathbf {x} ^ {*} - \mathbf {x} ^ {k} \| ^ {2} \leq \frac {\alpha L _ {f}}{2} \| \mathbf {x} ^ {*} - \mathbf {x} ^ {0} \| ^ {2}.
$$

By the monotonicity of $\{ F ( { \bf x } ^ { n } ) \} _ { n \ge 0 }$ (see Remark 10.20), we can conclude that 

$$
k (F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}}) \leq \sum_ {n = 0} ^ {k - 1} (F (\mathbf {x} ^ {n + 1}) - F _ {\mathrm{opt}}) \leq \frac {\alpha L _ {f}}{2} \| \mathbf {x} ^ {*} - \mathbf {x} ^ {0} \| ^ {2}.
$$

Consequently, 

$$
F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} \leq \frac {\alpha L _ {f} \| \mathbf {x} ^ {*} - \mathbf {x} ^ {0} \| ^ {2}}{2 k}. \quad \square
$$

Remark 10.22. Note that we did not utilize in the proof of Theorem 10.21 the fact that procedure B2 produces a nondecreasing sequence of constants $\{ L _ { k } \} _ { k \ge 0 }$ This implies in particular that the monotonicity of this sequence of constants is not essential, and we can actually prove the same convergence rate for any backtracking procedure that guarantees the validity of condition (10.23) and the bound $L _ { k } \le \alpha L _ { f }$ 

We can also prove that the generated sequence is Fej´er monotone, from which convergence of the sequence to an optimal solution readily follows. 

Theorem 10.23 (Fej´er monotonicity of the sequence generated by the proximal gradient method). Suppose that Assumption 10.1 holds and that in addition $f$ is convex. Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the proximal gradient method for solving problem (10.1) with either a constant stepsize rule in which $L _ { k } \equiv$ $L _ { f }$ for all $k \geq 0$ or the backtracking procedure B2. Then for any $\mathbf { x } ^ { * } \in X ^ { * }$ and $k \geq 0$ 2 

$$
\left\| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \right\| \leq \left\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \right\|.\tag{10.27}
$$

Proof. We will repeat some of the arguments used in the proof of Theorem 10.21. Substituting $L = L _ { k } , \mathbf { x } = \mathbf { x } ^ { * }$ , and $\mathbf { y } = \mathbf { x } ^ { k }$ in the fundamental prox-grad inequality (10.21) and taking into account the fact that in both stepsize rules condition (10.20) is satisfied, we obtain 

$$
\begin{array}{r l} & {\frac {2}{L _ {k}} (F (\mathbf {x} ^ {*}) - F (\mathbf {x} ^ {k + 1})) \geq \| \mathbf {x} ^ {*} - \mathbf {x} ^ {k + 1} \| ^ {2} - \| \mathbf {x} ^ {*} - \mathbf {x} ^ {k} \| ^ {2} + \frac {2}{L _ {k}} \ell_ {f} (\mathbf {x} ^ {*}, \mathbf {x} ^ {k})} \\ & {\qquad \geq \| \mathbf {x} ^ {*} - \mathbf {x} ^ {k + 1} \| ^ {2} - \| \mathbf {x} ^ {*} - \mathbf {x} ^ {k} \| ^ {2},} \end{array}
$$

where the convexity of f was used in the last inequality. The result (10.27) now follows by the inequality $F ( \mathbf { x } ^ { * } ) - F ( \mathbf { x } ^ { k + 1 } ) \leq 0$ 

Thanks to the Fej´er monotonicity property, we can now establish the convergence of the sequence generated by the proximal gradient method. 

Theorem 10.24 (convergence of the sequence generated by the proximal gradient method). Suppose that Assumption 10.1 holds and that in addition $f$ is convex. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the proximal gradient method for solving problem (10.1) with either a constant stepsize rule in which $L _ { k } \equiv L _ { f }$ for all $k \geq 0$ or the backtracking procedure B2. Then the sequence $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ converges to an optimal solution of problem (10.1). 

Proof. By Theorem 10.23, the sequence is Fej´er monotone w.r.t. $X ^ { * }$ . Therefore, by Theorem 8.16, to show convergence to a point in $X ^ { * }$ , it is enough to show that any limit point of the sequence $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ is necessarily in $X ^ { * }$ . Let then $\tilde { \bf x }$ be a limit point of the sequence. Then there exists a subsequence $\{ \mathbf { x } ^ { k _ { j } } \} _ { j \geq 0 }$ converging to x˜. By Theorem 10.21, 

$$
F (\mathbf {x} ^ {k _ {j}}) \to F _ {\mathrm{opt}} \mathrm{as} j \to \infty .\tag{10.28}
$$

Since $F$ is closed, it is also lower semicontinuous, and hence $F ( \tilde { \mathbf { x } } ) \leq \operatorname* { l i m } _ { j  \infty } F ( \mathbf { x } ^ { k _ { j } } )$ $= F _ { \mathrm { o p t } }$ , implying that $\tilde { \mathbf { x } } \in X ^ { * }$ □ 

To derive a complexity result for the proximal gradient method, we will assume that $\| \mathbf { x } ^ { 0 } - \mathbf { x } ^ { * } \| \leq R$ for some $\mathbf { x } ^ { * } \in X ^ { * }$ and some constant $R > 0$ ; for example, if $\operatorname { d o m } ( g )$ is bounded, then R might be taken as its diameter. By inequality (10.26) it follows that in order to obtain an ε-optimal solution of problem (10.1), it is enough to require that 

$$
\frac {\alpha L _ {f} R ^ {2}}{2 k} \leq \varepsilon ,
$$

which is the same as 

$$
k \geq \frac {\alpha L _ {f} R ^ {2}}{2 \varepsilon}.
$$

Thus, to obtain an ε-optimal solution, an order of $\frac { 1 } { \varepsilon }$ iterations is required, which is an improvement of the result for the projected subgradient method in which an order of $\textstyle { \frac { 1 } { \varepsilon ^ { 2 } } }$ iterations is needed (see, for example, Theorem 8.18). We summarize the above observations in the following theorem. 

Theorem 10.25 (complexity of the proximal gradient method). Under the setting of Theorem 10.21, for any k satisfying 

$$
k \geq \left\lceil \frac {\alpha L _ {f} R ^ {2}}{2 \varepsilon} \right\rceil ,
$$

it holds that $F ( \mathbf { x } ^ { k } ) - F _ { \mathrm { o p t } } \leq \varepsilon _ { \mathrm { . } }$ , where R is an upper bound on $\| \mathbf { x } ^ { * } - \mathbf { x } ^ { 0 } \|$ for some $\mathbf { x } ^ { * } \in X ^ { * }$ 

In the nonconvex case (meaning when f is not necessarily convex), an $O ( 1 / \sqrt { k } )$ rate of convergence of the norm of the gradient mapping was established in Theorem 10.15(c). We will now show that with the additional convexity assumption on $f ,$ this rate can be improved to $O ( 1 / k )$ 

Theorem 10.26 $( O ( 1 / k )$ rate of convergence of the minimal norm of the gradient mapping). Suppose that Assumption 10.1 holds and that in addition $f$ is convex. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the proximal gradient method for solving problem (10.1) with either a constant stepsize rule in which $L _ { k } \equiv L _ { f }$ for all $k \geq 0$ or the backtracking procedure B2. Then for any $\mathbf { x } ^ { * } \in X ^ { * }$ and $k \geq 1$ 

$$
\min _ {n = 0, 1, \dots , k} \| G _ {\alpha L _ {f}} (\mathbf {x} ^ {n}) \| \leq \frac {2 \alpha^ {1 . 5} L _ {f} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \|}{\sqrt {\beta} k},\tag{10.29}
$$

where $\alpha = \beta = 1$ in the constant stepsize setting and $\begin{array} { r } { \alpha = \operatorname* { m a x } \big \{ \eta , \frac { s } { L _ { f } } \big \} , \beta = \frac { s } { L _ { f } } } \end{array}$ if the backtracking rule is employed. 

Proof. By the suficient decrease lemma (Corollary 10.18), for any $n \geq 0$ 2 

$$
F (\mathbf {x} ^ {n}) - F (\mathbf {x} ^ {n + 1}) = F (\mathbf {x} ^ {n}) - F (T _ {L _ {n}} (\mathbf {x} ^ {n})) \geq \frac {1}{2 L _ {n}} \| G _ {L _ {n}} (\mathbf {x} ^ {n}) \| ^ {2}.\tag{10.30}
$$

By Theorem 10.9 and the fact that $\beta L _ { f } \leq L _ { n } \leq \alpha L _ { f }$ (see Remark 10.19), it follows that 

$$
\frac {1}{2 L _ {n}} \| G _ {L _ {n}} (\mathbf {x} ^ {n}) \| ^ {2} = \frac {L _ {n}}{2} \frac {\| G _ {L _ {n}} (\mathbf {x} ^ {n}) \| ^ {2}}{L _ {n} ^ {2}} \geq \frac {\beta L _ {f}}{2} \frac {\| G _ {\alpha L _ {f}} (\mathbf {x} ^ {n}) \| ^ {2}}{\alpha^ {2} L _ {f} ^ {2}} = \frac {\beta}{2 \alpha^ {2} L _ {f}} \| G _ {\alpha L _ {f}} (\mathbf {x} ^ {n}) \| ^ {2}.\tag{10.31}
$$

Therefore, combining (10.30) and (10.31), 

$$
F (\mathbf {x} ^ {n}) - F _ {\mathrm{opt}} \geq F (\mathbf {x} ^ {n + 1}) - F _ {\mathrm{opt}} + \frac {\beta}{2 \alpha^ {2} L _ {f}} \| G _ {\alpha L _ {f}} (\mathbf {x} ^ {n}) \| ^ {2}.\tag{10.32}
$$

Let $p$ be a positive integer. Summing (10.32) over $n = p , p + 1 , \dotsc , 2 p - 1$ yields 

$$
F (\mathbf {x} ^ {p}) - F _ {\mathrm{opt}} \geq F (\mathbf {x} ^ {2 p}) - F _ {\mathrm{opt}} + \frac {\beta}{2 \alpha^ {2} L _ {f}} \sum_ {n = p} ^ {2 p - 1} \| G _ {\alpha L _ {f}} (\mathbf {x} ^ {n}) \| ^ {2}.\tag{10.33}
$$

By Theorem 10.21, $\begin{array} { r } { F ( \mathbf { x } ^ { p } ) - F _ { \mathrm { o p t } } \leq \frac { \alpha L _ { f } \| \mathbf { x } ^ { 0 } - \mathbf { x } ^ { * } \| ^ { 2 } } { 2 p } } \end{array}$ , which, combined with the fact that $F ( \mathbf { x } ^ { 2 p } ) - F _ { \mathrm { o p t } } \geq 0$ and (10.33), implies 

$$
\frac {\beta p}{2 \alpha^ {2} L _ {f}} \min _ {n = 0, 1, \dots , 2 p - 1} \| G _ {\alpha L _ {f}} (\mathbf {x} ^ {n}) \| ^ {2} \leq \frac {\beta}{2 \alpha^ {2} L _ {f}} \sum_ {n = p} ^ {2 p - 1} \| G _ {\alpha L _ {f}} (\mathbf {x} ^ {n}) \| ^ {2} \leq \frac {\alpha L _ {f} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}}{2 p}.
$$

Thus, 

$$
\min _ {n = 0, 1, \dots , 2 p - 1} \| G _ {\alpha L _ {f}} (\mathbf {x} ^ {n}) \| ^ {2} \leq \frac {\alpha^ {3} L _ {f} ^ {2} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}}{\beta p ^ {2}}\tag{10.34}
$$

and also 

$$
\min _ {n = 0, 1, \dots , 2 p} \| G _ {\alpha L _ {f}} (\mathbf {x} ^ {n}) \| ^ {2} \leq \frac {\alpha^ {3} L _ {f} ^ {2} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}}{\beta p ^ {2}}.\tag{10.35}
$$

We conclude that for any $k \geq 1$ 2 

$$
\min _ {n = 0, 1, \dots , k} \| G _ {\alpha L _ {f}} (\mathbf {x} ^ {n}) \| ^ {2} \leq \frac {\alpha^ {3} L _ {f} ^ {2} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}}{\beta \min \{(k / 2) ^ {2} , ((k + 1) / 2) ^ {2} \}} = \frac {4 \alpha^ {3} L _ {f} ^ {2} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}}{\beta k ^ {2}}.
$$

When we assume further that $f$ is $L _ { f }$ -smooth over the entire space $\mathbb { E } .$ we can use Lemma 10.12 to obtain an improved result in the case of a constant stepsize. 

Theorem 10.27 $( O ( 1 / k )$ rate of convergence of the norm of the gradient mapping under the constant stepsize rule). Suppose that Assumption 10.1 holds and that in addition f is convex and $L _ { f } { - } s m o o t h$ over E. Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the proximal gradient method for solving problem (10.1) with a constant stepsize rule in which $L _ { k } \equiv L _ { f }$ for all $k \geq 0$ . Then for any $\mathbf { x } ^ { * } \in X ^ { * }$ and $k \geq 0$ 2 

$$
\left(\mathrm{a}\right) \left\| G _ {L _ {f}} (\mathbf {x} ^ {k + 1}) \right\| \leq \left\| G _ {L _ {f}} (\mathbf {x} ^ {k}) \right\|;
$$

$$
\left(\mathrm{b}\right) \left\| G _ {L _ {f}} (\mathbf {x} ^ {k}) \right\| \leq \frac {2 L _ {f} \left\| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \right\|}{k + 1}.
$$

Proof. Invoking Lemma 10.12 with $\mathbf { x } = \mathbf { x } ^ { k }$ , we obtain (a). Part (b) now followsby substituting $\alpha = \beta = 1$ in the result of Theorem 10.26 and noting that by part$\begin{array} { r } { \mathbf { \rho } ( \mathbf { a } ) , \| G _ { L _ { f } } ( \mathbf { x } ^ { k } ) \| = \operatorname* { m i n } _ { n = 0 , 1 , \dots , k } \| G _ { L _ { f } } ( \mathbf { x } ^ { n } ) \| } \end{array}$ 口

## 10.5 The Proximal Point Method

Consider the problem 

$$
\min _ {\mathbf {x} \in \mathbb {E}} g (\mathbf {x}),\tag{10.36}
$$

where $g : \mathbb { E }  ( - \infty , \infty ]$ is a proper closed and convex function. Problem (10.36) is actually a special case of the composite problem (10.1) with $f \equiv 0$ . The update step of the proximal gradient method in this case takes the form 

$$
\mathbf {x} ^ {k + 1} = \mathrm{prox} _ {\frac {1}{L _ {k}} g} (\mathbf {x} ^ {k}).
$$

Taking $\begin{array} { r } { L _ { k } = \frac { 1 } { c } } \end{array}$ for some $c > 0$ , we obtain the proximal point method. 

The Proximal Point Method
Initialization: pick $x^{0} \in E$ and c > 0.
General step ( $k \geq 0$ ): $\mathbf{x}^{k+1} = \operatorname{prox}_{cg}(\mathbf{x}^{k})$ . 

The proximal point method is actually not a practical algorithm since the general step asks to minimize the function $\begin{array} { r } { g ( \mathbf { x } ) + \frac { c } { 2 } \| \mathbf { x } - \mathbf { x } ^ { k } \| ^ { 2 } } \end{array}$ , which in general is as hard to accomplish as solving the original problem of minimizing $g .$ Since the proximal point method is a special case of the proximal gradient method, we can deduce its main convergence results from the corresponding results on the proximal gradient method. Specifically, since the smooth part $f \equiv 0$ is 0-smooth, we can take any constant stepsize to guarantee convergence and Theorems 10.21 and 10.24 imply the following result. 

Theorem 10.28 (convergence of the proximal point method). Let $g : \mathbb { E } $ $( - \infty , \infty ]$ be a proper closed and convex function. Assume that problem 

$$
\min _ {\mathbf {x} \in \mathbb {E}} g (\mathbf {x})
$$

has a nonempty optimal set $X ^ { \ast }$ , and let the optimal value be given by $g _ { \mathrm { o p t } }$ . Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the proximal point method with parameter $c > 0$ . Then 

(a) $\begin{array} { r } { g ( \mathbf { x } ^ { k } ) - g _ { \mathrm { o p t } } \leq \frac { \| \mathbf { x } ^ { 0 } - \mathbf { x } ^ { * } \| ^ { 2 } } { 2 c k } } \end{array}$ for any $\mathbf { x } ^ { * } \in X ^ { * }$ and $k \geq 0$ ; 

(b) the sequence $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ converges to some point in $X ^ { * }$ 

## 10.6 Convergence of the Proximal Gradient Method—The Strongly Convex Case

In the case where $f$ is assumed to be σ-strongly convex for some $\sigma > 0$ , the sublinear rate of convergence can be improved into a linear rate of convergence, meaning a rate of the form $O ( q ^ { k } )$ for some $q \in ( 0 , 1 )$ . Throughout the analysis of the strongly convex case we denote the unique optimal solution of problem (10.1) by $\mathbf { x } ^ { * }$ 

Theorem 10.29 (linear rate of convergence of the proximal gradient method—strongly convex case). Suppose that Assumption 10.1 holds and that in addition f is σ-strongly convex $( \sigma > 0 )$ . Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the proximal gradient method for solving problem (10.1) with either a constant stepsize rule in which $L _ { k } \equiv L _ { f }$ for all $k \geq 0$ or the backtracking procedure B2. Let 

$$
\alpha = \left\{ \begin{array}{l l} 1, & \text {constant stepsize}, \\ \max \left\{\eta , \frac {s}{L _ {f}} \right\}, & \text {backtracking}. \end{array} \right.
$$

Then for any $k \geq 0$ 

$$
\left(\mathrm{a}\right) \left\| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \right\| ^ {2} \leq \left(1 - \frac {\sigma}{\alpha L _ {f}}\right) \left\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \right\| ^ {2};
$$

$$
\left(\mathrm{b}\right) \left\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \right\| ^ {2} \leq \left(1 - \frac {\sigma}{\alpha L _ {f}}\right) ^ {k} \left\| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \right\| ^ {2};
$$

$$
\text {(c)} F (\mathbf {x} ^ {k + 1}) - F _ {\mathrm{opt}} \leq \frac {\alpha L _ {f}}{2} \left(1 - \frac {\sigma}{\alpha L _ {f}}\right) ^ {k + 1} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}.
$$

Proof. Plugging $L = L _ { k } , \mathbf { x } = \mathbf { x } ^ { * }$ , and $\mathbf { y } = \mathbf { x } ^ { k }$ into the fundamental prox-grad inequality (10.21) and taking into account the fact that in both stepsize rules condition (10.20) is satisfied, we obtain 

$$
F (\mathbf {x} ^ {*}) - F (\mathbf {x} ^ {k + 1}) \geq \frac {L _ {k}}{2} \| \mathbf {x} ^ {*} - \mathbf {x} ^ {k + 1} \| ^ {2} - \frac {L _ {k}}{2} \| \mathbf {x} ^ {*} - \mathbf {x} ^ {k} \| ^ {2} + \ell_ {f} (\mathbf {x} ^ {*}, \mathbf {x} ^ {k}).
$$

Since f is σ-strongly convex, it follows by Theorem 5.24(ii) that 

$$
\ell_ {f} \left(\mathbf {x} ^ {*}, \mathbf {x} ^ {k}\right) = f \left(\mathbf {x} ^ {*}\right) - f \left(\mathbf {x} ^ {k}\right) - \left\langle \nabla f \left(\mathbf {x} ^ {k}\right), \mathbf {x} ^ {*} - \mathbf {x} ^ {k} \right\rangle \geq \frac {\sigma}{2} \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2}.
$$

Thus, 

$$
F (\mathbf {x} ^ {*}) - F (\mathbf {x} ^ {k + 1}) \geq \frac {L _ {k}}{2} \| \mathbf {x} ^ {*} - \mathbf {x} ^ {k + 1} \| ^ {2} - \frac {L _ {k} - \sigma}{2} \| \mathbf {x} ^ {*} - \mathbf {x} ^ {k} \| ^ {2}.\tag{10.37}
$$

Since $\mathbf { x } ^ { * }$ is a minimizer of $F , F ( \mathbf { x } ^ { * } ) - F ( \mathbf { x } ^ { k + 1 } ) \leq 0$ , and hence, by (10.37) and the fact that $L _ { k } \le \alpha L _ { f }$ (see Remark 10.19), 

$$
\left\| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \right\| ^ {2} \leq \left(1 - \frac {\sigma}{L _ {k}}\right) \left\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \right\| ^ {2} \leq \left(1 - \frac {\sigma}{\alpha L _ {f}}\right) \left\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \right\| ^ {2},
$$

establishing part (a). Part (b) follows immediately by (a). To prove (c), note that by (10.37), 

$$
\begin{array}{l} F (\mathbf {x} ^ {k + 1}) - F _ {\mathrm{opt}} \leq \frac {L _ {k} - \sigma}{2} \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} - \frac {L _ {k}}{2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2} \\ \qquad \leq \frac {\alpha L _ {f} - \sigma}{2} \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} \\ \qquad = \frac {\alpha L _ {f}}{2} \left(1 - \frac {\sigma}{\alpha L _ {f}}\right) \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} \\ \qquad \leq \frac {\alpha L _ {f}}{2} \left(1 - \frac {\sigma}{\alpha L _ {f}}\right) ^ {k + 1} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}, \end{array}
$$

where part (b) was used in the last inequality. 

Theorem 10.29 immediately implies that in the strongly convex case, the proximal gradient method requires an order of $\log ( \frac { 1 } { \varepsilon } )$ iterations to obtain an ε-optimal solution. 

Theorem 10.30 (complexity of the proximal gradient method—The strongly convex case). Under the setting of Theorem 10.29, for any $k \geq 1$ satisfying 

$$
k \geq \alpha \kappa \log \left(\frac {1}{\varepsilon}\right) + \alpha \kappa \log \left(\frac {\alpha L _ {f} R ^ {2}}{2}\right),
$$

it holds that $F ( \mathbf { x } ^ { k } ) - F _ { \mathrm { o p t } } \leq \varepsilon$ , where $R$ is an upper bound on $\lVert \mathbf { x } ^ { 0 } - \mathbf { x } ^ { * } \rVert$ and $\begin{array} { r } { \kappa = \frac { L _ { f } } { \sigma } } \end{array}$ 

Proof. Let $k \geq 1$ . By Theorem 10.29 and the definition of $\kappa ,$ a suficient condition for the inequality $F ( \mathbf { x } ^ { k } ) - F _ { \mathrm { o p t } } \leq \varepsilon$ to hold is that 

$$
\frac {\alpha L _ {f}}{2} \left(1 - \frac {1}{\alpha \kappa}\right) ^ {k} R ^ {2} \leq \varepsilon ,
$$

which is the same as 

$$
k \log \left(1 - \frac {1}{\alpha \kappa}\right) \leq \log \left(\frac {2 \varepsilon}{\alpha L _ {f} R ^ {2}}\right).\tag{10.38}
$$

Since log $( 1 - x ) \leq - x$ for $\mathrm { a n y } ^ { 5 7 } \ x \ \leq \ 1$ , it follows that a suficient condition for (10.38) to hold is that 

$$
- \frac {1}{\alpha \kappa} k \leq \log \left(\frac {2 \varepsilon}{\alpha L _ {f} R ^ {2}}\right),
$$

namely, that 

$$
k \geq \alpha \kappa \log \left(\frac {1}{\varepsilon}\right) + \alpha \kappa \log \left(\frac {\alpha L _ {f} R ^ {2}}{2}\right). \quad \square
$$

## 10.7 The Fast Proximal Gradient Method—FISTA

## 10.7.1 The Method

The proximal gradient method achieves an $O ( 1 / k )$ rate of convergence in function values to the optimal value. In this section we will show how to accelerate the method in order to obtain a rate of $O ( 1 / k ^ { 2 } )$ in function values. The method is known as the “fast proximal gradient method,” but we will also refer to it as “FISTA,” which is an acronym for “fast iterative shrinkage-thresholding algorithm”; see Example 10.37 for further explanations. The method was devised and analyzed by Beck and Teboulle in the paper [18], from which the convergence analysis is taken. 

We will assume that $f$ is convex and that it is $L _ { f } { \mathrm { - s m o o t h } }$ , meaning that it is $L _ { f } .$ -smooth over the entire space E. We gather all the required properties in the following assumption. 

## Assumption 10.31.

(A) $g : \mathbb { E }  ( - \infty , \infty ]$ is proper closed and convex. 

(B) $f : \mathbb { E } \to \mathbb { R }$ is $L _ { f }$ -smooth and convex. 

(C) The optimal set of problem (10.1) is nonempty and denoted by $X ^ { * }$ . The optimal value of the problem is denoted by $F _ { \mathrm { o p t } }$ 

The description of FISTA now follows. 

## FISTA

Input: $( f , g , \mathbf { x } ^ { 0 } )$ where f and $g$ satisfy properties $\mathrm { ( A ) }$ and (B) in Assumption 10.31 and $\mathbf { x } ^ { 0 } \in \mathbb { E }$ Initialization: set $\mathbf { y } ^ { 0 } = \mathbf { x } ^ { 0 }$ and $t _ { 0 } = 1$ General step: for any $k = 0 , 1 , 2 , . . .$ . execute the following steps: 

(a) pick $L _ { k } > 0 ;$ 

(b) set $\begin{array} { r } { \mathbf { x } ^ { k + 1 } = \operatorname { p r o x } _ { \frac { 1 } { L _ { k } } g } \left( \mathbf { y } ^ { k } - \frac { 1 } { L _ { k } } \nabla f ( \mathbf { y } ^ { k } ) \right) } \end{array}$ ; 

(c) set $\begin{array} { r } { t _ { k + 1 } = \frac { 1 + \sqrt { 1 + 4 t _ { k } ^ { 2 } } } { 2 } ; } \end{array}$ 

(d) compute $\begin{array} { r } { \mathbf { y } ^ { k + 1 } = \mathbf { x } ^ { k + 1 } + \left( \frac { t _ { k } - 1 } { t _ { k + 1 } } \right) ( \mathbf { x } ^ { k + 1 } - \mathbf { x } ^ { k } ) . } \end{array}$ 

As usual, we will consider two options for the choice of $L _ { k }$ : constant and backtracking. The backtracking procedure for choosing the stepsize is referred to as “backtracking procedure $\mathrm { B 3 ^ { \prime } }$ and is identical to procedure B2 with the sole diference that it is invoked on the vector $\mathbf { y } ^ { k }$ rather than on $\mathbf { x } ^ { k }$ 

• Constant. $L _ { k } = L _ { f }$ for all $k .$ 

• Backtracking procedure B3. The procedure requires two parameters $( s , \eta )$ , where $s > 0$ and $\eta > 1$ . Define $L _ { - 1 } = s$ . At iteration $k \ ( k \geq 0 )$ the choice of $L _ { k }$ is done as follows: First, $L _ { k }$ is set to be equal to $L _ { k - 1 } .$ Then, while (recall that $\begin{array} { r } { T _ { L } ( \mathbf { y } ) \equiv T _ { L } ^ { f , g } ( \mathbf { y } ) = \mathrm { p r o x } _ { \frac { 1 } { L } g } \big ( \mathbf { y } - \frac { 1 } { L } \overbar { \nabla } f ( \mathbf { y } ) \big ) \big ) } \end{array}$ 2 

$$
f (T _ {L _ {k}} (\mathbf {y} ^ {k})) > f (\mathbf {y} ^ {k}) + \langle \nabla f (\mathbf {y} ^ {k}), T _ {L _ {k}} (\mathbf {y} ^ {k}) - \mathbf {y} ^ {k} \rangle + \frac {L _ {k}}{2} \| T _ {L _ {k}} (\mathbf {y} ^ {k}) - \mathbf {y} ^ {k} \| ^ {2},
$$

we set $L _ { k } : = \eta L _ { k }$ . In other words, the stepsize is chosen as $L _ { k } = L _ { k - 1 } \eta ^ { i _ { k } }$ • where $i _ { k }$ is the smallest nonnegative integer for which the condition 

$$
f (T _ {L _ {k - 1} \eta^ {i _ {k}}} (\mathbf {y} ^ {k})) \leq f (\mathbf {y} ^ {k}) + \langle \nabla f (\mathbf {y} ^ {k}), T _ {L _ {k - 1} \eta^ {i _ {k}}} (\mathbf {y} ^ {k}) - \mathbf {y} ^ {k} \rangle
$$

$$
+ \frac {L _ {k}}{2} \| T _ {L _ {k - 1} \eta^ {i _ {k}}} (\mathbf {y} ^ {k}) - \mathbf {y} ^ {k} \| ^ {2}
$$

is satisfied. 

In both stepsize rules, the following inequality is satisfied for any $k \geq 0$ 

$$
f (T _ {L _ {k}} (\mathbf {y} ^ {k})) \leq f (\mathbf {y} ^ {k}) + \langle \nabla f (\mathbf {y} ^ {k}), T _ {L _ {k}} (\mathbf {y} ^ {k}) - \mathbf {y} ^ {k} \rangle + \frac {L _ {k}}{2} \| T _ {L _ {k}} (\mathbf {y} ^ {k}) - \mathbf {y} ^ {k} \| ^ {2}.\tag{10.39}
$$

Remark 10.32. Since the backtracking procedure B3 is identical to the B2 procedure (only employed on $\mathbf { y } ^ { k } )$ , the arguments of Remark 10.19 are still valid, and we have that 

$$
\beta L _ {f} \leq L _ {k} \leq \alpha L _ {f},
$$

where α and $\beta$ are given in (10.25). 

The next lemma shows an important lower bound on the sequence $\{ t _ { k } \} _ { k \geq 0 }$ that will be used in the convergence proof. 

Lemma 10.33. Let $\{ t _ { k } \} _ { k \geq 0 }$ be the sequence defined by 

$$
t _ {0} = 1, t _ {k + 1} = \frac {1 + \sqrt {1 + 4 t _ {k} ^ {2}}}{2}, k \geq 0.
$$

Then $\begin{array} { r } { t _ { k } \ge \frac { k + 2 } { 2 } } \end{array}$ for all $k \geq 0$ 

Proof. The proof is by induction on k. Obviously, for $\begin{array} { r } { k = 0 , t _ { 0 } = 1 \geq \frac { 0 + 2 } { 2 } } \end{array}$ . Suppose that the claim holds for $k ,$ , meaning $\begin{array} { r } { t _ { k } \ge \frac { k + 2 } { 2 } } \end{array}$ . We will prove that $\textstyle t _ { k + 1 } \geq { \frac { k + 3 } { 2 } }$ . By the recursive relation defining the sequence and the induction assumption, 

$$
t _ {k + 1} = \frac {1 + \sqrt {1 + 4 t _ {k} ^ {2}}}{2} \geq \frac {1 + \sqrt {1 + (k + 2) ^ {2}}}{2} \geq \frac {1 + \sqrt {(k + 2) ^ {2}}}{2} = \frac {k + 3}{2}.
$$

## 10.7.2 Convergence Analysis of FISTA

Theorem 10.34 $( O ( 1 / k ^ { 2 } )$ rate of convergence of FISTA). Suppose that $A s -$ sumption 10.31 holds. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by FISTA for solving problem (10.1) with either a constant stepsize rule in which $L _ { k } \equiv L _ { f }$ for all $k \geq 0$ or the backtracking procedure B3. Then for any $\mathbf { x } ^ { * } \in X ^ { * }$ and $k \geq 1$ 

$$
F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} \leq \frac {2 \alpha L _ {f} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}}{(k + 1) ^ {2}},
$$

where $\alpha = 1$ in the constant stepsize setting and $\begin{array} { r } { \alpha = \operatorname* { m a x } \left\{ \eta , \frac { s } { L _ { f } } \right\} } \end{array}$ if the backtracking rule is employed. 

Proof. Let $k \geq 1$ . Substituting $\mathbf { x } = t _ { k } ^ { - 1 } \mathbf { x } ^ { * } + ( 1 - t _ { k } ^ { - 1 } ) \mathbf { x } ^ { k } , \mathbf { y } = \mathbf { y } ^ { k }$ , and $L = L _ { k }$ in the fundamental prox-grad inequality (10.21), taking into account that inequality (10.39) is satisfied and that $f$ is convex, we obtain that 

$$
\begin{array}{l} F (t _ {k} ^ {- 1} \mathbf {x} ^ {*} + (1 - t _ {k} ^ {- 1}) \mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \\ \geq \frac {L _ {k}}{2} \| \mathbf {x} ^ {k + 1} - (t _ {k} ^ {- 1} \mathbf {x} ^ {*} + (1 - t _ {k} ^ {- 1}) \mathbf {x} ^ {k}) \| ^ {2} - \frac {L _ {k}}{2} \| \mathbf {y} ^ {k} - (t _ {k} ^ {- 1} \mathbf {x} ^ {*} + (1 - t _ {k} ^ {- 1}) \mathbf {x} ^ {k}) \| ^ {2} \\ = \frac {L _ {k}}{2 t _ {k} ^ {2}} \| t _ {k} \mathbf {x} ^ {k + 1} - (\mathbf {x} ^ {*} + (t _ {k} - 1) \mathbf {x} ^ {k}) \| ^ {2} - \frac {L _ {k}}{2 t _ {k} ^ {2}} \| t _ {k} \mathbf {y} ^ {k} - (\mathbf {x} ^ {*} + (t _ {k} - 1) \mathbf {x} ^ {k}) \| ^ {2}. \end{array} \tag {16}\tag{10.40}
$$

By the convexity of $F$ , 

$$
F (t _ {k} ^ {- 1} \mathbf {x} ^ {*} + (1 - t _ {k} ^ {- 1}) \mathbf {x} ^ {k}) \leq t _ {k} ^ {- 1} F (\mathbf {x} ^ {*}) + (1 - t _ {k} ^ {- 1}) F (\mathbf {x} ^ {k}).
$$

Therefore, using the notation $v _ { n } \equiv F ( \mathbf { x } ^ { n } ) - F _ { \mathrm { o p t } }$ for any $n \geq 0$ 

$$
\begin{array}{r l} F (t _ {k} ^ {- 1} \mathbf {x} ^ {*} + (1 - t _ {k} ^ {- 1}) \mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) & \leq (1 - t _ {k} ^ {- 1}) (F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {*})) - (F (\mathbf {x} _ {k + 1}) - F (\mathbf {x} ^ {*})) \\ & = (1 - t _ {k} ^ {- 1}) v _ {k} - v _ {k + 1}. \end{array} \tag {10.41}
$$

On the other hand, using the relation $\begin{array} { r } { \mathbf { y } ^ { k } = \mathbf { x } ^ { k } + \left( \frac { t _ { k - 1 } - 1 } { t _ { k } } \right) ( \mathbf { x } ^ { k } - \mathbf { x } ^ { k - 1 } ) } \end{array}$ 

$$
\begin{array}{r l} \| t _ {k} \mathbf {y} ^ {k} - (\mathbf {x} ^ {*} + (t _ {k} - 1) \mathbf {x} ^ {k}) \| ^ {2} & = \| t _ {k} \mathbf {x} ^ {k} + (t _ {k - 1} - 1) (\mathbf {x} ^ {k} - \mathbf {x} ^ {k - 1}) - (\mathbf {x} ^ {*} + (t _ {k} - 1) \mathbf {x} ^ {k}) \| ^ {2} \\ & = \| t _ {k - 1} \mathbf {x} ^ {k} - (\mathbf {x} ^ {*} + (t _ {k - 1} - 1) \mathbf {x} ^ {k - 1}) \| ^ {2}. \end{array} \tag {10.42}
$$

Combining (10.40), (10.41), and (10.42), we obtain that 

$$
(t _ {k} ^ {2} - t _ {k}) v _ {k} - t _ {k} ^ {2} v _ {k + 1} \geq \frac {L _ {k}}{2} \| \mathbf {u} ^ {k + 1} \| ^ {2} - \frac {L _ {k}}{2} \| \mathbf {u} ^ {k} \| ^ {2},
$$

where we use the notation $\mathbf { u } ^ { n } = t _ { n - 1 } \mathbf { x } ^ { n } - ( \mathbf { x } ^ { * } + ( t _ { n - 1 } - 1 ) \mathbf { x } ^ { n - 1 } )$ for any $n \geq 0$ . By the update rule of $t _ { k + 1 }$ , we have $t _ { k } ^ { 2 } - t _ { k } = t _ { k - 1 } ^ { 2 }$ , and hence 

$$
\frac {2}{L _ {k}} t _ {k - 1} ^ {2} v _ {k} - \frac {2}{L _ {k}} t _ {k} ^ {2} v _ {k + 1} \geq \| \mathbf {u} ^ {k + 1} \| ^ {2} - \| \mathbf {u} ^ {k} \| ^ {2}.
$$

Since $L _ { k } \ge L _ { k - 1 }$ , we can conclude that 

$$
\frac {2}{L _ {k - 1}} t _ {k - 1} ^ {2} v _ {k} - \frac {2}{L _ {k}} t _ {k} ^ {2} v _ {k + 1} \geq \| \mathbf {u} ^ {k + 1} \| ^ {2} - \| \mathbf {u} ^ {k} \| ^ {2}.
$$

Thus, 

$$
\| \mathbf {u} ^ {k + 1} \| ^ {2} + \frac {2}{L _ {k}} t _ {k} ^ {2} v _ {k + 1} \leq \| \mathbf {u} ^ {k} \| ^ {2} + \frac {2}{L _ {k - 1}} t _ {k - 1} ^ {2} v _ {k},
$$

and hence, for any $k \geq 1$ 

$$
\| \mathbf {u} ^ {k} \| ^ {2} + \frac {2}{L _ {k - 1}} t _ {k - 1} ^ {2} v _ {k} \leq \| \mathbf {u} ^ {1} \| ^ {2} + \frac {2}{L _ {0}} t _ {0} ^ {2} v _ {1} = \| \mathbf {x} ^ {1} - \mathbf {x} ^ {*} \| ^ {2} + \frac {2}{L _ {0}} (F (\mathbf {x} ^ {1}) - F _ {\mathrm{opt}})\tag{10.43}
$$

Substituting $\mathbf x = \mathbf x ^ { * } , \mathbf y = \mathbf y ^ { 0 }$ , and $L = L _ { 0 }$ in the fundamental prox-grad inequality (10.21), taking into account the convexity of f yields 

$$
\frac {2}{L _ {0}} (F (\mathbf {x} ^ {*}) - F (\mathbf {x} ^ {1})) \geq \| \mathbf {x} ^ {1} - \mathbf {x} ^ {*} \| ^ {2} - \| \mathbf {y} ^ {0} - \mathbf {x} ^ {*} \| ^ {2},
$$

which, along with the fact that $\mathbf { y } ^ { 0 } = \mathbf { x } ^ { 0 }$ , implies the bound 

$$
\| \mathbf {x} ^ {1} - \mathbf {x} ^ {*} \| ^ {2} + \frac {2}{L _ {0}} (F (\mathbf {x} ^ {1}) - F _ {\mathrm{opt}}) \leq \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}.
$$

Combining the last inequality with (10.43), we get 

$$
\frac {2}{L _ {k - 1}} t _ {k - 1} ^ {2} v _ {k} \leq \| \mathbf {u} ^ {k} \| ^ {2} + \frac {2}{L _ {k - 1}} t _ {k - 1} ^ {2} v _ {k} \leq \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}.
$$

Thus, using the bound $L _ { k - 1 } \leq \alpha L _ { f }$ , the definition of $v _ { k }$ , and Lemma 10.33, 

$$
F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} \leq \frac {L _ {k - 1} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}}{2 t _ {k - 1} ^ {2}} \leq \frac {2 \alpha L _ {f} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}}{(k + 1) ^ {2}}.
$$

Remark 10.35 (alternative choice for $\scriptstyle t _ { k } )$ . A close inspection of the proof of Theorem 10.34 reveals that the result is correct $i f \left\{ t _ { k } \right\} _ { k \geq 0 }$ is any sequence satisfying the following two properties for any $\begin{array} { r } { k \geq 0 \colon \mathrm { ( a ) } \ t _ { k } \geq \frac { \bar { k } + \overline { { 2 } } } { 2 } ; \mathrm { ( b ) } \ t _ { k + 1 } ^ { 2 } - t _ { k + 1 } \leq t _ { k } ^ { 2 } } \end{array}$ . The choice $\begin{array} { r } { t _ { k } = \frac { k + 2 } { 2 } } \end{array}$ also satisfies these two properties. The validity of (a) is obvious; to show (b), note that 

$$
\begin{array}{c} t _ {k + 1} ^ {2} - t _ {k + 1} = t _ {k + 1} (t _ {k + 1} - 1) = \frac {k + 3}{2} \cdot \frac {k + 1}{2} = \frac {k ^ {2} + 4 k + 3}{4} \\ \leq \frac {k ^ {2} + 4 k + 4}{4} = \frac {(k + 2) ^ {2}}{4} = t _ {k} ^ {2}. \end{array}
$$

Remark 10.36. Note that FISTA has an $O ( 1 / k ^ { 2 } )$ rate of convergence in function values, while the proximal gradient method has an $O ( 1 / k )$ rate of convergence. This improvement was achieved despite the fact that the dominant computational steps at each iteration of both methods are essentially the same: one gradient evaluation and one prox computation. 

## 10.7.3 Examples

Example 10.37. Consider the following model, which was already discussed in Example 10.2: 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {n}} f (\mathbf {x}) + \lambda \| \mathbf {x} \| _ {1},
$$

where $\lambda > 0$ and $f : \mathbb { R } ^ { n }  \mathbb { R }$ is assumed to be convex and $L _ { f } { \mathrm { - s m o o t h } }$ . The update formula of the proximal gradient method with constant stepsize $\frac { 1 } { L _ { f } }$ has the form 

$$
\mathbf {x} ^ {k + 1} = \mathcal {T} _ {\frac {\lambda}{L _ {f}}} \left(\mathbf {x} ^ {k} - \frac {1}{L _ {f}} \nabla f (\mathbf {x} ^ {k})\right).
$$

As was already noted in Example 10.3, since at each iteration one shrinkage/softthresholding operation is performed, this method is also known as the iterative shrinkage-thresholding algorithm (ISTA). The general update step of the accelerated proximal gradient method discussed in this section takes the following form: 

$$
\text {(a)} \text {set} \mathbf {x} ^ {k + 1} = \mathcal {T} _ {\frac {\lambda}{L _ {f}}} \left(\mathbf {y} ^ {k} - \frac {1}{L _ {f}} \nabla f (\mathbf {y} ^ {k})\right);
$$

(b) set $\begin{array} { r } { t _ { k + 1 } = \frac { 1 + \sqrt { 1 + 4 t _ { k } ^ { 2 } } } { 2 } } \end{array}$ ; 

$$
\mathrm{(c)} \mathrm{compute} \mathbf {y} ^ {k + 1} = \mathbf {x} ^ {k + 1} + \left(\frac {t _ {k} - 1}{t _ {k + 1}}\right) (\mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k}).
$$

The above scheme truly deserves to be called “fast iterative shrinkage/thresholding algorithm” (FISTA) since it is an accelerated method that performs at each iteration a thresholding step. In this book we adopt the convention and use the acronym FISTA as the name of the fast proximal gradient method for a general nonsmooth part g. 

Example 10.38 (l -regularized least squares). As a special instance of Example 10.37, consider the problem 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {n}} \frac {1}{2} \| \mathbf {A x} - \mathbf {b} \| _ {2} ^ {2} + \lambda \| \mathbf {x} \| _ {1},\tag{10.44}
$$

where $\mathbf { A } \ \in \ \mathbb { R } ^ { m \times n } , \mathbf { b } \ \in \ \mathbb { R } ^ { m }$ , and $\lambda > 0$ . The problem fits model (10.1) with $f ( \mathbf { x } ) \ = \ { \textstyle \frac { 1 } { 2 } } \| \mathbf { A x } - \mathbf { b } \| _ { 2 } ^ { 2 }$ and $g ( \mathbf { x } ) ~ = ~ \lambda \| \mathbf { x } \| _ { 1 }$ The function $f$ is $L _ { f } { \mathrm { - s m o o t h } }$ with $L _ { f } = \| \mathbf { A } ^ { T } \mathbf { A } \| _ { 2 . 2 } = \bar { \lambda } _ { \mathrm { m a x } } ( \mathbf { A } ^ { T } \mathbf { A } )$ (see Example 5.2). The update step of FISTA / /has the following form: 

$$
\text {(a)} \mathrm{set} \mathbf {x} ^ {k + 1} = \mathcal {T} _ {\frac {\lambda}{L _ {k}}} \left(\mathbf {y} ^ {k} - \frac {1}{L _ {k}} \mathbf {A} ^ {T} (\mathbf {A y} ^ {k} - \mathbf {b})\right);
$$

(b) set $\begin{array} { r } { t _ { k + 1 } = \frac { 1 + \sqrt { 1 + 4 t _ { k } ^ { 2 } } } { 2 } } \end{array}$ ; 

(c) compute $\begin{array} { r } { \mathbf { y } ^ { k + 1 } = \mathbf { x } ^ { k + 1 } + \left( \frac { t _ { k } - 1 } { t _ { k + 1 } } \right) ( \mathbf { x } ^ { k + 1 } - \mathbf { x } ^ { k } ) . } \end{array}$ 

The update step of the proximal gradient method, which in this case is the same as ISTA, is 

$$
\mathbf {x} ^ {k + 1} = \mathcal {T} _ {\frac {\lambda}{L _ {k}}} \left(\mathbf {x} ^ {k} - \frac {1}{L _ {k}} \mathbf {A} ^ {T} (\mathbf {A x} ^ {k} - \mathbf {b})\right).
$$

The stepsizes in both methods can be chosen to be the constant $L _ { k } \equiv \lambda _ { \operatorname* { m a x } } ( \mathbf { A } ^ { T } \mathbf { A } )$ 

To illustrate the diference in the actual performance of ISTA and FISTA, we generated an instance of the problem with $\lambda = 1$ and $\mathbf { A } \in \mathbb { R } ^ { 1 0 0 \times 1 1 0 }$ The components of A were independently generated using a standard normal distribution. The $\mathrm { ^ { 6 4 } t r u e ^ { 7 } }$ vector is ${ \bf x } _ { \mathrm { t r u e } } = { \bf e } _ { 3 } - { \bf e } _ { 7 }$ , and b was chosen as $\mathbf { b } = \mathbf { A x } _ { \mathrm { t r u e } }$ . We ran 200 iterations of ISTA and FISTA in order to solve problem (10.44) with initial vector ${ \bf x } = { \bf e }$ , the vector of all ones. It is well known that the $l _ { 1 } { \mathrm { - n o r m } }$ element in the objective function is a regularizer that promotes sparsity, and we thus expect that the optimal solution of (10.44) will be close to the “true” sparse vector $\mathbf { x } _ { \mathrm { t r u e } } .$ The distances to optimality in terms of function values of the sequences generated by the two methods as a function of the iteration index are plotted in Figure 10.1, where it is apparent that FISTA is far superior to ISTA. 

In Figure 10.2 we plot the vectors that were obtained by the two methods. Obviously, the solution produced by 200 iterations of FISTA is much closer to the optimal solution (which is very close to ${ \bf e } _ { 3 } - { \bf e } _ { 7 } )$ than the solution obtained after 200 iterations of ISTA. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/811020f3884c1f6d52fe381464a17757a79c523d3218c3c3de09f8cf0f5a0b25.jpg)



Figure 10.1. Results of 200 iterations of ISTA and FISTA on an $l _ { 1 } -$ regularized least squares problem.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/c71f7a02a69c0c258a82ad8b294f3fe5a1bf1a1cb442fd729b62a06c8af61613.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/f25265b3f3a181f5dc9e99f54989e50efe1fef0b5f121dee5b5badc89faed072.jpg)



Figure 10.2. Solutions obtained by ISTA (left) and FISTA (right).


## 10.7.4 MFISTA<sup>58</sup>

FISTA is not a monotone method, meaning that the sequence of function values it produces is not necessarily nonincreasing. It is possible to define a monotone version of FISTA, which we call MFISTA, which is a descent method and at the same time preserves the same rate of convergence as FISTA. 

MFISTA
Input: $(f,g,\mathbf{x}^{0})$ , where f and g satisfy properties (A) and (B) in Assumption 10.31 and $x^{0}\inE$ .
Initialization: set $y^{0}=x^{0}$ and $t_{0}=1$ .
General step: for any $k=0,1,2,\ldots$ execute the following steps:
(a) pick $L_{k}>0;$ (b) set $z^{k}=\operatorname{prox}_{\frac{1}{L_{k}}g}\left(\mathbf{y}^{k}-\frac{1}{L_{k}}\nabla f(\mathbf{y}^{k})\right);$ (c) choose $x^{k+1}\in E$ such that $F(\mathbf{x}^{k+1})\leq\min\{F(\mathbf{z}^{k}),F(\mathbf{x}^{k})\}$ ;
(d) set $t_{k+1}=\frac{1+\sqrt{1+4t_{k}^{2}}}{2};$ (e) compute $y^{k+1}=x^{k+1}+\frac{t_{k}}{t_{k+1}}(z^{k}-x^{k+1})+\left(\frac{t_{k}-1}{t_{k+1}}\right)(x^{k+1}-x^{k}).$ 

Remark 10.39. The choice $\mathbf { x } ^ { k + 1 } \in \operatorname { a r g m i n } \{ F ( \mathbf { x } ) : \mathbf { x } = \mathbf { x } ^ { k } , \mathbf { z } ^ { k } \}$ is a very simple rule ensuring the condition $F ( \mathbf x ^ { k + 1 } ) \leq \mathrm { m i n } \{ \bar { F } ( \mathbf z ^ { k } ) , F ( \mathbf x ^ { k } ) \}$ . We also note that the convergence established in Theorem 10.40 only requires the condition $F ( \mathbf { x } ^ { k + 1 } ) \ \leq$ $F ( \mathbf { z } ^ { k } )$ 

The convergence result of MFISTA, whose proof is a minor adjustment of the proof of Theorem 10.34, is given below. 

Theorem 10.40 $( O ( 1 / k ^ { 2 } )$ rate of convergence of MFISTA). Suppose that Assumption 10.31 holds. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by MFISTA for solving problem (10.1) with either a constant stepsize rule in which $L _ { k } \equiv L _ { f }$ for all $k \geq 0$ or the backtracking procedure B3. Then for any $\mathbf { x } ^ { * } \in X ^ { * }$ and $k \geq 1$ 

$$
F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} \leq \frac {2 \alpha L _ {f} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}}{(k + 1) ^ {2}},
$$

where $\alpha = 1$ in the constant stepsize setting and $\begin{array} { r } { \alpha = \operatorname* { m a x } \left\{ \eta , \frac { s } { L _ { f } } \right\} } \end{array}$ if the backtracking rule is employed. 

Proof. Let $k \geq 1$ . Substituting $\mathbf { x } = t _ { k } ^ { - 1 } \mathbf { x } ^ { * } + ( 1 - t _ { k } ^ { - 1 } ) \mathbf { x } ^ { k } , \mathbf { y } = \mathbf { y } ^ { k }$ , and $L = L _ { k }$ in the fundamental prox-grad inequality (10.21), taking into account that inequality (10.39) is satisfied and that f is convex, we obtain that 

$$
\begin{array}{l} F (t _ {k} ^ {- 1} \mathbf {x} ^ {*} + (1 - t _ {k} ^ {- 1}) \mathbf {x} ^ {k}) - F (\mathbf {z} ^ {k}) \\ \geq \frac {L _ {k}}{2} \| \mathbf {z} ^ {k} - (t _ {k} ^ {- 1} \mathbf {x} ^ {*} + (1 - t _ {k} ^ {- 1}) \mathbf {x} ^ {k}) \| ^ {2} - \frac {L _ {k}}{2} \| \mathbf {y} ^ {k} - (t _ {k} ^ {- 1} \mathbf {x} ^ {*} + (1 - t _ {k} ^ {- 1}) \mathbf {x} ^ {k}) \| ^ {2} \\ = \frac {L _ {k}}{2 t _ {k} ^ {2}} \| t _ {k} \mathbf {z} ^ {k} - (\mathbf {x} ^ {*} + (t _ {k} - 1) \mathbf {x} ^ {k}) \| ^ {2} - \frac {L _ {k}}{2 t _ {k} ^ {2}} \| t _ {k} \mathbf {y} ^ {k} - (\mathbf {x} ^ {*} + (t _ {k} - 1) \mathbf {x} ^ {k}) \| ^ {2}. \end{array} (1\tag{10.45}
$$

By the convexity of $F$ , 

$$
F (t _ {k} ^ {- 1} \mathbf {x} ^ {*} + (1 - t _ {k} ^ {- 1}) \mathbf {x} ^ {k}) \leq t _ {k} ^ {- 1} F (\mathbf {x} ^ {*}) + (1 - t _ {k} ^ {- 1}) F (\mathbf {x} ^ {k}).
$$

Therefore, using the notation $v _ { n } \equiv F ( \mathbf { x } ^ { n } ) - F _ { \mathrm { o p t } }$ for any $n \geq 0$ and the fact that $F ( \mathbf { x } ^ { k + 1 } ) \leq F ( \mathbf { z } ^ { k } )$ , it follows that 

$$
F (t _ {k} ^ {- 1} \mathbf {x} ^ {*} + (1 - t _ {k} ^ {- 1}) \mathbf {x} ^ {k}) - F (\mathbf {z} ^ {k}) \leq (1 - t _ {k} ^ {- 1}) (F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {*})) - (F (\mathbf {x} _ {k + 1}) - F (\mathbf {x} ^ {*}))
$$

$$
= (1 - t _ {k} ^ {- 1}) v _ {k} - v _ {k + 1}.\tag{10.46}
$$

On the other hand, using the relation $\begin{array} { r } { { \mathbf { y } } ^ { k } = { \mathbf { x } } ^ { k } + \frac { t _ { k - 1 } } { t _ { k } } ( { \mathbf { z } } ^ { k - 1 } - { \mathbf { x } } ^ { k } ) + \left( \frac { t _ { k - 1 } - 1 } { t _ { k } } \right) ( { \mathbf { x } } ^ { k } - } \end{array}$ $\mathbf { x } ^ { k - 1 } )$ , we have 

$$
t _ {k} \mathbf {y} ^ {k} - (\mathbf {x} ^ {*} + (t _ {k} - 1) \mathbf {x} ^ {k}) = t _ {k - 1} \mathbf {z} ^ {k - 1} - (\mathbf {x} ^ {*} + (t _ {k - 1} - 1) \mathbf {x} ^ {k - 1}).\tag{10.47}
$$

Combining (10.45), (10.46), and (10.47), we obtain that 

$$
(t _ {k} ^ {2} - t _ {k}) v _ {k} - t _ {k} ^ {2} v _ {k + 1} \geq \frac {L _ {k}}{2} \| \mathbf {u} ^ {k + 1} \| ^ {2} - \frac {L _ {k}}{2} \| \mathbf {u} ^ {k} \| ^ {2},
$$

where we use the notation $\mathbf { u } ^ { n } = t _ { n - 1 } \mathbf { z } ^ { n - 1 } - ( \mathbf { x } ^ { * } + ( t _ { n - 1 } - 1 ) \mathbf { x } ^ { n - 1 } )$ for any $n \geq 0$ By the update rule of $t _ { k + 1 }$ , we have $t _ { k } ^ { 2 } - t _ { k } = t _ { k - 1 } ^ { 2 }$ , and hence 

$$
\frac {2}{L _ {k}} t _ {k - 1} ^ {2} v _ {k} - \frac {2}{L _ {k}} t _ {k} ^ {2} v _ {k + 1} \geq \| \mathbf {u} ^ {k + 1} \| ^ {2} - \| \mathbf {u} ^ {k} \| ^ {2}.
$$

Since $L _ { k } \ge L _ { k - 1 }$ , we can conclude that 

$$
\frac {2}{L _ {k - 1}} t _ {k - 1} ^ {2} v _ {k} - \frac {2}{L _ {k}} t _ {k} ^ {2} v _ {k + 1} \geq \| \mathbf {u} ^ {k + 1} \| ^ {2} - \| \mathbf {u} ^ {k} \| ^ {2}.
$$

Thus, 

$$
\| \mathbf {u} ^ {k + 1} \| ^ {2} + \frac {2}{L _ {k}} t _ {k} ^ {2} v _ {k + 1} \leq \| \mathbf {u} ^ {k} \| ^ {2} + \frac {2}{L _ {k - 1}} t _ {k - 1} ^ {2} v _ {k},
$$

and hence, for any $k \geq 1$ 

$$
\| \mathbf {u} ^ {k} \| ^ {2} + \frac {2}{L _ {k - 1}} t _ {k - 1} ^ {2} v _ {k} \leq \| \mathbf {u} ^ {1} \| ^ {2} + \frac {2}{L _ {0}} t _ {0} ^ {2} v _ {1} = \| \mathbf {z} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} + \frac {2}{L _ {0}} (F (\mathbf {x} ^ {1}) - F _ {\mathrm{opt}}).\tag{10.48}
$$

Substituting $\mathbf x = \mathbf x ^ { * } , \mathbf y = \mathbf y ^ { 0 }$ , and $L = L _ { 0 }$ in the fundamental prox-grad inequality (10.21), taking into account the convexity of $f _ { : }$ , yields 

$$
\frac {2}{L _ {0}} (F (\mathbf {x} ^ {*}) - F (\mathbf {z} ^ {0})) \geq \| \mathbf {z} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} - \| \mathbf {y} ^ {0} - \mathbf {x} ^ {*} \| ^ {2},
$$

which, along with the facts that $\mathbf { y } ^ { 0 } = \mathbf { x } ^ { 0 }$ and $F ( \mathbf { x } ^ { 1 } ) \leq F ( \mathbf { z } ^ { 0 } )$ , implies the bound 

$$
\| \mathbf {z} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} + \frac {2}{L _ {0}} (F (\mathbf {x} ^ {1}) - F _ {\mathrm{opt}}) \leq \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}.
$$

Combining the last inequality with (10.48), we get 

$$
\frac {2}{L _ {k - 1}} t _ {k - 1} ^ {2} v _ {k} \leq \| \mathbf {u} ^ {k} \| ^ {2} + \frac {2}{L _ {k - 1}} t _ {k - 1} ^ {2} v _ {k} \leq \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}.
$$

$$
F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} \leq \frac {L _ {k - 1} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}}{2 t _ {k - 1} ^ {2}} \leq \frac {2 \alpha L _ {f} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}}{(k + 1) ^ {2}}.
$$

Thus, using the bound $L _ { k - 1 } \leq \alpha L _ { f }$ , the definition of $v _ { k } .$ , and Lemma 10.33, 

## 10.7.5 Weighted FISTA

Consider the main composite model (10.1) under Assumption 10.31. Suppose that $\mathbb { E } = \mathbb { R } ^ { n }$ . Recall that a standing assumption in this chapter is that the underlying space is Euclidean, but this does not mean that the endowed inner product is the dot product. Assume that the endowed inner product is the Q-inner product: $\langle \mathbf { x } , \mathbf { y } \rangle = \mathbf { x } ^ { T } \mathbf { Q } \mathbf { y }$ , where $\mathbf { Q } \in \mathbb { S } _ { + + } ^ { n }$ . In this case, as explained in Remark 3.32, the gradient is given by 

$$
\nabla f (\mathbf {x}) = \mathbf {Q} ^ {- 1} D _ {f} (\mathbf {x}),
$$

where 

$$
D _ {f} (\mathbf {x}) = \left( \begin{array}{c} \frac {\partial f}{\partial x _ {1}} (\mathbf {x}) \\ \frac {\partial f}{\partial x _ {2}} (\mathbf {x}) \\ \vdots \\ \frac {\partial f}{\partial x _ {n}} (\mathbf {x}) \end{array} \right).
$$

We will use a Lipschitz constant of $\nabla f \ \mathrm { w . r . t } $ . the Q-norm, which we will denote by $L _ { f } ^ { \mathbf { Q } }$ . The constant is essentially defined by the relation 

$$
\| \mathbf {Q} ^ {- 1} D _ {f} (\mathbf {x}) - \mathbf {Q} ^ {- 1} D _ {f} (\mathbf {y}) \| _ {\mathbf {Q}} \leq L _ {f} ^ {\mathbf {Q}} \| \mathbf {x} - \mathbf {y} \| _ {\mathbf {Q}} \text {   for   any   } \mathbf {x}, \mathbf {y} \in \mathbb {R} ^ {n}.
$$

The general update rule for FISTA in this case will have the following form: 

(a) set $\begin{array} { r } { \mathbf { x } ^ { k + 1 } = \operatorname { p r o x } _ { \frac { 1 } { L _ { f } ^ { \mathbf { Q } } } g } \left( \mathbf { y } ^ { k } - \frac { 1 } { L _ { f } ^ { \mathbf { Q } } } \mathbf { Q } ^ { - 1 } D _ { f } ( \mathbf { y } ^ { k } ) \right) ; } \end{array}$ ; 

(b) set $\begin{array} { r } { t _ { k + 1 } = \frac { 1 + \sqrt { 1 + 4 t _ { k } ^ { 2 } } } { 2 } } \end{array}$ 

(c) compute $\begin{array} { r } { \mathbf { y } ^ { k + 1 } = \mathbf { x } ^ { k + 1 } + \left( \frac { t _ { k } - 1 } { t _ { k + 1 } } \right) ( \mathbf { x } ^ { k + 1 } - \mathbf { x } ^ { k } ) . } \end{array}$ 

Obviously, the prox operator in step (a) is computed in terms of the Q-norm, meaning that 

$$
\operatorname{prox} _ {h} (\mathbf {x}) = \operatorname{argmin} _ {\mathbf {u} \in \mathbb {R} ^ {n}} \left\{h (\mathbf {u}) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| _ {\mathbf {Q}} ^ {2} \right\}.
$$

The convergence result of Theorem 10.34 will also be written in terms of the $\mathbf { Q } -$ norm: O 

$$
F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} \leq \frac {2 L _ {f} ^ {\mathbf {Q}} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| _ {\mathbf {Q}} ^ {2}}{(k + 1) ^ {2}}.
$$

## 10.7.6 Restarting FISTA in the Strongly Convex Case

We will now assume that in addition to Assumption 10.31, f is σ-strongly convex for some $\sigma > 0$ . Recall that by Theorem 10.30, the proximal gradient method attains an ε-optimal solution after an order of $O ( \kappa \log ( \frac { 1 } { \varepsilon } ) )$ iterations $\begin{array} { r } { \left( \kappa = \frac { L _ { f } } { \sigma } \right) } \end{array}$ . The natural question is obviously how the complexity result improves when using FISTA instead of the proximal gradient method. Perhaps surprisingly, one option for obtaining such an improved result is by considering a version of FISTA that incorporates a restarting of the method after a constant amount of iterations. 

Restarted FISTA
Initialization: pick $z^{-1} \in E$ and a positive integer N. Set $\mathbf{z}^{0} = T_{L_{f}}(\mathbf{z}^{-1})$ .
General step ( $k \geq 0$ ):
• run N iterations of FISTA with constant stepsize ( $L_{k} \equiv L_{f}$ ) and input ( $f, g, z^{k}$ ) and obtain a sequence $\{x^{n}\}_{n=0}^{N}$ ;
• set $z^{k+1} = x^{N}$ . 

The algorithm essentially consists of “outer” iterations, and each one employs N iterations of FISTA. To avoid confusion, the outer iterations will be called cycles. Theorem 10.41 below shows that an order of $O ( \sqrt { \kappa } \log ( \frac { 1 } { \varepsilon } ) )$ FISTA iterations are enough to guarantee that an ε-optimal solution is attained. 

Theorem 10.41 $\begin{array} { r } { ( O ( \sqrt { \kappa } \log ( \frac { 1 } { \varepsilon } ) ) } \end{array}$ complexity of restarted FISTA). Suppose that Assumption 10.31 holds and that f is σ-strongly convex $( \sigma > 0 )$ . Let $\{ \mathbf { z } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the restarted FISTA method employed with $N = \lceil \sqrt { 8 \kappa } - 1 \rceil$ where $\begin{array} { r } { \kappa = \frac { L _ { f } } { \sigma } } \end{array}$ . Let R be an upper bound on $\lVert \mathbf { z } ^ { - 1 } - \mathbf { x } ^ { * } \rVert$ , where $\mathbf { x } ^ { * }$ is the unique optimal solution of problem (10.1). $T h e n ^ { 5 9 }$ 

(a) for any $k \geq 0 .$ 

$$
F (\mathbf {z} ^ {k}) - F _ {\text { opt }} \leq \frac {L _ {f} R ^ {2}}{2} \left(\frac {1}{2}\right) ^ {k};
$$

(b) after k iterations of FISTA with k satisfying 

$$
k \geq \sqrt {8 \kappa} \left(\frac {\log (\frac {1}{\varepsilon})}{\log (2)} + \frac {\log (L _ {f} R ^ {2})}{\log (2)}\right),
$$

an ε-optimal solution is obtained at the end of the last completed cycle. That $i s ,$ 

$$
F (\mathbf {z} ^ {\left\lfloor \frac {k}{N} \right\rfloor}) - F _ {\mathrm{opt}} \leq \varepsilon .
$$

Proof. (a) By Theorem 10.34, for any $n \geq 0$ 

$$
F (\mathbf {z} ^ {n + 1}) - F _ {\mathrm{opt}} \leq \frac {2 L _ {f} \| \mathbf {z} ^ {n} - \mathbf {x} ^ {*} \| ^ {2}}{(N + 1) ^ {2}}.\tag{10.49}
$$

Since $f$ is σ-strongly convex, it follows by Theorem 5.25(b) that 

$$
F (\mathbf {z} ^ {n}) - F _ {\mathrm{opt}} \geq \frac {\sigma}{2} \| \mathbf {z} ^ {n} - \mathbf {x} ^ {*} \| ^ {2},
$$

which, combined with (10.49), yields (recalling that $\kappa = L _ { f } / \sigma )$ 

$$
F (\mathbf {z} ^ {n + 1}) - F _ {\mathrm{opt}} \leq \frac {4 \kappa (F (\mathbf {z} ^ {n}) - F _ {\mathrm{opt}})}{(N + 1) ^ {2}}.\tag{10.50}
$$

Since $N \geq \sqrt { 8 \kappa } - 1$ , it follows that $\begin{array} { r } { \frac { 4 \kappa } { ( N + 1 ) ^ { 2 } } \leq \frac { 1 } { 2 } } \end{array}$ , and hence by (10.50) 

$$
F (\mathbf {z} ^ {n + 1}) - F _ {\mathrm{opt}} \leq \frac {1}{2} (F (\mathbf {z} ^ {n}) - F _ {\mathrm{opt}}).
$$

Employing the above inequality for $n = 0 , 1 , \ldots , k - 1$ , we conclude that 

$$
F (\mathbf {z} ^ {k}) - F _ {\mathrm{opt}} \leq \left(\frac {1}{2}\right) ^ {k} (F (\mathbf {z} ^ {0}) - F _ {\mathrm{opt}}).\tag{10.51}
$$

Note that $\mathbf { z } ^ { 0 } = T _ { L _ { f } } ( \mathbf { z } ^ { - 1 } )$ . Invoking the fundamental prox-grad inequality (10.21) with $\mathbf { x } = \mathbf { x } ^ { * } , \mathbf { y } = \mathbf { z } ^ { - 1 } , L = L _ { f }$ , and taking into account the convexity of $f ,$ we obtain T T 

$$
F (\mathbf {x} ^ {*}) - F (\mathbf {z} ^ {0}) \geq \frac {L _ {f}}{2} \| \mathbf {x} ^ {*} - \mathbf {z} ^ {0} \| ^ {2} - \frac {L _ {f}}{2} \| \mathbf {x} ^ {*} - \mathbf {z} ^ {- 1} \| ^ {2},
$$

and hence 

$$
F (\mathbf {z} ^ {0}) - F _ {\mathrm{opt}} \leq \frac {L _ {f}}{2} \| \mathbf {x} ^ {*} - \mathbf {z} ^ {- 1} \| ^ {2} \leq \frac {L _ {f} R ^ {2}}{2}.\tag{10.52}
$$

Combining (10.51) and (10.52), we obtain 

$$
F (\mathbf {z} ^ {k}) - F _ {\mathrm{opt}} \leq \frac {L _ {f} R ^ {2}}{2} \left(\frac {1}{2}\right) ^ {k}.
$$

(b) If k iterations of FISTA were employed, then $\left\lfloor { \frac { k } { N } } \right\rfloor$ cycles were completed. By part (a), 

$$
F (\mathbf {z} ^ {\lfloor \frac {k}{N} \rfloor}) - F _ {\mathrm{opt}} \leq \frac {L _ {f} R ^ {2}}{2} \left(\frac {1}{2}\right) ^ {\lfloor \frac {k}{N} \rfloor} \leq L _ {f} R ^ {2} \left(\frac {1}{2}\right) ^ {\frac {k}{N}}.
$$

Therefore, a suficient condition for the inequality $F ( \mathbf { z } ^ { \left\lfloor \frac { k } { N } \right\rfloor } ) - F _ { \mathrm { o p t } } \leq \varepsilon$ to hold is that 

$$
L _ {f} R ^ {2} \left(\frac {1}{2}\right) ^ {\frac {k}{N}} \leq \varepsilon ,
$$

which is equivalent to the inequality 

$$
k \geq N \left(\frac {\log (\frac {1}{\varepsilon})}{\log (2)} + \frac {\log (L _ {f} R ^ {2})}{\log (2)}\right).
$$

The claim now follows by the fact that $N = \lceil \sqrt { 8 \kappa } - 1 \rceil \leq \sqrt { 8 \kappa } .$ □ 

## 10.7.7 The Strongly Convex Case (Once Again)—Variation on FISTA

As in the previous section, we will assume that in addition to Assumption 10.31, f is σ-strongly convex for some $\sigma > 0$ . We will define a variant of FISTA, called V-FISTA, that will exhibit the improved linear rate of convergence of the restarted FISTA. This rate is established without any need of restarting of the method. 

## V-FISTA

Input: $(f,g,\mathbf{x}^0)$ , where $f$ and $g$ satisfy properties (A) and (B) in Assumption 10.31, $f$ is $\sigma$ -strongly convex ( $\sigma > 0$ ), and $\mathbf{x}^0 \in \mathbb{E}$ .  
Initialization: set $\mathbf{y}^0 = \mathbf{x}^0$ , $t_0 = 1$ and $\kappa = \frac{L_f}{\sigma}$ .  
General step: for any $k = 0,1,2,\ldots$ execute the following steps:  
(a) set $\mathbf{x}^{k+1} = \mathrm{prox}_{\frac{1}{L_f} g}\left(\mathbf{y}^k - \frac{1}{L_f}\nabla f(\mathbf{y}^k)\right)$ ;  
(b) compute $\mathbf{y}^{k+1} = \mathbf{x}^{k+1} + \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)(\mathbf{x}^{k+1} - \mathbf{x}^k)$ . 

The improved linear rate of convergence is established in the next result, whose proof is a variation on the proof of the rate of convergence of FISTA for the non– strongly convex case (Theorem 10.34). 

Theorem 10.42 $( O ( ( 1 - 1 / { \sqrt { \kappa } } ) ^ { k } )$ rate of convergence of V-FISTA).<sup>60</sup> Suppose that Assumption 10.31 holds and that f is σ-strongly convex $( \sigma > 0 )$ . Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by V-FISTA for solving problem (10.1). Then for any $\mathbf { x } ^ { * } \in X ^ { * }$ and $k \geq 0$ 

$$
F (\mathbf {x} ^ {k}) - F _ {\text { opt }} \leq \left(1 - \frac {1}{\sqrt {\kappa}}\right) ^ {k} \left(F (\mathbf {x} ^ {0}) - F _ {\text { opt }} + \frac {\sigma}{2} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}\right),\tag{10.53}
$$

where $\begin{array} { r } { \kappa = \frac { L _ { f } } { \sigma } } \end{array}$ 

Proof. By the fundamental prox-grad inequality (Theorem 10.16) and the σ-strong convexity of f (invoking Theorem 5.24), it follows that for any $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ 2 

$$
\begin{array}{c} F (\mathbf {x}) - F (T _ {L _ {f}} (\mathbf {y})) \geq \frac {L _ {f}}{2} \| \mathbf {x} - T _ {L _ {f}} \mathbf {y}) \| ^ {2} - \frac {L _ {f}}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2} + f (\mathbf {x}) - f (\mathbf {y}) - \langle \nabla f (\mathbf {y}), \mathbf {x} - \mathbf {y} \rangle \\ \geq \frac {L _ {f}}{2} \| \mathbf {x} - T _ {L _ {f}} (\mathbf {y}) \| ^ {2} - \frac {L _ {f}}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2} + \frac {\sigma}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}. \end{array}
$$

Therefore, 

$$
F (\mathbf {x}) - F (T _ {L _ {f}} (\mathbf {y})) \geq \frac {L _ {f}}{2} \| \mathbf {x} - T _ {L _ {f}} (\mathbf {y}) \| ^ {2} - \frac {L _ {f} - \sigma}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.\tag{10.54}
$$

Let $k \geq 0$ and $\begin{array} { r } { t = \sqrt { \kappa } = \sqrt { \frac { L _ { f } } { \sigma } } } \end{array}$ . Substituting $\mathbf { x } = t ^ { - 1 } \mathbf { x } ^ { * } + ( 1 - t ^ { - 1 } ) \mathbf { x } ^ { k }$ and $\mathbf { y } = \mathbf { y } ^ { k }$ into (10.54), we obtain that 

$$
\begin{array}{l} F (t ^ {- 1} \mathbf {x} ^ {*} + (1 - t ^ {- 1}) \mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \\ \geq \frac {L _ {f}}{2} \| \mathbf {x} ^ {k + 1} - (t ^ {- 1} \mathbf {x} ^ {*} + (1 - t ^ {- 1}) \mathbf {x} ^ {k}) \| ^ {2} - \frac {L _ {f} - \sigma}{2} \| \mathbf {y} ^ {k} - (t ^ {- 1} \mathbf {x} ^ {*} + (1 - t ^ {- 1}) \mathbf {x} ^ {k}) \| ^ {2} \\ = \frac {L _ {f}}{2 t ^ {2}} \| t \mathbf {x} ^ {k + 1} - (\mathbf {x} ^ {*} + (t - 1) \mathbf {x} ^ {k}) \| ^ {2} - \frac {L _ {f} - \sigma}{2 t ^ {2}} \| t \mathbf {y} ^ {k} - (\mathbf {x} ^ {*} + (t - 1) \mathbf {x} ^ {k}) \| ^ {2}. \end{array} \tag {10.55}
$$

By the σ-strong convexity of $F _ { ; }$ 

$$
F (t ^ {- 1} \mathbf {x} ^ {*} + (1 - t ^ {- 1}) \mathbf {x} ^ {k}) \leq t ^ {- 1} F (\mathbf {x} ^ {*}) + (1 - t ^ {- 1}) F (\mathbf {x} ^ {k}) - \frac {\sigma}{2} t ^ {- 1} (1 - t ^ {- 1}) \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2}.
$$

Therefore, using the notation $v _ { n } \equiv F ( \mathbf { x } ^ { n } ) - F _ { \mathrm { o p t } }$ for any $n \geq 0$ 2 

$$
\begin{array}{l} F (t ^ {- 1} \mathbf {x} ^ {*} + (1 - t ^ {- 1}) \mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \\ \leq (1 - t ^ {- 1}) (F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {*})) - (F (\mathbf {x} _ {k + 1}) - F (\mathbf {x} ^ {*})) - \frac {\sigma}{2} t ^ {- 1} (1 - t ^ {- 1}) \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} \\ = (1 - t ^ {- 1}) v _ {k} - v _ {k + 1} - \frac {\sigma}{2} t ^ {- 1} (1 - t ^ {- 1}) \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2}, \end{array}
$$

which, combined with (10.55), yields the inequality 

$$
\begin{array}{l} t (t - 1) v _ {k} + \frac {L _ {f} - \sigma}{2} \| t \mathbf {y} ^ {k} - (\mathbf {x} ^ {*} + (t - 1) \mathbf {x} ^ {k}) \| ^ {2} - \frac {\sigma (t - 1)}{2} \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} \\ \geq t ^ {2} v _ {k + 1} + \frac {L _ {f}}{2} \| t \mathbf {x} ^ {k + 1} - (\mathbf {x} ^ {*} + (t - 1) \mathbf {x} ^ {k}) \| ^ {2}. \end{array}\tag{10.56}
$$

We will use the following identity that holds for any a, b $\in \mathbb { E }$ and $\beta \in [ 0 , 1 )$ : 

$$
\| \mathbf {a} + \mathbf {b} \| ^ {2} - \beta \| \mathbf {a} \| ^ {2} = (1 - \beta) \left\| \mathbf {a} + \frac {1}{1 - \beta} \mathbf {b} \right\| ^ {2} - \frac {\beta}{1 - \beta} \| \mathbf {b} \| ^ {2}.
$$

Plugging $\mathbf { a } = \mathbf { x } ^ { k } - \mathbf { x } ^ { * } , \mathbf { b } = t ( \mathbf { y } ^ { k } - \mathbf { x } ^ { k } )$ , and $\begin{array} { r } { \beta = \frac { \sigma ( t - 1 ) } { L _ { f } - \sigma } } \end{array}$ into the above inequality, we obtain 

$$
\frac {L _ {f} - \sigma}{2} \| t (\mathbf {y} ^ {k} - \mathbf {x} ^ {k}) + \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} - \frac {\sigma (t - 1)}{2} \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2}
$$

$$
= \frac {L _ {f} - \sigma}{2} \left[ \| t (\mathbf {y} ^ {k} - \mathbf {x} ^ {k}) + \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} - \frac {\sigma (t - 1)}{L _ {f} - \sigma} \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} \right]
$$

$$
= \frac {L _ {f} - \sigma}{2} \left[ \frac {L _ {f} - \sigma t}{L _ {f} - \sigma} \left\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} + \frac {L _ {f} - \sigma}{L _ {f} - \sigma t} t (\mathbf {y} ^ {k} - \mathbf {x} ^ {k}) \right\| ^ {2} - \frac {\sigma (t - 1)}{L _ {f} - \sigma t} \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} \right]
$$

$$
\leq \frac {L _ {f} - \sigma t}{2} \left\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} + \frac {L _ {f} - \sigma}{L _ {f} - \sigma t} t (\mathbf {y} ^ {k} - \mathbf {x} ^ {k}) \right\| ^ {2}.
$$

/ /We can therefore conclude from the above inequality and (10.56) that 

$$
\begin{array}{l} t (t - 1) v _ {k} + \frac {L _ {f} - \sigma t}{2} \left\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} + \frac {L _ {f} - \sigma}{L _ {f} - \sigma t} t (\mathbf {y} ^ {k} - \mathbf {x} ^ {k}) \right\| ^ {2} \\ \geq t ^ {2} v _ {k + 1} + \frac {L _ {f}}{2} \| t \mathbf {x} ^ {k + 1} - (\mathbf {x} ^ {*} + (t - 1) \mathbf {x} ^ {k}) \| ^ {2}. \end{array}\tag{10.57}
$$

If $k \geq 1$ , then using the relations $\begin{array} { r } { \mathbf { y } ^ { k } = \mathbf { x } ^ { k } + \frac { \sqrt { \kappa } - 1 } { \sqrt { \kappa } + 1 } ( \mathbf { x } ^ { k } - \mathbf { x } ^ { k - 1 } ) } \end{array}$ and $\begin{array} { r } { t = \sqrt { \kappa } = \sqrt { \frac { L _ { f } } { \sigma } } , } \end{array}$ we obtain 

$$
\begin{array}{r l} & {\mathbf {x} ^ {k} - \mathbf {x} ^ {*} + \frac {L _ {f} - \sigma}{L _ {f} - \sigma t} t (\mathbf {y} ^ {k} - \mathbf {x} ^ {k}) = \mathbf {x} ^ {k} - \mathbf {x} ^ {*} + \frac {L _ {f} - \sigma}{L _ {f} - \sigma t} \frac {t (t - 1)}{t + 1} (\mathbf {x} ^ {k} - \mathbf {x} ^ {k - 1})} \\ & {\qquad = \mathbf {x} ^ {k} - \mathbf {x} ^ {*} + \frac {\kappa - 1}{\kappa - \sqrt {\kappa}} \frac {\sqrt {\kappa} (\sqrt {\kappa} - 1)}{\sqrt {\kappa} + 1} (\mathbf {x} ^ {k} - \mathbf {x} ^ {k - 1})} \\ & {\qquad = \mathbf {x} ^ {k} - \mathbf {x} ^ {*} + (\sqrt {\kappa} - 1) (\mathbf {x} ^ {k} - \mathbf {x} ^ {k - 1})} \\ & {\qquad = t \mathbf {x} ^ {k} - (\mathbf {x} ^ {*} + (t - 1) \mathbf {x} ^ {k - 1}),} \end{array}
$$

and obviously, for the case $k = 0$ (recalling that $\mathbf { y } ^ { 0 } = \mathbf { x } ^ { 0 } )$ 2 

$$
\mathbf {x} ^ {0} - \mathbf {x} ^ {*} + \frac {L _ {f} - \sigma}{L _ {f} - \sigma t} t (\mathbf {y} ^ {0} - \mathbf {x} ^ {0}) = \mathbf {x} ^ {0} - \mathbf {x} ^ {*}.
$$

We can thus deduce that (10.57) can be rewritten as (after division by $t ^ { 2 }$ and using again the definition of t as $\begin{array} { r } { t = \sqrt { \frac { L _ { f } } { \sigma } } ) } \end{array}$ 

$$
\begin{array}{l} v _ {k + 1} + \frac {\sigma}{2} \| t \mathbf {x} ^ {k + 1} - (\mathbf {x} ^ {*} + (t - 1) \mathbf {x} ^ {k}) \| ^ {2} \\ \leq \left\{ \begin{array}{l l} \left(1 - \frac {1}{t}\right) \left[ v _ {k} + \frac {\sigma}{2} \| t \mathbf {x} ^ {k} - (\mathbf {x} ^ {*} + (t - 1) \mathbf {x} ^ {k - 1}) \| ^ {2} \right], & k \geq 1, \\ \left(1 - \frac {1}{t}\right) \left[ v _ {0} + \frac {\sigma}{2} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2} \right], & k = 0. \end{array} \right. \end{array}
$$

We can thus conclude that for any $k \geq 0$ 

$$
v _ {k} \leq \left(1 - \frac {1}{t}\right) ^ {k} \left(v _ {0} + \frac {\sigma}{2} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}\right),
$$

which is the desired result (10.53). 

## 10.8 Smoothing<sup>61</sup>

## 10.8.1 Motivation

In Chapters 8 and 9 we considered methods for solving nonsmooth convex optimization problems with complexity ${ \cal O } ( 1 / \varepsilon ^ { 2 } )$ , meaning that an order of $1 / \varepsilon ^ { 2 }$ iterations were required in order to obtain an ε-optimal solution. On the other hand, FISTA requires $O ( 1 / \sqrt { \varepsilon } )$ iterations in order to find an ε-optimal solution of the composite model 

$$
\min _ {\mathbf {x} \in \mathbb {E}} f (\mathbf {x}) + g (\mathbf {x}),\tag{10.58}
$$

where $f$ is $L _ { f } { \mathrm { - s m o o t h } }$ and convex and $g$ is a proper closed and convex function. In this section we will show how FISTA can be used to devise a method for more general nonsmooth convex problems in an improved complexity of $O ( 1 / \varepsilon )$ . In particular, the model that will be considered includes an additional third term to (10.58): 

$$
\min \{f (\mathbf {x}) + h (\mathbf {x}) + g (\mathbf {x}): \mathbf {x} \in \mathbb {E} \}.\tag{10.59}
$$

The function h will be assumed to be real-valued and convex; we will not assume that it is easy to compute its prox operator (as is implicitly assumed on $g )$ , and hence solving it directly using FISTA with smooth and nonsmooth parts taken as $( f , g + h )$ is not a practical solution approach. The idea will be to find a smooth approximation of $h ,$ say ${ \tilde { h } } ,$ and solve the problem via FISTA with smooth and nonsmooth parts taken as $( f + \tilde { h } , g )$ This simple idea will be the basis for the improved $O ( 1 / \varepsilon )$ complexity. To be able to describe the method, we will need to study in more detail the notions of smooth approximations and smoothability. 

## 10.8.2 Smoothable Functions and Smooth Approximations

Definition 10.43 (smoothable functions). A convex function $h : \mathbb { E }  \mathbb { R }$ is called $( \alpha , \beta )$ -smoothable $( \alpha , \beta > 0 )$ if for any $\mu > 0$ there exists a convex diferentiable function $h _ { \mu } : \mathbb { E } $ R such that the following holds: 

(a) $h _ { \mu } ( \mathbf { x } ) \leq h ( \mathbf { x } ) \leq h _ { \mu } ( \mathbf { x } ) + \beta \mu$ for all $\mathbf { x } \in \mathbb { E }$ 

(b) $h _ { \mu }$ is $\frac { \alpha } { \mu }$ -smooth. 

The function $h _ { \mu }$ is called $a \ { \frac { \mathbf { 1 } } { \mu } }$ -smooth approximation of h with parameters $( \alpha , \beta )$ 

Example 10.44 (smooth approximation of $\left. \mathbf { x } \right. _ { 2 } )$ . Consider the function $h$ : $\mathbb { R } ^ { n } \to \mathbb { R }$ given by $h ( \mathbf { x } ) = \| \mathbf { x } \| _ { 2 }$ . For any $\mu > 0$ , define $h _ { \mu } ( { \bf x } ) \equiv \sqrt { \| { \bf x } \| _ { 2 } ^ { 2 } + \mu ^ { 2 } } - \mu$ Then for any $\mathbf { x } \in \mathbb { R } ^ { n }$ 

$$
h _ {\mu} (\mathbf {x}) = \sqrt {\| \mathbf {x} \| _ {2} ^ {2} + \mu^ {2}} - \mu \leq \| \mathbf {x} \| _ {2} + \mu - \mu = \| \mathbf {x} \| _ {2} = h (\mathbf {x}),
$$

$$
h (\mathbf {x}) = \| \mathbf {x} \| _ {2} \leq \sqrt {\| \mathbf {x} \| _ {2} ^ {2} + \mu^ {2}} = h _ {\mu} (\mathbf {x}) + \mu ,
$$

showing that property (a) in the definition of smoothable functions holds with $\beta = 1$ . To show that property $\underline { { ( \mathrm { b } ) } }$ holds with $\alpha = 1$ , note that by Example 5.14, the function $\varphi ( \mathbf x ) \equiv \sqrt { \| \mathbf x \| _ { 2 } ^ { 2 } + 1 }$ is 1-smooth, and hence $h _ { \mu } ( { \bf x } ) = \mu \varphi ( { \bf x } / \mu ) - \mu$ is $\textstyle { \frac { 1 } { \mu } }$ -smooth. We conclude that $h _ { \mu }$ is a $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation of h with parameters $( 1 , 1 )$ . In the terminology described in Definition 10.43, we showed that h is (1, 1)- smoothable. 

Example 10.45 (smooth approximation of $\mathbf { m a x } _ { i } \{ x _ { i } \} )$ . Consider the function $h : \mathbb { R } ^ { n }  \mathbb { R }$ given by $h ( \mathbf { x } ) = \operatorname* { m a x } \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { n } \}$ . For any $\mu > 0$ , define the function 

$$
h _ {\mu} (\mathbf {x}) = \mu \log \left(\sum_ {i = 1} ^ {n} e ^ {x _ {i} / \mu}\right) - \mu \log n.
$$

Then for any $\mathbf { x } \in \mathbb { R } ^ { n }$ 2 

$$
\begin{array}{l} h _ {\mu} (\mathbf {x}) = \mu \log \left(\sum_ {i = 1} ^ {n} e ^ {x _ {i} / \mu}\right) - \mu \log n \\ \qquad \leq \mu \log \left(n e ^ {\max _ {i} \{x _ {i} \} / \mu}\right) - \mu \log n = h (\mathbf {x}), \end{array}\tag{10.60}
$$

$$
h (\mathbf {x}) = \max _ {i} \{x _ {i} \} \leq \mu \log \left(\sum_ {i = 1} ^ {n} e ^ {x _ {i} / \mu}\right) = h _ {\mu} (\mathbf {x}) + \mu \log n.\tag{10.61}
$$

By Example 5.15, the function $\textstyle \varphi ( \mathbf { x } ) = \log ( \sum _ { i = 1 } ^ { n } e ^ { x _ { i } } )$ is 1-smooth, and hence the function $h _ { \mu } ( { \bf x } ) = \mu \varphi ( { \bf x } / \mu ) - \mu$ log n is $\scriptstyle { \frac { 1 } { \mu } } \displaystyle \mathrm { - s m o o t h }$ . Combining this with (10.60) and (10.61), it follows that $h _ { \mu }$ is a $\scriptstyle { \frac { 1 } { \mu } } \cdot$ -smooth approximation of h with parameters (1, log n). We conclude in particular that $h$ is (1, log n)-smoothable. 

The following result describes two important calculus rules of smooth approximations. 

## Theorem 10.46 (calculus of smooth approximations).

(a) Let $h ^ { 1 } , h ^ { 2 } : \mathbb { E }  \mathbb { R }$ be convex functions, and let $\gamma _ { 1 } , \gamma _ { 2 }$ be nonnegative numbers. Suppose that for a given $\mu > 0 , h _ { \mu } ^ { i }$ is a <sup>1</sup> -smooth approximation of h<sup>i</sup> with pa-μ rameters $( \alpha _ { i } , \beta _ { i } )$ for $i = 1 , 2$ . Then $\gamma _ { 1 } h _ { \mu } ^ { \mathrm { i } } + \gamma _ { 2 } h _ { \mu } ^ { 2 }$ is a $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation $o f \gamma _ { 1 } h ^ { 1 } + \gamma _ { 2 } h ^ { 2 }$ with parameters $\left( \gamma _ { 1 } \alpha _ { 1 } + \gamma _ { 2 } \alpha _ { 2 } , \gamma _ { 1 } \beta _ { 1 } + \gamma _ { 2 } \beta _ { 2 } \right)$ 

(b) Let $\mathcal { A } : \mathbb { E }  \mathbb { V }$ be a linear transformation between the Euclidean spaces $\mathbb { E }$ and V. Let $h : \mathbb { V } \to \mathbb { R }$ be a convex function and define 

$$
q (\mathbf {x}) \equiv h (\mathcal {A} (\mathbf {x}) + \mathbf {b}),
$$

where b $\in \mathbb { V }$ . Suppose that for a given $\mu > 0 , h _ { \mu }$ is $\scriptstyle { \lambda } { \frac { 1 } { \mu } } .$ smooth approximation of h with parameters $( \alpha , \beta )$ . Then the function $q _ { \mu } ( \dot { \mathbf { x } } ) \equiv h _ { \mu } ( \mathcal { A } ( \mathbf { x } ) + \mathbf { b } )$ is a $\textstyle { \frac { 1 } { \mu } } .$ -smooth approximation of q with parameters $( \alpha \| \boldsymbol { A } \| ^ { 2 } , \beta )$ 

Proof. (a) By its definition, $h _ { \mu } ^ { i } ~ ( i { \bf \alpha } = 1 , 2 )$ is convex, $\frac { \alpha _ { i } } { \mu }$ -smooth and satisfies $h _ { \mu } ^ { i } ( { \bf x } ) \leq h ^ { i } ( { \bf x } ) \leq h _ { \mu } ^ { i } ( { \bf x } ) + \beta _ { i } \mu$ for a $\operatorname { u n y } \mathbf { x } \in \mathbb { E }$ . We can thus conclude that $\gamma _ { 1 } h _ { \mu } ^ { 1 } + \gamma _ { 2 } h _ { \mu } ^ { 2 }$ is convex and that for any $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ 2 

$$
\gamma_ {1} h _ {\mu} ^ {1} (\mathbf {x}) + \gamma_ {2} h _ {\mu} ^ {2} (\mathbf {x}) \leq \gamma_ {1} h ^ {1} (\mathbf {x}) + \gamma_ {2} h ^ {2} (\mathbf {x}) \leq \gamma_ {1} h _ {\mu} ^ {1} (\mathbf {x}) + \gamma_ {2} h _ {\mu} ^ {2} (\mathbf {x}) + (\gamma_ {1} \beta_ {1} + \gamma_ {2} \beta_ {2}) \mu ,
$$

as well as 

$$
\begin{array}{r l} & {\| \nabla (\gamma_ {1} h _ {\mu} ^ {1} + \gamma_ {2} h _ {\mu} ^ {2}) (\mathbf {x}) - \nabla (\gamma_ {1} h _ {\mu} ^ {1} + \gamma_ {2} h _ {\mu} ^ {2}) (\mathbf {y}) \| \leq \gamma_ {1} \| \nabla h _ {\mu} ^ {1} (\mathbf {x}) - \nabla h _ {\mu} ^ {1} (\mathbf {y}) \|} \\ & {\qquad + \gamma_ {2} \| \nabla h _ {\mu} ^ {2} (\mathbf {x}) - \nabla h _ {\mu} ^ {2} (\mathbf {y}) \|} \\ & {\qquad \leq \gamma_ {1} \frac {\alpha_ {1}}{\mu} \| \mathbf {x} - \mathbf {y} \| + \gamma_ {2} \frac {\alpha_ {2}}{\mu} \| \mathbf {x} - \mathbf {y} \|} \\ & {\qquad = \frac {\gamma_ {1} \alpha_ {1} + \gamma_ {2} \alpha_ {2}}{\mu} \| \mathbf {x} - \mathbf {y} \|,} \end{array}
$$

establishing the fact that $\gamma _ { 1 } h _ { \mu } ^ { 1 } + \gamma _ { 2 } h _ { \mu } ^ { 2 }$ is a $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation of $\gamma _ { 1 } h ^ { 1 } + \gamma _ { 2 } h ^ { 2 }$ with parameters $\left( \gamma _ { 1 } \alpha _ { 1 } + \gamma _ { 2 } \alpha _ { 2 } , \gamma _ { 1 } \beta _ { 1 } + \gamma _ { 2 } \beta _ { 2 } \right)$ 

(b) Since $h _ { \mu }$ is a $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation of h with parameters $( \alpha , \beta )$ , it follows that $h _ { \mu }$ is $\mathrm { c o n v e x , } \ { \textstyle { \frac { \alpha } { \mu } } }$ -smooth and for any $\mathbf { y } \in \mathbb { V }$ 

$$
h _ {\mu} (\mathbf {y}) \leq h (\mathbf {y}) \leq h _ {\mu} (\mathbf {y}) + \beta \mu .\tag{10.62}
$$

Let $\mathbf { x } \in \mathbb { E }$ . Plugging $\mathbf { y } = \mathcal { A } ( \mathbf { x } ) + \mathbf { b }$ into (10.62), we obtain that 

$$
q _ {\mu} (\mathbf {x}) \leq q (\mathbf {x}) \leq q _ {\mu} (\mathbf {x}) + \beta \mu .\tag{10.63}
$$

In addition, by the $\textstyle { \frac { \alpha } { \mu } } .$ -smoothness of $h _ { \mu }$ , we have for any $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ 

$$
\begin{array}{l} \| \nabla q _ {\mu} (\mathbf {x}) - \nabla q _ {\mu} (\mathbf {y}) \| = \| \mathcal {A} ^ {T} \nabla h _ {\mu} (\mathcal {A} (\mathbf {x}) + \mathbf {b}) - \mathcal {A} ^ {T} \nabla h _ {\mu} (\mathcal {A} (\mathbf {y}) + \mathbf {b}) \| \\ \leq \| \mathcal {A} ^ {T} \| \cdot \| \nabla h _ {\mu} (\mathcal {A} (\mathbf {x}) + \mathbf {b}) - \nabla h _ {\mu} (\mathcal {A} (\mathbf {y}) + \mathbf {b}) \| \\ \leq \frac {\alpha}{\mu} \| \mathcal {A} ^ {T} \| \cdot \| \mathcal {A} (\mathbf {x}) + \mathbf {b} - \mathcal {A} (\mathbf {y}) - \mathbf {b} \| \\ \leq \frac {\alpha}{\mu} \| \mathcal {A} ^ {T} \| \cdot \| \mathcal {A} \| \cdot \| \mathbf {x} - \mathbf {y} \| \\ = \frac {\alpha \| \mathcal {A} \| ^ {2}}{\mu} \| \mathbf {x} - \mathbf {y} \|, \end{array}
$$

where the last equality follows by the fact that $\| \mathcal { A } \| = \| \mathcal { A } ^ { T } \|$ (see Section 1.14). We have thus shown that the convex function $h _ { \mu }$ is $\frac { \alpha \| \mathcal { A } \| ^ { 2 } } { \boldsymbol { \mathsf { I } } ^ { \mu } }$ -smooth and satisfies (10.63) for any $\mathbf { x } \in \mathbb { E }$ , establishing the desired result. 

A direct result of Theorem 10.46 is the following corollary stating the preservation of smoothability under nonnegative linear combinations and afine transformations of variables. 

Corollary 10.47 (operations preserving smoothability). 

(a) Let $h ^ { 1 } , h ^ { 2 } : \mathbb { E }  \mathbb { R }$ be convex functions which are $( \alpha _ { 1 } , \beta _ { 1 } )$ - and $( \alpha _ { 2 } , \beta _ { 2 } ) .$ smoothable, respectively, and let $\gamma _ { 1 } , \gamma _ { 2 }$ be nonnegative numbers. Then $\gamma _ { 1 } h ^ { 1 } +$ $\gamma _ { 2 } h ^ { 2 }$ is $a \left( \gamma _ { 1 } \alpha _ { 1 } + \gamma _ { 2 } \alpha _ { 2 } , \gamma _ { 1 } \beta _ { 1 } + \gamma _ { 2 } \beta _ { 2 } \right)$ -smoothable function. 

(b) Let $\mathcal { A } : \mathbb { E }  \mathbb { V }$ be a linear transformation between the Euclidean spaces $\mathbb { E }$ and V. Let $h : \mathbb { V } \to \mathbb { R }$ be a convex $( \alpha , \beta )$ -smoothable function and define 

$$
q (\mathbf {x}) \equiv h (\mathcal {A} (\mathbf {x}) + \mathbf {b}),
$$

where b $\in \mathbb { V }$ . Then $q$ is $( \alpha \| \boldsymbol { A } \| ^ { 2 } , \beta )$ -smoothable. 

Example 10.48 (smooth approximation of $\| \mathbf { A x } + \mathbf { b } \| _ { 2 } )$ . Let $q : \mathbb { R } ^ { n } $ R be given by $q ( \mathbf { x } ) = \lVert \mathbf { A } \mathbf { x } + \mathbf { b } \rVert _ { 2 }$ , where $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ and $\mathbf { b } \in \mathbb { R } ^ { m }$ . Then $q ( { \bf x } ) = g ( { \bf A x + b } )$ where $g : \mathbb { R } ^ { m }  \mathbb { R }$ is given by $g ( \mathbf { y } ) ~ = ~ \| \mathbf { y } \| _ { 2 }$ . Let $\mu > 0$ . By Example 10.44, $g _ { \mu } ( \mathbf { y } ) = \sqrt { \| \mathbf { y } \| _ { 2 } ^ { 2 } + \mu ^ { 2 } } - \mu$ is a $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation of $g$ with parameters (1, 1), and hence, by Theorem $1 0 . 4 6 \dot { ( } \mathrm { b ) }$ 

$$
q _ {\mu} (\mathbf {x}) \equiv g _ {\mu} (\mathbf {A x} + \mathbf {b}) = \sqrt {\| \mathbf {A x} + \mathbf {b} \| _ {2} ^ {2} + \mu^ {2}} - \mu
$$

is a $\scriptstyle { \frac { 1 } { \mu } } \to \operatorname { s m o o t h }$ approximation of $q$ with parameters $( \| \mathbf { A } \| _ { 2 , 2 } ^ { 2 } , 1 )$ 

Example 10.49 (smooth approximation of piecewise afine functions). Let $q : \mathbb { R } ^ { n } $ R be given by $q ( \mathbf { x } ) =$ ma $\mathbf { x } _ { i = 1 , \ldots , m } \{ \mathbf { a } _ { i } ^ { T } \mathbf { x } + b _ { i } \}$ , where $\mathbf { a } _ { i } \in \mathbb { R } ^ { n }$ and $b _ { i } \in$ R for any $i = 1 , 2 , \dots , m$ . Then $q ( { \bf x } ) = g ( { \bf A x } + { \bf b } )$ , where $g ( \mathbf { y } ) = \operatorname* { m a x } \{ y _ { 1 } , y _ { 2 } , \dots , y _ { m } \}$ , A is the matrix whose rows are $\mathbf { a } _ { 1 } ^ { T } , \mathbf { \dot { a } } _ { 2 } ^ { T } , \dots , \mathbf { \dot { a } } _ { m } ^ { T }$ , and $\mathbf { b } = ( b _ { 1 } , b _ { 2 } , \ldots , b _ { m } ) ^ { T }$ . Let $\mu > 0$ . By Example 10.45, $g _ { \mu } ( \mathbf { y } ) = \mu$ log $\begin{array} { r } { \left( \sum _ { i = 1 } ^ { m } e ^ { y _ { i } / \mu } \right) - \mu } \end{array}$ log m is a $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation of $g$ with parameters $( 1 , \log m )$ . Therefore, by Theorem 10.46(b), the function 

$$
q _ {\mu} (\mathbf {x}) \equiv g _ {\mu} (\mathbf {A x} + \mathbf {b}) = \mu \log \left(\sum_ {i = 1} ^ {m} e ^ {(\mathbf {a} _ {i} ^ {T} \mathbf {x} + b _ {i}) / \mu}\right) - \mu \log m
$$

is a $\scriptstyle { \frac { 1 } { \mu } } \to \operatorname { s m o o t h }$ approximation of $q$ with parameters $( \| \mathbf { A } \| _ { 2 , 2 } ^ { 2 } , \log m )$ 

Example 10.50 (tightness of the smoothing parameters). Consider the absolute value function $q : \mathbb { R }  \mathbb { R }$ given by $q ( x ) = | x |$ . By Example 10.44, for any $\mu > 0$ the function ${ \sqrt { x ^ { 2 } + \mu ^ { 2 } } } - \mu$ is a ${ \frac { 1 } { \mu } } \cdot$ -smooth approximation of $q$ with parameters (1, 1). Let us consider an alternative way to construct a smooth approximation of $q$ using Theorem 10.46. Note that $q ( x ) = \operatorname* { m a x } \{ x , - x \}$ . Thus, by Example 10.49 the function $q _ { \mu } ( x ) = \mu \log ( e ^ { x / \mu } + e ^ { - x / \mu } ) - \mu \log 2$ is a $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation of $q$ with parameters $( \| \mathbf { A } \| _ { 2 , 2 } ^ { 2 } , \log 2 )$ , where $\mathbf { A } \ = \ \left( \begin{array} { c } { 1 } \\ { - 1 } \end{array} \right)$ . Since $\| \mathbf { A } \| _ { 2 , 2 } ^ { 2 } ~ = ~ 2$ , we conclude that $q _ { \mu }$ is a $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation of $q$ with parameters (2, log 2). The question that arises is whether these parameters are tight, meaning whether they are the smallest ones possible. The β-parameter is indeed tight (since $\scriptstyle \operatorname* { l i m } _ { x \to \infty } q ( x ) \ -$ $q _ { \mu } ( x ) = \mu \log ( 2 ) )$ ; however, the α-parameter is not tight. To see this, note that for any $x \in \mathbb { R }$ 

$$
q _ {1} ^ {\prime \prime} (x) = \frac {4}{(e ^ {x} + e ^ {- x}) ^ {2}}.
$$

Therefore, for any $x \in \mathbb { R }$ , it holds that $| q _ { 1 } ^ { \prime \prime } ( x ) | \leq 1$ , and hence, by Theorem 5.12, $q _ { 1 }$ is 1-smooth. Consequently, $q _ { \mu } ,$ , which can also be written as $q _ { \mu } ( \mathbf { x } ) = \mu q _ { 1 } ( \mathbf { x } / \mu )$ , is $\scriptstyle { \frac { 1 } { \mu } } - \operatorname { s m o o t h }$ . We conclude that $q _ { \mu }$ , is a $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation of $q$ with parameters (1, log 2). 

## 10.8.3 The Moreau Envelope Revisited

A natural $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation of a given real-valued convex function $h : \mathbb { E } $ R is its Moreau envelope $M _ { h } ^ { \mu }$ , which was discussed in detail in Section 6.7. Recall that the Moreau envelope of h is given by 

$$
M _ {h} ^ {\mu} (\mathbf {x}) = \min _ {\mathbf {u} \in \mathbb {E}} \left\{h (\mathbf {u}) + \frac {1}{2 \mu} \| \mathbf {x} - \mathbf {u} \| ^ {2} \right\}.
$$

We will now show that whenever $h$ is in addition Lipschitz, the Moreau envelope is indeed a ${ \frac { 1 } { \mu } } \cdot$ -smooth approximation. 

Theorem 10.51 (smoothability of real-valued Lipschitz convex functions). Let $h : \mathbb { E }  \mathbb { R }$ be a convex function satisfying 

$$
| h (\mathbf {x}) - h (\mathbf {y}) | \leq \ell_ {h} \| \mathbf {x} - \mathbf {y} \| f o r a l l \mathbf {x}, \mathbf {y} \in \mathbb {E}.
$$

Then for any $\mu > 0 , M _ { h } ^ { \mu }$ is $\textstyle a { \frac { 1 } { \mu } }$ -smooth approximation of h with parameters $( 1 , \frac { \ell _ { h } ^ { 2 } } { 2 } )$ 

Proof. By Theorem 6.60, $M _ { h } ^ { \mu }$ is $\textstyle { \frac { 1 } { \mu } }$ -smooth. For any $\mathbf { x } \in \mathbb { E }$ 

$$
M _ {h} ^ {\mu} (\mathbf {x}) = \min _ {\mathbf {u} \in \mathbb {E}} \left\{h (\mathbf {u}) + \frac {1}{2 \mu} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\} \leq h (\mathbf {x}) + \frac {1}{2 \mu} \| \mathbf {x} - \mathbf {x} \| ^ {2} = h (\mathbf {x}).
$$

Let ${ \bf g } _ { \bf x } \in \partial h ( { \bf x } )$ . Since $h$ is Lipschitz with constant $\ell _ { h }$ , it follows by Theorem 3.61 that $\| \mathbf { g } _ { \mathbf { x } } \| \leq \ell _ { h }$ , and hence 

$$
\begin{array}{l} M _ {h} ^ {\mu} (\mathbf {x}) - h (\mathbf {x}) = \underset {\mathbf {u} \in \mathbb {E}} {\min} \left\{h (\mathbf {u}) - h (\mathbf {x}) + \frac {1}{2 \mu} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\} \\ \qquad \geq \underset {\mathbf {u} \in \mathbb {E}} {\min} \left\{\langle \mathbf {g} _ {\mathbf {x}}, \mathbf {u} - \mathbf {x} \rangle + \frac {1}{2 \mu} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\} \\ \qquad = - \frac {\mu}{2} \| \mathbf {g} _ {\mathbf {x}} \| ^ {2} \\ \qquad \geq - \frac {\ell_ {h} ^ {2}}{2} \mu , \end{array}
$$

where the subgradient inequality was used in the first inequality. To summarize, we obtained that the convex function $M _ { h } ^ { \mu }$ is $\scriptstyle { \frac { 1 } { \mu } } ,$ -smooth and satisfies 

$$
M _ {h} ^ {\mu} (\mathbf {x}) \leq h (\mathbf {x}) \leq M _ {h} ^ {\mu} (\mathbf {x}) + \frac {\ell_ {h} ^ {2}}{2} \mu ,
$$

showing that $\boldsymbol { M } _ { h } ^ { \mu }$ is a $\scriptstyle { \frac { 1 } { \mu } } ,$ -smooth approximation of h with parameters $( 1 , \frac { \ell _ { h } ^ { 2 } } { 2 } )$ 

Corollary 10.52. Let $h : \mathbb { E }  \mathbb { R }$ be convex and Lipschitz with constant $\ell _ { h }$ . Then h is $( 1 , \frac { \ell _ { h } ^ { 2 } } { 2 } )$ -smoothable. 

Example 10.53 (smooth approximation of the $l _ { 2 } { \bf - n o r m } )$ . Consider the function $h : \mathbb { R } ^ { n }  \mathbb { R }$ given by $h ( \mathbf { x } ) = \| \mathbf { x } \| _ { 2 }$ . Then h is convex and Lipschitz with constant $\ell _ { h } ~ = ~ 1$ Hence, by Theorem 10.51, for any $\mu > 0$ , the function (see Example 6.54) 

$$
M _ {h} ^ {\mu} (\mathbf {x}) = H _ {\mu} (\mathbf {x}) = \left\{ \begin{array}{l l} \frac {1}{2 \mu} \| \mathbf {x} \| _ {2} ^ {2}, & \| \mathbf {x} \| _ {2} \leq \mu , \\ \| \mathbf {x} \| _ {2} - \frac {\mu}{2}, & \| \mathbf {x} \| _ {2} > \mu , \end{array} \right.
$$

is a $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation of h with parameters $( 1 , { \frac { 1 } { 2 } } )$ 

Example 10.54 (smooth approximation of the $l _ { 1 } { \bf - n o r m } )$ . Consider the function $h : \mathbb { R } ^ { n }  \mathbb { R }$ given by $h ( \mathbf { x } ) = \| \mathbf { x } \| _ { 1 }$ . Then h is convex and Lipschitz with constant $\ell _ { h } = \sqrt { n }$ . Hence, by Theorem 10.51, for any $\mu > 0$ , the Moreau envelope of $h$ given by 

$$
M _ {h} ^ {\mu} (\mathbf {x}) = \sum_ {i = 1} ^ {n} H _ {\mu} (x _ {i})
$$

is a $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation of h with parameters $( 1 , { \frac { n } { 2 } } )$ 

Example 10.55 (smooth approximations of the absolute value function). Let us consider again the absolute value function $h ( x ) = \vert x \vert$ . In our discussions we actually considered three possible $\textstyle { \frac { 1 } { \mu } }$ -smooth approximations of $h$ , which are detailed below along with their parameters: 

• (Example 10.44) $h _ { \mu } ^ { 1 } ( x ) = \sqrt { x ^ { 2 } + \mu ^ { 2 } } - \mu , ( \alpha , \beta ) = ( 1 , 1 )$ 

• (Example 10.50) $h _ { \mu } ^ { 2 } ( x ) = \mu \log ( e ^ { x / \mu } + e ^ { - x / \mu } ) - \mu \log 2 , ( \alpha , \beta ) = ( 1 , \log 2 )$ 

• (Example 10.53) $\begin{array} { r } { h _ { \mu } ^ { 3 } ( x ) = H _ { \mu } ( x ) , ( \alpha , \beta ) = ( 1 , \frac { 1 } { 2 } ) } \end{array}$ 

Obviously, the Huber function is the best $\scriptstyle { \frac { 1 } { \mu } } \mathrm { - s m o o t h }$ approximation out of the three functions since all the functions have the same α-parameter, but $h _ { \mu } ^ { 3 }$ has the smallest β-parameter. This phenomenon is illustrated in Figure 10.3, where the three functions are plotted (for the case $\mu = 0 . 2 )$ ■ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/974749a1d063e33b8f228815beed8ca6c9c110fe4b91b7cb56a1bb93fa024c54.jpg)



Figure 10.3. The absolute value function along with its three 5-smooth approximations $( \mu = 0 . 2 )$ . “squared-based” is the function $h _ { \mu } ^ { 1 } ( x ) = \sqrt { x ^ { 2 } + \mu ^ { 2 } } - \mu ,$ “log-exp” is $h _ { \mu } ^ { 2 } ( x ) = \mu \log ( e ^ { x / \mu } + e ^ { - x / \mu } ) - \mu$ log 2, and “Huber” is $h _ { \mu } ^ { 3 } ( x ) = H _ { \mu } ( x )$


## 10.8.4 The S-FISTA Method

The optimization model that we consider is 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \{H (\mathbf {x}) \equiv f (\mathbf {x}) + h (\mathbf {x}) + g (\mathbf {x}) \},\tag{10.64}
$$

where the following assumptions are made. 

Assumption 10.56. 

(A) $f : \mathbb { E } \to \mathbb { R }$ is L<sub>f</sub>-smooth $\left( L _ { f } \geq 0 \right)$ 

(B) $h : \mathbb { E }  \mathbb { R }$ is $( \alpha , \beta )$ -smoothable $( \alpha , \beta > 0 )$ . For any $\mu > 0 , h _ { \mu }$ denotes a 1 -smooth approximation of h with parameters $( \alpha , \beta )$ μ 

(C) $g : \mathbb { E }  ( - \infty , \infty ]$ is proper closed and convex. 

(D) H has bounded level sets. Specifically, for any $\delta > 0$ , there exists $R _ { \delta } > 0$ such that 

$$
\| \mathbf {x} \| \leq R _ {\delta} f o r a n y \mathbf {x} s a t i s f y i n g H (\mathbf {x}) \leq \delta .
$$

(E) The optimal set of problem (10.64) is nonempty and denoted by $X ^ { * }$ . The optimal value of the problem is denoted by $H _ { \mathrm { o p t } }$ 

Assumption (E) is actually a consequence of assumptions $( \mathrm { A } ) { - } ( \mathrm { D } )$ . The idea is to consider the smoothed version of (10.64), 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \{H _ {\mu} (\mathbf {x}) \equiv \underbrace {f (\mathbf {x}) + h _ {\mu} (\mathbf {x})} _ {F _ {\mu} (\mathbf {x})} + g (\mathbf {x}) \},\tag{10.65}
$$

for some smoothing parameter $\mu > 0$ , and solve it using an accelerated method with convergence rate of $O ( 1 / k ^ { 2 } )$ in function values. Actually, any accelerated method can be employed, but we will describe the version in which FISTA with constant stepsize is employed on (10.65) with the smooth and nonsmooth parts taken as $F _ { \mu }$ and $g ,$ respectively. The method is described in detail below. Note that a Lipschitz constant of the gradient of $F _ { \mu }$ is $\begin{array} { r } { L _ { f } + \frac { \alpha } { \mu } } \end{array}$ , and thus the stepsize is taken as ${ \frac { 1 } { L _ { f } + { \frac { \alpha } { \mu } } } } ,$ 

S-FISTA

Input: $\mathbf{x}^{0} \in \text{dom}(g)$ , $\mu > 0$ .

Initialization: set $y^{0} = x^{0}$ , $t_{0} = 1$ ; construct $h_{\mu}$ —a $\frac{1}{\mu}$ -smooth approximation of h with parameters $(\alpha, \beta)$ ; set $F_{\mu} = f + h_{\mu}$ , $\tilde{L} = L_{f} + \frac{\alpha}{\mu}$ .

General step: for any $k = 0, 1, 2, \ldots$ execute the following steps:

(a) set $x^{k+1} = \text{prox}_{\frac{1}{L} g} \left( y^{k} - \frac{1}{\tilde{L}} \nabla F_{\mu}(y^{k}) \right)$ ;

(b) set $t_{k+1} = \frac{1 + \sqrt{1 + 4t_{k}^{2}}}{2}$ ;

(c) compute $y^{k+1} = x^{k+1} + \left( \frac{t_{k}-1}{t_{k+1}} \right) (x^{k+1} - x^{k})$ . 

The next result shows how, given an accuracy level $\varepsilon > 0$ , the parameter $\mu$ can be chosen to ensure that an ε-optimal solution of the original problem (10.64) is reached in $O ( 1 / \varepsilon )$ iterations. 

Theorem 10.57 $( O ( 1 / \varepsilon )$ complexity of S-FISTA). Suppose that Assumption 10.56 holds. Let $\varepsilon \in ( 0 , \bar { \varepsilon } )$ for some fixed $\bar { \varepsilon } > 0$ . Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by $S – F I S T A$ with smoothing parameter 

$$
\mu = \sqrt {\frac {\alpha}{\beta}} \frac {\varepsilon}{\sqrt {\alpha \beta} + \sqrt {\alpha \beta + L _ {f} \varepsilon}}.
$$

Then for any k satisfying 

$$
k \geq 2 \sqrt {2 \alpha \beta \Gamma} \frac {1}{\varepsilon} + \sqrt {2 L _ {f} \Gamma} \frac {1}{\sqrt {\varepsilon}},
$$

where $\Gamma = ( R _ { H ( \mathbf { x } ^ { 0 } ) + \frac { \bar { \varepsilon } } { 2 } } + | | \mathbf { x } ^ { 0 } | | ) ^ { 2 }$ , it holds that $H ( \mathbf { x } ^ { k } ) - H _ { \mathrm { o p t } } \leq \varepsilon$ 

Proof. By definition of S-FISTA, $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ is the sequence generated by FISTA employed on problem (10.65) with input $( \bar { F } _ { \mu } , g , { \bf x } ^ { 0 } )$ . Note that 

$$
\operatorname{argmin} _ {\mathbf {x} \in \mathbb {E}} H _ {\mu} (\mathbf {x}) = \operatorname{argmin} _ {\mathbf {x} \in \mathbb {E}} \left\{H _ {\mu} (\mathbf {x}): H _ {\mu} (\mathbf {x}) \leq H _ {\mu} (\mathbf {x} ^ {0}) \right\}.\tag{10.66}
$$

Since $H _ { \mu }$ is closed, the feasible set $C \equiv \{ \mathbf { x } \in \mathbb { E } : H _ { \mu } ( \mathbf { x } ) \leq H _ { \mu } ( \mathbf { x } ^ { 0 } ) \}$ of the right-hand side problem in (10.66) is closed. We will show that it is also bounded. Indeed, since $h _ { \mu }$ is a $\scriptstyle { \frac { 1 } { \mu } } .$ -smooth approximation of h with parameters $( \alpha , \beta )$ , it follows in particular that $h ( \mathbf { \dot { x } } ) \leq h _ { \mu } ( \mathbf { x } ) + \beta \mu$ for all $\mathbf { x } \in \mathbb { E }$ , and consequently $H ( \mathbf { x } ) \leq H _ { \mu } ( \mathbf { x } ) + \beta \mu$ for all $\mathbf { x } \in \mathbb { E }$ . Thus, 

$$
C \subseteq \{\mathbf {x} \in \mathbb {E}: H (\mathbf {x}) \leq H _ {\mu} \left(\mathbf {x} ^ {0}\right) + \beta \mu \},
$$

which by Assumption 10.56(D) implies that C is bounded and hence, by its closedness, also compact. We can therefore conclude by Weierstrass theorem for closed functions (Theorem 2.12) that an optimal solution of problem (10.65) is attained at some point $\mathbf { x } _ { \mu } ^ { \ast }$ with an optimal value $H _ { \mu , \mathrm { o p t } }$ . By Theorem 10.34, since $F _ { \mu }$ is $\begin{array} { r } { ( L _ { f } + \frac { \alpha } { \mu } ) } \end{array}$ )-smooth, 

$$
H _ {\mu} (\mathbf {x} ^ {k}) - H _ {\mu , \mathrm{opt}} \leq 2 \left(L _ {f} + \frac {\alpha}{\mu}\right) \frac {\| \mathbf {x} ^ {0} - \mathbf {x} _ {\mu} ^ {*} \| ^ {2}}{(k + 1) ^ {2}} = 2 \left(L _ {f} + \frac {\alpha}{\mu}\right) \frac {\Lambda}{(k + 1) ^ {2}},\tag{10.67}
$$

where $\Lambda = \| \mathbf { x } ^ { 0 } - \mathbf { x } _ { \mu } ^ { * } \| ^ { 2 }$ . We use again the fact that $h _ { \mu }$ is a ${ \frac { 1 } { \mu } } \cdot$ -smooth approximation of h with parameters $( \alpha , \beta )$ , from which it follows that for any $\mathbf { x } \in \mathbb { E }$ ， 

$$
H _ {\mu} (\mathbf {x}) \leq H (\mathbf {x}) \leq H _ {\mu} (\mathbf {x}) + \beta \mu .\tag{10.68}
$$

In particular, the following two inequalities hold: 

$$
H _ {\mathrm{opt}} \geq H _ {\mu , \mathrm{opt}} \quad \text { and } \quad H (\mathbf {x} ^ {k}) \leq H _ {\mu} (\mathbf {x} ^ {k}) + \beta \mu , k = 0, 1, \ldots ,\tag{10.69}
$$

which, combined with (10.67), yields 

$$
\begin{array}{c} H (\mathbf {x} ^ {k}) - H _ {\mathrm{opt}} \leq H _ {\mu} (\mathbf {x} ^ {k}) + \beta \mu - H _ {\mu , \mathrm{opt}} \leq 2 L _ {f} \frac {\Lambda}{(k + 1) ^ {2}} + \left(\frac {2 \alpha \Lambda}{(k + 1) ^ {2}}\right) \frac {1}{\mu} + \beta \mu \\ \leq 2 L _ {f} \frac {\Lambda}{k ^ {2}} + \left(\frac {2 \alpha \Lambda}{k ^ {2}}\right) \frac {1}{\mu} + \beta \mu . \end{array}
$$

Therefore, for a given $K > 0$ , it holds that for any $k \geq K$ 

$$
H \left(\mathbf {x} ^ {k}\right) - H _ {\text { opt }} \leq 2 L _ {f} \frac {\Lambda}{K ^ {2}} + \left(\frac {2 \alpha \Lambda}{K ^ {2}}\right) \frac {1}{\mu} + \beta \mu .\tag{10.70}
$$

Minimizing the right-hand side w.r.t. $\mu ,$ we obtain 

$$
\mu = \sqrt {\frac {2 \alpha \Lambda}{\beta}} \frac {1}{K}.\tag{10.71}
$$

Plugging the above expression into (10.70), we conclude that for any $k \geq K$ 2 

$$
H (\mathbf {x} ^ {k}) - H _ {\mathrm{opt}} \leq 2 L _ {f} \frac {\Lambda}{K ^ {2}} + 2 \sqrt {2 \alpha \beta \Lambda} \frac {1}{K}.
$$

Thus, to guarantee that $\mathbf { x } ^ { k }$ is an ε-optimal solution for any $k \geq K$ , it is enough that K will satisfy 

$$
2 L _ {f} \frac {\Lambda}{K ^ {2}} + 2 \sqrt {2 \alpha \beta \Lambda} \frac {1}{K} \leq \varepsilon .
$$

Denoting $\begin{array} { r } { t = \frac { \sqrt { 2 \Lambda } } { K } } \end{array}$ , the above inequality reduces to 

$$
L _ {f} t ^ {2} + 2 \sqrt {\alpha \beta} t - \varepsilon \leq 0,
$$

which, by the fact that $t > 0$ , is equivalent to 

$$
\frac {\sqrt {2 \Lambda}}{K} = t \leq \frac {- \sqrt {\alpha \beta} + \sqrt {\alpha \beta + L _ {f} \varepsilon}}{L _ {f}} = \frac {\varepsilon}{\sqrt {\alpha \beta} + \sqrt {\alpha \beta + L _ {f} \varepsilon}}.
$$

We conclude that K should satisfy 

$$
K \geq \frac {\sqrt {2 \Lambda \alpha \beta} + \sqrt {2 \Lambda \alpha \beta + 2 \Lambda L _ {f} \varepsilon}}{\varepsilon}.
$$

In particular, if we choose 

$$
K = K _ {1} \equiv \frac {\sqrt {2 \Lambda \alpha \beta} + \sqrt {2 \Lambda \alpha \beta + 2 \Lambda L _ {f} \varepsilon}}{\varepsilon}
$$

and $\mu$ according to (10.71), meaning that 

$$
\mu = \sqrt {\frac {2 \alpha \Lambda}{\beta}} \frac {1}{K _ {1}} = \sqrt {\frac {\alpha}{\beta}} \frac {\varepsilon}{\sqrt {\alpha \beta} + \sqrt {\alpha \beta + L _ {f} \varepsilon}},
$$

then for any $k \geq K _ { 1 }$ it holds that $H ( \mathbf { x } ^ { k } ) - H _ { \mathrm { o p t } } \leq \varepsilon$ . By (10.68) and (10.69), 

$$
H (\mathbf {x} _ {\mu} ^ {*}) - \beta \mu \leq H _ {\mu} (\mathbf {x} _ {\mu} ^ {*}) = H _ {\mu , \mathrm{opt}} \leq H _ {\mathrm{opt}} \leq H (\mathbf {x} ^ {0}),
$$

which along with the inequality 

$$
\mu = \sqrt {\frac {\alpha}{\beta}} \frac {\varepsilon}{\sqrt {\alpha \beta} + \sqrt {\alpha \beta + L _ {f} \varepsilon}} \leq \sqrt {\frac {\alpha}{\beta}} \frac {\varepsilon}{\sqrt {\alpha \beta} + \sqrt {\alpha \beta}} \leq \frac {\bar {\varepsilon}}{2 \beta}
$$

implies that $\begin{array} { r } { H ( \mathbf { x } _ { \mu } ^ { * } ) \leq H ( \mathbf { x } ^ { 0 } ) + \frac { \bar { \varepsilon } } { 2 } } \end{array}$ , and hence, by Assumption 10.56(D), it follows that $\| \mathbf { x } _ { \mu } ^ { * } \| \leq R _ { \delta }$ , where $\begin{array} { r } { \delta = H ( \mathbf { x } ^ { 0 } ) + \frac { \bar { \varepsilon } } { 2 } } \end{array}$ . Therefore, $\Lambda = \| \mathbf { x } _ { \mu } ^ { * } - \mathbf { x } ^ { 0 } \| ^ { 2 } \leq ( R _ { \delta } + \| \mathbf { x } ^ { 0 } \| ) ^ { 2 } = \Gamma$ Consequently, 

$$
\begin{array}{r l r} {K _ {1}} & = & {\frac {\sqrt {2 \Lambda \alpha \beta} + \sqrt {2 \Lambda \alpha \beta + 2 \Lambda L _ {f} \varepsilon}}{\varepsilon}} \\ & {\leq} & {\frac {2 \sqrt {2 \Lambda \alpha \beta} + \sqrt {2 \Lambda L _ {f} \varepsilon}}{\varepsilon}} \\ & {\leq} & {\frac {2 \sqrt {2 \Gamma \alpha \beta} + \sqrt {2 \Gamma L _ {f} \varepsilon}}{\varepsilon}} \\ & {\equiv} & {K _ {2},} \end{array}
$$

and hence for any $k \geq K _ { 2 }$ , we have that $H ( \mathbf { x } ^ { k } ) - H _ { \mathrm { o p t } } \leq \varepsilon$ , establishing the desired result. 

Remark 10.58. Note that the smoothing parameter chosen in Theorem 10.57 does not depend on Γ, although the number of iterations required to obtain an ε-optimal solution does depend on Γ. 

Example 10.59. Consider the problem 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \{h (\mathbf {x}): \mathbf {x} \in C \},\tag{10.72}
$$

where $C$ is a nonempty closed and convex set and $h : \mathbb { E }  \mathbb { R }$ is convex function, which is Lipschitz with constant $\ell _ { h }$ . Problem (10.72) fits model (10.64) with $f \equiv 0$ and $g \ = \ \delta _ { C }$ . By Theorem 10.51, for any $\mu > 0$ the Moreau envelope $M _ { h } ^ { \mu }$ is a $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation of h with parameters $\begin{array} { r } { ( \alpha , \beta ) = ( 1 , \frac { \ell _ { h } ^ { 2 } } { 2 } ) } \end{array}$ . In addition, by Theorem 6.60, $\begin{array} { r } { \nabla M _ { h } ^ { \mu } ( { \mathbf x } ) = \frac { 1 } { \mu } ( { \mathbf x } - \mathrm { p r o x } _ { \mu h } ( { \mathbf x } ) ) } \end{array}$ . We will pick $h _ { \mu } = M _ { h } ^ { \mu }$ , and therefore $F _ { \mu } = f + M _ { h } ^ { \mu } = M _ { h } ^ { \mu }$ . By Theorem 10.57, after employing $O ( 1 / \varepsilon )$ iterations of the S-FISTA method with (recalling that $L _ { f } = 0 )$ 

$$
\mu = \sqrt {\frac {\alpha}{\beta}} \frac {\varepsilon}{\sqrt {\alpha \beta} + \sqrt {\alpha \beta + L _ {f} \varepsilon}} = \sqrt {\frac {\alpha}{\beta}} \frac {\varepsilon}{\sqrt {\alpha \beta} + \sqrt {\alpha \beta}} = \frac {\varepsilon}{2 \beta} = \frac {\varepsilon}{\ell_ {h} ^ {2}},
$$

an ε-optimal solution will be achieved. The stepsize is $\textstyle { \frac { 1 } { \tilde { L } } }$ , where $\begin{array} { r } { \tilde { L } = \frac { \alpha } { \mu } = \frac { 1 } { \mu } } \end{array}$ . The main update step of S-FISTA has the following form: 

$$
\begin{array}{l} \mathbf {x} ^ {k + 1} = \operatorname{prox} _ {\frac {1}{L} g} \left(\mathbf {y} ^ {k} - \frac {1}{\tilde {L}} \nabla F _ {\mu} (\mathbf {y} ^ {k})\right) = P _ {C} \left(\mathbf {y} ^ {k} - \frac {1}{\tilde {L} \mu} (\mathbf {y} ^ {k} - \operatorname{prox} _ {\mu h} (\mathbf {y} ^ {k}))\right) \\ = P _ {C} (\operatorname{prox} _ {\mu h} (\mathbf {y} ^ {k})). \end{array}
$$

The S-FISTA method for solving (10.72) is described below. 

S-FISTA for solving (10.72)

Initialization: set $y^{0} = x^{0} \in C, t_{0} = 1, \mu = \frac{\varepsilon}{\ell_{h}^{2}}$ , and $\tilde{L} = \frac{\ell_{h}^{2}}{\varepsilon}$ .

General step: for any $k = 0, 1, 2, \ldots$ execute the following steps:

(a) $x^{k+1} = P_{C}(\text{prox}_{\mu h}(y^{k}))$ ;

(b) $t_{k+1} = \frac{1 + \sqrt{1 + 4t_{k}^{2}}}{2}$ ;

(c) $y^{k+1} = x^{k+1} + \left(\frac{t_{k}-1}{t_{k+1}}\right)(x^{k+1} - x^{k})$ . 

Example 10.60. Consider the problem 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {n}} \left\{\frac {1}{2} \| \mathbf {A x} - \mathbf {b} \| _ {2} ^ {2} + \| \mathbf {D x} \| _ {1} + \lambda \| \mathbf {x} \| _ {1} \right\},
$$

where $\mathbf { A } \in \mathbb { R } ^ { m \times n } , \mathbf { b } \in \mathbb { R } ^ { m } , \mathbf { D } \in \mathbb { R } ^ { p \times n }$ , and $\lambda > 0$ . Problem (P) fits model (10.64) with $\begin{array} { r } { f ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { A x } - \mathbf { b } \| _ { 2 } ^ { 2 } , ~ h ( \mathbf { x } ) = \| \mathbf { D x } \| _ { 1 } , } \end{array}$ , and $g ( \mathbf { x } ) = \lambda \| \mathbf { x } \| _ { 1 }$ . Assumption 10.56 holds: f is convex and $L _ { f } { \mathrm { - s m o o t h } }$ with $L _ { f } = \| \mathbf { A } ^ { T } \mathbf { A } \| _ { 2 , 2 } = \| \mathbf { A } \| _ { 2 , 2 } ^ { 2 } , g$ is proper closed and convex, h is real-valued and convex, and the level sets of the objective function are bounded. To show that h is smoothable, and to find its parameters, note that $h ( { \bf x } ) = q ( { \bf D } { \bf x } )$ , where $q : \mathbb { R } ^ { p }  \mathbb { R }$ is given by $q ( \mathbf { y } ) = \lVert \mathbf { y } \rVert _ { 1 }$ . By Example 10.54, for any $\textstyle \mu > 0 , q _ { \mu } ( \mathbf { y } ) = M _ { q } ^ { \mu } ( \mathbf { y } ) = \sum _ { i = 1 } ^ { p } H _ { \mu } ( y _ { i } )$ is a $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation of $q$ with parameters $( 1 , { \frac { p } { 2 } } )$ . By Theorem $1 0 . 4 6 ( \mathrm { b } ) , q _ { \mu } ( \mathbf { D } \mathbf { x } )$ is a $\textstyle { \frac { 1 } { \mu } }$ -smooth approximation of $h$ with parameters $( \alpha , \beta ) \ : = \ : ( \| \mathbf { D } \| _ { 2 , 2 } ^ { 2 } , \frac { p } { 2 } )$ , and we will set $h _ { \mu } ( { \bf x } ) = M _ { q } ^ { \mu } ( { \bf D } { \bf x } )$ and $F _ { \mu } ( { \bf x } ) = f ( { \bf x } ) + h _ { \mu } ( { \bf x } )$ . Therefore, invoking Theorem 10.57, to obtain an ε-optimal solution of problem (P), we need to employ the S-FISTA method with 

$$
\begin{array}{l} \mu = \sqrt {\frac {\alpha}{\beta}} \frac {\varepsilon}{\sqrt {\alpha \beta} + \sqrt {\alpha \beta + L _ {f} \varepsilon}} \\ = \frac {2 \| \mathbf {D} \| _ {2 , 2}}{\sqrt {p}} \cdot \frac {\varepsilon}{\sqrt {\| \mathbf {D} \| _ {2 , 2} ^ {2} p} + \sqrt {\| \mathbf {D} \| _ {2 , 2} ^ {2} p + 2 \| \mathbf {A} ^ {T} \mathbf {A} \| _ {2 , 2} \varepsilon}}. \end{array}\tag{10.73}
$$

Since $F _ { \mu } ( { \bf x } ) = f ( { \bf x } ) + M _ { q } ^ { \mu } ( { \bf D } { \bf x } )$ , it follows that 

$$
\begin{array}{r l} \nabla F _ {\mu} (\mathbf {x}) = \nabla f (\mathbf {x}) + \mathbf {D} ^ {T} \nabla M _ {q} ^ {\mu} (\mathbf {D x}) \\ = \nabla f (\mathbf {x}) + \frac {1}{\mu} \mathbf {D} ^ {T} (\mathbf {D x} - \mathrm{prox} _ {\mu q} (\mathbf {D x})) & [ \text {Theorem 6.60} ] \\ = \nabla f (\mathbf {x}) + \frac {1}{\mu} \mathbf {D} ^ {T} (\mathbf {D x} - \mathcal {T} _ {\mu} (\mathbf {D x})). & [ \text {Example 6.8} ] \end{array}
$$

Below we write the S-FISTA method for solving problem (P) for a given tolerance parameter $\varepsilon > 0$ 

## S-FISTA for solving (P)

Initialization: set $\mathbf { y } ^ { 0 } \ = \ \mathbf { x } ^ { 0 } \ \in \ \mathbb { R } ^ { n } , t _ { 0 } \ = \ 1$ ; set $\mu$ as in (10.73) and $\tilde { L } \ =$ $\begin{array} { r } { \| \mathbf { A } \| _ { 2 , 2 } ^ { 2 } + \frac { \| \mathbf { D } \| _ { 2 , 2 } ^ { 2 } } { \mu } . } \end{array}$ 

General step: for any $k = 0 , 1 , 2 , . . .$ . execute the following steps: 

$$
\text {(a)} \mathbf {x} ^ {k + 1} = \mathcal {T} _ {\lambda / \tilde {L}} \left(\mathbf {y} ^ {k} - \frac {1}{\tilde {L}} (\mathbf {A} ^ {T} (\mathbf {A y} ^ {k} - \mathbf {b}) + \frac {1}{\mu} \mathbf {D} ^ {T} (\mathbf {D y} ^ {k} - \mathcal {T} _ {\mu} (\mathbf {D y} ^ {k})))\right);
$$

$$
\mathrm{(b)} t _ {k + 1} = \frac {1 + \sqrt {1 + 4 t _ {k} ^ {2}}}{2};
$$

$$
\text {(c)} \mathbf {y} ^ {k + 1} = \mathbf {x} ^ {k + 1} + \left(\frac {t _ {k} - 1}{t _ {k + 1}}\right) (\mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k}).
$$

It is interesting to note that in the case of problem (P) we can actually compute the constant Γ that appears in Theorem 10.57. Indeed, if $H ( \mathbf { x } ) \leq \alpha$ , then 

$$
\lambda \| \mathbf {x} \| _ {1} \leq \frac {1}{2} \| \mathbf {A x} - \mathbf {b} \| _ {2} ^ {2} + \| \mathbf {D x} \| _ {1} + \lambda \| \mathbf {x} \| _ {1} \leq \alpha ,
$$

and since $\| \mathbf { x } \| _ { 2 } \leq \| \mathbf { x } \| _ { 1 }$ , it follows that $R _ { \alpha }$ can be chosen as $\frac { \alpha } { \lambda }$ , from which Γ can be computed. 

## 10.9 Non-Euclidean Proximal Gradient Methods

In this section, and in this section only, the underlying space will not be assumed to be Euclidean. We will consider two diferent approaches for handling this situation. 

The first tackles unconstrained smooth problems through a variation of the gradient method, and the second, which is aimed at solving the composite model, is based on replacing the Euclidean prox operator by a mapping based on the Bregman distance. 

## 10.9.1 The Non-Euclidean Gradient Method

Consider the unconstrained problem 

$$
\min \{f (\mathbf {x}): \mathbf {x} \in \mathbb {E} \},\tag{10.74}
$$

where we assume that $f$ is $L _ { f }$ -smooth w.r.t. the underlying norm. Recall that the gradient method (see Section 10.2) has the form 

$$
\mathbf {x} ^ {k + 1} = \mathbf {x} ^ {k} - t _ {k} \nabla f (\mathbf {x} ^ {k}).\tag{10.75}
$$

As was already discussed in Section 9.1 (in the context of the mirror descent method), this scheme has a “philosophical” flaw since $\mathbf { x } ^ { k } \in \mathbb { E }$ while $\nabla f ( \mathbf { x } ^ { k } ) \in \mathbb { E } ^ { * }$ Obviously, as the only diference between E and $\mathbb { E } ^ { * }$ in this book is their underlying norm, there is no practical problem to invoke the scheme (10.75). Nonetheless, we will change the scheme (10.75) and replace $\nabla f ( \mathbf { x } ^ { k } ) \in \mathbb { E } ^ { * }$ with a “primal counterpart” in E. For any vector $\mathbf { a } \in \mathbb { E } ^ { * }$ , we define the set of primal counterparts of a as 

$$
\Lambda_ {\mathbf {a}} = \operatorname{argmax} _ {\mathbf {v} \in \mathbb {E}} \{\langle \mathbf {a}, \mathbf {v} \rangle : \| \mathbf {v} \| \leq 1 \}.\tag{10.76}
$$

The lemma below presents some elementary properties of $\Lambda _ { \mathbf { a } }$ that follow immediately by its definition and the definition of the dual norm. 

Lemma 10.61 (basic properties of the set of primal counterparts). Let $\mathbf { a } \in \mathbb { E } ^ { * }$ 

(a) If $\mathbf { a } \neq \mathbf { 0 }$ , then $\| \mathbf { a } ^ { \dagger } \| = 1$ for any a<sup>†</sup> ∈ Λ<sub>a</sub>. 

(b) $I f \mathbf { a } = \mathbf { 0 } $ , then $\Lambda _ { \mathbf { a } } = B _ { \parallel \cdot \parallel } [ \mathbf { 0 } , 1 ]$ 

(c) $\left. \mathbf { a } , \mathbf { a } ^ { \dagger } \right. = \left\| \mathbf { a } \right\| _ { * } f o r a n y \mathbf { a } ^ { \dagger } \in \Lambda _ { \mathbf { a } } .$ 

We also note that by the conjugate subgradient theorem (Corollary 4.21), 

$$
\Lambda_ {\mathbf {a}} = \partial h (\mathbf {a}), \mathrm{where} h (\cdot) = \| \cdot \| _ {*}.
$$

Example 10.62. Suppose that $\mathbb { E } = \mathbb { R } ^ { n }$ endowed with the Euclidean $l _ { 2 } \mathrm { - n o r m }$ . In this case, for any a $\neq \mathbf { 0 }$ 2 

$$
\Lambda_ {\mathbf {a}} = \left\{\frac {\mathbf {a}}{\| \mathbf {a} \| _ {2}} \right\}. \quad \blacksquare
$$

Example 10.63. Suppose that $\mathbb { E } = \mathbb { R } ^ { n }$ endowed with the $l _ { 1 } { \mathrm { - n o r m } }$ . In this case, for any $\mathbf { a } \neq \mathbf { 0 }$ , by Example 3.52, 

$$
\Lambda_ {\mathbf {a}} = \partial \| \cdot \| _ {\infty} (\mathbf {a}) = \left\{\sum_ {i \in I (\mathbf {a})} \lambda_ {i} \mathrm{sgn} (a _ {i}) \mathbf {e} _ {i}: \sum_ {i \in I (\mathbf {a})} \lambda_ {i} = 1, \lambda_ {j} \geq 0, j \in I (\mathbf {a}) \right\},
$$

where $I ( \mathbf { a } ) = \mathrm { a r g m a x } _ { i = 1 , 2 , \ldots , n } | a _ { i } | .$ ■ 

Example 10.64. Suppose that $\mathbb { E } = \mathbb { R } ^ { n }$ endowed with the $l _ { \infty } \mathrm { - n o r m }$ . For any $\mathbf { a } \neq \mathbf { 0 }$ 2 $\Lambda _ { \mathbf { a } } = \partial h ( \mathbf { a } )$ , where $h ( \cdot ) = \| \cdot \| _ { 1 }$ . Then, by Example 3.41, 

$$
\Lambda_ {\mathbf {a}} = \left\{\mathbf {z} \in \mathbb {R} ^ {n}: z _ {i} = \operatorname{sgn} (a _ {i}), i \in I _ {\neq} (\mathbf {a}), | z _ {j} | \leq 1, j \in I _ {0} (\mathbf {a}) \right\},
$$

where 

$$
I _ {\neq} (\mathbf {a}) = \{i \in \{1, 2, \dots , n \}: a _ {i} \neq 0 \}, I _ {0} (\mathbf {a}) = \{i \in \{1, 2, \dots , n \}: a _ {i} = 0 \}.
$$

We are now ready to present the non-Euclidean gradient method in which the gradient $\nabla f ( \mathbf { x } ^ { k } )$ is replaced by a primal counterpart $\bar { \nabla } f ( { \mathbf { x } } ^ { k } ) ^ { \dagger } \in \Lambda _ { \nabla f ( { \mathbf { x } } ^ { k } ) }$ 

The Non-Euclidean Gradient Method
Initialization: pick $x^{0} \in E$ arbitrarily.
General step: for any $k = 0, 1, 2, \ldots$ execute the following steps:
(a) pick $\nabla f(\mathbf{x}^{k})^{\dagger} \in \Lambda_{\nabla f(\mathbf{x}^{k})}$ and $L_{k} > 0$ ;
(b) set $x^{k+1} = x^{k} - \frac{\|\nabla f(x^{k})\|_{*}}{L_{k}} \nabla f(x^{k})^{\dagger}$ . 

We begin by establishing a suficient decrease property. The proof is almost identical to the proof of Lemma 10.4. 

Lemma 10.65 (suficient decrease for the non-Euclidean gradient method). Let $f : \mathbb { E } $ R be an $L _ { f }$ -smooth function, and let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the non-Euclidean gradient method. Then for any $k \geq 0$ 

$$
f (\mathbf {x} ^ {k}) - f (\mathbf {x} ^ {k + 1}) \geq \frac {L _ {k} - \frac {L _ {f}}{2}}{L _ {k} ^ {2}} \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} ^ {2}.\tag{10.77}
$$

Proof. By the descent lemma (Lemma 5.7) we have 

$$
\begin{array}{l} f (\mathbf {x} ^ {k + 1}) \leq f (\mathbf {x} ^ {k}) + \langle \nabla f (\mathbf {x} ^ {k}), \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \rangle + \frac {L _ {f}}{2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \| ^ {2} \\ = f (\mathbf {x} ^ {k}) - \frac {\| \nabla f (\mathbf {x} ^ {k}) \| _ {*}}{L _ {k}} \langle \nabla f (\mathbf {x} ^ {k}), \nabla f (\mathbf {x} ^ {k}) ^ {\dagger} \rangle + \frac {L _ {f} \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} ^ {2}}{2 L _ {k} ^ {2}} \\ \stackrel {{(*)}} {{=}} f (\mathbf {x} ^ {k}) - \frac {\| \nabla f (\mathbf {x} ^ {k}) \| _ {*} ^ {2}}{L _ {k}} + \frac {L _ {f} \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} ^ {2}}{2 L _ {k} ^ {2}} \\ = f (\mathbf {x} ^ {k}) - \frac {L _ {k} - \frac {L _ {f}}{2}}{L _ {k} ^ {2}} \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} ^ {2}, \end{array}
$$

where (∗) follows by Lemma 10.61(c). 

Similarly to Section 10.3.3, we will consider both constant and backtracking stepsize strategies. In addition, we will also consider an exact line search procedure. 

• Constant. $\begin{array} { r } { L _ { k } = \bar { L } \in \left( \frac { L _ { f } } { 2 } , \infty \right) } \end{array}$ for all k. 

• Backtracking procedure B4. The procedure requires three parameters $( s , \gamma , \eta )$ , where $s > 0 , \gamma \in ( 0 , 1 )$ , and $\eta > 1$ . The choice of $L _ { k }$ is done as follows: First, $L _ { k }$ is set to be equal to the initial guess s. Then, while 

$$
f (\mathbf {x} ^ {k}) - f \left(\mathbf {x} ^ {k} - \frac {\| \nabla f (\mathbf {x} ^ {k}) \| _ {*}}{L _ {k}} \nabla f (\mathbf {x} ^ {k}) ^ {\dagger}\right) <   \frac {\gamma}{L _ {k}} \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} ^ {2},
$$

we set $L _ { k } : = \eta L _ { k }$ . In other words, $L _ { k }$ is chosen as $L _ { k } = s \eta ^ { i _ { k } }$ , where $i _ { k }$ is the smallest nonnegative integer for which the condition 

$$
f (\mathbf {x} ^ {k}) - f \left(\mathbf {x} ^ {k} - \frac {\| \nabla f (\mathbf {x} ^ {k}) \| _ {*}}{s \eta^ {i _ {k}}} \nabla f (\mathbf {x} ^ {k}) ^ {\dagger}\right) \geq \frac {\gamma}{s \eta^ {i _ {k}}} \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} ^ {2}
$$

is satisfied. 

• Exact line search. $L _ { k }$ is chosen as 

$$
L _ {k} \in \operatorname{argmin} _ {L > 0} f \left(\mathbf {x} ^ {k} - \frac {\| \nabla f (\mathbf {x} ^ {k}) \| _ {*}}{L} \nabla f (\mathbf {x} ^ {k}) ^ {\dagger}\right).
$$

By the same arguments given in Remark 10.13, it follows that if the backtracking procedure B4 is used, then 

$$
L _ {k} \leq \max \left\{s, \frac {\eta L _ {f}}{2 (1 - \gamma)} \right\}.\tag{10.78}
$$

## Convergence Analysis in the Nonconvex Case

The statements and proofs of the next two results (Lemma 10.66 and Theorem 10.67) are similar those of Lemma 10.14 and Theorem 10.15. 

Lemma 10.66 (suficient decrease of the non-Euclidean gradient method). Let f be an $L _ { f } { - } s m o o t h$ function. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the non-Euclidean gradient method for solving problem (10.74) with either a constant stepsize corresponding to $\begin{array} { r } { L _ { k } = \bar { L } \in \left( \frac { L _ { f } } { 2 } , \infty \right) } \end{array}$ ; a stepsize chosen by the backtracking procedure B4 with parameters $( s , \gamma , \eta )$ satisfying $s > 0 , \gamma \in ( 0 , 1 ) , \eta > 1 _ { \scriptscriptstyle }$ ; or an exact line search for computing the stepsize. Then for any $k \geq 0$ 2 

$$
f (\mathbf {x} ^ {k}) - f (\mathbf {x} ^ {k + 1}) \geq M \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} ^ {2},\tag{10.79}
$$

where 

$$
M = \left\{ \begin{array}{l l} \frac {\bar {L} - \frac {L _ {f}}{2}}{\big (\bar {L} \big) ^ {2}}, & \text {constant stepsize}, \\ \frac {\gamma}{\max \big \{s , \frac {\eta L _ {f}}{2 (1 - \gamma)} \big \}}, & \text {backtracking}, \\ \frac {1}{2 L _ {f}}, & \text {exact line search}. \end{array} \right.\tag{10.80}
$$

Proof. The result for the constant stepsize setting follows by plugging $L _ { k } = \bar { L }$ in (10.77). If $L _ { k }$ is chosen by the exact line search procedure, then, in particular, $f ( \mathbf { x } ^ { k + 1 } ) \leq f ( \tilde { \mathbf { x } } ^ { k } )$ , where $\begin{array} { r } { \tilde { { \mathbf { x } } } ^ { k } = { \mathbf { x } } ^ { k } - \frac { \| \nabla f ( { \mathbf { x } } ^ { k } ) \| _ { * } } { L _ { f } } \nabla f ( { \mathbf { x } } ^ { k } ) ^ { \dagger } } \end{array}$ , and hence 

$$
f (\mathbf {x} ^ {k}) - f (\mathbf {x} ^ {k + 1}) \geq f (\mathbf {x} ^ {k}) - f (\tilde {\mathbf {x}} ^ {k}) \geq \frac {1}{2 L _ {f}} \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} ^ {2},
$$

where we used the result already established for the constant stepsize in the second inequality. $\mathrm { A s }$ for the backtracking procedure, by its definition and the upper bound (10.78) on $L _ { k }$ we have 

$$
f (\mathbf {x} ^ {k}) - f (\mathbf {x} ^ {k + 1}) \geq \frac {\gamma}{L _ {k}} \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} ^ {2} \geq \frac {\gamma}{\max \left\{s , \frac {\eta L _ {f}}{2 (1 - \gamma)} \right\}} \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} ^ {2}.
$$

Theorem 10.67 (convergence of the non-Euclidean gradient method— nonconvex case). Suppose that f is an $L _ { f }$ -smooth function. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the non-Euclidean gradient method for solving the problem 

$$
\min _ {\mathbf {x} \in \mathbb {E}} f (\mathbf {x})\tag{10.81}
$$

with either a constant stepsize corresponding to $\begin{array} { r } { L _ { k } \ = \ \bar { L } \ \in \ \left( \frac { L _ { f } } { 2 } , \infty \right) } \end{array}$ ; a stepsize chosen by the backtracking procedure B4 with parameters $( s , \gamma , \eta )$ satisfying $s \mathrm { ~ } >$ $0 , \gamma \in ( 0 , 1 ) , \eta > 1$ ; or an exact line search for computing the stepsize. Then 

(a) the sequence $\{ f ( \mathbf { x } ^ { k } ) \} _ { k \geq 0 }$ is nonincreasing; in addition, $f ( \mathbf { x } ^ { k + 1 } ) < f ( \mathbf { x } ^ { k } )$ if and only if $\nabla f ( \mathbf { x } ^ { k } ) \neq \mathbf { 0 }$ ; 

(b) if the sequence $\{ f ( \mathbf { x } ^ { k } ) \} _ { k \geq 0 }$ is bounded below, then $\nabla f ( { \bf x } ^ { k } )  { \bf 0 } \ a s \ k  \infty$ 

(c) if the optimal value of (10.81) is finite and equal to $f _ { \mathrm { o p t } }$ , then 

$$
\min _ {n = 0, 1, \dots , k} \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} \leq \frac {\sqrt {f (\mathbf {x} ^ {0}) - f _ {\mathrm{opt}}}}{\sqrt {M (k + 1)}},\tag{10.82}
$$

where M is given in (10.80); 

(d) all limit points of the sequence $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ are stationary points of problem (10.81). 

Proof. (a) By Lemma 10.66, 

$$
f (\mathbf {x} ^ {k}) - f (\mathbf {x} ^ {k + 1}) \geq M \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} ^ {2},\tag{10.83}
$$

where $M > 0$ is given in (10.80). The inequality (10.83) readily implies that $f ( \mathbf { x } ^ { k } ) \geq$ $f ( \mathbf { x } ^ { k + 1 } )$ and that if $\nabla f ( \mathbf { x } ^ { k } ) \neq \mathbf { 0 }$ , then $\bar { f } ( \mathbf { x } ^ { k + 1 } ) < f ( \mathbf { x } ^ { k } )$ . Finally, if $\nabla f ( \mathbf { x } ^ { k } ) = \mathbf { 0 }$ then $\mathbf { x } ^ { \acute { k } } = \mathbf { x } ^ { k + 1 }$ , and hence $f ( \mathbf { x } ^ { k } ) = f ( \mathbf { x } ^ { k + 1 } )$ 

(b) Since the sequence $\{ f ( \mathbf { x } ^ { k } ) \} _ { k \geq 0 }$ is nonincreasing and bounded below, it converges. Thus, in particular $f ( \mathbf { x } ^ { k } ) { \overset { - } { - } } f ( \mathbf { x } ^ { k + 1 } )  0$ as $k  \infty$ , which, combined with (10.83), implies that $\nabla f ( \mathbf { x } ^ { k } )  \mathbf { 0 }$ as $k \to \infty$ 

(c) By Lemma 10.66, for any $n \geq 0$ 

$$
f (\mathbf {x} ^ {n}) - f (\mathbf {x} ^ {n + 1}) \geq M \| \nabla f (\mathbf {x} ^ {n}) \| _ {*} ^ {2}.
$$

Summing the above over $n = 0 , 1 , \ldots , k$ , we obtain 

$$
f (\mathbf {x} ^ {0}) - f (\mathbf {x} ^ {k + 1}) \geq M \sum_ {n = 0} ^ {k} \| \nabla f (\mathbf {x} ^ {n}) \| _ {*} ^ {2} \geq (k + 1) M \min _ {n = 0, 1, \dots , k} \| \nabla f (\mathbf {x} ^ {n}) \| _ {*} ^ {2}.
$$

Using the fact that $f ( \mathbf { x } ^ { k + 1 } ) \geq f _ { \mathrm { o p t } }$ , the inequality (10.82) follows. 

(d) Let x¯ be a limit point of $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ . Then there exists a subsequence $\{ \mathbf { x } ^ { k _ { j } } \} _ { j \ge 0 }$ converging to x¯. For any $j \geq 0$ 

$$
\| \nabla f (\bar {\mathbf {x}}) \| _ {*} \leq \| \nabla f (\mathbf {x} ^ {k _ {j}}) - \nabla f (\bar {\mathbf {x}}) \| _ {*} + \| \nabla f (\mathbf {x} ^ {k _ {j}}) \| _ {*} \leq L _ {f} \| \mathbf {x} ^ {k _ {j}} - \bar {\mathbf {x}} \| + \| \nabla f (\mathbf {x} ^ {k _ {j}}) \| _ {*}.\tag{10.84}
$$

Since the right-hand side of (10.84) goes to 0 as $j  \infty$ , it follows that $\nabla f ( { \bar { \mathbf { x } } } )$ ${ \bf \mu } = { \bf 0 }$ □ 

## Convergence Analysis in the Convex Case

To establish a rate of convergence in the case where $f$ is convex, we will require an additional boundedness-type assumption. We gather all the required assumptions in the following. 

Assumption 10.68. 

(A) $f : \mathbb { E } \to \mathbb { R }$ is $L _ { f }$ -smooth and convex. 

(B) The optimal set of the problem 

$$
\min _ {\mathbf {x} \in \mathbb {E}} f (\mathbf {x})
$$

is nonempty and denoted by $X ^ { \ast }$ . The optimal value is denoted by $f _ { \mathrm { o p t } }$ 

(C) For any $\alpha > 0$ , there exists $R _ { \alpha } > 0$ such that 

$$
\max _ {\mathbf {x}, \mathbf {x} ^ {*}} \{\| \mathbf {x} ^ {*} - \mathbf {x} \|: f (\mathbf {x}) \leq \alpha , \mathbf {x} ^ {*} \in X ^ {*} \} \leq R _ {\alpha}.
$$

The proof of the convergence rate is based on the following very simple lemma. 

Lemma 10.69. Suppose that Assumption 10.68 holds. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the non-Euclidean gradient method for solving the problem of minimizing f over E with either a constant stepsize corresponding to $\bar { L _ { k } } = \bar { L } \in \left( \frac { L _ { f } } { 2 } , \infty \right) ; a$ stepsize chosen by the backtracking procedure B4 with parameters $( s , \gamma , \eta )$ satisfying $s > 0 , \gamma \in ( 0 , 1 ) , \eta > 1$ ; or an exact line search for computing the stepsize. Then 

$$
f (\mathbf {x} ^ {k}) - f (\mathbf {x} ^ {k + 1}) \geq \frac {1}{C} (f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}) ^ {2},\tag{10.85}
$$

where 

$$
C = \left\{ \begin{array}{l l} \frac {R _ {\alpha} ^ {2} \bar {L} ^ {2}}{\bar {L} - \frac {L _ {f}}{2}}, & \text {constant stepsize}, \\ \frac {R _ {\alpha} ^ {2}}{\gamma} \max \left\{s, \frac {\eta L _ {f}}{2 (1 - \gamma)} \right\}, & \text {backtracking}, \\ 2 R _ {\alpha} ^ {2} L _ {f}, & \text {exact line search}, \end{array} \right.\tag{10.86}
$$

with $\alpha = f ( \mathbf { x } ^ { 0 } )$ . 

Proof. Note that, by Theorem $1 0 . 6 7 ( \mathrm { a } ) , \{ f ( \mathbf { x } ^ { k } ) \} _ { k \geq 0 }$ is nonincreasing, and in particular for any $k \geq 0$ it holds that $f ( \mathbf { x } ^ { k } ) \leq f ( \mathbf { x } ^ { 0 } )$ . Therefore, for any $\mathbf { x } ^ { * } \in X ^ { * }$ and $k \geq 0$ ， 

$$
\left\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \right\| \leq R _ {\alpha},
$$

where $\alpha = f ( \mathbf { x } ^ { 0 } )$ . To prove (10.85), we note that on the one hand, by Lemma 10.66, 

$$
f (\mathbf {x} ^ {k}) - f (\mathbf {x} ^ {k + 1}) \geq M \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} ^ {2},\tag{10.87}
$$

where M is given in (10.80). On the other hand, by the gradient inequality along with the generalized Cauchy–Schwarz inequality (Lemma 1.4), for any $\mathbf { x } ^ { * } \in X ^ { * }$ , 

$$
\begin{array}{r l} & f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}} = f (\mathbf {x} ^ {k}) - f (\mathbf {x} ^ {*}) \\ & \qquad \leq \langle \nabla f (\mathbf {x} ^ {k}), \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \rangle \\ & \qquad \leq \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| \\ & \qquad \leq R _ {\alpha} \| \nabla f (\mathbf {x} ^ {k}) \| _ {*}. \end{array}\tag{10.88}
$$

Combining (10.87) and (10.88), we obtain that 

$$
f (\mathbf {x} ^ {k}) - f (\mathbf {x} ^ {k + 1}) \geq M \| \nabla f (\mathbf {x} ^ {k}) \| _ {*} ^ {2} \geq \frac {M}{R _ {\alpha} ^ {2}} (f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}}) ^ {2}.
$$

Plugging the expression for M given in (10.80) into the above inequality, the result (10.85) is established. 

To derive the rate of convergence in function values, we will use the following lemma on convergence of nonnegative scalar sequences. 

Lemma 10.70. Let $\{ a _ { k } \} _ { k \ge 0 }$ be a sequence of nonnegative real numbers satisfying for any $k \geq 0$ 

$$
a _ {k} - a _ {k + 1} \geq \frac {1}{\gamma} a _ {k} ^ {2}
$$

for some $\gamma > 0$ . Then for any $k \geq 1$ 

$$
a _ {k} \leq \frac {\gamma}{k}.\tag{10.89}
$$

Proof. Let k be a positive integer. If $a _ { k } = 0$ , then obviously (10.89) holds. Suppose that $a _ { k } > 0$ . Then by the monotonicity of $\{ a _ { n } \} _ { n \geq 0 }$ , we have that $a _ { 0 } , a _ { 1 } , \dots , a _ { k } > 0$ For any $n = 1 , 2 , \ldots , k$ 2 

$$
\frac {1}{a _ {n}} - \frac {1}{a _ {n - 1}} = \frac {a _ {n - 1} - a _ {n}}{a _ {n - 1} a _ {n}} \geq \frac {1}{\gamma} \frac {a _ {n - 1} ^ {2}}{a _ {n - 1} a _ {n}} = \frac {1}{\gamma} \frac {a _ {n - 1}}{a _ {n}} \geq \frac {1}{\gamma},\tag{10.90}
$$

where the last inequality follows from the monotonicity of the sequence. Summing (10.90) over $n = 1 , 2 , \ldots , k$ , we obtain 

$$
\frac {1}{a _ {k}} \geq \frac {1}{a _ {0}} + \frac {k}{\gamma} \geq \frac {k}{\gamma},
$$

proving (10.89). 

Combining Lemmas 10.69 and 10.70, we can establish an $O ( 1 / k )$ rate of convergence in function values of the sequence generated by the non-Euclidean gradient method. 

Theorem 10.71 $( O ( 1 / k )$ rate of convergence of the non-Euclidean gradient method). Under the setting of Lemma 10.69, for any $k \geq 1$ 

$$
f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}} \leq \frac {C}{k},\tag{10.91}
$$

where $C$ is given in (10.86). 

Proof. By Lemma 10.69, 

$$
a _ {k} - a _ {k + 1} \geq \frac {1}{C} a _ {k} ^ {2},
$$

where $a _ { k } = f ( \mathbf { x } ^ { k } ) - f _ { \mathrm { o p t } }$ . Invoking Lemma 10.70 with $\gamma = C$ , the inequality $\begin{array} { r } { a _ { k } \le \frac { C } { k } } \end{array}$ 2 which is the same as (10.91), follows. 

Remark 10.72. When a constant stepsize $\frac { 1 } { L _ { f } }$ is used (meaning that $L _ { k } \equiv \bar { L } \equiv L _ { f } )$ 2 (10.91) has the form 

$$
f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}} \leq \frac {2 R _ {\alpha} ^ {2} L _ {f}}{k},
$$

which is similar in form to the result in the Euclidean setting in which the following bound was derived (see Theorem 10.21): 

$$
f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}} \leq \frac {L _ {f} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| ^ {2}}{2 k}.
$$

## The Non-Euclidean Gradient Method in $\mathbb { R } ^ { n }$ Endowed with the $l _ { 1 } \mathbf { - N o r m }$

Example 10.73. Suppose that the underlying space is $\mathbb { R } ^ { n }$ endowed with the $l _ { 1 } -$ norm, and let $f$ be an $L _ { f }$ -smooth function w.r.t. the $l _ { 1 } { \mathrm { - n o r m } }$ . Recall (see Example 10.63) that the set of primal counterparts in this case is given for any $\mathbf { a } \neq \mathbf { 0 }$ by 

$$
\Lambda_ {\mathbf {a}} = \left\{\sum_ {i \in I (\mathbf {a})} \lambda_ {i} \mathrm{sgn} (a _ {i}) \mathbf {e} _ {i}: \sum_ {i \in I (\mathbf {a})} \lambda_ {i} = 1, \lambda_ {j} \geq 0, j \in I (\mathbf {a}) \right\},
$$

where $I ( \mathbf { a } ) = \operatorname { a r g m a x } _ { i = 1 , 2 , \ldots , n } | a _ { i } |$ . When employing the method, we can always choose $\mathbf { a } ^ { \dagger } = \mathrm { s g n } ( a _ { i } ) \mathbf { e } _ { i }$ for some arbitrary $i \in I ( \mathbf { a } )$ . The method thus takes the following form: 

Non-Euclidean Gradient under the $l_{1}$ -Norm
- Initialization: pick $\mathbf{x}^0 \in \mathbb{R}^n$ .
- General step: for any $k = 0, 1, 2, \ldots$ execute the following steps:
- pick $i_k \in \operatorname{argmax}_i \left| \frac{\partial f(\mathbf{x}^k)}{\partial x_i} \right|$ ;
- set $\mathbf{x}^{k+1} = \mathbf{x}^k - \frac{\|\nabla f(\mathbf{x}^k)\|_\infty}{L_k} \text{sgn}\left(\frac{\partial f(\mathbf{x}^k)}{\partial x_{i_k}}\right) \mathbf{e}_{i_k}$ . 

The constants $L _ { k }$ can be chosen by either one of the three options: a constant stepsize rule $\begin{array} { r } { L _ { k } \equiv \bar { L } \in \left( \frac { L _ { f } } { 2 } , \infty \right) } \end{array}$ , the backtracking procedure B4, or an exact line  search. Note that at each iteration only one coordinate is altered. This is a variant of a coordinate descent method that actually has an interpretation as a non-Euclidean gradient method. 

Example 10.74. Consider the problem 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {n}} \left\{\frac {1}{2} \mathbf {x} ^ {T} \mathbf {A x} + \mathbf {b} ^ {T} \mathbf {x} \right\},
$$

where $\mathbf { A } \in \mathbb { S } _ { + + } ^ { n }$ and $\mathbf { b } \in \mathbb { R } ^ { n }$ . The underlying space is $\mathbb { E } = \mathbb { R } ^ { n }$ endowed with the $l _ { p ^ { - } } \mathrm { n o r m } \ ( p \in [ 1 , \infty ] )$ . By Example 5.2, f is ${ \cal L } _ { f } ^ { ( p ) }$ -smooth with 

$$
L _ {f} ^ {(p)} = \| \mathbf {A} \| _ {p, q} = \max _ {\mathbf {x}} \{\| \mathbf {A x} \| _ {q}: \| \mathbf {x} \| _ {p} \leq 1 \}
$$

with $q \in [ 1 , \infty ]$ satisfying $\textstyle { \frac { 1 } { p } } + { \frac { 1 } { q } } = 1$ . Two examples of smoothness parameters are the following: 

$p = 2$ . In this case, since A is positive definite, $L _ { f } ^ { ( 2 ) } = \| \mathbf { A } \| _ { 2 , 2 } = \lambda _ { \operatorname* { m a x } } ( \mathbf { A } )$ 

• p = 1. Here $L _ { f } ^ { ( 1 ) } = \| \mathbf { A } \| _ { 1 , \infty } = \operatorname* { m a x } _ { i , j } | A _ { i , j } | .$ 

The non-Euclidean gradient method for $p = 2$ is actually the Euclidean gradient method; taking a constant stepsize corresponding to $L _ { k } = L _ { f } ^ { ( 2 ) } = \lambda _ { \operatorname* { m a x } } ( \mathbf { A } )$ , the method takes the following form: 

Algorithm G2
- Initialization: pick $\mathbf{x}^0 \in \mathbb{R}^n$ .
- General step ( $k \geq 0$ ): $\mathbf{x}^{k+1} = \mathbf{x}^k - \frac{1}{L_f^{(2)}} (\mathbf{A}\mathbf{x}^k + \mathbf{b})$ . 

In the case $p = 1$ the method is a coordinate descent-type method, and with a constant stepsize corresponding to $L _ { k } = L _ { f } ^ { ( 1 ) } = \operatorname* { m a x } _ { i , j } | A _ { i , j } |$ it takes the following form: 

## Algorithm G1

• Initialization: pick $\mathbf { x } ^ { 0 } \in \mathbb { R } ^ { n }$ 

• General step $( k \geq 0 )$ 

– pick $\begin{array} { r } { i _ { k } \in \mathrm { a r g m a x } _ { i = 1 , 2 , \ldots , n } | \mathbf { A } _ { i } \mathbf { x } ^ { k } + b _ { i } | _ { \mathrm { ; } } } \end{array}$ , where $\mathbf { A } _ { i }$ denotes the ith row of A. 

$$
\mathrm{update} \mathbf {x} _ {j} ^ {k + 1} = \left\{ \begin{array}{l l} \mathbf {x} _ {j} ^ {k}, & j \neq i _ {k}, \\ \mathbf {x} _ {i _ {k}} ^ {k} - \frac {1}{L _ {f} ^ {(1)}} (\mathbf {A} _ {i _ {k}} \mathbf {x} ^ {k} + b _ {i _ {k}}), & j = i _ {k}. \end{array} \right.
$$

By Theorem 10.71,<sup>62</sup> 

$$
f (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}} \leq \frac {2 L _ {f} ^ {(p)} R _ {f (\mathbf {x} ^ {0})} ^ {2}}{k}.
$$

Therefore, the ratio $\frac { L _ { f } ^ { ( 2 ) } } { L _ { f } ^ { ( 1 ) } }$ might indicate which of the methods should have an advantage over the other. 

Remark 10.75. Note that Algorithm G2 (from Example $1 0 . 7 4 )$ requires $O ( n ^ { 2 } )$ operations at each iteration since the matrix/vector multiplication $\mathbf { A x } ^ { k }$ is computed. On the other hand, a careful implementation of Algorithm G1 will only require $O ( n )$ operations at each iteration; this can be accomplished by updating the gradient $\mathbf { g } ^ { k } \equiv \mathbf { A } \mathbf { x } ^ { k } +$ b using the relation $\begin{array} { r } { \mathbf { g } ^ { k + 1 } = \mathbf { g } ^ { k } - \frac { \mathbf { A } _ { i _ { k } } \mathbf { x } ^ { k } + b _ { i _ { k } } } { L _ { f } ^ { ( 1 ) } } \mathbf { A } \mathbf { e } _ { i _ { k } } \ \left( \mathbf { A } \mathbf { e } _ { i _ { k } } \right. } \end{array}$ is obviously the $i _ { k } t h$ column of A). Therefore, a fair comparison between Algorithms G1 and G2 will count each n iterations of algorithm G1 as “one iteration.” We will call such an iteration a “meta-iteration.” 

Example 10.76. Continuing Example 10.74, consider, for example, the matrix $\mathbf { A } = \bar { \mathbf { A } ( \mathfrak { d } ) } \equiv \mathbf { J } + d \mathbf { I }$ , where the matrix J is the matrix of all ones. Then for any $d > 0$ $\mathbf { A } ^ { ( d ) }$ is positive definite and $\begin{array} { r } { \lambda _ { \operatorname* { m a x } } ( \mathbf { A } ^ { ( d ) } ) = d + n , \operatorname* { m a x } _ { i , j } | A _ { i , j } ^ { ( d ) } | = d + 1 } \end{array}$ . Therefore, as the ratio $\begin{array} { r } { \rho _ { f } \equiv \frac { L _ { f } ^ { ( 2 ) } } { L _ { f } ^ { ( 1 ) } } = \frac { d + n } { d + 1 } } \end{array}$ gets larger, the Euclidean gradient method (Algorithm G2) should become more inferior to the non-Euclidean version (Algorithm G1). 

We ran the two algorithms for the choice $\mathbf { A } = \mathbf { A } ^ { ( 2 ) }$ and $\mathbf { b } = 1 0 \mathbf { e } _ { 1 }$ with initial point $ { \mathbf { x } } ^ { 0 } = \mathbf { e } _ { n }$ . The values $f ( \mathbf { x } ^ { k } ) - f _ { \mathrm { o p t } }$ as a function of the iteration index k are plotted in Figures 10.4 and 10.5 for $n = 1 0$ and $n = 1 0 0$ , respectively. As can be seen in the left images of both figures, when meta-iterations of algorithm G1 are compared with iterations of algorithm G2, the superiority of algorithm G1 is significant. We also made the comparison when each iteration of algorithm G1 is just an update of one coordinate, meaning that we do not consider meta-iterations. For $n = 1 0$ , the methods behave similarly, and there does not seem to be any preference to G1 or G2. However, when $n = 1 0 0$ , there is still a substantial advantage of algorithm G1 compared to G2, despite the fact that it is a much cheaper method w.r.t. the number of operations performed per iteration. A possible reason for this is the fact that for $\begin{array} { r } { n = 1 0 , \rho _ { f } = \frac { 2 + 1 0 } { 2 + 1 } = 4 } \end{array}$ , while for $\textstyle n = 1 0 0 , { \frac { 2 + 1 0 0 } { 2 + 1 } } = 3 4$ , and hence it is expected that the advantage of algorithm G1 over algorithm G2 will be more substantial when n = 100. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/fe30c2c4b39d8204fd08f4d9ba67e71f43f0db76daad9f002ee9673b99fbb332.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/a695bb5ae834f9dee480e20db32880e47b1030f664af0b72e48f6036a741fe6b.jpg)



Figure 10.4. Comparison of the Euclidean gradient method (G2) with the non-Euclidean gradient method (G1) applied on the problem from Example 10.76 with $n = 1 0$ . The left image considers “meta-iterations” of G1, meaning that 10 iterations of G1 are counted as one iteration, while the right image counts each coordinate update as one iteration.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/1bee645ab393f69ccc691a14b0334f2b834ecf64eb5c89fd7b387779d379bd77.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/25e578b9092f6a23f68d098a7ce8950c8dbc7825a78cc96a23d1b1372b513b57.jpg)



Figure 10.5. Comparison of the Euclidean gradient method (G2) with the non-Euclidean gradient method (G1) applied on the problem from Example 10.76 with $n = 1 0 0$ . The left image considers “meta-iterations” of G1, meaning that 100 iterations of G1 are counted as one iteration, while the right image counts each coordinate update as one iteration.


## 10.9.2 The Non-Euclidean Proximal Gradient Method<sup>63</sup>

In this section we return to the composite model 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \{F (\mathbf {x}) \equiv f (\mathbf {x}) + g (\mathbf {x}) \},\tag{10.92}
$$

where the endowed norm on E is not assume to be Euclidean. Our main objective will be to develop a non-Euclidean version of the proximal gradient method. We note that when $g \equiv 0$ , the method will not coincide with the non-Euclidean gradient method discussed in Section 10.9.1, meaning that the approach described here, which is similar to the generalization of projected subgradient to mirror descent (see Chapter 9), is fundamentally diferent than the approach considered in the non-Euclidean gradient method. We will make the following assumption. 

## Assumption 10.77.

(A) $g : \mathbb { E }  ( - \infty , \infty ]$ is proper closed and convex. 

(B) $f : \mathbb { E } \to ( - \infty , \infty ]$ is proper closed and convex; dom $( g ) \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ and $f$ is $L _ { f } { - } s m o o t h$ over int $\left( \operatorname { d o m } ( f ) \right)$ 

(C) The optimal solution of problem (10.1) is nonempty and denoted by $X ^ { * }$ . The optimal value of the problem is denoted by $F _ { \mathrm { o p t } }$ 

In the Euclidean setting, the general update rule of the proximal gradient method (see the discussion in Section 10.2) can be written in the following form: 

$$
\mathbf {x} ^ {k + 1} = \operatorname{argmin} _ {\mathbf {x} \in \mathbb {E}} \left\{f (\mathbf {x} ^ {k}) + \langle \nabla f (\mathbf {x} ^ {k}), \mathbf {x} - \mathbf {x} ^ {k} \rangle + g (\mathbf {x}) + \frac {L _ {k}}{2} \| \mathbf {x} - \mathbf {x} ^ {k} \| ^ {2} \right\}.
$$

We will use the same idea as in the mirror descent method and replace the half squared Euclidean distance with a Bregman distance, leading to the following update rule: 

$$
\mathbf {x} ^ {k + 1} = \operatorname * {a r g m i n} _ {\mathbf {x} \in \mathbb {E}} \left\{f (\mathbf {x} ^ {k}) + \langle \nabla f (\mathbf {x} ^ {k}), \mathbf {x} - \mathbf {x} ^ {k} \rangle + g (\mathbf {x}) + L _ {k} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {k}) \right\},
$$

where $B _ { \omega }$ is the Bregman distance associated with $\omega$ (see Definition 9.2). The function $\omega$ will satisfy the following properties. 

Assumption 10.78 (properties of ω). 

• ω is proper closed and convex. 

• ω is diferentiable over dom $( \partial \omega )$ 

$\operatorname { d o m } ( g ) \subseteq \operatorname { d o m } ( \omega )$ 

$\omega + \delta _ { \mathrm { d o m } ( g ) }$ is 1-strongly convex. 

The proximal gradient method is defined below. 

The Non-Euclidean Proximal Gradient Method
Initialization: pick $\mathbf{x}^{0}\in\mathrm{dom}(g)\cap\mathrm{dom}(\partial\omega)$ .
General step: for any $k=0,1,2,\ldots$ execute the following steps:
(a) pick $L_{k}>0;$ (b) compute $\mathbf{x}^{k+1}=\operatorname{argmin}_{\mathbf{x}\in\mathbb{E}}\left\{\left\langle\frac{1}{L_{k}}\nabla f(\mathbf{x}^{k})-\nabla\omega(\mathbf{x}^{k}),\mathbf{x}\right\rangle+\frac{1}{L_{k}}g(\mathbf{x})+\omega(\mathbf{x})\right\}.$ (10.93) 

Our first observation is that under Assumptions $1 0 . 7 7$ and 10.78, the non-Euclidean proximal gradient method is well defined, meaning that if $\mathbf { x } ^ { k } \in \operatorname { d o m } ( g )$ ∩ dom(∂ω), then the minimization problem in (10.93) has a unique optimal solution in dom $( g ) \cap$ dom(∂ω). This is a direct result of Lemma 9.7 invoked with $\begin{array} { r } { \psi ( \mathbf { x } ) = \Big \langle \frac { 1 } { L _ { k } } \nabla f ( \mathbf { x } ^ { k } ) - } \end{array}$ $\begin{array} { r } { \nabla \omega ( \mathbf { x } ^ { k } ) , \mathbf { x } \rangle + \frac { 1 } { L _ { k } } g ( \mathbf { x } ) } \end{array}$ . The two stepsize rules that will be analyzed are detailed below. We use the notation 

$$
V _ {L} (\bar {\mathbf {x}}) \equiv \operatorname * {a r g m i n} _ {\mathbf {x} \in \mathbb {E}} \left\{\left\langle \frac {1}{L} \nabla f (\bar {\mathbf {x}}) - \nabla \omega (\bar {\mathbf {x}}), \mathbf {x} \right\rangle + \frac {1}{L} g (\mathbf {x}) + \omega (\mathbf {x}) \right\}.
$$

• Constant. $L _ { k } = \bar { L } = L _ { f }$ for all k. 

• Backtracking procedure B5. The procedure requires two parameters $( s , \eta )$ , where $s > 0$ and $\eta > 1$ . Define $L _ { - 1 } = s$ . At iteration k $( k \geq 0 )$ the choice of $L _ { k }$ is done as follows: First, $L _ { k }$ is set to be equal to $L _ { k - 1 }$ Then, while 

$$
f (V _ {L _ {k}} (\mathbf {x} ^ {k})) > f (\mathbf {x} ^ {k}) + \langle \nabla f (\mathbf {x} ^ {k}), V _ {L _ {k}} (\mathbf {x} ^ {k}) - \mathbf {x} ^ {k} \rangle + \frac {L _ {k}}{2} \| V _ {L _ {k}} (\mathbf {x} ^ {k}) - \mathbf {x} ^ {k} \| ^ {2},
$$

set $L _ { k } : = \eta L _ { k }$ . In other words, the stepsize is chosen as $L _ { k } = L _ { k - 1 } \eta ^ { i _ { k } }$ 2 where $i _ { k }$ is the smallest nonnegative integer for which the condition 

$$
\begin{array}{r} f (V _ {L _ {k - 1} \eta^ {i _ {k}}} (\mathbf {x} ^ {k})) \leq f (\mathbf {x} ^ {k}) + \langle \nabla f (\mathbf {x} ^ {k}), V _ {L _ {k - 1} \eta^ {i _ {k}}} (\mathbf {x} ^ {k}) - \mathbf {x} ^ {k} \rangle \\ + \frac {L _ {k}}{2} \| V _ {L _ {k - 1} \eta^ {i _ {k}}} (\mathbf {x} ^ {k}) - \mathbf {x} ^ {k} \| ^ {2} \end{array}
$$

is satisfied. 

Remark 10.79. In both stepsize rules the following inequality holds: 

$$
f (\mathbf {x} ^ {k + 1}) \leq f (\mathbf {x} ^ {k}) + \langle \nabla f (\mathbf {x} ^ {k}), \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \rangle + \frac {L _ {k}}{2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \| ^ {2}.
$$

Remark 10.80. By the same arguments as in Remark 10.19 we have that $L _ { k } \ \leq$ $\alpha L _ { f }$ , where $\alpha = 1 f o r$ the constant stepsize case and $\begin{array} { r } { \alpha = \operatorname* { m a x } \left\{ \eta , \frac { s } { L _ { f } } \right\} } \end{array}$ in the setting of the backtracking procedure B5. 

The rate of convergence result will now be stated and proved. 

Theorem 10.81 $( O ( 1 / k )$ rate of convergence of the non-Euclidean proximal gradient method). Suppose that Assumptions 10.77 and 10.78 hold. Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the non-Euclidean proximal gradient method for solving problem (10.92) with either a constant stepsize rule in which $L _ { k } \equiv L _ { f }$ for all $k \geq 0$ or the backtracking procedure B5. Then 

(a) the sequence $\{ F ( \mathbf { x } ^ { k } ) \} _ { k \geq 0 }$ is nonincreasing; 

(b) for any $k \geq 1$ and $\mathbf { x } ^ { * } \in X ^ { * }$ 

$$
F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} \leq \frac {\alpha L _ {f} B _ {\omega} (\mathbf {x} ^ {*} , \mathbf {x} ^ {0})}{k},
$$

where $\alpha = 1$ in the constant stepsize setting and $\begin{array} { r } { \alpha = \operatorname* { m a x } \left. \eta , \frac { s } { L _ { f } } \right. } \end{array}$ if the backtracking rule is employed. 

Proof. (a) We will use the notation $m ( \mathbf x , \mathbf y ) \equiv f ( \mathbf y ) + \langle \nabla f ( \mathbf y ) , \mathbf x - \mathbf y \rangle$ . For both stepsize rules we have, for any $n \geq 0$ (see Remark 10.79), 

$$
f (\mathbf {x} ^ {n + 1}) \leq m (\mathbf {x} ^ {n + 1}, \mathbf {x} ^ {n}) + \frac {L _ {n}}{2} \| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {n} \| ^ {2}.
$$

Therefore, 

$$
\begin{array}{r l} & F (\mathbf {x} ^ {n + 1}) = f (\mathbf {x} ^ {n + 1}) + g (\mathbf {x} ^ {n + 1}) \\ & \qquad \leq m (\mathbf {x} ^ {n + 1}, \mathbf {x} ^ {n}) + g (\mathbf {x} ^ {n + 1}) + \frac {L _ {n}}{2} \| \mathbf {x} ^ {n + 1} - \mathbf {x} ^ {n} \| ^ {2} \\ & \qquad \leq m (\mathbf {x} ^ {n + 1}, \mathbf {x} ^ {n}) + g (\mathbf {x} ^ {n + 1}) + L _ {n} B _ {\omega} (\mathbf {x} ^ {n + 1}, \mathbf {x} ^ {n}), \end{array}\tag{10.94}
$$

where the 1-strong convexity of $\omega + \delta _ { \mathrm { d o m } ( g ) }$ was used in the last inequality. Note that 

$$
\mathbf {x} ^ {n + 1} = \operatorname{argmin} _ {\mathbf {x} \in \mathbb {E}} \left\{m \left(\mathbf {x}, \mathbf {x} ^ {n}\right) + g (\mathbf {x}) + L _ {n} B _ {\omega} \left(\mathbf {x}, \mathbf {x} ^ {n}\right) \right\}.\tag{10.95}
$$

Therefore, in particular, 

$$
\begin{array}{r l} & m (\mathbf {x} ^ {n + 1}, \mathbf {x} ^ {n}) + g (\mathbf {x} ^ {n + 1}) + L _ {n} B _ {\omega} (\mathbf {x} ^ {n + 1}, \mathbf {x} ^ {n}) \leq m (\mathbf {x} ^ {n}, \mathbf {x} ^ {n}) + g (\mathbf {x} ^ {n}) + L _ {n} B _ {\omega} (\mathbf {x} ^ {n}, \mathbf {x} ^ {n}) \\ & \qquad = f (\mathbf {x} ^ {n}) + g (\mathbf {x} ^ {n}) \\ & \qquad = F (\mathbf {x} ^ {n}), \end{array}
$$

which, combined with (10.94), implies that $F ( \mathbf { x } ^ { n + 1 } ) \leq F ( \mathbf { x } ^ { n } )$ , meaning that the sequence of function values $\{ F ( { \bf x } ^ { n } ) \} _ { n \ge 0 }$ is nonincreasing. 

(b) Let $k \geq 1$ and $\mathbf { x } ^ { * } \in X ^ { * }$ . Using the relation (10.95) and invoking the non-Euclidean second prox theorem (Theorem 9.12) with $\begin{array} { r } { \psi ( \mathbf { x } ) = \frac { m ( \mathbf { x } , \mathbf { x } ^ { n } ) + g ( \mathbf { x } ) } { L _ { n } } , \mathbf { b } = \mathbf { x } ^ { n } } \end{array}$ ， and $\mathbf { a } = \mathbf { x } ^ { n + 1 }$ , it follows that for all $\mathbf { x } \in \operatorname { d o m } ( g )$ 

$$
\langle \nabla \omega (\mathbf {x} ^ {n}) - \nabla \omega (\mathbf {x} ^ {n + 1}), \mathbf {x} - \mathbf {x} ^ {n + 1} \rangle \leq \frac {m (\mathbf {x} , \mathbf {x} ^ {n}) - m (\mathbf {x} ^ {n + 1} , \mathbf {x} ^ {n}) + g (\mathbf {x}) - g (\mathbf {x} ^ {n + 1})}{L _ {n}},
$$

which, combined with the three-points lemma (Lemma 9.11) with $\mathbf { a } = \mathbf { x } ^ { n + 1 } , \mathbf { b } = \mathbf { x } ^ { n }$ and $\mathbf { c } = \mathbf { x } .$ , yields the inequality 

$$
B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {n + 1}) + B _ {\omega} (\mathbf {x} ^ {n + 1}, \mathbf {x} ^ {n}) - B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {n}) \leq \frac {m (\mathbf {x} , \mathbf {x} ^ {n}) - m (\mathbf {x} ^ {n + 1} , \mathbf {x} ^ {n}) + g (\mathbf {x}) - g (\mathbf {x} ^ {n + 1})}{L _ {n}}.
$$

Rearranging terms, we obtain that 

$$
\begin{array}{r l} & m (\mathbf {x} ^ {n + 1}, \mathbf {x} ^ {n}) + g (\mathbf {x} ^ {n + 1}) + L _ {n} B _ {\omega} (\mathbf {x} ^ {n + 1}, \mathbf {x} ^ {n}) \leq m (\mathbf {x}, \mathbf {x} ^ {n}) + g (\mathbf {x}) + L _ {n} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {n}) \\ & \qquad - L _ {n} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {n + 1}), \end{array}
$$

which, combined with (10.94), yields the inequality 

$$
F (\mathbf {x} ^ {n + 1}) \leq m (\mathbf {x}, \mathbf {x} ^ {n}) + g (\mathbf {x}) + L _ {n} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {n}) - L _ {n} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {n + 1}).
$$

Since f is convex, $m ( \mathbf { x } , \mathbf { x } ^ { n } ) \leq f ( \mathbf { x } )$ , and hence 

$$
F (\mathbf {x} ^ {n + 1}) - F (\mathbf {x}) \leq L _ {n} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {n}) - L _ {n} B _ {\omega} (\mathbf {x}, \mathbf {x} ^ {n + 1}).
$$

Plugging in $\mathbf { x } = \mathbf { x } ^ { * }$ and dividing by $L _ { n }$ , we obtain 

$$
\frac {F (\mathbf {x} ^ {n + 1}) - F (\mathbf {x} ^ {*})}{L _ {n}} \leq B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {n}) - B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {n + 1}).
$$

Using the bound $L _ { n } \leq \alpha L _ { f }$ (see Remark 10.80), 

$$
\frac {F (\mathbf {x} ^ {n + 1}) - F (\mathbf {x} ^ {*})}{\alpha L _ {f}} \leq B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {n}) - B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {n + 1}),
$$

and hence 

$$
F (\mathbf {x} ^ {n + 1}) - F _ {\mathrm{opt}} \leq \alpha L _ {f} B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {n}) - \alpha L _ {f} B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {n + 1}).
$$

Summing the above inequality for $n = 0 , 1 , \ldots , k - 1$ , we obtain that 

$$
\sum_ {n = 0} ^ {k - 1} (F (\mathbf {x} ^ {n + 1}) - F _ {\mathrm{opt}}) \leq \alpha L _ {f} B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {0}) - \alpha L _ {f} B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {k}) \leq \alpha L _ {f} B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {0}).
$$

Using the monotonicity of the sequence of function values, we conclude that 

$$
k (F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}}) \leq \alpha L _ {f} B _ {\omega} (\mathbf {x} ^ {*}, \mathbf {x} ^ {0}),
$$

thus obtaining the result 

$$
F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} \leq \frac {\alpha L _ {f} B _ {\omega} (\mathbf {x} ^ {*} , \mathbf {x} ^ {0})}{k}. \quad \square
$$

# Chapter 11 The Block Proximal Gradient Method

Underlying Spaces: In this chapter, all the underlying spaces are Euclidean (see the details in Section 11.2). 

## 11.1 Decomposition Methods

Many of the methods discussed in this book are decomposition methods, which, loosely speaking, are methods that utilize at each step only a certain portion of the problem’s data or resort to solving a smaller-dimension problem at each step. One class of decomposition methods is the class of functional decomposition methods, in which the data of the problem comprise several functions, and at each iteration only a few of them (perhaps only one) are processed. Examples of functional decomposition methods were studied in the context of the model 

$$
\min _ {\mathbf {x}} \left\{\sum_ {i = 1} ^ {m} f _ {i} (\mathbf {x}): \mathbf {x} \in C \right\}.
$$

In Example 8.36 it was shown that an implementation of the stochastic projected subgradient method amounts to a method of the form 

$$
\mathbf {x} ^ {k + 1} = P _ {C} (\mathbf {x} ^ {k} - t _ {k} f _ {i _ {k}} ^ {\prime} (\mathbf {x} ^ {k})),
$$

where the index $i _ { k }$ is picked randomly by a uniform distribution. A deterministic version of this method is the incremental projected subgradient method, which was studied in Section 8.4, in which $i _ { k }$ is picked by a cyclic order. In both methods, each step exploits only one of the m functions that constitute the data of the problem. The proximal gradient method is actually another example of a functional decomposition method, where the relevant model (see Chapter 10) is 

$$
\min _ {\mathbf {x} \in \mathbb {E}} f (\mathbf {x}) + g (\mathbf {x}).
$$

The general step of the proximal gradient method is of the form 

$$
\mathbf {x} ^ {k + 1} = \operatorname{prox} _ {t _ {k} g} (\mathbf {x} ^ {k} - t _ {k} \nabla f (\mathbf {x} ^ {k})).
$$

The functions $f$ and $g$ are treated separately in the above update formula. First, a gradient step w.r.t. f is taken, and then a prox operator w.r.t. g is computed. 

Another class of decomposition methods is the class of variables decomposition methods, in which at each iteration only a subset of the decision variables is altered while all the other variables remain fixed. One example for such a method was given in Example 10.73, where the problem of minimizing a diferentiable function over $\mathbb { R } ^ { n }$ was considered. The method described in Example 10.73 (non-Euclidean gradient method under the $l _ { 1 } { \mathrm { - n o r m } } )$ picks one variable at each iteration by a certain greedy rule and performs a gradient step w.r.t. the chosen variable while keeping all the other variables fixed. 

In this chapter we will consider additional variables decomposition methods; these methods pick at each iteration one block of variables and perform a proximal gradient step w.r.t. the chosen block. 

## 11.2 Model and Assumptions

In this chapter we will consider methods for solving the composite model $f + g$ in the case where $g$ has a block separable structure. More specifically, the main model of this chapter is 

$$
\min _ {\mathbf {x} _ {1} \in \mathbb {E} _ {1}, \mathbf {x} _ {2} \in \mathbb {E} _ {2}, \dots , \mathbf {x} _ {p} \in \mathbb {E} _ {p}} \left\{F (\mathbf {x} _ {1}, \mathbf {x} _ {2}, \dots , \mathbf {x} _ {p}) = f (\mathbf {x} _ {1}, \mathbf {x} _ {2}, \dots , \mathbf {x} _ {p}) + \sum_ {j = 1} ^ {p} g _ {j} (\mathbf {x} _ {j}) \right\},\tag{11.1}
$$

where $\mathbb { E } _ { 1 } , \mathbb { E } _ { 2 } , \ldots , \mathbb { E } _ { p }$ are Euclidean spaces. We will denote the product space by $\mathbb { E } = \mathbb { E } _ { 1 } \times \mathbb { E } _ { 2 } \times \cdot \cdot \cdot \times \mathbb { E } _ { p }$ and use our convention (see Section 1.9) that the product space is also Euclidean with endowed norm 

$$
\| (\mathbf {u} _ {1}, \mathbf {u} _ {2}, \dots , \mathbf {u} _ {p}) \| _ {\mathbb {E}} = \sqrt {\sum_ {i = 1} ^ {p} \| \mathbf {u} _ {i} \| _ {\mathbb {E} _ {i}} ^ {2}}.
$$

In most cases we will omit the subscript of the norm indicating the underlying vector space (whose identity will be clear from the context). The function $g : \mathbb { E }  ( - \infty , \infty ]$ is defined by 

$$
g (\mathbf {x} _ {1}, \mathbf {x} _ {2}, \dots , \mathbf {x} _ {p}) \equiv \sum_ {i = 1} ^ {p} g _ {i} (\mathbf {x} _ {i}).
$$

The gradient w.r.t. the ith block $( i \in \{ 1 , 2 , . . . , p \} )$ is denoted by $\nabla _ { i } f$ , and whenever the function is diferentiable it holds tha 

$$
\nabla f (\mathbf {x}) = \left(\nabla_ {1} f (\mathbf {x}), \nabla_ {2} f (\mathbf {x}), \dots , \nabla_ {p} f (\mathbf {x})\right).
$$

For any $i \in \{ 1 , 2 , \ldots , p \}$ we define $\mathcal { U } _ { i } : \mathbb { E } _ { i }  \mathbb { E }$ to be the linear transformation given by 

$$
\mathcal {U} _ {i} (\mathbf {d}) = (\mathbf {0}, \ldots , \mathbf {0}, \underbrace {\mathbf {d}} _ {i t h b l o c k}, \mathbf {0}, \ldots , \mathbf {0}), \quad \mathbf {d} \in \mathbb {E} _ {i}.
$$

We also use throughout this chapter the notation that a vector $\mathbf { x } \in \mathbb { E }$ can be written as 

$$
\mathbf {x} = (\mathbf {x} _ {1}, \mathbf {x} _ {2}, \dots , \mathbf {x} _ {p}),
$$

and this relation will also be written as $\mathbf { x } = ( \mathbf { x } _ { i } ) _ { i = 1 } ^ { p }$ . Thus, in our notation, the main model (11.1) can be simply written as 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \{F (\mathbf {x}) = f (\mathbf {x}) + g (\mathbf {x}) \}.
$$

The basic assumptions on the model are summarized below. 

## Assumption 11.1.

(A) $g _ { i } : \mathbb { E } _ { i }  ( - \infty , \infty ]$ is proper closed and convex for any $i \in \{ 1 , 2 , \ldots , p \}$ 

(B) $f : \mathbb { E }  ( - \infty , \infty ]$ is proper and closed, and dom(f) is convex; dom $( g ) \subseteq$ int $\left( \operatorname { d o m } ( f ) \right)$ , and f is diferentiable over int $\left( \operatorname { d o m } ( f ) \right)$ 

(C) f is L<sub>f</sub>-smooth over int $( \mathrm { d o m } ( f ) ) \ ( L _ { f } > 0 )$ 

(D) There exist $L _ { 1 } , L _ { 2 } , \dots , L _ { p } > 0$ such that for any $i \in \{ 1 , 2 , \ldots , p \}$ it holds that 

$$
\left\| \nabla_ {i} f (\mathbf {x}) - \nabla_ {i} f \left(\mathbf {x} + \mathcal {U} _ {i} (\mathbf {d})\right) \right\| \leq L _ {i} \| \mathbf {d} \|\tag{11.2}
$$

for all $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ and d $\in \mathbb { E } _ { i }$ for which $\mathbf { x } + { \mathcal { U } } _ { i } ( \mathbf { d } ) \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ 

(E) The optimal set of problem (11.1) is nonempty and denoted by $X ^ { * }$ . The optimal value is denoted by $F _ { \mathrm { o p t } }$ 

Remark 11.2 (block/global Lipschitz constants). The constant $L _ { f }$ will be called the “global Lipschitz constant,” while the constants $L _ { 1 } , L _ { 2 } , \ldots , L _ { p }$ are the “block Lipschitz constants.” Obviously, we can choose $L _ { i } = L _ { f }$ for all i since by the definition of $L _ { f }$ , (11.2) holds for $L _ { i } = L _ { f }$ . However, the block Lipschitz constants can be significantly smaller than the global Lipschitz constant—a fact that might have significant influence on the performance of the derived algorithms, as well as their convergence rate. 

## 11.3 The Toolbox

## 11.3.1 The Partial Gradient Mapping

Recall that the gradient mapping associated with the functions $f , g$ and a constant $L > 0$ , as defined in Section 10.3.2, is a mapping from int $\left( \operatorname { d o m } ( f ) \right)$ to E given by 

$$
G _ {L} ^ {f, g} (\mathbf {x}) = L \left(\mathbf {x} - T _ {L} ^ {f, g} (\mathbf {x})\right),
$$

where $T _ { L } ^ { f , g }$ : int $( \operatorname { d o m } ( f ) ) \to { \mathbb { E } }$ is the prox-grad mapping given by 

$$
T _ {L} ^ {f, g} (\mathbf {x}) = \mathrm{prox} _ {\frac {1}{L} g} \left(\mathbf {x} - \frac {1}{L} \nabla f (\mathbf {x})\right).
$$

From now on we will always omit the superscripts and write $T _ { L }$ and $G _ { L }$ instead of $T _ { L } ^ { f , g }$ and $G _ { L } ^ { f , g }$ . In the context of block variables decomposition methods, it is also important to consider the notions of partial prox-grad mappings and partial gradient mappings. 

Definition 11.3 (partial prox-grad mapping). Suppose that f and $g _ { 1 } , g _ { 2 } , \ldots , g _ { p }$ satisfy properties (A) and (B) of Assumption 11.1, $L > 0$ , and let $i \in \{ 1 , 2 , \ldots , p \}$ Then the ith partial prox-grad mapping is the operator $T _ { L } ^ { i }$ : int $( \operatorname { d o m } ( f ) ) \to { \mathbb { E } } _ { i }$ defined $b y$ 

$$
T _ {L} ^ {i} (\mathbf {x}) = \mathrm{prox} _ {\frac {1}{L} g _ {i}} \left(\mathbf {x} _ {i} - \frac {1}{L} \nabla_ {i} f (\mathbf {x})\right).
$$

Definition 11.4 (partial gradient mapping). Suppose that f and $g _ { 1 } , g _ { 2 } , \ldots , g _ { p }$ satisfy properties (A) and (B) of Assumption 11.1, $L > 0$ , and let $i \in \{ 1 , 2 , \ldots , p \}$ Then the ith partial gradient mapping is the operator $G _ { L } ^ { i }$ : int $( \operatorname { d o m } ( f ) ) \to { \mathbb { E } } _ { i }$ defined by 

$$
G _ {L} ^ {i} (\mathbf {x}) = L \left(\mathbf {x} _ {i} - T _ {L} ^ {i} (\mathbf {x})\right).
$$

The ith partial prox-grad and gradient mappings depend on $f$ and $g _ { i } .$ , but this dependence is not indicated in our notation. If $g _ { i } \equiv 0$ for some $i \in \{ 1 , 2 , . . . , p \}$ , then $G _ { L } ^ { i } ( { \bf x } ) = \nabla _ { i } f ( { \bf x } )$ ; that is, in this case the partial gradient mapping coincides with the mapping $\mathbf { x } \mapsto \nabla _ { i } f ( \mathbf { x } )$ . Some basic properties of the partial prox-grad and gradient mappings are summarized in the following lemma. 

Lemma 11.5. Suppose that f and $g _ { 1 } , g _ { 2 } , \ldots , g _ { p }$ satisfy properties (A) and (B) of Assumption 11.1, $L > 0$ , and let $i \in \{ 1 , 2 , \ldots , p \}$ . Then for any $\mathbf { x \in }$ int(dom(f)), 

$$
\begin{array}{l} T _ {L} (\mathbf {x}) = (T _ {L} ^ {1} (\mathbf {x}), T _ {L} ^ {2} (\mathbf {x}), \ldots , T _ {L} ^ {p} (\mathbf {x})), \\ G _ {L} (\mathbf {x}) = (G _ {L} ^ {1} (\mathbf {x}), G _ {L} ^ {2} (\mathbf {x}), \ldots , G _ {L} ^ {p} (\mathbf {x})). \end{array}\tag{11.3}
$$

Proof. By Theorem 6.6, we have that for any $\mathbf { y } \in \operatorname { d o m } ( f )$ 

$$
\mathrm{prox} _ {\frac {1}{L} g} (\mathbf {y}) = (\mathrm{prox} _ {\frac {1}{L} g _ {i}} (\mathbf {y} _ {i})) _ {i = 1} ^ {p}.
$$

Thus, for any $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ , 

$$
\begin{array}{l} T _ {L} (\mathbf {x}) = \mathrm{prox} _ {\frac {1}{L} g} \left(\mathbf {x} - \frac {1}{L} \nabla f (\mathbf {x})\right) = \left(\mathrm{prox} _ {\frac {1}{L} g _ {i}} \left(\left[ \mathbf {x} - \frac {1}{L} \nabla f (\mathbf {x}) \right] _ {i}\right)\right) _ {i = 1} ^ {p} \\ \qquad = \left(\mathrm{prox} _ {\frac {1}{L} g _ {i}} \left(\mathbf {x} _ {i} - \frac {1}{L} \nabla_ {i} f (\mathbf {x})\right)\right) _ {i = 1} ^ {p} \\ \qquad = (T _ {L} ^ {i} (\mathbf {x})) _ {i = 1} ^ {p}. \end{array}
$$

The second identity follows immediately: 

$$
\begin{array}{r l} & G _ {L} (\mathbf {x}) = L (\mathbf {x} - T _ {L} (\mathbf {x})) = L \left((\mathbf {x} _ {i}) _ {i = 1} ^ {p} - (T _ {L} ^ {i} (\mathbf {x})) _ {i = 1} ^ {p}\right) \\ & \qquad = \left(L (\mathbf {x} _ {i} - T _ {L} ^ {i} (\mathbf {x}))\right) _ {i = 1} ^ {p} \\ & \qquad = \left(G _ {L} ^ {i} (\mathbf {x})\right) _ {i = 1} ^ {p}. \quad \square \end{array}
$$

A point $\mathbf { x } ^ { * } \in \mathrm { d o m } ( g )$ is a stationary point of problem (11.1) if $- \nabla f ( \mathbf { x } ^ { * } ) \in$ $\partial g ( \mathbf { x } ^ { * } )$ (see Definition 3.73). The following simple theorem shows that the stationarity condition for problem (11.1) can be decomposed into $p$ conditions expressed in terms of the partial gradient mappings. 

Theorem 11.6. Suppose that f and $g _ { 1 } , g _ { 2 } , \dotsc , g _ { p }$ satisfy properties (A) and (B) of Assumption 11.1. Then 

(a) $\mathbf { x } ^ { * } \in \mathrm { d o m } ( g )$ is a stationary point of problem (11.1) if and only if 

$$
- \nabla_ {i} f (\mathbf {x} ^ {*}) \in \partial g _ {i} (\mathbf {x} _ {i} ^ {*}), i = 1, 2, \ldots , p;\tag{11.4}
$$

(b) for any p positive numbers $M _ { 1 } , M _ { 2 } , \ldots , M _ { p } > 0 , { \bf x } ^ { * } \in$ dom(g) is a stationary point of problem (11.1) if and only if 

$$
G _ {M _ {i}} ^ {i} (\mathbf {x} ^ {*}) = \mathbf {0}, i = 1, 2, \ldots , p.
$$

Proof. (a) By definition, $\mathbf { x } ^ { * } \in \mathrm { d o m } ( g )$ is a stationary point of problem (11.1) if and only if 

$$
- \nabla f (\mathbf {x} ^ {*}) \in \partial g (\mathbf {x} ^ {*}).\tag{11.5}
$$

By the block separable structure of $g ,$ it is easy to show that 

$$
\partial g (\mathbf {x} ^ {*}) = \partial g _ {1} (\mathbf {x} _ {1} ^ {*}) \times \partial g _ {2} (\mathbf {x} _ {2} ^ {*}) \times \dots \times \partial g _ {p} (\mathbf {x} _ {p} ^ {*}),
$$

which, combined with the fact that $\nabla f ( \mathbf { x } ^ { * } ) = ( \nabla _ { 1 } f ( \mathbf { x } ^ { * } ) , \nabla _ { 2 } f ( \mathbf { x } ^ { * } ) , \ldots , \nabla _ { p } f ( \mathbf { x } ^ { * } ) )$ ， implies that the relation (11.5) is equivalent to 

$$
- \left(\nabla_ {1} f \left(\mathbf {x} ^ {*}\right), \nabla_ {2} f \left(\mathbf {x} ^ {*}\right), \dots , \nabla_ {p} f \left(\mathbf {x} ^ {*}\right)\right) \in \partial g _ {1} \left(\mathbf {x} _ {1} ^ {*}\right) \times \partial g _ {2} \left(\mathbf {x} _ {2} ^ {*}\right) \times \dots \times \partial g _ {p} \left(\mathbf {x} _ {p} ^ {*}\right),
$$

that is, to (11.4). 

(b) By the definition of the partial gradient mapping, $G _ { M _ { i } } ^ { i } ( \mathbf { x } ^ { * } ) = \mathbf { 0 }$ if and only $\begin{array} { r } { \mathrm { i f } \ \mathbf { x } _ { i } ^ { * } = \mathrm { p r o x } _ { \frac { 1 } { M _ { i } } g _ { i } } \left( \mathbf { x } _ { i } ^ { * } - \frac { 1 } { M _ { i } } \nabla _ { i } f ( \mathbf { x } ^ { * } ) \right) } \end{array}$ , which, by the second prox theorem (Theorem 6.39), is equivalent to 

$$
\left(\mathbf {x} _ {i} ^ {*} - \frac {1}{M _ {i}} \nabla_ {i} f (\mathbf {x} ^ {*})\right) - \mathbf {x} _ {i} ^ {*} \in \frac {1}{M _ {i}} \partial g _ {i} (\mathbf {x} _ {i} ^ {*}),
$$

that is, to 

$$
- \nabla_ {i} f (\mathbf {x} ^ {*}) \in \partial g _ {i} (\mathbf {x} _ {i} ^ {*}).
$$

To summarize, $G _ { M _ { i } } ^ { i } ( \mathbf { x } ^ { * } ) = \mathbf { 0 }$ for all i if and only if $- \nabla _ { i } f ( \mathbf { x } ^ { * } ) \in \partial g _ { i } ( \mathbf { x } _ { i } ^ { * } )$ for all i, which, by part (a), is equivalent to saying that $\mathbf { x } ^ { * }$ is a stationary point of problem (11.1). 

The next results shows some monotonicity properties of the partial gradient mapping w.r.t. its parameter. The result is presented without its proof, which is an almost verbatim repetition of the arguments in Theorem 10.9. 

Theorem 11.7 (monotonicity of the partial gradient mapping). Suppose that f and $g _ { 1 } , g _ { 2 } , \ldots , g _ { p }$ satisfy properties (A) and (B) of Assumption 11.1, and let $i \in \{ 1 , 2 , \ldots , p \}$ . Suppose that $L _ { 1 } \ge L _ { 2 } > 0$ . Then 

$$
\| G _ {L _ {1}} ^ {i} (\mathbf {x}) \| \geq \| G _ {L _ {2}} ^ {i} (\mathbf {x}) \|
$$

and 

$$
\frac {\| G _ {L _ {1}} ^ {i} (\mathbf {x}) \|}{L _ {1}} \leq \frac {\| G _ {L _ {2}} ^ {i} (\mathbf {x}) \|}{L _ {2}}
$$

for any x ∈ int(dom(f)). 

## 11.3.2 The Block Descent Lemma

The block descent lemma is a variant of the descent lemma (Lemma 5.7), and its proof is almost identical. 

Lemma 11.8 (block descent lemma). Let $f : \mathbb { E } _ { 1 } \times \mathbb { E } _ { 2 } \times \cdot \cdot \cdot \times \mathbb { E } _ { p } \to ( - \infty , \infty ]$ be a proper function whose domain dom(f) is convex. Assume that f is diferentiable over int $\left( \operatorname { d o m } ( f ) \right)$ . Let $i \in \{ 1 , 2 , \ldots , p \}$ . Suppose that there exists $L _ { i } > 0$ for which 

$$
\left\| \nabla_ {i} f (\mathbf {y}) - \nabla_ {i} f (\mathbf {y} + \mathcal {U} _ {i} (\mathbf {d})) \right\| \leq L _ {i} \| \mathbf {d} \|
$$

for any $\mathbf { y } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ and $\mathbf { d } \in \mathbb { E } _ { i }$ for which $\mathbf { y } + \mathcal { U } _ { i } ( \mathbf { d } ) \in \mathrm { i n t } ( \mathrm { d o m } ( f ) )$ . Then 

$$
f (\mathbf {x} + \mathcal {U} _ {i} (\mathbf {d})) \leq f (\mathbf {x}) + \langle \nabla_ {i} f (\mathbf {x}), \mathbf {d} \rangle + \frac {L _ {i}}{2} \| \mathbf {d} \| ^ {2}
$$

for any $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ and d $\in \mathbb { E } _ { i }$ for which $\mathbf { x } + { \mathcal { U } } _ { i } ( \mathbf { d } ) \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ 

Proof. Let $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ and $\mathbf { d } \in \mathbb { E } _ { i }$ such that $\mathbf { x } + { \mathcal { U } } _ { i } ( \mathbf { d } ) \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ . Denote $\mathbf { x } ^ { ( t ) } = \mathbf { x } + t \mathcal { U } _ { i } ( \mathbf { d } )$ and define $g ( t ) = f ( \mathbf { x } ^ { ( t ) } )$ . By the fundamental theorem of calculus, 

$$
\begin{array}{r l} & f (\mathbf {x} ^ {(1)}) - f (\mathbf {x}) = g (1) - g (0) = \int_ {0} ^ {1} g ^ {\prime} (t) d t \\ & \qquad = \int_ {0} ^ {1} \langle \nabla f (\mathbf {x} ^ {(t)}), \mathcal {U} _ {i} (\mathbf {d}) \rangle d t = \int_ {0} ^ {1} \langle \nabla_ {i} f (\mathbf {x} ^ {(t)}), \mathbf {d} \rangle d t \\ & \qquad = \langle \nabla_ {i} f (\mathbf {x}), \mathbf {d} \rangle + \int_ {0} ^ {1} \langle \nabla_ {i} f (\mathbf {x} ^ {(t)}) - \nabla_ {i} f (\mathbf {x}), \mathbf {d} \rangle d t. \end{array}
$$

Thus, 

$$
\begin{array}{l} | f (\mathbf {x} ^ {(1)}) - f (\mathbf {x}) - \langle \nabla_ {i} f (\mathbf {x}), \mathbf {d} \rangle | = \left| \int_ {0} ^ {1} \langle \nabla_ {i} f (\mathbf {x} ^ {(t)}) - \nabla_ {i} f (\mathbf {x}), \mathbf {d} \rangle d t \right| \\ \leq \int_ {0} ^ {1} | \langle \nabla_ {i} f (\mathbf {x} ^ {(t)}) - \nabla_ {i} f (\mathbf {x}), \mathbf {d} \rangle | d t \\ \stackrel {{(*)}} {{\leq}} \int_ {0} ^ {1} \| \nabla_ {i} f (\mathbf {x} ^ {(t)}) - \nabla_ {i} f (\mathbf {x}) \| \cdot \| \mathbf {d} \| d t \\ \leq \int_ {0} ^ {1} t L _ {i} \| \mathbf {d} \| ^ {2} d t \\ = \frac {L _ {i}}{2} \| \mathbf {d} \| ^ {2}, \end{array}
$$

where the Cauchy–Schwarz inequality was used in (∗). 

## 11.3.3 Suficient Decrease

The basic step that will be employed by all the methods discussed in this chapter is a proximal gradient step w.r.t. a given block. Specifically, for a given $\mathbf { x } \in \mathbb { E }$ and $i \in \{ 1 , 2 , \ldots , p \}$ , the next updated vector $\mathbf { x } ^ { + }$ will have the form 

$$
\mathbf {x} _ {j} ^ {+} = \left\{ \begin{array}{l l} \mathbf {x} _ {j}, & j \neq i, \\ T _ {L _ {i}} ^ {i} (\mathbf {x}), & j = i. \end{array} \right.
$$

The above update formula can be compactly written as 

$$
\mathbf {x} ^ {+} = \mathbf {x} + \mathcal {U} _ {i} (T _ {L _ {i}} ^ {i} (\mathbf {x}) - \mathbf {x} _ {i}).
$$

We will now prove a variant of the suficient decrease lemma (Lemma 10.4), in which only Lipschitz continuity w.r.t. a certain block of the gradient of the function is assumed. 

Lemma 11.9 (block suficient decrease lemma). Suppose that f and $g _ { 1 } , g _ { 2 }$ $\cdot \cdot \cdot g _ { p }$ satisfy properties (A) and (B) of Assumption 11.1. Let $i \in \{ 1 , 2 , . . . , p \}$ Suppose that there exists $L _ { i } > 0$ for which 

$$
\left\| \nabla_ {i} f (\mathbf {y}) - \nabla_ {i} f (\mathbf {y} + \mathcal {U} _ {i} (\mathbf {d})) \right\| \leq L _ {i} \| \mathbf {d} \|
$$

for any $\mathbf { y } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ and d $\in \mathbb { E } _ { i }$ for which $\mathbf { y } + \mathcal { U } _ { i } ( \mathbf { d } ) \in \mathrm { i n t } ( \mathrm { d o m } ( f ) )$ . Then 

$$
F (\mathbf {x}) - F (\mathbf {x} + \mathcal {U} _ {i} (T _ {L _ {i}} ^ {i} (\mathbf {x}) - \mathbf {x} _ {i})) \geq \frac {1}{2 L _ {i}} \| G _ {L _ {i}} ^ {i} (\mathbf {x}) \| ^ {2}\tag{11.6}
$$

for all $\mathbf { x \in }$ int(dom(f)). 

Proof. For the sake of simplicity, we use the shorthand notation $\mathbf { x } ^ { + } ~ = ~ \mathbf { x } ~ +$ $\mathcal { U } _ { i } ( T _ { L _ { i } } ^ { i } ( \mathbf { x } ) - \mathbf { x } _ { i } )$ . By the block descent lemma (Lemma 11.8), we have that 

$$
f (\mathbf {x} ^ {+}) \leq f (\mathbf {x}) + \left\langle \nabla_ {i} f (\mathbf {x}), T _ {L _ {i}} ^ {i} (\mathbf {x}) - \mathbf {x} _ {i} \right\rangle + \frac {L _ {i}}{2} \| T _ {L _ {i}} ^ {i} (\mathbf {x}) - \mathbf {x} _ {i} \| ^ {2}.\tag{11.7}
$$

By the second prox theorem (Theorem 6.39), since $T _ { L _ { i } } ^ { i } ( { \bf x } ) ~ = ~ \mathrm { p r o x } _ { \frac { 1 } { L _ { i } } g _ { i } } \left( { \bf x } _ { i } ~ - ~ \right.$ $\textstyle { \frac { 1 } { L _ { i } } } \nabla _ { i } f ( \mathbf { x } ) )$ , we have 

$$
\left\langle \mathbf {x} _ {i} - \frac {1}{L _ {i}} \nabla_ {i} f (\mathbf {x}) - T _ {L _ {i}} ^ {i} (\mathbf {x}), \mathbf {x} _ {i} - T _ {L _ {i}} ^ {i} (\mathbf {x}) \right\rangle \leq \frac {1}{L _ {i}} g _ {i} (\mathbf {x} _ {i}) - \frac {1}{L _ {i}} g _ {i} (T _ {L _ {i}} ^ {i} (\mathbf {x})),
$$

and hence 

$$
\left\langle \nabla_ {i} f (\mathbf {x}), T _ {L _ {i}} ^ {i} (\mathbf {x}) - \mathbf {x} _ {i} \right\rangle \leq - L _ {i} \left\| T _ {L _ {i}} ^ {i} (\mathbf {x}) - \mathbf {x} _ {i} \right\| ^ {2} + g _ {i} (\mathbf {x} _ {i}) - g _ {i} (\mathbf {x} _ {i} ^ {+}),
$$

which, combined with (11.7), yields 

$$
f (\mathbf {x} ^ {+}) + g _ {i} (\mathbf {x} _ {i} ^ {+}) \leq f (\mathbf {x}) + g _ {i} (\mathbf {x} _ {i}) - \frac {L _ {i}}{2} \left\| T _ {L _ {i}} ^ {i} (\mathbf {x}) - \mathbf {x} _ {i} \right\| ^ {2}.
$$

Adding the identity $\begin{array} { r } { \sum _ { j \neq i } g _ { j } ( \mathbf { x } _ { j } ^ { + } ) = \sum _ { j \neq i } g _ { j } ( \mathbf { x } _ { j } ) } \end{array}$ to the last inequality yields 

$$
F (\mathbf {x} ^ {+}) \leq F (\mathbf {x}) - \frac {L _ {i}}{2} \left\| T _ {L _ {i}} ^ {i} (\mathbf {x}) - \mathbf {x} _ {i} \right\| ^ {2},
$$

which, by the definition of the partial gradient mapping, is equivalent to the desired result (11.6). 

Remark 11.10. Under the setting of Lemma 11.9, if we denote $\mathbf { x } ^ { + } ~ = ~ \mathbf { x } ~ +$ $\mathcal { U } _ { i } ( T _ { L _ { i } } ^ { i } ( \mathbf { x } ) - \mathbf { x } _ { i } )$ , then the suficient decrease condition (11.6) can be written in the following form: 

$$
F (\mathbf {x}) - F (\mathbf {x} ^ {+}) \geq \frac {L _ {i}}{2} \| \mathbf {x} - \mathbf {x} ^ {+} \| ^ {2}.
$$

## 11.4 The Cyclic Block Proximal Gradient Method

In the cyclic block proximal gradient (CBPG) method we successively pick a block in a cyclic manner and perform a prox-grad step w.r.t. the chosen block. The kth iterate is denoted by $\mathbf { x } ^ { k } \hat { = } ( \mathbf { x } _ { 1 } ^ { k } , \mathbf { x } _ { 2 } ^ { k } , \cdot \cdot \cdot , \mathbf { x } _ { p } ^ { k } )$ . Each iteration of the CBPG method involves $p$ “subiterations,” and the by-products of these subiterations will be denoted by the following auxiliary subsequences: 

$$
\begin{array}{r l} & {\mathbf {x} ^ {k, 0} = \mathbf {x} ^ {k} = (\mathbf {x} _ {1} ^ {k}, \mathbf {x} _ {2} ^ {k}, \ldots , \mathbf {x} _ {p} ^ {k}),} \\ & {\mathbf {x} ^ {k, 1} = (\mathbf {x} _ {1} ^ {k + 1}, \mathbf {x} _ {2} ^ {k}, \ldots , \mathbf {x} _ {p} ^ {k}),} \\ & {\mathbf {x} ^ {k, 2} = (\mathbf {x} _ {1} ^ {k + 1}, \mathbf {x} _ {2} ^ {k + 1}, \mathbf {x} _ {3} ^ {k}, \ldots , \mathbf {x} _ {p} ^ {k}),} \\ & {\quad \vdots} \\ & {\mathbf {x} ^ {k, p} = \mathbf {x} ^ {k + 1} = (\mathbf {x} _ {1} ^ {k + 1}, \mathbf {x} _ {2} ^ {k + 1}, \ldots , \mathbf {x} _ {p} ^ {k + 1}).} \end{array}
$$

We can also write the following formula for the kth member of the ith auxiliary sequence: 

$$
\mathbf {x} ^ {k, i} = \sum_ {j = 1} ^ {i} \mathcal {U} _ {j} (\mathbf {x} _ {j} ^ {k + 1}) + \sum_ {j = i + 1} ^ {p} \mathcal {U} _ {j} (\mathbf {x} _ {j} ^ {k}).\tag{11.8}
$$

We are now ready to present the method. 

The Cyclic Block Proximal Gradient (CBPG) Method
Initialization: pick $\mathbf{x}^{0} = (\mathbf{x}_{1}^{0}, \mathbf{x}_{2}^{0}, \ldots, \mathbf{x}_{p}^{0}) \in \text{int}(\text{dom}(f))$ .
General step: for any $k = 0, 1, 2, \ldots$ execute the following steps:
• set $x^{k,0} = x^{k}$ ;
• for $i = 1, 2, \ldots, p$ , compute $x^{k,i} = x^{k,i-1} + \mathcal{U}_i(T_{L_i}^i(x^{k,i-1}) - x_i^{k,i-1})$ ;
• set $x^{k+1} = x^{k,p}$ . 

## 11.4.1 Convergence Analysis of the CBPG Method—The Nonconvex Case

The convergence analysis of the CBPG method relies on the following technical lemma, which is a direct consequence of the suficient decrease property of Lemma 11.9. 

Lemma 11.11 (suficient decrease of the CBPG method—version I). Suppose that Assumption 11.1 holds, and let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the CBPG method for solving problem (11.1) with the auxiliary sequences defined in (11.8). Then 

(a) for all $k \geq 0$ and $j \in \{ 0 , 1 , \ldots , p - 1 \}$ it holds that 

$$
F (\mathbf {x} ^ {k, j}) - F (\mathbf {x} ^ {k, j + 1}) \geq \frac {1}{2 L _ {j + 1}} \| G _ {L _ {j + 1}} ^ {j + 1} (\mathbf {x} ^ {k, j}) \| ^ {2},\tag{11.9}
$$

or equivalently, 

$$
F (\mathbf {x} ^ {k, j}) - F (\mathbf {x} ^ {k, j + 1}) \geq \frac {L _ {j + 1}}{2} \| \mathbf {x} ^ {k, j} - \mathbf {x} ^ {k, j + 1} \| ^ {2};\tag{11.10}
$$

(b) for all $k \geq 0$ 

$$
F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \geq \frac {L _ {\mathrm{min}}}{2} \| \mathbf {x} ^ {k} - \mathbf {x} ^ {k + 1} \| ^ {2},\tag{11.11}
$$

where $\begin{array} { r } { L _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { i = 1 , 2 , . . . , p } L _ { i } } \end{array}$ 

Proof. (a) Inequality (11.9) follows by invoking Lemma 11.9 with $\mathbf { x } = \mathbf { x } ^ { k , j }$ and $i = j + 1$ . The result (11.10) now follows by the identity $\| \mathbf { x } ^ { k , j } - \mathbf { x } ^ { k , j + 1 } \| ^ { 2 } =$ $\begin{array} { r } { \| T _ { L _ { j + 1 } } ^ { j + \frac { \check { 1 } } { 1 } } ( \mathbf { x } ^ { k , j } ) - \mathbf { x } _ { j + 1 } ^ { k } \| ^ { 2 } = \frac { \check { 1 } } { L _ { j + 1 } ^ { 2 } } \| \mathring { G } _ { L _ { j + 1 } } ^ { j + 1 } ( \mathbf { x } ^ { k , j } ) \| ^ { 2 } } \end{array}$ 

(b) Summing the inequality (11.10) over $j = 0 , 1 , \ldots , p - 1$ , we obtain 

$$
\begin{array}{l} F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) = \sum_ {j = 0} ^ {p - 1} (F (\mathbf {x} ^ {k, j}) - F (\mathbf {x} ^ {k, j + 1})) \geq \sum_ {j = 0} ^ {p - 1} \frac {L _ {j + 1}}{2} \| \mathbf {x} ^ {k, j} - \mathbf {x} ^ {k, j + 1} \| ^ {2} \\ = \sum_ {j = 0} ^ {p - 1} \frac {L _ {j + 1}}{2} \| \mathbf {x} _ {j + 1} ^ {k} - \mathbf {x} _ {j + 1} ^ {k + 1} \| ^ {2} \geq \frac {L _ {\min}}{2} \sum_ {j = 0} ^ {p - 1} \| \mathbf {x} _ {j + 1} ^ {k} - \mathbf {x} _ {j + 1} ^ {k + 1} \| ^ {2} \\ = \frac {L _ {\min}}{2} \| \mathbf {x} ^ {k} - \mathbf {x} ^ {k + 1} \| ^ {2}. \quad \square \end{array}
$$

A direct result of the last lemma is the monotonicity in function values of the sequence generated by the CBPG method. 

Corollary 11.12 (monotonicity of the sequence generated by the CBPG method). Under the setting of Lemma 11.11, for any $k \geq 0 , F ( \mathbf { x } ^ { k + 1 } ) \leq F ( \mathbf { x } ^ { k } )$ and equality holds if and only if $\mathbf { \Delta x } ^ { k } = \mathbf { x } ^ { k + 1 }$ 

We can now prove a suficient decrease property of the CBPG method in terms of the (nonpartial) gradient mapping. 

Lemma 11.13 (suficient decrease of the CBPG method—version II). Suppose that Assumption 11.1 holds, and let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the CBPG method for solving problem (11.1). Then for any $k \geq 0$ 2 

$$
F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \geq \frac {C}{p} \| G _ {L _ {\mathrm{min}}} (\mathbf {x} ^ {k}) \| ^ {2},\tag{11.12}
$$

where 

$$
C = \frac {L _ {\mathrm{min}}}{2 (L _ {f} + 2 L _ {\mathrm{max}} + \sqrt {L _ {\mathrm{min}} L _ {\mathrm{max}}}) ^ {2}}\tag{11.13}
$$

and 

$$
L _ {\mathrm{min}} = \min _ {i = 1, 2, \dots , p} L _ {i}, L _ {\mathrm{max}} = \max _ {i = 1, 2, \dots , p} L _ {i}.
$$

Proof. Let $i \in \{ 0 , 1 , \ldots , p - 1 \}$ }. By (11.9), 

$$
F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \geq F (\mathbf {x} ^ {k, i}) - F (\mathbf {x} ^ {k, i + 1}) \geq \frac {1}{2 L _ {i + 1}} \| G _ {L _ {i + 1}} ^ {i + 1} (\mathbf {x} ^ {k, i}) \| ^ {2}.\tag{11.14}
$$

We can bound $\| G _ { L _ { i + 1 } } ^ { i + 1 } ( \mathbf { x } ^ { k } ) \|$ as follows: 

$$
\begin{array}{l l} \| G _ {L _ {i + 1}} ^ {i + 1} (\mathbf {x} ^ {k}) \| \leq \| G _ {L _ {i + 1}} ^ {i + 1} (\mathbf {x} ^ {k}) - G _ {L _ {i + 1}} ^ {i + 1} (\mathbf {x} ^ {k, i}) \| + \| G _ {L _ {i + 1}} ^ {i + 1} (\mathbf {x} ^ {k, i}) \| & [ \text { triangle   inequality } ] \\ \leq \| G _ {L _ {i + 1}} (\mathbf {x} ^ {k}) - G _ {L _ {i + 1}} (\mathbf {x} ^ {k, i}) \| + \| G _ {L _ {i + 1}} ^ {i + 1} (\mathbf {x} ^ {k, i}) \| & [ (1 1. 3) ] \\ \leq (2 L _ {i + 1} + L _ {f}) \| \mathbf {x} ^ {k} - \mathbf {x} ^ {k, i} \| + \| G _ {L _ {i + 1}} ^ {i + 1} (\mathbf {x} ^ {k, i}) \| & [ \text { Lemma   10.10(a)} ] \\ \leq (2 L _ {i + 1} + L _ {f}) \| \mathbf {x} ^ {k} - \mathbf {x} ^ {k + 1} \| + \| G _ {L _ {i + 1}} ^ {i + 1} (\mathbf {x} ^ {k, i}) \|, \end{array}
$$

where the last inequality follows by the following argument: 

$$
\| \mathbf {x} ^ {k} - \mathbf {x} ^ {k, i} \| = \sqrt {\sum_ {j = 1} ^ {i} \| \mathbf {x} _ {j} ^ {k} - \mathbf {x} _ {j} ^ {k + 1} \| ^ {2}} \leq \sqrt {\sum_ {j = 1} ^ {p} \| \mathbf {x} _ {j} ^ {k} - \mathbf {x} _ {j} ^ {k + 1} \| ^ {2}} = \| \mathbf {x} ^ {k} - \mathbf {x} ^ {k + 1} \|.
$$

Using the inequalities (11.11) and (11.14), it follows that we can continue to bound $\| G _ { L _ { i + 1 } } ^ { i + 1 } ( \mathbf { x } ^ { k } ) \|$ as follows: 

$$
\begin{array}{r c l} \| G _ {L _ {i + 1}} ^ {i + 1} (\mathbf {x} ^ {k}) \| & \leq & (2 L _ {i + 1} + L _ {f}) \| \mathbf {x} ^ {k} - \mathbf {x} ^ {k + 1} \| + \| G _ {L _ {i + 1}} ^ {i + 1} (\mathbf {x} ^ {k, i}) \| \\ & \leq & \left[ \frac {\sqrt {2} (2 L _ {i + 1} + L _ {f})}{\sqrt {L _ {\min}}} + \sqrt {2 L _ {i + 1}} \right] \sqrt {F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1})} \\ & \overset {L _ {i + 1} \leq L _ {\max}} {\leq} & \sqrt {\frac {2}{L _ {\min}}} (L _ {f} + 2 L _ {\max} + \sqrt {L _ {\min} L _ {\max}}) \sqrt {F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1})}. \end{array}
$$

By the monotonicity of the partial gradient mapping (Theorem 11.7), it follows that $\| G _ { L _ { \operatorname* { m i n } } } ^ { i + 1 } ( \mathbf { x } ^ { k } ) \| \leq \| G _ { L _ { i + 1 } } ^ { i + 1 } ( \mathbf { x } ^ { k } ) \|$ , and hence, for any $i \in \{ 0 , 1 , \ldots , p - 1 \}$ , 

$$
F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \geq C \| G _ {L _ {\mathrm{min}}} ^ {i + 1} (\mathbf {x} ^ {k}) \| ^ {2},
$$

where C is given in (11.13). We can thus conclude that 

$$
\| G _ {L _ {\min}} (\mathbf {x} ^ {k}) \| ^ {2} = \sum_ {i = 0} ^ {p - 1} \| G _ {L _ {\min}} ^ {i + 1} (\mathbf {x} ^ {k}) \| ^ {2} \leq \sum_ {i = 0} ^ {p - 1} \frac {F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1})}{C} = \frac {p}{C} (F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1})),
$$

which is the same as (11.12). 

Equipped with Lemma 11.13, it is easy to show some standard convergence properties of the CBPG method. 

Theorem 11.14 (convergence of the CBPG method—nonconvex case). Suppose that Assumption 11.1 holds, and let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the CBPG method for solving problem (11.1). Denote 

$$
L _ {\min} = \min _ {i = 1, 2, \dots , p} L _ {i}, L _ {\max} = \max _ {i = 1, 2, \dots , p} L _ {i},
$$

and let C be given in (11.13). Then 

(a) $G _ { L _ { \operatorname* { m i n } } } ( \mathbf { x } ^ { k } ) \to \mathbf { 0 }$ as $k  \infty ;$ 

$$
\text {(b)} \min _ {n = 0, 1, \dots , k} \| G _ {L _ {\min}} (\mathbf {x} ^ {n}) \| \leq \frac {\sqrt {p (F (\mathbf {x} ^ {0}) - F _ {\mathrm{opt}})}}{\sqrt {C (k + 1)}};
$$

(c) all limit points of the sequence $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ are stationary points of problem (11.1). 

Proof. (a) Since the sequence $\{ F ( \mathbf { x } ^ { k } ) \} _ { k \geq 0 }$ is nonincreasing (Corollary 11.12) and bounded below (by Assumption 11.1(E)), it converges. Thus, in particular $F ( \mathbf { x } ^ { k } ) -$ $F ( \mathbf { x } ^ { k + 1 } )  0$ as $k \to \infty$ , which, combined with (11.12), implies that $\| G _ { L _ { \operatorname* { m i n } } } ( \mathbf { x } ^ { k } ) \| $ 0 as $k \to \infty$ 

(b) By Lemma 11.13, for any $n \geq 0$ 2 

$$
F (\mathbf {x} ^ {n}) - F (\mathbf {x} ^ {n + 1}) \geq \frac {C}{p} \| G _ {L _ {\min}} (\mathbf {x} ^ {n}) \| ^ {2}.\tag{11.15}
$$

Summing the above inequality over $n = 0 , 1 , \ldots , k$ , we obtain 

$$
F (\mathbf {x} ^ {0}) - F (\mathbf {x} ^ {k + 1}) \geq \frac {C}{p} \sum_ {n = 0} ^ {k} \| G _ {L _ {\min}} (\mathbf {x} ^ {n}) \| ^ {2} \geq \frac {C (k + 1)}{p} \min _ {n = 0, 1, \dots , k} \| G _ {L _ {\min}} (\mathbf {x} ^ {n}) \| ^ {2}.
$$

Using the fact that $F ( \mathbf { x } ^ { k + 1 } ) \geq F _ { \mathrm { o p t } }$ , the result follows. 

(c) Let x¯ be a limit point of $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ . Then there exists a subsequence $\{ \mathbf { x } ^ { k _ { j } } \} _ { j \ge 0 }$ converging to x¯. For any $j \geq 0$ 2 

$$
\begin{array}{r l} & {\| G _ {L _ {\min}} (\bar {\mathbf {x}}) \| \leq \| G _ {L _ {\min}} (\mathbf {x} ^ {k _ {j}}) - G _ {L _ {\min}} (\bar {\mathbf {x}}) \| + \| G _ {L _ {\min}} (\mathbf {x} ^ {k _ {j}}) \|} \\ & {\qquad \leq (2 L _ {\min} + L _ {f}) \| \mathbf {x} ^ {k _ {j}} - \bar {\mathbf {x}} \| + \| G _ {L _ {\min}} (\mathbf {x} ^ {k _ {j}}) \|,} \end{array}\tag{11.16}
$$

where Lemma $1 0 . 1 0 ( \mathrm { a } )$ was used in the last inequality. Since the expression in (11.16) goes to 0 as $j  \infty$ , it follows that $G _ { L _ { \mathrm { m i n } } } ( \bar { \bf x } ) = { \bf 0 }$ , which, by Theorem 10.7(b), implies that x¯ is a stationary point of problem (11.1). 

## 11.4.2 Convergence Analysis of the CBPG Method—The 64 Convex Case

We will now show a rate of convergence in function values of the CBPG method in the case where f is assumed to be convex and a certain boundedness property of the level sets of F holds. 

## Assumption 11.15.

(A) f is convex. 

(B) For any $\alpha > 0$ , there exists $R _ { \alpha } > 0$ such that 

$$
\max _ {\mathbf {x}, \mathbf {x} ^ {*} \in \mathbb {E}} \left\{\| \mathbf {x} - \mathbf {x} ^ {*} \|: F (\mathbf {x}) \leq \alpha , \mathbf {x} ^ {*} \in X ^ {*} \right\} \leq R _ {\alpha}.
$$

The analysis in the convex case is based on the following key lemma describing a recursive inequality relation of the sequence of function values. 

Lemma 11.16. Suppose that Assumptions 11.1 and 11.15 hold. Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the CBPG method for solving problem (11.1). Then for any $k \geq 0$ 

$$
F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \geq \frac {L _ {\mathrm{min}}}{2 p (L _ {f} + L _ {\mathrm{max}}) ^ {2} R ^ {2}} (F (\mathbf {x} ^ {k + 1}) - F _ {\mathrm{opt}}) ^ {2},
$$

where $\begin{array} { r } { R = R _ { F ( \mathbf { x } ^ { 0 } ) } , L _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { j = 1 , 2 , \dots , p } L _ { j } , a n d L _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { j = 1 , 2 , \dots , p } L _ { j } } \end{array}$ 

Proof. Let $\mathbf { x } ^ { * } \in X ^ { * }$ . By the definition of the CBPG method, for any $k \geq 0$ and $j \in \{ 1 , 2 , . . . , p \}$ 

$$
\mathbf {x} _ {j} ^ {k, j} = \operatorname{prox} _ {\frac {1}{L _ {j}} g _ {j}} \left(\mathbf {x} _ {j} ^ {k, j - 1} - \frac {1}{L _ {j}} \nabla_ {j} f (\mathbf {x} ^ {k, j - 1})\right).
$$

Thus, invoking the second prox theorem (Theorem 6.39), for any $\mathbf { y } \in \mathbb { E } _ { j }$ 

$$
g _ {j} (\mathbf {y}) \geq g _ {j} (\mathbf {x} _ {j} ^ {k, j}) + L _ {j} \left\langle \mathbf {x} _ {j} ^ {k, j - 1} - \frac {1}{L _ {j}} \nabla_ {j} f (\mathbf {x} ^ {k, j - 1}) - \mathbf {x} _ {j} ^ {k, j}, \mathbf {y} - \mathbf {x} _ {j} ^ {k, j} \right\rangle .
$$

By the definition of the auxiliary sequences given in (11.8), $\mathbf { x } _ { j } ^ { k , j - 1 } = \mathbf { x } _ { j } ^ { k } , \mathbf { x } _ { j } ^ { k , j } =$ $\mathbf { x } _ { j } ^ { k + 1 }$ , and therefore 

$$
g _ {j} (\mathbf {y}) \geq g _ {j} \left(\mathbf {x} _ {j} ^ {k + 1}\right) + L _ {j} \left\langle \mathbf {x} _ {j} ^ {k} - \frac {1}{L _ {j}} \nabla_ {j} f \left(\mathbf {x} ^ {k, j - 1}\right) - \mathbf {x} _ {j} ^ {k + 1}, \mathbf {y} - \mathbf {x} _ {j} ^ {k + 1} \right\rangle .
$$

Thus, in particular, if we substitute $\mathbf { y } = \mathbf { x } _ { j } ^ { * }$ 2 

$$
g _ {j} \left(\mathbf {x} _ {j} ^ {*}\right) \geq g _ {j} \left(\mathbf {x} _ {j} ^ {k + 1}\right) + L _ {j} \left\langle \mathbf {x} _ {j} ^ {k} - \frac {1}{L _ {j}} \nabla_ {j} f \left(\mathbf {x} ^ {k, j - 1}\right) - \mathbf {x} _ {j} ^ {k + 1}, \mathbf {x} _ {j} ^ {*} - \mathbf {x} _ {j} ^ {k + 1} \right\rangle .
$$

Summing the above inequality over $j = 1 , 2 , \dotsc , p$ yields the inequality 

$$
g \left(\mathbf {x} ^ {*}\right) \geq g \left(\mathbf {x} ^ {k + 1}\right) + \sum_ {j = 1} ^ {p} L _ {j} \left\langle \mathbf {x} _ {j} ^ {k} - \frac {1}{L _ {j}} \nabla_ {j} f \left(\mathbf {x} ^ {k, j - 1}\right) - \mathbf {x} _ {j} ^ {k + 1}, \mathbf {x} _ {j} ^ {*} - \mathbf {x} _ {j} ^ {k + 1} \right\rangle .\tag{11.17}
$$

We can now utilize the convexity of $f$ and write 

$$
\begin{array}{l} F (\mathbf {x} ^ {k + 1}) - F (\mathbf {x} ^ {*}) = f (\mathbf {x} ^ {k + 1}) - f (\mathbf {x} ^ {*}) + g (\mathbf {x} ^ {k + 1}) - g (\mathbf {x} ^ {*}) \\ \qquad \leq \langle \nabla f (\mathbf {x} ^ {k + 1}), \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \rangle + g (\mathbf {x} ^ {k + 1}) - g (\mathbf {x} ^ {*}) \\ \qquad = \sum_ {j = 1} ^ {p} \big \langle \nabla_ {j} f (\mathbf {x} ^ {k + 1}), \mathbf {x} _ {j} ^ {k + 1} - \mathbf {x} _ {j} ^ {*} \big \rangle + g (\mathbf {x} ^ {k + 1}) - g (\mathbf {x} ^ {*}), \end{array}
$$

which, combined with (11.17), implies 

$$
\begin{array}{l} F (\mathbf {x} ^ {k + 1}) - F (\mathbf {x} ^ {*}) \leq \sum_ {j = 1} ^ {p} \left\langle \nabla_ {j} f (\mathbf {x} ^ {k + 1}), \mathbf {x} _ {j} ^ {k + 1} - \mathbf {x} _ {j} ^ {*} \right\rangle \\ \qquad + \sum_ {j = 1} ^ {p} L _ {j} \left\langle \mathbf {x} _ {j} ^ {k} - \frac {1}{L _ {j}} \nabla_ {j} f (\mathbf {x} ^ {k, j - 1}) - \mathbf {x} _ {j} ^ {k + 1}, \mathbf {x} _ {j} ^ {k + 1} - \mathbf {x} _ {j} ^ {*} \right\rangle \\ \qquad = \sum_ {j = 1} ^ {p} \left\langle \nabla_ {j} f (\mathbf {x} ^ {k + 1}) - \nabla_ {j} f (\mathbf {x} ^ {k, j - 1}) + L _ {j} (\mathbf {x} _ {j} ^ {k} - \mathbf {x} _ {j} ^ {k + 1}), \mathbf {x} _ {j} ^ {k + 1} - \mathbf {x} _ {j} ^ {*} \right\rangle . \end{array}
$$

Using the Cauchy–Schwarz and triangle inequalities, we can conclude that 

$$
\begin{array}{l} F (\mathbf {x} ^ {k + 1}) - F (\mathbf {x} ^ {*}) \leq \sum_ {j = 1} ^ {p} \left(\| \nabla_ {j} f (\mathbf {x} ^ {k + 1}) - \nabla_ {j} f (\mathbf {x} ^ {k, j - 1}) \| + L _ {j} \| \mathbf {x} _ {j} ^ {k} - \mathbf {x} _ {j} ^ {k + 1} \|\right) \| \mathbf {x} _ {j} ^ {k + 1} - \mathbf {x} _ {j} ^ {*} \| \\ \leq \sum_ {j = 1} ^ {p} \left(\| \nabla f (\mathbf {x} ^ {k + 1}) - \nabla f (\mathbf {x} ^ {k, j - 1}) \| + L _ {j} \| \mathbf {x} _ {j} ^ {k} - \mathbf {x} _ {j} ^ {k + 1} \|\right) \| \mathbf {x} _ {j} ^ {k + 1} - \mathbf {x} _ {j} ^ {*} \| \\ \leq \sum_ {\substack {j = 1}} ^ {p} \left(L _ {f} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k, j - 1} \| + L _ {\max} \| \mathbf {x} ^ {k} - \mathbf {x} ^ {k + 1} \|\right) \| \mathbf {x} _ {j} ^ {k + 1} - \mathbf {x} _ {j} ^ {*} \| \\ \leq (L _ {f} + L _ {\max}) \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \| \sum_ {j = 1} ^ {p} \| \mathbf {x} _ {j} ^ {k + 1} - \mathbf {x} _ {j} ^ {*} \|. \end{array}
$$

Hence, 

$$
\begin{array}{l} (F (\mathbf {x} _ {k + 1}) - F (\mathbf {x} ^ {*})) ^ {2} \leq (L _ {f} + L _ {\max}) ^ {2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \| ^ {2} \left(\sum_ {j = 1} ^ {p} \| \mathbf {x} _ {j} ^ {k + 1} - \mathbf {x} _ {j} ^ {*} \|\right) ^ {2} \\ \leq p (L _ {f} + L _ {\max}) ^ {2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \| ^ {2} \sum_ {j = 1} ^ {p} \| \mathbf {x} _ {j} ^ {k + 1} - \mathbf {x} _ {j} ^ {*} \| ^ {2} \\ = p (L _ {f} + L _ {\max}) ^ {2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \| ^ {2} \cdot \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2} \\ \leq p (L _ {f} + L _ {\max}) ^ {2} R _ {F (\mathbf {x} ^ {0})} ^ {2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \| ^ {2}, \end{array}\tag{11.18}
$$

where the last inequality follows by the monotonicity of the sequence of function values (Corollary 11.12) and Assumption 11.15(B). Combining (11.18) with (11.11), we obtain that 

$$
F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \geq \frac {L _ {\mathrm{min}}}{2} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {k} \| ^ {2} \geq \frac {L _ {\mathrm{min}}}{2 p (L _ {f} + L _ {\mathrm{max}}) ^ {2} R ^ {2}} (F (\mathbf {x} ^ {k + 1}) - F (\mathbf {x} ^ {*})) ^ {2},
$$

where $R = R _ { F ( \mathbf { x } ^ { 0 } ) }$ □ 

To derive the rate of convergence in function values, we will use the following lemma on the convergence of nonnegative scalar sequences satisfying a certain recursive inequality relation. The result resembles the one derived in Lemma 10.70, but the recursive inequality is diferent. 

Lemma 11.17. Let $\{ a _ { k } \} _ { k \ge 0 }$ be a nonnegative sequence of real numbers satisfying 

$$
a _ {k} - a _ {k + 1} \geq \frac {1}{\gamma} a _ {k + 1} ^ {2}, \quad k = 0, 1, \ldots ,\tag{11.19}
$$

for some $\gamma > 0$ . Then for any $n \geq 2$ 

$$
a _ {n} \leq \max \left\{\left(\frac {1}{2}\right) ^ {(n - 1) / 2} a _ {0}, \frac {4 \gamma}{n - 1} \right\}.\tag{11.20}
$$

In addition, for any $\varepsilon > 0$ , if $n \geq 2$ satisfies 

$$
n \geq \max \left\{\frac {2}{\log (2)} (\log (a _ {0}) + \log (1 / \varepsilon)), \frac {4 \gamma}{\varepsilon} \right\} + 1,
$$

then $a _ { n } \leq \varepsilon$ 

Proof. Let $n \geq 2$ . If $a _ { n } = 0$ , then (11.20) is trivial. We can thus assume that $a _ { n } > 0$ , from which it follows that $a _ { 1 } , a _ { 2 } , \dotsc , a _ { n - 1 } > 0$ . For any $k \in \{ 0 , 1 , . . . , n { - } 1 \}$ 

$$
{\frac {1}{a _ {k + 1}}} - {\frac {1}{a _ {k}}} = {\frac {a _ {k} - a _ {k + 1}}{a _ {k} a _ {k + 1}}} \geq {\frac {1}{\gamma}} {\frac {a _ {k + 1}}{a _ {k}}}.\tag{11.21}
$$

For each $k ,$ , there are two options: 

(i) $\begin{array} { r } { \frac { a _ { k + 1 } } { a _ { k } } \leq \frac { 1 } { 2 } , } \end{array}$ 

(ii) $\begin{array} { r } { \frac { a _ { k + 1 } } { a _ { k } } > \frac { 1 } { 2 } } \end{array}$ 

By (11.21), under option (ii) we have 

$$
\frac {1}{a _ {k + 1}} - \frac {1}{a _ {k}} \geq \frac {1}{2 \gamma}.
$$

Suppose that n is a positive even integer. If there are at least $\textstyle { \frac { n } { 2 } }$ indices (out of $k = 0 , 1 , \ldots , n - 1 )$ for which option (ii) occurs, then 

$$
\frac {1}{a _ {n}} \geq \frac {n}{4 \gamma},
$$

and hence 

$$
a _ {n} \leq \frac {4 \gamma}{n}.
$$

On the other hand, if this is not the case, then there are at least $\textstyle { \frac { n } { 2 } }$ indices for which option (i) occurs, and consequently 

$$
a _ {n} \leq \left(\frac {1}{2}\right) ^ {n / 2} a _ {0}.
$$

We therefore obtain that in any case, for an even n, 

$$
a _ {n} \leq \max \left\{\left(\frac {1}{2}\right) ^ {n / 2} a _ {0}, \frac {4 \gamma}{n} \right\}.\tag{11.22}
$$

If $n \geq 3$ is a positive odd integer, then 

$$
a _ {n} \leq a _ {n - 1} \leq \max \left\{\left(\frac {1}{2}\right) ^ {(n - 1) / 2} a _ {0}, \frac {4 \gamma}{n - 1} \right\}.\tag{11.23}
$$

Since the right-hand side of (11.23) is larger than the right-hand side of (11.22), the result (11.20) follows. Let $n \geq 2$ . To guarantee that the inequality $a _ { n } \leq \varepsilon$ holds, it is suficient that the inequality 

$$
\max \left\{\left(\frac {1}{2}\right) ^ {(n - 1) / 2} a _ {0}, \frac {4 \gamma}{n - 1} \right\} \leq \varepsilon
$$

will hold, meaning that the following two inequalities will be satisfied: 

$$
\left(\frac {1}{2}\right) ^ {(n - 1) / 2} a _ {0} \leq \varepsilon , \frac {4 \gamma}{n - 1} \leq \varepsilon .
$$

These inequalities are obviously equivalent to 

$$
n \geq \frac {2}{\log (2)} (\log (a _ {0}) + \log (1 / \varepsilon)) + 1, n \geq \frac {4 \gamma}{\varepsilon} + 1.
$$

Therefore, if 

$$
n \geq \max \left\{\frac {2}{\log (2)} (\log (a _ {0}) + \log (1 / \varepsilon)), \frac {4 \gamma}{\varepsilon} \right\} + 1,
$$

then the inequality $a _ { n } \leq \varepsilon$ is guaranteed. 

Combining Lemmas 11.16 and 11.17, we can establish an $O ( 1 / k )$ rate of convergence in function values of the sequence generated by the CBPG method, as well as a complexity result. 

Theorem 11.18 $( O ( 1 / k )$ rate of convergence of CBPG). Suppose that Assumptions 11.1 and 11.15 hold. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the CBPG method for solving problem (11.1). For any $k \geq 2$ 2 

$$
F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} \leq \max \left\{\left(\frac {1}{2}\right) ^ {(k - 1) / 2} (F (\mathbf {x} ^ {0}) - F _ {\mathrm{opt}}), \frac {8 p (L _ {f} + L _ {\max}) ^ {2} R ^ {2}}{L _ {\min} (k - 1)} \right\},\tag{11.24}
$$

where $L _ { \mathrm { m i n } } = \mathrm { m i n } _ { i = 1 , 2 , . . . , p } L _ { i } , L _ { \mathrm { m a x } } = \mathrm { m a x } _ { i = 1 , 2 , . . . , p } L _ { i }$ , and $R = R _ { F ( { \bf x } ^ { 0 } ) }$ . In addition, if $n \geq 2$ satisfies 

$$
n \geq \max \left\{\frac {2}{\log (2)} (\log (F (\mathbf {x} ^ {0}) - F _ {\mathrm{opt}}) + \log (1 / \varepsilon)), \frac {8 p (L _ {f} + L _ {\max}) ^ {2} R ^ {2}}{L _ {\min} \varepsilon} \right\} + 1,
$$

then $F ( \mathbf { x } ^ { n } ) - F _ { \mathrm { o p t } } \leq \varepsilon .$ 

Proof. Denote $a _ { k } = F ( \mathbf { x } ^ { k } ) - F _ { \mathrm { o p t } }$ . Then by Lemma 11.16, 

$$
a _ {k} - a _ {k + 1} \geq \frac {1}{D} a _ {k + 1} ^ {2},
$$

where $\begin{array} { r } { D = \frac { 2 p ( L _ { f } + L _ { \mathrm { m a x } } ) ^ { 2 } R ^ { 2 } } { L _ { \mathrm { m i n } } } } \end{array}$ . The result now follows by invoking Lemma 11.17 with $\gamma = D$ 

Remark 11.19 (index order). The analysis of the CBPG method was done under the assumption that the index selection strategy is cyclic. However, it is easy to see that the same analysis, and consequently the main results (Theorems 11.14 and 11.18), hold for any index selection strategy in which each block is updated exactly once between consecutive iterations. One example of such an index selection strategy is the “cyclic shufle” order in which the order of blocks is picked at the beginning of each iteration by a random permutation; in a sense, this is a “quasi-randomized” strategy. In the next section we will study a fully randomized approach. 

We end this section by showing that for convex diferentiable functions (over the entire space) block Lipschitz continuity (Assumption 11.1(D)) implies that the function is L-smooth (Assumption 11.1(C)) with L being the sum of the block Lipschitz constants. This means that in this situation we can actually drop Assumption 11.1(C). 

Theorem 11.20.<sup>65</sup> Let $\phi : \mathbb { E }  \mathbb { R } \ ( \mathbb { E } = \mathbb { E } _ { 1 } \times \mathbb { E } _ { 2 } \times \cdot \cdot \cdot \times \mathbb { E } _ { p } )$ be a convex function satisfying the following assumptions: 

(A) $\phi$ is diferentiable over $\mathbb { E } ,$ 

(B) there exist $L _ { 1 } , L _ { 2 } , \dots , L _ { p } > 0$ such that for any $i \in \{ 1 , 2 , \ldots , p \}$ it holds that 

$$
\left\| \nabla_ {i} \phi (\mathbf {x}) - \nabla_ {i} \phi (\mathbf {x} + \mathcal {U} _ {i} (\mathbf {d})) \right\| \leq L _ {i} \| \mathbf {d} \|
$$

for all $\mathbf { x } \in \mathbb { E }$ and d $\in \mathbb { E } _ { i }$ 

Then $\phi$ is L-smooth with $L = L _ { 1 } + L _ { 2 } + \cdot \cdot \cdot + L _ { p } $ 

Proof. Let $\mathbf { y } \in \mathbb { E }$ . Define the function 

$$
f (\mathbf {x}) = \phi (\mathbf {x}) - \phi (\mathbf {y}) - \langle \nabla \phi (\mathbf {y}), \mathbf {x} - \mathbf {y} \rangle .\tag{11.25}
$$

Then it is immediate to show that $f$ also satisfies properties (A) and (B). In addition, the convexity of $\phi$ implies the convexity of $f$ as well as the fact that f is nonnegative. 

Invoking Lemma 11.9 with $g _ { 1 } = g _ { 2 } = \cdot \cdot \cdot = g _ { p } \equiv 0$ , we obtain that for all $i \in$ $\{ 1 , 2 , \ldots , p \}$ and $\mathbf { x } \in \mathbb { E }$ 

$$
f (\mathbf {x}) - f \left(\mathbf {x} - \frac {1}{L _ {i}} \mathcal {U} _ {i} (\nabla_ {i} f (\mathbf {x}))\right) \geq \frac {1}{2 L _ {i}} \| \nabla_ {i} f (\mathbf {x}) \| ^ {2},
$$

which, along with the nonnegativity of $f _ { : }$ , implies that 

$$
f (\mathbf {x}) \geq \frac {1}{2 L _ {i}} \| \nabla_ {i} f (\mathbf {x}) \| ^ {2}.
$$

Since the last inequality holds for any $i \in \{ 1 , 2 , \ldots , p \}$ , it follows that 

$$
\begin{array}{r l} & f (\mathbf {x}) \geq \max _ {i = 1, 2, \ldots , p} \left\{\frac {1}{2 L _ {i}} \| \nabla_ {i} f (\mathbf {x}) \| ^ {2} \right\} \geq \sum_ {i = 1} ^ {p} \frac {L _ {i}}{\sum_ {j = 1} ^ {p} L _ {j}} \frac {1}{2 L _ {i}} \| \nabla_ {i} f (\mathbf {x}) \| ^ {2} \\ & \quad = \frac {1}{2 (\sum_ {j = 1} ^ {p} L _ {j})} \| \nabla f (\mathbf {x}) \| ^ {2}. \end{array}
$$

Plugging the expression (11.25) for f into the above inequality, we obtain 

$$
\phi (\mathbf {x}) \geq \phi (\mathbf {y}) + \langle \nabla \phi (\mathbf {y}), \mathbf {x} - \mathbf {y} \rangle + \frac {1}{2 (\sum_ {j = 1} ^ {p} L _ {j})} \| \nabla \phi (\mathbf {x}) - \nabla \phi (\mathbf {y}) \| ^ {2}.
$$

Since the above inequality holds for any $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ , it follows by Theorem 5.8 (equivalence between (i) and (iii)) that $\phi$ is $( L _ { 1 } + L _ { 2 } + \cdots + L _ { p } )$ -smooth. 

## 11.5 The Randomized Block Proximal Gradient Method<sup>66</sup>

In this section we will analyze a version of the block proximal gradient method in which at each iteration a prox-grad step is performed at a randomly chosen block. The analysis is made under Assumption 11.21 given below. Note that at this point we do not assume that $f$ is convex, but the main convergence result, Theorem 11.25, will require the convexity of $f .$ . 

## Assumption 11.21.

(A) $g _ { i } : \mathbb { E } _ { i }  ( - \infty , \infty ]$ is proper closed and convex for any $i \in \{ 1 , 2 , \ldots , p \}$ 

(B) $f : \mathbb { E } \to ( - \infty , \infty ]$ is proper closed and convex, dom $( g ) \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ , and f is diferentiable over int(dom(f)). 

(C) There exist $L _ { 1 } , L _ { 2 } , \dots , L _ { p } > 0$ such that for any $i \in \{ 1 , 2 , \ldots , p \}$ it holds that 

$$
\left\| \nabla_ {i} f (\mathbf {x}) - \nabla_ {i} f (\mathbf {x} + \mathcal {U} _ {i} (\mathbf {d})) \right\| \leq L _ {i} \| \mathbf {d} \|
$$

for all $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ and d $\in \mathbb { E } _ { i }$ for which $\mathbf { x } + { \mathcal { U } } _ { i } ( \mathbf { d } ) \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ 

(D) The optimal set of problem (11.1) is nonempty and denoted by $X ^ { * }$ . The optimal value is denoted by $F _ { \mathrm { o p t } }$ 

Initialization: pick $\mathbf { x } ^ { 0 } = ( \mathbf { x } _ { 1 } ^ { 0 } , \mathbf { x } _ { 2 } ^ { 0 } , \ldots , \mathbf { x } _ { p } ^ { 0 } ) \in$ int(dom(f)). General step: for any $k = 0 , 1 , 2 , . . .$ . execute the following steps: 

(a) pick $i _ { k } \in \{ 1 , 2 , . . . , p \}$ randomly via a uniform distribution; 

(b) $\mathbf { x } ^ { k + 1 } = \mathbf { x } ^ { k } + \mathcal { U } _ { i _ { k } } \big ( T _ { L _ { i _ { k } } } ^ { i _ { k } } \big ( \mathbf { x } ^ { k } \big ) - \mathbf { x } _ { i _ { k } } ^ { k } \big ) .$ 

Remark 11.22. Step (b) of the algorithm can also be written as 

$$
\mathbf {x} ^ {k + 1} = \mathbf {x} ^ {k} - \frac {1}{L _ {i _ {k}}} \mathcal {U} _ {i _ {k}} (G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k})).
$$

From the point of view of computational complexity, loosely speaking, each p iterations of the RBPG method are comparable to one iteration of the CBPG method. 

Using the block suficient decrease lemma (Lemma 11.9), it is easy to show a suficient decrease property of the RBPG method. 

Theorem 11.23 (suficient decrease of the RBPG method). Suppose that Assumption 11.21 holds, and let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the RBPG method. Then for any $k \geq 0$ , 

$$
F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \geq \frac {1}{2 L _ {i _ {k}}} \| G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}) \| ^ {2}.
$$

Proof. Invoke Lemma 11.9 with $\mathbf { x } = \mathbf { x } ^ { k }$ and $i = i _ { k }$ □ 

Remark 11.24. A direct consequence of Theorem 11.23 is that the sequence of function values $\{ F ( \mathbf { x } ^ { k } ) \} _ { k \geq 0 }$ generated by the RBPG method is nonincreasing. As a result, it is also correct that the sequence of expected function values 

$$
\left\{\mathsf {E} _ {i _ {0}, \dots , i _ {k - 1}} \left(F \left(\mathbf {x} ^ {k}\right)\right) \right\} _ {k \geq 0}
$$

is nonincreasing. 

In our analysis the following notation is used: 

$\xi _ { k - 1 } \equiv \left\{ i _ { 0 } , i _ { 1 } , \dots , i _ { k - 1 } \right\}$ is a multivariate random variable. 

• In addition to the underlying Euclidean norm of the space $\mathbb { E } .$ , we define the following weighted norm: 

$$
\| \mathbf {x} \| _ {L} \equiv \sqrt {\sum_ {i = 1} ^ {p} L _ {i} \| \mathbf {x} _ {i} \| ^ {2}}
$$

and its dual norm 

$$
\| \mathbf {x} \| _ {L, *} = \sqrt {\sum_ {i = 1} ^ {p} \frac {1}{L _ {i}} \| \mathbf {x} _ {i} \| ^ {2}}.
$$

• We will consider the following variation of the gradient mapping: 

$$
\widetilde {G} (\mathbf {x}) = (G _ {L _ {1}} ^ {1} (\mathbf {x} ^ {k}), G _ {L _ {2}} ^ {2} (\mathbf {x} ^ {k}), \ldots , G _ {L _ {p}} ^ {p} (\mathbf {x} ^ {k})).\tag{11.26}
$$

Obviously, if $L _ { 1 } = L _ { 2 } = \cdot \cdot \cdot = L _ { p } = L$ , then $\widetilde { G } ( \mathbf { x } ) = G _ { L } ( \mathbf { x } )$ 

The main convergence result will now be stated and proved. 

Theorem 11.25 $( O ( 1 / k )$ rate of convergence of the RBPG method). Suppose that Assumption 11.21 holds and that f is convex. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the RBPG method for solving problem (11.1). Let $\mathbf { x } ^ { * } \in X ^ { * }$ . Then for any $k \geq 0$ , 

$$
\mathsf {E} _ {\xi_ {k}} (F (\mathbf {x} ^ {k + 1})) - F _ {\mathrm{opt}} \leq \frac {p}{p + k + 1} \left(\frac {1}{2} \| \mathbf {x} ^ {0} - \mathbf {x} ^ {*} \| _ {L} ^ {2} + F (\mathbf {x} ^ {0}) - F _ {\mathrm{opt}}\right).\tag{11.27}
$$

Proof. Let $\mathbf { x } ^ { * } \in X ^ { * }$ . We denote for any $n \geq 0 , r _ { n } \equiv \left\| \mathbf { x } ^ { n } - \mathbf { x } ^ { * } \right\| _ { L }$ . Then for any $k \geq 0$ 

$$
\begin{array}{r l} & {r _ {k + 1} ^ {2} = \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| _ {L} ^ {2}} \\ & {\quad = \left\| \mathbf {x} ^ {k} - \frac {1}{L _ {i _ {k}}} \mathcal {U} _ {i _ {k}} \left(G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k})\right) - \mathbf {x} ^ {*} \right\| _ {L} ^ {2}} \\ & {\quad = \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| _ {L} ^ {2} - \frac {2}{L _ {i _ {k}}} L _ {i _ {k}} \langle G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}), \mathbf {x} _ {i _ {k}} ^ {k} - \mathbf {x} _ {i _ {k}} ^ {*} \rangle + \frac {L _ {i _ {k}}}{L _ {i _ {k}} ^ {2}} \| G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}) \| ^ {2}} \\ & {\quad = \| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| _ {L} ^ {2} - 2 \langle G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}), \mathbf {x} _ {i _ {k}} ^ {k} - \mathbf {x} _ {i _ {k}} ^ {*} \rangle + \frac {1}{L _ {i _ {k}}} \| G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}) \| ^ {2}} \\ & {\quad = r _ {k} ^ {2} - 2 \langle G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}), \mathbf {x} _ {i _ {k}} ^ {k} - \mathbf {x} _ {i _ {k}} ^ {*} \rangle + \frac {1}{L _ {i _ {k}}} \| G _ {L _ {i _ {k}}} (\mathbf {x} ^ {k}) \| ^ {2}.} \end{array}
$$

Taking expectation w.r.t. $i _ { k }$ , we obtain (using the notation (11.26)) 

$$
\begin{array}{l} \mathsf {E} _ {i _ {k}} \left(\frac {1}{2} r _ {k + 1} ^ {2}\right) = \frac {1}{2} r _ {k} ^ {2} - \frac {1}{p} \sum_ {i = 1} ^ {p} \langle G _ {L _ {i}} ^ {i} (\mathbf {x} _ {k}), \mathbf {x} _ {i} ^ {k} - \mathbf {x} _ {i} ^ {*} \rangle + \frac {1}{2 p} \sum_ {i = 1} ^ {p} \frac {1}{L _ {i}} \| G _ {L _ {i}} ^ {i} (\mathbf {x} ^ {k}) \| ^ {2} \\ = \frac {1}{2} r _ {k} ^ {2} - \frac {1}{p} \langle \widetilde {G} (\mathbf {x} ^ {k}), \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \rangle + \frac {1}{2 p} \| \widetilde {G} (\mathbf {x} ^ {k}) \| _ {L, *} ^ {2}. \end{array} \tag {1}\tag{11.28}
$$

By the block descent lemma (Lemma 11.8), 

$$
\begin{array}{r l} & f (\mathbf {x} ^ {k + 1}) = f \left(\mathbf {x} ^ {k} - \frac {1}{L _ {i _ {k}}} \mathcal {U} _ {i _ {k}} (G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}))\right) \\ & \qquad \leq f (\mathbf {x} ^ {k}) - \frac {1}{L _ {i _ {k}}} \langle \nabla_ {i _ {k}} f (\mathbf {x} ^ {k}), G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}) \rangle + \frac {1}{2 L _ {i _ {k}}} \| G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}) \| ^ {2}. \end{array}
$$

Hence, 

$$
F (\mathbf {x} ^ {k + 1}) \leq f (\mathbf {x} ^ {k}) - \frac {1}{L _ {i _ {k}}} \langle \nabla_ {i _ {k}} f (\mathbf {x} ^ {k}), G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}) \rangle + \frac {1}{2 L _ {i _ {k}}} \| G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}) \| ^ {2} + g (\mathbf {x} ^ {k + 1}).
$$

Taking expectation of both sides of the last inequality w.r.t. $i _ { k }$ , we obtain 

$$
\mathsf {E} _ {i _ {k}} (F (\mathbf {x} ^ {k + 1})) \leq f (\mathbf {x} ^ {k}) - \frac {1}{p} \sum_ {i = 1} ^ {p} \frac {1}{L _ {i}} \langle \nabla_ {i} f (\mathbf {x} ^ {k}), G _ {L _ {i}} ^ {i} (\mathbf {x} ^ {k}) \rangle + \frac {1}{2 p} \| \widetilde {G} (\mathbf {x} ^ {k}) \| _ {L, *} ^ {2} + \mathsf {E} _ {i _ {k}} (g (\mathbf {x} ^ {k + 1})).\tag{11.29}
$$

Since $\begin{array} { r } { \mathbf { x } _ { i _ { k } } ^ { k + 1 } = \mathbf { x } _ { i _ { k } } ^ { k } - \frac { 1 } { L _ { i _ { k } } } G _ { L _ { i _ { k } } } ^ { i _ { k } } ( \mathbf { x } ^ { k } ) = \mathrm { p r o x } _ { \frac { 1 } { L _ { i _ { k } } } g _ { i _ { k } } } \left( \mathbf { x } _ { i _ { k } } ^ { k } - \frac { 1 } { L _ { i _ { k } } } \nabla _ { i _ { k } } f ( \mathbf { x } ^ { k } ) \right) } \end{array}$ , it follows by the second prox theorem (Theorem 6.39) that 

$$
\begin{array}{r l} & g _ {i _ {k}} (\mathbf {x} _ {i _ {k}} ^ {*}) \geq g _ {i _ {k}} \left(\mathbf {x} _ {i _ {k}} ^ {k} - \frac {1}{L _ {i _ {k}}} G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k})\right) \\ & \qquad + L _ {i _ {k}} \left\langle \mathbf {x} _ {i _ {k}} ^ {k} - \frac {1}{L _ {i _ {k}}} \nabla_ {i _ {k}} f (\mathbf {x} ^ {k}) - \mathbf {x} _ {i _ {k}} ^ {k} + \frac {1}{L _ {i _ {k}}} G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}), \mathbf {x} _ {i _ {k}} ^ {*} - \mathbf {x} _ {i _ {k}} ^ {k} + \frac {1}{L _ {i _ {k}}} G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}) \right\rangle . \end{array}
$$

That is, 

$$
\begin{array}{r l} & g _ {i _ {k}} (\mathbf {x} _ {i _ {k}} ^ {*}) \geq g _ {i _ {k}} \left(\mathbf {x} _ {i _ {k}} ^ {k} - \frac {1}{L _ {i _ {k}}} G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k})\right) \\ & \qquad + \left\langle - \nabla_ {i _ {k}} f (\mathbf {x} ^ {k}) + G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}), \mathbf {x} _ {i _ {k}} ^ {*} - \mathbf {x} _ {i _ {k}} ^ {k} + \frac {1}{L _ {i _ {k}}} G _ {L _ {i _ {k}}} ^ {i _ {k}} (\mathbf {x} ^ {k}) \right\rangle . \end{array}\tag{11.30}
$$

Note that 

$$
\mathsf {E} _ {i _ {k}} (g _ {i _ {k}} (\mathbf {x} _ {i _ {k}} ^ {*})) = \frac {1}{p} g (\mathbf {x} ^ {*}),\tag{11.31}
$$

$$
\mathsf {E} _ {i _ {k}} (g (\mathbf {x} ^ {k + 1})) = \frac {p - 1}{p} g (\mathbf {x} ^ {k}) + \frac {1}{p} \sum_ {i = 1} ^ {p} g _ {i} \left(\mathbf {x} _ {i} ^ {k} - \frac {1}{L _ {i}} G _ {L _ {i}} ^ {i} (\mathbf {x} ^ {k})\right).\tag{11.32}
$$

Taking expectation w.r.t. $i _ { k }$ in (11.30) and plugging in the relations (11.31) and (11.32) leads to the following inequality: 

$$
\begin{array}{l} \frac {1}{p} g (\mathbf {x} ^ {*}) \geq \mathsf {E} _ {i _ {k}} (g (\mathbf {x} ^ {k + 1})) - \frac {p - 1}{p} g (\mathbf {x} ^ {k}) + \frac {1}{p} \langle - \nabla f (\mathbf {x} ^ {k}) + \widetilde {G} (\mathbf {x} ^ {k}), \mathbf {x} ^ {*} - \mathbf {x} ^ {k} \rangle \\ \qquad - \frac {1}{p} \sum_ {i = 1} ^ {p} \frac {1}{L _ {i}} \langle \nabla_ {i} f (\mathbf {x} ^ {k}), G _ {L _ {i}} ^ {i} (\mathbf {x} ^ {k}) \rangle + \frac {1}{p} \| \widetilde {G} (\mathbf {x} ^ {k}) \| _ {L, *} ^ {2}. \end{array}
$$

The last inequality can be equivalently written as 

$$
\begin{array}{r l} & {\mathsf {E} _ {i _ {k}} (g (\mathbf {x} ^ {k + 1})) - \frac {1}{p} \sum_ {i = 1} ^ {p} \frac {1}{L _ {i}} \langle \nabla_ {i} f (\mathbf {x} ^ {k}), G _ {L _ {i}} ^ {i} (\mathbf {x} ^ {k}) \rangle} \\ & {\leq \frac {1}{p} g (\mathbf {x} ^ {*}) + \frac {p - 1}{p} g (\mathbf {x} ^ {k}) + \frac {1}{p} \langle \nabla f (\mathbf {x} ^ {k}) - \widetilde {G} (\mathbf {x} ^ {k}), \mathbf {x} ^ {*} - \mathbf {x} ^ {k} \rangle - \frac {1}{p} \| \widetilde {G} (\mathbf {x} ^ {k}) \| _ {L, *} ^ {2}.} \end{array}
$$

Plugging the last inequality into (11.29) we obtain that 

$$
\begin{array}{r l} & {\mathsf {E} _ {i _ {k}} (F (\mathbf {x} ^ {k + 1})) \leq f (\mathbf {x} ^ {k}) - \frac {1}{2 p} \| \tilde {G} (\mathbf {x} ^ {k}) \| _ {L, *} ^ {2} + \frac {1}{p} g (\mathbf {x} ^ {*}) + \frac {1}{p} \langle \nabla f (\mathbf {x} ^ {k}) - \widetilde {G} (\mathbf {x} ^ {k}), \mathbf {x} ^ {*} - \mathbf {x} ^ {k} \rangle} \\ & {\qquad + \frac {p - 1}{p} g (\mathbf {x} ^ {k}),} \end{array}
$$

which, along with the gradient inequality $\langle \nabla f ( { \mathbf { x } } ^ { k } ) , { \mathbf { x } } ^ { * } - { \mathbf { x } } ^ { k } \rangle \leq f ( { \mathbf { x } } ^ { * } ) - f ( { \mathbf { x } } ^ { k } )$ , implies 

$$
\mathsf {E} _ {i _ {k}} (F (\mathbf {x} ^ {k + 1})) \leq \frac {p - 1}{p} F (\mathbf {x} ^ {k}) + \frac {1}{p} F (\mathbf {x} ^ {*}) - \frac {1}{2 p} \| \widetilde {G} (\mathbf {x} ^ {k}) \| _ {L, *} ^ {2} - \frac {1}{p} \langle \widetilde {G} (\mathbf {x} ^ {k}), \mathbf {x} ^ {*} - \mathbf {x} ^ {k} \rangle .
$$

The last inequality, combined with (11.28), yields the relation 

$$
\mathsf {E} _ {i _ {k}} \left(\frac {1}{2} r _ {k + 1} ^ {2}\right) \leq \frac {1}{2} r _ {k} ^ {2} + \frac {p - 1}{p} F (\mathbf {x} ^ {k}) + \frac {1}{p} F (\mathbf {x} ^ {*}) - \mathsf {E} _ {i _ {k}} (F (\mathbf {x} ^ {k + 1})),
$$

which can be rearranged as 

$$
\mathsf {E} _ {i _ {k}} \left(\frac {1}{2} r _ {k + 1} ^ {2} + F (\mathbf {x} ^ {k + 1}) - F _ {\mathrm{opt}}\right) \leq \left(\frac {1}{2} r _ {k} ^ {2} + F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}}\right) - \frac {1}{p} (F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}}).
$$

Taking expectation over $\xi _ { k - 1 }$ of both sides we obtain (where we make the convention that the expression $\mathbb { E } _ { \xi _ { - 1 } } ( F ( \mathbf { x } ^ { 0 } ) )$ means $F ( \mathbf { x } ^ { 0 } ) )$ 

$$
\begin{array}{c} \mathsf {E} _ {\xi_ {k}} \left(\frac {1}{2} r _ {k + 1} ^ {2} + F (\mathbf {x} ^ {k + 1}) - F _ {\mathrm{opt}}\right) \leq \mathsf {E} _ {\xi_ {k - 1}} \left(\frac {1}{2} r _ {k} ^ {2} + F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}}\right) \\ - \frac {1}{p} (\mathsf {E} _ {\xi_ {k - 1}} (F (\mathbf {x} ^ {k})) - F _ {\mathrm{opt}}). \end{array}
$$

We can thus conclude that 

$$
\begin{array}{r l} & {\mathsf {E} _ {\xi_ {k}} (F (\mathbf {x} ^ {k + 1})) - F _ {\mathrm{opt}} \leq \mathsf {E} _ {\xi_ {k}} \left(\frac {1}{2} r _ {k + 1} ^ {2} + F (\mathbf {x} ^ {k + 1}) - F _ {\mathrm{opt}}\right)} \\ & {\qquad \leq \frac {1}{2} r _ {0} ^ {2} + F (\mathbf {x} ^ {0}) - F _ {\mathrm{opt}} - \frac {1}{p} \sum_ {j = 0} ^ {k} \left(\mathsf {E} _ {\xi_ {j - 1}} (F (\mathbf {x} ^ {j})) - F _ {\mathrm{opt}}\right),} \end{array}
$$

which, together with the monotonicity of the sequence of expected values $\{ \mathsf { E } _ { \xi _ { k - 1 } } ( F ( \mathbf { x } ^ { k } ) ) \} _ { k \geq 0 }$ (see Remark 11.24), implies that 

$$
\mathsf {E} _ {\xi_ {k}} (F (\mathbf {x} ^ {k + 1})) - F _ {\mathrm{opt}} \leq \frac {1}{2} r _ {0} ^ {2} + F (\mathbf {x} ^ {0}) - F _ {\mathrm{opt}} - \frac {k + 1}{p} \left(\mathsf {E} _ {\xi_ {k}} (F (\mathbf {x} ^ {k + 1})) - F _ {\mathrm{opt}}\right).\tag{11.33}
$$

The desired result (11.27) follows immediately from (11.33). 

# Chapter 12 Dual-Based Proximal Gradient Methods

Underlying Spaces: In this chapter, all the underlying spaces are Euclidean. 

## 12.1 The Primal and Dual Models

The main model discussed in this chapter is 

$$
f _ {\mathrm{opt}} = \min _ {\mathbf {x} \in \mathbb {E}} \left\{f (\mathbf {x}) + g (\mathcal {A} (\mathbf {x})) \right\},\tag{12.1}
$$

where the following assumptions are made. 

Assumption 12.1. 

(A) $f : \mathbb { E } \to ( - \infty , + \infty ]$ is proper closed and σ-strongly convex $( \sigma > 0 )$ 

(B) $g : \mathbb { V } \to ( - \infty , + \infty ]$ is proper closed and convex. 

(C) $\mathcal { A } : \mathbb { E }  \mathbb { V }$ is a linear transformation. 

(D) There exists ${ \hat { \mathbf { x } } } \in \operatorname { r i } ( \operatorname { d o m } ( f ) )$ and $\hat { \mathbf { z } } \in \operatorname { r i } ( \operatorname { d o m } ( g ) )$ such that $\boldsymbol { \mathcal { A } } ( \hat { \mathbf { x } } ) = \hat { \mathbf { z } }$ 

Under Assumption 12.1 the function $\mathbf { x } \mapsto f ( \mathbf { x } ) + g ( \mathcal { A } ( \mathbf { x } ) )$ is proper closed and σ-strongly convex, and hence, by Theorem $5 . 2 5 ( \mathrm { a } )$ , problem (12.1) has a unique optimal solution, which we denote throughout this chapter by $\mathbf { x } ^ { * }$ 

To construct a dual problem to (12.1), we first rewrite it in the form 

$$
\begin{array}{l l} \min _ {\mathbf {x}, \mathbf {z}} & f (\mathbf {x}) + g (\mathbf {z}) \\ \text {s.t.} & \mathcal {A} (\mathbf {x}) - \mathbf {z} = \mathbf {0}. \end{array}\tag{12.2}
$$

Associating a Lagrange dual vector $\mathbf { y } \in \mathbb { V }$ to the equality constraints in (12.2), the Lagrangian can be written as 

$$
L (\mathbf {x}, \mathbf {z}; \mathbf {y}) = f (\mathbf {x}) + g (\mathbf {z}) - \langle \mathbf {y}, \mathcal {A} (\mathbf {x}) - \mathbf {z} \rangle = f (\mathbf {x}) + g (\mathbf {z}) - \langle \mathcal {A} ^ {T} (\mathbf {y}), \mathbf {x} \rangle + \langle \mathbf {y}, \mathbf {z} \rangle .\tag{12.3}
$$

Minimizing the Lagrangian w.r.t. x and z, the obtained dual problem is 

$$
q _ {\text { opt }} = \max _ {\mathbf {y} \in \mathbb {V}} \left\{q (\mathbf {y}) \equiv - f ^ {*} (\mathcal {A} ^ {T} (\mathbf {y})) - g ^ {*} (- \mathbf {y}) \right\}.\tag{12.4}
$$

By the strong duality theorem for convex problems (see Theorem A.1), it follows that strong duality holds for the pair of problems (12.1) and (12.4). 

Theorem 12.2 (strong duality for the pair of problems (12.1) and (12.4)). Suppose that Assumption 12.1 holds, and let $f _ { \mathrm { o p t } } , q _ { \mathrm { o p t } }$ be the optimal values of the primal and dual problems (12.1) and (12.4), respectively. Then $f _ { \mathrm { o p t } } = q _ { \mathrm { o p t } }$ , and the dual problem (12.4) possesses an optimal solution. 

We will consider the dual problem in its minimization form: 

$$
\min _ {\mathbf {y} \in \mathbb {V}} \{F (\mathbf {y}) + G (\mathbf {y}) \},\tag{12.5}
$$

where 

$$
F (\mathbf {y}) \equiv f ^ {*} (\mathcal {A} ^ {T} (\mathbf {y})),
$$

$$
G (\mathbf {y}) \equiv g ^ {*} (- \mathbf {y}).\tag{12.6}
$$

(12.7) 

The basic properties of F and G are gathered in the following lemma. 

Lemma 12.3 (properties of F and G). Suppose that Assumption 12.1 holds, and let F and G be defined by (12.6) and (12.7), respectively. Then 

(a) $F : \mathbb { V }  \mathbb { R }$ is convex and L -smooth with $\begin{array} { r } { L _ { F } = \frac { \| \boldsymbol { \mathcal { A } } \| ^ { 2 } } { \sigma } } \end{array}$ 

(b) $G : \mathbb { V } \to ( - \infty , \infty ]$ is proper closed and convex. 

Proof. (a) Since f is proper closed and σ-strongly convex, then by the conjugate correspondence theorem (Theorem 5.26(b)), $f ^ { * }$ is $\scriptstyle { \frac { 1 } { \sigma } } .$ -smooth. Therefore, for any $\mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } \in \mathbb { V }$ 

$$
\begin{array}{l} \| \nabla F (\mathbf {y} _ {1}) - \nabla F (\mathbf {y} _ {2}) \| = \| \mathcal {A} (\nabla f ^ {*} (\mathcal {A} ^ {T} (\mathbf {y} _ {1}))) - \mathcal {A} (\nabla f ^ {*} (\mathcal {A} ^ {T} (\mathbf {y} _ {2}))) \| \\ \qquad \leq \| \mathcal {A} \| \cdot \| \nabla f ^ {*} (\mathcal {A} ^ {T} (\mathbf {y} _ {1})) - \nabla f ^ {*} (\mathcal {A} ^ {T} (\mathbf {y} _ {2})) \| \\ \qquad \leq \frac {1}{\sigma} \| \mathcal {A} \| \cdot \| \mathcal {A} ^ {T} (\mathbf {y} _ {1}) - \mathcal {A} ^ {T} (\mathbf {y} _ {2}) \| \\ \qquad \leq \frac {\| \mathcal {A} \| \cdot \| \mathcal {A} ^ {T} \|}{\sigma} \| \mathbf {y} _ {1} - \mathbf {y} _ {2} \| = \frac {\| \mathcal {A} \| ^ {2}}{\sigma} \| \mathbf {y} _ {1} - \mathbf {y} _ {2} \|, \end{array}
$$

where we used in the last equality the fact that $\| \mathcal { A } \| = \| \mathcal { A } ^ { T } \|$ (see Section 1.14). To show the convexity of $F$ , note that $f ^ { * }$ is convex as a conjugate function (Theorem 4.3), and hence, by Theorem 2.16, F, as a composition of a convex function and a linear mapping, is convex. 

(b) Since g is proper closed and convex, so is $g ^ { * }$ (Theorems 4.3 and 4.5). Thus, $G ( \mathbf { y } ) \equiv g ^ { * } ( - \mathbf { y } )$ is also proper closed and convex. 

## 12.2 The Dual Proximal Gradient Method<sup>67</sup>

Problem (12.5) consists of minimizing the sum of a convex L-smooth function and a proper closed and convex function. It is therefore possible to employ in this setting the proximal gradient method on problem (12.5), which is equivalent to the dual problem of (12.1). Naturally we will refer to this algorithm as the “dual proximal gradient” $\left( \mathrm { D P G } \right)$ method. The dual representation of the method is given below. 

Dual Proximal Gradient—dual representation
- Initialization: pick $\mathbf{y}^0 \in \mathbb{V}$ and $L \geq L_F = \frac{\|\mathcal{A}\|^2}{\sigma}$ .
- General step ( $k \geq 0$ ): $\mathbf{y}^{k+1} = \text{prox}_{\frac{1}{L} G} \left( \mathbf{y}^k - \frac{1}{L} \nabla F(\mathbf{y}^k) \right)$ . (12.8) 

Since $F$ is convex and L<sub>F</sub>-smooth and G is proper closed and convex, we can invoke Theorem 10.21 to obtain an $O ( 1 / k )$ rate of convergence in terms of the dual objective function values. 

Theorem 12.4. Suppose that Assumption 12.1 holds, and let $\{ \mathbf { y } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the DPG method with $\begin{array} { r } { L \geq L _ { F } = \frac { \Vert \mathcal { A } \Vert ^ { 2 } } { \sigma } } \end{array}$ . Then for any optimal solution $\mathbf { y } ^ { * }$ of the dual problem (12.4) and $k \geq 1$ 

$$
q _ {\mathrm{opt}} - q (\mathbf {y} ^ {k}) \leq \frac {L \| \mathbf {y} ^ {0} - \mathbf {y} ^ {*} \| ^ {2}}{2 k}.
$$

Our goal now will be to find a primal representation of the method, which will be written in a more explicit way in terms of the data of the problem, meaning $( f , g , A )$ . To achieve this goal, we will require the following technical lemma. 

Lemma 12.5. Let ${ \cal F } ( { \bf y } ) = f ^ { \ast } ( { \cal A } ^ { T } ( { \bf y } ) + { \bf b } ) , G ( { \bf y } ) = g ^ { \ast } ( - { \bf y } )$ , where $f , g ,$ and A satisfy properties $( \mathrm { A } ) , ~ ( \mathrm { B } )$ , and (C) of Assumption 12.1 and b $\in \mathbb { E }$ . Then for any $\mathbf { y } , \mathbf { v } \in \mathbb { V }$ and $L > 0$ the relation 

$$
\mathbf {y} = \operatorname{prox} _ {\frac {1}{L} G} \left(\mathbf {v} - \frac {1}{L} \nabla F (\mathbf {v})\right)\tag{12.9}
$$

holds if and only if 

$$
\mathbf {y} = \mathbf {v} - \frac {1}{L} \mathcal {A} (\tilde {\mathbf {x}}) + \frac {1}{L} \mathrm{prox} _ {L g} (\mathcal {A} (\tilde {\mathbf {x}}) - L \mathbf {v}),
$$

where 

$$
\tilde {\mathbf {x}} = \operatorname{argmax} _ {\mathbf {x}} \left\{\langle \mathbf {x}, \mathcal {A} ^ {T} (\mathbf {v}) + \mathbf {b} \rangle - f (\mathbf {x}) \right\}.
$$

Proof. By the conjugate subgradient theorem (Corollary 4.21), since $f$ is proper closed and convex, 

$$
\nabla f ^ {*} (\mathcal {A} ^ {T} (\mathbf {v}) + \mathbf {b}) = \tilde {\mathbf {x}} \in \mathbb {E} \equiv \operatorname{argmax} _ {\mathbf {x}} \left\{\langle \mathbf {x}, \mathcal {A} ^ {T} (\mathbf {v}) + \mathbf {b} \rangle - f (\mathbf {x}) \right\}.
$$

Therefore, since $\nabla F ( \mathbf { v } ) =  { \mathcal { A } ( \nabla f ^ { \ast } (  { \mathcal { A } } ^ { T } ( \mathbf { v } ) + \mathbf { b } ) ) } =  { \mathcal { A } ( \tilde { \mathbf { x } } ) } .$ 

$$
\mathbf {y} = \mathrm{prox} _ {\frac {1}{L} G} \left(\mathbf {v} - \frac {1}{L} \mathcal {A} (\tilde {\mathbf {x}})\right).\tag{12.10}
$$

Invoking Theorem 6.15 with $g \gets \frac { 1 } { L } g ^ { * } , \boldsymbol { \mathcal { A } } = - \boldsymbol { \mathcal { T } } , \mathbf { b } = \mathbf { 0 }$ , we obtain that for any $\mathbf { z } \in \mathbb { V }$ 

$$
\operatorname{prox} _ {\frac {1}{L} G} (\mathbf {z}) = - \operatorname{prox} _ {\frac {1}{L} g ^ {*}} (- \mathbf {z}).\tag{12.11}
$$

Combining (12.10) and (12.11) and using the extended Moreau decomposition formula (Theorem 6.45), we finally obtain that 

$$
\begin{array}{r l} & {\mathbf {y} = \mathrm{prox} _ {\frac {1}{L} G} \left(\mathbf {v} - \frac {1}{L} \mathcal {A} (\tilde {\mathbf {x}})\right) = - \mathrm{prox} _ {\frac {1}{L} g ^ {*}} \left(\frac {1}{L} \mathcal {A} (\tilde {\mathbf {x}}) - \mathbf {v}\right)} \\ & {\quad = - \left[ \frac {1}{L} \mathcal {A} (\tilde {\mathbf {x}}) - \mathbf {v} - \frac {1}{L} \mathrm{prox} _ {L g} (\mathcal {A} (\tilde {\mathbf {x}}) - L \mathbf {v}) \right]} \\ & {\quad = \mathbf {v} - \frac {1}{L} \mathcal {A} (\tilde {\mathbf {x}}) + \frac {1}{L} \mathrm{prox} _ {L g} (\mathcal {A} (\tilde {\mathbf {x}}) - L \mathbf {v}). \quad \square} \end{array}
$$

Equipped with Lemma 12.5, we can write a primal representation of the DPG method. 

The Dual Proximal Gradient (DPG) Method—primal representation
Initialization: pick $y^{0} \in V$ , and $L \geq \frac{\|\mathcal{A}\|^{2}}{\sigma}$ .
General step: for any $k = 0, 1, 2, \ldots$ execute the following steps:
(a) set $x^{k} = \arg\max_{x} \left\{ \langle x, \mathcal{A}^{T}(y^{k}) \rangle - f(x) \right\}$ ;
(b) set $y^{k+1} = y^{k} - \frac{1}{L} \mathcal{A}(x^{k}) + \frac{1}{L} \text{prox}_{Lg}(\mathcal{A}(x^{k}) - L y^{k})$ . 

Remark 12.6 (the primal sequence). The sequence $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ generated by the method will be called “the primal sequence.” The elements of the sequence are actually not necessarily feasible w.r.t. the primal problem (12.1) since they are not guaranteed to belong to dom $( g )$ ; nevertheless, we will show that the primal sequence does converge to the optimal solution $\mathbf { x } ^ { * }$ 

To prove a convergence result in terms of the primal sequence, we will require the following fundamental primal-dual relation. 

Lemma 12.7 (primal-dual relation). Suppose that Assumption 12.1 holds. Let $\bar { \mathbf { y } } \in \mathrm { d o m } ( G )$ , where G is given in (12.7), and let 

$$
\bar {\mathbf {x}} = \operatorname{argmax} _ {\mathbf {x} \in \mathbb {E}} \left\{\langle \mathbf {x}, \mathcal {A} ^ {T} (\bar {\mathbf {y}}) \rangle - f (\mathbf {x}) \right\}.\tag{12.12}
$$

Then 

$$
\left\| \bar {\mathbf {x}} - \mathbf {x} ^ {*} \right\| ^ {2} \leq \frac {2}{\sigma} (q _ {\mathrm{opt}} - q (\bar {\mathbf {y}})).\tag{12.13}
$$

Proof. Recall that the primal problem (12.1) can be equivalently written as the problem 

$$
\min _ {\mathbf {x} \in \mathbb {E}, \mathbf {z} \in \mathbb {V}} \{f (\mathbf {x}) + g (\mathbf {z}): \mathcal {A} (\mathbf {x}) - \mathbf {z} = \mathbf {0} \},
$$

whose Lagrangian is (see also (12.3)) 

$$
L (\mathbf {x}, \mathbf {z}; \mathbf {y}) = f (\mathbf {x}) - \langle \mathcal {A} ^ {T} (\mathbf {y}), \mathbf {x} \rangle + g (\mathbf {z}) + \langle \mathbf {y}, \mathbf {z} \rangle .
$$

In particular, 

$$
L (\mathbf {x}, \mathbf {z}; \bar {\mathbf {y}}) = h (\mathbf {x}) + s (\mathbf {z}),\tag{12.14}
$$

where 

$$
\begin{array}{r l} & h (\mathbf {x}) = f (\mathbf {x}) - \langle \mathcal {A} ^ {T} (\bar {\mathbf {y}}), \mathbf {x} \rangle , \\ & s (\mathbf {z}) = g (\mathbf {z}) + \langle \bar {\mathbf {y}}, \mathbf {z} \rangle . \end{array}
$$

Since $h$ is σ-strongly convex and x¯ is its minimizer (see relation (12.12)), it follows by Theorem 5.25(b) that 

$$
h (\mathbf {x}) - h (\bar {\mathbf {x}}) \geq \frac {\sigma}{2} \| \mathbf {x} - \bar {\mathbf {x}} \| ^ {2}.\tag{12.15}
$$

Since the relation $\bar { \mathbf { y } } \in \mathrm { d o m } ( G )$ is equivalent to $- { \bar { \mathbf { y } } } \in \operatorname { d o m } ( g ^ { * } )$ , it follows that 

$$
\min _ {\mathbf {z} \in \mathbb {V}} \{g (\mathbf {z}) + \langle \bar {\mathbf {y}}, \mathbf {z} \rangle \} = \min _ {\mathbf {z} \in \mathbb {V}} s (\mathbf {z}) > - \infty .
$$

Let $\varepsilon > 0$ . Then there exists $\bar { \mathbf { z } } _ { \varepsilon }$ for which 

$$
s (\bar {\mathbf {z}} _ {\varepsilon}) \leq \min _ {\mathbf {z} \in \mathbb {V}} s (\mathbf {z}) + \varepsilon .\tag{12.16}
$$

Combining (12.14), (12.15), and (12.16), we obtain that for all $\mathbf { x } \in \operatorname { d o m } ( f )$ and $\mathbf { z } \in \operatorname { d o m } ( g )$ 2 

$$
L (\mathbf {x}, \mathbf {z}; \bar {\mathbf {y}}) - L (\bar {\mathbf {x}}, \bar {\mathbf {z}} _ {\varepsilon}; \bar {\mathbf {y}}) = h (\mathbf {x}) - h (\bar {\mathbf {x}}) + s (\mathbf {z}) - s (\bar {\mathbf {z}} _ {\varepsilon}) \geq \frac {\sigma}{2} \| \mathbf {x} - \bar {\mathbf {x}} \| ^ {2} - \varepsilon .
$$

In particular, substituting $\mathbf { x } = \mathbf { x } ^ { * } , \mathbf { z } = \mathbf { z } ^ { * } \equiv \mathcal { A } ( \mathbf { x } ^ { * } )$ , then $L ( \mathbf { x } ^ { * } , \mathbf { z } ^ { * } ; \bar { \mathbf { y } } ) = f ( \mathbf { x } ^ { * } ) +$ $g ( \mathcal { A } ( \mathbf { x } ^ { * } ) ) = f _ { \mathrm { o p t } } = q _ { \mathrm { o p t } }$ (by Theorem 12.2), and we obtain 

$$
q _ {\mathrm{opt}} - L (\bar {\mathbf {x}}, \bar {\mathbf {z}} _ {\varepsilon}; \bar {\mathbf {y}}) \geq \frac {\sigma}{2} \| \mathbf {x} ^ {*} - \bar {\mathbf {x}} \| ^ {2} - \varepsilon .\tag{12.17}
$$

In addition, by the definition of the dual objective function value, 

$$
L (\bar {\mathbf {x}}, \bar {\mathbf {z}} _ {\varepsilon}; \bar {\mathbf {y}}) \geq \min _ {\mathbf {x} \in \mathbb {E}, \mathbf {z} \in \mathbb {V}} L (\mathbf {x}, \mathbf {z}; \bar {\mathbf {y}}) = q (\bar {\mathbf {y}}),
$$

which, combined with (12.17), results in the inequality 

$$
\| \bar {\mathbf {x}} - \mathbf {x} ^ {*} \| ^ {2} \leq \frac {2}{\sigma} (q _ {\mathrm{opt}} - q (\bar {\mathbf {y}})) + \frac {2}{\sigma} \varepsilon .
$$

Since the above inequality holds for any $\varepsilon > 0$ , the desired result (inequality (12.13)) follows. 

The Fast Dual Proximal Gradient (FDPG) Method—dual representation
- Initialization: $L \geq L_F = \frac{\|\mathcal{A}\|^2}{\sigma}$ , $\mathbf{w}^0 = \mathbf{y}^0 \in \mathbb{E}$ , $t_0 = 1$ .
- General step ( $k \geq 0$ ):
(a) $\mathbf{y}^{k+1} = \text{prox}_{\frac{1}{L} G} \left( \mathbf{w}^k - \frac{1}{L} \nabla F(\mathbf{w}^k) \right)$ ;
(b) $t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}$ ;
(c) $\mathbf{w}^{k+1} = \mathbf{y}^{k+1} + \left( \frac{t_k - 1}{t_{k+1}} \right) (\mathbf{y}^{k+1} - \mathbf{y}^k)$ . 

Combining the primal-dual relation of Lemma 12.7 with the rate of convergence of the sequence of dual objective function values stated in Theorem 12.4, we can deduce a rate of convergence result for the primal sequence to the unique optimal solution. 

Theorem 12.8 $( O ( 1 / k )$ rate of convergence of the primal sequence of the DPG method). Suppose that Assumption 12.1 holds, and let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ and $\{ \mathbf { y } ^ { k } \} _ { k \geq 0 }$ be the primal and dual sequences generated by the DPG method with $L \geq$ $\begin{array} { r } { L _ { F } = \frac { \| \ b { \mathcal { A } } \| ^ { 2 } } { \sigma } } \end{array}$ . Then for any optimal solution $\mathbf { y } ^ { * }$ of the dual problem (12.4) and $k \geq 1$ ， 

$$
\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} \leq \frac {L \| \mathbf {y} ^ {0} - \mathbf {y} ^ {*} \| ^ {2}}{\sigma k}.\tag{12.18}
$$

Proof. Invoking Lemma 12.7 with $\bar { \mathbf { y } } = \mathbf { y } ^ { k }$ , we obtain by the definition of $\bar { \bf x }$ (equation (12.12)) that $\bar { \mathbf { x } } = \mathbf { x } ^ { k }$ , and hence (12.13) reads as 

$$
\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} \leq \frac {2}{\sigma} (q _ {\mathrm{opt}} - q (\mathbf {y} ^ {k})),
$$

which, combined with Theorem 12.4, yields the desired result. 

## 12.3 Fast Dual Proximal Gradient

The DPG method employs the proximal gradient method on the dual problem. Alternatively, we can also employ FISTA (see Section 10.7) on the dual problem (12.4). The dual representation of the method is given below. 

Since this is exactly FISTA employed on the dual problem, we can invoke Theorem 10.34 and obtain a convergence result in terms of dual objective function values. 

Theorem 12.9. Suppose that Assumption 12.1 holds and that $\{ \mathbf { y } ^ { k } \} _ { k \geq 0 }$ is the sequence generated by the FDPG method with $\begin{array} { r } { L \ge L _ { F } = \frac { \| \mathcal { A } \| ^ { 2 } } { \sigma } } \end{array}$ . Then for any optimal solution $\mathbf { y } ^ { * }$ of problem (12.4) and $k \geq 1$ 

$$
q _ {\mathrm{opt}} - q (\mathbf {y} ^ {k}) \leq \frac {2 L \| \mathbf {y} ^ {0} - \mathbf {y} ^ {*} \| ^ {2}}{(k + 1) ^ {2}}.
$$

Using Lemma 12.5 with $\mathbf v = \mathbf w ^ { k } , \mathbf y = \mathbf y ^ { k + 1 }$ , and $\mathbf b = \mathbf 0$ , we obtain that step (a) of the FDPG method, namely, 

$$
\mathbf {y} ^ {k + 1} = \mathrm{prox} _ {\frac {1}{L} G} \left(\mathbf {w} ^ {k} - \frac {1}{L} \nabla F (\mathbf {w} ^ {k})\right),
$$

can be equivalently written as 

$$
\begin{array}{c} \mathbf {u} ^ {k} = \operatorname{argmax} _ {\mathbf {u}} \left\{\langle \mathbf {u}, \mathcal {A} ^ {T} (\mathbf {w} ^ {k}) \rangle - f (\mathbf {u}) \right\}, \\ \mathbf {y} ^ {k + 1} = \mathbf {w} ^ {k} - \frac {1}{L} \mathcal {A} (\mathbf {u} ^ {k}) + \frac {1}{L} \mathrm{prox} _ {L g} (\mathcal {A} (\mathbf {u} ^ {k}) - L \mathbf {w} ^ {k}). \end{array}
$$

We can thus formulate a primal representation of the method. 

The Fast Dual Proximal Gradient (FDPG) Method—primal representation

Initialization: $L \geq L_{F} = \frac{\|\mathcal{A}\|^{2}}{\sigma}$ , $w^{0} = y^{0} \in V$ , $t_{0} = 1$ .

General step ( $k \geq 0$ ):
(a) $u^{k} = \arg\max_{u} \left\{ \langle u, \mathcal{A}^{T}(w^{k}) \rangle - f(u) \right\}$ ;
(b) $y^{k+1} = w^{k} - \frac{1}{L} A(u^{k}) + \frac{1}{L} \text{prox}_{Lg}(A(u^{k}) - L w^{k})$ ;
(c) $t_{k+1} = \frac{1 + \sqrt{1 + 4t_{k}^{2}}}{2}$ ;
(d) $w^{k+1} = y^{k+1} + \left( \frac{t_{k}-1}{t_{k+1}} \right) (y^{k+1} - y^{k})$ . 

The primal sequence that we will be interested in is actually not computed during the steps of the FDPG method. The definition of the primal sequence on which a convergence result will be proved is 

$$
\mathbf {x} ^ {k} = \operatorname{argmax} _ {\mathbf {x} \in \mathbb {E}} \left\{\langle \mathbf {x}, \mathcal {A} ^ {T} (\mathbf {y} ^ {k}) \rangle - f (\mathbf {x}) \right\}.\tag{12.19}
$$

The convergence result on the primal sequence is given below, and its proof is almost a verbatim repetition of the proof of Theorem 12.8. 

Theorem 12.10 $( O ( 1 / k ^ { 2 } )$ convergence of the primal sequence of the FDPG method). Suppose that Assumption 12.1 holds, and let $\{ \mathbf { y } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the FDPG method with $\begin{array} { r } { L \geq L _ { F } = \frac { \Vert \mathcal { A } \Vert ^ { 2 } } { \sigma } } \end{array}$ . Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence defined by (12.19). Then for any optimal solution $\mathbf { y } ^ { * }$ of the dual problem (12.4) and $k \geq 1$ ， 

$$
\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} \leq \frac {4 L \| \mathbf {y} ^ {0} - \mathbf {y} ^ {*} \| ^ {2}}{\sigma (k + 1) ^ {2}}.
$$

Proof. Invoking Lemma 12.7 with $\bar { \mathbf { y } } = \mathbf { y } ^ { k }$ , we obtain by the definition of $\bar { \bf x }$ (equation (12.12)) that $\bar { \mathbf { x } } = \mathbf { x } ^ { k }$ , and hence the result (12.13) reads as 

$$
\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \| ^ {2} \leq \frac {2}{\sigma} (q _ {\mathrm{opt}} - q (\mathbf {y} ^ {k})),
$$

which, combined with Theorem 12.9, yields the desired result. 

## 12.4 Examples I

## 12.4.1 Orthogonal Projection onto a Polyhedral Set

$$
S = \left\{\mathbf {x} \in \mathbb {R} ^ {n}: \mathbf {A x} \leq \mathbf {b} \right\},
$$

where $\mathbf { A } \in \mathbb { R } ^ { p \times n }$ , b $\in ~ \mathbb { R } ^ { p }$ . We assume that $S$ is nonempty. Let $\mathbf { d } \in \mathbb { R } ^ { n }$ . The orthogonal projection of d onto $S$ is the unique optimal solution of the problem 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {n}} \left\{\frac {1}{2} \| \mathbf {x} - \mathbf {d} \| ^ {2}: \mathbf {A x} \leq \mathbf {b} \right\}.\tag{12.20}
$$

Problem (12.20) fits model (12.1) with $\begin{array} { r } { \mathbb { E } = \mathbb { R } ^ { n } , \mathbb { V } = \mathbb { R } ^ { p } , f ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { x } - \mathbf { d } \| ^ { 2 } } \end{array}$ 

$$
g (\mathbf {z}) = \delta_ {\operatorname{Box} [ - \infty \mathbf {e}, \mathbf {b} ]} (\mathbf {z}) = \left\{ \begin{array}{l l} \mathbf {0}, & \mathbf {z} \leq \mathbf {b}, \\ \infty & \text {else}, \end{array} \right.
$$

and $\mathcal { A } ( \mathbf { x } ) \equiv \mathbf { A } \mathbf { x }$ . We have 

$\operatorname { a r g m a x } _ { \mathbf { x } } \{ \langle \mathbf { v } , \mathbf { x } \rangle - f ( \mathbf { x } ) \} = \mathbf { v } + \mathbf { d } { \mathrm { ~ f o r ~ a n y ~ } } \mathbf { v } \in \mathbb { R } ^ { n }$ ; 

$\| \mathcal { A } \| = \| \mathbf { A } \| _ { 2 , 2 } ;$ 

$\sigma = 1$ 

$\mathcal { A } ^ { T } ( \mathbf { y } ) = \mathbf { A } ^ { T } \mathbf { y }$ for any $\mathbf { y } \in \mathbb { R } ^ { p } $ ; 

$\begin{array} { r c l } { { \mathrm { p r o x } _ { L g } ( \mathbf { z } ) ~ { = } ~ { \cal P } _ { \mathrm { B o x } [ - \infty \mathbf { e } , \mathbf { b } ] } ( \mathbf { z } ) ~ { = } ~ { \mathrm { m i n } } \{ \mathbf { z } , \mathbf { b } \} } } \\ { { ( { \mathrm { m i n } } \{ z _ { i } , b _ { i } \} ) _ { i = 1 } ^ { p } . } } \end{array}$ , where $\operatorname* { m i n } \{ \mathbf { z } , \mathbf { b } \}$ is the vector 

Using these facts, the DPG and FDPG methods for solving problem (12.20) can be explicitly written. 

Algorithm 1 [DPG for solving (12.20)] 

• Initialization: $L \geq \| \mathbf { A } \| _ { 2 , 2 } ^ { 2 } , \mathbf { y } ^ { 0 } \in \mathbb { R } ^ { p }$ 

• General step $( k \geq 0 )$ : 

(a) $\mathbf { x } ^ { k } = \mathbf { A } ^ { T } \mathbf { y } ^ { k } + \mathbf { d } ;$ 

$$
\mathbf {y} ^ {k + 1} = \mathbf {y} ^ {k} - \frac {1}{L} \mathbf {A x} ^ {k} + \frac {1}{L} \min \{\mathbf {A x} ^ {k} - L \mathbf {y} ^ {k}, \mathbf {b} \}. \tag {b}
$$

Algorithm 2 [FDPG for solving (12.20)]
- Initialization: $L \geq \| \mathbf{A} \|_{2,2}^2$ , $\mathbf{w}^0 = \mathbf{y}^0 \in \mathbb{R}^p$ , $t_0 = 1$ .
- General step ( $k \geq 0$ ):
(a) $\mathbf{u}^k = \mathbf{A}^T \mathbf{w}^k + \mathbf{d}$ ;
(b) $\mathbf{y}^{k+1} = \mathbf{w}^k - \frac{1}{L} \mathbf{A} \mathbf{u}^k + \frac{1}{L} \min\{\mathbf{A} \mathbf{u}^k - L \mathbf{w}^k, \mathbf{b}\}$ ;
(c) $t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}$ ;
(d) $\mathbf{w}^{k+1} = \mathbf{y}^{k+1} + \left( \frac{t_k - 1}{t_{k+1}} \right)(\mathbf{y}^{k+1} - \mathbf{y}^k)$ . 

The primal sequence for the FDPG method is given by $\mathbf { x } ^ { k } = \mathbf { A } ^ { T } \mathbf { y } ^ { k } + \mathbf { d }$ 

## 12.4.2 Orthogonal Projection onto the Intersection of Closed Convex Sets

Given p closed and convex sets $C _ { 1 } , C _ { 2 } , \ldots , C _ { p } \subseteq \mathbb { E }$ and a point d $\in \mathbb { E }$ , the orthogonal projection of d onto the intersection $\cap _ { i = 1 } ^ { p } C _ { i }$ is the optimal solution of the problem 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \left\{\frac {1}{2} \| \mathbf {x} - \mathbf {d} \| ^ {2}: \mathbf {x} \in \cap_ {i = 1} ^ {p} C _ {i} \right\}.\tag{12.21}
$$

We will assume that the intersection $\cap _ { i = 1 } ^ { p } C _ { i }$ is nonempty and that projecting onto each set $C _ { i }$ is an easy task. Our purpose will be to devise a method for solving problem (12.21) that only requires computing at each iteration—in addition to elementary linear algebra operations—orthogonal projections onto the sets $C _ { i }$ . Problem $\left( 1 2 . 2 1 \right)$ fits model (12.1) with $\begin{array} { r } { \mathbb { V } = \mathbb { E } ^ { \tilde { p } } , f ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { x } - \mathbf { d } \| ^ { 2 } , g ( \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { p } ) = } \end{array}$ $\textstyle \sum _ { i = 1 } ^ { p } \delta _ { C _ { i } } ( \mathbf { x } _ { i } )$ , and $\mathcal { A } : \mathbb { E }  \mathbb { V }$ given by 

$$
\mathcal {A} (\mathbf {z}) = (\underbrace {\mathbf {z} , \mathbf {z} , \ldots , \mathbf {z}} _ {p \text { times }}) \text {   for   any   } \mathbf {z} \in \mathbb {E}.
$$

We have 

• argmax ${ \bf \dot { \bf \dot { x } } } \{ \langle { \bf v } , { \bf x } \rangle - f ( { \bf x } ) \} = { \bf v } + { \bf d } \mathrm { ~ f o r ~ a n y ~ } { \bf v } \in \mathbb { E } ;$ 

$\| A \| ^ { 2 } = p ;$ 

$\sigma = 1$ 

$\begin{array} { r } { \mathcal { A } ^ { T } ( \mathbf { y } ) = \sum _ { i = 1 } ^ { p } y _ { i } } \end{array}$ for any $\mathbf { y } \in \mathbb { E } ^ { p }$ ; 

$\operatorname { p r o x } _ { L g } ( \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } , \ldots , \mathbf { v } _ { p } ) = \left( P _ { C _ { 1 } } ( \mathbf { v } _ { 1 } ) , P _ { C _ { 2 } } ( \mathbf { v } _ { 2 } ) , \ldots , P _ { C _ { p } } ( \mathbf { v } _ { p } ) \right)$ for any $\mathbf { v } \in \mathbb { E } ^ { p }$ 

Using these facts, the DPG and FDPG methods for solving problem (12.21) can be explicitly written. 

Algorithm 3 [DPG for solving (12.21)] 

• Initialization: $L \geq p , \mathbf { y } ^ { 0 } \in \mathbb { E } ^ { p }$ 

• General step $( k \geq 0 )$ 

(a) $\begin{array} { r } { \mathbf { x } ^ { k } = \sum _ { i = 1 } ^ { p } \mathbf { y } _ { i } ^ { k } + \mathbf { d } ; } \end{array}$ 

$$
\mathbf {\Phi} _ {i} ^ {k + 1} = \mathbf {y} _ {i} ^ {k} - \frac {1}{L} \mathbf {x} ^ {k} + \frac {1}{L} P _ {C _ {i}} (\mathbf {x} ^ {k} - L \mathbf {y} _ {i} ^ {k}), i = 1, 2, \ldots , p.
$$

Algorithm 4 [FDPG for solving (12.21)] 

• Initialization: $L \geq p , \mathbf { w } ^ { 0 } = \mathbf { y } ^ { 0 } \in \mathbb { E } ^ { p } , t _ { 0 } = 1 .$ 

• General step $( k \geq 0 )$ 

(a) $\begin{array} { r } { \mathbf { u } ^ { k } = \sum _ { i = 1 } ^ { p } \mathbf { w } _ { i } ^ { k } + \mathbf { d } ; } \end{array}$ 

(b) $\begin{array} { r } { { \bf y } _ { i } ^ { k + 1 } = { \bf w } _ { i } ^ { k } - \frac { 1 } { L } { \bf u } ^ { k } + \frac { 1 } { L } P _ { C _ { i } } ( { \bf u } ^ { k } - L { \bf w } _ { i } ^ { k } ) , i = 1 , 2 , \ldots , p ; } \end{array}$ 

(c) $\begin{array} { r } { t _ { k + 1 } = \frac { 1 + \sqrt { 1 + 4 t _ { k } ^ { 2 } } } { 2 } ; } \end{array}$ 

$$
\mathbf {w} ^ {k + 1} = \mathbf {y} ^ {k + 1} + \left(\frac {t _ {k} - 1}{t _ {k + 1}}\right) (\mathbf {y} ^ {k + 1} - \mathbf {y} ^ {k}). \tag {d}
$$

To actually guarantee convergence of the method, Assumption 12.1 needs to be satisfied, meaning that we assume that $\cap _ { i = 1 } ^ { p } \operatorname { r i } ( C _ { i } ) \neq \emptyset$ 

The primal sequence for the FDPG method is given by $\begin{array} { r } { \mathbf { x } ^ { k } = \sum _ { i = 1 } ^ { p } \mathbf { y } _ { i } ^ { k } + \mathbf { d } } \end{array}$ 

Example 12.11 (orthogonal projection onto a polyhedral set revisited). Note that Algorithm 4 can also be used to find an orthogonal projection of a point $\mathbf { d } \in \mathbb { R } ^ { n }$ onto the polyhedral set $C = \{ \mathbf { x } \in \mathbb { R } ^ { n } : \mathbf { A x } \leq \mathbf { b } \}$ , where $\mathbf { A } \in \mathbb { R } ^ { p \times n } , \mathbf { b } \in \mathbb { R } ^ { p }$ Indeed, $C$ can be written as the following intersection of half-spaces: 

$$
C = \cap_ {i = 1} ^ {p} C _ {i},
$$

where 

$$
C _ {i} = \{\mathbf {x} \in \mathbb {R} ^ {n}: \mathbf {a} _ {i} ^ {T} \mathbf {x} \leq b _ {i} \},\tag{12.22}
$$

with $\mathbf { a } _ { 1 } ^ { T } , \mathbf { a } _ { 2 } ^ { T } , \ldots , \mathbf { a } _ { p } ^ { T }$ being the rows of A. The projections on the half-spaces are simple and given by (see Lemma 6.26) $\begin{array} { r } { P _ { C _ { i } } ( \mathbf { x } ) = \mathbf { x } - \frac { [ \mathbf { a } _ { i } ^ { T } \mathbf { x } - b _ { i } ] _ { + } } { \Vert \mathbf { a } _ { i } \Vert ^ { 2 } } \mathbf { a } _ { i } } \end{array}$ . To summarize, the problem 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {n}} \left\{\frac {1}{2} \| \mathbf {x} - \mathbf {d} \| _ {2} ^ {2}: \mathbf {A x} \leq \mathbf {b} \right\}
$$

can be solved by two diferent FDPG methods. The first one is Algorithm $^ { 2 , }$ and the second one is the following algorithm, which is Algorithm 4 specified to the case where $C _ { i }$ is given by (12.22) for any $i \in \{ 1 , 2 , \ldots , p \}$ 

Algorithm 5 [second version of FDPG for solving (12.20)]
- Initialization: $L \geq p$ , $\mathbf{w}^0 = \mathbf{y}^0 \in \mathbb{E}^p$ , $t_0 = 1$ .
- General step ( $k \geq 0$ ):
(a) $\mathbf{u}^k = \sum_{i=1}^{p} \mathbf{w}_i^k + \mathbf{d}$ ;
(b) $\mathbf{y}_i^{k+1} = -\frac{1}{L \| \mathbf{a}_i \|^2} [\mathbf{a}_i^T (\mathbf{u}^k - L \mathbf{w}_i^k) - b_i]_+\mathbf{a}_i$ , $i = 1, 2, \ldots, p$ ;
(c) $t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}$ ;
(d) $\mathbf{w}^{k+1} = \mathbf{y}^{k+1} + \left(\frac{t_k - 1}{t_{k+1}}\right)(\mathbf{y}^{k+1} - \mathbf{y}^k)$ . 

Example 12.12 (comparison between DPG and FDPG). The $O ( 1 / k ^ { 2 } )$ rate of convergence obtained for the FDPG method (Theorem 12.10) is better than the $O ( 1 / k )$ result obtained for the DPG method (Theorem 12.8). To illustrate that this theoretical advantage is also reflected in practice, we consider the problem of projecting the point $( 0 . 5 , 1 . 9 ) ^ { T }$ onto a dodecagon—a regular polygon with 12 edges, which is represented as the intersection of 12 half-spaces. The first 10 iterations of the DPG and FDPG methods with $L = p = 1 2$ can be seen in Figure 12.1, where the DPG and FDPG methods that were used are those described by Algorithms 3 and 4 for the intersection of closed convex sets (which are taken as the 12 half-spaces in this example) and not Algorithms 1 and 2. Evidently, the FDPG method was able to find a good approximation of the projection after 10 iterations, while the DPG method was rather far from the required solution. ■ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/148310d8df3da926a892d51f3e1804e74f901a4041e6ce71cc3ae46ac4a7d62e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/b3d751f63a34c22176a44eaac880e8dea5ac6c4015c7021cf8602db01a3fd37a.jpg)



Figure 12.1. First 10 iterations of the DPG method (Algorithm 3) and the FDPG method (Algorithm $4 / 5 )$ . The initial value of the dual vector y was the zeros vector in both methods.


## 12.4.3 One-Dimensional Total Variation Denoising

In the denoising problem we are given a signal $\mathbf { d } \in \mathbb { E }$ , which is contaminated by noise, and we seek to find another vector $\mathbf { x } \in \mathbb { E }$ , which, on the one hand, is close to d in the sense that the norm $\| \mathbf { x } - \mathbf { d } \|$ is small and, on the other hand, yields a small regularization term $R ( \mathcal { A } ( \mathbf { x } ) )$ , where here $\mathcal { A } : \mathbb { E }  \mathbb { V }$ is a linear transformation that in many applications accounts for the smoothness of the signal and $R : \mathbb { V } \to \mathbb { R } _ { + }$ is a given convex function that measures the magnitude of its argument in some sense. The denoising problem is then defined to be 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \left\{\frac {1}{2} \| \mathbf {x} - \mathbf {d} \| ^ {2} + R (\mathcal {A} (\mathbf {x})) \right\}.\tag{12.23}
$$

In the one-dimensional total variation denoising problem, we are interested in the case where $\mathbb { E } = \mathbb { R } ^ { n } , \mathbb { V } = \mathbb { R } ^ { n - 1 } , \mathcal { A } ( \mathbf { x } ) = \mathbf { D } \mathbf { x }$ , and $R ( \mathbf { z } ) = \lambda \| \mathbf { z } \| _ { 1 }$ with $\lambda > 0$ being a “regularization parameter” and D being the matrix satisfying $\mathbf { D x } = ( x _ { 1 } - x _ { 2 } , x _ { 2 } -$ $x _ { 3 } , . . . , x _ { n - 1 } - x _ { n } \big ) ^ { T }$ for all $\mathbf { x } \in \mathbb { R } ^ { n }$ . Thus, problem (12.23) takes the form<sup>68</sup> 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {n}} \left\{\frac {1}{2} \| \mathbf {x} - \mathbf {d} \| _ {2} ^ {2} + \lambda \| \mathbf {D x} \| _ {1} \right\}\tag{12.24}
$$

or, more explicitly, 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {n}} \left\{\frac {1}{2} \| \mathbf {x} - \mathbf {d} \| _ {2} ^ {2} + \lambda \sum_ {i = 1} ^ {n - 1} | x _ {i} - x _ {i + 1} | \right\}.
$$

The function $\mathbf { x } \mapsto \| \mathbf { D x } \| _ { 1 }$ is known as a one-dimensional total variation function and is actually only one instance of many variants of total variation functions. Problem (12.24) fits model (12.1) with $\begin{array} { r } { \mathbb { E } = \mathbb { R } ^ { n } , \mathbb { V } = \mathbb { R } ^ { n - 1 } , f ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { x } - \mathbf { d } \| _ { 2 } ^ { 2 } , g ( \mathbf { y } ) = \lambda \| \mathbf { y } \| _ { 1 } } \end{array}$ 2 and $\mathcal { A } ( \mathbf { x } ) \equiv \mathbf { D } \mathbf { x }$ . In order to explicitly write the DPG and FDPG methods, we note that 

• argmax {v, x − f(x)} = v + d for any $\mathbf { v } \in \mathbb { E } ;$ 

$\| \mathcal { A } \| ^ { 2 } = \| \mathbf { D } \| _ { 2 , 2 } ^ { 2 } \leq 4 ;$ 

$\sigma = 1$ 

$\mathcal { A } ^ { T } ( \mathbf { y } ) = \mathbf { D } ^ { T } \mathbf { y }$ for any $\mathbf { y } \in \mathbb { R } ^ { n - 1 }$ ; 

$\mathrm { p r o x } _ { L g } ( \mathbf { y } ) = \mathcal { T } _ { \lambda L } ( \mathbf { y } )$ 

The bound on $\| \mathbf { D } \| _ { 2 , 2 }$ was achieved by the following argument: 

$$
\| \mathbf {D x} \| _ {2} ^ {2} = \sum_ {i = 1} ^ {n - 1} (x _ {i} - x _ {i + 1}) ^ {2} \leq 2 \sum_ {i = 1} ^ {n - 1} (x _ {i} ^ {2} + x _ {i + 1} ^ {2}) \leq 4 \| \mathbf {x} \| ^ {2}.
$$

The DPG and FDPG methods with $L = 4$ are explicitly written below. 

Algorithm 6 [DPG for solving (12.24)] 

• Initialization: $\mathbf { y } ^ { 0 } \in \mathbb { R } ^ { n - 1 }$ 

• General step $( k \geq 0 )$ 

(a) $\mathbf { x } ^ { k } = \mathbf { D } ^ { T } \mathbf { y } ^ { k } + \mathbf { d } ;$ 

$$
\mathbf {y} ^ {k + 1} = \mathbf {y} ^ {k} - \frac {1}{4} \mathbf {D x} ^ {k} + \frac {1}{4} \mathcal {T} _ {4 \lambda} (\mathbf {D x} ^ {k} - 4 \mathbf {y} ^ {k}).
$$

Algorithm 7 [FDPG for solving (12.24)] 

• Initialization: $\mathbf { w } ^ { 0 } = \mathbf { y } ^ { 0 } \in \mathbb { R } ^ { n - 1 } , t _ { 0 } = 1 .$ 

• General step $( k \geq 0 )$ 

(a) $\mathbf { u } ^ { k } = \mathbf { D } ^ { T } \mathbf { w } ^ { k } + \mathbf { d } ;$ 

(b) $\begin{array} { r } { \mathbf { y } ^ { k + 1 } = \mathbf { w } ^ { k } - \frac { 1 } { 4 } \mathbf { D } \mathbf { u } ^ { k } + \frac { 1 } { 4 } \mathcal { T } _ { 4 \lambda } ( \mathbf { D } \mathbf { u } ^ { k } - 4 \mathbf { w } ^ { k } ) ; } \end{array}$ 

(c) $\begin{array} { r } { t _ { k + 1 } = \frac { 1 + \sqrt { 1 + 4 t _ { k } ^ { 2 } } } { 2 } ; } \end{array}$ 

$$
\mathbf {w} ^ {k + 1} = \mathbf {y} ^ {k + 1} + \left(\frac {t _ {k} - 1}{t _ {k + 1}}\right) (\mathbf {y} ^ {k + 1} - \mathbf {y} ^ {k}).
$$

Example 12.13. Consider the case where $n = 1 0 0 0$ and the “clean” (actually unknown) signal is the vector $\mathbf { d } ^ { \mathrm { t r u e } }$ , which is a discretization of a step function: 

$$
d _ {i} ^ {\text {true}} = \left\{ \begin{array}{l l} 1, & 1 \leq i \leq 2 5 0, \\ 3, & 2 5 1 \leq i \leq 5 0 0, \\ 0, & 5 0 1 \leq i \leq 7 5 0, \\ 2, & 7 5 1 \leq i \leq 1 0 0 0. \end{array} \right.
$$

The observed vector d was constructed by adding independently to each componentof $\mathbf { d } ^ { \mathrm { t r u e } }$ a normally distributed noise with zero mean and standard deviation 0.05.The true and noisy signals can be seen in Figure 12.2. We ran 100 iterations ofAlgorithms $6 ~ ( \mathrm { D P G } )$ and $7 ~ \mathrm { ( F D P G ) }$ initialized with $\mathbf { y } ^ { 0 } = \mathbf { 0 }$ , and the resultingsignals can be seen in Figure 12.3. Clearly, the FDPG method produces a muchbetter quality reconstruction of the original step function than the DPG method.This is reflected in the objective function values of the vectors produced by each ofthe methods. The objective function values of the vectors generated by the DPGand FDPG methods after 100 iterations are 9.1667 and 8.4621, respectively, wherethe optimal value is 8.3031. 厂

## 12.4.4 Two-Dimensional Total Variation Denoising

In the two-dimensional total variation denoising problem, we are given an observed noisy matrix d $\in \mathbb { R } ^ { m \times n }$ , and we seek to solve the problem 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {m \times n}} \left\{\frac {1}{2} \| \mathbf {x} - \mathbf {d} \| _ {F} ^ {2} + \lambda \mathrm{TV} (\mathbf {x}) \right\}.\tag{12.25}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/ecef806a32644a49895efa1b7c620b31a8b5a1ac2684404641b27cd610484b53.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/328c8e95b591b23f48ae2dabea8e21b799df1b710e0b51db9ee37e0b9b7b2111.jpg)



Figure 12.2. True signal (left) and noisy signal (right).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/5c7048af35781953e13fbde56751ba6c182cc83d9faad9b08346bb6a86a6cfd8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/c6cf919381e387ff06fec17d009af2b92e34f7e23af2dd837752e062e566f6f5.jpg)



Figure 12.3. Results of the DPG and FDPG methods.


There are many possible choices for the two-dimensional total variation function $\mathrm { T V } ( \cdot )$ . Two popular choices are the isotropic TV defined for any $\mathbf { x } \in \mathbb { R } ^ { m \times n }$ by 

$$
\begin{array}{l} \mathrm{TV} _ {I} (\mathbf {x}) = \sum_ {i = 1} ^ {m - 1} \sum_ {j = 1} ^ {n - 1} \sqrt {(x _ {i , j} - x _ {i , j + 1}) ^ {2} + (x _ {i , j} - x _ {i + 1 , j}) ^ {2}} \\ \qquad + \sum_ {j = 1} ^ {n - 1} | x _ {m, j} - x _ {m, j + 1} | + \sum_ {i = 1} ^ {m - 1} | x _ {i, n} - x _ {i + 1, n} | \end{array}\tag{12.26}
$$

and the $l _ { \mathrm { 1 } } { \mathrm { - b a s e d } }$ , anisotropic TV defined by 

$$
\begin{array}{l l} \mathbf {x} \in \mathbb {R} ^ {m \times n}, & \mathrm{TV} _ {l _ {1}} (\mathbf {x}) = \sum_ {i = 1} ^ {m - 1} \sum_ {j = 1} ^ {n - 1} \{| x _ {i, j} - x _ {i, j + 1} | + | x _ {i, j} - x _ {i + 1, j} | \} \\ & \qquad + \sum_ {j = 1} ^ {n - 1} | x _ {m, j} - x _ {m, j + 1} | + \sum_ {i = 1} ^ {m - 1} | x _ {i, n} - x _ {i + 1, n} |. \end{array}
$$

Problem (12.25) fits the main model (12.1) with $\mathbb { E } = \mathbb { R } ^ { m \times n } , \mathbb { V } = \mathbb { R } ^ { m \times ( n - 1 ) } \times$ $\begin{array} { r } { \mathbb { R } ^ { ( m - 1 ) \times n } , f ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { x } - \mathbf { d } \| _ { F } ^ { 2 } } \end{array}$ , and $\mathcal { A } ( \mathbf { x } ) = \left( \mathbf { p } ^ { \mathbf { x } } , \mathbf { q } ^ { \mathbf { x } } \right)$ , where $\mathbf { p ^ { x } } \in \mathbb { R } ^ { m \times ( n - 1 ) }$ and $\mathbf { q } ^ { \mathbf { x } } \in \mathbb { R } ^ { ( m - 1 ) \times n }$ are given by 

$$
p _ {i, j} ^ {\mathbf {x}} = x _ {i, j} - x _ {i, j + 1}, \quad i = 1, 2, \ldots , m, j = 1, 2, \ldots , n - 1,
$$

$$
q _ {i, j} ^ {\mathbf {x}} = x _ {i, j} - x _ {i + 1, j}, \quad i = 1, 2, \dots , m - 1, j = 1, 2, \dots , n.
$$

The function $g : \mathbb { V } \to \mathbb { R }$ is given in the isotropic case by 

$$
g (\mathbf {p}, \mathbf {q}) = g _ {\mathrm{I}} (\mathbf {p}, \mathbf {q}) \equiv \sum_ {i = 1} ^ {m - 1} \sum_ {j = 1} ^ {n - 1} \sqrt {p _ {i , j} ^ {2} + q _ {i , j} ^ {2}} + \sum_ {j = 1} ^ {n - 1} | p _ {m, j} | + \sum_ {i = 1} ^ {m - 1} | q _ {i, n} |
$$

and in the anisotropic case by 

$$
g (\mathbf {p}, \mathbf {q}) = g _ {l _ {1}} (\mathbf {p}, \mathbf {q}) \equiv \sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n - 1} | p _ {i, j} | + \sum_ {i = 1} ^ {m - 1} \sum_ {j = 1} ^ {n} | q _ {i, j} |.
$$

Since $g _ { \mathrm { I } }$ and $g _ { l _ { 1 } }$ are a separable sum of either absolute values or $l _ { 2 }$ norms, it is easy to compute their prox mappings using Theorem 6.6 (prox of separable functions), Example 6.8 (prox of the $l _ { 1 } { \mathrm { - n o r m } } )$ , and Example 6.19 (prox of Euclidean norms) and obtain that for any $\mathbf { p } \in \mathbb { R } ^ { m \times ( n - 1 ) }$ and $\mathbf { q } \in \mathbb { R } ^ { ( m - 1 ) \times n }$ 

$$
\mathrm{prox} _ {\lambda g _ {\mathrm{I}}} (\mathbf {p}, \mathbf {q}) = (\bar {\mathbf {p}}, \bar {\mathbf {q}}),
$$

where 

$$
\bar {p} _ {i, j} = \left(1 - \lambda / \max \left\{\sqrt {p _ {i , j} ^ {2} + q _ {i , j} ^ {2}}, \lambda \right\}\right) p _ {i, j}, \quad i = 1, 2, \dots , m - 1, j = 1, 2, \dots n - 1,
$$

$$
\bar {p} _ {m, j} = \mathcal {T} _ {\lambda} (p _ {m, j}), \quad j = 1, 2, \dots , n - 1,
$$

$$
\bar {q} _ {i, j} = \left(1 - \lambda / \max \left\{\sqrt {p _ {i , j} ^ {2} + q _ {i , j} ^ {2}}, \lambda \right\}\right) q _ {i, j}, \quad i = 1, 2, \dots , m - 1, j = 1, 2, \dots n - 1,
$$

$$
\bar {q} _ {i, n} = \mathcal {T} _ {\lambda} (q _ {i, n}), \quad i = 1, 2, \ldots , m - 1,
$$

and 

$$
\operatorname{prox} _ {\lambda g _ {l _ {1}}} (\mathbf {p}, \mathbf {q}) = (\tilde {\mathbf {p}}, \tilde {\mathbf {q}}),
$$

where 

$$
\tilde {p} _ {i, j} = \mathcal {T} _ {\lambda} (p _ {i, j}), \quad i = 1, 2, \ldots , m, j = 1, 2, \ldots n - 1,
$$

$$
\tilde {q} _ {i, j} = \mathcal {T} _ {\lambda} (q _ {i, j}), \quad i = 1, 2, \ldots , m - 1, j = 1, 2, \ldots , n.
$$

The last detail that is missing in order to explicitly write the DPG or FDPG methods for solving problem (12.25) is the computation of $\mathcal { A } ^ { T } : \mathbb { V } $ E at points in $\mathbb { V } .$ . For that, note that for any $\mathbf { x } \in \mathbb { E }$ and $( \mathbf { p } , \mathbf { q } ) \in \mathbb { V }$ 2 

$$
\begin{array}{l} \langle \mathcal {A} (\mathbf {x}), (\mathbf {p}, \mathbf {q}) \rangle = \sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n - 1} (x _ {i, j} - x _ {i, j + 1}) p _ {i, j} + \sum_ {i = 1} ^ {m - 1} \sum_ {j = 1} ^ {n} (x _ {i, j} - x _ {i + 1, j}) q _ {i, j} \\ \qquad = \sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n} x _ {i, j} (p _ {i, j} + q _ {i, j} - p _ {i, j - 1} - q _ {i - 1, j}) \\ \qquad = \langle \mathbf {x}, \mathcal {A} ^ {T} (\mathbf {p}, \mathbf {q}) \rangle , \end{array}
$$

where we use a convention that 

$$
p _ {i, 0} = p _ {i, n} = q _ {0, j} = q _ {m, j} = 0 \quad \mathrm{forany} i = 1, 2, \ldots , m, j = 1, 2, \ldots , n.
$$

Therefore, with the above convention in mind, for any $( \mathbf { p } , \mathbf { q } ) \in \mathbb { V }$ 2 

$$
\mathcal {A} ^ {T} (\mathbf {p}, \mathbf {q}) _ {i, j} = p _ {i, j} + q _ {i, j} - p _ {i, j - 1} - q _ {i - 1, j}, \quad i = 1, 2, \ldots , m, j = 1, 2, \ldots , n.
$$

We also want to compute an upper bound on $\| { \mathcal { A } } \| ^ { 2 }$ . This can be done using the same technique as in the one-dimensional case; note that for any $\mathbf { x } \in \mathbb { R } ^ { m \times n }$ , 

$$
\begin{array}{l} \| \mathcal {A} (\mathbf {x}) \| ^ {2} = \sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n - 1} (x _ {i, j} - x _ {i, j + 1}) ^ {2} + \sum_ {i = 1} ^ {m - 1} \sum_ {j = 1} ^ {n} (x _ {i, j} - x _ {i + 1, j}) ^ {2} \\ \qquad \leq 2 \sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n - 1} (x _ {i, j} ^ {2} + x _ {i, j + 1} ^ {2}) + 2 \sum_ {i = 1} ^ {m - 1} \sum_ {j = 1} ^ {n} (x _ {i, j} ^ {2} + x _ {i + 1, j} ^ {2}) \\ \qquad \leq 8 \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} x _ {i, j} ^ {2}. \end{array}
$$

Therefore, $\| \mathcal { A } \| ^ { 2 } \leq 8$ . We will now explicitly write the FDPG method for solving the two-dimensional anisotropic total variation problem, meaning problem (12.25) with $g = g _ { l _ { 1 } }$ . For the stepsize, we use $L = 8$ 

Algorithm 8 [FDPG for solving (12.25) with $g = \lambda \mathrm{TV}_{l_1}$ ]
- Initialization: $\tilde{\mathbf{p}}^0 = \mathbf{p}^0 \in \mathbb{R}^{m \times (n-1)}, \tilde{\mathbf{q}}^0 = \mathbf{q}^0 \in \mathbb{R}^{(m-1) \times n}, t_0 = 1$ .
- General step ( $k \geq 0$ ):
(a) compute $\mathbf{u}^k \in \mathbb{R}^{m \times n}$ by setting for $i = 1, 2, \ldots, m, j = 1, 2, \ldots, n$ , $u_{i,j}^k = \tilde{p}_{i,j}^k + \tilde{q}_{i,j}^k - \tilde{p}_{i,j-1}^k - \tilde{q}_{i-1,j}^k + d_{i,j}$ ;
(b) set ( $\mathbf{p}^{k+1}, \mathbf{q}^{k+1}$ ) as $p_{i,j}^{k+1} = \tilde{p}_{i,j}^k - \frac{1}{8}(u_{i,j}^k - u_{i,j+1}^k) + \frac{1}{8}\mathcal{T}_{8\lambda}(u_{i,j}^k - u_{i,j+1}^k - 8\tilde{p}_{i,j}^k),$ $q_{i,j}^{k+1} = \tilde{q}_{i,j}^k - \frac{1}{8}(u_{i,j}^k - u_{i+1,j}^k) + \frac{1}{8}\mathcal{T}_{8\lambda}(u_{i,j}^k - u_{i+1,j}^k - 8\tilde{q}_{i,j}^k);$ (c) $t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2};$ (d) ( $\tilde{\mathbf{p}}^{k+1}, \tilde{\mathbf{q}}^{k+1}$ ) = ( $\mathbf{p}^{k+1}, \mathbf{q}^{k+1}$ ) + ( $\frac{t_k - 1}{t_{k+1}}$ ) ( $\mathbf{p}^{k+1} - \mathbf{p}^k, \mathbf{q}^{k+1} - \mathbf{q}^k$ ). 

## 12.5 The Dual Block Proximal Gradient Method

## 12.5.1 Preliminaries

In this section we will consider the problem 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \left\{f (\mathbf {x}) + \sum_ {i = 1} ^ {p} g _ {i} (\mathbf {x}) \right\},\tag{12.27}
$$

where the following assumptions are made. 

Assumption 12.14. 

(A) $f : \mathbb { E } \to ( - \infty , + \infty ]$ is proper closed and σ-strongly convex $( \sigma > 0 )$ 

(B) $g _ { i } : \mathbb { E }  ( - \infty , + \infty ]$ is proper closed and convex for any $i \in \{ 1 , 2 , \ldots , p \}$ 

(C) $\operatorname { r i } ( \operatorname { d o m } ( f ) ) \cap ( \cap _ { i = 1 } ^ { p } \operatorname { r i } ( \operatorname { d o m } ( g _ { i } ) ) ) \neq \emptyset .$ 

Problem (12.27) is actually a generalization of the projection problem discussed in Section 12.4.2, and we can use a similar observation to the one made there and note that problem (12.27) fits model (12.1) with $\mathbb { V } = \mathbb { E } ^ { p } , g ( \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { p } ) =$ $\textstyle \sum _ { i = 1 } ^ { p } g _ { i } ( \mathbf { x } _ { i } )$ , and $\mathcal { A } : \mathbb { E }  \mathbb { V }$ given by 

$$
\mathcal {A} (\mathbf {z}) = (\underbrace {\mathbf {z} , \mathbf {z} , \dots , \mathbf {z}}) \text {   for   any   } \mathbf {z} \in \mathbb {E}.
$$

p times 

Noting that 

$\| A \| ^ { 2 } = p ;$ 

$\begin{array} { r } { \mathcal { A } ^ { T } ( \mathbf { y } ) = \sum _ { i = 1 } ^ { p } y _ { i } } \end{array}$ for any $\mathbf { y } \in \mathbb { E } ^ { p }$ 

• prox<sub>Lg</sub>(v<sub>1</sub>, v<sub>2</sub>, . . . , v<sub>p</sub>) = (prox<sub>Lg</sub> (v<sub>1</sub>), prox<sub>Lg</sub> (v<sub>2</sub>), . . . , prox<sub>Lg</sub> (v<sub>p</sub>)) for any $\mathbf { v } _ { i } \in \mathbb { E } , i = 1 , 2 , \dots , p ,$ 

we can explicitly write the FDPG method with $\begin{array} { r } { L = \frac { \| \mathcal { A } \| ^ { 2 } } { \sigma } = \frac { p } { \sigma } } \end{array}$ 

Algorithm 9 [FDPG for solving (12.27)] 

• Initialization: $\mathbf { w } ^ { 0 } = \mathbf { y } ^ { 0 } \in \mathbb { E } ^ { p } , t _ { 0 } = 1$ 

• General step $( k \geq 0 )$ 

$$
\text {(a)} \mathbf {u} ^ {k} = \operatorname{argmax} _ {\mathbf {u} \in \mathbb {E}} \left\{\langle \mathbf {u}, \sum_ {i = 1} ^ {p} \mathbf {w} _ {i} ^ {k} \rangle - f (\mathbf {u}) \right\};
$$

$$
\mathbf {y} _ {i} ^ {k + 1} = \mathbf {w} _ {i} ^ {k} - \frac {\sigma}{p} \mathbf {u} ^ {k} + \frac {\sigma}{p} \mathrm{prox} _ {\frac {p}{\sigma} g _ {i}} (\mathbf {u} ^ {k} - \frac {p}{\sigma} \mathbf {w} _ {i} ^ {k}), i = 1, 2, \ldots , p;
$$

$$
\mathrm{(c)} t _ {k + 1} = \frac {1 + \sqrt {1 + 4 t _ {k} ^ {2}}}{2};
$$

$$
\text {(d)} \mathbf {w} ^ {k + 1} = \mathbf {y} ^ {k + 1} + \left(\frac {t _ {k} - 1}{t _ {k + 1}}\right) (\mathbf {y} ^ {k + 1} - \mathbf {y} ^ {k}).
$$

The primal sequence is given by 

$$
\mathbf {x} ^ {k} = \operatorname{argmax} _ {\mathbf {x} \in \mathbb {E}} \left\{\left\langle \mathbf {x}, \sum_ {i = 1} ^ {p} \mathbf {y} _ {i} ^ {k} \right\rangle - f (\mathbf {x}) \right\}.
$$

The Dual Block Proximal Gradient (DBPG) Method—dual representation
- Initialization: pick $\mathbf{y}^0 = (\mathbf{y}_1^0, \mathbf{y}_2^0, \ldots, \mathbf{y}_p^0) \in \mathbb{E}^p$ .
- General step ( $k \geq 0$ ):
    - pick an index $i_k \in \{1, 2, \ldots, p\}$ ;
    - compute $\mathbf{y}_j^{k+1} = \begin{cases} \text{prox}_{\sigma G_{i_k}} \left( \mathbf{y}_{i_k}^k - \sigma \nabla f^*(\sum_{j=1}^p \mathbf{y}_j^k) \right), & j = i_k, \\ \mathbf{y}_j^k, & j \neq i_k. \end{cases}$ 

## 12.5.2 The Dual Block Proximal Gradient Method

Note that the stepsize taken at each iteration of Algorithm 9 is $\textstyle { \frac { \sigma } { p } }$ , which might be extremely small when the number of blocks $( p )$ is large. The natural question is therefore whether it is possible to define a dual-based method whose stepsize is independent of the dimension. For that, let us consider the dual of problem (12.27), meaning problem (12.4). Keeping in mind that $\begin{array} { r } { \mathcal { A } ^ { T } ( \mathbf { y } ) = \sum _ { i = 1 } ^ { p } \mathbf { y } _ { i } } \end{array}$ and the fact that $\begin{array} { r } { g ^ { * } ( \mathbf { y } ) = \sum _ { i = 1 } ^ { p } g _ { i } ^ { * } ( \mathbf { y } _ { i } ) } \end{array}$ (see Theorem 4.12), we obtain the following form of the dual problem: 

$$
q _ {\mathrm{opt}} = \max _ {\mathbf {y} \in \mathbb {E} ^ {p}} \left\{- f ^ {*} \left(\sum_ {i = 1} ^ {p} \mathbf {y} _ {i}\right) - \sum_ {i = 1} ^ {p} \underbrace {g _ {i} ^ {*} (- \mathbf {y} _ {i})} _ {G _ {i} (\mathbf {y} _ {i})} \right\}.\tag{12.28}
$$

Since the nonsmooth part in (12.28) is block separable, we can employ a block proximal gradient method (see Chapter 11) on the dual problem (in its minimization form). Suppose that the current point is $\mathbf { \widetilde { y } } ^ { k } = ( \mathbf { y } _ { 1 } ^ { k } , \mathbf { y } _ { 2 } ^ { k } , \ldots , \mathbf { y } _ { p } ^ { k } )$ . At each iteration of a block proximal gradient method we pick an index i according to some rule and perform a proximal gradient step only on the ith block which is thus updated by the formula 

$$
\mathbf {y} _ {i} ^ {k + 1} = \operatorname{prox} _ {\sigma G _ {i}} \left(\mathbf {y} _ {i} ^ {k} - \sigma \nabla f ^ {*} \left(\sum_ {j = 1} ^ {p} \mathbf {y} _ {j} ^ {k}\right)\right).
$$

The stepsize was chosen to be $\sigma$ since f is proper closed and σ-strongly convex, and thus, by the conjugate correspondence theorem (Theorem 5.26), $f ^ { * }$ is <sup>1</sup> -smooth, from which it follows that the block Lipschitz constants of the function $( \mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } , \ldots , \mathbf { y } _ { p } ) \mapsto f ^ { * } ( \sum _ { i = 1 } ^ { p } \mathbf { y } _ { i } )$ are $\textstyle { \frac { 1 } { \sigma } }$ . Thus, the constant stepsize can be taken as $\sigma .$  We can now write a dual representation of the dual block proximal gradient (DBPG) method. 

We can utilize Lemma 12.5 to obtain a primal representation of the general step of the DBPG method. 

Lemma 12.15. Let f and $g _ { 1 } , g _ { 2 } , \ldots , g _ { p }$ satisfy properties $\mathrm { ( A ) }$ and (B) of Assumption 12.14. Let $i \in \{ 1 , 2 , . . . , p \}$ and $G _ { i } ( \mathbf { y } _ { i } ) \equiv g _ { i } ^ { * } ( - \mathbf { y } _ { i } )$ . Let $L > 0$ . Then $\mathbf { y } _ { i } \in \mathbb { E }$ and $\mathbf { v } \in \mathbb { E } ^ { p }$ satisfy the relation 

$$
\mathbf {y} _ {i} = \operatorname{prox} _ {\frac {1}{L} G _ {i}} \left(\mathbf {v} _ {i} - \frac {1}{L} \nabla f ^ {*} \left(\sum_ {j = 1} ^ {p} \mathbf {v} _ {j}\right)\right)
$$

$i f$ and only if 

$$
\mathbf {y} _ {i} = \mathbf {v} _ {i} - \frac {1}{L} \tilde {\mathbf {x}} + \frac {1}{L} \mathrm{prox} _ {L g _ {i}} \left(\tilde {\mathbf {x}} - L \mathbf {v} _ {i}\right),
$$

where 

$$
\tilde {\mathbf {x}} = \operatorname{argmax} _ {\mathbf {x} \in \mathbb {E}} \left\{\left\langle \mathbf {x}, \sum_ {j = 1} ^ {p} \mathbf {v} _ {j} \right\rangle - f (\mathbf {x}) \right\}.
$$

Proof. Follows by invoking Lemma 12.5 with $\begin{array} { r } { \mathbb { V } = \mathbb { E } , \boldsymbol { \mathcal { A } } = \boldsymbol { \mathcal { T } } , \mathbf { b } = \sum _ { j \neq i } \mathbf { v } _ { j } , g = g _ { i } } \end{array}$ $\mathbf { y } = \mathbf { y } _ { i }$ , and $\mathbf { v } = \mathbf { v } _ { i }$ 口

Using Lemma 12.15, we can now write a primal representation of the DBPG method. 

The Dual Block Proximal Gradient (DBPG) Method—primal representation

Initialization: pick $\mathbf{y}^{0} = (\mathbf{y}_{1}^{0}, \mathbf{y}_{2}^{0}, \ldots, \mathbf{y}_{p}^{0}) \in \mathbb{E}$ .

General step: for any $k = 0, 1, 2, \ldots$ execute the following steps:

(a) pick $i_{k} \in \{1, 2, \ldots, p\}$ ;

(b) set $x^{k} = \arg\max_{x \in E} \left\{ \langle x, \sum_{j=1}^{p} y_{j}^{k} \rangle - f(x) \right\}$ ;

(c) set $y_{j}^{k+1} = \begin{cases} y_{i_{k}}^{k} - \sigma x^{k} + \sigma \text{prox}_{g_{i}/\sigma}(x^{k} - y_{i_{k}}^{k}/\sigma), & j = i_{k}, \\ y_{j}^{k}, & j \neq i_{k}. \end{cases}$ 

Note that the derived DBPG method is a functional decomposition method, as it utilizes only one of the functions $g _ { 1 } , g _ { 2 } , \dotsc , g _ { p }$ at each iteration, and in addition the computation involving the function $f ( \mathrm { s t e p ~ ( b ) } )$ does not involve any other function. Thus, we obtained that in this case a variables decomposition method in the dual space gives rise to a functional decomposition method in the primal space. 

What is missing from the above description of the DBPG method is the index selection strategy, meaning the rule for choosing $i _ { k }$ at each iteration. We will consider two variations. 

• Cyclic. $i _ { k } = ( k \mod p ) + 1$ 

• Randomized. $i _ { k }$ is randomly picked from $\{ 1 , 2 , \ldots , p \}$ by a uniform distribution. 

## 12.5.3 Convergence Analysis

The rate of convergence of the DBPG method is a simple consequence of the rates of convergence already established for the block proximal gradient method in Chapter 11 combined with the primal-dual relation presented in Lemma 12.7. 

## Cyclic Block Order

Recall that since the model (12.27) is a special case of the general model (12.1) (with $\begin{array} { r } { \mathbb { V } = \mathbb { E } ^ { p } , \boldsymbol { \mathcal { A } } : \mathbf { z } \mapsto ( \mathbf { z } , \mathbf { z } , \ldots , \mathbf { z } ) , g ( \mathbf { x } ) = \sum _ { i = 1 } ^ { p } g _ { i } ( \mathbf { x } _ { i } ) ) } \end{array}$ , then under Assumption 12.14 the strong duality theorem (Theorem 12.2) holds, and thus the dual problem (12.28) has a nonempty optimal set. We will denote the set of dual optimal solutions by $\Lambda ^ { * }$ . The following assumption is required to present a convergence result for the DBPG method with a cyclic index selection strategy. 

Assumption 12.16. For any $\alpha > 0$ , there exists $R _ { \alpha } > 0$ such that 

$$
\max _ {\mathbf {y}, \mathbf {y} ^ {*} \in \mathbb {E} ^ {p}} \left\{\| \mathbf {y} - \mathbf {y} ^ {*} \|: q (\mathbf {y}) \geq \alpha , \mathbf {y} ^ {*} \in \Lambda^ {*} \right\} \leq R _ {\alpha},
$$

where $\begin{array} { r } { q ( \mathbf { y } ) \equiv - f ^ { * } ( \sum _ { i = 1 } ^ { p } \mathbf { y } _ { i } ) - \sum _ { i = 1 } ^ { p } g _ { i } ^ { * } ( - \mathbf { y } _ { i } ) . } \end{array}$ 

Theorem 12.17 $( O ( 1 / k )$ rate of convergence of DBPG with cyclic order). Suppose that Assumptions 12.14 and 12.16 hold. Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ and $\{ \mathbf { y } ^ { k } \} _ { k \geq 0 }$ be the primal and dual sequences generated by the DBPG method with cyclic index selection strategy for solving problem (12.27). Then for any $k \geq 2$ 2 

$$
\text {(a)} q _ {\mathrm{opt}} - q (\mathbf {y} ^ {p k}) \leq \max \left\{\left(\frac {1}{2}\right) ^ {(k - 1) / 2} (q _ {\mathrm{opt}} - q (\mathbf {y} ^ {0})), \frac {8 p (p + 1) ^ {2} R ^ {2}}{\sigma (k - 1)} \right\};
$$

$$
\text {(b)} \| \mathbf {x} ^ {p k} - \mathbf {x} ^ {*} \| ^ {2} \leq \frac {2}{\sigma} \max \left\{\left(\frac {1}{2}\right) ^ {(k - 1) / 2} (q _ {\mathrm{opt}} - q (\mathbf {y} ^ {0})), \frac {8 p (p + 1) ^ {2} R ^ {2}}{\sigma (k - 1)} \right\}.
$$

In the above two formulas $R = R _ { q ( \mathbf { y } ^ { 0 } ) }$ 

Proof. (a) The proof follows by invoking Theorem 11.18 while taking into account that in this case the constants in (11.24) are given by $\begin{array} { r } { L _ { \operatorname* { m a x } } = L _ { \operatorname* { m i n } } = \frac { 1 } { \sigma } , L _ { f } = \frac { p } { \sigma } } \end{array}$ 

(b) By the primal-dual relation, Lemma 12.7, $\begin{array} { r } { \| \mathbf { x } ^ { p k } - \mathbf { x } ^ { * } \| ^ { 2 } \leq \frac { 2 } { \sigma } ( q _ { \mathrm { o p t } } - \mathbf { \bar { q } } ( \mathbf { y } ^ { p \check { k } } ) ) } \end{array}$ which, combined with part (a), yields the inequality of part (b). 

## Randomized Block Order

A direct result of the $O ( 1 / k )$ rate of convergence of the RBPG method presented in Theorem 11.25 along with the primal-dual relation (Lemma 12.7) yields the following result on the convergence of the DBPG method with random index selection strategy. As in Section 11.5, we will use the notation of the random variable ${ \xi } _ { k } \equiv \left\{ i _ { 0 } , i _ { 1 } , \ldots , i _ { k } \right\}$ . Note that in the randomized setting we do not require $\mathrm { A s } _ { - }$ sumption 12.16 to hold. 

Theorem 12.18 $( O ( 1 / k )$ rate of convergence of DBPG with randomized order). Suppose that Assumption 12.14 holds. Let $\{ \mathbf { x } ^ { k } \} _ { k \geq 0 }$ and $\{ \mathbf { y } ^ { k } \} _ { k \geq 0 }$ be primal and dual sequences generated by the DBPG method with randomized index selection strategy. Then for any $k \geq 0$ 2 

$$
\text {(a)} q _ {\mathrm{opt}} - \mathsf {E} _ {\xi_ {k}} (q (\mathbf {y} ^ {k + 1})) \leq \frac {p}{p + k + 1} \left(\frac {1}{2 \sigma} \| \mathbf {y} ^ {0} - \mathbf {y} ^ {*} \| ^ {2} + q _ {\mathrm{opt}} - q (\mathbf {y} ^ {0})\right);
$$

$$
\text {(b)} \mathsf {E} _ {\xi_ {k}} \| \mathbf {x} ^ {k + 1} - \mathbf {x} ^ {*} \| ^ {2} \leq \frac {2 p}{\sigma (p + k + 1)} \left(\frac {1}{2 \sigma} \| \mathbf {y} ^ {0} - \mathbf {y} ^ {*} \| ^ {2} + q _ {\mathrm{opt}} - q (\mathbf {y} ^ {0})\right).
$$

## 12.5.4 Acceleration in the Two-Block Case<sup>69</sup>

Both the deterministic and the randomized DBPG methods are not accelerated methods, and consequently it was only possible to show that they exhibit an $O ( 1 / k )$ rate of convergence. In the case where $p = 2$ , we will show that it is actually possible to derive an accelerated dual block proximal gradient method by using a simple trick. For that, note that when $p = 2$ , the model amounts to 

$$
f _ {\mathrm{opt}} = \min _ {\mathbf {x} \in \mathbb {E}} \{F (\mathbf {x}) \equiv f (\mathbf {x}) + g _ {1} (\mathbf {x}) + g _ {2} (\mathbf {x}) \}.\tag{12.29}
$$

We can rewrite the problem as 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \{\tilde {f} (\mathbf {x}) + g _ {2} (\mathbf {x}) \},\tag{12.30}
$$

where $\tilde { f } = f + g _ { 1 }$ . If Assumption 12.14 holds with $p = 2 .$ , then $\tilde { f }$ is proper closed and σ-strongly convex, $g _ { 2 }$ is proper closed and convex, and the regularity condition $\operatorname { r i } ( \operatorname { d o m } ( { \tilde { f } } ) ) \cap \operatorname { r i } ( \operatorname { d o m } ( g _ { 2 } ) ) \neq \varnothing$ is satisfied. This means that Assumption 12.1 holds for $f = { \bar { f } } , g = g _ { 2 }$ , and $\mathcal A = \mathcal T$ . We can now define the accelerated dual block proximal gradient (ADBPG), which is the FDPG method with stepsize $\sigma$ employed on the model (12.1) with $f = \tilde { f } , g = g _ { 2 }$ , and $\mathcal A = \mathcal T$ 

The ADBPG Method
Initialization: $w^{0} = y^{0} \in E, t_{0} = 1$ .
General step ( $k \geq 0$ ):
(a) $u^{k} = \arg\max_{u} \left\{ \langle u, w^{k} \rangle - f(u) - g_{1}(u) \right\}$ ;
(b) $y^{k+1} = w^{k} - \sigma u^{k} + \sigma \text{prox}_{g_{2}/\sigma}(u^{k} - w^{k}/\sigma)$ ;
(c) $t_{k+1} = \frac{1 + \sqrt{1 + 4t_{k}^{2}}}{2}$ ;
(d) $w^{k+1} = y^{k+1} + \left(\frac{t_{k}-1}{t_{k+1}}\right)(y^{k+1} - y^{k})$ . 

A direct consequence of Theorem 12.10 is the following result on the rate of convergence of the ADBPG method. 

Theorem 12.19 $( O ( 1 / k ^ { 2 } )$ rate of convergence of ADBPG). Suppose that Assumption 12.14 holds with $p = 2$ , and let $\{ \mathbf { y } ^ { k } \} _ { k \geq 0 }$ be the sequence generated by the ADBPG method. Then for any optimal solution $\mathbf { y } ^ { * }$ of the dual problem 

$$
\min _ {\mathbf {y} \in \mathbb {E}} \{(\tilde {f}) ^ {*} (\mathbf {y}) + g _ {2} ^ {*} (- \mathbf {y}) \}
$$

and $k \geq 1$ , it holds that 

$$
\left\| \mathbf {x} ^ {k} - \mathbf {x} ^ {*} \right\| ^ {2} \leq \frac {4 \left\| \mathbf {y} ^ {0} - \mathbf {y} ^ {*} \right\| ^ {2}}{\sigma^ {2} (k + 1) ^ {2}},
$$

where $\mathbf { x } ^ { k } = \operatorname { a r g m a x } _ { \mathbf { x } } \left\{ \langle \mathbf { x } , \mathbf { y } ^ { k } \rangle - f ( \mathbf { x } ) - g _ { 1 } ( \mathbf { x } ) \right\}$ 

Remark 12.20. When $\begin{array} { r } { f ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { x } - \mathbf { d } \| ^ { 2 } } \end{array}$ for some d $\in \mathbb { E }$ , step (a) of the ADBPG can be written as a prox computation: 

$$
\mathbf {u} ^ {k} = \mathrm{prox} _ {g _ {1}} (\mathbf {d} + \mathbf {w} ^ {k}).
$$

Remark 12.21. Note that the ADBPG is not a full functional decomposition method since step (a) is a computation involving both f and $g _ { 1 }$ , but it still separates between $g _ { 1 }$ and $g _ { 2 }$ . The method has two main features. First, it is an accelerated method. Second, the stepsize taken in the method is $\sigma _ { i }$ , in contrast to the stepsize of $\textstyle { \frac { \sigma } { 2 } }$ that is used in Algorithm $9 _ { i }$ which is another type of an FDPG method. 

## 12.6 Examples II

Example 12.22 (one-dimensional total variation denoising). In this example we will compare the performance of the ADBPG method and Algorithm 9 (with $p = 2 )$ —both are FDPG methods, although quite diferent. We will consider the one-dimensional total variation problem (see also Section 12.4.3) 

$$
f _ {\text { opt }} = \min _ {\mathbf {x} \in \mathbb {R} ^ {n}} \left\{F (\mathbf {x}) \equiv \frac {1}{2} \| \mathbf {x} - \mathbf {d} \| _ {2} ^ {2} + \lambda \sum_ {i = 1} ^ {n - 1} | x _ {i - 1} - x _ {i} | \right\},\tag{12.31}
$$

where d $\in \mathbb { R } ^ { n }$ and $\lambda > 0$ . The above problem can be written as 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {n}} \{f (\mathbf {x}) + g _ {1} (\mathbf {x}) + g _ {2} (\mathbf {x}) \},
$$

where 

$$
f (\mathbf {x}) = \frac {1}{2} \| \mathbf {x} - \mathbf {d} \| _ {2} ^ {2},
$$

$$
g _ {1} (\mathbf {x}) = \lambda \sum_ {i = 1} ^ {\lfloor \frac {n}{2} \rfloor} | x _ {2 i - 1} - x _ {2 i} |,
$$

$$
g _ {2} (\mathbf {x}) = \lambda \sum_ {i = 1} ^ {\lfloor \frac {n - 1}{2} \rfloor} | x _ {2 i} - x _ {2 i + 1} |.
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/bec5368e778fc5bf3acf9ced09ddbd880659d528e2fdbe3592984de1a75b5e89.jpg)



Figure 12.4. Comparison of the ADBPG method and Algorithm 9 employed on the one-dimensional total variation denoising problem.


By Example 6.17 we have that the prox of λ times the two-dimensional function $h ( y , z ) = | y - z |$ is given by 

$$
\begin{array}{c} \operatorname{prox} _ {\lambda h} (y, z) = (y, z) + \frac {1}{2 \lambda^ {2}} (\mathcal {T} _ {2 \lambda^ {2}} (\lambda y - \lambda z) - \lambda y + \lambda z) (\lambda , - \lambda) \\ = (y, z) + \frac {1}{2} ([ | y - z | - 2 \lambda ] _ {+} \mathrm{sgn} (y - z) - y + z) (1, - 1). \end{array}
$$

Therefore, using the separability of $g _ { 1 }$ w.r.t. the pairs of variables $\{ x _ { 1 } , x _ { 2 } \}$ $\{ x _ { 3 } , x _ { 4 } \} , \ldots ,$ it follows that 

$$
\operatorname{prox} _ {g _ {1}} (\mathbf {x}) = \mathbf {x} + \frac {1}{2} \sum_ {i = 1} ^ {\lfloor \frac {n}{2} \rfloor} ([ | x _ {2 i - 1} - x _ {2 i} | - 2 \lambda ] _ {+} \operatorname{sgn} (x _ {2 i - 1} - x _ {2 i}) - x _ {2 i - 1} + x _ {2 i}) (\mathbf {e} _ {2 i - 1} - \mathbf {e} _ {2 i}),
$$

and similarly 

$$
\mathrm{prox} _ {g _ {2}} (\mathbf {x}) = \mathbf {x} + \frac {1}{2} \sum_ {i = 1} ^ {\lfloor \frac {n - 1}{2} \rfloor} ([ | x _ {2 i} - x _ {2 i + 1} | - 2 \lambda ] _ {+} \mathrm{sgn} (x _ {2 i} - x _ {2 i + 1}) - x _ {2 i} + x _ {2 i + 1}) (\mathbf {e} _ {2 i} - \mathbf {e} _ {2 i + 1}).
$$

Equipped with the above expressions for $\operatorname { p r o x } _ { g _ { 1 } }$ and $\operatorname { p r o x } _ { g _ { 2 } }$ (recalling that step (a) only requires a single computation of $\operatorname { p r o x } _ { g _ { 1 } } ;$ see Remark 12.20), we can employ the ADBPG method and Algorithm 9 on problem (12.31). The computational efort per iteration in both methods is almost identical and is dominated by single evaluations of the prox mappings of $g _ { 1 }$ and $g _ { 2 }$ . We ran 1000 iterations of both algorithms starting with a dual vector which is all zeros. In Figure 12.4 we plot the distance in function values<sup>70</sup> $F ( \mathbf { x } ^ { k } ) - f _ { \mathrm { o p t } }$ as a function of the iteration index k. Evidently, the ADBPG method exhibits the superior performance. Most likely, the reason is the fact that the ADBPG method uses a larger stepsize (σ) than the one used by Algorithm 9 ( <sup>σ</sup> ). 

Example 12.23 (two-dimensional total variation denoising). Consider the isotropic two-dimensional total variation problem 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {m \times n}} \left\{\frac {1}{2} \| \mathbf {x} - \mathbf {d} \| _ {F} ^ {2} + \lambda \mathrm{TV} _ {I} (\mathbf {x}) \right\},
$$

where d $\in \mathbb { R } ^ { m \times n } , \lambda > 0$ , and $\mathrm { T V } _ { I }$ is given in (12.26). It does not seem possible to decompose $\mathrm { T V } _ { I }$ into two functions whose prox can be directly computed as in the one-dimensional case. However, a decomposition into three separable functions (w.r.t. triplets of variables) is possible. To describe the decomposition, we introduce the following notation. Let $D _ { k }$ denote the set of indices that correspond to the elements of the kth diagonal of an $m \times n$ matrix, where $D _ { 0 }$ represents the indices set of the main diagonal, and $D _ { k }$ for $k > 0$ and $k < 0$ stand for the diagonals above and below the main diagonal, respectively. In addition, consider the partition of the diagonal indices set, $\{ - ( m - 1 ) , \ldots , n - 1 \}$ , into three sets 

$$
K _ {i} \equiv \bigl \{k \in \{- (m - 1), \ldots , n - 1 \}: (k + 1 - i) \mod 3 = 0 \bigr \}, \qquad i = 1, 2, 3.
$$

With the above notation, we are now ready to write the function $\mathrm { T V } _ { I }$ as 

$$
\begin{array}{l} \mathrm{TV} _ {I} (\mathbf {x}) = \sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n} \sqrt {(x _ {i , j} - x _ {i , j + 1}) ^ {2} + (x _ {i , j} - x _ {i + 1 , j}) ^ {2}} \\ \qquad = \sum_ {k \in K _ {1}} \sum_ {(i, j) \in D _ {k}} \sqrt {(x _ {i , j} - x _ {i , j + 1}) ^ {2} + (x _ {i , j} - x _ {i + 1 , j}) ^ {2}} \\ \qquad + \sum_ {k \in K _ {2}} \sum_ {(i, j) \in D _ {k}} \sqrt {(x _ {i , j} - x _ {i , j + 1}) ^ {2} + (x _ {i , j} - x _ {i + 1 , j}) ^ {2}} \\ \qquad + \sum_ {k \in K _ {3}} \sum_ {(i, j) \in D _ {k}} \sqrt {(x _ {i , j} - x _ {i , j + 1}) ^ {2} + (x _ {i , j} - x _ {i + 1 , j}) ^ {2}} \\ \qquad = \psi_ {1} (\mathbf {x}) + \psi_ {2} (\mathbf {x}) + \psi_ {3} (\mathbf {x}), \end{array}
$$

where we assume in the above expressions that $x _ { i , n + 1 } = x _ { i , n }$ and $x _ { m + 1 , j } = x _ { m , j }$ The fact that each of the functions $\psi _ { i }$ is separable w.r.t. triplets of variables $\{ x _ { i , j }$ $x _ { i + 1 , j } , x _ { i , j + 1 } \}$ is evident from the illustration in Figure 12.5. 

The denoising problem can thus be rewritten as 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {m \times n}} \left\{\frac {1}{2} \| \mathbf {x} - \mathbf {d} \| _ {F} ^ {2} + \lambda \psi_ {1} (\mathbf {x}) + \lambda \psi_ {2} (\mathbf {x}) + \lambda \psi_ {3} (\mathbf {x}) \right\}.
$$

It is not possible to employ the ADBPG method since the nonsmooth part is decomposed into three functions. However, it is possible to employ the DBPG method, which has no restriction on the number of functions. The algorithm requires evaluating a prox mapping of one of the functions $\lambda \psi _ { i }$ at each iteration. By the separability of these functions, it follows that each prox computation involves several prox computations of three-dimensional functions of the form $\lambda h$ , where 

$$
h (x, y, z) = \sqrt {(x - y) ^ {2} + (x - z) ^ {2}}.
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/d3844e068a46b70f234e838b68958a7eaf91e6939f51921eae6d2fecea6f6e83.jpg)


Figure 12.5. The decomposition of a $1 6 \times 1 2$ pixels Mario image according to the isotropic TV into three separable functions. The images are partitioned into blocks of three pixels positioned in an r-shaped structure. Each block encompasses the three pixels that form the term $\sqrt { ( x _ { i , j } - x _ { i + 1 , j } ) ^ { 2 } + ( x _ { i , j } - x _ { i , j + 1 } ) ^ { 2 } }$ . Summing over all the terms represented by the blocks of any of the above images yields the appropriate separable function. Reprinted with permission from Elsevier. [23] 

The prox of λh can be computed using Lemma 6.68 and is given by 

$$
\operatorname{prox} _ {\lambda h} (\mathbf {x}) = \left\{ \begin{array}{l l} \mathbf {x} - \mathbf {A} ^ {T} (\mathbf {A A} ^ {T}) ^ {- 1} \mathbf {A x}, & \| (\mathbf {A A} ^ {T}) ^ {- 1} \mathbf {A x} \| _ {2} \leq \lambda , \\ \mathbf {x} - \mathbf {A} ^ {T} (\mathbf {A A} ^ {T} + \alpha^ {*} \mathbf {I}) ^ {- 1} \mathbf {A x}, & \| (\mathbf {A A} ^ {T}) ^ {- 1} \mathbf {A x} \| _ {2} > \lambda , \end{array} \right.
$$

where $\alpha ^ { * }$ is the unique root of the decreasing function 

$$
g (\alpha) = \| (\mathbf {A A} ^ {T} + \alpha^ {*} \mathbf {I}) ^ {- 1} \mathbf {A x} \| _ {2} ^ {2} - \lambda^ {2}
$$

and A is the matrix 

$$
\mathbf {A} = \left( \begin{array}{c c c} 1 & - 1 & 0 \\ 1 & 0 & - 1 \end{array} \right). \quad \blacksquare
$$

# Chapter 13 The Generalized Conditional Gradient Method

Underlying Spaces: In this chapter, all the underlying spaces are Euclidean. 

## 13.1 The Frank–Wolfe/Conditional Gradient Method

Consider the problem 

$$
\min \{f (\mathbf {x}): \mathbf {x} \in C \},\tag{13.1}
$$

where $C \subseteq \mathbb { E }$ is a nonempty convex and compact set and $f : \mathbb { E }  ( - \infty , \infty ]$ is a convex function satisfying $C \subseteq \operatorname { d o m } ( f )$ . We further assume that dom(f) is open and that $f$ is diferentiable over dom(f). One method that can be employed in order to solve the problem is the projected gradient method (see Section 10.2) whose update step is 

$$
\mathbf {x} ^ {k + 1} = P _ {C} (\mathbf {x} ^ {k} - t _ {k} \nabla f (\mathbf {x} ^ {k})),
$$

with $t _ { k }$ being an appropriately chosen stepsize. In this chapter we will consider an alternative approach that does not require the evaluation of the orthogonal projection operator at each iteration. Instead, the approach, known as the conditional gradient method or $F r a n k  – W o l f e$ algorithm, computes the next step as a convex combination of the current iterate and a minimizer of a linearized version of the objective function over C. 

The Conditional Gradient Method
Initialization: pick $x^{0} \in C$ .
General step: for any $k = 0, 1, 2, \ldots$ execute the following steps:
(a) compute $p^{k} \in \arg\min_{p \in C} \langle p, \nabla f(x^{k}) \rangle$ ;
(b) choose $t_{k} \in [0, 1]$ and set $x^{k+1} = x^{k} + t_{k}(p^{k} - x^{k})$ . 

The conditional gradient approach is potentially beneficial in cases where computation of a linear oracle over the feasible set (that is, computation of a minimizer of a linear function over $C )$ is a simpler task than evaluating the orthogonal projection onto $C .$ . We will actually analyze an extension of the method that tackles the problem of minimizing a composite function $f + g ,$ where the case $g = \delta _ { C }$ brings us back to the model (13.1). 

## 13.2 The Generalized Conditional Gradient Method

## 13.2.1 Model and Method

Consider the composite problem 

$$
\min \left\{F (\mathbf {x}) \equiv f (\mathbf {x}) + g (\mathbf {x}) \right\},\tag{13.2}
$$

where we assume the following set of properties. 

Assumption 13.1. 

(A) $g : \mathbb { E }  ( - \infty , \infty ]$ is proper closed and convex and dom $( g )$ is compact. 

(B) $f : \mathbb { E } \to ( - \infty , \infty ]$ is $L _ { f }$ -smooth over dom $( f ) \ ( L _ { f } > 0 )$ , which is assumed to be an open and convex set satisfying dom $( g ) \subseteq \operatorname { d o m } ( f )$ 

(C) The optimal set of problem (13.2) is nonempty and denoted by $X ^ { * }$ . The optimal value of the problem is denoted by $F _ { \mathrm { o p t } }$ 

It is not dificult to deduce that property (C) is implied by properties $\mathrm { ( A ) }$ and (B). The generalized conditional gradient method for solving the composite model (13.2) is similar to the conditional gradient method, but instead of linearizing the entire objective function, the algorithm computes a minimizer of the sum of the linearized smooth part $f$ around the current iterate and leaves $g$ unchanged. 

The Generalized Conditional Gradient Method
Initialization: pick $\mathbf{x}^{0}\in\operatorname{dom}(g)$ .
General step: for any $k=0,1,2,\ldots$ execute the following steps:
(a) compute $p^{k} \in \arg\min_{p \in E} \left\{ \langle p, \nabla f(x^{k}) \rangle + g(p) \right\}$ ;
(b) choose $t_{k} \in [0,1]$ and set $x^{k+1} = x^{k} + t_{k}(p^{k} - x^{k})$ . 

## 13.2.2 The Conditional Gradient Norm

Throughout this chapter we will use the following notation: 

$$
\mathbf {p} (\mathbf {x}) \in \operatorname{argmin} _ {\mathbf {p}} \left\{\langle \mathbf {p}, \nabla f (\mathbf {x}) \rangle + g (\mathbf {p}) \right\}.\tag{13.3}
$$

Of course, $\mathbf { p } ( \mathbf { x } )$ is not uniquely defined in the sense that the above minimization problem might have multiple optimal solutions. We assume that there exists some rule for choosing an optimal solution whenever the optimal set of (13.3) is not a singleton and that the vector $\mathbf { p } ^ { k }$ computed by the generalized conditional gradient method is chosen by the same rule, meaning that $\mathbf { \bar { p } } ^ { k } = \mathbf { p } ( \mathbf { x } ^ { k } )$ . We can write the update step of the generalized conditional gradient method as 

$$
\mathbf {x} ^ {k + 1} = \mathbf {x} ^ {k} + t _ {k} (\mathbf {p} (\mathbf {x} ^ {k}) - \mathbf {x} ^ {k}).
$$

A natural optimality measure in the context of proximal gradient methods is the gradient mapping (see Section 10.3.2). However, the analysis of the conditional gradient method relies on a diferent optimality measure, which we will refer to as the conditional gradient norm. 

Definition 13.2 (conditional gradient norm). Suppose that f and g satisfy properties (A) and (B) of Assumption 13.1. Then the conditional gradient norm is the function S : dom(f) → R defined by 

$$
S (\mathbf {x}) \equiv \langle \nabla f (\mathbf {x}), \mathbf {x} - \mathbf {p} (\mathbf {x}) \rangle + g (\mathbf {x}) - g (\mathbf {p} (\mathbf {x})).
$$

Remark 13.3. The conditional gradient norm obviously depends on f and $^ { g , }$ , so a more precise notation would be $S ^ { f , g } ( \mathbf { x } )$ . However, since the identities of f and g will be clear from the context, we will keep the notation $S ( \mathbf { x } )$ 

Remark 13.4. By the definition of $\mathbf { p } ( \mathbf { x } )$ (equation (13.3)), we can also write $S ( \mathbf { x } )$ as 

$$
S (\mathbf {x}) = \max _ {\mathbf {p} \in \mathbb {E}} \left\{\langle \nabla f (\mathbf {x}), \mathbf {x} - \mathbf {p} \rangle + g (\mathbf {x}) - g (\mathbf {p}) \right\}.\tag{13.4}
$$

The following lemma shows how to write the conditional gradient norm in terms of the conjugate of $g .$ . 

Lemma 13.5. Suppose that f and g satisfy properties (A) and (B) of Assumption 13.1. Then for any $\mathbf { x } \in \operatorname { d o m } ( f )$ ， 

$$
S (\mathbf {x}) = \langle \nabla f (\mathbf {x}), \mathbf {x} \rangle + g (\mathbf {x}) + g ^ {*} (- \nabla f (\mathbf {x})).\tag{13.5}
$$

Proof. Follows by the definition of the conjugate function: 

$$
\begin{array}{l} S (\mathbf {x}) = \max _ {\mathbf {p} \in \mathbb {E}} \{\langle \nabla f (\mathbf {x}), \mathbf {x} - \mathbf {p} \rangle + g (\mathbf {x}) - g (\mathbf {p}) \} \\ \qquad = \langle \nabla f (\mathbf {x}), \mathbf {x} \rangle + g (\mathbf {x}) + \max _ {\mathbf {p} \in \mathbb {E}} \{\langle - \nabla f (\mathbf {x}), \mathbf {p} \rangle - g (\mathbf {p}) \} \\ \qquad = \langle \nabla f (\mathbf {x}), \mathbf {x} \rangle + g (\mathbf {x}) + g ^ {*} (- \nabla f (\mathbf {x})). \quad \square \end{array}
$$

A direct consequence of Lemma 13.5 is that $S ( \cdot )$ is an optimality measure in the sense that it is always nonnegative and is equal to zero only at stationary points of problem (13.2). 

Theorem 13.6 (conditional gradient norm as an optimality measure). Suppose that f and g satisfy properties (A) and (B) of Assumption 13.1. Then 

(a) $S ( \mathbf { x } ) \geq 0$ for any $\mathbf { x } \in \operatorname { d o m } ( f )$ 

(b) $S ( \mathbf { x } ^ { * } ) = 0$ if and only $i f - \nabla f ( \mathbf { x } ^ { * } ) \in \partial g ( \mathbf { x } ^ { * } )$ , that is, if and only if $\mathbf { x } ^ { * }$ is a stationary point of problem (13.2). 

Proof. (a) Follows by the expression (13.5) for the conditional gradient norm and Fenchel’s inequality (Theorem 4.6). 

(b) By part (a), it follows that $S ( \mathbf { x } ^ { * } ) = 0$ if and only if $S ( \mathbf { x } ^ { * } ) \leq 0$ , which is the same as the relation (using the expression (13.4) for $S ( \mathbf { x } ^ { * } ) )$ ) 

$$
\langle \nabla f (\mathbf {x} ^ {*}), \mathbf {x} ^ {*} - \mathbf {p} \rangle + g (\mathbf {x} ^ {*}) - g (\mathbf {p}) \leq 0 \text {   for   all   } \mathbf {p} \in \mathbb {E}.
$$

After some rearrangement of terms, the above can be rewritten as 

$$
g (\mathbf {p}) \geq g \left(\mathbf {x} ^ {*}\right) + \langle - \nabla f \left(\mathbf {x} ^ {*}\right), \mathbf {p} - \mathbf {x} ^ {*} \rangle ,
$$

which is equivalent to the relation $- \nabla f ( \mathbf { x } ^ { * } ) \in \partial g ( \mathbf { x } ^ { * } )$ , namely, to stationarity (see Definition 3.73). 

The basic inequality that will be used in the analysis of the generalized conditional gradient method is the following recursive inequality. 

Lemma 13.7 (fundamental inequality for generalized conditional gradient). Suppose that f and g satisfy properties $o f \left( \mathrm { A } \right)$ and (B) of Assumption 13.1. $L e t \mathbf { x } \in \mathrm { d o m } ( g )$ and $t \in [ 0 , 1 ]$ . Then 

$$
F (\mathbf {x} + t (\mathbf {p} (\mathbf {x}) - \mathbf {x})) \leq F (\mathbf {x}) - t S (\mathbf {x}) + \frac {t ^ {2} L _ {f}}{2} \| \mathbf {p} (\mathbf {x}) - \mathbf {x} \| ^ {2}.\tag{13.6}
$$

Proof. Using the descent lemma (Lemma 5.7), the convexity of $^ { g , }$ and the notation $\mathbf { p } ^ { + } = \mathbf { p } ( \mathbf { x } )$ , we can write the following: 

$$
\begin{array}{r l} & F (\mathbf {x} + t (\mathbf {p} ^ {+} - \mathbf {x})) = f (\mathbf {x} + t (\mathbf {p} ^ {+} - \mathbf {x})) + g (\mathbf {x} + t (\mathbf {p} ^ {+} - \mathbf {x})) \\ & \quad \leq f (\mathbf {x}) - t \langle \nabla f (\mathbf {x}), \mathbf {x} - \mathbf {p} ^ {+} \rangle + \frac {t ^ {2} L _ {f}}{2} \| \mathbf {p} ^ {+} - \mathbf {x} \| ^ {2} + g ((1 - t) \mathbf {x} + t \mathbf {p} ^ {+}) \\ & \quad \leq f (\mathbf {x}) - t \langle \nabla f (\mathbf {x}), \mathbf {x} - \mathbf {p} ^ {+} \rangle + \frac {t ^ {2} L _ {f}}{2} \| \mathbf {p} ^ {+} - \mathbf {x} \| ^ {2} + (1 - t) g (\mathbf {x}) + t g (\mathbf {p} ^ {+}) \\ & \quad = F (\mathbf {x}) - t (\langle \nabla f (\mathbf {x}), \mathbf {x} - \mathbf {p} ^ {+} \rangle + g (\mathbf {x}) - g (\mathbf {p} ^ {+})) + \frac {t ^ {2} L _ {f}}{2} \| \mathbf {p} ^ {+} - \mathbf {x} \| ^ {2} \\ & \quad = F (\mathbf {x}) - t S (\mathbf {x}) + \frac {t ^ {2} L _ {f}}{2} \| \mathbf {p} ^ {+} - \mathbf {x} \| ^ {2}. \quad \square \end{array}
$$

## 13.2.3 Convergence Analysis in the Nonconvex Case

Note that we do not assume at this point that $f$ is convex, and therefore convergence (if any) will be proven to stationary points. Before we delve into the convergence analysis, we mention the diferent options of stepsize strategies that will be considered. 

• Predefined diminishing stepsize. $\begin{array} { r } { t _ { k } = \frac { 2 } { k + 2 } } \end{array}$ 

• Adaptive stepsize. $\begin{array} { r } { t _ { k } = \operatorname* { m i n } \left\{ 1 , \frac { S ( \mathbf { x } ^ { k } ) } { L _ { f } \| \mathbf { p } ^ { k } - \mathbf { x } ^ { k } \| ^ { 2 } } \right\} } \end{array}$ 

• Exact line search. $\begin{array} { r } { t _ { k } \in \mathrm { a r g m i n } _ { t \in [ 0 , 1 ] } F ( \mathbf { x } ^ { k } + t ( \mathbf { p } ^ { k } - \mathbf { x } ^ { k } ) ) . } \end{array}$ 

The motivation for considering the adaptive stepsize comes from the fundamental inequality (13.6)—it is easy to verify that $t _ { k } =$ min $\begin{array} { r } { \left\{ 1 , \frac { S ( \mathbf { x } ^ { k } ) } { L _ { f } \| \mathbf { p } ^ { k } - \mathbf { x } ^ { k } \| ^ { 2 } } \right\} } \end{array}$ is the minimizer of the right-hand side of (13.6) w.r.t. $t \in [ 0 , 1 ]$ when $\mathbf { x } = \mathbf { x } ^ { k }$ . Much like the analysis of the proximal gradient method, the convergence of the generalized conditional gradient method is based on a suficient decrease property. 

Lemma 13.8 (suficient decrease for the generalized conditional gradient method). Suppose that f and g satisfy properties (A) and (B) of Assumption 13.1, and let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the generalized conditional gradient method for solving problem (13.2) with stepsizes chosen by either the adaptive or exact line search strategies. Then for any $k \geq 0$ 

$$
F (\mathbf {x} ^ {k}) - F (\mathbf {x} ^ {k + 1}) \geq \frac {1}{2} \min \left\{S (\mathbf {x} ^ {k}), \frac {S ^ {2} (\mathbf {x} ^ {k})}{L _ {f} \Omega^ {2}} \right\},\tag{13.7}
$$

where Ω be an upper bound on the diameter of dom(g): 

$$
\Omega \geq \max _ {\mathbf {x}, \mathbf {y} \in \operatorname{dom} (g)} \| \mathbf {x} - \mathbf {y} \|.
$$

Proof. Let $k \geq 0$ and let $\tilde { \mathbf { x } } ^ { k } = \mathbf { x } ^ { k } + s _ { k } \big ( \mathbf { p } ^ { k } - \mathbf { x } ^ { k } \big )$ , where 

$$
s _ {k} = \min \left\{1, \frac {S (\mathbf {x} ^ {k})}{L _ {f} \| \mathbf {x} ^ {k} - \mathbf {p} ^ {k} \| ^ {2}} \right\}.
$$

By the fundamental inequality (13.6) invoked with $\mathbf { x } = \mathbf { x } ^ { k }$ and $t = s _ { k }$ , we have 

$$
F (\mathbf {x} ^ {k}) - F (\tilde {\mathbf {x}} ^ {k}) \geq s _ {k} S (\mathbf {x} ^ {k}) - \frac {s _ {k} ^ {2} L _ {f}}{2} \| \mathbf {p} ^ {k} - \mathbf {x} ^ {k} \| ^ {2}.\tag{13.8}
$$

There are two options: Either $\begin{array} { r } { \frac { S ( \mathbf { x } ^ { k } ) } { L _ { f } \| \mathbf { x } ^ { k } - \mathbf { p } ^ { k } \| ^ { 2 } } \leq 1 } \end{array}$ , and in this case $\begin{array} { r } { s _ { k } = \frac { S ( \mathbf { x } ^ { k } ) } { L _ { f } \| \mathbf { x } ^ { k } - \mathbf { p } ^ { k } \| ^ { 2 } } . } \end{array}$ and hence, by (13.8), 

$$
F (\mathbf {x} ^ {k}) - F (\tilde {\mathbf {x}} ^ {k}) \geq \frac {S ^ {2} (\mathbf {x} ^ {k})}{2 L _ {f} \| \mathbf {p} ^ {k} - \mathbf {x} ^ {k} \| ^ {2}} \geq \frac {S ^ {2} (\mathbf {x} ^ {k})}{2 L _ {f} \Omega^ {2}}.
$$

${ \mathrm { O r } } ,$ on the other hand, if 

$$
\frac {S (\mathbf {x} ^ {k})}{L _ {f} \| \mathbf {x} ^ {k} - \mathbf {p} ^ {k} \| ^ {2}} \geq 1,\tag{13.9}
$$

then $s _ { k } = 1$ , and by (13.8), 

$$
F (\mathbf {x} ^ {k}) - F (\tilde {\mathbf {x}} ^ {k}) \geq S (\mathbf {x} ^ {k}) - \frac {L _ {f}}{2} \| \mathbf {p} ^ {k} - \mathbf {x} ^ {k} \| ^ {2} \stackrel {(1 3. 9)} {\geq} \frac {1}{2} S (\mathbf {x} ^ {k}).
$$

Combining the two cases, we obtain 

$$
F (\mathbf {x} ^ {k}) - F (\tilde {\mathbf {x}} ^ {k}) \geq \frac {1}{2} \min \left\{S (\mathbf {x} ^ {k}), \frac {S ^ {2} (\mathbf {x} ^ {k})}{L _ {f} \Omega^ {2}} \right\}.\tag{13.10}
$$

If the adaptive stepsize strategy is used, then $\tilde { \mathbf { x } } ^ { k } = \mathbf { x } ^ { k + 1 }$ and (13.10) is the same as (13.7). If an exact line search strategy is employed, then 

$$
F (\mathbf {x} ^ {k + 1}) = \min _ {t \in [ 0, 1 ]} F (\mathbf {x} ^ {k} + t (\mathbf {p} ^ {k} - \mathbf {x} ^ {k})) \leq F (\mathbf {x} ^ {k} + s _ {k} (\mathbf {p} ^ {k} - \mathbf {x} ^ {k})) = F (\tilde {\mathbf {x}} ^ {k}),
$$

which, combined with (13.10), implies that also in this case (13.7) holds. 

Using Lemma 13.8 we can establish the main convergence result for the generalized conditional gradient method with stepsizes chosen by either the adaptive or exact line search strategies. 

Theorem 13.9 (convergence of the generalized conditional gradient). Suppose that Assumption 13.1 holds, and let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the generalized conditional gradient method for solving problem (13.2) with stepsizes chosen by either the adaptive or exact line search strategies. Then 

(a) for any $k \geq 0 , F ( \mathbf { x } ^ { k } ) \geq F ( \mathbf { x } ^ { k + 1 } )$ and $F ( \mathbf { x } ^ { k } ) > F ( \mathbf { x } ^ { k + 1 } ) \ i f \ \mathbf { x } ^ { k }$ is not a stationary point of problem (13.2); 

(b) $S ( \mathbf { x } ^ { k } )  0$ as $k  \infty ;$ 

(c) for any $k \geq 0$ 

$$
\min _ {n = 0, 1, \dots , k} S (\mathbf {x} ^ {n}) \leq \max \left\{\frac {2 (F (\mathbf {x} ^ {0}) - F _ {\mathrm{opt}})}{k + 1}, \frac {\sqrt {2 L _ {f} \Omega^ {2} (F (\mathbf {x} ^ {0}) - F _ {\mathrm{opt}})}}{\sqrt {k + 1}} \right\},\tag{13.11}
$$

where Ω is an upper bound on the diameter of dom(g); 

(d) all limit points of the sequence $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ are stationary points of problem (13.2). 

Proof. (a) The monotonicity of $\{ F ( \mathbf { x } ^ { k } ) \} _ { k \geq 0 }$ is a direct result of the suficient decrease inequality (13.7) and the nonnegativity of $S ( \mathbf { x } ^ { k } )$ (Theorem $\mathrm { 1 3 . 6 ( a ) } )$ . As for the second claim, if $\mathbf { x } ^ { k }$ is not a stationary point of problem (13.2), then $S ( \mathbf { x } ^ { k } ) > 0$ (see Theorem 13.6(b)), and hence, by the suficient decrease inequality, $F ( \mathbf { x } ^ { k } ) >$ $\dot { F } ( \mathbf { x } ^ { k + 1 } )$ 

(b) Since $\{ F ( \mathbf { x } ^ { k } ) \} _ { k \geq 0 }$ is nonincreasing and bounded below (by $F _ { \mathrm { o p t } } )$ , it follows that it is convergent, and in particular, $F ( \mathbf { x } ^ { k } ) - F ( \mathbf { x } ^ { k + 1 } ) \to 0 { \mathrm { ~ a s ~ } } k \to \infty$ . Therefore, by the suficient decrease inequality (13.7), it follows that min $\begin{array} { r } { \{ S ( \mathbf { x } ^ { k } ) , \frac { S ^ { 2 } ( \mathbf { x } ^ { k } ) } { L _ { f } \Omega ^ { 2 } } \}  0 } \end{array}$ as $k \to \infty$ , implying that $S ( \mathbf { x } ^ { k } ) \to 0 { \mathrm { ~ a s ~ } } k \to \infty$ 

(c) By the suficient decrease inequality (13.7), for all $n \geq 0$ ， 

$$
F (\mathbf {x} ^ {n}) - F (\mathbf {x} ^ {n + 1}) \geq \frac {1}{2} \min \left\{S (\mathbf {x} ^ {n}), \frac {S ^ {2} (\mathbf {x} ^ {n})}{L _ {f} \Omega^ {2}} \right\}.\tag{13.12}
$$

Summing the above inequality over $n = 0 , 1 , \ldots , k$ 2 

$$
F (\mathbf {x} ^ {0}) - F (\mathbf {x} ^ {k + 1}) \geq \frac {1}{2} \sum_ {n = 0} ^ {k} \min \left\{S (\mathbf {x} ^ {n}), \frac {S ^ {2} (\mathbf {x} ^ {n})}{L _ {f} \Omega^ {2}} \right\}.\tag{13.13}
$$

Using the facts that $F ( \mathbf { x } ^ { k + 1 } ) \geq F _ { \mathrm { o p t } }$ and 

$$
\sum_ {n = 0} ^ {k} \min \left\{S (\mathbf {x} ^ {n}), \frac {S ^ {2} (\mathbf {x} ^ {n})}{L _ {f} \Omega^ {2}} \right\} \geq (k + 1) \min _ {n = 0, 1, \dots , k} \left[ \min \left\{S (\mathbf {x} ^ {n}), \frac {S ^ {2} (\mathbf {x} ^ {n})}{L _ {f} \Omega^ {2}} \right\} \right],
$$

we obtain that 

$$
\min _ {n = 0, 1, \ldots , k} \left[ \min \left\{S (\mathbf {x} ^ {n}), \frac {S ^ {2} (\mathbf {x} ^ {n})}{L _ {f} \Omega^ {2}} \right\} \right] \leq \frac {2 (F (\mathbf {x} ^ {0}) - F _ {\mathrm{opt}})}{k + 1},
$$

which implies in particular that there exists an $n \in \{ 0 , 1 , \ldots , k \}$ for which 

$$
\min \left\{S (\mathbf {x} ^ {n}), \frac {S ^ {2} (\mathbf {x} ^ {n})}{L _ {f} \Omega^ {2}} \right\} \leq \frac {2 (F (\mathbf {x} ^ {0}) - F _ {\mathrm{opt}})}{k + 1},
$$

that is, 

$$
S (\mathbf {x} ^ {n}) \leq \max \left\{\frac {2 (F (\mathbf {x} ^ {0}) - F _ {\mathrm{opt}})}{k + 1}, \frac {\sqrt {2 L _ {f} \Omega^ {2} (F (\mathbf {x} ^ {0}) - F _ {\mathrm{opt}})}}{\sqrt {k + 1}} \right\}.
$$

Since there exists $n \in \{ 0 , 1 , \ldots , k \}$ for which the above inequality holds, the result (13.11) immediately follows. 

(d) Suppose that x¯ is a limit point of $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ . Then there exists a subsequence $\{ \mathbf { x } ^ { k _ { j } } \} _ { j \geq 0 }$ that converges to x¯. By the definition of the conditional gradient norm $S ( \cdot )$ , it follows that for any $\mathbf { v } \in \mathbb { E }$ , 

$$
S \left(\mathbf {x} ^ {k _ {j}}\right) \geq \left\langle \nabla f \left(\mathbf {x} ^ {k _ {j}}\right), \mathbf {x} ^ {k _ {j}} - \mathbf {v} \right\rangle + g \left(\mathbf {x} ^ {k _ {j}}\right) - g (\mathbf {v}).
$$

Passing to the limit $j \to \infty$ and using the fact that $S ( \mathbf { x } ^ { k _ { j } } )  0$ as $j \to \infty$ , as well as the continuity of $\nabla f$ and the lower semicontinuity of $g .$ , we obtain that 

$$
0 \geq \langle \nabla f (\bar {\mathbf {x}}), \bar {\mathbf {x}} - \mathbf {v} \rangle + g (\bar {\mathbf {x}}) - g (\mathbf {v}) \text {for any} \mathbf {v} \in \mathbb {E},
$$

which is the same as the relation $- \nabla f ( { \bar { \mathbf { x } } } ) \in \partial g ( { \bar { \mathbf { x } } } )$ , that is, the same as stationarity. 

Example 13.10 (optimization over the unit ball). Consider the problem 

$$
\min \{f (\mathbf {x}): \| \mathbf {x} \| \leq 1 \},\tag{13.14}
$$

where $f : \mathbb { E } \to \mathbb { R }$ is $L _ { f } { \mathrm { - s m o o t h } }$ . Problem (13.14) fits the general model (13.2) with $g = \delta _ { B _ { \parallel } . \parallel } [ \mathbf { 0 } , 1 ]$ . Obviously, in this case the generalized conditional gradient method amounts to the conditional gradient method with feasible set $C = B _ { \parallel \cdot \parallel } [ 0 , 1 ]$ . Take $\mathbf { x } \in B _ { \| \cdot \| } [ \mathbf { 0 } , 1 ]$ . In order to find an expression for the conditional gradient norm $S ( \mathbf { x } )$ , we first note that 

$$
\mathbf {p} (\mathbf {x}) \in \operatorname{argmin} _ {\mathbf {p}: \| \mathbf {p} \| \leq 1} \langle \mathbf {p}, \nabla f (\mathbf {x}) \rangle
$$

is given by $\begin{array} { r } { \mathbf { p } ( \mathbf { x } ) ~ = ~ - { \frac { \nabla f ( \mathbf { x } ) } { \left\| \nabla f ( \mathbf { x } ) \right\| } } ~ \mathrm { i f } ~ \nabla f ( \mathbf { x } ) ~ \neq ~ 0 } \end{array}$ and can be chosen as $\mathbf { p } ( \mathbf { x } ) = \mathbf { 0 }$ if $\nabla f ( \mathbf { x } ) = \mathbf { 0 }$ . Thus, in both cases, we obtain that for any $\mathbf { x } \in B _ { \Vert \cdot \Vert } [ \mathbf { 0 } , 1 ]$ ， 

$$
S (\mathbf {x}) = \langle \nabla f (\mathbf {x}), \mathbf {x} - \mathbf {p} (\mathbf {x}) \rangle = \langle \nabla f (\mathbf {x}), \mathbf {x} \rangle + \| \nabla f (\mathbf {x}) \|.\tag{13.15}
$$

By its definition, $S ( \mathbf { x } ) = \infty$ for any $\mathbf { x } \notin B _ { \| \cdot \| } [ \mathbf { 0 } , 1 ]$ . By Theorem 13.6 the above expression (13.15) is nonnegative and is equal to zero if and only if x is a stationary point of (13.14), which in this case means that either $\nabla f ( \mathbf { x } ) = \mathbf { 0 } { \mathrm { ~ o r ~ } } \nabla f ( \mathbf { x } ) = \lambda \mathbf { x }$ for some $\lambda \leq 0$ (see [10, Example 9.6]). 

Assuming that $S ( \mathbf { x } ^ { k } ) \neq 0 ,$ the general update formula of the conditional gradient method for solving (13.14) is 

$$
\mathbf {x} ^ {k + 1} = (1 - t _ {k}) \mathbf {x} ^ {k} - t _ {k} \frac {\nabla f (\mathbf {x} ^ {k})}{\| \nabla f (\mathbf {x} ^ {k}) \|},
$$

where $t _ { k } \in [ 0 , 1 ]$ is an appropriately chosen stepsize. By Theorem 13.9 if the stepsize is chosen by either an adaptive or exact line search strategies, convergence of $S ( \mathbf { x } ^ { k } )$ to zero is guaranteed. 

Example 13.11 (the power method).<sup>71</sup> Continuing Example 13.10, let us consider the problem 

$$
\max _ {\mathbf {x} \in \mathbb {R} ^ {n}} \left\{\frac {1}{2} \mathbf {x} ^ {T} \mathbf {A x}: \| \mathbf {x} \| _ {2} \leq 1 \right\},\tag{13.16}
$$

where $\mathbf { A } \in \mathbb { S } _ { + } ^ { n }$ . Problem (13.16) fits the model (13.14) with $f : \mathbb { R } ^ { n }  \mathbb { R }$ given by $\begin{array} { r } { f ( \mathbf { x } ) = - \frac { 1 } { 2 } \mathbf { \dot { x } } ^ { T } \mathbf { A } \mathbf { x } } \end{array}$ . Consider the conditional gradient method for solving (13.16) and assume that $\mathbf { x } ^ { k }$ is not a stationary point of problem (13.2). Then 

$$
\mathbf {x} ^ {k + 1} = (1 - t _ {k}) \mathbf {x} ^ {k} + t _ {k} \underbrace {\frac {\mathbf {A x} ^ {k}}{\| \mathbf {A x} ^ {k} \| _ {2}}} _ {\mathbf {p} ^ {k}}.\tag{13.17}
$$

If the stepsizes are chosen by an exact line search strategy, then 

$$
t _ {k} \in \operatorname * {a r g m i n} _ {t \in [ 0, 1 ]} f (\mathbf {x} ^ {k} + t (\mathbf {p} ^ {k} - \mathbf {x} ^ {k})).\tag{13.18}
$$

Since $f$ is concave, it follows that either 0 or 1 is an optimal solution of (13.18), and by the fact that $\mathbf { x } ^ { k }$ is not a stationary point of problem (13.2), we can conclude by Theorem $1 3 . 9 ( \mathrm { a } )$ that $t _ { k } \neq 0$ . We can thus choose $t _ { k } = 1$ , and the method (13.17) becomes 

$$
\mathbf {x} ^ {k + 1} = \frac {\mathbf {A x} ^ {k}}{\| \mathbf {A x} ^ {k} \| _ {2}},
$$

which is the well-known power method for finding the eigenvector of A corresponding to the maximal eigenvalue. Theorem 13.9 guarantees that limit points of the method are stationary points of problem (13.16), meaning eigenvectors A corresponding to nonnegative eigenvalues. 

(13.3)] 

## 13.2.4 Convergence Analysis in the Convex Case

We will now further assume that f is convex. In this case, obviously all stationary points of problem (13.2) are also optimal points (Theorem $3 . 7 2 ( \mathrm { b } ) )$ , so that Theorem 13.9 guarantees that all limit points of the sequence generated by the generalized conditional gradient method with either adaptive or exact line search stepsize strategies are optimal points. We also showed in Theorem 13.9 an $O ( 1 / \sqrt { k } )$ rate of convergence of the conditional gradient norm. Our objectives will be to show an $O ( 1 / k )$ rate of convergence of function values to the optimal value, as well as of the conditional gradient norm to zero. 

We begin by showing that when f is convex, the conditional gradient norm is lower bounded by the distance to optimality in terms of function values. 

Lemma 13.12. Suppose that Assumption 13.1 holds and that $f$ is convex. Then for any $\mathbf { x } \in \operatorname { d o m } ( g )$ , 

$$
S (\mathbf {x}) \geq F (\mathbf {x}) - F _ {\mathrm{opt}}.
$$

Proof. Let $\mathbf { x } ^ { * } \in X ^ { * }$ . Then for any $\mathbf { x } \in \operatorname { d o m } ( g )$ 

$$
\begin{array}{l l} S (\mathbf {x}) = \langle \nabla f (\mathbf {x}), \mathbf {x} - \mathbf {p} (\mathbf {x}) \rangle + g (\mathbf {x}) - g (\mathbf {p} (\mathbf {x})) & [ \text {definition of} S ] \\ = \langle \nabla f (\mathbf {x}), \mathbf {x} \rangle + g (\mathbf {x}) - (\langle \nabla f (\mathbf {x}), \mathbf {p} (\mathbf {x}) \rangle + g (\mathbf {p} (\mathbf {x}))) \\ \geq \langle \nabla f (\mathbf {x}), \mathbf {x} \rangle + g (\mathbf {x}) - (\langle \nabla f (\mathbf {x}), \mathbf {x} ^ {*} \rangle + g (\mathbf {x} ^ {*})) & [ \text {definition of} \mathbf {p} (\cdot) (f) ] \\ = \langle \nabla f (\mathbf {x}), \mathbf {x} - \mathbf {x} ^ {*} \rangle + g (\mathbf {x}) - g (\mathbf {x} ^ {*}) \\ \geq f (\mathbf {x}) - f (\mathbf {x} ^ {*}) + g (\mathbf {x}) - g (\mathbf {x} ^ {*}) & [ \text {convexity of} f ] \\ = F (\mathbf {x}) - F _ {\text {opt}}. \quad \square \end{array}
$$

The convergence analysis relies on the following technical lemma on sequences of scalars. 

Lemma 13.13.<sup>72</sup> Let p be a positive integer, and let $\{ a _ { k } \} _ { k \ge 0 }$ and $\{ b _ { k } \} _ { k \ge 0 }$ be nonnegative sequences satisfying for any $k \geq 0$ 

$$
a _ {k + 1} \leq a _ {k} - \gamma_ {k} b _ {k} + \frac {A}{2} \gamma_ {k} ^ {2},\tag{13.19}
$$

where $\begin{array} { r } { \gamma _ { k } = \frac { 2 } { k + 2 p } } \end{array}$ and A is a positive number. Suppose that $a _ { k } \le b _ { k }$ for all k. Then 

(a) $\begin{array} { r } { a _ { k } \leq \frac { 2 \operatorname* { m a x } \{ A , ( p - 1 ) a _ { 0 } \} } { k + 2 p - 2 } } \end{array}$ for any $k \geq 1$ 

(b) for any $k \geq 3$ 

$$
\min _ {n = \lfloor k / 2 \rfloor + 2, \dots , k} b _ {n} \leq \frac {8 \max \{A , (p - 1) a _ {0} \}}{k - 2}.
$$

Proof. (a) By (13.19) and the fact that $a _ { k } \le b _ { k }$ , it follows that 

$$
a _ {k + 1} \leq (1 - \gamma_ {k}) a _ {k} + \frac {A}{2} \gamma_ {k} ^ {2}.
$$

Therefore, 

$$
\begin{array}{l} a _ {1} \leq (1 - \gamma_ {0}) a _ {0} + \frac {A}{2} \gamma_ {0} ^ {2}, \\ a _ {2} \leq (1 - \gamma_ {1}) a _ {1} + \frac {A}{2} \gamma_ {1} ^ {2} = (1 - \gamma_ {1}) (1 - \gamma_ {0}) a _ {0} + \frac {A}{2} (1 - \gamma_ {1}) \gamma_ {0} ^ {2} + \frac {A}{2} \gamma_ {1} ^ {2}, \\ a _ {3} \leq (1 - \gamma_ {2}) a _ {2} + \frac {A}{2} \gamma_ {2} ^ {2} = (1 - \gamma_ {2}) (1 - \gamma_ {1}) (1 - \gamma_ {0}) a _ {0} \\ \quad + \frac {A}{2} \left[ (1 - \gamma_ {2}) (1 - \gamma_ {1}) \gamma_ {0} ^ {2} + (1 - \gamma_ {2}) \gamma_ {1} ^ {2} + \gamma_ {2} ^ {2} \right]. \end{array}
$$

In general,<sup>73</sup> 

$$
a _ {k} \leq a _ {0} \prod_ {s = 0} ^ {k - 1} (1 - \gamma_ {s}) + \frac {A}{2} \sum_ {u = 0} ^ {k - 1} \left[ \prod_ {s = u + 1} ^ {k - 1} (1 - \gamma_ {s}) \right] \gamma_ {u} ^ {2}.\tag{13.20}
$$

Since $\begin{array} { r } { \gamma _ { k } = \frac { 2 } { k + 2 p } } \end{array}$ , it follows that 

$$
\begin{array}{l} \frac {A}{2} \sum_ {u = 0} ^ {k - 1} \left[ \prod_ {s = u + 1} ^ {k - 1} (1 - \gamma_ {s}) \gamma_ {u} ^ {2} \right] = \frac {A}{2} \sum_ {u = 0} ^ {k - 1} \left[ \prod_ {s = u + 1} ^ {k - 1} \frac {s + 2 p - 2}{s + 2 p} \gamma_ {u} ^ {2} \right] \\ = \frac {A}{2} \sum_ {u = 0} ^ {k - 1} \frac {(u + 2 p - 1) (u + 2 p)}{(k + 2 p - 2) (k + 2 p - 1)} \cdot \frac {4}{(u + 2 p) ^ {2}} \\ = \frac {A}{2} \sum_ {u = 0} ^ {k - 1} \frac {u + 2 p - 1}{(k + 2 p - 2) (k + 2 p - 1)} \cdot \frac {4}{u + 2 p} \\ \leq \frac {2 A k}{(k + 2 p - 2) (k + 2 p - 1)}. \end{array}\tag{13.21}
$$

In addition, 

$$
a _ {0} \prod_ {s = 0} ^ {k - 1} (1 - \gamma_ {s}) = a _ {0} \prod_ {s = 0} ^ {k - 1} \frac {s + 2 p - 2}{s + 2 p} = a _ {0} \frac {(2 p - 2) (2 p - 1)}{(k + 2 p - 2) (k + 2 p - 1)}.\tag{13.22}
$$

Therefore, combining (13.20), (13.21), and (13.22), 

$$
\begin{array}{l} a _ {k} \leq \frac {2 A k}{(k + 2 p - 2) (k + 2 p - 1)} + \frac {a _ {0} (2 p - 2) (2 p - 1)}{(k + 2 p - 2) (k + 2 p - 1)} \\ \leq \frac {2 \max \{A , (p - 1) a _ {0} \} (k + 2 p - 1)}{(k + 2 p - 2) (k + 2 p - 1)} \\ = \frac {2 \max \{A , (p - 1) a _ {0} \}}{k + 2 p - 2}. \end{array}
$$

(b) Replacing the index k with n in (13.19), we have 

$$
a _ {n + 1} \leq a _ {n} - \gamma_ {n} b _ {n} + \frac {A}{2} \gamma_ {n} ^ {2}.
$$

Summing the above inequality over $n = j , j + 1 , \dotsc , k$ , we obtain that 

$$
a _ {k + 1} \leq a _ {j} - \sum_ {n = j} ^ {k} \gamma_ {n} b _ {n} + \frac {A}{2} \sum_ {n = j} ^ {k} \gamma_ {n} ^ {2}.
$$

Thus, using the result of part (a) (assuming that $j \geq 1 )$ , 

$$
\begin{array}{l} \left(\sum_ {n = j} ^ {k} \gamma_ {n}\right) \min _ {n = j, \dots , k} b _ {n} \leq a _ {j} + \frac {A}{2} \sum_ {n = j} ^ {k} \gamma_ {n} ^ {2} \\ \leq \frac {2 \max \{A , (p - 1) a _ {0} \}}{j + 2 p - 2} + 2 A \sum_ {n = j} ^ {k} \frac {1}{(n + 2 p) ^ {2}} \\ \leq \frac {2 \max \{A , (p - 1) a _ {0} \}}{j + 2 p - 2} + 2 A \sum_ {n = j} ^ {k} \frac {1}{(n + 2 p - 1) (n + 2 p)} \\ = \frac {2 \max \{A , (p - 1) a _ {0} \}}{j + 2 p - 2} + 2 A \sum_ {n = j} ^ {k} \left[ \frac {1}{n + 2 p - 1} - \frac {1}{n + 2 p} \right] \\ = \frac {2 \max \{A , (p - 1) a _ {0} \}}{j + 2 p - 2} + 2 A \left[ \frac {1}{j + 2 p - 1} - \frac {1}{k + 2 p} \right] \\ \leq \frac {4 \max \{A , (p - 1) a _ {0} \}}{j + 2 p - 2}. \end{array} \tag {13.}\tag{13.23}
$$

On the other hand, 

$$
\sum_ {n = j} ^ {k} \gamma_ {n} = 2 \sum_ {n = j} ^ {k} \frac {1}{n + 2 p} \geq 2 \frac {k - j + 1}{k + 2 p},
$$

which, combined with (13.23), yields 

$$
\min _ {n = j, \dots , k} b _ {n} \leq \frac {2 \max \{A , (p - 1) a _ {0} \} (k + 2 p)}{(j + 2 p - 2) (k - j + 1)}.
$$

Taking $j = \lfloor k / 2 \rfloor + 2$ , we conclude that for any $k \geq 3$ 2 

$$
\min _ {n = \lfloor k / 2 \rfloor + 2, \dots , k} b _ {n} \leq \frac {2 \max \{A , (p - 1) a _ {0} \} (k + 2 p)}{(\lfloor k / 2 \rfloor + 2 p) (k - \lfloor k / 2 \rfloor - 1)}.\tag{13.24}
$$

Now, 

$$
\begin{array}{l} \frac {k + 2 p}{(\lfloor k / 2 \rfloor + 2 p) (k - \lfloor k / 2 \rfloor - 1)} \leq \frac {k + 2 p}{(k / 2 + 2 p - 0 . 5) (k - \lfloor k / 2 \rfloor - 1)} \\ = 2 \frac {k + 2 p}{k + 4 p - 1} \cdot \frac {1}{k - \lfloor k / 2 \rfloor - 1} \\ \leq \frac {2}{k - \lfloor k / 2 \rfloor - 1} \\ \leq \frac {2}{k / 2 - 1}, \end{array}
$$

which, combined with (13.24), yields 

$$
\min _ {n = \lfloor k / 2 \rfloor + 2, \dots , k} b _ {n} \leq \frac {8 \max \{A , (p - 1) a _ {0} \}}{k - 2}. \quad \square
$$

Equipped with Lemma 13.13, we will now establish a sublinear rate of convergence of the generalized conditional gradient method under the three stepsize strategies described at the beginning of Section 13.2.3: predefined, adaptive, and exact line search. 

Theorem 13.14. Suppose that Assumption 13.1 holds and that f is convex. Let $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ be the sequence generated by the generalized conditional gradient method for solving problem (13.2) with either a predefined stepsize $\begin{array} { r } { t _ { k } = \alpha _ { k } \equiv \frac { 2 } { k + 2 } } \end{array}$ , adaptive stepsize, or exact line search. Let Ω be an upper bound on the diameter of dom(g): 

$$
\Omega \geq \max _ {\mathbf {x}, \mathbf {y} \in \operatorname{dom} (g)} \| \mathbf {x} - \mathbf {y} \|.
$$

Then 

$$
\mathrm{(a)} F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} \leq \frac {2 L _ {f} \Omega^ {2}}{k} f o r a n y k \geq 1;
$$

$$
\mathrm{(b)} \min _ {n = \lfloor k / 2 \rfloor + 2, \dots , k} S (\mathbf {x} ^ {n}) \leq \frac {8 L _ {f} \Omega^ {2}}{k - 2} f o r a n y k \geq 3.
$$

Proof. By the fundamental inequality (13.6) invoked with $\mathbf { x } = \mathbf { x } ^ { k }$ and $t = t _ { k }$ , it follows that for any $k \geq 0$ 2 

$$
F (\mathbf {x} ^ {k} + t _ {k} (\mathbf {p} ^ {k} - \mathbf {x} ^ {k})) - F _ {\mathrm{opt}} \leq F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} - t _ {k} S (\mathbf {x} ^ {k}) + \frac {t _ {k} ^ {2} L _ {f}}{2} \| \mathbf {p} ^ {k} - \mathbf {x} ^ {k} \| ^ {2},\tag{13.25}
$$

where $\mathbf { p } ^ { k } \mathbf { \Psi } = \mathbf { \Psi } \mathbf { p } ( \mathbf { x } ^ { k } )$ . Specifically, if a predefined stepsize is used, meaning that $\begin{array} { r } { t _ { k } = \alpha _ { k } \equiv \frac { 2 } { k + 2 } } \end{array}$ , then 

$$
F (\mathbf {x} ^ {k} + \alpha_ {k} (\mathbf {p} ^ {k} - \mathbf {x} ^ {k})) - F _ {\mathrm{opt}} \leq F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} - \alpha_ {k} S (\mathbf {x} ^ {k}) + \frac {\alpha_ {k} ^ {2} L _ {f}}{2} \| \mathbf {p} ^ {k} - \mathbf {x} ^ {k} \| ^ {2}.\tag{13.26}
$$

If an exact line search is used, meaning that $t _ { k } = u _ { k } \in \mathrm { a r g m i n } _ { t \in [ 0 , 1 ] } F ( \mathbf { x } ^ { k } + t ( \mathbf { p } ^ { k } -$ $\mathbf { x } ^ { k } ) )$ , then 

$$
F (\mathbf {x} ^ {k} + u _ {k} (\mathbf {p} ^ {k} - \mathbf {x} ^ {k})) - F _ {\mathrm{opt}} \leq F (\mathbf {x} ^ {k} + \alpha_ {k} (\mathbf {p} ^ {k} - \mathbf {x} ^ {k})) - F _ {\mathrm{opt}}\tag{13.27}
$$

$$
\leq F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} - \alpha_ {k} S (\mathbf {x} ^ {k}) + \frac {\alpha_ {k} ^ {2} L _ {f}}{2} \| \mathbf {p} ^ {k} - \mathbf {x} ^ {k} \| ^ {2},
$$

where the first inequality follows by the definition of $u _ { k }$ and the second is the inequality (13.26). Finally, in the adaptive stepsize strategy, $\begin{array} { r } { t _ { k } = v _ { k } \equiv \operatorname* { m i n } \bigg \{ 1 , \frac { S ( \mathbf { x } ^ { k } ) } { L _ { f } \| \mathbf { p } ^ { k } - \mathbf { x } ^ { k } \| ^ { 2 } } \bigg \} } \end{array}$ Note that $v _ { k }$ satisfies 

$$
v _ {k} = \operatorname{argmin} _ {t \in [ 0, 1 ]} \left\{- t S (\mathbf {x} ^ {k}) + \frac {t ^ {2} L _ {f}}{2} \| \mathbf {p} ^ {k} - \mathbf {x} ^ {k} \| ^ {2} \right\}.\tag{13.28}
$$

Thus, 

$$
\begin{array}{r l r} & & {F (\mathbf {x} ^ {k} + v _ {k} (\mathbf {p} ^ {k} - \mathbf {x} ^ {k})) - F _ {\mathrm{opt}} \leq F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} - v _ {k} S (\mathbf {x} ^ {k}) + \frac {v _ {k} ^ {2} L _ {f}}{2} \| \mathbf {p} ^ {k} - \mathbf {x} ^ {k} \| ^ {2}} \\ & & {\leq F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} - \alpha_ {k} S (\mathbf {x} ^ {k}) + \frac {\alpha_ {k} ^ {2} L _ {f}}{2} \| \mathbf {p} ^ {k} - \mathbf {x} ^ {k} \| ^ {2},} \end{array}
$$

where the first inequality is the inequality (13.25) with $t _ { k } = v _ { k }$ and the second is due to (13.28). Combining the last inequality with (13.26) and (13.27), we conclude that for the three stepsize strategies, the following inequality holds: 

$$
F (\mathbf {x} ^ {k + 1}) - F _ {\mathrm{opt}} \leq F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} - \alpha_ {k} S (\mathbf {x} ^ {k}) + \frac {\alpha_ {k} ^ {2} L _ {f}}{2} \| \mathbf {p} ^ {k} - \mathbf {x} ^ {k} \| ^ {2},
$$

which, combined with the inequality $\| \mathbf { p } ^ { k } - \mathbf { x } ^ { k } \| \leq \Omega$ , implies that 

$$
F (\mathbf {x} ^ {k + 1}) - F _ {\mathrm{opt}} \leq F (\mathbf {x} ^ {k}) - F _ {\mathrm{opt}} - \alpha_ {k} S (\mathbf {x} ^ {k}) + \frac {\alpha_ {k} ^ {2} L _ {f} \Omega^ {2}}{2}.
$$

Invoking Lemma 13.13 with $a _ { k } = F ( \mathbf { x } ^ { k } ) - F _ { \mathrm { o p t } } , b _ { k } = S ( \mathbf { x } ^ { k } ) , A = L _ { f } \Omega ^ { 2 }$ , and $p = 1$ and noting that $a _ { k } \le b _ { k }$ by Lemma 13.12, both parts (a) and (b) follow. 

## 13.3 The Strongly Convex Case

We will focus on the case where the nonsmooth part is an indicator of a compact and convex set $C ,$ meaning that $g = \delta _ { C }$ , so that problem (13.2) becomes 

$$
\min \{f (\mathbf {x}): \mathbf {x} \in C \},
$$

and the method under consideration is the conditional gradient method. In Section 10.6 we showed that the proximal gradient method enjoys an improved linear convergence when the smooth part (in the composite model) is strongly convex. Unfortunately, as we will see in Section 13.3.1, in general, the conditional gradient method does not converge in a linear rate even if an additional strong convexity assumption is made on the objective function. Later on, in Section 13.3.2 we will show how, under a strong convexity assumption on the feasible set (and not on the objective function), linear rate can be established. 

## 13.3.1 The Negative Result of Canon and Cullum

The arguments go back to Canon and Cullum [37], and we follow them. We begin with some technical lemmas. 

Lemma 13.15. Let $\{ a _ { n } \} _ { n \geq 0 }$ be a sequence of real numbers such that $\textstyle \sum _ { n = 0 } ^ { \infty } \left| a _ { n } \right|$ diverges. Then for every $\begin{array} { r } { \varepsilon > 0 , \sum _ { n = k } ^ { \infty } \bar { a } _ { n } ^ { 2 } \ge \frac { 1 } { k ^ { 1 + \varepsilon } } } \end{array}$ for infinitely many $k ' s$ 

Proof. Suppose by contradiction that there is $\varepsilon > 0$ and a positive integer K such that for all $k \geq K$ 

$$
\sum_ {n = k} ^ {\infty} a _ {n} ^ {2} <   \frac {1}{k ^ {1 + 2 \varepsilon}}.\tag{13.29}
$$

We will show that $\textstyle \sum _ { n = 1 } ^ { \infty } \left| a _ { n } \right|$ converges. Note that by the Cauchy–Schwarz inequality, 

$$
\sum_ {n = 1} ^ {\infty} \left| a _ {n} \right| = \sum_ {n = 1} ^ {\infty} \left| a _ {n} \right| n ^ {(1 + \varepsilon) / 2} n ^ {- (1 + \varepsilon) / 2} \leq \sqrt {\sum_ {n = 1} ^ {\infty} n ^ {1 + \varepsilon} a _ {n} ^ {2}} \sqrt {\sum_ {n = 1} ^ {\infty} n ^ {- (1 + \varepsilon)}}.\tag{13.30}
$$

Since $\scriptstyle \sum _ { n = 1 } ^ { \infty } n ^ { - ( 1 + \varepsilon ) }$ converges, it is enough to show that $\textstyle \sum _ { n = 1 } ^ { \infty } n ^ { 1 + \varepsilon } a _ { n } ^ { 2 }$ converges. For that, note that by (13.29), for any $m \geq K$ 2 

$$
\sum_ {k = K} ^ {m} \left[ k ^ {\varepsilon} \sum_ {n = k} ^ {m} a _ {n} ^ {2} \right] \leq \sum_ {k = K} ^ {m} \left[ k ^ {\varepsilon} \sum_ {n = k} ^ {\infty} a _ {n} ^ {2} \right] \leq \sum_ {k = K} ^ {m} \frac {1}{k ^ {1 + \varepsilon}}.\tag{13.31}
$$

On the other hand, 

$$
\sum_ {k = K} ^ {m} \left[ k ^ {\varepsilon} \sum_ {n = k} ^ {m} a _ {n} ^ {2} \right] = \sum_ {n = K} ^ {m} \left[ a _ {n} ^ {2} \sum_ {k = K} ^ {n} k ^ {\varepsilon} \right],
$$

which, combined with the inequality 

$$
\sum_ {k = K} ^ {n} k ^ {\varepsilon} \geq \int_ {K} ^ {n} x ^ {\varepsilon} d x = \frac {1}{1 + \varepsilon} (n ^ {1 + \varepsilon} - K ^ {1 + \varepsilon})
$$

and (13.31), implies that (taking $m  \infty ,$ ) 

$$
\frac {1}{1 + \varepsilon} \sum_ {n = K} ^ {\infty} (n ^ {1 + \varepsilon} - K ^ {1 + \varepsilon}) a _ {n} ^ {2} \leq \sum_ {k = K} ^ {\infty} \frac {1}{k ^ {1 + \varepsilon}}.
$$

Since both $\scriptstyle \sum _ { k = K } ^ { \infty } { \frac { 1 } { k ^ { 1 + \varepsilon } } }$ and $\textstyle \sum _ { n = K } ^ { \infty } a _ { n } ^ { 2 }$ converge, it follows that $\scriptstyle \sum _ { n = K } ^ { \infty } n ^ { 1 + \varepsilon } a _ { n } ^ { 2 }$ con- verges and hence, by (13.30), that $\textstyle \sum _ { n = 1 } ^ { \infty } \left| a _ { n } \right|$ converges, which is a contradiction to our underlying assumptions. 

We will also use the following well-known lemma. 

Lemma 13.16 (see [75, Chapter VII, Theorem 4]). Let $\{ b _ { n } \} _ { n \geq 0 }$ be a sequence satisfying $0 \leq b _ { n } < 1$ for any n. Then $\textstyle \prod _ { n = 0 } ^ { m } ( 1 - b _ { n } ) \to 0$ as $m \to \infty$ if and only if $\textstyle \sum _ { n = 0 } ^ { \infty } b _ { n }$ diverges. 

Our main goal will be to describe an example of a minimization problem of a strongly convex function over a nonempty compact convex set for which the conditional gradient method does not exhibit a linear rate of convergence. For that, let us consider the following quadratic problem over $\mathbb { R } ^ { n }$ : 

$$
f _ {\text { opt }} \equiv \min _ {\mathbf {x} \in \mathbb {R} ^ {n}} \left\{f _ {q} (\mathbf {x}) \equiv \frac {1}{2} \mathbf {x} ^ {T} \mathbf {Q} \mathbf {x} + \mathbf {b} ^ {T} \mathbf {x}: \mathbf {x} \in \Omega \right\},\tag{13.32}
$$

where $\mathbf { Q } \in \mathbb { S } _ { + + } ^ { n } , \mathbf { b } \in \mathbb { R } ^ { n }$ , and $\Omega = \mathrm { c o n v } \{ \mathbf { a } _ { 1 } , \mathbf { a } _ { 2 } , \dots , \mathbf { a } _ { l } \}$ , where $\mathbf { a } _ { 1 } , \mathbf { a } _ { 2 } , \dots , \mathbf { a } _ { l } \in \mathbb { R } ^ { n }$ We will make the following assumption on problem (13.32). 

Assumption 13.17. int $( \Omega ) \ \ne \ \emptyset$ and the optimal solution of problem (13.32), denoted by $\mathbf { x } ^ { * }$ , is on the boundary of Ω and is not an extreme point of Ω. 

Denoting $\mathbf { A } \in \mathbb { R } ^ { n \times l }$ as the matrix whose columns are $\mathbf { a } _ { 1 } , \ldots , \mathbf { a } _ { l }$ , we can also write problem (13.32) as 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {n}, \mathbf {v} \in \mathbb {R} ^ {l}} \left\{\frac {1}{2} \mathbf {x} ^ {T} \mathbf {Q} \mathbf {x} + \mathbf {b} ^ {T} \mathbf {x}: \mathbf {x} = \mathbf {A} \mathbf {v}, \mathbf {v} \in \Delta_ {l} \right\}.
$$

The conditional gradient method with exact line search strategy for solving (13.32) reads as follows. Given the kth iterate $\mathbf { x } ^ { k }$ , the next point $\mathbf { x } ^ { k + 1 }$ is computed as follows: 

• Choose 

$$
i _ {k} \in \operatorname{argmin} _ {i = 1, 2, \dots , l} \langle \mathbf {a} _ {i}, \nabla f _ {q} (\mathbf {x} ^ {k}) \rangle .
$$

• Define 

$$
\mathbf {d} ^ {k} = \mathbf {a} _ {i _ {k}} - \mathbf {x} ^ {k}.\tag{13.33}
$$

If $\langle \mathbf { d } ^ { k } , \nabla f _ { q } ( \mathbf { x } ^ { k } ) \rangle \geq 0$ , then $\mathbf { x } ^ { k }$ is the optimal solution of problem (13.32). Otherwise, set 

$$
\mathbf {x} ^ {k + 1} = \mathbf {x} ^ {k} + t _ {k} \mathbf {d} ^ {k},
$$

where 

$$
t _ {k} = \mathrm{argmin} _ {t \in [ 0, 1 ]} f _ {q} (\mathbf {x} ^ {k} + t \mathbf {d} ^ {k}) = \min \left\{\lambda_ {k}, 1 \right\},
$$

with $\lambda _ { k }$ defined as 

$$
\lambda_ {k} = - \frac {\left\langle \mathbf {d} ^ {k} , \nabla f _ {q} (\mathbf {x} ^ {k}) \right\rangle}{(\mathbf {d} ^ {k}) ^ {T} \mathbf {Q d} ^ {k}}.\tag{13.34}
$$

We will make the following assumption on the starting point of the conditional gradient method. 

Assumption 13.18. $f _ { q } ( { \bf x } ^ { 0 } ) < \mathrm { m i n } _ { i = 1 , 2 , . . . , l } f _ { q } ( { \bf a } _ { i } )$ and $\mathbf { x } ^ { 0 } = \mathbf { A } \mathbf { v } ^ { 0 } \in \Omega$ , where $\mathbf { v } ^ { 0 } \in \Delta _ { l } \cap \mathbb { R } _ { + + } ^ { n }$ . In particular, $\mathbf { x } ^ { 0 } \in \mathrm { i n t } ( \Omega )$ 

A vector $\mathbf { x } ^ { 0 }$ satisfying Assumption 13.18 can be easily obtained by the following procedure. 

• Pick $p \in \mathrm { a r g m i n } _ { i = 1 , 2 , . . . , l } f _ { q } ( \mathbf { a } _ { i } )$ 

• Employ one step of the conditional gradient method starting from $\mathbf { a } _ { p }$ and obtain a point $\tilde { \mathbf { x } } ^ { 0 } \in \Omega$ for which $f _ { q } ( \tilde { \mathbf { x } } ^ { 0 } ) < f _ { q } ( \mathbf { a } ^ { p } )$ (the latter is satisfied since $\mathbf { a } ^ { p }$ is not an optimal solution—see Theorem 13.9(a)). 

• Find $\tilde { \mathbf { v } } ^ { 0 } \in \Delta _ { l }$ for which $\tilde { \mathbf { x } } ^ { 0 } = \mathbf { A } \tilde { \mathbf { v } } ^ { 0 }$ 

• If $\tilde { \mathbf { v } } ^ { 0 } \in \mathbb { R } _ { + + } ^ { l } .$ define $\mathbf { v } ^ { 0 } = \tilde { \mathbf { v } } ^ { 0 }$ and $\mathbf { x } ^ { 0 } = \tilde { \mathbf { x } } ^ { 0 }$ . If $\tilde { \mathbf { v } } ^ { 0 } \notin \mathbb { R } _ { + + } ^ { l }$ , then take a point $\mathbf { v } ^ { 0 } \in \dot { \Delta } _ { l } \cap \mathbb { R } _ { + + } ^ { l }$ close enough to $\tilde { \mathbf { v } } ^ { 0 }$ such that $\mathbf { x } ^ { 0 } \equiv \mathbf { \dot { A } } \mathbf { \dot { v } } ^ { 0 }$ will satisfy $f _ { q } ( \mathbf { x } ^ { 0 } ) < f _ { q } ( \mathbf { a } ^ { p } )$ 

The following lemma gathers several technical results that will be key to establishing the slow rate of the conditional gradient method. 

Lemma 13.19. Suppose that Assumption 13.17 holds and that $\{ \mathbf { x } ^ { k } \}$ is the sequence generated by the conditional gradient method with exact line search employed on problem (13.32) with a starting point $\mathbf { x } ^ { 0 }$ satisfying Assumption 13.18. Let $\mathbf { d } ^ { k }$ and $\lambda _ { k }$ be given by (13.33) and (13.34), respectively. Then 

(a) $\mathbf { x } ^ { k } \in \operatorname { i n t } ( \Omega )$ and $t _ { k } = \lambda _ { k } < 1$ for any $k \geq 0 ,$ 

(b) $\begin{array} { r } { f _ { q } ( \mathbf { x } ^ { k + 1 } ) = f _ { q } ( \mathbf { x } ^ { k } ) - \frac { 1 } { 2 } ( ( \mathbf { d } ^ { k } ) ^ { T } \mathbf { Q } \mathbf { d } ^ { k } ) \lambda _ { k } ^ { 2 } ~ f o r ~ a n y ~ k \ge 0 , } \end{array}$ 

(c) $\begin{array} { r } { \sum _ { k = 0 } ^ { \infty } \lambda _ { k } = \infty ; } \end{array}$ 

(d) there exists $\beta > 0$ such that $( \mathbf { d } ^ { k } ) ^ { T } \mathbf { Q } \mathbf { d } ^ { k } \geq \beta$ for all $k \geq 0$ 

Proof. (a) The stepsizes must satisfy $t _ { k } = \lambda _ { k } < 1$ , since otherwise, if $t _ { k } = 1$ for some $k .$ , then this means that $\mathbf { x } ^ { k + 1 } = \mathbf { a } _ { i _ { k } }$ . But $f _ { q } ( \mathbf { x } ^ { k + 1 } ) = f _ { q } ( \mathbf { a } _ { i _ { k } } ) > f _ { q } ( \mathbf { x } ^ { 0 } )$ , which is a contradiction to the monotonicity of the sequence of function values generated by the conditional gradient method (Theorem $\mathrm { 1 3 . 9 ( a ) } )$ ). The proof that $\mathbf { x } ^ { k } \in \operatorname { i n t } ( \Omega )$ is by induction on $k .$ . For $k = 0$ , by Assumption 13.18, $\mathbf { x } ^ { 0 } \in \mathrm { i n t } ( \Omega )$ . Now suppose that $\mathbf { x } ^ { k } \in \operatorname { i n t } ( \Omega )$ . To prove that the same holds for $k + 1$ , note that since $t _ { k } < 1$ , it follows by the line segment principle (Lemma 5.23) that $\mathbf { x } ^ { k + 1 } = ( 1 - t _ { k } ) \mathbf { x } ^ { k } + t _ { k } \mathbf { a } _ { i _ { k } }$ is also in int(Ω). 

(b) Since $t _ { k } = \lambda _ { k }$ , it follows that 

$$
\begin{array}{l} f _ {q} (\mathbf {x} ^ {k + 1}) = f _ {q} \left(\mathbf {x} ^ {k} + \lambda_ {k} \mathbf {d} ^ {k}\right) \\ \qquad = \frac {1}{2} (\mathbf {x} ^ {k} + \lambda_ {k} \mathbf {d} ^ {k}) ^ {T} \mathbf {Q} (\mathbf {x} ^ {k} + \lambda_ {k} \mathbf {d} ^ {k}) + \mathbf {b} ^ {T} (\mathbf {x} ^ {k} + \lambda_ {k} \mathbf {d} ^ {k}) \\ \qquad = f _ {q} (\mathbf {x} ^ {k}) + \lambda_ {k} (\mathbf {d} ^ {k}) ^ {T} (\mathbf {Q x} ^ {k} + \mathbf {b}) + \frac {\lambda_ {k} ^ {2}}{2} (\mathbf {d} ^ {k}) ^ {T} \mathbf {Q d} ^ {k} \\ \qquad = f _ {q} (\mathbf {x} ^ {k}) + ((\mathbf {d} ^ {k}) ^ {T} \mathbf {Q d} ^ {k}) \left(- \lambda_ {k} ^ {2} + \frac {\lambda_ {k} ^ {2}}{2}\right) \\ \qquad = f _ {q} (\mathbf {x} ^ {k}) - \frac {1}{2} ((\mathbf {d} ^ {k}) ^ {T} \mathbf {Q d} ^ {k}) \lambda_ {k} ^ {2}. \end{array}
$$

(c) Suppose by contradiction that $\textstyle \sum _ { k = 0 } ^ { \infty } \lambda _ { k } < \infty$ , then by Lemma 13.16, it follows that $\begin{array} { r } { \prod _ { k = 0 } ^ { \infty } ( 1 - \lambda _ { k } ) = \delta } \end{array}$ for some $\delta > 0$ . Note that by the definition of the Lmethod, for any $\bar { k } \geq 0 , \mathbf { x } ^ { k } = \mathbf { A } \mathbf { v } ^ { k }$ , where $\{ \mathbf { v } ^ { k } \} _ { k \geq 0 }$ satisfies 

$$
\mathbf {v} ^ {k + 1} = (1 - \lambda_ {k}) \mathbf {v} ^ {k} + \lambda_ {k} \mathbf {e} _ {i _ {k}}.
$$

Hence, 

$$
\mathbf {v} ^ {k + 1} \geq (1 - \lambda_ {k}) \mathbf {v} ^ {k},
$$

implying that 

$$
\mathbf {v} ^ {k} \geq \delta \mathbf {v} ^ {0}.\tag{13.35}
$$

By Theorem 13.9(d), the limit points of $\{ { \bf x } ^ { k } \} _ { k \ge 0 }$ are stationary points of problem (13.32). Let $\mathbf { x } ^ { * }$ be the unique optimal solution of problem (13.32). Since $\mathbf { x } ^ { * }$ is the only stationary point of problem (13.32), we can conclude that $\mathbf { x } ^ { k } \to \mathbf { x } ^ { * }$ . The sequence $\{ \mathbf { v } ^ { k } \} _ { k \geq 0 }$ is bounded and hence has a convergent subsequence $\{ \mathbf { v } ^ { k _ { j } } \} _ { j \geq 0 }$ Denoting the limit of the subsequence by $\mathbf { v } ^ { * } \in \Delta _ { l }$ , we note that by (13.35) it follows that $\mathbf { v } ^ { * } \geq \delta \mathbf { v } ^ { 0 }$ , and hence $\mathbf { v } ^ { * } \in \bar { \Delta } _ { l } \cap \mathbb { R } _ { + + } ^ { l }$ . Taking j to ∞ in the identity $\mathbf { x } ^ { k _ { j } } = \mathbf { A } \mathbf { v } ^ { k _ { j } }$ ， we obtain that $\mathbf { x } ^ { * } = \mathbf { A } \mathbf { v } ^ { * }$ , where $\mathbf { v } ^ { * } \in \Delta _ { l } \cap \mathbb { R } _ { + + } ^ { l }$ , implying that the $\mathbf { x } ^ { * } \in$ int(Ω), in contradiction to Assumption 13.17. 

(d) Since 

$$
(\mathbf {d} ^ {k}) ^ {T} \mathbf {Q} \mathbf {d} ^ {k} \geq \gamma \| \mathbf {d} ^ {k} \| _ {2} ^ {2}\tag{13.36}
$$

with $\gamma = \lambda _ { \mathrm { m i n } } ( \mathbf { Q } ) > 0$ , it follows that we need to show that $\| \mathbf { d } ^ { k } \| _ { 2 }$ is bounded below by a positive number. Note that by Assumption $1 3 . 1 7 , \mathbf { x } ^ { * } \notin \{ \mathbf { a } _ { 1 } , \mathbf { a } _ { 2 } , \ldots , \mathbf { a } _ { l } \}$ , and therefore there exists a positive integer K and $\beta _ { 1 } > 0$ such that $\| \mathbf { a } _ { i } - \mathbf { x } ^ { k } \| \geq \beta _ { 1 }$ for all $k > K$ and $i \in \{ 1 , 2 , \ldots , l \}$ . Since $\mathbf { x } ^ { k } \in \mathrm { i n t } ( \Omega )$ for all $k ,$ it follows that for $\beta _ { 2 }$ defined as 

$$
\beta_ {2} \equiv \min \{\beta_ {1}, \| \mathbf {a} _ {i _ {0}} - \mathbf {x} ^ {0} \| _ {2}, \| \mathbf {a} _ {i _ {1}} - \mathbf {x} ^ {1} \| _ {2}, \ldots , \| \mathbf {a} _ {i _ {K}} - \mathbf {x} ^ {K} \| _ {2} \} > 0,
$$

it holds that $\| \mathbf { d } ^ { k } \| _ { 2 } = \| \mathbf { a } _ { i _ { k } } - \mathbf { x } ^ { k } \| \geq \beta _ { 2 }$ for all $k \geq 0$ , and we can finally conclude by (13.36) that for $\beta = \gamma \beta _ { 2 } ^ { 2 } , ( { \bf d } ^ { k } ) ^ { T } { \bf Q } { \bf d } ^ { k } \geq \beta$ for all $k \geq 0$ □ 

The main negative result showing that the rate of convergence of the method cannot be linear is stated in Theorem 13.20 below. 

Theorem 13.20 (Canon and Cullum’s negative result). Suppose that Assumption 13.17 holds and that $\{ \mathbf { x } ^ { k } \}$ is the sequence generated by the conditional gradient method with exact line search for solving problem (13.32) with a starting point $\mathbf { x } ^ { 0 }$ satisfying Assumption 13.18. Then for every $\varepsilon > 0$ we have that $\begin{array} { r } { f _ { q } ( \mathbf { x } ^ { k } ) - f _ { \mathrm { o p t } } \geq \frac { 1 } { k ^ { 1 + \varepsilon } } } \end{array}$ for infinitely many $k ' s .$ 

Proof. Let $\mathbf { d } ^ { k }$ and $\lambda _ { k }$ be given by (13.33) and (13.34), respectively. By Lemma 13.19(b), we have for any two positive integers satisfying $K \geq k$ 

$$
f _ {q} (\mathbf {x} ^ {K}) - f _ {\mathrm{opt}} = f _ {q} (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}} - \frac {1}{2} \sum_ {n = k} ^ {K - 1} ((\mathbf {d} ^ {n}) ^ {T} \mathbf {Q} \mathbf {d} ^ {n}) \lambda_ {n} ^ {2}.
$$

Taking $K  \infty$ and using the fact that $f _ { q } ( { \bf x } ^ { K } )  f _ { \mathrm { o p t } }$ and Lemma 13.19(d), we obtain that 

$$
f _ {q} (\mathbf {x} ^ {k}) - f _ {\mathrm{opt}} = \frac {1}{2} \sum_ {n = k} ^ {\infty} ((\mathbf {d} ^ {n}) \mathbf {Q} (\mathbf {d} ^ {n})) \lambda_ {n} ^ {2} \geq \frac {\beta}{2} \sum_ {n = k} ^ {\infty} \lambda_ {n} ^ {2}.\tag{13.37}
$$

By Lemma $1 3 . 1 9 ( \mathrm { c } ) , \sum _ { k = 0 } ^ { \infty } \lambda _ { k } = \infty$ , and hence by Lemma 13.15 and (13.37), weconclude that $\begin{array} { r } { f _ { q } ( \mathbf { x } ^ { k } ) - \ddot { f } _ { \mathrm { o p t } } \geq \frac { 1 } { k ^ { 1 + \varepsilon } } } \end{array}$ for infinitely many $k \mathrm { { s } }$ .口

Example 13.21. Consider the problem 

$$
\min \{f _ {q} (x _ {1}, x _ {2}) \equiv x _ {1} ^ {2} + x _ {2} ^ {2}: (x _ {1}, x _ {2}) \in \operatorname{conv} \{(- 1, 0), (1, 0), (0, 1) \} \}.\tag{13.38}
$$

Assumption 13.17 is satisfied since the feasible set of problem (13.38) has a nonempty interior and the optimal solution, $( x _ { 1 } ^ { * } , x _ { 2 } ^ { * } ) = ( 0 , 0 )$ , is on the boundary of the feasible set but is not an extreme point. The starting point $\mathbf { x } ^ { 0 } = ( 0 , \textstyle { \frac { 1 } { 2 } } )$ satisfies Assumption 13.18 since 

$$
f _ {q} (\mathbf {x} ^ {0}) = \frac {1}{4} <   1 = \min \{f _ {q} (- 1, 0), f _ {q} (1, 0), f _ {q} (0, 1) \}
$$

and $\begin{array} { r } { \mathbf { x } ^ { 0 } = \frac { 1 } { 4 } ( - 1 , 0 ) + \frac { 1 } { 4 } ( 1 , 0 ) + \frac { 1 } { 2 } ( 0 , 1 ) } \end{array}$ . The first 100 iterations produced by the conditional gradient method are plotted in Figure 13.1. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/35ec2296-f55f-43fc-927c-40450837b5fa/49391a1714353a1171a9eed3565ccf00b169afbb0f15ce281a94e4be5b496370.jpg)



Figure 13.1. First 100 iterations of the conditional gradient method employed on the problem from Example 13.21.


## 13.3.2 Linear Rate under Strong Convexity of the Feasible Set

Canon and Cullum’s negative result shows that diferent assumptions than strong convexity of the objective are required in order to establish a linear rate of convergence of the conditional gradient method. One example of such an assumption is strong convexity of the feasible set. 

Definition 13.22 (strongly convex set). A nonempty set $C \subseteq \mathbb { E }$ is called $\sigma \mathrm { - }$ strongly convex $( \sigma > 0 )$ if for any $\mathbf { x } , \mathbf { y } \in C$ and $\lambda \in [ 0 , 1 ]$ the inclusion 

$$
B \left[ \lambda \mathbf {x} + (1 - \lambda) \mathbf {y}, \frac {\sigma}{2} \lambda (1 - \lambda) \| \mathbf {x} - \mathbf {y} \| ^ {2} \right] \subseteq C
$$

holds. 

A set is called strongly convex if it is σ-strongly convex for some $\sigma > 0$ Obviously, any strongly convex set is also convex. The next result states that level sets of nonnegative strongly convex and smooth functions are strongly convex sets. 

Theorem 13.23 (strong convexity of level sets of strongly convex and smooth functions).<sup>74</sup> Suppose that $g : \mathbb { E } \to \mathbb { R } _ { + }$ is nonnegative, $L _ { g } – s m o o t h _ { \scriptscriptstyle 3 }$ , and $\sigma _ { g } - s t r o n g l y$ convex. Let $\alpha > 0$ . Then the set 

$$
C _ {\alpha} = \{\mathbf {x} \in \mathbb {E}: g (\mathbf {x}) \leq \alpha \}
$$

is $\frac { \sigma _ { g } } { \sqrt { 2 \alpha L _ { g } } }$ -strongly convex. 

Proof. Let x, $\textbf { y } \in \ C _ { \alpha }$ and $\lambda ~ \in ~ [ 0 , 1 ]$ . Define $\mathbf { x } _ { \lambda } = \lambda \mathbf { x } + ( 1 - \lambda ) \mathbf { y }$ . By the nonnegativity of g and the suficient decrease lemma (Lemma 10.4), we have 

$$
g (\mathbf {x} _ {\lambda}) \geq g (\mathbf {x} _ {\lambda}) - g \left(\mathbf {x} _ {\lambda} - \frac {1}{L _ {g}} \nabla g (\mathbf {x} _ {\lambda})\right) \geq \frac {1}{2 L _ {g}} \| \nabla g (\mathbf {x} _ {\lambda}) \| ^ {2}.
$$

Thus, 

$$
\| \nabla g (\mathbf {x} _ {\lambda}) \| \leq \sqrt {2 L _ {g} g (\mathbf {x} _ {\lambda})}.\tag{13.39}
$$

By the $\sigma _ { g } \mathrm { - s t r o n g }$ convexity of $g$ and the inequalities $g ( \mathbf { x } ) , g ( \mathbf { y } ) \leq \alpha$ 

$$
g (\mathbf {x} _ {\lambda}) \leq \lambda g (\mathbf {x}) + (1 - \lambda) g (\mathbf {y}) - \frac {\sigma_ {g}}{2} \lambda (1 - \lambda) \| \mathbf {x} - \mathbf {y} \| ^ {2} \leq \alpha - \beta ,\tag{13.40}
$$

where $\begin{array} { r } { \beta \equiv \frac { \sigma _ { g } } { 2 } \lambda ( 1 - \lambda ) \lVert \mathbf { x } - \mathbf { y } \rVert ^ { 2 } } \end{array}$ 

Denote $\begin{array} { r } { \tilde { \sigma } = \frac { \sigma _ { g } } { \sqrt { 2 \alpha L _ { g } } } } \end{array}$ . In order to show that $C _ { \alpha }$ is ˜σ-strongly convex, we will take $\mathbf { u } \in B [ \mathbf { 0 } , 1 ]$ and show that $\mathbf { x } _ { \lambda } + \gamma \mathbf { u } \in C _ { \alpha }$ , where $\begin{array} { r } { \gamma = \frac { \tilde { \sigma } } { 2 } \lambda ( 1 - \lambda ) \| \mathbf { x } - \mathbf { y } \| ^ { 2 } } \end{array}$ Indeed, 

$$
\begin{array}{l l} g (\mathbf {x} _ {\lambda} + \gamma \mathbf {u}) \leq g (\mathbf {x} _ {\lambda}) + \gamma \langle \nabla g (\mathbf {x} _ {\lambda}), \mathbf {u} \rangle + \frac {\gamma^ {2} L _ {g}}{2} \| \mathbf {u} \| ^ {2} & [ \text { descent lemma } ] \\ \leq g (\mathbf {x} _ {\lambda}) + \gamma \| \nabla g (\mathbf {x} _ {\lambda}) \| \cdot \| \mathbf {u} \| + \frac {\gamma^ {2} L _ {g}}{2} \| \mathbf {u} \| ^ {2} & [ \text { Cauchy - Schwarz } ] \\ \leq g (\mathbf {x} _ {\lambda}) + \gamma \sqrt {2 L _ {g} g (\mathbf {x} _ {\lambda})} \| \mathbf {u} \| + \frac {\gamma^ {2} L _ {g}}{2} \| \mathbf {u} \| ^ {2} & [ (1 3. 3 9) ] \\ = \left(\sqrt {g (\mathbf {x} _ {\lambda})} + \gamma \sqrt {\frac {L _ {g}}{2}} \| \mathbf {u} \|\right) ^ {2}, \end{array}
$$

which, combined with (13.40) and the fact that $\| \mathbf { u } \| \leq 1$ , implies that 

$$
g (\mathbf {x} _ {\lambda} + \gamma \mathbf {u}) \leq \left(\sqrt {\alpha - \beta} + \gamma \sqrt {\frac {L _ {g}}{2}}\right) ^ {2}.\tag{13.41}
$$

By the concavity of the square root function $\varphi ( t ) = \sqrt { t }$ , we have 

$$
\begin{array}{r l} & {\sqrt {\alpha - \beta} = \varphi (\alpha - \beta) \leq \varphi (\alpha) - \varphi^ {\prime} (\alpha) \beta = \sqrt {\alpha} - \frac {\beta}{2 \sqrt {\alpha}}} \\ & {\qquad = \sqrt {\alpha} - \frac {\sigma_ {g} \lambda (1 - \lambda) \| \mathbf {x} - \mathbf {y} \| ^ {2}}{4 \sqrt {\alpha}}} \\ & {\qquad = \sqrt {\alpha} - \frac {\sqrt {2 \alpha L _ {g}} \tilde {\sigma} \lambda (1 - \lambda) \| \mathbf {x} - \mathbf {y} \| ^ {2}}{4 \sqrt {\alpha}}} \\ & {\qquad = \sqrt {\alpha} - \sqrt {\frac {L _ {g}}{2}} \frac {\tilde {\sigma} \lambda (1 - \lambda) \| \mathbf {x} - \mathbf {y} \| ^ {2}}{2}} \\ & {\qquad = \sqrt {\alpha} - \gamma \sqrt {\frac {L _ {g}}{2}},} \end{array}
$$

which, along with (13.41), leads to the inequality $g ( \mathbf { x } _ { \lambda } + \gamma \mathbf { u } ) \leq \alpha$ 

Example 13.24 (strong convexity of Euclidean balls). Consider the $\mathrm { s e t } ^ { 7 5 }$ $C = B [ \mathbf { c } , r ] \subseteq \mathbb { E }$ , where $\mathbf { c } \in \mathbb { E }$ and $r \ > \ 0$ . Note that $\dot { C } = \mathrm { L e v } ( g , r ^ { 2 } )$ , where $g ( \mathbf { x } ) \ = \ \| \mathbf { x } - \mathbf { c } \| ^ { 2 }$ . Since here $L _ { q } = \sigma _ { q } = 2 , \alpha = r ^ { 2 }$ , it follows that the strong convexity parameter of the set is $\begin{array} { r } { \dot { \overline { { 2 \cdot 2 \cdot r ^ { 2 } } } } = \frac { 1 } { r } } \end{array}$ 

We will consider the problem 

$$
\min _ {\mathbf {x} \in C} f (\mathbf {x}),\tag{13.42}
$$

where we assume the following set of properties. 

Assumption 13.25. 

(A) $C$ is nonempty, compact, and σ-strongly convex. 

(B) $f : \mathbb { E } \to ( - \infty , \infty ]$ is convex $L _ { f } { - } s m o o t h$ over dom $( f )$ , which is assumed to be an open and convex set satisfying $C \subseteq \operatorname { d o m } ( f )$ 

(C) There exists $\delta > 0$ such that $\| \nabla f ( \mathbf { x } ) \| \geq \delta$ for any $\mathbf { x } \in C$ 

(D) The optimal set of problem (13.42) is nonempty and denoted by $X ^ { \ast }$ . The optimal value of the problem is denoted by $f _ { \mathrm { o p t } }$ 

As usual, for any $\mathbf { x } \in C .$ we use the notation 

$$
\mathbf {p} (\mathbf {x}) \in \operatorname * {a r g m i n} _ {\mathbf {p} \in C} \langle \nabla f (\mathbf {x}), \mathbf {p} \rangle , S (\mathbf {x}) = \langle \nabla f (\mathbf {x}), \mathbf {x} - \mathbf {p} (\mathbf {x}) \rangle .
$$

We begin by establishing the following result connecting $S ( \mathbf { x } )$ and the distance between x and $\mathbf { p } ( \mathbf { x } )$ 