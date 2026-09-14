# First-Order Methods in Optimization

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/06afcb51-43be-4db1-8b62-226fc4277ca3/4db016432898cda3d948f130bcd17784953f665788f3df1d0783740c4d4274b0.jpg)


## MOS-SIAM Series on Optimization

This series is published jointly by the Mathematical Optimization Society and the Society for Industrial and Applied Mathematics. It includes research monographs, books on applications, textbooks at all levels, and tutorials. Besides being of high scientific quality, books in the series must advance the understanding and practice of optimization. They must also be written clearly and at an appropriate level for the intended audience. 

## Editor-in-Chief

Katya Scheinberg 

Lehigh University 

## Editorial Board

Santanu S. Dey, Georgia Institute of Technology Maryam Fazel, University of Washington Andrea Lodi, University of Bologna Arkadi Nemirovski, Georgia Institute of Technology Stefan Ulbrich, Technische Universität Darmstadt Luis Nunes Vicente, University of Coimbra David Williamson, Cornell University Stephen J. Wright, University of Wisconsin 

## Series Volumes

Beck, Amir, First-Order Methods in Optimization 

Terlaky, Tamás, Anjos, Miguel F., and Ahmed, Shabbir, editors, Advances and Trends in Optimization with Engineering Applications 

Todd, Michael J., Minimum-Volume Ellipsoids: Theory and Algorithms 

Bienstock, Daniel, Electrical Transmission System Cascades and Vulnerability: An Operations Research Viewpoint 

Koch, Thorsten, Hiller, Benjamin, Pfetsch, Marc E., and Schewe, Lars, editors, Evaluating Gas Network Capacities 

Corberán, Ángel, and Laporte, Gilbert, Arc Routing: Problems, Methods, and Applications 

Toth, Paolo, and Vigo, Daniele, Vehicle Routing: Problems, Methods, and Applications, Second Edition 

Beck, Amir, Introduction to Nonlinear Optimization: Theory, Algorithms, and Applications with MATLAB 

Attouch, Hedy, Buttazzo, Giuseppe, and Michaille, Gérard, Variational Analysis in Sobolev and BV Spaces: 

Delfour, M. C., Introduction to Optimization and Semidifferential Calculus 

Conn, Andrew R., Scheinberg, Katya, and Vicente, Luis N., Introduction to Derivative-Free Optimization 

Ferris, Michael C., Mangasarian, Olvi L., and Wright, Stephen J., Linear Programming with MATLAB 

Attouch, Hedy, Buttazzo, Giuseppe, and Michaille, Gérard, Variational Analysis in Sobolev and BV Spaces: Applications to PDEs and Optimization 

Wallace, Stein W. and Ziemba, William T., editors, Applications of Stochastic Programming 

Grötschel, Martin, editor, The Sharpest Cut: The Impact of Manfred Padberg and His Work 

Renegar, James, A Mathematical View of Interior-Point Methods in Convex Optimization 

Ben-Tal, Aharon and Nemirovski, Arkadi, Lectures on Modern Convex Optimization: Analysis, Algorithms, and Engineering Applications 

Conn, Andrew R., Gould, Nicholas I. M., and Toint, Phillippe L., Trust-Region Methods 

# First-Order Methods in Optimization

Amir Beck 

Tel-Aviv University 

Tel-Aviv 

Israel 

Society for Industrial and Applied Mathematics Philadelphia 

Copyright © 2017 by the Society for Industrial and Applied Mathematics and the Mathematical Optimization Society 

## 10 9 8 7 6 5 4 3 2 1

All rights reserved. Printed in the United States of America. No part of this book may be reproduced, stored, or transmitted in any manner without the written permission of the publisher. For information, write to the Society for Industrial and Applied Mathematics, 3600 Market Street, 6th Floor, Philadelphia, PA 19104-2688 USA. 

Trademarked names may be used in this book without the inclusion of a trademark symbol. These names are used in an editorial context only; no infringement of trademark is intended. 

Publisher Kivmars Bowling 

Acquisitions Editor Paula Callaghan 

Developmental Editor Gina Rinelli Harris 

Managing Editor Kelly Thomas 

Production Editor Louis R. Primus 

Copy Editor Bruce Owens 

Production Manager Donna Witzleben 

Production Coordinator Cally Shrader 

Compositor Cheryl Hufnagle 

Graphic Designer Lois Sellers 

## Library of Congress Cataloging-in-Publication Data

Please visit www.siam.org/books/mo25 to view the CIP data. 

For My wife, Nili My daughters, Noy and Vered My parents, Nili and Itzhak 

## Contents

Preface xi
1 Vector Spaces 1
1.1 Definition 1
1.2 Dimension 2
1.3 Norms 2
1.4 Inner Products 2
1.5 Affine Sets and Convex Sets 3
1.6 Euclidean Spaces 3
1.7 The Space $\mathbb{R}^n$ 4
1.8 The Space $\mathbb{R}^{m\times n}$ 6
1.9 Cartesian Product of Vector Spaces 7
1.10 Linear Transformations 8
1.11 The Dual Space 9
1.12 The Bidual Space 10
1.13 Adjoint Transformations 11
1.14 Norms of Linear Transformations 12
2 Extended Real-Valued Functions 13
2.1 Extended Real-Valued Functions and Closedness 13
2.2 Closedness versus Continuity 18
2.3 Convex Functions 21
2.4 Support Functions 26
3 Subgradients 35
3.1 Definitions and First Examples 35
3.2 Properties of the Subdifferential Set 39
3.3 Directional Derivatives 44
3.4 Computing Subgradients 53
3.5 The Value Function 67
3.6 Lipschitz Continuity and Boundedness of Subgradients 71
3.7 Optimality Conditions 72
3.8 Summary of Weak and Strong Subgradient Calculus Results 84
4 Conjugate Functions 87
4.1 Definition and Basic Properties 87
4.2 The Biconjugate 89 

4.3 Conjugate Calculus Rules . . . . . . . . . . . . . . . . . . . . . . . . . . 91
4.4 Examples . . . . . . . . . . . . . . . . . . . . . . 93
4.5 Infimal Convolution and Conjugacy . . . . . . . . . . . . . . . 102
4.6 Subdifferentials of Conjugate Functions . . . . . . . . . . . 104

5 Smoothness and Strong Convexity 107
5.1 L-Smooth Functions . . . . . . . . . . . . 107
5.2 Strong Convexity . . . . . . . . . . . 117
5.3 Smoothness and Strong Convexity Correspondence 123

6 The Proximal Operator 129
6.1 Definition, Existence, and Uniqueness 129
6.2 First Set of Examples of Proximal Mappings 131
6.3 Prox Calculus Rules 135
6.4 Prox of Indicators—Orthogonal Projections 146
6.5 The Second Prox Theorem 157
6.6 Moreau Decomposition 160
6.7 The Moreau Envelope 163
6.8 Miscellaneous Prox Computations 170
6.9 Summary of Prox Computations 177

7 Spectral Functions 179
7.1 Symmetric Functions 179
7.2 Symmetric Spectral Functions over $\mathbb{S}^n$ 182
7.3 Symmetric Spectral Functions over $\mathbb{R}^{m\times n}$ 188

8 Primal and Dual Projected Subgradient Methods 195
8.1 From Gradient Descent to Subgradient Descent 195
8.2 The Projected Subgradient Method 201
8.3 The Stochastic Projected Subgradient Method 221
8.4 The Incremental Projected Subgradient Method 229
8.5 The Dual Projected Subgradient Method 232

9 Mirror Descent 247
9.1 From Projected Subgradient to Mirror Descent 247
9.2 Convergence Analysis 252
9.3 Mirror Descent for the Composite Model 260

10 The Proximal Gradient Method 269
10.1 The Composite Model 269
10.2 The Proximal Gradient Method 270
10.3 Analysis of the Proximal Gradient Method—The Nonconvex Case 272
10.4 Analysis of the Proximal Gradient Method—The Convex Case 281
10.5 The Proximal Point Method 288
10.6 Convergence of the Proximal Gradient Method—The Strongly Convex Case 288
10.7 The Fast Proximal Gradient Method—FISTA 290
10.8 Smoothing 304 

10.9 Non-Euclidean Proximal Gradient Methods . . . . . . . . . . 315   
11 The Block Proximal Gradient Method 331   
11.1 Decomposition Methods . . . . . . . . . . . . . . . . 331   
11.2 Model and Assumptions . . . . . . . . . . . . . . . . . 332   
11.3 The Toolbox . . . . . . . . . . . . . . . . . . 333   
11.4 The Cyclic Block Proximal Gradient Method . . . . . . . . . 338   
11.5 The Randomized Block Proximal Gradient Method . . . . . 347   
12 Dual-Based Proximal Gradient Methods 353   
12.1 The Primal and Dual Models . . . . . . . . . . 353   
12.2 The Dual Proximal Gradient Method . . . . . . . 355   
12.3 Fast Dual Proximal Gradient . . . . . . . 358   
12.4 Examples I . . . . . . . . . . 360   
12.5 The Dual Block Proximal Gradient Method 369   
12.6 Examples II 374   
13 The Generalized Conditional Gradient Method 379   
13.1 The Frank-Wolfe/Conditional Gradient Method 379   
13.2 The Generalized Conditional Gradient Method 380   
13.3 The Strongly Convex Case 391   
13.4 The Randomized Generalized Block Conditional Gradient Method 400   
14 Alternating Minimization 405   
14.1 The Method 405   
14.2 Coordinate-wise Minima 407   
14.3 The Composite Model 411   
14.4 Convergence in the Convex Case 413   
14.5 Rate of Convergence in the Convex Case 415   
15 ADMM 423   
15.1 The Augmented Lagrangian Method 423   
15.2 Alternating Direction Method of Multipliers (ADMM) 425   
15.3 Convergence Analysis of AD-PMM 427   
15.4 Minimizing $f_{1}(\mathbf{x}) + f_{2}(\mathbf{A}\mathbf{x})$ 432   
A Strong Duality and Optimality Conditions 439   
B Tables 443   
C Symbols and Notation 451   
D Bibliographic Notes 457   
Bibliography 463   
Index 473 

## Preface

This book, as the title suggests, is about first-order methods, namely, methods that exploit information on values and gradients/subgradients (but not Hessians) of the functions comprising the model under consideration. First-order methods go back to 1847 with the work of Cauchy on the steepest descent method. With the increase in the amount of applications that can be modeled as large- or even huge-scale optimization problems, there has been a revived interest in using simple methods that require low iteration cost as well as low memory storage. 

The primary goal of the book is to provide in a self-contained manner a comprehensive study of the main first-order methods that are frequently used in solving large-scale problems. This is done by gathering and reorganizing in a unified manner many results that are currently scattered throughout the literature. Special emphasis is placed on rates of convergence and complexity analysis. Although the name of the book is “first-order methods in optimization,” two disclaimers are in order. First, we will actually also consider methods that exploit additional operations at each iteration such as prox evaluations, linear oracles, exact minimization w.r.t. blocks of variables, and more, so perhaps a more suitable name would have been “simple methods in optimization.” Second, in order to be truly self-contained, the first part of the book (Chapters 1–7) is actually purely theoretical and contains essential topics that are crucial for the developments in the algorithmic part (Chapters 8–15). 

The book is intended for students and researchers with a background in advanced calculus and linear algebra, as well as prior knowledge in the fundamentals of optimization (some convex analysis, optimality conditions, and duality). A MATLAB toolbox implementing many of the algorithms described in the book was developed by the author and Nili Guttmann-Beck and can be found at www.siam.org/books/mo25. 

The outline of the book is as follows. Chapter 1 reviews important facts about vector spaces. Although the material is quite fundamental, it is advisable not to skip this chapter since many of the conventions regarding the underlying spaces used in the book are explained. Chapter 2 focuses on extended real-valued functions with a special emphasis on properties such as convexity, closedness, and continuity. Chapter 3 covers the topic of subgradients starting from basic definitions, continuing with directional derivatives, diferentiability, and subdiferentiability and ending with calculus rules. Optimality conditions are derived for convex problems (Fermat’s optimality condition), but also for the nonconvex composite model, which will be discussed extensively throughout the book. Conjugate functions are the subject of Chapter 4, which covers several issues, such as Fenchel’s inequality, the biconjugate, calculus rules, conjugate subgradient theorem, relations with the infimal convolution, and Fenchel’s duality theorem. Chapter 5 covers two diferent but closely related subjects: smoothness and strong convexity—several characterizations of each of these concepts are given, and their relation via the conjugate correspondence theorem is established. The proximal operator is discussed in Chapter 6, which includes a large amount of prox computations as well as calculus rules. The basic properties of the proximal mapping (first and second prox theorems and Moreau decomposition) are proved, and the Moreau envelope concludes the theoretical part of the chapter. The first part of the book ends with Chapter 7, which contains a study of symmetric spectral functions. The second, algorithmic part of the book starts with Chapter 8 with primal and dual projected subgradient methods. Several stepsize rules are discussed, and complexity results for both the convex and the strongly convex cases are established. The chapter also includes discussions on the stochastic as well as the incremental projected subgradient methods. The non-Euclidean version of the projected subgradient method, a.k.a. the mirror descent method, is discussed in Chapter 9. Chapter 10 is concerned with the proximal gradient method as well as its many variants and extensions. The chapter also studies several theoretical results concerning the so-called gradient mapping, which plays an important part in the convergence analysis of proximal gradient–based methods. The extension of the proximal gradient method to the block proximal gradient method is discussed in Chapter 11, while Chapter 12 considers the dual proximal gradient method and contains a result on a primal-dual relation that allows one to transfer rate of convergence results from the dual problem to the primal problem. The generalized conditional gradient method is the topic of Chapter 13, which contains the basic rate of convergence results of the method, as well as its block version, and discusses the efect of strong convexity assumptions on the model. The alternating minimization method is the subject of Chapter 14, where its convergence (as well as divergence) in many settings is established and illustrated. The book concludes with a discussion on the ADMM method in Chapter 15. 

My deepest thanks to Marc Teboulle, whose fundamental works in first-order methods form the basis of many of the results in the book. Marc introduced me to the world of optimization, and he is a constant source and inspiration and admiration. I would like to thank Luba Tetruashvili for reading the book and for her helpful remarks. It has been a pleasure to work with the extremely devoted and eficient SIAM staf. Finally, I would like to acknowledge the support of the Israel Science Foundation for supporting me while writing this book. 

# Chapter 1

# Vector Spaces

This chapter reviews several important facts about diferent aspects of vectors spaces that will be used throughout the book. More comprehensive and detailed accounts of these subjects can be found in advanced linear algebra books. 

## 1.1 Definition

A vector space E over R (or a “real vector space”) is a set of elements called vectors such that the following holds. 

(A) For any two vectors x, $\mathbf { y } \in \mathbb { E }$ , there corresponds a vector $\mathbf x + \mathbf y$ , called the sum of x and y, satisfying the following properties: 

1. ${ \bf x } + { \bf y } = { \bf y } + { \bf x }$ for any x, $\mathbf { y } \in \mathbb { E }$ 

2. ${ \bf x } + ( { \bf y } + { \bf z } ) = ( { \bf x } + { \bf y } ) + { \bf z }$ for any $\mathbf { x } , \mathbf { y } , \mathbf { z } \in \mathbb { E }$ 

3. There exists in E a unique vector 0 (called the zeros vector ) such that $\mathbf { x } + \mathbf { 0 } = \mathbf { x }$ for any x. 

4. For any $\mathbf { x } \in \mathbb { E }$ , there exists a vector $- \mathbf { x } \in \mathbb { E }$ such that $\mathbf { x } + ( - \mathbf { x } ) = \mathbf { 0 }$ 

(B) For any real number (also called scalar ) $\alpha \in \mathbb { R }$ and $\mathbf { x } \in \mathbb { E }$ , there corresponds a vector αx called the scalar multiplication of α and x satisfying the following properties: 

1. $\alpha ( \beta \mathbf { x } ) = ( \alpha \beta ) \mathbf { x }$ for any $\alpha , \beta \in \mathbb { R } , \mathbf { x } \in \mathbb { E }$ 

2. 1x = x for any $\mathbf { x } \in \mathbb { E }$ 

(C) The two operations (summation, scalar multiplication) satisfy the following properties: 

1. α(x + y) = αx + αy for any $\alpha \in \mathbb { R } , \mathbf { x } , \mathbf { y } \in \mathbb { E }$ 

2. $( \alpha + \beta ) \mathbf { x } = \alpha \mathbf { x } + \beta \mathbf { x }$ for any $\alpha , \beta \in \mathbb { R } , \mathbf { x } \in \mathbb { E }$ 

## 1.2 Dimension

A set of vectors $\left\{ \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } , \ldots , \mathbf { v } _ { n } \right\}$ in a vector space E is called linearly independent or just independent if the linear system 

$$
\sum_ {i = 1} ^ {n} \alpha_ {i} \mathbf {v} _ {i} = \mathbf {0}
$$

implies that $\alpha _ { 1 } = \alpha _ { 2 } = \cdot \cdot \cdot = \alpha _ { n } = 0$ . In other words, there does not exist a nontrivial linear combination of vectors that is equal to the zeros vector. A set of vectors $\{ \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } , \ldots , \mathbf { v } _ { n } \}$ is said to span E if for any $\mathbf { x } \in \mathbb { E }$ , there exist $\beta _ { 1 } , \beta _ { 2 } , \ldots , \beta _ { n } \in$ R such that 

$$
\mathbf {x} = \sum_ {i = 1} ^ {n} \beta_ {i} \mathbf {v} _ {i}.
$$

A basis of a vector space E is an independent set of vectors that spans $\mathbb { E } .$ It is well known that the number of vectors in all the bases of a vector space E is the same; this number is called the dimension of the space and is denoted by dim(E). In this book we will discuss only vector spaces with a finite dimension, namely, finite-dimensional vector spaces. 

## 1.3 Norms

A norm $\| \cdot \|$ on a vector space $\mathbb { E }$ is a function $\| \cdot \| : \mathbb { E } \to \mathbb { R }$ satisfying the following properties: 

1. (nonnegativity) $\| \mathbf { x } \| \geq 0$ for any $\mathbf { x } \in \mathbb { E }$ and $\| \mathbf { x } \| = 0$ if and only if $\mathbf { x } = \mathbf { 0 }$ 

2. (positive homogeneity) $\| \lambda \mathbf { x } \| = | \lambda | \cdot \| \mathbf { x } \|$ for any $\mathbf { x } \in \mathbb { E }$ and $\lambda \in \mathbb { R }$ 

3. (triangle inequality) $\| \mathbf { x } + \mathbf { y } \| \leq \| \mathbf { x } \| + \| \mathbf { y } \|$ for any $\mathbf { x } , \mathbf { y } \in \mathbb { E } .$ 

We will sometimes denote the norm of a space E by $\| \cdot \| _ { \mathbb { E } }$ to emphasize the identity of the space and to distinguish it from other norms. The open ball with center $\mathbf { c } \in \mathbb { E }$ and radius $r > 0$ is denoted by $B ( \mathbf { c } , r )$ and defined by 

$$
B (\mathbf {c}, r) = \{\mathbf {x} \in \mathbb {E}: \| \mathbf {x} - \mathbf {c} \| <   r \}.
$$

The closed ball with center $\mathbf { c } \in \mathbb { E }$ and radius $r > 0$ is denoted by $B [ \mathbf { c } , r ]$ and defined by 

$$
B [ \mathbf {c}, r ] = \{\mathbf {x} \in \mathbb {E}: \| \mathbf {x} - \mathbf {c} \| \leq r \}.
$$

We will sometimes use the notation $B _ { \parallel \cdot \parallel } [ \mathbf { c } , r ]$ or $B _ { \parallel \cdot \parallel } ( \mathbf { c } , r )$ to identify the specific norm that is being used. 

## 1.4 Inner Products

An inner product of a real vector space E is a function that associates to each pair of vectors x, y a real number, which is denoted by $\langle \mathbf { x } , \mathbf { y } \rangle$ and satisfies the following properties: 

1. (commutativity) $\langle \mathbf { x } , \mathbf { y } \rangle = \langle \mathbf { y } , \mathbf { x } \rangle$ for any x, $\mathbf { y } \in \mathbb { E }$ 

2. (linearity) $\langle \alpha _ { 1 } { \bf x } _ { 1 } + \alpha _ { 2 } { \bf x } _ { 2 } , { \bf y } \rangle = \alpha _ { 1 } \langle { \bf x } _ { 1 } , { \bf y } \rangle + \alpha _ { 2 } \langle { \bf x } _ { 2 } , { \bf y } \rangle$ for any $\alpha _ { 1 } , \alpha _ { 2 } \in \mathbb { R }$ and $\mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \mathbf { y } \in \mathbb { E }$ 

3. (positive definiteness) $\langle \mathbf { x } , \mathbf { x } \rangle \geq 0$ for any $\mathbf { x } \in \mathbb { E }$ and $\langle \mathbf { x } , \mathbf { x } \rangle = 0$ if and only if $\mathbf { x } = \mathbf { 0 }$ 

A vector space endowed with an inner product is also called an inner product space. At this point we would like to make the following important note: 

Underlying Spaces: In this book the underlying vector spaces, usually denoted by V or E, are always finite dimensional real inner product spaces with endowed inner product $\langle \cdot , \cdot \rangle$ and endowed norm  · . 

## 1.5 Afine Sets and Convex Sets

Given a real vector space E, a set $S \subseteq \mathbb { E }$ is called $a f f i n e$ if for any $\mathbf { x } , \mathbf { y } \in S$ and $\lambda \in \mathbb { R }$ , the inclusion $\lambda \mathbf { x } + ( 1 - \lambda ) \mathbf { y } \in S$ holds. For a set $S \subseteq \mathbb { E } ,$ , the afine hull of S, denoted by af(S), is the intersection of all afine sets containing S. Clearly, af(S) is by itself an afine set, and it is the smallest afine set containing S (w.r.t. inclusion). A hyperplane is a subset of E given by 

$$
H _ {\mathbf {a}, b} = \{\mathbf {x} \in \mathbb {E}: \langle \mathbf {a}, \mathbf {x} \rangle = b \},
$$

where $\mathbf { a } \in \mathbb { E }$ and $b \in \mathbb { R }$ . It is an easy exercise to show that hyperplanes are afine sets. 

A set $C \subseteq \mathbb { E }$ is called convex if for any $\mathbf { x } , \mathbf { y } \in C$ and $\lambda \in [ 0 , 1 ]$ it holds that $\lambda \mathbf { x } + ( 1 - \lambda ) \mathbf { y } \in C$ . Evidently, afine sets are always convex. Open and closed balls are always convex regardless of the choice of norm. For given $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ , the closed line segment between x and y is a subset of E denoted by $\left[ \mathbf { x } , \mathbf { y } \right]$ and defined as 

$$
[ \mathbf {x}, \mathbf {y} ] = \left\{\alpha \mathbf {x} + (1 - \alpha) \mathbf {y}: \alpha \in [ 0, 1 ] \right\}.
$$

The open line segment $\displaystyle ( \mathbf { x } , \mathbf { y } )$ is similarly defined as 

$$
(\mathbf {x}, \mathbf {y}) = \{\alpha \mathbf {x} + (1 - \alpha) \mathbf {y}: \alpha \in (0, 1) \}
$$

when $\mathbf x \neq \mathbf y$ and is the empty set ∅ when $\mathbf x = \mathbf y$ . Closed and open line segments are convex sets. Another example of convex sets are half-spaces, which are sets of the form 

$$
H _ {\mathbf {a}, b} ^ {-} = \{\mathbf {x} \in \mathbb {E}: \langle \mathbf {a}, \mathbf {x} \rangle \leq b \},
$$

where $\mathbf { a } \in \mathbb { E }$ and $b \in \mathbb { R }$ 

## 1.6 Euclidean Spaces

A finite dimensional real vector space equipped with an inner product $\langle \cdot , \cdot \rangle$ is called a Euclidean space if it is endowed with the norm $\| \mathbf { x } \| = { \sqrt { \langle \mathbf { x } , \mathbf { x } \rangle } }$ , which is referred to as the Euclidean norm. 

## 1.7 The Space $\mathbb { R } ^ { n }$

The vector space $\mathbb { R } ^ { n }$ (n being a positive integer) is the set of n-dimensional column vectors with real components endowed with the component-wise addition operator, 

$$
\left( \begin{array}{c} x _ {1} \\ x _ {2} \\ \vdots \\ x _ {n} \end{array} \right) + \left( \begin{array}{c} y _ {1} \\ y _ {2} \\ \vdots \\ y _ {n} \end{array} \right) = \left( \begin{array}{c} x _ {1} + y _ {1} \\ x _ {2} + y _ {2} \\ \vdots \\ x _ {n} + y _ {n} \end{array} \right),
$$

and the scalar-vector product, 

$$
\lambda \left( \begin{array}{c} x _ {1} \\ x _ {2} \\ \vdots \\ x _ {n} \end{array} \right) = \left( \begin{array}{c} \lambda x _ {1} \\ \lambda x _ {2} \\ \vdots \\ \lambda x _ {n} \end{array} \right),
$$

where in the above $x _ { 1 } , x _ { 2 } , \ldots , x _ { n } , \lambda$ are real numbers. We will denote the standard basis of $\mathbb { R } ^ { n }$ by $\mathbf { e } _ { 1 } , \mathbf { e } _ { 2 } , \ldots , \mathbf { e } _ { n }$ , where $\mathbf { e } _ { i }$ is the n-length column vector whose ith component is one while all the others are zeros. The column vectors of all ones and all zeros will be denoted by e and $\mathbf { 0 } _ { ; }$ , respectively, where the length of the vectors will be clear from the context. 

By far the most used inner product in $\mathbb { R } ^ { n }$ is the dot product defined by 

$$
\langle \mathbf {x}, \mathbf {y} \rangle = \sum_ {i = 1} ^ {n} x _ {i} y _ {i}.
$$

Inner Product in $\mathbb { R } ^ { n }$ : In this book, unless otherwise stated, the endowed inner product in $\mathbb { R } ^ { n }$ is the dot product. 

Of course, the dot product is not the only possible inner product that can be defined over $\mathbb { R } ^ { n }$ . Another useful option is the $\mathbf { Q } \mathbf { \cdot }$ -inner product, which is defined as 

$$
\langle \mathbf {x}, \mathbf {y} \rangle_ {\mathbf {Q}} = \mathbf {x} ^ {T} \mathbf {Q} \mathbf {y},
$$

where $\mathbf { Q }$ is a positive definite $n \times n$ matrix. Obviously, the Q-inner product amounts to the dot product when $\mathbf Q = \mathbf I$ $\operatorname { I f } \mathbb { R } ^ { n }$ is endowed with the dot product, then the associated Euclidean norm is the l -norm 

$$
\| \mathbf {x} \| _ {2} = \sqrt {\langle \mathbf {x} , \mathbf {x} \rangle} = \sqrt {\sum_ {i = 1} ^ {n} x _ {i} ^ {2}}.
$$

If $\mathbb { R } ^ { n }$ is endowed with the $\mathbf { Q } \mathbf { - }$ -inner product, then the associated Euclidean norm is the Q-norm 

$$
\| \mathbf {x} \| _ {\mathbf {Q}} = \sqrt {\mathbf {x} ^ {T} \mathbf {Q} \mathbf {x}}.
$$

For a given $p \geq 1$ , the $l _ { p } { \mathrm { - n o r m } }$ on $\mathbb { R } ^ { n }$ is given by the formula 

$$
\| \mathbf {x} \| _ {p} = \sqrt [ p ]{\sum_ {i = 1} ^ {n} | x _ {i} | ^ {p}}.
$$

The $l _ { \infty } \mathrm { - n o r m }$ on $\mathbb { R } ^ { n }$ is defined by 

$$
\| \mathbf {x} \| _ {\infty} = \max _ {i = 1, 2, \dots , n} | x _ {i} |.
$$

## 1.7.1 Subsets of $\mathbb { R } ^ { n }$

The nonnegative orthant is the subset of $\mathbb { R } ^ { n }$ consisting of all vectors in $\mathbb { R } ^ { n }$ with nonnegative components and is denoted by $\mathbb { R } _ { + } ^ { n }$ : 

$$
\mathbb {R} _ {+} ^ {n} = \left\{\left(x _ {1}, x _ {2}, \dots , x _ {n}\right) ^ {T}: x _ {1}, x _ {2}, \dots , x _ {n} \geq 0 \right\}.
$$

Similarly, the positive orthant consists of all the vectors in $\mathbb { R } ^ { n }$ with positive components and is denoted by $\mathbb { R } _ { + + } ^ { n }$ 

$$
\mathbb {R} _ {+ +} ^ {n} = \left\{(x _ {1}, x _ {2}, \ldots , x _ {n}) ^ {T}: x _ {1}, x _ {2}, \ldots , x _ {n} > 0 \right\}.
$$

The unit simplex, denoted by $\Delta _ { n } .$ , is the subset of $\mathbb { R } ^ { n }$ comprising all nonnegative vectors whose components sum up to one: 

$$
\Delta_ {n} = \left\{\mathbf {x} \in \mathbb {R} ^ {n}: \mathbf {x} \geq \mathbf {0}, \mathbf {e} ^ {T} \mathbf {x} = 1 \right\}.
$$

Given two vectors $\boldsymbol { \ell } , { \mathbf { u } } \in \mathbb { R } ^ { n }$ that satisfy $\ell \leq \mathbf { u } .$ , the box with lower bounds - and upper bounds u is denoted by Box[-, u] and defined as 

$$
\operatorname{Box} [ \boldsymbol {\ell}, \mathbf {u} ] = \left\{\mathbf {x} \in \mathbb {R} ^ {n}: \boldsymbol {\ell} \leq \mathbf {x} \leq \mathbf {u} \right\}.
$$

Thus, for example, Box $[ - \mathbf { e } , \mathbf { e } ] = [ - 1 , 1 ] ^ { n }$ 

## 1.7.2 Operations on Vectors in $\mathbb { R } ^ { n }$

There are several operations on vectors in $\mathbb { R } ^ { n }$ that will be frequently used in the book. For a given vector $\mathbf { x } \in \mathbb { R } ^ { n }$ , the vector $[ \mathbf { x } ] _ { + }$ is the nonnegative part of x defined by $[ \mathbf { x } ] _ { + } = ( \operatorname* { m a x } \{ x _ { i } , 0 \} ) _ { i = 1 } ^ { n }$ . For a given $\mathbf { x } \in \mathbb { R } ^ { n }$ , the vector $| \mathbf { x } |$ is the vector of component-wise absolute values $( | x _ { i } | ) _ { i = 1 } ^ { n }$ , and the vector $\operatorname { s g n } ( \mathbf { x } )$ is defined as 

$$
\operatorname{sgn} (\mathbf {x}) _ {i} = \left\{ \begin{array}{l l} 1, & x _ {i} \geq 0, \\ - 1, & x _ {i} <   0. \end{array} \right.
$$

For two vectors a, $\mathbf { b } \in \mathbb { R } ^ { n }$ , their Hadamard product, denoted by $\mathbf { a } \odot \mathbf { b }$ , is the vector comprising the component-wise products: a $\odot \mathbf { b } = ( a _ { i } b _ { i } ) _ { i = 1 } ^ { n }$ 

## 1.8 The Space $\mathbb { R } ^ { m \times n }$

The set of all real-valued $m \times n$ matrices is denoted by $\mathbb { R } ^ { m \times n }$ . This is a vector space with the component-wise addition as the summation operation and the componentwise scalar multiplication as the “scalar-vector multiplication” operation. The dot product in $\mathbb { R } ^ { m \times n }$ is defined by 

$$
\langle \mathbf {A}, \mathbf {B} \rangle = \mathrm{Tr} (\mathbf {A} ^ {T} \mathbf {B}) = \sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n} A _ {i j} B _ {i j}, \quad \mathbf {A}, \mathbf {B} \in \mathbb {R} ^ {m \times n}.
$$

The space $\mathbb { R } ^ { m \times n }$ is sometimes associated with the space $\mathbb { R } ^ { m n }$ in the sense that each matrix in $\mathbb { R } ^ { m \times n }$ corresponds to the mn-length vector constructed by stacking the columns of the matrix. Unless otherwise stated, we will assume that the inner product in $\mathbb { R } ^ { m \times n }$ is the dot product. 

Inner Product in $\mathbb { R } ^ { m \times n }$ : In this book, unless otherwise stated, the endowed inner product in $\mathbb { R } ^ { m \times n }$ is the dot product. 

## 1.8.1 Subsets of $\mathbb { R } ^ { n \times n }$

The set of all $n \times n$ symmetric matrices is denoted by $\mathbb { S } ^ { n }$ : 

$$
\mathbb {S} ^ {n} = \left\{\mathbf {A} \in \mathbb {R} ^ {n \times n}: \mathbf {A} = \mathbf {A} ^ {T} \right\}.
$$

Note that $\mathbb { S } ^ { n }$ is also a vector space with the same summation and scalar multiplication operations as in $\mathbb { R } ^ { n \times n }$ . The inner product in $\mathbb { S } ^ { n }$ , unless otherwise stated, is the dot product. 

The set of all $n \times n$ positive semidefinite matrices is denoted by $\mathbb { S } _ { + } ^ { n }$ : 

$$
\mathbb {S} _ {+} ^ {n} = \left\{\mathbf {A} \in \mathbb {R} ^ {n \times n}: \mathbf {A} \succeq \mathbf {0} \right\}.
$$

The set of all $n \times n$ positive definite matrices is denoted by ${ \mathbb S } _ { + + } ^ { n }$ 

$$
\mathbb {S} _ {+ +} ^ {n} = \left\{\mathbf {A} \in \mathbb {R} ^ {n \times n}: \mathbf {A} \succ \mathbf {0} \right\}.
$$

Obviously, the inclusion $\mathbb { S } _ { + + } ^ { n } \subseteq \mathbb { S } _ { + } ^ { n } \subseteq \mathbb { S } ^ { n }$ holds. Similarly, $\mathbb { S } _ { - } ^ { n }$ is the set of all $n \times n$ negative semidefinite matrices, and $\mathbb { S } _ { -- } ^ { n }$ is the set of all $n \times n$ negative definite matrices: 

$$
\begin{array}{c} \mathbb {S} _ {-} ^ {n} = \left\{\mathbf {A} \in \mathbb {R} ^ {n \times n}: \mathbf {A} \preceq \mathbf {0} \right\}, \\ \mathbb {S} _ {- - } ^ {n} = \left\{\mathbf {A} \in \mathbb {R} ^ {n \times n}: \mathbf {A} \prec \mathbf {0} \right\}. \end{array}
$$

The set of all $n \times n$ orthogonal matrices is denoted by $\mathbb { O } ^ { n }$ : 

$$
\mathbb {O} ^ {n} = \left\{\mathbf {A} \in \mathbb {R} ^ {n \times n}: \mathbf {A A} ^ {T} = \mathbf {A} ^ {T} \mathbf {A} = \mathbf {I} \right\}.
$$

## 1.8.2 Norms in $\mathbb { R } ^ { m \times n }$

If ${ \mathbb { R } } ^ { m \times n }$ is endowed with the dot product, then the corresponding Euclidean norm is the Frobenius norm defined by 

$$
\| \mathbf {A} \| _ {F} = \sqrt {\operatorname{Tr} (\mathbf {A} ^ {T} \mathbf {A})} = \sqrt {\sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n} A _ {i j} ^ {2}}, \quad \mathbf {A} \in \mathbb {R} ^ {m \times n}.
$$

Many examples of matrix norms are generated by using the concept of induced norms, which we now describe. Given a matrix $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ and two norms $\| \cdot \| _ { a }$ and $\| \cdot \| _ { b }$ on $\mathbb { R } ^ { n }$ and $\mathbb { R } ^ { m }$ , respectively, the induced matrix norm $\| \mathbf { A } \| _ { a , b }$ is defined by 

$$
\| \mathbf {A} \| _ {a, b} = \max _ {\mathbf {x}} \{\| \mathbf {A x} \| _ {b}: \| \mathbf {x} \| _ {a} \leq 1 \}.
$$

It can be easily shown that the above definition implies that for any $\mathbf { x } \in \mathbb { R } ^ { n }$ , the inequality 

$$
\| \mathbf {A} \mathbf {x} \| _ {b} \leq \| \mathbf {A} \| _ {a, b} \| \mathbf {x} \| _ {a}
$$

holds. We refer to the matrix norm $\Vert \cdot \Vert _ { a , b }$ as the $( a , b ) – \mathrm { n o r m }$ . When $a = b ,$ , we will simply refer to it as an a-norm and omit one of the subscripts in its notation, that is, use the notation $\| \cdot \| _ { a }$ instead of $\| \cdot \| _ { a , a }$ 

Example 1.1 (spectral norm). $\operatorname { I f } \parallel \cdot \parallel _ { a } = \parallel \cdot \parallel _ { b } = \parallel \cdot \parallel _ { 2 }$ , then the induced norm of a matrix $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ is the maximum singular value of A: 

$$
\| \mathbf {A} \| _ {2} = \| \mathbf {A} \| _ {2, 2} = \sqrt {\lambda_ {\max} (\mathbf {A} ^ {T} \mathbf {A})} \equiv \sigma_ {\max} (\mathbf {A}).
$$

Example 1.2 (1-norm). When $\| \cdot \| _ { a } = \| \cdot \| _ { b } = \| \cdot \| _ { 1 }$ , the induced matrix norm of a matrix $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ is given by 

$$
\| \mathbf {A} \| _ {1} = \max _ {j = 1, 2, \dots , n} \sum_ {i = 1} ^ {m} | A _ {i, j} |.
$$

This norm is also called the maximum absolute column sum norm. ■ 

Example 1.3 (∞-norm). When $\| \cdot \| _ { a } = \| \cdot \| _ { b } = \| \cdot \| _ { \infty }$ , the induced matrix norm of a matrix $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ is given by 

$$
\| \mathbf {A} \| _ {\infty} = \max _ {i = 1, 2, \dots , m} \sum_ {j = 1} ^ {n} | A _ {i, j} |.
$$

This norm is also called the maximum absolute row sum norm. 

## 1.9 Cartesian Product of Vector Spaces

Given m vector spaces $\mathbb { E } _ { 1 } , \mathbb { E } _ { 2 } , \ldots , \mathbb { E } _ { m }$ equipped with inner products $\langle \cdot , \cdot \rangle _ { \mathbb { E } _ { i } }$ , their Cartesian product $\mathbb { E } _ { 1 } \times \mathbb { E } _ { 2 } \times \cdot \cdot \cdot \times \mathbb { E } _ { m }$ is the vector space of all m-tuples $( \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 }$ $\ldots , \mathbf { v } _ { m } )$ equipped with the component-wise addition between vectors: 

$$
\left(\mathbf {v} _ {1}, \mathbf {v} _ {2}, \dots , \mathbf {v} _ {m}\right) + \left(\mathbf {w} _ {1}, \mathbf {w} _ {2}, \dots , \mathbf {w} _ {m}\right) = \left(\mathbf {v} _ {1} + \mathbf {w} _ {1}, \mathbf {v} _ {2} + \mathbf {w} _ {2}, \dots , \mathbf {v} _ {m} + \mathbf {w} _ {m}\right)
$$

and the scalar-vector multiplication operation given by 

$$
\alpha (\mathbf {v} _ {1}, \mathbf {v} _ {2}, \dots , \mathbf {v} _ {m}) = (\alpha \mathbf {v} _ {1}, \alpha \mathbf {v} _ {2}, \dots , \alpha \mathbf {v} _ {m}).
$$

The inner product in the Cartesian product space is defined as 

$$
\langle (\mathbf {v} _ {1}, \mathbf {v} _ {2}, \dots , \mathbf {v} _ {m}), (\mathbf {w} _ {1}, \mathbf {w} _ {2}, \dots , \mathbf {w} _ {m}) \rangle_ {\mathbb {E} _ {1} \times \mathbb {E} _ {2} \times \dots \times \mathbb {E} _ {m}} = \sum_ {i = 1} ^ {m} \langle \mathbf {v} _ {i}, \mathbf {w} _ {i} \rangle_ {\mathbb {E} _ {i}}.\tag{1.1}
$$

The space $\mathbb { R } \times \mathbb { R }$ , for example, consists of all two-dimensional row vectors, so in that respect it is diferent than $\mathbb { R } ^ { 2 }$ , which comprises all two-dimensional column vectors. However, with only a slight abuse of notation, we will occasionally refer to $\mathbb { R } \times \mathbb { R }$ as $\mathbb { R } ^ { 2 }$ 

Suppose that $\mathbb { E } _ { 1 } , \mathbb { E } _ { 2 } , \ldots , \mathbb { E } _ { m }$ are vector spaces with endowed norms $\| \cdot \| _ { \mathbb { E } _ { 1 } , \| }$ $\| _ { \mathbb { E } _ { 2 } } , \ldots , \| \cdot \| _ { \mathbb { E } _ { m } } .$ respectively. There are many ways to define a norm on the Cartesian product space $\mathbb { E } _ { 1 } \times \mathbb { E } _ { 2 } \times \cdot \cdot \cdot \times \mathbb { E } _ { m }$ . For example, for any $p \geq 1$ , we can define the composite $l _ { p } { \mathrm { - n o r m } }$ as 

$$
\left\| \left(\mathbf {u} _ {1}, \mathbf {u} _ {2}, \dots , \mathbf {u} _ {m}\right) \right\| = \sqrt [ p ]{\sum_ {i = 1} ^ {m} \left\| \mathbf {u} _ {i} \right\| _ {\mathbb {E} _ {i}} ^ {p}}.
$$

Another norm is a composite weighted $l _ { 2 } .$ -norm: 

$$
\| (\mathbf {u} _ {1}, \mathbf {u} _ {2}, \dots , \mathbf {u} _ {m}) \| = \sqrt {\sum_ {i = 1} ^ {m} \omega_ {i} \| \mathbf {u} _ {i} \| _ {\mathbb {E} _ {i}} ^ {2}},
$$

where $\omega _ { 1 } , \omega _ { 2 } , \ldots , \omega _ { m }$ are given positive real numbers. 

We will use the convention that if $\mathbb { E } _ { 1 } , \mathbb { E } _ { 2 } , \ldots , \mathbb { E } _ { m }$ are Euclidean spaces, then $\mathbb { E } _ { 1 } \times \mathbb { E } _ { 2 } \times \cdot \cdot \cdot \times \mathbb { E } _ { m }$ is also a Euclidean space, and consequently, by the definition (1.1) of the inner product in product spaces, 

$$
\| (\mathbf {u} _ {1}, \mathbf {u} _ {2}, \dots , \mathbf {u} _ {m}) \| _ {\mathbb {E} _ {1} \times \mathbb {E} _ {2} \times \dots \times \mathbb {E} _ {m}} = \sqrt {\sum_ {i = 1} ^ {m} \| \mathbf {u} _ {i} \| _ {\mathbb {E} _ {i}} ^ {2}}.
$$

## 1.10 Linear Transformations

Given two vector spaces E and V, a function $\mathcal { A } : \mathbb { E }  \mathbb { V }$ is called a linear transformation if the following property holds for any $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ and $\alpha , \beta \in \mathbb { R }$ : 

$$
\mathcal {A} (\alpha \mathbf {x} + \beta \mathbf {y}) = \alpha \mathcal {A} (\mathbf {x}) + \beta \mathcal {A} (\mathbf {y}).
$$

All linear transformations from $\mathbb { R } ^ { n }$ to $\mathbb { R } ^ { m }$ have the form 

$$
\mathcal {A} (\mathbf {x}) = \mathbf {A x}
$$

for some matrix $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ . All linear transformations from $\mathbb { R } ^ { m \times n }$ to $\mathbb { R } ^ { k }$ have the form 

$$
\mathcal {A} (\mathbf {X}) = \left( \begin{array}{c} \operatorname{Tr} (\mathbf {A} _ {1} ^ {T} \mathbf {X}) \\ \operatorname{Tr} (\mathbf {A} _ {2} ^ {T} \mathbf {X}) \\ \vdots \\ \operatorname{Tr} (\mathbf {A} _ {k} ^ {T} \mathbf {X}) \end{array} \right)
$$

for some $\mathbf { A } _ { 1 } , \mathbf { A } _ { 2 } , \ldots , \mathbf { A } _ { k } \in \mathbb { R } ^ { m \times n }$ . The identity transformation, denoted by $\mathcal { T } ,$ is defined by the relation $\mathcal { T } ( \mathbf { x } ) = \mathbf { x }$ for all $\mathbf { x } \in \mathbb { E }$ 

## 1.11 The Dual Space

A linear functional on a vector space E is a linear transformation from E to R. Given a vector space $\mathbb { E } .$ , the set of all linear functionals on E is called the dual space and is denoted by $\mathbb { E } ^ { * }$ . For inner product spaces, it is known that given a linear functional $f \in \mathbb { E } ^ { * }$ , there always exists $\mathbf { v } \in \mathbb { E }$ such that 

$$
f (\mathbf {x}) = \langle \mathbf {v}, \mathbf {x} \rangle .\tag{1.2}
$$

For the sake of simplicity of notation, we will represent the linear functional f given in (1.2) by the vector v. This correspondence between linear functionals and elements in E leads us to consider the elements in $\mathbb { E } ^ { * }$ as exactly the same as those in E. The inner product in $\mathbb { E } ^ { * }$ is the same as the inner product in E. Essentially, the only diference between E and $\mathbb { E } ^ { * }$ will be in the choice of norms of each of the spaces. Suppose that E is endowed with a norm $\| \cdot \|$ . Then the norm of the dual space, called the dual norm, is given by 

$$
\| \mathbf {y} \| _ {*} \equiv \max _ {\mathbf {x}} \{\langle \mathbf {y}, \mathbf {x} \rangle : \| \mathbf {x} \| \leq 1 \}, \quad \mathbf {y} \in \mathbb {E} ^ {*}.\tag{1.3}
$$

It is not dificult to show that the dual norm is indeed a norm. A useful property is that the maximum in (1.3) can be taken over the unit sphere rather than over the unit ball, meaning that the following formula is valid: 

$$
\| \mathbf {y} \| _ {*} = \max _ {\mathbf {x}} \{\langle \mathbf {y}, \mathbf {x} \rangle : \| \mathbf {x} \| = 1 \}, \quad \mathbf {y} \in \mathbb {E} ^ {*}.
$$

The definition of the dual norm readily implies the following generalized version of the Cauchy–Schwarz inequality. 

Lemma 1.4 (generalized Cauchy–Schwarz inequality). Let E be an inner product vector space endowed with a norm $\| \cdot \|$ . Then 

$$
| \langle \mathbf {y}, \mathbf {x} \rangle | \leq \| \mathbf {y} \| _ {*} \| \mathbf {x} \| f o r a n y \mathbf {y} \in \mathbb {E} ^ {*}, \mathbf {x} \in \mathbb {E}.\tag{1.4}
$$

Proof. If $\mathbf { x } = \mathbf { 0 }$ , the inequality is trivially satisfied. Otherwise, take $\tilde { \mathbf { x } } = \frac { \mathbf { x } } { \| \mathbf { x } \| }$ Obviously, $\| \tilde { \mathbf { x } } \| = 1$ , and hence, by the definition of the dual norm, we have 

$$
\left\| \mathbf {y} \right\| _ {*} \geq \langle \mathbf {y}, \tilde {\mathbf {x}} \rangle = \frac {1}{\left\| \mathbf {x} \right\|} \langle \mathbf {y}, \mathbf {x} \rangle ,
$$

showing that $\langle \mathbf { y } , \mathbf { x } \rangle \leq \| \mathbf { y } \| _ { * } \| \mathbf { x } \|$ . Plugging $- \mathbf { x }$ instead of $\mathbf { x }$ in the latter inequality, we obtain that $\langle \mathbf { y } , \mathbf { x } \rangle \geq - \| \mathbf { y } \| _ { * } \| \mathbf { x } \|$ , thus showing the validity of inequality (1.4). 

Another important result is that Euclidean norms are self-dual, meaning that $\| \cdot \| = \| \cdot \| _ { * }$ . Here of course we use our convention that the elements in the dual space $\mathbb { E } ^ { * }$ are the same as the elements in E. We can thus write, in only a slight abuse of notation,<sup>1</sup> that for any Euclidean space $\mathbb { E } , \mathbb { E } = \mathbb { E } ^ { * }$ 

Example 1.5 $\left( l _ { p } { \mathbf { - n o r m s } } \right)$ . Consider the space $\mathbb { R } ^ { n }$ endowed with the $l _ { p } { \mathrm { - n o r m } }$ When $p > 1$ , the dual norm is the $l _ { q } \mathrm { - n o r m }$ , where $q > 1$ is the number satisfying $\begin{array} { r } { \frac { 1 } { p } + \frac { 1 } { q } = 1 } \end{array}$ . When $p = 1$ , the dual norm is the $l _ { \infty } \mathrm { - n o r m }$ , and vice versa—the dual norm of the $l _ { \infty } \mathrm { - n o r m }$ is the $l _ { 1 } { \mathrm { - n o r m } }$ ■ 

Example 1.6 (Q-norms). Consider the space $\mathbb { R } ^ { n }$ endowed with the Q-norm, where $\mathbf { Q } \in \mathbb { S } _ { + + } ^ { n }$ . The dual norm of $\| \cdot \| _ { \mathbf { Q } } \mathrm { ~ i s ~ } \| \cdot \| _ { \mathbf { Q } ^ { - 1 } }$ , meaning 

$$
\left\| \mathbf {x} \right\| _ {\mathbf {Q} ^ {- 1}} = \sqrt {\mathbf {x} ^ {T} \mathbf {Q} ^ {- 1} \mathbf {x}}.
$$

As an example, consider the case where $\mathbf { Q }$ is diagonal: $\mathbf { Q } = \operatorname { d i a g } ( w _ { 1 } , w _ { 2 } , \dots , w _ { n } )$ with $w _ { 1 } , w _ { 2 } , \ldots , w _ { n } > 0$ . The Q-norm in this case takes the form 

$$
\| \mathbf {x} \| = \sqrt {\sum_ {i = 1} ^ {n} w _ {i} x _ {i} ^ {2}},
$$

and its dual norm is 

$$
\| \mathbf {x} \| _ {*} = \sqrt {\sum_ {i = 1} ^ {n} \frac {1}{w _ {i}} x _ {i} ^ {2}}. \quad \blacksquare
$$

Example 1.7 (dual norm of Cartesian products of spaces). Consider the space $\mathbb { E } = \mathbb { E } _ { 1 } \times \mathbb { E } _ { 1 } \times \cdot \cdot \cdot \times \mathbb { E } _ { m }$ , where $\mathbb { E } _ { 1 } , \mathbb { E } _ { 2 } , \ldots , \mathbb { E } _ { m }$ are inner product vectors spaces with norms $\| \cdot \| _ { \mathbb { E } _ { 1 } } , \| \cdot \| _ { \mathbb { E } _ { 2 } } , \ldots , \| \cdot \| _ { \mathbb { E } _ { m } } .$ , respectively. Recall that we assume that the inner product in the product space is given by 

$$
\langle (\mathbf {v} _ {1}, \mathbf {v} _ {2}, \dots , \mathbf {v} _ {m}), (\mathbf {w} _ {1}, \mathbf {w} _ {2}, \dots , \mathbf {w} _ {m}) \rangle = \sum_ {i = 1} ^ {m} \langle \mathbf {v} _ {i}, \mathbf {w} _ {i} \rangle .
$$

The dual space to $\mathbb { E } _ { 1 } \times \mathbb { E } _ { 2 } \times \cdot \cdot \cdot \times \mathbb { E } _ { m }$ is the product space $\mathbb { E } _ { 1 } ^ { * } \times \mathbb { E } _ { 2 } ^ { * } \times \cdot \cdot \cdot \times \mathbb { E } _ { m } ^ { * }$ with endowed norm defined as usual in dual spaces. For example, suppose that the norm on the product space is the composite weighted $l _ { 2 } { \mathrm { - n o r m } } \colon$ 

$$
\| (\mathbf {u} _ {1}, \mathbf {u} _ {2}, \dots , \mathbf {u} _ {m}) \| = \sqrt {\sum_ {i = 1} ^ {m} \omega_ {i} \| \mathbf {u} _ {i} \| _ {\mathbb {E} _ {i}} ^ {2}}, \quad \mathbf {u} _ {i} \in \mathbb {E} _ {i}, i = 1, 2, \dots , p,
$$

where $\omega _ { 1 } , \omega _ { 2 } , \ldots , \omega _ { m } > 0$ are given positive weights. Then it is simple to show that the dual norm in this case is given by 

$$
\| (\mathbf {v} _ {1}, \mathbf {v} _ {2}, \dots , \mathbf {v} _ {m}) \| _ {*} = \sqrt {\sum_ {i = 1} ^ {m} \frac {1}{\omega_ {i}} \| \mathbf {v} _ {i} \| _ {\mathbb {E} _ {i} ^ {*}} ^ {2}}, \quad \mathbf {v} _ {i} \in \mathbb {E} _ {i} ^ {*}, i = 1, 2, \dots , p.
$$

where $\| \cdot \| _ { \mathbb { E } _ { i } ^ { * } }$ is the dual norm to $\| \cdot \| _ { \mathbb { E } _ { i } }$ , namely, the norm of the dual space $\mathbb { E } _ { i } ^ { * }$ . ■ 

## 1.12 The Bidual Space

Given a vector space E, the dual space $\mathbb { E } ^ { * }$ is also a vector space, and we can also consider its dual space, namely, $\mathbb { E } ^ { * * }$ . This is the so-called bidual space. In the setting of finite dimensional spaces, the bidual space is the same as the original space (under our convention that the elements in the dual space are the same as the elements in the original space), and the corresponding norm (bidual norm) is the same as the original norm. 

## 1.13 Adjoint Transformations

Given two inner product vector spaces $\mathbb { E } , \mathbb { V }$ and a linear transformation $\mathcal { A }$ from $\mathbb { V }$ to E, the adjoint transformation, denoted by $\mathcal { A } ^ { T }$ , is a transformation from $\mathbb { E } ^ { * }$ $\mathbb { V } ^ { * }$ defined by the relation 

$$
\langle \mathbf {y}, \mathcal {A} (\mathbf {x}) \rangle = \langle \mathcal {A} ^ {T} (\mathbf {y}), \mathbf {x} \rangle
$$

for any $\mathbf x \in \mathbb V , \mathbf y \in \mathbb E ^ { * }$ . When $\mathbb { V } = \mathbb { R } ^ { n } , \mathbb { E } = \mathbb { R } ^ { m }$ (endowed with the dot product), and $\mathcal { A } ( \mathbf { x } ) = \mathbf { A } \mathbf { x }$ for some matrix $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ , then the adjoint transformation is given by $\mathbf { \mathcal { A } } ^ { T } ( \mathbf { x } ) = \mathbf { A } ^ { T } \mathbf { x }$ 

Example 1.8 (adjoint of a transformation from $\mathbb { R } ^ { m \times n } { \bf \Pi } _ { { \bf t o } } \mathbb { R } ^ { k } )$ . Consider now a linear transformation from the space $\mathbb { R } ^ { m \times n }$ to $\mathbb { R } ^ { k }$ . As was already mentioned in Section 1.10, such a transformation has the form 

$$
\mathcal {A} (\mathbf {X}) = \left( \begin{array}{c} \operatorname{Tr} (\mathbf {A} _ {1} ^ {T} \mathbf {X}) \\ \operatorname{Tr} (\mathbf {A} _ {2} ^ {T} \mathbf {X}) \\ \vdots \\ \operatorname{Tr} (\mathbf {A} _ {k} ^ {T} \mathbf {X}) \end{array} \right),
$$

where $\mathbf { A } _ { i } \in \mathbb { R } ^ { m \times n }$ are given matrices. The adjoint transformation $\mathcal { A } ^ { T }$ will be a transformation from $\mathbb { R } ^ { k }$ to $\mathbb { R } ^ { m \times n }$ . To find it, let us write the defining relation of the adjoint operator: 

$$
\langle \mathbf {y}, \mathcal {A} (\mathbf {X}) \rangle = \left\langle \mathcal {A} ^ {T} (\mathbf {y}), \mathbf {X} \right\rangle \text {   for   all   } \mathbf {X} \in \mathbb {R} ^ {m \times n}, \mathbf {y} \in \mathbb {R} ^ {k},
$$

which is the same as (recall that unless otherwise stated, the inner products in $\mathbb { R } ^ { m \times n }$ and $\mathbb { R } ^ { k }$ are the dot products) 

$$
\sum_ {i = 1} ^ {k} y _ {i} \mathrm{Tr} (\mathbf {A} _ {i} ^ {T} \mathbf {X}) = \langle \mathcal {A} ^ {T} (\mathbf {y}), \mathbf {X} \rangle \mathrm{forall} \mathbf {X} \in \mathbb {R} ^ {m \times n}, \mathbf {y} \in \mathbb {R} ^ {k},
$$

that is, 

$$
\operatorname{Tr} \left(\left[ \sum_ {i = 1} ^ {k} y _ {i} \mathbf {A} _ {i} \right] ^ {T} \mathbf {X}\right) = \langle \mathcal {A} ^ {T} (\mathbf {y}), \mathbf {X} \rangle \text {   for   all   } \mathbf {X} \in \mathbb {R} ^ {m \times n}, \mathbf {y} \in \mathbb {R} ^ {k}.
$$

Obviously, the above relation implies that the adjoint transformation is given by 

$$
\mathcal {A} ^ {T} (\mathbf {y}) = \sum_ {i = 1} ^ {k} y _ {i} \mathbf {A} _ {i}. \quad \blacksquare
$$

The adjoint of the adjoint transformation is the original transformation: $( A ^ { T } ) ^ { T }$ $= A .$ . It also holds that whenever A is an invertible transformation, 

$$
(\mathcal {A} ^ {T}) ^ {- 1} = (\mathcal {A} ^ {- 1}) ^ {T}.
$$

## 1.14 Norms of Linear Transformations

Let $\mathcal { A } : \mathbb { E }  \mathbb { V }$ be a linear transformation from a vector space E to a vector space V. Assume that E and V are endowed with the norms $\| \cdot \| _ { \mathbb { E } }$ and $\| \cdot \| _ { \mathbb { V } }$ , respectively. The norm of the linear transformation is defined by 

$$
\| \mathcal {A} \| \equiv \max \{\| \mathcal {A} (\mathbf {x}) \| _ {\mathbb {V}}: \| \mathbf {x} \| _ {\mathbb {E}} \leq 1 \}.
$$

It is not dificult to show that $\| \mathcal { A } \| = \| \mathcal { A } ^ { T } \|$ . There is a close connection between the notion of induced norms discussed in Section 1.8.2 and norms of linear transformations. Specifically, suppose that A is a linear transformation from $\mathbb { R } ^ { n }$ to R<sup>m</sup> given by 

$$
\mathcal {A} (\mathbf {x}) = \mathbf {A x},\tag{1.5}
$$

where $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ , and assume that $\mathbb { R } ^ { n }$ and $\mathbb { R } ^ { m }$ are endowed with the norms $\| \cdot \| _ { a }$ and $\| \cdot \| _ { b }$ , respectively. Then $\| \mathcal { A } \| = \| \mathbf { A } \| _ { a , b }$ , meaning that the induced norm of a matrix is actually the norm of the corresponding linear transformation given by the relation (1.5). 

Chapter 2 

# Extended Real-Valued Functions

Underlying Space: Recall that in this book, the underlying spaces (denoted usually by E or V) are finite-dimensional inner product vector spaces with inner product ·, · and norm  · . 

## 2.1 Extended Real-Valued Functions and Closedness

An extended real-valued function is a function defined over the entire underlying space that can take any real value, as well as the infinite values −∞ and ∞. Since infinite values are allowed, we also define the appropriate arithmetic operations with −∞ and ∞ as follows: 

$$
a + \infty = \infty + a = \infty (- \infty <   a <   \infty),
$$

$$
a - \infty = - \infty + a = - \infty (- \infty <   a <   \infty),
$$

$$
a \cdot \infty = \infty \cdot a = \infty (0 <   a <   \infty),
$$

$$
a \cdot (- \infty) = (- \infty) \cdot a = - \infty (0 <   a <   \infty),
$$

$$
a \cdot \infty = \infty \cdot a = - \infty (- \infty <   a <   0),
$$

$$
a \cdot (- \infty) = (- \infty) \cdot a = \infty (- \infty <   a <   0),
$$

$$
0 \cdot \infty = \infty \cdot 0 = 0 \cdot (- \infty) = (- \infty) \cdot 0 = 0.
$$

In a sense, the only “unnatural” rule is the last one, since the expression $^ { 6 } 0 \cdot \infty ^ { \prime \ }$ is considered to be undefined in some branches of mathematics, but in the context of extended real-valued functions, defining it as zero is the “correct” choice in the sense of consistency. We will also use the following natural order between finite and infinite numbers: 

$$
\begin{array}{r l} \infty > a & (- \infty \leq a <   \infty), \\ - \infty <   a & (- \infty <   a \leq \infty). \end{array}
$$

For an extended real-valued function $f : \mathbb { E } \to [ - \infty , \infty ]$ , the efective domain or just the domain is the set 

$$
\operatorname{dom} (f) = \{\mathbf {x} \in \mathbb {E}: f (\mathbf {x}) <   \infty \}.
$$

The notation ${ } ^ { * } f : \mathbb { E } \to [ - \infty , \infty ] ^ { * }$ means that $f$ is (potentially) extended real-valued (even if not explicitly stated). The notation ${ } ^ { * } f : \mathbb { E } \to ( - \infty , \infty ] ^ { * }$ means that $f$ is extended real-valued and does not attain the value $- \infty$ 

The simplest examples of extended real-valued functions are indicators. 

Example 2.1 (indicator functions). For any subset $C \subseteq \mathbb { E }$ , the indicator function of $C$ is defined to be the extended real-valued function given by 

$$
\delta_ {C} (\mathbf {x}) = \left\{ \begin{array}{l l} 0, & \mathbf {x} \in C, \\ \infty , & \mathbf {x} \notin C. \end{array} \right. \quad \blacksquare
$$

We obviously have 

$$
\mathrm{dom} (\delta_ {C}) = C.
$$

The epigraph of an extended real-valued function $f : \mathbb { E } \to [ - \infty , \infty ]$ is defined by 

$$
\operatorname{epi} (f) = \left\{\left(\mathbf {x}, y\right): f (\mathbf {x}) \leq y, \mathbf {x} \in \mathbb {E}, y \in \mathbb {R} \right\}.
$$

The epigraph is a subset of $\mathbb { E } \times \mathbb { R }$ . Note that if $( \mathbf { x } , y ) \in \mathrm { e p i } ( f )$ , then obviously $\mathbf { x } \in \operatorname { d o m } ( f )$ . A function $f : \mathbb { E } \to [ - \infty , \infty ]$ is called proper if it does not attain the value $- \infty$ and there exists at least one $\mathbf { x } \in \mathbb { E }$ such that $f ( \mathbf { x } ) < \infty$ , meaning that dom $( f )$ is nonempty. The notion of closedness will play an important role in much of the analysis in this book. 

Definition 2.2 (closed functions). A function $f : \mathbb { E } \to [ - \infty , \infty ]$ is closed $i f$ its epigraph is closed. 

The indicator function $\delta _ { C }$ is closed if and only if its underlying set $C$ is closed. 

Proposition 2.3 (closedness of indicators of closed sets). The indicator function $\delta _ { C }$ is closed if and only if $C$ is a closed set. 

Proof. The epigraph of $\delta _ { C }$ is given by 

$$
\operatorname{epi} (\delta_ {C}) = \left\{\left(\mathbf {x}, y\right) \in \mathbb {E} \times \mathbb {R}: \delta_ {C} (\mathbf {x}) \leq y \right\} = C \times \mathbb {R} _ {+},
$$

which is evidently closed if and only if C is closed. 

We thus obtained in particular that the domain of a closed indicator function is necessarily a closed set. However, in general, we note that the domain of a closed function might not be closed. A classical example for this observation is given below. 

Example 2.4. Consider the function $f : \mathbb { R } \to [ - \infty , \infty ]$ given by 

$$
f (x) = \left\{ \begin{array}{l l} \frac {1}{x}, & x > 0, \\ \infty , & \text { else }. \end{array} \right.
$$

The domain of the function, which is the open interval $( 0 , \infty )$ , is obviously not closed, but the function is closed since its epigraph 

$$
\operatorname{epi} (f) = \{(x, y): x y \geq 1, x > 0 \}
$$

is a closed set; see Figure 2.1. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/06afcb51-43be-4db1-8b62-226fc4277ca3/0aabfd34d58ccc4602c58c69f5d8d9d5976dc342cd380beecfa78731ffc4168a.jpg)



Figure 2.1. The epigraph of the function $\textstyle f ( x ) = { \frac { 1 } { x } }$ for $x > 0$ and ∞ otherwise.


A property that will be later shown to be equivalent to closedness is lower semicontinuity. 

Definition 2.5 (lower semicontinuity). A function $f : \mathbb { E } \to [ - \infty , \infty ]$ is called lower semicontinuous at $\mathbf x \in  { \mathbb { E } } \ i f$ 

$$
f (\mathbf {x}) \leq \liminf _ {n \to \infty} f (\mathbf {x} _ {n})
$$

for any sequence $\{ \mathbf { x } _ { n } \} _ { n \geq 1 } \subseteq \mathbb { E }$ for which ${ \bf x } _ { n }  { \bf x } \ a s \ n  \infty$ . A function $f : \mathbb { E } $ $[ - \infty , \infty ]$ is called lower semicontinuous if it is lower semicontinuous at each point in E. 

For any $\alpha \in \mathbb { R }$ , the α-level set of a function $f : \mathbb { E } \to [ - \infty , \infty ]$ is the set 

$$
\operatorname{Lev} (f, \alpha) = \{\mathbf {x} \in \mathbb {E}: f (\mathbf {x}) \leq \alpha \}.
$$

The following theorem shows that closedness and lower semicontinuity are equivalent properties, and they are both equivalent to the property that all the level sets of the function are closed. 

Theorem 2.6 (equivalence of closedness, lower semicontinuity, and closedness of level sets). Let $f : \mathbb { E } \to [ - \infty , \infty ]$ . Then the following three claims are equivalent: 

(i) f is lower semicontinuous. 

(ii) f is closed. 

(iii) For any $\alpha \in \mathbb { R }$ , the level set 

$$
\operatorname{Lev} (f, \alpha) = \{\mathbf {x} \in \mathbb {E}: f (\mathbf {x}) \leq \alpha \}
$$

is closed. 

Proof. $( \mathrm { i } \Rightarrow \mathrm { i i } )$ Suppose that f is lower semicontinuous. We will show that $\operatorname { e p i } ( f )$ is closed. For that, take $\{ ( \mathbf { x } _ { n } , y _ { n } ) \} _ { n \geq 1 } \subseteq \mathrm { e p i } ( f )$ such that $( { \bf x } _ { n } , y _ { n } )  ( { \bf x } ^ { * } , y ^ { * } )$ as $n \to \infty$ . Then for any $n \geq 1$ ， 

$$
f (\mathbf {x} _ {n}) \leq y _ {n}.
$$

Therefore, by the lower semicontinuity of $f$ at $\mathbf { x } ^ { * }$ , we have 

$$
f (\mathbf {x} ^ {*}) \leq \operatorname * {l i m i n f} _ {n \to \infty} f (\mathbf {x} _ {n}) \leq \operatorname * {l i m i n f} _ {n \to \infty} y _ {n} = y ^ {*},
$$

showing that $( \mathbf { x } ^ { * } , y ^ { * } ) \in \mathrm { e p i } ( f )$ and hence that f is closed. 

$( \mathrm { i i } \Rightarrow \mathrm { i i i } )$ Suppose that f is closed, namely, that $\operatorname { e p i } ( f )$ is closed. Let $\alpha \in \mathbb { R }$ We will show that $\operatorname { L e v } ( f , \alpha )$ is closed. If $\operatorname { L e v } ( f , \alpha ) = \varnothing$ , we are done. Otherwise, take a sequence $\{ \mathbf { x } _ { n } \} _ { n \geq 1 } \subseteq \operatorname { L e v } ( f , \alpha )$ that converges to x¯. Obviously $( \mathbf { x } _ { n } , \alpha ) \in \mathrm { e p i } ( f )$ for any n and $( { \bf x } _ { n } , \alpha )  ( \bar { \bf x } , \alpha )$ as $n  \infty$ . By the closedness of $\operatorname { e p i } ( f )$ , it follows that $( \bar { \mathbf { x } } , \alpha ) \in \mathrm { e p i } ( f )$ , establishing the fact that $\bar { \mathbf { x } } \in \mathrm { L e v } ( f , \alpha )$ 

(iii ⇒ i) Suppose that all the level sets of $f$ are closed. We will show that it is lower semicontinuous. Assume by contradiction that f is not lower semicontinuous, meaning that there exists $\mathbf { x } ^ { * } \in \mathbb { E }$ and $\{ \mathbf { x } _ { n } \} _ { n \geq 1 } \subseteq \mathbb { E }$ such that ${ \bf x } _ { n }  { \bf x } ^ { * }$ and lim in $\dot { \mathbf { \zeta } } _ { n  \infty } f ( \mathbf { x } _ { n } ) < f ( \mathbf { x } ^ { * } )$ . Take α that satisfies 

$$
\liminf _ {n \to \infty} f (\mathbf {x} _ {n}) <   \alpha <   f (\mathbf {x} ^ {*}).\tag{2.1}
$$

Then there exists a subsequence $\{ \mathbf { x } _ { n _ { k } } \} _ { k \geq 1 }$ such that $f ( \mathbf { x } _ { n _ { k } } ) \leq \alpha$ for all $k \geq 1$ . By the closedness of the level set $\operatorname { L e v } ( f , \alpha )$ and the fact that $\mathbf { x } _ { n _ { k } } \to \mathbf { x } ^ { * }$ as $k  \infty$ , it follows that $f ( \mathbf { x } ^ { * } ) \leq \alpha ,$ , which is a contradiction to (2.1), showing that (iii) implies (i). 

The next result shows that closedness of functions is preserved under afine change of variables, summation, multiplication by a nonnegative number, and maximization. Before stating the theorem, we note that in this book we will not use the $\mathrm { i n f / s u p }$ notation but rather use only the min/max notation, where the usage of this notation does not imply that the maximum or minimum is actually attained. 

## Theorem 2.7 (operations preserving closedness).

(a) Let $\mathcal { A } : \mathbb { E }  \mathbb { V }$ be a linear transformation from E to V and b $\in \mathbb { V }$ and let $f : \mathbb { V } \to [ - \infty , \infty ]$ be an extended real-valued closed function. Then the function $g : \mathbb { E }  [ - \infty , \infty ]$ given by 

$$
g (\mathbf {x}) = f (\mathcal {A} (\mathbf {x}) + \mathbf {b})
$$

is closed. 

(b) Let $f _ { 1 } , f _ { 2 } , \ldots , f _ { m } : \mathbb { E } \to ( - \infty , \infty ]$ be extended real-valued closed functions and let $\alpha _ { 1 } , \alpha _ { 2 } , \ldots , \alpha _ { m } \in \mathbb { R } _ { + }$ . Then the function $\begin{array} { r } { f = \sum _ { i = 1 } ^ { m } \alpha _ { i } f _ { i } } \end{array}$ is closed. 

(c) Let $f _ { i } : \mathbb { E } \to ( - \infty , \infty ] , i \in I$ be extended real-valued closed functions, where I is a given index set. Then the function 

$$
f (\mathbf {x}) = \max _ {i \in I} f _ {i} (\mathbf {x})
$$

is closed. 

Proof. (a) To show that $g$ is closed, take a sequence $\{ ( \mathbf { x } _ { n } , y _ { n } ) \} _ { n \geq 1 } \subseteq \mathrm { e p i } ( g )$ such that $( { \bf x } _ { n } , y _ { n } )  ( { \bf x } ^ { * } , y ^ { * } )$ as $n  \infty$ , where $\mathbf { x } ^ { \ast } \in \mathbb { E }$ and $y ^ { \ast } \in \mathbb { R }$ . The relation $\{ ( \mathbf { x } _ { n } , y _ { n } ) \} _ { n \geq 1 } \subseteq \mathrm { e p i } ( g )$ can be written equivalently as 

$$
f (\mathcal {A} (\mathbf {x} _ {n}) + \mathbf {b}) \leq y _ {n} \text {   for   all   } n \geq 1.
$$

Therefore, $( A ( \mathbf { x } _ { n } ) + \mathbf { b } , y _ { n } ) \in \mathrm { e p i } ( f )$ . Hence, since f is closed and $\boldsymbol { \mathcal { A } } ( \mathbf { x } _ { n } ) + \mathbf { b } $ $\boldsymbol { \mathcal { A } } ( \mathbf { x } ^ { * } ) + \mathbf { b } , y _ { n } \to y ^ { * }$ as $n  \infty ~ ( \mathrm { b y }$ the continuity of linear transformations), it follows that $( A ( \mathbf { x } ^ { * } ) + \mathbf { b } , y ^ { * } ) \in \mathrm { e p i } ( f )$ , meaning that 

$$
f (\mathcal {A} (\mathbf {x} ^ {*}) + \mathbf {b}) \leq y ^ {*},
$$

which is the same as the relation $( \mathbf { x } ^ { * } , y ^ { * } ) \in \mathrm { e p i } ( g )$ . We have shown that $\operatorname { e p i } ( g )$ is closed or, equivalently, that $g$ is closed. 

(b) We will prove that f is lower semicontinuous, which by Theorem 2.6 is equivalent to the closedness of f. Let $\{ { \bf x } _ { n } \} _ { n \ge 1 }$ be a sequence converging to $\mathbf { x } ^ { * }$ Then by the lower semicontinuity of $f _ { i }$ , for any $i = 1 , 2 , \dots , m$ 

$$
f _ {i} (\mathbf {x} ^ {*}) \leq \liminf _ {n \to \infty} f _ {i} (\mathbf {x} _ {n}).
$$

Multiplying the above inequality by $\alpha _ { i }$ and summing for $i = 1 , 2 , \dots , m$ gives 

$$
\left(\sum_ {i = 1} ^ {m} \alpha_ {i} f _ {i}\right) (\mathbf {x} ^ {*}) \leq \sum_ {i = 1} ^ {m} \operatorname * {l i m i n f} _ {n \to \infty} \alpha_ {i} f _ {i} (\mathbf {x} _ {n}) \leq \operatorname * {l i m i n f} _ {n \to \infty} \left(\sum_ {i = 1} ^ {m} \alpha_ {i} f _ {i}\right) (\mathbf {x} _ {n}),
$$

where in the last inequality we used the fact that for any two sequences of real numbers $\{ a _ { n } \} _ { n \geq 1 }$ and $\{ b _ { n } \} _ { n \geq 1 }$ , it holds that 

$$
\liminf _ {n \to \infty} a _ {n} + \liminf _ {n \to \infty} b _ {n} \leq \liminf _ {n \to \infty} (a _ {n} + b _ {n}).
$$

A simple induction argument shows that this property holds for an arbitrary number of sequences. We have thus established the lower semicontinuity and hence closedness of $\textstyle \sum _ { i = 1 } ^ { m } \alpha _ { i } f _ { i }$ 

(c) Since f<sub>i</sub> is closed for any $i \in I ,$ , it follows that $\operatorname { e p i } ( f _ { i } )$ is closed for any $i ,$ and hence epi $\begin{array} { r } { ( f ) = \bigcap _ { i \in I } \operatorname { e p i } ( f _ { i } ) } \end{array}$ is closed as an intersection of closed sets, implying that f is closed. 

## 2.2 Closedness versus Continuity

A relation between continuity and closedness is described in the following theorem stating that if an extended real-valued function is continuous over its domain,<sup>2</sup> which is assumed to be closed, then it is closed. 

Theorem 2.8. Let $f : \mathbb { E }  ( - \infty , \infty ]$ be an extended real-valued function that is continuous over its domain and suppose that dom(f) is closed. Then f is closed. 

Proof. To show that epi(f) is closed (which is the same as saying that f is closed), take a sequence $\{ ( \mathbf { x } _ { n } , y _ { n } ) \} _ { n \geq 1 } \subseteq \mathrm { e p i } ( f )$ for which $( { \bf x } _ { n } , y _ { n } )  ( { \bf x } ^ { * } , y ^ { * } )$ as $n  \infty$ for some $\mathbf { x } ^ { * } \in \mathbb { E }$ and $y \in \mathbb { R }$ . Since $\{ \mathbf { x } _ { n } \} _ { n \geq 1 } \subseteq \operatorname { d o m } ( f ) , \mathbf { x } _ { n } \to \mathbf { x } ^ { * }$ and dom(f) is closed, it follows that $\mathbf { x } ^ { * } \in$ dom(f). By the definition of the epigraph, we have for all $n \geq 1$ 2 

$$
f (\mathbf {x} _ {n}) \leq y _ {n}.\tag{2.2}
$$

Since f is continuous over dom $( f )$ , and in particular at $\mathbf { x } ^ { * }$ , it follows by taking n to $\infty$ in (2.2) that 

$$
f (\mathbf {x} ^ {*}) \leq y ^ {*},
$$

showing that $( \mathbf { x } ^ { * } , y ^ { * } ) \in \mathrm { e p i } ( f )$ , thus establishing the closedness of $\operatorname { e p i } ( f )$ □ 

In particular, any real-valued continuous function over E is closed. 

Corollary 2.9. Let $f : \mathbb { E } \to \mathbb { R }$ be continuous. Then f is closed. 

The above results demonstrate that there is a connection between continuity and closedness. However, these two notions are diferent, as the following example illustrates. 

Example 2.10. Consider the function $f _ { \alpha } : \mathbb { R }  ( - \infty , \infty ]$ given by 

$$
f _ {\alpha} (x) = \left\{ \begin{array}{l l} \alpha , & x = 0, \\ x, & 0 <   x \leq 1, \\ \infty , & \text { else }. \end{array} \right.
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/06afcb51-43be-4db1-8b62-226fc4277ca3/78bbb926dff0f074681806678df421f56a86e822924df64daaa23a33cd667af9.jpg)



Figure 2.2. An example of a closed function, which is not continuous over its domain.


This function is closed if and only if $\alpha \leq 0$ , and it is continuous over its domain if and only if $\alpha = 0$ . Thus, the function $f _ { - 0 . 1 }$ , plotted in Figure 2.2, is closed but not continuous over its domain. 

Example 2.11 $\left( l _ { 0 } { \bf - n o r m } \right)$ . Consider the $l _ { 0 } \mathrm { - n o r m }$ function $f : \mathbb { R } ^ { n }  \mathbb { R }$ given by 

$$
f (\mathbf {x}) = \| \mathbf {x} \| _ {0} \equiv \# \{i: x _ {i} \neq 0 \}.
$$

That is, $\| \mathbf { x } \| _ { 0 }$ is the number of nonzero elements in x. Note the $l _ { 0 } \mathrm { - n o r m }$ is actually not a norm. It does not satisfy the homogeneity property. Nevertheless, this terminology is widely used in the literature, and we will therefore adopt it. Although f is obviously not continuous, it is closed. To show this, note that 

$$
f (\mathbf {x}) = \sum_ {i = 1} ^ {n} I (x _ {i}),
$$

where $I : \mathbb { R }  \{ 0 , 1 \}$ is given by 

$$
I (y) = \left\{ \begin{array}{l l} 0, & y = 0, \\ 1, & y \neq 0. \end{array} \right.
$$

The function I is closed since its level sets, which are given by 

$$
\operatorname{Lev} (I, \alpha) = \left\{ \begin{array}{l l} \emptyset , & \alpha <   0, \\ \{0 \}, & \alpha \in [ 0, 1), \\ \mathbb {R}, & \alpha \geq 1, \end{array} \right.
$$

are closed sets. Therefore, $f ,$ as a sum of closed functions, is closed (Theorem $2 . 7 ( \mathrm { b } ) )$ . 

It is well known that a continuous function over a nonempty compact<sup>3</sup> set attains a minimum. This is the well-known Weierstrass theorem. We will now show that this property also holds for closed functions. 

Theorem 2.12 (Weierstrass theorem for closed functions). Let $f : \mathbb { E } $ $( - \infty , \infty ]$ be a proper closed function and assume that $C$ is a compact set satisfying C ∩ dom $( f ) \neq \emptyset$ . Then 

(a) f is bounded below over $C .$ 

(b) f attains its minimal value over $C .$ 

Proof. (a) Suppose by contradiction that f is not bounded below over C. Then there exists a sequence $\{ \mathbf { x } _ { n } \} _ { n \geq 1 } \subseteq C$ such that 

$$
\lim _ {n \to \infty} f (\mathbf {x} _ {n}) = - \infty .\tag{2.3}
$$

By the Bolzano–Weierstrass theorem, since C is compact, there exists a subsequence $\{ \mathbf { x } _ { n _ { k } } \} _ { k \geq 1 }$ that converges to a point $\bar { \bf x } \in C$ . By Theorem 2.6, $f$ is lower semicontinuous, and hence 

$$
f (\bar {\mathbf {x}}) \leq \liminf _ {k \to \infty} f (\mathbf {x} _ {n _ {k}}),
$$

which is a contradiction to (2.3). 

(b) Denote by $f _ { \mathrm { o p t } }$ the minimal value of f over C. Then there exists a sequence $\{ { \bf x } _ { n } \} _ { n \ge 1 }$ for which $f (  { \mathbf { x } } _ { n } ) \to f _ { \mathrm { o p t } }$ as n → ∞. As before, take a subsequence $\{ \mathbf { x } _ { n _ { k } } \} _ { k \ge 1 }$ that converges to some point $\bar { \bf x } \in C$ . By the lower semicontinuity of $f ,$ it follows that 

$$
f (\bar {\mathbf {x}}) \leq \lim _ {k \to \infty} f (\mathbf {x} _ {n _ {k}}) = f _ {\mathrm{opt}},
$$

showing that x¯ is a minimizer of f over $C . \ \mathsf { \Pi } \circ$ 

When the set C in the premise of Theorem 2.12 is not compact, the Weierstrass theorem does not guarantee the attainment of a minimizer, but attainment of a minimizer can be shown when the compactness of C is replaced by closedness if the function has a property called coerciveness. 

Definition 2.13 (coerciveness). A proper function $f : \mathbb { E }  ( - \infty , \infty ]$ is called coercive if 

$$
\lim _ {\| \mathbf {x} \| \to \infty} f (\mathbf {x}) = \infty .
$$

An important property of closed coercive functions is that they possess a minimizer on any closed set that has a nonempty intersection with the domain of the function. 

Theorem 2.14 (attainment under coerciveness). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper closed and coercive function and let $S \subseteq \mathbb { E }$ be a nonempty closed set satisfying $S \cap$ dom $( f ) \neq \emptyset$ . Then $f$ attains its minimal value over $S$ 

Proof. Let $\mathbf { x } _ { \mathrm { 0 } }$ be an arbitrary point in $S \cap \operatorname { d o m } ( f )$ . By the coerciveness of $f ,$ there exists an $M > 0$ such that 

$$
f (\mathbf {x}) > f \left(\mathbf {x} _ {0}\right) \text {   for   any   } \mathbf {x} \text {   satisfying   } \| \mathbf {x} \| > M.\tag{2.4}
$$

Since any minimizer $\mathbf { x } ^ { * }$ of $f$ over $S$ satisfies $f ( \mathbf { x } ^ { * } ) \leq f ( \mathbf { x } _ { 0 } )$ , it follows from (2.4) that the set of minimizers of $f$ over $S$ is the same as the set of minimizers of $f$ over $S \cap B _ { \parallel \cdot \parallel } [ \mathbf { 0 } , M ]$ , which is compact (both sets are closed, and $B _ { \parallel \cdot \parallel } [ \mathbf { 0 } , M ]$ is bounded) and nonempty (as it contains $\mathbf { x } _ { \mathrm { 0 } } )$ . Therefore, by the Weierstrass theorem for closed functions (Theorem 2.12), there exists a minimizer of $f$ over $S \cap B [ \mathbf { 0 } , M ]$ and hence also over $S _ { ☉ }$ . 

## 2.3 Convex Functions

## 2.3.1 Definition and Basic Properties

Like closedness, the definition of convexity for extended real-valued functions can be written in terms of the epigraph. 

Definition 2.15 (convex functions). An extended real-valued function $f : \mathbb { E } $ $[ - \infty , \infty ]$ is called convex $i f \operatorname { e p i } ( f )$ is a convex set. 

It is not dificult to show that a proper extended real-valued function $f : \mathbb { E } $ $( - \infty , \infty ]$ is convex if and only if dom(f) is convex and the restriction of $f$ to dom(f) is convex over dom $( f )$ in the sense of convexity of real-valued functions over convex domains. Using this observation, we conclude that a proper extended real-valued function $f$ is convex if and only if 

$$
f (\lambda \mathbf {x} + (1 - \lambda) \mathbf {y}) \leq \lambda f (\mathbf {x}) + (1 - \lambda) f (\mathbf {y}) \text { for   all } \mathbf {x}, \mathbf {y} \in \mathbb {E}, \lambda \in [ 0, 1 ],\tag{2.5}
$$

or, equivalently, if and only if dom(f) is convex and (2.5) is satisfied for any $\mathbf { x } , \mathbf { y } \in$ dom $( f )$ and $\lambda \in [ 0 , 1 ]$ . Inequality (2.5) is a special case of Jensen’s inequality, stating that for any $\mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { k } \in \mathbb { E }$ and $\lambda \in \Delta _ { k }$ , the following inequality holds: 

$$
f \left(\sum_ {i = 1} ^ {k} \lambda_ {i} \mathbf {x} _ {i}\right) \leq \sum_ {i = 1} ^ {k} \lambda_ {i} f (\mathbf {x} _ {i}).
$$

There are several operations that preserve convexity of extended real-valued convex functions. Some of them are summarized in Theorem 2.16 below. The proof can be easily deduced by combining two facts: (i) the same properties are known to hold for real-valued convex functions defined on a given convex domain, and (ii) the observation that a proper extended real-valued function is convex if and only if its domain is convex and its restriction to its domain is a real-valued convex function. 

Theorem 2.16 (operations preserving convexity). 

(a) Let $\mathcal { A } : \mathbb { E }  \mathbb { V }$ be a linear transformation from E to $\mathbb { V }$ (two underlying vector spaces) and $\mathbf { \Sigma } ) ~ \in ~ \mathbb { V } _ { : }$ , and let $f : \mathbb { V } \to ( - \infty , \infty ]$ be an extended real-valued convex function. Then the extended real-valued function $g : \mathbb { E }  ( - \infty , \infty ]$ given by 

$$
g (\mathbf {x}) = f (\mathcal {A} (\mathbf {x}) + \mathbf {b})
$$

is convex. 

(b) Let $f _ { 1 } , f _ { 2 } , \ldots , f _ { m } : \mathbb { E } \to ( - \infty , \infty ]$ be extended real-valued convex functions, and let $\alpha _ { 1 } , \alpha _ { 2 } , \ldots , \alpha _ { m } \in \mathbb { R } _ { + }$ . Then the function $\textstyle \sum _ { i = 1 } ^ { m } \alpha _ { i } f _ { i }$ is convex. 

(c) Let $f _ { i } : \mathbb { E } \to ( - \infty , \infty ] , i \in I ,$ be extended real-valued convex functions, where I is a given index set. Then the function 

$$
f (\mathbf {x}) = \max _ {i \in I} f _ {i} (\mathbf {x})
$$

is convex. 

Given a nonempty set $C \subseteq \mathbb { E } .$ , the distance function to C is defined by 

$$
d _ {C} (\mathbf {x}) = \min _ {\mathbf {y} \in C} \| \mathbf {x} - \mathbf {y} \|.
$$

The next example shows that for Euclidean spaces, the function $\begin{array} { r } { \frac { 1 } { 2 } \left( \| \mathbf { x } \| ^ { 2 } - d _ { C } ^ { 2 } ( \mathbf { x } ) \right) } \end{array}$ is always convex, regardless of whether C is convex or not. 

Example $\mathbf { 2 . 1 7 . ^ { 4 } }$ Suppose that the underlying space E is Euclidean (meaning that $\| \cdot \| = \sqrt { \langle \cdot , \cdot \rangle } )$ . Let $C \subseteq \mathbb { E }$ be a nonempty set, and consider the function 

$$
\varphi_ {C} (\mathbf {x}) = \frac {1}{2} \left(\| \mathbf {x} \| ^ {2} - d _ {C} ^ {2} (\mathbf {x})\right).
$$

To show that $\varphi _ { C }$ is convex, note that 

$$
d _ {C} ^ {2} (\mathbf {x}) = \min _ {\mathbf {y} \in C} \| \mathbf {x} - \mathbf {y} \| ^ {2} = \| \mathbf {x} \| ^ {2} - \max _ {\mathbf {y} \in C} [ 2 \langle \mathbf {y}, \mathbf {x} \rangle - \| \mathbf {y} \| ^ {2} ].
$$

Hence, 

$$
\varphi_ {C} (\mathbf {x}) = \max _ {\mathbf {y} \in C} \left[ \langle \mathbf {y}, \mathbf {x} \rangle - \frac {1}{2} \| \mathbf {y} \| ^ {2} \right].\tag{2.6}
$$

Therefore, since $\varphi _ { C }$ is a maximization of afine—and hence convex—functions, by Theorem 2.16(c), it is necessarily convex. 

Another operation that preserves convexity is partial minimization of jointly convex functions. 

Theorem 2.18 (convexity under partial minimization). Let $f : \mathbb { E } \times \mathbb { V } $ $( - \infty , \infty ]$ be a convex function satisfying the following property: 

$$
\text {   for   any   } \mathbf {x} \in \mathbb {E} \text {   there   exists   } \mathbf {y} \in \mathbb {V} \text {   for   which   } f (\mathbf {x}, \mathbf {y}) <   \infty .\tag{2.7}
$$

$L e t ^ { 5 } ~ g : \mathbb { E } \to [ - \infty , \infty )$ be defined by 

$$
g (\mathbf {x}) \equiv \min _ {\mathbf {y} \in \mathbb {E}} f (\mathbf {x}, \mathbf {y}).
$$

Then $g$ is convex. 

Proof. Let $\mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } \in \mathbb { E }$ and $\lambda \in [ 0 , 1 ]$ . To show the convexity of $^ { g , }$ we will prove that 

$$
g \left(\lambda \mathbf {x} _ {1} + (1 - \lambda) \mathbf {x} _ {2}\right) \leq \lambda g \left(\mathbf {x} _ {1}\right) + (1 - \lambda) g \left(\mathbf {x} _ {2}\right).\tag{2.8}
$$

The inequality is obvious if $\lambda = 0 \ \mathrm { o r } \ 1$ . We will therefore assume that $\lambda \in ( 0 , 1 )$ The proof is split into two cases. 

Case I: Here we assume that $g ( \mathbf { x } _ { 1 } ) , g ( \mathbf { x } _ { 2 } ) > - \infty$ . Take $\varepsilon > 0$ . Then there exist $\mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } \in \mathbb { V }$ such that 

$$
f (\mathbf {x} _ {1}, \mathbf {y} _ {1}) \leq g (\mathbf {x} _ {1}) + \varepsilon ,\tag{2.9}
$$

$$
f (\mathbf {x} _ {2}, \mathbf {y} _ {2}) \leq g (\mathbf {x} _ {2}) + \varepsilon .\tag{2.10}
$$

By the convexity of $f ,$ we have 

$$
\begin{array}{r c l} f (\lambda \mathbf {x} _ {1} + (1 - \lambda) \mathbf {x} _ {2}, \lambda \mathbf {y} _ {1} + (1 - \lambda) \mathbf {y} _ {2}) & \leq & \lambda f (\mathbf {x} _ {1}, \mathbf {y} _ {1}) + (1 - \lambda) f (\mathbf {x} _ {2}, \mathbf {y} _ {2}) \\ & \stackrel {(2. 9), (2. 1 0)} {\leq} & \lambda (g (\mathbf {x} _ {1}) + \varepsilon) + (1 - \lambda) (g (\mathbf {x} _ {2}) + \varepsilon) \\ & = & \lambda g (\mathbf {x} _ {1}) + (1 - \lambda) g (\mathbf {x} _ {2}) + \varepsilon . \end{array}
$$

Therefore, by the definition of $^ { g , }$ we can conclude that 

$$
g \left(\lambda \mathbf {x} _ {1} + (1 - \lambda) \mathbf {x} _ {2}\right) \leq \lambda g \left(\mathbf {x} _ {1}\right) + (1 - \lambda) g \left(\mathbf {x} _ {2}\right) + \varepsilon .
$$

Since the above inequality holds for any $\varepsilon > 0$ , it follows that (2.8) holds. 

Case II: Assume that at least one of the values $g ( \mathbf { x } _ { 1 } ) , g ( \mathbf { x } _ { 2 } )$ is equal −∞. We will assume without loss of generality that $g ( \mathbf { x } _ { 1 } ) = - \infty$ . In this case, (2.8) is equivalent to saying that $g ( \lambda \mathbf { x } _ { 1 } + ( 1 - \lambda ) \mathbf { x } _ { 2 } ) = - \infty$ . Take any $M \in \mathbb { R }$ . Then since $g ( \mathbf { x } _ { 1 } ) = - \infty$ 2 it follows that there exists $\mathbf { y } _ { 1 } \in \mathbb { V }$ for which 

$$
f (\mathbf {x} _ {1}, \mathbf {y} _ {1}) \leq M.
$$

By property (2.7), there exists $\mathbf { y } _ { 2 } \in \mathbb { V }$ for which $f ( \mathbf { x } _ { 2 } , \mathbf { y } _ { 2 } ) < \infty$ . Using the convexity of $f _ { ; }$ we obtain that 

$$
\begin{array}{c} f (\lambda \mathbf {x} _ {1} + (1 - \lambda) \mathbf {x} _ {2}, \lambda \mathbf {y} _ {1} + (1 - \lambda) \mathbf {y} _ {2}) \leq \lambda f (\mathbf {x} _ {1}, \mathbf {y} _ {1}) + (1 - \lambda) f (\mathbf {x} _ {2}, \mathbf {y} _ {2}) \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \end{array}
$$

which by the definition of $g$ implies the inequality 

$$
g (\lambda \mathbf {x} _ {1} + (1 - \lambda) \mathbf {x} _ {2}) \leq \lambda M + (1 - \lambda) f (\mathbf {x} _ {2}, \mathbf {y} _ {2}).
$$

Since the latter inequality holds for any $M \in \mathbb { R }$ and since $f ( \mathbf { x } _ { 2 } , \mathbf { y } _ { 2 } ) < \infty$ , it follows that $g ( \lambda \mathbf { x } _ { 1 } + ( 1 - \lambda ) \mathbf { x } _ { 2 } ) = - \infty$ , proving the result for the second case. 

## 2.3.2 The Infimal Convolution

Let $h _ { 1 } , h _ { 2 } : \mathbb { E }  ( - \infty , \infty ]$ be two proper functions. The infimal convolution of $h _ { 1 } , h _ { 2 }$ is defined by the following formula: 

$$
(h _ {1} \square h _ {2}) (\mathbf {x}) \equiv \min _ {\mathbf {u} \in \mathbb {E}} \{h _ {1} (\mathbf {u}) + h _ {2} (\mathbf {x} - \mathbf {u}) \}.
$$

A direct consequence of Theorem 2.18 is the following result stating that the infimal convolution of a proper convex function and a real-valued convex function is always convex. 

Theorem 2.19 (convexity of the infimal convolution). Let $h _ { 1 } : \mathbb { E }  ( - \infty , \infty ]$ be a proper convex function and let $h _ { 2 } : \mathbb { E }  \mathbb { R }$ be a real-valued convex function. Then $h _ { 1 } \sqcup h _ { 2 }$ is convex. 

Proof. Define $f ( { \bf x } , { \bf y } ) \equiv h _ { 1 } ( { \bf y } ) + h _ { 2 } ( { \bf x } - { \bf y } )$ . The convexity of $h _ { 1 }$ and $h _ { 2 }$ implies that $f$ is convex. In addition, property (2.7) holds since for any $\mathbf { x } \in \mathbb { E }$ , we can pick any $\mathbf { y } \in \mathrm { d o m } ( h _ { 1 } )$ and obtain that $f ( \mathbf { x } , \mathbf { y } ) = h _ { 1 } ( \mathbf { y } ) + h _ { 2 } ( \mathbf { x } - \mathbf { y } ) < \infty$ . Thus, by Theorem 2.18, the function $h _ { 1 } \boxed  \} h _ { 2 }$ , as a partial minimization function of $f ( \cdot , \cdot )$ w.r.t. the second argument is a convex function. 

Example 2.20 (convexity of the distance function). Let $C \subseteq \mathbb { E }$ be a nonempty convex set. The distance function can be written as the following infimal convolution: 

$$
d _ {C} (\mathbf {x}) = \min _ {\mathbf {y}} \{\| \mathbf {x} - \mathbf {y} \|: \mathbf {y} \in C \} = \min _ {\mathbf {y} \in \mathbb {E}} \left\{\delta_ {C} (\mathbf {y}) + \| \mathbf {x} - \mathbf {y} \| \right\} = (\delta_ {C} \square h _ {1}) (\mathbf {x}),
$$

where $h _ { 1 } ( \cdot ) = \| \cdot \|$ . Since $\delta _ { C }$ is proper and convex and $h _ { 1 }$ is real-valued convex, it follows by Theorem 2.19 that $d _ { C }$ is convex. 

## 2.3.3 Continuity of Convex Functions

It is well known that convex functions are continuous at points in the interior of their domain. This is explicitly recalled in the next result, which actually states a stronger property of convex functions—local Lipschitz continuity. 

Theorem 2.21 (local Lipschitz continuity of convex functions [10, Theorem 7.36]). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be convex. Let ${ \bf x } _ { 0 } \in  { }$ int(dom(f))). Then there exist $\varepsilon > 0$ and $L > 0$ such that $B [ { \bf x } _ { 0 } , \varepsilon ] \subseteq C$ and 

$$
| f (\mathbf {x}) - f (\mathbf {x} _ {0}) | \leq L \| \mathbf {x} - \mathbf {x} _ {0} \|\tag{2.11}
$$

for all $\mathbf { x } \in B [ \mathbf { x } _ { 0 } , \varepsilon ]$ 

Convex functions are not necessarily continuous at boundary points. Continuity is not guaranteed even when the function at hand is closed and convex (cf. Example 2.32). However, for univariate functions we will now show that closed and convex functions are continuous. 

Theorem 2.22 (continuity of closed convex univariate functions). Let $f : \mathbb { R }  ( - \infty , \infty ]$ be a proper closed and convex function. Then f is continuous over dom(f). 

Proof. Since f is convex, its domain is some interval $I = \operatorname { d o m } ( f )$ . If int $( I ) = \emptyset$ then I is a singleton, and consequently the continuity of f over I is obvious. Assume then that int $( I ) \neq \emptyset$ . The fact that $f$ is continuous over int(I) follows from Theorem 2.21. We only need to show the continuity of $f$ at the endpoints of I (if it exists). For that, we can assume without loss of generality that the interval I has a left endpoint $^ { a , }$ and we will prove the right continuity of $f$ at $a .$ We begin by showing that lim $\scriptstyle { \mathrm { 1 } } _ { t \to a ^ { + } } f ( t )$ exists. Let $c > a$ be an arbitrary scalar in I and define the function 

$$
g (t) \equiv \frac {f (c - t) - f (c)}{t}.
$$

Obviously, $g$ is defined on $( 0 , c - a ]$ . We will show that g is nondecreasing and upper bounded over $( 0 , c - a ]$ . For that, take $0 < t \leq s \leq c - a$ . Then 

$$
c - t = \left(1 - \frac {t}{s}\right) c + \frac {t}{s} (c - s),
$$

and hence, by the convexity of $f ,$ 

$$
f (c - t) \leq \left(1 - \frac {t}{s}\right) f (c) + \frac {t}{s} f (c - s),
$$

which after some rearrangement of terms can be seen to be equivalent to 

$$
\frac {f (c - t) - f (c)}{t} \leq \frac {f (c - s) - f (c)}{s}.
$$

Thus, 

$$
g (t) \leq g (s) \text {   for   any   } 0 <   t \leq s \leq c - a.\tag{2.12}
$$

Namely, $g$ is nondecreasing over $( 0 , c - a ]$ . To show the upper boundedness, just plug $s = c - a$ into (2.12) and obtain that 

$$
g (t) \leq g (c - a) \text {   for   any   } t \in (0, c - a ].\tag{2.13}
$$

We can thus conclude that $\begin{array} { r } { \operatorname* { l i m } _ { t \to ( c - a ) ^ { - } } g ( t ) } \end{array}$ exists and is equal to some real number . Hence, 

$$
f (c - t) = f (c) + t g (t) \rightarrow f (c) + (c - a) \ell ,
$$

as $t \to ( c - a ) ^ { - }$ , and consequently $\scriptstyle \operatorname* { l i m } _ { t \to a ^ { + } } f ( t )$ exists and is equal to $f ( c ) + ( c - a ) \ell .$ Using (2.13), we obtain that for any $t \in ( 0 , c - a ]$ ， 

$$
f (c - t) = f (c) + t g (t) \leq f (c) + (c - a) g (c - a) = f (c) + (c - a) \frac {f (a) - f (c)}{c - a} = f (a),
$$

implying the inequality $\begin{array} { r } { \operatorname* { l i m } _ { t \to a ^ { + } } f ( t ) \le f ( a ) } \end{array}$ . On the other hand, since $f$ is closed, it is also lower semicontinuous (Theorem 2.6), and thus $\begin{array} { r } { \operatorname* { l i m } _ { t \to a ^ { + } } f ( t ) \geq f ( a ) } \end{array}$ . Consequently, lim $\mathfrak { l } _ { t \to a ^ { + } } f ( t ) = f ( a )$ , proving the right continuity of $f$ at $a . \quad \sqcup$ 

## 2.4 Support Functions

Let $C \subseteq \mathbb { E }$ be a nonempty set. Then the support function of C is the function $\sigma _ { C } : \mathbb { E } ^ { * }  ( - \infty , \infty ]$ given by 

$$
\sigma_ {C} (\mathbf {y}) = \max _ {\mathbf {x} \in C} \langle \mathbf {y}, \mathbf {x} \rangle .
$$

For a fixed x, the linear function $\mathbf { y } \mapsto \langle \mathbf { y } , \mathbf { x } \rangle$ is obviously closed and convex. Therefore, by Theorems $2 . 7 ( \mathrm { c } )$ and $2 . 1 6 ( \mathrm { c } )$ , the support function, as a maximum of closed and convex functions, is always closed and convex, regardless of whether C is closed and/or convex. We summarize this property in the next lemma. 

Lemma 2.23 (closedness and convexity of support functions). Let $C \subseteq \mathbb { E }$ be a nonempty set. Then $\sigma _ { C }$ is a closed and convex function. 

In most of our discussions on support functions in this chapter, the fact that $\sigma _ { C }$ operates on the dual space $\mathbb { E } ^ { * }$ instead of E will have no importance—recall that we use the convention that the elements of $\mathbb { E } ^ { * }$ and E are the same. However, when norms will be involved, naturally, the dual norm will have to be used (cf. Example 2.31). 

Additional properties of support functions that follow directly by definition are given in Lemma 2.24 below. Note that for two sets A, B that reside in the same space, the sum $A + B$ stands for the Minkowski sum given by 

$$
A + B = \{\mathbf {a} + \mathbf {b}: \mathbf {a} \in A, \mathbf {b} \in B \}.
$$

Also, for a scalar $\alpha \in \mathbb { R }$ and a set $A \subseteq \mathbb { E }$ , the set αA is 

$$
\alpha A = \{\alpha \mathbf {a}: \mathbf {a} \in A \}.
$$

Lemma 2.24. 

(a) (positive homogeneity) For any nonempty set $C \subseteq \mathbb { E } , \mathbf { y } \in \mathbb { E } ^ { * }$ and $\alpha \geq 0$ 

$$
\sigma_ {C} (\alpha \mathbf {y}) = \alpha \sigma_ {C} (\mathbf {y}).
$$

(b) (subadditivity) For any nonempty set $C \subseteq \mathbb { E }$ and $\mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } \in \mathbb { E } ^ { * }$ 

$$
\sigma_ {C} (\mathbf {y} _ {1} + \mathbf {y} _ {2}) \leq \sigma_ {C} (\mathbf {y} _ {1}) + \sigma_ {C} (\mathbf {y} _ {2}).
$$

(c) For any nonempty set $C \subseteq \mathbb { E } , \mathbf { y } \in \mathbb { E } ^ { * }$ and $\alpha \geq 0$ 

$$
\sigma_ {\alpha C} (\mathbf {y}) = \alpha \sigma_ {C} (\mathbf {y}).
$$

(d) For any two nonempty sets $A , B \subseteq \mathbb { E }$ and $\mathbf { y } \in \mathbb { E } ^ { * }$ ， 

$$
\sigma_ {A + B} (\mathbf {y}) = \sigma_ {A} (\mathbf {y}) + \sigma_ {B} (\mathbf {y}).
$$

Proof. (a) $\begin{array} { r } { \sigma _ { C } ( \alpha \mathbf { y } ) = \operatorname* { m a x } _ { \mathbf { x } \in C } \langle \alpha \mathbf { y } , \mathbf { x } \rangle = \alpha \operatorname* { m a x } _ { \mathbf { x } \in C } \langle \mathbf { y } , \mathbf { x } \rangle = \alpha \sigma _ { C } ( \mathbf { y } ) . } \end{array}$ (b) 

$$
\begin{array}{l} \sigma_ {C} (\mathbf {y} _ {1} + \mathbf {y} _ {2}) = \underset {\mathbf {x} \in C} {\max} \langle \mathbf {y} _ {1} + \mathbf {y} _ {2}, \mathbf {x} \rangle = \underset {\mathbf {x} \in C} {\max} [ \langle \mathbf {y} _ {1}, \mathbf {x} \rangle + \langle \mathbf {y} _ {2}, \mathbf {x} \rangle ] \\ \qquad \leq \underset {\mathbf {x} \in C} {\max} \langle \mathbf {y} _ {1}, \mathbf {x} \rangle + \underset {\mathbf {x} \in C} {\max} \langle \mathbf {y} _ {2}, \mathbf {x} \rangle = \sigma_ {C} (\mathbf {y} _ {1}) + \sigma_ {C} (\mathbf {y} _ {2}). \end{array}
$$

(c) 

$$
\sigma_ {\alpha C} (\mathbf {y}) = \max _ {\mathbf {x} \in \alpha C} \langle \mathbf {y}, \mathbf {x} \rangle = \max _ {\mathbf {x} _ {1} \in C} \langle \mathbf {y}, \alpha \mathbf {x} _ {1} \rangle = \alpha \max _ {\mathbf {x} _ {1} \in C} \langle \mathbf {y}, \mathbf {x} _ {1} \rangle = \alpha \sigma_ {C} (\mathbf {y}).\tag{d}
$$

$$
\begin{array}{l} \sigma_ {A + B} (\mathbf {y}) = \max _ {\mathbf {x} \in A + B} \langle \mathbf {y}, \mathbf {x} \rangle = \max _ {\mathbf {x} _ {1} \in A, \mathbf {x} _ {2} \in B} \langle \mathbf {y}, \mathbf {x} _ {1} + \mathbf {x} _ {2} \rangle \\ = \max _ {\mathbf {x} _ {1} \in A, \mathbf {x} _ {2} \in B} [ \langle \mathbf {y}, \mathbf {x} _ {1} \rangle + \langle \mathbf {y}, \mathbf {x} _ {2} \rangle ] = \max _ {\mathbf {x} _ {1} \in A} \langle \mathbf {y}, \mathbf {x} _ {1} \rangle + \max _ {\mathbf {x} _ {2} \in B} \langle \mathbf {y}, \mathbf {x} _ {2} \rangle \\ = \sigma_ {A} (\mathbf {y}) + \sigma_ {B} (\mathbf {y}). \quad \square \end{array}
$$

Following are some basic examples of support functions. 

Example 2.25 (support functions of finite sets). Suppose that 

$$
C = \{\mathbf {b} _ {1}, \mathbf {b} _ {2}, \dots , \mathbf {b} _ {m} \},
$$

where $\mathbf { b } _ { 1 } , \mathbf { b } _ { 2 } , \ldots , \mathbf { b } _ { m } \in \mathbb { E }$ . Then 

$$
\sigma_ {C} (\mathbf {y}) = \max \{\langle \mathbf {b} _ {1}, \mathbf {y} \rangle , \langle \mathbf {b} _ {2}, \mathbf {y} \rangle , \dots , \langle \mathbf {b} _ {m}, \mathbf {y} \rangle \}.
$$

Recall that $S \subseteq \mathbb { E }$ is called a cone if it satisfies the following property: for any $\mathbf { x } \in S$ and $\lambda \geq 0$ , the inclusion $\lambda \mathbf { x } \in S$ holds. 

Example 2.26 (support functions of cones). Let $K \subseteq \mathbb { E }$ be a cone. Define the polar cone of K as 

$$
K ^ {\circ} = \left\{\mathbf {y} \in \mathbb {E} ^ {*}: \langle \mathbf {y}, \mathbf {x} \rangle \leq 0 \text {   for   all   } \mathbf {x} \in K \right\}.
$$

We will show that 

$$
\sigma_ {K} (\mathbf {y}) = \delta_ {K ^ {\circ}} (\mathbf {y}).\tag{2.14}
$$

Indeed, if $\mathbf { y } ~ \in ~ K ^ { \circ }$ , then $\langle \mathbf { y } , \mathbf { x } \rangle ~ \leq ~ 0$ for all $\textbf { x } \in \ K$ and for $\mathbf { x } = \mathbf { 0 } , \langle \mathbf { y } , \mathbf { x } \rangle = 0$ Therefore, 

$$
\sigma_ {K} (\mathbf {y}) = \max _ {\mathbf {x} \in K} \langle \mathbf {y}, \mathbf {x} \rangle = 0.
$$

If $\textbf { y } \notin \ K ^ { \circ }$ , then there exists $\tilde { \textbf { x } } \in \ K$ such that $\langle \mathbf { y } , \tilde { \mathbf { x } } \rangle > 0$ . Since $\lambda \tilde { \mathbf { x } } \in K$ for all $\lambda \geq 0$ , it follows that 

$$
\sigma_ {K} (\mathbf {y}) \geq \langle \mathbf {y}, \lambda \tilde {\mathbf {x}} \rangle = \lambda \langle \mathbf {y}, \tilde {\mathbf {x}} \rangle \text {   for   all   } \lambda \geq 0.
$$

Taking $\lambda \to \infty$ , we obtain that $\sigma _ { K } ( { \bf y } ) = \infty \mathrm { ~ f o r ~ } { \bf y } \notin K ^ { \circ }$ , and hence formula (2.14) is proven. ■ 

Example 2.27 (support function of the nonnegative orthant). Consider the space $\mathbb { E } = \mathbb { R } ^ { n }$ . As a special case of Example 2.26, since $( \mathbb { R } _ { + } ^ { n } ) ^ { \circ } = \bar { \mathbb { R } } _ { - } ^ { n }$ , it follows that 

$$
\sigma_ {\mathbb {R} _ {+} ^ {n}} (\mathbf {y}) = \delta_ {\mathbb {R} _ {-} ^ {n}} (\mathbf {y}).
$$

The next example uses Farkas’s lemma,<sup>6</sup> which we now recall. 

Lemma 2.28 (Farkas’s lemma—second formulation). Let $\mathbf { c } \in \mathbb { R } ^ { n }$ and $\mathbf { A \in }$ $\mathbb { R } ^ { m \times n }$ . Then the following two claims are equivalent: 

A. The implication $\mathbf { A } \mathbf { x } \leq \mathbf { 0 } \Rightarrow \mathbf { c } ^ { T } \mathbf { x } \leq 0$ holds true. 

B. There exists $\mathbf { y } \in \mathbb { R } _ { + } ^ { m }$ such that $\mathbf { A } ^ { T } \mathbf { y } = \mathbf { c } .$ 

Example 2.29 (support functions of convex polyhedral cones). Let the underlying space be $\mathbb { E } = \mathbb { R } ^ { n }$ and let $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ . Define the set 

$$
S = \{\mathbf {x} \in \mathbb {R} ^ {n}: \mathbf {A x} \leq \mathbf {0} \}.
$$

Since $S$ is a cone, we can use Example 2.26 to conclude that 

$$
\sigma_ {S} (\mathbf {y}) = \delta_ {S ^ {\circ}} (\mathbf {y}).
$$

Note that $\mathbf { y } \in S ^ { \circ }$ if and only if 

$$
\langle \mathbf {y}, \mathbf {x} \rangle \leq 0 \mathrm{foranyxsatisfying} \mathbf {A x} \leq \mathbf {0}.\tag{2.15}
$$

By Farkas’s lemma (Lemma 2.28), (2.15) is equivalent to the statement 

there exists $\pmb { \lambda } \in \mathbb { R } _ { + } ^ { m }$ such that $\mathbf { A } ^ { T } \lambda = \mathbf { y }$ 

Hence, 

$$
S ^ {\circ} = \left\{\mathbf {A} ^ {T} \boldsymbol {\lambda}: \boldsymbol {\lambda} \in \mathbb {R} _ {+} ^ {m} \right\}.
$$

To conclude, 

$$
\sigma_ {S} (\mathbf {y}) = \delta_ {\{\mathbf {A} ^ {T} \boldsymbol {\lambda}: \boldsymbol {\lambda} \in \mathbb {R} _ {+} ^ {m} \}} (\mathbf {y}).
$$

Example 2.30 (support functions of afine sets). Let the underlying space be $\mathbb { E } = \mathbb { R } ^ { n }$ and let $\dot { \mathbf { B } } \in \mathbb { R } ^ { m \times n }$ , $\mathbf { b } \in \mathbb { R } ^ { m }$ . Define the afine set 

$$
C = \{\mathbf {x} \in \mathbb {R} ^ {n}: \mathbf {B x} = \mathbf {b} \}.
$$

We assume that C is nonempty, namely, that there exists $\mathbf { x } _ { 0 } \in \mathbb { R } ^ { n }$ for which $\mathbf { B x } _ { \mathrm { 0 } } =$ b. The support function is obviously given by 

$$
\sigma_ {C} (\mathbf {y}) = \max _ {\mathbf {x}} \left\{\langle \mathbf {y}, \mathbf {x} \rangle : \mathbf {B x} = \mathbf {b} \right\}.
$$

Making the change of variables ${ \bf x } = { \bf z } + { \bf x } _ { 0 }$ , we obtain that the support function can be rewritten as 

$$
\begin{array}{l} \sigma_ {C} (\mathbf {y}) = \max _ {\mathbf {z}} \{\langle \mathbf {y}, \mathbf {z} \rangle + \langle \mathbf {y}, \mathbf {x} _ {0} \rangle : \mathbf {B z} = \mathbf {0} \} \\ \qquad = \langle \mathbf {y}, \mathbf {x} _ {0} \rangle + \max _ {\mathbf {z}} \{\langle \mathbf {y}, \mathbf {z} \rangle : \mathbf {B z} = \mathbf {0} \} \\ \qquad = \langle \mathbf {y}, \mathbf {x} _ {0} \rangle + \sigma_ {\tilde {C}} (\mathbf {y}), \end{array}\tag{2.16}
$$

where $\tilde { C } = \{ \mathbf { x } \in \mathbb { R } ^ { n } : \mathbf { B x } = \mathbf { 0 } \}$ . The set $\tilde { C }$ is a convex polyhedral cone that can be written as 

$$
\tilde {C} = \left\{\mathbf {x} \in \mathbb {R} ^ {n}: \mathbf {A x} \leq \mathbf {0} \right\},
$$

where $\mathbf { A } = \left( \begin{array} { c } { \mathbf { B } } \\ { - \mathbf { B } } \end{array} \right)$ . By Example 2.29, it follows that 

$$
\sigma_ {\tilde {C}} = \delta_ {\tilde {C} ^ {\circ}},\tag{2.17}
$$

where $\tilde { C } ^ { \circ }$ is the polar cone of ${ \tilde { C } } _ { ; }$ , which is given by 

$$
\tilde {C} ^ {\circ} = \left\{\mathbf {B} ^ {T} \boldsymbol {\lambda} _ {1} - \mathbf {B} ^ {T} \boldsymbol {\lambda} _ {2}: \boldsymbol {\lambda} _ {1}, \boldsymbol {\lambda} _ {2} \in \mathbb {R} _ {+} ^ {m} \right\}.
$$

We will show that 

$$
\tilde {C} ^ {\circ} = \operatorname{Range} (\mathbf {B} ^ {T}).\tag{2.18}
$$

Indeed, if $\mathbf { v } \in \tilde { C } ^ { \circ }$ , then there exists $\lambda _ { 1 } , \lambda _ { 2 } \in \mathbb { R } _ { + } ^ { m }$ for which $\mathbf { v } = \mathbf { B } ^ { T } \lambda _ { 1 } - \mathbf { B } ^ { T } \lambda _ { 2 } =$ $\mathbf { B } ^ { T } ( \lambda _ { 1 } - \lambda _ { 2 } ) \in \mathrm { R a n g e } ( \mathbf { B } ^ { T } )$ . In the other direction, if $\mathbf { v } \in \mathrm { R a n g e } ( \mathbf { B } ^ { T } )$ , then there exists $\boldsymbol { \lambda } \in \mathbb { R } ^ { m }$ for which $\dot { \mathbf { v } } = \mathbf { B } ^ { T } \boldsymbol { \lambda }$ . Defining $\lambda _ { 1 } = [ \lambda ] _ { + } , \lambda _ { 2 } = [ - \lambda ] _ { + }$ , we obtain that $\lambda = \lambda _ { 1 } - \lambda _ { 2 }$ with $\lambda _ { 1 } , \lambda _ { 2 } \in \mathbb { R } _ { + } ^ { m }$ , and hence 

$$
\mathbf {v} = \mathbf {B} ^ {T} \boldsymbol {\lambda} = \mathbf {B} ^ {T} (\boldsymbol {\lambda} _ {1} - \boldsymbol {\lambda} _ {2}) = \mathbf {B} ^ {T} \boldsymbol {\lambda} _ {1} - \mathbf {B} ^ {T} \boldsymbol {\lambda} _ {2} \in \tilde {C} ^ {\circ}.
$$

Combining (2.16), (2.17), and (2.18), we finally conclude that 

$$
\sigma_ {C} (\mathbf {y}) = \langle \mathbf {y}, \mathbf {x} _ {0} \rangle + \delta_ {\mathrm{Range} (\mathbf {B} ^ {T})} (\mathbf {y}).
$$

Example 2.31 (support functions of unit balls). Suppose that E is the underlying space endowed with a norm $\| \cdot \|$ . Consider the unit ball given by 

$$
B _ {\| \cdot \|} [ \mathbf {0}, 1 ] = \{\mathbf {x} \in \mathbb {E}: \| \mathbf {x} \| \leq 1 \}.
$$

By the definition of the dual norm, we have for any $\mathbf { y } \in \mathbb { E } ^ { * }$ 

$$
\sigma_ {B _ {\| \cdot \|} [ \mathbf {0}, 1 ]} (\mathbf {y}) = \max _ {\| \mathbf {x} \| \leq 1} \left\langle \mathbf {y}, \mathbf {x} \right\rangle = \| \mathbf {y} \| _ {*}.
$$

Thus, for example, for the space $\mathbb { R } ^ { n }$ we have 

$$
\begin{array}{l} \sigma_ {B _ {\| \cdot \| _ {p}} [ \mathbf {0}, 1 ]} (\mathbf {y}) = \| \mathbf {y} \| _ {q} \quad \left(1 \leq p \leq \infty , \frac {1}{p} + \frac {1}{q} = 1\right), \\ \sigma_ {B _ {\| \cdot \| _ {\mathbf {Q}}} [ \mathbf {0}, 1 ]} (\mathbf {y}) = \| \mathbf {y} \| _ {\mathbf {Q} ^ {- 1}} \quad (\mathbf {Q} \in \mathbb {S} _ {+ +} ^ {n}). \end{array}
$$

In the first formula we use the convention that $p = 1 / \infty$ corresponds to $q = \infty / 1$ 

The next example is also an example of a closed and convex function that is not continuous (recall that such an example does not exist for one-dimensional functions; see Theorem 2.22). 

Example 2.32.<sup>7</sup> Consider the following set in $\mathbb { R } ^ { 2 }$ : 

$$
C = \left\{\left(x _ {1}, x _ {2}\right) ^ {T}: x _ {1} + \frac {x _ {2} ^ {2}}{2} \leq 0 \right\}.
$$

Then the support function of $C$ is given by 

$$
\sigma_ {C} (\mathbf {y}) = \max _ {x _ {1}, x _ {2}} \left\{y _ {1} x _ {1} + y _ {2} x _ {2}: x _ {1} + \frac {x _ {2} ^ {2}}{2} \leq 0 \right\}.\tag{2.19}
$$

Obviously, $\sigma _ { C } ( \mathbf { 0 } ) = 0$ . We will compute the support function at $\mathbf y \neq \mathbf 0$ . In this case, it is easy to see that the maximum of problem (2.19) is attained at the boundary of $C . ^ { 8 }$ Therefore, 

$$
\sigma_ {C} (\mathbf {y}) = \max _ {x _ {1}, x _ {2}} \left\{y _ {1} x _ {1} + y _ {2} x _ {2}: x _ {1} + \frac {x _ {2} ^ {2}}{2} = 0 \right\} = \max _ {x _ {2}} \left\{- \frac {y _ {1}}{2} x _ {2} ^ {2} + y _ {2} x _ {2} \right\}.
$$

If $y _ { 1 } < 0$ , then the maximal value is $\infty$ . If $y _ { 1 } = 0$ and $y _ { 2 } \neq 0$ , then the maximal value is also $\infty$ . If $y _ { 1 } > 0$ , the maximum is attained at $\begin{array} { r } { x _ { 2 } = \frac { y _ { 2 } } { y _ { 1 } } } \end{array}$ , and the corresponding maximal value is $\frac { y _ { 2 } ^ { 2 } } { 2 y _ { 1 } }$ . Thus, the support function is given by 

$$
\sigma_ {C} (\mathbf {y}) = \left\{ \begin{array}{l l} \frac {y _ {2} ^ {2}}{2 y _ {1}}, & y _ {1} > 0, \\ 0, & y _ {1} = y _ {2} = 0, \\ \infty & \text { else }. \end{array} \right.
$$

By Lemma 2.23, $\sigma _ { C }$ is closed and convex. However, it is not continuous at $( y _ { 1 } , y _ { 2 } ) = ( 0 , 0 )$ . Indeed, taking for any $\alpha > 0$ the path $\begin{array} { r } { y _ { 1 } ( t ) = \frac { t ^ { 2 } } { 2 \alpha } , y _ { 2 } ( t ) = t ( t > 0 ) } \end{array}$ we obtain that 

$$
\sigma_ {C} (y _ {1} (t), y _ {2} (t)) = \alpha ,
$$

and hence the limit of $\sigma _ { C } ( y _ { 1 } ( t ) , y _ { 2 } ( t ) )$ as $t \to 0 ^ { + }$ is $\alpha .$ , which combined with the fact that $\sigma _ { C } ( 0 , 0 ) = 0$ implies the discontinuity of f at (0, 0). The contour lines of $\sigma _ { C }$ are plotted in Figure 2.3. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/06afcb51-43be-4db1-8b62-226fc4277ca3/3038d0d6a379f424a17b423669e918e22eb09b8580cf1e3e4f867f70606f480e.jpg)



Figure 2.3. Contour lines of the closed, convex, and noncontinuous function from Example 2.32.


An important property of support functions is that they are completely determined by their underlying sets as long as these sets are closed and convex. The proof of this result requires the strict separation theorem,<sup>9</sup> which is now recalled. 

Theorem 2.33 (strict separation theorem). Let $C \subseteq \mathbb { E }$ be a nonempty closed and convex set, and let ${ \bf y } \notin C$ . Then there exist $\mathbf { p } \in \mathbb { E } ^ { * } \backslash \{ \mathbf { 0 } \}$ and $\alpha \in \mathbb { R }$ such that 

$$
\langle \mathbf {p}, \mathbf {y} \rangle > \alpha
$$

and 

$$
\langle \mathbf {p}, \mathbf {x} \rangle \leq \alpha f o r a l l \mathbf {x} \in C.
$$

Lemma 2.34. Let $A , B \subseteq \mathbb { E }$ be nonempty closed and convex sets. Then $A = B$ if and only if $\sigma _ { A } = \sigma _ { B }$ 

Proof. If $A = B$ , then obviously $\sigma _ { A } = \sigma _ { B }$ . Suppose now that $\sigma _ { A } = \sigma _ { B }$ . We will prove that $A = B$ . Assume by contradiction that this is not the case, and without loss of generality suppose that there exists $\mathbf { y } \in A$ such that $\mathbf { y } \notin B$ . Since $\mathbf { y } \notin B$ and B is nonempty closed and convex, by the strict separation theorem, there exists a hyperplane separating y from B, meaning that there exists $\mathbf { p } \in \mathbb { E } ^ { * } \backslash \{ \mathbf { 0 } \}$ and $\alpha > 0$ such that 

$$
\langle \mathbf {p}, \mathbf {x} \rangle \leq \alpha <   \langle \mathbf {p}, \mathbf {y} \rangle \text {   for   any   } \mathbf {x} \in B.
$$

Taking the maximum over $\mathbf x \in B$ , we conclude that $\sigma _ { B } ( \mathbf { p } ) \leq \alpha < \langle \mathbf { p } , \mathbf { y } \rangle \leq \sigma _ { A } ( \mathbf { y } )$ 5 a contradiction to the assertion that the support functions are the same. 

A related result states that the support function stays the same under the operations of closure and convex hull of the underlying set. 

## Lemma 2.35. Let $A \subseteq \mathbb { E }$ be nonempty. Then

(a) $\sigma _ { A } = \sigma _ { \mathrm { c l } ( A ) } ;$ 

(b) $\sigma _ { A } = \sigma _ { \mathrm { c o n v } ( A ) } .$ 

Proof. (a) Since $A \subseteq \operatorname { c l } ( A )$ 

$$
\sigma_ {A} (\mathbf {y}) \leq \sigma_ {\operatorname{cl} (A)} (\mathbf {y}) \text {   for   any   } \mathbf {y} \in \mathbb {E} ^ {*}.\tag{2.20}
$$

We will show the reverse inequality. Let $\textbf { y } \in \mathbb { E } ^ { * }$ . Then by the definition of the support function, there exists a sequence $\{ \mathbf { x } ^ { k } \} _ { k \geq 1 } \subseteq \operatorname { c l } ( A )$ such that 

$$
\langle \mathbf {y}, \mathbf {x} ^ {k} \rangle \rightarrow \sigma_ {\mathrm{cl} (A)} (\mathbf {y}) \mathrm{as} k \rightarrow \infty .\tag{2.21}
$$

By the definition of the closure, it follows that there exists a sequence $\{ \mathbf { z } ^ { k } \} _ { k \geq 1 } \subseteq A$ such that $\begin{array} { r } { \| \mathbf { z } ^ { k } - \mathbf { x } ^ { k } \| \leq \frac { 1 } { k } } \end{array}$ for all k, and hence 

$$
\mathbf {z} ^ {k} - \mathbf {x} ^ {k} \rightarrow 0 \mathrm{as} k \rightarrow \infty .\tag{2.22}
$$

Now, since $\mathbf { z } ^ { k } \in A$ 

$$
\sigma_ {A} (\mathbf {y}) \geq \langle \mathbf {y}, \mathbf {z} ^ {k} \rangle = \langle \mathbf {y}, \mathbf {x} ^ {k} \rangle + \langle \mathbf {y}, \mathbf {z} ^ {k} - \mathbf {x} ^ {k} \rangle .
$$

Taking $k \to \infty$ and using (2.21), (2.22), we obtain that 

$$
\sigma_ {A} (\mathbf {y}) \geq \sigma_ {\mathrm{cl} (A)} (\mathbf {y}) + 0 = \sigma_ {\mathrm{cl} (A)} (\mathbf {y}),
$$

which combined with (2.20) yields the desired result $\sigma _ { A } = \sigma _ { \mathrm { c l } ( A ) }$ 

(b) Since $A \subseteq \mathrm { c o n v } ( A )$ , we have that $\sigma _ { A } ( \mathbf { y } ) \leq \sigma _ { \mathrm { c o n v } ( A ) } ( \mathbf { y } )$ for any $\mathbf { y } \in \mathbb { E } ^ { * }$ . We will show the reverse inequality. Let $\mathbf { y } \in \mathbb { E } ^ { * }$ . Then by the definition of the support function, there exists a sequence $\{ \mathbf { x } ^ { k } \} _ { k \geq 1 } \subseteq \mathrm { c o n v } ( A )$ such that 

$$
\langle \mathbf {y}, \mathbf {x} ^ {k} \rangle \rightarrow \sigma_ {\mathrm{conv} (A)} (\mathbf {y}) \text {as} k \rightarrow \infty .\tag{2.23}
$$

By the definition of the convex hull, it follows that for any $k ,$ , there exist vectors $\mathbf { z } _ { 1 } ^ { k } , \mathbf { z } _ { 2 } ^ { k } , \ldots , \mathbf { z } _ { n _ { k } } ^ { k } \in A ( n _ { k }$ is a positive integer) and $\lambda ^ { k } \in \Delta _ { n _ { k } }$ such that 

$$
\mathbf {x} ^ {k} = \sum_ {i = 1} ^ {n _ {k}} \lambda_ {i} ^ {k} \mathbf {z} _ {i} ^ {k}.
$$

Now, 

$$
\langle \mathbf {y}, \mathbf {x} ^ {k} \rangle = \left\langle \mathbf {y}, \sum_ {i = 1} ^ {n _ {k}} \lambda_ {i} ^ {k} \mathbf {z} _ {i} ^ {k} \right\rangle = \sum_ {i = 1} ^ {n _ {k}} \lambda_ {i} ^ {k} \langle \mathbf {y}, \mathbf {z} _ {i} ^ {k} \rangle \leq \sum_ {i = 1} ^ {n _ {k}} \lambda_ {i} ^ {k} \sigma_ {A} (\mathbf {y}) = \sigma_ {A} (\mathbf {y}),
$$

where the inequality follows by the fact that $\mathbf { z } _ { i } ^ { k } \in A$ . Taking the limit as $k  \infty$ and using (2.23), we obtain that $\sigma _ { \mathrm { c o n v } ( A ) } ( \mathbf { y } ) \leq \sigma _ { A } ( \mathbf { y } )$ 

Example 2.36 (support of the unit simplex). Suppose that the underlying space is $\mathbb { R } ^ { n }$ and consider the unit simplex set $\Delta _ { n } = \{ \mathbf { x } \in \mathbb { R } ^ { n } : \mathbf { e } ^ { T } \mathbf { x } = 1 , \mathbf { x } \geq \mathbf { 0 } \}$ Since the unit simplex can be written as the convex hull of the standard basis of $\mathbb { R } ^ { n }$ ， 

$$
\Delta_ {n} = \mathrm{conv} \{\mathbf {e} _ {1}, \mathbf {e} _ {2}, \ldots , \mathbf {e} _ {n} \},
$$

it follows by Lemma 2.35(b) that 

$$
\sigma_ {\Delta_ {n}} (\mathbf {y}) = \sigma_ {\{\mathbf {e} _ {1}, \dots , \mathbf {e} _ {n} \}} (\mathbf {y}) = \max \left\{\langle \mathbf {e} _ {1}, \mathbf {y} \rangle , \langle \mathbf {e} _ {2}, \mathbf {y} \rangle , \dots , \langle \mathbf {e} _ {n}, \mathbf {y} \rangle \right\}.
$$

Since we always assume (unless otherwise stated) that $\mathbb { R } ^ { n }$ is endowed with the dot product, the support function is 

$$
\sigma_ {\Delta_ {n}} (\mathbf {y}) = \max \{y _ {1}, y _ {2}, \dots , y _ {n} \}.
$$


The table below summarizes the main support function computations that were considered in this section.


<table><tr><td>C</td><td><eq>\sigma_C(\mathbf{y})</eq></td><td>Assumptions</td><td>Reference</td></tr><tr><td><eq>\{\mathbf{b}_1, \mathbf{b}_2, \dots, \mathbf{b}_n\}</eq></td><td><eq>\max_{i=1,2,\dots,n}\langle\mathbf{b}_i, \mathbf{y}\rangle</eq></td><td><eq>\mathbf{b}_i \in \mathbb{E}</eq></td><td>Example 2.25</td></tr><tr><td>K</td><td><eq>\delta_{K^o}(\mathbf{y})</eq></td><td>K—cone</td><td>Example 2.26</td></tr><tr><td><eq>\mathbb{R}_+^n</eq></td><td><eq>\delta_{\mathbb{R}_-^n}(\mathbf{y})</eq></td><td><eq>\mathbb{E} = \mathbb{R}^n</eq></td><td>Example 2.27</td></tr><tr><td><eq>\Delta_n</eq></td><td><eq>\max\{y_1, y_2, \dots, y_n\}</eq></td><td><eq>\mathbb{E} = \mathbb{R}^n</eq></td><td>Example 2.36</td></tr><tr><td><eq>\{\mathbf{x} \in \mathbb{R}^n : \mathbf{A}\mathbf{x} \leq \mathbf{0}\}</eq></td><td><eq>\delta_{\{\mathbf{A}^T \lambda: \lambda \in \mathbb{R}_+^m\}}(\mathbf{y})</eq></td><td><eq>\mathbb{E} = \mathbb{R}^n, \mathbf{A} \in \mathbb{R}^{m \times n}</eq></td><td>Example 2.29</td></tr><tr><td><eq>\{\mathbf{x} \in \mathbb{R}^n : \mathbf{B}\mathbf{x} = \mathbf{b}\}</eq></td><td><eq>\langle \mathbf{y}, \mathbf{x}_0 \rangle + \delta_{\text{Range}(\mathbf{B}^T)}(\mathbf{y})</eq></td><td><eq>\mathbb{E} = \mathbb{R}^n, \mathbf{B} \in \mathbb{R}^{m \times n}, \mathbf{b} \in \mathbb{R}^m, \mathbf{B}\mathbf{x}_0 = \mathbf{b}</eq></td><td>Example 2.30</td></tr><tr><td><eq>B_{\|\cdot\|}[0,1]</eq></td><td><eq>\| \mathbf{y}\|_*</eq></td><td>-</td><td>Example 2.31</td></tr></table>

# Chapter 3 Subgradients

## 3.1 Definitions and First Examples

Definition 3.1 (subgradient). Let $f : \mathbb { E }  ( - \infty , \infty ]$ be a proper function and let $\mathbf { x } \in \operatorname { d o m } ( f )$ . A vector $\mathbf { g } \in \mathbb { E } ^ { * }$ is called a subgradient of f at $\textbf { x } i f$ 

$$
f (\mathbf {y}) \geq f (\mathbf {x}) + \langle \mathbf {g}, \mathbf {y} - \mathbf {x} \rangle f o r a l l \mathbf {y} \in \mathbb {E}.\tag{3.1}
$$

Recall (see Section 1.11) that we use in this book the convention that the elements of $\mathbb { E } ^ { * }$ are exactly the elements of E, whereas the asterisk just marks the fact that the endowed norm on $\mathbb { E } ^ { * }$ is the dual norm $\| \cdot \| _ { : }$ rather than the endowed norm $\| \cdot \|$ on E. 

The inequality (3.1) is also called the subgradient inequality. It actually says that each subgradient is associated with an underestimate afine function, which is tangent to the surface of the function at x. Since the subgradient inequality (3.1) is trivial for $\mathbf { y } \not \in \operatorname { d o m } ( f )$ , it is frequently restricted to points in $\operatorname { d o m } ( f )$ and is thus written as 

$$
f (\mathbf {y}) \geq f (\mathbf {x}) + \langle \mathbf {g}, \mathbf {y} - \mathbf {x} \rangle \text {   for   all   } \mathbf {y} \in \operatorname{dom} (f).
$$

Given a point $\mathbf { x } \in \operatorname { d o m } ( f )$ , there might be more than one subgradient of f at x, and the set of all subgradients is called the subdiferential. 

Definition 3.2 (subdiferential). The set of all subgradients of f at $\mathbf { x }$ is called the subdiferential of f at x and is denoted by $\partial f ( \mathbf { x } )$ 

$$
\partial f (\mathbf {x}) \equiv \left\{\mathbf {g} \in \mathbb {E} ^ {*}: f (\mathbf {y}) \geq f (\mathbf {x}) + \langle \mathbf {g}, \mathbf {y} - \mathbf {x} \rangle \text {   for   all   } \mathbf {y} \in \mathbb {E} \right\}.
$$

When $\mathbf { x } \not \in \operatorname { d o m } ( f )$ , we define $\partial f ( \mathbf { x } ) = \varnothing$ . Actually, for proper functions, this is a direct consequence of the definition of the subdiferential set since the subgradient inequality (3.1) does not hold for ${ \textbf { x } } { \notin }$ domf and $\textbf { y } \in$ domf. 

Example 3.3 (subdiferential of norms at 0). Let $f : \mathbb { E }  \mathbb { R }$ be given by $f ( \mathbf { x } ) = \| \mathbf { x } \|$ , where $\| \cdot \|$ is the endowed norm on E. We will show that the subdiferential of $f$ at $\mathbf { x } = \mathbf { 0 }$ is the dual norm unit ball: 

$$
\partial f (\mathbf {0}) = B _ {\| \cdot \| _ {*}} [ \mathbf {0}, 1 ] = \{\mathbf {g} \in \mathbb {E} ^ {*}: \| \mathbf {g} \| _ {*} \leq 1 \}.\tag{3.2}
$$

To show (3.2), note that $\mathbf { g } \in \partial f ( \mathbf { 0 } )$ if and only if 

$$
f (\mathbf {y}) \geq f (\mathbf {0}) + \langle \mathbf {g}, \mathbf {y} - \mathbf {0} \rangle \text {   for   all   } \mathbf {y} \in \mathbb {E},
$$

which is the same as 

$$
\| \mathbf {y} \| \geq \langle \mathbf {g}, \mathbf {y} \rangle \text {   for   all   } \mathbf {y} \in \mathbb {E}.\tag{3.3}
$$

We will prove that the latter holds true if and only if $\| \mathbf { g } \| _ { * } \leq 1$ . Indeed, if $\| \mathbf { g } \| _ { * } \leq 1$ , then by the generalized Cauchy–Schwarz inequality (Lemma 1.4), 

$$
\langle \mathbf {g}, \mathbf {y} \rangle \leq \| \mathbf {g} \| _ {*} \| \mathbf {y} \| \leq \| \mathbf {y} \| \text {   for   any   } \mathbf {y} \in \mathbb {E},
$$

implying (3.3). In the reverse direction, assume that (3.3) holds. Taking the maximum of both sides of (3.3) over all y satisfying $\| \mathbf { y } \| \leq 1$ , we get 

$$
\| \mathbf {g} \| _ {*} = \max _ {\mathbf {y}: \| \mathbf {y} \| \leq 1} \langle \mathbf {g}, \mathbf {y} \rangle \leq \max _ {\mathbf {y}: \| \mathbf {y} \| \leq 1} \| \mathbf {y} \| = 1.
$$

We have thus established the equivalence between (3.3) and the inequality $\| \mathbf { g } \| _ { * } \leq 1$ which is the same as the result (3.2). 

Example 3.4 (subdiferential of the $l _ { 1 } { \bf - n o r m }$ at $\mathbf { 0 } )$ . Let $f : \mathbb { R } ^ { n } $ R be given by $f ( \mathbf { x } ) = \| \mathbf { x } \| _ { 1 }$ . Then, since this is a special case of Example 3.3 with $\| \cdot \| = \| \cdot \| _ { 1 }$ and since the $l _ { \infty } \mathrm { - n o r m }$ is the dual of the $l _ { 1 } { \mathrm { - n o r m } } .$ , it follows that 

$$
\partial f (0) = B _ {\| \cdot \| _ {\infty}} [ \mathbf {0}, 1 ] = [ - 1, 1 ] ^ {n}.
$$

In particular, when $n = 1$ , then $f ( x ) = \left| x \right|$ , and we have 

$$
\partial f (0) = [ - 1, 1 ].
$$

The linear underestimators that correspond to $- 0 . 8 , - 0 . 3$ , and $0 . 7 \in \partial f ( 0 )$ , meaning $- 0 . 8 x , - 0 . 3 x$ , and 0.7x, are described in Figure 3.1. 

For the next example, we need the definition of the normal cone. Given a set $S \subseteq \mathbb { E }$ and a point $\mathbf { x } \in S$ , the normal cone of $S$ at x is defined as 

$$
N _ {S} (\mathbf {x}) = \{\mathbf {y} \in \mathbb {E} ^ {*}: \langle \mathbf {y}, \mathbf {z} - \mathbf {x} \rangle \leq 0 \text { for   any } \mathbf {z} \in S \}.
$$

The normal cone, in addition to being a cone, is closed and convex as an intersection of half-spaces. When $\mathbf { x } \not \in S$ , we define $N _ { S } ( { \bf x } ) = \emptyset$ 

Example 3.5 (subdiferential of indicator functions). Suppose that $S \subseteq \mathbb { E }$ is nonempty and consider the indicator function $\delta _ { S }$ . Then for any $\mathbf { x } \in S$ , we have that $\mathbf { y } \in \partial \delta _ { S } ( \mathbf { x } )$ if and only if 

$$
\delta_ {S} (\mathbf {z}) \geq \delta_ {S} (\mathbf {x}) + \langle \mathbf {y}, \mathbf {z} - \mathbf {x} \rangle \mathrm{forall} \mathbf {z} \in S,
$$

which is the same as 

$$
\langle \mathbf {y}, \mathbf {z} - \mathbf {x} \rangle \leq 0 \text {   for   all   } \mathbf {z} \in S.
$$

Therefore, we have that 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/06afcb51-43be-4db1-8b62-226fc4277ca3/0cf18c5ae2c67e7746e8ae4613ee168d5f0ed0d3da97184d8440555ee03d1ccc.jpg)



Figure 3.1. The linear underestimators $o f \left| x \right|$ corresponding to $- 0 . 8 , - 0 . 3$ $0 . 7 \in \partial f ( 0 )$ ; see Example 3.4.


$$
\partial \delta_ {S} (\mathbf {x}) = N _ {S} (\mathbf {x}) \text {   for   all   } \mathbf {x} \in S.\tag{3.4}
$$

For ${ \bf x } \notin S , \partial \delta _ { S } ( { \bf x } ) = N _ { S } ( { \bf x } ) = \emptyset$ by convention, so we obtain that (3.4) holds also for $\mathbf { x } \not \in S$ 

Example 3.6 (subdiferential of the indicator function of the unit ball). As a special case of Example 3.5, let 

$$
S = B [ \mathbf {0}, 1 ] = \{\mathbf {x} \in \mathbb {E}: \| \mathbf {x} \| \leq 1 \}.
$$

Then $\partial \delta _ { S } ( { \bf x } ) = N _ { S } ( { \bf x } )$ , where $N _ { S } ( { \bf x } )$ is given by 

$$
N _ {S} (\mathbf {x}) = \left\{\mathbf {y} \in \mathbb {E} ^ {*}: \langle \mathbf {y}, \mathbf {z} - \mathbf {x} \rangle \leq 0 \text {   for   all   } \mathbf {z} \in S \right\}.
$$

We will find a more explicit representation for $N _ { S } ( { \bf x } ) . \mathrm { ~ I f ~ } { \bf x } \notin S$ , then $N _ { S } ( { \bf x } ) = \emptyset$ Suppose that $\| \mathbf { x } \| \leq 1$ . A vector $\mathbf { y } \in \mathbb { E } ^ { * }$ satisfies $\mathbf { y } \in N _ { S } ( \mathbf { x } )$ if and only if 

$$
\langle \mathbf {y}, \mathbf {z} - \mathbf {x} \rangle \leq 0 \text {   for   any   } \mathbf {z} \text {   satisfying   } \| \mathbf {z} \| \leq 1,
$$

which is the same as the inequality, 

$$
\max _ {\mathbf {z}: \| \mathbf {z} \| \leq 1} \langle \mathbf {y}, \mathbf {z} \rangle \leq \langle \mathbf {y}, \mathbf {x} \rangle .
$$

Using the definition of the dual norm, we obtain that the latter can be rewritten as 

$$
\left\| \mathbf {y} \right\| _ {*} \leq \langle \mathbf {y}, \mathbf {x} \rangle .
$$

Therefore, 

$$
\partial \delta_ {B [ \mathbf {0}, 1 ]} (\mathbf {x}) = N _ {B [ \mathbf {0}, 1 ]} (\mathbf {x}) = \left\{ \begin{array}{l l} \{\mathbf {y} \in \mathbb {E} ^ {*}: \| \mathbf {y} \| _ {*} \leq \langle \mathbf {y}, \mathbf {x} \rangle \}, & \| \mathbf {x} \| \leq 1, \\ \emptyset , & \| \mathbf {x} \| > 1. \end{array} \right.
$$

Example 3.7 (subgradient of the dual function). Consider the minimization problem 

$$
\min \{f (\mathbf {x}): \mathbf {g} (\mathbf {x}) \leq \mathbf {0}, \mathbf {x} \in X \},\tag{3.5}
$$

where $\varnothing \neq X \subseteq \mathbb { E } , f : \mathbb { E } \to \mathbb { R }$ and $\mathbf { g } : \mathbb { E }  \mathbb { R } ^ { m }$ is a vector-valued function. The Lagrangian dual objective function is given by 

$$
q (\pmb {\lambda}) = \min _ {\mathbf {x} \in X} \left\{L (\mathbf {x}; \pmb {\lambda}) \equiv f (\mathbf {x}) + \pmb {\lambda} ^ {T} \mathbf {g} (\mathbf {x}) \right\}.
$$

The dual problem consists of maximizing q on its efective domain, which is given by 

$$
\operatorname{dom} (- q) = \{\boldsymbol {\lambda} \in \mathbb {R} _ {+} ^ {m}: q (\boldsymbol {\lambda}) > - \infty \}.
$$

No matter whether the primal problem (3.5) is convex or not, the dual problem 

$$
\max _ {\boldsymbol {\lambda} \in \mathbb {R} ^ {m}} \left\{q (\boldsymbol {\lambda}): \boldsymbol {\lambda} \in \operatorname{dom} (- q) \right\}
$$

is always convex, meaning that $q$ is a concave function and $\mathrm { d o m } ( - q )$ is a convex set. Let $\lambda _ { 0 } \in \operatorname { d o m } ( - q )$ and assume that the minimum in the minimization problem defining $q ( \lambda _ { 0 } )$ ， 

$$
q (\boldsymbol {\lambda} _ {0}) = \min _ {\mathbf {x} \in X} \left\{f (\mathbf {x}) + \boldsymbol {\lambda} _ {0} ^ {T} \mathbf {g} (\mathbf {x}) \right\},
$$

is attained at $\mathbf { x } _ { 0 } \in X$ , that is, 

$$
L (\mathbf {x} _ {0}; \pmb {\lambda} _ {0}) = f (\mathbf {x} _ {0}) + \pmb {\lambda} _ {0} ^ {T} \mathbf {g} (\mathbf {x} _ {0}) = q (\pmb {\lambda} _ {0}).
$$

We seek to find a subgradient of the convex function $- q$ at $\lambda _ { 0 }$ . For that, note that for any $\lambda \in \operatorname { d o m } ( - q )$ ， 

$$
\begin{array}{l} q (\boldsymbol {\lambda}) = \min _ {\mathbf {x} \in X} \Big \{f (\mathbf {x}) + \boldsymbol {\lambda} ^ {T} \mathbf {g} (\mathbf {x}) \Big \} \\ \leq f (\mathbf {x} _ {0}) + \boldsymbol {\lambda} ^ {T} \mathbf {g} (\mathbf {x} _ {0}) \\ = f (\mathbf {x} _ {0}) + \boldsymbol {\lambda} _ {0} ^ {T} \mathbf {g} (\mathbf {x} _ {0}) + (\boldsymbol {\lambda} - \boldsymbol {\lambda} _ {0}) ^ {T} \mathbf {g} (\mathbf {x} _ {0}) \\ = q (\boldsymbol {\lambda} _ {0}) + \mathbf {g} (\mathbf {x} _ {0}) ^ {T} (\boldsymbol {\lambda} - \boldsymbol {\lambda} _ {0}). \end{array}
$$

Thus, 

$$
- q (\boldsymbol {\lambda}) \geq - q \left(\boldsymbol {\lambda} _ {0}\right) + \left(- \mathbf {g} \left(\mathbf {x} _ {0}\right)\right) ^ {T} \left(\boldsymbol {\lambda} - \boldsymbol {\lambda} _ {0}\right) \text {   for   any   } \boldsymbol {\lambda} \in \operatorname{dom} (- q),
$$

concluding that 

$$
- \mathbf {g} (\mathbf {x} _ {0}) \in \partial (- q) (\boldsymbol {\lambda} _ {0}).
$$

Example 3.8 (subgradient of the maximum eigenvalue function). Consider the function $f : \mathbb { S } ^ { n }  \mathbb { R }$ given by $f ( \mathbf { X } ) = \lambda _ { \operatorname* { m a x } } ( \mathbf { X } )$ (recall that $\mathbb { S } ^ { n }$ is the set of all $n \times n$ symmetric matrices). Let $\mathbf { X } \in \mathbb { S } ^ { n }$ and let v be a normalized eigenvector of $\mathbf { X } \left( \| \mathbf { v } \| _ { 2 } = 1 \right)$ associated with the maximum eigenvalue of X. We will establish the relation 

$$
\mathbf {v v} ^ {T} \in \partial f (\mathbf {X}).\tag{3.6}
$$

To show this, note that for any $\mathbf { Y } \in \mathbb { S } ^ { n }$ 

$$
\begin{array}{r l} & {\lambda_ {\max} (\mathbf {Y}) = \underset {\mathbf {u}} {\max} \{\mathbf {u} ^ {T} \mathbf {Y} \mathbf {u}: \| \mathbf {u} \| _ {2} = 1 \}} \\ & {\qquad \geq \mathbf {v} ^ {T} \mathbf {Y} \mathbf {v}} \\ & {\qquad = \mathbf {v} ^ {T} \mathbf {X} \mathbf {v} + \mathbf {v} ^ {T} (\mathbf {Y} - \mathbf {X}) \mathbf {v}} \\ & {\qquad = \lambda_ {\max} (\mathbf {X}) \| \mathbf {v} \| _ {2} ^ {2} + \mathrm{Tr} (\mathbf {v} ^ {T} (\mathbf {Y} - \mathbf {X}) \mathbf {v})} \\ & {\qquad = \lambda_ {\max} (\mathbf {X}) + \mathrm{Tr} (\mathbf {v v} ^ {T} (\mathbf {Y} - \mathbf {X}))} \\ & {\qquad = \lambda_ {\max} (\mathbf {X}) + \langle \mathbf {v v} ^ {T}, \mathbf {Y} - \mathbf {X} \rangle ,} \end{array}
$$

establishing (3.6). 

There is an intrinsic diference between the results in Examples 3.7 and 3.8 and the results in Examples 3.3, 3.4, 3.5, and 3.6. Only one subgradient is computed in Examples 3.7 and 3.8; such results are referred to as weak results. On the other hand, in Examples 3.3, 3.4, 3.5, and 3.6 the entire subdiferential set is characterized—such results are called strong results. 

## 3.2 Properties of the Subdiferential Set

Note that the subdiferential sets computed in the previous section are all closed and convex. This is not a coincidence. Subdiferential sets are always closed and convex. 

Theorem 3.9 (closedness and convexity of the subdiferential set). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper function. Then the set $\partial f ( \mathbf { x } )$ is closed and convex for any $\mathbf { x } \in \mathbb { E }$ 

Proof. For any $\mathbf { x } \in \mathbb { E }$ , the subdiferential set can be represented as 

$$
\partial f (\mathbf {x}) = \bigcap_ {\mathbf {y} \in \mathbb {E}} H _ {\mathbf {y}},
$$

where $H _ { \mathbf { y } } = \{ \mathbf { g } \in \mathbb { E } ^ { * } : f ( \mathbf { y } ) \geq f ( \mathbf { x } ) + \langle \mathbf { g } , \mathbf { y } - \mathbf { x } \rangle \}$ . Since the sets $H _ { \mathbf { y } }$ are half-spaces and, in particular, closed and convex, it follows that $\partial f ( \mathbf { x } )$ is closed and convex. 

The subdiferential set $\partial f ( \mathbf { x } )$ may be empty. When it is nonempty at a given $\mathbf { x } \in \mathbb { E }$ , the function $f$ is called subdiferentiable at x. 

Definition 3.10 (subdiferentiability). A proper function $f : \mathbb { E } \to ( - \infty , \infty ]$ is called subdiferentiable at $\mathbf { x } \in \operatorname { d o m } ( f ) \ i f \partial f ( \mathbf { x } ) \neq \emptyset$ 

The collection of points of subdiferentiability is denoted by dom $( \partial f )$ : 

$$
\operatorname{dom} (\partial f) = \left\{\mathbf {x} \in \mathbb {E}: \partial f (\mathbf {x}) \neq \emptyset \right\}.
$$

We will now show that if a function is subdiferentiable at any point in its domain, which is assumed to be convex, then it is necessarily convex. 

Lemma 3.11 (nonemptiness of subdiferential sets $\mathbf { \tau } \Rightarrow \mathbf { c o n v e x i t y } )$ . Let $f$ : $\mathbb { E }  ( - \infty , \infty ]$ be a proper function and assume that dom $( f )$ is convex. Suppose that for any $\mathbf { x } \in \operatorname { d o m } ( f )$ , the set $\partial f ( \mathbf { x } )$ is nonempty. Then f is convex. 

Proof. Let $\mathbf { c } , \mathbf { y } \in \mathrm { d o m } ( f )$ and $\alpha \in [ 0 , 1 ]$ . Define ${ \bf z } _ { \alpha } = ( 1 - \alpha ) { \bf x } + \alpha { \bf y }$ . By the convexity of dom $( f ) , \mathbf { z } _ { \alpha } \in \mathrm { d o m } ( f )$ , and hence there exists $\mathbf { g } \in \partial f ( \mathbf { z } _ { \alpha } )$ , which in particular implies the following two inequalities: 

$$
\begin{array}{l} f (\mathbf {y}) \geq f (\mathbf {z} _ {\alpha}) + \langle \mathbf {g}, \mathbf {y} - \mathbf {z} _ {\alpha} \rangle = f (\mathbf {z} _ {\alpha}) + (1 - \alpha) \langle \mathbf {g}, \mathbf {y} - \mathbf {x} \rangle , \\ f (\mathbf {x}) \geq f (\mathbf {z} _ {\alpha}) + \langle \mathbf {g}, \mathbf {x} - \mathbf {z} _ {\alpha} \rangle = f (\mathbf {z} _ {\alpha}) - \alpha \langle \mathbf {g}, \mathbf {y} - \mathbf {x} \rangle . \end{array}
$$

Multiplying the first inequality by $\alpha ,$ the second by $1 - \alpha$ , and summing them yields 

$$
f ((1 - \alpha) \mathbf {x} + \alpha \mathbf {y}) = f (\mathbf {z} _ {\alpha}) \leq (1 - \alpha) f (\mathbf {x}) + \alpha f (\mathbf {y}).
$$

Since the latter holds for any $\mathbf { x } , \mathbf { y } \in \operatorname { d o m } ( f )$ with $\operatorname { d o m } ( f )$ being convex, it followsthat the function f is convex. 口

We have thus shown that if a function is subdiferentiable at any point in its (assumed to be) convex domain, then it is convex. However, this does not mean that the reverse direction is correct. The next example describes a convex function, which is not subdiferentiable at one of the points in its domain. 

Example 3.12. Consider the convex function $f : \mathbb { R } \to ( - \infty , \infty ]$ defined by 

$$
f (x) = \left\{ \begin{array}{l l} - \sqrt {x}, & x \geq 0, \\ \infty , & \text { else }. \end{array} \right.
$$

The function is plotted in Figure 3.2. It is not subdiferentiable at $x = 0$ . To show this, suppose by contradiction that there exists $g \in \mathbb { R }$ such that $g \in \partial f ( 0 )$ . Then 

$$
f (y) \geq f (0) + g (y - 0) \mathrm{forany} y \geq 0,
$$

which is the same as 

$$
- \sqrt {y} \geq g y \text {   for   any   } y \geq 0.\tag{3.7}
$$

The above is impossible since substituting $y = 1$ , we obtain that $g \le - 1$ (and in particular $g < 0 )$ , while substituting $\begin{array} { r } { y = \frac { 1 } { 2 g ^ { 2 } } } \end{array}$ in (3.7) yields the inequality 

$$
- \sqrt {1 / (2 g ^ {2})} \geq 1 / (2 g),
$$

$$
\frac {1}{2 g ^ {2}} \leq \frac {1}{4 g ^ {2}}. \quad \blacksquare
$$

which is equivalent to the impossible inequality (utilizing the fact that $g < 0 )$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/06afcb51-43be-4db1-8b62-226fc4277ca3/7c960971afbc696c75e6a447c2f2d500b61780da47304e40519e2af49c0c1594.jpg)



Figure 3.2. The function $f ( x ) = - { \sqrt { x } }$ with dom $( f ) = \mathbb { R } _ { + }$ . The function is not subdiferentiable at $x = 0$


Although, as demonstrated in Example 3.12, convex functions are not necessarily subdiferentiable at any point in their domain, they must be subdiferentiable at any point in the interior of their domain. This is stated in Theorem 3.14 below, which also shows the boundedness of the subdiferential set in this setting. The proof of the theorem strongly relies on the supporting hyperplane theorem stated explicitly below. 

Theorem 3.13 (supporting hyperplane theorem [29, Proposition $\mathbf { 2 . 4 . 1 } ]$ . Let $\varnothing \neq C \subseteq \mathbb { E }$ be a convex set, and let y ∈/ int(C). Then there exists $\mathbf { 0 } \neq \mathbf { p } \in \mathbb { E } ^ { * }$ such that 

$$
\langle \mathbf {p}, \mathbf {x} \rangle \leq \langle \mathbf {p}, \mathbf {y} \rangle f o r a n y \mathbf {x} \in C.
$$

Theorem 3.14 (nonemptiness and boundedness of the subdiferential set at interior points of the domain). Let $f : \mathbb { E }  ( - \infty , \infty ]$ be a proper convex function, and assume that $\tilde { \mathbf { x } } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ . Then $\partial f ( { \widetilde { \mathbf { x } } } )$ is nonempty and bounded. 

Proof. Recall that the inner product in the product space $\mathbb { E } \times \mathbb { R }$ is defined as (see Section 1.9) 

$$
\langle (\mathbf {y} _ {1}, \beta_ {1}), (\mathbf {y} _ {2}, \beta_ {2}) \rangle \equiv \langle \mathbf {y} _ {1}, \mathbf {y} _ {2} \rangle + \beta_ {1} \beta_ {2}, \quad (\mathbf {y} _ {1}, \beta_ {1}), (\mathbf {y} _ {2}, \beta_ {2}) \in \mathbb {E} \times \mathbb {R}.
$$

Since $( \tilde { \mathbf { x } } , f ( \tilde { \mathbf { x } } ) )$ is on the boundary of epi $( f ) \subseteq \mathbb { E } \times \mathbb { R }$ , it follows by the supporting hyperplane theorem (Theorem 3.13) that there exists a separating hyperplane between $( \tilde { \mathbf { x } } , f ( \tilde { \mathbf { x } } ) )$ ) and $\operatorname { e p i } ( f )$ , meaning that there exists a nonzero vector $( \mathbf { p } , - \alpha ) \in \mathbb { E } ^ { * } \times \mathbb { R }$ for which 

$$
\langle \mathbf {p}, \tilde {\mathbf {x}} \rangle - \alpha f (\tilde {\mathbf {x}}) \geq \langle \mathbf {p}, \mathbf {x} \rangle - \alpha t \text {   for   any   } (\mathbf {x}, t) \in \operatorname{epi} (f).\tag{3.8}
$$

Note that $\alpha \geq 0$ since $( \tilde { \mathbf { x } } , f ( \tilde { \mathbf { x } } ) + 1 ) \in \mathrm { e p i } ( f )$ , and hence plugging $\textbf { x } = \tilde { \textbf { x } }$ and $t = f ( \tilde { \mathbf { x } } ) + 1$ into (3.8) yields 

$$
\langle \mathbf {p}, \tilde {\mathbf {x}} \rangle - \alpha f (\tilde {\mathbf {x}}) \geq \langle \mathbf {p}, \tilde {\mathbf {x}} \rangle - \alpha (f (\tilde {\mathbf {x}}) + 1),
$$

implying that $\alpha \geq 0$ . Since $\tilde { \mathbf { x } } \in \mathrm { i n t } ( \mathrm { d o m } ( f ) )$ , it follows by the local Lipschitz continuity property of convex functions (Theorem 2.21) that there exist $\varepsilon > 0$ and $L > 0$ such that $B _ { \| \cdot \| } [ \tilde { \mathbf { x } } , \varepsilon ] \subseteq \mathrm { d o m } ( f )$ and 

$$
| f (\mathbf {x}) - f (\tilde {\mathbf {x}}) | \leq L \| \mathbf {x} - \tilde {\mathbf {x}} \| \text {   for   any   } \mathbf {x} \in B _ {\|. \|} [ \tilde {\mathbf {x}}, \varepsilon ].\tag{3.9}
$$

Since $B _ { \| \cdot \| } [ \tilde { \mathbf { x } } , \varepsilon ] \subseteq \mathrm { d o m } ( f )$ , it follows that $( \mathbf { x } , f ( \mathbf { x } ) ) \in \mathrm { e p i } ( f )$ for any $\mathbf { x } \in B _ { \parallel \cdot \parallel } [ \tilde { \mathbf { x } } , \varepsilon ]$ Therefore, plugging $t = f ( \mathbf { x } )$ into (3.8), yields that 

$$
\langle \mathbf {p}, \mathbf {x} - \tilde {\mathbf {x}} \rangle \leq \alpha (f (\mathbf {x}) - f (\tilde {\mathbf {x}})) \text {for any} \mathbf {x} \in B _ {\| \cdot \|} [ \tilde {\mathbf {x}}, \varepsilon ].\tag{3.10}
$$

Combining (3.9) and (3.10), we obtain that for any $\mathbf { x } \in B _ { \parallel \cdot \parallel } [ \tilde { \mathbf { x } } , \varepsilon ]$ ， 

$$
\langle \mathbf {p}, \mathbf {x} - \tilde {\mathbf {x}} \rangle \leq \alpha (f (\mathbf {x}) - f (\tilde {\mathbf {x}})) \leq \alpha L \| \mathbf {x} - \tilde {\mathbf {x}} \|.\tag{3.11}
$$

Take $\mathbf { p } ^ { \dagger } \in \mathbb { E }$ satisfying $\langle \mathbf { p } , \mathbf { p } ^ { \dagger } \rangle = \| \mathbf { p } \|$ <sub>∗</sub> and $\| \mathbf { p } ^ { \dagger } \| = 1$ . Since $\tilde { \mathbf { x } } + \varepsilon \mathbf { p } ^ { \dagger } \in B _ { \parallel \cdot \parallel } [ \tilde { \mathbf { x } } , \varepsilon ]$ , we can plug $\mathbf { x } = \tilde { \mathbf { x } } + \varepsilon \mathbf { p } ^ { \dagger }$ into (3.11) and obtain that 

$$
\varepsilon \| \mathbf {p} \| _ {*} = \varepsilon \langle \mathbf {p}, \mathbf {p} ^ {\dagger} \rangle \leq \alpha L \varepsilon \| \mathbf {p} ^ {\dagger} \| = \alpha L \varepsilon .
$$

Therefore, $\alpha > 0$ , since otherwise we would have $\alpha = 0$ and $\mathrm { ~ \bf ~ p ~ } = \mathrm { ~ \bf ~ 0 ~ }$ , which is impossible by the fact that the vector $( \mathbf { p } , \alpha )$ is not the zeros vector. Taking $t = f ( \mathbf { x } )$ in (3.8) and dividing by α yields 

$$
f (\mathbf {x}) \geq f (\tilde {\mathbf {x}}) + \langle \mathbf {g}, \mathbf {x} - \tilde {\mathbf {x}} \rangle \mathrm{forall} \mathbf {x} \in \mathrm{dom} (f),\tag{3.12}
$$

where $\mathbf { g } = \mathbf { p } / \alpha$ . Thus, $\mathbf { g } \in \partial f ( { \widetilde { \mathbf { x } } } )$ , establishing the nonemptiness of $\partial f ( { \widetilde { \mathbf { x } } } )$ . To show the boundedness of $\partial f ( { \widetilde { \mathbf { x } } } )$ , let $\mathbf { g } \in \partial f ( \tilde { \mathbf { x } } )$ , meaning that (3.12) holds. Take $\mathbf { g } ^ { \dagger } \in \mathbb { E }$ for which $\| \mathbf { g } \| _ { * } = \langle \mathbf { g } , \mathbf { g } ^ { \dag } \rangle$ and $\| \mathbf { g } ^ { \dagger } \| = 1$ . Then plugging $\mathbf { x } = \tilde { \mathbf { x } } + \varepsilon \mathbf { g } ^ { \dagger }$ in (3.12) yields 

$$
\varepsilon \| \mathbf {g} \| _ {*} = \varepsilon \langle \mathbf {g}, \mathbf {g} ^ {\dagger} \rangle = \langle \mathbf {g}, \mathbf {x} - \tilde {\mathbf {x}} \rangle \leq f (\mathbf {x}) - f (\tilde {\mathbf {x}}) \stackrel {(3. 9)} {\leq} L \| \mathbf {x} - \tilde {\mathbf {x}} \| = L \varepsilon ,
$$

showing that $\partial f ( \tilde { \mathbf { x } } ) ~ \subseteq ~ B _ { \| \cdot \| _ { * } } [ \mathbf { 0 } , L ]$ , and hence establishing the boundedness of $\partial f ( { \widetilde { \mathbf { x } } } )$ 

The result of Theorem 3.14 can be stated as the following inclusion relation: 

$$
\operatorname{int} (\operatorname{dom} (f)) \subseteq \operatorname{dom} (\partial f).
$$

A direct consequence of Theorem 3.14 is that real-valued convex functions (namely, convex functions $f$ with dom $( f ) = \mathbb { E } )$ are subdiferentiable at any point. 

Corollary 3.15 (subdiferentiability of real-valued convex functions). Let $f : \mathbb { E } \to \mathbb { R }$ be a convex function. Then f is subdiferentiable over E. 

We can extend the boundedness result of Theorem 3.14 and show that subgradients of points in a given compact set contained in the interior of the domain are always bounded. 

Theorem 3.16 (boundedness of subgradients over compact sets). Let f : $\mathbb { E }  ( - \infty , \infty ]$ be a proper convex function, and assume that $X \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ is nonempty and compact. Then $\textstyle Y = \bigcup _ { \mathbf { x } \in X } \partial f ( \mathbf { x } )$ is nonempty and bounded. 

Proof. The set Y is nonempty, since by Theorem 3.14 $\partial f ( \mathbf { x } ) \neq \emptyset$ for any $\mathbf { x } \in X$ To prove the boundedness, assume by contradiction that there exists a sequence $\{ \mathbf { x } _ { k } \} _ { k \geq 1 } \subseteq X$ and $\mathbf { g } _ { k } \in \partial f ( \mathbf { x } _ { k } )$ such that $\left\| \mathbf { g } _ { k } \right\| _ { * } \to \infty$ as $k  \infty$ . For any $k ,$ let $\mathbf { g } _ { k } ^ { \dagger }$ be a vector satisfying $\begin{array} { r } { \langle \mathbf { g } _ { k } , \mathbf { g } _ { k } ^ { \dagger } \rangle = \| \mathbf { g } _ { k } \| , } \end{array}$ and $\| \mathbf { g } _ { k } ^ { \dagger } \| = 1$ . Since X is compact, (int $( \operatorname { d o m } ( f ) ) ) ^ { c }$ (the complement of int(dom(f))) is closed, and $X \cap ( \operatorname { i n t } ( \operatorname { d o m } ( f ) ) ) ^ { c } =$ ∅, it follows that the distance between the two sets is nonempty, meaning in particular that there exists an $\varepsilon > 0$ for $\mathrm { w h i c h } ^ { 1 0 }$ 

$$
\| \mathbf {x} - \mathbf {y} \| \geq \varepsilon \text {   for   any   } \mathbf {x} \in X, \mathbf {y} \notin \operatorname{int} (\operatorname{dom} (f)).\tag{3.13}
$$

The relation $\mathbf { g } _ { k } \in \partial f ( \mathbf { x } _ { k } )$ implies in particular that 

$$
f \left(\mathbf {x} _ {k} + \frac {\varepsilon}{2} \mathbf {g} _ {k} ^ {\dagger}\right) - f (\mathbf {x} _ {k}) \geq \frac {\varepsilon}{2} \langle \mathbf {g} _ {k}, \mathbf {g} _ {k} ^ {\dagger} \rangle = \frac {\varepsilon}{2} \| \mathbf {g} _ {k} \| _ {*},\tag{3.14}
$$

where we used the fact that by (3.13), $\begin{array} { r } { \mathbf { x } _ { k } + \frac { \varepsilon } { 2 } \mathbf { g } _ { k } ^ { \dagger } \in \mathrm { i n t } ( \mathrm { d o m } ( f ) ) } \end{array}$ . We will show that the left-hand side of (3.14) is bounded. Suppose by contradiction that it is not bounded. Then there exist subsequences $\{ \mathbf { x } _ { k } \} _ { k \in T } , \{ \mathbf { g } _ { k } ^ { \dag } \} _ { k \in T } \ ( T$ being the set of indices of the subsequences) for which 

$$
f \left(\mathbf {x} _ {k} + \frac {\varepsilon}{2} \mathbf {g} _ {k} ^ {\dagger}\right) - f (\mathbf {x} _ {k}) \rightarrow \infty \text {as} k \stackrel {T} {\longrightarrow} \infty .\tag{3.15}
$$

Since both $\{ { \bf { x } } _ { k } \} _ { k \in T }$ and $\{ \mathbf { g } _ { k } ^ { \dagger } \} _ { k \in T }$ are bounded, it follows that there exist convergent subsequences $\{ \mathbf { x } _ { k } \} _ { k \in S } , \{ \mathbf { g } _ { k } ^ { \dag } \} _ { k \in S } \left( S \subseteq T \right)$ whose limits will be denoted by x¯ and g¯. Consequently, $\begin{array} { r } { \mathbf { x } _ { k } + \frac { \varepsilon } { 2 } \mathbf { g } _ { k } ^ { \dagger }  \bar { \mathbf { x } } + \frac { \varepsilon } { 2 } \bar { \mathbf { g } } } \end{array}$ as $k { \stackrel { S } { \to } } \infty$ . Since $\begin{array} { r } { { \bf x } _ { k } , { \bf x } _ { k } + \frac { \varepsilon } { 2 } { \bf g } _ { k } ^ { \dagger } , \bar { \bf x } + \frac { \varepsilon } { 2 } \bar { \bf g } } \end{array}$ are $\mathrm { a l l } ^ { 1 1 }$ in int $\left( \operatorname { d o m } ( f ) \right)$ , it follows by the continuity of $f$ over int(dom(f)) (Theorem 2.21) that 

$$
f \left(\mathbf {x} _ {k} + \frac {\varepsilon}{2} \mathbf {g} _ {k} ^ {\dagger}\right) - f (\mathbf {x} _ {k}) \rightarrow f \left(\bar {\mathbf {x}} + \frac {\varepsilon}{2} \bar {\mathbf {g}} ^ {\dagger}\right) - f (\bar {\mathbf {x}}) \text {as} k \stackrel {{S}} {{\rightarrow}} \infty ,
$$

which is a contradiction of (3.15). We can thus conclude that the left-hand side of (3.14) is bounded and hence that the right-hand side of (3.14) is also bounded, in contradiction to our assumption that $\| \mathbf { g } _ { k } \|$ <sub>∗</sub> goes to $\infty$ as $k \to \infty$ 

Subdiferentiability can be guaranteed for points that are not necessarily in the interior of the domain but are in the interior of the domain w.r.t. its afine hull. This is the notion of relative interior that we now recall: 

$$
\operatorname{ri} (S) = \{\mathbf {x} \in \operatorname{aff} (S): B [ \mathbf {x}, \varepsilon ] \cap \operatorname{aff} (S) \subseteq S \text {   for   some   } \varepsilon > 0 \}.
$$

One key property of the relative interior is that it is nonempty for convex sets. 

Theorem 3.17 (nonemptiness of the relative interior [108, Theorem 6.2]). Let $C \subseteq \mathbb { E }$ be a nonempty convex set. Then ri(C) is nonempty. 

<sup>10</sup>The proof of $_ { ( 3 . 1 3 ) }$ is simple. Suppose by contradiction that there exist sequences $\{ \mathbf { x } _ { k } \} _ { k \geq 1 } \subseteq$ X and $\{ \mathbf { y } _ { k } \} _ { k \geq 1 } \subseteq ( \mathrm { i n t } ( \mathrm { d o m } ( f ) ) ) ^ { c }$ satisfying $\| \mathbf { x } _ { k } - \mathbf { y } _ { k } \| \to 0$ as $k \to \infty$ . Since $\{ \mathbf { x } _ { k } \} _ { k \ge 1 }$ is bounded, there exists $\bar { M } > 0$ for which $\lvert \lvert \mathbf { x } _ { k } \rvert \rvert \leq \dot { M }$ for all k. Therefore, $\| \mathbf { y } _ { k } \| \leq \| \mathbf { x } _ { k } - \mathbf { y } _ { k } \| + \| \mathbf { x } _ { k } \| \leq \| \mathbf { x } _ { k }$ $\mathbf { y } _ { k } \| + M$ , and we can conclude by the boundedness of $\{ \| \mathbf { x } _ { k } - \mathbf { y } _ { k } \| \} _ { k \geq 1 }$ that $\{ \mathbf { y } _ { k } \} _ { k \ge 1 }$ is bounded. By the Bolzano–Weierstrass theorem, there exist convergent subsequences $\mathbf { x } _ { k _ { i } }  \bar { \mathbf { x } } , \mathbf { y } _ { k _ { i } }  \bar { \mathbf { y } }$ , and by the closedness of X and $( \operatorname { i n t } ( \operatorname { d o m } ( f ) ) ) ^ { c }$ , we have that $\bar { \mathbf { x } } \in X , \bar { \mathbf { y } } \in ( \mathrm { i n t } ( \mathrm { d o m } ( f ) ) ) ^ { c } .$ . The limit $\| \mathbf { x } _ { k _ { j } } - \mathbf { y } _ { k _ { j } } \| \to 0 { \mathrm { ~ a s ~ } } j \to \infty$ now brings us to the impossible equality $\bar { \mathbf { x } } = \bar { \mathbf { y } }$ 

A well-known result is that a proper convex function is always subdiferentiable at relative interior points of its domain. We state this result without a proof. 

Theorem 3.18 (nonemptiness of the subdiferential set at relative interior points [108, Theorem 23.4]). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper convex function, and let $\tilde { \mathbf { x } } \in \mathrm { r i } ( \mathrm { d o m } ( f ) )$ . Then $\partial f ( { \widetilde { \mathbf { x } } } )$ is nonempty. 

The result stated in Theorem 3.18 can be written as the inclusion 

$$
\operatorname{ri} (\operatorname{dom} (f)) \subseteq \operatorname{dom} (\partial f).
$$

Since the relative interior of $\operatorname { d o m } ( f )$ is always nonempty (Theorem 3.17), we can conclude that there always exists a point in the domain for which the subdiferential set is nonempty. 

Corollary 3.19. Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper convex function. Then there exists $\mathbf { x } \in \operatorname { d o m } ( f )$ for which $\partial f ( \mathbf { x } )$ is nonempty. 

One instance in which the subdiferential set $\partial f ( \mathbf { x } )$ is guaranteed to be unbounded is when the dimension of the domain of the function is strictly smaller than the dimension of the underlying space E. 

Theorem 3.20 (unboundedness of the subdiferential set when dim(dom(f)) $< \dim ( \mathbb { E } ) )$ . Let $f : \mathbb { E }  ( - \infty , \infty ]$ be a proper convex function. Suppose that $\dim ( \operatorname { d o m } ( f ) ) \ < \ \dim ( \mathbb { E } )$ and let $\mathbf { x } \in \mathop { \mathrm { d o m } } ( f )$ . If $\partial f ( \mathbf { x } ) \neq \emptyset .$ , then $\partial f ( \mathbf { x } )$ is unbounded. 

Proof. Let $\eta$ be an arbitrary vector in $\partial f ( \mathbf { x } )$ . The $\sec ^ { 1 2 } \bigvee \equiv \operatorname { a f f } ( \operatorname { d o m } ( f ) ) - \{ \mathbf { x } \}$ is a vector space. The dimension condition translates to $\dim ( \mathbb { V } ) < \dim ( \mathbb { E } )$ , which in particular implies that there exists a nonzero vector $\mathbf { v } \in \mathbb { E }$ such that $\langle \mathbf { v } , \mathbf { w } \rangle = 0$ for any $\mathbf { w } \in \mathbb { V }$ . Take any $\beta \in \mathbb { R }$ . For any $\mathbf { y } \in \operatorname { d o m } ( f )$ 2 

$$
f (\mathbf {y}) \geq f (\mathbf {x}) + \langle \boldsymbol {\eta}, \mathbf {y} - \mathbf {x} \rangle = f (\mathbf {x}) + \langle \boldsymbol {\eta} + \beta \mathbf {v}, \mathbf {y} - \mathbf {x} \rangle ,
$$

where the equality is due to the fact that $\mathbf { y } - \mathbf { x } \in \mathbb { V } .$ . We thus obtained that $\eta + \beta \mathbf { v } \in \partial f ( \mathbf { x } )$ for any $\beta _ { ; }$ , implying the unboundedness of $\partial f ( \mathbf { x } )$ □ 

## 3.3 Directional Derivatives

## 3.3.1 Definition and Basic Properties

Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper function and let $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ . The directional derivative of $f$ at x in a given direction d $\in \mathbb { E }$ , if it exists, is defined by 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) \equiv \lim _ {\alpha \to 0 ^ {+}} \frac {f (\mathbf {x} + \alpha \mathbf {d}) - f (\mathbf {x})}{\alpha}.
$$

A well-known result states that convex functions have directional derivatives in all directions at points in the interior of their domains. 

Theorem 3.21 ([108, Theorem $\mathbf { 2 3 . 1 } ] ^ { 1 3 } )$ . Let $f : \mathbb { E }  ( - \infty , \infty ]$ be a proper convex function and let ${ \textbf { x } } \in$ int $\left( \operatorname { d o m } ( f ) \right)$ . Then $f o r$ any d $\in \ \mathbb { E }$ , the directional derivative $f ^ { \prime } ( \mathbf { x } ; \mathbf { d } )$ exists. 

It is important to establish some basic properties of the function d $\mapsto f ^ { \prime } ( \mathbf { x } ; \mathbf { d } )$ The next theorem shows that it is convex and homogeneous of degree 1. 

Lemma 3.22 (convexity and homogeneity of d $\mapsto \ f ^ { \prime } ( { \bf x } ; { \bf d } ) )$ . Let $f : \mathbb { E } $ $( - \infty , \infty ]$ be a proper convex function and let $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ . Then 

(a) the function d $\mapsto f ^ { \prime } ( \mathbf { x } ; \mathbf { d } )$ is convex; 

(b) for any $\lambda \geq 0$ and $\mathbf { d } \in \mathbb { E } ,$ it holds that $f ^ { \prime } ( \mathbf { x } ; \lambda \mathbf { d } ) = \lambda f ^ { \prime } ( \mathbf { x } ; \mathbf { d } )$ 

Proof. (a) To show that the function $g ( \mathbf { d } ) \equiv f ^ { \prime } ( \mathbf { x } ; \mathbf { d } )$ is convex, take $\mathbf { d } _ { 1 } , \mathbf { d } _ { 2 } \in \mathbb { E }$ and $\lambda \in [ 0 , 1 ]$ . Then 

$$
\begin{array}{l} f ^ {\prime} (\mathbf {x}; \lambda \mathbf {d} _ {1} + (1 - \lambda) \mathbf {d} _ {2}) \\ = \lim _ {\alpha \to 0 ^ {+}} \frac {f (\mathbf {x} + \alpha [ \lambda \mathbf {d} _ {1} + (1 - \lambda) \mathbf {d} _ {2} ]) - f (\mathbf {x})}{\alpha} \\ = \lim _ {\alpha \to 0 ^ {+}} \frac {f (\lambda (\mathbf {x} + \alpha \mathbf {d} _ {1}) + (1 - \lambda) (\mathbf {x} + \alpha \mathbf {d} _ {2})) - f (\mathbf {x})}{\alpha} \\ \leq \lim _ {\alpha \to 0 ^ {+}} \frac {\lambda f (\mathbf {x} + \alpha \mathbf {d} _ {1}) + (1 - \lambda) f (\mathbf {x} + \alpha \mathbf {d} _ {2}) - f (\mathbf {x})}{\alpha} \\ = \lambda \lim _ {\alpha \to 0 ^ {+}} \frac {f (\mathbf {x} + \alpha \mathbf {d} _ {1}) - f (\mathbf {x})}{\alpha} + (1 - \lambda) \lim _ {\alpha \to 0 ^ {+}} \frac {f (\mathbf {x} + \alpha \mathbf {d} _ {2}) - f (\mathbf {x})}{\alpha} \\ = \lambda f ^ {\prime} (\mathbf {x}; \mathbf {d} _ {1}) + (1 - \lambda) f ^ {\prime} (\mathbf {x}; \mathbf {d} _ {2}), \end{array}
$$

where the inequality follows from Jensen’s inequality for convex functions. 

(b) If $\lambda = 0$ , the claim is trivial. Take $\lambda > 0$ . Then 

$$
f ^ {\prime} (\mathbf {x}; \lambda \mathbf {d}) = \lim _ {\alpha \to 0 ^ {+}} \frac {f (\mathbf {x} + \alpha \lambda \mathbf {d}) - f (\mathbf {x})}{\alpha} = \lambda \lim _ {\alpha \to 0 ^ {+}} \frac {f (\mathbf {x} + \alpha \lambda \mathbf {d}) - f (\mathbf {x})}{\alpha \lambda} = \lambda f ^ {\prime} (\mathbf {x}; \mathbf {d}).
$$

The next result highlights a connection between function values and directional derivatives under a convexity assumption. 

Lemma 3.23. Let $f : \mathbb { E }  ( - \infty , \infty ]$ be a proper convex function, and let ${ \textbf { x } } \in$ int $\left( \operatorname { d o m } ( f ) \right)$ ). Then 

$$
f (\mathbf {y}) \geq f (\mathbf {x}) + f ^ {\prime} (\mathbf {x}; \mathbf {y} - \mathbf {x}) \text {   for   all   } \mathbf {y} \in \operatorname{dom} (f).
$$

Proof. By the definition of the directional derivative, 

$$
\begin{array}{l} f ^ {\prime} (\mathbf {x}; \mathbf {y} - \mathbf {x}) = \lim _ {\alpha \to 0 ^ {+}} \frac {f (\mathbf {x} + \alpha (\mathbf {y} - \mathbf {x})) - f (\mathbf {x})}{\alpha} \\ \qquad = \lim _ {\alpha \to 0 ^ {+}} \frac {f ((1 - \alpha) \mathbf {x} + \alpha \mathbf {y}) - f (\mathbf {x})}{\alpha} \\ \qquad \leq \lim _ {\alpha \to 0 ^ {+}} \frac {- \alpha f (\mathbf {x}) + \alpha f (\mathbf {y})}{\alpha} \\ \qquad = f (\mathbf {y}) - f (\mathbf {x}), \end{array}
$$

where the inequality follows by Jensen’s inequality. 

A useful “calculus” rule for directional derivatives shows how to compute the directional derivative of maximum of a finite collection of functions without any convexity assumptions. 

Theorem 3.24 (directional derivative of maximum of functions). Suppose that $f ( \mathbf { x } ) = \operatorname* { m a x } \{ f _ { 1 } ( \mathbf { x } ) , f _ { 2 } ( \mathbf { x } ) , \ldots , f _ { m } ( \mathbf { x } ) \}$ , where $f _ { 1 } , f _ { 2 } , \ldots , f _ { m } : \mathbb { E } \to ( - \infty , \infty ]$ are proper functions. Let $\mathbf { x } \in \bigcap _ { i = 1 } ^ { m }$ int(dom(f<sub>i</sub>)) and $\mathbf { d } \in \mathbb { E }$ . Assume that $f _ { i } ^ { \prime } ( \mathbf { x } ; \mathbf { d } )$ exist for any $i \in \{ 1 , 2 , . . . , m \}$ . Then 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \max _ {i \in I (\mathbf {x})} f _ {i} ^ {\prime} (\mathbf {x}; \mathbf {d}),
$$

where $I ( \mathbf { x } ) = \{ i : f _ { i } ( \mathbf { x } ) = f ( \mathbf { x } ) \}$ 

Proof. For any $i \in \{ 1 , 2 , . . . , m \}$ 

$$
\lim _ {t \rightarrow 0 ^ {+}} f _ {i} (\mathbf {x} + t \mathbf {d}) = \lim _ {t \rightarrow 0 ^ {+}} \left[ t \frac {f _ {i} (\mathbf {x} + t \mathbf {d}) - f _ {i} (\mathbf {x})}{t} + f _ {i} (\mathbf {x}) \right] = 0 \cdot f _ {i} ^ {\prime} (\mathbf {x}; \mathbf {d}) + f _ {i} (\mathbf {x}) = f _ {i} (\mathbf {x}).\tag{3.16}
$$

By the definition of $I ( { \bf x } ) , f _ { i } ( { \bf x } ) > f _ { j } ( { \bf x } )$ for any $i \in I ( \mathbf { x } ) , j \notin I ( \mathbf { x } )$ . Utilizing (3.16), it follows that there exists an $\varepsilon > 0$ such that $f _ { i } ( \mathbf { x } + t \mathbf { d } ) > f _ { j } ( \mathbf { x } + t \mathbf { d } )$ for any $i \in I ( \mathbf { x } ) , j \notin I ( \mathbf { x } )$ and $t \in ( 0 , \varepsilon ]$ . Therefore, for any $t \in ( 0 , \varepsilon ]$ 

$$
f (\mathbf {x} + t \mathbf {d}) = \max _ {i = 1, 2, \dots , m} f _ {i} (\mathbf {x} + t \mathbf {d}) = \max _ {i \in I (\mathbf {x})} f _ {i} (\mathbf {x} + t \mathbf {d}).
$$

Consequently, for any $t \in ( 0 , \varepsilon ]$ 

$$
\frac {f (\mathbf {x} + t \mathbf {d}) - f (\mathbf {x})}{t} = \frac {\max _ {i \in I (\mathbf {x})} f _ {i} (\mathbf {x} + t \mathbf {d}) - f (\mathbf {x})}{t} = \max _ {i \in I (\mathbf {x})} \frac {f _ {i} (\mathbf {x} + t \mathbf {d}) - f _ {i} (\mathbf {x})}{t},
$$

where the last equality follows from the fact that $f ( \mathbf { x } ) = f _ { i } ( \mathbf { x } )$ for any $i \in I ( \mathbf { x } )$ Finally, taking $t \to 0 ^ { + }$ , we obtain that 

$$
\begin{array}{l} f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \lim _ {t \to 0 ^ {+}} \frac {f (\mathbf {x} + t \mathbf {d}) - f (\mathbf {x})}{t} \\ \qquad = \lim _ {t \to 0 ^ {+}} \max _ {i \in I (\mathbf {x})} \frac {f _ {i} (\mathbf {x} + t \mathbf {d}) - f _ {i} (\mathbf {x})}{t} \\ \qquad = \max _ {i \in I (\mathbf {x})} \lim _ {t \to 0 ^ {+}} \frac {f _ {i} (\mathbf {x} + t \mathbf {d}) - f _ {i} (\mathbf {x})}{t} \\ \qquad = \max _ {i \in I (\mathbf {x})} f _ {i} ^ {\prime} (\mathbf {x}; \mathbf {d}). \quad \square \end{array}
$$

Note that an assumption of Theorem 3.24 is that the directional derivatives $f _ { i } ^ { \prime } ( \mathbf { x } ; \mathbf { d } )$ exist. This assumption is automatically satisfied when the functions $f _ { 1 } , f _ { 2 } , \ldots , f _ { m }$ are convex. We can thus write the next corollary that replaces the condition on the existence of the directional derivatives by a convexity assumption. 

Corollary 3.25 (directional derivative of maximum of functions—convex case). Suppose that $f ( \mathbf { x } ) = \operatorname* { m a x } \{ f _ { 1 } ( \mathbf { x } ) , f _ { 2 } ( \mathbf { x } ) , \ldots , f _ { m } ( \mathbf { x } ) \}$ , where $f _ { 1 } , f _ { 2 } , \ldots , f _ { m }$ $\mathbb { E } \to ( - \infty , \infty ]$ are proper convex functions. Let $\mathbf { x } \in \cap _ { i = 1 } ^ { m }$ int $\left( \operatorname { d o m } ( f _ { i } ) \right)$ and d $\in \mathbb { E }$ Then 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \max _ {i \in I (\mathbf {x})} f _ {i} ^ {\prime} (\mathbf {x}; \mathbf {d}),
$$

where $I ( \mathbf { x } ) = \{ i : f _ { i } ( \mathbf { x } ) = f ( \mathbf { x } ) \}$ 

## 3.3.2 The Max Formula

We will now prove an extremely important and useful result, known as the max formula, that connects subgradients and directional derivatives. 

Theorem 3.26 (max formula). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper convex function. Then for any $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ and d $\in \mathbb { E }$ 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \max \left\{\langle \mathbf {g}, \mathbf {d} \rangle : \mathbf {g} \in \partial f (\mathbf {x}) \right\}.\tag{3.17}
$$

Proof. Let $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ and $\mathbf { d } \in \mathbb { E }$ . By the subgradient inequality, we have that for any $\mathbf { g } \in \partial f ( \mathbf { x } )$ 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \lim _ {\alpha \to 0 ^ {+}} \frac {1}{\alpha} (f (\mathbf {x} + \alpha \mathbf {d}) - f (\mathbf {x})) \geq \lim _ {\alpha \to 0 ^ {+}} \langle \mathbf {g}, \mathbf {d} \rangle = \langle \mathbf {g}, \mathbf {d} \rangle\tag{3.18}
$$

and, consequently, 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) \geq \max \{\langle \mathbf {g}, \mathbf {d} \rangle : \mathbf {g} \in \partial f (\mathbf {x}) \}.\tag{3.19}
$$

All that is left is to show the reverse direction of the above inequality. For that, define the function $h ( \mathbf { w } ) \equiv f ^ { \prime } ( \mathbf { x } ; \mathbf { w } )$ . Then by Lemma $3 . 2 2 ( \mathrm { a } )$ , h is a real-valued convex function and is thus subdiferentiable over E (Corollary 3.15). Let $\tilde { \mathbf { g } } \in \partial h ( \mathbf { d } )$ Then for any $\mathbf { v } \in \mathbb { E }$ and $\alpha \geq 0$ , using the homogeneity of h (Lemma 3.22(b)), 

$$
\alpha f ^ {\prime} (\mathbf {x}; \mathbf {v}) = f ^ {\prime} (\mathbf {x}; \alpha \mathbf {v}) = h (\alpha \mathbf {v}) \geq h (\mathbf {d}) + \langle \tilde {\mathbf {g}}, \alpha \mathbf {v} - \mathbf {d} \rangle = f ^ {\prime} (\mathbf {x}; \mathbf {d}) + \langle \tilde {\mathbf {g}}, \alpha \mathbf {v} - \mathbf {d} \rangle .
$$

Therefore, 

$$
\alpha (f ^ {\prime} (\mathbf {x}; \mathbf {v}) - \langle \tilde {\mathbf {g}}, \mathbf {v} \rangle) \geq f ^ {\prime} (\mathbf {x}; \mathbf {d}) - \langle \tilde {\mathbf {g}}, \mathbf {d} \rangle .\tag{3.20}
$$

Since the above inequality holds for any $\alpha \geq 0$ , it follows that the coeficient of α in the left-hand side expression is nonnegative (otherwise, inequality (3.20) would be violated for large enough α), meaning that 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {v}) \geq \langle \tilde {\mathbf {g}}, \mathbf {v} \rangle .
$$

Thus, by Lemma 3.23, for any $\mathbf { y } \in \operatorname { d o m } ( f )$ 

$$
f (\mathbf {y}) \geq f (\mathbf {x}) + f ^ {\prime} (\mathbf {x}; \mathbf {y} - \mathbf {x}) \geq f (\mathbf {x}) + \langle \tilde {\mathbf {g}}, \mathbf {y} - \mathbf {x} \rangle ,
$$

showing that $\tilde { \mathbf { g } } \in \partial f ( \mathbf { x } )$ . Taking $\alpha = 0$ in (3.20), we have that $f ^ { \prime } ( \mathbf { x } ; \mathbf { d } ) \leq \langle \tilde { \mathbf { g } } , \mathbf { d } \rangle$ , so that 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) \leq \langle \tilde {\mathbf {g}}, \mathbf {d} \rangle \leq \max \{\langle \mathbf {g}, \mathbf {d} \rangle : \mathbf {g} \in \partial f (\mathbf {x}) \},
$$

establishing the desired result. 

Remark 3.27. The max formula (3.17) can also be rewritten using the support function notation as follows: 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \sigma_ {\partial f (\mathbf {x})} (\mathbf {d}).
$$

## 3.3.3 Diferentiability

Definition 3.28 (diferentiability). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ and $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } f )$ The function f is said to be diferentiable at x if there exists $\mathbf { g } \in \mathbb { E } ^ { * }$ such that 

$$
\lim _ {\mathbf {h} \to \mathbf {0}} \frac {f (\mathbf {x} + \mathbf {h}) - f (\mathbf {x}) - \langle \mathbf {g} , \mathbf {h} \rangle}{\| \mathbf {h} \|} = 0.\tag{3.21}
$$

The $u n i q u e ^ { 1 4 }$ vector g satisfying (3.21) is called the gradient of f at x and is denoted by $\nabla f ( \mathbf { x } )$ 

The above is actually a definition of Fr´echet diferentiability, which is the one used in this book. 

If f is diferentiable at $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } f )$ , then the directional derivative has a simple formula. 

Theorem 3.29 (directional derivatives at points of diferentiability). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be proper, and suppose that f is diferentiable at ${ \textbf { x } } \in$ int(domf). Then for any $\mathbf { d } \in \mathbb { E }$ 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \langle \nabla f (\mathbf {x}), \mathbf {d} \rangle .\tag{3.22}
$$

Proof. The formula is obviously correct for $\mathbf { d } = \mathbf { 0 }$ . Suppose that d $\neq \mathbf { 0 }$ . The diferentiability of f implies that 

$$
\begin{array}{l} 0 = \lim _ {\alpha \to 0 ^ {+}} \frac {f (\mathbf {x} + \alpha \mathbf {d}) - f (\mathbf {x}) - \langle \nabla f (\mathbf {x}) , \alpha \mathbf {d} \rangle}{\| \alpha \mathbf {d} \|} \\ = \lim _ {\alpha \to 0 ^ {+}} \left[ \frac {f (\mathbf {x} + \alpha \mathbf {d}) - f (\mathbf {x})}{\alpha \| \mathbf {d} \|} - \frac {\langle \nabla f (\mathbf {x}) , \mathbf {d} \rangle}{\| \mathbf {d} \|} \right]. \end{array}
$$

Therefore, 

$$
\begin{array}{l} f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \lim _ {\alpha \to 0 ^ {+}} \frac {f (\mathbf {x} + \alpha \mathbf {d}) - f (\mathbf {x})}{\alpha} \\ \qquad = \lim _ {\alpha \to 0 ^ {+}} \left\{\| \mathbf {d} \| \left[ \frac {f (\mathbf {x} + \alpha \mathbf {d}) - f (\mathbf {x})}{\alpha \| \mathbf {d} \|} - \frac {\langle \nabla f (\mathbf {x}) , \mathbf {d} \rangle}{\| \mathbf {d} \|} \right] + \langle \nabla f (\mathbf {x}), \mathbf {d} \rangle \right\} \\ \qquad = \langle \nabla f (\mathbf {x}), \mathbf {d} \rangle . \quad \square \end{array}
$$

Example 3.30 (directional derivative of maximum of diferentiable functions). Consider the function $f ( \mathbf { x } ) = \mathrm { m a x } _ { i = 1 , 2 , . . . , m } f _ { i } ( \mathbf { x } )$ , where $f _ { i } : \mathbb { E }  ( - \infty , \infty ]$ are proper functions. Assume that $f _ { 1 } , f _ { 2 } , \ldots , f _ { m }$ are diferentiable at a given point $\mathbf { x } \in \cap _ { i = 1 } ^ { m }$ int(dom(f<sub>i</sub>). Then by Theorem 3.29, for any d $\in \mathbb { E } , f _ { i } ^ { \prime } ( \mathbf { x } ; \mathbf { d } ) = \langle \nabla f _ { i } ( \mathbf { x } ) , \mathbf { d } \rangle$ Therefore, invoking Theorem 3.24, 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \max _ {i \in I (\mathbf {x})} f _ {i} ^ {\prime} (\mathbf {x}; \mathbf {d}) = \max _ {i \in I (\mathbf {x})} \langle \nabla f _ {i} (\mathbf {x}), \mathbf {d} \rangle ,
$$

where $I ( \mathbf { x } ) = \{ i : f _ { i } ( \mathbf { x } ) = f ( \mathbf { x } ) \} .$ ■ 

Example 3.31 (gradient of $\textstyle { \frac { 1 } { 2 } } d _ { C } ^ { 2 } ( \cdot ) )$ .<sup>15</sup> Suppose that E is a Euclidean space, and let $C \subseteq \mathbb { E }$ be nonempty closed and convex set. Consider the function $\varphi _ { C } : \mathbb { E } $ R given by $\varphi _ { C } ( { \bf x } ) \equiv { \textstyle \frac { 1 } { 2 } } d _ { C } ^ { 2 } ( { \bf x } ) = { \textstyle \frac { 1 } { 2 } } \| { \bf x } - P _ { C } ( { \bf x } ) \| ^ { 2 }$ , where $P _ { C }$ is the so-called orthogonal projection mapping defined by 

$$
P _ {C} (\mathbf {x}) \equiv \operatorname{argmin} _ {\mathbf {y} \in C} \| \mathbf {y} - \mathbf {x} \|.
$$

It is well known that $P _ { C }$ is well defined (exists and unique) when the underlying set $C$ is nonempty, closed, and convex.<sup>16</sup> We will show that for any $\mathbf { x } \in \mathbb { E }$ 4 

$$
\nabla \varphi_ {C} (\mathbf {x}) = \mathbf {x} - P _ {C} (\mathbf {x}).\tag{3.23}
$$

For that, fix $\mathbf { x } \in \mathbb { E }$ and define the function $g _ { \mathbf x }$ by 

$$
g _ {\mathbf {x}} (\mathbf {d}) \equiv \varphi_ {C} (\mathbf {x} + \mathbf {d}) - \varphi_ {C} (\mathbf {x}) - \langle \mathbf {d}, \mathbf {z} _ {\mathbf {x}} \rangle ,
$$

where ${ \bf z } _ { \bf x } = { \bf x } - P _ { C } ( { \bf x } )$ . By the definition of the gradient, to show (3.23), it is enough to establish that 

$$
\frac {g _ {\mathbf {x}} (\mathbf {d})}{\| \mathbf {d} \|} \rightarrow \mathbf {0} \text {as} \mathbf {d} \rightarrow \mathbf {0}.\tag{3.24}
$$

To prove (3.24), note that by the definition of the orthogonal projection, we have for any $\mathbf { d } \in \mathbb { E }$ 

$$
\left\| \mathbf {x} + \mathbf {d} - P _ {C} (\mathbf {x} + \mathbf {d}) \right\| ^ {2} \leq \left\| \mathbf {x} + \mathbf {d} - P _ {C} (\mathbf {x}) \right\| ^ {2},
$$

which implies that for any $\mathbf { d } \in \mathbb { E }$ 

$$
\begin{array}{l} g _ {\mathbf {x}} (\mathbf {d}) = \frac {1}{2} \| \mathbf {x} + \mathbf {d} - P _ {C} (\mathbf {x} + \mathbf {d}) \| ^ {2} - \frac {1}{2} \| \mathbf {x} - P _ {C} (\mathbf {x}) \| ^ {2} - \langle \mathbf {d}, \mathbf {z} _ {\mathbf {x}} \rangle \\ \leq \frac {1}{2} \| \mathbf {x} + \mathbf {d} - P _ {C} (\mathbf {x}) \| ^ {2} - \frac {1}{2} \| \mathbf {x} - P _ {C} (\mathbf {x}) \| ^ {2} - \langle \mathbf {d}, \mathbf {z} _ {\mathbf {x}} \rangle \\ = \frac {1}{2} \| \mathbf {x} - P _ {C} (\mathbf {x}) \| ^ {2} + \langle \mathbf {d}, \mathbf {x} - P _ {C} (\mathbf {x}) \rangle + \frac {1}{2} \| \mathbf {d} \| ^ {2} - \frac {1}{2} \| \mathbf {x} - P _ {C} (\mathbf {x}) \| ^ {2} - \langle \mathbf {d}, \mathbf {z} _ {\mathbf {x}} \rangle \\ = \frac {1}{2} \| \mathbf {d} \| ^ {2}. \end{array} \tag {3.2}\tag{3.25}
$$

In particular, we also have 

$$
g _ {\mathbf {x}} (- \mathbf {d}) \leq \frac {1}{2} \| \mathbf {d} \| ^ {2}.\tag{3.26}
$$

Since $\varphi _ { C }$ is convex, so is $g _ { \mathbf { x } }$ . Therefore, by Jensen’s inequality, and noting that $g _ { \mathbf { x } } ( \mathbf { 0 } ) = 0$ 

$$
0 = g _ {\mathbf {x}} (\mathbf {0}) = g _ {\mathbf {x}} \left(\frac {\mathbf {d} + (- \mathbf {d})}{2}\right) \leq \frac {1}{2} (g _ {\mathbf {x}} (\mathbf {d}) + g _ {\mathbf {x}} (- \mathbf {d})).\tag{3.27}
$$

Combining (3.26) and (3.27), we get 

$$
g _ {\mathbf {x}} (\mathbf {d}) \geq - g _ {\mathbf {x}} (- \mathbf {d}) \geq - \frac {1}{2} \| \mathbf {d} \| ^ {2}.\tag{3.28}
$$

Finally, by (3.25) and (3.28), it follows that $\begin{array} { r } { | g _ { \mathbf { x } } ( \mathbf { d } ) | \leq \frac { 1 } { 2 } \| \mathbf { d } \| ^ { 2 } } \end{array}$ , from which the limit (3.24) follows and hence also the desired result (3.23). 

Remark 3.32 (what is the gradient?). We will now illustrate the fact that the gradient depends on the choice of the inner product in the underlying space. Let $\mathbb { E } = \mathbb { R } ^ { n }$ be endowed with the dot product. By Theorem 3.29 we know that when f is diferentiable at x, then 

$$
(\nabla f (\mathbf {x})) _ {i} = \langle \nabla f (\mathbf {x}), \mathbf {e} _ {i} \rangle = f ^ {\prime} (\mathbf {x}; \mathbf {e} _ {i});
$$

that is, in this case, the ith component of $\nabla f ( \mathbf { x } )$ is equal to $\textstyle { \frac { \partial f } { \partial x _ { i } } } ( \mathbf { x } ) = f ^ { \prime } ( \mathbf { x } ; \mathbf { e } _ { i } )$ —the ith partial derivative of f at x—so that $\nabla f ( \mathbf { x } ) = D _ { f } ( \mathbf { x } )$ , where 

$$
D _ {f} (\mathbf {x}) \equiv \left( \begin{array}{c} \frac {\partial f}{\partial x _ {1}} (\mathbf {x}) \\ \frac {\partial f}{\partial x _ {2}} (\mathbf {x}) \\ \vdots \\ \frac {\partial f}{\partial x _ {n}} (\mathbf {x}) \end{array} \right).\tag{3.29}
$$

Note that the definition of the directional derivative does not depend on the choice of the inner product in the underlying space, so we can arbitrarily choose the inner product in the formula (3.22) as the dot product and obtain (recalling that in this case $\nabla f ( \mathbf { x } ) = D _ { f } ( \mathbf { x } ) )$ ) 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) = D _ {f} (\mathbf {x}) ^ {T} \mathbf {d} = \sum_ {i = 1} ^ {n} \frac {\partial f}{\partial x _ {i}} (\mathbf {x}) d _ {i}.\tag{3.30}
$$

Formula (3.30) holds for any choice of inner product in the space. However, $\nabla f ( \mathbf { x } )$ is not necessarily equal to $D _ { f } ( \mathbf { x } )$ when the endowed inner product is not the dot product. For example, suppose that the inner product is given by 

$$
\langle \mathbf {x}, \mathbf {y} \rangle = \mathbf {x} ^ {T} \mathbf {H} \mathbf {y},\tag{3.31}
$$

where H is a given $n \times n$ positive definite matrix. In this case, 

$$
(\nabla f (\mathbf {x})) _ {i} = \nabla f (\mathbf {x}) ^ {T} \mathbf {e} _ {i} = \nabla f (\mathbf {x}) ^ {T} \mathbf {H} \left(\mathbf {H} ^ {- 1} \mathbf {e} _ {i}\right)
$$

$$
= \langle \nabla f (\mathbf {x}), \mathbf {H} ^ {- 1} \mathbf {e} _ {i} \rangle\tag{[by (3.31)]}
$$

$$
= f ^ {\prime} (\mathbf {x}; \mathbf {H} ^ {- 1} \mathbf {e} _ {i})\tag{[by (3.22)]}
$$

$$
= D _ {f} (\mathbf {x}) ^ {T} \mathbf {H} ^ {- 1} \mathbf {e} _ {i}.\tag{[by (3.30)]}
$$

Hence, we obtain that with respect to the inner product (3.31), the gradient is actually a “scaled”/“weighted” gradient: 

$$
\nabla f (\mathbf {x}) = \mathbf {H} ^ {- 1} D _ {f} (\mathbf {x}).
$$

Now consider the space $\mathbb { E } = \mathbb { R } ^ { m \times n }$ of all $m \times n$ real-valued matrices with the dot product as the endowed inner product: 

$$
\langle \mathbf {x}, \mathbf {y} \rangle = \operatorname{Tr} \left(\mathbf {x} ^ {T} \mathbf {y}\right) f o r a n y \mathbf {x}, \mathbf {y} \in \mathbb {R} ^ {m \times n}.
$$

Given a proper function $f : \mathbb { R } ^ { m \times n }  ( - \infty , \infty ]$ and $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ , the gradient, if it exists, is given by $\nabla f ( \mathbf { x } ) = D _ { f } ( \mathbf { x } )$ , where $D _ { f } ( \mathbf { x } )$ is the $m \times n$ matrix 

$$
D _ {f} (\mathbf {x}) = \left(\frac {\partial f}{\partial x _ {i j}} (\mathbf {x})\right) _ {i, j}.
$$

If the inner product is replaced by 

$$
\langle \mathbf {x}, \mathbf {y} \rangle = \mathrm{Tr} (\mathbf {x} ^ {T} \mathbf {H} \mathbf {y}),
$$

where H is a given m × m positive definite matrix, then a similar argument to the one given previously shows that 

$$
\nabla f (\mathbf {x}) = \mathbf {H} ^ {- 1} D _ {f} (\mathbf {x}).
$$

We will now show that when a convex function is diferentiable at a point in the interior of its domain, then the subdiferential set is the singleton (i.e., a set containing a single vector) consisting of the gradient at the point. The reverse is also correct in the sense that if the subdiferential set is a singleton $\{ \mathbf { g } \}$ , then the function is diferentiable at the given point with g being its gradient. 

Theorem 3.33 (the subdiferential at points of diferentiability). Let $f : \mathbb { E }  ( - \infty , \infty ]$ be a proper convex function, and let $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ . If f is diferentiable at x, then $\partial f ( \mathbf { x } ) = \{ \nabla f ( \mathbf { x } ) \}$ . Conversely, if f has a unique subgradient at x, then it is diferentiable at x and $\partial f ( \mathbf { x } ) = \{ \nabla f ( \mathbf { x } ) \}$ 

Proof. Let $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ and assume that f is diferentiable at x. Then by Theorem 3.29 it follows that for any $\mathbf { d } \in \mathbb { E }$ 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \langle \nabla f (\mathbf {x}), \mathbf {d} \rangle .\tag{3.32}
$$

Let $\mathbf { g } \in \partial f ( \mathbf { x } )$ . We will show that ${ \bf g } = \nabla f ( { \bf x } )$ . Combining (3.32) with the max formula (Theorem 3.26) we have 

$$
\langle \nabla f (\mathbf {x}), \mathbf {d} \rangle = f ^ {\prime} (\mathbf {x}; \mathbf {d}) \geq \langle \mathbf {g}, \mathbf {d} \rangle ,
$$

so that 

$$
\langle \mathbf {g} - \nabla f (\mathbf {x}), \mathbf {d} \rangle \leq 0.
$$

Taking the maximum over all d satisfying $\| \mathbf { d } \| \leq 1$ , we obtain that $\| \mathbf { g } - \nabla f ( \mathbf { x } ) \| _ { * } \leq 0$ and consequently that $\nabla f ( \mathbf { x } ) = \mathbf { g } .$ . We have thus shown that the only possible subgradient in $\partial f ( \mathbf { x } )$ is $\nabla f ( \mathbf { x } )$ . Combining this with the fact that the subdiferential set is nonempty (Theorem 3.14) yields the desired result $\partial f ( \mathbf { x } ) = \{ \nabla f ( \mathbf { x } ) \}$ 

For the reverse direction, suppose that f has a unique subgradient g at ${ \textbf { x } } \in$ int $\left( \operatorname { d o m } ( f ) \right)$ . Consider the convex function 

$$
h (\mathbf {u}) \equiv f (\mathbf {x} + \mathbf {u}) - f (\mathbf {x}) - \langle \mathbf {g}, \mathbf {u} \rangle .
$$

We will show that 

$$
\lim _ {\mathbf {u} \to \mathbf {0}} \frac {h (\mathbf {u})}{\| \mathbf {u} \|} = 0.
$$

This will establish (by definition) that ${ \bf g } = \nabla f ( { \bf x } )$ . Obviously, 0 is the unique subgradient of h at 0 and $\mathbf { 0 } \in \operatorname { i n t } ( \operatorname { d o m } ( h ) )$ , and hence by the max formula (Theorem 3.26), for any $\mathbf { d } \in \mathbb { E }$ 2 

$$
h ^ {\prime} (\mathbf {0}; \mathbf {d}) = \sigma_ {\partial h (\mathbf {0})} (\mathbf {d}) = 0.
$$

We can thus conclude that for any d $\in \mathbb { E }$ , 

$$
0 = h ^ {\prime} (\mathbf {0}; \mathbf {d}) = \lim _ {\alpha \to 0 ^ {+}} \frac {h (\alpha \mathbf {d}) - h (\mathbf {0})}{\alpha} = \lim _ {\alpha \to 0 ^ {+}} \frac {h (\alpha \mathbf {d})}{\alpha}.\tag{3.33}
$$

Let $\{ \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } , \ldots , \mathbf { v } _ { k } \}$ be an orthonormal basis of E. Since $\mathbf { 0 } \in \operatorname { i n t } ( \operatorname { d o m } ( h ) )$ , there exists $\varepsilon \in ( 0 , 1 )$ such that $\varepsilon \mathbf { v } _ { i } , - \varepsilon \mathbf { v } _ { i } \in \mathrm { d o m } ( h )$ for any $i = 1 , 2 , \ldots , k .$ Therefore, since dom(h) is convex, the set 

$$
D = \operatorname{conv} \left(\{\pm \varepsilon \mathbf {v} _ {i} \} _ {i = 1} ^ {k}\right)
$$

satisfies $D \subseteq \operatorname { d o m } ( h )$ . Let $\| \cdot \| = \sqrt { \langle \cdot , \cdot \rangle }$ be the Euclidean norm corresponding to the endowed inner product on E. Note that $B _ { \| \cdot \| } [ 0 , \gamma ] \subseteq D$ , where $\begin{array} { r } { \gamma = \frac { \varepsilon } { k } } \end{array}$ . Indeed, let $\mathbf { w } \in B _ { \lVert \cdot \rVert } [ \mathbf { 0 } , \gamma ]$ . Then since $\{ \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } , \ldots , \mathbf { v } _ { k } \}$ is an orthonormal basis of E, we have 

$$
\mathbf {w} = \sum_ {i = 1} ^ {k} \langle \mathbf {w}, \mathbf {v} _ {i} \rangle \mathbf {v} _ {i}
$$

as well as 

$$
\left\| \mathbf {w} \right\| ^ {2} = \sum_ {i = 1} ^ {k} \left\langle \mathbf {w}, \mathbf {v} _ {i} \right\rangle^ {2}.\tag{3.34}
$$

Since $\| \mathbf { w } \| ^ { 2 } \leq \gamma ^ { 2 }$ , it follows by (3.34) that $| \langle \mathbf { w } , \mathbf { v } _ { i } \rangle | \leq \gamma$ , and hence 

$$
\mathbf {w} = \sum_ {i = 1} ^ {k} \langle \mathbf {w}, \mathbf {v} _ {i} \rangle \mathbf {v} _ {i} = \sum_ {i = 1} ^ {k} \frac {| \langle \mathbf {w} , \mathbf {v} _ {i} \rangle |}{\varepsilon} [ \operatorname{sgn} (\langle \mathbf {w}, \mathbf {v} _ {i} \rangle) \varepsilon \mathbf {v} _ {i} ] + \left(1 - \sum_ {i = 1} ^ {k} \frac {| \langle \mathbf {w} , \mathbf {v} _ {i} \rangle |}{\varepsilon}\right) \cdot \mathbf {0} \in D,
$$

where the membership in D follows by the fact that $\mathbf { 0 } , \pm \varepsilon \mathbf { v } _ { i } \in D$ and $\begin{array} { r } { \sum _ { i = 1 } ^ { k } \frac { | \langle \mathbf w , \mathbf v _ { i } \rangle | } { \varepsilon } \leq } \end{array}$ $\begin{array} { r } { \frac { k \gamma } { \varepsilon } = 1 } \end{array}$ . We have therefore established the inclusion $B _ { \| \cdot \| } [ 0 , \gamma ] \subseteq D$ . Denote the 2k vectors $\{ \pm \varepsilon { \mathbf { v } } _ { i } \} _ { i = 1 } ^ { k }$ by $\mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , \ldots , \mathbf { z } _ { 2 k }$ . Take $\mathbf { 0 } \neq \mathbf { u } \in B _ { \| \cdot \| } [ \mathbf { 0 } , \gamma ^ { 2 } ]$ . We have that $\gamma \frac { \mathbf { u } } { \lVert \mathbf { u } \rVert } \in B _ { \lVert \cdot \rVert } \mathbf { \bar { [ 0 , \gamma ] } } \stackrel { \mathbf { \bar { \mathbf { C } } } } { = } \bar { D }$ , and hence there exists $\lambda \in \Delta _ { 2 k }$ such that 

$$
\gamma \frac {\mathbf {u}}{\| \mathbf {u} \|} = \sum_ {i = 1} ^ {2 k} \lambda_ {i} \mathbf {z} _ {i}.
$$

Therefore, 

$$
\begin{array}{l} \frac {h (\mathbf {u})}{\| \mathbf {u} \|} = \frac {h \left(\frac {\| \mathbf {u} \|}{\gamma} \gamma \frac {\mathbf {u}}{\| \mathbf {u} \|}\right)}{\| \mathbf {u} \|} = \frac {h \left(\sum_ {i = 1} ^ {2 k} \lambda_ {i} \frac {\| \mathbf {u} \|}{\gamma} \mathbf {z} _ {i}\right)}{\| \mathbf {u} \|} \\ \leq \sum_ {i = 1} ^ {2 k} \lambda_ {i} \frac {h \left(\| \mathbf {u} \| \frac {\mathbf {z} _ {i}}{\gamma}\right)}{\| \mathbf {u} \|} \\ \leq \max _ {i = 1, 2, \ldots , 2 k} \left\{\frac {h \left(\| \mathbf {u} \| \frac {\mathbf {z} _ {i}}{\gamma}\right)}{\| \mathbf {u} \|} \right\}, \end{array}\tag{3.35}
$$

where the first inequality follows by the convexity of h and by the fact that $\| \mathbf { u } \| _ { \gamma } ^ { \mathbf { \underline { { z } } } _ { i } } \in$ $B _ { \| \cdot \| } [ 0 , \gamma ] \subseteq D \subseteq$ dom(h). By (3.33), 

$$
\lim _ {\mathbf {u} \to \mathbf {0}} \frac {h \left(\| \mathbf {u} \| \frac {\mathbf {z} _ {i}}{\gamma}\right)}{\| \mathbf {u} \|} = \lim _ {\| \mathbf {u} \| \to \mathbf {0}} \frac {h \left(\| \mathbf {u} \| \frac {\mathbf {z} _ {i}}{\gamma}\right)}{\| \mathbf {u} \|} = \lim _ {\alpha \to 0 ^ {+}} \frac {h \left(\alpha \frac {\mathbf {z} _ {i}}{\gamma}\right)}{\alpha} = 0,
$$

which, combined with (3.35), implies that $\frac { h ( \mathbf { u } ) } { \| \mathbf { u } \| }  0$ as $\mathbf u \to \mathbf 0$ , proving the desired result. 

Example 3.34 (subdiferential of the $l _ { 2 } { \bf - n o r m } )$ . Let $f : \mathbb { R } ^ { n }  \mathbb { R }$ be given by $f ( \mathbf { x } ) = \| \mathbf { x } \| _ { 2 }$ . Then the subdiferential set of $f$ at $\mathbf { x } = \mathbf { 0 }$ was already computed in Example 3.3. When $\mathbf { x } \neq \mathbf { 0 }$ , the function is diferentiable with gradient $\frac { \mathbf { x } } { \| \mathbf { x } \| _ { 2 } }$ . Thus, using Theorem 3.33, we can summarize and write the subdiferential set as 

$$
\partial f (\mathbf {x}) = \left\{ \begin{array}{l l} \left\{\frac {\mathbf {x}}{\| \mathbf {x} \| _ {2}} \right\}, & \mathbf {x} \neq \mathbf {0}, \\ B _ {\| \cdot \| _ {2}} [ \mathbf {0}, 1 ], & \mathbf {x} = \mathbf {0}. \end{array} \right.
$$

In particular, when considering the case $n = 1$ , we obtain that for the one-dimensional function $g ( x ) = \vert x \vert$ , we have 

$$
\partial g (x) = \left\{ \begin{array}{l l} \{\operatorname{sgn} (x) \}, & x \neq 0, \\ [ - 1, 1 ], & x = 0. \end{array} \right.
$$

## 3.4 Computing Subgradients

This section establishes several useful calculus rules for subgradients and subdiferentials. Some of the results are “weak results” (rules for computing some of the subgradients in the subdiferential set), and some are “strong” (full characterization of the subdiferential set). 

## 3.4.1 Multiplication by a Positive Scalar

Theorem 3.35. Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper function and let $\alpha > 0$ . Then for any $\mathbf { x } \in \operatorname { d o m } ( f )$ 

$$
\partial (\alpha f) (\mathbf {x}) = \alpha \partial f (\mathbf {x}).
$$

Proof. We have that $\mathbf { g } \in \partial f ( \mathbf { x } )$ if and only if 

$$
f (\mathbf {y}) \geq f (\mathbf {x}) + \langle \mathbf {g}, \mathbf {y} - \mathbf {x} \rangle \text {   for   any   } \mathbf {y} \in \operatorname{dom} (f).
$$

Multiplying the inequality by $\alpha > 0$ , we can conclude that the above inequality holds if and only if 

$$
\alpha f (\mathbf {y}) \geq \alpha f (\mathbf {x}) + \langle \alpha \mathbf {g}, \mathbf {y} - \mathbf {x} \rangle \text {   for   any   } \mathbf {y} \in \operatorname{dom} (\alpha f),\tag{3.36}
$$

where we used the obvious fact that dom $( \alpha f ) = \mathrm { d o m } ( f )$ . The statement (3.36) is equivalent to the relation αg $\in \partial ( \alpha f ) ( \mathbf { x } )$ □ 

## 3.4.2 Summation

The following result contains both weak and strong results on the subdiferential set of a sum of functions. The weak result is also “weak” in the sense that its proof only requires the definition of the subgradient. The strong result utilizes the max formula. 

Theorem 3.36. Let $f _ { 1 } , f _ { 2 } : \mathbb { E }  ( - \infty , \infty ]$ be proper convex functions, and let $\mathbf { x } \in \operatorname { d o m } ( f _ { 1 } ) \cap \operatorname { d o m } ( f _ { 2 } )$ 

(a) The following inclusion holds: 

$$
\partial f _ {1} (\mathbf {x}) + \partial f _ {2} (\mathbf {x}) \subseteq \partial (f _ {1} + f _ {2}) (\mathbf {x}).
$$

(b) If x ∈ int(dom(f<sub>1</sub>)) ∩ int(dom(f<sub>2</sub>)), then 

$$
\partial (f _ {1} + f _ {2}) (\mathbf {x}) = \partial f _ {1} (\mathbf {x}) + \partial f _ {2} (\mathbf {x}).
$$

Proof. (a) Let $\mathbf { g } \in \partial f _ { 1 } ( \mathbf { x } ) + \partial f _ { 2 } ( \mathbf { x } )$ . Then there exist ${ \bf g } _ { 1 } \in \partial f _ { 1 } ( { \bf x } )$ and $\mathbf { g } _ { 2 } \in \partial f _ { 2 } ( \mathbf { x } )$ such that $\mathbf { g } = \mathbf { g } _ { 1 } + \mathbf { g } _ { 2 }$ . By the definition of $\mathbf { g } _ { 1 }$ and $\mathbf { g } _ { 2 }$ , it follows that for any $\mathbf { y } \in \operatorname { d o m } ( f _ { 1 } ) \cap \operatorname { d o m } ( f _ { 2 } )$ 

$$
\begin{array}{l} f _ {1} (\mathbf {y}) \geq f _ {1} (\mathbf {x}) + \langle \mathbf {g} _ {1}, \mathbf {y} - \mathbf {x} \rangle , \\ f _ {2} (\mathbf {y}) \geq f _ {2} (\mathbf {x}) + \langle \mathbf {g} _ {2}, \mathbf {y} - \mathbf {x} \rangle . \end{array}
$$

Summing the two inequalities, we obtain that for any $\mathbf { y } \in \operatorname { d o m } ( f _ { 1 } ) \cap \operatorname { d o m } ( f _ { 2 } )$ 

$$
f _ {1} (\mathbf {y}) + f _ {2} (\mathbf {y}) \geq f _ {1} (\mathbf {x}) + f _ {2} (\mathbf {x}) + \langle \mathbf {g} _ {1} + \mathbf {g} _ {2}, \mathbf {y} - \mathbf {x} \rangle ,
$$

that is, ${ \bf g } = { \bf g } _ { 1 } + { \bf g } _ { 2 } \in \partial ( f _ { 1 } + f _ { 2 } ) ( { \bf x } )$ 

(b) Let $\textbf { d } \in \ \mathbb { E }$ and define $f \equiv f _ { 1 } + f _ { 2 }$ . Then since $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) ) \ =$ int $( \mathrm { d o m } ( f _ { 1 } ) ) \cap \mathrm { i n t } ( \mathrm { d o m } ( f _ { 2 } ) )$ , it follows by the max formula (Theorem 3.26) that 

$$
\sigma_ {\partial f (\mathbf {x})} (\mathbf {d}) = \max \left\{\langle \mathbf {g}, \mathbf {d} \rangle : \mathbf {g} \in \partial f (\mathbf {x}) \right\} = f ^ {\prime} (\mathbf {x}; \mathbf {d}).
$$

Using the additivity of the directional derivative and the max formula (again), we also obtain 

$$
\begin{array}{l} \sigma_ {\partial f (\mathbf {x})} (\mathbf {d}) = f ^ {\prime} (\mathbf {x}; \mathbf {d}) \\ \qquad = f _ {1} ^ {\prime} (\mathbf {x}; \mathbf {d}) + f _ {2} ^ {\prime} (\mathbf {x}; \mathbf {d}) \\ \qquad = \max \left\{\langle \mathbf {g} _ {1}, \mathbf {d} \rangle : \mathbf {g} _ {1} \in \partial f _ {1} (\mathbf {x}) \right\} + \max \left\{\langle \mathbf {g} _ {2}, \mathbf {d} \rangle : \mathbf {g} _ {2} \in \partial f _ {2} (\mathbf {x}) \right\} \\ \qquad = \max \left\{\langle \mathbf {g} _ {1} + \mathbf {g} _ {2}, \mathbf {d} \rangle : \mathbf {g} _ {1} \in \partial f _ {1} (\mathbf {x}), \mathbf {g} _ {2} \in \partial f _ {2} (\mathbf {x}) \right\} \\ \qquad = \sigma_ {\partial f _ {1} (\mathbf {x}) + \partial f _ {2} (\mathbf {x})} (\mathbf {d}). \end{array}
$$

By Theorems 3.9 and 3.14, $\partial f ( \mathbf { x } ) , \partial f _ { 1 } ( \mathbf { x } )$ , and $\partial f _ { 2 } ( \mathbf { x } )$ are nonempty compact convex sets, which also implies (simple exercise) that $\partial f _ { 1 } ( \mathbf { x } ) + \partial f _ { 2 } ( \mathbf { x } )$ is nonempty compact and convex. Finally, invoking Lemma 2.34, it follows that $\partial f ( \mathbf { x } ) = \partial f _ { 1 } ( \mathbf { x } ) +$ $\partial f _ { 2 } ( \mathbf { x } )$ 

Remark 3.37. Note that the proof of part (a) of Theorem 3.36 does not require a convexity assumption on $f _ { 1 }$ and $f _ { 2 }$ . 

A simple induction argument can be used to generalize the last result to an arbitrary number of functions. 

Corollary 3.38. Let $f _ { 1 } , f _ { 2 } , \ldots , f _ { m } : \mathbb { E } \to ( - \infty , \infty ]$ be proper convex functions, and let $\mathbf { x } \in \cap _ { i = 1 } ^ { m } \mathrm { d o m } ( f _ { i } )$ 

(a) (weak sum rule of subdiferential calculus) The following inclusion holds: 

$$
\sum_ {i = 1} ^ {m} \partial f _ {i} (\mathbf {x}) \subseteq \partial \left(\sum_ {i = 1} ^ {m} f _ {i}\right) (\mathbf {x}).
$$

(b) If $\mathbf { x } \in \cap _ { i = 1 } ^ { m }$ int(dom(f<sub>i</sub>)), then 

$$
\partial \left(\sum_ {i = 1} ^ {m} f _ {i}\right) (\mathbf {x}) = \sum_ {i = 1} ^ {m} \partial f _ {i} (\mathbf {x}).\tag{3.37}
$$

Another direct consequence is that if $f _ { 1 } , f _ { 2 } , \ldots , f _ { m }$ are real-valued, meaning that their domain is the entire space $\mathbb { E } ,$ then the sum formula (3.37) holds. 

Corollary 3.39. Let $f _ { 1 } , f _ { 2 } , \ldots , f _ { m } : \mathbb { E } \to \mathbb { R }$ be real-valued convex functions. Then for any $\mathbf { x } \in \mathbb { E }$ 

$$
\partial \left(\sum_ {i = 1} ^ {m} f _ {i}\right) (\mathbf {x}) = \sum_ {i = 1} ^ {m} \partial f _ {i} (\mathbf {x}).
$$

A result with a less restrictive assumption than the one in Corollary 3.38(b) states that if the intersection $\cap _ { i = 1 } ^ { m } \operatorname { r i } ( \operatorname { d o m } ( f _ { i } ) )$ is nonempty, then the sum formula is correct at any point. We state this result without a proof. 

Theorem 3.40 (sum rule of subdiferential calculus [108, Theorem 23.8]). Let $f _ { 1 } , f _ { 2 } , \ldots , f _ { m } : \mathbb { E } \to ( - \infty , \infty ]$ be proper convex functions, and assume that $\bigcap _ { i = 1 } ^ { m } \operatorname { r i } ( \operatorname { d o m } ( f _ { i } ) ) \neq \emptyset$ . Then for any $\mathbf { x } \in \mathbb { E }$ 

$$
\partial \left(\sum_ {i = 1} ^ {m} f _ {i}\right) (\mathbf {x}) = \sum_ {i = 1} ^ {m} \partial f _ {i} (\mathbf {x}).
$$

Example 3.41 (subdiferential set of the l -norm function—strong result). Consider the function $f : \mathbb { R } ^ { n } $ R given by $f ( \mathbf { x } ) = \| \mathbf { x } \| _ { 1 } = \textstyle \sum _ { i = 1 } ^ { n } | x _ { i } |$ . Then $\textstyle f = \sum _ { i = 1 } ^ { n } f _ { i } .$ , where $f _ { i } ( \mathbf { x } ) \equiv | x _ { i } |$ . We have (see also Example 3.34) 

$$
\partial f _ {i} (\mathbf {x}) = \left\{ \begin{array}{l l} \{\operatorname{sgn} (x _ {i}) \mathbf {e} _ {i} \}, & x _ {i} \neq 0, \\ [ - \mathbf {e} _ {i}, \mathbf {e} _ {i} ], & x _ {i} = 0. \end{array} \right.
$$

Thus, by Corollary 3.39, 

$$
\partial f (\mathbf {x}) = \sum_ {i = 1} ^ {n} \partial f _ {i} (\mathbf {x}) = \sum_ {i \in I _ {\neq} (\mathbf {x})} \operatorname{sgn} (x _ {i}) \mathbf {e} _ {i} + \sum_ {i \in I _ {0} (\mathbf {x})} [ - \mathbf {e} _ {i}, \mathbf {e} _ {i} ],
$$

where 

$$
I _ {\neq} (\mathbf {x}) = \{i: x _ {i} \neq 0 \}, I _ {0} (\mathbf {x}) = \{i: x _ {i} = 0 \},
$$

and hence 

$$
\partial f (\mathbf {x}) = \left\{\mathbf {z} \in \mathbb {R} ^ {n}: z _ {i} = \operatorname{sgn} (x _ {i}), i \in I _ {\neq} (\mathbf {x}), | z _ {j} | \leq 1, j \in I _ {0} (\mathbf {x}) \right\}.
$$

Example 3.42 (a subgradient of the $l _ { 1 } { \bf - n o r m }$ function—weak result). Using the formula for the subdiferential set described in Example 3.41, we can readily conclude that 

$$
\operatorname{sgn} (\mathbf {x}) \in \partial f (\mathbf {x}).
$$

## 3.4.3 Afine Transformation

The following theorem states strong and weak results on the subdiferential set of a composition of a convex function with an afine transformation. 

Theorem 3.43. Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper convex function and $\mathcal { A } : \mathbb { V } \to \mathbb { E }$ be a linear transformation. Let $h ( \mathbf { x } ) = f ( A ( \mathbf { x } ) + \mathbf { b } )$ with $\mathbf { b } \in \mathbb { E }$ . Assume that h is proper, meaning that 

$$
\operatorname{dom} (h) = \left\{\mathbf {x} \in \mathbb {V}: \mathcal {A} (\mathbf {x}) + \mathbf {b} \in \operatorname{dom} (f) \right\} \neq \emptyset .
$$

(a) (weak afine transformation rule of subdiferential calculus) For any $\mathbf { x } \in \mathrm { d o m } ( h )$ 2 

$$
\mathcal {A} ^ {T} (\partial f (\mathcal {A} (\mathbf {x}) + \mathbf {b})) \subseteq \partial h (\mathbf {x}).
$$

(b) (afine transformation rule of subdiferential calculus) $I f \mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( h ) )$ and $\mathcal { A } ( \mathbf { x } ) + \mathbf { b } \in \mathrm { i n t } ( \mathrm { d o m } ( f ) )$ , then 

$$
\partial h (\mathbf {x}) = \mathcal {A} ^ {T} (\partial f (\mathcal {A} (\mathbf {x}) + \mathbf {b})).
$$

Proof. (a) Let $\mathbf { x } \in \mathrm { d o m } ( h )$ and assume that $\mathbf { g } \in \mathcal { A } ^ { T } ( \partial f ( \mathcal { A } ( \mathbf { x } ) + \mathbf { b } ) )$ . Then there exists d $\in \mathbb { E } ^ { * }$ for which $\mathbf { g } = \mathcal { A } ^ { T } ( \mathbf { d } )$ , where 

$$
\mathbf {d} \in \partial f (\mathcal {A} (\mathbf {x}) + \mathbf {b}).\tag{3.38}
$$

For any $\mathbf { y } \in \mathrm { d o m } ( h )$ , we have $\mathcal { A } ( \mathbf { y } ) + \mathbf { b } \in \mathrm { d o m } ( f )$ , and hence, by (3.38), 

$$
f (\mathcal {A} (\mathbf {y}) + \mathbf {b}) \geq f (\mathcal {A} (\mathbf {x}) + \mathbf {b}) + \langle \mathbf {d}, \mathcal {A} (\mathbf {y}) + \mathbf {b} - \mathcal {A} (\mathbf {x}) - \mathbf {b} \rangle ,
$$

and therefore 

$$
h (\mathbf {y}) \geq h (\mathbf {x}) + \langle \mathcal {A} ^ {T} (\mathbf {d}), \mathbf {y} - \mathbf {x} \rangle \text {   for   all   } \mathbf {y} \in \mathrm{dom} (h).
$$

Hence, $\mathbf { g } = \mathcal { A } ^ { T } ( \mathbf { d } ) \in \partial h ( \mathbf { x } )$ , proving that $\mathcal { A } ^ { T } ( \partial f ( \mathcal { A } ( \mathbf { x } ) + \mathbf { b } ) ) \subseteq \partial h ( \mathbf { x } )$ 

(b) Since $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( h ) )$ , then for any d $\in \mathbb { V }$ , by the max formula (Theorem 3.26), 

$$
h ^ {\prime} (\mathbf {x}; \mathbf {d}) = \sigma_ {\partial h (\mathbf {x})} (\mathbf {d}).\tag{3.39}
$$

In addition, by the definition of the directional derivative, we have 

$$
\begin{array}{l} h ^ {\prime} (\mathbf {x}; \mathbf {d}) = \lim _ {\alpha \to 0 ^ {+}} \frac {h (\mathbf {x} + \alpha \mathbf {d}) - h (\mathbf {x})}{\alpha} \\ \quad = \lim _ {\alpha \to 0 ^ {+}} \frac {f (\mathcal {A} (\mathbf {x}) + \mathbf {b} + \alpha \mathcal {A} (\mathbf {d})) - f (\mathcal {A} (\mathbf {x}) + \mathbf {b})}{\alpha} \\ \quad = f ^ {\prime} (\mathcal {A} (\mathbf {x}) + \mathbf {b}; \mathcal {A} (\mathbf {d})), \end{array}
$$

which, combined with (3.39), yields 

$$
\sigma_ {\partial h (\mathbf {x})} (\mathbf {d}) = f ^ {\prime} (\mathcal {A} (\mathbf {x}) + \mathbf {b}; \mathcal {A} (\mathbf {d})).
$$

Therefore, using the max formula again and the assumption that $\mathcal { A } ( \mathbf { x } ) + \mathbf { b } \ \in$ $\operatorname { i n t } ( \operatorname { d o m } ( f ) )$ ), we obtain that 

$$
\begin{array}{l} \sigma_ {\partial h (\mathbf {x})} (\mathbf {d}) = f ^ {\prime} (\mathcal {A} (\mathbf {x}) + \mathbf {b}; \mathcal {A} (\mathbf {d})) \\ \qquad = \underset {\mathbf {g}} {\max} \left\{\langle \mathbf {g}, \mathcal {A} (\mathbf {d}) \rangle : \mathbf {g} \in \partial f (\mathcal {A} (\mathbf {x}) + \mathbf {b}) \right\} \\ \qquad = \underset {\mathbf {g}} {\max} \left\{\langle \mathcal {A} ^ {T} (\mathbf {g}), \mathbf {d} \rangle : \mathbf {g} \in \partial f (\mathcal {A} (\mathbf {x}) + \mathbf {b}) \right\} \\ \qquad = \underset {\mathbf {g}} {\max} \left\{\langle \tilde {\mathbf {g}}, \mathbf {d} \rangle : \tilde {\mathbf {g}} \in \mathcal {A} ^ {T} (\partial f (\mathcal {A} (\mathbf {x}) + \mathbf {b})) \right\} \\ \qquad = \sigma_ {\mathcal {A} ^ {T} (\partial f (\mathcal {A} (\mathbf {x}) + \mathbf {b}))} (\mathbf {d}). \end{array}
$$

Since $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( h ) )$ , it follows by Theorems 3.9 and 3.14 that $\partial h ( { \bf x } )$ is nonempty compact and convex. Similarly, since $\pmb { \mathcal { A } } ( \mathbf { x } ) + \mathbf { b } \in \mathrm { i n t } ( \mathrm { d o m } ( f ) )$ , the set $\partial f ( \mathcal { A } ( \mathbf { x } ) + \mathbf { b } )$ is nonempty, compact, and convex, which implies that $\mathcal { A } ^ { \acute { T } } ( \partial f ( \mathcal { A } ( \mathbf { x } ) + \mathbf { b } ) )$ is also nonempty, compact, and convex. Finally, invoking Lemma 2.34, we obtain that $\partial h ( \mathbf { x } ) = \mathcal { A } ^ { T } ( \partial f ( \mathcal { A } ( \mathbf { x } ) + \mathbf { b } ) )$ 

Example 3.44 (subdiferential of $\| \mathbf { A x } + \mathbf { b } \| _ { 1 } )$ . Let $f : \mathbb { R } ^ { n } $ R be the function given by $f ( \mathbf { x } ) = \| \mathbf { A x } + \mathbf { b } \| _ { 1 }$ , where $\mathbf { A } \in \mathbb { R } ^ { m \times n } , \mathbf { b } \in \mathbb { R } ^ { m }$ . Then $f ( \mathbf { x } ) = g ( \mathbf { A x } + \mathbf { b } )$ with $g : \mathbb { R } ^ { m }  \mathbb { R }$ given by $g ( \mathbf { y } ) \mathbf { \Psi } = \mathbf { \Psi } \| \mathbf { y } \| _ { 1 }$ . By the afine transformation rule of subdiferential calculus (Theorem $3 . 4 3 ( \mathrm { b } ) )$ , we have that 

$$
\partial f (\mathbf {x}) = \mathbf {A} ^ {T} \partial g (\mathbf {A x} + \mathbf {b}).\tag{3.40}
$$

Denote the ith row of A by $\mathbf { a } _ { i } ^ { T }$ and define 

$$
\begin{array}{r} I _ {\neq} (\mathbf {x}) = \{i: \mathbf {a} _ {i} ^ {T} \mathbf {x} + b _ {i} \neq 0 \}, \\ I _ {0} (\mathbf {x}) = \{i: \mathbf {a} _ {i} ^ {T} \mathbf {x} + b _ {i} = 0 \}. \end{array}
$$

In this terminology, by Example 3.41, 

$$
\partial g (\mathbf {A} \mathbf {x} + \mathbf {b}) = \sum_ {i \in I _ {\neq} (\mathbf {x})} \operatorname{sgn} \left(\mathbf {a} _ {i} ^ {T} \mathbf {x} + b _ {i}\right) \mathbf {e} _ {i} + \sum_ {i \in I _ {0} (\mathbf {x})} \left[ - \mathbf {e} _ {i}, \mathbf {e} _ {i} \right].
$$

Thus, by (3.40), 

$$
\begin{array}{l} \partial f (\mathbf {x}) = \mathbf {A} ^ {T} \partial g (\mathbf {A x} + \mathbf {b}) \\ \qquad = \sum_ {i \in I _ {\neq} (\mathbf {x})} \operatorname{sgn} (\mathbf {a} _ {i} ^ {T} \mathbf {x} + b _ {i}) \mathbf {A} ^ {T} \mathbf {e} _ {i} + \sum_ {i \in I _ {0} (\mathbf {x})} [ - \mathbf {A} ^ {T} \mathbf {e} _ {i}, \mathbf {A} ^ {T} \mathbf {e} _ {i} ]. \end{array}
$$

Using the relation ${ \mathbf { A } } ^ { T } { \mathbf { e } } _ { i } = { \mathbf { a } } _ { i }$ , we finally conclude that 

$$
\partial f (\mathbf {x}) = \sum_ {i \in I _ {\neq} (\mathbf {x})} \operatorname{sgn} \left(\mathbf {a} _ {i} ^ {T} \mathbf {x} + b _ {i}\right) \mathbf {a} _ {i} + \sum_ {i \in I _ {0} (\mathbf {x})} \left[ - \mathbf {a} _ {i}, \mathbf {a} _ {i} \right].
$$

The above is a strong result characterizing the entire subdiferential set. A weak result indicating one possible subgradient is 

$$
\mathbf {A} ^ {T} \mathrm{sgn} (\mathbf {A x} + \mathbf {b}) \in \partial f (\mathbf {x}).
$$

Example 3.45 (subdiferential of $\| \mathbf { A x } + \mathbf { b } \| _ { 2 } )$ . Let $f : \mathbb { R } ^ { n } $ R be the function $f ( \mathbf { x } ) = \| \mathbf { A x } + \mathbf { b } \| _ { 2 }$ , where $\mathbf { A } \in \mathbb { R } ^ { m \times n } , \mathbf { b } \in \mathbb { R } ^ { m }$ . Then $f ( \mathbf { x } ) = g ( \mathbf { A x } + \mathbf { b } )$ with $g : \mathbb { R } ^ { m }  \mathbb { R }$ given by $g ( \mathbf { y } ) = \| \mathbf { y } \| _ { 2 }$ . By Example 3.34, 

$$
\partial g (\mathbf {A x} + \mathbf {b}) = \left\{ \begin{array}{l l} \frac {\mathbf {A x} + \mathbf {b}}{\| \mathbf {A x} + \mathbf {b} \| _ {2}}, & \mathbf {A x} + \mathbf {b} \neq \mathbf {0}, \\ B _ {\| \cdot \| _ {2}} [ \mathbf {0}, 1 ], & \mathbf {A x} + \mathbf {b} = \mathbf {0}. \end{array} \right.
$$

Thus, by the afine transformation rule of subdiferential calculus (Theorem 3.43(b)), 

$$
\partial f (\mathbf {x}) = \mathbf {A} ^ {T} \partial g (\mathbf {A x} + \mathbf {b}) = \left\{ \begin{array}{l l} \frac {\mathbf {A} ^ {T} (\mathbf {A x} + \mathbf {b})}{\| \mathbf {A x} + \mathbf {b} \| _ {2}}, & \mathbf {A x} + \mathbf {b} \neq \mathbf {0}, \\ \mathbf {A} ^ {T} B _ {\| \cdot \| _ {2}} [ \mathbf {0}, 1 ], & \mathbf {A x} + \mathbf {b} = \mathbf {0}. \end{array} \right.
$$

Note that at a vector x satisfying $\mathbf { A } \mathbf { x } + \mathbf { b } = \mathbf { 0 }$ , the subdiferential set can be explicitly written as 

$$
\partial f (\mathbf {x}) = \mathbf {A} ^ {T} B _ {\| \cdot \| _ {2}} [ \mathbf {0}, 1 ] = \left\{\mathbf {A} ^ {T} \mathbf {y}: \| \mathbf {y} \| _ {2} \leq 1 \right\}.
$$

If a weak result is required, then since $\mathbf { 0 } \in B _ { \parallel \cdot \parallel _ { 2 } } [ \mathbf { 0 } , 1 ]$ , we can write $\mathbf { 0 } = \mathbf { A } ^ { T } \mathbf { 0 } \in$ $\partial f ( \mathbf { x } )$ for any x satisfying $\mathbf { A x } + \mathbf { b } = \mathbf { 0 }$ 

## 3.4.4 Composition

The derivative of a composition of diferentiable functions can be computed by using the well-known chain rule. We recall here the classical result on the derivative of the composition of two one-dimensional functions. The result is a small variation of the result from [112, Theorem 5.5]. 

Theorem 3.46. Suppose that f is continuous on $[ a , b ] \ ( a < b )$ and that $f _ { + } ^ { \prime } ( a )$ exists. Let g be a function defined on an open interval I which contains the range $o f f$ , and assume that g is diferentiable at $f ( a )$ . Then the function 

$$
h (t) = g (f (t)) \quad (a \leq t \leq b)
$$

is right diferentiable at $t = a$ and 

$$
h _ {+} ^ {\prime} (a) = g ^ {\prime} (f (a)) f _ {+} ^ {\prime} (a).
$$

Proof. 

$$
\begin{array}{l} h _ {+} ^ {\prime} (a) = \lim _ {t \to a ^ {+}} \frac {g (f (t)) - g (f (a))}{t - a} \\ = \lim _ {t \to a ^ {+}} \frac {g (f (t)) - g (f (a))}{f (t) - f (a)} \cdot \frac {f (t) - f (a)}{t - a} = g ^ {\prime} (f (a)) f _ {+} ^ {\prime} (a). \end{array}
$$

We will now show how the one-dimensional chain rule can be used with the help of the max formula (Theorem 3.26) to show a multidimensional version of the chain rule. 

Theorem 3.47 (chain rule of subdiferential calculus). Let $f : \mathbb { E } \to \mathbb { R }$ be a convex function and $g : \mathbb { R }  \mathbb { R }$ be a nondecreasing convex function. Let $\mathbf { x } \in \mathbb { E }$ , and suppose that g is diferentiable at the point $f ( \mathbf { x } )$ . Let $h = g \circ f$ . Then 

$$
\partial h (\mathbf {x}) = g ^ {\prime} (f (\mathbf {x})) \partial f (\mathbf {x}).
$$

Proof. For any $\mathbf { d } \in \mathbb { E }$ , define the following one-dimensional functions: 

$$
\begin{array}{l l} f _ {\mathbf {x}, \mathbf {d}} (t) = f (\mathbf {x} + t \mathbf {d}), & t \in \mathbb {R}, \\ h _ {\mathbf {x}, \mathbf {d}} (t) = h (\mathbf {x} + t \mathbf {d}), & t \in \mathbb {R}. \end{array}
$$

We have the following simple relation between $f _ { \mathbf { x } , \mathbf { d } }$ and $h _ { \mathbf { x } , \mathbf { d } }$ 

$$
h _ {\mathbf {x}, \mathbf {d}} (t) = h (\mathbf {x} + t \mathbf {d}) = g (f (\mathbf {x} + t \mathbf {d})) = g (f _ {\mathbf {x}, \mathbf {d}} (t)), \quad t \in \mathbb {R}.\tag{3.41}
$$

The function $f$ is convex by the premise of the theorem, and h is convex since it is a composition of a nondecreasing convex function with a convex function. Therefore, the directional derivatives of $f$ and $h$ exist in every direction (Theorem 3.21), and we have by the definition of the directional derivative that 

$$
(f _ {\mathbf {x}, \mathbf {d}}) _ {+} ^ {\prime} (0) = f ^ {\prime} (\mathbf {x}; \mathbf {d}),\tag{3.42}
$$

$$
(h _ {\mathbf {x}, \mathbf {d}}) _ {+} ^ {\prime} (0) = h ^ {\prime} (\mathbf {x}; \mathbf {d}).\tag{3.43}
$$

Since $h _ { \mathbf { x } , \mathbf { d } } = g \circ f _ { \mathbf { x } , \mathbf { d } } ~ ( \mathrm { b y ~ ( 3 . 4 1 ) } )$ $f _ { \mathbf { x } , \mathbf { d } }$ is right diferentiable at $0 ,$ and $g$ is diferentiable at $f _ { \mathbf { x } , \mathbf { d } } ( 0 ) = f ( \mathbf { x } )$ , it follows by the chain rule for one-dimensional functions (Theorem 3.46) that 

$$
(h _ {\mathbf {x}, \mathbf {d}}) _ {+} ^ {\prime} (0) = g ^ {\prime} (f (\mathbf {x})) (f _ {\mathbf {x}; \mathbf {d}}) _ {+} ^ {\prime} (0).
$$

Plugging (3.42) and (3.43) into the latter equality, we obtain 

$$
h ^ {\prime} (\mathbf {x}; \mathbf {d}) = g ^ {\prime} (f (\mathbf {x})) f ^ {\prime} (\mathbf {x}; \mathbf {d}).
$$

By the max formula (Theorem 3.26), since f and h are convex and $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) ) =$ int $( \mathrm { d o m } ( h ) ) = \mathbb { E }$ 2 

$$
h ^ {\prime} (\mathbf {x}; \mathbf {d}) = \sigma_ {\partial h (\mathbf {x})} (\mathbf {d}), f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \sigma_ {\partial f (\mathbf {x})} (\mathbf {d}),
$$

and hence 

$$
\sigma_ {\partial h (\mathbf {x})} (\mathbf {d}) = h ^ {\prime} (\mathbf {x}; \mathbf {d}) = g ^ {\prime} (f (\mathbf {x})) f ^ {\prime} (\mathbf {x}; \mathbf {d}) = g ^ {\prime} (f (\mathbf {x})) \sigma_ {\partial f (\mathbf {x})} (\mathbf {d}) = \sigma_ {g ^ {\prime} (f (\mathbf {x})) \partial f (\mathbf {x})} (\mathbf {d}),
$$

where the last equality is due to Lemma $2 . 2 4 ( \mathrm { c } )$ and the fact that $g ^ { \prime } ( f ( \mathbf { x } ) ) \geq 0$ Finally, by Theorems 3.9 and 3.14 the sets $\partial h ( \mathbf { x } ) , \partial f ( \mathbf { x } )$ are nonempty, closed, and convex, and thus by Lemma 2.34 

$$
\partial h (\mathbf {x}) = g ^ {\prime} (f (\mathbf {x})) \partial f (\mathbf {x}). \quad \square
$$

Example 3.48 (subdiferential of $\| \cdot \| _ { 1 } ^ { 2 } )$ . Consider the function $h : \mathbb { R } ^ { n } $ R given by $h ( \mathbf { x } ) = \| \mathbf { x } \| _ { 1 } ^ { 2 }$ , which can be written as the composition $h = g \circ f ;$ , where $f ( \mathbf { x } ) = \| \mathbf { x } \| _ { 1 }$ and $g ( t ) = [ t ] _ { + } ^ { 2 } = \operatorname* { m a x } \{ t , 0 \} ^ { 2 }$ . Both $f$ and $g$ are real-valued convex functions, and $g$ is nondecreasing and diferentiable over R with derivative $g ^ { \prime } ( t ) =$ $2 [ t ] _ { + }$ . Therefore, by the chain rule of subdiferential calculus (Theorem 3.47), for any $\mathbf { x } \in \mathbb { R } ^ { n }$ • 

$$
\partial h (\mathbf {x}) = g ^ {\prime} (f (\mathbf {x})) \partial f (\mathbf {x}) = 2 \left[ \| \mathbf {x} \| _ {1} \right] _ {+} \partial f (\mathbf {x}) = 2 \| \mathbf {x} \| _ {1} \partial f (\mathbf {x}).
$$

Using the general form of $\partial f ( \mathbf { x } )$ as derived in Example 3.41, we can write $\partial h ( { \bf x } )$ explicitly as follows: 

$$
\begin{array}{c} \partial h (\mathbf {x}) = 2 \| \mathbf {x} \| _ {1} \left\{\mathbf {z} \in \mathbb {R} ^ {n}: z _ {i} = \mathrm{sgn} (x _ {i}), i \in I _ {\neq} (\mathbf {x}), | z _ {j} | \leq 1, j \in I _ {0} (\mathbf {x}) \right\}, \\ \text { where } I _ {\neq} (\mathbf {x}) = \{i: x _ {i} \neq 0 \}, I _ {0} (\mathbf {x}) = \{i: x _ {i} = 0 \}. \end{array}
$$

Plugging $\mathbf { x } = \mathbf { 0 }$ into the above formula, we obtain that 

$$
\partial h (\mathbf {0}) = \{\mathbf {0} \}.
$$

Since h has a unique subgradient at $\mathbf { x } = \mathbf { 0 }$ , it follows by Theorem 3.33 that h is diferentiable at $\mathbf { x } = \mathbf { 0 }$ and $\nabla h ( \mathbf { 0 } ) = \mathbf { 0 }$ . Note that the function is obviously not diferentiable over $\mathbb { R } ^ { n }$ . For example, when $n = 2$ , the nondiferentiability points are $\{ ( x _ { 1 } , 0 ) ^ { T } : x _ { 1 } \neq 0 \} \cup \{ ( 0 , x _ { 2 } ) ^ { T } : \bar { x } _ { 2 } \neq 0 \}$ , as illustrated in Figure 3.3. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/06afcb51-43be-4db1-8b62-226fc4277ca3/0bded447e42c8a4b90d1eb2b537244f50cbcd735b4b4e561d8665423a4e9e174.jpg)



Figure 3.3. Surface plot of the function $f ( x _ { 1 } , x _ { 2 } ) = ( | x _ { 1 } | + | x _ { 2 } | ) ^ { 2 }$


Example 3.49 (subdiferential of $d _ { C } ( \cdot ) )$ . Suppose that E is a Euclidean space, and let $C \subseteq \mathbb { E }$ be a nonempty closed and convex set (see Example 2.20). The distance function $d _ { C }$ is convex. We will show that 

$$
\partial d _ {C} (\mathbf {x}) = \left\{ \begin{array}{l l} \left\{\frac {\mathbf {x} - P _ {C} (\mathbf {x})}{d _ {C} (\mathbf {x})} \right\}, & \mathbf {x} \notin C, \\ N _ {C} (\mathbf {x}) \cap B [ \mathbf {0}, 1 ], & \mathbf {x} \in C. \end{array} \right.
$$

By Example 3.31, we know that the function $\begin{array} { r } { \varphi _ { C } ( \mathbf { x } ) = \frac { 1 } { 2 } d _ { C } ^ { 2 } ( \mathbf { x } ) } \end{array}$ is diferentiable and 

$$
\partial \varphi_ {C} (\mathbf {x}) = \{\mathbf {x} - P _ {C} (\mathbf {x}) \}\tag{3.44}
$$

for any $\textbf { x } \in \ \mathbb { E }$ . Note that $\varphi _ { C } = g \circ d _ { C }$ , where $\begin{array} { r } { g ( t ) \ = \ \frac { 1 } { 2 } [ t ] _ { + } ^ { 2 } } \end{array}$ is a nonincreasing real-valued convex diferentiable function. Then by the chain rule of subdiferential calculus (Theorem 3.47), 

$$
\partial \varphi_ {C} (\mathbf {x}) = g ^ {\prime} (d _ {C} (\mathbf {x})) \partial d _ {C} (\mathbf {x}) = [ d _ {C} (\mathbf {x}) ] _ {+} \partial d _ {C} (\mathbf {x}) = d _ {C} (\mathbf {x}) \partial d _ {C} (\mathbf {x}).\tag{3.45}
$$

$\operatorname { I f } \mathbf { x } \notin C$ , then $d _ { C } ( \mathbf { x } ) \neq 0$ , and thus by (3.44) and (3.45), 

$$
\partial d _ {C} (\mathbf {x}) = \left\{\frac {\mathbf {x} - P _ {C} (\mathbf {x})}{d _ {C} (\mathbf {x})} \right\} \text {for any} \mathbf {x} \notin C.
$$

Since $\partial d _ { C } ( { \bf x } )$ is a singleton for any $\mathbf { x } \notin C ,$ , it follows in particular, by Theorem 3.33, that $d _ { C }$ is diferentiable at points outside $C .$ 

Now assume that $\mathbf { x } \in C$ . We will show that 

$$
\partial d _ {C} (\mathbf {x}) = N _ {C} (\mathbf {x}) \cap B [ \mathbf {0}, 1 ].
$$

Indeed, if $\mathbf { d } \in \partial d _ { C } ( \mathbf { x } )$ , then 

$$
d _ {C} (\mathbf {y}) \geq \langle \mathbf {d}, \mathbf {y} - \mathbf {x} \rangle \text {   for   any   } \mathbf {y} \in \mathbb {E}.\tag{3.46}
$$

This means in particular that for any $\mathbf { y } \in C$ 

$$
\langle \mathbf {d}, \mathbf {y} - \mathbf {x} \rangle \leq 0,
$$

that is, $\mathbf { d } \in N _ { C } ( \mathbf { x } )$ . In addition, taking $\mathbf { y } = \mathbf { x } + \mathbf { d }$ in (3.46), we get 

$$
\left\| \mathbf {d} \right\| ^ {2} = \langle \mathbf {d}, \mathbf {x} + \mathbf {d} - \mathbf {x} \rangle \leq d _ {C} (\mathbf {x} + \mathbf {d}) \leq \left\| \mathbf {x} + \mathbf {d} - \mathbf {x} \right\| = \| \mathbf {d} \|,
$$

which readily implies that $\| \mathbf { d } \| \leq 1$ . We conclude that $\partial d _ { C } ( \mathbf { x } ) \subseteq N _ { C } ( \mathbf { x } ) \cap B [ \mathbf { 0 } , 1 ]$ To show the reverse direction, take $\mathbf { d } \in N _ { C } ( \mathbf { x } ) \cap B [ \mathbf { 0 } , 1 ]$ . Then for any $\mathbf { y } \in \mathbb { E }$ 2 

$$
\langle \mathbf {d}, \mathbf {y} - \mathbf {x} \rangle = \langle \mathbf {d}, \mathbf {y} - P _ {C} (\mathbf {y}) \rangle + \langle \mathbf {d}, P _ {C} (\mathbf {y}) - \mathbf {x} \rangle .\tag{3.47}
$$

Since $\mathbf { d } \in N _ { C } ( \mathbf { x } )$ and $P _ { C } ( \mathbf { y } ) \in C$ , it follows by the definition of the normal cone that $\left. \mathbf { d } , P _ { C } ( \mathbf { y } ) - \mathbf { x } \right. \le 0$ , which, combined with (3.47), the Cauchy–Schwarz inequality, and the assertion that $\| \mathbf { d } \| \leq 1$ , implies that for any $\mathbf { y } \in \mathbb { E }$ 

$$
\langle \mathbf {d}, \mathbf {y} - \mathbf {x} \rangle \leq \langle \mathbf {d}, \mathbf {y} - P _ {C} (\mathbf {y}) \rangle \leq \| \mathbf {d} \| \cdot \| \mathbf {y} - P _ {C} (\mathbf {y}) \| \leq \| \mathbf {y} - P _ {C} (\mathbf {y}) \| = d _ {C} (\mathbf {y}),
$$

and hence d $\in \partial d _ { C } ( { \bf x } )$ ■ 

## 3.4.5 Maximization

The following result shows how to compute the subdiferential set of a maximum of a finite collection of convex functions. 

Theorem 3.50 (max rule of subdiferential calculus). Let $f _ { 1 } , f _ { 2 } , \ldots , f _ { m }$ $\mathbb { E } \to ( - \infty , \infty ]$ be proper convex functions, and define 

$$
f (\mathbf {x}) = \max \left\{f _ {1} (\mathbf {x}), f _ {2} (\mathbf {x}), \dots , f _ {m} (\mathbf {x}) \right\}.
$$

Let $\mathbf { x } \in \bigcap _ { i = 1 } ^ { m }$ int(dom(f<sub>i</sub>)). Then 

$$
\partial f (\mathbf {x}) = \mathrm{conv} \left(\cup_ {i \in I (\mathbf {x})} \partial f _ {i} (\mathbf {x})\right),
$$

where $I ( \mathbf { x } ) = \{ i \in \{ 1 , 2 , \dots , m \} : f _ { i } ( \mathbf { x } ) = f ( \mathbf { x } ) \}$ 

Proof. First note that $f ,$ as a maximum of convex functions, is convex (see Theorem 2.16(c)) and that by Corollary 3.25 for any $\mathbf { d } \in \mathbb { E }$ ， 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \max _ {i \in I (\mathbf {x})} f _ {i} ^ {\prime} (\mathbf {x}; \mathbf {d}).
$$

For the sake of simplicity of notation, we will assume that $I ( \mathbf { x } ) = \{ 1 , 2 , \ldots , k \}$ for some $k \in \{ 1 , 2 , \ldots , m \}$ . Now, using the max formula (Theorem 3.26), we obtain 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \max _ {i = 1, 2, \dots , k} f _ {i} ^ {\prime} (\mathbf {x}; \mathbf {d}) = \max _ {i = 1, 2, \dots , k} \max _ {\mathbf {g} _ {i} \in \partial f _ {i} (\mathbf {x})} \langle \mathbf {g} _ {i}, \mathbf {d} \rangle .\tag{3.48}
$$

Using the fact that for any $a _ { 1 } , a _ { 2 } , \ldots , a _ { k } \in \mathbb { R }$ the identity 

$$
\max \{a _ {1}, a _ {2}, \dots , a _ {k} \} = \max _ {\boldsymbol {\lambda} \in \Delta_ {k}} \sum_ {i = 1} ^ {k} \lambda_ {i} a _ {i}
$$

holds, we can continue (3.48) and write 

$$
\begin{array}{l} f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \max _ {\boldsymbol {\lambda} \in \Delta_ {k}} \left\{\sum_ {i = 1} ^ {k} \lambda_ {i} \max \{\langle \mathbf {g} _ {i}, \mathbf {d} \rangle : \mathbf {g} _ {i} \in \partial f _ {i} (\mathbf {x}) \} \right\} \\ = \max \left\{\left\langle \sum_ {i = 1} ^ {k} \lambda_ {i} \mathbf {g} _ {i}, \mathbf {d} \right\rangle : \mathbf {g} _ {i} \in \partial f _ {i} (\mathbf {x}), \boldsymbol {\lambda} \in \Delta_ {k} \right\} \\ = \max \left\{\langle \mathbf {g}, \mathbf {d} \rangle : \mathbf {g} \in \operatorname{conv} \left(\cup_ {i = 1} ^ {k} \partial f _ {i} (\mathbf {x})\right) \right\} \\ = \sigma_ {A} (\mathbf {d}), \end{array}
$$

where $A = \mathrm { c o n v } \left( \cup _ { i = 1 } ^ { k } \partial f _ { i } ( \mathbf { x } ) \right)$ . By the max formula (Theorem 3.26), since ${ \textbf { x } } \in$ int $\left( \operatorname { d o m } ( f ) \right)$ 

$$
f ^ {\prime} (\mathbf {x}; \mathbf {d}) = \sigma_ {\partial f (\mathbf {x})} (\mathbf {d}),
$$

and hence 

$$
\sigma_ {A} (\mathbf {d}) = \sigma_ {\partial f (\mathbf {x})} (\mathbf {d}) \text {   for   any   } \mathbf {d} \in \mathbb {E}.\tag{3.49}
$$

The set $\partial f ( \mathbf { x } )$ is closed and convex by Theorem 3.9, and since $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ , it is also nonempty and bounded by Theorem 3.14. Similarly, $\begin{array} { r } { \partial f _ { i } ( \mathbf { x } ) , i = 1 , 2 , \ldots , k . } \end{array}$ are nonempty and compact sets, and hence also is $\cup _ { i = 1 } ^ { k } \partial f _ { i } ( \mathbf { x } )$ . We can conclude that the set $A = \mathrm { c o n v } ( \cup _ { i = 1 } ^ { k } \partial f _ { i } ( \mathbf { x } ) )$ , as a convex hull of a nonempty compact set, is also nonempty and compact.<sup>17</sup> In addition, by the definition of the convex hull, A is convex. 

To conclude, both A and $\partial f ( \mathbf { x } )$ are nonempty closed and convex, and thus (3.49) implies by Lemma 2.34 that 

$$
\partial f (\mathbf {x}) = A,
$$

which is the desired result. 

Example 3.51 (subdiferential of the max function). Let $f : \mathbb { R } ^ { n } $ R be given by $f ( \mathbf { x } ) = \operatorname* { m a x } \{ x _ { 1 } , x _ { 2 } , . . . , x _ { n } \}$ . Obviously, $f ( \mathbf { x } ) = \operatorname* { m a x } \{ f _ { 1 } ( \mathbf { x } ) , f _ { 2 } ( \mathbf { x } ) , \ldots , f _ { n } ( \mathbf { x } ) \}$ , where $f _ { i } ( \mathbf { x } ) = x _ { i }$ , and hence $\partial f _ { i } ( \mathbf { x } ) = \{ \mathbf { e } _ { i } \}$ for any $i = 1 , 2 , \ldots , n$ . Denote 

$$
I (\mathbf {x}) = \{i: f (\mathbf {x}) = x _ {i} \}.
$$

Then by the max rule of subdiferential calculus (Theorem 3.50), 

$$
\partial f (\mathbf {x}) = \mathrm{conv} (\cup_ {i \in I (\mathbf {x})} \partial f _ {i} (\mathbf {x})) = \mathrm{conv} (\cup_ {i \in I (\mathbf {x})} \{\mathbf {e} _ {i} \}),
$$

and hence 

$$
\partial f (\mathbf {x}) = \left\{\sum_ {i \in I (\mathbf {x})} \lambda_ {i} \mathbf {e} _ {i}: \sum_ {i \in I (\mathbf {x})} \lambda_ {i} = 1, \lambda_ {j} \geq 0, j \in I (\mathbf {x}) \right\}.
$$

In particular, 

$$
\partial f (\alpha \mathbf {e}) = \Delta_ {n} \text { for   any } \alpha \in \mathbb {R}.
$$

Example 3.52 (subdiferential of the $l _ { \infty } \mathbf { - n o r m } )$ . Let $f : \mathbb { R } ^ { n }  \mathbb { R }$ be given by $f ( \mathbf { x } ) = \| \mathbf { x } \| _ { \infty }$ . There are two options. $\mathrm { I f } \ \mathbf { x } = \mathbf { 0 }$ , then by Example $3 . 3 \ \partial f ( \mathbf { 0 } )$ is the dual-norm unit ball, and in this case, 

$$
\partial f (\mathbf {0}) = B _ {\| \cdot \| _ {1}} [ \mathbf {0}, 1 ] = \{\mathbf {x} \in \mathbb {R} ^ {n}: \| \mathbf {x} \| _ {1} \leq 1 \}.
$$

Suppose that $\mathbf { x } \neq \mathbf { 0 }$ . Note that $f ( \mathbf { x } ) = \operatorname* { m a x } \{ f _ { 1 } ( \mathbf { x } ) , f _ { 2 } ( \mathbf { x } ) , \ldots , f _ { n } ( \mathbf { x } ) \}$ with $f _ { i } ( \mathbf { x } ) =$ $\left| x _ { i } \right|$ and set 

$$
I (\mathbf {x}) = \{i: | x _ {i} | = \| \mathbf {x} \| _ {\infty} \}.
$$

For any $i \in I ( \mathbf { x } )$ we have $x _ { i } \neq 0$ , and hence for any such $i , \partial f _ { i } ( \mathbf { x } ) = \{ \mathrm { s g n } ( x _ { i } ) \mathbf { e } _ { i } \}$ Thus, by the max rule of subdiferential calculus (Theorem 3.50), 

$$
\begin{array}{l} \partial f (\mathbf {x}) = \operatorname{conv} \left(\cup_ {i \in I (\mathbf {x})} \partial f _ {i} (\mathbf {x})\right) \\ \qquad = \operatorname{conv} \left(\cup_ {i \in I (\mathbf {x})} \{\operatorname{sgn} (x _ {i}) \mathbf {e} _ {i} \}\right) \\ \qquad = \left\{\sum_ {i \in I (\mathbf {x})} \lambda_ {i} \operatorname{sgn} (x _ {i}) \mathbf {e} _ {i}: \sum_ {i \in I (\mathbf {x})} \lambda_ {i} = 1, \lambda_ {j} \geq 0, j \in I (\mathbf {x}) \right\}. \end{array}
$$

To conclude, 

$$
\partial f (\mathbf {x}) = \left\{ \begin{array}{l l} B _ {\| \cdot \| _ {1}} [ \mathbf {0}, 1 ], & \mathbf {x} = \mathbf {0}, \\ \left\{\sum_ {i \in I (\mathbf {x})} \lambda_ {i} \mathrm{sgn} (x _ {i}) \mathbf {e} _ {i}: \sum_ {i \in I (\mathbf {x})} \lambda_ {i} = 1, \lambda_ {j} \geq 0, j \in I (\mathbf {x}) \right\}, & \mathbf {x} \neq \mathbf {0}. \end{array} \right.
$$

Example 3.53 (subdiferential of piecewise linear functions). Consider the piecewise linear function $f : \mathbb { R } ^ { n }  \mathbb { R }$ given by 

$$
f (\mathbf {x}) = \max _ {i = 1, 2, \dots , m} \{\mathbf {a} _ {i} ^ {T} \mathbf {x} + b _ {i} \},
$$

where $\mathbf { a } _ { i } \in \mathbb { R } ^ { n } , b _ { i } \in \mathbb { R } , i = 1 , 2 , \dots , m$ . The function $f$ can be written as $f ( \mathbf { x } ) =$ max $\{ f _ { 1 } ( \mathbf { x } ) , f _ { 2 } ( \mathbf { x } ) , \hdots , f _ { m } ( \mathbf { x } ) \}$ , where $f _ { i } ( \mathbf { x } ) = \mathbf { a } _ { i } ^ { T } \mathbf { x } + b _ { i } , i = 1 , 2 , \ldots , m$ . Obviously, $\partial f _ { i } ( \mathbf { x } ) = \{ \mathbf { a } _ { i } \}$ . Thus, by the max rule of subdiferential calculus (Theorem 3.50), 

$$
\begin{array}{c} \partial f (\mathbf {x}) = \left\{\sum_ {i \in I (\mathbf {x})} \lambda_ {i} \mathbf {a} _ {i}: \sum_ {i \in I (\mathbf {x})} \lambda_ {i} = 1, \lambda_ {j} \geq 0, j \in I (\mathbf {x}) \right\}, \\ \text {where} I (\mathbf {x}) = \{i: f (\mathbf {x}) = \mathbf {a} _ {i} ^ {T} \mathbf {x} + b _ {i} \}. \end{array}
$$

Example 3.54 (subdiferential of $\| \mathbf { A x } + \mathbf { b } \| _ { \infty } )$ . Consider the function $f ~ :$ : $\mathbb { R } ^ { n } \to \mathbb { R }$ given by $f ( \mathbf { x } ) = \| \mathbf { A } \mathbf { x } + \mathbf { b } \| _ { \infty }$ , where ${ \textbf { A } } \in$ R<sup>m×n</sup> and b $\in \mathbb { R } ^ { m }$ . Then $f ( \mathbf { x } ) = g ( \mathbf { A x } + \mathbf { b } )$ , where $g : \mathbb { R } ^ { m }  \mathbb { R }$ is given by $g ( \mathbf { y } ) = \| \mathbf { y } \| _ { \infty }$ . By Example 3.52, we have, for any $\mathbf { y } \in \mathbb { R } ^ { m }$ 

$$
\partial g (\mathbf {y}) = \left\{ \begin{array}{l l} B _ {\| \cdot \| _ {1}} [ \mathbf {0}, 1 ], & \mathbf {y} = \mathbf {0}, \\ \left\{\sum_ {i \in I (\mathbf {y})} \lambda_ {i} \mathrm{sgn} (y _ {i}) \mathbf {e} _ {i}: \sum_ {i \in I (\mathbf {y})} \lambda_ {i} = 1, \lambda_ {j} \geq 0, j \in I (\mathbf {y}) \right\}, & \mathbf {y} \neq \mathbf {0}, \end{array} \right.
$$

where 

$$
I (\mathbf {y}) = \{i \in \{1, 2, \dots , m \}: | y _ {i} | = \| \mathbf {y} \| _ {\infty} \}.
$$

We can thus use the afine transformation rule of subdiferential calculus (Theorem $3 . 4 3 ( \mathrm { b } ) )$ to conclude that $\partial f ( \mathbf { x } ) = \mathbf { A } ^ { T } \partial g ( \mathbf { A x } + \mathbf { b } )$ is given by 

$$
\partial f (\mathbf {x}) = \left\{ \begin{array}{l l} \mathbf {A} ^ {T} B _ {\| \cdot \| _ {1}} [ \mathbf {0}, 1 ], & \mathbf {A x} + \mathbf {b} = \mathbf {0}, \\ \left\{\sum_ {i \in I _ {\mathbf {x}}} \lambda_ {i} \mathrm{sgn} (\mathbf {a} _ {i} ^ {T} \mathbf {x} + b _ {i}) \mathbf {a} _ {i}: \sum_ {i \in I _ {\mathbf {x}}} \lambda_ {i} = 1, \lambda_ {j} \geq 0, j \in I _ {\mathbf {x}} \right\}, & \mathbf {A x} + \mathbf {b} \neq \mathbf {0}, \end{array} \right.
$$

where $\mathbf { a } _ { 1 } ^ { T } , \mathbf { a } _ { 2 } ^ { T } , \ldots , \mathbf { a } _ { m } ^ { T }$ are the rows of A and $I _ { \mathbf { x } } = I ( \mathbf { A x } + \mathbf { b } )$ 

When the index set is arbitrary (for example, infinite), it is still possible to prove a weak subdiferential calculus rule. 

Theorem 3.55 (weak maximum rule of subdiferential calculus). Let I be an arbitrary set, and suppose that any $i \in I$ is associated with a proper convex function $f _ { i } : \mathbb { E }  ( - \infty , \infty ]$ . Let 

$$
f (\mathbf {x}) = \max _ {i \in I} f _ {i} (\mathbf {x}).\tag{3.50}
$$

Then for any $\mathbf { x } \in \operatorname { d o m } ( f )$ 

$$
\operatorname{conv} \left(\cup_ {i \in I (\mathbf {x})} \partial f _ {i} (\mathbf {x})\right) \subseteq \partial f (\mathbf {x}),\tag{3.51}
$$

where $I ( \mathbf { x } ) = \{ i \in I : f ( \mathbf { x } ) = f _ { i } ( \mathbf { x } ) \}$ 

Proof. Let $\mathbf { x } \in \operatorname { d o m } ( f )$ . Then for any $\mathbf { z } \in \mathrm { d o m } ( f ) , i \in I ( \mathbf { x } )$ and $\mathbf { g } \in \partial f _ { i } ( \mathbf { x } )$ ， 

$$
f (\mathbf {z}) \geq f _ {i} (\mathbf {z}) \geq f _ {i} (\mathbf {x}) + \langle \mathbf {g}, \mathbf {z} - \mathbf {x} \rangle = f (\mathbf {x}) + \langle \mathbf {g}, \mathbf {z} - \mathbf {x} \rangle ,\tag{3.52}
$$

where the first inequality follows from (3.50), the second inequality is the subgradient inequality, and the equality is due to the assertion that $i \in I ( \mathbf { x } )$ . Since (3.52) holds for any $\mathbf { z } \in \operatorname { d o m } ( f )$ , we can conclude that $\mathbf { g } \in \partial f ( \mathbf { x } )$ . Thus, $\partial f _ { i } ( \mathbf { x } ) \subseteq \partial f ( \mathbf { x } )$ Finally, by the convexity of $\partial f ( \mathbf { x } )$ (Theorem 3.9), the result (3.51) follows. 

Example 3.56 (subgradient of $\begin{array} { r } { \lambda _ { \operatorname* { m a x } } ( \mathbf { A _ { 0 } } + \sum _ { i = 1 } ^ { m } x _ { i } \mathbf { A } _ { i } ) ) } \end{array}$ . Let $\mathbf { A } _ { 0 } , \mathbf { A } _ { 1 } , \dots , \mathbf { A } _ { m }$ $\in \mathbb { S } ^ { n }$ . Let $\mathcal { A } : \mathbb { R } ^ { m }  \mathbb { S } ^ { n }$ be the afine transformation given by 

$$
\mathcal {A} (\mathbf {x}) = \mathbf {A} _ {0} + \sum_ {i = 1} ^ {m} x _ {i} \mathbf {A} _ {i} \text {   for   any   } \mathbf {x} \in \mathbb {R} ^ {m}.
$$

Consider the function $f : \mathbb { R } ^ { m }  \mathbb { R }$ given by $f ( \mathbf { x } ) = \lambda _ { \mathrm { m a x } } ( \mathcal { A } ( \mathbf { x } ) )$ . Since for any $\mathbf { x } \in \mathbb { R } ^ { m }$ , 

$$
f (\mathbf {x}) = \max _ {\mathbf {y} \in \mathbb {R} ^ {n}: \| \mathbf {y} \| _ {2} = 1} \mathbf {y} ^ {T} \mathcal {A} (\mathbf {x}) \mathbf {y},\tag{3.53}
$$

and since the function 

$$
f _ {\mathbf {y}} (\mathbf {x}) \equiv \mathbf {y} ^ {T} \mathcal {A} (\mathbf {x}) \mathbf {y} = \mathbf {y} ^ {T} \mathbf {A} _ {0} \mathbf {y} + \sum_ {i = 1} ^ {m} (\mathbf {y} ^ {T} \mathbf {A} _ {i} \mathbf {y}) x _ {i}
$$

is afine in $\mathbf { x } ,$ and in particular convex in $\mathbf { x } ,$ it follows by Theorem $2 . 1 6 ( \mathrm { c } )$ that $f$ is convex. For a given $\mathbf { x } \in \mathbb { R } ^ { n }$ , the maximum in (3.53) is attained at normalized eigenvectors which correspond to the maximum eigenvalue of $\mathcal { A } ( \mathbf { x } )$ . Let $\tilde { \mathbf { y } }$ be such a normalized eigenvector. Then it follows by the weak maximum rule of subdiferential calculus (Theorem 3.55) that a subgradient of the afine function $f _ { \tilde { \mathbf { y } } }$ at x is a subgradient of $f$ at x, that is, 

$$
(\tilde {\mathbf {y}} ^ {T} \mathbf {A} _ {1} \tilde {\mathbf {y}}, \tilde {\mathbf {y}} ^ {T} \mathbf {A} _ {2} \tilde {\mathbf {y}}, \dots , \tilde {\mathbf {y}} ^ {T} \mathbf {A} _ {m} \tilde {\mathbf {y}}) ^ {T} \in \partial f (\mathbf {x}),\tag{3.54}
$$

where $\tilde { \mathbf { y } }$ is a normalized eigenvector of $\mathcal { A } ( \mathbf { x } )$ corresponding to the maximum eigenvalue. 

It is interesting to note that the result (3.54) can also be deduced by the afine transformation rule of subdiferential calculus (Theorem $3 . 4 3 ( \mathrm { b } ) )$ . Indeed, let $\tilde { \mathbf { y } }$ be as defined above. The function $f$ can be written as $f ( \mathbf { x } ) = g ( \mathcal { B } ( \mathbf { x } ) + \mathbf { A } _ { 0 } )$ , where $\begin{array} { r } { B ( \mathbf { x } ) \equiv \sum _ { i = 1 } ^ { m } x _ { i } \mathbf { A } _ { i } } \end{array}$ and $g ( \mathbf { X } ) \equiv \lambda _ { \operatorname* { m a x } } ( \mathbf { X } )$ . Then by the afine transformation rule of subdiferential calculus, 

$$
\partial f (\mathbf {x}) = \mathcal {B} ^ {T} (\partial g (\mathcal {B} (\mathbf {x}) + \mathbf {A} _ {0})).\tag{3.55}
$$

By Example 3.8, we know that $\tilde { \mathbf { y } } \tilde { \mathbf { y } } ^ { T } \in \partial g ( \mathcal { B } ( \mathbf { x } ) + \mathbf { A } _ { 0 } )$ , and hence, by (3.55), 

$$
\mathcal {B} ^ {T} (\tilde {\mathbf {y}} \tilde {\mathbf {y}} ^ {T}) \in \partial f (\mathbf {x}).
$$

The result now follows by noting that 

$$
\begin{array}{c} \mathcal {B} ^ {T} (\tilde {\mathbf {y}} \tilde {\mathbf {y}} ^ {T}) = (\mathrm{Tr} (\mathbf {A} _ {1} \tilde {\mathbf {y}} \tilde {\mathbf {y}} ^ {T}), \mathrm{Tr} (\mathbf {A} _ {2} \tilde {\mathbf {y}} \tilde {\mathbf {y}} ^ {T}), \ldots , \mathrm{Tr} (\mathbf {A} _ {m} \tilde {\mathbf {y}} \tilde {\mathbf {y}} ^ {T})) ^ {T} \\ = (\tilde {\mathbf {y}} ^ {T} \mathbf {A} _ {1} \tilde {\mathbf {y}}, \tilde {\mathbf {y}} ^ {T} \mathbf {A} _ {2} \tilde {\mathbf {y}}, \ldots , \tilde {\mathbf {y}} ^ {T} \mathbf {A} _ {m} \tilde {\mathbf {y}}) ^ {T}. \end{array}
$$

## 3.5 The Value Function

Consider the minimization problem 

$$
f _ {\text { opt }} = \min _ {\mathbf {x} \in X} \{f (\mathbf {x}): g _ {i} (\mathbf {x}) \leq 0, i = 1, 2, \dots , m, \mathbf {A x} + \mathbf {b} = \mathbf {0} \},\tag{3.56}
$$

where $f , g _ { 1 } , g _ { 2 } , \ldots , g _ { m } : \mathbb { E } \to ( - \infty , \infty ]$ are extended real-valued functions, $X \subseteq \mathbb { E }$ a nonempty set, $\mathbf { A } \in \mathbb { R } ^ { p \times n }$ , and b $\in \mathbb { R } ^ { p }$ . We will define the vector-valued function $\mathbf { g } : \mathbb { E }  \mathbb { R } ^ { m }$ as 

$$
\mathbf {g} (\mathbf {x}) \equiv (g _ {1} (\mathbf {x}), g _ {2} (\mathbf {x}), \dots , g _ {m} (\mathbf {x})) ^ {T},
$$

so that problem (3.56) can be rewritten more compactly as 

$$
\min _ {\mathbf {x} \in X} \{f (\mathbf {x}): \mathbf {g} (\mathbf {x}) \leq \mathbf {0}, \mathbf {A x} + \mathbf {b} = \mathbf {0} \}.
$$

The value function associated with problem (3.56) is the function $v : \mathbb { R } ^ { m } \times$ $\mathbb { R } ^ { p } \to [ - \infty , \infty ]$ given by 

$$
v (\mathbf {u}, \mathbf {t}) = \min _ {\mathbf {x} \in X} \left\{f (\mathbf {x}): g (\mathbf {x}) \leq \mathbf {u}, \mathbf {A x} + \mathbf {b} = \mathbf {t} \right\}.\tag{3.57}
$$

The feasible set of the minimization problem in (3.57) will be denoted by 

$$
C (\mathbf {u}, \mathbf {t}) = \left\{\mathbf {x} \in X: g (\mathbf {x}) \leq \mathbf {u}, \mathbf {A x} + \mathbf {b} = \mathbf {t} \right\},
$$

so that the value function can also be rewritten as $v ( \mathbf { u } , \mathbf { t } ) = \operatorname* { m i n } \{ f ( \mathbf { x } ) : \mathbf { x } \in C ( \mathbf { u } , \mathbf { t } ) \}$ By convention $v ( \mathbf { u , t } ) = \infty \mathrm { ~ i f ~ } C ( \mathbf { u , t } )$ is empty. A simple property of the value function $v ( \cdot , \cdot )$ is that it is monotone $\mathrm { w . r . t }$ . its first argument. 

Lemma 3.57 (monotonicity of the value function). Let $f , g _ { 1 } , g _ { 2 } , \ldots , g _ { m } : \mathbb { E } \to$ $( - \infty , \infty ]$ be extended real-valued functions, $X \subseteq \mathbb { E }$ a nonempty set, $\mathbf { A } \in \mathbb { R } ^ { p \times n }$ , and b $\in \mathbb { R } ^ { p }$ . Let v be the value function given in (3.57). Then 

$$
v (\mathbf {u}, \mathbf {t}) \geq v (\mathbf {w}, \mathbf {t}) f o r a n y \mathbf {u}, \mathbf {w} \in \mathbb {R} ^ {m}, \mathbf {t} \in \mathbb {R} ^ {p} s a t i s f y i n g \mathbf {u} \leq \mathbf {w}.
$$

Proof. Follows by the obvious fact that $C ( \mathbf { u } , \mathbf { t } ) \subseteq C ( \mathbf { w } , \mathbf { t } )$ whenever $\mathbf { u } \leq \mathbf { w }$ □ 

From now on we will also assume in addition that $f , g _ { 1 } , g _ { 2 } , \ldots , g _ { m }$ , and X are convex. With these additional assumptions, we now show that the value function is convex as long as it is proper. 

Lemma 3.58 (convexity of the value function). Let $f , g _ { 1 } , g _ { 2 } , \ldots , g _ { m } : \mathbb { E } \to$ $( - \infty , \infty ]$ be convex functions, $X \subseteq \mathbb { E }$ a nonempty convex set, $\textbf { A } \in \mathbb { R } ^ { p \times n }$ , and $\mathbf { b } \in \mathbb { R } ^ { p }$ . Suppose that the value function v given in (3.57) is proper. Then v is convex over $\mathbb { R } ^ { m } \times \mathbb { R } ^ { p }$ 

Proof. Let $( \mathbf { u } , \mathbf { t } ) , ( \mathbf { w } , \mathbf { s } ) \in \mathrm { d o m } ( v )$ and $\lambda \in [ 0 , 1 ]$ . Since v is proper, to prove the convexity, we need to show that 

$$
v (\lambda \mathbf {u} + (1 - \lambda) \mathbf {w}, \lambda \mathbf {t} + (1 - \lambda) \mathbf {s}) \leq \lambda v (\mathbf {u}, \mathbf {t}) + (1 - \lambda) v (\mathbf {w}, \mathbf {s}).
$$

By the definition of the value function v, there exist sequences $\{ \mathbf { x } _ { k } \} _ { k \geq 1 } , \{ \mathbf { y } _ { k } \} _ { k \geq 1 }$ satisfying 

$$
\mathbf {x} _ {k} \in C (\mathbf {u}, \mathbf {t}), \mathbf {y} _ {k} \in C (\mathbf {w}, \mathbf {s}), f (\mathbf {x} _ {k}) \rightarrow v (\mathbf {u}, \mathbf {t}), f (\mathbf {y} _ {k}) \rightarrow v (\mathbf {w}, \mathbf {s}) \text {   as   } k \rightarrow \infty .
$$

Since $\mathbf { x } _ { k } \in C ( \mathbf { u } , \mathbf { t } )$ and $\mathbf { y } _ { k } \in C ( \mathbf { w } , \mathbf { s } )$ , we have $\mathbf { g } ( \mathbf { x } _ { k } ) \leq \mathbf { u } , \mathbf { g } ( \mathbf { y } _ { k } ) \leq \mathbf { w }$ . Therefore, by the convexity of the components of $\mathbf { g } _ { \mathrm { : } }$ , 

$$
\mathbf {g} \left(\lambda \mathbf {x} _ {k} + (1 - \lambda) \mathbf {y} _ {k}\right) \leq \lambda \mathbf {g} \left(\mathbf {x} _ {k}\right) + (1 - \lambda) \mathbf {g} \left(\mathbf {y} _ {k}\right) \leq \lambda \mathbf {u} + (1 - \lambda) \mathbf {w}.\tag{3.58}
$$

Moreover, 

$$
\mathbf {A} \left(\lambda \mathbf {x} _ {k} + (1 - \lambda) \mathbf {y} _ {k}\right) + \mathbf {b} = \lambda \left(\mathbf {A x} _ {k} + \mathbf {b}\right) + (1 - \lambda) \left(\mathbf {A y} _ {k} + \mathbf {b}\right) = \lambda \mathbf {s} + (1 - \lambda) \mathbf {t}.\tag{3.59}
$$

Combining (3.58) and (3.59), we conclude that 

$$
\lambda \mathbf {x} _ {k} + (1 - \lambda) \mathbf {y} _ {k} \in C (\lambda \mathbf {u} + (1 - \lambda) \mathbf {w}, \lambda \mathbf {s} + (1 - \lambda) \mathbf {t}).\tag{3.60}
$$

By the convexity of $f _ { ; }$ 9 

$$
f (\lambda \mathbf {x} _ {k} + (1 - \lambda) \mathbf {y} _ {k}) \leq \lambda f (\mathbf {x} _ {k}) + (1 - \lambda) f (\mathbf {y} _ {k}).\tag{3.61}
$$

Since $\lambda f ( \mathbf { x } _ { k } ) + ( 1 - \lambda ) f ( \mathbf { y } _ { k } ) \to \lambda v ( \mathbf { u } , \mathbf { t } ) + ( 1 - \lambda ) v ( \mathbf { w } , \mathbf { s } ) { \mathrm { ~ a s ~ } } k \to \infty$ , by (3.61) we have 

$$
\liminf _ {k \to \infty} f (\lambda \mathbf {x} _ {k} + (1 - \lambda) \mathbf {y} _ {k}) \leq \lambda v (\mathbf {u}, \mathbf {t}) + (1 - \lambda) v (\mathbf {w}, \mathbf {s}).\tag{3.62}
$$

Finally, since (3.60) holds, by the definition of v, for all $k .$ 

$$
v (\lambda \mathbf {u} + (1 - \lambda) \mathbf {w}, \lambda \mathbf {t} + (1 - \lambda) \mathbf {s}) \leq f (\lambda \mathbf {x} _ {k} + (1 - \lambda) \mathbf {y} _ {k}),
$$

and hence 

$$
v (\lambda \mathbf {u} + (1 - \lambda) \mathbf {w}, \lambda \mathbf {t} + (1 - \lambda) \mathbf {s}) \leq \operatorname * {l i m i n f} _ {k \to \infty} f (\lambda \mathbf {x} _ {k} + (1 - \lambda) \mathbf {y} _ {k}),
$$

which, combined with (3.62), yields the inequality 

$$
v (\lambda \mathbf {u} + (1 - \lambda) \mathbf {w}, \lambda \mathbf {t} + (1 - \lambda) \mathbf {s}) \leq \lambda v (\mathbf {u}, \mathbf {t}) + (1 - \lambda) v (\mathbf {w}, \mathbf {s}),
$$

establishing the convexity of v. 

The dual objective function $q : \mathbb { R } _ { + } ^ { m } \times \mathbb { R } ^ { q } \to [ - \infty , \infty )$ of problem (3.56) is 

$$
q (\mathbf {y}, \mathbf {z}) = \min _ {\mathbf {x} \in X} \left\{L (\mathbf {x}; \mathbf {y}, \mathbf {z}) = f (\mathbf {x}) + \mathbf {y} ^ {T} \mathbf {g} (\mathbf {x}) + \mathbf {z} ^ {T} (\mathbf {A x} + \mathbf {b}) \right\}, \mathbf {y} \in \mathbb {R} _ {+} ^ {m}, \mathbf {z} \in \mathbb {R} ^ {p}.
$$

The dual problem consists of maximizing q on its efective domain given by 

$$
\operatorname{dom} (- q) = \left\{\left(\mathbf {y}, \mathbf {z}\right) \in \mathbb {R} _ {+} ^ {m} \times \mathbb {R} ^ {p}: q (\mathbf {y}, \mathbf {z}) > - \infty \right\}.
$$

The dual problem 

$$
q _ {\text { opt }} = \max _ {\mathbf {y} \in \mathbb {R} _ {+} ^ {m}, \mathbf {z} \in \mathbb {R} ^ {p}} \left\{q (\mathbf {y}, \mathbf {z}): (\mathbf {y}, \mathbf {z}) \in \operatorname{dom} (- q) \right\}\tag{3.63}
$$

is convex in the sense that it consists of maximizing the concave function q over the convex feasible set dom $( - q )$ . We are now ready to show the main result of this section, which is a relation between the subdiferential set of the value function at the zeros vector and the set of optimal solutions of the dual problem. The result is established under the assumption that strong duality holds, meaning under the assumptions that the optimal values of the primal and dual problems are finite and equal $( f _ { \mathrm { o p t } } = q _ { \mathrm { o p t } } )$ and the optimal set of the dual problem is nonempty. By the strong duality theorem stated as Theorem A.1 in the appendix, it follows that these assumptions are met if the optimal value of problem (3.56) is finite, and if there exists a feasible solution x¯ satisfying ${ \bf g } ( \bar { \bf x } ) < { \bf 0 }$ and a vector $\hat { \mathbf { x } } \in \operatorname { r i } ( X )$ satisfying $\mathbf { A } { \hat { \mathbf { x } } } + \mathbf { b } = \mathbf { 0 }$ 

Theorem 3.59 (characterization of the subdiferential of the value function at 0). Let $f , g _ { 1 } , g _ { 2 } , \ldots , g _ { m } : \mathbb { E } \to ( - \infty , \infty ]$ be convex functions, $X \subseteq \mathbb { E }$ a nonempty convex set, $\mathbf { A } \in \mathbb { R } ^ { p \times n }$ , and b $\in \mathbb { R } ^ { p }$ . Let v be the value function given by (3.57). Suppose that $f _ { \mathrm { o p t } } = q _ { \mathrm { o p t } } \in ( - \infty , \infty )$ and that the optimal set of the dual problem is nonempty. Then 

(a) v is proper and convex; 

(b) $( \mathbf { y } , \mathbf { z } )$ is an optimal solution of problem (3.63) if and only $i f - ( \mathbf { y } , \mathbf { z } ) \in \partial v ( \mathbf { 0 } , \mathbf { 0 } )$ 

Proof. Let $( \mathbf { y } , \mathbf { z } ) \in \mathrm { d o m } ( - q )$ be an optimal solution of the dual problem. Then (recalling that $v ( \mathbf { 0 } , \mathbf { 0 } ) = f _ { \mathrm { o p t } } )$ 

$$
L (\mathbf {x}; \mathbf {y}, \mathbf {z}) \geq \min _ {\mathbf {w} \in X} L (\mathbf {w}; \mathbf {y}, \mathbf {z}) = q (\mathbf {y}, \mathbf {z}) = q _ {\text {opt}} = f _ {\text {opt}} = v (\mathbf {0}, \mathbf {0}) \text {for all} \mathbf {x} \in X.
$$

Therefore, for any $\mathbf { x } \in C ( \mathbf { u } , \mathbf { t } )$ 

$$
\begin{array}{l} v (\mathbf {0}, \mathbf {0}) - \mathbf {y} ^ {T} \mathbf {u} - \mathbf {z} ^ {T} \mathbf {t} \leq L (\mathbf {x}; \mathbf {y}, \mathbf {z}) - \mathbf {y} ^ {T} \mathbf {u} - \mathbf {z} ^ {T} \mathbf {t} \\ \qquad = f (\mathbf {x}) + \mathbf {y} ^ {T} \mathbf {g} (\mathbf {x}) + \mathbf {z} ^ {T} (\mathbf {A x} + \mathbf {b}) - \mathbf {y} ^ {T} \mathbf {u} - \mathbf {z} ^ {T} \mathbf {t} \\ \qquad = f (\mathbf {x}) + \mathbf {y} ^ {T} (\mathbf {g} (\mathbf {x}) - \mathbf {u}) + \mathbf {z} ^ {T} (\mathbf {A x} + \mathbf {b} - \mathbf {t}) \\ \qquad \leq f (\mathbf {x}), \end{array}
$$

where the last inequality follows from the facts that $\mathbf { g } ( \mathbf { x } ) \leq \mathbf { u } , \mathbf { y } \geq \mathbf { 0 }$ , and $\mathbf { A } \mathbf { x } + \mathbf { b } =$ t. We thus obtained the bound 

$$
f (\mathbf {x}) \geq v (\mathbf {0}, \mathbf {0}) - \mathbf {y} ^ {T} \mathbf {u} - \mathbf {z} ^ {T} \mathbf {t} \text {   for   any   } \mathbf {x} \in C (\mathbf {u}, \mathbf {t}).
$$

Minimizing the left-hand side w.r.t. $\mathbf { x } \in C ( \mathbf { u } , \mathbf { t } )$ yields 

$$
v (\mathbf {u}, \mathbf {t}) \geq v (\mathbf {0}, \mathbf {0}) - \mathbf {y} ^ {T} \mathbf {u} - \mathbf {z} ^ {T} \mathbf {t},\tag{3.64}
$$

which is equivalent to saying that $- ( \mathbf { y } , \mathbf { z } ) \ \in \ \partial v ( \mathbf { 0 } , \mathbf { 0 } )$ . We actually showed one direction of claim (b), as well as the properness of v since by (3.64), $v ( { \bf u } , { \bf t } ) > - \infty$ for any $( \mathbf { u } , \mathbf { t } ) \in \mathbb { R } ^ { m } \times \mathbb { R } ^ { p }$ , and by the premise of the theorem, $v ( \mathbf { 0 } , \mathbf { 0 } ) = f _ { \mathrm { o p t } } < \infty$ Invoking Lemma 3.58, it follows that v is convex, establishing claim (a). 

All that is left is to show the reverse direction of claim (b). Assume that $- ( \mathbf { y } , \mathbf { z } ) \in \partial v ( \mathbf { 0 } , \mathbf { 0 } )$ , meaning that 

$$
v (\mathbf {u}, \mathbf {t}) \geq v (\mathbf {0}, \mathbf {0}) - \mathbf {y} ^ {T} \mathbf {u} - \mathbf {z} ^ {T} \mathbf {t} \text {   for   any   } (\mathbf {u}, \mathbf {t}) \in \mathbb {R} ^ {m} \times \mathbb {R} ^ {p}.\tag{3.65}
$$

Let $\mathbf { x } \in X$ . Then 

$$
f (\mathbf {x}) \geq v (\mathbf {g} (\mathbf {x}), \mathbf {A x} + \mathbf {b}) \stackrel {(3. 6 5)} {\geq} v (\mathbf {0}, \mathbf {0}) - \mathbf {y} ^ {T} \mathbf {g} (\mathbf {x}) - \mathbf {z} ^ {T} (\mathbf {A x} + \mathbf {b}).
$$

Therefore, 

$$
v (\mathbf {0}, \mathbf {0}) \leq f (\mathbf {x}) + \mathbf {y} ^ {T} \mathbf {g} (\mathbf {x}) + \mathbf {z} ^ {T} (\mathbf {A x} + \mathbf {b}) = L (\mathbf {x}; \mathbf {y}, \mathbf {z}) \text {   for   any   } \mathbf {x} \in X.
$$

Minimizing the right-hand side w.r.t. $\mathbf { x } \in X$ yields 

$$
v (\mathbf {0}, \mathbf {0}) \leq \min _ {\mathbf {x} \in X} L (\mathbf {x}; \mathbf {y}, \mathbf {z}) = q (\mathbf {y}, \mathbf {z}).\tag{3.66}
$$

Let $j \in \{ 1 , 2 , \dots , m \}$ . Plugging $\mathbf { u } = \mathbf { e } _ { j } , \mathbf { t } = \mathbf { 0 }$ into (3.65), we obtain 

$$
y _ {j} \geq v (\mathbf {0}, \mathbf {0}) - v (\mathbf {e} _ {j}, \mathbf {0}) \geq 0,
$$

where the second inequality follows from the monotonicity property of the value function stated in Lemma 3.57. We thus obtained that $\mathbf { y } \geq \mathbf { 0 }$ , and we can consequently write using (3.66) 

$$
q _ {\mathrm{opt}} = f _ {\mathrm{opt}} = v (\mathbf {0}, \mathbf {0}) \leq q (\mathbf {y}, \mathbf {z}) \leq q _ {\mathrm{opt}},
$$

showing that $q ( \mathbf { y } , \mathbf { z } ) = q _ { \mathrm { o p t } }$ , meaning that $( \mathbf { y } , \mathbf { z } )$ is an optimal solution of the dual problem. 

Theorem 3.59 can be used to prove a result concerning an optimality measure of problem (3.56). Consider the following expression: 

$$
D (\mathbf {x}) \equiv f (\mathbf {x}) - f _ {\mathrm{opt}} + \rho_ {1} \| [ \mathbf {g} (\mathbf {x}) ] _ {+} \| _ {2} + \rho_ {2} \| \mathbf {A x} + \mathbf {b} \| _ {2}.
$$

Now assume that 

$$
D (\tilde {\mathbf {x}}) \leq \delta\tag{3.67}
$$

for some $\tilde { \mathbf { x } } \in X$ and a small $\delta > 0$ . The question that now arises is whether (3.67) implies that the expressions $f ( \tilde { \mathbf { x } } ) - f _ { \mathrm { o p t } }$ as well as $| | [ \mathbf { g } ( \tilde { \mathbf { x } } ) ] _ { + } | | _ { 2 }$ and $\| \mathbf { A } \tilde { \mathbf { x } } + \mathbf { b } \| _ { 2 }$ are also “small” in the sense that they are smaller than a constant times δ. In general, the answer is no. The vector $\tilde { \bf x }$ is not guaranteed to be feasible, and therefore, in principle, $f ( \tilde { \mathbf { x } } ) - f _ { \mathrm { o p t } }$ might be very small (and negative), and $| | [ \mathbf { g } ( \tilde { \mathbf { x } } ) ] _ { + } | | _ { 2 }$ and $\| \mathbf { A } \tilde { \mathbf { x } } + \mathbf { b } \| _ { 2 }$ can be very large. However, we will show in the next theorem that if $\rho _ { 1 }$ and $\rho _ { 2 }$ are chosen to be large enough, then under the setting of Theorem 3.59, such a conclusion can be drawn. 

Theorem 3.60.<sup>19</sup> Let $f , g _ { 1 } , g _ { 2 } , \ldots , g _ { m } : \mathbb { E } \to ( - \infty , \infty ]$ be convex functions, $X \subseteq \mathbb { E }$ a nonempty convex set, $\mathbf { A } \in \mathbb { R } ^ { p \times n }$ , and b $\in \mathbb { R } ^ { p }$ . Let $f _ { \mathrm { o p t } }$ and $q _ { \mathrm { o p t } }$ be the optimal values of the primal and dual problems (3.56) and (3.63), respectively. Suppose that $f _ { \mathrm { o p t } } = q _ { \mathrm { o p t } } \in ( - \infty , \infty )$ and that the optimal set of the dual problem is nonempty. Let $\left( \mathbf { y } ^ { * } , \mathbf { z } ^ { * } \right)$ be an optimal solution of the dual problem. Assume that ${ \tilde { \mathbf { x } } } \in X$ satisfies 

$$
f (\tilde {\mathbf {x}}) - f _ {\mathrm{opt}} + \rho_ {1} \| [ \mathbf {g} (\tilde {\mathbf {x}}) ] _ {+} \| _ {2} + \rho_ {2} \| \mathbf {A} \tilde {\mathbf {x}} + \mathbf {b} \| _ {2} \leq \delta ,\tag{3.68}
$$

where $\delta > 0$ and $\rho _ { 1 } , \rho _ { 2 }$ are constants satisfying $\rho _ { 1 } \geq 2 \| \mathbf { y } ^ { * } \| _ { 2 } , \rho _ { 2 } \geq 2 \| \mathbf { z } ^ { * } \| _ { 2 }$ . Then 

$$
f (\tilde {\mathbf {x}}) - f _ {\mathrm{opt}} \leq \delta ,
$$

$$
\| [ \mathbf {g} (\tilde {\mathbf {x}}) ] _ {+} \| _ {2} \leq \frac {2}{\rho_ {1}} \delta ,
$$

$$
\left\| \mathbf {A} \tilde {\mathbf {x}} + \mathbf {b} \right\| _ {2} \leq \frac {2}{\rho_ {2}} \delta .
$$

Proof. The inequality $f ( \tilde { \mathbf { x } } ) - f _ { \mathrm { o p t } } \leq \delta$ trivially follows from (3.68) and the fact that the expressions $\rho _ { 1 } \| [ \mathbf { g } ( \tilde { \mathbf { x } } ) ] _ { + } \| _ { 2 }$ and $\rho _ { 2 } \| \mathbf { A } \tilde { \mathbf { x } } + \mathbf { b } \| _ { 2 }$ are nonnegative. 

Define the function 

$$
v (\mathbf {u}, \mathbf {t}) = \min _ {\mathbf {x} \in X} \{f (\mathbf {x}): \mathbf {g} (\mathbf {x}) \leq \mathbf {u}, \mathbf {A x} + \mathbf {b} = \mathbf {t} \}.
$$

Since $\left( \mathbf { y } ^ { * } , \mathbf { z } ^ { * } \right)$ is an optimal solution of the dual problem, it follows by Theorem 3.59 that $( - \mathbf { y } ^ { * } , - \mathbf { z } ^ { * } ) \in \partial v ( \mathbf { 0 } , \mathbf { 0 } )$ . Therefore, for any $( \mathbf { u } , \mathbf { t } ) \in \mathrm { d o m } ( v )$ 2 

$$
v (\mathbf {u}, \mathbf {t}) - v (\mathbf {0}, \mathbf {0}) \geq \langle - \mathbf {y} ^ {*}, \mathbf {u} \rangle + \langle - \mathbf {z} ^ {*}, \mathbf {t} \rangle .\tag{3.69}
$$

Plugging $\mathbf { u } = \tilde { \mathbf { u } } \equiv [ \mathbf { g } ( \tilde { \mathbf { x } } ) ] _ { + }$ and $\mathbf { t } = \tilde { \mathbf { t } } \equiv \mathbf { A } \tilde { \mathbf { x } } + \mathbf { b }$ into (3.69), while using the inequality $v ( \tilde { \mathbf { u } } , \tilde { \mathbf { t } } ) \leq f ( \tilde { \mathbf { x } } )$ and the equality $v ( { \bf 0 } , { \bf 0 } ) = f _ { \mathrm { o p t } } .$ , we obtain 

$$
\begin{array}{l} (\rho_ {1} - \| \mathbf {y} ^ {*} \| _ {2}) \| \tilde {\mathbf {u}} \| _ {2} + (\rho_ {2} - \| \mathbf {z} ^ {*} \| _ {2}) \| \tilde {\mathbf {t}} \| _ {2} = - \| \mathbf {y} ^ {*} \| _ {2} \| \tilde {\mathbf {u}} \| _ {2} - \| \mathbf {z} ^ {*} \| _ {2} \| \tilde {\mathbf {t}} \| _ {2} + \rho_ {1} \| \tilde {\mathbf {u}} \| _ {2} + \rho_ {2} \| \tilde {\mathbf {t}} \| _ {2} \\ \qquad \qquad \qquad \leq \langle - \mathbf {y} ^ {*}, \tilde {\mathbf {u}} \rangle + \langle - \mathbf {z} ^ {*}, \tilde {\mathbf {t}} \rangle + \rho_ {1} \| \tilde {\mathbf {u}} \| _ {2} + \rho_ {2} \| \tilde {\mathbf {t}} \| _ {2} \\ \qquad \qquad \qquad \leq v (\tilde {\mathbf {u}}, \tilde {\mathbf {t}}) - v (\mathbf {0}, \mathbf {0}) + \rho_ {1} \| \tilde {\mathbf {u}} \| _ {2} + \rho_ {2} \| \tilde {\mathbf {t}} \| _ {2} \\ \qquad \qquad \qquad \leq f (\tilde {\mathbf {x}}) - f _ {\mathrm{opt}} + \rho_ {1} \| \tilde {\mathbf {u}} \| _ {2} + \rho_ {2} \| \tilde {\mathbf {t}} \| _ {2} \\ \qquad \qquad \qquad \leq \delta . \end{array}
$$

Therefore, since both expressions $( \rho _ { 1 } - \| \mathbf { y } ^ { * } \| _ { 2 } ) \| \tilde { \mathbf { u } } \| _ { 2 }$ and $( \rho _ { 2 } - \| \mathbf { z } ^ { * } \| _ { 2 } ) \| \tilde { \mathbf { t } } \| _ { 2 }$ are nonnegative, it follows that 

$$
\begin{array}{r} (\rho_ {1} - \| \mathbf {y} ^ {*} \| _ {2}) \| \tilde {\mathbf {u}} \| _ {2} \leq \delta , \\ (\rho_ {2} - \| \mathbf {z} ^ {*} \| _ {2}) \| \tilde {\mathbf {t}} \| _ {2} \leq \delta , \end{array}
$$

and hence, using the assumptions that $\rho _ { 1 } \geq 2 \| \mathbf { y } ^ { * } \| _ { 2 }$ and $\rho _ { 2 } \geq 2 \| \mathbf { t } ^ { * } \| _ { 2 }$ , 

$$
\| [ \mathbf {g} (\tilde {\mathbf {x}}) ] _ {+} \| _ {2} = \| \tilde {\mathbf {u}} \| _ {2} \leq \frac {\delta}{\rho_ {1} - \| \mathbf {y} ^ {*} \| _ {2}} \leq \frac {2}{\rho_ {1}} \delta ,
$$

$$
\| \mathbf {A} \tilde {\mathbf {x}} + \mathbf {b} \| _ {2} = \| \tilde {\mathbf {t}} \| _ {2} \leq \frac {\delta}{\rho_ {2} - \| \mathbf {z} ^ {*} \| _ {2}} \leq \frac {2}{\rho_ {2}} \delta .
$$

## 3.6 Lipschitz Continuity and Boundedness of Subgradients

This section considers an important relation between Lipschitz continuity of a convex function and boundedness of its subgradients. 

Theorem 3.61 (Lipschitz continuity and boundedness of the subdiferential sets). Let $f : \mathbb { E }  ( - \infty , \infty ]$ be a proper and convex function. Suppose that $X \subseteq \operatorname { i n t } ( \operatorname { d o m } f )$ . Consider the following two claims: 

(i) $| f ( \mathbf { x } ) - f ( \mathbf { y } ) | \leq L \| \mathbf { x } - \mathbf { y } \| \ f o r \ a n y \ \mathbf { x } , \mathbf { y } \in X$ 

(ii) $\| \mathbf { g } \| _ { * } \leq L$ for any $\mathbf { g } \in \partial f ( \mathbf { x } ) , \mathbf { x } \in X$ 

Then 

(a) the implication $\mathrm { ( i i ) } \Rightarrow \mathrm { ( i ) }$ holds; 

(b) if X is open, then (i) holds if and only if (ii) holds. 

Proof. (a) Suppose that (ii) is satisfied and let $\mathbf { x } , \mathbf { y } \in X$ . Let $\mathbf { g _ { x } } \in \partial f ( \mathbf { x } )$ and $\mathbf { g } _ { \mathbf { y } } \in \partial f ( \mathbf { y } )$ . The existence of these subgradients is guaranteed by Theorem 3.14. Then by the definitions of $\mathbf { g } _ { \mathbf { x } } , \mathbf { g } _ { \mathbf { y } }$ and the generalized Cauchy–Schwarz inequality (Lemma 1.4), 

$$
f (\mathbf {x}) - f (\mathbf {y}) \leq \left\langle \mathbf {g} _ {\mathbf {x}}, \mathbf {x} - \mathbf {y} \right\rangle \leq \| \mathbf {g} _ {\mathbf {x}} \| _ {*} \| \mathbf {x} - \mathbf {y} \| \leq L \| \mathbf {x} - \mathbf {y} \|,
$$

$$
f (\mathbf {y}) - f (\mathbf {x}) \leq \left\langle \mathbf {g} _ {\mathbf {y}}, \mathbf {y} - \mathbf {x} \right\rangle \leq \| \mathbf {g} _ {\mathbf {y}} \| _ {*} \| \mathbf {x} - \mathbf {y} \| \leq L \| \mathbf {x} - \mathbf {y} \|,
$$

showing the validity of (i). 

(b) The implication $\mathrm { ( i i ) } \Rightarrow \mathrm { ( i ) }$ was already shown. Now assume that (i) is satisfied. Take $\mathbf { x } \in X$ and $\mathbf { g } \in \partial f ( \mathbf { x } )$ . We will show that $\| \mathbf { g } \| _ { * } \leq L$ . Define $\mathbf { g } ^ { \dagger } \in \mathbb { E }$ as a vector that satisfies $\| \mathbf { g } ^ { \dag } \| = 1 , \langle \mathbf { g } ^ { \dag } , \mathbf { g } \rangle = \| \mathbf { g } \| _ { * }$ (the existence of such a vector is warranted by the definition of the dual norm). Take $\varepsilon > 0$ small enough such that $\mathbf { x } + \varepsilon \mathbf { g } ^ { \dagger } \in X$ . By the subgradient inequality we have 

$$
f (\mathbf {x} + \varepsilon \mathbf {g} ^ {\dagger}) \geq f (\mathbf {x}) + \langle \mathbf {g}, \varepsilon \mathbf {g} ^ {\dagger} \rangle .
$$

Thus, 

$$
\varepsilon \| \mathbf {g} \| _ {*} = \langle \mathbf {g}, \varepsilon \mathbf {g} ^ {\dagger} \rangle \leq f (\mathbf {x} + \varepsilon \mathbf {g} ^ {\dagger}) - f (\mathbf {x}) \leq L \| \mathbf {x} + \varepsilon \mathbf {g} ^ {\dagger} - \mathbf {x} \| = L \varepsilon ,
$$

showing that $\| \mathbf { g } \| _ { * } \leq L$ 

Recall that by Theorem 3.16, the subgradients of a given convex function $f$ are bounded over compact sets contained in int $\left( \operatorname { d o m } ( f ) \right)$ . Combining this with Theorem 3.61, we can conclude that convex functions are always Lipschitz continuous over compact sets contained in the interior of their domain. 

Corollary 3.62 (Lipschitz continuity of convex functions over compact domains). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper and convex function. Suppose that $X \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ is compact. Then there exists $L > 0$ for which 

$$
| f (\mathbf {x}) - f (\mathbf {y}) | \leq L \| \mathbf {x} - \mathbf {y} \| \text {   for   any   } \mathbf {x}, \mathbf {y} \in X.
$$

## 3.7 Optimality Conditions

## 3.7.1 Fermat’s Optimality Condition

Subdiferential sets are extremely useful in characterizing minima points. Perhaps the most basic optimality condition states that a point is a global minimum of a proper extended real-valued convex function if and only if 0 belongs to the subdiferential set at the point. In a sense, this is a generalization of Fermat’s optimality condition at points of diferentiability $( ^ { * } \nabla f ( \mathbf { x } ^ { * } ) = \mathbf { 0 } ^ { * } )$ ). We will refer to this condition as Fermat’s optimality condition. 

Theorem 3.63 (Fermat’s optimality condition). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper convex function. Then 

$$
\mathbf {x} ^ {*} \in \operatorname{argmin} \{f (\mathbf {x}): \mathbf {x} \in \mathbb {E} \}\tag{3.70}
$$

if and only if $\mathbf { 0 } \in \partial f ( \mathbf { x } ^ { * } )$ 

Proof. Follows by the definition of the subgradient. Indeed, (3.70) is satisfied if and only if 

$$
f (\mathbf {x}) \geq f \left(\mathbf {x} ^ {*}\right) + \langle \mathbf {0}, \mathbf {x} - \mathbf {x} ^ {*} \rangle \text {   for   any   } \mathbf {x} \in \operatorname{dom} (f),
$$

which is the same as the inclusion $\mathbf { 0 } \in \partial f ( \mathbf { x } ^ { * } )$ 

Example 3.64 (minimizing piecewise linear functions). Consider the problem 1 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {n}} \left[ f (\mathbf {x}) \equiv \max _ {i = 1, 2, \dots , m} \left\{\mathbf {a} _ {i} ^ {T} \mathbf {x} + b _ {i} \right\} \right],\tag{3.71}
$$

where $\mathbf { a } _ { i } \in \mathbb { R } ^ { n } , b _ { i } \in \mathbb { R } , i = 1 , 2 , \dots , m$ . Denote 

$$
I (\mathbf {x}) = \left\{i: f (\mathbf {x}) = \mathbf {a} _ {i} ^ {T} \mathbf {x} + b _ {i} \right\}.
$$

Then, by Example 3.53, 

$$
\partial f (\mathbf {x}) = \left\{\sum_ {i \in I (\mathbf {x})} \lambda_ {i} \mathbf {a} _ {i}: \sum_ {i \in I (\mathbf {x})} \lambda_ {i} = 1, \lambda_ {j} \geq 0, j \in I (\mathbf {x}) \right\}.
$$

Therefore, since by Fermat’s optimality condition $\mathbf { x } ^ { * }$ is an optimal solution of (3.71) if and only if $\mathbf { 0 } \in \partial f ( \mathbf { x } ^ { * } )$ , it follows that $\mathbf { x } ^ { * }$ is an optimal solution of problem (3.71) if and only if there exists $\lambda \in \Delta _ { m }$ such that 

$$
\mathbf {0} = \sum_ {i = 1} ^ {m} \lambda_ {i} \mathbf {a} _ {i} \text {   and   } \lambda_ {j} = 0 \text {   for   any   } j \notin I (\mathbf {x} ^ {*}).\tag{3.72}
$$

We can rewrite this condition in a more compact way by denoting $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ to be the matrix whose rows are $\mathbf { a } _ { 1 } ^ { T } , \mathbf { a } _ { 2 } ^ { T } , \ldots , \mathbf { a } _ { m } ^ { T }$ . Then the optimality condition (3.72) can be written as 

$$
\exists \pmb {\lambda} \in \Delta_ {m} \mathrm{s.t.} \mathbf {A} ^ {T} \pmb {\lambda} = \mathbf {0} \mathrm{and} \lambda_ {j} (\mathbf {a} _ {j} ^ {T} \mathbf {x} ^ {*} + b _ {j} - f (\mathbf {x} ^ {*})) = 0, j = 1, 2, \ldots , m.
$$

Example 3.65 (medians). Suppose that we are given n diferent<sup>20</sup> and ordered numbers $a _ { 1 } < a _ { 2 } < \cdots < a _ { n }$ . Denote $A = \left\{ a _ { 1 } , a _ { 2 } , \dotsc , a _ { n } \right\} \subseteq \mathbb { R }$ . The median of A is a number $\beta$ that satisfies 

$$
\# \{i: a _ {i} \leq \beta \} \geq \frac {n}{2} \text { and } \# \{i: a _ {i} \geq \beta \} \geq \frac {n}{2}.
$$

That is, a median of A is a number that satisfies that at least half of the numbers in A are smaller or equal to it and that at least half are larger or equal. It is not dificult to see that if A has an odd number of elements, then the median is the middlemost number. For example, the median of {5, 8, 11, 60, 100} is 11. If the number of elements in A is even, then there is no unique median. The set of medians comprises all numbers between the two middle values. For example, if $A = \{ 5 , 8 , 1 1 , 2 0 , 6 0 , 1 0 0 \}$ , then the set of medians of A is the interval [11, 20]. In general, 

$$
\operatorname{median} (A) = \left\{ \begin{array}{l l} a _ {\frac {n + 1}{2}}, & n \text { odd }, \\ [ a _ {\frac {n}{2}}, a _ {\frac {n}{2} + 1} ], & n \text { even }. \end{array} \right.
$$

From an optimization perspective, the set of possible medians is the optimal solution set of the problem 

$$
\min \left\{f (x) \equiv \sum_ {i = 1} ^ {n} | x - a _ {i} | \right\}.\tag{3.73}
$$

To show this, denote $f _ { i } ( x ) \equiv | x - a _ { i } |$ , so that $f ( x ) = f _ { 1 } ( x ) + f _ { 2 } ( x ) + \cdot \cdot \cdot + f _ { n } ( x )$ 2 and note that for any $i \in \{ 1 , 2 , \ldots , n \}$ , 

$$
\partial f _ {i} (x) = \left\{ \begin{array}{l l} 1, & x > a _ {i}, \\ - 1, & x <   a _ {i}, \\ [ - 1, 1 ], & x = a _ {i}. \end{array} \right.
$$

By the sum rule of subdiferential calculus (Theorem 3.40), 

$$
\begin{array}{l} \partial f (x) = \partial f _ {1} (x) + \partial f _ {2} (x) + \dots + \partial f _ {n} (x) \\ = \left\{ \begin{array}{l l} \# \{i: a _ {i} <   x \} - \# \{i: a _ {i} > x \}, & x \notin A, \\ \# \{i: a _ {i} <   x \} - \# \{i: a _ {i} > x \} + [ - 1, 1 ], & x \in A. \end{array} \right. \end{array}
$$

We can further elaborate and write 

$$
\partial f (x) = \left\{ \begin{array}{l l} 2 i - n, & x \in (a _ {i}, a _ {i + 1}), \\ 2 i - 1 - n + [ - 1, 1 ], & x = a _ {i}, \\ - n, & x <   a _ {1}, \\ n, & x > a _ {n}. \end{array} \right.\tag{3.74}
$$

Let $i \in \{ 1 , 2 , . . . , n \} . \mathrm { ~ B y ~ } ( 3 . 7 4 ) , 0 \in \partial f ( a _ { i } )$ if and only $\mathrm { i f ~ } | 2 i - 1 - n | \leq 1$ , which is equivalent to $\begin{array} { r } { \frac { n } { 2 } \leq i \leq \frac { n } { 2 } + 1 } \end{array}$ and $0 \in \partial f ( x )$ for some $x \in \left( a _ { i } , a _ { i + 1 } \right)$ if and only if $\begin{array} { r } { i = \frac { n } { 2 } } \end{array}$ . We can thus conclude that if n is odd, then the only optimal point is $a _ { \frac { n + 1 } { 2 } }$ and when n is even, the optimal set is the interval $[ a _ { \frac { n } { 2 } } , a _ { \frac { n } { 2 } + 1 } ]$ , establishing the fact that the optimal set of (3.73) is exactly the set of medians. 

Example 3.66 (Fermat–Weber problem). Given m diferent points in $\mathbb { R } ^ { d } , A =$ $\{ \mathbf { a } _ { 1 } , \mathbf { a } _ { 2 } , \dots , \mathbf { a } _ { m } \}$ , and m positive weights $\omega _ { 1 } , \omega _ { 2 } , \ldots , \omega _ { m }$ , the Fermat–Weber problem is given by 

$$
\text {(FW)} \quad \min _ {\mathbf {x} \in \mathbb {R} ^ {d}} \left\{f (\mathbf {x}) \equiv \sum_ {i = 1} ^ {m} \omega_ {i} \| \mathbf {x} - \mathbf {a} _ {i} \| _ {2} \right\}.
$$

The Fermat–Weber problem is actually a weighted multidimensional version of the median problem (3.73) discussed in the previous example and is therefore also referred to in the literature as the geometric median problem. Let us write explicitly the optimality conditions for problem (FW). Denote $f _ { i } ( { \bf x } ) = \omega _ { i } g _ { i } ( { \bf x } )$ , where $g _ { i } ( \mathbf { x } ) \equiv \| \mathbf { x } - \mathbf { a } _ { i } \| _ { 2 }$ . Then for any $i \in \{ 1 , 2 , \dots , m \}$ 

$$
\partial f _ {i} (\mathbf {x}) = \left\{ \begin{array}{l l} \omega_ {i} \frac {\mathbf {x} - \mathbf {a} _ {i}}{\| \mathbf {x} - \mathbf {a} _ {i} \| _ {2}}, & \mathbf {x} \neq \mathbf {a} _ {i}, \\ B _ {\| \cdot \| _ {2}} [ \mathbf {0}, \omega_ {i} ], & \mathbf {x} = \mathbf {a} _ {i}, \end{array} \right.
$$

where here we used Theorems 3.35 (“multiplication by a positive scalar”), the afine transformation rule of subdiferential calculus (Theorem 3.43(b)), and Example 3.34, in which the subdiferential set of the $l _ { 2 } { \mathrm { - n o r m } }$ was computed. Obviously, $\textstyle f = \sum _ { i = 1 } ^ { m } f _ { i }$ , and hence, by the sum rule of subdiferential calculus (Theorem $3 . 4 0 ^ { 2 1 } )$ , we obtain that 

$$
\partial f (\mathbf {x}) = \sum_ {i = 1} ^ {m} \partial f _ {i} (\mathbf {x}) = \left\{ \begin{array}{l l} \sum_ {i = 1} ^ {m} \omega_ {i} \frac {\mathbf {x} - \mathbf {a} _ {i}}{\| \mathbf {x} - \mathbf {a} _ {i} \| _ {2}}, & \mathbf {x} \notin \mathcal {A}, \\ \sum_ {i = 1, i \neq j} ^ {m} \omega_ {i} \frac {\mathbf {x} - \mathbf {a} _ {i}}{\| \mathbf {x} - \mathbf {a} _ {i} \| _ {2}} + B [ \mathbf {0}, \omega_ {j} ], & \mathbf {x} = \mathbf {a} _ {j} (j = 1, 2, \ldots , m). \end{array} \right.
$$

Using Fermat’s optimality condition (Theorem 3.63), we can conclude that $\mathbf { x } ^ { * } \in \mathbb { R } ^ { d }$ is an optimal solution of problem (FW) if and only if either 

$$
\mathbf {x} ^ {*} \notin \mathcal {A} \mathrm{and} \sum_ {i = 1} ^ {m} \omega_ {i} \frac {\mathbf {x} ^ {*} - \mathbf {a} _ {i}}{\| \mathbf {x} ^ {*} - \mathbf {a} _ {i} \| _ {2}} = \mathbf {0}
$$

or for some $j \in \{ 1 , 2 , \dots , m \}$ 

$$
\mathbf {x} ^ {*} = \mathbf {a} _ {j} \text {and} \left\| \sum_ {i = 1, i \neq j} ^ {m} \omega_ {i} \frac {\mathbf {x} ^ {*} - \mathbf {a} _ {i}}{\| \mathbf {x} ^ {*} - \mathbf {a} _ {i} \| _ {2}} \right\| _ {2} \leq \omega_ {j}. \quad \blacksquare
$$

## 3.7.2 Convex Constrained Optimization

Consider the constrained optimization problem 

$$
\min \{f (\mathbf {x}): \mathbf {x} \in C \},\tag{3.75}
$$

where $f$ is an extended real-valued convex function and $C \subseteq \mathbb { E }$ is a convex set. Using Fermat’s optimality condition (Theorem 3.63) and the convexity assumptions, it is easy to write a necessary and suficient optimality condition for problem (3.75) in terms of the subdiferential set of f and the normal cone of C. 

Theorem 3.67 (necessary and suficient optimality conditions for convex constrained optimization). Let $f : \mathbb { E }  ( - \infty , \infty ]$ be a proper and convex function, and let $C \subseteq \mathbb { E }$ be a convex set for which $\operatorname { r i } ( \operatorname { d o m } ( f ) ) \cap \operatorname { r i } ( C ) \neq \emptyset$ . Then $\mathbf { x } ^ { \ast } \in C$ is an optimal solution of (3.75) if and only if 

$$
\text { there   exists   } \mathbf {g} \in \partial f (\mathbf {x} ^ {*}) \text {   for   which   } - \mathbf {g} \in N _ {C} (\mathbf {x} ^ {*}).\tag{3.76}
$$

Proof. Problem (3.75) can be rewritten as 

$$
\min _ {\mathbf {x} \in \mathbb {E}} f (\mathbf {x}) + \delta_ {C} (\mathbf {x}).
$$

Since $\operatorname { r i } ( \operatorname { d o m } ( f ) ) \cap \operatorname { r i } ( C ) \neq \emptyset$ , it follows by the sum rule of subdiferential calculus (Theorem 3.40) that for any $\mathbf { x } \in C$ , 

$$
\partial (f + \delta_ {C}) (\mathbf {x}) = \partial f (\mathbf {x}) + \partial \delta_ {C} (\mathbf {x}).
$$

By Example 3.5, $\partial \delta _ { C } ( { \bf x } ) = N _ { C } ( { \bf x } )$ , and consequently for any $\mathbf { x } \in C$ 

$$
\partial (f + \delta_ {C}) (\mathbf {x}) = \partial f (\mathbf {x}) + N _ {C} (\mathbf {x}).
$$

Therefore, invoking Fermat’s optimality condition (Theorem 3.63), $\mathbf { x } ^ { \ast } \in C$ is an optimal solution of (3.75) if and only if ${ \mathbf 0 } \in \partial f ( { \mathbf x } ^ { * } ) + N _ { C } ( { \mathbf x } ^ { * } )$ , that is, if and only if 

$$
(- \partial f (\mathbf {x} ^ {*})) \cap N _ {C} (\mathbf {x} ^ {*}) \neq \emptyset ,
$$

which is the same as condition (3.76). 

Using the definition of the normal cone, we can write the optimality condition in a slightly more explicit manner. 

Corollary 3.68 (necessary and suficient optimality conditions for convex constrained optimization—second version). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper and convex function, and let C be a convex set satisfying $\operatorname { r i } ( \operatorname { d o m } ( f ) ) \cap \operatorname { r i } ( C ) \neq \emptyset$ Then $\mathbf { x } ^ { \ast } \in C$ is an optimal solution of (3.75) if and only if 

$$
\text { there   exists } \mathbf {g} \in \partial f (\mathbf {x} ^ {*}) \text { for   which } \langle \mathbf {g}, \mathbf {x} - \mathbf {x} ^ {*} \rangle \geq 0 \text { for   any } \mathbf {x} \in C.\tag{3.77}
$$

Condition (3.77) is not particularly explicit. We will show in the next example how to write it in an explicit way for the case where $C = \Delta _ { n }$ 

Example 3.69 (optimality conditions over the unit simplex). Suppose that the assumptions in Corollary 3.68 hold and that $C = \Delta _ { n } , \mathbb { E } = \mathbb { R } ^ { n }$ . Given $\mathbf { x } ^ { * } \in \Delta _ { n }$ we will show that the condition 

$$
\left(\mathrm{I}\right) \mathbf {g} ^ {T} (\mathbf {x} - \mathbf {x} ^ {*}) \geq 0 \text { for   all } \mathbf {x} \in \Delta_ {n}
$$

is satisfied if and only if the following condition is satisfied: 

$$
\text {(II) there exist} \mu \in \mathbb {R} \text {such that} g _ {i} \left\{ \begin{array}{l l} = \mu , & x _ {i} ^ {*} > 0, \\ \geq \mu , & x _ {i} ^ {*} = 0. \end{array} \right.
$$

Assume first that (II) is satisfied. Then for any $\mathbf { x } \in \Delta _ { n }$ 

$$
\begin{array}{l} \mathbf {g} ^ {T} (\mathbf {x} - \mathbf {x} ^ {*}) = \sum_ {i = 1} ^ {n} g _ {i} (x _ {i} - x _ {i} ^ {*}) \\ \qquad = \sum_ {i: x _ {i} ^ {*} > 0} g _ {i} (x _ {i} - x _ {i} ^ {*}) + \sum_ {i: x _ {i} ^ {*} = 0} g _ {i} x _ {i} \\ \qquad \geq \sum_ {i: x _ {i} ^ {*} > 0} \mu (x _ {i} - x _ {i} ^ {*}) + \mu \sum_ {i: x _ {i} ^ {*} = 0} x _ {i} \\ \qquad = \mu \sum_ {i = 1} ^ {n} x _ {i} - \mu \sum_ {i: x _ {i} ^ {*} > 0} x _ {i} ^ {*} = \mu - \mu = 0, \end{array}
$$

proving that condition (I) is satisfied. To show the reverse direction, assume that (I) is satisfied. Let i and $j$ be two diferent indices for which $x _ { i } ^ { * } > 0$ . Define the vector $\mathbf { x } \in \Delta _ { n }$ as 

$$
x _ {k} = \left\{ \begin{array}{l l} x _ {k} ^ {*}, & k \notin \{i, j \}, \\ x _ {i} ^ {*} - \frac {x _ {i} ^ {*}}{2}, & k = i, \\ x _ {j} ^ {*} + \frac {x _ {i} ^ {*}}{2}, & k = j. \end{array} \right.
$$

The inequality $\mathbf { g } ^ { T } ( \mathbf { x } - \mathbf { x } ^ { * } ) \geq 0$ then amounts to 

$$
- \frac {x _ {i} ^ {*}}{2} g _ {i} + \frac {x _ {i} ^ {*}}{2} g _ {j} \geq 0,
$$

which by the fact that $x _ { i } ^ { * } > 0$ implies that 

$$
g _ {i} \leq g _ {j}.\tag{3.78}
$$

In particular, for any two indices $i \neq j$ for which $x _ { i } ^ { * } , x _ { j } ^ { * } > 0$ , the two inequalities $g _ { i } \ \leq \ g _ { j }$ and $g _ { j } \ \leq \ g _ { i }$ hold, and hence $g _ { i } ~ = ~ g _ { j }$ . Therefore, all the components of g corresponding to positive components of $\mathbf { x } ^ { * }$ have the same value, which we will denote by $\mu .$ Let i be any index for which $x _ { i } ^ { * } > 0$ . Then for any index $j$ for which $x _ { j } ^ { * } = 0$ , the inequality (3.78) holds. Therefore, $g _ { j } \geq \mu$ , and condition (II) is thus established. 

We summarize the discussion in Example 3.69 with the following corollary. 

Corollary 3.70 (necessary and suficient optimality conditions for convex problems over the unit simplex). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper and convex function. Suppose that $\operatorname { r i } ( \Delta _ { n } ) \cap \operatorname { r i } ( \operatorname { d o m } ( f ) ) \neq \emptyset$ . Then $\mathbf { x } ^ { * } \in \Delta _ { n }$ is an optimal solution of 

$$
\min \{f (\mathbf {x}): \mathbf {x} \in \Delta_ {n} \}
$$

if and only if there exists $\mathbf { g } \in \partial f ( \mathbf { x } ^ { * } )$ and $\mu \in \mathbb { R }$ for which 

$$
g _ {i} \left\{ \begin{array}{l l} = \mu , & x _ {i} ^ {*} > 0, \\ \geq \mu , & x _ {i} ^ {*} = 0. \end{array} \right.
$$

The following example illustrates one instance in which the optimal solution of a convex problem over the unit simplex can be found using Corollary 3.70. 

Example 3.71. Consider the problem 

$$
\min _ {\mathbf {x}} \left\{\sum_ {i = 1} ^ {n} x _ {i} \log x _ {i} - \sum_ {i = 1} ^ {n} y _ {i} x _ {i}: \mathbf {x} \in \Delta_ {n} \right\},\tag{3.79}
$$

where $\mathbf { y } \in \mathbb { R } ^ { n }$ is a given vector. Problem (3.79) can be written as 

$$
\min \{f (\mathbf {x}): \mathbf {x} \in \Delta_ {n} \},
$$

where $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ is given by 

$$
f (\mathbf {x}) = \left\{ \begin{array}{l l} \sum_ {i = 1} ^ {n} x _ {i} \log x _ {i} - \sum_ {i = 1} ^ {n} y _ {i} x _ {i}, & \mathbf {x} \geq \mathbf {0}, \\ \infty & \text { else }. \end{array} \right.
$$

Let us assume that there exists an optimal solution<sup>22</sup> x<sup>∗</sup> satisfying $\mathbf { x } ^ { * } > \mathbf { 0 }$ . Then under this assumption, by Corollary 3.70 and the fact that $f$ is diferentiable at any positive vector, it follows that there exists $\mu \in \mathbb { R }$ such that for any i, $\begin{array} { r } { \frac { \partial f } { \partial x _ { i } } ( \mathbf { x } ^ { * } ) = \grave { \mu } _ { : } } \end{array}$ which is the same as log $x _ { i } ^ { * } + 1 - y _ { i } = \mu .$ . Therefore, for any $i ,$ , 

$$
x _ {i} ^ {*} = e ^ {\mu - 1 + y _ {i}} = \alpha e ^ {y _ {i}}, \quad i = 1, 2, \dots , n
$$

where $\alpha = e ^ { \mu - 1 }$ . Since $\textstyle \sum _ { i = 1 } ^ { n } x _ { i } ^ { * } = 1$ , it follows that $\begin{array} { r } { \alpha = \frac { 1 } { \sum _ { j = 1 } ^ { n } e ^ { y _ { j } } } } \end{array}$ . Therefore, 

$$
x _ {i} ^ {*} = \frac {e ^ {y _ {i}}}{\sum_ {j = 1} ^ {n} e ^ {y _ {j}}}, \quad i = 1, 2, \ldots , n.
$$

This is indeed an optimal solution of problem (3.79) since it satisfies the conditions of Corollary 3.70, which are (also) suficient conditions for optimality. 

## 3.7.3 The Nonconvex Composite Model

It is also possible to write a necessary optimality condition for nonconvex problems in terms of subgradients. We will write such a condition for problems consisting of minimizing a composite function $f + g .$ , where $f$ possesses some diferentiability properties but is not assumed to be convex while $g$ is convex but not assumed to have any special diferentiability properties. 

Theorem 3.72 (optimality conditions for the composite problem). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper function, and let $g : \mathbb { E }  ( - \infty , \infty ]$ be a proper convex function such that dom $( g ) \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ . Consider the problem 

$$
\text {(P)} \quad \min _ {\mathbf {x} \in \mathbb {E}} f (\mathbf {x}) + g (\mathbf {x}).
$$

(a) (necessary condition) $I f \mathbf { x } ^ { * } \in \mathrm { d o m } ( g )$ is a local optimal solution of (P) and f is diferentiable at $\mathbf { x } ^ { * }$ , then 

$$
- \nabla f (\mathbf {x} ^ {*}) \in \partial g (\mathbf {x} ^ {*}).\tag{3.80}
$$

(b) (necessary and suficient condition for convex problems) Suppose that f is convex. $I f f$ is diferentiable at $\mathbf { x } ^ { * } \in \mathrm { d o m } ( g )$ , then $\mathbf { x } ^ { * }$ is a global optimal solution of (P) if and only if (3.80) is satisfied. 

Proof. (a) Let $\mathbf { y } \in \operatorname { d o m } ( g )$ . Then by the convexity of dom $( g )$ , for any $\lambda \in ( 0 , 1 )$ ), the point $\mathbf { x } _ { \lambda } = ( 1 - \lambda ) \mathbf { x } ^ { * } + \lambda \mathbf { y }$ is in dom $( g )$ , and by the local optimality of $\mathbf { x } ^ { * }$ , it follows that, for small enough λ, 

$$
f (\mathbf {x} _ {\lambda}) + g (\mathbf {x} _ {\lambda}) \geq f (\mathbf {x} ^ {*}) + g (\mathbf {x} ^ {*}).
$$

That is, 

$$
f ((1 - \lambda) \mathbf {x} ^ {*} + \lambda \mathbf {y}) + g ((1 - \lambda) \mathbf {x} ^ {*} + \lambda \mathbf {y}) \geq f (\mathbf {x} ^ {*}) + g (\mathbf {x} ^ {*}).
$$

Using the convexity of $g ,$ , it follows that 

$$
f ((1 - \lambda) \mathbf {x} ^ {*} + \lambda \mathbf {y}) + (1 - \lambda) g (\mathbf {x} ^ {*}) + \lambda g (\mathbf {y}) \geq f (\mathbf {x} ^ {*}) + g (\mathbf {x} ^ {*}),
$$

which is the same as 

$$
\frac {f ((1 - \lambda) \mathbf {x} ^ {*} + \lambda \mathbf {y}) - f (\mathbf {x} ^ {*})}{\lambda} \geq g (\mathbf {x} ^ {*}) - g (\mathbf {y}).
$$

Taking $\lambda \to 0 ^ { + }$ in the last inequality yields 

$$
f ^ {\prime} (\mathbf {x} ^ {*}; \mathbf {y} - \mathbf {x} ^ {*}) \geq g (\mathbf {x} ^ {*}) - g (\mathbf {y}),
$$

where we used the fact that since $f$ is diferentiable at $\mathbf { x } ^ { * }$ , its directional derivatives exist. In fact, by Theorem 3.29, we have $f ^ { \prime } ( \mathbf { x } ^ { * } ; \mathbf { y } - \mathbf { x } ^ { * } ) = \langle \nabla f ( \mathbf { x } ^ { * } ) , \mathbf { y } - \mathbf { x } ^ { * } \rangle$ , and hence for any $\mathbf { y } \in \operatorname { d o m } ( g )$ , 

$$
g (\mathbf {y}) \geq g \left(\mathbf {x} ^ {*}\right) + \langle - \nabla f \left(\mathbf {x} ^ {*}\right), \mathbf {y} - \mathbf {x} ^ {*} \rangle ,
$$

showing that indeed $- \nabla f ( \mathbf { x } ^ { * } ) \in \partial g ( \mathbf { x } ^ { * } )$ 

(b) Suppose in addition that $f$ is convex. If $\mathbf { x } ^ { * }$ is an optimal solution of $( \mathrm { P } )$ then we already proved in part (a) that (3.80) is satisfied. Suppose now that (3.80) is satisfied. Then for any $\mathbf { y } \in \operatorname { d o m } ( g )$ 2 

$$
g (\mathbf {y}) \geq g \left(\mathbf {x} ^ {*}\right) + \langle - \nabla f \left(\mathbf {x} ^ {*}\right), \mathbf {y} - \mathbf {x} ^ {*} \rangle .\tag{3.81}
$$

By the convexity of $f _ { ; }$ for any $\mathbf { y } \in \operatorname { d o m } ( g )$ 

$$
f (\mathbf {y}) \geq f \left(\mathbf {x} ^ {*}\right) + \langle \nabla f \left(\mathbf {x} ^ {*}\right), \mathbf {y} - \mathbf {x} ^ {*} \rangle .\tag{3.82}
$$

Adding (3.81) and (3.82), we obtain that 

$$
f (\mathbf {y}) + g (\mathbf {y}) \geq f (\mathbf {x} ^ {*}) + g (\mathbf {x} ^ {*})
$$

for any $\mathbf { y } \in \operatorname { d o m } ( g )$ , meaning that $\mathbf { x } ^ { * }$ is an optimal solution of (P). 

The condition (3.80) is an important optimality condition, and we will refer to it as the “stationarity” condition. 

Definition 3.73 (stationarity). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be proper and let $g : \mathbb { E } $ $( - \infty , \infty ]$ be a proper convex function such that dom $\operatorname { \rho } ( g ) \subseteq \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ . Consider the problem 

$$
\text {(P)} \quad \min _ {\mathbf {x} \in \mathbb {E}} f (\mathbf {x}) + g (\mathbf {x}).
$$

A point $\mathbf { x } ^ { * }$ in which f is diferentiable is called a stationary point $o f \left( \mathrm { P } \right)$ if 

$$
- \nabla f (\mathbf {x} ^ {*}) \in \partial g (\mathbf {x} ^ {*}).
$$

Under the setting of Definition 3.73, $\mathbf { x } ^ { * }$ is also called a stationary point of the function $f + g$ 

We have shown in Theorem 3.72 that stationarity is a necessary local optimality condition for problem (P), and that if $f$ is convex, then stationarity is a necessary and suficient global optimality condition. The case $g = \delta _ { C }$ deserves a separate discussion. 

Example 3.74 (convex constrained nonconvex programming). When $g =$ $\delta _ { C }$ for a nonempty convex set $C \subseteq \mathbb { E }$ , problem (P) becomes 

$$
\min \{f (\mathbf {x}): \mathbf {x} \in C \},
$$

which is a problem consisting of minimizing a (possibly) nonconvex function over a convex set. A point $\mathbf { x ^ { \ast } } \in C$ in which f is diferentiable is a stationary point of (P) if and only if 

$$
- \nabla f (\mathbf {x} ^ {*}) \in \partial \delta_ {C} (\mathbf {x} ^ {*}) = N _ {C} (\mathbf {x} ^ {*}),\tag{3.83}
$$

where the equality is due to Example 3.5. By the definition of the normal cone, condition (3.83) can be rewritten as 

$$
\langle - \nabla f (\mathbf {x} ^ {*}), \mathbf {x} - \mathbf {x} ^ {*} \rangle \leq 0 \text {   for   any   } \mathbf {x} \in C,
$$

which is the same as 

$$
\langle \nabla f (\mathbf {x} ^ {*}), \mathbf {x} - \mathbf {x} ^ {*} \rangle \geq 0 \mathrm{forany} \mathbf {x} \in C.
$$

Example 3.75. Consider the problem 

$$
\min _ {\mathbf {x} \in \mathbb {R} ^ {n}} f (\mathbf {x}) + \lambda \| \mathbf {x} \| _ {1},\tag{3.84}
$$

where $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ is an extended real-valued function. A point $\mathbf { x } ^ { * } \in$ int(dom(f)) in which f is diferentiable is a stationary point of (3.84) if 

$$
- \nabla f (\mathbf {x} ^ {*}) \in \lambda \partial g (\mathbf {x} ^ {*}),
$$

where $g ( \cdot ) = \| \cdot \| _ { 1 }$ . Using the expression for the subdiferential set of the $l _ { 1 }$ -norm given in Example 3.41, we obtain that $\mathbf { x } ^ { * }$ is a stationary point of problem (3.84) if 

and only if 

$$
\frac {\partial f (\mathbf {x} ^ {*})}{\partial x _ {i}} \left\{ \begin{array}{l l} = - \lambda , & x _ {i} ^ {*} > 0, \\ = \lambda , & x _ {i} ^ {*} <   0, \\ \in [ - \lambda , \lambda ], & x _ {i} ^ {*} = 0. \end{array} \right.\tag{3.85}
$$

By Theorem 3.72, condition (3.85) is a necessary condition for $\mathbf { x } ^ { * }$ to be a local minimum of problem (3.84). If f is also convex, then condition (3.85) is a necessary and suficient condition for $\mathbf { x } ^ { * }$ to be a global optimal solution of problem (3.84). 

## 3.7.4 The KKT Conditions

In this section we will show that the KKT conditions for constrained convex problems can be directly deduced by Fermat’s optimality condition. For that, we begin by establishing an equivalent reformulation of general inequality constrained problems. 

Lemma 3.76. Let $f , g _ { 1 } , g _ { 2 } , \ldots , g _ { m } : \mathbb { E } \to \mathbb { R }$ be real-valued functions. Consider the problem 

$$
\begin{array}{l l} \min & f (\mathbf {x}) \\ s. t. & g _ {i} (\mathbf {x}) \leq 0, i = 1, 2, \ldots , m. \end{array}\tag{3.86}
$$

Assume that the minimum value of problem (3.86) is finite and equal to ${ \bar { f } } .$ Define the function 

$$
F (\mathbf {x}) \equiv \max \{f (\mathbf {x}) - \bar {f}, g _ {1} (\mathbf {x}), g _ {2} (\mathbf {x}), \ldots , g _ {m} (\mathbf {x}) \}.\tag{3.87}
$$

Then the optimal set of problem (3.86) is the same as the set of minimizers of $F$ . 

Proof. Let $X ^ { * }$ be the optimal set of problem (3.86). To establish the result, we will show that F satisfies the following two properties: 

$$
\text {(i)} F (\mathbf {x}) > 0 \text { for   any } \mathbf {x} \notin X ^ {*}.
$$

$$
\text {(ii)} F (\mathbf {x}) = 0 \text { for   any } \mathbf {x} \in X ^ {*}.
$$

To prove property (i), let ${ \textbf { x } } \notin X ^ { * }$ . There are two options. Either x is not feasible,meaning that $g _ { i } ( \mathbf { x } ) > 0$ for some i, and hence by its definition $F ( \mathbf { x } ) > 0$ . If x isfeasible but not optimal, then $g _ { i } ( \mathbf { x } ) \leq 0$ for all $i = 1 , 2 , \dots , m$ and $f \left( \mathbf { x } \right) > \bar { f }$ , whichalso implies that $F ( \mathbf { x } ) > 0$ . To prove (ii), suppose that $\mathbf { x } \in X ^ { * }$ . Then $g _ { i } ( \mathbf { x } ) \leq 0$ forall $i = 1 , 2 , \dots , m$ and $f ( \mathbf { x } ) = { \bar { f } }$ , implying that $F ( \mathbf { x } ) = 0$ 口

Using Lemma 3.76, we can conclude that problem (3.86) reduces to 

$$
\min _ {\mathbf {x} \in \mathbb {E}} F (\mathbf {x})\tag{3.88}
$$

in the sense that the optimal sets of the two problems are the same. Using this equivalence, we can now establish under additional convexity assumptions the wellknown Fritz-John optimality conditions for problem (3.86). 

Theorem 3.77 (Fritz-John necessary optimality conditions). Consider the minimization problem 

$$
\begin{array}{l l} \min & f (\mathbf {x}) \\ s. t. & g _ {i} (\mathbf {x}) \leq 0, \quad i = 1, 2, \ldots , m, \end{array}\tag{3.89}
$$

where $f , g _ { 1 } , g _ { 2 } , \ldots , g _ { m } : \mathbb { E } \to \mathbb { R }$ are real-valued convex functions. Let $\mathbf { x } ^ { * }$ be an optimal solution of (3.89). Then there exist $\lambda _ { 0 } , \lambda _ { 1 } , \ldots , \lambda _ { m } \geq 0$ , not all zeros, $f o r$ which 

$$
\mathbf {0} \in \lambda_ {0} \partial f (\mathbf {x} ^ {*}) + \sum_ {i = 1} ^ {m} \lambda_ {i} \partial g _ {i} (\mathbf {x} ^ {*})\tag{3.90}
$$

$$
\lambda_ {i} g _ {i} (\mathbf {x} ^ {*}) = 0, \quad i = 1, 2, \ldots , m.\tag{3.91}
$$

Proof. Let $\mathbf { x } ^ { * }$ be an optimal solution of problem (3.89). Denote the optimal value of problem (3.89) by $\bar { f } = f ( \mathbf { x } ^ { * } )$ . Using Lemma 3.76, it follows that $\mathbf { x } ^ { * }$ is an optimal solution of the problem 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \{F (\mathbf {x}) \equiv \max \{g _ {0} (\mathbf {x}), g _ {1} (\mathbf {x}), \ldots , g _ {m} (\mathbf {x}) \},
$$

where $g _ { 0 } ( { \bf x } ) \equiv f ( { \bf x } ) - \bar { f } .$ . Obviously, $F ( \mathbf { x } ^ { * } ) = 0$ . Since $F$ is a maximum of convex functions, it is convex, and hence, using Fermat’s optimality condition (Theorem 3.63), 

$$
\mathbf {0} \in \partial F (\mathbf {x} ^ {*}).\tag{3.92}
$$

By the max rule of subdiferential calculus (Theorem 3.50), 

$$
\partial F (\mathbf {x} ^ {*}) = \mathrm{conv} \left(\left(\cup_ {i \in I (\mathbf {x} ^ {*})} \partial g _ {i} (\mathbf {x} ^ {*})\right), \right.\tag{3.93}
$$

where $I ( \mathbf { x } ^ { * } ) = \{ i \in \{ 0 , 1 , \ldots , m \} : g _ { i } ( \mathbf { x } ^ { * } ) = 0 \}$ . Combining (3.92) and (3.93), we can deduce that there exists $\lambda _ { i } \geq 0 , i \in I ( \mathbf { x } ^ { * } )$ , such that $\begin{array} { r } { \sum _ { i \in I ( { \bf x } ^ { * } ) } \lambda _ { i } = 1 } \end{array}$ for which 

$$
\mathbf {0} \in \sum_ {i \in I (\mathbf {x} ^ {*})} \lambda_ {i} \partial g _ {i} (\mathbf {x} ^ {*}).\tag{3.94}
$$

Since $g _ { 0 } ( \mathbf { x } ^ { * } ) = f ( \mathbf { x } ^ { * } ) - \bar { f } = 0$ , it follows that $0 \in I ( \mathbf { x } ^ { * } )$ , and hence (3.94) can be rewritten as 

$$
\mathbf {0} \in \lambda_ {0} \partial f (\mathbf {x} ^ {*}) + \sum_ {i \in I (\mathbf {x} ^ {*}) \backslash \{0 \}} \lambda_ {i} \partial g _ {i} (\mathbf {x} ^ {*}).
$$

Defining $\lambda _ { i } = 0$ for any $i \in \{ 1 , 2 , . . . , m \} \backslash I ( \mathbf { x } ^ { * } )$ , we conclude that conditions (3.90) and (3.91) are satisfied. Finally, not all the $\lambda _ { i } ^ { \cdot }$ s are zeros since $\begin{array} { r } { \sum _ { i \in I ( { \bf x } ^ { * } ) } \lambda _ { i } = 1 } \end{array}$ □ 

We will now establish the KKT conditions, which are the same as the Fritz-John conditions, but with $\lambda _ { 0 } = 1$ . The necessity of these conditions requires the following additional condition, which we refer to as Slater’s condition: 

$$
\text { there   exists   } \bar {\mathbf {x}} \in \mathbb {E} \text {   for   which   } g _ {i} (\bar {\mathbf {x}}) <   0, \quad i = 1, 2, \ldots , m.\tag{3.95}
$$

The suficiency of the KKT conditions does not require any additional assumptions (besides convexity) and is actually easily derived without using the result on the Fritz-John conditions. 

Theorem 3.78 (KKT conditions). Consider the minimization problem 

$$
\begin{array}{l l} \min & f (\mathbf {x}) \\ s. t. & g _ {i} (\mathbf {x}) \leq 0, \quad i = 1, 2, \ldots , m, \end{array}\tag{3.96}
$$

where $f , g _ { 1 } , g _ { 2 } , \ldots , g _ { m } : \mathbb { E } \to \mathbb { R }$ are real-valued convex functions. 

(a) Let $\mathbf { x } ^ { * }$ be an optimal solution of (3.96), and assume that Slater’s condition (3.95) is satisfied. Then there exist $\lambda _ { 1 } , \ldots , \lambda _ { m } \geq 0$ for which 

$$
\mathbf {0} \in \partial f (\mathbf {x} ^ {*}) + \sum_ {i = 1} ^ {m} \lambda_ {i} \partial g _ {i} (\mathbf {x} ^ {*})\tag{3.97}
$$

$$
\lambda_ {i} g _ {i} (\mathbf {x} ^ {*}) = 0, \quad i = 1, 2, \ldots , m.\tag{3.98}
$$

(b) If $\mathbf { x } ^ { * } \in \mathbb { E }$ satisfies conditions (3.97) and (3.98) for some $\lambda _ { 1 } , \lambda _ { 2 } , \ldots , \lambda _ { m } \geq 0$ then it is an optimal solution of problem (3.96). 

Proof. (a) By the Fritz-John conditions (Theorem 3.77) there exist $\tilde { \lambda } _ { 0 } , \tilde { \lambda } _ { 1 } , \ldots , \tilde { \lambda } _ { m } \geq$ 0, not all zeros, for which 

$$
\mathbf {0} \in \tilde {\lambda} _ {0} \partial f (\mathbf {x} ^ {*}) + \sum_ {i = 1} ^ {m} \tilde {\lambda} _ {i} \partial g _ {i} (\mathbf {x} ^ {*}),\tag{3.99}
$$

$$
\tilde {\lambda} _ {i} g _ {i} (\mathbf {x} ^ {*}) = 0, \quad i = 1, 2, \ldots , m.\tag{3.100}
$$

We will show that $\tilde { \lambda } _ { 0 } \neq 0$ . Assume by contradiction that $\tilde { \lambda } _ { 0 } = 0$ . Then, by (3.99), 

$$
\mathbf {0} \in \sum_ {i = 1} ^ {m} \tilde {\lambda} _ {i} \partial g _ {i} (\mathbf {x} ^ {*});
$$

that is, there exist $\pmb { \xi } _ { i } \in \partial g _ { i } ( \mathbf { x } ^ { * } ) , i = 1 , 2 , . . . , m$ , such that 

$$
\sum_ {i = 1} ^ {m} \tilde {\lambda} _ {i} \boldsymbol {\xi} _ {i} = \mathbf {0}.\tag{3.101}
$$

Let $\bar { \bf x }$ be a point satisfying Slater’s condition (3.95). By the subgradient inequality employed on the pair of points $\bar { \mathbf { x } } , \mathbf { x } ^ { * }$ w.r.t. the functions $g _ { i } , i = 1 , 2 , \ldots , m$ , we have 

$$
g _ {i} (\mathbf {x} ^ {*}) + \langle \pmb {\xi} _ {i}, \bar {\mathbf {x}} - \mathbf {x} ^ {*} \rangle \leq g _ {i} (\bar {\mathbf {x}}), \quad i = 1, 2, \dots , m.
$$

Multiplying the ith inequality by $\tilde { \lambda } _ { i } \geq 0$ and summing over $i = 1 , 2 , \dots , m$ yields 

$$
\sum_ {i = 1} ^ {m} \tilde {\lambda} _ {i} g _ {i} (\mathbf {x} ^ {*}) + \left\langle \sum_ {i = 1} ^ {m} \tilde {\lambda} _ {i} \boldsymbol {\xi} _ {i}, \bar {\mathbf {x}} - \mathbf {x} ^ {*} \right\rangle \leq \sum_ {i = 1} ^ {m} \tilde {\lambda} _ {i} g _ {i} (\bar {\mathbf {x}}), \quad i = 1, 2, \dots , m.
$$

Using (3.100) and (3.101), we obtain the inequality $\begin{array} { r } { \sum _ { i = 1 } ^ { m } \tilde { \lambda } _ { i } g _ { i } ( \bar { \bf x } ) \ge 0 } \end{array}$ , which is impossible since $\tilde { \lambda } _ { i } \geq 0$ and $g _ { i } ( \bar { \bf x } ) < 0$ for any $i ,$ and not all the $\ddot { \lambda } _ { i } \mathrm { ' s }$ are zeros. Therefore, $\tilde { \lambda } _ { 0 } > 0$ , and we can thus divide both the relation (3.99) and the equalities (3.100) by $\tilde { \lambda } _ { 0 }$ to obtain that (3.97) and (3.98) are satisfied with $\begin{array} { r } { \lambda _ { i } ~ = ~ \frac { \tilde { \lambda } _ { i } } { \tilde { \lambda } _ { 0 } } , i = } \end{array}$ $1 , 2 , \ldots , m$ 

(b) Suppose then that $\mathbf { x } ^ { * }$ satisfies (3.97) and (3.98) for some nonnegative numbers $\lambda _ { 1 } , \lambda _ { 2 } , \ldots , \lambda _ { m }$ . Let $\hat { \textbf { x } } \in \ \mathbb { E }$ be a feasible point of (3.96), meaning that $g _ { i } ( \hat { \mathbf { x } } ) \leq 0 , i = 1 , 2 , \ldots , m$ . We will show that $f ( \hat { \mathbf { x } } ) \geq f ( \mathbf { x } ^ { * } )$ . Define the function 

$$
h (\mathbf {x}) = f (\mathbf {x}) + \sum_ {i = 1} ^ {m} \lambda_ {i} g _ {i} (\mathbf {x}).
$$

The function $h$ is convex, and the condition (3.97) along with the sum rule of subdiferential calculus (Theorem 3.40) yields the relation 

$$
\mathbf {0} \in \partial h (\mathbf {x} ^ {*}),
$$

which by Fermat’s optimality condition (Theorem 3.63) implies that $\mathbf { x } ^ { * }$ is a minimizer of h over E. Combining this fact with (3.98) implies that 

$$
f (\mathbf {x} ^ {*}) = f (\mathbf {x} ^ {*}) + \sum_ {i = 1} ^ {m} \lambda_ {i} g _ {i} (\mathbf {x} ^ {*}) = h (\mathbf {x} ^ {*}) \leq h (\hat {\mathbf {x}}) = f (\hat {\mathbf {x}}) + \sum_ {i = 1} ^ {m} \lambda_ {i} g _ {i} (\hat {\mathbf {x}}) \leq f (\hat {\mathbf {x}}),
$$

where the last inequality follows from the facts that $\lambda _ { i } \geq 0$ and $g _ { i } ( \hat { \mathbf { x } } ) ~ \leq ~ 0$ for $i = 1 , 2 , \dots , m$ . We have thus proven that $\mathbf { x } ^ { * }$ is an optimal solution of (3.96). 

## 3.8 Summary of Weak and Strong Subgradient Calculus Results

This section contains a summary of most of the rules and results concerning the computation of subdiferential sets (strong results), as well as rules for computing specific subgradients in the subdiferential sets (weak results). Before that, we begin by summarizing the rules of subdiferential calculus. 

• Multiplication by a positive scalar 

$$
\partial (\alpha f) (\mathbf {x}) = \alpha \partial f (\mathbf {x}).
$$

Assumptions: $f : \mathbb { E } \to ( - \infty , \infty ]$ proper, $\alpha > 0 , \mathbf { x } \in \operatorname { d o m } ( f )$ . [Theorem 3.35] 

• Diferentiability 

f is diferentiable at x if and only if $\partial f ( \mathbf { x } )$ is a singleton, and in that case 

$$
\partial f (\mathbf {x}) = \{\nabla f (\mathbf {x}) \}.
$$

Assumptions: $f : \mathbb { E } \to ( - \infty , \infty ]$ proper convex, $\mathbf { x } \in \operatorname { i n t } ( \operatorname { d o m } ( f ) )$ . [Theorem 3.33] 

• Weak sum rule of subdiferential calculus 

$$
\sum_ {i = 1} ^ {m} \partial f _ {i} (\mathbf {x}) \subseteq \partial (\sum_ {i = 1} ^ {m} f _ {i}) (\mathbf {x}).
$$

Assumptions: $f _ { 1 } , f _ { 2 } , \ldots , f _ { m } : \mathbb { E } \to ( - \infty , \infty ]$ proper convex. [Corollary 3.38(a)] 

• Sum rule of subdiferential calculus 

$$
\partial (\sum_ {i = 1} ^ {m} f _ {i}) (\mathbf {x}) = \sum_ {i = 1} ^ {m} \partial f _ {i} (\mathbf {x}).
$$

Assumptions: $f _ { 1 } , f _ { 2 } , \ldots , f _ { m } : \mathbb { E } \to ( - \infty , \infty ]$ proper convex, $\cap _ { i = 1 } ^ { m } \operatorname { r i } ( \operatorname { d o m } ( f _ { i } ) ) \neq \emptyset$ . [Theorem 3.40] 

• Weak afine transformation rule of subdiferential calculus 

$$
\mathcal {A} ^ {T} (\partial f (\mathcal {A} (\mathbf {x}) + \mathbf {b})) \subseteq \partial h (\mathbf {x}) \quad (h (\mathbf {x}) \equiv f (\mathcal {A} (\mathbf {x}) + \mathbf {b})).
$$

Assumptions: $f , h : \mathbb { E }  ( - \infty , \infty ]$ proper convex, $\mathbf { x } \in \mathrm { d o m } ( h )$ . [Theorem 3.43(a)] 

• Afine transformation rule of subdiferential calculus 

$$
\partial h (\mathbf {x}) = \mathcal {A} ^ {T} (\partial f (\mathcal {A} (\mathbf {x}) + \mathbf {b})) \quad (h (\mathbf {x}) \equiv f (\mathcal {A} (\mathbf {x}) + \mathbf {b})).
$$

Assumptions: $f , h : \mathbb { E }  ( - \infty , \infty ]$ proper convex, $\mathbf { x } \in \mathrm { i n t } ( \mathrm { d o m } ( h ) ) , A ( \mathbf { x } ) + \mathbf { b } \in \mathrm { i n t } ( \mathrm { d o m } ( f ) )$ [Theorem 3.43(b)] 

• Chain rule of subdiferential calculus 

$$
\partial h (\mathbf {x}) = g ^ {\prime} (f (\mathbf {x})) \partial f (\mathbf {x}) \quad (h = g \circ f).
$$

Assumptions: $f : \mathbb { E } \to \mathbb { R }$ convex, $g : \mathbb { R } $ R nondecreasing, diferentiable and convex. [Theorem 3.47] 

• Max rule of subdiferential calculus 

$$
\partial (\max (f _ {1}, f _ {2}, \dots , f _ {m})) (\mathbf {x}) = \operatorname{conv} \left(\cup_ {i \in I (\mathbf {x})} \partial f _ {i} (\mathbf {x})\right),
$$

where 

$$
I (\mathbf {x}) = \left\{i: f _ {i} (\mathbf {x}) = \max \left\{f _ {1} (\mathbf {x}), f _ {2} (\mathbf {x}), \dots , f _ {m} (\mathbf {x}) \right\} \right\}.
$$

Assumptions: $f _ { 1 } , f _ { 2 } , \ldots , f _ { m }$ proper, convex, $\mathbf { x } \in \cap _ { i = 1 } ^ { m } \operatorname { i n t } ( \operatorname { d o m } ( f _ { i } ) )$ . [Theorem 3.50] 

• Weak max rule of subdiferential calculus 

$$
\operatorname{conv} \left(\cup_ {i \in I (\mathbf {x})} \partial f _ {i} (\mathbf {x})\right) \subseteq \partial (\max _ {i \in I} f _ {i}) (\mathbf {x}),
$$

where 

$$
I (\mathbf {x}) = \{i \in I: f _ {i} (\mathbf {x}) = \max _ {i \in I} f _ {i} (\mathbf {x}) \}.
$$

Assumptions: $I = \mathrm { a r b i t r a r y }$ index set. $f _ { i } ~ : ~ \mathbb { E } \ \to \ ( - \infty , \infty ] \ ( i ~ \in ~ I )$ proper, convex, $\textbf { x } \in$ $\cap _ { i \in I } { \mathrm { d o } { \bar { \operatorname { m } } } } ( f _ { i } )$ . [Theorem 3.55] 


The table below contains the main examples from the chapter related to weak results of subgradients computations.


<table><tr><td>Function</td><td>Weak result</td><td>Setting</td><td>Reference</td></tr><tr><td>-q = negative dual function</td><td><eq>-\mathbf{g}(\mathbf{x}_{0}) \in \partial(-q)(\boldsymbol{\lambda}_{0})</eq></td><td><eq>q(\boldsymbol{\lambda}) = \min_{\mathbf{x} \in X} f(\mathbf{x}) + \boldsymbol{\lambda}^{T} \mathbf{g}(\mathbf{x}), f: \mathbb{E} \to \mathbb{R}, \mathbf{g}: \mathbb{E} \to \mathbb{R}^{m}, \mathbf{x}_{0} = \text{a minimizer of } f(\mathbf{x}) + \boldsymbol{\lambda}_{0}^{T} \mathbf{g}(\mathbf{x}) \text{ over } X</eq></td><td>Example 3.7</td></tr><tr><td><eq>f(\mathbf{X}) = \lambda_{\max}(\mathbf{X})</eq></td><td><eq>\mathbf{v}\mathbf{v}^{T} \in \partial f(\mathbf{X})</eq></td><td><eq>f: \mathbb{S}^{n} \to \mathbb{R}, \mathbf{v} = \text{normalized maximum eigenvector of } X \in \mathbb{S}^{n}</eq></td><td>Example 3.8</td></tr><tr><td><eq>f(\mathbf{x}) = \| \mathbf{x} \|_{1}</eq></td><td><eq>\operatorname{sgn}(\mathbf{x}) \in \partial f(\mathbf{x})</eq></td><td><eq>\mathbb{E} = \mathbb{R}^{n}</eq></td><td>Example 3.42</td></tr><tr><td><eq>f(\mathbf{x}) = \lambda_{\max}(\mathbf{A}_{0} + \sum_{i=1}^{m} x_{i} \mathbf{A}_{i})</eq></td><td><eq>(\tilde{\mathbf{y}}^{T} \mathbf{A}_{i} \tilde{\mathbf{y}})^{m}_{i=1} \in \partial f(\mathbf{x})</eq></td><td><eq>\tilde{\mathbf{y}} = \text{normalized maximum eigenvector of } \mathbf{A}_{0} + \sum_{i=1}^{m} x_{i} \mathbf{A}_{i}</eq></td><td>Example 3.56</td></tr></table>


The following table contains the main strong results of subdiferential sets computations derived in this chapter.


<table><tr><td>f(x)</td><td>∂f(x)</td><td>Assumptions</td><td>Reference</td></tr><tr><td>||x||</td><td><eq>B_{\parallel \cdot \|_{*}}[0,1]</eq></td><td>x = 0</td><td>Example 3.3</td></tr><tr><td>||x||1</td><td><eq>\left\{ \sum_{i \in I_{\neq}(\mathbf{x})} \text{sgn}(x_i)\mathbf{e}_i + \sum_{i \in I_0(\mathbf{x})} [-\mathbf{e}_i,\mathbf{e}_i] \right\}</eq></td><td><eq>\mathbb{E} = \mathbb{R}^n</eq>, <eq>I_{\neq}(\mathbf{x}) = \{i : x_i \neq 0\}</eq>, <eq>I_0(\mathbf{x}) = \{i : x_i = 0\}</eq>.</td><td>Example 3.41</td></tr><tr><td>||x||2</td><td><eq>\left\{ \begin{array}{l} \left\{\frac{\mathbf{x}}{\|\mathbf{x}\|_2} \right\}, \quad \mathbf{x} \neq \mathbf{0}, \\ B_{\parallel \cdot \|_2}[0,1], \quad \mathbf{x} = \mathbf{0}. \end{array} \right.</eq></td><td><eq>\mathbb{E} = \mathbb{R}^n</eq></td><td>Example 3.34</td></tr><tr><td>||x||∞</td><td><eq>\left\{ \sum_{i \in I(\mathbf{x})} \lambda_i \text{sgn}(x_i)\mathbf{e}_i : \sum_{i \in I(\mathbf{x})} \lambda_i = 1 \atop \lambda_i \geq 0 \right\}</eq></td><td><eq>\mathbb{E} = \mathbb{R}^n</eq>, <eq>I(\mathbf{x}) = \{i : \| \mathbf{x}\|_\infty = |x_i|\}, \mathbf{x} \neq \mathbf{0}</eq></td><td>Example 3.52</td></tr><tr><td>max(x)</td><td><eq>\left\{ \sum_{i \in I(\mathbf{x})} \lambda_i \mathbf{e}_i : \sum_{i \in I(\mathbf{x})} \lambda_i = 1, \lambda_i \geq 0 \right\}</eq></td><td><eq>\mathbb{E} = \mathbb{R}^n</eq>, <eq>I(\mathbf{x}) = \{i : \max(\mathbf{x}) = x_i\}</eq></td><td>Example 3.51</td></tr><tr><td>max(x)</td><td><eq>\Delta_n</eq></td><td><eq>\mathbb{E} = \mathbb{R}^n</eq>, <eq>\mathbf{x} = \alpha \mathbf{e}</eq> for some <eq>\alpha \in \mathbb{R}</eq></td><td>Example 3.51</td></tr><tr><td><eq>\delta_S(\mathbf{x})</eq></td><td><eq>N_S(\mathbf{x})</eq></td><td><eq>\emptyset \neq S \subseteq \mathbb{E}</eq></td><td>Example 3.5</td></tr><tr><td><eq>\delta_B[\mathbf{0},1](\mathbf{x})</eq></td><td><eq>\left\{ \begin{array}{ll} \{\mathbf{y} \in \mathbb{E}^* : \| \mathbf{y}\|_* \leq \langle \mathbf{y}, \mathbf{x} \rangle\}, &amp; \| \mathbf{x}\| \leq 1, \\ \emptyset, &amp; \| \mathbf{x}\| &gt; 1. \end{array} \right.</eq></td><td></td><td>Example 3.6</td></tr><tr><td>||Ax+b||1</td><td><eq>\sum_{i \in I_{\neq}(\mathbf{x})} \text{sgn}(\mathbf{a}_i^T \mathbf{x} + b_i)\mathbf{a}_i + \sum_{i \in I_0(\mathbf{x})} [-\mathbf{a}_i, \mathbf{a}_i]</eq></td><td><eq>\mathbb{E} = \mathbb{R}^n</eq>, <eq>\mathbf{A} \in \mathbb{R}^{m \times n}</eq>, <eq>\mathbf{b} \in \mathbb{R}^m</eq>, <eq>I_{\neq}(\mathbf{x}) = \{i : \mathbf{a}_i^T \mathbf{x} + b_i \neq 0\}</eq>, <eq>I_0(\mathbf{x}) = \{i : \mathbf{a}_i^T \mathbf{x} + b_i = 0\}</eq></td><td>Example 3.44</td></tr><tr><td>||Ax+b||2</td><td><eq>\left\{ \begin{array}{ll} \frac{\mathbf{A}^T(\mathbf{A}\mathbf{x}+\mathbf{b})}{\|\mathbf{A}\mathbf{x}+\mathbf{b}\|_2}, &amp; \mathbf{A}\mathbf{x} + \mathbf{b} \neq \mathbf{0}, \\ \mathbf{A}^T B_{\parallel \cdot \|_2}[0,1], &amp; \mathbf{A}\mathbf{x} + \mathbf{b} = \mathbf{0}. \end{array} \right.</eq></td><td><eq>\mathbb{E} = \mathbb{R}^n</eq>, <eq>\mathbf{A} \in \mathbb{R}^{m \times n}</eq>, <eq>\mathbf{b} \in \mathbb{R}^m</eq></td><td>Example 3.45</td></tr><tr><td>||Ax+b||∞</td><td><eq>\left\{ \sum_{i \in I_{\mathbf{x}}} \lambda_i \text{sgn}(\mathbf{a}_i^T \mathbf{x} + b_i)\mathbf{a}_i : \sum_{i \in I_{\mathbf{x}}} \lambda_i = 1 \atop \lambda_i \geq 0 \right\}</eq></td><td><eq>\mathbb{E} = \mathbb{R}^n</eq>, <eq>\mathbf{A} \in \mathbb{R}^{m \times n}</eq>, <eq>\mathbf{b} \in \mathbb{R}^m</eq>, <eq>I_{\mathbf{x}} = \{i : \| \mathbf{A}\mathbf{x} + \mathbf{b}\|_\infty = |\mathbf{a}_i^T \mathbf{x} + b_i|\}, \mathbf{A}\mathbf{x} + \mathbf{b} \neq \mathbf{0}</eq></td><td>Example 3.54</td></tr><tr><td>||Ax+b||∞</td><td><eq>\mathbf{A}^T B_{\parallel \cdot \|_1}[0,1]</eq></td><td>same as above but with <eq>\mathbf{A}\mathbf{x} + \mathbf{b} = \mathbf{0}</eq></td><td>Example 3.54</td></tr><tr><td><eq>\max_i\{\mathbf{a}_i^T \mathbf{x} + \mathbf{b}\}</eq></td><td><eq>\left\{ \sum_{i \in I(\mathbf{x})} \lambda_i \mathbf{a}_i : \sum_{i \in I(\mathbf{x})} \lambda_i = 1, \lambda_i \geq 0 \right\}</eq></td><td><eq>\mathbb{E} = \mathbb{R}^n, \mathbf{a}_i \in \mathbb{R}^n</eq>, <eq>b_i \in \mathbb{R}</eq>, <eq>I(\mathbf{x}) = \{i : f(\mathbf{x}) = \mathbf{a}_i^T \mathbf{x} + b_i\}</eq></td><td>Example 3.53</td></tr><tr><td><eq>\frac{1}{2}d_C^2(\mathbf{x})</eq></td><td><eq>\{\mathbf{x} - P_C(\mathbf{x})\}</eq></td><td><eq>C =</eq> nonempty closed and convex, <eq>\mathbb{E} =</eq> Euclidean</td><td>Example 3.31</td></tr><tr><td><eq>d_C(\mathbf{x})</eq></td><td><eq>\left\{ \begin{array}{ll} \left\{\frac{\mathbf{x} - P_C(\mathbf{x})}{d_C(\mathbf{x})}\right\}, &amp; \mathbf{x} \notin C, \\ N_C(\mathbf{x}) \cap B[\mathbf{0},1] &amp; \mathbf{x} \in C. \end{array} \right.</eq></td><td><eq>C =</eq> nonempty closed and convex, <eq>\mathbb{E} =</eq> Euclidean</td><td>Example 3.49</td></tr></table>

# Chapter 4 Conjugate Functions

## 4.1 Definition and Basic Properties

We begin with the definition of the conjugate function. 

Definition 4.1 (conjugate functions). Let $f : \mathbb { E }  [ - \infty , \infty ]$ be an extended real-valued function. The function $f ^ { * } : \mathbb { E } ^ { * }  [ - \infty , \infty ]$ , defined by 

$$
f ^ {*} (\mathbf {y}) = \max _ {\mathbf {x} \in \mathbb {E}} \left\{\langle \mathbf {y}, \mathbf {x} \rangle - f (\mathbf {x}) \right\}, \quad \mathbf {y} \in \mathbb {E} ^ {*},
$$

is called the conjugate function of $f .$ 

Example 4.2 (conjugate of indicator functions). Let $f = \delta _ { C }$ , where $C \subseteq \mathbb { E }$ is nonempty. Then for any $\mathbf { y } \in \mathbb { E } ^ { * }$ 

$$
f ^ {*} (\mathbf {y}) = \max _ {\mathbf {x} \in \mathbb {E}} \{\langle \mathbf {y}, \mathbf {x} \rangle - \delta_ {C} (\mathbf {x}) \} = \max _ {\mathbf {x} \in C} \langle \mathbf {y}, \mathbf {x} \rangle = \sigma_ {C} (\mathbf {y}).
$$

That is, the conjugate of the indicator function is the support function of the same underlying set: 

$$
\delta_ {C} ^ {*} = \sigma_ {C}.
$$

Two fundamental properties of conjugate functions are their convexity and closedness (regardless of whether the original function is closed or convex). 

Theorem 4.3 (convexity and closedness of conjugate functions). Let $f$ : $\mathbb { E } \to ( - \infty , \infty ]$ be an extended real-valued function. Then the conjugate function $f ^ { * }$ is closed and convex. 

Proof. Note that $f ^ { * }$ is the pointwise maximum of afine functions, which are convex and closed, and thus, invoking Theorems $2 . 1 6 ( \mathrm { c } )$ and $2 . 7 ( \mathrm { c } )$ , it follows that $f ^ { * }$ is closed and convex. 

Example 4.4 (conjugate of $\frac { 1 } { 2 } | | \cdot | | ^ { 2 } + \delta _ { C } )$ . Suppose that $\mathbb { E }$ is Euclidean and that $C \subseteq \mathbb { E }$ is nonempty. Define $\begin{array} { r } { f ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { x } \| ^ { 2 } + \delta _ { C } ( \mathbf { x } ) } \end{array}$ . Then by Example 2.17 (specifically, (2.6)), it follows that 

$$
f ^ {*} (\mathbf {y}) = \frac {1}{2} \| \mathbf {y} \| ^ {2} - \frac {1}{2} d _ {C} ^ {2} (\mathbf {y}).
$$

Note that while $f$ is convex only if $C$ is convex, the convexity of $f ^ { * }$ is guaranteed regardless of whether $C$ is convex or not. 

The next result states that the conjugate function of a proper convex function is also proper. 

Theorem 4.5 (properness of conjugate functions). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper convex function. Then $f ^ { * }$ is proper. 

Proof. Since $f$ is proper, it follows that there exists ${ \hat { \mathbf { x } } } \in \mathbb { E }$ such that $f ( \hat { \mathbf { x } } ) < \infty$ By the definition of the conjugate function, for any $\mathbf { y } \in \mathbb { E } ^ { * }$ 

$$
f ^ {*} (\mathbf {y}) \geq \langle \mathbf {y}, \hat {\mathbf {x}} \rangle - f (\hat {\mathbf {x}}),
$$

and hence $f ^ { * } ( \mathbf { y } ) > - \infty$ . What remains in order to establish the properness of $f ^ { * }$ is to show that there exists $\mathbf { g } \in \mathbb { E } ^ { * }$ such that $f ^ { * } ( \mathbf { g } ) < \infty$ . By Corollary 3.19, there exists $\mathbf { x } \in \operatorname { d o m } ( f )$ such that $\partial f ( \mathbf { x } ) \neq \emptyset$ . Take $\mathbf { g } \in \partial f ( \mathbf { x } )$ . Then by the definition of the subgradient, for any $\mathbf { z } \in \mathbb { E }$ , 

$$
f (\mathbf {z}) \geq f (\mathbf {x}) + \langle \mathbf {g}, \mathbf {z} - \mathbf {x} \rangle .
$$

Hence, 

$$
f ^ {*} (\mathbf {g}) = \max _ {\mathbf {z} \in \mathbb {E}} \left\{\langle \mathbf {g}, \mathbf {z} \rangle - f (\mathbf {z}) \right\} \leq \langle \mathbf {g}, \mathbf {x} \rangle - f (\mathbf {x}) <   \infty ,
$$

concluding that $f ^ { * }$ is a proper function. 

The following result, called Fenchel’s inequality, is a trivial implication of the definition of conjugacy. 

Theorem 4.6 (Fenchel’s inequality). Let $f : \mathbb { E }  ( - \infty , \infty ]$ be an extended real-valued proper function. Then for any $\mathbf { x } \in \mathbb { E }$ and $\mathbf { y } \in \mathbb { E } ^ { * }$ 

$$
f (\mathbf {x}) + f ^ {*} (\mathbf {y}) \geq \langle \mathbf {y}, \mathbf {x} \rangle .
$$

Proof. By the definition of the conjugate function we have that for any $\mathbf { x } \in \mathbb { E }$ and $\mathbf { y } \in \mathbb { E } ^ { * }$ 

$$
f ^ {*} (\mathbf {y}) \geq \langle \mathbf {y}, \mathbf {x} \rangle - f (\mathbf {x}).\tag{4.1}
$$

Since $f$ is proper, it follows that $f ( \mathbf { x } ) , f ^ { * } ( \mathbf { y } ) > - \infty$ . We can thus add $f ( \mathbf { x } )$ to both sides of (4.1) and obtain the desired result. 

## 4.2 The Biconjugate

The conjugacy operation can be invoked twice resulting in the biconjugate operation. Specifically, for a function $f : \mathbb { E } \to [ - \infty , \infty ]$ we define (recall that in this book E and $\mathbb { E } ^ { * * }$ are considered to be identical) 

$$
f ^ {* *} (\mathbf {x}) = \max _ {\mathbf {y} \in \mathbb {E} ^ {*}} \left\{\langle \mathbf {x}, \mathbf {y} \rangle - f ^ {*} (\mathbf {y}) \right\}, \quad \mathbf {x} \in \mathbb {E}.
$$

The biconjugate function is always a lower bound on the original function, as the following result states. 

Lemma 4.7 $( f ^ { * * } \leq f )$ . Let $f : \mathbb { E } \to [ - \infty , \infty ]$ be an extended real-valued function. Then $f ( \mathbf { x } ) \geq f ^ { * * } ( \mathbf { x } )$ for any $\mathbf { x } \in \mathbb { E }$ 

Proof. By the definition of the conjugate function we have for any $\mathbf { x } \in \mathbb { E }$ and $\mathbf { y } \in \mathbb { E } ^ { * }$ , 

$$
f ^ {*} (\mathbf {y}) \geq \langle \mathbf {y}, \mathbf {x} \rangle - f (\mathbf {x}).
$$

Thus, 

$$
f (\mathbf {x}) \geq \langle \mathbf {y}, \mathbf {x} \rangle - f ^ {*} (\mathbf {y}),
$$

implying that 

$$
f (\mathbf {x}) \geq \max _ {\mathbf {y} \in \mathbb {E} ^ {*}} \{\langle \mathbf {y}, \mathbf {x} \rangle - f ^ {*} (\mathbf {y}) \} = f ^ {* *} (\mathbf {x}).
$$

If we assume that $f$ is proper closed and convex, then the biconjugate is not just a lower bound on $f { \mathrm { - } } { \mathrm { i t } }$ is equal to $f .$ 

Theorem 4.8 $( f = f ^ { * * }$ for proper closed convex functions). Let $f : \mathbb { E } $ $( - \infty , \infty ]$ be a proper closed and convex function. Then $f ^ { * * } = f$ 

Proof. By Lemma 4.7, $f ^ { * * } \leq f .$ . We thus need to show that $f ^ { * * } \geq f$ . Suppose by contradiction that for some $\mathbf { x } \in \mathbb { E }$ we have $f ^ { * * } ( \mathbf { x } ) < f ( \mathbf { x } )$ . This means that $( \mathbf { x } , f ^ { * * } ( \mathbf { x } ) ) \not \in \mathrm { e p i } ( f ) \subseteq \mathbb { E } \times \mathbb { R }$ . We assume as usual that the product space $\mathbb { V } = \mathbb { E } \times \mathbb { R }$ is endowed with the inner product $\langle ( \mathbf { u } , s ) , ( \mathbf { v } , t ) \rangle _ { \mathbb { V } } = \langle \mathbf { u } , \mathbf { v } \rangle + s t$ , where $\langle \cdot , \cdot \rangle$ is the inner product associated with E (see Section 1.9). Since f is proper closed and convex, the set $\operatorname { e p i } ( f )$ is nonempty closed and convex, and hence, by the strict separation theorem (Theorem 2.33), there exist a $\in \mathbb { E } ^ { * } , b , c _ { 1 } , c _ { 2 } \in \mathbb { R }$ such that 

$$
\langle \mathbf {a}, \mathbf {z} \rangle + b s \leq c _ {1} <   c _ {2} \leq \langle \mathbf {a}, \mathbf {x} \rangle + b f ^ {* *} (\mathbf {x}) \text {   for   all   } (\mathbf {z}, s) \in \operatorname{epi} (f).
$$

We can thus conclude that 

$$
\langle \mathbf {a}, \mathbf {z} - \mathbf {x} \rangle + b (s - f ^ {* *} (\mathbf {x})) \leq c _ {1} - c _ {2} \equiv c <   0 \text {   for   all   } (\mathbf {z}, s) \in \operatorname{epi} (f).\tag{4.2}
$$

The scalar b must be nonpositive, since otherwise, if it was positive, the inequality would have been violated by taking a fixed z and large enough s. We will now consider two cases. 

• If $b < 0$ , then dividing (4.2) by −b and taking $\mathbf { y } = - { \frac { \mathbf { a } } { b } }$ , we get 

$$
\langle \mathbf {y}, \mathbf {z} - \mathbf {x} \rangle - s + f ^ {* *} (\mathbf {x}) \leq \frac {c}{- b} <   0 \text {   for   all   } (\mathbf {z}, s) \in \operatorname{epi} (f).
$$

In particular, taking $s = f ( \mathbf { z } )$ (which is possible since $( \mathbf { z } , f ( \mathbf { z } ) ) \in \mathrm { e p i } ( f ) )$ , we obtain that 

$$
\langle \mathbf {y}, \mathbf {z} \rangle - f (\mathbf {z}) - \langle \mathbf {y}, \mathbf {x} \rangle + f ^ {* *} (\mathbf {x}) \leq \frac {c}{- b} <   0 \text {   for   all   } \mathbf {z} \in \mathbb {E}.
$$

Taking the maximum over z yields the inequality 

$$
f ^ {*} (\mathbf {y}) - \langle \mathbf {y}, \mathbf {x} \rangle + f ^ {* *} (\mathbf {x}) \leq \frac {c}{- b} <   0,
$$

which is a contradiction of Fenchel’s inequality (Theorem 4.6). 

• If $b = 0$ , then take some $\hat { \mathbf { y } } \in \operatorname { d o m } ( f ^ { * } )$ . Such a vector exists since $f ^ { * }$ is proper by the properness and convexity of f (Theorem 4.5). Let $\varepsilon > 0$ and define $\hat { \mathbf { a } } = \mathbf { a } + \varepsilon \hat { \mathbf { y } }$ and $\hat { b } = - \varepsilon$ . Then for any $\mathbf { z } \in \operatorname { d o m } ( f )$ ， 

$$
\begin{array}{r l} & {\langle \hat {\mathbf {a}}, \mathbf {z} - \mathbf {x} \rangle + \hat {b} (f (\mathbf {z}) - f ^ {* *} (\mathbf {x})) = \langle \mathbf {a}, \mathbf {z} - \mathbf {x} \rangle + \varepsilon [ \langle \hat {\mathbf {y}}, \mathbf {z} \rangle - f (\mathbf {z}) + f ^ {* *} (\mathbf {x}) - \langle \hat {\mathbf {y}}, \mathbf {x} \rangle ]} \\ & {\qquad \leq c + \varepsilon [ \langle \hat {\mathbf {y}}, \mathbf {z} \rangle - f (\mathbf {z}) + f ^ {* *} (\mathbf {x}) - \langle \hat {\mathbf {y}}, \mathbf {x} \rangle ]} \\ & {\qquad \leq c + \varepsilon [ f ^ {*} (\hat {\mathbf {y}}) - \langle \hat {\mathbf {y}}, \mathbf {x} \rangle + f ^ {* *} (\mathbf {x}) ],} \end{array}
$$

where the first inequality is due to (4.2) and the second by the definition of $f ^ { * } ( \hat { \mathbf { y } } )$ as the maximum of $\langle \hat { \bf y } , { \bf z } \rangle - f ( { \bf z } )$ over all possible $\textbf { z } \in \ \mathbb { E }$ . We thus obtained the inequality 

$$
\langle \hat {\mathbf {a}}, \mathbf {z} - \mathbf {x} \rangle + \hat {b} (f (\mathbf {z}) - f ^ {* *} (\mathbf {x})) \leq \hat {c},\tag{4.3}
$$

where $\hat { c } \equiv c + \varepsilon \big [ f ^ { * } ( \hat { \mathbf { y } } ) - \langle \hat { \mathbf { y } } , \mathbf { x } \rangle + f ^ { * * } ( \mathbf { x } ) \big ]$ . Since $c < 0$ , we can pick $\varepsilon > 0$ small enough to ensure that $\hat { c } < 0$ . At this point we employ exactly the same argument used in the first case. Dividing (4.3) by −<sup>ˆ</sup>b and denoting $\begin{array} { r } { \tilde { \mathbf { y } } = - \frac { 1 } { \hat { b } } \hat { \mathbf { a } } } \end{array}$ yields the inequality 

$$
\langle \tilde {\mathbf {y}}, \mathbf {z} \rangle - f (\mathbf {z}) - \langle \tilde {\mathbf {y}}, \mathbf {x} \rangle + f ^ {* *} (\mathbf {x}) \leq - \frac {\hat {c}}{\hat {b}} <   0 \text {   for   any   } \mathbf {z} \in \operatorname{dom} (f).
$$

Taking the maximum over z results in 

$$
f ^ {*} (\tilde {\mathbf {y}}) - \langle \tilde {\mathbf {y}}, \mathbf {x} \rangle + f ^ {* *} (\mathbf {x}) \leq \frac {\hat {c}}{- \hat {b}} <   0,
$$

which, again, is a contradiction of Fenchel’s inequality. 

Example 4.9 (conjugate of support functions). We will now show how to exploit Theorem 4.8 in order to compute the conjugate of support functions. Suppose that $C \subseteq \mathbb { E }$ is a given nonempty set. Since cl(conv(C)) is closed and convex, it follows that $\delta _ { \mathrm { c l ( c o n v ( } C ) ) }$ is closed and convex, and hence, by Example 4.2 and Theorem 4.8, 

$$
\sigma_ {\mathrm{cl} (\mathrm{conv} (C))} ^ {*} = \bigl (\delta_ {\mathrm{cl} (\mathrm{conv} (C))} ^ {*} \bigr) ^ {*} = \delta_ {\mathrm{cl} (\mathrm{conv} (C))} ^ {* *} = \delta_ {\mathrm{cl} (\mathrm{conv} (C))}.\tag{4.4}
$$

Finally, by Lemma 2.35, 

$$
\sigma_ {C} = \sigma_ {\mathrm{cl} (\mathrm{conv} (C))},
$$

which, combined with (4.4), establishes the result 

$$
\sigma_ {C} ^ {*} = \delta_ {\mathrm{cl} (\mathrm{conv} (C))}.
$$

Example 4.10 (conjugate of the max function). Consider the function $f$ : $\mathbb { R } ^ { n } \to \mathbb { R }$ given by $f ( \mathbf { x } ) = \operatorname* { m a x } \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { n } \}$ . Note that the following elementary identity holds for any $\mathbf { x } \in \mathbb { R } ^ { n }$ 

$$
\max \{x _ {1}, x _ {2}, \ldots , x _ {n} \} = \max _ {\mathbf {y} \in \Delta_ {n}} \mathbf {y} ^ {T} \mathbf {x} = \sigma_ {\Delta_ {n}} (\mathbf {x}).
$$

Therefore, using Example 4.9, we can conclude, exploiting the convexity and closedness of $\Delta _ { n } .$ , that 

$$
f ^ {*} = \delta_ {\Delta_ {n}}.
$$

Example 4.11 (conjugate of $\frac { 1 } { 2 } | | \cdot | | ^ { 2 } - d _ { C } ^ { 2 } )$ . Let E be Euclidean, and let $C \subseteq \mathbb { E }$ be a nonempty closed and convex set. Define $\begin{array} { r } { f ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { x } \| ^ { 2 } - \frac { 1 } { 2 } d _ { C } ^ { 2 } ( \mathbf { x } ) } \end{array}$ . By Example 4.4, $f = g ^ { * }$ , where $\begin{array} { r } { g ( \mathbf { y } ) = \frac { 1 } { 2 } \| \mathbf { y } \| ^ { 2 } + \delta _ { C } ( \mathbf { y } ) } \end{array}$ . By the nonemptiness, closedness, and convexity of $C .$ , it follows that $g$ is proper closed and convex, and hence, by Theorem 4.8, 

$$
f ^ {*} (\mathbf {y}) = g ^ {* *} (\mathbf {y}) = g (\mathbf {y}) = \frac {1}{2} \| \mathbf {y} \| ^ {2} + \delta_ {C} (\mathbf {y}).
$$

## 4.3 Conjugate Calculus Rules

In this section we present the basic calculus rules for computing conjugate functions. We begin with a very simple rule for separable functions. 

Theorem 4.12 (conjugate of separable functions). Let $g : \mathbb { E } _ { 1 } \times \mathbb { E } _ { 2 } \times \cdot \cdot \cdot \times \mathbb { E } _ { p } \to$ $( - \infty , \infty ]$ be given by $\begin{array} { r } { g ( \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { p } ) = \sum _ { i = 1 } ^ { p } f _ { i } ( \mathbf { x } _ { i } ) } \end{array}$ , where $f _ { i } : \mathbb { E } _ { i }  ( - \infty , \infty ]$ is a proper function for any $i = 1 , 2 , \dotsc , p$ . Then 

$$
g ^ {*} (\mathbf {y} _ {1}, \mathbf {y} _ {2}, \dots , \mathbf {y} _ {p}) = \sum_ {i = 1} ^ {p} f _ {i} ^ {*} (\mathbf {y} _ {i}) \text {   for   any   } \mathbf {y} _ {i} \in \mathbb {E} _ {i} ^ {*}, \quad i = 1, 2, \dots , p.
$$

Proof. For any $( \mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } , \ldots , \mathbf { y } _ { p } ) \in \mathbb { E } _ { 1 } ^ { * } \times \mathbb { E } _ { 2 } ^ { * } \times \dots \times \mathbb { E } _ { p } ^ { * }$ , it holds that 

$$
\begin{array}{l} g ^ {*} (\mathbf {y} _ {1}, \mathbf {y} _ {2}, \ldots , \mathbf {y} _ {p}) = \max _ {\mathbf {x} _ {1}, \mathbf {x} _ {2}, \ldots , \mathbf {x} _ {p}} \{\langle (\mathbf {y} _ {1}, \mathbf {y} _ {2}, \ldots , \mathbf {y} _ {p}), (\mathbf {x} _ {1}, \mathbf {x} _ {2}, \ldots , \mathbf {x} _ {p}) \rangle - g (\mathbf {x} _ {1}, \mathbf {x} _ {2}, \ldots , \mathbf {x} _ {p}) \} \\ = \max _ {\mathbf {x} _ {1}, \mathbf {x} _ {2}, \ldots , \mathbf {x} _ {p}} \left\{\sum_ {i = 1} ^ {p} \langle \mathbf {y} _ {i}, \mathbf {x} _ {i} \rangle - \sum_ {i = 1} ^ {p} f _ {i} (\mathbf {x} _ {i}) \right\} \\ = \sum_ {i = 1} ^ {p} \max _ {\mathbf {x} _ {i}} \{\langle \mathbf {y} _ {i}, \mathbf {x} _ {i} \rangle - f _ {i} (\mathbf {x} _ {i}) \} \\ = \sum_ {i = 1} ^ {p} f _ {i} ^ {*} (\mathbf {y} _ {i}). \quad \square \end{array}
$$

The next result shows how the conjugate operation is afected by invertible afine change of variables as well as by addition of an afine function. 

Theorem 4.13 (conjugate of $f ( A ( \mathbf { x } - \mathbf { a } ) ) + \langle \mathbf { b } , \mathbf { x } \rangle + c )$ . Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be an extended real-valued function, $a n d \ l e t \ A : \ \mathbb { V } \to \mathbb { E }$ be an invertible linear transformation, $\mathbf { a } \in \mathbb { V } , \ \mathbf { b } \in \mathbb { V } ^ { * }$ , and $c \in \mathbb { R }$ . Then the conjugate of the function $g ( \mathbf { x } ) = f ( A ( \mathbf { x } - \mathbf { a } ) ) + \langle \mathbf { b } , \mathbf { x } \rangle + c$ is given by 

$$
g ^ {*} (\mathbf {y}) = f ^ {*} \left((\mathcal {A} ^ {T}) ^ {- 1} (\mathbf {y} - \mathbf {b})\right) + \langle \mathbf {a}, \mathbf {y} \rangle - c - \langle \mathbf {a}, \mathbf {b} \rangle , \quad \mathbf {y} \in \mathbb {V} ^ {*}.
$$

Proof. Making the change of variables $\mathbf { z } ~ = ~ \mathcal { A } ( \mathbf { x } - \mathbf { a } )$ , which is equivalent to $\mathbf { x } = \mathcal { A } ^ { - 1 } ( \mathbf { z } ) + \mathbf { a }$ , we can write for any $\mathbf { y } \in \mathbb { V } ^ { * }$ 

$$
\begin{array}{l} g ^ {*} (\mathbf {y}) = \underset {\mathbf {x}} {\max} \{\langle \mathbf {y}, \mathbf {x} \rangle - g (\mathbf {x}) \} \\ \qquad = \underset {\mathbf {x}} {\max} \{\langle \mathbf {y}, \mathbf {x} \rangle - f (\mathcal {A} (\mathbf {x} - \mathbf {a})) - \langle \mathbf {b}, \mathbf {x} \rangle - c \} \\ \qquad = \underset {\mathbf {z}} {\max} \{\langle \mathbf {y}, \mathcal {A} ^ {- 1} (\mathbf {z}) + \mathbf {a} \rangle - f (\mathbf {z}) - \langle \mathbf {b}, \mathcal {A} ^ {- 1} (\mathbf {z}) + \mathbf {a} \rangle - c \} \\ \qquad = \underset {\mathbf {z}} {\max} \left\{\langle \mathbf {y} - \mathbf {b}, \mathcal {A} ^ {- 1} (\mathbf {z}) \rangle - f (\mathbf {z}) + \langle \mathbf {a}, \mathbf {y} \rangle - \langle \mathbf {a}, \mathbf {b} \rangle - c \right\} \\ \qquad = \underset {\mathbf {z}} {\max} \left\{\langle (\mathcal {A} ^ {- 1}) ^ {T} (\mathbf {y} - \mathbf {b}), \mathbf {z} \rangle - f (\mathbf {z}) + \langle \mathbf {a}, \mathbf {y} \rangle - \langle \mathbf {a}, \mathbf {b} \rangle - c \right\} \\ \qquad = f ^ {*} ((\mathcal {A} ^ {T}) ^ {- 1} (\mathbf {y} - \mathbf {b})) + \langle \mathbf {a}, \mathbf {y} \rangle - c - \langle \mathbf {a}, \mathbf {b} \rangle , \end{array}
$$

where in the last equality we also used the fact that $( \mathcal { A } ^ { - 1 } ) ^ { T } = ( \mathcal { A } ^ { T } ) ^ { - 1 }$ 

Theorem 4.14 (conjugate of $\alpha f ( \cdot )$ and $\alpha f ( \cdot / \alpha ) )$ . Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be an extended real-valued function and let $\alpha \in \mathbb { R } _ { + + }$ 

(a) The conjugate of the function $g ( \mathbf { x } ) = \alpha f ( \mathbf { x } )$ is given by 

$$
g ^ {*} (\mathbf {y}) = \alpha f ^ {*} \left(\frac {\mathbf {y}}{\alpha}\right), \quad \mathbf {y} \in \mathbb {E} ^ {*}.
$$

(b) The conjugate of the function $\begin{array} { r } { h ( \mathbf { x } ) = \alpha f \left( \frac { \mathbf { x } } { \alpha } \right) } \end{array}$ is given by 

$$
h ^ {*} (\mathbf {y}) = \alpha f ^ {*} (\mathbf {y}), \quad \mathbf {y} \in \mathbb {E} ^ {*}.
$$

Proof. For any $\mathbf { y } \in \mathbb { E } ^ { * }$ 

$$
\begin{array}{l} g ^ {*} (\mathbf {y}) = \underset {\mathbf {x}} {\max} \{\langle \mathbf {y}, \mathbf {x} \rangle - g (\mathbf {x}) \} \\ \qquad = \underset {\mathbf {x}} {\max} \{\langle \mathbf {y}, \mathbf {x} \rangle - \alpha f (\mathbf {x}) \} \\ \qquad = \alpha \underset {\mathbf {x}} {\max} \left\{\left\langle \frac {\mathbf {y}}{\alpha}, \mathbf {x} \right\rangle - f (\mathbf {x}) \right\} \\ \qquad = \alpha f ^ {*} \left(\frac {\mathbf {y}}{\alpha}\right), \end{array}
$$

proving (a). The proof of (b) follows by the following chain of equalities: 

$$
\begin{array}{r c l} h ^ {*} (\mathbf {y}) & = & \max _ {\mathbf {x}} \{\langle \mathbf {y}, \mathbf {x} \rangle - h (\mathbf {x}) \} \\ & = & \max _ {\mathbf {x}} \left\{\langle \mathbf {y}, \mathbf {x} \rangle - \alpha f \left(\frac {\mathbf {x}}{\alpha}\right) \right\} \\ & = & \alpha \max _ {\mathbf {x}} \left\{\left\langle \mathbf {y}, \frac {\mathbf {x}}{\alpha} \right\rangle - f \left(\frac {\mathbf {x}}{\alpha}\right) \right\} \\ & = & \alpha \max _ {\mathbf {z}} \left\{\langle \mathbf {y}, \mathbf {z} \rangle - f (\mathbf {z}) \right\} \\ & = & \alpha f ^ {*} (\mathbf {y}). \quad \square \end{array}
$$

The table below summarizes the four calculus rules discussed in this section. 

<table><tr><td>g(x)</td><td>g*(y)</td><td>Reference</td></tr><tr><td><eq>\sum_{i=1}^{m} f_i(x_i)</eq></td><td><eq>\sum_{i=1}^{m} f_i^*(y_i)</eq></td><td>Theorem 4.12</td></tr><tr><td><eq>\alpha f(x) (\alpha &gt;0)</eq></td><td><eq>\alpha f^*(y/\alpha)</eq></td><td>Theorem 4.14</td></tr><tr><td><eq>\alpha f(x/\alpha) (\alpha &gt;0)</eq></td><td><eq>\alpha f^*(y)</eq></td><td>Theorem 4.14</td></tr><tr><td><eq>f(\mathcal{A}(x-a)) + \langle b,x\rangle + c</eq></td><td><eq>f^*((\mathcal{A}^T)^{-1}(y-b)) + \langle a,y\rangle - c - \langle a,b\rangle</eq></td><td>Theorem 4.13</td></tr></table>

## 4.4 Examples

In this section we compute the conjugate functions of several fundamental convex functions. The first examples are one-dimensional, while the rest are multidimensional. 

## 4.4.1 Exponent

Let $f : \mathbb { R } \to \mathbb { R }$ be given by $f ( x ) = e ^ { x }$ . Then for any $y \in \mathbb { R }$ 2 

$$
f ^ {*} (y) = \max _ {x} \left\{x y - e ^ {x} \right\}.\tag{4.5}
$$

If $y < 0$ , then the maximum value of the above problem is $\infty$ (easily seen by taking $x \to - \infty )$ . If $y = 0$ , then obviously the maximal value (which is not attained) is $0 . ~ \mathrm { ~ H ~ } y ~ > ~ 0$ , the unique maximizer of (4.5) is $x = \tilde { x } \equiv \log y$ . Consequently, $f ^ { * } ( y ) = \tilde { x } y - e ^ { \tilde { x } } = y \log y - y$ for any $y > 0$ . Using the convention 0 log $0 \equiv 0$ , we can finally deduce that 

$$
f ^ {*} (y) = \left\{ \begin{array}{l l} y \log y - y, & y \geq 0, \\ \infty & \text { else }. \end{array} \right.
$$

## 4.4.2 Negative Log

Let $f : \mathbb { R } \to ( - \infty , \infty ]$ be given by 

$$
f (x) = \left\{ \begin{array}{l l} - \log (x), & x > 0, \\ \infty , & x \leq 0. \end{array} \right.
$$

For any $y \in \mathbb R$ 

$$
f ^ {*} (y) = \max _ {x > 0} \{x y - f (x) \} = \max _ {x > 0} \{x y + \log (x) \}.\tag{4.6}
$$

If $y \geq 0$ , then the maximum value of the above problem is $\infty$ (since the objective function in (4.6) goes to ∞ as $x \to \infty )$ . If $y < 0$ , the unique optimal solution of (4.6) is attained at $\begin{array} { r } { \tilde { x } = - \frac { 1 } { y } } \end{array}$ , and hence for $y < 0$ we have $f ^ { * } ( y ) = \tilde { x } y + \log ( \tilde { x } ) =$ $- 1 - \log ( - y )$ . To conclude, 

$$
f ^ {*} (y) = \left\{ \begin{array}{l l} - 1 - \log (- y), & y <   0, \\ \infty , & y \geq 0. \end{array} \right.
$$

## 4.4.3 Hinge Loss

Consider the one-dimensional function $f : \mathbb { R } \to \mathbb { R }$ given by 

$$
f (x) = \max \{1 - x, 0 \}.
$$

Then for any $y \in \mathbb R$ 

$$
f ^ {*} (y) = \max _ {x} [ y x - \max \{1 - x, 0 \} ] = \max _ {x} [ \min \{(1 + y) x - 1, y x \} ].\tag{4.7}
$$

The objective function in the above maximization problem can be rewritten as 

$$
\min \left\{(1 + y) x - 1, y x \right\} = \left\{ \begin{array}{l l} (1 + y) x - 1, & x <   1, \\ y x, & x \geq 1. \end{array} \right.
$$

Thus, the objective function is a continuous piecewise linear function comprising two pieces: a line with slope $1 + y$ over $( - \infty , 1 ]$ and a line with slope y over $\lbrack 1 , \infty )$ Therefore, a maximizer exists if the slope of the left line is nonnegative $( 1 + y \ge 0 )$ and the slope of the right line is nonpositive $( y \leq 0 )$ . Consequently, a maximizer exists for the problem in (4.7) if and only if $y \in [ - 1 , 0 ]$ , and in that case it is attained at $x = 1$ , with y being the corresponding optimal value. To summarize, 

$$
f ^ {*} (y) = y + \delta_ {[ - 1, 0 ]} (y), \quad y \in \mathbb {R}.
$$

## 4.4.4 $\begin{array} { r } { \frac { 1 } { p } \big | \cdot \big | ^ { p } ( p > 1 ) \ } \end{array}$

Let $f : \mathbb { R } \to \mathbb { R }$ be given by $\begin{array} { r } { f ( x ) = \frac { 1 } { p } | x | ^ { p } } \end{array}$ , where $p > 1$ . For any $y \in \mathbb R$ 

$$
f ^ {*} (y) = \max _ {x} \left\{x y - \frac {1}{p} | x | ^ {p} \right\}.\tag{4.8}
$$

Since the problem in (4.8) consists of maximizing a diferentiable concave function over R, its optimal solutions are the points $\tilde { x }$ in which the derivative vanishes: 

$$
y - \mathrm{sgn} (\tilde {x}) | \tilde {x} | ^ {p - 1} = 0.
$$

Therefore, $\operatorname { s g n } ( \tilde { x } ) = \operatorname { s g n } ( y )$ and $| \tilde { x } | ^ { p - 1 } = | y |$ , implying that $\tilde { x } = \mathrm { s g n } ( y ) | y | ^ { \frac { 1 } { p - 1 } }$ . Thus, 

$$
f ^ {*} (y) = \tilde {x} y - \frac {1}{p} | \tilde {x} | ^ {p} = | y | ^ {1 + \frac {1}{p - 1}} - \frac {1}{p} | y | ^ {\frac {p}{p - 1}} = \left(1 - \frac {1}{p}\right) | y | ^ {\frac {p}{p - 1}} = \frac {1}{q} | y | ^ {q},
$$

where $q$ is the positive number satisfying $\begin{array} { r } { \frac { 1 } { p } + \frac { 1 } { q } = 1 } \end{array}$ . To summarize, 

$$
f ^ {*} (y) = \frac {1}{q} | y | ^ {q}, \quad y \in \mathbb {R}.
$$

4.4.5 $\begin{array} { r l } { - \frac { ( \cdot ) ^ { p } } { p } } & { { } ( 0 < p < 1 ) } \end{array}$ 

Let $f : \mathbb { R } \to ( - \infty , \infty ]$ be given by 

$$
f (x) = \left\{ \begin{array}{l l} - \frac {x ^ {p}}{p}, & x \geq 0, \\ \infty , & x <   0. \end{array} \right.
$$

For any $y \in \mathbb R$ 

$$
f ^ {*} (y) = \max _ {x} \{x y - f (x) \} = \max _ {x \geq 0} \left\{g (x) \equiv x y + \frac {x ^ {p}}{p} \right\}.
$$

When $y \geq 0$ , the value of the above problem is ∞ since $g ( x )  \infty$ as $x \to \infty$ . If $y < 0$ , then the derivative of $g ( x )$ vanishes at $x = \tilde { x } \equiv ( - y ) ^ { \frac { 1 } { p - 1 } } > 0$ , and since $g$ is concave, it follows that $\tilde { x }$ is a global maximizer of g. Therefore, 

$$
f ^ {*} (y) = \tilde {x} y + \frac {\tilde {x} ^ {p}}{p} = - (- y) ^ {\frac {p}{p - 1}} + \frac {1}{p} (- y) ^ {\frac {p}{p - 1}} = - \frac {(- y) ^ {q}}{q},
$$

where $q$ is the negative number for which $\begin{array} { r } { \frac { 1 } { p } + \frac { 1 } { q } = 1 } \end{array}$ . To summarize, 

$$
f ^ {*} (y) = \left\{ \begin{array}{l l} - \frac {(- y) ^ {q}}{q}, & y <   0, \\ \infty , & \text { else }. \end{array} \right.
$$

## 4.4.6 Strictly Convex Quadratic Functions

Let $f : \mathbb { R } ^ { n }  \mathbb { R }$ be given by $\begin{array} { r } { f ( \mathbf { x } ) = \frac { 1 } { 2 } \mathbf { x } ^ { T } \mathbf { A } \mathbf { x } + \mathbf { b } ^ { T } \mathbf { x } + c . } \end{array}$ , where $\mathbf { A } \in \mathbb { S } _ { + + } ^ { n } , \mathbf { b } \in \mathbb { R } ^ { n }$ and $c \in \mathbb { R }$ . We use our convention that (unless otherwise stated) $\mathbb { R } ^ { n }$ is endowed with the dot product, meaning that $\langle \mathbf { x } , \mathbf { y } \rangle = \mathbf { x } ^ { T } \mathbf { y }$ . For any $\mathbf { y } \in \mathbb { R } ^ { n }$ 

$$
\begin{array}{l} f ^ {*} (\mathbf {y}) = \underset {\mathbf {x}} {\max} \{\mathbf {y} ^ {T} \mathbf {x} - f (\mathbf {x}) \} \\ \qquad = \underset {\mathbf {x}} {\max} \left\{\mathbf {y} ^ {T} \mathbf {x} - \frac {1}{2} \mathbf {x} ^ {T} \mathbf {A x} - \mathbf {b} ^ {T} \mathbf {x} - c \right\} \\ \qquad = \underset {\mathbf {x}} {\max} \left\{- \frac {1}{2} \mathbf {x} ^ {T} \mathbf {A x} - (\mathbf {b} - \mathbf {y}) ^ {T} \mathbf {x} - c \right\}. \end{array}
$$

The maximum in the above problem is attained at $\mathbf { x } = \mathbf { A } ^ { - 1 } \left( \mathbf { y } - \mathbf { b } \right)$ , leading to the following expression for the conjugate function: 

$$
f ^ {*} (\mathbf {y}) = \frac {1}{2} (\mathbf {y} - \mathbf {b}) ^ {T} \mathbf {A} ^ {- 1} (\mathbf {y} - \mathbf {b}) - c.
$$

## 4.4.7 Convex Quadratic Functions

Let $f : \mathbb { R } ^ { n } $ R be given by $\begin{array} { r } { f ( \mathbf { x } ) = \frac { 1 } { 2 } \mathbf { x } ^ { T } \mathbf { A } \mathbf { x } + \mathbf { b } ^ { T } \mathbf { x } + c _ { \dagger } } \end{array}$ , where $\mathbf { A } \in \mathbb { S } _ { + } ^ { n }$ , b $\in \mathbb { R } ^ { n }$ and $c \in \mathbb { R }$ . The only diference between this example and the previous one is the fact that here A is not necessarily positive definite but is assumed to be only positive semidefinite. For any $\mathbf { y } \in \mathbb { R } ^ { n }$ 

$$
f ^ {*} (\mathbf {y}) = \max _ {\mathbf {x}} \{\mathbf {y} ^ {T} \mathbf {x} - f (\mathbf {x}) \} = \max _ {\mathbf {x}} \left\{g (\mathbf {x}) \equiv - \frac {1}{2} \mathbf {x} ^ {T} \mathbf {A x} + (\mathbf {y} - \mathbf {b}) ^ {T} \mathbf {x} - c \right\}.
$$

Since $g$ is concave and diferentiable over $\mathbb { R } ^ { n }$ , it follows that the maximizers of the above problem are the points for which the gradient vanishes, namely, points x satisfying 

$$
\mathbf {A} \mathbf {x} = \mathbf {y} - \mathbf {b}.\tag{4.9}
$$

This system has a solution if and only if $\mathbf { y } \in \mathbf { b } + \mathrm { R a n g e } ( \mathbf { A } )$ , and in that case we can choose one of the solutions to the system (4.9), for example, $\tilde { \mathbf { x } } = \mathbf { A } ^ { \dagger } ( \mathbf { y } - \mathbf { b } )$ where $\mathbf { A } ^ { \dagger }$ is the Moore–Penrose pseudoinverse of A. We can now compute $f ^ { * } ( \mathbf { y } )$ as follows: 

$$
\begin{array}{l} f ^ {*} (\mathbf {y}) = - \frac {1}{2} \tilde {\mathbf {x}} ^ {T} \mathbf {A} \tilde {\mathbf {x}} - (\mathbf {b} - \mathbf {y}) ^ {T} \tilde {\mathbf {x}} - c \\ \qquad = - \frac {1}{2} (\mathbf {y} - \mathbf {b}) \mathbf {A} ^ {\dagger} \mathbf {A} \mathbf {A} ^ {\dagger} (\mathbf {y} - \mathbf {b}) - (\mathbf {b} - \mathbf {y}) ^ {T} \mathbf {A} ^ {\dagger} (\mathbf {y} - \mathbf {b}) - c \\ \qquad = \frac {1}{2} (\mathbf {y} - \mathbf {b}) ^ {T} \mathbf {A} ^ {\dagger} (\mathbf {y} - \mathbf {b}) - c, \end{array}
$$

where we used the fact that the Moore–Penrose pseudoinverse of a symmetric matrix is symmetric, as well as the known identity $\mathbf { A } ^ { \dagger } \mathbf { A } \mathbf { A } ^ { \dagger } = \mathbf { A } ^ { \dagger }$ . We are left with the case where $\mathbf { y } - \mathbf { b } \notin \mathrm { R a n g e } ( \mathbf { A } )$ . We will show that in this case $f ^ { * } ( \mathbf { y } ) = \infty$ . Indeed, since $\operatorname { R a n g e } ( \mathbf { A } ) = \operatorname { N u l l } ( \mathbf { A } ) ^ { \perp }$ , it follows that $\mathbf { y } - \mathbf { b } \not \in \mathrm { N u l l } ( \mathbf { A } ) ^ { \perp }$ , meaning that there exists a vector $\mathbf { v } \in \mathrm { N u l l } ( \mathbf { A } )$ such that $( \mathbf { y } - \mathbf { b } ) ^ { T } \mathbf { v } > 0$ . Note that for any $\alpha \in \mathbb { R }$ 2 

$$
g (\alpha \mathbf {v}) = \alpha (\mathbf {y} - \mathbf {b}) ^ {T} \mathbf {v} - c,
$$

and hence $g ( \alpha \mathbf { v } )  \infty$ as α tends to $\infty { \mathrm { , } }$ , establishing the fact that $f ^ { * } ( \mathbf { y } ) = \infty$ whenever $\mathbf { y } \not \in \mathbf { b } + \mathrm { R a n g e } ( \mathbf { A } )$ . To conclude, 

$$
f ^ {*} (\mathbf {y}) = \left\{ \begin{array}{l l} \frac {1}{2} (\mathbf {y} - \mathbf {b}) ^ {T} \mathbf {A} ^ {\dagger} (\mathbf {y} - \mathbf {b}) - c, & \mathbf {y} \in \mathbf {b} + \operatorname{Range} (\mathbf {A}), \\ \infty & \text {else.} \end{array} \right.
$$

## 4.4.8 Negative Entropy

Let $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ be given by 

$$
f (\mathbf {x}) = \left\{ \begin{array}{l l} \sum_ {i = 1} ^ {n} x _ {i} \log x _ {i}, & \mathbf {x} \geq \mathbf {0}, \\ \infty & \text { else }. \end{array} \right.
$$

Since the function is separable, it is enough to compute the conjugate of the scalar function $g$ defined by $g ( t ) = t \log t$ for $t \geq 0$ and $\infty$ for $t < 0$ . For any $s \in \mathbb { R }$ à， 

$$
g ^ {*} (s) = \max _ {t} \{t s - g (t) \} = \max _ {t \geq 0} \{t s - t \log t \}.
$$

The maximum of the above problem is attained at $t = e ^ { s - 1 }$ , and hence the conjugate is given by 

$$
g ^ {*} (s) = s e ^ {s - 1} - (s - 1) e ^ {s - 1} = e ^ {s - 1}.
$$

Since $\begin{array} { r } { f ( \mathbf { x } ) = \sum _ { i = 1 } ^ { n } g ( x _ { i } ) } \end{array}$ , it follows by Theorem 4.12 that for any $\mathbf { y } \in \mathbb { R } ^ { n }$ ， 

$$
f ^ {*} (\mathbf {y}) = \sum_ {i = 1} ^ {n} g ^ {*} (y _ {i}) = \sum_ {i = 1} ^ {n} e ^ {y _ {i} - 1}.
$$

## 4.4.9 Negative Sum of Logs

Let $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ be given by 

$$
f (\mathbf {x}) = \left\{ \begin{array}{l l} - \sum_ {i = 1} ^ {n} \log x _ {i}, & \mathbf {x} > \mathbf {0}, \\ \infty & \text { else }. \end{array} \right.
$$

Note that $\begin{array} { r } { f ( \mathbf { x } ) = \sum _ { i = 1 } ^ { n } g ( x _ { i } ) } \end{array}$ , where $g ( t ) = - \log t$ for $t > 0$ and $\infty$ for $t \leq 0$ Therefore, invoking Theorem 4.12, 

$$
f ^ {*} (\mathbf {x}) = \sum_ {i = 1} ^ {n} g ^ {*} (x _ {i}).
$$

By Section 4.4.2, $g ^ { * } ( y ) = - 1 - \log ( - y )$ for $y < 0$ and $\infty$ otherwise. Therefore, 

$$
f ^ {*} (\mathbf {y}) = \left\{ \begin{array}{l l} - n - \sum_ {i = 1} ^ {n} \log (- y _ {i}), & \mathbf {y} <   \mathbf {0}, \\ \infty & \text { else }. \end{array} \right.
$$

## 4.4.10 Negative Entropy over the Unit Simplex

Let $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ be given by 

$$
f (\mathbf {x}) = \left\{ \begin{array}{l l} \sum_ {i = 1} ^ {n} x _ {i} \log x _ {i}, & \mathbf {x} \in \Delta_ {n}, \\ \infty & \text { else }. \end{array} \right.\tag{4.10}
$$

For any $\mathbf { y } \in \mathbb { R } ^ { n }$ 

$$
f ^ {*} (\mathbf {y}) = \max \left\{\sum_ {i = 1} ^ {n} y _ {i} x _ {i} - \sum_ {i = 1} ^ {n} x _ {i} \log x _ {i}: \sum_ {i = 1} ^ {n} x _ {i} = 1, x _ {1}, x _ {2}, \dots , x _ {n} \geq 0 \right\}.
$$

By Example 3.71, the optimal solution of the above maximization problem is 

$$
x _ {i} ^ {*} = \frac {e ^ {y _ {i}}}{\sum_ {j = 1} ^ {n} e ^ {y _ {j}}}, \quad i = 1, 2, \ldots , n,
$$

with a corresponding optimal value of 

$$
f ^ {*} (\mathbf {y}) = \sum_ {i = 1} ^ {n} y _ {i} x _ {i} ^ {*} - \sum_ {i = 1} ^ {n} x _ {i} ^ {*} \log x _ {i} ^ {*} = \log \left(\sum_ {j = 1} ^ {n} e ^ {y _ {j}}\right).
$$

That is, the conjugate of the negative entropy is the log-sum-exp function. 

## 4.4.11 log-sum-exp

Let $g : \mathbb { R } ^ { n } $ R be given by 

$$
g (\mathbf {x}) = \log \left(\sum_ {j = 1} ^ {n} e ^ {x _ {j}}\right).
$$

By Section 4.4.10, $g = f ^ { * }$ , where f is the negative entropy over the unit simplex given by (4.10). Since $f$ is proper closed and convex, it follows by Theorem 4.8 that $f ^ { * * } = f$ , and hence 

$$
g ^ {*} = f ^ {* *} = f,
$$

meaning that 

$$
g ^ {*} (\mathbf {y}) = \left\{ \begin{array}{l l} \sum_ {i = 1} ^ {n} y _ {i} \log y _ {i}, & \mathbf {y} \in \Delta_ {n}, \\ \infty & \text { else }. \end{array} \right.
$$

## 4.4.12 Norms

Let $f : \mathbb { E } \to \mathbb { R }$ be given by $f ( \mathbf { x } ) = \| \mathbf { x } \|$ . Then, by Example 2.31, 

$$
f = \sigma_ {B _ {\| \cdot \| *} [ \mathbf {0}, 1 ]},
$$

where we used the fact that the bidual norm $\| \cdot \| _ { * * }$ is identical to the norm $\| \cdot \|$ Hence, by Example 4.9, 

$$
f ^ {*} = \delta_ {\mathrm{cl} (\mathrm{conv} (B _ {\| \cdot \| *} [ \mathbf {0}, 1 ]))},
$$

but since $B _ { \parallel \cdot \parallel _ { * } } [ \mathbf { 0 } , 1 ]$ is closed and convex, cl(conv $( B _ { \parallel \cdot \parallel _ { * } } [ \mathbf { 0 } , 1 ] ) ) = B _ { \parallel \cdot \parallel _ { * } } [ \mathbf { 0 } , 1 ]$ , and therefore for any $\mathbf { y } \in \mathbb { E } ^ { * }$ , 

$$
f ^ {*} (\mathbf {y}) = \delta_ {B _ {\| \cdot \| _ {*}} [ \mathbf {0}, 1 ]} (\mathbf {y}) = \left\{ \begin{array}{l l} 0, & \| \mathbf {y} \| _ {*} \leq 1, \\ \infty & \text { else }. \end{array} \right.
$$

## 4.4.13 Ball-Pen

Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be given by 

$$
f (\mathbf {x}) = \left\{ \begin{array}{l l} - \sqrt {1 - \| \mathbf {x} \| ^ {2}}, & \| \mathbf {x} \| \leq 1, \\ \infty & \text { else }. \end{array} \right.
$$

To compute the conjugate function, we begin by rewriting it in a double maximization form: 

$$
\begin{array}{r} f ^ {*} (\mathbf {y}) = \max _ {\mathbf {x}} \left\{\langle \mathbf {y}, \mathbf {x} \rangle + \sqrt {1 - \| \mathbf {x} \| ^ {2}}: \| \mathbf {x} \| \leq 1 \right\} \\ = \max _ {\alpha \in [ 0, 1 ]} \max _ {\mathbf {x}: \| \mathbf {x} \| = \alpha} \left\{\langle \mathbf {y}, \mathbf {x} \rangle + \sqrt {1 - \alpha^ {2}} \right\}. \end{array}
$$

By the definition of the dual norm, the optimal value of the inner maximization problem is $\alpha \| \mathbf { y } \| _ { * } + { \sqrt { 1 - \alpha ^ { 2 } } }$ , and we can therefore write, for any $\mathbf { y } \in \mathbb { E } ^ { * }$ , 

$$
f ^ {*} (\mathbf {y}) = \max _ {\alpha \in [ 0, 1 ]} \left\{g (\alpha) \equiv \alpha \| \mathbf {y} \| _ {*} + \sqrt {1 - \alpha^ {2}} \right\}.\tag{4.11}
$$

It is easy to see that the maximizer of $g$ over [0, 1] is 

$$
\tilde {\alpha} = \frac {\| \mathbf {y} \| _ {*}}{\sqrt {\| \mathbf {y} \| _ {*} ^ {2} + 1}}.
$$

Plugging $\alpha = \tilde { \alpha }$ into (4.11), we finally obtain that for any $\mathbf { y } \in \mathbb { E } ^ { * }$ ， 

$$
f ^ {*} (\mathbf {y}) = \sqrt {\| \mathbf {y} \| _ {*} ^ {2} + 1}.
$$

It is also possible to generalize the result to functions of the form 

$$
f _ {\alpha} (\mathbf {x}) = \left\{ \begin{array}{l l} - \sqrt {\alpha^ {2} - \| \mathbf {x} \| ^ {2}}, & \| \mathbf {x} \| \leq \alpha , \\ \infty & \text {else}, \end{array} \right.
$$

where $\alpha \in \mathbb { R } _ { + + }$ . In this notation, $f = f _ { 1 }$ . To compute $f _ { \alpha } ^ { * }$ , note that $f _ { \alpha } ( \mathbf { x } ) =$ $\alpha f \left( { \pmb { \underline { { x } } } } \right)$ , and hence by Theorem 4.14(b) it follows that for any $\mathbf { y } \in \mathbb { E } ^ { * }$ 

$$
f _ {\alpha} ^ {*} (\mathbf {y}) = \alpha f ^ {*} (\mathbf {y}) = \alpha \sqrt {1 + \| \mathbf {y} \| _ {*} ^ {2}}.
$$

## 4.4.14 $\sqrt { \alpha ^ { 2 } + \| \cdot \| ^ { 2 } }$

Consider the function $g _ { \alpha } : \mathbb { E }  \mathbb { R }$ given by $g _ { \alpha } ( \mathbf { x } ) = \sqrt { \alpha ^ { 2 } + \| \mathbf { x } \| ^ { 2 } }$ , where $\alpha > 0$ Then $\begin{array} { r } { g _ { \alpha } ( \mathbf { x } ) = \alpha g \left( \frac { \mathbf { x } } { \alpha } \right) } \end{array}$ , where $g ( \mathbf { x } ) = \sqrt { 1 + \| \mathbf { x } \| ^ { 2 } }$ . By Section 4.4.13, it follows that $g = f ^ { * }$  , where f is given by 

$$
f (\mathbf {y}) = \left\{ \begin{array}{l l} - \sqrt {1 - \| \mathbf {y} \| _ {*} ^ {2}}, & \| \mathbf {y} \| _ {*} \leq 1, \\ \infty & \text { else }. \end{array} \right.
$$

Since $f$ is proper closed and convex, it follows by Theorem 4.8 that 

$$
g ^ {*} = f ^ {* *} = f.
$$

Finally, invoking Theorem 4.14(b), we conclude that for any $\mathbf { y } \in \mathbb { E } ^ { * }$ ， 

$$
g _ {\alpha} ^ {*} (\mathbf {y}) = \alpha g ^ {*} (\mathbf {y}) = \alpha f (\mathbf {y}) = \left\{ \begin{array}{l l} - \alpha \sqrt {1 - \| \mathbf {y} \| _ {*} ^ {2}}, & \| \mathbf {y} \| _ {*} \leq 1, \\ \infty & \text { else }. \end{array} \right.
$$

## 4.4.15 Squared Norm

Let $f : \mathbb { E } $ R be given by $\begin{array} { r } { f ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { x } \| ^ { 2 } } \end{array}$ , where $\| \cdot \|$ is the norm associated with the space E. For any $\mathbf { y } \in \mathbb { E } ^ { * }$ , we can write $f ^ { * } ( \mathbf { y } )$ as the optimal value of the following double maximization problem: 

$$
f ^ {*} (\mathbf {y}) = \max _ {\mathbf {x} \in \mathbb {E}} \left\{\langle \mathbf {y}, \mathbf {x} \rangle - \frac {1}{2} \| \mathbf {x} \| ^ {2} \right\} = \max _ {\alpha \geq 0} \max _ {\mathbf {x}: \| \mathbf {x} \| = \alpha} \left\{\langle \mathbf {y}, \mathbf {x} \rangle - \frac {1}{2} \alpha^ {2} \right\}.
$$

Using the definition of the dual norm, it follows that 

$$
\max _ {\mathbf {x} \in \mathbb {E}: \| \mathbf {x} \| = \alpha} \left\langle \mathbf {y}, \mathbf {x} \right\rangle = \alpha \| \mathbf {y} \| _ {*},
$$

Hence, 

$$
f ^ {*} (\mathbf {y}) = \max _ {\alpha \geq 0} \left\{\alpha \| \mathbf {y} \| _ {*} - \frac {1}{2} \alpha^ {2} \right\} = \frac {1}{2} \| \mathbf {y} \| _ {*} ^ {2}.
$$

## 4.4.16 Summary of Conjugate Computations

The table below summarizes all the computations of conjugate functions described in this chapter. 

<table><tr><td>f(x)</td><td>dom(f)</td><td>f*</td><td>Assumptions</td><td>Reference</td></tr><tr><td>ex</td><td><eq>\mathbb{R}</eq></td><td>y log y - y (dom(f*) = <eq>\mathbb{R}_{+}</eq>)</td><td>-</td><td>Section 4.4.1</td></tr><tr><td>-log x</td><td><eq>\mathbb{R}_{++}</eq></td><td>-1 - log(-y) (dom(f*) = <eq>\mathbb{R}_{--}</eq>)</td><td>-</td><td>Section 4.4.2</td></tr><tr><td>max{1 - x,0}</td><td><eq>\mathbb{R}</eq></td><td>y + δ[−1,0](y)</td><td>-</td><td>Section 4.4.3</td></tr><tr><td><eq>\frac{1}{p}|x|^p</eq></td><td><eq>\mathbb{R}</eq></td><td><eq>\frac{1}{q} |y|^q</eq></td><td>p &gt; 1, <eq>\frac{1}{p} + \frac{1}{q} = 1</eq></td><td>Section 4.4.4</td></tr><tr><td><eq>-\frac{x^p}{p}</eq></td><td><eq>\mathbb{R}_{+}</eq></td><td><eq>-\frac{(-y)^q}{q}</eq> (dom(f*) = <eq>\mathbb{R}_{--}</eq>)</td><td>0 &lt; p &lt; 1, <eq>\frac{1}{p} + \frac{1}{q} = 1</eq></td><td>Section 4.4.5</td></tr><tr><td><eq>\frac{1}{2}\mathbf{x}^T\mathbf{A}\mathbf{x} + \mathbf{b}^T\mathbf{x} + c</eq></td><td><eq>\mathbb{R}^n</eq></td><td><eq>\frac{1}{2}(\mathbf{y} - \mathbf{b})^T\mathbf{A}^{-1}(\mathbf{y} - \mathbf{b}) - c</eq></td><td><eq>\mathbf{A} \in \mathbb{S}_{++}^n</eq>, b ∈ <eq>\mathbb{R}^n</eq>, c ∈ <eq>\mathbb{R}</eq></td><td>Section 4.4.6</td></tr><tr><td><eq>\frac{1}{2}\mathbf{x}^T\mathbf{A}\mathbf{x} + \mathbf{b}^T\mathbf{x} + c</eq></td><td><eq>\mathbb{R}^n</eq></td><td><eq>\frac{1}{2}(\mathbf{y} - \mathbf{b})^T\mathbf{A}^\dagger(\mathbf{y} - \mathbf{b}) - c</eq> (dom(f*) = b + Range(A))</td><td><eq>\mathbf{A} \in \mathbb{S}_{+}^n</eq>, b ∈ <eq>\mathbb{R}^n</eq>, c ∈ <eq>\mathbb{R}</eq></td><td>Section 4.4.7</td></tr><tr><td><eq>\sum_{i=1}^{n} x_i \log x_i</eq></td><td><eq>\mathbb{R}_+^n</eq></td><td><eq>\sum_{i=1}^{n} e^{y_i - 1}</eq></td><td>-</td><td>Section 4.4.8</td></tr><tr><td><eq>\sum_{i=1}^{n} x_i \log x_i</eq></td><td><eq>\Delta_n</eq></td><td><eq>\log (\sum_{i=1}^{n} e^{y_i})</eq></td><td>-</td><td>Section 4.4.10</td></tr><tr><td><eq>-\sum_{i=1}^{n} \log x_i</eq></td><td><eq>\mathbb{R}_{++}^n</eq></td><td><eq>-n - \sum_{i=1}^{n} \log(-y_i)</eq> (dom(f*) = <eq>\mathbb{R}_{--}^n</eq>)</td><td>-</td><td>Section 4.4.9</td></tr><tr><td><eq>\log (\sum_{i=1}^{n} e^{x_i})</eq></td><td><eq>\mathbb{R}^n</eq></td><td><eq>\sum_{i=1}^{n} y_i \log y_i</eq> (dom(f*) = <eq>\Delta_n</eq>)</td><td>-</td><td>Section 4.4.11</td></tr><tr><td><eq>\max_i\{x_i\}</eq></td><td><eq>\mathbb{R}^n</eq></td><td><eq>\delta_{\Delta_n}(\mathbf{y})</eq></td><td>-</td><td>Example 4.10</td></tr><tr><td><eq>\delta_C(\mathbf{x})</eq></td><td>C</td><td><eq>\sigma_C(\mathbf{y})</eq></td><td><eq>\emptyset \neq C \subseteq \mathbb{E}</eq></td><td>Example 4.2</td></tr><tr><td><eq>\sigma_C(\mathbf{x})</eq></td><td><eq>\text{dom}(\sigma_C)</eq></td><td><eq>\delta_{\text{cl}(\text{conv}(C))}(\mathbf{y})</eq></td><td><eq>\emptyset \neq C \subseteq \mathbb{E}</eq></td><td>Example 4.9</td></tr><tr><td><eq>\| \mathbf{x} \|</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>\delta_{B_{\|\cdot\|*}}[\mathbf{o},1](\mathbf{y})</eq></td><td>-</td><td>Section 4.4.12</td></tr><tr><td><eq>-\sqrt{\alpha^2 - \| \mathbf{x} \|^2}</eq></td><td>B[0,α]</td><td><eq>\alpha\sqrt{\|\mathbf{y}\|_*^2 + 1}</eq></td><td>α &gt; 0</td><td>Section 4.4.13</td></tr><tr><td><eq>\sqrt{\alpha^2 + \| \mathbf{x} \|^2}</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>-\alpha\sqrt{1 - \| \mathbf{y}\|_*^2}</eq> (dom f* = B<eq>_{\|\cdot\|*}</eq>[0,1])</td><td>α &gt; 0</td><td>Section 4.4.14</td></tr><tr><td><eq>\frac{1}{2}\|\mathbf{x}\|^2</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>\frac{1}{2}\|\mathbf{y}\|^2</eq></td><td>-</td><td>Section 4.4.15</td></tr><tr><td><eq>\frac{1}{2}\|\mathbf{x}\|^2 + \delta_C(\mathbf{x})</eq></td><td>C</td><td><eq>\frac{1}{2}\|\mathbf{y}\|^2 - \frac{1}{2}d_C^2(\mathbf{y})</eq></td><td><eq>\emptyset \neq C \subseteq \mathbb{E}</eq>, E Euclidean</td><td>Example 4.4</td></tr><tr><td><eq>\frac{1}{2}\|\mathbf{x}\|^2 - \frac{1}{2}d_C^2(\mathbf{x})</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>\frac{1}{2}\|\mathbf{y}\|^2 + \delta_C(\mathbf{y})</eq></td><td><eq>\emptyset \neq C \subseteq \mathbb{E}</eq> closed convex.E Euclidean</td><td>Example 4.11</td></tr></table>

## 4.4.17 Fenchel’s Duality Theorem

Conjugate functions naturally appear in dual problems most prominently in the celebrated Fenchel’s duality theorem, which we now recall. Consider the problem 

$$
\text {(P)} \min _ {\mathbf {x} \in \mathbb {E}} f (\mathbf {x}) + g (\mathbf {x}).
$$

We begin by rewriting the problem as 

$$
\min _ {\mathbf {x}, \mathbf {z} \in \mathbb {E}} \left\{f (\mathbf {x}) + g (\mathbf {z}): \mathbf {x} = \mathbf {z} \right\}
$$

and then constructing the Lagrangian 

$$
L (\mathbf {x}, \mathbf {z}; \mathbf {y}) = f (\mathbf {x}) + g (\mathbf {z}) + \langle \mathbf {y}, \mathbf {z} - \mathbf {x} \rangle = - [ \langle \mathbf {y}, \mathbf {x} \rangle - f (\mathbf {x}) ] - [ \langle - \mathbf {y}, \mathbf {z} \rangle - g (\mathbf {z}) ].
$$

The dual objective function is computed by minimizing the Lagrangian w.r.t. the primal variables x, z: 

$$
q (\mathbf {y}) = \min _ {\mathbf {x}, \mathbf {z}} L (\mathbf {x}, \mathbf {z}; \mathbf {y}) = - f ^ {*} (\mathbf {y}) - g ^ {*} (- \mathbf {y}).
$$

We thus obtain the following dual problem, which is also called Fenchel’s dual: 

$$
\text {(D)} \max _ {\mathbf {y} \in \mathbb {E} ^ {*}} \{- f ^ {*} (\mathbf {y}) - g ^ {*} (- \mathbf {y}) \}.
$$

Fenchel’s duality theorem, which we recall below, provides conditions under which strong duality holds for the pair of problems (P) and (D). 

Theorem 4.15 (Fenchel’s duality theorem [108, Theorem 31.1]). Let $f , g \ .$ $\mathbb { E } \to ( - \infty , \infty ]$ be proper convex functions. If ri(dom(f)) ∩ ri(dom(g)) = ∅, then 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \{f (\mathbf {x}) + g (\mathbf {x}) \} = \max _ {\mathbf {y} \in \mathbb {E} ^ {*}} \{- f ^ {*} (\mathbf {y}) - g ^ {*} (- \mathbf {y}) \},
$$

and the maximum in the right-hand problem is attained whenever it is finite. 

## 4.5 Infimal Convolution and Conjugacy

We will now show that in some sense the operations of addition and infimal convolution are dual to each other under the conjugacy operation. The first result holds under the very mild condition of properness of the functions. 

Theorem 4.16 (conjugate of infimal convolution). For two proper functions $h _ { 1 } , h _ { 2 } : \mathbb { E }  ( - \infty , \infty ]$ it holds that 

$$
(h _ {1} \square h _ {2}) ^ {*} = h _ {1} ^ {*} + h _ {2} ^ {*}.
$$

Proof. For every $\mathbf { y } \in \mathbb { E } ^ { * }$ one has 

$$
\begin{array}{l} (h _ {1} \square h _ {2}) ^ {*} (\mathbf {y}) = \max _ {\mathbf {x} \in \mathbb {E}} \{\langle \mathbf {y}, \mathbf {x} \rangle - (h _ {1} \square h _ {2}) (\mathbf {x}) \} \\ \qquad = \max _ {\mathbf {x} \in \mathbb {E}} \{\langle \mathbf {y}, \mathbf {x} \rangle - \min _ {\mathbf {u} \in \mathbb {E}} \{h _ {1} (\mathbf {u}) + h _ {2} (\mathbf {x} - \mathbf {u}) \} \} \\ \qquad = \max _ {\mathbf {x} \in \mathbb {E}} \max _ {\mathbf {u} \in \mathbb {E}} \{\langle \mathbf {y}, \mathbf {x} \rangle - h _ {1} (\mathbf {u}) - h _ {2} (\mathbf {x} - \mathbf {u}) \} \\ \qquad = \max _ {\mathbf {x} \in \mathbb {E}} \max _ {\mathbf {u} \in \mathbb {E}} \{\langle \mathbf {y}, \mathbf {x} - \mathbf {u} \rangle + \langle \mathbf {y}, \mathbf {u} \rangle - h _ {1} (\mathbf {u}) - h _ {2} (\mathbf {x} - \mathbf {u}) \} \\ \qquad = \max _ {\mathbf {u} \in \mathbb {E}} \max _ {\mathbf {x} \in \mathbb {E}} \{\langle \mathbf {y}, \mathbf {x} - \mathbf {u} \rangle + \langle \mathbf {y}, \mathbf {u} \rangle - h _ {1} (\mathbf {u}) - h _ {2} (\mathbf {x} - \mathbf {u}) \} \\ \qquad = \max _ {\mathbf {u} \in E} \{h _ {2} ^ {*} (\mathbf {y}) + \langle \mathbf {y}, \mathbf {u} \rangle - h _ {1} (\mathbf {u}) \} \\ \qquad = h _ {1} ^ {*} (\mathbf {y}) + h _ {2} ^ {*} (\mathbf {y}). \quad Ⅱ \\ \qquad = h _ {1} ^ {*} (\mathbf {y}) + h _ {2} ^ {*} (\mathbf {y}). \end{array}
$$

The second “direction” is a much deeper result requiring additional assumptions like convexity of the functions under consideration. 

Theorem 4.17 (conjugate of sum). Let $h _ { 1 } : \mathbb { E }  ( - \infty , \infty ]$ be a proper convex function and $h _ { 2 } : \mathbb { E }  \mathbb { R }$ be a real-valued convex function. Then 

$$
(h _ {1} + h _ {2}) ^ {*} = h _ {1} ^ {*} \square h _ {2} ^ {*}.
$$

Proof. For any $\mathbf { y } \in \mathbb { E } ^ { * }$ ， 

$$
\begin{array}{l} (h _ {1} + h _ {2}) ^ {*} (\mathbf {y}) = \max _ {\mathbf {x} \in \mathbb {E}} \left\{\langle \mathbf {y}, \mathbf {x} \rangle - h _ {1} (\mathbf {x}) - h _ {2} (\mathbf {x}) \right\} \\ = - \min _ {\mathbf {x} \in \mathbb {E}} \left\{h _ {1} (\mathbf {x}) + h _ {2} (\mathbf {x}) - \langle \mathbf {y}, \mathbf {x} \rangle \right\} \\ = - \min _ {\mathbf {x} \in \mathbb {E}} \left\{h _ {1} (\mathbf {x}) + g (\mathbf {x}) \right\}, \end{array}\tag{4.12}
$$

where $g ( \mathbf { x } ) \equiv h _ { 2 } ( \mathbf { x } ) - \langle \mathbf { y } , \mathbf { x } \rangle$ . Note that 

$$
\operatorname{ri} \left(\operatorname{dom} \left(h _ {1}\right)\right) \cap \operatorname{ri} \left(\operatorname{dom} (g)\right) = \operatorname{ri} \left(\operatorname{dom} \left(h _ {1}\right)\right) \cap \mathbb {E} = \operatorname{ri} \left(\operatorname{dom} \left(h _ {1}\right)\right) \neq \emptyset ,
$$

and we can thus employ Fenchel’s duality theorem (Theorem 4.15) and obtain the following equality: 

$$
\min _ {\mathbf {x} \in \mathbb {E}} \{h _ {1} (\mathbf {x}) + g (\mathbf {x}) \} = \max _ {\mathbf {z} \in \mathbb {E} ^ {*}} \{- h _ {1} ^ {*} (\mathbf {z}) - g ^ {*} (- \mathbf {z}) \} = \max _ {\mathbf {z} \in \mathbb {E} ^ {*}} \{- h _ {1} ^ {*} (\mathbf {z}) - h _ {2} ^ {*} (\mathbf {y} - \mathbf {z}) \}\tag{4.13}
$$

Combining (4.12) and (4.13), we finally obtain that for any $\mathbf { y } \in \mathbb { E } ^ { * }$ 

$$
(h _ {1} + h _ {2}) ^ {*} (\mathbf {y}) = \min _ {\mathbf {z} \in \mathbb {E} ^ {*}} \{h _ {1} ^ {*} (\mathbf {z}) + h _ {2} ^ {*} (\mathbf {y} - \mathbf {z}) \} = (h _ {1} ^ {*} \Box h _ {2} ^ {*}) (\mathbf {y}),
$$

establishing the desired result. 

Corollary 4.18. Let $h _ { 1 } : \mathbb { E }  ( - \infty , \infty ]$ be a proper closed convex function and $h _ { 2 } : \mathbb { E }  \mathbb { R }$ be a real-valued convex function. Then 

$$
h _ {1} + h _ {2} = (h _ {1} ^ {*} \Box h _ {2} ^ {*}) ^ {*}.
$$

Proof. The function $h _ { 1 } + h _ { 2 }$ is obviously proper and is closed by the closedness of $h _ { 1 } , h _ { 2 }$ (Theorem $2 . 7 ( \mathrm { b } ) )$ . Therefore, by Theorem 4.8, $( h _ { 1 } + h _ { 2 } ) ^ { * * } = h _ { 1 } + h _ { 2 }$ , which, combined with Theorem 4.17, yields 

$$
h _ {1} + h _ {2} = (h _ {1} + h _ {2}) ^ {* *} = [ (h _ {1} + h _ {2}) ^ {*} ] ^ {*} = (h _ {1} ^ {*} \Box h _ {2} ^ {*}) ^ {*}. \quad \Box
$$

The next result shows a representation of the infimal convolution in terms of the corresponding conjugate functions. 

Theorem 4.19 (representation of the infimal convolution by conjugates). Let $h _ { 1 } : \mathbb { E }  ( - \infty , \infty ]$ be a proper convex function, and let $h _ { 2 } : \mathbb { E }  \mathbb { R }$ be a real-valued convex function. Suppose that $h _ { 1 } \sqcup h _ { 2 }$ is a real-valued function. Then 

$$
h _ {1} \square h _ {2} = (h _ {1} ^ {*} + h _ {2} ^ {*}) ^ {*}.\tag{4.14}
$$

Proof. By Theorem 4.16, 

$$
(h _ {1} \square h _ {2}) ^ {*} = h _ {1} ^ {*} + h _ {2} ^ {*}.\tag{4.15}
$$

Since $h _ { 1 }$ is proper and convex and $h _ { 2 }$ is real-valued and convex, it follows by Theorem 2.19 that $h _ { 1 } \boxed  \} h _ { 2 }$ is convex. Since $h _ { 1 } \boxed  \} h _ { 2 }$ is real-valued, it is in particular proper and closed. Therefore, by Theorem 4.8, $( h _ { 1 } \sqcup h _ { 2 } ) ^ { * * } = h _ { 1 } \sqcup h _ { 2 }$ . Hence, taking the conjugate of both sides of (4.15), the identity (4.14) follows. 

## 4.6 Subdiferentials of Conjugate Functions

The main result concerning the subdiferential of a conjugate function is the socalled conjugate subgradient theorem. 

Theorem 4.20 (conjugate subgradient theorem). Let $f : \mathbb { E }  ( - \infty , \infty ]$ be proper and convex. The following two claims are equivalent for any $\mathbf { x } \in \mathbb { E } , \mathbf { y } \in \mathbb { E } ^ { * }$ : 

(i) $\langle \mathbf { x } , \mathbf { y } \rangle = f ( \mathbf { x } ) + f ^ { * } ( \mathbf { y } ) .$ 

(ii) $\mathbf { y } \in \partial f ( \mathbf { x } )$ 

If in addition f is closed, then (i) and (ii) are equivalent to 

$$
\left(\text { iii }\right) \mathbf {x} \in \partial f ^ {*} (\mathbf {y}).
$$

Proof. The relation $\mathbf { y } \in \partial f ( \mathbf { x } )$ holds if and only if 

$$
f (\mathbf {z}) \geq f (\mathbf {x}) + \langle \mathbf {y}, \mathbf {z} - \mathbf {x} \rangle \text {   for   all   } \mathbf {z} \in \mathbb {E},
$$

which is the same as 

$$
\langle \mathbf {y}, \mathbf {x} \rangle - f (\mathbf {x}) \geq \langle \mathbf {y}, \mathbf {z} \rangle - f (\mathbf {z}) \text { for   all } \mathbf {z} \in \mathbb {E}.\tag{4.16}
$$

Taking the maximum over z, we obtain that (4.16) is the same as 

$$
\langle \mathbf {y}, \mathbf {x} \rangle - f (\mathbf {x}) \geq f ^ {*} (\mathbf {y}),
$$

which by Fenchel’s inequality (Theorem 4.6) is equivalent to the equality $\langle \mathbf { x } , \mathbf { y } \rangle =$ $f ( \mathbf { x } ) + f ^ { * } ( \mathbf { y } )$ . We have thus established the equivalence between (i) and (ii). Assume now that in addition f is closed. Then by Theorem 4.8, $f ^ { * * } = f$ , which in particular implies that (i) is equivalent to 

$$
\langle \mathbf {x}, \mathbf {y} \rangle = g (\mathbf {y}) + g ^ {*} (\mathbf {x}),
$$

where $g = f ^ { * }$ . By the same equivalence that was already established between (i) and (ii) (but here employed on $g )$ , we conclude that (i) is equivalent to $\mathbf { x } \in \partial g ( \mathbf { y } ) =$ $\partial f ^ { * } ( \mathbf { y } )$ □ 

By the definition of the conjugate function, claim (i) in Theorem 4.20 can be rewritten as 

$$
\mathbf {x} \in \operatorname{argmax} _ {\tilde {\mathbf {x}} \in \mathbb {E}} \left\{\langle \mathbf {y}, \tilde {\mathbf {x}} \rangle - f (\tilde {\mathbf {x}}) \right\},
$$

and, when $f$ is closed, also as 

$$
\mathbf {y} \in \operatorname{argmax} _ {\tilde {\mathbf {y}} \in \mathbb {E} ^ {*}} \left\{\langle \mathbf {x}, \tilde {\mathbf {y}} \rangle - f ^ {*} (\tilde {\mathbf {y}}) \right\}.
$$

Equipped with the above observation, we can conclude that the conjugate subgradient theorem, in the case where $f$ is closed, can also be equivalently formulated as follows. 

Corollary 4.21 (conjugate subgradient theorem—second formulation). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper closed convex function. Then for any ${ \textbf { x } } \in$ $\mathbb { E } , \mathbf { y } \in \mathbb { E } ^ { * }$ 

$$
\partial f (\mathbf {x}) = \operatorname{argmax} _ {\tilde {\mathbf {y}} \in \mathbb {E} ^ {*}} \left\{\langle \mathbf {x}, \tilde {\mathbf {y}} \rangle - f ^ {*} (\tilde {\mathbf {y}}) \right\}
$$

and 

$$
\partial f ^ {*} (\mathbf {y}) = \operatorname{argmax} _ {\tilde {\mathbf {x}} \in \mathbb {E}} \left\{\langle \mathbf {y}, \tilde {\mathbf {x}} \rangle - f (\tilde {\mathbf {x}}) \right\}.
$$

In particular, we can also conclude that for any proper closed convex function $f ,$ 

$$
\partial f (0) = \operatorname * {a r g m i n} _ {\mathbf {y} \in \mathbb {E} ^ {*}} f ^ {*} (\mathbf {y})
$$

and 

$$
\partial f ^ {*} (0) = \operatorname{argmin} _ {\mathbf {x} \in \mathbb {E}} f (\mathbf {x}).
$$

Example 4.22. Let $f : \mathbb { E } \to \mathbb { R }$ be given by $f ( \mathbf { x } ) = \| \mathbf { x } \|$ . Obviously, $f$ is proper, closed, and convex. By Example 2.31, $f = \sigma _ { B _ { \parallel \cdot \parallel _ { * } } [ \mathbf { 0 } , 1 ] }$ . Therefore, by Example 4.9, $f ^ { * } = \delta _ { B _ { \parallel } . \parallel _ { * } [ \mathbf { 0 } , 1 ] }$ . We can now use the conjugate subgradient theorem (Corollary 4.21) and compute the subdiferential set of $f$ at 0 as follows: 

$$
\partial f (\mathbf {0}) = \operatorname{argmin} _ {\mathbf {y} \in \mathbb {E} ^ {*}} f ^ {*} (\mathbf {y}) = \operatorname{argmin} _ {\mathbf {y} \in \mathbb {E} ^ {*}} \delta_ {B _ {\| \cdot \| _ {*}} [ \mathbf {0}, 1 ]} = B _ {\| \cdot \| _ {*}} [ \mathbf {0}, 1 ].
$$

This result was already established in Example 3.3. 

A relation between Lipschitz continuity of a function and the boundedness of its subgradients over a given set was established in Theorem 3.61. We end this chapter with a related result showing that Lipschitz continuity over the entire space is also equivalent to boundedness of the domain of the conjugate. 

Theorem 4.23 (Lipschitz continuity and boundedness of the domain of the conjugate). Let $f : \mathbb { E }  \mathbb { I }$ R be convex. Then the following three claims are equivalent for a given constant $L > 0$ 

(i) $| f ( \mathbf { x } ) - f ( \mathbf { y } ) | \leq L \| \mathbf { x } - \mathbf { y } \| ~ f o r ~ a n y ~ \mathbf { x } , \mathbf { y } \in \mathbb { E } .$ 

(ii) $\| \mathbf { g } \| _ { * } \leq L$ for any $\mathbf { g } \in \partial f ( \mathbf { x } ) , \mathbf { x } \in \mathbb { E }$ 

(iii) dom $\left( f ^ { * } \right) \subseteq B _ { \| \cdot \| _ { * } } [ \mathbf { 0 } , L ]$ 

Proof. The equivalence between (i) and (ii) follows from Theorem 3.61. We will show that (iii) implies (ii). Indeed, assume that (iii) holds, that is, dom $( f ^ { \ast } ) \subseteq$ $B _ { \parallel \cdot \parallel _ { * } } [ \mathbf { 0 } , L ]$ . Since by the conjugate subgradient theorem (Corollary 4.21) for any $\mathbf { x } \in \mathbb { E } ,$ 

$$
\partial f (\mathbf {x}) = \operatorname{argmax} _ {\mathbf {y} \in \mathbb {E} ^ {*}} \left\{\langle \mathbf {x}, \mathbf {y} \rangle - f ^ {*} (\mathbf {y}) \right\},
$$

it follows that $\partial f ( \mathbf { x } ) \subseteq$ dom $( f ^ { \ast } )$ , and hence in particular $\partial f ( \mathbf { x } ) \subseteq B _ { \| \cdot \| _ { * } } [ \mathbf { 0 } , L ]$ for any $\mathbf { x } \in \mathbb { E }$ , establishing (ii). In the reverse direction, we will show that the implication ${ \mathrm { ( i ) } \Rightarrow \mathrm { ( i i i ) } }$ holds. Suppose that (i) holds. Then in particular 

$$
f (\mathbf {x}) - f (\mathbf {0}) \leq | f (\mathbf {x}) - f (\mathbf {0}) | \leq L \| \mathbf {x} \|,
$$

and hence 

$$
- f (\mathbf {x}) \geq - f (\mathbf {0}) - L \| \mathbf {x} \|.
$$

Therefore, for any $\mathbf { y } \in \mathbb { E } ^ { * }$ 

$$
f ^ {*} (\mathbf {y}) = \max _ {\mathbf {x} \in \mathbb {E}} \left\{\langle \mathbf {x}, \mathbf {y} \rangle - f (\mathbf {x}) \right\} \geq \max _ {\mathbf {x} \in \mathbb {E}} \left\{\langle \mathbf {x}, \mathbf {y} \rangle - f (\mathbf {0}) - L \| \mathbf {x} \| \right\}.\tag{4.17}
$$

To show (iii), we take $\tilde { \mathbf { y } } \in \mathbb { E } ^ { * }$ that satisfies $\| \tilde { \mathbf { y } } \| _ { * } > L$ and show that ${ \tilde { \textbf { y } } } \notin$ dom $( f ^ { * } )$ Take a vector $\mathbf { y } ^ { \dagger } \in \mathbb { E }$ satisfying $\| \mathbf { y } ^ { \dagger } \| = 1$ for which $\langle \tilde { \mathbf { y } } , \mathbf { y } ^ { \dagger } \rangle = \| \tilde { \mathbf { y } } \|$ ∗ (such a vector exists by the definition of the dual norm). Define $C = \{ \alpha \mathbf { y } ^ { \dag } : \alpha \geq 0 \} \subseteq \mathbb { E }$ . We can now continue (4.17) (with $\mathbf { y } = \tilde { \mathbf { y } } )$ and write 

$$
\begin{array}{r c l} f ^ {*} (\tilde {\mathbf {y}}) & \geq & \max _ {\mathbf {x} \in \mathbb {E}} \left\{\langle \mathbf {x}, \tilde {\mathbf {y}} \rangle - f (\mathbf {0}) - L \| \mathbf {x} \| \right\} \\ & \geq & \max _ {\mathbf {x} \in C} \left\{\langle \mathbf {x}, \tilde {\mathbf {y}} \rangle - f (\mathbf {0}) - L \| \mathbf {x} \| \right\} \\ & = & \max _ {\alpha \geq 0} \left\{\langle \alpha \tilde {\mathbf {y}}, \mathbf {y} ^ {\dagger} \rangle - f (\mathbf {0}) - L \alpha \| \mathbf {y} ^ {\dagger} \| \right\} \\ & = & \max _ {\alpha \geq 0} \left\{\alpha \| \tilde {\mathbf {y}} \| _ {*} - f (\mathbf {0}) - L \alpha \right\} \\ & = & \max _ {\alpha \geq 0} \left\{\alpha (\| \tilde {\mathbf {y}} \| _ {*} - L) - f (\mathbf {0}) \right\} \\ & \| \tilde {\mathbf {y}} \| _ {*} > L & \\ & = & \infty , \end{array}
$$

thus showing that ${ \tilde { \textbf { y } } } \notin$ dom $( f ^ { * } )$ , establishing claim (iii). 

Chapter 5 

# Smoothness and Strong Convexity

## 5.1 L-Smooth Functions

We begin with the definition of L-smoothness. 

Definition 5.1 (L-smoothness). Let $L \geq 0$ . A function $f : \mathbb { E }  ( - \infty , \infty ]$ is said to be L-smooth over a set $D \subseteq \mathbb { E }$ if it is diferentiable over D and satisfies 

$$
\left\| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \right\| _ {*} \leq L \| \mathbf {x} - \mathbf {y} \| \text {   for   all   } \mathbf {x}, \mathbf {y} \in D.
$$

The constant L is called the smoothness parameter. 

Obviously, by the definition of diferentiability, if f is L-smooth over a set $D \subseteq \mathbb { E }$ , this means in particular that $D \subseteq \operatorname { i n t } ( \operatorname { d o m } f )$ . If a function is L-smooth over E, then we will just refer to it as L-smooth (without mentioning the entire space). Another frequent terminology in the literature refers to an L-smooth function over D as $^ { 6 6 } \mathrm { a }$ function with Lipschitz gradient with constant $L . ^ { \dprime }$ The class of L-smooth functions is denoted by $\bar { C } _ { L } ^ { 1 , 1 } ( D )$ . When $D = \mathbb { E }$ , the class is often denoted by ${ C } _ { L } ^ { 1 , 1 }$ instead of $C _ { L } ^ { 1 , 1 } ( \mathbb { E } )$ . The class of functions which are L-smooth for some $L \geq 0$ is denoted by $C ^ { 1 , 1 }$ 

By the definition of L-smoothness, it is clear that if a function is $L _ { \mathrm { 1 } \mathrm { - } \mathrm { s m o o t h } }$ 8 then it is also $L _ { 2 }$ -smooth for any $L _ { 2 } \ge L _ { 1 }$ . It is therefore sometimes interesting to discuss the value of the smallest possible smoothness parameter of a given function. 

Example 5.2 (smoothness of quadratic functions). Consider the function $f : \mathbb { R } ^ { n }  \mathbb { R }$ given by $\begin{array} { r } { f ( \mathbf { x } ) = \frac { 1 } { 2 } \mathbf { x } ^ { T } \mathbf { A } \mathbf { x } + \mathbf { b } ^ { T } \mathbf { x } + c , } \end{array}$ , where $\mathbf { A } \in \mathbb { S } ^ { n } , \mathbf { b } \in \mathbb { R } ^ { n }$ , and $c \in \mathbb { R }$ We assume that $\mathbb { R } ^ { n }$ is endowed with the l<sub>p</sub>-norm $( 1 \leq p \leq \infty )$ . Then, for any x, $\mathbf { y } \in \mathbb { R } ^ { n }$ 

$$
\| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| _ {q} = \| \mathbf {A x} - \mathbf {A y} \| _ {q} \leq \| \mathbf {A} \| _ {p, q} \| \mathbf {x} - \mathbf {y} \| _ {p},
$$

where $\| \cdot \| _ { p , q }$ is the induced norm given by (see also Section 1.8.2) 

$$
\left\| \mathbf {A} \right\| _ {p, q} = \max \{\left\| \mathbf {A x} \right\| _ {q}: \left\| \mathbf {x} \right\| _ {p} \leq 1 \},
$$

with $q \in [ 1 , \infty ]$ satisfying $\textstyle { \frac { 1 } { p } } + { \frac { 1 } { q } } = 1$ . We can thus conclude that f is $\| \mathbf { A } \| _ { p , q }$ -smooth. We will show that $\| \mathbf { A } \| _ { p , q }$ is the smallest smoothness parameter. For that, assume that $f$ is L-smooth. Take a vector $\tilde { \mathbf { x } }$ satisfying $\| \tilde { \mathbf { x } } \| _ { p } = 1$ and $\| \mathbf { A } \tilde { \mathbf { x } } \| _ { q } = \| \mathbf { A } \| _ { p , q }$ . The existence of such a vector is guaranteed by the definition the induced matrix norm. Then 

$$
\| \mathbf {A} \| _ {p, q} = \| \mathbf {A} \tilde {\mathbf {x}} \| _ {q} = \| \nabla f (\tilde {\mathbf {x}}) - \nabla f (\mathbf {0}) \| _ {q} \leq L \| \tilde {\mathbf {x}} - \mathbf {0} \| _ {p} = L.
$$

We thus showed that if $f$ is L-smooth, then $L \geq \| \mathbf { A } \| _ { p , q } ,$ proving that $\| \mathbf { A } \| _ { p , q }$ is indeed the smallest possible smoothness parameter. 

Example 5.3 (0-smoothness of afine functions). Let $f : \mathbb { E } \to \mathbb { R }$ be given by $f ( \mathbf { x } ) = \left. \mathbf { b } , \mathbf { x } \right. + c$ , where b $\in \mathbb { E } ^ { * }$ and $c \in \mathbb { R }$ . For any $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ 

$$
\left\| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \right\| _ {*} = \left\| \mathbf {b} - \mathbf {b} \right\| _ {*} = 0 \leq 0 \| \mathbf {x} - \mathbf {y} \|,
$$

showing that afine functions are 0-smooth. 

The next example will utilize a well-known result on the orthogonal projection operator, which was introduced in Example 3.31. A more general result will be shown later on in Theorem 6.42. 

Theorem 5.4 (see [10, Theorem 9.9]). Let E be a Euclidean space, and let $C \subseteq \mathbb { E }$ be a nonempty closed and convex set. Then 

(a) (firm nonexpansiveness) For any v, $\mathbf { w } \in \mathbb { E }$ 2 

$$
\langle P _ {C} (\mathbf {v}) - P _ {C} (\mathbf {w}), \mathbf {v} - \mathbf {w} \rangle \geq \| P _ {C} (\mathbf {v}) - P _ {C} (\mathbf {w}) \| ^ {2}.\tag{5.1}
$$

(b) (nonexpansiveness) For any $\mathbf { v } , \mathbf { w } \in \mathbb { E }$ 2 

$$
\left\| P _ {C} (\mathbf {v}) - P _ {C} (\mathbf {w}) \right\| \leq \left\| \mathbf {v} - \mathbf {w} \right\|.\tag{5.2}
$$

Example 5.5 (1-smoothness of $\scriptstyle { \frac { 1 } { 2 } } d _ { C } ^ { 2 } )$ . Suppose that E is a Euclidean space, and let $C \subseteq \mathbb { E }$ be a nonempty closed and convex set. Consider the function $\varphi _ { C } ( \mathbf { x } ) =$ $\textstyle { \frac { 1 } { 2 } } d _ { C } ^ { 2 } ( \mathbf { x } )$ . By Example 3.31, $\varphi _ { C }$ is diferentiable over E and $\nabla \varphi _ { C } ( \mathbf { x } ) = \mathbf { x } - P _ { C } ( \mathbf { x } )$ We will show that $\varphi _ { C }$ is 1-smooth. Indeed, for any $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ 

$$
\begin{array}{r l} \| \nabla \varphi_ {C} (\mathbf {x}) - \nabla \varphi_ {C} (\mathbf {y}) \| ^ {2} & = \| \mathbf {x} - \mathbf {y} - P _ {C} (\mathbf {x}) + P _ {C} (\mathbf {y}) \| ^ {2} \\ & = \| \mathbf {x} - \mathbf {y} \| ^ {2} - 2 \langle P _ {C} (\mathbf {x}) - P _ {C} (\mathbf {y}), \mathbf {x} - \mathbf {y} \rangle + \| P _ {C} (\mathbf {x}) - P _ {C} (\mathbf {y}) \| ^ {2} \\ & \overset {(*)} {\leq} \| \mathbf {x} - \mathbf {y} \| ^ {2} - 2 \| P _ {C} (\mathbf {x}) - P _ {C} (\mathbf {y}) \| ^ {2} + \| P _ {C} (\mathbf {x}) - P _ {C} (\mathbf {y}) \| ^ {2} \\ & = \| \mathbf {x} - \mathbf {y} \| ^ {2} - \| P _ {C} (\mathbf {x}) - P _ {C} (\mathbf {y}) \| ^ {2} \\ & \leq \| \mathbf {x} - \mathbf {y} \| ^ {2}, \end{array}
$$

where the inequality (∗) follows by the firm nonexpansivity of the orthogonal projection operator (Theorem $5 . 4 ( \mathrm { a } ) )$ ). 

Example 5.6 (1-smoothness of $\begin{array} { r } { \frac { 1 } { 2 } | | \cdot | | ^ { 2 } - \frac { 1 } { 2 } d _ { C } ^ { 2 } ) } \end{array}$ . Suppose that E is a Euclidean space, and let $C \subseteq \mathbb { E }$ be a nonempty closed convex set. Consider the function $\begin{array} { r } { \psi _ { C } ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { x } \| ^ { 2 } - \frac { 1 } { 2 } d _ { C } ^ { 2 } ( \mathbf { x } ) } \end{array}$ . By Example 2.17, ψ<sub>C</sub> is convex.<sup>23</sup> We will now show that it is 1-smooth. By Example 3.31, $\textstyle { \frac { 1 } { 2 } } d _ { C } ^ { 2 } ( \mathbf { x } )$ is diferentiable over $\mathbb { E } ,$ and its gradient is given by $\mathbf { x } - P _ { C } ( \mathbf { x } )$ . Therefore, 

$$
\nabla \psi_ {C} (\mathbf {x}) = \mathbf {x} - (\mathbf {x} - P _ {C} (\mathbf {x})) = P _ {C} (\mathbf {x}).
$$

The 1-smoothness of $\psi _ { C }$ now follows by the nonexpansivity of the projection operator (Theorem 5.4(b))—for any $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ 

$$
\| \nabla \psi_ {C} (\mathbf {x}) - \nabla \psi_ {C} (\mathbf {y}) \| = \| P _ {C} (\mathbf {x}) - P _ {C} (\mathbf {y}) \| \leq \| \mathbf {x} - \mathbf {y} \|.
$$

## 5.1.1 The Descent Lemma

An extremely useful result on L-smooth functions is the descent lemma, which states that they can be upper bounded by a certain quadratic function. 

Lemma 5.7 (descent lemma). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be an L-smooth function $( L \geq 0 )$ over a given convex set $D$ . Then for any $\mathbf { x } , \mathbf { y } \in D$ 2 

$$
f (\mathbf {y}) \leq f (\mathbf {x}) + \langle \nabla f (\mathbf {x}), \mathbf {y} - \mathbf {x} \rangle + \frac {L}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.\tag{5.3}
$$

Proof. By the fundamental theorem of calculus, 

$$
f (\mathbf {y}) - f (\mathbf {x}) = \int_ {0} ^ {1} \langle \nabla f (\mathbf {x} + t (\mathbf {y} - \mathbf {x})), \mathbf {y} - \mathbf {x} \rangle d t.
$$

Therefore, 

$$
f (\mathbf {y}) - f (\mathbf {x}) = \langle \nabla f (\mathbf {x}), \mathbf {y} - \mathbf {x} \rangle + \int_ {0} ^ {1} \langle \nabla f (\mathbf {x} + t (\mathbf {y} - \mathbf {x})) - \nabla f (\mathbf {x}), \mathbf {y} - \mathbf {x} \rangle d t.
$$

Thus, 

$$
\begin{array}{l} | f (\mathbf {y}) - f (\mathbf {x}) - \langle \nabla f (\mathbf {x}), \mathbf {y} - \mathbf {x} \rangle | = \left| \int_ {0} ^ {1} \langle \nabla f (\mathbf {x} + t (\mathbf {y} - \mathbf {x})) - \nabla f (\mathbf {x}), \mathbf {y} - \mathbf {x} \rangle d t \right| \\ \leq \int_ {0} ^ {1} | \langle \nabla f (\mathbf {x} + t (\mathbf {y} - \mathbf {x})) - \nabla f (\mathbf {x}), \mathbf {y} - \mathbf {x} \rangle | d t \\ \stackrel {{(*)}} {{\leq}} \int_ {0} ^ {1} \| \nabla f (\mathbf {x} + t (\mathbf {y} - \mathbf {x})) - \nabla f (\mathbf {x}) \| _ {*} \cdot \| \mathbf {y} - \mathbf {x} \| d t \\ \leq \int_ {0} ^ {1} t L \| \mathbf {y} - \mathbf {x} \| ^ {2} d t \\ = \frac {L}{2} \| \mathbf {y} - \mathbf {x} \| ^ {2}, \end{array}
$$

where in (∗) we used the generalized Cauchy–Schwarz inequality (Lemma 1.4). 

## 5.1.2 Characterizations of L-Smooth Functions

When f is convex, the next result gives several diferent and equivalent characterizations of the L-smoothness property of $f$ over the entire space. Note that property (5.3) from the descent lemma is one of the mentioned equivalent properties. 

Theorem 5.8 (characterizations of L-smoothness). Let $f : \mathbb { E }  \mathbb { R }$ be a convex function, diferentiable over $\mathbb { E }$ , and let $L > 0$ . Then the following claims are equivalent: 

(i) f is L-smooth. 

(ii) $\begin{array} { r } { f ( \mathbf { y } ) \leq f ( \mathbf { x } ) + \langle \nabla f ( \mathbf { x } ) , \mathbf { y } - \mathbf { x } \rangle + \frac { L } { 2 } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } } \end{array}$ for all x, $\mathbf { y } \in \mathbb { E }$ 

(iii) $\begin{array} { r } { f ( \mathbf { y } ) \geq f ( \mathbf { x } ) + \langle \nabla f ( \mathbf { x } ) , \mathbf { y } - \mathbf { x } \rangle + \frac { 1 } { 2 L } \| \nabla f ( \mathbf { x } ) - \nabla f ( \mathbf { y } ) \| _ { * } ^ { 2 } f o r ~ a l l ~ \mathbf { x } , \mathbf { y } \in \mathbb { E } . } \end{array}$ 

(iv) $\begin{array} { r } { \langle \nabla f ( \mathbf { x } ) - \nabla f ( \mathbf { y } ) , \mathbf { x } - \mathbf { y } \rangle \geq \frac { 1 } { L } \| \nabla f ( \mathbf { x } ) - \nabla f ( \mathbf { y } ) \| _ { * } ^ { 2 } ~ f o r ~ a l l ~ \mathbf { x } , \mathbf { y } \in \mathbb { E } } \end{array}$ 

(v) $f ( \lambda \mathbf { x } + ( 1 - \lambda ) \mathbf { y } ) \geq \lambda f ( \mathbf { x } ) + ( 1 - \lambda ) f ( \mathbf { y } ) - { \frac { L } { 2 } } \lambda ( 1 - \lambda ) \| \mathbf { x } - \mathbf { y } \| ^ { 2 }$ for any $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ and $\lambda \in [ 0 , 1 ]$ 

Proof. $( \mathrm { i } ) \Rightarrow ( \mathrm { i i } )$ . The fact that (i) implies (ii) is just the descent lemma (Lemma 5.7). 

$( \mathrm { i i } ) \Rightarrow ( \mathrm { i i i } )$ . Suppose that (ii) is satisfied. We can assume that $\nabla f ( \mathbf { x } ) \neq \nabla f ( \mathbf { y } )$ since otherwise the inequality (iii) is trivial by the convexity of $f .$ . For a fixed $\mathbf { x } \in \mathbb { E }$ consider the function 

$$
g _ {\mathbf {x}} (\mathbf {y}) = f (\mathbf {y}) - f (\mathbf {x}) - \langle \nabla f (\mathbf {x}), \mathbf {y} - \mathbf {x} \rangle , \quad \mathbf {y} \in \mathbb {E}.
$$

The function $g _ { \mathbf { x } }$ also satisfies property (ii). Indeed, for any $\mathbf { y } , \mathbf { z } \in \mathbb { E } ,$ 

$$
\begin{array}{l} g _ {\mathbf {x}} (\mathbf {z}) = f (\mathbf {z}) - f (\mathbf {x}) - \langle \nabla f (\mathbf {x}), \mathbf {z} - \mathbf {x} \rangle \\ \leq f (\mathbf {y}) + \langle \nabla f (\mathbf {y}), \mathbf {z} - \mathbf {y} \rangle + \frac {L}{2} \| \mathbf {z} - \mathbf {y} \| ^ {2} - f (\mathbf {x}) - \langle \nabla f (\mathbf {x}), \mathbf {z} - \mathbf {x} \rangle \\ = f (\mathbf {y}) - f (\mathbf {x}) - \langle \nabla f (\mathbf {x}), \mathbf {y} - \mathbf {x} \rangle + \langle \nabla f (\mathbf {y}) - \nabla f (\mathbf {x}), \mathbf {z} - \mathbf {y} \rangle + \frac {L}{2} \| \mathbf {z} - \mathbf {y} \| ^ {2} \\ = g _ {\mathbf {x}} (\mathbf {y}) + \langle \nabla g _ {\mathbf {x}} (\mathbf {y}), \mathbf {z} - \mathbf {y} \rangle + \frac {L}{2} \| \mathbf {z} - \mathbf {y} \| ^ {2}, \end{array} \tag {5.}\tag{5.4}
$$

where we used in the last equality the fact that $\nabla g _ { \mathbf { x } } ( \mathbf { y } ) = \nabla f ( \mathbf { y } ) - \nabla f ( \mathbf { x } )$ for any $\mathbf { y } \in \mathbb { E }$ . In particular, $\nabla g _ { \mathbf { x } } ( \mathbf { x } ) = \mathbf { 0 }$ , which by the convexity of $g _ { \mathbf { x } }$ implies that x is a global minimizer of $g$ , meaning that 

$$
g _ {\mathbf {x}} (\mathbf {x}) \leq g _ {\mathbf {x}} (\mathbf {z}) \text {   for   all   } \mathbf {z} \in \mathbb {E}.\tag{5.5}
$$

Let $\textbf { y } \in \ \mathbb { E }$ , and let ${ \textbf { v } } \in \ \mathbb { E }$ be a vector satisfying $\| \mathbf { v } \| = 1$ and $\left. \nabla g _ { \mathbf { x } } ( \mathbf { y } ) , \mathbf { v } \right. =$ $\| \nabla g _ { \mathbf { x } } ( \mathbf { y } ) \| _ { * }$ . Substituting 

$$
\mathbf {z} = \mathbf {y} - \frac {\| \nabla g _ {\mathbf {x}} (\mathbf {y}) \| _ {*}}{L} \mathbf {v}\tag{5.6}
$$

into (5.5) yields 

$$
0 = g _ {\mathbf {x}} (\mathbf {x}) \leq g _ {\mathbf {x}} \left(\mathbf {y} - \frac {\| \nabla g _ {\mathbf {x}} (\mathbf {y}) \| _ {*} \mathbf {v}}{L} \mathbf {v}\right).
$$

Combining the last inequality with (5.4) (using the specific choice of z given in (5.6)), we obtain 

$$
\begin{array}{r l} & 0 = g _ {\mathbf {x}} (\mathbf {x}) \\ & \leq g _ {\mathbf {x}} (\mathbf {y}) - \frac {\| \nabla g _ {\mathbf {x}} (\mathbf {y}) \| _ {*}}{L} \langle \nabla g _ {\mathbf {x}} (\mathbf {y}), \mathbf {v} \rangle + \frac {1}{2 L} \| \nabla g _ {\mathbf {x}} (\mathbf {y}) \| _ {*} ^ {2} \cdot \| \mathbf {v} \| ^ {2} \\ & = g _ {\mathbf {x}} (\mathbf {y}) - \frac {1}{2 L} \| \nabla g _ {\mathbf {x}} (\mathbf {y}) \| _ {*} ^ {2} \\ & = f (\mathbf {y}) - f (\mathbf {x}) - \langle \nabla f (\mathbf {x}), \mathbf {y} - \mathbf {x} \rangle - \frac {1}{2 L} \| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| _ {*} ^ {2}, \end{array}
$$

which is claim (iii). 

$( \mathrm { i i i } ) \Rightarrow ( \mathrm { i v } )$ . Writing the inequality (iii) for the two pairs $( \mathbf { x } , \mathbf { y } ) , ( \mathbf { y } , \mathbf { x } )$ yields 

$$
f (\mathbf {y}) \geq f (\mathbf {x}) + \langle \nabla f (\mathbf {x}), \mathbf {y} - \mathbf {x} \rangle + \frac {1}{2 L} \| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| _ {*} ^ {2},
$$

$$
f (\mathbf {x}) \geq f (\mathbf {y}) + \langle \nabla f (\mathbf {y}), \mathbf {x} - \mathbf {y} \rangle + \frac {1}{2 L} \| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| _ {*} ^ {2}.
$$

Adding the two inequalities and rearranging terms results in (iv). 

$\mathrm { ( i v ) \Rightarrow ( i ) }$ . The Lipschitz condition 

$$
\| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| _ {*} \leq L \| \mathbf {x} - \mathbf {y} \|
$$

is trivial when $\nabla f ( \mathbf { x } ) = \nabla f ( \mathbf { y } )$ . We will therefore assume that $\nabla f ( \mathbf { x } ) \neq \nabla f ( \mathbf { y } )$ By (iv) and the generalized Cauchy–Schwarz inequality (Lemma 1.4) we have for any $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ 2 

$$
\| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| _ {*} \cdot \| \mathbf {x} - \mathbf {y} \| \geq \langle \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}), \mathbf {x} - \mathbf {y} \rangle \geq \frac {1}{L} \| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| _ {*} ^ {2}.
$$

Dividing by $\| \nabla f ( \mathbf { x } ) - \nabla f ( \mathbf { y } ) \|$ and multiplying by $L , ( \mathrm { i } )$ is obtained. 

We have just shown the equivalence between (i), (ii), (iii), and (iv). To prove that (v) is also equivalent to each of these four claims, we will establish the equivalence $( \operatorname { i i } ) \Leftrightarrow ( \operatorname { v } )$ 

$( \mathrm { i i } ) \Rightarrow ( \mathrm { v } )$ . Let $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ and $\lambda \in [ 0 , 1 ]$ . Denote $\mathbf { x } _ { \lambda } = \lambda \mathbf { x } + ( 1 - \lambda ) \mathbf { y }$ . Then by (ii), 

$$
f (\mathbf {x}) \leq f (\mathbf {x} _ {\lambda}) + \langle \nabla f (\mathbf {x} _ {\lambda}), \mathbf {x} - \mathbf {x} _ {\lambda} \rangle + \frac {L}{2} \| \mathbf {x} - \mathbf {x} _ {\lambda} \| ^ {2},
$$

$$
f (\mathbf {y}) \leq f (\mathbf {x} _ {\lambda}) + \langle \nabla f (\mathbf {x} _ {\lambda}), \mathbf {y} - \mathbf {x} _ {\lambda} \rangle + \frac {L}{2} \| \mathbf {y} - \mathbf {x} _ {\lambda} \| ^ {2},
$$

which is the same as 

$$
f (\mathbf {x}) \leq f (\mathbf {x} _ {\lambda}) + (1 - \lambda) \langle \nabla f (\mathbf {x} _ {\lambda}), \mathbf {x} - \mathbf {y} \rangle + \frac {L (1 - \lambda) ^ {2}}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}
$$

$$
f (\mathbf {y}) \leq f (\mathbf {x} _ {\lambda}) + \lambda \langle \nabla f (\mathbf {x} _ {\lambda}), \mathbf {y} - \mathbf {x} \rangle + \frac {L \lambda^ {2}}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.
$$

Multiplying the first inequality by λ and the second by $1 - \lambda$ and adding them yields the inequality (iv). 

${ \bf { \Psi } } ( { \bf { v } } ) \Rightarrow ( \mathrm { { i i } } )$ . Rearranging terms in the inequality (v), we obtain that it is equivalent to 

$$
f (\mathbf {y}) \leq f (\mathbf {x}) + \frac {f (\mathbf {x} + (1 - \lambda) (\mathbf {y} - \mathbf {x})) - f (\mathbf {x})}{1 - \lambda} + \frac {L}{2} \lambda \| \mathbf {x} - \mathbf {y} \| ^ {2}.
$$

Taking $\lambda \to 1 ^ { - }$ , the last inequality becomes 

$$
f (\mathbf {y}) \leq f (\mathbf {x}) + f ^ {\prime} (\mathbf {x}; \mathbf {y} - \mathbf {x}) + \frac {L}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2},
$$

which, by the fact that $f ^ { \prime } ( \mathbf { x } ; \mathbf { y } - \mathbf { x } ) = \langle \nabla f ( \mathbf { x } ) , \mathbf { y } - \mathbf { x } \rangle$ (see Theorem 3.29), implies (ii). 

Remark 5.9 (necessity of convexity in Theorem 5.8). The convexity assumption in Theorem 5.8 is essential. Consider, for example, the function $f : \mathbb { R } ^ { n } $ R given by $\begin{array} { r } { f ( \mathbf { x } ) = - \frac { 1 } { 2 } \| \mathbf { x } \| _ { 2 } ^ { 2 } } \end{array}$ , which is 1-smooth w.r.t. the l -norm but is not L-smooth for $L < 1$ (see Example 5.2). However, $f$ is concave, and hence 

$$
f (\mathbf {y}) \leq f (\mathbf {x}) + \langle \nabla f (\mathbf {x}), \mathbf {y} - \mathbf {x} \rangle ,
$$

which implies that property (ii) of Theorem 5.8 is satisfied with $L = 0$ , although the function is obviously not 0-smooth. 

The next example will require the linear approximation theorem, which we now recall. 

Theorem 5.10 (linear approximation theorem, [10, Theorem 1.24], [101, Fact 3.3.10]). Let $f : U \to \mathbb { R }$ be a twice continuously diferentiable function<sup>24</sup> over an open set $U \subseteq \mathbb { R } ^ { n }$ , and let $\mathbf { x } \in U , r > 0$ satisfy $B ( \mathbf { x } , r ) \subseteq U$ . Then for any $\mathbf { y } \in B ( \mathbf { x } , r )$ there exists $\pmb { \xi } \in [ \mathbf { x } , \mathbf { y } ]$ such that 

$$
f (\mathbf {y}) = f (\mathbf {x}) + \nabla f (\mathbf {x}) ^ {T} (\mathbf {y} - \mathbf {x}) + \frac {1}{2} (\mathbf {y} - \mathbf {x}) ^ {T} \nabla^ {2} f (\boldsymbol {\xi}) (\mathbf {y} - \mathbf {x}).
$$

Example 5.11 $\left( \left( p \textrm { -- } \mathbf { 1 } \right) \right.$ -smoothness of the half-squared ${ l } _ { p } .$ -norm function) $\cdot ^ { 2 5 }$ Consider the convex function $f : \mathbb { R } ^ { n }  \mathbb { R }$ given by 

$$
f (\mathbf {x}) = \frac {1}{2} \| \mathbf {x} \| _ {p} ^ {2} = \frac {1}{2} \left(\sum_ {i = 1} ^ {n} | x _ {i} | ^ {p}\right) ^ {\frac {2}{p}},
$$

where $p \in [ 2 , \infty )$ . We assume that $\mathbb { R } ^ { n }$ is endowed with the $l _ { p } { \mathrm { - n o r m } }$ and show that $f$ is $( p - 1 )$ )-smooth w.r.t. the $l _ { p } { \mathrm { - n o r m } }$ . The result was already established for the case $p = 2$ in Example 5.2, and we will henceforth assume that $p > 2$ . We begin by computing the partial derivatives: 

$$
\frac {\partial f}{\partial x _ {i}} (\mathbf {x}) = \left\{ \begin{array}{l l} \operatorname{sgn} (x _ {i}) \frac {| x _ {i} | ^ {p - 1}}{\| \mathbf {x} \| _ {p} ^ {p - 2}}, & \mathbf {x} \neq \mathbf {0}, \\ 0, & \mathbf {x} = \mathbf {0}, \end{array} \right.
$$

The partial derivatives are continuous over $\mathbb { R } ^ { n }$ , and hence $f$ is diferentiable over $\mathbb { R } ^ { n }$ (in the sense of Definition 3.28).<sup>26</sup> The second-order partial derivatives exist for any $\mathbf { x } \neq \mathbf { 0 }$ and are given by 

$$
\frac {\partial^ {2} f}{\partial x _ {i} \partial x _ {j}} (\mathbf {x}) = \left\{ \begin{array}{l l} (2 - p) \mathrm{sgn} (x _ {i}) \mathrm{sgn} (x _ {j}) \frac {| x _ {i} | ^ {p - 1} | x _ {j} | ^ {p - 1}}{\| \mathbf {x} \| _ {p} ^ {2 p - 2}}, & i \neq j, \\ (p - 1) \frac {| x _ {i} | ^ {p - 2}}{\| \mathbf {x} \| _ {p} ^ {p - 2}} + (2 - p) \frac {| x _ {i} | ^ {2 p - 2}}{\| \mathbf {x} \| _ {p} ^ {2 p - 2}}, & i = j. \end{array} \right.
$$

It is easy to see that the second-order partial derivatives are continuous for any $\mathbf { x } \neq \mathbf { 0 }$ . We will show that property (ii) of Theorem 5.8 is satisfied with $L = p - 1$ Let $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { n }$ be such that $\mathbf { 0 } \not \in [ \mathbf { x } , \mathbf { y } ]$ . Then by the linear approximation theorem (Theorem 5.10)—taking U to be some open set containing $\left[ \mathbf { x } , \mathbf { y } \right]$ but not containing 0—there exists $\pmb { \xi } \in [ \mathbf { x } , \mathbf { y } ]$ for which 

$$
f (\mathbf {y}) = f (\mathbf {x}) + \nabla f (\mathbf {x}) ^ {T} (\mathbf {y} - \mathbf {x}) + \frac {1}{2} (\mathbf {y} - \mathbf {x}) ^ {T} \nabla^ {2} f (\boldsymbol {\xi}) (\mathbf {y} - \mathbf {x}).\tag{5.7}
$$

We will show that $\mathbf { d } ^ { T } \nabla ^ { 2 } f ( \boldsymbol { \xi } ) \mathbf { d } \mathbf { \xi } \leq ( p - 1 ) \| \mathbf { d } \| _ { p } ^ { 2 }$ for any d $\in \mathbb { R } ^ { n }$ . Since $\nabla ^ { 2 } f ( t \pmb { \xi } ) =$ $\nabla ^ { 2 } f ( \pmb { \xi } )$ for any $t \in \mathbb { R }$ , we can assume without loss of generality that $\| \pmb { \xi } \| _ { p } = 1$ Now, for any d $\in \mathbb { R } ^ { n }$ 

$$
\begin{array}{l} \mathbf {d} ^ {T} \nabla^ {2} f (\boldsymbol {\xi}) \mathbf {d} = (2 - p) \| \boldsymbol {\xi} \| _ {p} ^ {2 - 2 p} \left(\sum_ {i = 1} ^ {n} | \xi_ {i} | ^ {p - 1} \mathrm{sgn} (\xi_ {i}) d _ {i}\right) ^ {2} + (p - 1) \| \boldsymbol {\xi} \| _ {p} ^ {2 - p} \sum_ {i = 1} ^ {n} | \xi_ {i} | ^ {p - 2} d _ {i} ^ {2} \\ \leq (p - 1) \| \boldsymbol {\xi} \| _ {p} ^ {2 - p} \sum_ {i = 1} ^ {n} | \xi_ {i} | ^ {p - 2} d _ {i} ^ {2}, \end{array} \tag {5.8}
$$

where the last inequality follows by the fact that $p > 2$ . Using the generalized Cauchy–Schwarz inequality (Lemma 1.4) with $\begin{array} { r } { \| \cdot \| = \| \cdot \| _ { \frac { p } { p - 2 } } } \end{array}$ , we have 

$$
\begin{array}{r l} & {\sum_ {i = 1} ^ {n} | \xi_ {i} | ^ {p - 2} d _ {i} ^ {2} \leq \left(\sum_ {i = 1} ^ {n} (| \xi_ {i} | ^ {p - 2}) ^ {\frac {p}{p - 2}}\right) ^ {\frac {p - 2}{p}} \left(\sum_ {i = 1} ^ {n} (d _ {i} ^ {2}) ^ {\frac {p}{2}}\right) ^ {\frac {2}{p}}} \\ & {\qquad = \left(\sum_ {i = 1} ^ {n} | \xi_ {i} | ^ {p}\right) ^ {\frac {p - 2}{p}} \left(\sum_ {i = 1} ^ {n} | d _ {i} | ^ {p}\right) ^ {\frac {2}{p}}} \\ & {\qquad = \| \mathbf {d} \| _ {p} ^ {2}.} \end{array}\tag{5.9}
$$

Combining (5.8) and (5.9), we obtain that for any $\mathbf { d } \in \mathbb { R } ^ { n }$ ， 

$$
\mathbf {d} ^ {T} \nabla^ {2} f (\boldsymbol {\xi}) \mathbf {d} \leq (p - 1) \| \mathbf {d} \| _ {p} ^ {2},
$$

and specifically, for $\mathbf { d } = \mathbf { x } - \mathbf { y }$ 7 

$$
(\mathbf {y} - \mathbf {x}) ^ {T} \nabla^ {2} f (\pmb {\xi}) (\mathbf {y} - \mathbf {x}) \leq (p - 1) \| \mathbf {x} - \mathbf {y} \| _ {p} ^ {2}.
$$

Plugging the above inequality into (5.7) implies the inequality 

$$
f (\mathbf {y}) \leq f (\mathbf {x}) + \nabla f (\mathbf {x}) ^ {T} (\mathbf {y} - \mathbf {x}) + \frac {p - 1}{2} \| \mathbf {x} - \mathbf {y} \| _ {p} ^ {2}.\tag{5.10}
$$

The inequality (5.10) was proven for any x, $\mathbf { y } \in \mathbb { R } ^ { n }$ that satisfy $\mathbf { 0 } \not \in [ \mathbf { x } , \mathbf { y } ]$ . We can show that it holds for any $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { n }$ using a continuity argument. Indeed, assume that $\mathbf { 0 } \in [ \mathbf { x } , \mathbf { y } ]$ . Then we can find a sequence $\{ \mathbf { y } _ { k } \} _ { k \ge 0 }$ converging to $\mathbf { y }$ for which 0 $\notin [ \mathbf { x } , \mathbf { y } _ { k } ]$ . Thus, by what was already proven, for any $k \geq 0$ 

$$
f (\mathbf {y} _ {k}) \leq f (\mathbf {x}) + \nabla f (\mathbf {x}) ^ {T} (\mathbf {y} _ {k} - \mathbf {x}) + \frac {p - 1}{2} \| \mathbf {x} - \mathbf {y} _ {k} \| _ {p} ^ {2}.
$$

Taking $k  \infty$ in the last inequality and using the continuity of $f ,$ , we obtain that (5.10) holds. To conclude, we established that (5.10) holds for any $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { n }$ , and thus by Theorem 5.8 (equivalence between properties (i) and (ii)) and the convexity of $f ,$ it follows that $f$ is $( p - 1 )$ -smooth w.r.t. the $l _ { p } { \mathrm { - n o r m } }$ 

## 5.1.3 Second-Order Characterization

We will now consider the space $\mathbb { E } = \mathbb { R } ^ { n }$ endowed with the $l _ { p } { \mathrm { - n o r m ~ } } ( p \geq 1 )$ . For twice continuously diferentiable functions, it is possible to fully characterize the property of L-smoothness via the norm of the Hessian matrix. 

Theorem 5.12 (L-smoothness and boundedness of the Hessian). Let $f$ : $\mathbb { R } ^ { n } \to \mathbb { R }$ be a twice continuously diferentiable function over $\mathbb { R } ^ { n }$ . Then for a given $L \geq 0$ , the following two claims are equivalent: 

(i) f is L-smooth w.r.t. the l<sub>p</sub>-norm $( p \in [ 1 , \infty ] )$ 

(ii) $\| \nabla ^ { 2 } f ( \mathbf { x } ) \| _ { p , q } \leq L$ for any $\mathbf { x } \in \mathbb { R } ^ { n }$ , where $q \in [ 1 , \infty ]$ satisfies $\textstyle { \frac { 1 } { p } } + { \frac { 1 } { q } } = 1$ 

Proof. $\mathrm { ( i i ) } \Rightarrow \mathrm { ( i ) }$ . Suppose that $\| \nabla ^ { 2 } f ( \mathbf { x } ) \| _ { p , q } \leq L$ for any $\mathbf { x } \in \mathbb { R } ^ { n }$ . Then by the fundamental theorem of calculus, for all $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { n }$ 

$$
\begin{array}{l} \nabla f (\mathbf {y}) = \nabla f (\mathbf {x}) + \int_ {0} ^ {1} \nabla^ {2} f (\mathbf {x} + t (\mathbf {y} - \mathbf {x})) (\mathbf {y} - \mathbf {x}) d t \\ \qquad = \nabla f (\mathbf {x}) + \left(\int_ {0} ^ {1} \nabla^ {2} f (\mathbf {x} + t (\mathbf {y} - \mathbf {x})) d t\right) \cdot (\mathbf {y} - \mathbf {x}). \end{array}
$$

Then 

$$
\begin{array}{l} \| \nabla f (\mathbf {y}) - \nabla f (\mathbf {x}) \| _ {q} = \left\| \left(\int_ {0} ^ {1} \nabla^ {2} f (\mathbf {x} + t (\mathbf {y} - \mathbf {x})) d t\right) \cdot (\mathbf {y} - \mathbf {x}) \right\| _ {q} \\ \leq \left\| \int_ {0} ^ {1} \nabla^ {2} f (\mathbf {x} + t (\mathbf {y} - \mathbf {x})) d t \right\| _ {p, q} \| \mathbf {y} - \mathbf {x} \| _ {p} \\ \leq \left(\int_ {0} ^ {1} \| \nabla^ {2} f (\mathbf {x} + t (\mathbf {y} - \mathbf {x})) \| _ {p, q} d t\right) \| \mathbf {y} - \mathbf {x} \| _ {p} \\ \leq L \| \mathbf {y} - \mathbf {x} \| _ {p}, \end{array}
$$

establishing (i). 

$( \mathrm { i } ) \Rightarrow ( \mathrm { i i } )$ . Suppose now that $f$ is L-smooth w.r.t. the $l _ { p } { \mathrm { - n o r m } }$ . Then by the fundamental theorem of calculus, for any d $\in \mathbb { R } ^ { n }$ and $\alpha > 0$ 2 

$$
\nabla f (\mathbf {x} + \alpha \mathbf {d}) - \nabla f (\mathbf {x}) = \int_ {0} ^ {\alpha} \nabla^ {2} f (\mathbf {x} + t \mathbf {d}) \mathbf {d} d t.
$$

Thus, 

$$
\left\| \left(\int_ {0} ^ {\alpha} \nabla^ {2} f (\mathbf {x} + t \mathbf {d}) d t\right) \mathbf {d} \right\| _ {q} = \| \nabla f (\mathbf {x} + \alpha \mathbf {d}) - \nabla f (\mathbf {x}) \| _ {q} \leq \alpha L \| \mathbf {d} \| _ {p}.
$$

Dividing by α and taking the limit $\alpha \to 0 ^ { + }$ , we obtain 

$$
\left\| \nabla^ {2} f (\mathbf {x}) \mathbf {d} \right\| _ {q} \leq L \| \mathbf {d} \| _ {p} \text {   for   any   } \mathbf {d} \in \mathbb {R} ^ {n},
$$

implying that $\| \nabla ^ { 2 } f ( \mathbf { x } ) \| _ { p , q } \leq L$ 

A direct consequence is that for twice continuously diferentiable convex functions, L-smoothness w.r.t. the $l _ { 2 } { \mathrm { - n o r m } }$ is equivalent to the property that the maximum eigenvalue of the Hessian matrix is smaller than or equal to L. 

Corollary 5.13. Let $f : \mathbb { R } ^ { n }  \mathbb { R }$ be a twice continuously diferentiable convex function over R<sup>n</sup>. Then f is L-smooth w.r.t. the $l _ { 2 } { - } n o r m$ if and only if $\lambda _ { \operatorname* { m a x } } ( \nabla ^ { 2 } f ( \mathbf { x } ) ) \leq$ L for any $\mathbf { x } \in \mathbb { R } ^ { n }$ 

Proof. Since f is convex, it follows that $\nabla ^ { 2 } f ( \mathbf { x } ) \succeq \mathbf { 0 }$ for any $\mathbf { x } \in \mathbb { R } ^ { n }$ . Therefore, in this case, 

$$
\| \nabla^ {2} f (\mathbf {x}) \| _ {2, 2} = \sqrt {\lambda_ {\max} ((\nabla^ {2} f (\mathbf {x})) ^ {2})} = \lambda_ {\max} (\nabla^ {2} f (\mathbf {x})),
$$

which, combined with Theorem 5.12, establishes the desired result. 

Example 5.14 (1-smoothness of $\sqrt { { \bf 1 } + { \bf \| } \cdot { \bf \| } _ { 2 } ^ { 2 } }$ w.r.t. the $l _ { 2 } { \bf - n o r m } )$ . Let $f$ : $\mathbb { R } ^ { n } \to \mathbb { R }$ be the convex function given by 

$$
f (\mathbf {x}) = \sqrt {1 + \| \mathbf {x} \| _ {2} ^ {2}}.
$$

We will show that $f$ is 1-smooth w.r.t. the $l _ { 2 } { \mathrm { - n o r m } }$ . For any $\mathbf { x } \in \mathbb { R } ^ { n }$ ， 

$$
\nabla f (\mathbf {x}) = \frac {\mathbf {x}}{\sqrt {\| \mathbf {x} \| _ {2} ^ {2} + 1}}
$$

and 

$$
\nabla^ {2} f (\mathbf {x}) = \frac {1}{\sqrt {\| \mathbf {x} \| _ {2} ^ {2} + 1}} \mathbf {I} - \frac {\mathbf {x x} ^ {T}}{(\| \mathbf {x} \| _ {2} ^ {2} + 1) ^ {3 / 2}} \preceq \frac {1}{\sqrt {\| \mathbf {x} \| _ {2} ^ {2} + 1}} \mathbf {I} \preceq \mathbf {I}.
$$

Therefore, $\lambda _ { \operatorname* { m a x } } ( \nabla ^ { 2 } f ( \mathbf { x } ) ) \leq 1$ for all $\mathbf { x } \in \mathbb { R } ^ { n }$ , and hence by Corollary 5.13 it follows that f is 1-smooth w.r.t. the $l _ { 2 } { \mathrm { - n o r m } }$ ■ 

Example 5.15 (1-smoothness of the log-sum-exp function w.r.t. the ${ \mathbf { } } l _ { 2 } , { \mathbf { } } l _ { \infty }$ norms). Consider the log-sum-exp function $f : \mathbb { R } ^ { n }  \mathbb { R }$ given by 

$$
f (\mathbf {x}) = \log \left(e ^ {x _ {1}} + e ^ {x _ {2}} + \dots + e ^ {x _ {n}}\right).
$$

We will first show that it is 1-smooth w.r.t. the $l _ { 2 } \mathrm { - n o r m }$ . The partial derivatives of $f$ are 

$$
\frac {\partial f}{\partial x _ {i}} (\mathbf {x}) = \frac {e ^ {x _ {i}}}{\sum_ {k = 1} ^ {n} e ^ {x _ {k}}}, i = 1, 2, \ldots , n,
$$

and the second-order partial derivatives are 

$$
\frac {\partial^ {2} f}{\partial x _ {i} \partial x _ {j}} (\mathbf {x}) = \left\{ \begin{array}{l l} - \frac {e ^ {x _ {i}} e ^ {x _ {j}}}{(\sum_ {k = 1} ^ {n} e ^ {x _ {k}}) ^ {2}}, & i \neq j, \\ - \frac {e ^ {x _ {i}} e ^ {x _ {i}}}{(\sum_ {k = 1} ^ {n} e ^ {x _ {k}}) ^ {2}} + \frac {e ^ {x _ {i}}}{\sum_ {k = 1} ^ {n} e ^ {x _ {k}}}, & i = j. \end{array} \right.
$$

We can thus write the Hessian matrix as 

$$
\nabla^ {2} f (\mathbf {x}) = \mathrm{diag} (\mathbf {w}) - \mathbf {w w} ^ {T},
$$

where $\begin{array} { r } { w _ { i } = \frac { e ^ { x _ { i } } } { \sum _ { k = 1 } ^ { n } e ^ { x _ { k } } } } \end{array}$ . To show that f is 1-smooth w.r.t. the $l _ { 2 } { \mathrm { - n o r m } }$ , note that for any $\mathbf { x } \in \mathbb { R } ^ { n }$ 

$$
\nabla^ {2} f (\mathbf {x}) = \mathrm{diag} (\mathbf {w}) - \mathbf {w w} ^ {T} \preceq \mathrm{diag} (\mathbf {w}) \preceq \mathbf {I},
$$

and hence $\lambda _ { \operatorname* { m a x } } ( \nabla ^ { 2 } f ( \mathbf { x } ) ) \leq 1$ for any $\mathbf { x } \in \mathbb { R } ^ { n }$ . Noting that the log-sum-exp function is convex, we can invoke Corollary 5.13 and conclude that f is 1-smooth w.r.t. the $l _ { 2 } { \mathrm { - n o r m } }$ 

We will show that f is 1-smooth also w.r.t. the $l _ { \infty } \mathrm { - n o r m }$ . For that, we begin by proving that for any $\mathbf { d } \in \mathbb { R } ^ { n }$ ， 

$$
\mathbf {d} ^ {T} \nabla^ {2} f (\mathbf {x}) \mathbf {d} \leq \| \mathbf {d} \| _ {\infty} ^ {2}.\tag{5.11}
$$

Indeed, 

$$
\begin{array}{l} \mathbf {d} ^ {T} \nabla^ {2} f (\mathbf {x}) \mathbf {d} = \mathbf {d} ^ {T} (\mathrm{diag} (\mathbf {w}) - \mathbf {w w} ^ {T}) \mathbf {d} = \mathbf {d} ^ {T} \mathrm{diag} (\mathbf {w}) \mathbf {d} - (\mathbf {w} ^ {T} \mathbf {d}) ^ {2} \\ \qquad \leq \mathbf {d} ^ {T} \mathrm{diag} (\mathbf {w}) \mathbf {d} \\ \qquad = \sum_ {i = 1} ^ {n} w _ {i} d _ {i} ^ {2} \\ \qquad \leq \| \mathbf {d} \| _ {\infty} ^ {2} \sum_ {i = 1} ^ {n} w _ {i} \\ \qquad = \| \mathbf {d} \| _ {\infty} ^ {2}. \end{array}
$$

Now, since $f$ is twice continuously diferentiable over $\mathbb { R } ^ { n }$ , it follows by the linear approximation theorem (Theorem 5.10) that for any $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { n }$ there exists $\pmb { \xi } \in [ \mathbf { x } , \mathbf { y } ]$ for which 

$$
f (\mathbf {y}) = f (\mathbf {x}) + \nabla f (\mathbf {x}) ^ {T} (\mathbf {y} - \mathbf {x}) + \frac {1}{2} (\mathbf {y} - \mathbf {x}) ^ {T} \nabla^ {2} f (\boldsymbol {\xi}) (\mathbf {y} - \mathbf {x}).\tag{5.12}
$$

Combining (5.12) (taking $\mathbf { d } = \mathbf { y } - \mathbf { x } )$ and (5.11), we obtain the inequality 

$$
f (\mathbf {y}) \leq f (\mathbf {x}) + \nabla f (\mathbf {x}) ^ {T} (\mathbf {y} - \mathbf {x}) + \frac {1}{2} \| \mathbf {y} - \mathbf {x} \| _ {\infty} ^ {2},
$$

which by Theorem 5.8 (equivalence between properties (i) and (ii)) implies the 1-smoothness of $f$ w.r.t. the $l _ { \infty } \mathrm { - n o r m }$ ■ 

## 5.1.4 Summary of Smoothness Parameter Computations

The table below summarizes the smoothness parameters of the functions discussed in this section. The last function will only be discussed later on in Example 6.62. 

<table><tr><td><eq>f(\mathbf{x})</eq></td><td>dom(f)</td><td>Parameter</td><td>Norm</td><td>Reference</td></tr><tr><td><eq>\frac{1}{2}\mathbf{x}^{T}\mathbf{A}\mathbf{x}+\mathbf{b}^{T}\mathbf{x}+c</eq><eq>(\mathbf{A}\in\mathbb{S}^{n},\mathbf{b}\in\mathbb{R}^{n},c\in\mathbb{R})</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>\| \mathbf{A}\|_{p,q}</eq></td><td><eq>l_{p}</eq></td><td>Example 5.2</td></tr><tr><td><eq>\langle\mathbf{b},\mathbf{x}\rangle+c</eq><eq>(\mathbf{b}\in\mathbb{E}^{*},c\in\mathbb{R})</eq></td><td><eq>\mathbb{E}</eq></td><td>0</td><td>any norm</td><td>Example 5.3</td></tr><tr><td><eq>\frac{1}{2}\|\mathbf{x}\|_{p}^{2},\ p\in[2,\infty)</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>p-1</eq></td><td><eq>l_{p}</eq></td><td>Example 5.11</td></tr><tr><td><eq>\sqrt{1+\|\mathbf{x}\|_{2}^{2}}</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td>1</td><td><eq>l_{2}</eq></td><td>Example 5.14</td></tr><tr><td><eq>\log(\sum_{i=1}^{n}e^{x_{i}})</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td>1</td><td><eq>l_{2},l_{\infty}</eq></td><td>Example 5.15</td></tr><tr><td><eq>\frac{1}{2}d_{C}^{2}(\mathbf{x})</eq><eq>(\emptyset\neq C\subseteq\mathbb{E}\text{ closed convex})</eq></td><td><eq>\mathbb{E}</eq></td><td>1</td><td>Euclidean</td><td>Example 5.5</td></tr><tr><td><eq>\frac{1}{2}\|\mathbf{x}\|^{2}-\frac{1}{2}d_{C}^{2}(\mathbf{x})</eq><eq>(\emptyset\neq C\subseteq\mathbb{E}\text{ closed convex})</eq></td><td><eq>\mathbb{E}</eq></td><td>1</td><td>Euclidean</td><td>Example 5.6</td></tr><tr><td><eq>H_{\mu}(\mathbf{x}) \ ( \mu &gt;0 )</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>\frac{1}{\mu}</eq></td><td>Euclidean</td><td>Example 6.62</td></tr></table>

## 5.2 Strong Convexity

Definition 5.16 (strong convexity). A function $f : \mathbb { E }  ( - \infty , \infty ]$ is called σ-strongly convex for a given $\sigma > 0$ if dom(f) is convex and the following inequality holds for any x, $\mathbf { y } \in \operatorname { d o m } ( f )$ and $\lambda \in [ 0 , 1 ]$ 

$$
f (\lambda \mathbf {x} + (1 - \lambda) \mathbf {y}) \leq \lambda f (\mathbf {x}) + (1 - \lambda) f (\mathbf {y}) - \frac {\sigma}{2} \lambda (1 - \lambda) \| \mathbf {x} - \mathbf {y} \| ^ {2}.\tag{5.13}
$$

We will sometimes use the terminology “strongly convex with parameter $\sigma ^ { \mathfrak { N } }$ instead of “σ-strongly convex.” It is important to note that the strong convexity parameter σ depends on the underlying norm, and we will therefore sometimes refer to it as the “strong convexity parameter w.r.t. $\| \cdot \| . ^ { \mathfrak { s } }$ Obviously, strongly convex functions are necessarily also convex since their domain is assumed to be convex and inequality (5.13) implies that for any $\mathbf { x } , \mathbf { y } \in \operatorname { d o m } ( f )$ and $\lambda \in [ 0 , 1 ]$ , Jensen’s inequality is satisfied: 

$$
f (\lambda \mathbf {x} + (1 - \lambda) \mathbf {y}) \leq \lambda f (\mathbf {x}) + (1 - \lambda) f (\mathbf {y}).
$$

When the underlying set E is Euclidean, meaning that $\| \mathbf { x } \| = { \sqrt { \langle \mathbf { x } , \mathbf { x } \rangle } }$ for any $\mathbf { x } \in \mathbb { E }$ we can write a diferent and simple property that is equivalent to strong convexity. 

Theorem 5.17. Let E be a Euclidean space. Then $f : \mathbb { E } \to ( - \infty , \infty ]$ is a σ-strongly convex function $( \sigma > 0 )$ if and only if the function $f ( \cdot ) - \frac { \sigma } { 2 } \| \cdot \| ^ { 2 }$ is convex. 

Proof. The function $\begin{array} { r } { g ( \mathbf { x } ) \equiv f ( \mathbf { x } ) - \frac { \sigma } { 2 } \| \mathbf { x } \| ^ { 2 } } \end{array}$ is convex if and only if its domain dom $( g ) = \operatorname { d o m } ( f )$ is convex and for any $\mathbf { \bar { x } } , \mathbf { y } \in \operatorname { d o m } ( f )$ and $\lambda \in [ 0 , 1 ]$ 

$$
g (\lambda \mathbf {x} + (1 - \lambda) \mathbf {y}) \leq \lambda g (\mathbf {x}) + (1 - \lambda) g (\mathbf {y}).
$$

The latter inequality is the same as 

$$
f (\lambda \mathbf {x} + (1 - \lambda) \mathbf {y}) \leq \lambda f (\mathbf {x}) + (1 - \lambda) f (\mathbf {y}) + \frac {\sigma}{2} \left[ \| \lambda \mathbf {x} + (1 - \lambda) \mathbf {y} \| ^ {2} - \lambda \| \mathbf {x} \| ^ {2} - (1 - \lambda) \| \mathbf {y} \| ^ {2} \right].\tag{5.14}
$$

Now, using the identity (which holds since the norm is assumed to be Euclidean) 

$$
\| \lambda \mathbf {x} + (1 - \lambda) \mathbf {y} \| ^ {2} - \lambda \| \mathbf {x} \| ^ {2} - (1 - \lambda) \| \mathbf {y} \| ^ {2} = - \lambda (1 - \lambda) \| \mathbf {x} - \mathbf {y} \| ^ {2},
$$

combined with (5.14), we can conclude that the convexity of $g$ is equivalent to the convexity of dom(f) and the validity of the inequality 

$$
f (\lambda \mathbf {x} + (1 - \lambda) \mathbf {y}) \leq \lambda f (\mathbf {x}) + (1 - \lambda) f (\mathbf {y}) - \frac {\sigma}{2} \lambda (1 - \lambda) \| \mathbf {x} - \mathbf {y} \| ^ {2}
$$

for any $\mathbf { x } , \mathbf { y } \in \operatorname { d o m } ( f )$ and $\lambda \in [ 0 , 1 ]$ , namely, to the σ-strong convexity of $f . \quad \sqcup$ 

Remark 5.18. The assumption that the underlying space is Euclidean is essential in Theorem 5.17. As an example, consider the negative entropy function over the unit simplex 

$$
f (\mathbf {x}) \equiv \left\{ \begin{array}{l l} \sum_ {i = 1} ^ {n} x _ {i} \log x _ {i}, & \mathbf {x} \in \Delta_ {n}, \\ \infty & e l s e. \end{array} \right.
$$

We will later show (in Example 5.27) that f is a 1-strongly convex function with respect to the $l _ { 1 } { - } n o r m$ . Regardless of this fact, note that the function 

$$
g (\mathbf {x}) = f (\mathbf {x}) - \alpha \| \mathbf {x} \| _ {1} ^ {2}
$$

is convex for any $\alpha > 0$ since over the domain of f, we have that $\| \mathbf { x } \| _ { 1 } = 1$ Obviously, it is impossible that a function will be α-strongly convex for any $\alpha > 0$ Therefore, the characterization of strong convexity in Theorem 5.17 is not correct for any norm. 

Note that if a function f is $\sigma _ { \mathrm { { 1 } ^ { - s t r o n g l y } } }$ convex $\left( \sigma _ { 1 } > 0 \right)$ , then it is necessarily also σ<sub>2</sub>-strongly convex for any $\sigma _ { 2 } \in ( 0 , \sigma _ { 1 } )$ . An interesting problem is to find the largest possible strong convexity parameter of a given function. 

Example 5.19 (strong convexity of quadratic functions). Suppose that $\mathbb { E } =$ $\mathbb { R } ^ { n }$ is endowed with the $l _ { 2 } { \mathrm { - n o r m } }$ , and consider the quadratic function $f : \mathbb { R } ^ { n } $ R given by 

$$
f (\mathbf {x}) = \frac {1}{2} \mathbf {x} ^ {T} \mathbf {A x} + \mathbf {b} ^ {T} \mathbf {x} + c,
$$

where $\mathbf { A } \in \mathbb { S } ^ { n }$ , b $\in \mathbb { R } ^ { n }$ , and $c \in \mathbb { R }$ . Then by Theorem 5.17, f is strongly convex with parameter $\sigma > 0$ if and only if the function ${ \textstyle { \frac { 1 } { 2 } } } \mathbf { x } ^ { T } \left( \mathbf { A } - \sigma \mathbf { I } \right) \mathbf { x } + \mathbf { b } ^ { T } \mathbf { x } + c$ is convex, which is equivalent to the matrix inequality $\mathbf { A } \bar { - } \sigma \mathbf { I } \succeq \mathbf { 0 }$ , namely, to the inequality $\lambda _ { \operatorname* { m i n } } ( \mathbf { A } ) \geq \sigma$ . Thus, f is strongly convex if and only if A is positive definite, and in that case, $\lambda _ { \mathrm { m i n } } ( \mathbf { A } )$ is its largest possible strong convexity parameter. 

A simple result is that the sum of a strongly convex function and a convex function is always a strongly convex function. 

Lemma 5.20. Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a σ-strongly convex function $( \sigma > 0 )$ , and let $g : \mathbb { E }  ( - \infty , \infty ]$ be convex. Then $f + g$ is σ-strongly convex. 

Proof. Follows directly from the definitions of strong convexity and convexity. Since $f$ and $g$ are convex, both $\operatorname { d o m } ( f )$ and dom(g) are convex sets, and hence also dom $( f + g ) = \operatorname { d o m } ( f ) \cap \operatorname { d o m } ( g )$ is a convex set. Let $\mathbf { x } , \mathbf { y } \in \operatorname { d o m } ( f ) \cap \operatorname { d o m } ( g )$ and $\lambda \in [ 0 , 1 ]$ . Then by the σ-strong convexity of $f ,$ , 

$$
f (\lambda \mathbf {x} + (1 - \lambda) \mathbf {y}) \leq \lambda f (\mathbf {x}) + (1 - \lambda) f (\mathbf {y}) - \frac {\sigma}{2} \lambda (1 - \lambda) \| \mathbf {x} - \mathbf {y} \| ^ {2}.
$$

Since g is convex, 

$$
g (\lambda \mathbf {x} + (1 - \lambda) \mathbf {y}) \leq \lambda g (\mathbf {x}) + (1 - \lambda) g (\mathbf {y}).
$$

Adding the two inequalities, we obtain 

$$
(f + g) (\lambda \mathbf {x} + (1 - \lambda) \mathbf {y}) \leq \lambda (f + g) (\mathbf {x}) + (1 - \lambda) (f + g) (\mathbf {y}) - \frac {\sigma}{2} \lambda (1 - \lambda) \| \mathbf {x} - \mathbf {y} \| ^ {2},
$$

showing that $f + g$ is σ-strongly convex. 

Example 5.21 (strong convexity of $\frac { 1 } { 2 } \lVert \cdot \rVert ^ { 2 } + \delta _ { C } )$ . Suppose that E is a Euclidean space, and let $C \subseteq \mathbb { E }$ be a nonempty convex set. The function $\scriptstyle { \frac { 1 } { 2 } } \left\| \mathbf { x } \right\| ^ { 2 }$ is 1-strongly convex (Example 5.19), and by the convexity of $C , \delta _ { C }$ is convex. Therefore, by Lemma 5.20, the function $\begin{array} { r } { \frac { 1 } { 2 } \| \mathbf { x } \| ^ { 2 } + \delta _ { C } ( \mathbf { x } ) } \end{array}$ is 1-strongly convex. 

Theorem 5.24 below describes two properties that are equivalent to σ-strong convexity. The two properties are of a first-order nature in the sense that they are written in terms of the function and its subgradients. The proof uses the following version of the mean-value theorem for one-dimensional functions. 

Lemma 5.22 (see $\left[ { \bf 6 7 , 9 . 2 6 } \right] )$ . Let $f : \mathbb { R } \to ( - \infty , \infty ]$ be a closed convex function, and let $[ a , b ] \subseteq \mathrm { d o m } ( f ) ( a < b )$ . Then 

$$
f (b) - f (a) = \int_ {a} ^ {b} h (t) d t,
$$

where $h : ( a , b )  \mathbb { R }$ satisfies $h ( t ) \in \partial f ( t )$ for any $t \in ( a , b )$ 

Another technical lemma that is being used in the proof is the so-called line segment principle. 

Lemma 5.23 (line segment principle [108, Theorem 6.1]). Let C be a convex set. Suppose that $\mathbf { x } \in \operatorname { r i } ( C ) , \mathbf { y } \in \operatorname { c l } ( C )$ , and let $\lambda \in ( 0 , 1 ]$ . Then $\lambda \mathbf { x } + ( 1 - \lambda ) \mathbf { y } \in \mathrm { r i } ( C )$ 

Theorem 5.24 (first-order characterizations of strong convexity). Let $f$ : $\mathbb { E } \to ( - \infty , \infty ]$ be a proper closed and convex function. Then for a given $\sigma > 0$ , the following three claims are equivalent: 

(i) f is σ-strongly convex. 

(ii) 

$$
f (\mathbf {y}) \geq f (\mathbf {x}) + \langle \mathbf {g}, \mathbf {y} - \mathbf {x} \rangle + \frac {\sigma}{2} \| \mathbf {y} - \mathbf {x} \| ^ {2}
$$

for any $\mathbf { x } \in \mathrm { d o m } ( \partial f ) , \mathbf { y } \in \mathrm { d o m } ( f ) \ a n d \ \mathbf { g } \in \partial f ( \mathbf { x } )$ 

(iii) 

$$
\left\langle \mathbf {g} _ {\mathbf {x}} - \mathbf {g} _ {\mathbf {y}}, \mathbf {x} - \mathbf {y} \right\rangle \geq \sigma \| \mathbf {x} - \mathbf {y} \| ^ {2}\tag{5.15}
$$

for any x, y ∈ dom(∂f), and $\mathbf { g _ { x } } \in \partial f ( \mathbf { x } ) , \mathbf { g _ { y } } \in \partial f ( \mathbf { y } )$ 

Proof. $\mathrm { ( i i ) } \Rightarrow \mathrm { ( i ) }$ . Assume that (ii) is satisfied. To show (i), take $\mathbf { x } , \mathbf { y } \in \operatorname { d o m } ( f )$ and $\lambda \in \mathsf { \Gamma } ( 0 , 1 )$ Take some $\mathbf { z } \in \operatorname { r i } ( \operatorname { d o m } ( f ) )$ . Then for any $\alpha \in ( 0 , 1 ]$ , by the line segment principle (Lemma 5.23), the vector $\tilde { \mathbf { x } } = ( 1 - \alpha ) \mathbf { x } + \alpha \mathbf { z }$ is in $\operatorname { r i } ( \operatorname { d o m } ( f ) )$ At this point we fix $\alpha .$ Using the notation ${ \bf x } _ { \lambda } = \lambda \tilde { \bf x } + ( 1 - \lambda ) { \bf y } ,$ , we obtain that $\mathbf { x } _ { \lambda } \in \operatorname { r i } ( \operatorname { d o m } ( f ) )$ for any $\lambda \in \mathsf { \Gamma } ( 0 , 1 )$ , and hence, by Theorem 3.18, $\partial f ( \mathbf { x } _ { \lambda } ) \neq \varnothing$ , meaning that $\mathbf { x } _ { \lambda } \in \mathrm { d o m } ( \partial f )$ . Take $\mathbf { g } \in \partial f ( \mathbf { x } _ { \lambda } )$ . Then by (ii), 

$$
f (\tilde {\mathbf {x}}) \geq f (\mathbf {x} _ {\lambda}) + \langle \mathbf {g}, \tilde {\mathbf {x}} - \mathbf {x} _ {\lambda} \rangle + \frac {\sigma}{2} \| \tilde {\mathbf {x}} - \mathbf {x} _ {\lambda} \| ^ {2},
$$

which is the same as 

$$
f (\tilde {\mathbf {x}}) \geq f (\mathbf {x} _ {\lambda}) + (1 - \lambda) \langle \mathbf {g}, \tilde {\mathbf {x}} - \mathbf {y} \rangle + \frac {\sigma (1 - \lambda) ^ {2}}{2} \| \mathbf {y} - \tilde {\mathbf {x}} \| ^ {2}.\tag{5.16}
$$

Similarly, 

$$
f (\mathbf {y}) \geq f (\mathbf {x} _ {\lambda}) + \lambda \langle \mathbf {g}, \mathbf {y} - \tilde {\mathbf {x}} \rangle + \frac {\sigma \lambda^ {2}}{2} \| \mathbf {y} - \tilde {\mathbf {x}} \| ^ {2}.\tag{5.17}
$$

Multiplying (5.16) by λ and (5.17) by $1 - \lambda$ and adding the two resulting inequalities, we obtain that 

$$
f (\lambda \tilde {\mathbf {x}} + (1 - \lambda) \mathbf {y}) \leq \lambda f (\tilde {\mathbf {x}}) + (1 - \lambda) f (\mathbf {y}) - \frac {\sigma \lambda (1 - \lambda)}{2} \| \tilde {\mathbf {x}} - \mathbf {y} \| ^ {2}.
$$

Plugging the expression for $\tilde { \mathbf { x } }$ in the above inequality, we obtain that 

$$
g _ {1} (\alpha) \leq \lambda g _ {2} (\alpha) + (1 - \lambda) f (\mathbf {y}) - \frac {\sigma \lambda (1 - \lambda)}{2} \| (1 - \alpha) \mathbf {x} + \alpha \mathbf {z} - \mathbf {y} \| ^ {2},\tag{5.18}
$$

where $g _ { 1 } ( \alpha ) \equiv f ( \lambda ( 1 - \alpha ) \mathbf { x } + ( 1 - \lambda ) \mathbf { y } + \lambda \alpha \mathbf { z } )$ and $g _ { 2 } ( \alpha ) \equiv f ( ( 1 - \alpha ) \mathbf { x } + \alpha \mathbf { z } )$ . The functions $g _ { 1 }$ and $g _ { 2 }$ are one-dimensional proper closed and convex functions, and consequently, by Theorem 2.22, they are also continuous over their domain. Thus, taking $\alpha \to 0 ^ { + }$ in (5.18), it follows that 

$$
g _ {1} (0) \leq \lambda g _ {2} (0) + (1 - \lambda) f (\mathbf {y}) - \frac {\sigma \lambda (1 - \lambda)}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.
$$

Finally, since $g _ { 1 } ( 0 ) = f ( \lambda \mathbf { x } + ( 1 - \lambda ) \mathbf { y } )$ and $g _ { 2 } ( 0 ) = f ( \mathbf { x } )$ , we obtain the inequality 

$$
f (\lambda \mathbf {x} + (1 - \lambda) \mathbf {y}) \leq \lambda f (\mathbf {x}) + (1 - \lambda) f (\mathbf {y}) - \frac {\sigma \lambda (1 - \lambda)}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2},
$$

establishing the σ-strong convexity of $f .$ 

$( \mathrm { i } ) \ \Rightarrow \ \mathrm { ( i i i ) }$ . Assume that (i) is satisfied. Let $\mathbf { x } , \mathbf { y } \in \mathsf { d o m } ( \partial f )$ and $\mathbf { g } _ { \mathbf { x } } \in$ $\partial f ( \mathbf { x } ) , \mathbf { g } _ { \mathbf { y } } \in \partial f ( \mathbf { y } )$ . We will show that inequality (5.15) is satisfied. Let $\lambda \in [ 0 , 1 )$ and denote $\mathbf { x } _ { \lambda } = \lambda \mathbf { x } + ( 1 - \lambda ) \mathbf { y }$ . By condition (i), 

$$
f (\mathbf {x} _ {\lambda}) \leq \lambda f (\mathbf {x}) + (1 - \lambda) f (\mathbf {y}) - \frac {\sigma}{2} \lambda (1 - \lambda) \| \mathbf {x} - \mathbf {y} \| ^ {2},
$$

which is the same as 

$$
\frac {f (\mathbf {x} _ {\lambda}) - f (\mathbf {x})}{1 - \lambda} \leq f (\mathbf {y}) - f (\mathbf {x}) - \frac {\sigma}{2} \lambda \| \mathbf {x} - \mathbf {y} \| ^ {2},\tag{5.19}
$$

Since $\mathbf { g } _ { \mathbf { x } } \in \partial f ( \mathbf { x } )$ 

$$
\frac {f (\mathbf {x} _ {\lambda}) - f (\mathbf {x})}{1 - \lambda} \geq \frac {\langle \mathbf {g} _ {\mathbf {x}} , \mathbf {x} _ {\lambda} - \mathbf {x} \rangle}{1 - \lambda} = \langle \mathbf {g} _ {\mathbf {x}}, \mathbf {y} - \mathbf {x} \rangle ,
$$

which, combined with (5.19), yields the inequality 

$$
\langle \mathbf {g} _ {\mathbf {x}}, \mathbf {y} - \mathbf {x} \rangle \leq f (\mathbf {y}) - f (\mathbf {x}) - \frac {\sigma \lambda}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.\tag{5.20}
$$

Inequality (5.20) holds for any $\lambda \in [ 0 , 1 )$ . Taking the limit $\lambda  1 ^ { - }$ , we conclude that 

$$
\left\langle \mathbf {g} _ {\mathbf {x}}, \mathbf {y} - \mathbf {x} \right\rangle \leq f (\mathbf {y}) - f (\mathbf {x}) - \frac {\sigma}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.\tag{5.21}
$$

Changing the roles of x and y yields the inequality 

$$
\langle \mathbf {g} _ {\mathbf {y}}, \mathbf {x} - \mathbf {y} \rangle \leq f (\mathbf {x}) - f (\mathbf {y}) - \frac {\sigma}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.\tag{5.22}
$$

Adding inequalities (5.21) and (5.22), we can finally conclude that 

$$
\left\langle \mathbf {g} _ {\mathbf {x}} - \mathbf {g} _ {\mathbf {y}}, \mathbf {x} - \mathbf {y} \right\rangle \geq \sigma \| \mathbf {x} - \mathbf {y} \| ^ {2},
$$

which is the desired inequality. 

(iii) ⇒ (ii) Suppose that (iii) is satisfied, and let $\mathbf { x } \in \mathrm { d o m } ( \partial f ) , \mathbf { y } \in \mathrm { d o m } ( f )$ and $\mathbf { g } \in \partial f ( \mathbf { x } )$ . Let z be any vector in $\operatorname { r i } ( \operatorname { d o m } ( f ) )$ , and define $\tilde { \mathbf { y } } = ( 1 - \alpha ) \mathbf { y } + \alpha \mathbf { z }$ for some $\alpha \in ( 0 , 1 )$ , which at this point we fix. By the line segment principle (Lemma 5.23), $\tilde { \mathbf { y } } \in \operatorname { r i } ( \operatorname { d o m } ( f ) )$ . Consider now the one-dimensional function 

$$
\varphi (\lambda) = f (\mathbf {x} _ {\lambda}), \quad \lambda \in [ 0, 1 ],
$$

where $\mathbf { x } _ { \lambda } = ( 1 - \lambda ) \mathbf { x } + \lambda \tilde { \mathbf { y } }$ . For any $\lambda \in ( 0 , 1 )$ , let $\mathbf { g } _ { \lambda } \in \partial f ( \mathbf { x } _ { \lambda } )$ (whose existence is guaranteed since $\mathbf { x } _ { \lambda } \in \operatorname { r i } ( \operatorname { d o m } ( f ) )$ by the line segment principle). Then $\left. \mathbf { g } _ { \lambda } , \tilde { \mathbf { y } } - \mathbf { x } \right. \in$ $\partial \varphi ( \lambda )$ , and hence by the mean-value theorem (Lemma 5.22), 

$$
f (\tilde {\mathbf {y}}) - f (\mathbf {x}) = \varphi (1) - \varphi (0) = \int_ {0} ^ {1} \langle \mathbf {g} _ {\lambda}, \tilde {\mathbf {y}} - \mathbf {x} \rangle d \lambda .\tag{5.23}
$$

Since $\mathbf { g } \in \partial f ( \mathbf { x } )$ and $\mathbf { g } _ { \lambda } \in \partial f ( \mathbf { x } _ { \lambda } )$ , by property (iii), 

$$
\left\langle \mathbf {g} _ {\lambda} - \mathbf {g}, \mathbf {x} _ {\lambda} - \mathbf {x} \right\rangle \geq \sigma \| \mathbf {x} _ {\lambda} - \mathbf {x} \| ^ {2},
$$

which is equivalent to 

$$
\left\langle \mathbf {g} _ {\lambda}, \tilde {\mathbf {y}} - \mathbf {x} \right\rangle \geq \left\langle \mathbf {g}, \tilde {\mathbf {y}} - \mathbf {x} \right\rangle + \sigma \lambda \| \tilde {\mathbf {y}} - \mathbf {x} \| ^ {2}.
$$

Plugging the last inequality into (5.23), we obtain that 

$$
\begin{array}{c} f (\tilde {\mathbf {y}}) - f (\mathbf {x}) \geq \int_ {0} ^ {1} \left[ \langle \mathbf {g}, \tilde {\mathbf {y}} - \mathbf {x} \rangle + \sigma \lambda \| \tilde {\mathbf {y}} - \mathbf {x} \| ^ {2} \right] d \lambda \\ = \langle \mathbf {g}, \tilde {\mathbf {y}} - \mathbf {x} \rangle + \frac {\sigma}{2} \| \tilde {\mathbf {y}} - \mathbf {x} \| ^ {2}. \end{array}
$$

Recalling the definition of $\tilde { \mathbf { y } } .$ we obtain that for any $\alpha \in ( 0 , 1 )$ ， 

$$
f ((1 - \alpha) \mathbf {y} + \alpha \mathbf {z}) \geq f (\mathbf {x}) + \langle \mathbf {g}, (1 - \alpha) \mathbf {y} + \alpha \mathbf {z} - \mathbf {x} \rangle + \frac {\sigma}{2} \| (1 - \alpha) \mathbf {y} + \alpha \mathbf {z} - \mathbf {x} \| ^ {2}.
$$

Taking $\alpha  0 ^ { + }$ and using the continuity of the one-dimensional function $\alpha \mapsto$ $f ( ( 1 - \alpha ) \mathbf { y } + \alpha \mathbf { z } )$ over [0, 1] (follows by invoking Theorem 2.22 and recalling that the one-dimensional function is closed and convex), we obtain the desired result 

$$
f (\mathbf {y}) \geq f (\mathbf {x}) + \langle \mathbf {g}, \mathbf {y} - \mathbf {x} \rangle + \frac {\sigma}{2} \| \mathbf {y} - \mathbf {x} \| ^ {2}. \quad \square
$$

The next theorem states that a proper closed and strongly convex function has a unique minimizer and that it satisfies a certain growth property around the minimizer. 

Theorem 5.25 (existence and uniqueness of a minimizer of closed strongly convex functions). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper closed and σ-strongly convex function $( \sigma > 0 )$ . Then 

(a) f has a unique minimizer; 

(b) $\begin{array} { r } { f ( \mathbf { x } ) - f ( \mathbf { x } ^ { * } ) \geq \frac { \sigma } { 2 } \Vert \mathbf { x } - \mathbf { x } ^ { * } \Vert ^ { 2 } } \end{array}$ for all $\mathbf { x } \in \operatorname { d o m } ( f )$ , where $\mathbf { x } ^ { * }$ is the unique minimizer of f. 

Proof. (a) Since $\operatorname { d o m } ( f )$ is nonempty and convex, it follows that there exists $\mathbf { x } _ { 0 } \in \operatorname { r i } ( \operatorname { d o m } ( f ) )$ (Theorem 3.17), and consequently, by Theorem 3.18, $\partial f ( \mathbf { x } _ { 0 } ) \neq \emptyset$ Let $\mathbf { g } \in \partial f ( \mathbf { x } _ { 0 } )$ . Then by the equivalence between σ-strong convexity and property (ii) of Theorem 5.24, it follows that 

$$
f (\mathbf {x}) \geq f \left(\mathbf {x} _ {0}\right) + \langle \mathbf {g}, \mathbf {x} - \mathbf {x} _ {0} \rangle + \frac {\sigma}{2} \| \mathbf {x} - \mathbf {x} _ {0} \| ^ {2} \text {   for   all   } \mathbf {x} \in \mathbb {E}.
$$

Since all norms in finite dimensional spaces are equivalent, there exists a constant $C > 0$ such that 

$$
\| \mathbf {y} \| \geq \sqrt {C} \| \mathbf {y} \| _ {a},
$$

where $\| \cdot \| _ { a } \equiv \sqrt { \langle \cdot , \cdot \rangle }$ denotes the Euclidean norm associated with the inner product of the space E (which might be diferent than the endowed norm $\| \cdot \| )$ . Therefore, 

$$
f (\mathbf {x}) \geq f (\mathbf {x} _ {0}) + \langle \mathbf {g}, \mathbf {x} - \mathbf {x} _ {0} \rangle + \frac {C \sigma}{2} \| \mathbf {x} - \mathbf {x} _ {0} \| _ {a} ^ {2} \text {   for   any   } \mathbf {x} \in \mathbb {E},
$$

which is the same as 

$$
f (\mathbf {x}) \geq f (\mathbf {x} _ {0}) - \frac {1}{2 C \sigma} \| \mathbf {g} \| _ {a} ^ {2} + \frac {C \sigma}{2} \left\| \mathbf {x} - \left(\mathbf {x} _ {0} - \frac {1}{C \sigma} \mathbf {g}\right) \right\| _ {a} ^ {2} \text {for any} \mathbf {x} \in \mathbb {E}.
$$

In particular, it follows that 

$$
\operatorname{Lev} (f, f (\mathbf {x} _ {0})) \subseteq B _ {\| \cdot \| _ {a}} \left[ \mathbf {x} _ {0} - \frac {1}{C \sigma} \mathbf {g}, \frac {1}{C \sigma} \| \mathbf {g} \| _ {a} \right].
$$

Since $f$ is closed, the above level set is closed (Theorem 2.6), and since it is contained in a ball, it is also bounded. Therefore, $\operatorname { L e v } ( f , f ( \mathbf { x } _ { 0 } ) )$ is compact. We can thus deduce that the optimal set of the problem of minimizing f over dom(f) is the same as the optimal set of the problem of minimizing $f$ over the nonempty compact set $\operatorname { L e v } ( f , f ( \mathbf { x } _ { 0 } ) )$ . Invoking Weierstrass theorem for closed functions (Theorem 2.12), it follows that a minimizer exists. To show the uniqueness, assume that $\tilde { \mathbf { x } }$ and xˆ are minimizers of $f$ . Then $f ( \tilde { \mathbf { x } } ) = f ( \hat { \mathbf { x } } ) = f _ { \mathrm { o p t } }$ , where $f _ { \mathrm { o p t } }$ is the minimal value of $f .$ . Then by the definition of σ-strong convexity of $f$ , 

$$
f _ {\mathrm{opt}} \leq f \left(\frac {1}{2} \tilde {\mathbf {x}} + \frac {1}{2} \hat {\mathbf {x}}\right) \leq \frac {1}{2} f (\tilde {\mathbf {x}}) + \frac {1}{2} f (\hat {\mathbf {x}}) - \frac {\sigma}{8} \| \tilde {\mathbf {x}} - \hat {\mathbf {x}} \| ^ {2} = f _ {\mathrm{opt}} - \frac {\sigma}{8} \| \tilde {\mathbf {x}} - \hat {\mathbf {x}} \| ^ {2},
$$

implying that $\tilde { \mathbf { x } } = \hat { \mathbf { x } }$ and hence establishing the uniqueness of the minimizer of $f .$ (b) Let $\mathbf { x } ^ { * }$ be the unique minimizer of $f .$ Then by Fermat’s optimality condition (Theorem 3.63), $\mathbf { 0 } \in \partial f ( \mathbf { x } ^ { * } )$ and hence by using the equivalence between σ-strong convexity and property (ii) of Theorem 5.24, it follows that 

$$
f (\mathbf {x}) - f \left(\mathbf {x} ^ {*}\right) \geq \langle \mathbf {0}, \mathbf {x} - \mathbf {x} ^ {*} \rangle + \frac {\sigma}{2} \| \mathbf {x} - \mathbf {x} ^ {*} \| ^ {2} = \frac {\sigma}{2} \| \mathbf {x} - \mathbf {x} ^ {*} \| ^ {2}\tag{5.24}
$$

for any $\mathbf { x } \in \mathbb { E }$ , establishing claim (b). 

## 5.3 Smoothness and Strong Convexity Correspondence

## 5.3.1 The Conjugate Correspondence Theorem

An extremely useful connection between smoothness and strong convexity is given in the conjugate correspondence theorem that, loosely speaking, states that $f$ is σ-strongly convex if and only if $f ^ { * }$ is $\scriptstyle { \frac { 1 } { \sigma } } \to \operatorname { s m o o t h }$ 

Theorem 5.26 (conjugate correspondence theorem). Let $\sigma > 0$ . Then 

(a) If $f : \mathbb { E }  \mathbb { R }$ is $a ~ { \frac { 1 } { \sigma } } - s m o o t h$ convex function, then $f ^ { * }$ is σ-strongly convex w.r.t. the dual norm $\Vert \cdot \Vert _ { * }$ 

(b) $I f \ f : \mathbb { E } \to ( - \infty , \infty ]$ is a proper closed σ-strongly convex function, then $f ^ { * }$ : $\mathbb { E } ^ { * } \to \mathbb { R } \ i s \ \frac { \mathrm { 1 } } { \sigma } \mathrm { - } s m o o t h$ 

Proof. (a) Suppose that $f : \mathbb { E }  \mathbb { R }$ is a $\scriptstyle { \frac { 1 } { \sigma } } \to \operatorname { s m o o t h }$ convex function. To prove that $f ^ { * }$ is σ-strongly convex, take $\mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } \in \mathrm { d o m } ( \partial f ^ { * } )$ and ${ \bf v } _ { 1 } \in \partial f ^ { * } ( { \bf y } _ { 1 } ) , { \bf v } _ { 2 } \in$ $\partial f ^ { * } ( \mathbf { y } _ { 2 } )$ . Then by the conjugate subgradient theorem (Theorem 4.20), using also the properness closedness and convexity of $f ,$ it follows that $\mathbf { y } _ { 1 } ~ \in ~ \partial f ( \mathbf { v } _ { 1 } )$ and $\mathbf { y } _ { 2 } \in \partial f ( \mathbf { v } _ { 2 } )$ , which, by the diferentiability of $f ,$ implies that ${ \bf y } _ { 1 } = \nabla f ( { \bf v } _ { 1 } )$ and ${ \bf y } _ { 2 } = \nabla f ( { \bf v } _ { 2 } )$ (see Theorem 3.33). By the equivalence between properties (i) and (iv) in Theorem 5.8, we can write 

$$
\left\langle \mathbf {y} _ {1} - \mathbf {y} _ {2}, \mathbf {v} _ {1} - \mathbf {v} _ {2} \right\rangle \geq \sigma \| \mathbf {y} _ {1} - \mathbf {y} _ {2} \| _ {*} ^ {2}.
$$

Since the last inequality holds for any $\mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } \in \mathrm { d o m } ( \partial f ^ { * } )$ and ${ \bf v } _ { 1 } \in \partial f ^ { * } ( { \bf y } _ { 1 } ) , { \bf v } _ { 2 } \in$ $\partial f ^ { * } ( \mathbf { y } _ { 2 } )$ , it follows by the equivalence between σ-strong convexity and property (iii) of Theorem 5.24 that $f ^ { * }$ is a σ-strongly convex function. 

(b) Suppose that $f$ is a proper closed σ-strongly convex function. By the conjugate subgradient theorem (Corollary 4.21), 

$$
\partial f ^ {*} (\mathbf {y}) = \operatorname{argmax} _ {\mathbf {x} \in \mathbb {E}} \{\langle \mathbf {x}, \mathbf {y} \rangle - f (\mathbf {x}) \} \text {for any} \mathbf {y} \in \mathbb {E} ^ {*}.
$$

Thus, by the strong convexity and closedness of $f ,$ , along with Theorem $5 . 2 5 ( \mathrm { a } )$ , it follows that $\partial f ^ { * } ( \mathbf { y } )$ is a singleton for any $\mathbf { y } \in \mathbb { E } ^ { * }$ . Therefore, by Theorem 3.33, $f ^ { * }$ is diferentiable over the entire dual space $\mathbb { E } ^ { * }$ . To show the $\begin{array} { r } { \frac { 1 } { \sigma } } \end{array}$ -smoothness of $f ^ { * }$ take $\mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } \in \mathbb { E } ^ { * }$ and denote $\mathbf { v } _ { 1 } = \nabla f ^ { * } ( \mathbf { y } _ { 1 } ) , \mathbf { v } _ { 2 } = \nabla f ^ { * } ( \mathbf { y } _ { 2 } )$ . These relations, by the conjugate subgradient theorem (Theorem 4.20), are equivalent to $\mathbf { y } _ { 1 } \in \partial f ( \mathbf { v } _ { 1 } ) , \mathbf { y } _ { 2 } \in$ $\partial f ( \mathbf { v } _ { 2 } )$ . Therefore, by Theorem 5.24 (equivalence between properties (i) and (iii)), 

$$
\left\langle \mathbf {y} _ {1} - \mathbf {y} _ {2}, \mathbf {v} _ {1} - \mathbf {v} _ {2} \right\rangle \geq \sigma \| \mathbf {v} _ {1} - \mathbf {v} _ {2} \| ^ {2},
$$

that is, 

$$
\left\langle \mathbf {y} _ {1} - \mathbf {y} _ {2}, \nabla f ^ {*} (\mathbf {y} _ {1}) - \nabla f ^ {*} (\mathbf {y} _ {2}) \right\rangle \geq \sigma \| \nabla f ^ {*} (\mathbf {y} _ {1}) - \nabla f ^ {*} (\mathbf {y} _ {2}) \| ^ {2},
$$

which, combined with the generalized Cauchy–Schwarz inequality (Lemma 1.4), implies the inequality 

$$
\left\| \nabla f ^ {*} (\mathbf {y} _ {1}) - \nabla f ^ {*} (\mathbf {y} _ {2}) \right\| \leq \frac {1}{\sigma} \left\| \mathbf {y} _ {1} - \mathbf {y} _ {2} \right\| _ {*},
$$

proving the <sup>1</sup> -smoothness of $f ^ { * } . \sqsubseteq$ σ 

## 5.3.2 Examples of Strongly Convex Functions

We can use the conjugate correspondence theorem (Theorem 5.26) to conclude several results on the strong convexity of several important functions. 

Example 5.27 (negative entropy over the unit simplex). Consider the function $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ given by 

$$
f (\mathbf {x}) = \left\{ \begin{array}{l l} \sum_ {i = 1} ^ {n} x _ {i} \log x _ {i}, & \mathbf {x} \in \Delta_ {n}, \\ \infty & \text { else }. \end{array} \right.
$$

Then, by Section 4.4.10, the conjugate of this function is the log-sum-exp function $\begin{array} { r } { f ^ { * } ( \mathbf { y } ) = \log \left( \sum _ { i = 1 } ^ { n } e ^ { y _ { i } } \right) } \end{array}$ , which, by Example 5.15, is a 1-smooth function w.r.t. both the $l _ { \infty } -$ and $l _ { 2 } { \mathrm { - n o r m s } }$ . Consequently, by the conjugate correspondence theorem, f is 1-strongly convex w.r.t. both the $l _ { 1 } \cdot$ and $l _ { 2 } .$ -norms. 

Example 5.28 (squared p-norm for $\pmb { \mathnormal { p } } \in \left( \mathbf { 1 } , \mathbf { 2 } \right] )$ . Consider the function $f$ : $\mathbb { R } ^ { n } \to \mathbb { R }$ given by $\begin{array} { r } { f ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { x } \| _ { p } ^ { 2 } ~ ( p \in ( 1 , 2 ] ) } \end{array}$ . Then, by Section 4.4.15, $f ^ { * } ( \mathbf { y } ) =$ $\begin{array} { r } { \frac { 1 } { 2 } \| \mathbf { y } \| _ { q } ^ { 2 } , } \end{array}$ , where $q \geq 2$ is determined by the relation $\textstyle { \frac { 1 } { p } } + { \frac { 1 } { q } } = 1$ . By Example 5.11, $f ^ { * }$ is a $\left( q - 1 \right)$ -smooth function w.r.t. the $l _ { q } { \mathrm { - n o r m } }$ , which, by the conjugate correspondence theorem, implies that the function $f \ \mathrm { i s } \ \frac { 1 } { q - 1 }$ -strongly convex w.r.t. the $l _ { p } { \mathrm { - n o r m } }$ . Since $\textstyle { \frac { 1 } { q - 1 } } = p - 1$ , we conclude that f is $( p - 1 )$ -strongly convex w.r.t. the $l _ { p } { \mathrm { - n o r m } }$ ■ 

Example 5.29 $( l _ { 2 }$ ball-pen function). Consider the ball-pen function $f : \mathbb { R } ^ { n } $ $( - \infty , \infty ]$ given by 

$$
f (\mathbf {x}) = \left\{ \begin{array}{l l} - \sqrt {1 - \| \mathbf {x} \| _ {2} ^ {2}}, & \| \mathbf {x} \| _ {2} \leq 1, \\ \infty & \text { else }. \end{array} \right.
$$

By Section 4.4.13, the conjugate of $f$ is 

$$
f ^ {*} (\mathbf {y}) = \sqrt {\| \mathbf {y} \| _ {2} ^ {2} + 1},
$$

which, by Example 5.14, is known to be 1-smooth w.r.t. the $l _ { 2 } \mathrm { - n o r m }$ , and hence, by the conjugate correspondence theorem, f is 1-strongly convex w.r.t. the $l _ { 2 ^ { - } }$ norm. 


The table below contains all the strongly convex functions described in this chapter.


<table><tr><td><eq>f(\mathbf{x})</eq></td><td><eq>\text{dom}(f)</eq></td><td>Strong convexity parameter</td><td>Norm</td><td>Reference</td></tr><tr><td><eq>\frac{1}{2}\mathbf{x}^{T}\mathbf{A}\mathbf{x} + 2\mathbf{b}^{T}\mathbf{x} + c</eq><eq>(\mathbf{A} \in \mathbb{S}_{++}^{n}, \mathbf{b} \in \mathbb{R}^{n}, c \in \mathbb{R})</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>\lambda_{\min}(\mathbf{A})</eq></td><td><eq>l_{2}</eq></td><td>Example 5.19</td></tr><tr><td><eq>\frac{1}{2}\|\mathbf{x}\|^{2} + \delta_{C}(\mathbf{x})</eq><eq>(\emptyset \neq C \subseteq \mathbb{E} \text{convex})</eq></td><td><eq>C</eq></td><td>1</td><td>Euclidean</td><td>Example 5.21</td></tr><tr><td><eq>-\sqrt{1 - \|\mathbf{x}\|_{2}^{2}}</eq></td><td><eq>B_{\|\cdot\|_{2}}[\mathbf{0}, 1]</eq></td><td>1</td><td><eq>l_{2}</eq></td><td>Example 5.29</td></tr><tr><td><eq>\frac{1}{2}\|\mathbf{x}\|_{p}^{2} (p \in (1,2])</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>p - 1</eq></td><td><eq>l_{p}</eq></td><td>Example 5.28</td></tr><tr><td><eq>\sum_{i=1}^{n} x_{i} \log x_{i}</eq></td><td><eq>\Delta_{n}</eq></td><td>1</td><td><eq>l_{2} \text{ or } l_{1}</eq></td><td>Example 5.27</td></tr></table>

## 5.3.3 Smoothness and Diferentiability of the Infimal Convolution

We will now show that under appropriate conditions, the infimal convolution of a convex function and an L-smooth convex function is also L-smooth; in addition, we will derive an expression for the gradient. The proof of the result is based on the conjugate correspondence theorem. 

Theorem 5.30 (smoothness of the infimal convolution). Let $f : \mathbb { E } $ $( - \infty , \infty ]$ be a proper closed and convex function, and let $\omega : \mathbb { E } $ R be an L-smooth convex function. Assume that f-ω is real-valued. Then the following hold: 

(a) $f \boxed { \begin{array} { r l } \end{array} }$ is L-smooth. 

(b) Let $\mathbf { x } \in \mathbb { E }$ , and assume that $\mathbf { u } ( \mathbf { x } )$ is a minimizer of 

$$
\min _ {\mathbf {u}} \left\{f (\mathbf {u}) + \omega (\mathbf {x} - \mathbf {u}) \right\}.\tag{5.25}
$$

$$
\text { Then } \nabla (f \square \omega) (\mathbf {x}) = \nabla \omega (\mathbf {x} - \mathbf {u} (\mathbf {x})).
$$

Proof. (a) By Theorem 4.19, 

$$
f \square \omega = (f ^ {*} + \omega^ {*}) ^ {*}.
$$

Since f and $\omega$ are proper closed and convex, then so are $f ^ { * } , \omega ^ { * }$ (Theorems $4 . 3 ,$ 4.5). In addition, by the conjugate correspondence theorem (Theorem 5.26), $\omega ^ { * }$ is $\scriptstyle { \frac { 1 } { L } } \to \operatorname { s t r o n g l y }$ convex. Therefore, by Lemma 5.20, $f ^ { * } + \omega ^ { * }$ is $\scriptstyle { \frac { 1 } { L } } \to \operatorname { s t r o n g l y }$ convex, and it is also closed as a sum of closed functions; we will prove that it is also proper. Indeed, by Theorem 4.16, 

$$
(f \square \omega) ^ {*} = f ^ {*} + \omega^ {*}.
$$

Since $f \boxed { \begin{array} { r l } \end{array} }$ is convex (by Theorem 2.19) and proper, it follows that $f ^ { * } + \omega ^ { * }$ is proper as a conjugate of a proper and convex function (Theorem 4.5). Thus, since $f ^ { * } + \omega ^ { * }$ is proper closed and $\scriptstyle { \frac { 1 } { L } } \to \operatorname { s t r o n g l y }$ convex function, by the conjugate correspondence theorem, it follows that $f \sqcup \omega = ( f ^ { * } + \omega ^ { * } ) ^ { * }$ is L-smooth. 

(b) Let $\mathbf { x } \in \mathbb { E }$ be such that $\mathbf { u } ( \mathbf { x } )$ is a minimizer of (5.25), namely, 

$$
(f \square \omega) (\mathbf {x}) = f (\mathbf {u} (\mathbf {x})) + \omega (\mathbf {x} - \mathbf {u} (\mathbf {x})).\tag{5.26}
$$

For convenience, define ${ \bf z } \equiv \nabla \omega ( { \bf x } - { \bf u } ( { \bf x } ) )$ . Our objective is to show that $\nabla ( f \boxed { \mathbf { \xi } } \omega ) ( \mathbf { x } )$ $\ l = \textbf { z }$ . This means that we have to show that for any $\pmb { \xi } \in \mathbb { E } , \operatorname* { l i m } _ { \| \pmb { \xi } \|  0 } | \phi ( \pmb { \xi } ) | / \| \pmb { \xi } \| = 0$ where $\phi ( \pmb { \xi } ) \equiv ( f \sqcup \omega ) ( \mathbf { x } + \pmb { \xi } ) - ( f \sqcup \omega ) ( \mathbf { x } ) - \langle \pmb { \xi } , \mathbf { z } \rangle$ . By the definition of the infimal convolution, 

$$
(f \square \omega) (\mathbf {x} + \boldsymbol {\xi}) \leq f (\mathbf {u} (\mathbf {x})) + \omega (\mathbf {x} + \boldsymbol {\xi} - \mathbf {u} (\mathbf {x})),\tag{5.27}
$$

which, combined with (5.26), yields 

$$
\begin{array}{l l} \phi (\boldsymbol {\xi}) = (f \square \omega) (\mathbf {x} + \boldsymbol {\xi}) - (f \square \omega) (\mathbf {x}) - \langle \boldsymbol {\xi}, \mathbf {z} \rangle \\ \leq \omega (\mathbf {x} + \boldsymbol {\xi} - \mathbf {u} (\mathbf {x})) - \omega (\mathbf {x} - \mathbf {u} (\mathbf {x})) - \langle \boldsymbol {\xi}, \mathbf {z} \rangle & [ (5. 2 6), (5. 2 7) ] \\ \leq \langle \boldsymbol {\xi}, \nabla \omega (\mathbf {x} + \boldsymbol {\xi} - \mathbf {u} (\mathbf {x})) \rangle - \langle \boldsymbol {\xi}, \mathbf {z} \rangle , & [ \text {gradient inequality for} \omega ] \\ = \langle \boldsymbol {\xi}, \nabla \omega (\mathbf {x} + \boldsymbol {\xi} - \mathbf {u} (\mathbf {x})) - \nabla \omega (\mathbf {x} - \mathbf {u} (\mathbf {x})) \rangle & [ \text {substitution of} \mathbf {z} ] \\ \leq \| \boldsymbol {\xi} \| \cdot \| \nabla \omega (\mathbf {x} + \boldsymbol {\xi} - \mathbf {u} (\mathbf {x})) - \nabla \omega (\mathbf {x} - \mathbf {u} (\mathbf {x})) \| _ {*} & [ \text {generalized Cauchy - Schwarz} ] \\ \leq L \| \boldsymbol {\xi} \| ^ {2}. & [ L \text {-smoothness of} \omega ] \end{array}
$$

To complete the proof, it is enough to show that we also have $\phi ( \pmb { \xi } ) \geq - L \| \pmb { \xi } \| ^ { 2 }$ Since $f \boxed { \begin{array} { r l } \end{array} }$ is convex, so is $\phi ,$ which, along the fact that $\phi ( \mathbf { 0 } ) = 0$ , implies that $\phi ( \pmb { \xi } ) \geq - \phi ( - \pmb { \xi } )$ , and hence the desired result follows. 

Example 5.31 (revisiting the 1-smoothness of $\scriptstyle { \frac { 1 } { 2 } } d _ { C } ^ { 2 } )$ . Suppose that E is a Euclidean space and let $C \subseteq \mathbb { E }$ be a nonempty closed and convex set. Consider the function $\begin{array} { r } { \varphi _ { C } ( \mathbf { x } ) = \frac { 1 } { 2 } d _ { C } ^ { 2 } ( \mathbf { x } ) } \end{array}$ . We have already shown in Example 5.5 that it is 1-smooth. We will provide here a second proof for this result, which is based on Theorem 5.30. Note that $\varphi _ { C } = \delta _ { C } \bigtriangledown h$ , where $\begin{array} { r } { h ( \mathbf { x } ) = \frac { 1 } { 2 } \| \mathbf { x } \| ^ { 2 } } \end{array}$ . Since $h$ is a real-valued 1-smooth convex function, and since $\delta _ { C }$ is proper closed and convex, it follows by Theorem 5.30 that $\varphi _ { C }$ is 1-smooth. 

# Chapter 6 The Proximal Operator

Underlying Space: In this chapter E is a Euclidean space, meaning a finite dimensional space endowed with an inner product $\langle \cdot , \cdot \rangle$ and the Euclidean norm $\| \cdot \| = \sqrt { \langle \cdot , \cdot \rangle }$ 

This chapter is devoted to the study of the proximal mapping, which will be fundamental in many of the algorithms that will be explored later in the book. The operator and its properties were first studied by Moreau, and hence it is also referred to as “Moreau’s proximal mapping.” 

## 6.1 Definition, Existence, and Uniqueness

Definition 6.1 (proximal mapping). Given a function $f : \mathbb { E } \to ( - \infty , \infty ]$ , the proximal mapping of f is the operator given by 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \operatorname * {a r g m i n} _ {\mathbf {u} \in \mathbb {E}} \left\{f (\mathbf {u}) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\} f o r a n y \mathbf {x} \in \mathbb {E}.
$$

We will often use the term “prox” instead of “proximal.” The mapping prox<sub>f</sub> takes a vector $\textbf { x } \in \ \mathbb { E }$ and maps it into a subset of E, which might be empty, a singleton, or a set with multiple vectors as the following example illustrates. 

Example 6.2. Consider the following three functions from R to R: 

$$
\begin{array}{l} g _ {1} (x) \equiv 0, \\ g _ {2} (x) = \left\{ \begin{array}{l l} 0, & x \neq 0, \\ - \lambda , & x = 0, \end{array} \right. \end{array}
$$

$$
g _ {3} (x) = \left\{ \begin{array}{l l} 0, & x \neq 0, \\ \lambda , & x = 0, \end{array} \right.
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/06afcb51-43be-4db1-8b62-226fc4277ca3/919ebdb56e86878fe1648d379f467dad13d32f1b52e0b27570e7fd12c3314026.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/06afcb51-43be-4db1-8b62-226fc4277ca3/459305990c78f854a22c33d12f692617e88fa39b2d0a18ede46328fcc6c1bbe1.jpg)



Figure 6.1. The left and right images are the plots of the functions $g _ { 2 }$ and g<sub>3</sub>, respectively, with $\lambda = 0 . 5$ from Example 6.2.


where $\lambda > 0$ is a given constant. The plots of the noncontinuous functions $g _ { 2 }$ and g<sub>3</sub> are given in Figure 6.1. The prox of $g _ { 1 }$ can computed as follows: 

$$
\operatorname{prox} _ {g _ {1}} (x) = \operatorname{argmin} _ {u \in \mathbb {R}} \left\{g _ {1} (u) + \frac {1}{2} (u - x) ^ {2} \right\} = \operatorname{argmin} _ {u \in \mathbb {R}} \left\{\frac {1}{2} (u - x) ^ {2} \right\} = \{x \}.
$$

To compute the prox of $g _ { 2 }$ , note that $\begin{array} { r } { \operatorname { p r o x } _ { g _ { 2 } } ( x ) = \operatorname { a r g m i n } _ { u \in \mathbb { R } } \tilde { g } _ { 2 } ( u , x ) } \end{array}$ , where 

$$
\tilde {g} _ {2} (u, x) \equiv g _ {2} (u) + \frac {1}{2} (u - x) ^ {2} = \left\{ \begin{array}{l l} - \lambda + \frac {x ^ {2}}{2}, & u = 0, \\ \frac {1}{2} (u - x) ^ {2}, & u \neq 0. \end{array} \right.
$$

For $x \neq 0$ , the minimum of ${ \scriptstyle { \frac { 1 } { 2 } } } ( u - x ) ^ { 2 }$ over $\mathbb { R } \backslash \{ 0 \}$ is attained at $u = x ( \neq 0 )$ with a minimal value of 0. Therefore, in this case, if $\textstyle 0 > - \lambda + { \frac { x ^ { 2 } } { 2 } }$ , then the unique minimizer of $\tilde { g } _ { 2 } ( \cdot , x )$ is $u = 0$ , and if $\textstyle 0 < - \lambda + { \frac { x ^ { 2 } } { 2 } }$ , then $u \ = \ x$ is the unique minimizer of $\tilde { g } _ { 2 } ( \cdot , x )$ ; finally, if $\textstyle { 0 = - \lambda + { \frac { x ^ { 2 } } { 2 } } }$ , then 0 and x are the two minimizers $\tilde { g } _ { 2 } ( \cdot , x )$ . When $x = 0$ , the minimizer of $\tilde { g } _ { 2 } ( \bar { \cdot , } 0 )$ is $u = 0$ . To conclude, 

$$
\operatorname{prox} _ {g _ {2}} (x) = \left\{ \begin{array}{l l} \{0 \}, & | x | <   \sqrt {2 \lambda}, \\ \{x \}, & | x | > \sqrt {2 \lambda}, \\ \{0, x \}, & | x | = \sqrt {2 \lambda}. \end{array} \right.
$$

Similar arguments show that 

$$
\operatorname{prox} _ {g _ {3}} (x) = \left\{ \begin{array}{l l} \{x \}, & x \neq 0, \\ \emptyset , & x = 0. \end{array} \right.
$$

The next theorem, called the first prox theorem, states that if $f$ is proper closed and convex, then $\mathrm { p r o x } _ { f } ( \mathbf { x } )$ is always a singleton, meaning that the prox exists and is unique. This is the reason why in the last example only $g _ { 1 }$ , which was proper closed and convex, had a unique prox at any point. 

Theorem 6.3 (first prox theorem). Let $f : \mathbb { E }  ( - \infty , \infty ]$ be a proper closed and convex function. Then $\mathrm { p r o x } _ { f } ( \mathbf { x } )$ is a singleton for any $\mathbf { x } \in \mathbb { E }$ 

Proof. For any $\mathbf { x } \in \mathbb { E }$ 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \operatorname{argmin} _ {\mathbf {u} \in \mathbb {E}} \tilde {f} (\mathbf {u}, \mathbf {x}),\tag{6.1}
$$

where $\begin{array} { r } { \widetilde { f } ( \mathbf { u } , \mathbf { x } ) \equiv f ( \mathbf { u } ) + \frac { 1 } { 2 } \| \mathbf { u } - \mathbf { x } \| ^ { 2 } } \end{array}$ . The function $\tilde { f } ( \cdot , \mathbf { x } )$ is a closed and strongly convex function as a sum of the closed and strongly convex function $\textstyle { \frac { 1 } { 2 } } \| \cdot - \mathbf { x } \| ^ { 2 }$ and the closed and convex function $f$ (see Lemma 5.20 and Theorem $2 . 7 \mathrm { \tilde { ( b ) ) } }$ . The properness of $\tilde { f } ( \cdot , \mathbf { x } )$ immediately follows from the properness of $f .$ Therefore, by Theorem 5.25(a), there exists a unique minimizer to the problem in (6.1). 

When $f$ is proper closed and convex, the last result shows that prox ${ \bf \nabla } _ { f } ( { \bf x } )$ is a singleton for any $\mathbf { x } \in \mathbb { E }$ . In these cases, which will constitute the vast majority of cases that will be discussed in this chapter, we will treat prox as a singlevalued mapping from $\mathbb { E }$ to $\mathbb { E }$ , meaning that we will write prox $ \mathbf { \sigma } _ { f } ( \mathbf { x } ) \mathbf { \bar { \sigma } } = \mathbf { y }$ and not $\mathrm { p r o x } _ { f } ( \mathbf { x } ) = \{ \mathbf { y } \}$ 

If we relax the assumptions in the first prox theorem and only require closedness of the function, then it is possible to show under some coerciveness assumptions that $\mathrm { p r o x } _ { f } ( \mathbf { x } )$ is never an empty set. 

Theorem 6.4 (nonemptiness of the prox under closedness and coerciveness). Let $f : \mathbb { E }  ( - \infty , \infty ]$ be a proper closed function, and assume that the following condition is satisfied: 

$$
\text {   the   function   } \mathbf {u} \mapsto f (\mathbf {u}) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \text {   is   coercive   for   any   } \mathbf {x} \in \mathbb {E}.\tag{6.2}
$$

Then $\mathrm { p r o x } _ { f } ( \mathbf { x } )$ is nonempty for any $\mathbf { x } \in \mathbb { E }$ 

Proof. For any $\mathbf { x } \in \mathbb { E }$ , the proper function $\begin{array} { r } { h ( \mathbf { u } ) \equiv f ( \mathbf { u } ) + \frac { 1 } { 2 } \| \mathbf { u } - \mathbf { x } \| ^ { 2 } } \end{array}$ is closed as a sum of two closed functions. Since by the premise of the theorem it is also coercive, it follows by Theorem 2.14 (with $S = \mathbb { E } )$ that $\mathrm { p r o x } _ { f } ( \mathbf { x } )$ , which consists of the minimizers of $h ,$ , is nonempty. 

Example 6.2 actually gave an illustration of Theorem 6.4 since although both $g _ { 2 }$ and $g _ { 3 }$ satisfy the coercivity assumption (6.2), only $g _ { 2 }$ was closed, and thus the fact that $\operatorname { p r o x } _ { g _ { 3 } } ( x )$ was empty for a certain value of $x ,$ as opposed to $\operatorname { p r o x } _ { g _ { 2 } } ( x )$ which was never empty, is not surprising. 

## 6.2 First Set of Examples of Proximal Mappings

Equipped just with the definition of the proximal mapping, we will now compute the proximal mapping of several proper closed and convex functions. 

## 6.2.1 Constant

If $f \equiv c$ for some $c \in \mathbb { R }$ , then 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \operatorname{argmin} _ {\mathbf {u} \in \mathbb {E}} \left\{c + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\} = \mathbf {x}.
$$

Therefore, 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \mathbf {x}
$$

is the identity mapping. 

## 6.2.2 Afine

Let $f ( \mathbf { x } ) = \left. \mathbf { a } , \mathbf { x } \right. + b$ , where $\mathbf { a } \in \mathbb { E }$ and $b \in \mathbb { R }$ . Then 

$$
\begin{array}{r l} & {\mathrm{prox} _ {f} (\mathbf {x}) = \mathrm{argmin} _ {\mathbf {u} \in \mathbb {E}} \left\{\langle \mathbf {a}, \mathbf {u} \rangle + b + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\}} \\ & {\qquad = \mathrm{argmin} _ {\mathbf {u} \in \mathbb {E}} \left\{\langle \mathbf {a}, \mathbf {x} \rangle + b - \frac {1}{2} \| \mathbf {a} \| ^ {2} + \frac {1}{2} \| \mathbf {u} - (\mathbf {x} - \mathbf {a}) \| ^ {2} \right\}} \\ & {\qquad = \mathbf {x} - \mathbf {a}.} \end{array}
$$

Therefore, 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \mathbf {x} - \mathbf {a}
$$

is a translation mapping. 

## 6.2.3 Convex Quadratic

Let $f : \mathbb { R } ^ { n }  \mathbb { R }$ be given by $\begin{array} { r } { f ( \mathbf { x } ) = \frac { 1 } { 2 } \mathbf { x } ^ { T } \mathbf { A } \mathbf { x } + \mathbf { b } ^ { T } \mathbf { x } + c } \end{array}$ , where $\mathbf { A } \in \mathbb { S } _ { + } ^ { n } , \mathbf { b } \in \mathbb { R } ^ { n }$ , and $c \in \mathbb { R }$ . The vector prox ${ \bf \nabla } _ { f } ( { \bf x } )$ is the minimizer of the problem 

$$
\min _ {\mathbf {u} \in \mathbb {E}} \left\{\frac {1}{2} \mathbf {u} ^ {T} \mathbf {A u} + \mathbf {b} ^ {T} \mathbf {u} + c + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\}.
$$

The optimal solution of the last problem is attained when the gradient of the objective function vanishes: 

$$
\mathbf {A} \mathbf {u} + \mathbf {b} + \mathbf {u} - \mathbf {x} = \mathbf {0},
$$

that is, when 

$$
(\mathbf {A} + \mathbf {I}) \mathbf {u} = \mathbf {x} - \mathbf {b},
$$

and hence 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = (\mathbf {A} + \mathbf {I}) ^ {- 1} (\mathbf {x} - \mathbf {b}).
$$

## 6.2.4 One-Dimensional Examples

The following lemma contains several prox computations of one-dimensional functions. 

Lemma 6.5. The following are pairs of proper closed and convex functions and their prox mappings: 

$$
\begin{array}{l l} g _ {1} (x) = \left\{ \begin{array}{l l} \mu x, & x \geq 0, \\ \infty , & x <   0, \end{array} \right. & \operatorname{prox} _ {g _ {1}} (x) = [ x - \mu ] _ {+}, \\ g _ {2} (x) = \lambda | x |, & \operatorname{prox} _ {g _ {2}} (x) = [ | x | - \lambda ] _ {+} \mathrm{sgn} (x), \\ g _ {3} (x) = \left\{ \begin{array}{l l} \lambda x ^ {3}, & x \geq 0, \\ \infty , & x <   0, \end{array} \right. & \operatorname{prox} _ {g _ {3}} (x) = \frac {- 1 + \sqrt {1 + 1 2 \lambda [ x ] _ {+}}}{6 \lambda}, \\ g _ {4} (x) = \left\{ \begin{array}{l l} - \lambda \log x, & x > 0, \\ \infty , & x \leq 0, \end{array} \right. & \operatorname{prox} _ {g _ {4}} (x) = \frac {x + \sqrt {x ^ {2} + 4 \lambda}}{2}, \\ g _ {5} (x) = \delta_ {[ 0, \eta ] \cap \mathbb {R}} (x), & \operatorname{prox} _ {g _ {5}} (x) = \min \{\max \{x, 0 \}, \eta \}, \end{array}
$$

where $\lambda \in \mathbb { R } _ { + } , \eta \in [ 0 , \infty ]$ , and $\mu \in \mathbb { R }$ 

Proof. The proofs repeatedly use the following trivial arguments: (i) if $f ^ { \prime } ( u ) = 0$ for a convex function $f _ { ; }$ then u must be one of its minimizers; (ii) if a minimizer of a convex function exists and is not attained at any point of diferentiability, then it must be attained at a point of nondiferentiability. 

[prox of $g _ { 1 } ]$ By definition, prox<sub>g</sub> (x) is the minimizer of the function 

$$
f (u) = \left\{ \begin{array}{l l} \infty , & u <   0, \\ f _ {1} (u), & u \geq 0, \end{array} \right.
$$

where $\begin{array} { r } { f _ { 1 } ( u ) = \mu u + \frac { 1 } { 2 } ( u - x ) ^ { 2 } } \end{array}$ . First note that $f _ { 1 } ^ { \prime } ( u ) = 0$ if and only if $u = x - \mu$ . If $x > \mu ,$ , then $f ^ { \prime } ( x - \mu ) \bar { = } f _ { 1 } ^ { \prime } ( x - \mu ) = 0$ , implying that in this case $\mathrm { p r o x } _ { q _ { 1 } } ( x ) = x - \mu$ Otherwise, if $x \leq \mu$ , the minimizer of $f$ is not attained at a point of diferentiability, meaning that it has to be attained at 0, which is the only point of nondiferentiability in the domain of $f ,$ so that $\mathrm { p r o x } _ { g _ { 1 } } ( x ) = 0$ 

[prox of g<sub>2</sub>] $\operatorname { p r o x } _ { g _ { 2 } } ( x )$ is the minimizer of the function 

$$
h (u) = \left\{ \begin{array}{l l} h _ {1} (u) \equiv \lambda u + \frac {1}{2} (u - x) ^ {2}, & u > 0, \\ h _ {2} (u) \equiv - \lambda u + \frac {1}{2} (u - x) ^ {2}, & u \leq 0. \end{array} \right.
$$

If the minimizer is attained at $u > 0$ , then $0 = h _ { 1 } ^ { \prime } ( u ) = \lambda + u - x$ , meaning that $u = x - \lambda$ . Therefore, if $x > \lambda .$ , then pro $\mathrm { { x } } _ { q _ { 2 } } ( x ) = x - \lambda$ . The same argument shows that if $x < - \lambda$ , then $\mathrm { p r o x } _ { g _ { 2 } } ( x ) = x + \lambda$ λ. If | $| x | \le \lambda$ , then $\operatorname { p r o x } _ { g _ { 2 } } ( x )$ must be the only point of nondiferentiability of $h$ , namely, 0. 

[prox of $g _ { 3 } ] \ \mathrm { p r o x } _ { g _ { 3 } } ( x )$ is the minimizer of the function 

$$
s (u) = \left\{ \begin{array}{l l} \lambda u ^ {3} + \frac {1}{2} (u - x) ^ {2}, & u \geq 0, \\ \infty , & u <   0. \end{array} \right.
$$

If the minimizer is positive, then $\tilde { u } = \mathrm { p r o x } _ { g _ { 3 } } ( x )$ satisfies $s ^ { \prime } ( \tilde { u } ) = 0$ , that is, 

$$
3 \lambda \tilde {u} ^ {2} + \tilde {u} - x = 0.
$$

The above equation has a positive root if and only if $x > 0$ , and in this case the (unique) positive root is prox $\begin{array} { r } { \mathfrak { c } _ { g _ { 3 } } ( x ) = \tilde { u } = \frac { - 1 + \sqrt { 1 + 1 2 \lambda x } } { 6 \lambda } } \end{array}$ . If $x \leq 0$ , the minimizer of s is attained at the only point of nondiferentiability of s in its domain, that is, at 0. 

[prox of $g _ { 4 } ] \ \tilde { u } = \mathrm { p r o x } _ { g _ { 4 } } ( x )$ is a minimizer over $\mathbb { R } _ { + + }$ of 

$$
t (u) = - \lambda \log u + \frac {1}{2} (u - x) ^ {2},
$$

which is determined by the condition that the derivative vanishes: 

$$
- \frac {\lambda}{\tilde {u}} + (\tilde {u} - x) = 0,
$$

that is, 

$$
\tilde {u} ^ {2} - \tilde {u} x - \lambda = 0.
$$

Therefore (taking the positive root), 

$$
\mathrm{prox} _ {g _ {4}} (x) = \tilde {u} = \frac {x + \sqrt {x ^ {2} + 4 \lambda}}{2}.
$$

[prox of $g _ { 5 } ]$ We will first assume that $\eta < \infty$ . Note that $\tilde { u } = \mathrm { p r o x } _ { g _ { 5 } } ( x )$ is the minimizer of 

$$
w (u) = \frac {1}{2} (u - x) ^ {2}
$$

over $[ 0 , \eta ]$ . The minimizer of w over $\mathbb { R }$ is $u = x .$ Therefore, if $0 \leq x \leq \eta ,$ then $\tilde { u } = x$ . If $x < 0$ , then $w$ is increasing over $[ 0 , \eta ]$ , and hence $\tilde { u } = 0$ . Finally, if $x > \eta$ 2 then w is decreasing over $[ 0 , \eta ]$ , and thus $\tilde { u } = \eta$ . To conclude, 

$$
\operatorname{prox} _ {g _ {5}} (x) = \tilde {u} = \left\{ \begin{array}{l l} x, & 0 \leq x \leq \eta , \\ 0, & x <   0, \\ \eta , & x > \eta , \end{array} \right. = \min \{\max \{x, 0 \}, \eta \}.
$$

For $\eta = \infty , g _ { 5 } ( x ) = \delta _ { [ 0 , \infty ) } ( x )$ , and in this case, $g _ { 5 }$ is identical to $g _ { 1 }$ with $\mu = 0$ implying that $\mathrm { p r o x } _ { g _ { 5 } } ( x ) = [ x ] _ { + }$ , which can also be written as 

$$
\operatorname{prox} _ {g _ {5}} (x) = \min \{\max \{x, 0 \}, \infty \}. \quad \square
$$

## 6.3 Prox Calculus Rules

In this section we gather several important results on the calculus of proximal mappings. Note that some of the results do not require any convexity/closedness assumptions. 

Theorem 6.6 (prox of separable functions). Suppose that $f : \mathbb { E } _ { 1 } \times \mathbb { E } _ { 2 } \times \dots \times$ $\mathbb { E } _ { m } \to ( - \infty , \infty ]$ is given by 

$$
f \left(\mathbf {x} _ {1}, \mathbf {x} _ {2}, \dots , \mathbf {x} _ {m}\right) = \sum_ {i = 1} ^ {m} f _ {i} \left(\mathbf {x} _ {i}\right) \text {   for   any   } \mathbf {x} _ {i} \in \mathbb {E} _ {i}, \quad i = 1, 2, \dots , m.
$$

Then for any $\mathbf { x } _ { 1 } \in \mathbb { E } _ { 1 } , \mathbf { x } _ { 2 } \in \mathbb { E } _ { 2 } , \ldots , \mathbf { x } _ { m } \in \mathbb { E } _ { m }$ 

$$
\operatorname{prox} _ {f} (\mathbf {x} _ {1}, \mathbf {x} _ {2}, \dots , \mathbf {x} _ {m}) = \operatorname{prox} _ {f _ {1}} (\mathbf {x} _ {1}) \times \operatorname{prox} _ {f _ {2}} (\mathbf {x} _ {2}) \times \dots \times \operatorname{prox} _ {f _ {m}} (\mathbf {x} _ {m}).\tag{6.3}
$$

Proof. Formula (6.3) is a result of the following chain of equalities: 

$$
\begin{array}{l} \operatorname{prox} _ {f} (\mathbf {x} _ {1}, \mathbf {x} _ {2}, \ldots , \mathbf {x} _ {m}) = \operatorname * {a r g m i n} _ {\mathbf {y} _ {1}, \mathbf {y} _ {2}, \ldots , \mathbf {y} _ {m}} \sum_ {i = 1} ^ {m} \left[ \frac {1}{2} \| \mathbf {y} _ {i} - \mathbf {x} _ {i} \| ^ {2} + f _ {i} (\mathbf {y} _ {i}) \right] \\ = \prod_ {i = 1} ^ {m} \operatorname * {a r g m i n} _ {\mathbf {y} _ {i}} \left[ \frac {1}{2} \| \mathbf {y} _ {i} - \mathbf {x} _ {i} \| ^ {2} + f _ {i} (\mathbf {y} _ {i}) \right] \\ = \prod_ {i = 1} ^ {m} \operatorname{prox} _ {f _ {i}} (\mathbf {x} _ {i}). \quad \square \end{array}
$$

Remark 6.7. If $f : \mathbb { R } ^ { n }  \mathbb { R }$ is proper closed convex and separable, 

$$
f (\mathbf {x}) = \sum_ {i = 1} ^ {n} f _ {i} (x _ {i}),
$$

with $f _ { i }$ being proper closed and convex univariate functions, then the result of Theorem 6.6 can be rewritten as 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \left(\operatorname{prox} _ {f _ {i}} (x _ {i})\right) _ {i = 1} ^ {n}.
$$

Example 6.8 $\left( l _ { 1 } { \mathbf { - n o r m } } \right)$ . Suppose that $g : \mathbb { R } ^ { n }  \mathbb { R }$ is given by $g ( \mathbf { x } ) = \lambda \| \mathbf { x } \| _ { 1 }$ where $\lambda > 0$ . Then 

$$
g (\mathbf {x}) = \sum_ {i = 1} ^ {n} \varphi (x _ {i}),\tag{6.4}
$$

where $\varphi ( t ) = \lambda | t |$ . By Lemma 6.5 (computation of $\mathrm { p r o x } _ { g _ { 2 } } ) , \mathrm { p r o x } _ { \varphi } ( s ) = \mathcal { T } _ { \lambda } ( s )$ , where $\mathcal { T } _ { \lambda }$ is defined as 

$$
\mathcal {T} _ {\lambda} (y) = [ | y | - \lambda ] _ {+} \mathrm{sgn} (y) = \left\{ \begin{array}{l l} y - \lambda , & y \geq \lambda , \\ 0, & | y | <   \lambda , \\ y + \lambda , & y \leq - \lambda . \end{array} \right.
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/06afcb51-43be-4db1-8b62-226fc4277ca3/98c50bb1566f9af5cfc051042618239ff5360ebaa50cfdad69d4503499f21944.jpg)



Figure 6.2. The soft thresholding function $\mathcal { T } _ { 1 }$


The function $\mathcal { T } _ { \lambda }$ is called the soft thresholding function, and its description is given in Figure 6.2. 

By Theorem 6.6, 

$$
\operatorname{prox} _ {g} (\mathbf {x}) = \left(\mathcal {T} _ {\lambda} (x _ {j})\right) _ {j = 1} ^ {n}.
$$

We will expand the definition of the soft thresholding function for vectors by applying it componentwise, that is, for any $\mathbf { x } \in \mathbb { R } ^ { n }$ , 

$$
\mathcal {T} _ {\lambda} (\mathbf {x}) \equiv (\mathcal {T} _ {\lambda} (x _ {j})) _ {j = 1} ^ {n} = [ | \mathbf {x} | - \lambda \mathbf {e} ] _ {+} \odot \operatorname{sgn} (\mathbf {x}).
$$

In this notation, 

$$
\operatorname{prox} _ {g} (\mathbf {x}) = \mathcal {T} _ {\lambda} (\mathbf {x}).
$$

Example 6.9 (negative sum of logs). Let $g : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ be given by 

$$
g (\mathbf {x}) = \left\{ \begin{array}{l l} - \lambda \sum_ {j = 1} ^ {n} \log x _ {j}, & \mathbf {x} > \mathbf {0}, \\ \infty & \text { else }, \end{array} \right.
$$

where $\lambda > 0$ . Then $\begin{array} { r } { g ( \mathbf { x } ) = \sum _ { i = 1 } ^ { n } \varphi ( x _ { i } ) } \end{array}$ , where 

$$
\varphi (t) = \left\{ \begin{array}{l l} - \lambda \log t, & t > 0, \\ \infty , & t <   0. \end{array} \right.
$$

By Lemma 6.5 (computation of $\mathrm { p r o x } _ { g _ { 4 } } )$ , 

$$
\mathrm{prox} _ {\varphi} (s) = \frac {s + \sqrt {s ^ {2} + 4 \lambda}}{2}.
$$

Thus, by Theorem 6.6, 

$$
\operatorname{prox} _ {g} (\mathbf {x}) = (\operatorname{prox} _ {\varphi} (x _ {j})) _ {j = 1} ^ {n} = \left(\frac {x _ {j} + \sqrt {x _ {j} ^ {2} + 4 \lambda}}{2}\right) _ {j = 1} ^ {n}.
$$

Example 6.10 $\left( l _ { 0 } { \bf - n o r m } \right)$ . Let $f : \mathbb { R } ^ { n }  \mathbb { R }$ be given by $f ( \mathbf { x } ) = \lambda \| \mathbf { x } \| _ { 0 } ,$ where $\lambda > 0$ and $\| \mathbf { x } \| _ { 0 } = \# \{ i : x _ { i } \neq 0 \}$ is the $l _ { 0 } { \mathrm { - n o r m } }$ discussed in Example 2.11. For any $\mathbf { x } \in \mathbb { R } ^ { n }$ ， 

$$
f (\mathbf {x}) = \sum_ {i = 1} ^ {n} I (x _ {i}),
$$

where 

$$
I (t) = \left\{ \begin{array}{l l} \lambda , & t \neq 0, \\ 0, & t = 0. \end{array} \right.
$$

Note that $I ( \cdot ) = J ( \cdot ) + \lambda$ , where 

$$
J (t) = \left\{ \begin{array}{l l} 0, & t \neq 0, \\ - \lambda , & t = 0, \end{array} \right.
$$

and that by Example 6.2, 

$$
\operatorname{prox} _ {J} (s) = \left\{ \begin{array}{l l} \{0 \}, & | s | <   \sqrt {2 \lambda}, \\ \{s \}, & | s | > \sqrt {2 \lambda}, \\ \{0, s \}, & | s | = \sqrt {2 \lambda}. \end{array} \right.\tag{6.5}
$$

We can write the above as prox $\gimel ( s ) = \mathcal { H } _ { { \sqrt { 2 \lambda } } } ( s )$ , where ${ \mathcal { H } } _ { \alpha }$ is the so-called hard thresholding operator defined by 

$$
\mathcal {H} _ {\alpha} (s) \equiv \left\{ \begin{array}{l l} \{0 \}, & | s | <   \alpha , \\ \{s \}, & | s | > \alpha , \\ \{0, s \}, & | s | = \alpha . \end{array} \right.
$$

The operators $\mathrm { p r o x } _ { J }$ and $\mathrm { p r o x } _ { I }$ are the same since for any $s \in \mathbb { R }$ 

$$
\begin{array}{r l} & {\mathrm{prox} _ {I} (s) = \mathrm{argmin} _ {t} \left\{I (t) + \frac {1}{2} (t - s) ^ {2} \right\}} \\ & {\qquad = \mathrm{argmin} _ {t} \left\{J (t) + \lambda + \frac {1}{2} (t - s) ^ {2} \right\}} \\ & {\qquad = \mathrm{argmin} _ {t} \left\{J (t) + \frac {1}{2} (t - s) ^ {2} \right\}} \\ & {\qquad = \mathrm{prox} _ {J} (s).} \end{array}
$$

Thus, invoking Theorem 6.6, it follows that<sup>27</sup> 

$$
\operatorname{prox} _ {g} (\mathbf {x}) = \mathcal {H} _ {\sqrt {2 \lambda}} (x _ {1}) \times \mathcal {H} _ {\sqrt {2 \lambda}} (x _ {2}) \times \dots \times \mathcal {H} _ {\sqrt {2 \lambda}} (x _ {n}).
$$

Theorem 6.11 (scaling and translation). Let $g : \mathbb { E }  ( - \infty , \infty ]$ be a proper function. Let $\lambda \neq 0$ and $\mathbf { a } \in \mathbb { E }$ . Define $f ( \mathbf { x } ) = g ( \lambda \mathbf { x } + \mathbf { a } )$ . Then 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \frac {1}{\lambda} \left[ \operatorname{prox} _ {\lambda^ {2} g} (\lambda \mathbf {x} + \mathbf {a}) - \mathbf {a} \right].\tag{6.6}
$$

Proof. By definition of the prox, 

$$
\begin{array}{c} \operatorname{prox} _ {f} (\mathbf {x}) = \operatorname{argmin} _ {\mathbf {u}} \left\{f (\mathbf {u}) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\} \\ = \operatorname{argmin} _ {\mathbf {u}} \left\{g (\lambda \mathbf {u} + \mathbf {a}) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\}. \end{array}\tag{6.7}
$$

Making the change of variables 

$$
\mathbf {z} = \lambda \mathbf {u} + \mathbf {a},\tag{6.8}
$$

the objective function in the minimization problem (6.7) becomes 

$$
g (\mathbf {z}) + \frac {1}{2} \left\| \frac {1}{\lambda} (\mathbf {z} - \mathbf {a}) - \mathbf {x} \right\| ^ {2} = \frac {1}{\lambda^ {2}} \left[ \lambda^ {2} g (\mathbf {z}) + \frac {1}{2} \left\| \mathbf {z} - (\lambda \mathbf {x} + \mathbf {a}) \right\| ^ {2} \right].\tag{6.9}
$$

The minimizer of (6.9) is $\mathbf { z } = \mathrm { p r o x } _ { \lambda ^ { 2 } g } ( \lambda \mathbf { x } + \mathbf { a } )$ , and hence by (6.8), it follows that (6.6) holds. 

Theorem 6.12 (prox of $\lambda g ( \cdot / \lambda ) )$ . Let $g : \mathbb { E }  ( - \infty , \infty ]$ be proper, and let $\lambda \neq 0$ Define $f ( \mathbf { x } ) = \lambda g ( \mathbf { x } / \lambda )$ . Then 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \lambda \operatorname{prox} _ {g / \lambda} (\mathbf {x} / \lambda).
$$

Proof. Note that 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \operatorname{argmin} _ {\mathbf {u}} \left\{f (\mathbf {u}) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\} = \operatorname{argmin} _ {\mathbf {u}} \left\{\lambda g \left(\frac {\mathbf {u}}{\lambda}\right) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\}.
$$

Making the change of variables $\begin{array} { r } { { \bf z } = \frac { { \bf u } } { \lambda } } \end{array}$ , we can continue to write 

$$
\begin{array}{r l} & {\mathrm{prox} _ {f} (\mathbf {x}) = \lambda \mathrm{argmin} _ {\mathbf {z}} \left\{\lambda g (\mathbf {z}) + \frac {1}{2} \| \lambda \mathbf {z} - \mathbf {x} \| ^ {2} \right\}} \\ & {\qquad = \lambda \mathrm{argmin} _ {\mathbf {z}} \left\{\lambda^ {2} \left[ \frac {g (\mathbf {z})}{\lambda} + \frac {1}{2} \left\| \mathbf {z} - \frac {\mathbf {x}}{\lambda} \right\| ^ {2} \right] \right\}} \\ & {\qquad = \lambda \mathrm{argmin} _ {\mathbf {z}} \left\{\frac {g (\mathbf {z})}{\lambda} + \frac {1}{2} \left\| \mathbf {z} - \frac {\mathbf {x}}{\lambda} \right\| ^ {2} \right\}} \\ & {\qquad = \lambda \mathrm{prox} _ {g / \lambda} (\mathbf {x} / \lambda). \quad \square} \end{array}
$$

Theorem 6.13 (quadratic perturbation). Let $g : \mathbb { E }  ( - \infty , \infty ]$ be proper, and let $\begin{array} { r } { f ( \mathbf { x } ) = g ( \mathbf { x } ) + \frac { c } { 2 } \| \mathbf { x } \| ^ { 2 } + \langle \mathbf { a } , \mathbf { x } \rangle + \gamma } \end{array}$ , where $c > 0 , \mathbf { a } \in \mathbb { E }$ , and $\gamma \in \mathbb { R }$ . Then 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \operatorname{prox} _ {\frac {1}{c + 1} g} \left(\frac {\mathbf {x} - \mathbf {a}}{c + 1}\right).
$$

Proof. Follows by the following simple computation: 

$$
\begin{array}{l} \operatorname{prox} _ {f} (\mathbf {x}) = \operatorname{argmin} _ {\mathbf {u}} \left\{f (\mathbf {u}) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\} \\ \qquad = \operatorname{argmin} _ {\mathbf {u}} \left\{g (\mathbf {u}) + \frac {c}{2} \| \mathbf {u} \| ^ {2} + \langle \mathbf {a}, \mathbf {u} \rangle + \gamma + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\} \\ \qquad = \operatorname{argmin} _ {\mathbf {u}} \left\{g (\mathbf {u}) + \frac {c + 1}{2} \left\| \mathbf {u} - \left(\frac {\mathbf {x} - \mathbf {a}}{c + 1}\right) \right\| ^ {2} \right\} \\ \qquad = \operatorname{prox} _ {\frac {1}{c + 1} g} \left(\frac {\mathbf {x} - \mathbf {a}}{c + 1}\right). \quad \square \end{array}
$$

Example 6.14. Consider the function $f : \mathbb { R } \to ( - \infty , \infty ]$ given for any $x \in \mathbb { R }$ by 

$$
f (x) = \left\{ \begin{array}{l l} \mu x, & 0 \leq x \leq \alpha , \\ \infty & \text { else }, \end{array} \right.
$$

where $\mu \in \mathbb { R }$ and $\alpha \in [ 0 , \infty ]$ . To compute the prox of $f ,$ note first that $f$ can be represented as 

$$
f (x) = \delta_ {[ 0, \alpha ] \cap \mathbb {R}} (x) + \mu x.
$$

By Lemma 6.5 (computation of ${ \mathrm { p r o x } } _ { g _ { 5 } } ) , { \mathrm { p r o x } } _ { \delta _ { [ 0 , \alpha ] \cap \mathbb { R } } } ( x ) = { \mathrm { m i n } } \{ { \mathrm { m a x } } \{ x , 0 \} , \alpha \}$ . Therefore, using Theorem 6.13 with $c = 0 , \mathbf { a } = \mu , \dot { \gamma } = \mathbf { \dot { 0 } } .$ , we obtain that for any $x \in \mathbb { R }$ 

$$
\operatorname{prox} _ {f} (x) = \operatorname{prox} _ {g} (x - \mu) = \min \{\max \{x - \mu , 0 \}, \alpha \}.
$$

Unfortunately, there is no useful calculus rule for computing the prox mapping of a composition of a function with a general afine mapping. However, if the associated linear transformation satisfies a certain orthogonality condition, such a rule exists. 

Theorem 6.15 (composition with an afine mapping). Let $g : \mathbb { R } ^ { m } $ $( - \infty , \infty ]$ be a proper closed convex function, and let $f ( \mathbf { x } ) = g ( \mathcal { A } ( \mathbf { x } ) + \mathbf { b } )$ , where b $\in \mathbb { R } ^ { m }$ and $\mathcal { A } : \mathbb { V } \to \mathbb { R } ^ { m }$ is a linear transformation satisfying<sup>28</sup> A ◦ A<sup>T</sup> = αI for some constant $\alpha > 0$ . Then for any $\mathbf { x } \in \mathbb { V }$ 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \mathbf {x} + \frac {1}{\alpha} \mathcal {A} ^ {T} (\operatorname{prox} _ {\alpha g} (\mathcal {A} (\mathbf {x}) + \mathbf {b}) - \mathcal {A} (\mathbf {x}) - \mathbf {b}).
$$

Proof. By definition, $\mathrm { p r o x } _ { f } ( \mathbf { x } )$ is the optimal solution of 

$$
\min _ {\mathbf {u} \in \mathbb {V}} \left\{f (\mathbf {u}) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\},
$$

which can be rewritten as 

$$
\min _ {\mathbf {u} \in \mathbb {V}} \left\{g (\mathcal {A} (\mathbf {u}) + \mathbf {b}) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\}.
$$

The above problem can be formulated as the following constrained problem: 

$$
\begin{array}{l l} \min _ {\mathbf {u} \in \mathbb {V}, \mathbf {z} \in \mathbb {R} ^ {m}} & g (\mathbf {z}) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \\ \text {s.t.} & \mathbf {z} = \mathcal {A} (\mathbf {u}) + \mathbf {b}. \end{array}\tag{6.10}
$$

Denote the optimal solution of (6.10) by (˜z, u˜) (the existence and uniqueness of ˜z and u˜ follow by the underlying assumption that g is proper closed and convex). Note that $\tilde { \mathbf { u } } = \mathrm { p r o x } _ { f } ( \mathbf { x } )$ . Fixing $\mathbf { z } = \tilde { \mathbf { z } }$ , we obtain that u˜ is the optimal solution of 

$$
\begin{array}{l l} \min _ {\mathbf {u} \in \mathbb {V}} & \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \\ \text {s.t.} & \mathcal {A} (\mathbf {u}) = \tilde {\mathbf {z}} - \mathbf {b}. \end{array}\tag{6.11}
$$

Since strong duality holds for problem (6.11) (see Theorem A.1), by Theorem A.2, it follows that there exists $\mathbf { y } \in \mathbb { R } ^ { m }$ for which 

$$
\tilde {\mathbf {u}} \in \operatorname{argmin} _ {\mathbf {u} \in \mathbb {V}} \left\{\frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} + \langle \mathbf {y}, \mathcal {A} (\mathbf {u}) - \tilde {\mathbf {z}} + \mathbf {b} \rangle \right\}\tag{6.12}
$$

$$
\mathcal {A} (\tilde {\mathbf {u}}) = \tilde {\mathbf {z}} - \mathbf {b}.\tag{6.13}
$$

By (6.12), 

$$
\tilde {\mathbf {u}} = \mathbf {x} - \mathcal {A} ^ {T} (\mathbf {y}).\tag{6.14}
$$

Substituting this expression of u˜ into (6.13), we obtain 

$$
\mathcal {A} (\mathbf {x} - \mathcal {A} ^ {T} (\mathbf {y})) = \tilde {\mathbf {z}} - \mathbf {b},
$$

and hence, using the assumption that $\mathcal { A } \circ \mathcal { A } ^ { T } = \alpha \mathcal { Z } .$ 

$$
\alpha \mathbf {y} = \mathcal {A} (\mathbf {x}) + \mathbf {b} - \tilde {\mathbf {z}},
$$

which, combined with (6.14), yields an explicit expression for $\tilde { \mathbf { u } } = \mathrm { p r o x } _ { f } ( \mathbf { x } )$ in terms of $\tilde { \mathbf { z } } \mathrm { : }$ 

$$
\mathrm{prox} _ {f} (\mathbf {x}) = \tilde {\mathbf {u}} = \mathbf {x} + \frac {1}{\alpha} \mathcal {A} ^ {T} (\tilde {\mathbf {z}} - \mathcal {A} (\mathbf {x}) - \mathbf {b}).\tag{6.15}
$$

Substituting $\mathbf { u } = \tilde { \mathbf { u } }$ in the minimization problem (6.10), we obtain that $\tilde { \mathbf { z } }$ is given by 

$$
\begin{array}{l} \tilde {\mathbf {z}} = \operatorname * {a r g m i n} _ {\mathbf {z} \in \mathbb {R} ^ {m}} \left\{g (\mathbf {z}) + \frac {1}{2} \left\| \mathbf {x} + \frac {1}{\alpha} \mathcal {A} ^ {T} (\mathbf {z} - \mathcal {A} (\mathbf {x}) - \mathbf {b}) - \mathbf {x} \right\| ^ {2} \right\} \\ = \operatorname * {a r g m i n} _ {\mathbf {z} \in \mathbb {R} ^ {m}} \left\{g (\mathbf {z}) + \frac {1}{2 \alpha^ {2}} \| \mathcal {A} ^ {T} (\mathbf {z} - \mathcal {A} (\mathbf {x}) - \mathbf {b}) \| ^ {2} \right\} \\ \stackrel {{(*)}} {{=}} \operatorname * {a r g m i n} _ {\mathbf {z} \in \mathbb {R} ^ {m}} \left\{\alpha g (\mathbf {z}) + \frac {1}{2} \| \mathbf {z} - \mathcal {A} (\mathbf {x}) - \mathbf {b} \| ^ {2} \right\} \\ = \operatorname{prox} _ {\alpha g} (\mathcal {A} (\mathbf {x}) + \mathbf {b}), \end{array}
$$

where the equality (∗) uses the assumption that $\mathcal { A } \circ \mathcal { A } ^ { T } = \alpha \mathcal { T }$ . Plugging the expression for z˜ into (6.15) produces the desired result. 

Example 6.16. Let $g : \mathbb { E }  ( - \infty , \infty ]$ be proper closed and convex where $\mathbb { E } = \mathbb { R } ^ { d }$ 2 and let $f : \mathbb { E } ^ { m }  ( - \infty , \infty ]$ be defined as 

$$
f (\mathbf {x} _ {1}, \mathbf {x} _ {2}, \dots , \mathbf {x} _ {m}) = g (\mathbf {x} _ {1} + \mathbf {x} _ {2} + \dots + \mathbf {x} _ {m}).
$$

The above can be written as $f ( \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { m } ) = g ( { \cal A } ( \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { m } ) )$ , where $\mathcal { A }$ : $\mathbb { E } ^ { m } \to \mathbb { E }$ is the linear transformation 

$$
\mathcal {A} (\mathbf {x} _ {1}, \mathbf {x} _ {2}, \ldots , \mathbf {x} _ {m}) = \mathbf {x} _ {1} + \mathbf {x} _ {2} + \dots + \mathbf {x} _ {m}.
$$

Obviously, the adjoint operator $\mathcal { A } ^ { T } : \mathbb { E }  \mathbb { E } ^ { m }$ is given by 

$$
\mathcal {A} ^ {T} (\mathbf {x}) = (\mathbf {x}, \mathbf {x}, \dots , \mathbf {x}),
$$

and for any $\mathbf { x } \in \mathbb { E }$ 

$$
\mathcal {A} (\mathcal {A} ^ {T} (\mathbf {x})) = m \mathbf {x}.
$$

Thus, the conditions of Theorem 6.15 are satisfied with $\alpha = m$ and $\mathbf b = 0$ , and consequently, for any $\left( \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { m } \right) \in \mathbb { E } ^ { m }$ ， 

$$
\operatorname{prox} _ {f} (\mathbf {x} _ {1}, \mathbf {x} _ {2}, \dots , \mathbf {x} _ {m}) _ {j} = \mathbf {x} _ {j} + \frac {1}{m} \left(\operatorname{prox} _ {m g} \left(\sum_ {i = 1} ^ {m} \mathbf {x} _ {i}\right) - \sum_ {i = 1} ^ {m} \mathbf {x} _ {i}\right), \quad j = 1, 2, \dots , m.
$$

Example 6.17. Let $f : \mathbb { R } ^ { n }  \mathbb { R }$ be given by $f ( \mathbf { x } ) = | \mathbf { a } ^ { T } \mathbf { x } |$ , where $\mathbf { a } \in \mathbb { R } ^ { n } \setminus \{ \mathbf { 0 } \}$ We can write f as $f ( \mathbf { x } ) = g ( \mathbf { a } ^ { T } \mathbf { x } )$ , where $g ( t ) = | t |$ . By Lemma $6 . 5 \ \left( \mathrm { p r o x } _ { g _ { 2 } } \right.$ computation), $\mathrm { p r o x } _ { \lambda g } = \mathcal { T } _ { \lambda }$ , with $\mathcal { T } _ { \lambda } ( x ) = [ | x | - \lambda ] _ { + } \mathrm { s g n } ( x )$ being the soft thresholding operator defined in Example 6.8. Invoking Theorem 6.15 with $\alpha = \| \mathbf { a } \| ^ { 2 }$ , b = 0, and $\mathcal { A }$ defined as the transformation ${ \mathbf { x } } \mapsto { \mathbf { a } } ^ { T } { \mathbf { x } } .$ we obtain that 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \mathbf {x} + \frac {1}{\| \mathbf {a} \| ^ {2}} \left(\mathcal {T} _ {\| \mathbf {a} \| ^ {2}} \left(\mathbf {a} ^ {T} \mathbf {x}\right) - \mathbf {a} ^ {T} \mathbf {x}\right) \mathbf {a}.
$$

Theorem 6.18 (norm composition). Let $f : \mathbb { E } \to \mathbb { R }$ be given by $f ( \mathbf { x } ) = g ( \| \mathbf { x } \| )$ where $g : \mathbb { R }  ( - \infty , \infty ]$ is a proper closed and convex function satisfying dom $( g ) \subseteq$ $[ 0 , \infty )$ . Then 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \left\{ \begin{array}{l l} \operatorname{prox} _ {g} (\| \mathbf {x} \|) \frac {\mathbf {x}}{\| \mathbf {x} \|}, & \mathbf {x} \neq \mathbf {0}, \\ \{\mathbf {u} \in \mathbb {E}: \| \mathbf {u} \| = \operatorname{prox} _ {g} (0) \}, & \mathbf {x} = \mathbf {0}. \end{array} \right.\tag{6.16}
$$

Proof. By definition, $\mathrm { p r o x } _ { f } ( \mathbf { 0 } )$ is the set of minimizers of the problem 

$$
\min _ {\mathbf {u} \in \mathbb {E}} \left\{f (\mathbf {u}) + \frac {1}{2} \| \mathbf {u} \| ^ {2} \right\} = \min _ {\mathbf {u} \in \mathbb {E}} \left\{g (\| \mathbf {u} \|) + \frac {1}{2} \| \mathbf {u} \| ^ {2} \right\}.
$$

Making the change of variables $w = \left\| \mathbf { u } \right\|$ , the problem reduces to (recalling that dom $( g ) \subseteq [ 0 , \infty ) )$ 1 

$$
\min _ {w \in \mathbb {R}} \left\{g (w) + \frac {1}{2} w ^ {2} \right\}.
$$

The optimal set of the above problem is $\operatorname { p r o x } _ { g } ( 0 )$ , and hence $\mathrm { p r o x } _ { f } ( \mathbf { 0 } )$ is the set of vectors u satisfying $\| { \mathbf { u } } \| = \operatorname { p r o x } _ { g } ( 0 )$ . We will now compute $\mathrm { p r o x } _ { f } ( \mathbf { x } )$ for $\mathbf { x } \neq \mathbf { 0 }$ The optimization problem associated with the prox computation can be rewritten as the following double minimization problem: 

$$
\begin{array}{c} \min _ {\mathbf {u} \in \mathbb {E}} \left\{g (\| \mathbf {u} \|) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\} = \min _ {\mathbf {u} \in \mathbb {E}} \left\{g (\| \mathbf {u} \|) + \frac {1}{2} \| \mathbf {u} \| ^ {2} - \langle \mathbf {u}, \mathbf {x} \rangle + \frac {1}{2} \| \mathbf {x} \| ^ {2} \right\} \\ = \min _ {\alpha \in \mathbb {R} _ {+}} \min _ {\mathbf {u} \in \mathbb {E}: \| \mathbf {u} \| = \alpha} \left\{g (\alpha) + \frac {1}{2} \alpha^ {2} - \langle \mathbf {u}, \mathbf {x} \rangle + \frac {1}{2} \| \mathbf {x} \| ^ {2} \right\}. \end{array}
$$

Using the Cauchy–Schwarz inequality, it is easy to see that the minimizer of the inner minimization problem is 

$$
\mathbf {u} = \alpha \frac {\mathbf {x}}{\| \mathbf {x} \|},\tag{6.17}
$$

and the corresponding optimal value is 

$$
g (\alpha) + \frac {1}{2} \alpha^ {2} - \alpha \| \mathbf {x} \| + \frac {1}{2} \| \mathbf {x} \| ^ {2} = g (\alpha) + \frac {1}{2} (\alpha - \| \mathbf {x} \|) ^ {2}.
$$

Therefore, prox ${ \bf \nabla } _ { f } ( { \bf x } )$ is given by u in (6.17) with α given by 

$$
\begin{array}{l} \alpha = \operatorname * {a r g m i n} _ {\alpha \in \mathbb {R} _ {+}} \bigg \{g (\alpha) + \frac {1}{2} (\alpha - \| \mathbf {x} \|) ^ {2} \bigg \} \\ \qquad = \operatorname * {a r g m i n} _ {\alpha \in \mathbb {R}} \bigg \{g (\alpha) + \frac {1}{2} (\alpha - \| \mathbf {x} \|) ^ {2} \bigg \} \\ \qquad = \operatorname * {p r o x} _ {g} (\| \mathbf {x} \|), \end{array}
$$

where the second equality is due to the assumption that dom $( g ) \subseteq [ 0 , \infty )$ . Thus, $\begin{array} { r } { \mathrm { p r o x } _ { f } ( \mathbf { x } ) = \mathrm { p r o x } _ { g } ( \| \mathbf { x } \| ) \frac { \mathbf { x } } { \| \mathbf { x } \| } } \end{array}$ 

Example 6.19 (prox of Euclidean norm). Let $f : \mathbb { E } $ R be given by $f ( \mathbf { x } ) =$ $\lambda \| \mathbf { x } \|$ , where $\lambda > 0$ and $\| \cdot \|$ is the underlying Euclidean norm (recall that in this section we assume that the underlying space is Euclidean). Then $f ( \mathbf { x } ) = g ( \| \mathbf { x } \| )$ where 

$$
g (t) = \left\{ \begin{array}{l l} \lambda t, & t \geq 0, \\ \infty , & t <   0. \end{array} \right.
$$

Then by Theorem 6.18, for any $\mathbf { x } \in \mathbb { E }$ 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \left\{ \begin{array}{l l} \operatorname{prox} _ {g} (\| \mathbf {x} \|) \frac {\mathbf {x}}{\| \mathbf {x} \|}, & \mathbf {x} \neq \mathbf {0}, \\ \{\mathbf {u} \in \mathbb {E}: \| \mathbf {u} \| = \operatorname{prox} _ {g} (0) \}, & \mathbf {x} = \mathbf {0}. \end{array} \right.
$$

By Lemma 6.5 (computation of $\mathrm { p r o x } _ { g _ { 1 } } )$ $\operatorname { p r o x } _ { g } ( t ) = [ t - \lambda ] _ { + }$ . Thus, $\mathrm { p r o x } _ { g } ( 0 ) = 0$ and $\operatorname { p r o x } _ { g } ( \| \mathbf { x } \| ) = [ \| \mathbf { x } \| - \lambda ] _ { + }$ , and therefore 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \left\{ \begin{array}{l l} [ \| \mathbf {x} \| - \lambda ] _ {+} \frac {\mathbf {x}}{\| \mathbf {x} \|}, & \mathbf {x} \neq \mathbf {0}, \\ \mathbf {0}, & \mathbf {x} = \mathbf {0}. \end{array} \right.
$$

Finally, we can write the above formula in the following compact form: 

$$
\operatorname{prox} _ {\lambda \| \cdot \|} (\mathbf {x}) = \left(1 - \frac {\lambda}{\max \{\| \mathbf {x} \| , \lambda \}}\right) \mathbf {x}.
$$

Example 6.20 (prox of cubic Euclidean norm). Let $f ( \mathbf { x } ) = \lambda \| \mathbf { x } \| ^ { 3 }$ , where $\lambda > 0$ . Then $f ( \mathbf { x } ) = \lambda g ( \| \mathbf { x } \| )$ , where 

$$
g (t) = \left\{ \begin{array}{l l} t ^ {3}, & t \geq 0, \\ \infty , & t <   0. \end{array} \right.
$$

Then by Theorem 6.18, for any $\mathbf { x } \in \mathbb { R }$ 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \left\{ \begin{array}{l l} \operatorname{prox} _ {g} (\| \mathbf {x} \|) \frac {\mathbf {x}}{\| \mathbf {x} \|}, & \mathbf {x} \neq \mathbf {0}, \\ \{\mathbf {u} \in \mathbb {E}: \| \mathbf {u} \| = \operatorname{prox} _ {g} (0) \}, & \mathbf {x} = \mathbf {0}. \end{array} \right.
$$

By Lemma 6.5 (computation of $\operatorname { p r o x } _ { g _ { 3 } } )$ $\begin{array} { r } { \operatorname { p r o x } _ { g } ( t ) \ = \ \frac { - 1 + \sqrt { 1 + 1 2 \lambda [ t ] } + } { 6 \lambda } } \end{array}$ . Therefore, $\mathrm { p r o x } _ { g } ( 0 ) = 0$ and 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \left\{ \begin{array}{l l} \frac {- 1 + \sqrt {1 + 1 2 \lambda \| \mathbf {x} \|}}{6 \lambda} \frac {\mathbf {x}}{\| \mathbf {x} \|}, & \mathbf {x} \neq \mathbf {0}, \\ \mathbf {0}, & \mathbf {x} = \mathbf {0}, \end{array} \right.
$$

and thus 

$$
\mathrm{prox} _ {\lambda \| \cdot \| ^ {3}} (\mathbf {x}) = \frac {2}{1 + \sqrt {1 + 1 2 \lambda \| \mathbf {x} \|}} \mathbf {x}.
$$

Example 6.21 (prox of negative Euclidean norm). Let $f : \mathbb { E } \to \mathbb { R }$ be given by $f ( \mathbf { x } ) = - \lambda \| \mathbf { x } \|$ , where $\lambda > 0$ . Since $f$ is not convex, we do not expect the prox to be a single-valued mapping. However, since $f$ is closed, and since the function $\begin{array} { r } { \mathbf { u } \mapsto f ( \mathbf { u } ) + \frac { 1 } { 2 } \Vert \mathbf { u } - \mathbf { x } \Vert ^ { 2 } } \end{array}$ is coercive for any $\mathbf { x } \in \mathbb { E }$ , it follows by Theorem 6.4 that the set $\mathrm { p r o x } _ { f } ( \mathbf { x } )$ is always nonempty. To compute the prox, note that $f ( \mathbf { x } ) = g ( \| \mathbf { x } \| )$ ， where 

$$
g (t) = \left\{ \begin{array}{l l} - \lambda t, & t \geq 0, \\ \infty , & t <   0. \end{array} \right.
$$

By Theorem 6.18, for any $\mathbf { x } \in \mathbb { R }$ 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \left\{ \begin{array}{l l} \operatorname{prox} _ {g} (\| \mathbf {x} \|) \frac {\mathbf {x}}{\| \mathbf {x} \|}, & \mathbf {x} \neq \mathbf {0}, \\ \{\mathbf {u} \in \mathbb {E}: \| \mathbf {u} \| = \operatorname{prox} _ {g} (0) \}, & \mathbf {x} = \mathbf {0}. \end{array} \right.
$$

By Lemma 6.5 (computation of $\mathrm { p r o x } _ { g _ { 1 } } )$ , $\mathrm { p r o x } _ { g } ( t ) = [ t + \lambda ] _ { + }$ . Therefore, $\mathrm { p r o x } _ { g } ( 0 ) = \lambda$ and 

$$
\operatorname{prox} _ {- \lambda \| \cdot \|} (\mathbf {x}) = \left\{ \begin{array}{l l} \Big (1 + \frac {\lambda}{\| \mathbf {x} \|} \Big) \mathbf {x}, & \mathbf {x} \neq \mathbf {0}, \\ \{\mathbf {u}: \| \mathbf {u} \| = \lambda \}, & \mathbf {x} = \mathbf {0}. \end{array} \right.
$$

Example 6.22 (prox of absolute value over symmetric intervals). Consider the function $f : \mathbb { R } \to ( - \infty , \infty ]$ given by 

$$
f (x) = \left\{ \begin{array}{l l} \lambda | x |, & | x | \leq \alpha , \\ \infty & \text { else }, \end{array} \right.
$$

where $\lambda \in [ 0 , \infty )$ and $\alpha \in [ 0 , \infty ]$ . Then $f ( x ) = g ( | x | )$ , where 

$$
g (x) = \left\{ \begin{array}{l l} \lambda x, & 0 \leq x \leq \alpha , \\ \infty & \text { else }. \end{array} \right.
$$

Thus, by Theorem 6.18, for any $x ,$ 

$$
\operatorname{prox} _ {f} (\mathbf {x}) = \left\{ \begin{array}{l l} \operatorname{prox} _ {g} (| x |) \frac {x}{| x |}, & x \neq 0, \\ \{u \in \mathbb {R}: | u | = \operatorname{prox} _ {g} (0) \}, & x = 0. \end{array} \right.\tag{6.18}
$$

By Example 6.14, $\operatorname { p r o x } _ { q } ( x ) = \operatorname* { m i n } \{ \operatorname* { m a x } \{ x - \lambda , 0 \} , \alpha \}$ , which, combined with (6.18) and the fact that $\begin{array} { r } { \frac { x } { | x | } = \mathrm { s g n } ( x ) } \end{array}$ for any $x \neq 0 ,$ yields the formula 

$$
\operatorname{prox} _ {\lambda | \cdot | + \delta_ {[ - \alpha , \alpha ]}} (x) = \min \{\max \{| x | - \lambda , 0 \}, \alpha \} \mathrm{sgn} (x).
$$

Using the previous example, we can compute the prox of weighted $l _ { 1 } { \mathrm { - n o r m s } }$ over boxes. 

Example 6.23 (prox of weighted $\boldsymbol { l } _ { 1 }$ over a box). Consider the function $f$ : $\mathbb { R } ^ { n } \to \mathbb { R }$ given by 

$$
f (\mathbf {x}) = \left\{ \begin{array}{l l} \sum_ {i = 1} ^ {n} \omega_ {i} | x _ {i} |, & - \boldsymbol {\alpha} \leq \mathbf {x} \leq \boldsymbol {\alpha}, \\ \infty , & \text { else }, \end{array} \right.
$$

for any $\mathbf { x } \in \mathbb { R } ^ { n }$ , where $\boldsymbol \omega \in \mathbb { R } _ { + } ^ { n }$ and $\alpha \in [ 0 , \infty ] ^ { n }$ . Then $\textstyle f = \sum _ { i = 1 } ^ { n } f _ { i }$ , where 

$$
f _ {i} (x) = \left\{ \begin{array}{l l} w _ {i} | x |, & - \alpha_ {i} \leq x \leq \alpha_ {i}, \\ \infty , & \text { else }. \end{array} \right.
$$


Using Example 6.22 and invoking Theorem 6.6, we finally obtain that


<table><tr><td>$ \text{prox}_f(\mathbf{x}) = (\min\{\max\{|x_i| - \omega_i, 0\}, \alpha_i\}\text{sgn}(x_i))_{i=1}^n $.</td></tr></table>


The table below summarizes the main prox calculus rules discussed in this section.


<table><tr><td><eq>f(\mathbf{x})</eq></td><td><eq>\text{prox}_{f}(\mathbf{x})</eq></td><td>Assumptions</td><td>Reference</td></tr><tr><td><eq>\sum_{i=1}^{m} f_{i}(\mathbf{x}_{i})</eq></td><td><eq>\text{prox}_{f_{1}}(\mathbf{x}_{1}) \times \cdots \times \text{prox}_{f_{m}}(\mathbf{x}_{m})</eq></td><td></td><td>Theorem 6.6</td></tr><tr><td><eq>g(\lambda\mathbf{x} + \mathbf{a})</eq></td><td><eq>\frac{1}{\lambda} \left[ \text{prox}_{\lambda^{2}g}(\lambda\mathbf{x} + \mathbf{a}) - \mathbf{a} \right]</eq></td><td><eq>\lambda \neq 0, \mathbf{a} \in \mathbb{E}, g \text{ proper}</eq></td><td>Theorem 6.11</td></tr><tr><td><eq>\lambda g(\mathbf{x}/\lambda)</eq></td><td><eq>\lambda \text{prox}_{g/\lambda}(\mathbf{x}/\lambda)</eq></td><td><eq>\lambda \neq 0, g \text{ proper}</eq></td><td>Theorem 6.12</td></tr><tr><td><eq>g(\mathbf{x}) + \frac{c}{2} \| \mathbf{x} \|^{2} + \langle \mathbf{a}, \mathbf{x} \rangle + \gamma</eq></td><td><eq>\text{prox}_{\frac{1}{c+1}g}(\frac{\mathbf{x}-\mathbf{a}}{c+1})</eq></td><td><eq>\mathbf{a} \in \mathbb{E}, c &gt; 0, \gamma \in \mathbb{R}, g \text{ proper}</eq></td><td>Theorem 6.13</td></tr><tr><td><eq>g(\mathcal{A}(\mathbf{x}) + \mathbf{b})</eq></td><td><eq>\mathbf{x} + \frac{1}{\alpha} \mathcal{A}^{T}(\text{prox}_{\alpha g}(\mathcal{A}(\mathbf{x}) + \mathbf{b}) - \mathcal{A}(\mathbf{x}) - \mathbf{b})</eq></td><td><eq>\mathbf{b} \in \mathbb{R}^{m}, \mathcal{A}: \mathbb{V} \to \mathbb{R}^{m}, g \text{ proper closed convex, } \mathcal{A} \circ \mathcal{A}^{T} = \alpha I, \alpha &gt; 0</eq></td><td>Theorem 6.15</td></tr><tr><td><eq>g(\| \mathbf{x}\|)</eq></td><td><eq>\text{prox}_{g}(\| \mathbf{x}\|) \frac{\mathbf{x}}{\|\mathbf{x}\|}, \quad \mathbf{x} \neq \mathbf{0}</eq><eq>\{\mathbf{u}: \| \mathbf{u}\| = \text{prox}_{g}(0)\}, \quad \mathbf{x} = \mathbf{0}</eq></td><td><eq>g \text{ proper closed convex, dom}(g) \subseteq [0,\infty)</eq></td><td>Theorem 6.18</td></tr></table>

## 6.4 Prox of Indicators—Orthogonal Projections

## 6.4.1 The First Projection Theorem

Let $g : \mathbb { E }  ( - \infty , \infty ]$ be given by $g ( \mathbf { x } ) = \delta _ { C } ( \mathbf { x } )$ , where C is a nonempty set. Then 

$$
\operatorname{prox} _ {g} (\mathbf {x}) = \operatorname * {a r g m i n} _ {\mathbf {u} \in \mathbb {E}} \left\{\delta_ {C} (\mathbf {u}) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\} = \operatorname * {a r g m i n} _ {\mathbf {u} \in C} \| \mathbf {u} - \mathbf {x} \| ^ {2} = P _ {C} (\mathbf {x}).
$$

Thus, the proximal mapping of the indicator function of a given set is the orthogonal projection<sup>29</sup> operator onto the same set. 

Theorem 6.24. Let $C \subseteq \mathbb { E }$ be nonempty. Then $\mathrm { p r o x } _ { \delta _ { C } } ( \mathbf { x } ) = P _ { C } ( \mathbf { x } )$ for any $\mathbf { x } \in \mathbb { E }$ 

If C is closed and convex, in addition to being nonempty, the indicator function $\delta _ { C }$ is proper closed and convex, and hence by the first prox theorem (Theorem 6.3), the orthogonal projection mapping (which coincides with the proximal mapping) exists and is unique. This is the first projection theorem. 

Theorem 6.25 (first projection theorem). Let $C \subseteq \mathbb { E }$ be a nonempty closed convex set. Then $P _ { C } ( \mathbf { x } )$ is a singleton for any $\mathbf { x } \in \mathbb { E }$ 

## 6.4.2 First Examples in $\mathbb { R } ^ { n }$

We begin by $\mathrm { r e c a l l i n g ^ { 3 0 } }$ several known expressions for the orthogonal projection onto some basic subsets of $\mathbb { R } ^ { n }$ . Since the assumption made throughout the book is that (unless otherwise stated) $\mathbb { R } ^ { n }$ is endowed with the dot product, and since the standing assumption in this chapter is that the underlying space is Euclidean, it follows that the endowed norm is the $l _ { 2 } \mathrm { - n o r m }$ 

Lemma 6.26 (projection onto subsets of $\mathbb { R } ^ { n } )$ . Following are pairs of nonempty closed and convex sets and their corresponding orthogonal projections: 

nonnegative orthant C<sub>1</sub> = R<sup>n</sup><sub>+</sub>, 

box 

$$
C _ {2} = \operatorname{Box} [ \boldsymbol {\ell}, \mathbf {u} ], \quad (\min \{\max \{x _ {i}, \ell_ {i} \}, u _ {i} \}) _ {i = 1} ^ {n},
$$

afine set 

$$
C _ {3} = \{\mathbf {x} \in \mathbb {R} ^ {n}: \mathbf {A x} = \mathbf {b} \}, \quad \mathbf {x} - \mathbf {A} ^ {T} (\mathbf {A A} ^ {T}) ^ {- 1} (\mathbf {A x} - \mathbf {b}),
$$

$l _ { 2 }$ ball 

$$
C _ {4} = B _ {\| \cdot \| _ {2}} [ \mathbf {c}, r ], \quad \mathbf {c} + \frac {r}{\max \left\{\| \mathbf {x} - \mathbf {c} \| _ {2} , r \right\}} (\mathbf {x} - \mathbf {c}),
$$

half-space 

$$
C _ {5} = \{\mathbf {x}: \mathbf {a} ^ {T} \mathbf {x} \leq \alpha \}, \quad \mathbf {x} - \frac {[ \mathbf {a} ^ {T} \mathbf {x} - \alpha ] _ {+}}{\| \mathbf {a} \| ^ {2}} \mathbf {a},
$$

where $\ell \in [ - \infty , \infty ) ^ { n } , \mathbf { u } \in ( - \infty , \infty ] ^ { n }$ are such that $\ell \leq \mathbf { u } , \mathbf { A } \in \mathbb { R } ^ { m \times n }$ has full row rank, $\mathbf { b } \in \mathbb { R } ^ { m } , \mathbf { c } \in \mathbb { R } ^ { n } , r > 0 , \mathbf { a } \in \mathbb { R } ^ { n } \ \backslash \ \{ \mathbf { 0 } \}$ , and $\alpha \in \mathbb { R }$ 

Note that we extended the definition of box sets given in Section 1.7.1 to include unbounded intervals, meaning that $\operatorname { B o x } [ \ell , \mathbf { u } ]$ is also defined when the components of - might also take the value $- \infty$ , and the components of u might take the value $\infty$ . However, boxes are always subsets of $\mathbb { R } ^ { n }$ , and the formula 

$$
\operatorname{Box} [ \boldsymbol {\ell}, \mathbf {u} ] = \left\{\mathbf {x} \in \mathbb {R} ^ {n}: \boldsymbol {\ell} \leq \mathbf {x} \leq \mathbf {u} \right\}
$$

still holds. For example, $\mathrm { B o x } [ \mathbf { 0 } , \infty \mathbf { e } ] = \mathbb { R } _ { + } ^ { n }$ 

## 6.4.3 Projection onto the Intersection of a Hyperplane and a Box

The next result develops an expression for the orthogonal projection onto another subset of $\mathbb { R } ^ { n }$ —the intersection of an hyperplane and a box. 

Theorem 6.27 (projection onto the intersection of a hyperplane and a box). Let $C \subseteq \mathbb { R } ^ { n }$ be given by 

$$
C = H _ {\mathbf {a}, b} \cap \operatorname{Box} [ \boldsymbol {\ell}, \mathbf {u} ] = \left\{\mathbf {x} \in \mathbb {R} ^ {n}: \mathbf {a} ^ {T} \mathbf {x} = b, \boldsymbol {\ell} \leq \mathbf {x} \leq \mathbf {u} \right\},
$$

where $\mathbf { a } \in \mathbb { R } ^ { n } \backslash \{ \mathbf { 0 } \} , b \in \mathbb { R } , \ell \in [ - \infty , \infty ) ^ { n } , \mathbf { u } \in ( - \infty , \infty ] ^ { n }$ . Assume that $C \neq \emptyset$ . Then 

$$
P _ {C} (\mathbf {x}) = P _ {\mathrm{Box} [ \boldsymbol {\ell}, \mathbf {u} ]} (\mathbf {x} - \boldsymbol {\mu} ^ {*} \mathbf {a}),
$$

where $\operatorname { B o x } [ \ell , \mathbf { u } ] = \{ \mathbf { y } \in \mathbb { R } ^ { n } : \ell _ { i } \leq y _ { i } \leq u _ { i } , i = 1 , 2 , \ldots , n \}$ and $\mu ^ { * }$ is a solution of the equation 

$$
\mathbf {a} ^ {T} P _ {\mathrm{Box} [ \ell , \mathbf {u} ]} (\mathbf {x} - \mu \mathbf {a}) = b.\tag{6.19}
$$

Proof. The orthogonal projection of x onto C is the unique optimal solution of 

$$
\min _ {\mathbf {y}} \left\{\frac {1}{2} \| \mathbf {y} - \mathbf {x} \| _ {2} ^ {2}: \mathbf {a} ^ {T} \mathbf {y} = b, \ell \leq \mathbf {y} \leq \mathbf {u} \right\}.\tag{6.20}
$$

A Lagrangian of the problem is 

$$
L (\mathbf {y}; \mu) = \frac {1}{2} \| \mathbf {y} - \mathbf {x} \| _ {2} ^ {2} + \mu (\mathbf {a} ^ {T} \mathbf {y} - b) = \frac {1}{2} \| \mathbf {y} - (\mathbf {x} - \mu \mathbf {a}) \| _ {2} ^ {2} - \frac {\mu^ {2}}{2} \| \mathbf {a} \| _ {2} ^ {2} + \mu (\mathbf {a} ^ {T} \mathbf {x} - b).\tag{6.21}
$$

Since strong duality holds for problem (6.20) (see Theorem A.1), it follows by Theorem $\mathrm { A . 2 }$ that $\mathbf { y } ^ { * }$ is an optimal solution of problem (6.20) if and only if there exists $\mu ^ { * } \in \mathbb { R }$ (which will actually be an optimal solution of the dual problem) for which 

$$
\mathbf {y} ^ {*} \in \operatorname{argmin} _ {\ell \leq \mathbf {y} \leq \mathbf {u}} L (\mathbf {y}; \mu^ {*}),
$$

$$
\mathbf {a} ^ {T} \mathbf {y} ^ {*} = b.\tag{6.22}
$$

(6.23) 

Using the expression of the Lagrangian given in (6.21), the relation (6.22) can be equivalently written as 

$$
\mathbf {y} ^ {*} = P _ {\mathrm{Box} [ \ell , \mathbf {u} ]} (\mathbf {x} - \mu^ {*} \mathbf {a}).
$$

The feasibility condition (6.23) can then be rewritten as 

$$
\mathbf {a} ^ {T} P _ {\mathrm{Box} [ \ell , \mathbf {u} ]} (\mathbf {x} - \boldsymbol {\mu} ^ {*} \mathbf {a}) = b. \quad \square
$$

Remark 6.28. The projection onto the box Box[-, u] is extremely simple and is done component-wise as described in Lemma 6.26. Note also that (6.19) actually consists in finding a root of the nonincreasing function $\varphi ( \mu ) = { \mathbf a } ^ { T } P _ { \mathrm { B o x } } ( { \mathbf x } - \mu { \mathbf a } ) - b $ which is $a$ task that can be performed eficiently even by simple procedures such as bisection. The fact that $\varphi$ is nonincreasing follows from the observation that $\varphi ( \mu ) =$ $\textstyle \sum _ { i = 1 } ^ { n } a _ { i }$ min $\{ \operatorname* { m a x } \{ x _ { i } - \mu a _ { i } , \ell _ { i } \} , u _ { i } \} - b$ and the fact that $\mu \mapsto a _ { i }$ min $\{ \operatorname* { m a x } \{ x _ { i } -$ $\mu a _ { i } , \ell _ { i } \} , u _ { i } \}$ is a nonincreasing function for any i. 

A direct consequence of Theorem 6.27 is an expression for the orthogonal projection onto the unit simplex. 

Corollary 6.29 (orthogonal projection onto the unit simplex). For any $\mathbf { x } \in \mathbb { R } ^ { n }$ ， 

$$
P _ {\Delta_ {n}} (\mathbf {x}) = [ \mathbf {x} - \mu^ {*} \mathbf {e} ] _ {+},
$$

where $\mu ^ { * }$ is a root of the equation 

$$
\mathbf {e} ^ {T} [ \mathbf {x} - \boldsymbol {\mu} ^ {*} \mathbf {e} ] _ {+} - 1 = 0.
$$

Proof. Invoking Theorem 6.27 with $\mathbf { a } = \mathbf { e } , b = 1 , \ell _ { i } = 0 , u _ { i } = \infty , i = 1 , 2 , \ldots , n$ and noting that in this case $P _ { \mathrm { B o x } [ \ell , { \bf u } ] } ( { \bf x } ) = [ { \bf x } ] _ { + }$ , the result follows. 

In order to expend the variety of sets on which we will be able to find simple expressions for the orthogonal projection mapping, in the next two subsections, we will discuss how to project onto level sets and epigraphs. 

## 6.4.4 Projection onto Level Sets

Theorem 6.30 (orthogonal projection onto level sets). Let $C = L e v ( f , \alpha ) =$ $\{ \mathbf { x } \in \mathbb { E } : f ( \mathbf { x } ) \leq \alpha \}$ , where $f : \mathbb { E }  ( - \infty , \infty ]$ is proper closed and convex, and $\alpha \in \mathbb { R }$ . Assume that there exists $\hat { \mathbf { x } } \in \mathbb { E }$ for which $f ( \hat { \mathbf { x } } ) < \alpha$ . Then 

$$
P _ {C} (\mathbf {x}) = \left\{ \begin{array}{l l} P _ {\mathrm{dom} (f)} (\mathbf {x}), & f (P _ {\mathrm{dom} (f)} (\mathbf {x})) \leq \alpha , \\ \operatorname{prox} _ {\lambda^ {*} f} (\mathbf {x}) & e l s e, \end{array} \right.\tag{6.24}
$$

where $\lambda ^ { * }$ is any positive root of the equation 

$$
\varphi (\lambda) \equiv f (\operatorname{prox} _ {\lambda f} (\mathbf {x})) - \alpha = 0.
$$

In addition, the function $\varphi$ is nonincreasing. 

Proof. The orthogonal projection of x onto $C$ is an optimal solution of the problem 

$$
\min _ {\mathbf {y} \in \mathbb {E}} \left\{\frac {1}{2} \| \mathbf {y} - \mathbf {x} \| ^ {2}: f (\mathbf {y}) \leq \alpha , \mathbf {y} \in X \right\},
$$

where $X = \operatorname { d o m } ( f )$ . A Lagrangian of the problem is $( \lambda \geq 0 )$ 

$$
L (\mathbf {y}; \lambda) = \frac {1}{2} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \lambda f (\mathbf {y}) - \alpha \lambda .\tag{6.25}
$$

Since the problem is convex and satisfies Slater’s condition, strong duality holds (see Theorem $\mathrm { A . 1 } )$ , and therefore it follows by the optimality conditions in Theorem A.2 that $\mathbf { y } ^ { * }$ is an optimal solution of problem (6.25) if and only if there exists $\lambda ^ { \ast } \in \mathbb { R } _ { + }$ for which 

$$
\mathbf {y} ^ {*} \in \operatorname{argmin} _ {\mathbf {y} \in X} L (\mathbf {y}; \lambda^ {*}),\tag{6.26}
$$

$$
f (\mathbf {y} ^ {*}) \leq \alpha ,\tag{6.27}
$$

$$
\lambda^ {*} (f (\mathbf {y} ^ {*}) - \alpha) = 0.\tag{6.28}
$$

There are two cases. If $P _ { X } ( \mathbf { x } )$ exists and $f ( P _ { X } \mathbf { ( x ) } ) \leq \alpha .$ , then $\mathbf { y } ^ { * } = P _ { X } ( \mathbf { x } )$ , and $\lambda ^ { * } = 0$ is a solution to the system (6.26), (6.27), (6.28). Otherwise, if $P _ { X } ( \mathbf { x } )$ does not exist or $f ( P _ { X } ( \mathbf { x } ) ) > \alpha$ , then $\lambda ^ { * } > 0$ , and in this case the system (6.26), (6.27), (6.28) reduces to $\mathbf { y } ^ { * } = \mathrm { p r o x } _ { \lambda ^ { * } f } ( \mathbf { x } )$ and $f ( \mathrm { p r o x } _ { \lambda ^ { * } f } ( \mathbf { x } ) ) = \alpha$ , which yields the formula (6.24). 

To prove that $\varphi$ is nonincreasing, recall that 

$$
\operatorname{prox} _ {\lambda f} (\mathbf {x}) = \operatorname{argmin} _ {\mathbf {y} \in X} \left\{\frac {1}{2} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \lambda (f (\mathbf {y}) - \alpha) \right\}.
$$

Take $0 \leq \lambda _ { 1 } < \lambda _ { 2 }$ . Then denoting $\mathbf { v } _ { 1 } = \mathrm { p r o x } _ { \lambda _ { 1 } f } ( \mathbf { x } )$ and $\mathbf { v } _ { 2 } = \mathrm { p r o x } _ { \lambda _ { 2 } f } ( \mathbf { x } )$ , we have 

$$
\begin{array}{l} \frac {1}{2} \| \mathbf {v} _ {2} - \mathbf {x} \| ^ {2} + \lambda_ {2} (f (\mathbf {v} _ {2}) - \alpha) \\ = \frac {1}{2} \| \mathbf {v} _ {2} - \mathbf {x} \| ^ {2} + \lambda_ {1} (f (\mathbf {v} _ {2}) - \alpha) + (\lambda_ {2} - \lambda_ {1}) (f (\mathbf {v} _ {2}) - \alpha) \\ \geq \frac {1}{2} \| \mathbf {v} _ {1} - \mathbf {x} \| ^ {2} + \lambda_ {1} (f (\mathbf {v} _ {1}) - \alpha) + (\lambda_ {2} - \lambda_ {1}) (f (\mathbf {v} _ {2}) - \alpha) \\ = \frac {1}{2} \| \mathbf {v} _ {1} - \mathbf {x} \| ^ {2} + \lambda_ {2} (f (\mathbf {v} _ {1}) - \alpha) + (\lambda_ {2} - \lambda_ {1}) (f (\mathbf {v} _ {2}) - f (\mathbf {v} _ {1})) \\ \geq \frac {1}{2} \| \mathbf {v} _ {2} - \mathbf {x} \| ^ {2} + \lambda_ {2} (f (\mathbf {v} _ {2}) - \alpha) + (\lambda_ {2} - \lambda_ {1}) (f (\mathbf {v} _ {2}) - f (\mathbf {v} _ {1})). \end{array}
$$

Therefore, $( \lambda _ { 2 } - \lambda _ { 1 } ) ( f ( \mathbf { v } _ { 2 } ) - f ( \mathbf { v } _ { 1 } ) ) \leq 0$ . Since $\lambda _ { 1 } ~ < ~ \lambda _ { 2 }$ , we can conclude that $f ( \mathbf { v } _ { 2 } ) \leq f ( \mathbf { v } _ { 1 } )$ . Finally, 

$$
\varphi (\lambda_ {2}) = f (\mathbf {v} _ {2}) - \alpha \leq f (\mathbf {v} _ {1}) - \alpha = \varphi (\lambda_ {1}),
$$

establishing the monotonicity of $\varphi .$ 

Remark 6.31. Note that in Theorem 6.30 f is assumed to be closed, but this does not necessarily imply that dom(f) is closed. In cases where dom $( f )$ is not closed, it might happen that $P _ { \mathrm { d o m } ( f ) } ( \mathbf { x } )$ does not exist and formula (6.24) amounts to $P _ { C } ( \mathbf { x } ) = \mathrm { p r o x } _ { \lambda ^ { * } f } ( \mathbf { x } )$ 

Example 6.32 (projection onto the intersection of a half-space and a box). Consider the set 

$$
C = H _ {\mathbf {a}, b} ^ {-} \cap \mathrm{Box} [ \boldsymbol {\ell}, \mathbf {u} ] = \{\mathbf {x} \in \mathbb {R} ^ {n}: \mathbf {a} ^ {T} \mathbf {x} \leq b, \boldsymbol {\ell} \leq \mathbf {x} \leq \mathbf {u} \},
$$

where $\mathbf { a } \in \mathbb { R } ^ { n } \setminus \{ \mathbf { 0 } \} , b \in \mathbb { R } , \ell \in [ - \infty , \infty ) ^ { n }$ and $\mathbf { u } \in ( - \infty , \infty ] ^ { n }$ . Assume that $C \neq \emptyset$ Then $C = \operatorname { L e v } ( f , b )$ , where $f ( \mathbf { x } ) = \mathbf { a } ^ { T } \mathbf { x } + \delta _ { \mathrm { B o x } [ \ell , \mathbf { u } ] } ( \mathbf { x } )$ . For any $\lambda > 0$ 2 

$$
\operatorname{prox} _ {\lambda f} (\mathbf {x}) = \operatorname{prox} _ {\lambda \mathbf {a} ^ {T} (\cdot) + \delta_ {\operatorname{Box} [ \ell , \mathbf {u} ]} (\cdot)} (\mathbf {x}) \stackrel {(*)} {=} \operatorname{prox} _ {\delta_ {\operatorname{Box} [ \ell , \mathbf {u} ]}} (\mathbf {x} - \lambda \mathbf {a}) = P _ {\operatorname{Box} [ \ell , \mathbf {u} ]} (\mathbf {x} - \lambda \mathbf {a}),
$$

where in the equality (∗) we used Theorem 6.13. Invoking Theorem 6.30, we obtain the following formula for the projection on C: 

$$
P _ {C} (\mathbf {x}) = \left\{ \begin{array}{l l} P _ {\mathrm{Box} [ \boldsymbol {\ell}, \mathbf {u} ]} (\mathbf {x}), & \mathbf {a} ^ {T} P _ {\mathrm{Box} [ \boldsymbol {\ell}, \mathbf {u} ]} (\mathbf {x}) \leq b, \\ P _ {\mathrm{Box} [ \boldsymbol {\ell}, \mathbf {u} ]} (\mathbf {x} - \lambda^ {*} \mathbf {a}), & \mathbf {a} ^ {T} P _ {\mathrm{Box} [ \boldsymbol {\ell}, \mathbf {u} ]} (\mathbf {x}) > b, \end{array} \right.
$$

where $\lambda ^ { * }$ is any positive root of the nonincreasing function 

$$
\varphi (\lambda) = \mathbf {a} ^ {T} P _ {\mathrm{Box} [ \ell , \mathbf {u} ]} (\mathbf {x} - \lambda \mathbf {a}) - b.
$$

Example 6.33 (projection onto the $\boldsymbol { l } _ { 1 }$ ball). Let $C = B _ { \parallel . \parallel _ { 1 } } [ \mathbf { 0 } , \alpha ] = \{ \mathbf { x } \in \mathbb { R } ^ { n }$ $\| \mathbf { x } \| _ { 1 } \leq \alpha \}$ , where $\alpha > 0$ . Then $C = \operatorname { L e v } ( f , \alpha )$ with $f ( \mathbf { x } ) = \| \mathbf { x } \| _ { 1 }$ . The prox of $\lambda f = \lambda \| \cdot \|$ <sub>1</sub> for any $\lambda > 0$ was computed in Example 6.8, where it was shown that 

$$
\operatorname{prox} _ {\lambda f} (\mathbf {x}) = \mathcal {T} _ {\lambda} (\mathbf {x}) \text {   for   all   } \mathbf {x} \in \mathbb {R} ^ {n}
$$

with $\mathcal { T } _ { \lambda }$ being the soft thresholding operator given by $\mathcal { T } _ { \lambda } ( \mathbf { x } ) = [ \mathbf { x } - \lambda \mathbf { e } ] _ { + } \odot \mathrm { s g n } ( \mathbf { x } )$ Invoking Theorem 6.30, we obtain that 

$$
P _ {B _ {\| \cdot \| _ {1}} [ \mathbf {0}, \alpha ]} (\mathbf {x}) = \left\{ \begin{array}{l l} \mathbf {x}, & \| \mathbf {x} \| _ {1} \leq \alpha , \\ \mathcal {T} _ {\lambda^ {*}} (\mathbf {x}), & \| \mathbf {x} \| _ {1} > \alpha , \end{array} \right.
$$

where $\lambda ^ { * }$ is any positive root of the nonincreasing function 

$$
\varphi (\lambda) = \left\| \mathcal {T} _ {\lambda} (\mathbf {x}) \right\| _ {1} - \alpha .
$$

The next example uses a generalization of the soft thresholding mapping, which will be called the two-sided soft thresholding operator, and is defined for any a, $\mathbf { b } \in ( - \infty , \infty ] ^ { n }$ as 

$$
\mathcal {S} _ {\mathbf {a}, \mathbf {b}} (\mathbf {x}) = (\min \{\max \{| x _ {i} | - a _ {i}, 0 \}, b _ {i} \} \mathrm{sgn} (x _ {i})) _ {i = 1} ^ {n}, \quad \mathbf {x} \in \mathbb {R} ^ {n}.
$$

Obviously, 

$$
\mathcal {S} _ {\lambda \mathbf {e}, \infty \mathbf {e}} = \mathcal {T} _ {\lambda}.
$$

Here ∞e is the n-dimensional column vector whose elements are all $\infty$ . A plot of the function $t \mapsto S _ { 1 , 2 } ( t )$ is given in Figure 6.3. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/06afcb51-43be-4db1-8b62-226fc4277ca3/218211e0f7896a3f55a7076fa03804f21b2502c803a7347cc2a34ca5dc817a9f.jpg)



Figure 6.3. The two-sided soft thresholding function $t \ \mapsto \ S _ { 1 , 2 } ( t ) \ =$ min $\{ \operatorname* { m a x } \{ | t | - 1 , 0 \} , 2 \} \mathrm { s g n } ( t )$


Example 6.34 (projection onto the intersection of weighted $\boldsymbol { l } _ { 1 }$ ball and a box). Let $C \subseteq \mathbb { R } ^ { n }$ be given by 

$$
C = \left\{\mathbf {x} \in \mathbb {R} ^ {n}: \sum_ {i = 1} ^ {n} \omega_ {i} | x _ {i} | \leq \beta , - \boldsymbol {\alpha} \leq \mathbf {x} \leq \boldsymbol {\alpha} \right\},
$$

where $\omega \in \mathbb { R } _ { + } ^ { n } , \alpha \in [ 0 , \infty ] ^ { n }$ , and $\beta \in \mathbb { R } _ { + + }$ . Then obviously $C = \mathrm { L e v } ( f , \beta )$ , where 

$$
f (\mathbf {x}) = \boldsymbol {\omega} ^ {T} | \mathbf {x} | + \delta_ {\mathrm{Box} [ - \boldsymbol {\alpha}, \boldsymbol {\alpha} ]} (\mathbf {x}) = \left\{ \begin{array}{l l} \sum_ {i = 1} ^ {n} \omega_ {i} | x _ {i} |, & - \boldsymbol {\alpha} \leq \mathbf {x} \leq \boldsymbol {\alpha}, \\ \infty & \text {else} \end{array} \right.
$$

for any $\mathbf { x } \in \mathbb { R } ^ { n }$ . By Example 6.23, for any $\lambda > 0$ and $\mathbf { x } \in \mathbb { R } ^ { n }$ 2 

$$
\operatorname{prox} _ {\lambda f} (\mathbf {x}) = \left(\min \{\max \{| x _ {i} | - \lambda \omega_ {i}, 0 \}, \alpha_ {i} \} \operatorname{sgn} (x _ {i})\right) _ {i = 1} ^ {n} = \mathcal {S} _ {\lambda \boldsymbol {\omega}, \boldsymbol {\alpha}} (\mathbf {x}).
$$

Therefore, invoking Theorem 6.30, we obtain that 

$$
P _ {C} (\mathbf {x}) = \left\{ \begin{array}{l l} P _ {\mathrm{Box} [ - \boldsymbol {\alpha}, \boldsymbol {\alpha} ]} (\mathbf {x}), & \boldsymbol {\omega} ^ {T} | P _ {\mathrm{Box} [ - \boldsymbol {\alpha}, \boldsymbol {\alpha} ]} (\mathbf {x}) | \leq \beta , \\ \mathcal {S} _ {\lambda^ {*} \boldsymbol {\omega}, \boldsymbol {\alpha}} (\mathbf {x}), & \boldsymbol {\omega} ^ {T} | P _ {\mathrm{Box} [ - \boldsymbol {\alpha}, \boldsymbol {\alpha} ]} (\mathbf {x}) | > \beta , \end{array} \right.
$$

where $\lambda ^ { * }$ is any positive root of the nonincreasing function 

$$
\varphi (\lambda) = \pmb {\omega} ^ {T} | \mathcal {S} _ {\lambda \pmb {\omega}, \pmb {\alpha}} (\mathbf {x}) | - \beta .
$$

As a final illustration of Theorem 6.30, we give an example in which the domain of f is not closed. 

Example 6.35. Let 

$$
C = \left\{\mathbf {x} \in \mathbb {R} _ {+ +} ^ {n}: \Pi_ {i = 1} ^ {n} x _ {i} \geq \alpha \right\},
$$

where $\alpha > 0$ . The key property that will allow us to compute the orthogonal projection onto C is the fact that it can be rewritten as 

$$
C = \left\{\mathbf {x} \in \mathbb {R} _ {+ +} ^ {n}: - \sum_ {i = 1} ^ {n} \log x _ {i} \leq - \log \alpha \right\}.
$$

Thus, $C = \operatorname { L e v } ( f , - \log \alpha )$ , where $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ is the negative sum of logs function: 

$$
f (\mathbf {x}) = \left\{ \begin{array}{l l} - \sum_ {i = 1} ^ {n} \log x _ {i}, & \mathbf {x} \in \mathbb {R} _ {+ +} ^ {n}, \\ \infty & \text { else }. \end{array} \right.
$$

In Example 6.9 it was shown that for any $\mathbf { x } \in \mathbb { R } ^ { n }$ 

$$
\mathrm{prox} _ {\lambda f} (\mathbf {x}) = \left(\frac {x _ {j} + \sqrt {x _ {j} ^ {2} + 4 \lambda}}{2}\right) _ {j = 1} ^ {n}.
$$

We can now invoke Theorem 6.30 to obtain a formula (up to a single parameter that can be found by a one-dimensional search) for the projection onto $C ,$ but there is one issue that needs to be treated delicately. If $\mathbf { x } \notin \mathbb { R } _ { + + } ^ { n }$ , meaning that it has at least one nonpositive element, then $P _ { \mathbb { R } _ { + + } ^ { n } } ( \mathbf { x } )$ does not exist. In this case only the second part of (6.24) is relevant, meaning that $P _ { C } ( \mathbf { x } ) = \mathrm { p r o x } _ { \lambda ^ { * } f } ( \mathbf { x } )$ . To conclude, 

$$
P _ {C} (\mathbf {x}) = \left\{ \begin{array}{l l} \mathbf {x}, & \mathbf {x} \in C, \\ \left(\frac {x _ {j} + \sqrt {x _ {j} ^ {2} + 4 \lambda^ {*}}}{2}\right) _ {j = 1} ^ {n}, & \mathbf {x} \notin C, \end{array} \right.
$$

where $\lambda ^ { * }$ is any positive root of the nonincreasing function 

$$
\varphi (\lambda) = - \sum_ {j = 1} ^ {n} \log \left(\frac {x _ {j} + \sqrt {x _ {j} ^ {2} + 4 \lambda}}{2}\right) + \log \alpha .
$$

## 6.4.5 Projection onto Epigraphs

We can use Theorem 6.30 to prove a theorem on the projection onto epigraphs of convex functions. 

Theorem 6.36 (orthogonal projection onto epigraphs). Let 

$$
C = \operatorname{epi} (g) = \left\{\left(\mathbf {x}, t\right) \in \mathbb {E} \times \mathbb {R}: g (\mathbf {x}) \leq t \right\},
$$

where $g : \mathbb { E }  \mathbb { R }$ is convex. Then 

$$
P _ {C} ((\mathbf {x}, s)) = \left\{ \begin{array}{l l} (\mathbf {x}, s), & g (\mathbf {x}) \leq s, \\ (\operatorname{prox} _ {\lambda^ {*} g} (\mathbf {x}), s + \lambda^ {*}), & g (\mathbf {x}) > s, \end{array} \right.
$$

where $\lambda ^ { * }$ is any positive root of the function 

$$
\psi (\lambda) = g (\mathrm{prox} _ {\lambda g} (\mathbf {x})) - \lambda - s.
$$

In addition, $\psi$ is nonincreasing. 

Proof. Define $f : \mathbb { E } \times \mathbb { R } \to \mathbb { R }$ as $f ( \mathbf { x } , t ) \equiv g ( \mathbf { x } ) - t$ . By definition of the prox, 

$$
\begin{array}{c} \operatorname{prox} _ {\lambda f} (\mathbf {x}, s) = \operatorname{argmin} _ {\mathbf {y}, t} \left\{\frac {1}{2} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \frac {1}{2} (t - s) ^ {2} + \lambda f (\mathbf {y}, t) \right\} \\ = \operatorname{argmin} _ {\mathbf {y}, t} \left\{\frac {1}{2} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \frac {1}{2} (t - s) ^ {2} + \lambda g (\mathbf {y}) - \lambda t \right\}. \end{array}
$$

The above problem is separable in y and $\mathbf { t } ,$ and thus 

$$
\begin{array}{r l} & {\mathrm{prox} _ {\lambda f} (\mathbf {x}, s) = \left(\mathrm{argmin} _ {\mathbf {y}} \left\{\frac {1}{2} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \lambda g (\mathbf {y}) \right\}, \mathrm{argmin} _ {t} \left\{\frac {1}{2} (t - s) ^ {2} - \lambda t \right\}\right)} \\ & {\qquad = \left(\mathrm{prox} _ {\lambda g} (\mathbf {x}), \mathrm{prox} _ {\lambda h} (s)\right),} \end{array}
$$

where $h ( t ) \equiv - t$ . Since λh is linear, then by Section 6.2.2, $\mathrm { p r o x } _ { \lambda h } ( z ) = z + \lambda$ for $\mathrm { a n y } \ z \in \mathbb { R }$ . Thus, 

$$
\operatorname{prox} _ {\lambda f} (\mathbf {x}, s) = \left(\operatorname{prox} _ {\lambda g} (\mathbf {x}), s + \lambda\right).
$$

Since $\operatorname { e p i } ( g ) = \operatorname { L e v } ( f , 0 )$ , we can invoke Theorem 6.30 (noting that dom $( f ) = \mathbb { E } )$ and obtain that 

$$
P _ {C} ((\mathbf {x}, s)) = \left\{ \begin{array}{l l} (\mathbf {x}, s), & g (\mathbf {x}) \leq s, \\ (\operatorname{prox} _ {\lambda^ {*} g} (\mathbf {x}), s + \lambda), & g (\mathbf {x}) > s, \end{array} \right.
$$

where $\lambda ^ { * }$ is any positive root of the function 

$$
\psi (\lambda) = g (\mathrm{prox} _ {\lambda g} (\mathbf {x})) - \lambda - s,
$$

which by Theorem 6.30 is nonincreasing. 

Example 6.37 (projection onto the Lorentz cone). Consider the Lorentz cone, which is given by $L ^ { n } = \{ ( \mathbf { x } , t ) \in \mathbb { R } ^ { n } \times \mathbb { R } : \| \mathbf { x } \| _ { 2 } \leq t \}$ . We will show that for any $( \mathbf { x } , s ) \in \mathbb { R } ^ { n } \times \mathbb { R }$ 

$$
P _ {L ^ {n}} (\mathbf {x}, s) = \left\{ \begin{array}{l l} \left(\frac {\| \mathbf {x} \| _ {2} + s}{2 \| \mathbf {x} \| _ {2}} \mathbf {x}, \frac {\| \mathbf {x} \| _ {2} + s}{2}\right), & \| \mathbf {x} \| _ {2} \geq | s |, \\ (\mathbf {0}, 0), & s <   \| \mathbf {x} \| _ {2} <   - s, \\ (\mathbf {x}, s), & \| \mathbf {x} \| _ {2} \leq s. \end{array} \right.
$$

To show the above,<sup>31</sup> we invoke Theorem 6.36 to obtain the formula 

$$
P _ {L ^ {n}} ((\mathbf {x}, s)) = \left\{ \begin{array}{l l} (\mathbf {x}, s), & \| \mathbf {x} \| _ {2} \leq s, \\ (\operatorname{prox} _ {\lambda^ {*} \| \cdot \| _ {2}} (\mathbf {x}), s + \lambda^ {*}), & \| \mathbf {x} \| _ {2} > s, \end{array} \right.
$$

where $\lambda ^ { * }$ is any positive root of the nonincreasing function 

$$
\psi (\lambda) = \| \mathrm{prox} _ {\lambda \| \cdot \| _ {2}} (\mathbf {x}) \| _ {2} - \lambda - s.\tag{6.29}
$$

Let $( \mathbf { x } , s ) \in \mathbb { R } ^ { n } \times \mathbb { R }$ be such that $\| \mathbf { x } \| _ { 2 } > s$ . Recall that by Example 6.19, 

$$
\operatorname{prox} _ {\lambda \| \cdot \| _ {2}} (\mathbf {x}) = \left[ 1 - \frac {\lambda}{\max \{\| \mathbf {x} \| _ {2} , \lambda \}} \right] _ {+} \mathbf {x}.
$$

Plugging the above into the expression of $\psi$ in (6.29) yields 

$$
\psi (\lambda) = \left\{ \begin{array}{l l} \| \mathbf {x} \| _ {2} - 2 \lambda - s, & \lambda \leq \| \mathbf {x} \| _ {2}, \\ - \lambda - s, & \lambda \geq \| \mathbf {x} \| _ {2}. \end{array} \right.
$$

The unique positive root $\lambda ^ { * }$ of the piecewise linear function $\psi$ is 

$$
\lambda^ {*} = \left\{ \begin{array}{l l} \frac {\| \mathbf {x} \| _ {2} - s}{2}, & \| \mathbf {x} \| _ {2} \geq - s, \\ - s, & \| \mathbf {x} \| _ {2} <   - s. \end{array} \right.
$$

Thus, in the case $\| \mathbf { x } \| _ { 2 } > s$ (noting that $\| \mathbf { x } \| _ { 2 } \geq - s$ corresponds to the case where $\| \mathbf { x } \| _ { 2 } \geq \lambda ^ { * }$ and $\| \mathbf { x } \| _ { 2 } < - s$ corresponds to $\| \mathbf { x } \| _ { 2 } \leq \lambda ^ { * } )$ , 

$$
\begin{array}{l} (\operatorname{prox} _ {\lambda^ {*} \| \cdot \| _ {2}} (\mathbf {x}), s + \lambda^ {*}) = \left(\left[ 1 - \frac {\lambda^ {*}}{\max \{\| \mathbf {x} \| _ {2} , \lambda^ {*} \}} \right] _ {+} \mathbf {x}, s + \lambda^ {*}\right), \\ = \left\{ \begin{array}{l l} \left(\left[ 1 - \frac {\| \mathbf {x} \| _ {2} - s}{2 \| \mathbf {x} \| _ {2}} \right] _ {+} \mathbf {x}, \frac {\| \mathbf {x} \| _ {2} + s}{2}\right), & \| \mathbf {x} \| _ {2} \geq - s, \\ (\mathbf {0}, 0), & \| \mathbf {x} \| _ {2} <   - s. \end{array} \right. \\ = \left\{ \begin{array}{l l} \left(\frac {\| \mathbf {x} \| _ {2} + s}{2 \| \mathbf {x} \| _ {2}} \mathbf {x}, \frac {\| \mathbf {x} \| _ {2} + s}{2}\right), & \| \mathbf {x} \| _ {2} \geq - s, \\ (\mathbf {0}, 0), & \| \mathbf {x} \| _ {2} <   - s. \end{array} \right. \end{array}
$$

Recalling that $\| \mathbf { x } \| _ { 2 } > s$ , we have thus established that $P _ { L ^ { n } } ( \mathbf { x } , s ) = ( \mathbf { 0 } , 0 )$ when $s < \| \mathbf { x } \| _ { 2 } < - s$ and that whenever $( \mathbf { x } , s )$ satisfies $\| \mathbf { x } \| _ { 2 } > s$ and $\| \mathbf { x } \| _ { 2 } ~ \ge ~ - s$ , the formula 

$$
P _ {L ^ {n}} (\mathbf {x}, s) = \left(\frac {\| \mathbf {x} \| _ {2} + s}{2 \| \mathbf {x} \| _ {2}} \mathbf {x}, \frac {\| \mathbf {x} \| _ {2} + s}{2}\right)\tag{6.30}
$$

holds. The result now follows by noting that 

$$
\left\{\left(\mathbf {x}, s\right): \| \mathbf {x} \| _ {2} \geq | s | \right\} = \left\{\left(\mathbf {x}, s\right): \| \mathbf {x} \| _ {2} > s, \| \mathbf {x} \| _ {2} \geq - s \right\} \cup \left\{\left(\mathbf {x}, s\right): \| \mathbf {x} \| _ {2} = s \right\},
$$

and that formula (6.30) is trivial for the case $\| \mathbf { x } \| _ { 2 } = s$ (amounts to $P _ { L ^ { n } } ( \mathbf { x } , s ) =$ $\mathbf { \rho } ( \mathbf { x } , s ) )$ 

Example 6.38 (projection onto the epigraph of the $l _ { 1 } { \bf - n o r m ) }$ . Let 

$$
C = \left\{\left(\mathbf {y}, t\right) \in \mathbb {R} ^ {n} \times \mathbb {R}: \| \mathbf {y} \| _ {1} \leq t \right\}.
$$

Invoking Theorem 6.36 and recalling that for any $\lambda > 0 , \mathrm { p r o x } _ { \lambda \| \cdot \| _ { 1 } } = \mathcal { T } _ { \lambda }$ , where $\mathcal { T } _ { \lambda }$ is the soft thresholding operator (see Example 6.8), it follows that 

$$
P _ {C} ((\mathbf {x}, s)) = \left\{ \begin{array}{l l} (\mathbf {x}, s), & \| \mathbf {x} \| _ {1} \leq s, \\ (\mathcal {T} _ {\lambda^ {*}} (\mathbf {x}), s + \lambda^ {*}), & \| \mathbf {x} \| _ {1} > s, \end{array} \right.
$$

where $\lambda ^ { * }$ is any positive root of the nonincreasing function 

$$
\varphi (\lambda) = \left\| \mathcal {T} _ {\lambda} (\mathbf {x}) \right\| _ {1} - \lambda - s.
$$

## 6.4.6 Summary of Orthogonal Projection Computations

Table 6.1 describes all the examples of orthogonal projection computations onto subsets of $\mathbb { R } ^ { n }$ and $\mathbb { R } ^ { n } \times \mathbb { R }$ that were discussed so far. 


Table 6.1. The following notation is used in the table. $[ \mathbf { x } ] _ { + }$ is the nonnegative part of x, $, \mathcal { T } _ { \lambda } ( \mathbf { y } ) = ( [ | y _ { i } | - \lambda ] _ { + } \mathrm { s g n } ( y _ { i } ) ) _ { i = 1 } ^ { n }$ , and $S _ { \mathbf { a } , \mathbf { b } } ( \mathbf { x } ) = ( \operatorname* { m i n } \{ \operatorname* { m a x } \{ | x _ { i } | - $ $a _ { i } , 0 \} , b _ { i } \} \mathrm { s g n } ( x _ { i } ) ) _ { i = 1 } ^ { n }$


<table><tr><td>set (C)</td><td><eq>P_C(\mathbf{x})</eq></td><td>Assumptions</td><td>Reference</td></tr><tr><td><eq>\mathbb{R}_+^n</eq></td><td><eq>[\mathbf{x}]_+</eq></td><td>-</td><td>Lemma 6.26</td></tr><tr><td>Box<eq>[\ell,\mathbf{u}]</eq></td><td><eq>P_C(\mathbf{x})_i = \min\{\max\{x_i,\ell_i\},u_i\}</eq></td><td><eq>\ell_i \leq u_i</eq></td><td>Lemma 6.26</td></tr><tr><td><eq>B_{\|\cdot\|_2}[\mathbf{c},r]</eq></td><td><eq>\mathbf{c} + \frac{r}{\max\{ \| \mathbf{x}-\mathbf{c}\|_2,r\}} (\mathbf{x}-\mathbf{c})</eq></td><td><eq>\mathbf{c} \in \mathbb{R}^n, r &gt; 0</eq></td><td>Lemma 6.26</td></tr><tr><td><eq>\{\mathbf{x} : \mathbf{A}\mathbf{x} = \mathbf{b}\}</eq></td><td><eq>\mathbf{x} - \mathbf{A}^T (\mathbf{A}\mathbf{A}^T)^{-1} (\mathbf{A}\mathbf{x} - \mathbf{b})</eq></td><td><eq>\mathbf{A} \in \mathbb{R}^{m \times n}, \mathbf{b} \in \mathbb{R}^m, \mathbf{A} \text{ full row rank}</eq></td><td>Lemma 6.26</td></tr><tr><td><eq>\{\mathbf{x} : \mathbf{a}^T \mathbf{x} \leq b\}</eq></td><td><eq>\mathbf{x} - \frac{[\mathbf{a}^T \mathbf{x}-b]_+}{\|\mathbf{a}\|^2} \mathbf{a}</eq></td><td><eq>\mathbf{0} \neq \mathbf{a} \in \mathbb{R}^n, b \in \mathbb{R}</eq></td><td>Lemma 6.26</td></tr><tr><td><eq>\Delta_n</eq></td><td><eq>[\mathbf{x} - \mu^*\mathbf{e}]_+ \text{ where } \mu^* \in \mathbb{R} \text{ satisfies } \mathbf{e}^T [\mathbf{x} - \mu^*\mathbf{e}]_+ = 1</eq></td><td></td><td>Corollary 6.29</td></tr><tr><td><eq>H_{\mathbf{a},b} \cap \text{Box}[\ell,\mathbf{u}]</eq></td><td><eq>P_{\text{Box}[\ell,\mathbf{u}]}(\mathbf{x} - \mu^*\mathbf{a}) \text{ where } \mu^* \in \mathbb{R} \text{ satisfies } \mathbf{a}^T P_{\text{Box}[\ell,\mathbf{u}]}(\mathbf{x} - \mu^*\mathbf{a}) = b</eq></td><td><eq>\mathbf{a} \in \mathbb{R}^n \setminus \{\mathbf{0}\}, b \in \mathbb{R}</eq></td><td>Theorem 6.27</td></tr><tr><td><eq>H_{\mathbf{a},b}^{-} \cap \text{Box}[\ell,\mathbf{u}]</eq></td><td><eq>\begin{cases} P_{\text{Box}[\ell,\mathbf{u}]}(\mathbf{x}), &amp; \mathbf{a}^T \mathbf{v}_{\mathbf{x}} \leq b, \\ P_{\text{Box}[\ell,\mathbf{u}]}(\mathbf{x} - \lambda^*\mathbf{a}), &amp; \mathbf{a}^T \mathbf{v}_{\mathbf{x}} &gt; b, \\ \mathbf{v}_{\mathbf{x}} &amp; = P_{\text{Box}[\ell,\mathbf{u}]}(\mathbf{x}), \\ \mathbf{a}^T P_{\text{Box}[\ell,\mathbf{u}]}(\mathbf{x} - \lambda^*\mathbf{a}) = b, \lambda^* &gt; 0 \end{cases}</eq></td><td><eq>\mathbf{a} \in \mathbb{R}^n \setminus \{\mathbf{0}\}, b \in \mathbb{R}</eq></td><td>Example 6.32</td></tr><tr><td><eq>B_{\|\cdot\|_1}[\mathbf{0},\alpha]</eq></td><td><eq>\begin{cases} \mathbf{x}, &amp; \| \mathbf{x}\|_1 \leq \alpha, \\ \mathcal{T}_{\lambda^*}(\mathbf{x}), &amp; \| \mathbf{x}\|_1 &gt; \alpha, \\ \| \mathcal{T}_{\lambda^*}(\mathbf{x})\|_1 = \alpha, &amp; \lambda^* &gt; 0 \end{cases}</eq></td><td><eq>\alpha &gt; 0</eq></td><td>Example 6.33</td></tr><tr><td><eq>\{\mathbf{x} : \boldsymbol{\omega}^T |\mathbf{x}| \leq \beta, -\boldsymbol{\alpha} \leq \mathbf{x} \leq \boldsymbol{\alpha}\}</eq></td><td><eq>\begin{cases} \mathbf{v}_{\mathbf{x}}, &amp; \boldsymbol{\omega}^T |\mathbf{v}_{\mathbf{x}}| \leq \beta, \\ \mathcal{S}_{\lambda^*\boldsymbol{\omega},\boldsymbol{\alpha}}(\mathbf{x}), &amp; \boldsymbol{\omega}^T |\mathbf{v}_{\mathbf{x}}| &gt; \beta, \\ \mathbf{v}_{\mathbf{x}} &amp; = P_{\text{Box}[-\boldsymbol{\alpha},\boldsymbol{\alpha}]}(\mathbf{x}), \\ \boldsymbol{\omega}^T |\mathcal{S}_{\lambda^*\boldsymbol{\omega},\boldsymbol{\alpha}}(\mathbf{x})| = \beta, \lambda^* &gt; 0 \end{cases}</eq></td><td><eq>\boldsymbol{\omega} \in \mathbb{R}_+^n, \boldsymbol{\alpha} \in [0,\infty]^n, \beta \in \mathbb{R}_{++}</eq></td><td>Example 6.34</td></tr><tr><td><eq>\{\mathbf{x} &gt; \mathbf{0} : \Pi x_i \geq \alpha\}</eq></td><td><eq>\begin{cases} \mathbf{x}, &amp; \mathbf{x} \in C, \\ \left( \frac{x_j + \sqrt{x_j^2 + 4\lambda^*}}{2} \right)^n, &amp; \mathbf{x} \notin C, \\ \Pi_{j=1}^n \left( (x_j + \sqrt{x_j^2 + 4\lambda^*})/2 \right) &amp; = \\ \alpha, &amp; \lambda^* &gt; 0 \end{cases}</eq></td><td><eq>\alpha &gt; 0</eq></td><td>Example 6.35</td></tr><tr><td><eq>\{(\mathbf{x},s) : \| \mathbf{x}\|_2 \leq s\}</eq></td><td><eq>\left( \frac{\|\mathbf{x}\|_2 + s}{2\|\mathbf{x}\|_2} \mathbf{x}, \frac{\|\mathbf{x}\|_2 + s}{2} \right) \text{ if } \| \mathbf{x}\|_2 \geq |s| (0,0) \text{ if } s &lt; \| \mathbf{x}\|_2 &lt; -s, \\ (\mathbf{x},s) \text{ if } \| \mathbf{x}\|_2 \leq s.</eq></td><td></td><td>Example 6.37</td></tr><tr><td><eq>\{(\mathbf{x},s) : \| \mathbf{x}\|_1 \leq s\}</eq></td><td><eq>\begin{cases} (\mathbf{x},s), &amp; \| \mathbf{x}\|_1 \leq s, \\ (\mathcal{T}_{\lambda^*}(\mathbf{x}), s + \lambda^*), &amp; \| \mathbf{x}\|_1 &gt; s, \\ \| \mathcal{T}_{\lambda^*}(\mathbf{x})\|_1 - \lambda^* - s = 0, &amp; \lambda^* &gt; 0 \end{cases}</eq></td><td></td><td>Example 6.38</td></tr></table>

## 6.5 The Second Prox Theorem

We can use Fermat’s optimality condition (Theorem 3.63) in order to prove the second prox theorem. 

Theorem 6.39 (second prox theorem). Let $f : \mathbb { E }  ( - \infty , \infty ]$ be a proper closed and convex function. Then for any x, u ∈ E, the following three claims are equivalent: 

(i) $\mathbf { u } = \mathrm { p r o x } _ { f } ( \mathbf { x } )$ 

(ii) $\mathbf { x } - \mathbf { u } \in \partial f ( \mathbf { u } )$ 

(iii) $\langle \mathbf { x } - \mathbf { u } , \mathbf { y } - \mathbf { u } \rangle \leq f ( \mathbf { y } ) - f ( \mathbf { u } ) ~ f o r ~ a n y ~ \mathbf { y } \in \mathbb { E } .$ 

Proof. By definition, $\mathbf { u } = \mathrm { p r o x } _ { f } ( \mathbf { x } )$ if and only if u is the minimizer of the problem 

$$
\min _ {\mathbf {v}} \left\{f (\mathbf {v}) + \frac {1}{2} \| \mathbf {v} - \mathbf {x} \| ^ {2} \right\},
$$

which, by Fermat’s optimality condition (Theorem 3.63) and the sum rule of subdiferential calculus (Theorem 3.40), is equivalent to the relation 

$$
\mathbf {0} \in \partial f (\mathbf {u}) + \mathbf {u} - \mathbf {x}.\tag{6.31}
$$

We have thus shown the equivalence between claims (i) and (ii). Finally, by the definition of the subgradient, the membership relation of claim (ii) is equivalent to (iii). 

A direct consequence of the second prox theorem is that for a proper closed and convex function, ${ \bf x } = \mathrm { p r o x } _ { f } ( { \bf x } )$ if and only x is a minimizer of $f .$ 

Corollary 6.40. Let f be a proper closed and convex function. Then x is a minimizer of f if and only if $\mathbf { x } = \mathrm { p r o x } _ { f } ( \mathbf { x } )$ 

Proof. x is a minimizer of f if and only if $\mathbf { 0 } ~ \in \partial f ( \mathbf { x } )$ , that is, if and only if$\mathbf { x } - \mathbf { x } \in \partial f ( \mathbf { x } )$ , which by the second prox theorem (equivalence between (i) and (ii))is the same as $\mathbf { x } = \mathrm { p r o x } _ { f } ( \mathbf { x } )$ 口

When $f = \delta _ { C }$ , with C being a nonempty closed and convex set, the equivalence between claims (i) and (iii) in the second prox theorem amounts to the second projection theorem. 

Theorem 6.41 (second projection theorem). Let $C \subseteq \mathbb { E }$ be a nonempty closed and convex set. Let $\mathbf { u } \in C$ . Then $\mathbf { u } = P _ { C } ( \mathbf { x } )$ if and only if 

$$
\left\langle \mathbf {x} - \mathbf {u}, \mathbf {y} - \mathbf {u} \right\rangle \leq 0 f o r a n y \mathbf {y} \in C.
$$

Another rather direct result of the second prox theorem is the firm nonexpansivity of the prox operator. 

Theorem 6.42 (firm nonexpansivity of the prox operator). Let f be a proper closed and convex function. Then for any $\mathbf { x } , \mathbf { y } \in \mathbb { E }$ 

(a) (firm nonexpansivity) 

$$
\langle \mathbf {x} - \mathbf {y}, \operatorname{prox} _ {f} (\mathbf {x}) - \operatorname{prox} _ {f} (\mathbf {y}) \rangle \geq \| \operatorname{prox} _ {f} (\mathbf {x}) - \operatorname{prox} _ {f} (\mathbf {y}) \| ^ {2}.
$$

(b) (nonexpansivity) 

$$
\left\| \operatorname{prox} _ {f} (\mathbf {x}) - \operatorname{prox} _ {f} (\mathbf {y}) \right\| \leq \left\| \mathbf {x} - \mathbf {y} \right\|.
$$

Proof. (a) Denoting $\mathbf { u } = \mathrm { p r o x } _ { f } ( \mathbf { x } ) , \mathbf { v } = \mathrm { p r o x } _ { f } ( \mathbf { y } )$ , by the equivalence of (i) and (ii) in the second prox theorem (Theorem 6.39), it follows that 

$$
\mathbf {x} - \mathbf {u} \in \partial f (\mathbf {u}), \mathbf {y} - \mathbf {v} \in \partial f (\mathbf {v}).
$$

Thus, by the subgradient inequality, 

$$
\begin{array}{l} f (\mathbf {v}) \geq f (\mathbf {u}) + \langle \mathbf {x} - \mathbf {u}, \mathbf {v} - \mathbf {u} \rangle , \\ f (\mathbf {u}) \geq f (\mathbf {v}) + \langle \mathbf {y} - \mathbf {v}, \mathbf {u} - \mathbf {v} \rangle . \end{array}
$$

Summing the above two inequalities, we obtain 

$$
0 \geq \langle \mathbf {y} - \mathbf {x} + \mathbf {u} - \mathbf {v}, \mathbf {u} - \mathbf {v} \rangle ,
$$

which is the same as 

$$
\left\langle \mathbf {x} - \mathbf {y}, \mathbf {u} - \mathbf {v} \right\rangle \geq \left\| \mathbf {u} - \mathbf {v} \right\| ^ {2},
$$

that is, 

$$
\langle \mathbf {x} - \mathbf {y}, \operatorname{prox} _ {f} (\mathbf {x}) - \operatorname{prox} _ {f} (\mathbf {y}) \rangle \geq \| \operatorname{prox} _ {f} (\mathbf {x}) - \operatorname{prox} _ {f} (\mathbf {y}) \| ^ {2}.
$$

(b) If prox ${ \bf \Lambda } _ { f } ( { \bf x } ) = \mathrm { p r o x } _ { f } ( { \bf y } )$ , then the inequality is obvious. Assume that prox $_ f ( \mathbf { x } ) \neq \mathrm { p r o x } _ { f } ( \mathbf { y } )$ . Using (a) and the Cauchy–Schwarz inequality, it follows that 

$$
\begin{array}{r l} & {\| \mathrm{prox} _ {f} (\mathbf {x}) - \mathrm{prox} _ {f} (\mathbf {y}) \| ^ {2} \leq \langle \mathrm{prox} _ {h} (\mathbf {x}) - \mathrm{prox} _ {h} (\mathbf {y}), \mathbf {x} - \mathbf {y} \rangle} \\ & {\qquad \leq \| \mathrm{prox} _ {h} (\mathbf {x}) - \mathrm{prox} _ {h} (\mathbf {y}) \| \cdot \| \mathbf {x} - \mathbf {y} \|.} \end{array}
$$

Dividing by $\| \mathrm { p r o x } _ { h } ( { \bf x } ) - \mathrm { p r o x } _ { h } ( { \bf y } ) \|$ , the desired result is established. 

The following result shows how to compute the prox of the distance function to a nonempty closed and convex set. The proof is heavily based on the second prox theorem. 

Lemma 6.43 (prox of the distance function). Let $C \subseteq \mathbb { E }$ be a nonempty, closed, and convex set. Let $\lambda > 0$ . Then for any $\mathbf { x } \in \mathbb { E }$ 

$$
\operatorname{prox} _ {\lambda d _ {C}} (\mathbf {x}) = \left\{ \begin{array}{l l} (1 - \theta) \mathbf {x} + \theta P _ {C} (\mathbf {x}), & d _ {C} (\mathbf {x}) > \lambda , \\ P _ {C} (\mathbf {x}), & d _ {C} (\mathbf {x}) \leq \lambda , \end{array} \right.\tag{6.32}
$$

where<sup>32</sup> 

$$
\theta = \frac {\lambda}{d _ {C} (\mathbf {x})}.\tag{6.33}
$$

Proof. Let $\mathbf { u } = \mathrm { p r o x } _ { \lambda d _ { C } } ( \mathbf { x } )$ . By the second prox theorem (Theorem 6.39), 

$$
\mathbf {x} - \mathbf {u} \in \lambda \partial d _ {C} (\mathbf {u}).\tag{6.34}
$$

We will split the analysis into two cases. 

Case I. u $\notin C .$ . By Example 3.49, (6.34) is the same as 

$$
\mathbf {x} - \mathbf {u} = \lambda \frac {\mathbf {u} - P _ {C} (\mathbf {u})}{d _ {C} (\mathbf {u})}.
$$

Denoting $\begin{array} { r } { \alpha = \frac { \lambda } { d _ { C } ( \mathbf { u } ) } } \end{array}$ , the last equality can be rewritten as 

$$
\mathbf {u} = \frac {1}{\alpha + 1} \mathbf {x} + \frac {\alpha}{\alpha + 1} P _ {C} (\mathbf {u})\tag{6.35}
$$

or as 

$$
\mathbf {x} - P _ {C} (\mathbf {u}) = (\alpha + 1) (\mathbf {u} - P _ {C} (\mathbf {u})).\tag{6.36}
$$

By the second projection theorem (Theorem 6.41), in order to show that $P _ { C } ( \mathbf { u } ) =$ $P _ { C } ( \mathbf { x } )$ , it is enough to show that 

$$
\langle \mathbf {x} - P _ {C} (\mathbf {u}), \mathbf {y} - P _ {C} (\mathbf {u}) \rangle \leq 0 \text {   for   any   } \mathbf {y} \in C.\tag{6.37}
$$

Using (6.36), we can deduce that (6.37) is equivalent to 

$$
(\alpha + 1) \langle \mathbf {u} - P _ {C} (\mathbf {u}), \mathbf {y} - P _ {C} (\mathbf {u}) \rangle \leq 0 \text {   for   any   } \mathbf {y} \in C,
$$

which is a valid inequality by the second projection theorem, and hence $P _ { C } ( \mathbf { u } ) =$ $P _ { C } ( \mathbf { x } )$ . Using this fact and taking the norm in both sides of (6.36), we obtain that 

$$
d _ {C} (\mathbf {x}) = (\alpha + 1) d _ {C} (\mathbf {u}) = d _ {C} (\mathbf {u}) + \lambda ,
$$

which also shows that in this case $d _ { C } ( \mathbf { x } ) > \lambda$ (since $d _ { C } ( { \bf u } ) > 0 )$ and that 

$$
{\frac {1}{\alpha + 1}} = {\frac {d _ {C} (\mathbf {u})}{\lambda + d _ {C} (\mathbf {u})}} = {\frac {d _ {C} (\mathbf {x}) - \lambda}{d _ {C} (\mathbf {x})}} = 1 - \theta ,
$$

where θ is given in (6.33). Therefore, (6.35) can also be written as (recalling also that $P _ { C } ( \mathbf { u } ) = P _ { C } ( \mathbf { x } ) )$ 

$$
\mathrm{prox} _ {\lambda d _ {C}} (\mathbf {x}) = (1 - \theta) \mathbf {x} + \theta P _ {C} (\mathbf {x}).\tag{6.38}
$$

Case II. If $\mathbf { u } \in C .$ , then $\mathbf { u } = P _ { C } ( \mathbf { x } )$ . To show this, let $\mathbf { v } \in C$ . Since $\mathbf { u } = \mathrm { p r o x } _ { \lambda d _ { C } } ( \mathbf { x } )$ 2 it follows in particular that 

$$
\lambda d _ {C} (\mathbf {u}) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \leq \lambda d _ {C} (\mathbf {v}) + \frac {1}{2} \| \mathbf {v} - \mathbf {x} \| ^ {2},
$$

and hence, since $d _ { C } ( \mathbf { u } ) = d _ { C } ( \mathbf { v } ) = 0$ 

$$
\left\| \mathbf {u} - \mathbf {x} \right\| \leq \left\| \mathbf {v} - \mathbf {x} \right\|.
$$

Therefore, 

$$
\mathbf {u} = \operatorname{argmin} _ {\mathbf {v} \in C} \| \mathbf {v} - \mathbf {x} \| = P _ {C} (\mathbf {x}).
$$

By Example 3.49, the optimality condition (6.34) becomes 

$$
\frac {\mathbf {x} - P _ {C} (\mathbf {x})}{\lambda} \in N _ {C} (\mathbf {u}) \cap B [ \mathbf {0}, 1 ],
$$

which in particular implies that 

$$
\left\| \frac {\mathbf {x} - P _ {C} (\mathbf {x})}{\lambda} \right\| \leq 1,
$$

that is, 

$$
d _ {C} (\mathbf {x}) = \left\| P _ {C} (\mathbf {x}) - \mathbf {x} \right\| \leq \lambda .
$$

Since the first case in which (6.38) holds corresponds to vectors satisfying $d _ { C } ( \mathbf { x } ) > \lambda _ { : }$ while the second case in which $\mathrm { p r o x } _ { \lambda d _ { C } } ( { \bf x } ) = P _ { C } ( { \bf x } )$ corresponds to vectors satisfying $d _ { C } ( \mathbf { x } ) \leq \lambda$ , the desired result (6.32) is established. 

## 6.6 Moreau Decomposition

A key property of the prox operator is the so-called Moreau decomposition theorem, which connects the prox operator of proper closed convex functions and their conjugates. 

Theorem 6.44 (Moreau decomposition). Let $f : \mathbb { E }  ( - \infty , \infty ]$ be proper closed and convex. Then for any $\mathbf { x } \in \mathbb { E }$ 

$$
\operatorname{prox} _ {f} (\mathbf {x}) + \operatorname{prox} _ {f *} (\mathbf {x}) = \mathbf {x}.
$$

Proof. Let $\mathbf { x } \in \mathbb { E }$ and denote $\mathbf { u } = \mathrm { p r o x } _ { f } ( \mathbf { x } )$ . Then by the equivalence between claims (i) and (ii) in the second prox theorem (Theorem 6.39), it follows that ${ \bf x } - { \bf u } \in$ $\partial f ( \mathbf { u } )$ , which by the conjugate subgradient theorem (Theorem 4.20) is equivalent to $\mathbf { u } \in \partial f ^ { * } ( \mathbf { x } - \mathbf { u } )$ . Using the second prox theorem again, we conclude that ${ \bf x } - { \bf u } =$ prox<sub>f∗</sub> (x). Therefore, 

$$
\operatorname{prox} _ {f} (\mathbf {x}) + \operatorname{prox} _ {f *} (\mathbf {x}) = \mathbf {u} + (\mathbf {x} - \mathbf {u}) = \mathbf {x}. \quad \square
$$

The next result is a useful extension of the Moreau decomposition theorem. 

Theorem 6.45 (extended Moreau decomposition). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be proper closed and convex, and let $\lambda > 0$ . Then for any $\mathbf { x } \in \mathbb { E }$ 8 

$$
\operatorname{prox} _ {\lambda f} (\mathbf {x}) + \lambda \operatorname{prox} _ {\lambda^ {- 1} f ^ {*}} (\mathbf {x} / \lambda) = \mathbf {x}.\tag{6.39}
$$

Proof. Using Moreau decomposition, for any $\mathbf { x } \in \mathbb { E }$ 

$$
\mathrm{prox} _ {\lambda f} (\mathbf {x}) = \mathbf {x} - \mathrm{prox} _ {(\lambda f) ^ {*}} (\mathbf {x}) = \mathbf {x} - \mathrm{prox} _ {\lambda f ^ {*} (. / \lambda)} (\mathbf {x}),\tag{6.40}
$$

where the second equality follows by Theorem $4 . 1 4 ( \mathrm { a } )$ . By Theorem 6.12, 

$$
\operatorname{prox} _ {\lambda f ^ {*} (\cdot / \lambda)} (\mathbf {x}) = \lambda \operatorname{prox} _ {\lambda^ {- 1} f ^ {*}} (\mathbf {x} / \lambda),
$$

which, combined with (6.40), yields (6.39). 

## 6.6.1 Support Functions

Using Moreau decomposition, we can develop a formula for computing the prox of a support function of a given nonempty closed and convex set in terms of the orthogonal projection operator. 

Theorem 6.46 (prox of support functions). Let $C \subseteq \mathbb { E }$ be a nonempty closed and convex set, and let $\lambda > 0$ . Then for any $\mathbf { x } \in \mathbb { E }$ 

$$
\mathrm{prox} _ {\lambda \sigma_ {C}} (\mathbf {x}) = \mathbf {x} - \lambda P _ {C} (\mathbf {x} / \lambda).\tag{6.41}
$$

Proof. A direct consequence of the extended Moreau decomposition formula (Theorem 6.45) along with the fact that $( \sigma _ { C } ) ^ { * } = \delta _ { C }$ (Example 4.9). 

Following are several examples of prox computations using formula (6.41). 

Example 6.47 (prox of norms). Let $f : \mathbb { E }  \mathbb { R }$ be given by $f ( \mathbf { x } ) = \lambda \| \mathbf { x } \| _ { \alpha } ,$ where $\lambda > 0$ and $\| \cdot \| _ { \alpha }$ is any norm on E. Note that $\| \cdot \| _ { \alpha }$ is not necessarily the endowed norm on E, which is denoted by $\| \cdot \|$ and in this chapter is always assumed to be the Euclidean norm. We know by Example 2.31 that 

$$
\left\| \mathbf {x} \right\| _ {\alpha} = \sigma_ {C} (\mathbf {x}),
$$

where 

$$
C = B _ {\| \cdot \| _ {\alpha , *}} [ \mathbf {0}, 1 ] = \left\{\mathbf {x} \in \mathbb {E}: \| \mathbf {x} \| _ {\alpha , *} \leq 1 \right\}
$$

with $\| \cdot \| _ { \alpha , * }$ being the dual norm of $\| \cdot \| _ { \alpha }$ . Invoking Theorem 6.46, we obtain 

$$
\operatorname{prox} _ {\lambda \| \cdot \| _ {\alpha}} (\mathbf {x}) = \mathbf {x} - \lambda P _ {B _ {\| \cdot \| _ {\alpha}, *} [ \mathbf {0}, 1 ]} (\mathbf {x} / \lambda).
$$

Example 6.48 (prox of $\scriptstyle l _ { \infty } \to \mathbf { n o r m } )$ . By Example 6.47 we have for all $\lambda > 0$ and $\mathbf { x } \in \mathbb { R } ^ { n }$ 

$$
\operatorname{prox} _ {\lambda \| \cdot \| _ {\infty}} (\mathbf {x}) = \mathbf {x} - \lambda P _ {B _ {\| \cdot \| _ {1}} [ \mathbf {0}, 1 ]} (\mathbf {x} / \lambda).
$$

The projection onto the $l _ { 1 }$ unit ball can be easily computed by finding a root of a nonincreasing one-dimensional function; see Example 6.33. 

Example 6.49 (prox of the max function). Consider the max function $g :$ $\mathbb { R } ^ { n } \to$ R given by $g ( \mathbf { x } ) = \operatorname* { m a x } ( \mathbf { x } ) \equiv \operatorname* { m a x } \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { n } \}$ . It is easy to see that the max function is actually the support function of the unit simplex: 

$$
\max (\mathbf {x}) = \sigma_ {\Delta_ {n}} (\mathbf {x}).
$$

Therefore, by Theorem 6.46, for any $\lambda > 0$ and $\mathbf { x } \in \mathbb { R } ^ { n }$ 2 

$$
\operatorname{prox} _ {\lambda \max (\cdot)} (\mathbf {x}) = \mathbf {x} - \lambda P _ {\Delta_ {n}} (\mathbf {x} / \lambda).
$$

The projection onto the unit simplex can be eficiently computed by finding a root of a nonincreasing one-dimensional function; see Corollary 6.29. 

Example 6.50 (prox of the sum-of-k-largest-values function). Let $f : \mathbb { R } ^ { n } $ R be given by 

$$
f (\mathbf {x}) = x _ {[ 1 ]} + x _ {[ 2 ]} + \dots + x _ {[ k ]},
$$

where $k \in \{ 1 , 2 , \dots , n \}$ and for any $i , x _ { [ i ] }$ denotes ith largest value in the vector x. It is not dificult to show that $f = \sigma _ { C }$ , where 

$$
C = \{\mathbf {y} \in \mathbb {R} ^ {n}: \mathbf {e} ^ {T} \mathbf {y} = k, \mathbf {0} \leq \mathbf {y} \leq \mathbf {e} \}.
$$

Therefore, by Theorem 6.46, 

$$
\operatorname{prox} _ {\lambda f} (\mathbf {x}) = \mathbf {x} - \lambda P _ {C} (\mathbf {x} / \lambda).
$$

That is, for any $\mathbf { x } \in \mathbb { R } ^ { n }$ 2 

$$
\operatorname{prox} _ {\lambda f} (\mathbf {x}) = \mathbf {x} - \lambda P _ {\{\mathbf {y}: \mathbf {e} ^ {T} \mathbf {y} = k, \mathbf {0} \leq \mathbf {y} \leq \mathbf {e} \}} (\mathbf {x} / \lambda).
$$

As in the previous examples, computing the projection onto $C$ amounts to finding a root of a monotone one-dimensional function; see Theorem 6.27. 

Example 6.51 (prox of the sum-of-k-largest-absolute-values function). Let $f : \mathbb { R } ^ { n }  \mathbb { R }$ be given by 

$$
f (\mathbf {x}) = \sum_ {i = 1} ^ {k} | x _ {\langle i \rangle} |,
$$

where $k \in \{ 1 , 2 , \ldots , n \}$ and $x _ { \langle i \rangle }$ is the component of x with the ith largest absolute value, meaning in particular that $| x _ { \langle 1 \rangle } | \geq | x _ { \langle 2 \rangle } | \geq \cdot \cdot \cdot \geq | x _ { \langle n \rangle } |$ . Then 

$$
f (\mathbf {x}) = \max \left\{\sum_ {i = 1} ^ {n} z _ {i} x _ {i}: \| \mathbf {z} \| _ {1} \leq k, - \mathbf {e} \leq \mathbf {z} \leq \mathbf {e} \right\}.
$$

Therefore, $f = \sigma _ { C }$ , where 

$$
C = \left\{\mathbf {z} \in \mathbb {R} ^ {n}: \| \mathbf {z} \| _ {1} \leq k, - \mathbf {e} \leq \mathbf {z} \leq \mathbf {e} \right\},
$$

and consequently, by Theorem 6.46, 

$$
\operatorname{prox} _ {\lambda f} (\mathbf {x}) = \mathbf {x} - \lambda P _ {C} (\mathbf {x} / \lambda).
$$

That is, for any $\mathbf { x } \in \mathbb { R } ^ { n }$ 

$$
\operatorname{prox} _ {\lambda f} (\mathbf {x}) = \mathbf {x} - \lambda P _ {\{\mathbf {y}: \| \mathbf {y} \| _ {1} \leq k, - \mathbf {e} \leq \mathbf {y} \leq \mathbf {e} \}} (\mathbf {x} / \lambda).
$$

The orthogonal projection in the above formula amounts to finding a root of a nonincreasing one-dimensional function; see Example 6.34. 

## 6.7 The Moreau Envelope

## 6.7.1 Definition and Basic Properties

Definition 6.52 (Moreau envelope). Given a proper closed convex function $f : \mathbb { E } \to ( - \infty , \infty ]$ and $\mu > 0$ , the Moreau envelope of f is the function 

$$
M _ {f} ^ {\mu} (\mathbf {x}) = \min _ {\mathbf {u} \in \mathbb {E}} \left\{f (\mathbf {u}) + \frac {1}{2 \mu} \| \mathbf {x} - \mathbf {u} \| ^ {2} \right\}.\tag{6.42}
$$

The parameter $\mu$ is called the smoothing parameter. The explanation for this terminology will be given in Section 6.7.2. By the first prox theorem (Theorem 6.3), the minimization problem in (6.42) has a unique solution, given by $\operatorname { p r o x } _ { \mu f } ( \mathbf { x } )$ Therefore, ${ \cal M } _ { f } ^ { \mu } ( { \bf x } )$ is always a real number and 

$$
M _ {f} ^ {\mu} (\mathbf {x}) = f (\mathrm{prox} _ {\mu f} (\mathbf {x})) + \frac {1}{2 \mu} \| \mathbf {x} - \mathrm{prox} _ {\mu f} (\mathbf {x}) \| ^ {2}.
$$

Example 6.53 (Moreau envelope of indicators). Let $f = \delta _ { C }$ , where $C \subseteq \mathbb { E }$ is a nonempty closed and convex set. By Theorem 6.24, $\mathrm { p r o x } _ { \mu f } ( { \bf x } ) = P _ { C } ( { \bf x } )$ . Thus, for any $\mathbf { x } \in \mathbb { E }$ 

$$
M _ {f} ^ {\mu} (\mathbf {x}) = \delta_ {C} (P _ {C} (\mathbf {x})) + \frac {1}{2 \mu} \| \mathbf {x} - P _ {C} (\mathbf {x})) \| ^ {2},
$$

and hence 

$$
M _ {\delta_ {C}} ^ {\mu} = \frac {1}{2 \mu} d _ {C} ^ {2}.
$$

The next example will show that the Moreau envelope of the (Euclidean) norm is the so-called Huber function defined as 

$$
H _ {\mu} (\mathbf {x}) = \left\{ \begin{array}{l l} \frac {1}{2 \mu} \| \mathbf {x} \| ^ {2}, & \| \mathbf {x} \| \leq \mu , \\ \| \mathbf {x} \| - \frac {\mu}{2}, & \| \mathbf {x} \| > \mu . \end{array} \right.\tag{6.43}
$$

The one-dimensional Huber function is plotted in Figure 6.4, where it is illustrated that the function becomes smoother as $\mu$ becomes larger. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-14/06afcb51-43be-4db1-8b62-226fc4277ca3/e084fe01e6a3b992c7c87fd7072c04fd1918486a33ebd1ccd15982f198219dd9.jpg)



Figure 6.4. The Huber function with parameters $\mu = 0 . 1 , 1 , 4$ . The function becomes smoother as $\mu$ gets larger.


Example 6.54 (Huber function). Let $f : \mathbb { E } \to \mathbb { R }$ be given by $f ( \mathbf { x } ) = \| \mathbf { x } \|$ . Then by Example 6.19, for any $\mathbf { x } \in \mathbb { E }$ and $\mu > 0$ 

$$
\mathrm{prox} _ {\mu f} (\mathbf {x}) = \left(1 - \frac {\mu}{\max \{\| \mathbf {x} \| , \mu \}}\right) \mathbf {x}.
$$

Therefore, 

$$
M _ {f} ^ {\mu} (\mathbf {x}) = \| \mathrm{prox} _ {\mu f} (\mathbf {x}) \| + \frac {1}{2 \mu} \| \mathbf {x} - \mathrm{prox} _ {\mu f} (\mathbf {x}) \| ^ {2} = \left\{ \begin{array}{l l} \frac {1}{2 \mu} \| \mathbf {x} \| ^ {2}, & \| \mathbf {x} \| \leq \mu , \\ \| \mathbf {x} \| - \frac {\mu}{2}, & \| \mathbf {x} \| > \mu . \end{array} \right.
$$

Thus, for any $\mu > 0$ 

$$
M _ {\| \cdot \|} ^ {\mu} = H _ {\mu}.
$$

Note that the Moreau envelope function is actually a result of an infimal convolution operation between the function $f$ and the function 

$$
\omega_ {\mu} (\mathbf {x}) = \frac {1}{2 \mu} \| \mathbf {x} \| ^ {2}.\tag{6.44}
$$

That is, 

$$
M _ {f} ^ {\mu} = f \Box \omega_ {\mu}.
$$

One consequence of this observation is that by Theorem 2.19, if $f$ is a proper closed<sup>33</sup> and convex function, then $M _ { f } ^ { \mu }$ is convex. We summarize the above discussion in the following theorem. 

Theorem 6.55. Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper closed and convex function, and let $\omega _ { \mu }$ be given in (6.44), where $\mu > 0$ . Then 

(a) ${ \cal M } _ { f } ^ { \mu } = f \sqcup \omega _ { \mu }$ 

(b) $M _ { f } ^ { \mu } : \mathbb { E }  \mathbb { R }$ is real-valued and convex. 

We can immediately conclude from Theorem $6 . 5 5 ( \mathrm { a } )$ along with the formula for the conjugate of the infimal convolution (Theorem 4.16) an expression for the conjugate of the Moreau envelope. 

Corollary 6.56. Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper closed and convex function and let $\omega _ { \mu }$ be given in (6.44), where $\mu > 0$ . Then 

$$
(M _ {f} ^ {\mu}) ^ {*} = f ^ {*} + \omega_ {\frac {1}{\mu}}.
$$

Another useful algebraic property of the Moreau envelope is described in the following result. 

Lemma 6.57. Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper closed and convex function, and let $\lambda , \mu > 0$ . Then for any $\mathbf { x } \in \mathbb { E }$ 

$$
\lambda M _ {f} ^ {\mu} (\mathbf {x}) = M _ {\lambda f} ^ {\mu / \lambda} (\mathbf {x}).\tag{6.45}
$$

Proof. For any $\mathbf { x } \in \mathbb { E }$ 

$$
\begin{array}{l} \lambda M _ {f} ^ {\mu} (\mathbf {x}) = \lambda \min _ {\mathbf {u}} \left\{f (\mathbf {u}) + \frac {1}{2 \mu} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\} \\ = \min _ {\mathbf {u}} \left\{\lambda f (\mathbf {u}) + \frac {1}{2 \mu / \lambda} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\} \\ = M _ {\lambda f} ^ {\mu / \lambda} (\mathbf {x}). \quad \square \end{array}
$$

A simple calculus rule states that the Moreau envelope of a separable sum of functions is the sum of the corresponding Moreau envelopes. 

Theorem 6.58 (Moreau envelope of separable functions). Suppose that $\mathbb { E } = \mathbb { E } _ { 1 } \times \mathbb { E } _ { 2 } \times \cdot \cdot \cdot \times \mathbb { E } _ { m }$ , and let $f : \mathbb { E } \to ( - \infty , \infty ]$ be given by 

$$
f \left(\mathbf {x} _ {1}, \mathbf {x} _ {2}, \dots , \mathbf {x} _ {m}\right) = \sum_ {i = 1} ^ {m} f _ {i} \left(\mathbf {x} _ {i}\right), \quad \mathbf {x} _ {1} \in \mathbb {E} _ {1}, \mathbf {x} _ {2} \in \mathbb {E} _ {2}, \dots , \mathbf {x} _ {m} \in \mathbb {E} _ {m},
$$

with $f _ { i } : \mathbb { E } _ { i }  ( - \infty , \infty ]$ being a proper closed and convex function for any $i .$ . Then given $\mu > 0$ , for any $\mathbf { x } _ { 1 } \in \mathbb { E } _ { 1 } , \mathbf { x } _ { 2 } \in \mathbb { E } _ { 2 } , \ldots , \mathbf { x } _ { m } \in \mathbb { E } _ { m }$ 

$$
M _ {f} ^ {\mu} (\mathbf {x} _ {1}, \mathbf {x} _ {2}, \dots , \mathbf {x} _ {m}) = \sum_ {i = 1} ^ {m} M _ {f _ {i}} ^ {\mu} (\mathbf {x} _ {i}).
$$

Proof. For any $\mathbf { x } _ { 1 } \in \mathbb { E } _ { 1 } , \mathbf { x } _ { 2 } \in \mathbb { E } _ { 2 } , \ldots , \mathbf { x } _ { m } \in \mathbb { E } _ { m }$ , denoting $\mathbf { x } = \left( \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { m } \right)$ , we have 

$$
\begin{array}{l} M _ {f} ^ {\mu} (\mathbf {x}) = \min _ {\mathbf {u} _ {i} \in \mathbb {E} _ {i}, i = 1, 2, \ldots , m} \left\{f (\mathbf {u} _ {1}, \mathbf {u} _ {2}, \ldots , \mathbf {u} _ {m}) + \frac {1}{2 \mu} \| (\mathbf {u} _ {1}, \mathbf {u} _ {2}, \ldots , \mathbf {u} _ {m}) - \mathbf {x} \| ^ {2} \right\} \\ = \min _ {\mathbf {u} _ {i} \in \mathbb {E} _ {i}, i = 1, 2, \ldots , m} \left\{\sum_ {i = 1} ^ {m} f _ {i} (\mathbf {u} _ {i}) + \frac {1}{2 \mu} \sum_ {i = 1} ^ {m} \| \mathbf {u} _ {i} - \mathbf {x} _ {i} \| ^ {2} \right\} \\ = \sum_ {i = 1} ^ {m} \min _ {\mathbf {u} _ {i} \in \mathbb {E} _ {i}} \left\{f _ {i} (\mathbf {u} _ {i}) + \frac {1}{2 \mu} \| \mathbf {u} _ {i} - \mathbf {x} _ {i} \| ^ {2} \right\} \\ = \sum_ {i = 1} ^ {m} M _ {f _ {i}} ^ {\mu} (\mathbf {x} _ {i}). \quad \square \end{array}
$$

Example 6.59 (Moreau envelope of the $l _ { 1 } { \bf - n o r m ) }$ . Consider the function $f$ : $\mathbb { R } ^ { n } \to \mathbb { R }$ given by $f ( \mathbf { x } ) = \| \mathbf { x } \| _ { 1 }$ . Note that 

$$
f (\mathbf {x}) = \| \mathbf {x} \| _ {1} = \sum_ {i = 1} ^ {n} g (x _ {i}),
$$

where $g ( t ) = | t |$ . By Example 6.54, ${ \cal M } _ { g } ^ { \mu } = { \cal H } _ { \mu }$ . Thus, invoking Theorem 6.58, we obtain that for any $\mathbf { x } \in \mathbb { R } ^ { n }$ 

$$
M _ {f} ^ {\mu} (\mathbf {x}) = \sum_ {i = 1} ^ {n} M _ {g} ^ {\mu} (x _ {i}) = \sum_ {i = 1} ^ {n} H _ {\mu} (x _ {i}).
$$

## 6.7.2 Diferentiability of the Moreau Envelope

The main diferentiability properties of the Moreau envelope function are stated in the next result. 

Theorem 6.60 (smoothness of the Moreau envelope). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper closed and convex function. Let $\mu > 0$ . Then $\dot { M } _ { f } ^ { \mu }$ is $\frac { 1 } { \mu } - s m o o t h$ over $\mathbb { E } _ { : }$ and for any $\mathbf { x } \in \mathbb { E }$ , 

$$
\nabla M _ {f} ^ {\mu} (\mathbf {x}) = \frac {1}{\mu} \left(\mathbf {x} - \mathrm{prox} _ {\mu f} (\mathbf {x})\right).
$$

Proof. By Theorem $6 . 5 5 ( \mathrm { a } ) , M _ { f } ^ { \mu } = f \sqcup \omega _ { \mu } .$ where $\begin{array} { r } { \omega _ { \mu } = \frac { 1 } { 2 \mu } \| \cdot \| ^ { 2 } } \end{array}$ . We can therefore invoke Theorem 5.30, whose assumptions are satisfied (taking $\omega = \omega _ { \mu }$ and $\begin{array} { r } { L = \frac { 1 } { \mu } ) } \end{array}$ and conclude that $M _ { f } ^ { \mu }$ is $\textstyle { \frac { 1 } { \mu } }$ -smooth. In addition, since 

$$
\operatorname{prox} _ {\mu f} (\mathbf {x}) = \operatorname{argmin} _ {\mathbf {u} \in \mathbb {E}} \left\{f (\mathbf {u}) + \frac {1}{2 \mu} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\},
$$

it follows that the vector $\mathbf { u } ( \mathbf { x } )$ defined in Theorem 5.30 is equal to $\operatorname { p r o x } _ { \mu f } ( \mathbf { x } )$ and that 

$$
\nabla M _ {f} ^ {\mu} (\mathbf {x}) = \nabla \omega_ {\mu} (\mathbf {x} - \mathbf {u} (\mathbf {x})) = \frac {1}{\mu} (\mathbf {x} - \mathrm{prox} _ {\mu f} (\mathbf {x})).
$$

Example 6.61 (1-smoothness of $\scriptstyle { \frac { 1 } { 2 } } d _ { C } ^ { 2 } )$ . Let $C \subseteq \mathbb { E }$ be a nonempty closed and convex set. Recall that by Example 6.53, $\begin{array} { r } { \frac { 1 } { 2 } d _ { C } ^ { 2 } = M _ { \delta _ { C } } ^ { 1 } } \end{array}$ . Then by Theorem 6.60, $\textstyle { \frac { 1 } { 2 } } d _ { C } ^ { 2 }$ is 1-smooth and 

$$
\nabla \left(\frac {1}{2} d _ {C} ^ {2}\right) (\mathbf {x}) = \mathbf {x} - \mathrm{prox} _ {\delta_ {C}} (\mathbf {x}) = \mathbf {x} - P _ {C} (\mathbf {x}).
$$

Note that the above expression for the gradient was already derived in Example 3.31 and that the 1-smoothness of $\textstyle { \frac { 1 } { 2 } } d _ { C } ^ { 2 }$ was already established twice in Examples 5.5 and 5.31. ■ 

Example 6.62 (smoothness of the Huber function). Recall that the Huber function is given by 

$$
H _ {\mu} (\mathbf {x}) = \left\{ \begin{array}{l l} \frac {1}{2 \mu} \| \mathbf {x} \| ^ {2}, & \| \mathbf {x} \| \leq \mu , \\ \| \mathbf {x} \| - \frac {\mu}{2}, & \| \mathbf {x} \| > \mu . \end{array} \right.
$$

By Example 6.54, $H _ { \mu } = M _ { f } ^ { \mu }$ , where $f ( \mathbf { x } ) = \| \mathbf { x } \|$ . Then, by Theorem 6.60, $H _ { \mu }$ is $\textstyle { \frac { 1 } { \mu } }$ -smooth and 

$$
\begin{array}{l} \nabla H _ {\mu} (\mathbf {x}) = \frac {1}{\mu} \left(\mathbf {x} - \operatorname{prox} _ {\mu f} (\mathbf {x})\right) \\ \stackrel {{(*)}} {{=}} \frac {1}{\mu} \left(\mathbf {x} - \left(1 - \frac {\mu}{\max \{\| \mathbf {x} \| , \mu \}}\right) \mathbf {x}\right) \\ = \left\{ \begin{array}{l l} \frac {1}{\mu} \mathbf {x}, & \| \mathbf {x} \| \leq \mu , \\ \frac {\mathbf {x}}{\| \mathbf {x} \|}, & \| \mathbf {x} \| > \mu , \end{array} \right. \end{array}
$$

where the equality (∗) uses the expression for $\operatorname { p r o x } _ { \mu f }$ developed in Example 6.19. ■ 

## 6.7.3 Prox of the Moreau Envelope

An interesting and important result states that if we can compute the prox of a proper closed and convex function $f _ { : }$ then we can also compute the prox of its Moreau envelope. 

Theorem 6.63 (prox of Moreau envelope). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper closed and convex function, and let $\mu > 0$ . Then for any $\mathbf { x } \in \mathbb { E }$ 2 

$$
\operatorname{prox} _ {M _ {f} ^ {\mu}} (\mathbf {x}) = \mathbf {x} + \frac {1}{\mu + 1} \left(\operatorname{prox} _ {(\mu + 1) f} (\mathbf {x}) - \mathbf {x}\right).
$$

Proof. First note that 

$$
\min _ {\mathbf {u}} \left\{M _ {f} ^ {\mu} (\mathbf {u}) + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\} = \min _ {\mathbf {u}} \min _ {\mathbf {y}} \left\{f (\mathbf {y}) + \frac {1}{2 \mu} \| \mathbf {u} - \mathbf {y} \| ^ {2} + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\}.\tag{6.46}
$$

Exchanging the order of minimizations, we obtain the following problem: 

$$
\min _ {\mathbf {y}} \min _ {\mathbf {u}} \left\{f (\mathbf {y}) + \frac {1}{2 \mu} \| \mathbf {u} - \mathbf {y} \| ^ {2} + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| ^ {2} \right\}.\tag{6.47}
$$

The optimal solution of the inner minimization problem in u is attained when the gradient w.r.t. u vanishes: 

$$
\frac {1}{\mu} (\mathbf {u} - \mathbf {y}) + (\mathbf {u} - \mathbf {x}) = \mathbf {0},
$$

that is, when 

$$
\mathbf {u} = \mathbf {u} _ {\mu} \equiv \frac {\mu \mathbf {x} + \mathbf {y}}{\mu + 1}.\tag{6.48}
$$

Therefore, the optimal value of the inner minimization problem in (6.47) is 

$$
\begin{array}{c} f (\mathbf {y}) + \frac {1}{2 \mu} \| \mathbf {u} _ {\mu} - \mathbf {y} \| ^ {2} + \frac {1}{2} \| \mathbf {u} _ {\mu} - \mathbf {x} \| ^ {2} = f (\mathbf {y}) + \frac {1}{2 \mu} \left\| \frac {\mu \mathbf {x} - \mu \mathbf {y}}{\mu + 1} \right\| ^ {2} + \frac {1}{2} \left\| \frac {\mathbf {y} - \mathbf {x}}{\mu + 1} \right\| ^ {2} \\ = f (\mathbf {y}) + \frac {1}{2 (\mu + 1)} \| \mathbf {x} - \mathbf {y} \| ^ {2}. \end{array}
$$

Therefore, the optimal solution of (6.46) is given by (6.48), where y is the solution of 

$$
\min _ {\mathbf {y}} \left\{f (\mathbf {y}) + \frac {1}{2 (\mu + 1)} \| \mathbf {x} - \mathbf {y} \| ^ {2} \right\},
$$

that is, $\mathbf { y } = \operatorname { p r o x } _ { ( \mu + 1 ) f } ( \mathbf { x } )$ . To summarize, 

$$
\operatorname{prox} _ {M _ {f} ^ {\mu}} (\mathbf {x}) = \frac {1}{\mu + 1} \left(\mu \mathbf {x} + \operatorname{prox} _ {(\mu + 1) f} (\mathbf {x})\right). \quad \square
$$

Combining Theorem 6.63 with Lemma 6.57 leads to the following corollary. 

Corollary 6.64. Let $f : \mathbb { E }  ( - \infty , \infty ]$ be a proper closed and convex function, and let $\lambda , \mu > 0$ . Then for any $\mathbf { x } \in \mathbb { E }$ 

$$
\operatorname{prox} _ {\lambda M _ {f} ^ {\mu}} (\mathbf {x}) = \mathbf {x} + \frac {\lambda}{\mu + \lambda} \left(\operatorname{prox} _ {(\mu + \lambda) f} (\mathbf {x}) - \mathbf {x}\right).
$$

$$
\text {Proof.} \operatorname{prox} _ {\lambda M _ {f} ^ {\mu}} (\mathbf {x}) = \operatorname{prox} _ {M _ {\lambda f} ^ {\mu / \lambda}} (\mathbf {x}) = \mathbf {x} + \frac {\lambda}{\mu + \lambda} \left(\operatorname{prox} _ {(\mu + \lambda) f} (\mathbf {x}) - \mathbf {x}\right). \quad \square
$$

Example 6.65 (prox of $\scriptstyle { \frac { \lambda } { 2 } } d _ { C } ^ { 2 } )$ . Let $C \subseteq \mathbb { E }$ be a nonempty closed and convex set, and let $\lambda > 0$ . Consider the function $\begin{array} { r } { f = \frac { 1 } { 2 } d _ { C } ^ { 2 } } \end{array}$ . Then, by Example 6.53, $f = M _ { g } ^ { 1 }$ where $g = \delta _ { C }$ . Recall that $\mathrm { p r o x } _ { g } = P _ { C }$ . Therefore, invoking Corollary 6.64, we obtain that for any $\mathbf { x } \in \mathbb { E }$ 44 

$$
\operatorname{prox} _ {\lambda f} (\mathbf {x}) = \operatorname{prox} _ {\lambda M _ {g} ^ {1}} (\mathbf {x}) = \mathbf {x} + \frac {\lambda}{\lambda + 1} \left(\operatorname{prox} _ {(\lambda + 1) g} (\mathbf {x}) - \mathbf {x}\right) = \mathbf {x} + \frac {\lambda}{\lambda + 1} \left(P _ {C} (\mathbf {x}) - \mathbf {x}\right).
$$

To conclude, 

$$
\mathrm{prox} _ {\frac {\lambda}{2} d _ {C} ^ {2}} (\mathbf {x}) = \frac {\lambda}{\lambda + 1} P _ {C} (\mathbf {x}) + \frac {1}{\lambda + 1} \mathbf {x}.
$$

Example 6.66 (prox of the Huber function). Consider the function 

$$
f (\mathbf {x}) = \lambda H _ {\mu} (\mathbf {x}),
$$

where $H _ { \mu }$ is the Huber function with a smoothing parameter $\mu > 0$ given in (6.43). By Example 6.54, $H _ { \mu } = M _ { q } ^ { \mu }$ , where $g ( \mathbf { x } ) = \| \mathbf { x } \|$ . Therefore, by Corollary 6.64, it follows that for any $\lambda > 0$ and $\mathbf { x } \in \mathbb { E }$ (recalling the expression for the prox of the Euclidean norm derived in Example 6.19), 

$$
\begin{array}{r} \mathrm{prox} _ {\lambda H _ {\mu}} (\mathbf {x}) = \mathrm{prox} _ {\lambda M _ {g} ^ {\mu}} (\mathbf {x}) = \mathbf {x} + \frac {\lambda}{\mu + \lambda} \left(\mathrm{prox} _ {(\mu + \lambda) g} (\mathbf {x}) - \mathbf {x}\right) \\ = \mathbf {x} + \frac {\lambda}{\mu + \lambda} \left(\left(1 - \frac {\mu + \lambda}{\max \{\| \mathbf {x} \| , \mu + \lambda \}}\right) \mathbf {x} - \mathbf {x}\right), \end{array}
$$

which, after some algebraic cancellations, reduces to 

$$
\operatorname{prox} _ {\lambda H _ {\mu}} (\mathbf {x}) = \left(1 - \frac {\lambda}{\max \{\| \mathbf {x} \| , \mu + \lambda \}}\right) \mathbf {x}.
$$

Similarly to the Moreau decomposition formula for the prox operator (Theorem 6.45), we can obtain a decomposition formula for the Moreau envelope function. 

Theorem 6.67 (Moreau envelope decomposition). Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a proper closed and convex function, and let $\mu > 0$ . Then for any $\mathbf { x } \in \mathbb { E }$ 

$$
M _ {f} ^ {\mu} (\mathbf {x}) + M _ {f ^ {*}} ^ {1 / \mu} (\mathbf {x} / \mu) = \frac {1}{2 \mu} \| \mathbf {x} \| ^ {2}.
$$

Proof. Recall that for any $\mathbf { x } \in \mathbb { E }$ 

$$
M _ {f} ^ {\mu} (\mathbf {x}) = \min _ {\mathbf {u} \in \mathbb {E}} \left\{f (\mathbf {u}) + \psi (\mathbf {u}) \right\},
$$

where $\begin{array} { r } { \psi ( \mathbf { u } ) \equiv \frac { 1 } { 2 \mu } \| \mathbf { u } - \mathbf { x } \| ^ { 2 } } \end{array}$ . By Fenchel’s duality theorem (Theorem 4.15), we have 

$$
M _ {f} ^ {\mu} (\mathbf {x}) = \max _ {\mathbf {v} \in \mathbb {E}} \left\{- f ^ {*} (\mathbf {v}) - \psi^ {*} (- \mathbf {v}) \right\} = - \min _ {\mathbf {v} \in \mathbb {E}} \left\{f ^ {*} (\mathbf {v}) + \psi^ {*} (- \mathbf {v}) \right\}.
$$

Denote $\begin{array} { r } { \phi ( \cdot ) = \frac { 1 } { 2 } \| \cdot - \mathbf { x } \| ^ { 2 } } \end{array}$ . Then 

$$
\phi^ {*} (\mathbf {v}) = \frac {1}{2} \| \mathbf {v} \| ^ {2} + \langle \mathbf {x}, \mathbf {v} \rangle .
$$

Since $\begin{array} { r } { \psi = \frac { 1 } { \mu } \phi } \end{array}$ , it follows by Theorem 4.14 that 

$$
\psi^ {*} (\mathbf {v}) = \frac {1}{\mu} \phi^ {*} (\mu \mathbf {v}) = \frac {\mu}{2} \| \mathbf {v} \| ^ {2} + \langle \mathbf {x}, \mathbf {v} \rangle .
$$

Therefore, 

$$
M _ {f} ^ {\mu} (\mathbf {x}) = - \min _ {\mathbf {v} \in \mathbb {E}} \left\{f ^ {*} (\mathbf {v}) + \frac {\mu}{2} \| \mathbf {v} \| ^ {2} - \langle \mathbf {x}, \mathbf {v} \rangle \right\},
$$

and hence 

$$
M _ {f} ^ {\mu} (\mathbf {x}) = - \min _ {\mathbf {v} \in \mathbb {E}} \left\{f ^ {*} (\mathbf {v}) + \frac {\mu}{2} \| \mathbf {v} - \mathbf {x} / \mu \| ^ {2} - \frac {1}{2 \mu} \| \mathbf {x} \| ^ {2} \right\} = \frac {1}{2 \mu} \| \mathbf {x} \| ^ {2} - M _ {f ^ {*}} ^ {1 / \mu} (\mathbf {x} / \mu),
$$

establishing the desired result. 

## 6.8 Miscellaneous Prox Computations

In this section we gather several examples of prox computations that are not linked to any specific result established in this chapter. 

## 6.8.1 Norm of a Linear Transformation over $\mathbb { R } ^ { n }$

Lemma 6.68. Let $f : \mathbb { R } ^ { n }  \mathbb { R }$ be given by $f ( \mathbf { x } ) = \| \mathbf { A x } \| _ { 2 }$ , where $\mathbf { A } \in \mathbb { R } ^ { m \times n }$ is with full row rank, and $l e t \lambda > 0$ . Then 

$$
\operatorname{prox} _ {\lambda f} (\mathbf {x}) = \left\{ \begin{array}{l l} \mathbf {x} - \mathbf {A} ^ {T} (\mathbf {A A} ^ {T}) ^ {- 1} \mathbf {A x}, & \| (\mathbf {A A} ^ {T}) ^ {- 1} \mathbf {A x} \| _ {2} \leq \lambda , \\ \mathbf {x} - \mathbf {A} ^ {T} (\mathbf {A A} ^ {T} + \alpha^ {*} \mathbf {I}) ^ {- 1} \mathbf {A x}, & \| (\mathbf {A A} ^ {T}) ^ {- 1} \mathbf {A x} \| _ {2} > \lambda , \end{array} \right.
$$

where $\alpha ^ { * }$ is the unique positive root of the decreasing function 

$$
g (\alpha) = \| (\mathbf {A A} ^ {T} + \alpha \mathbf {I}) ^ {- 1} \mathbf {A x} \| _ {2} ^ {2} - \lambda^ {2}.
$$

Proof. The vector $\mathrm { p r o x } _ { \lambda f } ( \mathbf { x } )$ is the unique optimal solution to 

$$
\min _ {\mathbf {u} \in \mathbb {R} ^ {n}} \left\{\lambda \| \mathbf {A u} \| _ {2} + \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| _ {2} ^ {2} \right\},
$$

which can also be rewritten as 

$$
\min _ {\mathbf {u} \in \mathbb {R} ^ {n}, \mathbf {z} \in \mathbb {R} ^ {m}} \left\{\frac {1}{2} \| \mathbf {u} - \mathbf {x} \| _ {2} ^ {2} + \lambda \| \mathbf {z} \| _ {2}: \mathbf {z} = \mathbf {A u} \right\}.\tag{6.49}
$$

To construct a Lagrangian dual problem, we first form the Lagrangian: 

$$
\begin{array}{l} L (\mathbf {u}, \mathbf {z}; \mathbf {y}) = \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| _ {2} ^ {2} + \lambda \| \mathbf {z} \| _ {2} + \mathbf {y} ^ {T} (\mathbf {z} - \mathbf {A u}) \\ \qquad = \left[ \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| _ {2} ^ {2} - (\mathbf {A} ^ {T} \mathbf {y}) ^ {T} \mathbf {u} \right] + \left[ \lambda \| \mathbf {z} \| _ {2} + \mathbf {y} ^ {T} \mathbf {z} \right]. \end{array}
$$

Since the Lagrangian is separable w.r.t. u and $\mathbf { z } ,$ the dual objective function can be rewritten as 

$$
\min _ {\mathbf {u}, \mathbf {z}} L (\mathbf {u}, \mathbf {z}; \mathbf {y}) = \min _ {\mathbf {u}} \left[ \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| _ {2} ^ {2} - (\mathbf {A} ^ {T} \mathbf {y}) ^ {T} \mathbf {u} \right] + \min _ {\mathbf {z}} \left[ \lambda \| \mathbf {z} \| _ {2} + \mathbf {y} ^ {T} \mathbf {z} \right].\tag{6.50}
$$

The minimizer of the minimization problem in u is $\tilde { \mathbf { u } } = \mathbf { x } + \mathbf { A } ^ { T } \mathbf { y }$ with a corresponding optimal value of 

$$
\begin{array}{c} \min _ {\mathbf {u}} \left[ \frac {1}{2} \| \mathbf {u} - \mathbf {x} \| _ {2} ^ {2} - (\mathbf {A} ^ {T} \mathbf {y}) ^ {T} \mathbf {u} \right] = \frac {1}{2} \| \tilde {\mathbf {u}} - \mathbf {x} \| _ {2} ^ {2} - (\mathbf {A} ^ {T} \mathbf {y}) ^ {T} \tilde {\mathbf {u}} \\ = - \frac {1}{2} \mathbf {y} ^ {T} \mathbf {A} \mathbf {A} ^ {T} \mathbf {y} - (\mathbf {A x}) ^ {T} \mathbf {y}. \end{array}\tag{6.51}
$$

As for the second minimization problem, note that 

$$
\min _ {\mathbf {z}} \left[ \lambda \| \mathbf {z} \| _ {2} + \mathbf {y} ^ {T} \mathbf {z} \right] = - \max _ {\mathbf {z}} [ (- \mathbf {y}) ^ {T} \mathbf {z} - \lambda \| \mathbf {z} \| _ {2} ] = - g ^ {*} (- \mathbf {y}),
$$

where $g ( \cdot ) = \lambda \| \cdot \| _ { 2 }$ . Since $g ^ { * } ( \mathbf { w } ) = \lambda \delta _ { B _ { \| \cdot \| _ { 2 } } [ \mathbf { 0 } , 1 ] } ( \mathbf { w } / \lambda ) = \delta _ { B _ { \| \cdot \| _ { 2 } } [ \mathbf { 0 } , \lambda ] }$ (see Section 4.4.12 and Theorem 4.14), we can conclude that 

$$
\min _ {\mathbf {z}} \left[ \lambda \| \mathbf {z} \| _ {2} + \mathbf {y} ^ {T} \mathbf {z} \right] = \left\{ \begin{array}{l l} 0, & \| \mathbf {y} \| _ {2} \leq \lambda , \\ - \infty , & \| \mathbf {y} \| _ {2} > \lambda . \end{array} \right.
$$

Combining this with (6.51), we obtain the following dual problem: 

$$
\max _ {\mathbf {y} \in \mathbb {R} ^ {m}} \left\{- \frac {1}{2} \mathbf {y} ^ {T} \mathbf {A} \mathbf {A} ^ {T} \mathbf {y} - (\mathbf {A x}) ^ {T} \mathbf {y}: \| \mathbf {y} \| _ {2} \leq \lambda \right\}.\tag{6.52}
$$

Note that strong duality holds for the primal-dual pair of problems (6.49) and (6.52) (see Theorem A.1). To solve problem (6.52), we will first rewrite it as a minimization problem: 

$$
\min _ {\mathbf {y} \in \mathbb {R} ^ {m}} \left\{\frac {1}{2} \mathbf {y} ^ {T} \mathbf {A} \mathbf {A} ^ {T} \mathbf {y} + (\mathbf {A x}) ^ {T} \mathbf {y}: \| \mathbf {y} \| _ {2} ^ {2} \leq \lambda^ {2} \right\}.\tag{6.53}
$$

So far we have shown that 

$$
\operatorname{prox} _ {\lambda f} (\mathbf {x}) = \mathbf {x} + \mathbf {A} ^ {T} \mathbf {y},\tag{6.54}
$$

where $\mathbf { y }$ is an optimal solution of problem (6.53). Since problem (6.53) is convex and satisfies Slater’s condition, it follows by the KKT conditions that y is an optimal solution of (6.53) if and only if there exists $\alpha ^ { * }$ (optimal dual variable) for which 

$$
(\mathbf {A} \mathbf {A} ^ {T} + \alpha^ {*} \mathbf {I}) \mathbf {y} + \mathbf {A x} = \mathbf {0},\tag{6.55}
$$

$$
\alpha^ {*} (\| \mathbf {y} \| _ {2} ^ {2} - \lambda^ {2}) = 0,\tag{6.56}
$$

$$
\| \mathbf {y} \| _ {2} ^ {2} \leq \lambda^ {2},\tag{6.57}
$$

$$
\alpha^ {*} \geq 0.\tag{6.58}
$$

There are two options. In the first, $\alpha ^ { * } = 0$ , and then by (6.55), 

$$
\mathbf {y} = - \left(\mathbf {A A} ^ {T}\right) ^ {- 1} \mathbf {A x}.\tag{6.59}
$$

Since (6.56) and (6.58) are automatically satisfied for $\alpha ^ { * } = 0$ , we can conclude that y given in (6.59) is the optimal solution of (6.53) if and only if (6.57) is satisfied, meaning if and only if $\| ( \mathbf { A } \mathbf { A } ^ { T } ) ^ { - 1 } \mathbf { A } \mathbf { x } \| _ { 2 } \leq \lambda$ In this case, by (6.54), $\mathrm { p r o x } _ { \lambda f } ( { \bf x } ) = { \bf x } - \bar { \bf A } ^ { T } ( { \bf A } { \bf A } ^ { T } ) ^ { - 1 } { \bf A } { \bf x } .$ 

On the other hand, if $\| ( \mathbf { A } \mathbf { A } ^ { T } ) ^ { - 1 } \mathbf { A } \mathbf { x } \| _ { 2 } > \lambda$ , then $\alpha ^ { * } > 0$ , and hence by the complementary slackness condition (6.56), 

$$
\| \mathbf {y} \| _ {2} ^ {2} = \lambda^ {2}.\tag{6.60}
$$

By (6.55), 

$$
\mathbf {y} = - (\mathbf {A A} ^ {T} + \alpha^ {*} \mathbf {I}) ^ {- 1} \mathbf {A x}.
$$

Using (6.60), we can conclude that $\alpha ^ { * }$ can be uniquely determined as the positive root of the function 

$$
g (\alpha) = \| (\mathbf {A} \mathbf {A} ^ {T} + \alpha \mathbf {I}) ^ {- 1} \mathbf {A x} \| _ {2} ^ {2} - \lambda^ {2}.
$$

It is easy to see that $g$ is strictly decreasing for $\alpha \geq 0$ , and therefore $g$ has a unique root. 

## 6.8.2 Squared $l _ { 1 } \mathbf { - N o r m }$

The prox of the $l _ { 1 } { \mathrm { - n o r m } }$ has a simple formula. In this section we will show how to compute the prox of the squared $l _ { 1 } { \mathrm { - n o r m } } { \mathrm { - a } }$ task that will prove itself to be much more complicated. We will require the following lemma that expresses $\| \mathbf { x } \| _ { 1 } ^ { 2 }$ as the optimal value of an optimization problem written in terms of the function 

$$
\varphi (s, t) = \left\{ \begin{array}{l l} \frac {s ^ {2}}{t}, & t > 0, \\ 0, & s = 0, t = 0, \\ \infty & \text { else }. \end{array} \right.\tag{6.61}
$$

By Example 2.32, $\varphi$ is closed and convex (even though it is not continuous at $( s , t ) = ( 0 , 0 ) )$ ). 

Lemma 6.69 (variational representation of $\| \cdot \| _ { 1 } ^ { 2 } )$ . For any $\mathbf { x } \in \mathbb { R } ^ { n }$ the following holds: 

$$
\min _ {\boldsymbol {\lambda} \in \Delta_ {n}} \sum_ {j = 1} ^ {n} \varphi (x _ {j}, \lambda_ {j}) = \| \mathbf {x} \| _ {1} ^ {2},\tag{6.62}
$$

where $\varphi$ is defined in (6.61). An optimal solution of the minimization problem in (6.62) is given by 

$$
\tilde {\lambda} _ {j} = \left\{ \begin{array}{l l} \frac {| x _ {j} |}{\| \mathbf {x} \| _ {1}}, & \mathbf {x} \neq \mathbf {0}, \\ \frac {1}{n}, & \mathbf {x} = \mathbf {0}, \end{array} \right. \quad j = 1, 2, \ldots , n.\tag{6.63}
$$

Proof. Since problem (6.62) consists of minimizing a closed and convex function (by Example 2.32) over a compact set, then by the Weierstrass theorem for closed functions (Theorem 2.12), it possesses an optimal solution, which we denote by $\boldsymbol { \lambda } ^ { * } \in \Delta _ { n }$ . Define 

$$
\begin{array}{l} I _ {0} = \{i \in \{1, 2, \ldots , n \}: \lambda_ {i} ^ {*} = 0 \}, \\ I _ {1} = \{i \in \{1, 2, \ldots , n \}: \lambda_ {i} ^ {*} > 0 \}. \end{array}
$$

By the definitions of $I _ { 0 }$ and $I _ { 1 }$ , we have 

$$
\sum_ {i \in I _ {1}} \lambda_ {j} ^ {*} = \sum_ {i = 1} ^ {n} \lambda_ {j} ^ {*} = 1.\tag{6.64}
$$

It holds that $x _ { i } = 0$ for any $i \in I _ { 0 }$ , since otherwise we will have that $\varphi ( x _ { i } , \lambda _ { i } ^ { * } ) = \infty .$ which is a clear contradiction to the optimality of $\lambda ^ { * }$ . Therefore, using the Cauchy– Schwarz inequality, 

$$
\sum_ {j = 1} ^ {n} | x _ {j} | = \sum_ {j \in I _ {1}} | x _ {j} | = \sum_ {j \in I _ {1}} \frac {| x _ {j} |}{\sqrt {\lambda_ {j} ^ {*}}} \sqrt {\lambda_ {j} ^ {*}} \leq \sqrt {\sum_ {j \in I _ {1}} \frac {x _ {j} ^ {2}}{\lambda_ {j} ^ {*}}} \cdot \sqrt {\sum_ {j \in I _ {1}} \lambda_ {j} ^ {*}} \stackrel {(6. 6 4)} {=} \sqrt {\sum_ {j \in I _ {1}} \frac {x _ {j} ^ {2}}{\lambda_ {j} ^ {*}}}.
$$

We can thus conclude that 

$$
\sum_ {j = 1} ^ {n} \varphi (x _ {j}, \lambda_ {j} ^ {*}) = \sum_ {j \in I _ {1}} \varphi (x _ {j}, \lambda_ {j} ^ {*}) = \sum_ {j \in I _ {1}} \frac {x _ {j} ^ {2}}{\lambda_ {j} ^ {*}} \geq \| \mathbf {x} \| _ {1} ^ {2}.\tag{6.65}
$$

On the other hand, since $\lambda ^ { * }$ is an optimal solution of the problem in (6.62), 

$$
\sum_ {j = 1} ^ {n} \varphi (x _ {j}, \lambda_ {j} ^ {*}) \leq \sum_ {j = 1} ^ {n} \varphi (x _ {j}, \tilde {\lambda} _ {j}) = \| \mathbf {x} \| _ {1} ^ {2},\tag{6.66}
$$

where $\tilde { \lambda }$ is $\mathrm { g i }$ ven by (6.63). Combining (6.65) and (6.66), we finally conclude that the optimal value of the minimization problem in (6.62) is $\| \mathbf { x } \| _ { 1 } ^ { 2 }$ and that $\tilde { \lambda }$ is an optimal solution. 

Lemma 6.70 (prox of $\| \cdot \| _ { 1 } ^ { 2 } ) . ^ { 3 4 }$ Let $f : \mathbb { R } ^ { n }  \mathbb { R }$ be given by $f ( \mathbf { x } ) = \| \mathbf { x } \| _ { 1 } ^ { 2 }$ , and let $\rho > 0$ . Then 

$$
\operatorname{prox} _ {\rho f} (\mathbf {x}) = \left\{ \begin{array}{l l} \left(\frac {\lambda_ {i} x _ {i}}{\lambda_ {i} + 2 \rho}\right) _ {i = 1} ^ {n}, & \mathbf {x} \neq \mathbf {0}, \\ \mathbf {0}, & \mathbf {x} = \mathbf {0}, \end{array} \right.
$$

where $\begin{array} { r } { \lambda _ { i } ~ = ~ \left[ \frac { \sqrt { \rho } | x _ { i } | } { \sqrt { \mu ^ { * } } } - 2 \rho \right] _ { + } } \end{array}$ with $\mu ^ { * }$ being any positive root of the nonincreasing function 

$$
\psi (\mu) = \sum_ {i = 1} ^ {n} \left[ \frac {\sqrt {\rho} | x _ {i} |}{\sqrt {\mu}} - 2 \rho \right] _ {+} - 1.
$$

Proof. $\mathrm { I f } \ \mathbf { x } = \ \mathbf { 0 }$ , then obviously pro ${ \mathrm {  ~ \bar { \ s } ~ } } _ { \rho f } ( { \bf { x } } ) \ = \ \mathrm { a r g m i n } _ { \bf { u } } \left\{ \frac { 1 } { 2 } \| { \bf { u } } \| _ { 2 } ^ { 2 } + \rho \| { \bf { u } } \| _ { 1 } ^ { 2 } \right\} \ = \ { \bf { 0 } }$ Assume that $\mathbf { x } \neq \mathbf { 0 }$ . By Lemma 6.69, $\mathbf { u } \overset { \cdot } { = } \operatorname { p r o x } _ { \rho f } ( \mathbf { x } )$  if and only if it is the u-part of the optimal solution of 

$$
\min _ {\mathbf {u} \in \mathbb {R} ^ {n}, \boldsymbol {\lambda} \in \Delta_ {n}} \left\{\frac {1}{2} \| \mathbf {u} - \mathbf {x} \| _ {2} ^ {2} + \rho \sum_ {i = 1} ^ {n} \varphi (u _ {i}, \lambda_ {i}) \right\},
$$

where $\varphi$ is defined in (6.61). Minimizing first with respect to u, we obtain that $\begin{array} { r } { u _ { i } = \frac { \lambda _ { i } x _ { i } } { \lambda _ { i } + 2 \rho } } \end{array}$ , and the problem thus reduces to 

$$
\begin{array}{l l} \min _ {\boldsymbol {\lambda}} & \sum_ {i = 1} ^ {n} \frac {\rho x _ {i} ^ {2}}{\lambda_ {i} + 2 \rho} \\ \text {s.t.} & \mathbf {e} ^ {T} \boldsymbol {\lambda} = 1, \\ & \boldsymbol {\lambda} \geq \mathbf {0}. \end{array}\tag{6.67}
$$

By Theorem $\mathrm { A . 1 }$ , strong duality holds for problem (6.67) (taking the underlying set as $X = \mathbb { R } _ { + } ^ { n } )$ . Associating a Lagrange multiplier $\mu$ to the equality constraint, the Lagrangian is 

$$
L (\boldsymbol {\lambda}; \mu) = \sum_ {i = 1} ^ {n} \left(\frac {\rho x _ {i} ^ {2}}{\lambda_ {i} + 2 \rho} + \lambda_ {i} \mu\right) - \mu .
$$

By Theorem A.2, $\lambda ^ { * }$ is an optimal solution of (6.67) if and only if there exists $\mu ^ { * }$ for which 

$$
\boldsymbol {\lambda} ^ {*} \in \operatorname{argmin} _ {\boldsymbol {\lambda} \geq \mathbf {0}} L (\boldsymbol {\lambda}; \mu^ {*}),\tag{6.68}
$$

$$
\mathbf {e} ^ {T} \pmb {\lambda} ^ {*} = 1.\tag{6.69}
$$

Since the minimum in (6.68) is finite and attained, and since $\mathbf { x } \neq \mathbf { 0 } ,$ , it follows that $\mu ^ { * } > 0$ (otherwise, if $\mu ^ { * } = 0$ , the minimum in (6.68) would not be attained). Exploiting the separability of the Lagrangian, it follows that (6.68) is the same as 

$$
\lambda_ {i} ^ {*} = \left[ \frac {\sqrt {\rho} | x _ {i} |}{\sqrt {\mu}} - 2 \rho \right] _ {+}.
$$

The dual optimal variable $\mu ^ { * }$ is chosen to satisfy (6.69): 

$$
\sum_ {i = 1} ^ {n} \left[ \frac {\sqrt {\rho} | x _ {i} |}{\sqrt {\mu}} - 2 \rho \right] _ {+} = 1. \quad \square
$$

## 6.8.3 Projection onto the Set of s-Sparse Vectors

Let $s \in \{ 1 , 2 , \ldots , n \}$ and consider the set 

$$
C _ {s} = \left\{\mathbf {x} \in \mathbb {R} ^ {n}: \| \mathbf {x} \| _ {0} \leq s \right\}.
$$

The set $C _ { s }$ comprises all s-sparse vectors, meaning all vectors with at most s nonzero elements. Obviously $C _ { s }$ is not convex; for example, for $n = 2 , ( 0 , 1 ) ^ { T } , ( 1 , 0 ) ^ { T } \in C _ { 1 }$ but $( 0 . 5 , 0 . 5 ) ^ { T } = 0 . 5 ( 0 , 1 ) ^ { T } + 0 . 5 ( 1 , 0 ) ^ { T } \notin C _ { 1 }$ . The set $C _ { s }$ is closed as a level set of the closed function $\| \cdot \| _ { 0 }$ (see Example 2.11). Therefore, by Theorem 6.4, $P _ { C _ { s } } = \mathrm { p r o x } _ { \delta _ { C } }$ is nonempty; however, the nonconvexity of $C _ { s }$ implies that $P _ { C _ { s } } ( \mathbf { x } )$ is not necessarily a singleton. 

The set $P _ { C _ { s } } ( \mathbf { x } )$ is described in Lemma 6.71 below. The description requires some additional notation. For a vector $\mathbf { x } \in \mathbb { R } ^ { n }$ and a set of indices $S \subseteq \{ 1 , 2 , \ldots , n \}$ ， the vector $\mathbf { x } _ { S }$ is the subvector of x that corresponds to the indices in S. For example, for $n = 4 , \mathrm { i f } \ \mathbf { x } = ( 4 , 3 , 5 , - 1 ) ^ { T }$ , then $\mathbf { x } _ { \{ 1 , 4 \} } = ( 4 , - 1 ) ^ { T } , \mathbf { x } _ { \{ 2 , 3 \} } = ( 3 , 5 ) ^ { T }$ . For a given indices set $S \subseteq \{ 1 , 2 , \ldots , n \}$ , the matrix $\mathbf { U } _ { S }$ is the submatrix of the identity matrix ${ \mathbf I } _ { n }$ comprising the columns corresponding to the indices in S. For example, for $n = 3$ ， 

$$
\mathbf {U} _ {\{1, 3 \}} = \left( \begin{array}{c c} 1 & 0 \\ 0 & 0 \\ 0 & 1 \end{array} \right), \qquad \mathbf {U} _ {\{2 \}} = \left( \begin{array}{c} 0 \\ 1 \\ 0 \end{array} \right).
$$

For a given indices set $S \subseteq \{ 1 , 2 , \ldots , n \}$ , the complement set $S ^ { c }$ is given by $S ^ { c } =$ $\{ 1 , 2 , \ldots , n \} \setminus S$ 

Finally, we recall our notation (that was also used in Example 6.51) that for a given $\mathbf { x } \in \mathbb { R } ^ { n } , x _ { \langle i \rangle }$ is the ith largest value among $\vert x _ { 1 } \vert , \vert x _ { 2 } \vert , \dots , \vert x _ { n } \vert$ . Therefore, in particular, $| x _ { \langle 1 \rangle } | \geq | x _ { \langle 2 \rangle } | \geq \cdot \cdot \cdot \geq | x _ { \langle n \rangle } |$ . Lemma 6.71 shows that $P _ { C _ { s } } ( \mathbf { x } )$ comprises all vectors consisting of the s components of x with the largest absolute values and with zeros elsewhere. There may be several choices for the s components with largest absolute values, and this is why $P _ { C _ { s } } ( \mathbf { x } )$ might consist of several vectors. Note that in the statement of the lemma, we characterize the property of an index set $S$ to “comprise s indices corresponding to the s largest absolute values in $\mathbf { x } ^ { \mathfrak { N } }$ by the relation 

$$
S \subseteq \{1, 2, \dots , n \}, \quad | S | = s, \quad \sum_ {i \in S} | x _ {i} | = \sum_ {i = 1} ^ {s} | x _ {\langle i \rangle} |.
$$

Lemma 6.71 (projection onto $C _ { s } )$ . Let $s \in \{ 1 , 2 , \ldots , n \}$ and $\mathbf { x } \in \mathbb { R } ^ { n }$ . Then 

$$
P _ {C _ {s}} (\mathbf {x}) = \left\{\mathbf {U} _ {S} \mathbf {x} _ {S}: | S | = s, S \subseteq \{1, 2, \dots , n \}, \sum_ {i \in S} | x _ {i} | = \sum_ {i = 1} ^ {s} | x _ {\langle i \rangle} | \right\}.
$$

Proof. Since $C _ { s }$ consists of all s-sparse vectors, it can be represented as the following union: 

$$
C _ {s} = \bigcup_ {S \subseteq \{1, 2, \dots , n \}, | S | = s} A _ {S},
$$

where $A _ { S } = \left\{ \mathbf { x } \in \mathbb { R } ^ { n } : \mathbf { x } _ { S ^ { c } } = \mathbf { 0 } \right\}$ . Therefore,<sup>35</sup> 

$$
P _ {C _ {s}} (\mathbf {x}) \subseteq \bigcup_ {S \subseteq \{1, 2, \dots , n \}, | S | = s} \left\{P _ {A _ {S}} (\mathbf {x}) \right\}.\tag{6.70}
$$

The vectors in $P _ { C _ { s } } ( \mathbf { x } )$ will be the vectors $P _ { A _ { S } } ( \mathbf { x } )$ with the smallest possible value of $\| P _ { A _ { S } } ( \mathbf { x } ) - \mathbf { x } \| ^ { 2 }$ . The vector $P _ { A _ { S } } ( \mathbf { x } )$ is the optimal solution of the problem 

$$
\min _ {\mathbf {y} \in \mathbb {R} ^ {n}} \left\{\| \mathbf {y} - \mathbf {x} \| _ {2} ^ {2}: \mathbf {y} _ {S ^ {c}} = \mathbf {0} \right\},
$$

which can be rewritten as 

$$
\min _ {\mathbf {y} \in \mathbb {R} ^ {n}} \left\{\| \mathbf {y} _ {S} - \mathbf {x} _ {S} \| _ {2} ^ {2} + \| \mathbf {x} _ {S ^ {c}} \| _ {2} ^ {2}: \mathbf {y} _ {S ^ {c}} = \mathbf {0} \right\}.
$$

The optimal solution of the above problem is obviously given by $\mathbf { y } _ { S } = \mathbf { x } _ { S } , \mathbf { y } _ { S ^ { c } } = \mathbf { 0 }$ 9 that is, $\mathbf { y } = \mathbf { U } _ { S } \mathbf { x } _ { S }$ , and the optimal value is $\| \mathbf { x } _ { S ^ { c } } \| _ { 2 } ^ { 2 }$ . The vectors in $P _ { C _ { s } } ( \mathbf { x } )$ will therefore be of the form $\mathbf { U } _ { S } \mathbf { x } _ { S }$ , with indices sets S with cardinality s and with minimal value $\| \mathbf { x } _ { S ^ { c } } \| _ { 2 } ^ { 2 }$ , which is equivalent to the condition that S consists of s indices corresponding to the s largest absolute values in x. 

Example 6.72. Suppose that $n = 4$ . Then 

$$
P _ {C _ {2}} [ (2, 3, - 2, 1) ^ {T} ] = \{(2, 3, 0, 0) ^ {T}, (0, 3, - 2, 0) ^ {T} \}.
$$


6.9 Summary of Prox Computations


<table><tr><td>f(x)</td><td>dom(f)</td><td>prox<eq>_{f}</eq>(x)</td><td>Assumptions</td><td>Reference</td></tr><tr><td><eq>\frac{1}{2}\mathbf{x}^{T}\mathbf{A}\mathbf{x}+\mathbf{b}^{T}\mathbf{x}+c</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>(\mathbf{A}+\mathbf{I})^{-1}(\mathbf{x}-\mathbf{b})</eq></td><td><eq>\mathbf{A}\in\mathbb{S}_{+}^{n},\mathbf{b}\in\mathbb{R}^{n},c\in\mathbb{R}</eq></td><td>Section 6.2.3</td></tr><tr><td><eq>\lambda x^{3}</eq></td><td><eq>\mathbb{R}_{+}</eq></td><td><eq>\frac{-1+\sqrt{1+12\lambda[x]_{+}}}{6\lambda}</eq></td><td><eq>\lambda&gt;0</eq></td><td>Lemma 6.5</td></tr><tr><td><eq>\mu x</eq></td><td><eq>[0,\alpha]\cap\mathbb{R}</eq></td><td><eq>\min\{\max\{x-\mu,0\},\alpha\}</eq></td><td><eq>\mu\in\mathbb{R},\alpha\in[0,\infty]</eq></td><td>Example 6.14</td></tr><tr><td><eq>\lambda\|\mathbf{x}\|</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>\left(1-\frac{\lambda}{\max\{\|\mathbf{x}\|,\lambda\}}\right)\mathbf{x}</eq></td><td><eq>\|\cdot\|—Euclidean norm,\lambda&gt;0</eq></td><td>Example 6.19</td></tr><tr><td><eq>-\lambda\|\mathbf{x}\|</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>\left(1+\frac{\lambda}{\|\mathbf{x}\|}\right)\mathbf{x},\quad\mathbf{x}\neq\mathbf{0},\quad\{\mathbf{u}:\|\mathbf{u}\|=\lambda\},\quad\mathbf{x}=\mathbf{0}.</eq></td><td><eq>\|\cdot\|—Euclidean norm,\lambda&gt;0</eq></td><td>Example 6.21</td></tr><tr><td><eq>\lambda\|\mathbf{x}\|_{1}</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>\mathcal{T}_{\lambda}(\mathbf{x})=[|\mathbf{x}|-\lambda\mathbf{e}]_{+}\odot\operatorname{sgn}(\mathbf{x})</eq></td><td><eq>\lambda&gt;0</eq></td><td>Example 6.8</td></tr><tr><td><eq>\|\boldsymbol{\omega}\odot\mathbf{x}\|_{1}</eq></td><td><eq>\text{Box}[-\boldsymbol{\alpha},\boldsymbol{\alpha}]</eq></td><td><eq>\mathcal{S}_{\boldsymbol{\omega},\boldsymbol{\alpha}}(\mathbf{x})</eq></td><td><eq>\boldsymbol{\alpha}\in[0,\infty]^{n},\boldsymbol{\omega}\in\mathbb{R}_{+}^{n}</eq></td><td>Example 6.23</td></tr><tr><td><eq>\lambda\|\mathbf{x}\|_{\infty}</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>\mathbf{x}-\lambda P_{B_{\|\cdot\|_{1}}[\mathbf{0},1]}(\mathbf{x}/\lambda)</eq></td><td><eq>\lambda&gt;0</eq></td><td>Example 6.48</td></tr><tr><td><eq>\lambda\|\mathbf{x}\|_{a}</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>\mathbf{x}-\lambda P_{B_{\|\cdot\|_{a},*}}[\mathbf{0},1](\mathbf{x}/\lambda)</eq></td><td><eq>\|\mathbf{x}\|_{a}—arbitrary norm,\lambda&gt;0</eq></td><td>Example 6.47</td></tr><tr><td><eq>\lambda\|\mathbf{x}\|_{0}</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>\mathcal{H}_{\sqrt{2\lambda}}(x_{1})\times\cdots\times\mathcal{H}_{\sqrt{2\lambda}}(x_{n})</eq></td><td><eq>\lambda&gt;0</eq></td><td>Example 6.10</td></tr><tr><td><eq>\lambda\|\mathbf{x}\|^{3}</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>\frac{2}{1+\sqrt{1+12\lambda\|\mathbf{x}\|}}\mathbf{x}</eq></td><td><eq>\|\cdot\|—Euclidean norm,\lambda&gt;0,</eq></td><td>Example 6.20</td></tr><tr><td><eq>-\lambda\sum_{j=1}^{n}\log x_{j}</eq></td><td><eq>\mathbb{R}_{++}^{n}</eq></td><td><eq>\left(\frac{x_{j}+\sqrt{x_{j}^{2}+4\lambda}}{2}\right)_{j=1}^{n}</eq></td><td><eq>\lambda&gt;0</eq></td><td>Example 6.9</td></tr><tr><td><eq>\delta_{C}(\mathbf{x})</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>P_{C}(\mathbf{x})</eq></td><td><eq>\emptyset\neq C\subseteq\mathbb{E}</eq></td><td>Theorem 6.24</td></tr><tr><td><eq>\lambda\sigma_{C}(\mathbf{x})</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>\mathbf{x}-\lambda P_{C}(\mathbf{x}/\lambda)</eq></td><td><eq>\lambda&gt;0,C\neq\emptyset</eq>closed convex</td><td>Theorem 6.46</td></tr><tr><td><eq>\lambda\max\{x_{i}\}</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>\mathbf{x}-\lambda P_{\Delta n}(\mathbf{x}/\lambda)</eq></td><td><eq>\lambda&gt;0</eq></td><td>Example 6.49</td></tr><tr><td><eq>\lambda\sum_{i=1}^{k}x_{[i]}</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>\mathbf{x}-\lambda P_{C}(\mathbf{x}/\lambda),\quad C=H_{\mathbf{e},k}\cap\text{Box}[\mathbf{0},\mathbf{e}]</eq></td><td><eq>\lambda&gt;0</eq></td><td>Example 6.50</td></tr><tr><td><eq>\lambda\sum_{i=1}^{k}|x_{\langle i\rangle}|</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>\mathbf{x}-\lambda P_{C}(\mathbf{x}/\lambda),\quad C=B_{\|\cdot\|_{1}}[\mathbf{0},k]\cap\text{Box}[-\mathbf{e},\mathbf{e}]</eq></td><td><eq>\lambda&gt;0</eq></td><td>Example 6.51</td></tr><tr><td><eq>\lambda M_{f}^{\mu}(\mathbf{x})</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>\mathbf{x}+\frac{\lambda}{\mu+\lambda}\left(\text{prox}_{(\mu+\lambda)f}(\mathbf{x})-\mathbf{x}\right)</eq></td><td><eq>\lambda,\mu&gt;0,f</eq>proper closed convex</td><td>Corollary 6.64</td></tr><tr><td><eq>\lambda d_{C}(\mathbf{x})</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>\mathbf{x}+\min\left\{\frac{\lambda}{d_{C}(\mathbf{x})},1\right\}(P_{C}(\mathbf{x})-\mathbf{x})</eq></td><td><eq>\emptyset\neq C</eq>closed convex, <eq>\lambda&gt;0</eq></td><td>Lemma 6.43</td></tr><tr><td><eq>\frac{\lambda}{2}d_{C}^{2}(\mathbf{x})</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>\frac{\lambda}{\lambda+1}P_{C}(\mathbf{x})+\frac{1}{\lambda+1}\mathbf{x}</eq></td><td><eq>\emptyset\neq C</eq>closed convex, <eq>\lambda&gt;0</eq></td><td>Example 6.65</td></tr><tr><td><eq>\lambda H_{\mu}(\mathbf{x})</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>\left(1-\frac{\lambda}{\max\{\|\mathbf{x}\|,\mu+\lambda\}}\right)\mathbf{x}</eq></td><td><eq>\lambda,\mu&gt;0</eq></td><td>Example 6.66</td></tr><tr><td><eq>\rho\|\mathbf{x}\|_{1}^{2}</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>\left(\frac{v_{i}x_{i}}{v_{i}+2\rho}\right)_{i=1}^{n},\mathbf{v}=\left[\sqrt{\frac{\nu}{\mu}}|\mathbf{x}|-2\rho\right]_{+},\mathbf{e}^{T}\mathbf{v}=1(0 when \mathbf{x}=0)</eq></td><td><eq>\rho&gt;0</eq></td><td>Lemma 6.70</td></tr><tr><td><eq>\lambda\|\mathbf{A}\mathbf{x}\|_{2}</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>\mathbf{x}-\mathbf{A}^{T}(\mathbf{A}\mathbf{A}^{T}+\alpha^{*}\mathbf{I})^{-1}\mathbf{A}\mathbf{x},\alpha^{*}=0 \text{ if }\|\mathbf{v}_{0}\|_{2}\leq\lambda;\text{ otherwise },\|\mathbf{v}_{\alpha^{*}}\|_{2}=\lambda;\mathbf{v}_{\alpha}\equiv(\mathbf{A}\mathbf{A}^{T}+\alpha\mathbf{I})^{-1}\mathbf{A}\mathbf{x}</eq></td><td><eq>\mathbf{A}\in\mathbb{R}^{m\times n}</eq>with full row rank, <eq>\lambda&gt;0</eq></td><td>Lemma 6.68</td></tr></table>

# Chapter 7

# Spectral Functions

In this chapter we will concentrate on spectral functions, which are functions of matrices that depend only on their singular values or on their eigenvalues. The underlying spaces in this chapter are all Euclidean. We start by defining the notion of symmetry w.r.t. a given set of orthogonal matrices. 

## 7.1 Symmetric Functions

## 7.1.1 Definition and Examples

Definition 7.1 (symmetric functions). Let ${ \mathcal { A } } \subseteq \mathbb { O } ^ { n }$ be a set of orthogonal matrices. A proper function $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ is called symmetric w.r.t. A if 

$$
f (\mathbf {A} \mathbf {x}) = f (\mathbf {x}) \text {   for   all   } \mathbf {x} \in \mathbb {R} ^ {n}, \mathbf {A} \in \mathcal {A}.
$$

The following are five types of symmetric functions, each one dictated by the choice of orthogonal matrices in A. 

Example 7.2 (even functions). If ${ \mathcal { A } } = \{ - { \bf I } \}$ , then $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ is symmetric w.r.t. A if 

$$
f (\mathbf {x}) = f (- \mathbf {x}) \text {   for   all   } \mathbf {x} \in \mathbb {R} ^ {n}.
$$

Such functions will be called even functions. 

Example 7.3 (absolutely symmetric functions). Take $\mathcal { A } = \{ \mathbf { D } _ { 1 } , \mathbf { D } _ { 2 } , \dots , \mathbf { D } _ { n } \}$ $\subseteq \mathbb { R } ^ { n \times n }$ , where $\mathbf { D } _ { i }$ is the diagonal matrix whose diagonal elements are all ones except for the $( i , i )$ component which is equal to −1. Then a proper function $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ is symmetric w.r.t. A if and only if 

$$
f (\mathbf {x}) = f (| \mathbf {x} |) \text {   for   all   } \mathbf {x} \in \mathbb {R} ^ {n}.
$$

We will call such a function an absolutely symmetric function. It is easy to show that $f$ is absolutely symmetric if and only if there exists a function $g : \mathbb { R } _ { + } ^ { n }  ( - \infty , \infty ]$ such that $f ( \mathbf { x } ) = g ( | \mathbf { x } | )$ for all $\mathbf { x } \in \mathbb { R } ^ { n }$ ■ 

Example 7.4 (norm-dependent functions). A proper function $f : \mathbb { R } ^ { n } $ $( - \infty , \infty ]$ is symmetric w.r.t. $\ b { \mathcal { A } } = \mathbb { O } ^ { n }$ if and only if 

$$
f (\mathbf {x}) = f (\mathbf {U x}) \text {   for   all   } \mathbf {x} \in \mathbb {R} ^ {n}, \mathbf {U} \in \mathbb {O} ^ {n}.
$$

The above holds if and only if there exists a proper function $g : \mathbb { R }  ( - \infty , \infty ]$ such that 

$$
f (\mathbf {x}) = g \left(\| \mathbf {x} \| _ {2}\right) \text {   for   all   } \mathbf {x} \in \mathbb {R} ^ {n}.
$$

A function satisfying the above is called a norm-dependent function. 

We will require some additional notation before describing the next two examples. For a given vector $\mathbf { x } \in \mathbb { R } ^ { n }$ , the vector $\mathbf { x } ^ { \downarrow }$ is the vector x reordered nonincreasingly. For example, if $\mathbf { x } = ( 2 , - 9 , 2 , 1 0 ) ^ { T }$ , then $\mathbf { x } ^ { \downarrow } = ( 1 0 , 2 , 2 , - 9 ) ^ { T }$ 

Definition 7.5 (permutation matrices). An $n \times n$ matrix is called a permutation matrix if all its components are either 0 or 1 and each row and each column has exactly one nonzero element. The set of all $n \times n$ permutation matrices is denoted by $\Lambda _ { n }$ 

Definition 7.6 (generalized permutation matrices). An $n \times n$ matrix is called a generalized permutation matrix if all its components are either 0, 1, $o r - 1$ and each row and each column has exactly one nonzero element. The set of all $n \times n$ generalized permutation matrices is denoted by $\Lambda _ { n } ^ { \mathrm { G } }$ 

Thus, for example, 

$$
\left( \begin{array}{c c c} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{array} \right) \in \Lambda_ {3}, \qquad \left( \begin{array}{c c c} 0 & - 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & - 1 \end{array} \right) \in \Lambda_ {3} ^ {\mathrm{G}}.
$$

By the definition of permutation and generalized permutation matrices, it is easy to see that for any $\mathbf { x } \in \mathbb { R } ^ { n }$ there exists a permutation matrix $\mathbf { P } \in \Lambda _ { n }$ for which $\mathbf { P x } = \mathbf { x } ^ { \downarrow }$ and a generalized permutation matrix $\mathbf { Q } \in \Lambda _ { n } ^ { \mathrm { G } }$ for which $\mathbf { Q x } = | \mathbf { x } | ^ { \downarrow }$ . It can be readily verified that permutation matrices, as well as generalized permutation matrices, are orthogonal. 

Example 7.7 (permutation symmetric functions). A proper function $f$ : $\mathbb { R } ^ { n } \to ( - \infty , \infty ]$ is symmetric w.r.t. $\Lambda _ { n }$ if and only if 

$$
f (\mathbf {x}) = f (\mathbf {P x}) \text {   for   all   } \mathbf {x} \in \mathbb {R} ^ {n}, \mathbf {P} \in \Lambda_ {n}.
$$

Such a function will be called a permutation symmetric function. It is easy to show that $f$ is permutation symmetric if and only if 

$$
f (\mathbf {x}) = f \left(\mathbf {x} ^ {\downarrow}\right) \text {for all} \mathbf {x} \in \mathbb {R} ^ {n}.
$$

Example 7.8 (absolutely permutation symmetric functions). A proper function $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ is symmetric w.r.t. $\Lambda _ { n } ^ { \mathrm { G } }$ if and only if 

$$
f (\mathbf {x}) = f (\mathbf {P x}) \text {   for   all   } \mathbf {x} \in \mathbb {R} ^ {n}, \mathbf {P} \in \Lambda_ {n} ^ {\mathrm{G}}.
$$

Such a function will be called an absolutely permutation symmetric function. It is easy to show that f is absolutely permutation symmetric if and only if 

$$
f (\mathbf {x}) = f (| \mathbf {x} | ^ {\downarrow}) \text {for all} \mathbf {x} \in \mathbb {R} ^ {n}.
$$

## 7.1.2 The Symmetric Conjugate Theorem

We will now show that the conjugate of a symmetric function w.r.t. a set of orthogonal matrices is always symmetric w.r.t. the same set of matrices. 

Theorem 7.9 (symmetric conjugate theorem).<sup>36</sup> Let $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ be a proper function which is symmetric w.r.t. a set of orthogonal matrices ${ \mathcal { A } } \subseteq \mathbb { O } ^ { n }$ Then $f ^ { * }$ is symmetric w.r.t. A. 

Proof. Let $\mathbf { A } \in { \mathcal { A } }$ . Then by the symmetry assumption, $h = f$ , where $h ( \mathbf { x } ) \equiv$ $f ( \mathbf { A x } )$ . Thus, 

$$
f ^ {*} (\mathbf {y}) = h ^ {*} (\mathbf {y}) \text {   for   all   } \mathbf {y} \in \mathbb {R} ^ {n}.\tag{7.1}
$$

By Theorem 4.13 and the orthogonality of $\mathbf { A } ,$ , for any $\mathbf { y } \in \mathbb { R } ^ { n }$ 

$$
h ^ {*} (\mathbf {y}) = f ^ {*} \left(\left(\mathbf {A} ^ {T}\right) ^ {- 1} \mathbf {y}\right) = f ^ {*} (\mathbf {A y}),
$$

which, combined with (7.1), yields 

$$
f ^ {*} (\mathbf {y}) = f ^ {*} (\mathbf {A y}) \text {   for   all   } \mathbf {y} \in \mathbb {R} ^ {n}.
$$

Since the above holds for any $\mathbf A \in { \mathcal A }$ , it follows that $f ^ { * }$ is symmetric w.r.t. A. 

Example 7.10. In this example we will illustrate the symmetric conjugate theorem by verifying that the types of symmetries satisfied by the functions in the table of Section 4.4.16 also hold for their conjugates. 

## • even functions

<table><tr><td><eq>f(\mathbf{x})</eq></td><td><eq>\text{dom}(f)</eq></td><td><eq>f^{*}(\mathbf{y})</eq></td><td>Assumptions</td><td>Reference</td></tr><tr><td><eq>\frac{1}{p}|x|^p</eq></td><td><eq>\mathbb{R}</eq></td><td><eq>\frac{1}{q}|y|^q</eq></td><td><eq>p &gt; 1, \frac{1}{p} + \frac{1}{q} = 1</eq></td><td>Section 4.4.4</td></tr><tr><td><eq>\frac{1}{2}\mathbf{x}^T\mathbf{A}\mathbf{x} + c</eq></td><td><eq>\mathbb{R}^n</eq></td><td><eq>\frac{1}{2}\mathbf{y}^T\mathbf{A}^{-1}\mathbf{y} - c</eq></td><td><eq>\mathbf{A} \in \mathbb{S}_{++}^n, c \in \mathbb{R}</eq></td><td>Section 4.4.6</td></tr></table>

• permutation symmetric functions 

<table><tr><td><eq>f(\mathbf{x})</eq></td><td><eq>\text{dom}(f)</eq></td><td><eq>f^{*}(\mathbf{y})</eq></td><td>Reference</td></tr><tr><td><eq>\sum_{i=1}^{n} x_{i} \log x_{i}</eq></td><td><eq>\mathbb{R}_{+}^{n}</eq></td><td><eq>\sum_{i=1}^{n} e^{y_{i}-1}</eq></td><td>Section 4.4.8</td></tr><tr><td><eq>\sum_{i=1}^{n} x_{i} \log x_{i}</eq></td><td><eq>\Delta_{n}</eq></td><td><eq>\log\left(\sum_{i=1}^{n} e^{y_{i}}\right)</eq></td><td>Section 4.4.10</td></tr><tr><td><eq>\log\left(\sum_{i=1}^{n} e^{x_{i}}\right)</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>\sum_{i=1}^{n} y_{i} \log y_{i}</eq><eq>(\text{dom}(f^{*}) = \Delta_{n})</eq></td><td>Section 4.4.11</td></tr><tr><td><eq>\max_{i}\{x_{i}\}</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>\delta_{\Delta_{n}}(\mathbf{y})</eq></td><td>Example 4.10</td></tr></table>


<sup>36</sup>The symmetric conjugate theorem (Theorem 7.9) is from Rockafellar [108, Corollary 12.3.1]. 


## • absolutely permutation symmetric functions

<table><tr><td><eq>f(\mathbf{x})</eq></td><td><eq>\text{dom}(f)</eq></td><td><eq>f^{*}(\mathbf{y})</eq></td><td>Reference</td></tr><tr><td><eq>\| \mathbf{x}\|_{p}</eq></td><td><eq>\mathbb{R}^{n}</eq></td><td><eq>\delta_{B_{\|\cdot\|_{q}}[\mathbf{0},1]}(\mathbf{y})</eq></td><td>Section 4.4.12</td></tr><tr><td><eq>\frac{1}{2}\| \mathbf{x}\|_{p}^{2}</eq></td><td><eq>\mathbb{E}</eq></td><td><eq>\frac{1}{2}\| \mathbf{y}\|_{q}^{2}</eq></td><td>Section 4.4.15</td></tr></table>

where $\begin{array} { r } { p , q \in [ 1 , \infty ] , \frac { 1 } { p } + \frac { 1 } { q } = 1 } \end{array}$ (with the convention that if $p = 1 , \infty$ , then $q = \infty , 1$ respectively). 

## • norm-dependent functions

<table><tr><td>f</td><td>dom(f)</td><td><eq>f^*</eq></td><td>Reference</td></tr><tr><td><eq>\| \mathbf{x} \|_2</eq></td><td><eq>\mathbb{R}^n</eq></td><td><eq>\delta_{B_{\|\cdot\|_2}[0,1]}(\mathbf{y})</eq></td><td>Section 4.4.12</td></tr><tr><td><eq>-\sqrt{\alpha^2 - \| \mathbf{x} \|_2^2}_{(\alpha &gt;0)}</eq></td><td><eq>B_{\|\cdot\|_2}[\mathbf{0},\alpha]</eq></td><td><eq>\alpha\sqrt{\|\mathbf{y}\|_2^2 + 1}</eq></td><td>Section 4.4.13</td></tr><tr><td><eq>\sqrt{\alpha^2 + \| \mathbf{x} \|_2^2}_{(\alpha &gt;0)}</eq></td><td><eq>\mathbb{R}^n</eq></td><td><eq>-\alpha\sqrt{1 - \| \mathbf{y}\|_2^2}_{(\text{dom }f^* = B_{\|\cdot\|_2}[0,1])}</eq></td><td>Section 4.4.14</td></tr><tr><td><eq>\frac{1}{2}\|\mathbf{x}\|_2^2</eq></td><td><eq>\mathbb{R}^n</eq></td><td><eq>\frac{1}{2}\|\mathbf{y}\|_2^2</eq></td><td>Section 4.4.15</td></tr></table>

## 7.2 Symmetric Spectral Functions over $\mathbb { S } ^ { n ^ { 3 7 } }$

The main concern of this chapter are functions of matrices that are defined on either the set of symmetric matrices $\mathbb { S } ^ { n }$ or the set of matrices $\mathbb { R } ^ { m \times n }$ . We will deal only with functions that depend either on the eigenvalues of their argument (if the underlying space is $\mathbb { S } ^ { n } )$ or on the singular values (if the underlying space is $\mathbb { R } ^ { m \times n } )$ Such functions are called spectral functions. We first consider functions over S<sup>n</sup>. Given a matrix $\mathbf { X } \in \mathbb { S } ^ { n }$ , its eigenvalues ordered nonincreasingly are denoted by 

$$
\lambda_ {1} (\mathbf {X}) \geq \lambda_ {2} (\mathbf {X}) \geq \dots \geq \lambda_ {n} (\mathbf {X}).
$$

The eigenvalues function $\pmb { \lambda } : \mathbb { S } ^ { n }  \mathbb { R } ^ { n }$ is defined as 

$$
\boldsymbol {\lambda} (\mathbf {X}) \equiv (\lambda_ {1} (\mathbf {X}), \lambda_ {2} (\mathbf {X}), \dots , \lambda_ {n} (\mathbf {X})) ^ {T}.
$$

A key fact from linear algebra is that any symmetric matrix $\mathbf { X } \in \mathbb { S } ^ { n }$ has a spectral decomposition, meaning an orthogonal matrix $\mathbf { U } \in \mathbb { O } ^ { n }$ for which $\mathbf { X } = \mathbf { U } \mathrm { d i a g } ( \lambda ( \mathbf { X } ) ) \mathbf { U } ^ { T }$ We begin by formally defining the notion of spectral functions over $\mathbb { S } ^ { n }$ 

Definition 7.11 (spectral functions over $\mathbb { S } ^ { n } )$ . A proper function $g : { \mathbb { S } } ^ { n } $ $( - \infty , \infty ]$ is called a spectral function over $\mathbb { S } ^ { n }$ if there exists a proper function $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ for which $g = f \circ \lambda$ 

If $g = f \circ \lambda$ , we will refer to $f$ (which is actually not necessarily unique) as the associated function. Our main interest will be to study spectral functions whose associated functions are permutation symmetric. 

Definition 7.12 (symmetric spectral functions over $\mathbb { S } ^ { n } )$ . A proper function $f : \mathbb { S } ^ { n }  ( - \infty , \infty ]$ is called a symmetric spectral function over $\mathbb { S } ^ { n }$ if there exists a proper permutation symmetric function $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ for which $g =$ $f \circ \lambda$ 


Example 7.13. Following is a list of permutation symmetric functions along with their associated symmetric spectral functions.


<table><tr><td>#</td><td>f(x)</td><td>dom(f)</td><td>f(λ(X))</td><td>dom(f∘λ)</td></tr><tr><td>1</td><td><eq>\sum_{i=1}^{n} x_i</eq></td><td><eq>\mathbb{R}^n</eq></td><td>Tr(X)</td><td><eq>\mathbb{S}^n</eq></td></tr><tr><td>2</td><td><eq>\max_{i=1,2,\ldots,n} x_i</eq></td><td><eq>\mathbb{R}^n</eq></td><td><eq>\lambda_{\max}(X)</eq></td><td><eq>\mathbb{S}^n</eq></td></tr><tr><td>3</td><td><eq>\alpha \|x\|_2 (\alpha \in \mathbb{R})</eq></td><td><eq>\mathbb{R}^n</eq></td><td><eq>\alpha \|X\|_F</eq></td><td><eq>\mathbb{S}^n</eq></td></tr><tr><td>4</td><td><eq>\alpha \|x\|_2^2 (\alpha \in \mathbb{R})</eq></td><td><eq>\mathbb{R}^n</eq></td><td><eq>\alpha \|X\|_F^2</eq></td><td><eq>\mathbb{S}^n</eq></td></tr><tr><td>5</td><td><eq>\alpha \|x\|_\infty (\alpha \in \mathbb{R})</eq></td><td><eq>\mathbb{R}^n</eq></td><td><eq>\alpha \|X\|_{2,2}</eq></td><td><eq>\mathbb{S}^n</eq></td></tr><tr><td>6</td><td><eq>\alpha \|x\|_1 (\alpha \in \mathbb{R})</eq></td><td><eq>\mathbb{R}^n</eq></td><td><eq>\alpha \|X\|_{S_1}</eq></td><td><eq>\mathbb{S}^n</eq></td></tr><tr><td>7</td><td><eq>-\sum_{i=1}^{n} \log(x_i)</eq></td><td><eq>\mathbb{R}_{++}^n</eq></td><td><eq>-\log \det(X)</eq></td><td><eq>\mathbb{S}_{++}^n</eq></td></tr><tr><td>8</td><td><eq>\sum_{i=1}^{n} x_i \log(x_i)</eq></td><td><eq>\mathbb{R}_+^n</eq></td><td><eq>\sum_{i=1}^{n} \lambda_i(X) \log(\lambda_i(X))</eq></td><td><eq>\mathbb{S}_+^n</eq></td></tr><tr><td>9</td><td><eq>\sum_{i=1}^{n} x_i \log(x_i)</eq></td><td><eq>\Delta_n</eq></td><td><eq>\sum_{i=1}^{n} \lambda_i(X) \log(\lambda_i(X))</eq></td><td><eq>\Upsilon_n</eq></td></tr></table>

The domain of the last function in the above table is the spectahedron set given by 

$$
\Upsilon_ {n} = \{\mathbf {X} \in \mathbb {S} _ {+} ^ {n}: \operatorname{Tr} (\mathbf {X}) = 1 \}.
$$

The norm used in the sixth function is the Schatten 1-norm whose expression for symmetric matrices is given by 

$$
\| \mathbf {X} \| _ {S _ {1}} = \sum_ {i = 1} ^ {n} | \lambda_ {i} (\mathbf {X}) |, \quad \mathbf {X} \in \mathbb {S} ^ {n}.
$$

Schatten p-norms will be discussed in detail in Section 7.3 

A fundamental inequality that will be a key argument in establishing the main results of this section is the so-called Fan inequality stating that the inner product of two symmetric matrices is upper bounded by the inner product of their eigenvalues vectors and that equality holds if and only if the two matrices are simultaneously orthogonally diagonalizable. 

Theorem 7.14 (Fan’s Inequality [32, 119]). For any two symmetric matrices X, $\mathbf { Y } \in \mathbb { S } ^ { n }$ it holds that 

$$
\operatorname{Tr} (\mathbf {X Y}) \leq \langle \boldsymbol {\lambda} (\mathbf {X}), \boldsymbol {\lambda} (\mathbf {Y}) \rangle ,
$$

and equality holds if and only if there exists $\mathbf { V } \in \mathbb { O } ^ { n }$ for which 

$$
\mathbf {X} = \mathbf {V} \mathrm{diag} (\lambda (\mathbf {X})) \mathbf {V} ^ {T},
$$

$$
\mathbf {Y} = \mathbf {V} \mathrm{diag} (\lambda (\mathbf {Y})) \mathbf {V} ^ {T}.
$$

## 7.2.1 The Spectral Conjugate Formula

A rather direct result of Fan’s inequality is the spectral conjugate formula that shows how to compute the conjugate of a symmetric spectral function over $\mathbb { S } ^ { n }$ in terms of the conjugate of its associated function. 

Theorem 7.15 (spectral conjugate formula over $\mathbb { S } ^ { n } )$ . Let $f : \mathbb { E } \to ( - \infty , \infty ]$ be a permutation symmetric function. Then 

$$
(f \circ \boldsymbol {\lambda}) ^ {*} = f ^ {*} \circ \boldsymbol {\lambda}.
$$

Proof. Let $\mathbf { Y } \in \mathbb { S } ^ { n }$ . Then 

$$
\begin{array}{l} (f \circ \boldsymbol {\lambda}) ^ {*} (\mathbf {Y}) = \max _ {\mathbf {X} \in \mathbb {S} ^ {n}} \{\mathrm{Tr} (\mathbf {X Y}) - f (\boldsymbol {\lambda} (\mathbf {X})) \} \\ \qquad \leq \max _ {\mathbf {X} \in \mathbb {S} ^ {n}} \{\langle \boldsymbol {\lambda} (\mathbf {X}), \boldsymbol {\lambda} (\mathbf {Y}) \rangle - f (\boldsymbol {\lambda} (\mathbf {X})) \} \\ \qquad \leq \max _ {\mathbf {x} \in \mathbb {R} ^ {n}} \{\langle \mathbf {x}, \boldsymbol {\lambda} (\mathbf {Y}) \rangle - f (\mathbf {x}) \} \\ \qquad = (f ^ {*} \circ \boldsymbol {\lambda}) (\mathbf {Y}), \end{array}
$$

where Fan’s inequality (Theorem 7.14) was used in the first inequality. To show the reverse inequality, take a spectral decomposition of Y: 

$$
\mathbf {Y} = \mathbf {U} \mathrm{diag} (\boldsymbol {\lambda} (\mathbf {Y})) \mathbf {U} ^ {T} (\mathbf {U} \in \mathbb {O} ^ {n}).
$$

Then 

$$
\begin{array}{l} (f ^ {*} \circ \boldsymbol {\lambda}) (\mathbf {Y}) = \max _ {\mathbf {x} \in \mathbb {R} ^ {n}} \{\langle \mathbf {x}, \boldsymbol {\lambda} (\mathbf {Y}) \rangle - f (\mathbf {x}) \} \\ \qquad = \max _ {\mathbf {x} \in \mathbb {R} ^ {n}} \{\mathrm{Tr} (\mathrm{diag} (\mathbf {x}) \mathrm{diag} (\boldsymbol {\lambda} (\mathbf {Y})) - f (\mathbf {x}) \} \\ \qquad = \max _ {\mathbf {x} \in \mathbb {R} ^ {n}} \{\mathrm{Tr} (\mathrm{diag} (\mathbf {x}) \mathbf {U} ^ {T} \mathbf {Y} \mathbf {U}) - f (\mathbf {x} ^ {\downarrow}) \} \\ \qquad = \max _ {\mathbf {x} \in \mathbb {R} ^ {n}} \{\mathrm{Tr} (\mathrm{diag} (\mathbf {x}) \mathbf {U} ^ {T} \mathbf {Y} \mathbf {U}) - f (\boldsymbol {\lambda} (\mathbf {U} \mathrm{diag} (\mathbf {x}) \mathbf {U} ^ {T})) \} \\ \qquad = \max _ {\mathbf {x} \in \mathbb {R} ^ {n}} \{\mathrm{Tr} (\mathbf {U} \mathrm{diag} (\mathbf {x}) \mathbf {U} ^ {T} \mathbf {Y}) - f (\boldsymbol {\lambda} (\mathbf {U} \mathrm{diag} (\mathbf {x}) \mathbf {U} ^ {T})) \} \\ \qquad \leq \max _ {\mathbf {Z} \in \mathbb {S} ^ {n}} \{\mathrm{Tr} (\mathbf {Z Y}) - f (\boldsymbol {\lambda} (\mathbf {Z})) \} \\ \qquad = (f \circ \boldsymbol {\lambda}) ^ {*} (\mathbf {Y}). \quad \square \end{array}
$$

Example 7.16. Using the spectral conjugate formula, we can compute the conjugates of the functions from the table of Example 7.13. The conjugates appear in the following table, which also includes references to the corresponding results for functions over $\mathbb { R } ^ { n }$ . The numbering is the same as in the table of Example 7.13. 


7.2. Symmetric Spectral Functions over $\mathbb { S } ^ { n }$


<table><tr><td>#</td><td>g(X)</td><td>dom(g)</td><td>g*(Y)</td><td>dom(g*)</td><td>Reference</td></tr><tr><td>1</td><td><eq>\text{Tr}(\mathbf{X})</eq></td><td><eq>\mathbb{S}^n</eq></td><td><eq>\delta_{\{\mathbf{I}\}}(\mathbf{Y})</eq></td><td><eq>\{\mathbf{I}\}</eq></td><td>Section 4.4.7</td></tr><tr><td>2</td><td><eq>\lambda_{\max}(\mathbf{X})</eq></td><td><eq>\mathbb{S}^n</eq></td><td><eq>\delta_{\Upsilon_n}(\mathbf{Y})</eq></td><td><eq>\Upsilon_n</eq></td><td>Example 4.10</td></tr><tr><td>3</td><td><eq>\alpha\|\mathbf{X}\|_F (\alpha&gt;0)</eq></td><td><eq>\mathbb{S}^n</eq></td><td><eq>\delta_{B_{\|\cdot\|_F}[\mathbf{0},\alpha]}(\mathbf{Y})</eq></td><td><eq>B_{\|\cdot\|_F}[\mathbf{0},\alpha]</eq></td><td>Section 4.4.12</td></tr><tr><td>4</td><td><eq>\alpha\|\mathbf{X}\|_F^2 (\alpha&gt;0)</eq></td><td><eq>\mathbb{S}^n</eq></td><td><eq>\frac{1}{4\alpha}\|\mathbf{Y}\|_F^2</eq></td><td><eq>\mathbb{S}^n</eq></td><td>Section 4.4.6</td></tr><tr><td>5</td><td><eq>\alpha\|\mathbf{X}\|_{2,2} (\alpha&gt;0)</eq></td><td><eq>\mathbb{S}^n</eq></td><td><eq>\delta_{B_{\|\cdot\|_{S_1}}[\mathbf{0},\alpha]}(\mathbf{Y})</eq></td><td><eq>B_{\|\cdot\|_{S_1}}[\mathbf{0},\alpha]</eq></td><td>Section 4.4.12</td></tr><tr><td>6</td><td><eq>\alpha\|\mathbf{X}\|_{S_1} (\alpha&gt;0)</eq></td><td><eq>\mathbb{S}^n</eq></td><td><eq>\delta_{B_{\|\cdot\|_{2,2}}[\mathbf{0},\alpha]}(\mathbf{Y})</eq></td><td><eq>B_{\|\cdot\|_{2,2}}[\mathbf{0},\alpha]</eq></td><td>Section 4.4.12</td></tr><tr><td>7</td><td><eq>-\log \det(\mathbf{X})</eq></td><td><eq>\mathbb{S}_{++}^n</eq></td><td><eq>-n-\log \det(-\mathbf{Y})</eq></td><td><eq>\mathbb{S}_{--}^n</eq></td><td>Section 4.4.9</td></tr><tr><td>8</td><td><eq>\sum_{i=1}^{n} \lambda_i(\mathbf{X}) \log(\lambda_i(\mathbf{X}))</eq></td><td><eq>\mathbb{S}_+^n</eq></td><td><eq>\sum_{i=1}^{n} e^{\lambda_i(\mathbf{Y})-1}</eq></td><td><eq>\mathbb{S}^n</eq></td><td>Section 4.4.8</td></tr><tr><td>9</td><td><eq>\sum_{i=1}^{n} \lambda_i(\mathbf{X}) \log(\lambda_i(\mathbf{X}))</eq></td><td><eq>\Upsilon_n</eq></td><td><eq>\log\left(\sum_{i=1}^{n} e^{\lambda_i(\mathbf{Y})}\right)</eq></td><td><eq>\mathbb{S}^n</eq></td><td>Section 4.4.10</td></tr></table>

The spectral conjugate formula has several important consequences, one of which is the following theorem stating that a symmetric spectral function is closed and convex if and only if its associated function is closed and convex. 

Theorem 7.17 (closedness and convexity of symmetric spectral functions over $\mathbb { S } ^ { n } )$ . Let $F : \mathbb { S } ^ { n }  ( - \infty , \infty ]$ be given by $F = f \circ \lambda$ , where $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ is a permutation symmetric proper function. Then F is closed and convex $i f$ and only if f is closed and convex. 

Proof. By the spectral conjugate formula (Theorem 7.15), 

$$
F ^ {*} = (f \circ \boldsymbol {\lambda}) ^ {*} = f ^ {*} \circ \boldsymbol {\lambda}.
$$

Since by the symmetric conjugate theorem (Theorem 7.9) $f ^ { * }$ is permutation symmetric, we can invoke once again the spectral conjugate formula to obtain 

$$
F ^ {* *} = (f ^ {*} \circ \boldsymbol {\lambda}) ^ {*} = f ^ {* *} \circ \boldsymbol {\lambda}.\tag{7.2}
$$

If f is closed and convex, then by Theorem 4.8 (taking also into account the properness of $f )$ , it follows that $f ^ { * * } = f$ . Therefore, by (7.2), 

$$
F ^ {* *} = f \circ \boldsymbol {\lambda} = F.
$$

Thus, since $F$ is a conjugate of another function $( F ^ { * } )$ , it follows by Theorem 4.3 that it is closed and convex. Now assume that F is closed and convex. Since F is in addition proper, it follows by Theorem 4.8 that $F ^ { * * } = F$ , which, combined with (7.2), yields the equality 

$$
f \circ \boldsymbol {\lambda} = F = F ^ {* *} = f ^ {* *} \circ \boldsymbol {\lambda}.
$$

Therefore, for any $\mathbf { x } \in \mathbb { R } ^ { n }$ 

$$
f (\mathbf {x} ^ {\downarrow}) = f (\boldsymbol {\lambda} (\mathrm{diag} (\mathbf {x}))) = f ^ {* *} (\boldsymbol {\lambda} (\mathrm{diag} (\mathbf {x}))) = f ^ {* *} (\mathbf {x} ^ {\downarrow}).
$$

By the permutation symmetry property of both $f$ and $f ^ { * * }$ , it follows that $f ( \mathbf { x } ^ { \downarrow } ) =$ $f ( \mathbf { x } )$ and $f ^ { * * } ( { \bf x } ^ { \downarrow } ) = f ^ { * * } ( { \bf x } )$ , and we thus obtained that $f ( { \bf x } ) = f ^ { * * } ( { \bf x } )$ for any $\mathbf { x } \in \mathbb { R } ^ { n }$ , meaning that $f = f ^ { * * }$ . Therefore, $f ,$ as a conjugate of another function $( f ^ { * } )$ is closed and convex. 

## 7.2.2 The Proximal Operator of Symmetric Spectral Functions over $\mathbb { S } ^ { n }$

The next result shows a simple formula for computing the prox operator of a symmetric spectral function over $\mathbb { S } ^ { n }$ which is also proper closed and convex. The prox is expressed in terms of the spectral decomposition of the argument and the prox operator of the associated function. 

Theorem 7.18 (spectral prox formula over $\mathbb { S } ^ { n } )$ . Let $F : \mathbb { S } ^ { n }  ( - \infty , \infty ]$ be given by $F = f \circ \lambda$ , where $f : \mathbb { R } ^ { n }  ( - \infty , \infty ]$ is a permutation symmetric proper closed and convex function. Let $\mathbf { X } \in \mathbb { S } ^ { n }$ , and suppose that $\mathbf { X } = \bar { \mathbf { U } } \mathrm { d i a g } ( \lambda ( \mathbf { X } ) ) \bar { \mathbf { U } } ^ { T }$ where $\mathbf { U } \in \mathbb { O } ^ { n }$ . Then 

$$
\operatorname{prox} _ {F} (\mathbf {X}) = \mathbf {U} \text { diag } (\operatorname{prox} _ {f} (\boldsymbol {\lambda} (\mathbf {X}))) \mathbf {U} ^ {T}.
$$

Proof. Recall that 

$$
\operatorname{prox} _ {F} (\mathbf {X}) = \operatorname{argmin} _ {\mathbf {Z} \in \mathbb {S} ^ {n}} \left\{F (\mathbf {Z}) + \frac {1}{2} \| \mathbf {Z} - \mathbf {X} \| _ {F} ^ {2} \right\}.\tag{7.3}
$$

Denoting $\mathbf { D } = \mathrm { d i a g } ( \lambda ( \mathbf { X } ) )$ , we note that for any $\mathbf { Z } \in \mathbb { S } ^ { n }$ 

$$
F (\mathbf {Z}) + \frac {1}{2} \| \mathbf {Z} - \mathbf {X} \| _ {F} ^ {2} = F (\mathbf {Z}) + \frac {1}{2} \| \mathbf {Z} - \mathbf {U D U} ^ {T} \| _ {F} ^ {2} \stackrel {(*)} {=} F (\mathbf {U} ^ {T} \mathbf {Z U}) + \frac {1}{2} \| \mathbf {U} ^ {T} \mathbf {Z U} - \mathbf {D} \| _ {F} ^ {2},
$$

where the transition $( * )$ is due to the fact that $F ( \mathbf { Z } ) = f ( \lambda ( \mathbf { Z } ) ) = f ( \lambda ( \mathbf { U } ^ { T } \mathbf { Z } \mathbf { U } ) ) =$ ${ \cal F } ( { \bf U } ^ { T } { \bf Z U } )$ . Making the change of variables $\mathbf { W } = \mathbf { U } ^ { T } \mathbf { Z } \mathbf { U }$ , we conclude that the optimal solution of (7.3) is given by 

$$
\mathbf {Z} = \mathbf {U W} ^ {*} \mathbf {U} ^ {T},\tag{7.4}
$$

where $\mathbf { W } ^ { * } \in \mathbb { S } ^ { n }$ is the unique optimal solution of 

$$
\min _ {\mathbf {W} \in \mathbb {S} ^ {n}} \left\{G (\mathbf {W}) \equiv F (\mathbf {W}) + \frac {1}{2} \| \mathbf {W} - \mathbf {D} \| _ {F} ^ {2} \right\}.\tag{7.5}
$$

We will prove that $\mathbf { W } ^ { * }$ is diagonal. Let $i \in \{ 1 , 2 , \ldots , n \}$ . Take $\mathbf { V } _ { i }$ to be the diagonal matrix whose diagonal elements are all ones except for the $( i , i )$ th component, which is −1. Define $\widetilde { \mathbf { W } } _ { i } = \mathbf { V } _ { i } \mathbf { W } ^ { * } \mathbf { V } _ { i } ^ { T }$ . Obviously, by the fact that $\mathbf { V } _ { i } \in \mathbb { O } ^ { n }$ 

$$
F (\mathbf {V} _ {i} \mathbf {W} ^ {*} \mathbf {V} _ {i} ^ {T}) = f (\boldsymbol {\lambda} (\mathbf {V} _ {i} \mathbf {W} ^ {*} \mathbf {V} _ {i} ^ {T})) = f (\boldsymbol {\lambda} (\mathbf {W} ^ {*})) = F (\mathbf {W} ^ {*}),
$$

and we thus obtain 

$$
\begin{array}{r c l} G (\widetilde {\mathbf {W}} _ {i}) & = & F (\widetilde {\mathbf {W}} _ {i}) + \frac {1}{2} \| \widetilde {\mathbf {W}} _ {i} - \mathbf {D} \| _ {F} ^ {2} \\ & = & F (\mathbf {V} _ {i} \mathbf {W} ^ {*} \mathbf {V} _ {i} ^ {T}) + \frac {1}{2} \| \mathbf {V} _ {i} \mathbf {W} ^ {*} \mathbf {V} _ {i} ^ {T} - \mathbf {D} \| _ {F} ^ {2} \\ & = & F (\mathbf {W} ^ {*}) + \frac {1}{2} \| \mathbf {W} ^ {*} - \mathbf {V} _ {i} ^ {T} \mathbf {D V} _ {i} \| _ {F} ^ {2} \\ & \stackrel {(* *)} {=} & F (\mathbf {W} ^ {*}) + \frac {1}{2} \| \mathbf {W} ^ {*} - \mathbf {D} \| _ {F} ^ {2}, \\ & = & G (\mathbf {W} ^ {*}), \end{array}
$$

where (**) follows from the fact that $\mathbf { V } _ { i }$ and D are both diagonal, and hence $\mathbf { V } _ { i } ^ { T } \mathbf { D } \mathbf { V } _ { i } = \mathbf { V } _ { i } ^ { T } \mathbf { V } _ { i } \mathbf { D } = \mathbf { D }$ . We conclude that $\widetilde { \mathbf { W } } _ { i }$ is also an optimal solution, but ;by the uniqueness of the optimal solution of problem (7.5), it follows that $\mathbf { W ^ { * } } =$ $\mathbf { V } _ { i } \mathbf { W } ^ { * } \mathbf { V } _ { i } ^ { T }$ . Comparing the ith rows of the two matrices, we deduce that $W _ { i j } ^ { * } = 0$ for any $j \neq i$ . Since this argument is valid for any $i \in \{ 1 , 2 , \ldots , n \}$ , it follows that $\mathbf { W } ^ { * }$ is a diagonal matrix, and consequently the optimal solution of (7.5) is given by $\mathbf { W ^ { * } } = \mathrm { d i a g } ( \mathbf { w } ^ { * } )$ , where $\mathbf { w } ^ { * }$ is the optimal solution of 

$$
\min _ {\mathbf {w}} \left\{F (\mathrm{diag} (\mathbf {w})) + \frac {1}{2} \| \mathrm{diag} (\mathbf {w}) - \mathbf {D} \| _ {F} ^ {2} \right\}.
$$

Since $F ( \mathrm { d i a g } ( \mathbf { w } ) ) = f ( \mathbf { w } ^ { \downarrow } ) = f ( \mathbf { w } ) \mathrm { a n d } \| \mathrm { d i a g } ( \mathbf { w } ) - \mathbf { D } \| _ { F } ^ { 2 } = \| \mathbf { w } - \pmb { \lambda } ( \mathbf { X } ) \| _ { 2 } ^ { 2 }$ , it follows that $\mathbf { w } ^ { * }$ is given by 

$$
\mathbf {w} ^ {*} = \operatorname{argmin} _ {\mathbf {w}} \left\{f (\mathbf {w}) + \frac {1}{2} \| \mathbf {w} - \boldsymbol {\lambda} (\mathbf {X}) \| _ {2} ^ {2} \right\} = \operatorname{prox} _ {f} (\boldsymbol {\lambda} (\mathbf {X})).
$$

Therefore, $\mathbf { W } ^ { * } = \mathrm { d i a g } ( \mathrm { p r o x } _ { f } ( \boldsymbol { \lambda } ( \mathbf { X } ) ) )$ , which, along with (7.4), establishes the desired result. 


Example 7.19. Using the spectral prox formula, we can compute the prox of symmetric spectral functions in terms of the prox of their associated functions. Using this observation, we present in the table below expressions of prox operators of several functions. The parameter $\alpha$ is always assumed to be positive, and U is assumed to be an orthogonal matrix satisfying $\mathbf { X } = \mathbf { U } \mathrm { d i a g } ( \lambda ( \mathbf { X } ) ) \mathbf { U } ^ { T }$ . The table also includes references to the corresponding results for the associated functions, which are always defined over $\mathbb { R } ^ { n }$


<table><tr><td><eq>F(\mathbf{X})</eq></td><td><eq>\text{dom}(F)</eq></td><td><eq>\text{prox}_F(\mathbf{X})</eq></td><td>Reference</td></tr><tr><td><eq>\alpha\|\mathbf{X}\|_F^2</eq></td><td><eq>\mathbb{S}^n</eq></td><td><eq>\frac{1}{1+2\alpha}\mathbf{X}</eq></td><td>Section 6.2.3</td></tr><tr><td><eq>\alpha\|\mathbf{X}\|_F</eq></td><td><eq>\mathbb{S}^n</eq></td><td><eq>\left(1-\frac{\alpha}{\max\{ \| \mathbf{X}\|_F,\alpha\}}\right)\mathbf{X}</eq></td><td>Example 6.19</td></tr><tr><td><eq>\alpha\|\mathbf{X}\|_{S_1}</eq></td><td><eq>\mathbb{S}^n</eq></td><td><eq>\mathbf{Udiag}(\mathcal{T}_{\alpha}(\boldsymbol{\lambda}(\mathbf{X})))\mathbf{U}^T</eq></td><td>Example 6.8</td></tr><tr><td><eq>\alpha\|\mathbf{X}\|_{2,2}</eq></td><td><eq>\mathbb{S}^n</eq></td><td><eq>\mathbf{Udiag}(\boldsymbol{\lambda}(\mathbf{X})-\alpha P_{B_{\|\cdot\|_1}[0,1]}(\boldsymbol{\lambda}(\mathbf{X})/\alpha))\mathbf{U}^T</eq></td><td>Example 6.48</td></tr><tr><td><eq>-\alpha\log\det(\mathbf{X})</eq></td><td><eq>\mathbb{S}_{++}^n</eq></td><td><eq>\mathbf{Udiag}\left(\frac{\lambda_j(\mathbf{X})+\sqrt{\lambda_j(\mathbf{X})^2+4\alpha}}{2}\right)\mathbf{U}^T</eq></td><td>Example 6.9</td></tr><tr><td><eq>\alpha\lambda_1(\mathbf{X})</eq></td><td><eq>\mathbb{S}^n</eq></td><td><eq>\mathbf{Udiag}(\boldsymbol{\lambda}(\mathbf{X})-\alpha P_{\Delta_n}(\boldsymbol{\lambda}(\mathbf{X})/\alpha))\mathbf{U}^T</eq></td><td>Example 6.49</td></tr><tr><td><eq>\alpha\sum_{i=1}^{k}\lambda_i(\mathbf{X})</eq></td><td><eq>\mathbb{S}^n</eq></td><td><eq>\mathbf{X}-\alpha\mathbf{Udiag}(P_C(\boldsymbol{\lambda}(\mathbf{X})/\alpha))\mathbf{U}^T,</eq><eq>C = H_{\mathbf{e},k} \cap \text{Box}[\mathbf{0},\mathbf{e}]</eq></td><td>Example 6.50</td></tr></table>

A set $T \subseteq \mathbb { S } ^ { n }$ is called a symmetric spectral set in $\mathbb { S } ^ { n }$ if the indicator function $\delta _ { T }$ is a symmetric spectral function over $\mathbb { S } ^ { n }$ , meaning that it has the form $\delta _ { T } = \delta _ { C } \circ \lambda$ 2 where $\delta _ { C }$ is a permutation symmetric function. The set $C \subseteq \mathbb { R } ^ { n }$ is the associated set. Since $\mathrm { p r o x } _ { \delta _ { T } } = P _ { T }$ and $\mathrm { p r o x } _ { \delta _ { C } } = P _ { C }$ , it follows by the spectral prox formula that if C is nonempty closed and convex, then 

$$
P _ {T} (\mathbf {X}) = \mathbf {U} \mathrm{diag} (P _ {C} (\boldsymbol {\lambda} (\mathbf {X}))) \mathbf {U} ^ {T}, \mathbf {X} \in \mathbb {S} ^ {n},\tag{7.6}
$$

where U is an orthogonal matrix satisfying $\mathbf { X } = \mathbf { U } \mathrm { d i a g } ( \lambda ( \mathbf { X } ) ) \mathbf { U } ^ { T }$ 

Example 7.20. Using formula (7.6), we present in the following table expressions for the orthogonal projection onto several symmetric spectral sets in $\mathbb { S } ^ { n }$ . The table also includes references to the corresponding results on orthogonal projections onto the associated subsets of $\mathbb { R } ^ { n }$ . The matrix U is assumed to be an orthogonal matrix satisfying $\mathbf { X } = \mathbf { U } \mathrm { d i a g } ( \lambda ( \mathbf { X } ) ) \mathbf { U } ^ { T }$ 

<table><tr><td>set (T)</td><td><eq>P_{T}(\mathbf{X})</eq></td><td>Assumptions</td><td>Reference</td></tr><tr><td><eq>\mathbb{S}_{+}^{n}</eq></td><td><eq>\textbf{Udiag}([\boldsymbol{\lambda}(\mathbf{X})]_{+})\mathbf{U}^{T}</eq></td><td>-</td><td>Lemma 6.26</td></tr><tr><td><eq>\{\mathbf{X}:\ell\mathbf{I}\preceq\mathbf{X}\preceq u\mathbf{I}\}</eq></td><td><eq>\textbf{Udiag}(\mathbf{v})\mathbf{U}^{T},</eq><eq>v_{i}=\min\{\max\{\lambda_{i}(\mathbf{X}),\ell\},u\}</eq></td><td><eq>\ell\leq u</eq></td><td>Lemma 6.26</td></tr><tr><td><eq>B_{\parallel\cdot\parallel_{F}}[\mathbf{0},r]</eq></td><td><eq>\frac{r}{\max\{\|\mathbf{X}\|_{F},r\}}\mathbf{X}</eq></td><td><eq>r&gt;0</eq></td><td>Lemma 6.26</td></tr><tr><td><eq>\{\mathbf{X}:\text{Tr}(\mathbf{X})\leq b\}</eq></td><td><eq>\textbf{Udiag}(\mathbf{v})\mathbf{U}^{T},\mathbf{v}=\boldsymbol{\lambda}(\mathbf{X})-\frac{[\mathbf{e}^{T}\boldsymbol{\lambda}(\mathbf{X})-b]_{+}}{n}\mathbf{e}</eq></td><td><eq>b\in\mathbb{R}</eq></td><td>Lemma 6.26</td></tr><tr><td><eq>\Upsilon_{n}</eq></td><td><eq>\textbf{Udiag}(\mathbf{v})\mathbf{U}^{T},\mathbf{v}=[\boldsymbol{\lambda}(\mathbf{X})-\boldsymbol{\mu}^{*}\mathbf{e}]_{+}</eq> where <eq>\boldsymbol{\mu}^{*}\in\mathbb{R}</eq> satisfies <eq>\mathbf{e}^{T}[\boldsymbol{\lambda}(\mathbf{X})-\boldsymbol{\mu}^{*}\mathbf{e}]_{+}=1</eq></td><td>-</td><td>Corollary 6.29</td></tr><tr><td><eq>B_{\parallel\cdot\parallel_{S_{1}}}[0,\alpha]</eq></td><td><eq>\left\{\begin{array}{ll}\mathbf{X},&amp;\|\mathbf{X}\|_{S_{1}}\leq\alpha,\\\mathbf{Udiag}(\mathcal{T}_{\beta^{*}}(\boldsymbol{\lambda}(\mathbf{X})))\mathbf{U}^{T},&amp;\|\mathbf{X}\|_{S_{1}}&gt; \alpha,\\ \|\mathcal{T}_{\beta^{*}}(\boldsymbol{\lambda}(\mathbf{X}))\|_{1}=\alpha,\beta^{*}&gt;0\end{array}\right.</eq></td><td><eq>\alpha&gt;0</eq></td><td>Example 6.33</td></tr></table>

## 7.3 Symmetric Spectral Functions over $\mathbb { R } ^ { m \times n }$

Let $m , n$ be two positive integers and $r ~ = ~ \mathrm { m i n } \{ m , n \}$ . We will denote by $\sigma :$ $\mathbb { R } ^ { m \times n }  \mathbb { R } ^ { r }$ the singular values function that assigns to each matrix $\mathbf { X } \in \mathbb { R }$ m×n the vector of singular values $( \sigma _ { 1 } ( { \mathbf { X } } ) , \sigma _ { 2 } ( { \mathbf { X } } ) , \ldots , \sigma _ { r } ( { \mathbf { X } } ) ) ^ { T }$ , where $\sigma _ { 1 } ( \mathbf { X } ) \geq \sigma _ { 2 } ( \mathbf { X } ) \geq$ $\begin{array} { r } { \cdots \geq \sigma _ { r } ( { \bf X } ) \geq 0 } \end{array}$ . We will also require the following notation. For a vector $\mathbf { v } \in \mathbb { R } ^ { r }$ ， the matrix $\mathrm { d g } ( \mathbf { v } )$ is the $m \times n$ matrix defined by 

$$
\mathrm{dg} (\mathbf {v}) _ {i, j} = \left\{ \begin{array}{l l} v _ {i}, & i = j, \\ 0 & \text { else }. \end{array} \right.
$$

The operator $\deg ( \cdot )$ maps r-dimensional vectors to $\mathrm { g e n e r a l i z e d } ^ { 3 8 } \ m \ \times \ n$ diagonal matrices. The integers m and $n$ (and hence also $r )$ will be fixed throughout this section, and hence there is no need to indicate their values in the operator dg. We do not use the $\mathrm { \ddot { \Phi } d i a g \mathrm { \uparrow } }$ notation since it is reserved to square diagonal matrices. 

It is well known (see Golub and Van Loan [60, Theorem 2.5.2]) that any matrix $\mathbf { X } \in \mathbb { R } ^ { m \times n }$ has a singular value decomposition, meaning matrices $\mathbf { U } \in \mathbb { O } ^ { m } , \mathbf { V } \in \mathbb { O } ^ { n }$ for which ${ \bf X } = { \bf U } \mathrm { d g } ( { \bf \bar { \sigma } } ( { \bf X } ) ) { \bf V } ^ { T }$ 

The analysis in this section uses very similar arguments to those used in the previous section; however, for the sake of completeness we will provide the results with their complete proofs. 

We begin by formally defining the notion of spectral functions over $\mathbb { R } ^ { m \times n }$ 

Definition 7.21 (spectral functions over $\mathbb { R } ^ { m \times n } )$ . A proper function $\textit { g } :$ $\mathbb { R } ^ { m \times n }  ( - \infty , \infty ]$ is called a spectral function over $\mathbb { R } ^ { m \times n }$ if there exists a proper function $f : \mathbb { R } ^ { r }  ( - \infty , \infty ]$ for which $g = f \circ \sigma$ 

Similarly to the notation in Section 7.2, if $g = f \circ \sigma$ , we will refer to $f$ (which is actually not necessarily unique) as the associated function. Our main interest will be with spectral functions whose associated functions are absolutely permutation symmetric. 

Definition 7.22 (symmetric spectral functions over $\mathbb { R } ^ { m \times n } )$ . A proper function $f ~ : ~ \mathbb { R } ^ { m \times n } ~  ~ ( - \infty , \infty ]$ is called a symmetric spectral function over R<sup>m</sup>×<sup>n</sup> if there exists a proper absolutely permutation symmetric function $f : \mathbb { R } ^ { r } $ $( - \infty , \infty ]$ for which $g = f \circ \sigma$ 

Example 7.23 (Schatten p-norms). Let $p \in [ 1 , \infty ]$ . Then the Schatten p-norm is the norm defined by 

$$
\| \mathbf {X} \| _ {S _ {p}} \equiv \| \boldsymbol {\sigma} (\mathbf {X}) \| _ {p}, \mathbf {X} \in \mathbb {R} ^ {m \times n}.
$$

It is well $\mathrm { k n o w n ^ { 3 9 } }$ that $\| \cdot \| _ { S _ { p } }$ is indeed a norm for any $p \in [ 1 , \infty ]$ . Specific examples are the following: 

• trace-norm (Schatten 1-norm)—also called the nuclear norm: 

$$
\| \mathbf {X} \| _ {S _ {1}} = \sum_ {i = 1} ^ {r} \sigma_ {i} (\mathbf {X}).
$$

• spectral norm (Schatten ∞-norm): 

$$
\| \mathbf {X} \| _ {S _ {\infty}} = \sigma_ {1} (\mathbf {X}) = \| \mathbf {X} \| _ {2, 2}.
$$

• Frobenius norm (Schatten 2-norm): 

$$
\| \mathbf {X} \| _ {S _ {2}} = \sqrt {\sum_ {i = 1} ^ {r} \sigma_ {i} (\mathbf {X}) ^ {2}} = \sqrt {\operatorname{Tr} \left(\mathbf {X} ^ {T} \mathbf {X}\right)}.
$$

The Schatten p-norm is a symmetric spectral function over $\mathbb { R } ^ { m \times n }$ whose associated function is the $l _ { p } .$ -norm on $\mathbb { R } ^ { r }$ , which is obviously an absolutely permutation symmetric function. 

Example 7.24 (Ky Fan k-norms). Recall the notation from Example 6.51— given a vector $\mathbf { x } \in \mathbb { R } ^ { r } , \ x _ { \langle i \rangle }$ is the component of x with the ith largest absolute value, meaning in particular that 

$$
\left| x _ {\langle 1 \rangle} \right| \geq \left| x _ {\langle 2 \rangle} \right| \geq \dots \geq \left| x _ {\langle r \rangle} \right|.
$$

The function $\begin{array} { r } { f _ { k } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { k } | x _ { \langle i \rangle } | } \end{array}$ is an absolutely permutation symmetric function. The corresponding symmetric spectral function is the so-called $K y$ Fan k-norm given by 

$$
\| \mathbf {X} \| _ {\langle k \rangle} = f _ {k} (\sigma (\mathbf {X})) = \sum_ {i = 1} ^ {k} \sigma_ {i} (\mathbf {X}).
$$

Obviously, $\| \cdot \| _ { \langle 1 \rangle }$ is the spectral norm, which is also the Schatten ∞-norm; the norm $\| \cdot \| _ { \langle r \rangle }$ is the trace-norm, which is also the Schatten 1-norm. 

A key inequality that is used in the analysis of spectral functions over $\mathbb { R } ^ { m \times n }$ is an inequality bounding the inner product of two matrices via the inner product of their singular vectors. The inequality, which is credited to von Neumann and is in a sense the $\mathbb { R } ^ { m \times n }$ -counterpart” of Fan’s inequality (Theorem 7.14). 

Theorem 7.25 (von Neumann’s trace inequality [123]). For any two matrices $\mathbf { X } , \mathbf { Y } \in \mathbb { R } ^ { m \times n }$ , the inequality 

$$
\langle \mathbf {X}, \mathbf {Y} \rangle \leq \langle \boldsymbol {\sigma} (\mathbf {X}), \boldsymbol {\sigma} (\mathbf {Y}) \rangle
$$

holds. Equality holds if and only if there exists a simultaneous nonincreasing singular value decomposition of X, Y, meaning that there exist $\mathbf { U } \in \mathbb { O } ^ { m }$ and $\mathbf { V } \in \mathbb { O } ^ { n }$ for which 

$$
\begin{array}{r} \mathbf {X} = \mathbf {U} \mathrm{dg} (\pmb {\sigma} (\mathbf {X})) \mathbf {V} ^ {T}, \\ \mathbf {Y} = \mathbf {U} \mathrm{dg} (\pmb {\sigma} (\mathbf {Y})) \mathbf {V} ^ {T}. \end{array}
$$

## 7.3.1 The Spectral Conjugate Formula

A direct result of von Neumann’s trace inequality is the spectral conjugate formula over $\mathbb { R } ^ { m \times n }$ 

Theorem 7.26 (spectral conjugate formula over $\mathbb { R } ^ { m \times n } )$ . Let $f : \mathbb { E } $ $( - \infty , \infty ]$ be an absolutely permutation symmetric function. Then 

$$
(f \circ \boldsymbol {\sigma}) ^ {*} = f ^ {*} \circ \boldsymbol {\sigma}.
$$

Proof. Let $\mathbf { Y } \in \mathbb { R } ^ { m \times n }$ . Then 

$$
\begin{array}{l} (f \circ \boldsymbol {\sigma}) ^ {*} (\mathbf {Y}) = \max _ {\mathbf {X} \in \mathbb {R} ^ {m \times n}} \{\mathrm{Tr} (\mathbf {X Y}) - f (\boldsymbol {\sigma} (\mathbf {X})) \} \\ \leq \max _ {\mathbf {X} \in \mathbb {R} ^ {m \times n}} \{\langle \boldsymbol {\sigma} (\mathbf {X}), \boldsymbol {\sigma} (\mathbf {Y}) \rangle - f (\boldsymbol {\sigma} (\mathbf {X})) \} \\ \leq \max _ {\mathbf {x} \in \mathbb {R} ^ {r}} \{\langle \mathbf {x}, \boldsymbol {\sigma} (\mathbf {Y}) \rangle - f (\mathbf {x}) \} \\ = (f ^ {*} \circ \boldsymbol {\sigma}) (\mathbf {Y}), \end{array}
$$

where Von Neumann’s trace inequality (Theorem 7.25) was used in the first inequality. To show the reverse inequality, take a singular value decomposition of Y: 

$$
\mathbf {Y} = \mathbf {U} \mathrm{dg} (\boldsymbol {\sigma} (\mathbf {Y})) \mathbf {V} ^ {T} (\mathbf {U} \in \mathbb {O} ^ {m}, \mathbf {V} \in \mathbb {O} ^ {n}).
$$

Then 

$$
\begin{array}{l} (f ^ {*} \circ \boldsymbol {\sigma}) (\mathbf {Y}) = \underset {\mathbf {x} \in \mathbb {R} ^ {r}} {\max} \{\langle \mathbf {x}, \boldsymbol {\sigma} (\mathbf {Y}) \rangle - f (\mathbf {x}) \} \\ \qquad = \underset {\mathbf {x} \in \mathbb {R} ^ {r}} {\max} \{\mathrm{Tr} (\mathrm{dg} (\mathbf {x}) ^ {T} \mathrm{dg} (\boldsymbol {\sigma} (\mathbf {Y}))) - f (\mathbf {x}) \} \\ \qquad = \underset {\mathbf {x} \in \mathbb {R} ^ {r}} {\max} \{\mathrm{Tr} (\mathrm{dg} (\mathbf {x}) ^ {T} \mathbf {U} ^ {T} \mathbf {Y V}) - f (\mathbf {x} ^ {\downarrow}) \} \\ \qquad = \underset {\mathbf {x} \in \mathbb {R} ^ {r}} {\max} \{\mathrm{Tr} (\mathrm{dg} (\mathbf {x}) ^ {T} \mathbf {U} ^ {T} \mathbf {Y V}) - f (\boldsymbol {\sigma} (\mathbf {U d g} (\mathbf {x}) \mathbf {V} ^ {T})) \} \\ \qquad = \underset {\mathbf {x} \in \mathbb {R} ^ {r}} {\max} \{\mathrm{Tr} (\mathbf {V d g} (\mathbf {x}) ^ {T} \mathbf {U} ^ {T} \mathbf {Y}) - f (\boldsymbol {\sigma} (\mathbf {U d g} (\mathbf {x}) \mathbf {V} ^ {T})) \} \\ \qquad \leq \underset {\mathbf {Z} \in \mathbb {R} ^ {m \times n}} {\max} \{\mathrm{Tr} (\mathbf {Z} ^ {T} \mathbf {Y}) - f (\boldsymbol {\sigma} (\mathbf {Z})) \} \\ \qquad = (f \circ \boldsymbol {\sigma}) ^ {*} (\mathbf {Y}). \quad \square \end{array}
$$


Example 7.27. Using the spectral conjugate formula over $\mathbb { R } ^ { m \times n }$ , we present below expressions for the conjugate functions of several symmetric spectral functions over $\mathbb { R } ^ { m \times n }$ (all with full domain). The table also includes the references to the corresponding results on functions over $\mathbb { R } ^ { r }$ . The constant α is assumed to be positive.


<table><tr><td>g(X)</td><td>dom(g)</td><td>g*(Y)</td><td>dom(g*)</td><td>Reference</td></tr><tr><td><eq>\alpha\sigma_1(\mathbf{X}) (\alpha &gt;0)</eq></td><td><eq>\mathbb{R}^{m\times n}</eq></td><td><eq>\delta_{B_{\|\cdot\|_{S_1}}[\mathbf{0},\alpha]}(\mathbf{Y})</eq></td><td><eq>B_{\|\cdot\|_{S_1}}[\mathbf{0},\alpha]</eq></td><td>Section 4.4.12</td></tr><tr><td><eq>\alpha\|\mathbf{X}\|_F (\alpha &gt;0)</eq></td><td><eq>\mathbb{R}^{m\times n}</eq></td><td><eq>\delta_{B_{\|\cdot\|_F}[\mathbf{0},\alpha]}(\mathbf{Y})</eq></td><td><eq>B_{\|\cdot\|_F}[\mathbf{0},\alpha]</eq></td><td>Section 4.4.12</td></tr><tr><td><eq>\alpha\|\mathbf{X}\|_F^2 (\alpha &gt;0)</eq></td><td><eq>\mathbb{R}^{m\times n}</eq></td><td><eq>\frac{1}{4\alpha}\|\mathbf{Y}\|_F^2</eq></td><td><eq>\mathbb{R}^{m\times n}</eq></td><td>Section 4.4.6</td></tr><tr><td><eq>\alpha\|\mathbf{X}\|_{S_1} (\alpha &gt;0)</eq></td><td><eq>\mathbb{R}^{m\times n}</eq></td><td><eq>\delta_{B_{\|\cdot\|_{S_\infty}}[\mathbf{0},\alpha]}(\mathbf{Y})</eq></td><td><eq>B_{\|\cdot\|_{S_\infty}}[\mathbf{0},\alpha]</eq></td><td>Section 4.4.12</td></tr></table>

The spectral conjugate formula can be used to show that a symmetric spectral function over $\mathbb { R } ^ { m \times n }$ is closed and convex if and only if its associated function is closed and convex. 

Theorem 7.28 (closedness and convexity of symmetric spectral functions over $\mathbb { R } ^ { m \times n } )$ . Let $F : \mathbb { R } ^ { m \times n }  ( - \infty , \infty ]$ be given by $F = f \circ \sigma _ { ; }$ , where $f : \mathbb { R } ^ { r } $ $( - \infty , \infty ]$ is an absolutely permutation symmetric proper function. Then F is closed and convex if and only if f is closed and convex. 

Proof. By the spectral conjugate formula (Theorem 7.26), 

$$
F ^ {*} = (f \circ \boldsymbol {\sigma}) ^ {*} = f ^ {*} \circ \boldsymbol {\sigma}.
$$

Since by the symmetric conjugate theorem (Theorem 7.9) $f ^ { * }$ is absolutely permutation symmetric, we can invoke once again the spectral conjugate formula to obtain 

$$
F ^ {* *} = (f ^ {*} \circ \boldsymbol {\sigma}) ^ {*} = f ^ {* *} \circ \boldsymbol {\sigma}.\tag{7.7}
$$

If $f$ is closed and convex, then by Theorem 4.8 (taking also in account the properness of $f )$ it follows that $f ^ { * * } = f$ . Therefore, by (7.7), 

$$
F ^ {* *} = f \circ \pmb {\sigma} = F.
$$

Thus, since $F$ is a conjugate of another function $( F ^ { * } )$ , it follows by Theorem 4.3 that it is closed and convex. Now assume that $F$ is closed and convex. Since $F$ is in addition proper, it follows by Theorem 4.8 that $F ^ { * * } = F$ , which, combined with (7.7), yields the equality 

$$
f \circ \boldsymbol {\sigma} = F = F ^ {* *} = f ^ {* *} \circ \boldsymbol {\sigma}.
$$

Therefore, for any $\mathbf { x } \in \mathbb { R } ^ { r }$ 

$$
f (| \mathbf {x} | ^ {\downarrow}) = f (\pmb {\sigma} (\mathrm{dg} (\mathbf {x}))) = f ^ {* *} (\pmb {\sigma} (\mathrm{dg} (\mathbf {x}))) = f ^ {* *} (| \mathbf {x} | ^ {\downarrow}).
$$

By the absolutely permutation symmetry property of both f and $f ^ { * * }$ , it follows that $f ( | \mathbf { x } | ^ { \downarrow } ) = f ( \mathbf { x } )$ and $f ^ { * * } ( | \mathbf { x } | ^ { \downarrow } ) = f ^ { * * } ( \mathbf { x } )$ , and therefore $f ( { \bf x } ) = f ^ { * * } ( { \bf x } )$ for any $\mathbf { x } \in \mathbb { R } ^ { r }$ , meaning that $f = f ^ { * * }$ . Therefore, $f ,$ as a conjugate of another function $( f ^ { * } )$ , is closed and convex. 

## 7.3.2 The Proximal Operator of Symmetric Spectral Functions over R<sup>m</sup>×<sup>n</sup>

The next result shows a simple formula for computing the prox operator of a symmetric spectral function over $\mathbb { R } ^ { m \times n }$ , which is also proper closed and convex. The prox is expressed in terms of the singular value decomposition of the argument and the prox operator of the associated function. 

Theorem 7.29 (spectral prox formula over $\mathbb { R } ^ { m \times n } )$ . Let $F : \mathbb { R } ^ { m \times n }  ( - \infty , \infty ]$ be given by $F = f \circ \sigma$ , where $f : \mathbb { R } ^ { r }  ( - \infty , \infty ]$ is an absolutely permutation symmetric proper closed and convex function. Let $\mathbf { X } \in \mathbb { R } ^ { m \times n }$ , and suppose that ${ \bf X } = { \bf U } \mathrm { d } { \bf g } ( { \pmb \sigma } ( { \bf X } ) ) { \bf V } ^ { T }$ , where $\mathbf { U } \in \mathbb { O } ^ { m } , \mathbf { V } \in \mathbb { O } ^ { n }$ . Then 

$$
\operatorname{prox} _ {F} (\mathbf {X}) = \mathbf {U} \mathrm{dg} (\operatorname{prox} _ {f} (\boldsymbol {\sigma} (\mathbf {X}))) \mathbf {V} ^ {T}.
$$

Proof. Recall that 

$$
\operatorname{prox} _ {F} (\mathbf {X}) = \operatorname{argmin} _ {\mathbf {Z} \in \mathbb {R} ^ {m \times n}} \left\{F (\mathbf {Z}) + \frac {1}{2} \| \mathbf {Z} - \mathbf {X} \| _ {F} ^ {2} \right\}.\tag{7.8}
$$

Denoting $\mathbf { D } = \deg ( \pmb { \sigma } ( \mathbf { X } ) )$ , we note that for any $\mathbf { Z } \in \mathbb { R } ^ { m \times n }$ 

$$
F (\mathbf {Z}) + \frac {1}{2} \| \mathbf {Z} - \mathbf {X} \| _ {F} ^ {2} = F (\mathbf {Z}) + \frac {1}{2} \| \mathbf {Z} - \mathbf {U D V} ^ {T} \| _ {F} ^ {2} \stackrel {(*)} {=} F (\mathbf {U} ^ {T} \mathbf {Z V}) + \frac {1}{2} \| \mathbf {U} ^ {T} \mathbf {Z V} - \mathbf {D} \| _ {F} ^ {2},
$$

where the transition $( * )$ is due to the fact that $F ( \mathbf { Z } ) = f ( \pmb { \sigma } ( \mathbf { Z } ) ) = f ( \pmb { \sigma } ( \mathbf { U } ^ { T } \mathbf { Z } \mathbf { V } ) ) =$ $F ( \mathbf { U } ^ { T } \mathbf { Z } \mathbf { V } )$ . Making the change of variables $\mathbf { W } = \mathbf { U } ^ { T } \mathbf { Z } \mathbf { V }$ , we conclude that the unique optimal solution of (7.8) is given by 

$$
\mathbf {Z} = \mathbf {U W} ^ {*} \mathbf {V} ^ {T},\tag{7.9}
$$

where $\mathbf { W } ^ { * }$ is the unique optimal solution of 

$$
\min _ {\mathbf {W} \in \mathbb {R} ^ {m \times n}} \left\{G (\mathbf {W}) \equiv F (\mathbf {W}) + \frac {1}{2} \| \mathbf {W} - \mathbf {D} \| _ {F} ^ {2} \right\}.\tag{7.10}
$$

We will prove that $\mathbf { W } ^ { * }$ is a generalized diagonal matrix (meaning that all ofdiagonal components are zeros). Let $i \in \{ 1 , 2 , . . . , r \}$ . Take $\pmb { \Sigma } _ { i } ^ { ( 1 ) } \in \mathbb { R } ^ { m \times m }$ and $\pmb { \Sigma } _ { i } ^ { ( 2 ) } \in \mathbb { R } ^ { n \times n }$ to be the $m \times m$ and $n \times n$ diagonal matrices whose diagonal elements are all ones except for the $( i , i ) \mathrm { t h }$ component, which is −1. Define $\widetilde { \mathbf { W } } _ { i } =$ $\Sigma _ { i } ^ { ( 1 ) } \mathbf { W } ^ { * } \Sigma _ { i } ^ { ( 2 ) }$ . Obviously, by the fact that $\Sigma _ { i } ^ { ( 1 ) } \in \mathbb { O } ^ { m } , \Sigma _ { i } ^ { ( 2 ) } \in \mathbb { O } ^ { n }$ 2 

$$
F (\pmb {\Sigma} _ {i} ^ {(1)} \mathbf {W} ^ {*} \pmb {\Sigma} _ {i} ^ {(2)}) = f (\pmb {\sigma} (\pmb {\Sigma} _ {i} ^ {(1)} \mathbf {W} ^ {*} \pmb {\Sigma} _ {i} ^ {(2)})) = f (\pmb {\sigma} (\mathbf {W} ^ {*})) = F (\mathbf {W} ^ {*}),
$$

and we thus obtain 

$$
\begin{array}{l} G (\widetilde {\mathbf {W}} _ {i}) = F (\widetilde {\mathbf {W}} _ {i}) + \frac {1}{2} \| \widetilde {\mathbf {W}} _ {i} - \mathbf {D} \| _ {F} ^ {2} \\ \qquad = F (\boldsymbol {\Sigma} _ {i} ^ {(1)} \mathbf {W} ^ {*} \boldsymbol {\Sigma} _ {i} ^ {(2)}) + \frac {1}{2} \| \boldsymbol {\Sigma} _ {i} ^ {(1)} \mathbf {W} ^ {*} \boldsymbol {\Sigma} _ {i} ^ {(2)} - \mathbf {D} \| _ {F} ^ {2} \\ \qquad = F (\mathbf {W} ^ {*}) + \frac {1}{2} \| \mathbf {W} ^ {*} - \boldsymbol {\Sigma} _ {i} ^ {(1)} \mathbf {D} \boldsymbol {\Sigma} _ {i} ^ {(2)} \| _ {F} ^ {2} \\ \qquad = F (\mathbf {W} ^ {*}) + \frac {1}{2} \| \mathbf {W} ^ {*} - \mathbf {D} \| _ {F} ^ {2}, \\ \qquad = G (\mathbf {W} ^ {*}). \end{array}
$$

Consequently, $\widetilde { \mathbf { W } } _ { i }$ is also an optimal solution of (7.10), but by the uniqueness ;of the optimal solution of problem (7.10), we conclude that $\mathbf { W } ^ { * } = \Sigma _ { i } ^ { ( 1 ) } \mathbf { W } ^ { * } \Sigma _ { i } ^ { ( 2 ) }$ Comparing the ith rows and columns of the two matrices we obtain that $W _ { i j } ^ { * } = 0$ and $W _ { j i } ^ { * } = 0$ for any $j \neq i$ . Since this argument is valid for any $i \in \{ 1 , 2 , . . . , r \}$ it follows that $\mathbf { W } ^ { * }$ is a generalized diagonal matrix, and consequently the optimal solution of (7.10) is given by $\mathbf { W } ^ { * } = \mathrm { d g } ( \mathbf { w } ^ { * } )$ , where $\mathbf { w } ^ { * }$ is the optimal solution of 

$$
\min _ {\mathbf {w}} \left\{F (\mathrm{dg} (\mathbf {w})) + \frac {1}{2} \| \mathrm{dg} (\mathbf {w}) - \mathbf {D} \| _ {F} ^ {2} \right\}.
$$

Since $F ( \mathrm { d g } ( \mathbf { w } ) ) = f ( | \mathbf { w } | ^ { 4 } ) = f ( \mathbf { w } ) \mathrm { ~ a n d ~ } \| \mathrm { d g } ( \mathbf { w } ) - \mathbf { D } \| _ { F } ^ { 2 } = \| \mathbf { w } - \pmb { \sigma } ( \mathbf { X } ) \| _ { 2 } ^ { 2 }$ , it follows that $\mathbf { w } ^ { * }$ is given by 

$$
\mathbf {w} ^ {*} = \operatorname{argmin} _ {\mathbf {w}} \left\{f (\mathbf {w}) + \frac {1}{2} \| \mathbf {w} - \boldsymbol {\sigma} (\mathbf {X}) \| _ {2} ^ {2} \right\} = \operatorname{prox} _ {f} (\boldsymbol {\sigma} (\mathbf {X})).
$$

Therefore, $\mathbf { W } ^ { * } = \mathrm { d } \mathbf { g } ( \mathrm { p r o x } _ { f } ( { \pmb \sigma } ( \mathbf { X } ) ) )$ , which, along with (7.9), establishes the desired result. 

Example 7.30. Using the spectral prox formula over $\mathbb { R } ^ { m \times n }$ , we can compute the prox of symmetric spectral functions in terms of the prox of their associated functions. Using this observation, we present in the table below expressions of prox operators of several functions. The parameter $\alpha$ is always assumed to be positive, and $\mathbf { U } \in \mathbb { O } ^ { m } , \mathbf { V } \in \mathbb { O } ^ { n }$ are assumed to satisfy ${ \bf X } = { \bf U } \mathrm { d } { \bf g } ( { \pmb \sigma } ( { \bf X } ) ) { \bf V } ^ { T }$ . The table also includes a reference to the corresponding results for the associated functions, which are always defined over $\mathbb { R } ^ { r }$ 