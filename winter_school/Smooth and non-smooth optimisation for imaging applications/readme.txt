Lecture 1: Basics of convex analysis and basic algorithms 
In this lecture we revise the basic notions on smoothness (Lipschitz continuity, Gateaux/Frechet differentiability..) and convex analysis (subdifferentials and subgradients, Fenchel conjugation, strong convexity..) that will be required for defining two basic algorithms solving convex optimisation problems: gradient descent (GD) and proximal gradient descent (GD).

Lecture 2: Acceleration strategies
In this lecture we will describe acceleration strategies for improving convergence performance of GD (Nesterov acceleration), PGD (FISTA) and show how strong convexity can be explicitly dealt with to define even faster acceleration schemes. 

Lecture 3: Sparsity-based problems: from convex to non-convex approaches
In this lecture we will give a focus on how sparsity appears in applications. We will comment on the use of l_1 VS l_0 optimisation-based approaches. For the latter ones, we will review Iterative hard thresholding and greedy algorithms, and introduce the notion of continuous relaxations, both for  constrained and unconstrained l_0 problems. Applications to microscopy image analysis will be shown.

Lab: ISTA and FISTA for sparse image reconstruction 
In this lab we will consider convex optimisation algorithms for solving problems with an l_2 data term and a sparsity promoting term for problems of image reconstruction such as molecule localisation (for undersampled, blurred and noisy data, with l_1 penalty). We will compare ISTA with its accelerated version (FISTA) and, for a slightly modified model, with its strongly convex variant.

Exercise class: Proof of convergence of FISTA in convex/strongly convex setting
Proof of convergence of FISTA in function values + strongly convex variant (with explicit knowledge of strong convexity parameter).