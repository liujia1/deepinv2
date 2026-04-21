# UNIT5_assignment.pdf

## 第1页

Regularisation Theory
Topic: Error Estimates for Variational Regularisation
Date: 13 June 2025
Assessment: Error Estimates for Variational Regularisation with
Fenchel-Young Fidelities
Consider the variational regularisation problem of finding an approximate solution uδ
α to the
inverse problem Ku = f from noisy data fδ by solving the following minimisation problem
uδ
α ∈arg min
u∈U
n
Hfδ
Φ (Ku) + αJ(u)
o
,
where U and V are Hilbert spaces, K : U →V is a linear and bounded operator, J : U →R ∪
{+∞} is a proper, convex, and lower semi-continuous functional, and α > 0 is the regularisation
parameter.
The data fidelity term Hfδ
Φ : V →R ∪{+∞} is given by the Fenchel-Young fidelity,
Hfδ
Φ (v) = Φ(v) + Φ∗(fδ) −⟨v, fδ⟩
where Φ : V →R ∪{+∞} is a proper, convex, lower semi-continuous functional, and Φ∗is its
convex conjugate.
Your task is to derive an error estimate for the Bregman distance DK∗v†
J
(uδ
α, u†)
between the regularised solution uδ
α and a desired ”true” solution u†.
Assumptions
1. Let u† ∈U be a solution corresponding to exact data f ∈V, i.e., Ku† = f.
2. Assume a source condition holds for u†: there exists an element v† ∈V such that
K∗v† ∈∂J(u†)
where ∂J(u†) denotes the subdifferential of J at u†.
3. Assume the noise in the data is bounded in the sense of the data fidelity term, i.e.
Hfδ
Φ (f) ≤δ2
for some noise level δ > 0.
Guidance
1. Start from the optimality of uδ
α for the minimisation problem.
2. Introduce the Bregman distance with respect to the functional J to obtain an initial
inequality.
3. Add and subtract Hfδ
Φ to (and from) your right-hand-side.
4. Apply the source condition and the noise model assumption (Hfδ
Φ (f) ≤δ2).
1


---

## 第2页

5. Expand the remaining fidelity terms (Hfδ
Φ ) using their definition and collect terms into
logical groups.
6. Apply the Fenchel-Young inequality, ⟨a, b⟩≤Φ(a) + Φ∗(b), to simplify the expression.
7. Based on your final estimate, briefly discuss the conditions on the parameter choice rule
α(δ) that would ensure convergence.
2


---

