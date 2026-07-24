Assessment: Error Estimates for Variational Regularisation with Fenchel-Young Fidelities 

Consider the variational regularisation problem of finding an approximate solution $u _ { \alpha } ^ { \delta }$ to the inverse problem $K u = f$ from noisy data $f ^ { \delta }$ by solving the following minimisation problem 

$$
u _ {\alpha} ^ {\delta} \in \underset {u \in \mathcal {U}} {\arg \min} \left\{H _ {\Phi} ^ {f ^ {\delta}} (K u) + \alpha J (u) \right\},
$$

where $\mathcal { U }$ and $\nu$ are Hilbert spaces, $K : \mathcal { U } \to \mathcal { V }$ is a linear and bounded operator, $J : \mathcal { U } \to \mathbb { R } \cup$ $\{ + \infty \}$ is a proper, convex, and lower semi-continuous functional, and $\alpha > 0$ is the regularisation parameter. 

The data fidelity term $H _ { \Phi } ^ { f ^ { \delta } } : \mathcal { V } \to \mathbb { R } \cup \{ + \infty \}$ is given by the Fenchel-Young fidelity, 

$$
H _ {\Phi} ^ {f ^ {\delta}} (v) = \Phi (v) + \Phi^ {*} (f ^ {\delta}) - \langle v, f ^ {\delta} \rangle
$$

where $\Phi : \mathcal { V } \to \mathbb { R } \cup \{ + \infty \}$ is a proper, convex, lower semi-continuous functional, and $\Phi ^ { * }$ is its convex conjugate. 

Your task is to derive an error estimate for the Bregman distance $D _ { J } ^ { K ^ { * } v ^ { \dagger } } ( u _ { \alpha } ^ { \delta } , u ^ { \dagger } )$ between the regularised solution $u _ { \alpha } ^ { \delta }$ and a desired ”true” solution $u ^ { \dagger }$ 

## Assumptions

1. Let $u ^ { \dag } \in \mathcal { U }$ be a solution corresponding to exact data $f \in \mathcal V$ , i.e., $K u ^ { \dagger } = f$ 

2. Assume a source condition holds for $u ^ { \dagger } \colon$ : there exists an element $v ^ { \dagger } \in \mathcal V$ such that 

$$
K ^ {*} v ^ {\dagger} \in \partial J (u ^ {\dagger})
$$

where $\partial J ( u ^ { \dagger } )$ denotes the subdiferential of J at $u ^ { \dagger }$ 

3. Assume the noise in the data is bounded in the sense of the data fidelity term, i.e. 

$$
H _ {\Phi} ^ {f ^ {\delta}} (f) \leq \delta^ {2}
$$

for some noise level $\delta > 0$ 

## Guidance

1. Start from the optimality of $u _ { \alpha } ^ { \delta }$ for the minimisation problem. 

2. Introduce the Bregman distance with respect to the functional J to obtain an initial inequality. 

3. Add and subtract $H _ { \Phi } ^ { f ^ { \delta } }$ to (and from) your right-hand-side. 

4. Apply the source condition and the noise model assumption $( H _ { \Phi } ^ { f ^ { \delta } } ( f ) \leq \delta ^ { 2 } )$ 

5. Expand the remaining fidelity terms $( H _ { \Phi } ^ { f ^ { \delta } } )$ using their definition and collect terms into logical groups. 

6. Apply the Fenchel-Young inequality, $\langle a , b \rangle \leq \Phi ( a ) + \Phi ^ { * } ( b )$ , to simplify the expression. 

7. Based on your final estimate, briefly discuss the conditions on the parameter choice rule α(δ) that would ensure convergence. 