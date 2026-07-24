# ERASMUS+ INTERNATIONAL PHD SUMMER SCHOOL 2025 Mathematics and Machine Learning for image analysis Tom Pock’s final exam

June 15, 2025 

## Maximum likelihood learning for the total variation prior

The goal of this exam is to perform maximum likelihood estimation of the parameter $\lambda > 0$ in a total variation prior defined as 

$$
p _ {\lambda} (x) = \frac {\exp (- \lambda T V (x))}{\int \exp (- \lambda T V (x)) \mathrm{d} x}
$$

on natural images. The assignment consists of the following tasks: 

1. Implement an image sampling algorithm for the TV prior with arbitrary λ. Modify the notebook from Lab 2 by removing the denoising data term to ensure the Gaussian latent machine samples solely from the TV prior. 

2. Generate N = 10 samples of size $1 0 0 \times 1 0 0$ using λ = 1. Although the samples may appear noisy, this is expected. Visualize at least one sample. 

3. Compute the finite diferences $u _ { i + 1 , j } - u _ { i , j }$ and $u _ { i , j + 1 } - u _ { i , j }$ on your samples (your code already does this at some point). Plot the negative log histogram of these diferences. Does it resemble an absolute value function? Explain any deviations. 

4. Use some code snippets from the bilevel learning example from Lab 1 to load and extract $N = 1 0$ natural image patches of size $1 0 0 \times 1 0 0$ from the BSDS500 data base. 

5. Repeat the finite diference and histogram visualization for the natural patches. How does the plot compare? What insights does it ofer? 

6. Estimate the optimal λ of the TV prior that best fits natural images using maximum likelihood, as outlined on Slide 50 of Lecture 1. Replace $C N N _ { \theta } ( x )$ with λT V (x) (i.e. θ ≡ λ), set the gradient of the log-likelihood to zero, and solve for λ. Since $\lambda T V ( x ) = T V ( \lambda x )$ , you can rescale samples generated with $\lambda = 1$ to simulate other values. This leads to a simple closed form solution for the ML estimate of λ :-) 

7. Compute and report the optimal λ. Plot the negative log histogram of the natural image finite diferences alongside those from the TV prior samples using the optimal value of λ. 

8. Enjoy! 