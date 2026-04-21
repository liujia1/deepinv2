# Seminar_Tachella_09_june_2025.pdf

## 第1页

Self-Supervised Learning for 
Imaging Inverse Problems
Julián Tachella, CNRS, École Normale Supérieure de Lyon


---

## 第2页

2
Linear Inverse Problems
Goal: recover signal 𝒙from 𝒚
𝒚= 𝐴𝒙+ 𝝐


---

## 第3页

3
Examples
Magnetic resonance imaging
•
𝑨= subset of Fourier modes 
(𝑘−space)
Computed tomography
•
𝑨= 1D projections  
(sinograms)
𝒙
𝒚
𝒚
𝒙
Image inpainting
•
𝑨= diagonal matrix 
with 1’s and 0s.
𝒚
𝒙
Image denoising
•
𝑨= identity
𝒚
𝒙


---

## 第4页

4
Regularised reconstruction
Idea: define a regularisation 𝜌𝒙promoting plausible reconstructions
ෝ𝒙= argmin
𝒚−𝐴𝒙
2 + 𝜌(𝒙)
Examples: total-variation, sparsity, etc.
Disadvantages: hard to define a good 𝜌𝒙in real world problems, 
loose with respect to the true signal distribution
𝒙


---

## 第5页

5
Learning approach
Idea: use training pairs of signals and measurements to directly learn 
the inversion function
𝑓
input
target
input
target
…
supervised
dataset
argmin
𝑓
𝔼𝒙,𝒚𝒙−𝑓𝒚
2


---

## 第6页

6
Advantages: 
•
State-of-the-art reconstructions
•
Once trained, 𝑓is easy to evaluate
x8 accelerated MRI [Zbontar et al., 2019]
Deep network 
(34.5 dB)
Ground-truth
Total variation
(28.2 dB)
Learning approach


---

## 第7页

7
Learning approach
Main disadvantage: Obtaining training signals 𝒙𝑖can be expensive or impossible. 
•
Medical and scientific imaging
•
Distribution shift [Belthangady & Royer, 2019]


---

## 第8页

8
AI for Knowledge Discovery?


---

## 第9页

9
Outline
How can we learn 𝑓from measurement {𝒚𝑖} data alone?
1.
Noisy:  𝒚= 𝒙+ 𝝐
2.
Incomplete and noisy:  𝒚= 𝐴𝒙+ 𝝐
Goal: build a self-supervised loss ℒSELF such that
𝔼𝒚ℒSELF 𝒚, 𝑓= 𝔼𝒙,𝒚ℒSUP 𝒙, 𝒚, 𝑓+ const.


---

## 第10页

10
How can we learn 𝑓from measurement {𝒚𝑖} data alone?
Example
• Cryo-Electron microscopy images
• Extremely low SNR
• Noise distribution is unknown


---

## 第11页

11
Self-Supervised Risk Estimators
Goal: build self-supervised loss ℒSELF:
•
Unbiased estimator: 𝔼𝒚ℒSELF 𝒚, 𝑓∝𝔼𝒙,𝒚||𝒙−𝑓𝒚||2
•
Same global minimum: argmin
𝑓
𝔼𝒚ℒSELF 𝒚, 𝑓= argmin
𝑓
𝔼𝒙,𝒚||𝒙−𝑓𝒚||2
•
Same minima constrained set: argmin
𝑓∈ℱ
𝔼𝒚ℒSELF 𝒚, 𝑓= argmin
𝑓∈ℱ𝔼𝒙,𝒚||𝒙−𝑓𝒚||2
What is the best we can do? MMSE estimator
= 𝔼{𝒙|𝒚}
= 𝔼{𝒙|𝒚}
≠𝔼{𝒙|𝒚}


---

## 第12页

12
Stein’s Unbiased Risk Estimator
ℒSURE 𝒚, 𝑓
= || 𝒚−𝑓𝒚||2 + 2𝜎2 ෍
𝑖
𝛿𝑓𝑖
𝛿𝑦𝑖
𝒚
•
Stein’s lemma: Let 𝒚|𝒙∼𝒩𝒙, 𝑰𝜎2 , then 𝔼𝒚ℒSURE 𝒚, 𝑓∝𝔼𝒙,𝒚||𝒙−𝑓𝒚||2
•
Extensions for Poisson, Poisson-Gaussian [Hudson, 1978], 
•
MMSE estimator 𝑓∗𝒚= 𝔼{𝒙|𝒚}
Measurement
consistency 
Degrees of freedom [Efron, 2004] 


---

## 第13页

13
1) Monte Carlo SURE: approx. divergence as [Ramani et al., 2007]
෍
𝑖
𝛿𝑓𝑖
𝛿𝑦𝑖
𝒚
≈𝝎
𝛼
⊤
𝑓𝒚−𝑓𝒚+ 𝝎𝛼
2) Autodiff SURE: use auto-diff [Soltanayev, 2020]
෍
𝑖
𝛿𝑓𝑖
𝛿𝑦𝑖
𝒚
≈𝝎⊤𝛿𝑓
𝛿𝒚𝝎
3) Recorrupted2Recorrupted [Pang et al., 2021], [Monroy, Bacca & T., CVPR 2025]
Efficient SURE
ℒR2R 𝒚, 𝑓, 𝛼= 𝔼𝝎||𝒚+ 𝛼𝝎−𝑓𝒚−𝝎/𝛼||2
Extensions to Poisson, Gamma and Binomial
𝝎∼𝒩𝟎, 𝑰𝜎2
and 𝛼∈ℝ


---

## 第14页

14
Tweedie’s Formula
The solution to SURE is Tweedie’s Formula
•
Noise2Score [Kim and Ye, 2021] learns ∇log 𝑝𝒚𝒚from noisy data + denoises with Tweedie.
•
Key formula behind diffusion models, which can be trained self-supervised [Daras et al., 2024]
min
𝑓
𝔼𝒚|| 𝑓𝒚−𝒚−𝜎2∇log 𝑝𝒚𝒚||2
min
𝑓
𝔼𝒚|| 𝒚−𝑓𝒚||2 + 2𝜎2 ෍
𝑖
𝛿𝑓𝑖
𝛿𝑦𝑖
𝒚
min
𝑓
𝔼𝒚|| 𝒚−𝑓𝒚||2 −2𝜎2 ෍
𝑖
𝑓𝑖𝒚
𝛿log 𝑝𝒚(𝒚)
𝛿𝑦𝑖
Complete squares
Integration by parts
𝑓(𝒚) = 𝒚+ 𝜎2∇log 𝑝𝒚𝒚


---

## 第15页

15
Cross-Validation Methods
Noise2Void [Krull et al., 2019], Noise2Self [Batson, 2019], Neighbor2Neighbor [Huang, 2023]
•
During training flip centre pixel
•
Computes loss only on flipped pixels
Blind spot networks [Laine et al., 2019]
•
Convolutional architecture that doesn’t ‘see’ centre 
•
pixel by construction 
What happens if we only know that 𝑝𝒚𝒙= ς 𝑝𝑦𝑖𝑥𝑖?
Idea: if 𝒇𝒊doesn’t depend on 𝒚𝒊we cannot overfit the noise!


---

## 第16页

16
Cross-Validation Methods
•
SURE’s perspective:
•
These methods are not MMSE optimal!
min
𝑓
|| 𝒚−𝑓𝒚||2 + 2𝜎2 ෍
𝑖
𝛿𝑓𝑖
𝛿𝑦𝑖
𝒚
min
𝑓
|| 𝒚−𝑓𝒚||2 subject to  
𝛿𝑓𝑖
𝛿𝑦𝑖(𝒚) = 0 ∀𝑖, ∀𝒚
•
We can write these losses as


---

## 第17页

17
Are We Missing Something?
         
      
un nown
un nown
un nown
 ess e pressive
esti ators
                                         
 nown
un nown 
 
 
???


---

## 第18页

18
•
SURE’s perspective:
ℒUNSURE 𝒚, 𝑓= || 𝒚−𝑓𝒚||2 + 2𝜎2 ෍
𝑖
𝛿𝑓𝑖
𝛿𝑦𝑖
𝒚
ℒUNSURE 𝒚, 𝑓= || 𝒚−𝑓𝒚||2 subject to 𝔼𝒚σ𝑖
𝛿𝑓𝑖
𝛿𝑦𝑖(𝒚) = 0
•
Impose zero-expected divergence [Tachella et al., ICLR25]
min
𝑓
max
𝜼
𝔼𝒚|| 𝒚−𝑓𝒚||2 + 2𝜂෍
𝑖
𝛿𝑓𝑖
𝛿𝑦𝑖
(𝒚)
•
In practice, we use Lagrange multipliers
UNSURE


---

## 第19页

19
𝑓ZED 𝒚= 𝒚+ Ƹ𝜂∇log 𝑝𝒚𝒚
Ƹ𝜂=
1
𝑛𝔼𝒚||∇log 𝑝𝒚𝒚||2 −1
UNSURE
min
𝑓
max
𝜼
𝔼𝒚|| 𝒚−𝑓𝒚||2 + 2𝜂෍
𝑖
𝛿𝑓𝑖
𝛿𝑦𝑖
(𝒚)
•
Closed-form solution (similar to Tweedie’s form
ula)
1
𝑛𝔼𝒙,𝒚|| 𝑓ZED 𝒚−𝒙|| = 𝜎2
1
1 −MMSE
𝜎2
−1
≈MMSE + MMSE2
𝜎2
•
Expected error
•
UNSURE can be extended to unknown noise covariance and Poisson Gaussian noise


---

## 第20页

20
Denoising Experiments
•
MNIST dataset (28x28 grayscale)
•
Isotropic Gaussian noise


---

## 第21页

21
Real data experiments
•
Cryo electron microscopy images
•
Extremely low SNR
•
Approx. Poisson-Gaussian noise
Poisson-Gaussian UNSURE


---

## 第22页

22
Outline
How can we learn 𝑓from measurement {𝒚𝑖} data alone?
1.
Noisy:  𝒚= 𝒙+ 𝝐
2.
Incomplete and noisy:  𝒚= 𝑨𝒙+ 𝝐


---

## 第23页

23
For 𝐴≠𝐼, most estimators can be adapted to approximate
𝔼𝒙,𝒚||𝐴(𝒙−𝑓𝒚)||2
In this case, the risk does not penalise 𝑓(𝒚) in the nullspace of 𝐴!
Incomplete Measurements?
𝐴
𝒇


---

## 第24页

Learning from Measurements
How to learn from only 𝒚?
•
Access multiple operators 𝒚𝑖= 𝐴𝑔𝑖𝒙𝑖with 𝑔∈{1, … , 𝐺}
•
Each 𝐴𝑔with different nullspace
•
Offers the possibility for learning using multiple measurement operators
𝐴1𝑥1
𝐴3𝑥3
𝐴2𝑥2
𝐴1𝑥1
𝐴2𝑥2
𝐴3𝑥3


---

## 第25页

Necessary Condition
Proposition: Learning reconstruction mapping 𝑓from observed 
measurements possible only if 
rank 𝔼𝑔𝐴𝑔⊤𝐴𝑔= 𝑛
and thus, if 𝑚≥𝑛/𝐺.
Intuition: we need that the operators 𝐴1, 𝐴2, … … 𝐴𝐺cover the whole ambient space
[Tachella et al., 2023, JMLR].
𝐴1
𝐴2
𝐴3
𝐴4


---

## 第26页

Multi Operator Imaging 
Multi Operator Imaging (MOI) [Tachella et al., 2022, NeurIPS] 
ℒMOI 𝒚, 𝑓=
𝒚−𝐴𝑔𝑓𝒚, 𝐴𝑔
2
+ ෍
𝑠
𝑓𝐴𝑠ෝ𝒙, 𝐴𝑠−ෝ𝒙
2
with ෝ𝒙= 𝑓𝒚, 𝐴𝑔
Can be replaced by SURE,
UNSURE, etc. 
Enforces 𝑓𝐴𝑔𝒙, 𝐴𝑔≈𝑓𝐴𝑠𝒙, 𝐴𝑠


---

## 第27页

Inpainting Experiments
• U-Net network
• CelebA dataset
• 𝐺= 40 different 𝐴𝑔inpainting masks
Signal 𝑥
MOI
Supervised
Measurements 𝑦


---

## 第28页

28
Symmetry Prior
Equivariant Imaging [Chen, Tachella and Davies, ICCV 2021]
For all 𝑔∈𝐺we have
𝒚= 𝐴𝒙= 𝐴𝑇𝑔𝑇𝑔−1𝒙
•
We get multiple virtual operators 𝐴𝑔𝑔∈𝐺‘for free’!
•
Each 𝐴𝑇𝑔might have a different nullspace
= 𝐴𝑔𝒙′


---

## 第29页

29
(Non)-Equivariant Operators
Translation
Rotation
Scaling
Amplitude
Gaussian 
Blur
Image 
Inpainting
Sparse-view
CT
Accelerated
MRI
Downsampling
(with antialias)
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
?
Theorem [T. et al., 2023]: The full rank condition requires that 𝐴is not equivariant: 𝐴𝐴𝑇𝑔≠𝑇𝑔𝐴𝐴
rank 𝔼𝑔𝑇𝑔⊤𝐴⊤𝐴𝑇𝑔= rank 𝐴⊤(𝔼𝑔෨𝑇𝑔⊤෨𝑇𝑔)𝐴= rank 𝐴⊤𝐴= 𝑚< 𝑛


---

## 第30页

30
Equivariant Imaging
How can we enforce equivariance in practice?
Idea: we should have 𝑓𝐴𝑇𝑔𝒙= 𝑇𝑔𝑓(𝐴𝒙), i.e. 𝑓∘𝐴should be 𝐺-equivariant
𝐴
𝑓


---

## 第31页

31
Equivariant Imaging
ℒ𝐸𝐼𝒚, 𝑓= 𝔼𝑔|| 𝑇𝑔ෝ𝒙−𝑓𝐴𝑇𝑔ෝ𝒙||2
We can leverage invariance of 𝑝(𝒙) to transformations 𝑇𝑔to learn in the 
nullspace
where ෝ𝒙= 𝑓(𝒚) is used as reference
Robust Equivariant Imaging++ [Chen, Tachella & Davies 2022, CVPR]
ℒREI 𝒚, 𝑓= ℒUNSURE 𝒚, 𝑓+ ℒEI 𝒚, 𝑓
Handles noisy measurements of unknown noise level
enforces equivariance of 𝑓∘𝐴


---

## 第32页

32
Experiments
• FastMRI dataset
• Gaussian noise
            
           
          
        
          
  
             


---

## 第33页

33
Experiments
• Operator 𝐴is isotropic blur with Gaussian noise
• Dataset is approximately scale invariant
Scanvic, Davies, Abry, T., arxiv 2024


---

## 第34页

34
Finetuning
•
Reconstruct Anything Model: general model solving many inverse problems [Terris et al., 2025]
 otion  lur
 easure ent
   
Sin lecoil  R 
 aussian
to o raph 
 oisson
to o raph 
 ulticoil  R 
 ero s ot per orman e
                                      
                               


---

## 第35页

35
Finetuning
•
The model can be finetuned with self-supervised losses on up to a single 𝒚𝑁= 1
•
Finetuning can be done in a few seconds 


---

## 第36页

36
References
Slides and codes of a recent 3-hour tutorial can be found here:
https://tachella.github.io/blog/selfsuptutorial/


---

## 第37页

37


---

## 第38页

38


---

## 第39页

39
1-bit Compressed Sensing
•
𝑦= sign(𝐴𝑥) with Gaussian 𝐴and 20% undersampling ratio 
•
Dataset is approximately translation invariant
      
       
        
          
      
     
[Tachella & Jacques, TMLR 2023]


---

## 第40页

40
•
DIV2K dataset (320x320 RGB)
•
Spatially correlated Gaussian noise 3x3 pixels
 
       
          
          
            
           
Correlated Noise Experiments


---

## 第41页

41
Efficient SURE
Recorrupted2Recorrupted [Pang et al., 2021], Noisier2Noise [Moran et al., 2020], etc.
ℒR2R 𝒚, 𝑓, 𝛼= 𝔼𝝎|| 𝒚𝑏−𝑓𝒚𝑎||2
Proposition: Let 𝒚∼𝒩𝒙, 𝑰𝜎2 and define 
𝒚𝑎= 𝒚+ 𝛼𝝎
𝒚𝑏= 𝒚−𝝎/𝛼
where 𝝎∼𝒩𝟎, 𝑰𝜎2 and 𝛼∈ℝ, then 𝒚𝑎and 𝒚𝑏are independent random variables (fixed 𝒙).
Generalized Recorrupted2Recorrupted [Monroy, Bacca & Tachella, CVPR 2025]
•
Equivalent to SURE in the limit: lim
𝛼→0 ℒR2R 𝒚, 𝑓, 𝛼= ℒSURE 𝒚, 𝑓
•
Extensions to Poisson, Gamma and Binomial


---

## 第42页

42
Uncertainty Quantification
Error estimates 𝑒𝑗
True error
Self-supervised losses can also be used for uncertainty quantification!
•
SURE can be used to assess reconstruction error in denoising 
•
SURE4SURE [Bellec et al., 2021] gives error variance estimates.
•
EI loss can be seen as a bootstrapping technique 
[T. & Pereyra, 2024] with well calibrated uncertainty estimates 
Can we measure the uncertainty of the reconstructions?


---

## 第43页

43
Implementation 
We parameterize 𝑓as a deep neural network, small 𝛼> 0, 𝝎∼𝒩𝟎, 𝐼
1) UNSURE: Solve Lagrangian problem, approximating divergence as [Ramani et al., 2007]
tr(Σ 𝛿𝑓
𝛿𝒚) ≈(Σ𝝎)
𝛼
⊤
𝑓𝒚−𝑓𝒚+ 𝝎𝛼
2) UNSURE via score: Learn score 𝑠𝒚≈∇log 𝑝𝒚(𝒚) via
arg min
𝑠
𝔼𝒚,𝝎||𝝎−𝛼𝑠𝒚+ 𝝎𝛼||2
and use 𝑓𝒚= 𝒚+ Σෝ𝜼𝑠𝒚to denoise at test time.


---

