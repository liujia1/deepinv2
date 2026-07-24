# Pre-course: imaging and learning, part I (images, noise, Bayesian/variational formulation, optimisation)

Luca Calatroni MaLGa Center, Department of Computer Science, University of Genoa, Italy 

# Mathematics & Machine Learning for image analysis ERASMUS+ International PhD Summer School Bologna, June 3-11 2025

Co-funded by the European Union (ERC, MALIN, 10117133). Views and opinions expressed are however those of the author only and do not necessarily refl ect those of the European Union or the European Research Council Executive Agency. Neither the EU nor the granting authority can be held responsible for them. 

## Meet the instructors (Luca’s) & program

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/fb94b548c815c0145fb19d4832b820a1de504a4f6e7782c2ca90d4cf96a2166b.jpg)



Luca Calatroni Part I (14:30-15:45)


- Digital images 

- Modelling noise 

- Quality measures (MSE, PSNR, SSIM) 

- Image denoising as a toy ‘inverse’ problem 

- Bayesian formulation 

- MAP estimators 

- Regularisation 

- Bits of optimisation: GD, proximal operator 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/9bcc05b8a9b878456aa75d3548fff55e8832b13dc15c2c6dec92327ab99f9513.jpg)



Luca Ratti Part II (16:00-17:15)


- Image denoising as a regression problem 

- Supervised/unsupervised learning 

- Neural networks 

- Bias-variance tradeoff 

- Over-parametrisation 

- NNs for imaging: CNN, U-NET 

- Training a NN: backpropagation, batches, optimisers.. 

- NumPy/Matplotlib/Pytorch (tomorrow) 

## Digital images

Digital images are discrete representations of the continuous world we live in. 

Analog Image 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/83f1c7ec293c761ad5cee72a58a4ae869532b3a215d6e17cfc7dc347bf7bab9f.jpg)



Digital Sampling


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/c331a79ddb4e34f4bd6a0ceab036b80e36f67e94d76fa20fba7c43a27c4afe99.jpg)



Pixel Quantization


<table><tr><td>249</td><td>244</td><td>240</td><td>230</td><td>209</td><td>233</td><td>227</td><td>251</td><td>255</td></tr><tr><td>248</td><td>245</td><td>210</td><td>93</td><td>81</td><td>120</td><td>97</td><td>193</td><td>254</td></tr><tr><td>250</td><td>170</td><td>133</td><td>94</td><td>137</td><td>120</td><td>104</td><td>145</td><td>253</td></tr><tr><td>241</td><td>116</td><td>118</td><td>107</td><td>134</td><td>138</td><td>96</td><td>92</td><td>163</td></tr><tr><td>277</td><td>142</td><td>121</td><td>113</td><td>124</td><td>115</td><td>107</td><td>71</td><td>179</td></tr><tr><td>234</td><td>106</td><td>84</td><td>125</td><td>97</td><td>108</td><td>125</td><td>106</td><td>204</td></tr><tr><td>241</td><td>202</td><td>102</td><td>132</td><td>75</td><td>73</td><td>141</td><td>246</td><td>252</td></tr><tr><td>253</td><td>252</td><td>244</td><td>239</td><td>178</td><td>199</td><td>242</td><td>250</td><td>245</td></tr><tr><td>255</td><td>249</td><td>244</td><td>250</td><td>226</td><td>231</td><td>240</td><td>251</td><td>253</td></tr></table>

Sampling: allows to represent a continuous image into a finite (pixel) grid. 

Quantisation: assigns a grey-level describing average brightness at each pixel. 

## Digital images

Digital images are discrete representations of the continuous world we live in. 

Analog Image 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/62f6874f17358e33281b6b6306145c1a3aa2fb039ddfe029e2d85b405ac6cb42.jpg)



Digital Sampling


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/64b8dee3ff800196731feaa16033ae3a9307c6b593847660ae1d7273b02e5b33.jpg)



Pixel Quantization


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/fe575eb2467cd78eeb43c4377611c7f870c7e7d624a63948f4853cc6019798a4.jpg)


$$
n _ {1}
$$

Sampling: allows to represent a continuous image into a finite (pixel) grid. 

$$
n _ {2}
$$

Quantisation: assigns a grey-level describing average brightness at each pixel. 

$$
\Omega = \left\{1, \dots , n _ {1} \right\} \times \left\{1, \dots , n _ {2} \right\}
$$

image domain 

## Digital images

Digital images are discrete representations of the continuous world we live in. 

Analog Image 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/fd1485c03d5887db0c845aef5ee52b2a542a6eca8e93fc42382faed117d168b2.jpg)



Digital Sampling


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/6536bd53cf5b5ec2b8d9c279e336bdb2be7944b7ad9df8592882841f9bc44e44.jpg)



Pixel Quantization


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/5320d428e62ddde09b98925231570578e16f2762485e8beff551410104d71708.jpg)


Sampling: allows to represent a continuous image into a finite (pixel) grid. 

$$
n _ {2}
$$

Quantisation: assigns a grey-level describing average brightness at each pixel. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/86384af6fe19688e3de113b1bd6c368f34c68ed2419219e7524b341b3f8ddd4d.jpg)


$$
\Omega = \left\{1, \dots , n _ {1} \right\} \times \left\{1, \dots , n _ {2} \right\}
$$

$$
\mathbf {X}: \Omega \to \{0, \dots , 2 5 5 \}
$$

image domain 

$$
\mathbf {X} = (x _ {i, j}) \in \left\{0, \dots , 2 5 5 \right\} ^ {n _ {1} \times n _ {2}}
$$

## Digital images

Digital images are discrete representations of the continuous world we live in. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/8a929206239d4447e13f99f29b64fa155367607539e1d8acc47e929209abc1e1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/7cc9f7c500b6af69c75960e94f259315f0eed8ac10be7b0ce7bb7a0ffdc83093.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/d17c8c8542039a771c5061b54056286d6df541dea0b3c7d7eac6a8e6ad234f07.jpg)


$$
n _ {1}
$$

Sampling: allows to represent a continuous image into a finite (pixel) grid. 

$$
n _ {2}
$$

Quantisation: assigns a grey-level describing average brightness at each pixel. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/93e7a519a93a4405597a9591c2ca8e6109a870fd5bf327bb2a384232b9b8ebd0.jpg)


$$
\Omega = \left\{1, \dots , n _ {1} \right\} \times \left\{1, \dots , n _ {2} \right\}
$$

image domain 

$$
\mathbf {X}: \Omega \to \{0, \ldots , 2 5 5 \} \qquad \mathbf {X} = (x _ {i, j}) \in \{0, \ldots , 2 5 5 \} ^ {n _ {1} \times n _ {2}} \underset {\text {normalisation, adjustments..}} {\longrightarrow} \qquad \mathbf {X} \in [ 0, 1 ] ^ {n _ {1} \times n _ {2}} \qquad \mathbf {X} \in \mathbb {R} ^ {n _ {1} \times n _ {2}}
$$

## Digital images

Digital images are discrete representations of the continuous world we live in. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/9040165995b2379a1e3cce1da3ccb3427fb7269a37734aa18e212518d3e30c97.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/b73d883bb782aec1ea719bbbfb4190192f9a44dac3f285001e1e0a9c5a0ab98a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/78cdfc574fc380372b17c30dd133bf16a610f335728ddd3279686d29d1709bf0.jpg)


Sampling: allows to represent a continuous image into a finite (pixel) grid. 

$$
n _ {2}
$$

Quantisation: assigns a grey-level describing average brightness at each pixel. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/86c35549c2cbabef47534872bbf7dddfaaff2c50149f9cb7f977dfef578e793a.jpg)


Ω = {1,…, n<sub>1</sub>} × {1,…, n<sub>2</sub>} image domain 

$$
\mathbf {X}: \Omega \to \{0, \ldots , 2 5 5 \} \qquad \mathbf {X} = (x _ {i, j}) \in \{0, \ldots , 2 5 5 \} ^ {n _ {1} \times n _ {2}} \underset {\text {normalisation, adjustments..}} {\longrightarrow} \qquad \mathbf {X} \in [ 0, 1 ] ^ {n _ {1} \times n _ {2}} \qquad \mathbf {X} \in \mathbb {R} ^ {n _ {1} \times n _ {2}}
$$

Upon vectorisation of the 2D image X, consider a vector $\mathbf { x } \in \mathbb { R } ^ { n } ,$ , with $n = n _ { 1 } n _ { 2 }$ 

Blue 

## Grayscale/RGB images

Natural scenes are not grayscale. Color is a combination of Red-Green-Blue channels. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/5d60c6478ff383804e59bc97464ae663fd67ecb95294171da9f857699fc4af44.jpg)


The higher the intensity in the individual channel, the more represented is the color. 

$$
x _ {i, j} = (r _ {i, j}, g _ {i, j}, b _ {i, j}) \in \mathbb {R} ^ {3}
$$

Hence, for RGB images: 

$$
\mathbf {x} \in \mathbb {R} ^ {n \times 3} \quad \mathrm{or} \quad \mathbb {R} ^ {3 n}
$$

- Other color spaces are possible (CMYK, HSV..) 

- Often, color channels are processed separately. 

## Grayscale/RGB images

Natural scenes are not grayscale. Color is a combination of Red-Green-Blue channels. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/b453bdcb4aed5f359a67088279569ffbcb68e203f7a856e5697b4efd069f841e.jpg)


The higher the intensity in the individual channel, the more represented is the color. 

$$
x _ {i, j} = (r _ {i, j}, g _ {i, j}, b _ {i, j}) \in \mathbb {R} ^ {3}
$$

Hence, for RGB images: 

$$
\mathbf {x} \in \mathbb {R} ^ {n \times 3} \quad \mathrm{or} \quad \mathbb {R} ^ {3 n}
$$

- Other color spaces are possible (CMYK, HSV..) 

- Often, color channels are processed separately. 

Assume x is grayscale (extension is trivial) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/11a1f5eaf02f8daa2e0b37b5829cc138335066641407502e0274e68c3ab5617d.jpg)


## Modelling degradation processes

Images are matrices/vectors. How to model acquisition processes? 

x ∈ ℝ<sup>n</sup> 

(unknown) 

(observed) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/04805a75e4d82c3cd30c69500828ef8eb8a407c50189aafadacedba3e2ff463d.jpg)



A ∈ ℝ<sup>m×n</sup>



linear input (x)-output (y) relation


## Modelling degradation processes

Images are matrices/vectors. How to model acquisition processes? 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/7b51599bc9ee8c7bc7eadd610c5b1100da03d7c84c6094353b02848cdeab9ff2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/034d81e039a882e7dabcf6d134688d2b476b1895e4ad0fbd39b75e5c4494bb75.jpg)



x ∈ ℝ<sup>n</sup>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/ee66d42e8adac1acc6d3af273f76d1406bdb3162f35aae98cebee87af2ba7205.jpg)



(unknown)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/15dcd3f92d15ebe4242023fc375c6f0438af00201528dbd2e62c07a97b4fd0df.jpg)



(observed)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/e83f25e7ff4aab4f0e0b605885b1c1c1cdb4af4d56f43824a57e915aa9d9066f.jpg)


$$
\mathbf {A} \in \mathbb {R} ^ {m \times n}
$$

linear input (x)-output (y) relation 

- Convolution: $\mathbf { A } \mathbf { x }  h * \mathbf { X } ,$ h is a kernel 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/0fa7a32b15896679586b35fe1b42eb22fa1178ced80cf2736e11175d64d7047e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/c9d168e58f909cb098fdb61dc70176453ed60789c7f3146538b904f1c56a524a.jpg)


## Modelling degradation processes

Images are matrices/vectors. How to model acquisition processes? 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/eabf69a3ae039271e89c401af47dd120c5b08b5ef903331f6b422d967b44fe8e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/08c6246d7d2eec969b07cbd085a55753edd0f140a5894183666f420b2c4582b4.jpg)



x ∈ ℝ<sup>n</sup>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/7ff02ce042408169528482b46e8ee94dec9ddb990bc7ba6e2ac4cf6f2fa2f3e4.jpg)



(unknown)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/d59fe1728031a15de88e9081aa48e22fbb34ddc974a742cb4c9bdf88a34e241d.jpg)



(observed)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/9a2a3a8ec2fdba7869bbf7afb73e5b899f3e4af4fba50cf67d045db5fc52f83c.jpg)



A ∈ ℝ<sup>m×n</sup>



linear input (x)-output (y) relation


- Convolution: $\mathbf { A } \mathbf { x }  h * \mathbf { X } ,$ h is a kernel 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/8f0bd5da0050b99998e89b88b71169e201b8511a05353128f49fe33f8e769b98.jpg)



- Masking:


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/7c651e7527586c27521ffd18b7fb105f7df13b4db11be6ee9e6c403517a5259f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/ab518ab0532ab9b13010b0541b491777916a602a32be9224d860ad9ff3bfcf40.jpg)



A = M ∈ {0,1}<sup>m×n</sup>


## Modelling degradation processes

Images are matrices/vectors. How to model acquisition processes? 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/754cc4d06fd4c907ccc755b1d19845d6f77dcdee3fdb198e0ee668dc8ac22681.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/1f63d363c6cfe8fbf9047bd8442364a689f18a31ee8cf7ff338914935fa272b6.jpg)



x ∈ ℝ<sup>n</sup>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/9bb010233a3a7f8fca7233b03462fd2b2c5085ec993ab2508c6df6ad7f786aa0.jpg)



(unknown)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/13fd18f5220af7863134aa7a72e93631faa189a7f69c58ec1f80f2e45fbab478.jpg)



(observed)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/c4260fad55651176d5158d9a00a750de7d00b0651f641a2b073a029284e33778.jpg)


$$
\mathbf {A} \in \mathbb {R} ^ {m \times n}
$$

linear input (x)-output (y) relation 

- Convolution: $\mathbf { A } \mathbf { x }  h * \mathbf { X } ,$ h is a kernel 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/7888527278e2b94bcff7df6c518ffbc855d2a762f0e06ebf8e20807d6d01bd5a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/989a35776bdf7c215d6298d5635340646576c616aa326468c5f89c57014b8c06.jpg)



- Masking:


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/8b7fab6911a72cc87809634e99224774a6d8f289c1d83d0525386adf33575fe9.jpg)



A = M ∈ {0,1}<sup>m×n</sup>



- Fourier transform + Masking:


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/fa2fc62789217d078ac2eb9fbcc0af5fcb8fd011f820d0fe68e051a7d74efb00.jpg)


$$
\mathbf {A} \cdot = \mathbf {M} \mathcal {F} (\cdot)
$$

## Modelling noise

Acquisitions are never perfect. Interferences, errors, faults may happen. 

Noise : $\mathbb { R } ^ { m } \to \mathbb { R } ^ { m }$ codifies instrumental errors. 

$$
\mathbf {y} = \operatorname{Noise} (\mathbf {A x})
$$

In the following: $\mathbf { A } = \mathbf { I } \in \mathbb { R } ^ { n \times n }$ . How to model noise? 

## Modelling noise

Acquisitions are never perfect. Interferences, errors, faults may happen. 

$$
\mathbf {y} = \operatorname{Noise} (\mathbf {A x})
$$

Noise : $\mathbb { R } ^ { m } \to \mathbb { R } ^ { m }$ codifies instrumental errors. 

In the following: $\mathbf { A } = \mathbf { I } \in \mathbb { R } ^ { n \times n }$ . How to model noise? 

Gaussian noise: 

Noise(x) = x + ε, $\pmb { \varepsilon } \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } )$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/7b1028e8a383ea5f5f32b6ded1fbb9f28d112d7c72c7ecdfb173dbfa881fdb51.jpg)



Mostly used due to CLT. Models signal-independent electronic noise


## Modelling noise

Acquisitions are never perfect. Interferences, errors, faults may happen. 

Noise : $\mathbb { R } ^ { m } \to \mathbb { R } ^ { m }$ codifies instrumental errors. 

$$
\mathbf {y} = \operatorname{Noise} (\mathbf {A x})
$$

In the following: $\mathbf { A } = \mathbf { I } \in \mathbb { R } ^ { n \times n }$ . How to model noise? 

Gaussian noise: 

Noise(x) = x + ε, $\pmb { \varepsilon } \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } )$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/e0075de4ee793952bff87daf16da32ac70bc53f9f5e56a34fbc1f5594a391bc9.jpg)



Mostly used due to CLT. Models signal-independent electronic noise


Poisson noise: 

Noise(x) = Pois(x+β), $\mathbf { x } \in \mathbb { R } _ { \geq 0 } ^ { n } , \beta \in \mathbb { R } _ { > 0 } ^ { n }$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/e712953aafe6430f1ca99ede0f5b0ddf2b3b1f06db99f240d51af15df1081291.jpg)



Used in low-photon imaging. Astronomical, microscopy imaging.



Bertero, Boccacci, ‘98


## Modelling noise

Acquisitions are never perfect. Interferences, errors, faults may happen. 

Noise : $\mathbb { R } ^ { m } \to \mathbb { R } ^ { m }$ codifies instrumental errors. 

$$
\mathbf {y} = \operatorname{Noise} (\mathbf {A x})
$$

In the following: $\mathbf { A } = \mathbf { I } \in \mathbb { R } ^ { n \times n }$ . How to model noise? 

Gaussian noise: 

Noise(x) = x + ε, $\pmb { \varepsilon } \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } )$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/85e1730389e17a2e151227c242d70381c108714c8f5e11cdf4d4996224cd38a3.jpg)



Mostly used due to CLT. Models signal-independent electronic noise


Poisson noise: 

Noise(x) = Pois(x+β), $\mathbf { x } \in \mathbb { R } _ { \geq 0 } ^ { n } , \beta \in \mathbb { R } _ { > 0 } ^ { n }$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/b78aad622a308e9e88ae55268d604c282e2ede1786888b2df56e32798c8c30be.jpg)



Used in low-photon imaging. Astronomical, microscopy imaging.


Bertero, Boccacci, ‘98 

Impulsive noise: 

Noise(x) = (1 − s) ⊙ x + s ⊙ c 

$$
c _ {i} = \mathcal {B} (1 / 2), s _ {i} = \mathcal {B} (p), p \in [ 0, 1 ]
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/512e771ab789b1f0a92608026525b07401a7d44c8457cb89eeaa8e5c42b35b83.jpg)


Used to describe faulty detectors and/or long time exposures under bad lighting. 

## Quality metrics: MSE, SNR, PSNR

How to assess image quality between two images using pixel information? 

## Quality metrics: MSE, SNR, PSNR

How to assess image quality between two images using pixel information? 

$$
\mathsf {M S E} (\mathbf {y}, \mathbf {x}) = \frac {1}{n} \sum_ {i = 1} ^ {n} | y _ {i} - x _ {i} | ^ {2} = \frac {1}{n _ {1}} \frac {1}{n _ {2}} \sum_ {i _ {1} = 1} ^ {n _ {1}} \sum_ {i _ {2} = 1} ^ {n _ {2}} | y _ {i _ {1}, i _ {2}} - x _ {i _ {1}, i _ {2}} | ^ {2}
$$

## Quality metrics: MSE, SNR, PSNR

How to assess image quality between two images using pixel information? 

$$
\mathsf {M S E} (\mathbf {y}, \mathbf {x}) = \frac {1}{n} \sum_ {i = 1} ^ {n} | y _ {i} - x _ {i} | ^ {2} = \frac {1}{n _ {1}} \frac {1}{n _ {2}} \sum_ {i _ {1} = 1} ^ {n _ {1}} \sum_ {i _ {2} = 1} ^ {n _ {2}} | y _ {i _ {1}, i _ {2}} - x _ {i _ {1}, i _ {2}} | ^ {2}
$$

$$
\mathsf {S N R} (\mathbf {y}, \mathbf {x}) = 1 0 \log_ {1 0} \left(\frac {\| \mathbf {x} \| ^ {2}}{\| \mathbf {x} - \mathbf {y} \| ^ {2}}\right) \mathrm{noise} \varepsilon
$$

## Quality metrics: MSE, SNR, PSNR

How to assess image quality between two images using pixel information? 

$$
\mathsf {M S E} (\mathbf {y}, \mathbf {x}) = \frac {1}{n} \sum_ {i = 1} ^ {n} | y _ {i} - x _ {i} | ^ {2} = \frac {1}{n _ {1}} \frac {1}{n _ {2}} \sum_ {i _ {1} = 1} ^ {n _ {1}} \sum_ {i _ {2} = 1} ^ {n _ {2}} | y _ {i _ {1}, i _ {2}} - x _ {i _ {1}, i _ {2}} | ^ {2}
$$

$$
\mathsf {S N R} (\mathbf {y}, \mathbf {x}) = 1 0 \log_ {1 0} \left(\frac {\| \mathbf {x} \| ^ {2}}{\| \mathbf {x} - \mathbf {y} \| ^ {2}}\right) \mathrm{noise} \varepsilon
$$

$$
\mathsf {P S N R} (\mathbf {y}, \mathbf {x}) = 1 0 \log_ {1 0} \left(\frac {\mathsf {M A X} ^ {2}}{\mathsf {M S E} (\mathbf {y} , \mathbf {x})}\right)
$$

where MAX is the highest possible value $( \boldsymbol { \mathrm { e } } , \boldsymbol { \mathrm { g } } _ { \flat }$ 255 or 1) 

## Quality metrics: MSE, SNR, PSNR

How to assess image quality between two images using pixel information? 

$$
\mathsf {M S E} (\mathbf {y}, \mathbf {x}) = \frac {1}{n} \sum_ {i = 1} ^ {n} | y _ {i} - x _ {i} | ^ {2} = \frac {1}{n _ {1}} \frac {1}{n _ {2}} \sum_ {i _ {1} = 1} ^ {n _ {1}} \sum_ {i _ {2} = 1} ^ {n _ {2}} | y _ {i _ {1}, i _ {2}} - x _ {i _ {1}, i _ {2}} | ^ {2}
$$

$$
\mathsf {S N R} (\mathbf {y}, \mathbf {x}) = 1 0 \log_ {1 0} \left(\frac {\| \mathbf {x} \| ^ {2}}{\| \mathbf {x} - \mathbf {y} \| ^ {2}}\right)
$$

noise ε 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/7e23298816bac6b774fc13eb09340b070fb24c75b5f35443eb8280a6c7c76504.jpg)



Original


$$
\mathsf {P S N R} (\mathbf {y}, \mathbf {x}) = 1 0 \log_ {1 0} \left(\frac {\mathsf {M A X} ^ {2}}{\mathsf {M S E} (\mathbf {y} , \mathbf {x})}\right)
$$

where MAX is the highest possible value (e.g., 255 or 1) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/ad06c8a072657179ce6e4948c3659305b7cce6e18a97038426e09389ea762350.jpg)



PSNR=26.547


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/bf9a0e325ae8179c098f74f1bcb31589463446279ef09bd94f0e5f15e2f5df82.jpg)



PSNR=26.547


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/a0d27691d24489d0d0f1dc7795d36e606e9aef648076a625ea4b4b843260c5b2.jpg)



PSNR=26.547


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/9739131250493a7a8bff50dc279f809f4f95a304e2dd3a7add07b73dd649262a.jpg)


Sensitive to pixel variations Is it a good quality metric for natural images?! 

## Quality metrics: SSIM

$$
\mathsf {S S I M} (x, y) = \frac {(2 \mu_ {\mathbf {x}} \mu_ {\mathbf {y}} + C _ {1}) (2 \sigma_ {\mathbf {x y}} + C _ {2})}{(\mu_ {\mathbf {x}} ^ {2} + \mu_ {\mathbf {y}} ^ {2} + C _ {1}) (\sigma_ {\mathbf {x}} ^ {2} + \sigma_ {\mathbf {y}} ^ {2} + C _ {2})} \in [ 0, 1 ]
$$

- Based on image statistics: mean, variance, covariance + constant $C _ { 1 } , C _ { 2 }$ stabilising the division. 

- Typically performed on small image patches + averaging 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/99660e0f113bbafbc44f93d6ce8b0d80a3a3f38e7d42f6aa23fc617d80104ec7.jpg)



Original SSIM=1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/9ef1ff64281c48cc9a6c0403bd013f40833c068dcc02e7d5c780471b3c9b3e3f.jpg)



PSNR=26.547 SSIM=0.988


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/1cb20dcac3c6fd8045f2eab8e2200661469a682639f5c1f24d7ace4530428cf1.jpg)



PSNR=26.547 SSIM=0.840


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/ad512f0a495b3bd12781fe1942a74ffd37523d50976506d45451a79c84b31605.jpg)



PSNR=26.547 SSIM=0.694


## Quality metrics: SSIM

$$
\mathsf {S S I M} (x, y) = \frac {(2 \mu_ {\mathbf {x}} \mu_ {\mathbf {y}} + C _ {1}) (2 \sigma_ {\mathbf {x y}} + C _ {2})}{(\mu_ {\mathbf {x}} ^ {2} + \mu_ {\mathbf {y}} ^ {2} + C _ {1}) (\sigma_ {\mathbf {x}} ^ {2} + \sigma_ {\mathbf {y}} ^ {2} + C _ {2})} \in [ 0, 1 ]
$$

- Based on image statistics: mean, variance, covariance + constant $C _ { 1 } , C _ { 2 }$ stabilising the division. 

- Typically performed on small image patches + averaging 

All such metrics are supervised. They depend on ground-truth x. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/f01879099bf5286b1891e9a6621544ab3de9857640937c7ee44cdad6a056f023.jpg)



Original SSIM=1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/fc4cbb975843f9b43de4ba03d58c4eb89169eb6e1b9079d68e80cba5464d2629.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/05ecbe0dfeadb0bee93b1cbb67e3abc6be408dcc589a1cbc44c9ee68c0ad17e1.jpg)



PSNR=26.547 SSIM=0.988



PSNR=26.547 SSIM=0.840


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/f6f3949f296990ee465826114a98ce6942be593e800151cd2de8ebfbf38297fe.jpg)



PSNR=26.547 SSIM=0.694


<table><tr><td></td><td>RANGE</td><td>IDEAL</td><td>GOOD</td></tr><tr><td>MSE</td><td><eq>[0,MAX^2]</eq></td><td>0</td><td>&lt; 100(for images in [0,255])</td></tr><tr><td>SNR</td><td><eq>(-∞,+∞)</eq></td><td>+∞</td><td>&gt;30</td></tr><tr><td>PSNR</td><td><eq>[0,+∞)</eq></td><td>+∞</td><td>&gt;30</td></tr><tr><td>SSIM</td><td>[0,1]</td><td>1</td><td>&gt;0.9</td></tr></table>

## Image denoising as a ‘toy’ inverse problem

Given $\mathbf { y } \in \mathbb { R } ^ { n }$ , find $\mathbf { x } \in \mathbb { R } ^ { n }$ such that: 

- “Inverse problem” (operator to invert is $\mathbf { A } = \mathbf { I } )$ 

$$
\mathbf {y} = \mathbf {x} + \pmb {\varepsilon}, \quad \pmb {\varepsilon} \sim \mathcal {N} (\mathbf {0}, \sigma^ {2} \mathbf {I})
$$

- Still challenging: the noise realisation is unknown 

- If noise is high ( is big), image content can be lostσ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/d4fb6de9af7e56f5f6ba5ab9e036aecd4f270017df3acdc1fe1abdabea4bba04.jpg)



σ = 0.002


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/00dbccc0718bed76179136ef2e4e28982a8bb362689ab17052811a520b45488b.jpg)



σ = 0.05


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/429f438e374be6595e312f0aaf4da9744ed23f845426747dae75e52d9f22e297.jpg)



σ = 0.3


$$
\mathbf {x} \in [ 0, 1 ] ^ {3 n}
$$

How to model this problem in mathematical terms? 

MSE 

PSNR/SSIM 

## Bayesian formulation

Idea: model also x and y as realisation of random variables $\mathcal { X } \sim \pi _ { \mathcal { X } } , \mathcal { Y } \sim \pi _ { \mathcal { Y } } +$ conditional laws. 

## Bayesian formulation

Idea: model also x and y as realisation of random variables $\mathcal { X } \sim \pi _ { \mathcal { X } } , \mathcal { Y } \sim \pi _ { \mathcal { Y } } +$ conditional laws. 

$\boxed { \pi _ { \mathcal { X } \mid \mathcal { Y } } ( \mathbf { X } \mid \mathbf { Y } ) }$ Posterior distribution: what we would like to maximise. 

$\boxed { \pi _ { \mathcal { Y } | \mathcal { X } } ( \mathbf { y } \mid \mathbf { x } ) }$ Likelihood: function describing the probability of observing the data given a choice of x 

$\boxed { \pi _ { \mathcal { Y } } ( \mathbf { y } ) }$ Evidence term: normally neglected, does not depend on x 

$\boxed { \pmb { \pi } _ { \mathcal { X } } ( \mathbf { X } ) }$ Prior: prior assumptions on the unknown quantity x 

## Bayesian formulation

Idea: model also x and y as realisation of random variables $\mathcal { X } \sim \pi _ { \mathcal { X } } , \mathcal { Y } \sim \pi _ { \mathcal { Y } } +$ conditional laws. 

$\boxed { \pi _ { \mathcal { X } \mid \mathcal { Y } } ( \mathbf { X } \mid \mathbf { Y } ) }$ Posterior distribution: what we would like to maximise. 

$\boxed { \pi _ { \mathcal { Y } | \mathcal { X } } ( \mathbf { y } \mid \mathbf { x } ) }$ Likelihood: function describing the probability of observing the data given a choice of x 

$\boxed { \pi _ { \mathcal { Y } } ( \mathbf { y } ) }$ Evidence term: normally neglected, does not depend on x 

$\boxed { \pi _ { \mathcal { X } } ( \mathbf { X } ) }$ Prior: prior assumptions on the unknown quantity x 

$$
\pi_ {\mathcal {X} | \mathcal {Y}} (\mathbf {x} | \mathbf {y}) = \frac {\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x}) \pi_ {\mathcal {X}} (\mathbf {x})}{\pi_ {\mathcal {Y}} (\mathbf {y})}
$$

Under Gaussian noise assumption: 

$$
\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x}) = \pi_ {E} (\pmb {\varepsilon} = \mathbf {y} - \mathbf {x})
$$

Bayes’ Theorem 

## Bayesian formulation: priors, score functions

$$
\mathbf {x} ^ {*} \in \operatorname{argmax} _ {\mathbf {x}} \pi_ {\mathcal {X} | \mathcal {Y}} (\mathbf {x} | \mathbf {y}) = \operatorname{argmax} _ {\mathbf {x}} \frac {\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x}) \pi_ {\mathcal {X}} (\mathbf {x})}{\pi_ {\mathcal {Y}} (\mathbf {y})} = \operatorname{argmax} _ {\mathbf {x}} \pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x}) \pi_ {\mathcal {X}} (\mathbf {x})
$$

By taking the negative logarithm: 

$$
\mathbf {x} ^ {*} \in \operatorname{argmin} _ {\mathbf {x}} - \ln \left(\pi_ {\mathcal {X} | \mathcal {Y}} (\mathbf {x} | \mathbf {y})\right) = \operatorname{argmin} _ {\mathbf {x}} - \ln \left(\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x}) \pi_ {\mathcal {X}} (\mathbf {x})\right) = \operatorname{argmin} _ {\mathbf {x}} - \ln \left(\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x})\right) - \ln \left(\pi_ {\mathcal {X}} (\mathbf {x})\right)
$$

## Bayesian formulation: priors, score functions

$$
\mathbf {x} ^ {*} \in \operatorname{argmax} _ {\mathbf {x}} \pi_ {\mathcal {X} | \mathcal {Y}} (\mathbf {x} | \mathbf {y}) = \operatorname{argmax} _ {\mathbf {x}} \frac {\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x}) \pi_ {\mathcal {X}} (\mathbf {x})}{\pi_ {\mathcal {Y}} (\mathbf {y})} = \operatorname{argmax} _ {\mathbf {x}} \pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x}) \pi_ {\mathcal {X}} (\mathbf {x})
$$

By taking the negative logarithm: 

$$
\mathbf {x} ^ {*} \in \operatorname{argmin} _ {\mathbf {x}} - \ln \left(\pi_ {\mathcal {X} | \mathcal {Y}} (\mathbf {x} | \mathbf {y})\right) = \operatorname{argmin} _ {\mathbf {x}} - \ln \left(\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x}) \pi_ {\mathcal {X}} (\mathbf {x})\right) = \operatorname{argmin} _ {\mathbf {x}} - \ln \left(\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x})\right) - \ln \left(\pi_ {\mathcal {X}} (\mathbf {x})\right)
$$

Example: 

$$
\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x}) = \pi_ {E} (\mathbf {y} - \mathbf {x}), \quad \pi_ {\mathcal {X}} (\mathbf {x}) = \mathcal {N} (\mu_ {\mathbf {x}}, \Sigma_ {\mathbf {x}})
$$

$$
\begin{array}{r l} & {\mathbf {x} ^ {*} \in \operatorname{argmin} _ {\mathbf {x}} - \ln \left(\prod_ {i = 1} ^ {n} \frac {1}{\sqrt {2 \pi \sigma_ {\varepsilon} ^ {2}}} \exp \left(- \frac {| y _ {i} - x _ {i} | ^ {2}}{2 \sigma_ {\varepsilon} ^ {2}}\right)\right) - \ln \left(\frac {1}{(2 \pi) ^ {n / 2} | \pmb {\Sigma} _ {x} | ^ {1 / 2}} \exp \left(- \frac {1}{2} (\mathbf {x} - \pmb {\mu} _ {x}) ^ {\top} \pmb {\Sigma} _ {x} ^ {- 1} (\mathbf {x} - \pmb {\mu} _ {x})\right)\right)} \\ & {\quad = \operatorname{argmin} _ {\mathbf {x}} \frac {1}{2 \sigma_ {\varepsilon} ^ {2}} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \frac {1}{2} \| \mathbf {x} - \pmb {\mu} _ {x} \| _ {\pmb {\Sigma} _ {x} ^ {- 1}} ^ {2} + \mathrm{neglectingconstants}} \end{array}
$$

—-> towards optimisation problem! 

From a statistical to a variational perspective: optimisation 

$$
\operatorname{argmin} _ {\mathbf {x}} J (\mathbf {x}) := - \ln \left(\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x})\right) - \ln \left(\pi_ {\mathcal {X}} (\mathbf {x})\right)
$$

Whenever likelihood + prior functionals belong to a log-concave exponential family we end up with convex optimisation problems for a functional J : $\mathbb { R } ^ { n } \to \mathbb { R } _ { \geq 0 } \cup \{ + \infty \}$ 

## From a statistical to a variational perspective: optimisation

$$
\operatorname{argmin} _ {\mathbf {x}} J (\mathbf {x}) := - \ln \left(\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x})\right) - \ln \left(\pi_ {\mathcal {X}} (\mathbf {x})\right)
$$

Whenever likelihood + prior functionals belong to a log-concave exponential family we end up with convex optimisation problems for a functional $J : \mathbb { R } ^ { n } \to \mathbb { R } _ { \geq 0 } \cup \{ + \infty \}$ 

We are going to see a very simple algorithm for minimising J based on gradients. Note: 

$$
\nabla J (\mathbf {x} ^ {*}) = - \nabla \ln \left(\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x} ^ {*})\right) - \nabla \ln \left(\pi_ {\mathcal {X}} (\mathbf {x} ^ {*})\right) = \mathbf {0}
$$

when looking for MAP estimators. 

## From a statistical to a variational perspective: optimisation

$$
\operatorname{argmin} _ {\mathbf {x}} J (\mathbf {x}) := - \ln \left(\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x})\right) - \ln \left(\pi_ {\mathcal {X}} (\mathbf {x})\right)
$$

Whenever likelihood + prior functionals belong to a log-concave exponential family we end up with convex optimisation problems for a functional J : $\mathbb { R } ^ { n } \to \mathbb { R } _ { \geq 0 } \cup \{ + \infty \}$ 

We are going to see a very simple algorithm for minimising J based on gradients. Note: 

$$
\nabla J (\mathbf {x} ^ {*}) = - \nabla \ln \left(\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x} ^ {*})\right) - \nabla \ln \left(\pi_ {\mathcal {X}} (\mathbf {x} ^ {*})\right) = \mathbf {0}
$$

when looking for MAP estimators. 

<table><tr><td></td><td>Bayesian model</td><td>Optimisation approach</td></tr><tr><td>Noise information</td><td>Likelihood</td><td>Data-term</td></tr><tr><td>A-priori information</td><td>Prior</td><td>Regularisation</td></tr><tr><td>Parameters</td><td>Hyperparameters</td><td>Regularisation parameters</td></tr></table>

## From a statistical to a variational perspective: optimisation

$$
\operatorname{argmin} _ {\mathbf {x}} J (\mathbf {x}) := - \ln \left(\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x})\right) - \ln \left(\pi_ {\mathcal {X}} (\mathbf {x})\right)
$$

Whenever likelihood + prior functionals belong to a log-concave exponential family we end up with convex optimisation problems for a functional J : $\mathbb { R } ^ { n } \to \mathbb { R } _ { \geq 0 } \cup \{ + \infty \}$ 

We are going to see a very simple algorithm for minimising J based on gradients. Note: 

$$
\nabla J (\mathbf {x} ^ {*}) = - \nabla \ln \left(\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x} ^ {*})\right) - \nabla \ln \left(\pi_ {\mathcal {X}} (\mathbf {x} ^ {*})\right) = \mathbf {0}
$$

when looking for MAP estimators. 

How to choose good noise and image models? 

<table><tr><td></td><td>Bayesian model</td><td>Optimisation approach</td></tr><tr><td>Noise information</td><td>Likelihood</td><td>Data-term</td></tr><tr><td>A-priori information</td><td>Prior</td><td>Regularisation</td></tr><tr><td>Parameters</td><td>Hyperparameters</td><td>Regularisation parameters</td></tr></table>

## Data terms

$$
\mathbf {y} = \text { Noise } (\mathbf {x}) \quad \xrightarrow {\cdots \cdots} \quad \pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y}   |   \mathbf {x}) \quad \xrightarrow [ +   \text { constants } ]{- \ln} \quad D (\mathbf {y}, \mathbf {x}) \quad \text { Tailored   distance   function   with   observation   related   to   noise   assumptions? }
$$

## Data terms

$$
\mathbf {y} = \text { Noise } (\mathbf {x}) \quad \xrightarrow {\quad \cdots \quad \cdots \quad \cdots \quad \to} \quad \pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} \mid \mathbf {x}) \quad \xrightarrow [ + \text { constants } ]{- \ln} \quad D (\mathbf {y}, \mathbf {x})   \text { Tailored   distance   function   with   observation   related   to   noise   assumptions? }
$$

Gaussian noise: 

$$
\operatorname{Noise} (\mathbf {x}) = \mathbf {x} + \boldsymbol {\varepsilon}, \quad \boldsymbol {\varepsilon} \sim \mathcal {N} (\mathbf {0}, \sigma_ {\varepsilon} ^ {2} \mathbf {I}) \quad \dots\dots\rightarrow \quad D (\mathbf {y}, \mathbf {x}) = D (\mathbf {y} - \mathbf {x}) = \frac {1}{2 \sigma_ {\varepsilon} ^ {2}} \| \mathbf {y} - \mathbf {x} \| _ {2} ^ {2}
$$

## Data terms

$$
\mathbf {y} = \text { Noise } (\mathbf {x}) \quad \xrightarrow {\cdots \cdots} \quad \pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y}   |   \mathbf {x}) \quad \xrightarrow [ + \text { constants } ]{- \ln} \quad D (\mathbf {y}, \mathbf {x}) \quad \text { Tailored   distance   function   with   observation   related   to   noise   assumptions? }
$$

## Gaussian noise:

$$
\operatorname{Noise} (\mathbf {x}) = \mathbf {x} + \boldsymbol {\varepsilon}, \quad \boldsymbol {\varepsilon} \sim \mathcal {N} (\mathbf {0}, \sigma_ {\varepsilon} ^ {2} \mathbf {I}) \quad \dots\dots\rightarrow \quad D (\mathbf {y}, \mathbf {x}) = D (\mathbf {y} - \mathbf {x}) = \frac {1}{2 \sigma_ {\varepsilon} ^ {2}} \| \mathbf {y} - \mathbf {x} \| _ {2} ^ {2}
$$

Poisson noise: 

$$
\mathsf {N o i s e} (\mathbf {x}) = \mathsf {P o i s} (\mathbf {x} + \boldsymbol {\beta}), \quad \mathbf {x} \in \mathbb {R} _ {\geq 0} ^ {n},   \boldsymbol {\beta} \in \mathbb {R} _ {> 0} ^ {n} \quad \dots\dots\rightarrow \quad D (\mathbf {y}, \mathbf {x}) = D (\mathbf {y}, \mathbf {x} + \boldsymbol {\beta}) = \mathsf {K L} (\mathbf {y}, \mathbf {x} + \boldsymbol {\beta}) = \sum_ {i = 1} ^ {n} x _ {i} + \beta_ {i} - y _ {i} \ln \left(x _ {i} + \beta_ {i}\right)
$$

## Data terms

$$
\mathbf {y} = \text { Noise } (\mathbf {x}) \quad \xrightarrow {\cdots \cdots} \quad \pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x}) \quad \xrightarrow [ + \text { constants } ]{- \ln} \quad D (\mathbf {y}, \mathbf {x})   \text { Tailored   distance   function   with   observation   related   to   noise   assumptions? }
$$

Gaussian noise: 

$$
\operatorname{Noise} (\mathbf {x}) = \mathbf {x} + \boldsymbol {\varepsilon}, \quad \boldsymbol {\varepsilon} \sim \mathcal {N} (\mathbf {0}, \sigma_ {\varepsilon} ^ {2} \mathbf {I}) \quad \dots\dots\rightarrow \quad D (\mathbf {y}, \mathbf {x}) = D (\mathbf {y} - \mathbf {x}) = \frac {1}{2 \sigma_ {\varepsilon} ^ {2}} \| \mathbf {y} - \mathbf {x} \| _ {2} ^ {2}
$$

Poisson noise: 

$$
\operatorname{Noise} (\mathbf {x}) = \operatorname{Pois} (\mathbf {x} + \boldsymbol {\beta}), \quad \mathbf {x} \in \mathbb {R} _ {\geq 0} ^ {n}, \boldsymbol {\beta} \in \mathbb {R} _ {> 0} ^ {n} \quad \dots\dots\rightarrow D (\mathbf {y}, \mathbf {x}) = D (\mathbf {y}, \mathbf {x} + \boldsymbol {\beta}) = \mathsf {K L} (\mathbf {y}, \mathbf {x} + \boldsymbol {\beta}) = \sum_ {i = 1} ^ {n} x _ {i} + \beta_ {i} - y _ {i} \ln \left(x _ {i} + \beta_ {i}\right)
$$

Impulsive noise: 

$$
\begin{array}{l l} \operatorname{Noise} (\mathbf {x}) = (\mathbf {1} - \mathbf {s}) \odot \mathbf {x} + \mathbf {s} \odot \mathbf {c} & \dots\dots\\ c _ {i} = \mathcal {B} (1 / 2), s _ {i} = \mathcal {B} (p), p \in [ 0, 1 ] & D (\mathbf {y}, \mathbf {x}) = D (\mathbf {y} - \mathbf {x}) = \frac {1}{\tau_ {\varepsilon}} \| \mathbf {y} - \mathbf {x} \| _ {1} \\ \approx \mathbf {x} + \boldsymbol {\varepsilon}, \quad \boldsymbol {\varepsilon} \sim \mathcal {L} (\mathbf {0}, \tau_ {\varepsilon} \mathbf {I}) & \text {noise "sparsity" (the residual is 0 only in few pixels)} \end{array}
$$

UniGe 

Regularisation terms 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) \underset {+ \text {constants}} {\overset {- \ln} {\longrightarrow}} R (\mathbf {x}) _ {\text {codify a - priori information}}
$$

Regularisation terms 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) \quad \begin{array}{c} - \ln \\ + \text { constants } \end{array} \quad R (\mathbf {x}) _ {\text { codify }}
$$

codify a-priori information 

Regularity around the mean: 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \mathcal {N} (\mu_ {\mathbf {x}}, \boldsymbol {\Sigma} _ {x}) \quad \dots\dots\blacktriangleright \quad R (\mathbf {x}) = \frac {1}{2} \| \mathbf {x} - \mu_ {\mathbf {x}} \| _ {\boldsymbol {\Sigma} _ {x} ^ {- 1}} ^ {2}
$$

## Regularisation terms

$$
\pi_ {\mathcal {X}} (\mathbf {x}) \quad \begin{array}{c} - \ln \\ + \text {constants} \end{array} \quad R (\mathbf {x}) _ {\text {codify}}
$$

codify a-priori information 

Regularity around the mean: 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \mathcal {N} (\mu_ {\mathbf {x}}, \boldsymbol {\Sigma} _ {x}) \quad \dots\dots\blacktriangleright \quad R (\mathbf {x}) = \frac {1}{2} \| \mathbf {x} - \mu_ {\mathbf {x}} \| _ {\boldsymbol {\Sigma} _ {x} ^ {- 1}} ^ {2}
$$

Sparsity: 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \mathcal {L} (\mathbf {0}, \tau \mathbf {I}) \qquad \dots\dots► R (\mathbf {x}) = \frac {1}{\tau} \| \mathbf {x} \| _ {1}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/ffcc18a423bc76f7ae6cd4584aaa7fa13467839d1a508fb43a48d313f5a9384b.jpg)


Regularisation terms 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) \quad \begin{array}{c} - \ln \\ + \text {constants} \end{array} \quad R (\mathbf {x}) _ {\text {codify}}
$$

codify a-priori information 

Regularity around the mean: 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \mathcal {N} (\mu_ {\mathbf {x}}, \boldsymbol {\Sigma} _ {x}) \quad \dots\dots\blacktriangleright \quad R (\mathbf {x}) = \frac {1}{2} \| \mathbf {x} - \mu_ {\mathbf {x}} \| _ {\boldsymbol {\Sigma} _ {x} ^ {- 1}} ^ {2}
$$

Sparsity: 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \mathcal {L} (\mathbf {0}, \tau \mathbf {I}) \qquad \dots\dots\blacktriangleright \qquad R (\mathbf {x}) = \frac {1}{\tau} \| \mathbf {x} \| _ {1}
$$

$$
q _ {i} = \| (\nabla \mathbf {x}) _ {i} \| _ {2}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/48ecd26c85bed4401126e95bcd57c3fe78fc80303ead18340698ca5046b68292.jpg)


## Regularisation terms

$$
\pi_ {\mathcal {X}} (\mathbf {x}) \quad \begin{array}{c} - \ln \\ + \text {constants} \end{array} \quad R (\mathbf {x}) _ {\text {codify}}
$$

codify a-priori information 

Regularity around the mean: 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \mathcal {N} (\mu_ {\mathbf {x}}, \boldsymbol {\Sigma} _ {x}) \quad \dots\dots\blacktriangleright \quad R (\mathbf {x}) = \frac {1}{2} \| \mathbf {x} - \mu_ {\mathbf {x}} \| _ {\boldsymbol {\Sigma} _ {x} ^ {- 1}} ^ {2}
$$

Sparsity: 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \mathcal {L} (\mathbf {0}, \tau \mathbf {I}) \qquad \dots\dots► R (\mathbf {x}) = \frac {1}{\tau} \| \mathbf {x} \| _ {1}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/789e5a06627b83b3f2624a31361cfc04459a37ace784a5596a0e2027ec525e1d.jpg)


Smoothness: 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \pi_ {\mathcal {Q}} (\mathbf {q}) = \mathcal {N} (\mathbf {0}, \sigma_ {\mathbf {q}} ^ {2} \mathbf {I}) \quad \dots\dots\blacktriangleright \quad R (\mathbf {x}) = \frac {1}{2 \sigma_ {\mathbf {q}} ^ {2}} \| \nabla \mathbf {x} \| _ {2} ^ {2}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/d8e60bb0b8bc7ad10d4ed95dcc92b2c9e900896756b18b8912cc0fb8a67cdcd7.jpg)


$$
q _ {i} = \| (\nabla \mathbf {x}) _ {i} \| _ {2}
$$

## Regularisation terms

Regularity around the mean: 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) \quad \begin{array}{c} - \ln \\ + \text {constants} \end{array}
$$

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \mathcal {N} (\mu_ {\mathbf {x}}, \boldsymbol {\Sigma} _ {x})
$$

$$
R (\mathbf {x})
$$

$$
R (\mathbf {x}) = \frac {1}{2} \| \mathbf {x} - \mu_ {\mathbf {x}} \| _ {\boldsymbol {\Sigma} _ {x} ^ {- 1}} ^ {2}
$$

codify a-priori information 

Sparsity: 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \mathcal {L} (\mathbf {0}, \tau \mathbf {I})
$$

$$
R (\mathbf {x}) = \frac {1}{\tau} \| \mathbf {x} \| _ {1}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/e48a5d701fe9141220f5c89d8d3356370219c8a2e6dd606237ffdec690c953fb.jpg)


Smoothness: 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \pi_ {\mathcal {Q}} (\mathbf {q}) = \mathcal {N} (\mathbf {0}, \sigma_ {\mathbf {q}} ^ {2} \mathbf {I})
$$

$$
R (\mathbf {x}) = \frac {1}{2 \sigma_ {\mathbf {q}} ^ {2}} \| \nabla \mathbf {x} \| _ {2} ^ {2}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/86ce6d0f0b686505c14f2c40c1ca180bb5b62908eb2710803567d29481a1b727.jpg)


Piece-wise constancy: 

$$
q _ {i} = \| (\nabla \mathbf {x}) _ {i} \| _ {2}
$$

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \pi_ {\mathcal {Q}} (\mathbf {q}) = \mathcal {L} (\mathbf {0}, \tau \mathbf {I})
$$

$$
R (\mathbf {x}) = \frac {1}{\tau} \mathsf {T V} (\mathbf {x}) = \frac {1}{\tau} \| \nabla \mathbf {x} \| _ {2, 1}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/fe0ac0721d6c509e216d650a6beb6ed0e7aded4e9952a3b0664281ff345f938e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/b9d39db67b92d6f8965b47303f0f76f7c0db5e94413715595bec41ae7d9e2056.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/4ea895cb40b7f6c55b08526767764d0927dae2cd430f2601b246d5dbf1680362.jpg)


Total Variation 

## Regularisation terms

Regularity around the mean: 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \mathcal {N} (\mu_ {\mathbf {x}}, \boldsymbol {\Sigma} _ {x})
$$

Sparsity: 

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \mathcal {L} (\mathbf {0}, \tau \mathbf {I})
$$

$$
R (\mathbf {x}) = \frac {1}{2} \| \mathbf {x} - \mu_ {\mathbf {x}} \| _ {\boldsymbol {\Sigma} _ {x} ^ {- 1}} ^ {2}
$$

$$
R (\mathbf {x}) = \frac {1}{\tau} \| \mathbf {x} \| _ {1}
$$

$$
\pi_ {\mathcal {X}} (\mathbf {x}) \begin{array}{c} - \ln \\ + \text {constants} \end{array}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/8a33bfd183c36d1d17a8e101eb8ee8c58d39b328e747ede04db31e916ab26d34.jpg)


Piece-wise constancy: 

Smoothness: 

$$
q _ {i} = \| (\nabla \mathbf {x}) _ {i} \| _ {2}
$$

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \pi_ {\mathcal {Q}} (\mathbf {q}) = \mathcal {L} (\mathbf {0}, \tau \mathbf {I})
$$

$$
\pi_ {\mathcal {X}} (\mathbf {x}) = \pi_ {\mathcal {Q}} (\mathbf {q}) = \mathcal {N} (\mathbf {0}, \sigma_ {\mathbf {q}} ^ {2} \mathbf {I})
$$

$$
R (\mathbf {x})
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/8efa47a08bcad3c5cc5c5bb4e75d02c30be9a4de3cbf024335b5b1b4f636c45c.jpg)


$$
R (\mathbf {x}) = \frac {1}{2 \sigma_ {\mathbf {q}} ^ {2}} \| \nabla \mathbf {x} \| _ {2} ^ {2}
$$

codify a-priori information 

Tikhonov 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/b4c651eebe18165cb5fbcca155978e07da3499748f3f98e5f711ae366e0f9cd8.jpg)


Sparse reg./ Compressed sensing 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/6d0e8ac9960010493c287b2e3a9ba208e56309105137dbcc3d1dfe82cc58a55d.jpg)


$$
R (\mathbf {x}) = \frac {1}{\tau} \mathsf {T V} (\mathbf {x}) = \frac {1}{\tau} \| \nabla \mathbf {x} \| _ {2, 1}
$$

Tikhonov/ Sobolev 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/5dede0ef583489ec351cc0845a435c4b10e8040575a25de1fe6c146aac22447c.jpg)


## The regularisation parameter

Example: 

$$
\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x}) = \pi_ {E} (\mathbf {y} - \mathbf {x}), \quad \pi_ {\mathcal {X}} (\mathbf {x}) = \mathcal {N} (\mathbf {0}, \sigma_ {\mathbf {x}} ^ {2} \mathbf {I})
$$

$$
\operatorname{argmin} _ {\mathbf {x}} \frac {1}{2 \sigma_ {\varepsilon} ^ {2}} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \frac {1}{2 \sigma_ {x} ^ {2}} \| \mathbf {x} \| _ {2} ^ {2} = \operatorname{argmin} _ {\mathbf {x}} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \frac {\sigma_ {\varepsilon} ^ {2}}{\sigma_ {x} ^ {2}} \| \mathbf {x} \| ^ {2} = \operatorname{argmin} _ {\mathbf {x}} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \lambda \| \mathbf {x} \| ^ {2}
$$

Ratio between noise level (can be estimated) and image statistical features (hard to estimate). 

## The regularisation parameter

Example: 

$$
\pi_ {\mathcal {Y} | \mathcal {X}} (\mathbf {y} | \mathbf {x}) = \pi_ {E} (\mathbf {y} - \mathbf {x}), \quad \pi_ {\mathcal {X}} (\mathbf {x}) = \mathcal {N} (\mathbf {0}, \sigma_ {\mathbf {x}} ^ {2} \mathbf {I})
$$

$$
\operatorname{argmin} _ {\mathbf {x}} \frac {1}{2 \sigma_ {\varepsilon} ^ {2}} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \frac {1}{2 \sigma_ {x} ^ {2}} \| \mathbf {x} \| _ {2} ^ {2} = \operatorname{argmin} _ {\mathbf {x}} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \frac {\sigma_ {\varepsilon} ^ {2}}{\sigma_ {x} ^ {2}} \| \mathbf {x} \| ^ {2} = \operatorname{argmin} _ {\mathbf {x}} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \lambda \| \mathbf {x} \| ^ {2}
$$

Ratio between noise level (can be estimated) and image statistical features (hard to estimate). 

More in general, the regularisation parameter $\lambda > 0$ weights data fit against regularisation. 

$$
\operatorname{argmin} _ {\mathbf {x}} D (\mathbf {y}, \mathbf {x}) + \lambda R (\mathbf {x})
$$

- Small λ: low regularisation, trust in the data, noise overfit 

- High λ: high regularisation, need to regularise the data, artefacts induced by $R$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/08d360ea20ded1c763f85e7ab1c7a3caec5d58bbae5b800eeb03f9b6eed214b8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/3e82f6894607038751aef9d8b625bd544d00d06dae5facffeb5adb180afcea9d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/d22cbba6d7ec4d528b27f5192e0762e8d3f1b30629f87234cb3a02a018879683.jpg)


$$
\lambda \rightarrow + \infty
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-23/41682cc9-a76a-4ab7-b014-90da5cfad9fb/20b7aa78289389a6644254fc88e7ab2cf53e059a71da70bed5512590c642eb94.jpg)


Examples 

$$
\mathbf {y} = \operatorname{Noise} (\mathbf {x})
$$

Gaussian noise + piece-wise constant image: reference model for many applications. 

$$
\operatorname{argmin} _ {\mathbf {x}} \frac {1}{2} \| \mathbf {y} - \mathbf {x} \| _ {2} ^ {2} + \lambda \mathsf {T V} (\mathbf {x})
$$

Poisson noise + sparse signal: used in microscopy/astronomical imaging 

$$
\operatorname{argmin} _ {\mathbf {x} \geq 0} \operatorname{KL} (\mathbf {y}, \mathbf {x} + \boldsymbol {\beta}) + \lambda \| \mathbf {x} \| _ {1}
$$

Gaussian noise + Tikhonov-type regularisation: often used when $\mathbf A \neq \mathbf I$ for applications 

$$
\operatorname{argmin} _ {\mathbf {x}} \frac {1}{2} \| \mathbf {y} - \mathbf {x} \| _ {2} ^ {2} + \frac {\lambda}{2} \| \mathbf {L x} \| _ {2} ^ {2} \qquad \mathbf {L} \in \mathbb {R} ^ {d \times n}
$$

## Solving the problem: nods on optimisation

argmin J(x) := D(y, x) + λR(x) 

variational formulation of the image denoising problem 

$J : \mathbb { R } ^ { n } \to \mathbb { R } _ { \geq 0 } \cup \{ + \infty \}$ is proper 

$$
\operatorname{dom} (J) = \left\{\mathbf {x}: J (\mathbf {x}) <   + \infty \right\} \neq \varnothing
$$

J is L-smooth, i.e. has L-Lipschitz (Gâteaux) gradient: 

$$
\exists L > 0: \quad \| \nabla J (\mathbf {x} _ {1}) - \nabla J (\mathbf {x} _ {2}) \| _ {2} \leq L \| \mathbf {x} _ {1} - \mathbf {x} _ {2} \| _ {2}
$$

J is convex: 

$$
(\forall \mathbf {x} _ {1}, \mathbf {x} _ {2} \in \mathbb {R} ^ {n}), (\forall \alpha \in [ 0, 1 ]): J (\alpha \mathbf {x} _ {1} + (1 - \alpha) \mathbf {x} _ {2}) \leq \alpha J (\mathbf {x} _ {1}) + (1 - \alpha) J (\mathbf {x} _ {2})
$$

## Solving the problem: nods on optimisation

argmin J(x) := D(y, x) + λR(x) 

variational formulation of the image denoising problem 

$J : \mathbb { R } ^ { n } \to \mathbb { R } _ { \geq 0 } \cup \{ + \infty \}$ is proper 

$$
\operatorname{dom} (J) = \left\{\mathbf {x}: J (\mathbf {x}) <   + \infty \right\} \neq \varnothing
$$

J is L-smooth, i.e. has L-Lipschitz (Gâteaux) gradient: 

$$
\begin{array}{r l} \exists L > 0: & \| \nabla J (\mathbf {x} _ {1}) - \nabla J (\mathbf {x} _ {2}) \| _ {2} \leq L \| \mathbf {x} _ {1} - \mathbf {x} _ {2} \| _ {2} \\ & \Longleftrightarrow \end{array}
$$

$$
J (\mathbf {x} _ {1}) \leq J (\mathbf {x} _ {2}) + \langle \nabla J (\mathbf {x} _ {2}), \mathbf {x} _ {2} - \mathbf {x} _ {1} \rangle + \frac {L}{2} \| \mathbf {x} _ {2} - \mathbf {x} _ {1} \| _ {2} ^ {2}
$$

under convexity 

J is convex: 

$$
(\forall \mathbf {x} _ {1}, \mathbf {x} _ {2} \in \mathbb {R} ^ {n}), (\forall \alpha \in [ 0, 1 ]): J (\alpha \mathbf {x} _ {1} + (1 - \alpha) \mathbf {x} _ {2}) \leq \alpha J (\mathbf {x} _ {1}) + (1 - \alpha) J (\mathbf {x} _ {2})
$$

## Solving the problem: nods on optimisation

argmin $J ( \mathbf { x } ) : = D ( \mathbf { y } , \mathbf { x } ) + \lambda R ( \mathbf { x } )$ 

J is proper, convex, L-smooth and coercive. 

$$
\lim _ {\| \mathbf {x} \| \to + \infty} J (\mathbf {x}) = + \infty
$$

## Theorem

There exists a minimiser for J. All local minimisers are global minimisers. For all $x^{*} \in \operatorname{Argmin}_{x} J(x)$ , there holds $\nabla J(x^{*}) = 0$ . 

## Solving the problem: nods on optimisation

argmin x $J ( \mathbf { x } ) : = D ( \mathbf { y } , \mathbf { x } ) + \lambda R ( \mathbf { x } )$ 

J is proper, convex, L-smooth and coercive. 

$$
\lim _ {\| \mathbf {x} \| \to + \infty} J (\mathbf {x}) = + \infty
$$

## Theorem

There exists a minimiser for . All local minimisers are global minimisers.J For all $\mathbf { x } ^ { * } \in \mathsf { A r g m i n } _ { \mathbf { x } } J ( \mathbf { x } )$ , there holds $\nabla J ( \mathbf { x } ^ { * } ) = \mathbf { 0 } .$ 

Algorithm (gradient descent): for $\mathbf { x } _ { 0 } \in \mathsf { d o m } ( J ) , \tau \in \bigg ( 0 , \frac { 2 } { L } \bigg ) , k \geq 0 \mathrm { : }$ 

$$
\mathbf {x} _ {k + 1} = \mathbf {x} _ {k} - \tau \nabla J (\mathbf {x} _ {k})
$$

## Theorem Theorem

There holds $\mathbf { x } _ { k } \to \mathbf { x } ^ { * }$ and for the function values: $J ( \mathbf { x } _ { k } ) - J ( \mathbf { x } ^ { * } ) \leq \frac { \| \mathbf { x } _ { 0 } - \mathbf { x } ^ { * } \| ^ { 2 } } { 2 \tau k } .$ 

## Example on Tikhonov regularisation

argmin x 

$$
J (\mathbf {x}) := \frac {1}{2} \| \mathbf {y} - \mathbf {x} \| _ {2} ^ {2} + \lambda \| \mathbf {L x} \| _ {2} ^ {2},
$$

$$
\mathbf {L} \in \mathbb {R} ^ {d \times n}
$$

Examples: 

$$
\mathbf {L} \in \left\{\mathbf {I}, \nabla , \nabla^ {2} \right\}
$$

Remark: the problem is quadratic, it can be solved by looking at the optimality condition: 

$$
(\mathbf {x} ^ {*} - \mathbf {y}) + \lambda \mathbf {L} ^ {T} \mathbf {L} \mathbf {x} ^ {*} = \mathbf {0} \quad \Rightarrow \quad (\mathbf {I} + \lambda \mathbf {L} ^ {T} \mathbf {L}) \mathbf {x} ^ {*} = \mathbf {y}
$$

and solving the linear system, e.g., using DFT. Also, faster iterative methods exploiting further regularity (strong convexity, $C ^ { 2 } )$ can be employed. 

Hansen, Nagy, O’leary, ’06, Nesterov, ’83 

## Example on Tikhonov regularisation

argmin x 

$$
J (\mathbf {x}) := \frac {1}{2} \| \mathbf {y} - \mathbf {x} \| _ {2} ^ {2} + \lambda \| \mathbf {L x} \| _ {2} ^ {2},
$$

$$
\mathbf {L} \in \mathbb {R} ^ {d \times n}
$$

Examples: 

$$
\mathbf {L} \in \left\{\mathbf {I}, \nabla , \nabla^ {2} \right\}
$$

Remark: the problem is quadratic, it can be solved by looking at the optimality condition: 

$$
(\mathbf {x} ^ {*} - \mathbf {y}) + \lambda \mathbf {L} ^ {T} \mathbf {L} \mathbf {x} ^ {*} = \mathbf {0} \quad \Rightarrow \quad (\mathbf {I} + \lambda \mathbf {L} ^ {T} \mathbf {L}) \mathbf {x} ^ {*} = \mathbf {y}
$$

and solving the linear system, e.g., using DFT. Also, faster iterative methods exploiting further regularity (strong convexity, $C ^ { 2 } )$ can be employed. 

Hansen, Nagy, O’leary, ’06, Nesterov, ’83 

$$
\nabla J (\mathbf {x}) = (\mathbf {x} - \mathbf {y}) + \lambda \mathbf {L} ^ {T} \mathbf {L} \mathbf {x}, L = 1 + \lambda \| \mathbf {L} ^ {T} \mathbf {L} \| _ {*}, \mathbf {x} _ {0} \in \mathbb {R} ^ {n}, \tau \in (0, 2 / L)
$$

while not converging 

$$
\mathbf {x} _ {k + 1} = \mathbf {x} _ {k} - \boldsymbol {\tau} \left((\mathbf {x} _ {k} - \mathbf {y}) + \lambda \mathbf {L} ^ {T} \mathbf {L} \mathbf {x} _ {k}\right)
$$

end 

## Image denoisers and proximal operators

For general (possibly non-smooth) regularisation functionals $R : \mathbb { R } ^ { n } \to \mathbb { R } _ { \geq 0 } \cup \{ + \infty \}$ , note that: 

$$
\operatorname{argmin} _ {\mathbf {x}} \frac {1}{2} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \lambda R (\mathbf {x}) = \operatorname{prox} _ {\lambda R} (\mathbf {y})
$$

where $\mathsf { p r o x } _ { \lambda R } : \mathbb { R } ^ { n } \Rightarrow \mathbb { R } ^ { n }$ is single-valued if is convex and multi-valued (multiple minimisers)R otherwise. 

## Image denoisers and proximal operators

For general (possibly non-smooth) regularisation functionals $R : \mathbb { R } ^ { n } \to \mathbb { R } _ { \geq 0 } \cup \{ + \infty \}$ , note that: 

$$
\operatorname{argmin} _ {\mathbf {x}} \frac {1}{2} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \lambda R (\mathbf {x}) = \operatorname{prox} _ {\lambda R} (\mathbf {y})
$$

where $\mathsf { p r o x } _ { \lambda R } : \mathbb { R } ^ { n } \Rightarrow \mathbb { R } ^ { n }$ is single-valued if is convex and multi-valued (multiple minimisers)R otherwise. 

## Examples:

$$
\begin{array}{l l l} R (\mathbf {x}) = \iota_ {C} (\mathbf {x}), C \text {is convex and closed.} & \operatorname{prox} _ {\iota_ {C}} (\mathbf {y}) = P _ {C} (\mathbf {y}) & \\ R (\mathbf {x}) = \| \mathbf {x} \| _ {1} & \operatorname{prox} _ {\lambda_ {\| \cdot \| _ {1}}} (\mathbf {y}) = \mathsf {S T} (\mathbf {y}; \lambda) & \text {Non - smooth regularisation functionals} \\ R (\mathbf {x}) = \mathsf {T V} (\mathbf {x}) & \operatorname{prox} _ {\lambda_ {\textit {T V (\cdot)}}} (\mathbf {y})? & \nabla R \text {not defined}. \end{array}
$$

## Image denoisers and proximal operators

For general (possibly non-smooth) regularisation functionals $R : \mathbb { R } ^ { n } \to \mathbb { R } _ { \geq 0 } \cup \{ + \infty \}$ , note that: 

$$
\operatorname{argmin} _ {\mathbf {x}} \frac {1}{2} \| \mathbf {y} - \mathbf {x} \| ^ {2} + \lambda R (\mathbf {x}) = \operatorname{prox} _ {\lambda R} (\mathbf {y})
$$

where $\mathsf { p r o x } _ { \lambda R } : \mathbb { R } ^ { n } \Rightarrow \mathbb { R } ^ { n }$ is single-valued if is convex and multi-valued (multiple minimisers)R otherwise. 

## Examples:

R(x) = ι is convex and closed.(x), C 

$$
\begin{array}{l} {R (\mathbf {x}) = \iota_ {C} (\mathbf {x}),} \\ {R (\mathbf {x}) = \| \mathbf {x} \| _ {1}} \\ {R (\mathbf {x}) = \mathsf {T V} (\mathbf {x})} \end{array}
$$

$$
\begin{array}{r l} & {\mathsf {p r o x} _ {\iota_ {C}} (\mathbf {y}) = P _ {C} (\mathbf {y})} \\ & {\mathsf {p r o x} _ {\lambda \| \cdot \| _ {1}} (\mathbf {y}) = \mathsf {S T} (\mathbf {y}; \lambda)} \\ & {\mathsf {p r o x} _ {\lambda T V (\cdot)} (\mathbf {y})?} \end{array}
$$

Non-smooth regularisation functionals ∇R not defined. 

Proximal operators are widely used as implicit variants of gradients for non-smooth optimisation: 

$$
\mathbf {x} _ {k + 1} = \mathsf {p r o x} _ {\tau \lambda R} \left(\mathbf {x} _ {k} - \tau \nabla_ {\mathbf {x}} D (\mathbf {y}, \mathbf {A x} _ {k})\right)
$$

Gradient-descent on data term + denoising 

This observation stands at the very basis of Plug & Play approaches where prox<sub>τλR</sub> 

## References



J. Kaipio, E. Somersalo, Statistical and computational inverse problems, Springer, 2005. 





Tony F. Chan, J. Shen, Image Processing and Analysis: variational, PDE, wavelet and stochastic methods, SIAM, 2005. 





M. Pragliola, L. Calatroni, A. Lanza, F. Sgallari, On and beyond Total Variation in imaging: the role of space variance, SIAM Review, 65 (3), (2023). 





S. Arridge, P. Maas, O. Öktem, C.B. Schönlieb, Solving inverse problems using data-driven models, Acta Numerica, 2019 





A. Chambolle, T. Pock, An introduction to continuous optimization for imaging, Acta Numerica, 2016 





A. Beck, First-order methods in optimization, Volume 25, MOS-SIAM series on Optimization, 2017. 



## Thanks for your attention!

## luca.calatroni@unige.it

UniGe 

## MaLGa