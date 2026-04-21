# Seminar_DiStefano_10_June_2025.pdf

## 第1页

Trends in Computer Vision Research 
and Applications
Luigi Di Stefano (luigi.distefano@unibo.it) 
Department of Computer Science and Engineering (DISI)
University of Bologna, Italy


---

## 第2页

Key Process Technology in Manufacturing
Food&Bevarage
Electronics
Textile
Automotive 
Pharma & Tobacco
Transportation & Logistics
Cognex DataMan 
(1982)
Inspection, Gauging, 
Guidance, Tracing, 
Compliance
Whole Process Optimization 


---

## 第3页

Mass-Market Consumer Products  


---

## 第4页

Deep Learning (aka AI)
2012
A Krizhevsky, I Sutskever, and GE Hinton, 
Imagenet classification with deep 
convolutional neural networks. NIPS 2012. 
Karen Simonyan and Andrew 
Zisserman, Very Deep Convolutional 
Networks for Large-scale Image 
Recognition, ICLR 2015
Alexey Dosovitskiy et. at., An image is 
worth 16x16 words: Transformers for 
image recognition at scale. ICLR 2020. 
2020
Deep learning is 
representation learning !


---

## 第5页

Impact of computer vision research 


---

## 第6页

Novel-View Synthesis (Radiance Field) 
https://lumalabs.ai/  
NeRF, 2020
3D Gaussian Splatting, 2023 


---

## 第7页

More Generative Tasks  
A photorealistic image of a panda wearing 
the jersey of Bologna FC and teaching to a 
group of very interested frogs 
DALL⸱E 3 (Bing)
A panda wearing the jersey 
of Bologna FC
Genie (Luma AI)
DDPM aka Diffusion Models, 2020 
Latent Diffusion Models, 
2022  (Stable Diffusion) 


---

## 第8页

MLLMs (Multimodal Large Language Models) 
LLaVA: Large 
Language and 
Vision Assistant
Timeline of MLLMs
(https://arxiv.org/pdf/2401.13601) 


---

## 第9页

Deep Learning (aka AI) ? 
Food&Bevarage
Electronics
Textile
Automotive 
Pharma & Tobacco
Transportation & Logistics
Cognex DataMan 
(1982)
Inspection, Gauging, 
Guidance, Tracing, 
Compliance
Whole Process Optimization 
It’s happening


---

## 第10页

Industrial Anomaly Detection (IAD)
Nominal Samples
Anomalous Samples
Anomaly Segmentation
Good (nominal) samples are abundant and easy to collect, whereas defects are rare 
and (often) unpredictable
Learn to model good samples only ! 
MVTec AD 
Dataset [1] 


---

## 第11页

11 |
Learning to reconstruct good samples (1)
Input Image
AutoEncoder Network  
Encoder-Decoder architecture trained to reconstruct the input images 
Reconstructed Image
E
D


---

## 第12页

12 |
Learning to reconstruct good samples (2)
Input Image
E
D
Reconstructed Image
The models learns to reconstruct good samples only.
Hence, when provided with an anomalous sample, the
reconstructed image looks different from the input one.


---

## 第13页

13 |
Learning to reconstruct good samples (3)
Input Image
Reconstructed Image
-
Anomaly Map
The difference
between the input 
and reconstructed
image highlights  
anomalies


---

## 第14页

14 |
State-of-the-art Anomaly Detection: PatchCore [2]


---

## 第15页

15 |
Self-Supervised Learning and Vision Foundation Models
•
Vision Foundation Models
(VFM) are image encoders 
trained on huge unlabelled
datasets by self-
supervised learning. 
•
VFM provide high-quality
and general-purpose
features to be used
without any further
training for a variety of 
diverse downstream tasks.
•
Prominent VFM are DINO 
[3,4] and MAE (left).
MAE – Masked AutoEncoder [5]


---

## 第16页

16 |
Industrial Anomaly Detection goes Multimodal (RGB+3D) 
MVTec 3D-AD Dataset [6]  
Eyecandies Dataset [7]  


---

## 第17页

17 |
Crossmodal Feature Mapping [8]
•
Two lightweight MLPs are trained solely with 
nominal samples so to predict foundational
features across the two modalities. 
•
At inference time, the differences between
predicted and actual fatures are aggregated
into the final anomaly map. 


---

## 第18页

18 |
Quantitative results on MVTec 3D-AD 


---

## 第19页

19 |
Quantitative results on MVTec 3D-AD 


---

## 第20页

20 |
Qualitative results on MVTec 3D-AD


---

## 第21页

Computer vision for personalized maxillofacial surgery
DISI
DIBINEM
IRCCS Sant’Orsola
DIMEC
HAMLET - Head and face Automated reconstruction 
via Machine LEarning Techniques


---

## 第22页

Neural Shape Completion for Surgical Planning [10,11]
Encoder-Decoder  Network 
How to train the neural
network ?


---

## 第23页

Dataset and Self-Supervised Training
We simulate malformations  
by randomly removing  
regions of different 
positions and sizes from the 
CT scans of CQ500. 
CQ500:  385 CT scans 
of eumorphic subjects. 


---

## 第24页

Pre-Operative 
Reconstructed
Surgeon Prompt
Ground Truth 
Reconstructed
Reconstructed
Ground Truth 
Mean Accuracy 2 mm
Experimental Results


---

## 第25页

Fields
A field is a function 
defined for all spatial  
coordinates
Signed Distance Field (SDF)
ℝ3 →ℝ
 Implicit Surface  
RGB Intensity Field
ℝ2 →ℝ3
Image
Magnetic Field
ℝ2 →ℝ2
And Neural Fields
𝑥∈ℝ𝑛
𝑓𝑥∈ℝ𝑚
A neural field is a field 
parameterized by a 
neural network


---

## 第26页

Neural Fields for Images and 3D Objects
(𝑥, 𝑦)
(𝑅, 𝐺, 𝐵)
(𝑥, 𝑦, 𝑧)
𝑠𝑑𝑓(𝑥, 𝑦, 𝑧)


---

## 第27页

Neural Radiance Fields (NeRFs) [12]
Given N (e.g. 100+) posed images 
So as to synthesize
novel images from any
viewing direction
(volumeric rendering)
(𝑥, 𝑦, 𝑧, 𝜃, 𝜙)
(𝑅, 𝐺, 𝐵, 𝜎)
Train a neural network (NeRF) to 
predict the directional radiance
and density of points in space


---

## 第28页

A NeRF is just an MLP that is queried with the 3D coordinates of a 
point in space and the viewing direction of the camera to predict
a  color and a density value.
How it works ? Volumetric Rendering
To render an image, we sample 3D points along rays passing through the camera center and 
pixels. Once color and density are predicted, they are accumulated along the rays to obtain pixel 
colors. At training time, the MLP is optimized by a photometric loss. 


---

## 第29页

Key Advantages
• A Neural Field is a highly compressed 
and continuous representation which 
disentangles memory cost and spatial 
resolution. Indeed, with a fixed and 
small number of parameters one can 
output a 3D surface or image  at any 
arbitrarily fine resolution.
• This representation is applicable to many signals of interest. 


---

## 第30页

NFs are a novel representation for 3D Objects….
Triangle
Mesh
…. can we design NNs capable of processing NFs ? 
Voxel
Grid
Multiple
Images  
(NeRF)
Point 
Cloud


---

## 第31页

nf2vec [13,14] 
nf2vec:
embeds NFs’ weights
into compact latent vectors
nf2vec latent vectors
can be input to standard NNs
to tackle downstream tasks
<
NF0
<
NF2
<
NF3
<
NF4
PART SEGMENTATION
UNCONDITIONED 
GENERATION
SURFACE
RECONSTRUCTION
CLASSIFICATION
RETRIEVAL
COMPLETION
<
NF1
NF


---

## 第32页

nf2vec encoder
embeddin
g
hidden layers
linear transforms:
•
Weights 
•
Biases
Stack of weights and biases
Max pooling
shared
shared
shared
shared
ENCODER
LINEAR
+
BATCH NORM
+
ReLU
=
input layer:
•
Weights 
•
Biases
output layer:
•
Weights 
•
Biases
PAD


---

## 第33页

Training and Inference


---

## 第34页

Is it a good latent space ?  
0
1
Interpolation factor
POINTS 
from UDF
MESH from 
SDF
VOXELS 
from OF
IMAGES 
from NeRF


---

## 第35页

Is it a good latent space ?
Encoder
Nf2vec Dataset 
Embeddings
Unseen 
NF
KNN Search
Retrieval with NeRFs


---

## 第36页

Neural Processing of Neural Fields
NF
Standard MLP 
Lamp
NeRF Classification
Point Cloud Segmentation
NF Classification
Generation (Point Clouds, Meshes, NeRFs) by Latent-GAN


---

## 第37页

Learning Mappings between Latent Spaces
Shape Completion
Point Cloud to Mesh
Point Cloud to NeRF


---

## 第38页

LLaNA: Large Language and NeRF Assistant [15]


---

## 第39页

LLaNA: Captioning and Question Answering


---

## 第40页

LLaNA vs. MLLMs that can ingest renders from NeRfs 


---

## 第41页

Thanks a lot 
for your 
attention ! 
41


---

## 第42页

References (1)
[1] Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. Mvtec ad – a comprehensive real-world dataset for 
unsupervised anomaly detection. In Proceedings of CVPR 2019. 
[2] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Scholkopf, Thomas Brox, and Peter Gehler. Towards total recall
in industrial anomaly detection. In Proceedings of CVPR 2022. 
[3] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jegou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging
properties in self-supervised vision transformers. In Proceedings of ICCV 2021.
[4] Maxime Oquab, Timothee Darcet, Theo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel 
Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. 
https://arxiv.org/abs/2304.07193 2023.
[5] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable
vision learners. In Proceedings of  CVPR 2022. 
[6] Paul Bergmann, Jin Xin, David Sattlegger, and Carsten Steger. The mvtec 3d-ad dataset for unsupervised 3d anomaly
detection and localization. In Proceedings of the 17th International Joint Conference on Computer Vision, Imaging and
Computer Graphics Theory and Applications, 2022.
[7] Luca Bonfiglioli, Marco Toschi, Davide Silvestri, Nicola Fioraio, and Daniele De Gregorio. The eyecandies dataset
for unsupervised multimodal anomaly detection and localization. In Proceedings ACCV 2022.
[8] Alex Costanzino, Pierluigi Zama Ramirez, Giuseppe Lisanti, and Luigi Di Stefano. Multimodal Industrial Anomaly Detection 
by Crossmodal Feature Mapping. In Proceedings of  CVPR 2024.


---

## 第43页

References (2)
[9] Chengjie Wang, Wenbing Zhu, Bin-Bin Gao, Zhenye Gan, Jiangning Zhang, Zhihao Gu, Shuguang Qian, Mingang Chen, and 
Lizhuang Ma. Real-IAD: A Real-World Multi-View Dataset for Benchmarking Versatile Industrial Anomaly Detection. In 
Proceedings of CVPR 2024 
[10] Stefano Mazzocchetti, Riccardo Spezialetti, Mirko Bevini, Giovanni Badiali, Giuseppe Lisanti, Samuele Salti, and  Luigi Di 
Stefano. Neural shape completion for personalized Maxillofacial surgery. Scientific Reports (Springer Nature), August 2024. 
[11] Stefano Mazzocchetti, Mirko Bevini, Giovanni Badiali, Giuseppe Lisanti, Luigi Di Stefano, and Samuele Salti, Automatic 
Implant Generation for Cranioplasty via Occupancy Networks. IEEE Access, July 2024. 
[12] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: 
Representing Scenes as Neural Radiance Fields for View Synthesis. In Proceedings of ECCV 2020.
[13] Luca De Luigi, Adriano Cardace, Riccardo Spezialetti, Pierluigi Zama Ramirez, Samuele Salti, and Luigi Di Stefano. Deep 
learning on implicit neural representations of shapes. In Proceedings of ICLR 2024 
[14 Pierluigi Zama Ramirez, Luca De Luigi, Daniele Sirocchi, Adriano Cardace, Riccardo Spezialetti, Francesco Ballerini, 
Samuele Salti, and Luigi Di Stefano. Deep Learning on Object-centric 3D Neural Fields. IEEE Trans. On Pattern Analysis and 
Machine Intelligence (TPAMI), December 2024.
[15] Andrea Amaduzzi, Pierluigi Zama Ramirez, Giuseppe Lisanti, Samuele Salti, and Luigi Di Stefano. LLaNA: Large 
Language and NeRF Assistant. Advances in Neural Information Processing Systems 37, NeurIPS 2024.


---

