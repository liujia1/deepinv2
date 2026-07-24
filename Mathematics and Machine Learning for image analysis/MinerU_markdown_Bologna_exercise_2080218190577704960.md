For the assignment you should submit the following: Codes for training and evaluation. And a short summary (1-2 pages) of results as outlined in the tasks. 

## Task 1: Implement Learned Primal Dual:

• Start from the provided Learned Gradient Scheme codes (LGS) 

• Extend the class function for LPD to include both networks for primal and dual. Remember that these are diferent networks. 

• Report your results on the test data for the three networks U-Net, LGS, LPD. Report PSNR in table for each method and compare the reconstructed images qualitatively, that means show a reconstruction in the report and discuss the visual: Did the network do a good job, or are there residual artefacts? Note that you should aim for similar performance of all methods. 

• Report also the amount of parameters and choices you made for LPD and LGS in terms of unrolled iterates. Did you need to change the training parameters? 

Potential dificulties: Astra kernel may crash after many iterations. It is advised to save intermediate iterates every few thousand training iterations. 

## Task 2: Noise2Inverse with U-Net

• Modify the supervised U-Net training into an unsupervised (self supervised) setting using Noise2Inverse as follows. 

• Create two disjoint subsets of the sinogram and corresponding FBP reconstructions. You can either pre-compute these or do it ”on-the-fly” within the training loop. 

• Use the two reconstructions as training pairs. 

• Report again on the performance of Noise2Inverse compared to supervised training in terms of PSNR (table) and qualitative appearance (images). 