### INTRODUCTION

This repository is accompanying the publication 

Real-time cardiac MRI using an undersampled spiral k-space trajectory and a reconstruction based on a Variational Network
by 
Jonas Kleineisel, Julius F. Heidenreich, Philipp Eirich, Nils Petri, Herbert Köstler, Bernhard Petritsch, Thorsten A. Bley, Tobias Wech
in Magnetic Resonance in Medicine 2022

which introduces a new reconstruction method of undersampled spiral MR data by means of a Variational Network (VN).


### CONTENT

It contains the full Python & Pytorch code for the VN and U-Net model as described in the publication. 

Under ./data, the shape of the spiral read-outs and a corresponding density compensation function can be found in the MATLAB-format ".mat". 

The demonstration script demo.py gives an example of how to use the code. It generates a synthetic dataset of undersampled data, then trains both network models on it and uses the obtained network models to reconstruct undersampled synthetic data.


### INSTALLATION

The code requires python 3.7 with the following packages:

torch==1.7.1 numpy torchvision pytorch-lightning==1.1 numpy torchkbnufft==1.1.0 matplotlib scipy

We recommend using Anaconda to create a virtual environment and install these packages in it. An example of how to do this can be found in ./install_environment.sh


### REFERENCES

Part of the code, in particular the U-Net, is modified from the fastMRI repository, available at https://github.com/facebookresearch/fastMRI. It is released under the MIT license.

The code for the openadapt-algorithm for computing coil sensitivity maps is adapted from https://github.com/andyschwarzl/gpuNUFFT/blob/master/matlab/demo/utils/adapt_array_2d.m, where it is released under the MIT license.
The procedure is originally described in
D. O. Walsh, A. F. Gmitro, and M. W. Marcellin, “Adaptive reconstruction of phased array MR imagery,” Magnetic Resonance in Medicine, vol. 43, no. 5, pp. 682–690, May 2000. doi:10.1002/(sici)1522-2594(200005)43:5h682::aid-mrm10i3.0.co;2-g
and 
Mark Griswold, David Walsh, Robin Heidemann, Axel Haase, Peter Jakob, "The Use of an Adaptive Reconstruction for Array Coil Sensitivity Mapping and Intensity Normalization", Proceedings of the Tenth Scientific Meeting of the International Society for Magnetic Resonance in Medicine pg 2410 (2002)


### LICENSE

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
