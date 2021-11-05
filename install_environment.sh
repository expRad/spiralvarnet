#!/bin/bash

# We recommend using a dedicated conda environment
# Create an environment with 
#   conda create --name pytorch17 python=3.7 pip -y
# Then activate with
#   conda activate pytorch17

pip install torch==1.7.1 numpy torchvision pytorch-lightning==1.1 numpy torchkbnufft==1.1.0

conda install matplotlib scipy -y
