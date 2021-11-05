"""Python script that demonstrates the Varnet and U-Net model training and reconstruction.

Simulates non-Cartesian MR data, then generates a training dataset and trains a Varnet and U-Net model on it. 
Shows a comparison of the obtained reconstructions.
Requires approximately 10 GB of GPU memory.
"""

import torch
from torchkbnufft import KbNufftAdjoint, KbNufft
import numpy as np
import os
import matplotlib

from net import nufft_common as CT 
from net import nufft_data as ND
from net.nufft_varnet import VNModel
from net.nufft_unet import UNetRekoModel
import math

### Configuration

np.random.seed(31415)
gpu = 0 # Which cuda device to use
cudadev = torch.device("cuda:" + str(gpu))  
root_dir = "./"
dcf_path = root_dir + "data/dcf.mat"
arm_path = root_dir + "data/arm.mat"
dataset_size = 100 # How many data items to generate for training
num_epochs_vn = 25 # How many epochs to train the Varnet model
num_epochs_unet = 25 # How many epochs to train the U-Net model

output_image_path = root_dir + "output/out.png"
data_save_path = root_dir + "output/dataset/"
varnet_chk_dir = root_dir + "output/checkpoints/VNmodel/" 
unet_chk_dir = root_dir + "output/checkpoints/Unetmodel/"

os.makedirs(data_save_path, exist_ok = True)
os.makedirs(varnet_chk_dir, exist_ok = True)
os.makedirs(unet_chk_dir, exist_ok = True)

#### Simulate non-Cartesian MR data, including trajectory, sensitivtity maps and k-space

print("Simualating data")

imsize= 448
num_coils = 30
adjnufft = KbNufftAdjoint(im_size=(imsize,imsize), device = cudadev)
nufft = KbNufft(im_size=(imsize,imsize), device = cudadev)

# Generate spiral trajectory
arm = ND.load_mat_complex(arm_path, "arm")
traj_all = torch.zeros(2, 10*dataset_size, 1408, device = cudadev)
for frame in range(dataset_size):
    frame_angle = math.pi*(3-math.sqrt(5))*frame # do golden angle rotations between real-time frames
    for spoke in range(10):
        traj_all[:,10*frame+spoke,:] = CT.rotate(2*math.pi*spoke /10 + frame_angle, arm).permute(2,0,1).squeeze()*2*math.pi
dcf = ND.load_mat(dcf_path, "dcf").to(cudadev).expand(10,1408)

# Simulate coil sensitivity maps with 2D gaussians. This is not a physically accurate coil sensitivity map simulation, but fine for this demonstration.
smaps = torch.zeros(num_coils,imsize,imsize,2, device = cudadev)
center = torch.tensor((0, 1)).unsqueeze(0)
for coil in range(num_coils):
    center = CT.rotate(2*math.pi/num_coils, center)
    smaps[coil,:,:,:] = CT.gaussian((imsize,imsize),mu=tuple(center[0]*imsize + torch.tensor([imsize/2,imsize/2])), 
                                    sigma = (2*imsize,2*imsize)).unsqueeze(2).repeat(1,1,2)
smaps = smaps / CT.complex_abs(smaps).sum(0).max()

# Generate synthetic k-space data
kspace_all = torch.zeros(num_coils, 10*dataset_size, 1408, 2, device = cudadev)
refs = torch.zeros(dataset_size, imsize,imsize, device = cudadev)
for frame in range(dataset_size):
    traj = traj_all[:,frame*10:10*(frame+1),:]
    image = CT.ball((imsize,imsize), mu = tuple(np.random.randint(0,imsize, 2)), sigma = tuple(np.random.randint(imsize//10,imsize//2, 2)))+\
        CT.cube((imsize,imsize), mu = tuple(np.random.randint(0,imsize, 2)), sigma = tuple(np.random.randint(imsize//10,imsize//2, 2)))
    image[image>1] = 1
    image = image.unsqueeze(2).repeat(1,1,2)*1e-2
    refs[frame] = CT.complex_abs(image) / CT.complex_abs(image).max()
    image_coilwise = CT.expand_op(image.unsqueeze(0).unsqueeze(0).to(cudadev), smaps.unsqueeze(0))
    kspace_all[:,10*frame:10*(frame+1),:,:] = CT.nufft_coilwise(nufft, image_coilwise, traj.unsqueeze(0).to(cudadev), 10, 1408)[0]

# Generate a training dataset of real-time frames with the temporal average as reference
for i in range(dataset_size):
    kspace = kspace_all[:,i*10:(i+1)*10,:,:]
    traj = traj_all[:,i*10:(i+1)*10,:]
    ND.save_datafile(data_save_path + "item" + str(i) + ".mat", kspace, traj, dcf, refs[i], smaps)

### Train models

# Train the Varnet model for X Epochs on the generated training dataset. Uses 3 of the data items as validation set.
hparams = VNModel.get_argument_parser().parse_known_args()[0] # Get default hyperparameters
hparams.val_files = data_save_path + "*.mat§3" # Use the first three items from the dataset as validation set
hparams.train_files = data_save_path + "*.mat+-" + data_save_path + "*.mat§3" # Use the rest as training set, excluding the ones used as val set.
hparams.chk_dir = varnet_chk_dir# Set location for saving checkpoints
hparams.num_epochs = num_epochs_vn # How many epochs to train
hparams.gpu = gpu
hparams.learn_dc_weights = 0
hparams.aug_rotate = 180.0
VNmodel = VNModel.run(hparams).to(cudadev)

# Use tensorboard to view the training process:
#   tensorboard --logdir_spec=label:/PATH/TO/CHEKPOINT/DIRECTORY/lightning_logs/version_X

# Train the U-Net model for X Epochs on the generated training dataset. Uses 3 of the data items as validation set.
hparams = UNetRekoModel.get_argument_parser().parse_known_args()[0] # Get default hyperparameters
hparams.val_files = data_save_path + "*.mat§3" # Use the first three items from the dataset as validation set
hparams.train_files = data_save_path + "*.mat+-" + data_save_path + "*.mat§3" # Use the rest as training set, excluding the ones used as val set.
hparams.chk_dir = unet_chk_dir # Set location for saving checkpoints
hparams.num_epochs = num_epochs_unet # How many epochs to train. Increase
hparams.gpu = gpu
unetmodel = UNetRekoModel.run(hparams).to(cudadev)

### Compare reconstructions

# Take a real-time frame not seen during training
dataitem_path = ND.get_filelist(data_save_path + "*.mat§3")[2]
kspace, traj, dcf, ref, smaps = ND.load_data_file(dataitem_path,towhere = cudadev)

# Compute the naive nuFFT reconstruction
naivereco = CT.sens_reco(kspace, traj, dcf, smaps, adjnufft)[0]

# Compute Varnet reconstruction
VNoutput = VNmodel.propagate(kspace, traj, dcf, smaps)

# Compute U-Net reconstruction
unetoutput = unetmodel.propagate(kspace, traj, dcf, smaps)

# Show reconstructions
image = CT.imshow([naivereco.cpu(), ref.cpu(), VNoutput.cpu(), unetoutput.cpu()], 
                  title= ["Naive", "Ground truth", "Varnet", "U-Net"], 
                  normalize_separately=True, export = True)
matplotlib.image.imsave(output_image_path, image.cpu().numpy(), cmap="gray")
