"""The main class for the reconstruction U-Net used for comparison with the Varnet model.

Wraps the NormUnet in the Pytorch Lightning framework and contains all the code for training and for using the U-Net for reconstruction. 
Can be called directly from the command line like
python nufft_unet.py --train-files=/PATH/TO/TRAIN/FILES/*.mat --val-files=/PATH/TO/VAL/FILES/*.mat --chk-dir=/PATH/TO/CHECKPOINT/DIR/*.mat
"""

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from pytorch_lightning.callbacks import LearningRateMonitor
from argparse import Namespace
import torch.backends.cudnn as cudnn
import argparse
from torchkbnufft import KbNufftAdjoint
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from . import nufft_common as CT
from . import nufft_data as ND
from .losses import SSIM
from .unet_model import NormUnet

class UNetRekoModel(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
         super().__init__()
         self.save_hyperparameters(hparams)
         if type(hparams) == dict:
             hparams = Namespace(**hparams)
         self.net = NormUnet(hparams.chans, hparams.pools)
         self.step = 0
         self.ssim_loss = SSIM()
         self.mse_loss = nn.MSELoss()
         self.cudadev = torch.device('cuda:' + str(self.hparams.gpu))
         self.adjnufft = KbNufftAdjoint(im_size=(448,448), numpoints=6, device=self.cudadev)
         return
         
    def forward(self, kspace, traj, dcf, smaps):
        zfreco = CT.sens_reco(kspace, traj, dcf, smaps, self.adjnufft).unsqueeze(0)
        # unet expects x as : batch x coils(= channels) x ximsize x yimsize x 2(real/imag)
        out = self.net(zfreco).squeeze(1)
        if self.hparams.out_norm:
            return out / CT.complex_abs(out).max()
        else:
            return out
        
    def propagate(self, kspace, traj, dcf, smaps):
        return self.forward(kspace, traj, dcf, smaps).detach()[0]

    def training_step(self, batch, batch_idx):
        kspace, traj, dcf, target, smaps, identifier  = batch
        out = self(kspace, traj, dcf, smaps)
                       
        loss = self.ssim_loss(out, target) #  takes abs of out automatically
        mseloss = self.mse_loss(CT.complex_abs(out), target)

        self.log("train_loss",loss,  on_step=True, on_epoch=True)
        self.log("train_loss",loss)
        self.log("trainlosses/ssim_train_loss",loss,  on_step=True, on_epoch=True)
        self.log("trainlosses/mse_train_loss",mseloss,  on_step=True, on_epoch=True)
        
        self.step = self.step+1
        return loss
    
    def validation_step(self, batch, batch_idx):
        kspace, traj, dcf, target, smaps, identifier  = batch
        out = self(kspace, traj, dcf, smaps)

        valloss = self.ssim_loss(out, target)
        msevalloss = self.mse_loss(CT.complex_abs(out), target)
        
        self.log("val_loss",valloss,  on_step=True, on_epoch=True)
        self.log("vallosses/ssim_train_loss",valloss,  on_step=True, on_epoch=True)
        self.log("vallosses/mse_train_loss",msevalloss,  on_step=True, on_epoch=True)
        
        if batch_idx == 0:
            errim = CT.plot_error_map(CT.complex_abs(out).squeeze(), target.squeeze())
            zfreco = CT.sens_reco(kspace, traj, dcf, smaps, self.adjnufft).unsqueeze(0)
            self.logger.experiment.add_image(identifier[0]+"/in", CT.imconvert(CT.complex_abs(zfreco).squeeze()), self.step)
            self.logger.experiment.add_image(identifier[0]+"/out", CT.imconvert(CT.complex_abs(out).squeeze()), self.step)
            self.logger.experiment.add_image(identifier[0]+"/ref", CT.imconvert(target.squeeze()), self.step)
            self.logger.experiment.add_image(identifier[0]+"/error", CT.imconvert(errim), self.step)
        
        return valloss
     
    def configure_optimizers(self):
        optimizer  = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {"optimizer" : optimizer}

    @staticmethod
    def get_argument_parser():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--bs', type=int, default=1, help='Batch size')
        parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
        parser.add_argument('--seed', type=int, default=31415, help='Torch random seed')
        parser.add_argument('--save_top_k', type=int, default=1)
        parser.add_argument('--gpu', type=int, default=0)
        parser.add_argument('--log_every_n_steps', type=int, default=100)
        parser.add_argument('--checkpoint_filename', type=str, default='reco_unet-{epoch:02d}-{train_loss:.2f}')
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--norm-targets', type=float, default=1, help="If this is any float > 0, then all the targets will be normalized to have values between 0 and NORM_TARGETS.")
        parser.add_argument('--pools', type=int, default=5,help='Number of U-Net pooling layers')
        parser.add_argument('--chans', type=int, default=30,help='Number of U-Net channels')
        parser.add_argument('--aug-rotate', type=float, default=180.0, help="Data augmentation rotation. If this is set to any value not equal 0, rotates the input trajectories together with the target images and sensitivity maps randomly by + - angle (in degrees)")
        parser.add_argument('--out-norm', type=int, choices=[0,1],  default=1)
        parser.add_argument('--num-epochs', type=int, default=10,help='Number of training epochs')

        parser.add_argument('--chk-dir', type=str, default="./path/for/checkpoints", help='Where to save logs') 
        parser.add_argument('--val-files', type=str, default="./path/to/val/dataset/*.mat")
        parser.add_argument('--train-files', type=str, default="./path/to/train/dataset/*.mat")
    
        return parser
    
    @staticmethod
    def build_trainer_loader(hparams):
        torch.manual_seed(hparams.seed)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              filename=hparams.checkpoint_filename,
                                              save_top_k=hparams.save_top_k)
        trainer = pl.Trainer(default_root_dir=hparams.chk_dir, callbacks=[checkpoint_callback, lr_monitor], gpus=str(hparams.gpu), 
                              log_every_n_steps=hparams.log_every_n_steps, max_epochs=hparams.num_epochs)
        return trainer
    
    def train_dataloader(self):
        return UNetRekoModel.create_dataloader(self.hparams, self.hparams.train_files)

    def val_dataloader(self):
        return UNetRekoModel.create_dataloader(self.hparams, self.hparams.val_files)
    
    @staticmethod
    def create_dataloader(hparams, data_files):
        bs = hparams.bs
        preci = torch.float32
        dset = ND.SpiralItemData(data_files=data_files,dtype=preci, seed=hparams.seed, 
                                      aug_rotate=hparams.aug_rotate, norm_targets=hparams.norm_targets, 
                                      ref_mask="none")
        return DataLoader(dataset=dset,batch_size=bs,num_workers=hparams.num_workers);
    @staticmethod

    def run(args):
        cudnn.benchmark = True
        cudnn.enabled = True
        if args.val_files == "train":
            args.val_files = args.train_files
        else:
            args.train_files = args.train_files + "+-" + args.val_files
        
        model = UNetRekoModel(args)
    
        print("Using arguments: ")
        print(model.hparams)
    
        trainer = UNetRekoModel.build_trainer_loader(args)       
        
        trainer.fit(model)    
        print("Training completed")
    
        return model

def main(args=None):
    parser = UNetRekoModel.get_argument_parser()
    if args is not None:
        parser.aults(**args)
        
    args, _ = parser.parse_known_args()
    UNetRekoModel.run(args)    
if __name__ == '__main__':
    main()