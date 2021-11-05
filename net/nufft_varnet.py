"""The main class of the proposed Varnet model.

Implements the Varnet model in the Pytorch Lightning framework and contains all the code for training and for using it for reconstruction. 
Can be called directly from the command line like
python nufft_vernat.py --train-files=/PATH/TO/TRAIN/FILES/*.mat --val-files=/PATH/TO/VAL/FILES/*.mat --chk-dir=/PATH/TO/CHECKPOINT/DIR/*.mat
"""

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from torch import nn
from torchkbnufft import KbNufft, KbNufftAdjoint
import pytorch_lightning as pl
import argparse
from pytorch_lightning.utilities.parsing import AttributeDict
from torch.utils.data import DataLoader

from . import nufft_common as CT
from . import nufft_data as ND
from .losses import SSIM
from .unet_model import NormUnet

class VarNetBlock(nn.Module):

    def __init__(self, unet, nufft, adjnufft, hparams, cudadev):
        super(VarNetBlock, self).__init__()
        self.hparams = hparams
        self.unet = unet
        self.nufft = nufft
        self.adjnufft = adjnufft
        if self.hparams.learn_dc_weights == 1:
            self.dc_weight = nn.Parameter(torch.ones(1).to(cudadev)*self.hparams.dc_weight_init)
        else:
            self.dc_weight = torch.ones(1).to(cudadev)*self.hparams.dc_weight_init       
    def forward(self, current_imspace, traj, kspace0, dcf, smaps):
        # expects as:   current_imspace     batch x ximsize x yinmsize x 2(real/imag)               float32 tensor
        #               smaps               batch x coils x ximsize x yinmsize x 2(real/imag)    float32 tensor
        #               traj                batch x 2(xydim) x nspokes x spokelength                float32 tensor
        #               dcf                 batch x nspokes x spokelength                     float32 tensor
        batch, coils, numspokes, spokelength, two = kspace0.shape
        coil_expanded = CT.expand_op(current_imspace.unsqueeze(1), smaps)
        kspace_mod = CT.nufft_coilwise(self.nufft, coil_expanded, traj, numspokes, spokelength)
        residuum = CT.adjnufft_coilwise(self.adjnufft, kspace_mod - kspace0, traj, dcf)
        residuum = CT.reduce_op(residuum, smaps)*self.dc_weight
        regularization = self.hparams.unet_scale*self.unet(current_imspace.unsqueeze(1)).squeeze(1)
        passon = current_imspace - residuum + regularization
        return passon, residuum, regularization

class VNModel(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        assert torch.__version__.startswith("1.7"), "Wrong Pytorch version. Needs 1.7"
        self.save_hyperparameters(hparams)
                 
        self.im_size = (self.hparams.ximsize, self.hparams.yimsize)

        self.cudadev = torch.device('cuda:' + str(self.hparams.gpu))
        self.nufft = KbNufft(im_size=self.im_size, numpoints=6, device=self.cudadev)
        self.adjnufft = KbNufftAdjoint(im_size=self.im_size, numpoints=6, device=self.cudadev)
        
        self.cascades = nn.ModuleList([VarNetBlock(NormUnet(self.hparams.chans, self.hparams.pools), self.nufft, 
                                                             self.adjnufft, self.hparams, 
                                                             self.cudadev) for _ in range(self.hparams.num_cascades)])
        self.ssim_loss = SSIM()
        self.mse_loss = nn.MSELoss()
        self.step_counter = 1;
            
    def forward(self,  kspace, traj, dcf, smaps):
        # expected shapes:  kspace         batch x coils x numspokes x spokelength x 2(real/complex)  float tensor
        #                   traj           batch x 2(xydim) x numspokes x spokelength float tensor              
        #                   dcf            batch x numspokes x spokelength  float tensor 
        #                   smaps          batch x ximsize x yimsize x 2(real/imag) float tensor 
       
        batchsize, coils, numspokes, spokelength, two = kspace.shape
        
        kspace0 = kspace        
        i = 0
        current_imspace = CT.sens_reco(kspace, traj, dcf, smaps, self.adjnufft)
        for cascade in self.cascades:
            current_imspace, imspace_after_unet, residuum = cascade(current_imspace, traj, kspace0, dcf, smaps)
            i = i+1
        return current_imspace
        
    def propagate(self, kspace, traj, dcf, smaps, norm=True, absolute=False):
        kspace, traj, dcf, smaps = CT.add_batchdim(kspace, traj, dcf, smaps)
        output = self.forward(kspace, traj, dcf, smaps)[0]
        output = output.detach()
        if norm:
            output = output / CT.complex_abs(output).max()
        if absolute:
            output = CT.complex_abs(output)
        return output
     
    def training_step(self, batch, batch_idx):
        self.step_counter+=1
        kspace, traj, dcf, target, smaps, _  = batch
        output = self.forward(kspace, traj, dcf, smaps)
        output = output / CT.complex_abs(output).max()
        ssim_loss = self.ssim_loss(output, target)
        mse_loss = self.mse_loss(CT.complex_abs(output), target)
        self.log('0train_loss/ssim_train_loss', ssim_loss, on_step=True, on_epoch=True)
        self.log('0train_loss/mse_train_loss', mse_loss, on_step=True, on_epoch=True)
        if self.hparams.loss == "mse":
            train_loss = mse_loss
        elif self.hparams.loss == "ssim":
            train_loss = ssim_loss
        return train_loss
   
    def _normalize(self, image):
        image = image[np.newaxis]
        image -= image.min()
        return image / image.max()
    
    def validation_step(self, batch, batch_idx):
        kspace, traj, dcf, target, smaps, identifier = batch
        output = self.forward(kspace, traj, dcf, smaps)         # output has shape    batch x xsize x ysize x 2 (real/complex)
        output = output / CT.complex_abs(output).max() # normalize output by maximum of abs
        out_image = CT.complex_abs(output) # shape    batch x xsize x ysize
        ssim_loss = self.ssim_loss(out_image, target)
        mse_loss = self.mse_loss(out_image, target)
        
        if batch_idx < self.hparams.show_batches:
            out_imdata = torch.zeros(3)
            out_imdata[0] =  torch.mean(torch.max(torch.max(out_image,1).values, 1).values)
            out_imdata[1] =  torch.mean(torch.min(torch.min(out_image,1).values, 1).values)
            out_imdata[2] =  torch.mean(out_image)
            tgt_imdata = torch.zeros(3)
            tgt_imdata[0] =  torch.mean(torch.max(torch.max(target,1).values, 1).values)
            tgt_imdata[1] =  torch.mean(torch.min(torch.min(target,1).values, 1).values)
            tgt_imdata[2] =  torch.mean(target)

            self.log('1val_loss/ssim_val_loss', ssim_loss, on_step=False, on_epoch=True)
            self.log('1val_loss/mse_val_loss', mse_loss, on_step=False, on_epoch=True)
            
            return  {'output': output,'target': target,'ssim_val_loss': ssim_loss, 
                 'mse_val_loss': mse_loss, 'ID' : identifier, 'out_imdata' : out_imdata,
                 'tgt_imdata' : tgt_imdata, 'kspace' : kspace, 'traj' : traj, 'dcf' : dcf}
        else:
            return  {'ssim_val_loss': ssim_loss, 'mse_val_loss': mse_loss}

    def validation_epoch_end(self, val_logs):
        mse_val_loss_mean = torch.stack([x['mse_val_loss'] for x in val_logs]).mean()
        ssim_val_loss_mean = torch.stack([x['ssim_val_loss'] for x in val_logs]).mean()

        if self.hparams.loss == "mse":
            val_loss = mse_val_loss_mean
        elif self.hparams.loss == "ssim":
            val_loss = ssim_val_loss_mean
        
        b = 1
        for  varnetblock in self.cascades:
            self.log('VNModel/dc_weight_cascade ' + str(b), varnetblock.dc_weight)
            b += 1
        
        # from here on only for the shown ones
        show_logs = []
        for x in val_logs:
            if "output" in x:
                show_logs.append(x)

        # add pictures
        num_items_per_batch = show_logs[0]['output'].shape[0]
        items_to_show = range(num_items_per_batch)       
        for v in show_logs:
            for item_idx in items_to_show:
                identifier = v['ID'][item_idx];
                output = v['output'][item_idx,:,:,:] # why tensor??
                target = v['target'][item_idx,:,:]
                self.add_item_images(identifier, output, target)
        return {"val_loss" : val_loss}
            
    def add_item_images(self, identifier, output, target):
        out_image = CT.complex_abs(output)
        self.logger.experiment.add_image(identifier +"/1output", self._normalize(out_image), self.step_counter)
        self.logger.experiment.add_image(identifier +"/2target", self._normalize(target), self.step_counter)
        emapplot = CT.plot_error_map(out_image, target)
        self.logger.experiment.add_image(identifier +"/3error_map", emapplot, self.step_counter)        
        
    def train_dataloader(self):
        return self.create_dataloader(self.hparams.train_files, self.hparams.aug_rotate)

    def val_dataloader(self):
        return self.create_dataloader(self.hparams.val_files, self.hparams.aug_rotate)

    def create_dataloader(self, files_path, aug_rotate=0.0):
        bs = self.hparams.batch_size
        dset = ND.SpiralItemData(data_files=files_path, seed=self.hparams.seed, 
                                 aug_rotate=aug_rotate, norm_targets=self.hparams.norm_targets)
        return DataLoader(dataset=dset,batch_size=bs,pin_memory=False,num_workers=self.hparams.num_workers);
    
    def configure_optimizers(self):
        self.optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return {'optimizer' : self.optim, 'monitor': 'val_loss'} # pytorch 1.7
    
    def checkpoint_load(checkpointpath, towhichgpu, args=None):
        print('Loading weights from ' + str(checkpointpath).split("/")[-1]) 

        checkpoint = torch.load(checkpointpath)
        old_params  = AttributeDict(checkpoint['hyper_parameters'])

        if args is None:
            args = VNModel.get_argument_parser().parse_known_args()[0]
    
        old_params.gpu = towhichgpu

        # add any parameters that may be in args but not in old_params 
        old_params = dict(old_params)
        for key in vars(args):
            if not key in old_params.keys():
                old_params[key] = vars(args)[key]
        
        toload_params = AttributeDict(old_params)
        model = VNModel.load_from_checkpoint(checkpoint_path=str(checkpointpath), map_location="cuda:" + \
                                                  str(towhichgpu), **toload_params) 
        return model.to(model.cudadev)
    
    @staticmethod
    def get_argument_parser():
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--seed', type=int, default=731, help='Seed for any kind of random process used in the net')
        parser.add_argument('--loss', type=str, default="ssim", choices=["mse", "ssim"], help='Wether to use MSE or SSIM loss for training.')
        parser.add_argument('--num-workers', type=int, default=16, help='Number of workers for data loading')
        parser.add_argument('--num-epochs', type=int, default=10,help='Number of training epochs')
        parser.add_argument('--gpu', type=int, default=0, help='Which gpu to use. Set as 0 for GPU0 and 1 for GPU1 etc')
        parser.add_argument('--num-cascades', type=int, default=10, help='Number of cascades')
        parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for the ADAM optimizer.')    
        parser.add_argument('--pools', type=int, default=4,help='Number of U-Net pooling layers') 
        parser.add_argument('--chans', type=int, default=18,help='Number of U-Net channels')
        parser.add_argument('--batch-size', default=1, type=int, help='Batch size')
        parser.add_argument('--weight-decay', type=float, default=0., help='Strength of weight decay regularization for Adam optimizer')   
        parser.add_argument('--show-batches', type=int, default=3, help='How many validation batches should be shown in tensorboard')
        parser.add_argument('--aug-rotate', type=float, default=0.0, help="Data augmentation rotation. If this is set to any value not equal 0, rotates the input trajectories together with the target images and sensitivity maps randomly by + - angle (in degrees)")
        parser.add_argument('--ximsize', type=int, default=448, help="Image size in dimension x. Assumes all images have this size.")
        parser.add_argument('--yimsize', type=int, default=448, help="Image size in dimension y. Assumes all images have this size.")
        parser.add_argument('--norm-targets', type=float, default=1, help="If this is any float > 0, then all the targets will be normalized to have values between 0 and NORM_TARGETS.")
        parser.add_argument('--train-log-interval', type=int, default=50, help="Save logs during the epoch after each TRAIN_LOG_INTEVAL steps. This is also used duing the val epoch.")
        parser.add_argument('--unet-scale', type=float, default=0.08, help="Factor in front of the Unet(s)")
        parser.add_argument('--dc-weight-init', type=float, default=1, help="Initial value for the DC weight")
        parser.add_argument('--save-top-checkpoints', type=int, default=1, help="Saves this number of the best checkpoints, based on val loss.")
        parser.add_argument('--learn-dc-weights', type=int, default=0, choices=[0,1], help="Wether to learn DC weights. 1 to learn and 0 for not learning.")
        

        parser.add_argument('--chk-dir', type=str, default="./path/for/checkpoints", help='Where to save logs') 
        parser.add_argument('--val-files', type=str, default="./path/to/val/dataset/*.mat")
        parser.add_argument('--train-files', type=str, default="./path/to/train/dataset/*.mat")
        
        return parser
    
    @staticmethod
    def create_trainer(args):
        kwargs = {'default_root_dir' : args.chk_dir,'max_epochs' : args.num_epochs, 
                  'gpus' : str(args.gpu), 'weights_summary' : None, 
                  'distributed_backend' : None, 'replace_sampler_ddp'  : False, 'check_val_every_n_epoch'  : 1, 
                  'benchmark' : True, 'auto_lr_find' : False, 'log_every_n_steps' : args.train_log_interval}
        return Trainer(**kwargs)
    
    @staticmethod
    def run(args):
        cudnn.benchmark = True
        cudnn.enabled = True
        if args.val_files == "train":
            args.val_files = args.train_files
        else:
            args.train_files = args.train_files + "+-" + args.val_files
        
        model = VNModel(args)
    
        print("Using arguments: ")
        print(model.hparams)
    
        trainer = VNModel.create_trainer(args)       
        
        trainer.fit(model)    
        print("Training completed")
    
        return model

def main(args=None):
    parser = VNModel.get_argument_parser()
    if args is not None:
        parser.aults(**args)
        
    args, _ = parser.parse_known_args()
    VNModel.run(args)
    
if __name__ == '__main__':
    main()