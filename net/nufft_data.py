""" Utility fucntions and classes for data loading and processing of non-Cartesian k-space data.
"""

import numpy as np
import torch 
import scipy.io as sio
from torch.utils.data import Dataset
import glob
import re
import random
import math
import PIL
import torchvision.transforms.functional as TTF

from . import nufft_common as CT

def load_mat_complex(path, key):
    npar = np.array(sio.loadmat(path)[key])
    return torch.stack([torch.tensor(np.real(npar)),torch.tensor(np.imag(npar))],-1)

def save_datafile(matpath, kspace, traj, dcf, ref, smaps):
    outdict = {"kspace" : kspace.cpu().numpy(), "traj" : traj.cpu().numpy(), "dcf" : dcf.cpu().numpy(), "smaps" : smaps.cpu().numpy(), "ref" : ref.cpu().numpy()}
    save_mat(matpath, outdict)

def save_mat(matpath, outdict):
    outdict_np = {}
    for key in outdict:
        outdict_np[key] = torch.tensor(outdict[key]).cpu().numpy()
    sio.savemat(matpath, outdict_np)

def load_mat(path, key=None, whereto = None):
    if type(key) == str:
        if whereto == None:
            return torch.tensor(np.array(sio.loadmat(path)[key]))
        else:
            return torch.tensor(np.array(sio.loadmat(path)[key])).to(whereto)
    else:
        ret = []
        a = sio.loadmat(path)
        for i in a:
            if type(a[i]) == np.ndarray:
                if whereto == None:
                    ret.append(torch.tensor(np.array(a[i])))
                else:
                    ret.append(torch.tensor(np.array(a[i])).to(whereto))
        return tuple(ret)

def load_data_file(data_file, load_smaps=True, load_kspace=True, load_traj=True, load_dcf=True, load_ref=True,  dtype=torch.float, towhere=None):
    # loads matlab files with all elements included, with keys in the matlab file : traj kspace dcf ref
    # the traj should already have the format numspokes x spokelength x 2, so we do not need to permute anything
    mat_file = sio.loadmat(data_file)
    kspace = torch.zeros(1,1,1,1)
    traj = torch.zeros(1,1,1)
    dcf = torch.zeros(1,1)
    ref = torch.zeros(1,1)
    smaps = torch.zeros(1,1,1,1)
    if load_kspace:
        kspace = torch.tensor(np.array(mat_file['kspace']), dtype=dtype)
    if load_traj:
        traj = torch.tensor(np.array(mat_file['traj']), dtype=dtype)    
    if load_dcf:
        dcf = torch.tensor(np.array(mat_file['dcf']), dtype=dtype)    
    if load_ref:
        ref = torch.tensor(np.array(mat_file['ref']), dtype=dtype)
    if load_smaps:
        smaps = torch.tensor(np.array(mat_file['smaps']), dtype=dtype)
        
    if isinstance(towhere, str):
        towhere = torch.device(towhere)
        
    if towhere != None:
        kspace = kspace.to(towhere)
        traj = traj.to(towhere)
        dcf = dcf.to(towhere)
        ref = ref.to(towhere)
        smaps= smaps.to(towhere)
        
    return kspace, traj, dcf, ref, smaps

def get_filelist(data_files):
    file_list = []
    hubs = data_files.split("+")
    for hub in hubs:
        subtract = False
        spec = hub
        if bool(re.match("\A-.*\Z", hub)):
            subtract = True  
            spec = hub[1:]
        itms = []
        if bool(re.match("\A.*§[0-9]+\Z", spec)):
            path = spec.split('§')[0]
            num = int(spec.split('§')[-1])
            itms = glob.glob(path, recursive=True)
            itms = itms[:num]
        elif bool(re.match("\A.*\.txt\Z", spec)):
            f = open(spec, "r")
            for x in f:
                itms.append(x.replace("\n", "")) 
            f.close()
        else:
            itms = glob.glob(spec, recursive=True)
         
        if subtract:
            file_list = [x for x in file_list if x not in itms]
        else:
            file_list.extend(itms)
    return file_list

def apply_data_augmentation(kspace, traj, ref, smaps, aug_rotate):
    if aug_rotate != 0:
        angle = (random.random() - 0.5)*2*aug_rotate
        if not torch.equal(ref, torch.zeros(1,1)):
            ref = TTF.rotate(ref.unsqueeze(0), angle, resample=PIL.Image.BILINEAR).squeeze(0)
        if not torch.equal(traj, torch.zeros(1,1,1)):
            traj = CT.rotate(angle/360.0*2*math.pi, traj.permute(1,2,0)).permute(2,0,1)
        if not torch.equal(smaps, torch.zeros(1,1,1,1)):
            smaps = TTF.rotate(smaps.permute(0,3,1,2), angle).permute(0,2,3,1)
            smapsabs = CT.complex_abs(smaps)
            smaps_real = torch.where(smapsabs == 0., torch.ones_like(smapsabs)*1e-5, smaps[...,0])
            smaps_ima = torch.where(smapsabs == 0., torch.ones_like(smapsabs)*0, smaps[...,1])
            smaps = torch.stack([smaps_real, smaps_ima],-1)    
    return kspace, traj, ref, smaps

class SpiralItemData(Dataset):
    def __init__(self, data_files, load_smaps=True, load_kspace=True, load_traj=True, load_dcf=True, 
                 load_ref=True, transform=None, dtype=torch.float, kspace_amplification=1,
                 seed=731, aug_rotate=0.0, norm_targets=-1, 
                 ref_mask="none", identifier_append=""):
        self.dtype = dtype 
        self.load_kspace = load_kspace;
        self.load_traj = load_traj;
        self.load_dcf = load_dcf;
        self.load_ref = load_ref;
        self.kspace_amplification = kspace_amplification
        self.load_smaps = load_smaps
        self.aug_rotate = aug_rotate
        self.norm_targets = norm_targets
        self.ref_mask = ref_mask
        self.identifier_append = identifier_append
        
        random.seed(seed)
        self.file_list = get_filelist(data_files)
        if not data_files.endswith(".txt"):
            random.shuffle(self.file_list)

        if not self.file_list:
            raise ValueError("Given list of data files for the dataloader is empty! Given data path is: " + data_files)
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        filename = self.file_list[i]
        kspace, traj, dcf, ref, smaps = load_data_file(data_file=filename,load_smaps=self.load_smaps,load_kspace=self.load_kspace,
                                                       load_traj=self.load_traj, load_dcf=self.load_dcf, load_ref = self.load_ref,
                                                       dtype=self.dtype)
        assert len(kspace.shape) == 4
        kspace, traj, ref, smaps = apply_data_augmentation(kspace, traj, ref, smaps, self.aug_rotate)
        if self.norm_targets > 0:
            ref = CT.normalize_image(ref)*self.norm_targets
        kspace = kspace *self.kspace_amplification
        identifier = CT.get_id_str(filename) + self.identifier_append
        
        if self.ref_mask != "none":
            ref = ref*self.ref_mask.to(torch.float32)
        return  kspace, traj, dcf, ref, smaps, identifier