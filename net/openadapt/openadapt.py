"""Python implementation of the method by Walsh for computing coil sensitivity maps.

Based on:
D. O. Walsh, A. F. Gmitro, and M. W. Marcellin, “Adaptive reconstruction of phased array MR imagery,” Magnetic Resonance in Medicine, vol. 43, no. 5, pp. 682–690, May 2000. doi:10.1002/(sici)1522-2594(200005)43:5h682::aid-mrm10i3.0.co;2-g
and 
Mark Griswold, David Walsh, Robin Heidemann, Axel Haase, Peter Jakob, "The Use of an Adaptive Reconstruction for Array Coil Sensitivity Mapping and Intensity Normalization", Proceedings of the Tenth Scientific Meeting of the International Society for Magnetic Resonance in Medicine pg 2410 (2002)

Adapted from https://github.com/andyschwarzl/gpuNUFFT/blob/master/matlab/demo/utils/adapt_array_2d.m 
"""

import torch
import numpy as np
from numpy import linalg as LA
import PIL

def openadapt(imdata):
    # takes pytorch input of shape    coils x ximsize x yimsize x 2    float tensor
    # returns smaps of size           coils x ximsize x yimsize x 2    float tensor
    return cnp_pyt(openadapt_np(pyt_cnp(imdata)))

def cnp_pyt(A):
    return torch.stack([torch.tensor(np.real(A)), torch.tensor(np.imag(A))],-1)

def pyt_cnp(A):
    return A[...,0].cpu().numpy() + 1j*A[...,1].cpu().numpy()

def imresize(im, targetsize, method = "bilinear"):
    if method == "bilinear":
        resmeth = PIL.Image.BILINEAR
    if method == "nearest":
        resmeth = PIL.Image.NEAREST
    pilimage = PIL.Image.fromarray(im)
    return np.asarray(pilimage.resize(targetsize, resample=resmeth) )

def openadapt_np(im):
    nc, ny, nx = im.shape
    
    absimd = np.abs(im)
    pm = absimd.sum((1,2)) 
    maxcoil = np.argmax(pm,0)
    
    rn = np.eye(nc, dtype=np.complex64)
    rn_is_eye = True
    
    bs1 = 8
    bs2 = 8
    st = 2

    wsmall = np.zeros((nc, ny // st, nx//st),dtype=np.complex64)
    cmapsmall = np.zeros((nc, ny // st, nx//st),dtype=np.complex64)
    
    for x in range(st, nx+1, st):
        for y in range(st, ny+1, st):
            
            ymin1 = int(max(y - bs1 / 2, 1)) - 1
            xmin1 = int(max(x - bs2 / 2, 1)) - 1
            
            ymax1=int(min(y + bs1 / 2, ny)) - 1
            xmax1=int(min(x + bs2 / 2, nx)) - 1
                    
            ly1=ymax1 - ymin1 + 1
            lx1=xmax1 - xmin1 + 1;
            
            m1 = np.reshape(im[:,ymin1:ymax1+1, xmin1:xmax1+1], (nc,lx1*ly1), order='F')
            
            m1strich = np.conj(np.transpose(m1))
            m = np.matmul(m1, m1strich)
            
            if rn_is_eye:
                eigs,vects = LA.eigh(m)
                mf = vects[:,nc-1]

            else:
                rni = LA.inv(rn)
                eigs,vects = LA.eig(np.matmul(rni,m))
                ind = np.argmax(eigs)
                mf = vects[:,ind]

            normmf = mf
            if rn_is_eye:
                mf = mf / np.dot(mf, np.conj(mf))
            else:
                mf = mf / np.matmul(np.conj(np.transpose(mf)), np.matmul(rni, mf))
            
            mf = mf*np.exp(-1j*np.angle(mf[maxcoil])) 
            normmf = normmf*np.exp(-1j*np.angle(normmf[maxcoil]))
            
            wsmall[:,y // st -1, x//st -1] = mf
            cmapsmall[:,y // st -1, x//st -1] = normmf
    
    wfull = np.zeros((nc,ny,nx), dtype=np.complex64)
    cmap = np.zeros((nc, ny,nx), dtype=np.complex64)
    
    for i in range(nc):
        bws = np.abs(wsmall[i,:,:])
        aws = np.angle(wsmall[i,:,:])
        bcs = np.abs(cmapsmall[i,:,:])
        acs = np.angle(cmapsmall[i,:,:])
        
        wfull[i,:,:] = np.conj(imresize(bws, (ny,nx), "bilinear")) * np.exp(1j*imresize(aws, (ny,nx), "nearest"))
        cmap[i,:,:] = imresize(bcs, (ny,nx), "bilinear") * np.exp(1j*imresize(acs, (ny,nx), "nearest"))
        
    return cmap