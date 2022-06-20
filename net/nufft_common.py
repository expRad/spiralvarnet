"""Containing various common utility functions used throughout the implementation.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
import math
import torchvision.transforms.functional as TTF
import torchvision.transforms as TT
import matplotlib.gridspec as gridspec
import matplotlib

def gaussian(dims, mu= None, sigma = None):
    # returns a tensor of size dims with a bell curve centered at mu and standard deviation sigma
    # expects sigma and mu to be vectors of length len(dims)
        
    B, mu, sm = get_pos(dims, mu, sigma)

    sm = 0.5 / torch.pow(sm, 2.)
    fs = torch.pow(B - mu, 2.)
    out = torch.exp( - torch.sum(fs*sm, dim = -1))
   
    return out / out.sum()

def ball(dims, mu= None, sigma = None):
    coords, center, d = get_pos(dims, mu, sigma)
    k = torch.sum(torch.pow((coords - center)/d, 2.), dim = -1) < 1
    return k.to(torch.float32)

def cube(dims, mu= None, sigma = None):
    coords, center, d = get_pos(dims, mu, sigma)
    k = torch.abs(coords - center) - d
    out = 1- torch.heaviside(  k, torch.tensor(0., dtype=k.dtype))
    return (out.sum(-1) > len(dims)-1).to(torch.float32)

def get_pos(dims, mu= None, sigma = None):
    if type(dims) != int:
        numdims = len(dims)
    else:
        numdims  = 1
        
    if mu == None:
        mu = torch.div(torch.tensor(dims), 2)
    else:
        if type(mu) != torch.Tensor:
            mu = torch.tensor(mu)   
    if sigma == None:
        sigma = torch.div(torch.tensor(dims), 4)
    else:
        if type(sigma) != torch.Tensor:
            sigma = torch.tensor(sigma)
    
    for i in range(numdims):
        mu = mu.unsqueeze(dim = 0)
        sigma = sigma.unsqueeze(dim = 0)

    if type(dims) != int:
        A = torch.meshgrid(*[torch.arange(0,a) for a in dims]) # A[0] are x coords and A[1] are y coords
    else:
        A = torch.meshgrid(*[torch.arange(0,dims)]) # A[0] are x coords and A[1] are y coords
    B = torch.stack(A, dim=-1)
    
    mu = mu.expand(B.shape)
    sigma = sigma.expand(B.shape)
    return B, mu, sigma

def show_traj(traj, title = "", axis=True, dpi=100):
    traj = traj.cpu()
    two, numspokes, spokelength = traj.shape
    # draw trajectory
    fig = plt.figure(figsize=(8,8), dpi=dpi)
    ax = fig.subplots()
    kx = traj[0,:,:];
    ky = traj[1,:,:];
    kx_masked = traj[0,:, :];
    ky_masked = traj[1,:,:];
    for j in range(kx.shape[0]):
        ax.plot(kx[j, :], ky[j, :], 'y')
        ax.plot(kx_masked[j, :], ky_masked[j, :], 'black')
    if not axis:
        plt.axis('off')
    if title != "":
        plt.title(title)
    plt.show()
    return 

def get_id_str(f):
    return  f.split("/")[-1]

def plot_error_map(image, target):
    emap = torch.abs(torch.abs(image) - torch.abs(target)).cpu()
    fig = Figure(figsize=(10,10))
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    pos = ax.imshow(emap,cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.colorbar(pos, ax=ax, shrink=0.8)
    fig.tight_layout()
    canvas.draw()
    emapplot = np.array(canvas.renderer.buffer_rgba())[:,:,0]
    return emapplot[np.newaxis]
            
def rotate(angle, vect):
    # vect has shape ... x 2
    # gives vect, rotated at every point counter-clockwise (mathematically positive) by angle 
    M = torch.tensor([[math.cos(angle), math.sin(angle)],[-math.sin(angle), math.cos(angle)]], dtype = torch.float, device = vect.device)
    return torch.matmul(vect.to(torch.float),M)

def complex_transpose(A):
    return torch.stack([A[...,0], -A[...,1]],-1)

def imrotate(imagetensor, angle):
    return TTF.rotate(imagetensor.unsqueeze(-3), angle).squeeze(-3)

def resize_image_padding(image, targetsize):
    return TT.CenterCrop(targetsize)(image.unsqueeze(-3)).squeeze(-3)
    
def sens_reco(kspace_coilwise_input, traj_input, dcf_input, smaps_input, adjnufft):
    kspace_coilwise, traj, dcf, smaps = add_batchdim(kspace_coilwise_input, traj_input, dcf_input, smaps_input);
    return reduce_op(adjnufft_coilwise(adjnufft, kspace_coilwise, traj, dcf), smaps)

def add_batchdim(kspace_coilwise_input, traj_input, dcf_input, smaps_input):
    if len(kspace_coilwise_input.shape) == 4:
        kspace_coilwise = kspace_coilwise_input.unsqueeze(0)
    else:
        kspace_coilwise = kspace_coilwise_input
    if len(traj_input.shape) == 3:
        traj = traj_input.unsqueeze(0)
    else:
        traj = traj_input
    if len(dcf_input.shape) == 2:
        dcf = dcf_input.unsqueeze(0)
    else:
        dcf = dcf_input  
    if len(smaps_input.shape) == 4:
        smaps = smaps_input.unsqueeze(0)
    else:
        smaps = smaps_input  
    return kspace_coilwise, traj, dcf, smaps

def imconvert(image, numpy = True):
    if not type(image) == torch.Tensor:
        image = torch.tensor(image)
    image = torch.squeeze(image.cpu())
    if numpy:
        image = image.numpy()
        image = image[np.newaxis]
    image -= image.min()
    image = image / image.max()
    return image

def normalize_image(ref):
    normref = ref - ref.min()
    normref = normref / normref.max()
    return normref 

def reduce_op(image_coilwise, smaps):
    # expects:  image_space as      batch x coils x xdim x ydim x 2(real/imag)     float32 array 
    #           smaps               batch x coils x xdim x ydim x 2(real/imag)     float32 array
    # returns:  coil combination as batch x xdim x ydim x 2(real/imag)          float array
    smaps_adj = complex_conj(smaps)
    return torch.sum(complex_mul(smaps_adj,image_coilwise), 1)

def expand_op(im, smaps):
    # expects image as    batch x 1(channels/coils) x xdim x ydim x 2(real/imag)          float32 tensor
    #         smaps as    batch x coils x xdim x ydim x 2(real/imag)  float32 tensor
    # returns image space coilwise as batch x coils x xdim x ydim x 2(real/imag)
    # im = im.unsqueeze(-1)
    # print(im.shape)
    im = im.expand(-1,smaps.shape[1], -1, -1, -1)
    return complex_mul(im, smaps)

def chans_to_batch_dim(x):
    b, c, *other = x.shape
    return x.contiguous().view(b * c, 1, *other), b

def batch_chans_to_chan_dim(x, batch_size):
    bc, one, *other = x.shape
    c = bc // batch_size
    return x.view(batch_size, c, *other)

def complex_mul(x, y):
    assert x.shape[-1] == y.shape[-1] == 2
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack((re, im), dim=-1)

def adjnufft_coilwise(adjnufft, kspace_coilwise_input, traj_input, dcf_input, normstring = "ortho", **kwargs): 
    # expects kspace as      batch x coils x nspokes x spokelength x 2(real/imag) float array
    #         ktraj as       batch x 2(xydim) x nspokes x spokelength
    #         dcf as         batch x nspokes x spokelength
    # if no batch dimension exists, it is assumed to be 1
    # returns image space as batch x coils x xdim x ydim where self.image_size = (xdim, ydim)
    if len(kspace_coilwise_input.shape) == 4:
        kspace_coilwise = kspace_coilwise_input.unsqueeze(0)
    else:
        kspace_coilwise = kspace_coilwise_input
    if len(traj_input.shape) == 3:
        traj = traj_input.unsqueeze(0)
    else:
        traj = traj_input
    if len(dcf_input.shape) == 2:
        dcf = dcf_input.unsqueeze(0)
    else:
        dcf = dcf_input
    flat_data,_,_ = kdata_to_flat(kspace_coilwise)
    flat_traj,numspokes,spokelength = ktraj_to_flat(traj)   
    flat_dcf,_,_ = dcf_to_flat(dcf)
    compl_dcfed_data = flat_dcf*flat_data
    return adjnufft(compl_dcfed_data, flat_traj, norm =normstring, **kwargs) #.permute([1,0,2,3,4])
   
def nufft_coilwise(nufft, image_input, traj_input, numspokes, spokelength, normstring = "ortho"):
    # expects image_coilwise as  batch x coils x xdim x ydim x 2(real/complex)   float array
    # returns kspace as batch x coils x numspokes x spokelength where self.image_size = (xdim, ydim)
    if len(traj_input.shape) == 3:
        traj = traj_input.unsqueeze(0)
    else:
        traj = traj_input
    if len(image_input.shape) == 4:
        image = image_input.unsqueeze(0)
    else:
        image = image_input    
    
    flat_traj,numspokes,spokelength = ktraj_to_flat(traj)
    return kdata_from_flat(nufft(image, flat_traj, norm=normstring), numspokes, spokelength)

def complex_abs(data):
    # expects as ... x 2
    # returns as ...
    if data.size(-1) == 2:
        return (data ** 2).sum(dim=-1).sqrt()
    else:
        return data

def complex_conj(x):
    assert x.shape[-1] == 2
    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)

def kdata_to_flat(kdata):
    # expects as batch x coils x numspokes x spokelength x 2(real/imag)
    # returns as batch x coils x numspokes*spokelength x 2(real/imag) 
    kdata_flat = kdata.reshape(kdata.shape[0],kdata.shape[1], -1, 2)
    return kdata_flat, kdata.shape[2], kdata.shape[3]
    
def kdata_from_flat(kdata_flat, numspokes, spokelength):
    # expects kdata as     batch x coils x nummspokes*spokelength x 2(real/imag) 
    # returns as           batch x coils x numspokes x spokelength x 2(real/imag)
    return kdata_flat.reshape(kdata_flat.shape[0], kdata_flat.shape[1], numspokes, spokelength, 2)

def complex_to_chan_dim(x):
    b, c, h, w, two = x.shape
    assert two == 2
    return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)

def chan_complex_to_last_dim(x):
    b, c2, h, w = x.shape
    assert c2 % 2 == 0
    c = c2 // 2
    return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1)

def ktraj_to_flat(ktraj):
    # expects as batch x 2(xydim) x nspokes x spokelength
    # returns as batch x 2(xydim) x numspokes*spokelength
    ktraj_flat = ktraj.reshape(ktraj.shape[0],ktraj.shape[1], -1)
    return ktraj_flat, ktraj.shape[2], ktraj.shape[3]
    
def ktraj_from_flat(ktraj_flat, numspokes, spokelength):
    # expects ktraj as    batch x 2(xydim) x numspokes*spokelength      float torch tensor
    # returns ktraj as    batch x 2(xydim) x numspokes x spokelength    float torch tensor
    return  ktraj_flat.reshape(ktraj_flat.shape[0], ktraj_flat.shape[1], numspokes,spokelength)

def dcf_to_flat(dcf):
    #    expects dcf as    batch x nspokes x spokelength                float32 tensor
    #    returns dcf as    batch x 1(coils of data) x numspokes*spokelength  x 1(real/imag of data)   float32 tensor
    dcf_flat = dcf.reshape(dcf.shape[0], 1, -1,1)
    return dcf_flat, dcf.shape[1], dcf.shape[2]


def imshow(image, title="", colorbar=False, axis=True, ba = 1, dpi=200, export=False, normalize_separately= False, **kwargs):
    if export:
        oldbackend = matplotlib.get_backend()
        matplotlib.use('Agg')
    if type(image) == torch.Tensor or type(image) == np.ndarray:
        if type(image) == np.ndarray:
            image = torch.tensor(image)
        if image.dtype == torch.complex64 or image.dtype == torch.complex128:
            image = torch.abs(image)
        if image.shape[-1] == 2:
            image = complex_abs(image)
        image = image.detach().cpu()
        image = torch.squeeze(image)
        # normalization and brightness adjnustment
        if ba != 1:
            image = image / torch.max(image)
            image[image > ba] = ba
            image = image / ba
        # plotting
        ximsize, yimsize = image.shape
        fig = plt.figure(figsize=(ximsize/80+1, yimsize/80+1), dpi = dpi)
        if export:
            canvas = FigureCanvas(fig)
        ax = fig.add_subplot()
        s = ax.imshow(image.cpu().numpy(), cmap="gray", **kwargs)
        if colorbar:
            fig.colorbar(s, ax=ax, shrink=0.8)
        if not axis:
            plt.axis("off")
        plt.title(title)
        if export:
            canvas.draw()
            out = torch.tensor(np.array(canvas.renderer.buffer_rgba())[:,:,0])
            plt.close(fig)
            return out
        else:
            # fig.show()
            plt.show()
    elif type(image) == list:
        mi, ma = get_range(image)
        gs = math.ceil(math.sqrt(len(image)))
        fig = plt.figure(dpi=dpi, figsize = (10,10), constrained_layout=True)
        if export:
            canvas = FigureCanvas(fig)
        mastergs = gridspec.GridSpec(gs, gs, figure=fig, wspace=0, hspace=0)
        for i in range(gs):
            for j in range(gs):
                ind = i + gs*j
                if ind < len(image):
                    ax = fig.add_subplot(mastergs[j, i])
                    im = image[ind].detach()
                    if im.is_complex():
                        im = torch.abs(im)
                    elif len(im.shape) == 3:
                        im = complex_abs(im)
                    im = torch.squeeze(im)
                    if not normalize_separately:
                        s = ax.imshow(im, vmin = mi, vmax = ma*ba, cmap = "gray")
                    else:
                        mi, ma = get_range(im)
                        s = ax.imshow(im, vmin = mi, vmax = ma*ba, cmap = "gray")
                    if colorbar:
                        fig.colorbar(s, ax=ax, shrink=0.8)
                    if not axis:
                        ax.axis("off")
                    if type(title) == list:
                        ax.set_title(title[ind])
        if type(title) == str and not title =="":
            fig.suptitle(title)
        if export:
            canvas.draw()
            out= torch.tensor(np.array(canvas.renderer.buffer_rgba())[:,:,0])
            plt.close(fig)
            matplotlib.use(oldbackend)
            return out
        else:
            plt.show()
        
def get_range(images):
    allmax = -1e20
    allmin = 1e20
    for i in images:
        if type(i) == list:
            currmin, currmax = get_range(i)
        elif type(i) == torch.Tensor:
            currmax = float(i.max())
            currmin = float(i.min())
        elif type(i) == np.ndarray:
            currmax = np.max(i)
            currmin = np.min(i)
        if allmax < currmax:
            allmax = currmax
        if allmin > currmin:
            allmin = currmin
    return allmin, allmax