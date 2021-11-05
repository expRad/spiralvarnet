import torch
from torch import nn
from torch.nn import functional as F
from . import nufft_common as CT

class SSIM(nn.Module):
    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer('w', torch.ones(
            1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)
        
    def convert(self, output, target):
        if len(output.shape) == 4:
            output = CT.complex_abs(output)
        if len(target.shape) == 4:
            target = CT.complex_abs(target)
        max_value = torch.stack([target[i].max() for i in range(target.shape[0])], 0) # compute max values of batch items separately 
        return output.unsqueeze(1), target.unsqueeze(1), max_value
    
    def forward(self, output, target):
        # expects     out_image    as     batch x xdim x ydim    float tensor  or  as      batch x xdim x ydim x 2(real/imag)  float tensor
        # expects     target       as     batch x xdim x ydim    float tensor  or  as      batch x xdim x ydim x 2(real/imag)  float tensor
        # returns ssim_loss as shape =   batch    tensor
        X, Y, data_range = self.convert(output, target)
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1  = 2 * ux * uy + C1
        A2 = 2 * vxy + C2
        B1 = ux ** 2 + uy ** 2 + C1
        B2 = vx + vy + C2
        D = B1 * B2
        S = (A1 * A2) / D
        return 1 - S.mean()