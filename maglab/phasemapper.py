import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
import numpy as np
import warnings
from .const import c_m, mu_0
from .helper import padding_into

__all__ = ['PhaseMapper', 'rotation_3d', 'projection']

def rotation_3d(x, theta, axis, expand=False, fill=0.0):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if type(x) is np.ndarray:
        x = torch.from_numpy(x)
        x = x.float()

    x = x.to(device)

    if axis == 0:
        x = rotate(x, interpolation=InterpolationMode.BILINEAR, angle=float(theta), expand=expand, fill=fill)
    elif axis == 1:
        x = x.permute((1, 0, 2))
        x = rotate(x, interpolation=InterpolationMode.BILINEAR, angle=-1*float(theta), expand=expand, fill=fill)
        x = x.permute((1, 0, 2))
    elif axis == 2:
        x = x.permute((2, 1, 0))
        x = rotate(x, interpolation=InterpolationMode.BILINEAR, angle=-float(theta), expand=expand, fill=fill)
        x = x.permute((2, 1, 0))
    else:
        raise ValueError('Not invalid axis')
    
    return x.squeeze(0)
    
def projection(x, theta, axis):
    x = rotation_3d(x, theta, axis)
    return torch.sum(x, axis=-1)

class PhaseMapper(nn.Module):
    def __init__(self, fov, cellsize, padding=True, rotation_padding=None):
        super().__init__()

        self.fov = fov
        if padding:
            if rotation_padding and rotation_padding <= fov:
                self.pd_sz = rotation_padding
            else:
                raise ValueError("Need specify rotation_padding, which can not be larger than fov.")
        else: 
            self.pd_sz = -1
        self._register_kernel(fov, cellsize)
        
    def get_device(self):
        return self.ker_x.device
 
    def get_uv(self, M, theta, axis):
        if isinstance(M, np.ndarray):
            M = torch.from_numpy(M).to(self.get_device())
            
        if self.pd_sz > 0:
            (_,nx,ny,nz) = M.shape
            M = padding_into(M, (3, self.pd_sz,self.pd_sz,self.pd_sz))
            
        mx = projection(M[0,], theta, axis)
        my = projection(M[1,], theta, axis)
        mz = projection(M[2,], theta, axis)
        rad = torch.tensor(theta*torch.pi/180).to(self.get_device())
        ct, st = torch.cos(rad), torch.sin(rad)
        if axis == 0:
            return mx, my*ct-mz*st
        elif axis == 1:
            return mx*ct+mz*st, my
        elif axis == 2:
            return mx*ct-my*st, my*ct+mx*st
        else:
            raise ValueError("axis mush be one of (0,1,2)!")
    
    def _register_kernel(self, fov, cellsize):
        kx, ky = 2*torch.pi * torch.fft.fftfreq(fov,cellsize), 2*torch.pi * torch.fft.rfftfreq(fov,cellsize)
        KX, KY = torch.meshgrid(kx,ky,indexing='ij')
        k2 = torch.pow(KX,2) + torch.pow(KY,2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ker_x, ker_y = KX/k2,KY/k2
        ker_x[0,0] = 0.
        ker_y[0,0] = 0.
            
        self.register_buffer('ker_x', ker_x)
        self.register_buffer('ker_y', ker_y)
        self.cellsize = cellsize

    def __call__(self, m, theta, axis, Ms=1/mu_0):
        """Return phase.

        Args:
            M : 3D magnetization tensor shaped (3,nx,ny,nz)
            theta : projection angle in degree
            axis : rotation axis, which can be chosen from (0,1,2) representing x,y,and z respectively.
            cellsize: unit length in meter.

        Returns:
            phase(ndarray): 2D phase.
        """
        # cellsize:unit length of the cubic meshgrid
        u, v = self.get_uv(m, theta, axis)
        (nx,ny) = u.shape
        if self.fov > nx or self.fov > ny:
            u = padding_into(u, (self.fov,self.fov))
            v = padding_into(v, (self.fov,self.fov))
            
        u, v = self.cellsize * torch.fft.ifftshift(u,dim=(0,1)), self.cellsize * torch.fft.ifftshift(v,dim=(0,1))
        fft_u, fft_v = torch.fft.rfft2(u), torch.fft.rfft2(v)
        A_k = -1j * mu_0 * Ms * (fft_u*self.ker_y - fft_v*self.ker_x)
        phi_k = -1 * c_m * A_k #beam along z+ direction
        phi = -1 * torch.fft.irfft2(phi_k) #beam along z- direction
        return torch.fft.fftshift(phi, dim=(0,1))