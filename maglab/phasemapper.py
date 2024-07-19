import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
import numpy as np
import warnings
from .const import c_m, mu_0
from .helper import padding_into
import numbers

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
    def __init__(self, fov, dx, padding=True, rotation_padding=None):
        super().__init__()

        self.fov = fov
        if padding:
            if rotation_padding and rotation_padding <= fov:
                self.pd_sz = rotation_padding
            else:
                raise ValueError("Need specify rotation_padding, which can not be larger than fov.")
        else: 
            self.pd_sz = -1
        self._register_kernel(fov, dx)
        
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
    
    def _register_kernel(self, fov, dx):
        kx, ky = 2*torch.pi * torch.fft.fftfreq(fov,dx), 2*torch.pi * torch.fft.rfftfreq(fov,dx)
        KX, KY = torch.meshgrid(kx,ky,indexing='ij')
        k2 = torch.pow(KX,2) + torch.pow(KY,2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ker_x, ker_y = KX/k2,KY/k2
        ker_x[0,0] = 0.
        ker_y[0,0] = 0.
            
        self.register_buffer('ker_x', ker_x)
        self.register_buffer('ker_y', ker_y)
        self.dx = dx

    def __call__(self, m, theta, axis, Ms=1/mu_0):
        """Return phase.

        Args:
            M : 3D magnetization tensor shaped (3,nx,ny,nz)
            theta : projection angle in degree
            axis : rotation axis, which can be chosen from (0,1,2) representing x,y,and z respectively.
            dx: unit length in meter.

        Returns:
            phase(ndarray): 2D phase.
        """
        if isinstance(theta, numbers.Number):
            theta = [theta]
            axis = [axis]
            
        N = len(theta)
        uvs = []
        for i in range(N):
            u, v = self.get_uv(m*Ms, theta[i], axis[i])
            uv = torch.stack((u,v), dim=0)
            uvs.append(uv)
        uvs = self.dx * torch.stack(uvs, dim=0)
        
        (N,_,nx,ny) = uvs.shape
        if self.fov > nx or self.fov > ny:
            uvs = padding_into(uvs, (N, 2, self.fov,self.fov))
        
        uvs =  torch.fft.ifftshift(uvs,dim=(2,3))
        uvs_q = torch.fft.rfftn(uvs, dim=(2,3))
        ker_x = self.ker_x.unsqueeze(0).repeat(N,1,1)
        ker_y = self.ker_y.unsqueeze(0).repeat(N,1,1)
        A_k = -1j * mu_0 * (uvs_q[:,0,:,:]*ker_y - uvs_q[:,1,:,:]*ker_x) #(N,1,nx,ny//2)
        phi_k = c_m * A_k[:,:,:] #beam along z+ direction
        phi = -1 * torch.fft.irfftn(phi_k, dim=(1,2)) #beam along z- direction
        return torch.fft.fftshift(phi, dim=(1,2))