import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
import numpy as np
import warnings
from .const import c_m, mu_0
from .helper import padding_into,Euler_XYZ
import numbers

__all__ = ['PhaseMapper', 'Euler_rotation', 'projection']

def Euler_rotation(x, alpha, beta, gamma, expand=False, fill=0.0):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if type(x) is np.ndarray:
        x = torch.from_numpy(x)
        x = x.float()

    x = x.to(device)
    
    if not gamma == 0:
        x = x.permute((2, 1, 0))
        x = rotate(x, interpolation=InterpolationMode.BILINEAR, angle=-float(gamma), expand=expand, fill=fill)
        x = x.permute((2, 1, 0))
        
    if not beta == 0:
        x = x.permute((1, 0, 2))
        x = rotate(x, interpolation=InterpolationMode.BILINEAR, angle=-1*float(beta), expand=expand, fill=fill)
        x = x.permute((1, 0, 2))
        
    if not alpha == 0:
        x = rotate(x, interpolation=InterpolationMode.BILINEAR, angle=float(alpha), expand=expand, fill=fill)
    
    return x.squeeze(0)

def Euler_rotate_vector_field(x, alpha, beta, gamma):
    mx = Euler_rotation(x[0], alpha, beta, gamma)
    my = Euler_rotation(x[1], alpha, beta, gamma)
    mz = Euler_rotation(x[2], alpha, beta, gamma)
    a,b,c = alpha*torch.pi/180, beta*torch.pi/180, gamma*torch.pi/180
    R = Euler_XYZ(a,b,c)
    mx1 = R[0,0] * mx + R[0,1]*my + R[0,2] * mz
    my1 = R[1,0] * mx + R[1,1]*my + R[1,2] * mz
    mz1 = R[2,0] * mx + R[2,1]*my + R[2,2] * mz
    return torch.stack([mx1,my1,mz1])
    
    
def projection(x, alpha=0., beta=0., gamma=0.):
    x = Euler_rotation(x, alpha, beta, gamma)
    return torch.sum(x, axis=-1)

class PhaseMapper(nn.Module):
    def __init__(self, fov, dx, rotation_padding=-1):
        super().__init__()

        self.fov = fov
        self.padding_size = rotation_padding

        self._register_kernel(fov, dx)
        
    def get_device(self):
        return self.ker_x.device
 
    def get_uv(self, M, alpha=0., beta=0., gamma=0.):
        if isinstance(M, np.ndarray):
            M = torch.from_numpy(M).to(self.get_device())
            
        if self.padding_size > 0:
            (_,nx,ny,nz) = M.shape
            M = padding_into(M, (3, self.padding_size,self.padding_size,self.padding_size))
            
        mx = projection(M[0,], alpha, beta, gamma) * self.dx
        my = projection(M[1,], alpha, beta, gamma) * self.dx
        mz = projection(M[2,], alpha, beta, gamma) * self.dx
        a,b,c = alpha*torch.pi/180, beta*torch.pi/180, gamma*torch.pi/180
        R = Euler_XYZ(a,b,c)
        mu = R[0,0] * mx + R[0,1]*my + R[0,2] * mz
        mv = R[1,0] * mx + R[1,1]*my + R[1,2] * mz
        return mu, mv
    
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

    def __call__(self, m, alpha=0., beta=0., gamma=0., Ms=1/mu_0):
        """Return phase.

        Args:
            M : magnetization tensor shaped (3,nx,ny,nz)
            alpha, beta, gamma : Euler angles in degree
            Ms: magnetization constant.

        Returns:
            phase(torch.tensor): 2D phase image.
        """            
        u, v = self.get_uv(m*Ms, alpha, beta, gamma)
        uv = torch.stack((u,v), dim=0)
        
        (_,nx,ny) = uv.shape
        if not (self.fov == nx and self.fov == ny):
            uv = padding_into(uv, (2, self.fov,self.fov))
        
        uv =  torch.fft.ifftshift(uv,dim=(1,2))
        uv_q = torch.fft.rfftn(uv, dim=(1,2))
        ker_x = self.ker_x
        ker_y = self.ker_y
        A_k = -1j * mu_0 * (uv_q[0,:,:]*ker_y - uv_q[1,:,:]*ker_x) #(N,1,nx,ny//2)
        phi_k = c_m * A_k #beam along z+ direction
        phi = -1 * torch.fft.irfftn(phi_k, dim=(0,1)) #beam along z- direction
        return torch.fft.fftshift(phi, dim=(0,1))