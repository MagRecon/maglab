import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.constants as const

import itertools

from .helper import padding_width
from .microfields import MicroField

import logging

logger = logging.getLogger(__name__)

__all__ = ['DeMag', 'coords']

sqrt = torch.sqrt     

def atan(x):
    x = torch.atan(x)
    x = torch.nan_to_num(x, nan=0.0)
    return x

def asinh(x):
    x = torch.asinh(x)
    x = torch.nan_to_num(x, nan=0.0)
    return x
    
#newell f
def f(p):
    x, y, z = p[0,], p[1,], p[2,]
    return + y / 2.0 * (z**2 - x**2) * asinh(y / (sqrt(x**2 + z**2))) \
            + z / 2.0 * (y**2 - x**2) * asinh(z / (sqrt(x**2 + y**2))) \
            - x*y*z * atan(y*z / (x * sqrt(x**2 + y**2 + z**2)))       \
            + 1.0 / 6.0 * (2*x**2 - y**2 - z**2) * sqrt(x**2 + y**2 + z**2)

#newell g
def g(p):
    x, y, z = p[0,], p[1,], p[2,]
    value = x*y*z * asinh(z / (sqrt(x**2 + y**2)))                         \
            + y / 6.0 * (3.0 * z**2 - y**2) * asinh(x / (sqrt(y**2 + z**2))) \
            + x / 6.0 * (3.0 * z**2 - x**2) * asinh(y / (sqrt(x**2 + z**2))) \
            - z**3 / 6.0 * atan(x*y / (z * sqrt(x**2 + y**2 + z**2)))        \
            - z * y**2 / 2.0 * atan(x*z / (y * sqrt(x**2 + y**2 + z**2)))    \
            - z * x**2 / 2.0 * atan(y*z / (x * sqrt(x**2 + y**2 + z**2)))    \
            - x*y * sqrt(x**2 + y**2 + z**2) / 3.0

    return value
       
def Nxx(p, dV, fun=f):
    x = 8 * fun(p)
    
    # nearest
    for i in range(1, 4):
        x = x - 4 * fun(torch.roll(p, shifts=(1), dims=(i)))
        x = x - 4 * fun(torch.roll(p, shifts=(-1), dims=(i)))
    
    # next nearest  
    for i in range(1, 4):     
        dims = tuple(j for j in [1,2,3] if not j==i)
        combinations = list(itertools.product([-1, 1], repeat=2))   
        for comb in combinations:
            x = x + 2 * fun(torch.roll(p, shifts=comb, dims=dims))
    
    # next next nearest
    combinations = list(itertools.product([-1, 1], repeat=3))           
    for comb in combinations:
        x = x - fun(torch.roll(p, shifts=comb, dims=(1,2,3)))
    
    x = x[1:-1,1:-1,1:-1]
    return x / (4*torch.pi*dV)

def Nxy(p, dV):
    return Nxx(p, dV, fun=g)

def coords(n, d=1):
    return d * (torch.arange(n)-n//2).double()

class DeMag(MicroField):        
    def __init__(self, nx, ny, nz, dx):
        super().__init__()
        self.shape = (nx,ny,nz)
        self.dV = dx**3
        
        x,y,z = coords(2*nx+2, dx).cuda(), coords(2*ny+2, dx).cuda(), coords(2*nz+2, dx).cuda()
            
        X,Y,Z = torch.meshgrid(x,y,z,indexing='ij')
        p = torch.stack([X,Y,Z], axis=0)
        
        N_xx = Nxx(p, self.dV)
        N_yy = Nxx(p[[1,0,2], :,:,:], self.dV)
        N_zz = Nxx(p[[2,1,0], :,:,:], self.dV)
        
        p1 = p[:,1:-1,1:-1,1:-1]
        N_xy = Nxy(p, self.dV) 
        N_xz = Nxy(p[[0,2,1], :,:,:], self.dV) 
        N_yz = Nxy(p[[1,2,0], :,:,:], self.dV) 

        # Hx = Nx . M
        N_demag_x = torch.stack([N_xx, N_xy, N_xz])
        N_demag_y = torch.stack([N_xy, N_yy, N_yz])
        N_demag_z = torch.stack([N_xz, N_yz, N_zz])
        
        N_demag = -1*torch.stack([N_demag_x, N_demag_y, N_demag_z])
        N_demag = torch.fft.ifftshift(N_demag, dim=(2,3,4))
        N_demag = torch.fft.rfftn(N_demag, dim=(2,3,4))
        
        self.register_buffer('N_demag', N_demag)   
        
        px1, px2 = padding_width(self.shape[0], 2*self.shape[0])    
        py1, py2 = padding_width(self.shape[1], 2*self.shape[1])    
        pz1, pz2 = padding_width(self.shape[2], 2*self.shape[2])  
        self.padding_dims = (pz1,pz2,py1,py2,px1,px2)
        self.crop_dims = tuple(-1*x for x in self.padding_dims)
        
    def _rfft(self, x):
        x = torch.fft.ifftshift(x, dim=(1,2,3))
        x = torch.fft.rfftn(x, dim=(1,2,3))
        return x
    
    def _irfft(self, x):
        x = torch.fft.irfftn(x, dim=(1,2,3))
        x = torch.fft.fftshift(x, dim=(1,2,3))
        return x
    
    def effective_field(self, M):
        M = F.pad(M, self.padding_dims, 'constant', 0.)
        mk = self._rfft(M)
        hk_x = torch.sum(self.N_demag[0,] * mk, axis=0)
        hk_y = torch.sum(self.N_demag[1,] * mk, axis=0)
        hk_z = torch.sum(self.N_demag[2,] * mk, axis=0)
        hk = torch.stack([hk_x, hk_y, hk_z], axis=0)
        H = self._irfft(hk)
        H = F.pad(H, self.crop_dims)
        M = F.pad(M, self.crop_dims)
        return H, M
    
    # input should be m*Ms    
    def forward(self, m, geo, Ms):
        M = m * Ms   
        for i in range(3):
            if not M.shape[i+1] == self.shape[i]:
                raise ValueError("DeMag: Shape not match! Need {}, but got {} instead".format(self.shape, M.shape[1:]))
            
        H,M = self.effective_field(M)
        
        E = -1/2 * const.mu_0 * torch.sum(H*M, axis=0)
        
        return E
    
    def get_params(self,):
        return {'classname': self.__class__.__name__}