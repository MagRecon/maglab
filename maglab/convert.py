import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import scipy.constants as const
from .helper import padding_into
import logging

logger = logging.getLogger(__name__)

def m2A(m, dx=1, Ms=1/const.mu_0, shape=()):
    if len(shape) == 0:
        shape = m.shape[1:]
    converter = MagToVec(*shape, dx)
    A = converter(m, Ms)
    return A

class MagToVec(nn.Module):
    """
    Initialize the MagToVec class.

    Args:
        shape (tuple): The shape of the output vector potential A, which should be (nx, ny, nz).
        dx (float): The unit size of the magnetization M.

    Attributes:
        shape (tuple): The shape of the output vector potential A.
        kernel (torch.Tensor): The Fourier transform of the kernel used in the calculation.
    """
    
    def __init__(self, nx,ny,nz,dx):
        super().__init__()
        self._init_kernel(nx,ny,nz,dx,)
        self.shape = (nx,ny,nz)
        self.dx = dx
    
    def _init_kernel(self, nx,ny,nz,dx):
        # r-grid kernel to calculate vector potential
        x = dx * torch.fft.fftfreq(nx) * nx
        y = dx * torch.fft.fftfreq(ny) * ny
        z = dx * torch.fft.fftfreq(nz) * nz
        X,Y,Z = torch.meshgrid(x,y,z, indexing='ij')
        R3 = torch.sqrt(X**2+Y**2+Z**2)**3 
        ker = dx**3 * torch.stack([X/R3, Y/R3, Z/R3]) / (4 * np.pi)
        ker[:,0,0,0] = 0
        
        ker = torch.fft.fftshift(ker, dim=(1,2,3))
        ker = padding_into(ker, (3,2*nx,2*ny,2*nz))
        ker = torch.fft.ifftshift(ker, dim=(1,2,3))
        ker = torch.fft.fftn(ker, dim=(1,2,3))
        
        self.register_buffer('kernel', ker)
        
    def conv_q(self, mag):
        mag = torch.fft.ifftshift(mag, dim=(1,2,3))
        mag_q = torch.fft.fftn(mag, dim=(1,2,3))
        logger.debug("kernel.shape:{}".format(self.kernel.shape))
        logger.debug("mag_q.shape:{}".format(mag_q.shape))
        logger.debug("kernel.dtype:{}".format(self.kernel.dtype))
        logger.debug("mag_q.dtype:{}".format(mag_q.dtype))
        A_k = torch.cross(mag_q, self.kernel, dim=0)
        A = torch.fft.ifftn(A_k, dim=(1,2,3))
        A = torch.fft.fftshift(A, dim=(1,2,3))
        return A
        
    def __call__(self, mag, Ms):
        mag = mag.float()
        device = mag.device
        self.to(device)
        nx,ny,nz = self.shape
        mag = padding_into(mag, (3,2*nx,2*ny,2*nz))
        A = const.mu_0 * Ms * self.conv_q(mag) 
        A = padding_into(A, (3,*self.shape))
        return A.real

def curl(A, dx):
    dAx_dy = torch.gradient(A[0], dim=1)[0] / dx
    dAx_dz = torch.gradient(A[0], dim=2)[0] / dx
    
    dAy_dx = torch.gradient(A[1], dim=0)[0] / dx
    dAy_dz = torch.gradient(A[1], dim=2)[0] / dx
    
    dAz_dx = torch.gradient(A[2], dim=0)[0] / dx
    dAz_dy = torch.gradient(A[2], dim=1)[0] / dx
    
    # Compute curl as the cross product of the gradient and the vector field
    curl_x = dAz_dy - dAy_dz
    curl_y = dAx_dz - dAz_dx
    curl_z = dAy_dx - dAx_dy
    
    # Stack the components to get the curl vector field
    curl_A = torch.stack((curl_x, curl_y, curl_z), dim=0)
    
    return curl_A


def ift(x):
    x = torch.fft.ifftn(x, dim=(1,2,3))
    x = torch.fft.fftshift(x, dim=(1,2,3))
    return x

def skyrmion_number(spin):
    pxm = torch.gradient(spin, dim=1)[0]
    pym = torch.gradient(spin, dim=2)[0]
    pzm = torch.gradient(spin, dim=3)[0]
    px_py_m = torch.cross(pxm,pym)
    py_pz_m = torch.cross(pym,pzm)
    pz_px_m = torch.cross(pzm,pxm)
    bx = torch.sum(spin*py_pz_m, dim=0)
    by = torch.sum(spin*pz_px_m, dim=0)
    bz = torch.sum(spin*px_py_m, dim=0)
    return 1/(4*torch.pi) * torch.stack([bx,by,bz])

    
def hopf_index(spin):
    B = skyrmion_number(spin)
    A = m2A(B, dx=1, Ms=1/const.mu_0)
    ab = torch.sum(A*B, dim=0)
    return ab