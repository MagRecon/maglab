import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.constants as const
from .helper import padding_width

class MagToVec(nn.Module):
    """Calculate vector potential A from magnetization M.

    Args:
        shape (tuple): shape of A: (nx,ny,nz)
        cellsize : unit size of M
    """
    def __init__(self, shape, cellsize):
        super().__init__()
        self.shape = shape
        c0 = 2*torch.pi/cellsize
        kx = c0 * torch.fft.fftfreq(shape[0]) 
        ky = c0 * torch.fft.fftfreq(shape[1]) 
        kz = c0 * torch.fft.rfftfreq(shape[2])
        KX,KY,KZ= torch.meshgrid(kx,ky,kz, indexing='ij')
        K2 = KX**2+KY**2+KZ**2
        ker = -1j * torch.stack([KX/K2,KY/K2,KZ/K2], axis=0)
        ker[:,0,0,0] = 0.
        self.register_buffer('kernel',  ker)
        
    def _rfft(self,x):
        x = torch.fft.ifftshift(x, dim=(1,2,3))
        x = torch.fft.rfftn(x, dim=(1,2,3))
        return x
    
    def _irfft(self,x):
        x = torch.fft.irfftn(x, dim=(1,2,3))
        x = torch.fft.fftshift(x, dim=(1,2,3))
        return x
        
    def __call__(self, M, Ms=None):
        if Ms:
            M = M * Ms
        (_,nx,ny,nz) = M.shape
        px1, px2 = padding_width(nx, self.shape[0])
        py1, py2 = padding_width(ny, self.shape[1])
        pz1, pz2 = padding_width(nz, self.shape[2])
        M = F.pad(M, (pz1,pz2,py1,py2,px1,px2), 'constant', 0.)
        M = self._rfft(M)
        Ak = torch.cross(M, self.kernel, axis=0)
        A = const.mu_0 * self._irfft(Ak)
        return A
    
def compute_curl(A, cellsize):
    dAx_dy = (torch.roll(A[0], shifts=1, dims=1) - A[0]) / cellsize
    dAx_dz = (torch.roll(A[0], shifts=1, dims=2) - A[0]) / cellsize
    
    dAy_dx = (torch.roll(A[1], shifts=1, dims=0) - A[1]) / cellsize
    dAy_dz = (torch.roll(A[1], shifts=1, dims=2) - A[1]) / cellsize
    
    dAz_dx = (torch.roll(A[2], shifts=1, dims=0) - A[2]) / cellsize
    dAz_dy = (torch.roll(A[2], shifts=1, dims=1) - A[2]) / cellsize
    
    # Compute curl as the cross product of the gradient and the vector field
    curl_x = dAz_dy - dAy_dz
    curl_y = dAx_dz - dAz_dx
    curl_z = dAy_dx - dAx_dy
    
    # Stack the components to get the curl vector field
    curl_A = torch.stack((curl_x, curl_y, curl_z), dim=0)
    
    return curl_A