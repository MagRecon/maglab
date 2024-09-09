import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.constants as const

from .helper import to_tensor, init_scalar, init_vector, normalize_tuple

__all__ = ['Exch', 'DMI', 'Zeeman', 'Anistropy', 'InterfacialDMI', 'MicroField']

class MicroField(nn.Module): 
    def forward(self):
        pass
    
    def _init_pbc(self, pbc:str):
        # if pbc, use roll without padding
        self.pbc_x = 0 if 'x' in pbc else 1
        self.pbc_y = 0 if 'y' in pbc else 1
        self.pbc_z = 0 if 'z' in pbc else 1
        self.padding = (self.pbc_z, self.pbc_z, self.pbc_y, self.pbc_y, self.pbc_x, self.pbc_x)
        self.crop = tuple(-1*v for v in self.padding)
       
        
class Exch(MicroField):
    def __init__(self, shape, dx, A, pbc:str,):
        """
        Initializes a new instance of the class.

        Args:
            shape (tuple): The shape of the tensor.
            dx (float): The spacing between adjacent elements in the tensor.
            A (float or Tensor): The exchange constant. If a float, it is broadcasted to the shape of the tensor.
            pbc (str): A string specifying the periodic boundary conditions.
            save_energy (bool, optional): Whether to save the energy. Defaults to False.
        """
        super().__init__()
        self.register_buffer('A', init_scalar(A, shape))
        self.dx = dx
        self.pbc = pbc
        
        self._init_pbc(pbc)
        
    def forward(self, spin, geo, Ms):   
        x = F.pad(spin, self.padding, 'constant', 0)
        geo = F.pad(geo, self.padding, 'constant', 0)
        f = 0.
        for i in range(3):
            f = f + (torch.roll(x, shifts=(1), dims=(i+1)) - x) * torch.roll(geo, shifts=(1), dims=(i))
            f = f + (torch.roll(x, shifts=(-1), dims=(i+1)) - x) * torch.roll(geo, shifts=(-1), dims=(i))
        f = -self.A * self.dx * F.pad(f, self.crop, 'constant', 0)
        E = torch.sum(f * spin, axis=0)          
        E = E / self.dx**3
        return E
    
    
    def get_params(self,):
        return {'classname': self.__class__.__name__,
            'A': self.A.detach().clone()}
    
class DMI(MicroField):
    # positive D -> left chirality
    def __init__(self, shape, dx, D_vector, pbc:str, ):
        super().__init__()
        self.dx = dx
        self.pbc = pbc
        self._init_pbc(pbc)
        
        D = init_vector(D_vector, shape)
        D = F.pad(D, self.padding, 'constant', 0)
        self.register_buffer('D', D)
        
    def forward(self, spin, geo, Ms):
        Dx, Dy, Dz = self.D[0,],self.D[1,],self.D[2,]
        x = F.pad(spin, self.padding, 'constant', 0)
            
        d1 = torch.cross(x, torch.roll(x, shifts=(1), dims=(1)), dim=0)[0,]
        d2 = -1 * torch.cross(x, torch.roll(x, shifts=(-1), dims=(1)), dim=0)[0,]
        d3 = torch.cross(x, torch.roll(x, shifts=(1), dims=(2)), dim=0)[1,]
        d4 = -1 * torch.cross(x, torch.roll(x, shifts=(-1), dims=(2)), dim=0)[1,]
        d5 = torch.cross(x, torch.roll(x, shifts=(1), dims=(3)), dim=0)[2,]
        d6 = -1 * torch.cross(x, torch.roll(x, shifts=(-1), dims=(3)), dim=0)[2,]
        E = Dx * (d1+d2) + Dy * (d3+d4) + Dz * (d5+d6)
        
        E = F.pad(E, self.crop, 'constant', 0)
        
        E = 0.5 * self.dx**2 * E    
        
        E = E / self.dx**3
            
        return E
    
    def get_params(self,):
        D = F.pad(self.D, self.crop, 'constant', 0)
        return {'classname': self.__class__.__name__,
            'D': D.detach().clone()}
    
class InterfacialDMI(MicroField):
    # positive D -> left chiral
    def __init__(self, shape, dx, D, pbc:str, ):
        super().__init__()
        self.dx = dx
        self._init_pbc(pbc)
        
        D = init_scalar(D, shape)
        D = F.pad(D, self.padding, 'constant', 0)
        self.register_buffer('D', D)
        
    def forward(self, spin, geo, Ms):
        x = F.pad(spin, self.padding, 'constant', 0)
            
        d1 = torch.cross(x, torch.roll(x, shifts=(1), dims=(1)), dim=0)[1,]
        d2 = -1 * torch.cross(x, torch.roll(x, shifts=(-1), dims=(1)), dim=0)[1,]
        d3 = -1 * torch.cross(x, torch.roll(x, shifts=(1), dims=(2)), dim=0)[0,]
        d4 = torch.cross(x, torch.roll(x, shifts=(-1), dims=(2)), dim=0)[0,]
        E = d1+d2+d3+d4
        E = 0.5 * self.D * self.dx**2 * E    
        
        E = F.pad(E, self.crop, 'constant', 0)
        E = E / self.dx**3
            
        return E
    
    def get_params(self,):
        D =  F.pad(self.D.detach().clone(), self.crop, 'constant', 0)
        return {'classname': self.__class__.__name__,
            'D': D}
    
    
class Anistropy(MicroField):
    def __init__(self, ku, anis_axis, normalize=True):
        super().__init__()
        self.ku = ku
        if normalize:
            anis_axis = normalize_tuple(anis_axis)
            
        self.axis_tuple = anis_axis
        self.register_buffer('axis', torch.tensor(anis_axis).view(3, 1, 1, 1))

        
    def forward(self, spin, geo, Ms, ):
        mh = (spin * self.axis).sum(dim=0)
        E = self.ku * (1 - torch.pow(mh, 2)) * geo
        return E
    
    def get_params(self,):
        return {'classname': self.__class__.__name__,
            'ku': self.ku,
                'anis_axis': self.axis_tuple}
        
class CubicAnistropy(MicroField):
    def __init__(self, kc, axis1, axis2, normalize=True):
        super().__init__()
        if normalize:
            axis1 = normalize_tuple(axis1)
            axis2 = normalize_tuple(axis2)
        self.axes = [axis1, axis2]
        
        axis3 = tuple(np.cross(np.array(axis1), np.array(axis2)))
        self.kc = kc
        self.register_buffer('axis1', torch.tensor(axis1).view(3, 1, 1, 1))
        self.register_buffer('axis2', torch.tensor(axis2).view(3, 1, 1, 1))
        self.register_buffer('axis3', torch.tensor(axis3).view(3, 1, 1, 1))
         
    def forward(self, spin, geo, Ms, ):
        mh1 = (spin * self.axis1).sum(dim=0)
        mh2 = (spin * self.axis2).sum(dim=0)
        mh3 = (spin * self.axis3).sum(dim=0)
        E = -1 * self.kc * (mh1**4 + mh2**4 + mh3**4) * geo    
        return E
    
    def get_params(self,):
        return {'classname': self.__class__.__name__,
                'kc': self.kc,
                'axis1': self.axes[0],
                'axis2': self.axes[1]}
    
class Zeeman(MicroField):
    def __init__(self, H):
        super().__init__()
        self.H_tuple = H
        self.register_buffer('H', torch.tensor(H).view(3, 1, 1, 1))
        
    def forward(self, spin, geo, Ms):   
        E = -1 * const.mu_0 * Ms * (spin * self.H).sum(dim=0)
        return E
    
    def get_params(self,):
        return {'classname': self.__class__.__name__,
                'H': self.H_tuple}
    


    
    
    
