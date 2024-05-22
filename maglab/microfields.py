import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.constants as const

from .helper import to_tensor, init_scalar, init_vector

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
    def __init__(self, shape, dx, A, pbc:str, save_energy=False):
        super().__init__()
        self.shape = shape
        self.A = nn.Parameter(init_scalar(A, self.shape), requires_grad=False)
        self.dx = dx
        self.pbc = pbc
        self.save_energy = save_energy
        self._init_pbc(pbc)
        
    def forward(self, x, Ms):   
        x = F.pad(x, self.padding, 'constant', 0)
        Ms = F.pad(Ms, self.padding, 'constant', 0)
        geo = Ms > 1e-3 #tolerence
            
        f = torch.zeros_like(x)
        for i in range(1,4):
            f = f + (torch.roll(x, shifts=(1), dims=(i)) - x) * torch.roll(geo, shifts=(1), dims=(i-1))
            f = f + (torch.roll(x, shifts=(-1), dims=(i)) - x) * torch.roll(geo, shifts=(-1), dims=(i-1))

        E = torch.sum(f * x, axis=0)
        
        E = F.pad(E, self.crop, 'constant', 0)
        
        E = -self.A * self.dx * E
        
        if self.save_energy:
            self.E = E.detach().clone()
                
        loss = torch.mean(E)
        return loss
    
    def get_params(self,):
        return {'A': self.A.detach().copy()}
    
class DMI(MicroField):
    # positive D -> left chiral
    def __init__(self, shape, dx, D, pbc:str, save_energy=False):
        super().__init__()
        self.shape = shape
        self.dx2 = dx**2
        self.D = nn.Parameter(init_scalar(D, self.shape), requires_grad=False)
        self.pbc = pbc
        self.save_energy = save_energy
        self._init_pbc(pbc)
        
    def forward(self, x, Ms):
        x = F.pad(x, self.padding, 'constant', 0)
            
        d1 = torch.cross(x, torch.roll(x, shifts=(1), dims=(1)), dim=0)[0,]
        d2 = -1 * torch.cross(x, torch.roll(x, shifts=(-1), dims=(1)), dim=0)[0,]
        d3 = torch.cross(x, torch.roll(x, shifts=(1), dims=(2)), dim=0)[1,]
        d4 = -1 * torch.cross(x, torch.roll(x, shifts=(-1), dims=(2)), dim=0)[1,]
        d5 = torch.cross(x, torch.roll(x, shifts=(1), dims=(3)), dim=0)[2,]
        d6 = -1 * torch.cross(x, torch.roll(x, shifts=(-1), dims=(3)), dim=0)[2,]
        E = d1+d2+d3+d4+d5+d6
        
        E = F.pad(E, self.crop, 'constant', 0)
        
        E = 0.5 * self.D * self.dx2 * E    
        
        if self.save_energy:
            self.E = E.detach().clone()
            
        loss = torch.mean(E)
        return loss
    
    def get_params(self,):
        return {'D': self.D.detach().copy()}
    
class InterfacialDMI(MicroField):
    # positive D -> left chiral
    def __init__(self, shape, dx, D, pbc:str, save_energy=False):
        super().__init__()
        self.shape = shape
        self.dx2 = dx**2
        self.D = nn.Parameter(init_scalar(D, self.shape), requires_grad=False)
        self.save_energy = save_energy
        self._init_pbc(pbc)
        
    def forward(self, x, Ms):
        x = F.pad(x, self.padding, 'constant', 0)
            
        d1 = torch.cross(x, torch.roll(x, shifts=(1), dims=(1)), dim=0)[1,]
        d2 = -1 * torch.cross(x, torch.roll(x, shifts=(-1), dims=(1)), dim=0)[1,]
        d3 = -1 * torch.cross(x, torch.roll(x, shifts=(1), dims=(2)), dim=0)[0,]
        d4 = torch.cross(x, torch.roll(x, shifts=(-1), dims=(2)), dim=0)[0,]
        E = d1+d2+d3+d4
        
        E = F.pad(E, self.crop, 'constant', 0)
        
        E = 0.5 * self.D * self.dx2 * E    
        
        if self.save_energy:
            self.E = E.detach().clone()
            
        loss = torch.mean(E)
        return loss
    
    def get_params(self,):
        return {'D': self.D.detach().copy()}
    
    
class Anistropy(MicroField):
    def __init__(self, shape, dx, ku, anis_axis, save_energy=False):
        super().__init__()
        self.shape = shape
        self.dV = dx**3
        self.ku = nn.Parameter(init_scalar(ku, self.shape), requires_grad=False)
        self.anis_axis = nn.Parameter(init_vector(anis_axis, self.shape, normalize=True), requires_grad=False)
        self.save_energy = save_energy
        
    def forward(self, x, Ms, ):
        geo = Ms > 1e-3
        mh = torch.sum(x*self.anis_axis, axis=0)
        E = self.ku * self.dV * (1 - torch.pow(mh, 2)) * geo
        
        if self.save_energy:
            self.E = E.detach().clone()
            
        loss = torch.mean(E)        
        return loss
    
    def get_params(self,):
        return {'ku': self.ku.detach().copy(),
                'anis_axis': self.anis_axis.detach().copy()}
    
class Zeeman(MicroField):
    def __init__(self, shape, dx, H, save_energy=False):
        super().__init__()
        self.shape = shape
        self.dV = dx**3
        self.H = nn.Parameter(init_vector(H, self.shape), requires_grad=False)
        self.save_energy = save_energy

    def forward(self, x, Ms):   
        E = -1 * const.mu_0 * Ms * self.dV * torch.sum(x*self.H, axis=0)
        
        if self.save_energy:
            self.E = E.detach().clone()
            
        loss = torch.mean(E)
        return loss
    
    def get_params(self,):
        return {'H': self.H.detach().copy()}
    
    
    