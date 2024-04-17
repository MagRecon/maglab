import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.constants as const

__all__ = ['Exch', 'DMI', 'Zeeman', 'Anistropy', 'InterfacialDMI', 'MicroField']

class MicroField(nn.Module): 
    def forward(self):
        pass
    
class Exch(MicroField):
    def __init__(self, A, cellsize, pbc=False, save_energy=False):
        super().__init__()
        self.A = nn.Parameter(torch.tensor([A]), requires_grad=False)
        self.cellsize = cellsize
        self.pbc = pbc
        self.save_energy = save_energy
        
    def forward(self, x, geo, Ms=None):   
        if not self.pbc:
            x = F.pad(x, (1, 1, 1, 1, 1, 1), 'constant', 0)
            geo = F.pad(geo, (1, 1, 1, 1, 1, 1), 'constant', 0)
            
        f = torch.zeros_like(x)
        for i in range(1,4):
            f = f + (torch.roll(x, shifts=(1), dims=(i)) - x) * torch.roll(geo, shifts=(1), dims=(i-1))
            f = f + (torch.roll(x, shifts=(-1), dims=(i)) - x) * torch.roll(geo, shifts=(-1), dims=(i-1))

        E = torch.sum(f * x, axis=0)
        
        if not self.pbc:
            E = E[1:-1,1:-1,1:-1]
        
        E = -self.A * self.cellsize * E
        
        if self.save_energy:
            self.E = E.detach().clone()
                
        loss = torch.mean(E)
        return loss
    
    def get_params(self,):
        return {'A': self.A.item()}
    
class DMI(MicroField):
    # positive D -> left chiral
    def __init__(self, D, cellsize, pbc=False, save_energy=False):
        super().__init__()
        self.cellsize2 = cellsize**2
        self.D = nn.Parameter(torch.tensor([D]), requires_grad=False)
        self.pbc = pbc
        self.save_energy = save_energy
        
    def forward(self, x, geo, Ms=None):
        if not self.pbc:
            x = F.pad(x,(1,1,1,1,1,1), 'constant', 0.)
            
        d1 = torch.cross(x, torch.roll(x, shifts=(1), dims=(1)), dim=0)[0,]
        d2 = -1 * torch.cross(x, torch.roll(x, shifts=(-1), dims=(1)), dim=0)[0,]
        d3 = torch.cross(x, torch.roll(x, shifts=(1), dims=(2)), dim=0)[1,]
        d4 = -1 * torch.cross(x, torch.roll(x, shifts=(-1), dims=(2)), dim=0)[1,]
        d5 = torch.cross(x, torch.roll(x, shifts=(1), dims=(3)), dim=0)[2,]
        d6 = -1 * torch.cross(x, torch.roll(x, shifts=(-1), dims=(3)), dim=0)[2,]
        E = d1+d2+d3+d4+d5+d6
        
        if not self.pbc:
            E = E[1:-1,1:-1,1:-1]
        
        E = 0.5 * self.D * self.cellsize2 * E    
        
        if self.save_energy:
            self.E = E.detach().clone()
            
        loss = torch.mean(E)
        return loss
    
    def get_params(self,):
        return {'D': self.D.item()}
    
class InterfacialDMI(MicroField):
    # positive D -> left chiral
    def __init__(self, D, cellsize, pbc=False, save_energy=False):
        super().__init__()
        self.cellsize2 = cellsize**2
        self.D = nn.Parameter(torch.tensor([D]), requires_grad=False)
        self.pbc = pbc
        self.save_energy = save_energy
        
    def forward(self, x, geo, Ms=None):
        if not self.pbc:
            x = F.pad(x,(1,1,1,1,1,1), 'constant', 0.)
            
        d1 = torch.cross(x, torch.roll(x, shifts=(1), dims=(1)), dim=0)[1,]
        d2 = -1 * torch.cross(x, torch.roll(x, shifts=(-1), dims=(1)), dim=0)[1,]
        d3 = -1 * torch.cross(x, torch.roll(x, shifts=(1), dims=(2)), dim=0)[0,]
        d4 = torch.cross(x, torch.roll(x, shifts=(-1), dims=(2)), dim=0)[0,]
        E = d1+d2+d3+d4
        
        if not self.pbc:
            E = E[1:-1,1:-1,1:-1]
        
        E = 0.5 * self.D * self.cellsize2 * E    
        
        if self.save_energy:
            self.E = E.detach().clone()
            
        loss = torch.mean(E)
        return loss
    
    def get_params(self,):
        return {'D': self.D.item()}
    
    
class Anistropy(MicroField):
    def __init__(self, Ku, cellsize, axis=(0,0,1), save_energy=False):
        super().__init__()
        assert len(axis) == 3, "axis should be tuple of 3 numbers"
        axis_norm = np.sqrt(axis[0]**2+axis[1]**2+axis[2]**2)
        self.axis_item = tuple([x/axis_norm for x in axis])
        axis_np = np.array(self.axis_item)
        axis_np = np.expand_dims(axis_np, axis=(1,2,3)).astype(float)
        self.register_buffer('axis', torch.tensor(axis_np, requires_grad=False).float())
        self.dV = cellsize**3
        self.Ku = nn.Parameter(torch.tensor([Ku]), requires_grad=False)
        self.save_energy = save_energy
        
    def forward(self, x, geo, Ms=None):
        mh = torch.sum(x*self.axis, axis=0)
        E = self.Ku * self.dV * (1 - torch.pow(mh, 2)) * geo
        
        if self.save_energy:
            self.E = E.detach().clone()
            
        loss = torch.mean(E)        
        return loss
    
    def get_params(self,):
        return {'Ku': self.Ku.item(),
                'axis': self.axis_item}
    
class Zeeman(MicroField):
    def __init__(self, H, cellsize, save_energy=False):
        super().__init__()
        assert isinstance(H, tuple), "H should be a tuple of 3 float"
        H_array = np.array(list(H))
        H = np.expand_dims(H_array, axis=(1,2,3))
        self.dV = cellsize**3
        self.H = nn.Parameter(torch.tensor(H), requires_grad=False)
        self.save_energy = save_energy

    def forward(self, x, geo, Ms=None):
        if not Ms:
            Ms = 1
            
        E = -1 * const.mu_0 * Ms * self.dV * torch.sum(x*self.H, axis=0)
        
        if self.save_energy:
            self.E = E.detach().clone()
            
        loss = torch.mean(E)
        return loss
    
    def get_params(self,):
        H_numpy = self.H.detach().cpu().numpy()[:,0,0,0]
        H_tuple = tuple([H_numpy[0], H_numpy[1], H_numpy[2]])
        return {'H': H_tuple}
    
    
    
