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
    def __init__(self, shape, dx, A, pbc:str,):
        """
        Initializes a new instance of the class.

        Args:
            shape (tuple): The shape of the tensor.
            dx (float): The spacing between adjacent elements in the tensor.
            A (float or Tensor): The exchange constant. If a float, it is broadcasted to the shape of the tensor.
            pbc (str): A string specifying the periodic boundary conditions.
            save_energy (bool, optional): Whether to save the energy. Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.shape = shape
        self.A = nn.Parameter(init_scalar(A, self.shape), requires_grad=False)
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
    # positive D -> left chiral
    def __init__(self, shape, dx, D_vector, pbc:str, ):
        super().__init__()
        self.shape = shape
        self.dx = dx
        #self.D = nn.Parameter(init_scalar(D, self.shape), requires_grad=False)
        self.pbc = pbc
        
        self._init_pbc(pbc)
        self._init_D(D_vector)
        
    def _init_D(self, D_vector):
        D = init_vector(D_vector, self.shape)
        D = F.pad(D, self.padding, 'constant', 0)
        self.D = nn.Parameter(D, requires_grad=False)
        
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
        self.shape = shape
        self.dx = dx
        self.D = nn.Parameter(init_scalar(D, self.shape), requires_grad=False)
        
        self._init_pbc(pbc)
        
    def forward(self, spin, geo, Ms):
        x = F.pad(spin, self.padding, 'constant', 0)
            
        d1 = torch.cross(x, torch.roll(x, shifts=(1), dims=(1)), dim=0)[1,]
        d2 = -1 * torch.cross(x, torch.roll(x, shifts=(-1), dims=(1)), dim=0)[1,]
        d3 = -1 * torch.cross(x, torch.roll(x, shifts=(1), dims=(2)), dim=0)[0,]
        d4 = torch.cross(x, torch.roll(x, shifts=(-1), dims=(2)), dim=0)[0,]
        E = d1+d2+d3+d4
        
        E = F.pad(E, self.crop, 'constant', 0)
        
        E = 0.5 * self.D * self.dx**2 * E    
        
        E = E / self.dx**3
            
        return E
    
    def get_params(self,):
        return {'classname': self.__class__.__name__,
            'D': self.D.detach().clone()}
    
    
class Anistropy(MicroField):
    def __init__(self, shape, dx, ku, anis_axis, ):
        super().__init__()
        self.shape = shape
        self.dx = dx
        self.ku = nn.Parameter(init_scalar(ku, self.shape), requires_grad=False)
        self.axis_tuple = anis_axis
        self.anis_axis = nn.Parameter(init_vector(anis_axis, self.shape, normalize=True), requires_grad=False)
        
        
    def forward(self, spin, geo, Ms, ):
        mh = torch.sum(spin*self.anis_axis, axis=0)
        E = self.ku * (1 - torch.pow(mh, 2)) * geo
                    
        return E
    
    def get_params(self,):
        return {'classname': self.__class__.__name__,
            'ku': self.ku.detach().clone(),
                'anis_axis': self.axis_tuple}
        
class CubicAnistropy(MicroField):
    def __init__(self, shape, dx, kc, axis1, axis2, ):
        super().__init__()
        self.shape = shape
        self.dx = dx
        self.kc = nn.Parameter(init_scalar(kc, self.shape), requires_grad=False)
        axis3 = tuple(np.cross(np.array(axis1), np.array(axis2)))
        self.axes = [axis1, axis2]
        self.axis1 = nn.Parameter(init_vector(axis1, self.shape, normalize=True), requires_grad=False)
        self.axis2 = nn.Parameter(init_vector(axis2, self.shape, normalize=True), requires_grad=False)
        self.axis3 = nn.Parameter(init_vector(axis3, self.shape, normalize=True), requires_grad=False)
        
        
    def forward(self, spin, geo, Ms, ):
        m_axis1 = torch.sum(spin*self.axis1, axis=0) ** 4
        m_axis2 = torch.sum(spin*self.axis2, axis=0) ** 4
        m_axis3 = torch.sum(spin*self.axis3, axis=0) ** 4
        E = -1 * self.kc * (m_axis1+m_axis2+m_axis3) * geo
                    
        return E
    
    def get_params(self,):
        return {'classname': self.__class__.__name__,
                'kc': self.kc.detach().clone(),
                'axis1': self.axes[0],
                'axis2': self.axes[1]}
    
class Zeeman(MicroField):
    def __init__(self, shape, dx, H, ):
        super().__init__()
        self.shape = shape
        self.dx = dx
        self.H_tuple = H
        self.H = nn.Parameter(init_vector(H, self.shape), requires_grad=False)
        

    def forward(self, spin, geo, Ms):   
        E = -1 * const.mu_0 * Ms * torch.sum(spin*self.H, axis=0)
        return E
    
    def get_params(self,):
        return {'classname': self.__class__.__name__,
                'H': self.H_tuple}
    


    
    
    
