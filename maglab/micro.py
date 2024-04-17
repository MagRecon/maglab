import torch
import torch.nn as nn
import scipy.constants as const
import numpy as np
import numbers
from .loss import Exch, DMI, Anistropy, Zeeman, DeMagPBC, DeMag, InterfacialDMI
from .helper import vector_to_angles

__all__ = ['Micro']

class Micro(nn.Module):
    def __init__(self, geo, cellsize, pbc=False):
        super().__init__()
        self.shape = geo.shape 
        
        geo = self._tensor(geo)
        geo[geo.abs() > 1e-3] = 1
        self.geo = nn.Parameter(geo, requires_grad=False)
        
        self.cellsize = cellsize
        self.pbc = pbc
        self.angles = nn.Parameter(torch.zeros((2,*self.shape)), requires_grad=True)
        self.interactions = nn.ModuleList()
        self.Ms = None
        
    @classmethod    
    def _tensor(cls, x, ):
        if isinstance(x, torch.Tensor):
            x = x.detach().clone()
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        else:
            raise ValueError("Only accept torch.tensor or np.ndarray!")

        return x.float()
        
    @classmethod
    def m2geo(cls, mag):
        mag = cls._tensor(mag)
        mag.requires_grad = False
        
        geo = torch.sum(mag**2, dim=0)
        geo[geo.abs() > 1e-3] = 1.
        geo[geo.abs() <= 1e-3] = 0.
        return geo
        
    def save_state(self, file_path, Ms=None):
        state = {}
        state['angles'] = self.angles.detach().clone()
        state['geo'] = self.geo.detach().clone()
        state['cellsize'] = self.cellsize
        state['pbc'] = self.pbc
        state['interactions'] = self.get_interactions()
        state['Ms'] = Ms
        torch.save(state, file_path)
    
    @classmethod
    def load_state(cls, state, init_interactions=True):
        if isinstance(state, str):
            state = torch.load(state)
        
        micro = cls(state['geo'], state['cellsize'], state['pbc'])
        micro.init_m0(state['angles'])
        micro.Ms = state['Ms']

        if init_interactions :
            interactions = state['interactions']
            set_interaction_dict = {
                'Exch':'add_exch', 
                'DMI':'add_dmi', 
                'DeMag':'add_demag',
                'Anistropy':'add_anis',
                'Zeeman':'add_zeeman',
                'InterfacialDMI': 'add_interfacial_dmi',
                }
            for key in interactions:
                getattr(micro, set_interaction_dict[key],)(**interactions[key])
            
        return micro.cuda()
    
    def init_m0(self, x): 
        if isinstance(x, tuple):
            m = torch.zeros((3,*self.shape))
            m[0],m[1],m[2] = x[0],x[1],x[2]
            x = vector_to_angles(m)
        else:
            x = self._tensor(x)
            if x.shape[0] == 3:
                x = vector_to_angles(x)
        self.angles.data.copy_(x)
    
    def init_m0_random(self, seed=None):
        if seed:
            torch.manual_seed(seed)
            
        self.angles[0,].data.copy_(torch.rand(self.shape) * torch.pi)
        self.angles[1,].data.copy_(torch.rand(self.shape) * 2*torch.pi)    
        
    def set_requires_grad(self, requires_grad):
        self.angles.requires_grad_(requires_grad)
       
    def add_exch(self, A, save_energy=False):
        self.interactions.append(Exch(A, self.cellsize,  self.pbc,save_energy))
        
    def add_dmi(self, D, save_energy=False):
        self.interactions.append(DMI(D,self.cellsize, self.pbc,save_energy))
        
    def add_interfacial_dmi(self, D, save_energy=False):
        self.interactions.append(InterfacialDMI(D,self.cellsize, self.pbc,save_energy))
        
    def add_demag(self, save_energy=False):
        if self.pbc:
            demag = DeMagPBC
        else:
            demag = DeMag
            
        self.interactions.append(demag(*self.shape, self.cellsize, save_energy))
        
    def add_anis(self, Ku, axis=(0,0,1),save_energy=False):
        self.interactions.append(Anistropy(Ku, self.cellsize, axis,save_energy))        

    def add_zeeman(self, H, save_energy=False): 
        self.interactions.append(Zeeman(H, self.cellsize, save_energy))
        
    def remove_interaction(self, i_name):
        self.interactions = nn.ModuleList([
            i for i in self.interactions
            if i.__class__.__name__ != i_name
        ])
        
    def get_interactions(self,):
        params = {}
        for i in self.interactions:
            params[i.__class__.__name__] = i.get_params()
        return params
    
    def get_energy(self):
        res = {}
        for i in self.interactions:
            res[i.__class__.__name__] = i.energy
        return res
    
    def get_m(self,):
        mx = torch.sin(self.angles[0]) * torch.cos(self.angles[1])
        my = torch.sin(self.angles[0]) * torch.sin(self.angles[1])
        mz = torch.cos(self.angles[0])
        M = self.geo * torch.stack([mx,my,mz])
        return M
   
    def loss(self, Ms=None, unit=const.eV):
        loss_m = 0.
        if len(self.interactions) > 0:
            m = self.get_m()            
            for i in self.interactions:
                loss_m = loss_m + i(m, self.geo, Ms)
                
        return 1/unit * loss_m
                
        