import torch
import torch.nn as nn
import scipy.constants as const
import os
from .microfields import Exch, DMI, Anistropy, Zeeman, InterfacialDMI
from .demag import DeMag
from .helper import Cartesian2Spherical, Spherical2Cartesian, to_tensor, init_scalar
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

__all__ = ['Micro']


class Micro(nn.Module):
    def __init__(self, nx, ny, nz, dx=5e-9, pbc:str=""):
        """Initiate Micro class.

    Args:
        nx (int): Dimension in the x-axis.
        ny (int): Dimension in the y-axis.
        nz (int): Dimension in the z-axis.
        dx (float, optional): Grid spacing in meters. Defaults to 5e-9.
        pbc (str, optional): Periodic boundary condition flag. If periodic boundary condition in the x-dimension is required, set pbc="x". Defaults to "".
    """
        super().__init__()
        self.shape = (nx, ny, nz)
        self._init_Ms()
        self.dx = dx   
        self.spherical = nn.Parameter(torch.zeros((2,*self.shape)), requires_grad=True)
        self.interactions = nn.ModuleList()
        self.pbc = pbc
        
    def set_Ms(self, Ms):        
        Ms = init_scalar(Ms, self.shape)
        self.Ms = nn.Parameter(Ms, requires_grad=False)
        self.geo = nn.Parameter((Ms>1e-3).float(), requires_grad=False)
        
    def _init_Ms(self):
        self.set_Ms(1/const.mu_0)
            
    @classmethod
    def m2geo(cls, mag, tol=1e-3):
        """Return geometry array by magnetization array.

    This method converts a given magnetization array into a binary geometry array.
    The geometry array indicates the presence (1) or absence (0) of magnetization
    based on a specified tolerance.

    Args:
        mag (np.array or torch.Tensor): Magnetization array, with shape of (3, nx, ny, nz).
        tol (float, optional): Critical value to judge if Ms is 0 or 1. Defaults to 1e-3.

    Returns:
        torch.Tensor: Binary array that contains the geometry information, with shape of (nx, ny, nz).
    """
        mag = to_tensor(mag)
        mag.requires_grad = False
        
        geo = torch.sum(mag**2, dim=0)
        geo[geo.abs() > tol] = 1.
        geo[geo.abs() <= tol] = 0.
        return geo
        
    def save_state(self, file_path, Ms=None):
        state = {}
        state['shape'] = self.shape
        state['dx'] = self.dx
        state['Ms'] = self.Ms.detach().clone()
        state['spherical'] = self.spherical.detach().clone()
        state['pbc'] = self.pbc
        state['interactions'] = self.get_interactions()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(state, file_path)
    
    @classmethod
    def load_state(cls, state, init_interactions=True):
        if isinstance(state, str):
            state = torch.load(state)
        
        if 'geo' in state:
            geo = state['geo']
            nx, ny, nz = geo.shape
            micro = cls(nx, ny, nz, state['cellsize'])
            micro.set_Ms(state['Ms']*geo)
            micro.init_m0(state['angles'])
        else:
            nx, ny, nz = state['shape']
            micro = cls(nx, ny, nz, state['dx'], state['pbc'])
            micro.set_Ms(state['Ms'])
            micro.init_m0(state['spherical'])


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
            x = Cartesian2Spherical(m)
            self.spherical.data.copy_(x[1:,])
        elif x.shape[0] == 3:
            x = to_tensor(x)
            x = Cartesian2Spherical(x)
            self.spherical.data.copy_(x[1:,])
        elif x.shape[0] == 2:
            self.spherical.data.copy_(x)
    
    def init_m0_random(self, seed=None):
        if seed:
            torch.manual_seed(seed)
            
        self.spherical[0,].data.copy_(torch.rand(self.shape) * torch.pi)
        self.spherical[1,].data.copy_(torch.rand(self.shape) * 2*torch.pi)    
        
    def set_requires_grad(self, requires_grad):
        self.spherical.requires_grad_(requires_grad)
       
    def add_exch(self, A, save_energy=False):
        self.interactions.append(Exch(self.shape, self.dx,  A, self.pbc, save_energy))
        
    def add_dmi(self, D, save_energy=False):
        self.interactions.append(DMI(self.shape, self.dx,  D, self.pbc, save_energy))
        
    def add_interfacial_dmi(self, D, save_energy=False):
        self.interactions.append(InterfacialDMI(self.shape, self.dx,  D, self.pbc, save_energy))
        
    def add_demag(self, save_energy=False):
        self.interactions.append(DeMag(*self.shape, self.dx, save_energy))
        
    def add_anis(self, ku, anis_axis=(0,0,1),save_energy=False):
        self.interactions.append(Anistropy(self.shape, self.dx, ku, anis_axis, save_energy))        

    def add_zeeman(self, H, save_energy=False): 
        self.interactions.append(Zeeman(self.shape, self.dx, H, save_energy))
        
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
    
    def get_field(self):
        for i in self.interactions:
            i.save_energy = True
            
        loss = micro.loss()
        H = 0.
        for i in self.interactions:
            H += i.field
        return H
    
    def effective_field(self, interaction):
        if isinstance(interaction, DeMag) or isinstance(interaction, Zeeman):
            return interaction.field
        
        nxyz = self.shape[0] * self.shape[1] * self.shape[2]
        dV = self.dx**3
        ms_inv = -1/(const.mu_0 * self.Ms)
        ms_inv[self.geo == 0] = 0
        return nxyz / dV * ms_inv * interaction.field
    
    def get_spin(self,):
        m = self.geo * Spherical2Cartesian(self.spherical)
        return m
    
    def get_mag(self,):
        return self.Ms * self.get_spin
   
    def loss(self, unit=const.eV):
        loss_m = 0.
        if len(self.interactions) > 0:
            spin = self.get_spin()        
            for i in self.interactions:
                loss_m = loss_m + i(spin, self.Ms)
                
        return 1/unit * loss_m
                
        