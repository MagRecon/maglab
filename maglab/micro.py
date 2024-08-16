import torch
import torch.nn as nn
import scipy.constants as const
import os
import numpy as np
from .microfields import Exch, DMI, Anistropy, Zeeman, InterfacialDMI, CubicAnistropy
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
        self.geo = nn.Parameter(Ms.abs() > 1e-3, requires_grad=False)
        
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
        # geo[geo.abs() > tol] = 1.
        # geo[geo.abs() <= tol] = 0.
        geo = geo.abs() > tol
        return geo
        
    def save_state(self, file_path, Ms=None):
        state = {}
        state['shape'] = self.shape
        state['dx'] = self.dx
        state['Ms'] = self.Ms.detach().clone()
        state['spherical'] = self.spherical.detach().clone()
        state['pbc'] = self.pbc
        state['interactions'] = self.get_interactions()
        dirname = os.path.dirname(file_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(state, file_path)
    
    @classmethod
    def load_state(cls, state, init_interactions=True):
        if isinstance(state, str):
            state = torch.load(state)
        
        if 'geo' in state: #old version
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
                'CubicAnistropy': 'add_cubic_anis'
                }
            if isinstance(interactions, list):
                for x in interactions:
                    name = x.pop('classname')
                    getattr(micro, set_interaction_dict[name])(**x)
            else: #old version
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
    

    def add_exch(self, A, ):
        self.interactions.append(Exch(self.shape, self.dx,  A, self.pbc, ))
        
    def add_dmi(self, D, ):
        if isinstance(D, (np.ndarray, torch.Tensor)):
            self.interactions.append(DMI(self.shape, self.dx,  D, self.pbc, ))
        elif isinstance(D, float):
            self.interactions.append(DMI(self.shape, self.dx,  (D,D,D), self.pbc, ))
        elif isinstance(D, tuple) and len(D)==3:
            self.interactions.append(DMI(self.shape, self.dx,  D, self.pbc, ))
        else:
            raise ValueError('D should be a float, a tuple of 3 floats, or a tensor with shape (3,nx,ny,nz).')
        
    def add_anis_dmi(self, D, ):
        self.interactions.append(DMI(self.shape, self.dx,  (-D,D,0), self.pbc, ))
        
    def add_interfacial_dmi(self, D, ):
        self.interactions.append(InterfacialDMI(self.shape, self.dx,  D, self.pbc, ))
        
    def add_demag(self, ):
        self.interactions.append(DeMag(*self.shape, self.dx, ))
        
    def add_anis(self, ku, anis_axis=(0,0,1),):
        self.interactions.append(Anistropy(self.shape, self.dx, ku, anis_axis, ))     
        
    def add_cubic_anis(self, kc, axis1=(1,0,0),axis2=(0,1,0), ):
        self.interactions.append(CubicAnistropy(self.shape, self.dx, kc, axis1, axis2, ))    


    def add_zeeman(self, H, ): 
        self.interactions.append(Zeeman(self.shape, self.dx, H, ))
        
        
    def remove_interaction(self, i_name):
        self.interactions = nn.ModuleList([
            i for i in self.interactions
            if i.__class__.__name__ != i_name
        ])
        
    def get_interactions(self,):
        interactions = []
        for i in self.interactions:
            interactions.append(i.get_params())
        return interactions

    def loss(self, spin=None, unit=const.eV):
        if spin is None:
            spin = self.get_spin()
        E = self.get_energy_density(spin)
        loss_m = self.dx**3 * torch.mean(E)     
        return 1/unit * loss_m
    
    def get_energy_density(self, spin):
        E = 0.
        if len(self.interactions) > 0:      
            for i in self.interactions:
                E = E + i(spin, self.geo, self.Ms)
        return E
    
    def get_total_energy(self, spin):
        energy_density = self.get_energy_density(spin)
        return torch.sum(energy_density) * self.dx**3
    
    def get_field_from_loss(self, loss, spin, create_graph=False):
        field = torch.autograd.grad(loss, spin, create_graph=create_graph)[0]
        field[:, self.geo] *= -1/(const.mu_0 * self.Ms[self.geo])
        field[:, ~self.geo] *= 0
        return field
    
    def get_total_field(self, spin, create_graph=False):
        E = self.get_energy_density(spin)
        total_energy = torch.sum(E)
        return self.get_field_from_loss(total_energy, spin, create_graph)
    
    def get_tau(self, spin):
        H = self.get_total_field(spin)
        tau = torch.cross(H, spin, dim=0)
        return tau    
    
    def get_energy_list(self):
        # for testing purposes
        # return a list of dicts, each of which is like 
        # {'classname': i.__class__.__name__, 'value': value}
        with torch.no_grad():
            spin = self.get_spin()
            energy_list = []
            for i in self.interactions:
                E =  self.dx**3 * i(spin, self.geo, self.Ms)
                item = {'classname': i.__class__.__name__, 'value': E}
                energy_list.append(item)
        return energy_list
    
    def get_field_list(self):
        # for testing purposes
        field_list = []
        if len(self.interactions) > 0:      
            for i in self.interactions:
                spin = self.get_spin()
                E = i(spin, self.geo, self.Ms)
                energy = torch.sum(E)
                field = self.get_field_from_loss(energy, spin)
                item = {'classname': i.__class__.__name__, 'value': field.detach().clone()}
                field_list.append(item)
        return field_list
    
    
    def get_spin(self,):
        m = self.geo * Spherical2Cartesian(self.spherical)
        return m
    
    def get_mag(self,):
        return self.Ms * self.get_spin()

        