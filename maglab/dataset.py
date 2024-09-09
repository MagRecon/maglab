from torch.utils.data import Dataset,DataLoader,Sampler
from scipy.ndimage import zoom
import numpy as np
import torch
import logging
import math

from .preprocess import add_Gaussian
from .alignment import shift_array

logger = logging.getLogger(__name__)

class PhaseMap(torch.nn.Module):
    def __init__(self, data, alpha=0., beta=0., gamma=0., mask=None, binary_mask=True, mask_tol=1e-20):
        super().__init__()
        self.register_buffer('data', self._tensor(data))
        
        if mask is None:
            mask = torch.ones_like(self.data)
        else:
            mask = self._tensor(mask)
            
        if binary_mask:
            mask = mask.abs() > mask_tol
            
        self.register_buffer('mask', mask)
      
        self.Euler = (alpha, beta, gamma)
    
    @classmethod
    def _tensor(self, array):
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu()
        elif isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        else:
            raise ValueError("Need a torch.Tensor or numpy.ndarray, got {} instead".format(type(array)))
        
        if len(array.shape) == 3 and array.shape[0] == 1:
            array = array[0,]
        elif len(array.shape) == 2:
            pass
        else:
            raise ValueError("Need 2D array(or with a single channel)")
        
        return array.float()
    
    def shift(self, shifts):
        data = shift_array(self.data, shifts)
        mask = shift_array(self.mask, shifts)
        return PhaseMap(data, *self.Euler, mask)
    
    def zoom(self, nx, ny):
        (x,y) = self.data.shape
        data = zoom(self.data, (nx/x, ny/y), order=2)
        (x,y) = self.mask.shape
        mask = zoom(self.mask, (nx/x, ny/y), order=0)
        return PhaseMap(data, *self.Euler, mask=mask)
       
    def remove_background(self, ):
        mask_data = self.data * self.mask
        background = torch.sum(mask_data) / torch.sum(self.mask)
        data = self.data.clone()
        data[self.mask] -= background
        data[~self.mask] = 0.
        return PhaseMap(data, *self.Euler, mask=self.mask)
    
    def add_Gaussian(self, mean=0., sigma=0.05, by='max_diff', seed=None):
        data = add_Gaussian(self.data.cpu().numpy(), mean, sigma, by, seed)
        return PhaseMap(data, *self.Euler, mask=self.mask)

class PhaseSet(Dataset):
    def __init__(self, dtype=float):
        super().__init__()
        self.dtype = dtype
        self.datalist = []
                      
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, index,):
        return self.datalist[index]

    def sort(self,):
        self.datalist.sort(key=lambda x: x.Euler)
           
    def load(self, phasemap, ang_tol=1e-1):
        if not isinstance(phasemap, PhaseMap):
            raise ValueError("PhaseSet.load need a PhaseMap")
        
        self._load_data(phasemap, ang_tol)

            
    def _load_data(self, phasemap, ang_tol):
        for phi in self.datalist:
            if np.allclose(np.array(phi.Euler), np.array(phasemap.Euler), atol=ang_tol):
                logger.info("PhaseMap with Euler{} already exsit, replacing original data."\
                .format(phasemap.Euler))
            
                self.datalist.remove(phi)
                self.datalist.append(phasemap)
                return 
        
        logger.info("Loading PhaseMap with Euler{}."\
                .format(phasemap.Euler))    
        self.datalist.append(phasemap)
        return   
        
            
    def remove_background(self,):
        newset = PhaseSet()
        for phasemap in self:
            phase = phasemap.remove_background()
            newset.load(phase)
        return newset
            
    def zoom(self, nx, ny):
        newset = PhaseSet()
        for phasemap in self:
            phase = phasemap.zoom(nx, ny)
            newset.load(phase)
        return newset

class PhaseLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        if 'collate_fn' not in kwargs:
            kwargs['collate_fn'] = self._default_collate_fn
        super().__init__(*args, **kwargs)
        
    def _default_collate_fn(self, batch):
        data = torch.stack([x.data for x in batch])
        mask = torch.stack([x.mask for x in batch])
        Eulers = [x.Euler for x in batch]
        return data,mask,Eulers
