from torch.utils.data import Dataset,DataLoader,Sampler
from scipy.ndimage import zoom
import numpy as np
import torch
import logging
import math

from .preprocess import add_Gaussian

logger = logging.getLogger(__name__)

class PhaseMap(torch.nn.Module):
    def __init__(self, data, tilt_angle, tilt_axis, mask=None, binary_mask=True, mask_tol=1e-20):
        super().__init__()
        self.register_buffer('data', self._tensor(data))
        
        if mask is None:
            mask = torch.ones_like(self.data)
        else:
            mask = self._tensor(mask)
            
        if binary_mask:
            mask = mask.abs() > mask_tol
            
        self.register_buffer('mask', mask)
      
        self.tilt_angle = float(tilt_angle)
        self.tilt_axis = tilt_axis
    
    def _tensor(self, array):
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu()
        elif isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        else:
            raise ValueError("Need a torch.Tensor or numpy.ndarray.")
        
        if len(array.shape) == 3 and array.shape[0] == 1:
            array = array[0,]
        elif len(array.shape) == 2:
            pass
        else:
            raise ValueError("Need 2D array(or with a single channel)")
        
        return array.float()
    
    def zoom(self, nx, ny):
        (x,y) = self.data.shape
        data = zoom(self.data, (nx/x, ny/y), order=2)
        (x,y) = self.mask.shape
        mask = zoom(self.mask, (nx/x, ny/y), order=0)
        return PhaseMap(data, mask, self.tilt_angle, self.tilt_axis)
       
    def remove_background(self, ):
        data = self.data - torch.mean(self.data)
        return PhaseMap(data, self.tilt_angle, self.tilt_axis, mask=self.mask)
    
    def add_Gaussian(self, mean=0., sigma=0.05, by='max_diff', seed=None):
        data = add_Gaussian(self.data.cpu().numpy(), mean, sigma, by, seed)
        return PhaseMap(data, self.tilt_angle, self.tilt_axis, mask=self.mask)

class PhaseSet(Dataset):
    def __init__(self, dtype=float):
        super().__init__()
        self.dtype = dtype
        self.list_x = []
        self.list_y = []
                      
    def __len__(self):
        return len(self.list_x) + len(self.list_y)
    
    def __getitem__(self, index,):
        if index < len(self.list_x):
            return self.list_x[index]
        else:
            return self.list_y[index - len(self.list_x)]

    def sort(self,):
        self.list_x.sort(key=lambda x: x.tilt_angle)
        self.list_y.sort(key=lambda x: x.tilt_angle)
           
    def load(self, phasemap, ang_tol=1e-1):
        if not isinstance(phasemap, PhaseMap):
            raise ValueError("PhaseSet.load need a PhaseMap")
            
        if phasemap.tilt_axis == 0:
            self._load_data(self.list_x, phasemap, ang_tol, 0)
        elif phasemap.tilt_axis == 1:
            self._load_data(self.list_y, phasemap, ang_tol, 1)
        else:
            raise ValueError("Can only load PhaseMap with tilt_axis 0 or 1")
            
    def _load_data(self, phase_list, phasemap, ang_tol, tilt_axis):
        for phi in phase_list:
            if abs(phi.tilt_angle - phasemap.tilt_angle) < ang_tol:
                logger.debug("PhaseMap with (Angle {:.2f} and Axis {:d}) already exsit, replacing original data."\
                    .format(phasemap.tilt_angle, tilt_axis))
                
                phase_list.remove(phi)
                phase_list.append(phasemap)
                return   
        
        logger.debug("Loading PhaseMap with (Angle {:.2f} and Axis {:d})"
              .format(phasemap.tilt_angle, tilt_axis))    
        phase_list.append(phasemap) 
            
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

# fill in the last batch if the sample size can not be divided by batch size
class MyBatchSampler(Sampler):
    def __init__(self, n, batch_size):
        self.n = n
        self.batch_size = batch_size
    
    def _random_choice(self, list_a, num):
        perm = torch.randperm(len(list_a)).tolist()
        idx = perm[:num]
        return [list_a[x] for x in idx]
        
    def __iter__(self):
        indices = torch.randperm(self.n).tolist()

        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        last_batch_size = len(batches[-1]) 
        if last_batch_size < self.batch_size:
            remain_size = self.n - last_batch_size
            remain_indices = indices[:remain_size]
            additional_indices = self._random_choice(remain_indices, self.batch_size-last_batch_size)  
            batches[-1].extend(additional_indices)

        return iter(batches)

    def __len__(self):
        return math.ceil(self.n / self.batch_size)
       
class PhaseLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        if 'collate_fn' not in kwargs:
            kwargs['collate_fn'] = self._default_collate_fn
        
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 4
        
        if 'shuffle' in kwargs:
            del kwargs['shuffle']   
            
        dataset = args[0]
        n = len(dataset)
        batch_sampler = MyBatchSampler(n, kwargs['batch_size'])
        kwargs['batch_sampler'] = batch_sampler
        del kwargs['batch_size']
        super().__init__(*args, **kwargs)
        
    def _default_collate_fn(self, batch):
        # data = torch.stack([x.data.unsqueeze(0) for x in batch])
        # mask = torch.stack([x.mask.unsqueeze(0) for x in batch])
        data = torch.stack([x.data for x in batch])
        mask = torch.stack([x.mask for x in batch])
        ang = [x.tilt_angle for x in batch]
        axis = [x.tilt_axis for x in batch]
        return data,mask,ang,axis

        