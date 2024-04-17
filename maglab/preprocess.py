import torch
import numpy as np
import skimage

def add_Gaussian(phi, mean=0., sigma=0.05, by='max_diff', seed=None):
    #if by == 'max_diff':
    sigma = sigma * (np.max(phi)-np.min(phi))
    if seed is not None:
        np.random.seed(seed)
        
    return phi + np.random.normal(mean, sigma, size=phi.shape)

def butterworth(phi, cutoff=0.1):
    phi = skimage.filters.butterworth(phi, \
        cutoff_frequency_ratio=cutoff, \
        high_pass=False)
    return phi