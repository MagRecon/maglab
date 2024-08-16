import torch
import numpy as np
import skimage
import torch.nn.functional as F

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


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    x = torch.arange(-size // 2 + 1, size // 2 + 1)
    y = torch.arange(-size // 2 + 1, size // 2 + 1)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    kernel = (1 / (2 * np.pi * sigma ** 2)) * torch.exp(-torch.sum(xy_grid ** 2, dim=-1) / (2 * sigma ** 2))
    kernel = kernel / torch.sum(kernel)
    return kernel

def compute_sigma(ker_size):
    return 0.3*((ker_size-1)*0.5 - 1) + 0.8

def generate_gauss_kernel(ksize, batch_size):
    # ksize ~ image size
    sigma = compute_sigma(ksize)
    kernel = gaussian_kernel(ksize, sigma)
    kernel = kernel.repeat(batch_size, 1, 1, 1).cuda()
    kernel.requires_grad = True
    return kernel

def remove_lowf_signal(data: torch.Tensor, kernel, ksize) -> torch.Tensor:
    blurred = F.conv2d(data, kernel, padding=ksize // 2)
    result = data - blurred
    return result.squeeze(), blurred.squeeze()
