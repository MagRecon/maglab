from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .helper import get_induction
from .const import mu_0

__all__ = ['show', 'show_list', 'estimate_Ms', 'estimate_m0','generate_circle_mask','get_meshgrid_3d']

def generate_circle_mask(nx,ny, radius):
    x = np.linspace(-nx/2, nx/2, nx, endpoint=True)
    y = np.linspace(-ny/2, ny/2, ny, endpoint=True)
    X,Y = np.meshgrid(x,y,indexing='ij')
    R2 = X**2 + Y**2
    mask = np.zeros((nx,ny))
    mask[R2 <= radius**2] =1
    return mask

def get_meshgrid_3d(nx,ny,nz,dx=1,dy=1,dz=1):
    x = np.linspace(-nx/2, nx/2, nx, endpoint=True)*dx
    y = np.linspace(-ny/2, ny/2, ny, endpoint=True)*dy
    z = np.linspace(-nz/2, nz/2, nz, endpoint=True)*dz
    return np.meshgrid(x, y, z, indexing='ij')

def estimate_Ms(phase, layer, dx, percentile=100):
    B = get_induction(phase, dx)
    Bxy = np.sqrt(B[0,]**2+B[1,]**2)
    vB = np.percentile(Bxy, percentile) / (layer*dx)
    vM = vB / mu_0
    return vM

def estimate_m0(phase, layer, dx):
    Ms = estimate_Ms(phase, layer, dx)
    B = get_induction(phase, dx)
    M0 = B / (layer*Ms)
    M = np.repeat(M0[:, :, :, np.newaxis], layer, axis=3)
    M = np.pad(M, ((0, 1), (0, 0), (0, 0), (0,0)), mode='constant', constant_values=0.)
    return M

def show(img, **kwargs):
    ax = plt.imshow(img.T, origin='lower', **kwargs)
    plt.colorbar()
    return ax
    
def show_list(fs, same_colorbar=True, cutoff=0, figsize=(-1, 5), titles=[], rows=1, **kwargs):
    l = len(fs)
    if l == 1:
        axes = plt.imshow(fs[0].T, origin='lower')
        return axes
    
    lt = len(titles)
    v1 = np.max(fs[0])
    v2 = np.min(fs[0])
    
    columns = l//rows
    if columns * rows < l:
        columns = columns + 1

    if figsize[0] < 0:
        figsize = (5 * columns, 5*rows)
        
    fig, ax = plt.subplots(rows, columns, figsize=figsize)
    for i in range(l):
        if cutoff > 0:
            s = (slice(cutoff, -1 * cutoff), slice(cutoff, -1 * cutoff))
        else:
            s = (slice(0, fs[i].shape[0]), slice(0, fs[i].shape[1]))
        
        if rows == 1:
            axes_index = i
        else:
            axes_index = (i // columns, i%columns)
            
        if same_colorbar:
            im = ax[axes_index].imshow(fs[i][s].T, vmax=v1, vmin=v2, origin='lower', **kwargs)
        else:
            im = ax[axes_index].imshow(fs[i][s].T, origin='lower', **kwargs)

        if i < lt:
            ax[axes_index].set_title(titles[i])

        divider = make_axes_locatable(ax[axes_index])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax[axes_index].set_xticks([])
        ax[axes_index].set_yticks([])
        
    return fig, ax
