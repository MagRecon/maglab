from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .helper import get_induction
from .const import mu_0

__all__ = ['show', 'show_list']

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
