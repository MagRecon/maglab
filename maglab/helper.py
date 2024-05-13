from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from .const import c_m
import scipy.constants as const

def get_lam(E):
    """Calculate the wavelength of an electron using the accelerating voltage.
    Args:
        E (float): Accelerating voltage (V)

    Returns:
        lam (float): Electron wavelength (m)
    """
    ce = const.elementary_charge
    me = const.electron_mass
    cc = const.speed_of_light
    E_e = me*cc**2 + E * ce
    c1 = E_e ** 2 - (me*cc**2)**2
    p = np.sqrt(c1/cc**2)
    lam = const.Planck / p
    return lam

def partial_z(image_stack, defocus_seris):
    l, nx, ny = image_stack.shape
    deriv = np.zeros((nx,ny))
    I0 = np.zeros((nx,ny))
    for i in range(nx):
        a, b, c = np.polyfit(defocus_seris, image_stack[:,i,:], 2)
        deriv[i,:] = b
        I0[i,:] = c
    return deriv, I0

def get_induction(phase, dx):
    By = np.gradient(phase, axis=0) * 1/dx * 1/c_m
    Bx = -1 * np.gradient(phase, axis=1) * 1/dx * 1/c_m
    return np.array([Bx, By])

def Cartesian2Spherical(m):
    norm = torch.sqrt(m[0,]**2 + m[1,]**2 + m[2,]**2)
    m = m/(norm+1e-30)
    theta = torch.arccos(m[2,])
    phi = torch.arctan2(m[1,], m[0])
    return torch.stack([theta, phi])
    
def padding_width(n, N):
    right = (N-n)//2
    return (N-n-right, right)

def padding_into(x, new_shape):
    dims = len(new_shape)
    if not len(x.shape) == dims:
        raise ValueError("Padding dims not match!")
    
    pad_dims = []
    for i in range(dims-1, -1, -1):
        w1, w2 = padding_width(x.shape[i], new_shape[i])
        pad_dims.append(w1)
        pad_dims.append(w2)
        
    return F.pad(x, tuple(pad_dims), 'constant', 0)

def pad(x, dims, axes):
    if not len(dims) == len(axes):
        raise ValueError("length of dims must equal to axes")
    
    odim = len(x.shape)    
    owidth = [x.shape[i] for i in axes]
    for (i, L) in enumerate(dims):
        if L < owidth[i]:
            raise ValueError("padding width not enough!")

    pad_list = []
    axe_list = list(axes)
    for i in range(odim):
        if i in axes:
            pad_list.append(padding_width(x.shape[i], dims[axe_list.index(i)]))
        else:
            pad_list.append((0,0))
    return np.pad(x, tuple(pad_list))


def crop_width(N, n):
    dn = N - n
    n1 = dn // 2
    if N%2 == 0 and n%2 == 1:
        n1 += 1
        
    n2 = n1 + n
    return (n1, n2)

def crop(x, dims, axes):
    if not len(dims) == len(axes):
        raise ValueError("length of dims must equal to axes")
    
    owidth = [x.shape[i] for i in axes]
    for (i, L) in enumerate(dims):
        if L > owidth[i]:
            raise ValueError("crop size too large!")

    tem = deepcopy(x)
    for (i, axis) in enumerate(axes):
        n1, n2 = crop_width(tem.shape[axis], dims[i])
        tem = np.moveaxis(tem, axis, 0)
        tem = tem[n1:n2,]
        tem = np.moveaxis(tem, 0, axis)
    return tem