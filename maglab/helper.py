from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from .const import c_m
import scipy.constants as const
import warnings
import numbers

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
    m = m.clone()
    rho = torch.sqrt(m[0,]**2 + m[1,]**2 + m[2,]**2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = m/rho
        m[:,rho==0.] = 0.
    theta = torch.arccos(m[2,])
    phi = torch.arctan2(m[1,], m[0,])
    return torch.stack([rho, theta, phi])

def Spherical2Cartesian(p):
    if p.shape[0] == 3:
        rho, theta, phi =  p[0,], p[1,], p[2,]
    else:
        theta, phi = p[0,], p[1,]
        rho = 1.
        
    vx = torch.sin(theta) * torch.cos(phi)
    vy = torch.sin(theta) * torch.sin(phi)
    vz = torch.cos(theta)
    return rho * torch.stack([vx, vy, vz])

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().clone()
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        raise ValueError("Function `to_tensor` only accept torch.tensor or np.ndarray!")

    return x.float()

def init_scalar(v, shape):
    if isinstance(v, numbers.Number):
        v_t = v * torch.ones(shape)
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        assert v.shape == shape, "Shape not match! Expect {}, got {} instead.".format(shape, v.shape)
        v_t = to_tensor(v)
    else:
        raise ValueError("Function `init_scalar` only accept one of (Number, np.ndarray, torch.Tensor), got `{}` instead"
                            .format(v.__class__.__name__))
        
    return v_t

def tuple_to_vector(v:tuple, shape:tuple):
    dims = len(v)
    vec = np.zeros((dims, *shape))
    for i in dims:
        vec[i,] = v[i]
    return vec
    
def normalize_tensor(v, tol=1e-3):
    v_copy = v.detach().clone()
    n2 = torch.sum(v_copy**2, dim=0)
    n = torch.sqrt(n2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v_new = v_copy / n
    v_new[:, n<tol] = 0.
        
    return v_new 
        

def init_vector(v, shape, normalize=False):
    # shape = (dim, nx, ny, nz)
    if isinstance(v, tuple):
        dims = len(v)
        v_t = torch.zeros((dims, *shape))
        for i in range(dims):
            v_t[i,] = init_scalar(v[i], shape)

    elif isinstance(v, (np.ndarray, torch.Tensor)):
        assert v.shape == shape, "Shape not match! Expect {}, got {} instead.".format(shape, v.shape)
        v_t = to_tensor(v_t)
        
    else:
        raise ValueError("Function `init_vector` only accept one of (tuple, np.ndarray, torch.Tensor), got `{}` instead"
                            .format(v.__class__.__name__))
        
    if normalize:
        v_t = normalize_tensor(v_t)
        
    return v_t

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

def center_padding(x, new_shape):
    x = torch.fft.fftshift(x)
    x = padding_into(x, new_shape)
    x = torch.fft.ifftshift(x)
    return x

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