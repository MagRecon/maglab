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
    #phi = np.unwrap(phi.cpu().numpy())
    #phi = torch.from_numpy(phi)
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

def SphericalUnitVectors(spherical):
    theta, phi = spherical[0,], spherical[1,]
    st, ct = torch.sin(theta), torch.cos(theta)
    sp, cp = torch.sin(phi), torch.cos(phi)
    
    e_rho = torch.stack([st * cp, st * sp, ct])
    e_theta = torch.stack([ct * cp, ct * sp, -1*st])
    e_phi =  torch.stack([-1*sp, cp, torch.zeros_like(sp)])
    return e_rho, e_theta, e_phi

def cartesian_grad(spherical):
    with torch.no_grad():
        spherical_grad = spherical.grad
        theta, phi = spherical[0,], spherical[1,]
        e_rho, e_theta, e_phi = SphericalUnitVectors(spherical) #(3,nx,ny,nz)
        grad_theta = spherical_grad[0,].unsqueeze(0).expand_as(e_theta) #(nx,ny,nz)
        grad_phi = spherical_grad[1,].unsqueeze(0).expand_as(e_theta)
        grad_m = grad_theta * e_theta + torch.sin(theta) * grad_phi * e_phi
        return grad_m
    
def l2_vector(m):
    return torch.sqrt(torch.sum(m**2, dim=0))

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


def normalize_tuple(v):
    v = np.array(list(v))
    n = np.sqrt(np.sum(v**2, axis=0))
    v = tuple([x/n for x in v])
    return v

def init_vector(v, shape, normalize=False, atol=1e-5):
    if isinstance(v, numbers.Number):
        v_t = v * torch.zeros((3, *shape))
        
    elif isinstance(v, tuple):
        dims = len(v)
        v_t = torch.zeros((dims, *shape))
        for i in range(dims):
            v_t[i,] = v[i]

    elif isinstance(v, (np.ndarray, torch.Tensor)):
        assert v.shape[1:] == shape, "Shape not match! Expect {}, got {} instead.".format(shape, v.shape)
        v_t = to_tensor(v)
        
    else:
        raise ValueError("Function `init_vector` only accept one of (tuple, np.ndarray, torch.Tensor), got `{}` instead"
                            .format(v.__class__.__name__))
    
    if normalize:
        norm = torch.sqrt(torch.sum(v_t**2, dim=0))
        geo = norm > atol
        v_t[:, geo] = v_t[:, geo] / norm[geo]
        
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

def cross_correlation(x, y):
    return torch.fft.ifft2(torch.fft.fft2(x) * torch.fft.fft2(y).conj())

def coords(nx):
    return np.linspace(-nx/2, nx/2, nx, endpoint=True)

def Euler_x(alpha):
    st, ct = np.sin(alpha), np.cos(alpha)
    return torch.tensor([[1, 0, 0],
                       [0, ct, -1*st],
                       [0, st, ct]])
    
def Euler_y(beta):
    st, ct = np.sin(beta), np.cos(beta)
    return torch.tensor([[ct, 0, st],
                       [0, 1, 0],
                       [-1*st, 0, ct]])

def Euler_z(gamma):
    st, ct = np.sin(gamma), np.cos(gamma)
    return torch.tensor([[ct, -1*st, 0],
                       [st, ct, 0],
                       [0, 0, 1]])
    
def Euler_XYZ(alpha, beta, gamma):
    return Euler_z(gamma) @ Euler_y(beta) @ Euler_x(alpha)


def generate_circular_mask(nx,ny, radius):
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
    vM = vB / const.mu_0
    return vM

def estimate_m0(phase, layer, dx):
    Ms = estimate_Ms(phase, layer, dx)
    B = get_induction(phase, dx)
    M0 = B / (layer*Ms)
    M = np.repeat(M0[:, :, :, np.newaxis], layer, axis=3)
    M = np.pad(M, ((0, 1), (0, 0), (0, 0), (0,0)), mode='constant', constant_values=0.)
    return M
