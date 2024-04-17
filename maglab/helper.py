from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

def vector_to_angles(m):
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