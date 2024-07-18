import torch
import numpy as np
import torch.nn.functional as F
def mutual_correlation(x1, x2):
    nx,ny = x1.shape
    x1 = F.pad(x1, (nx//2, nx//2, ny//2, ny//2), mode='constant', value=0)
    x2 = F.pad(x2, (nx//2, nx//2, ny//2, ny//2), mode='constant', value=0)
    f1 = torch.fft.fft2(torch.fft.ifftshift(x1, dim=(0,1)))
    f2 = torch.fft.fft2(torch.fft.ifftshift(x2, dim=(0,1)))
    fg = f1 * torch.conj(f2)
    fg_n = fg / torch.sqrt(torch.abs(fg))
    fg_n = torch.nan_to_num(fg_n, nan=0.0)
    mcf = torch.fft.ifft2(fg_n).real
    return torch.fft.fftshift(mcf, dim=(0,1))[nx//2:-nx//2, ny//2:-ny//2]

def find_shift(x1, x2):
    corre = mutual_correlation(x1,x2)
    max_idx = torch.argmax(corre)
    max_idx = np.unravel_index(max_idx.cpu().numpy(), corre.shape)
    cx, cy = corre.shape[0]//2, corre.shape[1]//2
    return cx-max_idx[0], cy-max_idx[1]

def shift_array(x, shifts):
    y = torch.roll(x, shifts, dims=(0,1))
    if shifts[0] > 0:
        y[:shifts[0], :] = 0
    elif shifts[0] < 0:
        y[shifts[0]:, :] = 0
        
    if shifts[1] > 0:
        y[:, :shifts[1]] = 0
    elif shifts[1] < 0:
        y[:, shifts[0]:] = 0
    return y