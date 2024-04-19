import sys
import numpy as np
import matplotlib.pyplot as plt
from maglab import PhaseMapper
from maglab.const import mu_0
from maglab.utils import show, show_array

import torch
import warnings

device = torch.device("cuda")

def F0(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t1 = x*np.log(x**2+y**2) - 2*x
        t2 = 2*y*np.arctan(x/y)
    res = t1+t2
    res = np.where(y==0, t1, res)
    res = np.where((x==0)*(y==0), 0, res)
    return res

def phim_uniformly_magnetized_slab(mx, my, x, y, Lx, Ly, Lz, Ms):
    mu0 = 4*np.pi*1e-7
    Phi0 = 2.067833e-15
    coeff = mu0*Ms*Lz/(4*Phi0)
    a = F0(x-Lx/2, y-Ly/2) - F0(x+Lx/2, y-Ly/2) - F0(x-Lx/2, y+Ly/2) + F0(x+Lx/2, y+Ly/2)
    b = F0(y-Ly/2, x-Lx/2) - F0(y+Ly/2, x-Lx/2) - F0(y-Ly/2, x+Lx/2) + F0(y+Ly/2, x+Lx/2)
    return coeff*(my*b-mx*a)

def phase_in_theory(mx,my, nx,ny,nz, dx,dy,dz, fov_x,fov_y, Ms):
    Lx,Ly,Lz = nx*dx, ny*dy, nz*dz
    xs = (np.arange(fov_x)-fov_x//2)*dx
    ys = (np.arange(fov_y)-fov_y//2)*dy
    X, Y = np.meshgrid(xs,ys,indexing='ij')
    phi = phim_uniformly_magnetized_slab(mx, my, X, Y, Lx, Ly, Lz, Ms)
    return phi


theta = 5/3*np.pi
mx,my=np.cos(theta), np.sin(theta)
nx,ny,nz = 32,64,16
dx,dy,dz = 1e-9,1e-9,1e-9
fov = 128
fov2 = fov*2
Ms = 1e5
phi1 = phase_in_theory(mx,my,nx,ny,nz,dx,dy,dz,fov,fov,Ms)

m = np.zeros((3,nx,ny,nz))
m[0,], m[1,] = mx,my
phasemapper = PhaseMapper(2*fov, dx, rotation_padding=100).to(device)
phi2 = phasemapper(m, theta=0., axis=0, Ms=Ms)
phi2 = phi2.detach().cpu().numpy()[fov//2:-fov//2, fov//2:-fov//2]
phi2 = -1 * phi2 # we are using beam along z- direction
show_array([phi1, phi2, phi1-phi2], titles=['theory', 'simu', 'diff'])
print("mean error:", np.mean(np.abs(phi1-phi2)) / np.mean(np.abs(phi1)))
plt.savefig("phase.png", dpi=100)
plt.close()
plt.plot(phi1[:, fov//2], label='theory')
plt.plot(phi2[:, fov//2], ls='-.', label='simu')
plt.legend()
plt.savefig("line.png", dpi=100)