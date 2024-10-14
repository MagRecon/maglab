import numpy as np
import torch
import warnings
    
def cylider(dia, height):
    radius = dia/2
    volumn = np.zeros((dia,dia, height),dtype=float)
    xc,yc = dia/2, dia/2
    cx, cz = np.linspace(0, dia, dia, endpoint=True), np.linspace(0, height, height, endpoint=True)
    x,y,z = np.meshgrid(cx,cx,cz,indexing='ij')
    r = np.sqrt((x-xc)**2+(y-yc)**2)
    volumn[r<=radius] = 1.
    return torch.tensor(volumn)

def mesh(nx,ny,nz):
    x,y,z = np.arange(nx), np.arange(ny), np.arange(nz)
    xc, yc, zc = nx//2+nx%2, ny//2+ny%2, nz//2+nz%2
    return np.meshgrid(x-xc, y-yc, z-zc, indexing='ij')

def cubiod(nx,ny,nz):
    return torch.ones((nx,ny,nz))

def sphere(radius):
    diameter = 2*radius
    volumn = np.ones((diameter,diameter, diameter),dtype=float)
    rc = diameter//2
    cx, cz = np.linspace(0, diameter, diameter, endpoint=False), np.linspace(0, diameter, diameter, endpoint=False)
    x,y,z = np.meshgrid(cx,cx,cz,indexing='ij')
    r = np.sqrt((x-rc)**2+(y-rc)**2+(z-rc)**2)
    volumn[r>radius] = 0
    return torch.tensor(volumn)

def color_wheel_sphere(radius):
    diameter = 2*radius
    x = np.arange(diameter) - radius
    X,Y,Z = np.meshgrid(x,x,x,indexing='ij')
    R = np.sqrt(X**2+Y**2+Z**2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mx,my,mz = X/R, Y/R, Z/R
    m = np.stack((mx,my,mz))
    m = np.nan_to_num(m, 0.)
    return m
    