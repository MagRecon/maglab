import numpy as np
import torch
    
def cylider(dia, height):
    radius = dia/2
    volumn = np.ones((dia,dia, height),dtype=float)
    xc,yc = dia//2, dia//2
    cx, cz = np.linspace(0, dia, dia, endpoint=False), np.linspace(0, height, height, endpoint=False)
    x,y,z = np.meshgrid(cx,cx,cz,indexing='ij')
    r = np.sqrt((x-xc)**2+(y-yc)**2)
    volumn[r>radius] = 0
    return torch.tensor(volumn)


def cubiod(nx,ny,nz):
    return torch.ones((nx,ny,nz))

def sphere(diameter):
    radius = diameter/2
    volumn = np.ones((diameter,diameter, diameter),dtype=float)
    rc = diameter//2
    cx, cz = np.linspace(0, diameter, diameter, endpoint=False), np.linspace(0, diameter, diameter, endpoint=False)
    x,y,z = np.meshgrid(cx,cx,cz,indexing='ij')
    r = np.sqrt((x-rc)**2+(y-rc)**2+(z-rc)**2)
    volumn[r>radius] = 0
    return torch.tensor(volumn)