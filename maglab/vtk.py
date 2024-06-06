import os

from pyevtk.hl import gridToVTK
import numpy as np 
from .coloring import spin_to_rgb

def cordns(N, d):
    return N*d*np.fft.fftshift(np.fft.fftfreq(N))

def write_vtk(filename, m, data_name= "M", dx=5e-9, save_rgb=True): 
    folder = os.path.dirname(filename)
    if len(folder) > 0:
        os.makedirs(folder, exist_ok=True)
        
    if len(m.shape) == 4:
        flag_vector = True
    elif len(m.shape) == 3:
        flag_vector = False
    else:
        raise ValueError("shape error!")

    if flag_vector:
        (_, nx, ny, nz) = m.shape
    else:
        (nx, ny, nz) = m.shape
        
    # Coordinates 
    X = cordns(nx, dx)
    Y = cordns(ny, dx)
    Z = cordns(nz, dx)
    
    if filename.endswith(".vtr") or filename.endswith(".vtk"):
        filename = filename[0:-4]
    
    data_dict = {}
    if flag_vector:
        mx = np.ascontiguousarray(m[0, ...], dtype='float64')
        my = np.ascontiguousarray(m[1, ...], dtype='float64')
        mz = np.ascontiguousarray(m[2, ...], dtype='float64')
        data_dict[data_name] = (mx, my, mz)
        if save_rgb:
            rgb = spin_to_rgb(m)
            rgb = np.moveaxis(rgb, -1, 0)
            r = np.ascontiguousarray(rgb[0, ...], dtype='float64')
            g = np.ascontiguousarray(rgb[1, ...], dtype='float64')
            b = np.ascontiguousarray(rgb[2, ...], dtype='float64')
            data_dict['rgb'] = (r,g,b)
    else:
        m = np.ascontiguousarray(m, dtype='float64')
        data_dict[data_name] = (m)
        
    gridToVTK(filename, X, Y, Z, pointData=data_dict)