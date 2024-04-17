import os

from pyevtk.hl import gridToVTK
import numpy as np 

def cordns(N, d):
    return N*d*np.fft.fftshift(np.fft.fftfreq(N))

def write_vtk(filename, m, data_name= "M", cellsize=5e-9): 
    folder = os.path.dirname(filename)
    if not os.path.isdir(folder):
        print(f"making path: {folder}")
        os.makedirs(folder)
        
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
    X = cordns(nx, cellsize)
    Y = cordns(ny, cellsize)
    Z = cordns(nz, cellsize)
    
    if filename.endswith(".vtr") or filename.endswith(".vtk"):
        filename = filename[0:-4]
    
    if flag_vector:
        mx = np.ascontiguousarray(m[0, ...], dtype='float64')
        my = np.ascontiguousarray(m[1, ...], dtype='float64')
        mz = np.ascontiguousarray(m[2, ...], dtype='float64')
        gridToVTK(filename, X, Y, Z, pointData={data_name:(mx,my,mz)})
    else:
        m = np.ascontiguousarray(m, dtype='float64')
        gridToVTK(filename, X, Y, Z, pointData={data_name:(m)})