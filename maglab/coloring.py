import numpy as np
import matplotlib as mpl
import warnings
import matplotlib.pyplot as plt

__all__ = ['spin_to_hsv','spin_to_rgb', 'radial', 'colorwheel']

def spin_to_hsv(v):
    """
    <This function is copied from PyLorentz>
    Convert a 3D vector field to HSV color space.

    Parameters:
    v (numpy.ndarray): A 3D vector field of shape (3, ..., ...), where the first dimension
                       represents the x, y, and z components of the vectors.

    Returns:
    numpy.ndarray: An array of the same shape as the input, but with the last dimension
                   representing the HSV color space (hue, saturation, value).
    """
    if v.shape[0] == 3:
        vx, vy, vz = v[0,...], v[1, ...], v[2, ...]
    elif v.shape[0] == 2:
        vx, vy = v[0,...], v[1, ...]
        vn = np.sqrt(vx**2+vy**2)
        vmax = np.max(vn)
        vz = np.sqrt(vmax**2-vx**2-vy**2)
    else:
        raise ValueError("First channel number need to be 2 or 3.")
        
    n = np.sqrt(vx**2+vy**2+vz**2)
    v = v / np.max(n)
    phi = np.arctan2(vy, vx)
    phi = np.mod(phi, 2*np.pi)
    hue = phi / (2 * np.pi)
    
    nxy = np.sqrt(vx**2+vy**2)
    
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     sat = nxy / n
    #     value = (vz/n + 1.) / 2
        
    theta = np.arctan2(vz, np.sqrt(vx**2 + vy**2))
    #value = np.where(theta<0, 1-1/(1+np.exp(10*theta*2/np.pi+5)), 1)#sigmoid
    #sat = np.where(theta>0, 1-1/(1+np.exp(-10*theta*2/np.pi+5)), 1)#sigmoid
    value = np.where(theta < 0, np.cos(2 * theta) / 2 + 1 / 2, 1)
    sat = np.where(theta > 0, np.cos(2 * theta) / 2 + 1 / 2, 1)
    
    hue[n==0.] = 0.    
    sat[n==0.] = 0.
    value[n==0.] = 1.
    hsv = np.stack((hue, sat, value), axis=-1)
    return hsv

def spin_to_rgb(v):
    hsv = spin_to_hsv(v)
    return mpl.colors.hsv_to_rgb(hsv)

def radial(n):
    x = np.linspace(-1,1,n,endpoint=False)
    y = np.linspace(-1,1,n,endpoint=False)
    X,Y = np.meshgrid(x,y, indexing='ij')
    R2 = X**2+Y**2
    Z = np.sqrt(1-R2)
    r = np.stack((X,Y,Z))
    r[:,R2>=1] = 0.
    return r

def colorwheel(n=100):
    m0 = radial(n)
    rgb = spin_to_rgb(m0)
    rgb = np.transpose(rgb, (1,0,2))
    plt.imshow(rgb, origin='lower')
