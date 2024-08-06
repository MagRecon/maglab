import sys
import numpy as np
import matplotlib.pyplot as plt
from maglab import PhaseMapper
import maglab
from maglab.utils import show, show_list

import torch
import warnings
import unittest

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
    xs = np.linspace(-fov_x/2, fov_x/2, fov_x, endpoint=True)*dx
    ys = np.linspace(-fov_y/2, fov_y/2, fov_y, endpoint=True)*dx
    X, Y = np.meshgrid(xs,ys,indexing='ij')
    phi = phim_uniformly_magnetized_slab(mx, my, X, Y, Lx, Ly, Lz, Ms)
    return phi

class Test(unittest.TestCase):
    def setUp(self):
        theta = 5/3*np.pi
        mx,my=np.cos(theta), np.sin(theta)
        
        self.dims = (32,64,16)
        self.dx = 1e-9
        self.fov = 128
        self.Ms = 1e5
        self.phi_theory = phase_in_theory(mx, my, *self.dims,self.dx,self.dx,self.dx,self.fov,self.fov,self.Ms)
        print("estimated Ms: {:.2e} A/m".format(maglab.estimate_Ms(self.phi_theory, 16, 1e-9)))
        if torch.cuda.is_available:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        m = np.zeros((3,*self.dims))
        m[0,], m[1,] = mx,my
        self.m = torch.from_numpy(m).to(self.device)
        self.phasemapper = PhaseMapper(2*self.fov, self.dx, rotation_padding=100).to(self.device)
        
    def phase_from_A(self):
        m = self.m.clone()
        A = maglab.convert.m2A(self.m, self.Ms, 1e-9, (self.fov, self.fov, self.fov))
        phase = maglab.const.c_m * self.dx * torch.sum(A[2,], dim=2)
        return phase.detach().cpu().numpy()
        
    def test_phase(self):
        phi_simulate = self.phasemapper(self.m, theta=0., axis=0, Ms=self.Ms)
        phi_simulate = phi_simulate.detach().cpu().numpy()[0,self.fov//2:-self.fov//2, self.fov//2:-self.fov//2]
        phi_simulate = -1 * phi_simulate # we are using beam along z- direction, but the theory solution is using z+.
        phase_A = self.phase_from_A()
        show_list([self.phi_theory, 
                   phi_simulate, 
                   phase_A,
                   self.phi_theory - phi_simulate,
                   self.phi_theory-phase_A,], 
                    rows=2,titles=['theory', 'simulation', 'from_A',
                                    'diff_simu', 'diff_from_A'])
        
        print("mean error:", np.mean(np.abs(self.phi_theory - phi_simulate)) / np.mean(np.abs(self.phi_theory)))
        plt.savefig("phase.png", dpi=100)
        plt.close()
        plt.plot(self.phi_theory[:, self.fov//2], label='theory')
        plt.plot(phi_simulate[:, self.fov//2], ls='-.', label='simulation')
        plt.legend()
        plt.savefig("line.png", dpi=100)
        plt.close()
        
    def test_ltem(self):
        # amp= np.ones((128, 128))
        # nx, ny = self.dims[0], self.dims[1]
        # x0, y0 = (128-nx)//2, (128-ny)//2
        # amp[x0:x0+nx, y0:y0+ny] = 0.1
        thickness = np.zeros((128, 128))
        nx, ny = self.dims[0], self.dims[1]
        x0, y0 = (128-nx)//2, (128-ny)//2
        thickness[x0:x0+nx, y0:y0+ny] = 1e-9*self.dims[2]

        ltem = maglab.LTEM((128,128), dx=self.dx, C_s=0., theta_c=0.)
        amp = ltem.get_amplitude(torch.from_numpy(thickness), 1.)
        image = ltem(amp, 
                     torch.from_numpy(self.phi_theory),
                     df=5e-6).detach().cpu().numpy()
        
        show_list([amp.numpy(), self.phi_theory, image], titles=["Amp", "Phase", "Image"], same_colorbar=False)
        plt.savefig("LTEM_image.png", dpi=100)
        plt.close()
        
        
        
if __name__ == '__main__':
    unittest.main()