import numpy as np
import warnings
from .helper import partial_z, get_lam

__all__ = ['TIE']

class TIE:
    def __init__(self, image_size, dx, E=200e3, qc=0.):
        self.image_size = image_size
        self.dx = dx
        self.lam = get_lam(E)
        self.init_q_grid()
        self.qc = qc
        
    def init_q_grid(self):
        qx = 2*np.pi*np.fft.fftfreq(self.image_size[0], self.dx)
        qy = 2*np.pi*np.fft.fftfreq(self.image_size[1], self.dx)
        self.QX, self.QY = np.meshgrid(qx, qy, indexing='ij')
        self.Q2 = self.QX ** 2 + self.QY ** 2
            
    def get_qi(self):
        if self.qc > 0:
            qi = self.Q2 / (self.Q2 + self.qc**2) ** 2 
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                qi = 1/self.Q2
                qi[0,] = 0
        return qi
        
    def inverse_laplacian_q(self, x):
        x_q = self._fft(x)
        il_x_q = -1 * x_q * self.get_qi()
        il_x_q[0, 0] = 0.
        return il_x_q
    
    def _fft(self, x):
        return np.fft.fft2(np.fft.ifftshift(x, axes=(0,1)))
    
    def _ifft(self, x):
        return np.fft.fftshift(np.fft.ifft2(x), axes=(0,1))
    
    def get_phase_transfer_function_f(self, df, C_a, phi_a, C_s):
        c1 = np.pi * self.lam * (df + C_a * np.cos(2*phi_a)) * self.Q2
        c2 = np.pi/2 * C_s * self.lam**3 * self.Q2**2
        chi_f = -1*c1 + c2
        return chi_f
    
    def __call__(self, image_series, defocus_series, C_a=0., phi_a=0., C_s=0.):
        """
        Args:
            image_series (np.ndarray): A three-dimensional array with shape (L, Nx, Ny), where L represents 
                                   the number of images, Nx and Ny are the dimensions of each image.
            defocus_series (list): A list of defocus values, each corresponding to an image in `image_series`.

        The following arguments are used to construct the phase transfer function, which propagates the phase 
        from the backfocal plane to the object plane:
            C_s (float): Spherical aberration (m). Default 1mm.
            C_a (float): 2-fold astigmatism (m). Default 0.
            phi_a (float): 2-fold astigmatism angle (rad). Default 0.
        """
        pzI0, I0 = partial_z(image_series, defocus_series)
        inv_ll = self.inverse_laplacian_q(pzI0)
        f1_q = 1j* self.QX * inv_ll
        f2_q = 1j* self.QY * inv_ll
        f1 = self._ifft(f1_q)
        f2 = self._ifft(f2_q)
        for (i,df) in enumerate(defocus_series):
            if df == 0.:
                I0 = image_series[i]
                
        f1_I = f1 / I0
        f2_I = f2 / I0
        f1_I_q = self._fft(f1_I)
        f2_I_q = self._fft(f2_I)
        f_q = 1j*(self.QX *f1_I_q + self.QY *f2_I_q)  * self.get_qi()
        f_q[0,0] = 0.
        f = self._ifft(f_q).real
        transfer_func = self.get_phase_transfer_function_f(0., C_a, phi_a, C_s)
        f = f + transfer_func # back propagate wave
        return -2 * np.pi / self.lam * f