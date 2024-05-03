from .helper import padding_into
import torch
import scipy.constants as const
import numpy as np
from .phasemapper import projection
from scipy.ndimage import gaussian_filter

__all__ = ['LTEM']

class LTEM(torch.nn.Module):
    """
    This is a PyTorch-based reimplementation of the 'Microscope' class from PyLorentz.

    Attributes:
        E (float): Accelerating voltage (V). Default 200kV.
        C_s (float): Spherical aberration (m). Default 1mm.
        C_a (float): 2-fold astigmatism (m). Default 0.
        phi_a (float): 2-fold astigmatism angle (rad). Default 0.
        theta_c (float): Beam coherence (rad). Default 0.6mrad.
    """
    def __init__(self, fov, dx=5e-9, E=200e3,
                 C_s=1e-3, C_a=0, phi_a=0, theta_c=6e-4,):
        super().__init__()
        self.fov = fov
        self.dx = dx
        self.C_s = C_s
        self.C_a = C_a
        self.phi_a = phi_a
        self.E = E
        self.lam = self.get_lam(E)
        self.theta_c = theta_c
        self.aperture = 1.0
        self._register_grid()
    
    def get_lam(self, E):
        """Calculate the wavelength of an electron using the accelerating voltage.
        Args:
            E (float): Accelerating voltage (V)

        Returns:
            lam (float): Electron wavelength (m)
        """
        ce = const.elementary_charge
        me = const.electron_mass
        cc = const.speed_of_light
        E_e = me*cc**2 + E * ce
        c1 = E_e ** 2 - (me*cc**2)**2
        p = np.sqrt(c1/cc**2)
        lam = const.Planck / p
        return lam
    
    def _register_grid(self):
        nx, ny = self.fov
        kx, ky= torch.fft.fftfreq(nx, self.dx), torch.fft.fftfreq(ny, self.dx)
        kx, ky = torch.meshgrid(kx, ky, indexing='ij')
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
        self.register_buffer('k2', kx**2+ky**2)
    
    def get_intensity(self, amp, phase, df=0, spread=120e-9):
        """Generates a Fresnel image using given amplitude and phase data.

        Args:
            amp(torch.Tensor): A 2d tensor of size (n, n) containing amplitude values.
            phase(torch.Tensor): A 2d tensor of size (n, n) containing phase values.
            df(float): defocus(m).
            spread(float): defocus spread(m).

        Returns:
            img(torch.Tensor): A 2D tensor of size (n, n) representing the Fresnel image.
        """
        obj_wave = self.get_obj_wave(amp, phase)
        obj_wave_f = torch.fft.fft2(torch.fft.ifftshift(obj_wave, dim=(0,1)))
        t_f = self.get_microscope_transfer_function_f(df, spread)
        img_wave_f = obj_wave_f * t_f
        img_wave = torch.fft.ifft2(img_wave_f)
        img = torch.abs(img_wave) ** 2
        return torch.fft.fftshift(img, dim=(0,1))
    
    #TODO: setAperture
    def setAperture(self, sz):
        pass
    
    def get_obj_wave(self, amp, phase, ):
        (nx,ny) = amp.shape
        if not (nx == self.fov[0] and ny == self.fov[1]):
            amp = padding_into(amp, self.fov)
            
        return amp * torch.exp(1j*phase)
    
    def get_thickness_info(self, micro, angle, axis, padding=True):
        geo = micro.geo
        if padding:
            nx = self.fov[0]
            geo = padding_into(geo, (nx,nx,nx))
        proj = projection(geo, angle, axis) * micro.cellsize
        return proj

    def get_amplitude(self, thickness, eta_0, filter=True):
        """Get amplitude of objective wave.

        Args:
            thickness (torch.Tensor): A 2d tensor of size (n, n) containing thickness values.
            eta_0 (float): absorption constant of the sample

        Returns:
            amp(torch.Tensor): A 2d tensor of size (n, n) containing amplitude values.
        """
        amp = torch.exp(-1*thickness/eta_0)
        if filter:
            amp = gaussian_filter(amp.detach().cpu().numpy(), sigma=2)
            amp = torch.tensor(amp).cuda()
   
        return amp
        
    def get_microscope_transfer_function_f(self, df, spread):
        a_f = self.aperture
        chi_f = self.get_phase_transfer_function_f(df)
        g_f = self.get_damping_envelope_f(df, spread)
        t_f = torch.exp(-1j * chi_f) * torch.exp(-1 * g_f)
        return t_f
    
    def get_phase_transfer_function_f(self, df):
        k2 = self.k2
        k4 = k2 ** 2  
        c1 = torch.pi * self.lam * (df + self.C_a * np.cos(2*self.phi_a)) * k2
        c2 = torch.pi/2 * self.C_s * self.lam**3 * k4
        chi_f = -1*c1 + c2
        return chi_f

    def get_damping_envelope_f(self, df, spread):
        k1 = torch.sqrt(self.k2)
        k3 = self.k2 * k1
        k4 = self.k2 ** 2
        
        u = 1 + 2 * (torch.pi * self.theta_c * spread)**2 * self.k2
        c0 = torch.pi**2 * self.theta_c**2 / (self.lam**2 * u)
        c1 = self.C_s * self.lam**3 * k3
        c2 = df * self.lam * k1
        c3 = (torch.pi * self.lam * spread)**2 * k4 / (2*u)
        g_f = c0*(c1-c2)**2 - c3
        return g_f
    