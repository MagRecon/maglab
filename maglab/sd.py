# Ref: Exl L, et al. Journal of Applied Physics, 2014, 115(17).

import torch
import torch.nn as nn

class SteepestDescent(nn.Module):
    def __init__(self, shape, maxtau=1., mintau=1e-10):
        super().__init__()
        self.shape = shape
        self.prespin = nn.Parameter(torch.zeros(3,*shape), requires_grad=False)
        self.g = nn.Parameter(torch.zeros(3,*shape), requires_grad=False)
        self.step = 0
        self.max_tau = maxtau
        self.min_tau = mintau
                      
    def _tau1(self, dspin, dg):
        dspin_dspin = torch.sum(dspin**2)
        dspin_dg = torch.sum(dspin*dg)
        return torch.abs(dspin_dspin / dspin_dg)
        
    def _tau2(self, dspin, dg):
        dg_dg = torch.sum(dg**2)
        dspin_dg = torch.sum(dspin*dg)
        return torch.abs(dspin_dg / dg_dg) 
    
    def _get_fg(self, spin, field):
        f = torch.cross(spin, field, dim=0)
        g = torch.cross(spin, f, dim=0)
        return f,g
        
    def _new_spin(self, spin, f, g):
        n1 = torch.sum(f**2, dim=0)
        factor = 0.25 * n1 * self.tau**2
        m = (1-factor)*spin - self.tau*g
        new_spin = m / (1+factor)
        return new_spin
        
    def __call__(self, spin, field):
        with torch.no_grad():
            if self.step == 0:
                f,g = self._get_fg(spin, field)
                self.tau = self.min_tau
            else:  
                f,g = self._get_fg(spin, field)
                dspin = spin - self.prespin
                dg = g - self.g

                if self.step % 2 == 0:
                    tau = self._tau1(dspin, dg)
                else:
                    tau = self._tau2(dspin, dg)
                if torch.isnan(tau) or abs(tau) < self.min_tau:
                    tau = self.min_tau
                elif abs(tau) > self.max_tau:
                    tau = self.max_tau
                self.tau = tau
                
            self.step += 1
            self.prespin.copy_(spin)
            self.g.copy_(g)
            new_spin = self._new_spin(spin, f, g)
        return new_spin