import maglab
import torch
import numpy as np
import unittest

sin, cos = np.sin, np.cos

class test1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass
    
    def create_micro(self, m0, Ms, pbc):
        dx = 1e-9
        nx,ny,nz = 11,11,11
        micro = maglab.Micro(nx, ny, nz, dx, pbc=pbc)
        micro.set_Ms(Ms)
        micro.add_exch(1e-12, save_energy=True)
        micro.add_dmi(1e-4, save_energy=True)
        micro.add_anis(1e3, anis_axis=(0.3,0.4,0.5), save_energy=True)
        micro.add_demag(save_energy=True)
        micro.add_zeeman((0,0,1e3), save_energy=True)
        micro.add_interfacial_dmi(2e-4, save_energy=True)
        micro.init_m0(m0)
        micro.cuda()
        return micro
            
    def tearDown(self):
        self.erergy_cubiod = None
        self.erergy_cylinder = None
        
    def maxl1(self, x):
        return torch.max(torch.abs(x)).item()
    
    def compare_energy(self, m0, Ms, e_jumag, field_jumag, pbc=""):
        micro = self.create_micro(m0, Ms, pbc)
        micro.loss()

        for (i, interaction) in enumerate(micro.interactions):
            energy = interaction.E.detach().cpu().numpy()
            energy_diff = e_jumag[i,] - energy
            error = self.maxl1(energy_diff) / self.maxl1(e_jumag[i,])
            print(interaction.__class__.__name__, error)
            self.assertTrue(error < 1e-5)

        for (i, interaction) in enumerate(micro.interactions):
            field = micro.effective_field(interaction).detach().cpu().numpy()
            field_diff = field_jumag[i,] - field
            error = self.maxl1(field_diff) / self.maxl1(field_jumag[i,])
            print(interaction.__class__.__name__, error)
            self.assertTrue(error < 1e-5)
        
    def test_cubiod(self):
        # m0 is filled with cubiod geo
        m0 = np.load("dataset/m0.npy")
        Ms = 8e5 * maglab.Micro.m2geo(m0)
        e_jumag = torch.from_numpy(np.load("dataset/energy.npy"))
        field_jumag = torch.from_numpy(np.load("dataset/field.npy"))
        self.compare_energy(m0, Ms, e_jumag, field_jumag)
    
    def test_cylinder(self):
        m0 = np.load("dataset/m0_cylinder.npy")
        Ms = 8e5 * maglab.Micro.m2geo(m0)
        e_jumag = torch.from_numpy(np.load("dataset/energy_cylinder.npy"))
        field_jumag = torch.from_numpy(np.load("dataset/field_cylinder.npy"))
        self.compare_energy(m0, Ms, e_jumag, field_jumag)
        
    def test_pbc(self):
        m0 = np.load("dataset/m0.npy")
        Ms = 8e5 * maglab.Micro.m2geo(m0)
        e_jumag = torch.from_numpy(np.load("dataset/energy_pbc_xy.npy"))
        field_jumag = torch.from_numpy(np.load("dataset/field_pbc_xy.npy"))
        self.compare_energy(m0, Ms, e_jumag, field_jumag, "xy")
        
if __name__ == '__main__':
    unittest.main()