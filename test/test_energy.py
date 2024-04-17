import maglab
import torch
import numpy as np
import unittest
from maglab.loss import coords

sin, cos = np.sin, np.cos

class test1(unittest.TestCase):
    def setUp(self):
        pass
        # nx,ny,nz = 11,11,11
        # cellsize = 1e-9
        # Ms = 8e5
    
    def create_micro(self, m0, cellsize):
        geo = np.sum(m0**2, axis=0)
        micro = maglab.Micro(geo, cellsize)
        micro.add_exch(1e-12, save_energy=True)
        micro.add_dmi(1e-4, save_energy=True)
        micro.add_anis(1e3, save_energy=True)
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
        return torch.max(torch.abs(x))
    
    def compare_error(self, m0, e_jumag):
        micro = self.create_micro(m0, 1e-9,)
        micro.loss(Ms=8e5)
        e = np.stack([micro.interactions[i].E.detach().cpu().numpy() \
                      for i in range(len(micro.interactions))])
        
        
        energy_diff = e_jumag - e
        for i in range(len(e)):
            error = self.maxl1(energy_diff[i,]) / self.maxl1(e_jumag[i,])
            print(error)
            self.assertTrue(error < 1e-5)
        
    def test_cubiod(self):
        # m0 is filled with cubiod geo
        m0 = np.load("dataset/m0.npy")
        e_jumag = torch.from_numpy(np.load("dataset/energy.npy"))
        self.compare_error(m0, e_jumag)
    
    def test_cylinder(self):
        m0 = np.load("dataset/m0_cyd.npy")
        e_jumag = torch.from_numpy(np.load("dataset/energy_cyd.npy"))
        self.compare_error(m0, e_jumag)
        
if __name__ == '__main__':
    unittest.main()