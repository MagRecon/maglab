import torch
import torch.nn as nn
import torch.nn.functional as F

import maglab
import unittest

class TestMicro(unittest.TestCase):
    @staticmethod
    def create_micro():
        nx,ny,nz=64,64,4
        dx=5e-9
        micro = maglab.Micro(nx, ny, nz, dx, pbc="x").cuda()
        return micro
    
    @staticmethod    
    def add_interactions(micro):
        micro.set_Ms(1e5)
        micro.add_exch(1e-12,)
        micro.add_dmi(1e-4)
        micro.add_anis(1e3, anis_axis=(0.3,0.4,0.5))
        micro.add_demag()
        micro.add_zeeman((0,0,1e3))
        micro.add_interfacial_dmi(2e-4)
        micro.add_cubic_anis(4.5e5)
        #micro.init_m0_random((0,0,1.))
        micro.init_m0_random()
        micro.cuda()
        
    def test_init_m0(self):
        micro = self.create_micro()
        micro.init_m0((0,0,1.))
        micro.init_m0(torch.zeros((3,*micro.shape)))
        
    def test_get_spin(self):
        micro = maglab.Micro(1,1,1, 1e-9)
        micro.set_Ms(1e5)
        theta = torch.pi/3
        phi = torch.pi/4
        sp = torch.ones(size=(2,1,1,1))
        sp[0,] = theta
        sp[1,] = phi
        m = maglab.helper.Spherical2Cartesian(sp)
        micro.init_m0(m)
        m0 = micro.get_spin()
        
    def test_set_requires_grad(self):
        micro = self.create_micro()
        micro.set_requires_grad(False)
        self.assertTrue(not micro.spin.requires_grad)
        
        micro.set_requires_grad(True)
        self.assertTrue(micro.spin.requires_grad)
        
    def maxl1(self, x):
        return torch.max(torch.abs(x)).item()
        
    def test_io(self):
        micro = self.create_micro()
        self.add_interactions(micro)
        old_spin = micro.get_spin().detach().clone()
        old_Ms = micro.Ms.detach().clone()
        old_E = micro.get_energy_list()
        old_H = micro.get_field_list()
        micro.save_state("state.pth")
        
        micro_load = maglab.Micro.load_state("state.pth", init_interactions=True).cuda()
        new_spin = micro_load.get_spin().detach().clone()
        new_Ms = micro_load.Ms.detach().clone()
        new_E = micro_load.get_energy_list()
        new_H = micro_load.get_field_list()
        self.assertTrue(torch.allclose(old_spin, new_spin))
        self.assertTrue(torch.allclose(old_Ms, new_Ms))
        for i, interaction in enumerate(old_E):
            name = interaction['classname']
            diff_e = old_E[i]['value']-new_E[i]['value']
            print(name, self.maxl1(old_E[i]['value']))
            error = self.maxl1(diff_e) / self.maxl1(old_E[i]['value'])
            print(error)
            self.assertTrue(error < 1e-5)
            
            diff_h = old_H[i]['value']-new_H[i]['value']
            error = self.maxl1(diff_h) / self.maxl1(old_H[i]['value'])
            self.assertTrue(error < 1e-5)
        

if __name__ == '__main__':
    unittest.main()