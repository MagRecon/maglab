import torch
import torch.nn as nn
import torch.nn.functional as F

import maglab
import unittest

class TestMicro(unittest.TestCase):
    def setUp(self):
        nx,ny,nz=64,64,4
        dx=5e-9
        geo = maglab.geo.cylider(nx,nz)
        self.micro = maglab.Micro(nx, ny, nz, dx, pbc="x")
        
    def tearDown(self):
        self.micro = None    
        
    def test_interactions(self):
        self.micro.add_exch(1.3e-11)
        self.micro.add_demag()
        self.micro.add_anis(1e3)
        self.micro.add_dmi(1e-5)
        self.micro.add_zeeman((0,0,1e5))
        
    def test_init_m0(self):
        self.micro.init_m0((0,0,1.))
        self.micro.init_m0(torch.zeros((2,*self.micro.shape)))

    def test_set_requires_grad(self):
        self.micro.set_requires_grad(False)
        self.assertTrue(not self.micro.spherical.requires_grad)
        
        self.micro.set_requires_grad(True)
        self.assertTrue(self.micro.spherical.requires_grad)
        
    def test_io(self):
        self.micro.save_state("state.pth")
        micro1 = maglab.Micro.load_state("state.pth")
        

if __name__ == '__main__':
    unittest.main()