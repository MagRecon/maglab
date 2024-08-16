import maglab
import unittest
import torch

import maglab.helper

# def SphericalUnitVectors(spherical):
#     theta, phi = spherical[0,], spherical[1,]
#     st, ct = torch.sin(theta), torch.cos(theta)
#     sp, cp = torch.sin(phi), torch.cos(phi)
    
#     e_rho = torch.stack([st * cp, st * sp, ct])
#     e_theta = torch.stack([ct * cp, ct * sp, -1*st])
#     e_phi =  torch.stack([-1*sp, cp, torch.zeros_like(sp)])
#     return e_rho, e_theta, e_phi
# class TestHelper(unittest.TestCase):
#     def setUp(self):
#         pass
    
#     def test_SphericalUnitVectors(self):
#         spherical = torch.random.random((2,5,5))
#         e_rho, e_theta, e_phi = maglab.helper.SphericalUnitVectors(spherical)