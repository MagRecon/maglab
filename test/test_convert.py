import maglab
import torch
import numpy as np
import unittest

sin, cos = np.sin, np.cos

def np_curl(F):
    Fx, Fy, Fz = F
    
    dFz_dx, dFz_dy = np.gradient(Fz, axis=(0, 1))
    dFy_dz, dFy_dx = np.gradient(Fy, axis=(2, 0))
    dFx_dz, dFx_dy = np.gradient(Fx, axis=(2, 1))
    
    curl_x = dFz_dy - dFy_dz
    curl_y = dFx_dz - dFz_dx
    curl_z = dFy_dx - dFx_dy
    
    curl = np.array([curl_x, curl_y, curl_z])
    
    return curl

class TestConvert(unittest.TestCase):
    def setUp(self):
        source_vec = np.zeros((3,3,3,3))
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    index = 9 * k + 3 * j + i
                    source_vec[0,i,j,k] = index**2
                    source_vec[1,i,j,k] = index**2 + index
                    source_vec[2,i,j,k] = index**2 + 5*index
                    
        self.source_vec = source_vec            
        self.expected_curl = np_curl(source_vec)

    def test_curl(self,):
        my_curl = maglab.convert.curl(torch.from_numpy(self.source_vec), 1)
        self.assertTrue(np.allclose(my_curl.detach().cpu().numpy(), self.expected_curl))
        
if __name__ == '__main__':
    unittest.main()