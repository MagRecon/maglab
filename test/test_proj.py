from maglab.phasemapper import projection
import unittest
import numpy as np

class TestRadon3D(unittest.TestCase):
    def setUp(self):
        A =  np.zeros((4,4,4))
        A[1,1,1] = 1e0
        A[2,1,1] = 1e1
        A[1,2,1] = 1e2
        A[2,2,1] = 1e3

        A[1,1,2] = 1e4
        A[2,1,2] = 1e5
        A[1,2,2] = 1e6
        A[2,2,2] = 1e7
        self.testfield = A

    def tearDown(self):
        self.testfield = None

    def test_proj_x(self):
        # positive direction: counter-clockwise
        proj = projection(self.testfield, alpha=0.).cpu().numpy()
        self.assertTrue(np.allclose(proj, 
            [   [0., 0., 0., 0.],
                [0., 1.0001e4, 1.0001e6, 0.], 
                [0., 1.0001e5, 1.0001e7, 0.],
                [0., 0., 0., 0.] ]))
        
        proj = projection(self.testfield, alpha=90).cpu().numpy()
        self.assertTrue(np.allclose(proj, 
            [   [0., 0., 0., 0.],
                [0., 1.01e6, 1.01e2, 0], 
                [0., 1.01e7, 1.01e3, 0.],
                [0., 0., 0., 0.] ]))
        
        proj = projection(self.testfield, alpha=-90).cpu().numpy()
        self.assertTrue(np.allclose(proj, 
            [   [0., 0., 0., 0.],
                [0., 1.01e2, 1.01e6, 0.], 
                [0., 1.01e3, 1.01e7, 0.],
                [0., 0., 0., 0.] ]))

    def test_proj_y(self):
        proj = projection(self.testfield, beta = 0).cpu().numpy()
        self.assertTrue(np.allclose(proj, 
            [   [0., 0., 0., 0.],
                [0., 1.0001e4, 1.0001e6, 0.], 
                [0., 1.0001e5, 1.0001e7, 0.],
                [0., 0., 0., 0.] ]))

        proj = projection(self.testfield, beta=90).cpu().numpy()
        self.assertTrue(np.allclose(proj, 
            [   [0., 0., 0., 0.],
                [0., 1.1e1, 1.1e3, 0.], 
                [0., 1.1e5, 1.1e7, 0.],
                [0., 0., 0., 0.] ]))

        proj = projection(self.testfield, beta=-90).cpu().numpy()
        self.assertTrue(np.allclose(proj, 
            [   [0., 0., 0., 0.],
                [0., 1.1e5, 1.1e7, 0.], 
                [0., 1.1e1, 1.1e3, 0.],
                [0., 0., 0., 0.],
                 ]))
        
    def test_proj_z(self):
        proj = projection(self.testfield, gamma=90).cpu().numpy()
        self.assertTrue(np.allclose(proj, 
            [   [0., 0., 0., 0.],
                [0., 1.0001e6, 1.0001e7, 0.], 
                [0., 1.0001e4, 1.0001e5, 0.],
                [0., 0., 0., 0.] ]))

        proj = projection(self.testfield, gamma=-90).cpu().numpy()
        self.assertTrue(np.allclose(proj, 
            [   [0., 0., 0., 0.],
                [0., 1.0001e5, 1.0001e4, 0.], 
                [0., 1.0001e7, 1.0001e6, 0.],
                [0., 0., 0., 0.],
                 ]))



if __name__ == '__main__':
    unittest.main()