from maglab.helper import padding_width
import unittest

class TestMicro(unittest.TestCase):
    def setUp(self):
        pass
            
    def test1(self):
        x1,x2 = padding_width(1,3)
        self.assertEqual(x1, 1)
        self.assertEqual(x2, 1)
        
        x1,x2 = padding_width(1,4)
        self.assertEqual(x1, 2)
        self.assertEqual(x2, 1)
        
        x1,x2 = padding_width(2,3)
        self.assertEqual(x1, 1)
        self.assertEqual(x2, 0)
        
        x1,x2 = padding_width(2,4)
        self.assertEqual(x1, 1)
        self.assertEqual(x2, 1)


if __name__ == '__main__':
    unittest.main()