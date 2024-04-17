import unittest

if __name__ == "__main__":
    test_suite = unittest.TestLoader().discover(start_dir='.', pattern="test_*.py")
    print(test_suite)
    unittest.TextTestRunner().run(test_suite)