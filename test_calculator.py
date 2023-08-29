import unittest
from calculator import exponent

class TestCalculator(unittest.TestCase):

    def test_exponent_positive_numbers(self):
        self.assertEqual(exponent(2, 3), 8)

    def test_exponent_negative_base(self):
        self.assertEqual(exponent(-2, 3), -8)

    def test_exponent_zero_base(self):
        self.assertEqual(exponent(0, 3), 0)

    def test_exponent_negative_exponent(self):
        self.assertEqual(exponent(2, -3), 0.125)

    def test_exponent_fractional_base(self):
        self.assertEqual(exponent(0.5, 3), 0.125)

    def test_exponent_fractional_exponent(self):
        self.assertEqual(exponent(2, 0.5), 1.41421356237)

    def test_exponent_error_negative_base_and_fractional_exponent(self):
        with self.assertRaises(ValueError):
            exponent(-2, 0.5)

if __name__ == '__main__':
    unittest.main()
