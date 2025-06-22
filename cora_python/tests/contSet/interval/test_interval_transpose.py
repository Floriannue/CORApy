"""
test_interval_transpose - unit tests for interval transpose operator

Syntax:
    python -m pytest test_interval_transpose.py

Authors: Dmitry Grebenyuk (MATLAB)
         Python translation by AI Assistant
Written: 07-February-2016 (MATLAB)
Python translation: 2025
"""

import unittest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalTranspose(unittest.TestCase):
    """Test cases for interval transpose operator"""
    
    def test_transpose_vector_interval(self):
        """Test transpose of vector interval"""
        # Column vector
        I = Interval([[1], [2], [3]], [[2], [3], [4]])
        I_T = I.T
        
        # Should become row vector
        expected_inf = [[1, 2, 3]]
        expected_sup = [[2, 3, 4]]
        np.testing.assert_array_equal(I_T.inf, expected_inf)
        np.testing.assert_array_equal(I_T.sup, expected_sup)
        
        # Check dimensions
        self.assertEqual(I_T.inf.shape, (1, 3))
        self.assertEqual(I_T.sup.shape, (1, 3))
    
    def test_transpose_matrix_interval(self):
        """Test transpose of matrix interval"""
        I = Interval([[1, 2], [3, 4]], [[2, 3], [4, 5]])
        I_T = I.T
        
        # Should transpose the matrix
        expected_inf = [[1, 3], [2, 4]]
        expected_sup = [[2, 4], [3, 5]]
        np.testing.assert_array_equal(I_T.inf, expected_inf)
        np.testing.assert_array_equal(I_T.sup, expected_sup)
        
        # Check dimensions
        self.assertEqual(I_T.inf.shape, (2, 2))
        self.assertEqual(I_T.sup.shape, (2, 2))
    
    def test_transpose_rectangular_matrix(self):
        """Test transpose of rectangular matrix interval"""
        I = Interval([[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]])
        I_T = I.T
        
        # Should transpose from 2x3 to 3x2
        expected_inf = [[1, 4], [2, 5], [3, 6]]
        expected_sup = [[2, 5], [3, 6], [4, 7]]
        np.testing.assert_array_equal(I_T.inf, expected_inf)
        np.testing.assert_array_equal(I_T.sup, expected_sup)
        
        # Check dimensions
        self.assertEqual(I_T.inf.shape, (3, 2))
        self.assertEqual(I_T.sup.shape, (3, 2))
    
    def test_transpose_1d_interval(self):
        """Test transpose of 1D interval"""
        I = Interval([1, 2, 3], [4, 5, 6])
        I_T = I.T
        
        # 1D array transpose in NumPy returns the same 1D array
        expected_inf = [1, 2, 3]
        expected_sup = [4, 5, 6]
        np.testing.assert_array_equal(I_T.inf, expected_inf)
        np.testing.assert_array_equal(I_T.sup, expected_sup)
    
    def test_transpose_point_interval(self):
        """Test transpose of point interval"""
        I = Interval([[1, 2], [3, 4]])  # Point interval
        I_T = I.T
        
        # Point interval transpose
        expected_inf = [[1, 3], [2, 4]]
        expected_sup = [[1, 3], [2, 4]]
        np.testing.assert_array_equal(I_T.inf, expected_inf)
        np.testing.assert_array_equal(I_T.sup, expected_sup)
    
    def test_transpose_single_element(self):
        """Test transpose of single element interval"""
        I = Interval([[5]], [[7]])
        I_T = I.T
        
        # Single element should remain unchanged
        np.testing.assert_array_equal(I_T.inf, [[5]])
        np.testing.assert_array_equal(I_T.sup, [[7]])
    
    def test_transpose_with_negative_values(self):
        """Test transpose with negative values"""
        I = Interval([[-1, 2], [-3, 4]], [[1, 3], [-2, 5]])
        I_T = I.T
        
        expected_inf = [[-1, -3], [2, 4]]
        expected_sup = [[1, -2], [3, 5]]
        np.testing.assert_array_equal(I_T.inf, expected_inf)
        np.testing.assert_array_equal(I_T.sup, expected_sup)
    
    def test_transpose_with_infinity(self):
        """Test transpose with infinite bounds"""
        I = Interval([[-np.inf, 1], [2, -np.inf]], [[np.inf, 2], [3, np.inf]])
        I_T = I.T
        
        expected_inf = [[-np.inf, 2], [1, -np.inf]]
        expected_sup = [[np.inf, 3], [2, np.inf]]
        np.testing.assert_array_equal(I_T.inf, expected_inf)
        np.testing.assert_array_equal(I_T.sup, expected_sup)
    
    def test_transpose_empty_interval(self):
        """Test transpose of empty interval"""
        I = Interval.empty(2)
        I_T = I.T
        
        # Empty interval should remain empty after transpose
        self.assertTrue(I_T.is_empty())
    
    def test_transpose_double_transpose(self):
        """Test double transpose: (I.T).T = I"""
        I = Interval([[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]])
        I_double_T = I.T.T
        
        # Double transpose should return original
        np.testing.assert_array_equal(I_double_T.inf, I.inf)
        np.testing.assert_array_equal(I_double_T.sup, I.sup)
    
    def test_transpose_preserves_properties(self):
        """Test that transpose preserves interval properties"""
        I = Interval([[1, 2], [3, 4]], [[2, 3], [4, 5]])
        I_T = I.T
        
        # Should preserve boundedness
        self.assertEqual(I_T.is_bounded(), I.is_bounded())
        
        # Should be a valid interval
        self.assertTrue(np.all(I_T.inf <= I_T.sup))
        
        # Should preserve total number of elements
        self.assertEqual(I_T.inf.size, I.inf.size)
        self.assertEqual(I_T.sup.size, I.sup.size)
    
    def test_transpose_method_vs_property(self):
        """Test that transpose method and property give same result"""
        I = Interval([[1, 2], [3, 4]], [[2, 3], [4, 5]])
        
        # Method call
        I_T_method = I.transpose()
        
        # Property access
        I_T_property = I.T
        
        # Should be equal
        np.testing.assert_array_equal(I_T_method.inf, I_T_property.inf)
        np.testing.assert_array_equal(I_T_method.sup, I_T_property.sup)
    
    def test_transpose_with_arithmetic(self):
        """Test transpose in combination with other operations"""
        I1 = Interval([[1, 2]], [[3, 4]])  # 1x2 interval
        I2 = Interval([[1], [2]], [[2], [3]])  # 2x1 interval
        
        # Test matrix multiplication with transpose
        # I1.T @ I2 should work (2x1 @ 2x1 won't work, but 2x1 @ 1x2 -> 2x2)
        result = I1.T @ I2.T  # (2x1) @ (1x2) -> (2x2)
        
        # Should be valid interval
        self.assertTrue(np.all(result.inf <= result.sup))
        self.assertEqual(result.inf.shape, (2, 2))
    
    def test_transpose_type_consistency(self):
        """Test that transpose returns correct type"""
        I = Interval([[1, 2], [3, 4]], [[2, 3], [4, 5]])
        I_T = I.T
        
        # Should return Interval object
        self.assertIsInstance(I_T, Interval)
        
        # Should have same data types
        self.assertEqual(I_T.inf.dtype, I.inf.dtype)
        self.assertEqual(I_T.sup.dtype, I.sup.dtype)
    
    def test_transpose_large_matrix(self):
        """Test transpose of larger matrix"""
        # Create 3x4 interval matrix
        inf_vals = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        sup_vals = [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]
        I = Interval(inf_vals, sup_vals)
        I_T = I.T
        
        # Should become 4x3
        self.assertEqual(I_T.inf.shape, (4, 3))
        self.assertEqual(I_T.sup.shape, (4, 3))
        
        # Check a few elements
        self.assertEqual(I_T.inf[0, 0], 1)  # Was I.inf[0, 0]
        self.assertEqual(I_T.inf[3, 2], 12)  # Was I.inf[2, 3]
        self.assertEqual(I_T.sup[1, 1], 7)  # Was I.sup[1, 1]


if __name__ == '__main__':
    unittest.main() 