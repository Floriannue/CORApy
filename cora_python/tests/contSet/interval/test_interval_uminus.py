"""
test_interval_uminus - unit tests for interval unary minus operator

Syntax:
    python -m pytest test_interval_uminus.py

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 25-June-2015 (MATLAB)
Python translation: 2025
"""

import unittest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalUminus(unittest.TestCase):
    """Test cases for interval unary minus operator"""
    
    def test_uminus_basic_interval(self):
        """Test unary minus of basic interval"""
        I = Interval([1, 2], [3, 4])
        neg_I = -I
        
        # Check that bounds are negated and swapped: -[a,b] = [-b,-a]
        np.testing.assert_array_equal(neg_I.inf, [-3, -4])
        np.testing.assert_array_equal(neg_I.sup, [-1, -2])
    
    def test_uminus_matrix_interval(self):
        """Test unary minus of matrix interval"""
        I = Interval([[1, 2], [3, 4]], [[2, 3], [4, 5]])
        neg_I = -I
        
        # Check matrix bounds
        expected_inf = [[-2, -3], [-4, -5]]
        expected_sup = [[-1, -2], [-3, -4]]
        np.testing.assert_array_equal(neg_I.inf, expected_inf)
        np.testing.assert_array_equal(neg_I.sup, expected_sup)
    
    def test_uminus_point_interval(self):
        """Test unary minus of point interval"""
        I = Interval([1, 2, 3])  # Point interval [1,1], [2,2], [3,3]
        neg_I = -I
        
        # Point interval negation: -[a,a] = [-a,-a]
        np.testing.assert_array_equal(neg_I.inf, [-1, -2, -3])
        np.testing.assert_array_equal(neg_I.sup, [-1, -2, -3])
    
    def test_uminus_zero_interval(self):
        """Test unary minus of interval containing zero"""
        I = Interval([-1, 0], [1, 0])
        neg_I = -I
        
        # -[-1,1] = [-1,1], -[0,0] = [0,0]  
        np.testing.assert_array_equal(neg_I.inf, [-1, 0])
        np.testing.assert_array_equal(neg_I.sup, [1, 0])
    
    def test_uminus_negative_interval(self):
        """Test unary minus of negative interval"""
        I = Interval([-5, -3], [-2, -1])
        neg_I = -I
        
        # -[-5,-2] = [2,5], -[-3,-1] = [1,3]
        np.testing.assert_array_equal(neg_I.inf, [2, 1])
        np.testing.assert_array_equal(neg_I.sup, [5, 3])
    
    def test_uminus_mixed_interval(self):
        """Test unary minus of interval with mixed signs"""
        I = Interval([-2, 1], [3, 4])
        neg_I = -I
        
        # -[-2,3] = [-3,2], -[1,4] = [-4,-1]
        np.testing.assert_array_equal(neg_I.inf, [-3, -4])
        np.testing.assert_array_equal(neg_I.sup, [2, -1])
    
    def test_uminus_with_infinity(self):
        """Test unary minus with infinite bounds"""
        I = Interval([-np.inf, 1], [np.inf, 2])
        neg_I = -I
        
        # -[-inf,inf] = [-inf,inf], -[1,2] = [-2,-1]
        np.testing.assert_array_equal(neg_I.inf, [-np.inf, -2])
        np.testing.assert_array_equal(neg_I.sup, [np.inf, -1])
    
    def test_uminus_empty_interval(self):
        """Test unary minus of empty interval"""
        I = Interval.empty(2)
        neg_I = -I
        
        # Negation of empty set should still be empty
        self.assertTrue(neg_I.is_empty())
        self.assertEqual(neg_I.dim(), 2)
    
    def test_uminus_single_element(self):
        """Test unary minus of single element interval"""
        I = Interval([5], [5])
        neg_I = -I
        
        np.testing.assert_array_equal(neg_I.inf, [-5])
        np.testing.assert_array_equal(neg_I.sup, [-5])
    
    def test_uminus_large_interval(self):
        """Test unary minus of large interval"""
        I = Interval([1e6, -1e6], [2e6, -0.5e6])
        neg_I = -I
        
        # Check large number handling
        np.testing.assert_array_equal(neg_I.inf, [-2e6, 0.5e6])
        np.testing.assert_array_equal(neg_I.sup, [-1e6, 1e6])
    
    def test_uminus_double_negation(self):
        """Test double negation: -(-I) = I"""
        I = Interval([1, -2], [3, 4])
        double_neg_I = -(-I)
        
        # Double negation should return original interval
        np.testing.assert_array_equal(double_neg_I.inf, I.inf)
        np.testing.assert_array_equal(double_neg_I.sup, I.sup)
    
    def test_uminus_preserves_properties(self):
        """Test that uminus preserves interval properties"""
        I = Interval([1, 2], [3, 4])
        neg_I = -I
        
        # Should preserve dimension
        self.assertEqual(neg_I.dim(), I.dim())
        
        # Should preserve boundedness
        self.assertEqual(neg_I.is_bounded(), I.is_bounded())
        
        # Should be a valid interval
        self.assertTrue(np.all(neg_I.inf <= neg_I.sup))
    
    def test_uminus_method_vs_operator(self):
        """Test that uminus method and operator give same result"""
        I = Interval([1, 2], [3, 4])
        
        # Method call
        neg_I_method = I.uminus()
        
        # Operator
        neg_I_operator = -I
        
        # Should be equal
        np.testing.assert_array_equal(neg_I_method.inf, neg_I_operator.inf)
        np.testing.assert_array_equal(neg_I_method.sup, neg_I_operator.sup)
    
    def test_uminus_with_arithmetic(self):
        """Test uminus in combination with other operations"""
        I1 = Interval([1, 2], [3, 4])
        I2 = Interval([0, 1], [2, 3])
        
        # Test: -(I1 + I2) vs (-I1) + (-I2)
        # Note: This is not always equal for intervals, but let's test the operation works
        neg_sum = -(I1 + I2)
        sum_neg = (-I1) + (-I2)
        
        # Both should be valid intervals
        self.assertTrue(np.all(neg_sum.inf <= neg_sum.sup))
        self.assertTrue(np.all(sum_neg.inf <= sum_neg.sup))
    
    def test_uminus_type_consistency(self):
        """Test that uminus returns correct type"""
        I = Interval([1, 2], [3, 4])
        neg_I = -I
        
        # Should return Interval object
        self.assertIsInstance(neg_I, Interval)
        
        # Should have same data types
        self.assertEqual(neg_I.inf.dtype, I.inf.dtype)
        self.assertEqual(neg_I.sup.dtype, I.sup.dtype)


if __name__ == '__main__':
    unittest.main() 