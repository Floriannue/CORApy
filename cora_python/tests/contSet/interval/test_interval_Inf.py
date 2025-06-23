"""
test_interval_Inf - unit tests for interval Inf method (R^n instantiation)

Syntax:
    python -m pytest test_interval_Inf.py

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 16-January-2024 (MATLAB)
Python translation: 2025
"""

import unittest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalInf(unittest.TestCase):
    """Test cases for interval Inf method"""
    
    def test_Inf_1d(self):
        """Test Inf interval in 1D"""
        n = 1
        I = Interval.Inf(n)
        
        self.assertTrue(I.representsa_('fullspace'))
        self.assertEqual(I.dim(), 1)
    
    def test_Inf_5d(self):
        """Test Inf interval in 5D"""
        n = 5
        I = Interval.Inf(n)
        
        self.assertTrue(I.representsa_('fullspace'))
        self.assertEqual(I.dim(), 5)
    
    def test_Inf_properties(self):
        """Test properties of Inf interval"""
        I = Interval.Inf(3)
        
        # Should represent fullspace
        self.assertTrue(I.representsa_('fullspace'))
        
        # Should not be empty
        self.assertFalse(I.isemptyobject())
        
        # Should be unbounded
        self.assertFalse(I.is_bounded())
        
        # Bounds should be infinite
        self.assertTrue(np.all(np.isinf(I.inf)))
        self.assertTrue(np.all(np.isinf(I.sup)))
        self.assertTrue(np.all(I.inf < 0))  # -inf
        self.assertTrue(np.all(I.sup > 0))  # +inf
    
    def test_Inf_different_dimensions(self):
        """Test Inf interval with different dimensions"""
        for n in [1, 2, 3, 10, 100]:
            I = Interval.Inf(n)
            
            self.assertTrue(I.representsa_('fullspace'))
            self.assertEqual(I.dim(), n)
            self.assertFalse(I.isemptyobject())
            self.assertFalse(I.is_bounded())
    
    def test_Inf_contains_everything(self):
        """Test that Inf interval contains any finite point"""
        I = Interval.Inf(3)
        
        # Should contain origin
        self.assertTrue(I.contains_(np.zeros((3, 1)))[0])
        
        # Should contain any finite point
        points = [
            np.array([[1], [2], [3]]),
            np.array([[-100], [50], [-25]]),
            np.array([[1e6], [-1e6], [0]])
        ]
        
        for point in points:
            self.assertTrue(I.contains_(point)[0])
    
    def test_Inf_contains_finite_intervals(self):
        """Test that Inf interval contains any finite interval"""
        I_inf = Interval.Inf(2)
        
        # Should contain any finite interval
        finite_intervals = [
            Interval([0, 0], [1, 1]),
            Interval([-100, -50], [100, 50]),
            Interval([-1e6, -1e6], [1e6, 1e6])
        ]
        
        for I_finite in finite_intervals:
            self.assertTrue(I_inf.contains_(I_finite)[0])
    
    def test_Inf_operations(self):
        """Test basic operations with Inf interval"""
        I = Interval.Inf(2)
        
        # Addition with finite interval should still be fullspace
        I_finite = Interval([1, 2], [3, 4])
        I_sum = I + I_finite
        self.assertTrue(I_sum.representsa_('fullspace'))
        
        # Intersection with finite interval should give finite interval
        I_intersect = I.and_(I_finite)
        self.assertTrue(I_intersect.isequal(I_finite))
    
    def test_Inf_wrong_calls(self):
        """Test wrong calls to Inf method"""
        # Zero dimension is allowed in MATLAB (nonnegative validation)
        # This should work without exception
        I_zero = Interval.Inf(0)
        self.assertEqual(I_zero.dim(), 0)
        
        # Negative dimension should raise exception
        with self.assertRaises(Exception):
            Interval.Inf(-1)
        
        # Non-integer dimension
        with self.assertRaises(Exception):
            Interval.Inf(2.5)
        
        # Array input
        with self.assertRaises(Exception):
            Interval.Inf([1, 2])


if __name__ == '__main__':
    unittest.main() 