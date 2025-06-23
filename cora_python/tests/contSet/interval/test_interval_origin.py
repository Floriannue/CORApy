"""
test_interval_origin - unit tests for interval origin method

Syntax:
    python -m pytest test_interval_origin.py

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 21-September-2024 (MATLAB)
Python translation: 2025
"""

import unittest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalOrigin(unittest.TestCase):
    """Test cases for interval origin method"""
    
    def test_origin_1d(self):
        """Test origin interval in 1D"""
        I = Interval.origin(1)
        I_true = Interval([0])
        
        self.assertTrue(I.isequal(I_true))
        self.assertTrue(I.contains([0]))
    
    def test_origin_2d(self):
        """Test origin interval in 2D"""
        I = Interval.origin(2)
        I_true = Interval([0, 0])
        
        self.assertTrue(I.isequal(I_true))
        self.assertTrue(I.contains([0, 0]))
    
    def test_origin_higher_dimensions(self):
        """Test origin interval in higher dimensions"""
        for n in [3, 5, 10]:
            I = Interval.origin(n)
            I_true = Interval(np.zeros(n))
            
            self.assertTrue(I.isequal(I_true))
            self.assertTrue(I.contains(np.zeros(n)))
            self.assertEqual(I.dim(), n)
    
    def test_origin_properties(self):
        """Test properties of origin interval"""
        I = Interval.origin(3)
        
        # Should be a point interval
        self.assertTrue(I.representsa_('point'))
        
        # Should represent origin
        self.assertTrue(I.representsa_('origin'))
        
        # Center should be zero
        center = I.center()
        self.assertTrue(np.allclose(center, np.zeros_like(center)))
        
        # Radius should be zero
        radius = I.rad()
        self.assertTrue(np.allclose(radius, np.zeros_like(radius)))
    
    def test_origin_wrong_calls(self):
        """Test wrong calls to origin method"""
        # Zero dimension
        with self.assertRaises(Exception):
            Interval.origin(0)
        
        # Negative dimension
        with self.assertRaises(Exception):
            Interval.origin(-1)
        
        # Non-integer dimension
        with self.assertRaises(Exception):
            Interval.origin(0.5)
        
        # Array input
        with self.assertRaises(Exception):
            Interval.origin([1, 2])
        
        # String input
        with self.assertRaises(Exception):
            Interval.origin('text')
    
    def test_origin_large_dimension(self):
        """Test origin interval with large dimension"""
        n = 100
        I = Interval.origin(n)
        
        self.assertEqual(I.dim(), n)
        self.assertTrue(I.representsa_('point'))
        self.assertTrue(I.representsa_('origin'))
    
    def test_origin_contains_itself(self):
        """Test that origin interval contains itself"""
        I = Interval.origin(4)
        
        # Should contain itself
        self.assertTrue(I.contains_(I))
        
        # Should be equal to itself
        self.assertTrue(I.isequal(I))


if __name__ == '__main__':
    unittest.main() 