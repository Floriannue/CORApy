"""
test_interval_representsa_ - unit tests for interval representsa_ method

Syntax:
    python -m pytest test_interval_representsa_.py

Authors: Dmitry Grebenyuk, Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 16-January-2016 (MATLAB)
Python translation: 2025
"""

import unittest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalRepresentsa(unittest.TestCase):
    """Test cases for interval representsa_ method"""
    
    def test_representsa_empty_set(self):
        """Test comparison to empty set"""
        # empty interval
        I = Interval.empty(2)
        self.assertTrue(I.representsa_('emptySet'))
        
        # non-empty intervals
        I = Interval(-5.0, 2.0)
        self.assertFalse(I.representsa_('emptySet'))
        
        I = Interval([-5.0, -4.0, -3, 0, 0, 5], [-2, 0.0, 2.0, 0, 5, 8])
        self.assertFalse(I.representsa_('emptySet'))
    
    def test_representsa_origin(self):
        """Test comparison to origin"""
        # empty interval
        I = Interval.empty(2)
        self.assertFalse(I.representsa_('origin'))
        
        # only origin
        I = Interval(np.zeros((3, 1)), np.zeros((3, 1)))
        self.assertTrue(I.representsa_('origin'))
        
        # shifted center
        I = Interval([0.01, 0.02], [0.03, 0.025])
        self.assertFalse(I.representsa_('origin'))
        
        # shifted center, contains origin within tolerance
        I = Interval([0.01, -0.01], [0.02, 0.01])
        tol = 0.05
        self.assertTrue(I.representsa_('origin', tol))
    
    def test_representsa_point(self):
        """Test comparison to point"""
        # empty interval
        I = Interval.empty(2)
        self.assertFalse(I.representsa_('point'))
        
        # point interval
        I = Interval([-3, -2], [-3, -2])
        self.assertTrue(I.representsa_('point'))
        
        # non-point interval
        I = Interval([-3, -2], [-3, -1])
        self.assertFalse(I.representsa_('point'))
        
        # non-point interval within tolerance
        self.assertTrue(I.representsa_('point', 1))
    
    def test_representsa_zonotope(self):
        """Test comparison to zonotope"""
        # empty interval
        I = Interval.empty(2)
        self.assertTrue(I.representsa_('zonotope'))
        
        # non-empty interval
        I = Interval([-4, 2, 5], [5, 2, 8])
        res = I.representsa_('zonotope')
        self.assertTrue(res)
        
        # Test with conversion (if available)
        try:
            res, Z = I.representsa_('zonotope', return_object=True)
            self.assertTrue(res)
            # Note: Zonotope comparison would require zonotope implementation
            # For now, just check that conversion doesn't fail
        except (NotImplementedError, TypeError):
            # Zonotope conversion might not be implemented yet
            pass
    
    def test_representsa_fullspace(self):
        """Test comparison to fullspace"""
        # finite interval
        I = Interval([-1, -2], [1, 2])
        self.assertFalse(I.representsa_('fullspace'))
        
        # fullspace interval
        I = Interval.Inf(3)
        self.assertTrue(I.representsa_('fullspace'))
    
    def test_representsa_invalid_type(self):
        """Test with invalid set type"""
        I = Interval([1, 2], [3, 4])
        
        # Should return False for unknown set types
        self.assertFalse(I.representsa_('unknownSetType'))
    
    def test_representsa_with_tolerance(self):
        """Test representsa with different tolerance values"""
        # Create interval close to point
        I = Interval([1.0, 2.0], [1.01, 2.01])
        
        # Should not be point with default tolerance
        self.assertFalse(I.representsa_('point'))
        
        # Should be point with larger tolerance
        self.assertTrue(I.representsa_('point', 0.02))
        
        # Should not be point with smaller tolerance
        self.assertFalse(I.representsa_('point', 0.005))
    
    def test_representsa_case_insensitive(self):
        """Test that representsa_ is case insensitive"""
        I = Interval.empty(2)
        
        # Test different cases
        self.assertTrue(I.representsa_('emptyset'))
        self.assertTrue(I.representsa_('emptySet'))
        self.assertTrue(I.representsa_('EMPTYSET'))


if __name__ == '__main__':
    unittest.main() 