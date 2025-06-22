"""
test_interval_display - unit tests for interval display method

Syntax:
    python -m pytest test_interval_display.py

Authors: Matthias Althoff (MATLAB)
         Python translation by AI Assistant
Written: 19-June-2015 (MATLAB)
Python translation: 2025
"""

import unittest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalDisplay(unittest.TestCase):
    """Test cases for interval display method"""
    
    def test_display_basic_interval(self):
        """Test display of basic 1D interval"""
        I = Interval([1, 2], [3, 4])
        result = I.display('I')
        
        # Check that it contains expected elements
        self.assertIn('I =', result)
        self.assertIn('[1, 3]', result)
        self.assertIn('[2, 4]', result)
        self.assertIn('Interval object with dimension: 2', result)
    
    def test_display_matrix_interval(self):
        """Test display of 2D matrix interval"""
        I = Interval([[1, 2], [3, 4]], [[2, 3], [4, 5]])
        result = I.display('I_matrix')
        
        # Check that it contains expected elements
        self.assertIn('I_matrix =', result)
        self.assertIn('[1, 2]', result)
        self.assertIn('[2, 3]', result)
        self.assertIn('[3, 4]', result)
        self.assertIn('[4, 5]', result)
        self.assertIn('[2, 2]', result)  # Dimension info
    
    def test_display_point_interval(self):
        """Test display of point interval (inf = sup)"""
        I = Interval([1, 2, 3])
        result = I.display('point')
        
        # Check that it contains expected elements
        self.assertIn('point =', result)
        self.assertIn('[1, 1]', result)
        self.assertIn('[2, 2]', result)
        self.assertIn('[3, 3]', result)
    
    def test_display_empty_interval(self):
        """Test display of empty interval"""
        I = Interval.empty(2)
        result = I.display('empty_I')
        
        # Check that it contains expected elements
        self.assertIn('empty_I =', result)
        self.assertIn('2-dimensional empty set', result)
        self.assertIn('represented as Interval', result)
    
    def test_display_fullspace_interval(self):
        """Test display of fullspace interval"""
        I = Interval.Inf(3)
        result = I.display('fullspace_I')
        
        # Check that it contains expected elements
        self.assertIn('fullspace_I =', result)
        self.assertIn('R^3', result)
        self.assertIn('represented as Interval', result)
    
    def test_display_default_name(self):
        """Test display with default name"""
        I = Interval([1], [2])
        result = I.display()
        
        # Should use default name
        self.assertIn('ans =', result)
        self.assertIn('[1, 2]', result)
    
    def test_display_with_infinity(self):
        """Test display with infinite bounds"""
        I = Interval([-np.inf, 1], [np.inf, 2])
        result = I.display('inf_I')
        
        # Check that infinity is displayed correctly
        self.assertIn('inf_I =', result)
        self.assertIn('[-Inf, Inf]', result)
        self.assertIn('[1, 2]', result)
    
    def test_display_with_nan(self):
        """Test display with NaN values"""
        # Note: NaN in bounds is typically caught during construction
        # This test is more for completeness
        I = Interval([0, 1], [1, 2])
        # Manually set NaN for testing display format
        I.inf[0] = np.nan
        result = I.display('nan_I')
        
        self.assertIn('nan_I =', result)
        self.assertIn('NaN', result)
    
    def test_display_large_numbers(self):
        """Test display with large numbers (scientific notation)"""
        I = Interval([1e-5, 1e5], [2e-5, 2e5])
        result = I.display('large_I')
        
        self.assertIn('large_I =', result)
        # Should use scientific notation for very small/large numbers
        self.assertIn('e', result.lower())
    
    def test_display_single_element(self):
        """Test display of single element interval"""
        I = Interval([5], [7])
        result = I.display('single')
        
        self.assertIn('single =', result)
        self.assertIn('[5, 7]', result)
        self.assertIn('dimension: 1', result)
    
    def test_display_zero_interval(self):
        """Test display of interval containing zero"""
        I = Interval([-1, 0, 1], [1, 0, 2])
        result = I.display('zero_I')
        
        self.assertIn('zero_I =', result)
        self.assertIn('[-1, 1]', result)
        self.assertIn('[0, 0]', result)
        self.assertIn('[1, 2]', result)
    
    def test_str_calls_display(self):
        """Test that __str__ method calls display"""
        I = Interval([1, 2], [3, 4])
        str_result = str(I)
        display_result = I.display('ans')
        
        # str() should return the same as display() with default name
        self.assertEqual(str_result, display_result)
    
    def test_print_integration(self):
        """Test that print() works correctly"""
        I = Interval([1], [2])
        
        # This should not raise an exception
        try:
            # Capture print output would require more complex setup
            # For now, just ensure str() works
            output = str(I)
            self.assertIn('[1, 2]', output)
        except Exception as e:
            self.fail(f"print() integration failed: {e}")
    
    def test_repr_brief_format(self):
        """Test that repr() gives brief format"""
        I = Interval([1, 2, 3], [4, 5, 6])
        repr_result = repr(I)
        
        # repr should be brief
        self.assertEqual(repr_result, "Interval(dim=3)")
        
        # Should not contain the full interval bounds
        self.assertNotIn('[1, 4]', repr_result)


if __name__ == '__main__':
    unittest.main() 