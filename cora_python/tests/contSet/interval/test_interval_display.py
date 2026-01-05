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
        # Use display_() to get the string
        result = I.display_('I')
        
        # Check that it contains expected elements
        self.assertIn('I =', result)
        self.assertIn('[1, 3]', result)
        self.assertIn('[2, 4]', result)
        self.assertIn('Interval object with dimension: 2', result)
        
        # Test that display() prints (capture stdout)
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            I.display('I')
            printed_output = buffer.getvalue()
            # Should print the same content (now they should be exactly equal)
            self.assertEqual(printed_output, result)
        finally:
            sys.stdout = old_stdout
    
    def test_display_matrix_interval(self):
        """Test display of 2D matrix interval"""
        I = Interval([[1, 2], [3, 4]], [[2, 3], [4, 5]])
        result = I.display_('I_matrix')
        
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
        result = I.display_('point')
        
        # Check that it contains expected elements
        self.assertIn('point =', result)
        self.assertIn('[1, 1]', result)
        self.assertIn('[2, 2]', result)
        self.assertIn('[3, 3]', result)
    
    def test_display_empty_interval(self):
        """Test display of empty interval"""
        I = Interval.empty(2)
        result = I.display_('empty_I')
        
        # Check that it contains expected elements
        self.assertIn('empty_I =', result)
        self.assertIn('2-dimensional empty set', result)
        self.assertIn('represented as Interval', result)
    
    def test_display_fullspace_interval(self):
        """Test display of fullspace interval"""
        I = Interval.Inf(3)
        result = I.display_('fullspace_I')
        
        # Check that it contains expected elements
        self.assertIn('fullspace_I =', result)
        self.assertIn('R^3', result)
        self.assertIn('represented as Interval', result)
    
    def test_display_default_name(self):
        """Test display with default name"""
        I = Interval([1], [2])
        result = I.display_()
        
        # Should use default name
        self.assertIn('ans =', result)
        self.assertIn('[1, 2]', result)
    
    def test_display_with_infinity(self):
        """Test display with infinite bounds"""
        I = Interval([-np.inf, 1], [np.inf, 2])
        result = I.display_('inf_I')
        
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
        result = I.display_('nan_I')
        
        self.assertIn('nan_I =', result)
        self.assertIn('NaN', result)
    
    def test_display_large_numbers(self):
        """Test display with large numbers (scientific notation)"""
        I = Interval([1e-5, 1e5], [2e-5, 2e5])
        result = I.display_('large_I')
        
        self.assertIn('large_I =', result)
        # Should use scientific notation for very small/large numbers
        self.assertIn('e', result.lower())
    
    def test_display_single_element(self):
        """Test display of single element interval"""
        I = Interval([5], [7])
        result = I.display_('single')
        
        self.assertIn('single =', result)
        self.assertIn('[5, 7]', result)
        self.assertIn('dimension: 1', result)
    
    def test_display_zero_interval(self):
        """Test display of interval containing zero"""
        I = Interval([-1, 0, 1], [1, 0, 2])
        result = I.display_('zero_I')
        
        self.assertIn('zero_I =', result)
        self.assertIn('[-1, 1]', result)
        self.assertIn('[0, 0]', result)
        self.assertIn('[1, 2]', result)
    
    def test_str_calls_display_(self):
        """Test that __str__ method calls display_"""
        I = Interval([1, 2], [3, 4])
        str_result = str(I)
        display_result = I.display_()
        
        # str() should return the same as display_() with default name
        self.assertEqual(str_result, display_result)
    
    def test_print_integration(self):
        """Test that print() works correctly"""
        I = Interval([1], [2])
        
        # This should not raise an exception
        try:
            # Test that str() works (calls display_)
            output = str(I)
            self.assertIn('[1, 2]', output)
            
            # Test that display() prints (capture stdout)
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            try:
                I.display()
                printed_output = buffer.getvalue()
                # Should print the same content as str() (now they should be exactly equal)
                self.assertEqual(printed_output, output)
            finally:
                sys.stdout = old_stdout
        except Exception as e:
            self.fail(f"print() integration failed: {e}")
    
    def test_repr_format(self):
        """Test that repr() gives informative format"""
        I = Interval([1, 2, 3], [4, 5, 6])
        repr_result = repr(I)
        
        # repr should show the bounds for debugging purposes
        self.assertEqual(repr_result, "Interval([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])")
        
        # Should contain the interval bounds for debugging
        self.assertIn('[1.0, 2.0, 3.0]', repr_result)
        self.assertIn('[4.0, 5.0, 6.0]', repr_result)


if __name__ == '__main__':
    unittest.main() 