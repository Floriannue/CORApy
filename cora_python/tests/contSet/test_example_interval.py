"""
Test file for example_interval

This test verifies that the interval example runs without errors
and produces the expected outputs.

Authors: AI Assistant
Date: 2025
"""

import pytest
import numpy as np
import matplotlib
# Use non-interactive backend for testing
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cora_python.examples.contSet.example_interval import example_interval


class TestExampleInterval:
    """Test class for the interval example"""
    
    def test_example_interval_runs(self):
        """Test that the example runs without errors"""
        # Close any existing figures to avoid interference
        plt.close('all')
        
        # Capture stdout to suppress print statements during testing
        import io
        import contextlib
        
        captured_output = io.StringIO()
        
        with contextlib.redirect_stdout(captured_output):
            result = example_interval()
        
        # Check that example completed successfully
        assert result is True, "Example should return True on successful completion"
        
        # Check that some output was produced
        output = captured_output.getvalue()
        assert "Interval Example" in output, "Expected example title in output"
        assert "Radius of I1" in output, "Expected radius computation message"
        assert "Center of I3" in output, "Expected center computation message"
        assert "Example completed successfully" in output, "Expected completion message"
        
        # Close any figures created during the test
        plt.close('all')
    
    def test_example_interval_operations(self):
        """Test that the interval operations work correctly"""
        from cora_python.contSet.interval import Interval
        
        # Create the same intervals as in the example
        I1 = Interval([0, -1], [3, 1])
        I2 = Interval([-1, -1.5], [1, -0.5])
        
        # Test radius computation
        r = I1.rad()
        expected_radius = np.array([1.5, 1.0])  # (3-0)/2, (1-(-1))/2
        np.testing.assert_array_almost_equal(r, expected_radius, decimal=10)
        
        # Test intersection
        I3 = I1 & I2
        
        # Expected intersection: [max(0,-1), min(3,1)] x [max(-1,-1.5), min(1,-0.5)]
        # = [0, 1] x [-1, -0.5]
        expected_inf = np.array([0, -1])
        expected_sup = np.array([1, -0.5])
        
        np.testing.assert_array_almost_equal(I3.inf, expected_inf, decimal=10)
        np.testing.assert_array_almost_equal(I3.sup, expected_sup, decimal=10)
        
        # Test center computation
        c = I3.center()
        expected_center = np.array([0.5, -0.75])  # midpoint of intersection
        np.testing.assert_array_almost_equal(c, expected_center, decimal=10)
    
    def test_example_interval_properties(self):
        """Test that the intervals have expected properties"""
        from cora_python.contSet.interval import Interval
        
        I1 = Interval([0, -1], [3, 1])
        I2 = Interval([-1, -1.5], [1, -0.5])
        
        # Test dimensions
        assert I1.dim() == 2, "I1 should be 2-dimensional"
        assert I2.dim() == 2, "I2 should be 2-dimensional"
        
        # Test that intervals are not empty
        assert not I1.representsa_('emptySet'), "I1 should not be empty"
        assert not I2.representsa_('emptySet'), "I2 should not be empty"
        
        # Test intersection is not empty
        I3 = I1 & I2
        assert not I3.representsa_('emptySet'), "Intersection should not be empty"
        
        # Test that intersection is contained in both original intervals
        # This would require a contains method, but we can check bounds
        assert np.all(I3.inf >= np.maximum(I1.inf, I2.inf))
        assert np.all(I3.sup <= np.minimum(I1.sup, I2.sup))
    
    def test_example_interval_edge_cases(self):
        """Test edge cases for interval operations"""
        from cora_python.contSet.interval import Interval
        
        # Test with point intervals (zero radius)
        I_point = Interval([1, 2], [1, 2])
        r_point = I_point.rad()
        expected_zero_radius = np.array([0, 0])
        np.testing.assert_array_almost_equal(r_point, expected_zero_radius, decimal=10)
        
        # Test with non-intersecting intervals
        I_left = Interval([0, 0], [1, 1])
        I_right = Interval([2, 2], [3, 3])
        I_empty = I_left & I_right
        
        # The intersection should be empty
        assert I_empty.representsa_('emptySet'), "Non-intersecting intervals should produce empty intersection" 