"""
test_interval_cos - unit test function for interval cos operation

This module tests the cos operation for intervals,
covering all cases including periodic behavior, edge cases, and matrix operations.

Authors: Python translation by AI Assistant
Written: 2025
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import cora_python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from cora_python.contSet import Interval


class TestIntervalCos:
    """Test class for interval cos operation"""
    
    def test_cos_empty_interval(self):
        """Test cos operation with empty intervals"""
        I = Interval.empty(1)
        I_cos = I.cos()
        assert I_cos.representsa_('emptySet')
    
    def test_cos_scalar_basic(self):
        """Test cos operation with basic scalar intervals"""
        tol = 1e-9
        
        # Simple interval [0, pi/2]
        I = Interval([0], [np.pi/2])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], 0, atol=tol)
        assert np.isclose(I_cos.sup[0], 1, atol=tol)
        
        # Interval [-pi/2, 0]
        I = Interval([-np.pi/2], [0])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], 0, atol=tol)
        assert np.isclose(I_cos.sup[0], 1, atol=tol)
    
    def test_cos_full_period(self):
        """Test cos operation with intervals spanning full periods"""
        tol = 1e-9
        
        # Interval spanning more than 2*pi -> [-1, 1]
        I = Interval([0], [3*np.pi])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], -1, atol=tol)
        assert np.isclose(I_cos.sup[0], 1, atol=tol)
        
        # Exact 2*pi interval
        I = Interval([0], [2*np.pi])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], -1, atol=tol)
        assert np.isclose(I_cos.sup[0], 1, atol=tol)
    
    def test_cos_quarter_periods(self):
        """Test cos with intervals in different quarter periods"""
        tol = 1e-9
        
        # First quarter: [0, pi/2] -> [cos(pi/2), cos(0)] = [0, 1]
        I = Interval([0], [np.pi/2])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], 0, atol=tol)
        assert np.isclose(I_cos.sup[0], 1, atol=tol)
        
        # Second quarter: [pi/2, pi] -> [cos(pi), cos(pi/2)] = [-1, 0]
        I = Interval([np.pi/2], [np.pi])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], -1, atol=tol)
        assert np.isclose(I_cos.sup[0], 0, atol=tol)
        
        # Third quarter: [pi, 3*pi/2] -> [cos(3*pi/2), cos(pi)] = [-1, 0]
        I = Interval([np.pi], [3*np.pi/2])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], -1, atol=tol)
        assert np.isclose(I_cos.sup[0], 0, atol=tol)
        
        # Fourth quarter: [3*pi/2, 2*pi] -> [cos(3*pi/2), cos(2*pi)] = [0, 1]
        I = Interval([3*np.pi/2], [2*np.pi])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], 0, atol=tol)
        assert np.isclose(I_cos.sup[0], 1, atol=tol)
    
    def test_cos_crossing_maxima_minima(self):
        """Test cos with intervals crossing maxima and minima"""
        tol = 1e-9
        
        # Crossing maximum at 0
        I = Interval([-np.pi/4], [np.pi/4])
        I_cos = I.cos()
        assert np.isclose(I_cos.sup[0], 1, atol=tol)  # Maximum value
        expected_min = min(np.cos(-np.pi/4), np.cos(np.pi/4))
        assert np.isclose(I_cos.inf[0], expected_min, atol=tol)
        
        # Crossing minimum at pi
        I = Interval([3*np.pi/4], [5*np.pi/4])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], -1, atol=tol)  # Minimum value
        expected_max = max(np.cos(3*np.pi/4), np.cos(5*np.pi/4))
        assert np.isclose(I_cos.sup[0], expected_max, atol=tol)
    
    def test_cos_negative_intervals(self):
        """Test cos with negative intervals"""
        tol = 1e-9
        
        # Negative interval [-pi, -pi/2]
        I = Interval([-np.pi], [-np.pi/2])
        I_cos = I.cos()
        expected_inf = np.cos(-np.pi)    # -1
        expected_sup = np.cos(-np.pi/2)  # 0
        assert np.isclose(I_cos.inf[0], expected_inf, atol=tol)
        assert np.isclose(I_cos.sup[0], expected_sup, atol=tol)
    
    def test_cos_matrix_intervals(self):
        """Test cos operation with matrix intervals"""
        tol = 1e-9
        
        # 2x2 matrix interval
        I = Interval([[0, np.pi/2], [-np.pi/4, np.pi]], 
                     [[np.pi/2, np.pi], [np.pi/4, 3*np.pi/2]])
        I_cos = I.cos()
        
        # Check shape preservation
        assert I_cos.shape == (2, 2)
        
        # Element [0,0]: [0, pi/2] -> [0, 1]
        assert np.isclose(I_cos.inf[0, 0], 0, atol=tol)
        assert np.isclose(I_cos.sup[0, 0], 1, atol=tol)
        
        # Element [0,1]: [pi/2, pi] -> [-1, 0]
        assert np.isclose(I_cos.inf[0, 1], -1, atol=tol)
        assert np.isclose(I_cos.sup[0, 1], 0, atol=tol)
    
    def test_cos_large_intervals(self):
        """Test cos with very large intervals"""
        tol = 1e-9
        
        # Very large positive interval
        I = Interval([0], [100*np.pi])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], -1, atol=tol)
        assert np.isclose(I_cos.sup[0], 1, atol=tol)
        
        # Very large negative interval
        I = Interval([-100*np.pi], [0])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], -1, atol=tol)
        assert np.isclose(I_cos.sup[0], 1, atol=tol)
    
    def test_cos_special_values(self):
        """Test cos with special mathematical values"""
        tol = 1e-9
        
        # cos(0) = 1
        I = Interval([0], [0])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], 1, atol=tol)
        assert np.isclose(I_cos.sup[0], 1, atol=tol)
        
        # cos(pi/2) = 0
        I = Interval([np.pi/2], [np.pi/2])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], 0, atol=tol)
        assert np.isclose(I_cos.sup[0], 0, atol=tol)
        
        # cos(pi) = -1
        I = Interval([np.pi], [np.pi])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], -1, atol=tol)
        assert np.isclose(I_cos.sup[0], -1, atol=tol)
        
        # cos(3*pi/2) = 0
        I = Interval([3*np.pi/2], [3*np.pi/2])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], 0, atol=tol)
        assert np.isclose(I_cos.sup[0], 0, atol=tol)
        
        # cos(2*pi) = 1
        I = Interval([2*np.pi], [2*np.pi])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], 1, atol=tol)
        assert np.isclose(I_cos.sup[0], 1, atol=tol)
    
    def test_cos_vector_intervals(self):
        """Test cos with vector intervals"""
        tol = 1e-9
        
        # Vector interval with multiple components
        I = Interval([0, np.pi/2, -np.pi/4], [np.pi/2, np.pi, np.pi/4])
        I_cos = I.cos()
        
        # Check each component
        assert np.isclose(I_cos.inf[0], 0, atol=tol)  # cos([0, pi/2])
        assert np.isclose(I_cos.sup[0], 1, atol=tol)
        
        assert np.isclose(I_cos.inf[1], -1, atol=tol)  # cos([pi/2, pi])
        assert np.isclose(I_cos.sup[1], 0, atol=tol)
        
        assert np.isclose(I_cos.sup[2], 1, atol=tol)  # crosses maximum at 0
    
    def test_cos_bounded_vs_unbounded(self):
        """Test that cos always produces bounded output"""
        # Even with unbounded input, cos output should be bounded
        I = Interval([-np.inf], [np.inf])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], -1)
        assert np.isclose(I_cos.sup[0], 1)
        
        # Semi-infinite intervals
        I = Interval([-np.inf], [0])
        I_cos = I.cos()
        assert np.isclose(I_cos.inf[0], -1)
        assert np.isclose(I_cos.sup[0], 1)
    
    def test_cos_symmetry_properties(self):
        """Test cos symmetry properties"""
        tol = 1e-9
        
        # cos(-x) = cos(x) - test even function property
        I_pos = Interval([0], [np.pi/3])
        I_neg = Interval([-np.pi/3], [0])
        
        I_cos_pos = I_pos.cos()
        I_cos_neg = I_neg.cos()
        
        # Should give same result for symmetric intervals
        assert np.isclose(I_cos_pos.inf[0], I_cos_neg.inf[0], atol=tol)
        assert np.isclose(I_cos_pos.sup[0], I_cos_neg.sup[0], atol=tol)
    
    def test_cos_periodicity(self):
        """Test cos periodicity properties"""
        tol = 1e-9
        
        # cos(x) = cos(x + 2*pi)
        I1 = Interval([0], [np.pi/4])
        I2 = Interval([2*np.pi], [2*np.pi + np.pi/4])
        
        I_cos1 = I1.cos()
        I_cos2 = I2.cos()
        
        # Should be approximately equal due to periodicity
        assert np.isclose(I_cos1.inf[0], I_cos2.inf[0], atol=tol)
        assert np.isclose(I_cos1.sup[0], I_cos2.sup[0], atol=tol)
    
    def test_cos_monotonicity_regions(self):
        """Test cos behavior in monotonic regions"""
        tol = 1e-9
        
        # Decreasing region [0, pi]
        I = Interval([np.pi/6], [np.pi/3])
        I_cos = I.cos()
        # In decreasing region, cos(inf) > cos(sup)
        assert I_cos.inf[0] <= np.cos(np.pi/3) + tol
        assert I_cos.sup[0] >= np.cos(np.pi/6) - tol
        
        # Increasing region [pi, 2*pi]
        I = Interval([4*np.pi/3], [5*np.pi/3])
        I_cos = I.cos()
        # In increasing region, cos(inf) < cos(sup)
        assert I_cos.inf[0] <= np.cos(4*np.pi/3) + tol
        assert I_cos.sup[0] >= np.cos(5*np.pi/3) - tol
    
    def test_cos_complex_intervals(self):
        """Test cos with complex interval patterns"""
        tol = 1e-9
        
        # Interval crossing multiple extrema
        I = Interval([-np.pi/2], [3*np.pi/2])
        I_cos = I.cos()
        # Should span full range [-1, 1] since it crosses both min and max
        assert np.isclose(I_cos.inf[0], -1, atol=tol)
        assert np.isclose(I_cos.sup[0], 1, atol=tol)
        
        # Small interval around extremum
        I = Interval([np.pi - 0.01], [np.pi + 0.01])
        I_cos = I.cos()
        # Should be close to minimum value -1
        assert I_cos.inf[0] <= -0.99
        assert I_cos.sup[0] <= -0.99


if __name__ == '__main__':
    pytest.main([__file__]) 