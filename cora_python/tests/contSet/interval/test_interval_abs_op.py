"""
test_interval_abs_op - unit test function for interval abs operation

This module tests the abs operation for intervals,
covering all cases including positive, negative, mixed intervals, matrix operations, and edge cases.

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


class TestIntervalAbsOp:
    """Test class for interval abs operation"""
    
    def test_abs_empty_interval(self):
        """Test abs operation with empty intervals"""
        I = Interval.empty(1)
        I_abs = I.abs()
        assert I_abs.representsa_('emptySet')
    
    def test_abs_positive_intervals(self):
        """Test abs operation with positive intervals"""
        tol = 1e-9
        
        # Positive interval [1, 3]
        I = Interval([1], [3])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 1, atol=tol)
        assert np.isclose(I_abs.sup[0], 3, atol=tol)
        
        # Positive interval [0.5, 2.5]
        I = Interval([0.5], [2.5])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0.5, atol=tol)
        assert np.isclose(I_abs.sup[0], 2.5, atol=tol)
        
        # Small positive interval [0.001, 0.002]
        I = Interval([0.001], [0.002])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0.001, atol=tol)
        assert np.isclose(I_abs.sup[0], 0.002, atol=tol)
    
    def test_abs_negative_intervals(self):
        """Test abs operation with negative intervals"""
        tol = 1e-9
        
        # Negative interval [-3, -1]
        I = Interval([-3], [-1])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 1, atol=tol)
        assert np.isclose(I_abs.sup[0], 3, atol=tol)
        
        # Negative interval [-2.5, -0.5]
        I = Interval([-2.5], [-0.5])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0.5, atol=tol)
        assert np.isclose(I_abs.sup[0], 2.5, atol=tol)
        
        # Small negative interval [-0.002, -0.001]
        I = Interval([-0.002], [-0.001])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0.001, atol=tol)
        assert np.isclose(I_abs.sup[0], 0.002, atol=tol)
    
    def test_abs_mixed_intervals(self):
        """Test abs operation with intervals containing zero"""
        tol = 1e-9
        
        # Mixed interval [-2, 3]
        I = Interval([-2], [3])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 3, atol=tol)
        
        # Mixed interval [-5, 1]
        I = Interval([-5], [1])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 5, atol=tol)
        
        # Symmetric interval [-2, 2]
        I = Interval([-2], [2])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 2, atol=tol)
        
        # Asymmetric interval [-10, 3]
        I = Interval([-10], [3])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 10, atol=tol)
        
        # Asymmetric interval [-1, 8]
        I = Interval([-1], [8])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 8, atol=tol)
    
    def test_abs_zero_intervals(self):
        """Test abs operation with intervals containing only zero"""
        tol = 1e-9
        
        # Point interval at zero [0, 0]
        I = Interval([0], [0])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 0, atol=tol)
        
        # Small interval around zero [-0.1, 0.1]
        I = Interval([-0.1], [0.1])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 0.1, atol=tol)
        
        # Very small interval around zero [-1e-10, 1e-10]
        I = Interval([-1e-10], [1e-10])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 1e-10, atol=tol)
    
    def test_abs_vector_intervals(self):
        """Test abs operation with vector intervals"""
        tol = 1e-9
        
        # Vector interval with different signs
        I = Interval([-2, -1, 1], [3, 0, 4])
        I_abs = I.abs()
        
        # Component 0: [-2, 3] -> [0, 3]
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 3, atol=tol)
        
        # Component 1: [-1, 0] -> [0, 1]
        assert np.isclose(I_abs.inf[1], 0, atol=tol)
        assert np.isclose(I_abs.sup[1], 1, atol=tol)
        
        # Component 2: [1, 4] -> [1, 4]
        assert np.isclose(I_abs.inf[2], 1, atol=tol)
        assert np.isclose(I_abs.sup[2], 4, atol=tol)
        
        # Vector with all negative components
        I = Interval([-5, -3, -1], [-2, -1, -0.5])
        I_abs = I.abs()
        
        assert np.isclose(I_abs.inf[0], 2, atol=tol)
        assert np.isclose(I_abs.sup[0], 5, atol=tol)
        assert np.isclose(I_abs.inf[1], 1, atol=tol)
        assert np.isclose(I_abs.sup[1], 3, atol=tol)
        assert np.isclose(I_abs.inf[2], 0.5, atol=tol)
        assert np.isclose(I_abs.sup[2], 1, atol=tol)
    
    def test_abs_matrix_intervals(self):
        """Test abs operation with matrix intervals"""
        tol = 1e-9
        
        # 2x2 matrix interval
        I = Interval([[-2, 1], [-1, -3]], [[3, 2], [0, -1]])
        I_abs = I.abs()
        
        # Check shape preservation
        assert I_abs.shape == (2, 2)
        
        # Element [0,0]: [-2, 3] -> [0, 3]
        assert np.isclose(I_abs.inf[0, 0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0, 0], 3, atol=tol)
        
        # Element [0,1]: [1, 2] -> [1, 2]
        assert np.isclose(I_abs.inf[0, 1], 1, atol=tol)
        assert np.isclose(I_abs.sup[0, 1], 2, atol=tol)
        
        # Element [1,0]: [-1, 0] -> [0, 1]
        assert np.isclose(I_abs.inf[1, 0], 0, atol=tol)
        assert np.isclose(I_abs.sup[1, 0], 1, atol=tol)
        
        # Element [1,1]: [-3, -1] -> [1, 3]
        assert np.isclose(I_abs.inf[1, 1], 1, atol=tol)
        assert np.isclose(I_abs.sup[1, 1], 3, atol=tol)
        
        # 3x3 matrix interval
        I = Interval([[-1, 0, 2], [3, -4, -1], [-2, 1, 0]], 
                     [[2, 3, 5], [6, -2, 1], [0, 4, 2]])
        I_abs = I.abs()
        
        # Check shape preservation
        assert I_abs.shape == (3, 3)
        
        # Spot check a few elements
        assert np.isclose(I_abs.inf[0, 0], 0, atol=tol)  # [-1, 2] -> [0, 2]
        assert np.isclose(I_abs.sup[0, 0], 2, atol=tol)
        assert np.isclose(I_abs.inf[1, 1], 2, atol=tol)  # [-4, -2] -> [2, 4]
        assert np.isclose(I_abs.sup[1, 1], 4, atol=tol)
    
    def test_abs_large_intervals(self):
        """Test abs operation with large intervals"""
        tol = 1e-9
        
        # Large positive interval
        I = Interval([100], [1000])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 100, atol=tol)
        assert np.isclose(I_abs.sup[0], 1000, atol=tol)
        
        # Large negative interval
        I = Interval([-1000], [-100])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 100, atol=tol)
        assert np.isclose(I_abs.sup[0], 1000, atol=tol)
        
        # Large mixed interval
        I = Interval([-500], [200])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 500, atol=tol)
        
        # Very large interval
        I = Interval([-1e6], [5e5])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 1e6, atol=tol)
    
    def test_abs_infinite_intervals(self):
        """Test abs operation with infinite intervals"""
        tol = 1e-9
        
        # Positive infinite interval
        I = Interval([1], [np.inf])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 1, atol=tol)
        assert np.isinf(I_abs.sup[0]) and I_abs.sup[0] > 0
        
        # Negative infinite interval
        I = Interval([-np.inf], [-1])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 1, atol=tol)
        assert np.isinf(I_abs.sup[0]) and I_abs.sup[0] > 0
        
        # Mixed infinite interval
        I = Interval([-np.inf], [np.inf])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isinf(I_abs.sup[0]) and I_abs.sup[0] > 0
        
        # Semi-infinite intervals
        I = Interval([-np.inf], [5])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isinf(I_abs.sup[0]) and I_abs.sup[0] > 0
        
        I = Interval([-5], [np.inf])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isinf(I_abs.sup[0]) and I_abs.sup[0] > 0
    
    def test_abs_mathematical_properties(self):
        """Test mathematical properties of abs operation"""
        tol = 1e-9
        
        # abs(x) >= 0 for all x
        test_intervals = [
            Interval([-5], [3]),
            Interval([1], [4]),
            Interval([-10], [-2]),
            Interval([0], [0]),
            Interval([-1e-6], [1e-6])
        ]
        
        for I in test_intervals:
            I_abs = I.abs()
            assert np.all(I_abs.inf >= -tol), f"abs result should be non-negative: {I_abs.inf}"
            assert np.all(I_abs.sup >= I_abs.inf), f"interval bounds should be ordered: {I_abs.inf} <= {I_abs.sup}"
        
        # abs(abs(x)) = abs(x) - idempotent property
        I = Interval([-3, -1, 2], [1, 5, 6])
        I_abs = I.abs()
        I_abs_abs = I_abs.abs()
        
        assert np.allclose(I_abs.inf, I_abs_abs.inf, atol=tol)
        assert np.allclose(I_abs.sup, I_abs_abs.sup, atol=tol)
        
        # abs(-x) = abs(x) - symmetry property for point intervals
        values = [-5, -1, 0, 1, 5]
        for val in values:
            I_pos = Interval([val], [val])
            I_neg = Interval([-val], [-val])
            
            I_abs_pos = I_pos.abs()
            I_abs_neg = I_neg.abs()
            
            assert np.isclose(I_abs_pos.inf[0], I_abs_neg.inf[0], atol=tol)
            assert np.isclose(I_abs_pos.sup[0], I_abs_neg.sup[0], atol=tol)
    
    def test_abs_edge_cases(self):
        """Test abs operation with edge cases"""
        tol = 1e-9
        
        # Very small intervals
        I = Interval([-1e-15], [1e-15])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 1e-15, atol=tol)
        
        # One-sided zero intervals
        I = Interval([-5], [0])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 5, atol=tol)
        
        I = Interval([0], [5])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 5, atol=tol)
        
        # Degenerate intervals (point intervals)
        values = [-10, -1, 0, 1, 10]
        for val in values:
            I = Interval([val], [val])
            I_abs = I.abs()
            expected = abs(val)
            assert np.isclose(I_abs.inf[0], expected, atol=tol)
            assert np.isclose(I_abs.sup[0], expected, atol=tol)
    
    def test_abs_matlab_reference_examples(self):
        """Test abs operation with examples from MATLAB reference"""
        tol = 1e-9
        
        # Example from MATLAB documentation: interval([[-2],[-1]],[[3],[4]])
        I = Interval([[-2], [-1]], [[3], [4]])
        I_abs = I.abs()
        
        # Expected: [[-2, 3] -> [0, 3], [-1, 4] -> [0, 4]]
        assert np.isclose(I_abs.inf[0, 0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0, 0], 3, atol=tol)
        assert np.isclose(I_abs.inf[1, 0], 0, atol=tol)
        assert np.isclose(I_abs.sup[1, 0], 4, atol=tol)
        
        # Additional vector example
        I = Interval([[-3, 2, -1]], [[1, 5, 0]])
        I_abs = I.abs()
        
        # Expected: [[-3, 1] -> [0, 3], [2, 5] -> [2, 5], [-1, 0] -> [0, 1]]
        assert np.isclose(I_abs.inf[0, 0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0, 0], 3, atol=tol)
        assert np.isclose(I_abs.inf[0, 1], 2, atol=tol)
        assert np.isclose(I_abs.sup[0, 1], 5, atol=tol)
        assert np.isclose(I_abs.inf[0, 2], 0, atol=tol)
        assert np.isclose(I_abs.sup[0, 2], 1, atol=tol)
    
    def test_abs_numerical_precision(self):
        """Test abs operation with challenging numerical cases"""
        tol = 1e-9
        
        # Nearly zero intervals
        I = Interval([-1e-12], [1e-12])
        I_abs = I.abs()
        assert I_abs.inf[0] >= 0
        assert I_abs.sup[0] >= I_abs.inf[0]
        
        # Intervals with very different magnitudes
        I = Interval([-1e10], [1e-10])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 1e10, atol=tol)
        
        # Mixed signs with close to zero bounds
        I = Interval([-1e-8], [1e-8])
        I_abs = I.abs()
        assert np.isclose(I_abs.inf[0], 0, atol=tol)
        assert np.isclose(I_abs.sup[0], 1e-8, atol=tol)


if __name__ == '__main__':
    pytest.main([__file__]) 