"""
test_interval_and_ - unit test function of logical conjunction,
overloaded '&' operator for intervals

Tests the and_ method for interval objects to check intersection operations.

Syntax:
    pytest cora_python/tests/contSet/interval/test_interval_and_.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.interval import Interval


class TestIntervalAnd:
    def test_and_bounded_intersection(self):
        """Test bounded, bounded intersection"""
        I1 = Interval(np.array([[-10], [-2], [10]]), np.array([[-5], [8], [15]]))
        I2 = Interval(np.array([[-11], [2], [11]]), np.array([[-6], [9], [12]]))
        I = I1.and_(I2)
        I_true = Interval(np.array([[-10], [2], [11]]), np.array([[-6], [8], [12]]))
        assert I.isequal(I_true)

    def test_and_degenerate_intersection(self):
        """Test bounded, degenerate intersection"""
        I1 = Interval(np.array([[-2], [3], [1]]), np.array([[5], [4], [1]]))
        I2 = Interval(np.array([[5], [3], [0]]), np.array([[8], [5], [2]]))
        I = I1.and_(I2)
        I_true = Interval(np.array([[5], [3], [1]]), np.array([[5], [4], [1]]))
        assert I.isequal(I_true)

    def test_and_empty_intersection(self):
        """Test bounded, empty intersection"""
        I1 = Interval(np.array([[-5], [-2]]), np.array([[2], [4]]))
        I2 = Interval(np.array([[-7], [6]]), np.array([[-3], [8]]))
        I = I1.and_(I2)
        assert I.representsa_('emptySet')

    def test_and_unbounded_bounded_intersection(self):
        """Test unbounded, bounded intersection"""
        I1 = Interval(np.array([[-np.inf]]), np.array([[2]]))
        I2 = Interval(np.array([[-2]]), np.array([[np.inf]]))
        I = I1.and_(I2)
        I_true = Interval(np.array([[-2]]), np.array([[2]]))
        assert I.isequal(I_true)

    def test_and_unbounded_unbounded_intersection(self):
        """Test unbounded, unbounded intersection"""
        I1 = Interval(np.array([[-np.inf]]), np.array([[2]]))
        I2 = Interval(np.array([[-np.inf]]), np.array([[np.inf]]))
        I = I1.and_(I2)
        I_true = Interval(np.array([[-np.inf]]), np.array([[2]]))
        assert I.isequal(I_true)

    def test_and_unbounded_empty_intersection(self):
        """Test unbounded, empty intersection"""
        I1 = Interval(np.array([[-np.inf]]), np.array([[-2]]))
        I2 = Interval(np.array([[2]]), np.array([[np.inf]]))
        I = I1.and_(I2)
        assert I.representsa_('emptySet')

    def test_and_unbounded_degenerate_intersection(self):
        """Test unbounded, degenerate intersection"""
        I1 = Interval(np.array([[-np.inf]]), np.array([[-1]]))
        I2 = Interval(np.array([[-1]]), np.array([[np.inf]]))
        I = I1.and_(I2)
        I_true = Interval(np.array([[-1]]), np.array([[-1]]))
        assert I.isequal(I_true)

    def test_and_empty_intervals(self):
        """Test empty intervals"""
        I = Interval(np.array([[1], [2]]), np.array([[3], [4]]))
        I_empty = Interval.empty(2)
        
        I_and = I.and_(I_empty)
        assert I_and.isequal(I_empty)
        
        I_and = I_empty.and_(I)
        assert I_and.isequal(I_empty)

    def test_and_operator_overload(self):
        """Test & operator overload"""
        I1 = Interval(np.array([[-10], [-2]]), np.array([[-5], [8]]))
        I2 = Interval(np.array([[-11], [2]]), np.array([[-6], [9]]))
        
        # Test both methods give same result
        I_method = I1.and_(I2)
        I_operator = I1 & I2
        assert I_method.isequal(I_operator)

    def test_and_matrix_intervals(self):
        """Test intersection of matrix intervals"""
        I1 = Interval(np.array([[-2, 1], [3, 2]]), np.array([[0, 2], [4, 3]]))
        I2 = Interval(np.array([[-1, 0], [2, 1]]), np.array([[1, 3], [5, 4]]))
        I = I1.and_(I2)
        I_true = Interval(np.array([[-1, 1], [3, 2]]), np.array([[0, 2], [4, 3]]))
        assert I.isequal(I_true)

    def test_and_self_intersection(self):
        """Test intersection of interval with itself"""
        I = Interval(np.array([[-2], [3]]), np.array([[5], [7]]))
        I_self = I.and_(I)
        assert I_self.isequal(I)

    def test_and_tolerance(self):
        """Test intersection with small numerical differences"""
        tol = 1e-12
        I1 = Interval(np.array([[0]]), np.array([[1]]))
        I2 = Interval(np.array([[1 - tol]]), np.array([[2]]))
        I = I1.and_(I2)
        
        # Should have degenerate intersection at boundary
        assert not I.representsa_('emptySet')
        assert I.rad()[0, 0] < tol * 10  # Very small radius

    def test_and_dimension_consistency(self):
        """Test that intersection maintains dimension consistency"""
        I1 = Interval(np.array([[-1], [2], [-3]]), np.array([[1], [4], [0]]))
        I2 = Interval(np.array([[0], [1], [-2]]), np.array([[2], [3], [1]]))
        I = I1.and_(I2)
        
        assert I.dim() == 3
        assert I.infimum().shape == (3, 1)
        assert I.supremum().shape == (3, 1)


if __name__ == "__main__":
    pytest.main([__file__]) 