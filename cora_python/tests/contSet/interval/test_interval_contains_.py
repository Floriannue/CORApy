"""
test_interval_contains_ - unit test function of contains_

Tests the contains_ method for interval objects to check containment.

Syntax:
    pytest cora_python/tests/contSet/interval/test_interval_contains_.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.interval.interval import Interval


class TestIntervalContains:
    def test_contains_point_inside(self):
        """Test containment of point inside interval"""
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        point = np.array([[0], [2]])
        assert I.contains_(point)

    def test_contains_point_outside(self):
        """Test non-containment of point outside interval"""
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        point = np.array([[5], [2]])
        assert not I.contains_(point)

    def test_contains_point_on_boundary(self):
        """Test containment of point on boundary"""
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        
        # Points on boundary should be contained
        point_lower = np.array([[-2], [1]])
        point_upper = np.array([[3], [4]])
        point_mixed = np.array([[-2], [4]])
        
        assert I.contains_(point_lower)
        assert I.contains_(point_upper)
        assert I.contains_(point_mixed)

    def test_contains_interval_subset(self):
        """Test containment of interval subset"""
        I1 = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        I2 = Interval(np.array([[-1], [2]]), np.array([[2], [3]]))
        assert I1.contains_(I2)

    def test_contains_interval_not_subset(self):
        """Test non-containment of interval that's not a subset"""
        I1 = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        I2 = Interval(np.array([[-3], [2]]), np.array([[2], [5]]))
        assert not I1.contains_(I2)

    def test_contains_interval_equal(self):
        """Test containment of equal intervals"""
        I1 = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        I2 = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        assert I1.contains_(I2)

    def test_contains_empty_set(self):
        """Test containment of empty set"""
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        I_empty = Interval.empty(2)
        
        # Any interval contains the empty set
        assert I.contains_(I_empty)

    def test_contains_empty_set_contains_nothing(self):
        """Test that empty set contains nothing except empty set"""
        I_empty = Interval.empty(2)
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        point = np.array([[0], [2]])
        
        # Empty set contains nothing
        assert not I_empty.contains_(I)
        assert not I_empty.contains_(point)
        
        # But empty set contains empty set
        assert I_empty.contains_(I_empty)

    def test_contains_unbounded_intervals(self):
        """Test containment with unbounded intervals"""
        I_unbounded = Interval(np.array([[-np.inf]]), np.array([[np.inf]]))
        I_bounded = Interval(np.array([[-2]]), np.array([[3]]))
        point = np.array([[100]])
        
        # Unbounded interval contains everything
        assert I_unbounded.contains_(I_bounded)
        assert I_unbounded.contains_(point)
        
        # Bounded interval doesn't contain unbounded
        assert not I_bounded.contains_(I_unbounded)

    def test_contains_with_tolerance(self):
        """Test containment with tolerance"""
        I = Interval(np.array([[0]]), np.array([[1]]))
        
        # Point slightly outside
        point = np.array([[1.001]])
        assert not I.contains_(point)
        
        # Should be contained with tolerance
        tol = 0.01
        assert I.contains_(point, tol)

    def test_contains_degenerate_interval(self):
        """Test containment for degenerate (point) intervals"""
        I_point = Interval(np.array([[2], [3]]), np.array([[2], [3]]))
        
        # Same point should be contained
        point_same = np.array([[2], [3]])
        assert I_point.contains_(point_same)
        
        # Different point should not be contained
        point_diff = np.array([[2.1], [3]])
        assert not I_point.contains_(point_diff)

    def test_contains_matrix_intervals(self):
        """Test containment for matrix intervals"""
        I1 = Interval(np.array([[-2, 0], [1, -1]]), np.array([[3, 2], [4, 1]]))
        I2 = Interval(np.array([[-1, 0.5], [2, 0]]), np.array([[2, 1.5], [3, 0.5]]))
        
        assert I1.contains_(I2)

    def test_contains_dimension_mismatch(self):
        """Test dimension mismatch in containment"""
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        point_wrong_dim = np.array([[0], [2], [1]])
        
        with pytest.raises((ValueError, AssertionError)):
            I.contains_(point_wrong_dim)

    def test_contains_multiple_points(self):
        """Test containment of multiple points"""
        I = Interval(np.array([[-2], [1]]), np.array([[3], [4]]))
        
        # Matrix of points (each column is a point)
        points = np.array([[0, 5, -1], [2, 2, 3]])
        
        # Should return array of results
        result = I.contains_(points)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__]) 