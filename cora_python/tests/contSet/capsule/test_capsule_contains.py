"""
test_capsule_contains - unit test function for capsule contains method

Tests the contains method of capsule objects.

Syntax:
    pytest cora_python/tests/contSet/capsule/test_capsule_contains.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.capsule.capsule import Capsule
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestCapsuleContains:
    def test_contains_same_center_different_radius(self):
        """Test containment with same center and generator, different radius"""
        c = np.array([2, -1, 1])
        g = np.array([4, -3, 2])
        r_small = 0.5
        r_big = 1.0
        
        # instantiate capsules
        C_small = Capsule(c, g, r_small)
        C_big = Capsule(c, g, r_big)
        
        # compute containment - larger should contain smaller
        assert not C_small.contains(C_big)
        assert C_big.contains(C_small)

    def test_contains_centers_far_apart(self):
        """Test containment when centers are too far away"""
        c = np.array([2, -1, 1])
        c_plus = 10 * c
        c_minus = -c
        
        # norm of generator = 1, radius <= 1
        g = np.array([4, -3, 2])
        g = g / np.linalg.norm(g)
        r = 0.5
        
        C_plus = Capsule(c_plus, g, r)
        C_minus = Capsule(c_minus, g, r)
        
        # compute containment - should not contain each other
        assert not C_plus.contains(C_minus)
        assert not C_minus.contains(C_plus)

    def test_contains_overlapping_different_generators(self):
        """Test containment with overlapping capsules, different generators"""
        c = np.array([2, -1, 1])
        g1 = np.array([4, -3, 2])
        g2 = g1 + np.array([-1, 0.2, 0.5])
        r = 0.5
        
        C1 = Capsule(c, g1, r)
        C2 = Capsule(c, g2, r)
        
        # compute containment - should not contain each other
        assert not C1.contains(C2)
        assert not C2.contains(C1)

    def test_contains_point_inside(self):
        """Test point containment - point inside"""
        C = Capsule(np.array([1, 1, 0]), np.array([0.5, -1, 1]), 0.5)
        
        # create a point inside (center)
        p_inside = C.center().flatten()
        
        # check if correct result for containment
        assert C.contains(p_inside)

    def test_contains_point_outside(self):
        """Test point containment - point outside"""
        C = Capsule(np.array([1, 1, 0]), np.array([0.5, -1, 1]), 0.5)
        
        # create a point outside
        p_outside = 10 * np.ones(C.dim())
        
        # check if correct result for containment
        assert not C.contains(p_outside)

    def test_contains_point_array_outside(self):
        """Test point array containment - all points outside"""
        C = Capsule(np.array([1, 1, 0]), np.array([0.5, -1, 1]), 0.5)
        
        # array of points (all outside)
        num = 10
        p_array = 10 * (np.ones((C.dim(), num)) + np.random.rand(C.dim(), num))
        
        # check that none are contained
        result = C.contains(p_array)
        if isinstance(result, np.ndarray):
            assert not np.any(result)
        else:
            assert not result

    def test_contains_dimension_mismatch(self):
        """Test dimension mismatch error"""
        C1 = Capsule(np.array([1]), np.array([1]), 1)
        C2 = Capsule(np.random.rand(2), np.random.rand(2), 1)
        
        with pytest.raises((CORAerror, ValueError, AssertionError)):
            C1.contains(C2)

    def test_contains_empty_capsule(self):
        """Test containment with empty capsule"""
        n = 2
        C_empty = Capsule.empty(n)
        C_normal = Capsule(np.array([1, 2]), np.array([0.5, -0.5]), 0.3)
        
        # Empty capsule should not contain anything, but normal capsule should contain empty
        assert not C_empty.contains(C_normal)
        assert C_normal.contains(C_empty)

    def test_contains_point_on_boundary(self):
        """Test point exactly on boundary"""
        c = np.array([0, 0])
        g = np.array([1, 0])
        r = 0.5
        C = Capsule(c, g, r)
        
        # Point on boundary (at distance r from line segment)
        p_boundary = np.array([0, r])  # Should be on boundary
        
        # This should be contained (boundary is included)
        assert C.contains(p_boundary)

    def test_contains_identical_capsules(self):
        """Test containment of identical capsules"""
        c = np.array([1, 2, 3])
        g = np.array([0, 1, 0])
        r = 0.8
        
        C1 = Capsule(c, g, r)
        C2 = Capsule(c, g, r)
        
        # Identical capsules should contain each other
        assert C1.contains(C2)
        assert C2.contains(C1)

    def test_contains_1d_capsule(self):
        """Test containment in 1D"""
        # 1D capsules are line segments with radius
        C1 = Capsule(np.array([0]), np.array([2]), 0.5)  # [-2, 2] with radius 0.5
        C2 = Capsule(np.array([0]), np.array([1]), 0.2)  # [-1, 1] with radius 0.2
        
        # C1 should contain C2
        assert C1.contains(C2)
        assert not C2.contains(C1)

    def test_contains_point_capsule(self):
        """Test containment with point capsule (r=0, g=0)"""
        c1 = np.array([1, 2])
        C_point = Capsule(c1, np.array([0, 0]), 0)
        C_normal = Capsule(c1, np.array([1, 0]), 0.5)
        
        # Normal capsule should contain point capsule at same center
        assert C_normal.contains(C_point)
        assert not C_point.contains(C_normal)


if __name__ == "__main__":
    pytest.main([__file__]) 