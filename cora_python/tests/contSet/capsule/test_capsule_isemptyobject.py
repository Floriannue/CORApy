"""
test_capsule_isemptyobject - unit test function for capsule isemptyobject method

Tests the isemptyobject method of capsule objects.

Syntax:
    pytest cora_python/tests/contSet/capsule/test_capsule_isemptyobject.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.capsule.capsule import Capsule


class TestCapsuleIsEmptyObject:
    def test_isemptyobject_empty_capsule(self):
        """Test isemptyobject for empty capsule"""
        # 2D, empty
        C1 = Capsule.empty(2)
        assert C1.isemptyobject()

    def test_isemptyobject_bounded_capsule(self):
        """Test isemptyobject for bounded capsule"""
        # 2D, bounded
        C2 = Capsule(np.array([1, 1]), np.array([0, 1]), 0.5)
        assert not C2.isemptyobject()

    def test_isemptyobject_different_dimensions(self):
        """Test isemptyobject for different dimensions"""
        # 1D empty
        C1 = Capsule.empty(1)
        assert C1.isemptyobject()
        
        # 3D empty
        C3 = Capsule.empty(3)
        assert C3.isemptyobject()
        
        # 5D empty
        C5 = Capsule.empty(5)
        assert C5.isemptyobject()

    def test_isemptyobject_point_capsule(self):
        """Test isemptyobject for point capsule (r=0, g=0)"""
        # Point capsule should not be empty object
        C_point = Capsule(np.array([1, 2]), np.array([0, 0]), 0)
        assert not C_point.isemptyobject()

    def test_isemptyobject_zero_radius(self):
        """Test isemptyobject for capsule with zero radius but non-zero generator"""
        # Line segment (r=0 but g!=0) should not be empty object
        C_line = Capsule(np.array([0, 0]), np.array([1, 1]), 0)
        assert not C_line.isemptyobject()

    def test_isemptyobject_origin_capsule(self):
        """Test isemptyobject for origin capsule"""
        # Origin capsule should not be empty object
        C_origin = Capsule.origin(3)
        assert not C_origin.isemptyobject()

    def test_isemptyobject_various_bounded_capsules(self):
        """Test isemptyobject for various bounded capsules"""
        # 1D bounded
        C1 = Capsule(np.array([5]), np.array([2]), 0.1)
        assert not C1.isemptyobject()
        
        # 2D bounded with large radius
        C2 = Capsule(np.array([0, 0]), np.array([1, 0]), 10.0)
        assert not C2.isemptyobject()
        
        # 3D bounded
        C3 = Capsule(np.array([1, 2, 3]), np.array([0, 1, 0]), 0.5)
        assert not C3.isemptyobject()

    def test_isemptyobject_high_dimension(self):
        """Test isemptyobject for high-dimensional capsules"""
        n = 10
        
        # Empty high-dimensional capsule
        C_empty = Capsule.empty(n)
        assert C_empty.isemptyobject()
        
        # Bounded high-dimensional capsule
        c = np.random.rand(n)
        g = np.random.rand(n)
        r = 0.1
        C_bounded = Capsule(c, g, r)
        assert not C_bounded.isemptyobject()


if __name__ == "__main__":
    pytest.main([__file__]) 