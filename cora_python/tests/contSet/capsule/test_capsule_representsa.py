"""
test_capsule_representsa - unit test function for capsule representsa_ method

Tests the representsa_ method of capsule objects.

Syntax:
    pytest cora_python/tests/contSet/capsule/test_capsule_representsa.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.capsule.capsule import Capsule


class TestCapsuleRepresentsa:
    def test_representsa_empty_set(self):
        """Test representsa for empty set"""
        # Empty capsule should represent empty set
        C = Capsule.empty(2)
        assert C.representsa_('emptySet')
        
        # Bounded capsule should not represent empty set
        C = Capsule(np.array([1, 1]), np.array([0, 1]), 0.5)
        assert not C.representsa_('emptySet')
        
        # Line segment (r=0) should not represent empty set
        C = Capsule(np.array([1, 1]), np.array([0, 1]), 0)
        assert not C.representsa_('emptySet')

    def test_representsa_interval(self):
        """Test representsa for interval"""
        # Empty capsule should represent interval
        C = Capsule.empty(2)
        assert C.representsa_('interval')
        
        # Full-dimensional capsule should not represent interval
        c = np.array([2, 0, -1])
        g = np.array([0.2, -0.7, 0.4])
        r = 1
        C = Capsule(c, g, r)
        assert not C.representsa_('interval')
        
        # One-dimensional capsule should represent interval
        C = Capsule(np.array([2]), np.array([1]), 0)
        result = C.representsa_('interval')
        assert result
        
        # Two-dimensional capsule with axis-aligned generator and no radius
        C = Capsule(np.array([1, -1]), np.array([1, 0]), 0)
        assert C.representsa_('interval')
        
        # Two-dimensional capsule with all-zero generator and radius
        C = Capsule(np.array([0, -1]), np.array([0, 0]), 1)
        assert not C.representsa_('interval')
        
        # Two-dimensional capsule with all-zero generator and no radius
        C = Capsule(np.array([0, -1]), np.array([0, 0]), 0)
        assert C.representsa_('interval')

    def test_representsa_origin(self):
        """Test representsa for origin"""
        # Empty case should not represent origin
        C = Capsule.empty(2)
        assert not C.representsa_('origin')
        
        # True cases - point at origin
        C = Capsule(np.zeros(3))
        assert C.representsa_('origin')
        
        C = Capsule(np.zeros(3), np.zeros(3), 0)
        assert C.representsa_('origin')
        
        # Shifted center should not represent origin
        C = Capsule(np.ones(3), np.zeros(3), 0)
        assert not C.representsa_('origin')
        
        # Including generator, no radius should not represent origin
        C = Capsule(np.zeros(2), np.ones(2), 0)
        assert not C.representsa_('origin')
        
        # No generator, but radius should not represent origin
        C = Capsule(np.zeros(4), np.zeros(4), 1)
        assert not C.representsa_('origin')

    def test_representsa_origin_with_tolerance(self):
        """Test representsa for origin with tolerance"""
        # Within tolerance
        C = Capsule(np.zeros(3), np.zeros(3), 1)
        tol = 2
        assert C.representsa_('origin', tol)
        
        # Does not contain origin, but within tolerance
        c = 0.5 * np.array([np.sqrt(2), np.sqrt(2)])
        g = 0.5 * np.array([np.sqrt(2), -np.sqrt(2)])
        r = 0.5
        tol = 2
        C = Capsule(c, g, r)
        assert C.representsa_('origin', tol)

    def test_representsa_ellipsoid(self):
        """Test representsa for ellipsoid"""
        # Only center (point) should represent ellipsoid
        C = Capsule(np.array([3, 1, 2]))
        result = C.representsa_('ellipsoid')
        assert result
        
        # Ball (zero generator, non-zero radius) should represent ellipsoid
        C = Capsule(np.array([4, 1, -2, 3]), np.zeros(4), 2)
        assert C.representsa_('ellipsoid')
        
        # Regular capsule should not represent ellipsoid
        C = Capsule(np.array([3, 1]), np.array([-1, 2]), 1)
        assert not C.representsa_('ellipsoid')

    def test_representsa_zonotope(self):
        """Test representsa for zonotope"""
        # Only center (point) should represent zonotope
        C = Capsule(np.array([3, 1, 2]))
        assert C.representsa_('zonotope')
        
        # Center and one generator (line segment) should represent zonotope
        C = Capsule(np.array([3, 1, 2]), np.array([-1, 1, 0]))
        assert C.representsa_('zonotope')
        
        # Regular capsule (with radius) should not represent zonotope
        C = Capsule(np.array([-1, 2]), np.array([1, -1]), 1)
        assert not C.representsa_('zonotope')

    def test_representsa_point(self):
        """Test representsa for point"""
        # Point capsule should represent point
        C = Capsule(np.array([3, 1, 2]))
        assert C.representsa_('point')
        
        # Regular capsule should not represent point
        C = Capsule(np.array([-1, 2]), np.array([1, -1]), 1)
        assert not C.representsa_('point')
        
        # Line segment should not represent point
        C = Capsule(np.array([0, 0]), np.array([1, 0]), 0)
        assert not C.representsa_('point')

    def test_representsa_different_dimensions(self):
        """Test representsa for different dimensions"""
        # 1D point
        C1 = Capsule(np.array([5]))
        assert C1.representsa_('point')
        assert C1.representsa_('zonotope')
        assert C1.representsa_('ellipsoid')
        
        # 4D empty
        C4 = Capsule.empty(4)
        assert C4.representsa_('emptySet')
        assert C4.representsa_('interval')

    def test_representsa_invalid_type(self):
        """Test representsa with invalid type"""
        C = Capsule(np.array([1, 2]), np.array([0, 1]), 0.5)
        
        # Should handle invalid types gracefully
        assert not C.representsa_('invalidType')

    def test_representsa_edge_cases(self):
        """Test representsa edge cases"""
        # Zero-dimensional (should not occur, but test robustness)
        # This might not be valid, but test that it doesn't crash
        
        # Very small values
        C = Capsule(np.array([1e-10, 1e-10]), np.array([1e-10, 1e-10]), 1e-10)
        # Should still work for basic types
        assert not C.representsa_('emptySet')


if __name__ == "__main__":
    pytest.main([__file__]) 