"""
test_capsule - unit test function for capsule constructor

Tests the basic constructor and properties of capsule objects.

Syntax:
    pytest cora_python/tests/contSet/capsule/test_capsule.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.capsule.capsule import Capsule


class TestCapsule:
    def test_capsule_constructor_center_only(self):
        """Test capsule constructor with center only"""
        c = np.array([2, 0])
        C = Capsule(c)
        
        # Capsule stores center as column vector
        expected_c = np.array([[2], [0]])
        expected_g = np.array([[0], [0]])
        
        assert np.allclose(C.c, expected_c)
        assert np.allclose(C.g, expected_g)
        assert C.r == 0

    def test_capsule_constructor_center_generator(self):
        """Test capsule constructor with center and generator"""
        c = np.array([2, 0])
        g = np.array([1, -1])
        C = Capsule(c, g)
        
        # Capsule stores center and generator as column vectors
        expected_c = np.array([[2], [0]])
        expected_g = np.array([[1], [-1]])
        
        assert np.allclose(C.c, expected_c)
        assert np.allclose(C.g, expected_g)
        assert C.r == 0

    def test_capsule_constructor_full(self):
        """Test capsule constructor with center, generator, and radius"""
        c = np.array([2, 0])
        g = np.array([1, -1])
        r = 0.5
        C = Capsule(c, g, r)
        
        # Capsule stores center and generator as column vectors
        expected_c = np.array([[2], [0]])
        expected_g = np.array([[1], [-1]])
        
        assert np.allclose(C.c, expected_c)
        assert np.allclose(C.g, expected_g)
        assert C.r == r

    def test_capsule_empty_constructor(self):
        """Test empty capsule constructor"""
        n = 2
        C = Capsule.empty(n)
        
        assert C.dim() == n
        assert C.representsa_('emptySet')

    def test_capsule_origin_constructor(self):
        """Test origin capsule constructor"""
        n = 3
        C = Capsule.origin(n)
        
        assert C.dim() == n
        assert np.allclose(C.c, np.zeros((n, 1)))
        assert np.allclose(C.g, np.zeros((n, 1)))
        assert C.r == 0

    def test_capsule_dimension_consistency(self):
        """Test dimension consistency in constructor"""
        c = np.array([1, 2, 3])
        g = np.array([0, 1, 0])
        r = 1.5
        
        C = Capsule(c, g, r)
        assert C.dim() == 3

    def test_capsule_input_validation(self):
        """Test input validation"""
        c = np.array([1, 2])
        g_wrong_dim = np.array([1, 2, 3])  # Wrong dimension
        
        with pytest.raises((ValueError, AssertionError)):
            Capsule(c, g_wrong_dim)

    def test_capsule_negative_radius(self):
        """Test handling of negative radius"""
        c = np.array([0, 0])
        g = np.array([1, 0])
        r = -0.5
        
        with pytest.raises((ValueError, AssertionError)):
            Capsule(c, g, r)

    def test_capsule_properties_access(self):
        """Test accessing capsule properties"""
        c = np.array([1, 2])
        g = np.array([3, 4])
        r = 2.5
        C = Capsule(c, g, r)
        
        # Test property access - capsule stores as column vectors
        expected_c = np.array([[1], [2]])
        expected_g = np.array([[3], [4]])
        
        assert np.allclose(C.c, expected_c)
        assert np.allclose(C.g, expected_g)
        assert C.r == r

    def test_capsule_different_dimensions(self):
        """Test capsule in different dimensions"""
        # 1D capsule
        C_1d = Capsule(np.array([5]), np.array([2]), 1.0)
        assert C_1d.dim() == 1
        
        # 4D capsule
        c_4d = np.array([1, 2, 3, 4])
        g_4d = np.array([0, 1, 0, 1])
        C_4d = Capsule(c_4d, g_4d, 0.5)
        assert C_4d.dim() == 4


if __name__ == "__main__":
    pytest.main([__file__]) 