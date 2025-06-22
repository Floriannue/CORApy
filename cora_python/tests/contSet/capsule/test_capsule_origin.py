"""
test_capsule_origin - unit test function for capsule origin method

Tests the origin static method of capsule objects.

Syntax:
    pytest cora_python/tests/contSet/capsule/test_capsule_origin.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.capsule.capsule import Capsule


class TestCapsuleOrigin:
    def test_origin_1d(self):
        """Test origin capsule in 1D"""
        C = Capsule.origin(1)
        C_true = Capsule(np.array([0]))
        
        # Check that center is at origin
        assert np.allclose(C.center(), np.array([[0]]))
        assert C.representsa_('origin')
        
        # Should contain the origin point
        # Note: contains method might not be implemented yet
        # assert C.contains(np.array([0]))

    def test_origin_2d(self):
        """Test origin capsule in 2D"""
        C = Capsule.origin(2)
        C_true = Capsule(np.zeros(2))
        
        # Check that center is at origin
        assert np.allclose(C.center(), np.zeros((2, 1)))
        assert C.representsa_('origin')
        
        # Should contain the origin point
        # assert C.contains(np.zeros(2))

    def test_origin_different_dimensions(self):
        """Test origin capsule in different dimensions"""
        for n in [1, 2, 3, 4, 5, 10]:
            C = Capsule.origin(n)
            
            # Check dimension
            assert C.dim() == n
            
            # Check that center is at origin
            expected_center = np.zeros((n, 1))
            assert np.allclose(C.center(), expected_center)
            
            # Should represent origin
            assert C.representsa_('origin')

    def test_origin_properties(self):
        """Test properties of origin capsule"""
        n = 3
        C = Capsule.origin(n)
        
        # Should be a point (no generator, no radius)
        assert np.allclose(C.g, np.zeros((n, 1)))
        assert C.r == 0
        assert np.allclose(C.c, np.zeros((n, 1)))

    def test_origin_wrong_calls(self):
        """Test wrong calls to origin method"""
        # Zero dimension
        with pytest.raises((ValueError, AssertionError)):
            Capsule.origin(0)
        
        # Negative dimension
        with pytest.raises((ValueError, AssertionError)):
            Capsule.origin(-1)
        
        # Non-integer dimension
        with pytest.raises((ValueError, TypeError, AssertionError)):
            Capsule.origin(0.5)
        
        # Array input
        with pytest.raises((ValueError, TypeError, AssertionError)):
            Capsule.origin([1, 2])
        
        # String input
        with pytest.raises((ValueError, TypeError, AssertionError)):
            Capsule.origin('text')

    def test_origin_consistency(self):
        """Test consistency of origin capsules"""
        n = 4
        C1 = Capsule.origin(n)
        C2 = Capsule.origin(n)
        
        # Both should be identical
        assert np.allclose(C1.center(), C2.center())
        assert np.allclose(C1.g, C2.g)
        assert C1.r == C2.r
        assert C1.dim() == C2.dim()

    def test_origin_vs_manual_construction(self):
        """Test origin capsule vs manual construction"""
        n = 5
        C_origin = Capsule.origin(n)
        C_manual = Capsule(np.zeros(n), np.zeros(n), 0)
        
        # Should be equivalent
        assert np.allclose(C_origin.center(), C_manual.center())
        assert np.allclose(C_origin.g, C_manual.g)
        assert C_origin.r == C_manual.r

    def test_origin_large_dimension(self):
        """Test origin capsule with large dimension"""
        n = 100
        C = Capsule.origin(n)
        
        assert C.dim() == n
        assert np.allclose(C.center(), np.zeros((n, 1)))
        assert C.representsa_('origin')


if __name__ == "__main__":
    pytest.main([__file__]) 