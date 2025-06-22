"""
test_capsule_empty - unit test function for capsule empty method

Tests the empty static method of capsule objects.

Syntax:
    pytest cora_python/tests/contSet/capsule/test_capsule_empty.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.capsule.capsule import Capsule


class TestCapsuleEmpty:
    def test_empty_2d_capsule(self):
        """Test empty 2D capsule"""
        n = 2
        C = Capsule.empty(n)
        
        assert C.dim() == n
        assert C.representsa_('emptySet')

    def test_empty_different_dimensions(self):
        """Test empty capsules in different dimensions"""
        for n in [1, 3, 4, 5, 10]:
            C = Capsule.empty(n)
            assert C.dim() == n
            assert C.representsa_('emptySet')
            assert C.isemptyobject()

    def test_empty_properties(self):
        """Test properties of empty capsule"""
        n = 3
        C = Capsule.empty(n)
        
        # Check that center has correct shape but is empty
        center = C.center()
        assert center.shape == (n, 0)
        assert center.size == 0
        
        # Check that it's recognized as empty
        assert C.isemptyobject()

    def test_empty_zero_dimension(self):
        """Test empty capsule with zero dimension"""
        # This might not be valid, but test that it handles gracefully
        with pytest.raises((ValueError, AssertionError)):
            Capsule.empty(0)

    def test_empty_negative_dimension(self):
        """Test empty capsule with negative dimension"""
        with pytest.raises((ValueError, AssertionError)):
            Capsule.empty(-1)

    def test_empty_large_dimension(self):
        """Test empty capsule with large dimension"""
        n = 100
        C = Capsule.empty(n)
        assert C.dim() == n
        assert C.representsa_('emptySet')

    def test_empty_consistency(self):
        """Test that multiple empty capsules of same dimension are consistent"""
        n = 4
        C1 = Capsule.empty(n)
        C2 = Capsule.empty(n)
        
        # Both should have same dimension and represent empty set
        assert C1.dim() == C2.dim()
        assert C1.representsa_('emptySet')
        assert C2.representsa_('emptySet')
        assert C1.isemptyobject()
        assert C2.isemptyobject()

    def test_empty_type_validation(self):
        """Test type validation for empty method"""
        # Non-integer dimension should raise error
        with pytest.raises((ValueError, TypeError, AssertionError)):
            Capsule.empty(2.5)
        
        # Non-numeric dimension should raise error
        with pytest.raises((ValueError, TypeError, AssertionError)):
            Capsule.empty("2")


if __name__ == "__main__":
    pytest.main([__file__]) 