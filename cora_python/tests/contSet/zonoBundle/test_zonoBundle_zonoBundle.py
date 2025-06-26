"""
test_zonoBundle - unit test function for ZonoBundle constructor

Tests the zonotope bundle constructor functionality.

Authors:       Matthias Althoff (MATLAB)
               Python translation by AI Assistant
Written:       02-August-2023
Last update:   ---
Last revision: ---
"""

import pytest
import numpy as np
from cora_python.contSet.zonoBundle.zonoBundle import ZonoBundle
from cora_python.contSet.zonotope.zonotope import Zonotope


class TestZonoBundle:
    """Test class for ZonoBundle constructor"""

    def test_constructor_basic(self):
        """Test basic constructor with zonotopes"""
        # Create simple zonotopes
        Z1 = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        Z2 = Zonotope(np.array([0, 1]), np.array([[2, 1], [1, 2]]))
        
        # Create zonotope bundle
        zB = ZonoBundle([Z1, Z2])
        
        assert isinstance(zB, ZonoBundle)
        assert zB.dim() == 2

    def test_constructor_single_zonotope(self):
        """Test constructor with single zonotope"""
        Z = Zonotope(np.array([1, 2]), np.array([[1, 0], [0, 1]]))
        zB = ZonoBundle([Z])
        
        assert isinstance(zB, ZonoBundle)
        assert zB.dim() == 2

    def test_constructor_multiple_zonotopes(self):
        """Test constructor with multiple zonotopes"""
        zonotopes = []
        for i in range(5):
            Z = Zonotope(np.array([i, i+1]), np.eye(2))
            zonotopes.append(Z)
        
        zB = ZonoBundle(zonotopes)
        assert isinstance(zB, ZonoBundle)
        assert zB.dim() == 2

    def test_constructor_validation(self):
        """Test constructor input validation"""
        Z = Zonotope(np.array([0, 0]), np.eye(2))
        zB = ZonoBundle([Z])
        assert isinstance(zB, ZonoBundle)

    def test_constructor_properties(self):
        """Test constructor sets proper properties"""
        Z1 = Zonotope(np.array([1, 2]), np.eye(2))
        Z2 = Zonotope(np.array([2, 3]), np.eye(2))
        zB = ZonoBundle([Z1, Z2])
        
        assert zB.dim() == 2
        assert not zB.isemptyobject()
        assert hasattr(zB, 'Z')
        assert hasattr(zB, 'parallelSets')
        assert zB.parallelSets == 2

    def test_constructor_various_dimensions(self):
        """Test constructor with various dimensions"""
        for n in [1, 3, 5]:
            Z1 = Zonotope(np.zeros(n), np.eye(n))
            Z2 = Zonotope(np.ones(n), np.eye(n))
            zB = ZonoBundle([Z1, Z2])
            assert isinstance(zB, ZonoBundle)
            assert zB.dim() == n

    def test_constructor_dimension_consistency(self):
        """Test that all zonotopes must have same dimension"""
        Z1 = Zonotope(np.array([1, 2]), np.eye(2))
        Z2 = Zonotope(np.array([1, 2, 3]), np.eye(3))  # Different dimension
        
        # Should raise error for inconsistent dimensions
        with pytest.raises((ValueError, Exception)):
            ZonoBundle([Z1, Z2])

    def test_constructor_copy(self):
        """Test copy constructor"""
        Z1 = Zonotope(np.array([1, 2]), np.eye(2))
        Z2 = Zonotope(np.array([2, 3]), np.eye(2))
        zB1 = ZonoBundle([Z1, Z2])
        
        # Copy constructor
        zB2 = ZonoBundle(zB1)
        assert isinstance(zB2, ZonoBundle)
        assert zB2.dim() == zB1.dim()
        assert zB2.parallelSets == zB1.parallelSets

    def test_constructor_empty_input(self):
        """Test constructor with empty input"""
        with pytest.raises((ValueError, Exception)):
            ZonoBundle()

    def test_constructor_invalid_zonotope(self):
        """Test constructor with invalid zonotope input"""
        # Non-zonotope object
        with pytest.raises((ValueError, Exception)):
            ZonoBundle([np.array([1, 2])])  # Not a zonotope 