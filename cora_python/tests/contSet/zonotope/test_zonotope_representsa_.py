"""
test_zonotope_representsa_ - unit test function of representsa_

Tests the representsa_ method for zonotope objects to check if a zonotope
can represent various set types like empty set, origin, interval, point, etc.

Syntax:
    pytest cora_python/tests/contSet/zonotope/test_zonotope_representsa_.py

Authors: Florian Lercher (translated from MATLAB)
Date: 2025-01-11
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


class TestZonotopeRepresentsa:
    def test_representsa_empty_set(self):
        """Test comparison to empty set"""
        # Empty zonotope
        Z_empty = Zonotope.empty(2)
        assert Z_empty.representsa_('emptySet')
        
        # Non-empty zonotope
        Z = Zonotope(np.array([[1], [5], [-1]]), 
                    np.array([[2, 4], [6, 0], [4, 8]]))
        assert not Z.representsa_('emptySet')

    def test_representsa_origin(self):
        """Test comparison to origin"""
        # Empty zonotope
        Z = Zonotope.empty(2)
        assert not Z.representsa_('origin')
        
        # Only origin (point at origin)
        Z = Zonotope(np.zeros((3, 1)))
        assert Z.representsa_('origin')
        
        # Shifted center
        Z = Zonotope(0.01 * np.ones((4, 1)))
        assert not Z.representsa_('origin')
        
        # ...add tolerance
        tol = 0.02
        assert Z.representsa_('origin', tol)
        
        # Include generator matrix
        Z = Zonotope(np.ones((2, 1)), 0.1 * np.eye(2))
        tol = 2
        assert Z.representsa_('origin', tol)

    def test_representsa_interval(self):
        """Test comparison to interval"""
        # Create zonotopes
        c1 = np.array([[0], [0]])
        G1 = np.array([[2, 0], [0, 1]])
        Z = Zonotope(c1, G1)
        res, I = Z.representsa_('interval', return_set=True)
        assert res
        I_true = Interval(np.array([[-2], [-1]]), np.array([[2], [1]]))
        assert I.isequal(I_true)
        
        # Non-axis aligned zonotope
        c2 = np.array([[1], [0]])
        G2 = np.array([[2, 1], [-1, 4]])
        Z = Zonotope(c2, G2)
        assert not Z.representsa_('interval')

    def test_representsa_parallelotope(self):
        """Test comparison to parallelotope"""
        # Check empty zonotope
        Z = Zonotope.empty(2)
        assert not Z.representsa_('parallelotope')
        
        # Instantiate parallelotope
        c = np.array([[-2], [1]])
        G = np.array([[2, 4], [-2, 3]])
        Z = Zonotope(c, G)
        assert Z.representsa_('parallelotope')
        
        # Add zero-length generators
        G_with_zeros = np.hstack([G, np.zeros((2, 2))])
        Z = Zonotope(c, G_with_zeros)
        assert Z.representsa_('parallelotope')
        
        # Add generator -> not a parallelotope anymore
        G_extra = np.hstack([G_with_zeros, np.array([[4], [-2]])])
        Z = Zonotope(c, G_extra)
        assert not Z.representsa_('parallelotope')
        
        # No generator matrix (point)
        Z = Zonotope(c)
        assert not Z.representsa_('parallelotope')

    def test_representsa_point(self):
        """Test comparison to point"""
        # Point zonotope
        Z = Zonotope(np.ones((4, 1)))
        assert Z.representsa_('point')
        
        # Point with small generator
        Z = Zonotope(np.array([[3], [2], [1]]), 
                    np.array([[0], [0], [np.finfo(float).eps]]))
        assert Z.representsa_('point', 1e-10)

    def test_representsa_conZonotope(self):
        """Test comparison to conZonotope"""
        # Always true for zonotope (every zonotope is a constrained zonotope)
        Z = Zonotope(np.array([[1], [-1], [2]]), 
                    np.array([[1, -3, 0], [3, -1, 1], [-2, 0, 1]]))
        res, S = Z.representsa_('conZonotope', return_set=True)
        assert res
        # The returned constrained zonotope should also represent a zonotope
        assert S.representsa_('zonotope')

    def test_representsa_polyZonotope(self):
        """Test comparison to polyZonotope"""
        # Always true for zonotope (every zonotope is a polynomial zonotope)
        Z = Zonotope(np.array([[1], [-1], [2]]), 
                    np.array([[1, -3, 0], [3, -1, 1], [-2, 0, 1]]))
        res, S = Z.representsa_('polyZonotope', return_set=True)
        assert res
        # The returned polynomial zonotope should also represent a zonotope
        assert S.representsa_('zonotope')

    def test_representsa_capsule(self):
        """Test comparison to capsule"""
        # 2D zonotope that can represent a capsule (line segment + radius)
        c = np.array([[0], [0]])
        g = np.array([[1], [0]])  # Single generator along x-axis
        Z = Zonotope(c, g)
        res, C = Z.representsa_('capsule', return_set=True)
        assert res
        # Verify the capsule properties
        assert np.allclose(C.c, c.flatten())
        assert np.allclose(C.g, g)  # Compare with original column vector
        assert np.isclose(C.r, 0)  # No radius for line segment

    def test_representsa_tolerance_handling(self):
        """Test tolerance handling in representsa_"""
        # Create zonotope that's almost a point
        Z = Zonotope(np.array([[1], [1]]), 
                    np.array([[1e-12], [1e-12]]))
        
        # Without tolerance should not be a point
        assert not Z.representsa_('point')
        
        # With tolerance should be a point
        assert Z.representsa_('point', 1e-10)

    def test_representsa_unsupported_type(self):
        """Test handling of unsupported set types"""
        Z = Zonotope(np.array([[1], [1]]), np.array([[1], [0]]))
        
        # Test unsupported type
        with pytest.raises(CORAerror):
            Z.representsa_('unsupported_type')


if __name__ == "__main__":
    pytest.main([__file__]) 