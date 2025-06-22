"""
Test cases for ellipsoid representsa_ method.
"""

import numpy as np
import pytest

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.representsa_ import representsa_
from cora_python.contSet.ellipsoid.empty import empty
from cora_python.contSet.zonotope.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class TestEllipsoidRepresentsa:
    """Test class for ellipsoid representsa_ method."""
    
    def test_representsa_empty_set(self):
        """Test comparison to empty set."""
        # Empty ellipsoid
        E = empty(2)
        assert representsa_(E, 'emptySet', 1e-9)
        
        # Non-empty ellipsoid
        E = Ellipsoid(np.array([[1, 0], [0, 2]]), np.array([[0], [1]]))
        assert not representsa_(E, 'emptySet', 1e-9)
    
    def test_representsa_origin(self):
        """Test comparison to origin."""
        # Empty case
        E = empty(2)
        assert not representsa_(E, 'origin')
        
        # Only origin (point at origin with zero shape matrix)
        E = Ellipsoid(np.zeros((3, 3)), np.zeros((3, 1)))
        assert representsa_(E, 'origin')
        
        # Shifted center
        E = Ellipsoid(np.zeros((3, 3)), 0.01 * np.ones((3, 1)))
        assert not representsa_(E, 'origin')
        
        # Shifted center, contains origin within tolerance
        E = Ellipsoid(0.01 * np.eye(3), 0.01 * np.ones((3, 1)))
        tol = 0.15
        assert representsa_(E, 'origin', tol)
    
    def test_representsa_point(self):
        """Test comparison to point."""
        # Point ellipsoid (zero shape matrix)
        E = Ellipsoid(np.zeros((4, 4)), np.array([[3], [2], [-1], [4]]))
        assert representsa_(E, 'point')
        
        # Degenerate ellipsoid (not a point)
        Q = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        E = Ellipsoid(Q, np.array([[1], [2], [-1]]))
        assert not representsa_(E, 'point')
        
        # Full-dimensional ellipsoid
        E = Ellipsoid(np.eye(4), np.array([[3], [2], [-1], [4]]))
        assert not representsa_(E, 'point')
    
    def test_representsa_capsule(self):
        """Test comparison to capsule."""
        # Point ellipsoid
        E = Ellipsoid(np.zeros((2, 2)), np.array([[1], [0]]))
        assert representsa_(E, 'capsule')
        
        # 1D ellipsoid
        E = Ellipsoid(np.array([[2]]), np.array([[1]]))
        assert representsa_(E, 'capsule')
        
        # Ball (diagonal matrix with equal entries)
        E = Ellipsoid(np.eye(3), np.array([[0], [0], [0]]))
        assert representsa_(E, 'capsule')
        
        # General ellipsoid (not a capsule)
        E = Ellipsoid(np.array([[1, 0], [0, 2]]), np.array([[0], [0]]))
        assert not representsa_(E, 'capsule')
    
    def test_representsa_interval(self):
        """Test comparison to interval."""
        # Point ellipsoid
        E = Ellipsoid(np.zeros((2, 2)), np.array([[1], [2]]))
        assert representsa_(E, 'interval')
        
        # 1D ellipsoid
        E = Ellipsoid(np.array([[4]]), np.array([[1]]))
        assert representsa_(E, 'interval')
        
        # Multi-dimensional ellipsoid
        E = Ellipsoid(np.eye(2), np.array([[0], [0]]))
        assert not representsa_(E, 'interval')
    
    def test_representsa_zonotope(self):
        """Test comparison to zonotope."""
        # Point ellipsoid
        E = Ellipsoid(np.zeros((2, 2)), np.array([[2], [1]]))
        res, Z = representsa_(E, 'zonotope', 1e-9, 'return_set')
        assert res
        assert isinstance(Z, Zonotope)
        assert np.allclose(Z.c, np.array([[2], [1]]))
        
        # 1D ellipsoid
        E = Ellipsoid(np.array([[1]]), np.array([[3]]))
        assert representsa_(E, 'zonotope')
        
        # Multi-dimensional ellipsoid
        E = Ellipsoid(np.eye(2), np.array([[0], [0]]))
        assert not representsa_(E, 'zonotope')
    
    def test_representsa_ellipsoid(self):
        """Test comparison to ellipsoid (always true)."""
        E = Ellipsoid(np.eye(3), np.array([[1], [2], [3]]))
        res, S = representsa_(E, 'ellipsoid', 1e-9, 'return_set')
        assert res
        assert S is E
    
    def test_representsa_hyperplane(self):
        """Test comparison to hyperplane."""
        # 1D ellipsoid
        E = Ellipsoid(np.array([[1]]), np.array([[0]]))
        assert representsa_(E, 'hyperplane')
        
        # Multi-dimensional ellipsoid
        E = Ellipsoid(np.eye(2), np.array([[0], [0]]))
        assert not representsa_(E, 'hyperplane')
    
    def test_representsa_convex_set(self):
        """Test comparison to convex set (always true)."""
        E = Ellipsoid(np.eye(2), np.array([[0], [0]]))
        assert representsa_(E, 'convexSet')
    
    def test_representsa_unbounded_sets(self):
        """Test comparison to unbounded sets (always false)."""
        E = Ellipsoid(np.eye(2), np.array([[0], [0]]))
        
        # Halfspace
        assert not representsa_(E, 'halfspace')
        
        # Fullspace
        assert not representsa_(E, 'fullspace')
        
        # Prob zonotope
        assert not representsa_(E, 'probZonotope')
    
    def test_representsa_constrained_sets(self):
        """Test comparison to constrained sets."""
        # Point ellipsoid
        E_point = Ellipsoid(np.zeros((2, 2)), np.array([[1], [0]]))
        
        # 1D ellipsoid
        E_1d = Ellipsoid(np.array([[1]]), np.array([[0]]))
        
        # Multi-dimensional ellipsoid
        E_multi = Ellipsoid(np.eye(2), np.array([[0], [0]]))
        
        # Constrained hyperplane
        assert representsa_(E_point, 'conHyperplane')
        assert representsa_(E_1d, 'conHyperplane')
        assert not representsa_(E_multi, 'conHyperplane')
        
        # Constrained zonotope
        assert representsa_(E_point, 'conZonotope')
        assert representsa_(E_1d, 'conZonotope')
        assert not representsa_(E_multi, 'conZonotope')
        
        # Polytope
        assert representsa_(E_point, 'polytope')
        assert representsa_(E_1d, 'polytope')
        assert not representsa_(E_multi, 'polytope')
        
        # Zonotope bundle
        assert representsa_(E_point, 'zonoBundle')
        assert representsa_(E_1d, 'zonoBundle')
        assert not representsa_(E_multi, 'zonoBundle')
    
    def test_representsa_unsupported_types(self):
        """Test comparison to unsupported types."""
        E = Ellipsoid(np.eye(2), np.array([[0], [0]]))
        
        unsupported_types = [
            'conPolyZono', 'levelSet', 'polyZonotope', 'parallelotope'
        ]
        
        for type_str in unsupported_types:
            with pytest.raises(CORAError):
                representsa_(E, type_str)
    
    def test_representsa_conversion_not_supported(self):
        """Test cases where conversion is not supported."""
        E = Ellipsoid(np.eye(2), np.array([[0], [0]]))
        
        conversion_not_supported = [
            'capsule', 'conHyperplane', 'conZonotope', 'polytope', 'zonoBundle'
        ]
        
        for type_str in conversion_not_supported:
            if representsa_(E, type_str):
                with pytest.raises(CORAError):
                    representsa_(E, type_str, 1e-9, 'return_set')
    
    def test_representsa_unknown_type(self):
        """Test unknown set type."""
        E = Ellipsoid(np.eye(2), np.array([[0], [0]]))
        
        with pytest.raises(CORAError):
            representsa_(E, 'unknownType')
    
    def test_representsa_tolerance_effects(self):
        """Test effects of different tolerance values."""
        # Create ellipsoid close to origin
        E = Ellipsoid(1e-6 * np.eye(2), 1e-6 * np.ones((2, 1)))
        
        # Strict tolerance
        assert not representsa_(E, 'origin', 1e-9)
        
        # Loose tolerance
        assert representsa_(E, 'origin', 1e-3)
    
    def test_representsa_edge_cases(self):
        """Test edge cases."""
        # Very small ellipsoid
        E = Ellipsoid(1e-12 * np.eye(2), np.zeros((2, 1)))
        assert representsa_(E, 'point', 1e-9)
        
        # Large ellipsoid
        E = Ellipsoid(1e6 * np.eye(3), np.zeros((3, 1)))
        assert not representsa_(E, 'point')
        assert representsa_(E, 'convexSet')
    
    def test_representsa_return_set_functionality(self):
        """Test the return_set functionality."""
        # Point ellipsoid to point conversion
        center = np.array([[1], [2], [3]])
        E = Ellipsoid(np.zeros((3, 3)), center)
        
        res, S = representsa_(E, 'point', 1e-9, 'return_set')
        assert res
        assert np.allclose(S, center)
        
        # Origin conversion
        E = Ellipsoid(np.zeros((2, 2)), np.zeros((2, 1)))
        res, S = representsa_(E, 'origin', 1e-9, 'return_set')
        assert res
        assert np.allclose(S, np.zeros((2, 1)))
        
        # Ellipsoid to ellipsoid
        E = Ellipsoid(np.eye(2), np.array([[1], [0]]))
        res, S = representsa_(E, 'ellipsoid', 1e-9, 'return_set')
        assert res
        assert S is E 