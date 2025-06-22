"""
Test cases for ellipsoid contains_ method.
"""

import numpy as np
import pytest

from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.contains_ import contains_
from cora_python.contSet.ellipsoid.empty import empty
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAError


class TestEllipsoidContains:
    """Test class for ellipsoid contains_ method."""
    
    def test_contains_point_basic(self):
        """Test basic point containment."""
        # Create ellipsoid E = {x | (x-c)^T * Q^(-1) * (x-c) <= 1}
        Q = np.array([[4, 0], [0, 1]])  # Ellipse with semi-axes 2 and 1
        c = np.array([[1], [2]])  # Center at (1, 2)
        E = Ellipsoid(Q, c)
        
        # Point inside ellipsoid
        p_inside = np.array([[1.5], [2.5]])  # Should be inside
        assert contains_(E, p_inside)
        
        # Point outside ellipsoid
        p_outside = np.array([[4], [4]])  # Should be outside
        assert not contains_(E, p_outside)
        
        # Point on boundary (approximately)
        p_boundary = np.array([[3], [2]])  # On boundary
        assert contains_(E, p_boundary, tol=1e-6)
    
    def test_contains_point_degenerate(self):
        """Test point containment for degenerate ellipsoids."""
        # Point ellipsoid (zero shape matrix)
        Q = np.zeros((2, 2))
        c = np.array([[1], [2]])
        E = Ellipsoid(Q, c)
        
        # Same point should be contained
        assert contains_(E, c)
        
        # Different point should not be contained
        p_diff = np.array([[2], [3]])
        assert not contains_(E, p_diff)
    
    def test_contains_multiple_points(self):
        """Test containment of multiple points."""
        Q = np.eye(2)
        c = np.zeros((2, 1))
        E = Ellipsoid(Q, c)
        
        # Multiple points
        points = np.array([[0, 0.5, 1.5], [0, 0.5, 0]])
        res, cert, scaling = contains_(E, points, cert_toggle=True, scaling_toggle=True)
        
        # First two points should be inside, third outside
        expected = np.array([True, True, False])
        assert np.array_equal(res, expected)
        assert np.all(cert)
        assert len(scaling) == 3
    
    def test_contains_ellipsoid_basic(self):
        """Test ellipsoid containment."""
        # Larger ellipsoid
        Q1 = 4 * np.eye(2)
        c1 = np.zeros((2, 1))
        E1 = Ellipsoid(Q1, c1)
        
        # Smaller ellipsoid (should be contained)
        Q2 = np.eye(2)
        c2 = np.zeros((2, 1))
        E2 = Ellipsoid(Q2, c2)
        
        assert contains_(E1, E2)
        
        # Reverse should not be true
        assert not contains_(E2, E1)
    
    def test_contains_ellipsoid_point(self):
        """Test ellipsoid containing a point ellipsoid."""
        # Regular ellipsoid
        Q1 = np.eye(2)
        c1 = np.zeros((2, 1))
        E1 = Ellipsoid(Q1, c1)
        
        # Point ellipsoid inside
        Q2 = np.zeros((2, 2))
        c2 = np.array([[0.5], [0]])
        E2 = Ellipsoid(Q2, c2)
        
        assert contains_(E1, E2)
        
        # Point ellipsoid outside
        Q3 = np.zeros((2, 2))
        c3 = np.array([[2], [0]])
        E3 = Ellipsoid(Q3, c3)
        
        assert not contains_(E1, E3)
    
    def test_contains_empty_sets(self):
        """Test containment with empty sets."""
        # Regular ellipsoid
        Q = np.eye(2)
        c = np.zeros((2, 1))
        E = Ellipsoid(Q, c)
        
        # Empty ellipsoid
        E_empty = empty(2)
        
        # Regular ellipsoid should contain empty set
        assert contains_(E, E_empty)
        
        # Empty ellipsoid should not contain regular ellipsoid
        assert not contains_(E_empty, E)
        
        # Empty ellipsoid should contain empty set
        assert contains_(E_empty, E_empty)
    
    def test_contains_point_ellipsoids(self):
        """Test containment between point ellipsoids."""
        # Point ellipsoid 1
        Q1 = np.zeros((2, 2))
        c1 = np.array([[1], [2]])
        E1 = Ellipsoid(Q1, c1)
        
        # Same point ellipsoid
        Q2 = np.zeros((2, 2))
        c2 = np.array([[1], [2]])
        E2 = Ellipsoid(Q2, c2)
        
        assert contains_(E1, E2)
        
        # Different point ellipsoid
        Q3 = np.zeros((2, 2))
        c3 = np.array([[2], [3]])
        E3 = Ellipsoid(Q3, c3)
        
        assert not contains_(E1, E3)
    
    def test_contains_tolerance_effects(self):
        """Test effects of tolerance on containment."""
        Q = np.eye(2)
        c = np.zeros((2, 1))
        E = Ellipsoid(Q, c)
        
        # Point slightly outside boundary
        p = np.array([[1.001], [0]])
        
        # Strict tolerance should reject
        assert not contains_(E, p, tol=1e-6)
        
        # Loose tolerance should accept
        assert contains_(E, p, tol=1e-2)
    
    def test_contains_scaling_certificate(self):
        """Test scaling and certificate outputs."""
        Q = np.eye(2)
        c = np.zeros((2, 1))
        E = Ellipsoid(Q, c)
        
        # Point inside
        p_inside = np.array([[0.5], [0]])
        res, cert, scaling = contains_(E, p_inside, cert_toggle=True, scaling_toggle=True)
        
        assert res
        assert cert
        assert 0 <= scaling <= 1
        
        # Point outside
        p_outside = np.array([[2], [0]])
        res, cert, scaling = contains_(E, p_outside, cert_toggle=True, scaling_toggle=True)
        
        assert not res
        assert cert
        assert scaling > 1
    
    def test_contains_numeric_edge_cases(self):
        """Test edge cases with numeric arrays."""
        Q = np.eye(2)
        c = np.zeros((2, 1))
        E = Ellipsoid(Q, c)
        
        # Empty numeric array
        empty_array = np.array([]).reshape(2, 0)
        assert contains_(E, empty_array)
        
        # 1D point (should be reshaped)
        p_1d = np.array([0.5, 0])
        assert contains_(E, p_1d)
    
    def test_contains_method_parameter(self):
        """Test different method parameters."""
        Q = np.eye(2)
        c = np.zeros((2, 1))
        E = Ellipsoid(Q, c)
        
        p = np.array([[0.5], [0]])
        
        # Exact method
        assert contains_(E, p, method='exact')
        
        # Approx method (should work for points)
        assert contains_(E, p, method='approx')
    
    def test_contains_unsupported_method(self):
        """Test unsupported method parameter."""
        Q = np.eye(2)
        c = np.zeros((2, 1))
        E = Ellipsoid(Q, c)
        
        p = np.array([[0.5], [0]])
        
        with pytest.raises(CORAError):
            contains_(E, p, method='unknown')
    
    def test_contains_high_dimensional(self):
        """Test containment in higher dimensions."""
        # 3D ellipsoid
        Q = np.diag([1, 4, 9])
        c = np.array([[1], [2], [3]])
        E = Ellipsoid(Q, c)
        
        # Point inside
        p_inside = np.array([[1.5], [2.5], [3.5]])
        assert contains_(E, p_inside)
        
        # Point outside
        p_outside = np.array([[5], [5], [5]])
        assert not contains_(E, p_outside)
    
    def test_contains_singular_ellipsoid(self):
        """Test containment with singular ellipsoids."""
        # Singular ellipsoid (rank deficient)
        Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        c = np.zeros((3, 1))
        E = Ellipsoid(Q, c)
        
        # Point in the same subspace
        p_in_subspace = np.array([[0.5], [0.5], [0]])
        assert contains_(E, p_in_subspace)
        
        # Point outside the subspace
        p_outside_subspace = np.array([[0], [0], [1]])
        assert not contains_(E, p_outside_subspace)
    
    def test_contains_return_formats(self):
        """Test different return formats based on toggle parameters."""
        Q = np.eye(2)
        c = np.zeros((2, 1))
        E = Ellipsoid(Q, c)
        
        p = np.array([[0.5], [0]])
        
        # Only result
        res = contains_(E, p, cert_toggle=False, scaling_toggle=False)
        assert isinstance(res, (bool, np.bool_))
        
        # Result with certificate and scaling
        res, cert, scaling = contains_(E, p, cert_toggle=True, scaling_toggle=True)
        assert isinstance(res, (bool, np.bool_))
        assert isinstance(cert, (bool, np.bool_))
        assert isinstance(scaling, (float, np.floating))
    
    def test_contains_boundary_cases(self):
        """Test containment of points exactly on the boundary."""
        Q = np.eye(2)
        c = np.zeros((2, 1))
        E = Ellipsoid(Q, c)
        
        # Points on unit circle boundary
        boundary_points = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
        
        for i in range(boundary_points.shape[1]):
            p = boundary_points[:, i:i+1]
            # Should be contained with appropriate tolerance
            assert contains_(E, p, tol=1e-10)
    
    def test_contains_error_handling(self):
        """Test error handling for invalid inputs."""
        Q = np.eye(2)
        c = np.zeros((2, 1))
        E = Ellipsoid(Q, c)
        
        # Test with mock object that has no supported methods
        class MockSet:
            pass
        
        mock_set = MockSet()
        
        # Should raise error for unsupported set type with exact method
        with pytest.raises(CORAError):
            contains_(E, mock_set, method='exact') 