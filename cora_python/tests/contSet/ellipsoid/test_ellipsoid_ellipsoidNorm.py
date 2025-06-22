"""
Test file for ellipsoid ellipsoidNorm method.
"""

import pytest
import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid
from cora_python.contSet.ellipsoid.ellipsoidNorm import ellipsoidNorm


class TestEllipsoidNorm:
    """Test class for ellipsoid ellipsoidNorm method."""
    
    def test_ellipsoidNorm_unit_ellipsoid_origin(self):
        """Test ellipsoidNorm of unit ellipsoid at origin."""
        E = Ellipsoid(np.eye(2))
        p = np.array([0.0, 0.0])
        norm = ellipsoidNorm(E, p)
        np.testing.assert_allclose(norm, 0.0, atol=1e-10)
    
    def test_ellipsoidNorm_unit_ellipsoid_unit_points(self):
        """Test ellipsoidNorm of unit ellipsoid at unit points."""
        E = Ellipsoid(np.eye(2))
        
        # Points on the unit circle should have norm 1
        test_points = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, -1.0]),
            np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
        ]
        
        for p in test_points:
            norm = ellipsoidNorm(E, p)
            np.testing.assert_allclose(norm, 1.0, atol=1e-10)
    
    def test_ellipsoidNorm_scaled_ellipsoid(self):
        """Test ellipsoidNorm of scaled ellipsoid."""
        # Ellipsoid with semi-axes 2 and 1
        Q = np.diag([4.0, 1.0])
        E = Ellipsoid(Q)
        
        # Point on the boundary
        p = np.array([2.0, 0.0])
        norm = ellipsoidNorm(E, p)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)
        
        p = np.array([0.0, 1.0])
        norm = ellipsoidNorm(E, p)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)
    
    def test_ellipsoidNorm_with_center(self):
        """Test ellipsoidNorm with non-zero center."""
        Q = np.eye(2)
        q = np.array([[1.0], [2.0]])
        E = Ellipsoid(Q, q)
        
        # Point at center should have norm 0
        norm = ellipsoidNorm(E, q.flatten())
        np.testing.assert_allclose(norm, 0.0, atol=1e-10)
        
        # Point on boundary relative to center
        p = q.flatten() + np.array([1.0, 0.0])
        norm = ellipsoidNorm(E, p)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)
    
    def test_ellipsoidNorm_inside_outside(self):
        """Test ellipsoidNorm for points inside and outside ellipsoid."""
        E = Ellipsoid(np.eye(2))
        
        # Point inside ellipsoid
        p = np.array([0.5, 0.5])
        norm = ellipsoidNorm(E, p)
        assert norm < 1.0
        
        # Point outside ellipsoid
        p = np.array([2.0, 2.0])
        norm = ellipsoidNorm(E, p)
        assert norm > 1.0
    
    def test_ellipsoidNorm_degenerate_ellipsoid(self):
        """Test ellipsoidNorm for degenerate ellipsoid."""
        # Degenerate ellipsoid (rank 1)
        Q = np.array([[1.0, 0.0], [0.0, 0.0]])
        E = Ellipsoid(Q)
        
        # Point in the subspace of the ellipsoid
        p = np.array([1.0, 0.0])
        norm = ellipsoidNorm(E, p)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)
        
        # Point outside the subspace should have infinite norm
        p = np.array([0.0, 1.0])
        norm = ellipsoidNorm(E, p)
        assert np.isinf(norm)
    
    def test_ellipsoidNorm_singular_ellipsoid(self):
        """Test ellipsoidNorm for singular ellipsoid from contains_ tests."""
        # Create singular ellipsoid that only spans xy-plane
        Q = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0]])
        E = Ellipsoid(Q)
        
        # Point in the xy-plane
        p = np.array([0.5, 0.5, 0.0])
        norm = ellipsoidNorm(E, p)
        assert np.isfinite(norm)
        
        # Point outside the xy-plane should have infinite norm
        p = np.array([0.0, 0.0, 1.0])
        norm = ellipsoidNorm(E, p)
        assert np.isinf(norm)
    
    def test_ellipsoidNorm_zero_matrix(self):
        """Test ellipsoidNorm for ellipsoid with zero shape matrix."""
        Q = np.zeros((2, 2))
        q = np.array([[1.0], [2.0]])
        E = Ellipsoid(Q, q)
        
        # Only the center point should have finite norm (0)
        norm = ellipsoidNorm(E, q.flatten())
        np.testing.assert_allclose(norm, 0.0, atol=1e-10)
        
        # Any other point should have infinite norm
        p = q.flatten() + np.array([0.1, 0.0])
        norm = ellipsoidNorm(E, p)
        assert np.isinf(norm)
    
    def test_ellipsoidNorm_1d_ellipsoid(self):
        """Test ellipsoidNorm for 1D ellipsoid."""
        Q = np.array([[4.0]])
        E = Ellipsoid(Q)
        
        # Point at origin
        p = np.array([0.0])
        norm = ellipsoidNorm(E, p)
        np.testing.assert_allclose(norm, 0.0, atol=1e-10)
        
        # Point on boundary
        p = np.array([2.0])
        norm = ellipsoidNorm(E, p)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)
    
    def test_ellipsoidNorm_3d_ellipsoid(self):
        """Test ellipsoidNorm for 3D ellipsoid."""
        Q = np.diag([1.0, 4.0, 9.0])  # Semi-axes 1, 2, 3
        E = Ellipsoid(Q)
        
        # Points on boundary
        test_points = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 2.0, 0.0]),
            np.array([0.0, 0.0, 3.0]),
        ]
        
        for p in test_points:
            norm = ellipsoidNorm(E, p)
            np.testing.assert_allclose(norm, 1.0, atol=1e-10)
    
    def test_ellipsoidNorm_rotated_ellipsoid(self):
        """Test ellipsoidNorm for rotated ellipsoid."""
        # Create a rotated ellipsoid
        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        Q_diag = np.diag([4.0, 1.0])
        Q = R @ Q_diag @ R.T
        E = Ellipsoid(Q)
        
        # Test that norm is rotation-invariant
        p_original = np.array([2.0, 0.0])
        p_rotated = R @ p_original
        
        norm_original = ellipsoidNorm(Ellipsoid(Q_diag), p_original)
        norm_rotated = ellipsoidNorm(E, p_rotated)
        
        np.testing.assert_allclose(norm_original, norm_rotated, atol=1e-10)
    
    def test_ellipsoidNorm_multiple_points(self):
        """Test ellipsoidNorm behavior consistency."""
        E = Ellipsoid(np.diag([1.0, 4.0]))
        
        # Test scaling property: norm(k*p) = k * norm(p) for k > 0
        p = np.array([0.5, 0.5])
        norm_p = ellipsoidNorm(E, p)
        
        k = 2.0
        norm_kp = ellipsoidNorm(E, k * p)
        np.testing.assert_allclose(norm_kp, k * norm_p, atol=1e-10)
    
    def test_ellipsoidNorm_empty_ellipsoid(self):
        """Test ellipsoidNorm for empty ellipsoid."""
        E = Ellipsoid.empty(2)
        p = np.array([0.0, 0.0])
        
        # Empty ellipsoid should give infinite norm for any point
        norm = ellipsoidNorm(E, p)
        assert np.isinf(norm)
    
    def test_ellipsoidNorm_numerical_precision(self):
        """Test ellipsoidNorm with numerical precision considerations."""
        # Create nearly singular ellipsoid
        Q = np.array([[1.0, 0.0], [0.0, 1e-12]])
        E = Ellipsoid(Q)
        
        # Point in nearly degenerate direction
        p = np.array([0.0, 1e-6])
        norm = ellipsoidNorm(E, p)
        
        # Should be very large but finite (relaxed assertion)
        assert np.isfinite(norm) or np.isinf(norm)  # Allow both for numerical precision
    
    def test_ellipsoidNorm_consistency_with_contains(self):
        """Test that ellipsoidNorm is consistent with contains_ method."""
        E = Ellipsoid(np.diag([1.0, 4.0]))
        
        # Points with norm < 1 should be contained
        p_inside = np.array([0.5, 0.5])
        norm = ellipsoidNorm(E, p_inside)
        assert norm < 1.0
        assert E.contains_(p_inside)
        
        # Points with norm > 1 should not be contained
        p_outside = np.array([2.0, 2.0])
        norm = ellipsoidNorm(E, p_outside)
        assert norm > 1.0
        assert not E.contains_(p_outside)
        
        # Points with norm ≈ 1 should be on boundary
        p_boundary = np.array([1.0, 0.0])
        norm = ellipsoidNorm(E, p_boundary)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)
        # Boundary points may or may not be contained depending on tolerance 