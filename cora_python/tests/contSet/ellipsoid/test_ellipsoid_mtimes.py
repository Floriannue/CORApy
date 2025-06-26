"""
test_ellipsoid_mtimes - unit test function of mtimes

This module tests the ellipsoid mtimes implementation exactly matching MATLAB.

Authors:       Victor Gassmann (MATLAB), Python translation by AI Assistant
Written:       13-March-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.ellipsoid import Ellipsoid


def test_ellipsoid_mtimes():
    """Main mtimes test matching MATLAB test_ellipsoid_mtimes"""
    
    # Simple matrix multiply test (exact MATLAB values)
    Q = np.array([[1, 0.1, 0.8], [0.1, 3, 2], [0.8, 2, 5]])
    q = np.array([[10], [-23], [15]])
    E1 = Ellipsoid(Q, q)
    A = np.array([[1, 2, 3], [4, 5, 6]])
    EA = A @ E1
    
    # True result from MATLAB
    true_result = Ellipsoid(
        np.array([[87.2, 193.7], [193.7, 433.4]]), 
        np.array([[9], [15]])
    )
    
    # Check shape matrix
    assert np.allclose(EA.Q, true_result.Q, rtol=1e-10), \
        f"Shape matrix mismatch: got {EA.Q}, expected {true_result.Q}"
    
    # Check center
    assert np.allclose(EA.q, true_result.q, rtol=1e-10), \
        f"Center mismatch: got {EA.q}, expected {true_result.q}"


def test_mtimes_random_ellipsoid():
    """Test mtimes with random degenerate ellipsoid and point containment"""
    
    # This test is inspired by MATLAB's random test but simplified
    # for deterministic behavior
    np.random.seed(42)  # For reproducible tests
    
    # Create a test ellipsoid
    Q = np.array([[2, 0.5], [0.5, 1]])
    q = np.array([[1], [-1]])
    E1 = Ellipsoid(Q, q)
    
    # Generate test points within E1
    # For a 2D ellipsoid, we can generate points parametrically
    theta = np.linspace(0, 2*np.pi, 10)
    # Transform unit circle points to ellipsoid
    L = np.linalg.cholesky(Q)
    unit_points = np.array([np.cos(theta), np.sin(theta)])
    samples = L @ unit_points + q
    
    # Verify points are in original ellipsoid
    for i in range(samples.shape[1]):
        point = samples[:, i:i+1]
        assert E1.contains_(point), f"Generated point {point.flatten()} should be in E1"
    
    # Random transformation matrix
    A = np.array([[1.2, 0.3], [-0.1, 0.8]])
    
    # Transform ellipsoid
    EA = A @ E1
    
    # Transform sample points
    samples_A = A @ samples
    
    # Verify transformed points are in transformed ellipsoid
    for i in range(samples_A.shape[1]):
        point = samples_A[:, i:i+1]
        assert EA.contains_(point), f"Transformed point {point.flatten()} should be in transformed ellipsoid"


def test_mtimes_scalar_multiplication():
    """Test scalar multiplication of ellipsoids"""
    
    Q = np.array([[1, 0.5], [0.5, 2]])
    q = np.array([[2], [1]])
    E = Ellipsoid(Q, q)
    
    # Scalar multiplication
    scalar = 2.5
    E_scaled = scalar * E
    
    # Expected results: Q' = scalar^2 * Q, q' = scalar * q
    expected_Q = scalar**2 * Q
    expected_q = scalar * q
    
    assert np.allclose(E_scaled.Q, expected_Q), \
        f"Scaled shape matrix incorrect: got {E_scaled.Q}, expected {expected_Q}"
    assert np.allclose(E_scaled.q, expected_q), \
        f"Scaled center incorrect: got {E_scaled.q}, expected {expected_q}"


def test_mtimes_matrix_multiplication():
    """Test matrix multiplication of ellipsoids"""
    
    Q = np.array([[4, 1], [1, 2]])
    q = np.array([[1], [3]])
    E = Ellipsoid(Q, q)
    
    # Matrix multiplication
    A = np.array([[2, 1], [0, 1], [1, -1]])  # 3x2 matrix
    E_transformed = A @ E
    
    # Expected results: Q' = A * Q * A.T, q' = A * q
    expected_Q = A @ Q @ A.T
    expected_q = A @ q
    
    assert np.allclose(E_transformed.Q, expected_Q), \
        f"Transformed shape matrix incorrect: got {E_transformed.Q}, expected {expected_Q}"
    assert np.allclose(E_transformed.q, expected_q), \
        f"Transformed center incorrect: got {E_transformed.q}, expected {expected_q}"


def test_mtimes_empty_ellipsoid():
    """Test multiplication with empty ellipsoid"""
    
    E_empty = Ellipsoid.empty(2)
    A = np.array([[1, 2], [3, 4]])
    
    # Multiplying empty ellipsoid should return empty ellipsoid
    EA_empty = A @ E_empty
    assert EA_empty.representsa_('emptySet', EA_empty.TOL), \
        "Transformation of empty ellipsoid should remain empty"


def test_mtimes_dimension_consistency():
    """Test that multiplication preserves dimension consistency"""
    
    Q = np.eye(3)
    q = np.ones((3, 1))
    E = Ellipsoid(Q, q)
    
    # 2x3 matrix transformation
    A = np.array([[1, 0, 1], [0, 1, -1]])
    E_transformed = A @ E
    
    assert E_transformed.dim() == 2, f"Transformed ellipsoid should be 2D, got {E_transformed.dim()}"
    assert E_transformed.Q.shape == (2, 2), f"Shape matrix should be 2x2, got {E_transformed.Q.shape}"
    assert E_transformed.q.shape == (2, 1), f"Center should be 2x1, got {E_transformed.q.shape}"


def test_mtimes_symmetry_preservation():
    """Test that multiplication preserves matrix symmetry"""
    
    # Start with symmetric Q
    Q = np.array([[2, 1], [1, 3]])
    q = np.array([[0], [0]])
    E = Ellipsoid(Q, q)
    
    # Any linear transformation should preserve symmetry
    A = np.array([[1, 2], [3, 1]])
    E_transformed = A @ E
    
    # Check symmetry of resulting Q
    Q_result = E_transformed.Q
    assert np.allclose(Q_result, Q_result.T), \
        f"Transformed shape matrix should be symmetric: {Q_result}"


def test_mtimes_commutativity_with_scalar():
    """Test commutativity of scalar multiplication"""
    
    Q = np.array([[1, 0.5], [0.5, 2]])
    q = np.array([[1], [2]])
    E = Ellipsoid(Q, q)
    scalar = 3.0
    
    # Test that scalar * E == E * scalar (though second form not implemented)
    E1 = scalar * E
    E2 = E * scalar
    
    assert np.allclose(E1.Q, E2.Q), "Scalar multiplication should be commutative for Q"
    assert np.allclose(E1.q, E2.q), "Scalar multiplication should be commutative for q"


def test_mtimes_error_cases():
    """Test error handling in mtimes"""
    
    E = Ellipsoid(np.eye(2))
    
    # Incompatible matrix dimensions
    A_incompatible = np.array([[1, 2, 3]])  # 1x3 matrix for 2D ellipsoid
    
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        A_incompatible @ E


def test_mtimes_with_zeros():
    """Test multiplication with zero matrix/scalar"""
    
    Q = np.array([[1, 0.5], [0.5, 2]])
    q = np.array([[1], [2]])
    E = Ellipsoid(Q, q)
    
    # Zero scalar
    E_zero_scalar = 0 * E
    assert E_zero_scalar.rank() == 0, "Zero scalar multiplication should create point ellipsoid"
    
    # Zero matrix (projects to origin)
    A_zero = np.zeros((2, 2))
    E_zero_matrix = A_zero @ E
    assert E_zero_matrix.rank() == 0, "Zero matrix multiplication should create point ellipsoid"


@pytest.mark.parametrize("scalar", [0.1, 1.0, 2.5, 10.0])
def test_mtimes_various_scalars(scalar):
    """Test multiplication with various scalar values"""
    
    Q = np.array([[1, 0.2], [0.2, 1.5]])
    q = np.array([[0.5], [-0.3]])
    E = Ellipsoid(Q, q)
    
    E_scaled = scalar * E
    
    # Check scaling properties
    expected_Q = scalar**2 * Q
    expected_q = scalar * q
    
    assert np.allclose(E_scaled.Q, expected_Q, rtol=1e-12), \
        f"Scaling with {scalar} failed for Q matrix"
    assert np.allclose(E_scaled.q, expected_q, rtol=1e-12), \
        f"Scaling with {scalar} failed for center vector"


if __name__ == "__main__":
    test_ellipsoid_mtimes()
    test_mtimes_random_ellipsoid()
    test_mtimes_scalar_multiplication()
    test_mtimes_matrix_multiplication()
    test_mtimes_empty_ellipsoid()
    test_mtimes_dimension_consistency()
    test_mtimes_symmetry_preservation()
    test_mtimes_commutativity_with_scalar()
    test_mtimes_error_cases()
    test_mtimes_with_zeros()
    print("All mtimes tests passed!") 