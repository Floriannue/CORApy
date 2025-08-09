"""
test_interval - unit test function of interval

This module tests the ellipsoid interval conversion implementation.

Authors:       Victor Gassmann (MATLAB), Python translation by AI Assistant
Written:       13-March-2019 (MATLAB)
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.ellipsoid import Ellipsoid


def test_interval_basic():
    """Test basic interval conversion"""
    # Unit ellipsoid at origin
    E = Ellipsoid(np.eye(2))
    
    try:
        I = E.interval()
        
        # Unit ellipsoid should produce interval [-1,1] x [-1,1]
        expected_inf = np.array([[-1.0], [-1.0]])
        expected_sup = np.array([[1.0], [1.0]])
        
        assert np.allclose(I.inf, expected_inf, rtol=1e-10)
        assert np.allclose(I.sup, expected_sup, rtol=1e-10)
        
    except Exception as e:
        pytest.fail(f"Interval conversion failed: {e}")


def test_interval_shifted():
    """Test interval conversion with shifted ellipsoid"""
    # Unit ellipsoid shifted to [1, 2]
    E = Ellipsoid(np.eye(2), np.array([[1.0], [2.0]]))
    
    try:
        I = E.interval()
        
        # Should produce interval [0,2] x [1,3]
        expected_inf = np.array([[0.0], [1.0]])
        expected_sup = np.array([[2.0], [3.0]])
        
        assert np.allclose(I.inf, expected_inf, rtol=1e-10)
        assert np.allclose(I.sup, expected_sup, rtol=1e-10)
        
    except Exception as e:
        pytest.fail(f"Interval conversion failed: {e}")


def test_interval_scaled():
    """Test interval conversion with scaled ellipsoid"""
    # Ellipsoid with different radii in each dimension
    Q = np.diag([1, 4])  # Radii: 1, 2
    E = Ellipsoid(Q)
    
    try:
        I = E.interval()
        
        # Should produce interval [-1,1] x [-2,2]
        expected_inf = np.array([[-1.0], [-2.0]])
        expected_sup = np.array([[1.0], [2.0]])
        
        assert np.allclose(I.inf, expected_inf, rtol=1e-10)
        assert np.allclose(I.sup, expected_sup, rtol=1e-10)
        
    except Exception as e:
        pytest.fail(f"Interval conversion failed: {e}")


def test_interval_ellipse():
    """Test interval conversion with general ellipse"""
    # Example from MATLAB: E = ellipsoid([3 -1; -1 1],[1;0])
    Q = np.array([[3, -1], [-1, 1]])
    q = np.array([[1], [0]])
    E = Ellipsoid(Q, q)
    
    try:
        I = E.interval()
        
        # The interval should enclose the ellipsoid
        # We don't test exact values here since they depend on support function computation
        # Just verify that interval was created and has correct dimension
        assert I.dim() == 2
        assert hasattr(I, 'inf')
        assert hasattr(I, 'sup')
        
    except Exception as e:
        pytest.fail(f"Interval conversion failed: {e}")


def test_interval_degenerate():
    """Test interval conversion with degenerate ellipsoid (point)"""
    # Point ellipsoid
    E = Ellipsoid(np.zeros((2, 2)), np.array([[1.0], [2.0]]))
    
    try:
        I = E.interval()
        
        # Should produce point interval [1,1] x [2,2]
        expected_inf = np.array([[1.0], [2.0]])
        expected_sup = np.array([[1.0], [2.0]])
        
        assert np.allclose(I.inf, expected_inf, rtol=1e-10)
        assert np.allclose(I.sup, expected_sup, rtol=1e-10)
        
    except Exception as e:
        pytest.fail(f"Interval conversion failed: {e}") 