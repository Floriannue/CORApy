"""
test_plus - unit test function of plus

This module tests the ellipsoid plus operation implementation.

Authors:       Victor Gassmann (MATLAB), Python translation by AI Assistant
Written:       27-July-2021 (MATLAB)
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contSet.ellipsoid import Ellipsoid


def test_plus_vector():
    """Test Minkowski sum with vector"""
    # Init ellipsoid
    E1 = Ellipsoid(np.array([[5.4387811500952807, 12.4977183618314545], 
                            [12.4977183618314545, 29.6662117284481646]]), 
                   np.array([[-0.7445068341257537], [3.5800647524843665]]))
    
    # Test addition with vector
    v = np.array([[2.0], [-1.0]])
    E_result = E1.plus(v)
    
    # Check that shape matrix is unchanged
    assert np.allclose(E_result.Q, E1.Q)
    # Check that center is shifted by the vector
    assert np.allclose(E_result.q, E1.q + v)


def test_plus_empty():
    """Test Minkowski sum with empty set"""
    E1 = Ellipsoid(np.array([[5.4387811500952807, 12.4977183618314545], 
                            [12.4977183618314545, 29.6662117284481646]]), 
                   np.array([[-0.7445068341257537], [3.5800647524843665]]))
    
    # Empty set addition
    E_empty = Ellipsoid.empty(2)
    E_result = E1.plus(E_empty)
    
    # Result should be empty
    assert E_result.representsa_('emptySet', E1.TOL)


def test_plus_origin():
    """Test Minkowski sum with origin"""
    E1 = Ellipsoid(np.array([[5.4387811500952807, 12.4977183618314545], 
                            [12.4977183618314545, 29.6662117284481646]]), 
                   np.array([[-0.7445068341257537], [3.5800647524843665]]))
    
    # Origin (zero vector)
    origin = np.array([[0.0], [0.0]])
    E_result = E1.plus(origin)
    
    # Should be equal to original ellipsoid
    assert np.allclose(E_result.Q, E1.Q)
    assert np.allclose(E_result.q, E1.q)


def test_plus_ellipsoid_ellipsoid():
    """Test Minkowski sum between two ellipsoids returns an ellipsoid"""
    E1 = Ellipsoid(np.array([[5.4387811500952807, 12.4977183618314545], 
                            [12.4977183618314545, 29.6662117284481646]]), 
                   np.array([[-0.7445068341257537], [3.5800647524843665]]))
    
    E2 = Ellipsoid(np.array([[4.2533342807136076, 0.6346400221575308], 
                            [0.6346400221575309, 0.0946946398147988]]), 
                   np.array([[-2.4653656883489115], [0.2717868749873985]]))
    
    E_out = E1.plus(E2)
    assert isinstance(E_out, Ellipsoid)
    assert E_out.Q.shape == E1.Q.shape


def test_plus_operator_overload():
    """Test that the + operator is overloaded correctly"""
    E1 = Ellipsoid(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[0.0], [0.0]]))
    v = np.array([[1.0], [2.0]])
    
    # Test using + operator
    E_result = E1 + v
    
    # Should be same as using plus method
    E_result_method = E1.plus(v)
    
    assert np.allclose(E_result.Q, E_result_method.Q)
    assert np.allclose(E_result.q, E_result_method.q) 


def test_plus_outer_directions():
    Q1 = np.array([[2.0, 0.2], [0.2, 1.0]])
    q1 = np.array([[0.0], [0.0]])
    E1 = Ellipsoid(Q1, q1)

    Q2 = np.array([[1.5, -0.1], [-0.1, 0.8]])
    q2 = np.array([[0.1], [0.1]])
    E2 = Ellipsoid(Q2, q2)

    # one direction along x-axis and one along y-axis
    L = np.array([[1.0, 0.0], [0.0, 1.0]])
    E_out = E1.plus(E2, 'outer', L)
    assert isinstance(E_out, Ellipsoid)
    assert E_out.Q.shape == (2, 2)


def test_plus_outer_halder():
    Q1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    q1 = np.array([[0.2], [-0.1]])
    E1 = Ellipsoid(Q1, q1)

    Q2 = np.array([[0.5, 0.0], [0.0, 0.8]])
    q2 = np.array([[0.0], [0.1]])
    E2 = Ellipsoid(Q2, q2)

    E_out = E1.plus(E2, 'outer:halder', np.zeros((2, 0)))
    assert isinstance(E_out, Ellipsoid)
    assert E_out.Q.shape == (2, 2)
    # Basic sanity on positive semidefiniteness
    evals = np.linalg.eigvalsh(E_out.Q)
    assert np.all(evals >= -1e-8)