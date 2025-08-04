"""
Test cases for aux_tightenHalfspaces function
"""

import numpy as np
import pytest
from cora_python.g.functions.helper.sets.contSet.zonotope.aux_tightenHalfspaces import aux_tightenHalfspaces


def test_aux_tightenHalfspaces_basic():
    """Test basic functionality with simple constraints"""
    # Simple 2D polytope: 0 <= x <= 1, 0 <= y <= 1
    C = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    delta_d = np.array([[1], [1], [1], [1]])
    
    result = aux_tightenHalfspaces(C, delta_d)
    
    assert result is not None
    assert result.shape == (4, 1)
    # The tightened constraints should still define the same polytope
    assert np.allclose(result, delta_d, atol=1e-10)


def test_aux_tightenHalfspaces_empty_polytope():
    """Test with infeasible constraints (empty polytope)"""
    # Contradictory constraints: x <= 1 and x >= 2
    C = np.array([[1], [-1]])
    delta_d = np.array([[1], [-2]])  # x <= 1 and x >= 2 (infeasible)
    
    result = aux_tightenHalfspaces(C, delta_d)
    
    assert result is None


def test_aux_tightenHalfspaces_tightened():
    """Test with constraints that can be tightened"""
    # Triangle: x >= 0, y >= 0, x + y <= 1
    C = np.array([[1, 0], [0, 1], [-1, -1]])
    delta_d = np.array([[0], [0], [1]])
    
    result = aux_tightenHalfspaces(C, delta_d)
    
    assert result is not None
    assert result.shape == (3, 1)
    # The result should be the same as input for this case
    assert np.allclose(result, delta_d, atol=1e-10)


def test_aux_tightenHalfspaces_3d():
    """Test with 3D constraints"""
    # 3D cube: 0 <= x,y,z <= 1
    C = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    delta_d = np.array([[1], [1], [1], [1], [1], [1]])
    
    result = aux_tightenHalfspaces(C, delta_d)
    
    assert result is not None
    assert result.shape == (6, 1)
    assert np.allclose(result, delta_d, atol=1e-10)


def test_aux_tightenHalfspaces_single_constraint():
    """Test with single constraint"""
    C = np.array([[1, 0]])
    delta_d = np.array([[1]])
    
    result = aux_tightenHalfspaces(C, delta_d)
    
    assert result is not None
    assert result.shape == (1, 1)
    assert np.allclose(result, delta_d, atol=1e-10)


def test_aux_tightenHalfspaces_input_validation():
    """Test input validation"""
    # Test with non-2D array
    with pytest.raises(ValueError):
        aux_tightenHalfspaces(np.array([1, 2, 3]), np.array([[1]]))
    
    # Test with mismatched dimensions
    with pytest.raises(ValueError):
        aux_tightenHalfspaces(np.array([[1, 0], [0, 1]]), np.array([[1], [1], [1]])) 