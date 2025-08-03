"""
test_zonotope_lift_ - unit test function of lift_

This module contains unit tests for the zonotope lift_ method, which lifts a zonotope
onto a higher-dimensional space.

Authors: Tobias Ladner (MATLAB)
         Python translation by AI Assistant
Written: 19-September-2023 (MATLAB)
               2025 (Tiange Yang, Florian Nüssel, Python translation by AI Assistant)
"""

import pytest
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror


def test_zonotope_lift_basic():
    """
    Test basic lift functionality when no new dimensions are created
    """
    # Initialize zonotope
    Z = Zonotope(np.array([[1], [1]]), np.array([[3, 0], [0, 2]]))
    
    # Test lift to same dimension (should work like project)
    Z_lift = Z.lift_(2, [1, 0])  # Note: Python uses 0-indexing, MATLAB uses 1-indexing
    Z_proj = Z.project([1, 0])
    
    # Check that lift and project give same result
    assert np.allclose(Z_lift.c, Z_proj.c)
    assert np.allclose(Z_lift.G, Z_proj.G)


def test_zonotope_lift_should_fail():
    """
    Test that lift fails when trying to create new dimensions
    """
    # Initialize zonotope
    Z = Zonotope(np.array([[1], [1]]), np.array([[3, 0], [0, 2]]))
    
    # Should always fail when trying to lift to higher dimension
    with pytest.raises(CORAerror) as exc_info:
        Z_lift = Z.lift_(5, [1, 0])  # Try to lift 2D zonotope to 5D
    
    # Check error identifier
    assert exc_info.value.identifier == 'CORA:notDefined'


def test_zonotope_lift_different_projection_order():
    """
    Test lift with different projection order
    """
    # Initialize zonotope
    Z = Zonotope(np.array([[1], [1]]), np.array([[3, 0], [0, 2]]))
    
    # Test lift with reversed projection order
    Z_lift = Z.lift_(2, [0, 1])  # Reverse order
    Z_proj = Z.project([0, 1])
    
    # Check that lift and project give same result
    assert np.allclose(Z_lift.c, Z_proj.c)
    assert np.allclose(Z_lift.G, Z_proj.G)



if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__]) 