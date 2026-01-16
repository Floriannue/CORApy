"""
Test file for affineSolution function

This module contains unit tests for the computation of the affine solution
for linear systems.

Authors: Mark Wetzlinger (MATLAB)
         Python translation by AI Assistant
Written: 16-October-2024 (MATLAB)
Python translation: 2025
"""

import numpy as np
import pytest
from cora_python.contDynamics.linearSys import LinearSys, affineSolution
from cora_python.contSet.zonotope import Zonotope
from cora_python.g.functions.matlab.validate.check import compareMatrices
from cora_python.contSet.contSet import isequal, contains
import scipy.linalg


def test_linearSys_affineSolution_basic():
    """Test basic functionality of affineSolution"""
    
    # Tolerance
    tol = 1e-14
    
    # Init system, state, input, and algorithm parameters
    A = np.array([[-1, -4], [4, -1]], dtype=float)
    sys = LinearSys(A)
    
    # Create test zonotope
    center = np.array([[40], [20]], dtype=float)
    generators = np.array([[1, 4, 2], [-1, 3, 5]], dtype=float)
    X = Zonotope(center, generators)
    
    u = np.array([[2], [-1]], dtype=float)
    timeStep = 0.05
    truncationOrder = 6
    
    # Compute reachable sets of first step
    Htp, Pu, Hti, C_state, C_input = affineSolution(sys, X, u, timeStep, truncationOrder)
    
    # Compare particular solution to analytical solution
    Pu_true = np.linalg.inv(A) @ (scipy.linalg.expm(A * timeStep) - np.eye(2)) @ u
    
    # Note: For the test, we need to handle the decomposed case
    if isinstance(Pu, list):
        # Extract center from first block
        Pu_center = Pu[0].center() if hasattr(Pu[0], 'center') else Pu[0]
    else:
        Pu_center = Pu.center() if hasattr(Pu, 'center') else Pu
    
    assert compareMatrices(Pu_center, Pu_true, tol)
    
    # Time-interval affine solution must contain time-point affine solution
    assert contains(Hti, Htp, tol=tol)
    
    # The affine time-point solution is e^At*X + Pu
    eAt_X = scipy.linalg.expm(A * timeStep) @ X
    expected_Htp = eAt_X + Pu
    assert isequal(Htp, expected_Htp, tol)
    
    # The affine time-interval solution is enclose(X,Htp) + error terms
    enclosed = X.enclose(Htp)
    expected_Hti = enclosed + C_state + C_input
    assert isequal(Hti, expected_Hti, tol)


def test_linearSys_affineSolution_with_blocks():
    """Test affineSolution with block decomposition"""
    
    # Init system with higher dimensions for block testing
    A = np.array([[-0.5, -1, 0, 0], 
                  [1, -0.5, 0, 0],
                  [0, 0, -0.3, -2],
                  [0, 0, 2, -0.3]], dtype=float)
    sys = LinearSys(A)
    
    # Create test zonotope
    center = np.array([[1], [2], [3], [4]], dtype=float)
    generators = np.array([[0.1, 0.2], [0.1, -0.1], [0.2, 0.1], [0.1, 0.2]], dtype=float)
    X = Zonotope(center, generators)
    
    u = np.array([[0.5], [0.3], [0.7], [0.2]], dtype=float)
    timeStep = 0.1
    truncationOrder = 4
    
    # Define blocks for decomposition
    blocks = np.array([[1, 2], [3, 4]])
    
    # Compute with blocks
    Htp, Pu, Hti, C_state, C_input = affineSolution(sys, X, u, timeStep, truncationOrder, blocks)
    
    # Results should be lists due to decomposition
    assert isinstance(Htp, list) or hasattr(Htp, '__len__')
    assert isinstance(Pu, list) or hasattr(Pu, '__len__')
    assert isinstance(Hti, list) or hasattr(Hti, '__len__')


def test_linearSys_affineSolution_homogeneous():
    """Test affineSolution for homogeneous case (zero input)"""
    
    # Simple 2D system
    A = np.array([[0, 1], [-1, 0]], dtype=float)
    sys = LinearSys(A)
    
    center = np.array([[1], [0]], dtype=float)
    generators = np.array([[0.1, 0], [0, 0.1]], dtype=float)
    X = Zonotope(center, generators)
    
    # Zero input
    u = np.array([[0], [0]], dtype=float)
    timeStep = 0.1
    truncationOrder = 10
    
    Htp, Pu, Hti, C_state, C_input = affineSolution(sys, X, u, timeStep, truncationOrder)
    
    # For zero input, particular solution should be (approximately) zero
    if isinstance(Pu, list):
        Pu_center = Pu[0].center() if hasattr(Pu[0], 'center') else Pu[0]
    else:
        Pu_center = Pu.center() if hasattr(Pu, 'center') else Pu
    
    assert np.allclose(Pu_center, np.zeros((2, 1)), atol=1e-10)


def test_linearSys_affineSolution_return_values():
    """Test different numbers of return values"""
    
    A = np.array([[-1, 0], [0, -2]], dtype=float)
    sys = LinearSys(A)
    
    center = np.array([[2], [1]], dtype=float)
    generators = np.array([[0.5, 0], [0, 0.3]], dtype=float)
    X = Zonotope(center, generators)
    
    u = np.array([[1], [0.5]], dtype=float)
    timeStep = 0.05
    truncationOrder = 5
    
    # Test full return
    result = affineSolution(sys, X, u, timeStep, truncationOrder)
    assert len(result) == 5  # Htp, Pu, Hti, C_state, C_input
    
    Htp, Pu, Hti, C_state, C_input = result
    assert Htp is not None
    assert Pu is not None
    assert Hti is not None
    assert C_state is not None
    assert C_input is not None


if __name__ == "__main__":
    test_linearSys_affineSolution_basic()
    test_linearSys_affineSolution_with_blocks()
    test_linearSys_affineSolution_homogeneous()
    test_linearSys_affineSolution_return_values()
    print("All tests passed!") 