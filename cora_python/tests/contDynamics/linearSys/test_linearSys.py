"""
test_linearSys - unit test for LinearSys constructor

This test verifies that the LinearSys constructor correctly sets all properties
for various input argument combinations.

Authors: Florian Nüssel (Python implementation)
Date: 2025-06-08
"""

import pytest
import numpy as np
from cora_python.contDynamics import LinearSys


def test_linearSys_constructor():
    """Test LinearSys constructor with various input combinations"""
    
    # Empty case
    sys = LinearSys()
    assert sys.nr_of_dims == 0
    assert sys.nr_of_inputs == 0
    assert sys.nr_of_outputs == 0
    
    # Stable system matrix: n x n
    A = np.array([[-0.3780, 0.2839, 0.5403, -0.2962],
                  [0.1362, 0.2742, 0.5195, 0.8266],
                  [0.0502, -0.1051, -0.6572, 0.3874],
                  [1.0227, -0.4877, 0.8342, -0.2372]])
    states = A.shape[0]
    
    # Input matrix: n x m
    B = 0.25 * np.array([[-2, 0, 3],
                         [2, 1, 0],
                         [0, 0, 1],
                         [0, -2, 1]])
    inputs = B.shape[1]
    
    # Constant offset: n x 1
    c = 0.05 * np.array([[-4], [2], [3], [1]])
    c_def = np.zeros((states, 1))
    
    # Output matrix: q x n
    C = np.array([[1, 1, 0, 0],
                  [0, -0.5, 0.5, 0]])
    outputs = C.shape[0]
    
    # Feedthrough matrix: q x m
    D = np.array([[0, 0, 1],
                  [0, 0, 0]])
    
    # Constant input: q x 1
    k = np.array([[0], [0.02]])
    k_def_n = np.zeros((states, 1))
    k_def_y = np.zeros((outputs, 1))
    
    # Disturbance matrix: n x r
    E = np.array([[1, 0.5], [0, -0.5], [1, -1], [0, 1]])
    E_def = np.zeros((states, 1))
    state_dist = E.shape[1]
    
    # Noise matrix: q x s
    F = np.array([[1], [0.5]])
    F_def_n = np.zeros((states, 1))
    F_def_y = np.zeros((outputs, 1))
    output_dist = F.shape[1]
    
    # Initialize different LinearSys objects
    sys_A = LinearSys(A)
    sys_AB = LinearSys(A, B)
    sys_ABC = LinearSys(A, B, None, C)
    sys_ABCD = LinearSys(A, B, None, C, D)
    sys_ABcCDk = LinearSys(A, B, c, C, D, k)
    sys_ABcCDkE = LinearSys(A, B, c, C, D, k, E)
    sys_ABcCDkEF = LinearSys(A, B, c, C, D, k, E, F)
    
    systems = [sys_A, sys_AB, sys_ABC, sys_ABCD, sys_ABcCDk, sys_ABcCDkE, sys_ABcCDkEF]
    
    # Expected values
    expected_states = [states, states, states, states, states, states, states]
    expected_inputs = [1, inputs, inputs, inputs, inputs, inputs, inputs]
    expected_outputs = [states, states, outputs, outputs, outputs, outputs, outputs]
    expected_state_dist = [1, 1, 1, 1, 1, state_dist, state_dist]
    expected_output_dist = [1, 1, 1, 1, 1, 1, output_dist]
    
    # Check properties of instantiated objects
    for i, sys in enumerate(systems):
        # Check dimensions
        assert sys.nr_of_dims == expected_states[i], f"Failed at system {i}: states"
        assert sys.nr_of_inputs == expected_inputs[i], f"Failed at system {i}: inputs"
        assert sys.nr_of_outputs == expected_outputs[i], f"Failed at system {i}: outputs"
        assert sys.nr_of_disturbances == expected_state_dist[i], f"Failed at system {i}: state_dist"
        assert sys.nr_of_noises == expected_output_dist[i], f"Failed at system {i}: output_dist"
        
        # Check that matrices have correct shapes
        assert sys.A.shape == (expected_states[i], expected_states[i])
        assert sys.B.shape[0] == expected_states[i]
        assert sys.c.shape == (expected_states[i], 1)


def test_linearSys_with_name():
    """Test LinearSys constructor with name parameter"""
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1], [0]])
    
    # Test with name as first argument
    sys1 = LinearSys("test_system", A, B)
    assert sys1.name == "test_system"
    assert np.array_equal(sys1.A, A)
    assert np.array_equal(sys1.B, B)
    
    # Test without name
    sys2 = LinearSys(A, B)
    assert sys2.name == "linearSys"
    assert np.array_equal(sys2.A, A)
    assert np.array_equal(sys2.B, B)


def test_linearSys_equality():
    """Test equality comparison between LinearSys objects"""
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1], [0]])
    
    sys1 = LinearSys(A, B)
    sys2 = LinearSys(A, B)
    sys3 = LinearSys(A, B + 0.1)
    
    # Test equality
    assert sys1 == sys2
    assert sys1.isequal(sys2)
    
    # Test inequality
    assert sys1 != sys3
    assert not sys1.isequal(sys3)


def test_linearSys_display():
    """Test display functionality"""
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1], [0]])
    
    sys = LinearSys("test_system", A, B)
    
    # This should not raise an exception
    sys.display()


def test_linearSys_input_validation():
    """Test input validation and error handling"""
    
    # Test invalid A matrix (not square)
    with pytest.raises(ValueError):
        LinearSys(np.array([[1, 2, 3], [4, 5, 6]]))
    
    # Test mismatched dimensions
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2, 3]])  # Wrong number of rows
    with pytest.raises(ValueError):
        LinearSys(A, B)


if __name__ == "__main__":
    test_linearSys_constructor()
    test_linearSys_with_name()
    test_linearSys_equality()
    test_linearSys_display()
    test_linearSys_input_validation()
    print("All tests passed!") 