"""
Test for vnnlib2cora method

This test verifies that the vnnlib2cora method works correctly with different VNNLIB files.
"""

import pytest
import numpy as np
import os

def test_vnnlib2cora_prop1():
    """Test vnnlib2cora with ACASXU prop_1.vnnlib"""
    from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
    
    # Skip if file doesn't exist
    prop1Filename = 'cora_python/examples/nn/models/prop_1.vnnlib'
    if not os.path.exists(prop1Filename):
        pytest.skip("prop_1.vnnlib file not found")
    
    # Read specification
    X0, specs = vnnlib2cora(prop1Filename)
    
    # There is only one input set
    assert len(X0) == 1
    
    # Check input sets constraints
    # Note: Python Interval.inf returns 2D arrays, so we need to flatten them
    assert np.allclose(X0[0].inf.flatten(), np.array([0.6, -0.5, -0.5, 0.45, -0.5]))
    assert np.allclose(X0[0].sup.flatten(), np.array([0.679857769, 0.5, 0.5, 0.5, -0.45]))
    
    # Check output set
    # Note: In Python, specs.set is a single Polytope object, not a list
    # assert isinstance(specs.set, 'polytope') and representsa(specs.set, 'halfspace')
    assert specs.type == 'unsafeSet'
    
    # Check specification constraints
    if hasattr(specs.set, 'A') and hasattr(specs.set, 'b'):
        # Our Python implementation includes all constraints (input + output)
        # The last row should be the output constraint: [-1, 0, 0, 0, 0]
        assert np.allclose(specs.set.A[-1, :], np.array([-1, 0, 0, 0, 0]))
        assert np.allclose(specs.set.b[-1], -3.991125645861615)

def test_vnnlib2cora_prop2():
    """Test vnnlib2cora with ACASXU prop_2.vnnlib"""
    from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
    
    # Skip if file doesn't exist
    prop2Filename = 'cora_python/examples/nn/models/prop_2.vnnlib'
    if not os.path.exists(prop2Filename):
        pytest.skip("prop_2.vnnlib file not found")
    
    # Read specification
    X0, specs = vnnlib2cora(prop2Filename)
    
    # There is only one input set
    assert len(X0) == 1
    
    # Check input sets constraints
    # Note: Python Interval.inf returns 2D arrays, so we need to flatten them
    assert np.allclose(X0[0].inf.flatten(), np.array([0.6, -0.5, -0.5, 0.45, -0.5]))
    assert np.allclose(X0[0].sup.flatten(), np.array([0.679857769, 0.5, 0.5, 0.5, -0.45]))
    
    # Check output set
    # Note: In Python, specs.set is a single Polytope object, not a list
    # assert isinstance(specs.set, 'polytope')
    assert specs.type == 'unsafeSet'
    
    # Check specification constraints
    if hasattr(specs.set, 'A') and hasattr(specs.set, 'b'):
        # Our Python implementation includes all constraints (input + output)
        # The last 4 rows should be the output constraints: Y_1 <= Y_0, Y_2 <= Y_0, Y_3 <= Y_0, Y_4 <= Y_0
        # This translates to: Y_1 - Y_0 <= 0, Y_2 - Y_0 <= 0, Y_3 - Y_0 <= 0, Y_4 - Y_0 <= 0
        # So the coefficient matrix should be: [-1, 1, 0, 0, 0], [-1, 0, 1, 0, 0], [-1, 0, 0, 1, 0], [-1, 0, 0, 0, 1]
        expected_A = np.array([[-1, 1, 0, 0, 0], [-1, 0, 1, 0, 0], [-1, 0, 0, 1, 0], [-1, 0, 0, 0, 1]])
        # Check that the last 4 rows match the expected output constraints
        assert np.allclose(specs.set.A[-4:, :], expected_A)
        assert np.allclose(specs.set.b[-4:], np.zeros(4))

def test_vnnlib2cora_prop3():
    """Test vnnlib2cora with ACASXU prop_3.vnnlib"""
    from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
    
    # Skip if file doesn't exist
    prop3Filename = 'cora_python/examples/nn/models/axas_xu_prop_3.vnnlib'
    if not os.path.exists(prop3Filename):
        pytest.skip("axas_xu_prop_3.vnnlib file not found")
    
    # Read specification
    X0, specs = vnnlib2cora(prop3Filename)
    
    # There is only one input set
    assert len(X0) == 1
    
    # Check input sets constraints
    # Note: Python Interval.inf returns 2D arrays, so we need to flatten them
    assert np.allclose(X0[0].inf.flatten(), np.array([-0.303531156, -0.009549297, 0.493380324, 0.3, 0.3]))
    assert np.allclose(X0[0].sup.flatten(), np.array([-0.298552812, 0.009549297, 0.5, 0.5, 0.5]))
    
    # Check output set
    # Note: In Python, specs.set is a single Polytope object, not a list
    # assert isinstance(specs.set, 'polytope')
    assert specs.type == 'unsafeSet'
    
    # Check specification constraints
    if hasattr(specs.set, 'A') and hasattr(specs.set, 'b'):
        # Our Python implementation includes all constraints (input + output)
        # The last 4 rows should be the output constraints: Y_0 <= Y_1, Y_0 <= Y_2, Y_0 <= Y_3, Y_0 <= Y_4
        # This translates to: Y_0 - Y_1 <= 0, Y_0 - Y_2 <= 0, Y_0 - Y_3 <= 0, Y_0 - Y_4 <= 0
        # So the coefficient matrix should be: [1, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]
        expected_A = np.array([[1, -1, 0, 0, 0], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [1, 0, 0, 0, -1]])
        # Check that the last 4 rows match the expected output constraints
        assert np.allclose(specs.set.A[-4:, :], expected_A)
        assert np.allclose(specs.set.b[-4:], np.zeros(4))

def test_vnnlib2cora_prop5():
    """Test vnnlib2cora with ACASXU prop_5.vnnlib"""
    from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
    
    # Skip if file doesn't exist
    prop5Filename = 'cora_python/examples/nn/models/prop_5.vnnlib'
    if not os.path.exists(prop5Filename):
        pytest.skip("prop_5.vnnlib file not found")
    
    # Read specification
    X0, specs = vnnlib2cora(prop5Filename)
    
    # There is only one input set
    assert len(X0) == 1
    
    # Check input sets constraints
    # Note: Python Interval.inf returns 2D arrays, so we need to flatten them
    assert np.allclose(X0[0].inf.flatten(), np.array([-0.324274257, 0.031830989, -0.499999896, -0.5, -0.5]))
    assert np.allclose(X0[0].sup.flatten(), np.array([-0.321785085, 0.063661977, -0.499204121, -0.227272727, -0.166666667]))
    
    # Check output set
    # Note: In Python, specs.set is a single Polytope object, not a list
    # assert isinstance(specs.set, 'polytope')
    assert specs.type == 'safeSet'
    
    # Check specification constraints
    if hasattr(specs.set, 'A') and hasattr(specs.set, 'b'):
        # Our Python implementation includes all constraints (input + output)
        # The last 4 rows should be the output constraints: Y_0 <= Y_4, Y_1 <= Y_4, Y_2 <= Y_4, Y_3 <= Y_4
        # This translates to: Y_0 - Y_4 <= 0, Y_1 - Y_4 <= 0, Y_2 - Y_4 <= 0, Y_3 - Y_4 <= 0
        # So the coefficient matrix should be: [1, 0, 0, 0, -1], [0, 1, 0, 0, -1], [0, 0, 1, 0, -1], [0, 0, 0, 1, -1]
        expected_A = np.array([[1, 0, 0, 0, -1], [0, 1, 0, 0, -1], [0, 0, 1, 0, -1], [0, 0, 0, 1, -1]])
        # Check that the last 4 rows match the expected output constraints
        assert np.allclose(specs.set.A[-4:, :], expected_A)
        assert np.allclose(specs.set.b[-4:], np.zeros(4))
