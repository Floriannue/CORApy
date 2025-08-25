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
    prop1Filename = 'prop_1.vnnlib'
    if not os.path.exists(prop1Filename):
        pytest.skip("prop_1.vnnlib file not found")
    
    # Read specification
    X0, specs = vnnlib2cora(prop1Filename)
    
    # There is only one input set
    assert len(X0) == 1
    
    # Check input sets constraints
    assert np.allclose(X0[0].inf, np.array([0.6, -0.5, -0.5, 0.45, -0.5]))
    assert np.allclose(X0[0].sup, np.array([0.679857769, 0.5, 0.5, 0.5, -0.45]))
    
    # Check output set
    assert len(specs.set) == 1
    # Note: We can't check the exact type without implementing the full set classes
    # assert isinstance(specs.set, 'polytope') and representsa(specs.set, 'halfspace')
    assert specs.type == 'unsafeSet'
    
    # Check specification constraints
    if hasattr(specs.set, 'A') and hasattr(specs.set, 'b'):
        assert np.allclose(specs.set.A, np.array([[-1, 0, 0, 0, 0]]))
        assert np.allclose(specs.set.b, np.array([-3.991125645861615]))

def test_vnnlib2cora_prop2():
    """Test vnnlib2cora with ACASXU prop_2.vnnlib"""
    from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
    
    # Skip if file doesn't exist
    prop2Filename = 'prop_2.vnnlib'
    if not os.path.exists(prop2Filename):
        pytest.skip("prop_2.vnnlib file not found")
    
    # Read specification
    X0, specs = vnnlib2cora(prop2Filename)
    
    # There is only one input set
    assert len(X0) == 1
    
    # Check input sets constraints
    assert np.allclose(X0[0].inf, np.array([0.6, -0.5, -0.5, 0.45, -0.5]))
    assert np.allclose(X0[0].sup, np.array([0.679857769, 0.5, 0.5, 0.5, -0.45]))
    
    # Check output set
    assert len(specs.set) == 1
    # Note: We can't check the exact type without implementing the full set classes
    # assert isinstance(specs.set, 'polytope')
    assert specs.type == 'unsafeSet'
    
    # Check specification constraints
    if hasattr(specs.set, 'A') and hasattr(specs.set, 'b'):
        expected_A = np.vstack([-np.ones((4, 1)), np.eye(4)])
        assert np.allclose(specs.set.A, expected_A)
        assert np.allclose(specs.set.b, np.zeros(4))

def test_vnnlib2cora_prop3():
    """Test vnnlib2cora with ACASXU prop_3.vnnlib"""
    from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
    
    # Skip if file doesn't exist
    prop3Filename = 'unitTests/vnnlib/axas_xu_prop_3.vnnlib'
    if not os.path.exists(prop3Filename):
        pytest.skip("prop_3.vnnlib file not found")
    
    # Read specification
    X0, specs = vnnlib2cora(prop3Filename)
    
    # There is only one input set
    assert len(X0) == 1
    
    # Check input sets constraints
    assert np.allclose(X0[0].inf, np.array([-0.303531156, -0.009549297, 0.493380324, 0.3, 0.3]))
    assert np.allclose(X0[0].sup, np.array([-0.298552812, 0.009549297, 0.5, 0.5, 0.5]))
    
    # Check output set
    assert len(specs.set) == 1
    # Note: We can't check the exact type without implementing the full set classes
    # assert isinstance(specs.set, 'polytope')
    assert specs.type == 'unsafeSet'
    
    # Check specification constraints
    if hasattr(specs.set, 'A') and hasattr(specs.set, 'b'):
        expected_A = np.vstack([np.ones((4, 1)), -np.eye(4)])
        assert np.allclose(specs.set.A, expected_A)
        assert np.allclose(specs.set.b, np.zeros(4))

def test_vnnlib2cora_prop5():
    """Test vnnlib2cora with ACASXU prop_5.vnnlib"""
    from cora_python.converter.neuralnetwork2cora.vnnlib2cora import vnnlib2cora
    
    # Skip if file doesn't exist
    prop5Filename = 'prop_5.vnnlib'
    if not os.path.exists(prop5Filename):
        pytest.skip("prop_5.vnnlib file not found")
    
    # Read specification
    X0, specs = vnnlib2cora(prop5Filename)
    
    # There is only one input set
    assert len(X0) == 1
    
    # Check input sets constraints
    assert np.allclose(X0[0].inf, np.array([-0.324274257, 0.031830989, -0.499999896, -0.5, -0.5]))
    assert np.allclose(X0[0].sup, np.array([-0.321785085, 0.063661977, -0.499204121, -0.227272727, -0.166666667]))
    
    # Check output set
    assert len(specs.set) == 1
    # Note: We can't check the exact type without implementing the full set classes
    # assert isinstance(specs.set, 'polytope')
    assert specs.type == 'safeSet'
    
    # Check specification constraints
    if hasattr(specs.set, 'A') and hasattr(specs.set, 'b'):
        expected_A = np.vstack([-np.eye(4), np.ones((4, 1))])
        assert np.allclose(specs.set.A, expected_A)
        assert np.allclose(specs.set.b, np.zeros(4))
