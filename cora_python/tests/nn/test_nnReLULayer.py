"""
Test for nnReLULayer class

This test verifies that the nnReLULayer class can be imported and basic functionality works.
"""

import pytest
import numpy as np

def test_import_nnReLULayer():
    """Test that nnReLULayer can be imported"""
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    assert nnReLULayer is not None

def test_nnReLULayer_constructor():
    """Test that nnReLULayer can be constructed"""
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    
    # Test with default name
    layer = nnReLULayer()
    assert layer.alpha == 0
    assert layer.is_refinable == True

def test_nnReLULayer_constructor_name():
    """Test that nnReLULayer can be constructed with custom name"""
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    
    layer = nnReLULayer("custom_relu")
    assert "custom_relu" in layer.name

def test_nnReLULayer_evaluateNumeric():
    """Test that nnReLULayer evaluates numeric input correctly"""
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    
    layer = nnReLULayer()
    
    # Test input with positive and negative values
    x = np.array([[-2], [-1], [0], [1], [2]])
    options = {}
    
    result = layer.evaluateNumeric(x, options)
    
    # Expected: max(0, x) = [0, 0, 0, 1, 2]
    expected = np.array([[0], [0], [0], [1], [2]])
    
    assert np.allclose(result, expected, atol=1e-10)

def test_nnReLULayer_evaluateNumeric_2d():
    """Test that nnReLULayer evaluates 2D numeric input correctly"""
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    
    layer = nnReLULayer()
    
    # Test 2D input
    x = np.array([[-1, 2], [0, -3], [1, 0]])
    options = {}
    
    result = layer.evaluateNumeric(x, options)
    
    # Expected: max(0, x) = [[0, 2], [0, 0], [1, 0]]
    expected = np.array([[0, 2], [0, 0], [1, 0]])
    
    assert np.allclose(result, expected, atol=1e-10)

def test_nnReLULayer_getNumNeurons():
    """Test that nnReLULayer returns correct neuron counts"""
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    
    layer = nnReLULayer()
    nin, nout = layer.getNumNeurons()
    
    # Activation layers don't change neuron counts
    assert nin is None
    assert nout is None

def test_nnReLULayer_getOutputSize():
    """Test that nnReLULayer returns correct output size"""
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    
    layer = nnReLULayer()
    output_size = layer.getOutputSize([5, 3])
    
    # Activation layers don't change output size
    assert output_size == [5, 3]

def test_nnReLULayer_getMergeBuckets():
    """Test that nnReLULayer returns correct merge buckets"""
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    
    layer = nnReLULayer()
    buckets = layer.getMergeBuckets()
    
    assert buckets == 0

def test_nnReLULayer_inheritance():
    """Test that nnReLULayer inherits correctly from nnLeakyReLULayer"""
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    from cora_python.nn.layers.nonlinear.nnActivationLayer import nnActivationLayer
    from cora_python.nn.layers.nnLayer import nnLayer
    
    layer = nnReLULayer()
    
    # Check inheritance chain
    assert isinstance(layer, nnReLULayer)
    assert isinstance(layer, nnLeakyReLULayer)
    assert isinstance(layer, nnActivationLayer)
    assert isinstance(layer, nnLayer)
