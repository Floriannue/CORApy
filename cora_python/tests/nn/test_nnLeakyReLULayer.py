"""
Test for nnLeakyReLULayer class

This test verifies that the nnLeakyReLULayer class can be imported and basic functionality works.
"""

import pytest
import numpy as np

def test_import_nnLeakyReLULayer():
    """Test that nnLeakyReLULayer can be imported"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    assert nnLeakyReLULayer is not None

def test_nnLeakyReLULayer_constructor():
    """Test that nnLeakyReLULayer can be constructed"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    
    # Test with default alpha
    layer = nnLeakyReLULayer()
    assert layer.alpha == 0.01
    assert layer.is_refinable == True

def test_nnLeakyReLULayer_constructor_custom_alpha():
    """Test that nnLeakyReLULayer can be constructed with custom alpha"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    
    layer = nnLeakyReLULayer(alpha=0.2)
    assert layer.alpha == 0.2

def test_nnLeakyReLULayer_constructor_name():
    """Test that nnLeakyReLULayer can be constructed with custom name"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    
    layer = nnLeakyReLULayer(alpha=0.1, name="custom_leaky_relu")
    assert "custom_leaky_relu" in layer.name
    assert layer.alpha == 0.1

def test_nnLeakyReLULayer_evaluateNumeric():
    """Test that nnLeakyReLULayer evaluates numeric input correctly"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    
    layer = nnLeakyReLULayer(alpha=0.1)
    
    # Test input with positive and negative values
    x = np.array([[-2], [-1], [0], [1], [2]])
    options = {}
    
    result = layer.evaluateNumeric(x, options)
    
    # Expected: max(0.1 * x, x) = [-0.2, -0.1, 0, 1, 2]
    expected = np.array([[-0.2], [-0.1], [0], [1], [2]])
    
    assert np.allclose(result, expected, atol=1e-10)

def test_nnLeakyReLULayer_evaluateNumeric_2d():
    """Test that nnLeakyReLULayer evaluates 2D numeric input correctly"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    
    layer = nnLeakyReLULayer(alpha=0.2)
    
    # Test 2D input
    x = np.array([[-1, 2], [0, -3], [1, 0]])
    options = {}
    
    result = layer.evaluateNumeric(x, options)
    
    # Expected: max(0.2 * x, x) = [[-0.2, 2], [0, -0.6], [1, 0]]
    expected = np.array([[-0.2, 2], [0, -0.6], [1, 0]])
    
    assert np.allclose(result, expected, atol=1e-10)

def test_nnLeakyReLULayer_getDf():
    """Test getDf method returns correct derivative functions"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    
    layer = nnLeakyReLULayer(alpha=0.1)
    
    # Test derivative function (i=1)
    df = layer.getDf(1)
    
    # Test derivative evaluation
    x = np.array([[-1], [0], [1]])
    result = df(x)
    
    # Expected: 1 * (x > 0) + 0.1 * (x <= 0) = [0.1, 0.1, 1]
    expected = np.array([[0.1], [0.1], [1]])
    
    assert np.allclose(result, expected, atol=1e-10)

def test_nnLeakyReLULayer_getDerBounds():
    """Test getDerBounds method"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    
    layer = nnLeakyReLULayer(alpha=0.1)
    
    # Test case 1: l <= 0, u < 0
    df_l, df_u = layer.getDerBounds(-2, -1)
    assert df_l == 0.1
    assert df_u == 0.1
    
    # Test case 2: l <= 0, u >= 0
    df_l, df_u = layer.getDerBounds(-1, 1)
    assert df_l == 0.1
    assert df_u == 1
    
    # Test case 3: l > 0, u > 0
    df_l, df_u = layer.getDerBounds(1, 2)
    assert df_l == 1
    assert df_u == 1

def test_nnLeakyReLULayer_computeApproxPoly():
    """Test computeApproxPoly method"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    
    layer = nnLeakyReLULayer(alpha=0.1)
    
    # Test case 1: u <= 0 (negative region)
    coeffs, d = layer.computeApproxPoly(-2, -1)
    assert coeffs == [0.1, 0]
    assert d == 0
    
    # Test case 2: 0 <= l (positive region)
    coeffs, d = layer.computeApproxPoly(1, 2)
    assert coeffs == [1, 0]
    assert d == 0

def test_nnLeakyReLULayer_getFieldStruct():
    """Test getFieldStruct method"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    
    layer = nnLeakyReLULayer(alpha=0.3)
    field_struct = layer.getFieldStruct()
    
    assert 'alpha' in field_struct
    assert field_struct['alpha'] == 0.3

def test_nnLeakyReLULayer_getNumNeurons():
    """Test that nnLeakyReLULayer returns correct neuron counts"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    
    layer = nnLeakyReLULayer()
    nin, nout = layer.getNumNeurons()
    
    # Activation layers don't change neuron counts
    assert nin is None
    assert nout is None

def test_nnLeakyReLULayer_getOutputSize():
    """Test that nnLeakyReLULayer returns correct output size"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    
    layer = nnLeakyReLULayer()
    output_size = layer.getOutputSize([5, 3])
    
    # Activation layers don't change output size
    assert output_size == [5, 3]

def test_nnLeakyReLULayer_inheritance():
    """Test that nnLeakyReLULayer inherits correctly from nnActivationLayer"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    from cora_python.nn.layers.nonlinear.nnActivationLayer import nnActivationLayer
    from cora_python.nn.layers.nnLayer import nnLayer
    
    layer = nnLeakyReLULayer()
    
    # Check inheritance chain
    assert isinstance(layer, nnLeakyReLULayer)
    assert isinstance(layer, nnActivationLayer)
    assert isinstance(layer, nnLayer)

def test_nnLeakyReLULayer_special_values():
    """Test nnLeakyReLULayer with special input values"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    
    layer = nnLeakyReLULayer(alpha=0.1)
    options = {}
    
    # Test with zeros
    x_zeros = np.zeros((3, 2))
    result_zeros = layer.evaluateNumeric(x_zeros, options)
    assert np.allclose(result_zeros, np.zeros((3, 2)))
    
    # Test with very small negative values
    x_small = np.array([[-1e-10], [1e-10]])
    result_small = layer.evaluateNumeric(x_small, options)
    expected_small = np.array([[-1e-11], [1e-10]])  # alpha * x for negative, x for positive
    assert np.allclose(result_small, expected_small, atol=1e-15)

def test_nnLeakyReLULayer_alpha_zero():
    """Test nnLeakyReLULayer with alpha=0 (should behave like ReLU)"""
    from cora_python.nn.layers.nonlinear.nnLeakyReLULayer import nnLeakyReLULayer
    
    layer = nnLeakyReLULayer(alpha=0.0)
    
    # Test input with positive and negative values
    x = np.array([[-2], [-1], [0], [1], [2]])
    options = {}
    
    result = layer.evaluateNumeric(x, options)
    
    # Expected: max(0 * x, x) = max(0, x) = [0, 0, 0, 1, 2]
    expected = np.array([[0], [0], [0], [1], [2]])
    
    assert np.allclose(result, expected, atol=1e-10)
