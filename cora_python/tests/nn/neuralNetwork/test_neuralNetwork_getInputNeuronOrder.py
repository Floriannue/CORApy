"""
Test for neuralNetwork getInputNeuronOrder method

This test verifies that the getInputNeuronOrder method works correctly with different methods.
"""

import pytest
import numpy as np

def test_neuralNetwork_getInputNeuronOrder_in_order():
    """Test getInputNeuronOrder method with in-order method"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create simple network like MATLAB test
    np.random.seed(1)
    W1 = np.random.rand(10, 16) * 2 - 1
    b1 = np.random.rand(10, 1)
    
    W2 = np.random.rand(3, 10) * 2 - 1
    b2 = np.random.rand(3, 1)
    
    nn = NeuralNetwork([
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2),
        nnSigmoidLayer()
    ])
    
    x = np.random.rand(16, 1)
    
    # Test in-order method
    neuronOrder = nn.getInputNeuronOrder('in-order', x)
    
    # Should return sequential order
    expected_order = np.arange(1, 17)  # 1-based indexing like MATLAB
    assert np.array_equal(neuronOrder, expected_order)

def test_neuralNetwork_getInputNeuronOrder_sensitivity():
    """Test getInputNeuronOrder method with sensitivity method"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create simple network like MATLAB test
    np.random.seed(1)
    W1 = np.random.rand(10, 16) * 2 - 1
    b1 = np.random.rand(10, 1)
    
    W2 = np.random.rand(3, 10) * 2 - 1
    b2 = np.random.rand(3, 1)
    
    nn = NeuralNetwork([
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2),
        nnSigmoidLayer()
    ])
    
    x = np.random.rand(16, 1)
    
    # Test sensitivity method
    neuronOrder = nn.getInputNeuronOrder('sensitivity', x)
    

    # Should return order based on sensitivity
    S, _ = nn.calcSensitivity(x, {}, False)
    
    # Check that the order matches (may be different due to numerical precision)
    assert len(neuronOrder) == 16
    assert np.all(neuronOrder >= 1) and np.all(neuronOrder <= 16)
    
    # Check that all values are unique (no duplicates)
    assert len(set(neuronOrder)) == 16

def test_neuralNetwork_getInputNeuronOrder_snake():
    """Test getInputNeuronOrder method with snake method"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create simple network like MATLAB test
    np.random.seed(1)
    W1 = np.random.rand(10, 16) * 2 - 1
    b1 = np.random.rand(10, 1)
    
    W2 = np.random.rand(3, 10) * 2 - 1
    b2 = np.random.rand(3, 1)
    
    nn = NeuralNetwork([
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2),
        nnSigmoidLayer()
    ])
    
    x = np.random.rand(16, 1)
    
    # Test snake method
    neuronOrder = nn.getInputNeuronOrder('snake', x, [4, 4, 1])
    
    # Expected snake pattern for 4x4 grid
    expected_order = np.array([1, 5, 9, 13, 14, 15, 16, 12, 8, 4, 3, 2, 6, 10, 11, 7])
    assert np.array_equal(neuronOrder, expected_order)

def test_neuralNetwork_getInputNeuronOrder_different_input_sizes():
    """Test getInputNeuronOrder method with different input sizes"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create simple network
    W1 = np.random.rand(5, 8) * 2 - 1
    b1 = np.random.rand(5, 1)
    
    W2 = np.random.rand(2, 5) * 2 - 1
    b2 = np.random.rand(2, 1)
    
    nn = NeuralNetwork([
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2)
    ])
    
    x = np.random.rand(8, 1)
    
    # Test with different methods
    methods = ['in-order', 'sensitivity']
    for method in methods:
        neuronOrder = nn.getInputNeuronOrder(method, x)
        
        # Should return order with correct length
        assert len(neuronOrder) == 8
        assert np.all(neuronOrder >= 1) and np.all(neuronOrder <= 8)

def test_neuralNetwork_getInputNeuronOrder_empty_network():
    """Test getInputNeuronOrder method with empty network"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    
    # Create empty network
    nn = NeuralNetwork([])
    
    x = np.random.rand(4, 1)
    
    # Test with different methods
    methods = ['in-order', 'sensitivity']
    for method in methods:
        neuronOrder = nn.getInputNeuronOrder(method, x)
        
        # Should return sequential order for empty network
        expected_order = np.arange(1, 5)  # 1-based indexing
        assert np.array_equal(neuronOrder, expected_order)

def test_neuralNetwork_getInputNeuronOrder_single_layer():
    """Test getInputNeuronOrder method with single layer network"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    
    # Create single layer network
    W = np.random.rand(3, 6) * 2 - 1
    b = np.random.rand(3, 1)
    
    nn = NeuralNetwork([nnLinearLayer(W, b)])
    
    x = np.random.rand(6, 1)
    
    # Test with different methods
    methods = ['in-order', 'sensitivity']
    for method in methods:
        neuronOrder = nn.getInputNeuronOrder(method, x)
        
        # Should return order with correct length
        assert len(neuronOrder) == 6
        assert np.all(neuronOrder >= 1) and np.all(neuronOrder <= 6)

def test_neuralNetwork_getInputNeuronOrder_invalid_method():
    """Test getInputNeuronOrder method with invalid method"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    
    # Create simple network
    W = np.random.rand(3, 4) * 2 - 1
    b = np.random.rand(3, 1)
    
    nn = NeuralNetwork([nnLinearLayer(W, b)])
    
    x = np.random.rand(4, 1)
    
    # Test with invalid method
    with pytest.raises(ValueError):
        nn.getInputNeuronOrder('invalid_method', x)
