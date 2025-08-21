"""
Test for neuralNetwork evaluate_ method

This test verifies that the evaluate_ method works correctly with different input types.
"""

import pytest
import numpy as np

def test_neuralNetwork_evaluate_numeric():
    """Test evaluate_ method with numeric input"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create a simple neural network
    W1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([[0], [0]])
    W2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([[0], [0]])
    
    layers = [
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2)
    ]
    
    nn = NeuralNetwork(layers)
    
    # Test numeric input
    x = np.array([[1], [2]])
    options = {}
    
    result = nn.evaluate_(x, options)
    
    # Should return numeric output
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 1)

def test_neuralNetwork_evaluate_interval():
    """Test evaluate_ method with interval input"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create a simple neural network
    W1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([[0], [0]])
    W2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([[0], [0]])
    
    layers = [
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2)
    ]
    
    nn = NeuralNetwork(layers)
    
    # Test interval input (mock interval object)
    class MockInterval:
        def __init__(self, inf, sup):
            self.inf = inf
            self.sup = sup
    
    x = MockInterval(np.array([[-1], [-1]]), np.array([[1], [1]]))
    options = {}
    
    result = nn.evaluate_(x, options)
    
    # Should return interval output
    assert hasattr(result, 'inf')
    assert hasattr(result, 'sup')

def test_neuralNetwork_evaluate_zonotope():
    """Test evaluate_ method with zonotope input"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create a simple neural network
    W1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([[0], [0]])
    W2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([[0], [0]])
    
    layers = [
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2)
    ]
    
    nn = NeuralNetwork(layers)
    
    # Test zonotope input (mock zonotope object)
    class MockZonotope:
        def __init__(self, center, generators):
            self.center = center
            self.generators = generators
    
    x = MockZonotope(np.array([[1], [2]]), np.array([[0.1, 0], [0, 0.1]]))
    options = {}
    
    result = nn.evaluate_(x, options)
    
    # Should return zonotope output
    assert hasattr(result, 'center')
    assert hasattr(result, 'generators')

def test_neuralNetwork_evaluate_polyZonotope():
    """Test evaluate_ method with polyZonotope input"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create a simple neural network
    W1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([[0], [0]])
    W2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([[0], [0]])
    
    layers = [
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2)
    ]
    
    nn = NeuralNetwork(layers)
    
    # Test polyZonotope input (mock polyZonotope object)
    class MockPolyZonotope:
        def __init__(self, center, generators):
            self.center = center
            self.generators = generators
    
    x = MockPolyZonotope(np.array([[1], [2]]), np.array([[0.1, 0], [0, 0.1]]))
    options = {}
    
    result = nn.evaluate_(x, options)
    
    # Should return polyZonotope output
    assert hasattr(result, 'center')
    assert hasattr(result, 'generators')

def test_neuralNetwork_evaluate_taylm():
    """Test evaluate_ method with taylm input"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create a simple neural network
    W1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([[0], [0]])
    W2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([[0], [0]])
    
    layers = [
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2)
    ]
    
    nn = NeuralNetwork(layers)
    
    # Test taylm input (mock taylm object)
    class MockTaylm:
        def __init__(self, monomials):
            self.monomials = monomials
    
    x = MockTaylm(np.array([[1], [2]]))
    options = {}
    
    result = nn.evaluate_(x, options)
    
    # Should return taylm output
    assert hasattr(result, 'monomials')

def test_neuralNetwork_evaluate_conZonotope():
    """Test evaluate_ method with conZonotope input"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create a simple neural network
    W1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([[0], [0]])
    W2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([[0], [0]])
    
    layers = [
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2)
    ]
    
    nn = NeuralNetwork(layers)
    
    # Test conZonotope input (mock conZonotope object)
    class MockConZonotope:
        def __init__(self, C, d):
            self.C = C
            self.d = d
    
    x = MockConZonotope(np.array([[1, 0], [0, 1]]), np.array([[0.1], [0.1]]))
    options = {}
    
    result = nn.evaluate_(x, options)
    
    # Should return conZonotope output
    assert hasattr(result, 'C')
    assert hasattr(result, 'd')

def test_neuralNetwork_evaluate_unsupported_type():
    """Test evaluate_ method with unsupported input type"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create a simple neural network
    W1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([[0], [0]])
    W2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([[0], [0]])
    
    layers = [
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2)
    ]
    
    nn = NeuralNetwork(layers)
    
    # Test unsupported input type
    x = "unsupported_type"
    options = {}
    
    with pytest.raises(NotImplementedError):
        nn.evaluate_(x, options)

def test_neuralNetwork_evaluate_with_layer_indices():
    """Test evaluate_ method with specific layer indices"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create a simple neural network
    W1 = np.array([[1, 2], [3, 4]])
    b1 = np.array([[0], [0]])
    W2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([[0], [0]])
    
    layers = [
        nnLinearLayer(W1, b1),
        nnSigmoidLayer(),
        nnLinearLayer(W2, b2)
    ]
    
    nn = NeuralNetwork(layers)
    
    # Test with specific layer indices
    x = np.array([[1], [2]])
    options = {}
    idxLayer = [1, 2]  # Only evaluate first two layers
    
    result = nn.evaluate_(x, options, idxLayer)
    
    # Should return output from specified layers
    assert isinstance(result, np.ndarray)
