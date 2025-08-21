"""
Test for neuralNetwork refine method

This test verifies that the refine method works correctly with different parameters.
"""

import pytest
import numpy as np

def test_neuralNetwork_refine_default():
    """Test refine method with default parameters"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create network
    nn = NeuralNetwork([
        nnLinearLayer(np.array([[2, 3], [4, 5]])),
        nnSigmoidLayer(),
        nnLinearLayer(np.array([[-1, 5], [2, -3]])),
        nnSigmoidLayer()
    ])
    
    # Evaluate point
    x = np.array([[5], [2]])
    y = nn.evaluate(x)
    
    # Add uncertainty (mock polyZonotope)
    class MockPolyZonotope:
        def __init__(self, center, generators):
            self.center = center
            self.generators = generators
    
    X = MockPolyZonotope(x, np.eye(2) * 0.01)
    
    # Test default refine
    nn.reset()
    nn.evaluate(X)
    nn.refine()

def test_neuralNetwork_refine_max_order():
    """Test refine method with max_order parameter"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create network
    nn = NeuralNetwork([
        nnLinearLayer(np.array([[2, 3], [4, 5]])),
        nnSigmoidLayer(),
        nnLinearLayer(np.array([[-1, 5], [2, -3]])),
        nnSigmoidLayer()
    ])
    
    # Evaluate point
    x = np.array([[5], [2]])
    y = nn.evaluate(x)
    
    # Add uncertainty (mock polyZonotope)
    class MockPolyZonotope:
        def __init__(self, center, generators):
            self.center = center
            self.generators = generators
    
    X = MockPolyZonotope(x, np.eye(2) * 0.01)
    
    # Test max_order = 2
    nn.reset()
    nn.evaluate(X)
    nn.refine(2)

def test_neuralNetwork_refine_max_order_5():
    """Test refine method with max_order = 5"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create network
    nn = NeuralNetwork([
        nnLinearLayer(np.array([[2, 3], [4, 5]])),
        nnSigmoidLayer(),
        nnLinearLayer(np.array([[-1, 5], [2, -3]])),
        nnSigmoidLayer()
    ])
    
    # Evaluate point
    x = np.array([[5], [2]])
    y = nn.evaluate(x)
    
    # Add uncertainty (mock polyZonotope)
    class MockPolyZonotope:
        def __init__(self, center, generators):
            self.center = center
            self.generators = generators
    
    X = MockPolyZonotope(x, np.eye(2) * 0.01)
    
    # Test max_order = 5 with different refinement types
    nn.reset()
    nn.evaluate(X)
    nn.refine(5, "layer")
    
    nn.reset()
    nn.evaluate(X)
    nn.refine(5, "neuron")
    
    nn.reset()
    nn.evaluate(X)
    nn.refine(5, "all")

def test_neuralNetwork_refine_different_types():
    """Test refine method with different refinement types"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create network
    nn = NeuralNetwork([
        nnLinearLayer(np.array([[2, 3], [4, 5]])),
        nnSigmoidLayer(),
        nnLinearLayer(np.array([[-1, 5], [2, -3]])),
        nnSigmoidLayer()
    ])
    
    # Evaluate point
    x = np.array([[5], [2]])
    y = nn.evaluate(x)
    
    # Add uncertainty (mock polyZonotope)
    class MockPolyZonotope:
        def __init__(self, center, generators):
            self.center = center
            self.generators = generators
    
    X = MockPolyZonotope(x, np.eye(2) * 0.01)
    
    # Test different refinement types
    nn.reset()
    nn.evaluate(X)
    nn.refine(5, "layer", "approx_error")
    
    nn.reset()
    nn.evaluate(X)
    nn.refine(5, "layer", "sensitivity", x)
    
    nn.reset()
    nn.evaluate(X)
    nn.refine(5, "layer", "both", x)
    
    nn.reset()
    nn.evaluate(X)
    nn.refine(5, "layer", "random")

def test_neuralNetwork_refine_empty_network():
    """Test refine method with empty network"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    
    # Create empty network
    nn = NeuralNetwork([])
    
    # Should not raise error
    nn.refine()
    nn.refine(5)
    nn.refine(5, "layer")

def test_neuralNetwork_refine_single_layer():
    """Test refine method with single layer network"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    
    # Create single layer network
    nn = NeuralNetwork([nnLinearLayer(np.array([[1, 2], [3, 4]]))])
    
    # Should not raise error
    nn.refine()
    nn.refine(3, "layer")
