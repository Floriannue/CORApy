"""
Test for neuralNetwork getRefinableLayers method

This test verifies that the getRefinableLayers method works correctly with different networks.
"""

import pytest
import numpy as np

def test_neuralNetwork_getRefinableLayers_basic():
    """Test getRefinableLayers method with basic network"""
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
    
    # Get refinable layers
    refinable_layers = nn.getRefinableLayers()
    
    # Should return list of refinable layers
    assert isinstance(refinable_layers, list)
    
    # Check that sigmoid layer is refinable
    assert any(isinstance(layer, nnSigmoidLayer) for layer in refinable_layers)

def test_neuralNetwork_getRefinableLayers_mixed():
    """Test getRefinableLayers method with mixed layer types"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
    from cora_python.nn.layers.nonlinear.nnTanhLayer import nnTanhLayer
    
    # Create a network with mixed layer types
    layers = [
        nnLinearLayer(np.array([[1, 2], [3, 4]])),
        nnSigmoidLayer(),
        nnLinearLayer(np.array([[1, 0], [0, 1]])),
        nnReLULayer(),
        nnLinearLayer(np.array([[1, 1], [1, 1]])),
        nnTanhLayer()
    ]
    
    nn = NeuralNetwork(layers)
    
    # Get refinable layers
    refinable_layers = nn.getRefinableLayers()
    
    # Should return list of refinable layers
    assert isinstance(refinable_layers, list)
    
    # Check that activation layers are refinable
    assert any(isinstance(layer, nnSigmoidLayer) for layer in refinable_layers)
    assert any(isinstance(layer, nnReLULayer) for layer in refinable_layers)
    assert any(isinstance(layer, nnTanhLayer) for layer in refinable_layers)
    
    # Check that linear layers are not refinable
    assert not any(isinstance(layer, nnLinearLayer) for layer in refinable_layers)

def test_neuralNetwork_getRefinableLayers_empty():
    """Test getRefinableLayers method with empty network"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    
    # Create empty network
    nn = NeuralNetwork([])
    
    # Get refinable layers
    refinable_layers = nn.getRefinableLayers()
    
    # Should return empty list
    assert isinstance(refinable_layers, list)
    assert len(refinable_layers) == 0

def test_neuralNetwork_getRefinableLayers_no_refinable():
    """Test getRefinableLayers method with network containing no refinable layers"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
    
    # Create network with only linear layers
    layers = [
        nnLinearLayer(np.array([[1, 2], [3, 4]])),
        nnLinearLayer(np.array([[1, 0], [0, 1]])),
        nnLinearLayer(np.array([[1, 1], [1, 1]]))
    ]
    
    nn = NeuralNetwork(layers)
    
    # Get refinable layers
    refinable_layers = nn.getRefinableLayers()
    
    # Should return empty list
    assert isinstance(refinable_layers, list)
    assert len(refinable_layers) == 0

def test_neuralNetwork_getRefinableLayers_single_refinable():
    """Test getRefinableLayers method with network containing single refinable layer"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer
    
    # Create network with single sigmoid layer
    layers = [nnSigmoidLayer()]
    
    nn = NeuralNetwork(layers)
    
    # Get refinable layers
    refinable_layers = nn.getRefinableLayers()
    
    # Should return list with single layer
    assert isinstance(refinable_layers, list)
    assert len(refinable_layers) == 1
    assert isinstance(refinable_layers[0], nnSigmoidLayer)

def test_neuralNetwork_getRefinableLayers_custom_refinable():
    """Test getRefinableLayers method with custom refinable layer"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.nnLayer import nnLayer
    
    # Create custom refinable layer
    class CustomRefinableLayer(nnLayer):
        def __init__(self):
            super().__init__()
            self.is_refinable = True
        
        def evaluate(self, input_data, options):
            return input_data
    
    # Create network with custom layer
    layers = [CustomRefinableLayer()]
    
    nn = NeuralNetwork(layers)
    
    # Get refinable layers
    refinable_layers = nn.getRefinableLayers()
    
    # Should return list with custom layer
    assert isinstance(refinable_layers, list)
    assert len(refinable_layers) == 1
    assert isinstance(refinable_layers[0], CustomRefinableLayer)
