"""
Test file for NeuralNetwork constructor

This file tests the basic constructor functionality of the NeuralNetwork class.
"""

import pytest
import numpy as np
from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer

class TestNeuralNetwork:
    """Test class for NeuralNetwork constructor"""
    
    def test_neuralNetwork_constructor_basic(self):
        """Test basic constructor with layers"""
        # Create simple layers
        W1 = np.array([[1, 2], [3, 4]])
        b1 = np.array([[0], [0]])
        W2 = np.array([[1, 0], [0, 1]])
        b2 = np.array([[0], [0]])
        
        layers = [
            nnLinearLayer(W1, b1),
            nnReLULayer(),
            nnLinearLayer(W2, b2)
        ]
        
        nn = NeuralNetwork(layers)
        
        assert len(nn.layers) == 3
        assert nn.name == "Neural Network"
        assert nn.neurons_in == 2  # Set by first layer (W1 has 2 input features)
        assert nn.neurons_out == 2  # Set by last layer (W2 has 2 output features)
        assert nn.reductionRate == 1  # Default value from MATLAB
    
    def test_neuralNetwork_constructor_empty(self):
        """Test constructor with empty layers list"""
        # MATLAB throws an error when calling setInputSize() with empty layers
        # because neurons_in and neurons_out are empty
        with pytest.raises(Exception):
            nn = NeuralNetwork([])
    
    def test_neuralNetwork_constructor_custom_name(self):
        """Test constructor with custom name"""
        # Create a valid network with layers to test custom name
        W1 = np.array([[1, 2], [3, 4]])
        b1 = np.array([[0], [0]])
        layers = [nnLinearLayer(W1, b1)]
        nn = NeuralNetwork(layers, name="Custom Network")
        
        assert nn.name == "Custom Network"
        assert nn.neurons_in == 2  # Set by layer
        assert nn.neurons_out == 2  # Set by layer
    
    def test_neuralNetwork_constructor_layer_properties(self):
        """Test that layer properties are properly initialized"""
        # Create layers with known properties
        W1 = np.array([[1, 2], [3, 4]])
        b1 = np.array([[0], [0]])
        W2 = np.array([[1, 0], [0, 1]])
        b2 = np.array([[0], [0]])
        
        layers = [
            nnLinearLayer(W1, b1),
            nnReLULayer(),
            nnLinearLayer(W2, b2)
        ]
        
        nn = NeuralNetwork(layers)
        
        # Test that layers are properly stored
        assert nn.layers[0] is layers[0]
        assert nn.layers[1] is layers[1]
        assert nn.layers[2] is layers[2]
        
        # Test that layers have the expected types
        assert isinstance(nn.layers[0], nnLinearLayer)
        assert isinstance(nn.layers[1], nnReLULayer)
        assert isinstance(nn.layers[2], nnLinearLayer)
