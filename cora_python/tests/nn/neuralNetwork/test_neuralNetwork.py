"""
Test file for NeuralNetwork constructor

This file tests the basic constructor functionality of the NeuralNetwork class.
"""

import pytest
import numpy as np
from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer
from cora_python.nn.layers.nonlinear.nnTanhLayer import nnTanhLayer
from cora_python.nn.layers.nonlinear.nnSigmoidLayer import nnSigmoidLayer

class TestNeuralNetwork:
    """Test class for NeuralNetwork constructor"""
    
    def test_neuralNetwork_constructor_basic(self):
        """Test basic constructor with layers - matches MATLAB test exactly"""
        # Create simple layers - matches MATLAB: layers = cell(2, 1);
        W = np.random.rand(4, 3)  # matches MATLAB: W = rand(4,3)
        b = np.random.rand(4, 1)  # matches MATLAB: b = rand(4,1)
        
        layers = [
            nnLinearLayer(W, b),  # matches MATLAB: layers{1} = nnLinearLayer(W, b)
            nnTanhLayer()          # matches MATLAB: layers{2} = nnTanhLayer()
        ]
        
        nn = NeuralNetwork(layers)  # matches MATLAB: nn = neuralNetwork(layers)
        
        # Check neuron counts - matches MATLAB: assert(nn.neurons_in == 3 && nn.neurons_out == 4)
        assert nn.neurons_in == 3
        assert nn.neurons_out == 4
    
    def test_neuralNetwork_constructor_larger_example(self):
        """Test larger neural network construction - matches MATLAB test exactly"""
        # Create larger network - matches MATLAB: layers = cell(4, 1);
        W1 = np.random.rand(10, 2)  # matches MATLAB: W1 = rand(10,2)
        b1 = np.random.rand(10, 1)  # matches MATLAB: b1 = rand(10, 1)
        W2 = np.random.rand(3, 10)  # matches MATLAB: W2 = rand(3, 10)
        b2 = np.random.rand(3, 1)   # matches MATLAB: b2 = rand(3,1)
        
        layers = [
            nnLinearLayer(W1, b1),   # matches MATLAB: layers{1} = nnLinearLayer(W1, b1)
            nnReLULayer(),            # matches MATLAB: layers{2} = nnReLULayer()
            nnLinearLayer(W2, b2),   # matches MATLAB: layers{3} = nnLinearLayer(W2, b2)
            nnSigmoidLayer()          # matches MATLAB: layers{4} = nnSigmoidLayer()
        ]
        
        nn = NeuralNetwork(layers)   # matches MATLAB: nn = neuralNetwork(layers)
        
        # Check neuron counts - matches MATLAB: assert(nn.neurons_in == 2 && nn.neurons_out == 3)
        assert nn.neurons_in == 2
        assert nn.neurons_out == 3
    
    def test_neuralNetwork_constructor_wrong_input(self):
        """Test that constructor fails with wrong input - matches MATLAB test exactly"""
        # matches MATLAB: assertThrowsAs(@neuralNetwork,'CORA:wrongInputInConstructor',nnSigmoidLayer());
        # Test with non-list input (should fail) 
        with pytest.raises(ValueError, match="First argument should be a list of type nnLayer"):
            NeuralNetwork(nnSigmoidLayer())  # This should fail as it's not a list
    
    def test_neuralNetwork_constructor_empty(self):
        """Test constructor with empty layers list"""
        # MATLAB allows empty layers list - it just doesn't set neurons_in/out
        nn = NeuralNetwork([])
        assert nn.neurons_in is None
        assert nn.neurons_out is None
        assert len(nn.layers) == 0
    
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
