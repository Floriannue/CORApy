"""
Test file for NeuralNetwork.evaluate_ method

This file tests the internal evaluation method of the NeuralNetwork class.
"""

import pytest
import numpy as np
from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer

class TestNeuralNetworkEvaluate:
    """Test class for NeuralNetwork.evaluate_ method"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create a simple neural network
        W1 = np.array([[1, 2], [3, 4]])
        b1 = np.array([[0], [0]])
        W2 = np.array([[1, 0], [0, 1]])
        b2 = np.array([[0], [0]])
        
        layers = [
            nnLinearLayer(W1, b1),
            nnReLULayer(),
            nnLinearLayer(W2, b2)
        ]
        
        self.nn = NeuralNetwork(layers)
    
    def test_evaluate_numeric_input(self):
        """Test evaluate_ with numeric input"""
        x = np.array([[1], [2]])
        options = {}
        
        result = self.nn.evaluate_(x, options)
        
        # Should return numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2  # Output dimension
    
    def test_evaluate_with_layer_indices(self):
        """Test evaluate_ with specific layer indices"""
        x = np.array([[1], [2]])
        options = {}
        idxLayer = [0, 1]  # Only first two layers
        
        result = self.nn.evaluate_(x, options, idxLayer)
        
        # Should return numpy array
        assert isinstance(result, np.ndarray)
    
    def test_evaluate_with_options(self):
        """Test evaluate_ with various options"""
        x = np.array([[1], [2]])
        options = {
            'nn': {
                'train': {
                    'backprop': True
                }
            }
        }
        
        result = self.nn.evaluate_(x, options)
        
        # Should return numpy array
        assert isinstance(result, np.ndarray)
        
        # Check if backprop storage was set up
        for layer in self.nn.layers:
            if hasattr(layer, 'backprop'):
                assert 'store' in layer.backprop
    
    def test_evaluate_unsupported_input_type(self):
        """Test evaluate_ with unsupported input type"""
        x = "unsupported_input"
        options = {}
        
        with pytest.raises(NotImplementedError):
            self.nn.evaluate_(x, options)
    
    def test_evaluate_none_options(self):
        """Test evaluate_ with None options"""
        x = np.array([[1], [2]])
        
        result = self.nn.evaluate_(x, None)
        
        # Should work with default options
        assert isinstance(result, np.ndarray)
    
    def test_evaluate_none_idxLayer(self):
        """Test evaluate_ with None idxLayer (should use all layers)"""
        x = np.array([[1], [2]])
        options = {}
        
        result = self.nn.evaluate_(x, options, None)
        
        # Should evaluate through all layers
        assert isinstance(result, np.ndarray)
    
    def test_evaluate_empty_layers(self):
        """Test evaluate_ with empty network"""
        empty_nn = NeuralNetwork([])
        x = np.array([[1], [2]])
        options = {}
        
        result = empty_nn.evaluate_(x, options)
        
        # Should return input unchanged
        np.testing.assert_array_equal(result, x)
