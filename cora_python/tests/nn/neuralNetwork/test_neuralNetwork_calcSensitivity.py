"""
Test file for NeuralNetwork.calcSensitivity method

This file tests the sensitivity calculation method of the NeuralNetwork class.
"""

import pytest
import numpy as np
from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer

class TestNeuralNetworkCalcSensitivity:
    """Test class for NeuralNetwork.calcSensitivity method"""
    
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
    
    def test_calcSensitivity_basic(self):
        """Test basic sensitivity calculation"""
        x = np.array([[1], [2]])
        options = {}
        
        S, y = self.nn.calcSensitivity(x, options=options)
        
        # Should return sensitivity matrix and output
        assert isinstance(S, np.ndarray)
        assert isinstance(y, np.ndarray)
        
        # Check shapes
        assert S.ndim == 3  # (output_dim, output_dim, batch_size)
        assert y.shape[0] == 2  # Output dimension
    
    def test_calcSensitivity_with_store_sensitivity_true(self):
        """Test sensitivity calculation with store_sensitivity=True"""
        x = np.array([[1], [2]])
        options = {}
        
        S, y = self.nn.calcSensitivity(x, options=options, store_sensitivity=True)
        
        # Check that sensitivity was stored in layers
        for layer in self.nn.layers:
            assert hasattr(layer, 'sensitivity')
            assert layer.sensitivity is not None
    
    def test_calcSensitivity_with_store_sensitivity_false(self):
        """Test sensitivity calculation with store_sensitivity=False"""
        x = np.array([[1], [2]])
        options = {}
        
        S, y = self.nn.calcSensitivity(x, options=options, store_sensitivity=False)
        
        # Check that sensitivity was not stored in layers
        for layer in self.nn.layers:
            if hasattr(layer, 'sensitivity'):
                assert layer.sensitivity is None
    
    def test_calcSensitivity_with_options(self):
        """Test sensitivity calculation with various options"""
        x = np.array([[1], [2]])
        options = {
            'nn': {
                'train': {}
            }
        }
        
        S, y = self.nn.calcSensitivity(x, options=options)
        
        # Should work with options
        assert isinstance(S, np.ndarray)
        assert isinstance(y, np.ndarray)
    
    def test_calcSensitivity_none_options(self):
        """Test sensitivity calculation with None options"""
        x = np.array([[1], [2]])
        
        S, y = self.nn.calcSensitivity(x, options=None)
        
        # Should work with default options
        assert isinstance(S, np.ndarray)
        assert isinstance(y, np.ndarray)
    
    def test_calcSensitivity_empty_network(self):
        """Test sensitivity calculation with empty network"""
        empty_nn = NeuralNetwork([])
        x = np.array([[1], [2]])
        
        S, y = empty_nn.calcSensitivity(x)
        
        # Should handle empty network gracefully
        assert isinstance(S, np.ndarray)
        assert isinstance(y, np.ndarray)
    
    def test_calcSensitivity_single_layer(self):
        """Test sensitivity calculation with single layer"""
        W = np.array([[1, 2], [3, 4]])
        b = np.array([[0], [0]])
        layer = nnLinearLayer(W, b)
        
        single_layer_nn = NeuralNetwork([layer])
        x = np.array([[1], [2]])
        
        S, y = single_layer_nn.calcSensitivity(x)
        
        # Should work with single layer
        assert isinstance(S, np.ndarray)
        assert isinstance(y, np.ndarray)
    
    def test_calcSensitivity_batch_input(self):
        """Test sensitivity calculation with batch input"""
        x = np.array([[1, 3], [2, 4]])  # 2 inputs, batch size 2
        options = {}
        
        S, y = self.nn.calcSensitivity(x, options=options)
        
        # Should handle batch input
        assert isinstance(S, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert y.shape[1] == 2  # Batch size should be preserved
