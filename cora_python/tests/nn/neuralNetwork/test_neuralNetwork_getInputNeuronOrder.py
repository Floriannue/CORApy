"""
Test file for NeuralNetwork.getInputNeuronOrder method

This file tests the getInputNeuronOrder method of the NeuralNetwork class.
"""

import pytest
import numpy as np
from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer

class TestNeuralNetworkGetInputNeuronOrder:
    """Test class for NeuralNetwork.getInputNeuronOrder method"""
    
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
    
    def test_getInputNeuronOrder_basic(self):
        """Test basic getInputNeuronOrder functionality"""
        x = np.array([[1], [2]])
        inputSize = [2, 1, 1]
        
        order = self.nn.getInputNeuronOrder('sensitivity', x, inputSize)
        
        # Should return a list
        assert isinstance(order, list)
        assert len(order) == 2  # 2 input features
    
    def test_getInputNeuronOrder_sensitivity_method(self):
        """Test getInputNeuronOrder with sensitivity method"""
        x = np.array([[1], [2]])
        inputSize = [2, 1, 1]
        
        order = self.nn.getInputNeuronOrder('sensitivity', x, inputSize)
        
        # Should work with sensitivity method
        assert isinstance(order, list)
        assert len(order) == 2
    
    def test_getInputNeuronOrder_random_method(self):
        """Test getInputNeuronOrder with random method"""
        x = np.array([[1], [2]])
        inputSize = [2, 1, 1]
        
        order = self.nn.getInputNeuronOrder('random', x, inputSize)
        
        # Should work with random method
        assert isinstance(order, list)
        assert len(order) == 2
    
    def test_getInputNeuronOrder_custom_method(self):
        """Test getInputNeuronOrder with custom method"""
        x = np.array([[1], [2]])
        inputSize = [2, 1, 1]
        
        order = self.nn.getInputNeuronOrder('custom', x, inputSize)
        
        # Should work with custom method
        assert isinstance(order, list)
        assert len(order) == 2
    
    def test_getInputNeuronOrder_none_x(self):
        """Test getInputNeuronOrder with None x (should use default)"""
        inputSize = [2, 1, 1]
        
        order = self.nn.getInputNeuronOrder('sensitivity', None, inputSize)
        
        # Should work with default x
        assert isinstance(order, list)
        assert len(order) == 2
    
    def test_getInputNeuronOrder_none_inputSize(self):
        """Test getInputNeuronOrder with None inputSize (should use default)"""
        x = np.array([[1], [2]])
        
        order = self.nn.getInputNeuronOrder('sensitivity', x, None)
        
        # Should work with default inputSize
        assert isinstance(order, list)
    
    def test_getInputNeuronOrder_empty_network(self):
        """Test getInputNeuronOrder with empty network"""
        empty_nn = NeuralNetwork([])
        x = np.array([[1], [2]])
        inputSize = [2, 1, 1]
        
        order = empty_nn.getInputNeuronOrder('sensitivity', x, inputSize)
        
        # Should handle empty network gracefully
        assert isinstance(order, list)
    
    def test_getInputNeuronOrder_single_layer(self):
        """Test getInputNeuronOrder with single layer"""
        single_layer_nn = NeuralNetwork([self.nn.layers[0]])
        x = np.array([[1], [2]])
        inputSize = [2, 1, 1]
        
        order = single_layer_nn.getInputNeuronOrder('sensitivity', x, inputSize)
        
        # Should work with single layer
        assert isinstance(order, list)
    
    def test_getInputNeuronOrder_different_input_sizes(self):
        """Test getInputNeuronOrder with different input sizes"""
        x = np.array([[1], [2]])
        
        # Test different input sizes
        inputSizes = [
            [2, 1, 1],
            [3, 1, 1],
            [2, 2, 1],
            [1, 1, 1]
        ]
        
        for inputSize in inputSizes:
            order = self.nn.getInputNeuronOrder('sensitivity', x, inputSize)
            assert isinstance(order, list)
            assert len(order) == inputSize[0]
    
    def test_getInputNeuronOrder_batch_input(self):
        """Test getInputNeuronOrder with batch input"""
        x = np.array([[1, 3], [2, 4]])  # 2 inputs, batch size 2
        inputSize = [2, 1, 1]
        
        order = self.nn.getInputNeuronOrder('sensitivity', x, inputSize)
        
        # Should handle batch input
        assert isinstance(order, list)
        assert len(order) == 2
    
    def test_getInputNeuronOrder_method_validation(self):
        """Test getInputNeuronOrder method validation"""
        x = np.array([[1], [2]])
        inputSize = [2, 1, 1]
        
        # Test valid methods
        valid_methods = ['sensitivity', 'random', 'custom']
        for method in valid_methods:
            order = self.nn.getInputNeuronOrder(method, x, inputSize)
            assert isinstance(order, list)
    
    def test_getInputNeuronOrder_input_validation(self):
        """Test getInputNeuronOrder input validation"""
        inputSize = [2, 1, 1]
        
        # Test different input types
        test_inputs = [
            np.array([[1], [2]]),
            np.array([1, 2]),
            np.array([[1, 2], [3, 4]]),
            np.array([[[1], [2]]])
        ]
        
        for x in test_inputs:
            order = self.nn.getInputNeuronOrder('sensitivity', x, inputSize)
            assert isinstance(order, list)
    
    def test_getInputNeuronOrder_consistency(self):
        """Test getInputNeuronOrder consistency for same inputs"""
        x = np.array([[1], [2]])
        inputSize = [2, 1, 1]
        
        # Call multiple times with same parameters
        order1 = self.nn.getInputNeuronOrder('sensitivity', x, inputSize)
        order2 = self.nn.getInputNeuronOrder('sensitivity', x, inputSize)
        
        # Results should be consistent for deterministic methods
        assert isinstance(order1, list)
        assert isinstance(order2, list)
        assert len(order1) == len(order2)
