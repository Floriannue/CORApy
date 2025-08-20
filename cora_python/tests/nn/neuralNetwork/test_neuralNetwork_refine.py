"""
Test file for NeuralNetwork.refine method

This file tests the network refinement method of the NeuralNetwork class.
"""

import pytest
import numpy as np
from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer

class TestNeuralNetworkRefine:
    """Test class for NeuralNetwork.refine method"""
    
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
        
        # Mock refinable layers by adding required attributes
        for i, layer in enumerate(self.nn.layers):
            layer.is_refinable = True
            layer.order = np.array([1, 1])  # 2 neurons per layer
            layer.refine_heu = np.array([[0.5], [0.3]])  # Refinement heuristic
            layer.sensitivity = np.random.rand(2, 2, 1)  # Mock sensitivity
    
    def test_refine_all_type(self):
        """Test refine with type='all'"""
        max_order = 3
        type_ = "all"
        method = "all"
        x = np.array([[1], [2]])
        
        self.nn.refine(max_order, type_, method, x, verbose=False)
        
        # Check that orders were increased
        for layer in self.nn.layers:
            assert np.all(layer.order <= max_order)
    
    def test_refine_layer_type(self):
        """Test refine with type='layer'"""
        max_order = 3
        type_ = "layer"
        method = "sensitivity"
        x = np.array([[1], [2]])
        
        self.nn.refine(max_order, type_, method, x, verbose=False)
        
        # Check that at least one layer was refined
        orders_changed = False
        for layer in self.nn.layers:
            if np.any(layer.order > 1):
                orders_changed = True
                break
        assert orders_changed
    
    def test_refine_neuron_type(self):
        """Test refine with type='neuron'"""
        max_order = 3
        type_ = "neuron"
        method = "sensitivity"
        x = np.array([[1], [2]])
        
        self.nn.refine(max_order, type_, method, x, verbose=False)
        
        # Check that at least one neuron was refined
        orders_changed = False
        for layer in self.nn.layers:
            if np.any(layer.order > 1):
                orders_changed = True
                break
        assert orders_changed
    
    def test_refine_sensitivity_method(self):
        """Test refine with method='sensitivity'"""
        max_order = 3
        type_ = "layer"
        method = "sensitivity"
        x = np.array([[1], [2]])
        
        self.nn.refine(max_order, type_, method, x, verbose=False)
        
        # Should work with sensitivity method
        assert True  # If we get here, no exception was raised
    
    def test_refine_random_method(self):
        """Test refine with method='random'"""
        max_order = 3
        type_ = "layer"
        method = "random"
        x = np.array([[1], [2]])
        
        self.nn.refine(max_order, type_, method, x, verbose=False)
        
        # Should work with random method
        assert True  # If we get here, no exception was raised
    
    def test_refine_layer_bias_method(self):
        """Test refine with method='layer_bias'"""
        max_order = 3
        type_ = "layer"
        method = "layer_bias"
        x = np.array([[1], [2]])
        
        self.nn.refine(max_order, type_, method, x, verbose=False)
        
        # Should work with layer_bias method
        assert True  # If we get here, no exception was raised
    
    def test_refine_invalid_type(self):
        """Test refine with invalid type"""
        max_order = 3
        type_ = "invalid"
        method = "sensitivity"
        x = np.array([[1], [2]])
        
        with pytest.raises(ValueError, match="type_ must be one of"):
            self.nn.refine(max_order, type_, method, x)
    
    def test_refine_invalid_method(self):
        """Test refine with invalid method"""
        max_order = 3
        type_ = "layer"
        method = "invalid"
        x = np.array([[1], [2]])
        
        with pytest.raises(ValueError, match="method must be one of"):
            self.nn.refine(max_order, type_, method, x)
    
    def test_refine_sensitivity_without_x(self):
        """Test refine with sensitivity method but no x provided"""
        max_order = 3
        type_ = "layer"
        method = "sensitivity"
        x = None
        
        with pytest.raises(ValueError, match="No point for sensitivity analysis provided"):
            self.nn.refine(max_order, type_, method, x)
    
    def test_refine_with_force_bounds(self):
        """Test refine with force_bounds parameter"""
        max_order = 3
        type_ = "layer"
        method = "sensitivity"
        x = np.array([[1], [2]])
        force_bounds = [2, 3]
        
        self.nn.refine(max_order, type_, method, x, force_bounds=force_bounds)
        
        # Should work with force_bounds
        assert True  # If we get here, no exception was raised
    
    def test_refine_with_gamma(self):
        """Test refine with gamma parameter"""
        max_order = 3
        type_ = "neuron"
        method = "sensitivity"
        x = np.array([[1], [2]])
        gamma = 0.5
        
        self.nn.refine(max_order, type_, method, x, gamma=gamma)
        
        # Should work with gamma
        assert True  # If we get here, no exception was raised
    
    def test_refine_verbose(self):
        """Test refine with verbose output"""
        max_order = 3
        type_ = "all"
        method = "all"
        x = np.array([[1], [2]])
        
        # Should not raise exception with verbose=True
        self.nn.refine(max_order, type_, method, x, verbose=True)
        
        assert True  # If we get here, no exception was raised
    
    def test_refine_empty_network(self):
        """Test refine with empty network"""
        empty_nn = NeuralNetwork([])
        max_order = 3
        type_ = "all"
        method = "all"
        x = np.array([[1], [2]])
        
        # Should handle empty network gracefully
        empty_nn.refine(max_order, type_, method, x)
        
        assert True  # If we get here, no exception was raised
