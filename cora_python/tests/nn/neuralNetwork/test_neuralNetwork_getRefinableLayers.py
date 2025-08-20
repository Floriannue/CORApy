"""
Test file for NeuralNetwork.getRefinableLayers method

This file tests the getRefinableLayers method of the NeuralNetwork class.
"""

import pytest
import numpy as np
from cora_python.nn.neuralNetwork import NeuralNetwork
from cora_python.nn.layers.linear.nnLinearLayer import nnLinearLayer
from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer

class TestNeuralNetworkGetRefinableLayers:
    """Test class for NeuralNetwork.getRefinableLayers method"""
    
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
    
    def test_getRefinableLayers_basic(self):
        """Test basic getRefinableLayers functionality"""
        refinable_layers = self.nn.getRefinableLayers()
        
        # Should return a list
        assert isinstance(refinable_layers, list)
        
        # Initially no layers should be refinable
        assert len(refinable_layers) == 0
    
    def test_getRefinableLayers_with_refinable_layers(self):
        """Test getRefinableLayers with refinable layers"""
        # Make some layers refinable
        self.nn.layers[0].is_refinable = True
        self.nn.layers[1].is_refinable = True
        
        refinable_layers = self.nn.getRefinableLayers()
        
        # Should return refinable layers
        assert len(refinable_layers) == 2
        assert self.nn.layers[0] in refinable_layers
        assert self.nn.layers[1] in refinable_layers
        assert self.nn.layers[2] not in refinable_layers
    
    def test_getRefinableLayers_mixed_refinable(self):
        """Test getRefinableLayers with mixed refinable status"""
        # Make only one layer refinable
        self.nn.layers[0].is_refinable = True
        self.nn.layers[1].is_refinable = False
        self.nn.layers[2].is_refinable = True
        
        refinable_layers = self.nn.getRefinableLayers()
        
        # Should return only refinable layers
        assert len(refinable_layers) == 2
        assert self.nn.layers[0] in refinable_layers
        assert self.nn.layers[1] not in refinable_layers
        assert self.nn.layers[2] in refinable_layers
    
    def test_getRefinableLayers_all_refinable(self):
        """Test getRefinableLayers with all layers refinable"""
        # Make all layers refinable
        for layer in self.nn.layers:
            layer.is_refinable = True
        
        refinable_layers = self.nn.getRefinableLayers()
        
        # Should return all layers
        assert len(refinable_layers) == 3
        for layer in self.nn.layers:
            assert layer in refinable_layers
    
    def test_getRefinableLayers_none_refinable(self):
        """Test getRefinableLayers with no refinable layers"""
        # Make no layers refinable
        for layer in self.nn.layers:
            layer.is_refinable = False
        
        refinable_layers = self.nn.getRefinableLayers()
        
        # Should return empty list
        assert len(refinable_layers) == 0
    
    def test_getRefinableLayers_empty_network(self):
        """Test getRefinableLayers with empty network"""
        empty_nn = NeuralNetwork([])
        
        refinable_layers = empty_nn.getRefinableLayers()
        
        # Should return empty list
        assert isinstance(refinable_layers, list)
        assert len(refinable_layers) == 0
    
    def test_getRefinableLayers_single_layer(self):
        """Test getRefinableLayers with single layer"""
        single_layer_nn = NeuralNetwork([self.nn.layers[0]])
        
        # Make layer refinable
        single_layer_nn.layers[0].is_refinable = True
        
        refinable_layers = single_layer_nn.getRefinableLayers()
        
        # Should return the single layer
        assert len(refinable_layers) == 1
        assert single_layer_nn.layers[0] in refinable_layers
    
    def test_getRefinableLayers_missing_attribute(self):
        """Test getRefinableLayers with layers missing is_refinable attribute"""
        # Remove is_refinable attribute from some layers
        delattr(self.nn.layers[0], 'is_refinable')
        self.nn.layers[1].is_refinable = True
        delattr(self.nn.layers[2], 'is_refinable')
        
        refinable_layers = self.nn.getRefinableLayers()
        
        # Should only return layers with is_refinable=True
        assert len(refinable_layers) == 1
        assert self.nn.layers[1] in refinable_layers
    
    def test_getRefinableLayers_false_values(self):
        """Test getRefinableLayers with various false values"""
        # Test different false values
        self.nn.layers[0].is_refinable = False
        self.nn.layers[1].is_refinable = 0
        self.nn.layers[2].is_refinable = None
        
        refinable_layers = self.nn.getRefinableLayers()
        
        # Should return empty list (all values are falsy)
        assert len(refinable_layers) == 0
    
    def test_getRefinableLayers_true_values(self):
        """Test getRefinableLayers with various true values"""
        # Test different true values
        self.nn.layers[0].is_refinable = True
        self.nn.layers[1].is_refinable = 1
        self.nn.layers[2].is_refinable = "any_string"
        
        refinable_layers = self.nn.getRefinableLayers()
        
        # Should return all layers (all values are truthy)
        assert len(refinable_layers) == 3
