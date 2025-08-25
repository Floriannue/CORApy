"""
Test for nnLayer abstract base class

This test verifies that the nnLayer class can be imported and basic functionality works.
"""

import pytest
import numpy as np
from unittest.mock import Mock

# Test that we can import the base class
def test_import_nnLayer():
    """Test that nnLayer can be imported"""
    from cora_python.nn.layers.nnLayer import nnLayer
    assert nnLayer is not None

def test_import_neuralNetwork():
    """Test that NeuralNetwork can be imported"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    assert NeuralNetwork is not None

def test_nnLayer_abstract():
    """Test that nnLayer is abstract and cannot be instantiated directly"""
    from cora_python.nn.layers.nnLayer import nnLayer
    
    with pytest.raises(TypeError):
        # Should raise TypeError because nnLayer is abstract
        nnLayer()

def test_neuralNetwork_constructor():
    """Test that NeuralNetwork can be constructed with empty layers"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    
    # Should work with empty layers
    nn = NeuralNetwork()
    assert len(nn.layers) == 0
    assert nn.neurons_in is None
    assert nn.neurons_out is None

def test_neuralNetwork_length():
    """Test that NeuralNetwork length method works"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    from cora_python.nn.layers.nnLayer import nnLayer
    
    nn = NeuralNetwork()
    assert len(nn) == 0
    
    # Create a concrete mock layer that inherits from nnLayer
    class MockLayer(nnLayer):
        def getNumNeurons(self):
            return (2, 3)
        
        def getOutputSize(self, inputSize):
            return [3]
        
        def evaluateNumeric(self, input_data, options):
            return np.zeros((3, input_data.shape[1]))
    
    mock_layer = MockLayer()
    nn2 = NeuralNetwork([mock_layer])
    assert len(nn2) == 1
