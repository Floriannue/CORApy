"""
Test for neuralNetwork readONNXNetwork method

This test verifies that the readONNXNetwork method works correctly with different ONNX files.
"""

import pytest
import numpy as np
import os

def test_neuralNetwork_readONNXNetwork_basic():
    """Test readONNXNetwork with basic network"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    
    # Skip if file doesn't exist
    modelPath = 'attitude_control_3_64_torch.onnx'
    if not os.path.exists(modelPath):
        pytest.skip("attitude_control_3_64_torch.onnx file not found")
    
    # Test reading basic network
    nn = NeuralNetwork.readONNXNetwork(modelPath)
    
    # Should return a NeuralNetwork object
    assert isinstance(nn, NeuralNetwork)

def test_neuralNetwork_readONNXNetwork_verbose():
    """Test readONNXNetwork with verbose output and input/output formats"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    
    # Skip if file doesn't exist
    modelPath = 'controller_airplane.onnx'
    if not os.path.exists(modelPath):
        pytest.skip("controller_airplane.onnx file not found")
    
    # Test verbose output + input/output formats
    nn = NeuralNetwork.readONNXNetwork(modelPath, True, 'BC', 'BC')
    
    # Should return a NeuralNetwork object
    assert isinstance(nn, NeuralNetwork)

def test_neuralNetwork_readONNXNetwork_custom_layer():
    """Test readONNXNetwork with custom layer"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    
    # Skip if file doesn't exist
    modelPath = 'vnn_verivital_avgpool.onnx'
    if not os.path.exists(modelPath):
        pytest.skip("vnn_verivital_avgpool.onnx file not found")
    
    # Test reading network with custom layer
    nn = NeuralNetwork.readONNXNetwork(modelPath, False, 'BCSS')
    
    # Should return a NeuralNetwork object
    assert isinstance(nn, NeuralNetwork)

def test_neuralNetwork_readONNXNetwork_acasxu():
    """Test readONNXNetwork with ACASXU network"""
    from cora_python.nn.neuralNetwork import NeuralNetwork
    
    # Skip if file doesn't exist
    modelPath = 'ACASXU_run2a_1_2_batch_2000.onnx'
    if not os.path.exists(modelPath):
        pytest.skip("ACASXU_run2a_1_2_batch_2000.onnx file not found")
    
    # Test reading ACASXU network
    nn = NeuralNetwork.readONNXNetwork(modelPath, False, 'BSSC')
    
    # Should return a NeuralNetwork object
    assert isinstance(nn, NeuralNetwork)
    
    # Check that it has layers
    assert hasattr(nn, 'layers')
    assert len(nn.layers) > 0
