#!/usr/bin/env python3
"""
Test script to understand the network input size requirements
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

import numpy as np
from cora_python.nn.neuralNetwork import NeuralNetwork

def test_network_input_size():
    """Test what input size the network actually expects"""
    
    print("=== TESTING NETWORK INPUT SIZE ===")
    
    # Load the model
    modelPath = 'cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx'
    print(f"Loading model: {modelPath}")
    
    nn = NeuralNetwork.readONNXNetwork(modelPath, True, 'BSSC')
    
    print(f"\n=== NETWORK PROPERTIES ===")
    print(f"neurons_in: {nn.neurons_in}")
    print(f"neurons_out: {nn.neurons_out}")
    print(f"Number of layers: {len(nn.layers)}")
    
    # Test with different input sizes
    test_sizes = [
        (5, 1),      # 5 inputs
        (50, 1),     # 50 inputs (matches neurons_in)
        (1, 1, 1, 5), # Original ONNX shape
    ]
    
    for test_size in test_sizes:
        print(f"\n=== TESTING WITH INPUT SIZE {test_size} ===")
        try:
            test_input = np.random.rand(*test_size).astype(np.float32)
            print(f"Input shape: {test_input.shape}")
            
            # Try to evaluate the network
            output = nn.evaluate(test_input)
            print(f"SUCCESS: Output shape: {output.shape}")
            
        except Exception as e:
            print(f"FAILED: {e}")
    
    # Check the first layer specifically
    print(f"\n=== FIRST LAYER ANALYSIS ===")
    first_layer = nn.layers[0]
    print(f"First layer type: {type(first_layer).__name__}")
    
    if hasattr(first_layer, 'idx_out'):
        print(f"Reshape idx_out: {first_layer.idx_out}")
        print(f"Reshape idx_out shape: {np.array(first_layer.idx_out).shape}")
    
    # Check the second layer (first linear layer)
    if len(nn.layers) > 1:
        second_layer = nn.layers[1]
        print(f"Second layer type: {type(second_layer).__name__}")
        if hasattr(second_layer, 'W'):
            print(f"Linear layer W shape: {second_layer.W.shape}")
            print(f"Linear layer expects {second_layer.W.shape[1]} inputs")

if __name__ == "__main__":
    test_network_input_size()
