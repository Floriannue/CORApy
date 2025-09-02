#!/usr/bin/env python3
"""
Debug script to trace neural network data flow and compare with MATLAB
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

import numpy as np
from cora_python.nn.neuralNetwork import NeuralNetwork

def debug_neural_network_flow():
    """Debug the neural network data flow step by step"""
    
    print("=== DEBUGGING NEURAL NETWORK DATA FLOW ===")
    
    # Load the model
    modelPath = 'cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx'
    print(f"Loading model: {modelPath}")
    
    nn = NeuralNetwork.readONNXNetwork(modelPath, True, 'BSSC')
    
    print(f"\n=== NETWORK PROPERTIES ===")
    print(f"neurons_in: {nn.neurons_in}")
    print(f"neurons_out: {nn.neurons_out}")
    print(f"Number of layers: {len(nn.layers)}")
    
    print(f"\n=== LAYER DETAILS ===")
    for i, layer in enumerate(nn.layers):
        print(f"Layer {i}: {type(layer).__name__}")
        if hasattr(layer, 'getNumNeurons'):
            try:
                nin, nout = layer.getNumNeurons()
                print(f"  Input neurons: {nin}, Output neurons: {nout}")
            except Exception as e:
                print(f"  getNumNeurons failed: {e}")
        
        # Check reshape layer specifically
        if hasattr(layer, 'idx_out'):
            print(f"  Reshape idx_out: {layer.idx_out}")
    
    print(f"\n=== TESTING DATA FLOW ===")
    
    # Create test input with the expected shape
    # The ONNX model expects input with shape (1, 1, 1, 5)
    test_input_4d = np.random.rand(1, 1, 1, 5).astype(np.float32)
    print(f"Test input 4D shape: {test_input_4d.shape}")
    
    # Also test with 2D input (what the example provides)
    test_input_2d = np.random.rand(5, 1).astype(np.float32)
    print(f"Test input 2D shape: {test_input_2d.shape}")
    
    print(f"\n=== TESTING LAYER BY LAYER ===")
    
    # Test each layer individually
    current_data = test_input_2d
    print(f"Starting with input shape: {current_data.shape}")
    
    for i, layer in enumerate(nn.layers):
        print(f"\n--- Layer {i}: {type(layer).__name__} ---")
        print(f"Input shape: {current_data.shape}")
        
        try:
            # Test the layer
            if hasattr(layer, 'evaluateNumeric'):
                output = layer.evaluateNumeric(current_data, {})
                print(f"Output shape: {output.shape}")
                current_data = output
            else:
                print("Layer has no evaluateNumeric method")
        except Exception as e:
            print(f"ERROR in layer {i}: {e}")
            print(f"Layer type: {type(layer)}")
            if hasattr(layer, 'W'):
                print(f"Layer W shape: {layer.W.shape}")
            if hasattr(layer, 'b'):
                print(f"Layer b shape: {layer.b.shape}")
            break
    
    print(f"\n=== COMPARING WITH MATLAB EXPECTATIONS ===")
    print("MATLAB expects:")
    print("- Input size: 5 (from neurons_in)")
    print("- First layer should be reshape layer with idx_out: [1, 2, 3, 4, 5]")
    print("- Second layer should be linear layer expecting 5 inputs")

if __name__ == "__main__":
    debug_neural_network_flow()
