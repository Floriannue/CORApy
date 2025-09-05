#!/usr/bin/env python3
"""
Debug script to compare weight loading between Python and MATLAB
"""

import numpy as np
import sys
import os

# Add the cora_python directory to the path
sys.path.insert(0, 'cora_python')

from nn.neuralNetwork.readONNXNetwork import readONNXNetwork

def debug_weight_comparison():
    print("=== DEBUGGING WEIGHT COMPARISON ===")
    
    # Load the Python network
    model_path = "cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx"
    print(f"Loading Python network from: {model_path}")
    
    nn = readONNXNetwork(model_path, True, 'BSSC')
    
    print(f"\nPython Network Properties:")
    print(f"  neurons_in: {nn.neurons_in}")
    print(f"  neurons_out: {nn.neurons_out}")
    print(f"  Number of layers: {len(nn.layers)}")
    
    # Check the first few linear layers
    linear_layers = []
    for i, layer in enumerate(nn.layers):
        if hasattr(layer, 'W') and hasattr(layer, 'b'):
            linear_layers.append((i, layer))
            print(f"\nLayer {i} (Linear):")
            print(f"  W shape: {layer.W.shape}")
            print(f"  b shape: {layer.b.shape}")
            print(f"  W first row: {layer.W[0, :5]}")
            print(f"  b first values: {layer.b.flatten()[:5]}")
            
            if len(linear_layers) >= 3:  # Only check first 3 linear layers
                break
    
    # Test with the MATLAB counterexample input
    print(f"\n=== TESTING WITH MATLAB COUNTEREXAMPLE ===")
    matlab_x = np.array([[0.679858], [0.100000], [0.500000], [0.500000], [-0.450000]], dtype=np.float32)
    print(f"Input: {matlab_x.flatten()}")
    
    # Test first linear layer (Layer 2) manually
    if len(linear_layers) >= 1:
        layer_idx, first_linear = linear_layers[0]
        print(f"\nTesting Layer {layer_idx} manually:")
        
        # Get input to this layer (after reshape)
        layer_input = matlab_x  # Should be same after elementwise + reshape
        for i in range(layer_idx):
            layer_input = nn.layers[i].evaluateNumeric(layer_input, {})
        
        print(f"  Layer input: {layer_input.flatten()}")
        
        # Manual calculation
        manual_output = first_linear.W @ layer_input + first_linear.b
        print(f"  Manual W @ x + b: {manual_output.flatten()}")
        
        # Layer calculation
        layer_output = first_linear.evaluateNumeric(layer_input, {})
        print(f"  Layer output: {layer_output.flatten()}")
        
        # Check if they match
        if np.allclose(manual_output, layer_output):
            print("  ✓ Manual and layer calculations match")
        else:
            print("  ✗ Manual and layer calculations differ!")
            print(f"    Difference: {(manual_output - layer_output).flatten()}")

if __name__ == "__main__":
    debug_weight_comparison()
