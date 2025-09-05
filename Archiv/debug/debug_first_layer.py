#!/usr/bin/env python3
"""
Debug the first ElementwiseAffineLayer to see what preprocessing it's doing
"""

import numpy as np
import sys
import os

# Add the cora_python directory to the path
sys.path.insert(0, 'cora_python')

from nn.neuralNetwork.readONNXNetwork import readONNXNetwork

def debug_first_layer():
    print("=== DEBUGGING FIRST ELEMENTWISE AFFINE LAYER ===")
    
    # Load the Python network
    model_path = "cora_python/examples/nn/models/ACASXU_run2a_1_2_batch_2000.onnx"
    nn = readONNXNetwork(model_path, True, 'BSSC')
    
    # Check the first layer (ElementwiseAffineLayer)
    first_layer = nn.layers[0]
    print(f"First layer type: {type(first_layer).__name__}")
    
    if hasattr(first_layer, 'scale') and hasattr(first_layer, 'offset'):
        print(f"Scale: {first_layer.scale.flatten()}")
        print(f"Offset: {first_layer.offset.flatten()}")
        
        # Test with the MATLAB counterexample input
        matlab_x = np.array([[0.679858], [0.100000], [0.500000], [0.500000], [-0.450000]], dtype=np.float32)
        print(f"\nInput: {matlab_x.flatten()}")
        
        # Apply the transformation manually
        manual_output = first_layer.scale * matlab_x + first_layer.offset
        print(f"Manual scale * x + offset: {manual_output.flatten()}")
        
        # Apply via layer
        layer_output = first_layer.evaluateNumeric(matlab_x, {})
        print(f"Layer output: {layer_output.flatten()}")
        
        # Check if this is just identity (no change)
        if np.allclose(matlab_x, layer_output):
            print("✓ First layer is identity (no preprocessing)")
        else:
            print("✗ First layer applies preprocessing!")
            print(f"  Difference: {(layer_output - matlab_x).flatten()}")
    else:
        print("First layer doesn't have scale/offset attributes")

if __name__ == "__main__":
    debug_first_layer()
