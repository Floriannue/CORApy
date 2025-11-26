"""Simple test of AvgPool convolution"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.nn.layers.linear.nnAvgPool2DLayer import nnAvgPool2DLayer

# Create AvgPool layer
poolSize = [4, 4]
layer = nnAvgPool2DLayer(poolSize)

# Set input size
imgSize = [8, 8, 1]  # Simple 8x8 single channel
outputSize = layer.getOutputSize(imgSize)
print(f"Input: {imgSize}, Output: {outputSize}")

# Create simple test input: all 1s
# Input format for conv2d: [features, batch] = [8*8*1, 1] = [64, 1]
input_flat = np.ones((64, 1))
print(f"\nInput (flattened): shape {input_flat.shape}, first 10: {input_flat[:10].flatten()}")

# Check weights
print(f"\nW shape: {layer.W.shape}")
print(f"W[:, :, 0, 0]:\n{layer.W[:, :, 0, 0]}")

# Evaluate
output = layer.evaluateNumeric(input_flat, {})
print(f"\nOutput shape: {output.shape}")
print(f"Output: {output.flatten()}")

# Expected: if input is all 1s, output should be all 1s (average of 1s is 1)
# Output should be [2, 2, 1] = 4 values, all = 1.0
expected_output = np.ones((4, 1))
print(f"Expected output: {expected_output.flatten()}")
print(f"Match? {np.allclose(output, expected_output)}")

