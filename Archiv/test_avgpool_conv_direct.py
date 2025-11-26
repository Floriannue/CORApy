"""Test AvgPool convolution directly with known input"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.nn.layers.linear.nnAvgPool2DLayer import nnAvgPool2DLayer

# Create AvgPool layer
poolSize = [4, 4]
layer = nnAvgPool2DLayer(poolSize)

# Set input size to trigger getOutputSize
imgSize = [8, 8, 1]  # Simple 8x8 single channel
outputSize = layer.getOutputSize(imgSize)
layer.inputSize = imgSize  # Set manually for testing

print(f"Input: {imgSize}, Output: {outputSize}")
print(f"W shape: {layer.W.shape}")
print(f"W[:, :, 0, 0]:\n{layer.W[:, :, 0, 0]}")

# Create test input with known pattern
# Input format: [8*8*1, 1] = [64, 1]
# But we need to set it up so first 4x4 window has all 1s
input_flat = np.zeros((64, 1))
# Set first 4x4 window (first 16 values) to 1.0
input_flat[:16] = 1.0
print(f"\nInput (flattened): shape {input_flat.shape}")
print(f"Input first 16 values: {input_flat[:16].flatten()}")

# Evaluate
output = layer.evaluateNumeric(input_flat, {})
print(f"\nOutput shape: {output.shape}")
print(f"Output: {output.flatten()}")

# Expected: first output should be average of first 4x4 window = 1.0
# But if convolution is working, it should be 1.0
# If it's just taking first value, it would also be 1.0
# So let's try a different pattern

# Try pattern where first window has mixed values
input_flat2 = np.zeros((64, 1))
# First 4x4 window: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
input_flat2[0] = 1.0
print(f"\n\nTest 2: Input with single 1.0 in first position")
print(f"Input first 16 values: {input_flat2[:16].flatten()}")

output2 = layer.evaluateNumeric(input_flat2, {})
print(f"Output: {output2.flatten()}")
print(f"Expected first output: 1.0 / 16 = 0.0625")
print(f"Actual first output: {output2[0, 0]}")
print(f"Match? {np.isclose(output2[0, 0], 0.0625)}")

