"""Debug script to test AvgPool convolution directly"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.nn.layers.linear.nnAvgPool2DLayer import nnAvgPool2DLayer

# Create AvgPool layer
poolSize = [4, 4]
layer = nnAvgPool2DLayer(poolSize)

# Set input size to trigger getOutputSize
imgSize = [27, 27, 32]
outputSize = layer.getOutputSize(imgSize)
print(f"Input size: {imgSize}")
print(f"Output size: {outputSize}")
print(f"W shape: {layer.W.shape}")
print(f"b shape: {layer.b.shape}")

# Check W structure
print(f"\nW[:, :, 0, 0] (should be all 0.0625):")
print(layer.W[:, :, 0, 0])
print(f"All 0.0625? {np.allclose(layer.W[:, :, 0, 0], 0.0625)}")

print(f"\nW[:, :, 1, 1] (should be all 0.0625):")
print(layer.W[:, :, 1, 1])
print(f"All 0.0625? {np.allclose(layer.W[:, :, 1, 1], 0.0625)}")

print(f"\nW[:, :, 0, 1] (should be all 0):")
print(layer.W[:, :, 0, 1])
print(f"All 0? {np.allclose(layer.W[:, :, 0, 1], 0)}")

# Test with simple input
print(f"\n\nTesting convolution with simple input:")
# Create input: [27, 27, 32, 1] format
# But conv2d expects flattened input [features, batch]
input_flat = np.ones((27 * 27 * 32, 1))
print(f"Input shape (flattened): {input_flat.shape}")

# Evaluate
output = layer.evaluateNumeric(input_flat, {})
print(f"Output shape: {output.shape}")
print(f"Output (first 20): {output[:20].flatten()}")

# Expected: if input is all 1s, and we pool with 4x4 kernel and stride 4,
# each output should be the average of a 4x4 region = 1.0
# But we're getting the same values as input, which suggests pooling isn't working

