"""
Debug script to compare weight matrix construction between Python and MATLAB
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from cora_python.nn.layers.linear.nnConv2DLayer import nnConv2DLayer

# Create the same layer
W = np.zeros((2, 2, 1, 2), dtype=np.float64)
W[:, :, 0, 0] = np.array([[1, -1], [-1, 2]])
W[:, :, 0, 1] = np.array([[2, 3], [-1, -2]])
b = np.array([1.0, -2.0])

layer = nnConv2DLayer(W, b)
layer.inputSize = [4, 4, 1]

# Get the weight matrix
Wff = layer.aux_conv2Mat()

print("Python Wff (first 5 rows):")
for i in range(min(5, Wff.shape[0])):
    print(f"  Row {i}: {Wff[i, :]}")
print()

print("MATLAB Wff (first 5 rows, from matlab_zonotope_io_pairs.txt):")
print("  Row 0: [1.000000 -1.000000 0.000000 0.000000 -1.000000 2.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000]")
print("  Row 1: [0.000000 1.000000 -1.000000 0.000000 0.000000 -1.000000 2.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000]")
print("  Row 2: [0.000000 0.000000 1.000000 -1.000000 0.000000 0.000000 -1.000000 2.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000]")
print("  Row 3: [0.000000 0.000000 0.000000 0.000000 1.000000 -1.000000 0.000000 0.000000 -1.000000 2.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000]")
print("  Row 4: [0.000000 0.000000 0.000000 0.000000 0.000000 1.000000 -1.000000 0.000000 0.000000 -1.000000 2.000000 0.000000 0.000000 0.000000 0.000000 0.000000]")
print()

# Check linFfilter construction
k_h, k_w, in_c, out_c = W.shape
Filter = W

# Python construction
Filter_transposed = np.transpose(Filter, [1, 0, 2, 3])
Filter_reshaped = Filter_transposed.reshape(k_h * k_w, in_c, out_c)
linFfilter = np.concatenate([
    np.zeros((1, in_c, out_c), dtype=Filter.dtype),
    Filter_reshaped
], axis=0)
linFfilter_flat = np.transpose(linFfilter, [1, 2, 0]).flatten()

print("Python linFfilter (first 10 values):", linFfilter_flat[:10])
print("MATLAB linFfilter (first 10 values): [0.000000 1.000000 -1.000000 -1.000000 2.000000 0.000000 2.000000 -1.000000 3.000000 -2.000000]")
print()

# Check if they match
matlab_linFfilter = np.array([0.000000, 1.000000, -1.000000, -1.000000, 2.000000, 0.000000, 2.000000, -1.000000, 3.000000, -2.000000])
print("linFfilter matches MATLAB:", np.allclose(linFfilter_flat[:10], matlab_linFfilter, atol=1e-6))
print()

# Check WffIdx computation
WffIdx, _ = layer.aux_computeWeightMatIdx('', W, [4, 4, 1], np.array([1, 1]), np.array([0, 0, 0, 0]), np.array([1, 1]))
print("WffIdx (first 5x5):")
print(WffIdx[:5, :5])
print()

# Check what values WffIdx points to in linFfilter
WffIdx_0based = WffIdx - 1
WffIdx_0based = np.clip(WffIdx_0based, 0, len(linFfilter_flat) - 1).astype(np.int64)
Wff_values = linFfilter_flat[WffIdx_0based]
print("Values from linFfilter using WffIdx (first row, first 5):")
print(Wff_values[0, :5])
print("Expected (from MATLAB Wff first row, first 5): [1, -1, 0, 0, -1]")
print()

# Check what the actual Wff first row first 5 values are
print("Actual Wff[0, :5]:", Wff[0, :5])
print("Expected Wff[0, :5]: [1, -1, 0, 0, -1]")

