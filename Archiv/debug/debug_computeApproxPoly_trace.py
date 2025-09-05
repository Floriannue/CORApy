#!/usr/bin/env python3
"""
Debug script to trace through computeApproxPoly method step by step
"""

import sys
sys.path.append('cora_python')
from nn.layers.nonlinear.nnReLULayer import nnReLULayer
from nn.nnHelper.leastSquarePolyFunc import leastSquarePolyFunc
import numpy as np

# Create ReLU layer
layer = nnReLULayer()

# Test parameters
l, u = -1, 1
order = 2
method = 'regression'

print("=== computeApproxPoly Trace Debug ===")

# Get the coefficients from regression
num_points = 10 * (order + 1)
x = np.linspace(l, u, num_points)
y = layer.f(x)
coeffs = leastSquarePolyFunc(x, y, order)

print(f"Initial coeffs from leastSquarePolyFunc: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"Initial coeffs id: {id(coeffs)}")

# Now let's manually trace through computeApproxPoly
print("\nManually tracing computeApproxPoly...")

# Initialize
coeffs_manual = coeffs.copy()
d_manual = []

print(f"After initialization - coeffs: [{coeffs_manual[0]:.8f}, {coeffs_manual[1]:.8f}, {coeffs_manual[2]:.8f}]")
print(f"After initialization - d: {d_manual}")

# Check if d is empty
if len(d_manual) == 0:
    print("d is empty, calling computeApproxError...")
    coeffs_manual, d_manual = layer.computeApproxError(l, u, coeffs_manual)
    print(f"After computeApproxError - coeffs: [{coeffs_manual[0]:.8f}, {coeffs_manual[1]:.8f}, {coeffs_manual[2]:.8f}]")
    print(f"After computeApproxError - d: {d_manual}")

print(f"\nOriginal coeffs after manual trace: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")

# Now let's call the actual method
print(f"\nCalling actual computeApproxPoly...")
coeffs_actual, d_actual = layer.computeApproxPoly(l, u, order, method)
print(f"After computeApproxPoly - original coeffs: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"After computeApproxPoly - returned coeffs: [{coeffs_actual[0]:.8f}, {coeffs_actual[1]:.8f}, {coeffs_actual[2]:.8f}]")
print(f"After computeApproxPoly - returned d: {d_actual}")

print(f"\nComparison:")
print(f"Manual coeffs: [{coeffs_manual[0]:.8f}, {coeffs_manual[1]:.8f}, {coeffs_manual[2]:.8f}]")
print(f"Actual coeffs: [{coeffs_actual[0]:.8f}, {coeffs_actual[1]:.8f}, {coeffs_actual[2]:.8f}]")
print(f"Manual d: {d_manual}")
print(f"Actual d: {d_actual}")
