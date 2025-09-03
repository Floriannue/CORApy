#!/usr/bin/env python3
"""
Debug script to find the exact line where coefficients are modified
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

# Get the coefficients from regression
num_points = 10 * (order + 1)
x = np.linspace(l, u, num_points)
y = layer.f(x)
coeffs = leastSquarePolyFunc(x, y, order)

print("=== Exact Modification Debug ===")
print(f"Initial coeffs: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"Initial coeffs id: {id(coeffs)}")

# Let's manually trace through the computeApproxError method step by step
print("\nManually tracing computeApproxError...")

# Step 1: Make a copy
coeffs_copy = coeffs.copy()
print(f"After copy: [{coeffs_copy[0]:.8f}, {coeffs_copy[1]:.8f}, {coeffs_copy[2]:.8f}]")
print(f"Copy id: {id(coeffs_copy)}")
print(f"Original still: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")

# Step 2: Get derivative bounds
df_l, df_u = layer.getDerBounds(l, u)
print(f"After getDerBounds - original: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"After getDerBounds - copy: [{coeffs_copy[0]:.8f}, {coeffs_copy[1]:.8f}, {coeffs_copy[2]:.8f}]")

# Step 3: Call minMaxDiffOrder
from nn.nnHelper.minMaxDiffOrder import minMaxDiffOrder
diffl, diffu = minMaxDiffOrder(coeffs_copy, l, u, layer.f, df_l, df_u)
print(f"After minMaxDiffOrder - original: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"After minMaxDiffOrder - copy: [{coeffs_copy[0]:.8f}, {coeffs_copy[1]:.8f}, {coeffs_copy[2]:.8f}]")

# Step 4: Calculate diffc and modify
diffc = (diffl + diffu) / 2
coeffs_copy[-1] = coeffs_copy[-1] + diffc
print(f"After modification - original: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"After modification - copy: [{coeffs_copy[0]:.8f}, {coeffs_copy[1]:.8f}, {coeffs_copy[2]:.8f}]")

# Now let's call the actual method
print(f"\nCalling actual computeApproxError...")
coeffs_error, d_error = layer.computeApproxError(l, u, coeffs)
print(f"After computeApproxError - original: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"Returned: [{coeffs_error[0]:.8f}, {coeffs_error[1]:.8f}, {coeffs_error[2]:.8f}]")
print(f"Returned id: {id(coeffs_error)}")
