#!/usr/bin/env python3
"""
Debug script to test the computeApproxError method signature
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

print("=== Method Signature Debug ===")
print(f"Initial coeffs: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"Initial coeffs id: {id(coeffs)}")

# Let's manually implement the computeApproxError method
print("\nManually implementing computeApproxError...")

# Make a copy
coeffs_manual = coeffs.copy()
print(f"After copy - original: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"After copy - manual: [{coeffs_manual[0]:.8f}, {coeffs_manual[1]:.8f}, {coeffs_manual[2]:.8f}]")
print(f"After copy - original id: {id(coeffs)}")
print(f"After copy - manual id: {id(coeffs_manual)}")

# Get derivative bounds
df_l, df_u = layer.getDerBounds(l, u)

# Call minMaxDiffOrder
from nn.nnHelper.minMaxDiffOrder import minMaxDiffOrder
diffl, diffu = minMaxDiffOrder(coeffs_manual, l, u, layer.f, df_l, df_u)

# Calculate diffc and modify
diffc = (diffl + diffu) / 2
coeffs_manual[-1] = coeffs_manual[-1] + diffc
d_manual = diffu - diffc

print(f"After manual computation - original: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"After manual computation - manual: [{coeffs_manual[0]:.8f}, {coeffs_manual[1]:.8f}, {coeffs_manual[2]:.8f}]")
print(f"After manual computation - original id: {id(coeffs)}")
print(f"After manual computation - manual id: {id(coeffs_manual)}")

# Now let's call the actual method
print(f"\nCalling actual computeApproxError...")
coeffs_error, d_error = layer.computeApproxError(l, u, coeffs)
print(f"After actual method - original: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"After actual method - returned: [{coeffs_error[0]:.8f}, {coeffs_error[1]:.8f}, {coeffs_error[2]:.8f}]")
print(f"After actual method - original id: {id(coeffs)}")
print(f"After actual method - returned id: {id(coeffs_error)}")

print(f"\nComparison:")
print(f"Manual coeffs: [{coeffs_manual[0]:.8f}, {coeffs_manual[1]:.8f}, {coeffs_manual[2]:.8f}]")
print(f"Actual coeffs: [{coeffs_error[0]:.8f}, {coeffs_error[1]:.8f}, {coeffs_error[2]:.8f}]")
print(f"Manual d: {d_manual}")
print(f"Actual d: {d_error}")
print(f"Match: {np.allclose(coeffs_manual, coeffs_error) and np.isclose(d_manual, d_error)}")
