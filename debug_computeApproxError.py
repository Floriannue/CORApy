#!/usr/bin/env python3
"""
Debug script to trace computeApproxError method step by step
"""

import sys
sys.path.append('cora_python')
from nn.layers.nonlinear.nnReLULayer import nnReLULayer
from nn.nnHelper.leastSquarePolyFunc import leastSquarePolyFunc
from nn.nnHelper.minMaxDiffOrder import minMaxDiffOrder
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

print("=== Python computeApproxError Debug ===")
print(f"l: {l}, u: {u}")
print(f"coeffs: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")

# Test computeApproxError directly
print("\nCalling computeApproxError directly...")
coeffs_error, d_error = layer.computeApproxError(l, u, coeffs)
print(f"coeffs_after_error: [{coeffs_error[0]:.8f}, {coeffs_error[1]:.8f}, {coeffs_error[2]:.8f}]")
print(f"d_after_error: {d_error:.8f}")

# Now let's trace through computeApproxError step by step
print("\nTracing computeApproxError step by step...")

# Get derivative bounds
df_l, df_u = layer.getDerBounds(l, u)
print(f"df_l: {df_l:.8f}, df_u: {df_u:.8f}")

# Call minMaxDiffOrder
diffl, diffu = minMaxDiffOrder(coeffs, l, u, layer.f, df_l, df_u)
print(f"diffl: {diffl:.8f}, diffu: {diffu:.8f}")

# Compute final values (this is what computeApproxError does)
diffc = (diffl + diffu) / 2
coeffs_final = coeffs.copy()
coeffs_final[-1] = coeffs_final[-1] + diffc
d_final = diffu - diffc

print(f"diffc: {diffc:.8f}")
print(f"coeffs_final: [{coeffs_final[0]:.8f}, {coeffs_final[1]:.8f}, {coeffs_final[2]:.8f}]")
print(f"d_final: {d_final:.8f}")

# Compare with direct call
print(f"\nComparison:")
print(f"Direct call coeffs: [{coeffs_error[0]:.8f}, {coeffs_error[1]:.8f}, {coeffs_error[2]:.8f}]")
print(f"Manual computation coeffs: [{coeffs_final[0]:.8f}, {coeffs_final[1]:.8f}, {coeffs_final[2]:.8f}]")
print(f"Direct call d: {d_error:.8f}")
print(f"Manual computation d: {d_final:.8f}")
print(f"Match: {np.allclose(coeffs_error, coeffs_final) and np.isclose(d_error, d_final)}")
