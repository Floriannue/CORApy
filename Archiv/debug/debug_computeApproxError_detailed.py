#!/usr/bin/env python3
"""
Detailed debug script to trace computeApproxError method step by step
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

print("=== Detailed computeApproxError Debug ===")
print(f"Initial coeffs: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")

# Let's manually trace through the computeApproxError method
print("\nStep 1: Get derivative bounds")
df_l, df_u = layer.getDerBounds(l, u)
print(f"df_l: {df_l:.8f}, df_u: {df_u:.8f}")

print("\nStep 2: Call minMaxDiffOrder")
diffl, diffu = minMaxDiffOrder(coeffs, l, u, layer.f, df_l, df_u)
print(f"diffl: {diffl:.8f}, diffu: {diffu:.8f}")

print("\nStep 3: Calculate diffc")
diffc = (diffl + diffu) / 2
print(f"diffc: {diffc:.8f}")

print("\nStep 4: Modify coefficients")
coeffs_copy = coeffs.copy()
print(f"Before modification: [{coeffs_copy[0]:.8f}, {coeffs_copy[1]:.8f}, {coeffs_copy[2]:.8f}]")
coeffs_copy[-1] = coeffs_copy[-1] + diffc
print(f"After modification: [{coeffs_copy[0]:.8f}, {coeffs_copy[1]:.8f}, {coeffs_copy[2]:.8f}]")

print("\nStep 5: Calculate d")
d = diffu - diffc
print(f"d: {d:.8f}")

print("\nNow let's call the actual method and compare:")
coeffs_error, d_error = layer.computeApproxError(l, u, coeffs)
print(f"Method coeffs: [{coeffs_error[0]:.8f}, {coeffs_error[1]:.8f}, {coeffs_error[2]:.8f}]")
print(f"Method d: {d_error:.8f}")

print(f"\nComparison:")
print(f"Manual coeffs: [{coeffs_copy[0]:.8f}, {coeffs_copy[1]:.8f}, {coeffs_copy[2]:.8f}]")
print(f"Method coeffs: [{coeffs_error[0]:.8f}, {coeffs_error[1]:.8f}, {coeffs_error[2]:.8f}]")
print(f"Manual d: {d:.8f}")
print(f"Method d: {d_error:.8f}")

# Let's also check if the method is calling minMaxDiffOrder multiple times
print(f"\nLet's check if minMaxDiffOrder is called multiple times...")
print(f"Initial coeffs (should be unchanged): [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
