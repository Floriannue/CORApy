#!/usr/bin/env python3
"""
Debug script to find exactly when coefficients are being modified
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

print("=== Coefficient Modification Debug ===")
print(f"Initial coeffs: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"Initial coeffs id: {id(coeffs)}")

# Let's check if getDerBounds modifies coeffs
print("\nStep 1: Call getDerBounds")
df_l, df_u = layer.getDerBounds(l, u)
print(f"After getDerBounds: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"coeffs id: {id(coeffs)}")

# Let's check if minMaxDiffOrder modifies coeffs
print("\nStep 2: Call minMaxDiffOrder")
from nn.nnHelper.minMaxDiffOrder import minMaxDiffOrder
diffl, diffu = minMaxDiffOrder(coeffs, l, u, layer.f, df_l, df_u)
print(f"After minMaxDiffOrder: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"coeffs id: {id(coeffs)}")

# Now let's call computeApproxError and see what happens
print("\nStep 3: Call computeApproxError")
coeffs_error, d_error = layer.computeApproxError(l, u, coeffs)
print(f"After computeApproxError: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"coeffs id: {id(coeffs)}")
print(f"Returned coeffs: [{coeffs_error[0]:.8f}, {coeffs_error[1]:.8f}, {coeffs_error[2]:.8f}]")
print(f"Returned coeffs id: {id(coeffs_error)}")

# Let's also check if the issue is in the computeApproxPoly method
print("\nStep 4: Call computeApproxPoly")
coeffs_poly, d_poly = layer.computeApproxPoly(l, u, order, 'regression')
print(f"After computeApproxPoly: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"coeffs id: {id(coeffs)}")
print(f"Returned coeffs: [{coeffs_poly[0]:.8f}, {coeffs_poly[1]:.8f}, {coeffs_poly[2]:.8f}]")
print(f"Returned coeffs id: {id(coeffs_poly)}")
