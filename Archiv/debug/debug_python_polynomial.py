#!/usr/bin/env python3
"""
Debug script to test Python polynomial approximation step by step
"""

import sys
sys.path.append('cora_python')
from nn.layers.nonlinear.nnReLULayer import nnReLULayer
import numpy as np

# Create ReLU layer
layer = nnReLULayer()

# Test parameters
l, u = -1, 1
order = 2
method = 'regression'

print("Python Debug Results:")
print(f"  l: {l}, u: {u}")
print(f"  order: {order}, method: {method}")

# Test the method step by step
coeffs, d = layer.computeApproxPoly(l, u, order, method)

print(f"  coeffs: {coeffs}")
print(f"  d: {d}")

# Test evaluation at some points
x_test = np.linspace(l, u, 5)
y_true = layer.f(x_test)
y_approx = np.polyval(coeffs, x_test)

print(f"  x_test: {x_test}")
print(f"  y_true: {y_true}")
print(f"  y_approx: {y_approx}")

# Let's also test the regression method directly
print("\nTesting regression method directly:")
num_points = 10 * (order + 1)
x = np.linspace(l, u, num_points)
y = layer.f(x)

print(f"  num_points: {num_points}")
print(f"  x: {x[:5]}... (first 5 points)")
print(f"  y: {y[:5]}... (first 5 points)")

# Test leastSquarePolyFunc directly
from cora_python.nn.nnHelper.leastSquarePolyFunc import leastSquarePolyFunc
coeffs_direct = leastSquarePolyFunc(x, y, order)
print(f"  coeffs_direct: {coeffs_direct}")

# Test computeApproxError directly
coeffs_error, d_error = layer.computeApproxError(l, u, coeffs_direct)
print(f"  coeffs_after_error: {coeffs_error}")
print(f"  d_after_error: {d_error}")
