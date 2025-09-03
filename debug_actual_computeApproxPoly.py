#!/usr/bin/env python3
"""
Debug script to test the actual computeApproxPoly method
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

print("=== Actual computeApproxPoly Debug ===")
print(f"l: {l}, u: {u}, order: {order}, method: {method}")

# Call the actual method
coeffs, d = layer.computeApproxPoly(l, u, order, method)

print(f"coeffs: {coeffs}")
print(f"d: {d}")

# Test evaluation at some points
x_test = np.linspace(l, u, 5)
y_true = layer.f(x_test)
y_approx = np.polyval(coeffs, x_test)

print(f"x_test: {x_test}")
print(f"y_true: {y_true}")
print(f"y_approx: {y_approx}")
