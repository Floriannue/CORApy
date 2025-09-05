#!/usr/bin/env python3
"""
Debug script to test if .copy() is working properly
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

print("=== Copy Test Debug ===")
print(f"Initial coeffs: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"Initial coeffs id: {id(coeffs)}")

# Test .copy() method
coeffs_copy = coeffs.copy()
print(f"After .copy() - original: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"After .copy() - copy: [{coeffs_copy[0]:.8f}, {coeffs_copy[1]:.8f}, {coeffs_copy[2]:.8f}]")
print(f"After .copy() - original id: {id(coeffs)}")
print(f"After .copy() - copy id: {id(coeffs_copy)}")

# Modify the copy
coeffs_copy[-1] = coeffs_copy[-1] + 0.1
print(f"After modifying copy - original: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"After modifying copy - copy: [{coeffs_copy[0]:.8f}, {coeffs_copy[1]:.8f}, {coeffs_copy[2]:.8f}]")

# Now let's test the computeApproxError method
print(f"\n=== computeApproxError Test ===")
coeffs_test = coeffs.copy()
print(f"Before computeApproxError - original: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"Before computeApproxError - test: [{coeffs_test[0]:.8f}, {coeffs_test[1]:.8f}, {coeffs_test[2]:.8f}]")

coeffs_error, d_error = layer.computeApproxError(l, u, coeffs_test)
print(f"After computeApproxError - original: [{coeffs[0]:.8f}, {coeffs[1]:.8f}, {coeffs[2]:.8f}]")
print(f"After computeApproxError - test: [{coeffs_test[0]:.8f}, {coeffs_test[1]:.8f}, {coeffs_test[2]:.8f}]")
print(f"After computeApproxError - returned: [{coeffs_error[0]:.8f}, {coeffs_error[1]:.8f}, {coeffs_error[2]:.8f}]")
print(f"After computeApproxError - test id: {id(coeffs_test)}")
print(f"After computeApproxError - returned id: {id(coeffs_error)}")
