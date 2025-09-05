#!/usr/bin/env python3
"""
Debug script to test minMaxDiffPoly function
"""

import sys
sys.path.append('cora_python')
from nn.layers.nonlinear.nnReLULayer import nnReLULayer
from nn.nnHelper.leastSquarePolyFunc import leastSquarePolyFunc
from nn.nnHelper.minMaxDiffPoly import minMaxDiffPoly
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

print("=== minMaxDiffPoly Debug ===")
print(f"l: {l}, u: {u}")
print(f"coeffs: {coeffs}")

# Test the minMaxDiffPoly function like MATLAB exactly
# x < 0: p(x) - alpha*x
# MATLAB: minMaxDiffPoly(coeffs,[obj.alpha,0],l,0)
coeffs1 = coeffs  # Original coefficients
coeffs2 = np.array([layer.alpha, 0])  # [alpha, 0] - no padding
print(f"coeffs1: {coeffs1}")
print(f"coeffs2: {coeffs2}")

diffl1, diffu1 = minMaxDiffPoly(coeffs1, coeffs2, l, 0)
print(f"diffl1: {diffl1}, diffu1: {diffu1}")

# x > 0: p(x) - 1*x
# MATLAB: minMaxDiffPoly(coeffs,[1,0],0,u)
coeffs3 = np.array([1, 0])  # [1, 0] - no padding
print(f"coeffs3: {coeffs3}")

diffl2, diffu2 = minMaxDiffPoly(coeffs1, coeffs3, 0, u)
print(f"diffl2: {diffl2}, diffu2: {diffu2}")

# compute final approx error
diffl = min(diffl1, diffl2)
diffu = max(diffu1, diffu2)
diffc = (diffu + diffl) / 2
coeffs_final = coeffs.copy()
coeffs_final[-1] = coeffs_final[-1] - diffc
d_final = diffu - diffc

print(f"diffl: {diffl}, diffu: {diffu}")
print(f"diffc: {diffc}")
print(f"coeffs_final: {coeffs_final}")
print(f"d_final: {d_final}")

# Compare with actual method
print(f"\nActual method result:")
coeffs_actual, d_actual = layer.computeApproxError(l, u, coeffs)
print(f"coeffs_actual: {coeffs_actual}")
print(f"d_actual: {d_actual}")
