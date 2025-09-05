#!/usr/bin/env python3
"""
Compare old vs new implementation to see which matches MATLAB
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

print("=== Old vs New Implementation Comparison ===")
print(f"coeffs: {coeffs}")

# Current implementation (NEW - no padding)
print(f"\n=== NEW Implementation (no padding) ===")
coeffs1 = coeffs
coeffs2 = np.array([layer.alpha, 0])
coeffs3 = np.array([1, 0])

diffl1_new, diffu1_new = minMaxDiffPoly(coeffs1, coeffs2, l, 0)
diffl2_new, diffu2_new = minMaxDiffPoly(coeffs1, coeffs3, 0, u)

diffl_new = min(diffl1_new, diffl2_new)
diffu_new = max(diffu1_new, diffu2_new)
diffc_new = (diffu_new + diffl_new) / 2
coeffs_final_new = coeffs.copy()
coeffs_final_new[-1] = coeffs_final_new[-1] - diffc_new
d_final_new = diffu_new - diffc_new

print(f"NEW result: coeffs={coeffs_final_new}, d={d_final_new}")

# Old implementation (with padding)
print(f"\n=== OLD Implementation (with padding) ===")
coeffs_copy = coeffs.copy()
max_len = max(len(coeffs_copy), 2)

# x < 0: p(x) - alpha*x
coeffs1_old = np.pad(coeffs_copy, (0, max_len - len(coeffs_copy)))
coeffs2_old = np.pad(np.array([layer.alpha, 0]), (0, max_len - 2))
diffl1_old, diffu1_old = minMaxDiffPoly(coeffs1_old, coeffs2_old, l, 0)

# x > 0: p(x) - 1*x
coeffs3_old = np.pad(np.array([1, 0]), (0, max_len - 2))
diffl2_old, diffu2_old = minMaxDiffPoly(coeffs1_old, coeffs3_old, 0, u)

diffl_old = min(diffl1_old, diffl2_old)
diffu_old = max(diffu1_old, diffu2_old)
diffc_old = (diffu_old + diffl_old) / 2
coeffs_final_old = coeffs.copy()
coeffs_final_old[-1] = coeffs_final_old[-1] - diffc_old
d_final_old = diffu_old - diffc_old

print(f"OLD result: coeffs={coeffs_final_old}, d={d_final_old}")

# MATLAB values
print(f"\n=== MATLAB Values ===")
matlab_coeffs = np.array([0.45362903, 0.50000000, 0.06889408])
matlab_d = 0.06898369
print(f"MATLAB: coeffs={matlab_coeffs}, d={matlab_d}")

# Compare
print(f"\n=== Comparison ===")
new_coeffs_diff = np.abs(coeffs_final_new - matlab_coeffs)
old_coeffs_diff = np.abs(coeffs_final_old - matlab_coeffs)
new_d_diff = abs(d_final_new - matlab_d)
old_d_diff = abs(d_final_old - matlab_d)

print(f"NEW coeffs diff: {new_coeffs_diff}")
print(f"OLD coeffs diff: {old_coeffs_diff}")
print(f"NEW d diff: {new_d_diff}")
print(f"OLD d diff: {old_d_diff}")

print(f"\nNEW matches MATLAB better: {np.all(new_coeffs_diff < old_coeffs_diff) and new_d_diff < old_d_diff}")
print(f"OLD matches MATLAB better: {np.all(old_coeffs_diff < new_coeffs_diff) and old_d_diff < new_d_diff}")

# Check which is closer
new_total_diff = np.sum(new_coeffs_diff) + new_d_diff
old_total_diff = np.sum(old_coeffs_diff) + old_d_diff
print(f"NEW total difference: {new_total_diff}")
print(f"OLD total difference: {old_total_diff}")
print(f"Which is closer: {'NEW' if new_total_diff < old_total_diff else 'OLD'}")
