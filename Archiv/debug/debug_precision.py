#!/usr/bin/env python3
"""
Debug script to investigate precision differences between MATLAB and Python
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

print("=== Precision Investigation ===")
print(f"l: {l}, u: {u}")
print(f"coeffs: {coeffs}")
print(f"coeffs precision: {coeffs.dtype}")

# Test the minMaxDiffPoly function step by step
print(f"\n=== Step 1: x < 0: p(x) - alpha*x ===")
coeffs1 = coeffs
coeffs2 = np.array([layer.alpha, 0])
print(f"coeffs1: {coeffs1}")
print(f"coeffs2: {coeffs2}")
print(f"layer.alpha: {layer.alpha}")

# Let's manually compute the difference polynomial like MATLAB does
print(f"\n=== Manual polynomial difference computation ===")
# MATLAB: p = zeros(1,max(length(coeffs1),length(coeffs2)));
#         p(end-length(coeffs1)+1:end) = coeffs1;
#         p(end-length(coeffs2)+1:end) = p(end-length(coeffs2)+1:end)-coeffs2;
max_len = max(len(coeffs1), len(coeffs2))
p = np.zeros(max_len)
p[-len(coeffs1):] = coeffs1
p[-len(coeffs2):] = p[-len(coeffs2):] - coeffs2
print(f"Difference polynomial p: {p}")

# Compute derivative
from nn.nnHelper.fpolyder import fpolyder
dp = fpolyder(p)
print(f"Derivative dp: {dp}")

# Find roots
roots = np.roots(dp)
print(f"All roots: {roots}")

# Filter real roots
real_roots = roots[np.abs(roots.imag) < 1e-10].real
print(f"Real roots: {real_roots}")

# Filter roots within domain
real_roots = real_roots[(real_roots > l) & (real_roots < u)]
print(f"Roots in domain: {real_roots}")

# Add boundary points
points = np.concatenate([[l], real_roots, [u]])
print(f"Evaluation points: {points}")

# Evaluate difference at all points
values = np.polyval(p, points)
print(f"Values at points: {values}")

# Find bounds
diffl1 = np.min(values)
diffu1 = np.max(values)
print(f"diffl1: {diffl1}, diffu1: {diffu1}")

print(f"\n=== Step 2: x > 0: p(x) - 1*x ===")
coeffs3 = np.array([1, 0])
print(f"coeffs3: {coeffs3}")

# Manual computation for second case
p2 = np.zeros(max_len)
p2[-len(coeffs1):] = coeffs1
p2[-len(coeffs3):] = p2[-len(coeffs3):] - coeffs3
print(f"Difference polynomial p2: {p2}")

dp2 = fpolyder(p2)
print(f"Derivative dp2: {dp2}")

roots2 = np.roots(dp2)
real_roots2 = roots2[np.abs(roots2.imag) < 1e-10].real
real_roots2 = real_roots2[(real_roots2 > 0) & (real_roots2 < u)]
points2 = np.concatenate([[0], real_roots2, [u]])
print(f"Evaluation points2: {points2}")

values2 = np.polyval(p2, points2)
print(f"Values at points2: {values2}")

diffl2 = np.min(values2)
diffu2 = np.max(values2)
print(f"diffl2: {diffl2}, diffu2: {diffu2}")

print(f"\n=== Final computation ===")
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

# Compare with minMaxDiffPoly function
print(f"\n=== Comparison with minMaxDiffPoly function ===")
diffl1_func, diffu1_func = minMaxDiffPoly(coeffs1, coeffs2, l, 0)
diffl2_func, diffu2_func = minMaxDiffPoly(coeffs1, coeffs3, 0, u)
print(f"Function results: diffl1={diffl1_func}, diffu1={diffu1_func}")
print(f"Function results: diffl2={diffl2_func}, diffu2={diffu2_func}")

diffl_func = min(diffl1_func, diffl2_func)
diffu_func = max(diffu1_func, diffu2_func)
diffc_func = (diffu_func + diffl_func) / 2
coeffs_final_func = coeffs.copy()
coeffs_final_func[-1] = coeffs_final_func[-1] - diffc_func
d_final_func = diffu_func - diffc_func

print(f"Function final: coeffs={coeffs_final_func}, d={d_final_func}")
