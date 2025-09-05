#!/usr/bin/env python3
"""
Detailed precision investigation - compare each step with MATLAB
"""

import sys
sys.path.append('cora_python')
from nn.layers.nonlinear.nnReLULayer import nnReLULayer
from nn.nnHelper.leastSquarePolyFunc import leastSquarePolyFunc
from nn.nnHelper.minMaxDiffPoly import minMaxDiffPoly
from nn.nnHelper.fpolyder import fpolyder
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

print("=== Detailed Precision Investigation ===")
print(f"l: {l}, u: {u}")
print(f"coeffs: {coeffs}")
print(f"coeffs dtype: {coeffs.dtype}")

# MATLAB values for comparison
matlab_coeffs = np.array([0.45362903, 0.50000000, 0.06889408])
matlab_d = 0.06898369

print(f"\n=== Step-by-step comparison ===")

# Step 1: minMaxDiffPoly(coeffs, [alpha,0], l, 0)
print(f"\n1. minMaxDiffPoly(coeffs, [alpha,0], {l}, 0)")
coeffs1 = coeffs
coeffs2 = np.array([layer.alpha, 0])
print(f"   coeffs1: {coeffs1}")
print(f"   coeffs2: {coeffs2}")

# Manual computation to see each step
max_len = max(len(coeffs1), len(coeffs2))
p = np.zeros(max_len)
p[-len(coeffs1):] = coeffs1
p[-len(coeffs2):] = p[-len(coeffs2):] - coeffs2
print(f"   difference polynomial p: {p}")

dp = fpolyder(p)
print(f"   derivative dp: {dp}")

roots = np.roots(dp)
print(f"   all roots: {roots}")

real_roots = roots[np.abs(roots.imag) < 1e-10].real
print(f"   real roots: {real_roots}")

real_roots = real_roots[(real_roots > l) & (real_roots < 0)]
print(f"   roots in domain [{l}, 0]: {real_roots}")

points = np.concatenate([[l], real_roots, [0]])
print(f"   evaluation points: {points}")

values = np.polyval(p, points)
print(f"   values at points: {values}")

diffl1 = np.min(values)
diffu1 = np.max(values)
print(f"   diffl1: {diffl1}, diffu1: {diffu1}")

# Step 2: minMaxDiffPoly(coeffs, [1,0], 0, u)
print(f"\n2. minMaxDiffPoly(coeffs, [1,0], 0, {u})")
coeffs3 = np.array([1, 0])
print(f"   coeffs3: {coeffs3}")

p2 = np.zeros(max_len)
p2[-len(coeffs1):] = coeffs1
p2[-len(coeffs3):] = p2[-len(coeffs3):] - coeffs3
print(f"   difference polynomial p2: {p2}")

dp2 = fpolyder(p2)
print(f"   derivative dp2: {dp2}")

roots2 = np.roots(dp2)
real_roots2 = roots2[np.abs(roots2.imag) < 1e-10].real
real_roots2 = real_roots2[(real_roots2 > 0) & (real_roots2 < u)]
points2 = np.concatenate([[0], real_roots2, [u]])
print(f"   evaluation points2: {points2}")

values2 = np.polyval(p2, points2)
print(f"   values at points2: {values2}")

diffl2 = np.min(values2)
diffu2 = np.max(values2)
print(f"   diffl2: {diffl2}, diffu2: {diffu2}")

# Final computation
print(f"\n3. Final computation")
diffl = min(diffl1, diffl2)
diffu = max(diffu1, diffu2)
diffc = (diffu + diffl) / 2
coeffs_final = coeffs.copy()
coeffs_final[-1] = coeffs_final[-1] - diffc
d_final = diffu - diffc

print(f"   diffl: {diffl}, diffu: {diffu}")
print(f"   diffc: {diffc}")
print(f"   coeffs_final: {coeffs_final}")
print(f"   d_final: {d_final}")

# Compare with MATLAB
print(f"\n=== MATLAB Comparison ===")
print(f"MATLAB coeffs: {matlab_coeffs}")
print(f"MATLAB d: {matlab_d}")
print(f"Python coeffs: {coeffs_final}")
print(f"Python d: {d_final}")
print(f"coeffs difference: {np.abs(coeffs_final - matlab_coeffs)}")
print(f"d difference: {abs(d_final - matlab_d)}")

# Check if the difference is within acceptable tolerance
tolerance = 1e-4
coeffs_close = np.allclose(coeffs_final, matlab_coeffs, rtol=tolerance)
d_close = np.isclose(d_final, matlab_d, rtol=tolerance)
print(f"\nWithin tolerance {tolerance}: coeffs={coeffs_close}, d={d_close}")

# Test with function calls
print(f"\n=== Function call comparison ===")
diffl1_func, diffu1_func = minMaxDiffPoly(coeffs1, coeffs2, l, 0)
diffl2_func, diffu2_func = minMaxDiffPoly(coeffs1, coeffs3, 0, u)
diffl_func = min(diffl1_func, diffl2_func)
diffu_func = max(diffu1_func, diffu2_func)
diffc_func = (diffu_func + diffl_func) / 2
coeffs_final_func = coeffs.copy()
coeffs_final_func[-1] = coeffs_final_func[-1] - diffc_func
d_final_func = diffu_func - diffc_func

print(f"Function call result: coeffs={coeffs_final_func}, d={d_final_func}")
print(f"Matches manual: {np.allclose(coeffs_final, coeffs_final_func) and np.isclose(d_final, d_final_func)}")
