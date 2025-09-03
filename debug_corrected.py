 #!/usr/bin/env python3
"""
Corrected debug script to verify the minMaxDiffPoly function
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

print("=== Corrected Debug ===")
print(f"l: {l}, u: {u}")
print(f"coeffs: {coeffs}")

# Test the minMaxDiffPoly function calls exactly like MATLAB
print(f"\n=== MATLAB calls ===")
# MATLAB: minMaxDiffPoly(coeffs,[obj.alpha,0],l,0)
diffl1, diffu1 = minMaxDiffPoly(coeffs, np.array([layer.alpha, 0]), l, 0)
print(f"minMaxDiffPoly(coeffs, [alpha,0], {l}, 0): diffl={diffl1}, diffu={diffu1}")

# MATLAB: minMaxDiffPoly(coeffs,[1,0],0,u)
diffl2, diffu2 = minMaxDiffPoly(coeffs, np.array([1, 0]), 0, u)
print(f"minMaxDiffPoly(coeffs, [1,0], 0, {u}): diffl={diffl2}, diffu={diffu2}")

# Final computation like MATLAB
diffl = min(diffl1, diffl2)
diffu = max(diffu1, diffu2)
diffc = (diffu + diffl) / 2
coeffs_final = coeffs.copy()
coeffs_final[-1] = coeffs_final[-1] - diffc
d_final = diffu - diffc

print(f"\n=== Final results ===")
print(f"diffl: {diffl}, diffu: {diffu}")
print(f"diffc: {diffc}")
print(f"coeffs_final: {coeffs_final}")
print(f"d_final: {d_final}")

# Compare with MATLAB values
print(f"\n=== MATLAB comparison ===")
matlab_coeffs = np.array([0.45362903, 0.50000000, 0.06889408])
matlab_d = 0.06898369
print(f"MATLAB coeffs: {matlab_coeffs}")
print(f"MATLAB d: {matlab_d}")
print(f"Python coeffs: {coeffs_final}")
print(f"Python d: {d_final}")
print(f"coeffs diff: {np.abs(coeffs_final - matlab_coeffs)}")
print(f"d diff: {abs(d_final - matlab_d)}")

# Test the actual method
print(f"\n=== Actual method test ===")
coeffs_actual, d_actual = layer.computeApproxError(l, u, coeffs)
print(f"Actual method: coeffs={coeffs_actual}, d={d_actual}")
print(f"Matches direct: {np.allclose(coeffs_actual, coeffs_final) and np.isclose(d_actual, d_final)}")
