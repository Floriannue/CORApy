#!/usr/bin/env python3
"""
Detailed investigation of minMaxDiffPoly differences
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

print("=== Detailed minMaxDiffPoly Investigation ===")
print(f"coeffs: {coeffs}")

# Test the exact same calls as MATLAB
coeffs1 = coeffs
coeffs2 = np.array([layer.alpha, 0])  # [0, 0]
coeffs3 = np.array([1, 0])

print(f"\n=== Step 1: minMaxDiffPoly(coeffs, [alpha,0], {l}, 0) ===")
print(f"coeffs1: {coeffs1}")
print(f"coeffs2: {coeffs2}")

# Manual step-by-step computation to match MATLAB exactly
print(f"\n--- Manual computation ---")
max_len = max(len(coeffs1), len(coeffs2))
print(f"max_len: {max_len}")

# MATLAB: p = zeros(1,max(length(coeffs1),length(coeffs2)));
p = np.zeros(max_len)
print(f"p (initial): {p}")

# MATLAB: p(end-length(coeffs1)+1:end) = coeffs1;
p[-len(coeffs1):] = coeffs1
print(f"p (after coeffs1): {p}")

# MATLAB: p(end-length(coeffs2)+1:end) = p(end-length(coeffs2)+1:end)-coeffs2;
p[-len(coeffs2):] = p[-len(coeffs2):] - coeffs2
print(f"p (after coeffs2): {p}")

# MATLAB: dp = fpolyder(p);
dp = fpolyder(p)
print(f"dp: {dp}")

# MATLAB: dp_roots = roots(dp);
roots = np.roots(dp)
print(f"roots: {roots}")

# MATLAB: dp_roots = dp_roots(imag(dp_roots) == 0);
real_roots = roots[np.abs(roots.imag) < 1e-10].real
print(f"real_roots: {real_roots}")

# MATLAB: dp_roots = dp_roots(l < dp_roots & dp_roots < u);
# But for this call, u=0, so domain is [l, 0] = [-1, 0]
real_roots = real_roots[(real_roots > l) & (real_roots < 0)]
print(f"roots in domain [{l}, 0]: {real_roots}")

# MATLAB: extrema = [l, dp_roots', u];
extrema = np.concatenate([[l], real_roots, [0]])
print(f"extrema: {extrema}")

# MATLAB: diff = polyval(p, extrema);
diff = np.polyval(p, extrema)
print(f"diff: {diff}")

# MATLAB: diffl = min(diff); diffu = max(diff);
diffl1 = np.min(diff)
diffu1 = np.max(diff)
print(f"diffl1: {diffl1}, diffu1: {diffu1}")

# Compare with function call
print(f"\n--- Function call comparison ---")
diffl1_func, diffu1_func = minMaxDiffPoly(coeffs1, coeffs2, l, 0)
print(f"Function result: diffl={diffl1_func}, diffu={diffu1_func}")
print(f"Match: {np.isclose(diffl1, diffl1_func) and np.isclose(diffu1, diffu1_func)}")

print(f"\n=== Step 2: minMaxDiffPoly(coeffs, [1,0], 0, {u}) ===")
print(f"coeffs1: {coeffs1}")
print(f"coeffs3: {coeffs3}")

# Manual computation for step 2
p2 = np.zeros(max_len)
p2[-len(coeffs1):] = coeffs1
p2[-len(coeffs3):] = p2[-len(coeffs3):] - coeffs3
print(f"p2: {p2}")

dp2 = fpolyder(p2)
print(f"dp2: {dp2}")

roots2 = np.roots(dp2)
real_roots2 = roots2[np.abs(roots2.imag) < 1e-10].real
real_roots2 = real_roots2[(real_roots2 > 0) & (real_roots2 < u)]
extrema2 = np.concatenate([[0], real_roots2, [u]])
diff2 = np.polyval(p2, extrema2)
diffl2 = np.min(diff2)
diffu2 = np.max(diff2)
print(f"diffl2: {diffl2}, diffu2: {diffu2}")

# Compare with function call
diffl2_func, diffu2_func = minMaxDiffPoly(coeffs1, coeffs3, 0, u)
print(f"Function result: diffl={diffl2_func}, diffu={diffu2_func}")
print(f"Match: {np.isclose(diffl2, diffl2_func) and np.isclose(diffu2, diffu2_func)}")

print(f"\n=== Final Error Computation ===")
# MATLAB: diffl = min(diffl1,diffl2); diffu = max(diffu1,diffu2);
diffl = min(diffl1, diffl2)
diffu = max(diffu1, diffu2)
print(f"diffl: {diffl}, diffu: {diffu}")

# MATLAB: diffc = (diffu+diffl)/2;
diffc = (diffu + diffl) / 2
print(f"diffc: {diffc}")

# MATLAB: coeffs(end) = coeffs(end) - diffc;
coeffs_final = coeffs.copy()
coeffs_final[-1] = coeffs_final[-1] - diffc
print(f"coeffs_final: {coeffs_final}")

# MATLAB: d = diffu-diffc;
d_final = diffu - diffc
print(f"d_final: {d_final}")

# Compare with MATLAB values
print(f"\n=== MATLAB Comparison ===")
matlab_coeffs = np.array([0.45362903, 0.50000000, 0.06889408])
matlab_d = 0.06898369
print(f"MATLAB coeffs: {matlab_coeffs}")
print(f"MATLAB d: {matlab_d}")
print(f"Python coeffs: {coeffs_final}")
print(f"Python d: {d_final}")
print(f"coeffs diff: {np.abs(coeffs_final - matlab_coeffs)}")
print(f"d diff: {abs(d_final - matlab_d)}")

# Test with function calls
print(f"\n=== Function Call Results ===")
coeffs_actual, d_actual = layer.computeApproxError(l, u, coeffs)
print(f"Actual method: coeffs={coeffs_actual}, d={d_actual}")
print(f"Matches manual: {np.allclose(coeffs_actual, coeffs_final) and np.isclose(d_actual, d_final)}")

# Check if the difference is within 1e-5 tolerance
tolerance = 1e-5
coeffs_close = np.allclose(coeffs_final, matlab_coeffs, rtol=tolerance)
d_close = np.isclose(d_final, matlab_d, rtol=tolerance)
print(f"\nWithin tolerance {tolerance}: coeffs={coeffs_close}, d={d_close}")

if not d_close:
    print(f"d difference: {abs(d_final - matlab_d)}")
    print(f"Relative difference: {abs(d_final - matlab_d) / matlab_d}")
    print(f"Tolerance needed: {abs(d_final - matlab_d) / matlab_d}")
