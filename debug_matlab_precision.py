#!/usr/bin/env python3
"""
Investigate MATLAB precision differences
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

print("=== MATLAB Precision Investigation ===")
print(f"coeffs: {coeffs}")
print(f"coeffs dtype: {coeffs.dtype}")

# Test 1: Root finding precision
print(f"\n=== 1. Root Finding Precision ===")
from nn.nnHelper.fpolyder import fpolyder

# Test the derivative and roots for both cases
coeffs1 = coeffs
coeffs2 = np.array([layer.alpha, 0])  # [0, 0]
max_len = max(len(coeffs1), len(coeffs2))
p = np.zeros(max_len)
p[-len(coeffs1):] = coeffs1
p[-len(coeffs2):] = p[-len(coeffs2):] - coeffs2

dp = fpolyder(p)
print(f"derivative dp: {dp}")
print(f"dp dtype: {dp.dtype}")

roots = np.roots(dp)
print(f"roots: {roots}")
print(f"roots dtype: {roots.dtype}")

# Test with higher precision (use longdouble on Windows)
print(f"\nTesting with higher precision:")
try:
    dp_high = np.array(dp, dtype=np.longdouble)
    roots_high = np.roots(dp_high)
    print(f"roots (longdouble): {roots_high}")
except:
    print("longdouble not available, using float64")

# Test 2: Polynomial evaluation precision
print(f"\n=== 2. Polynomial Evaluation Precision ===")
points = np.array([-1.0, -0.55111111, 0.0])
print(f"evaluation points: {points}")

# Standard precision
values_std = np.polyval(p, points)
print(f"values (float64): {values_std}")

# Higher precision
try:
    p_high = np.array(p, dtype=np.longdouble)
    points_high = np.array(points, dtype=np.longdouble)
    values_high = np.polyval(p_high, points_high)
    print(f"values (longdouble): {values_high}")
except:
    print("longdouble not available for polynomial evaluation")

# Test 3: Matrix operations precision
print(f"\n=== 3. Matrix Operations Precision ===")
print(f"Original x: {x}")
print(f"Original y: {y}")

# Test leastSquarePolyFunc with different precisions
A = np.column_stack([x ** i for i in range(order + 1)])
print(f"A shape: {A.shape}, A dtype: {A.dtype}")

# Standard precision
coeffs_std = np.linalg.pinv(A) @ y
coeffs_std = np.flip(coeffs_std)
print(f"coeffs (float64): {coeffs_std}")

# Higher precision
try:
    A_high = np.array(A, dtype=np.longdouble)
    y_high = np.array(y, dtype=np.longdouble)
    coeffs_high = np.linalg.pinv(A_high) @ y_high
    coeffs_high = np.flip(coeffs_high)
    print(f"coeffs (longdouble): {coeffs_high}")
except:
    print("longdouble not available for matrix operations")

# Test 4: Check if MATLAB uses different algorithms
print(f"\n=== 4. Algorithm Differences ===")

# Check if there are any differences in the polynomial coefficient ordering
print(f"MATLAB expects: [0.45362903, 0.50000000, 0.06889408]")
print(f"Python gives:   {coeffs}")
print(f"Difference:     {np.abs(coeffs - np.array([0.45362903, 0.50000000, 0.06889408]))}")

# Test if the issue is in the initial polynomial fitting
print(f"\nTesting different polynomial fitting methods:")

# Method 1: Using np.polyfit (different algorithm)
coeffs_polyfit = np.polyfit(x, y, order)
print(f"np.polyfit: {coeffs_polyfit}")

# Method 2: Using np.linalg.lstsq (different algorithm)
coeffs_lstsq = np.linalg.lstsq(A, y, rcond=None)[0]
coeffs_lstsq = np.flip(coeffs_lstsq)
print(f"np.linalg.lstsq: {coeffs_lstsq}")

# Method 3: Using np.linalg.solve (if A is square)
if A.shape[0] == A.shape[1]:
    coeffs_solve = np.linalg.solve(A, y)
    coeffs_solve = np.flip(coeffs_solve)
    print(f"np.linalg.solve: {coeffs_solve}")

# Test 5: Check if MATLAB uses different tolerance settings
print(f"\n=== 5. Tolerance Settings ===")
print(f"NumPy default tolerance for pinv: {np.finfo(np.float64).eps}")
print(f"NumPy default tolerance for roots: {np.finfo(np.float64).eps}")

# Test with different tolerances for pinv
coeffs_pinv_tol1 = np.linalg.pinv(A, rcond=1e-15) @ y
coeffs_pinv_tol1 = np.flip(coeffs_pinv_tol1)
print(f"pinv (rcond=1e-15): {coeffs_pinv_tol1}")

coeffs_pinv_tol2 = np.linalg.pinv(A, rcond=1e-12) @ y
coeffs_pinv_tol2 = np.flip(coeffs_pinv_tol2)
print(f"pinv (rcond=1e-12): {coeffs_pinv_tol2}")

coeffs_pinv_tol3 = np.linalg.pinv(A, rcond=1e-9) @ y
coeffs_pinv_tol3 = np.flip(coeffs_pinv_tol3)
print(f"pinv (rcond=1e-9): {coeffs_pinv_tol3}")
