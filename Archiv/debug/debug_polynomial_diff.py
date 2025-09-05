import numpy as np
from cora_python.nn.nnHelper.minMaxDiffPoly import minMaxDiffPoly

# Test case from the failing scenario
coeffs1 = np.array([-0.1, 0.0])  # p1(x) = -0.1*x + 0
coeffs2 = np.array([0.01, 0.0])  # p2(x) = 0.01*x + 0
l, u = -1, 0

print("Python minMaxDiffPoly:")
print(f"coeffs1: {coeffs1}")
print(f"coeffs2: {coeffs2}")
print(f"l: {l}, u: {u}")

# Current Python implementation
diffl, diffu = minMaxDiffPoly(coeffs1, coeffs2, l, u)
print(f"Result: diffl = {diffl}, diffu = {diffu}")

# Let's manually compute what MATLAB does
print("\nManual computation (MATLAB style):")

# MATLAB style: pad to same length, subtract
max_len = max(len(coeffs1), len(coeffs2))
p1_padded = np.pad(coeffs1, (max_len - len(coeffs1), 0))
p2_padded = np.pad(coeffs2, (max_len - len(coeffs2), 0))
p_diff = p1_padded - p2_padded

print(f"p1_padded: {p1_padded}")
print(f"p2_padded: {p2_padded}")
print(f"p_diff: {p_diff}")

# Evaluate at boundaries
val_l = np.polyval(p_diff[::-1], l)  # Reverse for polyval
val_u = np.polyval(p_diff[::-1], u)

print(f"val_l: {val_l}")
print(f"val_u: {val_u}")

# For linear polynomials, no extrema in between
diffl_manual = min(val_l, val_u)
diffu_manual = max(val_l, val_u)

print(f"Manual result: diffl = {diffl_manual}, diffu = {diffu_manual}")
