"""Test -1 * P logic matches MATLAB exactly"""
import numpy as np

# MATLAB logic for fac < 0:
# P.A_.val = -P.A_.val;
# P.b_.val = -P.b_.val * fac;
# P.be_.val = P.be_.val * fac;

# For fac = -1:
# A_new = -A
# b_new = -b * (-1) = b
# be_new = be * (-1) = -be

# Original constraints: A x <= b
# After -1: -A x <= b
# This means: A x >= -b (flipping inequality)
# Which is equivalent to: -A x <= b (the form we have)

# Test with actual values
A = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
b = np.array([5., 1., -5., 1.])

print("Original constraints:")
print(f"A:\n{A}")
print(f"b: {b}")
print("\nConstraints: A x <= b")
for i in range(A.shape[0]):
    print(f"  {A[i]} x <= {b[i]}")

# Apply -1 scaling (MATLAB logic)
A_new = -A
b_new = -b * (-1)  # = b

print("\nAfter -1 * P (MATLAB logic):")
print(f"A_new:\n{A_new}")
print(f"b_new: {b_new}")
print("\nNew constraints: A_new x <= b_new")
for i in range(A_new.shape[0]):
    print(f"  {A_new[i]} x <= {b_new[i]}")

# Verify: original constraint A[i] x <= b[i] becomes -A[i] x <= b[i]
# Which means A[i] x >= -b[i]
print("\nVerification:")
print("Original: [1, 0] x <= 5  =>  x <= 5")
print("After -1: [-1, 0] x <= 5  =>  -x <= 5  =>  x >= -5")
print("But we want: -x <= 5, which is correct!")

# The issue: when we shift P - c, then do -1 * (P - c), we need to ensure
# the constraints are correct for the shifted polytope

