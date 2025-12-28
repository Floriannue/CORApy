"""Compare compact behavior between MATLAB and Python"""
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope.polytope import Polytope

# Create the same degenerate case
I = Interval(np.array([[5.], [-1.]]), np.array([[5.], [1.]]))
P = I.polytope()
P.constraints()

print("=== Original P ===")
print(f"A:\n{P.A}")
print(f"b: {P.b.flatten()}")

# Shift to origin (like in contains_)
c = P.center()
P_shifted = P - c
P_shifted.constraints()

print(f"\n=== After shifting ===")
print(f"A:\n{P_shifted.A}")
print(f"b: {P_shifted.b.flatten()}")

# Python compact
P_compact_py = P_shifted.compact_()
P_compact_py.constraints()

print(f"\n=== Python compact ===")
print(f"A:\n{P_compact_py.A}")
print(f"b: {P_compact_py.b.flatten()}")

# Check what MATLAB would do:
# MATLAB's compact_2D removes constraints based on:
# 1. priv_compact_zeros - removes 0*x <= b constraints where b >= 0
# 2. priv_compact_alignedIneq - removes aligned constraints, keeping the one with smallest b
# 3. priv_compact_2D - removes redundant constraints based on geometry

# For our case:
# A = [[1, 0], [0, 1], [-1, 0], [0, -1]]
# b = [0, 2, 0, 0]
# After shifting, we have constraints:
#   [1, 0] x <= 0  (x <= 0)
#   [0, 1] x <= 2  (y <= 2)
#   [-1, 0] x <= 0  (-x <= 0, i.e., x >= 0)
#   [0, -1] x <= 0  (-y <= 0, i.e., y >= 0)

# priv_compact_zeros should remove [1, 0] x <= 0 and [-1, 0] x <= 0
# because they are 0*x <= 0 (effectively) after normalization

# Let's check if Python's compact_zeros matches MATLAB
from cora_python.contSet.polytope.private.priv_compact_zeros import priv_compact_zeros

A = P_shifted.A.copy()
b = P_shifted.b.copy()
Ae = P_shifted.Ae.copy() if P_shifted.Ae.size > 0 else np.zeros((0, P_shifted.dim()))
be = P_shifted.be.copy() if P_shifted.be.size > 0 else np.zeros((0, 1))

print(f"\n=== Testing priv_compact_zeros ===")
A_z, b_z, Ae_z, be_z, empty_z = priv_compact_zeros(A, b, Ae, be, 1e-10)
print(f"After compact_zeros:")
print(f"A:\n{A_z}")
print(f"b: {b_z.flatten()}")
print(f"empty: {empty_z}")

# Check aligned constraints
from cora_python.contSet.polytope.private.priv_compact_alignedIneq import priv_compact_alignedIneq

if A_z.size > 0:
    print(f"\n=== Testing priv_compact_alignedIneq ===")
    A_a, b_a = priv_compact_alignedIneq(A_z, b_z, 1e-10)
    print(f"After compact_alignedIneq:")
    print(f"A:\n{A_a}")
    print(f"b: {b_a.flatten()}")

