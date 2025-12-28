"""Test compact step by step to match MATLAB"""
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.private.priv_normalizeConstraints import priv_normalizeConstraints
from cora_python.contSet.polytope.private.priv_compact_zeros import priv_compact_zeros
from cora_python.contSet.polytope.private.priv_compact_alignedIneq import priv_compact_alignedIneq
from cora_python.contSet.polytope.private.priv_compact_2D import priv_compact_2D

# Create the same degenerate case
I = Interval(np.array([[5.], [-1.]]), np.array([[5.], [1.]]))
P = I.polytope()
P.constraints()

print("=== Step 0: Original P ===")
print(f"A:\n{P.A}")
print(f"b: {P.b.flatten()}")

# Shift to origin
c = P.center()
P_shifted = P - c
P_shifted.constraints()

print(f"\n=== Step 1: After shifting ===")
print(f"A:\n{P_shifted.A}")
print(f"b: {P_shifted.b.flatten()}")

# Step 2: Normalize (like MATLAB compact_ does)
A_norm, b_norm, Ae_norm, be_norm = priv_normalizeConstraints(
    P_shifted.A, P_shifted.b.reshape(-1, 1), 
    P_shifted.Ae if P_shifted.Ae.size > 0 else np.zeros((0, P_shifted.dim())),
    P_shifted.be if P_shifted.be.size > 0 else np.zeros((0, 1)),
    'A'
)

print(f"\n=== Step 2: After normalizeConstraints(type='A') ===")
print(f"A_norm:\n{A_norm}")
print(f"b_norm: {b_norm.flatten()}")

# Step 3: priv_compact_zeros
A_z, b_z, Ae_z, be_z, empty_z = priv_compact_zeros(
    A_norm, b_norm, Ae_norm, be_norm, 1e-10
)

print(f"\n=== Step 3: After priv_compact_zeros ===")
print(f"A_z:\n{A_z}")
print(f"b_z: {b_z.flatten()}")
print(f"empty_z: {empty_z}")

# Step 4: priv_compact_alignedIneq
if A_z.size > 0:
    A_a, b_a = priv_compact_alignedIneq(A_z, b_z, 1e-10)
    print(f"\n=== Step 4: After priv_compact_alignedIneq ===")
    print(f"A_a:\n{A_a}")
    print(f"b_a: {b_a.flatten()}")
else:
    A_a, b_a = A_z, b_z

# Step 5: priv_compact_2D (if 2D)
if P_shifted.dim() == 2 and A_a.size > 0:
    A_2d, b_2d, Ae_2d, be_2d, empty_2d = priv_compact_2D(
        A_a, b_a, Ae_z, be_z, 1e-10
    )
    print(f"\n=== Step 5: After priv_compact_2D ===")
    print(f"A_2d:\n{A_2d}")
    print(f"b_2d: {b_2d.flatten()}")
    print(f"empty_2d: {empty_2d}")
else:
    A_2d, b_2d = A_a, b_a

# Compare with Python's compact_
P_compact = P_shifted.compact_()
P_compact.constraints()

print(f"\n=== Python compact_ result ===")
print(f"A:\n{P_compact.A}")
print(f"b: {P_compact.b.flatten()}")

# Check if they match
if P_shifted.dim() == 2:
    match = np.allclose(A_2d, P_compact.A) and np.allclose(b_2d.flatten(), P_compact.b.flatten())
else:
    match = np.allclose(A_a, P_compact.A) and np.allclose(b_a.flatten(), P_compact.b.flatten())

print(f"\n=== Match check ===")
print(f"Step-by-step matches compact_: {match}")

