"""Debug priv_compact_2D to see why it's not removing constraints"""
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.private.priv_normalizeConstraints import priv_normalizeConstraints
from cora_python.contSet.polytope.private.priv_compact_zeros import priv_compact_zeros
from cora_python.contSet.polytope.private.priv_compact_toEquality import priv_compact_toEquality
from cora_python.contSet.polytope.private.priv_compact_alignedEq import priv_compact_alignedEq
from cora_python.contSet.polytope.private.priv_compact_alignedIneq import priv_compact_alignedIneq
from cora_python.contSet.polytope.private.priv_compact_2D import priv_compact_2D

# Create the same degenerate case
I = Interval(np.array([[5.], [-1.]]), np.array([[5.], [1.]]))
P = I.polytope()
P.constraints()

# Shift to origin
c = P.center()
P_shifted = P - c

# Normalize
A_norm, b_norm, Ae_norm, be_norm = priv_normalizeConstraints(
    P_shifted.A, P_shifted.b.reshape(-1, 1), 
    P_shifted.Ae if P_shifted.Ae.size > 0 else np.zeros((0, P_shifted.dim())),
    P_shifted.be if P_shifted.be.size > 0 else np.zeros((0, 1)),
    'A'
)

print("=== After normalization ===")
print(f"A:\n{A_norm}")
print(f"b: {b_norm.flatten()}")

# priv_compact_zeros
A_z, b_z, Ae_z, be_z, empty_z = priv_compact_zeros(A_norm, b_norm, Ae_norm, be_norm, 1e-10)
print(f"\n=== After priv_compact_zeros ===")
print(f"A:\n{A_z}")
print(f"b: {b_z.flatten()}")

# priv_compact_toEquality
A_e, b_e, Ae_e, be_e = priv_compact_toEquality(A_z, b_z, Ae_z, be_z, 1e-10)
print(f"\n=== After priv_compact_toEquality ===")
print(f"A:\n{A_e}")
print(f"b: {b_e.flatten()}")
print(f"Ae:\n{Ae_e}")
print(f"be: {be_e.flatten()}")

# priv_compact_alignedEq
Ae_a, be_a, empty_a = priv_compact_alignedEq(Ae_e, be_e, 1e-10)
print(f"\n=== After priv_compact_alignedEq ===")
print(f"Ae:\n{Ae_a}")
print(f"be: {be_a.flatten()}")

# priv_compact_alignedIneq
A_i, b_i = priv_compact_alignedIneq(A_e, b_e, 1e-10)
print(f"\n=== After priv_compact_alignedIneq ===")
print(f"A:\n{A_i}")
print(f"b: {b_i.flatten()}")

# priv_compact_2D - THIS IS THE KEY STEP
A_2d, b_2d, Ae_2d, be_2d, empty_2d = priv_compact_2D(A_i, b_i, Ae_a, be_a, 1e-10)
print(f"\n=== After priv_compact_2D ===")
print(f"A:\n{A_2d}")
print(f"b: {b_2d.flatten()}")
print(f"empty: {empty_2d}")

# Compare with compact_
P_compact = P_shifted.compact_()
P_compact.constraints()
print(f"\n=== Python compact_ result ===")
print(f"A:\n{P_compact.A}")
print(f"b: {P_compact.b.flatten()}")

# Check if they match
if A_2d.size > 0 and P_compact.A.size > 0:
    # Sort both for comparison
    A_2d_sorted = A_2d[np.lexsort(A_2d.T[::-1])]
    P_compact_A_sorted = P_compact.A[np.lexsort(P_compact.A.T[::-1])]
    match = np.allclose(A_2d_sorted, P_compact_A_sorted) and np.allclose(np.sort(b_2d.flatten()), np.sort(P_compact.b.flatten()))
    print(f"\n=== Match check ===")
    print(f"priv_compact_2D result matches compact_: {match}")
    if not match:
        print(f"priv_compact_2D returned {A_2d.shape[0]} constraints")
        print(f"compact_ returned {P_compact.A.shape[0]} constraints")

