"""Test if compact removes necessary constraints for containment"""
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.zonotope import Zonotope

# Create the degenerate case from the test
c1 = np.array([[5.000], [0.000]])
G1 = np.array([[0.000], [1.000]])
Z1 = Zonotope(c1, G1)

c2 = np.array([[5.650], [0.000]])
G2 = np.array([[0.000, 0.050, 0.000, 0.000, 0.000],
               [0.937, 0.000, -0.005, -0.000, 0.000]])
Z2 = Zonotope(c2, G2)

# Get interval representation
isInterval, I = Z1.representsa_('interval', 1e-10, return_set=True)
P = I.polytope()
P.constraints()

print("=== Original P (before compact) ===")
print(f"A:\n{P.A}")
print(f"b: {P.b.flatten()}")

# Shift to origin (like in contains_)
c = P.center()
P_shifted = P - c
P_shifted.constraints()

print(f"\n=== After shifting (before compact) ===")
print(f"A:\n{P_shifted.A}")
print(f"b: {P_shifted.b.flatten()}")

# Now compact
P_compact = P_shifted.compact_()
P_compact.constraints()

print(f"\n=== After compact ===")
print(f"A:\n{P_compact.A}")
print(f"b: {P_compact.b.flatten()}")

# Check containment with original and compacted
S_shifted = Z2 - c

print(f"\n=== Containment check ===")
print(f"S_shifted center: {S_shifted.c.flatten()}")

# Check with original (non-compacted)
res_original, _, _ = P_shifted.contains_(S_shifted, 'exact', 1e-10, None, False, True)
print(f"Contains (original, non-compacted): {res_original}")

# Check with compacted
res_compact, _, _ = P_compact.contains_(S_shifted, 'exact', 1e-10, None, False, True)
print(f"Contains (compacted): {res_compact}")

# The expected result is False (Z2 is not contained in Z1)
print(f"\nExpected: False")
print(f"Original matches expected: {res_original == False}")
print(f"Compacted matches expected: {res_compact == False}")

