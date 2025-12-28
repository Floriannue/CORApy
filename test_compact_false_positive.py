"""Test if compact can cause false positives in containment"""
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.zonotope import Zonotope

# Create a case where compact might remove necessary constraints
# A degenerate polytope (line segment) represented as an interval
I = Interval(np.array([[5.], [-1.]]), np.array([[5.], [1.]]))
P = I.polytope()
P.constraints()

print("=== Original P ===")
print(f"A:\n{P.A}")
print(f"b: {P.b.flatten()}")

# Shift to origin
c = P.center()
P_shifted = P - c
P_shifted.constraints()

print(f"\n=== After shifting ===")
print(f"A:\n{P_shifted.A}")
print(f"b: {P_shifted.b.flatten()}")

# Compact removes x-direction constraints
P_compact = P_shifted.compact_()
P_compact.constraints()

print(f"\n=== After compact ===")
print(f"A:\n{P_compact.A}")
print(f"b: {P_compact.b.flatten()}")

# Create a zonotope that extends in x-direction (should not be contained)
c2 = np.array([[5.65], [0.0]])
G2 = np.array([[0.1, 0.0], [0.0, 0.5]])
Z2 = Zonotope(c2, G2)
S_shifted = Z2 - c

print(f"\n=== Test containment ===")
print(f"S_shifted center: {S_shifted.c.flatten()}")
print(f"S_shifted generators:\n{S_shifted.G}")

# Check with original (non-compacted) - should detect violation in x-direction
res_original, _, scaling_original = P_shifted.contains_(S_shifted, 'exact', 1e-10, None, False, True)
print(f"\nContains (original, non-compacted): {res_original}")
print(f"Scaling (original): {scaling_original}")

# Check with compacted - might miss violation if x-direction constraints removed
res_compact, _, scaling_compact = P_compact.contains_(S_shifted, 'exact', 1e-10, None, False, True)
print(f"Contains (compacted): {res_compact}")
print(f"Scaling (compacted): {scaling_compact}")

# Expected: False (Z2 extends beyond P in x-direction)
print(f"\nExpected: False")
print(f"Original correct: {res_original == False}")
print(f"Compacted correct: {res_compact == False}")
print(f"Both match: {res_original == res_compact}")

