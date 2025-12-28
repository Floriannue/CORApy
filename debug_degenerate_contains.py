"""Debug script for degenerate zonotope containment"""
import numpy as np
from cora_python.contSet.zonotope import Zonotope

# Test case from failing test
c1 = np.array([[5.000], [0.000]])
G1 = np.array([[0.000], [1.000]])
Z1 = Zonotope(c1, G1)

c2 = np.array([[5.650], [0.000]])
G2 = np.array([[0.000, 0.050, 0.000, 0.000, 0.000],
               [0.937, 0.000, -0.005, -0.000, 0.000]])
Z2 = Zonotope(c2, G2)

print("Z1 (degenerate):")
print(f"  center: {Z1.c.flatten()}")
print(f"  generators:\n{Z1.G}")
print(f"  isFullDim: {Z1.isFullDim(1e-10)}")

print("\nZ2 (full-dimensional):")
print(f"  center: {Z2.c.flatten()}")
print(f"  generators:\n{Z2.G}")
print(f"  isFullDim: {Z2.isFullDim(1e-10)}")

# Check containment
print("\nChecking Z1.contains_(Z2)...")
res, cert, scaling = Z1.contains_(Z2, 'exact', 1e-10)
print(f"  Result: {res} (expected: False)")
print(f"  Cert: {cert}")
print(f"  Scaling: {scaling}")

# Check what happens after buffering
print("\nAfter buffering Z1:")
from cora_python.contSet.interval import Interval
tol = 1e-10
I = tol * Interval(-np.ones((2, 1)), np.ones((2, 1)))
Z1_buffered = Z1 + I
print(f"  Z1_buffered center: {Z1_buffered.c.flatten()}")
print(f"  Z1_buffered generators shape: {Z1_buffered.G.shape}")

# Convert to polytope
from cora_python.contSet.polytope.polytope import Polytope
P1 = Polytope(Z1_buffered)
print(f"\nP1 (polytope from buffered Z1):")
print(f"  A shape: {P1.A.shape}")
print(f"  b shape: {P1.b.shape}")

# Check containment via polytope
print("\nChecking P1.contains_(Z2)...")
res2, cert2, scaling2 = P1.contains_(Z2, 'exact:polymax', 1e-10)
print(f"  Result: {res2} (expected: False)")
print(f"  Cert: {cert2}")

