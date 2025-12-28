"""Debug interval to polytope conversion for degenerate case"""
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval

# Test case
c1 = np.array([[5.000], [0.000]])
G1 = np.array([[0.000], [1.000]])
Z1 = Zonotope(c1, G1)

c2 = np.array([[5.650], [0.000]])
G2 = np.array([[0.000, 0.050, 0.000, 0.000, 0.000],
               [0.937, 0.000, -0.005, -0.000, 0.000]])
Z2 = Zonotope(c2, G2)

# Get interval representation of Z1
isInterval, I = Z1.representsa_('interval', 1e-10, return_set=True)
print(f"I (interval from Z1):")
print(f"  inf: {I.inf.flatten()}")
print(f"  sup: {I.sup.flatten()}")

# Convert to polytope
P = I.polytope()
print(f"\nP (polytope from I):")
print(f"  A shape: {P.A.shape}")
print(f"  b shape: {P.b.shape}")
print(f"  A:\n{P.A}")
print(f"  b:\n{P.b.flatten()}")

# Check containment
print("\nCalling P.contains_(Z2)...")
res, cert, scaling = P.contains_(Z2, 'exact', 1e-10, 200, True, True)
print(f"  Result: {res}")
print(f"  Cert: {cert}")
print(f"  Scaling: {scaling}")

# Also check what I.contains_ returns
print("\nCalling I.contains_(Z2)...")
res2, cert2, scaling2 = I.contains_(Z2, 'exact', 1e-10, 200, True, True)
print(f"  Result: {res2}")
print(f"  Cert: {cert2}")
print(f"  Scaling: {scaling2}")

