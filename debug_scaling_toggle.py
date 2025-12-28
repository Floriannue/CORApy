"""Debug script to check difference when scalingToggle is True vs False"""
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

print("Test case:")
print(f"Z1 (degenerate): center={Z1.c.flatten()}, generators shape={Z1.G.shape}")
print(f"Z2 (full-dim): center={Z2.c.flatten()}, generators shape={Z2.G.shape}")

# Check with scalingToggle=True (default)
print("\n=== With scalingToggle=True ===")
res1, cert1, scaling1 = Z1.contains_(Z2, 'exact', 1e-10, 200, True, True)
print(f"Result: {res1} (expected: False)")
print(f"Cert: {cert1}")
print(f"Scaling: {scaling1}")

# Check with scalingToggle=False
print("\n=== With scalingToggle=False ===")
res2, cert2, scaling2 = Z1.contains_(Z2, 'exact', 1e-10, 200, True, False)
print(f"Result: {res2} (expected: False)")
print(f"Cert: {cert2}")
print(f"Scaling: {scaling2}")

# Check what method is actually being used
print("\n=== Tracing method selection ===")
# After buffering, Z1 should go through aux_exactParser
# Since Z2 is a zonotope and method='exact', it should use 'exact:polymax'
# (because dim < 4)
print(f"Z2.dim(): {Z2.dim()}")
print(f"Method should be: 'exact:polymax' (since dim < 4)")

# Check directly with exact:polymax
print("\n=== Direct call with exact:polymax ===")
res3, cert3, scaling3 = Z1.contains_(Z2, 'exact:polymax', 1e-10, 200, True, True)
print(f"Result: {res3} (expected: False)")
print(f"Cert: {cert3}")
print(f"Scaling: {scaling3}")

