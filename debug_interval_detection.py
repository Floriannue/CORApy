"""Check if buffered degenerate zonotope is detected as interval"""
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

# Buffer Z1 (this happens in contains_)
tol = 1e-10
I = tol * Interval(-np.ones((2, 1)), np.ones((2, 1)))
Z1_buffered = Z1 + I

print("Z1_buffered:")
print(f"  center: {Z1_buffered.c.flatten()}")
print(f"  generators:\n{Z1_buffered.G}")

# Check if buffered Z1 represents an interval
print("\nChecking if Z1_buffered represents an interval...")
isInterval, I_result = Z1_buffered.representsa_('interval', tol, return_set=True)
print(f"  isInterval: {isInterval}")

if isInterval:
    print(f"  I_result: {I_result}")
    print(f"  I_result.inf: {I_result.inf.flatten()}")
    print(f"  I_result.sup: {I_result.sup.flatten()}")
    
    # Check what interval.contains_ returns
    print("\nCalling I_result.contains_(Z2)...")
    res, cert, scaling = I_result.contains_(Z2, 'exact', tol)
    print(f"  Result: {res}")
    print(f"  Cert: {cert}")
    print(f"  Scaling: {scaling}")

