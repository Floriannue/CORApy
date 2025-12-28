"""Debug script to check if buffered degenerate zonotope is detected as interval"""
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval

# Test case
c1 = np.array([[5.000], [0.000]])
G1 = np.array([[0.000], [1.000]])
Z1 = Zonotope(c1, G1)

print("Z1 (degenerate):")
print(f"  center: {Z1.c.flatten()}")
print(f"  generators:\n{Z1.G}")

# Check if Z1 represents an interval BEFORE buffering
print("\nBefore buffering:")
isInterval, I = Z1.representsa_('interval', 1e-10, return_set=True)
print(f"  Z1.representsa_('interval'): {isInterval}")

# Buffer Z1
tol = 1e-10
I_buf = tol * Interval(-np.ones((2, 1)), np.ones((2, 1)))
Z1_buffered = Z1 + I_buf

print("\nAfter buffering:")
print(f"  Z1_buffered center: {Z1_buffered.c.flatten()}")
print(f"  Z1_buffered generators shape: {Z1_buffered.G.shape}")
print(f"  Z1_buffered generators:\n{Z1_buffered.G}")

# Check if buffered Z1 represents an interval
isInterval2, I2 = Z1_buffered.representsa_('interval', 1e-10, return_set=True)
print(f"  Z1_buffered.representsa_('interval'): {isInterval2}")

if isInterval2:
    print(f"  Interval I2: {I2}")
    print(f"  I2.inf: {I2.inf.flatten()}")
    print(f"  I2.sup: {I2.sup.flatten()}")

