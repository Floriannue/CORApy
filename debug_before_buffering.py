"""Check if degenerate Z1 (before buffering) is detected as interval"""
import numpy as np
from cora_python.contSet.zonotope import Zonotope

# Test case
c1 = np.array([[5.000], [0.000]])
G1 = np.array([[0.000], [1.000]])
Z1 = Zonotope(c1, G1)

print("Z1 (before buffering):")
print(f"  center: {Z1.c.flatten()}")
print(f"  generators:\n{Z1.G}")

# Check if Z1 represents an interval BEFORE buffering
print("\nChecking if Z1 (before buffering) represents an interval...")
isInterval, I = Z1.representsa_('interval', 1e-10, return_set=True)
print(f"  isInterval: {isInterval}")

if isInterval:
    print(f"  I: {I}")
    print(f"  I.inf: {I.inf.flatten()}")
    print(f"  I.sup: {I.sup.flatten()}")
    
    # This would cause early return in contains_
    print("\n*** This would cause early return to interval.contains_ ***")
else:
    print("\n*** Z1 is NOT detected as interval, will proceed to buffering ***")

