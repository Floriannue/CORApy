"""Debug zonotope contains issue"""
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval

# Test case from failing test
Z1 = Zonotope(Interval(np.array([1,2]), np.array([4,6])))
Z2 = Zonotope(np.array([[3],[4]]), np.array([[1],[1]]))

print("Z1 (interval zonotope):")
print(f"  Z1.c: {Z1.c}")
print(f"  Z1.G: {Z1.G}")
print(f"  Z1.representsa_('interval'): {Z1.representsa_('interval', 1e-10)}")

print("\nZ2:")
print(f"  Z2.c: {Z2.c.flatten()}")
print(f"  Z2.G: {Z2.G.flatten()}")

# Check if Z1 represents an interval
Z_isInterval, I = Z1.representsa_('interval', 1e-10, return_set=True)
print(f"\nZ1 represents interval: {Z_isInterval}")
if I is not None:
    print(f"  Interval I.inf: {I.inf.flatten()}")
    print(f"  Interval I.sup: {I.sup.flatten()}")

# Check Z2 bounds
Z2_int = Z2.interval()
print(f"\nZ2 interval bounds:")
print(f"  Z2.inf: {Z2_int.inf.flatten()}")
print(f"  Z2.sup: {Z2_int.sup.flatten()}")

# Check containment manually
print(f"\nManual containment check:")
print(f"  I.inf <= Z2.inf: {I.inf.flatten() <= Z2_int.inf.flatten()}")
print(f"  I.sup >= Z2.sup: {I.sup.flatten() >= Z2_int.sup.flatten()}")

# Try interval.contains_ directly
print("\nCalling I.contains_(Z2)...")
res, cert, scaling = I.contains_(Z2, 'exact', 1e-12, 200, True, True)
print(f"  Result: {res}, cert: {cert}, scaling: {scaling}")

# Try zonotope.contains_ 
print("\nCalling Z1.contains_(Z2)...")
res2, cert2, scaling2 = Z1.contains_(Z2, 'exact', 1e-12, 200, True, True)
print(f"  Result: {res2}, cert: {cert2}, scaling: {scaling2}")

