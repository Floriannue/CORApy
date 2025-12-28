"""Debug the full contains_ call path"""
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope import Polytope

# Test case 1: test_outer_zonotope_is_interval
print("=" * 70)
print("Test 1: test_outer_zonotope_is_interval")
print("=" * 70)

Z1 = Zonotope(Interval(np.array([1,2]), np.array([4,6])))
Z2 = Zonotope(np.array([[3],[4]]), np.array([[1],[1]]))

print(f"Z1 (interval zonotope):")
print(f"  Z1.c: {Z1.c.flatten()}")
print(f"  Z1.G:\n{Z1.G}")

print(f"\nZ2:")
print(f"  Z2.c: {Z2.c.flatten()}")
print(f"  Z2.G:\n{Z2.G}")

# Check if Z1 represents interval
Z_isInterval, I = Z1.representsa_('interval', 1e-10, return_set=True)
print(f"\nZ1.representsa_('interval'): {Z_isInterval}")
if I is not None:
    print(f"  I.inf: {I.inf.flatten()}")
    print(f"  I.sup: {I.sup.flatten()}")

# Call contains_ with default parameters (scalingToggle=True)
print("\nCalling Z1.contains_(Z2) with defaults...")
print("  Default: method='exact', tol=1e-12, maxEval=200, certToggle=True, scalingToggle=True")
res, cert, scaling = Z1.contains_(Z2)
print(f"  Result: res={res}, cert={cert}, scaling={scaling}")

# Call contains_ with scalingToggle=False
print("\nCalling Z1.contains_(Z2, scalingToggle=False)...")
res2, cert2, scaling2 = Z1.contains_(Z2, scalingToggle=False)
print(f"  Result: res={res2}, cert={cert2}, scaling={scaling2}")

# Test case 2: test_zono_in_zono
print("\n" + "=" * 70)
print("Test 2: test_zono_in_zono")
print("=" * 70)

c1 = np.array([[0], [1]])
G1 = np.array([[1, 2, 1], [-1, 0, 1]])
Z1 = Zonotope(c1, G1)

c2 = np.array([[-1], [1.5]])
G2 = np.array([[0.2, 0], [-0.1, 0.1]])
Z2 = Zonotope(c2, G2)

print(f"Z1:")
print(f"  Z1.c: {Z1.c.flatten()}")
print(f"  Z1.G:\n{Z1.G}")

print(f"\nZ2:")
print(f"  Z2.c: {Z2.c.flatten()}")
print(f"  Z2.G:\n{Z2.G}")

print("\nCalling Z1.contains_(Z2) with defaults...")
res3, cert3, scaling3 = Z1.contains_(Z2)
print(f"  Result: res={res3}, cert={cert3}, scaling={scaling3}")

print("\nCalling Z1.contains_(Z2, scalingToggle=False)...")
res4, cert4, scaling4 = Z1.contains_(Z2, scalingToggle=False)
print(f"  Result: res={res4}, cert={cert4}, scaling={scaling4}")

