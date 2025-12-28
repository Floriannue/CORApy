"""Debug polytope contains zonotope"""
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.contSet.supportFunc_ import supportFunc_

# Test case
I = Interval(np.array([1,2]), np.array([4,6]))
Z2 = Zonotope(np.array([[3],[4]]), np.array([[1],[1]]))

print("Interval I:")
print(f"  I.inf: {I.inf.flatten()}")
print(f"  I.sup: {I.sup.flatten()}")

print("\nZonotope Z2:")
print(f"  Z2.c: {Z2.c.flatten()}")
print(f"  Z2.G: {Z2.G.flatten()}")

# Convert interval to polytope
P = I.polytope()
print(f"\nPolytope P from interval:")
print(f"  P.A shape: {P.A.shape}")
print(f"  P.b shape: {P.b.shape}")
print(f"  P.A:\n{P.A}")
print(f"  P.b:\n{P.b}")

# Check containment using polytope.contains_
print("\nCalling P.contains_(Z2, 'exact:polymax')...")
res, cert, scaling = P.contains_(Z2, 'exact:polymax', 1e-12, 200, True, True)
print(f"  Result: {res}, cert: {cert}, scaling: {scaling}")

# Debug support function calculation
print("\nDebugging support function calculation:")
print("  P constraints:")
for i in range(P.A.shape[0]):
    normal = P.A[i, :].reshape(-1, 1)
    b_poly = P.b[i, 0]
    sup_val, _, _ = supportFunc_(Z2, normal, 'upper')
    print(f"    Constraint {i}: normal={normal.flatten()}, b_poly={b_poly:.6f}, sup_Z2={sup_val:.6f}, contained={sup_val <= b_poly + 1e-12}")

