"""Debug polytope containment with exact constraints"""
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.contSet.supportFunc_ import supportFunc_
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

# Test case
c1 = np.array([[5.000], [0.000]])
G1 = np.array([[0.000], [1.000]])
Z1 = Zonotope(c1, G1)

c2 = np.array([[5.650], [0.000]])
G2 = np.array([[0.000, 0.050, 0.000, 0.000, 0.000],
               [0.937, 0.000, -0.005, -0.000, 0.000]])
Z2 = Zonotope(c2, G2)

# Get interval and convert to polytope
isInterval, I = Z1.representsa_('interval', 1e-10, return_set=True)
P = I.polytope()
P.constraints()

print("P constraints:")
print(f"A:\n{P.A}")
print(f"b:\n{P.b.flatten()}")

# Check with scalingToggle=True (shifting happens)
scalingToggle = True
c = P.center()
print(f"\nP center: {c.flatten()}")
P_shifted = P - c
Z2_shifted = Z2 - c

print(f"\nAfter shifting:")
print(f"P_shifted center: {P_shifted.center().flatten()}")
print(f"Z2_shifted center: {Z2_shifted.c.flatten()}")

# Check constraints of shifted polytope
P_shifted.constraints()
print(f"\nP_shifted constraints:")
print(f"A:\n{P_shifted.A}")
print(f"b:\n{P_shifted.b.flatten()}")

# Check support functions
print("\n=== Support function check ===")
for i in range(P_shifted.A.shape[0]):
    normal = P_shifted.A[i, :].reshape(-1, 1)
    b_polytope = P_shifted.b[i, 0]
    b_z2, _, _ = supportFunc_(Z2_shifted, normal, 'upper')
    
    print(f"\nConstraint {i}:")
    print(f"  Normal: {normal.flatten()}")
    print(f"  b_polytope: {b_polytope}")
    print(f"  b_z2: {b_z2}")
    print(f"  Contained? {b_z2 <= b_polytope}")
    print(f"  Within tol? {withinTol(b_polytope, b_z2, 1e-10)}")

