"""Debug support function values for degenerate case"""
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.contSet.supportFunc_ import supportFunc_

# Test case
c1 = np.array([[5.000], [0.000]])
G1 = np.array([[0.000], [1.000]])
Z1 = Zonotope(c1, G1)

c2 = np.array([[5.650], [0.000]])
G2 = np.array([[0.000, 0.050, 0.000, 0.000, 0.000],
               [0.937, 0.000, -0.005, -0.000, 0.000]])
Z2 = Zonotope(c2, G2)

# Buffer Z1
tol = 1e-10
I = tol * Interval(-np.ones((2, 1)), np.ones((2, 1)))
Z1_buffered = Z1 + I

# Convert to polytope
P1 = Polytope(Z1_buffered)
P1.constraints()  # Force H-rep

print("P1 constraints:")
print(f"A shape: {P1.A.shape}")
print(f"b shape: {P1.b.shape}")
print(f"A:\n{P1.A}")
print(f"b:\n{P1.b.flatten()}")

# Check support functions along each constraint normal
print("\n=== Support function comparison ===")
for i in range(P1.A.shape[0]):
    normal = P1.A[i, :].reshape(-1, 1)
    b_polytope = P1.b[i, 0]
    
    # Support function of Z2 along this normal
    b_z2, _, _ = supportFunc_(Z2, normal, 'upper')
    
    print(f"\nConstraint {i}:")
    print(f"  Normal: {normal.flatten()}")
    print(f"  P1 bound (b): {b_polytope}")
    print(f"  Z2 support func: {b_z2}")
    print(f"  Contained? {b_z2 <= b_polytope}")
    print(f"  Within tol? {abs(b_z2 - b_polytope) <= 1e-10}")

