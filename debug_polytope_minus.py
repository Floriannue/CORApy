"""Debug polytope subtraction to verify constraint update"""
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope import Polytope

# Create interval and convert to polytope
I = Interval(np.array([1,2]), np.array([4,6]))
P = I.polytope()

print("Original polytope P:")
print(f"  P.A:\n{P.A}")
print(f"  P.b:\n{P.b}")
print(f"  P.center(): {P.center().flatten()}")

# Subtract center
c = P.center()
P_shifted = P - c

print(f"\nAfter P - c (where c={c.flatten()}):")
print(f"  P_shifted.A:\n{P_shifted.A}")
print(f"  P_shifted.b:\n{P_shifted.b}")
print(f"  P_shifted.center(): {P_shifted.center().flatten()}")

# Verify: For constraint Ax <= b, after shifting by -c, we get A(x - c) <= b
# which expands to Ax <= b + Ac. So b_new should be b + Ac
print("\nVerification:")
for i in range(P.A.shape[0]):
    A_row = P.A[i, :].reshape(-1, 1)
    b_old = P.b[i, 0]
    b_new = P_shifted.b[i, 0]
    b_expected = b_old + (A_row.T @ c)[0, 0]
    print(f"  Constraint {i}:")
    print(f"    A_row: {A_row.flatten()}")
    print(f"    b_old: {b_old:.6f}")
    print(f"    b_new: {b_new:.6f}")
    print(f"    b_expected (b_old + A_row^T @ c): {b_expected:.6f}")
    print(f"    Match: {np.abs(b_new - b_expected) < 1e-10}")

