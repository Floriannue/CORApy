"""Test the full compact flow to understand constraint removal"""
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope.polytope import Polytope

# Create the same degenerate case
I = Interval(np.array([[5.], [-1.]]), np.array([[5.], [1.]]))
P = I.polytope()

# Shift to origin
c = P.center()
P_shifted = P - c
P_shifted.constraints()

print("=== Original constraints after shifting ===")
print(f"A:\n{P_shifted.A}")
print(f"b: {P_shifted.b.flatten()}")
print(f"Constraints:")
for i in range(P_shifted.A.shape[0]):
    print(f"  {P_shifted.A[i]} x <= {P_shifted.b[i]}")

# Now compact
P_compact = P_shifted.compact_()
P_compact.constraints()

print(f"\n=== After compact ===")
print(f"A:\n{P_compact.A}")
print(f"b: {P_compact.b.flatten()}")
print(f"Ae:\n{P_compact.Ae}")
print(f"be: {P_compact.be.flatten()}")
print(f"Constraints:")
for i in range(P_compact.A.shape[0]):
    print(f"  {P_compact.A[i]} x <= {P_compact.b[i]}")
if P_compact.Ae.size > 0:
    for i in range(P_compact.Ae.shape[0]):
        print(f"  {P_compact.Ae[i]} x = {P_compact.be[i]}")

print(f"\n=== Analysis ===")
print(f"Original: {P_shifted.A.shape[0]} inequality constraints")
print(f"After compact: {P_compact.A.shape[0]} inequality constraints, {P_compact.Ae.shape[0]} equality constraints")
print(f"Total removed: {P_shifted.A.shape[0] - P_compact.A.shape[0]} inequality constraints")

# The x-direction constraints [1, 0] x <= 0 and [-1, 0] x <= 0
# should have been converted to [1, 0] x = 0 (equality constraint)
# But we see no equality constraints in the result - why?

