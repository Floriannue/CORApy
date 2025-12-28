"""Debug -1 * P to ensure constraints are correct"""
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope.polytope import Polytope

# Create a polytope from an interval (like in the contains_ case)
I = Interval(np.array([[5.], [-1.]]), np.array([[5.], [1.]]))
P = I.polytope()
P.constraints()

print("Original P:")
print(f"  A:\n{P.A}")
print(f"  b:\n{P.b.flatten()}")
print(f"  Ae:\n{P.Ae}")
print(f"  be:\n{P.be.flatten()}")

# Test -1 * P
P_neg = -1 * P
P_neg.constraints()

print("\n-1 * P:")
print(f"  A:\n{P_neg.A}")
print(f"  b:\n{P_neg.b.flatten()}")
print(f"  Ae:\n{P_neg.Ae}")
print(f"  be:\n{P_neg.be.flatten()}")

# Test P - c (shifting)
c = P.center()
print(f"\nP center: {c.flatten()}")
P_shifted = P - c
P_shifted.constraints()

print("\nP - c:")
print(f"  A:\n{P_shifted.A}")
print(f"  b:\n{P_shifted.b.flatten()}")
print(f"  Ae:\n{P_shifted.Ae}")
print(f"  be:\n{P_shifted.be.flatten()}")

# Test -1 * (P - c) (combination)
P_neg_shifted = -1 * (P - c)
P_neg_shifted.constraints()

print("\n-1 * (P - c):")
print(f"  A:\n{P_neg_shifted.A}")
print(f"  b:\n{P_neg_shifted.b.flatten()}")
print(f"  Ae:\n{P_neg_shifted.Ae}")
print(f"  be:\n{P_neg_shifted.be.flatten()}")

# Expected: -1 * (P - c) should be equivalent to (-P) - (-c) = -P + c
# But we want to check if constraints are mathematically correct

