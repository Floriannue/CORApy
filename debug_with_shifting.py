"""Debug with shifting (scalingToggle=True)"""
import numpy as np
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.interval import Interval
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.contSet.supportFunc_ import supportFunc_
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.contSet.polytope.private.priv_equality_to_inequality import priv_equality_to_inequality
from cora_python.contSet.polytope.private.priv_normalizeConstraints import priv_normalizeConstraints
from cora_python.contSet.polytope.private.priv_compact_all import priv_compact_all

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

# Simulate scalingToggle=True: shift to origin
scalingToggle = True
c = P1.center()
print(f"P1 center: {c.flatten()}")
P_shifted = P1 - c
S_shifted = Z2 - c

print(f"\nAfter shifting:")
print(f"P_shifted center: {P_shifted.center().flatten()}")
print(f"S_shifted (Z2) center: {S_shifted.c.flatten()}")

# Now do the constraint check
combined_A, combined_b = priv_equality_to_inequality(P_shifted.A, P_shifted.b, P_shifted.Ae, P_shifted.be)
n_dim = P_shifted.dim()
A_n, b_n, Ae_n, be_n = priv_normalizeConstraints(combined_A, combined_b.reshape(-1, 1), np.zeros((0, n_dim)), np.zeros((0, 1)), 'A')
A_n, b_n, Ae_n, be_n, _, _ = priv_compact_all(A_n, b_n, Ae_n, be_n, n_dim, tol)
combined_A = A_n
combined_b = b_n.reshape(-1, 1)

print(f"\nAfter normalization and compaction:")
print(f"combined_A:\n{combined_A}")
print(f"combined_b:\n{combined_b.flatten()}")

scaling = 0.0
res = True

print("\n=== Constraint loop (with shifting) ===")
for i in range(combined_b.shape[0]):
    normal_vector = combined_A[i,:].reshape(-1,1)
    b_polytope = combined_b[i,0]
    
    b_set_sup_val, _, _ = supportFunc_(S_shifted, normal_vector, 'upper')
    
    condition = b_set_sup_val > b_polytope and not withinTol(b_polytope, b_set_sup_val, tol)
    
    if i < 2:  # Print first 2 constraints
        print(f"\nConstraint {i}:")
        print(f"  Normal: {normal_vector.flatten()}")
        print(f"  b_polytope: {b_polytope}")
        print(f"  b_set_sup_val: {b_set_sup_val}")
        print(f"  Violation: {condition}")
    
    if condition:
        res = False
        if not scalingToggle:
            break
        else:
            scaling_current = b_set_sup_val / b_polytope
            if np.isnan(scaling_current):
                scaling_current = 0.0
            if scaling_current > scaling:
                scaling = scaling_current

print(f"\nFinal result: {res}")
print(f"Final scaling: {scaling}")

