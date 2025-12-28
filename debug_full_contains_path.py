"""Debug the full contains path to see where it fails"""
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

# Get interval representation
isInterval, I = Z1.representsa_('interval', 1e-10, return_set=True)
P = I.polytope()
P.constraints()

print("=== Step 1: Original P ===")
print(f"A:\n{P.A}")
print(f"b: {P.b.flatten()}")

# Step 2: Shift to origin (scalingToggle=True)
scalingToggle = True
c = P.center()
print(f"\n=== Step 2: Center ===")
print(f"c: {c.flatten()}")

P_shifted = P - c
S_shifted = Z2 - c

P_shifted.constraints()
print(f"\n=== Step 3: After shifting ===")
print(f"P_shifted A:\n{P_shifted.A}")
print(f"P_shifted b: {P_shifted.b.flatten()}")
print(f"S_shifted center: {S_shifted.c.flatten()}")

# Step 3: Combine and normalize
combined_A, combined_b = priv_equality_to_inequality(P_shifted.A, P_shifted.b, P_shifted.Ae, P_shifted.be)
print(f"\n=== Step 4: After equality_to_inequality ===")
print(f"combined_A:\n{combined_A}")
print(f"combined_b: {combined_b.flatten()}")

n_dim = P_shifted.dim()
A_n, b_n, Ae_n, be_n = priv_normalizeConstraints(combined_A, combined_b.reshape(-1, 1), np.zeros((0, n_dim)), np.zeros((0, 1)), 'A')
print(f"\n=== Step 5: After normalizeConstraints ===")
print(f"A_n:\n{A_n}")
print(f"b_n: {b_n.flatten()}")

A_n, b_n, Ae_n, be_n, empty, _ = priv_compact_all(A_n, b_n, Ae_n, be_n, n_dim, 1e-10)
print(f"\n=== Step 6: After compact_all ===")
print(f"A_n:\n{A_n}")
print(f"b_n: {b_n.flatten()}")
print(f"empty: {empty}")

# Step 4: Check constraints
scaling = 0.0
res = True
tol = 1e-10

print(f"\n=== Step 7: Constraint loop ===")
for i in range(b_n.shape[0]):
    normal_vector = A_n[i,:].reshape(-1,1)
    b_polytope = b_n[i,0]
    
    b_set_sup_val, _, _ = supportFunc_(S_shifted, normal_vector, 'upper')
    
    condition = b_set_sup_val > b_polytope and not withinTol(b_polytope, b_set_sup_val, tol)
    
    print(f"\nConstraint {i}:")
    print(f"  Normal: {normal_vector.flatten()}")
    print(f"  b_polytope: {b_polytope}")
    print(f"  b_set_sup_val: {b_set_sup_val}")
    print(f"  Violation: {condition}")
    
    if condition:
        print(f"  *** SETTING res = False ***")
        res = False
        if not scalingToggle:
            print(f"  Early return")
            break
        else:
            if abs(b_polytope) < tol:
                scaling_current = np.inf
            else:
                scaling_current = b_set_sup_val / b_polytope
            if np.isnan(scaling_current):
                scaling_current = 0.0
            if scaling_current > scaling:
                scaling = scaling_current
            print(f"  Updated scaling: {scaling}")

print(f"\n=== Final Result ===")
print(f"res: {res}")
print(f"scaling: {scaling}")

