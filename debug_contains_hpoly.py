"""Debug _aux_contains_P_Hpoly"""
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.polytope.contains_ import _aux_contains_P_Hpoly
from cora_python.contSet.polytope.private.priv_equality_to_inequality import priv_equality_to_inequality
from cora_python.contSet.polytope.private.priv_normalizeConstraints import priv_normalizeConstraints
from cora_python.contSet.polytope.private.priv_compact_all import priv_compact_all
from cora_python.contSet.contSet.supportFunc_ import supportFunc_
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

# Test case
I = Interval(np.array([1,2]), np.array([4,6]))
Z2 = Zonotope(np.array([[3],[4]]), np.array([[1],[1]]))

P = I.polytope()
print("Polytope P:")
print(f"  P.A:\n{P.A}")
print(f"  P.b:\n{P.b}")

# Simulate _aux_contains_P_Hpoly logic
P_shifted = P
S_shifted = Z2

# Get combined A, b
combined_A, combined_b = priv_equality_to_inequality(P_shifted.A, P_shifted.b, P_shifted.Ae, P_shifted.be)
print(f"\nAfter equality_to_inequality:")
print(f"  combined_A shape: {combined_A.shape}")
print(f"  combined_b shape: {combined_b.shape}")

# Normalize and compact
n_dim = P_shifted.dim()
A_n, b_n, Ae_n, be_n = priv_normalizeConstraints(combined_A, combined_b.reshape(-1, 1), np.zeros((0, n_dim)), np.zeros((0, 1)), 'A')
print(f"\nAfter normalizeConstraints:")
print(f"  A_n shape: {A_n.shape}")
print(f"  b_n shape: {b_n.shape}")

A_n, b_n, Ae_n, be_n, _, _ = priv_compact_all(A_n, b_n, Ae_n, be_n, n_dim, 1e-12)
print(f"\nAfter compact_all:")
print(f"  A_n shape: {A_n.shape}")
print(f"  b_n shape: {b_n.shape}")
print(f"  A_n:\n{A_n}")
print(f"  b_n:\n{b_n}")

combined_A = A_n
combined_b = b_n.reshape(-1, 1)

scaling = 0.0
res = True

# Loop over all constraints
for i in range(combined_b.shape[0]):
    normal_vector = combined_A[i,:].reshape(-1,1)
    b_polytope = combined_b[i,0]
    
    # Compute support function of S along the normal vector
    b_set_sup_val, _, _ = supportFunc_(S_shifted, normal_vector, 'upper')
    
    print(f"\nConstraint {i}:")
    print(f"  normal: {normal_vector.flatten()}")
    print(f"  b_polytope: {b_polytope:.10f}")
    print(f"  b_set_sup_val: {b_set_sup_val:.10f}")
    print(f"  b_set_sup_val > b_polytope: {b_set_sup_val > b_polytope}")
    print(f"  withinTol: {withinTol(b_polytope, b_set_sup_val, 1e-12)}")
    
    # Check for containment condition
    if b_set_sup_val > b_polytope and not withinTol(b_polytope, b_set_sup_val, 1e-12):
        print(f"  -> NOT CONTAINED!")
        res = False
        scaling_current = b_set_sup_val / b_polytope
        if scaling_current > scaling:
            scaling = scaling_current
    else:
        print(f"  -> contained")

print(f"\nFinal result: res={res}, scaling={scaling}")

