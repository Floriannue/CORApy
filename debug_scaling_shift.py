"""Debug the scaling shift issue"""
import numpy as np
from cora_python.contSet.interval import Interval
from cora_python.contSet.zonotope import Zonotope
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.contSet.supportFunc_ import supportFunc_
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol
from cora_python.contSet.polytope.private.priv_equality_to_inequality import priv_equality_to_inequality
from cora_python.contSet.polytope.private.priv_normalizeConstraints import priv_normalizeConstraints
from cora_python.contSet.polytope.private.priv_compact_all import priv_compact_all

# Test case 1
I = Interval(np.array([1,2]), np.array([4,6]))
Z2 = Zonotope(np.array([[3],[4]]), np.array([[1],[1]]))

P = I.polytope()
print("Test case: Interval contains Zonotope")
print(f"I: inf={I.inf.flatten()}, sup={I.sup.flatten()}")
print(f"Z2: c={Z2.c.flatten()}, G={Z2.G.flatten()}")

# Without shifting (scalingToggle=False)
print("\n" + "="*70)
print("WITHOUT SHIFTING (scalingToggle=False):")
print("="*70)

P_shifted = P
S_shifted = Z2

combined_A, combined_b = priv_equality_to_inequality(P_shifted.A, P_shifted.b, P_shifted.Ae, P_shifted.be)
n_dim = P_shifted.dim()
A_n, b_n, Ae_n, be_n = priv_normalizeConstraints(combined_A, combined_b.reshape(-1, 1), np.zeros((0, n_dim)), np.zeros((0, 1)), 'A')
A_n, b_n, Ae_n, be_n, _, _ = priv_compact_all(A_n, b_n, Ae_n, be_n, n_dim, 1e-12)
combined_A = A_n
combined_b = b_n.reshape(-1, 1)

print(f"Polytope constraints (after normalization/compaction):")
for i in range(combined_b.shape[0]):
    normal = combined_A[i, :].reshape(-1, 1)
    b_poly = combined_b[i, 0]
    sup_val, _, _ = supportFunc_(S_shifted, normal, 'upper')
    contained = sup_val <= b_poly + 1e-12 or withinTol(b_poly, sup_val, 1e-12)
    print(f"  Constraint {i}: normal={normal.flatten()}, b_poly={b_poly:.6f}, sup_S={sup_val:.6f}, contained={contained}")

# With shifting (scalingToggle=True)
print("\n" + "="*70)
print("WITH SHIFTING (scalingToggle=True):")
print("="*70)

c = P.center()
print(f"Polytope center: {c.flatten()}")

P_shifted = P - c
S_shifted = Z2 - c

print(f"\nAfter shifting:")
print(f"  P_shifted center: {P_shifted.center().flatten()}")
print(f"  S_shifted (Z2) center: {S_shifted.c.flatten()}")
print(f"  S_shifted (Z2) G:\n{S_shifted.G}")

combined_A, combined_b = priv_equality_to_inequality(P_shifted.A, P_shifted.b, P_shifted.Ae, P_shifted.be)
n_dim = P_shifted.dim()
A_n, b_n, Ae_n, be_n = priv_normalizeConstraints(combined_A, combined_b.reshape(-1, 1), np.zeros((0, n_dim)), np.zeros((0, 1)), 'A')
A_n, b_n, Ae_n, be_n, _, _ = priv_compact_all(A_n, b_n, Ae_n, be_n, n_dim, 1e-12)
combined_A = A_n
combined_b = b_n.reshape(-1, 1)

print(f"\nPolytope constraints (after shifting and normalization/compaction):")
for i in range(combined_b.shape[0]):
    normal = combined_A[i, :].reshape(-1, 1)
    b_poly = combined_b[i, 0]
    sup_val, _, _ = supportFunc_(S_shifted, normal, 'upper')
    contained = sup_val <= b_poly + 1e-12 or withinTol(b_poly, sup_val, 1e-12)
    print(f"  Constraint {i}: normal={normal.flatten()}, b_poly={b_poly:.6f}, sup_S={sup_val:.6f}, contained={contained}")
    if not contained:
        print(f"    *** VIOLATION: sup_S={sup_val:.10f} > b_poly={b_poly:.10f}")

