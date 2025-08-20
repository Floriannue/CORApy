import numpy as np
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.contains_ import contains_, _aux_contains_Hpoly_pointcloud

# Create a degenerate polytope: x = -3
Ae = np.array([[1]])
be = np.array([-3])
P1 = Polytope(np.zeros((0, 1)), np.zeros(0), Ae, be)

# Test point containment
S = np.array([[-3]])
print("Testing contains_(P1, S.T)")
print("P1:", P1)
print("S.T shape:", S.T.shape)
print("S.T:", S.T)

# Check the constraints before normalization
print("\nBefore normalization:")
print("P1.A:", P1.A)
print("P1.b:", P1.b)
print("P1.Ae:", P1.Ae)
print("P1.be:", P1.be)

# Force H-representation
P1.constraints()
print("\nAfter P1.constraints():")
print("P1.A:", P1.A)
print("P1.b:", P1.b)
print("P1.Ae:", P1.Ae)
print("P1.be:", P1.be)

# Test the normalization functions
from cora_python.contSet.polytope.private.priv_normalizeConstraints import priv_normalizeConstraints
from cora_python.contSet.polytope.private.priv_compact_all import priv_compact_all

A = P1.A; b = P1.b.reshape(-1, 1); Ae = P1.Ae; be = P1.be.reshape(-1, 1)
print("\nBefore priv_normalizeConstraints:")
print("A:", A)
print("b:", b)
print("Ae:", Ae)
print("be:", be)

A, b, Ae, be = priv_normalizeConstraints(A, b, Ae, be, 'A')
print("\nAfter priv_normalizeConstraints:")
print("A:", A)
print("b:", b)
print("Ae:", Ae)
print("be:", be)

A, b, Ae, be, _, _ = priv_compact_all(A, b, Ae, be, P1.dim(), 1e-12)
print("\nAfter priv_compact_all:")
print("A:", A)
print("b:", b)
print("Ae:", Ae)
print("be:", be)

# Test the point containment manually
v = S.T.reshape(-1, 1)
print("\nTesting point containment manually:")
print("v:", v)
print("Ae @ v:", Ae @ v)
print("be:", be)
print("eq_res = Ae @ v - be:", Ae @ v - be)
print("np.abs(eq_res):", np.abs(Ae @ v - be))
print("np.any(np.abs(eq_res) > 1e-12):", np.any(np.abs(Ae @ v - be) > 1e-12))

# Test _aux_contains_Hpoly_pointcloud directly
print("\nTesting _aux_contains_Hpoly_pointcloud directly:")
P_temp = Polytope(np.zeros((0, 1)), np.zeros(0), Ae.copy(), be.copy().flatten())
res_direct, cert_direct, scaling_direct = _aux_contains_Hpoly_pointcloud(P_temp, S.T, 1e-12, False)
print("Direct result:", res_direct)
print("Direct cert:", cert_direct)
print("Direct scaling:", scaling_direct)

# Check if P1 is identified as empty
print("\nChecking P1.representsa_('emptySet', 1e-12):")

# Let's manually test the linprog call that representsa_ would make
from scipy.optimize import linprog
print("Manual linprog test:")
print("P1.A:", P1.A)
print("P1.b:", P1.b)
print("P1.Ae:", P1.Ae)  
print("P1.be:", P1.be)

n = P1.dim()
c = np.zeros(n)
A = P1.A; b = P1.b.flatten(); Ae = P1.Ae; be = P1.be.flatten()
print("A for linprog:", A)
print("b for linprog:", b)
print("Ae for linprog:", Ae)
print("be for linprog:", be)

res_lp = linprog(c, A_ub=A if A.size > 0 else None, b_ub=b if b.size > 0 else None,
                 A_eq=Ae if Ae.size > 0 else None, b_eq=be if be.size > 0 else None,
                 bounds=None)
print("Linprog result:", res_lp)
print("Linprog success:", res_lp.success)
print("Linprog status:", getattr(res_lp, 'status', None))

print("P1.representsa_('emptySet', 1e-12):", P1.representsa_('emptySet', 1e-12))

# Check if isinstance(S.T, np.ndarray) works correctly
print("\nChecking isinstance(S.T, np.ndarray):")
print("isinstance(S.T, np.ndarray):", isinstance(S.T, np.ndarray))
print("S.T.shape:", S.T.shape)
print("P1.dim():", P1.dim())

try:
    result = contains_(P1, S.T)
    print("\nResult:", result)
    print("Type:", type(result))
    print("First element:", result[0])
    print("First element type:", type(result[0]))
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
