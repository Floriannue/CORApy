import numpy as np
from cora_python.contSet.polytope.polytope import Polytope

# Test 1D unbounded & unbounded intersection
A = np.array([[1]])
b = np.array([1])
P1 = Polytope(A, b)

A = np.array([[-1]])
b = np.array([5])
P2 = Polytope(A, b)

print("P1:", P1)
print("P2:", P2)
print("P1 is empty:", P1.representsa_('emptySet', 0))
print("P2 is empty:", P2.representsa_('emptySet', 0))

# P1: x <= 1
# P2: -x <= 5 => x >= -5
# Intersection: -5 <= x <= 1 (should intersect)

# Debug P2 constraints
print("\nP2 debug:")
print("A:", P2.A)
print("b:", P2.b)
print("Ae:", P2.Ae if hasattr(P2, 'Ae') else None)
print("be:", P2.be if hasattr(P2, 'be') else None)

# Test vertices computation for P2
from cora_python.contSet.polytope.private.priv_vertices_1D import priv_vertices_1D
vertices_P2 = priv_vertices_1D(P2.A, P2.b, None, None)
print("P2 vertices:", vertices_P2)

# Test P1 vertices too
vertices_P1 = priv_vertices_1D(P1.A, P1.b, None, None)
print("P1 vertices:", vertices_P1)

result = P1.isIntersecting_(P2, 'exact')
print("Intersection result:", result)

# Let's also test approximate
result_approx = P1.isIntersecting_(P2, 'approx')
print("Approximate result:", result_approx) 