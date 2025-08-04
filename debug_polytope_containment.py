import numpy as np
import sys
import os

# Add the cora_python path
sys.path.insert(0, os.path.join(os.getcwd(), 'cora_python'))

from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.polytope import Polytope

# Test the failing case
Q = np.array([[2, 0], [0, 1]])
q = np.array([[2], [-1]])
M = np.array([[2, -1], [1, 1]])

# Create the ellipsoid
E = M @ Ellipsoid(Q, q)

# Create the polytope that should be outside
A_outside = np.array([[1, 0], [-1, 1], [-1, -1]])
b_outside = np.array([[7], [-2], [-4]])
P_outside = Polytope(A_outside, b_outside)

print("Ellipsoid E:")
print(f"Q: {E.Q}")
print(f"q: {E.q}")

print("\nPolytope P_outside:")
print(f"A: {P_outside.A}")
print(f"b: {P_outside.b}")

# Check if polytope is empty
print(f"\nPolytope empty: {P_outside.emptySet}")

# Check polytope vertices
try:
    vertices = P_outside.vertices_()
    print(f"Polytope vertices: {vertices}")
    print(f"Number of vertices: {vertices.shape[1] if vertices.size > 0 else 0}")
except Exception as e:
    print(f"Error getting vertices: {e}")

# Test containment
result = E.contains(P_outside)
print(f"\nE.contains(P_outside) = {result}")

# Let's also test the "inside" polytope for comparison
A_inside = np.array([[1, 0], [-1, 1], [-1, -1]])
b_inside = np.array([[4], [-2.2], [-4]])
P_inside = Polytope(A_inside, b_inside)

print(f"\nPolytope P_inside empty: {P_inside.emptySet}")
result_inside = E.contains(P_inside)
print(f"E.contains(P_inside) = {result_inside}") 