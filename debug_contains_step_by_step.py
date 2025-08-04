import numpy as np
import sys
import os

# Add the cora_python path
sys.path.insert(0, os.path.join(os.getcwd(), 'cora_python'))

from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.polytope import Polytope
from cora_python.contSet.ellipsoid.private.priv_containsPoint import priv_containsPoint

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

# Get the vertices
vertices = P_outside.vertices_()
print("Polytope vertices:")
print(vertices)

# Test priv_containsPoint directly
print("\nTesting priv_containsPoint:")
res, cert, scaling = priv_containsPoint(E, vertices, 1e-9)
print(f"res: {res}")
print(f"cert: {cert}")
print(f"scaling: {scaling}")

# Check if all vertices are contained
all_contained = np.all(res)
print(f"All vertices contained: {all_contained}")

# Test the ellipsoid norm for each vertex
print("\nTesting ellipsoid norm for each vertex:")
for i in range(vertices.shape[1]):
    vertex = vertices[:, i:i+1]
    norm = E.ellipsoidNorm(vertex - E.center())
    print(f"Vertex {i}: {vertex.flatten()}, norm: {norm}, inside: {norm <= 1}")

# Test the full contains method
print(f"\nTesting E.contains(P_outside):")
result = E.contains(P_outside)
print(f"Result: {result}") 