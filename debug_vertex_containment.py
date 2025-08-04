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

# Get the vertices
vertices = P_outside.vertices_()
print("Polytope vertices:")
print(vertices)

# Check each vertex individually
print("\nChecking each vertex:")
for i in range(vertices.shape[1]):
    vertex = vertices[:, i:i+1]
    print(f"Vertex {i}: {vertex.flatten()}")
    
    # Check if this vertex is inside the ellipsoid
    # For an ellipsoid (x-q)^T Q^(-1) (x-q) <= 1
    diff = vertex - E.q
    Q_inv = np.linalg.inv(E.Q)
    distance_squared = diff.T @ Q_inv @ diff
    print(f"  Distance squared: {distance_squared[0,0]}")
    print(f"  Inside ellipsoid: {distance_squared[0,0] <= 1}")

# Let's also check the ellipsoid center and some points
print(f"\nEllipsoid center: {E.q.flatten()}")
print(f"Ellipsoid Q matrix: {E.Q}")

# Check a point that should definitely be inside (the center)
center_check = (E.q - E.q).T @ np.linalg.inv(E.Q) @ (E.q - E.q)
print(f"Center distance squared: {center_check[0,0]}")

# Check a point that should be outside (far from center)
far_point = np.array([[20], [20]])
far_check = (far_point - E.q).T @ np.linalg.inv(E.Q) @ (far_point - E.q)
print(f"Far point distance squared: {far_check[0,0]}") 