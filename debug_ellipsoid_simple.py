import numpy as np
import sys
sys.path.append('cora_python')

from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

# Create the same polytope as in the test
Ae = np.array([[1, 0], [0, 1]])
be = np.array([2, 3])
P = Polytope(Ae=Ae, be=be)

print(f"Polytope created: dim={P.dim()}")
print(f"Polytope isHRep: {P.isHRep}")
print(f"Polytope isVRep: {P.isVRep}")

# Try to get vertices
try:
    V = P.vertices_()
    print(f"Vertices shape: {V.shape}")
    print(f"Vertices: {V}")
except Exception as e:
    print(f"Error getting vertices: {e}")

# Debug SVD directly
print("\n=== Debugging SVD directly ===")
points = V  # shape (2, 1)
print(f"Original points: {points}")
print(f"Original points shape: {points.shape}")

# Remove bias
c = np.mean(points, axis=1, keepdims=True)
points_centered = points - c
print(f"After bias removal: {points_centered}")
print(f"Center: {c}")

# SVD
U, S, Vt = np.linalg.svd(points_centered)
print(f"U shape: {U.shape}")
print(f"S (singular values): {S}")
print(f"Vt shape: {Vt.shape}")

# Try different rank calculations
print(f"\nRank calculations:")
print(f"np.linalg.matrix_rank(points_centered): {np.linalg.matrix_rank(points_centered)}")
print(f"np.linalg.matrix_rank(S): {np.linalg.matrix_rank(S)}")
print(f"np.linalg.matrix_rank(S, tol=1e-10): {np.linalg.matrix_rank(S, tol=1e-10)}")
print(f"np.linalg.matrix_rank(S, tol=1e-15): {np.linalg.matrix_rank(S, tol=1e-15)}")
print(f"np.linalg.matrix_rank(S, tol=1e-20): {np.linalg.matrix_rank(S, tol=1e-20)}")
print(f"np.linalg.matrix_rank(S, tol=0): {np.linalg.matrix_rank(S, tol=0)}")

# Try to get ellipsoid
try:
    E_outer = P.ellipsoid('outer')
    print(f"\nEllipsoid created: dim={E_outer.dim()}")
    print(f"Ellipsoid Q shape: {E_outer.Q.shape}")
    print(f"Ellipsoid q shape: {E_outer.q.shape}")
    print(f"Ellipsoid q: {E_outer.q}")
except Exception as e:
    print(f"Error getting ellipsoid: {e}")
    import traceback
    traceback.print_exc()
