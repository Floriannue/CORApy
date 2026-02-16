"""test_rerr1_computation - Test rerr1 computation to verify axis correctness"""

import numpy as np
from cora_python.contSet.zonotope import Zonotope

# Create a test zonotope: 2 dimensions, 3 generators
c = np.array([[0], [0]])
G = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3): 2 dims, 3 generators
Z = Zonotope(c, G)

print("Test zonotope:")
print(f"  Center shape: {c.shape}")
print(f"  Generators shape: {G.shape}")
print(f"  Generators:\n{G}")

# Test different sum operations
print("\nSum operations:")
print(f"  sum(abs(G), axis=0) (sum rows, reduce dim): {np.sum(np.abs(G), axis=0)}")
print(f"  sum(abs(G), axis=1) (sum cols, reduce generators): {np.sum(np.abs(G), axis=1)}")

# MATLAB equivalent: sum(abs(generators(Rerr)),2)
# This sums along dimension 2 (columns), producing (dim, 1)
# In Python, this is axis=1 (columns)
matlab_equivalent = np.sum(np.abs(G), axis=1)
print(f"\nMATLAB sum(...,2) equivalent (axis=1): {matlab_equivalent}")

# MATLAB: vecnorm(...,2) computes 2-norm along dimension 2
# For a (dim, 1) vector, this is just the 2-norm of the vector
matlab_result = np.linalg.norm(matlab_equivalent, 2)
print(f"MATLAB vecnorm result: {matlab_result}")

# Test with actual zonotope
rerr1_axis0 = np.linalg.norm(np.sum(np.abs(Z.generators()), axis=0), 2)
rerr1_axis1 = np.linalg.norm(np.sum(np.abs(Z.generators()), axis=1), 2)

print(f"\nWith Zonotope:")
print(f"  rerr1 with axis=0: {rerr1_axis0}")
print(f"  rerr1 with axis=1: {rerr1_axis1}")
print(f"  Correct (MATLAB equivalent): axis=1 = {rerr1_axis1}")
