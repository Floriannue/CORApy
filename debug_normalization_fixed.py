import numpy as np
import sys
import os

# Add the cora_python path
sys.path.insert(0, os.path.join(os.getcwd(), 'cora_python'))

# Import the fixed function
from cora_python.contSet.polytope.private.priv_normalizeConstraints import priv_normalizeConstraints

# Test the normalization logic
A = np.array([[1, 0], [-1, 1], [-1, -1]])
b = np.array([7, -2, -4]).reshape(-1, 1)

print("Original A:", A)
print("Original b:", b.flatten())

# Test the fixed normalization function
A_norm, b_norm, Ae_norm, be_norm = priv_normalizeConstraints(A, b, None, None, 'A')

print("\nAfter normalization:")
print("A_norm:", A_norm)
print("b_norm:", b_norm.flatten())

# Check if normalization worked correctly
norms_after = np.linalg.norm(A_norm, axis=1)
print("Norms after normalization:", norms_after)
print("All norms should be 1.0:", np.allclose(norms_after, 1.0))

# Check if any rows became zero
print("Any zero rows:", np.any(np.all(A_norm == 0, axis=1))) 