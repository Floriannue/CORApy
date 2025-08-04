import numpy as np
import sys
import os

# Add the cora_python path
sys.path.insert(0, os.path.join(os.getcwd(), 'cora_python'))

# Simple withinTol function to avoid circular imports
def withinTol(a, b, tol=1e-8):
    """Simple withinTol implementation"""
    return np.abs(a - b) <= tol

# Test the normalization logic
A = np.array([[1, 0], [-1, 1], [-1, -1]])
b = np.array([7, -2, -4]).reshape(-1, 1)

print("Original A:", A)
print("Original b:", b.flatten())

# Compute norms
normA = np.linalg.norm(A, axis=1)
print("Norms:", normA)

# Check which norms are non-zero using our simple withinTol
idx_nonzero = ~withinTol(normA, 0)
print("Non-zero indices:", idx_nonzero)

# Test the normalization
if np.any(idx_nonzero):
    normA_nonzero = normA[idx_nonzero]
    print("Non-zero norms:", normA_nonzero)
    
    # Use explicit indices to avoid boolean indexing assignment issues
    explicit_indices = np.where(idx_nonzero)[0]
    print("Explicit indices:", explicit_indices)
    
    # Apply normalization
    A_out = A.copy()
    b_out = b.copy()
    
    A_out[explicit_indices, :] = A_out[explicit_indices, :] / normA_nonzero.reshape(-1, 1)
    b_out[explicit_indices] = (b_out[explicit_indices].flatten() / normA_nonzero).reshape(-1, 1)
    
    print("A_out after normalization:", A_out)
    print("b_out after normalization:", b_out.flatten())
    
    # Check if normalization worked correctly
    norms_after = np.linalg.norm(A_out, axis=1)
    print("Norms after normalization:", norms_after)
    print("All norms should be 1.0:", np.allclose(norms_after, 1.0))
    
    # Check if any rows became zero
    print("Any zero rows:", np.any(np.all(A_out == 0, axis=1))) 