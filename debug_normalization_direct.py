import numpy as np

# Simple withinTol function to avoid circular imports
def withinTol(a, b, tol=1e-8):
    """Simple withinTol implementation"""
    return np.abs(a - b) <= tol

# Test the normalization logic directly
A = np.array([[1, 0], [-1, 1], [-1, -1]])
b = np.array([7, -2, -4]).reshape(-1, 1)

print("Original A:", A)
print("Original b:", b.flatten())

# Compute norms
normA = np.linalg.norm(A, axis=1)
print("Norms:", normA)

# Check which norms are non-zero
idx_nonzero = ~withinTol(normA, 0)
print("Non-zero indices:", idx_nonzero)
print("Non-zero indices as array:", np.array(idx_nonzero))

# Test the normalization
if np.any(idx_nonzero):
    normA_nonzero = normA[idx_nonzero]
    print("Non-zero norms:", normA_nonzero)
    
    # Get the rows to normalize
    A_nonzero = A[idx_nonzero, :]
    print("A_nonzero:", A_nonzero)
    
    # Normalize using the correct approach
    A_normalized = A_nonzero / normA_nonzero.reshape(-1, 1)
    print("A_normalized:", A_normalized)
    
    # Check norms after normalization
    norms_after = np.linalg.norm(A_normalized, axis=1)
    print("Norms after normalization:", norms_after)
    print("All norms should be 1.0:", np.allclose(norms_after, 1.0))
    
    # Now test the full function logic
    print("\n=== Testing full function logic ===")
    A_out = A.copy()
    b_out = b.copy()
    
    # Apply normalization to the full matrix
    A_out[idx_nonzero, :] = A_out[idx_nonzero, :] / normA_nonzero.reshape(-1, 1)
    b_out[idx_nonzero] = (b_out[idx_nonzero].flatten() / normA_nonzero).reshape(-1, 1)
    
    print("A_out after normalization:", A_out)
    print("b_out after normalization:", b_out.flatten())
    
    # Check final norms
    final_norms = np.linalg.norm(A_out, axis=1)
    print("Final norms:", final_norms)
    print("Any zero rows:", np.any(np.all(A_out == 0, axis=1))) 