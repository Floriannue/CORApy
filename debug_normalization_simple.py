import numpy as np

# Test the normalization logic
A = np.array([[1, 0], [-1, 1], [-1, -1]])
b = np.array([7, -2, -4]).reshape(-1, 1)

print("Original A:", A)
print("Original b:", b.flatten())

# Compute norms
normA = np.linalg.norm(A, axis=1)
print("Norms:", normA)

# Check which norms are non-zero (using simple tolerance)
tol = 1e-12
idx_nonzero = normA > tol
print("Non-zero indices:", idx_nonzero)

# Test the normalization
if np.any(idx_nonzero):
    normA_nonzero = normA[idx_nonzero]
    print("Non-zero norms:", normA_nonzero)
    
    # Normalize A
    A_normalized = A.copy()
    A_normalized[idx_nonzero, :] = (A_normalized[idx_nonzero, :].T / normA_nonzero).T
    print("Normalized A:", A_normalized)
    
    # Normalize b
    b_normalized = b.copy()
    b_normalized[idx_nonzero] = (b_normalized[idx_nonzero].flatten() / normA_nonzero).reshape(-1, 1)
    print("Normalized b:", b_normalized.flatten())
    
    # Check if any rows became zero
    norms_after = np.linalg.norm(A_normalized, axis=1)
    print("Norms after normalization:", norms_after)
    print("Any zero norms after normalization:", np.any(norms_after < tol)) 