import numpy as np

# Test the normalization logic step by step
A = np.array([[1, 0], [-1, 1], [-1, -1]])
b = np.array([7, -2, -4]).reshape(-1, 1)

print("Original A:", A)
print("Original b:", b.flatten())

# Compute norms
normA = np.linalg.norm(A, axis=1)
print("Norms:", normA)

# Check which norms are non-zero
tol = 1e-12
idx_nonzero = normA > tol
print("Non-zero indices:", idx_nonzero)

# Test the normalization
if np.any(idx_nonzero):
    normA_nonzero = normA[idx_nonzero]
    print("Non-zero norms:", normA_nonzero)
    print("Non-zero norms shape:", normA_nonzero.shape)
    
    # Get the rows to normalize
    A_nonzero = A[idx_nonzero, :]
    print("A_nonzero:", A_nonzero)
    print("A_nonzero shape:", A_nonzero.shape)
    
    # Test different normalization approaches
    print("\n=== Approach 1: Transpose-divide-transpose ===")
    A_norm1 = (A_nonzero.T / normA_nonzero).T
    print("Result:", A_norm1)
    
    print("\n=== Approach 2: Direct division with reshape ===")
    # Reshape normA_nonzero to (n, 1) for broadcasting
    normA_reshaped = normA_nonzero.reshape(-1, 1)
    print("normA_reshaped:", normA_reshaped)
    print("normA_reshaped shape:", normA_reshaped.shape)
    A_norm2 = A_nonzero / normA_reshaped
    print("Result:", A_norm2)
    
    print("\n=== Approach 3: Manual loop ===")
    A_norm3 = A_nonzero.copy()
    for i in range(len(normA_nonzero)):
        A_norm3[i, :] = A_nonzero[i, :] / normA_nonzero[i]
    print("Result:", A_norm3)
    
    # Check which approach gives correct results
    print("\n=== Verification ===")
    print("Approach 1 norms:", np.linalg.norm(A_norm1, axis=1))
    print("Approach 2 norms:", np.linalg.norm(A_norm2, axis=1))
    print("Approach 3 norms:", np.linalg.norm(A_norm3, axis=1))
    
    # Expected: all should be 1.0
    print("All should be 1.0") 