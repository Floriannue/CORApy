import numpy as np

# Test boolean indexing
A = np.array([[1, 0], [-1, 1], [-1, -1]])
print("Original A:", A)

# Create boolean mask
idx_nonzero = np.array([True, True, True])
print("idx_nonzero:", idx_nonzero)
print("idx_nonzero type:", type(idx_nonzero))
print("idx_nonzero shape:", idx_nonzero.shape)

# Test different indexing approaches
print("\n=== Testing indexing ===")
print("A[idx_nonzero, :]:", A[idx_nonzero, :])
print("A[idx_nonzero, :].shape:", A[idx_nonzero, :].shape)

# Test with explicit indices
explicit_indices = np.where(idx_nonzero)[0]
print("explicit_indices:", explicit_indices)
print("A[explicit_indices, :]:", A[explicit_indices, :])

# Test the normalization with explicit indices
normA = np.linalg.norm(A, axis=1)
normA_nonzero = normA[idx_nonzero]
print("normA_nonzero:", normA_nonzero)

# Test assignment with boolean indexing
A_out = A.copy()
print("Before assignment:", A_out)

# This should work but let's see what happens
A_out[idx_nonzero, :] = A_out[idx_nonzero, :] / normA_nonzero.reshape(-1, 1)
print("After assignment:", A_out)

# Check if the assignment worked
final_norms = np.linalg.norm(A_out, axis=1)
print("Final norms:", final_norms) 