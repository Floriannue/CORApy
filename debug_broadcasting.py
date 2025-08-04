import numpy as np

# Test the normalization logic step by step
A = np.array([[1, 0], [-1, 1], [-1, -1]])
b = np.array([7, -2, -4]).reshape(-1, 1)

print("Original A:", A)
print("Original b:", b.flatten())

# Compute norms
normA = np.linalg.norm(A, axis=1)
print("Norms:", normA)

# Get explicit indices
explicit_indices = np.array([0, 1, 2])
print("Explicit indices:", explicit_indices)

# Get the rows to normalize
A_rows = A[explicit_indices, :]
print("A_rows:", A_rows)
print("A_rows shape:", A_rows.shape)

# Get the norms for these rows
normA_nonzero = normA[explicit_indices]
print("normA_nonzero:", normA_nonzero)
print("normA_nonzero shape:", normA_nonzero.shape)

# Reshape for broadcasting
normA_reshaped = normA_nonzero.reshape(-1, 1)
print("normA_reshaped:", normA_reshaped)
print("normA_reshaped shape:", normA_reshaped.shape)

# Test the division
A_normalized = A_rows / normA_reshaped
print("A_normalized:", A_normalized)
print("A_normalized shape:", A_normalized.shape)

# Check norms after normalization
norms_after = np.linalg.norm(A_normalized, axis=1)
print("Norms after normalization:", norms_after)

# Now test the assignment
A_out = A.copy()
print("A_out before assignment:", A_out)

# This should work
A_out[explicit_indices, :] = A_normalized
print("A_out after assignment:", A_out)

# Check final norms
final_norms = np.linalg.norm(A_out, axis=1)
print("Final norms:", final_norms)
print("All norms should be 1.0:", np.allclose(final_norms, 1.0)) 