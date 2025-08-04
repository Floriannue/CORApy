import numpy as np

# Test the normalization logic with a simple approach
A = np.array([[1, 0], [-1, 1], [-1, -1]])
b = np.array([7, -2, -4]).reshape(-1, 1)

print("Original A:", A)
print("Original b:", b.flatten())

# Compute norms
normA = np.linalg.norm(A, axis=1)
print("Norms:", normA)

# Use simple tolerance check instead of withinTol
tol = 1e-8
idx_nonzero = normA > tol
print("Non-zero indices:", idx_nonzero)

# Test the normalization
if np.any(idx_nonzero):
    normA_nonzero = normA[idx_nonzero]
    print("Non-zero norms:", normA_nonzero)
    
    # Use explicit indices
    explicit_indices = np.where(idx_nonzero)[0]
    print("Explicit indices:", explicit_indices)
    
    # Apply normalization using loop
    A_out = A.copy()
    b_out = b.copy()
    
    for i, idx in enumerate(explicit_indices):
        print(f"Normalizing row {idx} with norm {normA_nonzero[i]}")
        A_out[idx, :] = A_out[idx, :] / normA_nonzero[i]
        b_out[idx] = b_out[idx] / normA_nonzero[i]
        print(f"Row {idx} after normalization: {A_out[idx, :]}")
    
    print("A_out after normalization:", A_out)
    print("b_out after normalization:", b_out.flatten())
    
    # Check if normalization worked correctly
    norms_after = np.linalg.norm(A_out, axis=1)
    print("Norms after normalization:", norms_after)
    print("All norms should be 1.0:", np.allclose(norms_after, 1.0))
    
    # Check if any rows became zero
    print("Any zero rows:", np.any(np.all(A_out == 0, axis=1))) 