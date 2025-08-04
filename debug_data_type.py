import numpy as np

# Test data type and view issues
A = np.array([[1, 0], [-1, 1], [-1, -1]])
print("Original A:", A)
print("A.dtype:", A.dtype)

# Test the failing case with explicit data type
A_out = A.copy()
normA_nonzero = np.array([1.0, 1.41421356, 1.41421356])
explicit_indices = np.array([0, 1, 2])

for i, idx in enumerate(explicit_indices):
    print(f"\n=== Row {idx} ===")
    print(f"Before: A_out[{idx}, :] = {A_out[idx, :]}")
    print(f"A_out[{idx}, :].dtype: {A_out[idx, :].dtype}")
    
    # Test the division step by step
    original_row = A_out[idx, :].copy()
    print(f"Original row: {original_row}")
    print(f"Original row.dtype: {original_row.dtype}")
    
    result = original_row / normA_nonzero[i]
    print(f"Division result: {result}")
    print(f"Division result.dtype: {result.dtype}")
    
    # Check if result is a view
    print(f"result.base is original_row: {result.base is original_row}")
    print(f"result.base is A_out: {result.base is A_out}")
    
    # Now assign it with explicit copy
    A_out[idx, :] = result.copy()
    print(f"After assignment: A_out[{idx}, :] = {A_out[idx, :]}")
    
    # Check if the assignment worked
    print(f"Assignment worked: {np.allclose(A_out[idx, :], result)}")

print(f"\nFinal A_out: {A_out}")
print(f"Final A_out.dtype: {A_out.dtype}") 