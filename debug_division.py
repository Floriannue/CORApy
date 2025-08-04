import numpy as np

# Test the division issue
A = np.array([[1, 0], [-1, 1], [-1, -1]])
print("Original A:", A)

# Test division for each row
for i in range(3):
    row = A[i, :]
    norm = np.linalg.norm(row)
    print(f"Row {i}: {row}, norm: {norm}")
    
    # Test division
    normalized_row = row / norm
    print(f"Row {i} after division: {normalized_row}")
    
    # Check if the division worked
    new_norm = np.linalg.norm(normalized_row)
    print(f"New norm: {new_norm}")
    print()

# Test the specific case that's failing
print("=== Testing the failing case ===")
A_out = A.copy()
normA_nonzero = np.array([1.0, 1.41421356, 1.41421356])
explicit_indices = np.array([0, 1, 2])

for i, idx in enumerate(explicit_indices):
    print(f"Before: A_out[{idx}, :] = {A_out[idx, :]}")
    print(f"Dividing by: {normA_nonzero[i]}")
    
    # Test the division step by step
    original_row = A_out[idx, :].copy()
    print(f"Original row: {original_row}")
    
    result = original_row / normA_nonzero[i]
    print(f"Division result: {result}")
    
    # Now assign it
    A_out[idx, :] = result
    print(f"After assignment: A_out[{idx}, :] = {A_out[idx, :]}")
    print() 