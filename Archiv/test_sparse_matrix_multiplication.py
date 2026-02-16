"""test_sparse_matrix_multiplication - Test if sparse matrix multiplication differs"""

import numpy as np
import scipy.sparse

print("=" * 80)
print("TESTING SPARSE MATRIX MULTIPLICATION")
print("=" * 80)

# Create test case matching actual usage
Zmat = np.array([[1.0, 1.0, 0.5], [0.0, 0.0, 0.3]])
H_sparse = scipy.sparse.csr_matrix(np.array([[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

print("\n1. Test matrices:")
print(f"Zmat shape: {Zmat.shape}")
print(f"Zmat:\n{Zmat}")
print(f"\nH_sparse shape: {H_sparse.shape}")
print(f"H_sparse (dense):\n{H_sparse.toarray()}")

# Test 1: Sparse multiplication (Python current)
print("\n2. Sparse multiplication (Zmat.T @ H_sparse @ Zmat):")
try:
    quadMat_sparse = Zmat.T @ H_sparse @ Zmat
    print(f"Result type: {type(quadMat_sparse)}")
    if scipy.sparse.issparse(quadMat_sparse):
        quadMat_sparse_dense = quadMat_sparse.toarray()
        print(f"Result (dense):\n{quadMat_sparse_dense}")
    else:
        print(f"Result:\n{quadMat_sparse}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 2: Dense multiplication (convert H first)
print("\n3. Dense multiplication (convert H to dense first):")
H_dense = H_sparse.toarray()
quadMat_dense = Zmat.T @ H_dense @ Zmat
print(f"Result:\n{quadMat_dense}")

# Compare
if scipy.sparse.issparse(quadMat_sparse):
    quadMat_sparse_dense = quadMat_sparse.toarray()
    diff = np.abs(quadMat_sparse_dense - quadMat_dense)
    print(f"\n4. Difference:")
    print(f"Max absolute difference: {np.max(diff)}")
    if np.max(diff) > 1e-10:
        print("WARNING: Sparse and dense multiplication differ!")
    else:
        print("OK: Sparse and dense multiplication match")
else:
    diff = np.abs(quadMat_sparse - quadMat_dense)
    print(f"\n4. Difference:")
    print(f"Max absolute difference: {np.max(diff)}")
    if np.max(diff) > 1e-10:
        print("WARNING: Results differ!")
    else:
        print("OK: Results match")

print("\n" + "=" * 80)
