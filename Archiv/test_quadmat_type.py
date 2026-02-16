"""test_quadmat_type - Test what type quadMat is"""

import numpy as np
import scipy.sparse
from cora_python.contSet.zonotope import Zonotope

# Create test case
Z = Zonotope(np.array([[1.0], [0.0]]), np.array([[1.0, 0.5], [0.0, 0.3]]))
Zmat = np.hstack([Z.c, Z.G])
print(f"Zmat type: {type(Zmat)}")
print(f"Zmat shape: {Zmat.shape}")

# Test with sparse matrix (like hessianTensor returns)
H_sparse = scipy.sparse.csr_matrix(np.array([[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
print(f"\nH_sparse type: {type(H_sparse)}")
print(f"H_sparse shape: {H_sparse.shape}")

# Test matrix multiplication
try:
    quadMat = Zmat.T @ H_sparse @ Zmat
    print(f"\nquadMat type: {type(quadMat)}")
    print(f"quadMat shape: {quadMat.shape if hasattr(quadMat, 'shape') else 'N/A'}")
    
    if scipy.sparse.issparse(quadMat):
        print("quadMat is sparse")
        quadMat_dense = quadMat.toarray()
        print(f"quadMat (dense) diagonal: {np.diag(quadMat_dense[1:3, 1:3])}")
    elif isinstance(quadMat, np.ndarray):
        print("quadMat is numpy array")
        print(f"quadMat diagonal: {np.diag(quadMat[1:3, 1:3])}")
    else:
        print(f"quadMat is unexpected type: {quadMat}")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test with Interval (to see what happens)
print("\n" + "="*80)
print("Testing with Interval:")
from cora_python.contSet.interval import Interval
H_interval = Interval(
    np.array([[-0.1, -0.05], [-0.05, -0.1]]),
    np.array([[0.1, 0.05], [0.05, 0.1]])
)

try:
    quadMat_int = Zmat.T @ H_interval @ Zmat
    print(f"quadMat_int type: {type(quadMat_int)}")
    if hasattr(quadMat_int, 'inf') and hasattr(quadMat_int, 'sup'):
        print("quadMat_int is Interval")
        print(f"  inf diagonal: {np.diag(quadMat_int.inf[1:3, 1:3])}")
        print(f"  sup diagonal: {np.diag(quadMat_int.sup[1:3, 1:3])}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
