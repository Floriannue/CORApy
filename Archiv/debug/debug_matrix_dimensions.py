#!/usr/bin/env python3
"""
Debug script to check matrix dimensions in poly2bernstein
"""

import numpy as np
import sys
import os

# Add the cora_python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.g.functions.helper.sets.contSet.polyZonotope.poly2bernstein import _aux_getIndices

def debug_matrix_dimensions():
    print("=== Debug Matrix Dimensions ===")
    
    # Test case from simple polynomial
    E = np.array([[1, 0], [0, 1]])  # 2 variables, 2 terms
    l = E.shape[0]  # 2
    n = np.max(E)   # 1
    len_coeff = (n + 1) ** (l - 1)  # 2^1 = 2
    
    print(f"l = {l}, n = {n}, len_coeff = {len_coeff}")
    print(f"Matrix A should be: ({n + 1}, {len_coeff}) = ({n + 1}, {len_coeff})")
    
    # Check what indices we get
    for i in range(E.shape[1]):
        exp = E[:, i]
        ind = _aux_getIndices(exp, len_coeff, n)
        print(f"Term {i}: exp = {exp} -> ind = {ind}")
        print(f"  Trying to access A[{ind[0]}, {ind[1]}] in matrix of size ({n + 1}, {len_coeff})")
        
        # Check if indices are valid
        if ind[0] >= n + 1 or ind[1] >= len_coeff:
            print(f"  ERROR: Index out of bounds!")
        else:
            print(f"  OK: Index is valid")

if __name__ == "__main__":
    debug_matrix_dimensions()
