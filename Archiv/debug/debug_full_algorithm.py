#!/usr/bin/env python3
"""
Debug script to trace through the full poly2bernstein algorithm
"""

import numpy as np
import sys
import os

# Add the cora_python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from g.functions.helper.sets.contSet.polyZonotope.poly2bernstein import _aux_getIndices, _aux_Binomial_coefficient, _aux_InverseUx, _aux_InverseVx, _aux_InverseWx, _aux_transposeMatrix
from contSet.interval.interval import Interval

def debug_full_algorithm():
    print("=== Debug Full Algorithm ===")
    
    G = np.array([[1, 1, 1]])
    E = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dom = Interval(np.array([-1, -1, -1]), np.array([1, 1, 1]))
    
    l = E.shape[0]  # number of variables = 3
    n = np.max(E)   # maximum degree = 1
    h = G.shape[0]  # number of dimensions = 1
    
    print(f"l = {l}, n = {n}, h = {h}")
    
    # Preprocessing: compute coefficient matrix A
    len_coeff = (n + 1) ** (l - 1)  # 2^(3-1) = 4
    A = [np.zeros((n + 1, len_coeff)) for _ in range(h)]
    
    for i in range(E.shape[1]):
        ind = _aux_getIndices(E[:, i], len_coeff, n)
        for j in range(h):
            A[j][ind[0], ind[1]] = G[j, i]
    
    print(f"Initial A[0] shape: {A[0].shape}")
    print(f"Initial A[0]:\n{A[0]}")
    
    # Step 1: Compute the binomial coefficients
    C = _aux_Binomial_coefficient(n)
    
    # Step 2: Compute inverse of U_x
    Ux = _aux_InverseUx(n, C)
    
    # Step 3: Iterate
    M = [None] * l
    infi = dom.inf
    sup = dom.sup
    
    for r in range(l):
        if r == 0 or infi[r] != infi[0] or sup[r] != sup[0]:
            # Compute inverse of V_x
            Vx = _aux_InverseVx(n, dom[r])
            
            # Compute inverse of W_x
            Wx = _aux_InverseWx(n, dom[r], C)
            
            # Product of all the inverse matrices
            M[r] = Ux @ Vx @ Wx
        else:
            M[r] = M[0]
    
    print(f"M[0] shape: {M[0].shape}")
    print(f"M[1] shape: {M[1].shape}")
    print(f"M[2] shape: {M[2].shape}")
    
    # Step 4: Iterate
    for j in range(h):
        print(f"\nProcessing dimension j = {j}")
        for r in range(l):
            print(f"  Processing variable r = {r}")
            print(f"    Before transpose: A[{j}] shape = {A[j].shape}")
            
            A[j] = _aux_transposeMatrix(A[j], r, len_coeff, l, n)
            print(f"    After transpose: A[{j}] shape = {A[j].shape}")
            
            print(f"    M[{r}] shape = {M[r].shape}")
            print(f"    A[{j}] shape = {A[j].shape}")
            
            try:
                A[j] = M[r] @ A[j]
                print(f"    After multiplication: A[{j}] shape = {A[j].shape}")
            except Exception as e:
                print(f"    ERROR in multiplication: {e}")
                return
            
            A[j] = _aux_transposeMatrix(A[j], r, len_coeff, l, n)
            print(f"    After second transpose: A[{j}] shape = {A[j].shape}")

if __name__ == "__main__":
    debug_full_algorithm()
