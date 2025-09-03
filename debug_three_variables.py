#!/usr/bin/env python3
"""
Debug script for three variables test case
"""

import numpy as np
import sys
import os

# Add the cora_python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from g.functions.helper.sets.contSet.polyZonotope.poly2bernstein import poly2bernstein, _aux_getIndices, _aux_Binomial_coefficient, _aux_InverseUx, _aux_InverseVx, _aux_InverseWx
from contSet.interval.interval import Interval

def debug_three_variables():
    print("=== Debug Three Variables ===")
    
    G = np.array([[1, 1, 1]])
    E = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dom = Interval(np.array([-1, -1, -1]), np.array([1, 1, 1]))
    
    print(f"G shape: {G.shape}")
    print(f"E shape: {E.shape}")
    
    l = E.shape[0]  # number of variables = 3
    n = np.max(E)   # maximum degree = 1
    h = G.shape[0]  # number of dimensions = 1
    
    print(f"l = {l}, n = {n}, h = {h}")
    
    # Preprocessing: compute coefficient matrix A
    len_coeff = (n + 1) ** (l - 1)  # 2^(3-1) = 4
    print(f"len_coeff = {len_coeff}")
    
    A = [np.zeros((n + 1, len_coeff)) for _ in range(h)]
    print(f"A[0] shape: {A[0].shape}")
    
    # Check indices
    for i in range(E.shape[1]):
        exp = E[:, i]
        ind = _aux_getIndices(exp, len_coeff, n)
        print(f"Term {i}: exp = {exp} -> ind = {ind}")
        A[0][ind[0], ind[1]] = G[0, i]
    
    print(f"A[0] after filling:\n{A[0]}")
    
    # Check matrix dimensions
    C = _aux_Binomial_coefficient(n)
    print(f"C shape: {C.shape}")
    
    Ux = _aux_InverseUx(n, C)
    print(f"Ux shape: {Ux.shape}")
    
    Vx = _aux_InverseVx(n, dom[0])
    print(f"Vx shape: {Vx.shape}")
    
    Wx = _aux_InverseWx(n, dom[0], C)
    print(f"Wx shape: {Wx.shape}")
    
    M = Ux @ Vx @ Wx
    print(f"M shape: {M.shape}")
    print(f"A[0] shape: {A[0].shape}")
    
    # Try the multiplication
    try:
        result = M @ A[0]
        print(f"Multiplication successful! Result shape: {result.shape}")
    except Exception as e:
        print(f"Multiplication failed: {e}")

if __name__ == "__main__":
    debug_three_variables()
