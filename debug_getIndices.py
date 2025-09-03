#!/usr/bin/env python3
"""
Debug script to test _aux_getIndices function
"""

import numpy as np
import sys
import os

# Add the cora_python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from g.functions.helper.sets.contSet.polyZonotope.poly2bernstein import _aux_getIndices

def test_getIndices():
    print("=== Test _aux_getIndices ===")
    
    # Test case from simple polynomial
    E = np.array([[1, 0], [0, 1]])  # 2 variables, 2 terms
    l = E.shape[0]  # 2
    n = np.max(E)   # 1
    len_coeff = (n + 1) ** (l - 1)  # 2^1 = 2
    
    print(f"l = {l}, n = {n}, len_coeff = {len_coeff}")
    print(f"E shape: {E.shape}")
    
    for i in range(E.shape[1]):
        exp = E[:, i]
        print(f"\nTerm {i}: exp = {exp}")
        try:
            ind = _aux_getIndices(exp, len_coeff, n)
            print(f"  ind = {ind}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    test_getIndices()
