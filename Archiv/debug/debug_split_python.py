#!/usr/bin/env python3
"""
Debug script to understand Python splitLongestGen behavior and compare with MATLAB
"""

import numpy as np
import sys
import os

# Add the cora_python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.contSet.polyZonotope import PolyZonotope

def debug_split_methods():
    """Debug split methods step by step"""
    
    # Create a simple polynomial zonotope (same as MATLAB)
    c = np.array([0, 0])
    G = np.array([[2, 0, 1], [0, 2, 1]])
    GI = np.array([[0], [0]])
    E = np.array([[1, 0, 3], [0, 1, 1]])
    id_list = np.array([1, 2])
    
    pZ = PolyZonotope(c, G, GI, E, id_list)
    
    print("Python PolyZonotope:")
    print(f"  c: {pZ.c}")
    print(f"  G shape: {pZ.G.shape}")
    print(f"  E shape: {pZ.E.shape}")
    print(f"  id: {pZ.id}")
    print(f"  id shape: {pZ.id.shape}")
    print(f"  id dtype: {pZ.id.dtype}")
    
    # Test splitLongestGen step by step
    print("\nTesting splitLongestGen step by step...")
    
    # Determine longest generator
    len_gen = np.sum(pZ.G**2, axis=0)
    print(f"  Generator lengths: {len_gen}")
    ind = np.argmax(len_gen)
    print(f"  Longest generator index: {ind}")
    
    # Find factor with the largest exponent
    print(f"  E[:, {ind}]: {pZ.E[:, ind]}")
    factor_idx = np.argmax(pZ.E[:, ind])
    print(f"  Factor with largest exponent index: {factor_idx}")
    print(f"  pZ.id[{factor_idx}]: {pZ.id[factor_idx]}")
    print(f"  pZ.id[{factor_idx}, 0]: {pZ.id[factor_idx, 0]}")
    factor = pZ.id[factor_idx, 0]  # Access the scalar value from 2D array
    print(f"  Factor value: {factor}")
    
    # Test splitDepFactor step by step
    print(f"\nTesting splitDepFactor with factor {factor}...")
    
    # Find selected dependent factor
    print(f"  pZ.id == {factor}: {pZ.id == factor}")
    print(f"  (pZ.id == {factor}).shape: {(pZ.id == factor).shape}")
    ind_mask = (pZ.id == factor).flatten()  # Flatten to 1D boolean array
    print(f"  ind_mask: {ind_mask}")
    print(f"  sum(ind_mask): {np.sum(ind_mask)}")
    
    E_ind = pZ.E[ind_mask, :]
    print(f"  E_ind shape: {E_ind.shape}")
    print(f"  E_ind: {E_ind}")
    
    # Parse input arguments
    polyOrd = np.max(E_ind)
    print(f"  polyOrd: {polyOrd}")
    
    # Determine all generators in which the selected dependent factor occurs
    genInd = (E_ind > 0) & (E_ind <= polyOrd)
    print(f"  genInd: {genInd}")
    
    print("\nPython splitLongestGen debug completed!")

if __name__ == "__main__":
    debug_split_methods()
