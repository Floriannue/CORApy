#!/usr/bin/env python3
"""
Test script to verify that splitLongestGen and splitDepFactor work as instance methods
"""

import numpy as np
import sys
import os

# Add the cora_python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.contSet.polyZonotope import PolyZonotope

def test_split_methods():
    """Test that split methods work as instance methods"""
    
    # Create a simple polynomial zonotope
    c = np.array([0, 0])
    G = np.array([[2, 0, 1], [0, 2, 1]])
    GI = np.array([[0], [0]])
    E = np.array([[1, 0, 3], [0, 1, 1]])
    id_list = np.array([1, 2])
    
    pZ = PolyZonotope(c, G, GI, E, id_list)
    
    print("Original PolyZonotope:")
    print(f"  c: {pZ.c}")
    print(f"  G shape: {pZ.G.shape}")
    print(f"  E shape: {pZ.E.shape}")
    print(f"  id: {pZ.id}")
    
    # Test splitLongestGen as instance method
    print("\nTesting splitLongestGen as instance method...")
    try:
        pZsplit = pZ.splitLongestGen()
        print(f"  Success! Split into {len(pZsplit)} parts")
        for i, split_pZ in enumerate(pZsplit):
            print(f"    Part {i+1}: c={split_pZ.c}, G shape={split_pZ.G.shape}")
    except Exception as e:
        print(f"  Error: {e}")
        return False
    
    # Test splitDepFactor as instance method
    print("\nTesting splitDepFactor as instance method...")
    try:
        pZsplit2 = pZ.splitDepFactor(1)  # Split factor 1
        print(f"  Success! Split into {len(pZsplit2)} parts")
        for i, split_pZ in enumerate(pZsplit2):
            print(f"    Part {i+1}: c={split_pZ.c}, G shape={split_pZ.G.shape}")
    except Exception as e:
        print(f"  Error: {e}")
        return False
    
    print("\nAll tests passed!")
    return True

if __name__ == "__main__":
    test_split_methods()
