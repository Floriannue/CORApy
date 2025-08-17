#!/usr/bin/env python3
"""
Debug script to test polytope vertices computation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

import numpy as np
from cora_python.contSet.polytope.polytope import Polytope

def test_vertices():
    """Test vertices computation for different polytopes"""
    
    print("=== Testing vertices computation ===")
    
    # Test 1: Single point (equality constraints only)
    print("\n1. Single point (equality constraints only):")
    Ae = np.array([[1, 0], [0, 1]])
    be = np.array([2, 3])
    P1 = Polytope(Ae=Ae, be=be)
    
    print(f"  Polytope: Ae={Ae}, be={be}")
    print(f"  Dimension: {P1.dim()}")
    print(f"  Is H-rep: {P1.isHRep}")
    print(f"  Is V-rep: {P1.isVRep}")
    
    try:
        V1 = P1.vertices_()
        print(f"  Vertices: shape={V1.shape}, data={V1}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 2: Bounded polytope
    print("\n2. Bounded polytope:")
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([1, 1, 1, 1])
    P2 = Polytope(A, b)
    
    print(f"  Polytope: A={A}, b={b}")
    print(f"  Dimension: {P2.dim()}")
    print(f"  Is H-rep: {P2.isHRep}")
    print(f"  Is V-rep: {P2.isVRep}")
    
    try:
        V2 = P2.vertices_()
        print(f"  Vertices: shape={V2.shape}, data={V2}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 3: Empty polytope
    print("\n3. Empty polytope:")
    P3 = Polytope.empty(2)
    
    print(f"  Polytope: empty 2D")
    print(f"  Dimension: {P3.dim()}")
    print(f"  Is H-rep: {P3.isHRep}")
    print(f"  Is V-rep: {P3.isVRep}")
    
    try:
        V3 = P3.vertices_()
        print(f"  Vertices: shape={V3.shape}, data={V3}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    test_vertices()
