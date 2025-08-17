#!/usr/bin/env python3
"""
Debug script to test ellipsoid enclosePoints method
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

import numpy as np
from cora_python.contSet.ellipsoid.ellipsoid import Ellipsoid

def test_enclosePoints():
    """Test enclosePoints method for different point sets"""
    
    print("=== Testing ellipsoid enclosePoints method ===")
    
    # Test 1: Single point
    print("\n1. Single point:")
    points1 = np.array([[2], [3]])  # 2D point at (2, 3)
    print(f"  Points: shape={points1.shape}, data={points1}")
    
    try:
        E1 = Ellipsoid.enclosePoints(points1, 'cov')
        print(f"  Ellipsoid: Q shape={E1.Q.shape}, q shape={E1.q.shape}")
        print(f"  Center: {E1.q.flatten()}")
        print(f"  Shape matrix: {E1.Q}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 2: Multiple points
    print("\n2. Multiple points:")
    points2 = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])  # Square vertices
    print(f"  Points: shape={points2.shape}, data={points2}")
    
    try:
        E2 = Ellipsoid.enclosePoints(points2, 'cov')
        print(f"  Ellipsoid: Q shape={E2.Q.shape}, q shape={E2.q.shape}")
        print(f"  Center: {E2.q.flatten()}")
        print(f"  Shape matrix: {E2.Q}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 3: Empty points
    print("\n3. Empty points:")
    points3 = np.zeros((2, 0))  # Empty point set
    print(f"  Points: shape={points3.shape}, data={points3}")
    
    try:
        E3 = Ellipsoid.enclosePoints(points3, 'cov')
        print(f"  Ellipsoid: Q shape={E3.Q.shape}, q shape={E3.q.shape}")
        print(f"  Center: {E3.q.flatten()}")
        print(f"  Shape matrix: {E3.Q}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    test_enclosePoints()
