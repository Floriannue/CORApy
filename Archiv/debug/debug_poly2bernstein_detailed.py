#!/usr/bin/env python3
"""
Detailed debug script to trace the exact error in poly2bernstein
"""

import numpy as np
import sys
import os

# Add the cora_python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from g.functions.helper.sets.contSet.polyZonotope.poly2bernstein import poly2bernstein
from contSet.interval.interval import Interval

def debug_detailed():
    print("=== Detailed Debug of Simple Polynomial ===")
    
    G = np.array([[1, 2]])  # 1 dimension, 2 terms
    E = np.array([[1, 0], [0, 1]])  # 2 variables, 2 terms
    dom = Interval(np.array([-1, -1]), np.array([1, 1]))
    
    print(f"G shape: {G.shape}")
    print(f"E shape: {E.shape}")
    print(f"G = {G}")
    print(f"E = {E}")
    
    try:
        B = poly2bernstein(G, E, dom)
        print(f"Success! B = {B}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_detailed()
