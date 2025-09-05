#!/usr/bin/env python3
"""
Debug script to test poly2bernstein edge cases in Python
"""

import numpy as np
import sys
import os

# Add the cora_python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from g.functions.helper.sets.contSet.polyZonotope.poly2bernstein import poly2bernstein
from contSet.interval.interval import Interval

def test_poly2bernstein():
    print("=== Test 1: Empty generators ===")
    try:
        G = np.array([]).reshape(1, 0)
        E = np.array([]).reshape(0, 0)
        dom = Interval(np.array([-1, -1]), np.array([1, 1]))
        B = poly2bernstein(G, E, dom)
        print(f"Empty generators - B type: {type(B)}")
        if hasattr(B, 'shape'):
            print(f"Empty generators - B shape: {B.shape}")
        else:
            print(f"Empty generators - B: {B}")
    except Exception as e:
        print(f"Empty generators - Error: {e}")

    print("\n=== Test 2: Zero polynomial ===")
    try:
        G = np.array([[0]])
        E = np.array([[0, 0]])
        dom = Interval(np.array([-1, -1]), np.array([1, 1]))
        B = poly2bernstein(G, E, dom)
        print(f"Zero polynomial - B: {B}")
    except Exception as e:
        print(f"Zero polynomial - Error: {e}")

    print("\n=== Test 3: Constant polynomial ===")
    try:
        G = np.array([[5]])
        E = np.array([[0, 0]])
        dom = Interval(np.array([-1, -1]), np.array([1, 1]))
        B = poly2bernstein(G, E, dom)
        print(f"Constant polynomial - B: {B}")
    except Exception as e:
        print(f"Constant polynomial - Error: {e}")

    print("\n=== Test 4: Simple polynomial ===")
    try:
        G = np.array([[1], [2]])
        E = np.array([[1, 0], [0, 1]])
        dom = Interval(np.array([-1, -1]), np.array([1, 1]))
        B = poly2bernstein(G, E, dom)
        print(f"Simple polynomial - B type: {type(B)}")
        if hasattr(B, 'shape'):
            print(f"Simple polynomial - B shape: {B.shape}")
        else:
            print(f"Simple polynomial - B: {B}")
    except Exception as e:
        print(f"Simple polynomial - Error: {e}")

if __name__ == "__main__":
    test_poly2bernstein()
