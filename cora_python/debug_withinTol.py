#!/usr/bin/env python3
"""
Debug script to test the withinTol function directly
"""

import numpy as np
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

# Test the exact values from the empty polytope
A = np.array([[0.0, 0.0]])
b = np.array([[-1.0]])
tol = 1e-9

print("Testing withinTol function:")
print(f"A: {A}")
print(f"b: {b}")
print(f"tol: {tol}")

print(f"\nwithinTol(A, 0, tol): {withinTol(A, 0, tol)}")
print(f"np.all(withinTol(A, 0, tol)): {np.all(withinTol(A, 0, tol))}")

print(f"\nwithinTol(b, 0, tol): {withinTol(b, 0, tol)}")
print(f"b > 0: {b > 0}")
print(f"(b > 0) | withinTol(b, 0, tol): {(b > 0) | withinTol(b, 0, tol)}")
print(f"np.all((b > 0) | withinTol(b, 0, tol)): {np.all((b > 0) | withinTol(b, 0, tol))}")

# Test the full logic
hrep_fullspace = (np.all(withinTol(A, 0, tol)) and 
                  np.all((b > 0) | withinTol(b, 0, tol)))
print(f"\nhrep_fullspace = {hrep_fullspace}")
