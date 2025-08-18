#!/usr/bin/env python3
"""
Simple debug script to trace representsa_ call
"""

import numpy as np
import sys
import os

# Add the cora_python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cora_python'))

from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.representsa_ import representsa_

print("=== Debug representsa_ call ===")

# Create the same polytope as in the test
A = np.zeros((0, 1))
b = np.zeros(0)
P = Polytope(A, b)

print(f"P type: {type(P)}")
print(f"P.A shape: {P.A.shape}, P.A size: {P.A.size}")
print(f"P.b shape: {P.b.shape}, P.b size: {P.b.size}")
print(f"P.Ae shape: {P.Ae.shape}, P.Ae size: {P.Ae.size}")
print(f"P.be shape: {P.be.shape}, P.be size: {P.be.size}")
print(f"P.dim(): {P.dim()}")
print(f"P.isemptyobject(): {P.isemptyobject()}")

print("\n--- Calling representsa_(P, 'emptySet') ---")
try:
    result = representsa_(P, 'emptySet')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Calling P.representsa_('emptySet') ---")
try:
    result = P.representsa_('emptySet')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
