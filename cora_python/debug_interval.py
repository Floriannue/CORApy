#!/usr/bin/env python3
"""
Debug script to test the interval method
"""

import numpy as np
from cora_python.contSet.polytope.polytope import Polytope

# Create empty polytope
P = Polytope.empty(2)

print("Empty polytope properties:")
print(f"P.A: {P.A}")
print(f"P.b: {P.b}")
print(f"P.dim(): {P.dim()}")

# Test interval method
print("\nTesting P.interval():")
try:
    I = P.interval()
    print(f"Interval created: {I}")
    print(f"Interval type: {type(I)}")
    print(f"Interval dim: {I.dim()}")
except Exception as e:
    print(f"Error creating interval: {e}")
    import traceback
    traceback.print_exc()

# Test interval function directly
print("\nTesting interval(P) directly:")
try:
    from cora_python.contSet.polytope.interval import interval
    I = interval(P)
    print(f"Interval created: {I}")
    print(f"Interval type: {type(I)}")
    print(f"Interval dim: {I.dim()}")
except Exception as e:
    print(f"Error creating interval: {e}")
    import traceback
    traceback.print_exc()
