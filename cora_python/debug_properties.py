#!/usr/bin/env python3
"""
Debug script to test the isHRep and isVRep properties
"""

import numpy as np
from cora_python.contSet.polytope.polytope import Polytope

# Create empty polytope
P = Polytope.empty(2)

print("Empty polytope properties:")
print(f"P.A: {P.A}")
print(f"P.b: {P.b}")
print(f"P.Ae: {P.Ae}")
print(f"P.be: {P.be}")
print(f"P.dim(): {P.dim()}")

# Test the properties
print("\nTesting properties:")
try:
    print(f"P.isHRep: {P.isHRep}")
    print(f"P.isVRep: {P.isVRep}")
except Exception as e:
    print(f"Error accessing properties: {e}")

# Test the methods
print("\nTesting methods:")
try:
    print(f"P.isHRep(): {P.isHRep()}")
    print(f"P.isVRep(): {P.isVRep()}")
except Exception as e:
    print(f"Error calling methods: {e}")

# Test the internal attributes
print("\nTesting internal attributes:")
try:
    print(f"P._isHRep: {P._isHRep}")
    print(f"P._isVRep: {P._isVRep}")
except Exception as e:
    print(f"Error accessing internal attributes: {e}")

# Test representsa_ for fullspace
print("\nTesting representsa_('fullspace', 0):")
try:
    result = P.representsa_('fullspace', 0)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error calling representsa_: {e}")
    import traceback
    traceback.print_exc()
