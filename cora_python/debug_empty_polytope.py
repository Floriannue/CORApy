#!/usr/bin/env python3
"""
Debug script to understand why empty polytope is detected as fullspace
"""

import numpy as np
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.contSet.polytope.representsa_ import representsa_

# Create empty polytope
P = Polytope.empty(2)

print("Empty polytope properties:")
print(f"P.A: {P.A}")
print(f"P.b: {P.b}")
print(f"P.Ae: {P.Ae}")
print(f"P.be: {P.be}")
print(f"P.dim(): {P.dim()}")
print(f"P._emptySet_val: {getattr(P, '_emptySet_val', 'Not found')}")
print(f"P._bounded_val: {getattr(P, '_bounded_val', 'Not found')}")
print(f"P._fullDim_val: {getattr(P, '_fullDim_val', 'Not found')}")

# Test representsa_ for fullspace
print("\nTesting representsa_('fullspace', 0):")
result = P.representsa_('fullspace', 0)
print(f"Result: {result}")

# Test representsa_ for emptySet
print("\nTesting representsa_('emptySet', 0):")
result = P.representsa_('emptySet', 0)
print(f"Result: {result}")

# Test the specific logic from representsa_
print("\nTesting the fullspace logic:")
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

tol = 1e-9
print(f"np.all(withinTol(P.A, 0, tol)): {np.all(withinTol(P.A, 0, tol))}")
print(f"np.all((P.b > 0) | withinTol(P.b, 0, tol)): {np.all((P.b > 0) | withinTol(P.b, 0, tol))}")

hrep_fullspace = (np.all(withinTol(P.A, 0, tol)) and 
                  np.all((P.b > 0) | withinTol(P.b, 0, tol)))
print(f"hrep_fullspace: {hrep_fullspace}")

# Check if there are equality constraints
if P.Ae is not None and P.Ae.size > 0:
    print(f"P.Ae check: {np.all(withinTol(P.Ae, 0, tol))}")
    print(f"P.be check: {np.all(withinTol(P.be, 0, tol))}")
    hrep_fullspace = hrep_fullspace and \
                     np.all(withinTol(P.Ae, 0, tol)) and \
                     np.all(withinTol(P.be, 0, tol))
    print(f"hrep_fullspace after equality check: {hrep_fullspace}")
else:
    print("No equality constraints")

print(f"Final hrep_fullspace: {hrep_fullspace}")
print(f"Final result should be: {hrep_fullspace}")
