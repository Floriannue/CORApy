#!/usr/bin/env python3
"""
Debug script to trace through the representsa_ method step by step
"""

import numpy as np
from cora_python.contSet.polytope.polytope import Polytope
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

# Create empty polytope
P = Polytope.empty(2)

print("Empty polytope properties:")
print(f"P.A: {P.A}")
print(f"P.b: {P.b}")
print(f"P.Ae: {P.Ae}")
print(f"P.be: {P.be}")
print(f"P.dim(): {P.dim()}")
print(f"P.isHRep: {P.isHRep()}")
print(f"P.isVRep: {P.isVRep()}")

# Test representsa_ for fullspace
print("\nTesting representsa_('fullspace', 0):")
result = P.representsa_('fullspace', 0)
print(f"Result: {result}")

# Now let's trace through the logic manually
print("\nManual trace through the logic:")
tol = 1e-9

# Check H-representation: no constraints means fullspace
# or all A rows zero and b >= 0
hrep_fullspace = False
if P.isHRep():
    print(f"P.isHRep: {P.isHRep}")
    print(f"P.A.size: {P.A.size}")
    print(f"P.Ae.size: {P.Ae.size}")
    
    if P.A.size == 0 and P.Ae.size == 0:
        print("Case 1: No constraints -> hrep_fullspace = True")
        hrep_fullspace = True
    else:
        print("Case 2: Check A and b constraints")
        print(f"np.all(withinTol(P.A, 0, tol)): {np.all(withinTol(P.A, 0, tol))}")
        print(f"P.A: {P.A}")
        print(f"withinTol(P.A, 0, tol): {withinTol(P.A, 0, tol)}")
        
        print(f"np.all((P.b > 0) | withinTol(P.b, 0, tol)): {np.all((P.b > 0) | withinTol(P.b, 0, tol))}")
        print(f"P.b: {P.b}")
        print(f"P.b > 0: {P.b > 0}")
        print(f"withinTol(P.b, 0, tol): {withinTol(P.b, 0, tol)}")
        print(f"(P.b > 0) | withinTol(P.b, 0, tol): {(P.b > 0) | withinTol(P.b, 0, tol)}")
        
        hrep_fullspace = (np.all(withinTol(P.A, 0, tol)) and 
                          np.all((P.b > 0) | withinTol(P.b, 0, tol)))
        print(f"hrep_fullspace = {hrep_fullspace}")

print(f"\nFinal hrep_fullspace: {hrep_fullspace}")

# Check equality constraints for H-representation
if P.Ae.size > 0:
    print(f"P.Ae.size > 0: {P.Ae.size > 0}")
    print("Checking equality constraints...")
    hrep_fullspace = hrep_fullspace and \
                     np.all(withinTol(P.Ae, 0, tol)) and \
                     np.all(withinTol(P.be, 0, tol))
    print(f"hrep_fullspace after equality check: {hrep_fullspace}")
else:
    print("No equality constraints to check")

print(f"\nFinal hrep_fullspace after all checks: {hrep_fullspace}")
