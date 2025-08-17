#!/usr/bin/env python3
"""
Comprehensive debug script to trace through the entire representsa_ method
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
print(f"P.isHRep(): {P.isHRep()}")
print(f"P.isVRep(): {P.isVRep()}")

# Test representsa_ for fullspace
print("\nTesting representsa_('fullspace', 0):")
result = P.representsa_('fullspace', 0)
print(f"Result: {result}")

# Now let's trace through the logic manually
print("\nManual trace through the logic:")
tol = 1e-9
n = P.dim()

# Check H-representation: no constraints means fullspace
# or all A rows zero and b >= 0
hrep_fullspace = False
if P.isHRep():
    print(f"P.isHRep(): {P.isHRep()}")
    print(f"P.A.size: {P.A.size}")
    print(f"P.Ae.size: {P.Ae.size if P.Ae is not None else 'None'}")
    
    if P.A.size == 0 and (P.Ae is None or P.Ae.size == 0):
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
if P.Ae is not None and P.Ae.size > 0:
    print(f"P.Ae.size > 0: {P.Ae.size > 0}")
    print("Checking equality constraints...")
    hrep_fullspace = hrep_fullspace and \
                     np.all(withinTol(P.Ae, 0, tol)) and \
                     np.all(withinTol(P.be, 0, tol))
    print(f"hrep_fullspace after equality check: {hrep_fullspace}")
else:
    print("No equality constraints to check")

print(f"\nFinal hrep_fullspace after all checks: {hrep_fullspace}")

# Check V-representation (1D case with -Inf and Inf)
vrep_fullspace = False
if P.isVRep() and n == 1:
    print(f"P.isVRep() and n == 1: {P.isVRep() and n == 1}")
    print("Checking V-representation...")
    try:
        vertices = P.V
        vrep_fullspace = (n == 1 and np.any(np.isinf(vertices) & (vertices < 0)) and \
                          np.any(np.isinf(vertices) & (vertices > 0)))
        print(f"vrep_fullspace = {vrep_fullspace}")
    except Exception as e:
        print(f"Error accessing P.V: {e}")
        vrep_fullspace = False
else:
    print(f"V-representation check skipped: P.isVRep() = {P.isVRep()}, n = {n}")

print(f"\nFinal vrep_fullspace: {vrep_fullspace}")

# Final result
res = hrep_fullspace or vrep_fullspace
print(f"\nFinal result: hrep_fullspace OR vrep_fullspace = {hrep_fullspace} OR {vrep_fullspace} = {res}")

# Check if this matches the actual result
print(f"\nActual representsa_('fullspace', 0) result: {result}")
print(f"Manual calculation matches: {res == result}")
