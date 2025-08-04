import numpy as np
import sys
import os

# Add the cora_python path
sys.path.insert(0, os.path.join(os.getcwd(), 'cora_python'))

from cora_python.contSet.ellipsoid import Ellipsoid
from cora_python.contSet.polytope import Polytope

# Test the failing case
Q = np.array([[2, 0], [0, 1]])
q = np.array([[2], [-1]])
M = np.array([[2, -1], [1, 1]])

# Create the ellipsoid
E = M @ Ellipsoid(Q, q)

# Create the polytope that should be outside
A_outside = np.array([[1, 0], [-1, 1], [-1, -1]])
b_outside = np.array([[7], [-2], [-4]])
P_outside = Polytope(A_outside, b_outside)

# Test the contains method with different parameters
print("Testing contains method:")

# Test with default parameters
result1 = E.contains(P_outside)
print(f"E.contains(P_outside) = {result1}")

# Test with explicit method='exact'
result2 = E.contains(P_outside, method='exact')
print(f"E.contains(P_outside, method='exact') = {result2}")

# Test with different tolerance
result3 = E.contains(P_outside, tol=1e-12)
print(f"E.contains(P_outside, tol=1e-12) = {result3}")

# Check polytope properties
print(f"\nPolytope properties:")
print(f"P_outside.emptySet = {P_outside.emptySet}")
print(f"P_outside.isFullDim = {P_outside.isFullDim}")
print(f"P_outside.isBounded = {P_outside.isBounded}")

# Check if polytope represents a point
try:
    is_point, point = P_outside.representsa_('point', return_set=True)
    print(f"P_outside represents a point: {is_point}")
    if is_point:
        print(f"Point: {point}")
except Exception as e:
    print(f"Error checking if polytope represents a point: {e}")

# Check if polytope represents emptySet
try:
    is_empty = P_outside.representsa_('emptySet')
    print(f"P_outside represents emptySet: {is_empty}")
except Exception as e:
    print(f"Error checking if polytope represents emptySet: {e}")

# Check if polytope represents fullspace
try:
    is_fullspace = P_outside.representsa_('fullspace')
    print(f"P_outside represents fullspace: {is_fullspace}")
except Exception as e:
    print(f"Error checking if polytope represents fullspace: {e}") 