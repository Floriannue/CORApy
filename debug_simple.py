import numpy as np
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

# Test the constraint checking logic
A = np.array([[-0.70710678, -0.70710678],
              [1.0, 0.0],
              [-0.70710678, 0.70710678]])
b = np.array([-2.82842712, 7.0, -1.41421356])

# Test point that should be inside
test_point = np.array([7.0, 0.0]).reshape(-1, 1)
print(f"Test point: {test_point.flatten()}")

# Check constraints
val = A @ test_point
print(f"Constraint values: {val.flatten()}")
print(f"Constraint bounds: {b}")
print(f"Constraint violations: {val.flatten() > b}")

# Check with tolerance
tol = 1e-8
satisfied = (val.flatten() < b + tol) | withinTol(val.flatten(), b, tol)
print(f"Constraints satisfied: {satisfied}")
print(f"All satisfied: {np.all(satisfied)}")

# Test the intersection points from debug output
points = [
    np.array([7.0, -3.0]),
    np.array([3.0, 1.0]),
    np.array([7.0, 5.0])
]

for i, point in enumerate(points):
    print(f"\nPoint {i+1}: {point}")
    val = A @ point.reshape(-1, 1)
    print(f"Constraint values: {val.flatten()}")
    print(f"Constraint bounds: {b}")
    satisfied = (val.flatten() < b + tol) | withinTol(val.flatten(), b, tol)
    print(f"Constraints satisfied: {satisfied}")
    print(f"All satisfied: {np.all(satisfied)}") 