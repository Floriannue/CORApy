import numpy as np
from cora_python.g.functions.matlab.validate.check.withinTol import withinTol

# Original constraints
A_orig = np.array([[1, 0], [-1, 1], [-1, -1]])
b_orig = np.array([[7], [-2], [-4]])

# Compacted constraints (from debug output)
A_comp = np.array([[-0.70710678, -0.70710678],
                   [1.0, 0.0],
                   [-0.70710678, 0.70710678]])
b_comp = np.array([-2.82842712, 7.0, -1.41421356])

print("Original constraints:")
print("A:", A_orig)
print("b:", b_orig.flatten())

print("\nCompacted constraints:")
print("A:", A_comp)
print("b:", b_comp)

# Test points
points = [
    np.array([7.0, -3.0]),
    np.array([3.0, 1.0]),
    np.array([7.0, 5.0])
]

for i, point in enumerate(points):
    print(f"\nPoint {i+1}: {point}")
    
    # Check with original constraints
    val_orig = A_orig @ point.reshape(-1, 1)
    satisfied_orig = np.all(val_orig.flatten() <= b_orig.flatten())
    print(f"Original constraints satisfied: {satisfied_orig}")
    print(f"Original constraint values: {val_orig.flatten()}")
    print(f"Original bounds: {b_orig.flatten()}")
    
    # Check with compacted constraints
    val_comp = A_comp @ point.reshape(-1, 1)
    tol = 1e-8
    satisfied_comp = np.all((val_comp.flatten() < b_comp + tol) | withinTol(val_comp.flatten(), b_comp, tol))
    print(f"Compacted constraints satisfied: {satisfied_comp}")
    print(f"Compacted constraint values: {val_comp.flatten()}")
    print(f"Compacted bounds: {b_comp}") 