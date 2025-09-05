import numpy as np

# Test polynomial: -0.11*x + 0
# MATLAB: p = [-0.11, 0] represents -0.11*x + 0
# Python: np.polyval expects descending order

p_matlab_style = np.array([-0.11, 0])  # ascending order like MATLAB
p_python_style = np.array([0, -0.11])  # descending order like Python

x_points = np.array([-1, 0])

print("Testing polynomial evaluation:")
print(f"Polynomial: -0.11*x + 0")
print(f"x_points: {x_points}")

print(f"\nMATLAB style p = {p_matlab_style} (ascending order):")
for x in x_points:
    # Manual evaluation: -0.11*x + 0
    val_manual = -0.11 * x + 0
    print(f"  Manual: p({x}) = {val_manual}")

print(f"\nPython np.polyval with p = {p_matlab_style} (wrong order):")
for x in x_points:
    val = np.polyval(p_matlab_style, x)
    print(f"  np.polyval: p({x}) = {val}")

print(f"\nPython np.polyval with p = {p_python_style} (correct order):")
for x in x_points:
    val = np.polyval(p_python_style, x)
    print(f"  np.polyval: p({x}) = {val}")

print(f"\nPython np.polyval with reversed p = {p_matlab_style[::-1]}:")
for x in x_points:
    val = np.polyval(p_matlab_style[::-1], x)
    print(f"  np.polyval: p({x}) = {val}")
