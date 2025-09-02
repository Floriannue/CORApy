import numpy as np

# Test the polynomial evaluation difference
p = np.array([-0.11, 0])  # MATLAB: [-0.11, 0]
x_points = np.array([-1, 0])

print("Polynomial coefficients:", p)
print("Evaluation points:", x_points)

# Python np.polyval (expects descending order)
print("\nPython np.polyval (descending order):")
for x in x_points:
    val = np.polyval(p, x)
    print(f"  p({x}) = {val}")

# Manual evaluation (ascending order like MATLAB)
print("\nManual evaluation (ascending order like MATLAB):")
for x in x_points:
    val = p[0] * x + p[1]  # -0.11*x + 0
    print(f"  p({x}) = {val}")

# Test with reversed coefficients
p_reversed = p[::-1]  # [0, -0.11]
print(f"\nReversed coefficients: {p_reversed}")
print("Python np.polyval with reversed coefficients:")
for x in x_points:
    val = np.polyval(p_reversed, x)
    print(f"  p({x}) = {val}")
