# Debug script to test Python findBernsteinPoly function
# Compare with MATLAB implementation

import numpy as np
from cora_python.nn.nnHelper.findBernsteinPoly import findBernsteinPoly

# Test with constant function
def f(x):
    return 5.0

l, u, n = 0, 1, 3
coeffs_const = findBernsteinPoly(f, l, u, n)
print(f'Python Constant function coeffs: {coeffs_const}')

# Test polynomial evaluation at endpoints
x0, x1 = 0, 1
val0 = np.polyval(coeffs_const, x0)
val1 = np.polyval(coeffs_const, x1)
print(f'At x=0: {val0:.6f}')
print(f'At x=1: {val1:.6f}')

# Test with linear function
def g(x):
    return 2*x + 1

coeffs_linear = findBernsteinPoly(g, l, u, n)
print(f'\nPython Linear function coeffs: {coeffs_linear}')

val0_linear = np.polyval(coeffs_linear, x0)
val1_linear = np.polyval(coeffs_linear, x1)
print(f'At x=0: {val0_linear:.6f} (expected: {g(x0):.6f})')
print(f'At x=1: {val1_linear:.6f} (expected: {g(x1):.6f})')

# Test with quadratic function
def h(x):
    return x**2

coeffs_quad = findBernsteinPoly(h, l, u, n)
print(f'\nPython Quadratic function coeffs: {coeffs_quad}')

# Test accuracy at several points
test_points = np.linspace(l, u, 5)
print('\nQuadratic function accuracy test:')
for x in test_points:
    bernstein_val = np.polyval(coeffs_quad, x)
    original_val = h(x)
    error = abs(bernstein_val - original_val)
    print(f'x={x:.3f}: Bernstein={bernstein_val:.6f}, Original={original_val:.6f}, Error={error:.6f}')

# Test with higher order
n_high = 5
coeffs_quad_high = findBernsteinPoly(h, l, u, n_high)
print(f'\nPython Quadratic function (n=5) coeffs: {coeffs_quad_high}')

# Test accuracy with higher order
print('\nQuadratic function accuracy test (n=5):')
for x in test_points:
    bernstein_val = np.polyval(coeffs_quad_high, x)
    original_val = h(x)
    error = abs(bernstein_val - original_val)
    print(f'x={x:.3f}: Bernstein={bernstein_val:.6f}, Original={original_val:.6f}, Error={error:.6f}')
