import numpy as np
from cora_python.nn.nnHelper.leastSquareRidgePolyFunc import leastSquareRidgePolyFunc

print('Testing Python leastSquareRidgePolyFunc:\n')

# Test 1: Linear function y = 2x + 1
print('=== Test 1: Linear function y = 2x + 1 ===')
x = np.array([0, 1, 2, 3])
y = 2 * x + 1
order = 1

# Test with default lambda (0.001)
coeffs_default = leastSquareRidgePolyFunc(x, y, order)
print(f'Default lambda (0.001): coeffs = [{coeffs_default[0]:.6f}, {coeffs_default[1]:.6f}]')

# Test with lambda = 0 (should be exact)
coeffs_exact = leastSquareRidgePolyFunc(x, y, order, 0)
print(f'Lambda = 0 (exact): coeffs = [{coeffs_exact[0]:.6f}, {coeffs_exact[1]:.6f}]')

# Test with lambda = 0.01
coeffs_reg = leastSquareRidgePolyFunc(x, y, order, 0.01)
print(f'Lambda = 0.01: coeffs = [{coeffs_reg[0]:.6f}, {coeffs_reg[1]:.6f}]')

print()

# Test 2: Quadratic function y = x² + 2x + 1
print('=== Test 2: Quadratic function y = x² + 2x + 1 ===')
x = np.array([0, 1, 2, 3, 4])
y = x**2 + 2*x + 1
order = 2

# Test with default lambda (0.001)
coeffs_default = leastSquareRidgePolyFunc(x, y, order)
print(f'Default lambda (0.001): coeffs = [{coeffs_default[0]:.6f}, {coeffs_default[1]:.6f}, {coeffs_default[2]:.6f}]')

# Test with lambda = 0 (should be exact)
coeffs_exact = leastSquareRidgePolyFunc(x, y, order, 0)
print(f'Lambda = 0 (exact): coeffs = [{coeffs_exact[0]:.6f}, {coeffs_exact[1]:.6f}, {coeffs_exact[2]:.6f}]')

# Test with lambda = 0.01
coeffs_reg = leastSquareRidgePolyFunc(x, y, order, 0.01)
print(f'Lambda = 0.01: coeffs = [{coeffs_reg[0]:.6f}, {coeffs_reg[1]:.6f}, {coeffs_reg[2]:.6f}]')

print()

# Test 3: Edge cases
print('=== Test 3: Edge cases ===')

# Single point
x = np.array([1])
y = np.array([5])
order = 0
coeffs = leastSquareRidgePolyFunc(x, y, order)
print(f'Single point (order 0): coeffs = [{coeffs[0]:.6f}]')

# Two points
x = np.array([1, 2])
y = np.array([3, 5])
order = 1
coeffs = leastSquareRidgePolyFunc(x, y, order)
print(f'Two points (order 1): coeffs = [{coeffs[0]:.6f}, {coeffs[1]:.6f}]')

print()

# Test 4: Error cases (Python behavior)
print('=== Test 4: Error cases (Python behavior) ===')

try:
    # Negative order
    coeffs = leastSquareRidgePolyFunc([1, 2, 3], [2, 4, 6], -1)
    print('Negative order: SUCCESS (unexpected!)')
except Exception as e:
    print(f'Negative order: ERROR - {e}')

try:
    # Order >= number of points
    coeffs = leastSquareRidgePolyFunc([1, 2, 3], [2, 4, 6], 3)
    print('Order >= points: SUCCESS (unexpected!)')
except Exception as e:
    print(f'Order >= points: ERROR - {e}')

try:
    # Mismatched lengths
    coeffs = leastSquareRidgePolyFunc([1, 2, 3], [2, 4], 1)
    print('Mismatched lengths: SUCCESS (unexpected!)')
except Exception as e:
    print(f'Mismatched lengths: ERROR - {e}')

try:
    # Empty arrays
    coeffs = leastSquareRidgePolyFunc([], [], 1)
    print('Empty arrays: SUCCESS (unexpected!)')
except Exception as e:
    print(f'Empty arrays: ERROR - {e}')

try:
    # Negative lambda
    coeffs = leastSquareRidgePolyFunc([1, 2, 3], [2, 4, 6], 1, -0.1)
    print('Negative lambda: SUCCESS (unexpected!)')
except Exception as e:
    print(f'Negative lambda: ERROR - {e}')

print('\nDone.')
