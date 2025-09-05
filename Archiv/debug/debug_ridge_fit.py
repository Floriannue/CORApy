import numpy as np
from cora_python.nn.nnHelper.leastSquareRidgePolyFunc import leastSquareRidgePolyFunc

# Test the exact same case as the failing test
x = np.array([0, 1, 2, 3])
y = 2 * x + 1
order = 1

print('Input:')
print(f'x = {x}')
print(f'y = {y}')
print(f'order = {order}')

# Test with default lambda (0.001)
coeffs = leastSquareRidgePolyFunc(x, y, order)
print(f'\nDefault lambda (0.001):')
print(f'coeffs = {coeffs}')

# Check fit
y_poly = np.polyval(coeffs, x)
print(f'y_poly = {y_poly}')
print(f'y = {y}')
print(f'difference = {y - y_poly}')
print(f'max difference = {np.max(np.abs(y - y_poly))}')

# Test with lambda = 0 (should be exact)
coeffs_exact = leastSquareRidgePolyFunc(x, y, order, 0.0)
print(f'\nLambda = 0 (exact):')
print(f'coeffs = {coeffs_exact}')

y_poly_exact = np.polyval(coeffs_exact, x)
print(f'y_poly = {y_poly_exact}')
print(f'y = {y}')
print(f'difference = {y - y_poly_exact}')
print(f'max difference = {np.max(np.abs(y - y_poly_exact))}')

# Let's also check what MATLAB would give us
print(f'\nExpected MATLAB results:')
print('Default lambda (0.001): coeffs = [1.999900, 0.999900]')
print('Lambda = 0 (exact): coeffs = [2.000000, 1.000000]')
