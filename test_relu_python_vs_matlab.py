import numpy as np
from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer

# Test ReLU polynomial approximation
layer = nnReLULayer()

# Test polynomial approximation
l = -1
u = 1
order = 1
method = "regression"

print('Testing ReLU polynomial approximation:')
print(f'l = {l}, u = {u}, order = {order}, method = {method}')

# Call computeApproxPoly
coeffs, d = layer.computeApproxPoly(l, u, order, method)

print(f'coeffs = [{coeffs[0]}, {coeffs[1]}]')
print(f'd = {d}')

# Test evaluation at some points
x_test = np.linspace(l, u, 10)
y_true = layer.f(x_test)
y_approx = np.polyval(coeffs, x_test)

print('\nEvaluation results:')
for i in range(len(x_test)):
    print(f'x = {x_test[i]:.6f}: true = {y_true[i]:.6f}, approx = {y_approx[i]:.6f}')

# Expected MATLAB values:
# coeffs = [0.5, 0.25]
# d = 0.25
print(f'\nExpected MATLAB:')
print(f'coeffs = [0.5, 0.25]')
print(f'd = 0.25')
print(f'Match coeffs: {np.allclose(coeffs, [0.5, 0.25])}')
print(f'Match d: {np.isclose(d, 0.25)}')
