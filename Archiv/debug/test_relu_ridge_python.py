import numpy as np
from cora_python.nn.layers.nonlinear.nnReLULayer import nnReLULayer

# Test ReLU ridge regression specifically
layer = nnReLULayer()

# Test ridge regression order 1
l = -1
u = 1
order = 1
method = "ridgeregression"

print('Testing ReLU ridge regression order 1:')
print(f'l = {l}, u = {u}, order = {order}, method = {method}')

# Call computeApproxPoly
coeffs, d = layer.computeApproxPoly(l, u, order, method)

print(f'Python result: coeffs = [{coeffs[0]}, {coeffs[1]}], d = {d}')
print(f'Expected MATLAB: coeffs = [0.499932, 0.250034], d = 0.250034')

# Test regular regression for comparison
method2 = "regression"
coeffs2, d2 = layer.computeApproxPoly(l, u, order, method2)
print(f'Python regression: coeffs = [{coeffs2[0]}, {coeffs2[1]}], d = {d2}')
print(f'Expected MATLAB: coeffs = [0.5, 0.25], d = 0.25')
