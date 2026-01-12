import numpy as np
import sympy as sp
from cora_python.g.functions.matlab.function_handle.isequalFunctionHandle import isequalFunctionHandle

# Test case 1: A @ x vs [x[0], x[1]] where A is identity
A = np.array([[1, 0], [0, 1]])

def f1(x, u):
    return A @ x

def f2(x, u):
    return np.array([[x[0]], [x[1]]])

print('Test 1: A @ x vs [x[0], x[1]]')
print('Result:', isequalFunctionHandle(f1, f2))

# Test case 2: A @ x + c vs [x[0] + 1, x[1] - 1]
A2 = np.array([[1, 0], [0, 1]])
c2 = np.array([[1], [-1]])

def f3(x, u):
    return A2 @ x + c2

def f4(x, u):
    return np.array([[x[0] + 1], [x[1] - 1]])

print('\nTest 2: A @ x + c vs [x[0] + 1, x[1] - 1]')
print('Result:', isequalFunctionHandle(f3, f4))

