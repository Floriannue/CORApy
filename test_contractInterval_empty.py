"""Test if contractInterval returns None or empty interval"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractInterval import contractInterval
import sympy as sp

def f(x):
    return x[0]**2 + x[1]**2 - 4

# Test with a domain that should result in empty set
dom = Interval([2, 1], [3, 3])  # This domain doesn't contain the solution

# Create jacobian
vars_sym = sp.symbols('x0:2')
f_sym = f(list(vars_sym))
jac = sp.Matrix([[sp.diff(f_sym, var) for var in vars_sym]])
jac_func = sp.lambdify(vars_sym, jac, 'numpy')

def jacHan(x):
    if isinstance(x, Interval):
        # For interval input, return center (simplified)
        x_center = x.center()
        result = jac_func(*x_center.flatten())
        return np.asarray(result).reshape(1, -1)
    else:
        x_flat = np.asarray(x).flatten()
        result = jac_func(*x_flat)
        result = np.asarray(result)
        if result.ndim == 0:
            return np.array([[float(result)]], dtype=float)
        elif result.ndim == 1:
            return result.reshape(1, -1)
        else:
            return result

print(f"Testing contractInterval with domain: {dom}")
result = contractInterval(f, dom, jacHan)
print(f"Result: {result}")
print(f"Result type: {type(result)}")
if result is None:
    print("Result is None")
else:
    print(f"Result is empty: {result.representsa_('emptySet', np.finfo(float).eps)}")

