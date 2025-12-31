"""Debug 'all' contractor step by step"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractForwardBackward import contractForwardBackward
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractInterval import contractInterval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractParallelLinearization import contractParallelLinearization
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractPolyBoxRevise import contractPolyBoxRevise
import sympy as sp

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

# Create jacHan like contract does
n = dom.dim()
vars_sym = sp.symbols('x0:{}'.format(n))
if n == 1:
    vars_sym = [vars_sym]
else:
    vars_sym = list(vars_sym)
vars_list = list(vars_sym)
f_sym = f(vars_list)
jac = sp.Matrix([[sp.diff(f_sym, var) for var in vars_sym]])
jac_func = sp.lambdify(vars_sym, jac, 'numpy')

def jacHan_func(x):
    if isinstance(x, Interval):
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

jacHan = jacHan_func

print("=== Simulating 'all' contractor logic ===")
dom_all = Interval(dom)
print(f"Step 1: forwardBackward")
dom_all = contractForwardBackward(f, dom_all)
print(f"  Result: {dom_all}")
if dom_all is None:
    print("  ERROR: forwardBackward returned None!")
    exit(1)
if dom_all.representsa_('emptySet', np.finfo(float).eps):
    print("  ERROR: forwardBackward returned empty set!")
    exit(1)

print(f"\nStep 2: interval")
dom_all = contractInterval(f, dom_all, jacHan)
print(f"  Result: {dom_all}")
if dom_all is None:
    print("  ERROR: interval returned None!")
    exit(1)
if dom_all.representsa_('emptySet', np.finfo(float).eps):
    print("  ERROR: interval returned empty set!")
    exit(1)

print(f"\nStep 3: linearize")
dom_all = contractParallelLinearization(f, dom_all, jacHan)
print(f"  Result: {dom_all}")
if dom_all is None:
    print("  ERROR: linearize returned None!")
    exit(1)
if dom_all.representsa_('emptySet', np.finfo(float).eps):
    print("  ERROR: linearize returned empty set!")
    exit(1)

print(f"\nStep 4: polynomial")
dom_all = contractPolyBoxRevise(f, dom_all)
print(f"  Result: {dom_all}")
if dom_all is None:
    print("  ERROR: polynomial returned None!")
    exit(1)
if dom_all.representsa_('emptySet', np.finfo(float).eps):
    print("  ERROR: polynomial returned empty set!")
    exit(1)

print("\nAll steps succeeded!")

