"""Debug 'all' contractor with 2 iterations"""
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

# Create jacHan
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

print("=== Simulating 'all' contractor with 2 iterations ===")
dom_all = Interval(dom)
dom_ = dom_all

for i in range(2):
    print(f"\n--- Iteration {i+1} ---")
    print(f"dom before: {dom_all}")
    print(f"dom_ before: {dom_}")
    
    # 'all' contractor logic
    dom_all = contractForwardBackward(f, dom_all)
    print(f"  After forwardBackward: {dom_all}")
    if dom_all is None or dom_all.representsa_('emptySet', np.finfo(float).eps):
        print("  ERROR: forwardBackward failed!")
        break
    
    dom_all = contractInterval(f, dom_all, jacHan)
    print(f"  After interval: {dom_all}")
    if dom_all is None or dom_all.representsa_('emptySet', np.finfo(float).eps):
        print("  ERROR: interval failed!")
        break
    
    dom_all = contractParallelLinearization(f, dom_all, jacHan)
    print(f"  After linearize: {dom_all}")
    if dom_all is None or dom_all.representsa_('emptySet', np.finfo(float).eps):
        print("  ERROR: linearize failed!")
        break
    
    dom_all = contractPolyBoxRevise(f, dom_all)
    print(f"  After polynomial: {dom_all}")
    if dom_all is None or dom_all.representsa_('emptySet', np.finfo(float).eps):
        print("  ERROR: polynomial failed!")
        break
    
    # Check if set is empty
    if dom_all is None or dom_all.representsa_('emptySet', np.finfo(float).eps):
        print("  ERROR: Set is empty after iteration!")
        break
    
    # Check for convergence
    diff_inf = np.abs(dom_all.inf - dom_.inf)
    diff_sup = np.abs(dom_all.sup - dom_.sup)
    print(f"  diff_inf: {diff_inf}")
    print(f"  diff_sup: {diff_sup}")
    print(f"  eps: {np.finfo(float).eps}")
    converged = np.all(diff_inf < np.finfo(float).eps) and np.all(diff_sup < np.finfo(float).eps)
    print(f"  Converged: {converged}")
    
    if converged:
        print("  Convergence detected, breaking")
        break
    else:
        dom_ = dom_all

print(f"\nFinal result: {dom_all}")

