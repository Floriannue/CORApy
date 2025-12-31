"""Test all contractors with point intervals"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractForwardBackward import contractForwardBackward
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractInterval import contractInterval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractParallelLinearization import contractParallelLinearization
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractPolyBoxRevise import contractPolyBoxRevise
import sympy as sp

def f(x):
    return x[0]**2 + x[1]**2 - 4

# Test with point interval that doesn't satisfy constraint exactly
dom_point = Interval([1.098076211353316, 1.098076211353316], [1.098076211353316, 1.098076211353316])
print("=== Testing contractors with point interval ===")
print(f"Domain: {dom_point}")
print(f"f(center): {f([1.098076211353316, 1.098076211353316])}")

# Create Jacobian (same way as in contract.py)
n = dom_point.dim()
vars_sym = sp.symbols('x0:{}'.format(n))
if n == 1:
    vars_sym = [vars_sym]
else:
    vars_sym = list(vars_sym)

f_sym = f(vars_sym)
if not isinstance(f_sym, (list, tuple, np.ndarray)):
    f_sym = [f_sym]
elif isinstance(f_sym, np.ndarray):
    f_sym = f_sym.flatten().tolist()

# Compute Jacobian
J_sym = sp.Matrix(f_sym).jacobian(vars_sym)
# lambdify with proper argument handling
def jacHan_func(x):
    if isinstance(x, Interval):
        x_vals = [x.inf[i] for i in range(n)]
    elif hasattr(x, '__len__'):
        x_vals = list(x[:n]) if len(x) >= n else list(x) + [0] * (n - len(x))
    else:
        x_vals = [x] * n
    return np.array(J_sym.subs([(vars_sym[i], x_vals[i]) for i in range(n)])).astype(float)

jacHan = jacHan_func

print("\n1. Testing contractForwardBackward:")
try:
    result1 = contractForwardBackward(f, dom_point)
    print(f"   Result: {result1}")
    if result1 is None:
        print("   -> Returns None (rejects point interval)")
    else:
        print("   -> Returns interval (accepts point interval)")
except Exception as e:
    print(f"   Error: {e}")

print("\n2. Testing contractInterval:")
try:
    result2 = contractInterval(f, dom_point, jacHan)
    print(f"   Result: {result2}")
    if result2 is None:
        print("   -> Returns None (rejects point interval)")
    else:
        print("   -> Returns interval (accepts point interval)")
except Exception as e:
    print(f"   Error: {e}")

print("\n3. Testing contractParallelLinearization:")
try:
    result3 = contractParallelLinearization(f, dom_point, jacHan)
    print(f"   Result: {result3}")
    if result3 is None:
        print("   -> Returns None (rejects point interval)")
    else:
        print("   -> Returns interval (accepts point interval)")
except Exception as e:
    print(f"   Error: {e}")

print("\n4. Testing contractPolyBoxRevise:")
try:
    result4 = contractPolyBoxRevise(f, dom_point)
    print(f"   Result: {result4}")
    if result4 is None:
        print("   -> Returns None (rejects point interval)")
    else:
        print("   -> Returns interval (accepts point interval)")
except Exception as e:
    print(f"   Error: {e}")

