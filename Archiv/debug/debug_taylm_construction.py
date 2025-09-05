# Debug script to test Taylm construction from intervals
import numpy as np
from cora_python.contSet.taylm.taylm import Taylm
from cora_python.contSet.interval.interval import Interval

print("Testing Taylm construction from intervals:")

# Test 1: 2D interval
print("\n=== Test 1: 2D interval ===")
lb = np.array([-1, 0])
ub = np.array([1, 2])
I = Interval(lb, ub)
print(f"Interval: {I}")

tay = Taylm(I)
print(f"Taylm type: {type(tay)}")
print(f"Taylm dim: {tay.dim()}")
print(f"Monomials type: {type(tay.monomials)}")
print(f"Monomials: {tay.monomials}")
print(f"Coefficients: {tay.coefficients}")
print(f"Remainder: {tay.remainder}")

# Test 2: 3D interval
print("\n=== Test 2: 3D interval ===")
lb = np.array([-3, -2, -5])
ub = np.array([4, 2, 1])
I = Interval(lb, ub)
print(f"Interval: {I}")

tay = Taylm(I)
print(f"Taylm type: {type(tay)}")
print(f"Taylm dim: {tay.dim()}")
print(f"Monomials type: {type(tay.monomials)}")
print(f"Monomials: {tay.monomials}")
print(f"Coefficients: {tay.coefficients}")
print(f"Remainder: {tay.remainder}")

# Test 3: Check what _aux_parseInputArgs returns
print("\n=== Test 3: _aux_parseInputArgs debug ===")
from cora_python.contSet.taylm.taylm import _aux_parseInputArgs

func, int_obj, max_order, names, opt_method, eps, tolerance = _aux_parseInputArgs(I)
print(f"func: {func}")
print(f"int_obj: {int_obj}")
print(f"int_obj type: {type(int_obj)}")
print(f"max_order: {max_order}")
print(f"names: {names}")
print(f"opt_method: {opt_method}")
print(f"eps: {eps}")
print(f"tolerance: {tolerance}")
