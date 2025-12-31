"""Test contract with interval algorithm and splits"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

# Test exactly as the test does
cont = ['forwardBackward', 'polynomial', 'linearize', 'interval', 'all']
iter_val = 2
splits = 2

for i, alg in enumerate(cont):
    print(f"\n=== Testing contractor: {alg} ===")
    try:
        I1 = contract(f, dom, alg, iter_val, splits)
        if I1 is None:
            print(f"  Result: None")
        else:
            print(f"  Result: {I1}")
            print(f"  Result dim: {I1.dim()}")
            print(f"  Result inf: {I1.inf}")
            print(f"  Result sup: {I1.sup}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

