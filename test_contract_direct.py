"""Direct test of contract without monkey patching"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("=== Direct test of contract with splits=2, alg='interval' ===")
print(f"Initial domain: {dom}")

result = contract(f, dom, 'interval', 2, 2)
print(f"\nResult: {result}")
print(f"Result type: {type(result)}")
if result is None:
    print("RESULT IS NONE - THIS IS THE BUG!")
else:
    print(f"Result dim: {result.dim()}")
    print(f"Result inf: {result.inf}")
    print(f"Result sup: {result.sup}")

