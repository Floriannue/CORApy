"""Debug contract with interval algorithm and splits"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract, _aux_bestSplit

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("=== Testing splitting ===")
domSplit = _aux_bestSplit(f, dom)
print(f"domSplit: {domSplit}")
for i, d in enumerate(domSplit):
    print(f"  Split {i}: {d}")

print("\n=== Testing contract on each split ===")
for i, d in enumerate(domSplit):
    print(f"\nSplit {i}: {d}")
    result = contract(f, d, 'interval', 2, None)
    print(f"  Result: {result}")
    if result is None:
        print(f"  Result is None!")
    else:
        print(f"  Result dim: {result.dim()}")
        print(f"  Result inf: {result.inf}")
        print(f"  Result sup: {result.sup}")

