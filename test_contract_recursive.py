"""Test recursive contract call to see what happens"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract, _aux_bestSplit

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("=== Testing recursive contract call ===")
# Simulate what happens in the splitting code
domSplit = _aux_bestSplit(f, dom)
print(f"Split result: {domSplit}")

# Test the recursive call that happens in splitting
print("\n=== Testing recursive contract call (splits=None) ===")
for i, split_dom in enumerate(domSplit):
    print(f"\nSplit {i}: {split_dom}")
    result = contract(f, split_dom, 'interval', 2, None)
    print(f"  Result: {result}")
    print(f"  Result type: {type(result)}")
    if result is None:
        print(f"  WARNING: Result is None!")
    elif result.representsa_('emptySet', np.finfo(float).eps):
        print(f"  WARNING: Result is empty set!")
    else:
        print(f"  Result is valid")

