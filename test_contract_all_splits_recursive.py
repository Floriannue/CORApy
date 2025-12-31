"""Test 'all' contractor with splitting - recursive call"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract, _aux_bestSplit

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("=== Testing 'all' with splits=2 ===")
print(f"Initial domain: {dom}")

# Simulate what happens in splitting
domSplit = _aux_bestSplit(f, dom)
print(f"\nFirst split: {domSplit}")

# Test recursive call on first split
print(f"\n=== Testing recursive contract call on split 0 ===")
split0 = domSplit[0]
print(f"Split 0: {split0}")

# This is what happens in the splitting code: contract(f, split0, 'all', 2, None, jacHan)
# But we don't have jacHan, so let's test without it first
result_recursive = contract(f, split0, 'all', 2, None)
print(f"Recursive result: {result_recursive}")
if result_recursive is None:
    print("ERROR: Recursive call returned None!")
else:
    print("Success!")

