"""Debug 'all' contractor with splits=2 - final check"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("=== Testing 'all' with splits=2 ===")
print(f"Initial domain: {dom}")

# Test the recursive call that happens in splitting
print("\n=== Testing recursive call (what happens in splitting) ===")
# Simulate what happens: contract(f, domSplit[k], 'all', 2, None)
# First, let's see what the first split would be
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import _aux_bestSplit
domSplit = _aux_bestSplit(f, dom)
print(f"First split: {domSplit}")

# Test recursive call on first split
print(f"\nTesting recursive call on split[0]: {domSplit[0]}")
result_recursive = contract(f, domSplit[0], 'all', 2, None)
print(f"Recursive result: {result_recursive}")
if result_recursive is None:
    print("ERROR: Recursive call returned None!")
else:
    print("OK: Recursive call succeeded")

# Now test the full splitting
print(f"\n=== Testing full splitting ===")
result_full = contract(f, dom, 'all', 2, 2)
print(f"Full result: {result_full}")
if result_full is None:
    print("ERROR: Full splitting returned None!")
else:
    print("OK: Full splitting succeeded")

