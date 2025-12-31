"""Reproduce the 'all' contractor issue with splits=2"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("=== Testing 'all' contractor with splits=2 ===")
print(f"Initial domain: {dom}")

result = contract(f, dom, 'all', 2, 2)
print(f"\nResult: {result}")
if result is None:
    print("ERROR: Result is None!")
else:
    print("Success!")

