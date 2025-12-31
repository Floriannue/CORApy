"""Test 'all' contractor with iterations"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("=== Testing 'all' with iter_val=1 ===")
result1 = contract(f, dom, 'all', 1, None)
print(f"Result: {result1}")
if result1 is None:
    print("ERROR: Result is None!")
else:
    print("Success!")

print("\n=== Testing 'all' with iter_val=2 ===")
dom2 = Interval([1, 1], [3, 3])
result2 = contract(f, dom2, 'all', 2, None)
print(f"Result: {result2}")
if result2 is None:
    print("ERROR: Result is None!")
    print("This is the issue - after 2 iterations, 'all' returns None")
else:
    print("Success!")

