"""Test 'all' contractor with fresh domain"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract

def f(x):
    return x[0]**2 + x[1]**2 - 4

# Test with fresh domain each time
print("=== Testing 'all' with fresh domain ===")
dom = Interval([1, 1], [3, 3])
result = contract(f, dom, 'all', 2, 2)
print(f"Result: {result}")
if result is None:
    print("ERROR: Result is None!")
else:
    print("Success!")

# Test 'all' without splitting
print("\n=== Testing 'all' without splitting ===")
dom2 = Interval([1, 1], [3, 3])
result2 = contract(f, dom2, 'all', 2, None)
print(f"Result: {result2}")
if result2 is None:
    print("ERROR: Result is None!")
else:
    print("Success!")

