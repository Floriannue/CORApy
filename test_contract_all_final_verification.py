"""Final verification of contract fixes"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("=== Testing all contractors with splits=2 ===")
contractors = ['forwardBackward', 'polynomial', 'linearize', 'interval', 'all']

for alg in contractors:
    print(f"\n--- Testing {alg} ---")
    result = contract(f, dom, alg, 2, 2)
    if result is None:
        print(f"  ERROR: {alg} returned None")
    else:
        print(f"  OK: {alg} returned valid result: {result}")

print("\n=== Testing all contractors without splitting ===")
for alg in contractors:
    print(f"\n--- Testing {alg} ---")
    result = contract(f, dom, alg, 2, None)
    if result is None:
        print(f"  ERROR: {alg} returned None")
    else:
        print(f"  OK: {alg} returned valid result: {result}")

