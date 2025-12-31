"""Test the 'all' contractor"""
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
    print(f"Success: Result is valid")

# Test step by step
print("\n=== Testing 'all' contractor step by step ===")
dom_step = Interval([1, 1], [3, 3])
print(f"Step 1: forwardBackward")
dom_step = contract(f, dom_step, 'forwardBackward', 2, 2)
print(f"  Result: {dom_step}")
if dom_step is None:
    print("  ERROR: forwardBackward returned None!")
    exit(1)

print(f"\nStep 2: interval")
dom_step = contract(f, dom_step, 'interval', 2, 2)
print(f"  Result: {dom_step}")
if dom_step is None:
    print("  ERROR: interval returned None!")
    exit(1)

print(f"\nStep 3: linearize")
dom_step = contract(f, dom_step, 'linearize', 2, 2)
print(f"  Result: {dom_step}")
if dom_step is None:
    print("  ERROR: linearize returned None!")
    exit(1)

print(f"\nStep 4: polynomial")
dom_step = contract(f, dom_step, 'polynomial', 2, 2)
print(f"  Result: {dom_step}")
if dom_step is None:
    print("  ERROR: polynomial returned None!")
    exit(1)

print("\nAll steps succeeded!")

