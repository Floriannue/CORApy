"""Test contract with interval algorithm"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("Testing contract with 'interval' algorithm (splits=None)...")
result = contract(f, dom, 'interval', 2, None)
print(f"Result: {result}")
print(f"Result type: {type(result)}")
if result is not None:
    print(f"Result dim: {result.dim()}")
    print(f"Result inf: {result.inf}")
    print(f"Result sup: {result.sup}")

print("\nTesting contract with 'interval' algorithm (splits=2)...")
result2 = contract(f, dom, 'interval', 2, 2)
print(f"Result: {result2}")
print(f"Result type: {type(result2)}")
if result2 is not None:
    print(f"Result dim: {result2.dim()}")
    print(f"Result inf: {result2.inf}")
    print(f"Result sup: {result2.sup}")

