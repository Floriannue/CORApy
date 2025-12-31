"""Test contract in a loop like the test does"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract

def f(x):
    return x[0]**2 + x[1]**2 - 4

# Test exactly like the test does
cont = ['forwardBackward', 'polynomial', 'linearize', 'interval', 'all']
iter_val = 2
splits = 2

for i, alg in enumerate(cont):
    print(f"\n=== Testing contractor {i}: {alg} ===")
    # Create a fresh domain for each contractor
    dom = Interval([1, 1], [3, 3])
    print(f"Domain before: {dom}")
    
    I1 = contract(f, dom, alg, iter_val, splits)
    print(f"Result: {I1}")
    
    if I1 is None:
        print(f"ERROR: Contractor {alg} returned None!")
        break
    else:
        print(f"Success: Contractor {alg} returned valid result")

