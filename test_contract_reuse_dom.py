"""Test contract with reused domain like the test does"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract

def f(x):
    return x[0]**2 + x[1]**2 - 4

# Create domain once and reuse it (like the test does)
dom = Interval([1, 1], [3, 3])

cont = ['forwardBackward', 'polynomial', 'linearize', 'interval', 'all']
iter_val = 2
splits = 2

for i, alg in enumerate(cont):
    print(f"\n=== Testing contractor {i}: {alg} ===")
    print(f"Domain before: {dom}")
    print(f"Domain id: {id(dom)}")
    
    I1 = contract(f, dom, alg, iter_val, splits)
    print(f"Result: {I1}")
    print(f"Domain after: {dom}")
    print(f"Domain id after: {id(dom)}")
    
    if I1 is None:
        print(f"ERROR: Contractor {alg} returned None!")
        break
    else:
        print(f"Success: Contractor {alg} returned valid result")

