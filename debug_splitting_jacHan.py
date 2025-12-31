"""Debug jacHan passing in splitting"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("=== Testing contract with splits=2 ===")
print(f"Initial domain: {dom}")

# Add debug prints to see what's happening
import sys
original_contract = contract

def debug_contract(f, dom, *varargin):
    """Wrapper to debug contract calls"""
    print(f"\n  DEBUG: contract called with:")
    print(f"    f: {f}")
    print(f"    dom: {dom}")
    print(f"    varargin: {varargin}")
    
    # Check if jacHan is in varargin
    if len(varargin) >= 4:
        jacHan = varargin[3]
        print(f"    jacHan: {jacHan}")
        if jacHan is None:
            print(f"    WARNING: jacHan is None!")
    
    result = original_contract(f, dom, *varargin)
    print(f"    Result: {result}")
    return result

# Monkey patch for debugging (not ideal, but let's see)
# Actually, let's just call it normally and add print statements to contract.py temporarily
print("\nCalling contract with splits=2...")
result = contract(f, dom, 'interval', 2, 2)
print(f"\nFinal result: {result}")

