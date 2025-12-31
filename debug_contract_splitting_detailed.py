"""Detailed debug of contract splitting issue"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contract import contract

def f(x):
    return x[0]**2 + x[1]**2 - 4

dom = Interval([1, 1], [3, 3])

print("=== Testing contract with splits=2, alg='interval' ===")
print(f"Initial domain: {dom}")

# Add a wrapper to trace recursive calls
original_contract = contract
call_depth = [0]

def traced_contract(f, dom, *varargin):
    call_depth[0] += 1
    depth = call_depth[0]
    indent = "  " * depth
    
    # Parse arguments
    if len(varargin) >= 1:
        alg = varargin[0]
    else:
        alg = 'forwardBackward'
    if len(varargin) >= 2:
        iter_val = varargin[1]
    else:
        iter_val = 1
    if len(varargin) >= 3:
        splits = varargin[2]
    else:
        splits = None
    if len(varargin) >= 4:
        jacHan = varargin[3]
    else:
        jacHan = None
    
    print(f"{indent}[Depth {depth}] contract called:")
    print(f"{indent}  dom: {dom}")
    print(f"{indent}  alg: {alg}")
    print(f"{indent}  iter_val: {iter_val}")
    print(f"{indent}  splits: {splits}")
    print(f"{indent}  jacHan: {jacHan is not None}")
    
    result = original_contract(f, dom, *varargin)
    
    print(f"{indent}[Depth {depth}] contract returned: {result}")
    if result is None:
        print(f"{indent}  WARNING: Result is None!")
    elif hasattr(result, 'representsa_') and result.representsa_('emptySet', np.finfo(float).eps):
        print(f"{indent}  WARNING: Result is empty set!")
    else:
        print(f"{indent}  Result is valid")
    
    call_depth[0] -= 1
    return result

# Monkey patch for debugging
import cora_python.g.functions.helper.sets.contSet.interval.contractors.contract as contract_module
contract_module.contract = traced_contract

# Now call it
result = contract(f, dom, 'interval', 2, 2)
print(f"\n=== Final Result ===")
print(f"Result: {result}")

