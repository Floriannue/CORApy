"""Test backpropagation with point interval"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.specification.syntaxTree import syntaxTree
from cora_python.specification.backpropagation import backpropagation

def f(x):
    return x[0]**2 + x[1]**2 - 4

# Test with point interval
dom_point = Interval([1.0981, 1.0981], [1.0981, 1.0981])
print(f"=== Testing backpropagation with point interval ===")
print(f"Domain: {dom_point}")

# Create syntax tree
vars_list = []
for i in range(dom_point.dim()):
    dom_i = Interval(dom_point.inf[i], dom_point.sup[i])
    vars_list.append(syntaxTree(dom_i, i))

print(f"vars_list: {vars_list}")

# Forward iteration
try:
    synTree = f(vars_list)
    print(f"synTree: {synTree}")
    print(f"synTree type: {type(synTree)}")
    
    # Backward iteration
    res = dom_point
    print(f"\nStarting backpropagation with res: {res}")
    
    if isinstance(synTree, list):
        synTree_list = synTree
    elif hasattr(synTree, '__len__') and not isinstance(synTree, str):
        synTree_list = list(synTree) if hasattr(synTree, '__iter__') else [synTree]
    else:
        synTree_list = [synTree]
    
    print(f"synTree_list: {synTree_list}")
    
    for i, tree in enumerate(synTree_list):
        print(f"\nProcessing tree {i}: {tree}")
        try:
            res = backpropagation(tree, Interval(0, 0), res)
            print(f"  Result after backpropagation: {res}")
        except Exception as ex:
            print(f"  Exception: {ex}")
            print(f"  Exception type: {type(ex)}")
            if hasattr(ex, 'identifier'):
                print(f"  Exception identifier: {ex.identifier}")
            raise
    
    print(f"\nFinal result: {res}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

