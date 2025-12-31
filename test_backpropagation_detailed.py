"""Detailed test of backpropagation with point interval"""
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
print(f"f(domain center): {f([1.0981, 1.0981])}")

# Create syntax tree
vars_list = []
for i in range(dom_point.dim()):
    dom_i = Interval(dom_point.inf[i], dom_point.sup[i])
    vars_list.append(syntaxTree(dom_i, i))

# Forward iteration
synTree = f(vars_list)
print(f"\nsynTree.value: {synTree.value}")
print(f"synTree.value type: {type(synTree.value)}")

# Check what the forward pass gives us
if isinstance(synTree.value, Interval):
    print(f"synTree.value.inf: {synTree.value.inf}")
    print(f"synTree.value.sup: {synTree.value.sup}")
    print(f"synTree.value contains 0: {synTree.value.contains_(0)}")
    print(f"synTree.value is empty: {synTree.value.representsa_('emptySet', np.finfo(float).eps)}")

# Now try backpropagation
print(f"\n=== Starting backpropagation ===")
res = dom_point
print(f"Initial res: {res}")

try:
    # The target value is Interval(0, 0) - we want f(x) = 0
    target = Interval(0, 0)
    print(f"Target value: {target}")
    print(f"Target contains 0: {target.contains_(0)}")
    print(f"Target is empty: {target.representsa_('emptySet', np.finfo(float).eps)}")
    
    res = backpropagation(synTree, target, res)
    print(f"Result: {res}")
except Exception as ex:
    print(f"Exception: {ex}")
    print(f"Exception type: {type(ex)}")
    if hasattr(ex, 'identifier'):
        print(f"Exception identifier: {ex.identifier}")
    
    # Let's see what the intermediate values are
    print("\n=== Debugging intermediate values ===")
    # Check if synTree.value intersects with target
    if isinstance(synTree.value, Interval):
        print(f"synTree.value: {synTree.value}")
        print(f"target: {target}")
        print(f"Are they intersecting: {synTree.value.isIntersecting_(target)}")
        print(f"synTree.value - target: {synTree.value - target}")

