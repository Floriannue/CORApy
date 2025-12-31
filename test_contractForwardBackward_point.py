"""Test contractForwardBackward with point interval"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractForwardBackward import contractForwardBackward

def f(x):
    return x[0]**2 + x[1]**2 - 4

# Test with point interval
dom_point = Interval([1.0981, 1.0981], [1.0981, 1.0981])
print(f"=== Testing contractForwardBackward with point interval ===")
print(f"Domain: {dom_point}")
print(f"Domain width: {dom_point.sup - dom_point.inf}")

result = contractForwardBackward(f, dom_point)
print(f"\nResult: {result}")
print(f"Result type: {type(result)}")
if result is None:
    print("ERROR: contractForwardBackward returned None for point interval!")
else:
    print(f"Success: Result is {result}")

# Test with small interval (not quite a point)
dom_small = Interval([1.0981, 1.0981], [1.0982, 1.0982])
print(f"\n=== Testing contractForwardBackward with small interval ===")
print(f"Domain: {dom_small}")
print(f"Domain width: {dom_small.sup - dom_small.inf}")

result2 = contractForwardBackward(f, dom_small)
print(f"\nResult: {result2}")
if result2 is None:
    print("ERROR: contractForwardBackward returned None!")
else:
    print(f"Success: Result is {result2}")

