"""Debug contractPoly forwardBackward in detail"""
import numpy as np
from cora_python.contSet.interval.interval import Interval
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractPoly import contractPoly
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractForwardBackward import contractForwardBackward

# Equivalent formulation with a polynomial function
c = -4
G = np.array([[1, 1]])
GI = np.array([])
E = 2 * np.eye(2)

# Function handle for polynomial constrained function
from cora_python.g.functions.helper.sets.contSet.interval.contractors.contractPoly import _aux_funcPoly

def f(x):
    return _aux_funcPoly(x, c, G, GI, E)

dom = Interval([1, 1], [3, 3])

print("=== Testing contractForwardBackward with polynomial function ===")
print(f"Initial domain: {dom}")

# Test contractForwardBackward directly
result_fb = contractForwardBackward(f, dom)
print(f"\ncontractForwardBackward result: {result_fb}")

# Test contractPoly with forwardBackward, no splitting
print("\n=== Testing contractPoly with forwardBackward, no splitting ===")
result_poly = contractPoly(c, G, GI, E, dom, 'forwardBackward', 2, None)
print(f"contractPoly result: {result_poly}")




